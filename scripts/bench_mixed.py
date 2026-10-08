#!/usr/bin/env python
"""混合长度测试(c=4):看 **prefill 与 decode 同时发生** 时,解码被拖慢多少。

用户 2026-10-08 追加要求:"有没有 mixed 测试方式,即 cs=4 但是 prompt 长短混合,
看看 prefill 的同时 decode?" ✓

设计:
* `c=4` 固定;每个项目是一组**长度组合**(混合 long 16384 / short 128)✓
* c 个请求**同时发出** ⇒ 短请求很快进入解码,而长请求**仍在预填** ⇒ 天然形成"预填 ‖ 解码"✓
* ⭐ **同位对照**:对每个**短**请求,把它的解码按时间切成两段 ——
    * **段 A**:长请求**仍在预填**期间(时间 `< max_i 长请求的 TTFT`)
    * **段 B**:长预填**已结束**之后
  ⇒ `A 段速度 / B 段速度` = **该请求在原地被长预填拖慢的倍数** ✓✓
* 每个请求记录**逐 token 到达时刻** ⇒ 才能做上面的切分 ✓

关键口径(**先写死** ✓):
  · 段 A 的时长 = `min(t_last, t_longprefill_end) − t_first`,token 数 = 该窗口内到达的 token 数
  · 段 B 的时长 = `t_last − max(t_first, t_longprefill_end)`
  · 若某段窗口 < 0.2 s 或 token < 3 ⇒ 该段**不参与统计**(样本太短)✓
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time

import requests

PORT = int(os.environ.get("PORT", 8070))
URL = "http://127.0.0.1:%d/v1/completions" % PORT
MODEL = os.environ.get("SERVED", "DeepSeek-V4.1-Flash")

_SEED_LOCK = threading.Lock()
_SEED = [0]


def _next_seed() -> int:
    with _SEED_LOCK:
        _SEED[0] += 1
        return (_SEED[0] * 1000003) ^ (time.time_ns() & 0xFFFFFFFF)


def _rand_ids(n: int, seed: int):
    import random
    rng = random.Random(seed)
    return [rng.randrange(1000, 100000) for _ in range(n)]


def one(i: int, n_prompt: int, max_tokens: int, t0: float, out: dict) -> None:
    """记录【逐 token 到达时刻】(相对 t0)⇒ 才能把解码按"长预填是否还在跑"切分 ✓"""
    body = {
        "model": MODEL,
        "prompt": _rand_ids(n_prompt, _next_seed()),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    times = []
    try:
        with requests.post(URL, json=body, stream=True, timeout=7200) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                p = line[6:].strip()
                if p == b"[DONE]":
                    break
                try:
                    j = json.loads(p)
                except Exception:  # noqa: BLE001
                    continue
                ch = j.get("choices") or []
                if ch and ch[0].get("text"):
                    times.append(time.perf_counter() - t0)
        out[i] = {"n_prompt": n_prompt, "tok_times": times, "err": None}
    except Exception as e:  # noqa: BLE001
        out[i] = {"n_prompt": n_prompt, "tok_times": [], "err": repr(e)[:200]}


def run_mix(mix: list, max_tokens: int) -> dict:
    """一轮:len(mix) 个不同长度的请求【同时】发出 ✓"""
    out = {}
    ths = []
    n = len(mix)
    # ⭐ 同时发出的关键:先把线程都建好,再用一个统一的 t0 启动
    t0 = time.perf_counter()
    for i, L in enumerate(mix):
        ths.append(threading.Thread(target=one, args=(i, L, max_tokens, t0, out)))
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    wall = time.perf_counter() - t0

    ok = {i: v for i, v in out.items() if v["err"] is None and v["tok_times"]}
    if not ok:
        return {"err": [v["err"] for v in out.values()][:1] or ["no ok"]}
    longs = [i for i, v in ok.items() if v["n_prompt"] >= 1024]
    shorts = [i for i, v in ok.items() if v["n_prompt"] < 1024]
    # ⭐ 长请求的预填结束时刻(取最大 TTFT)⇒ 短请求解码的分界线 ✓
    lpe = max(ok[i]["tok_times"][0] for i in longs) if longs else 0.0
    rec = {"n": n, "wall": wall, "lpe": lpe, "longs": [], "shorts": [], "err": None}
    for i, v in ok.items():
        tt = v["tok_times"]
        e = {"i": i, "L": v["n_prompt"], "ttft": tt[0], "n_out": len(tt), "last": tt[-1]}
        if v["n_prompt"] >= 1024 or not longs:
            e["cls"] = "long"
            rec["longs"].append(e)
        else:
            e["cls"] = "short"
            # ── ⭐ 同位对照:段 A(长预填仍在)vs 段 B(长预填已结束)──
            a_t = [t for t in tt if t <= lpe]           # A 段到达的 token
            b_t = [t for t in tt if t > lpe]
            durA = max(0.0, min(tt[-1], lpe) - tt[0])
            durB = max(0.0, tt[-1] - max(tt[0], lpe))
            e["A_tokens"] = max(0, len(a_t) - 1)         # 减首 token(prefill)
            e["B_tokens"] = len(b_t)
            e["A_dur"] = durA
            e["B_dur"] = durB
            e["A_tok_s"] = (e["A_tokens"] / durA) if (durA >= 0.2 and e["A_tokens"] >= 3) else None
            e["B_tok_s"] = (e["B_tokens"] / durB) if (durB >= 0.2 and e["B_tokens"] >= 3) else None
            rec["shorts"].append(e)
    return rec


def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def fmt(x, p="%.1f"):
    return (p % x) if x is not None else "—"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", type=int, default=16384)
    ap.add_argument("--short", type=int, default=128)
    ap.add_argument("--c", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--json", default="dev-docs/report/tuning/logs/bench_mixed.json")
    args = ap.parse_args()
    Lo, Sh, c = args.long, args.short, args.c

    mixes = [
        ("全部长 (4L)", [Lo] * c),
        ("3L + 1S", [Lo] * (c - 1) + [Sh]),
        ("2L + 2S", [Lo] * (c // 2) + [Sh] * (c - c // 2)),
        ("1L + 3S", [Lo] + [Sh] * (c - 1)),
        ("全部短 (4S)", [Sh] * c),
    ]
    print("# 混合长度测试:c=%d · 长=%d / 短=%d · 每项目 %d 轮(+%d 预热)· max_tokens=%d"
          % (c, Lo, Sh, args.rounds, args.warm, args.max_tokens), flush=True)
    print("# ⭐ 核心量:短请求解码的【A 段(长预填仍在)/ B 段(长预填已结束)】速度之比", flush=True)

    results = []
    for name, mix in mixes:
        for w in range(args.warm):
            r = run_mix(mix, args.max_tokens)
            print("  [预热 %s %d/%d] %s" % (name, w + 1, args.warm,
                  r.get("err") or ("wall %.1fs" % r["wall"])), flush=True)
        rows = []
        for k in range(args.rounds):
            r = run_mix(mix, args.max_tokens)
            if r.get("err"):
                print("  [%s 测量 %d] ⛔ %s" % (name, k + 1, r["err"]), flush=True)
                continue
            rows.append(r)
            sl = r["shorts"]
            a = med([e["A_tok_s"] for e in sl])
            b = med([e["B_tok_s"] for e in sl])
            ratio = (a / b) if (a and b) else None
            print("  [%s 测量 %d/%d] wall %6.1fs · 短请求 TTFT %s s · "
                  "A段(长预填中) %s tok/s · B段(长预填后) %s tok/s · ⭐拖慢倍数 %s"
                  % (name, k + 1, args.rounds, r["wall"],
                     fmt(med([e["ttft"] for e in sl]), "%.2f"),
                     fmt(a), fmt(b), fmt(ratio, "%.2fx")), flush=True)
        if not rows:
            results.append({"name": name, "mix": mix, "err": "no ok"})
            continue
        allsh = [e for r in rows for e in r["shorts"]]
        alllg = [e for r in rows for e in r["longs"]]
        A = med([e["A_tok_s"] for e in allsh])
        B = med([e["B_tok_s"] for e in allsh])
        proj = {
            "name": name, "mix": mix,
            "long_ttft": med([e["ttft"] for e in alllg]),
            "short_ttft": med([e["ttft"] for e in allsh]),
            "long_decode": med([e["n_out"] / max(e["last"] - e["ttft"], 1e-9) for e in alllg]),
            "short_A": A, "short_B": B,
            "ratio": (A / B) if (A and B) else None,
            "n_rounds": len(rows),
            "short_A_n": len([e for e in allsh if e["A_tok_s"] is not None]),
            "short_B_n": len([e for e in allsh if e["B_tok_s"] is not None]),
        }
        results.append(proj)
        print("  ⇒ **%s**:长TTFT %s s · 短TTFT %s s · 短解码 A %s → B %s tok/s · "
              "⭐**拖慢 %s**" % (name, fmt(proj["long_ttft"], "%.2f"),
                                fmt(proj["short_ttft"], "%.2f"),
                                fmt(A), fmt(B), fmt(proj["ratio"], "%.2fx")), flush=True)

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    json.dump({"c": c, "long": Lo, "short": Sh, "max_tokens": args.max_tokens,
               "rounds": args.rounds, "results": results},
              open(args.json, "w"), ensure_ascii=False, indent=1)
    print("\n# 原始数据 → %s" % args.json, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

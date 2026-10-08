#!/usr/bin/env python
"""性能矩阵:c ∈ {1,2,4} × L ∈ {128, 16384} = **6 个项目**。

每项目三项指标(用户 2026-10-08 口径):
  ① **中位数 TTFT(s)**   —— 该项目所有请求(含 c>1 的并发请求)TTFT 的中位数
  ② **聚合 prefill 速度(tok/s)** = (本轮 prompt token 总数) / (max_i TTFT_i)
  ③ **聚合 decode 速度(tok/s)**  = (本轮 output token 总数 − c) / (max_i 末token时刻 − min_i 首token时刻)

纪律:
* **random 数据集**:每个请求一份全新随机 token 序列 ⇒ 与基线无共享前缀 ⇒ **不触发 prefix cache** ✓
* **充分预热**:全局预热(两种长度各若干轮)+ **每项目**预热若干轮,预热不计入统计 ✓
* **每项目多轮取中位**,并报离散度 ✓
* 只用标准库 + requests;并发用线程(c≤4)✓
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import threading
import time

import requests

PORT = int(os.environ.get("PORT", 8070))
URL = "http://127.0.0.1:%d/v1/completions" % PORT
MODEL = os.environ.get("SERVED", "DeepSeek-V4.1-Flash")
MAXTOK = int(os.environ.get("MAXTOK", "128"))


_SEED_LOCK = threading.Lock()
_SEED = [0]


def _next_seed() -> int:
    """⭐ B385:【每个请求】一个全局唯一、且带时间熵的种子 ——
    原来用线程下标 pid 当 seed ⇒ **每轮生成同一串 prompt** ⇒ 第 2 轮起全是 **prefix cache 命中** ✗
    (实测后果:L=16384 的 TTFT 从应有的 ~10 s 掉到 **0.79 s**、聚合 prefill 虚高到 **20,633 tok/s** ✗)
    """
    with _SEED_LOCK:
        _SEED[0] += 1
        return (_SEED[0] * 1000003) ^ (time.time_ns() & 0xFFFFFFFF)


def rand_prompt_ids(n: int, seed: int) -> list:
    """random 数据集:均匀随机 token id(避开特殊 token 区);seed 必须**每请求唯一** ✓"""
    rng = random.Random(seed)
    return [rng.randrange(1000, 100000) for _ in range(n)]


def one_request(pid: int, n_prompt: int, max_tokens: int, out: dict) -> None:
    _pids = rand_prompt_ids(n_prompt, _next_seed())      # ⭐ 每请求唯一 ✓
    _fp = hash(tuple(_pids[:8])) ^ len(_pids)            # 指纹(用于验证唯一性)✓
    body = {
        "model": MODEL,
        "prompt": _pids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    t_last = None
    tok_times = []
    n_out = 0
    n_prompt_actual = None
    try:
        with requests.post(URL, json=body, stream=True, timeout=7200) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                try:
                    j = json.loads(payload)
                except Exception:  # noqa: BLE001
                    continue
                ch = j.get("choices") or []
                if ch and ch[0].get("text"):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    t_last = now
                    tok_times.append(now - t0)      # ⭐ MEDIAN-ITL:留下逐 token 时刻 ✓
                    n_out += 1
                u = j.get("usage")
                if u:
                    n_prompt_actual = u.get("prompt_tokens", n_prompt_actual)
                    if u.get("completion_tokens"):
                        n_out = u["completion_tokens"]
        out[pid] = {"ttft": ttft, "t_last": t_last, "n_out": n_out,
                    "n_prompt": n_prompt_actual or n_prompt, "err": None, "fp": _fp,
                    "tok_times": tok_times}
    except Exception as e:  # noqa: BLE001
        out[pid] = {"ttft": None, "t_last": None, "n_out": 0,
                    "n_prompt": n_prompt, "err": repr(e)[:200], "fp": _fp}


def run_round(c: int, n_prompt: int, max_tokens: int, tag: str) -> dict:
    out = {}
    ths = []
    for i in range(c):
        th = threading.Thread(target=one_request, args=(i, n_prompt, max_tokens, out))
        ths.append(th)
    t_start = time.perf_counter()
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    wall = time.perf_counter() - t_start

    ok = [v for v in out.values() if v["err"] is None and v["ttft"] is not None]
    errs = [v["err"] for v in out.values() if v["err"]]
    if not ok:
        return {"err": errs[:1] or ["no successful request"], "wall": wall}

    ttfts = [v["ttft"] for v in ok]
    t_first_min = min(ttfts)
    t_last_max = max((v["t_last"] or 0) for v in ok) - t_start
    # ⭐ MEDIAN-ITL(README 口径):ITL = 相邻 token 间隔;TPOT = (末-首)/(n-1) ✓
    _itls, _tpots = [], []
    for v in ok:
        tt = v.get("tok_times") or []
        if len(tt) >= 3:
            _itls += [tt[i + 1] - tt[i] for i in range(len(tt) - 1)]
            _tpots.append((tt[-1] - tt[0]) / (len(tt) - 1))
    itl_med = statistics.median(_itls) if _itls else None
    tpot_med = statistics.median(_tpots) if _tpots else None
    tok_in = sum(v["n_prompt"] for v in ok)
    tok_out = sum(v["n_out"] for v in ok)
    dec_tokens = max(tok_out - len(ok), 0)
    dec_win = max(t_last_max - t_first_min, 1e-9)
    return {
        "c": len(ok),
        "ttft_med": statistics.median(ttfts),
        "ttft_all": ttfts,
        "prefill_tok_s": tok_in / max(max(ttfts), 1e-9),
        "decode_tok_s": dec_tokens / dec_win,
        "tok_in": tok_in, "tok_out": tok_out, "wall": wall,
        "itl_med": itl_med, "tpot_med": tpot_med,
        "decode_itl": (1.0 / itl_med) if itl_med else None,   # ⭐ README 口径 ✓
        "fps": [v.get("fp") for v in ok], "err": None,
    }


def _line(prefix: str, r: dict) -> str:
    """把一轮结果渲染成一行(避免在 f-string 里嵌引号)"""
    if r.get("err"):
        return "%s ⛔ %s" % (prefix, r["err"])
    return ("%s TTFT中位 %6.2fs · 聚合prefill %8.1f tok/s · 聚合decode %7.1f tok/s"
            "  (tok_in=%d, tok_out=%d)"
            % (prefix, r["ttft_med"], r["prefill_tok_s"], r["decode_tok_s"],
               r["tok_in"], r["tok_out"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cs", default="1,2,4")
    ap.add_argument("--lens", default="128,16384")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--gwarm", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=MAXTOK)
    ap.add_argument("--json", default="dev-docs/report/tuning/logs/bench_matrix_itl.json")
    args = ap.parse_args()
    cs = [int(x) for x in args.cs.split(",")]
    lens = [int(x) for x in args.lens.split(",")]

    print("# 性能矩阵:c=%s × L=%s = %d 个项目" % (cs, lens, len(cs) * len(lens)), flush=True)
    print("# max_tokens=%d · 预热 %d 轮/项目(全局 %d 轮/形状) · 测量 %d 轮/项目 · random 数据集"
          % (args.max_tokens, args.warm, args.gwarm, args.rounds), flush=True)

    for L in lens:
        for g in range(args.gwarm):
            r = run_round(max(cs), L, args.max_tokens, "gwarm")
            print(_line("  [全局预热 L=%d 第%d轮]" % (L, g + 1), r), flush=True)

    results = []
    for L in lens:
        for c in cs:
            print("\n===== 项目 c=%d × L=%d =====" % (c, L), flush=True)
            for w in range(args.warm):
                r = run_round(c, L, args.max_tokens, "warm")
                print(_line("  [预热 %d/%d]" % (w + 1, args.warm), r), flush=True)
            rows = []
            for k in range(args.rounds):
                r = run_round(c, L, args.max_tokens, "meas")
                if r.get("err"):
                    print("  [测量 %d] ⛔ %s" % (k + 1, r["err"]), flush=True)
                    continue
                rows.append(r)
                print(_line("  [测量 %d/%d]" % (k + 1, args.rounds), r), flush=True)
            if not rows:
                results.append({"c": c, "L": L, "err": "no successful round"})
                continue
            all_ttft = [t for r in rows for t in r["ttft_all"]]
            proj = {
                "c": c, "L": L,
                "n_rounds": len(rows), "n_req": len(all_ttft),
                "ttft_med": statistics.median(all_ttft),
                "ttft_median_of_rounds": statistics.median([r["ttft_med"] for r in rows]),
                "prefill_tok_s": statistics.median([r["prefill_tok_s"] for r in rows]),
                "decode_tok_s": statistics.median([r["decode_tok_s"] for r in rows]),
                "decode_itl": statistics.median([r["decode_itl"] for r in rows
                                                 if r.get("decode_itl")] or [0]) or None,
                "itl_med": statistics.median([r["itl_med"] for r in rows if r.get("itl_med")] or [0]) or None,
                "tpot_med": statistics.median([r["tpot_med"] for r in rows if r.get("tpot_med")] or [0]) or None,
                "prefill_spread_pct": _spread([r["prefill_tok_s"] for r in rows]),
                "decode_spread_pct": _spread([r["decode_tok_s"] for r in rows]),
                "ttft_spread_pct": _spread([r["ttft_med"] for r in rows]),
                "tok_in": statistics.median([r["tok_in"] for r in rows]),
                "tok_out": statistics.median([r["tok_out"] for r in rows]),
            }
            results.append(proj)
            print("  ⇒ **中位 TTFT %.2fs** · **聚合 prefill %.1f tok/s**(离散 %.1f%%) · "
                  "**聚合 decode %.1f tok/s**(离散 %.1f%%) · ⭐**README 口径 decode(1/median ITL) %.1f tok/s**"
                  % (proj["ttft_med"], proj["prefill_tok_s"], proj["prefill_spread_pct"],
                     proj["decode_tok_s"], proj["decode_spread_pct"],
                     proj.get("decode_itl") or -1), flush=True)

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    json.dump({"max_tokens": args.max_tokens, "rounds": args.rounds,
               "cs": cs, "lens": lens, "results": results},
              open(args.json, "w"), ensure_ascii=False, indent=1)
    print("\n# 原始数据 → %s" % args.json, flush=True)
    return 0


def _spread(v: list) -> float:
    m = statistics.median(v)
    return (max(v) - min(v)) / m * 100 if m else 0.0


if __name__ == "__main__":
    raise SystemExit(main())

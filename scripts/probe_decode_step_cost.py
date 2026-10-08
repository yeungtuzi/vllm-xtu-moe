#!/usr/bin/env python3
"""量【纯解码步的代价】`t_dec(C)` —— R4 的 D 公式的输入之一。

═══════════════════════════════════════════════════════════════════════════
 为什么需要它
═══════════════════════════════════════════════════════════════════════════
R4(在长预填的 chunk 之间插入 D 个【纯解码步】)的闭式解:
    ITL(D) = (t_chunk + D·t_dec) / (D+1)  ≤  T_alive
⇒ 需要 `t_chunk`(来自 probe_prefill_overlap.py)与 **`t_dec`(本脚本)**。

⚠️ 关键细节:**生产开了投机解码(dspark k=5)** ⇒ 一个调度步可能**接受多个 token**,
   ⇒ 流里的"相邻 token 间隔"**不等于**步时间。所以本脚本同时给出三个量:
     ① **per-token ITL**(用户实际感受到的)
     ② **步时间 t_step**(把同一批到达的 token 聚成一步)
     ③ **接受长度 tokens/step**
   ⇒ D 数的是【步】,所以公式里该用 ②;而 ① 会因投机而自动变好 ✓

用法:
  PORT=8070 python scripts/probe_decode_step_cost.py            # C=1,2,4
  PORT=8070 CONC=1,2,4 REPEAT=3 python scripts/probe_decode_step_cost.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time

import requests


def stream_one(url, model, prompt, max_tokens, sink, tag):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
    }
    t0 = time.time()
    marks, last, first = [], t0, None
    try:
        with requests.post(url, json=body, stream=True, timeout=1800) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                payload = line[6:]
                if payload.strip() == b"[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except Exception:
                    continue
                delta = ((d.get("choices") or [{}])[0].get("delta") or {})
                ch = (delta.get("content") or delta.get("reasoning")
                      or delta.get("reasoning_content"))
                if ch:
                    now = time.time()
                    if first is None:
                        first = now - t0
                    marks.append(now - t0)
                    last = now
    except Exception as e:  # noqa: BLE001
        sink[tag] = {"error": repr(e)}
        return
    sink[tag] = {"ttft": first, "marks": marks, "n": len(marks)}


def burst_split(marks, thresh_ms=10.0):
    """把到达时刻聚成【步】:相邻间隔 > thresh 视为新的一步(同一 forward 的 token 几乎同时到)。"""
    if not marks:
        return [], []
    gaps_ms = [(marks[i] - marks[i - 1]) * 1000.0 for i in range(1, len(marks))]
    steps = [[marks[0]]]
    for i, g in enumerate(gaps_ms, start=1):
        if g > thresh_ms:
            steps.append([marks[i]])
        else:
            steps[-1].append(marks[i])
    step_times = [(steps[i][-1] - steps[i - 1][0]) * 1000.0 for i in range(1, len(steps))]
    sizes = [len(s) for s in steps]
    return step_times, sizes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8070)))
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash")
    ap.add_argument("--concs", default=os.environ.get("CONC", "1,2,4"))
    ap.add_argument("--repeat", type=int, default=int(os.environ.get("REPEAT", 3)))
    ap.add_argument("--len-a", type=int, default=64, help="短 prompt 字数(不触发长预填)")
    ap.add_argument("--tok", type=int, default=192, help="每路输出 token 数")
    ap.add_argument("--burst-ms", type=float, default=float(os.environ.get("BURST_MS", 10.0)))
    args = ap.parse_args()
    concs = [int(c) for c in args.concs.split(",") if c.strip()]
    url = f"http://{args.host}:{args.port}/v1/chat/completions"

    def filler(n, seed):
        return ("请逐字复述下面这段无意义文本,不要总结、不要解释:" +
                "".join(chr(0x4E00 + ((i * 7 + seed) % 2000)) for i in range(n)))

    print(f"{'C':>3} {'轮':>3} {'TTFT(s)':>8} {'per-token ITL(ms)':>18} "
          f"{'步时间(ms)':>11} {'tok/步':>7} {'步数':>5} {'聚合tok/s':>9}")
    print("-" * 82)

    summary = {}
    for C in concs:
        for rep in range(args.repeat):
            sink: dict = {}
            ths = []
            t_start = time.time()
            for j in range(C):
                # ⚠️ 每路不同 seed ⇒ 避开前缀缓存
                th = threading.Thread(target=stream_one,
                                      args=(url, args.model, filler(args.len_a, 10 + j + 100 * rep),
                                            args.tok, sink, f"r{j}"), daemon=True)
                th.start(); ths.append(th)
            for th in ths:
                th.join()
            wall = time.time() - t_start

            all_gaps, all_steps, all_sizes, ttfts, ntok = [], [], [], [], 0
            for j in range(C):
                r = sink.get(f"r{j}", {})
                if r.get("error") or not r.get("marks"):
                    continue
                m = r["marks"]
                ntok += len(m)
                if r.get("ttft"):
                    ttfts.append(r["ttft"])
                all_gaps += [(m[i] - m[i - 1]) * 1000.0 for i in range(1, len(m))]
                st, sz = burst_split(m, args.burst_ms)
                all_steps += st
                all_sizes += sz
            if not all_gaps:
                print(f"{C:>3} {rep:>3}   (无输出)")
                continue
            agg = ntok / wall if wall > 0 else 0
            print(f"{C:>3} {rep:>3} {statistics.median(ttfts):>8.2f} "
                  f"{statistics.median(all_gaps):>18.1f} "
                  f"{(statistics.median(all_steps) if all_steps else float('nan')):>11.1f} "
                  f"{(statistics.mean(all_sizes) if all_sizes else float('nan')):>7.2f} "
                  f"{len(all_steps):>5} {agg:>9.1f}")
            summary.setdefault(C, []).append({
                "itl_ms": statistics.median(all_gaps),
                "step_ms": statistics.median(all_steps) if all_steps else None,
                "tok_per_step": statistics.mean(all_sizes) if all_sizes else None,
                "agg_tok_s": agg,
            })
            time.sleep(1.5)

    print()
    print("=== 汇总(各轮中位)===")
    print(f"{'C':>3} {'per-token ITL':>14} {'步时间':>10} {'tok/步':>8} {'聚合 tok/s':>11}")
    out = {}
    for C in concs:
        rs = summary.get(C, [])
        if not rs:
            continue
        itl = statistics.median([r["itl_ms"] for r in rs])
        stp = statistics.median([r["step_ms"] for r in rs if r["step_ms"]])
        tps = statistics.mean([r["tok_per_step"] for r in rs if r["tok_per_step"]])
        agg = statistics.median([r["agg_tok_s"] for r in rs])
        out[C] = {"itl_ms": itl, "step_ms": stp, "tok_per_step": tps, "agg_tok_s": agg}
        print(f"{C:>3} {itl:>12.1f}ms {stp:>8.1f}ms {tps:>8.2f} {agg:>11.1f}")
    print()
    print("⇒ ⭐ R4 的 D 公式里用【步时间】:`D = ceil((t_chunk − T_alive)/(T_alive − step_ms))`")
    print("   而用户感受到的是【per-token ITL】—— 投机解码让后者自动优于前者 ✓")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""把 `vllm bench serve --save-result` 的 json 汇总成一张表(供 BENCH_REFERENCE 用)。

用法: python scripts/summarize_bench_serve.py dev-docs/report/tuning/logs/bench_serve_<tag>
"""
import glob
import json
import os
import sys

def g(d, *names, default=None):
    for n in names:
        if n in d:
            return d[n]
    return default

def main() -> int:
    d = sys.argv[1] if len(sys.argv) > 1 else "dev-docs/report/tuning/logs/bench_serve_acc"
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            j = json.load(open(p))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(j, dict):
            continue
        name = os.path.basename(p)[:-5]
        rows.append(dict(
            name=name,
            completed=g(j, "completed", "num_prompts", default=""),
            dur=g(j, "duration", default=None),
            req_tp=g(j, "request_throughput", default=None),
            out_tp=g(j, "output_throughput", default=None),
            tot_tp=g(j, "total_token_throughput", default=None),
            ttft=g(j, "mean_ttft_ms", default=None),
            ttft_p99=g(j, "p99_ttft_ms", default=None),
            tpot=g(j, "mean_tpot_ms", default=None),
            itl=g(j, "mean_itl_ms", default=None),
            e2el=g(j, "mean_e2el_ms", default=None),
        ))
    if not rows:
        print(f"[summarize] {d} 下没有可解析的 json"); return 1
    hdr = f"{'run':<22}{'完成':>6}{'时长s':>8}{'req/s':>8}{'out tok/s':>10}{'tot tok/s':>10}{'TTFT ms':>9}{'TTFTp99':>9}{'TPOT ms':>9}{'ITL ms':>8}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        def f(v, n=2):
            try: return f"{float(v):.{n}f}"
            except Exception: return str(v)[:8] if v is not None else "-"
        print(f"{r['name']:<22}{str(r['completed']):>6}{f(r['dur'],1):>8}{f(r['req_tp']):>8}"
              f"{f(r['out_tp']):>10}{f(r['tot_tp']):>10}{f(r['ttft'],1):>9}{f(r['ttft_p99'],1):>9}"
              f"{f(r['tpot']):>9}{f(r['itl']):>8}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

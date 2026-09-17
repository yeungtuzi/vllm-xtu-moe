#!/usr/bin/env python3
# 解析 vLLM torch profiler 的 chrome trace,按 **kernel 名**聚合耗时(§564)。
#
# 为什么需要它:一步解码到底花在哪些 kernel 上(以及 GPU 空档有多大),只能从 trace 读;
# 服务级/CD_TIMING 的 period/compute/rest 只是三个聚合数,定位不到具体算子。
#
# 用法:
#   python3 report/tuning/probes/trace_kernels.py /tmp/engprof/xxx.pt.trace.json.gz
#   ... --top 40 --grep engram|moe|allreduce
#   ... --annotations        # 另打 vLLM 的 gpu_user_annotation(到 Python 帧的归因)
#
# License: Apache-2.0
import argparse
import collections
import gzip
import json
import re
import sys


def _load(path: str):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="ignore") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=35)
    ap.add_argument("--grep", default="")
    ap.add_argument("--annotations", action="store_true")
    args = ap.parse_args()

    tr = _load(args.trace)
    ev = tr["traceEvents"] if isinstance(tr, dict) else tr
    print(f"[trace] {args.trace}: {len(ev)} events")

    by_cat = collections.Counter(e.get("cat", "?") for e in ev)
    print("[trace] cats:", dict(by_cat.most_common(12)))

    spans = [(e.get("ts", 0), e.get("dur", 0)) for e in ev if e.get("ph") == "X"]
    if spans:
        t0 = min(s for s, _ in spans)
        t1 = max(s + d for s, d in spans)
        print(f"[trace] 覆盖窗口 {t1 - t0:.0f} us = {(t1 - t0) / 1000:.1f} ms")

    def _agg(cat: str, label: str):
        agg = collections.defaultdict(lambda: [0, 0.0, 0.0])  # n, total us, max us
        for e in ev:
            if e.get("ph") != "X" or e.get("cat") != cat:
                continue
            a = agg[e.get("name", "?")]
            a[0] += 1
            a[1] += e.get("dur", 0.0)
            a[2] = max(a[2], e.get("dur", 0.0))
        tot = sum(a[1] for a in agg.values())
        print(f"\n[{'kernel' if cat == 'kernel' else cat}] 共 {len(agg)} 个名字,总 {tot / 1000:.1f} ms")
        for name, (n, t, mx) in sorted(agg.items(), key=lambda kv: -kv[1][1])[: args.top]:
            print(f"  {t / 1000:9.2f} ms  n={n:5d}  mean={t / n:8.1f} us  max={mx:8.1f} us  {name[:110]}")
        return agg

    ker = _agg("kernel", "kernel")
    if args.annotations:
        _agg("gpu_user_annotation", "gpu_user_annotation")

    if args.grep:
        pat = re.compile(args.grep, re.I)
        hit = {k: v for k, v in ker.items() if pat.search(k)}
        tot = sum(v[1] for v in hit.values())
        print(f"\n[grep {args.grep}] 命中 {len(hit)} 个名字,总 {tot / 1000:.1f} ms")
        for name, (n, t, mx) in sorted(hit.items(), key=lambda kv: -kv[1][1]):
            print(f"  {t / 1000:9.2f} ms  n={n:5d}  mean={t / n:8.1f} us  {name[:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

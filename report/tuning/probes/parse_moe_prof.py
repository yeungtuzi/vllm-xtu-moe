#!/usr/bin/env python
"""把引擎的 `[MOE-PROF]` 五段打点解析成**按 batch 规模(M)分组的表**。

用途(§615d):判定 CPU 引擎在小 batch 下的损失落在哪一段:
    A  = 分发/准备      A2 = setup(含 scratch 定尺)
    B0 / B = 主体(GEMM/反量化)   C = 收尾/归约   ovh = 其它开销
从而决定改 ①线程池/并行度 ②grouping 的 F 阈值 ③N-slice 内核选择。

用法:
    python report/tuning/probes/parse_moe_prof.py report/tuning/logs/<tag>.memfoot.log [--csv out.csv]
"""
import re
import sys
import statistics as st
from collections import defaultdict

# **实际格式是分桶版**(moe_v2.hpp 的 `[NS-PROF]`),按 M 分桶、每 200 次打印:
#   [NS-PROF] bucket=M<=2 calls=40 na=6.0 | per-call(us): setup=123 A=45 B=678 C=9 ovh=3 TOTAL=858
# (M<=2 = 解码/dspark draft;M3-8 = 小 prefill;M>8 = 真 prefill)
PAT = re.compile(
    r"\[NS-PROF\]\s*bucket=(\S+)\s+calls=(\d+)\s+na=([0-9.]+)\s*\|\s*per-call\(us\):\s*"
    r"setup=([0-9.]+)\s+A=([0-9.]+)\s+B=([0-9.]+)\s+C=([0-9.]+)\s+ovh=([0-9.]+)\s+TOTAL=([0-9.]+)")
# 旧的聚合格式([MOE-PROF])也保留兼容
PAT_OLD = re.compile(
    r"\[MOE-PROF\]\s*calls=(\d+)\s+na=(\S+)\s+M=(\S+).*?"
    r"A=([0-9.]+)ms\s+A2=([0-9.]+)ms\s+B0=([0-9.]+)ms\s+B=([0-9.]+)ms\s+C=([0-9.]+)ms\s+ovh=([0-9.]+)ms")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    rows = defaultdict(list)
    with open(path, errors="ignore") as fh:
        for line in fh:
            m = PAT.search(line)
            if m:
                bucket, calls, na = m.group(1), int(m.group(2)), float(m.group(3))
                setup, A, B, C, ovh = (float(m.group(i)) for i in range(4, 8))
                tot = float(m.group(9))
                rows[bucket].append((calls, na, setup, A, B, C, ovh, tot))
                continue
            mo = PAT_OLD.search(line)
            if mo:
                rows[f"M={mo.group(3)}"].append(
                    (int(mo.group(1)), float(mo.group(2)), float(mo.group(5)),
                     float(mo.group(4)), float(mo.group(6)) + float(mo.group(7)),
                     float(mo.group(8)), float(mo.group(9)),
                     sum(float(mo.group(i)) for i in range(4, 10))))
    hdr = (f"{'bucket':>8}{'calls':>7}{'na':>7}"
           f"{'setup':>9}{'A':>8}{'B':>8}{'C':>8}{'ovh':>7}{'TOTAL':>9}")
    print(hdr)
    print("-" * len(hdr))
    for b in sorted(rows):
        sm = rows[b]
        n = len(sm)
        avg = lambda i: st.mean(x[i] for x in sm)  # noqa: E731
        print(f"{b:>8}{avg(0):>7.0f}{avg(1):>7.1f}"
              + "".join(f"{avg(i):>9.1f}" for i in range(2, 8)))
    print("\n每列 = **per-call 微秒**(即每层的耗时);bucket 由 M=batch token 数决定。")
    print("  M<=2  = 解码(1 + 投机 token)、dspark draft")
    print("  M3-8  = 很小的 prefill")
    print("  M>8   = 真 prefill(ShareGPT/长 prompt)")
    print("\n读法:若 M>8 桶里 setup 或 A 占比大 ⇒ 改线程池/分组;若 B 占绝大部分 ⇒ 是内核算力本身。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

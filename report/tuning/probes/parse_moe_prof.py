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

# [MOE-PROF] calls=… na=… M=… maxme=… skew(...)=… A=…ms A2=…ms B0=…ms B=…ms C=…ms ovh=…ms (sum …ms)
NUM = r"([0-9.]+)"
PAT = re.compile(
    r"\[MOE-PROF\]\s*calls=(\d+)\s+na=(\S+)\s+M=(\S+).*?"
    rf"A={NUM}ms\s+A2={NUM}ms\s+B0={NUM}ms\s+B={NUM}ms\s+C={NUM}ms\s+ovh={NUM}ms")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    rows = defaultdict(list)
    with open(path, errors="ignore") as fh:
        for line in fh:
            m = PAT.search(line)
            if not m:
                continue
            na, M = float(m.group(2)), float(m.group(3))
            vals = [float(m.group(i)) for i in range(4, 10)]
            rows[int(round(M))].append((na, vals))
    if not rows:
        print(f"[parse] {path} 里没有可解析的 [MOE-PROF] 行")
        print("        提示:XIAOTU_MOE_PROFILE=1 才开;打印是**按窗口**的,窗口大小见 moe_v2.hpp")
        return 1
    hdr = (f"{'M(token)':>9}{'na(分配)':>10}{'样本':>5}"
           f"{'A':>8}{'A2':>8}{'B0':>8}{'B':>8}{'C':>8}{'ovh':>8}{'sum':>9}{'tok/s':>9}")
    print(hdr)
    print("-" * len(hdr))
    for M in sorted(rows):
        samples = rows[M]
        na = st.mean(s[0] for s in samples)
        cols = [st.mean(s[1][i] for s in samples) for i in range(6)]
        tot = sum(cols)
        # 单次调用(一层)的 ms ⇒ 该步 40 层 ⇒ 该 M 的 token 吞吐
        tps = M / (tot * 40 / 1000.0) if tot > 0 else 0.0
        print(f"{M:>9}{na:>10.0f}{len(samples):>5}"
              + "".join(f"{c:>8.1f}" for c in cols) + f"{tot:>9.1f}{tps:>9.1f}")
    print("\n列含义:A=分发/准备 A2=setup(含 scratch) B0/B=主体 C=收尾/归约 ovh=其它;"
          "\n       sum=单层合计(ms);tok/s=按 40 层估的该 M 的引擎吞吐(与 cd-timing 交叉验证)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

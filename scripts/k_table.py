#!/usr/bin/env python3
"""⛔ **本脚本已作废(2026-10-08,见 docs/EXPERIMENTS.md B351)** ——
  它把 k 当【截断长度】建模(取前 k 个),**前提就是错的**:
  k 是**并行块宽度**(`parallel_drafting_token_id` 重复 k 遍构成草稿 query),
  必须 = `dspark_block_size`(本 checkpoint = 5)⇒ **不可调** ✗
  保留文件仅供追溯;不要再用它下结论 ✓

查表(作废):**dspark 的 k 该取几?** —— 用两份实测拼出 `tokens/步` 与 `成本`,选最优 k。

═══════════════════════════════════════════════════════════════════════════
 为什么要"查表"而不是"扫 3 个 k"
═══════════════════════════════════════════════════════════════════════════
`num_speculative_tokens` 是**启动参数** ⇒ 每个 k 要一次重启(≈40 min)✗
⇒ ⭐ 只**真跑一个点**(用户 2026-10-08 定:`k=3`),其余 k 由本表**推算** ✓

═══════════════════════════════════════════════════════════════════════════
 输入(都是本仓实测,【实测】)
═══════════════════════════════════════════════════════════════════════════
 ① **逐位接受率**(B344,DSH bench 回放,50 条):
      pos0 80.8% / pos1 62.1% / pos2 47.1% / pos3 36.8% / pos4 29.3%
      ⇒ `tokens/步(k) = 1 + Σ_{i<k} a_i`(**可直接算,无需假设**)
      ⭐ 自洽校验:k=5 ⇒ 1+2.561 = **3.561**,与实测 **3.56/3.64** 吻合 ✓
 ② **成本 c(k)**:`docs/QFN_BF16_M16_ABCPERF_2026-10-07.md` 的 CPU MoE `compute`(ms/层,base 臂):
      M=1 → 0.25 · M=4 → 0.55 · M=16 → 1.14
      验证批次有 (k+1) 行 ⇒ 取 M = k+1,分段线性插值 ✓
      ⚠️ **这就是本表最弱的一环**:它只含 **CPU MoE 计算**,不含
         attention(∝k+1)、draft 层 forward、verify/sampling 开销 ⇒ **c(k) 被低估其增长** ✗
      ⇒ 所以另给一个**敏感性模型(GLM 口径)**:`c(5)/c(0) = 2.97`
        (见 `docs/KNOWN_LIMITATIONS.md:107`,GLM 实测"验证一批 5 个 ≈ 2.97× 单个")

 ⚠️ 本表**不是**"k=3 更好"的证明 —— 它是**给窗口一个先验**,并由那一枪去校准 c(k) ✓
"""
from __future__ import annotations

import argparse

# ── ① 逐位接受率(B344 实测)────────────────────────────────────
A = [0.808, 0.621, 0.471, 0.368, 0.293]
# ── ② 成本表(ABCPERF 实测,ms/层)──────────────────────────────
COST_M = {1: 0.25, 4: 0.55, 16: 1.14}


def c_interp(k: int) -> float:
    """验证 (k+1) 行 ⇒ c ≈ interp(M=k+1)。"""
    m = k + 1
    xs = sorted(COST_M)
    if m <= xs[0]:
        return COST_M[xs[0]]
    if m >= xs[-1]:
        return COST_M[xs[-1]]
    for lo, hi in zip(xs, xs[1:]):
        if lo <= m <= hi:
            t = (m - lo) / (hi - lo)
            return COST_M[lo] + t * (COST_M[hi] - COST_M[lo])
    raise AssertionError


def c_glm(k: int) -> float:
    """GLM 口径的敏感性:c(k)/c(0) 线性到 c(5)/c(0)=2.97。"""
    return 1.0 + (2.97 - 1.0) * k / 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conc", type=int, default=1, help="并发数(只影响 M 的绝对尺度,不影响排序)")
    args = ap.parse_args()
    print(f"逐位接受率(B344 实测): {['%.3f' % a for a in A]}")
    print(f"成本表(ABCPERF 实测,ms/层): {COST_M}")
    print()
    for name, fn in (("ABCPERF 口径(仅 CPU MoE)", c_interp), ("GLM 口径(含验证开销,敏感性)", c_glm)):
        print(f"=== {name} ===")
        print(f"{'k':>2} {'tokens/步':>10} {'c(k)':>8} {'收益/成本':>10} {'相对 k=0':>9} {'相对 k=5':>9}")
        base = None
        rows = []
        for k in range(0, 6):
            tps = 1.0 + sum(A[:k])
            c = fn(k)
            eff = tps / c
            rows.append((k, tps, c, eff))
        k5 = [r[3] for r in rows if r[0] == 5][0]
        k0 = [r[3] for r in rows if r[0] == 0][0]
        for k, tps, c, eff in rows:
            print(f"{k:>2} {tps:>10.3f} {c:>8.3f} {eff:>10.3f} "
                  f"{eff/k0:>8.3f}× {eff/k5:>8.3f}×")
        best = max(rows, key=lambda r: r[3])
        print(f"  ⇒ 本口径最优 k = **{best[0]}**(收益/成本 {best[3]:.3f})")
        print()
    print("⚠️ 两口径若结论不同 ⇒ **必须靠窗口那一枪(k=3)校准 c(k)**,不能凭表定案 ✗")
    print("⭐ 判据(上机):同一份 DSH bench 回放,比 `tokens/步` 与 **输出 tok/s**;")
    print("   基线(B344 实测,k=5):tokens/步 3.56~3.64 · 输出 15.89 / 17.13 tok/s ✓")


if __name__ == "__main__":
    main()

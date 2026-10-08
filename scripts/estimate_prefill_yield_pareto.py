#!/usr/bin/env python3
"""R4 —— 「chunk 间插入解码步(协作式让步)」的离线 Pareto 估算。

⚠️ 本脚本是**纯离线计算器**:不联网、不碰 GPU、不启动/停止任何进程
   ⇒ 【不属于】AGENTS.md「脚本复审门(R38)」的范围(那条只管启动/杀进程的脚本)✓

═══════════════════════════════════════════════════════════════════════════
 动机(本仓 B273 自己标为「未做」的那条路)
═══════════════════════════════════════════════════════════════════════════
B273(2026-10-04)末尾原文:
  > **⇒ 待验证的另一条路(未做)**:为解码**预留步数**(如按时间片轮转),而不是限 token;
  >   vLLM 现成旋钮里没看到 ✓

B273/B277 实测否决的是【限 token】(压 chunk):`--long-prefill-token-threshold 2048`
⇒ 每个 chunk 都要**重新流式过 40 层权重** ⇒ 预填效率崩 ⇒ 窗口涨得比停顿降得多 ✗

本脚本量化的是**另一个自变量**:保持 chunk 大小不变(权重只流一次),
在**两个 chunk 之间插入 D 个「纯解码步」**。

═══════════════════════════════════════════════════════════════════════════
 物理模型(全部锚在 B277/B279 的实测上,不引入任何新常数)
═══════════════════════════════════════════════════════════════════════════
关键区分(本估算的全部要点):
  · **预填步**:一步内塞入 C 个 prefill token ⇒ 该步要流式 40 层专家权重 ⇒ t ≈ t_chunk
  · **纯解码步**:一步内只有 1 个 decode token ⇒ 专家走 CPU 路径 ⇒ t ≈ t_dec = 64.8 ms
  ⇒ 「步数变多」本身不贵;**贵的是「预填步变多」**(每个都要重流 136 GB)✗

今天的调度(B277 已钉死):
  每一步 = 一个 MBT 大小的预填 chunk + 1 个解码 token
  ⇒ 解码的 ITL = t_chunk,且整个预填窗口内持续

插入 D 个纯解码步之后(每个 chunk 之间):
  解码每个 chunk 周期拿到 (D+1) 个 token
  ITL(D)    = (t_chunk + D·t_dec) / (D+1)
  window(D) = N · (t_chunk + D·t_dec)          [N = chunk 数]
  代价      = D·t_dec 的墙钟;⚠️ **不额外流式权重**(流水线里的层已完成,不重流)✓

实测锚点(全部来自 docs/EXPERIMENTS.md):
  B277 ①  MAXSEQS=4 / MBT=8192 / LPT=0
          A 基线的 median ITL        = 64.8 ms      ← 即 t_dec
          A 在 B 的预填窗口内每 ~8.4 s 拿到 1 个 token
          A 时间轴 7 个间隔(7.95/8.15/8.38/8.62/8.85/9.05/6.29)⇒ N = 7 个 chunk
          B 的预填窗口                = 57.6 s      (= 7 × 8.2 s ✓ 自洽)
  B277 ②  LPT=2048(固定):每步停顿 5.0 s(1.7× 好),窗口 132.8 s(2.3× 差)
  B279    2048-token chunk ≈ 410 tok/s;scaling 出**每 chunk 固定成本**
  B279    staging 每 chunk 流式 40 × 3.40 GB = 136 GB H2D

⇒ 由 B277 的两个点反解「每 chunk 固定成本 F」与「每 token 计算成本 1/Rc」:
     t_chunk(C) = F + C/Rc
     C=2048 ⇒ 5.0 ; C=8192 ⇒ 8.2   ⇒  Rc = 1920 tok/s , F = 3.93 s
   ⭐ F = 3.93 s/chunk ⇒ 有效聚合 H2D ≈ 136 GB / 3.93 s ≈ **34.6 GB/s**(TP=2)
      ⇒ C=2048 时 **79% 的 chunk 时间**是纯流式开销;C=8192 时 48% ✗

用法:
  python scripts/estimate_prefill_yield_pareto.py            # 主表
  python scripts/estimate_prefill_yield_pareto.py --check    # 与实测对账(自检)
"""
from __future__ import annotations

import argparse

# ─── 实测锚点(唯一真源:docs/EXPERIMENTS.md B277 / B279)──────────────────
T_DEC = 0.0648        # s   —— A 的基线 median ITL(C=4,注入前)       [B277 ①]
T_CHUNK_8192 = 8.2    # s   —— 8192-token 预填 chunk 的步时长          [B277 ①]
N_CHUNKS = 7          # 个  —— A 在窗口内拿到的 token 数 = chunk 数    [B277 ① 时间轴]
WINDOW_BASE = 57.6    # s   —— B 的预填窗口(32768 字面量 token 的 prompt)[B277 ①]
# LPT=2048 实测臂(负结果,用于对照)
LPT_T_CHUNK = 5.0     # s                                            [B277 ②]
LPT_WINDOW = 132.8    # s                                            [B277 ②]
LPT_T_CHUNK_TOKENS = 2048
# V4.1 形状(B279):每 chunk 流式的专家字节
STREAM_GB_PER_CHUNK = 40 * 3.40    # 40 层 × 3.40 GB = 136 GB         [B279]


def fit_chunk_cost() -> tuple[float, float]:
    """由两个实测点反解 t_chunk(C) = F + C/Rc ⇒ (F 秒, Rc tok/s)。"""
    c1, t1 = LPT_T_CHUNK_TOKENS, LPT_T_CHUNK
    c2, t2 = 8192, T_CHUNK_8192
    rc = (c2 - c1) / (t2 - t1)          # tok/s
    f = t1 - c1 / rc                    # s
    return f, rc


def itl_and_window(d_steps: float, t_chunk: float = T_CHUNK_8192,
                   n_chunks: int = N_CHUNKS, t_dec: float = T_DEC) -> tuple[float, float]:
    """插入 d_steps 个纯解码步后的 (ITL, 窗口)。"""
    period = t_chunk + d_steps * t_dec
    itl = period / (d_steps + 1.0)
    return itl, n_chunks * period


def solve_d_for_window_mult(mult: float, t_chunk: float = T_CHUNK_8192,
                            n_chunks: int = N_CHUNKS, t_dec: float = T_DEC) -> float:
    """给定窗口倍率,解出需要的 d_steps。"""
    period = mult * t_chunk                      # 因 window_base = n_chunks * t_chunk
    return max(0.0, (period - t_chunk) / t_dec)


def solve_d_for_itl(target_itl: float, t_chunk: float = T_CHUNK_8192,
                    t_dec: float = T_DEC) -> float:
    """给定目标 ITL,解出需要的 d_steps(解 (t_chunk + d·t_dec)/(d+1) = target)。"""
    if target_itl >= t_chunk:
        return 0.0
    # t_chunk + d·t_dec = target·(d+1)  ⇒  d(t_dec - target) = target - t_chunk
    return (target_itl - t_chunk) / (t_dec - target_itl)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="与实测对账(自检)")
    ap.add_argument("--t-chunk", type=float, default=T_CHUNK_8192)
    ap.add_argument("--n-chunks", type=int, default=N_CHUNKS)
    ap.add_argument("--t-dec", type=float, default=T_DEC)
    args = ap.parse_args()

    t_chunk, n_chunks, t_dec = args.t_chunk, args.n_chunks, args.t_dec

    F, Rc = fit_chunk_cost()
    window_base = n_chunks * t_chunk

    if args.check:
        print("=== 自检:模型 vs 实测 ===")
        print(f"反解每-chunk 固定成本 F = {F:.3f} s ;每-token 计算 Rc = {Rc:.0f} tok/s")
        print(f"  ⇒ 有效聚合 H2D = {STREAM_GB_PER_CHUNK:.0f} GB / {F:.3f} s = "
              f"{STREAM_GB_PER_CHUNK / F:.1f} GB/s")
        for c, t_meas in ((2048, LPT_T_CHUNK), (8192, T_CHUNK_8192)):
            t_pred = F + c / Rc
            print(f"  C={c:5d}  实测 {t_meas:5.2f} s  模型 {t_pred:5.2f} s  "
                  f"误差 {100 * (t_pred - t_meas) / t_meas:+.1f}%   "
                  f"({c / t_pred:6.0f} tok/s)")
        print(f"  窗口自洽:N={n_chunks} × t_chunk={t_chunk:.1f} s = "
              f"{window_base:.1f} s (实测 {WINDOW_BASE:.1f} s,"
              f"误差 {100 * (window_base - WINDOW_BASE) / WINDOW_BASE:+.1f}%)")
        print(f"  LPT=2048 对照:窗口 {LPT_WINDOW:.1f} s = {LPT_WINDOW / LPT_T_CHUNK:.1f} × "
              f"chunk ⇒ 与 N={n_chunks} 相比 chunk 数 ×{LPT_WINDOW / LPT_T_CHUNK / n_chunks:.2f}")
        return

    print("=" * 108)
    print("R4 离线 Pareto:【chunk 间插入纯解码步】 vs 【压 chunk(LPT=2048,已实测否决)】")
    print("=" * 108)
    print(f"锚点(B277):t_chunk={t_chunk:.1f} s/步,N={n_chunks} 个 chunk,"
          f"基线窗口={window_base:.1f} s,t_dec={t_dec * 1000:.1f} ms")
    print(f"反解(B277 两点):每-chunk 固定流式成本 F={F:.2f} s "
          f"(≈{STREAM_GB_PER_CHUNK / F:.1f} GB/s 聚合);每-token 计算 Rc={Rc:.0f} tok/s")
    print()

    print("── 表 1:插入 D 个纯解码步的 Pareto 前沿 " + "─" * 62)
    print(f"{'D':>6} {'解码份额f':>10} {'窗口(s)':>9} {'窗口×':>7} "
          f"{'ITL(s)':>9} {'ITL 改善':>9} {'窗口内解码tok/s':>16}")
    for d in (0, 1, 2, 3, 5, 8, 13, 20, 32, 63, 100, 165, 250):
        itl, win = itl_and_window(d, t_chunk, n_chunks, t_dec)
        f = d * t_dec / t_chunk
        toks = n_chunks * (d + 1)
        print(f"{d:>6} {f:>10.3f} {win:>9.1f} {win / window_base:>7.3f} "
              f"{itl:>9.3f} {t_chunk / itl:>8.1f}× {toks / win:>16.2f}")
    print()

    print("── 表 2:与【已实测否决的 LPT=2048】同口径对比 " + "─" * 55)
    lpt_mult = LPT_WINDOW / window_base
    d_same_win = solve_d_for_window_mult(lpt_mult, t_chunk, n_chunks, t_dec)
    itl_same_win, _ = itl_and_window(d_same_win, t_chunk, n_chunks, t_dec)
    d_same_itl = solve_d_for_itl(LPT_T_CHUNK, t_chunk, t_dec)
    itl_chk, win_same_itl = itl_and_window(d_same_itl, t_chunk, n_chunks, t_dec)
    print(f"{'':38}{'LPT=2048(实测)':>18}{'插入解码步(模型)':>20}")
    print(f"{'ITL(每步停顿)':38}{LPT_T_CHUNK:>16.2f} s{itl_same_win:>18.3f} s")
    print(f"{'预填窗口':38}{LPT_WINDOW:>16.1f} s{n_chunks * (t_chunk + d_same_win * t_dec):>18.1f} s")
    print(f"{'窗口倍率':38}{lpt_mult:>17.3f}×{lpt_mult:>19.3f}×")
    print(f"  ⇒ ⭐ 同窗口代价下:插入解码步的 ITL 比 LPT 好 "
          f"{LPT_T_CHUNK / itl_same_win:.0f}×(需 D={d_same_win:.0f},整数)")
    print()
    print(f"  ⇒ ⭐ 同 ITL({LPT_T_CHUNK:.1f} s)代价下:插入解码步的窗口 "
          f"{win_same_itl:.1f} s vs LPT {LPT_WINDOW:.1f} s")
    print(f"      ⇒ LPT 的窗口代价是它的 {LPT_WINDOW / win_same_itl:.1f} 倍"
          f"(连续解 D={d_same_itl:.2f};⚠️ D 必须取整)")
    d_int = max(1, round(d_same_itl))
    itl_int, win_int = itl_and_window(d_int, t_chunk, n_chunks, t_dec)
    print(f"      ⇒ 取【可实现的 D={d_int}】:ITL {itl_int:.2f} s"
          f"({t_chunk / itl_int:.1f}× 改善),窗口 {win_int:.1f} s"
          f"({win_int / window_base:.3f}×)⇒ 仍远优于 LPT 的 {LPT_T_CHUNK:.1f} s / "
          f"{LPT_WINDOW / window_base:.3f}× ✓")
    print()

    print("── 表 3:若给生产设一个「窗口增长 ≤ X%」的预算 " + "─" * 55)
    print(f"{'窗口预算':>10}{'f':>10}{'D':>7}{'ITL(s)':>10}{'ITL 改善':>10}"
          f"{'窗口(s)':>10}{'窗口内解码tok/s':>16}")
    for budget in (0.01, 0.02, 0.05, 0.10, 0.20, 0.50):
        d = solve_d_for_window_mult(1.0 + budget, t_chunk, n_chunks, t_dec)
        itl, win = itl_and_window(d, t_chunk, n_chunks, t_dec)
        print(f"{budget:>9.0%} {d * t_dec / t_chunk:>10.4f} {d:>7.1f}"
              f"{itl:>10.3f}{t_chunk / itl:>9.1f}×{win:>10.1f}"
              f"{n_chunks * (d + 1) / win:>16.2f}")
    print()

    print("── 表 4:对照 —— 压 chunk(改 MBT/LPT)的 Pareto(模型,已由实测校准)" + "─" * 30)
    print(f"{'MBT':>8}{'chunk(s)':>10}{'chunk数':>9}{'窗口(s)':>10}{'窗口×':>8}"
          f"{'ITL(s)':>9}{'tok/s':>9}")
    base_tokens = n_chunks * 8192        # 由 N=7 × 8192 反推的 prompt token 量
    for c in (8192, 6144, 4096, 2048, 1024):
        tc = F + c / Rc
        nc = base_tokens / c
        print(f"{c:>8}{tc:>10.2f}{nc:>9.1f}{nc * tc:>10.1f}{nc * tc / window_base:>8.3f}"
              f"{tc:>9.2f}{c / tc:>9.0f}")
    print()
    print("⇒ ⭐ 结论:压 chunk 是**用「每个 chunk 重付 F=3.93 s 流式成本」换停顿**,")
    print("   而插入纯解码步是**用「纯解码时间」换停顿,不重付流式成本** ⇒ 后者 Pareto 显著更好 ✓")


if __name__ == "__main__":
    main()

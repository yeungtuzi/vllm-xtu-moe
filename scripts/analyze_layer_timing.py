#!/usr/bin/env python3
"""把 `XIAOTU_LAYER_TIMING=1` 的输出解析成"每层 CPU-MoE vs 等 GPU"的时间序列。

背景(目标 goal-e5d81755,步骤 ①):我们已知**单流解码时 CPU 只有 47-72% 在跑**(用
`utime+stime` 增速实测),且判据表明那是**真的在等**(访存停顿仍算 running)⇒ 每层
「CPU MoE」与「GPU 工作」是交替而非重叠。要动手之前必须先把**每层**的两段量化出来。

打点本身仓库里已有(`vllm_xiaotu_moe/mixed_experts.py:76 _lt_record`,env
`XIAOTU_LAYER_TIMING=1` + `XIAOTU_LAYER_TIMING_EVERY=1` 每层一行):
    [layer-timing] t=<墙钟秒> n=<累计调用数> layer=<层名> pre=A ms eng=B ms post=C ms total=T ms
其中 **eng = 引擎(CPU MoE)调用**,pre/post = 其前后(装配/结果转换)。
⚠️ 打印的是**累计平均**,逐层值要靠相邻两行反解:
    x_k = avg_n·n − avg_{n−1}·(n−1)          (注释原文给的公式)
而 **Δt = 相邻两行的 `t` 之差 = 两次 MoE 调用之间的墙钟** ⇒
    **outside = Δt − total** = 「apply 之外」= attention / indexer / GPU 段 ✓

输出三类判据:
  1. `CPU 占空比 = eng / Δt` —— 与 `utime+stime` 实测的 47-72% 对照
  2. `等 GPU 占比 = outside / Δt`
  3. `eng / total`(apply 内 CPU MoE 的占比)

用法:
  # 直接读日志
  python scripts/analyze_layer_timing.py <log> [--tail N] [--warmup K]
  # 自测(合成数据,验证反解公式与统计都对)
  python scripts/analyze_layer_timing.py --selftest
"""
from __future__ import annotations

import argparse
import re
import statistics as st
import sys

LINE = re.compile(
    r"\[layer-timing\]\s+t=(?P<t>[0-9.]+)\s+(?:pid=(?P<pid>\d+)\s+)?n=(?P<n>\d+)\s+layer=(?P<layer>\S+)\s+"
    r"pre=(?P<pre>[0-9.]+)ms\s+eng=(?P<eng>[0-9.]+)ms\s+post=(?P<post>[0-9.]+)ms\s+"
    r"total=(?P<total>[0-9.]+)ms"
)


def parse(lines):
    """⇒ [(t, n, layer, pre, eng, post, total)] —— 都是打印出来的【累计平均】。"""
    out = []
    for ln in lines:
        m = LINE.search(ln)
        if not m:
            continue
        out.append((float(m["t"]), int(m["n"]), m["layer"],
                    float(m["pre"]), float(m["eng"]), float(m["post"]), float(m["total"]),
                    m.group("pid") or "-"))
    return out


def split_runs(rows):
    """优先**按 pid 分组**(打印已带 pid ⇒ 最可靠);无 pid 时退回"按 n 回退切分"。"""
    """按 `n` 单调递增切分。

    ⚠️ **必须切**:插件用 `print()` 输出(走 stdout,**没有 pid 前缀**),而 TP=2 时**两个
    worker 各有一份 `_LT_ACC`** ⇒ 两路日志在同一个文件里**交错** ✗。若直接反解,
    `avg_n·n − avg_{n−1}·(n−1)` 会把两个累加器混在一起 ⇒ 出现**负值** ✗(本会话实测踩到)。
    `n` 回退即意味着"换了另一个累加器" ⇒ 在此处断开 ✓
    """
    if rows and len(rows[0]) > 7 and any(r[7] != "-" for r in rows):
        by = {}
        for r in rows:
            by.setdefault(r[7], []).append(r)
        return [by[k] for k in by]
    runs, cur = [], []
    for r in rows:
        if cur and r[1] <= cur[-1][1]:
            runs.append(cur)
            cur = []
        cur.append(r)
    if cur:
        runs.append(cur)
    return runs


def per_call(rows):
    """用 x_k = avg_n·n − avg_{n−1}·(n−1) 反解出**单次调用**的值。"""
    res = []
    for i in range(1, len(rows)):
        t, n, layer, pre, eng, post, total, _pid = rows[i]
        _, n0, _, pre0, eng0, post0, total0, _pid0 = rows[i - 1]
        dn = n - n0
        if dn <= 0:
            continue
        dt = (t - rows[i - 1][0]) * 1e3          # 相邻两行墙钟(ms)
        res.append({
            "layer": layer,
            "pre": (pre * n - pre0 * n0) / dn,
            "eng": (eng * n - eng0 * n0) / dn,
            "post": (post * n - post0 * n0) / dn,
            "total": (total * n - total0 * n0) / dn,
            "dt": dt,
            "outside": dt - (total * n - total0 * n0) / dn,
        })
    return res


def report(calls, warmup=0):
    if warmup:
        calls = calls[warmup:]
    if not calls:
        print("  没有可用的样本(检查日志里有没有 [layer-timing] 行、以及 EVERY 是否够小)")
        return 1
    eng = [c["eng"] for c in calls]
    tot = [c["total"] for c in calls]
    dt = [c["dt"] for c in calls]
    out = [c["outside"] for c in calls]
    duty = [e / d for e, d in zip(eng, dt) if d > 0]
    outr = [o / d for o, d in zip(out, dt) if d > 0]
    inr = [e / t for e, t in zip(eng, tot) if t > 0]
    f = lambda v: f"{st.median(v):.3f}" if v else "n/a"
    print(f"  样本 {len(calls)} 层(dt 异常剔除 {sum(1 for d in dt if d <= 0)} 条)")
    print(f"  {'量':34s} {'中位':>9} {'均值':>9} {'p10':>9} {'p90':>9}")
    for name, v in (("eng  = CPU MoE (ms)", eng),
                    ("total= apply 合计 (ms)", tot),
                    ("dt   = 两次 MoE 之间墙钟 (ms)", dt),
                    ("outside = dt − total(等 GPU)(ms)", out)):
        s = sorted(v)
        print(f"  {name:34s} {st.median(v):9.3f} {st.mean(v):9.3f} "
              f"{s[int(.1*len(s))-1]:9.3f} {s[int(.9*len(s))-1]:9.3f}")
    print()
    print(f"  ⭐ CPU 占空比 eng/dt      = {f(duty)}   ← 与实测 utime+stime 的 47–72% 对照")
    print(f"     等 GPU 占比 outside/dt = {f(outr)}")
    print(f"     apply 内 eng/total    = {f(inr)}")
    print()
    mid = st.median(duty)
    if mid >= 0.90:
        print(f"  结论:CPU 占空比 {mid*100:.0f}% ⇒ **已经接近满载**:重叠基本到位,余量 ~{1/mid:.2f}×")
    elif mid >= 0.65:
        print(f"  结论:CPU 占空比 {mid*100:.0f}% ⇒ **有明显空闲**:理论上限 ~{1/mid:.2f}× ⇒ 值得做重叠")
    else:
        print(f"  结论:CPU 占空比 {mid*100:.0f}% ⇒ **大量空闲**(可能含 worker park):上限 ~{1/mid:.2f}×")
    return 0


def selftest():
    """合成一份日志:每层 eng=1.2ms、outside=0.5ms、pre/post 各 0.05ms ⇒ 占空比应 ≈ 1.2/1.8 = 0.667"""
    eng_t, out_t, pre_t, post_t = 1.2, 0.5, 0.05, 0.05
    tot_t = pre_t + eng_t + post_t
    dt_t = tot_t + out_t
    lines, t, cn = [], 1000.0, 0
    cpre = ceng = cpost = ctot = 0.0
    for k in range(1, 201):
        cpre += pre_t; ceng += eng_t; cpost += post_t; ctot += tot_t
        cn += 1
        t += dt_t / 1e3
        lines.append(f"[layer-timing] t={t:.6f} n={cn} layer=language_model.model.layers.{k%40}.ffn.experts "
                     f"pre={cpre/cn:.3f}ms eng={ceng/cn:.3f}ms post={cpost/cn:.3f}ms total={ctot/cn:.3f}ms")
    calls = per_call(parse(lines))
    assert len(calls) == 199, len(calls)
    c = calls[-1]
    ok = (abs(c["eng"] - eng_t) < 1e-6 and abs(c["outside"] - out_t) < 1e-6
          and abs(c["total"] - tot_t) < 1e-6)
    print(f"  合成用例:eng={c['eng']:.4f}(期望 {eng_t}) outside={c['outside']:.4f}(期望 {out_t}) "
          f"total={c['total']:.4f}(期望 {tot_t:.4f}) ⇒ {'✅ 反解正确' if ok else '❌ 反解错误'}")
    rc = report(calls, warmup=5)
    expect = eng_t / dt_t
    print(f"  (自测期望占空比 {expect:.3f};上面应报告 ≈0.65–0.67 ⇒ 判定为『有明显空闲』)")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?")
    ap.add_argument("--tail", type=int, default=0, help="只看日志最后 N 行")
    ap.add_argument("--warmup", type=int, default=0, help="丢弃前 K 个样本(预热)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.log:
        return selftest()
    with open(a.log, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    if a.tail:
        lines = lines[-a.tail:]
    rows = parse(lines)
    runs = split_runs(rows)
    print(f"  从 {a.log} 抓到 {len(rows)} 行 [layer-timing] ⇒ 切成 **{len(runs)} 个 run**"
          f"(TP=2 时应有 2 个累加器;每 run ≥2 行才算得出逐层值)"
          f";正式跑请设 XIAOTU_LAYER_TIMING_EVERY=1")
    calls = []
    for r in runs:
        calls.extend(per_call(r))
    return report(calls, warmup=a.warmup)


if __name__ == "__main__":
    sys.exit(main())

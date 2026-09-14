#!/usr/bin/env python
"""生成项目报告(PPT)用的插图。

所有数字来自本机实测(见 report/tuning/NOTES.md §300–§327 与 report/tuning/raw/)。
输出:`docs/figures/*.png`(200 dpi,16:9 友好)。

用法:
  python scripts/make_report_figures.py            # 全部生成
  OUT=docs/figures python scripts/make_report_figures.py

字体:优先用 docs/figures/fonts/NotoSansSC-Regular.otf(中文);缺失时退化为英文标签。
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("OUT", os.path.join(ROOT, "docs", "figures"))
os.makedirs(OUT, exist_ok=True)

# ---- 中文字体 ---------------------------------------------------------------
# 字体不入库(8 MB);缺失时自动下载一次,失败则退化为英文标签。
FONT = os.path.join(OUT, "fonts", "NotoSansSC-Regular.otf")
FONT_URL = ("https://cdn.jsdelivr.net/gh/notofonts/noto-cjk@main/"
            "Sans/SubsetOTF/SC/NotoSansSC-Regular.otf")
if not os.path.exists(FONT):
    os.makedirs(os.path.dirname(FONT), exist_ok=True)
    try:
        import urllib.request
        with urllib.request.urlopen(FONT_URL, timeout=90) as r, open(FONT, "wb") as f:
            f.write(r.read())
        print("downloaded CJK font ->", FONT)
    except Exception as e:  # noqa: BLE001
        print("CJK font unavailable (%s); falling back to English labels" % e, file=sys.stderr)
CJK = False
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = ["Noto Sans SC"]
    CJK = True
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "#666666"
plt.rcParams["axes.labelcolor"] = "#222222"
plt.rcParams["text.color"] = "#222222"
plt.rcParams["xtick.color"] = "#444444"
plt.rcParams["ytick.color"] = "#444444"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#DDDDDD"
plt.rcParams["grid.linewidth"] = 0.8
plt.rcParams["axes.axisbelow"] = True
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"

# 品牌色
C_OURS = "#1F6FEB"      # 蓝:本项目
C_REF = "#B0B7C3"       # 灰:参考闭源引擎
C_ACC = "#E8590C"       # 橙:强调/改进
C_OK = "#2F9E44"        # 绿:达标
C_WARN = "#F08C00"      # 黄:未达标


def T(zh, en):
    return zh if CJK else en


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)


# =============================================================================
# 图 1:主结果 —— 同机 / 同协议 / 同配置,唯一差别是引擎
# =============================================================================
def fig_main_result():
    conc = ["C=1", "C=2", "C=4"]
    ours_tpot = [28.13, 30.55, 33.60]
    ref_tpot = [23.11, 31.75, 41.88]
    ours_agg = [32.67, 59.89, 100.16]
    ref_agg = [41.04, 59.73, 86.92]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    x = range(len(conc))
    w = 0.36

    ax = axes[0]
    b1 = ax.bar([i - w / 2 for i in x], ours_tpot, w, label=T("vllm-xtu-moe(本项目)", "vllm-xtu-moe (ours)"),
                color=C_OURS)
    b2 = ax.bar([i + w / 2 for i in x], ref_tpot, w, label=T("lk_moe(专有,参考)", "lk_moe (proprietary ref)"),
                color=C_REF)
    ax.axhline(30, ls="--", lw=1.4, color=C_OK)
    ax.text(-0.44, 30.8, T("验收线 30 ms", "target 30 ms"), color=C_OK, fontsize=9.5, ha="left")
    ax.bar_label(b1, fmt="%.2f", fontsize=9, padding=2)
    ax.bar_label(b2, fmt="%.2f", fontsize=9, padding=2)
    ax.set_title(T("单请求解码延迟 TPOT(越低越好)", "Decode latency per token (lower is better)"),
                 fontsize=11, fontweight="bold")
    ax.set_ylabel(T("TPOT (ms)", "TPOT (ms)"))
    ax.set_xticks(list(x))
    ax.set_xticklabels(conc)
    ax.set_ylim(0, 50)
    ax.legend(fontsize=9, frameon=False)

    ax = axes[1]
    b1 = ax.bar([i - w / 2 for i in x], ours_agg, w, label=T("本项目", "ours"), color=C_OURS)
    b2 = ax.bar([i + w / 2 for i in x], ref_agg, w, label=T("lk_moe(专有)", "lk_moe (proprietary)"),
                color=C_REF)
    ax.axhline(50, ls="--", lw=1.4, color=C_OK)
    ax.text(-0.44, 52, T("验收线 50 t/s", "target 50 t/s"), color=C_OK, fontsize=9.5, ha="left")
    ax.axhline(60, ls=":", lw=1.2, color=C_WARN)
    ax.text(-0.44, 62, T("目标 60 t/s", "goal 60 t/s"), color=C_WARN, fontsize=9.5, ha="left")
    ax.bar_label(b1, fmt="%.1f", fontsize=9, padding=2)
    ax.bar_label(b2, fmt="%.1f", fontsize=9, padding=2)
    ax.set_title(T("聚合解码吞吐(越高越好)", "Aggregate decode throughput (higher is better)"),
                 fontsize=11, fontweight="bold")
    ax.set_ylabel(T("聚合吞吐 (tok/s)", "aggregate (tok/s)"))
    ax.set_xticks(list(x))
    ax.set_xticklabels(conc)
    ax.set_ylim(0, 120)
    ax.legend(fontsize=9, frameon=False)

    fig.suptitle(T("主结果:2×A100-40GB · TP=2 · 12 层 GPU 常驻 · 不投机解码 · L=256/OUT=512",
                   "Main result: 2xA100-40GB, TP=2, 12 GPU-resident MoE layers, no spec decoding"),
                 fontsize=11.5, fontweight="bold", y=1.04)
    save(fig, "fig01_main_result.png")


# =============================================================================
# 图 2:手段 3(异步握手)的同协议 A/B —— 因果归因
# =============================================================================
def fig_async_ab():
    conc = ["C=1", "C=2", "C=4"]
    before = [28.76, 36.12, 48.35]
    after = [28.10, 30.49, 33.51]
    agg_b = [32.46, 51.29, 71.78]
    agg_a = [32.71, 59.98, 100.47]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    x = range(len(conc))
    w = 0.36
    ax = axes[0]
    b1 = ax.bar([i - w / 2 for i in x], before, w, label=T("改造前:cudaLaunchHostFunc", "before: cudaLaunchHostFunc"),
                color=C_REF)
    b2 = ax.bar([i + w / 2 for i in x], after, w, label=T("改造后:常驻 worker + mapped flag", "after: worker + mapped flag"),
                color=C_OURS)
    for i, (a, b) in enumerate(zip(before, after)):
        ax.text(i, max(a, b) + 1.4, f"{(b/a-1)*100:+.1f}%", ha="center", fontsize=9.5,
                color=C_OK if b < a else C_ACC, fontweight="bold")
    ax.bar_label(b1, fmt="%.2f", fontsize=9, padding=2)
    ax.bar_label(b2, fmt="%.2f", fontsize=9, padding=2)
    ax.set_title(T("每 token 解码延迟 TPOT", "TPOT per token"), fontsize=11, fontweight="bold")
    ax.set_ylabel("TPOT (ms)")
    ax.set_xticks(list(x)); ax.set_xticklabels(conc)
    ax.set_ylim(0, 58)
    ax.legend(fontsize=9, frameon=False)

    ax = axes[1]
    b1 = ax.bar([i - w / 2 for i in x], agg_b, w, label=T("改造前", "before"), color=C_REF)
    b2 = ax.bar([i + w / 2 for i in x], agg_a, w, label=T("改造后", "after"), color=C_OURS)
    for i, (a, b) in enumerate(zip(agg_b, agg_a)):
        ax.text(i, max(a, b) + 3.5, f"{(b/a-1)*100:+.0f}%", ha="center", fontsize=9.5,
                color=C_OK, fontweight="bold")
    ax.bar_label(b1, fmt="%.1f", fontsize=9, padding=2)
    ax.bar_label(b2, fmt="%.1f", fontsize=9, padding=2)
    ax.set_title(T("聚合吞吐", "Aggregate throughput"), fontsize=11, fontweight="bold")
    ax.set_ylabel(T("tok/s", "tok/s"))
    ax.set_xticks(list(x)); ax.set_xticklabels(conc)
    ax.set_ylim(0, 125)
    ax.legend(fontsize=9, frameon=False)

    fig.suptitle(T("手段 3 落地:异步握手取代 cudaLaunchHostFunc(同协议 A/B,其余配置完全相同)",
                   "Technique 3: async handshake replaces cudaLaunchHostFunc (same protocol, all else equal)"),
                 fontsize=11.5, fontweight="bold", y=1.04)
    save(fig, "fig02_async_ab.png")


# =============================================================================
# 图 3:优化历程
# =============================================================================
def fig_history():
    stages = [
        T("① 基线\n43 层全 CPU", "1. baseline\nall 43 layers on CPU"),
        T("② +5 层常驻", "2. +5 GPU-resident"),
        T("③ +11 层常驻\n+PERMV 解码", "3. +11 resident\n+PERMV decode"),
        T("④ +12 层常驻\n+异步握手", "4. +12 resident\n+async handshake"),
    ]
    tpot = [37.33, 33.82, 31.78, 28.13]
    agg = [34.29, 55.23, 60.17, 100.16]

    fig, ax = plt.subplots(figsize=(10.6, 4.5))
    x = range(len(stages))
    b1 = ax.bar([i - 0.2 for i in x], tpot, 0.4, color=C_OURS, label=T("C=1 TPOT (ms,左轴)", "C=1 TPOT ms (left)"))
    ax.set_ylabel(T("C=1 TPOT (ms)", "C=1 TPOT (ms)"), color=C_OURS)
    ax.set_ylim(0, 44)
    ax.bar_label(b1, fmt="%.2f", fontsize=9.5, padding=2, color=C_OURS)
    ax.axhline(30, ls="--", lw=1.4, color=C_OK,
               label=T("验收线:C=1 TPOT ≤ 30 ms", "target: C=1 TPOT <= 30 ms"))

    ax2 = ax.twinx()
    b2 = ax2.bar([i + 0.2 for i in x], agg, 0.4, color=C_ACC, label=T("C=4 聚合 (tok/s,右轴)", "C=4 agg tok/s (right)"))
    ax2.set_ylabel(T("C=4 聚合吞吐 (tok/s)", "C=4 aggregate (tok/s)"), color=C_ACC)
    ax2.set_ylim(0, 130)
    ax2.grid(False)
    ax2.bar_label(b2, fmt="%.1f", fontsize=9.5, padding=2, color=C_ACC)
    ax2.axhline(50, ls="--", lw=1.4, color=C_OK,
                label=T("验收线:C=4 聚合 ≥ 50 t/s", "target: C=4 agg >= 50 t/s"))

    ax.set_xticks(list(x)); ax.set_xticklabels(stages, fontsize=9.5)
    ax.set_title(T("优化历程:① ② ③ 为历史口径(客户端协议不同,仅看趋势);④ 为本轮同协议实测",
                   "Optimization history: stages 1-3 use a legacy client protocol (trend only); stage 4 is current"),
                 fontsize=10.5, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, frameon=False, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax.set_ylim(0, 44)
    save(fig, "fig03_history.png")


# =============================================================================
# 图 4:C=1 的每 token 成本分解(严格相加)
# =============================================================================
def fig_cost_breakdown():
    gpu, cop, cpu = 17.32, 2.46, 17.55
    fig, ax = plt.subplots(figsize=(10.4, 2.4))
    ax.barh([0], [gpu], color="#4C8DFF", label=T("纯 GPU 模型(FAKE_ALL)", "pure GPU (FAKE_ALL)"))
    ax.barh([0], [cop], left=[gpu], color="#F4B942",
            label=T("D2H/H2D 拷贝 + host-func 派发", "copies + host-func dispatch"))
    ax.barh([0], [cpu], left=[gpu + cop], color="#E8590C",
            label=T("CPU MoE 计算(43 层 × 0.41 ms)", "CPU MoE compute (43 x 0.41 ms)"))
    ax.text(gpu / 2, 0, f"{gpu:.2f}", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    ax.text(gpu + cop / 2, 0, f"{cop:.2f}", ha="center", va="center", fontsize=9, color="#3B2E00")
    ax.text(gpu + cop + cpu / 2, 0, f"{cpu:.2f}", ha="center", va="center", fontsize=10, color="white",
            fontweight="bold")
    ax.text(gpu + cop + cpu + 0.4, 0, T("= 37.33 ms/token(实测)", "= 37.33 ms/token (measured)"),
            va="center", fontsize=10.5, fontweight="bold")
    ax.set_yticks([]); ax.set_xlim(0, 46)
    ax.set_xlabel(T("每 token 毫秒数(C=1,TP=2,无任何优化)", "ms per token (C=1, TP=2, no optimization)"))
    ax.set_title(T("诊断:三个零成本开关把端到端精确拆成三段(分毫不差)",
                   "Diagnosis: three zero-cost switches split the token cost exactly"),
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.34))
    save(fig, "fig04_cost_breakdown.png")


# =============================================================================
# 图 5:步代价拟合 TPOT(C) = F + C·V
# =============================================================================
def fig_step_cost():
    C = [1, 2, 4]
    ours = [28.13, 30.55, 33.60]
    ref = [23.11, 31.75, 41.88]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(C, ours, "o-", lw=2.4, ms=8, color=C_OURS, label=T("本项目", "ours"))
    ax.plot(C, ref, "s--", lw=2.4, ms=8, color=C_REF, label=T("lk_moe(专有)", "lk_moe (proprietary)"))
    # 拟合线
    F_o, V_o = 28.13 - (33.60 - 28.13) / 3, (33.60 - 28.13) / 3
    F_r, V_r = 23.11 - (41.88 - 23.11) / 3, (41.88 - 23.11) / 3
    xs = [1, 2, 3, 4]
    ax.plot(xs, [F_o + V_o * c for c in xs], ":", lw=1.6, color=C_OURS, alpha=0.8,
            label=T(f"拟合 F={F_o:.1f} ms, V={V_o:.2f} ms", f"fit F={F_o:.1f} ms, V={V_o:.2f} ms"))
    ax.plot(xs, [F_r + V_r * c for c in xs], ":", lw=1.6, color=C_REF, alpha=0.9,
            label=T(f"拟合 F={F_r:.1f} ms, V={V_r:.2f} ms", f"fit F={F_r:.1f} ms, V={V_r:.2f} ms"))
    ax.annotate(T("每 token 边际成本\n本项目更优", "lower marginal\ncost per token"),
                xy=(3.0, F_o + V_o * 3 - 1.6), xytext=(2.35, 44), fontsize=9.5, color=C_OK,
                arrowprops=dict(arrowstyle="->", color=C_OK, lw=1.3))
    ax.set_xticks([1, 2, 4])
    ax.set_xlabel(T("并发数 C", "concurrency C"))
    ax.set_ylabel("TPOT (ms)")
    ax.set_ylim(18, 50)
    ax.set_title(T("步代价拟合:TPOT(C) = F + C·V", "Step-cost fit: TPOT(C) = F + C*V"),
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    save(fig, "fig05_step_cost.png")


# =============================================================================
# 图 6:带宽 roofline —— 证明"不是带宽墙"
# =============================================================================
def fig_bandwidth():
    labels = [T("21.5 tok/s\n(单流下界)", "21.5 tok/s"),
              T("50 tok/s\n(验收线)", "50 tok/s\n(target)"),
              T("100 tok/s\n(目标)", "100 tok/s\n(goal)")]
    need = [70, 162, 325]
    fig, ax = plt.subplots(figsize=(9.6, 4.3))
    b = ax.bar(labels, need, 0.5, color=["#A5C8FF", "#4C8DFF", "#1F6FEB"],
               label=T("按 3.246 GB/token 计算所需带宽", "required BW @ 3.246 GB/token"))
    ax.bar_label(b, fmt="%d GB/s", fontsize=10, padding=3)
    ax.axhline(740, lw=2.2, color=C_OK, label=T("本机实测可达 ~740 GB/s", "measured machine BW ~740 GB/s"))
    ax.axhline(254, lw=2.0, ls="--", color=C_ACC,
               label=T("当前引擎实测 ~254 GB/s(34%)", "current engine ~254 GB/s (34%)"))
    ax.fill_between([-0.5, 2.5], 0, 740, color=C_OK, alpha=0.05)
    ax.set_ylim(0, 900)
    ax.set_ylabel(T("内存带宽 (GB/s)", "bandwidth (GB/s)"))
    ax.set_title(T("Roofline:带宽放得下 ⇒ 当前瓶颈是每层关键路径延迟,不是带宽",
                   "Roofline: bandwidth is not the wall; per-layer latency is"),
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    save(fig, "fig06_bandwidth.png")


# =============================================================================
# 图 7:每层 rest / compute 拆分
# =============================================================================
def fig_rest_compute():
    labels = [T("历史(43 层全 CPU)", "historical\n(43 CPU layers)"),
              T("本轮 async(均值)", "this round\n(async, mean)"),
              T("本轮 async(稳态最小)", "this round\n(async, min)")]
    compute = [0.41, 0.318, 0.318]
    rest = [0.60, 0.601, 0.316]
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    x = range(3)
    b1 = ax.bar(x, compute, 0.5, color=C_ACC, label=T("compute(CPU MoE+EP)", "compute (CPU MoE + EP)"))
    b2 = ax.bar(x, rest, 0.5, bottom=compute, color="#9DBEF5",
                label=T("rest(GPU+拷贝+握手)", "rest (GPU + copies + handshake)"))
    for i, (c, r) in enumerate(zip(compute, rest)):
        tot = c + r
        ax.text(i, tot + 0.02, f"period {tot:.2f} ms", ha="center", fontsize=9.5, fontweight="bold")
    ax.axhline(0.40, ls="--", lw=1.5, color=C_OK)
    ax.text(2.45, 0.415, T("rest 验收线 0.40 ms", "rest target 0.40 ms"), color=C_OK, fontsize=9, ha="right")
    ax.bar_label(b1, fmt="%.3f", fontsize=9, label_type="center", color="white")
    ax.bar_label(b2, fmt="%.3f", fontsize=9, label_type="center", color="#14315E")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel(T("每次 cpu_decode 调用 (ms)", "per cpu_decode call (ms)"))
    ax.set_ylim(0, 1.15)
    ax.set_title(T("每层开销拆分:compute 不退化(0.41→0.318),rest 稳态最小值已达标",
                   "Per-layer split: compute improved (0.41 -> 0.318), steady-state rest meets target"),
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    save(fig, "fig07_rest_compute.png")


def main():
    fig_main_result()
    fig_async_ab()
    fig_history()
    fig_cost_breakdown()
    fig_step_cost()
    fig_bandwidth()
    fig_rest_compute()
    print("CJK font:", CJK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

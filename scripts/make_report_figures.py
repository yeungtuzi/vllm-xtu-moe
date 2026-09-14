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
_FONT_URL = ("https://cdn.jsdelivr.net/gh/notofonts/noto-cjk@main/"
             "Sans/SubsetOTF/SC/NotoSansSC-%s.otf")
FONT_DIR = os.path.join(OUT, "fonts")
for _w in ("Regular", "Bold"):
    _f = os.path.join(FONT_DIR, "NotoSansSC-%s.otf" % _w)
    if not os.path.exists(_f):
        os.makedirs(FONT_DIR, exist_ok=True)
        try:
            import urllib.request
            with urllib.request.urlopen(_FONT_URL % _w, timeout=90) as r, open(_f, "wb") as fh:
                fh.write(r.read())
            print("downloaded CJK font ->", _f)
        except Exception as e:  # noqa: BLE001
            print("CJK %s font unavailable (%s)" % (_w, e), file=sys.stderr)
CJK = False
if os.path.exists(os.path.join(FONT_DIR, "NotoSansSC-Regular.otf")):
    for _w in ("Regular", "Bold"):
        _f = os.path.join(FONT_DIR, "NotoSansSC-%s.otf" % _w)
        if os.path.exists(_f):
            font_manager.fontManager.addfont(_f)
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
    ours_tpot = [26.17, 28.41, 31.27]
    ref_tpot = [23.11, 31.75, 41.88]
    ours_agg = [35.03, 64.27, 107.34]
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
        T("⑤ +自旋 2000→200\n+每 CCD 5 核", "5. +spin 2000->200\n+5 cores/CCD"),
    ]
    tpot = [37.33, 33.82, 31.78, 28.13, 26.17]
    agg = [34.29, 55.23, 60.17, 100.16, 107.34]

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
    ax.set_title(T("优化历程:① ② ③ 为历史口径(客户端协议不同,仅看趋势);④ ⑤ 为本轮同协议实测",
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
    ours = [26.17, 28.41, 31.27]
    ref = [23.11, 31.75, 41.88]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(C, ours, "o-", lw=2.4, ms=8, color=C_OURS, label=T("本项目", "ours"))
    ax.plot(C, ref, "s--", lw=2.4, ms=8, color=C_REF, label=T("lk_moe(专有)", "lk_moe (proprietary)"))
    # 拟合线
    F_o, V_o = 26.17 - (31.27 - 26.17) / 3, (31.27 - 26.17) / 3
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
    labels = [T("历史\n(43 层全 CPU)", "historical\n(43 CPU layers)"),
              T("异步握手\n(settle=2000,均值)", "async\n(settle=2000, mean)"),
              T("异步+自旋 200\n(均值)", "async + spin 200\n(mean)"),
              T("异步+自旋 200\n+5 核/CCD(稳态最小)", "async + spin 200\n+5 cores/CCD (min)")]
    compute = [0.41, 0.304, 0.250, 0.195]
    rest = [0.60, 0.601, 0.666, 0.339]
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    x = range(4)
    b1 = ax.bar(x, compute, 0.6, color=C_ACC, label=T("compute(CPU MoE+EP)", "compute (CPU MoE + EP)"))
    b2 = ax.bar(x, rest, 0.6, bottom=compute, color="#9DBEF5",
                label=T("rest(GPU+拷贝+握手)", "rest (GPU + copies + handshake)"))
    for i, (c, r) in enumerate(zip(compute, rest)):
        tot = c + r
        ax.text(i, tot + 0.02, f"period {tot:.2f} ms", ha="center", fontsize=9.5, fontweight="bold")
    ax.axhline(0.40, ls="--", lw=1.5, color=C_OK)
    ax.text(3.45, 0.415, T("rest 验收线 0.40 ms", "rest target 0.40 ms"), color=C_OK, fontsize=9, ha="right")
    ax.bar_label(b1, fmt="%.3f", fontsize=9, label_type="center", color="white")
    ax.bar_label(b2, fmt="%.3f", fontsize=9, label_type="center", color="#14315E")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel(T("每次 cpu_decode 调用 (ms)", "per cpu_decode call (ms)"))
    ax.set_ylim(0, 1.20)
    ax.set_title(T("每层开销拆分:compute 0.41→0.195,rest 稳态最小 0.339 达标",
                   "Per-layer split: compute improved (0.41 -> 0.318), steady-state rest meets target"),
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    save(fig, "fig07_rest_compute.png")


# =============================================================================
# 图 0:项目定位二维图(替代 ASCII 图,避免 Markdown 渲染错位)
# =============================================================================
def fig_positioning():
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.grid(False)

    # 象限底色
    ax.add_patch(plt.Rectangle((0.6, 0.6), 4.35, 2.6, fc="#F5F7FA", ec="#D5DBE3"))
    ax.add_patch(plt.Rectangle((5.05, 0.6), 4.35, 2.6, fc="#F5F7FA", ec="#D5DBE3"))
    ax.add_patch(plt.Rectangle((0.6, 3.3), 4.35, 2.6, fc="#EFF5FF", ec="#D5DBE3"))
    ax.add_patch(plt.Rectangle((5.05, 3.3), 4.35, 2.6, fc="#E7F0FF", ec=C_OURS, lw=2.2))

    def box(x, y, w, h, title, lines, fc, ec, tc="#12233B"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.10",
                                    fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top",
                fontsize=11.5, fontweight="bold", color=tc)
        ax.text(x + w / 2, y + h - 0.62, "\n".join(lines), ha="center", va="top",
                fontsize=9.2, color="#33465F", linespacing=1.5)

    box(0.85, 3.55, 3.85, 2.1,
        T("vLLM 主线", "vLLM mainline"),
        [T("CPU = “虚拟显存”(官方原话)", "CPU = \"virtual GPU memory\""),
         T("CPU MoE kernel 仅在纯 CPU 后端", "CPU MoE kernel only on CPU backend"),
         T("MoE 卸载 RFC #38256 仍 open", "MoE-offload RFC #38256 still open"),
         T("须 --enforce-eager、同步 H2D、单卡", "needs --enforce-eager, sync H2D, 1 GPU")],
        "#FFFFFF", "#C7D0DC")

    box(5.3, 3.55, 3.85, 2.1,
        T("ktransformers(+SGLang)", "ktransformers (+SGLang)"),
        [T("CPU-GPU 异构,CPU 真的算", "real CPU-GPU heterogeneous compute"),
         T("重心转向微调 / 消费级卡 / Windows", "focus shifted to SFT/consumer/Windows"),
         T("原始一体化框架已 archive", "original framework archived"),
         T("依赖 Intel AMX(本机 EPYC 无)", "requires Intel AMX (EPYC has none)")],
        "#FFFFFF", "#8FB4E8")

    box(5.3, 0.85, 3.85, 2.1,
        T("lk_moe / Lvllmds4-x", "lk_moe / Lvllmds4-x"),
        [T("性能标杆(本机唯一比我们快)", "performance leader (this machine)"),
         T("LICENSE = PROPRIETARY", "LICENSE = PROPRIETARY"),
         T("明文禁止逆向 / 衍生 / 再分发", "reverse-engineering forbidden"),
         T("fork-of-a-fork,无法合入主线", "fork-of-a-fork, cannot upstream")],
        "#FBF3F3", "#E0B4B4", "#7A2E2E")

    box(0.85, 0.85, 3.85, 2.1,
        T("(空白象限)", "(empty quadrant)"),
        [T("闭源 + CPU 不参与计算", "closed + CPU does not compute"),
         T("没有长期维护的项目:", "no maintained project:"),
         T("闭源引擎必须有性能卖点,", "a closed engine needs a perf story,"),
         T("而卖点恰恰来自 CPU 参与计算", "which comes from CPU compute")],
        "#F7F7F7", "#DDDDDD", "#8A8A8A")

    # 四角标签
    ax.text(5.0, 6.24, T("开源 / 可改造", "Open / modifiable"), ha="center", fontsize=13,
            fontweight="bold", color="#12233B")
    ax.text(5.0, 0.28, T("闭源 / 不可改", "Closed / not modifiable"), ha="center", fontsize=13,
            fontweight="bold", color="#12233B")
    ax.text(0.28, 3.25, T("CPU 不参与计算", "CPU does not compute"), ha="center", va="center",
            rotation=90, fontsize=13, fontweight="bold", color="#12233B")
    ax.text(9.76, 3.25, T("CPU 参与计算", "CPU computes"), ha="center", va="center",
            rotation=270, fontsize=13, fontweight="bold", color="#12233B")

    # 我们的定位星标
    ax.text(7.2, 3.44, T("★ vllm-xtu-moe(本项目)", "★ vllm-xtu-moe (this project)"),
            ha="center", va="bottom", fontsize=12, fontweight="bold", color=C_OURS)

    ax.set_title(T("项目定位:开源可改 × CPU 参与计算 —— 与 ktransformers 同象限,但面向不同的机器与目标",
                   "Positioning: open-source x CPU-compute — same quadrant as KT, different target hardware"),
                 fontsize=11.8, fontweight="bold", pad=14)
    save(fig, "fig00_positioning.png")


# =============================================================================
# 图 8:系统架构图(替代 ASCII 图)
# =============================================================================
def fig_architecture():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(11.2, 7.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.6)
    ax.axis("off")
    ax.grid(False)

    def panel(y, h, title, fc, ec, tc="#12233B"):
        ax.add_patch(FancyBboxPatch((0.35, y), 9.3, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.8))
        ax.text(0.62, y + h - 0.26, title, ha="left", va="top", fontsize=11.5,
                fontweight="bold", color=tc)

    # --- vLLM 编排层 ---
    panel(6.05, 2.35, T("vLLM fork —— 编排链(与参考实现逐字节同源,只差 1 个空白 hunk)",
                        "vLLM fork — orchestration (byte-identical to the reference)"),
          "#F2F7FF", "#9DBEF5")
    ax.text(0.65, 7.62, T("Scheduler · PagedAttention · CUDA Graph(FULL_DECODE_ONLY)· DSpark 投机 · Prefix Cache",
                          "Scheduler · PagedAttention · CUDA Graph · DSpark speculative · Prefix cache"),
            fontsize=9.3, color="#33465F")
    ax.text(0.65, 6.92, T("RoutedExperts.forward() —— 按层分派", "RoutedExperts.forward() — per-layer dispatch"),
            fontsize=10.2, fontweight="bold", color="#12233B")

    rows = [
        (6.52, "常驻层(0–11)", "resident layers", "vLLM 原生 GPU MoE · MARLIN MXFP4", "#DFF3E3", "#8CC79A"),
        (6.16, "GPU 预填充层", "gpu-prefill layer", "_gpu_prefill(GPU)", "#E8F0FE", "#A9C3E8"),
        (5.80, "预填充", "prefill", "_cpu_prefill(CPU 引擎)", "#FFF6E5", "#E8C078"),
    ]
    for y, zh, en, val, fc, ec in rows:
        ax.add_patch(FancyBboxPatch((0.62, y - 0.24), 3.05, 0.30,
                                    boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.1))
        ax.text(2.14, y - 0.09, T(zh, en), ha="center", va="center", fontsize=9.2, color="#12233B")
        ax.add_patch(FancyArrowPatch((3.72, y - 0.09), (4.30, y - 0.09),
                                     arrowstyle="-|>", mutation_scale=11, color="#7A8798", lw=1.2))
        ax.text(4.40, y - 0.09, val, ha="left", va="center", fontsize=9.2, color="#33465F")

    # 解码行(高亮)
    ax.add_patch(FancyBboxPatch((0.62, 5.20), 3.05, 0.30,
                                boxstyle="round,pad=0.02,rounding_size=0.06", fc="#DCE9FF", ec=C_OURS, lw=1.4))
    ax.text(2.14, 5.35, T("解码(主战场)", "decode (main path)"), ha="center", va="center",
            fontsize=9.4, fontweight="bold", color="#12233B")
    ax.add_patch(FancyArrowPatch((3.72, 5.35), (4.30, 5.35), arrowstyle="-|>",
                                 mutation_scale=11, color=C_OURS, lw=1.6))
    ax.text(4.40, 5.35, "_cpu_decode(CPU 引擎)", ha="left", va="center", fontsize=9.4,
            fontweight="bold", color=C_OURS)

    # --- 引擎层 ---
    ax.add_patch(FancyArrowPatch((7.6, 5.15), (7.6, 4.62), arrowstyle="-|>",
                                 mutation_scale=14, color=C_OURS, lw=2.0))
    ax.text(7.72, 4.88, T("7 个指针 + stream", "7 pointers + stream"), ha="left", va="center",
            fontsize=9.2, color=C_OURS)

    panel(0.55, 3.95, T("xiaotu_moe —— 自研 CPU 引擎(C++ / AVX-512,无 nvcc 依赖)",
                        "xiaotu_moe — in-house CPU engine (C++/AVX-512, no nvcc)"),
          "#EFF5FF", C_OURS)
    items = [
        T("MXFP4 权重常驻主机内存;按 NUMA 节点 mbind 切片(实测 128/128 rc=0)",
          "MXFP4 weights resident in host DRAM; per-NUMA-node mbind sharding (128/128 rc=0)"),
        T("进程内共享工作线程池,每 CCD 4 核(实测 4→8 核/CCD 无增益)",
          "process-wide shared worker pool, 4 cores/CCD (4→8 cores/CCD: no gain)"),
        T("跨 rank EP 归约走 /dev/shm(不走 NCCL:每层 4~6.5 ms → ~14 µs)",
          "cross-rank EP reduction over /dev/shm (not NCCL: 4-6.5 ms → ~14 us per layer)"),
        T("异步握手:常驻 worker + mapped flag + 流内存操作,取代 cudaLaunchHostFunc",
          "async handshake: resident worker + mapped flags, replacing cudaLaunchHostFunc"),
        T("FP4 解码:fp32 LUT + vpermps(每 32 权重 19 → 10 条指令)",
          "FP4 decode: fp32 LUT + vpermps (19 → 10 instructions per 32 weights)"),
    ]
    for i, it in enumerate(items):
        y = 3.85 - i * 0.62
        ax.text(0.78, y, "•", ha="left", va="center", fontsize=12, color=C_OURS)
        ax.text(1.02, y, it, ha="left", va="center", fontsize=9.4, color="#22374F")

    # 侧栏:两个 rank
    ax.add_patch(FancyBboxPatch((7.35, 0.72), 2.05, 3.60,
                                boxstyle="round,pad=0.06,rounding_size=0.10", fc="#FFFFFF", ec="#C7D0DC", lw=1.2))
    ax.text(8.37, 4.14, T("TP=2 两 rank", "TP=2, two ranks"), ha="center", va="top",
            fontsize=10, fontweight="bold", color="#12233B")
    for i, (t, sub) in enumerate([
        (T("rank 0", "rank 0"), T("socket 0 · 12 CCD · 48 线程", "socket 0 · 12 CCD · 48 threads")),
        (T("rank 1", "rank 1"), T("socket 1 · 12 CCD · 48 线程", "socket 1 · 12 CCD · 48 threads")),
    ]):
        y = 3.55 - i * 1.05
        ax.add_patch(FancyBboxPatch((7.55, y - 0.62), 1.65, 0.80,
                                    boxstyle="round,pad=0.04,rounding_size=0.08", fc="#EFF5FF", ec="#9DBEF5", lw=1.1))
        ax.text(8.37, y - 0.12, t, ha="center", va="center", fontsize=9.8, fontweight="bold", color=C_OURS)
        ax.text(8.37, y - 0.42, sub, ha="center", va="center", fontsize=8.2, color="#33465F")
    ax.text(8.37, 1.18, T("只读本地内存\n跨 socket 仅做归约", "local reads only\ncross-socket: reduction only"),
            ha="center", va="center", fontsize=8.4, color="#33465F")

    ax.set_title(T("系统架构:把 MoE 层劈开 —— 注意力/KV 留 GPU,专家计算放 CPU",
                   "Architecture: split the MoE layer — attention/KV on GPU, experts on CPU"),
                 fontsize=12.5, fontweight="bold", pad=10)
    save(fig, "fig08_architecture.png")


# =============================================================================
# 图 9:AI 的局部搜索 vs 人类换搜索空间
# =============================================================================
def fig_search_space():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis("off"); ax.grid(False)

    def blob(cx, cy, w, h, title, sub, fc, ec, tc):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.08,rounding_size=0.14",
                                    fc=fc, ec=ec, lw=1.8, linestyle="--"))
        ax.text(cx, cy + h / 2 - 0.28, title, ha="center", va="top", fontsize=11,
                fontweight="bold", color=tc)
        ax.text(cx, cy + h / 2 - 0.62, sub, ha="center", va="top", fontsize=8.8,
                color="#33465F", linespacing=1.5)

    # 左:AI 的搜索空间(小,密)
    blob(2.6, 2.5, 4.2, 3.2, T("AI 能测、能改的邻域", "what AI can measure and change"),
         "", "#F4F6F9", "#B9C2CE", "#12233B")
    import numpy as np
    rng = np.random.default_rng(3)
    for i in range(34):
        ax.plot(2.6 + rng.normal(0, 0.72), 2.5 + rng.normal(0, 0.62), "o",
                ms=5.0, color="#A9B4C2", alpha=0.85)
    ax.text(2.6, 1.02, T("约 30 轮内核微优化\n(seqlock / padding / K-major / 列分块 / 线程数 …)",
                         "~30 rounds of kernel micro-tuning"),
            ha="center", va="center", fontsize=9.2, color="#7A2E2E", linespacing=1.5)
    ax.text(2.6, 4.42, T("局部都“正确”,但整个搜索空间是错的", "every step locally right, the space is wrong"),
            ha="center", va="center", fontsize=9.6, color="#7A2E2E", style="italic")

    # 右:人类换掉整个空间
    blob(7.7, 2.5, 4.2, 3.2, T("人类一句话换掉的搜索空间", "the space a human replaces in one sentence"),
         "", "#EFF5FF", C_OURS, C_OURS)
    cases = [
        T("① 划界:“第一条视为已完成”", "1. draw the boundary"),
        T("② 预期形状:“C=2 怎么不涨?”", "2. expected shape"),
        T("③ 参考数字:生产接受率 40–50%", "3. reference numbers"),
        T("④ 架构放置:“投机模型跑在 GPU 上吧”", "4. architectural placement"),
    ]
    for i, c in enumerate(cases):
        ax.text(7.7, 3.86 - i * 0.50, c, ha="center", va="center", fontsize=9.4, color="#12233B")

    # 箭头:人的一句话把 AI 从局部里"拎"出来
    ax.add_patch(FancyArrowPatch((5.05, 2.5), (5.9, 2.5), arrowstyle="-|>",
                                 mutation_scale=26, color=C_ACC, lw=3.2))
    ax.text(5.48, 2.86, T("一句话", "one sentence"), ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color=C_ACC)
    ax.text(5.48, 2.08, T("收益\n数轮~数十轮", "payoff\ndozens of rounds"), ha="center", va="top",
            fontsize=9, color=C_ACC, linespacing=1.4)

    ax.set_title(T("人类工程师的关键作用:AI 在邻域内穷举,人换掉整个搜索空间",
                   "Why the human matters: AI exhausts a neighbourhood, the human replaces the space"),
                 fontsize=12.5, fontweight="bold", pad=12)
    save(fig, "fig09_search_space.png")


def main():
    fig_positioning()
    fig_architecture()
    fig_search_space()
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

#!/usr/bin/env python
"""把 `vllm bench serve` 的实测数据画成报告用图(多面板)。

口径(用户 2026-09-17 裁定):
  * 产品摘要用 **ShareGPT**(真实对话;投机解码与前缀缓存才有意义);
  * 简报数字 = **output tok/s**(最好值),**TTFT 单列**,不摊进吞吐;
  * 数字必须与**硬件配置**同时出现,否则无意义。

数据源:
  A) ShareGPT:report/tuning/raw/sg*_c*.json  (scripts/bench_sharegpt.sh 产出)
  B) 长度×并发扫描(裸预填充):report/tuning/logs/bench_serve_acc2/L*_C*.json
输出:docs/figures/v0210_bench_serve.png(200 dpi)
用法: python report/tuning/probes/plot_bench_serve.py
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# probes → tuning → report → <repo 根>(少一层就会指到 report/,glob 全空 —— 实测踩过)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FIG = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIG, exist_ok=True)

# 中文字体(项目里已有就复用,没有就退英文标签)
_CJK = None
for cand in glob.glob(os.path.join(FIG, "fonts", "*.otf")) + \
        glob.glob("/usr/share/fonts/**/NotoSansCJK*.ttc", recursive=True):
    try:
        font_manager.fontManager.addfont(cand)
        _CJK = font_manager.FontProperties(fname=cand).get_name()
        break
    except Exception:  # noqa: BLE001
        pass
plt.rcParams["font.sans-serif"] = ([_CJK] if _CJK else []) + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
ZH = _CJK is not None


def T(zh, en):
    return zh if ZH else en


def load(d, pat):
    out = {}
    for p in glob.glob(os.path.join(ROOT, d, pat)):
        m = re.search(r"_c(\d+)\.json$", p) or re.search(r"_C(\d+)\.json$", p)
        if not m:
            continue
        try:
            out[int(m.group(1))] = json.load(open(p))
        except Exception:  # noqa: BLE001
            pass
    return out


def val(j, *ks):
    for k in ks:
        if j.get(k) is not None:
            return j[k]
    return None


def sg(tag):
    """ShareGPT:按 <tag>_c<C>.json 聚合"""
    out = {}
    for p in glob.glob(os.path.join(ROOT, "report/tuning/raw", f"{tag}_c*.json")):
        m = re.search(r"_c(\d+)\.json$", p)
        try:
            out[int(m.group(1))] = json.load(open(p))
        except Exception:  # noqa: BLE001
            pass
    return out


def main() -> int:
    spec1 = sg("sg21s1")
    spec0 = sg("sg21s0")
    sweep = {}
    for p in glob.glob(os.path.join(ROOT, "report/tuning/logs/bench_serve_acc2/L*.json")):
        m = re.match(r"L(\d+)_C(\d+)", os.path.basename(p))
        if m:
            sweep[(int(m.group(1)), int(m.group(2)))] = json.load(open(p))
    if not (spec1 or sweep):
        print("没有可画的数据"); return 1

    fig, ax = plt.subplots(2, 2, figsize=(13.2, 8.6))
    fig.suptitle(T("vllm-xtu-moe v0.2 性能实测  (2×A100-40GB TP=2 / 2×EPYC 9654 / 1.5 TiB DDR5 ~740 GB/s)",
                   "vllm-xtu-moe v0.2 measured performance  (2xA100-40GB TP2 / 2xEPYC 9654 / 1.5 TiB DDR5 ~740 GB/s)"),
                 fontsize=12, y=0.985)

    # (1) ShareGPT:output tok/s vs 并发(Plain vs dspark)
    a = ax[0][0]
    for tag, js, style, lab in (("plain", spec0, "o-", T("Plain decode", "Plain decode")),
                                ("dspark", spec1, "s-", "dspark")):
        if not js:
            continue
        cs = sorted(js)
        a.plot(cs, [val(js[c], "output_throughput") for c in cs], style, label=lab, lw=2)
    a.set_xlabel(T("并发 (max-concurrency)", "concurrency")); a.set_ylabel("output tok/s")
    a.set_title(T("ShareGPT:输出吞吐(简报数字,TTFT 单列)", "ShareGPT: output throughput"))
    a.grid(alpha=.3); a.legend()

    # (2) ShareGPT:TTFT 单列
    b = ax[0][1]
    for tag, js, style, lab in (("plain", spec0, "o-", "Plain"), ("dspark", spec1, "s-", "dspark")):
        if not js:
            continue
        cs = sorted(js)
        b.plot(cs, [val(js[c], "mean_ttft_ms") for c in cs], style, label=lab, lw=2)
    b.set_xlabel(T("并发", "concurrency")); b.set_ylabel("TTFT (ms)")
    b.set_title(T("ShareGPT:首 token 时延(TTFT)", "ShareGPT: TTFT")); b.grid(alpha=.3); b.legend()

    # (3) 长度×并发:output tok/s(裸预填充扫描)
    c = ax[1][0]
    for L in sorted({k[0] for k in sweep}):
        cs = sorted(k[1] for k in sweep if k[0] == L)
        c.plot(cs, [val(sweep[(L, x)], "output_throughput") for x in cs], "o-", label=f"{L}")
    c.set_xlabel(T("并发", "concurrency")); c.set_ylabel("output tok/s")
    c.set_title(T("长度×并发扫描:输出吞吐(裸预填充口径)", "length x concurrency: output throughput"))
    c.grid(alpha=.3); c.legend(title=T("prompt 长度", "prompt len"), fontsize=8)

    # (4) 长度×并发:TTFT(log)
    d = ax[1][1]
    for L in sorted({k[0] for k in sweep}):
        cs = sorted(k[1] for k in sweep if k[0] == L)
        d.semilogy(cs, [val(sweep[(L, x)], "mean_ttft_ms") for x in cs], "o-", label=f"{L}")
    d.set_xlabel(T("并发", "concurrency")); d.set_ylabel("TTFT (ms, log)")
    d.set_title(T("长度×并发扫描:TTFT", "length x concurrency: TTFT"))
    d.grid(alpha=.3, which="both"); d.legend(title=T("prompt 长度", "prompt len"), fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIG, "v0210_bench_serve.png")
    fig.savefig(out, dpi=200)
    print("已生成", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

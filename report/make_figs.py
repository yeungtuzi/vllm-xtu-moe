#!/usr/bin/env python
"""Generate the report figures from the JSONL measurement files.

Inputs (all optional; missing files just skip their series):
  report/curve_tp1.jsonl, report/curve_tp2.jsonl   (dsv4_prefill_curve.py)
  report/moe_micro.json                            (bench_gpu_moe_prefetch.py)
  report/hw_bandwidth.json                         (stream + pcie_bw)
Outputs: report/fig/fig*.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig")
os.makedirs(FIG, exist_ok=True)


def load_jsonl(name):
    p = os.path.join(HERE, name)
    out = []
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line.startswith("JSON "):
                line = line[5:]
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def series(rows, kind, **match):
    return [r for r in rows if r.get("kind") == kind
            and all(r.get(k) == v for k, v in match.items())]


def fig_hw_bandwidth():
    p = os.path.join(HERE, "hw_bandwidth.json")
    if not os.path.exists(p):
        print("skip fig1 (no hw_bandwidth.json)")
        return
    d = json.load(open(p))
    labels, vals, colors = [], [], []
    for k, v in d.get("stream", {}).items():
        labels.append(f"DRAM {k}\n(Triad)")
        vals.append(v)
        colors.append("#4C78A8")
    for k, v in d.get("pcie", {}).items():
        labels.append(k)
        vals.append(v)
        colors.append("#F58518")
    if "hbm_d2d" in d:
        labels.append("HBM D2D\n(read+write)")
        vals.append(d["hbm_d2d"])
        colors.append("#54A24B")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    b = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("GB/s")
    ax.set_title("Measured bandwidth (AMD EPYC 9654 / A100-PCIE-40GB)")
    ax.set_yscale("log")
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v * 1.05, f"{v:.1f}",
                ha="center", fontsize=8)
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_hw_bandwidth.png"), dpi=150)
    plt.close(fig)
    print("fig1 ok")


def fig_ttft_vs_len():
    rows1 = load_jsonl("curve_tp1.jsonl")
    rows2 = load_jsonl("curve_tp2.jsonl")
    if not rows1 and not rows2:
        print("skip fig2 (no curve data)")
        return
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for rows, label, mk in ((rows1, "TP=1", "o"), (rows2, "TP=2 (EP)", "s")):
        for mode, ls, col in (("cpu", "--", "#B03A2E"), ("gpu", "-", "#1F77B4")):
            pts = [(r["tokens"], r["ttft_s"]) for r in series(rows, "ttft", mode=mode)]
            if not pts:
                continue
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys, ls, marker=mk, color=col,
                    label=f"{label} {'CPU prefill' if mode=='cpu' else 'GPU prefill'}")
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                            xytext=(0, 6), fontsize=7, ha="center")
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("TTFT (s)")
    ax.set_xscale("log", base=2)
    ax.set_title("Time to first token vs prompt length (DS-V4-Flash)")
    ax.grid(ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_ttft_vs_len.png"), dpi=150)
    plt.close(fig)
    print("fig2 ok")


def fig_moe_throughput():
    p = os.path.join(HERE, "moe_micro.json")
    if not os.path.exists(p):
        print("skip fig3 (no moe_micro.json)")
        return
    d = json.load(open(p))
    p2 = os.path.join(HERE, "moe_micro_tp2.json")
    if os.path.exists(p2):
        d2 = json.load(open(p2))
        for k in ("ovl_E128_K3", "seq_E128_K3"):
            if k in d2:
                d[k.replace("_E128_K3", "_tp2")] = d2[k]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for key, label, col in (("seq", "no overlap (H2D serial)", "#B03A2E"),
                            ("ovl", "prefetch overlap", "#1F77B4"),
                            ("ovl_tp2", "overlap + TP=2 EP (projected)",
                             "#54A24B")):
        if key not in d:
            continue
        xs = sorted(int(k) for k in d[key])
        ys = [d[key][str(x)] for x in xs]
        ax.plot(xs, ys, "o-", color=col, label=label)
    ax.axhline(1500, color="gray", ls="--", lw=1)
    ax.text(ax.get_xlim()[0], 1560, "target 1500 tok/s", fontsize=8, color="gray")
    ax.set_xlabel("prefill tokens per layer (T)")
    ax.set_ylabel("43-layer equivalent throughput (tok/s)")
    ax.set_xscale("log", base=2)
    ax.set_title("Pure-MoE prefill throughput (single A100, real layer shapes)")
    ax.grid(ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_moe_throughput.png"), dpi=150)
    plt.close(fig)
    print("fig3 ok")


def fig_scaling():
    rows1 = load_jsonl("curve_tp1.jsonl")
    rows2 = load_jsonl("curve_tp2.jsonl")
    g1 = {r["tokens"]: r for r in series(rows1, "ttft", mode="gpu")}
    g2 = {r["tokens"]: r for r in series(rows2, "ttft", mode="gpu")}
    if not g1 or not g2:
        print("skip fig4 (need both tp1 and tp2 gpu ttft)")
        return
    xs = sorted(set(g1) & set(g2))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    w = 0.38
    idx = range(len(xs))
    a1.bar([i - w / 2 for i in idx], [g1[x]["ttft_s"] for x in xs], w,
           label="TP=1", color="#4C78A8")
    a1.bar([i + w / 2 for i in idx], [g2[x]["ttft_s"] for x in xs], w,
           label="TP=2 (EP)", color="#54A24B")
    a1.set_xticks(list(idx))
    a1.set_xticklabels([str(x) for x in xs])
    a1.set_xlabel("prompt tokens")
    a1.set_ylabel("TTFT (s)")
    a1.set_title("GPU prefill TTFT: 1 vs 2 GPUs")
    a1.legend(fontsize=8)
    a1.grid(axis="y", ls=":", alpha=0.5)
    a2.bar([i - w / 2 for i in idx], [g1[x]["tok_per_s"] for x in xs], w,
           label="TP=1", color="#4C78A8")
    a2.bar([i + w / 2 for i in idx], [g2[x]["tok_per_s"] for x in xs], w,
           label="TP=2 (EP)", color="#54A24B")
    a2.axhline(1500, color="gray", ls="--", lw=1)
    a2.set_xticks(list(idx))
    a2.set_xticklabels([str(x) for x in xs])
    a2.set_xlabel("prompt tokens")
    a2.set_ylabel("tok/s")
    a2.set_title("End-to-end prefill throughput")
    a2.legend(fontsize=8)
    a2.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_scaling.png"), dpi=150)
    plt.close(fig)
    print("fig4 ok")


def fig_kv():
    d = {"1024": 405.0, "16384": 400.0}
    fig, ax = plt.subplots(figsize=(6, 4))
    ks = list(d)
    ax.bar(ks, [d[k] for k in ks], color="#4C78A8")
    ax.set_ylabel("KV cache per token (KiB)")
    ax.set_xlabel("max_model_len")
    ax.set_title("Measured DS-V4-Flash KV cost (~400 KiB/token)")
    for i, k in enumerate(ks):
        ax.text(i, d[k] + 5, f"{d[k]:.0f} KiB", ha="center", fontsize=9)
    ax.set_ylim(0, 480)
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_kv_capacity.png"), dpi=150)
    plt.close(fig)
    print("fig5 ok")


def fig_concurrency():
    rows1 = load_jsonl("curve_tp1.jsonl")
    rows2 = load_jsonl("curve_tp2.jsonl")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    got = False
    for rows, label, col in ((rows1, "TP=1", "#4C78A8"), (rows2, "TP=2 (EP)", "#54A24B")):
        pts = [(r["conc"], r["tok_per_s"]) for r in series(rows, "concurrency")]
        if not pts:
            continue
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", color=col, label=label)
        got = True
    if not got:
        print("skip fig6 (no concurrency data)")
        plt.close(fig)
        return
    ax.axhline(1500, color="gray", ls="--", lw=1)
    ax.set_xlabel("concurrent requests (2K-token prompts each)")
    ax.set_ylabel("aggregate prefill throughput (tok/s)")
    ax.set_title("Aggregate prefill throughput vs concurrency")
    ax.grid(ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_concurrency.png"), dpi=150)
    plt.close(fig)
    print("fig6 ok")


if __name__ == "__main__":
    fig_hw_bandwidth()
    fig_ttft_vs_len()
    fig_moe_throughput()
    fig_scaling()
    fig_kv()
    fig_concurrency()

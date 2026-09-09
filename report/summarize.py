#!/usr/bin/env python
"""Turn the measurement JSONL files into markdown tables for the report."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    p = os.path.join(HERE, name)
    rows = []
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line.startswith("JSON "):
                line = line[5:]
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def tbl(rows, kind, **m):
    sel = [r for r in rows if r.get("kind") == kind
           and all(r.get(k) == v for k, v in m.items())]
    return sorted(sel, key=lambda r: r.get("tokens", 0))


def main():
    for name, label in (("curve_tp1.jsonl", "TP=1"),
                        ("curve_tp2.jsonl", "TP=2 (EP)")):
        rows = load(name)
        if not rows:
            print(f"\n### {label}: (no data in {name})")
            continue
        print(f"\n### {label}")
        for r in rows:
            if r.get("kind") == "load":
                print(f"- load: {r.get('load_s')} s "
                      f"(maxlen={r.get('maxlen')}, mnbt={r.get('mnbt')})")
            elif r.get("kind") == "warmup":
                print(f"- warmup (pinned cache + JIT): {r.get('wall_s')} s")
        print("\n| prompt tokens | CPU TTFT s | CPU tok/s | GPU TTFT s | GPU tok/s |")
        print("|---|---:|---:|---:|---:|")
        cpu = {r["tokens"]: r for r in tbl(rows, "ttft", mode="cpu")}
        gpu = {r["tokens"]: r for r in tbl(rows, "ttft", mode="gpu")}
        for t in sorted(set(cpu) | set(gpu)):
            c = cpu.get(t, {})
            g = gpu.get(t, {})
            print(f"| {t} | {c.get('ttft_s','-')} | {c.get('tok_per_s','-')} | "
                  f"{g.get('ttft_s','-')} | {g.get('tok_per_s','-')} |")
        print("\n| concurrency | total tokens | wall s | aggregate tok/s |")
        print("|---|---:|---:|---:|")
        for r in tbl(rows, "concurrency"):
            print(f"| {r['conc']} | {r['total_tokens']} | {r['wall_s']} | "
                  f"{r['tok_per_s']} |")
        for r in rows:
            if r.get("kind") == "decode":
                print(f"\ndecode: {r['out_tokens']} tok in {r['wall_s']} s = "
                      f"{r['decode_tok_per_s']} tok/s")
        print("\nfirst-token texts:")
        for r in tbl(rows, "ttft"):
            print(f"  L={r['target_len']} {r['mode']}: {r.get('text')!r}")


if __name__ == "__main__":
    sys.exit(main())

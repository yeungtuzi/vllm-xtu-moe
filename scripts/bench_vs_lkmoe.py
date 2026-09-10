#!/usr/bin/env python
"""Comprehensive compute-performance comparison: xiaotu-moe vs lk-moe.

Benchmarks the EXACT interfaces vLLM's RoutedExperts calls (see
Lvllmds4-x/vllm/model_executor/layers/fused_moe/routed_experts.py), so nothing
is inferred from the binaries:

    engine.cpu_prefill(qlen, top_k, eids_ptr, wts_ptr, x_ptr, out_ptr)
    engine.cpu_decode(stream_ptr, qlen, top_k, x_ptr, eids_ptr, wts_ptr, out_ptr)

Both lk_moe.MOE_MXFP4 and xiaotu_moe.MOE_MXFP4 are built from the SAME synthetic
packed weights and fed the same inputs, then timed. No full model is run — the
kernels are exercised at real DeepSeek-V4-Flash dimensions (synthetic bytes).

Sweeps:
  * model dims (--dims): deepseek_flash | small | large | custom E,H,I,K
  * batch sizes for decode (1..64) and prefill (128..2048)
  * routing concentration (--concentrate N): N<e close to grouped/NUMA path,
    N=e -> diverse per-token path

Output: per-(engine, batch) ms/layer + us/token + MoE tok/s, and the
xiaotu/lk_moe ratio (<1 -> xiaotu faster). A CSV is also written.

NOTE: this compares CPU kernels, so it competes with the main serving process
for CPU. Run it when the main service is quiesced for stable absolute numbers;
ratios are meaningful under any constant load.

Usage:
  python bench_vs_lkmoe.py                       # deepseek dims, both engines
  python bench_vs_lkmoe.py --dims small --mode prefill --concentrate 16
  python bench_vs_lkmoe.py --only lk             # xiaotu-only / lk-only
"""
import argparse
import csv
import os
import sys
import time

# lk_moe allocates GPU weight mirrors for its gpu_prefill path, so it needs a
# valid, free CUDA device even for a CPU-compute comparison. Set BENCH_GPU
# (e.g. "2") to steer to an idle GPU; CUDA_VISIBLE_DEVICES must be in place
# before torch initializes CUDA, so handle it here at the very top.
_BENCH_GPU = os.environ.get("BENCH_GPU")
if _BENCH_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = _BENCH_GPU

import numpy as np
import torch

# REPO = 本仓库根(引擎随仓库内置);用 PYTHONPATH 指向它以便 import xiaotu_moe
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIM_PRESETS = {
    # name: (expert_num E, hidden H, intermed I, top_k K, groupK, model label)
    "deepseek_flash": (256, 4096, 2048, 6, 32, "DeepSeek-V4-Flash"),
    "small":          (128, 2048, 1024, 6, 32, "small"),
    "large":          (256, 5120, 2560, 6, 32, "large"),
}
DECODE_BATCHES = (1, 2, 4, 8, 16, 32, 64)
PREFILL_BATCHES = (128, 256, 512, 1024, 2048)


def build_dims(name, custom):
    if custom:
        E, H, I, K = [int(x) for x in custom.split(",")]
        return E, H, I, K, 32, f"custom E{E} H{H} I{I} K{K}"
    if name in DIM_PRESETS:
        return DIM_PRESETS[name]
    raise SystemExit(f"unknown dims {name!r}; choose {list(DIM_PRESETS)} or custom")


def syspath_and_import():
    sys.path.insert(0, REPO)
    import lk_moe
    import xiaotu_moe
    vm = xiaotu_moe.load()
    return lk_moe, vm


def make_cfg(mod, E, H, I, K, GK, max_bs):
    c = mod.MOEConfigV2()
    c.num_processes = 1; c.process_id = 0; c.gpu_id = 0
    c.has_gate_proj = True; c.expert_num = E; c.top_k = K
    c.hidden_size = H; c.intermediate_size = I
    c.max_batch_size = max(256, max_bs); c.max_num_seqs = 256
    c.stride = 32; c.group_min_len = 10; c.group_max_len = 4096 + 128
    c.groupN = 1; c.groupK = GK; c.activation_type = 0
    return c


def build_engines(E, H, I, K, GK, max_bs, only):
    rng = np.random.default_rng(0)
    def t(a): return torch.from_numpy(np.ascontiguousarray(a))
    w13 = t(rng.integers(0, 256, (E, 2 * I, H // 2), dtype=np.uint8))
    w2 = t(rng.integers(0, 256, (E, H, I // 2), dtype=np.uint8))
    s13 = t(rng.integers(100, 145, (E, 2 * I, H // GK), dtype=np.uint8))
    s2 = t(rng.integers(100, 145, (E, H, I // GK), dtype=np.uint8))
    size_mb = (E * (2 * I * H // 2 + H * I // 2)) / 1e6 * 2  # x2 (both engines)
    print(f"[dims] E={E} H={H} I={I} K={K} GK={GK}  "
          f"both-engine footprint ~{size_mb:.0f} MB")

    lk_moe, vm = syspath_and_import()
    engines = {}
    if only["lk"]:
        engines["lk_moe"] = lk_moe.MOE_MXFP4(
            make_cfg(lk_moe, E, H, I, K, GK, max_bs),
            w13.data_ptr(), w2.data_ptr(), s13.data_ptr(), s2.data_ptr(), 0, 0)
    if only["xiaotu"]:
        engines["xiaotu"] = vm.MOE_MXFP4(
            make_cfg(vm, E, H, I, K, GK, max_bs),
            w13.data_ptr(), w2.data_ptr(), s13.data_ptr(), s2.data_ptr(), 0, 0)
    return engines


def bench_call(engine, method, B, K, Hn, ids, wts, xu, out, iters, warmup=1):
    fn = getattr(engine, method)
    if method == "cpu_decode":
        call = lambda: fn(0, B, K, xu.data_ptr(), ids.data_ptr(),
                          wts.data_ptr(), out.data_ptr())
    else:
        call = lambda: fn(B, K, ids.data_ptr(), wts.data_ptr(),
                          xu.data_ptr(), out.data_ptr())
    for _ in range(warmup):
        call()
    # min-of-iters is robust to scheduler noise; also keep median.
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        call()
        times.append(time.perf_counter() - t0)
    return min(times), float(np.median(times))


def run(args, only):
    E, H, I, K, GK, label = build_dims(args.dims, args.custom)
    if args.custom is None and args.dims == "deepseek_flash":
        pass
    engines = build_engines(E, H, I, K, GK, args.prefill_max, only)

    conc = args.concentrate if args.concentrate else E
    conc = min(conc, E)
    rng = np.random.default_rng(7)

    # cpu_prefill is the synchronous, apples-to-apples interface for BOTH
    # engines (cpu_decode is async/overlapped for lk_moe, so its host-side
    # wall-clock is not comparable). Optional --small-prefill extends the
    # prefill sweep down to decode-size batches so every size is compared on
    # the synchronous kernel.
    prefill_batches = [1, 2, 4, 8, 16, 32, 64] + list(PREFILL_BATCHES) \
        if args.small_prefill else list(PREFILL_BATCHES)

    all_rows = []

    def headline(txt):
        print("\n" + "=" * 78)
        print(txt)
        print("=" * 78)

    for method, batches in (("cpu_prefill", prefill_batches),
                            ("cpu_decode", DECODE_BATCHES)):
        if args.mode == "prefill" and method != "cpu_prefill":
            continue
        if args.mode == "decode" and method != "cpu_decode":
            continue
        headline(f"interface {method}  (model {label}, routing over {conc}/{E} experts)")
        hdr = (f"{'batch':>6} | {'ms/layer':>26} | {'us/token':>24} "
               f"| {'MoE tok/s':>22} | ratio")
        print(hdr)
        print("-" * len(hdr))
        for B in batches:
            xu = torch.randint(0, 65536, (B, H), dtype=torch.uint16).contiguous()
            ids = torch.randint(0, conc, (B, K), dtype=torch.int32).contiguous()
            wts = torch.rand(B, K, dtype=torch.float32).contiguous()
            iters = max(3, min(20, 200 // B + 3)) if method == "cpu_decode" else \
                    max(3, min(10, 256 // (B // 128 + 1) + 2))
            res = {}
            line = f"{B:>6} | "
            for eng_name, engine in engines.items():
                out = torch.zeros(B, H, dtype=torch.float32)
                mn, med = bench_call(engine, method, B, K, H, ids, wts,
                                     xu, out, iters)
                us_tok = mn / B * 1e6
                toks = B / mn
                res[eng_name] = (mn, med)
                line += (f"{mn*1e3:>8.2f}/{med*1e3:>7.2f}ms "
                         f"({us_tok:>7.1f}us) ({toks:>9.0f}/s)    ")
            row = {"method": method, "batch": B, "routing": conc}
            if "lk_moe" in res and "xiaotu" in res:
                ratio = res["xiaotu"][0] / res["lk_moe"][0]
                line += f"  {ratio:>5.3f}  xiaotu/lk"
                row.update({"lk_ms": res["lk_moe"][0] * 1e3,
                            "xiaotu_ms": res["xiaotu"][0] * 1e3,
                            "ratio": ratio})
            elif "xiaotu" in res:
                row["xiaotu_ms"] = res["xiaotu"][0] * 1e3
            else:
                row["lk_ms"] = res["lk_moe"][0] * 1e3
            print(line)
            all_rows.append(row)

    if args.csv:
        path = args.csv
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["method", "batch", "routing",
                                              "lk_ms", "xiaotu_ms", "ratio"])
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"\nCSV written to {path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dims", default="deepseek_flash",
                    choices=list(DIM_PRESETS), help="model dims preset")
    ap.add_argument("--custom", default=None,
                    help="custom dims as E,H,I,K")
    ap.add_argument("--mode", default="all", choices=["prefill", "decode", "all"])
    ap.add_argument("--concentrate", type=int, default=None,
                    help="route tokens over N experts (grouped path); "
                         "default=all E (diverse path)")
    ap.add_argument("--only", default=None, choices=["lk", "xiaotu"],
                    help="run only one engine (default both)")
    ap.add_argument("--prefill-max", type=int, default=2048,
                    help="max prefill batch (default 2048)")
    ap.add_argument("--small-prefill", action="store_true",
                    help="also run cpu_prefill at decode-size batches (1..64), "
                         "the fair synchronous comparison for every batch size")
    ap.add_argument("--csv", default=None, help="optional CSV output path")
    args = ap.parse_args()
    only = {"lk": True, "xiaotu": False} if args.only == "lk" else \
           {"lk": False, "xiaotu": True} if args.only == "xiaotu" else \
           {"lk": True, "xiaotu": True}
    return run(args, only)


if __name__ == "__main__":
    sys.exit(main())

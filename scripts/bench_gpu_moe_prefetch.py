#!/usr/bin/env python
"""Microbench: gpu_moe_layer with vs without H2D/compute overlap (prefetch).

Synthetic weights in the exact fused layout, full DS-V4 dims (E=256, H=4096,
I=2048, topk=6). Two modes:
  seq  - the pre-overlap path: CPU tensors passed straight in (H2D + GPU byte
         transpose + kernels, all serialised on the compute stream).
  ovl  - prefetch_layer() on a side stream into a 2-slot ring, kernels consume
         the slot (H2D of "layer L+1" overlaps kernels of "layer L").
Reports per-layer ms and the 43-layer-equivalent prefill throughput (tok/s).
"""
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_xiaotu_moe.gpu_prefill import (  # noqa: E402
    _pinned_kmajor,
    gpu_moe_layer,
    prefetch_layer,
)

H, I, E, K = 4096, 2048, 256, 6
LAYERS = 43
dev = torch.cuda.current_device()
gc = torch.Generator(device="cpu").manual_seed(0)
gg = torch.Generator(device=dev).manual_seed(0)


def make_weights(Ee=E):
    w13 = torch.randint(0, 16, (Ee, 2 * I, H // 2), generator=gc, dtype=torch.uint8)
    s13 = torch.randint(118, 140, (Ee, 2 * I, H // 32), generator=gc, dtype=torch.uint8)
    w2 = torch.randint(0, 16, (Ee, H, I // 2), generator=gc, dtype=torch.uint8)
    s2 = torch.randint(118, 140, (Ee, H, I // 32), generator=gc, dtype=torch.uint8)
    return w13, s13, w2, s2


def bench_seq(ws, x, ids, tw, n=3):
    w13, s13, w2, s2 = ws
    gpu_moe_layer(x, ids, tw, w13, s13, w2, s2, H=H, I=I, K=K, device=dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        gpu_moe_layer(x, ids, tw, w13, s13, w2, s2, H=H, I=I, K=K, device=dev)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def bench_ovl(ws, x, ids, tw, n=3):
    w13, s13, w2, s2 = ws
    pw = tuple(_pinned_kmajor(t) for t in (w13, s13, w2, s2))
    # ring of layers: prefetch L+1 while computing L
    slot = prefetch_layer(*pw, dev)
    gpu_moe_layer(x, ids, tw, w13, s13, w2, s2, H=H, I=I, K=K, device=dev, slot=slot)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        nxt = prefetch_layer(*pw, dev)
        gpu_moe_layer(x, ids, tw, w13, s13, w2, s2, H=H, I=I, K=K, device=dev, slot=slot)
        slot = nxt
    # last prefetch is not consumed: drain it
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main():
    import json

    Ts = [int(t) for t in os.environ.get(
        "TS", "2048,4096,8192,16384,32768,49152").split(",")]
    Es = [int(e) for e in os.environ.get("E_LIST", "256,128").split(",")]
    Ks = [int(k) for k in os.environ.get("TOPKS", "6").split(",")]
    if len(Ks) < len(Es):
        Ks = Ks + [Ks[-1]] * (len(Es) - len(Ks))
    out_path = os.environ.get("OUT", "")
    res: dict = {"H": H, "I": I, "K": K, "layers": LAYERS}

    for _ei, Ee in enumerate(Es):
        Kk = Ks[_ei]
        ws = make_weights(Ee)
        nbytes = sum(t.numel() for t in ws)
        tag = "ovl" if (Ee == 256 and Kk == 6) else f"ovl_E{Ee}_K{Kk}"
        seq_tag = "seq" if (Ee == 256 and Kk == 6) else f"seq_E{Ee}_K{Kk}"
        print(f"\n### E={Ee} topk={Kk}  raw bytes/layer = {nbytes/2**30:.3f} GiB  "
              f"({tag})", flush=True)
        t0 = time.perf_counter()
        pw = tuple(_pinned_kmajor(t) for t in ws)
        torch.cuda.synchronize()
        print(f"host K-major pin+transpose = {(time.perf_counter()-t0)*1e3:.0f} ms",
              flush=True)
        del pw
        res[tag] = {}
        res[seq_tag] = {}
        print(f"{'T':>7} {'seq ms':>9} {'ovl ms':>9} {'speedup':>8} "
              f"{'seq tok/s':>10} {'ovl tok/s':>10}", flush=True)
        for T in Ts:
            x = torch.randn(T, H, generator=gg, dtype=torch.bfloat16, device=dev)
            ids = torch.randint(0, Ee, (T, Kk), dtype=torch.int64, device=dev)
            tw = torch.rand(T, Kk, device=dev) + 0.5
            ms_s = bench_seq(ws, x, ids, tw) * 1e3
            ms_o = bench_ovl(ws, x, ids, tw) * 1e3
            res[seq_tag][str(T)] = T / (ms_s * LAYERS / 1e3)
            res[tag][str(T)] = T / (ms_o * LAYERS / 1e3)
            res.setdefault("ms", {})[f"{tag}@{T}"] = round(ms_o, 1)
            res["ms"][f"{seq_tag}@{T}"] = round(ms_s, 1)
            print(f"{T:7d} {ms_s:9.1f} {ms_o:9.1f} {ms_s/ms_o:8.2f}x "
                  f"{T/(ms_s*LAYERS/1e3):10.0f} {T/(ms_o*LAYERS/1e3):10.0f}",
                  flush=True)
            del x, ids, tw
            torch.cuda.empty_cache()
        del ws

    if out_path:
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)
        print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

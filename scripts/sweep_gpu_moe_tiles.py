#!/usr/bin/env python
"""Tile-size sweep for the MXFP4 grouped MoE kernels (overlap path).

Sweeps BM/BN/BK/BH/NS on synthetic full-DS-V4-dimension weights at a few token
counts and prints per-layer ms + 43-layer-equivalent tok/s, so the kernel config
can be chosen without reloading the 156 GB model.
Env: TS (comma list), SWEEP (comma list of BN values), etc.
"""
import itertools
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
dev = torch.cuda.current_device()
gc = torch.Generator(device="cpu").manual_seed(0)
gg = torch.Generator(device=dev).manual_seed(0)


def main():
    w13 = torch.randint(0, 16, (E, 2 * I, H // 2), generator=gc, dtype=torch.uint8)
    s13 = torch.randint(118, 140, (E, 2 * I, H // 32), generator=gc, dtype=torch.uint8)
    w2 = torch.randint(0, 16, (E, H, I // 2), generator=gc, dtype=torch.uint8)
    s2 = torch.randint(118, 140, (E, H, I // 32), generator=gc, dtype=torch.uint8)
    pw = tuple(_pinned_kmajor(t) for t in (w13, s13, w2, s2))
    Ts = [int(t) for t in os.environ.get("TS", "16384").split(",")]
    BMs = [int(v) for v in os.environ.get("BMS", "64,128").split(",")]
    BNs = [int(v) for v in os.environ.get("BNS", "64,128,256").split(",")]
    BKs = [int(v) for v in os.environ.get("BKS", "64,128").split(",")]
    BHs = [int(v) for v in os.environ.get("BHS", "64,128").split(",")]
    NSs = [int(v) for v in os.environ.get("NSS", "2,3,4").split(",")]
    NREP = int(os.environ.get("NREP", "3"))

    for T in Ts:
        x = torch.randn(T, H, generator=gg, dtype=torch.bfloat16, device=dev)
        ids = torch.randint(0, E, (T, K), dtype=torch.int64, device=dev)
        tw = torch.rand(T, K, device=dev) + 0.5
        print(f"\n=== T={T} ===", flush=True)
        print(f"{'BM':>4}{'BN':>5}{'BK':>5}{'BH':>5}{'NS':>3} {'ms/layer':>9} {'tok/s':>7}",
              flush=True)
        best = None
        for BM, BN, BK, BH, NS in itertools.product(BMs, BNs, BKs, BHs, NSs):
            os.environ.update(
                XIAOTU_GPU_PREFILL_BM=str(BM), XIAOTU_GPU_PREFILL_BN=str(BN),
                XIAOTU_GPU_PREFILL_BK=str(BK), XIAOTU_GPU_PREFILL_BH=str(BH),
                XIAOTU_GPU_PREFILL_STAGES=str(NS),
            )
            try:
                slot = prefetch_layer(*pw, dev)
                gpu_moe_layer(x, ids, tw, w13, s13, w2, s2, H=H, I=I, K=K,
                              device=dev, slot=slot)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(NREP):
                    nxt = prefetch_layer(*pw, dev)
                    gpu_moe_layer(x, ids, tw, w13, s13, w2, s2, H=H, I=I, K=K,
                                  device=dev, slot=slot)
                    slot = nxt
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) / NREP * 1e3
                tps = T / (ms * 43 / 1e3)
                print(f"{BM:>4}{BN:>5}{BK:>5}{BH:>5}{NS:>3} {ms:9.1f} {tps:7.0f}", flush=True)
                if best is None or ms < best[0]:
                    best = (ms, (BM, BN, BK, BH, NS))
            except Exception as e:  # noqa: BLE001
                print(f"{BM:>4}{BN:>5}{BK:>5}{BH:>5}{NS:>3}  FAIL "
                      f"{type(e).__name__}: {str(e)[:60]}", flush=True)
        if best:
            print(f"BEST T={T}: {best[0]:.1f} ms  cfg(BM,BN,BK,BH,NS)={best[1]}", flush=True)
        del x, ids, tw
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

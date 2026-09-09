#!/usr/bin/env python
"""Microbenchmark: cost of gpu_moe_layer (one layer, full DS-V4 dims) vs T.

Synthetic weights in the EXACT fused layout the plugin streams ([E,2I,H//2] u8
etc., realistic scale range) -> same memory traffic as real weights; used for
timing. Also breaks out the dequant-slice fixed cost vs the two kernels.
"""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
import torch
import triton
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_xiaotu_moe.gpu_prefill import gpu_moe_layer

H, I, E, K = 4096, 2048, 256, 6
dev = torch.cuda.current_device()
gc = torch.Generator(device='cpu').manual_seed(0)
gg = torch.Generator(device=dev).manual_seed(0)

def make_weights():
    w13 = torch.randint(0, 16, (E, 2*I, H//2), generator=gc, dtype=torch.uint8)
    s13 = torch.randint(118, 140, (E, 2*I, H//32), generator=gc, dtype=torch.uint8)
    w2 = torch.randint(0, 16, (E, H, I//2), generator=gc, dtype=torch.uint8)
    s2 = torch.randint(118, 140, (E, H, I//32), generator=gc, dtype=torch.uint8)
    return w13, s13, w2, s2

def main():
    w13, s13, w2, s2 = make_weights()
    # move to GPU (this is the per-layer H2D streaming step)
    t0 = time.perf_counter()
    w13d = w13.to(dev); s13d = s13.to(dev); w2d = w2.to(dev); s2d = s2.to(dev)
    torch.cuda.synchronize()
    print(f"H2D full layer: {(time.perf_counter()-t0)*1e3:.1f} ms ({ (w13d.numel()+s13d.numel()+w2d.numel()+s2d.numel())/1e6:.0f} MB)")

    for T in [512, 1024, 2048, 4096]:
        x = torch.randn(T, H, generator=gg, dtype=torch.bfloat16, device=dev)
        topk_ids = torch.randint(0, E, (T, K), dtype=torch.int64, device=dev)
        tw = torch.rand(T, K, device=dev) + 0.5
        # warmup
        gpu_moe_layer(x, topk_ids, tw, w13, s13, w2, s2, H=H, I=I, K=K, device=dev)
        torch.cuda.synchronize()
        n = 3
        t0 = time.perf_counter()
        for _ in range(n):
            out = gpu_moe_layer(x, topk_ids, tw, w13, s13, w2, s2, H=H, I=I, K=K, device=dev)
        torch.cuda.synchronize()
        dt = (time.perf_counter()-t0)/n
        print(f"T={T:5d}: gpu_moe_layer one layer = {dt*1e3:8.1f} ms  "
              f"({T*6*13e9/1e12:.1f} TFLOP -> {T*6*13e9/1e12/dt:.0f} TFLOP/s eff)")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Golden test: vllm_xiaotu_moe.gpu_prefill.gpu_moe_layer vs pure-torch reference.

Small dims (fast) + optional full DS-V4 dims (T=4). Requires a CUDA GPU; uses
random data in the exact MXFP4 nibble+e8m0 layout the CPU engine reads.
Env: TEST_FULL=1 to also run full DS-V4 dims.
"""
import math
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_xiaotu_moe.gpu_prefill import gpu_moe_layer, torch_reference_layer


def make_layer(H, I, E, seed=0):
    g = torch.Generator().manual_seed(seed)
    w13 = torch.zeros(E, 2 * I, H // 2, dtype=torch.uint8)
    w2 = torch.zeros(E, H, I // 2, dtype=torch.uint8)
    torch.randint(0, 16, w13.shape, generator=g, out=w13)
    torch.randint(0, 16, w2.shape, generator=g, out=w2)
    # realistic e8m0 byte range (2^-12..2^13) avoiding fp32 overflow
    s13 = (torch.randint(118, 127, (E, 2 * I, H // 32), generator=g)).to(torch.uint8)
    s2 = (torch.randint(118, 127, (E, H, I // 32), generator=g)).to(torch.uint8)
    return w13, s13, w2, s2


def run_case(H, I, E, T, K, device, seed):
    w13, s13, w2, s2 = make_layer(H, I, E, seed)
    w13g = w13.to(device); s13g = s13.to(device); w2g = w2.to(device); s2g = s2.to(device)
    x = torch.randn(T, H, dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(0, E, (T, K), device=device, dtype=torch.int64)
    tw = torch.rand(T, K, device=device) + 0.5
    got = gpu_moe_layer(x, topk_ids, tw, w13g, s13g, w2g, s2g,
                        H=H, I=I, K=K, device=device)
    ref = torch_reference_layer(x, topk_ids, tw, w13g, s13g, w2g, s2g, H, I)
    diff = (got.float() - ref.float()).abs()
    rel = diff / (ref.float().abs() + 1e-3)
    rms = (diff.pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt()).item()
    print(f"  H={H} I={I} E={E} T={T} K={K}: max_abs={diff.max().item():.5f} "
          f"max_rel={rel.max().item():.5f} rms_rel={rms:.2e} "
          f"out_absmean={ref.abs().mean().item():.4f}")
    assert diff.max().item() < 0.5 * ref.abs().mean().item() + 1e-2, "golden mismatch"
    return diff.max().item()


def main() -> int:
    dev = torch.cuda.current_device()
    torch.cuda.set_device(dev)
    print("device:", torch.cuda.get_device_name(dev))
    print("[golden] small dims:")
    run_case(H=256, I=128, E=8, T=32, K=3, device=dev, seed=1)
    run_case(H=256, I=128, E=8, T=7, K=2, device=dev, seed=2)   # ragged M
    if os.environ.get("TEST_FULL") == "1":
        print("[golden] full DS-V4 dims (T=4):")
        run_case(H=4096, I=2048, E=256, T=4, K=6, device=dev, seed=3)
    print("[golden] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

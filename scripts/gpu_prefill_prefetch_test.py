#!/usr/bin/env python
"""Golden test for the prefetch-ahead (slot) path of gpu_prefill.gpu_moe_layer.

Verifies that the async H2D ring (host K-major pinned cache -> device slot on a
side stream) produces the same result as the synchronous path, and that a
2-slot ring reused across >2 layers does not race (the consumer's busy event
must gate the next copy into the same slot).

Env: TEST_FULL=1 to also run full DS-V4 dims (T=4, E=8 slice).
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_xiaotu_moe.gpu_prefill import (  # noqa: E402
    _pinned_kmajor,
    gpu_moe_layer,
    prefetch_layer,
    torch_reference_layer,
)


def make_layer(H, I, E, seed=0):
    g = torch.Generator().manual_seed(seed)
    w13 = torch.zeros(E, 2 * I, H // 2, dtype=torch.uint8)
    w2 = torch.zeros(E, H, I // 2, dtype=torch.uint8)
    torch.randint(0, 16, w13.shape, generator=g, out=w13)
    torch.randint(0, 16, w2.shape, generator=g, out=w2)
    s13 = torch.randint(118, 127, (E, 2 * I, H // 32), generator=g).to(torch.uint8)
    s2 = torch.randint(118, 127, (E, H, I // 32), generator=g).to(torch.uint8)
    return w13, s13, w2, s2


def run_case(H, I, E, T, K, device, seed, nlayers=1):
    w13, s13, w2, s2 = make_layer(H, I, E, seed)
    x = torch.randn(T, H, dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(0, E, (T, K), device=device, dtype=torch.int64)
    tw = torch.rand(T, K, device=device) + 0.5

    # reference from the raw CPU layout
    ref = torch_reference_layer(
        x, topk_ids, tw,
        w13.to(device), s13.to(device), w2.to(device), s2.to(device), H, I,
    )

    # slot path: prefetch on a side stream, consume on the compute stream
    w13p, s13p = _pinned_kmajor(w13), _pinned_kmajor(s13)
    w2p, s2p = _pinned_kmajor(w2), _pinned_kmajor(s2)
    slot = prefetch_layer(w13p, s13p, w2p, s2p, device)
    assert slot is not None, "prefetch_layer returned None (weights already on device?)"
    got = gpu_moe_layer(x, topk_ids, tw, w13, s13, w2, s2,
                        H=H, I=I, K=K, device=device, slot=slot)
    diff = (got.float() - ref.float()).abs()
    rel = diff / (ref.float().abs() + 1e-3)
    print(f"  [slot] H={H} I={I} E={E} T={T} K={K}: max_abs={diff.max().item():.5f} "
          f"max_rel={rel.max().item():.5f}")
    assert diff.max().item() < 0.5 * ref.abs().mean().item() + 1e-2, "slot mismatch"

    # ring reuse: 4 sequential layers through a 2-slot ring, all must be correct
    outs = []
    for li in range(4):
        wl = make_layer(H, I, E, seed=100 + li)
        wl_ref = torch_reference_layer(
            x, topk_ids, tw,
            wl[0].to(device), wl[1].to(device), wl[2].to(device), wl[3].to(device), H, I,
        )
        sl = prefetch_layer(_pinned_kmajor(wl[0]), _pinned_kmajor(wl[1]),
                            _pinned_kmajor(wl[2]), _pinned_kmajor(wl[3]), device)
        o = gpu_moe_layer(x, topk_ids, tw, wl[0], wl[1], wl[2], wl[3],
                          H=H, I=I, K=K, device=device, slot=sl)
        d = (o.float() - wl_ref.float()).abs().max().item()
        tol = 0.5 * wl_ref.abs().mean().item() + 1e-2
        outs.append((d, tol))
    print(f"  [ring x4] max_abs/tol per layer = "
          f"{[(f'{d:.4f}', f'{t:.1f}') for d, t in outs]}")
    assert all(d < t for d, t in outs), "ring reuse mismatch"
    return diff.max().item()


def main() -> int:
    dev = torch.cuda.current_device()
    torch.cuda.set_device(dev)
    print("device:", torch.cuda.get_device_name(dev))
    print("[prefetch-golden] small dims:")
    run_case(H=256, I=128, E=8, T=32, K=3, device=dev, seed=1)
    run_case(H=256, I=128, E=8, T=7, K=2, device=dev, seed=2)
    if os.environ.get("TEST_FULL") == "1":
        print("[prefetch-golden] full DS-V4 dims (T=4, E=8):")
        run_case(H=4096, I=2048, E=8, T=4, K=6, device=dev, seed=3)
    print("[prefetch-golden] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

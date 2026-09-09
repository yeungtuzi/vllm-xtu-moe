#!/usr/bin/env python
"""Standalone probe: do the mHC tilelang kernels produce zeros on SM80?

Compares the small-M path (mhc_fused_tilelang, num_tokens <= 16) against the
large-M path (mhc_post + hc_prenorm_gemm) and the broadcast pre path, for a
sweep of token counts. Pure kernels, no model weights, no engine.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("PROBE_GPU", "2"))

import torch

from vllm.model_executor.kernels.mhc.tilelang import (
    mhc_fused_post_pre_tilelang,
    mhc_pre_broadcast_tilelang,
)

H = 4096
HC = 4
HC2 = HC * HC
HC3 = HC * 2 + HC2
DEV = "cuda"
DT = torch.bfloat16


def mk_inputs(m: int, seed: int = 0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(m, H, dtype=DT, device=DEV, generator=g) * 0.1
    residual = torch.randn(m, HC, H, dtype=DT, device=DEV, generator=g) * 0.1
    post_mix = torch.rand(m, HC, 1, dtype=torch.float32, device=DEV, generator=g) * 0.5
    comb = torch.randn(m, HC, HC, dtype=torch.float32, device=DEV, generator=g) * 0.1
    fn = torch.randn(HC3, HC * H, dtype=torch.float32, device=DEV, generator=g) * 0.01
    hc_scale = torch.randn(3, dtype=torch.float32, device=DEV, generator=g) * 0.1
    hc_base = torch.randn(HC3, dtype=torch.float32, device=DEV, generator=g) * 0.1
    norm_w = torch.ones(H, dtype=DT, device=DEV)
    fn_bcast = torch.randn(HC3, H, dtype=torch.float32, device=DEV, generator=g) * 0.01
    return x, residual, post_mix, comb, fn, hc_scale, hc_base, norm_w, fn_bcast


def stats(t: torch.Tensor) -> str:
    return f"{t.abs().mean().item():.3e}/nz{(t != 0).float().mean().item():.2f}"


def main() -> None:
    print("=== mhc_fused_post_pre_tilelang (num_tokens<=16 -> mhc_fused_tilelang) ===")
    for m in [1, 2, 5, 8, 16, 17, 32, 64, 128, 256, 1024, 4096]:
        x, res, pm, cm, fn, sc, bs, nw, _ = mk_inputs(m)
        r, p, c, li = mhc_fused_post_pre_tilelang(
            x, res, pm, cm, fn, sc, bs, 1e-6, 1e-6, 1e-6, 1.0, 20,
            n_splits=1, tile_n=1, norm_weight=nw, norm_eps=1e-6,
        )
        print(f"  M={m:5d} res={stats(r)} pm={stats(p)} cm={stats(c)} layer_input={stats(li)}")

    print("=== mhc_pre_broadcast_tilelang (layer 0) ===")
    for m in [1, 2, 5, 8, 16, 17, 64, 1024, 8192]:
        x, res, pm, cm, fn, sc, bs, nw, fb = mk_inputs(m)
        r, p, c, li = mhc_pre_broadcast_tilelang(
            x, fn, sc, bs, 1e-6, 1e-6, 1e-6, 1.0, 20,
            norm_weight=nw, norm_eps=1e-6, fn_broadcast=fb,
        )
        print(f"  M={m:5d} res={stats(r)} pm={stats(p)} cm={stats(c)} layer_input={stats(li)}")

    print("=== raw prenorm gemm variants (hidden 16384) ===")
    from vllm.model_executor.kernels.mhc.tilelang import _tilelang_hc_prenorm_gemm

    for m in [1, 5, 16, 64, 256, 1024, 4096]:
        g = torch.Generator(device=DEV).manual_seed(1)
        x2 = torch.randn(m, HC * H, dtype=DT, device=DEV, generator=g) * 0.1
        fn = torch.randn(HC3, HC * H, dtype=torch.float32, device=DEV, generator=g) * 0.01
        out = torch.empty(1, m, HC3, dtype=torch.float32, device=DEV)
        sq = torch.empty(1, m, dtype=torch.float32, device=DEV)
        _tilelang_hc_prenorm_gemm(x2, fn, out, sq, H, HC)
        ref = x2.float() @ fn.t()
        err = (out[0] - ref).abs().max().item()
        print(f"  M={m:5d} out={stats(out)} sqrsum={stats(sq)} maxerr_vs_torch={err:.3e}")


if __name__ == "__main__":
    main()

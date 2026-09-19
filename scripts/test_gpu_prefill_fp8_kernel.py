#!/usr/bin/env python
"""Numerical check for the FP8 (e4m3 block-128) GPU prefill kernels.

Runs ``gate_up_kernel_fp8`` + ``down_kernel_fp8`` on synthetic weights and
compares against a torch reference that uses the SAME segmentation tensors and
the SAME bf16-quantized operands the kernels use, so what is under test is the
kernel itself (indexing, the block-128 scale placement, the swiglu clamp), not
the accepted bf16-operand policy.

Differences from the reference are then only fp32 accumulation order.

Usage: python scripts/test_gpu_prefill_fp8_kernel.py [E H I T K]
"""
from __future__ import annotations

import os
import sys

import torch
import triton

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from vllm_xiaotu_moe.gpu_prefill import _build_segmentation  # noqa: E402
from vllm_xiaotu_moe.gpu_prefill_fp8 import (  # noqa: E402
    down_kernel_fp8,
    gate_up_kernel_fp8,
)

E4M3_MAX = 448.0


def rand_fp8(shape, gen):
    """Random e4m3 bytes with a realistic small magnitude (no NaN encodings)."""
    f = torch.randn(shape, generator=gen) * 0.1
    u8 = f.to(torch.float8_e4m3fn).view(torch.uint8).clone()
    u8[(u8 == 0x7F) | (u8 == 0xFF)] = 0
    return u8


def e4m3_to_f32(u8: torch.Tensor) -> torch.Tensor:
    """Reference decode; matches fp8_utils._e4m3_uint8_to_f32 (bias 7, 2^-6 sub)."""
    u = u8.to(torch.int32)
    sign = (u >> 7) & 1
    exp = (u >> 3) & 0xF
    man = u & 0x7
    mant = man.float() * 0.125
    val = torch.where(
        exp != 0,
        torch.exp2((exp - 7).float()) * (1.0 + mant),
        torch.full_like(mant, 0.015625) * mant,
    )
    return torch.where(sign != 0, -val, val)


def quant_operand(w_u8, scale, n_blk, k_blk):
    """(dequant(w) * scale) rounded to bf16, exactly as the kernel builds b.

    w_u8: [N, K] uint8; scale: [N//128, K//128] fp32.
    """
    N, K = w_u8.shape
    assert N == n_blk * 128 and K == k_blk * 128, (N, K, n_blk, k_blk)
    deq = e4m3_to_f32(w_u8)
    sc = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return (deq * sc).to(torch.bfloat16).float()


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    args = [int(x) for x in sys.argv[1:]]
    E, H, I, T, K = args if len(args) == 5 else (4, 512, 256, 8, 2)
    dev = "cuda:0"
    g = torch.Generator().manual_seed(0)

    # ---- weights: w13 [E, 2I, H], w2 [E, H, I]; scales block 128x128 ----------
    w13 = rand_fp8((E, 2 * I, H), g)
    w2 = rand_fp8((E, H, I), g)
    s13 = (10.0 ** torch.empty(E, 2 * I // 128, H // 128).uniform_(-2, 1, generator=g)).float()
    s2 = (10.0 ** torch.empty(E, H // 128, I // 128).uniform_(-2, 1, generator=g)).float()

    # K-major (what the streaming path stages): [E, K, N]
    w13t = w13.transpose(1, 2).contiguous().to(dev)
    w2t = w2.transpose(1, 2).contiguous().to(dev)
    s13t = s13.transpose(1, 2).contiguous().to(dev)     # [E, H/128, 2I/128]
    s2t = s2.transpose(1, 2).contiguous().to(dev)       # [E, I/128, H/128]

    # ---- activations + routing ----------------------------------------------
    x = (torch.randn(T, H, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    topk_ids = torch.randint(0, E, (T, K), generator=g).to(torch.int32).to(dev)
    topk_weights = torch.rand(T, K, generator=g).float().to(dev) * 0.2 + 0.05

    tok, wts, seg_start, A = _build_segmentation(topk_ids, topk_weights, E, dev)
    print(f"E={E} H={H} I={I} T={T} K={K} A={A} segments={seg_start.shape[0]}")

    BM, BN, BK, BH, NS = 64, 64, 64, 64, 2
    inter = torch.zeros((A, 2 * I), dtype=torch.bfloat16, device=dev)
    gate_up_kernel_fp8[(E, triton.cdiv(2 * I, BN))](
        x, x.stride(0), tok, seg_start,
        w13t, w13t.stride(1), s13t, s13t.stride(1),
        inter, inter.stride(0), w13t.stride(0), s13t.stride(0),
        H=H, BM=BM, BN=BN, BK=BK, NS=NS, num_warps=4,
    )
    LIMIT = 10.0
    # fp32 accumulation: a bf16 `out` makes tl.atomic_add accumulate in bf16
    # across the top_k partial sums, which is a real (avoidable) error source.
    out = torch.zeros((T, H), dtype=torch.float32, device=dev)
    down_kernel_fp8[(E, triton.cdiv(H, BH))](
        inter, inter.stride(0), tok, wts, seg_start,
        w2t, w2t.stride(1), s2t, s2t.stride(1),
        out, out.stride(0), w2t.stride(0), s2t.stride(0),
        H=H, I=I, BM=BM, BH=BH, BK=BK, NS=NS, LIMIT=LIMIT, num_warps=4,
    )
    torch.cuda.synchronize()

    # ---- reference ----------------------------------------------------------
    # Split into the two stages so a bf16 rounding-boundary flip in `act`
    # (values reach O(100); one bf16 ulp is 0.4%) cannot be mistaken for a kernel
    # bug: (a) compares gate_up on its own, (b) drives `down` from the KERNEL's
    # own gate/up output, so both sides round the identical f32 values to bf16.
    tok_c, wts_c, seg_c = tok.cpu(), wts.cpu().float(), seg_start.cpu()

    ref_inter = torch.zeros(A, 2 * I, dtype=torch.float32)
    for e in range(E):
        lo, hi = int(seg_c[e]), int(seg_c[e + 1])
        if hi <= lo:
            continue
        b13 = quant_operand(w13[e], s13[e], 2 * I // 128, H // 128)  # [2I, H]
        toks = tok_c[lo:hi]
        ref_inter[lo:hi] = x[toks].float().cpu() @ b13.T

    got_inter = inter.float().cpu()
    rms_i = ref_inter.pow(2).mean().sqrt().item()
    rel_i = ((got_inter - ref_inter).pow(2).mean().sqrt() / (rms_i + 1e-12)).item()
    # `inter` is bf16, so the only admissible difference is one bf16 ulp
    # (2^-8 = 0.39%); allow 2x for the round-to-nearest boundary.
    tol_i = (ref_inter.abs() * 0.0079) + 1e-6
    over_i = int((got_inter - ref_inter).abs().gt(tol_i).sum())
    rel_i_bf = (
        ((got_inter - ref_inter.to(torch.bfloat16).float()).pow(2).mean().sqrt()
         / (rms_i + 1e-12)).item()
    )

    ref_out = torch.zeros(T, H, dtype=torch.float32)
    for e in range(E):
        lo, hi = int(seg_c[e]), int(seg_c[e + 1])
        if hi <= lo:
            continue
        toks = tok_c[lo:hi]
        gate = got_inter[lo:hi, :I]
        up = got_inter[lo:hi, I:]
        gate = torch.clamp(gate, max=LIMIT)
        up = torch.clamp(up, -LIMIT, LIMIT)
        act = ((gate / (1.0 + torch.exp(-gate))) * up).to(torch.bfloat16).float()
        b2 = quant_operand(w2[e], s2[e], H // 128, I // 128)        # [H, I]
        y = act @ b2.T
        ref_out.index_add_(0, toks, y * wts_c[lo:hi][:, None])

    got = out.float().cpu()
    diff = (got - ref_out).abs()
    rms = ref_out.pow(2).mean().sqrt().item()
    rms_rel = (diff.pow(2).mean().sqrt() / (rms + 1e-12)).item()
    max_abs = diff.max().item()
    tol = (ref_out.abs() * 0.0079) + 1e-6
    over = int(diff.gt(tol).sum())
    # A small fraction of elements sit within a few fp32 ulps of a bf16 rounding
    # boundary and land on the far side, so an exact-zero count is the wrong
    # criterion. Expect ~1e-5..1e-4 of elements (fp32-order / bf16-ulp ratio);
    # a real indexing or scale bug would move a large fraction or blow up rms.
    n_i, n_o = got_inter.numel(), got.numel()
    ok = (over_i / n_i < 1e-3 and over / n_o < 1e-3
          and rel_i_bf < 1e-3 and rms_rel < 1e-4)
    print(f"[{'OK ' if ok else 'BAD'}] FP8 GPU kernels vs torch reference: "
          f"gate_up rms_rel={rel_i_bf:.3e} (over-1ulp={over_i}/{n_i}) | "
          f"down rms_rel={rms_rel:.3e} (over-1ulp={over}/{n_o}) "
          f"max_abs={max_abs:.3e} ref_rms={rms:.4f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

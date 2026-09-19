"""FP8 (e4m3, block-128) GPU prefill kernels for the hybrid plugin.

Companion to ``gpu_prefill.py`` (which does MXFP4/block-32). Differences that
matter:

* weights are **1 byte/element** (not two elements/byte), so the K-major byte
  transpose in ``byte_transpose.ktranspose_bytes`` works unchanged -- it is a
  plain uint8 transpose;
* the block scale is **128x128** (not groupN=1/groupK=32), so with tile sizes
  that divide 128 the scale is a **single fp32 scalar per (k-block, n-block)**
  and can be loaded as a scalar instead of a [BK/32, BN] tile plus broadcast;
* A100 has no FP8 tensor cores, so the tile is decoded to bf16 and the MMA runs
  in bf16 -- the same thing the MXFP4 path does.

Decoding uses ``_e4m3_uint8_to_f32`` from vLLM's fp8_utils, the arithmetic
(fp8e4nv-free) decoder added for Ampere by the DSv4/V4.1 SM80 work; Triton has no
fp8e4nv type on SM80.

The down kernel applies the gated-SiLU clamp (``swiglu_limit``) the way the CPU
engine does: gate clamped on the upper side only, up clamped on both sides.
"""

from __future__ import annotations

import triton
import triton.language as tl

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _e4m3_uint8_to_f32,
)


@triton.jit
def gate_up_kernel_fp8(
    x_ptr, x_stride, tok_ptr,
    seg_ptr,
    w13t_ptr, w13t_row_stride, s13t_ptr, s13t_row_stride,
    inter_ptr, inter_ld,
    W13_E, S13_E,
    H: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NS: tl.constexpr,
):
    """inter[g, 0:2I] = x[tok[g], :] @ dequant(w13)^T for one expert's rows.

    w13t is [E, H, 2I] uint8 K-major; s13t is [E, H/128, 2I/128] fp32 K-major.
    Requires BK <= 128 (divides it) and BN <= 128 with the n-tile aligned, so the
    block scale is constant across the tile.
    """
    pid_e = tl.program_id(0)
    pid_n = tl.program_id(1)
    base = tl.load(seg_ptr + pid_e)
    end = tl.load(seg_ptr + pid_e + 1)
    if end <= base:
        return
    n = pid_n * BN + tl.arange(0, BN)
    n_blk = (pid_n * BN) // 128
    # int64: an expert's K-major block exceeds 2^31 bytes for these models.
    w_off = pid_e.to(tl.int64) * W13_E
    s_off = pid_e.to(tl.int64) * S13_E
    mt = 0
    while mt * BM < (end - base):
        g_rows = base + mt * BM + tl.arange(0, BM)
        rmask = g_rows < end
        toks = tl.load(tok_ptr + g_rows, mask=rmask, other=0)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for kk in tl.range(0, H, BK, num_stages=NS):
            a = tl.load(
                x_ptr + toks[:, None] * x_stride + (kk + tl.arange(0, BK))[None, :],
                mask=rmask[:, None], other=0.0)
            b_u8 = tl.load(
                w13t_ptr + w_off
                + (kk + tl.arange(0, BK))[:, None] * w13t_row_stride + n[None, :])
            sc = tl.load(s13t_ptr + s_off + (kk >> 7) * s13t_row_stride + n_blk)
            b = (_e4m3_uint8_to_f32(b_u8) * sc).to(tl.bfloat16)
            acc = tl.dot(a, b, acc)
        tl.store(
            inter_ptr + g_rows[:, None] * inter_ld + n[None, :],
            acc.to(tl.bfloat16), mask=rmask[:, None])
        mt += 1


@triton.jit
def down_kernel_fp8(
    inter_ptr, inter_ld, tok_ptr, wts_ptr,
    seg_ptr,
    w2t_ptr, w2t_row_stride, s2t_ptr, s2t_row_stride,
    out_ptr, out_ld,
    W2_E, S2_E,
    H: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    NS: tl.constexpr,
    LIMIT: tl.constexpr,
):
    """out[tok, :] += w * (silu_gate(gate) * up) @ dequant(w2)^T, per assignment."""
    pid_e = tl.program_id(0)
    pid_h = tl.program_id(1)
    base = tl.load(seg_ptr + pid_e)
    end = tl.load(seg_ptr + pid_e + 1)
    if end <= base:
        return
    h = pid_h * BH + tl.arange(0, BH)
    h_blk = (pid_h * BH) // 128
    w_off = pid_e.to(tl.int64) * W2_E
    s_off = pid_e.to(tl.int64) * S2_E
    mt = 0
    while mt * BM < (end - base):
        g_rows = base + mt * BM + tl.arange(0, BM)
        rmask = g_rows < end
        toks = tl.load(tok_ptr + g_rows, mask=rmask, other=0)
        wts = tl.load(wts_ptr + g_rows, mask=rmask, other=0.0).to(tl.float32)
        acc = tl.zeros((BM, BH), dtype=tl.float32)
        for kk in tl.range(0, I, BK, num_stages=NS):
            gate = tl.load(
                inter_ptr + g_rows[:, None] * inter_ld + (kk + tl.arange(0, BK))[None, :],
                mask=rmask[:, None], other=0.0).to(tl.float32)
            up = tl.load(
                inter_ptr + g_rows[:, None] * inter_ld + (I + kk + tl.arange(0, BK))[None, :],
                mask=rmask[:, None], other=0.0).to(tl.float32)
            if LIMIT > 0.0:
                gate = tl.minimum(gate, LIMIT)
                up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
            act = (gate / (1.0 + tl.exp(-gate))) * up
            b_u8 = tl.load(
                w2t_ptr + w_off
                + (kk + tl.arange(0, BK))[:, None] * w2t_row_stride + h[None, :])
            sc = tl.load(s2t_ptr + s_off + (kk >> 7) * s2t_row_stride + h_blk)
            b = (_e4m3_uint8_to_f32(b_u8) * sc).to(tl.bfloat16)
            acc = tl.dot(act.to(tl.bfloat16), b, acc)
        tl.atomic_add(
            out_ptr + toks[:, None] * out_ld + h[None, :],
            acc * wts[:, None], mask=rmask[:, None], sem="relaxed")
        mt += 1

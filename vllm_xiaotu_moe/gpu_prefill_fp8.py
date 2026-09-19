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

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _e4m3_uint8_to_f32,
)

from vllm_xiaotu_moe.gpu_prefill import (
    _build_segmentation,
    _dma_hostbuf,
    _kmajor_bytes,
)

FP8_BLOCK = 128

# Dedicated named buffers. NOTE: we deliberately do NOT use gpu_prefill._reuse
# here -- its cache key is (device, shape) and ignores the logical `key`, so for
# GLM-5.3 (H == 2I == 4096) the w13 row layout [E, 2I, H] and its K-major target
# [E, H, 2I] have the SAME shape and would hand back the same tensor, i.e. an
# in-place transpose that corrupts itself. (w2 escaped it only because
# [E, H, I] != [E, I, H].) Names keep the buffers distinct.
_FP8_BUF: dict = {}


def _buf(name: str, shape, device, dtype=torch.uint8):
    sh = tuple(int(v) for v in shape)
    k = (str(device), str(name), sh, dtype)
    t = _FP8_BUF.get(k)
    if t is None:
        t = torch.empty(sh, dtype=dtype, device=device)
        _FP8_BUF[k] = t
    return t


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


# ---------------------------------------------------------------------------
# Host-side assembly + staging (FP8 twins of gpu_prefill's MXFP4 versions)
# ---------------------------------------------------------------------------


def fp8_staging_bytes(n_experts: int, hidden: int, inter: int, ns: int = 2) -> int:
    """Peak device bytes one layer's FP8 streaming path needs.

    Same three-part accounting as ``gpu_prefill.staging_bytes`` (2 ring slots +
    the assembly target + one node's DMA staging), but with FP8 widths: 1 byte
    per weight element and a [N/128, K/128] fp32 block scale. That makes it ~2x
    the MXFP4 number for the weights, which matters for the VRAM preflight.
    """
    E, H, I = int(n_experts), int(hidden), int(inter)
    b = FP8_BLOCK
    w13d = E * (2 * I) * H
    w2d = E * H * I
    s13d = E * ((2 * I) // b) * (H // b) * 4
    s2d = E * (H // b) * (I // b) * 4
    n = max(1, int(ns))
    slots = 2 * (w13d + w2d + s13d + s2d)
    raw = w13d + w2d
    tmp = (w13d + w2d + s13d + s2d) / n
    return int(slots + raw + tmp)


def kmajor_from_engine_shards_fp8(engine, device, hidden: int, inter: int,
                                  n_experts: int, dst=None):
    """K-major FP8 device weights built from the engine's own NUMA shards.

    FP8 twin of ``gpu_prefill.kmajor_from_engine_shards``. Two differences from
    the MXFP4 version, both structural (not just widths):

    * a shard row is ``H`` (gate/up) / ``I`` (down) **bytes**, not ``H/2``/``I/2``
      -- but the reassembly is byte-identical, and the K-major transpose is the
      same plain uint8 ``ktranspose_bytes``;
    * the block scale is ``[E, N/128, K/128]`` fp32 (groupN = groupK = 128),
      whereas MXFP4 is ``[E, N, K/32]`` e8m0. So the scale is read as fp32 and
      transposed to K-major with torch (it is ~1 MB/layer -- negligible).

    Returns ``(w13t, s13t, w2t, s2t)`` with the *weights* K-major and ready for
    the kernels, or ``None`` when the engine has no shards (caller falls back).
    """
    geo = engine.shard_geometry()
    ns = int(geo["ns"])
    if ns < 2 or not geo["w13_node_bytes"] or not geo["w2_node_bytes"]:
        return None
    H, I, E = int(hidden), int(inter), int(n_experts)
    b = FP8_BLOCK

    # Assemble the canonical [E, 2I, H] / [E, H, I] row layout; each node's gate
    # (resp. up) block is a contiguous cbytes run that maps to a contiguous row
    # range, so each is a plain copy_ (see the MXFP4 version's NOTES §466).
    w13_raw = _buf("raw13", (E, 2 * I, H), device)
    c13 = int(geo["w13_cbytes"])
    cr13 = int(geo["w13_crows"])
    for n in range(ns):
        buf = _dma_hostbuf(engine, 0, n, int(geo["w13_node_bytes"]), device)
        blk = buf.view(E, 2, c13)
        c0 = n * cr13
        w13_raw[:, c0:c0 + cr13, :].copy_(blk[:, 0, :].reshape(E, cr13, H))
        w13_raw[:, I + c0:I + c0 + cr13, :].copy_(blk[:, 1, :].reshape(E, cr13, H))
        del buf, blk

    w2_raw = _buf("raw2", (E, H, I), device)
    c2 = int(geo["w2_cbytes"])
    cr2 = int(geo["w2_crows"])
    for n in range(ns):
        buf = _dma_hostbuf(engine, 1, n, int(geo["w2_node_bytes"]), device)
        c0 = n * cr2
        w2_raw[:, c0:c0 + cr2, :].copy_(buf.view(E, cr2, I))
        del buf

    # Scales are a single full host copy (not sharded): [E, N/128, K/128] fp32.
    s13_raw = (_dma_hostbuf(engine, 2, 0, int(geo["w13_scale_bytes"]), device)
               .view(torch.float32).view(E, (2 * I) // b, H // b))
    s2_raw = (_dma_hostbuf(engine, 3, 0, int(geo["w2_scale_bytes"]), device)
              .view(torch.float32).view(E, H // b, I // b))
    s13t = s13_raw.transpose(1, 2).contiguous()      # [E, H/128, 2I/128]
    s2t = s2_raw.transpose(1, 2).contiguous()        # [E, I/128, H/128]

    if dst is not None:
        w13t = _kmajor_bytes(w13_raw, dst[0])
        w2t = _kmajor_bytes(w2_raw, dst[1])
    else:
        # Reuse the K-major targets too: fresh-per-layer allocation is the
        # documented ~365 ms/layer allocator churn (NOTES §466). One slot is
        # safe here because assembly and GEMM are strictly ordered on one stream.
        w13t = _kmajor_bytes(w13_raw, _buf("km13", (E, H, 2 * I), device))
        w2t = _kmajor_bytes(w2_raw, _buf("km2", (E, I, H), device))
    return w13t, s13t, w2t, s2t


def gpu_moe_layer_fp8(x, topk_ids, topk_weights, w13t, s13t, w2t, s2t,
                      H: int, I: int, K: int, *, device,
                      swiglu_limit: float = 0.0,
                      bm: int = 64, bn: int = 64, bk: int = 64, bh: int = 64,
                      ns: int = 2, warps: int = 4):
    """One layer's routed MoE on GPU from K-major FP8 weights.

    Returns an **fp32** tensor: accumulating the top_k partial sums into a bf16
    output (what the MXFP4 path does, since its `out` is bf16) costs ~2-4e-3 rms
    relative error; fp32 accumulation measures ~5e-7. The caller casts once.
    """
    T = x.shape[0]
    E = int(w13t.shape[0])
    tok, wts, seg_start, A = _build_segmentation(topk_ids, topk_weights, E, device)
    out = torch.zeros((T, H), dtype=torch.float32, device=device)
    if T == 0 or K == 0:
        return out
    inter = _buf("inter", (A, 2 * I), device, torch.bfloat16)
    gate_up_kernel_fp8[(E, triton.cdiv(2 * I, bn))](
        x, x.stride(0), tok, seg_start,
        w13t, w13t.stride(1), s13t, s13t.stride(1),
        inter, inter.stride(0), w13t.stride(0), s13t.stride(0),
        H=H, BM=bm, BN=bn, BK=bk, NS=ns, num_warps=warps,
    )
    down_kernel_fp8[(E, triton.cdiv(H, bh))](
        inter, inter.stride(0), tok, wts, seg_start,
        w2t, w2t.stride(1), s2t, s2t.stride(1),
        out, out.stride(0), w2t.stride(0), s2t.stride(0),
        H=H, I=I, BM=bm, BH=bh, BK=bk, NS=ns, LIMIT=float(swiglu_limit),
        num_warps=warps,
    )
    return out


def gpu_moe_layer_fp8_from_engine(x, topk_ids, topk_weights, engine,
                                  H: int, I: int, K: int, *, device,
                                  swiglu_limit: float = 0.0, n_experts: int,
                                  **kw):
    """Assemble this layer's K-major FP8 weights from the engine and run it."""
    built = kmajor_from_engine_shards_fp8(engine, device, H, I, n_experts)
    if built is None:
        return None
    return gpu_moe_layer_fp8(x, topk_ids, topk_weights, *built, H, I, K,
                             device=device, swiglu_limit=swiglu_limit, **kw)

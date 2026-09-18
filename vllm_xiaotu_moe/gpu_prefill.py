"""Long-prefill GPU MoE path for the vllm-xtu-moe hybrid plugin.

Design (mirrors ktransformers' ``KT_GPU_PREFILL_TOKEN_THRESHOLD`` and the fork's
``LVLLM_GPU_PREFILL_MIN_BATCH_SIZE``): when a layer's prefill token count is >=
``VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS``, compute that layer's routed MoE on the
GPU by streaming the layer's raw MXFP4 weights H2D per layer, then freeing them,
overlapping the next layer's H2D with the current layer's compute.

Why raw MXFP4 and not Marlin: ``marlin_utils_fp4`` only supports NVFP4
group-size-16, not MXFP4 block-size-32, so vLLM's Marlin MoE cannot be reused.
We dequantize fp4 (e2m1 nibbles) with per-(groupN=1, groupK=32) e8m0 block scales
in-kernel — the same MXFP4 math as the CPU engine (moe_v2_packed4.hpp).

Data layout (per layer, matching the checkpoint / CPU engine params):
  w13 [E, 2I, H//2] u8: low nibble = even k, high nibble = odd k; gate rows
      [0,I) then up rows [I,2I).  N=2I=4096, K=H=4096.
  s13 [E, 2I, H//32] u8 e8m0 (dequant 2^(byte-127)) per (n, k//32).
  w2  [E, H, I//2] u8 fp4 (N=H=4096, K=I=2048).
  s2  [E, H, I//32] u8 e8m0.
  out[t,:] = sum_s topk_weights[t,s] * f_{topk_ids[t,s]}(x[t,:]);
      f = W2^T * ( SiLU(x@W13_gate^T) .* (x@W13_up^T) ).

Per layer we stream the raw weights H2D (from cached pinned host copies => real
DMA), then transpose the fp4 bytes (and scales) once to K-major on the GPU via
``transpose().contiguous()`` (a cheap device copy at ~1 TB/s). The grouped GEMM
kernels then load ``b`` directly as ``[BK, BN]`` K-major, so no in-loop trans —
this removes the biggest in-kernel overhead. Only the raw ~3.4 GiB (bytes) is
in VRAM per layer.
"""

import os
import threading
import time

import torch

import triton
import triton.language as tl

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
# Same table as a Python tuple, for the in-kernel 16-entry LUT (see
# _dequant_e2m1_lut). Indices follow the e2m1 encoding: bit3 sign, bits2:1 exp,
# bit0 mantissa.
_E2M1_CONST = tuple(_E2M1.tolist())
_LUT_CACHE: dict = {}


def _e2m1_table(device) -> torch.Tensor:
    """16-entry fp32 e2m1 table on ``device`` (for the gather-based dequant)."""
    t = _LUT_CACHE.get(device)
    if t is None:
        t = _E2M1.to(device).contiguous()
        _LUT_CACHE[device] = t
    return t


def gpu_prefill_min_tokens() -> int:
    """Current threshold.

    ``XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE`` (if set) wins over the env var: with
    TP>1 the model runs in worker processes that do not see the client's
    ``os.environ`` changes, so a file is the only way to switch the threshold at
    runtime (used by the measurement harness).
    """
    # 注意:变量名必须同时接受带/不带 VLLM_ 前缀两种写法 —— `scripts/tune_serve.sh`
    # 导出的是 VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE,而这里原先只读
    # XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE ⇒ **阈值文件从未生效**,导致所有"运行时切换
    # 阈值"的 A/B(2026-09-11 预填充对比)实际上都走了同一条 GPU 路径。
    f = os.environ.get("XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE") or \
        os.environ.get("VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE")
    if f:
        try:
            with open(f) as fh:
                return int(fh.read().strip())
        except (OSError, ValueError):
            pass
    v = os.environ.get("VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS", "0")
    try:
        return int(v)
    except ValueError:
        return 0


@triton.jit
def _dequant_e2m1_nibble(nib):
    nib = nib.to(tl.int32)
    sign = (nib >> 3) & 1
    ex = (nib >> 1) & 3
    man = nib & 1
    val_n = tl.math.exp2(ex.to(tl.float32) - 1.0) * (1.0 + 0.5 * man.to(tl.float32))
    val_s = 0.5 * man.to(tl.float32)
    val = tl.where(ex == 0, val_s, val_n)
    return tl.where(sign == 1, -val, val)


@triton.jit
def _dequant_e2m1_gather(nib, lut_ptr):
    """e2m1 dequant via a 16-entry fp32 table in global memory (L1-resident).

    Replaces ~10 int/float ops per element with one 4-byte gather from a 64-byte
    table; only worth it if the dequant ALU work is the kernel's bottleneck.
    """
    return tl.load(lut_ptr + nib.to(tl.int32))


@triton.jit
def _dequant_e2m1_lut(nib):
    """Same e2m1 dequant as ``_dequant_e2m1_nibble`` via a 16-entry register LUT.

    The nibble is a 4-bit index into {0,±0.5,±1,±1.5,±2,±3,±4,±6}; comparing the
    index against every table entry and selecting is cheaper than the arithmetic
    form (exp2 + several int ops) on SM80.
    """
    nib = nib.to(tl.int32)
    t0 = tl.zeros_like(nib).to(tl.float32)
    for i in tl.static_range(16):
        t0 = tl.where(nib == i, _E2M1_CONST[i], t0)
    return t0


@triton.jit
def _gate_up_kernel_split(
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
    LUT: tl.constexpr,
):
    """gate_up without the per-k-block join+trans+reshape.

    A packed byte holds k=2j (low nibble) and k=2j+1 (high nibble). Instead of
    interleaving them back into a [BK, BN] tile with a register shuffle, we split
    the *activation* tile into its even/odd k columns and do two dots. The scale
    (block 32 along k) is identical for both halves after a 16x repeat, so no
    extra loads are needed.
    """
    pid_e = tl.program_id(0)
    pid_n = tl.program_id(1)
    base = tl.load(seg_ptr + pid_e)
    end = tl.load(seg_ptr + pid_e + 1)
    if end <= base:
        return
    n = pid_n * BN + tl.arange(0, BN)
    # int64: the expert base offset is pid_e * stride(0), and for V4.1 the
    # K-major w13/w2 tensors exceed 2^31 bytes (4.53 GB / 2.26 GB), so
    # `pid_e * W13_E` OVERFLOWS int32. For w2 that wraps negative and faults
    # (illegal memory access); for w13 it wraps POSITIVE (383*11796480 =
    # 4,518,051,840 -> 223,084,544), so it silently reads the WRONG expert.
    # Triton types a Python int arg by its value, so the cast must be here.
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
            a_e, a_o = tl.split(tl.reshape(a, (BM, BK // 2, 2)))
            b_lo = tl.load(
                w13t_ptr + w_off + ((kk >> 1) + tl.arange(0, BK // 2))[:, None] * w13t_row_stride
                + n[None, :])
            if LUT:
                w_e = _dequant_e2m1_lut(b_lo & 0x0F)
                w_o = _dequant_e2m1_lut((b_lo >> 4) & 0x0F)
            else:
                w_e = _dequant_e2m1_nibble(b_lo & 0x0F)
                w_o = _dequant_e2m1_nibble((b_lo >> 4) & 0x0F)
            scale = tl.load(
                s13t_ptr + s_off + ((kk >> 5) + tl.arange(0, BK // 32))[:, None] * s13t_row_stride
                + n[None, :])
            sc = tl.exp2(scale.to(tl.float32) - 127.0)
            # k = 2j+e shares the block-32 scale with k = 2j, and j>>4 selects
            # the block, so the even and odd halves use the SAME [BK//2, BN]
            # scale tile (each block-32 row repeats 16 times).
            sc_h = tl.reshape(
                tl.broadcast_to(sc[:, None, :], (BK // 32, 16, BN)), (BK // 2, BN))
            b_e = (w_e * sc_h).to(tl.bfloat16)
            b_o = (w_o * sc_h).to(tl.bfloat16)
            acc = tl.dot(a_e, b_e, acc)
            acc = tl.dot(a_o, b_o, acc)
        tl.store(
            inter_ptr + g_rows[:, None] * inter_ld + n[None, :],
            acc.to(tl.bfloat16), mask=rmask[:, None])
        mt += 1


@triton.jit
def _gate_up_kernel(
    x_ptr, x_stride, tok_ptr,
    seg_ptr,
    w13t_ptr, w13t_row_stride, s13t_ptr, s13t_row_stride,  # [E, H//2, 2I] / [E, H//32, 2I]
    inter_ptr, inter_ld,
    W13_E, S13_E, lut_ptr,
    H: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NS: tl.constexpr,
    LUT: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_n = tl.program_id(1)
    base = tl.load(seg_ptr + pid_e)
    end = tl.load(seg_ptr + pid_e + 1)
    if end <= base:
        return
    n = pid_n * BN + tl.arange(0, BN)
    # int64: the expert base offset is pid_e * stride(0), and for V4.1 the
    # K-major w13/w2 tensors exceed 2^31 bytes (4.53 GB / 2.26 GB), so
    # `pid_e * W13_E` OVERFLOWS int32. For w2 that wraps negative and faults
    # (illegal memory access); for w13 it wraps POSITIVE (383*11796480 =
    # 4,518,051,840 -> 223,084,544), so it silently reads the WRONG expert.
    # Triton types a Python int arg by its value, so the cast must be here.
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
            b_lo = tl.load(
                w13t_ptr + w_off + ((kk >> 1) + tl.arange(0, BK // 2))[:, None] * w13t_row_stride
                + n[None, :])                                        # [BK//2, BN]
            nib_even = b_lo & 0x0F
            nib_odd = (b_lo >> 4) & 0x0F
            if LUT:
                w_e = _dequant_e2m1_gather(nib_even, lut_ptr)
                w_o = _dequant_e2m1_gather(nib_odd, lut_ptr)
            else:
                w_e = _dequant_e2m1_nibble(nib_even)
                w_o = _dequant_e2m1_nibble(nib_odd)
            b_val = tl.reshape(tl.trans(tl.join(w_e, w_o)), (BK, BN)) # [BK, BN]
            scale = tl.load(
                s13t_ptr + s_off + ((kk >> 5) + tl.arange(0, BK // 32))[:, None] * s13t_row_stride
                + n[None, :])                                        # [BK//32, BN]
            sc = tl.exp2(scale.to(tl.float32) - 127.0)
            sc = tl.reshape(tl.broadcast_to(sc[:, None, :], (BK // 32, 32, BN)), (BK, BN))
            b = (b_val * sc).to(tl.bfloat16)                         # [BK, BN]
            acc = tl.dot(a, b, acc)
        tl.store(
            inter_ptr + g_rows[:, None] * inter_ld + n[None, :],
            acc.to(tl.bfloat16), mask=rmask[:, None])
        mt += 1


@triton.jit
def _down_kernel(
    inter_ptr, inter_ld, tok_ptr, wts_ptr,
    seg_ptr,
    w2t_ptr, w2t_row_stride, s2t_ptr, s2t_row_stride,  # [E, I//2, H] / [E, I//32, H]
    out_ptr, out_ld,
    W2_E, S2_E, lut_ptr,
    H: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    NS: tl.constexpr,
    LUT: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_h = tl.program_id(1)
    base = tl.load(seg_ptr + pid_e)
    end = tl.load(seg_ptr + pid_e + 1)
    if end <= base:
        return
    h = pid_h * BH + tl.arange(0, BH)
    # int64: see _gate_up_kernel -- V4.1's w2 is 2.26 GB, so pid_e * W2_E
    # overflows int32 and faults at high expert ids.
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
            sig = 1.0 / (1.0 + tl.math.exp(-gate))
            act = (gate * sig) * up
            b_lo = tl.load(
                w2t_ptr + w_off + ((kk >> 1) + tl.arange(0, BK // 2))[:, None] * w2t_row_stride
                + h[None, :])                                        # [BK//2, BH]
            nib_even = b_lo & 0x0F
            nib_odd = (b_lo >> 4) & 0x0F
            if LUT:
                w_e = _dequant_e2m1_gather(nib_even, lut_ptr)
                w_o = _dequant_e2m1_gather(nib_odd, lut_ptr)
            else:
                w_e = _dequant_e2m1_nibble(nib_even)
                w_o = _dequant_e2m1_nibble(nib_odd)
            b_val = tl.reshape(tl.trans(tl.join(w_e, w_o)), (BK, BH)) # [BK, BH]
            scale = tl.load(
                s2t_ptr + s_off + ((kk >> 5) + tl.arange(0, BK // 32))[:, None] * s2t_row_stride
                + h[None, :])                                        # [BK//32, BH]
            sc = tl.exp2(scale.to(tl.float32) - 127.0)
            sc = tl.reshape(tl.broadcast_to(sc[:, None, :], (BK // 32, 32, BH)), (BK, BH))
            b = (b_val * sc).to(tl.bfloat16)                         # [BK, BH]
            acc = tl.dot(act.to(tl.bfloat16), b, acc)
        # sem="relaxed": the accumulations into ``out`` are the only accesses to
        # that buffer in this kernel and the caller reads it after the launch,
        # so we do not need acq_rel ordering. Measured 198.2 -> 189.8 ms/layer
        # (T=16K); a full "no atomic" restructure (per-slot buffer + reduction)
        # would reach 177.4 ms but is a larger change.
        tl.atomic_add(
            out_ptr + toks[:, None] * out_ld + h[None, :],
            acc * wts[:, None], mask=rmask[:, None], sem="relaxed")
        mt += 1


@triton.jit
def _down_kernel_split(
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
    LUT: tl.constexpr,
):
    """down projection, even/odd-k split (see _gate_up_kernel_split)."""
    pid_e = tl.program_id(0)
    pid_h = tl.program_id(1)
    base = tl.load(seg_ptr + pid_e)
    end = tl.load(seg_ptr + pid_e + 1)
    if end <= base:
        return
    h = pid_h * BH + tl.arange(0, BH)
    # int64: see _gate_up_kernel -- V4.1's w2 is 2.26 GB, so pid_e * W2_E
    # overflows int32 and faults at high expert ids.
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
            sig = 1.0 / (1.0 + tl.math.exp(-gate))
            act = (gate * sig) * up
            act_e, act_o = tl.split(tl.reshape(act, (BM, BK // 2, 2)))
            b_lo = tl.load(
                w2t_ptr + w_off + ((kk >> 1) + tl.arange(0, BK // 2))[:, None] * w2t_row_stride
                + h[None, :])
            if LUT:
                w_e = _dequant_e2m1_lut(b_lo & 0x0F)
                w_o = _dequant_e2m1_lut((b_lo >> 4) & 0x0F)
            else:
                w_e = _dequant_e2m1_nibble(b_lo & 0x0F)
                w_o = _dequant_e2m1_nibble((b_lo >> 4) & 0x0F)
            scale = tl.load(
                s2t_ptr + s_off + ((kk >> 5) + tl.arange(0, BK // 32))[:, None] * s2t_row_stride
                + h[None, :])
            sc = tl.exp2(scale.to(tl.float32) - 127.0)
            sc_h = tl.reshape(
                tl.broadcast_to(sc[:, None, :], (BK // 32, 16, BH)), (BK // 2, BH))
            b_e = (w_e * sc_h).to(tl.bfloat16)
            b_o = (w_o * sc_h).to(tl.bfloat16)
            acc = tl.dot(act_e.to(tl.bfloat16), b_e, acc)
            acc = tl.dot(act_o.to(tl.bfloat16), b_o, acc)
        # sem="relaxed": the accumulations into ``out`` are the only accesses to
        # that buffer in this kernel and the caller reads it after the launch,
        # so we do not need acq_rel ordering. Measured 198.2 -> 189.8 ms/layer
        # (T=16K); a full "no atomic" restructure (per-slot buffer + reduction)
        # would reach 177.4 ms but is a larger change.
        tl.atomic_add(
            out_ptr + toks[:, None] * out_ld + h[None, :],
            acc * wts[:, None], mask=rmask[:, None], sem="relaxed")
        mt += 1


# 【§598】按形状复用 GPU MoE 的**中间缓冲**(`inter`)与输出(`out`)。
# 背景:每层新分配 inter=(T*K, 2I)(qlen=13816 时 382 MB)会让 `alloc/reserved`
# 逐层上涨 ~0.42 GiB,40 层就是 ~17 GiB —— 直接把 free 压到预填充阈值之下。
# `inter` 是 `torch.empty` 语义(只读被写过的段)⇒ 复用完全等价。
_BUF_MOE: dict = {}


def _reuse_moe(key, shape, dtype, device):
    k = (str(device), key, tuple(int(v) for v in shape), dtype)
    t = _BUF_MOE.get(k)
    if t is None or tuple(t.shape) != tuple(int(v) for v in shape):
        t = torch.empty(tuple(int(v) for v in shape), dtype=dtype, device=device)
        _BUF_MOE[k] = t
    return t


def _build_segmentation(topk_ids, topk_weights, num_experts, device):
    """把 [T,K] 路由压成按专家分段的排序列表(供两个 Triton 内核按段遍历)。

    **必须是形状静态的**:CUDA graph 捕获期间不允许出现数据相关的形状
    (`ids[ok]` 这种布尔掩码索引会触发 `cudaErrorStreamCaptureUnsupported`,
    见 report/tuning/TRIED_AND_REVERTED.md R14)。所以无效/不属于本 rank 的
    id 不删除,而是**归入垃圾桶专家桶 `num_experts`**(它排在最后,内核 grid=E
    不会读它)⇒ 与"过滤掉"完全等价。

    返回的 ``A`` 固定为 ``T*K``(内核只按段边界取行,``A`` 只用来定缓冲大小;
    注意 ``inter`` 是按 **token id** 索引的,所以行数本来就要 ≥ T)。
    """
    T, K = topk_ids.shape
    ids = topk_ids.to(torch.int32).reshape(-1)
    wts = topk_weights.to(torch.float32).reshape(-1)
    tok_ids = torch.arange(T, device=device).repeat_interleave(K)
    # 垃圾桶桶:任何 <0 / ≥E 的 id 都映射到 E(固定形状,无掩码索引)
    bad = (ids < 0) | (ids >= num_experts)
    ids_c = torch.where(
        bad, torch.full((), num_experts, dtype=torch.int32, device=device), ids)
    order = torch.argsort(ids_c, stable=True)
    sorted_tok = tok_ids[order]
    sorted_wts = wts[order]
    # scatter_add 而不是 bincount:bincount 会从数据里求 max 来决定输出长度(可能同步);
    # scatter_add 的输出长度固定为 E+1。取前 E 个桶(垃圾桶段排在最后,内核不读)。
    ones = torch.ones_like(ids_c, dtype=torch.int32)
    counts = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    counts.scatter_add_(0, ids_c.long(), ones)
    counts = counts[:num_experts]
    seg_start = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(counts, dim=0)]
    )
    return sorted_tok, sorted_wts, seg_start, T * K


_GP_TIMING = os.environ.get("XIAOTU_GP_TIMING") == "1"
_GP_H2D = {"n": 0, "ms": 0.0}


def _gp_h2d(ms: float) -> None:
    st = _GP_H2D
    st["n"] += 1
    st["ms"] += ms
    if st["n"] % 40 == 0:
        print(f"[gp-h2d] n={st['n']} per_layer={st['ms'] / st['n']:.1f}ms", flush=True)
        st.update(n=0, ms=0.0)


_GP_STAT = {"n": 0, "seg": 0.0, "rest": 0.0, "tok": 0}
_GP_EVERY = 40


def _gp_add(T: int, A: int, seg_s: float, rest_s: float) -> None:
    """GPU 预填充路径的**主机侧**分段计时(env XIAOTU_GP_TIMING=1)。

    seg  = _build_segmentation(argsort/scatter_add/arange 等)耗时
    rest = 其后到内核 launch 之间的主机耗时(out/inter 分配、zeros 等)
    两者都是**串在关键路径上的主机时间**,用来定位"服务里每层多出的 ~31ms"。
    """
    st = _GP_STAT
    st["n"] += 1
    st["seg"] += seg_s
    st["rest"] += rest_s
    st["tok"] += T
    if st["n"] % _GP_EVERY == 0:
        n = st["n"]
        print(f"[gp-timing] n={n} Tavg={st['tok'] / n:.0f} "
              f"seg={st['seg'] / n * 1e3:.2f}ms rest={st['rest'] / n * 1e3:.2f}ms "
              f"host_total={(st['seg'] + st['rest']) / n * 1e3:.2f}ms", flush=True)
        st.update(n=0, seg=0.0, rest=0.0, tok=0)


_CAPTURE_DEPTH = 0
_CAPTURE_GUARD = threading.Lock()
_CAPTURE_WATCH_INSTALLED = False


def capture_in_progress() -> bool:
    """True while a CUDA graph capture is running in this process."""
    return _CAPTURE_DEPTH > 0


def wait_no_capture(timeout_s: float = 120.0) -> None:
    """Block until no CUDA graph capture is in flight (best effort, bounded).

    Used by the background pinned prebuild: any CUDA call it makes
    (cudaHostRegister / pin_memory) during capture both fails and **invalidates
    the capture**, which aborts engine startup.
    """
    import time as _time

    end = _time.monotonic() + timeout_s
    while capture_in_progress() and _time.monotonic() < end:
        _time.sleep(0.05)


def wait_capture_done(quiet_s: float = 10.0, grace_s: float = 300.0,
                      poll_s: float = 0.1) -> bool:
    """Wait until CUDA-graph capture is finished **and has stayed quiet**.

    Rationale (2026-09-11, three failed startups): vLLM builds its graphs ~100 s
    after the first big warm-up forward — which is exactly the forward that
    triggers the pinned prebuild. A `cudaHostRegister` already inside the driver
    when capture begins cannot be recalled, and it both invalidates the capture
    (`cudaErrorStreamCaptureInvalidated` -> EngineCore init fails) and has been
    seen to segfault inside libcuda (`cuMemHostRegister_v2` -> SIGSEGV).

    Timeline measured on this box (deliver13 log): warm-up forward -> +100 s ->
    "Breakable CUDA graph enabled" -> main capture -> DSpark speculator capture.
    A fixed grace period therefore cannot work; we wait for the capture state to
    be *quiet* (no capture for `quiet_s`), and if this process never captures
    (``EAGER=1``) we give up after `grace_s` and proceed.
    """
    import time as _time

    t0 = _time.monotonic()
    last_end = 0.0
    seen = False
    while _time.monotonic() - t0 < grace_s:
        if capture_in_progress():
            seen = True
            while capture_in_progress() and _time.monotonic() - t0 < grace_s + 600.0:
                _time.sleep(poll_s)
            last_end = _time.monotonic()
            continue
        if seen and (_time.monotonic() - last_end) >= quiet_s:
            return True
        _time.sleep(poll_s)
    return not capture_in_progress()


def _install_capture_watch() -> None:
    """Count CUDA graph captures process-wide by wrapping CUDAGraph begin/end.

    There is no CUDA API to ask "is any stream in this context capturing?", and
    the capturing stream lives on another thread, so we track it ourselves. The
    wrapper is installed once and is a no-op otherwise.
    """
    global _CAPTURE_WATCH_INSTALLED
    if _CAPTURE_WATCH_INSTALLED:
        return
    try:
        import torch
        cls = torch.cuda.CUDAGraph
        if getattr(cls, "_xiaotu_capture_watched", False):
            _CAPTURE_WATCH_INSTALLED = True
            return
        _begin, _end = cls.capture_begin, cls.capture_end

        def capture_begin(self, *a, **k):
            global _CAPTURE_DEPTH
            with _CAPTURE_GUARD:
                _CAPTURE_DEPTH += 1
            try:
                return _begin(self, *a, **k)
            except BaseException:
                with _CAPTURE_GUARD:
                    _CAPTURE_DEPTH = max(0, _CAPTURE_DEPTH - 1)
                raise

        def capture_end(self, *a, **k):
            global _CAPTURE_DEPTH
            try:
                return _end(self, *a, **k)
            finally:
                with _CAPTURE_GUARD:
                    _CAPTURE_DEPTH = max(0, _CAPTURE_DEPTH - 1)

        cls.capture_begin = capture_begin
        cls.capture_end = capture_end
        cls._xiaotu_capture_watched = True
        _CAPTURE_WATCH_INSTALLED = True
    except Exception:  # noqa: BLE001 - torch/driver variants: degrade to no-op
        pass


_install_capture_watch()


def _register_host(t: torch.Tensor) -> bool:
    """Page-lock ``t`` **in place** (no copy) via cudaHostRegister.

    ``Tensor.pin_memory()`` allocates a new page-locked buffer and copies into it
    at ~1.6 GB/s (measured); cudaHostRegister locks the pages we already have at
    ~5.8 GB/s (measured) and keeps the same H2D throughput, so the one-time
    K-major cache build drops from ~114 s to ~45 s (and with the parallel
    prebuild below, to ~15-25 s).
    """
    if t.is_pinned():
        return True
    try:
        rt = torch.cuda.cudart()
        err = rt.cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)
        return int(err) == 0 and t.is_pinned()
    except Exception:  # noqa: BLE001 - driver/cudart missing
        return False


_PIN_LOCKS: dict[tuple, threading.Lock] = {}
_PIN_LOCKS_GUARD = threading.Lock()


def _key_lock(key: tuple) -> threading.Lock:
    with _PIN_LOCKS_GUARD:
        lk = _PIN_LOCKS.get(key)
        if lk is None:
            lk = threading.Lock()
            _PIN_LOCKS[key] = lk
        return lk


def _pin_key(t: torch.Tensor, tag: str):
    """缓存键 + 源 storage。

    键用**源 storage 地址**(对同一参数的切片是稳定的,如 ``param.data[a:b]``);
    同时持有源 storage 的引用,避免地址被无关分配复用后命中过期条目。
    """
    stor = t.untyped_storage()
    return (stor.data_ptr(), t.storage_offset(), tuple(t.shape), t.dtype, tag), stor


def _ensure_pinned(key: tuple, ent: tuple) -> torch.Tensor:
    """确保缓存项已锁页(**就地** cudaHostRegister,失败才退回 pin_memory)。"""
    c = ent[1]
    if c.is_pinned():
        return c
    if not _register_host(c):
        c = c.pin_memory()
        ent = (ent[0], c)
        _PIN_CACHE[key] = ent
    return c


def _pinned(t: torch.Tensor) -> torch.Tensor:
    key, stor = _pin_key(t, "plain")
    ent = _PIN_CACHE.get(key)
    if ent is not None:
        return _ensure_pinned(key, ent)
    with _key_lock(key):
        ent = _PIN_CACHE.get(key)
        if ent is None:
            ent = (stor, t.contiguous())
            _PIN_CACHE[key] = ent
        return _ensure_pinned(key, ent)


_PIN_CACHE: dict[tuple, tuple] = {}


def _kmajor_bytes(t, dst=None):
    """[E, A, B] u8 -> [E, B, A] u8(字节转置;nibble 仍留在字节内)。`dst` 给了就写进去。

    【§596】**默认改用分块 Triton kernel**:实测 `transpose(1,2).contiguous()`
    对这种 uint8 逐字节转置只有 **~90 GB/s**(4 个张量合计 42.8 ms/层),
    而 128×128 分块版有 **~600 GB/s**(合计 6.5 ms/层),快 5-7× 且逐位相同。
    `XIAOTU_GPF_TT=0` 可退回旧实现(出问题时的一键回滚)。
    """
    try:
        from vllm_xiaotu_moe.byte_transpose import ktranspose_bytes
        return ktranspose_bytes(t, dst)
    except Exception:  # noqa: BLE001
        r = t.transpose(1, 2).contiguous()
        if dst is not None:
            dst.copy_(r)
            return dst
        return r


def _kmajor_cached(t: torch.Tensor):
    """纯 CPU 阶段:做 K-major 转置并放入缓存(**不锁页**)。

    转置是普通 CPU 拷贝,任何时刻都安全(包括 CUDA graph 捕获期间);只有
    ``cudaHostRegister`` 不能在捕获窗口里调用(见 TRIED_AND_REVERTED R4)。
    把两件事拆开,后台就能在 warmup 期间先把转置做完。
    """
    key, stor = _pin_key(t, "kmajor")
    ent = _PIN_CACHE.get(key)
    if ent is not None:
        return key, ent
    with _key_lock(key):
        ent = _PIN_CACHE.get(key)
        if ent is None:
            ent = (stor, _kmajor_bytes(t))
            _PIN_CACHE[key] = ent
    return key, ent


def _pinned_kmajor(t: torch.Tensor) -> torch.Tensor:
    """Pin the K-major view of ``t`` (host-side transpose, cached).

    Doing the byte transpose once on the host (at cache-build time) instead of
    once per layer on the GPU removes a ~3.4 GiB device read+write per layer and
    lets the H2D land directly in the layout the kernels index.

    The cache is keyed on ``t``'s **own** storage (a long-lived model param), not
    on the transposed temporary: keying on the temporary would miss on every
    call and re-pin 3.19 GiB per layer (~2.6 s) while also leaking pinned
    entries until the host OOMs.
    """
    key, ent = _kmajor_cached(t)
    return _ensure_pinned(key, ent)


# --------------------------------------------------------------------------
# 显存预算与"优雅放弃"(用户 2026-09-15 指示)
#
# 逐层流式要在**设备上**再放一份当前层的权重(raw + K-major,约 2x 单层专家大小;
# V4.1@TP=1 单层 6.72 GiB ⇒ 峰值 ~14 GiB)。这块内存与 vLLM 的 KV cache 竞争,
# 而 vLLM 是在 **profile run** 之后按"峰值显存"给 KV 定容的。若不挡住,
# profile 期间走 GPU 路径会把峰值抬高,直接得到
#   `Available KV cache memory: -1.57 GiB` ⇒ **整个服务起不来**。
#
# 所以两层保护:
#   1. profile run 期间**绝不**走 GPU 路径(否则定容就崩);
#   2. 运行时先做一次**预检**:腾不出 staging 就留在 CPU,并打印一次明确提示 ——
#      "慢但能跑"优于"起不来"。用户的选择:TP>=2(每 rank staging 减半)、
#      换更大显存的卡、或调低 --gpu-memory-utilization 给它留地方。
# --------------------------------------------------------------------------
_IN_PROFILE_RUN = False
# 更宽的窗口:从进程起来到 KV cache 初始化完成之前都算 **startup**
# (profile run / CUDA graph 捕获 / warmup 都落在里面)。实测只挡 `profile_run`
# **不够** —— 峰值显存的测量发生在它外面,`Available KV cache memory` 仍然变成
# 负数(NOTES §460.1)。用 "_initialize_kv_caches 返回" 作为 startup 结束的锚点,
# 它与"KV cache 已定容"是同一件事,比去找具体的 profile 调用稳。
_IN_STARTUP = [True]


def in_profile_run() -> bool:
    """True during startup (profiling/capture/warmup) -- GPU prefill must not run."""
    return _IN_PROFILE_RUN or _IN_STARTUP[0]


def install_profile_guard() -> list[str]:
    """Make ``GPUModelRunner.profile_run`` visible to us as a flag."""
    global _IN_PROFILE_RUN
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception:  # noqa: BLE001
        return []
    orig = getattr(GPUModelRunner, "profile_run", None)
    if orig is None or getattr(orig, "_xtu_shim", False):
        return []

    import functools

    @functools.wraps(orig)
    def profile_run(self, *a, **kw):
        global _IN_PROFILE_RUN
        _IN_PROFILE_RUN = True
        try:
            return orig(self, *a, **kw)
        finally:
            _IN_PROFILE_RUN = False

    profile_run._xtu_shim = True  # type: ignore[attr-defined]
    GPUModelRunner.profile_run = profile_run
    applied = ["GPUModelRunner.profile_run"]

    # 关键的一半:KV cache 定容完成 = startup 结束。
    try:
        from vllm.v1.engine.core import EngineCore
    except Exception:  # noqa: BLE001
        return applied
    orig_kv = getattr(EngineCore, "_initialize_kv_caches", None)
    if orig_kv is None or getattr(orig_kv, "_xtu_shim", False):
        return applied

    @functools.wraps(orig_kv)
    def _initialize_kv_caches(self, *a, **kw):
        try:
            return orig_kv(self, *a, **kw)
        finally:
            _IN_STARTUP[0] = False
            print("[vllm-xtu-moe] startup finished (KV cache sized) -> "
                  "GPU prefill is now allowed subject to the VRAM preflight",
                  flush=True)

    _initialize_kv_caches._xtu_shim = True  # type: ignore[attr-defined]
    EngineCore._initialize_kv_caches = _initialize_kv_caches
    applied.append("EngineCore._initialize_kv_caches")

    # 【§589 修】**必须同时给每个 worker 进程装一个锚点!**
    # `_IN_STARTUP` 是**进程级**标志,而门控 `_gpu_pf` 是在 **worker 进程**里求值的;
    # `EngineCore._initialize_kv_caches` 只在 EngineCore 进程执行(实测日志里那行
    # "startup finished" 只有 EngineCore 打过)⇒ **worker 里 `_IN_STARTUP[0]` 永远是 True**
    # ⇒ `in_profile_run()` 恒真 ⇒ **GPU 预填充从来没真正生效过**(4K 输入 TTFT 15.6 s,
    # 40 层 × 382 ms 全是 CPU MoE)。正确的 per-worker 锚点是
    # `Worker.compile_or_warm_up_model`:它在 KV 定容之后、图捕获完成时返回,正好是
    # "该 worker 的 startup 结束"(且期间保持禁用,才能不污染 profile 的显存测量与捕获)。
    try:
        from vllm.v1.worker.gpu_worker import Worker
    except Exception:  # noqa: BLE001
        return applied
    orig_warm = getattr(Worker, "compile_or_warm_up_model", None)
    if orig_warm is not None and not getattr(orig_warm, "_xtu_shim", False):
        @functools.wraps(orig_warm)
        def compile_or_warm_up_model(self, *a, **kw):
            try:
                return orig_warm(self, *a, **kw)
            finally:
                if _IN_STARTUP[0]:
                    _IN_STARTUP[0] = False
                    print("[vllm-xtu-moe] worker startup finished (KV sized + graphs "
                          "captured) -> GPU prefill allowed subject to VRAM preflight",
                          flush=True)

        compile_or_warm_up_model._xtu_shim = True  # type: ignore[attr-defined]
        Worker.compile_or_warm_up_model = compile_or_warm_up_model
        applied.append("Worker.compile_or_warm_up_model(per-worker startup anchor)")
    return applied


def staging_bytes(n_experts: int, hidden: int, inter: int, group_k: int = 32,
                  ns: int = 2) -> int:
    """Peak device bytes one layer's streaming path needs.

    `kmajor_from_engine_shards` now fills the K-major target directly and finishes
    w13 (including freeing its shard buffers) before allocating w2, so the peak is
        max(w13_kmajor + w13_node_shard,
            w13_kmajor + w2_kmajor + w2_node_shard) + 2 * scales
    rather than the old raw+K-major double (which was ~2x dense = 13.4 GiB).
    """
    E, H, I = int(n_experts), int(hidden), int(inter)
    gk = int(group_k) if int(group_k) > 0 else 1
    n = max(1, int(ns))
    w13d = E * (2 * I) * (H // 2)
    w2d = E * H * (I // 2)
    s13d = E * (2 * I) * (H // gk)
    s2d = E * H * (I // gk)
    # 【§600 修口径】不是只有 ping/pong 两槽!一次 staging 的**真实峰值**是:
    #   槽 2×(w13+w2+s13+s2) + raw(w13+w2,组装目标) + DMA 暂存(N 张量各一份)
    # 原式只算第一项(6.72 GiB),于是 preflight 说 8.4 GiB 够、实际要 ~11 GiB ⇒
    # "预检通过但跑起来 OOM"。这里把三项都算上(暂存已改为每张量一份)。
    slots = 2 * (w13d + w2d + s13d + s2d)
    raw = w13d + w2d
    # 暂存现在**每张量一份**(见 `_dma_hostbuf`),所以是"一个 node 的量"而不是全部 node 之和
    tmp = (w13d + w2d + s13d + s2d) / n
    return slots + raw + tmp


def fits_device(need_bytes: int, device, margin: float = 1.10):
    """(ok, free_bytes). 把 staging 的**真实峰值**与**空闲**显存比。

    【§600】margin 从 1.25 收到 **1.10**:`need_bytes` 现在是诚实口径
    (槽 + raw + 单份暂存 = ~10.8 GiB,原来只有槽的 6.72),再乘 1.25 会把
    "实测能跑"的配置(如 free=12.95)误判为不够。1.10 覆盖碎片/激活抖动。
    """
    try:
        free, _total = torch.cuda.mem_get_info(torch.device(device))
    except Exception:  # noqa: BLE001
        return True, -1        # cannot tell -> let it try (still guarded by try/except)
    return (free >= int(need_bytes * margin)), int(free)


# 复用中间缓冲:实测一次性函数 789 ms/层,而各段相加只有 424 ms —— 差额 ~365 ms
# 全是**每层重新分配**那些中间张量(w13/w2 的 node 缓冲 + raw + K-major,约 13 GB)
# 造成的 allocator churn(NOTES §466)。这些缓冲在同一 stream 上是安全可复用的:
# 下一层的 copy_ 排在前一层 kernel 之后,stream 会保证顺序。
_BUF_CACHE: dict = {}


def _reuse(key, shape, device):
    """Get a reusable uint8 buffer of `shape` (allocates once per backend)."""
    k = (str(device), int(shape[0]) if len(shape) == 1 else -1, tuple(shape))
    t = _BUF_CACHE.get(k)
    if t is None or t.shape != tuple(shape) or t.device != torch.device(device):
        t = torch.empty(tuple(shape), dtype=torch.uint8, device=device)
        _BUF_CACHE[k] = t
    return t


def _dma_hostbuf(engine, which: int, node: int, nbytes: int, device) -> torch.Tensor:
    """DMA one of the ENGINE's own host buffers to a fresh device tensor."""
    if nbytes <= 0:
        raise RuntimeError(f"gpu-prefill: engine buffer which={which} node={node} is empty")
    # 【§600】暂存缓冲**每张量一份**(不再每 node 一份)。同一条 stream 上
    # "DMA n → copy n → DMA n+1" 是严格有序的 ⇒ 复用安全;而每 node 一份会让
    # ns 份暂存同时常驻(TP=2 共 3.59 GiB),把真正的 staging 峰值从 ~11 GiB 抬到 ~14 GiB。
    t = _reuse(("dma", int(which)), (int(nbytes),), device)
    stream = torch.cuda.current_stream(device).cuda_stream
    got = engine.copy_hostbuf_to_device(int(which), int(node), t.data_ptr(), stream)
    if int(got) != int(nbytes):
        raise RuntimeError(
            f"gpu-prefill: shard DMA failed (which={which} node={node}: "
            f"copied {got}/{nbytes} bytes)"
        )
    return t


_STAGE_ACC = {"n": 0, "dma": 0.0, "asm": 0.0, "tr": 0.0}


def _stage_mark(t0: float, key: str) -> float:
    """[§594 诊断] 把 `kmajor_from_engine_shards` 的三个子相累加起来。

    `XIAOTU_GPF_STAGE=1` 时,每 40 次(一层一次)打印一次平均,用来回答
    "asm 的 220~283 ms/层 里,DMA / 跨步组装 / K-major 转置 各占多少"。
    **每个子相后面都要 sync**,否则测到的是"发射耗时"而不是"完成耗时";
    诊断模式下多 2 次 sync 可以接受(它只在 env 打开时生效)。
    """
    import time as _t
    import torch as _torch
    if os.environ.get("XIAOTU_GPF_STAGE") != "1":
        return t0
    _torch.cuda.synchronize()
    now = _t.perf_counter()
    _STAGE_ACC[key] += (now - t0) * 1e3
    _STAGE_ACC["n"] += 1 if key == "tr" else 0
    if key == "tr" and _STAGE_ACC["n"] % 40 == 0:
        n = _STAGE_ACC["n"]
        d, a, r = _STAGE_ACC["dma"], _STAGE_ACC["asm"], _STAGE_ACC["tr"]
        print(f"[gpf-stage] n={n} dma={d/40:.1f}ms asm={a/40:.1f}ms tr={r/40:.1f}ms "
              f"total={(d+a+r)/40:.1f}ms/层", flush=True)
        _STAGE_ACC.update({"dma": 0.0, "asm": 0.0, "tr": 0.0})
    return now


def kmajor_from_engine_shards_v2(engine, device, hidden: int, inter: int,
                                 n_experts: int, group_k: int):
    """【§595 A/B】同 `kmajor_from_engine_shards`,但把三个阶段**分批**做:

        D) 先把所有节点的 DMA **一次性全发出去**(同一 stream,背靠背) → sync
        B) 再做全部跨步组装(纯 D2D)                                          → sync
        C) 最后做 4 个 K-major 转置                                            → sync

    为什么要这样:原实现是 **per-node `DMA → copy_` 交替**,每个 `copy_` 都排在
    自己那次 DMA 之后,于是 **PCIe 在每次 D2D 期间空转**,而且三段耗时混在一起、
    无法归因(§594 只能量到 asm 总量)。分批后既能让 DMA 连续占满链路,也能把
    "DMA = 多少 / 组装 = 多少 / 转置 = 多少" 分开报出来。
    """
    import time as _time
    import torch as _torch

    geo = engine.shard_geometry()
    ns = int(geo["ns"])
    if ns < 2 or not geo["w13_node_bytes"] or not geo["w2_node_bytes"]:
        return None
    _pin_engine_hostbufs(engine)
    H, I, E = int(hidden), int(inter), int(n_experts)
    rb13 = H // 2
    rb2 = I // 2
    gk = int(group_k) if int(group_k) > 0 else 1
    dbg = os.environ.get("XIAOTU_GPF_STAGE") == "1"

    # ---- D) 所有 DMA 背靠背发出(不夹 D2D)----------------------------------
    t0 = _time.perf_counter()
    b13 = [_dma_hostbuf(engine, 0, n, int(geo["w13_node_bytes"]), device) for n in range(ns)]
    b2 = [_dma_hostbuf(engine, 1, n, int(geo["w2_node_bytes"]), device) for n in range(ns)]
    bs13 = _dma_hostbuf(engine, 2, 0, int(geo["w13_scale_bytes"]), device)
    bs2 = _dma_hostbuf(engine, 3, 0, int(geo["w2_scale_bytes"]), device)
    if dbg:
        _torch.cuda.synchronize()
    t_d = _time.perf_counter() - t0

    # ---- B) 组装(纯 D2D,源是连续的 node 缓冲,目标是跨步行区间)----------
    t1 = _time.perf_counter()
    w13_raw = _reuse(("raw13",), (E, 2 * I, rb13), device)
    c13 = int(geo["w13_cbytes"])
    cr13 = int(geo["w13_crows"])
    for n in range(ns):
        blk = b13[n].view(E, 2, c13)
        c0 = n * cr13
        w13_raw[:, c0:c0 + cr13, :].copy_(blk[:, 0, :].reshape(E, cr13, rb13))
        w13_raw[:, I + c0:I + c0 + cr13, :].copy_(blk[:, 1, :].reshape(E, cr13, rb13))
    w2_raw = _reuse(("raw2",), (E, H, rb2), device)
    c2 = int(geo["w2_cbytes"])
    cr2 = int(geo["w2_crows"])
    for n in range(ns):
        c0 = n * cr2
        w2_raw[:, c0:c0 + cr2, :].copy_(b2[n].view(E, cr2, rb2))
    s13_raw = bs13.view(E, 2 * I, H // gk)
    s2_raw = bs2.view(E, H, I // gk)
    if dbg:
        _torch.cuda.synchronize()
    t_b = _time.perf_counter() - t1

    # ---- C) K-major 转置 ---------------------------------------------------
    t2 = _time.perf_counter()
    out = (_kmajor_bytes(w13_raw), _kmajor_bytes(s13_raw),
           _kmajor_bytes(w2_raw), _kmajor_bytes(s2_raw))
    if dbg:
        _torch.cuda.synchronize()
    t_c = _time.perf_counter() - t2

    if dbg:
        a = _STAGE_ACC
        a["n"] += 1
        a["dma"] += t_d * 1e3
        a["asm"] += t_b * 1e3
        a["tr"] += t_c * 1e3
        if a["n"] % 40 == 0:
            print(f"[gpf-v2] n={a['n']} dma={a['dma']/40:.1f}ms 组装={a['asm']/40:.1f}ms "
                  f"转置={a['tr']/40:.1f}ms total={(a['dma']+a['asm']+a['tr'])/40:.1f}ms/层",
                  flush=True)
            a.update({"dma": 0.0, "asm": 0.0, "tr": 0.0})
    return out


_PIN_DONE = {"n": -1}


def _pin_engine_hostbufs(engine) -> None:
    """【§612】首次走 GPU 预填充时,把引擎自有的 host 分片**就地锁页一次**。

    为什么:这些分片是 `mmap`/`numa_alloc_onnode` 的 **pageable** 内存,实测 DMA 只有
    16-21 GB/s;而同一机器上**锁页**拷贝是 **26.85 GB/s**(微基准 5.74 s/chunk)⇒
    每 chunk 固定成本 8.9 s → ~5.4 s。**不额外占内存**(只是锁住已有页,不复制)。
    幂等;失败只告警不抛(继续用 pageable)。`XIAOTU_GPF_PIN=0` 可关。
    """
    if _PIN_DONE["n"] >= 0 or os.environ.get("XIAOTU_GPF_PIN", "1") == "0":
        return
    try:
        n = int(engine.pin_hostbufs())
        _PIN_DONE["n"] = n
        print(f"[vllm-xtu-moe] GPU 预填充:引擎 host 分片锁页 {n} 个缓冲"
              f"(DMA 16-21 → 26.85 GB/s)", flush=True)
    except Exception as exc:  # noqa: BLE001
        _PIN_DONE["n"] = -1
        print(f"[vllm-xtu-moe] ⚠️ host 分片锁页不可用(继续 pageable): "
              f"{type(exc).__name__}: {exc}", flush=True)


def kmajor_from_engine_shards(engine, device, hidden: int, inter: int,
                              n_experts: int, group_k: int, dst=None):
    """K-major device weights built from the engine's OWN host buffers.

    WHY (report/tuning/NOTES.md §459): the GPU prefill path used to stream the
    checkpoint source tensors, which forced ``XIAOTU_RELEASE_SOURCE=0`` and cost an
    extra ~269 GiB of host memory (measured: EngineCore hit 1073 GB RSS before
    finishing the load, with ~3 GB free per NUMA node). But the engine already owns
    a complete copy of the expert weights -- as per-NUMA-node *compact* shards that
    together are exactly one dense copy, laid out per expert as
    ``[gate cbytes][up cbytes]`` with node n holding gate rows
    ``[n*crows,(n+1)*crows)`` and up rows ``[I+n*crows, I+(n+1)*crows)``
    (see ``shard_fill_w13``/``shard_fill_w2``) -- plus its own copy of the scales.

    So we DMA those and reassemble the canonical layout on the DEVICE. The
    reassembly is just ``view``+``cat``: node n's gate block IS the contiguous row
    range ``[n*crows,(n+1)*crows)`` of the canonical ``[E, I, H/2]`` gate matrix,
    so concatenating the nodes along dim 1 reproduces it exactly.

    Returns ``(w13_t, s13_t, w2_t, s2_t)`` already K-major (ready for a
    ``PrefetchSlot``), or ``None`` when the engine has no shards (e.g. NOSHARD) --
    the caller then falls back to the source tensors.
    """
    geo = engine.shard_geometry()
    ns = int(geo["ns"])
    if ns < 2 or not geo["w13_node_bytes"] or not geo["w2_node_bytes"]:
        return None
    H, I, E = int(hidden), int(inter), int(n_experts)
    rb13 = H // 2
    rb2 = I // 2
    gk = int(group_k) if int(group_k) > 0 else 1

    # Assemble by copy_ into the RAW target's row ranges: both sides are contiguous
    # runs of `cbytes` bytes, so each node's part is a plain contiguous copy.
    # (Measured: `view`+`cat` of the strided gate/up views then one transpose cost
    # **894 ms/layer**, because cat of strided views falls back to a slow kernel;
    # this form keeps the copies contiguous. See NOTES §466.)
    import time as _time
    _a0 = None
    if os.environ.get("XIAOTU_GPF_STAGE") == "1":
        import torch as _t0
        _a0 = _t0.cuda.memory_allocated(device)
    _t_dma = _time.perf_counter()
    w13_raw = _reuse(("raw13",), (E, 2 * I, rb13), device)
    c13 = int(geo["w13_cbytes"])
    cr13 = int(geo["w13_crows"])
    for n in range(ns):
        buf = _dma_hostbuf(engine, 0, n, int(geo["w13_node_bytes"]), device)
        blk = buf.view(E, 2, c13)
        c0 = n * cr13
        w13_raw[:, c0:c0 + cr13, :].copy_(blk[:, 0, :].reshape(E, cr13, rb13))
        w13_raw[:, I + c0:I + c0 + cr13, :].copy_(blk[:, 1, :].reshape(E, cr13, rb13))
        del buf, blk

    w2_raw = _reuse(("raw2",), (E, H, rb2), device)
    c2 = int(geo["w2_cbytes"])
    cr2 = int(geo["w2_crows"])
    for n in range(ns):
        buf = _dma_hostbuf(engine, 1, n, int(geo["w2_node_bytes"]), device)
        c0 = n * cr2
        w2_raw[:, c0:c0 + cr2, :].copy_(buf.view(E, cr2, rb2))
        del buf

    s13_raw = _dma_hostbuf(engine, 2, 0, int(geo["w13_scale_bytes"]), device) \
        .view(E, 2 * I, H // gk)
    s2_raw = _dma_hostbuf(engine, 3, 0, int(geo["w2_scale_bytes"]), device) \
        .view(E, H, I // gk)
    # [§594] 到此为止 = "DMA 发射 + 跨步组装" 都排在同一条流上;下面 sync 一次把它们
    # 一起结掉,所以先记 dma,再单独量转置(转置只在 GPU 上,不含 DMA)。
    _t_dma_end = _stage_mark(_t_dma, "dma")
    # K-major outputs are returned FRESH: `_kmajor_bytes` is a fast contiguous
    # transpose (73 ms total), whereas `reused.copy_(t.transpose(1,2))` is a
    # strided read and cost ~126 ms extra (NOTES §466). Reuse is worth it for the
    # big INTERMEDIATES (node DMA buffers + raw, ~13 GB of churn); not here.
    if dst is not None:
        # 【§597】写进调用方给的**复用缓冲**(环形槽),避免每层新分配 3.589 GiB ——
        # 实测每层新分配会让 `reserved` 单调上涨、free 跌破预填充阈值,
        # 于是第 8 层起逐层 DISABLED、退化成比纯 CPU 还慢的混合模式(76.5 s/13.8K)。
        _kmajor_bytes(w13_raw, dst[0])
        _kmajor_bytes(s13_raw, dst[1])
        _kmajor_bytes(w2_raw, dst[2])
        _kmajor_bytes(s2_raw, dst[3])
        _stage_mark(_t_dma_end, "tr")
        if _a0 is not None:
            import torch as _t2
            _d = (_t2.cuda.memory_allocated(device) - _a0) / 2**20
            print(f"[gpf-delta] staging_alloc={_d:+.1f} MiB (dst 复用路径)", flush=True)
        return tuple(dst)
    res = (_kmajor_bytes(w13_raw), _kmajor_bytes(s13_raw),
           _kmajor_bytes(w2_raw), _kmajor_bytes(s2_raw))
    _stage_mark(_t_dma_end, "tr")
    if _a0 is not None:
        import torch as _t1
        _d = (_t1.cuda.memory_allocated(device) - _a0) / 2**20
        print(f"[gpf-delta] staging_alloc={_d:+.1f} MiB", flush=True)
    return res




def prebuild_pinned_kmajor(tensors: list, workers: int = 6) -> None:
    """两阶段预建 K-major 缓存(见 report/tuning/TRIED_AND_REVERTED.md R4)。

    阶段 1(**纯 CPU**):并行做字节转置并放进缓存(不锁页)。任何时刻都安全,
      包括 CUDA graph 捕获期间,所以它能在 warmup 期间就跑起来 —— 这也是启动
      速度的关键:否则服务路径会单线程地逐层转置+锁页,启动要多花好几分钟。
    阶段 2(**CUDA**):等捕获静止(``wait_capture_done``)后并行 ``cudaHostRegister``。
      捕获窗口里调用它会作废捕获(`cudaErrorStreamCaptureInvalidated`,启动失败),
      甚至让 libcuda segfault(`cuMemHostRegister_v2` → SIGSEGV)。
    """
    from concurrent.futures import ThreadPoolExecutor

    def _transpose(quad):
        for t in quad:
            if t is not None:
                try:
                    _kmajor_cached(t)
                except Exception as e:  # noqa: BLE001
                    print(f"[xiaotu] prebuild transpose failed: {type(e).__name__}: {e}",
                          flush=True)

    def _register(quad):
        for t in quad:
            if t is not None:
                try:
                    wait_no_capture()
                    _pinned_kmajor(t)
                except Exception as e:  # noqa: BLE001
                    print(f"[xiaotu] prebuild pin failed: {type(e).__name__}: {e}",
                          flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_transpose, tensors))
    if not wait_capture_done():
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_register, tensors))


# --------------------------------------------------------------------------
# Prefetch-ahead: overlap layer L+1's H2D with layer L's compute.
#
# A small ring of device slots per (device, shape) is filled by an async copy on
# a dedicated side stream; the compute stream only waits on the slot's ready
# event. Reuse of a slot is guarded by the consumer's ``busy`` event so a copy
# never overwrites bytes the GPU is still reading.
# --------------------------------------------------------------------------
class PrefetchSlot:
    __slots__ = ("bufs", "ready", "busy", "t0", "t1")

    def __init__(self):
        self.bufs = None
        self.ready = None
        self.busy = None
        self.t0 = None   # XIAOTU_GP_TIMING: H2D 起止事件(量服务里真实传输时间)
        self.t1 = None

    def alloc(self, tensors, device):
        self.alloc_shapes(tuple(tuple(t.shape) for t in tensors), device)

    def alloc_shapes(self, shapes, device):
        shapes = tuple(tuple(int(v) for v in sh) for sh in shapes)
        if self.bufs is None or tuple(tuple(b.shape) for b in self.bufs) != shapes:
            self.bufs = tuple(
                torch.empty(s, dtype=torch.uint8, device=device) for s in shapes
            )


def slot_for_shapes(tensors, device, nslots: int = 0):
    """按**形状**取/建全局环形槽(所有层共享同一 key),并 alloc 出复用缓冲。

    与 `prefetch_layer` 共用 `_SLOTS`。**注意**:`prefetch_layer` 的"≥2 槽"下限是因为
    它在**侧流上为下一层预取**,单槽会覆盖正在用的权重(正确性 bug);
    而 **engine-shards 的同步路径** staging 与 GEMM 都在**同一条流**上严格有序
    ⇒ **单槽是安全的**,且能省 3.59 GiB(§601)。
    返回的 `slot.bufs` 可直接当 `kmajor_from_engine_shards(..., dst=slot.bufs)` 的目标。
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    shapes = tuple(tuple(t.shape) if hasattr(t, "shape") else tuple(t) for t in tensors)
    key = (dev.index,) + shapes
    slots = _SLOTS.get(key)
    if slots is None:
        if nslots and int(nslots) > 0:
            n = max(1, int(nslots))
        else:
            n = max(2, int(os.environ.get("XIAOTU_MOE_PREFETCH_SLOTS", "2") or 2))
        slots = [PrefetchSlot() for _ in range(n)]
        _SLOTS[key] = slots
    i = _RING.get(key, 0)
    _RING[key] = i + 1
    slot = slots[i % len(slots)]
    slot.alloc_shapes(shapes, dev)
    return slot


_SLOTS: dict[tuple, list] = {}
_RING: dict[tuple, int] = {}
_PREFETCH_STREAMS: dict[int, torch.cuda.Stream] = {}
_PREFETCH_DISABLED = False


def _prefetch_stream(device: torch.device) -> torch.cuda.Stream:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    s = _PREFETCH_STREAMS.get(idx)
    if s is None:
        s = torch.cuda.Stream(device=device)
        _PREFETCH_STREAMS[idx] = s
    return s


def prefetch_layer(w13, s13, w2, s2, device):
    """Issue the async H2D for one layer's K-major raw weights; return a slot.

    ``w13/s13/w2/s2`` must already be the K-major host (pinned) tensors, i.e.
    ``_pinned_kmajor(...)`` of the layer's checkpoint tensors (optionally an
    expert-parallel slice of them).

    Returns ``None`` when overlap is not possible (weights already on device, or
    the device is too full to hold the ping-pong buffers) — the caller then
    falls back to the synchronous per-layer H2D path.
    """
    global _PREFETCH_DISABLED
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if _PREFETCH_DISABLED or w13.device == dev:
        return None
    key = (dev.index, tuple(w13.shape), tuple(s13.shape),
           tuple(w2.shape), tuple(s2.shape))
    slots = _SLOTS.get(key)
    if slots is None:
        # 环深度可配(默认 2 = 一层的预取窗口)。更深 = 给 H2D 更多提前量,代价是
        # VRAM:每个槽 ≈ 一层权重(TP=2 每 rank 1.7 GB,TP=1 3.4 GB)。
        # 【下限必须是 2,不能用 1 槽省显存】实测(/tmp/test_1slot.py,2026-09-11):
        # 只有 1 槽时,为 L+1 发起的 H2D 会覆盖 L 正在用的那份权重 ⇒ 本层被算成
        # 下一层的权重(err_vs_L=17.5 vs err_vs_L+1=0.096),是**正确性 bug**。
        # 想省这 ~3GB 请改 XIAOTU_MOE_RESIDENT_BUDGET_GB(少放常驻层),不要动这里。
        nslots = max(2, int(os.environ.get("XIAOTU_MOE_PREFETCH_SLOTS", "2") or 2))
        slots = [PrefetchSlot() for _ in range(nslots)]
        _SLOTS[key] = slots
    i = _RING.get(key, 0)
    _RING[key] = i + 1
    slot = slots[i % len(slots)]
    try:
        slot.alloc((w13, s13, w2, s2), dev)
    except RuntimeError as e:  # includes torch.cuda.OutOfMemoryError
        if "out of memory" not in str(e).lower():
            raise
        # vLLM sized the KV cache first; there is no room for ping-pong buffers.
        # Drop the (now useless) slots and disable overlap for this process.
        _SLOTS.pop(key, None)
        _PREFETCH_DISABLED = True
        print("[xiaotu] GPU prefill: no VRAM for ping-pong slots, "
              "falling back to synchronous H2D (no overlap)", flush=True)
        return None
    st = _prefetch_stream(dev)
    if slot.busy is not None:
        st.wait_event(slot.busy)
    timing = os.environ.get("XIAOTU_GP_TIMING") == "1"
    if timing:
        slot.t0 = torch.cuda.Event(enable_timing=True)
        slot.t1 = torch.cuda.Event(enable_timing=True)
        slot.t0.record(st)
    with torch.cuda.stream(st):
        for buf, src in zip(slot.bufs, (w13, s13, w2, s2)):
            buf.copy_(src, non_blocking=True)
        slot.ready = torch.cuda.Event()
        slot.ready.record(st)
        if timing:
            slot.t1.record(st)
    return slot


def gpu_moe_layer(
    x, topk_ids, topk_weights,
    w13, s13, w2, s2,
    H: int, I: int, K: int,
    *,
    device: torch.device,
    slot: "PrefetchSlot | None" = None,
):
    """One layer's routed MoE on GPU (K-major raw bytes, in-kernel fp4 dequant).

    Streams this layer's raw MXFP4 weights H2D once (from cached pinned copies,
    a real DMA), then transposes the bytes to K-major on the GPU so the grouped
    GEMM kernels load ``b`` as ``[BK, BN]`` with no in-loop transpose.

    If ``slot`` is given it is a device buffer already filled (or being filled)
    by ``prefetch_layer``: we only wait for its ready event, so this layer's H2D
    overlaps the previous layer's kernels.
    """
    T = x.shape[0]
    E = w13.shape[0]
    device = torch.device(device) if not isinstance(device, torch.device) else device
    _t0 = time.perf_counter() if _GP_TIMING else 0.0
    if slot is not None:
        # 【不要在图捕获期间 wait_event】常驻层的 slot.ready 是在**捕获外**
        # (构建时)record 的,图内等待它会触发 cudaErrorStreamCaptureIsolation
        # ("dependency created on uncaptured work in another stream"),整个捕获作废。
        # 常驻槽位的数据在构建时就已写完(阻塞 H2D + 之后早已完成的事件),图内
        # 不需要这个等待;非捕获时仍然保留(预取路径靠它保证顺序)。
        # 见 report/tuning/TRIED_AND_REVERTED.md R14。
        if not torch.cuda.is_current_stream_capturing():
            torch.cuda.current_stream(device).wait_event(slot.ready)
        w13_t, s13_t, w2_t, s2_t = slot.bufs
    elif w13.device == device:
        w13_t = _kmajor_bytes(w13); s13_t = _kmajor_bytes(s13)
        w2_t = _kmajor_bytes(w2); s2_t = _kmajor_bytes(s2)
    else:
        w13_t = _kmajor_bytes(_pinned(w13).to(device, non_blocking=True))
        s13_t = _kmajor_bytes(_pinned(s13).to(device, non_blocking=True))
        w2_t = _kmajor_bytes(_pinned(w2).to(device, non_blocking=True))
        s2_t = _kmajor_bytes(_pinned(s2).to(device, non_blocking=True))

    tok, wts, seg_start, A = _build_segmentation(topk_ids, topk_weights, E, device)
    _t1 = time.perf_counter() if _GP_TIMING else 0.0
    out = torch.zeros((T, H), dtype=torch.bfloat16, device=device)
    if T == 0 or K == 0:
        # 形状判空(不是数据判空):A 现在是固定的 T*K,空批之外不会为 0。
        # 全部 id 无效时它们都落进垃圾桶段,每个专家的段为空 ⇒ 内核不写 out。
        return out
    # 【inter 需要 A=T*K 行,不是 T 行】gate_up 内核用**排序位置** g_rows 索引
    # inter(`inter_ptr + g_rows*inter_ld`),只有写 out 时才换成 token id。
    # 曾经按 T 行分配 ⇒ 越界写,cudaErrorIllegalAddress(第 19 轮实测)。
    # 想省这块分配只能改成"排序后行数=有效项数"或持久缓冲,不能再动行数语义。
    # 【§598】复用:见 `_reuse_moe` 的注释。`XIAOTU_GPF_REUSE=0` 可回退到每层新分配。
    if os.environ.get("XIAOTU_GPF_REUSE", "1") == "1":
        inter = _reuse_moe("inter", (A, 2 * I), torch.bfloat16, device)
    else:
        inter = torch.empty((A, 2 * I), dtype=torch.bfloat16, device=device)
    if os.environ.get("XIAOTU_GPF_STAGE") == "1":
        import torch as _tp
        print(f"[gpf-ptr] out_ptr={out.data_ptr()} inter_ptr={inter.data_ptr()} "
              f"alloc={_tp.cuda.memory_allocated(device)/2**20:.1f}MiB "
              f"reserved={_tp.cuda.memory_reserved(device)/2**20:.1f}MiB", flush=True)

    W13_E = w13_t.stride(0)
    S13_E = s13_t.stride(0)
    W2_E = w2_t.stride(0)
    S2_E = s2_t.stride(0)
    BM = int(os.environ.get("XIAOTU_GPU_PREFILL_BM", "64"))
    BN = int(os.environ.get("XIAOTU_GPU_PREFILL_BN", "64"))
    BK = int(os.environ.get("XIAOTU_GPU_PREFILL_BK", "64"))
    BH = int(os.environ.get("XIAOTU_GPU_PREFILL_BH", "64"))
    NS = int(os.environ.get("XIAOTU_GPU_PREFILL_STAGES", "2"))
    # Kernel variant: "split" avoids the per-k-block join+trans+reshape by
    # splitting the activation tile into even/odd k columns and doing two dots.
    _variant = os.environ.get("XIAOTU_GPU_PREFILL_KERNEL", "base")
    _lut = os.environ.get("XIAOTU_GPU_PREFILL_LUT", "0") == "1"
    NW = int(os.environ.get("XIAOTU_GPU_PREFILL_WARPS", "4"))
    lut_t = _e2m1_table(device)
    if _variant == "split":
        _gate_up_kernel_split[(E, triton.cdiv(2 * I, BN))](
            x, x.stride(0), tok, seg_start,
            w13_t, w13_t.stride(1), s13_t, s13_t.stride(1),
            inter, inter.stride(0), W13_E, S13_E,
            H=H, BM=BM, BN=BN, BK=BK, NS=NS, LUT=_lut, num_warps=NW,
        )
        _down_kernel_split[(E, triton.cdiv(H, BH))](
            inter, inter.stride(0), tok, wts, seg_start,
            w2_t, w2_t.stride(1), s2_t, s2_t.stride(1),
            out, out.stride(0), W2_E, S2_E,
            H=H, I=I, BM=BM, BH=BH, BK=BK, NS=NS, LUT=_lut, num_warps=NW,
        )
    else:
        _gate_up_kernel[(E, triton.cdiv(2 * I, BN))](
            x, x.stride(0), tok, seg_start,
            w13_t, w13_t.stride(1), s13_t, s13_t.stride(1),
            inter, inter.stride(0), W13_E, S13_E, lut_t,
            H=H, BM=BM, BN=BN, BK=BK, NS=NS, LUT=_lut, num_warps=NW,
        )
        _down_kernel[(E, triton.cdiv(H, BH))](
            inter, inter.stride(0), tok, wts, seg_start,
            w2_t, w2_t.stride(1), s2_t, s2_t.stride(1),
            out, out.stride(0), W2_E, S2_E, lut_t,
            H=H, I=I, BM=BM, BH=BH, BK=BK, NS=NS, LUT=_lut, num_warps=NW,
        )
    if _GP_TIMING:
        _gp_add(T, A, _t1 - _t0, time.perf_counter() - _t1)
        if slot is not None and getattr(slot, "t0", None) is not None:
            try:   # H2D 传输时间(事件已完成:本层已在等 ready)
                _gp_h2d(slot.t0.elapsed_time(slot.t1))
            except Exception:  # noqa: BLE001
                pass
    if slot is not None:
        # Mark the slot free only after these kernels have actually finished.
        slot.busy = torch.cuda.Event()
        slot.busy.record(torch.cuda.current_stream(device))
    return out


def torch_reference_layer(
    x, topk_ids, topk_weights, w13, s13, w2, s2, H: int, I: int,
) -> torch.Tensor:
    """Pure-torch reference of the SAME MXFP4 layout for golden testing."""
    T = x.shape[0]
    K = topk_ids.shape[1]
    dev = x.device
    E = w13.shape[0]
    vals = _E2M1.to(dev)

    def exp_dequant(e):
        lo13 = (w13[e, :, :] & 0x0F).to(torch.int64)
        hi13 = ((w13[e, :, :] >> 4) & 0x0F).to(torch.int64)
        w13e = torch.stack([vals[lo13], vals[hi13]], dim=-1).reshape(2 * I, H)
        sc13 = torch.exp2(s13[e].float() - 127.0).repeat_interleave(32, dim=1)[..., :H]
        W13e = (w13e * sc13).to(x.dtype)
        lo2 = (w2[e, :, :] & 0x0F).to(torch.int64)
        hi2 = ((w2[e, :, :] >> 4) & 0x0F).to(torch.int64)
        w2e = torch.stack([vals[lo2], vals[hi2]], dim=-1).reshape(H, I)
        sc2 = torch.exp2(s2[e].float() - 127.0).repeat_interleave(32, dim=1)[..., :I]
        W2e = (w2e * sc2).to(x.dtype)
        return W13e, W2e

    cache = {}
    out = torch.zeros(T, H, dtype=x.dtype, device=dev)
    for t in range(T):
        for ssl in range(K):
            e = int(topk_ids[t, ssl])
            if e not in cache:
                cache[e] = exp_dequant(e)
            W13e, W2e = cache[e]
            gate = x[t] @ W13e[:I].t()
            up = x[t] @ W13e[I:].t()
            act = torch.nn.functional.silu(gate) * up
            out[t] += topk_weights[t, ssl] * (act @ W2e.t())
    return out

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
    f = os.environ.get("XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE")
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
    w_off = pid_e * W13_E
    s_off = pid_e * S13_E
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
    w_off = pid_e * W13_E
    s_off = pid_e * S13_E
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
    w_off = pid_e * W2_E
    s_off = pid_e * S2_E
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
    w_off = pid_e * W2_E
    s_off = pid_e * S2_E
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


def _build_segmentation(topk_ids, topk_weights, num_experts, device):
    T, K = topk_ids.shape
    ids = topk_ids.to(torch.int32).reshape(-1)
    wts = topk_weights.to(torch.float32).reshape(-1)
    tok_ids = torch.arange(T, device=device).repeat_interleave(K)
    ok = (ids >= 0) & (ids < num_experts)
    ids = ids[ok]; wts = wts[ok]; tok_ids = tok_ids[ok]
    if ids.numel() == 0:
        return (torch.empty(0, dtype=torch.int64, device=device),
                torch.empty(0, dtype=torch.float32, device=device),
                torch.zeros(1, dtype=torch.int32, device=device), 0)
    sorted_order = ids.argsort(stable=True)
    sorted_tok = tok_ids[sorted_order]
    sorted_wts = wts[sorted_order]
    counts = torch.bincount(ids[sorted_order], minlength=num_experts).to(torch.int32)
    seg_start = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(counts, dim=0)]
    )
    return sorted_tok, sorted_wts, seg_start, int(sorted_tok.numel())


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


def _pinned(t: torch.Tensor) -> torch.Tensor:
    # Keyed by the *storage* address, which is stable across slices of the same
    # parameter (``param.data[a:b]``). We also hold the source storage alive so
    # the address cannot be reused by an unrelated allocation while the entry
    # exists (otherwise a freed tensor could alias a stale pinned copy).
    stor = t.untyped_storage()
    key = (stor.data_ptr(), t.storage_offset(), tuple(t.shape), t.dtype)
    ent = _PIN_CACHE.get(key)
    if ent is not None:
        return ent[1]
    with _key_lock(key):
        ent = _PIN_CACHE.get(key)
        if ent is None:
            c = t.contiguous()
            if not _register_host(c):
                c = c.pin_memory()
            ent = (stor, c)
            _PIN_CACHE[key] = ent
    return ent[1]


_PIN_CACHE: dict[tuple, tuple] = {}


def _kmajor_bytes(t):
    """[E, A, B] u8 -> [E, B, A] u8 (byte transpose; nibbles stay within bytes)."""
    return t.transpose(1, 2).contiguous()


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
    stor = t.untyped_storage()
    key = (stor.data_ptr(), t.storage_offset(), tuple(t.shape), t.dtype, "kmajor")
    ent = _PIN_CACHE.get(key)
    if ent is not None:
        return ent[1]
    with _key_lock(key):
        ent = _PIN_CACHE.get(key)
        if ent is None:
            c = _kmajor_bytes(t)
            if not _register_host(c):
                c = c.pin_memory()
            ent = (stor, c)
            _PIN_CACHE[key] = ent
    return ent[1]


def prebuild_pinned_kmajor(tensors: list, workers: int = 6) -> None:
    """Build the K-major pinned cache for many layers in parallel (once).

    Called on the first long-prefill forward; the caller does not wait, so the
    copies overlap with the first few layers' kernels. The per-key lock in
    ``_pinned_kmajor`` makes a concurrent rebuild of the same layer safe.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _one(quad):
        for t in quad:
            if t is not None:
                try:
                    _pinned_kmajor(t)
                except Exception as e:  # noqa: BLE001
                    print(f"[xiaotu] prebuild failed: {type(e).__name__}: {e}",
                          flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_one, tensors))


# --------------------------------------------------------------------------
# Prefetch-ahead: overlap layer L+1's H2D with layer L's compute.
#
# A small ring of device slots per (device, shape) is filled by an async copy on
# a dedicated side stream; the compute stream only waits on the slot's ready
# event. Reuse of a slot is guarded by the consumer's ``busy`` event so a copy
# never overwrites bytes the GPU is still reading.
# --------------------------------------------------------------------------
class PrefetchSlot:
    __slots__ = ("bufs", "ready", "busy")

    def __init__(self):
        self.bufs = None
        self.ready = None
        self.busy = None

    def alloc(self, tensors, device):
        shapes = tuple(tuple(t.shape) for t in tensors)
        if self.bufs is None or self.bufs[0].shape != shapes[0] or \
                self.bufs[1].shape != shapes[1] or self.bufs[2].shape != shapes[2] or \
                self.bufs[3].shape != shapes[3]:
            self.bufs = tuple(
                torch.empty(s, dtype=torch.uint8, device=device) for s in shapes
            )


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
        slots = [PrefetchSlot(), PrefetchSlot()]
        _SLOTS[key] = slots
    i = _RING.get(key, 0)
    _RING[key] = i + 1
    slot = slots[i & 1]
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
    with torch.cuda.stream(st):
        for buf, src in zip(slot.bufs, (w13, s13, w2, s2)):
            buf.copy_(src, non_blocking=True)
        slot.ready = torch.cuda.Event()
        slot.ready.record(st)
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
    if slot is not None:
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
    out = torch.zeros((T, H), dtype=torch.bfloat16, device=device)
    if A == 0:
        return out
    inter = torch.empty((A, 2 * I), dtype=torch.bfloat16, device=device)

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

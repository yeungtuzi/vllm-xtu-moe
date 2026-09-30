# SPDX-License-Identifier: Apache-2.0
"""UVA 零拷贝预填路径(实验一产物 ✓;`XIAOTU_GPF_UVA=1` 打开,默认关 ✓)。

思路(用户 2026-09-30 定的两条原则 ✓):
1. **内存廉价、prefill 期 CPU 廉价** ⇒ 让 CPU 把权重预排成 GPU 最舒服的形态:
   **严格连续读** ⇒ CPU 侧按 ``[E, n_block, rows, BLK]`` 重排一次(一次性、可缓存 ✓)
2. **永远有准备好的层** ⇒ 深环常备(由调用方 `mixed_experts` 的 slot 机制提供 ✓)

实测依据(见 `dev-docs/UVA_ZEROCOPY_EXPERIMENT.md` ✓):
* E1:UVA **顺序读 = 25.02 GiB/s**(= 批量 DMA 线速 ✓);跨步读崩塌 ✗
* E2:原寻址 UVA = 9.47 GiB/s ✗(内核是"读 64B 连续 + 跳 4096B" ✗)
* E3:仅换寻址 ⇒ **25.03 GiB/s = 线速 100%** ✓
* E3-full:真实整层 ⇒ 710 → **389 ms**(1.83× ✓),但仍比 230 ms 基线慢 1.7× ✗
  ⇒ 缺口在"读与计算串行" ✗ ⇒ 本模块用 `num_stages` 流水 + 深环来补 ✓

⚠️ 与主路径的关系:本模块**不改变**默认行为 ✓;`XIAOTU_GPF_UVA=0`(默认)时一切照旧 ✓。
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from vllm_xiaotu_moe.gpu_prefill import _dequant_e2m1_gather, _dequant_e2m1_nibble

UVA_BLK = 64


def uva_enabled() -> bool:
    return os.environ.get("XIAOTU_GPF_UVA", "0") == "1"


def _permute_kmajor(t: torch.Tensor, blk: int = UVA_BLK) -> torch.Tensor:
    """把 K-major ``[E, rows, cols]`` 预排成 ``[E, n_block, rows, blk]``(CPU ✓)。

    目的:内核按"固定 n_block、遍历 rows"读取时**地址连续** ✓(E3 实测 25.03 GiB/s ✓)。
    """
    e, r, c = t.shape
    if c % blk:
        raise ValueError(f"cols {c} 不能被 blk {blk} 整除")
    return t.reshape(e, r, c // blk, blk).permute(0, 2, 1, 3).contiguous()


def build_seq_mirror(w13, s13, w2, s2, blk: int = UVA_BLK):
    """输入 4 个 **宿主 K-major** 张量(如 `_pinned_kmajor` 的产物 ✓)⇒ 返回 UVA 视图四元组 ✓。"""
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    out = []
    for t in (w13, s13, w2, s2):
        host = t.detach().to("cpu", torch.uint8).contiguous()
        p = _permute_kmajor(host, blk)
        if not p.is_pinned():
            p = p.pin_memory()
        out.append(get_accelerator_view_from_cpu_tensor(p))
    return tuple(out)


@triton.jit
def _gate_up_uva(x_ptr, x_stride, tok_ptr, seg_ptr, w13_ptr, s13_ptr,
                 inter_ptr, inter_ld, W13_E, S13_E,
                 H: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 ROWS: tl.constexpr, SROWS: tl.constexpr, NS: tl.constexpr, LUT: tl.constexpr):
    pid_e = tl.program_id(0)
    pid_n = tl.program_id(1)
    base = tl.load(seg_ptr + pid_e)
    end = tl.load(seg_ptr + pid_e + 1)
    if end <= base:
        return
    w_off = pid_e.to(tl.int64) * W13_E + pid_n * (ROWS * BN)
    s_off = pid_e.to(tl.int64) * S13_E + pid_n * (SROWS * BN)
    c = tl.arange(0, BN)
    mt = 0
    while mt * BM < (end - base):
        g_rows = base + mt * BM + tl.arange(0, BM)
        rmask = g_rows < end
        toks = tl.load(tok_ptr + g_rows, mask=rmask, other=0)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for kk in tl.range(0, H, BK, num_stages=NS):
            a = tl.load(x_ptr + toks[:, None] * x_stride + (kk + tl.arange(0, BK))[None, :],
                        mask=rmask[:, None], other=0.0)
            r = (kk >> 1) + tl.arange(0, BK // 2)
            b_lo = tl.load(w13_ptr + w_off + r[:, None] * BN + c[None, :])
            if LUT:
                w_e = _dequant_e2m1_gather(b_lo & 0x0F, None)
                w_o = _dequant_e2m1_gather((b_lo >> 4) & 0x0F, None)
            else:
                w_e = _dequant_e2m1_nibble(b_lo & 0x0F)
                w_o = _dequant_e2m1_nibble((b_lo >> 4) & 0x0F)
            b_val = tl.reshape(tl.trans(tl.join(w_e, w_o)), (BK, BN))
            sr = (kk >> 5) + tl.arange(0, BK // 32)
            scale = tl.load(s13_ptr + s_off + sr[:, None] * BN + c[None, :])
            sc = tl.exp2(scale.to(tl.float32) - 127.0)
            sc = tl.reshape(tl.broadcast_to(sc[:, None, :], (BK // 32, 32, BN)), (BK, BN))
            acc = tl.dot(a, (b_val * sc).to(tl.bfloat16), acc)
        tl.store(inter_ptr + g_rows[:, None] * inter_ld + (pid_n * BN + c)[None, :],
                 acc.to(tl.bfloat16), mask=rmask[:, None])
        mt += 1


@triton.jit
def _down_uva(inter_ptr, inter_ld, tok_ptr, wts_ptr, seg_ptr, w2_ptr, s2_ptr,
              out_ptr, out_ld, W2_E, S2_E,
              I: tl.constexpr, BM: tl.constexpr, BH: tl.constexpr, BK: tl.constexpr,
              ROWS: tl.constexpr, SROWS: tl.constexpr, NS: tl.constexpr, LUT: tl.constexpr):
    pid_e = tl.program_id(0)
    pid_h = tl.program_id(1)
    base = tl.load(seg_ptr + pid_e)
    end = tl.load(seg_ptr + pid_e + 1)
    if end <= base:
        return
    w_off = pid_e.to(tl.int64) * W2_E + pid_h * (ROWS * BH)
    s_off = pid_e.to(tl.int64) * S2_E + pid_h * (SROWS * BH)
    c = tl.arange(0, BH)
    mt = 0
    while mt * BM < (end - base):
        g_rows = base + mt * BM + tl.arange(0, BM)
        rmask = g_rows < end
        toks = tl.load(tok_ptr + g_rows, mask=rmask, other=0)
        wts = tl.load(wts_ptr + g_rows, mask=rmask, other=0.0).to(tl.float32)
        acc = tl.zeros((BM, BH), dtype=tl.float32)
        for kk in tl.range(0, I, BK, num_stages=NS):
            gate = tl.load(inter_ptr + g_rows[:, None] * inter_ld + (kk + tl.arange(0, BK))[None, :],
                           mask=rmask[:, None], other=0.0).to(tl.float32)
            up = tl.load(inter_ptr + g_rows[:, None] * inter_ld
                         + (I + kk + tl.arange(0, BK))[None, :],
                         mask=rmask[:, None], other=0.0).to(tl.float32)
            act = (gate / (1.0 + tl.math.exp(-gate))) * up
            r = (kk >> 1) + tl.arange(0, BK // 2)
            b_lo = tl.load(w2_ptr + w_off + r[:, None] * BH + c[None, :])
            if LUT:
                w_e = _dequant_e2m1_gather(b_lo & 0x0F, None)
                w_o = _dequant_e2m1_gather((b_lo >> 4) & 0x0F, None)
            else:
                w_e = _dequant_e2m1_nibble(b_lo & 0x0F)
                w_o = _dequant_e2m1_nibble((b_lo >> 4) & 0x0F)
            b_val = tl.reshape(tl.trans(tl.join(w_e, w_o)), (BK, BH))
            sr = (kk >> 5) + tl.arange(0, BK // 32)
            scale = tl.load(s2_ptr + s_off + sr[:, None] * BH + c[None, :])
            sc = tl.exp2(scale.to(tl.float32) - 127.0)
            sc = tl.reshape(tl.broadcast_to(sc[:, None, :], (BK // 32, 32, BH)), (BK, BH))
            acc = tl.dot(act.to(tl.bfloat16), (b_val * sc).to(tl.bfloat16), acc)
        tl.atomic_add(out_ptr + toks[:, None] * out_ld + (pid_h * BH + c)[None, :],
                      acc * wts[:, None], mask=rmask[:, None], sem="relaxed")
        mt += 1


def gpu_moe_layer_uva(x, topk_ids, topk_weights, mirror, H: int, I: int, K: int, *,
                      device: torch.device, inter=None, out=None,
                      BM: int = 64, BN: int = 64, BK: int = 64, BH: int = 64,
                      NS: int = 2, LUT: bool = False):
    """UVA 版一层 MoE(语义与 `gpu_prefill.gpu_moe_layer` 对齐 ✓,权重从宿主直读 ✓)。"""
    from vllm_xiaotu_moe.gpu_prefill import _build_segmentation

    device = torch.device(device)
    w13, s13, w2, s2 = mirror
    T, E = x.shape[0], w13.shape[0]
    tok, wts, seg, A = _build_segmentation(topk_ids, topk_weights, E, device)
    if out is None:
        out = torch.zeros((T, H), dtype=torch.bfloat16, device=device)
    if T == 0 or K == 0:
        return out
    if inter is None:
        inter = torch.empty((A, 2 * I), dtype=torch.bfloat16, device=device)
    _gate_up_uva[(E, triton.cdiv(2 * I, BN))](
        x, x.stride(0), tok, seg, w13, s13, inter, inter.stride(0),
        w13.stride(0), s13.stride(0),
        H=H, BM=BM, BN=BN, BK=BK, ROWS=H // 2, SROWS=H // 32, NS=NS, LUT=LUT,
        num_warps=int(os.environ.get("XIAOTU_GPU_PREFILL_WARPS", "4")))
    _down_uva[(E, triton.cdiv(H, BH))](
        inter, inter.stride(0), tok, wts, seg, w2, s2, out, out.stride(0),
        w2.stride(0), s2.stride(0),
        I=I, BM=BM, BH=BH, BK=BK, ROWS=I // 2, SROWS=I // 32, NS=NS, LUT=LUT,
        num_warps=int(os.environ.get("XIAOTU_GPU_PREFILL_WARPS", "4")))
    return out

# ── 镜像缓存:每个 (engine, layer) 只做一次「装配 → 拷宿主 → 预排 → pin → UVA」✓ ──
_MIRRORS: dict = {}
_MIRROR_LOCK = None


def get_or_build_mirror(cache_key, build_device_fn, *, blk: int = UVA_BLK, free_device: bool = True):
    """返回 UVA 镜像(4 元组 ✓);首次调用时用 ``build_device_fn()`` 造设备端 K-major ✓。

    ★ 关键:建成后**释放设备端那几份** ✓ ⇒ 稳态下设备侧不再持有 staging ✓
    (这正是"省 10.73 GiB/卡"的落地方式 ✓;官方 design 见 dev-docs/UVA_ZEROCOPY_EXPERIMENT.md ✓)

    ⚠️ 必须在 **CUDA graph 捕获之前**调用 ✗(cudaHostRegister 会作废捕获 ✓)。
    """
    global _MIRROR_LOCK
    import threading

    if _MIRROR_LOCK is None:
        _MIRROR_LOCK = threading.Lock()
    hit = _MIRRORS.get(cache_key)
    if hit is not None:
        return hit
    with _MIRROR_LOCK:
        hit = _MIRRORS.get(cache_key)
        if hit is not None:
            return hit
        devbufs = build_device_fn()
        if devbufs is None:
            return None
        hosts = []
        for t in devbufs:
            hosts.append(t.detach().to("cpu", torch.uint8).contiguous())
        if free_device:
            del devbufs
            torch.cuda.empty_cache()
        mirror = build_seq_mirror(*hosts, blk=blk)
        _MIRRORS[cache_key] = mirror
        return mirror


def mirror_bytes(mirror) -> int:
    return sum(t.numel() * t.element_size() for t in mirror) if mirror else 0

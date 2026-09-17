# ---------------------------------------------------------------------------
# **字节矩阵转置**(uint8)的分块 Triton 实现。
#
# 为什么需要它(§596 实测):权重是 fp4 打包的 uint8,**K-major 需要在最后一维做
# 逐字节转置**,而 `t.transpose(1,2).contiguous()` 对这种 dtype 只能跑到
# **~90-100 GB/s**(逐字节 elementwise kernel,无法向量化 + 非合并访问)。
# 每层 4 个张量合计 **42.8 ms**;换成 128×128 分块(经 shared/寄存器中转)后
# **6.5 ms/层**,快 **5-7×**,且 `torch.equal` 逐位相同。
#
# 用法:
#   from vllm_xiaotu_moe.byte_transpose import ktranspose_bytes
#   y = ktranspose_bytes(x)          # [E, A, B] -> [E, B, A]
# 关掉(退回 torch):`XIAOTU_GPF_TT=0`。
# ---------------------------------------------------------------------------
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # noqa: BLE001
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _ktranspose_bytes_kernel(x, y, A, B, sx, sy,
                                 BM: tl.constexpr, BN: tl.constexpr):
        e = tl.program_id(0).to(tl.int64)      # ⚠️ 必须 int64:e*stride 会超 2^31
        pm = tl.program_id(1)
        pn = tl.program_id(2)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        v = tl.load(x + e * sx + rm[:, None] * B + rn[None, :],
                    mask=(rm[:, None] < A) & (rn[None, :] < B), other=0)
        tl.store(y + e * sy + rn[:, None] * A + rm[None, :], tl.trans(v),
                 mask=(rn[:, None] < B) & (rm[None, :] < A))


def _enabled() -> bool:
    return os.environ.get("XIAOTU_GPF_TT", "1") != "0"


def ktranspose_bytes(t: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """[E, A, B] uint8 -> [E, B, A] uint8(K-major 字节转置)。

    `out` 给了就**写进去**(不分配)——staging 路径靠这个复用环形缓冲、消除每层分配。
    只在 3 维、uint8/byte、CUDA、且开着 `XIAOTU_GPF_TT` 时走分块 kernel;
    其余情况(含 triton 不可用、或 out 给了但没法原地转置)退回 torch。
    """
    if (not _HAVE_TRITON or not _enabled() or t.dim() != 3
            or t.dtype != torch.uint8 or not t.is_cuda or not t.is_contiguous()
            or (out is not None and (not out.is_contiguous()
                                     or tuple(out.shape) != (t.shape[0], t.shape[2], t.shape[1])))):
        r = t.transpose(1, 2).contiguous()
        if out is not None:
            out.copy_(r)
            return out
        return r
    E, A, B = (int(v) for v in t.shape)
    y = out if out is not None else torch.empty((E, B, A), dtype=t.dtype, device=t.device)
    BM = 128 if A % 128 == 0 else 64
    BN = 128 if B % 128 == 0 else (64 if B % 64 == 0 else 32)
    grid = (E, triton.cdiv(A, BM), triton.cdiv(B, BN))
    _ktranspose_bytes_kernel[grid](t, y, A, B, t.stride(0), y.stride(0),
                                   BM=BM, BN=BN)
    return y

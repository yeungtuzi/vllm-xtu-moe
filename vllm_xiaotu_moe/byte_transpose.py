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


# ---------------------------------------------------------------------------
# 【§459 / 方案 B】**任意 stride** 的版本:直接把每个 NUMA node 的分片转置进
# K-major 目标的一个切片,从而**不需要 raw 规范布局缓冲**(省 3.16 GiB/rank)。
#
# 为什么现有 `ktranspose_bytes` 不够:它把源的中间维 stride 硬编码成 `B`、
# 目标的中间维 stride 硬编码成 `A`。而我们要写的是
#     dst[:, :, a0 : a0+A]        # dst = [E, B, 2I] 连续
# 它的中间维 stride 是 **2I**(整行),不是 A。所以把两个中间 stride 都参数化。
# ---------------------------------------------------------------------------
if _HAVE_TRITON:

    @triton.jit
    def _ktranspose_bytes_strided_kernel(x, y, A, B,
                                         sx_e, sx_m, sy_e, sy_m,
                                         BM: tl.constexpr, BN: tl.constexpr):
        """`y[e, b, a] = x[e, a, b]`;两端都允许任意 e / 中间维 stride(末维须连续)。"""
        e = tl.program_id(0).to(tl.int64)      # ⚠️ int64:e*stride 会超 2^31
        pm = tl.program_id(1)
        pn = tl.program_id(2)
        rm = pm * BM + tl.arange(0, BM)        # 沿 A
        rn = pn * BN + tl.arange(0, BN)        # 沿 B
        v = tl.load(x + e * sx_e + rm[:, None] * sx_m + rn[None, :],
                    mask=(rm[:, None] < A) & (rn[None, :] < B), other=0)
        tl.store(y + e * sy_e + rn[:, None] * sy_m + rm[None, :], tl.trans(v),
                 mask=(rn[:, None] < B) & (rm[None, :] < A))


def ktranspose_into(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """`y[e, b, a] = x[e, a, b]` —— **允许两端都是任意 strided 的 3 维视图**(末维须连续)。

    与 `ktranspose_bytes` 的差别:那个要求 `x` 连续、且 `y` 的中间维 stride == A;
    这个把两个中间 stride 都作为参数传下去,于是可以**直接写进 K-major 目标的切片**
    (`dst[:, :, a0:a0+A]`,其中间维 stride 是整行 2I 而非 A)。

    用途(§459 / 方案 B):逐 NUMA node 直填 K-major,省掉每 rank 3.16 GiB 的 raw 缓冲。
    形状必须精确匹配 `x:[E,A,B] -> y:[E,B,A]`,不做广播 —— 索引算错会**静默算错**,
    所以宁可在这里直接报错。
    """
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError(f"ktranspose_into expects 3-D tensors, got {x.dim()}/{y.dim()}")
    E, A, B = (int(v) for v in x.shape)
    if tuple(y.shape) != (E, B, A):
        raise ValueError(f"ktranspose_into shape mismatch: x{tuple(x.shape)} -> y{tuple(y.shape)}")
    if x.stride(2) != 1 or y.stride(2) != 1:
        raise ValueError(
            f"ktranspose_into requires a contiguous last dim, got "
            f"x.stride(2)={x.stride(2)} y.stride(2)={y.stride(2)}"
        )
    if (not _HAVE_TRITON or not _enabled() or x.dtype != torch.uint8
            or y.dtype != torch.uint8 or not x.is_cuda or not y.is_cuda):
        y.copy_(x.transpose(1, 2))
        return y
    # ⚠️【实测】这里的 BM 决定**目标侧每次连续写多少字节**:strided 目标的中间维 stride
    # 是整行(2I),所以一次只写 BM 字节。原先按 `A % 128` 选,遇到 A=288 会落到 **BM=32**
    # ⇒ 只有 32 字节连续、实测 ~228 GB/s;改成"A≥128 一律 BM=128(尾巴交给 mask)"
    # 后每次写 128 字节。**不要**为了"整除好看"把 BM 调小。
    BM = 128 if A >= 128 else (64 if A >= 64 else 32)
    BN = 128 if B % 128 == 0 else (64 if B % 64 == 0 else 32)
    grid = (E, triton.cdiv(A, BM), triton.cdiv(B, BN))
    _ktranspose_bytes_strided_kernel[grid](
        x, y, A, B, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
        BM=BM, BN=BN)
    return y

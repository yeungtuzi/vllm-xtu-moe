"""Engine 侧 clamped SwiGLU(swiglu_limit/alpha/beta)的数值一致性测试。

对标 vLLM `silu_and_mul_with_clamp(out, in, limit, alpha, beta)`:
  out = clamp(gate, max=limit) * sigmoid(alpha * clamp(gate, max=limit))
        * (clamp(up, +-limit) + beta)

用法: python scripts/test_swiglu_clamp.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import xiaotu_moe  # noqa: E402

E, H, I, K, M = 6, 256, 128, 2, 16
torch.manual_seed(0)
rng = np.random.default_rng(0)


def make_cfg(limit: float, alpha: float, beta: float):
    cfg = xiaotu_moe.MOEConfigV2()
    cfg.num_processes = 1
    cfg.gpu_id = 0
    cfg.has_gate_proj = True
    cfg.expert_num = E
    cfg.top_k = K
    cfg.hidden_size = H
    cfg.intermediate_size = I
    cfg.max_batch_size = 1024
    cfg.max_num_seqs = 16
    cfg.stride = 1
    cfg.groupN = 0
    cfg.groupK = 0
    cfg.activation_type = 1 if (limit > 0 or alpha != 1.0 or beta != 0.0) else 0
    cfg.swiglu_limit = limit
    cfg.swiglu_alpha = alpha
    cfg.swiglu_beta = beta
    return cfg


def to_bf16(a: np.ndarray) -> np.ndarray:
    """float32 -> bf16 位模式(uint16),再取回 float32(与引擎看到的数值一致)。"""
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(torch.bfloat16)
    return t.view(torch.uint16).numpy()


def from_bf16(bits: np.ndarray) -> np.ndarray:
    return torch.from_numpy(bits).view(torch.bfloat16).float().numpy()


def reference(x, w13, w2, ids, wts, limit, alpha, beta):
    out = np.zeros((M, H), dtype=np.float64)
    for t in range(M):
        for r in range(K):
            e = int(ids[t, r])
            g = w13[e, :I].astype(np.float64) @ x[t].astype(np.float64)
            u = w13[e, I:].astype(np.float64) @ x[t].astype(np.float64)
            if limit > 0:
                g = np.minimum(g, limit)
                u = np.clip(u, -limit, limit)
            act = (g / (1.0 + np.exp(-alpha * g))) * (u + beta)
            out[t] += float(wts[t, r]) * (w2[e].astype(np.float64) @ act)
    return out


def main() -> int:
    w13_b = to_bf16(rng.standard_normal((E, 2 * I, H)) * 0.05)
    w2_b = to_bf16(rng.standard_normal((E, H, I)) * 0.05)
    x_f = rng.standard_normal((M, H)) * 1.0
    x_f[: M // 2] *= 30.0  # 制造超出 clamp 的值
    x_b = to_bf16(x_f)
    w13 = from_bf16(w13_b)
    w2 = from_bf16(w2_b)
    x = from_bf16(x_b)
    ids = rng.integers(0, E, size=(M, K)).astype(np.uint32)
    wts = rng.random((M, K)).astype(np.float32)

    rc = 0
    for limit, alpha, beta in [
        (0.0, 1.0, 0.0),   # plain silu (activation_type=0)
        (10.0, 1.0, 0.0),  # GLM-5.x / DS-V4
        (7.0, 1.0, 0.0),   # MiniMax-M3
        (10.0, 1.5, 0.0),  # alpha != 1
        (10.0, 1.0, 0.5),  # beta != 0
    ]:
        eng = xiaotu_moe.MOE_BF16(
            make_cfg(limit, alpha, beta),
            w13_b.ctypes.data, w2_b.ctypes.data, 0, 0, 0, 0,
        )
        out = np.zeros((M, H), dtype=np.float32)
        eng.cpu_prefill(
            M, K,
            ids.ctypes.data, wts.ctypes.data,
            x_b.ctypes.data, out.ctypes.data,
        )
        ref = reference(x, w13, w2, ids, wts, limit, alpha, beta)
        denom = np.sqrt(np.mean(ref**2)) + 1e-12
        rel = float(np.sqrt(np.mean((out - ref) ** 2)) / denom)
        mx = float(np.max(np.abs(out - ref)))
        ok = rel < 5e-3
        rc |= 0 if ok else 1
        print(
            f"[{'OK ' if ok else 'BAD'}] limit={limit:<5} alpha={alpha:<4} beta={beta:<4} "
            f"rel_rms={rel:.3e} max_abs={mx:.3e}  ref_rms={denom:.4f}",
            flush=True,
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

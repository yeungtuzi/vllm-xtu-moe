#!/usr/bin/env python
"""Synthetic FP8 CPU-MoE equivalence + M sweep for the (MR, NR) tiled kernel.

Unlike ``test_glm53_fp8_layer.py`` (which needs the 328 GB GLM-5.3-Flash
checkpoint), this builds random e4m3 block-128 expert weights and checks the
engine against a numpy reference that implements the same semantics:

    gate = clamp(W13_gate @ x, max=LIMIT)          # upper clamp only
    up   = clamp(W13_up   @ x, -LIMIT, +LIMIT)     # both sides
    act  = bf16(silu(gate) * up)
    y    = sum_r w_r * (W2 @ act)

It sweeps M = 1..16 (which is where GLM-5.3-Flash's per-expert row counts live)
so every tile/tail branch of ``matmul_fp8_tiled_range`` is exercised, and it
re-runs one shape for a determinism check.

Usage:
    python scripts/test_fp8_tiled_equiv.py [--limit 10] [--iters 1]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import xiaotu_moe  # noqa: E402


def f32_to_bf16_bits(x):
    v = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    lsb = (v >> 16) & 1
    return ((v + 0x7FFF + lsb) >> 16).astype(np.uint16)


def bf16_bits_to_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


def e4m3_to_f32(u8: np.ndarray) -> np.ndarray:
    """fp8 e4m3fn bit pattern -> float32 (matches the engine's decode)."""
    u = u8.astype(np.uint32)
    sign = (u >> 7) & 1
    exp = (u >> 3) & 0xF
    mant = u & 0x7
    out = np.where(
        exp == 0,
        mant.astype(np.float32) / 8.0 * (2.0**-6),
        (1.0 + mant.astype(np.float32) / 8.0) * (2.0 ** (exp.astype(np.float32) - 7)),
    )
    out = np.where(sign == 1, -out, out).astype(np.float32)
    return np.where((u8 == 0x7F) | (u8 == 0xFF), 0.0, out).astype(np.float32)


def expand_scale(scale: np.ndarray, N: int, K: int, block: int) -> np.ndarray:
    return np.repeat(np.repeat(scale, block, axis=0), block, axis=1)[:N, :K]


def quantize_fp8(x: np.ndarray) -> np.ndarray:
    """Round a float array to e4m3 and return the raw byte patterns."""
    import torch

    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    return t.to(torch.float8_e4m3fn).view(torch.uint8).numpy().copy()


def make_weights(E: int, H: int, I: int, seed: int):
    rng = np.random.default_rng(seed)
    # Keep e4m3 values inside the normal range: a much smaller sigma pushes most
    # elements into the subnormal region, where fp8 quantization error (not the
    # kernel) dominates the comparison.
    w13 = quantize_fp8(rng.standard_normal((E, 2 * I, H)).astype(np.float32) * 0.5)
    w2 = quantize_fp8(rng.standard_normal((E, H, I)).astype(np.float32) * 0.5)
    s13 = (10.0 ** rng.uniform(-0.5, 0.5, size=(E, 2 * I // 128, H // 128))).astype(np.float32)
    s2 = (10.0 ** rng.uniform(-0.5, 0.5, size=(E, H // 128, I // 128))).astype(np.float32)
    return w13, w2, s13, s2


def reference(w13, w2, s13, s2, xf, ids, wts, LIMIT, E, H, I):
    deq13 = e4m3_to_f32(w13) * np.stack(
        [expand_scale(s13[e], 2 * I, H, 128) for e in range(E)]
    )
    deq2 = e4m3_to_f32(w2) * np.stack(
        [expand_scale(s2[e], H, I, 128) for e in range(E)]
    )
    M, K = ids.shape
    golden = np.zeros((M, H), dtype=np.float32)
    for t in range(M):
        for r in range(K):
            e, w = int(ids[t, r]), float(wts[t, r])
            g = deq13[e, :I] @ xf[t]
            u = deq13[e, I:] @ xf[t]
            g = np.minimum(g, LIMIT)
            u = np.clip(u, -LIMIT, LIMIT)
            act = (g / (1.0 + np.exp(-g))) * u
            act_b = bf16_bits_to_f32(f32_to_bf16_bits(act))
            golden[t] += w * (deq2[e] @ act_b)
    return golden


def run_engine(cfg, w13, w2, s13, s2, M, K, ids, wts, x_u16):
    out = np.zeros((M, cfg.hidden_size), dtype=np.float32)
    eng = xiaotu_moe.MOE_FP8(cfg, w13, w2, s13, s2, 0, 0)
    eng.cpu_prefill(M, K, ids, wts, x_u16, out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=float, default=10.0)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--H", type=int, default=1024)
    ap.add_argument("--I", type=int, default=512)
    ap.add_argument("--E", type=int, default=16)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    H, I, E, K = args.H, args.I, args.E, args.K
    LIMIT = args.limit

    w13, w2, s13, s2 = make_weights(E, H, I, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    cfg = xiaotu_moe.MOEConfigV2()
    cfg.num_processes = 1
    cfg.process_id = 0
    cfg.gpu_id = 0
    cfg.has_gate_proj = True
    cfg.expert_num = E
    cfg.top_k = K
    cfg.hidden_size = H
    cfg.intermediate_size = I
    cfg.max_batch_size = 256
    cfg.max_num_seqs = 64
    cfg.stride = 128
    cfg.group_min_len = 10
    cfg.group_max_len = 4096 + 128
    cfg.groupN = 128
    cfg.groupK = 128
    cfg.activation_type = 1
    cfg.swiglu_limit = LIMIT
    cfg.swiglu_alpha = 1.0
    cfg.swiglu_beta = 0.0

    print(f"H={H} I={I} E={E} top_k={K} limit={LIMIT} iters={args.iters}")
    allok = True
    first_out = {}
    for M in list(range(1, 17)):
        ids = rng.integers(0, E, size=(M, K)).astype(np.uint32)
        wts = rng.uniform(0.05, 0.2, size=(M, K)).astype(np.float32)
        x_u16 = f32_to_bf16_bits(
            rng.standard_normal((M, H)).astype(np.float32) * 0.5
        )
        xf = bf16_bits_to_f32(x_u16)
        golden = reference(w13, w2, s13, s2, xf, ids, wts, LIMIT, E, H, I)
        out = run_engine(cfg, w13, w2, s13, s2, M, K, ids, wts, x_u16)
        rms = float(np.sqrt(np.mean(golden**2)))
        rms_rel = float(np.sqrt(np.mean((out - golden) ** 2)) / (rms + 1e-12))
        max_abs = float(np.abs(out - golden).max())
        # The engine rounds `act` to bf16 before the down projection, so the
        # floor is bf16 rounding (~4e-3 relative), not fp32.
        ok = rms_rel < 1e-2 and max_abs < 5e-2 * max(rms, 1e-9)
        allok &= ok
        print(
            f"[{'OK ' if ok else 'BAD'}] M={M:2d} rms_rel={rms_rel:.3e} "
            f"max_abs={max_abs:.3e} max_abs/rms={max_abs/(rms+1e-12):.3e} rms={rms:.4f}"
        )
        first_out[M] = out
        # determinism: a second identical call must be bit-identical
        out2 = run_engine(cfg, w13, w2, s13, s2, M, K, ids, wts, x_u16)
        det = np.array_equal(out, out2)
        if not det:
            print(f"    BAD determinism at M={M}")
        allok &= det

    print("== ALL OK ==" if allok else "== FAILURES ==")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())

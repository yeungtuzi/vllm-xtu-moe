#!/usr/bin/env python
"""End-to-end check: FP8 GPU prefill path vs the CPU engine, same weights.

Builds a real sharded MOE_FP8 engine from synthetic e4m3/block-128 weights, runs
one layer on the CPU engine and on the GPU path (kmajor_from_engine_shards_fp8 +
gpu_moe_layer_fp8), and compares.

Expected difference: the GPU kernels round ``dequant(w) * scale`` to bf16 before
the MMA (A100 has no FP8 tensor cores, so bf16 is the compute type), while the
CPU engine folds the fp32 scale exactly. That is a ~0.4% relative perturbation of
the weights, so the comparison bound is ~1e-2 rms relative, not 1e-3.

Usage: python scripts/test_gpu_prefill_fp8_vs_cpu.py [E H I T K]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import xiaotu_moe  # noqa: E402
from vllm_xiaotu_moe.gpu_prefill_fp8 import (  # noqa: E402
    gpu_moe_layer_fp8,
    kmajor_from_engine_shards_fp8,
)

BLOCK = 128
LIMIT = 10.0


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    args = [int(x) for x in sys.argv[1:]]
    E, H, I, T, K = args if len(args) == 5 else (8, 2048, 1024, 64, 8)
    dev = torch.device("cuda:0")
    g = np.random.default_rng(0)

    # Realistic magnitudes: random BYTES include e4m3's 448 exponent and make the
    # layer output ~1e5, where any 0.4% operand perturbation is meaningless.
    def q(shape):
        u8 = (torch.randn(shape, generator=torch.Generator().manual_seed(1)) * 0.1)\
            .to(torch.float8_e4m3fn).view(torch.uint8).numpy().copy()
        u8[(u8 == 0x7F) | (u8 == 0xFF)] = 0
        return u8
    w13 = q((E, 2 * I, H))
    w2 = q((E, H, I))
    s13 = (10.0 ** g.uniform(-0.5, 0.5, size=(E, 2 * I // BLOCK, H // BLOCK))).astype(np.float32)
    s2 = (10.0 ** g.uniform(-0.5, 0.5, size=(E, H // BLOCK, I // BLOCK))).astype(np.float32)

    cfg = xiaotu_moe.MOEConfigV2()
    for k, v in dict(num_processes=1, process_id=0, gpu_id=0, has_gate_proj=True,
                     expert_num=E, top_k=K, hidden_size=H, intermediate_size=I,
                     max_batch_size=256, max_num_seqs=64, stride=BLOCK,
                     group_min_len=10, group_max_len=4096, groupN=BLOCK, groupK=BLOCK,
                     activation_type=1, swiglu_limit=LIMIT, swiglu_alpha=1.0,
                     swiglu_beta=0.0).items():
        setattr(cfg, k, v)
    eng = xiaotu_moe.MOE_FP8(cfg, w13, w2, s13, s2, 0, 0)
    print(f"E={E} H={H} I={I} T={T} K={K} shards={eng.shard_geometry()['ns']}")

    xf = (g.standard_normal((T, H)) * 0.5).astype(np.float32)
    x_u16 = (xf.view(np.uint32) >> 16).astype(np.uint16)
    ids = g.integers(0, E, size=(T, K)).astype(np.uint32)
    wts = g.uniform(0.05, 0.2, size=(T, K)).astype(np.float32)

    cpu_out = np.zeros((T, H), dtype=np.float32)
    eng.cpu_prefill(T, K, ids, wts, x_u16, cpu_out)

    built = kmajor_from_engine_shards_fp8(eng, dev, H, I, E)
    assert built is not None, "assembly returned None (engine not sharded?)"
    x_b = torch.from_numpy(x_u16).to(dev).view(torch.bfloat16).reshape(T, H)
    gpu_out = gpu_moe_layer_fp8(
        x_b, torch.from_numpy(ids).to(dev).to(torch.int32),
        torch.from_numpy(wts).to(dev), *built, H, I, K, device=dev,
        swiglu_limit=LIMIT,
    )
    torch.cuda.synchronize()

    a = torch.from_numpy(cpu_out)
    b = gpu_out.float().cpu()

    def rel(x, y):
        d = (x - y).abs()
        rms = y.pow(2).mean().sqrt().item()
        return ((x - y).pow(2).mean().sqrt() / (rms + 1e-12)).item(), d.max().item(), rms

    rms_rel, max_abs, rms = rel(b, a)

    # Decompose: a torch "ideal" with EXACT fp32 dequant + exact fp32 scale fold
    # and fp32 gate/up (the CPU engine's semantics). Then
    #   CPU vs ideal  -> does the CPU engine match its own spec?
    #   GPU vs ideal  -> how much is the GPU's bf16 operand rounding?
    ideal = None
    if T * K * (2 * I * H + H * I) <= 3e11:
        def deq(w_u8, sc, nb, kb):
            u = torch.from_numpy(w_u8).to(torch.int32)
            sign = (u >> 7) & 1
            exp = (u >> 3) & 0xF
            man = u & 0x7
            mant = man.float() * 0.125
            val = torch.where(exp != 0,
                              torch.exp2((exp - 7).float()) * (1.0 + mant),
                              torch.full_like(mant, 0.015625) * mant)
            val = torch.where(sign != 0, -val, val)
            scb = torch.from_numpy(sc).repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
            return val * scb
        # The engine consumes bf16 activations (x_u16), so the ideal must too.
        xa = torch.from_numpy(x_u16).view(torch.bfloat16).reshape(T, H).float()
        ideal = torch.zeros(T, H)
        for t in range(T):
            acc = torch.zeros(H)
            for s in range(K):
                e = int(ids[t, s])
                if wts[t, s] == 0:
                    continue
                d13 = deq(w13[e], s13[e], 2 * I // BLOCK, H // BLOCK)
                g = xa[t] @ d13[:I].T
                u_ = xa[t] @ d13[I:].T
                g = torch.clamp(g, max=LIMIT)
                u_ = torch.clamp(u_, -LIMIT, LIMIT)
                act = ((g / (1 + torch.exp(-g))) * u_).to(torch.bfloat16).float()
                d2 = deq(w2[e], s2[e], H // BLOCK, I // BLOCK)
                y = act @ d2.T
                acc += float(wts[t, s]) * y
            ideal[t] = acc

    if ideal is not None:
        c_i = rel(a, ideal)
        g_i = rel(b, ideal)
        print(f"    CPU vs ideal: rms_rel={c_i[0]:.3e} max_abs={c_i[1]:.3e}")
        print(f"    GPU vs ideal: rms_rel={g_i[0]:.3e} max_abs={g_i[1]:.3e}")

    ok = rms_rel < 2e-2
    print(f"[{'OK ' if ok else 'BAD'}] GPU prefill vs CPU engine: "
          f"rms_rel={rms_rel:.3e} max_abs={max_abs:.3e} ref_rms={rms:.4f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

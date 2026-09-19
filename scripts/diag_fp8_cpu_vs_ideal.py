#!/usr/bin/env python
"""Isolate where the CPU FP8 MoE engine departs from exact fp32 dequant math.

Builds an engine from synthetic e4m3 weights and compares `cpu_prefill` against a
torch "ideal" (bit-exact e4m3 decode, single fp32 scale fold, fp32 accumulate,
bf16 activation rounding) under progressively more realistic settings:

  A  unit scales, LIMIT=0, T=1 K=1 wts=1   -> pure gemm semantics
  B  + random s13 only
  C  + random s2 only
  D  unit scales + LIMIT=10
  E  full (random scales + LIMIT=10 + K=4)

A mismatch in A means the kernel mapping (rows/scales/act) is wrong; a mismatch
that appears only in B or C points at a scale-table layout; only D points at the
activation.

Usage: python scripts/diag_fp8_cpu_vs_ideal.py [E H I T K]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import xiaotu_moe  # noqa: E402

BLOCK = 128


def deq_t(w_u8, sc):
    """Bit-exact e4m3 -> fp32 times the block scale (torch, fp32)."""
    u = torch.from_numpy(w_u8).to(torch.int32)
    sign = (u >> 7) & 1
    exp = (u >> 3) & 0xF
    man = u & 0x7
    mant = man.float() * 0.125
    val = torch.where(exp != 0,
                      torch.exp2((exp - 7).float()) * (1.0 + mant),
                      torch.full_like(mant, 0.015625) * mant)
    val = torch.where(sign != 0, -val, val)
    if sc is None:
        return val
    scb = torch.from_numpy(sc).repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    return val * scb


def run(tag, *, E, H, I, T, K, rnd_s13, rnd_s2, limit, seed=0):
    g = np.random.default_rng(seed)
    def q(shape):
        u8 = (torch.randn(shape, generator=torch.Generator().manual_seed(1)) * 0.1)\
            .to(torch.float8_e4m3fn).view(torch.uint8).numpy().copy()
        u8[(u8 == 0x7F) | (u8 == 0xFF)] = 0
        return u8
    w13 = q((E, 2 * I, H))
    w2 = q((E, H, I))
    s13 = ((10.0 ** g.uniform(-0.5, 0.5, size=(E, 2 * I // BLOCK, H // BLOCK)))
           .astype(np.float32) if rnd_s13 else np.ones((E, 2 * I // BLOCK, H // BLOCK), np.float32))
    s2 = ((10.0 ** g.uniform(-0.5, 0.5, size=(E, H // BLOCK, I // BLOCK)))
          .astype(np.float32) if rnd_s2 else np.ones((E, H // BLOCK, I // BLOCK), np.float32))

    cfg = xiaotu_moe.MOEConfigV2()
    for k, v in dict(num_processes=1, process_id=0, gpu_id=0, has_gate_proj=True,
                     expert_num=E, top_k=K, hidden_size=H, intermediate_size=I,
                     max_batch_size=256, max_num_seqs=64, stride=BLOCK,
                     group_min_len=10, group_max_len=4096, groupN=BLOCK, groupK=BLOCK,
                     activation_type=1, swiglu_limit=limit, swiglu_alpha=1.0,
                     swiglu_beta=0.0).items():
        setattr(cfg, k, v)
    eng = xiaotu_moe.MOE_FP8(cfg, w13, w2, s13, s2, 0, 0)

    xf = (g.standard_normal((T, H)) * 0.5).astype(np.float32)
    x_u16 = (xf.view(np.uint32) >> 16).astype(np.uint16)
    ids = g.integers(0, E, size=(T, K)).astype(np.uint32)
    wts = np.ones((T, K), np.float32) if tag == "A" else g.uniform(0.05, 0.2, (T, K)).astype(np.float32)

    cpu = np.zeros((T, H), np.float32)
    eng.cpu_prefill(T, K, ids, wts, x_u16, cpu)

    xa = torch.from_numpy(x_u16).view(torch.bfloat16).reshape(T, H).float()
    ideal = torch.zeros(T, H)
    d13c = {e: deq_t(w13[e], None if not rnd_s13 else s13[e]) for e in range(E)}
    d2c = {e: deq_t(w2[e], None if not rnd_s2 else s2[e]) for e in range(E)}
    for t in range(T):
        acc = torch.zeros(H)
        for s in range(K):
            e = int(ids[t, s])
            d13, d2 = d13c[e], d2c[e]
            gt = xa[t] @ d13[:I].T
            up = xa[t] @ d13[I:].T
            if limit > 0:
                gt = torch.clamp(gt, max=limit)
                up = torch.clamp(up, -limit, limit)
            act = ((gt / (1 + torch.exp(-gt))) * up).to(torch.bfloat16).float()
            acc += float(wts[t, s]) * (act @ d2.T)
        ideal[t] = acc

    a = torch.from_numpy(cpu)
    d = (a - ideal)
    rms_rel = (d.pow(2).mean().sqrt() / (ideal.pow(2).mean().sqrt() + 1e-12)).item()
    print(f"  {tag}: rms_rel={rms_rel:.3e}  max_abs={d.abs().max().item():.3e}  "
          f"ref_rms={ideal.pow(2).mean().sqrt().item():.4f}  "
          f"cpu_rms={a.pow(2).mean().sqrt().item():.4f}")
    return rms_rel


def run_identity(tag, *, H, T=1, K=1, id13=True, id2=True, limit=0.0):
    """H==I, w2/ w13 optionally e4m3 identities (0x38 == 1.0), unit scales.

    id2=True makes the down projection an exact copy of the activation, so any
    residual error must come from the gate/up phase. id13=True additionally makes
    gate == up == bf16(x), pinning the test on the activation + row layout.
    """
    E, I = 1, H
    w13 = np.zeros((E, 2 * I, H), np.uint8)
    for j in range(2 * I):
        if id13:
            w13[0, j, j % I] = 0x38
        else:
            w13[0, j] = (torch.randn(H, generator=torch.Generator().manual_seed(7)) * 0.1)\
                .to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    w2 = np.zeros((E, H, I), np.uint8)
    for h in range(H):
        if id2:
            w2[0, h, h] = 0x38
        else:
            w2[0, h] = (torch.randn(I, generator=torch.Generator().manual_seed(8)) * 0.1)\
                .to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    s13 = np.ones((E, 2 * I // BLOCK, H // BLOCK), np.float32)
    s2 = np.ones((E, H // BLOCK, I // BLOCK), np.float32)
    cfg = xiaotu_moe.MOEConfigV2()
    for k, v in dict(num_processes=1, process_id=0, gpu_id=0, has_gate_proj=True,
                     expert_num=E, top_k=K, hidden_size=H, intermediate_size=I,
                     max_batch_size=256, max_num_seqs=64, stride=BLOCK,
                     group_min_len=10, group_max_len=4096, groupN=BLOCK, groupK=BLOCK,
                     activation_type=1, swiglu_limit=limit, swiglu_alpha=1.0,
                     swiglu_beta=0.0).items():
        setattr(cfg, k, v)
    eng = xiaotu_moe.MOE_FP8(cfg, w13, w2, s13, s2, 0, 0)
    g = np.random.default_rng(3)
    xf = (g.standard_normal((T, H)) * 0.5).astype(np.float32)
    x_u16 = (xf.view(np.uint32) >> 16).astype(np.uint16)
    ids = np.zeros((T, K), np.uint32)
    wts = np.ones((T, K), np.float32)
    cpu = np.zeros((T, H), np.float32)
    eng.cpu_prefill(T, K, ids, wts, x_u16, cpu)
    xa = torch.from_numpy(x_u16).view(torch.bfloat16).reshape(T, H).float()
    if id13:
        gt = up = xa
    else:
        d13 = deq_t(w13[0], None)
        gt, up = xa @ d13[:I].T, xa @ d13[I:].T
    if limit > 0:
        gt = torch.clamp(gt, max=limit)
        up = torch.clamp(up, -limit, limit)
    act = ((gt / (1 + torch.exp(-gt))) * up).to(torch.bfloat16).float()
    ideal = act if id2 else act @ deq_t(w2[0], None).T
    a = torch.from_numpy(cpu)
    d = a - ideal
    rr = (d.pow(2).mean().sqrt() / (ideal.pow(2).mean().sqrt() + 1e-12)).item()
    print(f"  {tag}: rms_rel={rr:.3e}  max_abs={d.abs().max().item():.3e}  "
          f"ref_rms={ideal.pow(2).mean().sqrt().item():.4f}")
    # first few entries, for eyeballing
    print(f"      cpu[:4]  ={a[0,:4].tolist()}")
    print(f"      ideal[:4]={ideal[0,:4].tolist()}")
    return rr


def main() -> int:
    args = [int(x) for x in sys.argv[1:]]
    E, H, I, T, K = args if len(args) == 5 else (4, 1024, 512, 16, 4)
    print(f"E={E} H={H} I={I} T={T} K={K}")
    base = dict(E=E, H=H, I=I, T=T, K=K)
    run("A unit-scale/limit=0/K=1", rnd_s13=False, rnd_s2=False, limit=0.0,
        **{**base, "T": 1, "K": 1})
    run("B +s13", rnd_s13=True, rnd_s2=False, limit=0.0, **{**base, "T": 1, "K": 1})
    run("C +s2 ", rnd_s13=False, rnd_s2=True, limit=0.0, **{**base, "T": 1, "K": 1})
    run("D unit+limit=10", rnd_s13=False, rnd_s2=False, limit=10.0, **{**base, "T": 1, "K": 1})
    run("E full", rnd_s13=True, rnd_s2=True, limit=10.0, **base)
    print("identity probes (H=I=512):")
    run_identity("F down=identity", H=512)
    run_identity("G gate/up/down=identity", H=512)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

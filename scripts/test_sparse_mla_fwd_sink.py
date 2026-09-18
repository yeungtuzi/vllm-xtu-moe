#!/usr/bin/env python
"""Numerical check for ``sparse_mla_fwd_with_sink`` (SM8x portable Triton path).

Builds synthetic (q, gathered kv, topk indices, topk_length, attn_sink) cases and
compares the Triton kernel against a float32 torch reference that implements the
exact same math: an online softmax over the gathered rows with the attention sink
folded in as one extra logit whose value contribution is zero.

The kernel is the SM80 replacement for the SM90+ ``flash_mla_sparse_fwd``
contract (flat row ids into a shared kv table + per-query valid length), used by
the GLM-5.3-Flash NoPE sparse-MLA (DSA) layers on Ampere.

Usage:
  python scripts/test_sparse_mla_fwd_sink.py            # all cases
  python scripts/test_sparse_mla_fwd_sink.py --device cuda:2
"""
from __future__ import annotations

import argparse

import torch

from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    sparse_mla_fwd_with_sink,
)


def reference(
    q: torch.Tensor,          # (Tq, H, D) fp32
    kv: torch.Tensor,         # (R, D) fp32
    indices: torch.Tensor,    # (Tq, K) int64 (already clamped to >=0 where valid)
    topk_length: torch.Tensor,  # (Tq,) int64
    scale: float,
    attn_sink: torch.Tensor,  # (H,) fp32
) -> torch.Tensor:
    tq, h, d = q.shape
    k = indices.shape[1]
    out = torch.zeros((tq, h, d), dtype=torch.float32, device=q.device)
    pos = torch.arange(k, device=q.device)[None, :]
    valid = pos < topk_length[:, None]  # (Tq, K)
    for t in range(tq):
        rows = indices[t]  # (K,)
        kv_rows = kv[rows.clamp(min=0)]  # (K, D)
        for hh in range(h):
            s = (kv_rows @ q[t, hh]) * scale  # (K,)
            s = s.masked_fill(~valid[t], float("-inf"))
            m = torch.maximum(attn_sink[hh], s.max())
            # sink contributes exp(attn_sink - m) to the denominator, 0 to acc
            w = torch.exp(s - m)
            w = torch.where(valid[t], w, torch.zeros_like(w))
            denom = torch.exp(attn_sink[hh] - m) + w.sum()
            out[t, hh] = (w[:, None] * kv_rows).sum(0) / denom
    return out


def make_case(
    tq: int, h: int, d: int, k: int, r: int, frac_valid: float,
    device: str, seed: int,
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn((tq, h, d), generator=g) * 0.5
    kv = torch.randn((r, d), generator=g)
    # row ids: valid prefix are distinct random rows; padding is -1
    idx = torch.full((tq, k), -1, dtype=torch.int32)
    lens = torch.clamp(
        (torch.rand((tq,), generator=g) * frac_valid * k).to(torch.int32) + 1,
        1, k,
    )
    for t in range(tq):
        n = int(lens[t])
        idx[t, :n] = torch.randint(0, r, (n,), generator=g, dtype=torch.int32)
    sink = torch.randn((h,), generator=g) * 2.0
    scale = 1.0 / (d ** 0.5)
    return (
        q.to(device), kv.to(device), idx.to(device), lens.to(device),
        sink.to(device), scale,
    )


def run_case(name, tq, h, d, k, r, frac_valid, device, seed, sink_value=None):
    q, kv, idx, lens, sink, scale = make_case(
        tq, h, d, k, r, frac_valid, device, seed
    )
    if sink_value is not None:
        sink = torch.full_like(sink, sink_value)
    # The kernel consumes bf16 q/kv, so the reference must see the very same
    # bf16-rounded operands; otherwise input rounding, not kernel math, dominates
    # the difference. Element-wise relative error is meaningless here (outputs may
    # be arbitrarily close to zero), so use the peak-normalized error.
    q_b = q.to(torch.bfloat16)
    kv_b = kv.to(torch.bfloat16)
    ref = reference(
        q_b.float(), kv_b.float(), idx.long(), lens.long(), scale, sink
    )

    out = torch.empty((tq, h, d), dtype=torch.bfloat16, device=device)
    sparse_mla_fwd_with_sink(
        q=q_b, kv=kv_b, indices=idx,
        topk_length=lens, scale=scale, attn_sink=sink, output=out,
    )
    got = out.float()
    diff = (got - ref).abs()
    max_abs = diff.max().item()
    max_rel = (max_abs / (ref.abs().max().item() + 1e-12))
    ok = max_abs <= 2e-2 and max_rel <= 2e-2
    print(
        f"[{'OK ' if ok else 'BAD'}] {name:34s} Tq={tq:4d} H={h:3d} D={d:4d} "
        f"K={k:5d} R={r:5d} valid~{frac_valid:.2f}  "
        f"max_abs={max_abs:.4e} peak_rel={max_rel:.4e}"
    )
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    dev = args.device
    print(f"device={dev} torch={torch.__version__}")
    cases = [
        # GLM-5.3-Flash head geometry: qk_nope=256 + rope=0 => D = kv_lora 512
        ("glm nope head 512, full", 8, 64, 512, 128, 4096, 1.00),
        ("glm nope head 512, partial", 8, 64, 512, 128, 4096, 0.60),
        ("v3.2 head 576 (nope+rope)", 4, 64, 576, 64, 2048, 0.50),
        ("index_topk=2048", 2, 64, 512, 2048, 8192, 0.30),
        ("single token, single head", 1, 1, 512, 16, 64, 1.00),
        ("many tokens", 64, 64, 512, 256, 4096, 0.80),
        ("heads padded (actual<H)", 4, 64, 512, 128, 1024, 0.75),
    ]
    allok = True
    for i, c in enumerate(cases):
        allok &= run_case(*c, dev, args.seed + i)
    # sink extremes: a strongly negative sink must reduce to plain softmax
    allok &= run_case(
        "sink = -1e30 (no sink)", 4, 64, 512, 128, 2048, 0.7, dev, 99,
        sink_value=-1e30,
    )
    # determinism: same input three times
    q, kv, idx, lens, sink, scale = make_case(4, 64, 512, 128, 2048, 0.7, dev, 7)
    outs = []
    for _ in range(3):
        o = torch.empty((4, 64, 512), dtype=torch.bfloat16, device=dev)
        sparse_mla_fwd_with_sink(
            q=q.to(torch.bfloat16), kv=kv.to(torch.bfloat16), indices=idx,
            topk_length=lens, scale=scale, attn_sink=sink, output=o,
        )
        outs.append(o.clone())
    det = all(torch.equal(outs[0], o) for o in outs[1:])
    print(f"[{'OK ' if det else 'BAD'}] determinism (3 identical runs): "
          f"{'bit-identical' if det else 'DIFFERS'}")
    allok &= det
    print("== ALL OK ==" if allok else "== FAILURES ==")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())

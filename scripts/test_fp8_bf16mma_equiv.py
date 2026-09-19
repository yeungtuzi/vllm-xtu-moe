#!/usr/bin/env python
"""A/B gate for the opt-in AVX512-BF16 (`vdpbf16ps`) FP8 tile.

Runs the same engine call with `XIAOTU_MOE_FP8_BF16_MMA` off and on and reports
the delta. The two paths are NOT expected to be bit-identical: the bf16 tile keeps
`e4m3 * scale` in bf16 where the fp32 tile keeps it in fp32, so the expert weights
are perturbed by ~0.2-0.4%. That is the documented cost of the flag (see
dev-docs/GLM53_SM80_PLAN.md §18.5) and why it is opt-in.

The two configurations run in **separate processes**: the engine caches the flag
in a function-local static at first use (same pattern as XIAOTU_MOE_FUSE_A2B), so
an in-process flip would silently measure the same path twice -- the exact trap
this repo has been bitten by before.

Bounds:
  * determinism: two runs in the SAME process must be bit-identical;
  * fp32 vs bf16-mma rms relative error <= 1e-2 (the bf16 operand rounding).

Usage: python scripts/test_fp8_bf16mma_equiv.py [E H I T K]
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

BLOCK = 128
LIMIT = 10.0


def build(E, H, I, K, seed=0):
    import xiaotu_moe
    g = np.random.default_rng(seed)
    def q(shape):
        u8 = (torch.randn(shape, generator=torch.Generator().manual_seed(7)) * 0.1)\
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
                     max_batch_size=512, max_num_seqs=64, stride=BLOCK,
                     group_min_len=10, group_max_len=4096, groupN=BLOCK, groupK=BLOCK,
                     activation_type=1, swiglu_limit=LIMIT, swiglu_alpha=1.0,
                     swiglu_beta=0.0).items():
        setattr(cfg, k, v)
    return xiaotu_moe.MOE_FP8(cfg, w13, w2, s13, s2, 0, 0)


def run_once(E, H, I, T, K, seed=1):
    """One engine layer on deterministic synthetic data."""
    import xiaotu_moe  # noqa: F811  (imported after the env is set by the caller)
    eng = build(E, H, I, K)
    g = np.random.default_rng(seed)
    xf = (g.standard_normal((T, H)) * 0.5).astype(np.float32)
    x_u16 = (xf.view(np.uint32) >> 16).astype(np.uint16)
    ids = g.integers(0, E, size=(T, K)).astype(np.uint32)
    wts = g.uniform(0.05, 0.2, size=(T, K)).astype(np.float32)
    out = np.zeros((T, H), dtype=np.float32)
    eng.cpu_prefill(T, K, ids, wts, x_u16, out)
    return out


def child(out_path: str, E, H, I, T, K) -> int:
    """Subprocess entry point: run once with whatever env the parent set."""
    np.save(out_path, run_once(E, H, I, T, K))
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _, _, path, E, H, I, T, K = sys.argv
        return child(path, *[int(v) for v in (E, H, I, T, K)])

    args = [int(x) for x in sys.argv[1:]]
    E, H, I, T, K = args if len(args) == 5 else (8, 2048, 1024, 128, 8)
    print(f"E={E} H={H} I={I} T={T} K={K}")
    env = dict(os.environ)
    env.pop("XIAOTU_MOE_FP8_BF16_MMA", None)
    with tempfile.TemporaryDirectory() as td:
        def spawn(name, e):
            p = os.path.join(td, f"{name}.npy")
            r = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--child", p,
                 str(E), str(H), str(I), str(T), str(K)],
                env=e, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-2000:]); print(r.stderr[-2000:])
                raise SystemExit(f"child failed for {name}")
            return np.load(p)

        off1 = spawn("off1", env)
        off2 = spawn("off2", env)                    # determinism, same config
        e_on = dict(env); e_on["XIAOTU_MOE_FP8_BF16_MMA"] = "1"
        on = spawn("on", e_on)

    det = float(np.abs(off1 - off2).max())
    rms_ref = float(np.sqrt((off1 ** 2).mean()))
    d = np.abs(on - off1)
    rms_rel = float(np.sqrt(((on - off1) ** 2).mean()) / (rms_ref + 1e-12))
    per_elem = d / (np.abs(off1) + 1e-3 * rms_ref)
    print(f"  fp32 path determinism : max_abs_diff={det:.3e} (must be 0)")
    print(f"  fp32 vs bf16-mma      : rms_rel={rms_rel:.3e} max_abs={d.max():.3e} "
          f"p99.9_rel={np.percentile(per_elem, 99.9):.3e}")
    ok = det == 0.0 and rms_rel < 1e-2
    print(f"[{'OK ' if ok else 'BAD'}] bf16-MMA is an opt-in switch with the "
          f"documented bf16-operand tolerance")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

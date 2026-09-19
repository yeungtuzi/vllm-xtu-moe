#!/usr/bin/env python
"""Validate kmajor_from_engine_shards_fp8 against the weights the engine got.

Builds a real (sharded) MOE_FP8 engine from synthetic e4m3 weights, asks the FP8
assembly for K-major device tensors, and compares byte-for-byte (weights) /
exactly (scales) against the expected K-major transpose of the source arrays.

This is the piece that differs structurally from MXFP4: 1-byte rows and a
[N/128, K/128] fp32 block scale instead of [N, K/32] e8m0.

Usage: python scripts/test_gpu_prefill_fp8_assembly.py [E H I]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import xiaotu_moe  # noqa: E402
from vllm_xiaotu_moe.byte_transpose import ktranspose_bytes  # noqa: E402
from vllm_xiaotu_moe.gpu_prefill_fp8 import (  # noqa: E402
    kmajor_from_engine_shards_fp8,
)

BLOCK = 128


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    args = [int(x) for x in sys.argv[1:]]
    E, H, I = args if len(args) == 3 else (4, 512, 256)
    dev = torch.device("cuda:0")
    g = np.random.default_rng(0)

    w13 = g.integers(0, 256, size=(E, 2 * I, H), dtype=np.uint8)
    w2 = g.integers(0, 256, size=(E, H, I), dtype=np.uint8)
    s13 = (10.0 ** g.uniform(-2, 1, size=(E, 2 * I // BLOCK, H // BLOCK))).astype(np.float32)
    s2 = (10.0 ** g.uniform(-2, 1, size=(E, H // BLOCK, I // BLOCK))).astype(np.float32)

    cfg = xiaotu_moe.MOEConfigV2()
    for k, v in dict(num_processes=1, process_id=0, gpu_id=0, has_gate_proj=True,
                     expert_num=E, top_k=1, hidden_size=H, intermediate_size=I,
                     max_batch_size=64, max_num_seqs=8, stride=BLOCK,
                     group_min_len=10, group_max_len=1024, groupN=BLOCK, groupK=BLOCK,
                     activation_type=1, swiglu_limit=10.0, swiglu_alpha=1.0,
                     swiglu_beta=0.0).items():
        setattr(cfg, k, v)
    eng = xiaotu_moe.MOE_FP8(cfg, w13, w2, s13, s2, 0, 0)
    geo = eng.shard_geometry()
    print(f"E={E} H={H} I={I} shards ns={geo['ns']} "
          f"w13_crows={geo['w13_crows']} w13_cbytes={geo['w13_cbytes']} "
          f"w2_crows={geo['w2_crows']} w2_cbytes={geo['w2_cbytes']}")
    if int(geo["ns"]) < 2:
        raise SystemExit("engine did not shard; cannot test the assembly path")

    built = kmajor_from_engine_shards_fp8(eng, dev, H, I, E)
    assert built is not None, "assembly returned None"
    w13t, s13t, w2t, s2t = built

    # Expected K-major, computed on the host from the SOURCE arrays.
    exp_w13t = torch.from_numpy(w13).transpose(1, 2).contiguous().to(dev)
    exp_w2t = torch.from_numpy(w2).transpose(1, 2).contiguous().to(dev)
    exp_s13t = torch.from_numpy(s13).transpose(1, 2).contiguous().to(dev)
    exp_s2t = torch.from_numpy(s2).transpose(1, 2).contiguous().to(dev)

    torch.cuda.synchronize()
    r = {}
    r["w13t"] = bool(torch.equal(w13t, exp_w13t))
    r["w2t"] = bool(torch.equal(w2t, exp_w2t))
    r["s13t"] = float((s13t - exp_s13t).abs().max().item())
    r["s2t"] = float((s2t - exp_s2t).abs().max().item())
    r["shape13"] = tuple(w13t.shape) == (E, H, 2 * I)
    r["shape2"] = tuple(w2t.shape) == (E, I, H)
    ok = r["w13t"] and r["w2t"] and r["shape13"] and r["shape2"] \
        and r["s13t"] == 0.0 and r["s2t"] == 0.0
    print(f"[{'OK ' if ok else 'BAD'}] assembly: w13t_bytes_equal={r['w13t']} "
          f"w2t_bytes_equal={r['w2t']} shapes_ok={r['shape13'] and r['shape2']} "
          f"scale_maxdiff=({r['s13t']:.1e}, {r['s2t']:.1e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

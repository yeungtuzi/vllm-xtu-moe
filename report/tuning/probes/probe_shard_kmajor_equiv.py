#!/usr/bin/env python
"""Verify `kmajor_from_engine_shards` reproduces the canonical layout BIT-EXACTLY.

The GPU prefill path can stream the engine's own compact NUMA shards instead of the
checkpoint source (NOTES §459), which removes the need for
XIAOTU_RELEASE_SOURCE=0. That only works if reassembling the shards on the device
yields exactly the bytes `_kmajor_bytes(source)` would have produced. This checks
that directly, with real weights, no service needed.

Run: XIAOTU_LAYER1_NPZ=<ckpt> python probe_shard_kmajor_equiv.py
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, HERE)

from probe_gpu_prefill_v41 import E, GK, H, I, K, load_layer  # noqa: E402


def main() -> int:
    import xiaotu_moe
    from vllm_xiaotu_moe.gpu_prefill import _kmajor_bytes, kmajor_from_engine_shards

    layer = int(os.environ.get("LAYER", "3"))
    dev = torch.device("cuda:0")
    print(f"[equiv] H={H} I={I} E={E} K={K} GK={GK} layer={layer}", flush=True)
    w13, w2, s13, s2 = load_layer(layer)

    m = xiaotu_moe.load()
    cfg = m.MOEConfigV2()
    cfg.num_processes = 1; cfg.process_id = 0; cfg.gpu_id = 0
    cfg.has_gate_proj = True; cfg.expert_num = E; cfg.top_k = K
    cfg.hidden_size = H; cfg.intermediate_size = I
    cfg.max_batch_size = 1024; cfg.max_num_seqs = 256
    cfg.stride = 32; cfg.group_min_len = 10; cfg.group_max_len = 1152
    cfg.groupN = 1; cfg.groupK = GK; cfg.activation_type = 0
    eng = m.MOE_MXFP4(cfg, w13, w2, s13, s2, 0, 0)
    geo = eng.shard_geometry()
    print(f"[equiv] shard_geometry = {dict(geo)}", flush=True)
    if int(geo["ns"]) < 2:
        print("[equiv] engine has no shards (NOSHARD?) -- nothing to verify")
        return 0

    # reference: exactly what the old path did (canonical source -> device -> K-major)
    ref = tuple(
        _kmajor_bytes(torch.from_numpy(a).to(dev))
        for a in (w13, s13, w2, s2)
    )
    got = kmajor_from_engine_shards(eng, dev, H, I, E, GK)
    if got is None:
        print("[equiv] kmajor_from_engine_shards returned None -- FAIL")
        return 1

    names = ("w13", "s13", "w2", "s2")
    ok = True
    for nm, r, g in zip(names, ref, got):
        same_shape = tuple(r.shape) == tuple(g.shape)
        eq = same_shape and bool(torch.equal(r, g))
        n_diff = 0 if eq else int((r.view(-1)[:g.numel()] != g.view(-1)).sum()) if same_shape else -1
        print(f"  {nm:4s} ref{tuple(r.shape)} got{tuple(g.shape)} "
              f"bit_equal={eq} n_diff={n_diff}", flush=True)
        ok = ok and eq
    # also make sure the shards carry non-trivial data (not all zeros)
    _sl = got[0].view(-1)[:1_000_000]
    nz = int((_sl != 0).sum())
    print(f"  w13 nonzero bytes = {nz} / {got[0].numel()} "
          f"({100.0*nz/max(1,_sl.numel()):.1f}% of first 1e6)", flush=True)
    print("VERDICT:", "bit-exact ✓" if ok else "MISMATCH ✗", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

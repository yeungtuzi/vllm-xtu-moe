#!/usr/bin/env python
"""Where does GPU prefill time actually go for V4.1, and which knobs help?

The service measured ~1050-1800 ms/layer at 7000 tokens, while my arithmetic from
the parts (shard assembly + 6.72 GiB H2D + kernels) predicted ~800. So measure the
parts instead of predicting them, and sweep the kernel knobs:

  * `XIAOTU_GPU_PREFILL_LUT=1` -- the default (0) dequantizes each nibble with
    exp2 math in-kernel; the LUT branch is a 16-entry gather. Worth checking.
  * BM/BN/BK/BH tiles and STAGES (software pipelining depth).
  * WARPS.

Also times `kmajor_from_engine_shards` (the assembly) separately from
`gpu_moe_layer`, so the split is explicit.

Run: XIAOTU_LAYER1_NPZ=<ckpt> MS="2048 7000" python probe_gpu_prefill_breakdown.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, HERE)

from probe_gpu_prefill_v41 import E, GK, H, I, K, f32_to_bf16_bits, load_layer  # noqa: E402


def main() -> int:
    import xiaotu_moe
    from vllm_xiaotu_moe.gpu_prefill import (
        PrefetchSlot,
        _kmajor_bytes,
        gpu_moe_layer,
        kmajor_from_engine_shards,
    )

    layer = int(os.environ.get("LAYER", "3"))
    Ms = [int(x) for x in os.environ.get("MS", "2048 7000").split()]
    reps = int(os.environ.get("REP", "3"))
    dev = torch.device("cuda:0")
    print(f"[bk] H={H} I={I} E={E} K={K} GK={GK} layer={layer}", flush=True)

    w13, w2, s13, s2 = load_layer(layer)
    m = xiaotu_moe.load()
    cfg = m.MOEConfigV2()
    cfg.num_processes = 1; cfg.process_id = 0; cfg.gpu_id = 0
    cfg.has_gate_proj = True; cfg.expert_num = E; cfg.top_k = K
    cfg.hidden_size = H; cfg.intermediate_size = I
    cfg.max_batch_size = max(Ms); cfg.max_num_seqs = 256
    cfg.stride = 32; cfg.group_min_len = 10; cfg.group_max_len = max(Ms) + 128
    cfg.groupN = 1; cfg.groupK = GK; cfg.activation_type = 0
    eng = m.MOE_MXFP4(cfg, w13, w2, s13, s2, 0, 0)

    # ---- (a) assembly: shard path (what the plugin uses) -------------------
    for _ in range(2):
        km = kmajor_from_engine_shards(eng, dev, H, I, E, GK)
        torch.cuda.synchronize()
        del km
        torch.cuda.empty_cache()
    ta = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter()
        km = kmajor_from_engine_shards(eng, dev, H, I, E, GK)
        torch.cuda.synchronize(); ta.append(time.perf_counter() - t)
        del km
        torch.cuda.empty_cache()
    asm_ms = min(ta) * 1e3
    print(f"[bk] assembly (shards -> K-major): {asm_ms:.1f} ms/layer", flush=True)
    km = kmajor_from_engine_shards(eng, dev, H, I, E, GK)
    slot = PrefetchSlot(); slot.bufs = km
    slot.ready = torch.cuda.Event(); slot.ready.record(torch.cuda.current_stream(dev))

    rng = np.random.default_rng(7)
    print(f"{'M':>6} {'LUT':>4} {'BM':>4} {'BN':>4} {'BK':>4} {'BH':>4} {'NS':>3} {'W':>3} "
          f"{'gpu ms':>9} {'tok/s':>9} {'finite':>7}", flush=True)
    for M in Ms:
        ids = rng.integers(0, E, size=(M, K)).astype(np.int32)
        wts = rng.uniform(-1, 1, size=(M, K)).astype(np.float32)
        x = f32_to_bf16_bits(rng.standard_normal((M, H)).astype(np.float32))
        xs = torch.from_numpy(x.view(np.uint16)).view(torch.bfloat16).to(dev)
        ids_t = torch.from_numpy(ids).to(dev)
        wts_t = torch.from_numpy(wts).to(dev)
        grid = [("0", {}), ("1", {"XIAOTU_GPU_PREFILL_LUT": "1"})]
        for lut, env in grid:
            for e in env.items():
                os.environ[e[0]] = e[1]
            # env is read per call; force re-read of module-level defaults
            import importlib
            from vllm_xiaotu_moe import gpu_prefill as gp
            importlib.reload(gp)
            for _ in range(2):
                o = gp.gpu_moe_layer(xs, ids_t, wts_t, km[0], km[1], km[2], km[3],
                                     H=H, I=I, K=K, device=dev, slot=slot)
                torch.cuda.synchronize()
            ts = []
            for _ in range(reps):
                torch.cuda.synchronize(); t = time.perf_counter()
                o = gp.gpu_moe_layer(xs, ids_t, wts_t, km[0], km[1], km[2], km[3],
                                     H=H, I=I, K=K, device=dev, slot=slot)
                torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
            dt = min(ts)
            print(f"{M:>6} {lut:>4} "
                  f"{os.environ.get('XIAOTU_GPU_PREFILL_BM','64'):>4} "
                  f"{os.environ.get('XIAOTU_GPU_PREFILL_BN','64'):>4} "
                  f"{os.environ.get('XIAOTU_GPU_PREFILL_BK','64'):>4} "
                  f"{os.environ.get('XIAOTU_GPU_PREFILL_BH','64'):>4} "
                  f"{os.environ.get('XIAOTU_GPU_PREFILL_STAGES','2'):>3} "
                  f"{os.environ.get('XIAOTU_GPU_PREFILL_WARPS','4'):>3} "
                  f"{dt*1e3:9.1f} {M/dt:9.1f} "
                  f"{str(bool(torch.isfinite(o.float()).all())):>7}", flush=True)
            for e in env:
                os.environ.pop(e, None)
            importlib.reload(gp)
        del xs, ids_t, wts_t
        torch.cuda.empty_cache()
    print(f"[bk] => per-layer total at best ≈ assembly {asm_ms:.0f} + gpu; "
          f"CPU参考(服务实测,7000 tok)≈ 880 ms/层", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

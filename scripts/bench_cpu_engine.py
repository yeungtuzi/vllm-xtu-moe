#!/usr/bin/env python
"""CPU xiaotu engine microbench: per-layer cost vs batch size (real weights).

Used to A/B compiler flags for the native extension (report/bench_engine_flags.sh).
Pure CPU: engine.cpu_prefill = forward_many compute, no GPU copies.
Env: LAYER (default 3), BS (comma list), REP (default 3), THREADS.
"""
import glob
import json
import os
import sys
import time

import numpy as np
from safetensors import safe_open

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

MODEL = os.environ.get("XIAOTU_LAYER1_NPZ", "")
if not MODEL:
    raise SystemExit("set XIAOTU_LAYER1_NPZ=<real layer-1 npz fixture> (see docs/BENCHMARKS.md)")
H, I, GK, E, K = 4096, 2048, 32, 256, 6
N_LAYERS = 43


def f32_to_bf16_bits(x):
    v = x.astype(np.float32).view(np.uint32)
    lsb = (v >> 16) & 1
    return ((v + 0x7FFF + lsb) >> 16).astype(np.uint16)


def main():
    import torch  # noqa: F401  (engine needs torch loaded)
    import xiaotu_moe

    layer = int(os.environ.get("LAYER", "3"))
    bs = [int(v) for v in os.environ.get("BS", "512,2048,8192").split(",")]
    rep = int(os.environ.get("REP", "3"))

    shard = None
    for p in sorted(glob.glob(os.path.join(MODEL, "model-*.safetensors"))):
        with open(p, "rb") as f:
            hl = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(hl))
        if f"layers.{layer}.ffn.experts.0.w1.weight" in hdr:
            shard = p
            break
    assert shard, "shard not found"

    w13 = np.empty((E, 2 * I, H // 2), dtype=np.uint8)
    w2 = np.empty((E, H, I // 2), dtype=np.uint8)
    s13 = np.empty((E, 2 * I, H // GK), dtype=np.uint8)
    s2 = np.empty((E, H, I // GK), dtype=np.uint8)
    with safe_open(shard, "pt") as f:
        def get(e, sfx):
            return f.get_tensor(f"layers.{layer}.ffn.experts.{e}.{sfx}").cpu()
        for e in range(E):
            w1 = get(e, "w1.weight").contiguous().numpy().copy()
            w3 = get(e, "w3.weight").contiguous().numpy().copy()
            w13[e, 0:I] = w1
            w13[e, I:2 * I] = w3
            w2[e] = get(e, "w2.weight").contiguous().numpy().copy()
            s13[e, 0:I] = get(e, "w1.scale").contiguous().view(torch.uint8).numpy().copy()
            s13[e, I:2 * I] = get(e, "w3.scale").contiguous().view(torch.uint8).numpy().copy()
            s2[e] = get(e, "w2.scale").contiguous().view(torch.uint8).numpy().copy()

    m = xiaotu_moe.load()
    cfg = m.MOEConfigV2()
    cfg.num_processes = 1; cfg.process_id = 0; cfg.gpu_id = 0
    cfg.has_gate_proj = True; cfg.expert_num = E; cfg.top_k = K
    cfg.hidden_size = H; cfg.intermediate_size = I
    cfg.max_batch_size = 8192; cfg.max_num_seqs = 256
    cfg.stride = 32; cfg.group_min_len = 10; cfg.group_max_len = 4096 + 128
    cfg.groupN = 1; cfg.groupK = 32; cfg.activation_type = 0
    engine = m.MOE_MXFP4(cfg, w13, w2, s13, s2, 0, 0)
    print(f"[cpu-bench] variant={xiaotu_moe.__variant__} layer={layer}", flush=True)

    rng = np.random.default_rng(7)
    ids = rng.integers(0, E, size=(1, K)).astype(np.int32)
    wts = rng.uniform(-1, 1, size=(1, K)).astype(np.float32)

    print(f"{'B':>6} {'ms/layer':>10} {'43L ms':>9} {'tok/s':>8} {'TFLOP/s':>8}",
          flush=True)
    for B in bs:
        x = f32_to_bf16_bits(rng.standard_normal((B, H)).astype(np.float32))
        out = np.zeros((B, H), dtype=np.float32)
        # Random per-token routing: with identical ids for every token the batch
        # collapses onto a handful of experts, which is NOT what the real model
        # does (256 experts x ~48 rows each) and changes the weight working set
        # from 3.2 GB (DRAM) to a few tens of MB (L3).
        ids_b = rng.integers(0, E, size=(B, K)).astype(np.int32)
        wts_b = rng.uniform(-1, 1, size=(B, K)).astype(np.float32)
        engine.cpu_prefill(B, K, ids_b, wts_b, x, out)
        t0 = time.perf_counter()
        for _ in range(rep):
            engine.cpu_prefill(B, K, ids_b, wts_b, x, out)
        dt = (time.perf_counter() - t0) / rep
        # per token: K experts x (2*I*H + H*I) MACs
        flop = B * K * (2.0 * I * H + H * I) * 2 / 1e12
        print(f"{B:6d} {dt*1e3:10.2f} {dt*1e3*N_LAYERS:9.1f} "
              f"{B/(dt*N_LAYERS):8.1f} {flop/dt:8.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

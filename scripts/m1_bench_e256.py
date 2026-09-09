#!/usr/bin/env python
"""Decisive M1/M2 bridge: real E=256 cpu throughput of xiaotu engine on this EPYC 9654.

Reads ALL 256 routed experts of one real DS-V4-Flash layer from the checkpoint,
builds the xiaotu MOE_MXFP4 contract (raw e8m0 bytes, groupK=32), sweeps batch B,
reports per-layer cpu_decode/prefill time per token + extrapolated 43-layer tok/s.

This isolates the routing/bandwidth question the ~100 tok/s target hinges on:
on this AMD 9654 (no AMX), what is the actual per-token MoE cost at E=256?

Pure CPU (engine cpu_prefill = forward_many compute, no GPU copies needed).
"""
import glob
import json
import os
import sys
import time

import numpy as np
from safetensors import safe_open

REPO = "/home/user/lvllm/vllm-xiaotu-moe"
sys.path.insert(0, REPO)

MODEL = ("/home/user/.cache/modelscope/models/deepseek-ai--"
         "DeepSeek-V4-Flash-0731/snapshots/master")
H, I, GK, LAYER, E, K = 4096, 2048, 32, 3, 256, 6
N_LAYERS = 43


def f32_to_bf16_bits(x):
    v = x.astype(np.float32).view(np.uint32)
    lsb = (v >> 16) & 1
    return ((v + 0x7FFF + lsb) >> 16).astype(np.uint16)


def bf16_bits_to_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


def main():
    # find shard holding this layer's experts
    shard = None
    for p in sorted(glob.glob(os.path.join(MODEL, "model-*.safetensors"))):
        with open(p, "rb") as f:
            hl = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(hl))
        if f"layers.{LAYER}.ffn.experts.0.w1.weight" in hdr:
            shard = p
            break
    assert shard, "shard not found"
    print(f"[bench] layer {LAYER} experts in {os.path.basename(shard)}", flush=True)

    w13 = np.empty((E, 2 * I, H // 2), dtype=np.uint8)
    w2 = np.empty((E, H, I // 2), dtype=np.uint8)
    s13 = np.empty((E, 2 * I, H // GK), dtype=np.uint8)
    s2 = np.empty((E, H, I // GK), dtype=np.uint8)

    with safe_open(shard, "pt") as f:
        def get(e, sfx):
            return f.get_tensor(f"layers.{LAYER}.ffn.experts.{e}.{sfx}").cpu()
        for e in range(E):
            w1 = get(e, "w1.weight").contiguous().numpy().copy()
            w3 = get(e, "w3.weight").contiguous().numpy().copy()
            w2w = get(e, "w2.weight").contiguous().numpy().copy()
            s1 = get(e, "w1.scale").contiguous().view(torch.uint8).numpy().copy()
            s3 = get(e, "w3.scale").contiguous().view(torch.uint8).numpy().copy()
            s2w = get(e, "w2.scale").contiguous().view(torch.uint8).numpy().copy()
            w13[e, 0:I] = w1
            w13[e, I:2 * I] = w3
            w2[e] = w2w
            s13[e, 0:I] = s1
            s13[e, I:2 * I] = s3
            s2[e] = s2w
    print(f"[bench] assembled w13{w13.shape} w2{w2.shape} "
          f"s13{s13.shape} s2{s2.shape}", flush=True)

    import xiaotu_moe
    m = xiaotu_moe.load()
    cfg = m.MOEConfigV2()
    cfg.num_processes = 1; cfg.process_id = 0; cfg.gpu_id = 0
    cfg.has_gate_proj = True; cfg.expert_num = E; cfg.top_k = K
    cfg.hidden_size = H; cfg.intermediate_size = I
    cfg.max_batch_size = 8192; cfg.max_num_seqs = 256
    cfg.stride = 32; cfg.group_min_len = 10; cfg.group_max_len = 4096 + 128
    cfg.groupN = 1; cfg.groupK = 32; cfg.activation_type = 0
    engine = m.MOE_MXFP4(cfg, w13, w2, s13, s2, 0, 0)
    print("[bench] engine built", flush=True)

    rng = np.random.default_rng(7)
    ids = rng.integers(0, E, size=(1, K)).astype(np.int32)
    wts = rng.uniform(-1, 1, size=(1, K)).astype(np.float32)
    x_u16 = np.empty((1, H), dtype=np.uint16)

    print("\n  B(tok) | per-layer ms | 43-layer ms | ceiling tok/s (MoE-only)", flush=True)
    results = []
    for B in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        x = f32_to_bf16_bits(rng.standard_normal((B, H)).astype(np.float32))
        out = np.zeros((B, H), dtype=np.float32)
        ids_b = np.tile(ids, (B, 1))
        wts_b = np.tile(wts, (B, 1))
        # warmup
        engine.cpu_prefill(B, K, ids_b, wts_b, x, out)
        # timed
        n = 5
        t0 = time.perf_counter()
        for _ in range(n):
            engine.cpu_prefill(B, K, ids_b, wts_b, x, out)
        dt = (time.perf_counter() - t0) / n
        per_layer_ms = dt * 1e3
        full_ms = per_layer_ms * N_LAYERS
        tok_per_s = B / (full_ms / 1e3)
        mc = mc_str = ""
        print(f"  {B:7d} | {per_layer_ms:8.3f} | {full_ms:9.1f} | {tok_per_s:10.1f}", flush=True)
        results.append((B, per_layer_ms, full_ms, tok_per_s))

    print("\n[bench] NOTE: MoE-only ceiling (no attention/shared/graph overhead).", flush=True)
    print("[bench] decode wall-clock target ~100 tok/s xiaotu fork; this is the bandwidth bound.", flush=True)
    return 0


if __name__ == "__main__":
    import torch  # noqa: F401
    sys.exit(main())

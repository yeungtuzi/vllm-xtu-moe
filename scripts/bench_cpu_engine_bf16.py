#!/usr/bin/env python
"""Answer: does pre-dequantizing MXFP4 -> BF16 (no in-kernel dequant) speed up
the CPU engine? Same real weights, same routing, MOE_MXFP4 vs MOE_BF16.

Env: LAYER (3), BS (512,2048), REP (3), THREADS via XIAOTU_MOE_THREADS.
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

MODEL = ("/home/user/.cache/modelscope/models/deepseek-ai--"
         "DeepSeek-V4-Flash-0731/snapshots/master")
H, I, GK, E, K = 4096, 2048, 32, 256, 6
N_LAYERS = 43
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


def f32_to_bf16_bits(x):
    v = x.astype(np.float32).view(np.uint32)
    lsb = (v >> 16) & 1
    return ((v + 0x7FFF + lsb) >> 16).astype(np.uint16)


def load_layer(layer):
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
            w13[e, 0:I] = get(e, "w1.weight").contiguous().numpy()
            w13[e, I:2 * I] = get(e, "w3.weight").contiguous().numpy()
            w2[e] = get(e, "w2.weight").contiguous().numpy()
            s13[e, 0:I] = get(e, "w1.scale").contiguous().view(__import__("torch").uint8).numpy()
            s13[e, I:2 * I] = get(e, "w3.scale").contiguous().view(__import__("torch").uint8).numpy()
            s2[e] = get(e, "w2.scale").contiguous().view(__import__("torch").uint8).numpy()
    return w13, w2, s13, s2


def dequant_to_bf16(w13, w2, s13, s2):
    """fp4 nibbles x e8m0 block-32 scales -> bf16 bits, same math as the kernel."""
    lo13 = (w13 & 0x0F).astype(np.int64)
    hi13 = ((w13 >> 4) & 0x0F).astype(np.int64)
    v13 = np.stack([E2M1[lo13], E2M1[hi13]], axis=-1).reshape(E, 2 * I, H)
    sc13 = np.exp2(s13.astype(np.float32) - 127.0)
    sc13 = np.repeat(sc13, 32, axis=2)[:, :, :H]
    w13_bf = f32_to_bf16_bits(v13 * sc13)
    lo2 = (w2 & 0x0F).astype(np.int64)
    hi2 = ((w2 >> 4) & 0x0F).astype(np.int64)
    v2 = np.stack([E2M1[lo2], E2M1[hi2]], axis=-1).reshape(E, H, I)
    sc2 = np.exp2(s2.astype(np.float32) - 127.0)
    sc2 = np.repeat(sc2, 32, axis=2)[:, :, :I]
    w2_bf = f32_to_bf16_bits(v2 * sc2)
    return w13_bf, w2_bf


def make_cfg(m):
    cfg = m.MOEConfigV2()
    cfg.num_processes = 1; cfg.process_id = 0; cfg.gpu_id = 0
    cfg.has_gate_proj = True; cfg.expert_num = E; cfg.top_k = K
    cfg.hidden_size = H; cfg.intermediate_size = I
    cfg.max_batch_size = 8192; cfg.max_num_seqs = 256
    cfg.stride = 32; cfg.group_min_len = 10; cfg.group_max_len = 4096 + 128
    cfg.groupN = 1; cfg.groupK = 32; cfg.activation_type = 0
    return cfg


def bench(engine, B, ids, wts, x, rep):
    out = np.zeros((B, H), dtype=np.float32)
    engine.cpu_prefill(B, K, ids, wts, x, out)
    t0 = time.perf_counter()
    for _ in range(rep):
        engine.cpu_prefill(B, K, ids, wts, x, out)
    dt = (time.perf_counter() - t0) / rep
    flop = B * K * (2.0 * I * H + H * I) * 2 / 1e12
    return dt, flop / dt, out


def main():
    import torch  # noqa: F401
    import xiaotu_moe
    m = xiaotu_moe.load()
    layer = int(os.environ.get("LAYER", "3"))
    bs = [int(v) for v in os.environ.get("BS", "512,2048").split(",")]
    rep = int(os.environ.get("REP", "3"))

    w13, w2, s13, s2 = load_layer(layer)
    t0 = time.perf_counter()
    w13_bf, w2_bf = dequant_to_bf16(w13, w2, s13, s2)
    print(f"[bf16-bench] dequantized to bf16 in {time.perf_counter()-t0:.1f}s "
          f"({(w13_bf.nbytes+w2_bf.nbytes)/2**30:.1f} GiB)", flush=True)

    eng4 = m.MOE_MXFP4(make_cfg(m), w13, w2, s13, s2, 0, 0)
    engb = m.MOE_BF16(make_cfg(m), w13_bf, w2_bf, 0, 0, 0, 0)

    rng = np.random.default_rng(7)
    ids1 = rng.integers(0, E, size=(1, K)).astype(np.int32)
    wts1 = rng.uniform(-1, 1, size=(1, K)).astype(np.float32)
    print(f"{'B':>6} {'MXFP4 ms':>10} {'BF16 ms':>9} {'speedup':>8} "
          f"{'MXFP4 TFLOP/s':>14} {'BF16 TFLOP/s':>13}", flush=True)
    for B in bs:
        x = f32_to_bf16_bits(rng.standard_normal((B, H)).astype(np.float32))
        # realistic per-token routing (see bench_cpu_engine.py): fixed routing
        # collapses onto 6 experts and inflates the per-expert row count 40x.
        ids = rng.integers(0, E, size=(B, K)).astype(np.int32)
        wts = rng.uniform(-1, 1, size=(B, K)).astype(np.float32)
        d4, f4, o4 = bench(eng4, B, ids, wts, x, rep)
        db, fb, ob = bench(engb, B, ids, wts, x, rep)
        # numerical agreement (different kernels: expect bf16-level differences)
        rel = float(np.abs(o4 - ob).mean() / (np.abs(o4).mean() + 1e-6))
        print(f"{B:6d} {d4*1e3:10.2f} {db*1e3:9.2f} {d4/db:7.2f}x "
              f"{f4:14.2f} {fb:13.2f}   (mean|Δ|/mean = {rel:.2e})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

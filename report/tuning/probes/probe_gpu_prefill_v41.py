#!/usr/bin/env python
"""Is the GPU prefill path viable for **V4.1** dims, and where is the real crossover?

WHY (user directive, 2026-09-15): our principle is that prefill should run on the
GPU, and the V4-Flash measurement said "input > 384 tokens -> GPU is a net win".
V4.1 is a different beast: its routed experts are 269 GiB (6.72 GiB/layer) vs
V4-Flash's ~69 GiB, and A100-PCIE has no NVLink, so the per-prefill weight DMA is
a large FIXED cost:

    268.9 GiB / 25.0 GiB/s = 10.8 s   (measured, see NOTES §449)

The CPU engine's per-layer cost, by contrast, grows with the token count. So the
crossover is where cpu_ms(M) == h2d_ms + gpu_ms(M), and it is NOT 384 for V4.1.
This script measures the GPU side for real V4.1 weights (auto-detected dims) so
the threshold can be set from data instead of inherited from another model.

Env: MS="256 1024 2048 4096" LAYER=3 REP=3 (XIAOTU_LAYER1_NPZ = ckpt dir)
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

MODEL = os.environ.get("XIAOTU_LAYER1_NPZ", "")
if not MODEL:
    raise SystemExit("set XIAOTU_LAYER1_NPZ=<ckpt dir>")
_c = json.load(open(os.path.join(MODEL, "config.json")))
_t = _c.get("text_config", _c) or {}
H = int(os.environ.get("HID") or _t.get("hidden_size") or 4096)
I = int(os.environ.get("I") or _t.get("moe_intermediate_size") or 2048)
E = int(os.environ.get("E") or _t.get("n_routed_experts") or 256)
K = int(os.environ.get("K") or _t.get("num_experts_per_tok") or 6)
GK = int(os.environ.get("GK") or 32)


def f32_to_bf16_bits(x):
    v = x.astype(np.float32).view(np.uint32)
    lsb = (v >> 16) & 1
    return ((v + 0x7FFF + lsb) >> 16).astype(np.uint16)


def load_layer(layer: int):
    from safetensors import safe_open
    shard = None
    for p in sorted(glob.glob(os.path.join(MODEL, "model-*.safetensors"))):
        with open(p, "rb") as f:
            hl = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(hl))
        if f"layers.{layer}.ffn.experts.0.w1.weight" in hdr:
            shard = p
            break
    assert shard, f"shard with layers.{layer} not found"
    w13 = np.empty((E, 2 * I, H // 2), dtype=np.uint8)
    w2 = np.empty((E, H, I // 2), dtype=np.uint8)
    s13 = np.empty((E, 2 * I, H // GK), dtype=np.uint8)
    s2 = np.empty((E, H, I // GK), dtype=np.uint8)
    with safe_open(shard, "pt") as f:
        def g(e, sfx):
            return f.get_tensor(f"layers.{layer}.ffn.experts.{e}.{sfx}").cpu()
        for e in range(E):
            w13[e, 0:I] = g(e, "w1.weight").contiguous().numpy().copy()
            w13[e, I:2 * I] = g(e, "w3.weight").contiguous().numpy().copy()
            w2[e] = g(e, "w2.weight").contiguous().numpy().copy()
            s13[e, 0:I] = g(e, "w1.scale").contiguous().view(torch.uint8).numpy().copy()
            s13[e, I:2 * I] = g(e, "w3.scale").contiguous().view(torch.uint8).numpy().copy()
            s2[e] = g(e, "w2.scale").contiguous().view(torch.uint8).numpy().copy()
    return w13, w2, s13, s2


def main() -> int:
    layer = int(os.environ.get("LAYER", "3"))
    reps = int(os.environ.get("REP", "3"))
    Ms = [int(x) for x in os.environ.get("MS", "256 1024 2048 4096").split()]
    print(f"[gpu-prefill] H={H} I={I} E={E} K={K} GK={GK} layer={layer}", flush=True)
    w13, w2, s13, s2 = load_layer(layer)
    per_layer = w13.nbytes + w2.nbytes + s13.nbytes + s2.nbytes
    print(f"[gpu-prefill] per-layer expert bytes = {per_layer/2**30:.3f} GiB", flush=True)

    import xiaotu_moe
    m = xiaotu_moe.load()
    cfg = m.MOEConfigV2()
    cfg.num_processes = 1; cfg.process_id = 0; cfg.gpu_id = 0
    cfg.has_gate_proj = True; cfg.expert_num = E; cfg.top_k = K
    cfg.hidden_size = H; cfg.intermediate_size = I
    cfg.max_batch_size = max(Ms); cfg.max_num_seqs = 256
    cfg.stride = 32; cfg.group_min_len = 10; cfg.group_max_len = max(Ms) + 128
    cfg.groupN = 1; cfg.groupK = GK; cfg.activation_type = 0
    eng = m.MOE_MXFP4(cfg, w13, w2, s13, s2, 0, 0)

    from vllm_xiaotu_moe.gpu_prefill import gpu_moe_layer
    dev = torch.device("cuda:0")
    tw13 = torch.from_numpy(w13); tw2 = torch.from_numpy(w2)
    ts13 = torch.from_numpy(s13); ts2 = torch.from_numpy(s2)

    rng = np.random.default_rng(7)
    print(f"{'M':>6} {'cpu ms':>9} {'cpu t/s':>9} | {'gpu+H2D ms':>11} {'gpu t/s':>9} "
          f"{'maxrell':>9} | 结论", flush=True)
    cross = None
    for M in Ms:
        ids = rng.integers(0, E, size=(M, K)).astype(np.int32)
        wts = rng.uniform(-1, 1, size=(M, K)).astype(np.float32)
        x = f32_to_bf16_bits(rng.standard_normal((M, H)).astype(np.float32))
        out_cpu = np.zeros((M, H), dtype=np.float32)
        eng.cpu_prefill(M, K, ids, wts, x, out_cpu)
        tc = 1e9
        for _ in range(reps):
            t0 = time.perf_counter(); eng.cpu_prefill(M, K, ids, wts, x, out_cpu)
            tc = min(tc, time.perf_counter() - t0)

        xs = torch.from_numpy(x.view(np.uint16)).view(torch.bfloat16).to(dev)
        ids_t = torch.from_numpy(ids).to(dev)
        wts_t = torch.from_numpy(wts).to(dev)
        # No slot => gpu_moe_layer does the pinned H2D itself, which is exactly the
        # per-layer cost a non-overlapped GPU prefill pays. (With the ping-pong
        # prefetch this can hide behind the previous layer's kernels, but then the
        # pipeline is DMA-bound, so the per-layer cost is the same 6.72 GiB / 25 GB/s.)
        out_gpu = gpu_moe_layer(xs, ids_t, wts_t, tw13, ts13, tw2, ts2,
                                H=H, I=I, K=K, device=dev, slot=None)
        torch.cuda.synchronize()
        tg = 1e9
        for _ in range(reps):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out_gpu = gpu_moe_layer(xs, ids_t, wts_t, tw13, ts13, tw2, ts2,
                                    H=H, I=I, K=K, device=dev, slot=None)
            torch.cuda.synchronize(); tg = min(tg, time.perf_counter() - t0)
        og = out_gpu.float().cpu().numpy()
        fin = np.isfinite(og).all()
        # tolerance: the two paths differ in accumulation order and the GPU path
        # returns bf16, so compare against the CPU's magnitude, not bit-exactly.
        rel = float(np.abs(og - out_cpu).max() / (np.abs(out_cpu).max() + 1e-9)) if fin else float("nan")
        verdict = "GPU 更快" if tg < tc else "CPU 更快"
        if cross is None and tg < tc:
            cross = M
        print(f"{M:>6} {tc*1e3:9.2f} {M/tc:9.1f} | {tg*1e3:11.2f} {M/tg:9.1f} "
              f"{rel:9.2e} | {verdict}{'' if fin else ' **NONFINITE**'}", flush=True)
        del xs, ids_t, wts_t, out_gpu
        torch.cuda.empty_cache()
    if cross is not None:
        print(f"=> first M (of those tested) where GPU wins: {cross}")
    print("=> fixed per-prefill DMA (40 layers) = "
          f"{per_layer*40/2**30/25.0:.1f} s at 25 GiB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    # NENGINES>1:像真实服务那样建多个引擎(每层一个),只对最后一个计时。
    # 目的:验证"服务内单次调用比单实例微基准慢 3x"是否来自**多实例的内存放置**
    # (服务里 43 层 x ~3.2 GB 分片 ≈ 139 GB,而微基准只有 3.2 GB)。
    neng = int(os.environ.get("NENGINES", "1"))
    keep = []
    for _ in range(max(1, neng)):
        keep.append(m.MOE_MXFP4(cfg, w13, w2, s13, s2, 0, 0))
    engine = keep[-1]
    # ROUNDROBIN=1:像真实服务那样**轮流**调用这 neng 个引擎(每层一个),这样每个
    # 引擎的 scratch 缓冲在一次调用后就被换掉 ⇒ scratch 常驻缓存的效果消失。
    # 用来验证"服务内单次调用比单实例微基准慢 2x"是否来自 scratch 变冷。
    rr = int(os.environ.get("ROUNDROBIN", "0"))
    print(f"[cpu-bench] variant={xiaotu_moe.__variant__} layer={layer} nengines={len(keep)}",
          flush=True)

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
        #
        # DEDUP=N 控制"去重后的活跃专家数"(不设 = 全随机,36 个都不同)。
        # 真实解码步(qlen=6/topk=6、6 个 draft token 属同一段文本)实测去重后只有
        # ~12 个专家 ⇒ 每个专家平均被 3 个 token 命中(me=3)。内核只有 me>=4 才走
        # "解码一次喂 4 行"的快路径,me=1..3 走单行路径(每行都把权重重新解码一遍)
        # ⇒ 不设 DEDUP 就测不到真实情形。
        dedup = int(os.environ.get("DEDUP", "0"))
        if dedup > 0:
            picks = rng.integers(0, E, size=(dedup,)).astype(np.int32)
            ids_b = np.tile(picks, (B * K) // dedup + 1)[: B * K].reshape(B, K)
            ids_b = ids_b.reshape(-1)[rng.permutation(B * K)].reshape(B, K)
        else:
            ids_b = rng.integers(0, E, size=(B, K)).astype(np.int32)
        wts_b = rng.uniform(-1, 1, size=(B, K)).astype(np.float32)
        engine.cpu_prefill(B, K, ids_b, wts_b, x, out)
        t0 = time.perf_counter()
        if rr:
            for i in range(rep):
                keep[i % len(keep)].cpu_prefill(B, K, ids_b, wts_b, x, out)
        else:
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

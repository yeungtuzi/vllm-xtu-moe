#!/usr/bin/env python
"""GPU prefill 的**性能证明**:上游 MARLIN MXFP4 vs 我们的 CPU 引擎(同一层真实权重)。

回答的问题:把大 batch 的预填充切到 GPU,到底快多少、在哪个 batch 之上才划算?
(设计文档 `dev-docs/GPU_PREFILL_MAINLINE.md` §2.2 的 DMA 预算模型:一次完整预填充
 ≈ 69 GiB / 20 GB/s ≈ 3.4 s,与 batch 无关 ⇒ batch 越大越划算。)

用法:
  CUDA_VISIBLE_DEVICES=2 XIAOTU_LAYER1_NPZ=<ckpt> \
    /path/to/vllm-env/bin/python scripts/bench_gpu_prefill.py
  环境变量:MS="64 256 1024" LAYER=3 REP=5
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
H, I, GK, E, K = 4096, 2048, 32, 256, 6


def load_layer(ckpt: str, layer: int):
    from safetensors import safe_open

    shard = None
    for p in sorted(glob.glob(os.path.join(ckpt, "model-*.safetensors"))):
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
    ckpt = os.environ["XIAOTU_LAYER1_NPZ"]
    layer = int(os.environ.get("LAYER", "3"))
    reps = int(os.environ.get("REP", "5"))
    Ms = [int(x) for x in os.environ.get("MS", "64 256 1024").split()]
    w13, w2, s13, s2 = load_layer(ckpt, layer)
    rng = np.random.default_rng(7)

    import xiaotu_moe
    m = xiaotu_moe.load()
    dev = torch.device("cuda:0")

    # ---- CPU 引擎(每个 M 单独建?不必:max_batch_size 设成 max(Ms) 即可)----
    cfg = m.MOEConfigV2()
    cfg.num_processes = 1; cfg.process_id = 0; cfg.gpu_id = 0
    cfg.has_gate_proj = True; cfg.expert_num = E; cfg.top_k = K
    cfg.hidden_size = H; cfg.intermediate_size = I
    cfg.max_batch_size = max(Ms); cfg.max_num_seqs = 256
    cfg.stride = 32; cfg.group_min_len = 10; cfg.group_max_len = max(Ms) + 128
    cfg.groupN = 1; cfg.groupK = GK; cfg.activation_type = 0
    t0 = time.time()
    eng = m.MOE_MXFP4(cfg, w13, w2, s13, s2, 0, 0)
    print(f"[bench] CPU engine built in {time.time()-t0:.2f}s")

    # ---- GPU:把权重搬上去 + **加载期一次 MARLIN 重排**(只做一次,与 M 无关)----
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        prepare_moe_mxfp4_layer_for_marlin,
    )
    from vllm.scalar_type import scalar_types

    class _Stub:
        params_dtype = torch.bfloat16

    gpu_build_t0 = time.time()
    t_w1 = torch.from_numpy(w13).to(dev)
    t_w2 = torch.from_numpy(w2).to(dev)
    t_s1 = torch.from_numpy(s13).to(dev)
    t_s2 = torch.from_numpy(s2).to(dev)
    h2d_w = time.time() - gpu_build_t0
    rep_t0 = time.time()
    t_w1, t_w2, t_s1, t_s2, _, _ = prepare_moe_mxfp4_layer_for_marlin(
        _Stub(), t_w1, t_w2, t_s1, t_s2, None, None)
    print(f"[bench] GPU upload {h2d_w:.2f}s + marlin repack {time.time()-rep_t0:.2f}s "
          f"= {time.time()-gpu_build_t0:.2f}s (one-time, 每层)")

    print(f"{'M':>6} {'cpu ms':>9} {'cpu t/s':>9} | {'h2d ms':>8} {'gpu ms':>9} {'gpu t/s':>9} | 结论")
    for M in Ms:
        ids = np.stack([rng.choice(E, size=K, replace=False) for _ in range(M)]).astype(np.uint32)
        wts = rng.uniform(-1, 1, size=(M, K)).astype(np.float32)
        xf = (rng.standard_normal((M, H)).astype(np.float32) * 0.1)
        x_bf = torch.from_numpy(xf).to(torch.bfloat16)
        x_u16 = x_bf.view(torch.uint16).numpy().copy()
        out_cpu = np.zeros((M, H), dtype=np.float32)

        eng.cpu_prefill(M, K, ids, wts, x_u16, out_cpu)   # warmup
        tc = 1e9
        for _ in range(reps):
            t = time.perf_counter(); eng.cpu_prefill(M, K, ids, wts, x_u16, out_cpu)
            tc = min(tc, time.perf_counter() - t)

        hs = x_bf.to(dev); t_ids = torch.from_numpy(ids.astype(np.int32)).to(dev)
        t_wts = torch.from_numpy(wts).to(dev)
        N = 2 * I
        ws13 = torch.empty((E * M, max(H, N)), dtype=torch.bfloat16, device=dev)
        ws2 = torch.empty((E * M, N), dtype=torch.bfloat16, device=dev)
        out = torch.empty(M, H, dtype=torch.bfloat16, device=dev)

        def gpu_once():
            fused_marlin_moe(hs, t_w1, t_w2, None, None, t_s1, t_s2, t_wts, t_ids,
                             scalar_types.float4_e2m1f.id, global_num_experts=E,
                             activation=MoEActivation.SILU,
                             intermediate_cache13=ws2, intermediate_cache2=ws13,
                             output=out, input_dtype=torch.bfloat16)
        gpu_once(); torch.cuda.synchronize()
        tg = 1e9
        for _ in range(reps):
            torch.cuda.synchronize(); t = time.perf_counter()
            gpu_once(); torch.cuda.synchronize()
            tg = min(tg, time.perf_counter() - t)
        # 纯 H2D 计时(每次预填充都要搬 1.61 GiB/rank)
        # 纯 H2D:生产里每次预填充要把**重排后的**权重从主机搬上去(≈1.61 GiB/rank)。
        host_w1 = t_w1.cpu(); host_w2 = t_w2.cpu()
        stage1 = torch.empty_like(t_w1); stage2 = torch.empty_like(t_w2)
        th = 1e9
        for _ in range(3):
            torch.cuda.synchronize(); t = time.perf_counter()
            stage1.copy_(host_w1, non_blocking=False)
            stage2.copy_(host_w2, non_blocking=False)
            torch.cuda.synchronize(); th = min(th, time.perf_counter() - t)
        stage1.copy_(t_w1); stage2.copy_(t_w2)
        del host_w1, host_w2

        verdict = "GPU 更快" if (tg + th) < tc else "CPU 更快"
        print(f"{M:>6} {tc*1e3:9.3f} {M/tc:9.1f} | {th*1e3:8.2f} {tg*1e3:9.3f} {M/tg:9.1f} | {verdict}"
              f"  (含H2D: {(tg+th)*1e3:.2f} ms, {M/(tg+th):.0f} t/s)")
        del ws13, ws2, out
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

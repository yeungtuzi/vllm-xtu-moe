#!/usr/bin/env python
"""FP8 (MOE_FP8) CPU 引擎微基准:真实路由下的 TFLOP/s 与每 token 成本。

对齐 `scripts/bench_cpu_engine.py` 的方法学(逐 token 随机路由,避免固定路由把
权重工作集压进 L3),但对象是 **FP8 块量化** 路径 —— 真实 fp8 模型(GLM-5.3 /
Qwen3 / DS-V4 fp8)走的就是它。

用法: python scripts/bench_fp8_engine.py [E H I topk] [B ...]
默认: E=128 H=2048 I=768 topk=8(Qwen3-30B-A3B-FP8 的形状)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import xiaotu_moe  # noqa: E402

E, H, I, TOPK = (int(x) for x in (sys.argv[1:5] if len(sys.argv) > 4 else (128, 2048, 768, 8)))
BATCHES = [int(x) for x in (sys.argv[5:] or [512, 2048, 8192])]
GROUP = 128


def main() -> int:
    rng = np.random.default_rng(0)
    # 引擎把权重拷贝进自己的缓冲,所以这里用小块随机数据即可(内存按 E 缩放)
    w13 = rng.integers(0, 256, size=(E, 2 * I, H), dtype=np.uint8)
    w2 = rng.integers(0, 256, size=(E, H, I), dtype=np.uint8)
    s13 = rng.uniform(1e-3, 2e-2, size=(E, 2 * I // GROUP, H // GROUP)).astype(np.float32)
    s2 = rng.uniform(1e-3, 2e-2, size=(E, H // GROUP, I // GROUP)).astype(np.float32)

    cfg = xiaotu_moe.MOEConfigV2()
    cfg.num_processes = 1
    cfg.process_id = 0
    cfg.gpu_id = 0
    cfg.has_gate_proj = True
    cfg.expert_num = E
    cfg.top_k = TOPK
    cfg.hidden_size = H
    cfg.intermediate_size = I
    cfg.max_batch_size = max(BATCHES) * 2
    cfg.max_num_seqs = 512
    cfg.stride = GROUP
    cfg.group_min_len = 10
    cfg.group_max_len = 4096 + 128
    cfg.groupN = GROUP
    cfg.groupK = GROUP
    cfg.activation_type = 0

    t0 = time.time()
    eng = xiaotu_moe.MOE_FP8(
        cfg, w13.ctypes.data, w2.ctypes.data, s13.ctypes.data, s2.ctypes.data, 0, 0
    )
    print(
        f"[fp8bench] engine built in {time.time() - t0:.1f}s  "
        f"E={E} H={H} I={I} topk={TOPK} weights="
        f"{(w13.nbytes + w2.nbytes) / 1e9:.2f} GB",
        flush=True,
    )

    flops_per_tok = 2 * TOPK * 3 * H * I  # gate+up+down, 2 FLOPs per MAC
    print(f"[fp8bench] {flops_per_tok / 1e6:.0f} MFLOP/token/layer", flush=True)

    for B in BATCHES:
        x = rng.integers(0, 65535, size=(B, H), dtype=np.uint16)  # bf16 位模式
        ids = rng.integers(0, E, size=(B, TOPK)).astype(np.uint32)
        wts = rng.random((B, TOPK)).astype(np.float32)
        out = np.zeros((B, H), dtype=np.float32)
        # 预热
        eng.cpu_prefill(B, TOPK, ids.ctypes.data, wts.ctypes.data, x.ctypes.data, out.ctypes.data)
        reps = max(1, int(2048 / B))
        t0 = time.time()
        for _ in range(reps):
            eng.cpu_prefill(
                B, TOPK, ids.ctypes.data, wts.ctypes.data, x.ctypes.data, out.ctypes.data
            )
        dt = (time.time() - t0) / reps
        tflops = B * flops_per_tok / dt / 1e12
        print(
            f"[fp8bench] B={B:5d}  {dt * 1e3:8.1f} ms  "
            f"{dt / B * 1e3:6.3f} ms/token  {tflops:5.2f} TFLOP/s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

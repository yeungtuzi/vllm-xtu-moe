#!/usr/bin/env python
"""Focused: E=256 engine repeated-call crash isolation. Prints before each call."""
import glob
import json
import os
import sys

import numpy as np
import torch  # noqa: F401
from safetensors import safe_open

REPO = "/home/user/lvllm/vllm-xiaotu-moe"
sys.path.insert(0, REPO)
MODEL = ("/home/user/.cache/modelscope/models/deepseek-ai--"
         "DeepSeek-V4-Flash-0731/snapshots/master")
H, I, GK, LAYER, E, K = 4096, 2048, 32, 3, 256, 6


def f32_to_bf16_bits(x):
    v = x.astype(np.float32).view(np.uint32)
    lsb = (v >> 16) & 1
    return ((v + 0x7FFF + lsb) >> 16).astype(np.uint16)


def main():
    shard = None
    for p in sorted(glob.glob(os.path.join(MODEL, "model-*.safetensors"))):
        with open(p, "rb") as f:
            hl = int.from_bytes(f.read(8), "little")
            json.loads(f.read(hl))
        with safe_open(p, "pt") as f:
            if f"layers.{LAYER}.ffn.experts.0.w1.weight" in f.keys():
                shard = p
                break
    print("shard", os.path.basename(shard), flush=True)

    w13 = np.empty((E, 2 * I, H // 2), dtype=np.uint8)
    w2 = np.empty((E, H, I // 2), dtype=np.uint8)
    s13 = np.empty((E, 2 * I, H // GK), dtype=np.uint8)
    s2 = np.empty((E, H, I // GK), dtype=np.uint8)
    with safe_open(shard, "pt") as f:
        def get(e, sfx):
            return f.get_tensor(f"layers.{LAYER}.ffn.experts.{e}.{sfx}").cpu()
        for e in range(E):
            w13[e, 0:I] = get(e, "w1.weight").numpy().copy()
            w13[e, I:2 * I] = get(e, "w3.weight").numpy().copy()
            w2[e] = get(e, "w2.weight").numpy().copy()
            s13[e, 0:I] = get(e, "w1.scale").view(torch.uint8).numpy().copy()
            s13[e, I:2 * I] = get(e, "w3.scale").view(torch.uint8).numpy().copy()
            s2[e] = get(e, "w2.scale").view(torch.uint8).numpy().copy()
    print("assembled", flush=True)

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
    print("engine built", flush=True)

    rng = np.random.default_rng(7)
    for B in [1, 2, 4, 8, 16, 32]:
        x = f32_to_bf16_bits(rng.standard_normal((B, H)).astype(np.float32))
        ids = rng.integers(0, E, size=(B, K)).astype(np.int32)
        wts = rng.uniform(-1, 1, size=(B, K)).astype(np.float32)
        out = np.zeros((B, H), dtype=np.float32)
        for it in range(3):
            print(f"B={B} it={it} calling", flush=True)
            engine.cpu_prefill(B, K, ids, wts, x, out)
            print(f"B={B} it={it} ok", flush=True)
    print("ALL SWEEP OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

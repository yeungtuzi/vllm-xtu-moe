#!/usr/bin/env python
"""M1-REAL: 用真实 DeepSeek-V4-Flash-0731 层1 权重建一层 MOE_MXFP4 并跑 forward_many.

离线,只用 GPU2。验证引擎在本机(AMD EPYC9654 / A100)对真实权重的可运行性。
布局遵循 xiaotu-moe/scripts/_prep_real.py(已验证的正确契约):
  w13 [E, 2I, H//2] uint8(packed), w2 [E, H, I//2] uint8,
  w13_scale [E, 2I, H//GK] fp16, w2_scale [E, H, I//GK] fp16, GK=32。
用法:
  conda activate vllm-xiaotu-moe
  PYTHONPATH=/home/user/lvllm/vllm-xiaotu-moe CUDA_VISIBLE_DEVICES=2 \\
    python vllm-xiaotu-moe/scripts/m1_real_smoke.py
"""
import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
for p in (REPO, os.path.join(REPO, "xiaotu_moe", "build")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
from safetensors import safe_open

import xiaotu_moe

MODEL = ("/home/user/.cache/modelscope/models/deepseek-ai--"
         "DeepSeek-V4-Flash-0731/snapshots/master")
H, I, GK, LAYER = 4096, 2048, 32, 1
E = 256
TOPK = 6
Q = 64


def find_shard(layer):
    for sh in sorted(glob.glob(os.path.join(MODEL, "model-*.safetensors"))):
        with safe_open(sh, "pt") as f:
            if f"layers.{layer}.ffn.experts.0.w1.weight" in f.keys():
                return sh
    raise FileNotFoundError(layer)


def build_layer(layer):
    shard = find_shard(layer)
    w13 = np.empty((E, 2 * I, H // 2), dtype=np.uint8)
    w2 = np.empty((E, H, I // 2), dtype=np.uint8)
    g13 = np.empty((E, 2 * I, H // GK), dtype=np.float16)
    g2 = np.empty((E, H, I // GK), dtype=np.float16)
    with safe_open(shard, "pt") as f:
        def get(e, sfx):
            return f.get_tensor(f"layers.{layer}.ffn.experts.{e}.{sfx}").cpu()
        for e in range(E):
            w13[e] = get(e, "w1.weight").numpy().view(np.uint8).reshape(2 * I, H // 2)
            w13[e] = np.concatenate([
                w13[e], get(e, "w3.weight").numpy().view(np.uint8).reshape(2 * I, H // 2)
            ]) if False else w13[e]  # placeholder
    return w13, w2, g13, g2


# 简化:直接按 w1/w3 拼接布局装配(与 fork _process_mxfp4 期望一致)
def assemble(layer=1):
    shard = find_shard(layer)
    w13 = torch.empty(E, 2 * I, H // 2, dtype=torch.uint8)
    w2 = torch.empty(E, H, I // 2, dtype=torch.uint8)
    g13 = torch.empty(E, 2 * I, H // GK, dtype=torch.float16)
    g2 = torch.empty(E, H, I // GK, dtype=torch.float16)
    with safe_open(shard, "pt") as f:
        def get(e, sfx):
            return f.get_tensor(f"layers.{layer}.ffn.experts.{e}.{sfx}").cpu()
        e13 = torch.empty(E, 2 * I, H // 2, dtype=torch.uint8)
        for e in range(E):
            w1 = get(e, "w1.weight")           # [2I?, H]? 实际见下
            w11 = get(e, "w1.weight").view(torch.uint8)
            w2w = get(e, "w2.weight").view(torch.uint8)
            w3 = get(e, "w3.weight").view(torch.uint8)
            w13[e] = torch.cat([w1, w3], dim=0)
            w2[e] = w2w
            s1 = get(e, "w1.weight_scale").to(torch.float16)
            s3 = get(e, "w3.weight_scale").to(torch.float16)
            g13[e] = torch.cat([s1, s3], dim=0)
            g2[e] = get(e, "w2.weight_scale").to(torch.float16)
    return w13.contiguous(), w2.contiguous(), g13.contiguous(), g2.contiguous()


if __name__ == "__main__":
    print("loading real layer-1 weights (E=256) ...")
    w13, w2, g13, g2 = assemble(LAYER)
    print("w13", tuple(w13.shape), "w2", tuple(w2.shape),
          "g13", tuple(g13.shape), "g2", tuple(g2.shape))

    cfg = xiaotu_moe.MOEConfigV2()
    cfg.num_processes = 1
    cfg.process_id = 0
    cfg.gpu_id = 2
    cfg.has_gate_proj = True
    cfg.expert_num = E
    cfg.top_k = TOPK
    cfg.hidden_size = H
    cfg.intermediate_size = I
    cfg.max_batch_size = 8192
    cfg.max_num_seqs = 256
    cfg.stride = 32
    cfg.group_min_len = 10
    cfg.group_max_len = 512
    cfg.activation_type = 0
    cfg.use_gpu_prefill = False
    cfg.groupN = 1
    cfg.groupK = GK

    engine = xiaotu_moe.MOE_MXFP4(
        cfg, w13.data_ptr(), w2.data_ptr(),
        g13.data_ptr(), g2.data_ptr(), 0, 0)
    print("engine constructed OK:", type(engine).__name__)

    x = torch.randint(0, 3, (Q, H), dtype=torch.uint16).contiguous()
    ids = torch.arange(0, TOPK, dtype=torch.int32).repeat(Q, 1).contiguous()
    tw = torch.ones(Q, TOPK, dtype=torch.float32).contiguous()
    out = torch.zeros(Q, H, dtype=torch.float32).contiguous()
    engine.cpu_prefill(Q, TOPK, ids, tw, x, out)
    print("forward_many OK; out", tuple(out.shape), out.dtype)
    print("M1-REAL PASS")

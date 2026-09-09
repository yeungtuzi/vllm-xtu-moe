#!/usr/bin/env python
"""M1: 验证 xiaotu 引擎在主线 env + 本机(AMD EPYC9654/A100)可构造一层 MOE_MXFP4 并跑 forward_many.

离线,只用 GPU2(CUDA_VISIBLE_DEVICES=2)。合成 DeepSeek-V4-Flash-0731 尺寸权重。
不加载真实模型、不起 serve、不碰 prod。

判据:一层 MOE_MXFP4 构造成功 + cpu_prefill(M*qlen 小批)前向不崩 + 输出形状/设备正确。
用法:
  conda activate vllm-xiaotu-moe
  PYTHONPATH=/home/user/lvllm/vllm-xiaotu-moe CUDA_VISIBLE_DEVICES=2 \\
    python vllm-xiaotu-moe/scripts/m1_engine_smoke.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
for p in (REPO, os.path.join(REPO, "xiaotu_moe", "build")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import xiaotu_moe

# 合成单层参数(DS-V4-Flash 尺寸):hidden 4096, moe_intermediate 2048, experts(少,便于快测)
H = 4096
I = 2048
E = 8          # 只用 8 个 expert 做冒烟(真实 256,这里够验证机制)
TOPK = 4
Q = 64         # qlen(小)

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

# 权重:MXFP4 需 uint8 存储 + fp16/bf16 scale。
# 引擎期望 w13 [E, 2I, H] uint8(packed), w2 [E, H, I] uint8; scale [E, N_groups, ...] fp16。
# 这里用随机小值制造合法形状即可(不做数值验证)。
def mxfp4_block(r, dt=torch.uint8):
    return torch.randint(0, 16, r, dtype=dt)

w13 = torch.randint(0, 16, (E, 2 * I, H), dtype=torch.uint8).contiguous()
w2 = torch.randint(0, 16, (E, H, I), dtype=torch.uint8).contiguous()
w13_s = torch.randn(E, 2 * I // 32, H // 32, dtype=torch.float16).contiguous()
w2_s = torch.randn(E, H // 32, I // 32, dtype=torch.float16).contiguous()
groupN = max(w13.shape[1] // w13_s.shape[1], w2.shape[1] // w2_s.shape[1])
groupK = max((w13.shape[2] * 2) // w13_s.shape[2], (w2.shape[2] * 2) // w2_s.shape[2])
cfg.groupN = groupN
cfg.groupK = groupK

engine = xiaotu_moe.MOE_MXFP4(
    cfg,
    w13.data_ptr(), w2.data_ptr(),
    w13_s.data_ptr(), w2_s.data_ptr(),
    0, 0,
)
print("engine constructed OK:", type(engine).__name__)

# 前向(纯 CPU 布局):input [Q,H] uint16(bf16), ids [Q,TOPK] int32, weights [Q,TOPK] fp32,
# output [Q,H] fp32
x = torch.randint(0, 3, (Q, H), dtype=torch.uint16).contiguous()
ids = torch.arange(0, TOPK, dtype=torch.int32).repeat(Q, 1).contiguous()
tw = torch.ones(Q, TOPK, dtype=torch.float32).contiguous()
out = torch.zeros(Q, H, dtype=torch.float32).contiguous()

engine.cpu_prefill(Q, TOPK, ids, tw, x, out)
print("forward_many (cpu_prefill) ran OK; out shape", tuple(out.shape), "dtype", out.dtype)
print("M1 PASS")

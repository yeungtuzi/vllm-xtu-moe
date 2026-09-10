#!/usr/bin/env python
"""WNA16(INT4 W4A16)布局重排 + 引擎数值自检(不依赖 vLLM,秒级)。

验证两件事:
  1. 插件对 GPTQ 检查点布局的重排(`w13 [E, K/8, N] int32` → `[E, N, K/2] u8` +
     scales `[E, K/g, N]` → `[E, N, K/g]`)与引擎期望的布局一致;
  2. 引擎的 `MOE_WNA16` 内核(gather 路径、`lut[nibble] = nibble - 8`)与 torch
     参考实现一致 —— 这条路径此前从未被数值验证过。

用法: python scripts/test_wna16_repack.py [E H I GROUP TOPK]
License: Apache-2.0
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import xiaotu_moe  # noqa: E402

E, H, I, GROUP, TOPK = (int(x) for x in (sys.argv[1:] if len(sys.argv) > 5
                                        else (4, 256, 128, 128, 2)))
N13 = 2 * I


def gptq_pack(nib: np.ndarray, axis: int) -> np.ndarray:
    """Pack int4 nibbles (0..15) along `axis` into int32 (8 nibbles/word)."""
    nib = np.moveaxis(nib, axis, -1).astype(np.uint32)      # [..., K]
    pad = (-nib.shape[-1]) % 8
    if pad:
        nib = np.pad(nib, [(0, 0)] * (nib.ndim - 1) + [(0, pad)])
    words = np.zeros(nib.shape[:-1] + (nib.shape[-1] // 8,), dtype=np.uint32)
    for i in range(8):
        words |= (nib[..., i::8] & 0xF) << (4 * i)
    return np.moveaxis(words, -1, axis).astype(np.int32)


def main() -> int:
    rng = np.random.default_rng(0)
    # 随机 int4 权重 + 对称零点(GPTQ 存 zp-1=7)
    w13_nib = rng.integers(0, 16, size=(E, N13, H), dtype=np.uint8)   # [E, N, K]
    w2_nib = rng.integers(0, 16, size=(E, H, I), dtype=np.uint8)      # [E, N, K]
    s13 = rng.uniform(1e-3, 2e-2, size=(E, H // GROUP, N13)).astype(np.float32)
    s2 = rng.uniform(1e-3, 2e-2, size=(E, I // GROUP, H)).astype(np.float32)

    # GPTQ 检查点布局:qweight [E, K/8, N] int32(沿 K 打包,N 在最后)
    q13 = gptq_pack(np.transpose(w13_nib, (0, 2, 1)), axis=1)   # [E, H/8, N]
    q2 = gptq_pack(np.transpose(w2_nib, (0, 2, 1)), axis=1)     # [E, I/8, H]
    q13_t = torch.from_numpy(q13)
    q2_t = torch.from_numpy(q2)
    s13_t = torch.from_numpy(s13)
    s2_t = torch.from_numpy(s2)

    # ---- 与插件相同的重排 ------------------------------------------------
    w13_u8 = q13_t.transpose(1, 2).contiguous().view(torch.uint8)   # [E, N, H/2]
    w2_u8 = q2_t.transpose(1, 2).contiguous().view(torch.uint8)     # [E, H, I/2]
    s13_e = s13_t.transpose(1, 2).contiguous()                      # [E, N, H/g]
    s2_e = s2_t.transpose(1, 2).contiguous()                        # [E, H, I/g]
    assert w13_u8.shape == (E, N13, H // 2), w13_u8.shape
    assert w2_u8.shape == (E, H, I // 2), w2_u8.shape

    cfg = xiaotu_moe.MOEConfigV2()
    cfg.num_processes = 1
    cfg.process_id = 0
    cfg.gpu_id = 0
    cfg.has_gate_proj = True
    cfg.expert_num = E
    cfg.top_k = TOPK
    cfg.hidden_size = H
    cfg.intermediate_size = I
    cfg.max_batch_size = 64
    cfg.max_num_seqs = 64
    cfg.stride = GROUP
    cfg.group_min_len = 10
    cfg.group_max_len = 4096 + 128
    cfg.groupN = 1              # WNA16: 逐行、沿 K 分组
    cfg.groupK = GROUP
    cfg.activation_type = 0
    eng = xiaotu_moe.MOE_WNA16(
        cfg, w13_u8.data_ptr(), w2_u8.data_ptr(),
        s13_e.data_ptr(), s2_e.data_ptr(), 0, 0,
    )

    xf = (rng.standard_normal((1, H)).astype(np.float32) * 0.5)
    x = (torch.from_numpy(xf).view(torch.float32).view(torch.int32) >> 16).to(torch.int16)
    ids = torch.from_numpy(rng.integers(0, E, size=(1, TOPK)).astype(np.uint32))
    wts = torch.from_numpy(rng.random((1, TOPK)).astype(np.float32))
    out = torch.zeros((1, H), dtype=torch.float32)
    eng.cpu_prefill(1, TOPK, ids.data_ptr(), wts.data_ptr(), x.data_ptr(), out.data_ptr())

    # ---- torch 参考 ------------------------------------------------------
    ref = torch.zeros(H, dtype=torch.float32)
    xb = x.view(torch.bfloat16).float()          # bf16 输入
    for r in range(TOPK):
        e, w = int(ids[0, r]), float(wts[0, r])
        if w == 0.0:
            continue
        deq13 = (w13_nib[e].astype(np.float32) - 8.0) * \
            np.repeat(s13[e].T, GROUP, axis=1)[:, :H]      # [N, K]
        d13 = torch.from_numpy(deq13)
        gate, up = d13[:I] @ xb[0], d13[I:] @ xb[0]
        act = ((gate / (1 + torch.exp(-gate))) * up).to(torch.bfloat16).float()
        deq2 = (w2_nib[e].astype(np.float32) - 8.0) * \
            np.repeat(s2[e].T, GROUP, axis=1)[:, :I]       # [N, K]
        ref += w * (torch.from_numpy(deq2) @ act)

    d = (out[0] - ref).abs()
    rel = float(d.max() / (ref.abs().max() + 1e-12))
    rms = float(torch.sqrt(torch.mean(d * d)) / (torch.sqrt(torch.mean(ref * ref)) + 1e-12))
    # 1e-4 阈值:引擎在激活后转 bf16(与 torch 的 exp/FMA 顺序略有差异),个别元素
    # 落在 bf16 舍入边界上会差 1 ULP,经 down 投影放大到 ~1e-5 量级(与 FP8 路径同源,
    # 见 scripts/test_glm53_fp8_layer.py 的 9.3e-5)。真正的布局/内核错误是 O(1)。
    ok = rel < 1e-4
    print(f"[wna16] E={E} H={H} I={I} group={GROUP} topk={TOPK}  "
          f"max_rel={rel:.3e} rms_rel={rms:.3e} ref_rms={float(torch.sqrt(torch.mean(ref*ref))):.4f}  "
          f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

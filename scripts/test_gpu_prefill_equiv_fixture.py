#!/usr/bin/env python
"""GPU prefill(P1)三方对拍 —— 基于**已验证的 fixture**,避免自己造 golden。

思路:直接复用 `scripts/test_block23_equiv.py` 用的那个 fixture
(`fixtures/real_layer1_model.npz`),它自带**权威的反量化权重** `gol13/gol2`
(已用 `max_abs=0.0` 验证:我用 nibble+e8m0 反量化得到的与它逐位相同)。

* **CPU**:`engine.cpu_prefill(...)` —— 该路径已被 `test_block23_equiv.py` 验证(OK=7 BAD=1)。
* **GPU**:`vllm...fused_experts(x_bf16, gol13_bf16, gol2_bf16, wts, ids)`(未量化路径)。
* 两边都应与 **numpy golden**(同 block23 语义,含激活的 bf16 取整)一致。

用法:
  CUDA_VISIBLE_DEVICES=2 /path/to/vllm-env/bin/python scripts/test_gpu_prefill_equiv_fixture.py
  环境变量:REP(默认 3)= 每个专家的重复次数(M = E*REP/K)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
NPZ = os.environ.get("NPZ", os.path.join(REPO, "fixtures", "real_layer1_model.npz"))


def bf16_bits_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    v = x.astype(np.float32).view(np.uint32)
    lsb = (v >> 16) & 1
    return ((v + 0x7FFF + lsb) >> 16).astype(np.uint16)


def main() -> int:
    d = np.load(NPZ)
    w13, w2, gol13, gol2 = d["w13"], d["w2"], d["gol13"], d["gol2"]
    E, I, H = int(d["E"]), int(d["I"]), int(d["H"])
    K = 6
    REP = int(os.environ.get("REP", "3"))

    rng = np.random.default_rng(23)
    # 每个专家 REP 次 ⇒ M = E*REP/K 个 token,每行 K 个**互不相同**的专家
    rows = [e for e in range(E) for _ in range(REP)]
    rows = rows[: len(rows) // K * K]
    M = len(rows) // K
    ids = np.array(rows, dtype=np.int32).reshape(M, K)
    wts = rng.uniform(-1, 1, size=(M, K)).astype(np.float32)
    x_u16 = f32_to_bf16_bits(rng.standard_normal((M, H)).astype(np.float32) * 6.0)
    xf = bf16_bits_to_f32(x_u16)

    import xiaotu_moe
    m = xiaotu_moe.load()
    print(f"[fx] variant={xiaotu_moe.__variant__} E={E} I={I} H={H} M={M} K={K} REP={REP}")

    # ---- CPU ----
    cfg = m.MOEConfigV2()
    cfg.num_processes = 1; cfg.process_id = 0; cfg.gpu_id = 0
    cfg.has_gate_proj = True; cfg.expert_num = E; cfg.top_k = K
    cfg.hidden_size = H; cfg.intermediate_size = I
    cfg.max_batch_size = 256; cfg.max_num_seqs = 256
    cfg.stride = 32; cfg.group_min_len = 10; cfg.group_max_len = 4096 + 128
    cfg.groupN = 1; cfg.groupK = 32; cfg.activation_type = 0
    def f32_to_e8m0(ss):      # 与 test_block23_equiv.py 完全一致
        lg = np.round(np.log2(np.maximum(ss, 1e-30))).astype(np.int32) + 127
        return np.clip(lg, 0, 255).astype(np.uint8)
    s13, s2 = f32_to_e8m0(d["g13"]), f32_to_e8m0(d["g2"])
    eng = m.MOE_MXFP4(cfg, w13, w2, s13, s2, 0, 0)
    out_cpu = np.zeros((M, H), dtype=np.float32)
    eng.cpu_prefill(M, K, ids, wts, x_u16, out_cpu)

    # ---- numpy golden(与 block23 完全同语义)----
    golden = np.zeros((M, H), dtype=np.float32)
    for t in range(M):
        for r in range(K):
            e, w = int(ids[t, r]), float(wts[t, r])
            g = gol13[e, 0:I] @ xf[t]
            u = gol13[e, I:2 * I] @ xf[t]
            act = (g / (1.0 + np.exp(-g))) * u
            golden[t] += w * (gol2[e] @ bf16_bits_to_f32(f32_to_bf16_bits(act)))

    # ---- bf16 一致的 golden:GPU 用的是 **bf16 权重**,所以要拿"权重也取整到 bf16"
    #      的 golden 去比,才能把"实现差异"与"精度差异"分开。
    g13b = bf16_bits_to_f32(f32_to_bf16_bits(gol13))
    g2b = bf16_bits_to_f32(f32_to_bf16_bits(gol2))
    gold_bf16 = np.zeros((M, H), dtype=np.float32)
    for t in range(M):
        for r in range(K):
            e, w = int(ids[t, r]), float(wts[t, r])
            g = g13b[e, 0:I] @ xf[t]
            u = g13b[e, I:2 * I] @ xf[t]
            act = (g / (1.0 + np.exp(-g))) * u
            gold_bf16[t] += w * (g2b[e] @ bf16_bits_to_f32(f32_to_bf16_bits(act)))

    # ---- GPU:上游 fused_experts(未量化,吃 gol13/gol2 的 bf16)----
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    dev = torch.device("cuda:0")
    _wd = torch.float32 if os.environ.get("W32") else torch.bfloat16
    t_w1 = torch.from_numpy(gol13).to(dev).to(_wd)
    t_w2 = torch.from_numpy(gol2).to(dev).to(_wd)
    hs = torch.from_numpy(xf).to(dev).to(torch.bfloat16)
    t_ids = torch.from_numpy(ids).to(dev)
    t_wts = torch.from_numpy(wts).to(dev)
    out_gpu = fused_experts(hs, t_w1, t_w2, t_wts, t_ids,
                            activation=MoEActivation.SILU, global_num_experts=E)
    y_gpu = out_gpu.float().cpu().numpy()

    def stat(name, a, ref=None):
        ref = golden if ref is None else ref
        ad = np.abs(a - ref)
        print(f"[fx] {name:4s} vs golden: max_abs={ad.max():.4e}  "
              f"mean|y|={np.abs(a).mean():.4e} (golden {np.abs(golden).mean():.4e})")
        return float(ad.max())

    c = stat("cpu", out_cpu)
    stat("gpu", y_gpu, gold_bf16)
    g = float(np.abs(y_gpu - gold_bf16).max())
    ok = max(c, g) < 2e-2
    print("[fx]", "OK" if ok else "MISMATCH", f"(阈值 max_abs<2e-2; cpu={c:.2e} gpu={g:.2e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

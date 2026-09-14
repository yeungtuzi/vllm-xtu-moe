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
  环境变量:REP(默认 3)= 每个专家的重复次数(M = E*REP/K);KE = top-k
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
    K = int(os.environ.get("KE", "6"))
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

    # ---- GPU 路径可切换 ----
    #  GPUMODE=bf16 (默认)  : 未量化 fused_experts(吃反量化的 bf16 权重),只验证语义
    #  GPUMODE=mxfp4         : **上游真正的 MXFP4 GPU 内核**(吃打包权重 + e8m0 scales),
    #                          走 modular 入口 triton_kernel_fused_experts(precomputed routing),
    #                          这样 DS-V4 自己的 sqrtsoftplus/group-topk 路由可以照用。
    GPUMODE = os.environ.get("GPUMODE", "bf16")
    if GPUMODE == "mxfp4":
        return _gpu_mxfp4(d, ids, wts, xf, M, K, E, I, H, golden)
    if GPUMODE == "marlin":
        return _gpu_marlin(d, ids, wts, xf, M, K, E, I, H, golden)
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
        # 单元素 max 对 bf16 计算没有统计意义(最坏元素可达几个 %),看分位数才对:
        scale = float(np.abs(ref).mean()) or 1.0
        rel = ad / (np.abs(ref) + 1e-3)
        q = [float(np.percentile(rel, p)) for p in (50, 99, 99.9)]
        aq = [float(np.percentile(ad, p)) for p in (50, 99, 99.9)]
        # 绝对误差相对"输出量级"的归一化,才是公平口径(y 的元素可以接近 0)
        print(f"[fx] {name:4s} vs golden: abs p50={aq[0]:.2e} p99={aq[1]:.2e} p99.9={aq[2]:.2e} "
              f"max={ad.max():.2e}  (|y| 均值 {scale:.3e})")
        print(f"[fx]     归一化: p50={aq[0]/scale:.2e} p99={aq[1]/scale:.2e} "
              f"p99.9={aq[2]/scale:.2e} max={ad.max()/scale:.2e}   [rel p50={q[0]:.1e} p99={q[1]:.1e}]")
        return (aq[0] / scale, aq[1] / scale, ad.max() / scale)  # 归一化 (p50,p99,max)

    c = stat("cpu", out_cpu)
    g = stat("gpu", y_gpu, gold_bf16)
    # 判定口径(归一化 = Δ / mean|y|):
    #   * CPU 是 fp32 累加 ⇒ **max** 也应 ~1e-6;
    #   * GPU 走 bf16 内核(权重/激活都取整到 bf16)⇒ 单元素 max 没有统计意义
    #     (最坏元素可达 eps 的十几倍),应看 **中位数**:中位数 ≈ bf16 eps(3.9e-3)
    #     就说明两边"在 bf16 精度内等价",差异来自 dtype 而不是实现。
    EPS_BF16 = 3.9e-3
    c50, c99, cmax = c
    g50, g99, gmax = g
    ok_cpu = cmax < 1e-4
    ok_gpu = (g50 < EPS_BF16) and (g99 < 10 * EPS_BF16)
    ok = ok_cpu and ok_gpu
    print(f"[fx] 判定: cpu(max={cmax:.2e} <1e-4? {ok_cpu})  "
          f"gpu(p50={g50:.2e} <eps? {g50 < EPS_BF16}; p99={g99:.2e} <10eps? {g99 < 10*EPS_BF16})  "
          f"=> {'OK(P1 通过)' if ok else 'MISMATCH'}")
    return 0 if ok else 1



def _gpu_mxfp4(d, ids, wts, x_bf16, M, K, E, I, H, golden):
    """上游真正的 MXFP4 GPU 内核 + **precomputed routing**(DS-V4 路由由我们给)。"""
    import torch
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        mxfp4_w4a16_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (
        make_routing_data,
        triton_kernel_fused_experts,
    )

    dev = torch.device("cuda:0")
    w13, w2 = d["w13"], d["w2"]
    s13, s2 = d["g13"], d["g2"]          # fixture 的 fp32 尺度 ⇒ 转 e8m0 字节
    def f32_to_e8m0(ss):
        lg = np.round(np.log2(np.maximum(ss, 1e-30))).astype(np.int32) + 127
        return np.clip(lg, 0, 255).astype(np.uint8)
    t_w1 = torch.from_numpy(w13).to(dev)                    # uint8 packed
    t_w2 = torch.from_numpy(w2).to(dev)
    t_s1 = torch.from_numpy(f32_to_e8m0(s13)).to(dev)
    t_s2 = torch.from_numpy(f32_to_e8m0(s2)).to(dev)
    hs = torch.from_numpy(x_bf16.reshape(M, H)).to(dev).to(torch.bfloat16)
    t_ids = torch.from_numpy(ids.astype(np.int64)).to(dev)
    t_wts = torch.from_numpy(wts).to(dev)
    qc = mxfp4_w4a16_moe_quant_config(t_s1, t_s2)
    routing_data, gather_idx, scatter_idx = make_routing_data(t_ids, t_wts, E)
    out = torch.empty_like(hs, dtype=torch.float32)
    triton_kernel_fused_experts(
        out, hs, t_w1, t_w2, routing_data, gather_idx, scatter_idx,
        topk=K, activation=MoEActivation.SILU, quant_config=qc,
        global_num_experts=E,
    )
    y = out.float().cpu().numpy()
    ad = np.abs(y - golden)
    scale = float(np.abs(golden).mean()) or 1.0
    print(f"[fx] gpu(mxfp4) vs golden: abs p50={np.percentile(ad,50):.2e} "
          f"p99={np.percentile(ad,99):.2e} max={ad.max():.2e}  "
          f"归一化 p50={np.percentile(ad,50)/scale:.2e} max={ad.max()/scale:.2e}")
    ok = np.percentile(ad, 50) / scale < 3.9e-3
    print("[fx] 判定(mxfp4):", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


def _gpu_marlin(d, ids, wts, xf, M, K, E, I, H, golden):
    """上游 **MARLIN MXFP4** 功能入口(`fused_marlin_moe`),吃显式打包权重 + e8m0 scales。

    为什么是它:`MarlinExperts.apply()` 内部就是直接调这个函数;triton 那条(OAITriton)
    只支持 SWIGLUOAI(实测断言),不适用 DS-V4。MARLIN 也正是 lk fork 在 A100 上用的内核。
    """
    import torch
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.scalar_type import scalar_types

    dev = torch.device("cuda:0")
    w13, w2 = d["w13"], d["w2"]
    def f32_to_e8m0(ss):
        lg = np.round(np.log2(np.maximum(ss, 1e-30))).astype(np.int32) + 127
        return np.clip(lg, 0, 255).astype(np.uint8)
    t_w1 = torch.from_numpy(w13).to(dev)
    t_w2 = torch.from_numpy(w2).to(dev)
    t_s1 = torch.from_numpy(f32_to_e8m0(d["g13"])).to(dev)
    t_s2 = torch.from_numpy(f32_to_e8m0(d["g2"])).to(dev)
    hs = torch.from_numpy(xf.reshape(M, H)).to(dev).to(torch.bfloat16)
    t_ids = torch.from_numpy(ids.astype(np.int32)).to(dev)
    t_wts = torch.from_numpy(wts).to(dev)
    # workspace_shapes(): ws13=(E*M, max(K,N*2)), ws2=(E*M, N);apply() 里两者是**交换**传的
    N = 2 * I
    ws13 = torch.empty((E * M, max(H, N)), dtype=torch.bfloat16, device=dev)
    ws2 = torch.empty((E * M, N), dtype=torch.bfloat16, device=dev)
    out = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
    fused_marlin_moe(
        hs, t_w1, t_w2, None, None, t_s1, t_s2, t_wts, t_ids,
        scalar_types.float4_e2m1f.id,
        global_num_experts=E, activation=MoEActivation.SILU,
        intermediate_cache13=ws2, intermediate_cache2=ws13,
        output=out, input_dtype=torch.bfloat16,
    )
    y = out.float().cpu().numpy()
    ad = np.abs(y - golden)
    scale = float(np.abs(golden).mean()) or 1.0
    print(f"[fx] gpu(marlin) vs golden: abs p50={np.percentile(ad,50):.2e} "
          f"p99={np.percentile(ad,99):.2e} max={ad.max():.2e}  "
          f"归一化 p50={np.percentile(ad,50)/scale:.2e} max={ad.max()/scale:.2e}")
    ok = np.percentile(ad, 50) / scale < 3.9e-3
    print("[fx] 判定(marlin):", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

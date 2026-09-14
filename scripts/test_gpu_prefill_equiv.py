#!/usr/bin/env python
"""GPU prefill(P1)的数值对拍:上游 `fused_experts`(MXFP4 W4A16)vs 我们的 CPU 引擎。

**为什么需要它**:v0.2 的 GPU-prefill 设计(`docs/GPU_PREFILL_MAINLINE.md`)要把大 batch 的
预填充从 CPU 引擎切到**主线上游的 GPU MoE 内核**,复用 `fused_experts(hidden, w1, w2, ...)`。
切之前必须证明**两边数值一致**,否则整条路线不可用。

本脚本**不加载模型**:直接吃一个真实 ckpt 的**一层** MXFP4 权重(与
`scripts/bench_cd_plumbing.py` 同一套加载器),
* CPU 侧:`engine.cpu_prefill(M, K, ids, wts, x_u16, out_f32)`(**纯主机调用**,无 D2H/H2D);
* GPU 侧:`vllm.model_executor.layers.fused_moe.fused_moe.fused_experts` +
  `mxfp4_w4a16_moe_quant_config`(与主线自己的 MXFP4 专家同一条内核);
* 输出:两边逐元素比较,给 max_abs / max_rel 与判定。

用法:
  CUDA_VISIBLE_DEVICES=2 XIAOTU_LAYER1_NPZ=<ckpt_dir> \
    /path/to/vllm-xiaotu-moe-env/bin/python scripts/test_gpu_prefill_equiv.py
  环境变量:M(默认 64) K(默认 6) LAYER(默认 3) CLAMP(默认 10.0,0=不夹)
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

H, I, GK, E, K = 4096, 2048, 32, 256, 6


def load_layer(ckpt: str, layer: int):
    """与 bench_cd_plumbing.py 完全一致的加载器(保证同一个布局)。"""
    shard = None
    for p in sorted(glob.glob(os.path.join(ckpt, "model-*.safetensors"))):
        with open(p, "rb") as f:
            hl = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(hl))
        if f"layers.{layer}.ffn.experts.0.w1.weight" in hdr:
            shard = p
            break
    assert shard, f"shard with layers.{layer} not found under {ckpt}"
    from safetensors import safe_open

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
    ckpt = os.environ.get("XIAOTU_LAYER1_NPZ", "")
    if not ckpt:
        raise SystemExit("set XIAOTU_LAYER1_NPZ=<dir with model-*.safetensors>")
    M = int(os.environ.get("M", "64"))
    K = int(os.environ.get("KE", str(globals()["K"])))   # 允许改 top-k 做隔离实验
    layer = int(os.environ.get("LAYER", "3"))
    clamp = float(os.environ.get("CLAMP", "10.0"))

    import xiaotu_moe

    w13, w2, s13, s2 = load_layer(ckpt, layer)
    rng = np.random.default_rng(7)

    # ---- 路由:每个 token 的 top-k,保证每个专家都可能被命中 ----
    ids = rng.integers(0, E, size=(M, K)).astype(np.uint32)
    wts = rng.random((M, K)).astype(np.float32)
    # 输入必须是**有限**的 bf16(随机字节会大量 NaN/Inf,对拍无意义):
    # 先造 fp32 正态,再转 bf16,取位模式给引擎。
    xf = (rng.standard_normal((M, H)).astype(np.float32) * 0.1)
    x_bf16 = torch.from_numpy(xf).to(torch.bfloat16)
    x = x_bf16.view(torch.uint16).numpy().copy()   # [M,H] uint16 = bf16 位模式

    # ---- CPU:引擎(纯主机调用) ----
    cfg = xiaotu_moe.MOEConfigV2()
    cfg.num_processes = 1
    cfg.process_id = 0
    cfg.gpu_id = 0
    cfg.has_gate_proj = True
    cfg.expert_num = E
    cfg.top_k = K
    cfg.hidden_size = H
    cfg.intermediate_size = I
    cfg.max_batch_size = max(M, 64)
    cfg.max_num_seqs = 256
    cfg.stride = 32
    cfg.group_min_len = 10
    cfg.group_max_len = max(M, 64) + 128
    cfg.groupN = 1
    cfg.groupK = GK
    cfg.activation_type = 1 if clamp > 0 else 0     # 1 = clamped SwiGLU(DS-V4)
    eng = xiaotu_moe.MOE_MXFP4(cfg, w13, w2, s13, s2, 0, 0)
    out_cpu = np.zeros((M, H), dtype=np.float32)
    eng.cpu_prefill(M, K, ids, wts, x, out_cpu)
    print(f"[cpu] done  shape={out_cpu.shape} mean|y|={np.abs(out_cpu).mean():.4e}")

    # ---- GPU:上游 fused_experts ----
    # 直接把 MXFP4 的 ocp_mx_scheme 丢给 functional 入口会被主线拒绝
    # ("Using ocp_mx_scheme=w_mxfp4 in functional fused_experts call is deprecated,
    #  please use OCP_MXQuantizationEmulationTritonExperts"),而那个类要求 AMD Quark。
    # ⇒ 对拍改为:**主机侧把 MXFP4 反量化成 bf16**,再走上游的**未量化** fused_experts。
    #    这样验证的是"路由 + 激活 + 夹取 + 累加"的语义一致性(量化格式本身已由
    #    scripts/test_block23_equiv.py 对过 numpy golden)。
    # 为了不让显存爆掉,只保留前 NE 个专家并把路由限制在它们上面。
    import torch as _t
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    NE = int(os.environ.get("NE", "8"))
    # 真实路由里**同一 token 的 top-k 专家互不相同**(vLLM 的 topk 就是在专家维上取的);
    # 早期版本用 ids%NE 会造出重复专家,触发引擎的"同行同专家去重"路径,
    # 与 golden 的"逐 assignment 累加"语义不同 ⇒ 对拍场景不真实。
    ids = np.stack([rng.choice(NE, size=K, replace=False) for _ in range(M)]).astype(np.uint32)
    LUT = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    lut16 = np.concatenate([LUT, -LUT]).astype(np.float32)     # 16 个 fp4 取值

    def deq(w_packed: np.ndarray, sc: np.ndarray, gn: int, gk: int):
        """[E, N, K/2] u8 + e8m0 [E, N/gn, K/gk] → bf16 [E, N, K]"""
        E_, N_, Kh = w_packed.shape
        Kk = Kh * 2
        lo = (w_packed & 0x0F).astype(np.uint8)
        hi = ((w_packed >> 4) & 0x0F).astype(np.uint8)
        out = np.empty((E_, N_, Kk), dtype=np.float32)
        out[..., 0::2] = lut16[lo]
        out[..., 1::2] = lut16[hi]
        s = np.exp2(sc.astype(np.int32) - 127).astype(np.float32)   # [E, N/gn, K/gk]
        s = np.repeat(np.repeat(s, gn, axis=1), gk, axis=2)
        return (out * s[:, :, :Kk]).astype(np.float32)

    dev = _t.device("cuda:0")
    w13b = deq(w13[:NE], s13[:NE], 1, GK)          # [NE, 2I, H]
    w2b = deq(w2[:NE], s2[:NE], 1, GK)             # [NE, H, I]
    t_w1 = _t.from_numpy(w13b).to(dev).to(_t.bfloat16)
    t_w2 = _t.from_numpy(w2b).to(dev).to(_t.bfloat16)
    hs = x_bf16.reshape(M, H).to(dev)
    t_ids = _t.from_numpy(ids.astype(np.int32)).to(dev)
    t_wts = _t.from_numpy(wts).to(dev)
    out_gpu = fused_experts(
        hs, t_w1, t_w2, t_wts, t_ids,
        activation=MoEActivation.SILU, global_num_experts=NE,
    )
    y_gpu = out_gpu.float().cpu().numpy()
    print(f"[gpu] done  shape={y_gpu.shape} mean|y|={np.abs(y_gpu).mean():.4e} (NE={NE})")

    # ---- numpy golden(三方对拍:谁错一目了然)----
    if os.environ.get("GOLDEN"):
        import torch.nn.functional as _F
        x0 = x_bf16.float().numpy()[0].astype(np.float32)
        # 必须把**全部 K 个 assignment** 按权重累加(引擎与 fused_experts 都是这么做的)
        ygold = np.zeros(H, dtype=np.float32)
        def bf16r(a):
            v = a.astype(np.float32).view(np.uint32)
            lsb = (v >> 16) & 1
            return (((v + 0x7FFF + lsb) >> 16) << 16).astype(np.uint32).view(np.float32)
        for kk in range(ids.shape[1]):
            e0 = int(ids[0, kk])
            gu = w13b[e0].astype(np.float32) @ x0
            act = _F.silu(torch.from_numpy(gu[:I])).numpy() * gu[I:]
            act = bf16r(act)      # ← 引擎在 gate/up 与 down 之间把激活存成 bf16
            ygold += float(wts[0, kk]) * (w2b[e0].astype(np.float32) @ act)
        print(f"[gold] K={ids.shape[1]} mean|y|={np.abs(ygold).mean():.4e}")
        print(f"[gold] cpu : max_abs={np.abs(out_cpu[0]-ygold).max():.4e}")
        print(f"[gold] gpu : max_abs={np.abs(y_gpu[0]-ygold).max():.4e}")

    # ---- 比较 ----
    a = out_cpu.reshape(-1)
    b = y_gpu.reshape(-1)
    denom = np.maximum(np.abs(a), 1e-6)
    max_abs = float(np.abs(a - b).max())
    rel = np.abs(a - b) / denom
    max_rel = float(rel.max())
    p99_rel = float(np.percentile(rel, 99))
    print(f"[cmp] max_abs={max_abs:.4e}  max_rel={max_rel:.4e}  p99_rel={p99_rel:.4e}")
    ok = max_rel < 2e-2
    print("[cmp]", "OK" if ok else "MISMATCH",
          f"(阈值 max_rel<2e-2;M={M} K={K} clamp={clamp})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

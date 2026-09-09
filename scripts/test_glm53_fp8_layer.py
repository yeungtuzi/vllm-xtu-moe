#!/usr/bin/env python
"""GLM-5.3-Flash 真实 fp8 专家权重的层内数值校验(不依赖注意力后端)。

GLM-5.3-Flash 在这台 A100(SM80)上**跑不起来**:主线没有支持它 MLA 维度
(qk_nope_head_dim=256 / v_head_dim=256 / qk_rope_head_dim=0)的 attention 后端
(见 logs/glm53_smoke.log / glm53_smoke2.log 的 selector 报错)。那是模型/硬件
层面的限制,与本插件的 CPU 后端无关。

本脚本绕开注意力,直接取 checkpoint 里某一层 MoE 的前 E 个专家(fp8 e4m3 +
128x128 块缩放,即主线的 kFp8Static128BlockSym),用 **通用路径同一套代码**
(mixed_experts.XiaotuCPUExpertsFp8 的引擎配置)跑一遍,和 numpy 参考实现比:

  gate = dequant(w1) @ x ; up = dequant(w3) @ x
  act  = clamp(gate,10) * sigmoid(clamp(gate,10)) * clamp(up,+-10)   # swiglu_limit
  act_bf16 = bf16(act)                                              # 引擎行为
  out  = sum_r w_r * dequant(w2_e) @ act_bf16

用法: python scripts/test_glm53_fp8_layer.py [层号] [专家数]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

MODEL = os.environ.get("GLM_MODEL", "")
if not MODEL:
    raise SystemExit("set GLM_MODEL=<local path of the GLM-5.3-Flash checkpoint>")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import xiaotu_moe  # noqa: E402


def f32_to_bf16_bits(x):
    v = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    lsb = (v >> 16) & 1
    return ((v + 0x7FFF + lsb) >> 16).astype(np.uint16)


def bf16_bits_to_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


# 引擎的 e4m3 次正规解码采用与 torch.float8_e4m3fn 一致的 2^-6;
# 置 GLM_FP8_ENGINE_DECODE=1 可复现旧版(2^-7)的语义,用于差异定位。
_ENGINE_DECODE = os.environ.get("GLM_FP8_ENGINE_DECODE") == "1"


def e4m3_to_f32(u8: np.ndarray) -> np.ndarray:
    """fp8 e4m3fn(sign=1,exp=4,mant=3)位模式 -> float32(与 torch 一致)。"""
    u = u8.astype(np.uint32)
    sign = (u >> 7) & 1
    exp = (u >> 3) & 0xF
    mant = u & 0x7
    sub_exp = -7 if _ENGINE_DECODE else -6
    out = np.where(exp == 0,
                   mant.astype(np.float32) / 8.0 * (2.0 ** sub_exp),
                   (1.0 + mant.astype(np.float32) / 8.0) * (2.0 ** (exp.astype(np.float32) - 7)))
    out = np.where(sign == 1, -out, out).astype(np.float32)
    if _ENGINE_DECODE:
        out = np.where((u8 == 0x7F) | (u8 == 0xFF), 0.0, out).astype(np.float32)
    return out


def expand_scale(scale: np.ndarray, N: int, K: int, block: int) -> np.ndarray:
    return np.repeat(np.repeat(scale, block, axis=0), block, axis=1)[:N, :K]


def main() -> int:
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    nexp = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    from safetensors import safe_open

    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    pre = f"model.language_model.layers.{layer}.mlp.experts."
    names = [
        "gate_proj.weight", "gate_proj.weight_scale_inv",
        "up_proj.weight", "up_proj.weight_scale_inv",
        "down_proj.weight", "down_proj.weight_scale_inv",
    ]
    if not any(pre + f"0.{n}" in wm for n in names):
        print(f"[glm53-layer] layer {layer} 没有专家权重,换一层试试")
        return 2

    # 按文件分组读取
    by_file: dict[str, list[str]] = {}
    for e in range(nexp):
        for n in names:
            key = pre + f"{e}.{n}"
            by_file.setdefault(wm[key], []).append(key)
    import torch

    tensors: dict[str, np.ndarray] = {}
    for fname, keys in by_file.items():
        with safe_open(os.path.join(MODEL, fname), framework="pt") as f:
            for k in keys:
                t = f.get_tensor(k)
                if t.dtype == torch.float8_e4m3fn:
                    tensors[k] = t.view(torch.uint8).numpy()   # e4m3 位模式
                else:
                    tensors[k] = t.float().numpy()

    def stack(suffix):
        return np.stack([tensors[pre + f"{e}.{suffix}"] for e in range(nexp)])

    gate_w, up_w, down_w = stack("gate_proj.weight"), stack("up_proj.weight"), stack("down_proj.weight")
    gate_s = stack("gate_proj.weight_scale_inv")
    up_s = stack("up_proj.weight_scale_inv")
    down_s = stack("down_proj.weight_scale_inv")
    E, I, H = gate_w.shape
    print(
        f"[glm53-layer] layer={layer} experts={E} I={I} H={H}\n"
        f"  gate_w {gate_w.shape} {gate_w.dtype} scale {gate_s.shape} {gate_s.dtype}\n"
        f"  down_w {down_w.shape} {down_w.dtype} scale {down_s.shape} {down_s.dtype}",
        flush=True,
    )
    assert gate_w.dtype == np.uint8 and gate_s.shape[-2:] == (I // 128, H // 128)

    # 引擎权重布局(与主线 FusedMoE 一致):w13=[E,2I,H](gate 在前,up 在后),
    # w2=[E,H,I];块缩放 [E,2I/128,H/128] / [E,H/128,I/128]
    w13 = np.ascontiguousarray(np.concatenate([gate_w, up_w], axis=1), dtype=np.uint8)
    s13 = np.ascontiguousarray(np.concatenate([gate_s, up_s], axis=1), dtype=np.float32)
    w2 = np.ascontiguousarray(down_w, dtype=np.uint8)
    s2 = down_s.astype(np.float32)

    rng = np.random.default_rng(3)
    M, K, LIMIT = 8, 8, 10.0
    x_u16 = f32_to_bf16_bits(rng.standard_normal((M, H)).astype(np.float32) * 3.0)
    xf = bf16_bits_to_f32(x_u16)
    ids = rng.integers(0, E, size=(M, K)).astype(np.uint32)
    wts = rng.uniform(0.05, 0.2, size=(M, K)).astype(np.float32)

    cfg = xiaotu_moe.MOEConfigV2()
    cfg.num_processes = 1
    cfg.process_id = 0
    cfg.gpu_id = 0
    cfg.has_gate_proj = True
    cfg.expert_num = E
    cfg.top_k = K
    cfg.hidden_size = H
    cfg.intermediate_size = I
    cfg.max_batch_size = 256
    cfg.max_num_seqs = 64
    cfg.stride = 128
    cfg.group_min_len = 10
    cfg.group_max_len = 4096 + 128
    cfg.groupN = 128
    cfg.groupK = 128
    cfg.activation_type = 1
    cfg.swiglu_limit = LIMIT
    cfg.swiglu_alpha = 1.0
    cfg.swiglu_beta = 0.0

    out = np.zeros((M, H), dtype=np.float32)
    eng = xiaotu_moe.MOE_FP8(cfg, w13, w2, s13, s2, 0, 0)
    eng.cpu_prefill(M, K, ids, wts, x_u16, out)

    # numpy 参考(与引擎同语义:块缩放反量化 + bf16(act))
    deq13 = e4m3_to_f32(w13) * np.stack(
        [expand_scale(s13[e], 2 * I, H, 128) for e in range(E)]
    )
    deq2 = e4m3_to_f32(w2) * np.stack(
        [expand_scale(s2[e], H, I, 128) for e in range(E)]
    )
    golden = np.zeros((M, H), dtype=np.float32)
    for t in range(M):
        for r in range(K):
            e, w = int(ids[t, r]), float(wts[t, r])
            g = deq13[e, :I] @ xf[t]
            u = deq13[e, I:] @ xf[t]
            g = np.minimum(g, LIMIT)
            u = np.clip(u, -LIMIT, LIMIT)
            act = (g / (1.0 + np.exp(-g))) * u
            act_b = bf16_bits_to_f32(f32_to_bf16_bits(act))
            golden[t] += w * (deq2[e] @ act_b)

    ad = np.abs(out - golden)
    rms = float(np.sqrt(np.mean(golden**2)))
    rms_rel = float(np.sqrt(np.mean((out - golden) ** 2)) / (rms + 1e-12))
    # 引擎在 act 之后转 bf16 再算 down,所以相对误差下限就是 bf16 精度(~4e-3/元素,
    # 整层 RMS 上远小于它)。用 RMS 相对误差 + max_abs/rms 两个稳健指标。
    ok = rms_rel < 1e-3 and float(ad.max()) < 5e-3 * rms
    print(
        f"[glm53-layer] FP8 block128 engine vs numpy-golden: "
        f"rms_rel={rms_rel:.3e} max_abs={ad.max():.3e} "
        f"max_abs/rms={float(ad.max())/rms:.3e} golden_rms={rms:.4f} "
        f"-> {'OK' if ok else 'FAIL'}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

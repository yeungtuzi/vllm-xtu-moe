"""MXFP4(N-parallel/nsliced 路径)上 clamped SwiGLU 的数值一致性测试。

`forward_many_nsliced` 里激活是和 f32->bf16 融合写的(与 forward_many 的两处
独立循环是**不同代码**),所以单独测一遍:用真实 DS-V4 layer-1 的 MXFP4 权重
(npz fixture)+ 引擎语义完全一致的 numpy golden。

用法: python scripts/test_swiglu_clamp_mxfp4.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.environ.get("XIAOTU_LAYER1_NPZ", "")
if not NPZ or not os.path.exists(NPZ):
    raise SystemExit(
        "set XIAOTU_LAYER1_NPZ=<npz with w13/w2/gol13/gol2/g13/g2/E/I/H of one real "
        "MXFP4 expert layer>; the self-contained numeric test is "
        "scripts/test_swiglu_clamp.py"
    )
K = 6


def f32_to_bf16_bits(x):
    v = x.astype(np.float32).view(np.uint32)
    lsb = (v >> 16) & 1
    return ((v + 0x7FFF + lsb) >> 16).astype(np.uint16)


def bf16_bits_to_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


def main() -> int:
    d = np.load(NPZ)
    w13, w2 = d["w13"], d["w2"]
    gol13, gol2 = d["gol13"], d["gol2"]
    E, I, H = int(d["E"]), int(d["I"]), int(d["H"])

    def f32_to_e8m0(s):
        lg = np.round(np.log2(np.maximum(s, 1e-30))).astype(np.int32) + 127
        return np.clip(lg, 0, 255).astype(np.uint8)

    s13 = f32_to_e8m0(d["g13"])
    s2 = f32_to_e8m0(d["g2"])

    sys.path.insert(0, REPO)
    import xiaotu_moe

    m = xiaotu_moe.load()
    rng = np.random.default_rng(11)
    M = 8
    x_u16 = f32_to_bf16_bits(rng.standard_normal((M, H)).astype(np.float32) * 6.0)
    xf = bf16_bits_to_f32(x_u16)
    ids = rng.integers(0, E, size=(M, K)).astype(np.int32)
    wts = rng.uniform(-1, 1, size=(M, K)).astype(np.float32)

    rc = 0
    for limit, alpha, beta in [(0.0, 1.0, 0.0), (10.0, 1.0, 0.0), (10.0, 1.5, 0.0)]:
        cfg = m.MOEConfigV2()
        cfg.num_processes = 1
        cfg.process_id = 0
        cfg.gpu_id = 0
        cfg.has_gate_proj = True
        cfg.expert_num = E
        cfg.top_k = K
        cfg.hidden_size = H
        cfg.intermediate_size = I
        cfg.max_batch_size = 256
        cfg.max_num_seqs = 256
        cfg.stride = 32
        cfg.group_min_len = 10
        cfg.group_max_len = 4096 + 128
        cfg.groupN = 1
        cfg.groupK = 32
        cfg.activation_type = 1 if (limit > 0 or alpha != 1.0 or beta != 0.0) else 0
        cfg.swiglu_limit = limit
        cfg.swiglu_alpha = alpha
        cfg.swiglu_beta = beta
        engine = m.MOE_MXFP4(cfg, w13, w2, s13, s2, 0, 0)
        out = np.zeros((M, H), dtype=np.float32)
        engine.cpu_prefill(M, K, ids, wts, x_u16, out)

        golden = np.zeros((M, H), dtype=np.float32)
        for t in range(M):
            for r in range(K):
                e, w = ids[t, r], wts[t, r]
                if w == 0.0:
                    continue
                g = gol13[e, 0:I] @ xf[t]
                u = gol13[e, I:2 * I] @ xf[t]
                if limit > 0:
                    g = np.minimum(g, limit)
                    u = np.clip(u, -limit, limit)
                act = (g / (1.0 + np.exp(-alpha * g))) * (u + beta)
                act_bf16 = bf16_bits_to_f32(f32_to_bf16_bits(act))
                golden[t] += w * (gol2[e] @ act_bf16)

        ad = np.abs(out - golden)
        rel = ad / (np.abs(golden) + 1e-6)
        big = np.abs(golden) > 0.05
        mx = float(rel[big].max()) if big.any() else float(rel.max())
        ok = mx < 2e-3
        rc |= 0 if ok else 1
        print(
            f"[{'OK ' if ok else 'BAD'}] MXFP4/nsliced limit={limit:<5} alpha={alpha:<4} "
            f"beta={beta:<4} max_abs={ad.max():.3e} max_rel(|g|>0.05)={mx:.3e}",
            flush=True,
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""强制 me=2 / me=3 路由下的 MXFP4-Kernel 数值对拍。

为什么要单独测:2026-09-10 给 `matmul_packed4_group` 增加了 **2/3 行分块**路径
(原来只有 4 行分块 + 单行兜底,而真实解码 me≈3 一直走单行,每行重复解码权重,
实测每 assignment 慢 1.78×)。新增代码必须对拍:本脚本用**受控路由**让每个专家
恰好被 2 个或 3 个 token 命中,逐元素对比 numpy golden。

用法: XIAOTU_LAYER1_NPZ=<real_layer1_model.npz> python scripts/test_block23_equiv.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.environ.get("XIAOTU_LAYER1_NPZ", "")
if not NPZ:
    raise SystemExit("set XIAOTU_LAYER1_NPZ=<npz with one real MXFP4 layer>")

H, I, K = 4096, 2048, 6


def f32_to_bf16_bits(x):
    v = x.astype(np.float32).view(np.uint32)
    lsb = (v >> 16) & 1
    return ((v + 0x7FFF + lsb) >> 16).astype(np.uint16)


def bf16_bits_to_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


def run_case(m, w13, w2, s13, s2, gol13, gol2, E, ids, wts, x_u16, xf, label):
    M = ids.shape[0]
    cfg = m.MOEConfigV2()
    cfg.num_processes = 1; cfg.process_id = 0; cfg.gpu_id = 0
    cfg.has_gate_proj = True; cfg.expert_num = E; cfg.top_k = K
    cfg.hidden_size = H; cfg.intermediate_size = I
    cfg.max_batch_size = 256; cfg.max_num_seqs = 256
    cfg.stride = 32; cfg.group_min_len = 10; cfg.group_max_len = 4096 + 128
    cfg.groupN = 1; cfg.groupK = 32; cfg.activation_type = 0
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
            act = (g / (1.0 + np.exp(-g))) * u
            golden[t] += w * (gol2[e] @ bf16_bits_to_f32(f32_to_bf16_bits(act)))

    ad = np.abs(out - golden)
    rel = ad / (np.abs(golden) + 1e-6)
    big = np.abs(golden) > 0.05
    mx = float(rel[big].max()) if big.any() else float(rel.max())
    ok = mx < 2e-3
    # 每个专家的命中次数(me)分布,确认真的覆盖了 2/3 行分块
    counts = {}
    for t in range(M):
        for r in range(K):
            counts[ids[t, r]] = counts.get(ids[t, r], 0) + 1
    mes = sorted(counts.values())
    print(f"[{'OK ' if ok else 'BAD'}] {label:26s} M={M} me分布={mes} "
          f"max_abs={ad.max():.3e} max_rel={mx:.3e}")
    return 0 if ok else 1


def main() -> int:
    d = np.load(NPZ)
    w13, w2, gol13, gol2 = d["w13"], d["w2"], d["gol13"], d["gol2"]
    E = int(d["E"])

    def f32_to_e8m0(s):
        lg = np.round(np.log2(np.maximum(s, 1e-30))).astype(np.int32) + 127
        return np.clip(lg, 0, 255).astype(np.uint8)

    s13, s2 = f32_to_e8m0(d["g13"]), f32_to_e8m0(d["g2"])
    sys.path.insert(0, REPO)
    import xiaotu_moe
    m = xiaotu_moe.load()
    print(f"[block23] variant={xiaotu_moe.__variant__}")
    rng = np.random.default_rng(23)
    wts_all = rng.uniform(-1, 1, size=(64, K)).astype(np.float32)

    def make(plan):
        """plan: list of (expert, repeat) -> 逐 token 填 K 个 assignment。"""
        rows = []
        for e, rep in plan:
            rows += [e] * rep
        rows = rows[: len(rows) // K * K]
        M = len(rows) // K
        ids = np.array(rows, dtype=np.int32).reshape(M, K)
        x_u16 = f32_to_bf16_bits(rng.standard_normal((M, H)).astype(np.float32) * 6.0)
        return ids, wts_all[:M].copy(), x_u16, bf16_bits_to_f32(x_u16)

    rc = 0
    # 每个 case 的 assignment 总数都取 6 的倍数(= K),且专家 id < E(本 fixture E=16)
    cases = {
        "me=1(6 个专家)": [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)],
        "me=2(9 个专家)": [(e, 2) for e in range(1, 10)],
        "me=3(12 个专家)": [(e, 3) for e in range(1, 13)],
        "me=2/3 混合": [(e, 3) for e in range(1, 7)] + [(e, 2) for e in range(7, 10)],
        "me=4(旧快路径)": [(e, 4) for e in range(1, 4)],
        "me=5(4+1)": [(e, 5) for e in range(1, 7)],
        "me=6(4+2)": [(e, 6) for e in range(1, 7)],
        "me=7(4+3)": [(e, 7) for e in range(1, 7)],
    }
    for label, plan in cases.items():
        ids, wts, x_u16, xf = make(plan)
        if ids.shape[0] == 0:
            continue
        rc |= run_case(m, w13, w2, s13, s2, gol13, gol2, E, ids, wts, x_u16, xf, label)
    return rc


if __name__ == "__main__":
    sys.exit(main())

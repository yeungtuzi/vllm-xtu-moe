#!/usr/bin/env python3
"""把 GPU 预填【装配相】拆开计时 —— 用于判定 250 ms/层 到底花在哪。

═══════════════════════════════════════════════════════════════════════════
 为什么要测(2026-10-08)
═══════════════════════════════════════════════════════════════════════════
目标 `goal-733e1da7` 的前提来自 **B285**:
    H2D ~3.84 GB @15–27 GB/s = **140–250 ms**  +  设备侧 K-major 转置 ~75 ms @84 GB/s
⇒ 但 `gpu_prefill.py:678-680`(**§596**)声称转置已改成 128×128 分块 Triton ⇒ **~600 GB/s、6.5 ms/层**
⇒ ⭐ **两个说法矛盾**(75 ms vs 6.5 ms)⇒ 必须实测,否则会去优化一个已经不存在的问题 ✗

本脚本**不改引擎代码**,只用合成分片引擎把相拆开:
    t_total      = `kmajor_from_engine_shards_fp8(...)` 全程(= H2D + 转置)
    t_transpose  = 单独跑 `_kmajor_bytes`(同形状)
    ⇒ t_H2D ≈ t_total − t_transpose
并算出各自的 GB/s,与链路实测上限对比(⚠️ **GPU0 = x8 ≈ 12.5 GB/s;GPU1/2 = x16 ≈ 25 GB/s**)

用法:
  GPU=1 E=64 H=7168 I=2048 REP=5 python scripts/bench_gpf_asm_phases.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import xiaotu_moe  # noqa: E402
from vllm_xiaotu_moe.gpu_prefill import _kmajor_bytes  # noqa: E402
from vllm_xiaotu_moe.gpu_prefill_fp8 import (  # noqa: E402
    kmajor_from_engine_shards_fp8,
)

BLOCK = 128


def gib(n: float) -> str:
    return f"{n / 2**30:.3f} GiB"


def rate(nbytes: float, ms: float) -> float:
    return (nbytes / 2**30) / (ms / 1e3) if ms > 0 else 0.0


def main() -> int:
    E = int(os.environ.get("E", 64))
    H = int(os.environ.get("H", 7168))
    I = int(os.environ.get("I", 2048))
    REP = int(os.environ.get("REP", 5))
    gi = int(os.environ.get("GPU", 1))
    dev = torch.device(f"cuda:{gi}")
    torch.cuda.set_device(gi)
    print(f"设备 cuda:{gi} · E={E} H={H} I={I} · REP={REP}")

    g = np.random.default_rng(0)
    w13 = g.integers(0, 256, size=(E, 2 * I, H), dtype=np.uint8)
    w2 = g.integers(0, 256, size=(E, H, I), dtype=np.uint8)
    s13 = (10.0 ** g.uniform(-2, 1, size=(E, 2 * I // BLOCK, H // BLOCK))).astype(np.float32)
    s2 = (10.0 ** g.uniform(-2, 1, size=(E, H // BLOCK, I // BLOCK))).astype(np.float32)

    cfg = xiaotu_moe.MOEConfigV2()
    for k, v in dict(num_processes=1, process_id=0, gpu_id=gi, has_gate_proj=True,
                     expert_num=E, top_k=1, hidden_size=H, intermediate_size=I,
                     max_batch_size=64, max_num_seqs=8, stride=BLOCK,
                     group_min_len=10, group_max_len=1024, groupN=BLOCK, groupK=BLOCK,
                     activation_type=1, swiglu_limit=10.0, swiglu_alpha=1.0,
                     swiglu_beta=0.0).items():
        setattr(cfg, k, v)
    eng = xiaotu_moe.MOE_FP8(cfg, w13, w2, s13, s2, 0, 0)
    geo = eng.shard_geometry()
    if int(geo["ns"]) < 2:
        raise SystemExit("engine did not shard")

    # ⭐ 每层要搬的字节:w13 [E,2I,H] + w2 [E,H,I](u8) + 两个 scales(fp32)
    bytes_w13 = E * 2 * I * H
    bytes_w2 = E * H * I
    bytes_s13 = E * (2 * I // BLOCK) * (H // BLOCK) * 4
    bytes_s2 = E * (H // BLOCK) * (I // BLOCK) * 4
    bytes_weights = bytes_w13 + bytes_w2
    print(f"  权重 {gib(bytes_weights)}(w13 {gib(bytes_w13)} + w2 {gib(bytes_w2)})"
          f" + scales {gib(bytes_s13 + bytes_s2)}")

    # ── 预热 ──
    built = kmajor_from_engine_shards_fp8(eng, dev, H, I, E)
    torch.cuda.synchronize()
    w13t, s13t, w2t, s2t = built

    # ── 相 1:t_total(H2D + 转置)──
    tot = []
    for _ in range(REP):
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        kmajor_from_engine_shards_fp8(eng, dev, H, I, E)
        e1.record(); torch.cuda.synchronize()
        tot.append(e0.elapsed_time(e1))

    # ── 相 2:t_transpose(单独跑 _kmajor_bytes,同形状)──
    raw13 = torch.empty((E, 2 * I, H), dtype=torch.uint8, device=dev)
    raw13.copy_(torch.from_numpy(w13).to(dev))
    kmajor_from_engine_shards_fp8(eng, dev, H, I, E)   # 再热一次
    dst13 = torch.empty((E, H, 2 * I), dtype=torch.uint8, device=dev)
    _kmajor_bytes(raw13, dst13); torch.cuda.synchronize()
    tr = []
    for _ in range(REP):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); _kmajor_bytes(raw13, dst13); e1.record(); torch.cuda.synchronize()
        tr.append(e0.elapsed_time(e1))

    med_tot = float(np.median(tot)); med_tr = float(np.median(tr))
    med_dma = med_tot - med_tr
    print()
    print("=== 相分离(中位,ms/层)===")
    print(f"  t_total(全程)        = {med_tot:8.2f} ms")
    print(f"  t_transpose(_kmajor) = {med_tr:8.2f} ms   ⇒ {rate(bytes_weights*2, med_tr):7.1f} GB/s"
          f"(读+写)")
    print(f"  t_H2D(≈ 差值)        = {med_dma:8.2f} ms   ⇒ {rate(bytes_weights+bytes_s13+bytes_s2, med_dma):7.1f} GB/s")
    print(f"  ⭐ 转置占全程 = {100*med_tr/med_tot:.1f}%")
    print()
    print("=== A/B:两臂交替,看 H2D 是否稳定(有无 lazy submission 的迹象)===")
    alt = []
    for _ in range(REP):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); kmajor_from_engine_shards_fp8(eng, dev, H, I, E); e1.record()
        torch.cuda.synchronize(); alt.append(e0.elapsed_time(e1))
    print(f"  交替臂 t_total 中位 = {float(np.median(alt)):.2f} ms(与上面 {med_tot:.2f} 对比)")
    print()
    print("⇒ ⭐ 判定:若 **转置占比已经很小(§596 的 6.5 ms)** ⇒ 目标里的 ② 只有 ~2.6% 天花板,")
    print("   **250 ms 几乎全在 H2D** ⇒ 应把力气放在 A 组 ① 与 C 组 ④⑤ ✗②")
    print("⇒ 若转置仍 ~75 ms(占 30%) ⇒ B285 的口径仍成立 ⇒ ② 值得做 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Is the xiaotu CPU engine **bit-deterministic** for identical inputs?

Why this exists (NOTES §365): on the real service, the same greedy request run
twice gave only 2/5 identical outputs (same config, same seed, temperature=0).
That makes "greedy consistency" unusable as an acceptance criterion until we know
*whether the non-determinism comes from our engine* or from vLLM.

This runs without loading a model: one engine, the same input N times, compare
bit-exactly. Reports where the first differing bit appears so we can tell a
reduction-order effect from a genuine bug.

Usage: python scripts/test_engine_determinism.py [N]
Env:   XIAOTU_LAYER1_NPZ=<ckpt dir with model-*.safetensors> (default: model snapshot)
Exit 0 = bit-identical across all runs.

License: Apache-2.0
"""

import os
import sys

os.environ.setdefault("XIAOTU_MOE_NO_AUTO_EP", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import xiaotu_moe  # noqa: E402

E, H, I = 8, 256, 512
TOPK, QLEN = 2, 4
GROUP_K, GROUP_N = 32, 1
SCALE_ZERO = 127          # e8m0: 2^(127-127) = 1.0
REP = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def main() -> int:
    rng = np.random.default_rng(2024)
    w13 = rng.integers(0, 256, size=(E, 2 * I, H // 2), dtype=np.uint8)
    w13_g = np.full((E, 2 * I, H // GROUP_K), SCALE_ZERO, dtype=np.uint8)
    w2 = rng.integers(0, 256, size=(E, H, I // 2), dtype=np.uint8)
    w2_g = np.full((E, H, I // GROUP_K), SCALE_ZERO, dtype=np.uint8)

    cfg = xiaotu_moe.MOEConfigV2()
    cfg.num_processes = 1
    cfg.process_id = 0
    cfg.gpu_id = 0
    cfg.has_gate_proj = True
    cfg.expert_num = E
    cfg.top_k = TOPK
    cfg.hidden_size = H
    cfg.intermediate_size = I
    cfg.max_batch_size = 1024
    cfg.max_num_seqs = 64
    cfg.stride = 32
    cfg.group_min_len = 1
    cfg.group_max_len = 1024
    cfg.activation_type = 0
    cfg.use_gpu_prefill = False
    cfg.groupN = GROUP_N
    cfg.groupK = GROUP_K
    eng = xiaotu_moe.MOE_MXFP4(
        cfg, w13.ctypes.data, w2.ctypes.data, w13_g.ctypes.data, w2_g.ctypes.data, 0, 0)

    # 固定输入:同一路由、同一激活,连跑 REP 次
    ids = np.tile(np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.uint32), (1, 1))
    wts = np.array([[0.6, 0.4], [0.5, 0.5], [0.7, 0.3], [0.2, 0.8]], dtype=np.float32)
    x = (rng.standard_normal((QLEN, H)) * 0.1).astype(np.float16).view(np.uint16)

    outs = []
    for _ in range(REP):
        o = np.zeros((QLEN, H), dtype=np.float32)
        eng.cpu_prefill(QLEN, TOPK, ids, wts, x, o)
        outs.append(o.copy())

    ref = outs[0]
    bad = 0
    for i, o in enumerate(outs[1:], 1):
        if np.array_equal(o, ref):
            continue
        bad += 1
        d = np.argwhere(o != ref)
        first = d[0] if len(d) else None
        print(f"  run{i:3d}: DIFFERS  n_diff={len(d):6d}  "
              f"first@[{(first[0], first[1]) if first is not None else None}] "
              f"ref={ref[tuple(first)] if first is not None else None} "
              f"got={o[tuple(first)] if first is not None else None}  "
              f"max|Δ|={np.abs(o - ref).max():.3e}")

    print(f"\nengine single-rank: {REP} runs, {REP - 1 - bad}/{REP - 1} bit-identical "
          f"to run0, {bad} differ")
    if bad == 0:
        print("VERDICT: engine is bit-deterministic for identical input "
              "=> look for the non-determinism in TP=2/EP reduction or in vLLM.")
        return 0
    print("VERDICT: engine is NON-deterministic => fix it here first.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

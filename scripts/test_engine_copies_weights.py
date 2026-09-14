#!/usr/bin/env python
"""Does the xiaotu engine COPY the caller's weights, or only keep pointers?

Premise for the v0.2 memory fix (NOTES §344–§346). `hybrid_model.finalize_mega_moe_weights`
deliberately keeps `self._w13 = w13` alive with the comment "引擎持有这些参数的内存
(达 data_ptr),必须防被替换/释放" — but `moe_v2.hpp` `shard_fill_w13/w2` appear to
`memcpy` into engine-owned `mbind`'d shards. If the engine really copies, that
defensive hold is unnecessary and the vLLM-side parameter can be shrunk to this
rank's EP shard, halving per-worker host RAM (138 GB -> 69 GB).

Test (no model load, ~1 s): build a tiny engine, run a forward, then **clobber the
caller's weight buffers in place** (same address, different bytes) and run again.
Identical outputs => the engine owns its own copy. Different => it aliases ours.

Usage: python scripts/test_engine_copies_weights.py
Exit code 0 = engine copies (safe to release the caller's weights).

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
GROUP_K = 32
GROUP_N = 1
SCALE_ZERO = 127          # e8m0: 2^(127-127) = 1.0


def make_weights(seed: int):
    rng = np.random.default_rng(seed)
    w13 = rng.integers(0, 256, size=(E, 2 * I, H // 2), dtype=np.uint8)
    w13_g = np.full((E, 2 * I, H // GROUP_K), SCALE_ZERO, dtype=np.uint8)
    w2 = rng.integers(0, 256, size=(E, H, I // 2), dtype=np.uint8)
    w2_g = np.full((E, H, I // GROUP_K), SCALE_ZERO, dtype=np.uint8)
    return w13, w13_g, w2, w2_g


def build():
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
    return cfg


def run(engine, ids, wts, x, out):
    engine.cpu_prefill(QLEN, TOPK, ids, wts, x, out)


def main() -> int:
    w13, w13_g, w2, w2_g = make_weights(1234)
    cfg = build()
    engine = xiaotu_moe.MOE_MXFP4(
        cfg, w13.ctypes.data, w2.ctypes.data, w13_g.ctypes.data, w2_g.ctypes.data, 0, 0)

    rng = np.random.default_rng(7)
    ids = rng.integers(0, E, size=(QLEN, TOPK), dtype=np.uint32)
    wts = rng.random((QLEN, TOPK), dtype=np.float32)
    x = (rng.standard_normal((QLEN, H)) * 0.1).astype(np.float16).view(np.uint16)

    out1 = np.zeros((QLEN, H), dtype=np.float32)
    run(engine, ids, wts, x, out1)

    # internal view of the engine's own weights (first 8 halves)
    d13_before, d2_before = engine.debug_first(16)

    # ---- clobber the caller's buffers IN PLACE (same address, new bytes) ----
    w13[:] = 0
    w13_g[:] = 255        # 2^(255-127) -> inf-ish, would explode if aliased
    w2[:] = 0
    w2_g[:] = 255

    out2 = np.zeros((QLEN, H), dtype=np.float32)
    run(engine, ids, wts, x, out2)

    d13_after, _ = engine.debug_first(16)

    same_out = np.array_equal(out1, out2)
    same_internal = np.array_equal(np.asarray(d13_before), np.asarray(d13_after))
    finite = bool(np.isfinite(out2).all())

    print(f"out1[:4]        = {out1.reshape(-1)[:4]}")
    print(f"out2[:4]        = {out2.reshape(-1)[:4]}")
    print(f"internal w13 same after clobber : {same_internal}")
    print(f"output identical after clobber  : {same_out}")
    print(f"output still finite             : {finite}")
    print(f"max|out1-out2|                  = {np.abs(out1 - out2).max():.6g}")

    if same_out and same_internal and finite:
        print("\nVERDICT: engine COPIES weights -> caller's buffers may be released.")
        return 0
    print("\nVERDICT: engine ALIASES the caller's weights -> must NOT release them.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Controlled FP8 CPU-MoE M-curve: ms and effective weight bandwidth vs M.

``M`` is the number of rows a single expert receives. We force it exactly by
using top_k = 1 and routing assignment ``t`` to expert ``t % E``, with
``B = M * E`` tokens -- so every expert gets exactly ``M`` rows (the same trick
``DEDUP`` uses for the packed4 engine).

Weight bytes per call are fixed at one full read of every expert
(``E * (2I*H + H*I)`` bytes for fp8), so:

  * if the kernel decodes each weight once per MR rows, the *effective* traffic
    is ``ceil(M/MR) * that``;
  * ``GB/s = weight_bytes / time`` therefore drops by ~MR when M grows past MR
    on an unblocked kernel, and stays flat on a blocked one.

Usage:
  python scripts/bench_fp8_mcurve.py [E H I] [M ...]
  default: GLM-5.3-Flash shape E=288 H=4096 I=2048, M=1..12
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import xiaotu_moe  # noqa: E402

GROUP = 128


def main() -> int:
    args = sys.argv[1:]
    if len(args) >= 3:
        E, H, I = (int(x) for x in args[:3])
        Ms = [int(x) for x in args[3:]] or list(range(1, 13))
    else:
        E, H, I = 288, 4096, 2048
        Ms = list(range(1, 13))
    warmup = int(os.environ.get("WARMUP", "20"))
    rep = int(os.environ.get("REP", "60"))
    threads = int(os.environ.get("XIAOTU_MOE_THREADS", "0"))
    if threads:
        os.environ["XIAOTU_MOE_THREADS"] = str(threads)

    rng = np.random.default_rng(0)
    w13 = rng.integers(0, 256, size=(E, 2 * I, H), dtype=np.uint8)
    w2 = rng.integers(0, 256, size=(E, H, I), dtype=np.uint8)
    s13 = rng.uniform(1e-3, 2e-2, size=(E, 2 * I // GROUP, H // GROUP)).astype(np.float32)
    s2 = rng.uniform(1e-3, 2e-2, size=(E, H // GROUP, I // GROUP)).astype(np.float32)

    cfg = xiaotu_moe.MOEConfigV2()
    cfg.num_processes = 1
    cfg.process_id = 0
    cfg.gpu_id = 0
    cfg.has_gate_proj = True
    cfg.expert_num = E
    cfg.top_k = 1
    cfg.hidden_size = H
    cfg.intermediate_size = I
    cfg.max_batch_size = max(Ms) * E + 8
    cfg.max_num_seqs = 512
    cfg.stride = GROUP
    cfg.group_min_len = 10
    cfg.group_max_len = 4096 + 128
    cfg.groupN = GROUP
    cfg.groupK = GROUP
    cfg.activation_type = 0

    t0 = time.time()
    eng = xiaotu_moe.MOE_FP8(
        cfg, w13.ctypes.data, w2.ctypes.data, s13.ctypes.data, s2.ctypes.data, 0, 0
    )
    wbytes = int(w13.nbytes + w2.nbytes)
    print(
        f"[mcurve] E={E} H={H} I={I} top_k=1  weights={wbytes/1e9:.2f} GB  "
        f"engine build {time.time()-t0:.1f}s  threads={os.environ.get('XIAOTU_MOE_THREADS','auto')}"
    )
    x = (rng.standard_normal((max(Ms) * E,) * 1 + (H,)) * 0.5).astype(np.float32)
    xb = (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)

    print(f"{'M':>3} {'B':>6} {'ms':>9} {'GB/s(eff)':>10} {'tok/s':>10} {'x@M=1':>7}")
    base = None
    for M in Ms:
        B = M * E
        ids = np.arange(B, dtype=np.uint32) % E
        ids = ids.reshape(B, 1)
        wts = np.ones((B, 1), dtype=np.float32)
        xin = np.ascontiguousarray(xb[:B].reshape(-1))
        out = np.zeros((B, H), dtype=np.float32)
        for _ in range(warmup):
            eng.cpu_prefill(B, 1, ids, wts, xin, out)
        t = time.perf_counter()
        for _ in range(rep):
            eng.cpu_prefill(B, 1, ids, wts, xin, out)
        dt = (time.perf_counter() - t) / rep
        ms = dt * 1e3
        gbs = wbytes / dt / 1e9
        toks = B / dt
        if base is None:
            base = ms
        print(f"{M:>3} {B:>6} {ms:>9.3f} {gbs:>10.1f} {toks:>10.0f} {ms/base:>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

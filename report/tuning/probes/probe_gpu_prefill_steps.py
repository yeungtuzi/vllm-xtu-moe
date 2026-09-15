#!/usr/bin/env python
"""Localize the V4.1-dim illegal memory access inside gpu_moe_layer.

`gpu_moe_layer` works on V4 dims (H=4096/I=2048/E=256) and faults on V4.1 dims
(H=5120/I=2304/E=384) even at M=8, so it is a shape problem, not a batch problem.
The traceback only ever points at `_down_kernel`'s first launch (a *sticky* CUDA
error surfacing at the next API call), which is why this walks the pipeline step by
step with a hard synchronize + error check after each stage.

Run: XIAOTU_LAYER1_NPZ=<ckpt> CUDA_LAUNCH_BLOCKING=1 python probe_gpu_prefill_steps.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, HERE)

from probe_gpu_prefill_v41 import E, GK, H, I, K, f32_to_bf16_bits, load_layer  # noqa: E402


def step(name, fn):
    try:
        r = fn()
        torch.cuda.synchronize()
        print(f"  OK   {name}", flush=True)
        return r
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {name}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        raise SystemExit(1)


def main() -> int:
    import triton
    from vllm_xiaotu_moe import gpu_prefill as gp

    layer = int(os.environ.get("LAYER", "3"))
    M = int(os.environ.get("M", "8"))
    print(f"[steps] H={H} I={I} E={E} K={K} GK={GK} M={M}", flush=True)
    w13, w2, s13, s2 = load_layer(layer)
    dev = torch.device("cuda:0")

    BM = int(os.environ.get("XIAOTU_GPU_PREFILL_BM", "64"))
    BN = int(os.environ.get("XIAOTU_GPU_PREFILL_BN", "64"))
    BK = int(os.environ.get("XIAOTU_GPU_PREFILL_BK", "64"))
    BH = int(os.environ.get("XIAOTU_GPU_PREFILL_BH", "64"))
    print(f"[steps] BM={BM} BN={BN} BK={BK} BH={BH}", flush=True)

    rng = np.random.default_rng(7)
    ids = rng.integers(0, E, size=(M, K)).astype(np.int32)
    wts = rng.uniform(-1, 1, size=(M, K)).astype(np.float32)
    x = f32_to_bf16_bits(rng.standard_normal((M, H)).astype(np.float32))

    xs = torch.from_numpy(x.view(np.uint16)).view(torch.bfloat16).to(dev)
    ids_t = torch.from_numpy(ids).to(dev)
    wts_t = torch.from_numpy(wts).to(dev)

    print("-- host->device upload of weights --", flush=True)
    w13_t = step("w13 kmajor H2D", lambda: gp._kmajor_bytes(gp._pinned(torch.from_numpy(w13)).to(dev, non_blocking=True)))
    s13_t = step("s13 kmajor H2D", lambda: gp._kmajor_bytes(gp._pinned(torch.from_numpy(s13)).to(dev, non_blocking=True)))
    w2_t = step("w2 kmajor H2D", lambda: gp._kmajor_bytes(gp._pinned(torch.from_numpy(w2)).to(dev, non_blocking=True)))
    s2_t = step("s2 kmajor H2D", lambda: gp._kmajor_bytes(gp._pinned(torch.from_numpy(s2)).to(dev, non_blocking=True)))
    print(f"  w13_t {tuple(w13_t.shape)} strides {w13_t.stride()}", flush=True)
    print(f"  s13_t {tuple(s13_t.shape)} strides {s13_t.stride()}", flush=True)
    print(f"  w2_t  {tuple(w2_t.shape)} strides {w2_t.stride()}", flush=True)
    print(f"  s2_t  {tuple(s2_t.shape)} strides {s2_t.stride()}", flush=True)

    tok, wts_s, seg_start, A = step("_build_segmentation", lambda: gp._build_segmentation(ids_t, wts_t, E, dev))
    print(f"  A={A} tok={tuple(tok.shape)} seg_start={tuple(seg_start.shape)} "
          f"max_seg={int(seg_start.max())} min={int(seg_start.min())}", flush=True)
    assert int(seg_start.max()) <= A, "segment end exceeds A!"

    out = torch.zeros((M, H), dtype=torch.bfloat16, device=dev)
    inter = step("alloc inter", lambda: torch.zeros((A, 2 * I), dtype=torch.bfloat16, device=dev))
    lut_t = gp._e2m1_table(dev)
    NW = int(os.environ.get("XIAOTU_GPU_PREFILL_WARPS", "4"))

    print("-- kernels, one at a time --", flush=True)
    step("_gate_up_kernel", lambda: gp._gate_up_kernel[(E, triton.cdiv(2 * I, BN))](
        xs, xs.stride(0), tok, seg_start,
        w13_t, w13_t.stride(1), s13_t, s13_t.stride(1),
        inter, inter.stride(0), w13_t.stride(0), s13_t.stride(0), lut_t,
        H=H, BM=BM, BN=BN, BK=BK, NS=2, LUT=False, num_warps=NW))
    print(f"  inter finite={bool(torch.isfinite(inter.float()).all())} "
          f"absmax={float(inter.float().abs().max()):.3e}", flush=True)
    step("_down_kernel", lambda: gp._down_kernel[(E, triton.cdiv(H, BH))](
        inter, inter.stride(0), tok, wts_s, seg_start,
        w2_t, w2_t.stride(1), s2_t, s2_t.stride(1),
        out, out.stride(0), w2_t.stride(0), s2_t.stride(0), lut_t,
        H=H, I=I, BM=BM, BH=BH, BK=BK, NS=2, LUT=False, num_warps=NW))
    print(f"  out finite={bool(torch.isfinite(out.float()).all())} "
          f"absmax={float(out.float().abs().max()):.3e}", flush=True)
    print("ALL STEPS OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

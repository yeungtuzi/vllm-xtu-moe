#!/usr/bin/env python
"""Isolate which dim breaks `_down_kernel`: H, I, or E?  One config per process.

A CUDA illegal-access poisons the context, so this takes H/I/E/M from the env and
does exactly one launch; the caller loops over the matrix in separate processes.

Run: H=5120 I=2304 E=384 M=8 python probe_down_kernel_dims.py
"""
from __future__ import annotations

import os
import sys

import torch

H = int(os.environ.get("H", "5120"))
I = int(os.environ.get("I", "2304"))
E = int(os.environ.get("E", "384"))
M = int(os.environ.get("M", "8"))
K = int(os.environ.get("K", "6"))
BM = int(os.environ.get("BM", "64"))
BH = int(os.environ.get("BH", "64"))
BK = int(os.environ.get("BK", "64"))
NS = int(os.environ.get("NS", "2"))

import triton  # noqa: E402
from vllm_xiaotu_moe.gpu_prefill import _down_kernel, _e2m1_table  # noqa: E402


def main() -> int:
    dev = torch.device("cuda:0")
    A = M * K
    # Synthetic but structurally identical to the real call.
    seg = torch.arange(0, E + 1, dtype=torch.int32, device=dev) * (A // E)
    seg[-1] = A
    tok = torch.arange(A, dtype=torch.int32, device=dev) % M
    wts = torch.ones(A, dtype=torch.float32, device=dev)
    inter = torch.zeros((A, 2 * I), dtype=torch.bfloat16, device=dev)
    w2_t = torch.zeros((E, I // 2, H), dtype=torch.uint8, device=dev)
    s2_t = torch.zeros((E, I // 32, H), dtype=torch.uint8, device=dev)
    out = torch.zeros((M, H), dtype=torch.bfloat16, device=dev)
    lut = _e2m1_table(dev)
    torch.cuda.synchronize()
    try:
        _down_kernel[(E, triton.cdiv(H, BH))](
            inter, inter.stride(0), tok, wts, seg,
            w2_t, w2_t.stride(1), s2_t, s2_t.stride(1),
            out, out.stride(0), w2_t.stride(0), s2_t.stride(0), lut,
            H=H, I=I, BM=BM, BH=BH, BK=BK, NS=NS, LUT=False, num_warps=4)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        print(f"H={H} I={I} E={E} M={M} BK={BK} BH={BH} -> FAULT "
              f"{type(exc).__name__}: {str(exc)[:80]}")
        return 1
    print(f"H={H} I={I} E={E} M={M} BK={BK} BH={BH} -> OK "
          f"finite={bool(torch.isfinite(out.float()).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

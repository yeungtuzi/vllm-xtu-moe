#!/usr/bin/env python
"""Decoder conformance gate: every e4m3 byte vs torch.float8_e4m3fn.

Compiles the engine's `kernels/fp8_dequant.hpp` decoders into a tiny standalone
program (one binary per ISA path) and checks all 256 byte values against
`torch.float8_e4m3fn`.

This gate exists because the *scalar/LUT* decoder was verified against torch
while the gather-free arithmetic decoders (`e4m3x16_to_fp32`, AVX-512, and
`e4m3x8_to_fp32_arith`, AVX2) were not -- and they decoded every subnormal as
`2^-9*(1+m/8)` instead of `m*2^-9`, i.e. byte 0x00 came out as 2^-9 instead of 0
(a DC bias on every zero weight) and 0x01..0x07 were mis-scaled by up to 3.7x.
Real block-128 checkpoints scale by `amax/448`, which keeps the error well under
1% of the layer output, so no accuracy gate had caught it.

Bound: every lane must match torch bit-for-bit, except the NaN encodings
(0x7F/0xFF), which both sides define as 0.

Usage: python scripts/test_fp8_decode_conformance.py [--gxx g++-16]
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HDR = os.path.join(REPO, "xiaotu_moe", "csrc", "kernels", "fp8_dequant.hpp")

SRC = r"""
#include <cstdio>
#include <cstdint>
#include "fp8_dequant.hpp"

int main() {
    alignas(64) uint8_t in[16];
    alignas(64) float out[16];
    for (int base = 0; base < 256; base += 16) {
        for (int i = 0; i < 16; ++i) in[i] = (uint8_t)(base + i);
#if defined(__AVX512F__)
        __m512 v = xiaotu_moe::fp8::e4m3x16_to_fp32(in);
        _mm512_storeu_ps(out, v);
#elif defined(__AVX2__)
        __m256 lo = xiaotu_moe::fp8::e4m3x8_to_fp32_arith(in);
        __m256 hi = xiaotu_moe::fp8::e4m3x8_to_fp32_arith(in + 8);
        _mm256_storeu_ps(out, lo);
        _mm256_storeu_ps(out + 8, hi);
#else
        for (int i = 0; i < 16; ++i)
            out[i] = xiaotu_moe::fp8::e4m3_to_fp32_scalar(in[i]);
#endif
        for (int i = 0; i < 16; ++i)
            printf("%d %a\n", base + i, (double)out[i]);
        /* scalar reference too, so both decoders are covered on every build */
        for (int i = 0; i < 16; ++i)
            printf("s%d %a\n", base + i,
                   (double)xiaotu_moe::fp8::e4m3_to_fp32_scalar(in[i]));
    }
    return 0;
}
"""

PATHS = [
    ("avx512", ["-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx512dq", "-mavx512bf16",
                "-mavx512vbmi", "-mfma"]),
    ("avx2", ["-mavx2", "-mfma"]),
]


def main() -> int:
    gxx = "g++-16"
    if "--gxx" in sys.argv:
        gxx = sys.argv[sys.argv.index("--gxx") + 1]

    import torch
    ref = {}
    tbl = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn)
    vals = tbl.float()
    for i in range(256):
        v = float(vals[i])
        ref[i] = 0.0 if v != v else v          # NaN encodings are defined as 0

    inc = os.path.dirname(HDR)                    # dir holding fp8_dequant.hpp
    bad_total = 0
    built = 0
    with tempfile.TemporaryDirectory() as td:
        cpp = os.path.join(td, "t.cpp")
        with open(cpp, "w") as f:
            f.write(SRC)
        for name, flags in PATHS:
            exe = os.path.join(td, name)
            cmd = [gxx, "-O2", "-std=c++17", *flags, "-I", inc, cpp, "-o", exe]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[BAD] build failed for {name}: {r.stderr.strip()[-500:]}")
                bad_total += 1
                continue
            built += 1
            r = subprocess.run([exe], capture_output=True, text=True, check=True)
            got = {}
            for line in r.stdout.split("\n"):
                if not line:
                    continue
                k, h = line.split()
                got[k] = float.fromhex(h)
            bad = []
            for key, v in got.items():
                idx = int(key.lstrip("s"))
                e = ref[idx]
                if v != e and not (v == 0.0 and e == 0.0):
                    bad.append((key, v, e))
            print(f"{name:>7}: {len(got)} lanes, {len(bad)} mismatches"
                  f"{'' if not bad else '  ' + str(bad[:6])}")
            bad_total += len(bad)
    if bad_total or built == 0:
        print(f"[BAD] {bad_total} decoder mismatches vs torch.float8_e4m3fn")
        return 1
    print("[OK ] all 256 e4m3 bytes decode bit-exactly (avx512 + avx2 + scalar)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

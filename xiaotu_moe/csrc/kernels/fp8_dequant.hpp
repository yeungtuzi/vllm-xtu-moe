// xiaotu-moe E4M3 (FP8) dequantization leaf. Self-contained, no external deps.
//
// Two implementations are provided:
//   * scalar:   reference, e4m3mfn -> fp32 with std::ldexp
//   * vector:   gather-free arithmetic conversion (AVX2 8-wide / AVX-512 16-wide)
// A 256-entry LUT variant is kept for A/B comparison; the hot kernel uses the
// arithmetic path because a microcoded gather is several times slower than a
// handful of integer ops (this was the dominant cost of the FP8 MoE kernel).
//
// Layout: exactly one e4m3 per byte. Scale is applied by the caller per
// groupN/groupK block.
//
// Copyright: Apache-2.0 (LUT idea from KVCache.AI ktransformers,
// /kt-kernel/operators/avx2/fp8_dequant.hpp).
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 大河马 (BigHippo) <dahema@me.com>

#ifndef XIAOTU_MOE_FP8_DEQUANT_HPP
#define XIAOTU_MOE_FP8_DEQUANT_HPP

#include <cstdint>
#include <cstring>
#include <cmath>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace xiaotu_moe {
namespace fp8 {

// ---- scalar: e4m3mfn byte -> fp32 -----------------------------------------
// e4m3mfn: 1 sign, 4 exponent (bias 7), 3 mantissa.
// 0x7F (NaN) and 0xFF -> 0, matching torch.float8_e4m3fn's NaN encodings.
inline float e4m3_to_fp32_scalar(uint8_t v) {
    if (v == 0x7F || v == 0xFF) return 0.0f;
    int s = (v >> 7) & 1;
    int e = (v >> 3) & 0xF;
    int m = v & 0x7;
    float val;
    if (e == 0) {
        // Subnormal: exponent field 0 has the same weight as e=1, i.e.
        // 2^(1-7) = 2^-6 (bias 7). The old 2^-7 made every subnormal byte
        // decode 2x too small (verified against torch.float8_e4m3fn).
        val = (m == 0) ? 0.0f : std::ldexp(m / 8.0f, -6);
    } else {
        val = std::ldexp(1.0f + m / 8.0f, e - 7);
    }
    return s ? -val : val;
}

#if defined(__AVX2__)
// ---- LUT variant (kept for A/B) -------------------------------------------
struct FP8LUT {
    float tab[256];
    FP8LUT() {
        for (int v = 0; v < 256; ++v) tab[v] = e4m3_to_fp32_scalar((uint8_t)v);
    }
};
inline const FP8LUT& fp8_lut() {
    static const FP8LUT lut;
    return lut;
}

// Dequantize 8 consecutive e4m3 bytes -> 8 fp32 lanes via LUT gather.
inline __m256 e4m3x8_to_fp32(const uint8_t* p) {
    __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    __m256i idx = _mm256_cvtepu8_epi32(b);
    const float* lut = fp8_lut().tab;
    return _mm256_i32gather_ps(lut, idx, 4);
}

// ---- gather-free arithmetic conversion ------------------------------------
// Bit-exact equivalent of e4m3_to_fp32_scalar:
//   normal  (e > 0): fp32 bits = sign<<31 | (e + 120)<<23 | m<<20
//   subnormal(e == 0): value = m * 2^-9, i.e. the e4m3 subnormal unit is
//     2^(1-7)/8 = 2^-9 (NOT 2^-6, and NOT the 2^-9*(1+m/8) that the naive
//     `118<<23 | m<<20` bit form produces). Looked up in `kE4M3Sub8` because the
//     correct bit pattern is not a linear function of m: m=1..7 maps to
//     2^-9,2^-8,3*2^-9,2^-7,5*2^-9,3*2^-8,7*2^-9. Getting this wrong makes 0x00
//     decode to 2^-9 instead of 0 (a DC bias on every zero weight) and mis-scales
//     0x01..0x07 by up to 3.7x. See scripts/test_fp8_decode_conformance.py.
//   NaN encodings 0x7F/0xFF -> 0
//
// Lane m (0..7) holds m * 2^-9 as fp32; lanes 8..15 are never selected because
// m = byte & 7.
static const float kE4M3Sub8[8] = {
    0.0f, 0x1p-9f, 2.0f * 0x1p-9f, 3.0f * 0x1p-9f,
    4.0f * 0x1p-9f, 5.0f * 0x1p-9f, 6.0f * 0x1p-9f, 7.0f * 0x1p-9f,
};

inline __m256 e4m3x8_to_fp32_arith(const uint8_t* p) {
    __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    __m256i v = _mm256_cvtepu8_epi32(b);
    __m256i sign = _mm256_slli_epi32(_mm256_and_si256(v, _mm256_set1_epi32(0x80)), 24);
    __m256i e = _mm256_and_si256(_mm256_srli_epi32(v, 3), _mm256_set1_epi32(0xF));
    __m256i m = _mm256_and_si256(v, _mm256_set1_epi32(0x7));
    __m256i bits = _mm256_or_si256(
        _mm256_slli_epi32(_mm256_add_epi32(e, _mm256_set1_epi32(120)), 23),
        _mm256_slli_epi32(m, 20));
    __m256i is_sub = _mm256_cmpeq_epi32(e, _mm256_setzero_si256());
    __m256 sub = _mm256_permutevar8x32_ps(_mm256_loadu_ps(kE4M3Sub8), m);
    bits = _mm256_blendv_epi8(bits, _mm256_castps_si256(sub), is_sub);
    bits = _mm256_or_si256(sign, bits);
    __m256i is_nan = _mm256_or_si256(_mm256_cmpeq_epi32(v, _mm256_set1_epi32(0x7F)),
                                     _mm256_cmpeq_epi32(v, _mm256_set1_epi32(0xFF)));
    bits = _mm256_andnot_si256(is_nan, bits);
    return _mm256_castsi256_ps(bits);
}
#endif  // __AVX2__

#if defined(__AVX512F__)
// Dequantize 16 consecutive e4m3 bytes -> 16 fp32 lanes (AVX-512).
// Subnormals use the exact `m * 2^-9` table (see kE4M3Sub8 above) instead of the
// old `118<<23 | m<<20`, which decoded 0x00 as 2^-9 (not 0) and mis-scaled
// 0x01..0x07 by up to 3.7x.
inline __m512 e4m3x16_to_fp32(const uint8_t* p) {
    __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    __m512i v = _mm512_cvtepu8_epi32(b);
    __m512i sign = _mm512_slli_epi32(_mm512_and_si512(v, _mm512_set1_epi32(0x80)), 24);
    __m512i e = _mm512_and_si512(_mm512_srli_epi32(v, 3), _mm512_set1_epi32(0xF));
    __m512i m = _mm512_and_si512(v, _mm512_set1_epi32(0x7));
    __mmask16 sub = _mm512_cmpeq_epi32_mask(e, _mm512_setzero_si512());
    __m512i bits = _mm512_or_si512(
        _mm512_slli_epi32(_mm512_add_epi32(e, _mm512_set1_epi32(120)), 23),
        _mm512_slli_epi32(m, 20));
    __m512 subv = _mm512_permutexvar_ps(m, _mm512_castps256_ps512(
        _mm256_loadu_ps(kE4M3Sub8)));
    bits = _mm512_mask_blend_epi32(sub, bits, _mm512_castps_si512(subv));
    bits = _mm512_or_si512(sign, bits);
    __mmask16 is_nan = _mm512_cmpeq_epi32_mask(v, _mm512_set1_epi32(0x7F)) |
                       _mm512_cmpeq_epi32_mask(v, _mm512_set1_epi32(0xFF));
    bits = _mm512_maskz_mov_epi32(~is_nan, bits);
    return _mm512_castsi512_ps(bits);
}
#endif  // __AVX512F__

}  // namespace fp8
}  // namespace xiaotu_moe

#endif  // XIAOTU_MOE_FP8_DEQUANT_HPP

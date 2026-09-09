// xiaotu-moe E4M3 (FP8) dequantization leaf. Self-contained, no external deps.
//
// Dequantizes a byte of FP8 e4m3mfn into fp32. Mirrors the approach in
// ktransformers' avx2/fp8_dequant.hpp (256-entry float LUT) but written fresh
// for this codebase. LUT avoids per-element float exponent math at decode time.
//
// Layout: exactly one e4m3 per byte. No grouping here; scale is applied by the
// caller (the gemm / moe traits), per the groupN/groupK block.
//
// Copyright: Apache-2.0 (dequantization idea from KVCache.AI ktransformers,
// /kt-kernel/operators/avx2/fp8_dequant.hpp).

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

// ---- scalar: e4m3mfn byte -> fp32. Follows ktransformers fp8_dequant.hpp
// (which itself follows the standard: NaN 0x7F handled as 0; values beyond
// fp16 range clamp). Kept deterministic and dependency-free.
inline float e4m3_to_fp32_scalar(uint8_t v) {
    // e4m3mfn: 1 sign, 4 exponent, 3 mantissa. Number = (-1)^s * 2^(e-7) * (1 + m/8)
    // special: 0x7F (NaN) and 0xFF -> treat as 0 like the reference kernel.
    if (v == 0x7F || v == 0xFF) return 0.0f;
    int s = (v >> 7) & 1;
    int e = (v >> 3) & 0xF;
    int m = v & 0x7;
    float val;
    if (e == 0) {
        val = (m == 0) ? 0.0f : std::ldexp(m / 8.0f, -7);       // subnormal
    } else {
        val = std::ldexp(1.0f + m / 8.0f, e - 7);               // normal
    }
    return s ? -val : val;
}

#if defined(__AVX2__)
// 256-entry LUT generated once (static init). Normal entries via ldexp like the
// reference; subnormal/zero/NaN handled in the helper above.
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

// Dequantize 8 consecutive e4m3 bytes -> 8 fp32 lanes (AVX2).
inline __m256 e4m3x8_to_fp32(const uint8_t* p) {
    __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));   // 8 bytes
    __m256i idx = _mm256_cvtepu8_epi32(b);                              // 8 zero-extended ints
    const float* lut = fp8_lut().tab;
    return _mm256_i32gather_ps(lut, idx, 4);
}
#endif  // __AVX2__

}  // namespace fp8
}  // namespace xiaotu_moe

#endif  // XIAOTU_MOE_FP8_DEQUANT_HPP

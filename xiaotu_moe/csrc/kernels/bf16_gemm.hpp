// xiaotu-moe BF16 GEMM kernel (self-contained, no external deps)
//
// Weight layout: [N][K] bf16 row-major, Activation: [M][K] bf16 row-major
// Output: [M][N] fp32
//
// Structure mirrors lktransformers' amx_gemm.cpp (AVX2 + AVX512 paths) but
// without the llama.cpp / ggml dependency tree, so the BF16 minimal closed
// loop builds standalone.
//
// Copyright: Apache-2.0 (structure inspired by KVCache.AI / Qiong GU open source)

#ifndef XIAOTU_MOE_BF16_GEMM_HPP
#define XIAOTU_MOE_BF16_GEMM_HPP

#include <cstdint>
#include <cstddef>
#include <cstring>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512BF16__)
    #define XIAOTU_MOE_HAVE_AVX512_BF16 1
#else
    #define XIAOTU_MOE_HAVE_AVX512_BF16 0
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
    // For the AVX2-with-512 support used as a fallback when BF16 is absent.
    #define XIAOTU_MOE_HAVE_AVX512 1
#else
    #define XIAOTU_MOE_HAVE_AVX512 0
#endif

#if defined(__AVX2__) && defined(__FMA__)
    #define XIAOTU_MOE_HAVE_AVX2 1
#else
    #define XIAOTU_MOE_HAVE_AVX2 0
#endif

#if XIAOTU_MOE_HAVE_AVX512_BF16 || XIAOTU_MOE_HAVE_AVX512 || XIAOTU_MOE_HAVE_AVX2
    #include <immintrin.h>
#endif

namespace xiaotu_moe {
namespace bf16 {

inline float bf16_to_fp32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f; std::memcpy(&f, &u, 4);
    return f;
}

// Round-to-nearest-even fp32 -> bf16 (truncation-based approximation).
inline uint16_t fp32_to_bf16(float v) {
    uint32_t u; std::memcpy(&u, &v, 4);
    // round to nearest even on bit 16
    uint32_t rounded = u + 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(rounded >> 16);
}

// Convert an fp32 buffer to bf16 in place / to a separate buffer.
inline void convert_f32_to_bf16(const float* src, uint16_t* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = fp32_to_bf16(src[i]);
}

// C[M,N] fp32 = A[M,K]bf16 x B[N,K]bf16^T  (both row-major)
inline void matmul_bf16(const uint16_t* A, const uint16_t* B, float* C,
                        int M, int N, int K) {
#if XIAOTU_MOE_HAVE_AVX512_BF16 || XIAOTU_MOE_HAVE_AVX512
    // 16-wide fp32 over K. B is [N][K] so a column is contiguous -> no transpose
    // needed; converts A/B segments to fp32 and FMA-accumulates. (dpbf16 needs a
    // transposed B layout and is deferred to a later pass with pre-transposed
    // weights.)
    constexpr int W = 16;
    for (int i = 0; i < M; ++i) {
        const uint16_t* Arow = A + (size_t)i * K;
        float* Crow = C + (size_t)i * N;
        for (int j = 0; j < N; ++j) {
            const uint16_t* Brow = B + (size_t)j * K;
            __m512 acc = _mm512_setzero_ps();
            int k = 0;
            for (; k + W <= K; k += W) {
                __m256i a16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Arow + k));
                __m512 av = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(a16), 16));
                __m256i b16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Brow + k));
                __m512 bv = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(b16), 16));
                acc = _mm512_fmadd_ps(av, bv, acc);
            }
            float s = 0.f;
            for (; k < K; ++k) s += bf16_to_fp32(Arow[k]) * bf16_to_fp32(Brow[k]);
            float tmp[W]; _mm512_storeu_ps(tmp, acc);
            float accf = 0.f; for (int t = 0; t < W; ++t) accf += tmp[t];
            Crow[j] = accf + s;
        }
    }
#elif XIAOTU_MOE_HAVE_AVX2
    // AVX2 fallback (hosts without avx512_bf16): scalar columns, 8-wide K loop.
    // B is [N][K] so a column is strided; vectorizing over K avoids a transpose.
    for (int i = 0; i < M; ++i) {
        const uint16_t* Arow = A + (size_t)i * K;
        float* Crow = C + (size_t)i * N;
        for (int j = 0; j < N; ++j) {
            const uint16_t* Brow = B + (size_t)j * K;
            __m256 acc = _mm256_setzero_ps();
            int k = 0;
            for (; k + 8 <= K; k += 8) {
                // 8 bf16 of A (contiguous, 16 bytes = full __m128i) -> 8 fp32
                __m128i a16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(Arow + k));
                __m256 av = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(a16), 16));
                // 8 bf16 of B (contiguous, 16 bytes = full __m128i) -> 8 fp32
                __m128i b16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(Brow + k));
                __m256 bv = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b16), 16));
                acc = _mm256_fmadd_ps(av, bv, acc);
            }
            float s = 0.f;
            for (; k < K; ++k) s += bf16_to_fp32(Arow[k]) * bf16_to_fp32(Brow[k]);
            float tmp[8]; _mm256_storeu_ps(tmp, acc);
            float accf = 0.f; for (int t = 0; t < 8; ++t) accf += tmp[t];
            Crow[j] = accf + s;
        }
    }
#else
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += bf16_to_fp32(A[i * K + k]) * bf16_to_fp32(B[j * K + k]);
            C[i * N + j] = acc;
        }
#endif
}

} // namespace bf16
} // namespace xiaotu_moe

#endif // XIAOTU_MOE_BF16_GEMM_HPP

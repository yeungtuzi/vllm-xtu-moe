// xiaotu-moe: FP8 (e4m3) WeightTraits for MOE_V2. Self-contained.
//
// Weight layout (mirrors what vLLM feeds the real lk_moe MOE_FP8 engine):
//   w13  : [expert_num][2*inter][hidden]   one e4m3 byte per element.
//          gate = rows [0, inter), up = rows [inter, 2*inter).
//   w2   : [expert_num][hidden][inter]     one e4m3 byte per element.
//   scales (w13_g / w2_g):
//          [expert_num][N/groupN][K/groupK] fp32 (per-group block scale).
//          For w13_g N=2*inter (gate+up rows map into the same N index),
//          K=hidden. For w2_g N=hidden, K=inter.
//   dequant: W[n,k] = e4m3_byte * scale[n/groupN, k/groupK]
//
// GEMM uses the deferred-hsum pattern from ktransformers' mxfp8-moe.hpp: inside
// a K-group only unscaled values accumulate; at each group boundary a single
// broadcast-FMA folds the group scale into the vector accumulator; one hsum per
// row. 8-wide AVX2 gather path (compiles under AVX512 too) + scalar fallback.
//
// Copyright: Apache-2.0. Dequantization approach from KVCache.AI ktransformers
// (/kt-kernel/operators/avx2/fp8_dequant.hpp, avx2/mxfp8-moe.hpp).

#ifndef XIAOTU_MOE_MOE_V2_FP8_HPP
#define XIAOTU_MOE_MOE_V2_FP8_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "../kernels/bf16_gemm.hpp"
#include "../kernels/fp8_dequant.hpp"

namespace xiaotu_moe {

namespace fp8_detail {

// C[M,N] fp32 = A[M,K]bf16 x W[N,K]fp8^T, group scale along N (groupN) and K
// (groupK). S = [N/groupN][K/groupK] fp32. groupN/groupK <= 0 treated as 1.
inline void matmul_fp8_quant(const uint16_t* A, const uint8_t* W, const float* S,
                             float* C, int M, int N, int K,
                             int groupN, int groupK) {
    const int gn = groupN > 0 ? groupN : 1;
    const int gk = groupK > 0 ? groupK : 1;
#if XIAOTU_MOE_HAVE_AVX2
    constexpr int Wv = 8;
#else
    constexpr int Wv = 8;
#endif
#if defined(__AVX2__)
    for (int i = 0; i < M; ++i) {
        const uint16_t* Arow = A + (size_t)i * K;
        float* Crow = C + (size_t)i * N;
        for (int j = 0; j < N; ++j) {
            const uint8_t* Wrow = W + (size_t)j * K;
            const float* Srow = S + (size_t)(j / gn) * ((K + gk - 1) / gk);
            __m256 total = _mm256_setzero_ps();
            int kbase = 0;
            for (; kbase < K; kbase += gk) {
                __m256 gacc = _mm256_setzero_ps();
                int k = kbase;
                int kend = (kbase + gk < K) ? (kbase + gk) : K;
                for (; k + Wv <= kend; k += Wv) {
                    // A: bf16 -> fp32 (load 8 bf16 = 16 bytes = full __m128i)
                    __m128i a16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(Arow + k));
                    __m256 av = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(a16), 16));
                    __m256 wv = fp8::e4m3x8_to_fp32(Wrow + k);
                    gacc = _mm256_fmadd_ps(av, wv, gacc);
                }
                float gscalar = 0.f;
                for (; k < kend; ++k)
                    gscalar += bf16::bf16_to_fp32(Arow[k]) * fp8::e4m3_to_fp32_scalar(Wrow[k]);
                float scale = Srow[kbase / gk];
                total = _mm256_fmadd_ps(gacc, _mm256_set1_ps(scale), total);
                if (gscalar != 0.f) {
                    float t[8]; _mm256_storeu_ps(t, total);
                    t[0] += gscalar * scale;
                    total = _mm256_loadu_ps(t);
                }
            }
            float tmp[Wv]; _mm256_storeu_ps(tmp, total);
            float acc = 0.f; for (int t = 0; t < Wv; ++t) acc += tmp[t];
            Crow[j] = acc;
        }
    }
#else
    for (int i = 0; i < M; ++i) {
        const uint16_t* Arow = A + (size_t)i * K;
        float* Crow = C + (size_t)i * N;
        for (int j = 0; j < N; ++j) {
            const uint8_t* Wrow = W + (size_t)j * K;
            const float* Srow = S + (size_t)(j / gn) * ((K + gk - 1) / gk);
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += bf16::bf16_to_fp32(Arow[k]) * fp8::e4m3_to_fp32_scalar(Wrow[k]) *
                       Srow[k / gk];
            Crow[j] = acc;
        }
    }
#endif
}

}  // namespace fp8_detail

// FP8 WeightTraits. Base class dispatches to *_impl via CRTP (see moe_v2.hpp).
struct FP8WeightTraits : WeightTraitsBase<FP8WeightTraits> {
    static constexpr size_t w13_bytes_impl(size_t E, size_t n2, size_t H) {
        return E * n2 * H * sizeof(uint8_t);  // [E][2I][H] fp8
    }
    static constexpr size_t w2_bytes_impl(size_t E, size_t H, size_t I) {
        return E * H * I * sizeof(uint8_t);   // [E][H][I] fp8
    }
    static void gate_up_impl(const uint16_t* x, const void* w13, const void* w13_g,
                            const float* w13_gs, float* gate, float* up,
                            int inter, int hidden, size_t eid,
                            int groupN, int groupK) {
        (void)w13_gs;
        // w13 : [E][2I][H]; scales [E][2I/gn][H/gk]. N = 2*inter (gate+up rows).
        const int n2 = 2 * inter;
        const uint8_t* base = static_cast<const uint8_t*>(w13) + eid * (size_t)n2 * hidden;
        std::vector<float> both(n2);
        if (w13_g) {
            const int gn = groupN > 0 ? groupN : 1;
            const int gk = groupK > 0 ? groupK : 1;
            // Block counts use ceil (a group larger than a dim collapses to one
            // block, but the per-expert stride must still advance by >=1).
            const size_t nb = (n2 + gn - 1) / gn;
            const size_t kb = (hidden + gk - 1) / gk;
            const float* sbase = static_cast<const float*>(w13_g) + eid * (nb * kb);
            fp8_detail::matmul_fp8_quant(x, base, sbase, both.data(), 1, n2, hidden, groupN, groupK);
        } else {
            // No scale: pass a degenerate all-ones scale table.
            const int gn = groupN > 0 ? groupN : 1;
            const int gk = groupK > 0 ? groupK : 1;
            const size_t nb = (n2 + gn - 1) / gn;
            const size_t kb = (hidden + gk - 1) / gk;
            std::vector<float> ones(nb * kb, 1.0f);
            fp8_detail::matmul_fp8_quant(x, base, ones.data(), both.data(), 1, n2, hidden, groupN, groupK);
        }
        std::copy(both.begin(), both.begin() + inter, gate);
        std::copy(both.begin() + inter, both.end(), up);
    }

    static void down_impl(const uint16_t* act, const void* w2, const void* w2_g,
                          const float* w2_gs, float* down, int hidden, int inter,
                          size_t eid, int groupN, int groupK) {
        (void)w2_gs;
        // w2 : [E][H][I]; scales [E][H/gn][I/gk].
        const uint8_t* base = static_cast<const uint8_t*>(w2) + eid * (size_t)hidden * inter;
        const int gn = groupN > 0 ? groupN : 1;
        const int gk = groupK > 0 ? groupK : 1;
        const size_t nb = (hidden + gn - 1) / gn;
        const size_t kb = (inter + gk - 1) / gk;
        if (w2_g) {
            const float* sbase = static_cast<const float*>(w2_g) + eid * (nb * kb);
            fp8_detail::matmul_fp8_quant(act, base, sbase, down, 1, hidden, inter, groupN, groupK);
        } else {
            std::vector<float> ones(nb * kb, 1.0f);
            fp8_detail::matmul_fp8_quant(act, base, ones.data(), down, 1, hidden, inter, groupN, groupK);
        }
    }
};

}  // namespace xiaotu_moe

#endif  // XIAOTU_MOE_MOE_V2_FP8_HPP

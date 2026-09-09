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
// Vector helpers: one instantiation per ISA level, so the kernel body below is
// written once. Arithmetic (gather-free) FP8 decode is used in the hot loop.
#if defined(__AVX512F__)
using Vec = __m512;
constexpr int kVW = 16;
inline Vec vzero() { return _mm512_setzero_ps(); }
inline Vec vbcast(float s) { return _mm512_set1_ps(s); }
inline Vec vfma(Vec a, Vec b, Vec c) { return _mm512_fmadd_ps(a, b, c); }
inline Vec load_bf16_as_f32(const uint16_t* p) {
    __m256i a16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(a16), 16));
}
inline Vec load_fp8_as_f32(const uint8_t* p) { return fp8::e4m3x16_to_fp32(p); }
inline float vhsum(Vec v) { return _mm512_reduce_add_ps(v); }
#elif defined(__AVX2__)
using Vec = __m256;
constexpr int kVW = 8;
inline Vec vzero() { return _mm256_setzero_ps(); }
inline Vec vbcast(float s) { return _mm256_set1_ps(s); }
inline Vec vfma(Vec a, Vec b, Vec c) { return _mm256_fmadd_ps(a, b, c); }
inline Vec load_bf16_as_f32(const uint16_t* p) {
    __m128i a16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(a16), 16));
}
inline Vec load_fp8_as_f32(const uint8_t* p) { return fp8::e4m3x8_to_fp32_arith(p); }
inline float vhsum(Vec v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
#endif

// Row-range variant: computes C[i][j] only for j in [n0, n1), indexing W/S by the
// GLOBAL row j so a caller can hand out disjoint row slices to different threads
// (N-sliced decode). C keeps its full [M][N] row stride.
inline void matmul_fp8_quant_range(const uint16_t* A, const uint8_t* W, const float* S,
                                   float* C, int M, int N, int K,
                                   int groupN, int groupK, int n0, int n1) {
    const int gn = groupN > 0 ? groupN : 1;
    const int gk = groupK > 0 ? groupK : 1;
    if (n0 < 0) n0 = 0;
    if (n1 > N) n1 = N;
    if (n1 <= n0) return;
#if defined(__AVX2__)
    for (int i = 0; i < M; ++i) {
        const uint16_t* Arow = A + (size_t)i * K;
        float* Crow = C + (size_t)i * N;
        for (int j = n0; j < n1; ++j) {
            const uint8_t* Wrow = W + (size_t)j * K;
            const float* Srow = S + (size_t)(j / gn) * ((K + gk - 1) / gk);
            Vec total = vzero();
            float tail = 0.f;   // 标量尾巴(不足一个向量的部分),最后一起加
            int kbase = 0;
            for (; kbase < K; kbase += gk) {
                Vec gacc = vzero();
                int k = kbase;
                int kend = (kbase + gk < K) ? (kbase + gk) : K;
                for (; k + kVW <= kend; k += kVW)
                    gacc = vfma(load_bf16_as_f32(Arow + k), load_fp8_as_f32(Wrow + k), gacc);
                float gscalar = 0.f;
                for (; k < kend; ++k)
                    gscalar += bf16::bf16_to_fp32(Arow[k]) * fp8::e4m3_to_fp32_scalar(Wrow[k]);
                float scale = Srow[kbase / gk];
                total = vfma(gacc, vbcast(scale), total);
                tail += gscalar * scale;
            }
            Crow[j] = vhsum(total) + tail;
        }
    }
#else
    for (int i = 0; i < M; ++i) {
        const uint16_t* Arow = A + (size_t)i * K;
        float* Crow = C + (size_t)i * N;
        for (int j = n0; j < n1; ++j) {
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

inline void matmul_fp8_quant(const uint16_t* A, const uint8_t* W, const float* S,
                             float* C, int M, int N, int K,
                             int groupN, int groupK) {
    matmul_fp8_quant_range(A, W, S, C, M, N, K, groupN, groupK, 0, N);
}

}  // namespace fp8_detail

// FP8 WeightTraits. Base class dispatches to *_impl via CRTP (see moe_v2.hpp).
struct FP8WeightTraits : WeightTraitsBase<FP8WeightTraits> {
    // Decode is compute-bound (a single token's GEMV re-decodes every weight row
    // on one thread), so small batches are fanned out over N slices as well.
    static constexpr bool kNSliceSmallM = true;
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

    // ---- N-sliced variants (small-batch decode) ----------------------------
    // Same kernels restricted to the N-row range [n0, n1): gate rows [n0, n1) and
    // up rows [inter+n0, inter+n1) of the [2*inter] block are written into
    // `both` (row stride 2*inter); down rows [n0, n1) of the [hidden] block into
    // `down`. Disjoint slices are race-free, which lets forward_many fan a single
    // token's GEMV over every worker thread.
    static void gate_up_slice_impl(const uint16_t* x, const void* w13, const void* w13_g,
                                   const float* w13_gs, float* both, int inter, int hidden,
                                   size_t eid, int groupN, int groupK, int n0, int n1) {
        (void)w13_gs;
        if (n0 < 0) n0 = 0;
        if (n1 > inter) n1 = inter;
        if (n1 <= n0) return;
        const int n2 = 2 * inter;
        const uint8_t* base = static_cast<const uint8_t*>(w13) + eid * (size_t)n2 * hidden;
        const float* sbase = fp8_scale_base(w13_g, eid, n2, hidden, groupN, groupK);
        fp8_detail::matmul_fp8_quant_range(x, base, sbase, both, 1, n2, hidden, groupN, groupK,
                                           n0, n1);
        fp8_detail::matmul_fp8_quant_range(x, base, sbase, both, 1, n2, hidden, groupN, groupK,
                                           inter + n0, inter + n1);
    }

    static void down_slice_impl(const uint16_t* act, const void* w2, const void* w2_g,
                                const float* w2_gs, float* down, int hidden, int inter,
                                size_t eid, int groupN, int groupK, int n0, int n1) {
        (void)w2_gs;
        if (n0 < 0) n0 = 0;
        if (n1 > hidden) n1 = hidden;
        if (n1 <= n0) return;
        const uint8_t* base = static_cast<const uint8_t*>(w2) + eid * (size_t)hidden * inter;
        const float* sbase = fp8_scale_base(w2_g, eid, hidden, inter, groupN, groupK);
        fp8_detail::matmul_fp8_quant_range(act, base, sbase, down, 1, hidden, inter, groupN, groupK,
                                           n0, n1);
    }

    // Per-expert scale table base; a degenerate all-ones table (thread-local, so
    // the sliced decode path allocates nothing per job) stands in when the model
    // ships no per-expert scales.
    static const float* fp8_scale_base(const void* s, size_t eid, int N, int K,
                                       int groupN, int groupK) {
        const int gn = groupN > 0 ? groupN : 1;
        const int gk = groupK > 0 ? groupK : 1;
        const size_t nb = ((size_t)N + gn - 1) / gn;
        const size_t kb = ((size_t)K + gk - 1) / gk;
        if (s) return static_cast<const float*>(s) + eid * (nb * kb);
        thread_local std::vector<float> ones;
        if (ones.size() < nb * kb) ones.assign(nb * kb, 1.0f);
        return ones.data();
    }
};

}  // namespace xiaotu_moe

#endif  // XIAOTU_MOE_MOE_V2_FP8_HPP

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

// ---------------------------------------------------------------------------
// FP8 (MR, NR) register blocking.
//
// The legacy path walks one row at a time, so for every (row, column) pair it
// re-decodes the same weight byte and re-loads the same activation chunk: a
// batch routed to one expert re-reads that expert's N*K bytes M times (M = rows
// this expert got). GLM-5.3-Flash's real shapes put M in 1..12, so weight
// traffic is up to 12x what it needs to be.
//
// This tile path decodes each weight vector once and feeds it to MR rows, and
// loads each activation vector once for NR columns:
//     instr/MAC ~ 1/16 + decode/(MR*16) + activation/(NR*16)
// Register budget: acc[MR][NR] + av[MR] + wv[NR] + ~4 <= 32 zmms, so (12,1),
// (8,1), (6,2), (5,3) and (4,4) all fit.
//
// Compact NUMA shard support: `rowshift` maps a global output row j to the
// shard-local row j-rowshift, so the same kernel serves the dense block and a
// node's compact [gate][up] slice (see MOE_V2::shard_fill_w13). `rowmap` lets
// the activation row come from a gathered buffer.
//
// The tiled path requires groupK (and K) to be a multiple of kVW; other
// geometries fall back to the legacy kernel (dense only).
// ---------------------------------------------------------------------------
#ifndef XIAOTU_MOE_FP8_MR
#define XIAOTU_MOE_FP8_MR 6
#endif
#ifndef XIAOTU_MOE_FP8_NR
#define XIAOTU_MOE_FP8_NR 2
#endif
// Adaptive plan: for the row counts where the primary plan would read the
// weights twice, switch to a taller/thinner plan that covers them in one pass.
// 0 disables it (always use the primary (MR,NR)).
#ifndef XIAOTU_MOE_FP8_ADAPT
#define XIAOTU_MOE_FP8_ADAPT 1
#endif
#ifndef XIAOTU_MOE_FP8_WIDE_MR
#define XIAOTU_MOE_FP8_WIDE_MR 12
#endif
// Upper row count for the wide plan; at 12+ the primary plan measured faster
// again (register pressure in the 12x1 tile outweighs the extra weight read).
#ifndef XIAOTU_MOE_FP8_WIDE_HI
#define XIAOTU_MOE_FP8_WIDE_HI 11
#endif
// Third plan for the very small row counts, where a wider column block pays off
// more than a taller row block: (4,4) measured 4-11% faster than (6,2) at M<=4.
#ifndef XIAOTU_MOE_FP8_SMALL_MR
#define XIAOTU_MOE_FP8_SMALL_MR 4
#endif
#ifndef XIAOTU_MOE_FP8_SMALL_NR
#define XIAOTU_MOE_FP8_SMALL_NR 4
#endif
#ifndef XIAOTU_MOE_FP8_SMALL_HI
#define XIAOTU_MOE_FP8_SMALL_HI 4
#endif

#if defined(__AVX512F__)
template <int MRT, int NRT>
inline void fp8_tile_avx512(const uint16_t* A, const uint8_t* W, const float* S,
                            float* C, int N, int K, int gn, int gk,
                            int group_count, int kb_stride, int j0, int rowshift,
                            const uint32_t* rowmap, int m0) {
    __m512 acc[MRT][NRT];
    for (int r = 0; r < MRT; ++r)
        for (int jj = 0; jj < NRT; ++jj) acc[r][jj] = _mm512_setzero_ps();

    // Column bases are loop-invariant; hoist the row pointers out of the k loop.
    const uint8_t* wrow[NRT];
    for (int jj = 0; jj < NRT; ++jj)
        wrow[jj] = W + (size_t)(j0 + jj - rowshift) * (size_t)K;

    for (int g = 0; g < group_count; ++g) {
        const int kbase = g * gk;
        const int kend = (kbase + gk < K) ? (kbase + gk) : K;
        __m512 sv[NRT];
        for (int jj = 0; jj < NRT; ++jj)
            sv[jj] = _mm512_set1_ps(S[(size_t)((j0 + jj) / gn) * kb_stride + g]);
        for (int k = kbase; k < kend; k += kVW) {
            __m512 wv[NRT];
            for (int jj = 0; jj < NRT; ++jj)
                wv[jj] = _mm512_mul_ps(fp8::e4m3x16_to_fp32(wrow[jj] + k), sv[jj]);
            for (int r = 0; r < MRT; ++r) {
                const uint16_t* ap =
                    A + (size_t)(rowmap ? rowmap[m0 + r]
                                        : (uint32_t)(m0 + r)) * (size_t)K + k;
                const __m512 av = load_bf16_as_f32(ap);
                for (int jj = 0; jj < NRT; ++jj)
                    acc[r][jj] = _mm512_fmadd_ps(wv[jj], av, acc[r][jj]);
            }
        }
    }
    for (int r = 0; r < MRT; ++r)
        for (int jj = 0; jj < NRT; ++jj)
            C[(size_t)(m0 + r) * N + j0 + jj] = vhsum(acc[r][jj]);
}

// ---------------------------------------------------------------------------
// Opt-in BF16-MMA tile (AVX512-BF16 `vdpbf16ps`), enabled by
// XIAOTU_MOE_FP8_BF16_MMA=1.
//
// WHY: the fp32 tile above spends most of its uops on the *activation* side --
// `load_bf16_as_f32` is 2 uops per 16 values and runs MR times per k-chunk -- and
// on a 1-vector (16-element) FMA. Measured uop mix per 16 weights at (6,2):
// fp8 decode 19+1, activation convert/load 6*2*2 = 24, FMA 12 => 56, of which
// only ~30% is real arithmetic (see dev-docs/GLM53_SM80_PLAN.md §14).
//
// `vdpbf16ps` consumes **32 bf16** per operand and does 32 MACs into 16 fp32
// lanes, so it halves the MAC instruction count and needs NO activation
// conversion at all (the activations are already bf16 in memory). Per 32 weights
// at (6,2): decode+cvt+scale 2*(19+1+1) = 42, activation loads 6*2 = 12,
// dpbf16ps 12 => 66 uops, vs 112 for the fp32 tile over the same 32 weights.
//
// COST, and why this is opt-in: both operands must be bf16, so the weights are
// rounded to bf16 (`_mm512_cvtneps_pbh`) where the fp32 tile keeps
// `e4m3 * scale` in fp32. That is a ~0.2-0.4% perturbation of the expert weights
// -- documented in §18.5, deliberately NOT the default because the FP8 CPU path
// is otherwise exact.
//
// Numerics otherwise stay close: the e4m3 decode is the same (already-correct)
// `e4m3x16_to_fp32`, the products of two bf16 values are exact in fp32, and the
// accumulator is fp32; only the summation order changes (pairwise within 32).
// The scale is applied per k-group as before (groupK = 128 covers whole 32-wide
// steps), so no group accumulator is needed.
//
// Requires K % 32 == 0 and groupK % 32 == 0 (the k loop steps 32); the dispatch
// falls back to the fp32 tile otherwise.
// ---------------------------------------------------------------------------
#if defined(__AVX512F__) && defined(__AVX512BF16__)
inline bool fp8_bf16_mma_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("XIAOTU_MOE_FP8_BF16_MMA");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

template <int MRT, int NRT>
inline void fp8_tile_bf16_avx512(const uint16_t* A, const uint8_t* W, const float* S,
                                 float* C, int N, int K, int gn, int gk,
                                 int group_count, int kb_stride, int j0, int rowshift,
                                 const uint32_t* rowmap, int m0) {
    __m512 acc[MRT][NRT];
    for (int r = 0; r < MRT; ++r)
        for (int jj = 0; jj < NRT; ++jj) acc[r][jj] = _mm512_setzero_ps();

    const uint8_t* wrow[NRT];
    for (int jj = 0; jj < NRT; ++jj)
        wrow[jj] = W + (size_t)(j0 + jj - rowshift) * (size_t)K;

    for (int g = 0; g < group_count; ++g) {
        const int kbase = g * gk;
        const int kend = (kbase + gk < K) ? (kbase + gk) : K;
        __m512 sv[NRT];
        for (int jj = 0; jj < NRT; ++jj)
            sv[jj] = _mm512_set1_ps(S[(size_t)((j0 + jj) / gn) * kb_stride + g]);
        int k = kbase;
        for (; k + 32 <= kend; k += 32) {
            // Activations: already bf16 in memory -- a plain 64-byte load, no
            // conversion. Loaded once and reused by every column.
            __m512bh av[MRT];
            for (int r = 0; r < MRT; ++r) {
                const uint16_t* ap =
                    A + (size_t)(rowmap ? rowmap[m0 + r] : (uint32_t)(m0 + r)) *
                            (size_t)K + k;
                av[r] = (__m512bh)_mm512_loadu_si512(reinterpret_cast<const void*>(ap));
            }
            for (int jj = 0; jj < NRT; ++jj) {
                // e4m3 -> fp32 (exact) -> * scale -> bf16, two 16-wide vectors
                // concatenated into the 32-bf16 operand `vdpbf16ps` wants.
                __m512 w0 = _mm512_mul_ps(fp8::e4m3x16_to_fp32(wrow[jj] + k), sv[jj]);
                __m512 w1 = _mm512_mul_ps(fp8::e4m3x16_to_fp32(wrow[jj] + k + 16), sv[jj]);
                __m512i wb = _mm512_castsi256_si512(
                    reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(w0)));
                wb = _mm512_inserti64x4(
                    wb, reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(w1)), 1);
                const __m512bh wbh = (__m512bh)wb;
                for (int r = 0; r < MRT; ++r)
                    acc[r][jj] = _mm512_dpbf16_ps(acc[r][jj], av[r], wbh);
            }
        }
        // Tail inside the group: unreachable under the dispatch guard (which
        // requires K % 32 == 0 and groupK % 32 == 0), but computed correctly
        // rather than skipped -- a silently short accumulator is far worse than
        // a slow one.
        for (; k < kend; ++k) {
            for (int jj = 0; jj < NRT; ++jj) {
                const float w = fp8::e4m3_to_fp32_scalar(wrow[jj][k]) *
                                S[(size_t)((j0 + jj) / gn) * kb_stride + g];
                const __m512 wv = _mm512_set1_ps(w);
                for (int r = 0; r < MRT; ++r) {
                    const uint16_t* ap =
                        A + (size_t)(rowmap ? rowmap[m0 + r]
                                            : (uint32_t)(m0 + r)) * (size_t)K + k;
                    acc[r][jj] = _mm512_fmadd_ps(
                        wv, _mm512_set1_ps(bf16::bf16_to_fp32(*ap)), acc[r][jj]);
                }
            }
        }
    }
    for (int r = 0; r < MRT; ++r)
        for (int jj = 0; jj < NRT; ++jj)
            C[(size_t)(m0 + r) * N + j0 + jj] = vhsum(acc[r][jj]);
}

// The tile geometry is compile time on BOTH axes, and it must equal the actual
// block size -- not merely bound it. A tile taller/wider than the block reads
// `rowmap[]` past this expert's instance list (a wild row index => segfault) and
// writes C out of range. This mirrors the fp32 path's fp8_emit_mr/fp8_emit_nj.
template <int MR, int NR, int MRT, int NRT>
inline void fp8_bf16_emit(const uint16_t* A, const uint8_t* W, const float* S,
                          float* C, int N, int K, int gn, int gk,
                          int group_count, int kb_stride, int j0, int rowshift,
                          const uint32_t* rowmap, int m0, int mr, int nj) {
    if (mr == MRT) {
        if (nj == NRT) {
            fp8_tile_bf16_avx512<MRT, NRT>(A, W, S, C, N, K, gn, gk, group_count,
                                           kb_stride, j0, rowshift, rowmap, m0);
            return;
        }
        if constexpr (NRT > 1)
            fp8_bf16_emit<MR, NR, MRT, NRT - 1>(A, W, S, C, N, K, gn, gk,
                                                group_count, kb_stride, j0, rowshift,
                                                rowmap, m0, mr, nj);
        return;
    }
    if constexpr (MRT > 1)
        fp8_bf16_emit<MR, NR, MRT - 1, NRT>(A, W, S, C, N, K, gn, gk, group_count,
                                            kb_stride, j0, rowshift, rowmap, m0,
                                            mr, nj);
}

template <int MRT, int NRT>
inline void fp8_bf16_sweep(const uint16_t* A, const uint8_t* W, const float* S,
                           float* C, int M, int N, int K, int gn, int gk,
                           int n0, int n1, int rowshift, const uint32_t* rowmap) {
    const int group_count = (K + gk - 1) / gk;
    const int kb_stride = group_count;
    for (int m0 = 0; m0 < M; m0 += MRT) {
        const int mr = (M - m0 < MRT) ? (M - m0) : MRT;
        for (int j0 = n0; j0 < n1; j0 += NRT) {
            const int rem = n1 - j0;
            const int nj = (rem < NRT) ? rem : NRT;
            fp8_bf16_emit<MRT, NRT, MRT, NRT>(A, W, S, C, N, K, gn, gk, group_count,
                                              kb_stride, j0, rowshift, rowmap, m0,
                                              mr, nj);
        }
    }
}
#endif  // __AVX512F__ && __AVX512BF16__

// nj is always in [1, NR]; pick the matching compile-time column width.
template <int MRT, int NR>
inline void fp8_emit_nj(const uint16_t* A, const uint8_t* W, const float* S,
                        float* C, int N, int K, int gn, int gk,
                        int group_count, int kb_stride, int j0, int rowshift,
                        const uint32_t* rowmap, int m0, int nj) {
    if (nj == 1) {
        fp8_tile_avx512<MRT, 1>(A, W, S, C, N, K, gn, gk, group_count, kb_stride,
                                j0, rowshift, rowmap, m0);
        return;
    }
    if constexpr (NR >= 2) {
        if (nj == 2) {
            fp8_tile_avx512<MRT, 2>(A, W, S, C, N, K, gn, gk, group_count,
                                    kb_stride, j0, rowshift, rowmap, m0);
            return;
        }
    }
    if constexpr (NR >= 3) {
        if (nj == 3) {
            fp8_tile_avx512<MRT, 3>(A, W, S, C, N, K, gn, gk, group_count,
                                    kb_stride, j0, rowshift, rowmap, m0);
            return;
        }
    }
    if constexpr (NR >= 4) {
        fp8_tile_avx512<MRT, 4>(A, W, S, C, N, K, gn, gk, group_count, kb_stride,
                                j0, rowshift, rowmap, m0);
    }
}

// mr is always in [1, MR]; walk down to the matching compile-time row height.
template <int MR, int NR, int MRT>
inline void fp8_emit_mr(const uint16_t* A, const uint8_t* W, const float* S,
                        float* C, int N, int K, int gn, int gk,
                        int group_count, int kb_stride, int j0, int rowshift,
                        const uint32_t* rowmap, int m0, int mr, int nj) {
    if (mr == MRT) {
        fp8_emit_nj<MRT, NR>(A, W, S, C, N, K, gn, gk, group_count, kb_stride,
                             j0, rowshift, rowmap, m0, nj);
        return;
    }
    if constexpr (MRT > 1)
        fp8_emit_mr<MR, NR, MRT - 1>(A, W, S, C, N, K, gn, gk, group_count,
                                     kb_stride, j0, rowshift, rowmap, m0, mr, nj);
}

// Full tile sweep for one (MR, NR) plan.
template <int MR, int NR>
inline void fp8_tile_sweep(const uint16_t* A, const uint8_t* W, const float* S,
                           float* C, int M, int N, int K, int gn, int gk,
                           int n0, int n1, int rowshift, const uint32_t* rowmap) {
    const int group_count = (K + gk - 1) / gk;
    const int kb_stride = group_count;
    for (int m0 = 0; m0 < M; m0 += MR) {
        const int mr = (M - m0 < MR) ? (M - m0) : MR;
        for (int j0 = n0; j0 < n1; j0 += NR) {
            const int rem = n1 - j0;
            const int nj = (rem < NR) ? rem : NR;
            fp8_emit_mr<MR, NR, MR>(A, W, S, C, N, K, gn, gk, group_count,
                                    kb_stride, j0, rowshift, rowmap, m0, mr, nj);
        }
    }
}
#endif

// C[M,N] fp32 = A[M,K]bf16 x W[N,K]fp8^T for columns [n0, n1), MR x NR tiled.
// W/S are indexed by GLOBAL row for the scale table (`(j0+jj)/gn`), and by
// SHARD-LOCAL row for the weights (`j0+jj-rowshift`).
inline void matmul_fp8_tiled_range(const uint16_t* A, const uint8_t* W, const float* S,
                                   float* C, int M, int N, int K,
                                   int groupN, int groupK, int n0, int n1,
                                   int rowshift, const uint32_t* rowmap) {
#if defined(__AVX512F__)
    if (M <= 0 || K <= 0 || n1 <= n0) return;
    const int gk = groupK > 0 ? groupK : 1;
    if ((gk % kVW) != 0 || (K % kVW) != 0) {
        // Geometry the tile path does not cover; the legacy kernel is dense-only,
        // so this is reachable only with rowshift == 0 (see the trait guards).
        matmul_fp8_quant_range(A, W, S, C, M, N, K, groupN, groupK, n0, n1);
        return;
    }
    const int gn = groupN > 0 ? groupN : 1;
    constexpr int MR = XIAOTU_MOE_FP8_MR;
    constexpr int NR = XIAOTU_MOE_FP8_NR;
#if defined(__AVX512BF16__)
    // Opt-in BF16-MMA tile (see the block above `fp8_tile_bf16_avx512`). Needs
    // 32-element k steps on both the K and the scale-group grid, and it is gated
    // to the large-M regime: measured at the GLM shape it is 1.17-1.20x faster
    // for M >= 6 but 0.87x SLOWER at M = 1 (the weight decode dominates there and
    // the tile adds a `vcvtneps2bf16` + a 256->512 insert per 32 weights), and M
    // = 1 is exactly the latency-critical single-stream decode case (IRON_RULE
    // R8). So the exact fp32 path keeps small-M and the faster bf16 path serves
    // prefill / batched decode.
    if (fp8_bf16_mma_enabled() && M > XIAOTU_MOE_FP8_SMALL_HI &&
        (K % 32) == 0 && (gk % 32) == 0) {
        fp8_bf16_sweep<MR, NR>(A, W, S, C, M, N, K, gn, gk, n0, n1, rowshift, rowmap);
        return;
    }
#endif
#if XIAOTU_MOE_FP8_ADAPT
    // Measured M-curve (dev-docs/GLM53_SM80_PLAN.md §7.4-7.5, GLM shape, 60
    // threads). Three plans, chosen per call (per expert):
    //   M <= 4          : (4,4)  -- widest column block wins, -4..-11%
    //   7 <= M <= 11    : (12,1) -- tall/thin, avoids the ceil(M/6)=2 double read
    //   otherwise       : (6,2)  -- best for 5..6 and for M >= 12 (at 12+ the
    //                               12x1 tile's register pressure costs more
    //                               than the second weight read)
    if (M <= XIAOTU_MOE_FP8_SMALL_HI) {
        fp8_tile_sweep<XIAOTU_MOE_FP8_SMALL_MR, XIAOTU_MOE_FP8_SMALL_NR>(
            A, W, S, C, M, N, K, gn, gk, n0, n1, rowshift, rowmap);
        return;
    }
    if (M > MR && M <= XIAOTU_MOE_FP8_WIDE_HI) {
        fp8_tile_sweep<XIAOTU_MOE_FP8_WIDE_MR, 1>(A, W, S, C, M, N, K, gn, gk,
                                                  n0, n1, rowshift, rowmap);
        return;
    }
#endif
    fp8_tile_sweep<MR, NR>(A, W, S, C, M, N, K, gn, gk, n0, n1, rowshift, rowmap);
#else
    // Non-AVX512 ISA: keep the portable legacy kernel.
    matmul_fp8_quant_range(A, W, S, C, M, N, K, groupN, groupK, n0, n1);
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
    // Single-copy NUMA node sharding: one node-local copy of the expert weights
    // instead of two per-socket replicas. The tiled kernel and the batched slice
    // impls both honor the compact [gate][up] geometry it produces.
    static constexpr bool kNodeShard = true;
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

    // ---- Batched (M>1) slice variants --------------------------------------
    // `me` rows of ONE expert in a single GEMM call, so each weight vector is
    // decoded once and shared across MR rows (matmul_fp8_tiled_range). Honors
    // the compact NUMA-shard geometry that forward_many_nsliced passes
    // (cstride/row0/up_off), where W points at this node's [gate][up] slice.
    static void gate_up_slice_batch_impl(int me, const uint16_t* xg, const void* w13,
                                         const void* w13_g, const float* w13_gs,
                                         float* both_buf, int inter, int hidden,
                                         size_t eid, int groupN, int groupK,
                                         int n0, int n1,
                                         const uint32_t* rowmap = nullptr,
                                         size_t cstride = 0, long row0 = 0,
                                         size_t up_off = 0) {
        (void)w13_gs;
        if (me <= 0) return;
        if (n1 < 0 || n1 > inter) n1 = inter;
        if (n1 <= n0) return;
        const int n2 = 2 * inter;
        const size_t rowb = (size_t)hidden;                        // fp8: 1 byte/elem
        const size_t S = cstride ? cstride : (size_t)n2 * rowb;    // per-expert stride
        const size_t r0 = (size_t)(row0 < 0 ? 0 : row0);
        const size_t uoff = up_off ? up_off : (size_t)inter * rowb;
        const uint8_t* gbase = static_cast<const uint8_t*>(w13) + eid * S;
        const float* sbase = fp8_scale_base(w13_g, eid, n2, hidden, groupN, groupK);
        // gate rows [0, inter): shard-local row = global row - r0
        fp8_detail::matmul_fp8_tiled_range(xg, gbase, sbase, both_buf, me, n2, hidden,
                                           groupN, groupK, n0, n1, (int)r0, rowmap);
        // up rows [inter, 2*inter): stored right after the gate slice in a shard
        fp8_detail::matmul_fp8_tiled_range(xg, gbase + uoff, sbase, both_buf, me, n2, hidden,
                                           groupN, groupK, inter + n0, inter + n1,
                                           (int)(r0 + (size_t)inter), rowmap);
    }

    static void down_slice_batch_impl(int me, const uint16_t* actg, const void* w2,
                                      const void* w2_g, const float* w2_gs,
                                      float* down_buf, int hidden, int inter,
                                      size_t eid, int groupN, int groupK,
                                      int n0, int n1,
                                      size_t cstride = 0, long row0 = 0) {
        (void)w2_gs;
        if (me <= 0) return;
        if (n1 < 0 || n1 > hidden) n1 = hidden;
        if (n1 <= n0) return;
        const size_t rowb = (size_t)inter;                         // fp8: 1 byte/elem
        const size_t S = cstride ? cstride : (size_t)hidden * rowb;
        const size_t r0 = (size_t)(row0 < 0 ? 0 : row0);
        const uint8_t* base = static_cast<const uint8_t*>(w2) + eid * S;
        const float* sbase = fp8_scale_base(w2_g, eid, hidden, inter, groupN, groupK);
        fp8_detail::matmul_fp8_tiled_range(actg, base, sbase, down_buf, me, hidden, inter,
                                           groupN, groupK, n0, n1, (int)r0, nullptr);
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

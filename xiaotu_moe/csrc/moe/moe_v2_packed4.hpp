// xiaotu-moe: packed 4-bit WeightTraits (MXFP4 / WNA16) for MOE_V2.
// Self-contained CPU kernels.
//
// Weight layout (mirrors what vLLM feeds the real lk_moe MOE_MXFP4 / MOE_WNA16
// engines after _process_mxfp4/_process_wna16):
//   w13  : [expert_num][2*inter][hidden/2]  uint8, 4-bit nibble-packed along K.
//          gate = rows [0, inter), up = rows [inter, 2*inter).
//   w2   : [expert_num][hidden][inter/2]    uint8, nibble-packed along K.
//   Each byte = 2 consecutive K elements: low nibble = even k, high nibble = odd k.
//   Scales (w13_g / w2_g):
//          [expert_num][N/groupN][K/groupK] fp32 per-group block scale.
//          N = 2*inter for w13_g, hidden for w2_g; K = hidden/inter respectively.
//   Dequant: W[n,k] = lut[nibble(W,n,k)] * scale[n/groupN, k/groupK]
//     - MXFP4 : lut = FP4 E2M1 table (no zero point).
//     - WNA16 : lut = int4 centered at 8  ((nibble - 8)), ktransformers GPTQ
//               convention. Exact vLLM-Marlin matching needs a real-model A/B.
//
// GEMM uses the deferred-hsum pattern: inside a K-group only unscaled values
// accumulate, at each group boundary one broadcast-FMA folds the group scale,
// one hsum per row, then an optional global scale is applied.
//
// Copyright: Apache-2.0. Dequantization approach and FP4 E2M1 table follow
// KVCache.AI ktransformers (/kt-kernel/operators/avx2/mxfp4-moe.hpp).

#ifndef XIAOTU_MOE_MOE_V2_PACKED4_HPP
#define XIAOTU_MOE_MOE_V2_PACKED4_HPP

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../kernels/bf16_gemm.hpp"

namespace xiaotu_moe {

namespace packed4 {

// Standard FP4 E2M1 dequant table (nibble -> value). Shared by MXFP4 & NVFP4.
// e2m1: 1 sign, 2 exponent (bias 1), 1 mantissa.
static constexpr float E2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f /* -0 */, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// WNA16 int4 table, centered at 8 (ktransformers GPTQ convention: (nibble-8)).
static constexpr float INT4_CENTER8[16] = {
    -8.0f, -7.0f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f,
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
};

// nibble value of packed element (W packed [N][K/2], low nibble = even k).
static inline int nibble_of(const uint8_t* Wrow, int k) {
    const uint8_t b = Wrow[k / 2];
    return (k & 1) ? (b >> 4) : (b & 0x0F);
}

// horizontal sum of a 256-bit vector into a scalar.
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

#if defined(__AVX512F__)
// horizontal sum of a 512-bit vector into a scalar (fnm/blend-free).
static inline float hsum512(__m512 v) {
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 s = _mm256_add_ps(lo, hi);
    return hsum256(s);
}
#endif

// ---------------------------------------------------------------------------
// Fast-path memory profiler (DISABLED unless XIAOTU_MOE_BYTEPROF is set).
// Counts raw weight bytes read per fast-path call + wall time to measure the
// engine's real streaming bandwidth (MB/s) for the packed4 gate/up/down GEMMs.
// ---------------------------------------------------------------------------
static std::atomic<uint64_t> g_bp_bytes{0}, g_bp_ns{0}, g_bp_calls{0};
static std::atomic<uint64_t> g_bp_Lbytes{0}, g_bp_Lns{0}, g_bp_Lcalls{0}; // large (>=512KB)
static inline bool byteprof_on() {
    static const bool on = (std::getenv("XIAOTU_MOE_BYTEPROF") != nullptr);
    return on;
}
static inline void byteprof_accum(size_t bytes, uint64_t ns) {
    g_bp_bytes += bytes; g_bp_ns += ns; g_bp_calls += 1;
    if (bytes >= (size_t)(512 << 10)) { g_bp_Lbytes += bytes; g_bp_Lns += ns; g_bp_Lcalls += 1; }
    uint64_t c = g_bp_calls.load(std::memory_order_relaxed);
    if ((c % 200) == 0) {
        double agg = (double)g_bp_bytes.load() / 1e6 / std::max((double)g_bp_ns.load()/1e9, 1e-12);
        double Lag = (double)g_bp_Lbytes.load() / 1e6 / std::max((double)g_bp_Lns.load()/1e9, 1e-12);
        fprintf(stderr, "[BYTEPROF] calls=%llu agg=%.0f MB/s  LARGE(>=512KB) calls=%llu bytes=%.1fMB ns=%.0f agg=%.0f MB/s\n",
                (unsigned long long)c, agg,
                (unsigned long long)g_bp_Lcalls.load(), (double)g_bp_Lbytes.load()/1e6, (double)g_bp_Lns.load(), Lag);
    }
}
// fp8_e8m0fnu ("ue8m0") decode: scale = 2^(byte - 127). This is how the fork's
// Mxfp4MoEMethod feeds DeepSeek-V4-Flash MXFP4 scales (raw uint8 bytes, verified
// against real lk_moe MOE_MXFP4: median ratio 1.0001 vs torch truth). Lazy
// 256-entry table so indexing costs one load, no per-group powf.
inline float e8m0_scale_table_gen(int i) { return std::pow(2.0f, i - 127); }
static inline const float* e8m0_table() {
    static const float table[256] = {
        e8m0_scale_table_gen(0), e8m0_scale_table_gen(1), e8m0_scale_table_gen(2),
        e8m0_scale_table_gen(3), e8m0_scale_table_gen(4), e8m0_scale_table_gen(5),
        e8m0_scale_table_gen(6), e8m0_scale_table_gen(7), e8m0_scale_table_gen(8),
        e8m0_scale_table_gen(9), e8m0_scale_table_gen(10), e8m0_scale_table_gen(11),
        e8m0_scale_table_gen(12), e8m0_scale_table_gen(13), e8m0_scale_table_gen(14),
        e8m0_scale_table_gen(15), e8m0_scale_table_gen(16), e8m0_scale_table_gen(17),
        e8m0_scale_table_gen(18), e8m0_scale_table_gen(19), e8m0_scale_table_gen(20),
        e8m0_scale_table_gen(21), e8m0_scale_table_gen(22), e8m0_scale_table_gen(23),
        e8m0_scale_table_gen(24), e8m0_scale_table_gen(25), e8m0_scale_table_gen(26),
        e8m0_scale_table_gen(27), e8m0_scale_table_gen(28), e8m0_scale_table_gen(29),
        e8m0_scale_table_gen(30), e8m0_scale_table_gen(31), e8m0_scale_table_gen(32),
        e8m0_scale_table_gen(33), e8m0_scale_table_gen(34), e8m0_scale_table_gen(35),
        e8m0_scale_table_gen(36), e8m0_scale_table_gen(37), e8m0_scale_table_gen(38),
        e8m0_scale_table_gen(39), e8m0_scale_table_gen(40), e8m0_scale_table_gen(41),
        e8m0_scale_table_gen(42), e8m0_scale_table_gen(43), e8m0_scale_table_gen(44),
        e8m0_scale_table_gen(45), e8m0_scale_table_gen(46), e8m0_scale_table_gen(47),
        e8m0_scale_table_gen(48), e8m0_scale_table_gen(49), e8m0_scale_table_gen(50),
        e8m0_scale_table_gen(51), e8m0_scale_table_gen(52), e8m0_scale_table_gen(53),
        e8m0_scale_table_gen(54), e8m0_scale_table_gen(55), e8m0_scale_table_gen(56),
        e8m0_scale_table_gen(57), e8m0_scale_table_gen(58), e8m0_scale_table_gen(59),
        e8m0_scale_table_gen(60), e8m0_scale_table_gen(61), e8m0_scale_table_gen(62),
        e8m0_scale_table_gen(63), e8m0_scale_table_gen(64), e8m0_scale_table_gen(65),
        e8m0_scale_table_gen(66), e8m0_scale_table_gen(67), e8m0_scale_table_gen(68),
        e8m0_scale_table_gen(69), e8m0_scale_table_gen(70), e8m0_scale_table_gen(71),
        e8m0_scale_table_gen(72), e8m0_scale_table_gen(73), e8m0_scale_table_gen(74),
        e8m0_scale_table_gen(75), e8m0_scale_table_gen(76), e8m0_scale_table_gen(77),
        e8m0_scale_table_gen(78), e8m0_scale_table_gen(79), e8m0_scale_table_gen(80),
        e8m0_scale_table_gen(81), e8m0_scale_table_gen(82), e8m0_scale_table_gen(83),
        e8m0_scale_table_gen(84), e8m0_scale_table_gen(85), e8m0_scale_table_gen(86),
        e8m0_scale_table_gen(87), e8m0_scale_table_gen(88), e8m0_scale_table_gen(89),
        e8m0_scale_table_gen(90), e8m0_scale_table_gen(91), e8m0_scale_table_gen(92),
        e8m0_scale_table_gen(93), e8m0_scale_table_gen(94), e8m0_scale_table_gen(95),
        e8m0_scale_table_gen(96), e8m0_scale_table_gen(97), e8m0_scale_table_gen(98),
        e8m0_scale_table_gen(99), e8m0_scale_table_gen(100), e8m0_scale_table_gen(101),
        e8m0_scale_table_gen(102), e8m0_scale_table_gen(103), e8m0_scale_table_gen(104),
        e8m0_scale_table_gen(105), e8m0_scale_table_gen(106), e8m0_scale_table_gen(107),
        e8m0_scale_table_gen(108), e8m0_scale_table_gen(109), e8m0_scale_table_gen(110),
        e8m0_scale_table_gen(111), e8m0_scale_table_gen(112), e8m0_scale_table_gen(113),
        e8m0_scale_table_gen(114), e8m0_scale_table_gen(115), e8m0_scale_table_gen(116),
        e8m0_scale_table_gen(117), e8m0_scale_table_gen(118), e8m0_scale_table_gen(119),
        e8m0_scale_table_gen(120), e8m0_scale_table_gen(121), e8m0_scale_table_gen(122),
        e8m0_scale_table_gen(123), e8m0_scale_table_gen(124), e8m0_scale_table_gen(125),
        e8m0_scale_table_gen(126), e8m0_scale_table_gen(127), e8m0_scale_table_gen(128),
        e8m0_scale_table_gen(129), e8m0_scale_table_gen(130), e8m0_scale_table_gen(131),
        e8m0_scale_table_gen(132), e8m0_scale_table_gen(133), e8m0_scale_table_gen(134),
        e8m0_scale_table_gen(135), e8m0_scale_table_gen(136), e8m0_scale_table_gen(137),
        e8m0_scale_table_gen(138), e8m0_scale_table_gen(139), e8m0_scale_table_gen(140),
        e8m0_scale_table_gen(141), e8m0_scale_table_gen(142), e8m0_scale_table_gen(143),
        e8m0_scale_table_gen(144), e8m0_scale_table_gen(145), e8m0_scale_table_gen(146),
        e8m0_scale_table_gen(147), e8m0_scale_table_gen(148), e8m0_scale_table_gen(149),
        e8m0_scale_table_gen(150), e8m0_scale_table_gen(151), e8m0_scale_table_gen(152),
        e8m0_scale_table_gen(153), e8m0_scale_table_gen(154), e8m0_scale_table_gen(155),
        e8m0_scale_table_gen(156), e8m0_scale_table_gen(157), e8m0_scale_table_gen(158),
        e8m0_scale_table_gen(159), e8m0_scale_table_gen(160), e8m0_scale_table_gen(161),
        e8m0_scale_table_gen(162), e8m0_scale_table_gen(163), e8m0_scale_table_gen(164),
        e8m0_scale_table_gen(165), e8m0_scale_table_gen(166), e8m0_scale_table_gen(167),
        e8m0_scale_table_gen(168), e8m0_scale_table_gen(169), e8m0_scale_table_gen(170),
        e8m0_scale_table_gen(171), e8m0_scale_table_gen(172), e8m0_scale_table_gen(173),
        e8m0_scale_table_gen(174), e8m0_scale_table_gen(175), e8m0_scale_table_gen(176),
        e8m0_scale_table_gen(177), e8m0_scale_table_gen(178), e8m0_scale_table_gen(179),
        e8m0_scale_table_gen(180), e8m0_scale_table_gen(181), e8m0_scale_table_gen(182),
        e8m0_scale_table_gen(183), e8m0_scale_table_gen(184), e8m0_scale_table_gen(185),
        e8m0_scale_table_gen(186), e8m0_scale_table_gen(187), e8m0_scale_table_gen(188),
        e8m0_scale_table_gen(189), e8m0_scale_table_gen(190), e8m0_scale_table_gen(191),
        e8m0_scale_table_gen(192), e8m0_scale_table_gen(193), e8m0_scale_table_gen(194),
        e8m0_scale_table_gen(195), e8m0_scale_table_gen(196), e8m0_scale_table_gen(197),
        e8m0_scale_table_gen(198), e8m0_scale_table_gen(199), e8m0_scale_table_gen(200),
        e8m0_scale_table_gen(201), e8m0_scale_table_gen(202), e8m0_scale_table_gen(203),
        e8m0_scale_table_gen(204), e8m0_scale_table_gen(205), e8m0_scale_table_gen(206),
        e8m0_scale_table_gen(207), e8m0_scale_table_gen(208), e8m0_scale_table_gen(209),
        e8m0_scale_table_gen(210), e8m0_scale_table_gen(211), e8m0_scale_table_gen(212),
        e8m0_scale_table_gen(213), e8m0_scale_table_gen(214), e8m0_scale_table_gen(215),
        e8m0_scale_table_gen(216), e8m0_scale_table_gen(217), e8m0_scale_table_gen(218),
        e8m0_scale_table_gen(219), e8m0_scale_table_gen(220), e8m0_scale_table_gen(221),
        e8m0_scale_table_gen(222), e8m0_scale_table_gen(223), e8m0_scale_table_gen(224),
        e8m0_scale_table_gen(225), e8m0_scale_table_gen(226), e8m0_scale_table_gen(227),
        e8m0_scale_table_gen(228), e8m0_scale_table_gen(229), e8m0_scale_table_gen(230),
        e8m0_scale_table_gen(231), e8m0_scale_table_gen(232), e8m0_scale_table_gen(233),
        e8m0_scale_table_gen(234), e8m0_scale_table_gen(235), e8m0_scale_table_gen(236),
        e8m0_scale_table_gen(237), e8m0_scale_table_gen(238), e8m0_scale_table_gen(239),
        e8m0_scale_table_gen(240), e8m0_scale_table_gen(241), e8m0_scale_table_gen(242),
        e8m0_scale_table_gen(243), e8m0_scale_table_gen(244), e8m0_scale_table_gen(245),
        e8m0_scale_table_gen(246), e8m0_scale_table_gen(247), e8m0_scale_table_gen(248),
        e8m0_scale_table_gen(249), e8m0_scale_table_gen(250), e8m0_scale_table_gen(251),
        e8m0_scale_table_gen(252), e8m0_scale_table_gen(253), e8m0_scale_table_gen(254),
        e8m0_scale_table_gen(255),
    };
    return table;
}

// C[M,N] fp32 = A[M,K]bf16 x W[N,K/2]fp4^T, group scale along N (groupN) and K
// (groupK), optional global scale. W row major [N][K/2]. S=[N/gn][K/gk].
// lut: 16-entry float table mapping nibble -> weight value.
// E8M0: when true, S is a raw uint8 fp8_e8m0 byte buffer (fork MXFP4 feeding),
// decoded as 2^(byte-127); when false, S is fp32 values (WNA16/NVFP4 feeding).
template <bool E8M0 = false, bool FAST_FP4 = false>
inline void matmul_packed4_group(const uint16_t* A, const uint8_t* W,
                                 const float* lut, const void* S,
                                 float global_scale, float* C,
                                 int M, int N, int K,
                                 int groupN, int groupK,
                                 int n0 = 0, int n1 = -1,
                                 int rowshift = 0) {
    if (K <= 0 || (K & 1)) return;  // packed layout requires even K
    if (n1 < 0 || n1 > N) n1 = N;
    if (n1 <= n0) return;
    // rowshift > 0 means W is a shard holding global rows [rowshift, rowshift+N):
    // local row j (global output row j) reads weight slice row j - rowshift.
    const int gn = groupN > 0 ? groupN : 1;
    const int gk = groupK > 0 ? groupK : 1;
    const float* Srow = static_cast<const float*>(S);
    auto scale_at = [&](int n, int kbase) -> float {
        const int idx = (n / gn) * ((K + gk - 1) / gk) + (kbase / gk);
        if constexpr (E8M0) return e8m0_table()[static_cast<const uint8_t*>(S)[idx]];
        return Srow[idx];
    };
#if defined(__AVX512F__)
    // ----------------------------------------------------------------------
    // FAST FP4 (E2M1) gather-free AVX512 path, matching the approach used by
    // lk-moe's `_avx512_*` engine (decompile: PSHUFB nibble decode -> vpmovzx
    // zero-extend -> vcvtdq2ps -> vfmadd231ps, FP32 accumulate on the 512-bit
    // datapath -- no BF16 materialization, no int8). Keeps weights FP4 (packed
    // 2 values/byte) and dequantizes to FP32 in-kernel; on Zen4 the 512-bit
    // `_mm512_fmadd_ps` runs ~2x the FMA rate of the old 256-bit mul+add path.
    // Decoded into NATURAL column order so activations need no permutation
    // (dot = sum_k A[k]*W[k]); semantics identical to the AVX2 path below.
    // ----------------------------------------------------------------------
    if (FAST_FP4 && gk == 32 && (K & 31) == 0 && (size_t)M * (size_t)K <= (size_t)(4 << 20)) {
        const bool bp_on = byteprof_on();
        uint64_t bp_t0 = 0;
        if (bp_on) bp_t0 = std::chrono::steady_clock::now().time_since_epoch().count();
        // Activations -> FP32 in NATURAL (non-permuted) order once, so the inner
        // loop pairs plain zmm loads with the decoded natural-order weights.
        thread_local std::vector<float> a32_storage;
        if (a32_storage.size() < (size_t)M * (size_t)K) a32_storage.resize((size_t)M * (size_t)K);
        float* a32 = a32_storage.data();
        for (int mi = 0; mi < M; mi++) {
            const uint16_t* a_row = A + (size_t)mi * K;
            float* p_row = a32 + (size_t)mi * K;
            for (int k = 0; k < K; k++) p_row[k] = bf16::bf16_to_fp32(a_row[k]);
        }
        // E2M1 -> BF16 byte LUTs (same values as packed4::E2M1, APACHE-2.0 ktransformers).
        alignas(16) static constexpr uint8_t fp4_bf16_lo[16] = {
            0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0,
            0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0};
        alignas(16) static constexpr uint8_t fp4_bf16_hi[16] = {
            0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40,
            0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0};
        const __m256i lut_lo256_ = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*)fp4_bf16_lo));
        const __m256i lut_hi256_ = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*)fp4_bf16_hi));
        const __m128i nib_mask = _mm_set1_epi8(0x0F);
        // Decode one 32-value K-group (16 packed bytes at b_row+g*16) into
        // 32 bf16(u16) in NATURAL column order [W0..W31] (nibble c%2 of byte c/2).
#define XIAOTU_DECODE_GROUP_AVX512(b_row, g_in)                                        \
        const __m128i raw_ = _mm_loadu_si128((const __m128i*)((b_row) + (size_t)(g_in) * 16)); \
        const __m128i lo_ = _mm_and_si128(raw_, nib_mask);                            \
        const __m128i hi_ = _mm_and_si128(_mm_srli_epi16(raw_, 4), nib_mask);         \
        const __m128i sello_ = _mm_unpacklo_epi8(lo_, hi_);  /* nibbles, cols 0..15 */ \
        const __m128i selhi_ = _mm_unpackhi_epi8(lo_, hi_);  /* nibbles, cols16..31 */ \
        const __m256i sel_ = _mm256_inserti128_si256(_mm256_castsi128_si256(sello_), selhi_, 1); \
        const __m256i bl_ = _mm256_shuffle_epi8(lut_lo256_, sel_); /* lo byte/col */  \
        const __m256i bh_ = _mm256_shuffle_epi8(lut_hi256_, sel_); /* hi byte/col */  \
        /* NOTE: _mm256_unpack*_epi8 operate per 128-bit lane, so the interleaved
           u16 land split as [lo-lane(u_lo=cols0-7) | hi-lane(u_lo=cols16-23)] and
           u_hi=[cols8-15 | cols24-31]. Recombine the two lane-halves into
           contiguous natural halves before the word->zmm widen:                */ \
        const __m256i u_lo_ = _mm256_unpacklo_epi8(bl_, bh_);  /* [c0..7 | c16..23] */ \
        const __m256i u_hi_ = _mm256_unpackhi_epi8(bl_, bh_);  /* [c8..15 | c24..31] */ \
        const __m256i A_ = _mm256_inserti128_si256(                                 \
            _mm256_castsi128_si256(_mm256_extracti128_si256(u_lo_, 0)),             \
            _mm256_extracti128_si256(u_hi_, 0), 1);  /* [c0..7 | c8..15] = c0..15 */ \
        const __m256i B_ = _mm256_inserti128_si256(                                 \
            _mm256_castsi128_si256(_mm256_extracti128_si256(u_lo_, 1)),             \
            _mm256_extracti128_si256(u_hi_, 1), 1);  /* [c16..23 | c24..31]        */ \
        const __m512i ilo_ = _mm512_cvtepu16_epi32(A_); /* u32 = 0x0000_XXXX */    \
        /* bf16->fp32 = pattern<<16 (round-toward-zero), matching the AVX2 path's
           _mm256_unpacklo_epi16(zero,u16). Shift (NOT vcvtdq2ps) keeps the numeric
           value: vcvtdq2ps would reinterpret the bf16 pattern as an integer. */   \
        const __m512 wlo_ = _mm512_castsi512_ps(_mm512_slli_epi32(ilo_, 16));      \
        const __m512i ihi_ = _mm512_cvtepu16_epi32(B_);                            \
        const __m512 whi_ = _mm512_castsi512_ps(_mm512_slli_epi32(ihi_, 16));

        const int group_count = K / 32;
        for (int j = n0; j < n1; ++j) {
            const uint8_t* b_row = W + (size_t)(j - rowshift) * (K / 2);
            if (j + 1 < n1) {
                const char* nr = (const char*)(W + (size_t)(j + 1 - rowshift) * (K / 2));
                _mm_prefetch(nr, _MM_HINT_T0);
                _mm_prefetch(nr + 64, _MM_HINT_T0);
                _mm_prefetch(nr + 128, _MM_HINT_T0);
                _mm_prefetch(nr + 192, _MM_HINT_T0);
            }
            // 4-token blocked path: decode each group once, feed 4x zmm accumulators.
            int mi = 0;
            for (; mi + 4 <= M; mi += 4) {
                const float* p0 = a32 + (size_t)(mi + 0) * K;
                const float* p1 = a32 + (size_t)(mi + 1) * K;
                const float* p2 = a32 + (size_t)(mi + 2) * K;
                const float* p3 = a32 + (size_t)(mi + 3) * K;
                __m512 a0[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
                __m512 a1[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
                __m512 a2[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
                __m512 a3[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
                for (int g = 0; g < group_count; g++) {
                    const int base = g * 32;
                    XIAOTU_DECODE_GROUP_AVX512(b_row, g);
                    const float scale = scale_at(j, g * 32);
                    const __m512 sv = _mm512_set1_ps(scale);
                    // ILP-16: round-robin over 4 independent partial chains per
                    // token (4 tokens x 4 partials = 16 in-flight vfmadd231ps),
                    // mirroring lk's 8x64 tile of 16 independent zmm accumulators.
                    // Breaks the serial FMA dependency chain so EPYC's ~4-cycle
                    // FMA latency is hidden; reassociates fp32 (fine for inference).
                    const int p = g & 3;
                    __m512 d0 = _mm512_mul_ps(wlo_, _mm512_loadu_ps(p0 + base));
                    d0 = _mm512_fmadd_ps(whi_, _mm512_loadu_ps(p0 + base + 16), d0);
                    a0[p] = _mm512_fmadd_ps(d0, sv, a0[p]);
                    __m512 d1 = _mm512_mul_ps(wlo_, _mm512_loadu_ps(p1 + base));
                    d1 = _mm512_fmadd_ps(whi_, _mm512_loadu_ps(p1 + base + 16), d1);
                    a1[p] = _mm512_fmadd_ps(d1, sv, a1[p]);
                    __m512 d2 = _mm512_mul_ps(wlo_, _mm512_loadu_ps(p2 + base));
                    d2 = _mm512_fmadd_ps(whi_, _mm512_loadu_ps(p2 + base + 16), d2);
                    a2[p] = _mm512_fmadd_ps(d2, sv, a2[p]);
                    __m512 d3 = _mm512_mul_ps(wlo_, _mm512_loadu_ps(p3 + base));
                    d3 = _mm512_fmadd_ps(whi_, _mm512_loadu_ps(p3 + base + 16), d3);
                    a3[p] = _mm512_fmadd_ps(d3, sv, a3[p]);
                }
                const __m512 s0 = _mm512_add_ps(_mm512_add_ps(a0[0], a0[1]), _mm512_add_ps(a0[2], a0[3]));
                const __m512 s1 = _mm512_add_ps(_mm512_add_ps(a1[0], a1[1]), _mm512_add_ps(a1[2], a1[3]));
                const __m512 s2 = _mm512_add_ps(_mm512_add_ps(a2[0], a2[1]), _mm512_add_ps(a2[2], a2[3]));
                const __m512 s3 = _mm512_add_ps(_mm512_add_ps(a3[0], a3[1]), _mm512_add_ps(a3[2], a3[3]));
                C[(size_t)(mi + 0) * N + j] = hsum512(s0) * global_scale;
                C[(size_t)(mi + 1) * N + j] = hsum512(s1) * global_scale;
                C[(size_t)(mi + 2) * N + j] = hsum512(s2) * global_scale;
                C[(size_t)(mi + 3) * N + j] = hsum512(s3) * global_scale;
            }
            // Single-row remainder (also the whole path when M == 1).
            for (; mi < M; mi++) {
                const float* p0 = a32 + (size_t)mi * K;
                __m512 total0 = _mm512_setzero_ps();
                for (int g = 0; g < group_count; g++) {
                    const int base = g * 32;
                    XIAOTU_DECODE_GROUP_AVX512(b_row, g);
                    __m512 d = _mm512_mul_ps(wlo_, _mm512_loadu_ps(p0 + base));
                    d = _mm512_fmadd_ps(whi_, _mm512_loadu_ps(p0 + base + 16), d);
                    const float scale = scale_at(j, g * 32);
                    total0 = _mm512_fmadd_ps(d, _mm512_set1_ps(scale), total0);
                }
                C[(size_t)mi * N + j] = hsum512(total0) * global_scale;
            }
        }
        if (bp_on) {
            auto bp_t1 = std::chrono::steady_clock::now().time_since_epoch().count();
            byteprof_accum((size_t)(n1 - n0) * (size_t)(K / 2), (uint64_t)(bp_t1 - bp_t0));
        }
        return;
    }
#endif  // __AVX512F__

#if defined(__AVX2__)
    // ----------------------------------------------------------------------
    // FAST FP4 (E2M1) gather-free path, ported from KVCache.AI ktransformers
    // /kt-kernel/operators/avx2/mxfp4-moe.hpp (Apache-2.0), KT_MXFP4_DECODE_GROUP.
    // The old per-8-elements _mm256_i32gather_ps from the 16-entry LUT is
    // latency-bound (~1GB/s vs ~48GB/s memory bandwidth) and was the decode
    // bottleneck (73ms/layer at M=1). This decode expands a whole 32-value
    // K-group (16 packed bytes) with one 256-bit PSHUFB into 4 ymm, then folds
    // with plain FMAs — no gather, bandwidth-bound. Layout is identical to
    // xiaotu's (byte = 2 consecutive K elements, low nibble = even k).
    // ----------------------------------------------------------------------
    if (FAST_FP4 && gk == 32 && (K & 31) == 0 && (size_t)M * (size_t)K <= (size_t)(4 << 20)) {
        const bool bp_on = byteprof_on();
        uint64_t bp_t0 = 0;
        if (bp_on) bp_t0 = std::chrono::steady_clock::now().time_since_epoch().count();
        // Decode emission order within each 32-value group (see macro below).
        static constexpr int kPerm[32] = {0,  2,  4,  6,  1,  3,  5,  7,  8,  10, 12, 14, 9,  11, 13, 15,
                                          16, 18, 20, 22, 17, 19, 21, 23, 24, 26, 28, 30, 25, 27, 29, 31};
        // Pre-permute each activation row to FP32 in decode order once, so the
        // inner loops pair plain loads with the decoded weights.
        thread_local std::vector<float> a_perm_storage;
        if (a_perm_storage.size() < (size_t)M * (size_t)K) a_perm_storage.resize((size_t)M * (size_t)K);
        float* a_perm = a_perm_storage.data();
        const int group_count = K / 32;
        for (int mi = 0; mi < M; mi++) {
            const uint16_t* a_row = A + (size_t)mi * K;
            float* p_row = a_perm + (size_t)mi * K;
            for (int g = 0; g < group_count; g++) {
                const int base = g * 32;
                for (int j = 0; j < 32; j++) p_row[base + j] = bf16::bf16_to_fp32(a_row[base + kPerm[j]]);
            }
        }
        // E2M1 -> BF16 byte LUTs (same values as packed4::E2M1, APACHE-2.0 ktransformers).
        alignas(16) static constexpr uint8_t fp4_bf16_lo[16] = {
            0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0,
            0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0};
        alignas(16) static constexpr uint8_t fp4_bf16_hi[16] = {
            0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40,
            0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0};
        const __m256i lut_lo256 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*)fp4_bf16_lo));
        const __m256i lut_hi256 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*)fp4_bf16_hi));
        const __m256i zero256 = _mm256_setzero_si256();
        const __m128i nib_mask = _mm_set1_epi8(0x0F);
        // Decode one 32-value K-group at b_row + g*16 bytes into w0..w3.
        // w0: cols {0,2,4,6|1,3,5,7}; w1: {8,10,12,14|9,11,13,15};
        // w2: {16,...,22|17,...,23};    w3: {24,...,30|25,...,31}.
#define XIAOTU_DECODE_GROUP(b_row, g_in)                                                \
        const __m128i raw_ = _mm_loadu_si128((const __m128i*)((b_row) + (size_t)(g_in) * 16)); \
        const __m128i lo_ = _mm_and_si128(raw_, nib_mask);                             \
        const __m128i hi_ = _mm_and_si128(_mm_srli_epi16(raw_, 4), nib_mask);          \
        const __m256i v_ = _mm256_set_m128i(hi_, lo_);                                 \
        const __m256i bl_ = _mm256_shuffle_epi8(lut_lo256, v_);                        \
        const __m256i bh_ = _mm256_shuffle_epi8(lut_hi256, v_);                        \
        const __m256i u16a_ = _mm256_unpacklo_epi8(bl_, bh_);                          \
        const __m256i u16b_ = _mm256_unpackhi_epi8(bl_, bh_);                          \
        const __m256 w0_ = _mm256_castsi256_ps(_mm256_unpacklo_epi16(zero256, u16a_)); \
        const __m256 w1_ = _mm256_castsi256_ps(_mm256_unpackhi_epi16(zero256, u16a_)); \
        const __m256 w2_ = _mm256_castsi256_ps(_mm256_unpacklo_epi16(zero256, u16b_)); \
        const __m256 w3_ = _mm256_castsi256_ps(_mm256_unpackhi_epi16(zero256, u16b_))

        for (int j = n0; j < n1; ++j) {
            const uint8_t* b_row = W + (size_t)(j - rowshift) * (K / 2);
            // ktransformers alignment: prefetch the next weight row ahead of the
            // FMA stream (bandwidth-bound; hides DRAM latency for the next row).
            if (j + 1 < n1) {
                const char* nr = (const char*)(W + (size_t)(j + 1 - rowshift) * (K / 2));
                _mm_prefetch(nr, _MM_HINT_T0);
                _mm_prefetch(nr + 64, _MM_HINT_T0);
                _mm_prefetch(nr + 128, _MM_HINT_T0);
                _mm_prefetch(nr + 192, _MM_HINT_T0);
            }
            // 4-token blocked path: decode each group once, feed 4 accumulators.
            int mi = 0;
            for (; mi + 4 <= M; mi += 4) {
                const float* p0 = a_perm + (size_t)(mi + 0) * K;
                const float* p1 = a_perm + (size_t)(mi + 1) * K;
                const float* p2 = a_perm + (size_t)(mi + 2) * K;
                const float* p3 = a_perm + (size_t)(mi + 3) * K;
                __m256 tot0 = _mm256_setzero_ps(), tot1 = _mm256_setzero_ps();
                __m256 tot2 = _mm256_setzero_ps(), tot3 = _mm256_setzero_ps();
                for (int g = 0; g < group_count; g++) {
                    const int base = g * 32;
                    XIAOTU_DECODE_GROUP(b_row, g);
                    const float scale = scale_at(j, g * 32);
                    const __m256 sv = _mm256_set1_ps(scale);
                    __m256 g0 = _mm256_mul_ps(_mm256_loadu_ps(p0 + base), w0_);
                    g0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 8), w1_, g0);
                    g0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 16), w2_, g0);
                    g0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 24), w3_, g0);
                    __m256 g1 = _mm256_mul_ps(_mm256_loadu_ps(p1 + base), w0_);
                    g1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1 + base + 8), w1_, g1);
                    g1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1 + base + 16), w2_, g1);
                    g1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1 + base + 24), w3_, g1);
                    __m256 g2 = _mm256_mul_ps(_mm256_loadu_ps(p2 + base), w0_);
                    g2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2 + base + 8), w1_, g2);
                    g2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2 + base + 16), w2_, g2);
                    g2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2 + base + 24), w3_, g2);
                    __m256 g3 = _mm256_mul_ps(_mm256_loadu_ps(p3 + base), w0_);
                    g3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3 + base + 8), w1_, g3);
                    g3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3 + base + 16), w2_, g3);
                    g3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3 + base + 24), w3_, g3);
                    tot0 = _mm256_fmadd_ps(g0, sv, tot0);
                    tot1 = _mm256_fmadd_ps(g1, sv, tot1);
                    tot2 = _mm256_fmadd_ps(g2, sv, tot2);
                    tot3 = _mm256_fmadd_ps(g3, sv, tot3);
                }
                C[(size_t)(mi + 0) * N + j] = hsum256(tot0) * global_scale;
                C[(size_t)(mi + 1) * N + j] = hsum256(tot1) * global_scale;
                C[(size_t)(mi + 2) * N + j] = hsum256(tot2) * global_scale;
                C[(size_t)(mi + 3) * N + j] = hsum256(tot3) * global_scale;
            }
            // Single-row remainder (also the whole decode path when M == 1).
            for (; mi < M; mi++) {
                const float* p0 = a_perm + (size_t)mi * K;
                __m256 total0 = _mm256_setzero_ps();
                __m256 total1 = _mm256_setzero_ps();
                for (int g = 0; g < group_count; g++) {
                    const int base = g * 32;
                    XIAOTU_DECODE_GROUP(b_row, g);
                    __m256 gacc = _mm256_mul_ps(_mm256_loadu_ps(p0 + base), w0_);
                    gacc = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 8), w1_, gacc);
                    gacc = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 16), w2_, gacc);
                    gacc = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 24), w3_, gacc);
                    const float scale = scale_at(j, g * 32);
                    const __m256 sv = _mm256_set1_ps(scale);
                    if (g & 1)
                        total1 = _mm256_fmadd_ps(gacc, sv, total1);
                    else
                        total0 = _mm256_fmadd_ps(gacc, sv, total0);
                }
                C[(size_t)mi * N + j] = hsum256(_mm256_add_ps(total0, total1)) * global_scale;
            }
        }
        if (bp_on) {
            auto bp_t1 = std::chrono::steady_clock::now().time_since_epoch().count();
            byteprof_accum((size_t)(n1 - n0) * (size_t)(K / 2), (uint64_t)(bp_t1 - bp_t0));
        }
        return;
    }
    // FAST_FP4 cross-parity fallback (gk != 32 or K%32 != 0): FP4 E2M1 dequant
    // without a per-8-elements gather. The values are mag[nib & 7] * sign, with
    // mag in {0,.5,1,1.5,2,3,4,6} and sign = +/-1 from bit 3. A gather from the
    // 16-entry LUT is latency-bound and was ~70x slower than memory bandwidth in
    // the M=1 GEMV hot loop; two permutevar8x32 lookups + one mul replace it.
    static constexpr float kMag8[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                      2.0f, 3.0f, 4.0f, 6.0f};
    const __m256 mag_tab = _mm256_loadu_ps(kMag8);
    const __m256 sign_tab = _mm256_setr_ps(1.0f, -1.0f, 0, 0, 0, 0, 0, 0);
    for (int i = 0; i < M; ++i) {
        const uint16_t* Arow = A + (size_t)i * K;
        float* Crow = C + (size_t)i * N;
        for (int j = n0; j < n1; ++j) {
            const uint8_t* Wrow = W + (size_t)(j - rowshift) * (K / 2);
            __m256 total = _mm256_setzero_ps();
            int kbase = 0;
            for (; kbase < K; kbase += gk) {
                __m256 gacc = _mm256_setzero_ps();
                int k = kbase;
                int kend = (kbase + gk < K) ? (kbase + gk) : K;
                if constexpr (FAST_FP4) {
                    __m256 pre = _mm256_setzero_ps();
                    for (; k + 8 <= kend; k += 8) {
                        __m128i a16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(Arow + k));
                        __m256 av = _mm256_castsi256_ps(
                            _mm256_slli_epi32(_mm256_cvtepu16_epi32(a16), 16));
                        uint32_t raw;
                        std::memcpy(&raw, Wrow + (k / 2), 4);
                        __m128i b = _mm_cvtsi32_si128((int)raw);
                        __m128i lo = _mm_and_si128(b, _mm_set1_epi8(0x0F));
                        __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(0x0F));
                        __m128i nib = _mm_unpacklo_epi8(lo, hi);  // 8 nibbles
                        __m256i magidx = _mm256_cvtepu8_epi32(
                            _mm_and_si128(nib, _mm_set1_epi8(7)));
                        __m256 mag = _mm256_permutevar8x32_ps(mag_tab, magidx);
                        // sign bit = bit 3 -> shift each 16-bit lane right 3,
                        // mask the low byte's bit0 (which was nib bit 3).
                        __m128i s3 = _mm_srli_epi16(nib, 3);
                        __m256i signidx = _mm256_cvtepu8_epi32(
                            _mm_and_si128(s3, _mm_set1_epi8(1)));
                        __m256 sign = _mm256_permutevar8x32_ps(sign_tab, signidx);
                        __m256 wv = _mm256_mul_ps(mag, sign);
                        // scale deferred: fold at group end
                        pre = _mm256_fmadd_ps(av, wv, pre);
                    }
                    float pre_scalar = 0.f;
                    for (; k < kend; ++k) {
                        int nib = (Wrow[k / 2] >> (k & 1 ? 4 : 0)) & 0x0F;
                        float v = kMag8[nib & 7] * ((nib & 8) ? -1.0f : 1.0f);
                        pre_scalar += bf16::bf16_to_fp32(Arow[k]) * v;
                    }
                    float scale = scale_at(j, kbase);
                    total = _mm256_fmadd_ps(pre, _mm256_set1_ps(scale), total);
                    if (pre_scalar != 0.f) {
                        float t[8]; _mm256_storeu_ps(t, total);
                        t[0] += pre_scalar * scale;
                        total = _mm256_loadu_ps(t);
                    }
                } else {
                    __m256 gacc = _mm256_setzero_ps();
                    for (; k + 8 <= kend; k += 8) {
                        __m128i a16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(Arow + k));
                        __m256 av = _mm256_castsi256_ps(
                            _mm256_slli_epi32(_mm256_cvtepu16_epi32(a16), 16));
                        // 8 elements = 4 packed bytes starting at byte k/2.
                        uint32_t raw;
                        std::memcpy(&raw, Wrow + (k / 2), 4);
                        __m128i b = _mm_cvtsi32_si128((int)raw);
                        __m128i lo = _mm_and_si128(b, _mm_set1_epi8(0x0F));
                        __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(0x0F));
                        __m128i nib = _mm_unpacklo_epi8(lo, hi);  // 8 nibble indices
                        __m256i idx = _mm256_cvtepu8_epi32(nib);
                        __m256 wv = _mm256_i32gather_ps(lut, idx, 4);
                        gacc = _mm256_fmadd_ps(av, wv, gacc);
                    }
                    float gscalar = 0.f;
                    for (; k < kend; ++k)
                        gscalar += bf16::bf16_to_fp32(Arow[k]) * lut[nibble_of(Wrow, k)];
                    float scale = scale_at(j, kbase);
                    total = _mm256_fmadd_ps(gacc, _mm256_set1_ps(scale), total);
                    if (gscalar != 0.f) {
                        float t[8]; _mm256_storeu_ps(t, total);
                        t[0] += gscalar * scale;
                        total = _mm256_loadu_ps(t);
                    }
                }
            }
            float tmp[8]; _mm256_storeu_ps(tmp, total);
            float acc = 0.f;
            for (int t = 0; t < 8; ++t) acc += tmp[t];
            Crow[j] = acc * global_scale;
        }
    }
#else
    (void)lut;
    for (int i = 0; i < M; ++i) {
        const uint16_t* Arow = A + (size_t)i * K;
        float* Crow = C + (size_t)i * N;
        for (int j = n0; j < n1; ++j) {
            const uint8_t* Wrow = W + (size_t)(j - rowshift) * (K / 2);
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += bf16::bf16_to_fp32(Arow[k]) *
                       lut[nibble_of(Wrow, k)] * scale_at(j, k - (k % gk));
            Crow[j] = acc * global_scale;
        }
    }
#endif
}

}  // namespace packed4

// Generic CRTP base for a packed-4bit quant type given a dequant LUT.
// The LM `w13_g`/`w2_g` are block-scale arrays and `w13_gs`/`w2_gs` are optional
// per-expert global scales (NVFP4). dequant = lut[nibble] * block_scale * global.
// `Tag` type parameter gives each *configuration* a distinct C++ type so that
// pybind11 can register MXFP4 and NVFP4 (which share the E2M1 LUT) as separate
// classes (pybind11 keys registered classes by C++ typeid).
struct MXFP4Tag;  // forward decl so the base can detect the fast FP4 path
template <const float (&LUT)[16], typename Tag = void, bool E8M0 = false>
struct Packed4WeightTraitsBase
    : WeightTraitsBase<Packed4WeightTraitsBase<LUT, Tag, E8M0>> {
    static constexpr bool kE8M0 = E8M0;   // scale stored as 1-byte e8m0 vs fp32
    // FAST path only for the FP4 E2M1 LUT (MXFP4/NVFP4), where dequant reduces to
    // mag[nib&7]*sign and avoids the slow gather. WNA16's INT4_CENTER8 table does
    // not decompose that way and keeps the general gather path.
    static constexpr bool kFastFP4 = true;
    // This trait can split the N (row) dimension of each GEMV across worker
    // threads (ktransformers `split_range_n`). Requires the *_slice_impl below.
    static constexpr bool kNParallel = true;
    static constexpr size_t w13_bytes_impl(size_t E, size_t n2, size_t H) {
        return E * n2 * (H / 2);          // [E][2I][H/2] packed (2 elem/byte)
    }
    static constexpr size_t w2_bytes_impl(size_t E, size_t H, size_t I) {
        return E * H * (I / 2);           // [E][H][I/2] packed (2 elem/byte)
    }
    static void gate_up_impl(const uint16_t* x, const void* w13, const void* w13_g,
                            const float* w13_gs, float* gate, float* up,
                            int inter, int hidden, size_t eid,
                            int groupN, int groupK) {
        // w13 : [E][2I][H/2] packed; scales [E][2I/gn][H/gk] (fp32 or e8m0 bytes).
        const int n2 = 2 * inter;
        const uint8_t* base = static_cast<const uint8_t*>(w13) + eid * (size_t)n2 * (hidden / 2);
        std::vector<float> both(n2);
        const int gn = groupN > 0 ? groupN : 1;
        const int gk = groupK > 0 ? groupK : 1;
        const size_t nb = (size_t)(n2 + gn - 1) / gn;
        const size_t kb = (size_t)(hidden + gk - 1) / gk;
        const void* sbase = nullptr;
        if (w13_g) {
            // scale element is 1 byte (e8m0) or 4 bytes (fp32) -> byte offset
            const size_t byte_off = eid * (nb * kb) * (E8M0 ? 1u : sizeof(float));
            sbase = static_cast<const char*>(w13_g) + byte_off;
        } else if constexpr (E8M0) {
            static const uint8_t ones[1] = {127};  // 2^(127-127)=1
            sbase = ones;
        } else {
            static const float ones[1] = {1.0f};
            sbase = ones;
        }
        const float gs = w13_gs ? w13_gs[eid] : 1.0f;
        packed4::matmul_packed4_group<E8M0, kFastFP4>(x, base, LUT, sbase, gs, both.data(),
                                            1, n2, hidden, groupN, groupK);
        std::copy(both.begin(), both.begin() + inter, gate);
        std::copy(both.begin() + inter, both.end(), up);
    }

    static void down_impl(const uint16_t* act, const void* w2, const void* w2_g,
                          const float* w2_gs, float* down, int hidden, int inter,
                          size_t eid, int groupN, int groupK) {
        // w2 : [E][H][I/2] packed; scales [E][H/gn][I/gk] (fp32 or e8m0 bytes).
        const uint8_t* base = static_cast<const uint8_t*>(w2) + eid * (size_t)hidden * (inter / 2);
        const int gn = groupN > 0 ? groupN : 1;
        const int gk = groupK > 0 ? groupK : 1;
        const size_t nb = (size_t)(hidden + gn - 1) / gn;
        const size_t kb = (size_t)(inter + gk - 1) / gk;
        const void* sbase = nullptr;
        if (w2_g) {
            const size_t byte_off = eid * (nb * kb) * (E8M0 ? 1u : sizeof(float));
            sbase = static_cast<const char*>(w2_g) + byte_off;
        } else if constexpr (E8M0) {
            static const uint8_t ones[1] = {127};
            sbase = ones;
        } else {
            static const float ones[1] = {1.0f};
            sbase = ones;
        }
        const float gs = w2_gs ? w2_gs[eid] : 1.0f;
        packed4::matmul_packed4_group<E8M0, kFastFP4>(act, base, LUT, sbase, gs, down,
                                            1, hidden, inter, groupN, groupK);
    }

    // ---- N-sliced variants (kNParallel) ------------------------------------
    // Compute gate/up only for rows i in [n0, n1) of `inter`. `both` is [2*inter]:
    //   gate in [0, inter), up in [inter, 2*inter).
    static void gate_up_slice_impl(const uint16_t* x, const void* w13, const void* w13_g,
                                   const float* w13_gs, float* both, int inter, int hidden,
                                   size_t eid, int groupN, int groupK, int n0, int n1) {
        if (n1 < 0 || n1 > inter) n1 = inter;
        if (n1 <= n0) return;
        const int n2 = 2 * inter;
        const uint8_t* base = static_cast<const uint8_t*>(w13) + eid * (size_t)n2 * (hidden / 2);
        const int gn = groupN > 0 ? groupN : 1;
        const int gk = groupK > 0 ? groupK : 1;
        const size_t nb = (size_t)(n2 + gn - 1) / gn;
        const size_t kb = (size_t)(hidden + gk - 1) / gk;
        const void* sbase = nullptr;
        if (w13_g) {
            const size_t byte_off = eid * (nb * kb) * (E8M0 ? 1u : sizeof(float));
            sbase = static_cast<const char*>(w13_g) + byte_off;
        } else if constexpr (E8M0) {
            static const uint8_t ones[1] = {127};
            sbase = ones;
        } else {
            static const float ones[1] = {1.0f};
            sbase = ones;
        }
        const float gs = w13_gs ? w13_gs[eid] : 1.0f;
        // w13 rows [0, inter) = gate, [inter, 2*inter) = up. gate slice first;
        // the up call indexes rows [inter+n0, inter+n1) of the same n2 block into
        // C=both (so both[inter+n0..inter+n1) holds up[n0..n1)).
        packed4::matmul_packed4_group<E8M0, kFastFP4>(x, base, LUT, sbase, gs, both,
                                            1, n2, hidden, groupN, groupK, n0, n1);
        packed4::matmul_packed4_group<E8M0, kFastFP4>(x, base, LUT, sbase, gs, both,
                                            1, n2, hidden, groupN, groupK, inter + n0, inter + n1);
    }

    static void down_slice_impl(const uint16_t* act, const void* w2, const void* w2_g,
                                const float* w2_gs, float* down, int hidden, int inter,
                                size_t eid, int groupN, int groupK, int n0, int n1) {
        if (n1 < 0 || n1 > hidden) n1 = hidden;
        if (n1 <= n0) return;
        const uint8_t* base = static_cast<const uint8_t*>(w2) + eid * (size_t)hidden * (inter / 2);
        const int gn = groupN > 0 ? groupN : 1;
        const int gk = groupK > 0 ? groupK : 1;
        const size_t nb = (size_t)(hidden + gn - 1) / gn;
        const size_t kb = (size_t)(inter + gk - 1) / gk;
        const void* sbase = nullptr;
        if (w2_g) {
            const size_t byte_off = eid * (nb * kb) * (E8M0 ? 1u : sizeof(float));
            sbase = static_cast<const char*>(w2_g) + byte_off;
        } else if constexpr (E8M0) {
            static const uint8_t ones[1] = {127};
            sbase = ones;
        } else {
            static const float ones[1] = {1.0f};
            sbase = ones;
        }
        const float gs = w2_gs ? w2_gs[eid] : 1.0f;
        packed4::matmul_packed4_group<E8M0, kFastFP4>(act, base, LUT, sbase, gs, down,
                                            1, hidden, inter, groupN, groupK, n0, n1);
    }

    // ---- Batched N-sliced variants (kNParallel). ----------------------------
    // Same as the single-instance *_slice_impl but with `me` token-rows of the
    // SAME expert run in one GEMM call, so the packed kernel (matmul_packed4_group
    // FAST path) decodes each weight row once and shares it across me rows and
    // gets M-way FMA ILP via its 4-token blocked loop (ktransformers pattern).
    static void gate_up_slice_batch_impl(int me, const uint16_t* xg, const void* w13, const void* w13_g,
                                         const float* w13_gs, float* both_buf, int inter, int hidden,
                                         size_t eid, int groupN, int groupK, int n0, int n1) {
        if (me <= 0) return;
        if (n1 < 0 || n1 > inter) n1 = inter;
        if (n1 <= n0) return;
        const int n2 = 2 * inter;
        const uint8_t* base = static_cast<const uint8_t*>(w13) + eid * (size_t)n2 * (hidden / 2);
        const int gn = groupN > 0 ? groupN : 1;
        const int gk = groupK > 0 ? groupK : 1;
        const size_t nb = (size_t)(n2 + gn - 1) / gn;
        const size_t kb = (size_t)(hidden + gk - 1) / gk;
        const void* sbase = nullptr;
        if (w13_g) {
            const size_t byte_off = eid * (nb * kb) * (E8M0 ? 1u : sizeof(float));
            sbase = static_cast<const char*>(w13_g) + byte_off;
        } else if constexpr (E8M0) {
            static const uint8_t ones[1] = {127};
            sbase = ones;
        } else {
            static const float ones[1] = {1.0f};
            sbase = ones;
        }
        const float gs = w13_gs ? w13_gs[eid] : 1.0f;
        // both_buf rows are strided by n2; indices [n0,n1) = gate chunk and
        // [inter+n0, inter+n1) = up chunk, both over ALL me rows at once.
        packed4::matmul_packed4_group<E8M0, kFastFP4>(xg, base, LUT, sbase, gs, both_buf,
                                            me, n2, hidden, groupN, groupK, n0, n1);
        packed4::matmul_packed4_group<E8M0, kFastFP4>(xg, base, LUT, sbase, gs, both_buf,
                                            me, n2, hidden, groupN, groupK, inter + n0, inter + n1);
    }

    static void down_slice_batch_impl(int me, const uint16_t* actg, const void* w2, const void* w2_g,
                                      const float* w2_gs, float* down_buf, int hidden, int inter,
                                      size_t eid, int groupN, int groupK, int n0, int n1) {
        if (me <= 0) return;
        if (n1 < 0 || n1 > hidden) n1 = hidden;
        if (n1 <= n0) return;
        const uint8_t* base = static_cast<const uint8_t*>(w2) + eid * (size_t)hidden * (inter / 2);
        const int gn = groupN > 0 ? groupN : 1;
        const int gk = groupK > 0 ? groupK : 1;
        const size_t nb = (size_t)(hidden + gn - 1) / gn;
        const size_t kb = (size_t)(inter + gk - 1) / gk;
        const void* sbase = nullptr;
        if (w2_g) {
            const size_t byte_off = eid * (nb * kb) * (E8M0 ? 1u : sizeof(float));
            sbase = static_cast<const char*>(w2_g) + byte_off;
        } else if constexpr (E8M0) {
            static const uint8_t ones[1] = {127};
            sbase = ones;
        } else {
            static const float ones[1] = {1.0f};
            sbase = ones;
        }
        const float gs = w2_gs ? w2_gs[eid] : 1.0f;
        // down_buf rows strided by hidden; write slice [n0,n1) for all me rows.
        packed4::matmul_packed4_group<E8M0, kFastFP4>(actg, base, LUT, sbase, gs, down_buf,
                                            me, hidden, inter, groupN, groupK, n0, n1);
    }
};

// MXFP4: FP4 E2M1 weights, raw fp8_e8m0 uint8 per-block scales (fork feeding),
// no global scale. groupK=32 when fed by the fork.
struct MXFP4Tag {};
using MXFP4WeightTraits = Packed4WeightTraitsBase<packed4::E2M1, MXFP4Tag, true>;

// WNA16: int4 weights (LUT centered at 8), fp32 per-group scales, no global scale.
struct WNA16Tag {};
using WNA16WeightTraits = Packed4WeightTraitsBase<packed4::INT4_CENTER8, WNA16Tag>;

// NVFP4: FP4 E2M1 weights (same nibble layout as MXFP4), fp32 per-group scales
// PLUS a per-expert global scale. groupK is 32 when fed by vLLM.
struct NVFP4Tag {};
using NVFP4WeightTraits = Packed4WeightTraitsBase<packed4::E2M1, NVFP4Tag>;

}  // namespace xiaotu_moe

#endif  // XIAOTU_MOE_MOE_V2_PACKED4_HPP

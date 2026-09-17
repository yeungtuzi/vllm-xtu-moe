// xiaotu-moe: MOE_V2<WeightTraits, ActivationType> — BF16 + quantized closed loop
//
// Phase 1 minimal closed loop: decode a batch of tokens routed through top_k
// experts entirely on the CPU. The aggregated forward loop mirrors the semantics
// of lktransformers' llamafile/moe.cpp (gate -> up -> gated activation -> down ->
// weighted accumulate) but is self-contained: it depends only on bf16_gemm.hpp and
// a small internal std::thread pool, with no ggml/llama.cpp or libnuma dependency.
// The NUMA work-stealing scheduler (backend_numa.cpp) can be dropped back in later
// without changing this interface.
//
// Quantization is factored out behind a CRTP "WeightTraits" interface so the
// orchestration loop is shared verbatim across BF16 / FP8 / WNA16 / MXFP4 / NVFP4:
//   - gate_up_impl computes (gate, up) = x @ [Wg; Wu]^T from a per-expert weight
//     block (and optional scale), writing inter floats each.
//   - down_impl computes down = act @ Wd^T from the (bf16) activation.
// M=1 per token by default; a traits can override to batch tokens if it wants to.
//
// License: Apache-2.0. Structure inspired by KVCache.AI / Qiong GU open source.

#ifndef XIAOTU_MOE_MOE_V2_HPP
#define XIAOTU_MOE_MOE_V2_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <sys/mman.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "../kernels/bf16_gemm.hpp"
#include "numa_pool.hpp"   // persistent NUMA-aware worker pool

namespace xiaotu_moe {

// Minimal engine configuration (field order and names follow the reference
// engine's config semantics).
struct MOEConfigV2 {
    int   num_processes = 1;
    int   process_id = 0;
    int   gpu_id = 0;
    bool  has_gate_proj = true;
    int   expert_num = 0;
    int   top_k = 0;
    int   hidden_size = 0;
    int   intermediate_size = 0;
    int   max_batch_size = 0;
    int   max_num_seqs = 0;
    int   stride = 32;
    int   group_min_len = 10;
    int   group_max_len = 4096;
    int   groupN = 0;
    int   groupK = 0;
    float swiglu_alpha = 1.f;   // sigmoid scale inside the gated activation
    float swiglu_beta = 0.f;    // additive term on `up` before the multiply
    float swiglu_limit = 0.f;   // clamp on both gate (max) and up (+-), 0=off
    int   activation_type = 0;  // 0=plain gated SiLU, 1=clamped SwiGLU
    bool  use_gpu_prefill = false;
};

namespace act {

// gated activation: out[i] = up[i] * silu(gate[i])
// Written as up * g / (1+exp(-g)) rather than up * (g*sig) so that a very large
// |g| (which drives sig -> 0) yields a finite 0 instead of inf*0 = NaN under
// -ffast-math -fno-finite-math-only.
inline void silu_gate(const float* gate, const float* up, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        float g = gate[i];
        out[i] = up[i] * (g / (1.f + std::exp(-g)));
    }
}

// in-place SiLU: v[i] = v[i] / (1 + exp(-v[i]))
inline void silu_single(float* v, int n) {
    for (int i = 0; i < n; ++i)
        v[i] = v[i] / (1.f + std::exp(-v[i]));
}

// Clamped gated activation, bit-for-bit the semantics of vLLM's
// `silu_and_mul_with_clamp(out, in, limit, alpha, beta)`:
//   out = clamp(gate, max=limit) * sigmoid(alpha * clamp(gate, max=limit))
//         * (clamp(up, +-limit) + beta)
// `limit <= 0` disables clamping; (limit=0, alpha=1, beta=0) reduces to
// silu_gate above. Models that need it: GLM-5.x (10.0), DeepSeek-V4 (10.0),
// MiniMax-M3 (7.0), HY-V4 (10.0) — all trained with the clamp, so ignoring it
// changes the experts' output.
__attribute__((always_inline)) inline float silu_gate_one(
    float g, float u, float limit, float alpha, float beta) {
    if (limit > 0.f) {
        if (g > limit) g = limit;
        if (u > limit) u = limit;
        else if (u < -limit) u = -limit;
    }
    return (g / (1.f + std::exp(-alpha * g))) * (u + beta);
}

inline void silu_gate_clamped(const float* gate, const float* up, float* out,
                              int n, float limit, float alpha, float beta) {
    for (int i = 0; i < n; ++i)
        out[i] = silu_gate_one(gate[i], up[i], limit, alpha, beta);
}

} // namespace act

// ---- CRTP base: dispatches to Derived::*_impl ----
template <typename Derived>
struct WeightTraitsBase {
    static constexpr bool kE8M0 = false;  // default: raw fp32 scales; overridden by packed4
    // NVFP4 除了 per-block scale 还有 per-expert 的 **global(tensor) scale**。
    // 丢了它动态范围会塌(MLX/Metal 就是直接拒绝该参数,损失 ~137x 量程)。
    // 我们的内核在 fp32 里应用它,但 `w13_gs/w2_gs` 为 null 时**默认 1.0f**
    // ⇒ 接线漏传会**静默**丢量程。用这个标志让构造函数不能保持沉默。
    static constexpr bool kNeedsGlobalScale = false;
    // Storage bytes for the full [E] weight tensors. Each Derived provides
    // *_impl which accounts for its own element width and packing.
    static constexpr size_t w13_bytes(size_t E, size_t n2, size_t H) {
        return Derived::w13_bytes_impl(E, n2, H);
    }
    static constexpr size_t w2_bytes(size_t E, size_t H, size_t I) {
        return Derived::w2_bytes_impl(E, H, I);
    }
    // gate_up: gate, up [inter] fp32 = x [hidden] bf16 x per-expert W13 [2*inter][hidden]^T
    // w13 points at expert block eid; w13_g optional per-expert block scale
    // [N/groupN][K/groupK]; w13_gs optional per-expert global scale (array) used
    // by NVFP4; groupN/groupK are the scale's grouping (<=0 -> treat as 1).
    static void gate_up(const uint16_t* x, const void* w13, const void* w13_g,
                        const float* w13_gs, float* gate, float* up,
                        int inter, int hidden, size_t eid,
                        int groupN, int groupK) {
        Derived::gate_up_impl(x, w13, w13_g, w13_gs, gate, up, inter, hidden, eid, groupN, groupK);
    }
    // down: down [hidden] fp32 = act [inter] bf16 x per-expert W2 [hidden][inter]^T
    static void down(const uint16_t* act, const void* w2, const void* w2_g,
                     const float* w2_gs, float* down, int hidden, int inter,
                     size_t eid, int groupN, int groupK) {
        Derived::down_impl(act, w2, w2_g, w2_gs, down, hidden, inter, eid, groupN, groupK);
    }
    // N-parallel capability flag. Packed4 (MXFP4/NVFP4) overrides this to true so
    // forward_many can split the (N-rows of the) GEMV across all worker threads
    // (ktransformers `split_range_n` technique). BF16/FP8 stay single-chunk.
    static constexpr bool kNParallel = false;
    // Small-batch N-slicing: when a batch has only a few (token, rank)
    // assignments the per-token loop leaves all but a handful of workers idle,
    // and a single token's GEMV is compute-bound rather than bandwidth-bound, so
    // it dominates decode latency. Traits that can slice their N rows (FP8) set
    // this so forward_many also routes small batches through the N-sliced
    // kernel; large batches keep their grouped/bandwidth-optimal path.
    static constexpr bool kNSliceSmallM = false;
    // N-sliced variants used by the N-parallel dispatch. `both` is [2*inter]:
    //   gate in [0, inter), up in [inter, 2*inter).
    // `down` is [hidden], sliced over hidden. Slice is [n0, n1) over `inter`
    // (gate/up) or `hidden` (down). Derived provides *_slice_impl.
    static void gate_up_slice(const uint16_t* x, const void* w13, const void* w13_g,
                              const float* w13_gs, float* both, int inter, int hidden,
                              size_t eid, int groupN, int groupK, int n0, int n1) {
        Derived::gate_up_slice_impl(x, w13, w13_g, w13_gs, both, inter, hidden, eid, groupN, groupK, n0, n1);
    }
    static void down_slice(const uint16_t* act, const void* w2, const void* w2_g,
                           const float* w2_gs, float* down, int hidden, int inter,
                           size_t eid, int groupN, int groupK, int n0, int n1) {
        Derived::down_slice_impl(act, w2, w2_g, w2_gs, down, hidden, inter, eid, groupN, groupK, n0, n1);
    }
    // Batched slice variants: process `me` token-instances of ONE expert in a
    // single call so the packed kernel decodes each weight row once and
    // amortizes it over me accumulators (M-way FMA ILP + reuse of the decoded
    // weight across tokens — ktransformers 4-token block pattern). `xg` is
    // [me * hidden] (gate/up) contiguous input rows, `actg` [me * inter] (down);
    // `both_buf` [me*2*inter] / `down_buf` [me*hidden]. Slice is [n0,n1) over
    // `inter` (gate/up) or `hidden` (down), covering ALL me rows. Dispatches to
    // Derived::*_batch_impl; the base default loops per row over the
    // single-instance *_slice_impl, and packed4 overrides with a true M>1 kernel
    // (matmul_packed4_group's 4-token blocked path).
    // Sharded-read geometry (single-copy NUMA shards; packed4 only). A compact
    // shard holds only its own slice of every expert block, so the reader has to
    // be told how the slice is laid out:
    //   cstride : bytes between consecutive experts INSIDE the shard
    //             (0 => the dense n2*(hidden/2) layout, i.e. a full block)
    //   row0    : global row index stored at compact offset 0 of the gate part
    //   up_off  : byte gap from the gate base to the up base (0 => dense layout,
    //             i.e. up begins inter*(hidden/2) after gate)
    // Dense callers (socket replicas / single-copy fallback) pass nothing and get
    // byte-identical behaviour to before.
    static void gate_up_slice_batched(int me, const uint16_t* xg, const void* w13, const void* w13_g,
                                      const float* w13_gs, float* both_buf, int inter, int hidden,
                                      size_t eid, int groupN, int groupK, int n0, int n1,
                                      const uint32_t* rowmap = nullptr,
                                      size_t cstride = 0, long row0 = 0, size_t up_off = 0) {
        Derived::gate_up_slice_batch_impl(me, xg, w13, w13_g, w13_gs, both_buf, inter, hidden,
                                          eid, groupN, groupK, n0, n1, rowmap,
                                          cstride, row0, up_off);
    }
    static void gate_up_slice_batch_impl(int me, const uint16_t* xg, const void* w13, const void* w13_g,
                                         const float* w13_gs, float* both_buf, int inter, int hidden,
                                         size_t eid, int groupN, int groupK, int n0, int n1,
                                         const uint32_t* rowmap = nullptr,
                                         size_t cstride = 0, long row0 = 0, size_t up_off = 0) {
        // Only packed4 implements compact shard addressing; the generic traits
        // never take the sharded path (kNParallel == false), so the geometry is
        // unused here (kept in the signature so the shared call site compiles).
        (void)cstride; (void)row0; (void)up_off;
        for (int mi = 0; mi < me; ++mi)
            Derived::gate_up_slice_impl(xg + (size_t)(rowmap ? rowmap[mi] : (uint32_t)mi) * hidden,
                                        w13, w13_g, w13_gs,
                                        both_buf + (size_t)mi * (2 * inter),
                                        inter, hidden, eid, groupN, groupK, n0, n1);
    }
    static void down_slice_batched(int me, const uint16_t* actg, const void* w2, const void* w2_g,
                                   const float* w2_gs, float* down_buf, int hidden, int inter,
                                   size_t eid, int groupN, int groupK, int n0, int n1,
                                   size_t cstride = 0, long row0 = 0) {
        Derived::down_slice_batch_impl(me, actg, w2, w2_g, w2_gs, down_buf, hidden, inter,
                                       eid, groupN, groupK, n0, n1, cstride, row0);
    }
    static void down_slice_batch_impl(int me, const uint16_t* actg, const void* w2, const void* w2_g,
                                      const float* w2_gs, float* down_buf, int hidden, int inter,
                                      size_t eid, int groupN, int groupK, int n0, int n1,
                                      size_t cstride = 0, long row0 = 0) {
        (void)cstride; (void)row0;   // see gate_up_slice_batch_impl
        for (int mi = 0; mi < me; ++mi)
            Derived::down_slice_impl(actg + (size_t)mi * inter, w2, w2_g, w2_gs,
                                     down_buf + (size_t)mi * hidden,
                                     hidden, inter, eid, groupN, groupK, n0, n1);
    }
};

// ---- WeightTraits: BF16 (no quantization) ----
struct BF16WeightTraits : WeightTraitsBase<BF16WeightTraits> {
    using weight_t = uint16_t;      // bf16 storage
    static constexpr size_t w13_bytes_impl(size_t E, size_t n2, size_t H) {
        return E * n2 * H * sizeof(uint16_t);  // [E][2I][H] bf16
    }
    static constexpr size_t w2_bytes_impl(size_t E, size_t H, size_t I) {
        return E * H * I * sizeof(uint16_t);   // [E][H][I] bf16
    }

    static void gate_up_impl(const uint16_t* x, const void* w13, const void* w13_g,
                             const float* w13_gs, float* gate, float* up,
                             int inter, int hidden, size_t eid,
                             int groupN, int groupK) {
        (void)w13_g; (void)w13_gs; (void)groupN; (void)groupK;
        // w13 : [E][2*I][H] bf16, gate block then up block.
        const uint16_t* base = static_cast<const uint16_t*>(w13) +
            eid * (size_t)2 * inter * hidden;
        bf16::matmul_bf16(x, base, gate, 1, inter, hidden);
        bf16::matmul_bf16(x, base + (size_t)inter * hidden, up, 1, inter, hidden);
    }

    static void down_impl(const uint16_t* act, const void* w2, const void* w2_g,
                          const float* w2_gs, float* down, int hidden, int inter,
                          size_t eid, int groupN, int groupK) {
        (void)w2_g; (void)w2_gs; (void)groupN; (void)groupK;
        // w2 : [E][H][I] bf16.
        const uint16_t* base = static_cast<const uint16_t*>(w2) +
            eid * (size_t)hidden * inter;
        bf16::matmul_bf16(act, base, down, 1, hidden, inter);
    }
};

// ---- ActivationType ----
struct BF16Activation {
    using act_t = uint16_t;         // bf16
    static constexpr bool is_fp16 = false;
};

struct FP16Activation {
    using act_t = uint16_t;         // fp16 storage (same width as bf16)
    static constexpr bool is_fp16 = true;
};

// Small parallel-for over a hardware-thread pool. Each worker processes a
// strided slice of [0, n). Used to split independent token work across cores.
class ThreadPool {
public:
    explicit ThreadPool(size_t n = 0)
        : nthreads_(n == 0 ? default_threads() : n) {}

    // run func(i) for i in [0, n). Thread-safe; spawns/joins per call.
    template <typename F>
    void parallel_for(size_t n, const F& func) const {
        if (nthreads_ <= 1 || n <= 1) {
            for (size_t i = 0; i < n; ++i) func(i);
            return;
        }
        size_t nw = std::min(nthreads_, n);
        std::vector<std::thread> pool;
        pool.reserve(nw);
        for (size_t t = 0; t < nw; ++t) {
            pool.emplace_back([&, t]() {
                for (size_t i = t; i < n; i += nw) func(i);
            });
        }
        for (auto& th : pool) th.join();
    }

    size_t nthreads() const { return nthreads_; }

private:
    static size_t default_threads() {
        unsigned hw = std::thread::hardware_concurrency();
        size_t nt = hw > 0 ? (size_t)hw : 1;
        // Allow capping worker count via env (helps isolate thread-count-related
        // faults in the full server; over-subscribing by spawning one thread per
        // core per forward call is wasteful anyway).
        if (const char* e = std::getenv("XIAOTU_MOE_THREADS")) {
            long v = std::atol(e);
            if (v > 0) nt = (size_t)v;
        }
        return nt;
    }
    size_t nthreads_;
};

// Lock-free float accumulate via CAS on the IEEE-754 bit pattern. Used when
// several expert-grouping jobs contribute to the same token's output in
// parallel (x86 aligned 32-bit float is not natively atomic to RMW).
inline void atomic_add_f32(float* p, float v) {
    std::atomic<uint32_t>* a = reinterpret_cast<std::atomic<uint32_t>*>(p);
    uint32_t old = a->load(std::memory_order_relaxed);
    for (;;) {
        float cur;
        std::memcpy(&cur, &old, sizeof(cur));
        float nv = cur + v;
        uint32_t nbits;
        std::memcpy(&nbits, &nv, sizeof(nbits));
        if (a->compare_exchange_weak(old, nbits, std::memory_order_relaxed,
                                     std::memory_order_relaxed))
            break;
    }
}

// ---- MOE_V2 ----
template <typename WeightTraits, typename ActivationType>
class MOE_V2 {
public:
    using wt = WeightTraits;
    using act = ActivationType;

    // w13/g13: routed expert weights. Layout assumption (verified against the
    // real feed at integration time):
    //   w13 = [expert_num][2][intermediate_size][hidden_size] (gate block 0,
    //         up block 1), base pointer (bf16 or quantized bytes).
    //   w13_g = optional per-expert scale for the gate/up block (unused for BF16;
    //           kept for ABI parity).
    // w2/g2: routed expert weights:
    //   w2 = [expert_num][hidden_size][intermediate_size] (down_proj).
    //   w2_g = optional per-expert scale (unused for BF16).
    // w13_gs/w2_gs: optional per-expert global scale arrays (NVFP4), may be null.
    explicit MOE_V2(const MOEConfigV2& cfg,
                    const void* w13, const void* w2,
                    const void* w13_g, const void* w2_g,
                    const float* w13_gs = nullptr, const float* w2_gs = nullptr)
        : cfg_(cfg), pool_(shared_numa_pool(cfg.process_id, cfg.num_processes)) {
        // Validate minimal dims so we never divide by zero downstream.
        if (cfg_.expert_num <= 0 || cfg_.top_k <= 0 ||
            cfg_.hidden_size <= 0 || cfg_.intermediate_size <= 0)
            throw std::runtime_error("MOE_V2: invalid config dims");

        // Resolve the activation once. activation_type: 0 = plain gated SiLU,
        // 1 = clamped SwiGLU (vLLM `silu_and_mul_with_clamp` semantics, used by
        // GLM-5.x / DeepSeek-V4 / MiniMax-M3 via `swiglu_limit`).
        if (cfg_.activation_type != 0 && cfg_.activation_type != 1)
            throw std::runtime_error("MOE_V2: unsupported activation_type");
        swiglu_limit_ = cfg_.swiglu_limit > 0.f ? cfg_.swiglu_limit : 0.f;
        swiglu_alpha_ = cfg_.swiglu_alpha != 0.f ? cfg_.swiglu_alpha : 1.f;
        swiglu_beta_  = cfg_.swiglu_beta;
        // 【2026-09-13 修正·与上游/ lk 语义对齐】
        // 上游把"夹紧的 SwiGLU"表达为 **`MoEActivation.SILU` + 独立的 `clamp_limit`**
        // (见 vllm/.../fused_moe/activation.py:122 "SwiGLU kernels: SILU + clamp_limit …"),
        // 也就是说**夹紧是 `clamp_limit` 的属性,不是激活族的属性**。
        // 我们原来要求 `activation_type == 1` 才夹紧,于是 lk 的编排链
        // (`activation_type = 0`(silu)+ `swiglu_limit = 10`)会**静默丢掉夹紧**
        // ⇒ 端到端出现"大部分对、偶发错"的稳定偏差(实测 NOTES §283/§284)。
        // 现在改为:只要给了 limit / alpha / beta 就夹紧,两种约定都对:
        //   lk 链  : activation_type=0 + limit=10   → 夹紧 ✅
        //   本插件 : activation_type=1 + limit=10   → 夹紧 ✅(行为不变)
        //   数值门禁: activation_type=0 + limit=0    → 不夹紧 ✅(门禁仍通过)
        clamped_ = (swiglu_limit_ > 0.f || swiglu_alpha_ != 1.f || swiglu_beta_ != 0.f);

        // COPY the weight blocks into engine-owned buffers. The fork hands us
        // pointers into the vLLM parameter tensors (self.w13_weight etc.), but
        // after loading vLLM (`clean_weights_after_loading`) drops those params
        // and the backing pages can be reclaimed / parked PROT_NONE, leaving the
        // stored pointers dangling (=> SIGSEGV on the first real forward). The
        // authoritative lk_moe engine likewise snapshots its weights at
        // construction. Here we mirror that: read the bytes once while the source
        // is live and keep our own copy for the lifetime of the engine.
        const int H = cfg_.hidden_size;
        const int I = cfg_.intermediate_size;
        const int E = cfg_.expert_num;
        const int gn = cfg_.groupN > 0 ? cfg_.groupN : 1;
        const int gk = cfg_.groupK > 0 ? cfg_.groupK : 1;
        const bool e8m0 = WeightTraits::kE8M0;
        const size_t n2 = (size_t)2 * I;                       // gate+up rows
        // NOTE: byte sizes are per-weight-trait (BF16 2B, FP8 1B, packed4 H/2).
        // The old H/2 formula was only valid for packed4 and under-sized BF16/FP8
        // by 4x/2x -> out-of-bounds reads -> NaN. Fixed via trait helpers.
        const size_t w13_bytes = WeightTraits::w13_bytes(E, n2, H);
        const size_t w2_bytes = WeightTraits::w2_bytes(E, H, I);
        const size_t w13g_bytes = (size_t)E * ((n2 + gn - 1) / gn) * ((H + gk - 1) / gk) * (e8m0 ? 1u : sizeof(float));
        const size_t w2g_bytes = (size_t)E * ((H + gn - 1) / gn) * ((I + gk - 1) / gk) * (e8m0 ? 1u : sizeof(float));
        // 供 GPU 流式路径查询(见 NOTES §459):引擎自有的这些主机缓冲合起来
        // 就是**一份**完整的专家权重,所以 GPU 路径可以直接 DMA 它们,
        // 不必为了流式而强留 checkpoint 源张量(那要多 269 GiB)。
        g_w13_dense_bytes_ = w13_bytes;
        g_w2_dense_bytes_ = w2_bytes;
        g_w13g_bytes_ = w13g_bytes;
        g_w2g_bytes_ = w2g_bytes;
        g_group_k_ = gk;
        if constexpr (WeightTraits::kNeedsGlobalScale) {
            if (!w13_gs || !w2_gs) {
                fprintf(stderr,
                        "[xtu-moe] WARNING: this backend needs a per-expert GLOBAL "
                        "(tensor) scale (NVFP4) but %s%s%s were not supplied; the "
                        "engine will use 1.0f, which SILENTLY collapses the dynamic "
                        "range (the MLX/Metal NVFP4 failure mode). Pass the tensor "
                        "scales (w13_gs/w2_gs).\n",
                        w13_gs ? "" : "w13_gs ", w2_gs ? "" : "w2_gs ",
                        (w13_gs && w2_gs) ? "" : "-> missing");
            }
        }
        // SINGLE-COPY NUMA SHARDING (new default for the N-parallel packed4 path).
        // The MoE re-reads the ~GB-scale weight blocks every decode step. Two
        // competing designs place those reads physically local to each core:
        //   (a) per-SOCKET replication (old): 2 socket copies, each worker reads
        //       its local copy -> weights read TWICE total, 2x memory.
        //   (b) single-copy sharding (here): split rows across ALL NUMA nodes;
        //       node n owns gate [n*I/NS,(n+1)*I/NS) & down [n*H/NS,(n+1)*H/NS).
        //       Each core reads ONLY its node's rows, all page-local; weights are
        //       read ONCE total, memory = 1 copy. Scales stay as one small full
        //       copy (indexed by absolute row). Env XIAOTU_MOE_NOSHARD=1 forces the
        //       old socket-replica path (A/B toggling). Falls back to socket
        //       replication then single copy on failure / non-divisible dims.
        // 【固定规则,不要再改】NUMA 分片是**唯一**的权重布局:每个 node 的
        // worker 只读写 MPOL_BIND 到自己那份的内存(全部 page-local),node 之间
        // 只交换很小的激活切片/部分和。旧的 XIAOTU_MOE_SINGLECOPY(单份连续拷贝)
        // 模式**已彻底删除**:它让全部线程读同一份内存(7/8 的读跨 node),解码
        // A 阶段慢 ~30 倍、端到端慢 1.67 倍。不要以任何形式恢复它。
        bool sharded_ok = false;
        if constexpr (wt::kNParallel) {
            if (std::getenv("XIAOTU_MOE_NOSHARD") == nullptr) {
                // 【第 212 轮】多 rank 同机:分片数与 node 基址都按 rank 切开,
                // 让 rank r 的 1/world 权重落在它自己那 1/world 个 NUMA node 上
                // (与 numa_pool 的核表切分一致)。world<=1 时行为与以前完全一致。
                // 【§519】核切分与分片单位由**同一个判据**决定 —— 见 numa_pool.hpp 的
                // resolve_shard_mode() 长注释与实测表。默认:node 数/world >= 2 时
                // 按 **node** 分片 + 核按 node 子集切(RANK_SPLIT=1);否则(会退化成
                // 每 rank 一片、等于权重存 2 份)按 socket 分片 + 核按 CCD 交错(=2)。
                // 显式 RANK_SPLIT 优先,保证"分片单位 == 核切分单位"这个不变量。
                int _rank_mode = 2; bool _sock = true;
                resolve_shard_mode(std::max(1, cfg_.num_processes), _rank_mode, _sock);
                const int _world = (_rank_mode >= 2) ? 1 : std::max(1, cfg_.num_processes);
                const int _units = _sock ? numa_socket_count() : numa_node_count();
                rank_node_base_ = std::max(0, cfg_.process_id) * (_units / _world);
                nshard_ = std::max(1, _units / _world);
                // 【§505 用户指导】EPYC/Xeon 上**无视 NPS,直接按 socket 分片**(+ 核心 CCD 交错):
                // 既保住访存局部性(socket 内 node 距离 10/12,跨 socket 32),又天然沿 CCD 均衡,
                // 而且同一份配置在 NPS1/NPS4 上行为一致(不需要"CPU 型号 ⇒ CCD 分布"配置表)。
                // 【§519 更正 §505】上面"node 分片只快一点、不做默认"的判断**已被实测推翻**
                // (2026-09-17):socket 分片让整层权重只落在 2 个 NUMA node 上、worker 却铺满
                // 8 个 node ⇒ 引擎每层 compute 0.44→0.96 ms、单流 24.94→14.17 t/s;
                // 微基准 B=8 上 node 分片比 socket **快 4.7×**(1.76 vs 8.25 ms/层)。
                // 现默认 = **node 分片 + 核按 node 子集切**,只有 node 数/world 会退化到 1 时
                // 才退回 socket(见 resolve_shard_mode 的判据与告警)。
                // 仍可显式覆盖:`XIAOTU_MOE_RANK_SPLIT=2`(socket)/`=1`(node)/
                // `XIAOTU_MOE_NSHARD=<n>`(直接钉分片数)。
                // 注:分片越细 ⇒ 每片的线程数越少(nthreads/NS),片内并行不足。
                if (const char* _ns = std::getenv("XIAOTU_MOE_NSHARD")) {
                    const int v = std::atoi(_ns);
                    if (v >= 2) nshard_ = v;
                }
                const int NS = nshard_;
                if (NS >= 2 && (I % NS == 0) && (H % NS == 0)) {
                    w13_shard_.assign((size_t)NS, nullptr);
                    w2_shard_.assign((size_t)NS, nullptr);
                    sharded_ok = shard_fill_w13(w13) && shard_fill_w2(w2);
                    if (sharded_ok) {
                        // scales: single full copy (tiny, indexed by absolute row j).
                        if (w13_g) { buf_w13_g_ = std::make_unique<uint8_t[]>(w13g_bytes); std::memcpy(buf_w13_g_.get(), w13_g, w13g_bytes); w13_g_ = buf_w13_g_.get(); }
                        else       { w13_g_ = nullptr; }
                        if (w2_g) { buf_w2_g_ = std::make_unique<uint8_t[]>(w2g_bytes); std::memcpy(buf_w2_g_.get(), w2_g, w2g_bytes); w2_g_ = buf_w2_g_.get(); }
                        else       { w2_g_ = nullptr; }
                        // node-0 shard referenced for debug/parity + any fallback
                        w13_ = w13_shard_[0];  w2_ = w2_shard_[0];
                    } else {
                        nshard_ = 0; w13_shard_.clear(); w2_shard_.clear();
                    }
                } else {
                    nshard_ = 0;
                }
            }
            if (!sharded_ok) {
                // 分片不可用(维度不整除 / 分配失败)时的安全网:每 socket 一份副本。
                nsock_ = numa_socket_count();
                sock_fill(w13, w13_bytes, sock_owned_, w13_s_);
                sock_fill(w2, w2_bytes, sock_owned_, w2_s_);
                sock_fill(w13_g, w13g_bytes, sock_owned_, w13g_s_);
                sock_fill(w2_g, w2g_bytes, sock_owned_, w2g_s_);
            }
        }
        if (sharded_ok) {
            // w13_/w2_/scales already set above (sharded mode).
        } else if (nsock_ >= 2) {
            // Replicas for the (unexpected) case that sharding could not be set
            // up; w13_/w2_ reference the socket-0 replica so debug/parity
            // accessors and any single-threaded fallback still see valid memory.
            w13_ = w13_s_[0];   w2_ = w2_s_[0];
            w13_g_ = w13g_s_[0]; w2_g_ = w2g_s_[0];
        } else {
            // Safety net only (single socket / sharding impossible): one copy.
            if (w13) { buf_w13_ = std::make_unique<uint8_t[]>(w13_bytes); std::memcpy(buf_w13_.get(), w13, w13_bytes); w13_ = buf_w13_.get(); }
            else     { w13_ = nullptr; }
            if (w2)  { buf_w2_ = std::make_unique<uint8_t[]>(w2_bytes); std::memcpy(buf_w2_.get(), w2, w2_bytes); w2_ = buf_w2_.get(); }
            else     { w2_ = nullptr; }
            if (w13_g) { buf_w13_g_ = std::make_unique<uint8_t[]>(w13g_bytes); std::memcpy(buf_w13_g_.get(), w13_g, w13g_bytes); w13_g_ = buf_w13_g_.get(); }
            else       { w13_g_ = nullptr; } // no per-expert scale for this block
            if (w2_g) { buf_w2_g_ = std::make_unique<uint8_t[]>(w2g_bytes); std::memcpy(buf_w2_g_.get(), w2_g, w2g_bytes); w2_g_ = buf_w2_g_.get(); }
            else       { w2_g_ = nullptr; } // no per-expert scale for this block
        }
        // global scales are tiny; copy if provided (shared across sockets)
        if (w13_gs) { buf_w13_gs_ = std::make_unique<float[]>(E); std::memcpy(buf_w13_gs_.get(), w13_gs, E * sizeof(float)); w13_gs_ = buf_w13_gs_.get(); }
        else        { w13_gs_ = nullptr; }
        if (w2_gs) { buf_w2_gs_ = std::make_unique<float[]>(E); std::memcpy(buf_w2_gs_.get(), w2_gs, E * sizeof(float)); w2_gs_ = buf_w2_gs_.get(); }
        else       { w2_gs_ = nullptr; }

        if (!w13_ || !w2_) throw std::runtime_error("MOE_V2: null w13/w2");
    }

    ~MOE_V2() {
        for (void* p : sock_owned_) munmap(p, 0);
        for (void* p : shard_owned_) munmap(p, 0);
    }

    // forward_many: M tokens, top_k=k. expert_ids/weights are [M][k] row-major.
    // input [M, hidden] bf16, output [M, hidden] fp32 (accumulated in place).
    void forward_many(int M, int k,
                      const uint32_t* expert_ids, const float* weights,
                      const uint16_t* input, float* output) {
        const int hidden = cfg_.hidden_size;
        const int inter = cfg_.intermediate_size;
        const int nel = cfg_.expert_num;
        const int groupN = cfg_.groupN;
        const int groupK = cfg_.groupK;
        const size_t NASS = (size_t)M * (size_t)k;
        if (M <= 0 || k <= 0 || inter <= 0 || hidden <= 0) return;

        std::lock_guard<std::mutex> lock(mtx_);

        // N-parallel dispatch (packed4): split each expert's GEMV over its N rows
        // across all worker threads so a single token's 4k-row GEMV uses the whole
        // pool, not 1 thread (ktransformers `split_range_n`). BF16/FP8 take the
        // existing single-chunk paths below.
        if constexpr (wt::kNParallel) {
            // 注意:这里**不要**传 wlimit。实测 `parallel_for_limited(limit == nt_)`
            // 会在 warmup 阶段让 worker 卡死(shm_broadcast 超时),
            // 而 `small_batch_workers()` 在 DS-V4 维度上恒等于 nt_ ⇒ 传了也等于没限制,
            // 只会踩到 limited 等待路径的边界竞态。真正要限制 worker 子集需要
            // 先把 numa_pool 的 limited 路径修好(见 report/tuning/NOTES.md §33)。
            //
            // 【第 421 轮】此前这里对 packed4 **无条件**走 N-slice 并 return,导致下面
            // "--- Expert grouping ---" 对 MXFP4 **永远不可达**:prefill 时
            // NASS = M*k 极大(2417 token ⇒ 14502),每个 (token,专家) 对各自读一遍
            // 专家块 ⇒ 流量 ~9.8 TB、冷 prefill 只有 ~59 tok/s(NOTES §420/§421)。
            //
            // 分组路径读的是成员 `w13_`;而 `nshard_ >= 2` 时 `w13_ = w13_shard_[0]`
            // **只是 node0 的紧凑分片**,布局与索引都不匹配 ⇒ 分片模式下**绝不能**放开。
            // 仅当 `XIAOTU_MOE_NOSHARD=1`(socket 副本 / 整块 reader)且批量足够大时,
            // 才允许落到分组路径。**默认(分片)路径行为逐字不变。**
            const size_t _nass_gate = (size_t)M * (size_t)k;
            static const size_t _grp_min = [] {
                const char* e = std::getenv("XIAOTU_MOE_GROUP_MIN_NASS");
                long v = e ? std::atol(e) : 512L;
                return (size_t)(v > 0 ? v : 512L);
            }();
            if ((nshard_ >= 2) || (_nass_gate <= _grp_min)) {
                forward_many_nsliced(M, k, expert_ids, weights, input, output, 0,
                                     wlimit_override());
                return;
            }
            if (std::getenv("XIAOTU_MOE_GROUP_DIAG") != nullptr)
                fprintf(stderr, "[group] packed4 large-batch -> grouping "
                                "M=%d k=%d NASS=%zu nshard=%d\n",
                        M, k, _nass_gate, nshard_);
        }
        // Small-batch N-slicing (decode): with few assignments the grouped and
        // per-token paths both leave most workers idle. Route them through the
        // N-sliced kernel, and cap the worker count at the point where the pool
        // barrier (~1.8us/worker, the caller waits for the slowest one) stops
        // being cheaper than the extra parallelism. The 4x slack keeps a few
        // tokens per thread on the N-sliced path instead of dropping to the
        // per-token loop.
        if constexpr (wt::kNSliceSmallM) {
            static const bool nslice_small = [] {
                const char* e = std::getenv("XIAOTU_MOE_NSLICE_SMALL");
                return !(e && std::atoi(e) == 0);   // =0 forces the legacy path
            }();
            if (nslice_small && NASS <= 4 * (size_t)pool_.nthreads()) {
                size_t _wl = wlimit_override();
                if (_wl == 0) _wl = small_batch_workers(NASS, inter, hidden);
                forward_many_nsliced(M, k, expert_ids, weights, input, output, -1, _wl);
                return;
            }
        }

        // Zero the whole output once, up front (was per-token before).
        std::fill(output, output + (size_t)M * (size_t)hidden, 0.f);

        // --- Expert grouping -------------------------------
        // Old per-token loop read each expert block once per token×rank, so DRAM
        // traffic was ~ batch*top_k * (12 MB per expert block). Instead we group
        // every (token, rank) assignment by its expert and process each active
        // expert's block ONCE, over all tokens routed to it (the block stays hot
        // in cache across the expert's token sub-batch). Bandwidth drops to
        // ~ active_experts * 12 MB (up to ~7x for typical routing diversity).
        //
        // Assignment index ai = t*k + r reconstructs the token t=ai/k.

        // per_expert[e] = list of assignment indexes routed to expert e (w!=0).
        std::vector<std::vector<size_t>> per_expert((size_t)nel);
        std::vector<int> active;
        {
            std::vector<size_t> count((size_t)nel, 0);
            for (size_t ai = 0; ai < NASS; ++ai) {
                uint32_t eid = expert_ids[ai];
                if (eid < (uint32_t)nel && weights[ai] != 0.f) count[eid]++;
            }
            for (int e = 0; e < nel; ++e)
                if (count[e]) { per_expert[e].reserve(count[e]); active.push_back(e); }
            for (size_t ai = 0; ai < NASS; ++ai) {
                uint32_t eid = expert_ids[ai];
                if (eid < (uint32_t)nel && weights[ai] != 0.f) per_expert[eid].push_back(ai);
            }
        }

        // SiLU output scratch: one [inter] f32 block per assignment.
        act_scratch_.resize(NASS * (size_t)inter);
        // Down output scratch: one [hidden] f32 block per assignment.
        down_scratch_.resize(NASS * (size_t)hidden);

        // --- Adaptive dispatch ---------------------------------------------
        // Grouping pays when routing is concentrated (several tokens share an
        // active expert, so its 12 MB block is re-read from cache instead of
        // DRAM per token). With diverse routing (almost every expert active for a
        // handful of tokens) the two-phase + CAS overhead dominates, so we fall
        // back to the direct per-token loop. Threshold: group only when we have
        // >= 8 assignments on average per active expert.
        const size_t nave = active.size();
        // Grouping-factor threshold: group only when we have >= F assignments on
        // average per active expert. F defaults to 8; XIAOTU_MOE_GROUP_FACTOR
        // overrides (a large value forces the per-token path — used to A/B the
        // two paths on identical routing).
        static constexpr size_t kDefaultF = 8;
        size_t F = kDefaultF;
        if (const char* e = std::getenv("XIAOTU_MOE_GROUP_FACTOR")) {
            long v = std::atol(e);
            if (v > 0) F = (size_t)v;
        }
        if (nave * F <= NASS) {
            // --- Grouped path: each active expert's block read once. ---
            // Jobs split each expert's token list into chunks so fewer active
            // experts still keep all worker threads busy. Job = (eid, [ab, ae)).
            struct Job { int eid; size_t ab, ae; };
            std::vector<Job> jobs;
            {
                size_t total_assign = 0;
                for (int e : active) total_assign += per_expert[e].size();
                size_t nt = pool_.nthreads();
                size_t target_jobs = (nt > 4 ? nt * 4 : 16);
                size_t perjob = total_assign / (target_jobs ? target_jobs : 1);
                if (perjob < 8) perjob = 8;   // amortize per-job fixed cost
                if (perjob < 1) perjob = 1;
                for (int e : active) {
                    const auto& lst = per_expert[e];
                    for (size_t b = 0; b < lst.size(); b += perjob)
                        jobs.push_back({e, b, std::min(b + perjob, lst.size())});
                }
            }
            if (jobs.empty()) return;

            // 【§528】XIAOTU_MOE_FUSE_A2B=1:把 Phase1(gate/up+SiLU)与 Phase2(bf16+down)
            // **合成一轮并行**。依据:两相用**完全相同**的 job 划分(jobs 按 eid+[ab,ae) 切),
            // 且每个 assignment 在 Phase2 只读自己在 Phase1 写出的 act_scratch_[ai] —— 
            // **assignment 之间没有任何跨依赖**,所以合成后每个 assignment 的算子顺序逐字不变
            // ⇒ 结果应当**逐位相同**(无归约重排)。动机:§527 实测每层固定开销 ~0.6-0.8 ms
            // 来自"A/B 两轮 per-phase 派发 + 等齐 120 线程"(ovh=0、A:B=2:1 与字节比吻合),
            // 而每层要付 2 轮;合成后只剩 1 轮,且同一专家的 w13/w2 变成背靠背读取(局部性更好)。
            static const bool fuse_a2b = std::getenv("XIAOTU_MOE_FUSE_A2B") != nullptr;
            if (fuse_a2b) {
                pool_.parallel_for(jobs.size(), [&](size_t ji) {
                    const Job& job = jobs[ji];
                    const auto& lst = per_expert[job.eid];
                    std::vector<float> gate_buf(inter), up_buf(inter);
                    std::vector<uint16_t> act_bf16(inter);
                    std::vector<float> down_buf(hidden);
                    float* act_base = act_scratch_.data();
                    float* down_base = down_scratch_.data();
                    for (size_t it = job.ab; it < job.ae; ++it) {
                        size_t ai = lst[it];
                        size_t t = ai / (size_t)k;
                        const uint16_t* xt = input + t * (size_t)hidden;
                        wt::gate_up(xt, w13_, w13_g_, w13_gs_, gate_buf.data(), up_buf.data(),
                                    inter, hidden, job.eid, groupN, groupK);
                        float* act_dst = act_base + ai * (size_t)inter;
                        if (clamped_)
                            ::xiaotu_moe::act::silu_gate_clamped(gate_buf.data(),
                                up_buf.data(), act_dst, inter,
                                swiglu_limit_, swiglu_alpha_, swiglu_beta_);
                        else
                            ::xiaotu_moe::act::silu_gate(gate_buf.data(), up_buf.data(),
                                                         act_dst, inter);
                        bf16::convert_f32_to_bf16(act_dst, act_bf16.data(), (size_t)inter);
                        wt::down(act_bf16.data(), w2_, w2_g_, w2_gs_, down_buf.data(),
                                 hidden, inter, job.eid, groupN, groupK);
                        std::memcpy(down_base + ai * (size_t)hidden, down_buf.data(),
                                    (size_t)hidden * sizeof(float));
                    }
                });
            } else {
            // Phase 1 (parallel over jobs): gate/up + SiLU -> act_scratch_.
            pool_.parallel_for(jobs.size(), [&](size_t ji) {
                const Job& job = jobs[ji];
                const auto& lst = per_expert[job.eid];
                std::vector<float> gate_buf(inter), up_buf(inter);
                float* act_base = act_scratch_.data();
                for (size_t it = job.ab; it < job.ae; ++it) {
                    size_t ai = lst[it];
                    size_t t = ai / (size_t)k;
                    const uint16_t* xt = input + t * (size_t)hidden;
                    wt::gate_up(xt, w13_, w13_g_, w13_gs_, gate_buf.data(), up_buf.data(),
                                inter, hidden, job.eid, groupN, groupK);
                    float* act_dst = act_base + ai * (size_t)inter;
                    if (clamped_)
                        ::xiaotu_moe::act::silu_gate_clamped(gate_buf.data(),
                            up_buf.data(), act_dst, inter,
                            swiglu_limit_, swiglu_alpha_, swiglu_beta_);
                    else
                        ::xiaotu_moe::act::silu_gate(gate_buf.data(), up_buf.data(),
                                                     act_dst, inter);
                }
            });

            // Phase 2 (parallel over jobs): bf16 -> down -> into down_scratch_.
            // (No cross-expert race on `output`: contributions are staged per
            // assignment and reduced token-wise in Phase 3.)
            pool_.parallel_for(jobs.size(), [&](size_t ji) {
                const Job& job = jobs[ji];
                const auto& lst = per_expert[job.eid];
                std::vector<uint16_t> act_bf16(inter);
                std::vector<float> down_buf(hidden);
                const float* act_base = act_scratch_.data();
                float* down_base = down_scratch_.data();
                for (size_t it = job.ab; it < job.ae; ++it) {
                    size_t ai = lst[it];
                    bf16::convert_f32_to_bf16(act_base + ai * (size_t)inter,
                                              act_bf16.data(), (size_t)inter);
                    wt::down(act_bf16.data(), w2_, w2_g_, w2_gs_, down_buf.data(),
                             hidden, inter, job.eid, groupN, groupK);
                    float* dst = down_base + ai * (size_t)hidden;
                    std::memcpy(dst, down_buf.data(), (size_t)hidden * sizeof(float));
                }
            });

            }

            // Phase 3 (parallel over tokens): weighted reduce, per token, in rank
            // order — one thread per token, so no output contention.
            pool_.parallel_for((size_t)M, [&](size_t t) {
                const float* down_base = down_scratch_.data();
                float* out_t = output + t * (size_t)hidden;
                for (int r = 0; r < k; ++r) {
                    size_t ai = t * (size_t)k + r;
                    uint32_t eid = expert_ids[ai];
                    float w = weights[ai];
                    if (eid >= (uint32_t)nel || w == 0.f) continue;
                    const float* d = down_base + ai * (size_t)hidden;
                    for (int h = 0; h < hidden; ++h) out_t[h] += w * d[h];
                }
            });
            return;
        }

        // --- Fallback: direct per-token loop (diverse routing). ---
        pool_.parallel_for((size_t)M, [&](size_t t) {
            const uint16_t* xt = input + t * (size_t)hidden;
            float* out_t = output + t * (size_t)hidden;   // already zeroed above
            std::vector<float> gate_out(inter), up_out(inter), act_out(inter);
            std::vector<uint16_t> act_bf16(inter);
            std::vector<float> down_out(hidden);
            for (int r = 0; r < k; ++r) {
                size_t ai = t * (size_t)k + r;
                uint32_t eid = expert_ids[ai];
                float w = weights[ai];
                if (eid >= (uint32_t)nel || w == 0.f) continue;
                wt::gate_up(xt, w13_, w13_g_, w13_gs_, gate_out.data(), up_out.data(),
                            inter, hidden, eid, groupN, groupK);
                if (clamped_)
                    ::xiaotu_moe::act::silu_gate_clamped(gate_out.data(),
                        up_out.data(), act_out.data(), inter,
                        swiglu_limit_, swiglu_alpha_, swiglu_beta_);
                else
                    ::xiaotu_moe::act::silu_gate(gate_out.data(), up_out.data(),
                                                 act_out.data(), inter);
                bf16::convert_f32_to_bf16(act_out.data(), act_bf16.data(), (size_t)inter);
                wt::down(act_bf16.data(), w2_, w2_g_, w2_gs_, down_out.data(),
                         hidden, inter, eid, groupN, groupK);
                for (int h = 0; h < hidden; ++h) out_t[h] += w * down_out[h];
            }
        });
    }

    // N-parallel forward_many (used when wt::kNParallel). Splits each active
    // expert's gate/up GEMV over chunks of the `inter` row dimension and the down
    // GEMV over chunks of `hidden`, so one token's big GEMV fans out across the
    // whole NUMA pool instead of a single worker thread (ktransformers
    // `split_range_n` technique). Race-free: every job writes disjoint N-slices.
    // Worker count for a small batch: single-thread work is ~NASS*3*I*H MACs at
    // ~8 MAC/cycle (fp8 decode arithmetic), and a pool call costs ~40us +
    // ~1.8us/participating worker, so minimise W/T + overhead(T).
    // 【第 21 轮】worker 子集上限的诊断/标定覆盖。
    // `small_batch_workers()` 的 `single_us` 取自**标称** MAC 率(8 MAC/cycle @3GHz),
    // 在 DS-V4 解码维度(NASS=1*6, I=2048, H=4096)上算得 **6292 µs**,而真实单线程
    // 是百 µs 级 ⇒ 高估约 50× ⇒ `lim` 恒 ≥ nt ⇒ `worker_limit_` 的门闸
    // (`if (w % stride != 0) goto park`) 因为 `stride = nt/lim = 1` **完全失效**
    // ⇒ 60 个 worker 全部参与每一相、并与 vLLM 自己的线程抢核。
    // `XIAOTU_MOE_WLIMIT=N`(N>0)强制该上限;未设/<=0 用公式。见 NOTES §355/§356。
    size_t wlimit_override() const {
        static const long v = [] {
            const char* e = std::getenv("XIAOTU_MOE_WLIMIT");
            return e ? std::atol(e) : 0L;
        }();
        if (v <= 0) return 0;
        const size_t nt = pool_.nthreads();
        size_t lim = (size_t)v;
        if (lim > nt) lim = nt;
        return lim;
    }

    size_t small_batch_workers(size_t NASS, int inter, int hidden) const {
        const size_t nt = pool_.nthreads();
        const double macs = (double)NASS * 3.0 * (double)inter * (double)hidden;
        const double single_us = macs / (8.0 * 3.0e3);          // us at 3GHz, 8 MAC/cycle
        double t = std::sqrt(single_us / 1.8);
        size_t lim = (size_t)(t + 0.5);
        if (lim < 4) lim = 4;
        if (lim > nt) lim = nt;
        return lim;
    }

// 【§536g】阶段探针:`XIAOTU_MOE_TRACE=1` 时每个 trace 点打印一次(默认完全静默)。
// 用途:定位"某形状到底走哪条路径/哪一支 pfor"—— 本 session 因缺它而连错四次(§527/§531/§533/§536),
// 现在把它固化成常驻诊断开关(与 SHARD_DIAG / GROUP_DIAG / BYTEPROF 同类)。
#define XTU_PROBE(tag) do { \
    static const bool _po = [](){ \
        const bool on = (std::getenv("XIAOTU_MOE_TRACE") != nullptr); \
        if (on) fprintf(stderr, "[trace] " tag "\n"); \
        return true; }(); (void)_po; } while (0)
    void forward_many_nsliced(int M, int k,
                              const uint32_t* expert_ids, const float* weights,
                              const uint16_t* input, float* output,
                              int chunk_hint = 0, size_t wlimit = 0) {
        XTU_PROBE("nsliced-ENTER");

        const int hidden = cfg_.hidden_size;
        const int inter = cfg_.intermediate_size;
        const int nel = cfg_.expert_num;
        const int groupN = cfg_.groupN;
        const int groupK = cfg_.groupK;
        const size_t NASS = (size_t)M * (size_t)k;
        if (M <= 0 || k <= 0 || inter <= 0 || hidden <= 0) return;
        prof_init();
        auto _suA = std::chrono::steady_clock::now();
        // 整个 forward_many 的入点(pA0 之前的 setup 也要计入,否则"引擎内部阶段
        // 之和"会小于 binding 侧量到的回调时长,差额无从归属)。
        const auto t_entry = std::chrono::steady_clock::now();

        // Every phase goes through these: wlimit>0 restricts the call to a worker
        // subset (small-batch decode, where the barrier tail would otherwise cost
        // more than the arithmetic it parallelises).
        auto pfor = [&](size_t n, auto&& fn) {
            if (wlimit) pool_.parallel_for_limited(n, wlimit, fn);
            else        pool_.parallel_for(n, fn);
        };
        auto pfor_sharded = [&](int nn, const size_t* jc, auto&& fn) {
            if (wlimit) pool_.parallel_for_sharded_limited(nn, jc, wlimit, fn);
            else        pool_.parallel_for_sharded(nn, jc, fn);
        };

        // 【轮 70】提前预取本层输入激活(去重后约 NASS/k 行 × hidden×2 字节 ≈ 188 KB):
        // gather 阶段实测仍在 ~40 µs(288 KB ⇒ 7 GB/s),瓶颈是**冷 DRAM 延迟**而非带宽,
        // 且它必须等 bookkeeping + exp_off_ 之后才发起访问,这段延迟完全暴露。
        // 在函数入口就把这些行拉进 cache(非阻塞提示,只改时序、不改任何语义/数值),
        // 让 DRAM 往返与后面的 bookkeeping 重叠。
        {
            const size_t grows = (size_t)(NASS / (size_t)(k > 0 ? k : 1)) + 1;
            const char* ip = (const char*)input;
            const size_t ibytes = grows * (size_t)hidden * sizeof(uint16_t);
            for (size_t o = 0; o < ibytes; o += 64) __builtin_prefetch(ip + o, 0, 3);
        }

        std::fill(output, output + (size_t)M * (size_t)hidden, 0.f);

        // Per-expert instance bookkeeping (persistent exp_/active_/inst_idx_
        // members re-used every call). An "instance" = one (token, rank)
        // assignment routing to an expert. Instances of one expert are gathered
        // into CONTIGUOUS rows so each batched matmul runs M>1 (decode a weight
        // row once, amortize over me slots + M-way ILP — ktransformers
        // 4-token block).
        exp_.resize((size_t)nel);
        active_.clear();
        count_.assign((size_t)nel, 0);
        inst_idx_.assign(NASS, (size_t)0);
        for (size_t ai = 0; ai < NASS; ++ai) {
            uint32_t eid = expert_ids[ai];
            if (eid < (uint32_t)nel && weights[ai] != 0.f) count_[eid]++;
        }
        for (int e = 0; e < nel; ++e)
            if (count_[e]) { exp_[e].ai_list.clear(); exp_[e].ai_list.reserve(count_[e]); active_.push_back(e); }
        for (size_t ai = 0; ai < NASS; ++ai) {
            uint32_t eid = expert_ids[ai];
            if (eid < (uint32_t)nel && weights[ai] != 0.f) {
                inst_idx_[ai] = exp_[eid].ai_list.size();
                exp_[eid].ai_list.push_back(ai);
            }
        }
        if (active_.empty()) return;
        auto _suB = std::chrono::steady_clock::now();

        // Gather contiguous per-expert input rows and size the output buffers.
        // resize() only grows capacity; steady state reuses it (no allocation).
        // 【轮 73】gather 彻底删除:改为把"每行在去重输入里的行号"作为 rowmap 交给内核,
        // gate/up 直接从 input 取数,不再先拷贝成每专家连续的 xg ⇒ 省掉一整段 memcpy
        // **和一个完整的并行区**(轮 67-72 实测 setup ~52 µs,其中 gather 区约 40 µs)。
        for (int e : active_) {
            ExpBuf& g = exp_[e];
            const size_t me = g.ai_list.size();
            // 【轮 71】只增不减的 resize:消除 me 在 2/3 间振荡导致的反复零填充
            // (实测 24-34 µs → 1.5 µs)。
            const size_t nx = me * (size_t)hidden;
            const size_t n2 = me * (size_t)2 * (size_t)inter;
            const size_t ni = me * (size_t)inter;
            // xg 已不再使用(保留成员仅为兼容);down 仍被阶段 B/C 使用,必须保证容量。
            if (g.down.size()  < nx) g.down.resize(nx);
            if (g.both.size()  < n2) g.both.resize(n2);
            if (g.act.size()   < ni) g.act.resize(ni);
            if (g.abf16.size() < ni) g.abf16.resize(ni);
            if (g.rowmap.size() < me) g.rowmap.resize(me);
            for (size_t m = 0; m < me; ++m)      // assignment ai 的激活行 = ai / k
                g.rowmap[m] = (uint32_t)(g.ai_list[m] / (size_t)k);
        }
        auto _suB2 = std::chrono::steady_clock::now();
        auto _suG = std::chrono::steady_clock::now();
        const size_t na = active_.size();

        // Number of N-chunks per active expert (~4x jobs/thread, coarse ~128 rows).
        // chunk_hint > 0 forces that many chunks; chunk_hint < 0 selects the
        // small-batch mode (jobs ~= 2x threads, finer ~64-row chunks) used for
        // decode, where a single token's GEMV must span the whole pool.
        const bool small_m = chunk_hint < 0;
        const int hint = chunk_hint > 0 ? chunk_hint : 0;
        // Chunk counts are sized against the EFFECTIVE worker count (the subset
        // when limited), so every participating worker gets ~2-4 tickets.
        const size_t nt_eff = wlimit ? wlimit : pool_.nthreads();
        int nc_gu = 1;
        {   const char* eov = std::getenv("XIAOTU_MOE_NCGU");
            if (hint > 0) nc_gu = hint;
            else if (eov && std::atoi(eov) > 0) nc_gu = std::atoi(eov);
            else if (nt_eff > 1) {
                size_t need = (nt_eff * (small_m ? 2 : 4)) / std::max<size_t>(1, na);
                size_t maxc = (size_t)(inter / (small_m ? 64 : 128));
                if (maxc < 1) maxc = 1;
                if (need > maxc) need = maxc;
                if (need < 1) need = 1;
                nc_gu = (int)need;
            }
        }
        int nc_d = 1;
        {   const char* eov = std::getenv("XIAOTU_MOE_NCD");
            if (hint > 0) nc_d = hint;
            else if (eov && std::atoi(eov) > 0) nc_d = std::atoi(eov);
            else if (nt_eff > 1) {
                size_t need = (nt_eff * (small_m ? 2 : 4)) / std::max<size_t>(1, na);
                size_t maxc = (size_t)(hidden / (small_m ? 64 : 128));
                if (maxc < 1) maxc = 1;
                if (need > maxc) need = maxc;
                if (need < 1) need = 1;
                nc_d = (int)need;
            }
        }

        // Sub-split for the SHARDED weight-read phases (A and B).
        //
        // 【轮 422】以前 subA 只看**活跃专家个数**:
        //     need = ceil(4*tpn/na);  subA = min(need, spanA/32)
        // 专家一多 subA 就塌到 ~3 —— 而真实路由是**长尾**的,不只是"多":
        //   NOTES §442 实测 qlen=2048 时 active=147.7、mean_me=83、**max_me=2048**,
        // 即一个热点专家独占 16.7% 的 assignment,却只被切成 3 刀 ⇒ 它单枪匹马
        // 成为整层的关键路径。微基准里可以精确复现这个形状:
        //   uniform(384 专家, me≈32) 175.8 ms  vs  SKEW=2048/POOL=147 → **552.7 ms**
        // 而服务实测正是 511 ms。**DEDUP 永远测不出这一条**,因为它让被选中的专家
        // 彼此**相等**,而"相等"恰恰会让按 na 推出来的 subA 自动均衡。
        //
        // 改为按**工作量**给每个专家分配刀数:subA_e ∝ me_e / NASS,让每个 job 的
        // 工作量大致相等(target = 4 jobs/worker 合计)。均匀路由下
        // me_e ≈ NASS/na ⇒ subA_e = ceil(4*tpn/na),与原公式**同阶**,
        // 所以均匀情形没有退化。
        //
        // Env XIAOTU_MOE_SHARDSPLIT=N(>0) 仍然强制所有专家都用 N(逐字保留原语义);
        // =0 关闭;不设/负数 = 上面的自适应。
        std::vector<int> subAe, subBe;
        std::vector<size_t> eoffA, eoffB;   // 每个活跃专家的 job 前缀和
        {
            const char* eov = std::getenv("XIAOTU_MOE_SHARDSPLIT");
            const long ov = eov ? std::atol(eov) : -1L;
            const bool sharded = (nshard_ >= 2) && (pool_.nthreads() > 1);
            const int NS = nshard_ > 0 ? nshard_ : 1;
            const size_t tpn = std::max<size_t>(1, pool_.nthreads() / (size_t)NS);
            const size_t spanA = (size_t)std::max(1, inter / NS);
            const size_t spanB = (size_t)std::max(1, hidden / NS);
            const size_t maxA = std::max<size_t>(1, spanA / 32);   // >=32 rows/job
            const size_t maxB = std::max<size_t>(1, spanB / 32);
            const size_t nass = NASS ? NASS : 1;
            const size_t target = 4 * tpn;                          // ~4 jobs/worker
            const bool auto_split = sharded && (ov < 0);
            subAe.assign(na, 1); subBe.assign(na, 1);
            eoffA.assign(na + 1, 0); eoffB.assign(na + 1, 0);
            size_t ta = 0, tb = 0;
            for (size_t e = 0; e < na; ++e) {
                const size_t me = exp_[active_[e]].ai_list.size();
                int ra, rb;
                if (ov > 0) {                       // 显式覆盖:所有专家同一刀数
                    ra = rb = (int)std::min<size_t>((size_t)ov, std::max(maxA, maxB) * 64);
                } else if (auto_split) {
                    size_t want = (target * me + nass - 1) / nass;
                    // Keep the old >=2 floor: measured, uniform routing at B=2048 is
                    // 174.2-178.8 ms at 2 slices vs 180.6-184.1 ms at 1 slice, i.e.
                    // the original `need<2 -> 2` was load-bearing (768 jobs vs 384).
                    // With this floor the UNIFORM case reproduces the old splitting
                    // exactly, so only the skewed case changes.
                    if (want < 2) want = 2;
                    ra = (int)std::min<size_t>(maxA, want);
                    rb = (int)std::min<size_t>(maxB, want);
                } else {
                    ra = rb = 1;
                }
                subAe[e] = ra; subBe[e] = rb;
                eoffA[e] = ta; ta += (size_t)ra;
                eoffB[e] = tb; tb += (size_t)rb;
            }
            eoffA[na] = ta; eoffB[na] = tb;

        }

        // Flattened job index for A2/B0 = sum over active experts of me*nc_gu.
        exp_off_.resize(na + 1);
        exp_off_[0] = 0;
        for (size_t e_idx = 0; e_idx < na; ++e_idx)
            exp_off_[e_idx + 1] = exp_off_[e_idx] + exp_[active_[e_idx]].ai_list.size() * (size_t)nc_gu;
        {   // A2 四段打点(XIAOTU_MOE_SETUP_PROF=1;NOTES §88.1/§113 的正确锚点)
            static const bool _sp = std::getenv("XIAOTU_MOE_SETUP_PROF") != nullptr;
            if (_sp) {
                auto _suC = std::chrono::steady_clock::now();
                static double s_pre = 0, s_res = 0, s_gath = 0, s_nc = 0; static int s_n = 0;
                s_pre  += std::chrono::duration<double, std::milli>(_suB - _suA).count();
                s_res  += std::chrono::duration<double, std::milli>(_suB2 - _suB).count();
                s_gath += std::chrono::duration<double, std::milli>(_suG - _suB2).count();
                s_nc   += std::chrono::duration<double, std::milli>(_suC - _suG).count();
                if (++s_n % 40 == 0) {
                    fprintf(stderr, "[setup-prof] n=%d per-call(us): pre_bookkeeping=%.1f resize=%.1f "
                            "gather=%.1f nc_sub=%.1f total=%.1f\n",
                            s_n, s_pre / s_n * 1e3, s_res / s_n * 1e3, s_gath / s_n * 1e3,
                            s_nc / s_n * 1e3, (s_pre + s_res + s_gath + s_nc) / s_n * 1e3);
                    s_pre = s_res = s_gath = s_nc = 0; s_n = 0;
                }
            }
        }
        const size_t a2_total = exp_off_[na];

        // XIAOTU_MOE_ME_DIAG=1: report the per-forward routing shape. WHY (NOTES
        // §438): the service's real-weight prefill measured 1535 ms/layer at
        // qlen=2048 -- 8.6x the isolated microbench -- and the leading explanation
        // is that routing concentrates enough that a single expert owns more than
        // hidden-based FAST_FP4 limit (4<<20)/hidden = 819 tokens, dropping it to
        // the slow kernel (§437). That is a claim about REAL routing, so it has to
        // be measured, not inferred.
        {
            static const bool me_diag = std::getenv("XIAOTU_MOE_ME_DIAG") != nullptr;
            if (me_diag) {
                const size_t thr = (size_t)((size_t)4 << 20) / (size_t)(hidden > 0 ? hidden : 1);
                size_t mx = 0, over = 0, sum = 0;
                for (size_t e_idx = 0; e_idx < active_.size(); ++e_idx) {
                    const size_t m = exp_[active_[e_idx]].ai_list.size();
                    if (m > mx) mx = m;
                    if (m > thr) ++over;
                    sum += m;
                }
                fprintf(stderr,
                        "[me-diag] M=%d k=%d NASS=%zu active=%zu max_me=%zu "
                        "n_over_thr=%zu thr=%zu mean_me=%.1f nshard=%d\n",
                        M, k, NASS, active_.size(), mx, over, thr,
                        active_.empty() ? 0.0 : (double)sum / (double)active_.size(),
                        nshard_);
                fflush(stderr);
            }
        }

        using clk = std::chrono::steady_clock;
        auto pA0 = clk::now();
        // Phase A: batched gate+up slices. Sharded: each NUMA node computes the
        // rows it owns ([n*I/NS,(n+1)*I/NS)) for EVERY active expert, reading its
        // node-local shard (na jobs per node). Otherwise the legacy flat path
        // splits (expert, inter-chunk) and reads the worker's socket replica.
        if (nshard_ >= 2) {
            const size_t NS = (size_t)nshard_;
            // Compact-shard geometry (matches shard_fill_w13): node n stores
            // [gate cbytes][up cbytes] per expert, with global row0 = n*gu_crows.
            const size_t gu_crows = (size_t)inter / NS;
            const size_t gu_rb = (size_t)hidden / 2;
            const size_t gu_cbytes = gu_crows * gu_rb;
            std::vector<size_t> jc(NS, eoffA[na]);
            pfor_sharded((int)NS, jc.data(), [&](size_t n, size_t job) {
                // job -> (expert, slice) via the per-expert prefix sums (eoffA is
                // strictly increasing because every subAe >= 1).
                size_t e_idx = (size_t)(std::upper_bound(eoffA.begin(), eoffA.end(), job)
                                        - eoffA.begin()) - 1;
                if (e_idx >= active_.size()) return;
                const int subA_e = subAe[e_idx];
                size_t s = job - eoffA[e_idx];
                int eid = active_[e_idx];
                ExpBuf& g = exp_[eid];
                const size_t me = g.ai_list.size();
                if (me == 0) return;
                int n0 = (int)(n * inter / NS);
                int n1 = (int)((n + 1) * inter / NS);
                if (n1 > inter) n1 = inter;
                if (n0 >= n1) return;
                if (subA_e > 1) {         // sub-split node span across node's threads
                    int sep = (n1 - n0 + subA_e - 1) / subA_e;
                    n0 = n0 + (int)s * sep;
                    n1 = std::min<int>(n1, n0 + sep);
                    if (n0 >= n1) return;
                }
                if (MOE_V2::diag_barrier()) {   // 诊断:XIAOTU_MOE_DIAG_BARRIER=1 时空转
                    for (size_t mi = 0; mi < me; ++mi) {
                        float* bs = g.both.data() + mi * (size_t)2 * (size_t)inter;
                        for (int i = n0; i < n1; ++i) { bs[i] = 1.f; bs[inter + i] = 1.f; }
                    }
                } else
                wt::gate_up_slice_batched((int)me, input, w13_shard_[n], w13_g_, w13_gs_,
                                          g.both.data(), inter, hidden, (size_t)eid, groupN, groupK, n0, n1,
                                          g.rowmap.data(),
                                          /*cstride=*/2 * gu_cbytes,
                                          /*row0=*/(long)((size_t)n * gu_crows),
                                          /*up_off=*/gu_cbytes);
                // FUSED (lk does 3 barriers, we now do 3): gated-SiLU + f32->bf16
                // applied inline on this node's chunk for every instance, replacing
                // the former separate A2/B0 global barriers. Identical numerics.
                const float* bc = g.both.data();
                uint16_t* ab = g.abf16.data();
                for (size_t mi = 0; mi < me; ++mi) {
                    const float* bs = bc + mi * (size_t)2 * (size_t)inter;
                    uint16_t* abd = ab + mi * (size_t)inter;
                    if (clamped_) {
                        for (int i = n0; i < n1; ++i)
                            abd[i] = bf16::fp32_to_bf16(::xiaotu_moe::act::silu_gate_one(
                                bs[i], bs[inter + i], swiglu_limit_, swiglu_alpha_,
                                swiglu_beta_));
                    } else {
                        for (int i = n0; i < n1; ++i) {
                            float gv = bs[i];
                            abd[i] = bf16::fp32_to_bf16(bs[inter + i] * (gv / (1.f + std::exp(-gv))));
                        }
                    }
                }
            });
        } else {
            XTU_PROBE("PHASE-A(gu)");
            pfor(active_.size() * (size_t)nc_gu, [&](size_t ji) {
                size_t e_idx = ji / (size_t)nc_gu;
                int c = (int)(ji % (size_t)nc_gu);
                int eid = active_[e_idx];
                ExpBuf& g = exp_[eid];
                const size_t me = g.ai_list.size();
                int n0 = c * inter / nc_gu;
                int n1 = (c + 1) * inter / nc_gu;
                if (n1 > inter) n1 = inter;
                if (n0 >= n1) return;
                const int s = xiaotu_moe::current_socket();   // worker's pinned socket
                wt::gate_up_slice_batched((int)me, input, w13_for(s), w13g_for(s), w13_gs_,
                                          g.both.data(), inter, hidden, (size_t)eid, groupN, groupK, n0, n1,
                                          g.rowmap.data());
                // FUSED gated-SiLU + f32->bf16 (lk-style single phase), replacing A2/B0.
                const float* bc = g.both.data();
                uint16_t* ab = g.abf16.data();
                for (size_t mi = 0; mi < me; ++mi) {
                    const float* bs = bc + mi * (size_t)2 * (size_t)inter;
                    uint16_t* abd = ab + mi * (size_t)inter;
                    if (clamped_) {
                        for (int i = n0; i < n1; ++i)
                            abd[i] = bf16::fp32_to_bf16(::xiaotu_moe::act::silu_gate_one(
                                bs[i], bs[inter + i], swiglu_limit_, swiglu_alpha_,
                                swiglu_beta_));
                    } else {
                        for (int i = n0; i < n1; ++i) {
                            float gv = bs[i];
                            abd[i] = bf16::fp32_to_bf16(bs[inter + i] * (gv / (1.f + std::exp(-gv))));
                        }
                    }
                }
            });
        }
        auto pA1 = clk::now();

        // Phase B: batched down slices. Sharded: node n computes down rows
        // [n*H/NS,(n+1)*H/NS) for every active expert from its node-local shard.
        // Otherwise flat (expert, h-chunk) over the worker's socket replica.
        if (nshard_ >= 2) {
            const size_t NS = (size_t)nshard_;
            // Compact-shard geometry (matches shard_fill_w2): node n stores one
            // cbytes slice per expert, with global row0 = n*d_crows.
            const size_t d_crows = (size_t)hidden / NS;
            const size_t d_rb = (size_t)inter / 2;
            const size_t d_cbytes = d_crows * d_rb;
            std::vector<size_t> jc(NS, eoffB[na]);
            pfor_sharded((int)NS, jc.data(), [&](size_t n, size_t job) {
                size_t e_idx = (size_t)(std::upper_bound(eoffB.begin(), eoffB.end(), job)
                                        - eoffB.begin()) - 1;
                if (e_idx >= active_.size()) return;
                const int subB_e = subBe[e_idx];
                size_t s = job - eoffB[e_idx];
                int eid = active_[e_idx];
                ExpBuf& g = exp_[eid];
                const size_t me = g.ai_list.size();
                if (me == 0) return;
                int n0 = (int)(n * hidden / NS);
                int n1 = (int)((n + 1) * hidden / NS);
                if (n1 > hidden) n1 = hidden;
                if (n0 >= n1) return;
                if (subB_e > 1) {         // sub-split node span across node's threads
                    int sep = (n1 - n0 + subB_e - 1) / subB_e;
                    n0 = n0 + (int)s * sep;
                    n1 = std::min<int>(n1, n0 + sep);
                    if (n0 >= n1) return;
                }
                if (MOE_V2::diag_barrier()) {   // 诊断:同上
                    for (size_t mi = 0; mi < me; ++mi) {
                        float* d = g.down.data() + mi * (size_t)hidden;
                        for (int i = n0; i < n1; ++i) d[i] = 1.f;
                    }
                } else
                wt::down_slice_batched((int)me, g.abf16.data(), w2_shard_[n], w2_g_, w2_gs_,
                                       g.down.data(), hidden, inter, (size_t)eid, groupN, groupK, n0, n1,
                                       /*cstride=*/d_cbytes,
                                       /*row0=*/(long)((size_t)n * d_crows));
            });
        } else {
            XTU_PROBE("PHASE-B(sock)");
            pfor(active_.size() * (size_t)nc_d, [&](size_t ji) {
                size_t e_idx = ji / (size_t)nc_d;
                int c = (int)(ji % (size_t)nc_d);
                int eid = active_[e_idx];
                ExpBuf& g = exp_[eid];
                int n0 = c * hidden / nc_d;
                int n1 = (c + 1) * hidden / nc_d;
                if (n1 > hidden) n1 = hidden;
                if (n0 >= n1) return;
                const int s = xiaotu_moe::current_socket();   // worker's pinned socket
                wt::down_slice_batched((int)g.ai_list.size(), g.abf16.data(), w2_for(s), w2g_for(s), w2_gs_,
                                       g.down.data(), hidden, inter, (size_t)eid, groupN, groupK, n0, n1);
            });
        }
        auto pB1 = clk::now();

        // Phase C: weighted reduce per token (rank order) - no output contention.
            XTU_PROBE("PHASE-C(reduce)");
        pfor((size_t)M, [&](size_t t) {
            float* out_t = output + t * (size_t)hidden;
            for (int r = 0; r < k; ++r) {
                size_t ai = t * (size_t)k + r;
                uint32_t eid = expert_ids[ai];
                float w = weights[ai];
                if (eid >= (uint32_t)nel || w == 0.f) continue;
                const float* d = exp_[eid].down.data() + inst_idx_[ai] * (size_t)hidden;
                for (int h = 0; h < hidden; ++h) out_t[h] += w * d[h];
            }
        });
        auto pC = clk::now();
        prof_add((size_t)M, active_.size(),
                 std::chrono::duration_cast<std::chrono::nanoseconds>(pA1 - pA0).count(),
                 std::chrono::duration_cast<std::chrono::nanoseconds>(pA0 - t_entry).count(),
                 0,
                 std::chrono::duration_cast<std::chrono::nanoseconds>(pB1 - pA1).count(),
                 std::chrono::duration_cast<std::chrono::nanoseconds>(pC - pB1).count(),
                 std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - pC).count());
    }

    // forward_one: single token, single routed expert (for warm-up / tests).
    void forward_one(uint32_t expert_id, float weight,
                     const uint16_t* x, float* out) {
        uint32_t ids[1] = {expert_id};
        float w[1] = {weight};
        forward_many(1, 1, ids, w, x, out);
    }

    // Matches the runtime method name seen in the closed binary (no-op here).
    void warm_up() {}

    // Debug accessors (parity-check helpers; not part of the public API).
    const MOEConfigV2& config() const { return cfg_; }
    const void* debug_w13() const { return w13_; }
    const void* debug_w2() const { return w2_; }

    // Per-socket accessors: return the replica for socket `s`, or the shared
    // single copy when replication is off (nsock_<2 => w13_s_ are null).
    const void* w13_for(int s) const { return s >= 0 && s < 2 && w13_s_[s] ? w13_s_[s] : w13_; }
    const void* w2_for(int s)  const { return s >= 0 && s < 2 && w2_s_[s]  ? w2_s_[s]  : w2_; }
    const void* w13g_for(int s) const { return s >= 0 && s < 2 && w13g_s_[s] ? w13g_s_[s] : w13_g_; }
    const void* w2g_for(int s) const { return s >= 0 && s < 2 && w2g_s_[s]  ? w2g_s_[s]  : w2_g_; }

private:
    MOEConfigV2 cfg_;
    // Activation parameters, resolved once from cfg_ (see act::silu_gate_one).
    // clamped_ == false keeps the original hot loop (plain `up * silu(gate)`).
    bool  clamped_ = false;
    float swiglu_limit_ = 0.f;
    float swiglu_alpha_ = 1.f;
    float swiglu_beta_ = 0.f;
    std::unique_ptr<uint8_t[]> buf_w13_, buf_w2_, buf_w13_g_, buf_w2_g_;
    std::unique_ptr<float[]> buf_w13_gs_, buf_w2_gs_;
    const void* w13_;
    const void* w2_;
    const void* w13_g_;
    const void* w2_g_;
    const float* w13_gs_;
    const float* w2_gs_;
    // Per-socket weight replicas (mmap'd, munmap'd in the destructor). Used by
    // the N-parallel packed4 hot path so each worker reads only its own socket's
    // pages. nsock_==1 when a single socket or when replication was not possible.
    int nsock_ = 1;
    std::vector<void*> sock_owned_;
    const void* w13_s_[2] = {nullptr, nullptr};
    const void* w2_s_[2] = {nullptr, nullptr};
    const void* w13g_s_[2] = {nullptr, nullptr};
    const void* w2g_s_[2] = {nullptr, nullptr};

    // Single-copy NUMA sharding (N-parallel path). When nshard_>=2 the two GB-scale
    // blocks are sharded across NUMA nodes: node n owns gate rows [n*I/NS,(n+1)*I/NS)
    // & down rows [n*H/NS,(n+1)*H/NS). Each node maps a COMPACT region holding
    // EXACTLY its own slice (w13: E*2*cbytes as [gate cbytes][up cbytes] per expert;
    // w2: E*cbytes), mmap'd and MPOL_BIND to that node. Physical RSS is ONE full copy
    // across all nodes and the virtual mapping is also 1x the weights (it used to be
    // NS x the full block). Every weight read is from node-local pages (no cross-node
    // traffic): each layer's weights are read exactly ONCE over the whole machine (vs
    // 2x with per-socket replication). Scales stay as one full copy (tiny, indexed
    // by absolute row); the row->compact-offset mapping is passed as `row0`/`cstride`.
    int nshard_ = 0;
    // 多 rank 同机时本 rank 的 NUMA node 起点(node = rank_node_base_ + shard 下标),
    // 与 numa_pool 的核表切分保持同一套划分。
    int rank_node_base_ = 0;
    std::vector<void*> shard_owned_;
    std::vector<const uint8_t*> w13_shard_;
    std::vector<const uint8_t*> w2_shard_;

    // ---- 引擎自有主机缓冲的尺寸/几何(GPU 流式路径用,NOTES §459)------------
    // 每 node 的紧凑分片合起来是**一份**完整专家权重;scale 是引擎自己复制的一份
    // (未分片)。把这些尺寸记下来,GPU 侧就能直接 DMA 引擎的缓冲,而不必保留
    // checkpoint 源张量(那要多 269 GiB,本机放不下)。
    size_t g_w13_shard_bytes_ = 0, g_w2_shard_bytes_ = 0;
    size_t g_w13_dense_bytes_ = 0, g_w2_dense_bytes_ = 0;
    size_t g_w13g_bytes_ = 0, g_w2g_bytes_ = 0;
    size_t g_w13_crows_ = 0, g_w13_cbytes_ = 0;
    size_t g_w2_crows_ = 0, g_w2_cbytes_ = 0;
    int g_group_k_ = 1;

    // Lightweight per-phase timing (env-gated print). Accumulates wall time of
    // the 5 phases across calls; prints a breakdown every prof_every_ calls.
    bool prof_ = false;
    size_t prof_every_ = 40;
    size_t prof_calls_ = 0;     // 累计调用数(只增,用于标注)
    // 【§566 修】**窗口计数必须与累计计数分开**:打印时把 prof_na_/prof_M_ 等累加器
    // 归零了,却用**从不归零的 prof_calls_** 当分母 ⇒ 打印出的 na/M 会逐次衰减
    // (实测 calls=40 na=12 → 80 na=6 → 120 na=4 → 160 na=3 → 200 na=2),
    // 而门禁/报告取的是最后一行 ⇒ 读到垃圾值(带宽被算成 36 GB/s)。
    size_t prof_win_ = 0;       // 本窗口内的调用数(打印后归零),用作分母
    int64_t prof_A_ = 0, prof_A2_ = 0, prof_B0_ = 0, prof_B_ = 0, prof_C_ = 0, prof_ovh_ = 0;
    size_t prof_M_ = 0, prof_na_ = 0;
    static bool diag_barrier() {
        static const bool v = (std::getenv("XIAOTU_MOE_DIAG_BARRIER") != nullptr);
        return v;
    }

    void prof_init() {
        prof_ = std::getenv("XIAOTU_MOE_PROFILE") != nullptr;
        if (prof_) {
            static bool once = [](){ fprintf(stderr, "[MOE-PROF] profiling ENABLED (XIAOTU_MOE_PROFILE set)\n"); return true; }();
        }
    }
    // 按调用规模分桶:解码主调用是 M=qlen(1+投机token 数),dspark draft 的 MoE 层
    // 会以小 M 混进来。混在一起平均会把两件事搅成一本糊涂账(实测被误导过),
    // 所以分开累计并分别打印。
    struct ProfBucket { size_t calls = 0, na = 0; int64_t setup = 0, A = 0, B = 0, C = 0, ovh = 0; };
    ProfBucket pbuf_[3];
    static int prof_bucket(size_t M) { return M <= 2 ? 0 : (M <= 8 ? 1 : 2); }

    void prof_add(size_t M, size_t na, int64_t dA, int64_t dA2, int64_t dB0,
                  int64_t dB, int64_t dC, int64_t dovh) {
        if (!prof_) return;
        {
            ProfBucket& b = pbuf_[prof_bucket(M)];
            b.calls++; b.na += na;
            b.setup += dA2; b.A += dA; b.B += dB + dB0; b.C += dC; b.ovh += dovh;
            static const char* names[3] = {"M<=2 ", "M3-8 ", "M>8  "};
            static size_t npb = 0;
            if (++npb % 200 == 0) {
                for (int i = 0; i < 3; ++i) {
                    ProfBucket& q = pbuf_[i];
                    if (!q.calls) continue;
                    double c = (double)q.calls;
                    fprintf(stderr,
                        "[NS-PROF] bucket=%s calls=%zu na=%.1f | per-call(us): "
                        "setup=%.0f A=%.0f B=%.0f C=%.0f ovh=%.0f TOTAL=%.0f\n",
                        names[i], q.calls, (double)q.na / c,
                        q.setup / c / 1e3, q.A / c / 1e3, q.B / c / 1e3,
                        q.C / c / 1e3, q.ovh / c / 1e3,
                        (q.setup + q.A + q.B + q.C + q.ovh) / c / 1e3);
                    q = ProfBucket{};
                }
                npb = 0;
            }
        }
        prof_A_ += dA; prof_A2_ += dA2; prof_B0_ += dB0; prof_B_ += dB;
        prof_C_ += dC; prof_ovh_ += dovh; prof_M_ += M; prof_na_ += na; ++prof_calls_; ++prof_win_;
        if (prof_win_ >= prof_every_) {
            double S = (double)(prof_A_ + prof_A2_ + prof_B0_ + prof_B_ + prof_C_ + prof_ovh_) / 1e6;
            // 分母用**窗口**计数(§566);累计值只用于标注 calls=
            double navg = (double)prof_na_ / (double)prof_win_;
            double mavg = (double)prof_M_ / (double)prof_win_;
            // routing-skew histogram over the last call's active experts
            int maxme = 0; long b1=0,b8=0,b32=0,b128=0,bb=0;
            for (int e : active_) {
                int m = (int)exp_[e].ai_list.size(); maxme = std::max(maxme, m);
                if (m<2)b1++; else if(m<8)b8++; else if(m<32)b32++; else if(m<128)b128++; else bb++;
            }
            fprintf(stderr,
                "[MOE-PROF] calls=%zu na=%.0f M=%.0f maxme=%d  skew(1|2-7|8-31|32-127|128+)=%ld|%ld|%ld|%ld|%ld  A=%.1fms A2=%.1fms B0=%.1fms B=%.1fms C=%.1fms ovh=%.1fms (sum %.0fms)\n",
                prof_calls_, navg, mavg, maxme, b1,b8,b32,b128,bb,
                prof_A_/1e6, prof_A2_/1e6, prof_B0_/1e6,
                prof_B_/1e6, prof_C_/1e6, prof_ovh_/1e6, S);
            prof_A_=prof_A2_=prof_B0_=prof_B_=prof_C_=prof_ovh_=prof_M_=prof_na_=0; prof_win_=0;
        }
    }

    // Fill one MPOL_BIND-to-node `node` region of `total` bytes with this node's
    // owned slice copied from `src`. `total` is the COMPACT shard size and the
    // copier emits exactly the node's rows into [0,total); writing faults the pages
    // in on `node`, so every backed page is physically local and the total physical
    // RSS across all nodes equals one full copy.
    static void* shard_region(size_t total, int node,
                              const uint8_t* src,
                              const std::function<void(uint8_t*, const uint8_t*, size_t)>& copier) {
        void* p = mmap(nullptr, total, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) return nullptr;
        // 【THP 默认关,不要再改回去】每个 node 只拥有每个专家的一段稀疏跨度
        // (w13 每专家 8.4MB stride 里只碰 2×512KB,w2 碰 1×512KB),MADV_HUGEPAGE
        // 会让跨度覆盖到的每个 2MB 页**整体**落地 ⇒ 分片区域 3.2GB/层 → 15-19GB/层,
        // 且分布不均(实测 node7 一层 ~5GB):29 层就把单 node 的 193GB 吃光 ⇒
        //   oom-kill: constraint=CONSTRAINT_MEMORY_POLICY, nodemask=7
        // 关掉后 6.9GB/层、速度不降反略快(同一窗口交错,B=6:1.20/1.20/1.23 关 vs
        // 1.20/1.28/1.32 开)。旧注释"不落大页会 thrash 4KB TLB"是错的:跨度连续。
        // 复现该 A/B 时显式 XIAOTU_MOE_SHARD_HUGEPAGE=1(仅调试用)。
        const char* hp = std::getenv("XIAOTU_MOE_SHARD_HUGEPAGE");
        if (hp && std::atoi(hp) != 0) madvise(p, total, MADV_HUGEPAGE);
        unsigned long mask = 1UL << node;
        // 【必须用 mbind,不许用 set_mempolicy】set_mempolicy 给**线程**设策略,
        // 会被之后新建的线程继承:引擎构建期间 vLLM 还在起 pinned 权重缓存等线程,
        // 它们在窗口内分配的大块内存就被绑到单个 node ⇒
        //   oom-kill: constraint=CONSTRAINT_MEMORY_POLICY, nodemask=7,
        //             task=VLLM::EngineCor, anon-rss 603GB
        // 整个 EngineCore 被杀(2026-09-11 实测)。mbind 只作用于这段映射,不会泄漏。
        // 【§520 实验,env 门控,默认关】把"按 socket 分组"**真正实现出来**再量一次:
        // 上面这条 `mask = 1UL << node` + MPOL_BIND 意味着"一个分片 = 一个**单 node**",
        // 所以 §505 的 socket 分片(nshard=2)实际上让**整层权重只落在 2 个 node 上**(§518)。
        // 本开关:每个分片改为 **MPOL_INTERLEAVE 到它所属 socket 的全部 node** ⇒ 保留 socket
        // 粒度,同时把该 socket 的 4 个内存域都用上、且同 socket 内线程距离 ≤12。
        // 用途:回答"是'无视 NPS、只按 socket 分组'这个方向错了,还是我们把它做砸了"。
        int mpol = MPOL_BIND;
        if (std::getenv("XIAOTU_MOE_SHARD_INTERLEAVE_SOCKET") != nullptr) {
            NumaTopology _t = discover_numa_topology();
            const int sock = (node >= 0 && node < (int)_t.node_socket.size())
                             ? _t.node_socket[(size_t)node] : 0;
            unsigned long m2 = 0;
            for (size_t j = 0; j < _t.node_socket.size(); ++j)
                if (_t.node_socket[j] == sock) m2 |= (1UL << j);
            if (m2) { mask = m2; mpol = MPOL_INTERLEAVE; }
        }
        long rc = syscall(SYS_mbind, p, total, mpol, &mask, sizeof(mask) * 8, 0);
        uint8_t* d = static_cast<uint8_t*>(p);
        copier(d, src, total);
        if (std::getenv("XIAOTU_MOE_SHARD_DIAG") != nullptr)
            fprintf(stderr, "[SHARD-DIAG] region node=%d rc=%ld vmasize=%.1fGiB addr=%p%s\n",
                    node, rc, (double)total / (1ULL<<30), p,
                    (rc==0) ? " (mbind)" : " (mbind FAILED -> first-touch!)");
        return p;
    }

    // Shard the gate+up block [E][2I][H/2] across nshard_ nodes: node n owns gate
    // rows [n*I/NS,(n+1)*I/NS) and up rows [I+n*I/NS, I+(n+1)*I/NS).
    //
    // COMPACT layout: each node maps EXACTLY what it stores — per expert
    // [gate cbytes][up cbytes], total E*2*cbytes — instead of mmap'ing the whole
    // E*stride block and writing sparse spans. The reader recovers global rows via
    // row0 == n*crows (see the packed4 gate_up_slice_batch_impl / rowshift).
public:
    // ---- GPU 流式路径的公共访问器(见 report/tuning/NOTES.md §459)---------
    // 这些缓冲是引擎**自有**的(不依赖 checkpoint 源张量),所以 GPU prefill 可以在
    // XIAOTU_RELEASE_SOURCE=1 下工作 —— 主机专家内存从 522 GiB 降到 253 GiB。
    int shard_ns() const { return nshard_; }
    size_t shard_w13_node_bytes() const { return g_w13_shard_bytes_; }
    size_t shard_w2_node_bytes() const { return g_w2_shard_bytes_; }
    size_t scale_w13_bytes() const { return g_w13g_bytes_; }
    size_t scale_w2_bytes() const { return g_w2g_bytes_; }
    size_t shard_w13_crows() const { return g_w13_crows_; }
    size_t shard_w13_cbytes() const { return g_w13_cbytes_; }
    size_t shard_w2_crows() const { return g_w2_crows_; }
    size_t shard_w2_cbytes() const { return g_w2_cbytes_; }
    // which: 0 = w13 (per-node shard), 1 = w2 (per-node shard),
    //        2 = w13 scales (single copy), 3 = w2 scales (single copy)
    const void* host_wbuf(int which, int node) const {
        if (which == 0) return (node >= 0 && node < (int)w13_shard_.size()) ? w13_shard_[node] : nullptr;
        if (which == 1) return (node >= 0 && node < (int)w2_shard_.size()) ? w2_shard_[node] : nullptr;
        if (which == 2) return w13_g_;
        if (which == 3) return w2_g_;
        return nullptr;
    }

private:
    bool shard_fill_w13(const void* src) {
        if (!src || nshard_ < 2) return false;
        const size_t H = cfg_.hidden_size, I = cfg_.intermediate_size, E = cfg_.expert_num;
        const size_t rowbytes = H / 2;
        const int NS = nshard_;
        const size_t crows = I / NS;               // caller guarantees NS | I
        const size_t cbytes = crows * rowbytes;
        const size_t stride = (size_t)2 * I * rowbytes;  // dense source expert block
        const size_t total = (size_t)2 * cbytes * E;     // compact shard size
        g_w13_shard_bytes_ = total;
        g_w13_crows_ = crows;
        g_w13_cbytes_ = cbytes;
        size_t base = shard_owned_.size();
        const uint8_t* s = static_cast<const uint8_t*>(src);
        for (int n = 0; n < NS; ++n) {
            const size_t rs = (size_t)n * crows;
            void* p = shard_region(total, rank_node_base_ + n, s, [&](uint8_t* d, const uint8_t* srcx, size_t) {
                for (size_t e = 0; e < E; ++e) {
                    const size_t sb = e * stride;
                    const size_t db = e * (2 * cbytes);
                    std::memcpy(d + db,          srcx + sb + rs * rowbytes,       cbytes); // gate
                    std::memcpy(d + db + cbytes, srcx + sb + (I + rs) * rowbytes, cbytes); // up
                }
            });
            if (!p) { while (shard_owned_.size() > base) { munmap(shard_owned_.back(), 0); shard_owned_.pop_back(); } return false; }
            w13_shard_[n] = static_cast<const uint8_t*>(p);
            shard_owned_.push_back(p);
        }
        if (std::getenv("XIAOTU_MOE_SHARD_DIAG") != nullptr)
            fprintf(stderr, "[SHARD-DIAG] w13 sharding OK NS=%d shard=%.1fGiB (full would be %.1fGiB)\n",
                    NS, (double)total/(1ULL<<30), (double)(stride * E)/(1ULL<<30));
        return true;
    }

    // Shard the down block [E][H][I/2] across nshard_ nodes: node n owns rows
    // [n*H/NS,(n+1)*H/NS). COMPACT: E*cbytes bytes, one cbytes slice per expert.
    bool shard_fill_w2(const void* src) {
        if (!src || nshard_ < 2) return false;
        const size_t H = cfg_.hidden_size, I = cfg_.intermediate_size, E = cfg_.expert_num;
        const size_t rowbytes = I / 2;
        const int NS = nshard_;
        const size_t crows = H / NS;               // caller guarantees NS | H
        const size_t cbytes = crows * rowbytes;
        const size_t stride = H * rowbytes;              // dense source expert block
        const size_t total = cbytes * E;                 // compact shard size
        g_w2_shard_bytes_ = total;
        g_w2_crows_ = crows;
        g_w2_cbytes_ = cbytes;
        size_t base = shard_owned_.size();
        const uint8_t* s = static_cast<const uint8_t*>(src);
        for (int n = 0; n < NS; ++n) {
            const size_t rs = (size_t)n * crows;
            void* p = shard_region(total, rank_node_base_ + n, s, [&](uint8_t* d, const uint8_t* srcx, size_t) {
                for (size_t e = 0; e < E; ++e)
                    std::memcpy(d + e * cbytes, srcx + e * stride + rs * rowbytes, cbytes);
            });
            if (!p) { while (shard_owned_.size() > base) { munmap(shard_owned_.back(), 0); shard_owned_.pop_back(); } return false; }
            w2_shard_[n] = static_cast<const uint8_t*>(p);
            shard_owned_.push_back(p);
        }
        if (std::getenv("XIAOTU_MOE_SHARD_DIAG") != nullptr)
            fprintf(stderr, "[SHARD-DIAG] w2 sharding OK NS=%d shard=%.1fGiB (full would be %.1fGiB)\n",
                    NS, (double)total/(1ULL<<30), (double)(stride * E)/(1ULL<<30));
        return true;
    }

    // Fill per-socket replicas of one weight block (src -> dst[s]), mmap'd and
    // interleaved within socket s. On partial failure, munmaps only this call's
    // buffers and clears nsock_ so the single-copy fallback is used.
    void sock_fill(const void* src, size_t bytes,
                   std::vector<void*>& owned, const void** dst) {
        dst[0] = dst[1] = nullptr;
        if (!src || nsock_ < 2) return;
        int ok = 0;
        size_t base = owned.size();
        for (int s = 0; s < 2; ++s) {
            void* p = numa_socket_alloc(bytes, s);
            if (!p) break;
            std::memcpy(p, src, bytes);
            owned.push_back(p);
            dst[s] = p;
            ++ok;
        }
        if (ok < 2) {
            while (owned.size() > base) { munmap(owned.back(), 0); owned.pop_back(); }
            dst[0] = dst[1] = nullptr;
            nsock_ = 1;
        }
    }
    // Scratch shared by the two expert-grouping phases within one forward call.
    // Guarded by mtx_ so accidental concurrent forward_many on the same engine is
    // safe (the fork processes layers sequentially; concurrent calls are serialized).
    mutable std::mutex mtx_;
    // Persistent per-expert scratch for the N-parallel batched path. Defeats the
    // per-forward malloc churn of locals: capacities persist across calls, so
    // the steady-state hot loop does zero allocation (the prior local-vector
    // version regressed ~25% because it mmap/munmap'd every buffer every call).
    struct ExpBuf {
        std::vector<uint16_t> xg, abf16;   // me*hidden / me*inter (bf16 rows)
        std::vector<uint32_t> rowmap;      // me:每行在"去重输入"里的行号(轮 73 取代 gather)
        std::vector<float>    both, act, down;  // me*2*inter / me*inter / me*hidden
        std::vector<size_t>   ai_list;
    };
    mutable std::vector<ExpBuf> exp_;       // sized to nel; only active experts used
    mutable std::vector<int> active_;       // reusable list of active expert ids
    mutable std::vector<size_t> count_;     // per-expert instance counts
    mutable std::vector<size_t> inst_idx_;  // ai -> instance rank within its expert
    mutable std::vector<size_t> exp_off_;   // prefix sums for A2/B0 job mapping
    std::vector<float> act_scratch_;
    std::vector<float> down_scratch_;
    std::vector<float> both_scratch_;          // N-parallel gate+up (2*inter/assign)
    std::vector<uint16_t> act_bf16_scratch_;   // N-parallel bf16 activation
    // Shared process-wide NUMA pool (one pool for ALL layers, mirroring lk_moe's
    // single Backend_NUMA engine). Per-layer pools would give ~61 x threads and
    // thrash the scheduler; a single pool keeps the total = XIAOTU_MOE_THREADS.
    NumaWorkPool& pool_;
};

} // namespace xiaotu_moe

#endif // XIAOTU_MOE_MOE_V2_HPP

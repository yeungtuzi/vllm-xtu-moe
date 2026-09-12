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
    static void gate_up_slice_batched(int me, const uint16_t* xg, const void* w13, const void* w13_g,
                                      const float* w13_gs, float* both_buf, int inter, int hidden,
                                      size_t eid, int groupN, int groupK, int n0, int n1) {
        Derived::gate_up_slice_batch_impl(me, xg, w13, w13_g, w13_gs, both_buf, inter, hidden,
                                          eid, groupN, groupK, n0, n1);
    }
    static void gate_up_slice_batch_impl(int me, const uint16_t* xg, const void* w13, const void* w13_g,
                                         const float* w13_gs, float* both_buf, int inter, int hidden,
                                         size_t eid, int groupN, int groupK, int n0, int n1) {
        for (int mi = 0; mi < me; ++mi)
            Derived::gate_up_slice_impl(xg + (size_t)mi * hidden, w13, w13_g, w13_gs,
                                        both_buf + (size_t)mi * (2 * inter),
                                        inter, hidden, eid, groupN, groupK, n0, n1);
    }
    static void down_slice_batched(int me, const uint16_t* actg, const void* w2, const void* w2_g,
                                   const float* w2_gs, float* down_buf, int hidden, int inter,
                                   size_t eid, int groupN, int groupK, int n0, int n1) {
        Derived::down_slice_batch_impl(me, actg, w2, w2_g, w2_gs, down_buf, hidden, inter,
                                       eid, groupN, groupK, n0, n1);
    }
    static void down_slice_batch_impl(int me, const uint16_t* actg, const void* w2, const void* w2_g,
                                      const float* w2_gs, float* down_buf, int hidden, int inter,
                                      size_t eid, int groupN, int groupK, int n0, int n1) {
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
        : cfg_(cfg), pool_(shared_numa_pool()) {
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
        clamped_ = cfg_.activation_type == 1 &&
                   (swiglu_limit_ > 0.f || swiglu_alpha_ != 1.f || swiglu_beta_ != 0.f);

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
                nshard_ = numa_node_count();
                // XIAOTU_MOE_NSHARD=N 覆盖分片数。本机 NPS=4 ⇒ numa_node_count()=8,
                // 但 ACPI 距离矩阵显示**同 socket 内 10/12/12/12、跨 socket 32**:
                // 真正的局部性边界是 socket,不是 NUMA node。NPS=1(BIOS)时
                // numa_node_count()=2 会自动得到同样的效果;在不能重启的机器上
                // 用 NSHARD=2 等价地得到"每个 socket 一份 1/2 分片、1 份权重"。
                // 分片越细 ⇒ 每片的线程数越少(nthreads/NS),node 内并行不足。
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
            forward_many_nsliced(M, k, expert_ids, weights, input, output);
            return;
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
                forward_many_nsliced(M, k, expert_ids, weights, input, output, -1,
                                     small_batch_workers(NASS, inter, hidden));
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

    void forward_many_nsliced(int M, int k,
                              const uint32_t* expert_ids, const float* weights,
                              const uint16_t* input, float* output,
                              int chunk_hint = 0, size_t wlimit = 0) {
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
        static const bool _nogather = std::getenv("XIAOTU_MOE_NOGATHER") != nullptr;
        // gather 每行的分段数(轮 69 引入,4 段 = 144 个 2 KB job)。轮 72 起可用
        // XIAOTU_MOE_GCHUNK 覆盖,便于零重建扫描(=1 表示每行一个 job,即轮 67 的形态)。
        const size_t kGChunk = [] {
            const char* e = std::getenv("XIAOTU_MOE_GCHUNK");
            const long v = e ? std::atol(e) : 0;
            return (size_t)(v > 0 ? v : 4);
        }();
        for (int e : active_) {
            ExpBuf& g = exp_[e];
            size_t me = g.ai_list.size();
            // 【轮 71】改成"只增不减"的 resize:原来 `resize(need)` 在路由变化使 me 在
            // 2/3 间振荡时会先 shrink 再 grow,而**每次 grow 都零填充新元素**,实测稳定
            // 花 24-34 µs/次(每层 ~4%)。现在 size 永远保持在历史最大值 ⇒ grow 只发生
            // 一次,之后不再零填充。与轮 70 的 reserve 方案(R58,直接算错)的区别:
            // 这里 `size()` 始终 ≥ need,零填充语义也完全不变,不依赖任何隐藏 size() 行为。
            const size_t nx = me * (size_t)hidden;
            const size_t n2 = me * (size_t)2 * (size_t)inter;
            const size_t ni = me * (size_t)inter;
            if (g.xg.size()    < nx) g.xg.resize(nx);
            if (g.both.size()  < n2) g.both.resize(n2);
            if (g.act.size()   < ni) g.act.resize(ni);
            if (g.abf16.size() < ni) g.abf16.resize(ni);
            if (g.down.size()  < nx) g.down.resize(nx);
        }
        // 轮 67【并行 gather】:原来是**串行 for**(见 NOTES §113)。12 个专家 x me=3 行
        // x hidden*2=8KB = 288 KB 的单线程 memcpy,把整段 DRAM 延迟暴露在关键路径上,
        // 同一时刻其余 ~119 个线程在下一个 barrier 上空等。实测 [setup-prof]
        // gather=267 us/层 = 每层总时间(1.19 ms)的 21%,而 A 相本身能跑到 210 GB/s,
        // 说明这段是纯延迟、不是带宽。改成按 assignment 并行(每个 job 搬一行 8 KB,
        // 共 NASS=BS*K 个 job),用 inst_idx_ 定位该 assignment 在专家内的行号,
        // 与串行版**逐字节等价**(同一地址、同一长度、同一数据)。
        auto _suB2 = std::chrono::steady_clock::now();
        if (!_nogather) {
            // 对照开关:XIAOTU_MOE_SERIAL_GATHER=1 回到旧的串行 gather(逐字节同结果),
            // 用于在同一 session 内做串行 vs 并行的真实 A/B(避免跨窗口漂移)。
            static const bool _serialg = std::getenv("XIAOTU_MOE_SERIAL_GATHER") != nullptr;
            static const bool _noopg = std::getenv("XIAOTU_MOE_NOOP_GATHER") != nullptr;
            if (_noopg) {
                // 计时诊断:保留 pfor 派发、body 为空 ⇒ 量出"并行区派发+同步"的固定开销。
                // 若此值 ≈ gather 实测值,则引擎的时间不是花在访存上而是花在同步上。
                // 值 N>1 时重复 N 次:用于判定该开销是"每区一次"还是"每次调用固定"。
                static const long _np = std::max<long>(1, std::atol(std::getenv("XIAOTU_MOE_NOOP_GATHER")));
                for (long _i = 0; _i < _np; ++_i) pfor(NASS, [&](size_t) {});
            } else if (_serialg) {
                for (int e : active_) {
                    ExpBuf& g = exp_[e];
                    const size_t me = g.ai_list.size();
                    for (size_t m = 0; m < me; ++m) {
                        const size_t t = g.ai_list[m] / (size_t)k;
                        std::memcpy(g.xg.data() + m * (size_t)hidden, input + t * (size_t)hidden,
                                    (size_t)hidden * sizeof(uint16_t));
                    }
                }
            } else
            pfor(NASS * kGChunk, [&](size_t j) {   // 计时诊断:XIAOTU_MOE_NOGATHER=1 跳过(数值无效)
                const size_t ai = j / kGChunk;
                const size_t c = j % kGChunk;
                const uint32_t eid = expert_ids[ai];
                if (eid >= (uint32_t)nel || weights[ai] == 0.f) return;
                const size_t m = inst_idx_[ai];
                const size_t t = ai / (size_t)k;
                // 【轮 69】每行切成 kGChunk 段:原来每行一个 job(8 KB),只有 NASS=36 个
                // 线程参与,单线程串行搬 8 KB 且源是冷 DRAM ⇒ 实测 ~110 µs(2.4 GB/s,
                // 纯延迟、不是带宽)。切成 4×2 KB 后 144 个 job 铺满 120 线程,
                // 把 DRAM 访问摊开到更多并行流上。**字节级等价**(同一区间、同一数据)。
                const size_t off = c * ((size_t)hidden / kGChunk);
                const size_t len = ((c + 1) == kGChunk) ? ((size_t)hidden - off)
                                                        : ((size_t)hidden / kGChunk);
                std::memcpy(exp_[eid].xg.data() + m * (size_t)hidden + off,
                            input + t * (size_t)hidden + off, len * sizeof(uint16_t));
            });
        }
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

        // Sub-split for the SHARDED weight-read phases (A and B). Unlike the flat
        // path (nc_gu/nc_d chunks), the sharded path launches exactly `na` jobs per
        // node -- na*NS total (e.g. 6*8=48) -- so each node's ~nthreads/NS workers
        // (168/8=21) have only 6 tickets and ~15 of them sit idle during the
        // bandwidth-critical w13/w2 reads. Sub-splitting each node's row span into
        // ~threads-per-node tickets engages ALL worker threads on its node-local
        // shard. Env XIAOTU_MOE_SHARDSPLIT=N forces sub; =0 disables (A/B toggle);
        // default picks ceil(tpn/na) capped to keep >=32 rows/job.
        int subA = 1, subB = 1;
        {
            const char* eov = std::getenv("XIAOTU_MOE_SHARDSPLIT");
            if (eov && std::atoi(eov) > 0) { subA = std::atoi(eov); subB = subA; }
            else if (nshard_ >= 2 && pool_.nthreads() > 1 && (!eov || std::atoi(eov) < 0)) {
                const int NS = nshard_;
                size_t tpn = std::max<size_t>(1, pool_.nthreads() / (size_t)NS);
                // 每 worker 约 4 个 job(原来 ceil(tpn/na) 在 na≈tpn 时只有 1 个
                // job/worker ⇒ 一个掉队 job 就拖慢整个阶段:实测 auto(=1) 时
                // A+B=1448-1469 µs、subA=2→1217、4→1166-1199、8→1174-1236,
                // 即 1.2x,每核带宽 1.15→1.44 GB/s)。仍受 spanA/32 的上限约束。
                size_t need = (4 * tpn + na - 1) / na;                // ≈ 4 jobs/worker
                if (need < 2) need = 2;
                size_t spanA = (size_t)(inter / NS);
                subA = (int)std::min<size_t>(need, std::max<size_t>(1, spanA / 32));
                if (subA < 1) subA = 1;
                size_t spanB = (size_t)(hidden / NS);
                subB = (int)std::min<size_t>(need, std::max<size_t>(1, spanB / 32));
                if (subB < 1) subB = 1;
            }
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

        using clk = std::chrono::steady_clock;
        auto pA0 = clk::now();
        // Phase A: batched gate+up slices. Sharded: each NUMA node computes the
        // rows it owns ([n*I/NS,(n+1)*I/NS)) for EVERY active expert, reading its
        // node-local shard (na jobs per node). Otherwise the legacy flat path
        // splits (expert, inter-chunk) and reads the worker's socket replica.
        if (nshard_ >= 2) {
            const size_t NS = (size_t)nshard_;
            std::vector<size_t> jc(NS, (size_t)active_.size() * (size_t)subA);
            pfor_sharded((int)NS, jc.data(), [&](size_t n, size_t job) {
                size_t e_idx = job / (size_t)subA;
                size_t s = job % (size_t)subA;
                if (e_idx >= active_.size()) return;
                int eid = active_[e_idx];
                ExpBuf& g = exp_[eid];
                const size_t me = g.ai_list.size();
                if (me == 0) return;
                int n0 = (int)(n * inter / NS);
                int n1 = (int)((n + 1) * inter / NS);
                if (n1 > inter) n1 = inter;
                if (n0 >= n1) return;
                if (subA > 1) {           // sub-split node span across node's threads
                    int sep = (n1 - n0 + subA - 1) / subA;
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
                wt::gate_up_slice_batched((int)me, g.xg.data(), w13_shard_[n], w13_g_, w13_gs_,
                                          g.both.data(), inter, hidden, (size_t)eid, groupN, groupK, n0, n1);
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
                wt::gate_up_slice_batched((int)me, g.xg.data(), w13_for(s), w13g_for(s), w13_gs_,
                                          g.both.data(), inter, hidden, (size_t)eid, groupN, groupK, n0, n1);
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
            std::vector<size_t> jc(NS, (size_t)active_.size() * (size_t)subB);
            pfor_sharded((int)NS, jc.data(), [&](size_t n, size_t job) {
                size_t e_idx = job / (size_t)subB;
                size_t s = job % (size_t)subB;
                if (e_idx >= active_.size()) return;
                int eid = active_[e_idx];
                ExpBuf& g = exp_[eid];
                const size_t me = g.ai_list.size();
                if (me == 0) return;
                int n0 = (int)(n * hidden / NS);
                int n1 = (int)((n + 1) * hidden / NS);
                if (n1 > hidden) n1 = hidden;
                if (n0 >= n1) return;
                if (subB > 1) {           // sub-split node span across node's threads
                    int sep = (n1 - n0 + subB - 1) / subB;
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
                                       g.down.data(), hidden, inter, (size_t)eid, groupN, groupK, n0, n1);
            });
        } else {
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
    // & down rows [n*H/NS,(n+1)*H/NS). Each node's region uses the FULL-layout stride
    // (per-expert offset UNCHANGED -> existing kernel path works with rowshift==0),
    // mmap'd and MPOL_BIND to that node, but only its OWNED rows are faulted in:
    // unowned rows have no physical backing and are never read, so physical RSS is
    // ONE full copy across all nodes. Every weight read is from node-local pages
    // (no cross-node traffic): each layer's weights are read exactly ONCE over the
    // whole machine (vs 2x with per-socket replication). Scales stay as one full
    // copy (tiny, indexed by absolute row).
    int nshard_ = 0;
    std::vector<void*> shard_owned_;
    std::vector<const uint8_t*> w13_shard_;
    std::vector<const uint8_t*> w2_shard_;

    // Lightweight per-phase timing (env-gated print). Accumulates wall time of
    // the 5 phases across calls; prints a breakdown every prof_every_ calls.
    bool prof_ = false;
    size_t prof_every_ = 40;
    size_t prof_calls_ = 0;
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
        prof_C_ += dC; prof_ovh_ += dovh; prof_M_ += M; prof_na_ += na; ++prof_calls_;
        if (prof_calls_ % prof_every_ == 0) {
            double S = (double)(prof_A_ + prof_A2_ + prof_B0_ + prof_B_ + prof_C_ + prof_ovh_) / 1e6;
            double navg = (double)prof_na_ / (double)prof_calls_;
            double mavg = (double)prof_M_ / (double)prof_calls_;
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
            prof_A_=prof_A2_=prof_B0_=prof_B_=prof_C_=prof_ovh_=prof_M_=prof_na_=0;
        }
    }

    // Fill one full-stride, MPOL_BIND-to-node `node` region of `total` bytes with
    // this node's owned rows copied from `src`. `copier(d,s,per_eid_bytes)` copies
    // the node's (stride-located) rows into the local region; writing faults the
    // pages in on `node`, so every backed page is physically local and the total
    // physical RSS across all nodes equals one full copy.
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
        long rc = syscall(SYS_mbind, p, total, MPOL_BIND, &mask, sizeof(mask) * 8, 0);
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
    bool shard_fill_w13(const void* src) {
        if (!src || nshard_ < 2) return false;
        const size_t H = cfg_.hidden_size, I = cfg_.intermediate_size, E = cfg_.expert_num;
        const size_t rowbytes = H / 2;
        const size_t n2 = 2 * I;
        const size_t stride = n2 * rowbytes;
        const size_t total = stride * E;
        const int NS = nshard_;
        size_t base = shard_owned_.size();
        const uint8_t* s = static_cast<const uint8_t*>(src);
        for (int n = 0; n < NS; ++n) {
            const size_t rs = (size_t)n * I / NS, re = (size_t)(n + 1) * I / NS;
            const size_t cbytes = (re - rs) * rowbytes;
            void* p = shard_region(total, n, s, [&](uint8_t* d, const uint8_t* srcx, size_t) {
                for (size_t e = 0; e < E; ++e) {
                    const size_t eb = e * stride;
                    std::memcpy(d + eb + rs * rowbytes, srcx + eb + rs * rowbytes, cbytes);         // gate
                    std::memcpy(d + eb + (I + rs) * rowbytes, srcx + eb + (I + rs) * rowbytes, cbytes); // up
                }
            });
            if (!p) { while (shard_owned_.size() > base) { munmap(shard_owned_.back(), 0); shard_owned_.pop_back(); } return false; }
            w13_shard_[n] = static_cast<const uint8_t*>(p);
            shard_owned_.push_back(p);
        }
        if (std::getenv("XIAOTU_MOE_SHARD_DIAG") != nullptr)
            fprintf(stderr, "[SHARD-DIAG] w13 sharding OK NS=%d total=%.1fGiB\n", NS, (double)total/(1ULL<<30));
        return true;
    }

    // Shard the down block [E][H][I/2] across nshard_ nodes: node n owns rows
    // [n*H/NS,(n+1)*H/NS).
    bool shard_fill_w2(const void* src) {
        if (!src || nshard_ < 2) return false;
        const size_t H = cfg_.hidden_size, I = cfg_.intermediate_size, E = cfg_.expert_num;
        const size_t rowbytes = I / 2;
        const size_t stride = H * rowbytes;
        const size_t total = stride * E;
        const int NS = nshard_;
        size_t base = shard_owned_.size();
        const uint8_t* s = static_cast<const uint8_t*>(src);
        for (int n = 0; n < NS; ++n) {
            const size_t rs = (size_t)n * H / NS, re = (size_t)(n + 1) * H / NS;
            const size_t cbytes = (re - rs) * rowbytes;
            void* p = shard_region(total, n, s, [&](uint8_t* d, const uint8_t* srcx, size_t) {
                for (size_t e = 0; e < E; ++e) {
                    const size_t eb = e * stride;
                    std::memcpy(d + eb + rs * rowbytes, srcx + eb + rs * rowbytes, cbytes);
                }
            });
            if (!p) { while (shard_owned_.size() > base) { munmap(shard_owned_.back(), 0); shard_owned_.pop_back(); } return false; }
            w2_shard_[n] = static_cast<const uint8_t*>(p);
            shard_owned_.push_back(p);
        }
        if (std::getenv("XIAOTU_MOE_SHARD_DIAG") != nullptr)
            fprintf(stderr, "[SHARD-DIAG] w2 sharding OK NS=%d total=%.1fGiB\n", NS, (double)total/(1ULL<<30));
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

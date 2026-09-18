// xiaotu-moe Python binding (pybind11)
//
// Exposes the MOE_V2 family plus MOEConfigV2 with a stable constructor ABI:
// every engine takes
// (cfg, w13_weight, w2_weight, w13_scale=0, w2_scale=0, w13_global_scale=0,
//  w2_global_scale=0) where the weight/scale args are integer data pointers
// (numpy arrays are also accepted for convenience; their data() is used).
//
// Phase 1 ships BF16 (MOE_BF16 / MOE_FP16); Phase 2 adds FP8
// (MOE_FP8 / MOE_FP8_FP16) then WNA16 / MXFP4. Quantized traits each implement
// gate_up/down and never touch the orchestration loop.
//
// License: Apache-2.0.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <csignal>
#include <cstdio>
#include <cstring>
#include <execinfo.h>
#define _GNU_SOURCE
#include <ucontext.h>
#include <sys/ucontext.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cuda.h>   // 驱动 API:cuStreamWriteValue32 / cuStreamWaitValue32(item 1/3)
#if defined(__AVX512F__)
#include <immintrin.h>
#endif
#include <algorithm>
#include <atomic>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "../moe/moe_v2.hpp"
#include "../moe/moe_v2_fp8.hpp"
#include "../moe/moe_v2_packed4.hpp"

namespace py = pybind11;
using namespace xiaotu_moe;

// ---- capture-safe cpu_decode (mirrors lk_moe) ------------------------------
// lk_moe's cpu_decode(stream_ptr, ...) is CUDA-graph-capture-safe: it reads the
// GPU input tensors with async host-to-device-pinned memcpy nodes, runs the CPU
// MoE compute inside a CUDA host-function node (cudaLaunchHostFunc), and writes
// the result back to a stable device buffer with an async memcpy node. During
// stream capture these become recorded graph nodes; at each graph replay the
// host callback re-runs the CPU compute on the *current* input data and issues
// the write-back. xiaotu-moe is a pure-CPU engine, so it exact-mirrors this:
// GPU -> pinned host, forward_many on CPU, pinned host -> device out buffer.
//
// A single per-engine CpuDecodeState owns the persistent pinned buffers (sized
// to the largest qlen seen) plus a reusable host-function context. Buffers are
// deliberately per-engine: each MoE layer has its own RoutedExperts/lk_moe, so
// there is no cross-layer aliasing, and decode is serialized per engine.
//
// Requires only host-side CUDA runtime API (no device kernels), so the whole
// module still builds with a plain host compiler once -lcudart is linked.
// 【§582】**默认改为 sync(即 `XIAOTU_MOE_ASYNC` 未设时不走 async)**;要 async 请显式 `=1`。
//
// 为什么翻默认:第 221 轮时 async 确实是最大单点收益(V 6.53→1.80 ms/token、
// C=4 聚合 71.8→107.3 t/s),但**后续实测推翻了它作为默认的资格**(NOTES §517b/§517c):
//   * qlen=1 / TP=1 每层:sync **1.05 ms** vs async **2.05 ms**(引擎段 0.44 vs 1.31)⇒ **async 慢 2×**;
//   * 且 async 跑里观察到"每步把整段 prompt 重算一遍"的病态(qlen=26 而非 1,ShareGPT 掉到 2.27 tok/s)。
// 我们的 `serve_v41.sh` / probe 一直显式钉 `ASYNC=0`,所以**生产路径从未被打中**;
// 但"默认 true"对任何不设该 env 的启动者都是颗地雷 ⇒ 现在把默认值改成与实测最优一致(fail-safe)。
// async 路径保留(高并发下曾显示收益),作为显式 opt-in 继续存在。
static bool xiaotu_async_enabled() {
    static const bool e = [] {
        const char* v = std::getenv("XIAOTU_MOE_ASYNC");
        return v && std::atoi(v) != 0;      // 只有显式 =1 才走 async
    }();
    return e;
}

struct CpuDecodeState {
    void* pin_hidden = nullptr;
    void* pin_ids = nullptr;
    void* pin_weights = nullptr;
    void* pin_out = nullptr;
    size_t cap_hidden = 0, cap_ids = 0, cap_weights = 0, cap_out = 0;

    cudaStream_t stream = nullptr;
    const void* engine = nullptr;      // MOE* (opaque; typed in the host fn)
    const uint16_t* hid = nullptr;     // pinned D2H destination (bf16)
    const uint32_t* ids = nullptr;     // pinned D2H destination (int32)
    const float* wts = nullptr;        // pinned D2H destination (fp32)
    float* out = nullptr;              // pinned compute output (fp32)
    float* outg = nullptr;             // device H2D destination (fp32)
    int qlen = 0, k = 0;
    size_t out_bytes = 0;
    void (*host_fn)(void*) = nullptr;

    // 捕获期间不能 free 的旧 pinned 缓冲:graph 节点可能仍引用它们。
    std::vector<void*> retired;

    // 返回 true 表示发生了重新分配(旧 pinned 指针失效)。
    // retire=true(CUDA graph capture 期间)时只解除引用、不 cudaFreeHost ——
    // 捕获中既不允许 cudaStreamSynchronize,也不允许释放被 graph 节点引用的内存。
    // ---- 异步握手(item 1/3,XIAOTU_MOE_ASYNC=1):用"mapped flag + 流内存操作"
    // 取代 cudaLaunchHostFunc —— host-func 会阻塞整条流并由驱动派发回调,实测每层 36 µs。
    void* pin_flags = nullptr;
    volatile uint32_t* hin = nullptr;    // GPU 写 1(输入就绪)/ 0(槽位归还);CPU 轮询
    volatile uint32_t* hout = nullptr;   // CPU 写 1(结果就绪)/ 0(复位);GPU 等
    CUdeviceptr din = 0, dout = 0;
    bool async_ready = false;
    int w_qlen = 0, w_k = 0;
    const uint16_t* w_hid = nullptr;
    const uint32_t* w_ids = nullptr;
    const float* w_wts = nullptr;
    float* w_out = nullptr;
    void* w_engine = nullptr;
    void (*w_fn)(void*, int, int, const uint16_t*, const uint32_t*, const float*, float*) = nullptr;
    void* out_ptr_seen = nullptr;   // out_gpu 指针缓存(避免每次调用做 Python shape 查询)
    size_t out_rows = 0;

    // 【必须】mapped flag 的**分配**放在构造期,不能在 cpu_decode 里惰性分配:
    // 服务里第一次 cpu_decode 很可能发生在 **CUDA graph 捕获区**内
    // (`gpu_model_runner.py:6481 Profiling CUDA graph memory`),
    // 而捕获期 `cudaHostAlloc` 会让整段 capture 作废
    // (实测:`cudaErrorStreamCaptureInvalidated`)。
    CpuDecodeState() {
        if (!xiaotu_async_enabled()) return;
        void* p = nullptr;
        if (cudaHostAlloc(&p, 256, cudaHostAllocMapped) != cudaSuccess) return;
        std::memset(p, 0, 256);
        pin_flags = p;
        hin = (volatile uint32_t*)p;
        hout = hin + 32;                    // 不同 cache line,避免伪共享
        void* dp = nullptr;
        if (cudaHostGetDevicePointer(&dp, (void*)hin, 0) == cudaSuccess)
            din = (CUdeviceptr)dp;
        if (cudaHostGetDevicePointer(&dp, (void*)hout, 0) == cudaSuccess)
            dout = (CUdeviceptr)dp;
    }

    bool ensure_buffers(size_t nh, size_t ni, size_t nw, size_t no, bool retire = false) {
        bool realloc = false;
        auto grow = [&](void*& p, size_t& cap, size_t need) {
            if (need <= cap) return;
            if (p) {
                if (retire) retired.push_back(p);
                else cudaFreeHost(p);
            }
            p = nullptr;
            cudaHostAlloc(&p, need, cudaHostAllocDefault);
            cap = need;
            realloc = true;
        };
        grow(pin_hidden, cap_hidden, nh);
        grow(pin_ids, cap_ids, ni);
        grow(pin_weights, cap_weights, nw);
        grow(pin_out, cap_out, no);
        return realloc;
    }

    ~CpuDecodeState() {
        if (pin_hidden) cudaFreeHost(pin_hidden);
        if (pin_ids) cudaFreeHost(pin_ids);
        if (pin_weights) cudaFreeHost(pin_weights);
        if (pin_out) cudaFreeHost(pin_out);
        if (pin_flags) cudaFreeHost(pin_flags);
        for (void* p : retired) cudaFreeHost(p);
    }
};

// ---- EP(专家并行)跨 rank 归约:走共享内存,不进 GPU -----------------------
// 背景:TP=2 时每个 rank 只算自己那半专家,部分和必须相加。用 GPU 的 NCCL
// all-reduce 是每层一次跨卡集合通信(实测每层 +4~6.5 ms);而两个 rank 在**同一台
// 机器**上,完全可以用 /dev/shm 直接交换 —— 一次 memcpy + 两个自旋 barrier,
// 量级是几十 µs。参考实现(lk-moe)也是把 num_processes=ep_size 交给引擎内部合并。
//
// 布局(每个引擎/每层一个文件,由插件按层名创建):
//   [EpShmHeader(64B)][rank0 部分和: stride B][rank1 部分和: stride B]
struct EpShmHeader {
    alignas(64) std::atomic<int> arrive{0};
    std::atomic<unsigned long long> gen{0};
    alignas(64) std::atomic<int> read_done{0};
    std::atomic<unsigned long long> gen2{0};
    alignas(64) int world{1};
    int rank{0};
    unsigned long long cap_bytes{0};
};

struct EpShmState {
    EpShmHeader* hdr = nullptr;
    char* parts = nullptr;       // 指向 [rank0][rank1] 部分和区
    float* partial_copy = nullptr;  // 本 rank 的部分和暂存(malloc 的对齐缓冲)
    int world = 1, rank = 0;
    size_t capacity = 0;         // partial_copy 容量(字节)
};

std::mutex g_ep_mtx;
std::unordered_map<const void*, std::unique_ptr<EpShmState>> g_ep_state;

// ---- 自建跨 rank 归约:mkl 的编排链只给 num_processes/process_id,不调 configure_ep ----
// lk 的 vLLM 侧(`routed_experts.py:_process_mxfp4`)把
//   num_processes = tp_size, process_id = tp_rank, intermediate_size = 每卡一半
// 交给引擎,而 `cpu_decode/cpu_prefill` 的签名里**没有**归约参数 ⇒ **部分和合并由引擎
// 自己负责**(专有 lk_moe 的二进制里确实用了 `shm_open`)。我们此前只在插件显式调用
// `configure_ep()` 时才建归约,所以走 lk 链时**根本没有归约**。
// 这里按 lk 的做法自行建立:两个 rank 以**相同顺序**构造同样的引擎,故用"构造序号"
// 命名即可对齐;文件名以 `xiaotu_ep_` 开头,沿用启动脚本里的 `rm -f /dev/shm/xiaotu_ep_*.bin`。
namespace {
std::atomic<int> g_auto_ep_seq{0};

bool auto_ep_setup(const void* key, const MOEConfigV2& cfg) {
    if (cfg.num_processes <= 1) return false;
    // 【v0.2】有些调用方(主线的 mixed-mode 插件)只用 num_processes/process_id 表达
    // "本进程该占哪一半 CCD / NUMA node",而**归约由框架自己做**(专家按 I 切分 +
    // expert_map,引擎不该再叠加一层 shm 归约)。此时必须能显式关掉自建归约,
    // 否则会多算一次、结果错误。
    static const bool no_auto_ep = [] {
        const char* v = std::getenv("XIAOTU_MOE_NO_AUTO_EP");
        return v && std::atoi(v) != 0;
    }();
    if (no_auto_ep) return false;
    const int world = cfg.num_processes;
    const int rank = cfg.process_id;
    if (rank < 0 || rank >= world) return false;
    const int H = cfg.hidden_size;
    const int tokens = cfg.max_batch_size > 0 ? cfg.max_batch_size : 1;
    if (H <= 0) return false;
    const size_t stride =
        ((size_t)tokens * (size_t)H * sizeof(float) + 63) / 64 * 64;
    const size_t total = ((sizeof(EpShmHeader) + stride * (size_t)world) + 63) / 64 * 64;

    const int seq = g_auto_ep_seq.fetch_add(1);   // 两 rank 构造顺序一致 ⇒ 同名
    char name[160];
    std::snprintf(name, sizeof(name), "/xiaotu_ep_auto_%d_%d_%d_%d.bin",
                  seq, world, H, tokens);

    int fd = ::shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600);
    bool created = (fd >= 0);
    if (fd < 0) fd = ::shm_open(name, O_RDWR, 0600);
    if (fd < 0) return false;
    if (created && ::ftruncate(fd, (off_t)total) != 0) { ::close(fd); return false; }
    void* base = ::mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) return false;
    if (created) std::memset(base, 0, sizeof(EpShmHeader));

    EpShmHeader* hdr = reinterpret_cast<EpShmHeader*>(base);
    hdr->world = world;

    std::lock_guard<std::mutex> lg(g_ep_mtx);
    auto& ep = g_ep_state[key];
    if (!ep) ep = std::make_unique<EpShmState>();
    ep->hdr = hdr;
    ep->parts = reinterpret_cast<char*>(base) + sizeof(EpShmHeader);
    ep->world = world;
    ep->rank = rank;
    ep->capacity = stride;
    ep->partial_copy =
        (float*)std::aligned_alloc(64, (stride + 63) / 64 * 64);
    std::fprintf(stderr,
                 "[xiaotu/engine] auto EP shm %s: rank %d/%d stride=%zu tokens=%d\n",
                 name, rank, world, stride, tokens);
    std::fflush(stderr);
    return true;
}
}  // namespace

// Map engine pointer -> its pinned buffers / host-fn context. Keyed by the
// MOE* identity; every MOE instance across every type has a unique address, so
// sharing one map across all template instantiations is safe.
std::mutex g_cd_mtx;
// 故意"泄漏"到进程退出(`*new`):常驻 async worker 线程在**静态析构期**仍可能轮询
// 这些 state,若此时 vector/map 已析构就会 SIGSEGV(实测,harness 退出时必崩)。
// 泄漏的只是几十个 pinned 缓冲,进程退出后由 OS 回收。
std::unordered_map<const void*, std::unique_ptr<CpuDecodeState>>& g_cd_state =
    *new std::unordered_map<const void*, std::unique_ptr<CpuDecodeState>>();

// ---- 异步枢纽(item 1/3)-----------------------------------------------
// 每层一对 mapped flag:GPU 在 D2H 之后 cuStreamWriteValue32(hin,1),
// 然后 cuStreamWaitValue32(hout,1) 等 CPU 结果,最后 H2D 并写回 hin=0。
// 一个常驻 worker 轮询所有层的 hin;它**只自旋**(单线程,占 ~0.5% CPU)以保证低延迟。
static std::mutex& g_async_mtx = *new std::mutex();
static std::vector<CpuDecodeState*>& g_async_slots = *new std::vector<CpuDecodeState*>();
static std::atomic<bool> g_async_stop{false};
static std::thread g_async_thr;
static std::atomic<bool> g_async_started{false};
// worker 线程上最近一次 run_moe_and_ep 的 EP 耗时(仅诊断计时用)。
static thread_local double g_async_ep_last = 0.0;

static void async_loop() {
    // per-layer 计时(与 host-func 路径同口径;XIAOTU_CD_TIMING=1 时打印)。
    const bool timing = std::getenv("XIAOTU_CD_TIMING") != nullptr;
    const int every = [] {
        const char* e = std::getenv("XIAOTU_CD_TIMING_EVERY");
        return e ? std::atoi(e) : 43;
    }();
    double sum_compute = 0, sum_period = 0, sum_ep = 0;
    int n = 0;
    double min_period = 1e18, min_compute = 1e18, min_rest = 1e18;
    // period = **相邻两次 cpu_decode 调用**的间隔(任意层),即每次调用实际覆盖的
    // "GPU/常驻层 + 拷贝 + 握手"时间;compute = 本次调用的 CPU MoE(+EP)。
    // ⚠️ 均值会被"请求之间的预填充空档"污染 ⇒ 同时给出 **min**(稳态下界,无空档)。
    // ⚠️ 有常驻层时 1 次调用平均跨 `总层数/CPU层数` 个模型层,换算"每模型层"要再除这个比。
    auto last_cb = std::chrono::steady_clock::now();
    bool seen = false;
    while (!g_async_stop.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> lg(g_async_mtx);
        for (auto* st : g_async_slots) {
            if (st->hin && st->hin[0] == 1) {
                auto t_cb = std::chrono::steady_clock::now();
                double ep = 0.0;
                if (st->w_fn) st->w_fn(st->w_engine, st->w_qlen, st->w_k,
                                       st->w_hid, st->w_ids, st->w_wts, st->w_out);
                if (timing) {
                    // EP 时间由 run_moe_and_ep 经 thread_local 传出(worker 是单线程)。
                    ep = g_async_ep_last;
                    auto t_end = std::chrono::steady_clock::now();
                    const double compute_ms =
                        std::chrono::duration<double, std::milli>(t_end - t_cb).count();
                    if (seen) {
                        const double period_ms = std::chrono::duration<double, std::milli>(
                                                     t_cb - last_cb).count();
                        sum_compute += compute_ms;
                        sum_period += period_ms;
                        sum_ep += ep;
                        if (period_ms < min_period) min_period = period_ms;
                        if (compute_ms < min_compute) min_compute = compute_ms;
                        // 【第 221 轮】rest 必须按**同一次采样**算,不能用
                        // min(period) − min(compute)(两者来自不同样本,会高估 rest)。
                        if (period_ms - compute_ms < min_rest) min_rest = period_ms - compute_ms;
                        if (++n % every == 0) {
                            fprintf(stderr,
                                    "[cd-timing/async] calls=%d qlen=%d k=%d "
                                    "period=%.3fms compute=%.3fms(engine=%.3f ep=%.3f) "
                                    "rest=%.3fms | MIN period=%.3fms compute=%.3fms "
                                    "rest=%.3fms\n",
                                    every, st->w_qlen, st->w_k,
                                    sum_period / every, sum_compute / every,
                                    (sum_compute - sum_ep) / every, sum_ep / every,
                                    (sum_period - sum_compute) / every,
                                    min_period, min_compute, min_rest);
                            fflush(stderr);
                            sum_compute = sum_period = sum_ep = 0.0;
                            min_period = min_compute = min_rest = 1e18;
                        }
                    }
                    seen = true;
                    last_cb = t_cb;
                }
                st->hout[0] = 1;                       // 通知 GPU:结果已写好
                while (st->hin[0] == 1 && !g_async_stop.load(std::memory_order_relaxed))
                    _mm_pause();                       // 等 GPU 归还槽位(hin=0)
                st->hout[0] = 0;                       // 复位,供下一轮 replay
            }
        }
    }
}

static bool async_enabled() { return xiaotu_async_enabled(); }

static bool async_init(CpuDecodeState* st) {
    if (st->async_ready) return true;
    if (!async_enabled()) return false;
    // flag 已在构造期分配(捕获期不能 cudaHostAlloc,见 CpuDecodeState 构造函数)。
    if (!st->pin_flags || !st->din || !st->dout) return false;
    {
        std::lock_guard<std::mutex> lg(g_async_mtx);
        g_async_slots.push_back(st);
    }
    bool expected = false;
    if (g_async_started.compare_exchange_strong(expected, true)) {
        g_async_thr = std::thread(async_loop);
        g_async_thr.detach();   // 常驻 worker:进程退出时由 g_async_stop 结束;
                                // 不 detach 会在退出时 terminate(joinable 析构)。
    }
    st->async_ready = true;
    return true;
}


// 【第 217 轮】"CPU MoE + EP 归约"抽成独立函数:host-func 路径与异步握手
// 路径(worker 线程)共用同一份实现,避免两份代码漂移。
template <typename MOET>
static void run_moe_and_ep(MOET* engine, int qlen, int k, const uint16_t* hid,
                           const uint32_t* ids, const float* wts, float* out,
                           double* ep_ms_out, double* fwd_ms_out = nullptr) {
        static const bool fake_cpu = std::getenv("XIAOTU_MOE_FAKE_CPU") != nullptr;
        // 【§619b】把 `forward_many` 自己的耗时单独量出来。`[cd-timing] compute` 包住的是
        // `run_moe_and_ep`(= forward_many + EP 归约),而 `[NS-PROF] TOTAL` 只包 forward_many;
        // 两者实测差 108 ms/层(qlen=1905)却无从归属 ⇒ 直接打点,别再靠推断。
        auto t_f0 = std::chrono::steady_clock::now();
        if (!fake_cpu) engine->forward_many(qlen, k, ids, wts, hid, out);
        auto t_f1 = std::chrono::steady_clock::now();
        if (fwd_ms_out)
            *fwd_ms_out = std::chrono::duration<double, std::milli>(t_f1 - t_f0).count();
        // ---- EP:把本 rank 的部分和与对端相加(共享内存,不进 GPU) ----
        {
            EpShmState* ep = nullptr;
            {
                std::lock_guard<std::mutex> lg(g_ep_mtx);
                auto it = g_ep_state.find(engine);
                if (it != g_ep_state.end()) ep = it->second.get();
            }
            auto t_ep0 = std::chrono::steady_clock::now();
            if (ep && ep->hdr && ep->world > 1) {
                const size_t bytes = (size_t)qlen * (size_t)engine->config().hidden_size
                                     * sizeof(float);
                if (bytes <= ep->capacity) {
                    EpShmHeader* h = ep->hdr;
                    // 1) 写自己的部分和(写进 shm 中本 rank 的槽位)
                    std::memcpy(ep->parts + (size_t)ep->rank * ep->capacity,
                                out, bytes);
                    // 2) 到达 barrier:最后一个到达者复位计数并推进 gen
                    const unsigned long long gen =
                        h->gen.load(std::memory_order_acquire);
                    if (h->arrive.fetch_add(1, std::memory_order_acq_rel) + 1
                        == ep->world) {
                        h->arrive.store(0, std::memory_order_release);
                        h->gen.fetch_add(1, std::memory_order_release);
                    } else {
                        // 【第 132 轮】自旋退避:原为 `std::this_thread::yield()` ——
                        // 那是系统调用,且在本机 120 个 MoE 线程 + 48 OMP 线程满负荷时
                        // 很容易把本线程排到队尾(等待被放大成毫秒级)。
                        // 改为:先纯 PAUSE 自旋(无系统调用),久等才退到 nano-sleep。
                        // 每层两处 barrier ⇒ 43 层 ×2 的收益。
                        int _spin = 0;
                        while (h->gen.load(std::memory_order_acquire) == gen) {
                            if (++_spin < 8192) __builtin_ia32_pause();
                            else std::this_thread::sleep_for(
                                std::chrono::nanoseconds(200));
                        }
                    }
                    // 3) 求和到自己的 pinned out(H2D 会把它送回 GPU)
                    //    通用 world(=TP 大小,2 或 3)个部分和相加
                    const float* base = (const float*)ep->parts;
                    const size_t stride_f = ep->capacity / sizeof(float);
                    const size_t n = bytes / sizeof(float);
                    for (size_t i = 0; i < n; ++i) {
                        float acc = base[i];
                        for (int r = 1; r < ep->world; ++r)
                            acc += base[(size_t)r * stride_f + i];
                        out[i] = acc;
                    }
                    // 4) 读完成 barrier:确保双方都读完再允许下一轮覆写
                    const unsigned long long g2 =
                        h->gen2.load(std::memory_order_acquire);
                    if (h->read_done.fetch_add(1, std::memory_order_acq_rel) + 1
                        == ep->world) {
                        h->read_done.store(0, std::memory_order_release);
                        h->gen2.fetch_add(1, std::memory_order_release);
                    } else {
                        // 【第 132 轮】同 barrier 1:PAUSE 自旋优先,久等才 nano-sleep。
                        int _spin2 = 0;
                        while (h->gen2.load(std::memory_order_acquire) == g2) {
                            if (++_spin2 < 8192) __builtin_ia32_pause();
                            else std::this_thread::sleep_for(
                                std::chrono::nanoseconds(200));
                        }
                    }
                }
            }
            auto t_ep1 = std::chrono::steady_clock::now();
            *ep_ms_out = std::chrono::duration<double, std::milli>(t_ep1 - t_ep0).count();
        }
}



// ---- 解码 pinned 缓冲的"捕获前预分配"(2026-09-13,R103)--------------------
// 背景:lk 链**从不调用** prepare_decode_buffers(`routed_experts.py:1694`
// `_initialize_cuda_graph_buffers()` 只设 `cuda_graphs`/`RoutedExperts.output_gpu`),
// 而它的 `_cpu_decode`(`routed_experts.py:1708`)直接把 stream 交给引擎。
// 于是 pinned 缓冲会在**第一次 cpu_decode** 时惰性分配 —— 若那一次(或之后某次更大的
// batch)落在 CUDA graph 捕获区内,`cudaHostAlloc` 会让整段捕获作废(实测
// `cudaErrorStreamCaptureInvalidated`,见 TRIED_AND_REVERTED R103)。
// 对策:引擎一造好就按"解码 token 上限"预分配一次,之后捕获期永不再分配。
// 成本:每引擎 ≈1.5 MB 锁页内存(64 token × H=4096: hidden 0.5MB + out 1MB)。
static constexpr int kDecodeTokenFloor = 64;

static void prealloc_decode_buffers(const void* key, const MOEConfigV2& cfg) {
    const int H = cfg.hidden_size;
    const int K = cfg.top_k;
    if (H <= 0 || K <= 0) return;
    int tok = kDecodeTokenFloor;
    if (cfg.max_batch_size > 0) tok = std::min(tok, cfg.max_batch_size);
    tok = std::max(tok, 1);
    std::lock_guard<std::mutex> lg(g_cd_mtx);
    auto& st = g_cd_state[key];
    if (!st) st = std::make_unique<CpuDecodeState>();
    st->ensure_buffers((size_t)tok * H * sizeof(uint16_t),
                       (size_t)tok * K * sizeof(uint32_t),
                       (size_t)tok * K * sizeof(float),
                       (size_t)tok * H * sizeof(float), false);
}


// ---- SIGSEGV diagnostics (debug build): print faulting addr + stack ----
namespace {
struct sigaction g_prev_segv {};
volatile sig_atomic_t g_in_handler = 0;

void xt_sigsegv_handler(int signo, siginfo_t* si, void* ctx) {
    fprintf(stderr, "\n[XTSIG] SIGSEGV at faulting address %p signal %d\n",
            (void*)(si ? si->si_addr : nullptr), signo);
    ucontext_t* uc = (ucontext_t*)ctx;
    if (uc) {
        auto& g = uc->uc_mcontext.gregs;
        fprintf(stderr,
                "[XTSIG] RIP=%p RSP=%p RBP=%p\n"
                "[XTSIG] RAX=%p RBX=%p RCX=%p RDX=%p\n"
                "[XTSIG] RSI=%p RDI=%p R8=%p R9=%p R10=%p R11=%p\n"
                "[XTSIG] R12=%p R13=%p R14=%p R15=%p\n",
                (void*)g[REG_RIP], (void*)g[REG_RSP], (void*)g[REG_RBP],
                (void*)g[REG_RAX], (void*)g[REG_RBX], (void*)g[REG_RCX], (void*)g[REG_RDX],
                (void*)g[REG_RSI], (void*)g[REG_RDI], (void*)g[REG_R8], (void*)g[REG_R9],
                (void*)g[REG_R10], (void*)g[REG_R11],
                (void*)g[REG_R12], (void*)g[REG_R13], (void*)g[REG_R14], (void*)g[REG_R15]);
    }
    void* bt[48]; int n = backtrace(bt, 48);
    backtrace_symbols_fd(bt, n, 2);
    fflush(stderr);
    // restore default so a second fault is a clean abort
    if (g_prev_segv.sa_handler == SIG_DFL || g_prev_segv.sa_handler == SIG_IGN)
        std::signal(SIGSEGV, SIG_DFL);
    else if (g_prev_segv.sa_sigaction)
        sigaction(SIGSEGV, &g_prev_segv, nullptr);
    std::raise(SIGSEGV);
}

void maybe_install_sigsegv_handler() {
    struct sigaction sa {};
    sa.sa_sigaction = xt_sigsegv_handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &g_prev_segv);
}
}  // namespace

// Extract a raw const pointer from an argument that may be an integer (data_ptr),
// a numpy array (.data()), or None/0 (-> nullptr).
static const void* as_ptr(const py::object& o) {
    if (o.is_none()) return nullptr;
    if (py::isinstance<py::int_>(o)) {
        long long v = py::cast<long long>(o);
        return reinterpret_cast<const void*>(v);
    }
    if (py::isinstance<py::array>(o))
        return py::cast<py::array>(o).data();
    return nullptr;
}

// Typed pointer extractors so cpu_prefill/cpu_decode accept BOTH numpy arrays
// and the integer `data_ptr()` values the Lvllmds4-x fork passes (mirroring the
// real lk_moe binding), exactly matching the fork's _cpu_prefill/_cpu_decode.
static const uint32_t* as_u32(const py::object& o) {
    return reinterpret_cast<const uint32_t*>(as_ptr(o));
}
static const float* as_f32(const py::object& o) {
    return reinterpret_cast<const float*>(as_ptr(o));
}
static const uint16_t* as_u16(const py::object& o) {
    return reinterpret_cast<const uint16_t*>(as_ptr(o));
}
static float* as_f32m(const py::object& o) {  // mutable output
    return reinterpret_cast<float*>(const_cast<void*>(as_ptr(o)));
}

template <typename WT, typename ACT>
static void bind_moe_class(py::module& m, const char* name) {
    using MOE = MOE_V2<WT, ACT>;
    py::class_<MOE>(m, name)
        .def(py::init([](const MOEConfigV2& cfg,
                         py::object w13, py::object w2,
                         py::object w13_g, py::object w2_g,
                         py::object w13_global, py::object w2_global) {
            MOE* p = new MOE(cfg, as_ptr(w13), as_ptr(w2), as_ptr(w13_g), as_ptr(w2_g),
                             static_cast<const float*>(as_ptr(w13_global)),
                             static_cast<const float*>(as_ptr(w2_global)));
            // lk 链只给 num_processes/process_id:引擎自建跨 rank 归约(见 auto_ep_setup)
            auto_ep_setup((const void*)p, cfg);
            // 必须在任何 CUDA graph 捕获之前把解码 pinned 缓冲开好(R103)。
            prealloc_decode_buffers((const void*)p, cfg);
            return p;
        }), py::arg("cfg"), py::arg("w13_weight"), py::arg("w2_weight"),
            py::arg("w13_scale") = py::int_(0), py::arg("w2_scale") = py::int_(0),
            py::arg("w13_global_scale") = py::int_(0), py::arg("w2_global_scale") = py::int_(0))
        // 配置 EP 共享内存归约(插件在引擎建好后调用一次)。
        //   shm: /dev/shm 里 mmap 的基址(两个 rank 同一文件,必须同一虚拟地址首选,
        //        否则用相对偏移即可 —— 这里要求调用方传入**相同的映射偏移**语义:
        //        我们只使用 (char*)shm + 64 + rank*stride 的偏移,不依赖绝对地址相等)
        .def("configure_ep", [](MOE& self, int rank, int world,
                                py::object shm, unsigned long long stride) {
            intptr_t base = 0;
            if (py::isinstance<py::int_>(shm)) {
                base = py::cast<intptr_t>(shm);
            } else {
                PyObject* buf = shm.ptr();
                if (PyObject_CheckBuffer(buf)) {
                    Py_buffer view{};
                    if (PyObject_GetBuffer(buf, &view, PyBUF_SIMPLE) == 0) {
                        base = reinterpret_cast<intptr_t>(view.buf);
                        PyBuffer_Release(&view);
                    }
                }
            }
            if (base == 0 || world < 2 || rank < 0 || rank >= world) return;
            std::lock_guard<std::mutex> lg(g_ep_mtx);
            auto& ep = g_ep_state[&self];
            if (!ep) ep = std::make_unique<EpShmState>();
            ep->hdr = reinterpret_cast<EpShmHeader*>(base);
            ep->parts = reinterpret_cast<char*>(base) + sizeof(EpShmHeader);
            ep->world = world;
            ep->rank = rank;
            ep->capacity = (size_t)stride;
            ep->hdr->world = world;
            ep->partial_copy =
                (float*)std::aligned_alloc(64, ((size_t)stride + 63) / 64 * 64);
        }, py::arg("rank"), py::arg("world"), py::arg("shm"), py::arg("stride"))
        .def("ep_enabled", [](MOE& self) {
            std::lock_guard<std::mutex> lg(g_ep_mtx);
            auto it = g_ep_state.find(&self);
            return it != g_ep_state.end() && it->second && it->second->world > 1;
        })
        // 显式预分配 pinned 缓冲(必须在 CUDA graph capture **之前**调用)。
        // 捕获期间调用 cudaHostAlloc/cudaFreeHost 都会让 capture 失效
        // (实测:cudaErrorStreamCaptureInvalidated → PyTorch CUDACachingAllocator
        //  断言失败、引擎初始化直接失败)。所以插件在引擎建好后就用
        // 捕获尺寸上限预分配一次,之后稳态永不扩容。
        .def("prepare_decode_buffers", [](MOE& self, int max_qlen, int top_k) {
            const int H = self.config().hidden_size;
            if (max_qlen <= 0 || top_k <= 0 || H <= 0) return;
            std::lock_guard<std::mutex> lg(g_cd_mtx);
            auto& st = g_cd_state[&self];
            if (!st) st = std::make_unique<CpuDecodeState>();
            st->ensure_buffers((size_t)max_qlen * H * sizeof(uint16_t),
                               (size_t)max_qlen * top_k * sizeof(uint32_t),
                               (size_t)max_qlen * top_k * sizeof(float),
                               (size_t)max_qlen * H * sizeof(float), false);
        }, py::arg("max_qlen"), py::arg("top_k"))
        .def("cpu_decode", [](MOE& self,
                              py::object stream_obj,
                              int qlen, int top_k,
                              py::object hidden, py::object expert_ids,
                              py::object weights, py::object out_gpu) {
            // Capture-safe API mirroring lk_moe.cpu_decode(stream_ptr, ...).
            //   hidden:  [qlen, hidden] bf16 (uint16 storage) DEVICE
            //   expert_ids/weights: [qlen, top_k] int32 / fp32 DEVICE
            //   out_gpu: [qlen, hidden] fp32 DEVICE (stable graph buffer)
            // CPU MoE runs as a CUDA host-function node: async D2H copies are
            // recorded into the graph (or just stream-enqueued when eager), the
            // host callback computes on pinned CPU buffers and writes back with
            // an async H2D memcpy into out_gpu.
            const int H = self.config().hidden_size;
            if (qlen <= 0 || top_k <= 0 || H <= 0) return;
            // 【诊断专用】XIAOTU_MOE_FAKE_ALL=1:整个 cpu_decode 立即返回(不做 D2H/
            // host-func/H2D)⇒ 服务里量出的就是"纯 GPU 模型"每 token 时间。
            // 与 FAKE_CPU(只跳计算、仍做拷贝)相减 = 拷贝+host-func 的真实成本。
            static const bool fake_all = std::getenv("XIAOTU_MOE_FAKE_ALL") != nullptr;
            if (fake_all) return;

            cudaStream_t s = nullptr;
            if (!stream_obj.is_none()) {
                s = reinterpret_cast<cudaStream_t>(
                        py::cast<long long>(stream_obj));
            }
            if (!s) s = 0;  // default stream fallback

            const auto* hid_dev = as_u16(hidden);
            const auto* ids_dev = as_u32(expert_ids);
            const auto* wts_dev = as_f32(weights);
            float* outg_dev = as_f32m(out_gpu);

            // lk 的 `RoutedExperts.output_gpu` 是**全层共享**的 `(max_num_seqs, hidden)`
            // 缓冲(`routed_experts.py:1700`);投机解码时一步要验证的 token 数可达
            // `num_seqs × (1 + num_spec_tokens)` > `max_num_seqs` ⇒ 直接写会**越界写显存**。
            // 这里显式检查:宁可少算这层并大声报错,也不要静默写坏设备内存。
            // 每次调用都做 Python 属性查找太贵(每 token 43 次)⇒ 只在 out 指针
            // 变化时才查一次形状,结果缓存在 state 里。
            try {
                void* op = (void*)outg_dev;
                size_t rows = 0;
                {
                    std::lock_guard<std::mutex> lg(g_cd_mtx);
                    auto& stx = g_cd_state[&self];
                    if (!stx) stx = std::make_unique<CpuDecodeState>();
                    if (stx->out_ptr_seen == op) {
                        rows = stx->out_rows;
                    } else {
                        py::tuple shp = out_gpu.attr("shape");
                        rows = shp.size() > 0 ? py::cast<size_t>(shp[0]) : 0;
                        stx->out_ptr_seen = op;
                        stx->out_rows = rows;
                    }
                }
                if (rows < (size_t)qlen) {
                    fprintf(stderr,
                            "[cd] ERROR: out_gpu rows=%zu < qlen=%d (lk 的 output_gpu 按 "
                            "max_num_seqs 开;投机解码下需按 token 数开)-> skip layer\n",
                            rows, qlen);
                    return;
                }
            } catch (...) {
                // 形状拿不到就不拦(旧调用方可能传入裸指针包装)
            }

            const size_t nh = (size_t)qlen * H * sizeof(uint16_t);
            const size_t ni = (size_t)qlen * top_k * sizeof(uint32_t);
            const size_t nw = (size_t)qlen * top_k * sizeof(float);
            const size_t no = (size_t)qlen * H * sizeof(float);

            // 先取 capture 状态:捕获期间 cudaStreamSynchronize 会返回
            // "operation not permitted when stream is capturing"(实测会让引擎初始化失败),
            // 且不允许释放被 graph 节点引用的 pinned 缓冲。
            cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing(s, &cap_status);
            const bool capturing = (cap_status != cudaStreamCaptureStatusNone);

            std::lock_guard<std::mutex> lg(g_cd_mtx);
            auto& st = g_cd_state[&self];
            if (!st) st = std::make_unique<CpuDecodeState>();
            if (st->ensure_buffers(nh, ni, nw, no, capturing)) {
                if (capturing) {
                    // 预分配(kDecodeTokenFloor)之后不该再走到这里;真走到说明解码
                    // batch 超过了预分配上限。**绝不能在捕获区里 cudaHostAlloc**
                    // (会让整段 capture 作废):明确报错并跳过本次写入,而不是写越界。
                    fprintf(stderr,
                            "[cd] ERROR: pinned decode buffers too small during CUDA graph "
                            "capture (qlen=%d > prealloc %d tokens); skipping this layer's "
                            "CPU decode. Raise kDecodeTokenFloor.\n",
                            qlen, kDecodeTokenFloor);
                    return;
                }
                // 重新分配了 pinned 缓冲:旧的缓冲可能还有 pending 的回调在读,
                // 先排空该 stream 再继续(只在缓冲区增长时发生,稳态下不会触发)。
                cudaError_t es = cudaStreamSynchronize(s);
                if (es != cudaSuccess)
                    fprintf(stderr, "[cd] streamSync err=%s\n", cudaGetErrorString(es));
            }

            st->stream = s;
            st->out_bytes = no;
            st->out = (float*)st->pin_out;   // pinned 计算输出(H2D 源)
            st->outg = outg_dev;             // 设备 H2D 目标

            // 【诊断】XIAOTU_MOE_FAKE_COPY=1:跳过 D2H/H2D 拷贝但保留 host-func 与 CPU 计算
            // ⇒ 与 FAKE_CPU(保留拷贝、跳过计算)/FAKE_ALL(全跳过)三者相减即可把
            // marshalling 精确拆成"拷贝"与"host-func 派发"两项。
            struct CpuDecodeCall {
                MOE* engine;
                int qlen;
                int k;
                const uint16_t* hid;
                const uint32_t* ids;
                const float* wts;
                float* out;
                bool graph_owned;   // true: 由 graph 节点持有,回调不得释放
            };

            // ---- 异步握手路径(item 1/3)---------------------------------
            // 参数(指针)在捕获期就固定下来;replay 时只有 pinned 内容与 flag 变化。
            if (async_init(st.get())) {
                st->w_qlen = qlen; st->w_k = top_k;
                st->w_hid = (const uint16_t*)st->pin_hidden;
                st->w_ids = (const uint32_t*)st->pin_ids;
                st->w_wts = (const float*)st->pin_weights;
                st->w_out = (float*)st->pin_out;
                st->w_engine = (void*)&self;
                if (!st->w_fn) {
                    st->w_fn = [](void* e, int q, int k, const uint16_t* h,
                                  const uint32_t* i, const float* w, float* o) {
                        double ep = 0.0;
                        run_moe_and_ep<MOE>((MOE*)e, q, k, h, i, w, o, &ep);
                        g_async_ep_last = ep;
                    };
                }
                CUstream cs = (CUstream)s;
                st->outg = outg_dev; st->out_bytes = no;
                st->out = (float*)st->pin_out;
                // D2H 拷贝(与旧路径一致)
                cudaMemcpyAsync(st->pin_hidden, hid_dev, nh, cudaMemcpyDeviceToHost, s);
                cudaMemcpyAsync(st->pin_ids, ids_dev, ni, cudaMemcpyDeviceToHost, s);
                cudaMemcpyAsync(st->pin_weights, wts_dev, nw, cudaMemcpyDeviceToHost, s);
                cuStreamWriteValue32(cs, st->din, 1, 0);
                cuStreamWaitValue32(cs, st->dout, 1, CU_STREAM_WAIT_VALUE_EQ);
                cudaMemcpyAsync(st->outg, st->out, no, cudaMemcpyHostToDevice, s);
                cuStreamWriteValue32(cs, st->din, 0, 0);
                return;
            }

            // 1) Async D2H copies on the caller stream (graph-capturable).
            cudaMemcpyAsync(st->pin_hidden, hid_dev, nh,
                            cudaMemcpyDeviceToHost, s);
            cudaMemcpyAsync(st->pin_ids, ids_dev, ni,
                            cudaMemcpyDeviceToHost, s);
            cudaMemcpyAsync(st->pin_weights, wts_dev, nw,
                            cudaMemcpyDeviceToHost, s);

            // 2) Host-function node: CPU MoE compute ONLY. A CUDA host callback
            //    may NOT itself enqueue CUDA work (that returns
            //    cudaErrorNotPermitted), so the H2D write-back is a separate
            //    stream node placed AFTER this host node. Stream ordering
            //    guarantees the H2D waits for the callback, i.e. for
            //    forward_many to finish writing st->out.
            // 每次调用一个不可变的参数块:回调只读它,不再读 per-engine 的
            // 可变字段。否则当 vLLM 的 async scheduling 让下一 step 的 host 代码
            // 跑在本 step 的回调之前时,回调会读到新 step 的参数(实测:第二个
            // 请求会拿到上一个请求的数据)。
            //
            // CUDA graph 的区别(重要):capture 期间 host function **不会执行**,
            // 它只是被记录成一个节点,并在**每次 replay** 时用同一个 arg 指针回调。
            // 因此 graph 捕获的块绝不能由回调释放 —— 否则第二个 replay 就是
            // use-after-free(实测 SIGSEGV 落在引擎的 forward 里,每个请求只出 2 个
            // token 引擎就死)。eager 路径下每个块只被调用一次,由回调释放。
            auto* call = new CpuDecodeCall{
                &self, qlen, top_k,
                (const uint16_t*)st->pin_hidden,
                (const uint32_t*)st->pin_ids,
                (const float*)st->pin_weights,
                (float*)st->pin_out,
                capturing,
            };
            st->host_fn = [](void* arg) {
                std::unique_ptr<CpuDecodeCall> c(
                    static_cast<CpuDecodeCall*>(arg));
                static int traced = 0;
                if (std::getenv("XIAOTU_CD_TRACE") && traced < 4000) {
                    ++traced;
                    fprintf(stderr, "[cd] eng=%p qlen=%d k=%d ids=[%u,%u,%u,%u] "
                                    "hid=[%04x,%04x]\n",
                            (void*)c->engine, c->qlen, c->k,
                            c->ids ? c->ids[0] : 0u,
                            c->ids && c->k > 1 ? c->ids[1] : 0u,
                            c->ids && c->k > 2 ? c->ids[2] : 0u,
                            c->ids && c->k > 3 ? c->ids[3] : 0u,
                            c->hid ? c->hid[0] : 0, c->hid ? c->hid[1] : 0);
                }
                MOE* engine = c->engine;
                const int qlen = c->qlen, k = c->k;
                const uint16_t* hid = c->hid;
                const uint32_t* ids = c->ids;
                const float* wts = c->wts;
                float* out = c->out;
                // ---- per-layer segmentation (XIAOTU_CD_TIMING=1) -------------
                // period = callback-entry to callback-entry: the TRUE serialized
                // per-layer time (the decode path is a strict layer-by-layer
                // chain GPU -> D2H -> CPU -> H2D -> GPU). compute = the CPU MoE
                // itself. period - compute = GPU work + copies + host-fn
                // dispatch latency, i.e. the part a GPU-resident layer removes.
                static const bool timing = std::getenv("XIAOTU_CD_TIMING") != nullptr;
                static const int every = [] {
                    const char* e = std::getenv("XIAOTU_CD_TIMING_EVERY");
                    return e ? std::atoi(e) : 43;
                }();
                static double sum_compute = 0, sum_period = 0, sum_wait = 0;
                // EP(跨 rank 归约)是 compute 的一部分,单独拆出来看它占多少:
                // TP=1 时 ep 恒为 0,所以"TP=1 vs TP=2 的 ep 差"就是这个归约的净成本。
                static double sum_ep = 0;
                static double sum_fwd = 0;
                double ep_ms_last = 0.0;
                double fwd_ms_last = 0.0;
                static int n = 0;
                static auto last_cb = std::chrono::steady_clock::now();
                auto t_cb = std::chrono::steady_clock::now();
                if (c->graph_owned) c.release();   // graph 还会再 replay 它
                // 【诊断专用】XIAOTU_MOE_FAKE_CPU=1:跳过 CPU MoE 计算(输出保持原值)。
                // 用途:在服务里量出"该层纯 GPU 侧(注意力/dense+拷贝+派发)每层耗多少",
                // 从而把 period 精确拆成 GPU 部分 与 CPU 部分。**绝不能用于正确性测试**。
                run_moe_and_ep(engine, qlen, k, hid, ids, wts, out, &ep_ms_last, &fwd_ms_last);
                if (timing) {
                    auto t_end = std::chrono::steady_clock::now();
                    const double compute_ms =
                        std::chrono::duration<double, std::milli>(t_end - t_cb).count();
                    const double period_ms =
                        std::chrono::duration<double, std::milli>(t_cb - last_cb).count();
                    last_cb = t_cb;
                    sum_compute += compute_ms;
                    sum_period += period_ms;
                    sum_wait += period_ms - compute_ms;
                    sum_ep += ep_ms_last;
                    sum_fwd += fwd_ms_last;
                    if (++n % every == 0) {
                        fprintf(stderr,
                                "[cd-timing] layers=%d qlen=%d k=%d "
                                "period=%.2fms compute=%.2fms(fwd=%.2f ep=%.2f other=%.2f) rest=%.2fms "
                                "(compute %.0f%%, rest %.0f%%)\n",
                                every, qlen, k,
                                sum_period / every, sum_compute / every,
                                sum_fwd / every, sum_ep / every,
                                (sum_compute - sum_fwd - sum_ep) / every,
                                sum_wait / every,
                                100.0 * sum_compute / std::max(1e-9, sum_period),
                                100.0 * sum_wait / std::max(1e-9, sum_period));
                        fflush(stderr);
                        sum_compute = sum_period = sum_wait = sum_ep = sum_fwd = 0.0;
                    }
                }
            };
            cudaLaunchHostFunc(s, st->host_fn, call);
            if (std::getenv("XIAOTU_CD_ERRCHECK")) {
                cudaError_t e1 = cudaGetLastError();
                if (e1 != cudaSuccess)
                    fprintf(stderr, "[cd] hostfunc err=%s\n", cudaGetErrorString(e1));
            }
            // 3) Async H2D write-back into the stable out_gpu device buffer.
            //    During capture this is recorded as a normal graph node; at
            //    replay the graph executes D2H -> host(CPU compute) -> H2D.
            cudaMemcpyAsync(st->outg, st->out, st->out_bytes,
                            cudaMemcpyHostToDevice, s);
#ifdef XIAOTU_DEBUG_CD
            cudaError_t le = cudaGetLastError();
            if (le != cudaSuccess)
                fprintf(stderr, "[cd] error after launch: %s\n", cudaGetErrorString(le));
#endif
        }, py::arg("stream"), py::arg("qlen"), py::arg("top_k"),
           py::arg("hidden"), py::arg("expert_ids"), py::arg("weights"),
           py::arg("out_gpu"))
        // ---- GPU 流式 prefill:让 Python 能 DMA 引擎自有的主机缓冲 -------------
        // 动机(NOTES §459):GPU prefill 原先要求保留 checkpoint 源张量才能逐层
        // 流式权重,代价是 +269 GiB 主机内存,本机放不下。而引擎的每 node 紧凑分片
        // **合起来正好是一份完整拷贝**,scale 也是引擎自己复制的一份 —— 直接 DMA
        // 这些缓冲即可,于是 XIAOTU_RELEASE_SOURCE 可以保持 1。
        .def("shard_geometry", [](MOE& self) {
            py::dict d;
            d["ns"] = self.shard_ns();
            d["w13_node_bytes"] = self.shard_w13_node_bytes();
            d["w2_node_bytes"] = self.shard_w2_node_bytes();
            d["w13_scale_bytes"] = self.scale_w13_bytes();
            d["w2_scale_bytes"] = self.scale_w2_bytes();
            d["w13_crows"] = self.shard_w13_crows();
            d["w13_cbytes"] = self.shard_w13_cbytes();
            d["w2_crows"] = self.shard_w2_crows();
            d["w2_cbytes"] = self.shard_w2_cbytes();
            return d;
        })
        // which: 0 = w13 shard(node), 1 = w2 shard(node), 2 = w13 scales, 3 = w2 scales
        // 【§610】一次性把自有 host 分片锁页(幂等;返回成功个数)。
        .def("pin_hostbufs", [](MOE& self) -> size_t { return self.pin_hostbufs(); })
        .def("copy_hostbuf_to_device",
             [](MOE& self, int which, int node, uintptr_t dst, uintptr_t stream) -> size_t {
            const void* src = self.host_wbuf(which, node);
            if (!src) return 0;
            size_t n = 0;
            switch (which) {
                case 0: n = self.shard_w13_node_bytes(); break;
                case 1: n = self.shard_w2_node_bytes(); break;
                case 2: n = self.scale_w13_bytes(); break;
                case 3: n = self.scale_w2_bytes(); break;
                default: return 0;
            }
            if (!n) return 0;
            cudaError_t rc = cudaMemcpyAsync(
                reinterpret_cast<void*>(dst), src, n,
                cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream));
            if (rc != cudaSuccess) {
                fprintf(stderr, "[shard-dma] cudaMemcpyAsync failed: %s (which=%d node=%d n=%zu)\n",
                        cudaGetErrorString(rc), which, node, n);
                return 0;
            }
            return n;
        }, py::arg("which"), py::arg("node"), py::arg("dst"), py::arg("stream"))
        .def("cpu_prefill", [](MOE& self,
                              int qlen, int top_k,
                              py::object expert_ids, py::object weights,
                              py::object input, py::object output) {
            self.forward_many(qlen, top_k,
                              as_u32(expert_ids),
                              as_f32(weights),
                              as_u16(input),
                              as_f32m(output));
        }, py::arg("qlen"), py::arg("top_k"), py::arg("expert_ids"),
           py::arg("weights"), py::arg("input"), py::arg("output"))
        .def("debug_first", [](MOE& self, int n) {
            return py::make_tuple(
                py::array(py::buffer_info(
                    const_cast<uint16_t*>(static_cast<const uint16_t*>(self.debug_w13())), sizeof(uint16_t),
                    py::format_descriptor<uint16_t>::format(), 1, { (size_t)n }, { sizeof(uint16_t) })),
                py::array(py::buffer_info(
                    const_cast<uint16_t*>(static_cast<const uint16_t*>(self.debug_w2())), sizeof(uint16_t),
                    py::format_descriptor<uint16_t>::format(), 1, { (size_t)n }, { sizeof(uint16_t) })));
        });
}

// The exported module name is parameterized so one source can be compiled into
// several ISA-specific variants (avx2 / avx512_base / avx512_vnni / ...) that
// coexist on disk and are selected at runtime by the _dynamic_loader. The base
// (no-suffix) build keeps the canonical name `_xiaotu_moe_C`.
//
// Build with e.g.  -DXIAOTU_MOE_MODULE_NAME=_xiaotu_moe_C_avx512_vnni  to emit
// `_xiaotu_moe_C_avx512_vnni`. This must be a single token (PYBIND11_MODULE
// token-pastes its first argument with the init-function name).
#ifndef XIAOTU_MOE_MODULE_NAME
#define XIAOTU_MOE_MODULE_NAME _xiaotu_moe_C
#endif

PYBIND11_MODULE(XIAOTU_MOE_MODULE_NAME, m) {
    m.doc() = "xiaotu-moe CPU MoE engine (BF16 + FP8 closed loops)";
    maybe_install_sigsegv_handler();

    py::class_<MOEConfigV2>(m, "MOEConfigV2")
        .def(py::init<>())
        .def_readwrite("num_processes", &MOEConfigV2::num_processes)
        .def_readwrite("process_id", &MOEConfigV2::process_id)
        .def_readwrite("gpu_id", &MOEConfigV2::gpu_id)
        .def_readwrite("has_gate_proj", &MOEConfigV2::has_gate_proj)
        .def_readwrite("expert_num", &MOEConfigV2::expert_num)
        .def_readwrite("top_k", &MOEConfigV2::top_k)
        .def_readwrite("hidden_size", &MOEConfigV2::hidden_size)
        .def_readwrite("intermediate_size", &MOEConfigV2::intermediate_size)
        .def_readwrite("max_batch_size", &MOEConfigV2::max_batch_size)
        .def_readwrite("max_num_seqs", &MOEConfigV2::max_num_seqs)
        .def_readwrite("stride", &MOEConfigV2::stride)
        .def_readwrite("group_min_len", &MOEConfigV2::group_min_len)
        .def_readwrite("group_max_len", &MOEConfigV2::group_max_len)
        .def_readwrite("groupN", &MOEConfigV2::groupN)
        .def_readwrite("groupK", &MOEConfigV2::groupK)
        .def_readwrite("swiglu_alpha", &MOEConfigV2::swiglu_alpha)
        .def_readwrite("swiglu_beta", &MOEConfigV2::swiglu_beta)
        .def_readwrite("swiglu_limit", &MOEConfigV2::swiglu_limit)
        .def_readwrite("activation_type", &MOEConfigV2::activation_type)
        .def_readwrite("use_gpu_prefill", &MOEConfigV2::use_gpu_prefill);

    bind_moe_class<BF16WeightTraits, BF16Activation>(m, "MOE_BF16");
    bind_moe_class<BF16WeightTraits, FP16Activation>(m, "MOE_FP16");
    // Backward-compatible alias from Phase 1 (real engine also exports MOE_BF16_FP16).
    // pybind11 forbids registering one C++ type twice, so alias the class object.
    m.attr("MOE_BF16_FP16") = m.attr("MOE_FP16");

    bind_moe_class<FP8WeightTraits, BF16Activation>(m, "MOE_FP8");
    bind_moe_class<FP8WeightTraits, FP16Activation>(m, "MOE_FP8_FP16");

    bind_moe_class<MXFP4WeightTraits, BF16Activation>(m, "MOE_MXFP4");
    bind_moe_class<MXFP4WeightTraits, FP16Activation>(m, "MOE_MXFP4_FP16");
    bind_moe_class<WNA16WeightTraits, BF16Activation>(m, "MOE_WNA16");
    bind_moe_class<WNA16WeightTraits, FP16Activation>(m, "MOE_WNA16_FP16");

    bind_moe_class<NVFP4WeightTraits, BF16Activation>(m, "MOE_NVFP4");
    bind_moe_class<NVFP4WeightTraits, FP16Activation>(m, "MOE_NVFP4_FP16");
}

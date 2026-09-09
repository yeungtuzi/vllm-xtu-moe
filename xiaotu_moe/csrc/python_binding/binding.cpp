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

#include <cuda_runtime.h>
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

    void ensure_buffers(size_t nh, size_t ni, size_t nw, size_t no) {
        if (nh > cap_hidden) {
            if (pin_hidden) cudaFreeHost(pin_hidden);
            cudaHostAlloc(&pin_hidden, nh, cudaHostAllocDefault);
            cap_hidden = nh;
        }
        if (ni > cap_ids) {
            if (pin_ids) cudaFreeHost(pin_ids);
            cudaHostAlloc(&pin_ids, ni, cudaHostAllocDefault);
            cap_ids = ni;
        }
        if (nw > cap_weights) {
            if (pin_weights) cudaFreeHost(pin_weights);
            cudaHostAlloc(&pin_weights, nw, cudaHostAllocDefault);
            cap_weights = nw;
        }
        if (no > cap_out) {
            if (pin_out) cudaFreeHost(pin_out);
            cudaHostAlloc(&pin_out, no, cudaHostAllocDefault);
            cap_out = no;
        }
    }

    ~CpuDecodeState() {
        if (pin_hidden) cudaFreeHost(pin_hidden);
        if (pin_ids) cudaFreeHost(pin_ids);
        if (pin_weights) cudaFreeHost(pin_weights);
        if (pin_out) cudaFreeHost(pin_out);
    }
};

// Map engine pointer -> its pinned buffers / host-fn context. Keyed by the
// MOE* identity; every MOE instance across every type has a unique address, so
// sharing one map across all template instantiations is safe.
std::mutex g_cd_mtx;
std::unordered_map<const void*, std::unique_ptr<CpuDecodeState>> g_cd_state;

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
            return new MOE(cfg, as_ptr(w13), as_ptr(w2), as_ptr(w13_g), as_ptr(w2_g),
                           static_cast<const float*>(as_ptr(w13_global)),
                           static_cast<const float*>(as_ptr(w2_global)));
        }), py::arg("cfg"), py::arg("w13_weight"), py::arg("w2_weight"),
            py::arg("w13_scale") = py::int_(0), py::arg("w2_scale") = py::int_(0),
            py::arg("w13_global_scale") = py::int_(0), py::arg("w2_global_scale") = py::int_(0))
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

            const size_t nh = (size_t)qlen * H * sizeof(uint16_t);
            const size_t ni = (size_t)qlen * top_k * sizeof(uint32_t);
            const size_t nw = (size_t)qlen * top_k * sizeof(float);
            const size_t no = (size_t)qlen * H * sizeof(float);

            std::lock_guard<std::mutex> lg(g_cd_mtx);
            auto& st = g_cd_state[&self];
            if (!st) st = std::make_unique<CpuDecodeState>();
            st->ensure_buffers(nh, ni, nw, no);

            st->stream = s;
            st->engine = &self;
            st->qlen = qlen;
            st->k = top_k;
            st->out_bytes = no;
            st->hid = (const uint16_t*)st->pin_hidden;
            st->ids = (const uint32_t*)st->pin_ids;
            st->wts = (const float*)st->pin_weights;
            st->out = (float*)st->pin_out;
            st->outg = outg_dev;

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
            st->host_fn = [](void* arg) {
                auto* d = static_cast<CpuDecodeState*>(arg);
                auto* e = static_cast<MOE*>(const_cast<void*>(d->engine));
                e->forward_many(d->qlen, d->k, d->ids, d->wts,
                                d->hid, d->out);
            };
            cudaLaunchHostFunc(s, st->host_fn, &*st);
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

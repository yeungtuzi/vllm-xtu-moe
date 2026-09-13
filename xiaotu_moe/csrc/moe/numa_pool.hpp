// xiaotu-moe: persistent NUMA-aware worker pool (Backend_NUMA equivalent).
//
// This is the CPU MoE thread backend, an open reimplementation of lk_moe's
// Backend_NUMA. It provides:
//   * persistent worker threads (created once, reused across forward calls —
//     no per-call thread spawn/join),
//   * NUMA-node-aware affinity: each worker pinned to a distinct physical core,
//     spread across NUMA nodes (cross-node communication minimized),
//   * dynamic scheduling via a shared atomic index (implicit work stealing:
//     workers pull the next index, so no worker starves while others run long),
//   * optional NUMA-interleaved allocation for engine-owned buffers (default on
//     for weight snapshots when >1 node), via the mbind syscall.
//
// IMPORTANT: builds WITHOUT libnuma (per project constraint). Topology comes from
// /proc/cpuinfo + /sys/devices/system/node; affinity via sched_setaffinity
// (glibc); memory interleave via the raw mbind syscall (<linux/mempolicy.h>).
//
// License: Apache-2.0.

#ifndef XIAOTU_MOE_NUMA_POOL_HPP
#define XIAOTU_MOE_NUMA_POOL_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <immintrin.h>   // _mm_pause for hot-restart spin loops
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <sched.h>

#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <utility>
#include <unordered_map>

// Signal-based native-stack dump (diagnostics for stuck-forward investigation).
// Registered in NumaWorkPool so every thread inherits the handler; on SIGUSR2
// the receiving thread appends its own glibc backtrace to a single file. The
// harness signals every /proc/<pid>/task/*/tid to get ALL threads' native
// stacks. Needs -g -fno-omit-frame-pointer for depth; still accurate with -O3.
#include <signal.h>
#include <execinfo.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <atomic>

namespace xiaotu_moe {
namespace {
std::atomic<int> g_stack_dump_fd{-1};

void stack_dump_handler(int /*sig*/) {
    int fd = g_stack_dump_fd.load(std::memory_order_relaxed);
    if (fd < 0) return;
    void* bt[160];
    int n = backtrace(bt, 160);
    int tid = (int)syscall(SYS_gettid);
    char hdr[96];
    int hl = snprintf(hdr, sizeof(hdr), "\n===== native stack tid=%d n=%d =====", tid, n);
    if (hl > 0) (void)!write(fd, hdr, (size_t)hl);
    (void)!write(fd, "\n", 1);
    backtrace_symbols_fd(bt, n, fd);
    fdatasync(fd);
}

struct StackDumperRegistrar {
    StackDumperRegistrar() {
        char path[128];
        int pid = (int)getpid();
        snprintf(path, sizeof(path), "/tmp/xiaotu_stack_%d.txt", pid);
        int fd = open(path, O_CREAT | O_WRONLY | O_TRUNC | O_APPEND, 0644);
        if (fd >= 0) {
            g_stack_dump_fd.store(fd, std::memory_order_relaxed);
            struct sigaction sa;
            memset(&sa, 0, sizeof(sa));
            sa.sa_handler = stack_dump_handler;
            sigemptyset(&sa.sa_mask);
            sa.sa_flags = 0;                     // no SA_RESTART -> interrupts futex wait
            sigaction(SIGUSR2, &sa, nullptr);
        }
    }
};
} // anonymous namespace
} // namespace xiaotu_moe

namespace xiaotu_moe {

// ---------------------------------------------------------------------------
// Topology discovery (no libnuma: /proc/cpuinfo + /sys/devices/system/node).
// ---------------------------------------------------------------------------

// parse a Linux cpulist ("0-3,8,11-15") into a sorted set of cpu ids.
inline std::set<int> parse_cpulist(const std::string& s) {
    std::set<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        auto dash = tok.find('-');
        if (dash == std::string::npos) {
            out.insert(std::atoi(tok.c_str()));
        } else {
            int a = std::atoi(tok.substr(0, dash).c_str());
            int b = std::atoi(tok.substr(dash + 1).c_str());
            if (a > b) std::swap(a, b);
            for (int c = a; c <= b; ++c) out.insert(c);
        }
    }
    return out;
}

struct NumaTopology {
    // For each configured NUMA node: the set of cpu ids in it.
    std::vector<std::vector<int>> node_cpus;
    // For each cpu id: node index.
    std::unordered_map<int, int> cpu_node;
    // For each cpu id: physical core id (from core id); used to avoid pinning
    // siblings of the same physical core.
    std::unordered_map<int, int> cpu_core;
    // Node index -> physical package (socket) index. Grouped so that all nodes
    // whose first cpu shares a physical_package_id map to the same socket. Called
    // node_socket so dist[10,12] within-socket vs dist[32] cross-socket can be
    // exploited by per-socket weight replication.
    std::vector<int> node_socket;
};

inline int cpu_package(int cpu) {
    char p[256];
    snprintf(p, sizeof(p), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
    std::ifstream f(p);
    if (!f.good()) return 0;
    int id = -1; f >> id;
    return id < 0 ? 0 : id;
}

inline NumaTopology discover_numa_topology() {
    NumaTopology t;
    // Nodes 0.. while nodeX/cpulist exists.
    for (int n = 0;; ++n) {
        std::string p = "/sys/devices/system/node/node" + std::to_string(n) + "/cpulist";
        std::ifstream f(p);
        if (!f.good()) { if (n == 0) return t; break; }
        std::string s; std::getline(f, s);
        std::vector<int> cpus;
        for (int c : parse_cpulist(s)) { cpus.push_back(c); t.cpu_node[c] = n; }
        if (!cpus.empty()) t.node_cpus.push_back(cpus);
    }
    // Physical core ids from /proc/cpuinfo (boarder.py reads the first block; we
    // use the standard "processor : N" / "core id : C" pairing).
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line; int cur_proc = -1; int cur_core = -1;
    auto flush = [&]() {
        if (cur_proc >= 0 && cur_core >= 0) t.cpu_core[cur_proc] = cur_core;
    };
    while (std::getline(cpuinfo, line)) {
        if (line.rfind("processor", 0) == 0) { flush(); cur_proc = std::atoi(line.c_str() + 10); cur_core = -1; }
        else if (line.rfind("core id", 0) == 0) { cur_core = std::atoi(line.c_str() + 8); }
    }
    flush();
    // Socket (physical package) per node: group nodes by the package of their
    // first cpu, assigning compact socket ids 0.. in node order.
    {
        std::unordered_map<int, int> pkg_to_sock;
        int next = 0;
        for (size_t n = 0; n < t.node_cpus.size(); ++n) {
            int cpu = t.node_cpus[n].front();
            int pkg = cpu_package(cpu);
            auto it = pkg_to_sock.find(pkg);
            if (it == pkg_to_sock.end()) { pkg_to_sock[pkg] = next++; }
            t.node_socket.push_back(pkg_to_sock[pkg]);
        }
        if (t.node_socket.empty()) t.node_socket.push_back(0);
    }
    return t;
}

// ---------------------------------------------------------------------------
// mbind-based NUMA-interleaved allocation (raw syscall, no libnuma).
// ---------------------------------------------------------------------------
inline void numa_set_interleaved(void* addr, size_t len) {
    static int nodes_initialized = []() {
        // gather configured nodes at first use
        NumaTopology t = discover_numa_topology();
        if (t.node_cpus.size() > 1) {
            unsigned long mask = (1UL << t.node_cpus.size()) - 1;
            return static_cast<int>(mask);
        }
        return 0;
    }();
    if (nodes_initialized <= 1) return;  // single node: default policy fine
    unsigned long nodemask = static_cast<unsigned long>(nodes_initialized);
    long rc = syscall(SYS_mbind, addr, len, MPOL_INTERLEAVE,
                      &nodemask, sizeof(nodemask) * 8, MPOL_MF_MOVE);
    (void)rc;
}

// ---------------------------------------------------------------------------
// Thread-policy based NUMA interleave: set the calling thread's memory policy
// to MPOL_INTERLEAVE so that its new allocations (operator new / mmap for the
// large weight snapshots) fault their pages in already interleaved across all
// NUMA nodes — the most reliable path (mbind-on-existing-pages with MF_MOVE is
// fragile: it can EINVAL on sub-ranges and only migrates existing pages). We
// bracket each weight copy: begin() sets interleave, end() restores default.
// lk-moe / ktransformers similarly place weights so both CPU sockets read their
// weight pages locally (the MoE is bandwidth-bound; with weights on one socket
// the other socket's 84 threads read remote DRAM at ~half bandwidth).
// ---------------------------------------------------------------------------
inline void numa_interleave_begin() {
    static int nmask = []() {
        NumaTopology t = discover_numa_topology();
        if (t.node_cpus.size() > 1) return (int)((1UL << t.node_cpus.size()) - 1);
        return 0;
    }();
    if (nmask > 1) {
        unsigned long mask = (unsigned long)nmask;
        syscall(SYS_set_mempolicy, MPOL_INTERLEAVE, &mask, sizeof(mask) * 8);
    }
}
inline void numa_interleave_end() {
    syscall(SYS_set_mempolicy, MPOL_DEFAULT, nullptr, 0);
}

// ---------------------------------------------------------------------------
// Socket helpers (per-socket weight replication).
// Linux memory policy for locality is a per-thread property; we expose the
// worker's pinned socket through a thread_local so the hot MoE job lambda can
// pick the replica that lives on the SAME socket (eliminating distance-32
// cross-socket weight reads entirely, mirroring lk_moe's ~3% cross-node design).
// ---------------------------------------------------------------------------

// Number of physical packages (sockets) on this host.
inline int numa_socket_count() {
    static const int n = []() {
        NumaTopology t = discover_numa_topology();
        if (t.node_socket.empty()) return 1;
        int mx = 0; for (int s : t.node_socket) mx = std::max(mx, s);
        return mx + 1;
    }();
    return n;
}

// Socket index that `node` belongs to.
inline int numa_socket_of_node(int node) {
    NumaTopology t = discover_numa_topology();
    if (node >= 0 && node < (int)t.node_socket.size()) return t.node_socket[node];
    return 0;
}

// Socket index of the calling worker thread (uses sched_getcpu -> node -> socket).
// The cpu->socket map is built ONCE; the worker is pinned stably so sched_getcpu
// returns its fixed core and this is a single array lookup per call.
inline int current_socket() {
    static const std::vector<int> g_cpusock = []() {
        NumaTopology t = discover_numa_topology();
        int maxcpu = 0;
        for (auto& kv : t.cpu_node) maxcpu = std::max(maxcpu, kv.first);
        std::vector<int> m(maxcpu + 64, -1);
        for (auto& kv : t.cpu_node) m[kv.first] = numa_socket_of_node(kv.second);
        return m;
    }();
    int cpu = sched_getcpu();
    if (cpu >= 0 && (size_t)cpu < g_cpusock.size() && g_cpusock[cpu] >= 0)
        return g_cpusock[cpu];
    return 0;
}

// Allocate `bytes` on the nodes of `socket` only, interleaved WITHIN that
// socket (so every page is ≤distance-12 local to the socket, never distance-32
// cross-socket). Uses mmap + a thread-scoped MPOL_INTERLEAVE on the socket's
// node subset, then touches (writes) each page so it faults in on that socket.
// Returns nullptr if the socket is unknown or the allocation fails (caller then
// falls back to a single shared copy).
inline void* numa_socket_alloc(size_t bytes, int socket) {
    if (bytes == 0) return nullptr;
    NumaTopology t = discover_numa_topology();
    unsigned long mask = 0;
    int nnodes = (int)t.node_cpus.size();
    for (int n = 0; n < nnodes; ++n)
        if (t.node_socket[n] == socket) mask |= (1UL << n);
    if (mask == 0) return nullptr;
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    // mbind() 只作用于**这段映射**。绝不要用 set_mempolicy():那是给"线程"设策略,
    // 会被随后新建的线程继承,窗口内别处的大块分配(如 vLLM 的 pinned 权重缓存)
    // 就被绑到单个 node 上 ⇒ CONSTRAINT_MEMORY_POLICY 的 OOM,整进程被杀。
    long rc = syscall(SYS_mbind, p, bytes, MPOL_INTERLEAVE, &mask, sizeof(mask) * 8, 0);
    {
        // Fault every page in on this socket (region policy is INTERLEAVE over
        // just this socket's nodes). volatile forces the stores.
        volatile char* cp = static_cast<volatile char*>(p);
        const size_t PS = 4096;
        for (size_t off = 0; off < bytes; off += PS) cp[off] = 0;
    }
    if (rc != 0) {
        // policy failed: pages were still touched above (default placement), so
        // the region is valid — just not socket-local. Caller falls back.
    }
    return p;
}

// Number of NUMA nodes (memory domains) on this host.
inline int numa_node_count() {
    static const int n = []() { return (int)discover_numa_topology().node_cpus.size(); }();
    return n;
}

// NUMA node of the calling worker thread (sched_getcpu -> node). Builds the
// cpu->node map once; the worker is pinned stably so this is a single lookup.
inline int current_node() {
    static const std::vector<int> g_cpunode = []() {
        NumaTopology t = discover_numa_topology();
        int maxcpu = 0; for (auto& kv : t.cpu_node) maxcpu = std::max(maxcpu, kv.first);
        std::vector<int> m(maxcpu + 64, 0);
        for (auto& kv : t.cpu_node) m[kv.first] = kv.second;
        return m;
    }();
    int cpu = sched_getcpu();
    if (cpu >= 0 && (size_t)cpu < g_cpunode.size()) return g_cpunode[cpu];
    return 0;
}

// Allocate `bytes` on ONE NUMA node (single physical copy), mirroring
// lktransformers `allocate_aligned_numa(size, nid)=numa_alloc_onnode`: mmap then
// fault each page in under a thread-scoped MPOL_BIND to `node`, so every page
// physically lands on that node's memory (never cross-QPI). Returns nullptr on
// failure (caller falls back).
inline void* numa_alloc_onnode(size_t bytes, int node) {
    if (bytes == 0) return nullptr;
    NumaTopology t = discover_numa_topology();
    int nn = (int)t.node_cpus.size();
    if (node < 0 || node >= nn) return nullptr;
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    unsigned long mask = 1UL << node;
    long rc = syscall(SYS_set_mempolicy, MPOL_BIND, &mask, sizeof(mask) * 8);
    volatile char* cp = static_cast<volatile char*>(p);
    const size_t PS = 4096;
    if (rc == 0) {
        for (size_t off = 0; off < bytes; off += PS) cp[off] = 0;  // fault on node
        syscall(SYS_set_mempolicy, MPOL_DEFAULT, nullptr, 0);
    } else {
        for (size_t off = 0; off < bytes; off += PS) cp[off] = 0;  // default placement
    }
    return p;
}

// ---------------------------------------------------------------------------
// NumaWorkPool
// ---------------------------------------------------------------------------
class NumaWorkPool {
public:
    explicit NumaWorkPool(size_t n = 0)
        : nt_(n == 0 ? default_threads() : n), stop_(false),
          current_gen_(0), counter_(0), remaining_(0), n_(0) {
        // per-worker completion slots kept only for diagnostics; the completion
        // barrier no longer waits on them (see parallel_for).
        worker_gen_.reset(new std::atomic<uint64_t>[nt_]);
        for (size_t w = 0; w < nt_; ++w) worker_gen_[w].store(0);
        worker_node_.assign(nt_, 0);
        node_ticket_.reset(new PaddedTicket[kMaxNodeShards]);
        for (size_t n = 0; n < kMaxNodeShards; ++n) node_ticket_[n].v.store(0);
        static StackDumperRegistrar registrar;  // SIGUSR2 native-stack dump
        const char* sp = std::getenv("XIAOTU_MOE_SPIN_IDLE_US");
        if (sp) { long v = std::atol(sp); if (v >= 0) spin_idle_us_.store((uint64_t)v); }
        start_workers();
    }

    ~NumaWorkPool() { stop(); }

    size_t nthreads() const { return nt_; }

    // Diagnostic: print core order + per-node worker counts (debug only).
    void dump_affinity(const char* tag = "") const {
        int per_node[64] = {0};
        size_t used = std::min(nt_, cores_.size());
        for (size_t w = 0; w < used; ++w) {
            int cpu = cores_[w % cores_.size()];
            auto it = topo_.cpu_node.find(cpu);
            if (it != topo_.cpu_node.end()) per_node[it->second]++;
        }
        fprintf(stderr, "[affinity%s] nt=%zu cores=%zu order:", tag, nt_, cores_.size());
        int shown = 0;
        for (size_t i = 0; i < used && shown < 48; ++i, ++shown)
            fprintf(stderr, "%d%s", cores_[i % cores_.size()], i + 1 < used ? "," : "");
        fprintf(stderr, "\n[affinity%s] node_present=0x%lx per_node:", tag, node_present_);
        for (int n = 0; n < 64; ++n) if (per_node[n]) fprintf(stderr, "n%d:%d ", n, per_node[n]);
        fprintf(stderr, "\n");
    }

    // run fn(i) for i in [0, n). Persistent workers; dynamic index scheduling.
    template <typename F>
    void parallel_for(size_t n, const F& fn) {
        parallel_for_impl(n, 0, fn);
    }

    // Same, but only ~`limit` workers may claim tickets. The completion barrier
    // waits only for claimed tickets, so non-participants cost nothing; this caps
    // the tail latency (the caller waits for the LAST worker, and a descheduled
    // worker on a busy 192-thread pool costs ~100us) that otherwise dominates
    // small-batch decode, where the useful work per call is a few microseconds.
    template <typename F>
    void parallel_for_limited(size_t n, size_t limit, const F& fn) {
        parallel_for_impl(n, limit, fn);
    }

    template <typename F>
    void parallel_for_impl(size_t n, size_t limit, const F& fn) {
        // Serialize access to the shared generation/counter state. When the pool
        // is shared process-wide (one pool for all MoE layers, mirroring lk_moe's
        // single Backend_NUMA), different layers could otherwise race on
        // counter_/task_/current_gen_. vLLM-eager drives layers sequentially, so
        // this lock is uncontended in practice; it only orders genuinely-concurrent
        // forward_many calls issued from multiple driver threads.
        std::lock_guard<std::mutex> call_lock(call_mtx_);
        if (nt_ <= 1 || n <= 1) {
            for (size_t i = 0; i < n; ++i) fn(i);
            return;
        }
        uint64_t gen;
        {
            std::lock_guard<std::mutex> lk(work_mtx_);
            // 【轮 68】删除 `task_ = std::function<void(size_t)>(fn);`:全文检查确认 task_
            // **只被赋值、从未被读取**(worker 走 pub_flat_/pub_shard_ 裸函数指针),每次调用
            // 白构造一个 std::function(可能堆分配)并持 work_mtx_。保留成员仅为兼容注释。
            pub_flat_ = &flat_thunk<F>; pub_flat_ctx_ = (void*)&fn;
            pub_shard_ = nullptr; pub_shard_ctx_ = nullptr;
            // 【第 129 轮修】`n_`/`worker_limit_`/`sharded_call_` 原**写在这里**(奇数 store
            // 之前),与分片发布(§184)是同一个缺陷:字段写在发布窗口之外 ⇒ 读者可能以
            // 旧 `start` 配新 `n` 做 `i = t - start` 判定,把属于本代的票判成越界/空洞而丢掉
            // ⇒ remaining_ 少减一次。实测(§190):并发压测
            // `WATCHDOG fired: n=8 start=960104 end=960112 counter=960232 remaining=1`。
            // 现已把这几个字段移到下面的"奇数→偶数"窗口内。
            // MONOTONIC ticket counter (never reset). Each call occupies the
            // ticket range [start_, start_+n). Workers bound to THIS call compute
            // i = ticket - start_ and only run for i in [0,n); any ticket beyond
            // that range means the worker's own range is exhausted and it re-arms
            // for the next generation.
            //
            // PUBLISH ORDER (both halves matter, see the sharded twin below):
            //  1) every field (task_, n_, remaining_) is written while holding
            //     work_mtx_, so a worker that anchors under the lock always sees a
            //     complete call;
            //  2) current_gen_ is bumped BEFORE the ticket counter is read: a
            //     worker that pulls a ticket while still observing the old
            //     generation must have pulled it before this load, i.e. below
            //     start_, so a stale ticket can never alias a job of this call.
            // Reversing (2) makes stale tickets land inside the new range and be
            // silently dropped; reading the counter before (1) let a worker
            // decrement remaining_ before it was stored, which hung the caller.
            // 【轮 69 奇偶 seqlock】current_gen_ 现在是**序号**:奇数 = "正在发布",
            // 偶数 = "已就绪"。调用方必须在读 counter_ 之前先 store 奇数、写完全部字段
            // 再 store 偶数。这样 worker 只需接受偶数即可,不会看到"新 gen + 旧 start_"
            // 的撕裂(轮 68 的朴素"复读 gen"正是死在这里:撕裂时 gen 已新、start_ 还旧,
            // 复读一致 ⇒ 查不出来 ⇒ worker 丢票 ⇒ remaining_ 永不归零)。
            // 原有的"先掀 gen 再读 counter_ 以保证陈旧票落在 start_ 之前"不变。
            gen = current_gen_.load(std::memory_order_relaxed) + 2;   // 下一个偶数代
            current_gen_.store(gen - 1, std::memory_order_release);   // 奇数:发布中
            start_ = counter_.load();
            n_ = n;
            worker_limit_.store(limit >= nt_ ? 0 : limit, std::memory_order_relaxed);
            sharded_call_ = 0;   // this is a flat call: task_ is valid, so reset
                                 // any stale sharded marker so late workers anchor flat.
            remaining_.store(n);      // outstanding work items in this call
            // diagnostic processed-bitset for this call (only when debug/trace on)
            if (diag_active()) proc_vec_.assign(n, 0);
            current_gen_.store(gen, std::memory_order_release);       // 偶数:就绪
        }
        // Unlimited calls wake everybody (prefill needs the whole pool). Limited
        // calls rely on their participants already spinning, so the fast path
        // issues no syscall at all; the wait loop below falls back to notify_all
        // if nobody claims a ticket (participants parked during an idle gap).
        const bool limited = (limit > 0 && limit < nt_);
        if (!limited) cv_.notify_all();
        {
            std::unique_lock<std::mutex> lk(done_mtx_);
            auto t0 = std::chrono::steady_clock::now();
            const auto deadline = t0 + std::chrono::seconds(300);
            const char* tre = std::getenv("XIAOTU_MOE_POOL_TRACE");
            long rstep_s = (tre && std::atoi(tre) > 0) ? (long)std::atoi(tre) : 0;
            bool late = false;
            if (rstep_s > 0) {
                // STALL-TRACE mode: sliced plain waits so live state can be
                // printed while a call is outstanding (no abort). Only enabled
                // explicitly for diagnosing the CPU-MoE conc stall. The normal
                // path below uses an immediate predicate wait.
                int last_report = -1;
                for (;;) {
                    if (stop_ || remaining_.load(std::memory_order_acquire) == 0) break;
                    auto now = std::chrono::steady_clock::now();
                    if (now >= deadline) { late = true; break; }
                    auto nw = now + std::chrono::seconds(rstep_s);
                    if (nw > deadline) nw = deadline;
                    done_cv_.wait_until(lk, nw);   // plain wait to allow periodic wake
                    long es = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count();
                    int rk = (int)(es / 1000 / rstep_s);
                    if (rk != last_report) {
                        last_report = rk;
                        if (remaining_.load(std::memory_order_acquire) > 0) {
                            std::string miss; size_t nmiss = 0;
                            size_t first_missing = (size_t)-1;
                            for (size_t z = 0; z < n_; ++z) {
                                if (!proc_vec_[z]) {
                                    if (first_missing == (size_t)-1) first_missing = z;
                                    if (nmiss < 16) miss += std::to_string(start_+z)+" ";
                                    ++nmiss;
                                }
                            }
                            fprintf(stderr,
                                "[pool] STALL t=%lds gen=%llu n=%zu start=%zu rem=%zu livegen=%llu dropped=%zu first=%zu missing=%zu {%s}\n",
                                es/1000, (unsigned long long)gen, n_, start_, remaining_.load(),
                                (unsigned long long)current_gen_.load(), dropped_.load(),
                                first_missing, nmiss, miss.c_str());
                        }
                    }
                }
            } else {
                // Completion countdown barrier (normal). We wait for the workers
                // that ACTUALLY claim an index; idle workers are never waited on.
                // Hot path: while the pool is spinning (decode), the countdown
                // closes in microseconds, so busy-wait instead of sleeping on the
                // condvar -> avoids a futex round-trip per phase. Only fall back
                // to the condvar after spin_idle_us_ of no progress.
                bool spun_done = false;
                const uint64_t idle_us = spin_idle_us_.load(std::memory_order_relaxed);
                if (idle_us > 0) {
                    auto dl = std::chrono::steady_clock::now()
                            + std::chrono::microseconds(idle_us);
                    bool woke = false;
                    while (std::chrono::steady_clock::now() < dl) {
                        if (remaining_.load(std::memory_order_acquire) == 0) {
                            spun_done = true; break;
                        }
                        // Limited call whose participants are ALL parked (idle
                        // gap): remaining_ is still n, so nobody claimed a ticket.
                        // Wake the pool once and keep spinning. Dynamic ticket
                        // pulling means one awake participant drains every ticket,
                        // so a partial countdown never needs a wake (and a clock
                        // check in this loop would cost more than it saves).
                        if (limited && !woke &&
                            remaining_.load(std::memory_order_acquire) == n) {
                            woke = true;
                            cv_.notify_all();
                        }
                        _mm_pause();
                    }
                }
                if (!spun_done) {
                    if (limited) cv_.notify_all();
                    late = !done_cv_.wait_until(lk, deadline, [&] {
                        if (stop_) return true;
                        return remaining_.load(std::memory_order_acquire) == 0;
                    });
                }
            }
            long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();
            // Genuine hang: remaining>0 past the 300s deadline -> dump + abort.
            if (late) {
                fprintf(stderr, "[pool] WATCHDOG fired: gen=%llu n=%zu start=%zu end=%zu counter=%zu remaining=%zu current_gen=%llu dropped=%zu\n",
                    (unsigned long long)gen, n_, start_, start_+n_, counter_.load(),
                    remaining_.load(), (unsigned long long)current_gen_.load(),
                    dropped_.load());
                if (diag_active()) {
                    size_t miss=0; std::string s;
                    for (size_t z=0;z<n_;++z) if(!proc_vec_[z]){ s+=std::to_string(start_+z)+" "; ++miss;}
                    fprintf(stderr, "  missing %zu tickets: %s\n", miss, s.c_str());
                }
                for (size_t w = 0; w < nt_; ++w)
                    fprintf(stderr, "  worker[%zu] slot=%llu\n",
                        w, (unsigned long long)worker_gen_[w].load());
                abort();
            }
            long slow_ms = 2000;
            if (const char* e = std::getenv("XIAOTU_MOE_POOL_SLOW_MS")) {
                long v = std::atol(e); if (v >= 0) slow_ms = v;
            }
            if (elapsed_ms >= slow_ms) {
                fprintf(stderr,
                    "[pool] SLOW parallel_for: elapsed=%ldms gen=%llu n=%zu counter=%zu remaining=%zu current_gen=%llu\n",
                    elapsed_ms, (unsigned long long)gen, n_, counter_.load(),
                    remaining_.load(), (unsigned long long)current_gen_.load());
                size_t nlag = 0;
                for (size_t w = 0; w < nt_; ++w) {
                    uint64_t s = worker_gen_[w].load(std::memory_order_acquire);
                    if (s != gen) {
                        fprintf(stderr, "  LAGGARD worker[%zu] slot=%llu\n",
                            (unsigned long long)w, (unsigned long long)s);
                        ++nlag;
                    }
                }
                fprintf(stderr, "  laggards=%zu/%zu remaining=%zu\n", nlag, (size_t)nt_,
                        remaining_.load());
            }
        }
        if (std::getenv("XIAOTU_MOE_POOL_DEBUG")) {
            fprintf(stderr, "[pool] parallel_for(n=%zu) returned at gen=%llu\n",
                    n, (unsigned long long)gen);
        }
    }

    // -----------------------------------------------------------------------
    // Node-scoped parallel execution for single-copy weight sharding.
    //   nnodes        : number of NUMA nodes participating (the shard count)
    //   job_counts[n] : how many jobs node n owns (total jobs = sum)
    //   fn(n, local)  : called as fn(node_index, local_index_in_node) — the caller
    //                   maps local_index to its per-node job list.
    // Every job runs ONLY on a worker pinned to the node that owns it (worker
    // pulls from node_ticket_[my_node_]), so weight reads are guaranteed local
    // to the node holding those blocks. Same completion barrier as parallel_for.
    // -----------------------------------------------------------------------
    template <typename F>
    void parallel_for_sharded(int nnodes, const size_t* job_counts, const F& fn) {
        parallel_for_sharded_impl(nnodes, job_counts, 0, fn);
    }

    // Same, with the worker-subset gate (see parallel_for_limited).
    template <typename F>
    void parallel_for_sharded_limited(int nnodes, const size_t* job_counts, size_t limit,
                                      const F& fn) {
        parallel_for_sharded_impl(nnodes, job_counts, limit, fn);
    }

    template <typename F>
    void parallel_for_sharded_impl(int nnodes, const size_t* job_counts, size_t limit,
                                   const F& fn) {
        std::lock_guard<std::mutex> call_lock(call_mtx_);
        if (nnodes <= 0) return;
        if (nt_ <= 1) {
            for (int n = 0; n < nnodes; ++n)
                for (size_t j = 0; j < job_counts[n]; ++j) fn((size_t)n, j);
            return;
        }
        // Must have >=1 worker on every participating node, else some jobs are
        // unclaimable -> hang. If not, degrade to a correct serial loop.
        unsigned long need = 0;
        for (int n = 0; n < nnodes; ++n) need |= (1UL << n);
        bool all_present = ((node_present_ & need) == need);
        if (nt_ == 1 || !all_present) {
            for (int n = 0; n < nnodes; ++n)
                for (size_t j = 0; j < job_counts[n]; ++j) fn((size_t)n, j);
            return;
        }
        if (nnodes > (int)kMaxNodeShards) {   // safety: should never happen
            for (int n = 0; n < nnodes; ++n)
                for (size_t j = 0; j < job_counts[n]; ++j) fn((size_t)n, j);
            return;
        }
        size_t total = 0;
        {
            std::lock_guard<std::mutex> lk(work_mtx_);
            node_base_.resize((size_t)nnodes);
            node_nj_.resize((size_t)nnodes);
            // 【轮 68】同 parallel_for_impl:sharded_task_ 也从未被读取,删除其构造。
            pub_shard_ = &shard_thunk<F>; pub_shard_ctx_ = (void*)&fn;
            pub_flat_ = nullptr; pub_flat_ctx_ = nullptr;
            // A sharded call has no flat task: clear it so a worker that somehow
            // takes the flat path can never invoke the previous call's function.
            task_ = nullptr;
            if (diag_active()) {
                shard_cnt_.assign((size_t)nnodes * kShardDiagStride, 0u);
            }
            // 【第 182 轮·消融 #2,env 控制,**绝不进交付配置**】
            // 中性堆数组填零,规模与 shard_cnt_ 相同,位置也与它相同(发布窗口之前)。
            // 目的:复现 POOL_TRACE 对"发布者节奏"的扰动,以判别停顿是否与发布节奏相关。
            static std::vector<unsigned char> s_ablate_buf;
            static const bool s_ablate2_on = [] {
                const char* e = std::getenv("XIAOTU_MOE_ABLATE_HEAPFILL");
                return e != nullptr && e[0] == '1';
            }();
            if (s_ablate2_on) {
                s_ablate_buf.assign((size_t)nnodes * (size_t)kShardDiagStride, 0u);
            }
            // 【第 124 轮·真根因修复】下面这些字段原本写在**奇数 store 之前**,
            // 直接违反本文件 :455-471 自己写下的不变式:"先 store 奇数 → 写完全部字段
            // → 再 store 偶数,worker 只接受偶数"。后果:仍在上一代(偶代)的 worker 会在
            // :937 读到**新一代的 node_nj_** 而 node_base_ 还是旧的 ⇒ 可领区间变小
            // ⇒ **丢掉一张票** ⇒ remaining_sh_ 永不归零 ⇒ 300s 后看门狗 abort()。
            // 实测证据(NOTES §184):inrange=entered=127 < total=128,
            // 即 worker 眼里"范围内"的票只有 127 张,而倒计时按 128 计。
            // 注意 node_base_ 必须**仍**在奇数 store 之后读(陈旧票要落在 base 之前)。
            uint64_t gen = current_gen_.load(std::memory_order_relaxed) + 2;
            current_gen_.store(gen - 1, std::memory_order_release);   // 奇数:发布中
            total = 0;
            for (int n = 0; n < nnodes; ++n) {
                node_nj_[n] = job_counts[n];
                total += job_counts[n];
            }
            n_ = total;
            worker_limit_.store(limit >= nt_ ? 0 : limit, std::memory_order_relaxed);
            sharded_call_ = nnodes;          // visible to workers at anchor
            start_ = 0;
            for (int n = 0; n < nnodes; ++n)
                node_base_[n] = node_ticket_[n].v.load(std::memory_order_relaxed);
            remaining_sh_[sh_slot(gen)].store(total);   // 【模式A修】分片专用分桶倒计时
            shard_exec_.store(0);      // diag reset per call
            abandoned_.store(0);       // diag reset per call
            skipped_dec_.store(0);     // (已失效)diag reset per call
            underflow_.store(0);       // diag reset per call
            entered_.store(0);         // diag reset per call
            left_.store(0);            // diag reset per call
            inrange_.store(0);         // diag reset per call
            inrange2_.store(0);        // diag reset per call
            snap_retry_.store(0);      // diag reset per call
            snap_mismatch_.store(0);   // diag reset per call
            issued_.store(0);          // diag reset per call
            current_gen_.store(gen, std::memory_order_release);       // 偶数:就绪
        }
        cv_.notify_all();
        {
            std::unique_lock<std::mutex> lk(done_mtx_);
            // 发布已完成 ⇒ 此刻 current_gen_ 就是本次分片调用的偶数代 `gen`(发布块的局部
            // `gen` 不在本作用域内)。等待期间不可能有新一代发布(那需要本次先归零)。
            const int _gslot = sh_slot(current_gen_.load(std::memory_order_acquire));
            auto t0 = std::chrono::steady_clock::now();
            long dsec = 300;
            if (const char* se = std::getenv("XIAOTU_MOE_SHARD_WD")) {
                long v = std::atol(se); if (v > 0) dsec = v;
            }
            const auto deadline = t0 + std::chrono::seconds(dsec);
            bool late = !done_cv_.wait_until(lk, deadline, [&] {
                if (stop_) return true;
                return remaining_sh_[_gslot].load(std::memory_order_acquire) == 0;
            });
            if (late) {
                fprintf(stderr, "[pool] WATCHDOG(sharded) gen=%llu total=%zu rem=%zu exec=%ld\n",
                        (unsigned long long)current_gen_.load(), total,
                        remaining_sh_[_gslot].load(),
                        shard_exec_.load());
                fprintf(stderr, "  [判据] abandoned=%ld underflow=%ld  "
                        "entered=%ld left=%ld inrange=%ld inrange2=%ld  "
                        "snap_retry=%ld snap_mismatch=%ld\n",
                        abandoned_.load(), underflow_.load(),
                        entered_.load(), left_.load(),
                        inrange_.load(), inrange2_.load(),
                        snap_retry_.load(), snap_mismatch_.load());
                for (int n = 0; n < (int)node_nj_.size(); ++n) {
                    long done = (long)node_ticket_[n].v.load() - (long)node_base_[n];
                    fprintf(stderr, "  node %d: jobs=%zu pulled=%ld\n", n, node_nj_[n], done);
                }
                abort();
            }
            if (diag_active()) {
                size_t dup = 0, miss = 0;
                for (int n = 0; n < nnodes; ++n) {
                    for (size_t j = 0; j < node_nj_[n] && j < kShardDiagStride; ++j) {
                        uint32_t c = __atomic_load_n(
                            &shard_cnt_[(size_t)n * kShardDiagStride + j], __ATOMIC_RELAXED);
                        if (c == 0) ++miss;
                        else if (c > 1) ++dup;
                    }
                }
                // 【第 137 轮】原来只在 `dup || miss` 时才打印 ⇒ **健康调用从不打印**,
                // 于是"`abandoned` 的基线"从未被观测过(§199c:我此前只见过**崩溃态**的值,
                // 把它当成了基线)。现在**总是打印**,并加上 abandoned/inrange/inrange2
                // ⇒ 可以直接读健康态的收敛基线,把 §199(c) 的二选一变成事实。
                fprintf(stderr,
                        "[pool] SHARD-JOBDIAG gen=%llu total=%zu exec=%ld dup=%zu miss=%zu "
                        "abandoned=%ld inrange=%ld inrange2=%ld issued=%ld **unclassified=%ld**\n",
                        (unsigned long long)current_gen_.load(), total,
                        shard_exec_.load(), dup, miss,
                        abandoned_.load(), inrange_.load(), inrange2_.load(),
                        issued_.load(),
                        issued_.load() - inrange_.load() - inrange2_.load() - abandoned_.load());
            }
        }
        // NOTE: deliberately do NOT clear sharded_call_ here. A worker whose wake
        // is delayed past the end of this call would otherwise anchor into a
        // "flat-looking" generation with stale n_ and an empty task_ -> calling it
        // throws std::bad_function_call (fatal terminate in the worker thread).
        // Leaving sharded_call_ set means any late worker anchors as sharded and
        // finds its node's ticket range exhausted -> a clean no-op. The flat
        // parallel_for clears sharded_call_=0 when it dispatches (task_ is valid).
    }

    static size_t default_threads() {        unsigned hw = std::thread::hardware_concurrency();
        size_t nt = hw > 0 ? (size_t)hw : 1;
        if (const char* e = std::getenv("XIAOTU_MOE_THREADS")) {
            long v = std::atol(e);
            if (v > 0) nt = (size_t)v;
        }
        return nt;
    }

private:
    void start_workers() {
        topo_ = discover_numa_topology();
        // ---------------------------------------------------------------------
        // CCD-first core ordering (this host: EPYC 9654, SMT off, 2x12 CCDs,
        // 8 cores/CCD, NPS=4 -> 8 NUMA nodes, distance 10).
        //
        // Bandwidth experiment (NUMA_BANDWIDTH_CCD.md) showed a single CCD can
        // NOT saturate its IOD's DDR5 channels; you need all 12 CCDs of a socket
        // together (~30 GB/s at 1 CCD -> ~70-80 GB/s at 12 CCDs). So the core
        // ORDER matters: we must spread any task across all CCDs — one thread per
        // CCD before filling a second one — rather than consuming all 8 cores of
        // one CCD. We therefore build `cores_` in SLOT-major / CCD-minor order:
        //   cores_ = [ccd0.cpu0, ccd1.cpu0, ..., ccd23.cpu0,   <- 1 thread/CCD
        //             ccd0.cpu1, ccd1.cpu1, ..., ccd23.cpu1,   <- 2nd thread/CCD
        //             ... ]
        // so workers 0..23 land on 24 DISTINCT CCDs, 24..47 on a 2nd core of each
        // CCD, etc. CCD identity = L3 cache index (24 instances on this box).
        // ---------------------------------------------------------------------
        std::map<int, std::vector<int>> ccd_of_cpu;  // L3 id -> sorted cpus
        for (int cpu = 0; cpu < 1024; ++cpu) {
            char p[256];
            snprintf(p, sizeof(p),
                     "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
            std::ifstream f(p);
            if (!f.good()) break;                     // past the last real cpu
            char l3[256];
            snprintf(l3, sizeof(l3), "/sys/devices/system/cpu/cpu%d/cache/index3/id", cpu);
            std::ifstream fl3(l3); int ccd = 0;
            if (!fl3.good()) {                        // no index3: default to node
                auto n_it = topo_.cpu_node.find(cpu);
                ccd = (n_it != topo_.cpu_node.end()) ? n_it->second : 0;
            } else { fl3 >> ccd; }
            ccd_of_cpu[ccd].push_back(cpu);
        }
        cores_ = {};
        std::set<std::pair<int, int>> seen_core;      // dedup by (core_id, ccd)
        std::vector<int> ccd_ids;
        for (auto& kv : ccd_of_cpu) ccd_ids.push_back(kv.first);
        size_t max_slots = 0;
        for (int c : ccd_ids) max_slots = std::max(max_slots, ccd_of_cpu[c].size());
        for (size_t s = 0; s < max_slots; ++s) {      // slot-major, ccd-minor
            for (int c : ccd_ids) {
                const auto& lst = ccd_of_cpu[c];
                if (s >= lst.size()) continue;
                int cpu = lst[s];
                int coreid = -1; auto ct = topo_.cpu_core.find(cpu);
                if (ct != topo_.cpu_core.end()) coreid = ct->second;
                auto key = std::make_pair(coreid, c);
                if (coreid >= 0 && seen_core.count(key)) continue;
                seen_core.insert(key);
                cores_.push_back(cpu);
            }
        }
        if (cores_.empty()) {
            for (int c = 0; c < (int)nt_; ++c) cores_.push_back(c);
        }
        // Record which NUMA nodes at least one worker is pinned to (used to
        // validate node-scoped sharding: every sharded node must have a worker).
        node_present_ = 0;
        for (int cpu : cores_) { auto it = topo_.cpu_node.find(cpu);
            if (it != topo_.cpu_node.end()) node_present_ |= (1UL << it->second); }
        // Publish every worker's NUMA node HERE, before the thread is spawned:
        // the worker loop reads worker_node_[w] to pick its node-scoped ticket
        // queue, so a first call racing with thread startup would otherwise see
        // the default 0 and steal another node's jobs (dropping them).
        for (size_t w = 0; w < nt_; ++w) {
            int cpu = cores_[w % cores_.size()];
            worker_node_[w] = topo_.cpu_node.count(cpu) ? topo_.cpu_node[cpu] : 0;
        }
        // pin workers round-robin across the physical-core list
        for (size_t w = 0; w < nt_; ++w) {
            int cpu = cores_[w % cores_.size()];
            workers_.emplace_back([this, cpu, w] {
                pin_to(cpu);
                worker_loop(w);
            });
        }
        // Wait until every worker is running before the pool accepts work, so a
        // call issued immediately after construction cannot race with startup.
        while (ready_.load(std::memory_order_acquire) < nt_) std::this_thread::yield();
    }

    static void pin_to(int cpu) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
        sched_setaffinity(0, sizeof(set), &set);  // best effort
    }

    void worker_loop(size_t w) {
        ready_.fetch_add(1, std::memory_order_release);
        uint64_t my_last_gen = 0;  // per-worker: generation this thread handled
        for (;;) {
            void (*lf_)(void*, size_t) = nullptr;   void* lc_ = nullptr;
            void (*sf_)(void*, size_t, size_t) = nullptr; void* sc_ = nullptr;
            size_t n = 0, start = 0;
            uint64_t gen = 0; bool sharded = false;
            // --- HOT-RESTART SPIN (mirrors lktransformers' lock-free spin).
            // While a new generation arrives within spin_idle_us_ (decode hot
            // path: back-to-back parallel_for calls across MoE phases/layers),
            // keep this worker awake on the generation counter instead of
            // sleeping on the condvar. Adjacent phases then hand off with ZERO
            // futex syscalls. Only when the pool stays idle past the budget do we
            // fall back to the condvar (so an idle pool does not burn CPU).
            if (current_gen_.load(std::memory_order_acquire) != my_last_gen) {
                goto have_work;   // generation already pending: skip all sync
            }
            // Only workers that may claim tickets in a LIMITED call spin. Idle
            // spinners cost real throughput: with 192 spinning threads (2 per
            // physical core) the ~30 participants of a small decode call run at
            // half speed, and the caller's spin-wait competes for a core too.
            // Non-participants park on the condvar instead and are woken by the
            // caller only for unlimited (prefill) calls.
            {
                const size_t wl = worker_limit_.load(std::memory_order_relaxed);
                if (wl > 0 && wl < nt_) {
                    size_t stride = nt_ / wl;
                    if (stride < 1) stride = 1;
                    if (w % stride != 0) goto park;
                }
            }
            if (spin_idle_us_.load(std::memory_order_relaxed) > 0) {
                const uint64_t idle_us = spin_idle_us_.load(std::memory_order_relaxed);
                auto dl = std::chrono::steady_clock::now()
                        + std::chrono::microseconds(idle_us);
                while (std::chrono::steady_clock::now() < dl) {
                    uint64_t g = current_gen_.load(std::memory_order_acquire);
                    if (g != my_last_gen) goto have_work;
                    _mm_pause();
                }
            }
        park:
            {
                std::unique_lock<std::mutex> lk(work_mtx_);
                cv_.wait(lk, [&] { return stop_ || current_gen_.load() != my_last_gen; });
                if (stop_) return;
            }
        have_work:
            {
                // 【轮 69 去锚定锁】§113 实测:120 个 worker 每次换 generation 都抢全局
                // work_mtx_ ⇒ **每个并行区固定 ~113 µs**(空 body pfor 也是 113 µs)。
                // 轮 68 只做"读 gen → 读字段 → 复读 gen"的朴素 seqlock ⇒ 确定性挂死,
                // 因为调用方在"掀 gen"与"读 counter_ 写 start_"之间有个窗口,撕裂形态是
                // **新 gen + 旧 start_**(复读 gen 一致,查不出来)。
                // 现在 current_gen_ 是奇偶序号(见 parallel_for_impl):奇数 = 发布中。
                // worker 只接受**偶数**代,并复读确认 ⇒ 绝不接受半发布的调用。
                // 未抢到票的 worker 不参与 remaining_ 计数,错过一代是安全的。
                for (;;) {
                    const uint64_t s = current_gen_.load(std::memory_order_acquire);
                    if (s & 1u) { _mm_pause(); continue; }   // 奇数:调用方正在发布,等它写完
                    lf_ = __atomic_load_n(&pub_flat_, __ATOMIC_RELAXED);
                    lc_ = __atomic_load_n(&pub_flat_ctx_, __ATOMIC_RELAXED);
                    sf_ = __atomic_load_n(&pub_shard_, __ATOMIC_RELAXED);
                    sc_ = __atomic_load_n(&pub_shard_ctx_, __ATOMIC_RELAXED);
                    n = __atomic_load_n(&n_, __ATOMIC_RELAXED);
                    start = __atomic_load_n(&start_, __ATOMIC_RELAXED);
                    sharded = (__atomic_load_n(&sharded_call_, __ATOMIC_RELAXED) > 0);
                    std::atomic_thread_fence(std::memory_order_acquire);
                    if (current_gen_.load(std::memory_order_relaxed) == s) {
                        gen = s;
                        my_last_gen = s;
                        worker_gen_[w].store(s, std::memory_order_release);
                        break;
                    }
                }
            }
            // Worker-subset gate: a limited call lets only ~limit workers claim
            // tickets (stride selection keeps them spread across NUMA nodes). The
            // others claim nothing and simply re-arm, so they are not waited on.
            // 【第 180 轮·消融实验,仅 env 控制,绝不进交付配置】
            // 中性争用:每任务对一条共享原子做一次 RMW,只为复现 POOL_TRACE 的争用特征
            // (内容与任何判定无关)。目的:验证"额外的一处共享内存争用"是否足以抑制停顿。
            static std::atomic<long> s_ablate_cont{0};
            static const bool s_ablate_on = [] {
                const char* e = std::getenv("XIAOTU_MOE_ABLATE_CONTENTION");
                return e != nullptr && e[0] == '1';
            }();
            const size_t wlim = worker_limit_.load(std::memory_order_relaxed);
            if (wlim > 0 && wlim < nt_) {
                size_t stride = nt_ / wlim;
                if (stride < 1) stride = 1;
                if (w % stride != 0) continue;
            }
            if (sharded) {
                // ---- node-scoped single-copy sharded path: pull only from my node.
                const int myn = worker_node_[w];
                if (myn >= 0 && myn < sharded_call_) {
                    size_t base = 0, nj = 0, loc = 0;
                    for (;;) {
                        // 【第 128 轮修·读端 seqlock,正确顺序】**先取一致快照,再领票**。
                        // 第 127 轮我把复读校验放在 fetch_add **之后**,校验失败就 `continue`
                        // ⇒ **把已经领走的票静默丢弃**(既不执行也不递减),等于自己造了一个
                        // `remaining_` 少减点(实测 §188:inrange=126 < entered=127,总数仍 127<128)。
                        // 不变式:**`fetch_add` 之后绝不允许无归属地 continue/break 走掉**。
                        // 因此快照阶段(可能自旋重试)**不得领票**;只有拿到一致快照后才领票,
                        // 且领到的票要么走快路径执行+递减,要么交给下面的 re-anchor 分支重新归属。
                        size_t base2 = 0, nj2 = 0; uint64_t g2 = 0;
                        for (;;) {
                            g2 = current_gen_.load(std::memory_order_acquire);
                            if (g2 & 1) { snap_retry_.fetch_add(1, std::memory_order_relaxed); continue; }            // 奇数=发布中 ⇒ 自旋(此时不领票,不会丢)
                            base2 = node_base_[myn];
                            nj2 = node_nj_[myn];
                            if (current_gen_.load(std::memory_order_acquire) == g2) break;  // 一致快照
                            snap_retry_.fetch_add(1, std::memory_order_relaxed);          // 复读不一致 ⇒ 重试
                        }
                        if (g2 != gen) snap_mismatch_.fetch_add(1, std::memory_order_relaxed);
                        size_t t = node_ticket_[myn].v.fetch_add(1, std::memory_order_relaxed);
                        issued_.fetch_add(1, std::memory_order_relaxed);   // 无损账:领票即记
                        uint64_t g = current_gen_.load(std::memory_order_acquire);
                        if (g == gen && g2 == gen) {
                            base = base2;
                            nj = nj2;
                            loc = t - base;
                            if (loc >= nj) { abandoned_.fetch_add(1, std::memory_order_relaxed); break; }  // this node's jobs exhausted
                            inrange_.fetch_add(1, std::memory_order_relaxed);   // diag: 范围内的票被领走
                            if (diag_active() && loc < kShardDiagStride) {
                                __atomic_fetch_add(
                                    &shard_cnt_[(size_t)myn * kShardDiagStride + loc],
                                    1u, __ATOMIC_RELAXED);
                            }
                            if (sf_) {
                                entered_.fetch_add(1, std::memory_order_relaxed);
                                sf_(sc_, (size_t)myn, loc);
                                left_.fetch_add(1, std::memory_order_relaxed);
                            }
                            shard_exec_.fetch_add(1, std::memory_order_relaxed);
                        if (s_ablate_on) s_ablate_cont.fetch_add(1, std::memory_order_relaxed);
                            if (s_ablate_on) s_ablate_cont.fetch_add(1, std::memory_order_relaxed);
                            // 【模式A修】无条件递减到**本代自己的桶**。
                            // 旧代码是 `if (current_gen_ == gen && remaining_.fetch_sub(..))`,
                            // 守卫为假时递减被静默跳过 ⇒ 调用方永不归零(实测 skipped_dec==缺口)。
                            // 分桶后迟到递减只影响自己那代,不会污染下一代 ⇒ 守卫可安全删除。
                            // skipped_dec_ 保留为**不变式检查**:修好后应恒为 0,非 0 即新 bug。
                            if (remaining_sh_[sh_slot(gen)].load(std::memory_order_relaxed) == 0)
                                underflow_.fetch_add(1, std::memory_order_relaxed);
                            if (remaining_sh_[sh_slot(gen)].fetch_sub(
                                    1, std::memory_order_acq_rel) == 1) {
                                std::lock_guard<std::mutex> gl(done_mtx_);
                                done_cv_.notify_all();
                            }
                            continue;
                        }
                        // generation advanced: NEVER abandon the consumed ticket.
                        // Re-anchor in place and reconcile it against the LIVE call:
                        // the ticket is a valid job of the new generation iff it
                        // falls in that call's monotonic node range -> execute it
                        // with the LIVE task (the snapshot `stask` belongs to the
                        // previous call and would run the wrong function).
                        void (*nfn)(void*, size_t, size_t) = nullptr; void* nctx = nullptr;
                        uint64_t ng; size_t nb, nnj; bool live;
                        {
                            std::lock_guard<std::mutex> lk(work_mtx_);
                            if (stop_) return;
                            ng = current_gen_.load();
                            live = (sharded_call_ > 0) && (int)myn < sharded_call_;
                            nb = live ? node_base_[myn] : 0;
                            nnj = live ? node_nj_[myn] : 0;
                            nfn = pub_shard_; nctx = pub_shard_ctx_;
                        }
                        if (!live) {                   // switched away from sharded:
                            abandoned_.fetch_add(1, std::memory_order_relaxed);
                            break;                     // outer wait will re-anchor flat
                        }
                        gen = ng; my_last_gen = ng;
                        worker_gen_[w].store(ng, std::memory_order_release);
                        sf_ = nfn; sc_ = nctx;
                        base = nb; nj = nnj;
                        loc = t - base;
                        if (loc >= nj) { abandoned_.fetch_add(1, std::memory_order_relaxed); break; }  // beyond live range -> re-arm outer wait
                        inrange2_.fetch_add(1, std::memory_order_relaxed);   // diag: re-anchor 分支的范围内票
                        if (sf_) {
                            entered_.fetch_add(1, std::memory_order_relaxed);
                            sf_(sc_, (size_t)myn, loc);
                            left_.fetch_add(1, std::memory_order_relaxed);
                        }
                        shard_exec_.fetch_add(1, std::memory_order_relaxed);
                        if (s_ablate_on) s_ablate_cont.fetch_add(1, std::memory_order_relaxed);
                        // 【模式A修】同 fast path:无条件递减到本代自己的桶(见上)。
                        if (remaining_sh_[sh_slot(gen)].fetch_sub(
                                1, std::memory_order_acq_rel) == 1) {
                            std::lock_guard<std::mutex> gl(done_mtx_);
                            done_cv_.notify_all();
                        }
                        continue;
                    }
                }
                // workers on a non-participating node stay idle for this generation
                continue;
            }
            for (;;) {
                size_t t = counter_.fetch_add(1, std::memory_order_relaxed);
                size_t i = t - start;
                uint64_t g = current_gen_.load(std::memory_order_acquire);
                if (g == gen) {
                    // Snapshot is authoritative: caller has not started the next
                    // call (publish-first), so start_/n_/task_ unchanged.
                    if (i >= n) break;   // true future-gap -> safe lock-free drop
                    if (lf_) lf_(lc_, i);
                    if (diag_active()) proc_vec_[i] = 1;
                    // 【第 126 轮修】原为 `if (current_gen_ != gen) continue;` —— 世代前进就
                    // **跳过递减**,而票已经从 counter_ 领走 ⇒ flat 调用方永远等不到 0
                    // (实测:`WATCHDOG fired: gen=35800 n=2 start=43431 end=43433
                    //  counter=43553 remaining=1 dropped=0`,即 2 个任务只减了 1 次)。
                    // 安全性论证:调用方只有在 remaining_==0 之后才会发布下一代并 store 新的 n,
                    // 因此只要本票还没递减,调用方必然仍在等待、remaining_ 仍是本代的值
                    // ⇒ **无条件递减不会污染下一代**,守卫是多余且有害的。
                    if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        std::lock_guard<std::mutex> gl(done_mtx_);
                        done_cv_.notify_all();
                    }
                    continue;
                }
                // Generation advanced: re-anchor to the current LIVE call under
                // the lock (consistent start_/n_/task_), then evaluate t against
                // the authoritative live range.
                {
                    void (*nlf)(void*, size_t) = nullptr; void* nlc = nullptr;
                    size_t nn, ns; uint64_t ng; bool live_sharded;
                    {
                        std::lock_guard<std::mutex> lk(work_mtx_);
                        if (stop_) return;
                        ng = current_gen_.load();
                        live_sharded = (sharded_call_ > 0);
                        nlf = pub_flat_; nlc = pub_flat_ctx_;
                        nn = n_;
                        ns = start_;
                    }
                    // The live call may be SHARDED: its jobs are claimed from the
                    // per-node counters and `task_` is stale, so executing the flat
                    // task here would run the wrong function and still decrement
                    // the sharded call's remaining_ (dropping its real jobs).
                    // Re-arm instead: the outer loop re-anchors with `sharded`
                    // read correctly.
                    if (live_sharded) {
                        my_last_gen = ng;
                        worker_gen_[w].store(ng, std::memory_order_release);
                        // 【第 126 轮修】票已经从 counter_ 领走(本代的),这里若直接 break
                        // 就会**丢掉一次递减** ⇒ 本代 remaining_ 永不归零。
                        // 必须先把本代的那一次递减补上,再回去重新 arm。
                        // (这是 §183b 记录的第二处同型缺陷。)
                        if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                            std::lock_guard<std::mutex> gl(done_mtx_);
                            done_cv_.notify_all();
                        }
                        break;
                    }
                    gen = ng;
                    my_last_gen = ng;
                    worker_gen_[w].store(ng, std::memory_order_release);
                    lf_ = nlf; lc_ = nlc;
                    n = nn; start = ns;
                    i = t - start;
                    if (i >= n) {               // genuinely beyond live range -> re-arm
                        dropped_.fetch_add(1, std::memory_order_relaxed);
                        break;
                    }
                    if (lf_) lf_(lc_, i);
                    if (diag_active()) proc_vec_[i] = 1;
                    // Lock-protected notify (same reasoning as the fast path): hold
                    // done_mtx_ so the completion notify can't be missed by the
                    // caller's pred-check -> condvar-wait transition. Guarded on the
                    // live generation like the fast path.
                    if (current_gen_.load(std::memory_order_acquire) != gen) continue;
                    if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        std::lock_guard<std::mutex> gl(done_mtx_);
                        done_cv_.notify_all();
                    }
                }
            }
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(work_mtx_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& th : workers_) if (th.joinable()) th.join();
    }

    size_t nt_;
    NumaTopology topo_;
    std::vector<int> cores_;

    static bool pool_dbg() {
        static const int v = []{ const char* e = std::getenv("XIAOTU_MOE_POOL_DBG");
            return e && std::atoi(e) > 0 ? 1 : 0; }();
        return v != 0;
    }

    // True when per-call processed-bitset should be maintained (pool debug or
    // stall-trace mode). Cheap enough for diagnosis; off in production.
    static bool diag_active() {
        if (pool_dbg()) return true;
        static const int v = []{ const char* e = std::getenv("XIAOTU_MOE_POOL_TRACE");
            return e && std::atoi(e) > 0 ? 1 : 0; }();
        return v != 0;
    }

    // work dispatch
    std::mutex work_mtx_;
    std::condition_variable cv_;
    std::mutex call_mtx_;  // serializes parallel_for entry (shared-pool safety)
    std::function<void(size_t)> task_;
    size_t n_;
    alignas(64) std::atomic<size_t> counter_;
    size_t start_ = 0;                 // first ticket index of current call (monotonic)
    alignas(64) std::atomic<size_t> remaining_;  // outstanding work items in current call (countdown barrier)
    alignas(64) std::atomic<uint64_t> current_gen_;
    // Hot-restart spin budget (us). While a new parallel_for generation arrives
    // within this window (decode hot path), workers stay awake spinning on the
    // generation counter and the caller spin-waits completion, avoiding the
    // futex/condvar wake-storm on every phase -> layer. 0 disables spin entirely
    // (legacy behavior). Mirrors lktransformers' lock-free status spin.
    std::atomic<uint64_t> spin_idle_us_{5000};
    // >0: only ~this many workers may claim tickets in the current call (0 = all).
    std::atomic<size_t> worker_limit_{0};
    std::atomic<size_t> dropped_{0};  // diagnostic: tickets dropped (future-gap)
    std::atomic<size_t> ready_{0};    // workers that reached their loop
    // diagnostic: per-(node, local job) execution counts for sharded calls
    static constexpr size_t kShardDiagStride = 4096;
    std::vector<uint32_t> shard_cnt_;
    std::vector<unsigned char> proc_vec_;  // diagnostic processed-bitset
    // per-worker completion slots: worker[w] writes only worker_gen_[w].
    std::unique_ptr<std::atomic<uint64_t>[]> worker_gen_;
    bool stop_;

    // completion barrier
    std::mutex done_mtx_;
    std::condition_variable done_cv_;

    std::vector<std::thread> workers_;
    std::vector<int> worker_node_;    // per-worker pinned node [w]
    unsigned long node_present_ = 0;  // bitset: nodes that have >=1 worker

    // --- node-scoped (single-copy sharded) scheduling state -----------------
    // parallel_for_sharded splits a call into per-node job lists; each node's
    // workers pull only from their OWN node's ticket counter so every job is
    // executed by a worker bound to the node that owns the weight rows it reads
    // (lktransformers intra-node model). Mirrors the flat parallel_for state.
    // ---- 无锁任务发布(2026-09-11)------------------------------------------
    // 原实现:每个 worker 每次拿任务都要在全局 work_mtx_ 下 `local_task = task_;
    // stask = sharded_task_;` —— 120 个 worker × 每层 3 个阶段 ⇒ 大量锁竞争 +
    // 每个 std::function 拷贝都可能是堆分配。实测把矩阵计算体挖空后,
    // "纯 job 分解 + 屏障"仍要 614 µs/次调用(占 41%)。
    // 现在改成发布 (fn, ctx) 两个机器字:调用方在 `++current_gen_` 之前写入,
    // worker 在确认了新一代之后直接读(不加锁、不拷贝、不分配)。
    template <typename F> static void flat_thunk(void* c, size_t i) {
        (*static_cast<F*>(c))(i);
    }
    template <typename F> static void shard_thunk(void* c, size_t n, size_t j) {
        (*static_cast<F*>(c))(n, j);
    }
    void (*pub_flat_)(void*, size_t) = nullptr;   void* pub_flat_ctx_ = nullptr;
    void (*pub_shard_)(void*, size_t, size_t) = nullptr; void* pub_shard_ctx_ = nullptr;

    std::function<void(size_t, size_t)> sharded_task_;  // fn(node, local)
    int sharded_call_ = 0;                              // #nodes if current call is sharded
    alignas(64) std::atomic<long> shard_exec_{0};                   // diag: actual stask executions
    // 【第 123 轮修·模式 A】分片路径按**世代奇偶**分桶的倒计时。
    // 原来分片路径与 flat 路径共用 `remaining_`,且在递减处加了"再读一次 current_gen_"的守卫
    // —— 守卫为假时递减被**静默跳过**,而该票据所属代的调用方**已把它计入 total**
    // ⇒ 调用方永远等不到 0 ⇒ 看门狗 `abort()`。实测证据(NOTES §179/§180):三份 dump 里
    // `skipped_dec` 与缺口**精确相等**(exec=total, rem=1, skipped_dec=1)。
    // 分桶后,属于第 g 代的递减**只能**落到 `remaining_sh_[sh_slot(g)]`,而 g+2 代用**另一个**
    // 桶 ⇒ **结构上不可能**污染下一代 ⇒ 守卫不再需要,可改为"领票即必减"。
    // 桶复用安全性:g+4 的 store(total) 只在 g+2 归零后才执行,而 g 代的递减在 g+2 归零前
    // 必已全部完成(g+2 的发布以 g 归零为前提)⇒ 复用安全。票据由 fetch_add 唯一领取 ⇒ 不漏减不多减。
    // 注意 `gen` 恒为**偶数**(就绪态),所以槽位取 (gen>>1)&1,**不能**用 gen&1(恒 0)。
    alignas(64) std::atomic<size_t> remaining_sh_[2] = {};
    static int sh_slot(uint64_t gen) { return (int)((gen >> 1) & 1ULL); }
    // 【第 123 轮纯诊断】不动任何控制流,只计数,用来区分"丢票"与"递减被世代守卫跳过"。
    // 崩溃签名是 exec==total 而 rem==1/2,而每个 exec 后都紧跟一个带守卫的递减
    // ⇒ 必然有一处 break 丢弃了已领票据,或有一处守卫把递减跳过了。谁非零即定位。
    alignas(64) std::atomic<long> abandoned_{0};                    // diag: 已领票后在 break 处被丢弃
    // 【第 122 轮·修埋点】`skipped_dec_` 已失效:第 121 轮删世代守卫时把它的自增一并删了,
    // 于是它结构上恒为 0、再无信息量。换成真正的不变式检查 `underflow_`:递减前桶值已是 0
    // ⇒ 说明发生了"多减/重复减"(那会让调用方提前归零)。
    std::atomic<long> skipped_dec_{0};                  // (已失效,保留仅为兼容 dump 字段)
    alignas(64) std::atomic<long> underflow_{0};                    // diag: 递减前桶值已为 0(多减/重复减)
    // 【第 122 轮·模式 B 直接证据】任务体入口/出口计数。若某 worker 领了票却卡在
    // `sf_(...)` 里面,看门狗触发时 `entered - left` 就等于"进了没出"的 worker 数。
    alignas(64) std::atomic<long> entered_{0};
    alignas(64) std::atomic<long> left_{0};
    // 【第 124 轮】"本代范围内"的票被领走的次数。与 `entered_` 比较即可二分:
    //   inrange == entered + 1 ⇒ 有一张范围内的票在 fast path 内消失了(继续查分支内);
    //   inrange == entered     ⇒ 那张票**根本没被领** ⇒ node_base_/node_nj_ 发布与 worker
    //                            读取之间撕裂(票号区间与实际 job 数不一致)。
    alignas(64) std::atomic<long> inrange_{0};
    // 【第 133 轮】re-anchor 分支里"范围内"的票(与快路径的 `inrange_` 分开计)。
    // 这样 `inrange_ + inrange2_` 与 `total` 一比即可判定丢票性质:
    //   两和 == total  ⇒ 票都被领且都执行了(问题在递减);
    //   两和 <  total  ⇒ **有票从未被领** ⇒ 发布/读取仍有残余(与 §184 同型)。
    alignas(64) std::atomic<long> inrange2_{0};
    // 【第 135 轮·只读诊断】快照自旋的流量。用于回答 §197 的遗留问题:
    // 那张票是否卡在"快照反复重试、始终不满足 g2 == 缓存 gen"的路径上。
    //   snap_retry_    : 快照循环重试次数(奇数代 / 复读不一致)
    //   snap_mismatch_ : 快照成功但 g2 != worker 缓存的 gen(⇒ 走 re-anchor)
    alignas(64) std::atomic<long> snap_retry_{0};
    alignas(64) std::atomic<long> snap_mismatch_{0};
    // 【第 138 轮·无损账】领票后**立刻**自增。此后无论走哪条分支都必须恰好一次计入
    // inrange_/inrange2_/abandoned_ 之一 ⇒ `issued - inrange - inrange2 - abandoned`
    // 就是"领了票却没有被分类"的票数(健康态应为 0)。用于回答:那张票到底去哪了。
    alignas(64) std::atomic<long> issued_{0};
    static constexpr size_t kMaxNodeShards = 128;       // ample for any EPYC topology
    // 【轮 76】每个 node 的票号计数器**各自独占一条 cacheline**。原来 8 个计数器挤在
    // 同一条线上,120 个 worker 的 fetch_add 让这条线在 core 之间来回弹(伪共享),
    // 是去锁后每个并行区仍剩 ~17 µs 固定成本的主要嫌疑。纯布局改动,零语义变化。
    struct alignas(64) PaddedTicket {
        std::atomic<size_t> v{0};
        char _pad[64 - sizeof(std::atomic<size_t>)];
    };
    std::unique_ptr<PaddedTicket[]> node_ticket_;  // per-node monotonic counters (padded)
    std::vector<size_t> node_base_;                     // per-node first ticket this call
    std::vector<size_t> node_nj_;                       // per-node job count this call
};

// ---------------------------------------------------------------------------
// Shared process-wide NUMA pool.
//
// One persistent worker pool for the whole process, shared by every MOE_V2
// layer. This mirrors lk_moe's architecture: a single shared CPU MoE engine
// (Backend_NUMA, LK_THREADS workers) is reused across all 61 layers, instead
// of giving each layer its own pool (which would balloon to ~61 x threads and
// thrash the scheduler). The C++11 magic-static guarantees thread-safe
// one-time construction; the worker count is set by XIAOTU_MOE_THREADS (or the
// hardware concurrency) exactly once.
// ---------------------------------------------------------------------------
inline NumaWorkPool& shared_numa_pool() {
    static NumaWorkPool pool;
    return pool;
}

} // namespace xiaotu_moe

#endif // XIAOTU_MOE_NUMA_POOL_HPP

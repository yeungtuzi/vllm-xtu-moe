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
    long rc = syscall(SYS_set_mempolicy, MPOL_INTERLEAVE, &mask, sizeof(mask) * 8);
    if (rc == 0) {
        // Fault every page in on this socket (thread policy is INTERLEAVE over
        // just this socket's nodes). volatile forces the stores.
        volatile char* cp = static_cast<volatile char*>(p);
        const size_t PS = 4096;
        for (size_t off = 0; off < bytes; off += PS) cp[off] = 0;
        syscall(SYS_set_mempolicy, MPOL_DEFAULT, nullptr, 0);
    } else {
        // policy failed: touch anyway (default placement) to keep the region valid
        volatile char* cp = static_cast<volatile char*>(p);
        const size_t PS = 4096;
        for (size_t off = 0; off < bytes; off += PS) cp[off] = 0;
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
        node_ticket_.reset(new std::atomic<size_t>[kMaxNodeShards]);
        for (size_t n = 0; n < kMaxNodeShards; ++n) node_ticket_[n].store(0);
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
            task_ = std::function<void(size_t)>(fn);  // type-erased copy
            n_ = n;
            worker_limit_.store(limit >= nt_ ? 0 : limit, std::memory_order_relaxed);
            sharded_call_ = 0;   // this is a flat call: task_ is valid, so reset
                                 // any stale sharded marker so late workers anchor flat.
            // MONOTONIC ticket counter (never reset). Each call occupies the
            // ticket range [start_+0, start_+n). Workers bound to THIS call
            // compute  i = ticket - start_  and only run for i in [0,n); any
            // ticket >= start_+n means the worker's own range is exhausted and
            // it re-arms for the next generation.
            //
            // PUBLISH ORDER: current_gen_ is bumped BEFORE start_/remaining_ are
            // written. A worker fast-path sees current_gen_==its snapshot gen only
            // while the caller has not yet started the next call, so start_ still
            // equals its snapshot start -> a range-exhausted drop there is a TRUE
            // future-gap (safe, no lock). Once current_gen_ advances, workers
            // re-anchor under the lock and read a consistent (start_,n_) pair.
            gen = ++current_gen_;        // open a new generation (publish FIRST)
            start_ = counter_.load();
            remaining_.store(n);      // outstanding work items in this call
            // diagnostic processed-bitset for this call (only when debug/trace on)
            if (diag_active()) proc_vec_.assign(n, 0);
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
            sharded_task_ = std::function<void(size_t, size_t)>(fn);
            for (int n = 0; n < nnodes; ++n) {
                node_nj_[n] = job_counts[n];
                node_base_[n] = 0;                   // reset: base is always 0
                node_ticket_[n].store(0);            // fresh per-call counter range [0,nj)
                total += job_counts[n];
            }
            n_ = total;
            worker_limit_.store(limit >= nt_ ? 0 : limit, std::memory_order_relaxed);
            uint64_t gen = ++current_gen_;   // publish first (workers anchor under lock)
            sharded_call_ = nnodes;          // visible to workers at anchor
            start_ = 0;
            remaining_.store(total);
            shard_exec_.store(0);      // diag reset per call
        }
        cv_.notify_all();
        {
            std::unique_lock<std::mutex> lk(done_mtx_);
            auto t0 = std::chrono::steady_clock::now();
            long dsec = 300;
            if (const char* se = std::getenv("XIAOTU_MOE_SHARD_WD")) {
                long v = std::atol(se); if (v > 0) dsec = v;
            }
            const auto deadline = t0 + std::chrono::seconds(dsec);
            bool late = !done_cv_.wait_until(lk, deadline, [&] {
                if (stop_) return true;
                return remaining_.load(std::memory_order_acquire) == 0;
            });
            if (late) {
                fprintf(stderr, "[pool] WATCHDOG(sharded) gen=%llu total=%zu rem=%zu exec=%ld\n",
                        (unsigned long long)current_gen_.load(), total, remaining_.load(),
                        shard_exec_.load());
                for (int n = 0; n < (int)node_nj_.size(); ++n) {
                    long done = (long)node_ticket_[n].load() - (long)node_base_[n];
                    fprintf(stderr, "  node %d: jobs=%zu pulled=%ld\n", n, node_nj_[n], done);
                }
                abort();
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
        // pin workers round-robin across the physical-core list
        for (size_t w = 0; w < nt_; ++w) {
            int cpu = cores_[w % cores_.size()];
            workers_.emplace_back([this, cpu, w] {
                pin_to(cpu);
                worker_node_[w] = topo_.cpu_node.count(cpu) ? topo_.cpu_node[cpu] : 0;
                worker_loop(w);
            });
        }
    }

    static void pin_to(int cpu) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
        sched_setaffinity(0, sizeof(set), &set);  // best effort
    }

    void worker_loop(size_t w) {
        uint64_t my_last_gen = 0;  // per-worker: generation this thread handled
        for (;;) {
            std::function<void(size_t)> local_task;
            std::function<void(size_t, size_t)> stask;
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
                std::lock_guard<std::mutex> lk(work_mtx_);
                gen = current_gen_.load();
                my_last_gen = gen;
                worker_gen_[w].store(gen, std::memory_order_release);
                local_task = task_;   // copy the type-erased function
                stask = sharded_task_;
                n = n_;
                start = start_;       // this call's first ticket
                sharded = (sharded_call_ > 0);
            }
            // Worker-subset gate: a limited call lets only ~limit workers claim
            // tickets (stride selection keeps them spread across NUMA nodes). The
            // others claim nothing and simply re-arm, so they are not waited on.
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
                    size_t base = node_base_[myn], nj = node_nj_[myn];
                    for (;;) {
                        size_t t = node_ticket_[myn].fetch_add(1, std::memory_order_relaxed);
                        size_t loc = t - base;
                        uint64_t g = current_gen_.load(std::memory_order_acquire);
                        if (g == gen) {
                            if (loc >= nj) break;      // this node's jobs exhausted
                            stask((size_t)myn, loc);
                            shard_exec_.fetch_add(1, std::memory_order_relaxed);
                            if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                                std::lock_guard<std::mutex> gl(done_mtx_);
                                done_cv_.notify_all();
                            }
                            continue;
                        }
                        // generation advanced: NEVER abandon the consumed ticket.
                        // Re-anchor in place and reconcile it against the LIVE call:
                        // with per-call base==0, if the ticket is in the live node
                        // range it is a valid job of the new generation -> execute it.
                        // (The flat parallel_for does the same; abandoning would let a
                        // stale worker consume a current-gen ticket and skip its job.)
                        uint64_t ng; size_t nb, nnj; bool live;
                        {
                            std::lock_guard<std::mutex> lk(work_mtx_);
                            if (stop_) return;
                            ng = current_gen_.load();
                            live = (sharded_call_ > 0) && (int)myn < sharded_call_;
                            nb = live ? node_base_[myn] : 0;
                            nnj = live ? node_nj_[myn] : 0;
                        }
                        if (!live) {                   // switched away from sharded:
                            break;                     // outer wait will re-anchor flat
                        }
                        gen = ng; my_last_gen = ng;
                        worker_gen_[w].store(ng, std::memory_order_release);
                        base = nb; nj = nnj;
                        loc = t - base;                // base==0: loc==t
                        if (loc >= nj) break;          // beyond live range -> re-arm outer wait
                        stask((size_t)myn, loc);
                        shard_exec_.fetch_add(1, std::memory_order_relaxed);
                        if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
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
                    local_task(i);
                    if (diag_active()) proc_vec_[i] = 1;
                    // Last worker to reach 0 notifies the completion condvar.
                    // It must hold done_mtx_ (the same mutex the caller's
                    // predicate-wait runs under) so the notify can never fall in
                    // the caller's pred-check -> condvar-wait window (lost wakeup).
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
                    std::function<void(size_t)> nt;
                    size_t nn, ns; uint64_t ng;
                    {
                        std::lock_guard<std::mutex> lk(work_mtx_);
                        if (stop_) return;
                        ng = current_gen_.load();
                        nt = task_;
                        nn = n_;
                        ns = start_;
                    }
                    gen = ng;
                    my_last_gen = ng;
                    worker_gen_[w].store(ng, std::memory_order_release);
                    local_task = std::move(nt);
                    n = nn; start = ns;
                    i = t - start;
                    if (i >= n) {               // genuinely beyond live range -> re-arm
                        dropped_.fetch_add(1, std::memory_order_relaxed);
                        break;
                    }
                    local_task(i);
                    if (diag_active()) proc_vec_[i] = 1;
                    // Lock-protected notify (same reasoning as the fast path): hold
                    // done_mtx_ so the completion notify can't be missed by the
                    // caller's pred-check -> condvar-wait transition.
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
    std::atomic<size_t> counter_;
    size_t start_ = 0;                 // first ticket index of current call (monotonic)
    std::atomic<size_t> remaining_;  // outstanding work items in current call (countdown barrier)
    std::atomic<uint64_t> current_gen_;
    // Hot-restart spin budget (us). While a new parallel_for generation arrives
    // within this window (decode hot path), workers stay awake spinning on the
    // generation counter and the caller spin-waits completion, avoiding the
    // futex/condvar wake-storm on every phase -> layer. 0 disables spin entirely
    // (legacy behavior). Mirrors lktransformers' lock-free status spin.
    std::atomic<uint64_t> spin_idle_us_{5000};
    // >0: only ~this many workers may claim tickets in the current call (0 = all).
    std::atomic<size_t> worker_limit_{0};
    std::atomic<size_t> dropped_{0};  // diagnostic: tickets dropped (future-gap)
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
    std::function<void(size_t, size_t)> sharded_task_;  // fn(node, local)
    int sharded_call_ = 0;                              // #nodes if current call is sharded
    std::atomic<long> shard_exec_{0};                   // diag: actual stask executions
    static constexpr size_t kMaxNodeShards = 128;       // ample for any EPYC topology
    std::unique_ptr<std::atomic<size_t>[]> node_ticket_;  // per-node monotonic counters
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

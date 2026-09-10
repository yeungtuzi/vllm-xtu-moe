// 机器内存带宽上限基准 —— 用来判断"我们的 MoE 权重流式读取"离硬件上限有多远。
//
// 背景(report/tuning/NOTES.md §35):单卡 256K 配置下,每层 MoE 要流 36 个
// (row,expert) 对 × 12.6 MB fp4 专家权重 = 454 MB,实测 1.94 ms ⇒ 234 GB/s。
// 本机 EPYC 9654 / NPS=4(8 节点,每节点 3 通道)理论 ~115 GB/s/节点、~920 GB/s 整机。
// 若本基准跑出的实际上限也只有 ~250 GB/s,那我们的内核已经贴上限,唯一出路是
// 减少字节数(更少的 draft token / 更小的 k);反之说明内核有 2-3x 空间。
//
// 编译/运行(与 sweep 分开跑,避免互相污染):
//   gcc -O2 -mavx512f -pthread -o /tmp/bwprobe scripts/bwprobe.c
//   T=1,24,192 /tmp/bwprobe          # 线程数扫描
//   BIND=node /tmp/bwprobe           # 线程绑到各自节点、只读本节点分片(模拟 shard 布局)
//   BIND=flat /tmp/bwprobe           # 不绑定、整块连续(模拟单拷贝布局)
//
// License: Apache-2.0
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef GB
#define GB 8.0
#endif

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

typedef struct { size_t begin, end; int cpu; double gbs; } Arg;

// 顺序读并累加(每行 64 字节取一个数,足够让 HW prefetcher 工作,
// 不做任何会掩盖带宽的额外运算)。AVX-512: 每迭代读 4 个 cache line。
static void* worker(void* p) {
    Arg* a = (Arg*)p;
    if (a->cpu >= 0) {
        cpu_set_t s; CPU_ZERO(&s); CPU_SET(a->cpu, &s);
        sched_setaffinity(0, sizeof(s), &s);
    }
    const uint8_t* base = (const uint8_t*)0;
    (void)base;
    double t0 = now_s();
    // 通过全局指针传入缓冲区(pthread 参数只带区间)
    extern uint8_t* g_buf;
    const uint64_t* q = (const uint64_t*)(g_buf + a->begin);
    size_t n = (a->end - a->begin) / 64;   // 每条 64B 取 1 个 u64
    uint64_t acc = 0;
    for (size_t i = 0; i < n; ++i) acc += q[i * 8];
    double dt = now_s() - t0;
    a->gbs = (double)(a->end - a->begin) / 1e9 / dt;
    return (void*)(uintptr_t)(acc & 1);
}

uint8_t* g_buf = NULL;

int main(void) {
    const char* ts = getenv("T");
    const char* bind = getenv("BIND");
    int nthr = ts ? atoi(ts) : 24;
    int flat = !(bind && strcmp(bind, "node") == 0);
    size_t bytes = (size_t)(GB * 1e9);
    bytes &= ~(size_t)63;
    g_buf = aligned_alloc(4096, bytes);
    memset(g_buf, 1, bytes);

    int ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    pthread_t th[512];
    Arg args[512];
    if (nthr > ncpu) nthr = ncpu;

    for (int i = 0; i < nthr; i++) {
        size_t per = bytes / (size_t)nthr;
        args[i].begin = (size_t)i * per;
        args[i].end = (i == nthr - 1) ? bytes : args[i].begin + per;
        // node 模式:把线程逐个钉到物理核上(和引擎的 CCD-slot-major 顺序一致),
        // 让"每个线程只读自己附近的分片"这一布局的带宽上限可测。
        args[i].cpu = flat ? -1 : (i < ncpu ? i : -1);
        args[i].gbs = 0;
        pthread_create(&th[i], NULL, worker, &args[i]);
    }
    double sum = 0; double t0 = now_s();
    for (int i = 0; i < nthr; i++) { pthread_join(th[i], NULL); sum += args[i].gbs; }
    double dt = now_s() - t0;
    printf("[bwprobe] threads=%3d layout=%-4s size=%.1fGB wall=%.3fs aggregate=%7.1f GB/s per-thread=%5.2f GB/s\n",
           nthr, flat ? "flat" : "node", GB, dt, (double)bytes / 1e9 / dt, sum / nthr);
    free(g_buf);
    return 0;
}

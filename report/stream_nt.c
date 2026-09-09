/* STREAM variants to find this box's real DRAM bandwidth.
 *
 * Plain STREAM Triad counts 3N bytes (2 reads + 1 write) but a normal write
 * triggers read-for-ownership, so the DRAM actually moves 4N. Using AVX-512
 * non-temporal stores (_mm512_stream_pd) removes the RFO, so the counted 3N
 * matches the real traffic and the reported number approaches the true limit.
 *
 * modes: copy / scale / triad / triad_nt / read-only (sum reduction)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <omp.h>
#include <immintrin.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static void *alloc_huge(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 2*1024*1024, bytes) != 0) return NULL;
    if (getenv("NO_HUGEPAGE") == NULL) madvise(p, bytes, MADV_HUGEPAGE);
    return p;
}

int main(int argc, char **argv) {
    size_t N = (argc > 1) ? strtoull(argv[1], 0, 10) : 200000000;
    int nt = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    double *a = alloc_huge(N * 8), *b = alloc_huge(N * 8), *c = alloc_huge(N * 8);
    if (!a || !b || !c) { fprintf(stderr, "alloc fail\n"); return 1; }
#pragma omp parallel for simd num_threads(nt) schedule(static)
    for (size_t i = 0; i < N; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }

    double q = 3.0, best[5] = {0, 0, 0, 0, 0};
    for (int r = 0; r < 5; r++) {
        double t0 = now();
#pragma omp parallel for simd num_threads(nt) schedule(static)
        for (size_t i = 0; i < N; i++) c[i] = a[i];
        double g = N * 8.0 / (now() - t0) / 1e9; if (g > best[0]) best[0] = g;

        t0 = now();
#pragma omp parallel for simd num_threads(nt) schedule(static)
        for (size_t i = 0; i < N; i++) c[i] = q * a[i];
        g = N * 8.0 * 2 / (now() - t0) / 1e9; if (g > best[1]) best[1] = g;

        t0 = now();
#pragma omp parallel for simd num_threads(nt) schedule(static)
        for (size_t i = 0; i < N; i++) a[i] = b[i] + q * c[i];
        g = N * 8.0 * 3 / (now() - t0) / 1e9; if (g > best[2]) best[2] = g;

        /* Triad with non-temporal stores: real traffic == counted 3N */
        t0 = now();
#pragma omp parallel num_threads(nt)
        {
            int tid = omp_get_thread_num();
            size_t lo = (size_t)((double)tid / nt * N) & ~7UL;
            size_t hi = (size_t)((double)(tid + 1) / nt * N) & ~7UL;
            for (size_t i = lo; i < hi; i += 8) {
                __m512d vb = _mm512_load_pd(&b[i]);
                __m512d vc = _mm512_load_pd(&c[i]);
                __m512d va = _mm512_fmadd_pd(_mm512_set1_pd(q), vc, vb);
                _mm512_stream_pd(&a[i], va);
            }
        }
        g = N * 8.0 * 3 / (now() - t0) / 1e9; if (g > best[3]) best[3] = g;

        /* read-only: pure read bandwidth (2 streams) */
        double sink = 0;
        t0 = now();
#pragma omp parallel for simd num_threads(nt) schedule(static) reduction(+ : sink)
        for (size_t i = 0; i < N; i++) sink += a[i] + b[i];
        g = N * 8.0 * 2 / (now() - t0) / 1e9; if (g > best[4]) best[4] = g;
        if (sink < 0) printf(" ");
    }
    printf("threads=%3d N=%zu hugepage=%s  Copy %6.1f  Scale %6.1f  Triad %6.1f  "
           "Triad_NT %6.1f  Read %6.1f GB/s\n",
           nt, N, getenv("NO_HUGEPAGE") ? "no" : "yes",
           best[0], best[1], best[2], best[3], best[4]);
    return 0;
}

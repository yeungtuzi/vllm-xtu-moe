/* STREAM with explicit AVX-512 width control + huge pages.
 *
 * Why this exists: `gcc -march=native` on this box (GCC 11.4) resolves to
 * znver3 and defaults to -mprefer-vector-width=128, so the plain STREAM runs
 * with 128-bit vectors and 4 KiB pages -> far below the machine's DRAM
 * bandwidth. This version makes the two effects switchable at compile time:
 *   -mprefer-vector-width=512   (512-bit loads/stores)
 *   MADV_HUGEPAGE               (2 MiB pages; THP is 'madvise' on this host)
 * and reports the best of N repetitions.
 *
 * Build matrix (see report/bench_stream.sh):
 *   gcc -O3 -march=native -fopenmp -o stream512 stream512.c
 *   gcc -O3 -march=native -mprefer-vector-width=512 -fopenmp -o stream512w ...
 *   gcc -O3 -march=znver4 ...            (if the compiler supports it)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <omp.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static void *alloc_huge(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 4096, bytes) != 0) return NULL;
    if (getenv("NO_HUGEPAGE") == NULL)
        madvise(p, bytes, MADV_HUGEPAGE);
    return p;
}

int main(int argc, char **argv) {
    size_t N = (argc > 1) ? strtoull(argv[1], 0, 10) : 200000000;
    int nt = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    double *a = alloc_huge(N * 8), *b = alloc_huge(N * 8), *c = alloc_huge(N * 8);
    if (!a || !b || !c) { fprintf(stderr, "alloc fail\n"); return 1; }

#pragma omp parallel for simd num_threads(nt) schedule(static)
    for (size_t i = 0; i < N; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }

    double q = 3.0, best_c = 0, best_s = 0, best_a = 0;
    for (int r = 0; r < 5; r++) {
        double t0 = now();
#pragma omp parallel for simd num_threads(nt) schedule(static)
        for (size_t i = 0; i < N; i++) c[i] = a[i];
        double gb = N * 8.0 / (now() - t0) / 1e9;
        if (gb > best_c) best_c = gb;

        t0 = now();
#pragma omp parallel for simd num_threads(nt) schedule(static)
        for (size_t i = 0; i < N; i++) c[i] = q * a[i];
        gb = N * 8.0 * 2 / (now() - t0) / 1e9;
        if (gb > best_s) best_s = gb;

        t0 = now();
#pragma omp parallel for simd num_threads(nt) schedule(static)
        for (size_t i = 0; i < N; i++) a[i] = b[i] + q * c[i];
        gb = N * 8.0 * 3 / (now() - t0) / 1e9;
        if (gb > best_a) best_a = gb;
    }
    printf("threads=%3d N=%zu hugepage=%s  Copy %6.1f  Scale %6.1f  Triad %6.1f GB/s\n",
           nt, N, getenv("NO_HUGEPAGE") ? "no" : "yes", best_c, best_s, best_a);
    return 0;
}

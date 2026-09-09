#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/mman.h>
#include <omp.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(int argc,char**argv){
  size_t N=(argc>1)?strtoull(argv[1],0,10):400000000;
  int nt=(argc>2)?atoi(argv[2]):192;
  double*p=NULL;
  if(posix_memalign((void**)&p,2*1024*1024,N*8))return 1;
  madvise(p,N*8,MADV_HUGEPAGE);
  #pragma omp parallel for simd num_threads(nt) schedule(static)
  for(size_t i=0;i<N;i++) p[i]=1.0;
  double best=0;
  for(int r=0;r<5;r++){
    double s=0,t0=now();
    #pragma omp parallel for simd num_threads(nt) schedule(static) reduction(+:s)
    for(size_t i=0;i<N;i++) s+=p[i];
    double g=N*8.0/(now()-t0)/1e9; if(g>best)best=g;
    if(s<0)printf(" ");
  }
  printf("single-stream read: threads=%d  %.1f GB/s\n",nt,best);
  return 0;
}

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(int argc,char**argv){
  size_t N = (argc>1)? strtoull(argv[1],0,10) : 200000000; /* doubles */
  int nt = (argc>2)? atoi(argv[2]) : omp_get_max_threads();
  double *a,*b,*c;
  a=(double*)malloc(N*8); b=(double*)malloc(N*8); c=(double*)malloc(N*8);
  if(!a||!b||!c){fprintf(stderr,"alloc fail\n");return 1;}
  #pragma omp parallel for num_threads(nt) schedule(static)
  for(size_t i=0;i<N;i++){a[i]=1.0;b[i]=2.0;c[i]=0.0;}
  double q=3.0, t0,t1; double best_c=0,best_s=0,best_a=0;
  for(int r=0;r<5;r++){
    t0=now();
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(size_t i=0;i<N;i++) c[i]=a[i];
    t1=now(); double gb=N*8.0/(t1-t0)/1e9; if(gb>best_c)best_c=gb;
    t0=now();
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(size_t i=0;i<N;i++) c[i]=q*a[i];
    t1=now(); gb=N*8.0*2/(t1-t0)/1e9; if(gb>best_s)best_s=gb;
    t0=now();
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(size_t i=0;i<N;i++) a[i]=b[i]+q*c[i];
    t1=now(); gb=N*8.0*3/(t1-t0)/1e9; if(gb>best_a)best_a=gb;
  }
  printf("threads=%d N=%zu  Copy %.1f GB/s  Scale %.1f GB/s  Triad %.1f GB/s\n",nt,N,best_c,best_s,best_a);
  return 0;
}

// nvcc -std=c++17 -O3 -arch=sm_120 -I<cutlass>/include -lcublas -lcuda
// Select ET_CUTLASS_M={16,32}, N={64,128}, K={32,64} at compile time.
#include "expert-cutlass-grouped.cuh"
#define cublasGemmGroupedBatchedEx et_cutlass_grouped
#include "expert-grouped-partials.cu"

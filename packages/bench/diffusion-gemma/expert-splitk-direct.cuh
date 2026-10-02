#ifndef ET_EXPERT_ROW_TILE
#define ET_EXPERT_ROW_TILE 32
#endif
// Exact BF16 split-K partial rounding, matching the expert cuBLAS schedule.
#include <cuda_bf16.h>
#include <mma.h>
struct EtExpertSplitKDescriptor {
    unsigned long long x, weight, out, partial_offset;
    unsigned int rows, splits, slice_k, compute_prefix, reduce_prefix, reserved[3];
};
static_assert(sizeof(EtExpertSplitKDescriptor) == 64, "descriptor ABI");
struct EtExpertSplitKUpload { EtExpertSplitKDescriptor descriptors[32]; };
extern "C" __global__ void et_expert_splitk_upload(EtExpertSplitKDescriptor* descriptors, unsigned int first, unsigned int count, EtExpertSplitKUpload batch) {
    unsigned int i = threadIdx.x;
    if (i < count) descriptors[first + i] = batch.descriptors[i];
}
__device__ __forceinline__ unsigned int et_expert_splitk_find(const EtExpertSplitKDescriptor* d, unsigned int count, unsigned int block, bool reduce) {
    unsigned int low = 0, high = count;
    while (low + 1 < high) {
        unsigned int mid = (low + high) >> 1;
        unsigned int prefix = reduce ? d[mid].reduce_prefix : d[mid].compute_prefix;
        if (prefix <= block) low = mid; else high = mid;
    }
    return low;
}
extern "C" __global__ void et_expert_splitk_compute(const EtExpertSplitKDescriptor* descriptors, unsigned int count, __nv_bfloat16* partials, unsigned int columns, unsigned int inner) {
 using namespace nvcuda;
 const auto d=descriptors[et_expert_splitk_find(descriptors,count,blockIdx.x,false)];
 const auto* x=reinterpret_cast<const __nv_bfloat16*>(d.x);
 const auto* w=reinterpret_cast<const __nv_bfloat16*>(d.weight);
 unsigned nt=(columns+31)/32,mt=(d.rows+ET_EXPERT_ROW_TILE-1)/ET_EXPERT_ROW_TILE,block=blockIdx.x-d.compute_prefix;
 unsigned nr=(block%nt)*32,mr=((block/nt)%mt)*ET_EXPERT_ROW_TILE,split=block/(nt*mt);
 unsigned begin=split*d.slice_k,end=min(inner,begin+d.slice_k),lane=threadIdx.x,warp=lane/32;
 __shared__ __align__(32) __nv_bfloat16 b[ET_EXPERT_ROW_TILE*16];
 __shared__ __align__(32) float c[ET_EXPERT_ROW_TILE*32];
 wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> af;
 wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> bf;
 wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.0f);
 for(unsigned k=begin;k<end;k+=16){
  if(mr+ET_EXPERT_ROW_TILE>d.rows){
   for(unsigned i=lane;i<ET_EXPERT_ROW_TILE*16;i+=blockDim.x){unsigned m=mr+i/16;b[i]=m<d.rows?x[m*inner+k+i%16]:__float2bfloat16(0.0f);}
   __syncthreads();
  }
  wmma::load_matrix_sync(af,w+(nr+(warp%2)*16)*inner+k,inner);
  if(mr+ET_EXPERT_ROW_TILE<=d.rows)wmma::load_matrix_sync(bf,x+(mr+(warp/2)*16)*inner+k,inner);
  else wmma::load_matrix_sync(bf,b+(warp/2)*16*16,16);
  wmma::mma_sync(acc,af,bf,acc);
  if(mr+ET_EXPERT_ROW_TILE>d.rows)__syncthreads();
 }
 wmma::store_matrix_sync(c+(warp%2)*16+(warp/2)*16*32,acc,32,wmma::mem_col_major);__syncthreads();
 for(unsigned i=lane;i<ET_EXPERT_ROW_TILE*32;i+=blockDim.x){unsigned n=nr+i%32,m=mr+i/32;if(n<columns&&m<d.rows)partials[d.partial_offset+(static_cast<unsigned long long>(split)*d.rows+m)*columns+n]=__float2bfloat16_rn(c[i]);}
}
extern "C" __global__ void et_expert_splitk_reduce(const EtExpertSplitKDescriptor* descriptors, unsigned int count, const __nv_bfloat16* partials, unsigned int columns) {
    const auto d = descriptors[et_expert_splitk_find(descriptors, count, blockIdx.x, true)];
    unsigned int i = (blockIdx.x - d.reduce_prefix) * blockDim.x + threadIdx.x, elements = d.rows * columns;
    if (i >= elements) return;
    float value = 0.0f;
    for (unsigned int split = 0; split < d.splits; ++split) value += __bfloat162float(partials[d.partial_offset + static_cast<unsigned long long>(split) * elements + i]);
    reinterpret_cast<__nv_bfloat16*>(d.out)[i] = __float2bfloat16_rn(value);
}

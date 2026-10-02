#include <cuda_bf16.h>
struct Descriptor{unsigned long long x,weight,out,partial_offset;unsigned rows,splits,slice_k,compute_prefix,reduce_prefix,reserved[3];};
struct Shape{int m,n,k;};
struct M1Upload{unsigned count;unsigned indices[128];};
static_assert(sizeof(Descriptor)==64&&sizeof(Shape)==12&&sizeof(M1Upload)==516,"metadata ABI");
__device__ void geometry59(unsigned rows,unsigned inner,unsigned&splits,unsigned&slice){splits=1;slice=inner;if(inner==704){if(rows>=257&&rows<=416){splits=2;slice=384;}return;}if(rows<=16||(rows>=65&&rows<=128)||(rows>=257&&rows<=448))return;if(rows<=30){splits=8;slice=384;}else if(rows<=32||(rows>=43&&rows<=57)||(rows>=193&&rows<=256)||rows>=449){splits=4;slice=704;}else if(rows<=64){splits=2;slice=1408;}else{splits=5;slice=576;}}
// CudaKernelArgs ABI: input0 control[130], input1 gathered BF16 X,
// input2 BF16 weights, input3 row_map, input4 inverse; output BF16 projection.
// scratch0 descriptors[128], scratch1 shapes[128], scratch2 M1meta[129],
// scratch3 shared native status. integers0 columns,1 inner,2 total,3 inverse-map enabled.
// Launch exactly one CTA of 128 threads. Host eligibility proves total <= 2048,
// each expert <= 256, supported paired geometry, and all allocation bounds.
extern "C" __global__ void et_expert_device_metadata59(CudaKernelArgs a) {
    const unsigned e = threadIdx.x;
    const int* control = (const int*)a.inputs[0];
    Descriptor* descriptors = (Descriptor*)a.scratch[0];
    Shape* shapes = (Shape*)a.scratch[1];
    unsigned* meta = (unsigned*)a.scratch[2];
    const unsigned columns = (unsigned)a.integers[0];
    const unsigned inner = (unsigned)a.integers[1];
    const unsigned total = (unsigned)a.integers[2];
    const __nv_bfloat16* x = (const __nv_bfloat16*)a.inputs[1];
    const __nv_bfloat16* w = (const __nv_bfloat16*)a.inputs[2];
    __nv_bfloat16* out = (__nv_bfloat16*)a.output;
    __shared__ unsigned invalid;
    if (e == 0) {
        invalid = (a.scratch[3] && *((const et_u64*)a.scratch[3]) != 0);
        meta[0] = 0;
    }
    meta[1 + e] = 0;
    __syncthreads();
    const int begin = control[e + 1], end = control[e + 2];
    if (control[0] != 0 || control[1] != 0 || control[129] != int(total)
        || (total != 512 && total != 2048) || begin < 0 || end < begin || end > int(total)
        || end - begin > int(total / 8)
        || !((columns == 1408 && inner == 2816) || (columns == 2816 && inner == 704)))
        atomicOr(&invalid, 1);
    __syncthreads();
    Descriptor d{};
    Shape shape{0, int(columns), int(inner)};
    if (!invalid) {
        unsigned rows = end - begin;
        d.x = (unsigned long long)(x + size_t(begin) * inner);
        d.weight = (unsigned long long)(w + size_t(e) * columns * inner);
        d.out = (unsigned long long)(out + size_t(begin) * columns);
        d.rows = rows >= 2 ? rows : 0;
        geometry59(rows, inner, d.splits, d.slice_k);
        shape.m = d.rows;
        unsigned prefix = 0;
        for (unsigned i = 0; i < e; ++i) prefix += control[i + 2] - control[i + 1] == 1;
        if (rows == 1) meta[1 + prefix] = e;
        if (e == 127) meta[0] = prefix + (rows == 1);
    } else if (e == 0) {
        et_error(a, 1);
    }
    descriptors[e] = d;
    shapes[e] = shape;
}

// Run on primary immediately after metadata, before guarded row-map kernels.
// Metadata has already zeroed every descriptor and M1 count on error. Bounded
// identities make subsequent gather/retained-finalizer reads safe; zeroing the
// private projection prevents use of poison even when subsequent math executes.
extern "C" __global__ void et_expert_device_sanitize59(CudaKernelArgs a) {
    if (!a.scratch[3] || *((const et_u64*)a.scratch[3]) == 0) return;
    const et_u64 total = a.integers[2];
    const et_u64 columns = a.integers[0];
    const et_u64 stride = (et_u64)gridDim.x * blockDim.x;
    for (et_u64 row = et_thread(); row < total; row += stride) {
        ((unsigned*)a.inputs[3])[row] = (unsigned)row;
        if (a.integers[3]) ((unsigned*)a.inputs[4])[row] = (unsigned)row;
    }
    for (et_u64 i = et_thread(); i < total * columns; i += stride)
        ((__nv_bfloat16*)a.output)[i] = __float2bfloat16_rn(0.0f);
}
// Literal54 first-M1 arithmetic, only dispatch changes to direct indexed persistent jobs.
extern "C" __global__ void et_expert_device_first59(const Descriptor*descriptors,const unsigned*meta){
 __shared__ float partial[64];
 for(unsigned job=blockIdx.x;job<meta[0]*352;job+=gridDim.x){
  const auto d=descriptors[meta[1+job/352]];const auto*x=(const __nv_bfloat16*)d.x;const auto*w=(const __nv_bfloat16*)d.weight;
  unsigned lane=threadIdx.x,col=(job%352)*4+threadIdx.y;float value=0.0f;
  for(unsigned i=0;i<176;++i){unsigned k=lane+i*16;value=fmaf(__bfloat162float(x[k]),__bfloat162float(w[col*2816+k]),value);}
  unsigned index=threadIdx.y*16+lane;partial[index]=value;__syncthreads();
  for(unsigned stride=8;stride;stride>>=1){if(lane<stride)partial[index]+=partial[index+stride];__syncthreads();}
  if(lane==0)((__nv_bfloat16*)d.out)[col]=__float2bfloat16_rn(partial[index]);
  // Every thread must finish consuming the previous shared tile before next job.
  __syncthreads();
 }
}
// Literal production second-M1 arithmetic, same22 products/lane and32-lane tree.
extern "C" __global__ void et_expert_device_second59(const Descriptor*descriptors,const unsigned*meta){
 __shared__ float partial[128];
 for(unsigned job=blockIdx.x;job<meta[0]*704;job+=gridDim.x){
  const auto d=descriptors[meta[1+job/704]];const auto*x=(const __nv_bfloat16*)d.x;const auto*w=(const __nv_bfloat16*)d.weight;
  unsigned lane=threadIdx.x,col=(job%704)*4+threadIdx.y;float value=0.0f;
  for(unsigned j=0;j<22;++j){unsigned k=lane*22+j;value=fmaf(__bfloat162float(x[k]),__bfloat162float(w[col*704+k]),value);}
  unsigned index=threadIdx.y*32+lane;partial[index]=value;__syncthreads();
  for(unsigned stride=16;stride;stride>>=1){if(lane<stride)partial[index]+=partial[index+stride];__syncthreads();}
  if(lane==0)((__nv_bfloat16*)d.out)[col]=__float2bfloat16_rn(partial[index]);
  __syncthreads();
 }
}

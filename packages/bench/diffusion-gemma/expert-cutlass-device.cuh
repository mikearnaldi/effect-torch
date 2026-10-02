// Standalone device-metadata exact expert execution. No runtime integration.
#include "expert-cutlass-grouped.cuh"
#include <cuda_bf16.h>
#if defined(ET_DEVICE_LOSSLESS) && ET_DEVICE_LOSSLESS
#include "expert-bf16-lossless.cuh"
#endif

__device__ inline void device_split_geometry(int rows,int inner,int& splits,int& slice) {
  splits=1;slice=inner;
  if(inner==704){if(rows>=257&&rows<=416){splits=2;slice=384;}return;}
  if(rows<=16||(rows>=65&&rows<=128)||(rows>=257&&rows<=448))return;
  if(rows<=30){splits=8;slice=384;}
  else if(rows<=32||(rows>=43&&rows<=57)||(rows>=193&&rows<=256)||rows>=449){splits=4;slice=704;}
  else if(rows<=64){splits=2;slice=1408;}
  else {splits=5;slice=576;}
}

struct DeviceExpertMetadata {
  cutlass::gemm::GemmCoord* shapes;
  EtBf16 **a,**b,**d;
  int64_t *lda,*ldb,*ldd;
  int *m1_indices,*m1_count,*offsets,*problem_count;
  EtGroupedKernel::Params* params;
};

__global__ void build_device_experts(const int* counts, int experts,int columns,int inner,
    EtBf16* x,EtBf16* w,EtBf16* out,EtBf16* partial,DeviceExpertMetadata metadata,bool compact,bool merge_slices=false) {
  int expert=threadIdx.x;if(expert>=experts)return;
  int m1_prefix=0;for(int i=0;i<expert;++i)m1_prefix+=counts[i]==1;
  if(counts[expert]==1)metadata.m1_indices[m1_prefix]=expert;
  if(expert==experts-1)*metadata.m1_count=m1_prefix+(counts[expert]==1);
  int rows=counts[expert],offset=0;for(int i=0;i<expert;++i)offset+=counts[i];
  metadata.offsets[expert]=offset;
  int splits,slice;device_split_geometry(rows,inner,splits,slice);
  if(merge_slices){splits=1;slice=inner;}
  int problem_prefix=0;
  if(compact)for(int i=0;i<expert;++i){int prior_splits,prior_slice;device_split_geometry(counts[i],inner,prior_splits,prior_slice);if(counts[i]>1)problem_prefix+=merge_slices?1:prior_splits;}
  if(expert==experts-1){
    int total=compact?problem_prefix+(rows>1?splits:0):experts*8;
    *metadata.problem_count=total;
    if(metadata.params)metadata.params->problem_visitor.problem_count=total;
  }
  for(int s=0;s<8;++s){
    bool active=rows>1&&s<splits;
    if(compact&&!active)continue;
    int slot=compact?problem_prefix+s:expert*8+s;
    metadata.shapes[slot]=cutlass::gemm::GemmCoord(active?rows:0,columns,active?min(slice,inner-s*slice):inner);
    metadata.a[slot]=x+size_t(offset)*inner+(active?s*slice:0);
    metadata.b[slot]=w+size_t(expert)*columns*inner+(active?s*slice:0);
    metadata.d[slot]=splits==1?out+size_t(offset)*columns:
      partial+size_t(offset)*columns*8+size_t(s)*rows*columns;
    metadata.lda[slot]=inner;metadata.ldb[slot]=inner;metadata.ldd[slot]=columns;
  }
}

// Device-only scheduling accepts a compact count loaded at launch time. Keep
// CUTLASS's original MMA/epilogue operator and shared layout unchanged.
__global__ void run_compact_experts(EtGroupedKernel::Params params,const int* problem_count){
  extern __shared__ int storage[];
  __shared__ int count;
  if(threadIdx.x==0)count=*problem_count;
  __syncthreads();
  params.problem_visitor.problem_count=count;
  EtGroupedKernel op;
  op(params,*reinterpret_cast<EtGroupedKernel::SharedStorage*>(storage));
}

// Alternative control: immutable pointer/stride Params live in invocation GPU
// storage; metadata only updates their count before this launch. Avoid creating
// a mutable thread-local Params object when passing the device count to CUTLASS.
__global__ void run_compact_device_params(const EtGroupedKernel::Params* params){
  extern __shared__ int storage[];
  EtGroupedKernel op;
  op(*params,*reinterpret_cast<EtGroupedKernel::SharedStorage*>(storage));
}

__global__ void reduce_device_experts(const int* counts,const int* offsets,int columns,int inner,
    const __nv_bfloat16* partial,__nv_bfloat16* out) {
  int expert=blockIdx.x,rows=counts[expert],splits,slice;
  device_split_geometry(rows,inner,splits,slice);
  if(rows<=1||splits==1)return;
  int offset=offsets[expert];
  size_t elements=size_t(rows)*columns;
  for(size_t index=blockIdx.y*blockDim.x+threadIdx.x;index<elements;index+=gridDim.y*blockDim.x){
    float value=0;
    for(int s=0;s<splits;++s)value+=__bfloat162float(partial[size_t(offset)*columns*8+size_t(s)*elements+index]);
    out[size_t(offset)*columns+index]=__float2bfloat16_rn(value);
  }
}

__global__ void gemv_device_experts(const int* counts,const int* offsets,int columns,int inner,
    const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* out,
    const int* m1_indices,const int* m1_count) {
  __shared__ float partial[128];
  for(unsigned job=blockIdx.x;job<unsigned(*m1_count)*(columns/4);job+=gridDim.x){
  int expert=m1_indices[job/(columns/4)];
  int offset=offsets[expert];
  unsigned lane=threadIdx.x,lanes=blockDim.x,col=(job%(columns/4))*4+threadIdx.y;
  unsigned grain=inner==2816?1:22;
  float value=0;
  for(unsigned base=lane*grain;base<unsigned(inner);base+=lanes*grain)
    for(unsigned j=0;j<grain&&base+j<unsigned(inner);++j)
      value=fmaf(__bfloat162float(x[size_t(offset)*inner+base+j]),
        __bfloat162float(w[(size_t(expert)*columns+col)*inner+base+j]),value);
  unsigned i=threadIdx.y*lanes+lane;
  partial[i]=value;__syncthreads();
  for(unsigned stride=lanes/2;stride;stride>>=1){if(lane<stride)partial[i]+=partial[i+stride];__syncthreads();}
  if(lane==0&&col<unsigned(columns))out[size_t(offset)*columns+col]=__float2bfloat16_rn(partial[i]);
  __syncthreads();
  }
}

// Second projection only: coalesce global reads without changing cuBLAS's
// contiguous 22-product lane sequence or the ordered 32-lane reduction.
__global__ void gemv_device_experts_shared(const int* offsets,int columns,
    const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* out,
    const int* m1_indices,const int* m1_count) {
  __shared__ __align__(16) __nv_bfloat16 shared_w[4*704],shared_x[704];
  __shared__ float partial[128];
  unsigned lane=threadIdx.x,thread=threadIdx.y*32+lane;
  for(unsigned job=blockIdx.x;job<unsigned(*m1_count)*(columns/4);job+=gridDim.x){
    int expert=m1_indices[job/(columns/4)],offset=offsets[expert];
    unsigned column_base=(job%(columns/4))*4;
    for(unsigned i=thread;i<4*704;i+=128)
      shared_w[i]=w[(size_t(expert)*columns+column_base)*704+i];
    for(unsigned i=thread;i<704;i+=128)shared_x[i]=x[size_t(offset)*704+i];
    __syncthreads();
    float value=0;
    for(unsigned j=0;j<22;++j){unsigned k=lane*22+j;
      value=fmaf(__bfloat162float(shared_x[k]),__bfloat162float(shared_w[threadIdx.y*704+k]),value);}
    partial[thread]=value;__syncthreads();
    for(unsigned stride=16;stride;stride>>=1){if(lane<stride)partial[thread]+=partial[thread+stride];__syncthreads();}
    if(lane==0)out[size_t(offset)*columns+column_base+threadIdx.y]=__float2bfloat16_rn(partial[thread]);
    __syncthreads();
  }
}

// Retain each split in registers and reuse the pipelined CUTLASS mainloop.
// BF16 rounding and ordered F32 summation match the materialized split path.
__global__ void run_merged_device_experts(const EtGroupedKernel::Params* params) {
#if defined(ET_DEVICE_LOSSLESS) && ET_DEVICE_LOSSLESS
  using Mma=typename EtLosslessMma<EtGroupedKernel::Mma>::Type;
#else
  using Mma=EtGroupedKernel::Mma;
#endif
  using Epilogue=EtGroupedKernel::Epilogue;
  using Visitor=EtGroupedKernel::ProblemVisitor;
  extern __shared__ int storage[];
  auto& shared=*reinterpret_cast<EtGroupedKernel::SharedStorage*>(storage);
  Visitor visitor(params->problem_visitor,shared.problem_visitor,blockIdx.x);
  int thread=threadIdx.x,warp=cutlass::canonical_warp_idx_sync(),lane=thread%32;
  while(visitor.next_tile()){
    auto shape=visitor.problem_size();int problem=visitor.problem_index();
    auto grid=visitor.grid_shape(shape);int tile=int(visitor.threadblock_idx());
    cutlass::MatrixCoord origin((tile/grid.n())*Mma::Shape::kM,(tile%grid.n())*Mma::Shape::kN);
    int splits,slice;device_split_geometry(shape.m(),shape.k(),splits,slice);
#if defined(ET_DEVICE_LOSSLESS) && ET_DEVICE_LOSSLESS
    auto lossless_context=et_lossless_context(params->ptr_B[problem]);
#endif
    typename Mma::FragmentC total;total.clear();
    for(int split=0;split<splits;++split){
      int begin=split*slice,length=min(slice,shape.k()-begin);
      typename Mma::IteratorA a(typename Mma::IteratorA::Layout(params->lda[problem]),
        params->ptr_A[problem]+begin,{shape.m(),length},thread,{origin.row(),0});
      typename Mma::IteratorB b(typename Mma::IteratorB::Layout(params->ldb[problem]),
        params->ptr_B[problem]+begin,{length,shape.n()},thread,{0,origin.column()});
      typename Mma::FragmentC partial;partial.clear();
      __syncthreads();
#if defined(ET_DEVICE_LOSSLESS) && ET_DEVICE_LOSSLESS
      Mma mma(shared.kernel.main_loop,thread,warp,lane,lossless_context);
#else
      Mma mma(shared.kernel.main_loop,thread,warp,lane);
#endif
      mma((length+Mma::Shape::kK-1)/Mma::Shape::kK,partial,a,b,partial);
      CUTLASS_PRAGMA_UNROLL
      for(int i=0;i<Mma::FragmentC::kElements;++i){
        if(splits==1)total[i]=partial[i];
        else total[i]=__fadd_rn(total[i],__bfloat162float(__float2bfloat16_rn(partial[i])));
      }
    }
    __syncthreads();
    typename Epilogue::OutputTileIterator::Params out_params(
      typename Epilogue::OutputTileIterator::Layout(params->ldd[problem]));
    typename Epilogue::OutputTileIterator output(out_params,params->ptr_D[problem],shape.mn(),thread,origin);
    Epilogue epilogue(shared.kernel.epilogue,thread,warp,lane);
    typename EtGroupedKernel::EpilogueOutputOp op(params->output_op);
    epilogue(op,output,total,output);
    __syncthreads();
    visitor.advance(gridDim.x);
  }
}

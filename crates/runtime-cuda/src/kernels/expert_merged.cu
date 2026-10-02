// Experimental ahead-of-time CUTLASS module. Built only by the explicit helper;
// the normal runtime has no CUTLASS or nvcc build dependency.
#include <cuda_bf16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/epilogue/thread/linear_combination.h>

using Element=cutlass::bfloat16_t;
using OutputOp=cutlass::epilogue::thread::LinearCombination<Element,8,float,float,
  cutlass::epilogue::thread::ScaleType::Nothing>;
using Kernel=typename cutlass::gemm::kernel::DefaultGemmGrouped<
  Element,cutlass::layout::RowMajor,cutlass::ComplexTransform::kNone,8,
  Element,cutlass::layout::ColumnMajor,cutlass::ComplexTransform::kNone,8,
  Element,cutlass::layout::RowMajor,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<32,64,64>,cutlass::gemm::GemmShape<16,32,64>,
  cutlass::gemm::GemmShape<16,8,16>,OutputOp,
  cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,3,
  cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,cutlass::arch::OpMultiplyAdd>::GemmKernel;
static_assert(Kernel::Mma::Shape::kK==Kernel::Mma::Operator::Shape::kK,"sequential K16 accumulation");
static_assert(Kernel::kThreadCount==128,"Rust launch ABI");
static_assert(sizeof(Kernel::SharedStorage)==36880,"Rust shared storage ABI");
static_assert(sizeof(cutlass::gemm::GemmCoord)==12,"Rust shape bank ABI");

// Matches ExpertSplitKDescriptor. All storage belongs to the invocation.
struct Descriptor {
  unsigned long long x,weight,out,partial_offset;
  unsigned rows,splits,slice_k,compute_prefix,reduce_prefix,reserved[3];
};
struct Upload {Descriptor descriptors[32];};
static_assert(sizeof(Descriptor)==64&&sizeof(Upload)==2048,"Rust descriptor ABI");
extern "C" __global__ void et_expert_merged_upload_v1(Descriptor* descriptors,
    cutlass::gemm::GemmCoord* shapes,unsigned first,unsigned count,unsigned columns,
    unsigned inner,Upload upload){
  unsigned lane=threadIdx.x;
  if(lane<count){descriptors[first+lane]=upload.descriptors[lane];
    shapes[first+lane]=cutlass::gemm::GemmCoord(upload.descriptors[lane].rows,columns,inner);}
}
// CUDA 12.1+ supports up to 32764 bytes of parameters on SM70+. Reading the
// immutable bank directly avoids an 8192-byte per-thread local parameter copy.
struct WideUpload {Descriptor descriptors[128];};
static_assert(sizeof(WideUpload)==8192,"Rust wide descriptor ABI");
extern "C" __global__ void et_expert_merged_upload_v2(Descriptor* descriptors,
    cutlass::gemm::GemmCoord* shapes,unsigned first,unsigned count,unsigned columns,
    unsigned inner,const __grid_constant__ WideUpload upload){
  unsigned lane=threadIdx.x;
  if(lane<count){descriptors[first+lane]=upload.descriptors[lane];
    shapes[first+lane]=cutlass::gemm::GemmCoord(upload.descriptors[lane].rows,columns,inner);}
}
extern "C" __global__ void et_expert_merged_compute_v1(const Descriptor* descriptors,
    cutlass::gemm::GemmCoord* shapes,unsigned count,unsigned columns,unsigned inner){
  using Mma=Kernel::Mma;using Epilogue=Kernel::Epilogue;using Visitor=Kernel::ProblemVisitor;
  extern __shared__ int storage[];
  auto& shared=*reinterpret_cast<Kernel::SharedStorage*>(storage);
  typename Visitor::Params visitor_params(shapes,count,nullptr,0);
  Visitor visitor(visitor_params,shared.problem_visitor,blockIdx.x);
  int thread=threadIdx.x,warp=cutlass::canonical_warp_idx_sync(),lane=thread%32;
  while(visitor.next_tile()){
    auto shape=visitor.problem_size();auto descriptor=descriptors[visitor.problem_index()];
    auto grid=visitor.grid_shape(shape);int tile=int(visitor.threadblock_idx());
    cutlass::MatrixCoord origin((tile/grid.n())*32,(tile%grid.n())*64);
    typename Mma::FragmentC total;total.clear();
    for(unsigned split=0;split<descriptor.splits;++split){
      unsigned begin=split*descriptor.slice_k,length=min(descriptor.slice_k,inner-begin);
      typename Mma::IteratorA a(typename Mma::IteratorA::Layout(inner),
        reinterpret_cast<Element*>(descriptor.x)+begin,{shape.m(),int(length)},thread,{origin.row(),0});
      typename Mma::IteratorB b(typename Mma::IteratorB::Layout(inner),
        reinterpret_cast<Element*>(descriptor.weight)+begin,{int(length),shape.n()},thread,{0,origin.column()});
      typename Mma::FragmentC partial;partial.clear();
      __syncthreads();
      Mma mma(shared.kernel.main_loop,thread,warp,lane);
      mma((length+63)/64,partial,a,b,partial);
      CUTLASS_PRAGMA_UNROLL
      for(int i=0;i<Mma::FragmentC::kElements;++i)
        total[i]=descriptor.splits==1?partial[i]:
          __fadd_rn(total[i],__bfloat162float(__float2bfloat16_rn(partial[i])));
    }
    __syncthreads();
    typename Epilogue::OutputTileIterator::Params output_params{
      typename Epilogue::OutputTileIterator::Layout(columns)};
    typename Epilogue::OutputTileIterator output(output_params,reinterpret_cast<Element*>(descriptor.out),shape.mn(),thread,origin);
    Epilogue epilogue(shared.kernel.epilogue,thread,warp,lane);
    OutputOp op(typename OutputOp::Params(1,0));
    epilogue(op,output,total,output);
    __syncthreads();visitor.advance(gridDim.x);
  }
}

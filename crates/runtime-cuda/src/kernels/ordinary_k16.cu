// Experimental fixed-shape artifact; normal builds have no CUTLASS dependency.
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
using Element=cutlass::bfloat16_t;
using OutputOp=cutlass::epilogue::thread::LinearCombination<Element,8,float,float,
  cutlass::epilogue::thread::ScaleType::Nothing>;
using DeviceGemm=cutlass::gemm::device::Gemm<Element,cutlass::layout::RowMajor,
  Element,cutlass::layout::ColumnMajor,Element,cutlass::layout::RowMajor,float,
  cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<64,64,64>,cutlass::gemm::GemmShape<32,32,64>,
  cutlass::gemm::GemmShape<16,8,16>,OutputOp,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,3,8,8,false,
  cutlass::arch::OpMultiplyAdd>;
using Kernel=DeviceGemm::GemmKernel;
static_assert(Kernel::Mma::Shape::kK==Kernel::Mma::Operator::Shape::kK,"sequential K16 accumulation");
static_assert(Kernel::kThreadCount==128,"Rust thread ABI");
static_assert(sizeof(Kernel::SharedStorage)==49152,"Rust shared storage ABI");
struct Arguments {unsigned long long x,weight,out;unsigned columns,reserved;};
static_assert(sizeof(Arguments)==32,"Rust argument ABI");
extern "C" __global__ void et_ordinary_k16_v1(Arguments args){
  using Mma=Kernel::Mma;using Epilogue=Kernel::Epilogue;
  extern __shared__ int storage[];
  auto& shared=*reinterpret_cast<Kernel::SharedStorage*>(storage);
  int thread=threadIdx.x,warp=cutlass::canonical_warp_idx_sync(),lane=thread%32;
  cutlass::MatrixCoord origin(int(blockIdx.y)*64,int(blockIdx.x)*64);
  typename Mma::IteratorA a(typename Mma::IteratorA::Layout(2816),
    reinterpret_cast<Element*>(args.x),{256,2816},thread,{origin.row(),0});
  typename Mma::IteratorB b(typename Mma::IteratorB::Layout(2816),
    reinterpret_cast<Element*>(args.weight),{2816,int(args.columns)},thread,{0,origin.column()});
  typename Mma::FragmentC accumulator;accumulator.clear();
  Mma mma(shared.main_loop,thread,warp,lane);
  mma(44,accumulator,a,b,accumulator);
  __syncthreads();
  typename Epilogue::OutputTileIterator::Params output_params{
    typename Epilogue::OutputTileIterator::Layout(args.columns)};
  typename Epilogue::OutputTileIterator output(output_params,reinterpret_cast<Element*>(args.out),
    {256,int(args.columns)},thread,origin);
  Epilogue epilogue(shared.epilogue,thread,warp,lane);
  OutputOp op(typename OutputOp::Params(1,0));
  epilogue(op,output,accumulator,output);
}

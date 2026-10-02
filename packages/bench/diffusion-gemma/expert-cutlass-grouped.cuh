// Standalone experiment adapter, not a runtime integration. Problem metadata is
// uploaded once per immutable pointer bank, before any timed repeat/capture.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <map>
#include <memory>
#include <tuple>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifndef ET_CUTLASS_M
#define ET_CUTLASS_M 32
#endif
#ifndef ET_CUTLASS_N
#define ET_CUTLASS_N 64
#endif
#ifndef ET_CUTLASS_K
#define ET_CUTLASS_K 32
#endif
#ifndef ET_CUTLASS_STAGES
#define ET_CUTLASS_STAGES 3
#endif
#ifndef ET_CUTLASS_WARP_M
#define ET_CUTLASS_WARP_M (ET_CUTLASS_M <= 32 ? 16 : 32)
#endif
#ifndef ET_CUTLASS_WARP_N
#define ET_CUTLASS_WARP_N (ET_CUTLASS_N <= 128 ? ET_CUTLASS_N / 2 : 64)
#endif
#ifndef ET_CUTLASS_HOST_SCHEDULE
#define ET_CUTLASS_HOST_SCHEDULE 0
#endif
using EtBf16 = cutlass::bfloat16_t;
using EtEpilogue = cutlass::epilogue::thread::LinearCombination<
    EtBf16, 8, float, float, cutlass::epilogue::thread::ScaleType::Nothing>;
using EtGroupedKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    EtBf16, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
    EtBf16, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,
    EtBf16, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<ET_CUTLASS_M, ET_CUTLASS_N, ET_CUTLASS_K>,
    cutlass::gemm::GemmShape<ET_CUTLASS_WARP_M, ET_CUTLASS_WARP_N, ET_CUTLASS_K>,
    cutlass::gemm::GemmShape<16, 8, 16>, EtEpilogue,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, ET_CUTLASS_STAGES,
    ET_CUTLASS_HOST_SCHEDULE ? cutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute : cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
    cutlass::arch::OpMultiplyAdd>::GemmKernel;
using EtGrouped = cutlass::gemm::device::GemmGrouped<EtGroupedKernel>;
static_assert(EtGroupedKernel::Mma::Shape::kK == EtGroupedKernel::Mma::Operator::Shape::kK,
              "Do not split K between warps: preserve sequential K16 accumulation");
inline void et_cutlass_cuda(cudaError_t status) {
  if (status != cudaSuccess) { std::fprintf(stderr,"CUTLASS metadata: %s\n",cudaGetErrorString(status));std::exit(7); }
}
struct EtCutlassPlan {
  EtGrouped op;
  std::vector<void*> allocations;
  template<class T> T* upload(const std::vector<T>& data) {
    T* p;et_cutlass_cuda(cudaMalloc(&p,data.size()*sizeof(T)));
    allocations.push_back(p);et_cutlass_cuda(cudaMemcpy(p,data.data(),data.size()*sizeof(T),cudaMemcpyHostToDevice));return p;
  }
  ~EtCutlassPlan(){for(void* p:allocations)cudaFree(p);}
};
// Signature mirrors the cuBLAS grouped call in the existing stress harness.
inline cublasStatus_t et_cutlass_grouped(
    cublasHandle_t handle,const cublasOperation_t* ta,const cublasOperation_t* tb,
    const int* m,const int* n,const int* k,const float* alpha,
    const void* const* weights,cudaDataType_t wt,const int* ldw,
    const void* const* inputs,cudaDataType_t xt,const int* ldx,const float* beta,
    void* const* outputs,cudaDataType_t ot,const int* ldo,int groups,
    const int* sizes,cublasComputeType_t compute) {
  using Key=std::tuple<const void*,const void*,const void*,int>;
  static std::map<Key,std::unique_ptr<EtCutlassPlan>> plans;
  Key key{weights,inputs,outputs,groups};auto it=plans.find(key);
  cudaStream_t stream;cublasGetStream(handle,&stream);
  if(it==plans.end()) {
    int count=0;for(int g=0;g<groups;++g){
      if(ta[g]!=CUBLAS_OP_T||tb[g]!=CUBLAS_OP_N||alpha[g]!=1||beta[g]!=0||wt!=CUDA_R_16BF||xt!=CUDA_R_16BF||ot!=CUDA_R_16BF||compute!=CUBLAS_COMPUTE_32F)return CUBLAS_STATUS_NOT_SUPPORTED;
      count+=sizes[g];
    }
    std::vector<EtBf16*> a(count),b(count),d(count);
    et_cutlass_cuda(cudaMemcpy(a.data(),inputs,count*sizeof(void*),cudaMemcpyDeviceToHost));
    et_cutlass_cuda(cudaMemcpy(b.data(),weights,count*sizeof(void*),cudaMemcpyDeviceToHost));
    et_cutlass_cuda(cudaMemcpy(d.data(),outputs,count*sizeof(void*),cudaMemcpyDeviceToHost));
    std::vector<cutlass::gemm::GemmCoord> shapes;
    std::vector<int64_t> lda,ldb,ldd;
    for(int g=0;g<groups;++g)for(int i=0;i<sizes[g];++i){
      shapes.emplace_back(n[g],m[g],k[g]);lda.push_back(ldx[g]);ldb.push_back(ldw[g]);ldd.push_back(ldo[g]);
    }
    auto plan=std::make_unique<EtCutlassPlan>();
    int blocks=EtGrouped::sufficient(shapes.data(),count);
    if(blocks<=0)return CUBLAS_STATUS_NOT_SUPPORTED;
    if(const char* scale=std::getenv("ET_CUTLASS_BLOCK_SCALE")) {
      double factor=std::atof(scale);
      if(!(factor>0.0 && factor<=8.0))return CUBLAS_STATUS_NOT_SUPPORTED;
      blocks=std::max(1,int(blocks*factor));
    }
    auto dd=plan->upload(d);auto strides=plan->upload(ldd);
    typename EtGrouped::Arguments args(plan->upload(shapes),count,blocks,
        typename EtEpilogue::Params(1,0),plan->upload(a),plan->upload(b),dd,dd,
        plan->upload(lda),plan->upload(ldb),strides,strides,shapes.data());
    const size_t schedule_bytes=EtGrouped::get_workspace_size(args);
    void* schedule_workspace=nullptr;
    if(schedule_bytes) {
      et_cutlass_cuda(cudaMalloc(&schedule_workspace,schedule_bytes));
      plan->allocations.push_back(schedule_workspace);
    }
    auto status=plan->op.initialize(args,schedule_workspace,stream);
    if(status!=cutlass::Status::kSuccess){std::fprintf(stderr,"CUTLASS initialize %d\n",int(status));return CUBLAS_STATUS_EXECUTION_FAILED;}
    std::fprintf(stderr,"CUTLASS tile=%dx%dx%d stages=%d sharedBytes=%zu threads=%d problems=%d blocks=%d\n",ET_CUTLASS_M,ET_CUTLASS_N,ET_CUTLASS_K,ET_CUTLASS_STAGES,sizeof(EtGroupedKernel::SharedStorage),EtGroupedKernel::kThreadCount,count,blocks);
    it=plans.emplace(key,std::move(plan)).first;
  }
  auto status=it->second->op.run(stream);
  return status==cutlass::Status::kSuccess?CUBLAS_STATUS_SUCCESS:CUBLAS_STATUS_EXECUTION_FAILED;
}

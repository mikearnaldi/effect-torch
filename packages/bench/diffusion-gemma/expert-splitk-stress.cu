// Dense numerical stress for the exact grouped split-K kernel.
#define main reference_benchmark_main
#include "expert-gemm-batching.cu"
#undef main
#ifdef ET_EXPERT_PIPELINE
#include "expert-splitk-pipeline.cuh"
#elif defined(ET_EXPERT_DIRECT)
#include "expert-splitk-direct.cuh"
#elif defined(ET_EXPERT_ROW_TILE)
#include "expert-splitk-variant.cuh"
#else
#include "../../../crates/runtime-cuda/src/kernels/expert_splitk.cu"
#define ET_EXPERT_ROW_TILE 32
#endif
std::pair<unsigned,unsigned> schedule(int n,int m) {
 if(n==2816) return m>=257&&m<=416?std::pair<unsigned,unsigned>{2,384}:std::pair<unsigned,unsigned>{1,704};
 if(m<=16)return {1,2816}; if(m<=30)return {8,384}; if(m<=32)return {4,704};
 if(m<=42)return {2,1408};if(m<=57)return {4,704};if(m<=64)return {2,1408};
 if(m<=128)return {1,2816};if(m<=192)return {5,576};if(m<=256)return {4,704};
 if(m<=448)return {1,2816};return {4,704};
}
int main(int argc, char** argv) {
 unsigned seed=argc>1?std::strtoul(argv[1],nullptr,10):0;
 bool edge=argc>2&&std::string(argv[2])=="edge";
 bool fixture=argc>2&&!edge;
 cudaStream_t stream;CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 cublasHandle_t handle;BLAS(cublasCreate(&handle));BLAS(cublasSetStream(handle,stream));
 void* workspace;CUDA(cudaMalloc(&workspace,workspace_bytes));BLAS(cublasSetWorkspace(handle,workspace,workspace_bytes));
 EtExpertSplitKDescriptor *dd;CUDA(cudaMalloc(&dd,sizeof(*dd)));
 for(int n:{1408,2816}) {
  int k=n==1408?2816:704;
  auto x=dense(size_t(512)*k,17+seed*113),w=dense(size_t(n)*k,29+seed*127);
  if(edge) {
   const unsigned short patterns[][8]={{0x0001,0x007f,0x0080,0x8001,0x807f,0x8080,0x3f80,0xbf80},{0x3f80,0xbf80,0x3f81,0xbf81,0x3f00,0xbf00,0x3f7f,0xbf7f},{0x3f80,0x3b80,0xbf80,0xbb80,0x3e80,0x3e81,0xbe80,0xbe81}};
   for(size_t i=0;i<x.size();++i)x[i]=patterns[seed%3][i%8];
   for(size_t i=0;i<w.size();++i)w[i]=patterns[seed%3][((i/k)*3+i%k)%8];
  }
  __nv_bfloat16 *dx,*dw,*out,*reference,*partials;
  CUDA(cudaMalloc(&dx,x.size()*2));CUDA(cudaMalloc(&dw,w.size()*2));
  CUDA(cudaMalloc(&out,size_t(512)*n*2));CUDA(cudaMalloc(&reference,size_t(512)*n*2));CUDA(cudaMalloc(&partials,size_t(512)*n*8*2));
  CUDA(cudaMemcpyAsync(dx,x.data(),x.size()*2,cudaMemcpyHostToDevice,stream));CUDA(cudaMemcpyAsync(dw,w.data(),w.size()*2,cudaMemcpyHostToDevice,stream));
  CUDA(cudaStreamSynchronize(stream));
  for(int m=2;m<=512;++m) {
   std::vector<unsigned short> official;
   if(fixture) {
    bool found=false;
    for(const char* family:{"encoder","decoder"}) for(int pair=0;pair<5;++pair) {
     if(argc>3&&std::string(family)!=argv[3])continue;
     int rows=std::string(family)=="encoder"?std::vector<int>{262,6,1,1,4}[pair]:std::vector<int>{149,17,8,3,6}[pair];
     if(rows!=m)continue;
     std::string name=std::string(family)+"-"+std::to_string(pair*2+(n==2816));
     x=load(std::string(argv[2])+"/"+name+".input.bf16",size_t(m)*k);w=load(std::string(argv[2])+"/"+name+".weight.bf16",size_t(n)*k);
     official=load(std::string(argv[2])+"/"+name+".official.bf16",size_t(m)*n);
     found=true;break;
    }
    if(!found)continue;
    CUDA(cudaMemcpyAsync(dx,x.data(),x.size()*2,cudaMemcpyHostToDevice,stream));CUDA(cudaMemcpyAsync(dw,w.data(),w.size()*2,cudaMemcpyHostToDevice,stream));
   }
   auto [splits,width]=schedule(n,m);
   EtExpertSplitKDescriptor d{(unsigned long long)dx,(unsigned long long)dw,(unsigned long long)out,0,(unsigned)m,splits,width,0,0,{0,0,0}};
   CUDA(cudaMemcpyAsync(dd,&d,sizeof(d),cudaMemcpyHostToDevice,stream));
   float alpha=1,beta=0;
   BLAS(cublasGemmStridedBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&alpha,dw,CUDA_R_16BF,k,0,dx,CUDA_R_16BF,k,size_t(m)*k,&beta,reference,CUDA_R_16BF,n,size_t(m)*n,1,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
   et_expert_splitk_compute<<<((n+31)/32)*((m+ET_EXPERT_ROW_TILE-1)/ET_EXPERT_ROW_TILE)*splits,ET_EXPERT_ROW_TILE*4,0,stream>>>(dd,1,partials,n,k);
   et_expert_splitk_reduce<<<(m*n+255)/256,256,0,stream>>>(dd,1,partials,n);
   CUDA(cudaGetLastError());CUDA(cudaStreamSynchronize(stream));
   std::vector<unsigned short> a(size_t(m)*n),b(a.size());
   CUDA(cudaMemcpy(a.data(),out,a.size()*2,cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(b.data(),reference,b.size()*2,cudaMemcpyDeviceToHost));
   size_t exact=0;for(size_t i=0;i<a.size();++i)exact+=a[i]==b[i];
   size_t official_exact=0;for(size_t i=0;i<official.size();++i)official_exact+=a[i]==official[i];
   if(fixture) {
    size_t cublas_official=0;for(size_t i=0;i<official.size();++i)cublas_official+=b[i]==official[i];
    std::fprintf(stderr,"fixture m%d n%d cublasofficial %zu/%zu\n",m,n,cublas_official,official.size());
    unsigned printed=0;for(size_t i=0;i<official.size()&&printed<12;++i)if(a[i]!=official[i]||b[i]!=official[i]) {std::fprintf(stderr,"index%zu custom%04x cublas%04x official%04x\n",i,a[i],b[i],official[i]);++printed;}
   }
   std::printf("{\"m\":%d,\"n\":%d,\"k\":%d,\"exact\":%zu,\"elements\":%zu,\"officialExact\":%zu,\"officialElements\":%zu}\n",m,n,k,exact,a.size(),official_exact,official.size());
  }
  CUDA(cudaFree(dx));CUDA(cudaFree(dw));CUDA(cudaFree(out));CUDA(cudaFree(reference));CUDA(cudaFree(partials));
 }
 CUDA(cudaFree(dd));BLAS(cublasDestroy(handle));CUDA(cudaFree(workspace));CUDA(cudaStreamDestroy(stream));
}

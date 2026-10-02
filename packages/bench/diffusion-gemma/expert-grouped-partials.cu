// Diagnostic: grouped cuBLAS slices followed by the reference BF16 partial reduction.
// nvcc -O3 -arch=sm_120 -lcublas -lcuda expert-grouped-partials.cu -o expert-grouped-partials
#define main batching_diagnostic_main
#include "expert-gemm-batching.cu"
#undef main

__global__ void sum_slices(const __nv_bfloat16* partial, __nv_bfloat16* output,
    size_t count, int slices) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  float value = 0;
  for (int s = 0; s < slices; ++s) value += __bfloat162float(partial[size_t(s) * count + i]);
  output[i] = __float2bfloat16_rn(value);
}

std::pair<int, int> schedule(int rows, int inner) {
  if (inner == 704) return rows >= 257 && rows <= 416 ? std::pair<int,int>{2,384} : std::pair<int,int>{1,704};
  if (rows <= 16 || (rows >= 65 && rows <= 128) || (rows >= 257 && rows <= 448)) return {1,2816};
  if (rows <= 30) return {8,384};
  if (rows <= 32 || (rows >= 43 && rows <= 57) || (rows >= 193 && rows <= 256) || rows >= 449) return {4,704};
  if (rows <= 64) return {2,1408};
  return {5,576};
}

int main(int argc, char** argv) {
  if (argc != 6) { std::fprintf(stderr,"usage: diagnostic rows columns inner seed fixture-prefix-or-dash\n"); return 2; }
  int rows=std::atoi(argv[1]), columns=std::atoi(argv[2]), inner=std::atoi(argv[3]), seed=std::atoi(argv[4]);
  if (rows < 2 || rows > 512) return 2; // M1 must remain ordinary GEMV.
  const int batch=std::getenv("ET_PARTIAL_BATCH")?std::atoi(std::getenv("ET_PARTIAL_BATCH")):16;
  if(batch<1 || batch>128)return 2;
  auto mapped=schedule(rows,inner); int slices=mapped.first,width=mapped.second;
  size_t xc=size_t(rows)*inner,wc=size_t(columns)*inner,oc=size_t(rows)*columns;
  std::string prefix=argv[5];
  auto hx=prefix=="-"?dense(xc,17+seed*113):load(prefix+".input.bf16",xc);
  auto hw=prefix=="-"?dense(wc,29+seed*127):load(prefix+".weight.bf16",wc);
  if(prefix=="-" && std::getenv("ET_PARTIAL_PATTERN")) {
    int pattern=std::atoi(std::getenv("ET_PARTIAL_PATTERN"));
    for(auto* data:{&hx,&hw})for(size_t i=0;i<data->size();++i){unsigned short v=(*data)[i];
      if(pattern==1)v=(v&0x807f)|((100+(i*37+seed*11)%36)<<7);
      if(pattern==2){const unsigned short edges[]={0,0x8000,1,0x8001,0x007f,0x807f,0x0080,0x8080,0x3f80,0xbf80,0x3f81,0xbf81};v=edges[(i*7+seed)%12];}
      (*data)[i]=v;
    }
  }
  std::vector<unsigned short> expected(oc),actual(batch*oc);
  unsigned short *x,*w,*reference,*partial,*out;
  CUDA(cudaMalloc(&x,batch*xc*2));CUDA(cudaMalloc(&w,batch*wc*2));CUDA(cudaMalloc(&reference,oc*2));
  CUDA(cudaMalloc(&partial,batch*slices*oc*2));CUDA(cudaMalloc(&out,batch*oc*2));
  for(int b=0;b<batch;++b){CUDA(cudaMemcpy(x+b*xc,hx.data(),xc*2,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(w+b*wc,hw.data(),wc*2,cudaMemcpyHostToDevice));}
  cublasHandle_t handle;BLAS(cublasCreate(&handle));void* workspace;CUDA(cudaMalloc(&workspace,workspace_bytes));
  BLAS(cublasSetWorkspace(handle,workspace,workspace_bytes));float one=1,zero=0;
  BLAS(cublasGemmStridedBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,columns,rows,inner,&one,w,CUDA_R_16BF,inner,0,x,CUDA_R_16BF,inner,xc,&zero,reference,CUDA_R_16BF,columns,oc,1,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  CUDA(cudaMemcpy(expected.data(),reference,oc*2,cudaMemcpyDeviceToHost));
  if(prefix!="-"){auto official=load(prefix+".official.bf16",oc);if(official!=expected){std::fprintf(stderr,"ordinary fixture mismatch\n");return 3;}}
  std::vector<const void*> xp,wp;std::vector<void*> op;
  std::vector<int> m(slices,columns),n(slices,rows),k(slices),ld(slices,inner),ldout(slices,columns),size(slices,batch);
  std::vector<float> alpha(slices,1),beta(slices,0);
  std::vector<cublasOperation_t> ta(slices,CUBLAS_OP_T),tb(slices,CUBLAS_OP_N);
  for(int s=0;s<slices;++s){k[s]=std::min(width,inner-s*width);for(int b=0;b<batch;++b){xp.push_back(x+b*xc+s*width);wp.push_back(w+b*wc+s*width);op.push_back(partial+(size_t(s)*batch+b)*oc);}}
  int groups=slices;
  if(std::getenv("ET_PARTIAL_SINGLE_GROUPS")){
    std::vector<int> expanded;
    for(int s=0;s<slices;++s)for(int b=0;b<batch;++b)expanded.push_back(k[s]);
    groups=slices*batch;k=expanded;m.assign(groups,columns);n.assign(groups,rows);ld.assign(groups,inner);ldout.assign(groups,columns);size.assign(groups,1);alpha.assign(groups,1);beta.assign(groups,0);ta.assign(groups,CUBLAS_OP_T);tb.assign(groups,CUBLAS_OP_N);
  }
  const void **dx,**dw;void** dout;CUDA(cudaMalloc(&dx,xp.size()*sizeof(void*)));CUDA(cudaMalloc(&dw,wp.size()*sizeof(void*)));CUDA(cudaMalloc(&dout,op.size()*sizeof(void*)));
  CUDA(cudaMemcpy(dx,xp.data(),xp.size()*sizeof(void*),cudaMemcpyHostToDevice));CUDA(cudaMemcpy(dw,wp.data(),wp.size()*sizeof(void*),cudaMemcpyHostToDevice));CUDA(cudaMemcpy(dout,op.data(),op.size()*sizeof(void*),cudaMemcpyHostToDevice));
  auto submit=[&](){BLAS(cublasGemmGroupedBatchedEx(handle,ta.data(),tb.data(),m.data(),n.data(),k.data(),alpha.data(),dw,CUDA_R_16BF,ld.data(),dx,CUDA_R_16BF,ld.data(),beta.data(),dout,CUDA_R_16BF,ldout.data(),groups,size.data(),CUBLAS_COMPUTE_32F));sum_slices<<<(batch*oc+255)/256,256>>>((__nv_bfloat16*)partial,(__nv_bfloat16*)out,batch*oc,slices);CUDA(cudaGetLastError());};
  submit();CUDA(cudaMemcpy(actual.data(),out,batch*oc*2,cudaMemcpyDeviceToHost));size_t exact=0;for(size_t i=0;i<actual.size();++i)exact+=actual[i]==expected[i%oc];
  std::printf("{\"m\":%d,\"n\":%d,\"k\":%d,\"seed\":%d,\"slices\":%d,\"width\":%d,\"batch\":%d,\"exact\":%zu,\"elements\":%zu",rows,columns,inner,seed,slices,width,batch,exact,actual.size());
  if(exact==actual.size() && !std::getenv("ET_PARTIAL_NO_TIMING")){cudaEvent_t a,b;CUDA(cudaEventCreate(&a));CUDA(cudaEventCreate(&b));
    for(int mode=0;mode<2;++mode){std::vector<float> times;for(int i=-3;i<11;++i){CUDA(cudaEventRecord(a));
      if(mode==0)submit();else for(int bank=0;bank<batch;++bank)BLAS(cublasGemmStridedBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,columns,rows,inner,&one,w+bank*wc,CUDA_R_16BF,inner,0,x+bank*xc,CUDA_R_16BF,inner,xc,&zero,out+bank*oc,CUDA_R_16BF,columns,oc,1,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
      CUDA(cudaEventRecord(b));CUDA(cudaEventSynchronize(b));float ms;CUDA(cudaEventElapsedTime(&ms,a,b));if(i>=0)times.push_back(ms);}std::sort(times.begin(),times.end());std::printf(",\"%s\":%.6f",mode==0?"groupedGpuMedianMs":"ordinaryOneStreamGpuMedianMs",times[times.size()/2]);}
  }
  std::printf("}\n");return exact==actual.size()?0:4;
}

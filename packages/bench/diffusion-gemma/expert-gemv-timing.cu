// Separate diagnostic tile variant; never changes the production kernel.
#define main reference_benchmark_main
#include "expert-gemm-batching.cu"
#undef main
#include "expert-gemv-grouped.cuh"
#define ET_EXPERT_ROW_TILE 1
int main() {
 Worker worker[32];cudaStream_t stream;CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 for(auto &w:worker){CUDA(cudaStreamCreateWithFlags(&w.stream,cudaStreamNonBlocking));BLAS(cublasCreate(&w.handle));BLAS(cublasSetStream(w.handle,w.stream));CUDA(cudaMalloc(&w.workspace,workspace_bytes));BLAS(cublasSetWorkspace(w.handle,w.workspace,workspace_bytes));CUDA(cudaEventCreateWithFlags(&w.done,cudaEventDisableTiming));}
 cudaEvent_t start,end;CUDA(cudaEventCreate(&start));CUDA(cudaEventCreate(&end));
 EtExpertSplitKDescriptor *dd;CUDA(cudaMalloc(&dd,32*sizeof(*dd)));
 for(int scenario=0;scenario<2;++scenario){
  int m=1,n=scenario==0?1408:2816,k=scenario==0?2816:704;
  unsigned splits=1,width=k;
  size_t xc=size_t(m)*k,wc=size_t(n)*k,oc=size_t(m)*n;
  __nv_bfloat16 *dx,*dw,*out,*ref,*partials;CUDA(cudaMalloc(&dx,32*xc*2));CUDA(cudaMalloc(&dw,32*wc*2));CUDA(cudaMalloc(&out,32*oc*2));CUDA(cudaMalloc(&ref,32*oc*2));CUDA(cudaMalloc(&partials,32*oc*splits*2));
  std::vector<EtExpertSplitKDescriptor> d;
  unsigned compute=0,reduce=0;
  for(int i=0;i<32;++i){auto x=dense(xc,17+i*113),w=dense(wc,29+i*127);CUDA(cudaMemcpyAsync(dx+i*xc,x.data(),xc*2,cudaMemcpyHostToDevice,stream));CUDA(cudaMemcpyAsync(dw+i*wc,w.data(),wc*2,cudaMemcpyHostToDevice,stream));CUDA(cudaStreamSynchronize(stream));d.push_back({(unsigned long long)(dx+i*xc),(unsigned long long)(dw+i*wc),(unsigned long long)(out+i*oc),i*oc*splits,(unsigned)m,splits,width,compute,reduce,{0,0,0}});compute+=(n+3)/4;reduce+=(oc+255)/256;}
  CUDA(cudaMemcpyAsync(dd,d.data(),32*sizeof(*dd),cudaMemcpyHostToDevice,stream));CUDA(cudaStreamSynchronize(stream));
  for(int mode=0;mode<2;++mode){std::vector<float> samples;
   for(int iteration=-3;iteration<15;++iteration){CUDA(cudaEventRecord(start,stream));
    if(mode==0){float alpha=1,beta=0;for(int i=0;i<32;++i){CUDA(cudaStreamWaitEvent(worker[i].stream,start));BLAS(cublasGemmStridedBatchedEx(worker[i].handle,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&alpha,dw+i*wc,CUDA_R_16BF,k,0,dx+i*xc,CUDA_R_16BF,k,xc,&beta,ref+i*oc,CUDA_R_16BF,n,oc,1,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));CUDA(cudaEventRecord(worker[i].done,worker[i].stream));CUDA(cudaStreamWaitEvent(stream,worker[i].done));}}
    else{et_expert_gemv_compute<<<compute,dim3(n==1408?16:32,4),0,stream>>>(dd,32,n,k);}
    CUDA(cudaEventRecord(end,stream));CUDA(cudaEventSynchronize(end));float ms;CUDA(cudaEventElapsedTime(&ms,start,end));if(iteration>=0)samples.push_back(ms);
   }
   std::sort(samples.begin(),samples.end());std::printf("{\"m\":%d,\"n\":%d,\"k\":%d,\"rowTile\":%d,\"mode\":%d,\"medianMs\":%.6f}\n",m,n,k,ET_EXPERT_ROW_TILE,mode,samples[samples.size()/2]);
  }
  std::vector<unsigned short>a(32*oc),b(a.size());CUDA(cudaMemcpy(a.data(),out,a.size()*2,cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(b.data(),ref,b.size()*2,cudaMemcpyDeviceToHost));size_t exact=0;for(size_t i=0;i<a.size();++i)exact+=a[i]==b[i];std::printf("{\"exact\":%zu,\"elements\":%zu}\n",exact,a.size());
  CUDA(cudaFree(dx));CUDA(cudaFree(dw));CUDA(cudaFree(out));CUDA(cudaFree(ref));CUDA(cudaFree(partials));
 }
 CUDA(cudaFree(dd));for(auto&w:worker){BLAS(cublasDestroy(w.handle));CUDA(cudaFree(w.workspace));CUDA(cudaEventDestroy(w.done));CUDA(cudaStreamDestroy(w.stream));}CUDA(cudaEventDestroy(start));CUDA(cudaEventDestroy(end));CUDA(cudaStreamDestroy(stream));
}

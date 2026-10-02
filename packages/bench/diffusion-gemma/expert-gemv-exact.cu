// Isolated cuBLAS GEMV reduction reconstruction for one routed row.
#define main reference_benchmark_main
#include "expert-gemm-batching.cu"
#undef main
#include "expert-gemv-grouped.cuh"
__global__ void candidate(const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* out,int n,int k,int grain,int order) {
 int lane=threadIdx.x,col=blockIdx.x*4+threadIdx.y,lanes=blockDim.x;
 float value=0;
 if(col<n)for(int base=lane*grain;base<k;base+=lanes*grain)for(int j=0;j<grain&&base+j<k;++j)value=fmaf(__bfloat162float(x[base+j]),__bfloat162float(w[col*k+base+j]),value);
 __shared__ float partial[128];int i=threadIdx.y*lanes+lane;partial[i]=value;__syncthreads();
 if(order==0)for(int stride=lanes/2;stride;stride/=2){if(lane<stride)partial[i]+=partial[i+stride];__syncthreads();}
 else for(int stride=1;stride<lanes;stride*=2){if(lane%(2*stride)==0)partial[i]+=partial[i+stride];__syncthreads();}
 if(lane==0&&col<n)out[col]=__float2bfloat16_rn(partial[i]);
}
int main(int argc,char**argv){
 if(argc<2)return 1;
 bool stress=argc>2; size_t mismatches=0;
 EtExpertSplitKDescriptor* descriptor; CUDA(cudaMalloc(&descriptor,sizeof(*descriptor)));
 cudaStream_t stream;CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 cublasHandle_t h;BLAS(cublasCreate(&h));BLAS(cublasSetStream(h,stream));void* workspace;CUDA(cudaMalloc(&workspace,workspace_bytes));BLAS(cublasSetWorkspace(h,workspace,workspace_bytes));
 for(int n:{1408,2816}){int k=n==1408?2816:704,lanes=n==1408?16:32;__nv_bfloat16 *dx,*dw,*out,*ref;CUDA(cudaMalloc(&dx,k*2));CUDA(cudaMalloc(&dw,size_t(n)*k*2));CUDA(cudaMalloc(&out,n*2));CUDA(cudaMalloc(&ref,n*2));
  for(int seed=-2;seed<(stress?64:4);++seed){std::vector<unsigned short>x,w,official;std::string name;
   if(seed<0){name="encoder-"+std::to_string((seed==-2?4:6)+(n==2816));x=load(std::string(argv[1])+"/"+name+".input.bf16",k);w=load(std::string(argv[1])+"/"+name+".weight.bf16",size_t(n)*k);official=load(std::string(argv[1])+"/"+name+".official.bf16",n);}else{x=dense(k,17+seed*113);w=dense(size_t(n)*k,29+seed*127);}
   if(stress&&seed>=4){
    unsigned mode=(seed-4)%5;
    if(mode==0){ // Signed values spanning eighty binary exponents.
     for(size_t i=0;i<x.size();++i)x[i]=(x[i]&0x807f)|((87+(i*31+seed)%81)<<7);
     for(size_t i=0;i<w.size();++i)w[i]=(w[i]&0x807f)|((87+(i*17+seed)%81)<<7);
    }else if(mode==1){ // BF16 subnormal inputs multiplied into the normal range.
     for(size_t i=0;i<x.size();++i)x[i]=(x[i]&0x8000)|((i+seed)%127+1);
     for(size_t i=0;i<w.size();++i)w[i]=(w[i]&0x807f)|((190+i%20)<<7);
    }else if(mode==2){ // Exact cancellation mixed with low-order residue.
     std::fill(x.begin(),x.end(),0x3f80);
     for(size_t i=0;i<w.size();i+=2){unsigned short v=w[i];w[i+1]=v^0x8000;}
     for(int col=0;col<n;++col)w[size_t(col)*k+(col+seed)%k]=0x3380;
    }else if(mode==3){ // Midpoint and adjacent BF16 rounding witnesses.
     std::fill(x.begin(),x.end(),0x3f80);std::fill(w.begin(),w.end(),0);
     for(int col=0;col<n;++col){auto base=size_t(col)*k;w[base+(col*7)%k]=0x3f80+(col%127);w[base+(col*7+1)%k]=0x3b80;w[base+(col*7+2)%k]=(col%2?0xb300:0x3300);}
    }else{ // Gradual underflow and signed zero.
     for(size_t i=0;i<x.size();++i)x[i]=(x[i]&0x807f)|((60+i%12)<<7);
     for(size_t i=0;i<w.size();++i)w[i]=(w[i]&0x807f)|((60+i%12)<<7);
    }
   }
   CUDA(cudaMemcpyAsync(dx,x.data(),x.size()*2,cudaMemcpyHostToDevice,stream));CUDA(cudaMemcpyAsync(dw,w.data(),w.size()*2,cudaMemcpyHostToDevice,stream));float alpha=1,beta=0;BLAS(cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N,n,1,k,&alpha,dw,CUDA_R_16BF,k,0,dx,CUDA_R_16BF,k,k,&beta,ref,CUDA_R_16BF,n,n,1,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));CUDA(cudaStreamSynchronize(stream));std::vector<unsigned short>b(n),a(n);CUDA(cudaMemcpy(b.data(),ref,n*2,cudaMemcpyDeviceToHost));
   for(int grain:{1,2,4,8,16,22,32,64,128})for(int order:{0,1}){if(stress&&(grain!=(n==1408?1:22)||order!=0))continue; if(stress){EtExpertSplitKDescriptor d={(unsigned long long)dx,(unsigned long long)dw,(unsigned long long)out,0,1,1,(unsigned)k,0,0,{0,0,0}};CUDA(cudaMemcpyAsync(descriptor,&d,sizeof(d),cudaMemcpyHostToDevice,stream));et_expert_gemv_compute<<<(n+3)/4,dim3(lanes,4),0,stream>>>(descriptor,1,n,k);}else candidate<<<(n+3)/4,dim3(lanes,4),0,stream>>>(dx,dw,out,n,k,grain,order);CUDA(cudaStreamSynchronize(stream));CUDA(cudaMemcpy(a.data(),out,n*2,cudaMemcpyDeviceToHost));size_t exact=0,oe=0,ce=0;for(int i=0;i<n;++i){exact+=a[i]==b[i];if(!official.empty()){oe+=a[i]==official[i];ce+=b[i]==official[i];}}if(stress){mismatches+=n-exact;if(!official.empty())mismatches+=n-oe;}std::printf("{\"n\":%d,\"k\":%d,\"seed\":%d,\"grain\":%d,\"order\":%d,\"exact\":%zu,\"elements\":%d,\"officialExact\":%zu,\"cublasOfficial\":%zu}\n",n,k,seed,grain,order,exact,n,oe,ce);}
  }
  CUDA(cudaFree(dx));CUDA(cudaFree(dw));CUDA(cudaFree(out));CUDA(cudaFree(ref));
 }
 BLAS(cublasDestroy(h));CUDA(cudaFree(workspace));CUDA(cudaFree(descriptor));CUDA(cudaStreamDestroy(stream));return mismatches?2:0;
}

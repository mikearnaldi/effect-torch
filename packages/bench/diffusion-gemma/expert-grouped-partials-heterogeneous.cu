// Mixed-row, mixed-split grouped partials exactness diagnostic; compilation as
// expert-grouped-partials.cu. Compares raw groups, equal-geometry compression, and ordinary streams.
#define main batching_diagnostic_main
#include "expert-gemm-batching.cu"
#undef main
#include <sstream>
#include <tuple>
struct Reduction { size_t offset, output; unsigned count, slices; };
__global__ void reduce_mixed(const __nv_bfloat16* partial,__nv_bfloat16* output,const Reduction* descriptors) {
  const auto d=descriptors[blockIdx.x];
  if(!d.slices)return;
  for(unsigned i=threadIdx.x;i<d.count;i+=blockDim.x){float value=0;for(unsigned s=0;s<d.slices;++s)value+=__bfloat162float(partial[d.offset+size_t(s)*d.count+i]);output[d.output+i]=__float2bfloat16_rn(value);}
}
std::pair<int,int> geometry(int m,int k){
  if(k==704)return m>=257&&m<=416?std::pair<int,int>{2,384}:std::pair<int,int>{1,704};
  if(m<=16||(m>=65&&m<=128)||(m>=257&&m<=448))return {1,2816};
  if(m<=30)return {8,384};
  if(m<=32||(m>=43&&m<=57)||(m>=193&&m<=256)||m>=449)return {4,704};
  if(m<=64)return {2,1408};return {5,576};
}
int main(int argc,char**argv){
  int projection=argc>1?std::atoi(argv[1]):0,seed=argc>2?std::atoi(argv[2]):0;
  int count=argc>3?std::atoi(argv[3]):128,pattern=argc>4?std::atoi(argv[4]):0;
  std::vector<int> recorded;
  if(argc>5){std::istringstream stream(argv[5]);std::string word;while(std::getline(stream,word,','))recorded.push_back(std::stoi(word));count=recorded.size();}
  if(count<1||count>128)return 2;int columns=projection?2816:1408,inner=projection?704:2816;
  size_t wc=size_t(columns)*inner,xc=0,oc=0,pc=0;
  std::vector<int> rows(count);std::vector<size_t> xo(count),oo(count),po(count);std::vector<Reduction> reduction;
  for(int b=0;b<count;++b){rows[b]=recorded.empty()?2+(b*37+seed*13)%511:recorded[b];if(rows[b]<1||rows[b]>512)return 2;auto mapped=geometry(rows[b],inner);if(rows[b]==1)mapped.first=0;xo[b]=xc;oo[b]=oc;po[b]=pc;size_t elements=size_t(rows[b])*columns;reduction.push_back({pc,oc,unsigned(elements),unsigned(mapped.first)});xc+=size_t(rows[b])*inner;oc+=elements;pc+=elements*mapped.first;}
  unsigned short *x,*w,*out,*partial;CUDA(cudaMalloc(&x,xc*2));CUDA(cudaMalloc(&w,count*wc*2));CUDA(cudaMalloc(&out,oc*2));CUDA(cudaMalloc(&partial,pc*2));
  std::vector<unsigned short> expected(oc),actual(oc);
  cublasHandle_t h;BLAS(cublasCreate(&h));void*workspace;CUDA(cudaMalloc(&workspace,workspace_bytes));BLAS(cublasSetWorkspace(h,workspace,workspace_bytes));float one=1,zero=0;
  std::vector<const void*>xp,wp;std::vector<void*>op;std::vector<int>m,n,k,ld,ldout,size;
  for(int b=0;b<count;++b){auto hx=dense(size_t(rows[b])*inner,17+seed*113+b*37),hw=dense(wc,29+seed*127+b*43);
    for(auto* data:{&hx,&hw})for(size_t i=0;i<data->size();++i){if(pattern==1)(*data)[i]=((*data)[i]&0x807f)|((100+(i*37+seed*11+b)%36)<<7);if(pattern==2){const unsigned short edges[]={0,0x8000,1,0x8001,0x007f,0x807f,0x0080,0x8080,0x3f80,0xbf80,0x3f81,0xbf81};(*data)[i]=edges[(i*7+seed+b)%12];}}
    CUDA(cudaMemcpy(x+xo[b],hx.data(),hx.size()*2,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(w+b*wc,hw.data(),wc*2,cudaMemcpyHostToDevice));
    BLAS(cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N,columns,rows[b],inner,&one,w+b*wc,CUDA_R_16BF,inner,0,x+xo[b],CUDA_R_16BF,inner,size_t(rows[b])*inner,&zero,out+oo[b],CUDA_R_16BF,columns,size_t(rows[b])*columns,1,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    auto mapped=geometry(rows[b],inner);if(rows[b]==1)mapped.first=0;for(int s=0;s<mapped.first;++s){m.push_back(columns);n.push_back(rows[b]);k.push_back(std::min(mapped.second,inner-s*mapped.second));ld.push_back(inner);ldout.push_back(columns);size.push_back(1);xp.push_back(x+xo[b]+s*mapped.second);wp.push_back(w+b*wc+s*mapped.second);op.push_back(partial+po[b]+size_t(s)*rows[b]*columns);}
  }
  CUDA(cudaMemcpy(expected.data(),out,oc*2,cudaMemcpyDeviceToHost));const void**dx,**dw;void**dout;Reduction*dr;
  CUDA(cudaMalloc(&dx,xp.size()*sizeof(void*)));CUDA(cudaMalloc(&dw,wp.size()*sizeof(void*)));CUDA(cudaMalloc(&dout,op.size()*sizeof(void*)));CUDA(cudaMalloc(&dr,count*sizeof(Reduction)));
  CUDA(cudaMemcpy(dx,xp.data(),xp.size()*sizeof(void*),cudaMemcpyHostToDevice));CUDA(cudaMemcpy(dw,wp.data(),wp.size()*sizeof(void*),cudaMemcpyHostToDevice));CUDA(cudaMemcpy(dout,op.data(),op.size()*sizeof(void*),cudaMemcpyHostToDevice));CUDA(cudaMemcpy(dr,reduction.data(),count*sizeof(Reduction),cudaMemcpyHostToDevice));
  std::vector<float>alpha(m.size(),1),beta(m.size(),0);std::vector<cublasOperation_t>ta(m.size(),CUBLAS_OP_T),tb(m.size(),CUBLAS_OP_N);
  std::vector<size_t> order(m.size());for(size_t i=0;i<order.size();++i)order[i]=i;
  std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return std::tie(m[a],n[a],k[a],ld[a])<std::tie(m[b],n[b],k[b],ld[b]);});
  std::vector<int>cm,cn,ck,cld,cldo,cs;std::vector<const void*>cx,cw;std::vector<void*>co;
  size_t previous=0;
  for(size_t position=0;position<order.size();++position){size_t i=order[position];if(position==0||std::tie(m[i],n[i],k[i],ld[i])!=std::tie(m[previous],n[previous],k[previous],ld[previous])){cm.push_back(m[i]);cn.push_back(n[i]);ck.push_back(k[i]);cld.push_back(ld[i]);cldo.push_back(ldout[i]);cs.push_back(0);}++cs.back();cx.push_back(xp[i]);cw.push_back(wp[i]);co.push_back(op[i]);previous=i;}
  const void**cdx,**cdw;void**cdo;CUDA(cudaMalloc(&cdx,cx.size()*sizeof(void*)));CUDA(cudaMalloc(&cdw,cw.size()*sizeof(void*)));CUDA(cudaMalloc(&cdo,co.size()*sizeof(void*)));
  CUDA(cudaMemcpy(cdx,cx.data(),cx.size()*sizeof(void*),cudaMemcpyHostToDevice));CUDA(cudaMemcpy(cdw,cw.data(),cw.size()*sizeof(void*),cudaMemcpyHostToDevice));CUDA(cudaMemcpy(cdo,co.data(),co.size()*sizeof(void*),cudaMemcpyHostToDevice));
  cudaStream_t primary;CUDA(cudaStreamCreateWithFlags(&primary,cudaStreamNonBlocking));BLAS(cublasSetStream(h,primary));BLAS(cublasSetWorkspace(h,workspace,workspace_bytes));
  Worker worker[workers];for(auto& item:worker){CUDA(cudaStreamCreateWithFlags(&item.stream,cudaStreamNonBlocking));BLAS(cublasCreate(&item.handle));BLAS(cublasSetStream(item.handle,item.stream));CUDA(cudaMalloc(&item.workspace,workspace_bytes));BLAS(cublasSetWorkspace(item.handle,item.workspace,workspace_bytes));CUDA(cudaEventCreate(&item.done));}
  cudaEvent_t start,end;CUDA(cudaEventCreate(&start));CUDA(cudaEventCreate(&end));
  auto run=[&](int mode,bool synchronize=true){CUDA(cudaEventRecord(start,primary));for(auto& item:worker)CUDA(cudaStreamWaitEvent(item.stream,start,0));
    for(int b=0;b<count;++b)if(mode==2||rows[b]==1)BLAS(cublasGemmStridedBatchedEx(worker[b%workers].handle,CUBLAS_OP_T,CUBLAS_OP_N,columns,rows[b],inner,&one,w+b*wc,CUDA_R_16BF,inner,0,x+xo[b],CUDA_R_16BF,inner,size_t(rows[b])*inner,&zero,out+oo[b],CUDA_R_16BF,columns,size_t(rows[b])*columns,1,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    if(mode!=2){auto& mm=mode==0?m:cm;auto& nn=mode==0?n:cn;auto& kk=mode==0?k:ck;auto& ll=mode==0?ld:cld;auto& lo=mode==0?ldout:cldo;auto& sizes=mode==0?size:cs;
      BLAS(cublasGemmGroupedBatchedEx(h,ta.data(),tb.data(),mm.data(),nn.data(),kk.data(),alpha.data(),mode==0?dw:cdw,CUDA_R_16BF,ll.data(),mode==0?dx:cdx,CUDA_R_16BF,ll.data(),beta.data(),mode==0?dout:cdo,CUDA_R_16BF,lo.data(),mm.size(),sizes.data(),CUBLAS_COMPUTE_32F));reduce_mixed<<<count,256,0,primary>>>((__nv_bfloat16*)partial,(__nv_bfloat16*)out,dr);CUDA(cudaGetLastError());}
    for(auto& item:worker){CUDA(cudaEventRecord(item.done,item.stream));CUDA(cudaStreamWaitEvent(primary,item.done,0));}CUDA(cudaEventRecord(end,primary));if(!synchronize)return 0.0f;CUDA(cudaEventSynchronize(end));float ms;CUDA(cudaEventElapsedTime(&ms,start,end));return ms;
  };
  for(int mode=0;mode<3;++mode){run(mode);CUDA(cudaMemcpy(actual.data(),out,oc*2,cudaMemcpyDeviceToHost));size_t exact=0;for(size_t i=0;i<oc;++i)exact+=actual[i]==expected[i];
    const char*label=mode==0?"group_size1":mode==1?"sorted_compressed":"ordinary32streams";
    std::printf("{\"projection\":%d,\"seed\":%d,\"pattern\":%d,\"experts\":%d,\"groups\":%zu,\"mode\":\"%s\",\"exact\":%zu,\"elements\":%zu",projection,seed,pattern,count,mode==0?m.size():mode==1?cm.size():size_t(count),label,exact,oc);
    if(exact!=oc){std::printf("}\n");return 4;}
    if(!recorded.empty()){std::vector<float>times;for(int i=-3;i<11;++i){float ms=run(mode);if(i>=0)times.push_back(ms);}std::sort(times.begin(),times.end());std::printf(",\"gpuMedianMs\":%.6f",times[times.size()/2]);
      cudaGraph_t graph;cudaGraphExec_t executable;
      CUDA(cudaStreamBeginCapture(primary,cudaStreamCaptureModeGlobal));run(mode,false);CUDA(cudaStreamEndCapture(primary,&graph));CUDA(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
      cudaEvent_t gs,ge;CUDA(cudaEventCreate(&gs));CUDA(cudaEventCreate(&ge));times.clear();for(int i=-3;i<11;++i){CUDA(cudaEventRecord(gs,primary));CUDA(cudaGraphLaunch(executable,primary));CUDA(cudaEventRecord(ge,primary));CUDA(cudaEventSynchronize(ge));float ms;CUDA(cudaEventElapsedTime(&ms,gs,ge));if(i>=0)times.push_back(ms);}std::sort(times.begin(),times.end());
      CUDA(cudaMemcpy(actual.data(),out,oc*2,cudaMemcpyDeviceToHost));size_t graph_exact=0;for(size_t i=0;i<oc;++i)graph_exact+=actual[i]==expected[i];
      std::printf(",\"graphMedianMs\":%.6f,\"graphExact\":%zu",times[times.size()/2],graph_exact);if(graph_exact!=oc){std::printf("}\n");return 5;}
      CUDA(cudaEventDestroy(gs));CUDA(cudaEventDestroy(ge));CUDA(cudaGraphExecDestroy(executable));CUDA(cudaGraphDestroy(graph));}
    std::printf("}\n");
  }
  return 0;
}

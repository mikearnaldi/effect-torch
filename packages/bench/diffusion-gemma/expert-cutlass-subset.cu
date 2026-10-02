// Direct-output replacement of ONLY the production exact grouped subset.
// Immutable metadata setup excluded; fallback scheduling and workspaces match
// production: grouped worker0, descending-row minimum-load workers1..31.
#include "expert-cutlass-grouped.cuh"
#define main unused_batching_main
#include "expert-gemm-batching.cu"
#undef main
#include <sstream>

int main(int argc, char** argv) {
  if (argc != 5) return 2; // projection, seed, edge-pattern, comma-separated rows
  const int projection=std::atoi(argv[1]), seed=std::atoi(argv[2]), pattern=std::atoi(argv[3]);
  if (projection<0 || projection>1 || pattern<0 || pattern>2) return 2;
  std::vector<int> rows; std::istringstream csv(argv[4]); std::string word;
  while(std::getline(csv,word,',')) rows.push_back(std::stoi(word));
  const int count=rows.size(), columns=projection?2816:1408, inner=projection?704:2816;
  if(count<1 || count>128) return 2;
  auto eligible=[&](int i){return rows[i]>=2 && rows[i]<=(projection?128:16);};
  int selected=0; for(int i=0;i<count;++i) {if(rows[i]<1||rows[i]>512)return 2;selected+=eligible(i);}
  const bool grouped=selected>=4;
  size_t xc=0,oc=0,wc=size_t(columns)*inner;
  std::vector<size_t> xo(count),oo(count);
  for(int i=0;i<count;++i){xo[i]=xc;oo[i]=oc;xc+=size_t(rows[i])*inner;oc+=size_t(rows[i])*columns;}
  unsigned short *x,*w,*out;
  CUDA(cudaMalloc(&x,xc*2));CUDA(cudaMalloc(&w,count*wc*2));CUDA(cudaMalloc(&out,oc*2));
  cudaStream_t primary;CUDA(cudaStreamCreateWithFlags(&primary,cudaStreamNonBlocking));
  Worker worker[workers];
  for(auto& v:worker){CUDA(cudaStreamCreateWithFlags(&v.stream,cudaStreamNonBlocking));BLAS(cublasCreate(&v.handle));BLAS(cublasSetStream(v.handle,v.stream));CUDA(cudaMalloc(&v.workspace,workspace_bytes));BLAS(cublasSetWorkspace(v.handle,v.workspace,workspace_bytes));CUDA(cudaEventCreate(&v.done));}
  const float one=1,zero=0;
  auto ordinary=[&](int i,int wi){BLAS(cublasGemmStridedBatchedEx(worker[wi].handle,CUBLAS_OP_T,CUBLAS_OP_N,columns,rows[i],inner,&one,w+i*wc,CUDA_R_16BF,inner,0,x+xo[i],CUDA_R_16BF,inner,size_t(rows[i])*inner,&zero,out+oo[i],CUDA_R_16BF,columns,size_t(rows[i])*columns,1,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));};
  for(int i=0;i<count;++i){
    auto hx=dense(size_t(rows[i])*inner,17+seed*113+i*37),hw=dense(wc,29+seed*127+i*43);
    for(auto* data:{&hx,&hw})for(size_t j=0;j<data->size();++j){
      if(pattern==1)(*data)[j]=((*data)[j]&0x807f)|((100+(j*37+seed*11+i)%36)<<7);
      if(pattern==2){const unsigned short edges[]={0,0x8000,1,0x8001,0x007f,0x807f,0x0080,0x8080,0x3f80,0xbf80,0x3f81,0xbf81};(*data)[j]=edges[(j*7+seed+i)%12];}
    }
    CUDA(cudaMemcpy(x+xo[i],hx.data(),hx.size()*2,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(w+i*wc,hw.data(),wc*2,cudaMemcpyHostToDevice));
  }
  // Pageable HtoD copies may return after staging; nonblocking workers do not
  // inherit legacy-stream ordering. Finish every upload before the reference.
  CUDA(cudaDeviceSynchronize());
  for(int i=0;i<count;++i)ordinary(i,0);
  CUDA(cudaStreamSynchronize(worker[0].stream));
  std::vector<unsigned short> expected(oc),actual(oc);CUDA(cudaMemcpy(expected.data(),out,oc*2,cudaMemcpyDeviceToHost));
  std::vector<const void*> xp,wp;std::vector<void*> op;
  std::vector<int> m,n,k,ld,ldout,sizes;
  for(int i=0;i<count;++i)if(grouped&&eligible(i)){
    xp.push_back(x+xo[i]);wp.push_back(w+i*wc);op.push_back(out+oo[i]);
    m.push_back(columns);n.push_back(rows[i]);k.push_back(inner);ld.push_back(inner);ldout.push_back(columns);sizes.push_back(1);
  }
  const void **dx=nullptr,**dw=nullptr;void** dout=nullptr;
  if(grouped){CUDA(cudaMalloc(&dx,selected*sizeof(void*)));CUDA(cudaMalloc(&dw,selected*sizeof(void*)));CUDA(cudaMalloc(&dout,selected*sizeof(void*)));
    CUDA(cudaMemcpy(dx,xp.data(),selected*sizeof(void*),cudaMemcpyHostToDevice));CUDA(cudaMemcpy(dw,wp.data(),selected*sizeof(void*),cudaMemcpyHostToDevice));CUDA(cudaMemcpy(dout,op.data(),selected*sizeof(void*),cudaMemcpyHostToDevice));}
  std::vector<float> alpha(selected,1),beta(selected,0);
  std::vector<cublasOperation_t> ta(selected,CUBLAS_OP_T),tb(selected,CUBLAS_OP_N);
  std::vector<int> fallback;for(int i=0;i<count;++i)if(!grouped||!eligible(i))fallback.push_back(i);
  std::stable_sort(fallback.begin(),fallback.end(),[&](int a,int b){return rows[a]>rows[b];});
  const int first=grouped?1:0, active=std::min(int(fallback.size()),workers-first);
  std::vector<std::vector<int>> partitions(active);std::vector<int> loads(active);
  for(int i:fallback){int slot=std::min_element(loads.begin(),loads.end())-loads.begin();partitions[slot].push_back(i);loads[slot]+=rows[i];}
  CUDA(cudaDeviceSynchronize()); // Publish immutable pointer banks to all streams.
  cudaEvent_t start,end;CUDA(cudaEventCreate(&start));CUDA(cudaEventCreate(&end));
  auto submit=[&](int mode,bool synchronize=true){
    CUDA(cudaEventRecord(start,primary));
    for(int i=0;i<first+active;++i)CUDA(cudaStreamWaitEvent(worker[i].stream,start,0));
    if(grouped){
#define SUBSET_ARGS worker[0].handle,ta.data(),tb.data(),m.data(),n.data(),k.data(),alpha.data(),dw,CUDA_R_16BF,ld.data(),dx,CUDA_R_16BF,ld.data(),beta.data(),dout,CUDA_R_16BF,ldout.data(),selected,sizes.data(),CUBLAS_COMPUTE_32F
      if(mode==0) { BLAS(cublasGemmGroupedBatchedEx(SUBSET_ARGS)); }
      else { BLAS(et_cutlass_grouped(SUBSET_ARGS)); }
#undef SUBSET_ARGS
    }
    for(int slot=0;slot<active;++slot)for(int i:partitions[slot])ordinary(i,slot+first);
    for(int i=0;i<first+active;++i){CUDA(cudaEventRecord(worker[i].done,worker[i].stream));CUDA(cudaStreamWaitEvent(primary,worker[i].done,0));}
    CUDA(cudaEventRecord(end,primary));if(!synchronize)return 0.0f;
    CUDA(cudaEventSynchronize(end));float ms;CUDA(cudaEventElapsedTime(&ms,start,end));return ms;
  };
  auto check=[&](){CUDA(cudaMemcpy(actual.data(),out,oc*2,cudaMemcpyDeviceToHost));size_t equal=0;for(size_t i=0;i<oc;++i)equal+=actual[i]==expected[i];return equal;};
  for(int mode=0;mode<2;++mode){
    submit(mode);size_t exact=check();if(exact!=oc){
      for(int i=0;i<count;++i){size_t bad=0;for(size_t j=0;j<size_t(rows[i])*columns;++j)bad+=actual[oo[i]+j]!=expected[oo[i]+j];if(bad)std::fprintf(stderr,"expert=%d rows=%d grouped=%d mismatches=%zu\n",i,rows[i],eligible(i),bad);}
      std::printf("{\"mode\":%d,\"exact\":%zu,\"elements\":%zu}\n",mode,exact,oc);return 4;}
    std::vector<float> gpu;std::vector<double> wall;
    for(int i=-3;i<15;++i){auto begin=std::chrono::steady_clock::now();float ms=submit(mode);double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();if(i>=0){gpu.push_back(ms);wall.push_back(elapsed);}}
    std::sort(gpu.begin(),gpu.end());std::sort(wall.begin(),wall.end());
    cudaGraph_t graph;cudaGraphExec_t exec;
    CUDA(cudaStreamBeginCapture(primary,cudaStreamCaptureModeGlobal));submit(mode,false);CUDA(cudaStreamEndCapture(primary,&graph));CUDA(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    cudaEvent_t gs,ge;CUDA(cudaEventCreate(&gs));CUDA(cudaEventCreate(&ge));std::vector<float> graphs;
    for(int i=-3;i<15;++i){CUDA(cudaEventRecord(gs,primary));CUDA(cudaGraphLaunch(exec,primary));CUDA(cudaEventRecord(ge,primary));CUDA(cudaEventSynchronize(ge));float ms;CUDA(cudaEventElapsedTime(&ms,gs,ge));if(i>=0)graphs.push_back(ms);}
    std::sort(graphs.begin(),graphs.end());size_t graph_exact=check();
    std::printf("{\"projection\":%d,\"seed\":%d,\"pattern\":%d,\"experts\":%d,\"selected\":%d,\"mode\":\"%s\",\"exact\":%zu,\"elements\":%zu,\"graphExact\":%zu,\"gpuMedianMs\":%.6f,\"wallMedianMs\":%.6f,\"graphMedianMs\":%.6f}\n",projection,seed,pattern,count,selected,mode==0?"production_grouped":"cutlass_subset",exact,oc,graph_exact,gpu[gpu.size()/2],wall[wall.size()/2],graphs[graphs.size()/2]);
    if(graph_exact!=oc)return 5;
    CUDA(cudaGraphExecDestroy(exec));CUDA(cudaGraphDestroy(graph));CUDA(cudaEventDestroy(gs));CUDA(cudaEventDestroy(ge));
  }
  return 0;
}

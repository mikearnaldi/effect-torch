// Fixed device metadata versus production host-count/grouped/ordinary dispatch.
#include "expert-cutlass-device.cuh"
#define main unused_batching_main
#include "expert-gemm-batching.cu"
#undef main
#include <sstream>

template<class T> T* device_alloc(size_t count){T* p;CUDA(cudaMalloc(&p,count*sizeof(T)));return p;}
struct HostPointerBank { const void* x[128];const void* w[128];void* out[128];const void** dx;const void** dw;void** dout;int count; };
__global__ void upload_host_pointer_bank(HostPointerBank bank){
  int i=threadIdx.x;if(i<bank.count){bank.dx[i]=bank.x[i];bank.dw[i]=bank.w[i];bank.dout[i]=bank.out[i];}
}

int main(int argc,char** argv){
  if(argc!=5)return 2;
  int projection=std::atoi(argv[1]),seed=std::atoi(argv[2]),pattern=std::atoi(argv[3]);
  if(projection<0||projection>1||pattern<0||pattern>11)return 2;
  std::vector<int> rows;std::string word;std::istringstream csv(argv[4]);
  while(std::getline(csv,word,','))rows.push_back(std::stoi(word));
  int count=rows.size(),n=projection?2816:1408,k=projection?704:2816;
  if(count<1||count>128)return 2;
  size_t xc=0,oc=0,wc=size_t(n)*k;std::vector<size_t> xo(count),oo(count);
  for(int i=0;i<count;++i){if(rows[i]<0||rows[i]>512)return 2;xo[i]=xc;oo[i]=oc;xc+=size_t(rows[i])*k;oc+=size_t(rows[i])*n;}
  if(!oc)return 2;
  auto* x=device_alloc<EtBf16>(xc);auto* w=device_alloc<EtBf16>(count*wc);
  auto* out=device_alloc<EtBf16>(oc);auto* partial=device_alloc<EtBf16>(oc*8);
  auto* counts=device_alloc<int>(count);
  CUDA(cudaMemcpy(counts,rows.data(),count*sizeof(int),cudaMemcpyHostToDevice));
#if defined(ET_DEVICE_LOSSLESS) && ET_DEVICE_LOSSLESS
  std::vector<EtLosslessContext> lossless_contexts;
#endif
  for(int i=0;i<count;++i){
    auto hx=dense(size_t(rows[i])*k,17+seed*113+i*37),hw=dense(wc,29+seed*127+i*43);
    for(auto* data:{&hx,&hw})for(size_t j=0;j<data->size();++j){
      if(pattern==1)(*data)[j]=((*data)[j]&0x807f)|((100+(j*37+seed*11+i)%36)<<7);
      if(pattern==2){const unsigned short edges[]={0,0x8000,1,0x8001,0x007f,0x807f,0x0080,0x8080,0x3f80,0xbf80,0x3f81,0xbf81};(*data)[j]=edges[(j*7+seed+i)%12];}
    }
    if(pattern==3){
      for(size_t j=0;j<hx.size();++j)hx[j]=(hx[j]&0x807f)|((87+(j*31+seed)%81)<<7);
      for(size_t j=0;j<hw.size();++j)hw[j]=(hw[j]&0x807f)|((87+(j*17+seed)%81)<<7);
    }else if(pattern==4){
      for(size_t j=0;j<hx.size();++j)hx[j]=(hx[j]&0x8000)|((j+seed)%127+1);
      for(size_t j=0;j<hw.size();++j)hw[j]=(hw[j]&0x807f)|((190+j%20)<<7);
    }else if(pattern==5){
      std::fill(hx.begin(),hx.end(),0x3f80);
      for(size_t j=0;j<hw.size();j+=2)hw[j+1]=hw[j]^0x8000;
      for(int col=0;col<n;++col)hw[size_t(col)*k+(col+seed)%k]=0x3380;
    }else if(pattern==6){
      std::fill(hx.begin(),hx.end(),0x3f80);std::fill(hw.begin(),hw.end(),0);
      for(int col=0;col<n;++col){auto base=size_t(col)*k;hw[base+(col*7)%k]=0x3f80+(col%127);hw[base+(col*7+1)%k]=0x3b80;hw[base+(col*7+2)%k]=col%2?0xb300:0x3300;}
    }else if(pattern==7){
      for(size_t j=0;j<hx.size();++j)hx[j]=(hx[j]&0x807f)|((60+j%12)<<7);
      for(size_t j=0;j<hw.size();++j)hw[j]=(hw[j]&0x807f)|((60+j%12)<<7);
    }
    if(pattern>=8){
      std::fill(hx.begin(),hx.end(),pattern==10?0x7f7f:pattern==11?0x8000:0x3f80);
      for(int col=0;col<n;++col)for(int t=0;t<k;++t){
        unsigned short value;
        if(pattern==8){const unsigned short values[]={0x7f80,0xff80,0x3f80,0xbf80};value=values[col%4];}
        else if(pattern==9){const unsigned short values[]={0x7fc1,0xffc1,0x7f81,0xff81};value=values[col%4];}
        else if(pattern==10)value=col%2?0xbf80:0x3f80;
        else value=(col+t)%2?0xbf80:0x3f80;
        hw[size_t(col)*k+t]=value;
      }
    }
    if(!hx.empty())CUDA(cudaMemcpy(x+xo[i],hx.data(),hx.size()*2,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(w+i*wc,hw.data(),wc*2,cudaMemcpyHostToDevice));
#if defined(ET_DEVICE_LOSSLESS) && ET_DEVICE_LOSSLESS
    lossless_contexts.push_back(et_lossless_upload(hw,w+i*wc));
#endif
  }
#if defined(ET_DEVICE_LOSSLESS) && ET_DEVICE_LOSSLESS
  et_lossless_bind(lossless_contexts,w,wc*2);
#endif
  cudaStream_t primary;CUDA(cudaStreamCreateWithFlags(&primary,cudaStreamNonBlocking));
  Worker worker[workers];for(auto& v:worker){CUDA(cudaStreamCreateWithFlags(&v.stream,cudaStreamNonBlocking));BLAS(cublasCreate(&v.handle));BLAS(cublasSetStream(v.handle,v.stream));CUDA(cudaMalloc(&v.workspace,workspace_bytes));BLAS(cublasSetWorkspace(v.handle,v.workspace,workspace_bytes));CUDA(cudaEventCreate(&v.done));}
  CUDA(cudaDeviceSynchronize());
  float one=1,zero=0;
  auto ordinary=[&](int expert,int wi,int m,size_t offset){BLAS(cublasGemmStridedBatchedEx(worker[wi].handle,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&one,w+expert*wc,CUDA_R_16BF,k,0,x+offset*k,CUDA_R_16BF,k,size_t(m)*k,&zero,out+offset*n,CUDA_R_16BF,n,size_t(m)*n,1,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));};
  for(int i=0;i<count;++i)if(rows[i])ordinary(i,0,rows[i],xo[i]/k);
  CUDA(cudaStreamSynchronize(worker[0].stream));std::vector<unsigned short> expected(oc),actual(oc);
  CUDA(cudaMemcpy(expected.data(),out,oc*2,cudaMemcpyDeviceToHost));

  // Stable invocation-owned metadata addresses; contents rebuilt on GPU every call.
  int problems=count*8;
  DeviceExpertMetadata meta{device_alloc<cutlass::gemm::GemmCoord>(problems),
    device_alloc<EtBf16*>(problems),device_alloc<EtBf16*>(problems),device_alloc<EtBf16*>(problems),
    device_alloc<int64_t>(problems),device_alloc<int64_t>(problems),device_alloc<int64_t>(problems),
    device_alloc<int>(count),device_alloc<int>(1),device_alloc<int>(count),device_alloc<int>(1),nullptr};
  cudaDeviceProp properties;CUDA(cudaGetDeviceProperties(&properties,0));
  int blocks_per_sm=2;
  if(const char* configured=std::getenv("ET_DEVICE_BLOCKS_PER_SM")){
    char* end=nullptr;long parsed=std::strtol(configured,&end,10);
    if(!end||*end||parsed<1||parsed>8){std::fprintf(stderr,"ET_DEVICE_BLOCKS_PER_SM must be 1..8\n");return 2;}
    blocks_per_sm=int(parsed);
  }
  int persistent_blocks=properties.multiProcessorCount*blocks_per_sm;
  int gemv_blocks=properties.multiProcessorCount*16;
  bool overlap_m1=!(std::getenv("ET_DEVICE_M1_OVERLAP")&&std::string(std::getenv("ET_DEVICE_M1_OVERLAP"))=="0");
  bool compact=!(std::getenv("ET_DEVICE_COMPACT")&&std::string(std::getenv("ET_DEVICE_COMPACT"))=="0");
  bool device_params=std::getenv("ET_DEVICE_PARAMS")&&std::string(std::getenv("ET_DEVICE_PARAMS"))=="1";
  bool merge_slices=std::getenv("ET_DEVICE_MERGE_SLICES")&&std::string(std::getenv("ET_DEVICE_MERGE_SLICES"))=="1";
  if(merge_slices)compact=true;
  bool cublas_m1=std::getenv("ET_DEVICE_CUBLAS_M1")&&std::string(std::getenv("ET_DEVICE_CUBLAS_M1"))=="1";
  int m1_workers=0;for(int m:rows)if(m==1)++m1_workers;m1_workers=std::min(m1_workers,workers);
  bool cached_counts=projection==1&&std::getenv("ET_CACHED_COUNTS")&&std::string(std::getenv("ET_CACHED_COUNTS"))=="1";
  bool shared_gemv=projection==1&&std::getenv("ET_DEVICE_GEMV_SHARED")&&std::string(std::getenv("ET_DEVICE_GEMV_SHARED"))=="1";
  build_device_experts<<<1,128,0,primary>>>(counts,count,n,k,x,w,out,partial,meta,compact,merge_slices);CUDA(cudaGetLastError());CUDA(cudaStreamSynchronize(primary));
  EtGrouped grouped_op;
  typename EtGrouped::Arguments args(meta.shapes,problems,persistent_blocks,
    typename EtEpilogue::Params(1,0),meta.a,meta.b,meta.d,meta.d,meta.lda,meta.ldb,meta.ldd,meta.ldd,nullptr);
  auto status=grouped_op.initialize(args,nullptr,primary);
  if(status!=cutlass::Status::kSuccess){std::fprintf(stderr,"CUTLASS init=%d\n",int(status));return 7;}
  typename EtGroupedKernel::Params compact_params(args);
  meta.params=device_alloc<EtGroupedKernel::Params>(1);
  CUDA(cudaMemcpy(meta.params,&compact_params,sizeof(compact_params),cudaMemcpyHostToDevice));
  CUDA(cudaDeviceSynchronize());
  size_t compact_shared=sizeof(typename EtGroupedKernel::SharedStorage);
  if(compact_shared>=48*1024)CUDA(cudaFuncSetAttribute(run_compact_experts,cudaFuncAttributeMaxDynamicSharedMemorySize,compact_shared));
  if(compact_shared>=48*1024)CUDA(cudaFuncSetAttribute(run_compact_device_params,cudaFuncAttributeMaxDynamicSharedMemorySize,compact_shared));
  if(compact_shared>=48*1024)CUDA(cudaFuncSetAttribute(run_merged_device_experts,cudaFuncAttributeMaxDynamicSharedMemorySize,compact_shared));
  cudaFuncAttributes original_attributes{},compact_attributes{},device_attributes{},merged_attributes{};
  CUDA(cudaFuncGetAttributes(&merged_attributes,run_merged_device_experts));
  int merged_active_blocks=0;
  CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&merged_active_blocks,run_merged_device_experts,EtGroupedKernel::kThreadCount,compact_shared));
  std::fprintf(stderr,"mergedShape tile=%dx%dx%d warp=%dx%dx%d stages=%d threads=%d blocksPerSM=%d occupancyBlocksPerSM=%d smCount=%d dynamicShared=%zu\n",
    ET_CUTLASS_M,ET_CUTLASS_N,ET_CUTLASS_K,ET_CUTLASS_WARP_M,ET_CUTLASS_WARP_N,ET_CUTLASS_K,
    ET_CUTLASS_STAGES,EtGroupedKernel::kThreadCount,blocks_per_sm,merged_active_blocks,properties.multiProcessorCount,compact_shared);
  std::fprintf(stderr,"mergedKernel registers=%d localBytes=%zu staticShared=%zu mergeSlices=%d\n",merged_attributes.numRegs,merged_attributes.localSizeBytes,merged_attributes.sharedSizeBytes,merge_slices);
  CUDA(cudaFuncGetAttributes(&original_attributes,cutlass::Kernel<EtGroupedKernel>));
  CUDA(cudaFuncGetAttributes(&compact_attributes,run_compact_experts));
  CUDA(cudaFuncGetAttributes(&device_attributes,run_compact_device_params));
  std::fprintf(stderr,"kernel_attributes original registers=%d localBytes=%zu staticShared=%zu; compact registers=%d localBytes=%zu staticShared=%zu; device registers=%d localBytes=%zu staticShared=%zu dynamicShared=%zu deviceParams=%d compact=%d overlapM1=%d\n",
    original_attributes.numRegs,original_attributes.localSizeBytes,original_attributes.sharedSizeBytes,
    compact_attributes.numRegs,compact_attributes.localSizeBytes,compact_attributes.sharedSizeBytes,
    device_attributes.numRegs,device_attributes.localSizeBytes,device_attributes.sharedSizeBytes,compact_shared,device_params,compact,overlap_m1);
  int* host_counts;CUDA(cudaMallocHost(&host_counts,count*sizeof(int)));
  std::copy(rows.begin(),rows.end(),host_counts);
  std::fprintf(stderr,"cachedSecondProjectionCounts=%d sharedGemv=%d\n",cached_counts,shared_gemv);
  auto** dx=device_alloc<const void*>(count);auto** dw=device_alloc<const void*>(count);auto** dout=device_alloc<void*>(count);
  cudaEvent_t start,end,ready;CUDA(cudaEventCreate(&start));CUDA(cudaEventCreate(&end));CUDA(cudaEventCreate(&ready));
  auto submit=[&](bool gpu_only,bool synchronize=true){
    CUDA(cudaEventRecord(start,primary));
    if(gpu_only){
      build_device_experts<<<1,128,0,primary>>>(counts,count,n,k,x,w,out,partial,meta,compact,merge_slices);CUDA(cudaGetLastError());
      if(cublas_m1){
        // Component bound: immutable host routing is prepared before capture.
        // Production integration must retain its first-projection count fence.
        CUDA(cudaEventRecord(ready,primary));
        for(int wi=0;wi<m1_workers;++wi)CUDA(cudaStreamWaitEvent(worker[wi].stream,ready,0));
        int job=0;for(int expert=0;expert<count;++expert)if(rows[expert]==1)
          ordinary(expert,job++%m1_workers,1,xo[expert]/k);
        for(int wi=0;wi<m1_workers;++wi)CUDA(cudaEventRecord(worker[wi].done,worker[wi].stream));
      }else{
        if(overlap_m1){CUDA(cudaEventRecord(ready,primary));CUDA(cudaStreamWaitEvent(worker[0].stream,ready,0));}
        auto gemv_stream=overlap_m1?worker[0].stream:primary;
        if(shared_gemv)gemv_device_experts_shared<<<gemv_blocks,dim3(32,4),0,gemv_stream>>>(meta.offsets,n,reinterpret_cast<__nv_bfloat16*>(x),reinterpret_cast<__nv_bfloat16*>(w),reinterpret_cast<__nv_bfloat16*>(out),meta.m1_indices,meta.m1_count);
        else gemv_device_experts<<<gemv_blocks,dim3(k==2816?16:32,4),0,gemv_stream>>>(counts,meta.offsets,n,k,reinterpret_cast<__nv_bfloat16*>(x),reinterpret_cast<__nv_bfloat16*>(w),reinterpret_cast<__nv_bfloat16*>(out),meta.m1_indices,meta.m1_count);
        CUDA(cudaGetLastError());
        if(overlap_m1)CUDA(cudaEventRecord(worker[0].done,worker[0].stream));
      }
      if(compact){
        if(merge_slices)run_merged_device_experts<<<persistent_blocks,EtGroupedKernel::kThreadCount,compact_shared,primary>>>(meta.params);
        else if(device_params)run_compact_device_params<<<persistent_blocks,EtGroupedKernel::kThreadCount,compact_shared,primary>>>(meta.params);
        else run_compact_experts<<<persistent_blocks,EtGroupedKernel::kThreadCount,compact_shared,primary>>>(compact_params,meta.problem_count);
        CUDA(cudaGetLastError());}
      else {auto status=grouped_op.run(primary);if(status!=cutlass::Status::kSuccess){std::fprintf(stderr,"CUTLASS run=%d\n",int(status));std::exit(8);}}
      if(!merge_slices)reduce_device_experts<<<dim3(count,32),256,0,primary>>>(counts,meta.offsets,n,k,reinterpret_cast<__nv_bfloat16*>(partial),reinterpret_cast<__nv_bfloat16*>(out));CUDA(cudaGetLastError());
      if(cublas_m1){for(int wi=0;wi<m1_workers;++wi)CUDA(cudaStreamWaitEvent(primary,worker[wi].done,0));}
      else if(overlap_m1)CUDA(cudaStreamWaitEvent(primary,worker[0].done,0));
    }else{
      // Same tiny count readback required by production's exact cuBLAS dispatch.
      if(!cached_counts){CUDA(cudaMemcpyAsync(host_counts,counts,count*sizeof(int),cudaMemcpyDeviceToHost,primary));CUDA(cudaStreamSynchronize(primary));}
      std::vector<int> offsets(count),eligible,fallback;int offset=0;
      for(int i=0;i<count;++i){offsets[i]=offset;offset+=host_counts[i];if(host_counts[i]>=2&&host_counts[i]<=(projection?128:16))eligible.push_back(i);}
      bool grouped=eligible.size()>=4;if(!grouped)eligible.clear();
      for(int i=0;i<count;++i)if(host_counts[i]&&(!grouped||host_counts[i]<2||host_counts[i]>(projection?128:16)))fallback.push_back(i);
      int first=grouped?1:0,active=std::min(int(fallback.size()),workers-first);
      std::vector<std::vector<int>> partitions(active);std::vector<int> loads(active);
      std::stable_sort(fallback.begin(),fallback.end(),[&](int a,int b){return host_counts[a]>host_counts[b];});
      for(int i:fallback){int slot=std::min_element(loads.begin(),loads.end())-loads.begin();partitions[slot].push_back(i);loads[slot]+=host_counts[i];}
      CUDA(cudaEventRecord(ready,primary));for(int i=0;i<first+active;++i)CUDA(cudaStreamWaitEvent(worker[i].stream,ready,0));
      if(grouped){
        int groups=eligible.size();HostPointerBank pointers{};pointers.dx=dx;pointers.dw=dw;pointers.dout=dout;pointers.count=groups;
        std::vector<int> mm(groups,n),nn,kk(groups,k),ld(groups,k),ldo(groups,n),sizes(groups,1);
        std::vector<float> alpha(groups,1),beta(groups,0);std::vector<cublasOperation_t> ta(groups,CUBLAS_OP_T),tb(groups,CUBLAS_OP_N);
        int pi=0;for(int i:eligible){nn.push_back(host_counts[i]);pointers.x[pi]=x+size_t(offsets[i])*k;pointers.w[pi]=w+i*wc;pointers.out[pi]=out+size_t(offsets[i])*n;++pi;}
        // By-value metadata upload matches production and owns host arguments
        // through submission without asynchronous host-pointer lifetimes.
        upload_host_pointer_bank<<<1,128,0,worker[0].stream>>>(pointers);CUDA(cudaGetLastError());
        BLAS(cublasSetMathMode(worker[0].handle,CUBLAS_DEFAULT_MATH));BLAS(cublasSetWorkspace(worker[0].handle,worker[0].workspace,workspace_bytes));
        BLAS(cublasGemmGroupedBatchedEx(worker[0].handle,ta.data(),tb.data(),mm.data(),nn.data(),kk.data(),alpha.data(),dw,CUDA_R_16BF,ld.data(),dx,CUDA_R_16BF,ld.data(),beta.data(),dout,CUDA_R_16BF,ldo.data(),groups,sizes.data(),CUBLAS_COMPUTE_32F));
      }
      for(int slot=0;slot<active;++slot){BLAS(cublasSetMathMode(worker[slot+first].handle,CUBLAS_DEFAULT_MATH));BLAS(cublasSetWorkspace(worker[slot+first].handle,worker[slot+first].workspace,workspace_bytes));for(int i:partitions[slot])ordinary(i,slot+first,host_counts[i],offsets[i]);}
      for(int i=0;i<first+active;++i){CUDA(cudaEventRecord(worker[i].done,worker[i].stream));CUDA(cudaStreamWaitEvent(primary,worker[i].done,0));}
    }
    CUDA(cudaEventRecord(end,primary));if(!synchronize)return 0.0f;
    CUDA(cudaEventSynchronize(end));float ms;CUDA(cudaEventElapsedTime(&ms,start,end));return ms;
  };
  auto check=[&](){CUDA(cudaMemcpy(actual.data(),out,oc*2,cudaMemcpyDeviceToHost));size_t exact=0;for(size_t i=0;i<oc;++i)exact+=actual[i]==expected[i];return exact;};
  for(bool gpu_only:{false,true}){
    submit(gpu_only);size_t exact=check();
    if(exact!=oc){for(int i=0;i<count;++i){size_t bad=0;for(size_t j=0;j<size_t(rows[i])*n;++j)bad+=actual[oo[i]+j]!=expected[oo[i]+j];if(bad)std::fprintf(stderr,"expert=%d rows=%d mismatches=%zu\n",i,rows[i],bad);}std::printf("{\"mode\":\"%s\",\"exact\":%zu,\"elements\":%zu}\n",gpu_only?"gpu_metadata":"production_host",exact,oc);return 4;}
    std::vector<float> gpu;std::vector<double> wall;
    for(int i=-3;i<15;++i){auto begin=std::chrono::steady_clock::now();float ms=submit(gpu_only);double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();if(i>=0){gpu.push_back(ms);wall.push_back(elapsed);}}
    std::sort(gpu.begin(),gpu.end());std::sort(wall.begin(),wall.end());float graph_ms=0;size_t graph_exact=exact;
    if(gpu_only){cudaGraph_t graph;cudaGraphExec_t exec;CUDA(cudaStreamBeginCapture(primary,cudaStreamCaptureModeGlobal));submit(true,false);CUDA(cudaStreamEndCapture(primary,&graph));CUDA(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));std::vector<float> times;
      for(int i=-3;i<15;++i){CUDA(cudaEventRecord(start,primary));CUDA(cudaGraphLaunch(exec,primary));CUDA(cudaEventRecord(end,primary));CUDA(cudaEventSynchronize(end));float ms;CUDA(cudaEventElapsedTime(&ms,start,end));if(i>=0)times.push_back(ms);}std::sort(times.begin(),times.end());graph_ms=times[times.size()/2];graph_exact=check();CUDA(cudaGraphExecDestroy(exec));CUDA(cudaGraphDestroy(graph));}
    std::printf("{\"projection\":%d,\"seed\":%d,\"pattern\":%d,\"experts\":%d,\"mode\":\"%s\",\"exact\":%zu,\"elements\":%zu,\"graphExact\":%zu,\"gpuMedianMs\":%.6f,\"wallMedianMs\":%.6f,\"graphMedianMs\":%.6f}\n",projection,seed,pattern,count,gpu_only?"gpu_metadata":"production_host",exact,oc,graph_exact,gpu[gpu.size()/2],wall[wall.size()/2],graph_ms);
    if(graph_exact!=oc)return 5;
  }
}

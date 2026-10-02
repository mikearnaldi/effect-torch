// One warp per token; output rank materializes U32 exactly as original chain.
extern "C" __global__ void et_expert_route_rank(CudaKernelArgs a) {
 unsigned lane=threadIdx.x&31U;et_u64 row=et_thread()/32;
 if(row>=a.integers[0])return;
 unsigned routes=(unsigned)a.integers[1];
 unsigned expert=lane<routes?((const unsigned*)a.inputs[0])[row*routes+lane]:0;
 unsigned position=lane<routes?((const unsigned*)a.inputs[1])[lane]:0;
 unsigned rank=0;
 for(unsigned c=0;c<routes;++c){
  unsigned other=__shfl_sync(0xffffffffU,expert,c),prior=__shfl_sync(0xffffffffU,position,c);
  rank+=(other<expert)||(other==expert&&prior<position);
 }
 if(lane<routes)((unsigned*)a.output)[row*routes+lane]=rank;
}

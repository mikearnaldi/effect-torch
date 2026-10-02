// Complete exact router tail; sorting keys are materialized F32 probabilities.
// Output packs F32 weights followed by U32 indices, both planned aliases.
extern "C" __global__ void et_router_tail(CudaKernelArgs a) {
 unsigned row=blockIdx.x,index=threadIdx.x,lane=index&31U;
 __shared__ float probabilities[128];
 __shared__ unsigned orders[128],indices[128],invalid;
 if(index==0)invalid=0;
 if(index<32){
  const float *input=((const float*)a.inputs[0])+row*128;
  float values[4],maximum=-1.f/0.f;
  #pragma unroll
  for(unsigned j=0;j<4;++j){values[j]=input[lane+j*32];maximum=fmaxf(maximum,values[j]);}
  for(unsigned offset=16;offset;offset>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffU,maximum,offset));
  maximum=__shfl_sync(0xffffffffU,maximum,0);
  if(maximum==0.f){if(!lane){maximum=-1.f/0.f;for(unsigned j=0;j<128;++j)maximum=fmaxf(maximum,input[j]);}maximum=__shfl_sync(0xffffffffU,maximum,0);}
  float total=0.f;
  #pragma unroll
  for(unsigned j=0;j<4;++j){values[j]=expf(values[j]-maximum);total+=values[j];}
  for(unsigned offset=16;offset;offset>>=1)total+=__shfl_down_sync(0xffffffffU,total,offset);
  total=__shfl_sync(0xffffffffU,total,0);
  #pragma unroll
  for(unsigned j=0;j<4;++j){float probability=values[j]/total;probabilities[lane+j*32]=probability;}
 }
 __syncthreads();
 if(index<128){float value=probabilities[index];if((__float_as_uint(value)&0x7fffffffU)>0x7f800000U)atomicExch(&invalid,1U);orders[index]=et_top_k_order(value);indices[index]=index;}
 __syncthreads();
 if(invalid){if(!index)et_error(a,5);return;}
 for(unsigned size=2;size<=128;size<<=1)for(unsigned stride=size>>1;stride;stride>>=1){unsigned other=index^stride;if(index<128&&index<other)et_top_k_pair(orders,indices,index,other,(index&size)==0);__syncthreads();}
 if(index<32){
  float selected=index<8?probabilities[indices[index]]:0.f;
  // Original Sum initializes+0 and adds each lane's value before shuffle tree.
  float total=0.f;if(index<8)total+=selected;
  for(unsigned offset=16;offset;offset>>=1)total+=__shfl_down_sync(0xffffffffU,total,offset);
  total=__shfl_sync(0xffffffffU,total,0);
  if(index<8){unsigned expert=indices[index];float normalized=selected/total;float scale=et_load<float>(a.inputs[1],a.input_dtypes[1],expert);((float*)a.output)[row*8+index]=normalized*scale;((unsigned*)a.output)[a.integers[0]*8+row*8+index]=expert;}
 }
}

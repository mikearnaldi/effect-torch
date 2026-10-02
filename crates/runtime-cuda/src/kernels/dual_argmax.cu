// Same1024logical lanes and two-level shuffle reduction as both production kernels.
// Inputs are materialized processed logits and external uniforms; no RNG changes.
__device__ float et_dual_arg_noisy_value(float value,float uniform){float x=logf(uniform);x=-x;x=logf(x);x=-x;return value+x;}
__device__ void et_dual_arg_update(float value,unsigned index,float&best,unsigned&best_index){
 if(!isnan(value)&&(value>best||(value==best&&index<best_index))){best=value;best_index=index;}
}
extern "C" __global__ __launch_bounds__(1024) void et_dual_argmax(CudaKernelArgs a){
 __shared__ float values[2][32];__shared__ unsigned indexes[2][32];
 unsigned lane=threadIdx.x&31,warp=threadIdx.x/32;et_u64 width=a.integers[1];
 for(et_u64 row=blockIdx.x;row<a.integers[0];row+=gridDim.x){
  float best[2]={-1.0f / 0.0f,-1.0f / 0.0f};unsigned best_index[2]={0xffffffffU,0xffffffffU};
  for(et_u64 column=threadIdx.x;column<width;column+=1024){
   et_u64 i=row*width+column;float value=et_load<float>(a.inputs[0],1,i),uniform=et_load<float>(a.inputs[1],1,i);
   et_dual_arg_update(value,column,best[0],best_index[0]);et_dual_arg_update(et_dual_arg_noisy_value(value,uniform),column,best[1],best_index[1]);
  }
  for(unsigned offset=16;offset;offset>>=1){
   #pragma unroll
   for(unsigned mode=0;mode<2;++mode){float other=__shfl_down_sync(0xffffffffU,best[mode],offset);unsigned index=__shfl_down_sync(0xffffffffU,best_index[mode],offset);et_dual_arg_update(other,index,best[mode],best_index[mode]);}
  }
  if(!lane){for(unsigned mode=0;mode<2;++mode){values[mode][warp]=best[mode];indexes[mode][warp]=best_index[mode];}}
  __syncthreads();
  if(!warp){
   for(unsigned mode=0;mode<2;++mode){best[mode]=values[mode][lane];best_index[mode]=indexes[mode][lane];}
   for(unsigned offset=16;offset;offset>>=1){
    #pragma unroll
    for(unsigned mode=0;mode<2;++mode){float other=__shfl_down_sync(0xffffffffU,best[mode],offset);unsigned index=__shfl_down_sync(0xffffffffU,best_index[mode],offset);et_dual_arg_update(other,index,best[mode],best_index[mode]);}
   }
   if(!lane){float first=et_load<float>(a.inputs[0],1,row*width),uniform=et_load<float>(a.inputs[1],1,row*width);if(isnan(first))best_index[0]=0;if(isnan(et_dual_arg_noisy_value(first,uniform)))best_index[1]=0;et_store(a.output,4,row,(et_i64)best_index[0]);et_store(a.output,4,a.integers[0]+row,(et_i64)best_index[1]);}
  }
  __syncthreads();
 }
}

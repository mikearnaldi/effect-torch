// Proven sampler83 component body; same F32 pointwise module as RNG80.
#ifdef ET_POINTWISE
extern "C" __global__ __launch_bounds__(1024) void et_random_sampler83(CudaKernelArgs a){
 const float* x=(const float*)a.inputs[0];
 const float* temperature=(const float*)a.inputs[1];
 unsigned short* feedback=(unsigned short*)a.output;
 et_i64* arg=(et_i64*)a.scratch[0];
 float* entropy=(float*)a.scratch[1];
 int rows=(int)a.integers[2],width=(int)a.integers[1],per_row=(int)a.integers[3];
 et_u64 seed=a.integers[0];
 unsigned row=blockIdx.x,lane=threadIdx.x&31,warp=threadIdx.x/32;
 float temp=temperature[per_row?row:0];
 float best[2]={(-1.0f/0.0f),(-1.0f/0.0f)},maximum=(-1.0f/0.0f);
 unsigned indexes[2]={0xffffffffU,0xffffffffU};
 __shared__ float values[2][32],maxima[32],sums[32],moments[32],shared_max;
 __shared__ unsigned owners[2][32];
 for(et_u64 col=threadIdx.x;col<(et_u64)width;col+=1024){
  et_u64 i=(et_u64)row*width+col;
  float value=x[i]/temp;
  et_store((et_u64)feedback,3,i,value);
  maximum=fmaxf(maximum,value);
  et_rng80_update(value,col,best[0],indexes[0]);
  et_rng80_update(et_rng80_noisy_value(value,et_rng80_uniform(seed,i)),col,best[1],indexes[1]);
 }
 for(unsigned offset=16;offset;offset>>=1){
  maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffU,maximum,offset));
  #pragma unroll
  for(unsigned mode=0;mode<2;++mode){
   float v=__shfl_down_sync(0xffffffffU,best[mode],offset);
   unsigned ix=__shfl_down_sync(0xffffffffU,indexes[mode],offset);
   et_rng80_update(v,ix,best[mode],indexes[mode]);
  }
 }
 if(!lane){maxima[warp]=maximum;for(unsigned mode=0;mode<2;++mode){values[mode][warp]=best[mode];owners[mode][warp]=indexes[mode];}}
 __syncthreads();
 if(!warp){
  maximum=maxima[lane];
  for(unsigned mode=0;mode<2;++mode){best[mode]=values[mode][lane];indexes[mode]=owners[mode][lane];}
  for(unsigned offset=16;offset;offset>>=1){
   maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffU,maximum,offset));
   #pragma unroll
   for(unsigned mode=0;mode<2;++mode){
    float v=__shfl_down_sync(0xffffffffU,best[mode],offset);
    unsigned ix=__shfl_down_sync(0xffffffffU,indexes[mode],offset);
    et_rng80_update(v,ix,best[mode],indexes[mode]);
   }
  }
  if(!lane){
   float first=x[(et_u64)row*width]/temp;
   if(isnan(first))indexes[0]=0;
   if(isnan(et_rng80_noisy_value(first,et_rng80_uniform(seed,(et_u64)row*width))))indexes[1]=0;
   arg[row]=indexes[0];arg[rows+row]=indexes[1];shared_max=maximum;
  }
 }
 __syncthreads();
 maximum=shared_max;float sum=0,moment=0;
 for(et_u64 col=threadIdx.x;col<(et_u64)width;col+=1024){
  float value=x[(et_u64)row*width+col]/temp;
  float shifted=value-maximum,e=expf(shifted);sum+=e;moment+=e==0?0:e*shifted;
 }
 for(unsigned offset=16;offset;offset>>=1){sum+=__shfl_down_sync(0xffffffffU,sum,offset);moment+=__shfl_down_sync(0xffffffffU,moment,offset);}
 if(!lane){sums[warp]=sum;moments[warp]=moment;}
 __syncthreads();
 if(!warp){
  sum=sums[lane];moment=moments[lane];
  for(unsigned offset=16;offset;offset>>=1){sum+=__shfl_down_sync(0xffffffffU,sum,offset);moment+=__shfl_down_sync(0xffffffffU,moment,offset);}
  if(!lane)entropy[row]=logf(sum)-moment/sum;
 }
}
#endif

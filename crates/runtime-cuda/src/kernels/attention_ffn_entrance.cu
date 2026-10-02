// Exact four-output owner layout: [R,D,E,S], each rows*2816 BF16.

struct EtAttentionFfnEntrance {const unsigned short *a,*h,*wa,*wd,*we,*wr; const float *rho; unsigned short *r,*d,*e,*s; unsigned rows;};
__device__ float et_entrance_round(float x){return et_bfloat_float(et_to16(x,true));}
// Fixed512threads; no alternate geometry or tuning policy.
extern "C" __global__ void et_attention_ffn_entrance_bf16(CudaKernelArgs a){
#if defined(ET_RELAXED_NORM76)
 et_attention_ffn_entrance_relaxed76(a);
#else
 unsigned n = (unsigned)a.integers[0] * 2816;
 auto out = (unsigned short*)a.output;
 EtAttentionFfnEntrance q{(const unsigned short*)a.inputs[0], (const unsigned short*)a.inputs[1],
  (const unsigned short*)a.inputs[2], (const unsigned short*)a.inputs[3],
  (const unsigned short*)a.inputs[4], (const unsigned short*)a.inputs[5], (const float*)a.inputs[6],
  out, out+n, out+2*n, out+3*n, (unsigned)a.integers[0]};
 constexpr unsigned W=2816;unsigned lane=threadIdx.x&31,row=blockIdx.x;
 __shared__ float residual[W],inverse;
 if(threadIdx.x<32){
  float p[4]={0,0,0,0};
  #pragma unroll
  for(unsigned step=0;step<22;++step){
   unsigned k=lane*4+step*128;
   et_u64 packed,addr=(et_u64)(q.a+row*W+k);
   bool aligned = (addr & 7) == 0;
   if (aligned) asm volatile("ld.global.u64 %0, [%1];":"=l"(packed):"l"(addr));
   #pragma unroll
   for(unsigned j=0;j<4;++j){float x=aligned ? et_bfloat_float((unsigned short)(packed>>(j*16))) : et_bfloat_float(q.a[row*W+k+j]);float square=x*x;p[j]+=square;}
  }
  float sum=((p[0]+p[1])+p[2])+p[3];
  for(unsigned off=16;off;off>>=1)sum+=__shfl_down_sync(0xffffffffU,sum,off);
  if(!lane)inverse=rsqrtf(sum*(1.0f/float(W))+float(1e-6));
 }
 __syncthreads();
 for(unsigned k=threadIdx.x;k<W;k+=blockDim.x){
  float x=et_bfloat_float(q.a[row*W+k])*inverse;
  x*=et_bfloat_float(q.wa[k]);
  float r=et_entrance_round(et_bfloat_float(q.h[row*W+k])+et_entrance_round(x));
  residual[k]=r;q.r[row*W+k]=et_to16(r,true);
 }
 __syncthreads();
 if(threadIdx.x<32){
  float p[4]={0,0,0,0};
  #pragma unroll
  for(unsigned step=0;step<22;++step){unsigned k=lane*4+step*128;
   #pragma unroll
   for(unsigned j=0;j<4;++j){float x=residual[k+j];float square=x*x;p[j]+=square;}
  }
  float sum=((p[0]+p[1])+p[2])+p[3];
  for(unsigned off=16;off;off>>=1)sum+=__shfl_down_sync(0xffffffffU,sum,off);
  if(!lane)inverse=rsqrtf(sum*(1.0f/float(W))+float(1e-6));
 }
 __syncthreads();
 for(unsigned k=threadIdx.x;k<W;k+=blockDim.x){unsigned i=row*W+k;float v=residual[k]*inverse;
  q.d[i]=et_to16(v*et_bfloat_float(q.wd[k]),true);
  q.e[i]=et_to16(v*et_bfloat_float(q.we[k]),true);
  float learned=et_entrance_round(et_entrance_round(v)*et_bfloat_float(q.wr[k]));q.s[i]=et_to16(learned*q.rho[0],true);
 }
#endif
}

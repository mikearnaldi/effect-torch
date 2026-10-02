// Diagnostic exact cuBLAS GEMV lane decomposition for the two expert shapes.
#include "../../../crates/runtime-cuda/src/kernels/expert_splitk.cu"
extern "C" __global__ void et_expert_gemv_compute(const EtExpertSplitKDescriptor* descriptors,unsigned count,unsigned columns,unsigned inner) {
 const auto d=descriptors[et_expert_splitk_find(descriptors,count,blockIdx.x,false)];
 const auto* x=reinterpret_cast<const __nv_bfloat16*>(d.x);
 const auto* w=reinterpret_cast<const __nv_bfloat16*>(d.weight);
 unsigned lane=threadIdx.x,lanes=blockDim.x,col=(blockIdx.x-d.compute_prefix)*4+threadIdx.y;
 unsigned grain=inner==2816?1:22;
 float value=0.0f;
 for(unsigned base=lane*grain;base<inner;base+=lanes*grain)
  for(unsigned j=0;j<grain&&base+j<inner;++j)
   value=fmaf(__bfloat162float(x[base+j]),__bfloat162float(w[col*inner+base+j]),value);
 __shared__ float partial[128];unsigned i=threadIdx.y*lanes+lane;partial[i]=value;__syncthreads();
 for(unsigned stride=lanes/2;stride;stride>>=1){if(lane<stride)partial[i]+=partial[i+stride];__syncthreads();}
 if(lane==0&&col<columns)reinterpret_cast<__nv_bfloat16*>(d.out)[col]=__float2bfloat16_rn(partial[i]);
}

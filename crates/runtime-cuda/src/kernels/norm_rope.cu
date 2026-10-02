// Private BF16 RMS and HalfSplit RoPE; precomputed tables retain their own
// rounding boundaries and ownership. Kernel-local casts reproduce materialization.
#if defined(ET_TENSOR) && defined(ET_COMPUTE_F32)
__device__ float et_norm_rope_round(float x) { return et_bfloat_float(et_to16(x, true)); }
__device__ et_u64 et_norm_rope_mod(unsigned index, et_u64 modulus, bool mask) {
    if (!modulus) return index;
    if (mask && !(modulus & (modulus - 1))) return et_u64(index) & (modulus - 1);
    return et_u64(index) % modulus;
}
__device__ et_u64 et_norm_rope_table_index(const CudaKernelArgs &a, unsigned slot,
    unsigned position, unsigned column) {
    if (!(a.operation & (1U << slot))) return (et_u64)position * a.integers[0] + column;
    const et_u64 *layout = et_tail(a) + slot * 5;
    et_u64 row = et_norm_rope_mod(position, layout[3], a.operation & 4);
    et_u64 col = et_norm_rope_mod(column, layout[4], a.operation & 4);
    return layout[0] + row * layout[1] + col * layout[2];
}
extern "C" __global__ void et_norm_rope_bf16(CudaKernelArgs a) {
    const auto* cosine = (const unsigned short*)a.inputs[2];
    const auto* sine = (const unsigned short*)a.inputs[3];
    unsigned sequence = a.integers[11];
    unsigned width=a.integers[0], half=width/2, rows=a.elements/width;
    unsigned lane=threadIdx.x&31;
    for(unsigned row=(blockIdx.x*blockDim.x+threadIdx.x)/32;row<rows;row+=gridDim.x*(blockDim.x/32)) {
        et_u64 source_row=et_rms_source_row(a,row);
        float partial[4]={0,0,0,0};
        for(unsigned k=lane*4;k<width;k+=128) {
            #pragma unroll
            for(unsigned j=0;j<4;++j) {
                float x=et_load<float>(a.inputs[0],3,source_row*width+k+j);
                partial[j]+=x*x;
            }
        }
        float sum=((partial[0]+partial[1])+partial[2])+partial[3];
        for(unsigned offset=16;offset;offset>>=1)sum+=__shfl_down_sync(0xffffffffU,sum,offset);
        sum=__shfl_sync(0xffffffffU,sum,0);
        float mean=sum*(1.0f/float(width));
        float inverse=rsqrtf(mean+float(a.scalars[0]));
        unsigned position=row%sequence;
        for(unsigned k=lane;k<half;k+=32) {
            float x0=et_load<float>(a.inputs[0],3,source_row*width+k)*inverse;
            float x1=et_load<float>(a.inputs[0],3,source_row*width+k+half)*inverse;
            if(a.inputs[1]) {
                x0*=et_load<float>(a.inputs[1],3,k);
                x1*=et_load<float>(a.inputs[1],3,k+half);
            }
            // Materialization-equivalent canonical BF16 narrowing, including sign/payload rules.
            unsigned short n0=et_to16(x0,true),n1=et_to16(x1,true);
            float direct0=et_norm_rope_round(et_bfloat_float(n0)*et_bfloat_float(cosine[et_norm_rope_table_index(a, 0, position, k)]));
            float cross0=et_norm_rope_round(et_bfloat_float(n1^0x8000)*et_bfloat_float(sine[et_norm_rope_table_index(a, 1, position, k)]));
            float direct1=et_norm_rope_round(et_bfloat_float(n1)*et_bfloat_float(cosine[et_norm_rope_table_index(a, 0, position, k+half)]));
            float cross1=et_norm_rope_round(et_bfloat_float(n0)*et_bfloat_float(sine[et_norm_rope_table_index(a, 1, position, k+half)]));
            ((unsigned short*)a.output)[row*width+k]=et_to16(direct0+cross0,true);
            ((unsigned short*)a.output)[row*width+k+half]=et_to16(direct1+cross1,true);
        }
    }
}

#endif

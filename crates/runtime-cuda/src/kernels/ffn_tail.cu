// Private BF16 FFN tail. All three reductions keep the production width2816
// four-chain/32-lane tree. No semantic BF16 materialization is elided numerically.
__device__ float et_ffn_tail_round(float value) {
    return et_bfloat_float(et_to16(value, true));
}
__device__ float et_ffn_tail_inverse(const unsigned short *source, unsigned row) {
    unsigned lane = threadIdx.x & 31;
    float partial[4] = {0, 0, 0, 0};
    #pragma unroll
    for (unsigned step = 0; step < 22; ++step) {
        unsigned k = lane * 4 + step * 128;
        et_u64 packed, address = (et_u64)(source + (et_u64)row * 2816 + k);
        if (address % 8 == 0) {
            asm volatile("ld.global.u64 %0, [%1];" : "=l"(packed) : "l"(address));
        } else {
            // Offset dense views need not retain the allocator's base alignment.
            packed = 0;
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j)
                packed |= (et_u64)source[(et_u64)row * 2816 + k + j] << (16 * j);
        }
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            float x = et_bfloat_float((unsigned short)(packed >> (16 * j)));
            float square = x * x;
            partial[j] += square;
        }
    }
    float sum = ((partial[0] + partial[1]) + partial[2]) + partial[3];
    for (unsigned offset = 16; offset; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    return rsqrtf(sum * (1.0f / 2816.0f) + 1e-6f);
}
extern "C" __global__ void et_ffn_tail_bf16(CudaKernelArgs a) {
#if defined(ET_RELAXED_NORM76)
    et_ffn_tail_relaxed76(a);
#else
    const auto *dense = (const unsigned short *)a.inputs[0];
    const auto *expert = (const unsigned short *)a.inputs[1];
    const auto *residual = (const unsigned short *)a.inputs[2];
    const auto *wd = (const unsigned short *)a.inputs[3];
    const auto *we = (const unsigned short *)a.inputs[4];
    const auto *wc = (const unsigned short *)a.inputs[5];
    float scale = et_load<float>(a.inputs[6], 3, 0);
    unsigned row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    __shared__ float inverse[3];
    __shared__ float summed[2816];
    if (warp < 2) {
        float value = et_ffn_tail_inverse(warp ? expert : dense, row);
        if (!lane) inverse[warp] = value;
    }
    __syncthreads();
    for (unsigned k = threadIdx.x; k < 2816; k += blockDim.x) {
        et_u64 i = (et_u64)row * 2816 + k;
        float d = et_bfloat_float(dense[i]) * inverse[0];
        d *= et_bfloat_float(wd[k]);
        d = et_ffn_tail_round(d);
        float e = et_bfloat_float(expert[i]) * inverse[1];
        e *= et_bfloat_float(we[k]);
        e = et_ffn_tail_round(e);
        summed[k] = et_ffn_tail_round(d + e);
    }
    __syncthreads();
    if (!warp) {
        float partial[4] = {0, 0, 0, 0};
        #pragma unroll
        for (unsigned step = 0; step < 22; ++step) {
            unsigned k = lane * 4 + step * 128;
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                float x = summed[k + j];
                float square = x * x;
                partial[j] += square;
            }
        }
        float sum = ((partial[0] + partial[1]) + partial[2]) + partial[3];
        for (unsigned offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffU, sum, offset);
        if (!lane) inverse[2] = rsqrtf(sum * (1.0f / 2816.0f) + 1e-6f);
    }
    __syncthreads();
    for (unsigned k = threadIdx.x; k < 2816; k += blockDim.x) {
        et_u64 i = (et_u64)row * 2816 + k;
        float combined = summed[k] * inverse[2];
        combined *= et_bfloat_float(wc[k]);
        combined = et_ffn_tail_round(combined);
        float added = et_ffn_tail_round(et_bfloat_float(residual[i]) + combined);
        ((unsigned short *)a.output)[i] = et_to16(added * scale, true);
    }
#endif
}

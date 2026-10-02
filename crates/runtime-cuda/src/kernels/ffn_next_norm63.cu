extern "C" __global__ void et_ffn_next_norm63_bf16(CudaKernelArgs a) {
    const auto *dense = (const unsigned short *)a.inputs[0];
    const auto *expert = (const unsigned short *)a.inputs[1];
    const auto *residual = (const unsigned short *)a.inputs[2];
    const auto *wd = (const unsigned short *)a.inputs[3];
    const auto *we = (const unsigned short *)a.inputs[4];
    const auto *wc = (const unsigned short *)a.inputs[5];
    float scale = et_load<float>(a.inputs[6], 3, 0);
    unsigned row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    __shared__ float inverse[4];
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
        unsigned short bits = et_to16(added * scale, true);
        ((unsigned short *)a.output)[i] = bits;
        summed[k] = et_bfloat_float(bits);
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
        if (!lane) inverse[3] = rsqrtf(sum * (1.0f / 2816.0f) + 1e-6f);
    }
    __syncthreads();
    for (unsigned k = threadIdx.x; k < 2816; k += blockDim.x) {
        float value = summed[k] * inverse[3];
        value *= et_load<float>(a.inputs[7], 3, k);
        et_store(a.output + a.integers[0] * 2816 * 2, 3, (et_u64)row * 2816 + k, value);
    }
}

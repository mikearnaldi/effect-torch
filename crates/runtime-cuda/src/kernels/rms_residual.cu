// Private RMS + residual. Dense inputs only; semantic views materialize first.
// Preserve production reduction order and the intermediate BF16 store numerically.
extern "C" __global__ void et_rms_residual_bf16(CudaKernelArgs a) {
    constexpr unsigned width = 2816, DType = 3, OutType = 3;
    unsigned row = blockIdx.x, lane = threadIdx.x & 31;
    __shared__ float inverse;
    auto source_row = row;
    if (threadIdx.x < 32) {
        float partial[4] = {0, 0, 0, 0};
        auto address = a.inputs[0] + source_row * width * (DType == 1 ? 4 : 2);
        bool aligned = address % (DType == 1 ? 16 : 8) == 0;
        #pragma unroll
        for (unsigned step = 0; step < 22; ++step) {
            unsigned k = lane * 4 + step * 128;
            float values[4];
            if (aligned) {
                if (DType == 1) {
                    auto source = address + k * 4;
                    asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                        : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]), "=f"(values[3]) : "l"(source));
                } else {
                    et_u64 packed, source = address + k * 2;
                    asm volatile("ld.global.u64 %0, [%1];" : "=l"(packed) : "l"(source));
                    #pragma unroll
                    for (unsigned j = 0; j < 4; ++j) {
                        unsigned short bits = packed >> (j * 16);
                        values[j] = DType == 3 ? et_bfloat_float(bits) : et_half_float(bits);
                    }
                }
            } else {
                #pragma unroll
                for (unsigned j = 0; j < 4; ++j)
                    values[j] = et_load<float>(a.inputs[0], DType, source_row * width + k + j);
            }
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                float square = values[j] * values[j];
                partial[j] += square;
            }
        }
        float sum = ((partial[0] + partial[1]) + partial[2]) + partial[3];
        for (unsigned offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffU, sum, offset);
        if (!lane) inverse = rsqrtf(sum * (1.0f / float(width)) + float(a.scalars[0]));
    }
    __syncthreads();
    for (unsigned k = threadIdx.x; k < width; k += blockDim.x) {
        float value = et_load<float>(a.inputs[0], DType, source_row * width + k) * inverse;
        if (a.inputs[1]) value *= et_load<float>(a.inputs[1], a.input_dtypes[1], k);
        unsigned short rounded;
        et_store((et_u64)&rounded, 3, 0, value);
        float residual = et_load<float>(a.inputs[2], 3, row * width + k);
        float result = et_add(residual, et_bfloat_float(rounded));
        et_store(a.output, 3, row * width + k, result);
    }
}

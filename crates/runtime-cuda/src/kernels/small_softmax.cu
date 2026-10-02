// Exact private width-128 F32 softmax. Each warp retains four lane-strided
// values, preserving the ordinary short Max and Sum reduction schedules.
#if defined(ET_TENSOR) && defined(ET_COMPUTE_F32)
extern "C" __global__ void et_small_softmax_f32(CudaKernelArgs a) {
    const float *input = reinterpret_cast<const float *>(a.inputs[0]);
    float *output = reinterpret_cast<float *>(a.output);
    unsigned int lane = threadIdx.x & 31U;
    et_u64 rows = a.elements / 128;
    for (et_u64 row = ((et_u64)blockIdx.x * blockDim.x + threadIdx.x) / 32;
         row < rows; row += (et_u64)gridDim.x * (blockDim.x / 32)) {
        float values[4], maximum = -1.0f / 0.0f;
        #pragma unroll
        for (unsigned int j = 0; j < 4; ++j) {
            values[j] = input[row * 128 + lane + j * 32];
            maximum = fmaxf(maximum, values[j]);
        }
        for (unsigned int offset = 16; offset; offset >>= 1)
            maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
        maximum = __shfl_sync(0xffffffffU, maximum, 0);
        // Match the ordinary Max's original-order signed-zero tie handling.
        if (maximum == 0.0f) {
            if (!lane) {
                maximum = -1.0f / 0.0f;
                for (unsigned int j = 0; j < 128; ++j)
                    maximum = fmaxf(maximum, input[row * 128 + j]);
            }
            maximum = __shfl_sync(0xffffffffU, maximum, 0);
        }
        float total = 0.0f;
        #pragma unroll
        for (unsigned int j = 0; j < 4; ++j) {
            values[j] = expf(values[j] - maximum);
            total += values[j];
        }
        for (unsigned int offset = 16; offset; offset >>= 1)
            total += __shfl_down_sync(0xffffffffU, total, offset);
        total = __shfl_sync(0xffffffffU, total, 0);
        #pragma unroll
        for (unsigned int j = 0; j < 4; ++j)
            output[row * 128 + lane + j * 32] = values[j] / total;
    }
}
#endif

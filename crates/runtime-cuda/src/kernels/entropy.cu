// Exact categorical entropy with private normalized/probability values recomputed.
#if defined(ET_TENSOR) && defined(ET_COMPUTE_F32)
template<int Stage> __device__ float et_entropy_value(const CudaKernelArgs &a, et_u64 row, et_u64 column) {
    float x = ((const float *)a.inputs[0])[row * a.integers[0] + column];
    if (Stage == 0) return x;
    float maximum = ((const float *)a.inputs[1])[row];
    if (Stage == 1) { float shifted = x - maximum; return expf(shifted); }
    float logarithm = logf(((const float *)a.inputs[2])[row]);
    float logsumexp = maximum + logarithm;
    float normalized = x - logsumexp;
    if (Stage == 2) return normalized;
    float shifted = normalized - ((const float *)a.inputs[3])[row];
    float exponential = expf(shifted);
    if (Stage == 3) return exponential;
    float clamped = fmaxf(normalized, -3.4028234663852886e38f);
    float probability = exponential / ((const float *)a.inputs[4])[row];
    return clamped * probability;
}
template<int Stage> __device__ void et_entropy_reduce(const CudaKernelArgs &a) {
    constexpr bool maximum = Stage == 0 || Stage == 2;
    et_u64 row = blockIdx.x, width = a.integers[0];
    unsigned int lane = threadIdx.x & 31U, warp = threadIdx.x / 32;
    __shared__ float partials[32];
    float value = maximum ? -1.0f / 0.0f : 0.0f;
    if (maximum) {
        for (et_u64 r = threadIdx.x; r < width; r += 1024)
            value = fmaxf(value, et_entropy_value<Stage>(a, row, r));
    } else {
        et_u64 complete = width - width % 4096;
        for (et_u64 r = threadIdx.x * 4; r < complete; r += 4096) {
            #pragma unroll
            for (unsigned int j = 0; j < 4; ++j)
                value += et_entropy_value<Stage>(a, row, r + j);
        }
        for (et_u64 r = complete + threadIdx.x; r < width; r += 1024)
            value += et_entropy_value<Stage>(a, row, r);
    }
    for (unsigned int offset = 16; offset; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffffU, value, offset);
        value = maximum ? fmaxf(value, other) : value + other;
    }
    if (!lane) partials[warp] = value;
    __syncthreads();
    if (!warp) {
        value = partials[lane];
        for (unsigned int offset = 16; offset; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffffU, value, offset);
            value = maximum ? fmaxf(value, other) : value + other;
        }
        if (!lane) ((float *)a.output)[row] = Stage == 4 ? -value : value;
    }
}
extern "C" __global__ __launch_bounds__(1024) void et_entropy_max(CudaKernelArgs a) { et_entropy_reduce<0>(a); }
extern "C" __global__ __launch_bounds__(1024) void et_entropy_sum(CudaKernelArgs a) { et_entropy_reduce<1>(a); }
extern "C" __global__ __launch_bounds__(1024) void et_entropy_normalized_max(CudaKernelArgs a) { et_entropy_reduce<2>(a); }
extern "C" __global__ __launch_bounds__(1024) void et_entropy_normalized_sum(CudaKernelArgs a) { et_entropy_reduce<3>(a); }
extern "C" __global__ __launch_bounds__(1024) void et_entropy_finish(CudaKernelArgs a) { et_entropy_reduce<4>(a); }
#endif

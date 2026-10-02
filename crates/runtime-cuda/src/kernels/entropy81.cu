// Frozen entropy81-screen identity: experimental F32 rounding, never default.
// Maximum is produced by the existing et_entropy_max kernel. All remaining
// values are invocation-owned; only the final entropy vector escapes.
#if defined(ET_TENSOR) && defined(ET_COMPUTE_F32)
extern "C" __global__ __launch_bounds__(1024) void et_entropy_relaxed81_moment_finish(CudaKernelArgs a) {
    unsigned row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    unsigned long long width = a.integers[0];
    float maximum = ((const float*)a.inputs[1])[row];
    float sum = 0, moment = 0;
    for (unsigned long long col = threadIdx.x; col < width; col += 1024) {
        float shifted = ((const float*)a.inputs[0])[row * width + col] - maximum;
        float e = expf(shifted);
        sum += e;
        // Underflow/-infinity contributes zero; NaN still propagates.
        moment += e == 0 ? 0 : e * shifted;
    }
    __shared__ float sums[32], moments[32];
    for (unsigned offset = 16; offset; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        moment += __shfl_down_sync(0xffffffff, moment, offset);
    }
    if (!lane) { sums[warp] = sum; moments[warp] = moment; }
    __syncthreads();
    if (!warp) {
        sum = sums[lane]; moment = moments[lane];
        for (unsigned offset = 16; offset; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
            moment += __shfl_down_sync(0xffffffff, moment, offset);
        }
        if (!lane) ((float*)a.output)[row] = logf(sum) - moment / sum;
    }
}
#endif

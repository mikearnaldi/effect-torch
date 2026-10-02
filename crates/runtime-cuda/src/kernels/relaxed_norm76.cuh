// Opt-in full-CTA RMS reductions. Fixed512-thread/width2816 ABI, unchanged
// BF16 intermediate materialization; only the F32 sum-of-squares tree changes.
#if defined(ET_RELAXED_NORM76)
__device__ float et_norm76_warp_sum(float value) {
    for (unsigned offset = 16; offset; offset >>= 1)
        value += __shfl_down_sync(0xffffffffU, value, offset);
    return value;
}
__device__ float et_norm76_inverse(float local, float *partials) {
    unsigned lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    local = et_norm76_warp_sum(local);
    if (!lane) partials[warp] = local;
    __syncthreads();
    if (!warp) {
        float total = et_norm76_warp_sum(lane < 16 ? partials[lane] : 0.0f);
        if (!lane) partials[0] = rsqrtf(total * (1.0f / 2816.0f) + 1e-6f);
    }
    __syncthreads();
    return partials[0];
}
__device__ float et_norm76_round(float x) {
    return et_bfloat_float(et_to16(x, true));
}
__device__ void et_ffn_tail_relaxed76(CudaKernelArgs a) {
    constexpr unsigned W = 2816, STEPS = 6;
    const auto *dense = (const unsigned short *)a.inputs[0];
    const auto *expert = (const unsigned short *)a.inputs[1];
    const auto *residual = (const unsigned short *)a.inputs[2];
    const auto *wd = (const unsigned short *)a.inputs[3];
    const auto *we = (const unsigned short *)a.inputs[4];
    const auto *wc = (const unsigned short *)a.inputs[5];
    unsigned row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    __shared__ float pd[16], pe[16], inverse[2];
    float dv[STEPS], ev[STEPS], combined[STEPS];
    float sd = 0.0f, se = 0.0f;
    #pragma unroll
    for (unsigned step = 0; step < STEPS; ++step) {
        unsigned k = threadIdx.x + step * 512;
        dv[step] = k < W ? et_bfloat_float(dense[(et_u64)row * W + k]) : 0.0f;
        ev[step] = k < W ? et_bfloat_float(expert[(et_u64)row * W + k]) : 0.0f;
        sd += dv[step] * dv[step];
        se += ev[step] * ev[step];
    }
    sd = et_norm76_warp_sum(sd); se = et_norm76_warp_sum(se);
    if (!lane) { pd[warp] = sd; pe[warp] = se; }
    __syncthreads();
    if (!warp) {
        sd = et_norm76_warp_sum(lane < 16 ? pd[lane] : 0.0f);
        se = et_norm76_warp_sum(lane < 16 ? pe[lane] : 0.0f);
        if (!lane) {
            inverse[0] = rsqrtf(sd * (1.0f / float(W)) + 1e-6f);
            inverse[1] = rsqrtf(se * (1.0f / float(W)) + 1e-6f);
        }
    }
    __syncthreads();
    float total = 0.0f;
    #pragma unroll
    for (unsigned step = 0; step < STEPS; ++step) {
        unsigned k = threadIdx.x + step * 512;
        combined[step] = 0.0f;
        if (k < W) {
            float d = dv[step] * inverse[0]; d *= et_bfloat_float(wd[k]);
            float e = ev[step] * inverse[1]; e *= et_bfloat_float(we[k]);
            combined[step] = et_norm76_round(et_norm76_round(d) + et_norm76_round(e));
            total += combined[step] * combined[step];
        }
    }
    float inv = et_norm76_inverse(total, pd);
    float scale = et_load<float>(a.inputs[6], 3, 0);
    #pragma unroll
    for (unsigned step = 0; step < STEPS; ++step) {
        unsigned k = threadIdx.x + step * 512;
        if (k < W) {
            float c = combined[step] * inv; c *= et_bfloat_float(wc[k]);
            float added = et_norm76_round(et_bfloat_float(residual[(et_u64)row * W + k]) + et_norm76_round(c));
            ((unsigned short *)a.output)[(et_u64)row * W + k] = et_to16(added * scale, true);
        }
    }
}
__device__ void et_attention_ffn_entrance_relaxed76(CudaKernelArgs a) {
    constexpr unsigned W = 2816, STEPS = 6;
    auto source = (const unsigned short *)a.inputs[0];
    auto hidden = (const unsigned short *)a.inputs[1];
    auto wa = (const unsigned short *)a.inputs[2];
    auto wd = (const unsigned short *)a.inputs[3];
    auto we = (const unsigned short *)a.inputs[4];
    auto wr = (const unsigned short *)a.inputs[5];
    float rho = ((const float *)a.inputs[6])[0];
    auto out = (unsigned short *)a.output;
    et_u64 n = a.integers[0] * W, row = blockIdx.x;
    __shared__ float partials[16];
    float values[STEPS], residuals[STEPS], total = 0.0f;
    #pragma unroll
    for (unsigned step = 0; step < STEPS; ++step) {
        unsigned k = threadIdx.x + step * 512;
        values[step] = k < W ? et_bfloat_float(source[row * W + k]) : 0.0f;
        total += values[step] * values[step];
    }
    float inverse = et_norm76_inverse(total, partials);
    total = 0.0f;
    #pragma unroll
    for (unsigned step = 0; step < STEPS; ++step) {
        unsigned k = threadIdx.x + step * 512;
        residuals[step] = 0.0f;
        if (k < W) {
            float x = values[step] * inverse; x *= et_bfloat_float(wa[k]);
            float r = et_norm76_round(et_bfloat_float(hidden[row * W + k]) + et_norm76_round(x));
            residuals[step] = r; out[row * W + k] = et_to16(r, true);
            total += r * r;
        }
    }
    // Every thread copied the previous inverse before this next shared write:
    // et_norm76_inverse's final barrier precedes its shared read. The explicit
    // barrier protects that read from a fast warp overwriting partials[0].
    __syncthreads();
    inverse = et_norm76_inverse(total, partials);
    #pragma unroll
    for (unsigned step = 0; step < STEPS; ++step) {
        unsigned k = threadIdx.x + step * 512;
        if (k < W) {
            et_u64 i = row * W + k;
            float v = residuals[step] * inverse;
            out[n + i] = et_to16(v * et_bfloat_float(wd[k]), true);
            out[2 * n + i] = et_to16(v * et_bfloat_float(we[k]), true);
            float learned = et_norm76_round(et_norm76_round(v) * et_bfloat_float(wr[k]));
            out[3 * n + i] = et_to16(learned * rho, true);
        }
    }
}
#endif

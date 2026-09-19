__device__ void et_cache_append(const CudaKernelArgs &a, et_u64 lane, et_u64 token, et_u64 heads, et_u64 tokens, et_u64 dim) {
    const unsigned int *cursors = (const unsigned int *)a.inputs[7];
    et_u64 cache_lane = lane / a.integers[2], capacity = a.integers[0], physical = (cursors[lane] + token) % capacity;
    unsigned int dtype = a.integers[1];
    for (et_u64 head = 0; head < heads; ++head) {
        et_u64 source = ((lane * heads + head) * tokens + token) * dim;
        et_u64 row = (cache_lane * capacity + physical) * heads + head, dest = row * dim;
        for (int role = 1; role <= 2; ++role) {
            float scale = 1;
            if (dtype == 6) {
                float maximum = 0;
                for (et_u64 d = 0; d < dim; ++d) maximum = fmaxf(maximum, fabsf(et_load<float>(a.inputs[role], a.input_dtypes[role], source + d)));
                scale = maximum == 0 ? 1.0f : maximum / 127.0f;
                ((float *)a.inputs[role + 4])[row] = scale;
            }
            for (et_u64 d = 0; d < dim; ++d) {
                float value = et_load<float>(a.inputs[role], a.input_dtypes[role], source + d);
                if (dtype == 6) {
                    int code = __float2int_rn(value / scale); code = code < -127 ? -127 : code > 127 ? 127 : code;
                    ((unsigned char *)a.inputs[role + 2])[dest + d] = (unsigned char)(code + 128);
                } else if (a.input_dtypes[role] == 0) {
                    et_store(a.inputs[role + 2], dtype, dest + d, et_load<double>(a.inputs[role], 0, source + d));
                } else et_store(a.inputs[role + 2], dtype, dest + d, value);
            }
        }
    }
}
__device__ float et_cache_load(const CudaKernelArgs &a, int role, et_u64 row, et_u64 d, et_u64 dim) {
    if (a.integers[1] == 6) return ((int)((const unsigned char *)a.inputs[role])[row * dim + d] - 128) * ((const float *)a.inputs[role + 2])[row];
    return et_load<float>(a.inputs[role], a.integers[1], row * dim + d);
}
template<class T> __device__ void et_kv_attention_impl(const CudaKernelArgs &a) {
    // One complete warp owns a sequence. Its lanes partition context positions;
    // lane zero alone appends, preserving row/token order through ring wraps.
    et_u64 sequence = et_thread() / 32, rows_per_sequence = a.integers[2], batch = a.integers[5];
    if (sequence >= batch) return;
    unsigned int warp_lane = threadIdx.x & 31U;
    unsigned int rank = et_meta(a)[1]; const et_u64 *qs = et_shape(a, 0), *ks = et_shape(a, 1);
    et_u64 dim = qs[rank - 1], tokens = qs[rank - 2], qheads = qs[rank - 3], kheads = ks[rank - 3];
    const unsigned int *valid = (const unsigned int *)a.scratch[0], *cursors = (const unsigned int *)a.inputs[7];
    et_u64 capacity = a.integers[0], window = a.integers[3]; bool bidirectional = a.integers[4] != 0;
    for (et_u64 lane = sequence * rows_per_sequence; lane < (sequence + 1) * rows_per_sequence; ++lane) {
        et_u64 count = valid[lane]; if (count > tokens) { et_error(a, 1); return; }
        if (bidirectional && warp_lane == 0) for (et_u64 t = 0; t < count; ++t) et_cache_append(a, lane, t, kheads, tokens, dim);
        __syncwarp();
        for (et_u64 t = 0; t < tokens; ++t) {
            if (t < count && !bidirectional && warp_lane == 0) et_cache_append(a, lane, t, kheads, tokens, dim);
            __syncwarp();
            for (et_u64 head = 0; head < qheads; ++head) for (et_u64 d = 0; d < dim; d += 4) {
                et_u64 query = ((lane * qheads + head) * tokens + t) * dim;
                if (t >= count) {
                    if (warp_lane == 0) for (unsigned int j = 0; j < 4 && d + j < dim; ++j) et_store(a.output, a.output_dtype, query + d + j, 0.0f);
                    continue;
                }
                et_u64 end = cursors[lane] + (bidirectional ? count : t + 1), start = end > capacity ? end - capacity : 0;
                if (window && end > window && end - window > start) start = end - window;
                T maximum = -T(1) / T(0), denominator = 0, result[4] = {0, 0, 0, 0};
                for (et_u64 pos = start + warp_lane; pos < end; pos += 32) {
                    et_u64 row = (sequence * capacity + pos % capacity) * kheads + head * kheads / qheads;
                    T score = 0;
                    for (et_u64 j = 0; j < dim; ++j) score += et_load<T>(a.inputs[0], a.input_dtypes[0], query + j) * T(et_cache_load(a, 3, row, j, dim));
                    score *= T(a.scalars[0]);
                    T next = fmax(maximum, score), previous = exp(maximum - next), current = exp(score - next);
                    denominator = denominator * previous + current;
                    for (unsigned int j = 0; j < 4; ++j) if (d + j < dim) result[j] = result[j] * previous + current * T(et_cache_load(a, 4, row, d + j, dim));
                    maximum = next;
                }
                // Merge stable online-softmax partials in the declared compute
                // dtype. The operation permits backend-defined reduction order.
                T global_maximum = maximum;
                for (unsigned int offset = 16; offset; offset >>= 1) global_maximum = fmax(global_maximum, __shfl_down_sync(0xffffffffU, global_maximum, offset));
                global_maximum = __shfl_sync(0xffffffffU, global_maximum, 0);
                T rescale = denominator == 0 ? T(0) : exp(maximum - global_maximum);
                denominator *= rescale;
                for (unsigned int j = 0; j < 4; ++j) result[j] *= rescale;
                for (unsigned int offset = 16; offset; offset >>= 1) {
                    denominator += __shfl_down_sync(0xffffffffU, denominator, offset);
                    for (unsigned int j = 0; j < 4; ++j) result[j] += __shfl_down_sync(0xffffffffU, result[j], offset);
                }
                if (warp_lane == 0) for (unsigned int j = 0; j < 4 && d + j < dim; ++j) et_store(a.output, a.output_dtype, query + d + j, result[j] / denominator);
            }
            // All readers must finish before lane zero can overwrite a slot.
            __syncwarp();
        }
    }
}

extern "C" __global__ void et_kv_attention(CudaKernelArgs a) {
    if (a.compute_dtype == 0) et_kv_attention_impl<double>(a);
    else et_kv_attention_impl<float>(a);
}

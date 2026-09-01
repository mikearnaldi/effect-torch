extern "C" __global__ void short_conv_f64(
    unsigned int op,
    const double *x,
    const double *weight,
    const double *g,
    double *out,
    unsigned int len,
    unsigned int outer,
    unsigned int time,
    unsigned int channels,
    unsigned int kernel,
    const double *initial_state,
    double *next_state,
    const unsigned int *valid,
    unsigned int stateful,
    unsigned int dtype
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    double total = 0.0;
    if (op == 0) {
        unsigned int c = index % channels;
        unsigned int t = (index / channels) % time;
        unsigned int batch = index / (channels * time);
        if (t < valid[batch]) {
            for (unsigned int j = 0; j < kernel; ++j) {
                long long source = (long long)t - kernel + 1 + j;
                if (source >= 0) {
                    total += x[(batch * time + source) * channels + c] * weight[c * kernel + j];
                } else if (stateful != 0) {
                    unsigned int history = (unsigned int)(source + kernel - 1);
                    total += initial_state[(batch * (kernel - 1) + history) * channels + c] * weight[c * kernel + j];
                }
            }
        }
        if (stateful != 0 && t == 0) {
            unsigned int advance = valid[batch];
            for (unsigned int j = 0; j + 1 < kernel; ++j) {
                long long source = (long long)advance - kernel + 1 + j;
                unsigned int history = (unsigned int)(source + kernel - 1);
                double value = source >= 0
                    ? x[(batch * time + source) * channels + c]
                    : initial_state[(batch * (kernel - 1) + history) * channels + c];
                next_state[(batch * (kernel - 1) + j) * channels + c] = value;
            }
        }
    } else if (op == 1) {
        unsigned int c = index % channels;
        unsigned int source = (index / channels) % time;
        unsigned int batch = index / (channels * time);
        for (unsigned int j = 0; j < kernel; ++j) {
            long long t = (long long)source + kernel - 1 - j;
            if (t >= 0 && t < time) total += g[(batch * time + t) * channels + c] * weight[c * kernel + j];
        }
    } else {
        unsigned int j = index % kernel;
        unsigned int c = index / kernel;
        for (unsigned int batch = 0; batch < outer; ++batch) for (unsigned int t = 0; t < time; ++t) {
            long long source = (long long)t - kernel + 1 + j;
            if (source >= 0) total += g[(batch * time + t) * channels + c] * x[(batch * time + source) * channels + c];
        }
    }
    out[index] = cast_dtype(total, dtype);
}

extern "C" __global__ void kda_forward_f64(
    const double *q, const double *k, const double *v, const double *decay, const double *beta,
    double *out, double *state, unsigned int outer, unsigned int time, unsigned int dk, unsigned int dv,
    double scale, const unsigned int *valid, unsigned int heads, unsigned int stateful, unsigned int dtype
) {
    unsigned int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= outer) return;
    double *s = state + (unsigned long long)batch * dk * dv;
    if (stateful == 0) for (unsigned int i = 0; i < dk * dv; ++i) s[i] = 0.0;
    for (unsigned int t = 0; t < time; ++t) {
        unsigned long long qbase = ((unsigned long long)batch * time + t) * dk;
        unsigned long long vbase = ((unsigned long long)batch * time + t) * dv;
        if (t >= valid[batch / heads]) {
            for (unsigned int j = 0; j < dv; ++j) out[vbase + j] = 0.0;
            continue;
        }
        for (unsigned int d = 0; d < dk; ++d) for (unsigned int j = 0; j < dv; ++j) s[d * dv + j] *= exp(decay[qbase + d]);
        for (unsigned int j = 0; j < dv; ++j) {
            double dot = 0.0;
            for (unsigned int d = 0; d < dk; ++d) dot += k[qbase + d] * s[d * dv + j];
            for (unsigned int d = 0; d < dk; ++d) s[d * dv + j] += beta[(unsigned long long)batch * time + t] * k[qbase + d] * (v[vbase + j] - dot);
            double result = 0.0;
            for (unsigned int d = 0; d < dk; ++d) result += s[d * dv + j] * q[qbase + d];
            out[vbase + j] = cast_dtype(scale * result, dtype);
        }
    }
}

extern "C" __global__ void kda_backward_f64(
    unsigned int output_kind, const double *q, const double *k, const double *v, const double *decay,
    const double *beta, const double *g, double *out, double *history, double *grad_state,
    unsigned int outer, unsigned int time, unsigned int dk, unsigned int dv, double scale, unsigned int dtype
) {
    unsigned int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= outer) return;
    unsigned long long state_size = (unsigned long long)dk * dv;
    double *states = history + (unsigned long long)batch * (time + 1) * state_size;
    double *gs = grad_state + (unsigned long long)batch * state_size * 2;
    double *next_gs = gs + state_size;
    for (unsigned long long i = 0; i < state_size; ++i) states[i] = 0.0;
    for (unsigned int t = 0; t < time; ++t) {
        double *previous = states + (unsigned long long)t * state_size;
        double *current = previous + state_size;
        unsigned long long qbase = ((unsigned long long)batch * time + t) * dk;
        unsigned long long vbase = ((unsigned long long)batch * time + t) * dv;
        for (unsigned int d = 0; d < dk; ++d) for (unsigned int j = 0; j < dv; ++j) current[d * dv + j] = previous[d * dv + j] * exp(decay[qbase + d]);
        for (unsigned int j = 0; j < dv; ++j) {
            double dot = 0.0; for (unsigned int d = 0; d < dk; ++d) dot += k[qbase + d] * current[d * dv + j];
            for (unsigned int d = 0; d < dk; ++d) current[d * dv + j] += beta[(unsigned long long)batch * time + t] * k[qbase + d] * (v[vbase + j] - dot);
        }
    }
    for (unsigned long long i = 0; i < state_size; ++i) gs[i] = 0.0;
    for (int ti = (int)time - 1; ti >= 0; --ti) {
        unsigned int t = (unsigned int)ti;
        double *previous = states + (unsigned long long)t * state_size;
        double *current = previous + state_size;
        unsigned long long qbase = ((unsigned long long)batch * time + t) * dk;
        unsigned long long vbase = ((unsigned long long)batch * time + t) * dv;
        for (unsigned int d = 0; d < dk; ++d) for (unsigned int j = 0; j < dv; ++j) gs[d * dv + j] += scale * q[qbase + d] * g[vbase + j];
        if (output_kind == 0) for (unsigned int d = 0; d < dk; ++d) { double total = 0.0; for (unsigned int j = 0; j < dv; ++j) total += current[d * dv + j] * g[vbase + j]; out[qbase + d] = cast_dtype(scale * total, dtype); }
        if (output_kind == 2) for (unsigned int j = 0; j < dv; ++j) { double total = 0.0; for (unsigned int d = 0; d < dk; ++d) total += gs[d * dv + j] * k[qbase + d]; out[vbase + j] = cast_dtype(beta[(unsigned long long)batch * time + t] * total, dtype); }
        double beta_grad = 0.0;
        for (unsigned int d = 0; d < dk; ++d) {
            double k_grad = 0.0; double decay_grad = 0.0;
            for (unsigned int j = 0; j < dv; ++j) {
                double decayed = previous[d * dv + j] * exp(decay[qbase + d]);
                double dot = 0.0; double kg = 0.0;
                for (unsigned int i = 0; i < dk; ++i) { dot += k[qbase + i] * previous[i * dv + j] * exp(decay[qbase + i]); kg += k[qbase + i] * gs[i * dv + j]; }
                beta_grad += gs[d * dv + j] * k[qbase + d] * (v[vbase + j] - dot);
                k_grad += beta[(unsigned long long)batch * time + t] * (gs[d * dv + j] * (v[vbase + j] - dot) - decayed * kg);
                double ddecayed = gs[d * dv + j] - beta[(unsigned long long)batch * time + t] * k[qbase + d] * kg;
                decay_grad += ddecayed * decayed;
            }
            if (output_kind == 1) out[qbase + d] = cast_dtype(k_grad, dtype);
            if (output_kind == 3) out[qbase + d] = cast_dtype(decay_grad, dtype);
        }
        if (output_kind == 4) out[(unsigned long long)batch * time + t] = cast_dtype(beta_grad, dtype);
        for (unsigned int d = 0; d < dk; ++d) for (unsigned int j = 0; j < dv; ++j) {
            double kg = 0.0; for (unsigned int i = 0; i < dk; ++i) kg += k[qbase + i] * gs[i * dv + j];
            next_gs[d * dv + j] = (gs[d * dv + j] - beta[(unsigned long long)batch * time + t] * k[qbase + d] * kg) * exp(decay[qbase + d]);
        }
        for (unsigned long long i = 0; i < state_size; ++i) gs[i] = next_gs[i];
    }
}

extern "C" __global__ void last_token_f64(
    const double *input,
    double *output,
    unsigned int width,
    unsigned int tokens,
    unsigned int valid,
    unsigned int dtype
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= width) return;
    output[index] = valid == 0 || valid > tokens
        ? 0.0
        : cast_dtype(input[(unsigned long long)(valid - 1) * width + index], dtype);
}

extern "C" __global__ void kv_append_f64(
    const double *k,
    const double *v,
    double *key_cache,
    double *value_cache,
    const unsigned int *cursors,
    const unsigned int *valid,
    unsigned int batch,
    unsigned int heads,
    unsigned int tokens,
    unsigned int dim,
    unsigned int capacity,
    unsigned int cache_dtype
) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long len = (unsigned long long)batch * heads * tokens * dim;
    if (index >= len) return;
    unsigned int d = index % dim;
    unsigned long long rest = index / dim;
    unsigned int token = rest % tokens;
    rest /= tokens;
    unsigned int head = rest % heads;
    unsigned int lane = rest / heads;
    if (token >= valid[lane]) return;
    unsigned int position = (cursors[lane] + token) % capacity;
    unsigned long long cache = (((unsigned long long)lane * capacity + position) * heads + head) * dim + d;
    if (cache_dtype == 6) {
        unsigned long long row = index - d;
        double key_max = 0.0;
        double value_max = 0.0;
        for (unsigned int i = 0; i < dim; ++i) {
            key_max = fmax(key_max, fabs(k[row + i]));
            value_max = fmax(value_max, fabs(v[row + i]));
        }
        double key_scale = key_max / 127.0 + 1e-12;
        double value_scale = value_max / 127.0 + 1e-12;
        double key_quantized = fmin(255.0, fmax(0.0, nearbyint(k[index] / key_scale) + 128.0));
        double value_quantized = fmin(255.0, fmax(0.0, nearbyint(v[index] / value_scale) + 128.0));
        key_cache[cache] = (key_quantized - 128.0) * key_scale;
        value_cache[cache] = (value_quantized - 128.0) * value_scale;
    } else {
        key_cache[cache] = cast_dtype(k[index], cache_dtype);
        value_cache[cache] = cast_dtype(v[index], cache_dtype);
    }
}

extern "C" __global__ void kv_append_packed_f64(
    const double *k,
    const double *v,
    double *key_cache,
    double *value_cache,
    const unsigned int *cursors,
    const unsigned int *valid,
    unsigned int batch,
    unsigned int heads,
    unsigned int tokens,
    unsigned int dim,
    unsigned int capacity,
    unsigned int rows_per_sequence,
    unsigned int cache_dtype
) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long len = (unsigned long long)batch * heads * tokens * dim;
    if (index >= len) return;
    unsigned int d = index % dim;
    unsigned long long rest = index / dim;
    unsigned int token = rest % tokens;
    rest /= tokens;
    unsigned int head = rest % heads;
    unsigned int lane = rest / heads;
    if (token >= valid[lane]) return;
    unsigned int position = (cursors[lane] + token) % capacity;
    unsigned int cache_lane = lane / rows_per_sequence;
    unsigned long long cache = (((unsigned long long)cache_lane * capacity + position) * heads + head) * dim + d;
    if (cache_dtype == 6) {
        unsigned long long row = index - d;
        double key_max = 0.0;
        double value_max = 0.0;
        for (unsigned int i = 0; i < dim; ++i) {
            key_max = fmax(key_max, fabs(k[row + i]));
            value_max = fmax(value_max, fabs(v[row + i]));
        }
        double key_scale = key_max / 127.0 + 1e-12;
        double value_scale = value_max / 127.0 + 1e-12;
        double key_quantized = fmin(255.0, fmax(0.0, nearbyint(k[index] / key_scale) + 128.0));
        double value_quantized = fmin(255.0, fmax(0.0, nearbyint(v[index] / value_scale) + 128.0));
        key_cache[cache] = (key_quantized - 128.0) * key_scale;
        value_cache[cache] = (value_quantized - 128.0) * value_scale;
    } else {
        key_cache[cache] = cast_dtype(k[index], cache_dtype);
        value_cache[cache] = cast_dtype(v[index], cache_dtype);
    }
}

extern "C" __global__ void kv_attention_f64(
    const double *q,
    const double *key_cache,
    const double *value_cache,
    double *output,
    const unsigned int *cursors,
    const unsigned int *valid,
    unsigned int batch,
    unsigned int q_heads,
    unsigned int kv_heads,
    unsigned int tokens,
    unsigned int dim,
    unsigned int capacity,
    double scale,
    unsigned int window,
    unsigned int bidirectional,
    unsigned int dtype
) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long len = (unsigned long long)batch * q_heads * tokens * dim;
    if (index >= len) return;
    unsigned int d = index % dim;
    unsigned long long rest = index / dim;
    unsigned int token = rest % tokens;
    rest /= tokens;
    unsigned int q_head = rest % q_heads;
    unsigned int lane = rest / q_heads;
    if (token >= valid[lane]) { output[index] = 0.0; return; }
    unsigned int kv_head = q_head * kv_heads / q_heads;
    unsigned int end = cursors[lane] + (bidirectional ? valid[lane] - 1 : token);
    unsigned int start = window == 0 || end + 1 <= window ? 0 : end + 1 - window;
    unsigned int retained_start = end + 1 <= capacity ? 0 : end + 1 - capacity;
    start = max(start, retained_start);
    double maximum = -1.7976931348623157e308;
    for (unsigned int position = start; position <= end; ++position) {
        unsigned int physical = position % capacity;
        unsigned long long cache = (((unsigned long long)lane * capacity + physical) * kv_heads + kv_head) * dim;
        unsigned long long query = (((unsigned long long)lane * q_heads + q_head) * tokens + token) * dim;
        double score = 0.0;
        for (unsigned int i = 0; i < dim; ++i) score += q[query + i] * key_cache[cache + i];
        maximum = fmax(maximum, score * scale);
    }
    double denominator = 0.0;
    double result = 0.0;
    for (unsigned int position = start; position <= end; ++position) {
        unsigned int physical = position % capacity;
        unsigned long long cache = (((unsigned long long)lane * capacity + physical) * kv_heads + kv_head) * dim;
        unsigned long long query = (((unsigned long long)lane * q_heads + q_head) * tokens + token) * dim;
        double score = 0.0;
        for (unsigned int i = 0; i < dim; ++i) score += q[query + i] * key_cache[cache + i];
        double weight = exp(score * scale - maximum);
        denominator += weight;
        result += weight * value_cache[cache + d];
    }
    output[index] = cast_dtype(result / denominator, dtype);
}

extern "C" __global__ void kv_attention_packed_f64(
    const double *q,
    const double *key_cache,
    const double *value_cache,
    double *output,
    const unsigned int *cursors,
    const unsigned int *valid,
    unsigned int batch,
    unsigned int q_heads,
    unsigned int kv_heads,
    unsigned int tokens,
    unsigned int dim,
    unsigned int capacity,
    unsigned int rows_per_sequence,
    double scale,
    unsigned int window,
    unsigned int bidirectional,
    unsigned int dtype
) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long len = (unsigned long long)batch * q_heads * tokens * dim;
    if (index >= len) return;
    unsigned int d = index % dim;
    unsigned long long rest = index / dim;
    unsigned int token = rest % tokens;
    rest /= tokens;
    unsigned int q_head = rest % q_heads;
    unsigned int lane = rest / q_heads;
    if (token >= valid[lane]) { output[index] = 0.0; return; }
    unsigned int kv_head = q_head * kv_heads / q_heads;
    unsigned int cache_lane = lane / rows_per_sequence;
    unsigned int end = cursors[lane] + (bidirectional ? valid[lane] - 1 : token);
    unsigned int start = window == 0 || end + 1 <= window ? 0 : end + 1 - window;
    unsigned int retained_start = end + 1 <= capacity ? 0 : end + 1 - capacity;
    start = max(start, retained_start);
    double maximum = -1.7976931348623157e308;
    for (unsigned int position = start; position <= end; ++position) {
        unsigned int physical = position % capacity;
        unsigned long long cache = (((unsigned long long)cache_lane * capacity + physical) * kv_heads + kv_head) * dim;
        unsigned long long query = (((unsigned long long)lane * q_heads + q_head) * tokens + token) * dim;
        double score = 0.0;
        for (unsigned int i = 0; i < dim; ++i) score += q[query + i] * key_cache[cache + i];
        maximum = fmax(maximum, score * scale);
    }
    double denominator = 0.0;
    double result = 0.0;
    for (unsigned int position = start; position <= end; ++position) {
        unsigned int physical = position % capacity;
        unsigned long long cache = (((unsigned long long)cache_lane * capacity + physical) * kv_heads + kv_head) * dim;
        unsigned long long query = (((unsigned long long)lane * q_heads + q_head) * tokens + token) * dim;
        double score = 0.0;
        for (unsigned int i = 0; i < dim; ++i) score += q[query + i] * key_cache[cache + i];
        double weight = exp(score * scale - maximum);
        denominator += weight;
        result += weight * value_cache[cache + d];
    }
    output[index] = cast_dtype(result / denominator, dtype);
}

__device__ __forceinline__ double warp_sum_f64(double value) {
    for (unsigned int offset = 16; offset != 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return __shfl_sync(0xffffffffU, value, 0);
}

__device__ __forceinline__ float warp_sum_f32(float value) {
    for (unsigned int offset = 16; offset != 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return __shfl_sync(0xffffffffU, value, 0);
}


extern "C" __global__ void muse_gqa_append_attention_split_f64(
    const double *q,
    const double *k,
    const double *v,
    const double *q_weight,
    const double *k_weight,
    double *key_cache,
    double *value_cache,
    double *output,
    const unsigned int *cursors,
    unsigned int capacity,
    double scale,
    double q_eps,
    double k_eps,
    double rope_theta,
    unsigned int apply_rope,
    unsigned int window,
    unsigned int cache_dtype,
    unsigned int output_dtype
) {
    constexpr unsigned int DIM = 128;
    constexpr unsigned int QUERY_HEADS_PER_KV = 16;
    constexpr unsigned int PARTITIONS = 4;
    __shared__ float prepared_query[DIM];
    __shared__ float current_key[DIM];
    __shared__ float current_value[DIM];
    __shared__ float partition_result[PARTITIONS][DIM];
    __shared__ float partition_maximum[PARTITIONS];
    __shared__ float partition_denominator[PARTITIONS];

    unsigned int query_head = blockIdx.x;
    unsigned int kv_head = query_head / QUERY_HEADS_PER_KV;
    unsigned int warp = threadIdx.x / 32;
    unsigned int lane = threadIdx.x & 31U;
    unsigned int cursor = cursors[0];
    unsigned int physical = cursor % capacity;
    unsigned long long cache =
        ((unsigned long long)physical * 2 + kv_head) * DIM;
    unsigned long long query = (unsigned long long)query_head * DIM;

    if (warp == 0) {
        float query_values[4];
        float key_values[4];
        float query_sum = 0.0f;
        float key_sum = 0.0f;
#pragma unroll
        for (unsigned int i = 0; i < 4; ++i) {
            unsigned int d = lane + i * 32;
            query_values[i] = (float)q[query + d];
            key_values[i] = (float)k[kv_head * DIM + d];
            query_sum += query_values[i] * query_values[i];
            key_sum += key_values[i] * key_values[i];
        }
        float query_inverse_rms =
            1.0f / sqrtf(warp_sum_f32(query_sum) / DIM + (float)q_eps);
        float key_inverse_rms =
            1.0f / sqrtf(warp_sum_f32(key_sum) / DIM + (float)k_eps);
#pragma unroll
        for (unsigned int i = 0; i < 4; ++i) {
            unsigned int d = lane + i * 32;
            float query_normalized =
                query_values[i] * query_inverse_rms * (float)q_weight[d];
            float key_normalized =
                key_values[i] * key_inverse_rms * (float)k_weight[d];
            if (apply_rope != 0) {
                float query_paired =
                    __shfl_xor_sync(0xffffffffU, query_normalized, 1);
                float key_paired =
                    __shfl_xor_sync(0xffffffffU, key_normalized, 1);
                unsigned int pair = d / 2;
                float angle = cursor * powf((float)rope_theta, -2.0f * pair / DIM);
                float c = cosf(angle);
                float s = sinf(angle);
                query_normalized = (d & 1U) == 0
                    ? query_normalized * c - query_paired * s
                    : query_paired * s + query_normalized * c;
                key_normalized = (d & 1U) == 0
                    ? key_normalized * c - key_paired * s
                    : key_paired * s + key_normalized * c;
            }
            prepared_query[d] = query_normalized;
            current_key[d] = (float)cast_dtype(key_normalized, cache_dtype);
            current_value[d] =
                (float)cast_dtype((float)v[kv_head * DIM + d], cache_dtype);
        }
        if (query_head % QUERY_HEADS_PER_KV == 0) {
#pragma unroll
            for (unsigned int i = 0; i < 4; ++i) {
                unsigned int d = lane + i * 32;
                key_cache[cache + d] = cast_dtype(current_key[d], cache_dtype);
                value_cache[cache + d] = cast_dtype(current_value[d], cache_dtype);
            }
        }
    }
    __syncthreads();

    float query_values[4];
#pragma unroll
    for (unsigned int i = 0; i < 4; ++i) {
        query_values[i] = prepared_query[lane + i * 32];
    }
    unsigned int start = window == 0 || cursor + 1 <= window
        ? 0
        : cursor + 1 - window;
    unsigned int retained_start = cursor + 1 <= capacity
        ? 0
        : cursor + 1 - capacity;
    start = max(start, retained_start);
    unsigned int positions = cursor + 1 - start;
    unsigned int partition_size = (positions + PARTITIONS - 1) / PARTITIONS;
    unsigned int partition_start = start + warp * partition_size;
    unsigned int partition_end = min(cursor + 1, partition_start + partition_size);

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float maximum = -3.402823466e38f;
    float denominator = 0.0f;
    for (unsigned int position = partition_start; position < partition_end; ++position) {
        bool current = position == cursor;
        if (!current) {
            physical = position % capacity;
            cache = ((unsigned long long)physical * 2 + kv_head) * DIM;
        }
        float partial = 0.0f;
#pragma unroll
        for (unsigned int i = 0; i < 4; ++i) {
            unsigned int d = lane + i * 32;
            float key_value = current
                ? current_key[d]
                : (float)key_cache[cache + d];
            partial += query_values[i] * key_value;
        }
        float score = warp_sum_f32(partial) * (float)scale;
        float next_maximum = fmaxf(maximum, score);
        float previous_weight = maximum == -3.402823466e38f
            ? 0.0f
            : expf(maximum - next_maximum);
        float current_weight = expf(score - next_maximum);
        denominator = denominator * previous_weight + current_weight;
#pragma unroll
        for (unsigned int i = 0; i < 4; ++i) {
            unsigned int d = lane + i * 32;
            float value = current
                ? current_value[d]
                : (float)value_cache[cache + d];
            result[i] = result[i] * previous_weight + current_weight * value;
        }
        maximum = next_maximum;
    }

#pragma unroll
    for (unsigned int i = 0; i < 4; ++i) {
        partition_result[warp][lane + i * 32] = result[i];
    }
    if (lane == 0) {
        partition_maximum[warp] = maximum;
        partition_denominator[warp] = denominator;
    }
    __syncthreads();

    if (warp == 0) {
        float merged_maximum = partition_maximum[0];
#pragma unroll
        for (unsigned int partition = 1; partition < PARTITIONS; ++partition) {
            merged_maximum = fmaxf(merged_maximum, partition_maximum[partition]);
        }
        float merged_denominator = 0.0f;
        float merged[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (unsigned int partition = 0; partition < PARTITIONS; ++partition) {
            float partition_weight = partition_denominator[partition] == 0.0f
                ? 0.0f
                : expf(partition_maximum[partition] - merged_maximum);
            merged_denominator +=
                partition_denominator[partition] * partition_weight;
#pragma unroll
            for (unsigned int i = 0; i < 4; ++i) {
                unsigned int d = lane + i * 32;
                merged[i] += partition_result[partition][d] * partition_weight;
            }
        }
#pragma unroll
        for (unsigned int i = 0; i < 4; ++i) {
            unsigned int d = lane + i * 32;
            output[query + d] =
                cast_dtype((double)(merged[i] / merged_denominator), output_dtype);
        }
    }
}

extern "C" __global__ void muse_gqa_append_attention_grouped_f64(
    const double *q,
    const double *k,
    const double *v,
    const double *q_weight,
    const double *k_weight,
    double *key_cache,
    double *value_cache,
    float *partials,
    const unsigned int *cursors,
    unsigned int capacity,
    double scale,
    double q_eps,
    double k_eps,
    double rope_theta,
    unsigned int apply_rope,
    unsigned int window,
    unsigned int cache_dtype
) {
    constexpr unsigned int DIM = 128;
    constexpr unsigned int QUERY_HEADS = 32;
    constexpr unsigned int QUERY_HEADS_PER_KV = 16;
    constexpr unsigned int PARTITIONS = 128;
    constexpr unsigned int TILE = 8;
    constexpr unsigned int PARTIAL_STRIDE = DIM + 2;
    __shared__ float prepared_query[QUERY_HEADS_PER_KV][DIM];
    __shared__ float current_key[DIM];
    __shared__ float current_value[DIM];
    __shared__ float key_tile[TILE][DIM];
    __shared__ float value_tile[TILE][DIM];
    __shared__ float rope_cos[DIM];
    __shared__ float rope_sin[DIM];

    unsigned int thread = threadIdx.x;
    unsigned int warp = thread / 32;
    unsigned int lane = thread & 31U;
    unsigned int kv_head = blockIdx.x / PARTITIONS;
    unsigned int partition = blockIdx.x % PARTITIONS;
    unsigned int query_head = kv_head * QUERY_HEADS_PER_KV + warp;
    unsigned int cursor = cursors[0];
    unsigned int physical = cursor % capacity;
    unsigned long long current_cache =
        ((unsigned long long)physical * 2 + kv_head) * DIM;
    unsigned long long query = (unsigned long long)query_head * DIM;

    if (apply_rope != 0) {
        if (warp == 0) {
#pragma unroll
            for (unsigned int i = 0; i < 4; ++i) {
                unsigned int d = lane + i * 32;
                unsigned int pair = d / 2;
                float angle = cursor * powf((float)rope_theta, -2.0f * pair / DIM);
                sincosf(angle, &rope_sin[d], &rope_cos[d]);
            }
        }
        __syncthreads();
    }

    float query_values[4];
    float query_sum = 0.0f;
#pragma unroll
    for (unsigned int i = 0; i < 4; ++i) {
        unsigned int d = lane + i * 32;
        query_values[i] = (float)q[query + d];
        query_sum += query_values[i] * query_values[i];
    }
    float query_inverse_rms =
        1.0f / sqrtf(warp_sum_f32(query_sum) / DIM + (float)q_eps);
#pragma unroll
    for (unsigned int i = 0; i < 4; ++i) {
        unsigned int d = lane + i * 32;
        float normalized = query_values[i] * query_inverse_rms * (float)q_weight[d];
        if (apply_rope != 0) {
            float paired = __shfl_xor_sync(0xffffffffU, normalized, 1);
            float c = rope_cos[d];
            float s = rope_sin[d];
            normalized = (d & 1U) == 0
                ? normalized * c - paired * s
                : paired * s + normalized * c;
        }
        prepared_query[warp][d] = normalized;
    }

    if (warp == 0) {
        float key_values[4];
        float key_sum = 0.0f;
#pragma unroll
        for (unsigned int i = 0; i < 4; ++i) {
            unsigned int d = lane + i * 32;
            key_values[i] = (float)k[kv_head * DIM + d];
            key_sum += key_values[i] * key_values[i];
        }
        float key_inverse_rms =
            1.0f / sqrtf(warp_sum_f32(key_sum) / DIM + (float)k_eps);
#pragma unroll
        for (unsigned int i = 0; i < 4; ++i) {
            unsigned int d = lane + i * 32;
            float normalized = key_values[i] * key_inverse_rms * (float)k_weight[d];
            if (apply_rope != 0) {
                float paired = __shfl_xor_sync(0xffffffffU, normalized, 1);
                float c = rope_cos[d];
                float s = rope_sin[d];
                normalized = (d & 1U) == 0
                    ? normalized * c - paired * s
                    : paired * s + normalized * c;
            }
            current_key[d] = (float)cast_dtype(normalized, cache_dtype);
            current_value[d] =
                (float)cast_dtype((float)v[kv_head * DIM + d], cache_dtype);
            if (partition == 0) {
                key_cache[current_cache + d] = cast_dtype(current_key[d], cache_dtype);
                value_cache[current_cache + d] = cast_dtype(current_value[d], cache_dtype);
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (unsigned int i = 0; i < 4; ++i) {
        query_values[i] = prepared_query[warp][lane + i * 32];
    }
    unsigned int start = window == 0 || cursor + 1 <= window
        ? 0
        : cursor + 1 - window;
    unsigned int retained_start = cursor + 1 <= capacity
        ? 0
        : cursor + 1 - capacity;
    start = max(start, retained_start);
    unsigned int positions = cursor + 1 - start;
    unsigned int partition_size = (positions + PARTITIONS - 1) / PARTITIONS;
    unsigned int partition_start = start + partition * partition_size;
    unsigned int partition_end = min(cursor + 1, partition_start + partition_size);
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float maximum = -3.402823466e38f;
    float denominator = 0.0f;

    for (unsigned int tile_start = partition_start; tile_start < partition_end; tile_start += TILE) {
        for (unsigned int item = thread; item < TILE * DIM; item += blockDim.x) {
            unsigned int tile = item / DIM;
            unsigned int d = item % DIM;
            unsigned int position = tile_start + tile;
            if (position < partition_end) {
                if (position == cursor) {
                    key_tile[tile][d] = current_key[d];
                    value_tile[tile][d] = current_value[d];
                } else {
                    unsigned int cached_physical = position % capacity;
                    unsigned long long cache =
                        ((unsigned long long)cached_physical * 2 + kv_head) * DIM + d;
                    key_tile[tile][d] = (float)key_cache[cache];
                    value_tile[tile][d] = (float)value_cache[cache];
                }
            }
        }
        __syncthreads();
        unsigned int tile_end = min(TILE, partition_end - tile_start);
        for (unsigned int tile = 0; tile < tile_end; ++tile) {
            float partial = 0.0f;
#pragma unroll
            for (unsigned int i = 0; i < 4; ++i) {
                unsigned int d = lane + i * 32;
                partial += query_values[i] * key_tile[tile][d];
            }
            float score = warp_sum_f32(partial) * (float)scale;
            float next_maximum = fmaxf(maximum, score);
            float previous_weight = maximum == -3.402823466e38f
                ? 0.0f
                : expf(maximum - next_maximum);
            float current_weight = expf(score - next_maximum);
            denominator = denominator * previous_weight + current_weight;
#pragma unroll
            for (unsigned int i = 0; i < 4; ++i) {
                unsigned int d = lane + i * 32;
                result[i] = result[i] * previous_weight
                    + current_weight * value_tile[tile][d];
            }
            maximum = next_maximum;
        }
        __syncthreads();
    }

    float *partial_output = partials
        + ((partition * QUERY_HEADS + query_head) * PARTIAL_STRIDE);
#pragma unroll
    for (unsigned int i = 0; i < 4; ++i) {
        partial_output[lane + i * 32] = result[i];
    }
    if (lane == 0) {
        partial_output[DIM] = maximum;
        partial_output[DIM + 1] = denominator;
    }
}

extern "C" __global__ void muse_gqa_merge_grouped_f64(
    const float *partials,
    double *output,
    unsigned int output_dtype
) {
    constexpr unsigned int DIM = 128;
    constexpr unsigned int QUERY_HEADS = 32;
    constexpr unsigned int PARTITIONS = 128;
    constexpr unsigned int PARTIAL_STRIDE = DIM + 2;
    unsigned int warp = threadIdx.x / 32;
    unsigned int lane = threadIdx.x & 31U;
    unsigned int query_head = blockIdx.x * 4 + warp;
    if (query_head >= QUERY_HEADS) return;

    float merged_maximum = -3.402823466e38f;
#pragma unroll
    for (unsigned int partition = 0; partition < PARTITIONS; ++partition) {
        const float *partial = partials
            + ((partition * QUERY_HEADS + query_head) * PARTIAL_STRIDE);
        merged_maximum = fmaxf(merged_maximum, partial[DIM]);
    }
    float merged_denominator = 0.0f;
    float merged[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (unsigned int partition = 0; partition < PARTITIONS; ++partition) {
        const float *partial = partials
            + ((partition * QUERY_HEADS + query_head) * PARTIAL_STRIDE);
        float partition_weight = partial[DIM + 1] == 0.0f
            ? 0.0f
            : expf(partial[DIM] - merged_maximum);
        merged_denominator += partial[DIM + 1] * partition_weight;
#pragma unroll
        for (unsigned int i = 0; i < 4; ++i) {
            merged[i] += partial[lane + i * 32] * partition_weight;
        }
    }
#pragma unroll
    for (unsigned int i = 0; i < 4; ++i) {
        unsigned int d = lane + i * 32;
        output[(unsigned long long)query_head * DIM + d] =
            cast_dtype((double)(merged[i] / merged_denominator), output_dtype);
    }
}

__device__ void run_kv_attention_warp_f32(
    const double *q,
    const double *key_cache,
    const double *value_cache,
    double *output,
    const unsigned int *cursors,
    const unsigned int *valid,
    unsigned int batch,
    unsigned int q_heads,
    unsigned int kv_heads,
    unsigned int tokens,
    unsigned int dim,
    unsigned int capacity,
    unsigned int rows_per_sequence,
    double scale,
    unsigned int window,
    unsigned int bidirectional,
    unsigned int dtype
) {
    unsigned int thread_lane = threadIdx.x & 31U;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int warps_per_block = blockDim.x >> 5;
    unsigned long long query_index =
        (unsigned long long)blockIdx.x * warps_per_block + warp;
    unsigned long long query_count =
        (unsigned long long)batch * q_heads * tokens;
    if (query_index >= query_count) return;

    unsigned int token = query_index % tokens;
    unsigned long long rest = query_index / tokens;
    unsigned int q_head = rest % q_heads;
    unsigned int lane = rest / q_heads;
    unsigned long long query = query_index * dim;
    if (token >= valid[lane]) {
        for (unsigned int d = thread_lane; d < dim; d += 32) {
            output[query + d] = 0.0;
        }
        return;
    }

    unsigned int kv_head = q_head * kv_heads / q_heads;
    unsigned int cache_lane = lane / rows_per_sequence;
    unsigned int end = cursors[lane] + (bidirectional ? valid[lane] - 1 : token);
    unsigned int start = window == 0 || end + 1 <= window ? 0 : end + 1 - window;
    unsigned int retained_start = end + 1 <= capacity ? 0 : end + 1 - capacity;
    start = max(start, retained_start);

    float result[8];
#pragma unroll
    for (unsigned int i = 0; i < 8; ++i) result[i] = 0.0f;
    float maximum = -3.402823466e38f;
    float denominator = 0.0f;
    float scale_f32 = (float)scale;
    for (unsigned int position = start; position <= end; ++position) {
        unsigned int physical = position % capacity;
        unsigned long long cache =
            (((unsigned long long)cache_lane * capacity + physical) * kv_heads + kv_head) * dim;
        float partial = 0.0f;
        for (unsigned int d = thread_lane; d < dim; d += 32) {
            partial += (float)q[query + d] * (float)key_cache[cache + d];
        }
        float score = warp_sum_f32(partial) * scale_f32;
        float next_maximum = fmaxf(maximum, score);
        float previous_weight = maximum == -3.402823466e38f
            ? 0.0f
            : expf(maximum - next_maximum);
        float current_weight = expf(score - next_maximum);
        denominator = denominator * previous_weight + current_weight;
#pragma unroll
        for (unsigned int i = 0; i < 8; ++i) {
            unsigned int d = thread_lane + i * 32;
            if (d < dim) {
                result[i] = result[i] * previous_weight
                    + current_weight * (float)value_cache[cache + d];
            }
        }
        maximum = next_maximum;
    }
#pragma unroll
    for (unsigned int i = 0; i < 8; ++i) {
        unsigned int d = thread_lane + i * 32;
        if (d < dim) {
            output[query + d] = cast_dtype((double)(result[i] / denominator), dtype);
        }
    }
}

__device__ void run_kv_attention_warp_f64(
    const double *q,
    const double *key_cache,
    const double *value_cache,
    double *output,
    const unsigned int *cursors,
    const unsigned int *valid,
    unsigned int batch,
    unsigned int q_heads,
    unsigned int kv_heads,
    unsigned int tokens,
    unsigned int dim,
    unsigned int capacity,
    unsigned int rows_per_sequence,
    double scale,
    unsigned int window,
    unsigned int bidirectional,
    unsigned int dtype
) {
    unsigned int thread_lane = threadIdx.x & 31U;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int warps_per_block = blockDim.x >> 5;
    unsigned long long query_index =
        (unsigned long long)blockIdx.x * warps_per_block + warp;
    unsigned long long query_count =
        (unsigned long long)batch * q_heads * tokens;
    if (query_index >= query_count) return;

    unsigned int token = query_index % tokens;
    unsigned long long rest = query_index / tokens;
    unsigned int q_head = rest % q_heads;
    unsigned int lane = rest / q_heads;
    unsigned long long query = query_index * dim;
    if (token >= valid[lane]) {
        for (unsigned int d = thread_lane; d < dim; d += 32) {
            output[query + d] = 0.0;
        }
        return;
    }

    unsigned int kv_head = q_head * kv_heads / q_heads;
    unsigned int cache_lane = lane / rows_per_sequence;
    unsigned int end = cursors[lane] + (bidirectional ? valid[lane] - 1 : token);
    unsigned int start = window == 0 || end + 1 <= window ? 0 : end + 1 - window;
    unsigned int retained_start = end + 1 <= capacity ? 0 : end + 1 - capacity;
    start = max(start, retained_start);

    double result[8];
#pragma unroll
    for (unsigned int i = 0; i < 8; ++i) result[i] = 0.0;
    double maximum = -1.7976931348623157e308;
    double denominator = 0.0;
    for (unsigned int position = start; position <= end; ++position) {
        unsigned int physical = position % capacity;
        unsigned long long cache =
            (((unsigned long long)cache_lane * capacity + physical) * kv_heads + kv_head) * dim;
        double partial = 0.0;
        for (unsigned int d = thread_lane; d < dim; d += 32) {
            partial += q[query + d] * key_cache[cache + d];
        }
        double score = warp_sum_f64(partial) * scale;
        double next_maximum = fmax(maximum, score);
        double previous_weight = maximum == -1.7976931348623157e308
            ? 0.0
            : exp(maximum - next_maximum);
        double current_weight = exp(score - next_maximum);
        denominator = denominator * previous_weight + current_weight;
#pragma unroll
        for (unsigned int i = 0; i < 8; ++i) {
            unsigned int d = thread_lane + i * 32;
            if (d < dim) {
                result[i] = result[i] * previous_weight
                    + current_weight * value_cache[cache + d];
            }
        }
        maximum = next_maximum;
    }
#pragma unroll
    for (unsigned int i = 0; i < 8; ++i) {
        unsigned int d = thread_lane + i * 32;
        if (d < dim) {
            output[query + d] = cast_dtype(result[i] / denominator, dtype);
        }
    }
}

extern "C" __global__ void kv_attention_warp_f64(
    const double *q,
    const double *key_cache,
    const double *value_cache,
    double *output,
    const unsigned int *cursors,
    const unsigned int *valid,
    unsigned int batch,
    unsigned int q_heads,
    unsigned int kv_heads,
    unsigned int tokens,
    unsigned int dim,
    unsigned int capacity,
    double scale,
    unsigned int window,
    unsigned int bidirectional,
    unsigned int dtype
) {
    run_kv_attention_warp_f64(
        q, key_cache, value_cache, output, cursors, valid, batch, q_heads,
        kv_heads, tokens, dim, capacity, 1, scale, window, bidirectional, dtype
    );
}

extern "C" __global__ void kv_attention_warp_f32(
    const double *q,
    const double *key_cache,
    const double *value_cache,
    double *output,
    const unsigned int *cursors,
    const unsigned int *valid,
    unsigned int batch,
    unsigned int q_heads,
    unsigned int kv_heads,
    unsigned int tokens,
    unsigned int dim,
    unsigned int capacity,
    double scale,
    unsigned int window,
    unsigned int bidirectional,
    unsigned int dtype
) {
    run_kv_attention_warp_f32(
        q, key_cache, value_cache, output, cursors, valid, batch, q_heads,
        kv_heads, tokens, dim, capacity, 1, scale, window, bidirectional, dtype
    );
}

extern "C" __global__ void kv_attention_packed_warp_f64(
    const double *q,
    const double *key_cache,
    const double *value_cache,
    double *output,
    const unsigned int *cursors,
    const unsigned int *valid,
    unsigned int batch,
    unsigned int q_heads,
    unsigned int kv_heads,
    unsigned int tokens,
    unsigned int dim,
    unsigned int capacity,
    unsigned int rows_per_sequence,
    double scale,
    unsigned int window,
    unsigned int bidirectional,
    unsigned int dtype
) {
    run_kv_attention_warp_f64(
        q, key_cache, value_cache, output, cursors, valid, batch, q_heads,
        kv_heads, tokens, dim, capacity, rows_per_sequence, scale, window,
        bidirectional, dtype
    );
}

extern "C" __global__ void kv_attention_packed_warp_f32(
    const double *q,
    const double *key_cache,
    const double *value_cache,
    double *output,
    const unsigned int *cursors,
    const unsigned int *valid,
    unsigned int batch,
    unsigned int q_heads,
    unsigned int kv_heads,
    unsigned int tokens,
    unsigned int dim,
    unsigned int capacity,
    unsigned int rows_per_sequence,
    double scale,
    unsigned int window,
    unsigned int bidirectional,
    unsigned int dtype
) {
    run_kv_attention_warp_f32(
        q, key_cache, value_cache, output, cursors, valid, batch, q_heads,
        kv_heads, tokens, dim, capacity, rows_per_sequence, scale, window,
        bidirectional, dtype
    );
}

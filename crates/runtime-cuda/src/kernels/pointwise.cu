extern "C" __global__ void greedy_argmax_f64(
    const double *logits,
    unsigned int len,
    unsigned int *result
) {
    __shared__ double values[256];
    __shared__ unsigned int tokens[256];
    __shared__ unsigned int invalid[256];

    unsigned int thread = threadIdx.x;
    double best = -3.40282346638528859812e38;
    unsigned int selected = 0;
    unsigned int first_invalid = 0xffffffffU;
    for (unsigned int token = thread; token < len; token += blockDim.x) {
        double value = logits[token];
        if (!isfinite(value)) {
            first_invalid = min(first_invalid, token);
        } else if (value > best || (value == best && token < selected)) {
            best = value;
            selected = token;
        }
    }
    values[thread] = best;
    tokens[thread] = selected;
    invalid[thread] = first_invalid;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
        if (thread < stride) {
            double other = values[thread + stride];
            unsigned int other_token = tokens[thread + stride];
            if (other > values[thread]
                || (other == values[thread] && other_token < tokens[thread])) {
                values[thread] = other;
                tokens[thread] = other_token;
            }
            invalid[thread] = min(invalid[thread], invalid[thread + stride]);
        }
        __syncthreads();
    }
    if (thread == 0) {
        result[0] = tokens[0];
        result[1] = invalid[0];
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void topk_f64(
    const double *logits,
    unsigned int len,
    unsigned int k,
    double *output
) {
    constexpr unsigned int MAX_TOP_K = 40;
    __shared__ double warp_values[8];
    __shared__ unsigned int warp_tokens[8];
    __shared__ unsigned int warp_owners[8];
    __shared__ unsigned int winner;
    __shared__ unsigned int invalid[256];
    double local_values[MAX_TOP_K];
    unsigned int local_tokens[MAX_TOP_K];
    unsigned int thread = threadIdx.x;
    unsigned int first_invalid = 0xffffffffU;

#pragma unroll
    for (unsigned int rank = 0; rank < MAX_TOP_K; ++rank) {
        local_values[rank] = -3.40282346638528859812e38;
        local_tokens[rank] = 0xffffffffU;
    }
    for (unsigned int token = blockIdx.x * blockDim.x + thread;
        token < len;
        token += blockDim.x * gridDim.x) {
        double value = logits[token];
        if (!isfinite(value)) {
            first_invalid = min(first_invalid, token);
            continue;
        }
        unsigned int rank = k - 1;
        if (value < local_values[rank]
            || (value == local_values[rank] && token >= local_tokens[rank])) {
            continue;
        }
        while (rank > 0
            && (value > local_values[rank - 1]
                || (value == local_values[rank - 1] && token < local_tokens[rank - 1]))) {
            local_values[rank] = local_values[rank - 1];
            local_tokens[rank] = local_tokens[rank - 1];
            --rank;
        }
        local_values[rank] = value;
        local_tokens[rank] = token;
    }

    invalid[thread] = first_invalid;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
        if (thread < stride) invalid[thread] = min(invalid[thread], invalid[thread + stride]);
        __syncthreads();
    }
    if (thread == 0) {
        ((unsigned int *)output)[2 * gridDim.x * MAX_TOP_K + blockIdx.x] = invalid[0];
    }

    unsigned int lane = thread & 31U;
    unsigned int warp = thread / 32;
    unsigned int local_rank = 0;
    for (unsigned int output_rank = 0; output_rank < k; ++output_rank) {
        double best_value = local_values[local_rank];
        unsigned int best_token = local_tokens[local_rank];
        unsigned int best_owner = thread;
#pragma unroll
        for (unsigned int offset = 16; offset != 0; offset >>= 1) {
            double other = __shfl_down_sync(0xffffffffU, best_value, offset);
            unsigned int other_token =
                __shfl_down_sync(0xffffffffU, best_token, offset);
            unsigned int other_owner =
                __shfl_down_sync(0xffffffffU, best_owner, offset);
            if (lane + offset < 32
                && (other > best_value
                    || (other == best_value && other_token < best_token))) {
                best_value = other;
                best_token = other_token;
                best_owner = other_owner;
            }
        }
        if (lane == 0) {
            warp_values[warp] = best_value;
            warp_tokens[warp] = best_token;
            warp_owners[warp] = best_owner;
        }
        __syncthreads();
        if (warp == 0) {
            best_value = lane < 8 ? warp_values[lane] : -3.40282346638528859812e38;
            best_token = lane < 8 ? warp_tokens[lane] : 0xffffffffU;
            best_owner = lane < 8 ? warp_owners[lane] : 0xffffffffU;
#pragma unroll
            for (unsigned int offset = 16; offset != 0; offset >>= 1) {
                double other = __shfl_down_sync(0xffffffffU, best_value, offset);
                unsigned int other_token =
                    __shfl_down_sync(0xffffffffU, best_token, offset);
                unsigned int other_owner =
                    __shfl_down_sync(0xffffffffU, best_owner, offset);
                if (lane + offset < 32
                    && (other > best_value
                        || (other == best_value && other_token < best_token))) {
                    best_value = other;
                    best_token = other_token;
                    best_owner = other_owner;
                }
            }
            if (lane == 0) {
                unsigned int output_offset = blockIdx.x * MAX_TOP_K + output_rank;
                output[output_offset] = best_value;
                ((unsigned int *)output)[gridDim.x * MAX_TOP_K + output_offset] = best_token;
                winner = best_owner;
            }
        }
        __syncthreads();
        if (thread == winner) ++local_rank;
    }
}

__device__ unsigned long long mix64(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ double random_unit(unsigned long long seed) {
    return ((mix64(seed) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

__device__ void random_f64(
    double *out,
    unsigned int len,
    unsigned long long seed,
    unsigned int normal,
    double lo,
    double hi,
    unsigned int dtype
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    double u1 = random_unit(seed + (unsigned long long)index * 2ULL);
    double value;
    if (normal != 0) {
        double u2 = random_unit(seed + (unsigned long long)index * 2ULL + 1ULL);
        value = sqrt(-2.0 * log(u1)) * cos(6.2831853071795864769 * u2);
    } else {
        value = lo + (hi - lo) * u1;
    }
    out[index] = cast_dtype(value, dtype);
}

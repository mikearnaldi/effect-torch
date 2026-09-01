extern "C" __global__ void fill_f64(
    double *out,
    double value,
    unsigned int len,
    unsigned int dtype
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) out[index] = cast_dtype(value, dtype);
}

extern "C" __global__ void muse_gate_f32(
    const float *value,
    const float *gate,
    const float *multiplier,
    float *out,
    unsigned int len,
    unsigned int has_multiplier
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    float half_gate = __fmul_rn(gate[index], 0.5f);
    float sigmoid = __fadd_rn(__fmul_rn(tanhf(half_gate), 0.5f), 0.5f);
    float result = __fmul_rn(value[index], sigmoid);
    out[index] = has_multiplier == 0
        ? result
        : __fmul_rn(result, multiplier[index]);
}

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
        output[2 * gridDim.x * MAX_TOP_K + blockIdx.x] =
            invalid[0] == 0xffffffffU ? -1.0 : (double)invalid[0];
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
                output[gridDim.x * MAX_TOP_K + output_offset] = (double)best_token;
                winner = best_owner;
            }
        }
        __syncthreads();
        if (thread == winner) ++local_rank;
    }
}

extern "C" __global__ void binary_f64(
    unsigned int op,
    const double *a,
    const double *b,
    double *out,
    unsigned int len,
    unsigned int rank,
    const unsigned long long *shapes,
    unsigned int dtype
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    const unsigned long long *a_shape = shapes + rank;
    const unsigned long long *b_shape = shapes + rank * 2;
    double x = a[mapped_index(index, rank, shapes, a_shape)];
    double y = b[mapped_index(index, rank, shapes, b_shape)];
    double value = 0.0;
    switch (op) {
        case 0: value = x + y; break;
        case 1: value = x - y; break;
        case 2: value = x * y; break;
        case 3: value = x / y; break;
        case 4: value = fmax(x, y); break;
        case 5: value = fmin(x, y); break;
        case 6: value = x == y; break;
        case 7: value = x > y; break;
        case 8: value = x < y; break;
        case 9: value = x >= y; break;
        case 10: value = x <= y; break;
    }
    out[index] = cast_dtype(value, dtype);
}

extern "C" __global__ void binary_i64(
    unsigned int op,
    const long long *a,
    const long long *b,
    long long *out,
    double *out_f64,
    unsigned int len,
    unsigned int rank,
    const unsigned long long *shapes
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    const unsigned long long *a_shape = shapes + rank;
    const unsigned long long *b_shape = shapes + rank * 2;
    long long x = a[mapped_index(index, rank, shapes, a_shape)];
    long long y = b[mapped_index(index, rank, shapes, b_shape)];
    long long value = 0;
    switch (op) {
        case 0: value = x + y; break;
        case 1: value = x - y; break;
        case 2: value = x * y; break;
        case 3: value = x / y; break;
        case 4: value = x > y ? x : y; break;
        case 5: value = x < y ? x : y; break;
    }
    out[index] = value;
    out_f64[index] = (double)value;
}

extern "C" __global__ void compare_i64(
    unsigned int op,
    const long long *a,
    const long long *b,
    double *out,
    unsigned int len,
    unsigned int rank,
    const unsigned long long *shapes
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    const unsigned long long *a_shape = shapes + rank;
    const unsigned long long *b_shape = shapes + rank * 2;
    long long x = a[mapped_index(index, rank, shapes, a_shape)];
    long long y = b[mapped_index(index, rank, shapes, b_shape)];
    switch (op) {
        case 6: out[index] = x == y; break;
        case 7: out[index] = x > y; break;
        case 8: out[index] = x < y; break;
        case 9: out[index] = x >= y; break;
        default: out[index] = x <= y; break;
    }
}

extern "C" __global__ void unary_i64(
    unsigned int op,
    const long long *a,
    long long *out,
    double *out_f64,
    unsigned int len
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    long long x = a[index];
    long long value;
    switch (op) {
        case 0: value = -x; break;
        case 1: value = x < 0 ? -x : x; break;
        case 8: value = x > 0 ? x : 0; break;
        case 13: value = (x > 0) - (x < 0); break;
        default: value = x; break;
    }
    out[index] = value;
    out_f64[index] = (double)value;
}

extern "C" __global__ void cast_from_i64(
    const long long *a,
    double *out,
    unsigned int len,
    unsigned int dtype
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) out[index] = cast_dtype((double)a[index], dtype);
}

extern "C" __global__ void cast_to_i64(
    const double *a,
    long long *out,
    double *out_f64,
    unsigned int len
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    long long value = (long long)a[index];
    out[index] = value;
    out_f64[index] = (double)value;
}

extern "C" __global__ void unary_f64(
    unsigned int op,
    const double *a,
    double *out,
    unsigned int len,
    double parameter,
    unsigned int dtype
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    double x = a[index];
    double value = x;
    switch (op) {
        case 0: value = -x; break;
        case 1: value = fabs(x); break;
        case 2: value = sqrt(x); break;
        case 3: value = exp(x); break;
        case 4: value = log(x); break;
        case 5: value = sin(x); break;
        case 6: value = cos(x); break;
        case 7: value = tanh(x); break;
        case 8: value = fmax(x, 0.0); break;
        case 9: value = erf(x); break;
        case 10: value = floor(x); break;
        case 11: value = ceil(x); break;
        case 12: value = round(x); break;
        case 13: value = (x > 0.0) - (x < 0.0); break;
        case 14: value = pow(x, parameter); break;
        case 15: value = 0.5 * x * (1.0 + erf(x * 0.7071067811865475244)); break;
        case 16: {
            double inner = 0.7978845608028653559 * (x + 0.044715 * x * x * x);
            value = 0.5 * x * (1.0 + tanh(inner));
            break;
        }
        case 17: value = x; break;
    }
    out[index] = cast_dtype(value, dtype);
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

extern "C" __global__ void random_f64(
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

extern "C" __global__ void sequence_f64(
    unsigned int op,
    double *out,
    unsigned int len,
    double start,
    double step,
    unsigned int width,
    unsigned int dtype
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    double value = op == 0 ? start + step * index : (index / width == index % width ? 1.0 : 0.0);
    out[index] = cast_dtype(value, dtype);
}

extern "C" __global__ void reindex_f64(
    unsigned int op,
    const double *a,
    double *out,
    unsigned int len,
    unsigned int rank,
    const unsigned long long *metadata,
    unsigned int dtype
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    const unsigned long long *out_shape = metadata;
    const unsigned long long *in_shape = metadata + rank;
    const unsigned long long *parameters = metadata + rank * 2;
    unsigned long long linear = index;
    unsigned long long input_index = 0;
    for (int axis = (int)rank - 1; axis >= 0; --axis) {
        unsigned long long width = out_shape[axis];
        unsigned long long coordinate = width == 0 ? 0 : linear % width;
        linear = width == 0 ? 0 : linear / width;
        unsigned long long input_axis = op == 1 ? parameters[axis] : (unsigned long long)axis;
        unsigned long long input_coordinate = coordinate;
        if (op == 0 && in_shape[input_axis] == 1) input_coordinate = 0;
        if (op == 2) input_coordinate = parameters[axis * 2] + coordinate * parameters[axis * 2 + 1];
        unsigned long long stride = 1;
        for (unsigned long long inner = input_axis + 1; inner < rank; ++inner) stride *= in_shape[inner];
        input_index += input_coordinate * stride;
    }
    out[index] = cast_dtype(a[input_index], dtype);
}

extern "C" __global__ void where_f64(
    const double *cond,
    const double *a,
    const double *b,
    double *out,
    unsigned int len,
    unsigned int rank,
    const unsigned long long *shapes,
    unsigned int dtype
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    const unsigned long long *cond_shape = shapes + rank;
    const unsigned long long *a_shape = shapes + rank * 2;
    const unsigned long long *b_shape = shapes + rank * 3;
    unsigned long long ci = mapped_index(index, rank, shapes, cond_shape);
    unsigned long long ai = mapped_index(index, rank, shapes, a_shape);
    unsigned long long bi = mapped_index(index, rank, shapes, b_shape);
    out[index] = cast_dtype(cond[ci] != 0.0 ? a[ai] : b[bi], dtype);
}

extern "C" __global__ void concat_f64(
    const double *a,
    const double *b,
    double *out,
    unsigned int len,
    unsigned int rank,
    unsigned int dim,
    const unsigned long long *shapes,
    unsigned int dtype
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    const unsigned long long *a_shape = shapes + rank;
    const unsigned long long *b_shape = shapes + rank * 2;
    unsigned long long linear = index;
    unsigned long long ai = 0;
    unsigned long long bi = 0;
    unsigned long long coordinate_at_dim = 0;
    for (int axis = (int)rank - 1; axis >= 0; --axis) {
        unsigned long long coordinate = linear % shapes[axis];
        linear /= shapes[axis];
        if ((unsigned int)axis == dim) coordinate_at_dim = coordinate;
        unsigned long long a_stride = 1;
        unsigned long long b_stride = 1;
        for (int inner = axis + 1; inner < (int)rank; ++inner) {
            a_stride *= a_shape[inner];
            b_stride *= b_shape[inner];
        }
        ai += coordinate * a_stride;
        bi += (coordinate - ((unsigned int)axis == dim ? a_shape[dim] : 0)) * b_stride;
    }
    out[index] = cast_dtype(coordinate_at_dim < a_shape[dim] ? a[ai] : b[bi], dtype);
}

extern "C" __global__ void reduce_f64(
    unsigned int op,
    const double *a,
    double *out,
    unsigned int out_len,
    unsigned int rank,
    const unsigned long long *metadata,
    unsigned int dtype
) {
    unsigned int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index >= out_len) return;
    const unsigned long long *shape = metadata;
    const unsigned long long *reduced = metadata + rank;
    unsigned long long reduction_count = 1;
    for (unsigned int axis = 0; axis < rank; ++axis) if (reduced[axis]) reduction_count *= shape[axis];
    double result = op == 1 ? 1.0 : op == 2 ? -1.0 / 0.0 : op == 3 ? 1.0 / 0.0 : 0.0;
    for (unsigned long long reduction = 0; reduction < reduction_count; ++reduction) {
        unsigned long long output_linear = output_index;
        unsigned long long reduction_linear = reduction;
        unsigned long long input_index = 0;
        unsigned long long stride = 1;
        for (int axis = (int)rank - 1; axis >= 0; --axis) {
            unsigned long long coordinate;
            if (reduced[axis]) {
                coordinate = reduction_linear % shape[axis];
                reduction_linear /= shape[axis];
            } else {
                coordinate = output_linear % shape[axis];
                output_linear /= shape[axis];
            }
            input_index += coordinate * stride;
            stride *= shape[axis];
        }
        double value = a[input_index];
        if (op == 0 || op == 4) result += value;
        else if (op == 1) result *= value;
        else if (op == 2) result = fmax(result, value);
        else if (op == 3) result = fmin(result, value);
    }
    if (op == 4) result /= reduction_count;
    out[output_index] = cast_dtype(result, dtype);
}

extern "C" __global__ void matmul_f64(
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
    unsigned long long linear = index;
    unsigned long long n = shapes[rank - 1];
    unsigned long long m = shapes[rank - 2];
    unsigned long long column = linear % n;
    linear /= n;
    unsigned long long row = linear % m;
    linear /= m;
    unsigned long long a_batch = 0;
    unsigned long long b_batch = 0;
    unsigned long long a_batch_stride = 1;
    unsigned long long b_batch_stride = 1;
    for (int axis = (int)rank - 3; axis >= 0; --axis) {
        unsigned long long coordinate = linear % shapes[axis];
        linear /= shapes[axis];
        if (a_shape[axis] != 1) a_batch += coordinate * a_batch_stride;
        if (b_shape[axis] != 1) b_batch += coordinate * b_batch_stride;
        a_batch_stride *= a_shape[axis];
        b_batch_stride *= b_shape[axis];
    }
    unsigned long long k_width = a_shape[rank - 1];
    unsigned long long a_matrix = a_shape[rank - 2] * k_width;
    unsigned long long b_matrix = k_width * b_shape[rank - 1];
    double value = 0.0;
    for (unsigned long long k = 0; k < k_width; ++k) {
        value += a[a_batch * a_matrix + row * k_width + k] *
            b[b_batch * b_matrix + k * b_shape[rank - 1] + column];
    }
    out[index] = cast_dtype(value, dtype);
}

extern "C" __global__ void index_f64(
    unsigned int op,
    const double *a,
    const double *indexes,
    const double *src,
    double *out,
    unsigned int out_len,
    unsigned int src_len,
    unsigned int rank,
    unsigned int dim,
    const unsigned long long *shapes,
    unsigned int dtype
) {
    unsigned int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index >= out_len) return;
    const unsigned long long *a_shape = shapes;
    const unsigned long long *out_shape = shapes + rank;
    unsigned long long linear = output_index;
    unsigned long long input_index = 0;
    unsigned long long dim_coordinate = 0;
    unsigned long long stride = 1;
    for (int axis = (int)rank - 1; axis >= 0; --axis) {
        unsigned long long coordinate;
        if ((op == 0 || op == 1) && (unsigned int)axis == dim) {
            coordinate = 0;
        } else {
            unsigned long long width = out_shape[axis];
            coordinate = linear % width;
            linear /= width;
        }
        if ((unsigned int)axis == dim) dim_coordinate = coordinate;
        input_index += coordinate * stride;
        stride *= a_shape[axis];
    }
    if (op == 0 || op == 1) {
        unsigned long long dim_stride = 1;
        for (unsigned int axis = dim + 1; axis < rank; ++axis) dim_stride *= a_shape[axis];
        double best = a[input_index];
        unsigned long long best_index = 0;
        for (unsigned long long candidate = 1; candidate < a_shape[dim]; ++candidate) {
            double value = a[input_index + candidate * dim_stride];
            if ((op == 0 && value > best) || (op == 1 && value < best)) {
                best = value;
                best_index = candidate;
            }
        }
        out[output_index] = (double)best_index;
    } else if (op == 2) {
        unsigned long long dim_stride = 1;
        for (unsigned int axis = dim + 1; axis < rank; ++axis) dim_stride *= a_shape[axis];
        double total = 0.0;
        unsigned long long base = input_index - dim_coordinate * dim_stride;
        for (unsigned long long candidate = 0; candidate <= dim_coordinate; ++candidate) {
            total += a[base + candidate * dim_stride];
        }
        out[output_index] = cast_dtype(total, dtype);
    } else if (op == 3) {
        unsigned long long selected = (unsigned long long)indexes[dim_coordinate];
        unsigned long long dim_stride = 1;
        for (unsigned int axis = dim + 1; axis < rank; ++axis) dim_stride *= a_shape[axis];
        out[output_index] = cast_dtype(a[input_index - dim_coordinate * dim_stride + selected * dim_stride], dtype);
    } else if (op == 4) {
        unsigned long long selected = (unsigned long long)indexes[output_index];
        unsigned long long dim_stride = 1;
        for (unsigned int axis = dim + 1; axis < rank; ++axis) dim_stride *= a_shape[axis];
        out[output_index] = cast_dtype(a[input_index - dim_coordinate * dim_stride + selected * dim_stride], dtype);
    } else {
        double total = a[output_index];
        for (unsigned long long source_index = 0; source_index < src_len; ++source_index) {
            unsigned long long source_linear = source_index;
            unsigned long long target_index = 0;
            unsigned long long target_stride = 1;
            for (int axis = (int)rank - 1; axis >= 0; --axis) {
                unsigned long long coordinate = source_linear % out_shape[axis];
                source_linear /= out_shape[axis];
                if ((unsigned int)axis == dim) coordinate = (unsigned long long)indexes[source_index];
                target_index += coordinate * target_stride;
                target_stride *= a_shape[axis];
            }
            if (target_index == output_index) total += src[source_index];
        }
        out[output_index] = cast_dtype(total, dtype);
    }
}

extern "C" __global__ void rms_norm_f64(
    const double *x,
    const double *weight,
    double *out,
    unsigned int len,
    unsigned int width,
    unsigned int weighted,
    double eps,
    unsigned int dtype
) {
    unsigned int lane = threadIdx.x & 31U;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int warps_per_block = blockDim.x >> 5;
    unsigned int row = blockIdx.x * warps_per_block + warp;
    unsigned int rows = len / width;
    if (row >= rows) return;

    double sum = 0.0;
    for (unsigned int column = lane; column < width; column += 32) {
        double value = x[(unsigned long long)row * width + column];
        sum += value * value;
    }
#pragma unroll
    for (unsigned int offset = 16; offset != 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    double inverse_rms = 1.0 / sqrt(__shfl_sync(0xffffffffU, sum, 0) / width + eps);
    for (unsigned int column = lane; column < width; column += 32) {
        unsigned long long index = (unsigned long long)row * width + column;
        double value = x[index] * inverse_rms;
        if (weighted != 0) value *= weight[column];
        out[index] = cast_dtype(value, dtype);
    }
}

extern "C" __global__ void rms_norm_block_f64(
    const double *x,
    const double *weight,
    double *out,
    unsigned int len,
    unsigned int width,
    unsigned int weighted,
    double eps,
    unsigned int dtype
) {
    unsigned int row = blockIdx.x;
    unsigned int rows = len / width;
    if (row >= rows) return;

    unsigned int lane = threadIdx.x & 31U;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int warps = blockDim.x >> 5;
    double sum = 0.0;
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        double value = x[(unsigned long long)row * width + column];
        sum += value * value;
    }
#pragma unroll
    for (unsigned int offset = 16; offset != 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    __shared__ double warp_sums[32];
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = lane < warps ? warp_sums[lane] : 0.0;
#pragma unroll
        for (unsigned int offset = 16; offset != 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffU, sum, offset);
        }
        if (lane == 0) warp_sums[0] = sum;
    }
    __syncthreads();

    double inverse_rms = 1.0 / sqrt(warp_sums[0] / width + eps);
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        unsigned long long index = (unsigned long long)row * width + column;
        double value = x[index] * inverse_rms;
        if (weighted != 0) value *= weight[column];
        out[index] = cast_dtype(value, dtype);
    }
}

extern "C" __global__ void muse_residual_rms_norm_f32(
    const float *x,
    const float *weight,
    const float *residual,
    float *out,
    unsigned int len,
    unsigned int width,
    unsigned int weighted,
    float eps
) {
    unsigned int row = blockIdx.x;
    unsigned int rows = len / width;
    if (row >= rows) return;

    unsigned int lane = threadIdx.x & 31U;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int warps = blockDim.x >> 5;
    float sum = 0.0f;
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        float value = x[(unsigned long long)row * width + column];
        sum += value * value;
    }
#pragma unroll
    for (unsigned int offset = 16; offset != 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    __shared__ float warp_sums[32];
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = lane < warps ? warp_sums[lane] : 0.0f;
#pragma unroll
        for (unsigned int offset = 16; offset != 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffU, sum, offset);
        }
        if (lane == 0) warp_sums[0] = sum;
    }
    __syncthreads();

    float inverse_rms = rsqrtf(warp_sums[0] / width + eps);
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        unsigned long long index = (unsigned long long)row * width + column;
        float value = x[index] * inverse_rms;
        if (weighted != 0) value *= weight[column];
        out[index] = __fadd_rn(residual[index], value);
    }
}

extern "C" __global__ void cross_entropy_f64(
    const double *logits,
    const double *target,
    double *out,
    unsigned int rows,
    unsigned int classes,
    long long ignore_index,
    unsigned int dtype
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double total = 0.0;
    unsigned int active = 0;
    for (unsigned int row = 0; row < rows; ++row) {
        long long selected = (long long)target[row];
        if (selected == ignore_index) continue;
        if (selected < 0 || selected >= classes) { out[0] = 0.0 / 0.0; return; }
        double maximum = -1.0 / 0.0;
        for (unsigned int column = 0; column < classes; ++column) maximum = fmax(maximum, logits[row * classes + column]);
        double sum = 0.0;
        for (unsigned int column = 0; column < classes; ++column) sum += exp(logits[row * classes + column] - maximum);
        total += maximum + log(sum) - logits[row * classes + selected];
        active += 1;
    }
    out[0] = active == 0 ? 0.0 / 0.0 : cast_dtype(total / active, dtype);
}

extern "C" __global__ void cross_entropy_backward_f64(
    const double *logits,
    const double *target,
    double *out,
    unsigned int len,
    unsigned int classes,
    long long ignore_index,
    unsigned int dtype
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    unsigned int rows = len / classes;
    unsigned int row = index / classes;
    unsigned int column = index % classes;
    long long selected = (long long)target[row];
    if (selected == ignore_index) { out[index] = 0.0; return; }
    unsigned int active = 0;
    for (unsigned int i = 0; i < rows; ++i) if ((long long)target[i] != ignore_index) active += 1;
    double maximum = -1.0 / 0.0;
    for (unsigned int i = 0; i < classes; ++i) maximum = fmax(maximum, logits[row * classes + i]);
    double sum = 0.0;
    for (unsigned int i = 0; i < classes; ++i) sum += exp(logits[row * classes + i] - maximum);
    double value = exp(logits[index] - maximum) / sum - (column == selected ? 1.0 : 0.0);
    out[index] = cast_dtype(value / active, dtype);
}

__device__ double chunked_head_logit_f64(
    const double *x,
    const double *weight,
    const double *bias,
    unsigned int row,
    unsigned int column,
    unsigned int inner,
    unsigned int vocab
) {
    double value = bias[column];
    for (unsigned int index = 0; index < inner; ++index) {
        value += x[(unsigned long long)row * inner + index]
            * weight[(unsigned long long)index * vocab + column];
    }
    return value;
}

__device__ void chunked_head_row_stats_f64(
    const double *x,
    const double *weight,
    const double *bias,
    unsigned int row,
    unsigned int inner,
    unsigned int vocab,
    double *maximum,
    double *sum
) {
    *maximum = -1.0 / 0.0;
    for (unsigned int column = 0; column < vocab; ++column) {
        *maximum = fmax(
            *maximum,
            chunked_head_logit_f64(x, weight, bias, row, column, inner, vocab)
        );
    }
    *sum = 0.0;
    for (unsigned int column = 0; column < vocab; ++column) {
        *sum += exp(
            chunked_head_logit_f64(x, weight, bias, row, column, inner, vocab)
                - *maximum
        );
    }
}

extern "C" __global__ void chunked_head_ce_f64(
    const double *x,
    const double *weight,
    const double *bias,
    const long long *target,
    double *out,
    unsigned int rows,
    unsigned int inner,
    unsigned int vocab,
    unsigned int active,
    long long ignore_index,
    unsigned int dtype
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double total = 0.0;
    for (unsigned int row = 0; row < rows; ++row) {
        long long selected = target[row];
        if (selected == ignore_index) continue;
        double maximum;
        double sum;
        chunked_head_row_stats_f64(
            x, weight, bias, row, inner, vocab, &maximum, &sum
        );
        total += maximum + log(sum)
            - chunked_head_logit_f64(
                x, weight, bias, row, (unsigned int)selected, inner, vocab
            );
    }
    out[0] = cast_dtype(total / active, dtype);
}

extern "C" __global__ void chunked_head_ce_backward_f64(
    unsigned int output_kind,
    const double *x,
    const double *weight,
    const double *bias,
    const long long *target,
    const double *gradient,
    double *out,
    unsigned int len,
    unsigned int rows,
    unsigned int inner,
    unsigned int vocab,
    unsigned int active,
    long long ignore_index,
    unsigned int dtype
) {
    unsigned int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index >= len) return;
    double scale = gradient[0] / active;
    double result = 0.0;
    if (output_kind == 0) {
        unsigned int row = output_index / inner;
        unsigned int feature = output_index % inner;
        long long selected = target[row];
        if (selected != ignore_index) {
            double maximum;
            double sum;
            chunked_head_row_stats_f64(
                x, weight, bias, row, inner, vocab, &maximum, &sum
            );
            for (unsigned int column = 0; column < vocab; ++column) {
                double probability = exp(
                    chunked_head_logit_f64(
                        x, weight, bias, row, column, inner, vocab
                    ) - maximum
                ) / sum;
                double grad = (probability - (column == (unsigned int)selected ? 1.0 : 0.0))
                    * scale;
                result += grad * weight[(unsigned long long)feature * vocab + column];
            }
        }
    } else {
        unsigned int feature = output_kind == 1 ? output_index / vocab : 0;
        unsigned int column = output_kind == 1 ? output_index % vocab : output_index;
        for (unsigned int row = 0; row < rows; ++row) {
            long long selected = target[row];
            if (selected == ignore_index) continue;
            double maximum;
            double sum;
            chunked_head_row_stats_f64(
                x, weight, bias, row, inner, vocab, &maximum, &sum
            );
            double probability = exp(
                chunked_head_logit_f64(
                    x, weight, bias, row, column, inner, vocab
                ) - maximum
            ) / sum;
            double grad = (probability - (column == (unsigned int)selected ? 1.0 : 0.0))
                * scale;
            result += output_kind == 1
                ? x[(unsigned long long)row * inner + feature] * grad
                : grad;
        }
    }
    out[output_index] = cast_dtype(result, dtype);
}

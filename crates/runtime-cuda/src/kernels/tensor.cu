__device__ void reduce_f64(
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

__device__ void matmul_f64(
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

__device__ void chunked_head_ce_f64(
    const double *x,
    const double *weight,
    const double *bias,
    const void *target, unsigned int target_dtype,
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
        long long selected = et_load<et_i64>((et_u64)target, target_dtype, row);
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

__device__ void chunked_head_ce_backward_f64(
    unsigned int output_kind,
    const double *x,
    const double *weight,
    const double *bias,
    const void *target, unsigned int target_dtype,
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
        long long selected = et_load<et_i64>((et_u64)target, target_dtype, row);
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
            long long selected = et_load<et_i64>((et_u64)target, target_dtype, row);
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

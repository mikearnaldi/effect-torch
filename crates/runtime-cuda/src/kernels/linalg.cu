extern "C" __global__ void conv_f64(
    unsigned int op,
    const double *x,
    const double *w,
    double *out,
    unsigned int len,
    const unsigned long long *shape,
    unsigned int stride,
    unsigned int padding,
    unsigned int dilation,
    unsigned int groups,
    unsigned int dtype
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    const unsigned long long *xs = shape;
    const unsigned long long *ws = shape + 4;
    const unsigned long long *os = shape + 8;
    double total = 0.0;
    if (op == 0) {
        unsigned long long ox = index % os[2];
        unsigned long long oc = (index / os[2]) % os[1];
        unsigned long long n = index / (os[1] * os[2]);
        unsigned long long cper = ws[1];
        unsigned long long group = oc / (ws[0] / groups);
        for (unsigned long long ci = 0; ci < cper; ++ci) for (unsigned long long k = 0; k < ws[2]; ++k) {
            long long ix = (long long)(ox * stride + k * dilation) - padding;
            if (ix >= 0 && ix < xs[2]) total += x[(n * xs[1] + group * cper + ci) * xs[2] + ix] * w[(oc * cper + ci) * ws[2] + k];
        }
    } else if (op == 1) {
        unsigned long long ox = index % os[3];
        unsigned long long oy = (index / os[3]) % os[2];
        unsigned long long oc = (index / (os[3] * os[2])) % os[1];
        unsigned long long n = index / (os[1] * os[2] * os[3]);
        unsigned long long cper = ws[1];
        unsigned long long group = oc / (ws[0] / groups);
        for (unsigned long long ci = 0; ci < cper; ++ci) for (unsigned long long ky = 0; ky < ws[2]; ++ky) for (unsigned long long kx = 0; kx < ws[3]; ++kx) {
            long long iy = (long long)(oy * stride + ky * dilation) - padding;
            long long ix = (long long)(ox * stride + kx * dilation) - padding;
            if (iy >= 0 && iy < xs[2] && ix >= 0 && ix < xs[3]) {
                unsigned long long xi = ((n * xs[1] + group * cper + ci) * xs[2] + iy) * xs[3] + ix;
                unsigned long long wi = ((oc * cper + ci) * ws[2] + ky) * ws[3] + kx;
                total += x[xi] * w[wi];
            }
        }
    } else if (op == 2) {
        unsigned long long ox = index % os[2];
        unsigned long long oc = (index / os[2]) % os[1];
        unsigned long long n = index / (os[1] * os[2]);
        unsigned long long out_per_group = ws[1];
        unsigned long long group = oc / out_per_group;
        for (unsigned long long ic = group * (xs[1] / groups); ic < (group + 1) * (xs[1] / groups); ++ic) for (unsigned long long k = 0; k < ws[2]; ++k) {
            long long numerator = (long long)ox + padding - (long long)(k * dilation);
            if (numerator >= 0 && numerator % stride == 0) {
                unsigned long long ix = numerator / stride;
                if (ix < xs[2]) total += x[(n * xs[1] + ic) * xs[2] + ix] * w[(ic * out_per_group + oc % out_per_group) * ws[2] + k];
            }
        }
    } else if (op == 3) {
        unsigned long long ox = index % os[3];
        unsigned long long oy = (index / os[3]) % os[2];
        unsigned long long oc = (index / (os[3] * os[2])) % os[1];
        unsigned long long n = index / (os[1] * os[2] * os[3]);
        unsigned long long out_per_group = ws[1];
        unsigned long long group = oc / out_per_group;
        for (unsigned long long ic = group * (xs[1] / groups); ic < (group + 1) * (xs[1] / groups); ++ic) for (unsigned long long ky = 0; ky < ws[2]; ++ky) for (unsigned long long kx = 0; kx < ws[3]; ++kx) {
            long long ny = (long long)oy + padding - (long long)(ky * dilation);
            long long nx = (long long)ox + padding - (long long)(kx * dilation);
            if (ny >= 0 && nx >= 0 && ny % stride == 0 && nx % stride == 0) {
                unsigned long long iy = ny / stride;
                unsigned long long ix = nx / stride;
                if (iy < xs[2] && ix < xs[3]) {
                    unsigned long long xi = ((n * xs[1] + ic) * xs[2] + iy) * xs[3] + ix;
                    unsigned long long wi = ((ic * out_per_group + oc % out_per_group) * ws[2] + ky) * ws[3] + kx;
                    total += x[xi] * w[wi];
                }
            }
        }
    } else if (op == 4) {
        unsigned long long k = index % os[2];
        unsigned long long ci = (index / os[2]) % os[1];
        unsigned long long oc = index / (os[1] * os[2]);
        unsigned long long group = oc / (os[0] / groups);
        for (unsigned long long n = 0; n < xs[0]; ++n) for (unsigned long long ox = 0; ox < ws[2]; ++ox) {
            long long ix = (long long)(ox * stride + k * dilation) - padding;
            if (ix >= 0 && ix < xs[2]) total += x[(n * xs[1] + group * os[1] + ci) * xs[2] + ix] * w[(n * ws[1] + oc) * ws[2] + ox];
        }
    } else {
        unsigned long long kx = index % os[3];
        unsigned long long ky = (index / os[3]) % os[2];
        unsigned long long ci = (index / (os[3] * os[2])) % os[1];
        unsigned long long oc = index / (os[1] * os[2] * os[3]);
        unsigned long long group = oc / (os[0] / groups);
        for (unsigned long long n = 0; n < xs[0]; ++n) for (unsigned long long oy = 0; oy < ws[2]; ++oy) for (unsigned long long ox = 0; ox < ws[3]; ++ox) {
            long long iy = (long long)(oy * stride + ky * dilation) - padding;
            long long ix = (long long)(ox * stride + kx * dilation) - padding;
            if (iy >= 0 && iy < xs[2] && ix >= 0 && ix < xs[3]) {
                unsigned long long xi = ((n * xs[1] + group * os[1] + ci) * xs[2] + iy) * xs[3] + ix;
                unsigned long long gi = ((n * ws[1] + oc) * ws[2] + oy) * ws[3] + ox;
                total += x[xi] * w[gi];
            }
        }
    }
    out[index] = cast_dtype(total, dtype);
}

extern "C" __global__ void linalg_f64(
    unsigned int op,
    const double *a,
    const double *b,
    double *out,
    double *workspace,
    unsigned int batches,
    unsigned int n,
    unsigned int rhs,
    unsigned int dtype
) {
    unsigned int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batches) return;
    unsigned long long matrix_size = (unsigned long long)n * n;
    unsigned long long right_size = op == 0 ? matrix_size : op == 2 ? (unsigned long long)n * rhs : 0;
    double *matrix = workspace + batch * (matrix_size + right_size);
    double *right = matrix + matrix_size;
    for (unsigned long long i = 0; i < matrix_size; ++i) matrix[i] = a[batch * matrix_size + i];
    if (op == 0) {
        for (unsigned int row = 0; row < n; ++row) for (unsigned int column = 0; column < n; ++column) right[row * n + column] = row == column ? 1.0 : 0.0;
    } else if (op == 2) {
        for (unsigned long long i = 0; i < right_size; ++i) right[i] = b[batch * right_size + i];
    }
    double determinant = 1.0;
    int sign = 1;
    for (unsigned int pivot = 0; pivot < n; ++pivot) {
        unsigned int selected = pivot;
        for (unsigned int row = pivot + 1; row < n; ++row) if (fabs(matrix[row * n + pivot]) > fabs(matrix[selected * n + pivot])) selected = row;
        if (matrix[selected * n + pivot] == 0.0) {
            if (op == 1) out[batch] = 0.0;
            else for (unsigned long long i = 0; i < right_size; ++i) out[batch * right_size + i] = 0.0 / 0.0;
            return;
        }
        if (selected != pivot) {
            for (unsigned int column = 0; column < n; ++column) { double temporary = matrix[pivot * n + column]; matrix[pivot * n + column] = matrix[selected * n + column]; matrix[selected * n + column] = temporary; }
            if (op != 1) for (unsigned int column = 0; column < (op == 0 ? n : rhs); ++column) { double temporary = right[pivot * (op == 0 ? n : rhs) + column]; right[pivot * (op == 0 ? n : rhs) + column] = right[selected * (op == 0 ? n : rhs) + column]; right[selected * (op == 0 ? n : rhs) + column] = temporary; }
            sign = -sign;
        }
        double diagonal = matrix[pivot * n + pivot];
        determinant *= diagonal;
        if (op != 1) {
            for (unsigned int column = pivot; column < n; ++column) matrix[pivot * n + column] /= diagonal;
            for (unsigned int column = 0; column < (op == 0 ? n : rhs); ++column) right[pivot * (op == 0 ? n : rhs) + column] /= diagonal;
        }
        for (unsigned int row = op == 1 ? pivot + 1 : 0; row < n; ++row) {
            if (row == pivot) continue;
            double factor = matrix[row * n + pivot];
            if (op == 1) factor /= diagonal;
            for (unsigned int column = pivot; column < n; ++column) matrix[row * n + column] -= factor * matrix[pivot * n + column];
            if (op != 1) for (unsigned int column = 0; column < (op == 0 ? n : rhs); ++column) right[row * (op == 0 ? n : rhs) + column] -= factor * right[pivot * (op == 0 ? n : rhs) + column];
        }
    }
    if (op == 1) out[batch] = cast_dtype(determinant * sign, dtype);
    else for (unsigned long long i = 0; i < right_size; ++i) out[batch * right_size + i] = cast_dtype(right[i], dtype);
}

extern "C" __global__ void linear_f64(
    const double *x,
    const double *weight,
    const double *bias,
    double *out,
    unsigned int len,
    unsigned int k_width,
    unsigned int n_width,
    unsigned int dtype
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    unsigned int row = index / n_width;
    unsigned int column = index % n_width;
    double value = bias[column];
    for (unsigned int k = 0; k < k_width; ++k) value += x[row * k_width + k] * weight[k * n_width + column];
    out[index] = cast_dtype(value, dtype);
}

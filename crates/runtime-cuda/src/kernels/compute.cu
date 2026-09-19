// Appended to each fixed-storage compute module. typed.cuh precedes the F32
// macro prelude, keeping descriptor scalars and independently typed roles intact.
#define ET_INPUT(n) ((const double *)a.inputs[n])
#define ET_OUTPUT ((double *)a.output)

#ifdef ET_TENSOR
extern "C" __global__ void et_reduce(CudaKernelArgs a) {
    unsigned int rank = et_meta(a)[1]; et_u64 metadata[128];
    if (rank > 64) { et_error(a, 4); return; }
    for (unsigned int d = 0; d < rank; ++d) { metadata[d] = et_shape(a, 0)[d]; metadata[rank + d] = 0; }
    for (et_u64 d = 0; d < a.integers[0]; ++d) metadata[rank + et_tail(a)[d]] = 1;
    reduce_f64(a.operation, ET_INPUT(0), ET_OUTPUT, a.elements, rank, metadata, a.compute_dtype);
}
extern "C" __global__ void et_matmul(CudaKernelArgs a) {
    unsigned int rank = et_meta(a)[0]; et_u64 shapes[192];
    if (rank > 64) { et_error(a, 4); return; }
    for (unsigned int d = 0; d < rank; ++d) shapes[d] = et_shape(a)[d];
    for (int role = 0; role < 2; ++role) {
        unsigned int irank = et_meta(a)[role + 1];
        for (unsigned int d = 0; d < rank; ++d) shapes[(role + 1) * rank + d] = d + irank < rank ? 1 : et_shape(a, role)[d + irank - rank];
    }
    matmul_f64(ET_INPUT(0), ET_INPUT(1), ET_OUTPUT, a.elements, rank, shapes, a.compute_dtype);
}
extern "C" __global__ void et_rms_norm(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    et_u64 width = et_shape(a, 0)[et_meta(a)[1] - 1], base = i / width * width;
    double sum = 0; for (et_u64 d = 0; d < width; ++d) { double v = ET_INPUT(0)[base + d]; sum += v * v; }
    double value = ET_INPUT(0)[i] / sqrt(sum / width + (double)a.scalars[0]);
    if (a.inputs[1]) value *= ET_INPUT(1)[i % width];
    ET_OUTPUT[i] = value;
}
__device__ unsigned int et_ce_active(const CudaKernelArgs &a, int role, et_u64 rows, et_u64 classes) {
    unsigned int active = 0;
    for (et_u64 row = 0; row < rows; ++row) {
        et_i64 target = et_load<et_i64>(a.inputs[role], a.input_dtypes[role], row);
        if (target == (et_i64)a.integers[0]) continue;
        if (target < 0 || (et_u64)target >= classes) { et_error(a, 1); return 0; }
        ++active;
    }
    if (!active) et_error(a, 2);
    return active;
}
extern "C" __global__ void et_cross_entropy(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    et_u64 classes = et_shape(a, 0)[et_meta(a)[1] - 1], rows = et_numel(a, 0) / classes;
    unsigned int active = et_ce_active(a, 1, rows, classes); if (!active) return;
    double total = 0;
    et_u64 begin = a.operation ? i / classes : 0, end = a.operation ? begin + 1 : rows;
    for (et_u64 row = begin; row < end; ++row) {
        et_i64 selected = et_load<et_i64>(a.inputs[1], a.input_dtypes[1], row);
        if (selected == (et_i64)a.integers[0]) continue;
        double maximum = -1.0 / 0.0, sum = 0;
        for (et_u64 c = 0; c < classes; ++c) maximum = fmax(maximum, ET_INPUT(0)[row * classes + c]);
        for (et_u64 c = 0; c < classes; ++c) sum += exp(ET_INPUT(0)[row * classes + c] - maximum);
        total += a.operation ? exp(ET_INPUT(0)[i] - maximum) / sum - (double)(i % classes == (et_u64)selected)
            : maximum + log(sum) - ET_INPUT(0)[row * classes + selected];
    }
    ET_OUTPUT[i] = total / active;
}
extern "C" __global__ void et_chunked_head_ce(CudaKernelArgs a) {
    if (et_thread() >= a.elements) return;
    unsigned int inner = et_shape(a, 0)[et_meta(a)[1] - 1], vocab = et_shape(a, 1)[1];
    unsigned int rows = et_numel(a, 0) / inner, active = et_ce_active(a, 3, rows, vocab); if (!active) return;
    if (!a.operation) chunked_head_ce_f64(ET_INPUT(0), ET_INPUT(1), ET_INPUT(2), (const void *)a.inputs[3], a.input_dtypes[3], ET_OUTPUT,
        rows, inner, vocab, active, (et_i64)a.integers[0], a.compute_dtype);
    else chunked_head_ce_backward_f64(a.operation - 1, ET_INPUT(0), ET_INPUT(1), ET_INPUT(2), (const void *)a.inputs[3], a.input_dtypes[3], ET_INPUT(4), ET_OUTPUT,
        a.elements, rows, inner, vocab, active, (et_i64)a.integers[0], a.compute_dtype);
}
#endif

#ifdef ET_LINALG
extern "C" __global__ void et_conv(CudaKernelArgs a) {
    et_u64 shapes[12];
    for (unsigned int d = 0; d < 4; ++d) {
        shapes[d] = d < et_meta(a)[1] ? et_shape(a, 0)[d] : 1;
        shapes[4 + d] = d < et_meta(a)[2] ? et_shape(a, 1)[d] : 1;
        shapes[8 + d] = d < et_meta(a)[0] ? et_shape(a)[d] : 1;
    }
    conv_f64(a.operation, ET_INPUT(0), ET_INPUT(1), ET_OUTPUT, a.elements, shapes,
        a.integers[0], a.integers[1], a.integers[2], a.integers[3], a.compute_dtype);
}
extern "C" __global__ void et_linalg(CudaKernelArgs a) {
    unsigned int rank = et_meta(a)[1], n = et_shape(a, 0)[rank - 1];
    unsigned int batches = et_numel(a, 0) / ((et_u64)n * n), rhs = 0;
    if (a.operation == 2) rhs = et_numel(a, 1) / ((et_u64)batches * n);
    linalg_f64(a.operation, ET_INPUT(0), ET_INPUT(1), ET_OUTPUT, (double *)a.scratch[0], (unsigned int *)a.scratch[3], batches, n, rhs, a.compute_dtype);
}
extern "C" __global__ void et_linear(CudaKernelArgs a) {
    linear_f64(ET_INPUT(0), ET_INPUT(1), ET_INPUT(2), ET_OUTPUT, a.elements, a.integers[0], a.integers[1], a.compute_dtype);
}
extern "C" __global__ void et_linear_bias(CudaKernelArgs a) {
    linear_bias_f64((const void *)a.inputs[0], (const void *)a.inputs[1], a.input_dtypes[1],
        (void *)a.output, a.output_dtype, a.input_dtypes[0], a.elements, (unsigned int)a.integers[0]);
}
#endif

#ifdef ET_NEURAL
extern "C" __global__ void et_layer_norm(CudaKernelArgs a) {
    layer_norm_f64(a.operation, ET_INPUT(0), ET_INPUT(1), ET_INPUT(2), ET_OUTPUT, a.elements,
        a.integers[0], a.integers[1], a.scalars[0], a.compute_dtype);
}
extern "C" __global__ void et_sdpa(CudaKernelArgs a) {
    unsigned int rank = et_meta(a)[1]; et_u64 shapes[192];
    if (rank > 64) { et_error(a, 4); return; }
    for (int role = 0; role < 3; ++role) for (unsigned int d = 0; d < rank; ++d) shapes[role * rank + d] = et_shape(a, role)[d];
    sdpa_f64(a.operation, ET_INPUT(0), ET_INPUT(1), ET_INPUT(2), ET_INPUT(3), ET_OUTPUT, a.elements,
        rank, shapes, a.scalars[0], a.integers[0], a.integers[1], a.compute_dtype);
}
extern "C" __global__ void et_rotary(CudaKernelArgs a) {
    unsigned int rank = et_meta(a)[1], width = et_shape(a, 0)[rank - 1];
    // The first leading axis is the sequence lane. Remaining leading axes are groups.
    unsigned int groups = 1; for (unsigned int d = 1; d + 2 < rank; ++d) groups *= et_shape(a, 0)[d];
    rotary_f64(ET_INPUT(0), ET_OUTPUT, a.elements, width, a.integers[0], (const unsigned int *)a.scratch[0], groups,
        a.scalars[0], a.integers[1], a.integers[2], a.compute_dtype);
}
#endif

#ifdef ET_STATEFUL
extern "C" __global__ void et_short_conv(CudaKernelArgs a) {
    unsigned int rank = et_meta(a)[1]; const et_u64 *s = et_shape(a, 0);
    unsigned int channels = s[rank - 1], time = s[rank - 2], outer = et_numel(a, 0) / ((et_u64)time * channels);
    unsigned int kernel = et_shape(a, 1)[et_meta(a)[2] - 1];
    short_conv_f64(a.operation, ET_INPUT(0), ET_INPUT(1), ET_INPUT(2), ET_OUTPUT, a.elements, outer, time, channels, kernel,
        (const float *)a.scratch[0], (float *)a.scratch[1], (const unsigned int *)a.scratch[2], a.integers[0], a.compute_dtype);
}
extern "C" __global__ void et_kda(CudaKernelArgs a) {
    unsigned int rank = et_meta(a)[1]; const et_u64 *s = et_shape(a, 0);
    unsigned int dk = s[rank - 1], time = s[rank - 2], outer = et_numel(a, 0) / ((et_u64)time * dk);
    unsigned int dv = et_shape(a, 2)[et_meta(a)[3] - 1], heads = rank >= 3 ? s[rank - 3] : 1;
    if (!a.integers[0]) {
        if (a.integers[1]) kda_forward_f64(ET_INPUT(0), ET_INPUT(1), ET_INPUT(2), ET_INPUT(3), ET_INPUT(4), ET_OUTPUT,
            (float *)a.scratch[0], outer, time, dk, dv, a.scalars[0], (const unsigned int *)a.scratch[2], heads, 1, a.compute_dtype);
        else kda_forward_f64(ET_INPUT(0), ET_INPUT(1), ET_INPUT(2), ET_INPUT(3), ET_INPUT(4), ET_OUTPUT,
            (double *)a.scratch[0], outer, time, dk, dv, a.scalars[0], (const unsigned int *)a.scratch[2], heads, 0, a.compute_dtype);
    } else kda_backward_f64(a.operation, ET_INPUT(0), ET_INPUT(1), ET_INPUT(2), ET_INPUT(3), ET_INPUT(4), ET_INPUT(5), ET_OUTPUT,
        (double *)a.scratch[0], (double *)a.scratch[1], outer, time, dk, dv, a.scalars[0], a.compute_dtype);
}
#endif

#ifdef ET_POINTWISE
extern "C" __global__ void et_random(CudaKernelArgs a) {
    random_f64(ET_OUTPUT, a.elements, a.integers[0], a.integers[1], a.scalars[0], a.scalars[1], a.compute_dtype);
}
#endif

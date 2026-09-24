__device__ void et_convert_lane(et_u64 input, unsigned int source, et_u64 si, et_u64 output, unsigned int destination, et_u64 di) {
    if (source == destination) { et_copy(input, output, source, si, di); return; }
    switch (source) {
        case 0: et_store(output, destination, di, et_load<double>(input, source, si)); break;
        case 4: et_store(output, destination, di, et_load<et_i64>(input, source, si)); break;
        case 5: et_store(output, destination, di, et_load<unsigned int>(input, source, si)); break;
        case 6: et_store(output, destination, di, et_load<unsigned char>(input, source, si)); break;
        default: et_store(output, destination, di, et_load<float>(input, source, si)); break;
    }
}
extern "C" __global__ void et_convert(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    et_convert_lane(a.inputs[0], a.input_dtypes[0], i, a.output, a.output_dtype, i);
}
__device__ unsigned int et_top_k_order(float value) {
    unsigned int bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) == 0) bits = 0; // Signed zeros tie.
    return (bits & 0x80000000U) ? ~bits : bits ^ 0x80000000U;
}
// Stable O(width*k) insertion. The planned output is also the workspace.
// No sampled tokens, host scores, or dynamically allocated device storage.
extern "C" __global__ void et_top_k_indices(CudaKernelArgs a) {
    if (threadIdx.x != 0) return;
    et_u64 row = blockIdx.x, k = a.integers[0], width = a.integers[1];
    if (row >= a.elements / k) return;
    const float *x = (const float *)a.inputs[0] + row * width;
    unsigned int *out = (unsigned int *)a.output + row * k;
    for (et_u64 i = 0; i < width; ++i) {
        float value = x[i];
        if ((__float_as_uint(value) & 0x7fffffffU) > 0x7f800000U) { et_error(a, 5); return; }
        unsigned int order = et_top_k_order(value);
        et_u64 used = i < k ? i : k, position = 0;
        while (position < used && order <= et_top_k_order(x[out[position]])) ++position;
        if (position < k) {
            for (et_u64 j = used < k ? used : k - 1; j > position; --j) out[j] = out[j - 1];
            out[position] = (unsigned int)i;
        }
    }
}
// One warp cooperates on a dot, with adjacent lanes reading adjacent weights.
// Only native F32/BF16 elements of the selected expert are loaded.
extern "C" __global__ void et_expert_linear_rows(CudaKernelArgs a) {
    const et_u64 rows = a.integers[0], columns = a.integers[1], inner = a.integers[2], experts = a.integers[3];
    const et_u64 width = columns ? columns : 1, work = rows * width;
    const unsigned int lane = threadIdx.x & 31U;
    const et_u64 stride = (et_u64)gridDim.x * (blockDim.x / 32);
    for (et_u64 out = et_thread() / 32; out < work; out += stride) {
        et_u64 row = out / width, column = out % width;
        unsigned int expert = lane == 0 ? ((const unsigned int *)a.inputs[2])[row] : 0;
        expert = __shfl_sync(0xffffffffU, expert, 0);
        if ((et_u64)expert >= experts) {
            if (lane == 0) {
                et_error(a, 6);
                if (columns) et_store(a.output, a.output_dtype, out, 0.0f);
            }
            continue;
        }
        if (!columns) continue;
        float sum = 0.0f;
        const et_u64 base = ((et_u64)expert * columns + column) * inner;
        for (et_u64 i = lane; i < inner; i += 32) {
            sum = fmaf(et_load<float>(a.inputs[0], a.input_dtypes[0], row * inner + i),
                et_load<float>(a.inputs[1], a.input_dtypes[1], base + i), sum);
        }
        for (unsigned int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffffU, sum, offset);
        if (lane == 0) et_store(a.output, a.output_dtype, out, sum);
    }
}
// Control is [error, offsets[E+1]], initially zero. During counting the
// count for expert e occupies slot e+2. No payload ever crosses the host.
extern "C" __global__ void et_grouped_counts(CudaKernelArgs a) {
    unsigned int *control = (unsigned int *)a.output;
    for (et_u64 row = et_thread(); row < a.elements; row += (et_u64)gridDim.x * blockDim.x) {
        unsigned int expert = ((const unsigned int *)a.inputs[0])[row];
        if ((et_u64)expert >= a.integers[0]) atomicCAS(control, 0U, 6U);
        else atomicAdd(control + (et_u64)expert + 2, 1U);
    }
}
extern "C" __global__ void et_grouped_offsets(CudaKernelArgs a) {
    if (et_thread()) return;
    unsigned int *control = (unsigned int *)a.output;
    unsigned int total = 0;
    for (et_u64 e = 0; e < a.integers[0]; ++e) {
        unsigned int count = control[e + 2];
        control[e + 1] = total; total += count;
    }
    control[a.integers[0] + 1] = total;
}
// One warp per expert, scanning input rows in ascending order. Ballot prefix
// ranks make the permutation stable independently of warp/block scheduling.
extern "C" __global__ void et_grouped_rows(CudaKernelArgs a) {
    const unsigned int lane = threadIdx.x & 31U;
    for (et_u64 e = et_thread() / 32; e < a.integers[0]; e += (et_u64)gridDim.x * (blockDim.x / 32)) {
        unsigned int offset = ((const unsigned int *)a.inputs[1])[e + 1];
        for (et_u64 base = 0; base < a.elements; base += 32) {
            et_u64 row = base + lane;
            bool selected = row < a.elements && ((const unsigned int *)a.inputs[0])[row] == e;
            unsigned int mask = __ballot_sync(0xffffffffU, selected);
            if (selected) ((unsigned int *)a.output)[offset + __popc(mask & ((1U << lane) - 1U))] = (unsigned int)row;
            offset += __popc(mask);
        }
    }
}
extern "C" __global__ void et_grouped_gather(CudaKernelArgs a) {
    et_u64 width = a.integers[0], rows = a.elements / width;
    for (et_u64 destination = blockIdx.x; destination < rows; destination += gridDim.x) {
        et_u64 source = ((const unsigned int *)a.inputs[1])[destination] % a.integers[1];
        for (et_u64 column = threadIdx.x; column < width; column += blockDim.x)
            et_copy(a.inputs[0], a.output, a.output_dtype, source * width + column, destination * width + column);
    }
}
extern "C" __global__ void et_grouped_scatter(CudaKernelArgs a) {
    et_u64 width = a.integers[0], rows = a.elements / width;
    for (et_u64 source = blockIdx.x; source < rows; source += gridDim.x) {
        et_u64 destination = ((const unsigned int *)a.inputs[1])[source];
        for (et_u64 column = threadIdx.x; column < width; column += blockDim.x)
            et_copy(a.inputs[0], a.output, a.output_dtype, source * width + column, destination * width + column);
    }
}
// Same sequential F32 multiply/add order as et_matmul_f32, with row-oriented
// weights. Both modules compile with fmad=false.
// This is the ordinary CUDA F32 matrix contract, not the strict warp expert dot.
extern "C" __global__ void et_grouped_matmul_f32(CudaKernelArgs a) {
    for (et_u64 i = et_thread(); i < a.elements; i += (et_u64)gridDim.x * blockDim.x) {
        et_u64 columns = a.integers[0], inner = a.integers[1];
        et_u64 row = i / columns, column = i % columns;
        float sum = 0.0f;
        for (et_u64 k = 0; k < inner; ++k)
            sum += ((const float *)a.inputs[0])[row * inner + k] * ((const float *)a.inputs[1])[column * inner + k];
        ((float *)a.output)[i] = sum;
    }
}
extern "C" __global__ void et_fill(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i < a.elements) et_store(a.output, a.output_dtype, i, a.scalars[0]);
}
template<class T> __device__ T et_add(T x, T y) { return x + y; }
template<> __device__ et_i64 et_add(et_i64 x, et_i64 y) { return (et_i64)((et_u64)x + (et_u64)y); }
template<class T> __device__ T et_sub(T x, T y) { return x - y; }
template<> __device__ et_i64 et_sub(et_i64 x, et_i64 y) { return (et_i64)((et_u64)x - (et_u64)y); }
template<class T> __device__ T et_mul(T x, T y) { return x * y; }
template<> __device__ et_i64 et_mul(et_i64 x, et_i64 y) { return (et_i64)((et_u64)x * (et_u64)y); }
template<class T> __device__ T et_div(T x, T y, const CudaKernelArgs &) { return x / y; }
template<> __device__ et_i64 et_div(et_i64 x, et_i64 y, const CudaKernelArgs &a) {
    if (y == 0) { et_error(a, 4); return 0; }
    if (x == (-0x7fffffffffffffffLL - 1) && y == -1) return x;
    return x / y;
}
template<class T> __device__ T et_max(T x, T y) { return x > y ? x : y; }
template<class T> __device__ T et_min(T x, T y) { return x < y ? x : y; }
template<> __device__ float et_max(float x, float y) { return fmaxf(x, y); }
template<> __device__ float et_min(float x, float y) { return fminf(x, y); }
template<> __device__ double et_max(double x, double y) { return fmax(x, y); }
template<> __device__ double et_min(double x, double y) { return fmin(x, y); }
template<class T> __device__ T et_binary_operand(const CudaKernelArgs &a, et_u64 i, int role) {
    et_u64 source_index = et_broadcast(a, i, role);
    if (!a.integers[role]) return et_load<T>(a.inputs[role], a.input_dtypes[role], source_index);
    // Optional planned semantic scalar coercion precedes arithmetic promotion.
    if (a.integers[role] > 7 || et_meta(a)[role + 1] != 0) { et_error(a, 4); return 0; }
    unsigned int target = a.integers[role] - 1;
    et_u64 rounded = 0;
    et_convert_lane(a.inputs[role], a.input_dtypes[role], source_index, (et_u64)&rounded, target, 0);
    return et_load<T>((et_u64)&rounded, target, 0);
}
template<class T> __device__ void et_binary_impl(const CudaKernelArgs &a, et_u64 i) {
    T x = et_binary_operand<T>(a, i, 0);
    T y = et_binary_operand<T>(a, i, 1), z = 0;
    switch (a.operation) {
        case 0: z = et_add(x, y); break; case 1: z = et_sub(x, y); break;
        case 2: z = et_mul(x, y); break; case 3: z = et_div(x, y, a); break;
        case 4: z = et_max(x, y); break; case 5: z = et_min(x, y); break;
        case 6: z = x == y; break; case 7: z = x > y; break; case 8: z = x < y; break;
        case 9: z = x >= y; break; case 10: z = x <= y; break;
    }
    et_store(a.output, a.output_dtype, i, z);
}
extern "C" __global__ void et_binary(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    if (a.compute_dtype >= 4) et_binary_impl<et_i64>(a, i);
    else if (a.compute_dtype == 0) et_binary_impl<double>(a, i);
    else et_binary_impl<float>(a, i);
}
template<class T> __device__ void et_unary_float(const CudaKernelArgs &a, et_u64 i) {
    T x = et_load<T>(a.inputs[0], a.input_dtypes[0], i), z = x, p = a.scalars[0];
    switch (a.operation) {
        case 0: z = -x; break; case 1: z = fabs(x); break; case 2: z = sqrt(x); break;
        case 3: z = exp(x); break; case 4: z = log(x); break; case 5: z = sin(x); break;
        case 6: z = cos(x); break; case 7: z = tanh(x); break; case 8: z = et_max(x, T(0)); break;
        case 9: z = erf(x); break; case 10: z = floor(x); break; case 11: z = ceil(x); break;
        case 12: z = round(x); break; case 13: z = (x > 0) - (x < 0); break;
        case 14: z = pow(x, p); break;
        case 15: z = T(0.5) * x * (T(1) + erf(x * T(0.7071067811865475244))); break;
        case 16: z = T(0.5) * x * (T(1) + tanh(T(0.7978845608028653559) * (x + T(0.044715) * x * x * x))); break;
    }
    et_store(a.output, a.output_dtype, i, z);
}
extern "C" __global__ void et_unary(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    if (a.input_dtypes[0] >= 4) {
        et_i64 x = et_load<et_i64>(a.inputs[0], a.input_dtypes[0], i), z = x;
        switch (a.operation) {
            case 0: z = et_sub<et_i64>(0, x); break;
            case 1: z = x < 0 ? et_sub<et_i64>(0, x) : x; break;
            case 8: z = x > 0 ? x : 0; break; case 13: z = (x > 0) - (x < 0); break;
        }
        et_store(a.output, a.output_dtype, i, z);
    } else if (a.input_dtypes[0] == 0) et_unary_float<double>(a, i);
    else et_unary_float<float>(a, i);
}
extern "C" __global__ void et_where(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    // Conditions are U8 masks, independent of the selected values' dtype.
    bool c = et_load<unsigned char>(a.inputs[0], a.input_dtypes[0], et_broadcast(a, i, 0)) != 0;
    int role = c ? 1 : 2;
    et_copy(a.inputs[role], a.output, a.output_dtype, et_broadcast(a, i, role), i);
}
extern "C" __global__ void et_reindex(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    if (a.operation == 0) { et_copy(a.inputs[0], a.output, a.output_dtype, et_broadcast(a, i, 0), i); return; }
    const et_u64 *os = et_shape(a), *is = et_shape(a, 0), *p = et_tail(a);
    et_u64 linear = i, source = 0; int rank = et_meta(a)[0];
    for (int d = rank - 1; d >= 0; --d) {
        et_u64 c = linear % os[d]; linear /= os[d];
        et_u64 axis = a.operation == 1 ? p[d] : d;
        if (a.operation == 2) c = p[2 * d] + c * p[2 * d + 1];
        et_u64 stride = 1; for (et_u64 j = axis + 1; j < (et_u64)rank; ++j) stride *= is[j];
        source += c * stride;
    }
    et_copy(a.inputs[0], a.output, a.output_dtype, source, i);
}
extern "C" __global__ void et_concat(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    const et_u64 *s = et_shape(a), *ls = et_shape(a, 0), *rs = et_shape(a, 1);
    unsigned int dim = a.integers[0], rank = et_meta(a)[0];
    et_u64 inner = 1; for (unsigned int d = dim + 1; d < rank; ++d) inner *= s[d];
    et_u64 c = (i / inner) % s[dim], outer = i / (inner * s[dim]);
    int role = c < ls[dim] ? 0 : 1; et_u64 width = role ? rs[dim] : ls[dim];
    et_u64 source = (outer * width + (role ? c - ls[dim] : c)) * inner + i % inner;
    et_copy(a.inputs[role], a.output, a.output_dtype, source, i);
}
template<class T> __device__ void et_index_impl(const CudaKernelArgs &a, et_u64 i) {
    const et_u64 *s = et_shape(a, 0); unsigned int rank = et_meta(a)[1], dim = a.integers[0];
    et_u64 inner = 1; for (unsigned int d = dim + 1; d < rank; ++d) inner *= s[d];
    if (a.operation <= 1) {
        if (s[dim] == 0) { et_error(a, 4); return; }
        et_u64 base = (i / inner) * inner * s[dim] + i % inner, best_index = 0;
        T best = et_load<T>(a.inputs[0], a.input_dtypes[0], base);
        for (et_u64 c = 1; c < s[dim]; ++c) {
            T value = et_load<T>(a.inputs[0], a.input_dtypes[0], base + c * inner);
            if ((a.operation == 0 && value > best) || (a.operation == 1 && value < best)) { best = value; best_index = c; }
        }
        et_store(a.output, a.output_dtype, i, (et_i64)best_index);
    } else if (a.operation == 2) {
        et_u64 c = (i / inner) % s[dim], base = i - c * inner; T sum = 0;
        for (et_u64 j = 0; j <= c; ++j) sum = et_add(sum, et_load<T>(a.inputs[0], a.input_dtypes[0], base + j * inner));
        et_store(a.output, a.output_dtype, i, sum);
    } else if (a.operation == 3 || a.operation == 4) {
        const et_u64 *os = et_shape(a); et_u64 linear = i, base = 0, stride = 1, coord = 0;
        for (int d = rank - 1; d >= 0; --d) {
            et_u64 c = linear % os[d]; linear /= os[d];
            if ((unsigned int)d == dim) coord = c; else base += c * stride;
            stride *= s[d];
        }
        et_u64 selected; if (!et_selected(a, 1, a.operation == 3 ? coord : i, s[dim], &selected)) return;
        et_copy(a.inputs[0], a.output, a.output_dtype, base + selected * inner, i);
    } else {
        T total = et_load<T>(a.inputs[0], a.input_dtypes[0], i);
        const et_u64 *ss = et_shape(a, 2);
        // Public scatter geometry fixes every coordinate outside the scatter
        // axis. Only that source slice can contribute to this output. Visit it
        // in the original flattened source order, without atomic updates.
        bool same_axes = et_meta(a)[3] == rank;
        for (unsigned int d = 0; d < rank && same_axes; ++d)
            if (d != dim && ss[d] != s[d]) same_axes = false;
        et_u64 coordinate = (i / inner) % s[dim];
        et_u64 begin = same_axes ? (i / (inner * s[dim])) * inner * ss[dim] + i % inner : 0;
        et_u64 step = same_axes ? inner : 1;
        et_u64 end = same_axes ? begin + ss[dim] * inner : et_numel(a, 2);
        for (et_u64 j = begin; j < end; j += step) {
            et_u64 selected; if (!et_selected(a, 1, j, s[dim], &selected)) return;
            bool contributes = selected == coordinate;
            if (!same_axes) {
                // Retain native partial-shape indexing behavior as well.
                et_u64 linear = j, target = 0, stride = 1;
                for (int d = rank - 1; d >= 0; --d) {
                    et_u64 c = linear % ss[d]; linear /= ss[d];
                    target += ((unsigned int)d == dim ? selected : c) * stride; stride *= s[d];
                }
                contributes = target == i;
            }
            if (contributes) {
                total = et_add(total, et_load<T>(a.inputs[2], a.input_dtypes[2], j));
                // Native half scatter updates round each addition to storage,
                // just as the CPU's typed updates do. Do not silently widen
                // repeated updates into one F32 reduction.
                if (a.output_dtype == 2) total = (T)et_half_float(et_to16(total, false));
                else if (a.output_dtype == 3) total = (T)et_bfloat_float(et_to16(total, true));
            }
        }
        et_store(a.output, a.output_dtype, i, total);
    }
}
extern "C" __global__ void et_index(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    if (a.input_dtypes[0] >= 4) et_index_impl<et_i64>(a, i);
    else if (a.input_dtypes[0] == 0) et_index_impl<double>(a, i);
    else et_index_impl<float>(a, i);
}
template<class T> __device__ void et_scatter_add_inner_impl(
    const CudaKernelArgs &a, et_u64 row, et_u64 coordinate, et_u64 d, const et_i64 *indexes
) {
    et_u64 routes = a.integers[0], inner = a.integers[1];
    et_u64 i = (row * routes + coordinate) * inner + d;
    T total = et_load<T>(a.inputs[0], a.input_dtypes[0], i);
    for (et_u64 route = 0; route < routes; ++route) {
        if ((et_u64)indexes[route] != coordinate) continue;
        et_u64 source = (row * routes + route) * inner + d;
        total = et_add(total, et_load<T>(a.inputs[2], a.input_dtypes[2], source));
        if (a.output_dtype == 2) total = (T)et_half_float(et_to16(total, false));
        else if (a.output_dtype == 3) total = (T)et_bfloat_float(et_to16(total, true));
    }
    et_store(a.output, a.output_dtype, i, total);
}
extern "C" __global__ void et_scatter_add_inner(CudaKernelArgs a) {
    et_u64 routes = a.integers[0], inner = a.integers[1];
    et_u64 tiles = (inner + blockDim.x - 1) / blockDim.x;
    et_u64 tile = blockIdx.x % tiles, row = blockIdx.x / tiles;
    et_u64 d = tile * blockDim.x + threadIdx.x;
    __shared__ et_i64 indexes[32];
    __shared__ unsigned int valid;
    if (!threadIdx.x) valid = 1;
    __syncthreads();
    if (threadIdx.x < routes) {
        et_i64 selected = et_load<et_i64>(a.inputs[1], a.input_dtypes[1], row * routes + threadIdx.x);
        if (selected < 0 || (et_u64)selected >= routes) {
            et_error(a, 1);
            atomicExch(&valid, 0U);
        }
        indexes[threadIdx.x] = selected;
    }
    __syncthreads();
    if (!valid || d >= inner) return;
    for (et_u64 coordinate = 0; coordinate < routes; ++coordinate) {
        if (a.input_dtypes[0] >= 4) et_scatter_add_inner_impl<et_i64>(a, row, coordinate, d, indexes);
        else if (a.input_dtypes[0] == 0) et_scatter_add_inner_impl<double>(a, row, coordinate, d, indexes);
        else et_scatter_add_inner_impl<float>(a, row, coordinate, d, indexes);
    }
}
extern "C" __global__ __launch_bounds__(1024) void et_arg_index_last_wide(CudaKernelArgs a) {
    et_u64 width = a.integers[1];
    __shared__ float values[32];
    __shared__ unsigned int indexes[32];
    unsigned int lane = threadIdx.x & 31U, warp = threadIdx.x / 32;
    for (et_u64 row = blockIdx.x; row < a.elements; row += gridDim.x) {
        const float *input = (const float *)a.inputs[0] + row * width;
        float best = a.operation == 0 ? -1.0f / 0.0f : 1.0f / 0.0f;
        unsigned int best_index = 0xffffffffU;
        for (et_u64 column = threadIdx.x; column < width; column += 1024) {
            float value = input[column];
            bool better = !isnan(value) && (
                (a.operation == 0 && value > best) ||
                (a.operation == 1 && value < best) ||
                (value == best && column < best_index)
            );
            if (better) { best = value; best_index = (unsigned int)column; }
        }
        for (unsigned int offset = 16; offset; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffffU, best, offset);
            unsigned int other_index = __shfl_down_sync(0xffffffffU, best_index, offset);
            bool better = (a.operation == 0 && other > best) ||
                (a.operation == 1 && other < best) ||
                (other == best && other_index < best_index);
            if (better) { best = other; best_index = other_index; }
        }
        if (!lane) { values[warp] = best; indexes[warp] = best_index; }
        __syncthreads();
        if (!warp) {
            best = values[lane]; best_index = indexes[lane];
            for (unsigned int offset = 16; offset; offset >>= 1) {
                float other = __shfl_down_sync(0xffffffffU, best, offset);
                unsigned int other_index = __shfl_down_sync(0xffffffffU, best_index, offset);
                bool better = (a.operation == 0 && other > best) ||
                    (a.operation == 1 && other < best) ||
                    (other == best && other_index < best_index);
                if (better) { best = other; best_index = other_index; }
            }
            if (!lane) {
                if (isnan(input[0])) best_index = 0;
                et_store(a.output, a.output_dtype, row, (et_i64)best_index);
            }
        }
        __syncthreads();
    }
}
extern "C" __global__ void et_sequence(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    if (a.integers[0] == 1) {
        et_u64 width = et_shape(a)[et_meta(a)[0] - 1];
        et_store(a.output, a.output_dtype, i, (unsigned int)(i / width == i % width));
    } else if (a.output_dtype >= 4) {
        et_i64 start = et_int(a.scalars[0]), step = et_int(a.scalars[1]);
        et_store(a.output, a.output_dtype, i, et_add(start, et_mul(step, (et_i64)i)));
    } else if (a.output_dtype == 0) et_store(a.output, a.output_dtype, i, a.scalars[0] + a.scalars[1] * (double)i);
    else et_store(a.output, a.output_dtype, i, (float)a.scalars[0] + (float)a.scalars[1] * (float)i);
}
extern "C" __global__ void et_last_token(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    et_u64 valid = a.integers[1], tokens = a.integers[0];
    if (!valid || valid > tokens) et_store(a.output, a.output_dtype, i, 0U);
    else et_copy(a.inputs[0], a.output, a.output_dtype, (valid - 1) * a.elements + i, i);
}
template<class T> __device__ void et_optimizer_impl(const CudaKernelArgs &a, et_u64 i) {
    T p = et_load<T>(a.inputs[0], a.input_dtypes[0], i), g = et_load<T>(a.inputs[1], a.input_dtypes[1], i);
    T s = et_load<T>(a.inputs[2], a.input_dtypes[2], i), lr = et_load<T>(a.inputs[4], a.input_dtypes[4], 0);
    T b1 = a.scalars[0], b2 = a.scalars[1], eps = a.scalars[2], wd = a.scalars[3], damp = a.scalars[4], z;
    if (a.operation == 0) {
        T v = et_load<T>(a.inputs[3], a.input_dtypes[3], i);
        T c1 = et_load<T>(a.inputs[5], a.input_dtypes[5], 0), c2 = et_load<T>(a.inputs[6], a.input_dtypes[6], 0);
        T m = b1 * s + (T(1) - b1) * g; v = b2 * v + (T(1) - b2) * g * g;
        z = a.integers[0] == 1 ? m : a.integers[0] == 2 ? v : p - lr * (m / c1) / (sqrt(v / c2) + eps) - lr * wd * p;
    } else {
        bool first = et_load<unsigned char>(a.inputs[3], a.input_dtypes[3], 0) != 0;
        T adjusted = g + wd * p, velocity = first ? adjusted : b1 * s + (T(1) - damp) * adjusted;
        T update = a.integers[1] ? adjusted + b1 * velocity : velocity;
        z = a.integers[0] ? velocity : p - lr * update;
    }
    et_store(a.output, a.output_dtype, i, z);
}
extern "C" __global__ void et_optimizer(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    if (a.compute_dtype == 0) et_optimizer_impl<double>(a, i); else et_optimizer_impl<float>(a, i);
}

extern "C" __global__ void et_reduce_integer(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    const et_u64 *s = et_shape(a, 0), *dims = et_tail(a); unsigned int rank = et_meta(a)[1];
    et_u64 count = 1;
    for (et_u64 j = 0; j < a.integers[0]; ++j) count *= s[dims[j]];
    et_i64 result = a.operation == 1 ? 1 : 0;
    for (et_u64 r = 0; r < count; ++r) {
        et_u64 out = i, reduced = r, source = 0, stride = 1;
        for (int d = rank - 1; d >= 0; --d) {
            bool selected = false;
            for (et_u64 j = 0; j < a.integers[0]; ++j) if (dims[j] == (et_u64)d) selected = true;
            et_u64 c = selected ? reduced % s[d] : out % s[d];
            if (selected) reduced /= s[d]; else out /= s[d];
            source += c * stride; stride *= s[d];
        }
        et_i64 value = et_load<et_i64>(a.inputs[0], a.input_dtypes[0], source);
        if (a.operation == 0 || a.operation == 4) result = et_add(result, value);
        else if (a.operation == 1) result = et_mul(result, value);
        else if (a.operation == 2) result = r == 0 ? value : et_max(result, value);
        else result = r == 0 ? value : et_min(result, value);
    }
    if (a.operation == 4) result = et_div(result, (et_i64)count, a);
    et_store(a.output, a.output_dtype, i, result);
}

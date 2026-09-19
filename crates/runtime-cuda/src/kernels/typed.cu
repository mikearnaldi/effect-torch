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
        const et_u64 *ss = et_shape(a, 2); et_u64 count = et_numel(a, 2);
        for (et_u64 j = 0; j < count; ++j) {
            et_u64 selected; if (!et_selected(a, 1, j, s[dim], &selected)) return;
            et_u64 linear = j, target = 0, stride = 1;
            for (int d = rank - 1; d >= 0; --d) {
                et_u64 c = linear % ss[d]; linear /= ss[d];
                target += ((unsigned int)d == dim ? selected : c) * stride; stride *= s[d];
            }
            if (target == i) total = et_add(total, et_load<T>(a.inputs[2], a.input_dtypes[2], j));
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

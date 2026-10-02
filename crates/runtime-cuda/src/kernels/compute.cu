// Appended to each fixed-storage compute module. typed.cuh precedes the F32
// macro prelude, keeping descriptor scalars and independently typed roles intact.
#define ET_INPUT(n) ((const double *)a.inputs[n])
#define ET_OUTPUT ((double *)a.output)

#ifdef ET_TENSOR
#ifdef ET_COMPUTE_F32
// The narrow selector guarantees dense F32 [1,256], axes[0,1], scalar result.
// Preserve generic Mean's ascending sequential additions and final division.
extern "C" __global__ void et_mean256_f32(CudaKernelArgs a) {
    if (blockIdx.x || threadIdx.x) return;
    const float *input = (const float *)a.inputs[0];
    float value = 0.0f;
    #pragma unroll 1
    for (unsigned int i = 0; i < 256; ++i) value = value + input[i];
    ((float *)a.output)[0] = value / 256.0f;
}
__device__ et_u64 et_sum_source(const et_u64 *shape, unsigned int rank, et_u64 mask,
                               et_u64 out, et_u64 reduced) {
    et_u64 source = 0, stride = 1;
    for (int d = (int)rank - 1; d >= 0; --d) {
        bool selected = mask & (1ULL << d);
        et_u64 coordinate = selected ? reduced % shape[d] : out % shape[d];
        if (selected) reduced /= shape[d]; else out /= shape[d];
        source += coordinate * stride; stride *= shape[d];
    }
    return source;
}
// One CTA per row preserves both existing wide F32 reduction trees. Only
// the exponential array is materialized; the BF16 input is widened on load.
extern "C" __global__ __launch_bounds__(1024) void et_bf16_softmax_prepare(CudaKernelArgs a) {
    et_u64 width = a.integers[0], rows = a.integers[1];
    et_u64 complete = width - width % 4096;
    unsigned int lane = threadIdx.x & 31U, warp = threadIdx.x / 32;
    __shared__ float partials[32];
    __shared__ float maximum;
    for (et_u64 row = blockIdx.x; row < rows; row += gridDim.x) {
        float value = -1.0f / 0.0f;
        for (et_u64 column = threadIdx.x; column < width; column += 1024)
            value = fmaxf(value, et_load<float>(a.inputs[0], 3, row * width + column));
        for (unsigned int offset = 16; offset; offset >>= 1)
            value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
        if (!lane) partials[warp] = value;
        __syncthreads();
        if (!warp) {
            value = partials[lane];
            for (unsigned int offset = 16; offset; offset >>= 1)
                value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
            if (!lane) maximum = value;
        }
        __syncthreads();
        float sum = 0.0f;
        for (et_u64 r = threadIdx.x * 4; r < complete; r += 4096) {
            #pragma unroll
            for (unsigned int j = 0; j < 4; ++j) {
                et_u64 i = row * width + r + j;
                float shifted = et_load<float>(a.inputs[0], 3, i) - maximum;
                float exponential = expf(shifted);
                et_store(a.output, 1, i, exponential);
                sum += exponential;
            }
        }
        for (et_u64 r = complete + threadIdx.x; r < width; r += 1024) {
            et_u64 i = row * width + r;
            float shifted = et_load<float>(a.inputs[0], 3, i) - maximum;
            float exponential = expf(shifted);
            et_store(a.output, 1, i, exponential);
            sum += exponential;
        }
        for (unsigned int offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffU, sum, offset);
        if (!lane) partials[warp] = sum;
        __syncthreads();
        if (!warp) {
            sum = partials[lane];
            for (unsigned int offset = 16; offset; offset >>= 1)
                sum += __shfl_down_sync(0xffffffffU, sum, offset);
            if (!lane) et_store(a.output, 1, rows * width + row, sum);
        }
        __syncthreads();
    }
}
extern "C" __global__ void et_bf16_softmax_store(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    float numerator = et_load<float>(a.inputs[0], 1, i);
    float denominator = et_load<float>(a.inputs[0], 1, a.elements + i / a.integers[0]);
    et_store(a.output, 3, i, numerator / denominator);
}

extern "C" __global__ __launch_bounds__(1024) void et_sum_wide(CudaKernelArgs a) {
    unsigned int rank = et_meta(a)[1], lane = threadIdx.x & 31U;
    if (rank > 64) { et_error(a, 4); return; }
    const et_u64 *shape = et_shape(a, 0), *dims = et_tail(a);
    et_u64 mask = 0, count = a.integers[1];
    for (et_u64 i = 0; i < a.integers[0]; ++i) mask |= 1ULL << dims[i];
    __shared__ float partials[32];
    unsigned int warp = threadIdx.x / 32;
    et_u64 complete = count - count % 4096;
    for (et_u64 row = blockIdx.x; row < a.elements; row += gridDim.x) {
        const double *input = ET_INPUT(0) + row * count;
        float sum = 0.0f;
        for (et_u64 r = threadIdx.x * 4; r < complete; r += 4096) {
            #pragma unroll
            for (unsigned int j = 0; j < 4; ++j)
                sum += a.integers[2] ? input[r + j] : ET_INPUT(0)[et_sum_source(shape, rank, mask, row, r + j)];
        }
        for (et_u64 r = complete + threadIdx.x; r < count; r += 1024)
            sum += a.integers[2] ? input[r] : ET_INPUT(0)[et_sum_source(shape, rank, mask, row, r)];
        for (unsigned int offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffU, sum, offset);
        if (!lane) partials[warp] = sum;
        __syncthreads();
        if (!warp) {
            sum = partials[lane];
            for (unsigned int offset = 16; offset; offset >>= 1)
                sum += __shfl_down_sync(0xffffffffU, sum, offset);
            if (!lane) ET_OUTPUT[row] = sum;
        }
        __syncthreads();
    }
}
extern "C" __global__ __launch_bounds__(1024) void et_reduce_last_wide(CudaKernelArgs a) {
    et_u64 count = a.integers[1];
    __shared__ float partials[32];
    unsigned int lane = threadIdx.x & 31U, warp = threadIdx.x / 32;
    for (et_u64 row = blockIdx.x; row < a.elements; row += gridDim.x) {
        const double *input = ET_INPUT(0) + row * count;
        float value = a.operation == 2 ? -1.0f / 0.0f : 1.0f / 0.0f;
        for (et_u64 r = threadIdx.x; r < count; r += 1024)
            value = a.operation == 2 ? fmaxf(value, input[r]) : fminf(value, input[r]);
        for (unsigned int offset = 16; offset; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffffU, value, offset);
            value = a.operation == 2 ? fmaxf(value, other) : fminf(value, other);
        }
        if (!lane) partials[warp] = value;
        __syncthreads();
        if (!warp) {
            value = partials[lane];
            for (unsigned int offset = 16; offset; offset >>= 1) {
                float other = __shfl_down_sync(0xffffffffU, value, offset);
                value = a.operation == 2 ? fmaxf(value, other) : fminf(value, other);
            }
            if (!lane) ET_OUTPUT[row] = value;
        }
        __syncthreads();
    }
}
#endif
extern "C" __global__ void et_reduce(CudaKernelArgs a) {
#ifdef ET_COMPUTE_F32
    if (a.operation == 2 && a.integers[7]) {
        // Last-axis Max: one warp per row, retaining the generic kernel's
        // -infinity identity and fmaxf NaN behavior. Revisit zero maxima in
        // original input order so signed-zero ties retain identical bits.
        unsigned int lane = threadIdx.x & 31U;
        et_u64 count = a.integers[1];
        for (et_u64 row = et_thread() / 32; row < a.elements;
             row += (et_u64)gridDim.x * (blockDim.x / 32)) {
            const double *input = ET_INPUT(0) + row * count;
            float value = -1.0f / 0.0f;
            for (et_u64 column = lane; column < count; column += 32)
                value = fmaxf(value, input[column]);
            for (unsigned int offset = 16; offset; offset >>= 1)
                value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
            if (!lane) {
                if (value == 0.0f) {
                    value = -1.0f / 0.0f;
                    for (et_u64 column = 0; column < count; ++column)
                        value = fmaxf(value, input[column]);
                }
                ET_OUTPUT[row] = cast_dtype(value, a.compute_dtype);
            }
        }
        return;
    }
    if (a.operation == 0) {
        // Backend-defined F32 Sum. Short reductions use a lane-strided warp;
        // wide reductions use 1024 threads with four consecutive values per
        // iteration and two descending warp trees. This bounds sequential
        // accumulation depth without changing opmath or semantic narrowing.
        unsigned int rank = et_meta(a)[1], lane = threadIdx.x & 31U;
        if (rank > 64) { et_error(a, 4); return; }
        const et_u64 *shape = et_shape(a, 0), *dims = et_tail(a);
        et_u64 mask = 0, count = 1;
        for (et_u64 i = 0; i < a.integers[0]; ++i) mask |= 1ULL << dims[i];
        for (unsigned int i = 0; i < rank; ++i) if (mask & (1ULL << i)) count *= shape[i];
        for (et_u64 row = et_thread() / 32; row < a.elements; row += (et_u64)gridDim.x * (blockDim.x / 32)) {
            float sum = 0.0f;
            for (et_u64 r = lane; r < count; r += 32) {
                et_u64 out = row, reduced = r, source = 0, stride = 1;
                for (int d = (int)rank - 1; d >= 0; --d) {
                    bool selected = mask & (1ULL << d);
                    et_u64 coordinate = selected ? reduced % shape[d] : out % shape[d];
                    if (selected) reduced /= shape[d]; else out /= shape[d];
                    source += coordinate * stride; stride *= shape[d];
                }
                sum += ET_INPUT(0)[source];
            }
            for (unsigned int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffffU, sum, offset);
            if (!lane) ET_OUTPUT[row] = sum;
        }
        return;
    }
#endif
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
#ifdef ET_COMPUTE_F32
__device__ et_u64 et_rms_source_row(const CudaKernelArgs &a, et_u64 row) {
    et_u64 rank = a.integers[1];
    if (!rank) return row;
    et_u64 source = a.integers[2];
    for (int dimension = (int)rank - 1; dimension >= 0; --dimension) {
        et_u64 size = a.integers[3 + dimension];
        source += (row % size) * a.integers[3 + rank + dimension];
        row /= size;
    }
    return source;
}
#endif
extern "C" __global__ void et_rms_norm(CudaKernelArgs a) {
#ifdef ET_COMPUTE_F32
    // One warp per row, four independently rounded F32 partials per lane.
    // fmad=false keeps square and accumulation boundaries distinct. Inputs
    // are materialized opmath values; narrowing remains in the output plan.
    et_u64 width = a.integers[0];
    if (!width) return;
    et_u64 rows = a.elements / width;
    unsigned int lane = threadIdx.x & 31U;
    for (et_u64 row = et_thread() / 32; row < rows; row += (et_u64)gridDim.x * (blockDim.x / 32)) {
        et_u64 source_row = et_rms_source_row(a, row);
        float partial[4] = {0, 0, 0, 0};
        for (et_u64 k = lane * 4; k < width; k += 128) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) if (k + j < width) {
                float value = et_load<float>(a.inputs[0], a.input_dtypes[0], source_row * width + k + j);
                partial[j] += value * value;
            }
        }
        float sum = ((partial[0] + partial[1]) + partial[2]) + partial[3];
        for (unsigned int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffffU, sum, offset);
        sum = __shfl_sync(0xffffffffU, sum, 0);
        float mean = sum * (1.0f / (float)width);
        float inverse = rsqrtf(mean + (float)a.scalars[0]);
        for (et_u64 k = lane; k < width; k += 32) {
            float value = et_load<float>(a.inputs[0], a.input_dtypes[0], source_row * width + k) * inverse;
            if (a.inputs[1]) value *= et_load<float>(a.inputs[1], a.input_dtypes[1], k);
            et_store(a.output, a.output_dtype, row * width + k, value);
        }
    }
#else
    et_u64 i = et_thread(); if (i >= a.elements) return;
    et_u64 width = et_shape(a, 0)[et_meta(a)[1] - 1], base = i / width * width;
    double sum = 0; for (et_u64 d = 0; d < width; ++d) { double v = et_load<double>(a.inputs[0], a.input_dtypes[0], base + d); sum += v * v; }
    double value = et_load<double>(a.inputs[0], a.input_dtypes[0], i) / sqrt(sum / width + (double)a.scalars[0]);
    if (a.inputs[1]) value *= et_load<double>(a.inputs[1], a.input_dtypes[1], i % width);
    et_store(a.output, a.output_dtype, i, value);
#endif
}
#ifdef ET_COMPUTE_F32
// Four adjacent values feed four separate partial chains. Vectorizing only
// the load must not combine these accumulators or change square/add rounding.
template<unsigned int DType>
__device__ void et_rms_vector_partials(et_u64 address, et_u64 width,
                                      unsigned int lane, float *partial) {
    for (et_u64 k = lane * 4; k < width; k += 128) {
        float values[4];
        if (DType == 1) {
            et_u64 source = address + k * 4;
            asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                         : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]), "=f"(values[3])
                         : "l"(source));
        } else {
            et_u64 packed, source = address + k * 2;
            asm volatile("ld.global.u64 %0, [%1];" : "=l"(packed) : "l"(source));
            #pragma unroll
            for (unsigned int j = 0; j < 4; ++j) {
                unsigned short bits = (unsigned short)(packed >> (j * 16));
                values[j] = DType == 3 ? et_bfloat_float(bits) : et_half_float(bits);
            }
        }
        #pragma unroll
        for (unsigned int j = 0; j < 4; ++j) {
            float square = values[j] * values[j];
            partial[j] += square;
        }
    }
}
// Preserve et_rms_norm's exact 32-lane reduction tree while giving wide rows
// a full block for the output pass. One block owns one row at a time.
template<bool Vectorized>
__device__ void et_rms_norm_wide_impl(CudaKernelArgs a) {
    et_u64 width = a.integers[0];
    if (!width) return;
    et_u64 rows = a.elements / width;
    unsigned int lane = threadIdx.x & 31U;
    __shared__ float inverse;
    for (et_u64 row = blockIdx.x; row < rows; row += gridDim.x) {
        et_u64 source_row = et_rms_source_row(a, row);
        if (threadIdx.x < 32) {
            float partial[4] = {0, 0, 0, 0};
            unsigned int dtype = a.input_dtypes[0];
            et_u64 bytes = dtype == 1 ? 4 : 2;
            et_u64 address = a.inputs[0] + source_row * width * bytes;
            bool vectorized = Vectorized && width % 4 == 0 &&
                (dtype == 1 || dtype == 2 || dtype == 3) && address % (bytes * 4) == 0;
            if (vectorized) {
                if (dtype == 1) et_rms_vector_partials<1>(address, width, lane, partial);
                else if (dtype == 2) et_rms_vector_partials<2>(address, width, lane, partial);
                else et_rms_vector_partials<3>(address, width, lane, partial);
            } else {
                for (et_u64 k = lane * 4; k < width; k += 128) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) if (k + j < width) {
                        float value = et_load<float>(a.inputs[0], a.input_dtypes[0], source_row * width + k + j);
                        partial[j] += value * value;
                    }
                }
            }
            float sum = ((partial[0] + partial[1]) + partial[2]) + partial[3];
            for (unsigned int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffffU, sum, offset);
            if (!lane) {
                float mean = sum * (1.0f / (float)width);
                inverse = rsqrtf(mean + (float)a.scalars[0]);
            }
        }
        __syncthreads();
        for (et_u64 k = threadIdx.x; k < width; k += blockDim.x) {
            float value = et_load<float>(a.inputs[0], a.input_dtypes[0], source_row * width + k) * inverse;
            if (a.inputs[1]) value *= et_load<float>(a.inputs[1], a.input_dtypes[1], k);
            et_store(a.output, a.output_dtype, row * width + k, value);
        }
        __syncthreads();
    }
}
// Opt-in one-CTA-per-row specialization. Keep four independently rounded
// partial chains, the original shuffle tree, and typed load/store conversions.
template<unsigned DType, unsigned OutType>
__device__ void et_rms_norm_static2816_impl(CudaKernelArgs a) {
    constexpr unsigned width = 2816;
    unsigned row = blockIdx.x, lane = threadIdx.x & 31;
    __shared__ float inverse;
    auto source_row = et_rms_source_row(a, row);
    if (threadIdx.x < 32) {
        float partial[4] = {0, 0, 0, 0};
        auto address = a.inputs[0] + source_row * width * (DType == 1 ? 4 : 2);
        bool aligned = address % (DType == 1 ? 16 : 8) == 0;
        #pragma unroll
        for (unsigned step = 0; step < 22; ++step) {
            unsigned k = lane * 4 + step * 128;
            float values[4];
            if (aligned) {
                if (DType == 1) {
                    auto source = address + k * 4;
                    asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                        : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]), "=f"(values[3]) : "l"(source));
                } else {
                    et_u64 packed, source = address + k * 2;
                    asm volatile("ld.global.u64 %0, [%1];" : "=l"(packed) : "l"(source));
                    #pragma unroll
                    for (unsigned j = 0; j < 4; ++j) {
                        unsigned short bits = packed >> (j * 16);
                        values[j] = DType == 3 ? et_bfloat_float(bits) : et_half_float(bits);
                    }
                }
            } else {
                #pragma unroll
                for (unsigned j = 0; j < 4; ++j)
                    values[j] = et_load<float>(a.inputs[0], DType, source_row * width + k + j);
            }
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                float square = values[j] * values[j];
                partial[j] += square;
            }
        }
        float sum = ((partial[0] + partial[1]) + partial[2]) + partial[3];
        for (unsigned offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffU, sum, offset);
        if (!lane) inverse = rsqrtf(sum * (1.0f / float(width)) + float(a.scalars[0]));
    }
    __syncthreads();
    for (unsigned k = threadIdx.x; k < width; k += blockDim.x) {
        float value = et_load<float>(a.inputs[0], DType, source_row * width + k) * inverse;
        if (a.inputs[1]) value *= et_load<float>(a.inputs[1], a.input_dtypes[1], k);
        et_store(a.output, OutType, row * width + k, value);
    }
}

extern "C" __global__ void et_rms_norm_static2816_1_1(CudaKernelArgs a) {
    et_rms_norm_static2816_impl<1, 1>(a);
}

extern "C" __global__ void et_rms_norm_static2816_1_3(CudaKernelArgs a) {
    et_rms_norm_static2816_impl<1, 3>(a);
}

extern "C" __global__ void et_rms_norm_static2816_3_1(CudaKernelArgs a) {
    et_rms_norm_static2816_impl<3, 1>(a);
}

extern "C" __global__ void et_rms_norm_static2816_3_3(CudaKernelArgs a) {
    et_rms_norm_static2816_impl<3, 3>(a);
}

extern "C" __global__ void et_shared_rms_norm_f32(CudaKernelArgs a) {
    et_u64 width = a.integers[0];
    if (!width) return;
    et_u64 elements = a.integers[1], outputs = a.integers[2];
    et_u64 rows = elements / width;
    unsigned int lane = threadIdx.x & 31U;
    __shared__ float inverse;
    for (et_u64 row = blockIdx.x; row < rows; row += gridDim.x) {
        et_u64 source_row = row;
        if (threadIdx.x < 32) {
            float partial[4] = {0, 0, 0, 0};
            unsigned int dtype = a.input_dtypes[0];
            et_u64 bytes = dtype == 1 ? 4 : 2;
            et_u64 address = a.inputs[0] + source_row * width * bytes;
            bool vectorized = width % 4 == 0 &&
                (dtype == 1 || dtype == 2 || dtype == 3) && address % (bytes * 4) == 0;
            if (vectorized) {
                if (dtype == 1) et_rms_vector_partials<1>(address, width, lane, partial);
                else if (dtype == 2) et_rms_vector_partials<2>(address, width, lane, partial);
                else et_rms_vector_partials<3>(address, width, lane, partial);
            } else {
                for (et_u64 k = lane * 4; k < width; k += 128) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) if (k + j < width) {
                        float value = et_load<float>(a.inputs[0], a.input_dtypes[0], source_row * width + k + j);
                        partial[j] += value * value;
                    }
                }
            }
            float sum = ((partial[0] + partial[1]) + partial[2]) + partial[3];
            for (unsigned int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffffU, sum, offset);
            if (!lane) {
                float mean = sum * (1.0f / (float)width);
                inverse = rsqrtf(mean + (float)a.scalars[0]);
            }
        }
        __syncthreads();
        for (et_u64 k = threadIdx.x; k < width; k += blockDim.x) {
            float value = et_load<float>(a.inputs[0], a.input_dtypes[0], source_row * width + k) * inverse;
            for (et_u64 output = 0; output < outputs; ++output) {
                float result = value;
                if (a.inputs[output + 1]) result *= et_load<float>(a.inputs[output + 1], a.input_dtypes[output + 1], k);
                et_store(a.output, a.output_dtype, output * elements + row * width + k, result);
            }
        }
        __syncthreads();
    }
}
extern "C" __global__ void et_rms_norm_wide(CudaKernelArgs a) {
    et_rms_norm_wide_impl<false>(a);
}
extern "C" __global__ void et_rms_norm_wide_vector(CudaKernelArgs a) {
    et_rms_norm_wide_impl<true>(a);
}
#endif
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

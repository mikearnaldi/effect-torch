__device__ double packed_f16(const unsigned char *bytes, unsigned int offset) {
    unsigned int bits = (unsigned int)bytes[offset] | ((unsigned int)bytes[offset + 1] << 8);
    double sign = (bits & 0x8000U) != 0 ? -1.0 : 1.0;
    unsigned int exponent = (bits >> 10) & 0x1fU;
    unsigned int mantissa = bits & 0x03ffU;
    if (exponent == 0) return sign * ldexp((double)mantissa, -24);
    if (exponent == 31) return mantissa == 0 ? sign / 0.0 : 0.0 / 0.0;
    return sign * ldexp(1024.0 + (double)mantissa, (int)exponent - 25);
}

__device__ __forceinline__ float packed_f16_f32(const unsigned char *bytes, unsigned int offset) {
    unsigned int bits = (unsigned int)bytes[offset] | ((unsigned int)bytes[offset + 1] << 8);
    unsigned int sign = (bits & 0x8000U) << 16;
    unsigned int exponent = (bits >> 10) & 0x1fU;
    unsigned int mantissa = bits & 0x03ffU;
    unsigned int output;
    if (exponent == 0) {
        if (mantissa == 0) return __uint_as_float(sign);
        int adjusted = -14;
        while ((mantissa & 0x0400U) == 0) {
            mantissa <<= 1;
            adjusted -= 1;
        }
        output = sign | ((unsigned int)(adjusted + 127) << 23) | ((mantissa & 0x03ffU) << 13);
    } else if (exponent == 31) {
        output = sign | 0x7f800000U | (mantissa << 13);
    } else {
        output = sign | ((exponent - 15U + 127U) << 23) | (mantissa << 13);
    }
    return __uint_as_float(output);
}

__device__ unsigned int kquant_block_bytes(unsigned int codec) {
    switch (codec) {
        case 0: return 84;
        case 1: return 110;
        case 2: return 144;
        case 3: return 176;
        default: return 210;
    }
}

__device__ void scale_min_k4(
    const unsigned char *scales,
    unsigned int group,
    unsigned int *scale,
    unsigned int *minimum
) {
    if (group < 4) {
        *scale = scales[group] & 63U;
        *minimum = scales[group + 4] & 63U;
    } else {
        *scale = (scales[group + 4] & 15U) | ((scales[group - 4] >> 6) << 4);
        *minimum = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
    }
}

__device__ double q2_k_value(const unsigned char *block, unsigned int index) {
    unsigned int group = index / 16;
    unsigned int lane = index % 16;
    unsigned int half = group / 8;
    unsigned int group_in_half = group % 8;
    unsigned int shift = (group_in_half / 2) * 2;
    unsigned int offset = (group_in_half % 2) * 16;
    unsigned int scale = block[group];
    unsigned int quant = (block[16 + half * 32 + offset + lane] >> shift) & 3U;
    return packed_f16(block, 80) * (double)(scale & 15U) * (double)quant
        - packed_f16(block, 82) * (double)(scale >> 4);
}

__device__ double q3_k_value(const unsigned char *block, unsigned int index) {
    unsigned int group = index / 16;
    unsigned int lane = index % 16;
    unsigned int half = group / 8;
    unsigned int group_in_half = group % 8;
    unsigned int quant_lane = group_in_half / 2;
    unsigned int shift = quant_lane * 2;
    unsigned int offset = (group_in_half % 2) * 16;
    unsigned int low_scale = group < 8 ? block[96 + group] & 15U : block[96 + group - 8] >> 4;
    unsigned int high_scale = (block[104 + group % 4] >> (2 * (group / 4))) & 3U;
    int scale = (int)(low_scale | (high_scale << 4)) - 32;
    int quant = (int)((block[32 + half * 32 + offset + lane] >> shift) & 3U);
    if ((block[offset + lane] & (1U << (half * 4 + quant_lane))) == 0) quant -= 4;
    return packed_f16(block, 108) * (double)scale * (double)quant;
}

__device__ double q4_k_value(const unsigned char *block, unsigned int index) {
    unsigned int group = index / 32;
    unsigned int lane = index % 32;
    unsigned int scale;
    unsigned int minimum;
    scale_min_k4(block + 4, group, &scale, &minimum);
    unsigned int packed = block[16 + (group / 2) * 32 + lane];
    unsigned int quant = group % 2 == 0 ? packed & 15U : packed >> 4;
    return packed_f16(block, 0) * (double)scale * (double)quant
        - packed_f16(block, 2) * (double)minimum;
}

__device__ double q5_k_value(const unsigned char *block, unsigned int index) {
    unsigned int group = index / 32;
    unsigned int lane = index % 32;
    unsigned int pair = group / 2;
    unsigned int side = group % 2;
    unsigned int scale;
    unsigned int minimum;
    scale_min_k4(block + 4, group, &scale, &minimum);
    unsigned int packed = block[48 + pair * 32 + lane];
    unsigned int quant = side == 0 ? packed & 15U : packed >> 4;
    if ((block[16 + lane] & (1U << (pair * 2 + side))) != 0) quant += 16;
    return packed_f16(block, 0) * (double)scale * (double)quant
        - packed_f16(block, 2) * (double)minimum;
}

__device__ double q6_k_value(const unsigned char *block, unsigned int index) {
    unsigned int half = index / 128;
    unsigned int within = index % 128;
    unsigned int quarter = within / 32;
    unsigned int lane = within % 32;
    unsigned int low_offset = half * 64 + lane + (quarter % 2) * 32;
    unsigned int low = quarter < 2 ? block[low_offset] & 15U : block[low_offset] >> 4;
    unsigned int high = (block[128 + half * 32 + lane] >> (quarter * 2)) & 3U;
    int quant = (int)(low | (high << 4)) - 32;
    unsigned int scale_lane = lane / 16;
    int scale = (int)(signed char)block[192 + half * 8 + scale_lane + quarter * 2];
    return packed_f16(block, 208) * (double)scale * (double)quant;
}

__device__ double kquant_value(
    const unsigned char *block,
    unsigned int index,
    unsigned int codec
) {
    switch (codec) {
        case 0: return q2_k_value(block, index);
        case 1: return q3_k_value(block, index);
        case 2: return q4_k_value(block, index);
        case 3: return q5_k_value(block, index);
        default: return q6_k_value(block, index);
    }
}

extern "C" __global__ void quantize_q8_f64(
    const double *x,
    signed char *quants,
    float *scales,
    unsigned int blocks
) {
    unsigned int lane = threadIdx.x & 31U;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int warps_per_block = blockDim.x >> 5;
    unsigned int block = blockIdx.x * warps_per_block + warp;
    if (block >= blocks) return;

    float value = (float)x[(unsigned long long)block * 32 + lane];
    float maximum = fabsf(value);
#pragma unroll
    for (unsigned int offset = 16; offset != 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
    }
    maximum = __shfl_sync(0xffffffffU, maximum, 0);
    float scale = maximum / 127.0f;
    int quant = maximum == 0.0f ? 0 : __float2int_rn(value / scale);
    quant = quant < -127 ? -127 : quant > 127 ? 127 : quant;
    quants[(unsigned long long)block * 32 + lane] = (signed char)quant;
    if (lane == 0) scales[block] = scale;
}

template <unsigned int codec>
__device__ __forceinline__ int kquant_code_i8(
    const unsigned char *block,
    unsigned int index
) {
    if (codec == 0) {
        unsigned int group = index / 16;
        unsigned int lane = index % 16;
        unsigned int half = group / 8;
        unsigned int group_in_half = group % 8;
        unsigned int shift = (group_in_half / 2) * 2;
        unsigned int offset = (group_in_half % 2) * 16;
        return (block[16 + half * 32 + offset + lane] >> shift) & 3U;
    }
    if (codec == 1) {
        unsigned int group = index / 16;
        unsigned int lane = index % 16;
        unsigned int half = group / 8;
        unsigned int group_in_half = group % 8;
        unsigned int quant_lane = group_in_half / 2;
        unsigned int shift = quant_lane * 2;
        unsigned int offset = (group_in_half % 2) * 16;
        int quant = (int)((block[32 + half * 32 + offset + lane] >> shift) & 3U);
        if ((block[offset + lane] & (1U << (half * 4 + quant_lane))) == 0) quant -= 4;
        return quant;
    }
    if (codec == 2) {
        unsigned int group = index / 32;
        unsigned int lane = index % 32;
        unsigned int packed = block[16 + (group / 2) * 32 + lane];
        return group % 2 == 0 ? packed & 15U : packed >> 4;
    }
    if (codec == 3) {
        unsigned int group = index / 32;
        unsigned int lane = index % 32;
        unsigned int pair = group / 2;
        unsigned int side = group % 2;
        unsigned int packed = block[48 + pair * 32 + lane];
        unsigned int quant = side == 0 ? packed & 15U : packed >> 4;
        if ((block[16 + lane] & (1U << (pair * 2 + side))) != 0) quant += 16;
        return quant;
    }
    unsigned int half = index / 128;
    unsigned int within = index % 128;
    unsigned int quarter = within / 32;
    unsigned int lane = within % 32;
    unsigned int low_offset = half * 64 + lane + (quarter % 2) * 32;
    unsigned int low = quarter < 2 ? block[low_offset] & 15U : block[low_offset] >> 4;
    unsigned int high = (block[128 + half * 32 + lane] >> (quarter * 2)) & 3U;
    return (int)(low | (high << 4)) - 32;
}

__device__ __forceinline__ unsigned int load_i8x4_unaligned(
    const unsigned char *pointer
) {
    return (unsigned int)pointer[0]
        | ((unsigned int)pointer[1] << 8)
        | ((unsigned int)pointer[2] << 16)
        | ((unsigned int)pointer[3] << 24);
}

template <unsigned int codec>
__device__ __forceinline__ int kquant_code_i8x4(
    const unsigned char *block,
    unsigned int index
) {
    if (codec == 0 || codec == 1) {
        unsigned int group = index / 16;
        unsigned int lane = index % 16;
        unsigned int half = group / 8;
        unsigned int group_in_half = group % 8;
        unsigned int quant_lane = group_in_half / 2;
        unsigned int shift = quant_lane * 2;
        unsigned int offset = (group_in_half % 2) * 16;
        const unsigned char *quants =
            block + (codec == 0 ? 16 : 32) + half * 32 + offset + lane;
        unsigned int packed = codec == 0
            ? *(const unsigned int *)quants
            : load_i8x4_unaligned(quants);
        packed = (packed >> shift) & 0x03030303U;
        if (codec == 1) {
            const unsigned char *masks = block + offset + lane;
            unsigned int high = load_i8x4_unaligned(masks);
            high = (high >> (half * 4 + quant_lane)) & 0x01010101U;
            packed |= high << 2;
        }
        return (int)packed;
    }
    if (codec == 2 || codec == 3) {
        unsigned int group = index / 32;
        unsigned int lane = index % 32;
        unsigned int pair = group / 2;
        unsigned int side = group % 2;
        unsigned int packed = *(const unsigned int *)(
            block + (codec == 2 ? 16 : 48) + pair * 32 + lane
        );
        packed = (packed >> (side * 4)) & 0x0f0f0f0fU;
        if (codec == 3) {
            unsigned int high = *(const unsigned int *)(block + 16 + lane);
            high = (high >> (pair * 2 + side)) & 0x01010101U;
            packed |= high << 4;
        }
        return (int)packed;
    }
    unsigned int half = index / 128;
    unsigned int within = index % 128;
    unsigned int quarter = within / 32;
    unsigned int lane = within % 32;
    unsigned int low_offset = half * 64 + lane + (quarter % 2) * 32;
    unsigned int low = load_i8x4_unaligned(block + low_offset);
    low = (low >> (quarter < 2 ? 0 : 4)) & 0x0f0f0f0fU;
    unsigned int high = load_i8x4_unaligned(block + 128 + half * 32 + lane);
    high = (high >> (quarter * 2)) & 0x03030303U;
    return (int)(low | (high << 4));
}

template <unsigned int codec>
__device__ __forceinline__ void kquant_group_coefficients(
    const unsigned char *block,
    unsigned int group,
    float *scale,
    float *minimum
) {
    if (codec == 0) {
        unsigned int packed = block[group];
        *scale = packed_f16_f32(block, 80) * (float)(packed & 15U);
        *minimum = packed_f16_f32(block, 82) * (float)(packed >> 4);
        return;
    }
    if (codec == 1) {
        unsigned int low = group < 8 ? block[96 + group] & 15U : block[96 + group - 8] >> 4;
        unsigned int high = (block[104 + group % 4] >> (2 * (group / 4))) & 3U;
        *scale = packed_f16_f32(block, 108) * (float)((int)(low | (high << 4)) - 32);
        *minimum = 0.0f;
        return;
    }
    if (codec == 2 || codec == 3) {
        unsigned int group_scale;
        unsigned int group_minimum;
        scale_min_k4(block + 4, group, &group_scale, &group_minimum);
        *scale = packed_f16_f32(block, 0) * (float)group_scale;
        *minimum = packed_f16_f32(block, 2) * (float)group_minimum;
        return;
    }
    *scale = packed_f16_f32(block, 208) * (float)(signed char)block[192 + group];
    *minimum = 0.0f;
}

__device__ __forceinline__ int pack_i8x4(int a, int b, int c, int d) {
    return (int)((unsigned int)(unsigned char)a
        | ((unsigned int)(unsigned char)b << 8)
        | ((unsigned int)(unsigned char)c << 16)
        | ((unsigned int)(unsigned char)d << 24));
}

__device__ __forceinline__ int dot_i8x4(int a, int b, int sum) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(sum));
    return result;
}

template <unsigned int codec, unsigned int vector_tile>
__device__ __forceinline__ void quantized_linear_k_q8_warp(
    const signed char *x,
    const float *x_scales,
    const unsigned char *weight,
    const double *bias,
    double *out,
    unsigned int vectors,
    unsigned int rows,
    unsigned int columns,
    unsigned int row_bytes,
    unsigned int has_bias
) {
    unsigned int lane = threadIdx.x & 31U;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int warps_per_block = blockDim.x >> 5;
    unsigned int row = blockIdx.x * warps_per_block + warp;
    unsigned int first_vector = blockIdx.y * vector_tile;
    if (row >= rows || first_vector >= vectors) return;

    float sums[vector_tile];
#pragma unroll
    for (unsigned int vector_offset = 0; vector_offset < vector_tile; ++vector_offset) {
        sums[vector_offset] = 0.0f;
    }
    const unsigned char *packed_row = weight + (unsigned long long)row * row_bytes;
    const unsigned int block_bytes = codec == 0 ? 84 : codec == 1 ? 110 : codec == 2 ? 144 : codec == 3 ? 176 : 210;
    const unsigned int group_values = codec == 2 || codec == 3 ? 32 : 16;
    for (unsigned int block_index = 0; block_index < columns / 256; ++block_index) {
        const unsigned char *block = packed_row + (unsigned long long)block_index * block_bytes;
        for (unsigned int chunk = lane; chunk < 64; chunk += 32) {
            unsigned int weight_index = chunk * 4;
            unsigned int group = weight_index / group_values;
            int weight_values = kquant_code_i8x4<codec>(block, weight_index);
            float weight_scale;
            float weight_minimum;
            kquant_group_coefficients<codec>(block, group, &weight_scale, &weight_minimum);
#pragma unroll
            for (unsigned int vector_offset = 0; vector_offset < vector_tile; ++vector_offset) {
                unsigned int vector = first_vector + vector_offset;
                if (vector < vectors) {
                    unsigned long long input_index = (unsigned long long)vector * columns
                        + block_index * 256 + weight_index;
                    int input_values = *(const int *)(x + input_index);
                    int dot = dot_i8x4(weight_values, input_values, 0);
                    int input_sum = dot_i8x4(0x01010101, input_values, 0);
                    if (codec == 1) {
                        dot -= 4 * input_sum;
                    }
                    if (codec == 4) dot -= 32 * input_sum;
                    float input_scale = x_scales[input_index / 32];
                    sums[vector_offset] += input_scale * (
                        weight_scale * (float)dot
                        - weight_minimum * (float)input_sum
                    );
                }
            }
        }
    }

#pragma unroll
    for (unsigned int offset = 16; offset != 0; offset >>= 1) {
#pragma unroll
        for (unsigned int vector_offset = 0; vector_offset < vector_tile; ++vector_offset) {
            sums[vector_offset] += __shfl_down_sync(0xffffffffU, sums[vector_offset], offset);
        }
    }
    if (lane == 0) {
        float bias_value = has_bias != 0 ? (float)bias[row] : 0.0f;
#pragma unroll
        for (unsigned int vector_offset = 0; vector_offset < vector_tile; ++vector_offset) {
            unsigned int vector = first_vector + vector_offset;
            if (vector < vectors) {
                out[(unsigned long long)vector * rows + row] = (double)(sums[vector_offset] + bias_value);
            }
        }
    }
}

template <unsigned int codec>
__device__ __forceinline__ void quantized_linear_k_q8_cooperative(
    const signed char *x,
    const float *x_scales,
    const unsigned char *weight,
    const double *bias,
    double *out,
    unsigned int vectors,
    unsigned int rows,
    unsigned int columns,
    unsigned int row_bytes,
    unsigned int has_bias,
    float *partials
) {
    unsigned int lane = threadIdx.x & 31U;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int warps_per_block = blockDim.x >> 5;
    unsigned int row = blockIdx.x;
    unsigned int vector = blockIdx.y;
    if (row >= rows || vector >= vectors) return;

    float sum = 0.0f;
    const unsigned char *packed_row = weight + (unsigned long long)row * row_bytes;
    const unsigned int block_bytes = codec == 0 ? 84 : codec == 1 ? 110 : codec == 2 ? 144 : codec == 3 ? 176 : 210;
    const unsigned int group_values = codec == 2 || codec == 3 ? 32 : 16;
    for (unsigned int block_index = warp; block_index < columns / 256; block_index += warps_per_block) {
        const unsigned char *block = packed_row + (unsigned long long)block_index * block_bytes;
        for (unsigned int chunk = lane; chunk < 64; chunk += 32) {
            unsigned int weight_index = chunk * 4;
            unsigned int group = weight_index / group_values;
            int weight_values = kquant_code_i8x4<codec>(block, weight_index);
            float weight_scale;
            float weight_minimum;
            kquant_group_coefficients<codec>(block, group, &weight_scale, &weight_minimum);
            unsigned long long input_index = (unsigned long long)vector * columns
                + block_index * 256 + weight_index;
            int input_values = *(const int *)(x + input_index);
            int dot = dot_i8x4(weight_values, input_values, 0);
            int input_sum = dot_i8x4(0x01010101, input_values, 0);
            if (codec == 1) {
                dot -= 4 * input_sum;
            }
            if (codec == 4) dot -= 32 * input_sum;
            float input_scale = x_scales[input_index / 32];
            sum += input_scale * (
                weight_scale * (float)dot
                - weight_minimum * (float)input_sum
            );
        }
    }

    partials[warp * 32 + lane] = sum;
    __syncthreads();
    if (warp != 0) return;
    sum = partials[lane];
    for (unsigned int source_warp = 1; source_warp < warps_per_block; ++source_warp) {
        sum += partials[source_warp * 32 + lane];
    }
    for (unsigned int offset = 16; offset != 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    if (lane == 0) {
        float bias_value = has_bias != 0 ? (float)bias[row] : 0.0f;
        out[(unsigned long long)vector * rows + row] = (double)(sum + bias_value);
    }
}

extern "C" __global__ void quantized_linear_k_mmv_q8(
    const signed char *x,
    const float *x_scales,
    const unsigned char *weight,
    const double *bias,
    double *out,
    unsigned int vectors,
    unsigned int rows,
    unsigned int columns,
    unsigned int row_bytes,
    unsigned int codec,
    unsigned int has_bias
) {
    if (columns / 256 >= 4) {
        extern __shared__ float partials[];
        switch (codec) {
            case 0: quantized_linear_k_q8_cooperative<0>(x, x_scales, weight, bias, out, vectors, rows, columns, row_bytes, has_bias, partials); break;
            case 1: quantized_linear_k_q8_cooperative<1>(x, x_scales, weight, bias, out, vectors, rows, columns, row_bytes, has_bias, partials); break;
            case 2: quantized_linear_k_q8_cooperative<2>(x, x_scales, weight, bias, out, vectors, rows, columns, row_bytes, has_bias, partials); break;
            case 3: quantized_linear_k_q8_cooperative<3>(x, x_scales, weight, bias, out, vectors, rows, columns, row_bytes, has_bias, partials); break;
            default: quantized_linear_k_q8_cooperative<4>(x, x_scales, weight, bias, out, vectors, rows, columns, row_bytes, has_bias, partials); break;
        }
        return;
    }
    switch (codec) {
        case 0: quantized_linear_k_q8_warp<0, 1>(x, x_scales, weight, bias, out, vectors, rows, columns, row_bytes, has_bias); break;
        case 1: quantized_linear_k_q8_warp<1, 1>(x, x_scales, weight, bias, out, vectors, rows, columns, row_bytes, has_bias); break;
        case 2: quantized_linear_k_q8_warp<2, 1>(x, x_scales, weight, bias, out, vectors, rows, columns, row_bytes, has_bias); break;
        case 3: quantized_linear_k_q8_warp<3, 1>(x, x_scales, weight, bias, out, vectors, rows, columns, row_bytes, has_bias); break;
        default: quantized_linear_k_q8_warp<4, 1>(x, x_scales, weight, bias, out, vectors, rows, columns, row_bytes, has_bias); break;
    }
}

extern "C" __global__ void quantized_linear_k_mmq_q8(
    const signed char *x,
    const float *x_scales,
    const unsigned char *weight,
    const double *bias,
    double *out,
    unsigned int vectors,
    unsigned int rows,
    unsigned int columns,
    unsigned int row_bytes,
    unsigned int codec,
    unsigned int has_bias
) {
    switch (codec) {
        case 0: quantized_linear_k_q8_warp<0, 16>(x, x_scales, weight, bias, out, vectors, rows, columns, row_bytes, has_bias); break;
        case 1: quantized_linear_k_q8_warp<1, 16>(x, x_scales, weight, bias, out, vectors, rows, columns, row_bytes, has_bias); break;
        case 2: quantized_linear_k_q8_warp<2, 16>(x, x_scales, weight, bias, out, vectors, rows, columns, row_bytes, has_bias); break;
        case 3: quantized_linear_k_q8_warp<3, 16>(x, x_scales, weight, bias, out, vectors, rows, columns, row_bytes, has_bias); break;
        default: quantized_linear_k_q8_warp<4, 16>(x, x_scales, weight, bias, out, vectors, rows, columns, row_bytes, has_bias); break;
    }
}

extern "C" __global__ void quantized_embedding_k_f64(
    const double *indexes,
    const unsigned char *weight,
    double *out,
    unsigned int len,
    unsigned int columns,
    unsigned int row_bytes,
    unsigned int codec
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    unsigned int lookup = index / columns;
    unsigned int column = index % columns;
    unsigned long long row = (unsigned long long)indexes[lookup];
    const unsigned char *packed_row = weight + row * row_bytes;
    const unsigned char *block = packed_row + (column / 256) * kquant_block_bytes(codec);
    out[index] = cast_dtype(kquant_value(block, column % 256, codec), 1);
}

extern "C" __global__ void quantized_embedding_k_f32(
    const double *indexes,
    const unsigned char *weight,
    float *out,
    unsigned int len,
    unsigned int columns,
    unsigned int row_bytes,
    unsigned int codec
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    unsigned int lookup = index / columns;
    unsigned int column = index % columns;
    unsigned long long row = (unsigned long long)indexes[lookup];
    const unsigned char *packed_row = weight + row * row_bytes;
    const unsigned char *block = packed_row + (column / 256) * kquant_block_bytes(codec);
    out[index] = kquant_value(block, column % 256, codec);
}

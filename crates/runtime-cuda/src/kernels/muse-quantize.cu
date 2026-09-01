extern "C" __global__ void muse_gate_quantize_q8_1_f32(
    const float *__restrict__ value,
    const float *__restrict__ gate,
    const float *__restrict__ multiplier,
    block_q8_1 *__restrict__ output,
    unsigned int columns,
    unsigned int padded_columns,
    unsigned int has_multiplier
) {
    unsigned int column = blockDim.x * blockIdx.x + threadIdx.x;
    if (column >= padded_columns) return;

    unsigned int vector = blockIdx.y;
    unsigned int padded_index = vector * padded_columns + column;
    unsigned int block = padded_index / QK8_1;
    unsigned int quant = padded_index % QK8_1;
    float gated = 0.0f;
    if (column < columns) {
        unsigned int index = vector * columns + column;
        float half_gate = __fmul_rn(gate[index], 0.5f);
        float sigmoid = __fadd_rn(__fmul_rn(tanhf(half_gate), 0.5f), 0.5f);
        gated = __fmul_rn(value[index], sigmoid);
        if (has_multiplier != 0) {
            gated = __fmul_rn(gated, multiplier[index]);
        }
    }

    float maximum = fabsf(gated);
    float sum = gated;
#pragma unroll
    for (unsigned int mask = 16; mask != 0; mask >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, mask, 32));
        sum += __shfl_xor_sync(0xffffffffU, sum, mask, 32);
    }

    float scale = maximum / 127.0f;
    output[block].qs[quant] = maximum == 0.0f
        ? 0
        : (signed char)roundf(gated / scale);
    if (quant == 0) {
        output[block].ds.x = __float2half(scale);
        output[block].ds.y = __float2half(sum);
    }
}

extern "C" __global__ void cast_f32_bf16(
    const float *__restrict__ input,
    c10::BFloat16 *__restrict__ output,
    unsigned int len
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) output[index] = __float2bfloat16(input[index]);
}

extern "C" __global__ void dequantize_q2_k_bf16(
    const block_q2_K *__restrict__ input,
    c10::BFloat16 *__restrict__ output
) {
    unsigned int block = blockIdx.x;
    unsigned int thread = threadIdx.x;
    unsigned int group = thread / 32;
    unsigned int lane = thread - 32 * group;
    unsigned int scale_index = 8 * group + lane / 16;
    unsigned int quant = input[block].qs[32 * group + lane];
    c10::BFloat16 *row = output + block * QK_K + 128 * group;
    float scale = __half2float(input[block].dm.x);
    float minimum = __half2float(input[block].dm.y);

#pragma unroll
    for (unsigned int part = 0; part < 4; ++part) {
        unsigned int packed_scale = input[block].scales[scale_index + 2 * part];
        float value = scale * (float)(packed_scale & 0x0fU) * (float)((quant >> (2 * part)) & 3U)
            - minimum * (float)(packed_scale >> 4);
        row[lane + 32 * part] = __float2bfloat16(value);
    }
}

extern "C" __global__ void dequantize_q3_k_bf16(
    const block_q3_K *__restrict__ input,
    c10::BFloat16 *__restrict__ output
) {
    unsigned int block = blockIdx.x;
    unsigned int thread = threadIdx.x;
    unsigned int row = thread / 4;
    unsigned int half_row = row / 2;
    unsigned int half = row % 2;
    unsigned int lane_start = 16 * half + 4 * (thread % 4);
    unsigned int group = half_row / 4;
    unsigned int part = half_row - 4 * group;
    unsigned int mask = 1U << (4 * group + part);
    unsigned int scale_index = 8 * group + 2 * part + half;
    unsigned int shift = 2 * part;

    int packed_scale = scale_index < 4
        ? (input[block].scales[scale_index] & 0x0f) | (((input[block].scales[scale_index + 8] >> 0) & 3) << 4)
        : scale_index < 8
            ? (input[block].scales[scale_index] & 0x0f) | (((input[block].scales[scale_index + 4] >> 2) & 3) << 4)
            : scale_index < 12
                ? (input[block].scales[scale_index - 8] >> 4) | (((input[block].scales[scale_index] >> 4) & 3) << 4)
                : (input[block].scales[scale_index - 8] >> 4) | (((input[block].scales[scale_index - 4] >> 6) & 3) << 4);
    float scale = __half2float(input[block].d) * (float)(packed_scale - 32);
    c10::BFloat16 *destination = output + block * QK_K + 128 * group + 32 * part;
    const uint8_t *quant = input[block].qs + 32 * group;
    const uint8_t *high = input[block].hmask;

#pragma unroll
    for (unsigned int lane = lane_start; lane < lane_start + 4; ++lane) {
        int value = (int)((quant[lane] >> shift) & 3) - ((high[lane] & mask) != 0 ? 0 : 4);
        destination[lane] = __float2bfloat16(scale * (float)value);
    }
}

static __device__ __forceinline__ void q4_k_scale_min(
    unsigned int index,
    const uint8_t *values,
    unsigned int *scale,
    unsigned int *minimum
) {
    if (index < 4) {
        *scale = values[index] & 63;
        *minimum = values[index + 4] & 63;
    } else {
        *scale = (values[index + 4] & 0x0f) | ((values[index - 4] >> 6) << 4);
        *minimum = (values[index + 4] >> 4) | ((values[index] >> 6) << 4);
    }
}

extern "C" __global__ void dequantize_q4_k_bf16(
    const block_q4_K *__restrict__ input,
    c10::BFloat16 *__restrict__ output
) {
    unsigned int block = blockIdx.x;
    unsigned int thread = threadIdx.x;
    unsigned int group = thread / 8;
    unsigned int lane = thread % 8;
    unsigned int scale_index = 2 * group;
    c10::BFloat16 *destination = output + block * QK_K + 64 * group + 4 * lane;
    const uint8_t *quant = input[block].qs + 32 * group + 4 * lane;
    float global_scale = __half2float(input[block].dm.x);
    float global_minimum = __half2float(input[block].dm.y);
    unsigned int scale0, minimum0, scale1, minimum1;
    q4_k_scale_min(scale_index, input[block].scales, &scale0, &minimum0);
    q4_k_scale_min(scale_index + 1, input[block].scales, &scale1, &minimum1);

#pragma unroll
    for (unsigned int value = 0; value < 4; ++value) {
        destination[value] = __float2bfloat16(
            global_scale * (float)scale0 * (float)(quant[value] & 0x0f)
                - global_minimum * (float)minimum0);
        destination[value + 32] = __float2bfloat16(
            global_scale * (float)scale1 * (float)(quant[value] >> 4)
                - global_minimum * (float)minimum1);
    }
}

extern "C" __global__ void dequantize_q5_k_bf16(
    const block_q5_K *__restrict__ input,
    c10::BFloat16 *__restrict__ output
) {
    unsigned int block = blockIdx.x;
    unsigned int thread = threadIdx.x;
    unsigned int group = thread / 16;
    unsigned int lane = thread % 16;
    unsigned int scale_index = 2 * group;
    c10::BFloat16 *destination = output + block * QK_K + 64 * group + 2 * lane;
    const uint8_t *low = input[block].qs + 32 * group + 2 * lane;
    const uint8_t *high = input[block].qh + 2 * lane;
    float global_scale = __half2float(input[block].dm.x);
    float global_minimum = __half2float(input[block].dm.y);
    unsigned int scale0, minimum0, scale1, minimum1;
    q4_k_scale_min(scale_index, input[block].scales, &scale0, &minimum0);
    q4_k_scale_min(scale_index + 1, input[block].scales, &scale1, &minimum1);
    unsigned int mask0 = 1U << (2 * group);
    unsigned int mask1 = mask0 << 1;

#pragma unroll
    for (unsigned int value = 0; value < 2; ++value) {
        destination[value] = __float2bfloat16(
            global_scale * (float)scale0 * (float)((low[value] & 0x0f) + ((high[value] & mask0) != 0 ? 16 : 0))
                - global_minimum * (float)minimum0);
        destination[value + 32] = __float2bfloat16(
            global_scale * (float)scale1 * (float)((low[value] >> 4) + ((high[value] & mask1) != 0 ? 16 : 0))
                - global_minimum * (float)minimum1);
    }
}

extern "C" __global__ void dequantize_q6_k_bf16(
    const block_q6_K *__restrict__ input,
    c10::BFloat16 *__restrict__ output
) {
    unsigned int block = blockIdx.x;
    unsigned int thread = threadIdx.x;
    unsigned int half = thread / 32;
    unsigned int lane = thread - 32 * half;
    unsigned int scale_index = 8 * half + lane / 16;
    c10::BFloat16 *destination = output + block * QK_K + 128 * half + lane;
    const uint8_t *low = input[block].ql + 64 * half + lane;
    uint8_t high = input[block].qh[32 * half + lane];
    const int8_t *scales = input[block].scales + scale_index;
    float scale = __half2float(input[block].d);

    int quant0 = (int8_t)((low[0] & 0x0f) | (((high >> 0) & 3) << 4)) - 32;
    int quant1 = (int8_t)((low[32] & 0x0f) | (((high >> 2) & 3) << 4)) - 32;
    int quant2 = (int8_t)((low[0] >> 4) | (((high >> 4) & 3) << 4)) - 32;
    int quant3 = (int8_t)((low[32] >> 4) | (((high >> 6) & 3) << 4)) - 32;
    destination[0] = __float2bfloat16(scale * (float)(scales[0] * quant0));
    destination[32] = __float2bfloat16(scale * (float)(scales[2] * quant1));
    destination[64] = __float2bfloat16(scale * (float)(scales[4] * quant2));
    destination[96] = __float2bfloat16(scale * (float)(scales[6] * quant3));
}

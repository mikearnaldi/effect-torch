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

__device__ float q2_k_value(const unsigned char *block, unsigned int index) {
    unsigned int group = index / 16;
    unsigned int lane = index % 16;
    unsigned int half = group / 8;
    unsigned int group_in_half = group % 8;
    unsigned int shift = (group_in_half / 2) * 2;
    unsigned int offset = (group_in_half % 2) * 16;
    unsigned int scale = block[group];
    unsigned int quant = (block[16 + half * 32 + offset + lane] >> shift) & 3U;
    return packed_f16_f32(block, 80) * (float)(scale & 15U) * (float)quant
        - packed_f16_f32(block, 82) * (float)(scale >> 4);
}

__device__ float q3_k_value(const unsigned char *block, unsigned int index) {
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
    return packed_f16_f32(block, 108) * (float)scale * (float)quant;
}

__device__ float q4_k_value(const unsigned char *block, unsigned int index) {
    unsigned int group = index / 32;
    unsigned int lane = index % 32;
    unsigned int scale;
    unsigned int minimum;
    scale_min_k4(block + 4, group, &scale, &minimum);
    unsigned int packed = block[16 + (group / 2) * 32 + lane];
    unsigned int quant = group % 2 == 0 ? packed & 15U : packed >> 4;
    return packed_f16_f32(block, 0) * (float)scale * (float)quant
        - packed_f16_f32(block, 2) * (float)minimum;
}

__device__ float q5_k_value(const unsigned char *block, unsigned int index) {
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
    return packed_f16_f32(block, 0) * (float)scale * (float)quant
        - packed_f16_f32(block, 2) * (float)minimum;
}

__device__ float q6_k_value(const unsigned char *block, unsigned int index) {
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
    return packed_f16_f32(block, 208) * (float)scale * (float)quant;
}

__device__ float kquant_value(
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

extern "C" __global__ void et_quantized_linear(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    unsigned int codec = a.integers[0]; et_u64 n = a.integers[1], k = a.integers[2], row_bytes = a.integers[3];
    et_u64 row = i / n, column = i % n;
    const unsigned char *weight = (const unsigned char *)a.inputs[1] + column * row_bytes;
    const float *x = (const float *)a.inputs[0];
    float total = 0.0f;
    unsigned int block_bytes = kquant_block_bytes(codec);
    for (et_u64 d = 0; d < k; ++d) {
        float w = kquant_value(weight + (d / 256) * block_bytes, d % 256, codec);
        total += x[row * k + d] * w;
    }
    if (a.inputs[2]) total += ((const float *)a.inputs[2])[column];
    ((float *)a.output)[i] = total;
}

extern "C" __global__ void et_quantized_embedding(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    unsigned int codec = a.integers[0]; et_u64 n = a.integers[1], k = a.integers[2], row_bytes = a.integers[3], selected;
    if (!et_selected(a, 0, i / k, n, &selected)) return;
    if (a.integers[5] && selected == a.integers[4]) { ((float *)a.output)[i] = 0.0f; return; }
    et_u64 column = i % k;
    const unsigned char *block = (const unsigned char *)a.inputs[1] + selected * row_bytes + (column / 256) * kquant_block_bytes(codec);
    ((float *)a.output)[i] = kquant_value(block, column % 256, codec);
}

__device__ float round_bf16(float value) {
    unsigned int bits = __float_as_uint(value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return __uint_as_float(bits & 0xffff0000U);
}

__device__ float round_f16(float value) {
    unsigned int bits = __float_as_uint(value);
    unsigned int sign = (bits >> 16) & 0x8000U;
    int exponent = (int)((bits >> 23) & 0xffU) - 127 + 15;
    unsigned int mantissa = bits & 0x7fffffU;
    unsigned int half;
    if (exponent <= 0) {
        if (exponent < -10) return __uint_as_float(sign << 16);
        mantissa |= 0x800000U;
        unsigned int shift = (unsigned int)(14 - exponent);
        half = sign | ((mantissa + ((1U << (shift - 1)) - 1U) + ((mantissa >> shift) & 1U)) >> shift);
    } else if (exponent >= 31) {
        half = sign | 0x7c00U | (mantissa != 0 ? 0x0200U : 0U);
    } else {
        mantissa += 0x0fffU + ((mantissa >> 13) & 1U);
        if ((mantissa & 0x800000U) != 0) {
            mantissa = 0;
            exponent += 1;
        }
        half = exponent >= 31 ? sign | 0x7c00U : sign | ((unsigned int)exponent << 10) | (mantissa >> 13);
    }
    unsigned int half_exponent = (half >> 10) & 0x1fU;
    unsigned int half_mantissa = half & 0x03ffU;
    unsigned int output;
    if (half_exponent == 0) {
        if (half_mantissa == 0) {
            output = (half & 0x8000U) << 16;
        } else {
            int adjusted = -14;
            while ((half_mantissa & 0x0400U) == 0) {
                half_mantissa <<= 1;
                adjusted -= 1;
            }
            output = ((half & 0x8000U) << 16) | ((unsigned int)(adjusted + 127) << 23) | ((half_mantissa & 0x03ffU) << 13);
        }
    } else if (half_exponent == 31) {
        output = ((half & 0x8000U) << 16) | 0x7f800000U | (half_mantissa << 13);
    } else {
        output = ((half & 0x8000U) << 16) | ((half_exponent - 15U + 127U) << 23) | (half_mantissa << 13);
    }
    return __uint_as_float(output);
}

__device__ double cast_dtype(double value, unsigned int dtype) {
    switch (dtype) {
        case 0: return value;
        case 1: return (double)(float)value;
        case 2: return (double)round_f16((float)value);
        case 3: return (double)round_bf16((float)value);
        case 4: return (double)(long long)value;
        case 5: return (double)(unsigned int)value;
        case 6: return (double)(unsigned char)value;
        default: return value;
    }
}

__device__ unsigned long long mapped_index(
    unsigned long long index,
    unsigned int rank,
    const unsigned long long *out_shape,
    const unsigned long long *in_shape
) {
    unsigned long long mapped = 0;
    unsigned long long stride = 1;
    for (int axis = (int)rank - 1; axis >= 0; --axis) {
        unsigned long long width = out_shape[axis];
        unsigned long long coordinate = width == 0 ? 0 : index % width;
        index = width == 0 ? 0 : index / width;
        if (in_shape[axis] != 1) mapped += coordinate * stride;
        stride *= in_shape[axis];
    }
    return mapped;
}

// The Q8_1 layout and vector-dot routines come from SGLang's GGUF AOT path.
// See sglang-gguf-common.cuh and sglang-gguf-vecdotq.cuh for upstream attribution.

extern "C" __global__ void quantize_q8_1_f32(
    const float *__restrict__ x,
    block_q8_1 *__restrict__ output,
    unsigned int columns,
    unsigned int padded_columns
) {
    unsigned int column = blockDim.x * blockIdx.x + threadIdx.x;
    if (column >= padded_columns) return;

    unsigned int vector = blockIdx.y;
    unsigned int padded_index = vector * padded_columns + column;
    unsigned int block = padded_index / QK8_1;
    unsigned int quant = padded_index % QK8_1;
    float value = column < columns ? x[vector * columns + column] : 0.0f;
    float maximum = fabsf(value);
    float sum = value;

#pragma unroll
    for (unsigned int mask = 16; mask != 0; mask >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, mask, 32));
        sum += __shfl_xor_sync(0xffffffffU, sum, mask, 32);
    }

    float scale = maximum / 127.0f;
    output[block].qs[quant] = maximum == 0.0f
        ? 0
        : (signed char)roundf(value / scale);
    if (quant == 0) {
        output[block].ds.x = __float2half(scale);
        output[block].ds.y = __float2half(sum);
    }
}

extern "C" __global__ void rms_norm_quantize_q8_1_f32(
    const float *__restrict__ x,
    const float *__restrict__ weight,
    block_q8_1 *__restrict__ output,
    unsigned int columns,
    unsigned int padded_columns,
    unsigned int weighted,
    float eps
) {
    __shared__ float warp_sums[8];
    __shared__ float inverse_rms;
    unsigned int vector = blockIdx.x;
    unsigned int lane = threadIdx.x & 31U;
    unsigned int warp = threadIdx.x >> 5;
    float square_sum = 0.0f;

    for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
        float value = x[vector * columns + column];
        square_sum += value * value;
    }
#pragma unroll
    for (unsigned int offset = 16; offset != 0; offset >>= 1) {
        square_sum += __shfl_down_sync(0xffffffffU, square_sum, offset);
    }
    if (lane == 0) warp_sums[warp] = square_sum;
    __syncthreads();
    if (warp == 0) {
        square_sum = lane < 8 ? warp_sums[lane] : 0.0f;
#pragma unroll
        for (unsigned int offset = 16; offset != 0; offset >>= 1) {
            square_sum += __shfl_down_sync(0xffffffffU, square_sum, offset);
        }
        if (lane == 0) inverse_rms = rsqrtf(square_sum / columns + eps);
    }
    __syncthreads();

    for (unsigned int base = 0; base < padded_columns; base += blockDim.x) {
        unsigned int column = base + threadIdx.x;
        float value = column < columns
            ? x[vector * columns + column] * inverse_rms
                * (weighted != 0 ? weight[column] : 1.0f)
            : 0.0f;
        float maximum = fabsf(value);
        float sum = value;
#pragma unroll
        for (unsigned int mask = 16; mask != 0; mask >>= 1) {
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, mask));
            sum += __shfl_xor_sync(0xffffffffU, sum, mask);
        }
        unsigned int block = vector * (padded_columns / QK8_1) + column / QK8_1;
        float scale = maximum / 127.0f;
        output[block].qs[lane] = maximum == 0.0f
            ? 0
            : (signed char)roundf(value / scale);
        if (lane == 0) {
            output[block].ds.x = __float2half(scale);
            output[block].ds.y = __float2half(sum);
        }
    }
}

extern "C" __global__ void residual_rms_norm_quantize_q8_1_f32(
    const float *__restrict__ branch,
    const float *__restrict__ post_weight,
    const float *__restrict__ residual,
    float *__restrict__ hidden,
    const float *__restrict__ next_weight,
    block_q8_1 *__restrict__ output,
    unsigned int columns,
    unsigned int padded_columns,
    unsigned int post_weighted,
    unsigned int next_weighted,
    float post_eps,
    float next_eps
) {
    __shared__ float warp_sums[8];
    __shared__ float inverse_rms;
    unsigned int vector = blockIdx.x;
    unsigned int lane = threadIdx.x & 31U;
    unsigned int warp = threadIdx.x >> 5;
    float square_sum = 0.0f;

    for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
        float value = branch[vector * columns + column];
        square_sum += value * value;
    }
#pragma unroll
    for (unsigned int offset = 16; offset != 0; offset >>= 1) {
        square_sum += __shfl_down_sync(0xffffffffU, square_sum, offset);
    }
    if (lane == 0) warp_sums[warp] = square_sum;
    __syncthreads();
    if (warp == 0) {
        square_sum = lane < 8 ? warp_sums[lane] : 0.0f;
#pragma unroll
        for (unsigned int offset = 16; offset != 0; offset >>= 1) {
            square_sum += __shfl_down_sync(0xffffffffU, square_sum, offset);
        }
        if (lane == 0) inverse_rms = rsqrtf(square_sum / columns + post_eps);
    }
    __syncthreads();

    square_sum = 0.0f;
    for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
        unsigned int index = vector * columns + column;
        float value = branch[index] * inverse_rms;
        if (post_weighted != 0) value *= post_weight[column];
        value = __fadd_rn(residual[index], value);
        hidden[index] = value;
        square_sum += value * value;
    }
#pragma unroll
    for (unsigned int offset = 16; offset != 0; offset >>= 1) {
        square_sum += __shfl_down_sync(0xffffffffU, square_sum, offset);
    }
    if (lane == 0) warp_sums[warp] = square_sum;
    __syncthreads();
    if (warp == 0) {
        square_sum = lane < 8 ? warp_sums[lane] : 0.0f;
#pragma unroll
        for (unsigned int offset = 16; offset != 0; offset >>= 1) {
            square_sum += __shfl_down_sync(0xffffffffU, square_sum, offset);
        }
        if (lane == 0) inverse_rms = rsqrtf(square_sum / columns + next_eps);
    }
    __syncthreads();

    for (unsigned int base = 0; base < padded_columns; base += blockDim.x) {
        unsigned int column = base + threadIdx.x;
        float value = column < columns
            ? hidden[vector * columns + column] * inverse_rms
                * (next_weighted != 0 ? next_weight[column] : 1.0f)
            : 0.0f;
        float maximum = fabsf(value);
        float sum = value;
#pragma unroll
        for (unsigned int mask = 16; mask != 0; mask >>= 1) {
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, mask));
            sum += __shfl_xor_sync(0xffffffffU, sum, mask);
        }
        unsigned int block = vector * (padded_columns / QK8_1) + column / QK8_1;
        float scale = maximum / 127.0f;
        output[block].qs[lane] = maximum == 0.0f
            ? 0
            : (signed char)roundf(value / scale);
        if (lane == 0) {
            output[block].ds.x = __float2half(scale);
            output[block].ds.y = __float2half(sum);
        }
    }
}

template <
    int qi,
    typename block_q_t,
    int vdr,
    vec_dot_q_cuda_t vec_dot_q_cuda
>
__device__ __forceinline__ void mmvq_k_f32(
    const block_q_t *__restrict__ weight,
    const block_q8_1 *__restrict__ input,
    const float *__restrict__ bias,
    float *__restrict__ output,
    unsigned int columns,
    unsigned int rows,
    unsigned int padded_columns,
    unsigned int has_bias
) {
    unsigned int row = blockIdx.x * blockDim.y + threadIdx.y;
    unsigned int vector = blockIdx.y;
    if (row >= rows) return;

    unsigned int blocks_per_row = columns / QK_K;
    unsigned int blocks_per_warp = vdr * WARP_SIZE / qi;
    float sum = 0.0f;

    for (unsigned int index = threadIdx.x / (qi / vdr);
         index < blocks_per_row;
         index += blocks_per_warp) {
        unsigned int weight_block = row * blocks_per_row + index;
        unsigned int input_block = vector * (padded_columns / QK8_1)
            + index * (QK_K / QK8_1);
        unsigned int quant = vdr * (threadIdx.x % (qi / vdr));
        sum += vec_dot_q_cuda(&weight[weight_block], &input[input_block], quant);
    }
#pragma unroll
    for (unsigned int mask = WARP_SIZE / 2; mask != 0; mask >>= 1) {
        sum += __shfl_xor_sync(0xffffffffU, sum, mask);
    }
    if (threadIdx.x == 0) {
        output[vector * rows + row] = sum + (has_bias != 0 ? bias[row] : 0.0f);
    }
}

#define DEFINE_K_MMVQ(NAME, BLOCK, QI, VDR, DOT) \
extern "C" __global__ void NAME( \
    const BLOCK *__restrict__ weight, \
    const block_q8_1 *__restrict__ input, \
    const float *__restrict__ bias, \
    float *__restrict__ output, \
    unsigned int columns, \
    unsigned int rows, \
    unsigned int padded_columns, \
    unsigned int has_bias \
) { \
    mmvq_k_f32<QI, BLOCK, VDR, DOT>( \
        weight, input, bias, output, columns, rows, padded_columns, has_bias \
    ); \
}

DEFINE_K_MMVQ(mmvq_q2_k_f32, block_q2_K, QI2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1)
DEFINE_K_MMVQ(mmvq_q3_k_f32, block_q3_K, QI3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1)
DEFINE_K_MMVQ(mmvq_q4_k_f32, block_q4_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1)
DEFINE_K_MMVQ(mmvq_q5_k_f32, block_q5_K, QI5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1)
DEFINE_K_MMVQ(mmvq_q6_k_f32, block_q6_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1)

#define DEFINE_K_MMVQ_PAIR(NAME, BLOCK, QI, VDR, DOT) \
extern "C" __global__ void NAME( \
    const BLOCK *__restrict__ weight0, \
    const BLOCK *__restrict__ weight1, \
    const block_q8_1 *__restrict__ input, \
    float *__restrict__ output0, \
    float *__restrict__ output1, \
    unsigned int columns, \
    unsigned int rows, \
    unsigned int padded_columns \
) { \
    const BLOCK *weight = blockIdx.z == 0 ? weight0 : weight1; \
    float *output = blockIdx.z == 0 ? output0 : output1; \
    mmvq_k_f32<QI, BLOCK, VDR, DOT>( \
        weight, input, (const float *)0, output, columns, rows, padded_columns, 0 \
    ); \
}

DEFINE_K_MMVQ_PAIR(
    mmvq_q2_k_f32_pair, block_q2_K, QI2_K,
    VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1)
DEFINE_K_MMVQ_PAIR(
    mmvq_q3_k_f32_pair, block_q3_K, QI3_K,
    VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1)

extern "C" __global__ void mmvq_q2_q3_k_f32_pair(
    const block_q2_K *__restrict__ weight0,
    const block_q3_K *__restrict__ weight1,
    const block_q8_1 *__restrict__ input,
    float *__restrict__ output0,
    float *__restrict__ output1,
    unsigned int columns,
    unsigned int rows,
    unsigned int padded_columns
) {
    if (blockIdx.z == 0) {
        mmvq_k_f32<QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
            weight0, input, (const float *)0, output0, columns, rows, padded_columns, 0);
    } else {
        mmvq_k_f32<QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
            weight1, input, (const float *)0, output1, columns, rows, padded_columns, 0);
    }
}

extern "C" __global__ void mmvq_q3_q2_k_f32_pair(
    const block_q3_K *__restrict__ weight0,
    const block_q2_K *__restrict__ weight1,
    const block_q8_1 *__restrict__ input,
    float *__restrict__ output0,
    float *__restrict__ output1,
    unsigned int columns,
    unsigned int rows,
    unsigned int padded_columns
) {
    if (blockIdx.z == 0) {
        mmvq_k_f32<QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
            weight0, input, (const float *)0, output0, columns, rows, padded_columns, 0);
    } else {
        mmvq_k_f32<QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
            weight1, input, (const float *)0, output1, columns, rows, padded_columns, 0);
    }
}

template <
    int qi,
    typename block_q_t,
    int vdr,
    vec_dot_q_cuda_t vec_dot_q_cuda
>
__device__ __forceinline__ void mmvq_k_f32_cooperative(
    const block_q_t *__restrict__ weight,
    const block_q8_1 *__restrict__ input,
    const float *__restrict__ bias,
    float *__restrict__ output,
    unsigned int columns,
    unsigned int rows,
    unsigned int padded_columns,
    unsigned int has_bias
) {
    unsigned int row = blockIdx.x;
    unsigned int vector = blockIdx.y;
    unsigned int warp = threadIdx.y;
    if (row >= rows) return;

    unsigned int blocks_per_row = columns / QK_K;
    unsigned int blocks_per_warp = vdr * WARP_SIZE / qi;
    float sum = 0.0f;
    for (unsigned int index = threadIdx.x / (qi / vdr) + warp * blocks_per_warp;
         index < blocks_per_row;
         index += blocks_per_warp * blockDim.y) {
        unsigned int weight_block = row * blocks_per_row + index;
        unsigned int input_block = vector * (padded_columns / QK8_1)
            + index * (QK_K / QK8_1);
        unsigned int quant = vdr * (threadIdx.x % (qi / vdr));
        sum += vec_dot_q_cuda(&weight[weight_block], &input[input_block], quant);
    }
    __shared__ float partial[3][WARP_SIZE];
    if (warp > 0) partial[warp - 1][threadIdx.x] = sum;
    __syncthreads();
    if (warp > 0) return;
#pragma unroll
    for (unsigned int other = 0; other < 3; ++other) {
        sum += partial[other][threadIdx.x];
    }
#pragma unroll
    for (unsigned int mask = WARP_SIZE / 2; mask != 0; mask >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, mask);
    }
    if (threadIdx.x == 0) {
        output[vector * rows + row] = sum + (has_bias != 0 ? bias[row] : 0.0f);
    }
}

#define COOPERATIVE_ARGS(BLOCK) \
    const BLOCK *__restrict__ weight, \
    const block_q8_1 *__restrict__ input, \
    const float *__restrict__ bias, \
    float *__restrict__ output, \
    unsigned int columns, \
    unsigned int rows, \
    unsigned int padded_columns, \
    unsigned int has_bias

extern "C" __global__ __launch_bounds__(128, 1)
void mmvq_q2_k_f32_cooperative(COOPERATIVE_ARGS(block_q2_K)) {
    mmvq_k_f32_cooperative<QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
        weight, input, bias, output, columns, rows, padded_columns, has_bias);
}
extern "C" __global__ __launch_bounds__(128, 1)
void mmvq_q3_k_f32_cooperative(COOPERATIVE_ARGS(block_q3_K)) {
    mmvq_k_f32_cooperative<QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
        weight, input, bias, output, columns, rows, padded_columns, has_bias);
}
extern "C" __global__ void mmvq_q4_k_f32_cooperative(COOPERATIVE_ARGS(block_q4_K)) {
    mmvq_k_f32_cooperative<QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
        weight, input, bias, output, columns, rows, padded_columns, has_bias);
}
extern "C" __global__ void mmvq_q5_k_f32_cooperative(COOPERATIVE_ARGS(block_q5_K)) {
    mmvq_k_f32_cooperative<QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
        weight, input, bias, output, columns, rows, padded_columns, has_bias);
}
extern "C" __global__ void mmvq_q6_k_f32_cooperative(COOPERATIVE_ARGS(block_q6_K)) {
    mmvq_k_f32_cooperative<QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
        weight, input, bias, output, columns, rows, padded_columns, has_bias);
}

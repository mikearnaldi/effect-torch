// K-quant MMQ wrappers around the pinned SGLang/llama.cpp tile implementation.

#define DEFINE_K_MMQ(NAME, QR, QI, BLOCK, NEED_SUM, ALLOCATE, LOAD, VDR, DOT) \
extern "C" __global__ void NAME( \
    const BLOCK *__restrict__ weight, \
    const block_q8_1 *__restrict__ input, \
    float *__restrict__ output, \
    unsigned int columns, \
    unsigned int rows, \
    unsigned int padded_columns, \
    unsigned int vectors \
) { \
    constexpr int MMQ_X = 64; \
    constexpr int MMQ_Y = 128; \
    constexpr int NWARPS = 8; \
    if (rows % MMQ_Y == 0) { \
        mul_mat_q< \
            float, QK_K, QR, QI, NEED_SUM, BLOCK, MMQ_X, MMQ_Y, NWARPS, \
            ALLOCATE<MMQ_Y>, LOAD<MMQ_Y, NWARPS, false>, VDR, DOT>( \
                weight, input, output, columns, rows, vectors, padded_columns, rows); \
    } else { \
        mul_mat_q< \
            float, QK_K, QR, QI, NEED_SUM, BLOCK, MMQ_X, MMQ_Y, NWARPS, \
            ALLOCATE<MMQ_Y>, LOAD<MMQ_Y, NWARPS, true>, VDR, DOT>( \
                weight, input, output, columns, rows, vectors, padded_columns, rows); \
    } \
}

DEFINE_K_MMQ(
    mmq_q2_k_f32, QR2_K, QI2_K, block_q2_K, false,
    allocate_tiles_q2_K, load_tiles_q2_K,
    VDR_Q2_K_Q8_1_MMQ, vec_dot_q2_K_q8_1_mul_mat)
DEFINE_K_MMQ(
    mmq_q3_k_f32, QR3_K, QI3_K, block_q3_K, false,
    allocate_tiles_q3_K, load_tiles_q3_K,
    VDR_Q3_K_Q8_1_MMQ, vec_dot_q3_K_q8_1_mul_mat)
DEFINE_K_MMQ(
    mmq_q4_k_f32, QR4_K, QI4_K, block_q4_K, true,
    allocate_tiles_q4_K, load_tiles_q4_K,
    VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat)
DEFINE_K_MMQ(
    mmq_q5_k_f32, QR5_K, QI5_K, block_q5_K, true,
    allocate_tiles_q5_K, load_tiles_q5_K,
    VDR_Q5_K_Q8_1_MMQ, vec_dot_q5_K_q8_1_mul_mat)
DEFINE_K_MMQ(
    mmq_q6_k_f32, QR6_K, QI6_K, block_q6_K, false,
    allocate_tiles_q6_K, load_tiles_q6_K,
    VDR_Q6_K_Q8_1_MMQ, vec_dot_q6_K_q8_1_mul_mat)

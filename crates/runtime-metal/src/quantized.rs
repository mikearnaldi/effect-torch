//! Correctness-first GGML K-quant execution. Each linear output is accumulated
//! directly from packed blocks, and embedding only visits selected packed rows.

use crate::runtime::dtype::DType;
use crate::runtime::metal::run::MetalTensor;
use effect_torch_runtime::GgmlKQuant;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearRequirements {
    pub codec: GgmlKQuant,
    pub input_shape: Box<[usize]>,
    pub output_shape: Box<[usize]>,
    pub rows: usize,
    pub columns: usize,
    pub vectors: usize,
    pub encoded_row_bytes: usize,
    pub has_bias: bool,
    pub output_bytes: usize,
    pub pipeline_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingRequirements {
    pub codec: GgmlKQuant,
    pub index_shape: Box<[usize]>,
    pub index_dtype: DType,
    pub output_shape: Box<[usize]>,
    pub rows: usize,
    pub columns: usize,
    pub indexes: usize,
    pub encoded_row_bytes: usize,
    pub output_bytes: usize,
    pub pipeline_count: usize,
}

fn checked_numel(shape: &[usize], operation: &str) -> Result<usize, String> {
    shape
        .iter()
        .try_fold(1usize, |total, &value| total.checked_mul(value))
        .ok_or_else(|| format!("{operation}: element count overflow"))
}

fn checked_bytes(elements: usize, dtype: DType, operation: &str) -> Result<usize, String> {
    elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| format!("{operation}: byte size overflow"))
}

fn block_bytes(codec: GgmlKQuant) -> usize {
    match codec {
        GgmlKQuant::Q2K => 84,
        GgmlKQuant::Q3K => 110,
        GgmlKQuant::Q4K => 144,
        GgmlKQuant::Q5K => 176,
        GgmlKQuant::Q6K => 210,
    }
}

fn rows_per_simd(codec: GgmlKQuant) -> usize {
    match codec {
        GgmlKQuant::Q2K => 2,
        GgmlKQuant::Q3K => 3,
        GgmlKQuant::Q4K | GgmlKQuant::Q6K => 2,
        GgmlKQuant::Q5K => 1,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn linear_requirements(
    input_shape: &[usize],
    input_dtype: DType,
    weight_shape: &[usize],
    weight_dtype: DType,
    bias: Option<(&[usize], DType)>,
    output_shape: &[usize],
    output_dtype: DType,
    codec: GgmlKQuant,
    logical_weight_shape: [usize; 2],
) -> Result<LinearRequirements, String> {
    let [rows, columns] = logical_weight_shape;
    if input_dtype != DType::F32 || input_shape.len() < 2 {
        return Err("quantized_linear: input must be rank >= 2 f32".to_string());
    }
    if input_shape.last() != Some(&columns) {
        return Err(format!(
            "quantized_linear: expected input trailing width {columns}, got {input_shape:?}"
        ));
    }
    checked_numel(input_shape, "quantized_linear")?;
    let vectors = checked_numel(&input_shape[..input_shape.len() - 1], "quantized_linear")?;
    let encoded_row_bytes = codec.encoded_row_bytes(columns).ok_or_else(|| {
        format!(
            "quantized_linear: logical row width {columns} is invalid for {}",
            codec.name()
        )
    })?;
    rows.checked_mul(encoded_row_bytes)
        .ok_or_else(|| "quantized_linear: packed weight byte size overflow".to_string())?;
    if weight_dtype != DType::U8 || weight_shape != [rows, encoded_row_bytes] {
        return Err(format!(
            "quantized_linear: expected exact contiguous packed {} weight [{rows}, {encoded_row_bytes}] u8",
            codec.name()
        ));
    }
    if bias.is_some_and(|(shape, dtype)| dtype != DType::F32 || shape != [rows]) {
        return Err(format!("quantized_linear: bias must be [{rows}] f32"));
    }
    let mut expected_output = input_shape.to_vec();
    *expected_output
        .last_mut()
        .expect("input rank was validated") = rows;
    if output_dtype != DType::F32 || output_shape != expected_output {
        return Err("quantized_linear: output does not match the logical node shape".to_string());
    }
    let output_elements = vectors
        .checked_mul(rows)
        .ok_or_else(|| "quantized_linear: output element count overflow".to_string())?;
    let output_bytes = checked_bytes(output_elements, DType::F32, "quantized_linear")?;
    Ok(LinearRequirements {
        codec,
        input_shape: input_shape.to_vec().into_boxed_slice(),
        output_shape: expected_output.into_boxed_slice(),
        rows,
        columns,
        vectors,
        encoded_row_bytes,
        has_bias: bias.is_some(),
        output_bytes,
        pipeline_count: usize::from(output_elements != 0),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn embedding_requirements(
    index_shape: &[usize],
    index_dtype: DType,
    weight_shape: &[usize],
    weight_dtype: DType,
    output_shape: &[usize],
    output_dtype: DType,
    codec: GgmlKQuant,
    logical_weight_shape: [usize; 2],
) -> Result<EmbeddingRequirements, String> {
    let [rows, columns] = logical_weight_shape;
    if !matches!(index_dtype, DType::U32 | DType::I64) {
        return Err(format!(
            "quantized_embedding: indexes must be u32 or i64, got {index_dtype}"
        ));
    }
    let indexes = checked_numel(index_shape, "quantized_embedding")?;
    let encoded_row_bytes = codec.encoded_row_bytes(columns).ok_or_else(|| {
        format!(
            "quantized_embedding: logical row width {columns} is invalid for {}",
            codec.name()
        )
    })?;
    rows.checked_mul(encoded_row_bytes)
        .ok_or_else(|| "quantized_embedding: packed weight byte size overflow".to_string())?;
    if weight_dtype != DType::U8 || weight_shape != [rows, encoded_row_bytes] {
        return Err(format!(
            "quantized_embedding: expected exact contiguous packed {} weight [{rows}, {encoded_row_bytes}] u8",
            codec.name()
        ));
    }
    let mut expected_output = index_shape.to_vec();
    expected_output.push(columns);
    if output_dtype != DType::F32 || output_shape != expected_output {
        return Err(
            "quantized_embedding: output does not match the logical node shape".to_string(),
        );
    }
    let output_elements = indexes
        .checked_mul(columns)
        .ok_or_else(|| "quantized_embedding: output element count overflow".to_string())?;
    let output_bytes = checked_bytes(output_elements, DType::F32, "quantized_embedding")?;
    Ok(EmbeddingRequirements {
        codec,
        index_shape: index_shape.to_vec().into_boxed_slice(),
        index_dtype,
        output_shape: expected_output.into_boxed_slice(),
        rows,
        columns,
        indexes,
        encoded_row_bytes,
        output_bytes,
        pipeline_count: 1,
    })
}

fn tensor_end(tensor: &MetalTensor, operation: &str, label: &str) -> Result<usize, String> {
    tensor
        .layout
        .offset()
        .checked_add(tensor.numel())
        .and_then(|elements| elements.checked_mul(tensor.dtype.size_in_bytes()))
        .ok_or_else(|| format!("{operation}: {label} byte range overflow"))
}

fn require_contiguous(
    tensor: &MetalTensor,
    operation: &str,
    label: &str,
    zero_offset: bool,
) -> Result<(), String> {
    if !tensor.layout.is_contiguous() || (zero_offset && tensor.layout.offset() != 0) {
        return Err(format!(
            "{operation}: {label} must be {}contiguous",
            if zero_offset { "zero-offset " } else { "" }
        ));
    }
    let end = tensor_end(tensor, operation, label)?;
    if end > tensor.buffer.size {
        return Err(format!(
            "{operation}: {label} view requires {end} bytes, buffer has {}",
            tensor.buffer.size
        ));
    }
    Ok(())
}

fn validate_linear(
    input: &MetalTensor,
    weight: &MetalTensor,
    bias: Option<&MetalTensor>,
    output: &MetalTensor,
    requirements: &LinearRequirements,
) -> Result<(), String> {
    if input.dtype != DType::F32 || input.layout.shape() != requirements.input_shape.as_ref() {
        return Err("quantized_linear: input does not match the immutable plan".to_string());
    }
    require_contiguous(input, "quantized_linear", "input", false)?;
    if weight.dtype != DType::U8
        || weight.layout.shape() != [requirements.rows, requirements.encoded_row_bytes]
    {
        return Err(format!(
            "quantized_linear: expected exact contiguous packed {} weight [{}, {}] u8",
            requirements.codec.name(),
            requirements.rows,
            requirements.encoded_row_bytes
        ));
    }
    require_contiguous(weight, "quantized_linear", "packed weight", true)?;
    if bias.is_some() != requirements.has_bias
        || bias.is_some_and(|bias| {
            bias.dtype != DType::F32
                || bias.layout.shape() != [requirements.rows]
                || !bias.layout.is_contiguous()
        })
    {
        return Err("quantized_linear: bias does not match the immutable plan".to_string());
    }
    if let Some(bias) = bias {
        require_contiguous(bias, "quantized_linear", "bias", false)?;
    }
    output.validate_destination("quantized_linear", &requirements.output_shape, DType::F32)
}

fn validate_embedding(
    indexes: &MetalTensor,
    weight: &MetalTensor,
    output: &MetalTensor,
    status: &MetalTensor,
    requirements: &EmbeddingRequirements,
) -> Result<(), String> {
    if indexes.dtype != requirements.index_dtype
        || indexes.layout.shape() != requirements.index_shape.as_ref()
    {
        return Err("quantized_embedding: indexes do not match the immutable plan".to_string());
    }
    require_contiguous(indexes, "quantized_embedding", "indexes", false)?;
    if weight.dtype != DType::U8
        || weight.layout.shape() != [requirements.rows, requirements.encoded_row_bytes]
    {
        return Err(format!(
            "quantized_embedding: expected exact contiguous packed {} weight [{}, {}] u8",
            requirements.codec.name(),
            requirements.rows,
            requirements.encoded_row_bytes
        ));
    }
    require_contiguous(weight, "quantized_embedding", "packed weight", true)?;
    output.validate_destination(
        "quantized_embedding",
        &requirements.output_shape,
        DType::F32,
    )?;
    status.validate_destination("quantized_embedding", &[1], DType::U32)
}

#[derive(Clone, Copy)]
enum KernelKind {
    Linear,
    Embedding,
}

fn codec_tag(codec: GgmlKQuant) -> u64 {
    match codec {
        GgmlKQuant::Q2K => 2,
        GgmlKQuant::Q3K => 3,
        GgmlKQuant::Q4K => 4,
        GgmlKQuant::Q5K => 5,
        GgmlKQuant::Q6K => 6,
    }
}

fn pipeline_key(kind: KernelKind, codec: GgmlKQuant, index_dtype: Option<DType>) -> u64 {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    "effect_torch_quantized_k_v2".hash(&mut hasher);
    match kind {
        KernelKind::Linear => 0u8,
        KernelKind::Embedding => 1u8,
    }
    .hash(&mut hasher);
    codec_tag(codec).hash(&mut hasher);
    index_dtype.map(|dtype| dtype as u8).hash(&mut hasher);
    hasher.finish()
}

const DECODE_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define ET_CODEC $CODEC
#define ET_BLOCK_BYTES $BLOCK_BYTES
#define ET_BLOCK_VALUES 256ul

inline float et_fp16_at(device const uchar* block, uint offset) {
    const ushort bits = ushort(block[offset]) | (ushort(block[offset + 1]) << 8);
    return float(as_type<half>(bits));
}

inline int et_signed_byte(uchar value) {
    return value < 128 ? int(value) : int(value) - 256;
}

inline float et_decode_k(device const uchar* block, uint lane) {
#if ET_CODEC == 2
    const uint group = lane / 16;
    const uint index = lane % 16;
    const uint half_index = group / 8;
    const uint within = group % 8;
    const uint shift = (within / 2) * 2;
    const uint quant_offset = (within % 2) * 16;
    const uchar scale = block[group];
    const uchar quant = (block[16 + half_index * 32 + quant_offset + index] >> shift) & 3;
    const float d = et_fp16_at(block, 80);
    const float dmin = et_fp16_at(block, 82);
    return d * float(scale & 15) * float(quant) - dmin * float(scale >> 4);
#elif ET_CODEC == 3
    const uint group = lane / 16;
    const uint index = lane % 16;
    const uint half_index = group / 8;
    const uint within = group % 8;
    const uint quant_lane = within / 2;
    const uint shift = quant_lane * 2;
    const uint quant_offset = (within % 2) * 16;
    const uchar low = group < 8 ? (block[96 + group] & 15) : (block[96 + group - 8] >> 4);
    const uchar high = (block[104 + group % 4] >> (2 * (group / 4))) & 3;
    const int scale = int(low | (high << 4)) - 32;
    const int low_quant = int((block[32 + half_index * 32 + quant_offset + index] >> shift) & 3);
    const uchar mask = uchar(1u << (half_index * 4 + quant_lane));
    const int high_quant = (block[quant_offset + index] & mask) == 0 ? 4 : 0;
    return et_fp16_at(block, 108) * float(scale) * float(low_quant - high_quant);
#elif ET_CODEC == 4 || ET_CODEC == 5
    const uint group = lane / 32;
    const uint index = lane % 32;
    const uint pair = group / 2;
    const uint side = group % 2;
    uint scale;
    uint minimum;
    if (group < 4) {
        scale = uint(block[4 + group] & 63);
        minimum = uint(block[8 + group] & 63);
    } else {
        scale = uint(block[8 + group] & 15) | (uint(block[group]) >> 6) << 4;
        minimum = uint(block[8 + group] >> 4) | (uint(block[4 + group]) >> 6) << 4;
    }
#if ET_CODEC == 4
    const uchar packed = block[16 + pair * 32 + index];
    const uint quant = side == 0 ? uint(packed & 15) : uint(packed >> 4);
#else
    const uchar packed = block[48 + pair * 32 + index];
    const uint low_quant = side == 0 ? uint(packed & 15) : uint(packed >> 4);
    const uchar mask = uchar(1u << (pair * 2 + side));
    const uint quant = low_quant + ((block[16 + index] & mask) == 0 ? 0u : 16u);
#endif
    return et_fp16_at(block, 0) * float(scale) * float(quant)
        - et_fp16_at(block, 2) * float(minimum);
#else
    const uint half_index = lane / 128;
    const uint within = lane % 128;
    const uint quarter = within / 32;
    const uint index = within % 32;
    const uint low_index = half_index * 64 + index + ((quarter & 1) == 0 ? 0 : 32);
    const uint low_shift = quarter < 2 ? 0 : 4;
    const uint high_shift = quarter * 2;
    const uint quant = ((uint(block[low_index]) >> low_shift) & 15)
        | (((uint(block[128 + half_index * 32 + index]) >> high_shift) & 3) << 4);
    const uint scale_lane = index / 16;
    const uint scale_index = half_index * 8 + scale_lane + quarter * 2;
    const int scale = et_signed_byte(block[192 + scale_index]);
    return et_fp16_at(block, 208) * float(scale) * float(int(quant) - 32);
#endif
}
"#;

const LINEAR_SOURCE: &str = r#"
#define ET_ROWS_PER_SIMD $ROWS_PER_SIMD
#define ET_SIMD_GROUPS 2ul
#define ET_UNROLL _Pragma("clang loop unroll(full)")

struct et_block_q2_k {
    uchar scales[16];
    uchar qs[64];
    half d;
    half dmin;
};

struct et_block_q3_k {
    uchar hmask[32];
    uchar qs[64];
    uchar scales[12];
    half d;
};

struct et_block_q4_k {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[128];
};

struct et_block_q5_k {
    half d;
    half dmin;
    uchar scales[12];
    uchar qh[32];
    uchar qs[128];
};

struct et_block_q6_k {
    uchar ql[128];
    uchar qh[64];
    char scales[16];
    half d;
};

static_assert(sizeof(et_block_q2_k) == 84, "invalid Q2_K block layout");
static_assert(sizeof(et_block_q3_k) == 110, "invalid Q3_K block layout");
static_assert(sizeof(et_block_q4_k) == 144, "invalid Q4_K block layout");
static_assert(sizeof(et_block_q5_k) == 176, "invalid Q5_K block layout");
static_assert(sizeof(et_block_q6_k) == 210, "invalid Q6_K block layout");

kernel void et_quantized_linear(
    device const float* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant ulong& vectors [[buffer(4)]],
    constant ulong& rows [[buffer(5)]],
    constant ulong& columns [[buffer(6)]],
    constant ulong& encoded_row_bytes [[buffer(7)]],
    constant uint& has_bias [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    const ulong vector = ulong(group.y);
    const ulong first_row = (ulong(group.x) * ET_SIMD_GROUPS + ulong(simd_group)) * ET_ROWS_PER_SIMD;
    if (vector >= vectors || first_row >= rows) {
        return;
    }
    const ulong blocks = columns / ET_BLOCK_VALUES;
    device const float* y = input + vector * columns;
    float sums[ET_ROWS_PER_SIMD] = { 0.0f };

#if ET_CODEC == 2
    const ushort ix = lane / 8;
    const ushort it = lane % 8;
    const ushort iq = it / 4;
    const ushort ir = it % 4;
    const ushort scale_offset = ir / 2;
    device const float* y4 = y + ulong(ix) * ET_BLOCK_VALUES + 128ul * iq + 8ul * ir;

    const uint packed_blocks = uint(blocks);
    for (uint block_index = ix; block_index < packed_blocks; block_index += 4) {
        float values[32];
        float4 input_sums = 0.0f;
        for (ushort i = 0; i < 8; ++i) {
            values[i] = y4[i];
            values[i + 8] = y4[i + 32];
            values[i + 16] = y4[i + 64];
            values[i + 24] = y4[i + 96];
            input_sums += float4(values[i], values[i + 8], values[i + 16], values[i + 24]);
        }
        for (ushort row_offset = 0; row_offset < ET_ROWS_PER_SIMD && first_row + row_offset < rows; ++row_offset) {
            device const et_block_q2_k* block = reinterpret_cast<device const et_block_q2_k*>(
                weight + (first_row + row_offset) * encoded_row_bytes + block_index * ET_BLOCK_BYTES);
            device const uchar* scales = block->scales + 8 * iq + scale_offset;
            device const ushort* quants = reinterpret_cast<device const ushort*>(block->qs) + 16 * iq + 4 * ir;
            float4 low = 0.0f;
            float4 high = 0.0f;
            ET_UNROLL for (ushort i = 0; i < 8; i += 2) {
                low[0] += values[i] * (quants[i / 2] & 0x0003);
                high[0] += values[i + 1] * (quants[i / 2] & 0x0300);
                low[1] += values[i + 8] * (quants[i / 2] & 0x000c);
                high[1] += values[i + 9] * (quants[i / 2] & 0x0c00);
                low[2] += values[i + 16] * (quants[i / 2] & 0x0030);
                high[2] += values[i + 17] * (quants[i / 2] & 0x3000);
                low[3] += values[i + 24] * (quants[i / 2] & 0x00c0);
                high[3] += values[i + 25] * (quants[i / 2] & 0xc000);
            }
            const float d = float(block->d);
            const float dmin = float(block->dmin) / 16.0f;
            sums[row_offset] += d * (
                (low[0] + high[0] / 256.0f) * (scales[0] & 0x0f) +
                (low[1] + high[1] / 256.0f) * (scales[2] & 0x0f) / 4.0f +
                (low[2] + high[2] / 256.0f) * (scales[4] & 0x0f) / 16.0f +
                (low[3] + high[3] / 256.0f) * (scales[6] & 0x0f) / 64.0f) -
                dmin * (input_sums[0] * (scales[0] & 0xf0) +
                        input_sums[1] * (scales[2] & 0xf0) +
                        input_sums[2] * (scales[4] & 0xf0) +
                        input_sums[3] * (scales[6] & 0xf0));
        }
        y4 += 4ul * ET_BLOCK_VALUES;
    }
#elif ET_CODEC == 3
    const ushort tid = (lane % 4) + 4 * (lane / 16);
    const ushort ix = (lane % 16) / 4;
    const ushort ip = tid / 4;
    const ushort il = 2 * ((tid % 4) / 2);
    const ushort ir = tid % 2;
    const ushort l0 = 8 * ir;
    const ushort4 high_masks[4] = {
        ushort4(0x0001, 0x0100, 0x0002, 0x0200),
        ushort4(0x0004, 0x0400, 0x0008, 0x0800),
        ushort4(0x0010, 0x1000, 0x0020, 0x2000),
        ushort4(0x0040, 0x4000, 0x0080, 0x8000)
    };
    const int4 quant_masks[2] = {
        int4(0x0003, 0x0300, 0x000c, 0x0c00),
        int4(0x0030, 0x3000, 0x00c0, 0xc000)
    };
    const ushort4 hm = high_masks[2 * ip + il / 2];
    const ushort scale_shift_low = 4 * ip;
    const ushort scale_shift_high = scale_shift_low + il;
    const ushort quant_offset = 32 * ip + l0;
    const ushort input_offset = 128 * ip + 32 * il + l0;
    const float absent_low = il == 0 ? 4.0f : 64.0f;
    const float absent_high = 4.0f * absent_low;
    device const float* y1 = y + ulong(ix) * ET_BLOCK_VALUES + input_offset;
    float sums_high[ET_ROWS_PER_SIMD] = { 0.0f };

    const uint q3_blocks = uint(blocks);
    for (uint block_index = ix; block_index < q3_blocks; block_index += 4) {
        float values[32];
        for (ushort i = 0; i < 8; ++i) {
            values[i] = y1[i];
            values[i + 8] = y1[i + 16];
            values[i + 16] = y1[i + 32];
            values[i + 24] = y1[i + 48];
        }
        for (ushort row_offset = 0; row_offset < ET_ROWS_PER_SIMD && first_row + row_offset < rows; ++row_offset) {
            device const et_block_q3_k* block = reinterpret_cast<device const et_block_q3_k*>(
                weight + (first_row + row_offset) * encoded_row_bytes + block_index * ET_BLOCK_BYTES);
            device const ushort* quants = reinterpret_cast<device const ushort*>(block->qs + quant_offset);
            device const ushort* high = reinterpret_cast<device const ushort*>(block->hmask + l0);
            device const ushort* packed_scales = reinterpret_cast<device const ushort*>(block->scales);
            uint packed;
            uint scale_high;
            thread ushort* packed16 = reinterpret_cast<thread ushort*>(&packed);
            thread const char* scales = reinterpret_cast<thread const char*>(&packed);
            packed16[0] = packed_scales[4];
            packed16[1] = packed_scales[5];
            scale_high = ((packed >> scale_shift_high) << 4) & 0x30303030;
            packed16[0] = packed_scales[il];
            packed16[1] = packed_scales[il + 1];
            packed = ((packed >> scale_shift_low) & 0x0f0f0f0f) | scale_high;

            float a0 = 0.0f, a1 = 0.0f, absent0 = 0.0f;
            float a2 = 0.0f, a3 = 0.0f, absent1 = 0.0f;
            ET_UNROLL for (ushort i = 0; i < 8; i += 2) {
                const int q = quants[i / 2];
                const ushort h = high[i / 2];
                const float4 dotted = float4(values[i], values[i + 1], values[i + 16], values[i + 17]) * float4(
                    q & quant_masks[il / 2][0],
                    q & quant_masks[il / 2][1],
                    q & quant_masks[il / 2][2],
                    q & quant_masks[il / 2][3]);
                a0 += dotted[0];
                a1 += dotted[1];
                a2 += dotted[2];
                a3 += dotted[3];
                absent0 += (h & hm[0] ? 0.0f : values[i]) +
                           (h & hm[1] ? 0.0f : values[i + 1]);
                absent1 += (h & hm[2] ? 0.0f : values[i + 16]) +
                           (h & hm[3] ? 0.0f : values[i + 17]);
            }
            const float d = float(block->d);
            float dot_low = d * (a0 + a1 / 256.0f - absent0 * absent_low);
            float dot_high = d * (a2 + a3 / 256.0f - absent1 * absent_high);
            sums[row_offset] += dot_low * (scales[0] - 32);
            sums_high[row_offset] += dot_high * (scales[2] - 32);

            a0 = a1 = absent0 = a2 = a3 = absent1 = 0.0f;
            for (ushort i = 0; i < 8; i += 2) {
                const int q = quants[i / 2 + 8];
                a0 += values[i + 8] * (q & quant_masks[il / 2][0]);
                a1 += values[i + 9] * (q & quant_masks[il / 2][1]);
                absent0 += (high[i / 2 + 8] & hm[0] ? 0.0f : values[i + 8]) +
                           (high[i / 2 + 8] & hm[1] ? 0.0f : values[i + 9]);
                a2 += values[i + 24] * (q & quant_masks[il / 2][2]);
                a3 += values[i + 25] * (q & quant_masks[il / 2][3]);
                absent1 += (high[i / 2 + 8] & hm[2] ? 0.0f : values[i + 24]) +
                           (high[i / 2 + 8] & hm[3] ? 0.0f : values[i + 25]);
            }
            dot_low = d * (a0 + a1 / 256.0f - absent0 * absent_low);
            dot_high = d * (a2 + a3 / 256.0f - absent1 * absent_high);
            sums[row_offset] += dot_low * (scales[1] - 32);
            sums_high[row_offset] += dot_high * (scales[3] - 32);
        }
        y1 += 4ul * ET_BLOCK_VALUES;
    }
    for (ushort row_offset = 0; row_offset < ET_ROWS_PER_SIMD; ++row_offset) {
        sums[row_offset] = (sums[row_offset] + 0.25f * sums_high[row_offset]) / float(1 << (2 * il));
    }
#elif ET_CODEC == 4
    constexpr ushort scale_mask = 0x3f3f;
    constexpr ushort low_mask = 0x0f0f;
    constexpr ushort high_mask = 0xc0c0;
    const ushort ix = lane / 8;
    const ushort it = lane % 8;
    const ushort iq = it / 4;
    const ushort ir = it % 4;
    device const float* y4 = y + ulong(ix) * ET_BLOCK_VALUES + 64ul * iq + 8ul * ir;

    const uint packed_blocks = uint(blocks);
    for (uint block_index = ix; block_index < packed_blocks; block_index += 4) {
        float low_values[16];
        float high_values[16];
        float4 input_sums = 0.0f;
        for (ushort i = 0; i < 8; ++i) {
            low_values[i] = y4[i];
            low_values[i + 8] = y4[i + 32];
            high_values[i] = y4[i + 128];
            high_values[i + 8] = y4[i + 160];
            input_sums += float4(low_values[i], low_values[i + 8], high_values[i], high_values[i + 8]);
        }
        for (ushort row_offset = 0; row_offset < ET_ROWS_PER_SIMD && first_row + row_offset < rows; ++row_offset) {
            device const et_block_q4_k* block = reinterpret_cast<device const et_block_q4_k*>(
                weight + (first_row + row_offset) * encoded_row_bytes + block_index * ET_BLOCK_BYTES);
            device const ushort* scales = reinterpret_cast<device const ushort*>(block->scales) + iq;
            device const ushort* q1 = reinterpret_cast<device const ushort*>(block->qs) + 16 * iq + 4 * ir;
            device const ushort* q2 = q1 + 32;
            ushort unpacked[4];
            thread const uchar* scale_bytes = reinterpret_cast<thread const uchar*>(unpacked);
            unpacked[0] = scales[0] & scale_mask;
            unpacked[1] = scales[2] & scale_mask;
            unpacked[2] = ((scales[4] >> 0) & low_mask) | ((scales[0] & high_mask) >> 2);
            unpacked[3] = ((scales[4] >> 4) & low_mask) | ((scales[2] & high_mask) >> 2);
            float4 dot_low = 0.0f;
            float4 dot_high = 0.0f;
            ET_UNROLL for (ushort i = 0; i < 4; ++i) {
                dot_low[0] += low_values[2 * i] * (q1[i] & 0x000f);
                dot_low[1] += low_values[2 * i + 1] * (q1[i] & 0x0f00);
                dot_low[2] += low_values[2 * i + 8] * (q1[i] & 0x00f0);
                dot_low[3] += low_values[2 * i + 9] * (q1[i] & 0xf000);
                dot_high[0] += high_values[2 * i] * (q2[i] & 0x000f);
                dot_high[1] += high_values[2 * i + 1] * (q2[i] & 0x0f00);
                dot_high[2] += high_values[2 * i + 8] * (q2[i] & 0x00f0);
                dot_high[3] += high_values[2 * i + 9] * (q2[i] & 0xf000);
            }
            sums[row_offset] += float(block->d) * (
                (dot_low[0] + dot_low[1] / 256.0f) * scale_bytes[0] +
                (dot_low[2] + dot_low[3] / 256.0f) * scale_bytes[1] / 16.0f +
                (dot_high[0] + dot_high[1] / 256.0f) * scale_bytes[4] +
                (dot_high[2] + dot_high[3] / 256.0f) * scale_bytes[5] / 16.0f) -
                float(block->dmin) * (input_sums[0] * scale_bytes[2] + input_sums[1] * scale_bytes[3] +
                                      input_sums[2] * scale_bytes[6] + input_sums[3] * scale_bytes[7]);
        }
        y4 += 4ul * ET_BLOCK_VALUES;
    }
#elif ET_CODEC == 5
    constexpr ushort scale_mask = 0x3f3f;
    constexpr ushort low_mask = 0x0f0f;
    constexpr ushort high_mask = 0xc0c0;
    const ushort tid = lane / 4;
    const ushort ix = lane % 4;
    const ushort iq = tid / 4;
    const ushort ir = tid % 4;
    const ushort l0 = 8 * ir;
    const ushort quant_offset = 32 * iq + l0;
    const ushort input_offset = 64 * iq + l0;
    const uchar hm0 = 1u << (2 * iq);
    const uchar hm1 = hm0 << 1;
    const uchar hm2 = hm0 << 4;
    const uchar hm3 = hm1 << 4;
    device const float* y1 = y + ulong(ix) * ET_BLOCK_VALUES + input_offset;

    const uint packed_blocks = uint(blocks);
    for (uint block_index = ix; block_index < packed_blocks; block_index += 4) {
        float low_values[16];
        float high_values[16];
        float4 input_sums = 0.0f;
        device const float* y2 = y1 + 128;
        for (ushort i = 0; i < 8; ++i) {
            low_values[i] = y1[i];
            low_values[i + 8] = y1[i + 32];
            high_values[i] = y2[i];
            high_values[i + 8] = y2[i + 32];
            input_sums += float4(low_values[i], low_values[i + 8], high_values[i], high_values[i + 8]);
        }
        for (ushort row_offset = 0; row_offset < ET_ROWS_PER_SIMD && first_row + row_offset < rows; ++row_offset) {
            device const et_block_q5_k* block = reinterpret_cast<device const et_block_q5_k*>(
                weight + (first_row + row_offset) * encoded_row_bytes + block_index * ET_BLOCK_BYTES);
            device const uchar* q1 = block->qs + quant_offset;
            device const uchar* q2 = q1 + 64;
            device const uchar* qh = block->qh + l0;
            device const ushort* scales = reinterpret_cast<device const ushort*>(block->scales) + iq;
            ushort unpacked[4];
            thread const uchar* scale_bytes = reinterpret_cast<thread const uchar*>(unpacked);
            unpacked[0] = scales[0] & scale_mask;
            unpacked[1] = scales[2] & scale_mask;
            unpacked[2] = ((scales[4] >> 0) & low_mask) | ((scales[0] & high_mask) >> 2);
            unpacked[3] = ((scales[4] >> 4) & low_mask) | ((scales[2] & high_mask) >> 2);
            float4 low_dot = 0.0f;
            float4 high_dot = 0.0f;
            ET_UNROLL for (ushort i = 0; i < 8; ++i) {
                const uchar high = qh[i];
                low_dot += float4(
                    low_values[i] * (q1[i] & 0x0f), low_values[i + 8] * (q1[i] & 0xf0),
                    high_values[i] * (q2[i] & 0x0f), high_values[i + 8] * (q2[i] & 0xf0));
                high_dot += float4(
                    high & hm0 ? low_values[i] : 0.0f, high & hm1 ? low_values[i + 8] : 0.0f,
                    high & hm2 ? high_values[i] : 0.0f, high & hm3 ? high_values[i + 8] : 0.0f);
            }
            sums[row_offset] += float(block->d) * (
                scale_bytes[0] * (low_dot[0] + 16.0f * high_dot[0]) +
                scale_bytes[1] * (low_dot[1] / 16.0f + 16.0f * high_dot[1]) +
                scale_bytes[4] * (low_dot[2] + 16.0f * high_dot[2]) +
                scale_bytes[5] * (low_dot[3] / 16.0f + 16.0f * high_dot[3])) -
                float(block->dmin) * (input_sums[0] * scale_bytes[2] + input_sums[1] * scale_bytes[3] +
                                      input_sums[2] * scale_bytes[6] + input_sums[3] * scale_bytes[7]);
        }
        y1 += 4ul * ET_BLOCK_VALUES;
    }
#else
    const ushort tid = lane / 2;
    const ushort ix = lane % 2;
    const ushort ip = tid / 8;
    const ushort il = tid % 8;
    const ushort l0 = 4 * il;
    const ushort scale_offset = 8 * ip + l0 / 16;
    const ushort input_offset = 128 * ip + l0;
    const ushort low_offset = 64 * ip + l0;
    const ushort high_offset = 32 * ip + l0;

    const uint packed_blocks = uint(blocks);
    for (uint block_index = ix; block_index < packed_blocks; block_index += 2) {
        float values[16];
        device const float* block_input = y + block_index * ET_BLOCK_VALUES + input_offset;
        for (ushort i = 0; i < 4; ++i) {
            values[4 * i] = block_input[i];
            values[4 * i + 1] = block_input[i + 32];
            values[4 * i + 2] = block_input[i + 64];
            values[4 * i + 3] = block_input[i + 96];
        }
        for (ushort row_offset = 0; row_offset < ET_ROWS_PER_SIMD && first_row + row_offset < rows; ++row_offset) {
            device const et_block_q6_k* block = reinterpret_cast<device const et_block_q6_k*>(
                weight + (first_row + row_offset) * encoded_row_bytes + block_index * ET_BLOCK_BYTES);
            device const uchar* q1 = block->ql + low_offset;
            device const uchar* q2 = q1 + 32;
            device const uchar* qh = block->qh + high_offset;
            device const char* scales = block->scales + scale_offset;
            float4 dot = 0.0f;
            ET_UNROLL for (ushort i = 0; i < 4; ++i) {
                dot[0] += values[4 * i] * (int((q1[i] & 0x0f) | ((qh[i] & 0x03) << 4)) - 32);
                dot[1] += values[4 * i + 1] * (int((q2[i] & 0x0f) | ((qh[i] & 0x0c) << 2)) - 32);
                dot[2] += values[4 * i + 2] * (int((q1[i] >> 4) | (qh[i] & 0x30)) - 32);
                dot[3] += values[4 * i + 3] * (int((q2[i] >> 4) | ((qh[i] & 0xc0) >> 2)) - 32);
            }
            sums[row_offset] += float(block->d) * (dot[0] * scales[0] + dot[1] * scales[2] +
                                                    dot[2] * scales[4] + dot[3] * scales[6]);
        }
    }
#endif

    for (ushort row_offset = 0; row_offset < ET_ROWS_PER_SIMD && first_row + row_offset < rows; ++row_offset) {
        const float sum = simd_sum(sums[row_offset]);
        if (lane == 0) {
            const ulong row = first_row + row_offset;
            output[vector * rows + row] = sum + (has_bias != 0 ? bias[row] : 0.0f);
        }
    }
}
"#;

const EMBEDDING_SOURCE: &str = r#"
#define ET_INDEX $INDEX_TYPE

kernel void et_quantized_embedding(
    device const ET_INDEX* indexes [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant ulong& index_count [[buffer(4)]],
    constant ulong& rows [[buffer(5)]],
    constant ulong& columns [[buffer(6)]],
    constant ulong& encoded_row_bytes [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
    for (ulong position = 0; position < index_count; ++position) {
        const long signed_row = long(indexes[position]);
        if (signed_row < 0 || ulong(signed_row) >= rows) {
            if (tid == 0) {
                atomic_store_explicit(status, 1u, memory_order_relaxed);
            }
            continue;
        }
        const ulong row = ulong(signed_row);
        for (ulong column = ulong(tid); column < columns; column += 256ul) {
            const ulong block_index = column / ET_BLOCK_VALUES;
            const uint lane = uint(column % ET_BLOCK_VALUES);
            device const uchar* block = weight + row * encoded_row_bytes + block_index * ET_BLOCK_BYTES;
            output[position * columns + column] = et_decode_k(block, lane);
        }
    }
}
"#;

fn source(kind: KernelKind, codec: GgmlKQuant, index_dtype: Option<DType>) -> String {
    let kernel = match kind {
        KernelKind::Linear => LINEAR_SOURCE,
        KernelKind::Embedding => EMBEDDING_SOURCE,
    };
    let mut source = String::with_capacity(DECODE_SOURCE.len() + kernel.len());
    source.push_str(DECODE_SOURCE);
    source.push_str(kernel);
    source = source.replace("$CODEC", &codec_tag(codec).to_string());
    source = source.replace("$BLOCK_BYTES", &block_bytes(codec).to_string());
    source = source.replace("$ROWS_PER_SIMD", &rows_per_simd(codec).to_string());
    if matches!(kind, KernelKind::Embedding) {
        source = source.replace(
            "$INDEX_TYPE",
            match index_dtype {
                Some(DType::U32) => "uint",
                Some(DType::I64) => "long",
                _ => unreachable!("embedding requirements validate index dtype"),
            },
        );
    }
    source
}

fn pipeline(
    kind: KernelKind,
    codec: GgmlKQuant,
    index_dtype: Option<DType>,
) -> Result<crate::device::Pipeline, String> {
    let name = match kind {
        KernelKind::Linear => "et_quantized_linear",
        KernelKind::Embedding => "et_quantized_embedding",
    };
    crate::device::MetalDevice::get().compile_lazy(
        pipeline_key(kind, codec, index_dtype),
        name,
        || source(kind, codec, index_dtype),
    )
}

fn cached_pipeline(
    kind: KernelKind,
    codec: GgmlKQuant,
    index_dtype: Option<DType>,
) -> Result<crate::device::Pipeline, String> {
    crate::device::MetalDevice::get()
        .pipeline_cached(pipeline_key(kind, codec, index_dtype))
        .ok_or_else(|| {
            format!(
                "quantized: exact {} {} pipeline is not warm",
                codec.name(),
                match kind {
                    KernelKind::Linear => "linear",
                    KernelKind::Embedding => "embedding",
                }
            )
        })
}

pub fn warm_linear_exact(requirements: &LinearRequirements) -> Result<(), String> {
    if requirements.pipeline_count != 0 {
        use objc2_metal::MTLComputePipelineState;
        let pipeline = pipeline(KernelKind::Linear, requirements.codec, None)?;
        if pipeline.as_raw().threadExecutionWidth() != 32
            || pipeline.as_raw().maxTotalThreadsPerThreadgroup() < 64
        {
            return Err(
                "quantized_linear: Metal pipeline requires SIMD width 32 and 64-thread groups"
                    .to_string(),
            );
        }
    }
    Ok(())
}

pub fn warm_embedding_exact(requirements: &EmbeddingRequirements) -> Result<(), String> {
    pipeline(
        KernelKind::Embedding,
        requirements.codec,
        Some(requirements.index_dtype),
    )?;
    Ok(())
}

pub fn linear_into(
    input: &MetalTensor,
    weight: &MetalTensor,
    bias: Option<&MetalTensor>,
    output: &MetalTensor,
    requirements: &LinearRequirements,
) -> Result<(), String> {
    use crate::runtime::metal::device::{set_buffer, set_bytes, MetalDevice};
    use objc2_metal::MTLComputeCommandEncoder;

    validate_linear(input, weight, bias, output, requirements)?;
    if requirements.pipeline_count == 0 {
        return Ok(());
    }
    let pipeline = cached_pipeline(KernelKind::Linear, requirements.codec, None)?;
    let has_bias = u32::from(requirements.has_bias);
    let vectors = requirements.vectors as u64;
    let rows = requirements.rows as u64;
    let columns = requirements.columns as u64;
    let encoded_row_bytes = requirements.encoded_row_bytes as u64;
    let rows_per_threadgroup = rows_per_simd(requirements.codec) * 2;
    MetalDevice::get().with_encoder(|encoder| {
        encoder.setComputePipelineState(pipeline.as_raw());
        set_buffer(encoder, 0, &input.buffer, input.layout.offset() * 4);
        set_buffer(encoder, 1, &weight.buffer, 0);
        let bias = bias.unwrap_or(output);
        set_buffer(encoder, 2, &bias.buffer, bias.layout.offset() * 4);
        set_buffer(encoder, 3, &output.buffer, output.layout.offset() * 4);
        set_bytes(encoder, 4, &vectors);
        set_bytes(encoder, 5, &rows);
        set_bytes(encoder, 6, &columns);
        set_bytes(encoder, 7, &encoded_row_bytes);
        set_bytes(encoder, 8, &has_bias);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MetalDevice::grid(
                requirements.rows.div_ceil(rows_per_threadgroup),
                requirements.vectors,
                1,
            ),
            MetalDevice::grid(64, 1, 1),
        );
    });
    Ok(())
}

pub fn embedding_into(
    indexes: &MetalTensor,
    weight: &MetalTensor,
    output: &MetalTensor,
    status: &MetalTensor,
    requirements: &EmbeddingRequirements,
) -> Result<(), String> {
    use crate::runtime::metal::device::{set_buffer, set_bytes, MetalDevice};
    use objc2_metal::MTLComputeCommandEncoder;

    validate_embedding(indexes, weight, output, status, requirements)?;
    unsafe {
        status
            .buffer
            .contents_ptr()
            .cast::<u32>()
            .add(status.layout.offset())
            .write(0);
    }
    let pipeline = cached_pipeline(
        KernelKind::Embedding,
        requirements.codec,
        Some(requirements.index_dtype),
    )?;
    let index_count = requirements.indexes as u64;
    let rows = requirements.rows as u64;
    let columns = requirements.columns as u64;
    let encoded_row_bytes = requirements.encoded_row_bytes as u64;
    MetalDevice::get().with_encoder(|encoder| {
        encoder.setComputePipelineState(pipeline.as_raw());
        set_buffer(
            encoder,
            0,
            &indexes.buffer,
            indexes.layout.offset() * indexes.dtype.size_in_bytes(),
        );
        set_buffer(encoder, 1, &weight.buffer, 0);
        set_buffer(encoder, 2, &output.buffer, output.layout.offset() * 4);
        set_buffer(encoder, 3, &status.buffer, status.layout.offset() * 4);
        set_bytes(encoder, 4, &index_count);
        set_bytes(encoder, 5, &rows);
        set_bytes(encoder, 6, &columns);
        set_bytes(encoder, 7, &encoded_row_bytes);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MetalDevice::grid(1, 1, 1),
            MetalDevice::grid(256, 1, 1),
        );
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;
    use effect_torch_runtime::Layout;
    use std::sync::Arc;

    #[test]
    fn requirements_reject_malformed_geometry_and_report_exact_resources() {
        let linear = linear_requirements(
            &[2, 256],
            DType::F32,
            &[3, 144],
            DType::U8,
            Some((&[3], DType::F32)),
            &[2, 3],
            DType::F32,
            GgmlKQuant::Q4K,
            [3, 256],
        )
        .unwrap();
        assert_eq!(linear.output_bytes, 2 * 3 * 4);
        assert_eq!(linear.pipeline_count, 1);

        let embedding = embedding_requirements(
            &[2],
            DType::I64,
            &[3, 176],
            DType::U8,
            &[2, 256],
            DType::F32,
            GgmlKQuant::Q5K,
            [3, 256],
        )
        .unwrap();
        assert_eq!(embedding.output_bytes, 2 * 256 * 4);
        assert_eq!(embedding.pipeline_count, 1);

        assert!(linear_requirements(
            &[1, 255],
            DType::F32,
            &[1, 0],
            DType::U8,
            None,
            &[1, 1],
            DType::F32,
            GgmlKQuant::Q2K,
            [1, 255],
        )
        .unwrap_err()
        .contains("logical row width"));
        assert!(embedding_requirements(
            &[1],
            DType::U32,
            &[1, 143],
            DType::U8,
            &[1, 256],
            DType::F32,
            GgmlKQuant::Q4K,
            [1, 256],
        )
        .unwrap_err()
        .contains("exact contiguous packed"));
    }

    #[test]
    fn cache_keys_are_operation_codec_and_index_type_specific() {
        assert_ne!(
            pipeline_key(KernelKind::Linear, GgmlKQuant::Q2K, None),
            pipeline_key(KernelKind::Linear, GgmlKQuant::Q3K, None)
        );
        assert_ne!(
            pipeline_key(KernelKind::Linear, GgmlKQuant::Q4K, None),
            pipeline_key(KernelKind::Embedding, GgmlKQuant::Q4K, Some(DType::U32))
        );
        assert_ne!(
            pipeline_key(KernelKind::Embedding, GgmlKQuant::Q6K, Some(DType::U32)),
            pipeline_key(KernelKind::Embedding, GgmlKQuant::Q6K, Some(DType::I64))
        );
    }

    #[test]
    fn execution_rejects_nonzero_offset_packed_rows() {
        let requirements = linear_requirements(
            &[1, 256],
            DType::F32,
            &[1, 144],
            DType::U8,
            None,
            &[1, 1],
            DType::F32,
            GgmlKQuant::Q4K,
            [1, 256],
        )
        .unwrap();
        let device = MetalDevice::get();
        let input = MetalTensor::from_f32(device, vec![0.0; 256], vec![1, 256]);
        let weight = MetalTensor {
            buffer: device.upload_bytes(&[0; 145]),
            layout: Layout::new(vec![1, 144], vec![144, 1], 1),
            dtype: DType::U8,
        };
        let output = MetalTensor {
            buffer: Arc::new(crate::device::Buffer::suballoc(&device.alloc_raw(4), 0, 4)),
            layout: Layout::contiguous(vec![1, 1]),
            dtype: DType::F32,
        };
        let error = linear_into(&input, &weight, None, &output, &requirements).unwrap_err();
        assert!(error.contains("packed weight must be zero-offset contiguous"));
    }

    #[test]
    #[ignore = "manual decode kernel bandwidth probe"]
    fn kquant_decode_bandwidth_probe() {
        for (codec, rows, columns, block_bytes) in [
            (GgmlKQuant::Q3K, 6656usize, 4096usize, 110usize),
            (GgmlKQuant::Q3K, 19968, 6656, 110),
            (GgmlKQuant::Q3K, 6656, 19968, 110),
            (GgmlKQuant::Q2K, 4096, 6656, 84),
            (GgmlKQuant::Q2K, 19968, 6656, 84),
            (GgmlKQuant::Q4K, 6656, 4096, 144),
            (GgmlKQuant::Q4K, 256, 6656, 144),
            (GgmlKQuant::Q6K, 202048, 6656, 210),
        ] {
            let encoded_row_bytes = columns / 256 * block_bytes;
            let requirements = linear_requirements(
                &[1, columns],
                DType::F32,
                &[rows, encoded_row_bytes],
                DType::U8,
                None,
                &[1, rows],
                DType::F32,
                codec,
                [rows, columns],
            )
            .unwrap();
            let device = MetalDevice::get();
            let input = MetalTensor::from_f32(device, vec![1.0; columns], vec![1, columns]);
            let weight = MetalTensor {
                buffer: device.upload_bytes(&vec![0u8; requirements.encoded_row_bytes * rows]),
                layout: Layout::contiguous(vec![rows, requirements.encoded_row_bytes]),
                dtype: DType::U8,
            };
            let output = MetalTensor::from_f32(device, vec![0.0; rows], vec![1, rows]);
            warm_linear_exact(&requirements).unwrap();
            for _ in 0..10 {
                linear_into(&input, &weight, None, &output, &requirements).unwrap();
            }
            device.synchronize().unwrap();
            let started = std::time::Instant::now();
            for _ in 0..100 {
                linear_into(&input, &weight, None, &output, &requirements).unwrap();
            }
            device.synchronize().unwrap();
            let seconds = started.elapsed().as_secs_f64();
            let bytes = (requirements.encoded_row_bytes * rows) as f64;
            eprintln!(
                "{} {rows}x{columns}: {:.3} ms/call, {:.1} GB/s",
                codec.name(),
                seconds * 10.0,
                bytes * 100.0 / seconds / 1e9
            );
        }
    }
}

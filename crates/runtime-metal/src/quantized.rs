//! GGML K-quant execution. Linear operations accumulate each output directly
//! from packed blocks. Embedding operations visit only selected packed rows.
//!
//! ## Packed layouts (GGML K-quants)
//!
//! Weights are opaque u8 tensors of `[rows, encoded_row_bytes]` where each
//! row is `columns / 256` consecutive super-blocks of 256 values. Block
//! sizes per codec: Q2_K 84 B, Q3_K 110 B, Q4_K 144 B, Q5_K 176 B,
//! Q6_K 210 B. The MSL struct declarations in `LINEAR_SOURCE` mirror
//! the GGML byte layouts exactly and are guarded by `static_assert`s on
//! their sizes; `et_decode_k` (per-lane scalar decode) and
//! `et_decode_k16` (16-value group decode into threadgroup memory) are
//! define the bit-level formats. Sub-block scales and mins use 4 to 6 bits.
//! Quants use 2 to 6 bits split into low nibbles and high-bit masks. Each block
//! has an f16 super-scale `d` and, where required, `dmin`.
//!
//! ## Metal SIMD assumptions
//!
//! - SIMD width is exactly 32 (`threadExecutionWidth` is asserted at
//!   warm time); lane indexes drive the sub-block decomposition.
//! - `et_quantized_linear`: threadgroups of 64 threads (2 simdgroups),
//!   each simdgroup producing `rows_per_simd(codec)` output rows
//!   (Q2K/Q4K/Q6K: 2, Q3K: 3, Q5K: 1). Partial dots fold with
//!   `simd_sum` (decode) or a short `simd_shuffle_xor` tree (legacy
//!   batched multi-vector variants, `ET_VECTOR_LANES` ∈ {1, 2, 4}).
//! - `LinearDirect` is an exact Q2_K/Q3_K candidate for 2 to 8 input vectors.
//!   One simdgroup owns one output row and reuses each packed fragment
//!   across a compile-time vector tile while retaining M=1 f32
//!   arithmetic, accumulation order, and `simd_sum` for every vector.
//! - `et_quantized_linear_mma` is the prefill path for `vectors ≥ 8` and
//!   large weights. It decodes blocks to threadgroup memory with
//!   `et_decode_k16`, then 8×8 simdgroup matrix multiply-accumulate.
//!   The default-selected variants (`et_quantized_linear_mma_half`)
//!   dequantize weights and stage the input tile as `threadgroup half`
//!   with `simdgroup_half8x8` A/B operands and `simdgroup_float8x8`
//!   accumulators in the llama.cpp `mul_mm` tile geometry; f32-operand
//!   variants remain compiled for rollback and parity tests.
//!   Threadgroup usage stays under 16 KB (8 KB for the half variants),
//!   asserted at warm time.
//! - `et_quantized_embedding`: one threadgroup of 256 threads visits
//!   only the selected packed rows; out-of-range indexes are skipped
//!   and reported through a u32 status word (atomic store, relaxed).
//!
//! Packed weights must be zero-offset contiguous because kernels index them
//! byte-wise from the buffer base. Inputs and outputs are f32.

use crate::runtime::dtype::DType;
use crate::runtime::metal::run::MetalTensor;
use effect_torch_runtime::GgmlKQuant;

/// Requirements for a fused quantized linear
/// (`y = x · dequant(W)ᵀ + b`) over packed GGML K-quant weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearRequirements {
    /// The K-quant codec of the packed weight.
    pub codec: GgmlKQuant,
    /// f32 input shape, rank ≥ 2, trailing dim = `columns`.
    pub input_shape: Box<[usize]>,
    /// f32 output shape (`input_shape` with trailing dim = `rows`).
    pub output_shape: Box<[usize]>,
    /// Logical weight rows (output width).
    pub rows: usize,
    /// Logical weight columns (contraction width).
    pub columns: usize,
    /// Number of input vectors (`numel(input) / columns`).
    pub vectors: usize,
    /// Bytes of one packed row (`columns / 256 * block_bytes(codec)`).
    pub encoded_row_bytes: usize,
    /// Whether an f32 `[rows]` bias participates.
    pub has_bias: bool,
    /// Bytes of the f32 output.
    pub output_bytes: usize,
    /// 1 when the problem is non-empty, 0 for zero-element outputs
    /// (no dispatch needed).
    pub pipeline_count: usize,
}

/// Requirements for a decode-time grouped quantized
/// linear with 2 to 4 independent bias-free members sharing one f32 input
/// and codec, dispatched as a single `et_quantized_linear_grouped`
/// command. Restricted to the packed-dot decode kernels
/// (`KernelKind::Linear`/`LinearBatched`, vectors < 8); MMA prefill
/// shapes are never grouped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupedLinearRequirements {
    /// The shared K-quant codec of every packed weight.
    pub codec: GgmlKQuant,
    /// The shared f32 input shape, rank ≥ 2, trailing dim = `columns`.
    pub input_shape: Box<[usize]>,
    /// Shared contraction width.
    pub columns: usize,
    /// Shared number of input vectors (`numel(input) / columns`).
    pub vectors: usize,
    /// Shared bytes of one packed row.
    pub encoded_row_bytes: usize,
    /// Exact per-member plans; row counts and output shapes may differ.
    pub members: Box<[LinearRequirements]>,
    /// Total bytes of all f32 member outputs.
    pub output_bytes: usize,
    /// 1 when any member is non-empty, 0 for all-empty groups
    /// (no dispatch needed).
    pub pipeline_count: usize,
}

/// Requirements for a quantized embedding gather.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingRequirements {
    /// The K-quant codec of the packed table.
    pub codec: GgmlKQuant,
    /// Index tensor shape (any rank).
    pub index_shape: Box<[usize]>,
    /// Index dtype (u32 or i64).
    pub index_dtype: DType,
    /// f32 output shape (`index_shape ++ [columns]`).
    pub output_shape: Box<[usize]>,
    /// Table rows.
    pub rows: usize,
    /// Table columns (embedding width).
    pub columns: usize,
    /// Number of indexes (`numel(index_shape)`).
    pub indexes: usize,
    /// Bytes of one packed row.
    pub encoded_row_bytes: usize,
    /// Bytes of the f32 output.
    pub output_bytes: usize,
    /// Always 1: a single embedding kernel per launch.
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

/// Plans a fused quantized linear: validates the f32 input geometry
/// against the logical `[rows, columns]` weight, requires the packed
/// weight to be exactly `[rows, encoded_row_bytes]` u8 and the bias
/// (if any) `[rows]` f32, and computes the exact output plan.
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

/// Plans a decode-time grouped quantized linear from 2 to 4 member
/// plans. Members must share codec, input geometry (shape, columns,
/// vectors, encoded row bytes), and be bias-free; row counts may
/// differ. Only non-MMA packed-dot decode shapes (vectors < 8) group.
pub fn grouped_linear_requirements(
    members: &[LinearRequirements],
) -> Result<GroupedLinearRequirements, String> {
    if !(2..=4).contains(&members.len()) {
        return Err(format!(
            "quantized_linear_group: expected 2..=4 members, got {}",
            members.len()
        ));
    }
    let first = &members[0];
    for member in members {
        if member.codec != first.codec
            || member.input_shape != first.input_shape
            || member.columns != first.columns
            || member.vectors != first.vectors
            || member.encoded_row_bytes != first.encoded_row_bytes
        {
            return Err(
                "quantized_linear_group: members must share codec and exact input geometry"
                    .to_string(),
            );
        }
        if member.has_bias {
            return Err("quantized_linear_group: biased members are not grouped".to_string());
        }
        let mut expected_output = first.input_shape.to_vec();
        *expected_output
            .last_mut()
            .expect("member input rank was validated") = member.rows;
        if member.output_shape.as_ref() != expected_output.as_slice() {
            return Err(
                "quantized_linear_group: member output does not match the shared input shape"
                    .to_string(),
            );
        }
        if !matches!(
            linear_kernel_kind(member),
            KernelKind::Linear | KernelKind::LinearBatched(_)
        ) {
            return Err(
                "quantized_linear_group: only non-MMA packed-dot decode shapes group".to_string(),
            );
        }
    }
    let output_bytes = members
        .iter()
        .try_fold(0usize, |total, member| {
            total.checked_add(member.output_bytes)
        })
        .ok_or_else(|| "quantized_linear_group: output byte size overflow".to_string())?;
    Ok(GroupedLinearRequirements {
        codec: first.codec,
        input_shape: first.input_shape.clone(),
        columns: first.columns,
        vectors: first.vectors,
        encoded_row_bytes: first.encoded_row_bytes,
        members: members.to_vec().into_boxed_slice(),
        output_bytes,
        pipeline_count: usize::from(members.iter().any(|member| member.pipeline_count != 0)),
    })
}

/// Plans a quantized embedding gather: u32/i64 indexes into a packed
/// `[rows, encoded_row_bytes]` u8 table, producing f32
/// `index_shape ++ [columns]`.
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

#[allow(dead_code)] // f32 MMA variants remain warm-selectable rollback kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KernelKind {
    Linear,
    LinearBatched(u8),
    LinearDirect(u8),
    LinearGrouped(u8),
    LinearMma,
    LinearMmaSimple,
    LinearMma32,
    LinearMma32Simple,
    LinearMma8,
    LinearMma8Simple,
    LinearMmaHalf,
    LinearMmaSimpleHalf,
    LinearMma32Half,
    LinearMma32SimpleHalf,
    LinearMma8Half,
    LinearMma8SimpleHalf,
    LinearMma32SwzHalf,
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
    "effect_torch_quantized_k_v6".hash(&mut hasher);
    match kind {
        KernelKind::Linear => 0u8.hash(&mut hasher),
        KernelKind::LinearBatched(vector_lanes) => {
            2u8.hash(&mut hasher);
            vector_lanes.hash(&mut hasher);
        }
        KernelKind::LinearDirect(vectors) => {
            18u8.hash(&mut hasher);
            vectors.hash(&mut hasher);
        }
        KernelKind::LinearGrouped(vector_lanes) => {
            9u8.hash(&mut hasher);
            vector_lanes.hash(&mut hasher);
        }
        KernelKind::LinearMma => 3u8.hash(&mut hasher),
        KernelKind::LinearMmaSimple => 4u8.hash(&mut hasher),
        KernelKind::LinearMma32 => 7u8.hash(&mut hasher),
        KernelKind::LinearMma32Simple => 8u8.hash(&mut hasher),
        KernelKind::LinearMma8 => 5u8.hash(&mut hasher),
        KernelKind::LinearMma8Simple => 6u8.hash(&mut hasher),
        KernelKind::LinearMmaHalf => 10u8.hash(&mut hasher),
        KernelKind::LinearMmaSimpleHalf => 11u8.hash(&mut hasher),
        KernelKind::LinearMma32Half => 12u8.hash(&mut hasher),
        KernelKind::LinearMma32SimpleHalf => 13u8.hash(&mut hasher),
        KernelKind::LinearMma8Half => 14u8.hash(&mut hasher),
        KernelKind::LinearMma8SimpleHalf => 15u8.hash(&mut hasher),
        KernelKind::LinearMma32SwzHalf => 17u8.hash(&mut hasher),
        KernelKind::Embedding => 1u8.hash(&mut hasher),
    }
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

template <typename P>
inline void et_decode_k16(device const uchar* block, uint group, P destination, uint stride) {
#if ET_CODEC == 2
    const uint half_index = group / 8;
    const uint within = group % 8;
    const uint shift = (within / 2) * 2;
    const uint quant_offset = (within % 2) * 16;
    const uchar packed_scale = block[group];
    const uint mask = 3u << shift;
    const float scale = et_fp16_at(block, 80) * float(packed_scale & 15) / float(1u << shift);
    const float offset = et_fp16_at(block, 82) * float(packed_scale >> 4);
    device const uchar* quants = block + 16 + half_index * 32 + quant_offset;
    for (uint index = 0; index < 16; ++index) {
        destination[index * stride] = scale * float(quants[index] & mask) - offset;
    }
#elif ET_CODEC == 3
    const uint half_index = group / 8;
    const uint within = group % 8;
    const uint quant_lane = within / 2;
    const uint shift = quant_lane * 2;
    const uint quant_offset = (within % 2) * 16;
    const uchar low = group < 8 ? (block[96 + group] & 15) : (block[96 + group - 8] >> 4);
    const uchar high = (block[104 + group % 4] >> (2 * (group / 4))) & 3;
    const uint divisor = 1u << shift;
    const float scale = et_fp16_at(block, 108) * float(int(low | (high << 4)) - 32) / float(divisor);
    const uint quant_mask = 3u << shift;
    const uchar mask = uchar(1u << (half_index * 4 + quant_lane));
    device const uchar* quants = block + 32 + half_index * 32 + quant_offset;
    device const uchar* hmask = block + quant_offset;
    for (uint index = 0; index < 16; ++index) {
        const int low_quant = int(quants[index] & quant_mask);
        const int high_quant = (hmask[index] & mask) == 0 ? int(4u * divisor) : 0;
        destination[index * stride] = scale * float(low_quant - high_quant);
    }
#elif ET_CODEC == 4
    const uint scale_group = group / 2;
    const uint index_offset = (group % 2) * 16;
    const uint pair = scale_group / 2;
    const uint side = scale_group % 2;
    uint packed_scale;
    uint minimum;
    if (scale_group < 4) {
        packed_scale = uint(block[4 + scale_group] & 63);
        minimum = uint(block[8 + scale_group] & 63);
    } else {
        packed_scale = uint(block[8 + scale_group] & 15) | (uint(block[scale_group]) >> 6) << 4;
        minimum = uint(block[8 + scale_group] >> 4) | (uint(block[4 + scale_group]) >> 6) << 4;
    }
    const float d = et_fp16_at(block, 0);
    const float dmin = et_fp16_at(block, 2);
    for (uint index = 0; index < 16; ++index) {
        const uint quant_index = index_offset + index;
        const uchar packed = block[16 + pair * 32 + quant_index];
        const uint quant = side == 0 ? uint(packed & 15) : uint(packed >> 4);
        destination[index * stride] = d * float(packed_scale) * float(quant) - dmin * float(minimum);
    }
#elif ET_CODEC == 5
    const uint scale_group = group / 2;
    const uint index_offset = (group % 2) * 16;
    const uint pair = scale_group / 2;
    const uint side = scale_group % 2;
    uint packed_scale;
    uint minimum;
    if (scale_group < 4) {
        packed_scale = uint(block[4 + scale_group] & 63);
        minimum = uint(block[8 + scale_group] & 63);
    } else {
        packed_scale = uint(block[8 + scale_group] & 15) | (uint(block[scale_group]) >> 6) << 4;
        minimum = uint(block[8 + scale_group] >> 4) | (uint(block[4 + scale_group]) >> 6) << 4;
    }
    const float d = et_fp16_at(block, 0);
    const float dmin = et_fp16_at(block, 2);
    for (uint index = 0; index < 16; ++index) {
        const uint quant_index = index_offset + index;
        const uchar packed = block[48 + pair * 32 + quant_index];
        const uint low_quant = side == 0 ? uint(packed & 15) : uint(packed >> 4);
        const uchar mask = uchar(1u << (pair * 2 + side));
        const uint quant = low_quant + ((block[16 + quant_index] & mask) == 0 ? 0u : 16u);
        destination[index * stride] = d * float(packed_scale) * float(quant) - dmin * float(minimum);
    }
#else
    const uint half_index = group / 8;
    const uint within = (group % 8) * 16;
    const uint quarter = within / 32;
    const uint index_offset = within % 32;
    const uint low_index = half_index * 64 + index_offset + ((quarter & 1) == 0 ? 0 : 32);
    const uint low_shift = quarter < 2 ? 0 : 4;
    const uint high_shift = quarter * 2;
    const uint scale_lane = index_offset / 16;
    const uint scale_index = half_index * 8 + scale_lane + quarter * 2;
    const float scale = et_fp16_at(block, 208) * float(et_signed_byte(block[192 + scale_index]));
    for (uint index = 0; index < 16; ++index) {
        const uint quant = ((uint(block[low_index + index]) >> low_shift) & 15)
            | (((uint(block[128 + half_index * 32 + index_offset + index]) >> high_shift) & 3) << 4);
        destination[index * stride] = scale * float(int(quant) - 32);
    }
#endif
}

// Specialized 16-value group decode for the half MMA path. Q2_K/Q4_K/Q5_K
// load aligned uints; Q3_K/Q6_K fields are only 2-byte aligned, so they
// load ushort pairs. Packed fields are unpacked SIMD-in-register and converted
// through the half magic constant (0x6400 | v encodes 1024 + v exactly).
// The f32 scale arithmetic and single rounding to half match et_decode_k16
// bit for bit.
inline void et_decode_k16_half(device const uchar* block, uint group, thread half* destination) {
#if ET_CODEC == 2
    const uint half_index = group / 8;
    const uint within = group % 8;
    const uint shift = (within / 2) * 2;
    const uint quant_offset = (within % 2) * 16;
    const uchar packed_scale = block[group];
    const float scale = et_fp16_at(block, 80) * float(packed_scale & 15);
    const float offset = et_fp16_at(block, 82) * float(packed_scale >> 4);
    device const uint* quants = reinterpret_cast<device const uint*>(
        block + 16 + half_index * 32 + quant_offset);
    const uint4 q = uint4(quants[0], quants[1], quants[2], quants[3]);
    const uint4 u = (q >> shift) & uint4(0x03030303u);
    for (uint word = 0; word < 4; ++word) {
        // Magic-lane split yields byte order (b0, b2) / (b1, b3).
        const uint lo = (u[word] & 0x00FF00FFu) | 0x64006400u;
        const uint hi = ((u[word] >> 8) & 0x00FF00FFu) | 0x64006400u;
        const half2 even = as_type<half2>(lo);
        const half2 odd = as_type<half2>(hi);
        const half4 unpacked = half4(even.x, odd.x, even.y, odd.y) - half4(1024.0h);
        const float4 values = float4(unpacked) * scale - offset;
        *reinterpret_cast<thread half4*>(destination + 4 * word) = half4(values);
    }
#elif ET_CODEC == 3
    const uint half_index = group / 8;
    const uint within = group % 8;
    const uint quant_lane = within / 2;
    const uint shift = quant_lane * 2;
    const uint quant_offset = (within % 2) * 16;
    const uchar low = group < 8 ? (block[96 + group] & 15) : (block[96 + group - 8] >> 4);
    const uchar high = (block[104 + group % 4] >> (2 * (group / 4))) & 3;
    const float scale = et_fp16_at(block, 108) * float(int(low | (high << 4)) - 32);
    device const ushort* quants = reinterpret_cast<device const ushort*>(
        block + 32 + half_index * 32 + quant_offset);
    device const ushort* hmask = reinterpret_cast<device const ushort*>(block + quant_offset);
    uint4 q;
    uint4 h;
    for (uint word = 0; word < 4; ++word) {
        q[word] = uint(quants[2 * word]) | (uint(quants[2 * word + 1]) << 16);
        h[word] = uint(hmask[2 * word]) | (uint(hmask[2 * word + 1]) << 16);
    }
    const uint4 u = ((q >> shift) & uint4(0x03030303u))
        | (((h >> (half_index * 4 + quant_lane)) & uint4(0x01010101u)) << 2);
    for (uint word = 0; word < 4; ++word) {
        // Magic-lane split yields byte order (b0, b2) / (b1, b3).
        const uint lo = (u[word] & 0x00FF00FFu) | 0x64006400u;
        const uint hi = ((u[word] >> 8) & 0x00FF00FFu) | 0x64006400u;
        const half2 even = as_type<half2>(lo);
        const half2 odd = as_type<half2>(hi);
        const half4 unpacked = half4(even.x, odd.x, even.y, odd.y) - half4(1028.0h);
        const float4 values = float4(unpacked) * scale;
        *reinterpret_cast<thread half4*>(destination + 4 * word) = half4(values);
    }
#elif ET_CODEC == 4 || ET_CODEC == 5
    const uint scale_group = group / 2;
    const uint index_offset = (group % 2) * 16;
    const uint pair = scale_group / 2;
    const uint side = scale_group % 2;
    uint packed_scale;
    uint minimum;
    if (scale_group < 4) {
        packed_scale = uint(block[4 + scale_group] & 63);
        minimum = uint(block[8 + scale_group] & 63);
    } else {
        packed_scale = uint(block[8 + scale_group] & 15) | (uint(block[scale_group]) >> 6) << 4;
        minimum = uint(block[8 + scale_group] >> 4) | (uint(block[4 + scale_group]) >> 6) << 4;
    }
    const float scale = et_fp16_at(block, 0) * float(packed_scale);
    const float offset = et_fp16_at(block, 2) * float(minimum);
#if ET_CODEC == 4
    device const uint* quants = reinterpret_cast<device const uint*>(
        block + 16 + pair * 32 + index_offset);
#else
    device const uint* quants = reinterpret_cast<device const uint*>(
        block + 48 + pair * 32 + index_offset);
    device const uint* high_bits = reinterpret_cast<device const uint*>(block + 16 + index_offset);
#endif
    uint4 u;
    for (uint word = 0; word < 4; ++word) {
        const uint low = side == 0
            ? quants[word] & 0x0F0F0F0Fu
            : (quants[word] >> 4) & 0x0F0F0F0Fu;
#if ET_CODEC == 4
        u[word] = low;
#else
        u[word] = low | (((high_bits[word] >> (pair * 2 + side)) & 0x01010101u) << 4);
#endif
    }
    for (uint word = 0; word < 4; ++word) {
        const uint lo = (u[word] & 0x00FF00FFu) | 0x64006400u;
        const uint hi = ((u[word] >> 8) & 0x00FF00FFu) | 0x64006400u;
        const half2 even = as_type<half2>(lo);
        const half2 odd = as_type<half2>(hi);
        const half4 unpacked = half4(even.x, odd.x, even.y, odd.y) - half4(1024.0h);
        const float4 values = float4(unpacked) * scale - offset;
        *reinterpret_cast<thread half4*>(destination + 4 * word) = half4(values);
    }
#elif ET_CODEC == 6
    const uint half_index = group / 8;
    const uint within = group % 8;
    const uint quarter = within / 2;
    const uint index_offset = (within % 2) * 16;
    const uint low_offset = half_index * 64 + index_offset + ((quarter & 1) == 0 ? 0 : 32);
    const uint low_shift = quarter < 2 ? 0 : 4;
    const uint high_shift = quarter * 2;
    const float scale = et_fp16_at(block, 208) * float(et_signed_byte(block[192 + group]));
    device const ushort* lows = reinterpret_cast<device const ushort*>(block + low_offset);
    device const ushort* highs = reinterpret_cast<device const ushort*>(
        block + 128 + half_index * 32 + index_offset);
    uint4 low;
    uint4 high;
    for (uint word = 0; word < 4; ++word) {
        low[word] = uint(lows[2 * word]) | (uint(lows[2 * word + 1]) << 16);
        high[word] = uint(highs[2 * word]) | (uint(highs[2 * word + 1]) << 16);
    }
    const uint4 u = ((low >> low_shift) & uint4(0x0F0F0F0Fu))
        | (((high >> high_shift) & uint4(0x03030303u)) << 4);
    for (uint word = 0; word < 4; ++word) {
        const uint lo = (u[word] & 0x00FF00FFu) | 0x64006400u;
        const uint hi = ((u[word] >> 8) & 0x00FF00FFu) | 0x64006400u;
        const half2 even = as_type<half2>(lo);
        const half2 odd = as_type<half2>(hi);
        const half4 unpacked = half4(even.x, odd.x, even.y, odd.y) - half4(1056.0h);
        const float4 values = float4(unpacked) * scale;
        *reinterpret_cast<thread half4*>(destination + 4 * word) = half4(values);
    }
#else
    et_decode_k16(block, group, destination, 1);
#endif
}
// Paired variant of et_decode_k16_half: decodes groups `group` and
// `group + 2`, which share the same 16 quant bytes (fields at shifts
// s and s+2) and, for Q3_K, the same 16 hmask bytes (bits b and b+1).
// Halves the device loads per decoded value. Writes 32 halves: the
// first 16 for `group`, the next 16 for `group + 2`.
#if ET_CODEC == 2
inline void et_decode_k16_half2(device const uchar* block, uint group, thread half* destination) {
    const uint half_index = group / 8;
    const uint within = group % 8;
    const uint shift = (within / 2) * 2;
    const uint quant_offset = (within % 2) * 16;
    const uchar packed_scale_a = block[group];
    const uchar packed_scale_b = block[group + 2];
    const float d = et_fp16_at(block, 80);
    const float dmin = et_fp16_at(block, 82);
    const float scale_a = d * float(packed_scale_a & 15);
    const float offset_a = dmin * float(packed_scale_a >> 4);
    const float scale_b = d * float(packed_scale_b & 15);
    const float offset_b = dmin * float(packed_scale_b >> 4);
    device const uint* quants = reinterpret_cast<device const uint*>(
        block + 16 + half_index * 32 + quant_offset);
    const uint4 q = uint4(quants[0], quants[1], quants[2], quants[3]);
    // Per byte: quant fields a|b<<2 side by side.
    const uint4 both = (q >> shift) & uint4(0x0F0F0F0Fu);
    const uint4 u_a = both & uint4(0x03030303u);
    const uint4 u_b = (both >> 2) & uint4(0x03030303u);
    for (uint word = 0; word < 4; ++word) {
        const uint lo_a = (u_a[word] & 0x00FF00FFu) | 0x64006400u;
        const uint hi_a = ((u_a[word] >> 8) & 0x00FF00FFu) | 0x64006400u;
        const half2 even_a = as_type<half2>(lo_a);
        const half2 odd_a = as_type<half2>(hi_a);
        const float4 values_a =
            float4(half4(even_a.x, odd_a.x, even_a.y, odd_a.y) - half4(1024.0h)) * scale_a
            - offset_a;
        *reinterpret_cast<thread half4*>(destination + 4 * word) = half4(values_a);
        const uint lo_b = (u_b[word] & 0x00FF00FFu) | 0x64006400u;
        const uint hi_b = ((u_b[word] >> 8) & 0x00FF00FFu) | 0x64006400u;
        const half2 even_b = as_type<half2>(lo_b);
        const half2 odd_b = as_type<half2>(hi_b);
        const float4 values_b =
            float4(half4(even_b.x, odd_b.x, even_b.y, odd_b.y) - half4(1024.0h)) * scale_b
            - offset_b;
        *reinterpret_cast<thread half4*>(destination + 16 + 4 * word) = half4(values_b);
    }
}
#elif ET_CODEC == 3
inline void et_decode_k16_half2(device const uchar* block, uint group, thread half* destination) {
    const uint half_index = group / 8;
    const uint within = group % 8;
    const uint quant_lane = within / 2;
    const uint shift = quant_lane * 2;
    const uint quant_offset = (within % 2) * 16;
    const uchar low_a = group < 8 ? (block[96 + group] & 15) : (block[96 + group - 8] >> 4);
    const uchar low_b = group + 2 < 8 ? (block[96 + group + 2] & 15) : (block[96 + group - 6] >> 4);
    const uchar high_a = (block[104 + group % 4] >> (2 * (group / 4))) & 3;
    const uchar high_b = (block[104 + (group + 2) % 4] >> (2 * ((group + 2) / 4))) & 3;
    const float d = et_fp16_at(block, 108);
    const float scale_a = d * float(int(low_a | (high_a << 4)) - 32);
    const float scale_b = d * float(int(low_b | (high_b << 4)) - 32);
    device const ushort* quants = reinterpret_cast<device const ushort*>(
        block + 32 + half_index * 32 + quant_offset);
    device const ushort* hmask = reinterpret_cast<device const ushort*>(block + quant_offset);
    uint4 q;
    uint4 h;
    for (uint word = 0; word < 4; ++word) {
        q[word] = uint(quants[2 * word]) | (uint(quants[2 * word + 1]) << 16);
        h[word] = uint(hmask[2 * word]) | (uint(hmask[2 * word + 1]) << 16);
    }
    const uint bit = half_index * 4 + quant_lane;
    // Per byte: quant fields a|b<<2 in v, hmask bits b|b+1 in w.
    const uint4 v = (q >> shift) & uint4(0x0F0F0F0Fu);
    const uint4 w = (h >> bit) & uint4(0x03030303u);
    const uint4 u_a = (v & uint4(0x03030303u)) | ((w & uint4(0x01010101u)) << 2);
    const uint4 u_b = ((v >> 2) & uint4(0x03030303u)) | ((w & uint4(0x02020202u)) << 1);
    for (uint word = 0; word < 4; ++word) {
        const uint lo_a = (u_a[word] & 0x00FF00FFu) | 0x64006400u;
        const uint hi_a = ((u_a[word] >> 8) & 0x00FF00FFu) | 0x64006400u;
        const half2 even_a = as_type<half2>(lo_a);
        const half2 odd_a = as_type<half2>(hi_a);
        const float4 values_a =
            float4(half4(even_a.x, odd_a.x, even_a.y, odd_a.y) - half4(1028.0h)) * scale_a;
        *reinterpret_cast<thread half4*>(destination + 4 * word) = half4(values_a);
        const uint lo_b = (u_b[word] & 0x00FF00FFu) | 0x64006400u;
        const uint hi_b = ((u_b[word] >> 8) & 0x00FF00FFu) | 0x64006400u;
        const half2 even_b = as_type<half2>(lo_b);
        const half2 odd_b = as_type<half2>(hi_b);
        const float4 values_b =
            float4(half4(even_b.x, odd_b.x, even_b.y, odd_b.y) - half4(1028.0h)) * scale_b;
        *reinterpret_cast<thread half4*>(destination + 16 + 4 * word) = half4(values_b);
    }
}
#else
// Never called (pair_decode is false for other codecs); defined so the
// dead branch in the half MMA kernel still compiles.
inline void et_decode_k16_half2(device const uchar* block, uint group, thread half* destination) {
    et_decode_k16_half(block, group, destination);
    et_decode_k16_half(block, group + 2, destination + 16);
}
#endif
"#;

const LINEAR_SOURCE: &str = r#"
#define ET_ROWS_PER_SIMD $ROWS_PER_SIMD
#define ET_VECTOR_LANES $VECTOR_LANES
#define ET_SIMD_GROUPS 2ul
#define ET_UNROLL _Pragma("clang loop unroll(full)")
#if ET_VECTOR_LANES == 1
#define ET_BLOCK_STRIDE 4u
#elif ET_VECTOR_LANES == 2
#define ET_BLOCK_STRIDE 2u
#else
#define ET_BLOCK_STRIDE 1u
#endif

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

inline void et_quantized_linear_body(
    device const float* input,
    device const uchar* weight,
    device const float* bias,
    device float* output,
    const ulong vectors,
    const ulong rows,
    const ulong columns,
    const ulong encoded_row_bytes,
    const uint has_bias,
    const uint2 group,
    const ushort lane,
    const ushort simd_group) {
    const ulong first_row = (ulong(group.x) * ET_SIMD_GROUPS + ulong(simd_group)) * ET_ROWS_PER_SIMD;
#if ET_VECTOR_LANES == 1
    const ulong vector = ulong(group.y);
    if (vector >= vectors || first_row >= rows) {
        return;
    }
    const ulong input_vector = vector;
#else
    if (first_row >= rows) {
        return;
    }
    const ushort vector_lane = lane / (32 / ET_VECTOR_LANES);
    const ulong vector = ulong(group.y) * ET_VECTOR_LANES + ulong(vector_lane);
    const bool active_vector = vector < vectors;
    const ulong input_vector = active_vector ? vector : 0ul;
#endif
    const ulong blocks = columns / ET_BLOCK_VALUES;
    device const float* y = input + input_vector * columns;
    float sums[ET_ROWS_PER_SIMD] = { 0.0f };

#if ET_CODEC == 2
#if ET_VECTOR_LANES == 1
    const ushort ix = lane / 8;
#elif ET_VECTOR_LANES == 2
    const ushort ix = (lane % 16) / 8;
#else
    const ushort ix = 0;
#endif
    const ushort it = lane % 8;
    const ushort iq = it / 4;
    const ushort ir = it % 4;
    const ushort scale_offset = ir / 2;
    device const float* y4 = y + ulong(ix) * ET_BLOCK_VALUES + 128ul * iq + 8ul * ir;

    const uint packed_blocks = uint(blocks);
    for (uint block_index = ix; block_index < packed_blocks; block_index += ET_BLOCK_STRIDE) {
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
        y4 += ulong(ET_BLOCK_STRIDE) * ET_BLOCK_VALUES;
    }
#elif ET_CODEC == 3
#if ET_VECTOR_LANES == 1
    const ushort tid = (lane % 4) + 4 * (lane / 16);
    const ushort ix = (lane % 16) / 4;
#elif ET_VECTOR_LANES == 2
    const ushort tid = lane % 8;
    const ushort ix = (lane % 16) / 8;
#else
    const ushort tid = lane % 8;
    const ushort ix = 0;
#endif
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
    for (uint block_index = ix; block_index < q3_blocks; block_index += ET_BLOCK_STRIDE) {
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
        y1 += ulong(ET_BLOCK_STRIDE) * ET_BLOCK_VALUES;
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
        float sum = sums[row_offset];
#if ET_VECTOR_LANES == 1
        sum = simd_sum(sum);
        if (lane == 0) {
            const ulong row = first_row + row_offset;
            output[vector * rows + row] = sum + (has_bias != 0 ? bias[row] : 0.0f);
        }
#else
        sum += simd_shuffle_xor(sum, ushort(4));
        sum += simd_shuffle_xor(sum, ushort(2));
        sum += simd_shuffle_xor(sum, ushort(1));
#if ET_VECTOR_LANES == 2
        sum += simd_shuffle_xor(sum, ushort(8));
#endif
        if (lane % (32 / ET_VECTOR_LANES) == 0 && active_vector) {
            const ulong row = first_row + row_offset;
            output[vector * rows + row] = sum + (has_bias != 0 ? bias[row] : 0.0f);
        }
#endif
    }
}

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
    et_quantized_linear_body(
        input,
        weight,
        bias,
        output,
        vectors,
        rows,
        columns,
        encoded_row_bytes,
        has_bias,
        group,
        lane,
        simd_group
    );
}
"#;

// Q2_K/Q3_K small-row verifier path. The lane decomposition, block stride,
// expressions, and simd_sum reduction match the one-vector packed-dot body
// above. Only the loop nesting changes. Packed fields are
// loaded before the compile-time vector-tile loop and reused within that tile.
const LINEAR_DIRECT_SOURCE: &str = r#"
#define ET_DIRECT_VECTORS $DIRECT_VECTORS
#define ET_DIRECT_TILE $DIRECT_TILE

inline void et_quantized_linear_direct_body(
    device const float* input,
    device const uchar* weight,
    device const float* bias,
    device float* output,
    const ulong vectors,
    const ulong rows,
    const ulong columns,
    const ulong encoded_row_bytes,
    const uint has_bias,
    const uint group_x,
    const uint group_y,
    const ushort lane,
    const ushort simd_group) {
    const ulong first_row =
        (ulong(group_x) * ET_SIMD_GROUPS + ulong(simd_group)) * ET_ROWS_PER_SIMD;
    const ulong first_vector = ulong(group_y) * ET_DIRECT_TILE;
    if (first_row >= rows || first_vector >= vectors || vectors != ET_DIRECT_VECTORS) {
        return;
    }
    const ushort active_vectors = ushort(min(ulong(ET_DIRECT_TILE), vectors - first_vector));
    const ulong blocks = columns / ET_BLOCK_VALUES;
    float sums[ET_ROWS_PER_SIMD][ET_DIRECT_TILE] = { 0.0f };

#if ET_CODEC == 2
    const ushort ix = lane / 8;
    const ushort it = lane % 8;
    const ushort iq = it / 4;
    const ushort ir = it % 4;
    const ushort scale_offset = ir / 2;
    const uint packed_blocks = uint(blocks);
    for (uint block_index = ix; block_index < packed_blocks; block_index += 4) {
        ET_UNROLL for (ushort row_offset = 0;
                       row_offset < ET_ROWS_PER_SIMD && first_row + row_offset < rows;
                       ++row_offset) {
            device const et_block_q2_k* block = reinterpret_cast<device const et_block_q2_k*>(
                weight + (first_row + row_offset) * encoded_row_bytes + block_index * ET_BLOCK_BYTES);
            device const uchar* scales = block->scales + 8 * iq + scale_offset;
            device const ushort* quants =
                reinterpret_cast<device const ushort*>(block->qs) + 16 * iq + 4 * ir;
            const uchar4 packed_scales = uchar4(scales[0], scales[2], scales[4], scales[6]);
            const ushort4 packed_quants = ushort4(quants[0], quants[1], quants[2], quants[3]);
            const float d = float(block->d);
            const float dmin = float(block->dmin) / 16.0f;
            ET_UNROLL for (ushort vector = 0; vector < ET_DIRECT_TILE; ++vector) {
                if (vector >= active_vectors) {
                    continue;
                }
                device const float* y4 = input + (first_vector + ulong(vector)) * columns +
                    ulong(block_index) * ET_BLOCK_VALUES + 128ul * iq + 8ul * ir;
                float values[32];
                float4 input_sums = 0.0f;
                ET_UNROLL for (ushort i = 0; i < 8; ++i) {
                    values[i] = y4[i];
                    values[i + 8] = y4[i + 32];
                    values[i + 16] = y4[i + 64];
                    values[i + 24] = y4[i + 96];
                    input_sums += float4(values[i], values[i + 8], values[i + 16], values[i + 24]);
                }
                float4 low = 0.0f;
                float4 high = 0.0f;
                ET_UNROLL for (ushort i = 0; i < 8; i += 2) {
                    const ushort q = packed_quants[i / 2];
                    low[0] += values[i] * (q & 0x0003);
                    high[0] += values[i + 1] * (q & 0x0300);
                    low[1] += values[i + 8] * (q & 0x000c);
                    high[1] += values[i + 9] * (q & 0x0c00);
                    low[2] += values[i + 16] * (q & 0x0030);
                    high[2] += values[i + 17] * (q & 0x3000);
                    low[3] += values[i + 24] * (q & 0x00c0);
                    high[3] += values[i + 25] * (q & 0xc000);
                }
                sums[row_offset][vector] += d * (
                    (low[0] + high[0] / 256.0f) * (packed_scales[0] & 0x0f) +
                    (low[1] + high[1] / 256.0f) * (packed_scales[1] & 0x0f) / 4.0f +
                    (low[2] + high[2] / 256.0f) * (packed_scales[2] & 0x0f) / 16.0f +
                    (low[3] + high[3] / 256.0f) * (packed_scales[3] & 0x0f) / 64.0f) -
                    dmin * (input_sums[0] * (packed_scales[0] & 0xf0) +
                            input_sums[1] * (packed_scales[1] & 0xf0) +
                            input_sums[2] * (packed_scales[2] & 0xf0) +
                            input_sums[3] * (packed_scales[3] & 0xf0));
            }
        }
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
    float sums_high[ET_ROWS_PER_SIMD][ET_DIRECT_TILE] = { 0.0f };
    const uint packed_blocks = uint(blocks);
    for (uint block_index = ix; block_index < packed_blocks; block_index += 4) {
        ET_UNROLL for (ushort row_offset = 0;
                       row_offset < ET_ROWS_PER_SIMD && first_row + row_offset < rows;
                       ++row_offset) {
            device const et_block_q3_k* block = reinterpret_cast<device const et_block_q3_k*>(
                weight + (first_row + row_offset) * encoded_row_bytes + block_index * ET_BLOCK_BYTES);
            device const ushort* quants = reinterpret_cast<device const ushort*>(block->qs + quant_offset);
            device const ushort* high = reinterpret_cast<device const ushort*>(block->hmask + l0);
            const ushort4 quants_low = ushort4(quants[0], quants[1], quants[2], quants[3]);
            const ushort4 quants_high = ushort4(quants[8], quants[9], quants[10], quants[11]);
            const ushort4 high_low = ushort4(high[0], high[1], high[2], high[3]);
            const ushort4 high_high = ushort4(high[8], high[9], high[10], high[11]);
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
            const char4 block_scales = char4(scales[0], scales[1], scales[2], scales[3]);
            const float d = float(block->d);
            ET_UNROLL for (ushort vector = 0; vector < ET_DIRECT_TILE; ++vector) {
                if (vector >= active_vectors) {
                    continue;
                }
                device const float* y1 = input + (first_vector + ulong(vector)) * columns +
                    ulong(block_index) * ET_BLOCK_VALUES + input_offset;
                float values[32];
                ET_UNROLL for (ushort i = 0; i < 8; ++i) {
                    values[i] = y1[i];
                    values[i + 8] = y1[i + 16];
                    values[i + 16] = y1[i + 32];
                    values[i + 24] = y1[i + 48];
                }
                float a0 = 0.0f, a1 = 0.0f, absent0 = 0.0f;
                float a2 = 0.0f, a3 = 0.0f, absent1 = 0.0f;
                ET_UNROLL for (ushort i = 0; i < 8; i += 2) {
                    const int q = quants_low[i / 2];
                    const ushort h = high_low[i / 2];
                    const float4 dotted = float4(values[i], values[i + 1], values[i + 16], values[i + 17]) *
                        float4(q & quant_masks[il / 2][0], q & quant_masks[il / 2][1],
                               q & quant_masks[il / 2][2], q & quant_masks[il / 2][3]);
                    a0 += dotted[0];
                    a1 += dotted[1];
                    a2 += dotted[2];
                    a3 += dotted[3];
                    absent0 += (h & hm[0] ? 0.0f : values[i]) +
                               (h & hm[1] ? 0.0f : values[i + 1]);
                    absent1 += (h & hm[2] ? 0.0f : values[i + 16]) +
                               (h & hm[3] ? 0.0f : values[i + 17]);
                }
                float dot_low = d * (a0 + a1 / 256.0f - absent0 * absent_low);
                float dot_high = d * (a2 + a3 / 256.0f - absent1 * absent_high);
                sums[row_offset][vector] += dot_low * (block_scales[0] - 32);
                sums_high[row_offset][vector] += dot_high * (block_scales[2] - 32);

                a0 = a1 = absent0 = a2 = a3 = absent1 = 0.0f;
                ET_UNROLL for (ushort i = 0; i < 8; i += 2) {
                    const int q = quants_high[i / 2];
                    const ushort h = high_high[i / 2];
                    a0 += values[i + 8] * (q & quant_masks[il / 2][0]);
                    a1 += values[i + 9] * (q & quant_masks[il / 2][1]);
                    absent0 += (h & hm[0] ? 0.0f : values[i + 8]) +
                               (h & hm[1] ? 0.0f : values[i + 9]);
                    a2 += values[i + 24] * (q & quant_masks[il / 2][2]);
                    a3 += values[i + 25] * (q & quant_masks[il / 2][3]);
                    absent1 += (h & hm[2] ? 0.0f : values[i + 24]) +
                               (h & hm[3] ? 0.0f : values[i + 25]);
                }
                dot_low = d * (a0 + a1 / 256.0f - absent0 * absent_low);
                dot_high = d * (a2 + a3 / 256.0f - absent1 * absent_high);
                sums[row_offset][vector] += dot_low * (block_scales[1] - 32);
                sums_high[row_offset][vector] += dot_high * (block_scales[3] - 32);
            }
        }
    }
    ET_UNROLL for (ushort row_offset = 0; row_offset < ET_ROWS_PER_SIMD; ++row_offset) {
        ET_UNROLL for (ushort vector = 0; vector < ET_DIRECT_TILE; ++vector) {
            sums[row_offset][vector] =
                (sums[row_offset][vector] + 0.25f * sums_high[row_offset][vector]) /
                float(1 << (2 * il));
        }
    }
#else
#error et_quantized_linear_direct only supports Q2_K and Q3_K
#endif

    ET_UNROLL for (ushort row_offset = 0;
                   row_offset < ET_ROWS_PER_SIMD && first_row + row_offset < rows;
                   ++row_offset) {
        ET_UNROLL for (ushort vector = 0; vector < ET_DIRECT_TILE; ++vector) {
            float sum = simd_sum(sums[row_offset][vector]);
            if (lane == 0 && vector < active_vectors) {
                const ulong row = first_row + row_offset;
                output[(first_vector + ulong(vector)) * rows + row] =
                    sum + (has_bias != 0 ? bias[row] : 0.0f);
            }
        }
    }
}

kernel void et_quantized_linear_direct(
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
    et_quantized_linear_direct_body(
        input, weight, bias, output, vectors, rows, columns, encoded_row_bytes,
        has_bias, group.x, group.y, lane, simd_group
    );
}
"#;

/// Grouped decode projection in which 2 to 4 members share an input and codec.
/// `group.z` selects the member; each member binds its own packed
/// weight and output and reads its own row count. Bias-free only
/// (grouping rejects biased members), so the bias slot of the shared
/// body receives the member output pointer and `has_bias == 0`.
const LINEAR_GROUPED_SOURCE: &str = r#"
kernel void et_quantized_linear_grouped(
    device const float* input [[buffer(0)]],
    device const uchar* weight0 [[buffer(1)]],
    device const uchar* weight1 [[buffer(2)]],
    device const uchar* weight2 [[buffer(3)]],
    device const uchar* weight3 [[buffer(4)]],
    device float* output0 [[buffer(5)]],
    device float* output1 [[buffer(6)]],
    device float* output2 [[buffer(7)]],
    device float* output3 [[buffer(8)]],
    constant ulong& vectors [[buffer(9)]],
    constant ulong* member_rows [[buffer(10)]],
    constant ulong& columns [[buffer(11)]],
    constant ulong& encoded_row_bytes [[buffer(12)]],
    uint3 group3 [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint member = group3.z;
    device const uchar* weight = member == 0 ? weight0
        : member == 1 ? weight1
        : member == 2 ? weight2
        : weight3;
    device float* output = member == 0 ? output0
        : member == 1 ? output1
        : member == 2 ? output2
        : output3;
    et_quantized_linear_body(
        input,
        weight,
        output,
        output,
        vectors,
        member_rows[member],
        columns,
        encoded_row_bytes,
        0u,
        uint2(group3.x, group3.y),
        lane,
        simd_group
    );
}
"#;

const MMA_LINEAR_SOURCE: &str = r#"
#include <metal_simdgroup_matrix>

#define ET_PIPELINED $PIPELINED
#define ET_TILE_M $TILE_M

inline void et_decode_mma_tile(
    device const uchar* weight,
    ulong rows,
    ulong encoded_row_bytes,
    ulong first_row,
    uint k0,
    uint local,
    threadgroup float* tile
) {
    constexpr uint tile_n = 16;
    constexpr uint tile_k = 128;
    const uint group_index = local;
    const uint row_offset = group_index / (tile_k / 16);
    const uint group_offset = group_index % (tile_k / 16);
    const ulong row = first_row + row_offset;
    const uint group_column = k0 + group_offset * 16;
    if (row < rows) {
        device const uchar* block = weight + row * encoded_row_bytes +
            ulong(group_column / 256) * ET_BLOCK_BYTES;
        et_decode_k16(
            block,
            (group_column % 256) / 16,
            &tile[row_offset * tile_k + group_offset * 16],
            1
        );
    } else {
        for (uint index = 0; index < 16; ++index) {
            tile[row_offset * tile_k + group_offset * 16 + index] = 0.0f;
        }
    }
}

kernel void et_quantized_linear_mma(
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
    uint tid [[thread_index_in_threadgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr uint tile_m = ET_TILE_M;
    constexpr uint tile_n = 16;
    constexpr uint tile_k = 128;
    constexpr uint decode_storage = ET_PIPELINED ? 2 * tile_k * tile_n : tile_k * tile_n;
    constexpr uint output_storage = 4 * tile_m * tile_n;
    constexpr uint storage_len = decode_storage > output_storage ? decode_storage : output_storage;
    threadgroup float tile_storage[storage_len];
    const ulong first_vector = ulong(group.y) * tile_m;
    const ulong first_row = ulong(group.x) * tile_n;
    simdgroup_float8x8 accumulators[ET_TILE_M / 8][2];
    for (uint i = 0; i < ET_TILE_M / 8; ++i) {
        for (uint j = 0; j < 2; ++j) {
            accumulators[i][j] = simdgroup_float8x8(0.0f);
        }
    }

#if ET_PIPELINED
    uint current = 0;
    uint next = tile_k * tile_n;
    if (tid >= 128) {
        et_decode_mma_tile(
            weight,
            rows,
            encoded_row_bytes,
            first_row,
            0,
            tid - 128,
            &tile_storage[current]
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k0 = 0; k0 < uint(columns); k0 += tile_k) {
        if (tid < 128) {
            const uint n0 = 0;
            const uint k_begin = uint(simd_group) * 32;
            for (uint k = k_begin; k < k_begin + 32; k += 8) {
                simdgroup_float8x8 b0;
                simdgroup_load(b0, &tile_storage[current + (n0 + 0) * tile_k + k], tile_k, 0, true);
                simdgroup_float8x8 b1;
                simdgroup_load(b1, &tile_storage[current + (n0 + 8) * tile_k + k], tile_k, 0, true);
                for (uint i = 0; i < ET_TILE_M / 8; ++i) {
                    simdgroup_float8x8 a;
                    simdgroup_load(a, input + (first_vector + i * 8) * columns + k0 + k, uint(columns));
                    simdgroup_multiply_accumulate(accumulators[i][0], a, b0, accumulators[i][0]);
                    simdgroup_multiply_accumulate(accumulators[i][1], a, b1, accumulators[i][1]);
                }
            }
        } else if (k0 + tile_k < uint(columns)) {
            et_decode_mma_tile(
                weight,
                rows,
                encoded_row_bytes,
                first_row,
                k0 + tile_k,
                tid - 128,
                &tile_storage[next]
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint swap = current;
        current = next;
        next = swap;
    }
#else
    for (uint k0 = 0; k0 < uint(columns); k0 += tile_k) {
        et_decode_mma_tile(
            weight,
            rows,
            encoded_row_bytes,
            first_row,
            k0,
            tid,
            tile_storage
        );
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint n0 = 0;
        const uint k_begin = uint(simd_group) * 32;
        for (uint k = k_begin; k < k_begin + 32; k += 8) {
            simdgroup_float8x8 b0;
            simdgroup_load(b0, &tile_storage[(n0 + 0) * tile_k + k], tile_k, 0, true);
            simdgroup_float8x8 b1;
            simdgroup_load(b1, &tile_storage[(n0 + 8) * tile_k + k], tile_k, 0, true);
            for (uint i = 0; i < ET_TILE_M / 8; ++i) {
                simdgroup_float8x8 a;
                simdgroup_load(a, input + (first_vector + i * 8) * columns + k0 + k, uint(columns));
                simdgroup_multiply_accumulate(accumulators[i][0], a, b0, accumulators[i][0]);
                simdgroup_multiply_accumulate(accumulators[i][1], a, b1, accumulators[i][1]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#endif

    const uint n0 = 0;
    const uint partial = uint(simd_group);
    if (tid < 128) {
        for (uint i = 0; i < ET_TILE_M / 8; ++i) {
            for (uint j = 0; j < 2; ++j) {
                simdgroup_store(
                    accumulators[i][j],
                    &tile_storage[partial * tile_m * tile_n + (i * 8) * tile_n + n0 + j * 8],
                    tile_n
                );
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#if ET_PIPELINED
    for (uint element = tid; element < tile_m * tile_n; element += 256) {
#else
    for (uint element = tid; element < tile_m * tile_n; element += 128) {
#endif
        const uint vector_offset = element / tile_n;
        const uint row_offset = element % tile_n;
        const ulong vector = first_vector + vector_offset;
        const ulong row = first_row + row_offset;
        if (vector < vectors && row < rows) {
            output[vector * rows + row] =
                tile_storage[vector_offset * tile_n + row_offset] +
                tile_storage[tile_m * tile_n + vector_offset * tile_n + row_offset] +
                tile_storage[2 * tile_m * tile_n + vector_offset * tile_n + row_offset] +
                tile_storage[3 * tile_m * tile_n + vector_offset * tile_n + row_offset] +
                (has_bias != 0 ? bias[row] : 0.0f);
        }
    }
}
"#;

/// Half-operand twin of `MMA_LINEAR_SOURCE`, following the shipped
/// llama.cpp `mul_mm` geometry: weight tiles are dequantized into
/// `threadgroup half`, the input tile is staged into `threadgroup half`
/// once per K tile, A/B operands are `simdgroup_half8x8`, and
/// accumulation stays in `simdgroup_float8x8`. Each threadgroup covers
/// ET_TILE_N rows (32 by default, 64 in the wide geometry) x ET_TILE_M
/// vectors over a 2048-element K tile. The production M=8/M=16 paths
/// partition complete output blocks among simdgroups and avoid a final
/// reduction; M=32 and rollback variants split K four ways and fold the
/// partials through threadgroup memory after the K loop. Decode runs
/// into registers before the tile barrier so it overlaps the previous
/// tile's MMA. Tiles use the interleaved 8x8-block layout (64
/// contiguous halves per block, ld=8 loads, no transposes) and stay
/// under 8 KB of threadgroup memory.
const MMA_LINEAR_HALF_SOURCE: &str = r#"
#include <metal_simdgroup_matrix>

#define ET_TILE_M $TILE_M
#define ET_TILE_N $TILE_N
#define ET_SWIZZLE $SWIZZLE
#define ET_OUTPUT_SPLIT $OUTPUT_SPLIT

kernel void et_quantized_linear_mma_half(
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
    uint tid [[thread_index_in_threadgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr uint tile_m = ET_TILE_M;
    constexpr uint tile_n = ET_TILE_N;
    // The paired Q2_K/Q3_K decoder is retained as reference code but stays
    // disabled: enabling it does not preserve nonzero-weight parity.
    constexpr bool pair_decode = false;
    constexpr uint tile_k = 2048 / tile_n;
    constexpr uint k_blocks = tile_k / 8;
    constexpr uint decode_chunks = tile_k / 16;
    constexpr uint temp_groups = pair_decode ? 2 : 1;
    constexpr uint decode_units = pair_decode ? decode_chunks / 2 : decode_chunks;
    constexpr uint decode_items = tile_n * decode_units;
    constexpr uint decode_rounds = (decode_items + 127) / 128;
    constexpr uint w_halves = tile_n * tile_k;
    constexpr uint a_halves = tile_m * tile_k;
    constexpr uint half_floats = (w_halves + a_halves + 1) / 2;
    constexpr uint out_floats = tile_n * tile_m;
    constexpr uint storage_floats = half_floats > out_floats ? half_floats : out_floats;
    constexpr uint storage_quads = storage_floats / 4;
    // K-split simdgroup geometry: each simdgroup owns the full output
    // tile (a_count x b_count 8x8 blocks) over k_blocks / 4 of the K
    // tile, so every 8x8 weight/input block is loaded from threadgroup
    // memory exactly once per K tile (halving load traffic against the
    // 2x2 output-quadrant split). The four K-quarter partials are
    // folded through threadgroup memory after the K loop.
    constexpr uint n_blocks = tile_n / 8;
    constexpr uint b_count = n_blocks;
    constexpr uint a_count = tile_m / 8;
    // Input staging: 8-element chunks, at most `stage_rounds` per thread.
    constexpr uint stage_chunks = tile_m * tile_k / 8;
    constexpr uint stage_rounds = (stage_chunks + 127) / 128;
    threadgroup uint4 storage[storage_quads];
    // Weight tile: [k][n] 8x8 blocks, block (nb, kb) at
    // (nb * k_blocks + kb) * 64.
    threadgroup half* wt = reinterpret_cast<threadgroup half*>(storage);
    // Input tile: [m][k] 8x8 blocks, block (mb, kb) at
    // (mb * k_blocks + kb) * 64.
    threadgroup half* it = wt + w_halves;
    // Output staging: [m][n] f32, aliases the half tiles (written only
    // after the last tile read).
    threadgroup float* st = reinterpret_cast<threadgroup float*>(storage);
#if ET_SWIZZLE
    // Vector-tile-fastest dispatch order: co-resident threadgroups share
    // one 32-row weight tile, so the L2 absorbs the M-tile replication
    // instead of DRAM (llama.cpp mul_mm grid order).
    const ulong first_vector = ulong(group.x) * tile_m;
    const ulong first_row = ulong(group.y) * tile_n;
#else
    const ulong first_vector = ulong(group.y) * tile_m;
    const ulong first_row = ulong(group.x) * tile_n;
#endif
#if ET_OUTPUT_SPLIT
    simdgroup_float8x8 output_accumulators[a_count];
    for (uint j = 0; j < a_count; ++j) {
        output_accumulators[j] = simdgroup_float8x8(0.0f);
    }
#else
    simdgroup_float8x8 accumulators[a_count][b_count];
    for (uint i = 0; i < a_count; ++i) {
        for (uint j = 0; j < b_count; ++j) {
            accumulators[i][j] = simdgroup_float8x8(0.0f);
        }
    }
#endif

    // Per-round decode coordinates (loop-invariant): item =
    // tid + round * 128 covers the tile_n x decode_units grid. In the
    // paired Q2_K/Q3_K path each item covers groups (c, c + 2) of one
    // superblock; chunk order interleaves so both stay in-block.
    uint decode_row_r[decode_rounds];
    uint decode_chunk_r[decode_rounds][2];
    for (uint round = 0; round < decode_rounds; ++round) {
        const uint item = tid + round * 128;
        const bool active = item < decode_items;
        decode_row_r[round] = active ? item / decode_units : 0;
        const uint unit = active ? item % decode_units : 0;
        if (pair_decode) {
            const uint first_chunk = (unit % 2) + (unit / 2) * (decode_chunks / 2);
            decode_chunk_r[round][0] = first_chunk;
            decode_chunk_r[round][1] = first_chunk + 2;
        } else {
            decode_chunk_r[round][0] = unit;
        }
    }
    const bool stage_active = tid < stage_chunks || stage_rounds > 1;
    const uint stage_m0 = tid / k_blocks;
    const uint stage_k0 = (tid % k_blocks) * 8;
    for (uint k0 = 0; k0 < uint(columns); k0 += tile_k) {
        // Dequantize 16 values of one row into registers before the
        // barrier, so the decode overlaps the previous tile's MMA
        // (llama.cpp mul_mm ordering). half4 storage keeps the
        // vectorized Q2_K/Q3_K decode stores aligned.
        half4 temp[decode_rounds][temp_groups][4];
        for (uint round = 0; round < decode_rounds; ++round) {
            const uint item = tid + round * 128;
            thread half* temp_h = reinterpret_cast<thread half*>(temp[round]);
            if (item < decode_items && first_row + decode_row_r[round] < rows) {
                const uint group_column = k0 + decode_chunk_r[round][0] * 16;
                device const uchar* block =
                    weight + (first_row + decode_row_r[round]) * encoded_row_bytes +
                    ulong(group_column / 256) * ET_BLOCK_BYTES;
                if (pair_decode) {
                    et_decode_k16_half2(block, (group_column % 256) / 16, temp_h);
                } else {
                    et_decode_k16_half(block, (group_column % 256) / 16, temp_h);
                }
            } else {
                for (uint g = 0; g < temp_groups; ++g) {
                    for (uint index = 0; index < 4; ++index) {
                        temp[round][g][index] = half4(0.0h);
                    }
                }
            }
        }
        // Likewise fetch this thread's input chunks early. The staged
        // offsets are always 16-byte aligned (columns and tile_k are
        // multiples of 256 and stage_k0 a multiple of 8), so float4
        // loads are safe.
        float4 staged[stage_rounds][2];
        if (stage_active) {
            for (uint round = 0; round < stage_rounds; ++round) {
                const uint m = stage_m0 + round * (128 / k_blocks);
                if (m < tile_m) {
                    device const float4* source =
                        reinterpret_cast<device const float4*>(
                            input + (first_vector + m) * columns + k0 + stage_k0);
                    staged[round][0] = source[0];
                    staged[round][1] = source[1];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Stride 8 lands the two 8-value halves in adjacent k blocks.
        for (uint round = 0; round < decode_rounds; ++round) {
            if (tid + round * 128 < decode_items) {
                for (uint g = 0; g < temp_groups; ++g) {
                    threadgroup half* base =
                        &wt[((decode_row_r[round] / 8) * k_blocks +
                                decode_chunk_r[round][g] * 2) * 64 +
                            decode_row_r[round] % 8];
                    thread const half* temp_h =
                        reinterpret_cast<thread const half*>(temp[round][g]);
                    for (uint index = 0; index < 16; ++index) {
                        base[index * 8] = temp_h[index];
                    }
                }
            }
        }
        if (stage_active) {
            for (uint round = 0; round < stage_rounds; ++round) {
                const uint m = stage_m0 + round * (128 / k_blocks);
                if (m < tile_m) {
                    threadgroup half4* base = reinterpret_cast<threadgroup half4*>(
                        it + ((m / 8) * k_blocks + stage_k0 / 8) * 64 + (m % 8) * 8
                    );
                    base[0] = half4(staged[round][0]);
                    base[1] = half4(staged[round][1]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#if ET_OUTPUT_SPLIT
        // Each simdgroup owns `a_count` output blocks across full K.
        // This rereads threadgroup tiles but needs no split-K reduction.
        const uint output_i = uint(simd_group) * a_count / 4;
        const uint output_j = uint(simd_group) * a_count % b_count;
        for (uint kb = 0; kb < k_blocks; ++kb) {
            simdgroup_half8x8 a;
            simdgroup_load(a, it + (output_i * k_blocks + kb) * 64, 8);
            for (uint j = 0; j < a_count; ++j) {
                simdgroup_half8x8 b;
                simdgroup_load(b, wt + ((output_j + j) * k_blocks + kb) * 64, 8);
                simdgroup_multiply_accumulate(
                    output_accumulators[j],
                    a,
                    b,
                    output_accumulators[j]
                );
            }
        }
#else
        // K-split MMA: simdgroup s reduces K blocks s, s + 4, ... over
        // the full output tile; 2x the MACs per threadgroup load of
        // the output-quadrant split.
        for (uint kb = uint(simd_group); kb < k_blocks; kb += 4) {
            simdgroup_half8x8 a[a_count];
            for (uint i = 0; i < a_count; ++i) {
                simdgroup_load(a[i], it + (i * k_blocks + kb) * 64, 8);
            }
            simdgroup_half8x8 b[b_count];
            for (uint j = 0; j < b_count; ++j) {
                simdgroup_load(b[j], wt + (j * k_blocks + kb) * 64, 8);
            }
            for (uint i = 0; i < a_count; ++i) {
                for (uint j = 0; j < b_count; ++j) {
                    simdgroup_multiply_accumulate(
                        accumulators[i][j],
                        a[i],
                        b[j],
                        accumulators[i][j]
                    );
                }
            }
        }
#endif
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
#if ET_OUTPUT_SPLIT
    const uint output_i = uint(simd_group) * a_count / 4;
    const uint output_j = uint(simd_group) * a_count % b_count;
    for (uint j = 0; j < a_count; ++j) {
        simdgroup_store(
            output_accumulators[j],
            &st[(output_i * 8) * tile_n + (output_j + j) * 8],
            tile_n
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#else
    // Fold the four K-quarter partial reductions into st: each
    // simdgroup adds its accumulators onto the running sum in turn.
    for (uint turn = 0; turn < 4; ++turn) {
        if (uint(simd_group) == turn) {
            for (uint i = 0; i < a_count; ++i) {
                for (uint j = 0; j < b_count; ++j) {
                    threadgroup float* dst = &st[(i * 8) * tile_n + j * 8];
                    if (turn == 0) {
                        simdgroup_store(accumulators[i][j], dst, tile_n);
                    } else {
                        simdgroup_float8x8 partial;
                        simdgroup_load(partial, dst, tile_n);
                        for (uint e = 0; e < 2; ++e) {
                            partial.thread_elements()[e] +=
                                accumulators[i][j].thread_elements()[e];
                        }
                        simdgroup_store(partial, dst, tile_n);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#endif
    for (uint element = tid; element < tile_n * tile_m; element += 128) {
        const uint vector_offset = element / tile_n;
        const uint row_offset = element % tile_n;
        const ulong vector = first_vector + vector_offset;
        const ulong out_row = first_row + row_offset;
        if (vector < vectors && out_row < rows) {
            output[vector * rows + out_row] =
                st[element] + (has_bias != 0 ? bias[out_row] : 0.0f);
        }
    }
}

// Warp-specialized pipelined twin (probe candidate): 256 threads, of
// which simdgroups 0-3 run the MMA over the current K tile while
// simdgroups 4-7 dequantize and stage the next K tile into the other
// half of the double-buffered tiles. One barrier per tile; decode and
// staging overlap the MMA within the threadgroup instead of relying on
// co-resident threadgroups.
template <uint TILE_M, uint TILE_N>
inline void et_mma_half_ws_fill(
    device const uchar* weight,
    device const float* input,
    ulong rows,
    ulong encoded_row_bytes,
    ulong first_row,
    ulong first_vector,
    ulong columns,
    uint k0,
    uint dtid,
    threadgroup half* wt_nxt,
    threadgroup half* it_nxt
) {
    constexpr uint tile_m = TILE_M;
    constexpr uint tile_k = 2048 / TILE_N;
    constexpr uint k_blocks = tile_k / 8;
    constexpr uint decode_chunks = tile_k / 16;
    constexpr uint stage_rounds = (tile_m * tile_k / 8 + 127) / 128;
    const uint decode_row = dtid / decode_chunks;
    const uint decode_chunk = dtid % decode_chunks;
    const ulong decode_row_global = first_row + decode_row;
    const uint stage_m0 = dtid / k_blocks;
    const uint stage_k0 = (dtid % k_blocks) * 8;
    half4 temp[4];
    if (decode_row_global < rows) {
        const uint group_column = k0 + decode_chunk * 16;
        device const uchar* block = weight +
            decode_row_global * encoded_row_bytes +
            ulong(group_column / 256) * ET_BLOCK_BYTES;
        et_decode_k16_half(
            block,
            (group_column % 256) / 16,
            reinterpret_cast<thread half*>(temp)
        );
    } else {
        for (uint index = 0; index < 4; ++index) {
            temp[index] = half4(0.0h);
        }
    }
    float4 staged[stage_rounds][2];
    for (uint round = 0; round < stage_rounds; ++round) {
        const uint m = stage_m0 + round * (128 / k_blocks);
        if (m < tile_m) {
            device const float4* source = reinterpret_cast<device const float4*>(
                input + (first_vector + m) * columns + k0 + stage_k0);
            staged[round][0] = source[0];
            staged[round][1] = source[1];
        }
    }
    threadgroup half* base =
        &wt_nxt[((decode_row / 8) * k_blocks + decode_chunk * 2) * 64 + decode_row % 8];
    thread const half* temp_h = reinterpret_cast<thread const half*>(temp);
    for (uint index = 0; index < 16; ++index) {
        base[index * 8] = temp_h[index];
    }
    for (uint round = 0; round < stage_rounds; ++round) {
        const uint m = stage_m0 + round * (128 / k_blocks);
        if (m < tile_m) {
            threadgroup half4* stage_base = reinterpret_cast<threadgroup half4*>(
                it_nxt + ((m / 8) * k_blocks + stage_k0 / 8) * 64 + (m % 8) * 8
            );
            stage_base[0] = half4(staged[round][0]);
            stage_base[1] = half4(staged[round][1]);
        }
    }
}

kernel void et_quantized_linear_mma_half_ws(
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
    uint tid [[thread_index_in_threadgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr uint tile_m = ET_TILE_M;
    constexpr uint tile_n = ET_TILE_N;
    constexpr uint tile_k = 2048 / tile_n;
    constexpr uint k_blocks = tile_k / 8;
    constexpr uint decode_chunks = tile_k / 16;
    constexpr uint decode_items = tile_n * decode_chunks;
    static_assert(decode_items == 128, "ws geometry decodes one group per thread");
    constexpr uint w_halves = tile_n * tile_k;
    constexpr uint a_halves = tile_m * tile_k;
    constexpr uint buf_halves = w_halves + a_halves;
    constexpr uint out_floats = tile_n * tile_m;
    constexpr uint storage_floats = buf_halves > out_floats ? buf_halves : out_floats;
    constexpr uint storage_quads = storage_floats / 4;
    constexpr uint n_blocks = tile_n / 8;
    constexpr uint sg_n = tile_m >= 16 ? (n_blocks >= 2 ? 2 : 1)
        : (n_blocks >= 4 ? 4 : n_blocks);
    constexpr uint sg_m = 4 / sg_n;
    constexpr uint b_count = n_blocks / sg_n;
    constexpr uint a_count = tile_m / 8 / sg_m;
    threadgroup uint4 storage[storage_quads];
    // Double-buffered: [wt0|wt1] then [it0|it1].
    threadgroup half* wt = reinterpret_cast<threadgroup half*>(storage);
    threadgroup half* it = wt + 2 * w_halves;
    threadgroup float* st = reinterpret_cast<threadgroup float*>(storage);
    const ulong first_vector = ulong(group.y) * tile_m;
    const ulong first_row = ulong(group.x) * tile_n;
    const bool mma_side = tid < 128;
    const uint dtid = tid >= 128 ? tid - 128 : tid;
    const uint block_row = (uint(simd_group) % sg_n) * b_count;
    const uint block_vector = (uint(simd_group) / sg_n) * a_count;
    simdgroup_float8x8 accumulators[a_count][b_count];
    for (uint i = 0; i < a_count; ++i) {
        for (uint j = 0; j < b_count; ++j) {
            accumulators[i][j] = simdgroup_float8x8(0.0f);
        }
    }

    // Prologue: decode side fills buffer 0 for the first K tile.
    if (!mma_side) {
        et_mma_half_ws_fill<ET_TILE_M, ET_TILE_N>(
            weight, input, rows, encoded_row_bytes, first_row, first_vector,
            columns, 0, dtid, wt, it
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k0 = 0, turn = 0; k0 < uint(columns); k0 += tile_k, ++turn) {
        const uint cur = (turn & 1) * w_halves;
        const uint cur_in = (turn & 1) * a_halves;
        if (mma_side) {
            threadgroup const half* wt_cur = wt + cur;
            threadgroup const half* it_cur = it + cur_in;
            for (uint kb = 0; kb < k_blocks; ++kb) {
                simdgroup_half8x8 a[a_count];
                for (uint i = 0; i < a_count; ++i) {
                    simdgroup_load(a[i], it_cur + ((block_vector + i) * k_blocks + kb) * 64, 8);
                }
                simdgroup_half8x8 b[b_count];
                for (uint j = 0; j < b_count; ++j) {
                    simdgroup_load(b[j], wt_cur + ((block_row + j) * k_blocks + kb) * 64, 8);
                }
                for (uint i = 0; i < a_count; ++i) {
                    for (uint j = 0; j < b_count; ++j) {
                        simdgroup_multiply_accumulate(
                            accumulators[i][j],
                            a[i],
                            b[j],
                            accumulators[i][j]
                        );
                    }
                }
            }
        } else if (k0 + tile_k < uint(columns)) {
            const uint nxt = ((turn + 1) & 1) * w_halves;
            const uint nxt_in = ((turn + 1) & 1) * a_halves;
            et_mma_half_ws_fill<ET_TILE_M, ET_TILE_N>(
                weight, input, rows, encoded_row_bytes, first_row, first_vector,
                columns, k0 + tile_k, dtid, wt + nxt, it + nxt_in
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (mma_side) {
        for (uint i = 0; i < a_count; ++i) {
            for (uint j = 0; j < b_count; ++j) {
                simdgroup_store(
                    accumulators[i][j],
                    &st[(block_vector * 8 + i * 8) * tile_n + (block_row + j) * 8],
                    tile_n
                );
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint element = tid; element < tile_n * tile_m; element += 256) {
        const uint vector_offset = element / tile_n;
        const uint row_offset = element % tile_n;
        const ulong vector = first_vector + vector_offset;
        const ulong out_row = first_row + row_offset;
        if (vector < vectors && out_row < rows) {
            output[vector * rows + out_row] =
                st[element] + (has_bias != 0 ? bias[out_row] : 0.0f);
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
        KernelKind::Linear | KernelKind::LinearBatched(_) => LINEAR_SOURCE.to_string(),
        KernelKind::LinearDirect(_) => {
            let mut direct =
                String::with_capacity(LINEAR_SOURCE.len() + LINEAR_DIRECT_SOURCE.len());
            direct.push_str(LINEAR_SOURCE);
            direct.push_str(LINEAR_DIRECT_SOURCE);
            direct
        }
        KernelKind::LinearGrouped(_) => {
            let mut grouped =
                String::with_capacity(LINEAR_SOURCE.len() + LINEAR_GROUPED_SOURCE.len());
            grouped.push_str(LINEAR_SOURCE);
            grouped.push_str(LINEAR_GROUPED_SOURCE);
            grouped
        }
        KernelKind::LinearMma
        | KernelKind::LinearMmaSimple
        | KernelKind::LinearMma32
        | KernelKind::LinearMma32Simple
        | KernelKind::LinearMma8
        | KernelKind::LinearMma8Simple => MMA_LINEAR_SOURCE.to_string(),
        KernelKind::LinearMmaHalf
        | KernelKind::LinearMmaSimpleHalf
        | KernelKind::LinearMma32Half
        | KernelKind::LinearMma32SimpleHalf
        | KernelKind::LinearMma8Half
        | KernelKind::LinearMma8SimpleHalf
        | KernelKind::LinearMma32SwzHalf => MMA_LINEAR_HALF_SOURCE.to_string(),
        KernelKind::Embedding => EMBEDDING_SOURCE.to_string(),
    };
    let mut source = String::with_capacity(DECODE_SOURCE.len() + kernel.len());
    source.push_str(DECODE_SOURCE);
    source.push_str(&kernel);
    source = source.replace("$CODEC", &codec_tag(codec).to_string());
    source = source.replace("$BLOCK_BYTES", &block_bytes(codec).to_string());
    let source_rows_per_simd = if matches!(kind, KernelKind::LinearDirect(_)) {
        1
    } else {
        rows_per_simd(codec)
    };
    source = source.replace("$ROWS_PER_SIMD", &source_rows_per_simd.to_string());
    let vector_lanes = match kind {
        KernelKind::LinearBatched(vector_lanes) | KernelKind::LinearGrouped(vector_lanes) => {
            vector_lanes
        }
        KernelKind::Linear
        | KernelKind::LinearMma
        | KernelKind::LinearMmaSimple
        | KernelKind::LinearMma32
        | KernelKind::LinearMma32Simple
        | KernelKind::LinearMma8
        | KernelKind::LinearMma8Simple
        | KernelKind::LinearMmaHalf
        | KernelKind::LinearMmaSimpleHalf
        | KernelKind::LinearMma32Half
        | KernelKind::LinearMma32SimpleHalf
        | KernelKind::LinearMma8Half
        | KernelKind::LinearMma8SimpleHalf
        | KernelKind::LinearMma32SwzHalf
        | KernelKind::LinearDirect(_)
        | KernelKind::Embedding => 1,
    };
    source = source.replace("$VECTOR_LANES", &vector_lanes.to_string());
    if let KernelKind::LinearDirect(vectors) = kind {
        source = source.replace("$DIRECT_VECTORS", &vectors.to_string());
        source = source.replace(
            "$DIRECT_TILE",
            &usize::from(vectors)
                .min(match codec {
                    GgmlKQuant::Q2K => 4,
                    GgmlKQuant::Q3K => 2,
                    _ => unreachable!("direct row reuse only supports Q2_K/Q3_K"),
                })
                .to_string(),
        );
    }
    source = source.replace(
        "$PIPELINED",
        if matches!(
            kind,
            KernelKind::LinearMma
                | KernelKind::LinearMma32
                | KernelKind::LinearMma8
                | KernelKind::LinearMmaHalf
                | KernelKind::LinearMma32Half
                | KernelKind::LinearMma8Half
        ) {
            "1"
        } else {
            "0"
        },
    );
    source = source.replace(
        "$TILE_M",
        if matches!(
            kind,
            KernelKind::LinearMma8
                | KernelKind::LinearMma8Simple
                | KernelKind::LinearMma8Half
                | KernelKind::LinearMma8SimpleHalf
        ) {
            "8"
        } else if matches!(
            kind,
            KernelKind::LinearMma32
                | KernelKind::LinearMma32Simple
                | KernelKind::LinearMma32Half
                | KernelKind::LinearMma32SimpleHalf
                | KernelKind::LinearMma32SwzHalf
        ) {
            "32"
        } else {
            "16"
        },
    );
    source = source.replace(
        "$TILE_N",
        if matches!(
            kind,
            KernelKind::LinearMmaSimpleHalf
                | KernelKind::LinearMma32SimpleHalf
                | KernelKind::LinearMma8SimpleHalf
                | KernelKind::LinearMma32SwzHalf
        ) {
            "32"
        } else {
            "64"
        },
    );
    source = source.replace(
        "$SWIZZLE",
        if matches!(kind, KernelKind::LinearMma32SwzHalf) {
            "1"
        } else {
            "0"
        },
    );
    source = source.replace(
        "$OUTPUT_SPLIT",
        if matches!(
            kind,
            KernelKind::LinearMmaSimpleHalf | KernelKind::LinearMma8SimpleHalf
        ) {
            "1"
        } else {
            "0"
        },
    );
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
        KernelKind::Linear | KernelKind::LinearBatched(_) => "et_quantized_linear",
        KernelKind::LinearDirect(_) => "et_quantized_linear_direct",
        KernelKind::LinearGrouped(_) => "et_quantized_linear_grouped",
        KernelKind::LinearMma
        | KernelKind::LinearMmaSimple
        | KernelKind::LinearMma32
        | KernelKind::LinearMma32Simple
        | KernelKind::LinearMma8
        | KernelKind::LinearMma8Simple => "et_quantized_linear_mma",
        KernelKind::LinearMmaHalf
        | KernelKind::LinearMmaSimpleHalf
        | KernelKind::LinearMma32Half
        | KernelKind::LinearMma32SimpleHalf
        | KernelKind::LinearMma8Half
        | KernelKind::LinearMma8SimpleHalf
        | KernelKind::LinearMma32SwzHalf => "et_quantized_linear_mma_half",
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
                    KernelKind::LinearBatched(_) => "linear_batched",
                    KernelKind::LinearDirect(_) => "linear_direct",
                    KernelKind::LinearGrouped(_) => "linear_grouped",
                    KernelKind::LinearMma => "linear_mma",
                    KernelKind::LinearMmaSimple => "linear_mma_simple",
                    KernelKind::LinearMma32 => "linear_mma32",
                    KernelKind::LinearMma32Simple => "linear_mma32_simple",
                    KernelKind::LinearMma8 => "linear_mma8",
                    KernelKind::LinearMma8Simple => "linear_mma8_simple",
                    KernelKind::LinearMmaHalf => "linear_mma_half",
                    KernelKind::LinearMmaSimpleHalf => "linear_mma_simple_half",
                    KernelKind::LinearMma32Half => "linear_mma32_half",
                    KernelKind::LinearMma32SimpleHalf => "linear_mma32_simple_half",
                    KernelKind::LinearMma8Half => "linear_mma8_half",
                    KernelKind::LinearMma8SimpleHalf => "linear_mma8_simple_half",
                    KernelKind::LinearMma32SwzHalf => "linear_mma32_swz_half",
                    KernelKind::Embedding => "embedding",
                }
            )
        })
}

/// Selects the measured production kernel. Direct f32-faithful
/// Q2_K/Q3_K row-reuse variants remain explicit parity/profiling
/// candidates: serializing vectors inside each SIMD lane loses to both
/// parallel qmv and M=8 MMA on Apple silicon.
fn linear_kernel_kind(requirements: &LinearRequirements) -> KernelKind {
    if requirements.vectors >= 32 && requirements.vectors % 32 == 0 {
        return KernelKind::LinearMma32SimpleHalf;
    }
    if requirements.vectors >= 16 && requirements.vectors % 16 == 0 {
        return KernelKind::LinearMmaSimpleHalf;
    }
    if requirements.vectors >= 8 && requirements.vectors % 8 == 0 {
        return KernelKind::LinearMma8SimpleHalf;
    }
    if requirements.vectors >= 4 {
        match requirements.codec {
            GgmlKQuant::Q2K => KernelKind::LinearBatched(2),
            GgmlKQuant::Q3K => KernelKind::LinearBatched(4),
            _ => KernelKind::Linear,
        }
    } else {
        KernelKind::Linear
    }
}

/// Warms exactly the linear pipeline selected by `requirements` and
/// asserts the Metal SIMD assumptions the kernels rely on (thread
/// execution width 32, sufficient threadgroup capacity; 16 KB
/// threadgroup memory for the f32 MMA variants, 8 KB for the
/// default half variants).
pub fn warm_linear_exact(requirements: &LinearRequirements) -> Result<(), String> {
    if requirements.pipeline_count != 0 {
        use objc2_metal::{MTLComputePipelineState, MTLDevice as _};

        let kernel_kind = linear_kernel_kind(requirements);
        let pipeline = pipeline(kernel_kind, requirements.codec, None)?;
        let required_threads = match kernel_kind {
            KernelKind::LinearMma | KernelKind::LinearMma32 | KernelKind::LinearMma8 => 256,
            KernelKind::LinearMmaSimple
            | KernelKind::LinearMma32Simple
            | KernelKind::LinearMma8Simple
            | KernelKind::LinearMmaHalf
            | KernelKind::LinearMmaSimpleHalf
            | KernelKind::LinearMma32Half
            | KernelKind::LinearMma32SimpleHalf
            | KernelKind::LinearMma8Half
            | KernelKind::LinearMma8SimpleHalf
            | KernelKind::LinearMma32SwzHalf => 128,
            _ => 64,
        };
        if pipeline.as_raw().threadExecutionWidth() != 32
            || pipeline.as_raw().maxTotalThreadsPerThreadgroup() < required_threads
        {
            return Err(
                "quantized_linear: Metal pipeline requires SIMD width 32 and exact threadgroup capacity"
                    .to_string(),
            );
        }
        let required_memory = match kernel_kind {
            KernelKind::LinearMma | KernelKind::LinearMma32 | KernelKind::LinearMma8 => 16 * 1024,
            KernelKind::LinearMmaSimple
            | KernelKind::LinearMma32Simple
            | KernelKind::LinearMma8Simple => 8 * 1024,
            // Half decode tiles free room for the staged input tile; the
            // 64-row x 32-vector half tile peaks at 8 KB (output staging
            // aliased onto the 6 KB of half tiles). The 32-row x 64-vector
            // tile needs 12 KB (2048 weight + 4096 input halves).
            KernelKind::LinearMmaHalf
            | KernelKind::LinearMmaSimpleHalf
            | KernelKind::LinearMma32Half
            | KernelKind::LinearMma32SimpleHalf
            | KernelKind::LinearMma8Half
            | KernelKind::LinearMma8SimpleHalf
            | KernelKind::LinearMma32SwzHalf => 8 * 1024,
            _ => 0,
        };
        if crate::device::MetalDevice::get()
            .raw()
            .maxThreadgroupMemoryLength()
            < required_memory
        {
            return Err(
                "quantized_linear_mma: Metal device has insufficient threadgroup memory"
                    .to_string(),
            );
        }
    }
    Ok(())
}

/// Warms exactly the embedding pipeline described by `requirements`.
pub fn warm_embedding_exact(requirements: &EmbeddingRequirements) -> Result<(), String> {
    pipeline(
        KernelKind::Embedding,
        requirements.codec,
        Some(requirements.index_dtype),
    )?;
    Ok(())
}

/// Selects the grouped packed-dot kernel matching the members' shared
/// vector/codec geometry. Grouped requirements reject MMA shapes, so
/// this always yields a `LinearGrouped` variant.
fn grouped_kernel_kind(requirements: &GroupedLinearRequirements) -> KernelKind {
    match linear_kernel_kind(
        requirements
            .members
            .first()
            .expect("grouped requirements hold 2..=4 members"),
    ) {
        KernelKind::Linear => KernelKind::LinearGrouped(1),
        KernelKind::LinearBatched(vector_lanes) => KernelKind::LinearGrouped(vector_lanes),
        _ => unreachable!("grouped requirements reject MMA shapes"),
    }
}

/// Warms exactly the grouped linear pipeline selected by
/// `requirements` and asserts the Metal SIMD assumptions the shared
/// kernel body relies on (thread execution width 32, 64-thread
/// threadgroups).
pub fn warm_linear_grouped_exact(requirements: &GroupedLinearRequirements) -> Result<(), String> {
    if requirements.pipeline_count != 0 {
        use objc2_metal::MTLComputePipelineState;

        let pipeline = pipeline(grouped_kernel_kind(requirements), requirements.codec, None)?;
        if pipeline.as_raw().threadExecutionWidth() != 32
            || pipeline.as_raw().maxTotalThreadsPerThreadgroup() < 64
        {
            return Err(
                "quantized_linear_group: Metal pipeline requires SIMD width 32 and exact threadgroup capacity"
                    .to_string(),
            );
        }
    }
    Ok(())
}

fn validate_grouped_member(
    weight: &MetalTensor,
    output: &MetalTensor,
    requirements: &LinearRequirements,
) -> Result<(), String> {
    if weight.dtype != DType::U8
        || weight.layout.shape() != [requirements.rows, requirements.encoded_row_bytes]
    {
        return Err(format!(
            "quantized_linear_group: expected exact contiguous packed {} weight [{}, {}] u8",
            requirements.codec.name(),
            requirements.rows,
            requirements.encoded_row_bytes
        ));
    }
    require_contiguous(weight, "quantized_linear_group", "packed weight", true)?;
    output.validate_destination(
        "quantized_linear_group",
        &requirements.output_shape,
        DType::F32,
    )
}

/// Non-allocating grouped quantized linear dispatch: validates every
/// member against the immutable plan and encodes one
/// `et_quantized_linear_grouped` command covering all members.
/// Allocates nothing; requires the exact pipeline to be warm. A plan
/// with `pipeline_count == 0` (all-empty outputs) is a no-op.
pub fn linear_grouped_into(
    input: &MetalTensor,
    members: &[(&MetalTensor, &MetalTensor)],
    requirements: &GroupedLinearRequirements,
) -> Result<(), String> {
    use crate::runtime::metal::device::{set_buffer, set_bytes, MetalDevice};
    use objc2_metal::MTLComputeCommandEncoder;

    if members.len() != requirements.members.len() {
        return Err(format!(
            "quantized_linear_group: expected {} members, got {}",
            requirements.members.len(),
            members.len()
        ));
    }
    if input.dtype != DType::F32 || input.layout.shape() != requirements.input_shape.as_ref() {
        return Err("quantized_linear_group: input does not match the immutable plan".to_string());
    }
    require_contiguous(input, "quantized_linear_group", "input", false)?;
    for (&(weight, output), member) in members.iter().zip(requirements.members.iter()) {
        validate_grouped_member(weight, output, member)?;
    }
    if requirements.pipeline_count == 0 {
        return Ok(());
    }
    let kernel_kind = grouped_kernel_kind(requirements);
    let pipeline = cached_pipeline(kernel_kind, requirements.codec, None)?;
    let vectors = requirements.vectors as u64;
    let columns = requirements.columns as u64;
    let encoded_row_bytes = requirements.encoded_row_bytes as u64;
    let mut member_rows = [0u64; 4];
    for (index, member) in requirements.members.iter().enumerate() {
        member_rows[index] = member.rows as u64;
    }
    let rows_per_threadgroup = rows_per_simd(requirements.codec) * 2;
    let row_groups = requirements
        .members
        .iter()
        .map(|member| member.rows.div_ceil(rows_per_threadgroup))
        .max()
        .expect("grouped requirements hold 2..=4 members");
    let vector_groups = match kernel_kind {
        KernelKind::LinearGrouped(1) => requirements.vectors,
        KernelKind::LinearGrouped(vector_lanes) => {
            requirements.vectors.div_ceil(usize::from(vector_lanes))
        }
        _ => unreachable!("grouped kernel kind is always LinearGrouped"),
    };
    MetalDevice::get().with_encoder(|encoder| {
        encoder.setComputePipelineState(pipeline.as_raw());
        set_buffer(encoder, 0, &input.buffer, input.layout.offset() * 4);
        for slot in 0..4 {
            let (weight, output) = members.get(slot).copied().unwrap_or_else(|| members[0]);
            set_buffer(encoder, 1 + slot, &weight.buffer, 0);
            set_buffer(
                encoder,
                5 + slot,
                &output.buffer,
                output.layout.offset() * 4,
            );
        }
        set_bytes(encoder, 9, &vectors);
        set_bytes(encoder, 10, &member_rows);
        set_bytes(encoder, 11, &columns);
        set_bytes(encoder, 12, &encoded_row_bytes);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MetalDevice::grid(row_groups, vector_groups, requirements.members.len()),
            MetalDevice::grid(64, 1, 1),
        );
    });
    Ok(())
}

/// Non-allocating quantized linear dispatch: validates every argument
/// against the immutable plan and encodes the selected kernel.
/// Allocates nothing; requires the exact pipeline to be warm. A plan
/// with `pipeline_count == 0` (empty output) is a no-op.
pub fn linear_into(
    input: &MetalTensor,
    weight: &MetalTensor,
    bias: Option<&MetalTensor>,
    output: &MetalTensor,
    requirements: &LinearRequirements,
) -> Result<(), String> {
    validate_linear(input, weight, bias, output, requirements)?;
    if requirements.pipeline_count == 0 {
        return Ok(());
    }
    encode_linear(
        input,
        weight,
        bias,
        output,
        requirements,
        linear_kernel_kind(requirements),
    )
}

/// Encodes one quantized linear dispatch with an explicit kernel kind.
/// `linear_into` passes the selected kind; tests use this to run the
/// f32-operand MMA variants against the default half-operand ones.
fn encode_linear(
    input: &MetalTensor,
    weight: &MetalTensor,
    bias: Option<&MetalTensor>,
    output: &MetalTensor,
    requirements: &LinearRequirements,
    kernel_kind: KernelKind,
) -> Result<(), String> {
    use crate::runtime::metal::device::{set_buffer, set_bytes, MetalDevice};
    use objc2_metal::MTLComputeCommandEncoder;

    let pipeline = cached_pipeline(kernel_kind, requirements.codec, None)?;
    let has_bias = u32::from(requirements.has_bias);
    let vectors = requirements.vectors as u64;
    let rows = requirements.rows as u64;
    let columns = requirements.columns as u64;
    let encoded_row_bytes = requirements.encoded_row_bytes as u64;
    let rows_per_threadgroup = rows_per_simd(requirements.codec) * 2;
    let (row_groups, threads_per_threadgroup) = match kernel_kind {
        KernelKind::LinearMma | KernelKind::LinearMma32 | KernelKind::LinearMma8 => {
            (requirements.rows.div_ceil(16), 256)
        }
        KernelKind::LinearMmaSimple
        | KernelKind::LinearMma32Simple
        | KernelKind::LinearMma8Simple => (requirements.rows.div_ceil(16), 128),
        KernelKind::LinearMmaHalf | KernelKind::LinearMma32Half | KernelKind::LinearMma8Half => {
            (requirements.rows.div_ceil(64), 128)
        }
        KernelKind::LinearMmaSimpleHalf
        | KernelKind::LinearMma32SimpleHalf
        | KernelKind::LinearMma8SimpleHalf
        | KernelKind::LinearMma32SwzHalf => (requirements.rows.div_ceil(32), 128),
        KernelKind::LinearDirect(_) => (requirements.rows.div_ceil(2), 64),
        _ => (requirements.rows.div_ceil(rows_per_threadgroup), 64),
    };
    let vector_groups = match kernel_kind {
        KernelKind::LinearMma
        | KernelKind::LinearMmaSimple
        | KernelKind::LinearMmaHalf
        | KernelKind::LinearMmaSimpleHalf => requirements.vectors.div_ceil(16),
        KernelKind::LinearMma32
        | KernelKind::LinearMma32Simple
        | KernelKind::LinearMma32Half
        | KernelKind::LinearMma32SimpleHalf => requirements.vectors.div_ceil(32),
        KernelKind::LinearMma32SwzHalf => requirements.vectors.div_ceil(32),
        KernelKind::LinearMma8
        | KernelKind::LinearMma8Simple
        | KernelKind::LinearMma8Half
        | KernelKind::LinearMma8SimpleHalf => requirements.vectors.div_ceil(8),
        KernelKind::LinearBatched(vector_lanes) => {
            requirements.vectors.div_ceil(usize::from(vector_lanes))
        }
        KernelKind::LinearDirect(vectors) => {
            requirements
                .vectors
                .div_ceil(usize::from(vectors).min(match requirements.codec {
                    GgmlKQuant::Q2K => 4,
                    GgmlKQuant::Q3K => 2,
                    _ => unreachable!("direct row reuse only supports Q2_K/Q3_K"),
                }))
        }
        KernelKind::Linear | KernelKind::Embedding => requirements.vectors,
        KernelKind::LinearGrouped(_) => {
            unreachable!("grouped dispatch computes its own grid")
        }
    };
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
        let (grid_x, grid_y) = if kernel_kind == KernelKind::LinearMma32SwzHalf {
            (vector_groups, row_groups)
        } else {
            (row_groups, vector_groups)
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MetalDevice::grid(grid_x, grid_y, 1),
            MetalDevice::grid(threads_per_threadgroup, 1, 1),
        );
    });
    Ok(())
}

/// Non-allocating quantized embedding dispatch: decodes the selected
/// packed rows into `output` and zeroes then conditionally sets the
/// u32 `status` word when any index is out of range. Allocates
/// nothing; requires the exact pipeline to be warm.
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
        // SAFETY: `validate_embedding` just verified that `status` is a
        // contiguous u32 `[1]` view whose byte range lies within its
        // buffer, so `contents_ptr() + offset` addresses a valid u32.
        // The buffer uses shared storage (host-visible), the view is
        // exclusively owned by this dispatch, and the write lands
        // before the encoder submits the kernel that reads it.
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
            pipeline_key(KernelKind::Linear, GgmlKQuant::Q2K, None),
            pipeline_key(KernelKind::LinearBatched(2), GgmlKQuant::Q2K, None)
        );
        assert_ne!(
            pipeline_key(KernelKind::LinearBatched(2), GgmlKQuant::Q2K, None),
            pipeline_key(KernelKind::LinearBatched(4), GgmlKQuant::Q2K, None)
        );
        assert_ne!(
            pipeline_key(KernelKind::LinearBatched(4), GgmlKQuant::Q2K, None),
            pipeline_key(KernelKind::LinearMma, GgmlKQuant::Q2K, None)
        );
        assert_ne!(
            pipeline_key(KernelKind::LinearDirect(2), GgmlKQuant::Q2K, None),
            pipeline_key(KernelKind::LinearDirect(8), GgmlKQuant::Q2K, None)
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
    fn grouped_requirements_validate_members() {
        let member = |rows: usize, vectors: usize, codec: GgmlKQuant, bias: bool| {
            let encoded_row_bytes = codec.encoded_row_bytes(256).unwrap();
            let bias_shape = [rows];
            linear_requirements(
                &[vectors, 256],
                DType::F32,
                &[rows, encoded_row_bytes],
                DType::U8,
                bias.then_some((&bias_shape[..], DType::F32)),
                &[vectors, rows],
                DType::F32,
                codec,
                [rows, 256],
            )
            .unwrap()
        };
        let grouped = grouped_linear_requirements(&[
            member(3, 1, GgmlKQuant::Q4K, false),
            member(5, 1, GgmlKQuant::Q4K, false),
        ])
        .unwrap();
        assert_eq!(grouped.members.len(), 2);
        assert_eq!(grouped.output_bytes, (3 + 5) * 4);
        assert_eq!(grouped.pipeline_count, 1);
        assert!(matches!(
            grouped_kernel_kind(&grouped),
            KernelKind::LinearGrouped(1)
        ));

        assert!(
            grouped_linear_requirements(&[member(3, 1, GgmlKQuant::Q4K, false)])
                .unwrap_err()
                .contains("2..=4")
        );
        assert!(
            grouped_linear_requirements(&vec![member(1, 1, GgmlKQuant::Q4K, false); 5])
                .unwrap_err()
                .contains("2..=4")
        );
        assert!(grouped_linear_requirements(&[
            member(3, 1, GgmlKQuant::Q4K, false),
            member(3, 1, GgmlKQuant::Q6K, false),
        ])
        .unwrap_err()
        .contains("share codec"));
        assert!(grouped_linear_requirements(&[
            member(3, 1, GgmlKQuant::Q4K, false),
            member(3, 2, GgmlKQuant::Q4K, false),
        ])
        .unwrap_err()
        .contains("share codec"));
        assert!(grouped_linear_requirements(&[
            member(3, 1, GgmlKQuant::Q4K, true),
            member(3, 1, GgmlKQuant::Q4K, false),
        ])
        .unwrap_err()
        .contains("bias"));
        assert!(grouped_linear_requirements(&[
            member(3, 16, GgmlKQuant::Q4K, false),
            member(5, 16, GgmlKQuant::Q4K, false),
        ])
        .unwrap_err()
        .contains("non-MMA"));
    }

    #[test]
    fn grouped_linear_preserves_values_across_members_and_codecs() {
        let device = MetalDevice::get();
        let columns = 512usize;
        let vectors = 1usize;
        let input_values = (0..vectors * columns)
            .map(|index| ((index * 7 + 3) % 17) as f32 * 0.125 - 1.0)
            .collect::<Vec<_>>();
        let input = MetalTensor::from_f32(device, input_values, vec![vectors, columns]);
        for codec in [
            GgmlKQuant::Q2K,
            GgmlKQuant::Q3K,
            GgmlKQuant::Q4K,
            GgmlKQuant::Q5K,
            GgmlKQuant::Q6K,
        ] {
            let encoded_row_bytes = codec.encoded_row_bytes(columns).unwrap();
            let mut weights = Vec::new();
            let mut plans = Vec::new();
            let mut references = Vec::new();
            for (member_index, rows) in [3usize, 5].into_iter().enumerate() {
                let mut packed = (0..rows * encoded_row_bytes)
                    .map(|index| ((index * 31 + member_index * 17 + 7) % 251) as u8)
                    .collect::<Vec<_>>();
                // Keep the f16 super-scales finite so the dotted sums
                // stay meaningful (other bytes may decode to anything).
                let block_bytes = encoded_row_bytes / (columns / 256);
                for block in packed.chunks_exact_mut(block_bytes) {
                    match codec {
                        GgmlKQuant::Q2K => block[80..84].copy_from_slice(&[0x00, 0x3c, 0x00, 0x38]),
                        GgmlKQuant::Q3K => block[108..110].copy_from_slice(&[0x00, 0x3c]),
                        GgmlKQuant::Q4K | GgmlKQuant::Q5K => {
                            block[0..4].copy_from_slice(&[0x00, 0x3c, 0x00, 0x38])
                        }
                        GgmlKQuant::Q6K => block[208..210].copy_from_slice(&[0x00, 0x38]),
                    }
                }
                let weight = MetalTensor {
                    buffer: device.upload_bytes(&packed),
                    layout: Layout::contiguous(vec![rows, encoded_row_bytes]),
                    dtype: DType::U8,
                };
                let requirements = linear_requirements(
                    &[vectors, columns],
                    DType::F32,
                    &[rows, encoded_row_bytes],
                    DType::U8,
                    None,
                    &[vectors, rows],
                    DType::F32,
                    codec,
                    [rows, columns],
                )
                .unwrap();
                let reference =
                    MetalTensor::from_f32(device, vec![0.0; vectors * rows], vec![vectors, rows]);
                warm_linear_exact(&requirements).unwrap();
                linear_into(&input, &weight, None, &reference, &requirements).unwrap();
                weights.push(weight);
                plans.push(requirements);
                references.push(reference);
            }
            let grouped = grouped_linear_requirements(&plans).unwrap();
            let outputs = plans
                .iter()
                .map(|member| {
                    MetalTensor::from_f32(
                        device,
                        vec![f32::NAN; member.vectors * member.rows],
                        vec![member.vectors, member.rows],
                    )
                })
                .collect::<Vec<_>>();
            warm_linear_grouped_exact(&grouped).unwrap();
            let members = weights
                .iter()
                .zip(&outputs)
                .map(|(weight, output)| (weight, output))
                .collect::<Vec<_>>();
            linear_grouped_into(&input, &members, &grouped).unwrap();
            device.synchronize().unwrap();

            for (member_index, ((plan, output), reference)) in
                plans.iter().zip(&outputs).zip(&references).enumerate()
            {
                let actual = output.buffer.contents_ptr().cast::<f32>();
                let expected = reference.buffer.contents_ptr().cast::<f32>();
                for index in 0..plan.vectors * plan.rows {
                    let (actual, expected) = unsafe { (*actual.add(index), *expected.add(index)) };
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{} member={member_index} index={index}: {actual} != {expected}",
                        codec.name()
                    );
                    assert!(!actual.is_nan(), "{} produced NaN", codec.name());
                }
            }
        }
    }

    #[test]
    fn grouped_execution_rejects_nonzero_offset_packed_rows() {
        let member = linear_requirements(
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
        let grouped = grouped_linear_requirements(&[member.clone(), member]).unwrap();
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
        let error =
            linear_grouped_into(&input, &[(&weight, &output), (&weight, &output)], &grouped)
                .unwrap_err();
        assert!(error.contains("packed weight must be zero-offset contiguous"));
    }

    #[test]
    fn q2_q3_direct_rows_match_sequential_decode_bitwise() {
        let device = MetalDevice::get();
        let rows = 5usize;
        let columns = 512usize;
        for codec in [GgmlKQuant::Q2K, GgmlKQuant::Q3K] {
            let encoded_row_bytes = codec.encoded_row_bytes(columns).unwrap();
            let block_bytes = block_bytes(codec);
            let mut packed = (0..rows * encoded_row_bytes)
                .map(|index| ((index * 31 + 7) % 251) as u8)
                .collect::<Vec<_>>();
            for block in packed.chunks_exact_mut(block_bytes) {
                match codec {
                    GgmlKQuant::Q2K => block[80..84].copy_from_slice(&[0x00, 0x38, 0x00, 0x34]),
                    GgmlKQuant::Q3K => block[108..110].copy_from_slice(&[0x00, 0x38]),
                    _ => unreachable!(),
                }
            }
            let weight = MetalTensor {
                buffer: device.upload_bytes(&packed),
                layout: Layout::contiguous(vec![rows, encoded_row_bytes]),
                dtype: DType::U8,
            };
            let bias_values = (0..rows)
                .map(|row| row as f32 * 0.03125 - 0.0625)
                .collect::<Vec<_>>();
            let bias = MetalTensor::from_f32(device, bias_values, vec![rows]);
            for vectors in 2usize..=8 {
                let input_values = (0..vectors * columns)
                    .map(|index| ((index * 13 + vectors * 5) % 97) as f32 * 0.013 - 0.6)
                    .collect::<Vec<_>>();
                let input =
                    MetalTensor::from_f32(device, input_values.clone(), vec![vectors, columns]);
                let requirements = linear_requirements(
                    &[vectors, columns],
                    DType::F32,
                    &[rows, encoded_row_bytes],
                    DType::U8,
                    Some((&[rows], DType::F32)),
                    &[vectors, rows],
                    DType::F32,
                    codec,
                    [rows, columns],
                )
                .unwrap();
                let direct_kind = KernelKind::LinearDirect(vectors as u8);
                let output = MetalTensor::from_f32(
                    device,
                    vec![f32::NAN; vectors * rows],
                    vec![vectors, rows],
                );
                pipeline(direct_kind, codec, None).unwrap();
                encode_linear(
                    &input,
                    &weight,
                    Some(&bias),
                    &output,
                    &requirements,
                    direct_kind,
                )
                .unwrap();

                let mut references = Vec::with_capacity(vectors);
                for vector in 0..vectors {
                    let reference_input = MetalTensor::from_f32(
                        device,
                        input_values[vector * columns..(vector + 1) * columns].to_vec(),
                        vec![1, columns],
                    );
                    let reference_requirements = linear_requirements(
                        &[1, columns],
                        DType::F32,
                        &[rows, encoded_row_bytes],
                        DType::U8,
                        Some((&[rows], DType::F32)),
                        &[1, rows],
                        DType::F32,
                        codec,
                        [rows, columns],
                    )
                    .unwrap();
                    let reference =
                        MetalTensor::from_f32(device, vec![f32::NAN; rows], vec![1, rows]);
                    warm_linear_exact(&reference_requirements).unwrap();
                    linear_into(
                        &reference_input,
                        &weight,
                        Some(&bias),
                        &reference,
                        &reference_requirements,
                    )
                    .unwrap();
                    references.push(reference);
                }
                device.synchronize().unwrap();

                let actual = output.buffer.contents_ptr().cast::<f32>();
                for (vector, reference) in references.iter().enumerate() {
                    let expected = reference.buffer.contents_ptr().cast::<f32>();
                    for row in 0..rows {
                        let (actual, expected) =
                            unsafe { (*actual.add(vector * rows + row), *expected.add(row)) };
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "{} vectors={vectors} vector={vector} row={row}: {actual} != {expected}",
                            codec.name()
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "manual prefill kernel bandwidth probe"]
    fn kquant_prefill_bandwidth_probe() {
        // Ramp GPU clocks with throwaway dispatches before timing.
        {
            let device = MetalDevice::get();
            let requirements = linear_requirements(
                &[32, 4096],
                DType::F32,
                &[8192, 1760],
                DType::U8,
                None,
                &[32, 8192],
                DType::F32,
                GgmlKQuant::Q3K,
                [8192, 4096],
            )
            .unwrap();
            let input = MetalTensor::from_f32(device, vec![1.0; 32 * 4096], vec![32, 4096]);
            let weight = MetalTensor {
                buffer: device.upload_bytes(&vec![0u8; 8192 * 1760]),
                layout: Layout::contiguous(vec![8192, 1760]),
                dtype: DType::U8,
            };
            let output = MetalTensor::from_f32(device, vec![0.0; 32 * 8192], vec![32, 8192]);
            warm_linear_exact(&requirements).unwrap();
            for _ in 0..100 {
                linear_into(&input, &weight, None, &output, &requirements).unwrap();
            }
            device.synchronize().unwrap();
        }
        for (codec, vectors, rows, columns, block_bytes) in [
            (GgmlKQuant::Q3K, 32usize, 6656usize, 4096usize, 110usize),
            (GgmlKQuant::Q3K, 16usize, 6656usize, 4096usize, 110usize),
            (GgmlKQuant::Q3K, 1, 6656, 4096, 110),
            (GgmlKQuant::Q3K, 2, 6656, 4096, 110),
            (GgmlKQuant::Q3K, 3, 6656, 4096, 110),
            (GgmlKQuant::Q3K, 4, 6656, 4096, 110),
            (GgmlKQuant::Q3K, 5, 6656, 4096, 110),
            (GgmlKQuant::Q3K, 6, 6656, 4096, 110),
            (GgmlKQuant::Q3K, 7, 6656, 4096, 110),
            (GgmlKQuant::Q3K, 8, 6656, 4096, 110),
            (GgmlKQuant::Q3K, 32, 19968, 6656, 110),
            (GgmlKQuant::Q3K, 256, 19968, 6656, 110),
            (GgmlKQuant::Q3K, 256, 6656, 4096, 110),
            (GgmlKQuant::Q3K, 16, 19968, 6656, 110),
            (GgmlKQuant::Q3K, 8, 19968, 6656, 110),
            (GgmlKQuant::Q3K, 16, 6656, 19968, 110),
            (GgmlKQuant::Q3K, 8, 6656, 19968, 110),
            (GgmlKQuant::Q2K, 32, 4096, 6656, 84),
            (GgmlKQuant::Q2K, 16, 4096, 6656, 84),
            (GgmlKQuant::Q2K, 1, 4096, 6656, 84),
            (GgmlKQuant::Q2K, 2, 4096, 6656, 84),
            (GgmlKQuant::Q2K, 3, 4096, 6656, 84),
            (GgmlKQuant::Q2K, 4, 4096, 6656, 84),
            (GgmlKQuant::Q2K, 5, 4096, 6656, 84),
            (GgmlKQuant::Q2K, 6, 4096, 6656, 84),
            (GgmlKQuant::Q2K, 7, 4096, 6656, 84),
            (GgmlKQuant::Q2K, 8, 4096, 6656, 84),
            (GgmlKQuant::Q2K, 32, 19968, 6656, 84),
            (GgmlKQuant::Q2K, 256, 19968, 6656, 84),
            (GgmlKQuant::Q2K, 256, 4096, 6656, 84),
            (GgmlKQuant::Q2K, 16, 19968, 6656, 84),
            (GgmlKQuant::Q2K, 8, 19968, 6656, 84),
            (GgmlKQuant::Q4K, 32, 6656, 4096, 144),
            (GgmlKQuant::Q4K, 16, 6656, 4096, 144),
            (GgmlKQuant::Q4K, 8, 6656, 4096, 144),
            (GgmlKQuant::Q5K, 16, 6656, 6656, 176),
            (GgmlKQuant::Q5K, 8, 6656, 6656, 176),
            (GgmlKQuant::Q6K, 32, 202048, 6656, 210),
            (GgmlKQuant::Q6K, 16, 202048, 6656, 210),
            (GgmlKQuant::Q6K, 8, 202048, 6656, 210),
        ] {
            let encoded_row_bytes = columns / 256 * block_bytes;
            let requirements = linear_requirements(
                &[vectors, columns],
                DType::F32,
                &[rows, encoded_row_bytes],
                DType::U8,
                None,
                &[vectors, rows],
                DType::F32,
                codec,
                [rows, columns],
            )
            .unwrap();
            let device = MetalDevice::get();
            let input =
                MetalTensor::from_f32(device, vec![1.0; vectors * columns], vec![vectors, columns]);
            let weight = MetalTensor {
                buffer: device.upload_bytes(&vec![0u8; requirements.encoded_row_bytes * rows]),
                layout: Layout::contiguous(vec![rows, requirements.encoded_row_bytes]),
                dtype: DType::U8,
            };
            let output =
                MetalTensor::from_f32(device, vec![0.0; vectors * rows], vec![vectors, rows]);
            warm_linear_exact(&requirements).unwrap();
            // Same-session before/after: time the f32-operand MMA twin
            // alongside the default-selected (half) kernel.
            let kernel_kind = linear_kernel_kind(&requirements);
            let f32_kind = match kernel_kind {
                KernelKind::LinearMmaHalf => Some(KernelKind::LinearMma),
                KernelKind::LinearMmaSimpleHalf => Some(KernelKind::LinearMmaSimple),
                KernelKind::LinearMma32Half => Some(KernelKind::LinearMma32),
                KernelKind::LinearMma32SimpleHalf => Some(KernelKind::LinearMma32Simple),
                KernelKind::LinearMma8Half => Some(KernelKind::LinearMma8),
                KernelKind::LinearMma8SimpleHalf => Some(KernelKind::LinearMma8Simple),
                _ => None,
            };
            if let Some(f32_kind) = f32_kind {
                pipeline(f32_kind, codec, None).unwrap();
            }
            for _ in 0..5 {
                linear_into(&input, &weight, None, &output, &requirements).unwrap();
            }
            device.synchronize().unwrap();
            let started = std::time::Instant::now();
            for _ in 0..20 {
                linear_into(&input, &weight, None, &output, &requirements).unwrap();
            }
            device.synchronize().unwrap();
            let seconds = started.elapsed().as_secs_f64();
            let bytes = (requirements.encoded_row_bytes * rows) as f64;
            eprintln!(
                "{} {vectors}x{rows}x{columns}: {:.3} ms/call, {:.1} GB/s packed",
                codec.name(),
                seconds * 50.0,
                bytes * 20.0 / seconds / 1e9
            );
            if let Some(f32_kind) = f32_kind {
                for _ in 0..5 {
                    encode_linear(&input, &weight, None, &output, &requirements, f32_kind).unwrap();
                }
                device.synchronize().unwrap();
                let started = std::time::Instant::now();
                for _ in 0..20 {
                    encode_linear(&input, &weight, None, &output, &requirements, f32_kind).unwrap();
                }
                device.synchronize().unwrap();
                let seconds = started.elapsed().as_secs_f64();
                eprintln!(
                    "{} {vectors}x{rows}x{columns} [f32 operands]: {:.3} ms/call, {:.1} GB/s packed",
                    codec.name(),
                    seconds * 50.0,
                    bytes * 20.0 / seconds / 1e9
                );
            }
            // Alternate row-tile geometry (64 vs 32 rows).
            let alt = match kernel_kind {
                KernelKind::LinearMmaHalf => Some(KernelKind::LinearMmaSimpleHalf),
                KernelKind::LinearMmaSimpleHalf => Some(KernelKind::LinearMmaHalf),
                KernelKind::LinearMma32Half => Some(KernelKind::LinearMma32SimpleHalf),
                KernelKind::LinearMma32SimpleHalf => Some(KernelKind::LinearMma32Half),
                KernelKind::LinearMma8Half => Some(KernelKind::LinearMma8SimpleHalf),
                KernelKind::LinearMma8SimpleHalf => Some(KernelKind::LinearMma8Half),
                _ => None,
            };
            if let Some(alt) = alt {
                pipeline(alt, codec, None).unwrap();
                for _ in 0..5 {
                    encode_linear(&input, &weight, None, &output, &requirements, alt).unwrap();
                }
                device.synchronize().unwrap();
                let started = std::time::Instant::now();
                for _ in 0..20 {
                    encode_linear(&input, &weight, None, &output, &requirements, alt).unwrap();
                }
                device.synchronize().unwrap();
                let seconds = started.elapsed().as_secs_f64();
                eprintln!(
                    "{} {vectors}x{rows}x{columns} [other tile_n]: {:.3} ms/call, {:.1} GB/s packed",
                    codec.name(),
                    seconds * 50.0,
                    bytes * 20.0 / seconds / 1e9
                );
            }
            // 64-vector tile (halves the weight-tile dequant replication).
            if kernel_kind == KernelKind::LinearMma32SimpleHalf {
                for (m64, tag) in [(KernelKind::LinearMma32SwzHalf, "swizzled")] {
                    pipeline(m64, codec, None).unwrap();
                    for _ in 0..5 {
                        encode_linear(&input, &weight, None, &output, &requirements, m64).unwrap();
                    }
                    device.synchronize().unwrap();
                    let started = std::time::Instant::now();
                    for _ in 0..20 {
                        encode_linear(&input, &weight, None, &output, &requirements, m64).unwrap();
                    }
                    device.synchronize().unwrap();
                    let seconds = started.elapsed().as_secs_f64();
                    eprintln!(
                        "{} {vectors}x{rows}x{columns} [{tag}]: {:.3} ms/call, {:.1} GB/s packed",
                        codec.name(),
                        seconds * 50.0,
                        bytes * 20.0 / seconds / 1e9
                    );
                }
            }
            if (2..=8).contains(&requirements.vectors)
                && matches!(codec, GgmlKQuant::Q2K | GgmlKQuant::Q3K)
            {
                let legacy = match codec {
                    GgmlKQuant::Q2K => KernelKind::LinearBatched(2),
                    GgmlKQuant::Q3K => KernelKind::LinearBatched(4),
                    _ => unreachable!(),
                };
                let mut candidates = vec![
                    (
                        KernelKind::LinearDirect(requirements.vectors as u8),
                        "direct-row-reuse",
                    ),
                    (KernelKind::Linear, "sequential-qmv"),
                    (legacy, "legacy-batched"),
                ];
                if requirements.vectors == 8 {
                    candidates.push((KernelKind::LinearMma8SimpleHalf, "half-mma"));
                }
                for (candidate, tag) in candidates {
                    pipeline(candidate, codec, None).unwrap();
                    for _ in 0..5 {
                        encode_linear(&input, &weight, None, &output, &requirements, candidate)
                            .unwrap();
                    }
                    device.synchronize().unwrap();
                    let started = std::time::Instant::now();
                    for _ in 0..20 {
                        encode_linear(&input, &weight, None, &output, &requirements, candidate)
                            .unwrap();
                    }
                    device.synchronize().unwrap();
                    let seconds = started.elapsed().as_secs_f64();
                    eprintln!(
                        "{} {vectors}x{rows}x{columns} [{tag}]: {:.3} ms/call, {:.1} GB/s packed",
                        codec.name(),
                        seconds * 50.0,
                        bytes * 20.0 / seconds / 1e9
                    );
                }
            }
        }
    }

    #[test]
    fn kquant_mma_preserves_values_across_tiles() {
        let vectors = 16usize;
        let rows = 129usize;
        let columns = 256usize;
        let encoded_row_bytes = 84usize;
        let requirements = linear_requirements(
            &[vectors, columns],
            DType::F32,
            &[rows, encoded_row_bytes],
            DType::U8,
            None,
            &[vectors, rows],
            DType::F32,
            GgmlKQuant::Q2K,
            [rows, columns],
        )
        .unwrap();
        assert!(matches!(
            linear_kernel_kind(&requirements),
            KernelKind::LinearMmaSimpleHalf
        ));

        let mut packed = vec![0u8; encoded_row_bytes * rows];
        for block in packed.chunks_exact_mut(encoded_row_bytes) {
            block[..16].fill(1);
            block[16..80].fill(0xff);
            block[80] = 0;
            block[81] = 0x3c;
        }
        let device = MetalDevice::get();
        let input =
            MetalTensor::from_f32(device, vec![1.0; vectors * columns], vec![vectors, columns]);
        let weight = MetalTensor {
            buffer: device.upload_bytes(&packed),
            layout: Layout::contiguous(vec![rows, encoded_row_bytes]),
            dtype: DType::U8,
        };
        let output = MetalTensor::from_f32(device, vec![0.0; vectors * rows], vec![vectors, rows]);
        warm_linear_exact(&requirements).unwrap();
        linear_into(&input, &weight, None, &output, &requirements).unwrap();
        device.synchronize().unwrap();

        let values = output.buffer.contents_ptr().cast::<f32>();
        for vector in [0usize, 7, 8, 15] {
            for row in [0usize, 31, 32, 63, 64, 127, 128] {
                let actual = unsafe { *values.add(vector * rows + row) };
                assert_eq!(actual, 3.0 * columns as f32, "vector={vector} row={row}");
            }
        }
    }

    #[test]
    fn kquant_mma8_preserves_values_across_tiles() {
        let vectors = 8usize;
        let rows = 129usize;
        let columns = 256usize;
        let encoded_row_bytes = 84usize;
        let requirements = linear_requirements(
            &[vectors, columns],
            DType::F32,
            &[rows, encoded_row_bytes],
            DType::U8,
            None,
            &[vectors, rows],
            DType::F32,
            GgmlKQuant::Q2K,
            [rows, columns],
        )
        .unwrap();
        assert!(matches!(
            linear_kernel_kind(&requirements),
            KernelKind::LinearMma8SimpleHalf
        ));

        let mut packed = vec![0u8; encoded_row_bytes * rows];
        for block in packed.chunks_exact_mut(encoded_row_bytes) {
            block[..16].fill(1);
            block[16..80].fill(0xff);
            block[80] = 0;
            block[81] = 0x3c;
        }
        let device = MetalDevice::get();
        let input =
            MetalTensor::from_f32(device, vec![1.0; vectors * columns], vec![vectors, columns]);
        let weight = MetalTensor {
            buffer: device.upload_bytes(&packed),
            layout: Layout::contiguous(vec![rows, encoded_row_bytes]),
            dtype: DType::U8,
        };
        let output = MetalTensor::from_f32(device, vec![0.0; vectors * rows], vec![vectors, rows]);
        warm_linear_exact(&requirements).unwrap();
        linear_into(&input, &weight, None, &output, &requirements).unwrap();
        device.synchronize().unwrap();

        let values = output.buffer.contents_ptr().cast::<f32>();
        for vector in [0usize, 7] {
            for row in [0usize, 31, 32, 63, 64, 127, 128] {
                let actual = unsafe { *values.add(vector * rows + row) };
                assert_eq!(actual, 3.0 * columns as f32, "vector={vector} row={row}");
            }
        }
    }

    #[test]
    fn kquant_pipelined_mma_preserves_values_for_32_16_and_8_vectors() {
        let rows = 33usize;
        let columns = 4352usize;
        let encoded_row_bytes = 1870usize;
        let mut packed = vec![0u8; encoded_row_bytes * rows];
        for row in packed.chunks_exact_mut(encoded_row_bytes) {
            for block in row.chunks_exact_mut(110) {
                block[108] = 0x00;
                block[109] = 0x3c;
            }
        }
        let device = MetalDevice::get();
        for vectors in [32usize, 16, 8] {
            let requirements = linear_requirements(
                &[vectors, columns],
                DType::F32,
                &[rows, encoded_row_bytes],
                DType::U8,
                None,
                &[vectors, rows],
                DType::F32,
                GgmlKQuant::Q3K,
                [rows, columns],
            )
            .unwrap();
            assert!(matches!(
                linear_kernel_kind(&requirements),
                KernelKind::LinearMma32SimpleHalf
                    | KernelKind::LinearMma32SwzHalf
                    | KernelKind::LinearMmaSimpleHalf
                    | KernelKind::LinearMma8SimpleHalf
            ));
            let input =
                MetalTensor::from_f32(device, vec![1.0; vectors * columns], vec![vectors, columns]);
            let weight = MetalTensor {
                buffer: device.upload_bytes(&packed),
                layout: Layout::contiguous(vec![rows, encoded_row_bytes]),
                dtype: DType::U8,
            };
            let output =
                MetalTensor::from_f32(device, vec![0.0; vectors * rows], vec![vectors, rows]);
            warm_linear_exact(&requirements).unwrap();
            linear_into(&input, &weight, None, &output, &requirements).unwrap();
            device.synchronize().unwrap();

            let values = output.buffer.contents_ptr().cast::<f32>();
            for vector in [0usize, vectors - 1] {
                for row in [0usize, 15, 16, 31, 32] {
                    let actual = unsafe { *values.add(vector * rows + row) };
                    assert_eq!(actual, 128.0 * columns as f32, "vector={vector} row={row}");
                }
            }
        }
    }

    #[test]
    fn kquant_mma_half_matches_f32_mma_within_half_precision() {
        // Non-half-exact inputs and decodes: the only difference between
        // the default half-operand kernels and the f32 MMA kernels is
        // rounding both operands to half (11-bit significand, ≤ 2^-11
        // relative per operand, ≤ 2^-10 per product). Both kernels accumulate
        // in f32, so errors stay near sqrt(K) * 2^-10 of the per-product
        // magnitude, far below 1% of the largest dot.
        let device = MetalDevice::get();
        for (codec, vectors, rows, columns, block_bytes, f32_kind, half_kind) in [
            (
                GgmlKQuant::Q3K,
                32usize,
                65usize,
                4352usize,
                110usize,
                KernelKind::LinearMma32,
                KernelKind::LinearMma32SimpleHalf,
            ),
            (
                GgmlKQuant::Q2K,
                16,
                33,
                2560,
                84,
                KernelKind::LinearMmaSimple,
                KernelKind::LinearMmaSimpleHalf,
            ),
            (
                GgmlKQuant::Q6K,
                8,
                17,
                768,
                210,
                KernelKind::LinearMma8Simple,
                KernelKind::LinearMma8SimpleHalf,
            ),
            (
                GgmlKQuant::Q4K,
                8,
                19,
                768,
                144,
                KernelKind::LinearMma8Simple,
                KernelKind::LinearMma8SimpleHalf,
            ),
            (
                GgmlKQuant::Q5K,
                8,
                35,
                768,
                176,
                KernelKind::LinearMma8Simple,
                KernelKind::LinearMma8SimpleHalf,
            ),
        ] {
            let encoded_row_bytes = columns / 256 * block_bytes;
            let requirements = linear_requirements(
                &[vectors, columns],
                DType::F32,
                &[rows, encoded_row_bytes],
                DType::U8,
                None,
                &[vectors, rows],
                DType::F32,
                codec,
                [rows, columns],
            )
            .unwrap();
            assert_eq!(linear_kernel_kind(&requirements), half_kind);

            let mut packed = (0..rows * encoded_row_bytes)
                .map(|index| ((index * 31 + 7) % 251) as u8)
                .collect::<Vec<_>>();
            // Keep the f16 super-scales finite and modest so dots stay
            // in a meaningful range (other bytes may decode to anything).
            for block in packed.chunks_exact_mut(block_bytes) {
                match codec {
                    GgmlKQuant::Q2K => block[80..84].copy_from_slice(&[0x00, 0x3c, 0x00, 0x38]),
                    GgmlKQuant::Q3K => block[108..110].copy_from_slice(&[0x00, 0x3c]),
                    GgmlKQuant::Q4K | GgmlKQuant::Q5K => {
                        block[..4].copy_from_slice(&[0x00, 0x38, 0x00, 0x34])
                    }
                    GgmlKQuant::Q6K => block[208..210].copy_from_slice(&[0x00, 0x38]),
                }
            }
            let input_values = (0..vectors * columns)
                .map(|index| ((index * 13 + 5) % 97) as f32 * 0.013 - 0.6)
                .collect::<Vec<_>>();
            let input = MetalTensor::from_f32(device, input_values, vec![vectors, columns]);
            let weight = MetalTensor {
                buffer: device.upload_bytes(&packed),
                layout: Layout::contiguous(vec![rows, encoded_row_bytes]),
                dtype: DType::U8,
            };
            let half_output =
                MetalTensor::from_f32(device, vec![0.0; vectors * rows], vec![vectors, rows]);
            let f32_output =
                MetalTensor::from_f32(device, vec![0.0; vectors * rows], vec![vectors, rows]);
            // The 64-row (wide) half geometry is covered against the
            // same f32 reference.
            let wide_kind = match half_kind {
                KernelKind::LinearMma32SimpleHalf | KernelKind::LinearMma32SwzHalf => {
                    KernelKind::LinearMma32Half
                }
                KernelKind::LinearMmaSimpleHalf => KernelKind::LinearMmaHalf,
                KernelKind::LinearMma8SimpleHalf => KernelKind::LinearMma8Half,
                _ => unreachable!("parity test uses simple half kinds"),
            };
            let wide_output =
                MetalTensor::from_f32(device, vec![0.0; vectors * rows], vec![vectors, rows]);
            warm_linear_exact(&requirements).unwrap();
            pipeline(f32_kind, codec, None).unwrap();
            pipeline(wide_kind, codec, None).unwrap();
            linear_into(&input, &weight, None, &half_output, &requirements).unwrap();
            encode_linear(&input, &weight, None, &f32_output, &requirements, f32_kind).unwrap();
            encode_linear(
                &input,
                &weight,
                None,
                &wide_output,
                &requirements,
                wide_kind,
            )
            .unwrap();
            device.synchronize().unwrap();

            let half_values = half_output.buffer.contents_ptr().cast::<f32>();
            let wide_values = wide_output.buffer.contents_ptr().cast::<f32>();
            let f32_values = f32_output.buffer.contents_ptr().cast::<f32>();
            let mut largest = 0.0f32;
            let mut worst = 0.0f32;
            for index in 0..vectors * rows {
                let (half, wide, f32) = unsafe {
                    (
                        *half_values.add(index),
                        *wide_values.add(index),
                        *f32_values.add(index),
                    )
                };
                assert!(
                    half.is_finite() && wide.is_finite(),
                    "{} index={index} produced {half}/{wide}",
                    codec.name()
                );
                largest = largest.max(f32.abs());
                worst = worst.max((half - f32).abs()).max((wide - f32).abs());
            }
            let tolerance = 0.01 * largest.max(1.0);
            eprintln!(
                "{} {vectors}x{rows}x{columns}: worst |half - f32| = {worst}, tolerance {tolerance}",
                codec.name()
            );
            assert!(
                worst <= tolerance,
                "{} {vectors}x{rows}x{columns}: worst |half - f32| = {worst} exceeds {tolerance}",
                codec.name()
            );
        }
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

    /// Same-session A/B timing of patched `et_quantized_linear_mma_half`
    /// variants against the default-selected kernel, for kernel micro-
    /// optimization experiments. Variants are source-level patches of
    /// the exact production source of the selected kernel kind.
    #[test]
    #[ignore = "manual prefill kernel variant A/B probe"]
    fn kquant_prefill_variant_probe() {
        use crate::runtime::metal::device::{set_buffer, set_bytes};
        use objc2_metal::MTLComputeCommandEncoder;
        use std::hash::{Hash, Hasher};

        // Ramp GPU clocks with throwaway dispatches before timing.
        {
            let device = MetalDevice::get();
            let requirements = linear_requirements(
                &[32, 4096],
                DType::F32,
                &[8192, 1760],
                DType::U8,
                None,
                &[32, 8192],
                DType::F32,
                GgmlKQuant::Q3K,
                [8192, 4096],
            )
            .unwrap();
            let input = MetalTensor::from_f32(device, vec![1.0; 32 * 4096], vec![32, 4096]);
            let weight = MetalTensor {
                buffer: device.upload_bytes(&vec![0u8; 8192 * 1760]),
                layout: Layout::contiguous(vec![8192, 1760]),
                dtype: DType::U8,
            };
            let output = MetalTensor::from_f32(device, vec![0.0; 32 * 8192], vec![32, 8192]);
            warm_linear_exact(&requirements).unwrap();
            for _ in 0..100 {
                linear_into(&input, &weight, None, &output, &requirements).unwrap();
            }
            device.synchronize().unwrap();
        }

        for (codec, vectors, rows, columns, block_bytes) in [
            (GgmlKQuant::Q3K, 32usize, 6656usize, 4096usize, 110usize),
            (GgmlKQuant::Q3K, 32, 19968, 6656, 110),
            (GgmlKQuant::Q3K, 16, 19968, 6656, 110),
            (GgmlKQuant::Q3K, 8, 6656, 4096, 110),
            (GgmlKQuant::Q3K, 8, 19968, 6656, 110),
            (GgmlKQuant::Q3K, 8, 6656, 19968, 110),
            (GgmlKQuant::Q2K, 32, 4096, 6656, 84),
            (GgmlKQuant::Q2K, 32, 19968, 6656, 84),
            (GgmlKQuant::Q2K, 16, 19968, 6656, 84),
            (GgmlKQuant::Q2K, 8, 4096, 6656, 84),
            (GgmlKQuant::Q2K, 8, 19968, 6656, 84),
            (GgmlKQuant::Q4K, 8, 6656, 4096, 144),
            (GgmlKQuant::Q5K, 8, 6656, 6656, 176),
            (GgmlKQuant::Q6K, 8, 202048, 6656, 210),
        ] {
            let encoded_row_bytes = columns / 256 * block_bytes;
            let requirements = linear_requirements(
                &[vectors, columns],
                DType::F32,
                &[rows, encoded_row_bytes],
                DType::U8,
                None,
                &[vectors, rows],
                DType::F32,
                codec,
                [rows, columns],
            )
            .unwrap();
            let device = MetalDevice::get();
            let input =
                MetalTensor::from_f32(device, vec![1.0; vectors * columns], vec![vectors, columns]);
            let weight = MetalTensor {
                buffer: device.upload_bytes(&vec![0u8; requirements.encoded_row_bytes * rows]),
                layout: Layout::contiguous(vec![rows, requirements.encoded_row_bytes]),
                dtype: DType::U8,
            };
            let output =
                MetalTensor::from_f32(device, vec![0.0; vectors * rows], vec![vectors, rows]);
            warm_linear_exact(&requirements).unwrap();
            let kind = linear_kernel_kind(&requirements);
            let (tile_n, tile_m) = match kind {
                KernelKind::LinearMma32SimpleHalf => (32usize, 32usize),
                KernelKind::LinearMmaSimpleHalf => (32, 16),
                KernelKind::LinearMma8SimpleHalf => (32, 8),
                _ => unreachable!("probe shapes select simple half kinds"),
            };
            let variants: [(&str, &str, u64, Box<dyn FnOnce(String) -> String>); 15] = [
                // Default production source (current selection).
                (
                    "current",
                    "et_quantized_linear_mma_half",
                    128,
                    Box::new(|src| src),
                ),
                // Rollback geometry for selected output-partition kernels.
                (
                    "k-split",
                    "et_quantized_linear_mma_half",
                    128,
                    Box::new(|src| {
                        src.replace("#define ET_OUTPUT_SPLIT 1", "#define ET_OUTPUT_SPLIT 0")
                    }),
                ),
                // Assign complete output blocks to simdgroups and remove
                // the four-way split-K reduction.
                (
                    "output-split",
                    "et_quantized_linear_mma_half",
                    128,
                    Box::new(|src| {
                        src.replace("#define ET_OUTPUT_SPLIT 0", "#define ET_OUTPUT_SPLIT 1")
                    }),
                ),
                // Warp-specialized double-buffered twin, 256 threads.
                (
                    "ws",
                    "et_quantized_linear_mma_half_ws",
                    256,
                    Box::new(|src| src),
                ),
                // Warp-specialized twin on the 64-row geometry.
                (
                    "ws tile_n 64",
                    "et_quantized_linear_mma_half_ws",
                    256,
                    Box::new(|src| src.replace("#define ET_TILE_N 32", "#define ET_TILE_N 64")),
                ),
                // Generic per-byte decode (pre-optimization).
                (
                    "generic-decode",
                    "et_quantized_linear_mma_half",
                    128,
                    Box::new(|src| {
                        src.replace(
                            "et_decode_k16_half(block, (group_column % 256) / 16, temp_h);",
                            "et_decode_k16(block, (group_column % 256) / 16, temp_h, 1);",
                        )
                    }),
                ),
                // Geometry: 64-row tile.
                (
                    "tile_n 64",
                    "et_quantized_linear_mma_half",
                    128,
                    Box::new(|src| src.replace("#define ET_TILE_N 32", "#define ET_TILE_N 64")),
                ),
                // Geometry: doubled K tile (fewer barriers, each superblock
                // loaded half as often).
                (
                    "tile_k 128",
                    "et_quantized_linear_mma_half",
                    128,
                    Box::new(|src| {
                        src.replacen(
                            "constexpr uint tile_k = 2048 / tile_n;",
                            "constexpr uint tile_k = 4096 / tile_n;",
                            1,
                        )
                    }),
                ),
                // Geometry: quadrupled K tile (full superblock per tile).
                (
                    "tile_k 256",
                    "et_quantized_linear_mma_half",
                    128,
                    Box::new(|src| {
                        src.replacen(
                            "constexpr uint tile_k = 2048 / tile_n;",
                            "constexpr uint tile_k = 8192 / tile_n;",
                            1,
                        )
                    }),
                ),
                // Diagnostics: skip the MMA loop (decode+stage+barriers).
                (
                    "no-mma",
                    "et_quantized_linear_mma_half",
                    128,
                    Box::new(|src| {
                        src.replace(
                            "for (uint kb = uint(simd_group); kb < k_blocks; kb += 4) {",
                            "for (uint kb = 0; kb < 0; ++kb) {",
                        )
                    }),
                ),
                // Diagnostics: skip weight decode (MMA+stage+barriers).
                (
                    "no-decode",
                    "et_quantized_linear_mma_half",
                    128,
                    Box::new(|src| {
                        src.replace(
                            "if (item < decode_items && first_row + decode_row_r[round] < rows) {",
                            "if (false) {",
                        )
                    }),
                ),
                // Diagnostics: skip input staging loads and stores.
                (
                    "no-stage",
                    "et_quantized_linear_mma_half",
                    128,
                    Box::new(|src| src.replace("if (stage_active) {", "if (false) {")),
                ),
                // Diagnostics: no decode, no staging (MMA+stores+barriers).
                (
                    "mma-only",
                    "et_quantized_linear_mma_half",
                    128,
                    Box::new(|src| {
                        src.replace(
                            "if (item < decode_items && first_row + decode_row_r[round] < rows) {",
                            "if (false) {",
                        )
                        .replace("if (stage_active) {", "if (false) {")
                    }),
                ),
                // Diagnostics: drop both in-loop barriers (racy; timing
                // floor for the fully pipelined structure).
                (
                    "no-barrier",
                    "et_quantized_linear_mma_half",
                    128,
                    Box::new(|src| {
                        src.replace(
                        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n        // Stride 8 lands",
                        "        // Stride 8 lands",
                    )
                    .replace(
                        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n        // K-split MMA",
                        "        // K-split MMA",
                    )
                    }),
                ),
                // Diagnostics: no-mma without the in-loop barriers.
                (
                    "loads-only",
                    "et_quantized_linear_mma_half",
                    128,
                    Box::new(|src| {
                        src.replace(
                        "for (uint kb = uint(simd_group); kb < k_blocks; kb += 4) {",
                        "for (uint kb = 0; kb < 0; ++kb) {",
                    )
                    .replace(
                        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n        // Stride 8 lands",
                        "        // Stride 8 lands",
                    )
                    .replace(
                        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n        // K-split MMA",
                        "        // K-split MMA",
                    )
                    }),
                ),
            ];
            let vectors64 = requirements.vectors as u64;
            let rows64 = requirements.rows as u64;
            let columns64 = requirements.columns as u64;
            let encoded64 = requirements.encoded_row_bytes as u64;
            let mut runners = Vec::new();
            let mut names = Vec::new();
            for (label, entry, threads, patch) in variants {
                let patched = patch(source(kind, codec, None));
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                "kquant_prefill_variant_probe".hash(&mut hasher);
                label.hash(&mut hasher);
                patched.hash(&mut hasher);
                let pipeline = device.compile(hasher.finish(), &patched, entry).unwrap();
                if label == "current" {
                    use objc2_metal::MTLComputePipelineState;
                    eprintln!(
                        "    [occupancy]: maxThreads {} staticTgMem {}",
                        pipeline.as_raw().maxTotalThreadsPerThreadgroup(),
                        pipeline.as_raw().staticThreadgroupMemoryLength()
                    );
                }
                let tile_n = if label.contains("tile_n 64") {
                    64
                } else {
                    tile_n
                };
                let row_groups = requirements.rows.div_ceil(tile_n);
                let vector_groups = requirements.vectors.div_ceil(tile_m);
                let input_buffer = input.buffer.clone();
                let input_offset = input.layout.offset();
                let weight_buffer = weight.buffer.clone();
                let output_buffer = output.buffer.clone();
                let output_offset = output.layout.offset();
                names.push(label);
                runners.push(move || {
                    device.with_encoder(|encoder| {
                        encoder.setComputePipelineState(pipeline.as_raw());
                        set_buffer(encoder, 0, &input_buffer, input_offset * 4);
                        set_buffer(encoder, 1, &weight_buffer, 0);
                        set_buffer(encoder, 2, &output_buffer, output_offset * 4);
                        set_buffer(encoder, 3, &output_buffer, output_offset * 4);
                        set_bytes(encoder, 4, &vectors64);
                        set_bytes(encoder, 5, &rows64);
                        set_bytes(encoder, 6, &columns64);
                        set_bytes(encoder, 7, &encoded64);
                        set_bytes(encoder, 8, &0u32);
                        encoder.dispatchThreadgroups_threadsPerThreadgroup(
                            MetalDevice::grid(row_groups, vector_groups, 1),
                            MetalDevice::grid(threads as usize, 1, 1),
                        );
                    });
                });
            }
            // Interleave variants round-robin so thermal throttling
            // affects them equally; report the best of 3 repetitions.
            let mut best = vec![f64::INFINITY; runners.len()];
            for _ in 0..3 {
                for (index, run) in runners.iter().enumerate() {
                    for _ in 0..3 {
                        run();
                    }
                    device.synchronize().unwrap();
                    let started = std::time::Instant::now();
                    for _ in 0..10 {
                        run();
                    }
                    device.synchronize().unwrap();
                    best[index] = best[index].min(started.elapsed().as_secs_f64());
                }
            }
            let bytes = (requirements.encoded_row_bytes * requirements.rows) as f64;
            eprintln!("{} {vectors}x{rows}x{columns}:", codec.name());
            for (label, seconds) in names.iter().zip(best.iter()) {
                eprintln!(
                    "    [{}]: {:.3} ms/call, {:.1} GB/s packed",
                    label,
                    seconds * 100.0,
                    bytes * 10.0 / seconds / 1e9
                );
            }
        }
    }
}

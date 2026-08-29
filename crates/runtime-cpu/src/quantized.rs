//! GGML K-quant weight decoding and quantized inference kernels.
//!
//! The GGUF K-quant formats in [`GgmlKQuant`] are `Q2_K`, `Q3_K`, `Q4_K`,
//! `Q5_K`, and `Q6_K`. They store weights in packed little-endian blocks.
//! Each block encodes [`BLOCK_VALUES`] values, which equals 256, along with
//! f16 global scale and minimum fields, bit-packed per-group scales, and
//! quantized codes.
//! The codec decoders mirror upstream ggml's `block_q*_K` and
//! `dequantize_row_q*_K` exactly.
//!
//! Two kernels consume packed weights without materializing them:
//!
//! - `quantized_linear`, through `*_requirements` and `*_into`, computes
//!   `input[.., columns] × weight[rows, columns]ᵀ (+ bias)`, decoding one
//!   block at a time into a 256-element stack buffer and accumulating in
//!   `f32`.
//! - `quantized_embedding` decodes only rows selected by `u32` or `i64`
//!   indexes, producing `[indexes..., columns]` f32 output.
//!
//! Packed weights must be exact contiguous `[rows, encoded_row_bytes]` `u8`
//! tensors with zero layout offset. Both kernels check the
//! [`CancellationFlag`] for each vector and block and abort with
//! `"operation aborted"`.

use crate::{CpuBuffer, CpuDestination, CpuTensorRequirement, Tensor};
use effect_torch_runtime::{CancellationFlag, DType, GgmlKQuant};
use half::f16;

/// Number of logical values in one K-quant block, for all codecs.
const BLOCK_VALUES: usize = 256;

/// Plan for one quantized linear or embedding invocation.
///
/// `output` is always f32. `encoded_row_bytes` is the packed size of one
/// weight row (`columns / 256 * block_bytes(codec)`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedRequirements {
    pub(crate) output: CpuTensorRequirement,
    rows: usize,
    columns: usize,
    encoded_row_bytes: usize,
    has_bias: bool,
}

fn checked_product(values: &[usize], operation: &str) -> Result<usize, String> {
    values
        .iter()
        .try_fold(1usize, |total, &value| total.checked_mul(value))
        .ok_or_else(|| format!("{operation}: element count overflow"))
}

fn checked_f32_output(shape: &[usize], operation: &str) -> Result<CpuTensorRequirement, String> {
    checked_product(shape, operation)?
        .checked_mul(DType::F32.size_in_bytes())
        .ok_or_else(|| format!("{operation}: output byte size overflow"))?;
    Ok(CpuTensorRequirement::new(shape, DType::F32))
}

/// Packed byte size of one 256-value block for `codec` (matches ggml's
/// `sizeof(block_q*_K)`).
fn block_bytes(codec: GgmlKQuant) -> usize {
    match codec {
        GgmlKQuant::Q2K => 84,
        GgmlKQuant::Q3K => 110,
        GgmlKQuant::Q4K => 144,
        GgmlKQuant::Q5K => 176,
        GgmlKQuant::Q6K => 210,
    }
}

/// Reads a little-endian f16 field of a packed block as f32.
fn fp16_at(block: &[u8], offset: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([block[offset], block[offset + 1]])).to_f32()
}

/// Unpacks the 6-bit (scale, min) pair for group `index` from the shared
/// 12-byte scale field of Q4_K/Q5_K blocks (ggml's packed `get_scale_min_k4`).
fn scale_min_k4(index: usize, scales: &[u8]) -> (u8, u8) {
    if index < 4 {
        (scales[index] & 63, scales[index + 4] & 63)
    } else {
        (
            (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4),
            (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
        )
    }
}

/// Decodes one packed block into 256 f32 values. `block` must be exactly the
/// codec's block size.
pub(crate) fn decode_block(
    codec: GgmlKQuant,
    block: &[u8],
    output: &mut [f32; BLOCK_VALUES],
) -> Result<(), String> {
    let expected = block_bytes(codec);
    if block.len() != expected {
        return Err(format!(
            "{} block has {} bytes, expected {expected}",
            codec.name(),
            block.len()
        ));
    }
    match codec {
        GgmlKQuant::Q2K => decode_q2_k(block, output),
        GgmlKQuant::Q3K => decode_q3_k(block, output),
        GgmlKQuant::Q4K => decode_q4_k(block, output),
        GgmlKQuant::Q5K => decode_q5_k(block, output),
        GgmlKQuant::Q6K => decode_q6_k(block, output),
    }
    Ok(())
}

// These offsets and bit traversals mirror block_q*_K and dequantize_row_q*_K
// in current ggml. K-quant blocks are little-endian GGUF payloads.
/// Q2_K: 16 groups of 16, two-bit codes, 4-bit scale/min per group, f16
/// global `d`/`dmin`.
fn decode_q2_k(block: &[u8], output: &mut [f32; BLOCK_VALUES]) {
    let scales = &block[..16];
    let quants = &block[16..80];
    let d = fp16_at(block, 80);
    let dmin = fp16_at(block, 82);
    let mut group = 0;
    for half in 0..2 {
        let q = &quants[half * 32..half * 32 + 32];
        for shift in [0, 2, 4, 6] {
            for q_offset in [0, 16] {
                let scale = scales[group];
                let dl = d * f32::from(scale & 0x0f);
                let ml = dmin * f32::from(scale >> 4);
                let out = &mut output[group * 16..group * 16 + 16];
                for (value, &quant) in out.iter_mut().zip(&q[q_offset..q_offset + 16]) {
                    *value = dl * f32::from((quant >> shift) & 3) - ml;
                }
                group += 1;
            }
        }
    }
}

/// Q3_K: 16 groups of 16, two-bit codes plus a one-bit high mask, 6-bit
/// signed group scales biased by 32, f16 global scale.
fn decode_q3_k(block: &[u8], output: &mut [f32; BLOCK_VALUES]) {
    let hmask = &block[..32];
    let quants = &block[32..96];
    let packed_scales = &block[96..108];
    let d = fp16_at(block, 108);
    let mut group = 0;
    for half in 0..2 {
        let q = &quants[half * 32..half * 32 + 32];
        for lane in 0..4 {
            let shift = lane * 2;
            let mask = 1u8 << (half * 4 + lane);
            for q_offset in [0, 16] {
                let low = if group < 8 {
                    packed_scales[group] & 0x0f
                } else {
                    packed_scales[group - 8] >> 4
                };
                let high = (packed_scales[8 + group % 4] >> (2 * (group / 4))) & 3;
                let scale = i16::from(low | (high << 4)) - 32;
                let dl = d * f32::from(scale);
                let out = &mut output[group * 16..group * 16 + 16];
                for index in 0..16 {
                    let low_quant = i16::from((q[q_offset + index] >> shift) & 3);
                    let high_quant = if hmask[q_offset + index] & mask == 0 {
                        4
                    } else {
                        0
                    };
                    out[index] = dl * f32::from(low_quant - high_quant);
                }
                group += 1;
            }
        }
    }
}

/// Q4_K: 8 groups of 32, four-bit codes, packed 6-bit scale/min pairs
/// ([`scale_min_k4`]), f16 global `d`/`dmin`.
fn decode_q4_k(block: &[u8], output: &mut [f32; BLOCK_VALUES]) {
    let d = fp16_at(block, 0);
    let dmin = fp16_at(block, 2);
    let scales = &block[4..16];
    let quants = &block[16..144];
    for pair in 0..4 {
        let q = &quants[pair * 32..pair * 32 + 32];
        for side in 0..2 {
            let group = pair * 2 + side;
            let (scale, min) = scale_min_k4(group, scales);
            let dl = d * f32::from(scale);
            let ml = dmin * f32::from(min);
            let out = &mut output[group * 32..group * 32 + 32];
            for (value, &quant) in out.iter_mut().zip(q) {
                let quant = if side == 0 { quant & 0x0f } else { quant >> 4 };
                *value = dl * f32::from(quant) - ml;
            }
        }
    }
}

/// Q5_K: like Q4_K plus a 32-byte high-bit plane extending codes to five
/// bits.
fn decode_q5_k(block: &[u8], output: &mut [f32; BLOCK_VALUES]) {
    let d = fp16_at(block, 0);
    let dmin = fp16_at(block, 2);
    let scales = &block[4..16];
    let high = &block[16..48];
    let low = &block[48..176];
    for pair in 0..4 {
        let ql = &low[pair * 32..pair * 32 + 32];
        let masks = [1u8 << (pair * 2), 1u8 << (pair * 2 + 1)];
        for side in 0..2 {
            let group = pair * 2 + side;
            let (scale, min) = scale_min_k4(group, scales);
            let dl = d * f32::from(scale);
            let ml = dmin * f32::from(min);
            let out = &mut output[group * 32..group * 32 + 32];
            for index in 0..32 {
                let low_quant = if side == 0 {
                    ql[index] & 0x0f
                } else {
                    ql[index] >> 4
                };
                let quant = low_quant
                    + if high[index] & masks[side] == 0 {
                        0
                    } else {
                        16
                    };
                out[index] = dl * f32::from(quant) - ml;
            }
        }
    }
}

/// Q6_K: 16 groups of 16, six-bit codes (4 low bits + 2 high bits) biased by
/// −32, signed 8-bit group scales, f16 global scale.
fn decode_q6_k(block: &[u8], output: &mut [f32; BLOCK_VALUES]) {
    let low = &block[..128];
    let high = &block[128..192];
    let d = fp16_at(block, 208);
    for half in 0..2 {
        let ql = &low[half * 64..half * 64 + 64];
        let qh = &high[half * 32..half * 32 + 32];
        for index in 0..32 {
            let scale_lane = index / 16;
            let quants = [
                (ql[index] & 0x0f) | (((qh[index] >> 0) & 3) << 4),
                (ql[index + 32] & 0x0f) | (((qh[index] >> 2) & 3) << 4),
                (ql[index] >> 4) | (((qh[index] >> 4) & 3) << 4),
                (ql[index + 32] >> 4) | (((qh[index] >> 6) & 3) << 4),
            ];
            for (quarter, &quant) in quants.iter().enumerate() {
                let scale_index = half * 8 + scale_lane + quarter * 2;
                let scale = block[192 + scale_index] as i8;
                let quant = i16::from(quant) - 32;
                output[half * 128 + quarter * 32 + index] = d * f32::from(scale) * f32::from(quant);
            }
        }
    }
}

/// Validates that `weight` is an exact contiguous packed `[rows,
/// encoded_row_bytes]` u8 tensor for `codec`, returning the encoded row
/// width.
fn validate_weight(
    operation: &str,
    weight: &Tensor,
    codec: GgmlKQuant,
    rows: usize,
    columns: usize,
) -> Result<usize, String> {
    let encoded_row_bytes = codec.encoded_row_bytes(columns).ok_or_else(|| {
        format!(
            "{operation}: logical row width {columns} is invalid for {}",
            codec.name()
        )
    })?;
    rows.checked_mul(encoded_row_bytes)
        .ok_or_else(|| format!("{operation}: packed weight byte size overflow"))?;
    if weight.dtype() != DType::U8
        || weight.shape() != [rows, encoded_row_bytes]
        || !weight.layout.is_contiguous()
        || weight.layout.offset() != 0
    {
        return Err(format!(
            "{operation}: expected exact contiguous packed {} weight [{rows}, {encoded_row_bytes}] u8",
            codec.name()
        ));
    }
    Ok(encoded_row_bytes)
}

/// Plans `quantized_linear`. Validates the f32 input of rank ≥ 2, the packed
/// weight, and the optional `[rows]` f32 bias. Computes the f32 output
/// requirement with `rows` as the trailing dimension.
pub(crate) fn linear_requirements(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    codec: GgmlKQuant,
    weight_shape: [usize; 2],
) -> Result<QuantizedRequirements, String> {
    let [rows, columns] = weight_shape;
    if input.dtype() != DType::F32 || input.shape().len() < 2 {
        return Err("quantized_linear: input must be rank >= 2 f32".to_string());
    }
    if input.shape().last() != Some(&columns) {
        return Err(format!(
            "quantized_linear: expected input trailing width {columns}, got {:?}",
            input.shape()
        ));
    }
    checked_product(input.shape(), "quantized_linear")?;
    let encoded_row_bytes = validate_weight("quantized_linear", weight, codec, rows, columns)?;
    if let Some(bias) = bias {
        if bias.dtype() != DType::F32 || bias.shape() != [rows] {
            return Err(format!(
                "quantized_linear: bias must be [{rows}] f32, got {:?} {}",
                bias.shape(),
                bias.dtype()
            ));
        }
        checked_product(bias.shape(), "quantized_linear")?;
    }
    let mut output_shape = input.shape().to_vec();
    *output_shape.last_mut().expect("input rank was validated") = rows;
    let output = checked_f32_output(&output_shape, "quantized_linear")?;
    Ok(QuantizedRequirements {
        output,
        rows,
        columns,
        encoded_row_bytes,
        has_bias: bias.is_some(),
    })
}

/// Plans `quantized_embedding`. Validates u32 or i64 indexes and the packed
/// weight, then computes the `[indexes..., columns]` f32 output requirement.
pub(crate) fn embedding_requirements(
    indexes: &Tensor,
    weight: &Tensor,
    codec: GgmlKQuant,
    weight_shape: [usize; 2],
) -> Result<QuantizedRequirements, String> {
    let [rows, columns] = weight_shape;
    if !matches!(indexes.dtype(), DType::U32 | DType::I64) {
        return Err(format!(
            "quantized_embedding: indexes must be u32 or i64, got {}",
            indexes.dtype()
        ));
    }
    checked_product(indexes.shape(), "quantized_embedding")?;
    let encoded_row_bytes = validate_weight("quantized_embedding", weight, codec, rows, columns)?;
    let mut output_shape = indexes.shape().to_vec();
    output_shape.push(columns);
    let output = checked_f32_output(&output_shape, "quantized_embedding")?;
    Ok(QuantizedRequirements {
        output,
        rows,
        columns,
        encoded_row_bytes,
        has_bias: false,
    })
}

fn validate_linear_execution(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    destination: &CpuDestination<'_>,
    codec: GgmlKQuant,
    requirements: &QuantizedRequirements,
) -> Result<usize, String> {
    if input.dtype() != DType::F32
        || input.shape().len() < 2
        || input.shape().last() != Some(&requirements.columns)
    {
        return Err("quantized_linear: input does not match the immutable plan".to_string());
    }
    let vectors = checked_product(
        &input.shape()[..input.shape().len() - 1],
        "quantized_linear",
    )?;
    let output_elements = vectors
        .checked_mul(requirements.rows)
        .ok_or_else(|| "quantized_linear: output element count overflow".to_string())?;
    output_elements
        .checked_mul(DType::F32.size_in_bytes())
        .ok_or_else(|| "quantized_linear: output byte size overflow".to_string())?;
    validate_weight(
        "quantized_linear",
        weight,
        codec,
        requirements.rows,
        requirements.columns,
    )?;
    if bias.is_some() != requirements.has_bias
        || bias
            .is_some_and(|bias| bias.dtype() != DType::F32 || bias.shape() != [requirements.rows])
    {
        return Err("quantized_linear: bias does not match the immutable plan".to_string());
    }
    let shape = destination.shape();
    if destination.dtype() != DType::F32
        || shape.len() != input.shape().len()
        || shape[..shape.len() - 1] != input.shape()[..input.shape().len() - 1]
        || shape.last() != Some(&requirements.rows)
    {
        return Err(
            "quantized_linear: destination does not match the logical node shape".to_string(),
        );
    }
    Ok(vectors)
}

/// Executes a planned quantized linear without allocating. Decodes blocks as
/// needed and checks `cancelled` for each vector and block.
pub(crate) fn linear_into(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    codec: GgmlKQuant,
    destination: &mut CpuDestination<'_>,
    requirements: &QuantizedRequirements,
    cancelled: &CancellationFlag,
) -> Result<(), String> {
    let vectors = validate_linear_execution(input, weight, bias, destination, codec, requirements)?;
    let CpuBuffer::F32(input_storage) = &input.buffer else {
        unreachable!("input dtype was validated")
    };
    let CpuBuffer::U8(weight_storage) = &weight.buffer else {
        unreachable!("weight dtype was validated")
    };
    let bias_storage = bias.map(|bias| {
        let CpuBuffer::F32(storage) = &bias.buffer else {
            unreachable!("bias dtype was validated")
        };
        (storage.as_slice(), &bias.layout)
    });
    let input_values = input_storage.as_slice();
    let packed = weight_storage.as_slice();
    let block_bytes = block_bytes(codec);
    let blocks = requirements.columns / BLOCK_VALUES;
    let columns = requirements.columns;
    let rows = requirements.rows;
    destination.write_current::<f32, _>("quantized_linear", |output| -> Result<(), String> {
        let mut decoded = [0.0f32; BLOCK_VALUES];
        for vector in 0..vectors {
            for row in 0..rows {
                if cancelled.is_cancelled() {
                    return Err("operation aborted".to_string());
                }
                let row_start = row
                    .checked_mul(requirements.encoded_row_bytes)
                    .ok_or_else(|| "quantized_linear: packed row offset overflow".to_string())?;
                let mut sum = 0.0f32;
                for block_index in 0..blocks {
                    if cancelled.is_cancelled() {
                        return Err("operation aborted".to_string());
                    }
                    let encoded_start =
                        row_start
                            .checked_add(block_index.checked_mul(block_bytes).ok_or_else(|| {
                                "quantized_linear: block offset overflow".to_string()
                            })?)
                            .ok_or_else(|| "quantized_linear: block offset overflow".to_string())?;
                    let encoded_end = encoded_start
                        .checked_add(block_bytes)
                        .ok_or_else(|| "quantized_linear: block end overflow".to_string())?;
                    decode_block(codec, &packed[encoded_start..encoded_end], &mut decoded)?;
                    let block_start = block_index
                        .checked_mul(BLOCK_VALUES)
                        .ok_or_else(|| "quantized_linear: input offset overflow".to_string())?;
                    let input_start = vector
                        .checked_mul(columns)
                        .and_then(|offset| offset.checked_add(block_start))
                        .ok_or_else(|| "quantized_linear: input offset overflow".to_string())?;
                    for (lane, &weight) in decoded.iter().enumerate() {
                        let logical = input_start
                            .checked_add(lane)
                            .ok_or_else(|| "quantized_linear: input offset overflow".to_string())?;
                        let input_index = crate::tensor::source_index(&input.layout, logical);
                        sum += input_values[input_index] * weight;
                    }
                }
                if let Some((values, layout)) = bias_storage {
                    sum += values[crate::tensor::source_index(layout, row)];
                }
                let output_index = vector
                    .checked_mul(rows)
                    .and_then(|offset| offset.checked_add(row))
                    .ok_or_else(|| "quantized_linear: output offset overflow".to_string())?;
                output[output_index] = sum;
            }
        }
        Ok(())
    })?
}

fn validate_embedding_execution(
    indexes: &Tensor,
    weight: &Tensor,
    destination: &CpuDestination<'_>,
    codec: GgmlKQuant,
    requirements: &QuantizedRequirements,
) -> Result<usize, String> {
    if !matches!(indexes.dtype(), DType::U32 | DType::I64) {
        return Err("quantized_embedding: indexes do not match the immutable plan".to_string());
    }
    let count = checked_product(indexes.shape(), "quantized_embedding")?;
    let output_elements = count
        .checked_mul(requirements.columns)
        .ok_or_else(|| "quantized_embedding: output element count overflow".to_string())?;
    output_elements
        .checked_mul(DType::F32.size_in_bytes())
        .ok_or_else(|| "quantized_embedding: output byte size overflow".to_string())?;
    validate_weight(
        "quantized_embedding",
        weight,
        codec,
        requirements.rows,
        requirements.columns,
    )?;
    let shape = destination.shape();
    let output_rank = indexes
        .shape()
        .len()
        .checked_add(1)
        .ok_or_else(|| "quantized_embedding: output rank overflow".to_string())?;
    if destination.dtype() != DType::F32
        || shape.len() != output_rank
        || shape[..indexes.shape().len()] != *indexes.shape()
        || shape.last() != Some(&requirements.columns)
    {
        return Err(
            "quantized_embedding: destination does not match the logical node shape".to_string(),
        );
    }
    Ok(count)
}

/// Executes a planned quantized embedding without allocating and decodes only
/// selected rows. Checks index bounds and rejects negative i64 indexes.
pub(crate) fn embedding_into(
    indexes: &Tensor,
    weight: &Tensor,
    codec: GgmlKQuant,
    destination: &mut CpuDestination<'_>,
    requirements: &QuantizedRequirements,
    cancelled: &CancellationFlag,
) -> Result<(), String> {
    let count = validate_embedding_execution(indexes, weight, destination, codec, requirements)?;
    let CpuBuffer::U8(weight_storage) = &weight.buffer else {
        unreachable!("weight dtype was validated")
    };
    let packed = weight_storage.as_slice();
    let block_bytes = block_bytes(codec);
    let blocks = requirements.columns / BLOCK_VALUES;
    destination.write_current::<f32, _>("quantized_embedding", |output| -> Result<(), String> {
        let mut decoded = [0.0f32; BLOCK_VALUES];
        for logical_index in 0..count {
            if cancelled.is_cancelled() {
                return Err("operation aborted".to_string());
            }
            let source_index = crate::tensor::source_index(&indexes.layout, logical_index);
            let row = match &indexes.buffer {
                CpuBuffer::U32(values) => values[source_index] as usize,
                CpuBuffer::I64(values) => usize::try_from(values[source_index]).map_err(|_| {
                    format!(
                        "quantized_embedding: index {} is outside 0..{}",
                        values[source_index], requirements.rows
                    )
                })?,
                _ => unreachable!("index dtype was validated"),
            };
            if row >= requirements.rows {
                return Err(format!(
                    "quantized_embedding: index {row} is outside 0..{}",
                    requirements.rows
                ));
            }
            let row_start = row
                .checked_mul(requirements.encoded_row_bytes)
                .ok_or_else(|| "quantized_embedding: packed row offset overflow".to_string())?;
            let output_start = logical_index
                .checked_mul(requirements.columns)
                .ok_or_else(|| "quantized_embedding: output offset overflow".to_string())?;
            for block_index in 0..blocks {
                if cancelled.is_cancelled() {
                    return Err("operation aborted".to_string());
                }
                let encoded_start =
                    row_start
                        .checked_add(block_index.checked_mul(block_bytes).ok_or_else(|| {
                            "quantized_embedding: block offset overflow".to_string()
                        })?)
                        .ok_or_else(|| "quantized_embedding: block offset overflow".to_string())?;
                let encoded_end = encoded_start
                    .checked_add(block_bytes)
                    .ok_or_else(|| "quantized_embedding: block end overflow".to_string())?;
                decode_block(codec, &packed[encoded_start..encoded_end], &mut decoded)?;
                let decoded_start = block_index
                    .checked_mul(BLOCK_VALUES)
                    .and_then(|offset| output_start.checked_add(offset))
                    .ok_or_else(|| "quantized_embedding: output offset overflow".to_string())?;
                let decoded_end = decoded_start
                    .checked_add(BLOCK_VALUES)
                    .ok_or_else(|| "quantized_embedding: output end overflow".to_string())?;
                output[decoded_start..decoded_end].copy_from_slice(&decoded);
            }
        }
        Ok(())
    })?
}

#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;

    pub(crate) struct Fixture {
        pub(crate) codec: GgmlKQuant,
        pub(crate) bytes: &'static [u8],
        pub(crate) group_width: usize,
        pub(crate) expected_groups: &'static [f32],
    }

    const Q2_BYTES: [u8; 84] = {
        let mut bytes = [0u8; 84];
        let mut index = 0;
        while index < 16 {
            bytes[index] = 0x10 | ((index % 15 + 1) as u8);
            index += 1;
        }
        index = 16;
        while index < 80 {
            bytes[index] = 0xe4;
            index += 1;
        }
        bytes[80] = 0x00;
        bytes[81] = 0x3c;
        bytes[82] = 0x00;
        bytes[83] = 0x38;
        bytes
    };
    const Q2_EXPECTED: [f32; 16] = [
        -0.5, -0.5, 2.5, 3.5, 9.5, 11.5, 20.5, 23.5, -0.5, -0.5, 10.5, 11.5, 25.5, 27.5, 44.5, 2.5,
    ];

    const Q3_BYTES: [u8; 110] = {
        let mut bytes = [0u8; 110];
        let mut index = 0;
        while index < 32 {
            bytes[index] = 0xaa;
            index += 1;
        }
        while index < 96 {
            bytes[index] = 0xe4;
            index += 1;
        }
        index = 96;
        while index < 104 {
            bytes[index] = 0x11;
            index += 1;
        }
        while index < 108 {
            bytes[index] = 0xaa;
            index += 1;
        }
        bytes[108] = 0x00;
        bytes[109] = 0x3c;
        bytes
    };
    const Q3_EXPECTED: [f32; 16] = [
        -4.0, -4.0, 1.0, 1.0, -2.0, -2.0, 3.0, 3.0, -4.0, -4.0, 1.0, 1.0, -2.0, -2.0, 3.0, 3.0,
    ];

    const fn scale_min_fixture<const N: usize>(
        quant_offset: usize,
        high_offset: Option<usize>,
    ) -> [u8; N] {
        let mut bytes = [0u8; N];
        bytes[0] = 0x00;
        bytes[1] = 0x3c;
        bytes[2] = 0x00;
        bytes[3] = 0x38;
        let mut index = 4;
        while index < 8 {
            bytes[index] = 1;
            index += 1;
        }
        while index < 12 {
            bytes[index] = 2;
            index += 1;
        }
        while index < 16 {
            bytes[index] = 0x21;
            index += 1;
        }
        if let Some(offset) = high_offset {
            index = offset;
            while index < offset + 32 {
                bytes[index] = 0x99;
                index += 1;
            }
        }
        index = quant_offset;
        while index < N {
            bytes[index] = 0xe4;
            index += 1;
        }
        bytes
    }

    const Q4_BYTES: [u8; 144] = scale_min_fixture(16, None);
    const Q4_EXPECTED: [f32; 8] = [3.0, 13.0, 3.0, 13.0, 3.0, 13.0, 3.0, 13.0];

    const Q5_BYTES: [u8; 176] = scale_min_fixture(48, Some(16));
    const Q5_EXPECTED: [f32; 8] = [19.0, 13.0, 3.0, 29.0, 19.0, 13.0, 3.0, 29.0];

    const Q6_BYTES: [u8; 210] = {
        let mut bytes = [0u8; 210];
        let mut index = 0;
        while index < 128 {
            bytes[index] = 0xe4;
            index += 1;
        }
        while index < 192 {
            bytes[index] = 0xe4;
            index += 1;
        }
        index = 0;
        while index < 16 {
            bytes[192 + index] = (index + 1) as u8;
            index += 1;
        }
        bytes[208] = 0x00;
        bytes[209] = 0x38;
        bytes
    };
    const Q6_EXPECTED: [f32; 16] = [
        -14.0, -28.0, -18.0, -24.0, 35.0, 42.0, 105.0, 120.0, -126.0, -140.0, -66.0, -72.0, 91.0,
        98.0, 225.0, 240.0,
    ];

    pub(crate) const CASES: [Fixture; 5] = [
        Fixture {
            codec: GgmlKQuant::Q2K,
            bytes: &Q2_BYTES,
            group_width: 16,
            expected_groups: &Q2_EXPECTED,
        },
        Fixture {
            codec: GgmlKQuant::Q3K,
            bytes: &Q3_BYTES,
            group_width: 16,
            expected_groups: &Q3_EXPECTED,
        },
        Fixture {
            codec: GgmlKQuant::Q4K,
            bytes: &Q4_BYTES,
            group_width: 32,
            expected_groups: &Q4_EXPECTED,
        },
        Fixture {
            codec: GgmlKQuant::Q5K,
            bytes: &Q5_BYTES,
            group_width: 32,
            expected_groups: &Q5_EXPECTED,
        },
        Fixture {
            codec: GgmlKQuant::Q6K,
            bytes: &Q6_BYTES,
            group_width: 16,
            expected_groups: &Q6_EXPECTED,
        },
    ];

    pub(crate) fn fixture(codec: GgmlKQuant) -> &'static Fixture {
        CASES.iter().find(|case| case.codec == codec).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::{fixture, CASES};
    use super::*;
    use effect_torch_runtime::Layout;

    fn assert_fixture_values(actual: &[f32], codec: GgmlKQuant) {
        let fixture = fixture(codec);
        assert_eq!(actual.len(), BLOCK_VALUES);
        for (group, &expected) in fixture.expected_groups.iter().enumerate() {
            assert_eq!(
                &actual[group * fixture.group_width..(group + 1) * fixture.group_width],
                vec![expected; fixture.group_width],
                "{} group {group}",
                codec.name()
            );
        }
    }

    #[test]
    fn canonical_k_quant_fixtures_decode_exactly() {
        for fixture in &CASES {
            let mut actual = [0.0; BLOCK_VALUES];
            decode_block(fixture.codec, fixture.bytes, &mut actual).unwrap();
            assert_fixture_values(&actual, fixture.codec);
        }
    }

    #[test]
    fn decoder_rejects_noncanonical_block_lengths() {
        for fixture in &CASES {
            let mut actual = [0.0; BLOCK_VALUES];
            let error = decode_block(
                fixture.codec,
                &fixture.bytes[..fixture.bytes.len() - 1],
                &mut actual,
            )
            .unwrap_err();
            assert!(error.contains("expected"));
        }
    }

    #[test]
    fn linear_decodes_blocks_during_prefill() {
        let fixture = fixture(GgmlKQuant::Q4K);
        let input = Tensor::from_vec(
            (0..1024)
                .map(|index| {
                    if index < 512 || index < 576 {
                        1.0f32
                    } else {
                        0.0
                    }
                })
                .collect(),
            vec![2, 512],
        );
        let mut packed = Vec::with_capacity(fixture.bytes.len() * 4);
        for _ in 0..4 {
            packed.extend_from_slice(fixture.bytes);
        }
        let weight = Tensor::from_vec(packed, vec![2, fixture.bytes.len() * 2]);
        let bias = Tensor::from_vec(vec![0.5f32, -0.5], vec![2]);
        let requirements =
            linear_requirements(&input, &weight, Some(&bias), fixture.codec, [2, 512]).unwrap();
        let mut output = Tensor::empty(&[2, 2], DType::F32);
        linear_into(
            &input,
            &weight,
            Some(&bias),
            fixture.codec,
            &mut CpuDestination::new(&mut output).unwrap(),
            &requirements,
            &CancellationFlag::new(),
        )
        .unwrap();
        let CpuBuffer::F32(values) = output.buffer else {
            panic!()
        };
        assert_eq!(values.as_slice(), &[4096.5, 4095.5, 512.5, 511.5]);
    }

    #[test]
    fn embedding_decodes_only_selected_rows_for_both_index_types() {
        for i64_indexes in [false, true] {
            let fixture = fixture(GgmlKQuant::Q5K);
            let mut packed = vec![0; fixture.bytes.len() * 3];
            packed[fixture.bytes.len()..fixture.bytes.len() * 2].copy_from_slice(fixture.bytes);
            let weight = Tensor::from_vec(packed, vec![3, fixture.bytes.len()]);
            let indexes = if i64_indexes {
                Tensor::from_vec(vec![1i64, 1], vec![2])
            } else {
                Tensor::from_vec(vec![1u32, 1], vec![2])
            };
            let requirements =
                embedding_requirements(&indexes, &weight, fixture.codec, [3, 256]).unwrap();
            let mut output = Tensor::empty(&[2, 256], DType::F32);
            embedding_into(
                &indexes,
                &weight,
                fixture.codec,
                &mut CpuDestination::new(&mut output).unwrap(),
                &requirements,
                &CancellationFlag::new(),
            )
            .unwrap();
            let CpuBuffer::F32(values) = output.buffer else {
                panic!()
            };
            assert_fixture_values(&values.as_slice()[..256], fixture.codec);
            assert_fixture_values(&values.as_slice()[256..], fixture.codec);
        }
    }

    #[test]
    fn embedding_rejects_negative_and_out_of_range_indexes() {
        let fixture = fixture(GgmlKQuant::Q2K);
        let weight = Tensor::from_vec(fixture.bytes.to_vec(), vec![1, fixture.bytes.len()]);
        for indexes in [
            Tensor::from_vec(vec![-1i64], vec![1]),
            Tensor::from_vec(vec![1i64], vec![1]),
        ] {
            let requirements =
                embedding_requirements(&indexes, &weight, fixture.codec, [1, 256]).unwrap();
            let mut output = Tensor::empty(&[1, 256], DType::F32);
            let error = embedding_into(
                &indexes,
                &weight,
                fixture.codec,
                &mut CpuDestination::new(&mut output).unwrap(),
                &requirements,
                &CancellationFlag::new(),
            )
            .unwrap_err();
            assert!(error.contains("outside 0..1"));
        }
    }

    #[test]
    fn malformed_packed_rows_and_layouts_are_rejected() {
        let fixture = fixture(GgmlKQuant::Q4K);
        let input = Tensor::from_vec(vec![0.0f32; 256], vec![1, 256]);
        let wrong_rank = Tensor::from_vec(fixture.bytes.to_vec(), vec![fixture.bytes.len()]);
        assert!(
            linear_requirements(&input, &wrong_rank, None, fixture.codec, [1, 256])
                .unwrap_err()
                .contains("exact contiguous")
        );

        let wrong_width = Tensor::from_vec(
            fixture.bytes[..fixture.bytes.len() - 1].to_vec(),
            vec![1, fixture.bytes.len() - 1],
        );
        assert!(
            linear_requirements(&input, &wrong_width, None, fixture.codec, [1, 256])
                .unwrap_err()
                .contains("exact contiguous")
        );

        let storage = Tensor::from_vec(
            vec![0u8; fixture.bytes.len() + 1],
            vec![fixture.bytes.len() + 1],
        );
        let offset = Tensor::new(
            storage.buffer,
            Layout::new(
                vec![1, fixture.bytes.len()],
                vec![fixture.bytes.len(), 1],
                1,
            ),
        );
        assert!(
            linear_requirements(&input, &offset, None, fixture.codec, [1, 256])
                .unwrap_err()
                .contains("exact contiguous")
        );

        let invalid_width_input = Tensor::from_vec(vec![0.0f32; 255], vec![1, 255]);
        let empty_weight = Tensor::from_vec(Vec::<u8>::new(), vec![1, 0]);
        assert!(linear_requirements(
            &invalid_width_input,
            &empty_weight,
            None,
            fixture.codec,
            [1, 255],
        )
        .unwrap_err()
        .contains("logical row width"));
        assert!(
            linear_requirements(&input, &wrong_rank, None, fixture.codec, [usize::MAX, 256],)
                .unwrap_err()
                .contains("byte size overflow")
        );
    }

    #[test]
    fn quantized_execution_observes_cancellation() {
        let fixture = fixture(GgmlKQuant::Q6K);
        let indexes = Tensor::from_vec(vec![0u32], vec![1]);
        let weight = Tensor::from_vec(fixture.bytes.to_vec(), vec![1, fixture.bytes.len()]);
        let requirements =
            embedding_requirements(&indexes, &weight, fixture.codec, [1, 256]).unwrap();
        let mut output = Tensor::empty(&[1, 256], DType::F32);
        let cancelled = CancellationFlag::new();
        cancelled.cancel();
        let error = embedding_into(
            &indexes,
            &weight,
            fixture.codec,
            &mut CpuDestination::new(&mut output).unwrap(),
            &requirements,
            &cancelled,
        )
        .unwrap_err();
        assert_eq!(error, "operation aborted");
    }
}

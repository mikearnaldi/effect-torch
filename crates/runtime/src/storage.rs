//! Closed logical-value storage contracts and canonical GGML K-quant interpretation.
//!
//! A representation identifies values, while a layout describes their byte arrangement.
//! Dense values retain the exact scalar dtype, including two-byte F16/BF16 bits.
//! Packed formats cannot be inferred from a byte buffer or an operation's attributes.

use crate::{DType, Layout};

/// Logical values in every supported GGML K-quant block.
pub const GGML_K_BLOCK_VALUES: usize = 256;
const BLOCK_VALUES: usize = GGML_K_BLOCK_VALUES;

/// Exact, source-defined packed formats. Unknown formats are never opaque values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PackedFormat {
    GgmlKQuant(GgmlKQuant),
}

/// Storage representation of a logical tensor. Dense uses its semantic dtype.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum StorageRepresentation {
    #[default]
    Dense,
    Packed(PackedFormat),
}

/// Source-defined private layout ABIs. No private binding layout is accepted yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendLayoutAbi {}

/// Owned caller-visible layout constraints. Dense strides and offsets use elements.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum StorageLayout {
    Unconstrained,
    Canonical,
    DenseStrided(Layout),
    BackendPrivate(BackendLayoutAbi),
}

/// Borrowed layout constraints, with exact dense element strides and offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutConstraintSpec<'a> {
    Unconstrained,
    Canonical,
    DenseStrided(&'a Layout),
    BackendPrivate(BackendLayoutAbi),
}

/// Owned representation and layout metadata. Logical shape/dtype belong to the value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StorageMetadata {
    pub representation: StorageRepresentation,
    pub layout: StorageLayout,
}

impl Default for StorageMetadata {
    fn default() -> Self {
        Self::dense()
    }
}

impl StorageMetadata {
    /// A materialized, zero-offset row-major dense value.
    pub const fn dense() -> Self {
        Self {
            representation: StorageRepresentation::Dense,
            layout: StorageLayout::Canonical,
        }
    }

    /// An internal dense graph result whose physical layout lowering may choose.
    pub const fn unconstrained() -> Self {
        Self {
            representation: StorageRepresentation::Dense,
            layout: StorageLayout::Unconstrained,
        }
    }

    /// Canonical packed GGML bytes. The value still has a logical F32 shape.
    pub const fn packed(codec: GgmlKQuant) -> Self {
        Self {
            representation: StorageRepresentation::Packed(PackedFormat::GgmlKQuant(codec)),
            layout: StorageLayout::Canonical,
        }
    }

    pub fn as_spec(&self) -> StorageSpec<'_> {
        StorageSpec {
            representation: self.representation,
            layout_constraint: match &self.layout {
                StorageLayout::Unconstrained => LayoutConstraintSpec::Unconstrained,
                StorageLayout::Canonical => LayoutConstraintSpec::Canonical,
                StorageLayout::DenseStrided(layout) => LayoutConstraintSpec::DenseStrided(layout),
                StorageLayout::BackendPrivate(abi) => LayoutConstraintSpec::BackendPrivate(*abi),
            },
        }
    }
}

/// Borrowed representation and layout, without duplicate logical metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StorageSpec<'a> {
    pub representation: StorageRepresentation,
    pub layout_constraint: LayoutConstraintSpec<'a>,
}

/// Complete logical value contract used by graphs, bindings, and target queries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValueSpec<'a> {
    pub semantic_dtype: DType,
    pub logical_shape: &'a [usize],
    pub storage: StorageSpec<'a>,
}

/// Derived physical geometry of a canonical materialization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageGeometry {
    pub physical_shape: Vec<usize>,
    pub physical_dtype: DType,
    pub byte_len: usize,
}

fn checked_product(shape: &[usize]) -> Result<usize, String> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape
        .iter()
        .try_fold(1usize, |total, &dim| total.checked_mul(dim))
        .ok_or_else(|| "storage: element count overflows usize".to_string())
}

impl<'a> ValueSpec<'a> {
    pub const fn dense(semantic_dtype: DType, logical_shape: &'a [usize]) -> Self {
        Self {
            semantic_dtype,
            logical_shape,
            storage: StorageSpec {
                representation: StorageRepresentation::Dense,
                layout_constraint: LayoutConstraintSpec::Canonical,
            },
        }
    }

    /// Validates representation, logical dtype, geometry, and layout compatibility.
    pub fn validate(self) -> Result<(), String> {
        self.canonical_geometry()?;
        match (self.storage.representation, self.storage.layout_constraint) {
            (_, LayoutConstraintSpec::BackendPrivate(abi)) => match abi {},
            (StorageRepresentation::Packed(_), LayoutConstraintSpec::Canonical) => Ok(()),
            (StorageRepresentation::Packed(_), _) => {
                Err("storage: packed values require canonical layout".to_string())
            }
            (StorageRepresentation::Dense, LayoutConstraintSpec::DenseStrided(layout)) => {
                if layout.shape() != self.logical_shape {
                    return Err(
                        "storage: dense layout shape differs from logical shape".to_string()
                    );
                }
                layout
                    .checked_byte_size(self.semantic_dtype)
                    .ok_or_else(|| "storage: dense layout byte extent overflows".to_string())?;
                Ok(())
            }
            (StorageRepresentation::Dense, _) => Ok(()),
        }
    }

    /// Computes canonical byte geometry with overflow checks. No allocation of payloads.
    pub fn canonical_geometry(self) -> Result<StorageGeometry, String> {
        let logical_elements = checked_product(self.logical_shape)?;
        match self.storage.representation {
            StorageRepresentation::Dense => {
                // A zero dimension makes numel zero but does not necessarily make
                // the inner products used by Layout::contiguous representable.
                self.logical_shape
                    .iter()
                    .skip(1)
                    .rev()
                    .try_fold(1usize, |stride, &dimension| stride.checked_mul(dimension))
                    .ok_or_else(|| "storage: canonical dense strides overflow usize".to_string())?;
                let byte_len = logical_elements
                    .checked_mul(self.semantic_dtype.size_in_bytes())
                    .ok_or_else(|| "storage: dense byte length overflows usize".to_string())?;
                Ok(StorageGeometry {
                    physical_shape: self.logical_shape.to_vec(),
                    physical_dtype: self.semantic_dtype,
                    byte_len,
                })
            }
            StorageRepresentation::Packed(PackedFormat::GgmlKQuant(codec)) => {
                if self.logical_shape.contains(&0) {
                    return Err("storage: packed logical dimensions must be positive".to_string());
                }
                if self.semantic_dtype != DType::F32 {
                    return Err(format!(
                        "storage: {} represents f32 values, got {}",
                        codec.name(),
                        self.semantic_dtype
                    ));
                }
                let (&columns, leading) = self.logical_shape.split_last().ok_or_else(|| {
                    "storage: packed values require a non-scalar logical shape".to_string()
                })?;
                let rows = checked_product(leading)?;
                let row_bytes = codec.encoded_row_bytes(columns)
                    .ok_or_else(|| format!("storage: {} row width {columns} is not a complete 256-value block or overflows", codec.name()))?;
                let byte_len = rows
                    .checked_mul(row_bytes)
                    .ok_or_else(|| "storage: packed byte length overflows usize".to_string())?;
                Ok(StorageGeometry {
                    physical_shape: vec![rows, row_bytes],
                    physical_dtype: DType::U8,
                    byte_len,
                })
            }
        }
    }

    /// Validates observed physical dtype, layout, and allocation extent before publication.
    /// Dense allocations may contain unused trailing bytes. Element-unit strides and
    /// offsets preserve scalar alignment relative to the base address; the backend
    /// must guarantee that address is aligned for the physical scalar type.
    pub fn validate_buffer(
        self,
        physical_dtype: DType,
        layout: &Layout,
        byte_len: usize,
    ) -> Result<(), String> {
        self.validate()?;
        let geometry = self.canonical_geometry()?;
        if physical_dtype != geometry.physical_dtype || layout.shape() != geometry.physical_shape {
            return Err(
                "storage: physical dtype or shape does not match representation".to_string(),
            );
        }
        match self.storage.layout_constraint {
            LayoutConstraintSpec::Canonical if !layout.is_contiguous() || layout.offset() != 0 => {
                return Err(
                    "storage: canonical storage must be contiguous at offset zero".to_string(),
                )
            }
            LayoutConstraintSpec::DenseStrided(expected) if layout != expected => {
                return Err(
                    "storage: dense strides or offset differ from declared layout".to_string(),
                )
            }
            _ => {}
        }
        let required = layout
            .checked_byte_size(physical_dtype)
            .ok_or_else(|| "storage: physical byte extent overflows".to_string())?;
        if byte_len < required
            || (matches!(
                self.storage.representation,
                StorageRepresentation::Packed(_)
            ) && byte_len != geometry.byte_len)
        {
            return Err(format!(
                "storage: expected {required} bytes, received {byte_len}"
            ));
        }
        Ok(())
    }

    /// Validated matrix operand metadata for quantized linear and embedding.
    pub fn packed_matrix(self) -> Result<(GgmlKQuant, [usize; 2]), String> {
        self.validate()?;
        let StorageRepresentation::Packed(PackedFormat::GgmlKQuant(codec)) =
            self.storage.representation
        else {
            return Err("expected a packed GGML K-quant weight".to_string());
        };
        let [rows, columns] = self.logical_shape else {
            return Err("packed weight must have logical shape [rows, columns]".to_string());
        };
        Ok((codec, [*rows, *columns]))
    }
}

impl PackedFormat {
    pub fn from_name(name: &str) -> Result<Self, String> {
        GgmlKQuant::from_name(name)
            .map(Self::GgmlKQuant)
            .ok_or_else(|| format!("unsupported packed format {name:?}"))
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::GgmlKQuant(codec) => codec.name(),
        }
    }
}

/// A GGML K-quant block encoding (256 values per block).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GgmlKQuant {
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl GgmlKQuant {
    /// Every supported format, in GGML type-code order.
    pub const ALL: [Self; 5] = [Self::Q2K, Self::Q3K, Self::Q4K, Self::Q5K, Self::Q6K];

    /// Returns the canonical GGML encoding name, such as `"Q4_K"`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
        }
    }

    /// Parses a canonical GGML name, or returns `None` if it is unknown.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Q2_K" => Some(Self::Q2K),
            "Q3_K" => Some(Self::Q3K),
            "Q4_K" => Some(Self::Q4K),
            "Q5_K" => Some(Self::Q5K),
            "Q6_K" => Some(Self::Q6K),
            _ => None,
        }
    }

    /// Exact canonical byte length of one 256-value block.
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
        }
    }

    /// Returns the packed byte length of a `columns`-element logical row.
    /// Returns `None` if `columns` is zero, is not a multiple of 256, or the
    /// result overflows.
    pub fn encoded_row_bytes(self, columns: usize) -> Option<usize> {
        (columns != 0 && columns.is_multiple_of(GGML_K_BLOCK_VALUES))
            .then(|| (columns / 256).checked_mul(self.block_bytes()))
            .flatten()
    }
}

/// Reads a little-endian f16 field of a packed block as f32.
fn fp16_at(block: &[u8], offset: usize) -> f32 {
    let bits = u16::from_le_bytes([block[offset], block[offset + 1]]);
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 31;
    let mut fraction = u32::from(bits & 1023);
    let encoded = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut exponent = 113u32;
            while fraction & 1024 == 0 {
                fraction <<= 1;
                exponent -= 1;
            }
            sign | (exponent << 23) | ((fraction & 1023) << 13)
        }
        31 => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(encoded)
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
pub fn decode_ggml_k_block(
    codec: GgmlKQuant,
    block: &[u8],
    output: &mut [f32; BLOCK_VALUES],
) -> Result<(), String> {
    let expected = codec.block_bytes();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_geometry_covers_every_accepted_scalar_and_packed_format() {
        for (dtype, width) in [
            (DType::F32, 4),
            (DType::F64, 8),
            (DType::F16, 2),
            (DType::BF16, 2),
            (DType::U8, 1),
            (DType::U32, 4),
            (DType::I64, 8),
        ] {
            let geometry = ValueSpec::dense(dtype, &[2, 3])
                .canonical_geometry()
                .unwrap();
            assert_eq!(geometry.physical_dtype, dtype);
            assert_eq!(geometry.physical_shape, [2, 3]);
            assert_eq!(geometry.byte_len, 6 * width);
        }
        for (codec, block_bytes) in GgmlKQuant::ALL.into_iter().zip([84, 110, 144, 176, 210]) {
            let storage = StorageMetadata::packed(codec);
            let value = ValueSpec {
                semantic_dtype: DType::F32,
                logical_shape: &[2, 3, 512],
                storage: storage.as_spec(),
            };
            let geometry = value.canonical_geometry().unwrap();
            assert_eq!(geometry.physical_dtype, DType::U8);
            assert_eq!(geometry.physical_shape, [6, block_bytes * 2]);
            assert_eq!(geometry.byte_len, 12 * block_bytes);
            value
                .validate_buffer(
                    DType::U8,
                    &Layout::contiguous(geometry.physical_shape),
                    geometry.byte_len,
                )
                .unwrap();
            let rank_one = ValueSpec {
                logical_shape: &[256],
                ..value
            }
            .canonical_geometry()
            .unwrap();
            assert_eq!(rank_one.physical_shape, [1, block_bytes]);
        }
    }

    #[test]
    fn unknown_formats_cannot_be_inferred_from_family_names() {
        for name in [
            "", "Q2_K_XL", "Q8_0", "NVFP4", "MXFP4", "ggml", "Q6_K:3", "q4_k",
        ] {
            assert!(PackedFormat::from_name(name).is_err(), "{name}");
            assert_eq!(GgmlKQuant::from_name(name), None);
        }
        for codec in GgmlKQuant::ALL {
            assert_eq!(
                PackedFormat::from_name(codec.name()).unwrap(),
                PackedFormat::GgmlKQuant(codec)
            );
        }
    }

    #[test]
    fn rejects_wrong_dtype_partial_blocks_overflow_and_physical_layout() {
        let storage = StorageMetadata::packed(GgmlKQuant::Q4K);
        let value = ValueSpec {
            semantic_dtype: DType::F32,
            logical_shape: &[2, 256],
            storage: storage.as_spec(),
        };
        for shape in [&[][..], &[2, 255], &[usize::MAX, 256], &[2, usize::MAX]] {
            assert!(ValueSpec {
                logical_shape: shape,
                ..value
            }
            .validate()
            .is_err());
        }
        assert!(ValueSpec {
            semantic_dtype: DType::U8,
            ..value
        }
        .validate()
        .is_err());
        let physical = Layout::contiguous(vec![2, 144]);
        for bytes in [287, 289] {
            assert!(value.validate_buffer(DType::U8, &physical, bytes).is_err());
        }
        assert!(value.validate_buffer(DType::F32, &physical, 288).is_err());
        assert!(value
            .validate_buffer(DType::U8, &Layout::new(vec![2, 144], vec![144, 1], 1), 289)
            .is_err());
        assert!(value
            .validate_buffer(DType::U8, &Layout::contiguous(vec![1, 288]), 288)
            .is_err());
        assert!(ValueSpec {
            storage: StorageSpec {
                layout_constraint: LayoutConstraintSpec::Unconstrained,
                ..value.storage
            },
            ..value
        }
        .validate()
        .is_err());
    }

    #[test]
    fn dense_strides_preserve_offsets_and_validate_allocation_extent() {
        let layout = Layout::new(vec![2, 3], vec![8, 2], 1);
        let value = ValueSpec {
            semantic_dtype: DType::BF16,
            logical_shape: &[2, 3],
            storage: StorageSpec {
                representation: StorageRepresentation::Dense,
                layout_constraint: LayoutConstraintSpec::DenseStrided(&layout),
            },
        };
        value.validate_buffer(DType::BF16, &layout, 28).unwrap();
        assert!(value.validate_buffer(DType::BF16, &layout, 26).is_err());
        assert!(value
            .validate_buffer(DType::BF16, &Layout::contiguous(vec![2, 3]), 28)
            .is_err());
    }

    #[test]
    fn packed_extents_are_positive_and_logical_numel_must_fit() {
        for codec in GgmlKQuant::ALL {
            let storage = StorageMetadata::packed(codec);
            assert_eq!(codec.encoded_row_bytes(0), None);
            for shape in [&[0][..], &[0, 256], &[2, 0], &[1, 0, 256]] {
                let value = ValueSpec {
                    semantic_dtype: DType::F32,
                    logical_shape: shape,
                    storage: storage.as_spec(),
                };
                assert!(value.canonical_geometry().unwrap_err().contains("positive"));
            }
            // Compression can make the byte count fit while logical indexing overflows.
            let rows = usize::MAX / GGML_K_BLOCK_VALUES + 1;
            assert!(rows.checked_mul(codec.block_bytes()).is_some());
            assert!(rows.checked_mul(GGML_K_BLOCK_VALUES).is_none());
            let value = ValueSpec {
                semantic_dtype: DType::F32,
                logical_shape: &[rows, GGML_K_BLOCK_VALUES],
                storage: storage.as_spec(),
            };
            assert!(value
                .validate()
                .unwrap_err()
                .contains("element count overflows"));
        }
    }

    #[test]
    fn empty_dense_shapes_keep_zero_numel_without_hiding_stride_overflow() {
        for shape in [vec![0], vec![2, 0, 3], vec![usize::MAX, 2, 0]] {
            let value = ValueSpec::dense(DType::F16, &shape);
            assert_eq!(value.canonical_geometry().unwrap().byte_len, 0);
            let layout = Layout::contiguous(shape.clone());
            assert_eq!(layout.checked_numel(), Some(0));
            value.validate_buffer(DType::F16, &layout, 0).unwrap();
        }
        let impossible = ValueSpec::dense(DType::F16, &[0, usize::MAX, 2]);
        assert!(impossible
            .canonical_geometry()
            .unwrap_err()
            .contains("strides overflow"));
        assert!(ValueSpec::dense(DType::U8, &[usize::MAX, 2])
            .validate()
            .is_err());
    }

    #[test]
    fn dense_strided_validation_checks_logical_size_and_reachable_byte_extent() {
        for layout in [
            Layout::new(vec![2], vec![usize::MAX], 0),
            Layout::new(vec![1], vec![1], usize::MAX),
            Layout::new(vec![2], vec![usize::MAX / 2], 0),
            // Broadcast storage is small, but the logical element count is invalid.
            Layout::new(vec![usize::MAX, 2], vec![0, 0], 0),
        ] {
            let value = ValueSpec {
                semantic_dtype: DType::F32,
                logical_shape: layout.shape(),
                storage: StorageSpec {
                    representation: StorageRepresentation::Dense,
                    layout_constraint: LayoutConstraintSpec::DenseStrided(&layout),
                },
            };
            assert!(value.validate().is_err());
        }
        let layout = Layout::new(vec![2, 3], vec![8, 2], 1);
        let value = ValueSpec {
            semantic_dtype: DType::BF16,
            logical_shape: &[2, 3],
            storage: StorageSpec {
                representation: StorageRepresentation::Dense,
                layout_constraint: LayoutConstraintSpec::DenseStrided(&layout),
            },
        };
        // Offset 1 and the largest element index 13 require exactly 28 reachable bytes.
        value.validate_buffer(DType::BF16, &layout, 28).unwrap();
        value.validate_buffer(DType::BF16, &layout, 29).unwrap();
        assert!(value.validate_buffer(DType::BF16, &layout, 27).is_err());
        assert!(ValueSpec {
            logical_shape: &[3, 2],
            ..value
        }
        .validate()
        .is_err());
    }

    #[test]
    fn decoder_requires_exact_blocks_and_preserves_half_scale_values() {
        for codec in GgmlKQuant::ALL {
            let mut output = [1.0; GGML_K_BLOCK_VALUES];
            assert!(
                decode_ggml_k_block(codec, &vec![0; codec.block_bytes() - 1], &mut output).is_err()
            );
            assert!(
                decode_ggml_k_block(codec, &vec![0; codec.block_bytes() + 1], &mut output).is_err()
            );
            decode_ggml_k_block(codec, &vec![0; codec.block_bytes()], &mut output).unwrap();
            assert!(output.iter().all(|x| *x == 0.0));
        }
        // Q4_K codes 1, scale 1, minimum 0 give exactly the F16 global scale.
        let mut block = [0u8; 144];
        block[4..8].fill(1);
        block[12..16].fill(1);
        block[16..].fill(0x11);
        for (bits, expected) in [
            (0x0001u16, 2.0f32.powi(-24)),
            (0x3c00, 1.0),
            (0x7bff, 65504.0),
        ] {
            block[..2].copy_from_slice(&bits.to_le_bytes());
            let mut output = [0.0; GGML_K_BLOCK_VALUES];
            decode_ggml_k_block(GgmlKQuant::Q4K, &block, &mut output).unwrap();
            assert!(output.iter().all(|x| x.to_bits() == expected.to_bits()));
        }
        assert_eq!(
            fp16_at(&0x8000u16.to_le_bytes(), 0).to_bits(),
            (-0.0f32).to_bits()
        );
        assert_eq!(fp16_at(&0x7c00u16.to_le_bytes(), 0), f32::INFINITY);
        assert!(fp16_at(&0x7e01u16.to_le_bytes(), 0).is_nan());
    }
}

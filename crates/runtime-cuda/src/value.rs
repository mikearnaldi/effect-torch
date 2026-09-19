use crate::buffer::CudaBuffer;
use crate::device::{CUDA_TOP_K_BLOCKS, CUDA_TOP_K_LIMIT};
use crate::CudaDevice;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use effect_torch_graph::{Device, LeafValue};
use effect_torch_runtime::{
    DType, LayoutConstraintSpec, PackedFormat, StorageLayout, StorageMetadata,
    StorageRepresentation, ValueSpec, MAX_SAMPLING_VOCABULARY,
};
use half::{bf16, f16};
use std::any::Any;
use std::sync::Arc;

pub(crate) fn element_count(shape: &[usize]) -> Result<usize, String> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or_else(|| "CUDA tensor element count overflowed usize".to_string())
    })
}

/// Validates exact canonical allocation geometry before upload or publication.
pub(crate) fn validate_storage_bytes(spec: ValueSpec<'_>, bytes: usize) -> Result<(), String> {
    spec.validate()?;
    match spec.storage.layout_constraint {
        LayoutConstraintSpec::Canonical | LayoutConstraintSpec::Unconstrained => {}
        _ => return Err("CUDA values require canonical contiguous storage".to_string()),
    }
    let expected = spec.canonical_geometry()?.byte_len;
    if bytes != expected {
        return Err(format!(
            "CUDA storage requires {expected} bytes, received {bytes}"
        ));
    }
    Ok(())
}

// Round the complete binary64 significand once. The half crate can discard
// sticky bits, or convert through F32 on some hosts, before its final rounding.
fn f64_to_half_bits(value: f64, fraction_bits: u32, exponent_bias: i32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 48) & 0x8000) as u16;
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    let infinity = ((2 * exponent_bias + 1) as u16) << fraction_bits;
    if exponent == 0x7ff {
        let payload = if fraction == 0 {
            0
        } else {
            (fraction >> (52 - fraction_bits)) as u16 | (1 << (fraction_bits - 1))
        };
        return sign | infinity | payload;
    }
    let exponent = exponent - 1023;
    let minimum_normal = 1 - exponent_bias;
    let minimum_subnormal = minimum_normal - fraction_bits as i32;
    if exponent < minimum_subnormal - 1 {
        return sign;
    }
    if exponent > exponent_bias {
        return sign | infinity;
    }
    let significand = fraction | (1u64 << 52);
    let shift = if exponent < minimum_normal {
        (52 + minimum_subnormal - exponent) as u32
    } else {
        52 - fraction_bits
    };
    let retained = significand >> shift;
    let remainder = significand & ((1u64 << shift) - 1);
    let midpoint = 1u64 << (shift - 1);
    let rounded =
        retained + u64::from(remainder > midpoint || (remainder == midpoint && retained & 1 != 0));
    let encoded = if exponent < minimum_normal {
        rounded as u16
    } else {
        (((exponent + exponent_bias) as u16) << fraction_bits) + rounded as u16
            - (1 << fraction_bits)
    };
    sign | encoded
}

pub(crate) fn dense_bytes_from_host(values: &[f64], dtype: DType) -> Vec<u8> {
    let mut bytes = Vec::new();
    for &value in values {
        match dtype {
            DType::F64 => bytes.extend_from_slice(&value.to_le_bytes()),
            DType::F32 => bytes.extend_from_slice(&(value as f32).to_le_bytes()),
            DType::F16 => bytes.extend_from_slice(&f64_to_half_bits(value, 10, 15).to_le_bytes()),
            DType::BF16 => bytes.extend_from_slice(&f64_to_half_bits(value, 7, 127).to_le_bytes()),
            DType::I64 => bytes.extend_from_slice(&(value as i64).to_le_bytes()),
            DType::U32 => bytes.extend_from_slice(&(value as u32).to_le_bytes()),
            DType::U8 => bytes.push(value as u8),
        }
    }
    bytes
}

/// Rust element types with exact dense CUDA storage.
pub(crate) trait CudaElement: Send + Sync + 'static {
    const DTYPE: DType;
}

impl CudaElement for f64 {
    const DTYPE: DType = DType::F64;
}
impl CudaElement for f32 {
    const DTYPE: DType = DType::F32;
}
impl CudaElement for f16 {
    const DTYPE: DType = DType::F16;
}
impl CudaElement for bf16 {
    const DTYPE: DType = DType::BF16;
}
impl CudaElement for i64 {
    const DTYPE: DType = DType::I64;
}
impl CudaElement for u32 {
    const DTYPE: DType = DType::U32;
}
impl CudaElement for u8 {
    const DTYPE: DType = DType::U8;
}

/// A contiguous allocation with exact dtype widths and logical representation metadata.
#[derive(Clone)]
pub struct CudaValue {
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) buffer: Arc<CudaBuffer<u8>>,
    shape: Arc<[usize]>,
    dtype: DType,
    storage: StorageMetadata,
}

impl CudaValue {
    pub(crate) fn from_dense_bytes(
        device: Arc<CudaDevice>,
        shape: Vec<usize>,
        dtype: DType,
        bytes: &[u8],
    ) -> Result<Self, String> {
        let spec = ValueSpec::dense(dtype, &shape);
        validate_storage_bytes(spec, bytes.len())?;
        let buffer = CudaBuffer::from_slice(
            device
                .stream
                .clone_htod(bytes)
                .map_err(|error| error.to_string())?,
        );
        Self::from_planned_buffer(device, spec, buffer)
    }

    pub(crate) fn from_packed_bytes(
        device: Arc<CudaDevice>,
        logical_shape: Vec<usize>,
        format: PackedFormat,
        bytes: &[u8],
    ) -> Result<Self, String> {
        let storage = StorageMetadata {
            representation: StorageRepresentation::Packed(format),
            layout: StorageLayout::Canonical,
        };
        let spec = ValueSpec {
            semantic_dtype: DType::F32,
            logical_shape: &logical_shape,
            storage: storage.as_spec(),
        };
        validate_storage_bytes(spec, bytes.len())?;
        let buffer = CudaBuffer::from_slice(
            device
                .stream
                .clone_htod(bytes)
                .map_err(|error| error.to_string())?,
        );
        Self::from_planned_buffer(device, spec, buffer)
    }

    pub(crate) fn from_planned_buffer(
        device: Arc<CudaDevice>,
        spec: ValueSpec<'_>,
        buffer: CudaBuffer<u8>,
    ) -> Result<Self, String> {
        validate_storage_bytes(spec, buffer.len())?;
        let alignment = spec.canonical_geometry()?.physical_dtype.size_in_bytes() as u64;
        if buffer.address() % alignment != 0 {
            return Err(format!(
                "CUDA storage pointer is not aligned to {alignment} bytes"
            ));
        }
        Ok(Self {
            device,
            buffer: Arc::new(buffer),
            shape: spec.logical_shape.into(),
            dtype: spec.semantic_dtype,
            storage: StorageMetadata {
                representation: spec.storage.representation,
                layout: StorageLayout::Canonical,
            },
        })
    }

    /// Explicit numerical conversion from host doubles, never used by raw archive I/O.
    pub(crate) fn from_host(
        device: Arc<CudaDevice>,
        shape: Vec<usize>,
        dtype: DType,
        values: &[f64],
    ) -> Result<Self, String> {
        if values.len() != element_count(&shape)? {
            return Err("CUDA host value count does not match shape".to_string());
        }
        Self::from_dense_bytes(device, shape, dtype, &dense_bytes_from_host(values, dtype))
    }

    pub(crate) fn write_storage_bytes(&self, bytes: &[u8]) -> Result<(), String> {
        self.require_dense()?;
        let expected = self.storage_bytes();
        if bytes.len() != expected {
            return Err(format!(
                "CUDA storage requires {expected} bytes, received {}",
                bytes.len()
            ));
        }
        let mut buffer = self.buffer.as_ref().clone();
        self.device
            .stream
            .memcpy_htod(bytes, &mut buffer)
            .map_err(|error| error.to_string())
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn ordinal(&self) -> u32 {
        self.device.ordinal
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn spec(&self) -> ValueSpec<'_> {
        ValueSpec {
            semantic_dtype: self.dtype,
            logical_shape: &self.shape,
            storage: self.storage.as_spec(),
        }
    }
    pub(crate) fn storage_bytes(&self) -> usize {
        self.buffer.len()
    }
    pub(crate) fn storage_address(&self) -> u64 {
        self.buffer.address()
    }

    pub(crate) fn reshape_dense(&self, shape: Vec<usize>) -> Result<Self, String> {
        self.require_dense()?;
        validate_storage_bytes(ValueSpec::dense(self.dtype, &shape), self.storage_bytes())?;
        let mut reshaped = self.clone();
        reshaped.shape = shape.into();
        Ok(reshaped)
    }

    fn require_dense(&self) -> Result<(), String> {
        if self.storage.representation != StorageRepresentation::Dense {
            return Err("CUDA packed values require representation-aware operations".to_string());
        }
        Ok(())
    }

    fn require_dtype(&self, dtype: DType) -> Result<(), String> {
        self.require_dense()?;
        if self.dtype != dtype {
            return Err(format!(
                "CUDA typed view requires {dtype}, received {}",
                self.dtype
            ));
        }
        Ok(())
    }

    pub(crate) fn typed_buffer<T: CudaElement>(&self) -> Result<CudaBuffer<T>, String> {
        self.require_dtype(T::DTYPE)?;
        self.buffer
            .cast(self.storage_bytes() / std::mem::size_of::<T>())
    }

    pub(crate) fn greedy_argmax(&self) -> Result<u32, String> {
        let logits = self.typed_buffer::<f32>()?;
        let len = element_count(&self.shape)?;
        let len = u32::try_from(len)
            .map_err(|_| "CUDA greedy argmax input exceeds u32 indexing".to_string())?;
        if len == 0 {
            return Err("sample: logits must not be empty".to_string());
        }
        let mut output = self
            .device
            .greedy_argmax_output
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut launch = self
            .device
            .stream
            .launch_builder(&self.device.f32.greedy_argmax);
        launch.arg(&logits);
        launch.arg(&len);
        launch.arg(&mut *output);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| error.to_string())?;
        let result = self
            .device
            .stream
            .clone_dtoh(&*output)
            .map_err(|error| error.to_string())?;
        if result[1] != u32::MAX {
            return Err(format!("sample: logit {} is not finite", result[1]));
        }
        Ok(result[0])
    }

    pub(crate) fn topk(&self, k: usize) -> Result<(Vec<f64>, Vec<u32>), String> {
        let logits = self.typed_buffer::<f32>()?;
        let len = element_count(&self.shape)?;
        if len == 0 {
            return Err("sample: logits must be non-empty".to_string());
        }
        if len > MAX_SAMPLING_VOCABULARY {
            return Err(format!(
                "sample: vocabulary {len} exceeds limit {MAX_SAMPLING_VOCABULARY}"
            ));
        }
        if k == 0 || k > len {
            return Err(format!("sample: topK must be in [1, {len}], got {k}"));
        }
        if k > CUDA_TOP_K_LIMIT {
            return Err(format!("CUDA top-k exceeds limit {CUDA_TOP_K_LIMIT}"));
        }
        let len =
            u32::try_from(len).map_err(|_| "CUDA top-k input exceeds u32 indexing".to_string())?;
        let k = u32::try_from(k).expect("CUDA top-k limit fits u32");
        let mut output = self
            .device
            .topk_output
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut launch = self.device.stream.launch_builder(&self.device.f32.topk);
        launch.arg(&logits);
        launch.arg(&len);
        launch.arg(&k);
        launch.arg(&mut *output);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (CUDA_TOP_K_BLOCKS as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| error.to_string())?;
        let result = self
            .device
            .stream
            .clone_dtoh(&*output)
            .map_err(|error| error.to_string())?;
        let output_count = CUDA_TOP_K_BLOCKS * CUDA_TOP_K_LIMIT;
        let invalid = result[2 * output_count..]
            .iter()
            .map(|value| value.to_bits())
            .filter(|index| *index != u32::MAX)
            .min();
        if let Some(invalid) = invalid {
            return Err(format!("sample: logit {invalid} is not finite"));
        }
        let k = k as usize;
        let mut values = Vec::with_capacity(CUDA_TOP_K_BLOCKS * k);
        let mut tokens = Vec::with_capacity(CUDA_TOP_K_BLOCKS * k);
        for block in 0..CUDA_TOP_K_BLOCKS {
            let offset = block * CUDA_TOP_K_LIMIT;
            values.extend(result[offset..offset + k].iter().copied().map(f64::from));
            tokens.extend(
                result[output_count + offset..output_count + offset + k]
                    .iter()
                    .map(|token| token.to_bits()),
            );
        }
        Ok((values, tokens))
    }

    pub fn read_storage_bytes(&self) -> Result<Vec<u8>, String> {
        self.device
            .stream
            .clone_dtoh(self.buffer.as_ref())
            .map_err(|error| error.to_string())
    }

    pub fn readback_i64(&self) -> Result<Vec<i64>, String> {
        self.device
            .stream
            .clone_dtoh(&self.typed_buffer::<i64>()?)
            .map_err(|error| error.to_string())
    }

    /// Numerical host readback. I64 callers needing exact values use readback_i64.
    pub fn readback(&self) -> Result<Vec<f64>, String> {
        self.require_dense()?;
        let bytes = self.read_storage_bytes()?;
        Ok(bytes
            .chunks_exact(self.dtype.size_in_bytes())
            .map(|chunk| match self.dtype {
                DType::F64 => f64::from_le_bytes(chunk.try_into().expect("validated f64 bytes")),
                DType::F32 => f64::from(f32::from_le_bytes(
                    chunk.try_into().expect("validated f32 bytes"),
                )),
                DType::F16 => {
                    f16::from_le_bytes(chunk.try_into().expect("validated f16 bytes")).to_f64()
                }
                DType::BF16 => {
                    bf16::from_le_bytes(chunk.try_into().expect("validated bf16 bytes")).to_f64()
                }
                DType::I64 => {
                    i64::from_le_bytes(chunk.try_into().expect("validated i64 bytes")) as f64
                }
                DType::U32 => f64::from(u32::from_le_bytes(
                    chunk.try_into().expect("validated u32 bytes"),
                )),
                DType::U8 => f64::from(chunk[0]),
            })
            .collect())
    }
}

impl LeafValue for CudaValue {
    fn shape(&self) -> Vec<usize> {
        self.shape.to_vec()
    }
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn device(&self) -> Device {
        Device::Cuda(self.device.ordinal)
    }
    fn storage(&self) -> StorageMetadata {
        self.storage.clone()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
#[path = "value_storage_tests.rs"]
mod tests;

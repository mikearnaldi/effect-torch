//! Leaf-value wrapper around [`MetalTensor`] for the graph layer.
//!
//! [`Value`] is the concrete `LeafValue` used by compiler graph walks. This
//! thin newtype exposes shape, dtype, and device metadata. It also provides
//! constructors to upload host bytes or reserve shared-memory destination
//! storage for the NAPI addon. The boundary rejects f64 because
//! this runtime has no double-precision Metal support.

use crate::device::MetalDevice;
#[cfg(test)]
use crate::kernels;
use crate::run::MetalTensor;
use effect_torch_graph::Device;
use effect_torch_runtime::{DType, Layout};
#[cfg(feature = "napi-addon")]
use std::sync::Arc;

/// A graph leaf value backed by a Metal tensor.
#[derive(Clone)]
pub struct Value(pub MetalTensor, Option<PackedValue>);

#[derive(Clone)]
struct PackedValue {
    shape: Vec<usize>,
    storage: effect_torch_runtime::StorageMetadata,
}

impl Value {
    pub fn dense(tensor: MetalTensor) -> Self {
        Self(tensor, None)
    }

    /// Attach a validated canonical representation to imported packed bytes.
    pub fn with_packed_storage(
        self,
        codec: effect_torch_runtime::GgmlKQuant,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        let storage = effect_torch_runtime::StorageMetadata::packed(codec);
        let spec = effect_torch_runtime::ValueSpec {
            semantic_dtype: DType::F32,
            logical_shape: &shape,
            storage: storage.as_spec(),
        };
        let bytes = self
            .0
            .layout
            .checked_byte_size(self.0.dtype)
            .ok_or("packed buffer extent overflow")?;
        if self.1.is_some() {
            return Err("value already has packed storage".to_string());
        }
        spec.validate_buffer(self.0.dtype, &self.0.layout, bytes)?;
        if self.0.buffer.size < bytes {
            return Err("packed bytes exceed the Metal allocation".to_string());
        }
        Ok(Self(self.0, Some(PackedValue { shape, storage })))
    }

    pub fn value_spec(&self) -> effect_torch_runtime::ValueSpec<'_> {
        match &self.1 {
            Some(packed) => effect_torch_runtime::ValueSpec {
                semantic_dtype: DType::F32,
                logical_shape: &packed.shape,
                storage: packed.storage.as_spec(),
            },
            None => effect_torch_runtime::Buffer::value_spec(&self.0),
        }
    }

    pub fn storage(&self) -> effect_torch_runtime::StorageMetadata {
        self.1
            .as_ref()
            .map(|packed| packed.storage.clone())
            .unwrap_or_else(|| effect_torch_runtime::StorageMetadata {
                representation: effect_torch_runtime::StorageRepresentation::Dense,
                layout: effect_torch_runtime::StorageLayout::DenseStrided(self.0.layout.clone()),
            })
    }
    /// The Metal device this value lives on.
    pub fn device(&self) -> Device {
        Device::Metal(self.0.buffer.device_ordinal())
    }

    /// The wrapped tensor (infallible: a `Value` is always Metal-backed).
    pub fn as_metal(&self) -> Result<&MetalTensor, String> {
        Ok(&self.0)
    }

    /// Element type of the wrapped tensor.
    pub fn dtype(&self) -> DType {
        self.value_spec().semantic_dtype
    }

    /// Logical shape of the wrapped tensor.
    pub fn shape(&self) -> &[usize] {
        self.value_spec().logical_shape
    }

    /// Physical byte size, including the packed representation when present.
    pub fn byte_size(&self) -> usize {
        self.0.numel() * self.0.dtype.size_in_bytes()
    }

    /// Synchronizes, copies to contiguous f32 on device, and reads back to
    /// the host (tests only).
    #[cfg(test)]
    pub fn to_f32_vec(&self) -> Result<Vec<f32>, String> {
        if self.1.is_some() {
            return Err("packed values require explicit dequantization".to_string());
        }
        MetalDevice::with_ordinal(self.0.buffer.device_ordinal() as usize, || {
            let device = MetalDevice::get();
            let tensor = kernels::strided_copy(device, &self.0)?;
            let tensor = if tensor.dtype == DType::F32 {
                tensor
            } else {
                kernels::cast(device, &tensor, DType::F32)?
            };
            device.synchronize()?;
            tensor.read_f32()
        })?
    }
}

/// Builds a contiguous value by uploading `bytes` (exactly
/// `numel(shape) * dtype.size_in_bytes()` of them) to the device.
pub(crate) fn value_from_bytes(
    bytes: &[u8],
    shape: &[usize],
    dtype: DType,
) -> Result<Value, String> {
    let elements = shape
        .iter()
        .try_fold(1usize, |total, &dimension| total.checked_mul(dimension))
        .ok_or_else(|| "tensor element count overflows".to_string())?;
    let expected = elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| "tensor byte length overflows".to_string())?;
    if bytes.len() != expected {
        return Err(format!(
            "expected {expected} bytes for {dtype} tensor with shape {shape:?}, got {}",
            bytes.len()
        ));
    }
    if dtype == DType::F64 {
        return Err("f64 is not supported on Metal".to_string());
    }
    Ok(Value::dense(MetalTensor {
        buffer: MetalDevice::get().upload_bytes(bytes),
        layout: Layout::contiguous(shape.to_vec()),
        dtype,
    }))
}

/// Allocates an uninitialized contiguous value in shared memory, suitable
/// as a zero-copy destination for host writes via [`write_value_bytes`].
#[cfg(feature = "napi-addon")]
pub(crate) fn empty_shared_value(shape: &[usize], dtype: DType) -> Result<Value, String> {
    let elements = shape
        .iter()
        .try_fold(1usize, |total, &dimension| total.checked_mul(dimension))
        .ok_or_else(|| "tensor element count overflows".to_string())?;
    let byte_len = elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| "tensor byte length overflows".to_string())?;
    if dtype == DType::F64 {
        return Err("f64 is not supported on Metal".to_string());
    }
    Ok(Value::dense(MetalTensor {
        buffer: MetalDevice::get().alloc_raw_checked(byte_len)?,
        layout: Layout::contiguous(shape.to_vec()),
        dtype,
    }))
}

/// Runs `write` against the value's raw bytes. Requires a contiguous
/// layout at offset zero and unique (`Arc`) ownership of the storage, so
/// no GPU work or other view can alias the bytes being written.
#[cfg(feature = "napi-addon")]
pub(crate) fn write_value_bytes<R>(
    value: &mut Value,
    write: impl FnOnce(&mut [u8]) -> R,
) -> Result<R, String> {
    if !value.0.layout.is_contiguous() || value.0.layout.offset() != 0 {
        return Err("tensor byte destination must be contiguous at offset zero".to_string());
    }
    let byte_len = value
        .0
        .numel()
        .checked_mul(value.0.dtype.size_in_bytes())
        .ok_or_else(|| "tensor byte length overflows".to_string())?;
    let buffer = Arc::get_mut(&mut value.0.buffer)
        .ok_or_else(|| "tensor byte destination storage is shared".to_string())?;
    if byte_len > buffer.size {
        return Err(format!(
            "tensor byte destination requires {byte_len} bytes, buffer has {}",
            buffer.size
        ));
    }
    // SAFETY: `Arc::get_mut` above proves unique ownership of the buffer,
    // so no other host view or GPU consumer aliases it; the buffer is
    // shared-mode so `contents_ptr()` is a valid host mapping, and the
    // `byte_len > buffer.size` check above bounds the slice.
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(buffer.contents_ptr().cast::<u8>(), byte_len) };
    Ok(write(bytes))
}

impl effect_torch_graph::LeafValue for Value {
    fn shape(&self) -> Vec<usize> {
        self.shape().to_vec()
    }

    fn dtype(&self) -> DType {
        self.dtype()
    }

    fn device(&self) -> Device {
        self.device()
    }

    fn storage(&self) -> effect_torch_runtime::StorageMetadata {
        self.storage()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

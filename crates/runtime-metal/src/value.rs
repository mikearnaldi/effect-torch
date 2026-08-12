use crate::device::MetalDevice;
#[cfg(test)]
use crate::kernels;
use crate::run::MetalTensor;
use effect_torch_graph::Device;
use effect_torch_runtime::{DType, Layout};
use std::sync::Arc;

#[derive(Clone)]
pub struct Value(pub MetalTensor);

impl Value {
    pub fn device(&self) -> Device {
        Device::Metal
    }

    pub fn as_metal(&self) -> Result<&MetalTensor, String> {
        Ok(&self.0)
    }

    pub fn dtype(&self) -> DType {
        self.0.dtype
    }

    pub fn shape(&self) -> &[usize] {
        self.0.layout.shape()
    }

    pub fn numel(&self) -> usize {
        self.0.numel()
    }

    pub fn byte_size(&self) -> usize {
        self.numel() * self.dtype().size_in_bytes()
    }

    #[cfg(test)]
    pub fn to_f32_vec(&self) -> Result<Vec<f32>, String> {
        let device = MetalDevice::get();
        let tensor = kernels::strided_copy(device, &self.0)?;
        let tensor = if tensor.dtype == DType::F32 {
            tensor
        } else {
            kernels::cast(device, &tensor, DType::F32)?
        };
        device.synchronize()?;
        tensor.read_f32()
    }
}

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
    Ok(Value(MetalTensor {
        buffer: MetalDevice::get().upload_bytes(bytes),
        layout: Layout::contiguous(shape.to_vec()),
        dtype,
    }))
}

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
    Ok(Value(MetalTensor {
        buffer: MetalDevice::get().alloc_raw_checked(byte_len)?,
        layout: Layout::contiguous(shape.to_vec()),
        dtype,
    }))
}

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
    // SAFETY: this is a fresh shared MTLBuffer with unique Arc ownership.
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

use crate::device::MetalDevice;
#[cfg(test)]
use crate::kernels;
use crate::run::MetalTensor;
use effect_torch_graph::Device;
use effect_torch_runtime::{DType, Layout};

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

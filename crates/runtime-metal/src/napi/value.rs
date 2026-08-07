use super::err::{err, Res};
use crate::{device::MetalDevice, kernels, run::MetalTensor};
use effect_torch_graph::Device;
use effect_torch_runtime::DType;

#[derive(Clone)]
pub struct Value(pub MetalTensor);

impl Value {
    pub fn device(&self) -> Device {
        Device::Metal
    }

    pub fn as_metal(&self) -> Res<&MetalTensor> {
        Ok(&self.0)
    }

    pub fn dtype(&self) -> DType {
        self.0.dtype
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.layout.shape().to_vec()
    }

    pub fn buffer_key(&self) -> Option<usize> {
        Some(std::sync::Arc::as_ptr(&self.0.buffer) as usize)
    }

    pub fn numel(&self) -> usize {
        self.0.numel()
    }

    pub fn byte_size(&self) -> usize {
        self.numel() * self.dtype().size_in_bytes()
    }

    pub fn synchronize(&self) -> Res<()> {
        MetalDevice::get().synchronize()
    }

    pub fn to_f32_vec(&self) -> Res<Vec<f32>> {
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

    pub fn to_f64_vec(&self) -> Res<Vec<f64>> {
        Ok(self
            .to_f32_vec()?
            .into_iter()
            .map(|value| value as f64)
            .collect())
    }

    pub fn to_u32_vec(&self) -> Res<Vec<u32>> {
        let device = MetalDevice::get();
        let tensor = kernels::strided_copy(device, &self.0)?;
        device.synchronize()?;
        let count = tensor.numel();
        let element_size = tensor.dtype.size_in_bytes();
        let pointer = unsafe {
            tensor
                .buffer
                .contents_ptr()
                .cast::<u8>()
                .add(tensor.layout.offset() * element_size)
        };
        let bytes = unsafe { std::slice::from_raw_parts(pointer, count * element_size) };
        let values = match tensor.dtype {
            DType::U32 => bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
                .collect(),
            DType::U8 => bytes.iter().map(|&value| value as u32).collect(),
            DType::I64 => bytes
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")) as u32)
                .collect(),
            _ => return err("to_u32_vec: dtype must be u8/u32/i64"),
        };
        Ok(values)
    }

    pub fn to_i64_vec(&self) -> Res<Vec<i64>> {
        Ok(self
            .to_u32_vec()?
            .into_iter()
            .map(|value| value as i64)
            .collect())
    }

    pub fn to_u8_vec(&self) -> Res<Vec<u8>> {
        let device = MetalDevice::get();
        let tensor = kernels::strided_copy(device, &self.0)?;
        device.synchronize()?;
        let count = tensor.numel();
        let element_size = tensor.dtype.size_in_bytes();
        let pointer = unsafe {
            tensor
                .buffer
                .contents_ptr()
                .cast::<u8>()
                .add(tensor.layout.offset() * element_size)
        };
        let bytes = unsafe { std::slice::from_raw_parts(pointer, count * element_size) };
        match tensor.dtype {
            DType::U8 => Ok(bytes.to_vec()),
            DType::U32 => Ok(bytes.chunks_exact(4).map(|chunk| chunk[0]).collect()),
            _ => err("to_u8_vec: dtype must be u8/u32"),
        }
    }
}

impl effect_torch_graph::LeafValue for Value {
    fn shape(&self) -> Vec<usize> {
        self.shape()
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

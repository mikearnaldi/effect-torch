use crate::{CpuBuffer, Tensor};
use effect_torch_graph::Device;
use effect_torch_runtime::DType;

#[derive(Clone, Debug)]
pub struct Value(pub Tensor);

impl Value {
    pub fn tensor(&self) -> &Tensor {
        &self.0
    }

    #[cfg(test)]
    pub fn into_tensor(self) -> Tensor {
        self.0
    }

    pub fn device(&self) -> Device {
        Device::Cpu
    }

    pub fn dtype(&self) -> DType {
        self.0.dtype()
    }

    pub fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    pub fn numel(&self) -> usize {
        self.0.numel()
    }

    pub fn byte_size(&self) -> usize {
        self.numel() * self.dtype().size_in_bytes()
    }

    pub fn to_f32_vec(&self) -> Result<Vec<f32>, String> {
        let tensor = self.0.cast(DType::F32).contiguous();
        let CpuBuffer::F32(values) = &tensor.buffer else {
            unreachable!()
        };
        Ok(values.as_slice().to_vec())
    }

    pub fn to_f64_vec(&self) -> Result<Vec<f64>, String> {
        let tensor = self.0.cast(DType::F64).contiguous();
        let CpuBuffer::F64(values) = &tensor.buffer else {
            unreachable!()
        };
        Ok(values.as_slice().to_vec())
    }

    pub fn to_u32_vec(&self) -> Result<Vec<u32>, String> {
        let tensor = self.0.cast(DType::U32).contiguous();
        let CpuBuffer::U32(values) = &tensor.buffer else {
            unreachable!()
        };
        Ok(values.as_slice().to_vec())
    }

    pub fn to_i64_vec(&self) -> Result<Vec<i64>, String> {
        let tensor = self.0.cast(DType::I64).contiguous();
        let CpuBuffer::I64(values) = &tensor.buffer else {
            unreachable!()
        };
        Ok(values.as_slice().to_vec())
    }

    pub fn to_u8_vec(&self) -> Result<Vec<u8>, String> {
        let tensor = self.0.cast(DType::U8).contiguous();
        let CpuBuffer::U8(values) = &tensor.buffer else {
            unreachable!()
        };
        Ok(values.as_slice().to_vec())
    }
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

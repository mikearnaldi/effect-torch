//! [`Value`]: the graph-leaf payload wrapper around a CPU [`Tensor`].
//!
//! The graph and compiler crates talk about opaque leaf values through the
//! [`effect_torch_graph::LeafValue`] trait; this module adapts CPU tensors to
//! that interface and provides the typed read-back helpers used by the NAPI
//! boundary and tests.

use crate::{CpuBuffer, Tensor};
use effect_torch_graph::Device;
use effect_torch_runtime::DType;

/// A CPU tensor packaged as a graph leaf value.
///
/// Cloning is cheap: the underlying buffer is reference-counted, so a cloned
/// `Value` aliases the same storage as the original.
#[derive(Clone, Debug)]
pub struct Value(pub Tensor);

impl Value {
    /// Borrows the wrapped tensor.
    pub fn tensor(&self) -> &Tensor {
        &self.0
    }

    #[cfg(test)]
    pub fn into_tensor(self) -> Tensor {
        self.0
    }

    /// Always [`Device::Cpu`]; this runtime only produces CPU values.
    pub fn device(&self) -> Device {
        Device::Cpu
    }

    /// Element type of the wrapped tensor.
    pub fn dtype(&self) -> DType {
        self.0.dtype()
    }

    /// Logical shape of the wrapped tensor.
    pub fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    /// Total number of logical elements.
    pub fn numel(&self) -> usize {
        self.0.numel()
    }

    /// Logical size in bytes (`numel * dtype size`); ignores any padding the
    /// backing allocation may carry.
    pub fn byte_size(&self) -> usize {
        self.numel() * self.dtype().size_in_bytes()
    }

    /// Materializes the value as a dense `f32` vector, casting and gathering
    /// strided layouts as needed.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>, String> {
        let tensor = self.0.cast(DType::F32).contiguous();
        let CpuBuffer::F32(values) = &tensor.buffer else {
            unreachable!()
        };
        Ok(values.as_slice().to_vec())
    }

    /// Materializes the value as a dense `f64` vector.
    pub fn to_f64_vec(&self) -> Result<Vec<f64>, String> {
        let tensor = self.0.cast(DType::F64).contiguous();
        let CpuBuffer::F64(values) = &tensor.buffer else {
            unreachable!()
        };
        Ok(values.as_slice().to_vec())
    }

    /// Materializes the value as a dense `u32` vector.
    pub fn to_u32_vec(&self) -> Result<Vec<u32>, String> {
        let tensor = self.0.cast(DType::U32).contiguous();
        let CpuBuffer::U32(values) = &tensor.buffer else {
            unreachable!()
        };
        Ok(values.as_slice().to_vec())
    }

    /// Materializes the value as a dense `i64` vector.
    pub fn to_i64_vec(&self) -> Result<Vec<i64>, String> {
        let tensor = self.0.cast(DType::I64).contiguous();
        let CpuBuffer::I64(values) = &tensor.buffer else {
            unreachable!()
        };
        Ok(values.as_slice().to_vec())
    }

    /// Materializes the value as a dense `u8` vector.
    pub fn to_u8_vec(&self) -> Result<Vec<u8>, String> {
        let tensor = self.0.cast(DType::U8).contiguous();
        let CpuBuffer::U8(values) = &tensor.buffer else {
            unreachable!()
        };
        Ok(values.as_slice().to_vec())
    }
}

/// Exposes the value to the graph crate without revealing the concrete
/// tensor type.
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

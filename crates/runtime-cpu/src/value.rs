//! [`Value`] wraps a CPU [`Tensor`] as a graph-leaf payload.
//!
//! The graph and compiler crates use the [`effect_torch_graph::LeafValue`]
//! trait for opaque leaf values. This module implements that interface for CPU
//! tensors and provides typed readback helpers for the NAPI boundary and tests.

use crate::{CpuBuffer, Tensor};
use effect_torch_graph::Device;
use effect_torch_runtime::DType;

/// A CPU tensor packaged as a graph leaf value.
///
/// Cloning shares the reference-counted buffer, so the new `Value` aliases
/// the original storage.
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

    /// Always [`Device::Cpu(0)`] because this runtime only produces CPU values.
    pub fn device(&self) -> Device {
        Device::Cpu(0)
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

    /// Logical size in bytes, `numel * dtype size`. Ignores padding in the
    /// backing allocation.
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

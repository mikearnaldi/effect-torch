//! [`Value`] wraps a CPU [`Tensor`] as a graph-leaf payload.
//!
//! The graph and compiler crates use the [`effect_torch_graph::LeafValue`]
//! trait for opaque leaf values. This module implements that interface for CPU
//! tensors and provides typed readback helpers for the NAPI boundary and tests.

use crate::{CpuBuffer, Tensor};
use effect_torch_graph::Device;
use effect_torch_runtime::{
    Buffer, DType, GgmlKQuant, StorageLayout, StorageMetadata, StorageRepresentation, ValueSpec,
};

/// A CPU tensor packaged as a graph leaf value.
///
/// Cloning shares the reference-counted buffer, so the new `Value` aliases
/// the original storage.
#[derive(Clone, Debug)]
pub struct Value(pub Tensor, Option<PackedValue>);

#[derive(Clone, Debug)]
struct PackedValue {
    shape: Vec<usize>,
    storage: StorageMetadata,
}

impl Value {
    pub fn require_dense(&self) -> Result<(), String> {
        if self.1.is_some() {
            Err(
                "packed tensor does not support dense readback, conversion, or serialization"
                    .into(),
            )
        } else {
            Ok(())
        }
    }
    pub fn dense(tensor: Tensor) -> Self {
        Self(tensor, None)
    }

    /// Attaches canonical packed metadata after checking physical storage.
    pub fn with_packed_storage(self, codec: GgmlKQuant, shape: Vec<usize>) -> Result<Self, String> {
        if self.1.is_some() {
            return Err("value already has packed storage".into());
        }
        let storage = StorageMetadata::packed(codec);
        let spec = ValueSpec {
            semantic_dtype: DType::F32,
            logical_shape: &shape,
            storage: storage.as_spec(),
        };
        spec.validate_buffer(self.0.dtype(), &self.0.layout, self.0.buffer.byte_len())?;
        Ok(Self(self.0, Some(PackedValue { shape, storage })))
    }

    pub fn value_spec(&self) -> ValueSpec<'_> {
        match &self.1 {
            Some(packed) => ValueSpec {
                semantic_dtype: DType::F32,
                logical_shape: &packed.shape,
                storage: packed.storage.as_spec(),
            },
            None => self.0.value_spec(),
        }
    }

    pub fn storage(&self) -> StorageMetadata {
        self.1
            .as_ref()
            .map(|packed| packed.storage.clone())
            .unwrap_or_else(|| StorageMetadata {
                representation: StorageRepresentation::Dense,
                layout: StorageLayout::DenseStrided(self.0.layout.clone()),
            })
    }
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
        self.value_spec().semantic_dtype
    }

    /// Logical shape of the wrapped tensor.
    pub fn shape(&self) -> &[usize] {
        self.value_spec().logical_shape
    }

    /// Total number of logical elements.
    pub fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// Physical payload size in bytes. Packed values retain their encoded size.
    pub fn byte_size(&self) -> usize {
        self.0.numel() * self.0.dtype().size_in_bytes()
    }

    /// Materializes the value as a dense `f32` vector, casting and gathering
    /// strided layouts as needed.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>, String> {
        self.require_dense()?;
        let tensor = self.0.cast(DType::F32).contiguous();
        let CpuBuffer::F32(values) = &tensor.buffer else {
            unreachable!()
        };
        Ok(values.as_slice().to_vec())
    }

    /// Materializes the value as a dense `f64` vector.
    pub fn to_f64_vec(&self) -> Result<Vec<f64>, String> {
        self.require_dense()?;
        let tensor = self.0.cast(DType::F64).contiguous();
        let CpuBuffer::F64(values) = &tensor.buffer else {
            unreachable!()
        };
        Ok(values.as_slice().to_vec())
    }

    /// Materializes the value as a dense `u32` vector.
    pub fn to_u32_vec(&self) -> Result<Vec<u32>, String> {
        self.require_dense()?;
        let tensor = self.0.cast(DType::U32).contiguous();
        let CpuBuffer::U32(values) = &tensor.buffer else {
            unreachable!()
        };
        Ok(values.as_slice().to_vec())
    }

    /// Materializes the value as a dense `i64` vector.
    pub fn to_i64_vec(&self) -> Result<Vec<i64>, String> {
        self.require_dense()?;
        let tensor = self.0.cast(DType::I64).contiguous();
        let CpuBuffer::I64(values) = &tensor.buffer else {
            unreachable!()
        };
        Ok(values.as_slice().to_vec())
    }

    /// Materializes the value as a dense `u8` vector.
    pub fn to_u8_vec(&self) -> Result<Vec<u8>, String> {
        self.require_dense()?;
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
    fn storage(&self) -> StorageMetadata {
        self.storage()
    }
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

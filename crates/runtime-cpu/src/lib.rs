pub mod composed;
pub mod conv;
pub mod indexing;
pub mod linalg;
pub mod matmul;
#[cfg(feature = "napi-addon")]
pub mod napi;
pub mod ops;
pub mod pool;
pub mod random;
pub mod reduce;
pub mod tensor;

pub use tensor::{CpuBuffer, Elem, Tensor};

use effect_torch_runtime::{
    Backend, BackendError, BackendResult, Buffer, Capabilities, Capability, DType, DeviceId,
    ErasedBuffer, Placement, RuntimeIdentity,
};
use std::any::Any;
use std::sync::OnceLock;

fn identity() -> &'static RuntimeIdentity {
    static IDENTITY: OnceLock<RuntimeIdentity> = OnceLock::new();
    IDENTITY.get_or_init(|| RuntimeIdentity::new("cpu"))
}

fn placement() -> &'static Placement {
    static PLACEMENT: OnceLock<Placement> = OnceLock::new();
    PLACEMENT.get_or_init(|| Placement::new(DeviceId::new("cpu:0")))
}

impl Buffer for Tensor {
    fn runtime_id(&self) -> effect_torch_runtime::RuntimeId {
        identity().id()
    }

    fn placement(&self) -> &Placement {
        placement()
    }

    fn dtype(&self) -> DType {
        self.dtype()
    }

    fn layout(&self) -> &effect_torch_runtime::Layout {
        &self.layout
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuOperation {
    Add,
    Matmul,
}

pub struct CpuBackend {
    identity: RuntimeIdentity,
    capabilities: Capabilities,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            identity: identity().clone(),
            capabilities: Capabilities::new(
                vec![
                    DType::F32,
                    DType::F64,
                    DType::F16,
                    DType::BF16,
                    DType::U8,
                    DType::U32,
                    DType::I64,
                ],
                vec![Capability::Compilation, Capability::AsyncExecution],
            ),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend<CpuOperation> for CpuBackend {
    fn identity(&self) -> &RuntimeIdentity {
        &self.identity
    }

    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn execute(
        &self,
        operation: &CpuOperation,
        inputs: &[ErasedBuffer],
    ) -> BackendResult<ErasedBuffer> {
        if inputs.len() != 2 {
            return Err(BackendError::Execution {
                operation: Some(format!("{operation:?}")),
                message: format!("expected 2 inputs, got {}", inputs.len()),
            });
        }
        let mut tensors = Vec::with_capacity(2);
        for input in inputs {
            input.validate_owner(self.identity.id())?;
            tensors.push(input.downcast_ref::<Tensor>().ok_or_else(|| {
                BackendError::Execution {
                    operation: Some(format!("{operation:?}")),
                    message: "CPU backend received a non-CPU buffer".to_string(),
                }
            })?);
        }
        let output =
            match operation {
                CpuOperation::Add => tensors[0].add(tensors[1]),
                CpuOperation::Matmul => tensors[0].try_matmul(tensors[1]).map_err(|message| {
                    BackendError::Execution {
                        operation: Some(format!("{operation:?}")),
                        message: message.to_string(),
                    }
                })?,
            };
        Ok(ErasedBuffer::new(output))
    }
}

#[cfg(test)]
mod runtime_tests {
    use super::*;

    #[test]
    fn shared_backend_contract_executes_production_cpu_kernels() {
        let backend = CpuBackend::new();
        let a = ErasedBuffer::new(Tensor::from_vec(vec![1.0f32, 2.0], vec![2]));
        let b = ErasedBuffer::new(Tensor::from_vec(vec![3.0f32, 4.0], vec![2]));
        let output = backend.execute(&CpuOperation::Add, &[a, b]).unwrap();
        let tensor = output.downcast_ref::<Tensor>().unwrap();
        let CpuBuffer::F32(values) = &tensor.buffer else {
            panic!("expected f32 output")
        };
        assert_eq!(values.as_slice(), &[4.0, 6.0]);
        assert_eq!(output.runtime_id(), backend.identity().id());
        let foreign = RuntimeIdentity::new("cpu");
        assert!(matches!(
            output.validate_owner(foreign.id()),
            Err(BackendError::InvalidHandle { .. })
        ));
    }
}

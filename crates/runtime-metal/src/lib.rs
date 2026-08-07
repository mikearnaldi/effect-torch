//! Metal backend. Everything Metal lives here:
//!
//! - `device` — device singleton, pool allocator, encoder manager,
//!   pipeline cache.
//! - `emit` — first-party IR → MSL emitter (SSA form).
//! - `run` — `MetalTensor` and the fused elementwise/reduce runners.
//! - `kernels`, `indexing`, `conv`, `gemm` — primitive kernels
//!   (creation/cast/copy/random/reductions, gather/scatter/select/cat,
//!   conv family, tiled matmul).
//! - `ops` — the dispatch helpers the evaluator calls for ordinary
//!   ops (binary/unary/compare/cast/matmul/contiguous/views).
//! - `composed` — composite fallbacks built from `ops` (sdpa,
//!   layer_norm, cross_entropy, rotary, optimizer steps).
//! - `flash`, `loss`, `layer_norm`, `rotary`, `paged`, `linear` —
//!   semantic fused kernels.

#![cfg(target_os = "macos")]

pub mod err {
    pub type Res<T> = Result<T, String>;

    pub fn err<T>(message: impl Into<String>) -> Res<T> {
        Err(message.into())
    }

    pub fn err_str(message: impl Into<String>) -> String {
        message.into()
    }
}

pub mod fusion {
    pub use effect_torch_compiler::*;
}

pub mod runtime {
    pub mod dtype {
        pub use effect_torch_runtime::DType;
    }

    pub mod layout {
        pub use effect_torch_runtime::Layout;
    }

    #[cfg(test)]
    pub mod cpu {
        pub use effect_torch_runtime_cpu::*;
    }

    pub mod metal {
        pub use crate::{
            arena, composed, conv, device, emit, flash, gemm, indexing, kernels, layer_norm,
            linear, loss, ops, paged, rotary, run,
        };
    }
}

pub use effect_torch_graph::CrossEntropyReduction as CeReduction;

use effect_torch_runtime::{
    Backend, BackendError, BackendResult, Capabilities, Capability, DeviceId, ErasedBuffer,
    Placement, RuntimeIdentity,
};
use std::any::Any;
use std::sync::OnceLock;

pub mod arena;
pub mod conv;
pub mod device;
pub mod emit;
pub mod gemm;
pub mod indexing;
pub mod kernels;
#[cfg(feature = "napi-addon")]
pub mod napi;
pub mod run;

pub mod composed;
pub mod ops;

pub mod flash;
pub mod layer_norm;
pub mod linear;
pub mod loss;
pub mod paged;
pub mod rotary;

fn identity() -> &'static RuntimeIdentity {
    static IDENTITY: OnceLock<RuntimeIdentity> = OnceLock::new();
    IDENTITY.get_or_init(|| RuntimeIdentity::new("metal"))
}

fn placement() -> &'static Placement {
    static PLACEMENT: OnceLock<Placement> = OnceLock::new();
    PLACEMENT.get_or_init(|| Placement::with_memory_space(DeviceId::new("metal:0"), "shared"))
}

impl std::fmt::Debug for run::MetalTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalTensor")
            .field("layout", &self.layout)
            .field("dtype", &self.dtype)
            .finish_non_exhaustive()
    }
}

impl effect_torch_runtime::Buffer for run::MetalTensor {
    fn runtime_id(&self) -> effect_torch_runtime::RuntimeId {
        identity().id()
    }

    fn placement(&self) -> &Placement {
        placement()
    }

    fn dtype(&self) -> effect_torch_runtime::DType {
        self.dtype
    }

    fn layout(&self) -> &effect_torch_runtime::Layout {
        &self.layout
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalOperation {
    Add,
    Matmul,
}

pub struct MetalBackend {
    identity: RuntimeIdentity,
    capabilities: Capabilities,
}

impl MetalBackend {
    pub fn new() -> Self {
        Self {
            identity: identity().clone(),
            capabilities: Capabilities::new(
                vec![
                    effect_torch_runtime::DType::F32,
                    effect_torch_runtime::DType::F16,
                    effect_torch_runtime::DType::BF16,
                    effect_torch_runtime::DType::U8,
                    effect_torch_runtime::DType::U32,
                    effect_torch_runtime::DType::I64,
                ],
                vec![
                    Capability::Compilation,
                    Capability::AsyncExecution,
                    Capability::UnifiedMemory,
                ],
            ),
        }
    }
}

impl Default for MetalBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend<MetalOperation> for MetalBackend {
    fn identity(&self) -> &RuntimeIdentity {
        &self.identity
    }

    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn execute(
        &self,
        operation: &MetalOperation,
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
            tensors.push(input.downcast_ref::<run::MetalTensor>().ok_or_else(|| {
                BackendError::Execution {
                    operation: Some(format!("{operation:?}")),
                    message: "Metal backend received a non-Metal buffer".to_string(),
                }
            })?);
        }
        let output = match operation {
            MetalOperation::Add => ops::binary(tensors[0], tensors[1], ops::BinOp::Add),
            MetalOperation::Matmul => ops::matmul(tensors[0], tensors[1]),
        }
        .map_err(|message| BackendError::Execution {
            operation: Some(format!("{operation:?}")),
            message,
        })?;
        Ok(ErasedBuffer::new(output))
    }

    fn synchronize(&self) -> BackendResult<()> {
        device::MetalDevice::get()
            .synchronize()
            .map_err(|message| BackendError::Execution {
                operation: Some("synchronize".to_string()),
                message,
            })
    }
}

#[cfg(test)]
mod runtime_tests {
    use super::*;

    #[test]
    fn shared_backend_contract_executes_production_metal_kernels() {
        let backend = MetalBackend::new();
        let device = device::MetalDevice::get();
        let a = ErasedBuffer::new(run::MetalTensor::from_f32(device, vec![1.0, 2.0], vec![2]));
        let b = ErasedBuffer::new(run::MetalTensor::from_f32(device, vec![3.0, 4.0], vec![2]));
        let output = backend.execute(&MetalOperation::Add, &[a, b]).unwrap();
        backend.synchronize().unwrap();
        assert_eq!(
            output
                .downcast_ref::<run::MetalTensor>()
                .unwrap()
                .read_f32()
                .unwrap(),
            [4.0, 6.0]
        );
        assert_eq!(output.runtime_id(), backend.identity().id());
    }
}

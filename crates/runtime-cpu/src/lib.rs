//! CPU runtime for effect-torch: tensor storage, kernels, and the native
//! execution engine.
//!
//! This crate implements the reference CPU backend. Its central contract is
//! that every operation comes in two forms: allocating convenience wrappers
//! ([`Tensor::add`], [`Tensor::matmul`], ...) and allocation-free `*_into`
//! entry points that write into caller-provided [`CpuDestination`] buffers
//! whose sizes are fixed ahead of time by `*_requirements` planners
//! ([`CpuTensorRequirement`], [`CpuOperationRequirements`], and the
//! per-operation requirement structs in [`matmul`], [`conv`], [`linalg`], and
//! [`composed`]).
//!
//! Layering of the modules:
//!
//! - [`storage`] owns raw memory: aligned [`CpuSegment`] allocations, typed
//!   [`CpuStorage`] views, and the [`ExecutableAllocationGuard`] that forbids
//!   ambient allocation while a compiled executable runs.
//! - `workspace` pools reusable scratch segments behind a global LRU pool
//!   so repeated invocations of a compiled program do not hit the allocator.
//! - [`tensor`] defines [`Tensor`], the dtype-tagged [`CpuBuffer`], and the
//!   [`CpuDestination`] write capability with its safety rules.
//! - [`ops`], [`reduce`], [`indexing`], [`matmul`], [`conv`], [`linalg`],
//!   [`random`], and `quantized` implement the primitive kernels.
//! - [`composed`] implements higher-level kernels (attention, normalization,
//!   cross-entropy, optimizers, KDA, short convolutions, rotary embeddings).
//! - [`fusion`] interprets compiler-fused elementwise and reduction
//!   expression trees directly against planned destinations.
//! - [`executable`] lowers a prepared graph into a [`executable::CpuExecutable`]
//!   with a fixed memory plan and runs it allocation-free under the guard,
//!   observing cancellation.
//! - [`pool`] provides dtype-generic KV-cache slabs for stateful decoding.
//! - `napi` (feature `napi-addon`) exports the runtime to Node.js.

pub mod composed;
pub mod conv;
pub mod executable;
pub mod fusion;
pub mod indexing;
pub mod linalg;
pub mod matmul;
#[cfg(feature = "napi-addon")]
pub mod napi;
pub mod ops;
pub mod pool;
mod quantized;
pub mod random;
pub mod reduce;
pub mod storage;
pub mod tensor;
pub mod value;
mod workspace;

pub use storage::{
    CpuElement, CpuSegment, CpuStorage, CpuStorageError, ExecutableAllocationGuard,
    CPU_STORAGE_ALIGNMENT,
};
pub use tensor::{
    CpuBuffer, CpuDestination, CpuOperationRequirements, CpuTensorRequirement, Elem, Tensor,
};
pub use value::Value;

use effect_torch_runtime::{Buffer, DType, DeviceId, Placement, RuntimeIdentity};
use std::any::Any;
use std::sync::OnceLock;

/// Process-wide runtime identity (`"cpu"`) shared by every tensor this crate
/// produces.
fn identity() -> &'static RuntimeIdentity {
    static IDENTITY: OnceLock<RuntimeIdentity> = OnceLock::new();
    IDENTITY.get_or_init(|| RuntimeIdentity::new("cpu"))
}

fn placement() -> &'static Placement {
    static PLACEMENT: OnceLock<Placement> = OnceLock::new();
    PLACEMENT.get_or_init(|| Placement::new(DeviceId::new("cpu:0")))
}

/// Every CPU [`Tensor`] is a [`Buffer`] rooted at the single `cpu:0` device.
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

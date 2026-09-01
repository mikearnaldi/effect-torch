//! Metal backend.
//!
//! - `device` owns the device singleton, pool allocator, encoder manager,
//!   and pipeline cache.
//! - `emit` converts the fusion IR to MSL in SSA form.
//! - `run` defines `MetalTensor` and the fused elementwise and reduction runners.
//! - `kernels`, `indexing`, `conv`, and `gemm` implement primitive kernels.
//! - `ops` dispatches ordinary executable operations.
//! - `composed` provides numerical references for tests.
//! - `flash`, `loss`, `layer_norm`, `rotary`, `paged`, `linear`, and
//!   `quantized` implement fused operations.
//!
//! # Runtime invariants
//!
//! - One process-wide [`device::MetalDevice`] singleton owns the `MTLDevice`
//!   and one `MTLCommandQueue`. Per-stream submission contexts handle all
//!   encoding. The queue is thread-safe at the Metal API boundary.
//! - Kernels are encoded ahead of GPU execution. Host reads require
//!   `synchronize` or `synchronize_buffer` to drain the producing stream.
//!   The pool never recycles a buffer while a command buffer can reference it.
//! - Every `*_into` API requires its pipeline to be warm. Call the matching
//!   `compile_*` or `warm_*` function, or use the allocating wrapper, before
//!   dispatch. Executable dispatch rejects allocation and compilation.
//! - `EFFECT_TORCH_MEMORY_CAP_MB` sets a hard live-byte ceiling.
//!   `EFFECT_TORCH_MEMORY_BUDGET_MB` sets the soft backpressure budget.
//!   `EFFECT_TORCH_PRIVATE_INTERMEDIATES`, `EFFECT_TORCH_NO_MMA`,
//!   `EFFECT_TORCH_METAL_PROFILE`, and `EFFECT_TORCH_SYNC_TRACE` control
//!   storage mode, MMA selection, profiling, and sync tracing. The runtime
//!   snapshots all six variables once.
//! - The fusion emitter computes in f32 and stores f32/bf16. Primitive kernels
//!   support every `DType` except f64, which Metal does not support.
//!   Destinations must be contiguous. Input strides become constants in the
//!   emitted source.

#![cfg(target_os = "macos")]
#![cfg_attr(not(feature = "napi-addon"), allow(dead_code))]

/// String-based errors shared by every Metal module.
pub mod err {
    /// The Metal runtime's result type. Errors have no taxonomy.
    pub type Res<T> = Result<T, String>;

    /// Builds an `Err` from anything string-like.
    pub fn err<T>(message: impl Into<String>) -> Res<T> {
        Err(message.into())
    }

    /// Builds the error string itself (for `map_err`/`ok_or_else` sites).
    pub fn err_str(message: impl Into<String>) -> String {
        message.into()
    }
}

/// Re-export of the compiler's fusion IR (`Expr`, `ReduceOp`, the
/// reference interpreters) that the Metal emitters and runners consume.
pub mod fusion {
    pub use effect_torch_compiler::*;
}

/// Re-exports arranged to match the cross-runtime module layout.
pub mod runtime {
    /// Tensor element types supported across runtimes.
    pub mod dtype {
        pub use effect_torch_runtime::DType;
    }

    /// Shape/stride/offset tensor layouts.
    pub mod layout {
        pub use effect_torch_runtime::Layout;
    }

    /// CPU runtime, available to tests for numerical parity checks.
    #[cfg(test)]
    pub mod cpu {
        pub use effect_torch_runtime_cpu::*;
    }

    /// The Metal backend module tree.
    pub mod metal {
        #[cfg(test)]
        pub use crate::composed;
        pub use crate::{
            conv, device, emit, flash, gemm, indexing, kda, kernels, layer_norm, linear, loss, ops,
            paged, quantized, rotary, run, sampling, shortconv,
        };
    }
}

/// Cross-entropy reduction mode, re-exported for the loss kernels.
pub use effect_torch_graph::CrossEntropyReduction as CeReduction;

use effect_torch_runtime::{Placement, RuntimeIdentity};
use std::any::Any;
use std::sync::OnceLock;

pub mod conv;
pub mod device;
pub mod emit;
pub(crate) mod executable;
pub mod gemm;
pub mod indexing;
pub mod kernels;
#[cfg(feature = "napi-addon")]
pub mod napi;
pub mod run;

#[cfg(test)]
pub mod composed;
pub mod ops;

pub mod flash;
pub mod kda;
pub mod layer_norm;
pub mod linear;
pub mod loss;
pub mod paged;
pub mod quantized;
pub mod rotary;
pub mod sampling;
pub mod shortconv;
pub(crate) mod value;
mod workspace;

/// Runtime identity singleton: this backend registers as `"metal"`.
fn identity() -> &'static RuntimeIdentity {
    static IDENTITY: OnceLock<RuntimeIdentity> = OnceLock::new();
    IDENTITY.get_or_init(|| RuntimeIdentity::new("metal"))
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
        self.buffer.placement()
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

//! Metal backend. Everything Metal lives here:
//!
//! - `device` — device singleton, pool allocator, encoder manager,
//!   pipeline cache.
//! - `emit` — first-party IR → MSL emitter (SSA form).
//! - `run` — `MetalTensor` and the fused elementwise/reduce runners.
//! - `kernels`, `indexing`, `conv`, `gemm` — primitive kernels
//!   (creation/cast/copy/random/reductions, gather/scatter/select/cat,
//!   conv family, tiled matmul).
//! - `ops` — dispatch helpers used by executable commands for ordinary
//!   operations (binary/unary/compare/cast/matmul/contiguous/views).
//! - `composed` (tests only) — primitive-built numerical references for
//!   destination-kernel parity tests.
//! - `flash`, `loss`, `layer_norm`, `rotary`, `paged`, `linear`, `quantized` —
//!   semantic fused kernels.
//!
//! # Runtime invariants
//!
//! - **Device/queue ownership.** One process-wide [`device::MetalDevice`]
//!   singleton owns the `MTLDevice` and a single `MTLCommandQueue`. All
//!   encoding goes through per-stream submission contexts; the queue is
//!   shared and thread-safe at the Metal API boundary.
//! - **Asynchronous execution.** Kernels are encoded ahead of GPU
//!   execution. Host reads of buffer contents are only defined after a
//!   `synchronize`/`synchronize_buffer` that drains the producing command
//!   stream; the pool allocator never recycles a buffer while a command
//!   buffer can still reference it.
//! - **Precompilation contract.** Every `*_into` destination API requires
//!   its exact pipeline to be warm: call the matching `compile_*`/`warm_*`
//!   (or let the allocating wrapper do it) before dispatch. During
//!   executable dispatch, allocation and compilation are hard errors.
//! - **Memory policy.** `EFFECT_TORCH_MEMORY_CAP_MB` sets a hard live-byte
//!   ceiling (panic); `EFFECT_TORCH_MEMORY_BUDGET_MB` sets the soft
//!   backpressure budget; `EFFECT_TORCH_PRIVATE_INTERMEDIATES`,
//!   `EFFECT_TORCH_NO_MMA`, `EFFECT_TORCH_METAL_PROFILE`, and
//!   `EFFECT_TORCH_SYNC_TRACE` toggle storage mode, MMA selection,
//!   profiling, and sync tracing. All are snapshotted once.
//! - **Dtype/layout.** The fusion emitter computes in f32 and stores
//!   f32/bf16; primitive kernels cover the full `DType` set except f64
//!   (unsupported on Metal). Destinations must be contiguous; inputs may
//!   carry arbitrary strides, which are baked into the emitted source.

#![cfg(target_os = "macos")]
#![cfg_attr(not(feature = "napi-addon"), allow(dead_code))]

/// Error plumbing shared by every Metal module: stringly-typed `Res<T>`
/// plus the `err`/`err_str` constructors used on hot paths.
pub mod err {
    /// The Metal runtime's uniform result type: errors are human-readable
    /// strings (validation failures, compile errors, GPU command buffer
    /// failures) with no error taxonomy.
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

/// Runtime-facing facade mirroring the cross-runtime module layout:
/// dtype/layout re-exports, the CPU runtime (tests only, for parity
/// references), and the Metal module tree itself.
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

use effect_torch_runtime::{DeviceId, Placement, RuntimeIdentity};
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

/// Placement singleton: all Metal tensors live on `metal:0` in the
/// `"shared"` memory space (unified memory; storage mode may still be
/// private for pure intermediates).
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

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

#![cfg(target_os = "macos")]
#![cfg_attr(not(feature = "napi-addon"), allow(dead_code))]

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
        #[cfg(test)]
        pub use crate::composed;
        pub use crate::{
            conv, device, emit, flash, gemm, indexing, kda, kernels, layer_norm, linear, loss, ops,
            paged, quantized, rotary, run, shortconv,
        };
    }
}

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
pub mod shortconv;
pub(crate) mod value;
mod workspace;

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

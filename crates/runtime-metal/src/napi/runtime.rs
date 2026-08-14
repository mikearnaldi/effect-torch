//! Compatibility namespace for backend-neutral runtime types.
//!
//! The NAPI module historically refers to `runtime::{dtype, layout, metal}`.
//! These re-exports keep that code explicit without introducing wrapper types
//! or duplicating the runtime's dtype/layout definitions.

/// Backend-neutral dtype definitions.
pub mod dtype {
    pub use effect_torch_runtime::DType;
}

/// Backend-neutral tensor layouts.
pub mod layout {
    pub use effect_torch_runtime::Layout;
}

/// Metal device and tensor implementation used by the addon.
pub mod metal {
    pub use crate::{device, run};
}

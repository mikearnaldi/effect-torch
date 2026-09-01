//! CUDA runtime backed by the CUDA driver API and NVRTC.
//!
//! The runtime compiles graph programs through the shared compiler driver and
//! dispatches CUDA kernels through a device-local NVRTC module.

mod buffer;
mod device;
mod executable;
mod lowering;
mod value;
mod workspace;

#[cfg(feature = "napi-addon")]
#[cfg_attr(test, allow(dead_code))]
mod napi;

pub use device::CudaDevice;
pub use executable::{
    compile, compile_stateful, compile_stateful_with_options, compile_with_options, CudaExecutable,
    CudaSequenceState, CudaStateInvocation,
};
pub use value::CudaValue;

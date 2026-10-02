//! CUDA runtime backed by the CUDA driver API and NVRTC.
//!
//! The runtime compiles graph programs through the shared compiler driver and
//! dispatches CUDA kernels through a device-local NVRTC module.

mod attention75;
mod buffer;
mod capabilities;
mod cublas;
mod device;
mod div_feedback;
mod emit;
mod executable;
mod expert_device;
#[cfg(test)]
mod expert_device_tests;
mod explicit_graph61;
mod fused_moe75;
mod kv_matmul;
mod kv_pair;
#[cfg(test)]
mod kv_pair_tests;
mod lowering;
mod norm98_pair;
mod normrope101_admission;
mod planned_overlap;
mod triton_norm98;
mod triton_normrope101;
mod triton_softmax100;
mod typed_binary;
mod value;
mod vnorm_store;
mod workspace;

#[cfg(test)]
mod cublas_tests;
#[cfg(test)]
mod grouped_expert_tests;
#[cfg(test)]
mod grouped_rows_tests;
#[cfg(test)]
mod planned_overlap_tests;
#[cfg(test)]
mod rms_tests;
#[cfg(test)]
mod rotary_tests;
#[cfg(test)]
mod sum_tests;

#[cfg(feature = "napi-addon")]
#[cfg_attr(test, allow(dead_code))]
mod napi;

pub use device::CudaDevice;
pub use executable::{
    compile, compile_stateful, compile_stateful_with_layout, compile_stateful_with_options,
    compile_with_options, CudaExecutable, CudaKvSnapshot, CudaSequenceState, CudaStateInvocation,
    CudaStateLayout,
};
pub use value::CudaValue;

mod entropy81;
mod rng_arg80;
mod sampler83;

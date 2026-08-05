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

pub mod device;
pub mod arena;
pub mod emit;
pub mod conv;
pub mod gemm;
pub mod indexing;
pub mod kernels;
pub mod run;

pub mod ops;
pub mod composed;

pub mod flash;
pub mod layer_norm;
pub mod linear;
pub mod loss;
pub mod paged;
pub mod rotary;

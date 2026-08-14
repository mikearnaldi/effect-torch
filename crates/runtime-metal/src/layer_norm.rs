//! Fused layer normalization on Metal: one kernel per direction over
//! per-row threadgroups (two in-kernel passes — mean, then variance —
//! one launch each way). The backward also emits x̂ so dw/db are two
//! plain reduce ops host-side instead of another kernel family. CPU
//! keeps the composed path in lib.rs.
//!
//! ## Kernel contracts
//!
//! - `et_ln_fwd`: one threadgroup of `NT = 1024` threads per row;
//!   mean and variance are reduced with `simd_sum` plus a
//!   threadgroup-memory tree across the `NT / 32` simdgroups. Input
//!   elements load to f32 for the statistics; storage may be
//!   f32/f16/bf16.
//! - `et_rms_fwd` / `et_rms_fwd_registers`: RMS norm over the last
//!   dim with an optional weight. The register variant keeps up to 8
//!   elements per thread in registers and is selected when
//!   `normalized_elements <= NT * 8`; its thread count is clamped to
//!   `[32, NT]` and rounded to a full simdgroup (32 threads).
//! - `et_ln_bwd`: recomputes mean/rstd, then
//!   `dx = (g·w − mean(g·w) − x̂·mean(g·w·x̂)) · rstd`, also storing
//!   `x̂` so the caller derives `dw`/`db` with plain reductions.
//!
//! The `*_into` entry points validate contiguity, shape, dtype, and
//! storage bounds, mark destination buffers written, and allocate
//! nothing; the allocating wrappers (`ln_forward`, `ln_backward`) are
//! for use outside planned executables.

use crate::runtime::metal::run::MetalTensor;

#[cfg(test)]
thread_local! {
    static TEST_DEVICE_ALLOCATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TEST_PIPELINE_REQUESTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn record_device_allocation() {
    TEST_DEVICE_ALLOCATIONS.with(|count| count.set(count.get() + 1));
}

#[cfg(test)]
fn record_pipeline_request() {
    TEST_PIPELINE_REQUESTS.with(|count| count.set(count.get() + 1));
}

#[cfg(test)]
fn reset_test_counts() {
    TEST_DEVICE_ALLOCATIONS.with(|count| count.set(0));
    TEST_PIPELINE_REQUESTS.with(|count| count.set(0));
}

#[cfg(test)]
fn test_counts() -> (usize, usize) {
    (
        TEST_DEVICE_ALLOCATIONS.with(std::cell::Cell::get),
        TEST_PIPELINE_REQUESTS.with(std::cell::Cell::get),
    )
}

/// Planner-facing description of one device buffer a launch needs
/// (shape, dtype, and derived element/byte counts, overflow-checked).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferRequirement {
    /// Logical shape of the buffer.
    pub shape: Vec<usize>,
    /// Element dtype.
    pub dtype: crate::runtime::dtype::DType,
    /// Total element count (product of `shape`).
    pub elements: usize,
    /// Total byte count (`elements.max(1) * dtype.size_in_bytes()`).
    pub bytes: usize,
}

impl BufferRequirement {
    fn new(
        shape: Vec<usize>,
        dtype: crate::runtime::dtype::DType,
        label: &str,
    ) -> crate::err::Res<Self> {
        let elements = shape
            .iter()
            .try_fold(1usize, |total, dimension| total.checked_mul(*dimension));
        let elements =
            elements.ok_or_else(|| format!("layer_norm: {label} element count overflow"))?;
        let bytes = elements
            .max(1)
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| format!("layer_norm: {label} byte size overflow"))?;
        Ok(Self {
            shape,
            dtype,
            elements,
            bytes,
        })
    }
}

/// Thread topology selected for a launch. Currently always row-parallel:
/// one threadgroup per normalized row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerNormTopology {
    /// One threadgroup per row, `threads` threads each, `dispatches`
    /// total kernel launches.
    Rows { threads: usize, dispatches: usize },
}

/// Planner-facing requirements of a forward (layer-norm or RMS-norm)
/// launch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerNormForwardRequirements {
    /// Output buffer (same shape/dtype as the input).
    pub output: BufferRequirement,
    /// Selected thread topology.
    pub topology: LayerNormTopology,
    /// Number of normalized rows (`numel(x) / normalized_elements`).
    pub rows: usize,
    /// Elements per normalized row.
    pub normalized_elements: usize,
    /// Storage dtype (f32/f16/bf16).
    pub dtype: crate::runtime::dtype::DType,
}

/// Planner-facing requirements of a backward launch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerNormBackwardRequirements {
    /// `dx` output buffer (same shape/dtype as the input).
    pub dx: BufferRequirement,
    /// `x̂` (normalized) output buffer, from which the caller derives
    /// `dw`/`db` via plain reductions.
    pub normalized: BufferRequirement,
    /// Selected thread topology.
    pub topology: LayerNormTopology,
    /// Number of normalized rows.
    pub rows: usize,
    /// Elements per normalized row.
    pub normalized_elements: usize,
    /// Storage dtype (f32/f16/bf16).
    pub dtype: crate::runtime::dtype::DType,
}

/// Whether the fused layer-norm path can run: Metal, f32, last-dim norm.
pub fn is_supported(x: &MetalTensor, weight: &MetalTensor) -> bool {
    matches!(
        x.dtype,
        crate::runtime::dtype::DType::F32
            | crate::runtime::dtype::DType::F16
            | crate::runtime::dtype::DType::BF16
    ) && weight.dtype == x.dtype
}

#[cfg(target_os = "macos")]
pub use metal::{
    ln_backward, ln_backward_into, ln_backward_requirements, ln_forward, ln_forward_into,
    ln_forward_requirements, rms_forward_into, rms_forward_requirements, warm_backward,
    warm_backward_exact, warm_forward, warm_forward_exact, warm_rms_exact,
};

#[cfg(target_os = "macos")]
mod metal {
    use super::{
        BufferRequirement, LayerNormBackwardRequirements, LayerNormForwardRequirements,
        LayerNormTopology,
    };
    use crate::runtime::metal::device::{set_buffer, set_bytes, MetalDevice, Pipeline};
    use crate::runtime::metal::run::MetalTensor;

    use objc2_metal::MTLComputeCommandEncoder;
    use std::sync::Arc;

    const NT: usize = 1024;

    fn checked_numel(shape: &[usize], label: &str) -> crate::err::Res<usize> {
        shape
            .iter()
            .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
            .ok_or_else(|| format!("layer_norm: {label} element count overflow"))
    }

    fn geometry(
        x_shape: &[usize],
        x_dtype: crate::runtime::dtype::DType,
        weight_shape: &[usize],
        weight_dtype: crate::runtime::dtype::DType,
        bias_shape: Option<&[usize]>,
        bias_dtype: Option<crate::runtime::dtype::DType>,
    ) -> crate::err::Res<(usize, usize)> {
        if !matches!(
            x_dtype,
            crate::runtime::dtype::DType::F32
                | crate::runtime::dtype::DType::F16
                | crate::runtime::dtype::DType::BF16
        ) {
            return Err(format!("layer_norm: unsupported dtype {x_dtype:?}"));
        }
        if weight_dtype != x_dtype || bias_dtype.is_some_and(|dtype| dtype != x_dtype) {
            return Err("layer_norm: x/weight/bias dtypes must match".to_string());
        }
        if weight_shape.is_empty() || weight_shape.len() > x_shape.len() {
            return Err("layer_norm: normalized shape must be a non-empty suffix of x".to_string());
        }
        if x_shape[x_shape.len() - weight_shape.len()..] != *weight_shape {
            return Err("layer_norm: weight shape must match the normalized suffix".to_string());
        }
        if bias_shape.is_some_and(|shape| shape != weight_shape) {
            return Err("layer_norm: bias shape must match weight shape".to_string());
        }
        let normalized_elements = checked_numel(weight_shape, "normalized shape")?;
        let x_elements = checked_numel(x_shape, "input")?;
        if normalized_elements == 0 || x_elements == 0 {
            return Err("layer_norm: zero-sized dimensions are unsupported".to_string());
        }
        let rows = x_elements / normalized_elements;
        if rows > u32::MAX as usize || normalized_elements > u32::MAX as usize {
            return Err("layer_norm: rows and normalized elements must fit u32".to_string());
        }
        Ok((rows, normalized_elements))
    }

    /// Plans a layer-norm forward: validates the geometry (weight/bias
    /// shapes must equal the normalized suffix of `x`; dtypes must
    /// match) and returns the exact requirements.
    pub fn ln_forward_requirements(
        x_shape: &[usize],
        x_dtype: crate::runtime::dtype::DType,
        weight_shape: &[usize],
        weight_dtype: crate::runtime::dtype::DType,
        bias_shape: &[usize],
        bias_dtype: crate::runtime::dtype::DType,
    ) -> crate::err::Res<LayerNormForwardRequirements> {
        let (rows, normalized_elements) = geometry(
            x_shape,
            x_dtype,
            weight_shape,
            weight_dtype,
            Some(bias_shape),
            Some(bias_dtype),
        )?;
        Ok(LayerNormForwardRequirements {
            output: BufferRequirement::new(x_shape.to_vec(), x_dtype, "output")?,
            topology: LayerNormTopology::Rows {
                threads: NT,
                dispatches: 1,
            },
            rows,
            normalized_elements,
            dtype: x_dtype,
        })
    }

    /// Plans a layer-norm backward: like the forward, plus the
    /// cotangent must match `x` exactly in shape and dtype.
    pub fn ln_backward_requirements(
        x_shape: &[usize],
        x_dtype: crate::runtime::dtype::DType,
        weight_shape: &[usize],
        weight_dtype: crate::runtime::dtype::DType,
        gradient_shape: &[usize],
        gradient_dtype: crate::runtime::dtype::DType,
    ) -> crate::err::Res<LayerNormBackwardRequirements> {
        let (rows, normalized_elements) =
            geometry(x_shape, x_dtype, weight_shape, weight_dtype, None, None)?;
        if gradient_shape != x_shape || gradient_dtype != x_dtype {
            return Err("layer_norm: gradient must match x shape and dtype".to_string());
        }
        Ok(LayerNormBackwardRequirements {
            dx: BufferRequirement::new(x_shape.to_vec(), x_dtype, "dx")?,
            normalized: BufferRequirement::new(x_shape.to_vec(), x_dtype, "normalized")?,
            topology: LayerNormTopology::Rows {
                threads: NT,
                dispatches: 1,
            },
            rows,
            normalized_elements,
            dtype: x_dtype,
        })
    }

    fn rms_threads(normalized_elements: usize) -> usize {
        normalized_elements.clamp(32, NT).next_multiple_of(32)
    }

    /// Plans an RMS-norm forward over the last dim with an optional
    /// weight; selects the register-caching kernel for small rows.
    pub fn rms_forward_requirements(
        x_shape: &[usize],
        x_dtype: crate::runtime::dtype::DType,
        weight: Option<(&[usize], crate::runtime::dtype::DType)>,
    ) -> crate::err::Res<LayerNormForwardRequirements> {
        if !matches!(
            x_dtype,
            crate::runtime::dtype::DType::F32
                | crate::runtime::dtype::DType::F16
                | crate::runtime::dtype::DType::BF16
        ) || x_shape.is_empty()
        {
            return Err(format!(
                "rms_norm: input must have rank at least 1 and a floating dtype, got {x_shape:?}:{x_dtype:?}"
            ));
        }
        let normalized_elements = *x_shape.last().expect("validated input rank");
        let elements = checked_numel(x_shape, "rms_norm input")?;
        if normalized_elements == 0 || elements == 0 {
            return Err("rms_norm: zero-sized dimensions are unsupported".to_string());
        }
        if weight.is_some_and(|(shape, dtype)| shape != [normalized_elements] || dtype != x_dtype) {
            return Err("rms_norm: weight must match the last dimension and dtype".to_string());
        }
        let rows = elements / normalized_elements;
        if rows > u32::MAX as usize || normalized_elements > u32::MAX as usize {
            return Err("rms_norm: rows and normalized elements must fit u32".to_string());
        }
        Ok(LayerNormForwardRequirements {
            output: BufferRequirement::new(x_shape.to_vec(), x_dtype, "rms output")?,
            topology: LayerNormTopology::Rows {
                threads: rms_threads(normalized_elements),
                dispatches: 1,
            },
            rows,
            normalized_elements,
            dtype: x_dtype,
        })
    }

    fn require_contiguous(tensor: &MetalTensor, label: &str) -> crate::err::Res<()> {
        if !tensor.layout.is_contiguous() {
            return Err(format!("layer_norm: {label} must be contiguous"));
        }
        let end = tensor
            .layout
            .offset()
            .checked_add(tensor.numel())
            .and_then(|elements| elements.checked_mul(tensor.dtype.size_in_bytes()))
            .ok_or_else(|| format!("layer_norm: {label} storage range overflow"))?;
        if end > tensor.buffer.size {
            return Err(format!("layer_norm: {label} storage is too small"));
        }
        Ok(())
    }

    fn require_exact(
        tensor: &MetalTensor,
        shape: &[usize],
        dtype: crate::runtime::dtype::DType,
        label: &str,
    ) -> crate::err::Res<()> {
        if tensor.layout.shape() != shape || tensor.dtype != dtype {
            return Err(format!(
                "layer_norm: {label} must be {shape:?}:{dtype:?}, got {:?}:{:?}",
                tensor.layout.shape(),
                tensor.dtype
            ));
        }
        require_contiguous(tensor, label)
    }

    fn wrap_contig(t: &MetalTensor) -> crate::err::Res<MetalTensor> {
        if t.layout.is_contiguous() {
            Ok(t.clone())
        } else {
            crate::runtime::metal::kernels::strided_copy(MetalDevice::get(), t)
        }
    }

    fn alloc_t(
        n: usize,
        dtype: crate::runtime::dtype::DType,
    ) -> Arc<crate::runtime::metal::device::Buffer> {
        #[cfg(test)]
        super::record_device_allocation();
        MetalDevice::get().alloc(n.max(1), dtype)
    }

    fn source(ty: &str, threads: usize) -> String {
        r#"
#include <metal_stdlib>
using namespace metal;

#define NT $NT
#define STOR {ty}
#define LD(x) float(x)
#define ST(p, v) ((p) = STOR(v))

// Per-row layer norm: mean and variance in two in-kernel passes, one
// launch. One threadgroup per row.
kernel void et_ln_fwd(
    device const STOR* X [[buffer(0)]],
    device const STOR* W [[buffer(1)]],
    device const STOR* B [[buffer(2)]],
    device STOR* O [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    const uint row = tgid;
    device const STOR* x = X + (ulong)row * D;
    device STOR* o = O + (ulong)row * D;
    float s = 0.0f;
    for (uint j = tid; j < D; j += NT) { s += LD(x[j]); }
    s = simd_sum(s);
    threadgroup float ps[NT / 32];
    const uint lane = tid % 32;
    const uint grp = tid / 32;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { ps[grp] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = 0.0f;
    for (uint g = 0; g < NT / 32; g++) { mean += ps[g]; }
    mean /= float(D);
    float q = 0.0f;
    for (uint j = tid; j < D; j += NT) { const float c = LD(x[j]) - mean; q += c * c; }
    q = simd_sum(q);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { ps[grp] = q; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0.0f;
    for (uint g = 0; g < NT / 32; g++) { var += ps[g]; }
    var /= float(D);
    const float rstd = rsqrt(var + eps);
    for (uint j = tid; j < D; j += NT) {
        ST(o[j], (LD(x[j]) - mean) * rstd * LD(W[j]) + LD(B[j]));
    }
}

kernel void et_rms_fwd(
    device const STOR* X [[buffer(0)]],
    device const STOR* W [[buffer(1)]],
    device STOR* O [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant uint& has_weight [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    device const STOR* x = X + (ulong)row * D;
    device STOR* o = O + (ulong)row * D;
    float sum_square = 0.0f;
    for (uint j = tid; j < D; j += NT) {
        const float value = LD(x[j]);
        sum_square += value * value;
    }
    sum_square = simd_sum(sum_square);
    threadgroup float partial[NT / 32];
    const uint lane = tid % 32;
    const uint group = tid / 32;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { partial[group] = sum_square; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < NT / 32; ++i) { total += partial[i]; }
    const float scale = rsqrt(total / float(D) + eps);
    for (uint j = tid; j < D; j += NT) {
        ST(o[j], LD(x[j]) * scale * (has_weight != 0 ? LD(W[j]) : 1.0f));
    }
}

kernel void et_rms_fwd_registers(
    device const STOR* X [[buffer(0)]],
    device const STOR* W [[buffer(1)]],
    device STOR* O [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant uint& has_weight [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    device const STOR* x = X + (ulong)row * D;
    device STOR* o = O + (ulong)row * D;
    float values[8];
    float sum_square = 0.0f;
#pragma unroll
    for (uint index = 0; index < 8; ++index) {
        const uint j = tid + index * NT;
        if (j < D) {
            const float value = LD(x[j]);
            values[index] = value;
            sum_square += value * value;
        }
    }
    sum_square = simd_sum(sum_square);
    threadgroup float partial[NT / 32];
    const uint lane = tid % 32;
    const uint group = tid / 32;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { partial[group] = sum_square; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < NT / 32; ++i) { total += partial[i]; }
    const float scale = rsqrt(total / float(D) + eps);
#pragma unroll
    for (uint index = 0; index < 8; ++index) {
        const uint j = tid + index * NT;
        if (j < D) {
            ST(o[j], values[index] * scale * (has_weight != 0 ? LD(W[j]) : 1.0f));
        }
    }
}

// Backward: dx plus x̂ (dw/db are plain reduces host-side).
kernel void et_ln_bwd(
    device const STOR* X [[buffer(0)]],
    device const STOR* W [[buffer(1)]],
    device const STOR* G [[buffer(2)]],
    device STOR* DX [[buffer(3)]],
    device STOR* XH [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    const uint row = tgid;
    device const STOR* x = X + (ulong)row * D;
    device const STOR* g = G + (ulong)row * D;
    device STOR* dx = DX + (ulong)row * D;
    device STOR* xh = XH + (ulong)row * D;
    float s = 0.0f;
    for (uint j = tid; j < D; j += NT) { s += LD(x[j]); }
    s = simd_sum(s);
    threadgroup float ps[NT / 32];
    const uint lane = tid % 32;
    const uint grp = tid / 32;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { ps[grp] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = 0.0f;
    for (uint g = 0; g < NT / 32; g++) { mean += ps[g]; }
    mean /= float(D);
    float q = 0.0f;
    for (uint j = tid; j < D; j += NT) { const float c = LD(x[j]) - mean; q += c * c; }
    q = simd_sum(q);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { ps[grp] = q; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0.0f;
    for (uint g = 0; g < NT / 32; g++) { var += ps[g]; }
    var /= float(D);
    const float rstd = rsqrt(var + eps);
    // dx = (dyw - mean(dyw) - x̂·mean(dyw·x̂)) · rstd
    float s1 = 0.0f;
    float s2 = 0.0f;
    for (uint j = tid; j < D; j += NT) {
        const float dyw = LD(g[j]) * LD(W[j]);
        const float hat = (LD(x[j]) - mean) * rstd;
        s1 += dyw;
        s2 += dyw * hat;
    }
    s1 = simd_sum(s1);
    s2 = simd_sum(s2);
    threadgroup float ps1[NT / 32];
    threadgroup float ps2[NT / 32];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { ps1[grp] = s1; ps2[grp] = s2; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m1 = 0.0f;
    float m2 = 0.0f;
    for (uint g = 0; g < NT / 32; g++) { m1 += ps1[g]; m2 += ps2[g]; }
    m1 /= float(D);
    m2 /= float(D);
    for (uint j = tid; j < D; j += NT) {
        const float hat = (LD(x[j]) - mean) * rstd;
        ST(xh[j], hat);
        ST(dx[j], (LD(g[j]) * LD(W[j]) - m1 - hat * m2) * rstd);
    }
}
"#
        .replace("{ty}", ty)
        .replace("$NT", &threads.to_string())
    }

    fn pipeline_key(
        name: &'static str,
        dtype: crate::runtime::dtype::DType,
        threads: usize,
    ) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        (name, dtype, threads).hash(&mut hasher);
        hasher.finish()
    }

    fn pipeline(
        name: &'static str,
        dtype: crate::runtime::dtype::DType,
        threads: usize,
    ) -> crate::err::Res<Pipeline> {
        #[cfg(test)]
        super::record_pipeline_request();
        let ty = match dtype {
            crate::runtime::dtype::DType::F32 => "float",
            crate::runtime::dtype::DType::F16 => "half",
            crate::runtime::dtype::DType::BF16 => "bfloat",
            other => return Err(format!("layer_norm: unsupported dtype {other:?}")),
        };
        MetalDevice::get().compile_lazy(pipeline_key(name, dtype, threads), name, || {
            source(ty, threads)
        })
    }

    fn cached_pipeline(
        name: &'static str,
        dtype: crate::runtime::dtype::DType,
        threads: usize,
    ) -> crate::err::Res<Pipeline> {
        MetalDevice::get()
            .pipeline_cached(pipeline_key(name, dtype, threads))
            .ok_or_else(|| {
                format!("layer_norm: {name} pipeline is not warm; call the exact warm function")
            })
    }

    /// Warms the forward layer-norm pipeline for `dtype`.
    pub fn warm_forward(dtype: crate::runtime::dtype::DType) -> crate::err::Res<()> {
        pipeline("et_ln_fwd", dtype, NT)?;
        Ok(())
    }

    /// Warms exactly the pipeline described by `requirements`.
    pub fn warm_forward_exact(requirements: &LayerNormForwardRequirements) -> crate::err::Res<()> {
        warm_forward(requirements.dtype)
    }

    fn rms_kernel_name(normalized_elements: usize) -> &'static str {
        if normalized_elements <= NT * 8 {
            "et_rms_fwd_registers"
        } else {
            "et_rms_fwd"
        }
    }

    /// Warms exactly the RMS pipeline (kernel variant and thread count)
    /// described by `requirements`.
    pub fn warm_rms_exact(requirements: &LayerNormForwardRequirements) -> crate::err::Res<()> {
        pipeline(
            rms_kernel_name(requirements.normalized_elements),
            requirements.dtype,
            rms_threads(requirements.normalized_elements),
        )?;
        Ok(())
    }

    /// Warms the backward layer-norm pipeline for `dtype`.
    pub fn warm_backward(dtype: crate::runtime::dtype::DType) -> crate::err::Res<()> {
        pipeline("et_ln_bwd", dtype, NT)?;
        Ok(())
    }

    /// Warms exactly the pipeline described by `requirements`.
    pub fn warm_backward_exact(
        requirements: &LayerNormBackwardRequirements,
    ) -> crate::err::Res<()> {
        warm_backward(requirements.dtype)
    }

    fn wrap(
        buf: Arc<crate::runtime::metal::device::Buffer>,
        shape: Vec<usize>,
        dtype: crate::runtime::dtype::DType,
    ) -> crate::err::Res<MetalTensor> {
        Ok(MetalTensor {
            buffer: buf,
            layout: crate::runtime::layout::Layout::contiguous(shape),
            dtype,
        })
    }

    /// Non-allocating layer-norm forward dispatch. All tensors must be
    /// contiguous; `output` must match `x` in shape and dtype. Requires
    /// the forward pipeline to be warm.
    pub fn ln_forward_into(
        x: &MetalTensor,
        weight: &MetalTensor,
        bias: &MetalTensor,
        eps: f64,
        output: &MetalTensor,
    ) -> crate::err::Res<()> {
        let (rows, d) = geometry(
            x.layout.shape(),
            x.dtype,
            weight.layout.shape(),
            weight.dtype,
            Some(bias.layout.shape()),
            Some(bias.dtype),
        )?;
        for (tensor, label) in [(x, "x"), (weight, "weight"), (bias, "bias")] {
            require_contiguous(tensor, label)?;
        }
        require_exact(output, x.layout.shape(), x.dtype, "output")?;
        MetalDevice::get().mark_buffer_write(&output.buffer)?;
        let pipe = cached_pipeline("et_ln_fwd", x.dtype, NT)?;
        let sz = x.dtype.size_in_bytes();
        MetalDevice::get().with_encoder(|e| {
            e.setComputePipelineState(pipe.as_raw());
            set_buffer(e, 0, &x.buffer, x.layout.offset() * sz);
            set_buffer(e, 1, &weight.buffer, weight.layout.offset() * sz);
            set_buffer(e, 2, &bias.buffer, bias.layout.offset() * sz);
            set_buffer(e, 3, &output.buffer, output.layout.offset() * sz);
            set_bytes(e, 4, &(d as u32));
            set_bytes(e, 5, &(eps as f32));
            e.dispatchThreadgroups_threadsPerThreadgroup(
                objc2_metal::MTLSize {
                    width: rows,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: NT,
                    height: 1,
                    depth: 1,
                },
            );
        });
        Ok(())
    }

    /// Non-allocating RMS-norm forward dispatch. Recomputes the exact
    /// plan from the arguments and rejects the call when it differs
    /// from the immutable `requirements` the pipeline was warmed for.
    pub fn rms_forward_into(
        x: &MetalTensor,
        weight: Option<&MetalTensor>,
        eps: f64,
        output: &MetalTensor,
        requirements: &LayerNormForwardRequirements,
    ) -> crate::err::Res<()> {
        let exact = rms_forward_requirements(
            x.layout.shape(),
            x.dtype,
            weight.map(|weight| (weight.layout.shape(), weight.dtype)),
        )?;
        if &exact != requirements {
            return Err("rms_norm: execution does not match the immutable plan".to_string());
        }
        require_contiguous(x, "x")?;
        if let Some(weight) = weight {
            require_contiguous(weight, "weight")?;
        }
        require_exact(output, x.layout.shape(), x.dtype, "output")?;
        MetalDevice::get().mark_buffer_write(&output.buffer)?;
        let threads = rms_threads(requirements.normalized_elements);
        let pipeline = cached_pipeline(
            rms_kernel_name(requirements.normalized_elements),
            x.dtype,
            threads,
        )?;
        let size = x.dtype.size_in_bytes();
        let has_weight = u32::from(weight.is_some());
        let weight = weight.unwrap_or(output);
        let d = requirements.normalized_elements as u32;
        MetalDevice::get().with_encoder(|encoder| {
            encoder.setComputePipelineState(pipeline.as_raw());
            set_buffer(encoder, 0, &x.buffer, x.layout.offset() * size);
            set_buffer(encoder, 1, &weight.buffer, weight.layout.offset() * size);
            set_buffer(encoder, 2, &output.buffer, output.layout.offset() * size);
            set_bytes(encoder, 3, &d);
            set_bytes(encoder, 4, &(eps as f32));
            set_bytes(encoder, 5, &has_weight);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MetalDevice::grid(requirements.rows, 1, 1),
                MetalDevice::grid(threads, 1, 1),
            );
        });
        Ok(())
    }

    /// Allocating convenience wrapper around [`ln_forward_into`].
    pub fn ln_forward(
        x: &MetalTensor,
        weight: &MetalTensor,
        bias: &MetalTensor,
        eps: f64,
    ) -> crate::err::Res<MetalTensor> {
        let requirements = ln_forward_requirements(
            x.layout.shape(),
            x.dtype,
            weight.layout.shape(),
            weight.dtype,
            bias.layout.shape(),
            bias.dtype,
        )?;
        warm_forward_exact(&requirements)?;
        let x = wrap_contig(x)?;
        let weight = wrap_contig(weight)?;
        let bias = wrap_contig(bias)?;
        let output = wrap(
            alloc_t(requirements.output.elements, requirements.output.dtype),
            requirements.output.shape.clone(),
            requirements.output.dtype,
        )?;
        ln_forward_into(&x, &weight, &bias, eps, &output)?;
        Ok(output)
    }

    /// Non-allocating layer-norm backward dispatch: writes `dx` and the
    /// normalized activations `x̂` (both same shape/dtype as `x`).
    /// Requires the backward pipeline to be warm.
    pub fn ln_backward_into(
        x: &MetalTensor,
        weight: &MetalTensor,
        g: &MetalTensor,
        eps: f64,
        dx: &MetalTensor,
        normalized: &MetalTensor,
    ) -> crate::err::Res<()> {
        let (rows, d) = geometry(
            x.layout.shape(),
            x.dtype,
            weight.layout.shape(),
            weight.dtype,
            None,
            None,
        )?;
        if g.layout.shape() != x.layout.shape() || g.dtype != x.dtype {
            return Err("layer_norm: gradient must match x shape and dtype".to_string());
        }
        for (tensor, label) in [(x, "x"), (weight, "weight"), (g, "gradient")] {
            require_contiguous(tensor, label)?;
        }
        require_exact(dx, x.layout.shape(), x.dtype, "dx")?;
        require_exact(normalized, x.layout.shape(), x.dtype, "normalized")?;
        MetalDevice::get().mark_buffer_write(&dx.buffer)?;
        MetalDevice::get().mark_buffer_write(&normalized.buffer)?;
        let pipe = cached_pipeline("et_ln_bwd", x.dtype, NT)?;
        let sz = x.dtype.size_in_bytes();
        MetalDevice::get().with_encoder(|e| {
            e.setComputePipelineState(pipe.as_raw());
            set_buffer(e, 0, &x.buffer, x.layout.offset() * sz);
            set_buffer(e, 1, &weight.buffer, weight.layout.offset() * sz);
            set_buffer(e, 2, &g.buffer, g.layout.offset() * sz);
            set_buffer(e, 3, &dx.buffer, dx.layout.offset() * sz);
            set_buffer(e, 4, &normalized.buffer, normalized.layout.offset() * sz);
            set_bytes(e, 5, &(d as u32));
            set_bytes(e, 6, &(eps as f32));
            e.dispatchThreadgroups_threadsPerThreadgroup(
                objc2_metal::MTLSize {
                    width: rows,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: NT,
                    height: 1,
                    depth: 1,
                },
            );
        });
        Ok(())
    }

    /// Allocating convenience wrapper around [`ln_backward_into`];
    /// returns `(dx, x_hat)`. `dw`/`db` are computed by the caller from
    /// `x_hat` via plain reductions.
    pub fn ln_backward(
        x: &MetalTensor,
        weight: &MetalTensor,
        g: &MetalTensor,
        eps: f64,
    ) -> crate::err::Res<(MetalTensor, MetalTensor)> {
        let requirements = ln_backward_requirements(
            x.layout.shape(),
            x.dtype,
            weight.layout.shape(),
            weight.dtype,
            g.layout.shape(),
            g.dtype,
        )?;
        warm_backward_exact(&requirements)?;
        let x = wrap_contig(x)?;
        let weight = wrap_contig(weight)?;
        let g = wrap_contig(g)?;
        let dx = wrap(
            alloc_t(requirements.dx.elements, requirements.dx.dtype),
            requirements.dx.shape.clone(),
            requirements.dx.dtype,
        )?;
        let normalized = wrap(
            alloc_t(
                requirements.normalized.elements,
                requirements.normalized.dtype,
            ),
            requirements.normalized.shape.clone(),
            requirements.normalized.dtype,
        )?;
        ln_backward_into(&x, &weight, &g, eps, &dx, &normalized)?;
        Ok((dx, normalized))
    }
}

#[cfg(test)]
mod tests {
    use super::LayerNormTopology;
    use crate::runtime::cpu::{CpuBuffer, Tensor};
    use crate::runtime::dtype::DType;
    use crate::runtime::metal::device::MetalDevice;
    use crate::runtime::metal::run::MetalTensor;

    fn cpu_f32(tensor: &Tensor) -> &[f32] {
        let CpuBuffer::F32(values) = &tensor.buffer else {
            panic!("expected f32 tensor")
        };
        values
    }

    #[test]
    fn requirements_include_auxiliary_and_topology() {
        let forward = crate::layer_norm::ln_forward_requirements(
            &[2, 3, 4],
            DType::F16,
            &[3, 4],
            DType::F16,
            &[3, 4],
            DType::F16,
        )
        .unwrap();
        assert_eq!(forward.output.shape, [2, 3, 4]);
        assert_eq!(forward.rows, 2);
        assert_eq!(forward.normalized_elements, 12);
        assert_eq!(
            forward.topology,
            LayerNormTopology::Rows {
                threads: 1024,
                dispatches: 1,
            }
        );

        let backward = crate::layer_norm::ln_backward_requirements(
            &[2, 3, 4],
            DType::F16,
            &[3, 4],
            DType::F16,
            &[2, 3, 4],
            DType::F16,
        )
        .unwrap();
        assert_eq!(backward.dx.shape, [2, 3, 4]);
        assert_eq!(backward.normalized.shape, [2, 3, 4]);
        assert_eq!(backward.normalized.bytes, 2 * 3 * 4 * 2);
        assert_eq!(backward.topology, forward.topology);
    }

    #[test]
    fn into_matches_allocating_and_cpu_without_allocating() {
        let dev = MetalDevice::get();
        let x_values: Vec<f32> = (0..24)
            .map(|index| (index as f32 * 0.37).sin() * 2.0)
            .collect();
        let weight_values: Vec<f32> = (0..12).map(|index| 0.5 + index as f32 * 0.07).collect();
        let bias_values: Vec<f32> = (0..12).map(|index| index as f32 * 0.03 - 0.1).collect();
        let gradient_values: Vec<f32> = (0..24).map(|index| (index as f32 * 0.19).cos()).collect();
        let x = MetalTensor::from_f32(dev, x_values.clone(), vec![2, 3, 4]);
        let weight = MetalTensor::from_f32(dev, weight_values.clone(), vec![3, 4]);
        let bias = MetalTensor::from_f32(dev, bias_values.clone(), vec![3, 4]);
        let gradient = MetalTensor::from_f32(dev, gradient_values.clone(), vec![2, 3, 4]);
        let eps = 1e-5;

        let forward_req = crate::layer_norm::ln_forward_requirements(
            x.layout.shape(),
            x.dtype,
            weight.layout.shape(),
            weight.dtype,
            bias.layout.shape(),
            bias.dtype,
        )
        .unwrap();
        crate::layer_norm::warm_forward_exact(&forward_req).unwrap();
        let output = MetalTensor::empty(dev, forward_req.output.shape.clone(), DType::F32);
        super::reset_test_counts();
        crate::layer_norm::ln_forward_into(&x, &weight, &bias, eps, &output).unwrap();
        assert_eq!(super::test_counts(), (0, 0));

        let backward_req = crate::layer_norm::ln_backward_requirements(
            x.layout.shape(),
            x.dtype,
            weight.layout.shape(),
            weight.dtype,
            gradient.layout.shape(),
            gradient.dtype,
        )
        .unwrap();
        crate::layer_norm::warm_backward_exact(&backward_req).unwrap();
        let dx = MetalTensor::empty(dev, backward_req.dx.shape.clone(), DType::F32);
        let normalized = MetalTensor::empty(dev, backward_req.normalized.shape.clone(), DType::F32);
        super::reset_test_counts();
        crate::layer_norm::ln_backward_into(&x, &weight, &gradient, eps, &dx, &normalized).unwrap();
        assert_eq!(super::test_counts(), (0, 0));

        let allocated_output = crate::layer_norm::ln_forward(&x, &weight, &bias, eps).unwrap();
        let (allocated_dx, allocated_normalized) =
            crate::layer_norm::ln_backward(&x, &weight, &gradient, eps).unwrap();
        dev.synchronize().unwrap();

        let cpu_x = Tensor::from_vec(x_values, vec![2, 3, 4]);
        let cpu_weight = Tensor::from_vec(weight_values, vec![3, 4]);
        let cpu_bias = Tensor::from_vec(bias_values, vec![3, 4]);
        let cpu_gradient = Tensor::from_vec(gradient_values, vec![2, 3, 4]);
        let cpu_output =
            crate::runtime::cpu::composed::layer_norm_forward(&cpu_x, &cpu_weight, &cpu_bias, eps)
                .contiguous();
        let (cpu_dx, _, _) = crate::runtime::cpu::composed::layer_norm_backward(
            &cpu_x,
            &cpu_weight,
            &cpu_gradient,
            eps,
        );
        let output_values = output.read_f32().unwrap();
        let allocated_output = allocated_output.read_f32().unwrap();
        for ((actual, allocated), expected) in output_values
            .iter()
            .zip(&allocated_output)
            .zip(cpu_f32(&cpu_output))
        {
            assert!((actual - expected).abs() < 2e-5);
            assert!((actual - allocated).abs() < 1e-6);
        }
        let dx_values = dx.read_f32().unwrap();
        let allocated_dx = allocated_dx.read_f32().unwrap();
        let cpu_dx = cpu_dx.contiguous();
        for ((actual, allocated), expected) in
            dx_values.iter().zip(&allocated_dx).zip(cpu_f32(&cpu_dx))
        {
            assert!((actual - expected).abs() < 3e-5);
            assert!((actual - allocated).abs() < 1e-6);
        }
        assert_eq!(
            normalized.read_f32().unwrap(),
            allocated_normalized.read_f32().unwrap()
        );
    }
}

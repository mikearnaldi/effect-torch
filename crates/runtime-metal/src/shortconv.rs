//! Fused causal depthwise short convolution on Metal (RFC 0018): the
//! KDA local-mixing conv `y[t,c] = sum_j w[c,j] * x[t-K+1+j, c]` with zero
//! or carried history. One launch per call replaces the composed
//! pad/conv1d/transpose chain; the stateful ConvState decode/prefill
//! path passes the slot's [K-1, C] window as history and receives the
//! shifted window back (only the `advance` real rows shift in — chunked
//! prefill right-pads). All channel counts and kernel sizes are
//! supported; the dtype gate fails loud.
//!
//! ## Kernel contracts
//!
//! - **Forward** (`et_shortconv_fwd`): `x` is `[.., steps, C]`, `w` is
//!   `[C, K]`; taps before step 0 read the optional f32 history window
//!   `[K-1, C]`. When the window-write flag is set, the last `K-1` real
//!   rows (of the `advance` prefix) plus surviving history are written
//!   back as the shifted f32 window — this is the ConvState transaction.
//!   History/state are single-sequence only (`batch == 1`).
//! - **Backward-x** (`et_shortconv_bwd_x`): full correlation of the
//!   cotangent against the time-reversed taps, `dx[s] = sum_j
//!   w[c, K-1-j] * g[s+j, c]`, implicitly right-zero-padded.
//! - **Backward-w** (`et_shortconv_bwd_w`): `dw[c, j] = sum_{b,t}
//!   g[t, c] * x[t-K+1+j, c]` over the causal window, one thread per
//!   (channel, tap) with serial time loops.
//!
//! All three kernels accumulate in f32 regardless of storage dtype
//! (f32 or bf16) and require contiguous, offset-0-compatible views;
//! the `*_into` entry points validate shapes and mark destination
//! buffers written, but allocate nothing.

use crate::runtime::dtype::DType;

/// Planner-facing resource requirements of one forward launch
/// (stateless or ConvState-transaction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForwardRequirements {
    /// Storage dtype of input, weight, and output (f32 or bf16).
    pub dtype: DType,
    /// Bytes of the `[batch, steps, channels]` output.
    pub output_bytes: usize,
    /// Bytes of the f32 `[K-1, C]` next-window state; 0 when the call
    /// does not carry state.
    pub state_next_bytes: usize,
    /// Always 0: the kernels take no staging views.
    pub staging_bytes: usize,
    /// Always 0: the kernels report no runtime status.
    pub status_bytes: usize,
    /// Always 0: the kernels need no scratch workspace.
    pub scratch_bytes: usize,
    /// Always 1: a single fused forward kernel per launch.
    pub pipeline_count: usize,
}

/// Planner-facing resource requirements of the combined backward
/// (backward-x + backward-w, two launches).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackwardRequirements {
    /// Storage dtype of gradients and weights (f32 or bf16).
    pub dtype: DType,
    /// Bytes of the `dx` output (`[batch, steps, channels]`).
    pub dx_bytes: usize,
    /// Bytes of the `dweight` output (`[channels, kernel]`).
    pub dweight_bytes: usize,
    /// Always 0: no staging views.
    pub staging_bytes: usize,
    /// Always 0: no runtime status.
    pub status_bytes: usize,
    /// Always 0: no scratch workspace.
    pub scratch_bytes: usize,
    /// Always 2: backward-x and backward-w pipelines.
    pub pipeline_count: usize,
}

/// Planner-facing resource requirements of the backward-x launch alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackwardXRequirements {
    /// Storage dtype of cotangent and `dx` (f32 or bf16).
    pub dtype: DType,
    /// Bytes of the `dx` output (`[batch, steps, channels]`).
    pub output_bytes: usize,
    /// Always 0: no staging views.
    pub staging_bytes: usize,
    /// Always 0: no runtime status.
    pub status_bytes: usize,
    /// Always 0: no scratch workspace.
    pub scratch_bytes: usize,
    /// Always 1: a single backward-x kernel per launch.
    pub pipeline_count: usize,
}

/// Planner-facing resource requirements of the backward-w launch alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackwardWRequirements {
    /// Storage dtype of `dweight` (f32 or bf16).
    pub dtype: DType,
    /// Bytes of the `dweight` output (`[channels, kernel]`).
    pub output_bytes: usize,
    /// Always 0: no staging views.
    pub staging_bytes: usize,
    /// Always 0: no runtime status.
    pub status_bytes: usize,
    /// Always 0: no scratch workspace.
    pub scratch_bytes: usize,
    /// Always 1: a single backward-w kernel per launch.
    pub pipeline_count: usize,
}

fn checked_bytes(elements: &[usize], dtype: DType, operation: &str) -> crate::err::Res<usize> {
    elements
        .iter()
        .try_fold(1usize, |count, value| count.checked_mul(*value))
        .and_then(|count| count.checked_mul(dtype.size_in_bytes()))
        .ok_or_else(|| format!("{operation}: requirement byte size overflow"))
}

/// Requirements of a stateless forward over `batch` sequences of
/// `steps` positions with `channels` channels and kernel size `kernel`.
/// `write_state_next` additionally sizes the f32 next-window state.
pub fn forward_requirements(
    dtype: DType,
    batch: usize,
    steps: usize,
    channels: usize,
    kernel: usize,
    write_state_next: bool,
) -> crate::err::Res<ForwardRequirements> {
    if !supported_dtype(dtype) {
        return Err(format!("shortconv forward: unsupported dtype {dtype:?}"));
    }
    Ok(ForwardRequirements {
        dtype,
        output_bytes: checked_bytes(&[batch, steps, channels], dtype, "shortconv forward")?,
        state_next_bytes: if write_state_next {
            checked_bytes(
                &[kernel.saturating_sub(1), channels],
                DType::F32,
                "shortconv state",
            )?
        } else {
            0
        },
        staging_bytes: 0,
        status_bytes: 0,
        scratch_bytes: 0,
        pipeline_count: 1,
    })
}

/// Requirements of a single-sequence ConvState transaction: forward
/// plus the shifted-window state write.
pub fn state_requirements(
    dtype: DType,
    steps: usize,
    channels: usize,
    kernel: usize,
) -> crate::err::Res<ForwardRequirements> {
    forward_requirements(dtype, 1, steps, channels, kernel, true)
}

/// Requirements of the combined backward (both dx and dweight launches).
pub fn backward_requirements(
    dtype: DType,
    batch: usize,
    steps: usize,
    channels: usize,
    kernel: usize,
) -> crate::err::Res<BackwardRequirements> {
    let dx = backward_x_requirements(dtype, batch, steps, channels)?;
    let dweight = backward_w_requirements(dtype, channels, kernel)?;
    Ok(BackwardRequirements {
        dtype,
        dx_bytes: dx.output_bytes,
        dweight_bytes: dweight.output_bytes,
        staging_bytes: 0,
        status_bytes: 0,
        scratch_bytes: 0,
        pipeline_count: 2,
    })
}

/// Requirements of the backward-x launch alone.
pub fn backward_x_requirements(
    dtype: DType,
    batch: usize,
    steps: usize,
    channels: usize,
) -> crate::err::Res<BackwardXRequirements> {
    if !supported_dtype(dtype) {
        return Err(format!("shortconv backward-x: unsupported dtype {dtype:?}"));
    }
    Ok(BackwardXRequirements {
        dtype,
        output_bytes: checked_bytes(&[batch, steps, channels], dtype, "shortconv backward-x")?,
        staging_bytes: 0,
        status_bytes: 0,
        scratch_bytes: 0,
        pipeline_count: 1,
    })
}

/// Requirements of the backward-w launch alone.
pub fn backward_w_requirements(
    dtype: DType,
    channels: usize,
    kernel: usize,
) -> crate::err::Res<BackwardWRequirements> {
    if !supported_dtype(dtype) {
        return Err(format!(
            "shortconv backward-weight: unsupported dtype {dtype:?}"
        ));
    }
    Ok(BackwardWRequirements {
        dtype,
        output_bytes: checked_bytes(&[channels, kernel], dtype, "shortconv backward-weight")?,
        staging_bytes: 0,
        status_bytes: 0,
        scratch_bytes: 0,
        pipeline_count: 1,
    })
}

#[cfg(test)]
mod requirement_tests {
    use super::*;

    #[test]
    fn requirements_are_exact_and_operation_specific() {
        let forward = forward_requirements(DType::BF16, 2, 9, 8, 4, false).unwrap();
        assert_eq!(forward.output_bytes, 2 * 9 * 8 * 2);
        assert_eq!(forward.state_next_bytes, 0);
        assert_eq!(forward.pipeline_count, 1);

        let state = state_requirements(DType::F32, 9, 8, 4).unwrap();
        assert_eq!(state.output_bytes, 9 * 8 * 4);
        assert_eq!(state.state_next_bytes, 3 * 8 * 4);

        let dx = backward_x_requirements(DType::F32, 2, 9, 8).unwrap();
        let dw = backward_w_requirements(DType::F32, 8, 4).unwrap();
        let both = backward_requirements(DType::F32, 2, 9, 8, 4).unwrap();
        assert_eq!(dx.output_bytes, 2 * 9 * 8 * 4);
        assert_eq!(dw.output_bytes, 8 * 4 * 4);
        assert_eq!(
            (both.dx_bytes, both.dweight_bytes),
            (dx.output_bytes, dw.output_bytes)
        );
        assert_eq!(
            (dx.pipeline_count, dw.pipeline_count, both.pipeline_count),
            (1, 1, 2)
        );
        assert!(forward_requirements(DType::F32, usize::MAX, 2, 1, 1, false).is_err());
    }
}

/// Whether the fused short-conv kernels support `dtype` (f32 or bf16;
/// the kernels accumulate in f32).
pub fn supported_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::BF16)
}

#[cfg(target_os = "macos")]
pub use metal::{
    backward_into, backward_w, backward_w_into, backward_x, backward_x_into, forward, forward_into,
    state_into, warm_all, warm_backward_exact, warm_backward_w, warm_backward_w_exact,
    warm_backward_x, warm_backward_x_exact, warm_forward, warm_forward_exact, warm_state,
    IntoResources,
};

#[cfg(target_os = "macos")]
mod metal {
    use crate::runtime::dtype::DType;
    use crate::runtime::metal::device::{set_buffer, set_bytes, MetalDevice, Pipeline};
    use crate::runtime::metal::run::MetalTensor;

    use objc2_metal::MTLComputeCommandEncoder;

    /// Borrowed resource views supplied by the executable planner. All
    /// short-conv kernels are self-contained, so every slice must be
    /// empty; see [`IntoResources::empty`].
    #[derive(Clone, Copy)]
    pub struct IntoResources<'a> {
        /// Must be empty.
        pub staging: &'a [MetalTensor],
        /// Must be empty.
        pub status: &'a [MetalTensor],
        /// Must be empty.
        pub scratch: &'a [MetalTensor],
    }

    impl IntoResources<'_> {
        /// The (only) valid resource set for short-conv dispatches.
        pub const fn empty() -> Self {
            Self {
                staging: &[],
                status: &[],
                scratch: &[],
            }
        }
    }

    fn wrap_contig(t: &MetalTensor) -> crate::err::Res<MetalTensor> {
        if t.layout.is_contiguous() && t.layout.offset() == 0 {
            Ok(t.clone())
        } else {
            crate::runtime::metal::kernels::strided_copy(MetalDevice::get(), t)
        }
    }

    fn validate_destination(
        tensor: &MetalTensor,
        shape: &[usize],
        dtype: DType,
        label: &str,
    ) -> crate::err::Res<()> {
        if tensor.dtype != dtype || tensor.layout.shape() != shape || !tensor.layout.is_contiguous()
        {
            return Err(format!(
                "{label}: expected contiguous {shape:?}:{dtype:?}, got {:?}:{:?}",
                tensor.layout.shape(),
                tensor.dtype
            ));
        }
        let bytes = tensor
            .layout
            .checked_max_index()
            .and_then(|elements| elements.checked_mul(dtype.size_in_bytes()))
            .ok_or_else(|| format!("{label}: byte size overflow"))?;
        if bytes > tensor.buffer.size {
            return Err(format!(
                "{label}: destination needs {bytes} bytes but buffer has {}",
                tensor.buffer.size
            ));
        }
        MetalDevice::get().mark_buffer_write(&tensor.buffer)?;
        Ok(())
    }

    fn validate_resources(resources: IntoResources<'_>, operation: &str) -> crate::err::Res<()> {
        if !resources.staging.is_empty()
            || !resources.status.is_empty()
            || !resources.scratch.is_empty()
        {
            return Err(format!(
                "{operation}: staging, status, and scratch views must be empty"
            ));
        }
        Ok(())
    }

    fn ty_of(dtype: DType) -> &'static str {
        match dtype {
            DType::F32 => "float",
            DType::BF16 => "bfloat",
            other => unreachable!("shortconv: unsupported dtype {other:?}"),
        }
    }

    fn fwd_source(dtype: DType) -> String {
        let ty = ty_of(dtype);
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;

#define T {ty}

// One thread per output element, addressed by a 3D grid (channel, row,
// batch) — no div/mod index math. x [.., steps, C], optional f32
// history [K-1, C] (single-sequence use only), y mirrors x; when flag 2
// is set the last K-1 real rows (of `advance`) plus surviving history
// write the shifted window to WIN.
kernel void et_shortconv_fwd(
    device const T* X [[buffer(0)]],
    device const T* W [[buffer(1)]],
    device const float* HIST [[buffer(2)]],
    device T* Y [[buffer(3)]],
    device float* WIN [[buffer(4)]],
    constant uint& steps [[buffer(5)]],
    constant uint& C [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& advance [[buffer(8)]],
    constant uint& flags [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    const uint c = gid.x;
    const uint t = gid.y;
    if (c >= C) return;
    const bool has_hist = (flags & 1u) != 0u;
    const bool write_win = (flags & 2u) != 0u;
    const ulong rowBase = (ulong)gid.z * steps * C;
    if (t < steps) {{
        float acc = 0.0f;
        for (uint j = 0; j < K; j++) {{
            const long s = long(t) + long(j) - long(K - 1);
            if (s >= 0) {{
                acc += float(W[c * K + j]) * float(X[rowBase + ulong(s) * C + c]);
            }} else if (has_hist) {{
                acc += float(W[c * K + j]) * HIST[ulong(K - 1 + s) * C + c];
            }}
        }}
        Y[rowBase + ulong(t) * C + c] = T(acc);
    }}
    if (write_win && t < K - 1 && gid.z == 0) {{
        // Window row t holds real position advance-(K-1)+t (negative:
        // the surviving history row K-1+s).
        const long s = long(advance) - long(K - 1) + long(t);
        float val = 0.0f;
        if (s >= 0) {{
            val = float(X[rowBase + ulong(s) * C + c]);
        }} else if (has_hist) {{
            val = HIST[ulong(K - 1 + s) * C + c];
        }}
        WIN[ulong(t) * C + c] = val;
    }}
}}
"#,
            ty = ty,
        )
    }
    fn bwd_x_source(dtype: DType) -> String {
        let ty = ty_of(dtype);
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;

#define T {ty}

// dx[s] = sum_j w[:, K-1-j] * g[s+j]: the full correlation against the
// right-zero-padded cotangent. One thread per element.
kernel void et_shortconv_bwd_x(
    device const T* G [[buffer(0)]],
    device const T* W [[buffer(1)]],
    device T* dX [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& steps [[buffer(4)]],
    constant uint& C [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    const uint c = gid.x;
    const uint t = gid.y;
    if (c >= C || t >= steps) return;
    const ulong rowBase = (ulong)gid.z * steps * C;
    float acc = 0.0f;
    for (uint j = 0; j < K; j++) {{
        if (t + j < steps) {{
            acc += float(W[c * K + (K - 1 - j)]) * float(G[rowBase + ulong(t + j) * C + c]);
        }}
    }}
    dX[rowBase + ulong(t) * C + c] = T(acc);
}}
"#,
            ty = ty,
        )
    }

    fn bwd_w_source(dtype: DType) -> String {
        let ty = ty_of(dtype);
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;

#define T {ty}

// dw[c, j] = sum over batch and t of g[t, c] * x[t-K+1+j, c] over the
// causal window. One thread per (channel, tap); the loops are serial.
kernel void et_shortconv_bwd_w(
    device const T* X [[buffer(0)]],
    device const T* G [[buffer(1)]],
    device T* dW [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& steps [[buffer(4)]],
    constant uint& C [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint2 gid2 [[thread_position_in_grid]]
) {{
    const ulong i = ulong(gid2.y) * {wide}ul + ulong(gid2.x);
    if (i >= ulong(C) * K) return;
    const uint j = uint(i % K);
    const uint c = uint(i / K);
    float acc = 0.0f;
    for (uint b = 0; b < batch; b++) {{
        const ulong base = ulong(b) * steps * C;
        for (uint t = K - 1 - j; t < steps; t++) {{
            acc += float(G[base + ulong(t) * C + c])
                * float(X[base + ulong(t - (K - 1 - j)) * C + c]);
        }}
    }}
    dW[i] = T(acc);
}}
"#,
            ty = ty,
            wide = MetalDevice::WIDE,
        )
    }

    fn pipeline(
        key_tag: u32,
        dtype: DType,
        name: &'static str,
        make_src: impl Fn() -> String,
    ) -> crate::err::Res<Pipeline> {
        MetalDevice::get().compile_lazy(((key_tag as u64) << 8) | dtype as u64, name, make_src)
    }

    fn cached_pipeline(key_tag: u32, dtype: DType, name: &str) -> crate::err::Res<Pipeline> {
        let key = ((key_tag as u64) << 8) | dtype as u64;
        MetalDevice::get().pipeline_cached(key).ok_or_else(|| {
            format!("shortconv: exact {name} pipeline is not warm; call the matching warm function")
        })
    }

    /// Warms all three short-conv pipelines (forward, backward-x,
    /// backward-w) for `dtype`.
    pub fn warm_all(dtype: DType) -> crate::err::Res<()> {
        pipeline(0xC041, dtype, "et_shortconv_fwd", || fwd_source(dtype))?;
        pipeline(0xC042, dtype, "et_shortconv_bwd_x", || bwd_x_source(dtype))?;
        pipeline(0xC043, dtype, "et_shortconv_bwd_w", || bwd_w_source(dtype))?;
        Ok(())
    }

    /// Warms the forward pipeline for `dtype`.
    pub fn warm_forward(dtype: DType) -> crate::err::Res<()> {
        pipeline(0xC041, dtype, "et_shortconv_fwd", || fwd_source(dtype))?;
        Ok(())
    }

    /// Warms exactly the pipeline described by `requirements`.
    pub fn warm_forward_exact(requirements: &super::ForwardRequirements) -> crate::err::Res<()> {
        warm_forward(requirements.dtype)
    }

    /// Warms the pipeline used by the ConvState transaction (same
    /// kernel as the plain forward).
    pub fn warm_state(dtype: DType) -> crate::err::Res<()> {
        warm_forward(dtype)
    }

    /// Warms the backward-x pipeline for `dtype`.
    pub fn warm_backward_x(dtype: DType) -> crate::err::Res<()> {
        pipeline(0xC042, dtype, "et_shortconv_bwd_x", || bwd_x_source(dtype))?;
        Ok(())
    }

    /// Warms exactly the pipeline described by `requirements`.
    pub fn warm_backward_x_exact(
        requirements: &super::BackwardXRequirements,
    ) -> crate::err::Res<()> {
        warm_backward_x(requirements.dtype)
    }

    /// Warms the backward-w pipeline for `dtype`.
    pub fn warm_backward_w(dtype: DType) -> crate::err::Res<()> {
        pipeline(0xC043, dtype, "et_shortconv_bwd_w", || bwd_w_source(dtype))?;
        Ok(())
    }

    /// Warms exactly the pipeline described by `requirements`.
    pub fn warm_backward_w_exact(
        requirements: &super::BackwardWRequirements,
    ) -> crate::err::Res<()> {
        warm_backward_w(requirements.dtype)
    }

    /// Warms both pipelines described by the combined backward plan.
    pub fn warm_backward_exact(requirements: &super::BackwardRequirements) -> crate::err::Res<()> {
        warm_backward_x(requirements.dtype)?;
        warm_backward_w(requirements.dtype)
    }

    fn grid_over(n: usize) -> (objc2_metal::MTLSize, objc2_metal::MTLSize) {
        // grid_flat yields total-THREAD sizes: pair with dispatchThreads.
        MetalDevice::grid_flat(n.div_ceil(256) * 256)
    }

    // 3D (channels, rows, batch) thread grid, 32×8 threadgroups.
    fn grid3d(
        c: usize,
        steps: usize,
        batch: usize,
    ) -> (objc2_metal::MTLSize, objc2_metal::MTLSize) {
        (
            MetalDevice::grid(c.div_ceil(32) * 32, steps.div_ceil(8) * 8, batch),
            MetalDevice::grid(32, 8, 1),
        )
    }

    /// Non-allocating forward dispatch: `x [.., steps, C]`, `weight
    /// [C, K]`, optional f32 `history [K-1, C]`, optional f32
    /// `state_next [K-1, C]`. `advance` is the number of real rows (≤
    /// `steps`; chunked prefill right-pads, so only the `advance`
    /// prefix shifts into the window). History/state are
    /// single-sequence only. Allocates nothing; requires the forward
    /// pipeline to be warm.
    pub fn forward_into(
        x: &MetalTensor,
        weight: &MetalTensor,
        history: Option<&MetalTensor>,
        advance: usize,
        output: &MetalTensor,
        state_next: Option<&MetalTensor>,
        resources: IntoResources<'_>,
    ) -> crate::err::Res<()> {
        let dtype = x.dtype;
        let shape = x.layout.shape();
        let r = shape.len();
        let (steps, c) = (shape[r - 2], shape[r - 1]);
        let batch: usize = shape[..r - 2].iter().product();
        let kk = weight.layout.shape()[1];
        validate_resources(resources, "shortconv forward")?;
        if advance > steps {
            return Err(format!(
                "shortconv forward: advance {advance} exceeds {steps} input steps"
            ));
        }
        if batch != 1 && (history.is_some() || state_next.is_some()) {
            return Err("shortconv state: history/state-next require one sequence".to_string());
        }
        if !x.layout.is_contiguous()
            || !weight.layout.is_contiguous()
            || history.is_some_and(|tensor| !tensor.layout.is_contiguous())
        {
            return Err(
                "shortconv forward: inputs must be contiguous before forward_into".to_string(),
            );
        }
        validate_destination(output, shape, dtype, "shortconv forward output")?;
        if let Some(history) = history {
            validate_destination(
                history,
                &[kk.saturating_sub(1), c],
                DType::F32,
                "shortconv history",
            )?;
        }
        if let Some(state) = state_next {
            validate_destination(
                state,
                &[kk.saturating_sub(1), c],
                DType::F32,
                "shortconv state-next",
            )?;
        }
        let dev = MetalDevice::get();
        let pipe = cached_pipeline(0xC041, dtype, "forward")?;
        let mut flags = 0u32;
        if history.is_some() {
            flags |= 1;
        }
        if state_next.is_some() {
            flags |= 2;
        }
        let elem = |off: usize| off * dtype.size_in_bytes();
        let (grid, tg) = grid3d(
            c,
            steps.max(if state_next.is_some() { kk - 1 } else { 0 }),
            batch,
        );
        dev.with_encoder(|e| {
            e.setComputePipelineState(pipe.as_raw());
            set_buffer(e, 0, &x.buffer, elem(x.layout.offset()));
            set_buffer(e, 1, &weight.buffer, elem(weight.layout.offset()));
            let history = history.unwrap_or(output);
            let state_next = state_next.unwrap_or(output);
            set_buffer(
                e,
                2,
                &history.buffer,
                history.layout.offset() * history.dtype.size_in_bytes(),
            );
            set_buffer(e, 3, &output.buffer, elem(output.layout.offset()));
            set_buffer(
                e,
                4,
                &state_next.buffer,
                state_next.layout.offset() * state_next.dtype.size_in_bytes(),
            );
            set_bytes(e, 5, &(steps as u32));
            set_bytes(e, 6, &(c as u32));
            set_bytes(e, 7, &(kk as u32));
            set_bytes(e, 8, &(advance as u32));
            set_bytes(e, 9, &flags);
            e.dispatchThreads_threadsPerThreadgroup(grid, tg);
        });
        Ok(())
    }

    /// ConvState transaction: [`forward_into`] with a mandatory
    /// `state_next` window write.
    #[allow(clippy::too_many_arguments)]
    pub fn state_into(
        x: &MetalTensor,
        weight: &MetalTensor,
        history: Option<&MetalTensor>,
        advance: usize,
        output: &MetalTensor,
        state_next: &MetalTensor,
        resources: IntoResources<'_>,
    ) -> crate::err::Res<()> {
        forward_into(
            x,
            weight,
            history,
            advance,
            output,
            Some(state_next),
            resources,
        )
    }

    /// Allocating convenience wrapper around [`forward_into`]: makes
    /// inputs contiguous, warms the pipeline, and returns the output
    /// plus the shifted window when `write_window` is set.
    pub fn forward(
        x: &MetalTensor,
        weight: &MetalTensor,
        history: Option<&MetalTensor>,
        advance: usize,
        write_window: bool,
    ) -> crate::err::Res<(MetalTensor, Option<MetalTensor>)> {
        let dtype = x.dtype;
        let shape = x.layout.shape().to_vec();
        let r = shape.len();
        let c = shape[r - 1];
        let kk = weight.layout.shape()[1];
        let x = wrap_contig(x)?;
        let weight = wrap_contig(weight)?;
        let history = history.map(wrap_contig).transpose()?;
        warm_forward(dtype)?;
        let output = MetalTensor::empty(MetalDevice::get(), shape, dtype);
        let state_next = write_window.then(|| {
            MetalTensor::empty(
                MetalDevice::get(),
                vec![kk.saturating_sub(1), c],
                DType::F32,
            )
        });
        forward_into(
            &x,
            &weight,
            history.as_ref(),
            advance,
            &output,
            state_next.as_ref(),
            IntoResources::empty(),
        )?;
        Ok((output, state_next))
    }

    /// Non-allocating backward-x dispatch: cotangent `g [.., steps,
    /// C]` and `weight [C, K]` produce `dx` with `g`'s shape. Allocates
    /// nothing; requires the backward-x pipeline to be warm.
    pub fn backward_x_into(
        g: &MetalTensor,
        weight: &MetalTensor,
        dx: &MetalTensor,
        resources: IntoResources<'_>,
    ) -> crate::err::Res<()> {
        let dtype = g.dtype;
        let shape = g.layout.shape();
        let r = shape.len();
        let (steps, c) = (shape[r - 2], shape[r - 1]);
        let batch: usize = shape[..r - 2].iter().product();
        let kk = weight.layout.shape()[1];
        validate_resources(resources, "shortconv backward-x")?;
        if !g.layout.is_contiguous() || !weight.layout.is_contiguous() {
            return Err(
                "shortconv backward-x: inputs must be contiguous before backward_x_into"
                    .to_string(),
            );
        }
        validate_destination(dx, shape, dtype, "shortconv backward-x output")?;
        let dev = MetalDevice::get();
        let pipe = cached_pipeline(0xC042, dtype, "backward-x")?;
        let elem = |off: usize| off * dtype.size_in_bytes();
        let (grid, tg) = grid3d(c, steps, batch);
        dev.with_encoder(|e| {
            e.setComputePipelineState(pipe.as_raw());
            set_buffer(e, 0, &g.buffer, elem(g.layout.offset()));
            set_buffer(e, 1, &weight.buffer, elem(weight.layout.offset()));
            set_buffer(e, 2, &dx.buffer, elem(dx.layout.offset()));
            set_bytes(e, 3, &(batch as u32));
            set_bytes(e, 4, &(steps as u32));
            set_bytes(e, 5, &(c as u32));
            set_bytes(e, 6, &(kk as u32));
            e.dispatchThreads_threadsPerThreadgroup(grid, tg);
        });
        Ok(())
    }

    /// Allocating convenience wrapper around [`backward_x_into`].
    pub fn backward_x(g: &MetalTensor, weight: &MetalTensor) -> crate::err::Res<MetalTensor> {
        let shape = g.layout.shape().to_vec();
        let dtype = g.dtype;
        let g = wrap_contig(g)?;
        let weight = wrap_contig(weight)?;
        warm_backward_x(dtype)?;
        let dx = MetalTensor::empty(MetalDevice::get(), shape, dtype);
        backward_x_into(&g, &weight, &dx, IntoResources::empty())?;
        Ok(dx)
    }

    /// Non-allocating backward-w dispatch: `x` and cotangent `g` (both
    /// `[.., steps, C]`) produce `dweight [C, kernel]`, summed over
    /// batch and time. Allocates nothing; requires the backward-w
    /// pipeline to be warm.
    pub fn backward_w_into(
        x: &MetalTensor,
        g: &MetalTensor,
        kernel: usize,
        dweight: &MetalTensor,
        resources: IntoResources<'_>,
    ) -> crate::err::Res<()> {
        let dtype = g.dtype;
        let shape = g.layout.shape();
        let r = shape.len();
        let (steps, c) = (shape[r - 2], shape[r - 1]);
        let batch: usize = shape[..r - 2].iter().product();
        validate_resources(resources, "shortconv backward-weight")?;
        if !x.layout.is_contiguous() || !g.layout.is_contiguous() {
            return Err(
                "shortconv backward-weight: inputs must be contiguous before backward_w_into"
                    .to_string(),
            );
        }
        validate_destination(
            dweight,
            &[c, kernel],
            dtype,
            "shortconv backward-weight output",
        )?;
        let dev = MetalDevice::get();
        let pipe = cached_pipeline(0xC043, dtype, "backward-weight")?;
        let elem = |off: usize| off * dtype.size_in_bytes();
        let (grid, tg) = grid_over(c * kernel);
        dev.with_encoder(|e| {
            e.setComputePipelineState(pipe.as_raw());
            set_buffer(e, 0, &x.buffer, elem(x.layout.offset()));
            set_buffer(e, 1, &g.buffer, elem(g.layout.offset()));
            set_buffer(e, 2, &dweight.buffer, elem(dweight.layout.offset()));
            set_bytes(e, 3, &(batch as u32));
            set_bytes(e, 4, &(steps as u32));
            set_bytes(e, 5, &(c as u32));
            set_bytes(e, 6, &(kernel as u32));
            e.dispatchThreads_threadsPerThreadgroup(grid, tg);
        });
        Ok(())
    }

    /// Allocating convenience wrapper around [`backward_w_into`].
    pub fn backward_w(
        x: &MetalTensor,
        g: &MetalTensor,
        kernel: usize,
    ) -> crate::err::Res<MetalTensor> {
        let dtype = g.dtype;
        let c = g.layout.shape()[g.layout.shape().len() - 1];
        let x = wrap_contig(x)?;
        let g = wrap_contig(g)?;
        warm_backward_w(dtype)?;
        let dweight = MetalTensor::empty(MetalDevice::get(), vec![c, kernel], dtype);
        backward_w_into(&x, &g, kernel, &dweight, IntoResources::empty())?;
        Ok(dweight)
    }

    /// Combined non-allocating backward: dispatches backward-x into
    /// `dx` and backward-w into `dweight` from one call.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_into(
        x: &MetalTensor,
        weight: &MetalTensor,
        gradient: &MetalTensor,
        dx: &MetalTensor,
        dweight: &MetalTensor,
        resources: IntoResources<'_>,
    ) -> crate::err::Res<()> {
        validate_resources(resources, "shortconv backward")?;
        backward_x_into(gradient, weight, dx, IntoResources::empty())?;
        backward_w_into(
            x,
            gradient,
            weight.layout.shape()[1],
            dweight,
            IntoResources::empty(),
        )
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use crate::runtime::metal::device::MetalDevice;
    use crate::runtime::metal::run::MetalTensor as MT;

    fn prand(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s % 2000) as f32 / 1000.0) - 1.0
            })
            .collect()
    }

    fn max_diff(x: &[f32], y: &[f32]) -> f32 {
        x.iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
    }

    #[test]
    fn kernels_match_composed() {
        let dev = MetalDevice::get();
        for (b, t, c, k) in [
            (2usize, 9usize, 8usize, 4usize),
            (1, 3, 5, 2),
            (2, 70, 64, 4),
        ] {
            let x = MT::from_f32(dev, prand(b * t * c, 31), vec![b, t, c]);
            let w = MT::from_f32(dev, prand(c * k, 32), vec![c, k]);
            let g = MT::from_f32(dev, prand(b * t * c, 33), vec![b, t, c]);

            crate::runtime::metal::indexing::warm_cat(&[&[b, k - 1, c], &[b, t, c]], x.dtype, 1)
                .unwrap();
            crate::runtime::metal::indexing::warm_cat(&[&[b, t, c], &[b, k - 1, c]], x.dtype, 1)
                .unwrap();
            let mut width = 1usize;
            while width < k {
                let right = width.min(k - width);
                crate::runtime::metal::indexing::warm_cat(
                    &[&[b, width, c], &[b, right, c]],
                    x.dtype,
                    1,
                )
                .unwrap();
                width += right;
            }

            let fwd = super::metal::forward(&x, &w, None, t, false).unwrap().0;
            let composed_fwd =
                crate::runtime::metal::composed::short_conv1d_forward(&x, &w).unwrap();
            let dx = super::metal::backward_x(&g, &w).unwrap();
            let composed_dx =
                crate::runtime::metal::composed::short_conv1d_backward_x(&x, &w, &g).unwrap();
            let dw = super::metal::backward_w(&x, &g, k).unwrap();
            let composed_dw =
                crate::runtime::metal::composed::short_conv1d_backward_w(&x, &w, &g).unwrap();
            dev.synchronize().unwrap();
            let df = max_diff(&fwd.read_f32().unwrap(), &composed_fwd.read_f32().unwrap());
            let ddx = max_diff(&dx.read_f32().unwrap(), &composed_dx.read_f32().unwrap());
            let ddw = max_diff(&dw.read_f32().unwrap(), &composed_dw.read_f32().unwrap());
            assert!(
                df < 1e-4 && ddx < 1e-4 && ddw < 1e-4,
                "({b},{t},{c},{k}): fwd {df} dx {ddx} dw {ddw}"
            );
        }
    }

    #[test]
    fn history_window_matches_stateless() {
        let dev = MetalDevice::get();
        let (t, c, k) = (11usize, 8usize, 4usize);
        let x = MT::from_f32(dev, prand(t * c, 41), vec![t, c]);
        let w = MT::from_f32(dev, prand(c * k, 42), vec![c, k]);
        // Split at 5: the second half with the first half's trailing
        // window must equal the full-sequence output tail.
        let full = super::metal::forward(&x, &w, None, t, false).unwrap().0;
        let first = MT {
            buffer: x.buffer.clone(),
            layout: x.layout.narrow(0, 0, 5),
            dtype: x.dtype,
        };
        let (_, window) = super::metal::forward(&first, &w, None, 5, true).unwrap();
        let second = MT {
            buffer: x.buffer.clone(),
            layout: x.layout.narrow(0, 5, t - 5),
            dtype: x.dtype,
        };
        let (tail, _) = super::metal::forward(&second, &w, window.as_ref(), t - 5, false).unwrap();
        let full_tail = crate::runtime::metal::ops::contiguous(&MT {
            buffer: full.buffer.clone(),
            layout: full.layout.narrow(0, 5, t - 5),
            dtype: full.dtype,
        })
        .unwrap();
        dev.synchronize().unwrap();
        let diff = max_diff(&full_tail.read_f32().unwrap(), &tail.read_f32().unwrap());
        assert!(diff < 1e-5, "stateful tail diff {diff}");
    }

    #[test]
    fn into_matches_wrappers_and_uses_only_supplied_views() {
        let dev = MetalDevice::get();
        let (b, t, c, k) = (1usize, 7usize, 6usize, 4usize);
        let x = MT::from_f32(dev, prand(b * t * c, 51), vec![b, t, c]);
        let weight = MT::from_f32(dev, prand(c * k, 52), vec![c, k]);
        let gradient = MT::from_f32(dev, prand(b * t * c, 53), vec![b, t, c]);
        let (expected_output, expected_state) =
            super::metal::forward(&x, &weight, None, t, true).unwrap();
        let expected_dx = super::metal::backward_x(&gradient, &weight).unwrap();
        let expected_dw = super::metal::backward_w(&x, &gradient, k).unwrap();

        let output = MT::empty(dev, vec![b, t, c], x.dtype);
        let state_next = MT::empty(dev, vec![k - 1, c], crate::runtime::dtype::DType::F32);
        let dx = MT::empty(dev, vec![b, t, c], x.dtype);
        let dweight = MT::empty(dev, vec![c, k], x.dtype);
        let forward_requirements = super::state_requirements(x.dtype, t, c, k).unwrap();
        let dx_requirements = super::backward_x_requirements(x.dtype, b, t, c).unwrap();
        let dw_requirements = super::backward_w_requirements(x.dtype, c, k).unwrap();
        super::metal::warm_forward_exact(&forward_requirements).unwrap();
        super::metal::warm_backward_x_exact(&dx_requirements).unwrap();
        super::metal::warm_backward_w_exact(&dw_requirements).unwrap();
        dev.synchronize().unwrap();

        let _dispatch_guard = dev.begin_executable_dispatch().unwrap();
        super::metal::state_into(
            &x,
            &weight,
            None,
            t,
            &output,
            &state_next,
            super::metal::IntoResources::empty(),
        )
        .unwrap();
        super::metal::backward_into(
            &x,
            &weight,
            &gradient,
            &dx,
            &dweight,
            super::metal::IntoResources::empty(),
        )
        .unwrap();
        dev.synchronize().unwrap();

        for (label, expected, actual) in [
            ("output", &expected_output, &output),
            ("state", expected_state.as_ref().unwrap(), &state_next),
            ("dx", &expected_dx, &dx),
            ("dweight", &expected_dw, &dweight),
        ] {
            let diff = max_diff(&expected.read_f32().unwrap(), &actual.read_f32().unwrap());
            assert!(diff < 1e-6, "{label} into parity diff {diff}");
        }
    }
}

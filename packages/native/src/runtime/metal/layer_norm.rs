//! Fused layer normalization on Metal: one kernel per direction over
//! per-row threadgroups (two in-kernel passes — mean, then variance —
//! one launch each way). The backward also emits x̂ so dw/db are two
//! plain reduce ops host-side instead of another kernel family. CPU
//! keeps the composed path in lib.rs.

use crate::runtime::metal::run::MetalTensor;

/// Whether the fused layer-norm path can run: Metal, f32, last-dim norm.
pub fn is_supported(x: &MetalTensor, weight: &MetalTensor) -> bool {
    x.dtype == crate::runtime::dtype::DType::F32 && weight.dtype == crate::runtime::dtype::DType::F32
}

#[cfg(target_os = "macos")]
pub use metal::{ln_backward, ln_forward};

#[cfg(target_os = "macos")]
mod metal {
    use crate::runtime::metal::device::{set_buffer, set_bytes, MetalDevice, Pipeline};
    use crate::runtime::metal::run::MetalTensor;

    use objc2_metal::MTLComputeCommandEncoder;
    use std::sync::Arc;

    const NT: usize = 128;

    fn wrap_contig(t: &MetalTensor) -> crate::err::Res<MetalTensor> {
        if t.layout.is_contiguous() {
            Ok(t.clone())
        } else {
            crate::runtime::metal::kernels::strided_copy(MetalDevice::get(), t)
        }
    }

    fn alloc_f32(n: usize) -> Arc<crate::runtime::metal::device::Buffer> {
        MetalDevice::get().alloc(n.max(1), crate::runtime::dtype::DType::F32)
    }

    fn source() -> &'static str {
        r#"
#include <metal_stdlib>
using namespace metal;

#define NT 128

// Per-row layer norm: mean and variance in two in-kernel passes, one
// launch. One threadgroup per row.
kernel void et_ln_fwd(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device const float* B [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    const uint row = tgid;
    device const float* x = X + (ulong)row * D;
    device float* o = O + (ulong)row * D;
    float s = 0.0f;
    for (uint j = tid; j < D; j += NT) { s += x[j]; }
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
    for (uint j = tid; j < D; j += NT) { const float c = x[j] - mean; q += c * c; }
    q = simd_sum(q);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { ps[grp] = q; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0.0f;
    for (uint g = 0; g < NT / 32; g++) { var += ps[g]; }
    var /= float(D);
    const float rstd = rsqrt(var + eps);
    for (uint j = tid; j < D; j += NT) {
        o[j] = (x[j] - mean) * rstd * W[j] + B[j];
    }
}

// Backward: dx plus x̂ (dw/db are plain reduces host-side).
kernel void et_ln_bwd(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device const float* G [[buffer(2)]],
    device float* DX [[buffer(3)]],
    device float* XH [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    const uint row = tgid;
    device const float* x = X + (ulong)row * D;
    device const float* g = G + (ulong)row * D;
    device float* dx = DX + (ulong)row * D;
    device float* xh = XH + (ulong)row * D;
    float s = 0.0f;
    for (uint j = tid; j < D; j += NT) { s += x[j]; }
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
    for (uint j = tid; j < D; j += NT) { const float c = x[j] - mean; q += c * c; }
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
        const float dyw = g[j] * W[j];
        const float hat = (x[j] - mean) * rstd;
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
        const float hat = (x[j] - mean) * rstd;
        xh[j] = hat;
        dx[j] = (g[j] * W[j] - m1 - hat * m2) * rstd;
    }
}
"#
    }

    fn pipeline(name: &'static str) -> crate::err::Res<Pipeline> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        name.hash(&mut hasher);
        let key = hasher.finish();
        MetalDevice::get().compile_lazy(key, name, || source().to_string())
    }

    fn wrap(buf: Arc<crate::runtime::metal::device::Buffer>, n: usize, shape: Vec<usize>) -> crate::err::Res<MetalTensor> {
        let _ = n;
        Ok(MetalTensor {
            buffer: buf,
            layout: crate::runtime::layout::Layout::contiguous(shape),
            dtype: crate::runtime::dtype::DType::F32,
        })
    }

    pub fn ln_forward(x: &MetalTensor, weight: &MetalTensor, bias: &MetalTensor, eps: f64) -> crate::err::Res<MetalTensor> {
        let d = weight.numel();
        let rows = x.numel() / d;
        let x = wrap_contig(x)?;
        let weight = wrap_contig(weight)?;
        let bias = wrap_contig(bias)?;
        let out_buf = alloc_f32(x.numel());
        let pipe = pipeline("et_ln_fwd")?;
        let off = |o: usize| o * 4;
        MetalDevice::get().with_encoder(|e| {
            e.setComputePipelineState(pipe.as_raw());
            set_buffer(e, 0, &x.buffer, off(x.layout.offset()));
            set_buffer(e, 1, &weight.buffer, off(weight.layout.offset()));
            set_buffer(e, 2, &bias.buffer, off(bias.layout.offset()));
            set_buffer(e, 3, &out_buf, 0);
            set_bytes(e, 4, &(d as u32));
            set_bytes(e, 5, &(eps as f32));
            e.dispatchThreadgroups_threadsPerThreadgroup(
                objc2_metal::MTLSize { width: rows, height: 1, depth: 1 },
                objc2_metal::MTLSize { width: NT, height: 1, depth: 1 },
            );
        });
        wrap(out_buf, x.numel(), x.layout.shape().to_vec())
    }

    // Returns (dx, x̂) — dw/db are computed host-side from x̂.
    pub fn ln_backward(x: &MetalTensor, weight: &MetalTensor, g: &MetalTensor, eps: f64) -> crate::err::Res<(MetalTensor, MetalTensor)> {
        let d = weight.numel();
        let rows = x.numel() / d;
        let x = wrap_contig(x)?;
        let weight = wrap_contig(weight)?;
        let g = wrap_contig(g)?;
        let dx_buf = alloc_f32(x.numel());
        let xh_buf = alloc_f32(x.numel());
        let pipe = pipeline("et_ln_bwd")?;
        let off = |o: usize| o * 4;
        MetalDevice::get().with_encoder(|e| {
            e.setComputePipelineState(pipe.as_raw());
            set_buffer(e, 0, &x.buffer, off(x.layout.offset()));
            set_buffer(e, 1, &weight.buffer, off(weight.layout.offset()));
            set_buffer(e, 2, &g.buffer, off(g.layout.offset()));
            set_buffer(e, 3, &dx_buf, 0);
            set_buffer(e, 4, &xh_buf, 0);
            set_bytes(e, 5, &(d as u32));
            set_bytes(e, 6, &(eps as f32));
            e.dispatchThreadgroups_threadsPerThreadgroup(
                objc2_metal::MTLSize { width: rows, height: 1, depth: 1 },
                objc2_metal::MTLSize { width: NT, height: 1, depth: 1 },
            );
        });
        Ok((
            wrap(dx_buf, x.numel(), x.layout.shape().to_vec())?,
            wrap(xh_buf, x.numel(), x.layout.shape().to_vec())?,
        ))
    }
}


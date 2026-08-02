//! Fused layer normalization on Metal: one kernel per direction over
//! per-row threadgroups (two in-kernel passes — mean, then variance —
//! one launch each way). The backward also emits x̂ so dw/db are two
//! plain reduce ops host-side instead of another kernel family. CPU
//! keeps the composed path in lib.rs.

use candle_core::{DType, Device, Tensor};

/// Whether the fused layer-norm path can run: Metal, f32, last-dim norm.
pub fn is_supported(x: &Tensor, weight: &Tensor) -> bool {
    matches!(x.device(), Device::Metal(_))
        && x.dtype() == DType::F32
        && weight.dtype() == DType::F32
        && x.dim(candle_core::D::Minus1).is_ok()
}

#[cfg(target_os = "macos")]
pub use metal::{ln_backward, ln_forward};

#[cfg(target_os = "macos")]
mod metal {
    use candle_core::{DType, MetalStorage, Storage, Tensor};
    use candle_metal_kernels::metal::ComputePipeline;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    const NT: usize = 128;

    fn pipelines() -> &'static Mutex<HashMap<u64, ComputePipeline>> {
        static CACHE: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
        CACHE.get_or_init(|| Mutex::new(HashMap::new()))
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

    fn pipeline(mdev: &candle_core::MetalDevice, name: &'static str) -> candle_core::Result<ComputePipeline> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        name.hash(&mut hasher);
        let key = hasher.finish();
        let mut cache = pipelines().lock().unwrap();
        if let Some(p) = cache.get(&key) {
            return Ok(p.clone());
        }
        #[allow(deprecated)]
        let opts = {
            let o = objc2_metal::MTLCompileOptions::new();
            o.setFastMathEnabled(false);
            o
        };
        let lib = mdev
            .device()
            .new_library_with_source(source(), Some(&opts))
            .map_err(|e| candle_core::Error::Msg(format!("layer_norm: {e}")))?;
        let func = lib
            .get_function(name, None)
            .map_err(|e| candle_core::Error::Msg(format!("layer_norm: {e}")))?;
        let p = mdev
            .device()
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| candle_core::Error::Msg(format!("layer_norm: {e}")))?;
        cache.insert(key, p.clone());
        Ok(p)
    }

    fn buffer_of(t: &Tensor) -> candle_core::Result<(candle_metal_kernels::metal::Buffer, usize)> {
        let (storage, layout) = t.storage_and_layout();
        match &*storage {
            Storage::Metal(m) => Ok((m.buffer().clone(), layout.start_offset())),
            _ => Err(candle_core::Error::Msg(
                "layer_norm: expected Metal storage".to_string(),
            )),
        }
    }

    fn wrap(buf: std::sync::Arc<candle_metal_kernels::metal::Buffer>, mdev: &candle_core::MetalDevice, n: usize, shape: Vec<usize>) -> Tensor {
        Tensor::from_storage(
            Storage::Metal(MetalStorage::new(buf, mdev.clone(), n, DType::F32)),
            shape,
            candle_core::op::BackpropOp::none(),
            false,
        )
    }

    pub fn ln_forward(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
        let rank = x.rank();
        let d = weight.elem_count();
        let rows = x.elem_count() / d;
        let device = x.device();
        let mdev = device.as_metal_device()?;
        let x = x.contiguous()?;
        let weight = weight.contiguous()?;
        let bias = bias.contiguous()?;
        let out_buf = mdev.new_buffer(x.elem_count(), DType::F32, "ln_fwd")?;
        let encoder = mdev.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline(mdev, "et_ln_fwd")?);
        let (xb, xo) = buffer_of(&x)?;
        let (wb, wo) = buffer_of(&weight)?;
        let (bb, bo) = buffer_of(&bias)?;
        let off = |o: usize| o * DType::F32.size_in_bytes();
        encoder.set_buffer(0, Some(&xb), off(xo));
        encoder.set_buffer(1, Some(&wb), off(wo));
        encoder.set_buffer(2, Some(&bb), off(bo));
        encoder.set_buffer(3, Some(&out_buf), 0);
        encoder.set_bytes(4, &(d as u32));
        encoder.set_bytes(5, &(eps as f32));
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize { width: rows, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: NT, height: 1, depth: 1 },
        );
        Ok(wrap(out_buf, mdev, x.elem_count(), x.dims().to_vec()))
    }

    // Returns (dx, x̂) — dw/db are computed host-side from x̂.
    pub fn ln_backward(x: &Tensor, weight: &Tensor, g: &Tensor, eps: f64) -> candle_core::Result<(Tensor, Tensor)> {
        let rank = x.rank();
        let d = weight.elem_count();
        let rows = x.elem_count() / d;
        let device = x.device();
        let mdev = device.as_metal_device()?;
        let x = x.contiguous()?;
        let weight = weight.contiguous()?;
        let g = g.contiguous()?;
        let dx_buf = mdev.new_buffer(x.elem_count(), DType::F32, "ln_dx")?;
        let xh_buf = mdev.new_buffer(x.elem_count(), DType::F32, "ln_xh")?;
        let encoder = mdev.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline(mdev, "et_ln_bwd")?);
        let (xb, xo) = buffer_of(&x)?;
        let (wb, wo) = buffer_of(&weight)?;
        let (gb, go) = buffer_of(&g)?;
        let off = |o: usize| o * DType::F32.size_in_bytes();
        encoder.set_buffer(0, Some(&xb), off(xo));
        encoder.set_buffer(1, Some(&wb), off(wo));
        encoder.set_buffer(2, Some(&gb), off(go));
        encoder.set_buffer(3, Some(&dx_buf), 0);
        encoder.set_buffer(4, Some(&xh_buf), 0);
        encoder.set_bytes(5, &(d as u32));
        encoder.set_bytes(6, &(eps as f32));
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize { width: rows, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: NT, height: 1, depth: 1 },
        );
        Ok((
            wrap(dx_buf, mdev, x.elem_count(), x.dims().to_vec()),
            wrap(xh_buf, mdev, x.elem_count(), x.dims().to_vec()),
        ))
    }
}

#[cfg(not(target_os = "macos"))]
mod metal {
    use candle_core::Tensor;
    pub fn ln_forward(_x: &Tensor, _w: &Tensor, _b: &Tensor, _eps: f64) -> candle_core::Result<Tensor> {
        unreachable!("fused layer norm is Metal-only")
    }
    pub fn ln_backward(_x: &Tensor, _w: &Tensor, _g: &Tensor, _eps: f64) -> candle_core::Result<(Tensor, Tensor)> {
        unreachable!("fused layer norm is Metal-only")
    }
}

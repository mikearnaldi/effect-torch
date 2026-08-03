//! Fused cross-entropy on Metal: the composed path (~30 ops with four
//! synchronous host readbacks for count/label checks) becomes one
//! kernel per row-block plus one tiny status read per direction.
//! Forward: per-row online logsumexp + nll in one pass, a single
//! status kernel (loss, active count, invalid count), one 12-byte
//! readback that preserves the exact error semantics. Backward:
//! device-side active count, then probs − one_hot in one pass — no
//! host round trip beyond the same zero-count check. CPU keeps the
//! composed reference path.

use crate::runtime::dtype::DType;
use crate::runtime::metal::run::MetalTensor;

/// Whether the fused CE path can run: Metal, f32 logits, integer targets.
pub fn is_supported(logits: &MetalTensor, target: &MetalTensor) -> bool {
    logits.dtype == DType::F32 && matches!(target.dtype, DType::U32 | DType::I64)
}

#[cfg(target_os = "macos")]
pub use metal::{ce_backward, ce_forward};

#[cfg(target_os = "macos")]
mod metal {
    use crate::runtime::metal::device::{set_buffer, set_bytes, MetalDevice, Pipeline};
    use crate::runtime::metal::run::MetalTensor;
    use crate::runtime::dtype::DType;

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

    fn alloc(n: usize, dtype: crate::runtime::dtype::DType) -> Arc<crate::runtime::metal::device::Buffer> {
        MetalDevice::get().alloc(n.max(1), dtype)
    }

    fn source(tgt: crate::runtime::dtype::DType) -> String {
        let tgt_ty = match tgt {
            DType::U32 => "uint",
            DType::I64 => "long",
            other => unreachable!("ce: unsupported target dtype {other:?}"),
        };
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;

#define NT {nt}
#define TGT {tgt_ty}

// One threadgroup per row: online (max, sumexp) over the row's V
// logits, then nll and status flags. flags bit0 = ignored,
// bit1 = invalid-but-active target.
kernel void et_ce_fwd(
    device const float* Z [[buffer(0)]],
    device const TGT* T [[buffer(1)]],
    device float* nll [[buffer(2)]],
    device uint* flags [[buffer(3)]],
    constant uint& V [[buffer(4)]],
    constant long& ignore [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {{
    const uint row = tgid;
    device const float* z = Z + (ulong)row * V;
    const long t = (long)T[row];
    const bool ignored = (t == ignore);
    const bool invalid = !ignored && (t < 0 || t >= (long)V);
    float m = -INFINITY;
    float l = 0.0f;
    for (uint j = tid; j < V; j += NT) {{
        const float v = z[j];
        const float mn = max(m, v);
        l = l * exp(m - mn) + exp(v - mn);
        m = mn;
    }}
    // Fold (m, l): group max, rescale, group sum (simd groups of 32).
    const float gm = simd_max(m);
    l *= (gm == -INFINITY) ? 0.0f : exp(m - gm);
    l = simd_sum(l);
    threadgroup float pm[NT / 32];
    threadgroup float pl[NT / 32];
    const uint lane = tid % 32;
    const uint grp = tid / 32;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {{ pm[grp] = gm; pl[grp] = l; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {{
        float fm = -INFINITY;
        for (uint g = 0; g < NT / 32; g++) {{ fm = max(fm, pm[g]); }}
        float fl = 0.0f;
        for (uint g = 0; g < NT / 32; g++) {{ fl += pl[g] * exp(pm[g] - fm); }}
        const float lse = fm + log(fl);
        nll[row] = (ignored || invalid) ? 0.0f : lse - z[t];
        flags[row] = (ignored ? 1u : 0u) | (invalid ? 2u : 0u);
    }}
}}

// Single threadgroup: loss = sum(nll) / active, plus active and
// invalid counts for the host-side error semantics.
kernel void et_ce_status(
    device const float* nll [[buffer(0)]],
    device const uint* flags [[buffer(1)]],
    device float* status [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]]
) {{
    float s = 0.0f;
    uint active = 0;
    uint invalid = 0;
    for (uint i = tid; i < N; i += NT) {{
        s += nll[i];
        active += (flags[i] & 1u) ? 0u : 1u;
        invalid += (flags[i] & 2u) ? 1u : 0u;
    }}
    s = simd_sum(s);
    active = simd_sum(active);
    invalid = simd_sum(invalid);
    threadgroup float ps[NT / 32];
    threadgroup uint pa[NT / 32];
    threadgroup uint pi[NT / 32];
    const uint lane = tid % 32;
    const uint grp = tid / 32;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {{ ps[grp] = s; pa[grp] = active; pi[grp] = invalid; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {{
        float ts = 0.0f;
        uint ta = 0, ti = 0;
        for (uint g = 0; g < NT / 32; g++) {{ ts += ps[g]; ta += pa[g]; ti += pi[g]; }}
        status[0] = ta > 0 ? ts / float(ta) : 0.0f;
        status[1] = float(ta);
        status[2] = float(ti);
    }}
}}

// Active target count on device (backward divides by it without a
// host round trip).
kernel void et_ce_count(
    device const TGT* T [[buffer(0)]],
    device float* count [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    constant long& ignore [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]]
) {{
    uint active = 0;
    for (uint i = tid; i < N; i += NT) {{ active += ((long)T[i] == ignore) ? 0u : 1u; }}
    active = simd_sum(active);
    threadgroup uint pa[NT / 32];
    const uint lane = tid % 32;
    const uint grp = tid / 32;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {{ pa[grp] = active; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {{
        uint ta = 0;
        for (uint g = 0; g < NT / 32; g++) {{ ta += pa[g]; }}
        count[0] = float(ta);
    }}
}}

// grad = (softmax(z) - one_hot(t)) / count for active rows, zeros
// where ignored. One threadgroup per row, one pass.
kernel void et_ce_bwd(
    device const float* Z [[buffer(0)]],
    device const TGT* T [[buffer(1)]],
    device const float* count [[buffer(2)]],
    device float* G [[buffer(3)]],
    constant uint& V [[buffer(4)]],
    constant long& ignore [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {{
    const uint row = tgid;
    device const float* z = Z + (ulong)row * V;
    device float* g = G + (ulong)row * V;
    const long t = (long)T[row];
    if (t == ignore) {{
        for (uint j = tid; j < V; j += NT) {{ g[j] = 0.0f; }}
        return;
    }}
    float m = -INFINITY;
    float l = 0.0f;
    for (uint j = tid; j < V; j += NT) {{
        const float v = z[j];
        const float mn = max(m, v);
        l = l * exp(m - mn) + exp(v - mn);
        m = mn;
    }}
    const float gm = simd_max(m);
    l *= (gm == -INFINITY) ? 0.0f : exp(m - gm);
    l = simd_sum(l);
    threadgroup float pm[NT / 32];
    threadgroup float pl[NT / 32];
    const uint lane = tid % 32;
    const uint grp = tid / 32;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {{ pm[grp] = gm; pl[grp] = l; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float lse;
    if (tid == 0) {{
        float fm = -INFINITY;
        for (uint g = 0; g < NT / 32; g++) {{ fm = max(fm, pm[g]); }}
        float fl = 0.0f;
        for (uint g = 0; g < NT / 32; g++) {{ fl += pl[g] * exp(pm[g] - fm); }}
        lse = fm + log(fl);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / count[0];
    for (uint j = tid; j < V; j += NT) {{
        const float p = exp(z[j] - lse);
        g[j] = (p - ((long)j == t ? 1.0f : 0.0f)) * inv;
    }}
}}
"#,
            nt = NT,
            tgt_ty = tgt_ty,
        )
    }

    fn pipeline(tgt: crate::runtime::dtype::DType, name: &'static str) -> crate::err::Res<Pipeline> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        (tgt, name).hash(&mut hasher);
        let key = hasher.finish();
        let src = source(tgt);
        MetalDevice::get().compile(key, &src, name)
    }

    fn wrap(buf: Arc<crate::runtime::metal::device::Buffer>, n: usize, shape: Vec<usize>) -> crate::err::Res<MetalTensor> {
        let _ = n;
        Ok(MetalTensor {
            buffer: buf,
            layout: crate::runtime::layout::Layout::contiguous(shape),
            dtype: crate::runtime::dtype::DType::F32,
        })
    }

    fn dispatch_grid(encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>, width: usize, threads: usize) {
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            objc2_metal::MTLSize { width, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: threads, height: 1, depth: 1 },
        );
    }

    // Returns (loss scalar, status [3]) — the status is NOT read here:
    // checking it requires a device sync, which would split the walk's
    // encode pipeline. The evaluator defers it to the walk's end.
    pub fn ce_forward(logits: &MetalTensor, target: &MetalTensor, ignore_index: i64) -> crate::err::Res<(MetalTensor, MetalTensor)> {
        if std::env::var_os("EFFECT_TORCH_FUSION_DEBUG").is_some() {
            eprintln!("[ce] fused forward");
        }
        let rank = logits.layout.shape().len();
        let v = logits.layout.shape()[rank - 1];
        let n = logits.numel() / v;
        let tgt = target.dtype;
        let flat = |x: &MetalTensor, shape: Vec<usize>| MetalTensor {
            buffer: x.buffer.clone(),
            layout: crate::runtime::layout::Layout::contiguous(shape),
            dtype: x.dtype,
        };
        let logits = wrap_contig(&flat(logits, vec![n, v]))?;
        let target = wrap_contig(&flat(target, vec![n]))?;
        let nll_buf = alloc(n, crate::runtime::dtype::DType::F32);
        let flags_buf = alloc(n, crate::runtime::dtype::DType::U32);
        let status_buf = alloc(3, crate::runtime::dtype::DType::F32);
        {
            let pipe = pipeline(tgt, "et_ce_fwd")?;
            let f32_off = |off: usize| off * DType::F32.size_in_bytes();
            let tgt_off = |off: usize| off * tgt.size_in_bytes();
            MetalDevice::get().with_encoder(|e| {
                e.setComputePipelineState(pipe.as_raw());
                set_buffer(e, 0, &logits.buffer, f32_off(logits.layout.offset()));
                set_buffer(e, 1, &target.buffer, tgt_off(target.layout.offset()));
                set_buffer(e, 2, &nll_buf, 0);
                set_buffer(e, 3, &flags_buf, 0);
                set_bytes(e, 4, &(v as u32));
                set_bytes(e, 5, &ignore_index);
                dispatch_grid(e, n, NT);
            });
        }
        {
            let pipe = pipeline(tgt, "et_ce_status")?;
            MetalDevice::get().with_encoder(|e| {
                e.setComputePipelineState(pipe.as_raw());
                set_buffer(e, 0, &nll_buf, 0);
                set_buffer(e, 1, &flags_buf, 0);
                set_buffer(e, 2, &status_buf, 0);
                set_bytes(e, 3, &(n as u32));
                dispatch_grid(e, 1, NT);
            });
        }
        MetalDevice::get().synchronize();
        let status = wrap(status_buf, 3, vec![3])?;
        let loss = MetalTensor {
            buffer: status.buffer.clone(),
            layout: crate::runtime::layout::Layout::contiguous(vec![]),
            dtype: status.dtype,
        };
        Ok((loss, status))
    }

    // Returns (grad, count [1]) — the count check is deferred like the
    // forward's status (see ce_forward).
    pub fn ce_backward(logits: &MetalTensor, target: &MetalTensor, ignore_index: i64) -> crate::err::Res<(MetalTensor, MetalTensor)> {
        let rank = logits.layout.shape().len();
        let v = logits.layout.shape()[rank - 1];
        let n = logits.numel() / v;
        let out_shape = logits.layout.shape().to_vec();
        let tgt = target.dtype;
        let flat = |x: &MetalTensor, shape: Vec<usize>| MetalTensor {
            buffer: x.buffer.clone(),
            layout: crate::runtime::layout::Layout::contiguous(shape),
            dtype: x.dtype,
        };
        let logits = wrap_contig(&flat(logits, vec![n, v]))?;
        let target = wrap_contig(&flat(target, vec![n]))?;
        let _ = out_shape;
        let count_buf = alloc(1, crate::runtime::dtype::DType::F32);
        let grad_buf = alloc(n * v, crate::runtime::dtype::DType::F32);
        {
            let pipe = pipeline(tgt, "et_ce_count")?;
            MetalDevice::get().with_encoder(|e| {
                e.setComputePipelineState(pipe.as_raw());
                set_buffer(e, 0, &target.buffer, target.layout.offset() * tgt.size_in_bytes());
                set_buffer(e, 1, &count_buf, 0);
                set_bytes(e, 2, &(n as u32));
                set_bytes(e, 3, &ignore_index);
                dispatch_grid(e, 1, NT);
            });
        }
        {
            let pipe = pipeline(tgt, "et_ce_bwd")?;
            MetalDevice::get().with_encoder(|e| {
                e.setComputePipelineState(pipe.as_raw());
                set_buffer(e, 0, &logits.buffer, logits.layout.offset() * DType::F32.size_in_bytes());
                set_buffer(e, 1, &target.buffer, target.layout.offset() * tgt.size_in_bytes());
                set_buffer(e, 2, &count_buf, 0);
                set_buffer(e, 3, &grad_buf, 0);
                set_bytes(e, 4, &(v as u32));
                set_bytes(e, 5, &ignore_index);
                dispatch_grid(e, n, NT);
            });
        }
        MetalDevice::get().synchronize();
        // The zero-active check is deferred to the walk's end (see
        // ce_forward); the count buffer is returned for it.
        let count = wrap(count_buf.clone(), 1, vec![1])?;
        let grad = wrap(grad_buf, n * v, vec![n, v])?;
        let grad = MetalTensor {
            buffer: grad.buffer,
            layout: crate::runtime::layout::Layout::contiguous(out_shape),
            dtype: grad.dtype,
        };
        Ok((grad, count))
    }
}

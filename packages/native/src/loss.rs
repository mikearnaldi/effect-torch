//! Fused cross-entropy on Metal: the composed path (~30 ops with four
//! synchronous host readbacks for count/label checks) becomes one
//! kernel per row-block plus one tiny status read per direction.
//! Forward: per-row online logsumexp + nll in one pass, a single
//! status kernel (loss, active count, invalid count), one 12-byte
//! readback that preserves the exact error semantics. Backward:
//! device-side active count, then probs − one_hot in one pass — no
//! host round trip beyond the same zero-count check. CPU keeps the
//! composed reference path.

use candle_core::{DType, Device, Tensor};

/// Whether the fused CE path can run: Metal, f32 logits, integer targets.
pub fn is_supported(logits: &Tensor, target: &Tensor) -> bool {
    matches!(logits.device(), Device::Metal(_))
        && logits.dtype() == DType::F32
        && matches!(target.dtype(), DType::U32 | DType::I64)
}

#[cfg(target_os = "macos")]
pub use metal::{ce_backward, ce_forward};

#[cfg(not(target_os = "macos"))]
mod metal {
    use candle_core::Tensor;
    pub fn ce_forward(_l: &Tensor, _t: &Tensor, _i: i64) -> candle_core::Result<(Tensor, Tensor)> {
        unreachable!("fused cross-entropy is Metal-only")
    }
    pub fn ce_backward(_l: &Tensor, _t: &Tensor, _i: i64) -> candle_core::Result<(Tensor, Tensor)> {
        unreachable!("fused cross-entropy is Metal-only")
    }
}

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

    fn source(tgt: DType) -> String {
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

    fn pipeline(mdev: &candle_core::MetalDevice, tgt: DType, name: &'static str) -> candle_core::Result<ComputePipeline> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        (tgt, name).hash(&mut hasher);
        let key = hasher.finish();
        let mut cache = pipelines().lock().unwrap();
        if let Some(p) = cache.get(&key) {
            return Ok(p.clone());
        }
        let src = source(tgt);
        #[allow(deprecated)]
        let opts = {
            let o = objc2_metal::MTLCompileOptions::new();
            o.setFastMathEnabled(false);
            o
        };
        let lib = mdev
            .device()
            .new_library_with_source(&src, Some(&opts))
            .map_err(|e| candle_core::Error::Msg(format!("ce: {e}")))?;
        let func = lib
            .get_function(name, None)
            .map_err(|e| candle_core::Error::Msg(format!("ce: {e}")))?;
        let p = mdev
            .device()
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| candle_core::Error::Msg(format!("ce: {e}")))?;
        cache.insert(key, p.clone());
        Ok(p)
    }

    fn buffer_of(t: &Tensor) -> candle_core::Result<(candle_metal_kernels::metal::Buffer, usize)> {
        let (storage, layout) = t.storage_and_layout();
        match &*storage {
            Storage::Metal(m) => Ok((m.buffer().clone(), layout.start_offset())),
            _ => Err(candle_core::Error::Msg("ce: expected Metal storage".to_string())),
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

    fn dispatch_grid(encoder: &candle_metal_kernels::metal::ComputeCommandEncoder, width: usize, threads: usize) {
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize { width, height: 1, depth: 1 },
            objc2_metal::MTLSize { width: threads, height: 1, depth: 1 },
        );
    }

    // Returns (loss scalar, status [3]) — the status is NOT read here:
    // checking it requires a device sync, which would split the walk's
    // encode pipeline. The evaluator defers it to the walk's end.
    pub fn ce_forward(logits: &Tensor, target: &Tensor, ignore_index: i64) -> candle_core::Result<(Tensor, Tensor)> {
        if std::env::var_os("EFFECT_TORCH_FUSION_DEBUG").is_some() {
            eprintln!("[ce] fused forward");
        }
        let rank = logits.rank();
        let v = logits.dim(rank - 1)?;
        let n = logits.elem_count() / v;
        let device = logits.device();
        let mdev = device.as_metal_device()?;
        let tgt = target.dtype();
        let logits = logits.reshape((n, v))?.contiguous()?;
        let target = target.reshape(n)?.contiguous()?;
        let nll_buf = mdev.new_buffer(n, DType::F32, "ce_nll")?;
        let flags_buf = mdev.new_buffer(n, DType::U32, "ce_flags")?;
        let status_buf = mdev.new_buffer(3, DType::F32, "ce_status")?;
        {
            let encoder = mdev.command_encoder()?;
            let encoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&pipeline(mdev, tgt, "et_ce_fwd")?);
            let (zb, zo) = buffer_of(&logits)?;
            let (tb, to) = buffer_of(&target)?;
            let f32_off = |off: usize| off * DType::F32.size_in_bytes();
            let tgt_off = |off: usize| off * tgt.size_in_bytes();
            encoder.set_input_buffer(0, Some(&zb), f32_off(zo));
            encoder.set_input_buffer(1, Some(&tb), tgt_off(to));
            encoder.set_output_buffer(2, Some(&nll_buf), 0);
            encoder.set_output_buffer(3, Some(&flags_buf), 0);
            encoder.set_bytes(4, &(v as u32));
            encoder.set_bytes(5, &ignore_index);
            dispatch_grid(encoder, n, NT);
        }
        {
            let encoder = mdev.command_encoder()?;
            let encoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&pipeline(mdev, tgt, "et_ce_status")?);
            encoder.set_input_buffer(0, Some(&nll_buf), 0);
            encoder.set_input_buffer(1, Some(&flags_buf), 0);
            encoder.set_output_buffer(2, Some(&status_buf), 0);
            encoder.set_bytes(3, &(n as u32));
            dispatch_grid(encoder, 1, NT);
        }
        let status = wrap(status_buf, mdev, 3, vec![3]);
        let loss = status.narrow(0, 0, 1)?.reshape(())?;
        Ok((loss, status))
    }

    // Returns (grad, count [1]) — the count check is deferred like the
    // forward's status (see ce_forward).
    pub fn ce_backward(logits: &Tensor, target: &Tensor, ignore_index: i64) -> candle_core::Result<(Tensor, Tensor)> {
        let rank = logits.rank();
        let v = logits.dim(rank - 1)?;
        let n = logits.elem_count() / v;
        let out_shape = logits.dims().to_vec();
        let device = logits.device();
        let mdev = device.as_metal_device()?;
        let tgt = target.dtype();
        let logits = logits.reshape((n, v))?.contiguous()?;
        let target = target.reshape(n)?.contiguous()?;
        let count_buf = mdev.new_buffer(1, DType::F32, "ce_count")?;
        let grad_buf = mdev.new_buffer(n * v, DType::F32, "ce_grad")?;
        {
            let encoder = mdev.command_encoder()?;
            let encoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&pipeline(mdev, tgt, "et_ce_count")?);
            let (tb, to) = buffer_of(&target)?;
            encoder.set_input_buffer(0, Some(&tb), to * tgt.size_in_bytes());
            encoder.set_output_buffer(1, Some(&count_buf), 0);
            encoder.set_bytes(2, &(n as u32));
            encoder.set_bytes(3, &ignore_index);
            dispatch_grid(encoder, 1, NT);
        }
        {
            let encoder = mdev.command_encoder()?;
            let encoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&pipeline(mdev, tgt, "et_ce_bwd")?);
            let (zb, zo) = buffer_of(&logits)?;
            let (tb, to) = buffer_of(&target)?;
            encoder.set_input_buffer(0, Some(&zb), zo * DType::F32.size_in_bytes());
            encoder.set_input_buffer(1, Some(&tb), to * tgt.size_in_bytes());
            encoder.set_input_buffer(2, Some(&count_buf), 0);
            encoder.set_output_buffer(3, Some(&grad_buf), 0);
            encoder.set_bytes(4, &(v as u32));
            encoder.set_bytes(5, &ignore_index);
            dispatch_grid(encoder, n, NT);
        }
        // The zero-active check is deferred to the walk's end (see
        // ce_forward); the count buffer is returned for it.
        let count = wrap(count_buf.clone(), mdev, 1, vec![1]);
        let grad = wrap(grad_buf, mdev, n * v, vec![n, v]);
        Ok((grad.reshape(out_shape)?, count))
    }
}

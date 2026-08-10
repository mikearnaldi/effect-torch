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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferRequirement {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub elements: usize,
    pub bytes: usize,
}

impl BufferRequirement {
    fn new(shape: Vec<usize>, dtype: DType, label: &str) -> crate::err::Res<Self> {
        let elements = shape
            .iter()
            .try_fold(1usize, |total, dimension| total.checked_mul(*dimension));
        let elements = elements.ok_or_else(|| format!("cross_entropy: {label} size overflow"))?;
        let bytes = elements
            .max(1)
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| format!("cross_entropy: {label} byte size overflow"))?;
        Ok(Self {
            shape,
            dtype,
            elements,
            bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CeForwardTopology {
    RowsThenStatus { threads: usize, dispatches: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CeForwardRequirements {
    pub loss: BufferRequirement,
    pub status: BufferRequirement,
    pub nll_scratch: BufferRequirement,
    pub flags_scratch: BufferRequirement,
    pub topology: CeForwardTopology,
    pub rows: usize,
    pub classes: usize,
    pub logits_dtype: DType,
    pub target_dtype: DType,
    pub reduction: crate::CeReduction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CeBackwardTopology {
    Rows { threads: usize, dispatches: usize },
    CountThenRows { threads: usize, dispatches: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CeBackwardRequirements {
    pub grad: BufferRequirement,
    pub count_status: Option<BufferRequirement>,
    pub topology: CeBackwardTopology,
    pub rows: usize,
    pub classes: usize,
    pub logits_dtype: DType,
    pub target_dtype: DType,
    pub reduction: crate::CeReduction,
}

/// Whether the fused CE path can run: Metal, f32 logits, integer targets.
pub fn is_supported(logits: &MetalTensor, target: &MetalTensor) -> bool {
    matches!(logits.dtype, DType::F32 | DType::F16 | DType::BF16)
        && matches!(target.dtype, DType::U32 | DType::I64)
}

#[cfg(target_os = "macos")]
pub use metal::{
    ce_backward, ce_backward_into, ce_backward_requirements, ce_backward_scaled_f32_into,
    ce_forward, ce_forward_into, ce_forward_requirements, ce_target_status_into, warm_backward,
    warm_backward_exact, warm_backward_scaled_f32, warm_forward, warm_forward_exact,
    warm_target_status,
};

#[cfg(target_os = "macos")]
mod metal {
    use super::{
        BufferRequirement, CeBackwardRequirements, CeBackwardTopology, CeForwardRequirements,
        CeForwardTopology,
    };
    use crate::runtime::dtype::DType;
    use crate::runtime::metal::device::{set_buffer, set_bytes, MetalDevice, Pipeline};
    use crate::runtime::metal::run::MetalTensor;

    use objc2_metal::MTLComputeCommandEncoder;
    use std::sync::Arc;

    const NT: usize = 128;

    fn checked_numel(shape: &[usize], label: &str) -> crate::err::Res<usize> {
        shape
            .iter()
            .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
            .ok_or_else(|| format!("cross_entropy: {label} element count overflow"))
    }

    fn geometry(
        logits_shape: &[usize],
        logits_dtype: DType,
        target_shape: &[usize],
        target_dtype: DType,
    ) -> crate::err::Res<(usize, usize)> {
        if !matches!(logits_dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(format!(
                "cross_entropy: unsupported logits dtype {logits_dtype:?}"
            ));
        }
        if !matches!(target_dtype, DType::U32 | DType::I64) {
            return Err(format!(
                "cross_entropy: unsupported target dtype {target_dtype:?}"
            ));
        }
        let Some(&classes) = logits_shape.last() else {
            return Err("cross_entropy: logits must have rank >= 1".to_string());
        };
        if classes == 0 {
            return Err("cross_entropy: zero classes are unsupported".to_string());
        }
        let logits_elements = checked_numel(logits_shape, "logits")?;
        let rows = logits_elements / classes;
        if rows == 0 {
            return Err("cross_entropy: zero rows are unsupported".to_string());
        }
        let targets = checked_numel(target_shape, "target")?;
        if targets != rows {
            return Err(format!(
                "cross_entropy: target has {targets} elements for {rows} logits rows"
            ));
        }
        if rows > u32::MAX as usize || classes > u32::MAX as usize {
            return Err("cross_entropy: rows and classes must fit u32".to_string());
        }
        Ok((rows, classes))
    }

    pub fn ce_forward_requirements(
        logits_shape: &[usize],
        logits_dtype: DType,
        target_shape: &[usize],
        target_dtype: DType,
        reduction: crate::CeReduction,
    ) -> crate::err::Res<CeForwardRequirements> {
        let (rows, classes) = geometry(logits_shape, logits_dtype, target_shape, target_dtype)?;
        Ok(CeForwardRequirements {
            loss: BufferRequirement::new(Vec::new(), DType::F32, "loss")?,
            status: BufferRequirement::new(vec![3], DType::F32, "status")?,
            nll_scratch: BufferRequirement::new(vec![rows], DType::F32, "nll scratch")?,
            flags_scratch: BufferRequirement::new(vec![rows], DType::U32, "flags scratch")?,
            topology: CeForwardTopology::RowsThenStatus {
                threads: NT,
                dispatches: 2,
            },
            rows,
            classes,
            logits_dtype,
            target_dtype,
            reduction,
        })
    }

    pub fn ce_backward_requirements(
        logits_shape: &[usize],
        logits_dtype: DType,
        target_shape: &[usize],
        target_dtype: DType,
        reduction: crate::CeReduction,
    ) -> crate::err::Res<CeBackwardRequirements> {
        let (rows, classes) = geometry(logits_shape, logits_dtype, target_shape, target_dtype)?;
        let (count_status, topology) = match reduction {
            crate::CeReduction::Mean => (
                Some(BufferRequirement::new(vec![1], DType::F32, "count status")?),
                CeBackwardTopology::CountThenRows {
                    threads: NT,
                    dispatches: 2,
                },
            ),
            crate::CeReduction::Sum => (
                None,
                CeBackwardTopology::Rows {
                    threads: NT,
                    dispatches: 1,
                },
            ),
        };
        Ok(CeBackwardRequirements {
            grad: BufferRequirement::new(logits_shape.to_vec(), logits_dtype, "gradient")?,
            count_status,
            topology,
            rows,
            classes,
            logits_dtype,
            target_dtype,
            reduction,
        })
    }

    fn require_contiguous(tensor: &MetalTensor, label: &str) -> crate::err::Res<()> {
        if !tensor.layout.is_contiguous() {
            return Err(format!("cross_entropy: {label} must be contiguous"));
        }
        let end = tensor
            .layout
            .offset()
            .checked_add(tensor.numel())
            .and_then(|elements| elements.checked_mul(tensor.dtype.size_in_bytes()))
            .ok_or_else(|| format!("cross_entropy: {label} storage range overflow"))?;
        if end > tensor.buffer.size {
            return Err(format!("cross_entropy: {label} storage is too small"));
        }
        Ok(())
    }

    fn require_exact(
        tensor: &MetalTensor,
        shape: &[usize],
        dtype: DType,
        label: &str,
    ) -> crate::err::Res<()> {
        if tensor.layout.shape() != shape || tensor.dtype != dtype {
            return Err(format!(
                "cross_entropy: {label} must be {shape:?}:{dtype:?}, got {:?}:{:?}",
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

    fn alloc(
        n: usize,
        dtype: crate::runtime::dtype::DType,
    ) -> Arc<crate::runtime::metal::device::Buffer> {
        #[cfg(test)]
        super::record_device_allocation();
        MetalDevice::get().alloc(n.max(1), dtype)
    }

    fn source(tgt: crate::runtime::dtype::DType, z: crate::runtime::dtype::DType) -> String {
        let tgt_ty = match tgt {
            DType::U32 => "uint",
            DType::I64 => "long",
            other => unreachable!("ce: unsupported target dtype {other:?}"),
        };
        let z_ty = match z {
            DType::F32 => "float",
            DType::F16 => "half",
            DType::BF16 => "bfloat",
            other => unreachable!("ce: unsupported logits dtype {other:?}"),
        };
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;

#define NT {nt}
#define TGT {tgt_ty}
#define ZS {z_ty}

// One threadgroup per row: online (max, sumexp) over the row's V
// logits, then nll and status flags. flags bit0 = ignored,
// bit1 = invalid-but-active target.
kernel void et_ce_fwd(
    device const ZS* Z [[buffer(0)]],
    device const TGT* T [[buffer(1)]],
    device float* nll [[buffer(2)]],
    device uint* flags [[buffer(3)]],
    constant uint& V [[buffer(4)]],
    constant long& ignore [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {{
    const uint row = tgid;
    device const ZS* z = Z + (ulong)row * V;
    const long t = (long)T[row];
    const bool ignored = (t == ignore);
    const bool invalid = !ignored && (t < 0 || t >= (long)V);
    float m = -INFINITY;
    float l = 0.0f;
    for (uint j = tid; j < V; j += NT) {{
        const float v = float(z[j]);
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

// Single threadgroup: loss = sum(nll) (/ active when mean), plus active
// and invalid counts for the host-side error semantics.
kernel void et_ce_status(
    device const float* nll [[buffer(0)]],
    device const uint* flags [[buffer(1)]],
    device float* loss [[buffer(2)]],
    device float* status [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& mean [[buffer(5)]],
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
        const float result = (mean != 0u) ? (ta > 0 ? ts / float(ta) : 0.0f) : ts;
        loss[0] = result;
        status[0] = result;
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

// Active and invalid target counts without touching logits. Chunked-head
// backward uses this once before recomputing each logits chunk.
kernel void et_ce_target_status(
    device const TGT* T [[buffer(0)]],
    device float* status [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    constant uint& V [[buffer(3)]],
    constant long& ignore [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]]
) {{
    uint active = 0;
    uint invalid = 0;
    for (uint i = tid; i < N; i += NT) {{
        const long t = (long)T[i];
        const bool ignored = (t == ignore);
        active += ignored ? 0u : 1u;
        invalid += (!ignored && (t < 0 || t >= (long)V)) ? 1u : 0u;
    }}
    active = simd_sum(active);
    invalid = simd_sum(invalid);
    threadgroup uint pa[NT / 32];
    threadgroup uint pi[NT / 32];
    const uint lane = tid % 32;
    const uint grp = tid / 32;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {{ pa[grp] = active; pi[grp] = invalid; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {{
        uint ta = 0, ti = 0;
        for (uint g = 0; g < NT / 32; g++) {{ ta += pa[g]; ti += pi[g]; }}
        status[0] = 0.0f;
        status[1] = float(ta);
        status[2] = float(ti);
    }}
}}

// grad = (softmax(z) - one_hot(t)) (/ count when mean) for active rows,
// zeros where ignored. One threadgroup per row, one pass.
kernel void et_ce_bwd(
    device const ZS* Z [[buffer(0)]],
    device const TGT* T [[buffer(1)]],
    device const float* count [[buffer(2)]],
    device ZS* G [[buffer(3)]],
    constant uint& V [[buffer(4)]],
    constant long& ignore [[buffer(5)]],
    constant uint& mean [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {{
    const uint row = tgid;
    device const ZS* z = Z + (ulong)row * V;
    device ZS* g = G + (ulong)row * V;
    const long t = (long)T[row];
    if (t == ignore) {{
        for (uint j = tid; j < V; j += NT) {{ g[j] = ZS(0.0f); }}
        return;
    }}
    float m = -INFINITY;
    float l = 0.0f;
    for (uint j = tid; j < V; j += NT) {{
        const float v = float(z[j]);
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
    const float inv = (mean != 0u) ? 1.0f / count[0] : 1.0f;
    for (uint j = tid; j < V; j += NT) {{
        const float p = exp(float(z[j]) - lse);
        g[j] = ZS((p - ((long)j == t ? 1.0f : 0.0f)) * inv);
    }}
}}

// Chunked LM-head backward consumes f32 gradients for its GEMMs. Preserve the
// logits-dtype rounding point while writing the scaled f32 value directly,
// avoiding separate full-vocabulary cast and multiply passes.
kernel void et_ce_bwd_scaled_f32(
    device const ZS* Z [[buffer(0)]],
    device const TGT* T [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    device float* G [[buffer(3)]],
    constant uint& V [[buffer(4)]],
    constant long& ignore [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {{
    const uint row = tgid;
    device const ZS* z = Z + (ulong)row * V;
    device float* g = G + (ulong)row * V;
    const long t = (long)T[row];
    if (t == ignore) {{
        for (uint j = tid; j < V; j += NT) {{ g[j] = 0.0f; }}
        return;
    }}
    float m = -INFINITY;
    float l = 0.0f;
    for (uint j = tid; j < V; j += NT) {{
        const float v = float(z[j]);
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
        for (uint group = 0; group < NT / 32; group++) {{ fm = max(fm, pm[group]); }}
        float fl = 0.0f;
        for (uint group = 0; group < NT / 32; group++) {{ fl += pl[group] * exp(pm[group] - fm); }}
        lse = fm + log(fl);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float s = scale[0];
    for (uint j = tid; j < V; j += NT) {{
        const float p = exp(float(z[j]) - lse);
        const float delta = p - ((long)j == t ? 1.0f : 0.0f);
        g[j] = float(ZS(delta)) * s;
    }}
}}
"#,
            nt = NT,
            tgt_ty = tgt_ty,
            z_ty = z_ty,
        )
    }

    fn pipeline_key(
        tgt: crate::runtime::dtype::DType,
        z: crate::runtime::dtype::DType,
        name: &'static str,
    ) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        (tgt, z, name).hash(&mut hasher);
        hasher.finish()
    }

    fn pipeline(
        tgt: crate::runtime::dtype::DType,
        z: crate::runtime::dtype::DType,
        name: &'static str,
    ) -> crate::err::Res<Pipeline> {
        #[cfg(test)]
        super::record_pipeline_request();
        MetalDevice::get().compile_lazy(pipeline_key(tgt, z, name), name, || source(tgt, z))
    }

    fn cached_pipeline(
        tgt: crate::runtime::dtype::DType,
        z: crate::runtime::dtype::DType,
        name: &'static str,
    ) -> crate::err::Res<Pipeline> {
        MetalDevice::get()
            .pipeline_cached(pipeline_key(tgt, z, name))
            .ok_or_else(|| {
                format!("cross_entropy: {name} pipeline is not warm; call the exact warm function")
            })
    }

    pub fn warm_forward(logits: DType, target: DType) -> crate::err::Res<()> {
        geometry(&[1, 1], logits, &[1], target)?;
        pipeline(target, logits, "et_ce_fwd")?;
        pipeline(target, logits, "et_ce_status")?;
        Ok(())
    }

    pub fn warm_forward_exact(requirements: &CeForwardRequirements) -> crate::err::Res<()> {
        pipeline(
            requirements.target_dtype,
            requirements.logits_dtype,
            "et_ce_fwd",
        )?;
        pipeline(
            requirements.target_dtype,
            requirements.logits_dtype,
            "et_ce_status",
        )?;
        Ok(())
    }

    pub fn warm_backward(
        logits: DType,
        target: DType,
        reduction: crate::CeReduction,
    ) -> crate::err::Res<()> {
        geometry(&[1, 1], logits, &[1], target)?;
        if reduction == crate::CeReduction::Mean {
            pipeline(target, logits, "et_ce_count")?;
        }
        pipeline(target, logits, "et_ce_bwd")?;
        Ok(())
    }

    pub fn warm_backward_exact(requirements: &CeBackwardRequirements) -> crate::err::Res<()> {
        if requirements.reduction == crate::CeReduction::Mean {
            pipeline(
                requirements.target_dtype,
                requirements.logits_dtype,
                "et_ce_count",
            )?;
        }
        pipeline(
            requirements.target_dtype,
            requirements.logits_dtype,
            "et_ce_bwd",
        )?;
        Ok(())
    }

    pub fn warm_backward_scaled_f32(logits: DType, target: DType) -> crate::err::Res<()> {
        geometry(&[1, 1], logits, &[1], target)?;
        pipeline(target, logits, "et_ce_bwd_scaled_f32")?;
        Ok(())
    }

    pub fn warm_target_status(logits: DType, target: DType) -> crate::err::Res<()> {
        geometry(&[1, 1], logits, &[1], target)?;
        pipeline(target, logits, "et_ce_target_status")?;
        Ok(())
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

    fn dispatch_grid(
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        width: usize,
        threads: usize,
    ) {
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            objc2_metal::MTLSize {
                width,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ce_forward_into(
        logits: &MetalTensor,
        target: &MetalTensor,
        ignore_index: i64,
        reduction: crate::CeReduction,
        loss: &MetalTensor,
        status: &MetalTensor,
        nll_scratch: &MetalTensor,
        flags_scratch: &MetalTensor,
    ) -> crate::err::Res<()> {
        let (n, v) = geometry(
            logits.layout.shape(),
            logits.dtype,
            target.layout.shape(),
            target.dtype,
        )?;
        require_contiguous(logits, "logits")?;
        require_contiguous(target, "target")?;
        require_exact(loss, &[], DType::F32, "loss")?;
        require_exact(status, &[3], DType::F32, "status")?;
        require_exact(nll_scratch, &[n], DType::F32, "nll scratch")?;
        require_exact(flags_scratch, &[n], DType::U32, "flags scratch")?;
        for tensor in [loss, status, nll_scratch, flags_scratch] {
            MetalDevice::get().mark_buffer_write(&tensor.buffer)?;
        }

        let forward_pipe = cached_pipeline(target.dtype, logits.dtype, "et_ce_fwd")?;
        let status_pipe = cached_pipeline(target.dtype, logits.dtype, "et_ce_status")?;
        let logits_size = logits.dtype.size_in_bytes();
        let target_size = target.dtype.size_in_bytes();
        MetalDevice::get().with_encoder(|e| {
            e.setComputePipelineState(forward_pipe.as_raw());
            set_buffer(e, 0, &logits.buffer, logits.layout.offset() * logits_size);
            set_buffer(e, 1, &target.buffer, target.layout.offset() * target_size);
            set_buffer(
                e,
                2,
                &nll_scratch.buffer,
                nll_scratch.layout.offset() * DType::F32.size_in_bytes(),
            );
            set_buffer(
                e,
                3,
                &flags_scratch.buffer,
                flags_scratch.layout.offset() * DType::U32.size_in_bytes(),
            );
            set_bytes(e, 4, &(v as u32));
            set_bytes(e, 5, &ignore_index);
            dispatch_grid(e, n, NT);
        });
        let mean: u32 = match reduction {
            crate::CeReduction::Mean => 1,
            crate::CeReduction::Sum => 0,
        };
        MetalDevice::get().with_encoder(|e| {
            e.setComputePipelineState(status_pipe.as_raw());
            set_buffer(
                e,
                0,
                &nll_scratch.buffer,
                nll_scratch.layout.offset() * DType::F32.size_in_bytes(),
            );
            set_buffer(
                e,
                1,
                &flags_scratch.buffer,
                flags_scratch.layout.offset() * DType::U32.size_in_bytes(),
            );
            set_buffer(e, 2, &loss.buffer, loss.layout.offset() * 4);
            set_buffer(e, 3, &status.buffer, status.layout.offset() * 4);
            set_bytes(e, 4, &(n as u32));
            set_bytes(e, 5, &mean);
            dispatch_grid(e, 1, NT);
        });
        Ok(())
    }

    // Returns (loss scalar, status [3]) without reading status. Validation
    // remains deferred to the evaluator's existing status gate.
    pub fn ce_forward(
        logits: &MetalTensor,
        target: &MetalTensor,
        ignore_index: i64,
        reduction: crate::CeReduction,
    ) -> crate::err::Res<(MetalTensor, MetalTensor)> {
        if std::env::var_os("EFFECT_TORCH_FUSION_DEBUG").is_some() {
            eprintln!("[ce] fused forward");
        }
        let requirements = ce_forward_requirements(
            logits.layout.shape(),
            logits.dtype,
            target.layout.shape(),
            target.dtype,
            reduction,
        )?;
        warm_forward_exact(&requirements)?;
        let logits = wrap_contig(logits)?;
        let target = wrap_contig(target)?;
        let status = wrap(
            alloc(requirements.status.elements, DType::F32),
            requirements.status.shape.clone(),
            DType::F32,
        )?;
        let loss = MetalTensor {
            buffer: status.buffer.clone(),
            layout: crate::runtime::layout::Layout::contiguous(requirements.loss.shape.clone()),
            dtype: DType::F32,
        };
        let nll_scratch = wrap(
            alloc(requirements.nll_scratch.elements, DType::F32),
            requirements.nll_scratch.shape.clone(),
            DType::F32,
        )?;
        let flags_scratch = wrap(
            alloc(requirements.flags_scratch.elements, DType::U32),
            requirements.flags_scratch.shape.clone(),
            DType::U32,
        )?;
        ce_forward_into(
            &logits,
            &target,
            ignore_index,
            reduction,
            &loss,
            &status,
            &nll_scratch,
            &flags_scratch,
        )?;
        Ok((loss, status))
    }

    pub fn ce_backward_into(
        logits: &MetalTensor,
        target: &MetalTensor,
        ignore_index: i64,
        reduction: crate::CeReduction,
        grad: &MetalTensor,
        count_status: Option<&MetalTensor>,
    ) -> crate::err::Res<()> {
        let (n, v) = geometry(
            logits.layout.shape(),
            logits.dtype,
            target.layout.shape(),
            target.dtype,
        )?;
        require_contiguous(logits, "logits")?;
        require_contiguous(target, "target")?;
        require_exact(grad, logits.layout.shape(), logits.dtype, "gradient")?;
        match (reduction, count_status) {
            (crate::CeReduction::Mean, Some(status)) => {
                require_exact(status, &[1], DType::F32, "count status")?
            }
            (crate::CeReduction::Mean, None) => {
                return Err("cross_entropy: mean backward requires count status".to_string())
            }
            (crate::CeReduction::Sum, Some(_)) => {
                return Err("cross_entropy: sum backward has no status buffer".to_string())
            }
            (crate::CeReduction::Sum, None) => {}
        }
        MetalDevice::get().mark_buffer_write(&grad.buffer)?;
        if let Some(status) = count_status {
            MetalDevice::get().mark_buffer_write(&status.buffer)?;
        }

        let count_pipe = if reduction == crate::CeReduction::Mean {
            Some(cached_pipeline(target.dtype, logits.dtype, "et_ce_count")?)
        } else {
            None
        };
        let backward_pipe = cached_pipeline(target.dtype, logits.dtype, "et_ce_bwd")?;
        let mean: u32 = match reduction {
            crate::CeReduction::Mean => 1,
            crate::CeReduction::Sum => 0,
        };
        if let (Some(pipe), Some(status)) = (count_pipe, count_status) {
            MetalDevice::get().with_encoder(|e| {
                e.setComputePipelineState(pipe.as_raw());
                set_buffer(
                    e,
                    0,
                    &target.buffer,
                    target.layout.offset() * target.dtype.size_in_bytes(),
                );
                set_buffer(e, 1, &status.buffer, status.layout.offset() * 4);
                set_bytes(e, 2, &(n as u32));
                set_bytes(e, 3, &ignore_index);
                dispatch_grid(e, 1, NT);
            });
        }
        MetalDevice::get().with_encoder(|e| {
            e.setComputePipelineState(backward_pipe.as_raw());
            set_buffer(
                e,
                0,
                &logits.buffer,
                logits.layout.offset() * logits.dtype.size_in_bytes(),
            );
            set_buffer(
                e,
                1,
                &target.buffer,
                target.layout.offset() * target.dtype.size_in_bytes(),
            );
            if let Some(status) = count_status {
                set_buffer(e, 2, &status.buffer, status.layout.offset() * 4);
            } else {
                // The sum-specialized runtime branch does not read count.
                set_buffer(
                    e,
                    2,
                    &grad.buffer,
                    grad.layout.offset() * grad.dtype.size_in_bytes(),
                );
            }
            set_buffer(
                e,
                3,
                &grad.buffer,
                grad.layout.offset() * grad.dtype.size_in_bytes(),
            );
            set_bytes(e, 4, &(v as u32));
            set_bytes(e, 5, &ignore_index);
            set_bytes(e, 6, &mean);
            dispatch_grid(e, n, NT);
        });
        Ok(())
    }

    pub fn ce_target_status_into(
        target: &MetalTensor,
        logits_dtype: DType,
        classes: usize,
        ignore_index: i64,
        status: &MetalTensor,
    ) -> crate::err::Res<()> {
        if !matches!(target.dtype, DType::U32 | DType::I64) {
            return Err(format!(
                "cross_entropy: unsupported target dtype {:?}",
                target.dtype
            ));
        }
        if !matches!(logits_dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(format!(
                "cross_entropy: unsupported logits dtype {logits_dtype:?}"
            ));
        }
        let rows = checked_numel(target.layout.shape(), "target")?;
        if rows == 0 || rows > u32::MAX as usize || classes == 0 || classes > u32::MAX as usize {
            return Err("cross_entropy: rows and classes must be nonzero and fit u32".to_string());
        }
        require_contiguous(target, "target")?;
        require_exact(status, &[3], DType::F32, "status")?;
        MetalDevice::get().mark_buffer_write(&status.buffer)?;
        let target_size = target.dtype.size_in_bytes();
        let pipe = cached_pipeline(target.dtype, logits_dtype, "et_ce_target_status")?;
        MetalDevice::get().with_encoder(|e| {
            e.setComputePipelineState(pipe.as_raw());
            set_buffer(e, 0, &target.buffer, target.layout.offset() * target_size);
            set_buffer(e, 1, &status.buffer, status.layout.offset() * 4);
            set_bytes(e, 2, &(rows as u32));
            set_bytes(e, 3, &(classes as u32));
            set_bytes(e, 4, &ignore_index);
            dispatch_grid(e, 1, NT);
        });
        Ok(())
    }

    pub fn ce_backward_scaled_f32_into(
        logits: &MetalTensor,
        target: &MetalTensor,
        ignore_index: i64,
        scale: &MetalTensor,
        grad: &MetalTensor,
    ) -> crate::err::Res<()> {
        let (rows, classes) = geometry(
            logits.layout.shape(),
            logits.dtype,
            target.layout.shape(),
            target.dtype,
        )?;
        require_contiguous(logits, "logits")?;
        require_contiguous(target, "target")?;
        require_exact(scale, &[1], DType::F32, "scale")?;
        require_exact(grad, logits.layout.shape(), DType::F32, "scaled gradient")?;
        MetalDevice::get().mark_buffer_write(&grad.buffer)?;
        let pipe = cached_pipeline(target.dtype, logits.dtype, "et_ce_bwd_scaled_f32")?;
        MetalDevice::get().with_encoder(|encoder| {
            encoder.setComputePipelineState(pipe.as_raw());
            set_buffer(
                encoder,
                0,
                &logits.buffer,
                logits.layout.offset() * logits.dtype.size_in_bytes(),
            );
            set_buffer(
                encoder,
                1,
                &target.buffer,
                target.layout.offset() * target.dtype.size_in_bytes(),
            );
            set_buffer(encoder, 2, &scale.buffer, scale.layout.offset() * 4);
            set_buffer(encoder, 3, &grad.buffer, grad.layout.offset() * 4);
            set_bytes(encoder, 4, &(classes as u32));
            set_bytes(encoder, 5, &ignore_index);
            dispatch_grid(encoder, rows, NT);
        });
        Ok(())
    }

    // Returns (grad, count [1]); only mean reduction writes and checks count.
    pub fn ce_backward(
        logits: &MetalTensor,
        target: &MetalTensor,
        ignore_index: i64,
        reduction: crate::CeReduction,
    ) -> crate::err::Res<(MetalTensor, MetalTensor)> {
        let requirements = ce_backward_requirements(
            logits.layout.shape(),
            logits.dtype,
            target.layout.shape(),
            target.dtype,
            reduction,
        )?;
        warm_backward_exact(&requirements)?;
        let logits = wrap_contig(logits)?;
        let target = wrap_contig(target)?;
        let grad = wrap(
            alloc(requirements.grad.elements, requirements.grad.dtype),
            requirements.grad.shape.clone(),
            requirements.grad.dtype,
        )?;
        // Preserve the allocating wrapper's historical pair return for sum;
        // the exact contract correctly declares that buffer only for mean.
        let count = wrap(alloc(1, DType::F32), vec![1], DType::F32)?;
        ce_backward_into(
            &logits,
            &target,
            ignore_index,
            reduction,
            &grad,
            (reduction == crate::CeReduction::Mean).then_some(&count),
        )?;
        Ok((grad, count))
    }
}

#[cfg(test)]
mod tests {
    use super::{CeBackwardTopology, CeForwardTopology};
    use crate::runtime::cpu::{CpuBuffer, Tensor};
    use crate::runtime::dtype::DType;
    use crate::runtime::metal::device::MetalDevice;
    use crate::runtime::metal::run::MetalTensor;

    fn target(values: &[u32], shape: Vec<usize>) -> MetalTensor {
        MetalTensor {
            buffer: MetalDevice::get().alloc_with_data_u32(values),
            layout: crate::runtime::layout::Layout::contiguous(shape),
            dtype: DType::U32,
        }
    }

    fn cpu_f32(tensor: &Tensor) -> &[f32] {
        let CpuBuffer::F32(values) = &tensor.buffer else {
            panic!("expected f32 tensor")
        };
        values
    }

    #[test]
    fn requirements_include_status_scratch_and_selected_topology() {
        let forward = crate::loss::ce_forward_requirements(
            &[2, 3, 7],
            DType::BF16,
            &[2, 3],
            DType::U32,
            crate::CeReduction::Mean,
        )
        .unwrap();
        assert!(forward.loss.shape.is_empty());
        assert_eq!(forward.status.shape, [3]);
        assert_eq!(forward.status.bytes, 12);
        assert_eq!(forward.nll_scratch.shape, [6]);
        assert_eq!(forward.flags_scratch.shape, [6]);
        assert_eq!(forward.flags_scratch.dtype, DType::U32);
        assert_eq!(
            forward.topology,
            CeForwardTopology::RowsThenStatus {
                threads: 128,
                dispatches: 2,
            }
        );

        let mean = crate::loss::ce_backward_requirements(
            &[2, 3, 7],
            DType::BF16,
            &[2, 3],
            DType::U32,
            crate::CeReduction::Mean,
        )
        .unwrap();
        assert_eq!(mean.grad.shape, [2, 3, 7]);
        assert_eq!(mean.count_status.unwrap().shape, [1]);
        assert_eq!(
            mean.topology,
            CeBackwardTopology::CountThenRows {
                threads: 128,
                dispatches: 2,
            }
        );

        let sum = crate::loss::ce_backward_requirements(
            &[2, 3, 7],
            DType::BF16,
            &[2, 3],
            DType::U32,
            crate::CeReduction::Sum,
        )
        .unwrap();
        assert!(sum.count_status.is_none());
        assert_eq!(
            sum.topology,
            CeBackwardTopology::Rows {
                threads: 128,
                dispatches: 1,
            }
        );
    }

    #[test]
    fn into_matches_allocating_and_cpu_without_allocating() {
        let dev = MetalDevice::get();
        let logits_values = vec![
            0.2, -0.4, 1.1, 0.7, -1.0, 0.3, 0.8, -0.2, 1.4, 0.5, -0.7, 0.1,
        ];
        let target_values = vec![2u32, 99, 1];
        let logits = MetalTensor::from_f32(dev, logits_values.clone(), vec![3, 4]);
        let target = target(&target_values, vec![3]);
        let reduction = crate::CeReduction::Mean;
        let forward_req = crate::loss::ce_forward_requirements(
            logits.layout.shape(),
            logits.dtype,
            target.layout.shape(),
            target.dtype,
            reduction,
        )
        .unwrap();
        crate::loss::warm_forward_exact(&forward_req).unwrap();
        let loss = MetalTensor::empty(dev, vec![], DType::F32);
        let status = MetalTensor::empty(dev, vec![3], DType::F32);
        let nll = MetalTensor::empty(dev, forward_req.nll_scratch.shape.clone(), DType::F32);
        let flags = MetalTensor::empty(dev, forward_req.flags_scratch.shape.clone(), DType::U32);
        super::reset_test_counts();
        crate::loss::ce_forward_into(
            &logits, &target, 99, reduction, &loss, &status, &nll, &flags,
        )
        .unwrap();
        assert_eq!(super::test_counts(), (0, 0));

        let backward_req = crate::loss::ce_backward_requirements(
            logits.layout.shape(),
            logits.dtype,
            target.layout.shape(),
            target.dtype,
            reduction,
        )
        .unwrap();
        crate::loss::warm_backward_exact(&backward_req).unwrap();
        let grad = MetalTensor::empty(dev, backward_req.grad.shape.clone(), DType::F32);
        let count = MetalTensor::empty(dev, vec![1], DType::F32);
        super::reset_test_counts();
        crate::loss::ce_backward_into(&logits, &target, 99, reduction, &grad, Some(&count))
            .unwrap();
        assert_eq!(super::test_counts(), (0, 0));

        let (allocated_loss, _) = crate::loss::ce_forward(&logits, &target, 99, reduction).unwrap();
        let (allocated_grad, _) =
            crate::loss::ce_backward(&logits, &target, 99, reduction).unwrap();
        dev.synchronize().unwrap();

        let cpu_logits = Tensor::from_vec(logits_values, vec![3, 4]);
        let cpu_target = Tensor::from_vec(target_values, vec![3]);
        let cpu_loss = crate::runtime::cpu::composed::cross_entropy_forward(
            &cpu_logits,
            &cpu_target,
            99,
            reduction,
        )
        .unwrap();
        let cpu_grad = crate::runtime::cpu::composed::cross_entropy_backward(
            &cpu_logits,
            &cpu_target,
            99,
            reduction,
        )
        .unwrap();
        let actual_loss = loss.read_f32().unwrap()[0];
        let expected_loss = cpu_f32(&cpu_loss.contiguous())[0];
        assert!((actual_loss - expected_loss).abs() < 1e-5);
        assert!((allocated_loss.read_f32().unwrap()[0] - actual_loss).abs() < 1e-6);
        let status_values = status.read_f32().unwrap();
        assert_eq!(status_values[1], 2.0);
        assert_eq!(status_values[2], 0.0);
        assert_eq!(count.read_f32().unwrap(), vec![2.0]);
        let actual_grad = grad.read_f32().unwrap();
        let allocated_grad = allocated_grad.read_f32().unwrap();
        for ((actual, allocated), expected) in actual_grad
            .iter()
            .zip(&allocated_grad)
            .zip(cpu_f32(&cpu_grad.contiguous()))
        {
            assert!((actual - expected).abs() < 1e-5);
            assert!((actual - allocated).abs() < 1e-6);
        }
    }

    #[test]
    fn supplied_forward_status_preserves_deferred_invalid_label_check() {
        let dev = MetalDevice::get();
        let logits = MetalTensor::from_f32(dev, vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]);
        let target = target(&[0, 7], vec![2]);
        let requirements = crate::loss::ce_forward_requirements(
            logits.layout.shape(),
            logits.dtype,
            target.layout.shape(),
            target.dtype,
            crate::CeReduction::Sum,
        )
        .unwrap();
        crate::loss::warm_forward_exact(&requirements).unwrap();
        let loss = MetalTensor::empty(dev, vec![], DType::F32);
        let status = MetalTensor::empty(dev, vec![3], DType::F32);
        let nll = MetalTensor::empty(dev, vec![2], DType::F32);
        let flags = MetalTensor::empty(dev, vec![2], DType::U32);
        crate::loss::ce_forward_into(
            &logits,
            &target,
            -1,
            crate::CeReduction::Sum,
            &loss,
            &status,
            &nll,
            &flags,
        )
        .unwrap();
        assert_eq!(status.read_f32().unwrap()[2], 1.0);
    }

    #[test]
    fn target_status_counts_active_and_invalid_labels_without_allocating() {
        let dev = MetalDevice::get();
        let target = target(&[0, 99, 7], vec![3]);
        let status = MetalTensor::empty(dev, vec![3], DType::F32);
        crate::loss::warm_target_status(DType::BF16, DType::U32).unwrap();
        super::reset_test_counts();
        crate::loss::ce_target_status_into(&target, DType::BF16, 4, 99, &status).unwrap();
        assert_eq!(super::test_counts(), (0, 0));
        assert_eq!(status.read_f32().unwrap(), vec![0.0, 2.0, 1.0]);
    }
}

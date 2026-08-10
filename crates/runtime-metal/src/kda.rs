//! Fused KDA recurrent decode on Metal (RFC 0018, phase 3): one kernel
//! launch per (sequence slot, layer) advances the gated delta-rule state
//! by one token — S = Diag(alpha) S + beta k (v - (Diag(alpha) S)^T k)^T,
//! o = scale * S^T q — with the [Dk, Dv] fp32 state distributed across
//! threadgroup registers (each of 32 lanes holds ceil(Dk/32) rows for one
//! of 4 value columns), simd_sum reductions for the k^T S and S^T q
//! contractions, and the state read and written in place. This replaces
//! the ~45-launch primitive reference for the T=1 decode step. All head dims are supported:
//! out-of-range lanes/rows are masked off (the project rule is to fail
//! loud on genuinely unsupported input, never to degrade silently).

use crate::runtime::dtype::DType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForwardRequirements {
    pub dtype: DType,
    pub batch_heads: usize,
    pub steps: usize,
    pub dk: usize,
    pub dv: usize,
    pub scale_bits: u64,
    pub output_bytes: usize,
    pub state_next_bytes: usize,
    pub staging_bytes: usize,
    pub status_bytes: usize,
    pub scratch_bytes: usize,
    pub pipeline_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackwardRequirements {
    pub dtype: DType,
    pub batch_heads: usize,
    pub steps: usize,
    pub dk: usize,
    pub dv: usize,
    pub scale_bits: u64,
    pub dq_bytes: usize,
    pub dk_bytes: usize,
    pub dv_bytes: usize,
    pub dg_bytes: usize,
    pub db_bytes: usize,
    pub staging_bytes: usize,
    pub status_bytes: usize,
    pub scratch_bytes: usize,
    pub pipeline_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeRequirements {
    pub dtype: DType,
    pub heads: usize,
    pub dk: usize,
    pub dv: usize,
    pub scale_bits: u64,
    pub output_bytes: usize,
    pub state_next_bytes: usize,
    pub staging_bytes: usize,
    pub status_bytes: usize,
    pub scratch_bytes: usize,
    pub pipeline_count: usize,
}

fn checked_bytes(elements: &[usize], dtype: DType, operation: &str) -> crate::err::Res<usize> {
    elements
        .iter()
        .try_fold(1usize, |count, value| count.checked_mul(*value))
        .and_then(|count| count.checked_mul(dtype.size_in_bytes()))
        .ok_or_else(|| format!("{operation}: requirement byte size overflow"))
}

pub fn forward_requirements(
    dtype: DType,
    batch_heads: usize,
    steps: usize,
    dk: usize,
    dv: usize,
    scale: f64,
    write_state_next: bool,
) -> crate::err::Res<ForwardRequirements> {
    if !supported_dtype(dtype) {
        return Err(format!("kda forward: unsupported dtype {dtype:?}"));
    }
    Ok(ForwardRequirements {
        dtype,
        batch_heads,
        steps,
        dk,
        dv,
        scale_bits: scale.to_bits(),
        output_bytes: checked_bytes(&[batch_heads, steps, dv], dtype, "kda forward")?,
        state_next_bytes: if write_state_next {
            checked_bytes(&[batch_heads, dk, dv], DType::F32, "kda forward")?
        } else {
            0
        },
        staging_bytes: 0,
        status_bytes: 0,
        scratch_bytes: 0,
        pipeline_count: 1,
    })
}

pub fn backward_requirements(
    dtype: DType,
    batch_heads: usize,
    steps: usize,
    dk: usize,
    dv: usize,
    scale: f64,
) -> crate::err::Res<BackwardRequirements> {
    if !supported_dtype(dtype) {
        return Err(format!("kda backward: unsupported dtype {dtype:?}"));
    }
    let dk_bytes = checked_bytes(&[batch_heads, steps, dk], dtype, "kda backward")?;
    let dv_bytes = checked_bytes(&[batch_heads, steps, dv], dtype, "kda backward")?;
    let nchunks = steps.div_ceil(64);
    let starts = nchunks
        .checked_add(64)
        .and_then(|value| value.checked_mul(dk))
        .and_then(|value| value.checked_mul(dv))
        .ok_or_else(|| "kda backward: requirement byte size overflow".to_string())?;
    let deltas = 64usize
        .checked_mul(dv)
        .ok_or_else(|| "kda backward: requirement byte size overflow".to_string())?;
    let per_head = starts
        .checked_add(deltas)
        .ok_or_else(|| "kda backward: requirement byte size overflow".to_string())?;
    Ok(BackwardRequirements {
        dtype,
        batch_heads,
        steps,
        dk,
        dv,
        scale_bits: scale.to_bits(),
        dq_bytes: dk_bytes,
        dk_bytes,
        dv_bytes,
        dg_bytes: dk_bytes,
        db_bytes: checked_bytes(&[batch_heads, steps], dtype, "kda backward")?,
        staging_bytes: 0,
        status_bytes: 0,
        scratch_bytes: checked_bytes(&[batch_heads, per_head], DType::F32, "kda backward")?,
        pipeline_count: 1,
    })
}

pub fn decode_requirements(
    dtype: DType,
    heads: usize,
    dk: usize,
    dv: usize,
    scale: f64,
) -> crate::err::Res<DecodeRequirements> {
    if !supported_dtype(dtype) {
        return Err(format!("kda decode: unsupported dtype {dtype:?}"));
    }
    Ok(DecodeRequirements {
        dtype,
        heads,
        dk,
        dv,
        scale_bits: scale.to_bits(),
        output_bytes: checked_bytes(&[heads, dv], dtype, "kda decode")?,
        state_next_bytes: checked_bytes(&[heads, dk, dv], DType::F32, "kda decode")?,
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
    fn requirements_are_exact_and_overflow_checked() {
        let forward = forward_requirements(DType::F32, 2, 3, 7, 5, 0.25, true).unwrap();
        assert_eq!(forward.output_bytes, 2 * 3 * 5 * 4);
        assert_eq!(forward.state_next_bytes, 2 * 7 * 5 * 4);
        assert_eq!(
            (
                forward.staging_bytes,
                forward.status_bytes,
                forward.scratch_bytes
            ),
            (0, 0, 0)
        );
        assert_eq!(forward.pipeline_count, 1);

        let backward = backward_requirements(DType::F32, 2, 3, 7, 5, 0.25).unwrap();
        assert_eq!(backward.dq_bytes, 2 * 3 * 7 * 4);
        assert_eq!(backward.dv_bytes, 2 * 3 * 5 * 4);
        assert_eq!(backward.db_bytes, 2 * 3 * 4);
        assert_eq!(backward.scratch_bytes, 2 * ((1 + 64) * 7 * 5 + 64 * 5) * 4);

        let decode = decode_requirements(DType::BF16, 2, 7, 5, 0.25).unwrap();
        assert_eq!(decode.output_bytes, 2 * 5 * 2);
        assert_eq!(decode.state_next_bytes, 2 * 7 * 5 * 4);
        assert!(forward_requirements(DType::F32, usize::MAX, 2, 1, 1, 1.0, false).is_err());
    }
}

/// f64 has no Metal compute support anywhere in this backend; everything
/// else (any head/value dims, f32/bf16) is handled by the masked kernels.
pub fn supported_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::BF16)
}

#[cfg(target_os = "macos")]
pub use metal::{
    backward, backward_into, decode, decode_into, forward, forward_into, warm_backward,
    warm_backward_exact, warm_decode, warm_decode_exact, warm_forward, warm_forward_exact,
    IntoResources,
};

#[cfg(target_os = "macos")]
mod metal {
    use crate::runtime::dtype::DType;
    use crate::runtime::metal::device::{set_buffer, set_bytes, MetalDevice, Pipeline};
    use crate::runtime::metal::run::MetalTensor;

    use objc2_metal::MTLComputeCommandEncoder;

    #[derive(Clone, Copy)]
    pub struct IntoResources<'a> {
        pub staging: &'a [MetalTensor],
        pub status: &'a [MetalTensor],
        pub scratch: &'a [MetalTensor],
    }

    impl IntoResources<'_> {
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

    fn require_contiguous(inputs: &[&MetalTensor], operation: &str) -> crate::err::Res<()> {
        if inputs.iter().any(|tensor| !tensor.layout.is_contiguous()) {
            return Err(format!(
                "{operation}: inputs must be contiguous before destination encoding"
            ));
        }
        Ok(())
    }

    fn kernel_source(dtype: DType, dk: usize, dv: usize, scale: f64) -> String {
        let ty = match dtype {
            DType::F32 => "float",
            DType::BF16 => "bfloat",
            other => unreachable!("kda decode: unsupported dtype {other:?}"),
        };
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;

#define T {ty}
#define DK {dk}
#define DV {dv}
#define M {m}
#define SCALE {scale:?}f

kernel void et_kda_decode(
    device const T* Q [[buffer(0)]],
    device const T* K [[buffer(1)]],
    device const T* V [[buffer(2)]],
    device const T* G [[buffer(3)]],
    device const T* B [[buffer(4)]],
    device const float* S0 [[buffer(5)]],
    device float* S1 [[buffer(6)]],
    device T* O [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {{
    const uint lane = tpitg.x;
    const uint dv = tgid.y * 4 + tpitg.y;
    const uint h = tgid.z;
    const bool dv_ok = dv < DV;
    device const T* qp = Q + h * DK;
    device const T* kp = K + h * DK;
    device const T* gp = G + h * DK;
    device const float* s0p = S0 + (ulong)h * DK * DV;
    device float* s1p = S1 + (ulong)h * DK * DV;
    float s[M];
    float kk[M];
    float kv = 0.0f;
    for (uint m = 0; m < M; m++) {{
        const uint d = m * 32 + lane;
        const bool ok = (d < DK) && dv_ok;
        const float km = ok ? float(kp[d]) : 0.0f;
        float sm = ok ? s0p[(ulong)d * DV + dv] * exp(float(gp[d])) : 0.0f;
        kv += sm * km;
        s[m] = sm;
        kk[m] = km;
    }}
    const float kvm = simd_sum(kv);
    const float vv = dv_ok ? float(V[h * DV + dv]) : 0.0f;
    const float delta = (vv - kvm) * float(B[h]);
    float qo = 0.0f;
    for (uint m = 0; m < M; m++) {{
        const uint d = m * 32 + lane;
        const bool ok = (d < DK) && dv_ok;
        s[m] += kk[m] * delta;
        qo += ok ? s[m] * float(qp[d]) : 0.0f;
    }}
    const float o = simd_sum(qo) * SCALE;
    if (lane == 0 && dv_ok) {{
        O[h * DV + dv] = T(o);
    }}
    for (uint m = 0; m < M; m++) {{
        const uint d = m * 32 + lane;
        if (d < DK && dv_ok) s1p[(ulong)d * DV + dv] = s[m];
    }}
}}
"#,
            ty = ty,
            dk = dk,
            dv = dv,
            m = dk.div_ceil(32),
            scale = scale,
        )
    }

    fn pipeline_key(tag: u32, dtype: DType, dk: usize, dv: usize, scale: f64) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        (tag, dtype, dk, dv, scale.to_bits()).hash(&mut hasher);
        hasher.finish()
    }

    fn cached_pipeline(
        tag: u32,
        dtype: DType,
        dk: usize,
        dv: usize,
        scale: f64,
        name: &str,
    ) -> crate::err::Res<Pipeline> {
        let key = pipeline_key(tag, dtype, dk, dv, scale);
        MetalDevice::get().pipeline_cached(key).ok_or_else(|| {
            format!("kda: exact {name} pipeline is not warm; call the matching warm function")
        })
    }

    fn pipeline(dtype: DType, dk: usize, dv: usize, scale: f64) -> crate::err::Res<Pipeline> {
        MetalDevice::get().compile_lazy(
            pipeline_key(0xDA01, dtype, dk, dv, scale),
            "et_kda_decode",
            || kernel_source(dtype, dk, dv, scale),
        )
    }

    pub fn warm_decode(dtype: DType, dk: usize, dv: usize, scale: f64) -> crate::err::Res<()> {
        pipeline(dtype, dk, dv, scale)?;
        Ok(())
    }

    pub fn warm_decode_exact(requirements: &super::DecodeRequirements) -> crate::err::Res<()> {
        warm_decode(
            requirements.dtype,
            requirements.dk,
            requirements.dv,
            f64::from_bits(requirements.scale_bits),
        )
    }

    fn fwd_source(dtype: DType, dk: usize, dv: usize, scale: f64) -> String {
        let ty = match dtype {
            DType::F32 => "float",
            DType::BF16 => "bfloat",
            other => unreachable!("kda forward: unsupported dtype {other:?}"),
        };
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;

#define T {ty}
#define DK {dk}
#define DV {dv}
#define M {m}
#define SCALE {scale:?}f

// Sequential gated delta-rule forward: one threadgroup per (batch·head,
// 4-column value strip), state in registers, one launch per layer. The
// chunked WY form wins on tensor-core throughput at large dims; at
// Dk/Dv <= 128 the register-resident scan is strictly cheaper than
// materializing the chunk algebra.
kernel void et_kda_forward(
    device const T* Q [[buffer(0)]],
    device const T* K [[buffer(1)]],
    device const T* V [[buffer(2)]],
    device const T* G [[buffer(3)]],
    device const T* B [[buffer(4)]],
    device const float* S0 [[buffer(5)]],
    device float* S1 [[buffer(6)]],
    device T* O [[buffer(7)]],
    constant uint& steps [[buffer(8)]],
    constant uint& flags [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {{
    const uint lane = tpitg.x;
    const uint dv = tgid.y * 4 + tpitg.y;
    const uint bh = tgid.z;
    const bool dv_ok = dv < DV;
    device const T* qp = Q + (ulong)bh * steps * DK;
    device const T* kp = K + (ulong)bh * steps * DK;
    device const T* vp = V + (ulong)bh * steps * DV;
    device const T* gp = G + (ulong)bh * steps * DK;
    device const T* bp = B + (ulong)bh * steps;
    device T* op = O + (ulong)bh * steps * DV;
    const bool has_s0 = (flags & 1u) != 0u;
    const bool write_s1 = (flags & 2u) != 0u;
    float s[M];
    if (has_s0) {{
        device const float* s0p = S0 + (ulong)bh * DK * DV;
        for (uint m = 0; m < M; m++) {{
            const uint d = m * 32 + lane;
            s[m] = (d < DK && dv_ok) ? s0p[(ulong)d * DV + dv] : 0.0f;
        }}
    }} else {{
        for (uint m = 0; m < M; m++) s[m] = 0.0f;
    }}
    for (uint t = 0; t < steps; t++) {{
        float kk[M];
        float kv = 0.0f;
        for (uint m = 0; m < M; m++) {{
            const uint d = m * 32 + lane;
            const bool ok = (d < DK) && dv_ok;
            const float km = ok ? float(kp[t * DK + d]) : 0.0f;
            float sm = ok ? s[m] * exp(float(gp[t * DK + d])) : 0.0f;
            kv += sm * km;
            s[m] = sm;
            kk[m] = km;
        }}
        const float kvm = simd_sum(kv);
        const float vv = dv_ok ? float(vp[t * DV + dv]) : 0.0f;
        const float delta = (vv - kvm) * float(bp[t]);
        float qo = 0.0f;
        for (uint m = 0; m < M; m++) {{
            const uint d = m * 32 + lane;
            const bool ok = (d < DK) && dv_ok;
            s[m] += kk[m] * delta;
            qo += ok ? s[m] * float(qp[t * DK + d]) : 0.0f;
        }}
        const float o = simd_sum(qo) * SCALE;
        if (lane == 0 && dv_ok) op[t * DV + dv] = T(o);
    }}
    if (write_s1 && dv_ok) {{
        device float* s1p = S1 + (ulong)bh * DK * DV;
        for (uint m = 0; m < M; m++) {{
            const uint d = m * 32 + lane;
            if (d < DK) s1p[(ulong)d * DV + dv] = s[m];
        }}
    }}
}}
"#,
            ty = ty,
            dk = dk,
            dv = dv,
            m = dk.div_ceil(32),
            scale = scale,
        )
    }

    fn bwd_source(dtype: DType, dk: usize, dv: usize, scale: f64) -> String {
        let ty = match dtype {
            DType::F32 => "float",
            DType::BF16 => "bfloat",
            other => unreachable!("kda backward: unsupported dtype {other:?}"),
        };
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;

#define T {ty}
#define DK {dk}
#define DV {dv}
#define M {m}
#define C4 {c4}
#define CHUNK {chunk}
#define SCALE {scale:?}f

// Closed-form adjoint of the gated delta rule. One threadgroup per
// batch·head (the dq/dk/dg/db gradients sum over the FULL value dim, so
// strips would need cross-threadgroup reduction); thread (lane, row)
// owns dk rows m*32+lane (m < M) and value columns row*C4..+C4, so the
// fp32 state/adjoint tiles live in registers. Phase A recomputes
// chunk-start states into the workspace; phase B walks chunks in
// reverse, recomputing each chunk's per-token states into the workspace
// and stepping the adjoint state L back through the tokens:
//   L += scale·q·do^T;  dq = scale·S_t·do;  dv = β·L^T k;
//   dk = β·(L δ − S̃ (L^T k));  dβ = k^T L δ;
//   dg = α ⊙ sum_dv(S_{{t-1}} ⊙ M);  L ← Diag(α) M,  M = (I − β k k^T) L
// Per-DV sums reduce across lanes with simd_sum; per-DK sums reduce
// across the 4 rows through threadgroup memory.
kernel void et_kda_backward(
    device const T* Q [[buffer(0)]],
    device const T* K [[buffer(1)]],
    device const T* V [[buffer(2)]],
    device const T* G [[buffer(3)]],
    device const T* B [[buffer(4)]],
    device const T* dO [[buffer(5)]],
    device float* WS [[buffer(6)]],
    device T* dQ [[buffer(7)]],
    device T* dK [[buffer(8)]],
    device T* dV [[buffer(9)]],
    device T* dG [[buffer(10)]],
    device T* dB [[buffer(11)]],
    constant uint& steps [[buffer(12)]],
    constant uint& nchunks [[buffer(13)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {{
    const uint lane = tpitg.x;
    const uint row = tpitg.y;
    const uint bh = tgid.z;
    const ulong rowElems = (ulong)DK * DV;
    const ulong wsElems = ((ulong)nchunks + CHUNK) * rowElems + CHUNK * DV;
    device float* starts = WS + (ulong)bh * wsElems;
    device float* wss = starts + (ulong)nchunks * rowElems;
    device float* wsd = wss + CHUNK * rowElems;
    device const T* qp = Q + (ulong)bh * steps * DK;
    device const T* kp = K + (ulong)bh * steps * DK;
    device const T* vp = V + (ulong)bh * steps * DV;
    device const T* gp = G + (ulong)bh * steps * DK;
    device const T* bp = B + (ulong)bh * steps;
    device const T* dop = dO + (ulong)bh * steps * DV;
    threadgroup float p_dq[DK * 4];
    threadgroup float p_ldel[DK * 4];
    threadgroup float p_sdec[DK * 4];
    threadgroup float p_dga[DK * 4];
    threadgroup float r_dq[DK];
    threadgroup float r_ldel[DK];
    threadgroup float r_sdec[DK];
    threadgroup float r_dga[DK];

    // Phase A: chunk-start states via the register recurrence.
    float s[M][C4];
    for (uint m = 0; m < M; m++)
        for (uint c = 0; c < C4; c++) s[m][c] = 0.0f;
    for (uint t = 0; t < steps; t++) {{
        if (t % CHUNK == 0) {{
            device float* dst = starts + (t / CHUNK) * rowElems;
            for (uint m = 0; m < M; m++) {{
                const uint d = m * 32 + lane;
                if (d >= DK) continue;
                for (uint c = 0; c < C4; c++) {{
                    const uint col = row * C4 + c;
                    if (col < DV) dst[(ulong)d * DV + col] = s[m][c];
                }}
            }}
        }}
        float kk[M];
        float al[M];
        for (uint m = 0; m < M; m++) {{
            const uint d = m * 32 + lane;
            kk[m] = (d < DK) ? float(kp[t * DK + d]) : 0.0f;
            al[m] = (d < DK) ? exp(float(gp[t * DK + d])) : 0.0f;
        }}
        const float beta = float(bp[t]);
        for (uint m = 0; m < M; m++)
            for (uint c = 0; c < C4; c++) s[m][c] *= al[m];
        for (uint c = 0; c < C4; c++) {{
            const uint col = row * C4 + c;
            float kv = 0.0f;
            for (uint m = 0; m < M; m++) kv += s[m][c] * kk[m];
            const float kvm = simd_sum(kv);
            const float vv = (col < DV) ? float(vp[t * DV + col]) : 0.0f;
            const float delta = (vv - kvm) * beta;
            for (uint m = 0; m < M; m++) s[m][c] += kk[m] * delta;
        }}
    }}

    float lam[M][C4];
    for (uint m = 0; m < M; m++)
        for (uint c = 0; c < C4; c++) lam[m][c] = 0.0f;
    for (uint cdown = 0; cdown < nchunks; cdown++) {{
        const uint ci = nchunks - 1 - cdown;
        const uint t0 = ci * CHUNK;
        const uint clen = min((uint)CHUNK, steps - t0);
        // Recompute the chunk's per-token states and deltas.
        device const float* st = starts + ci * rowElems;
        for (uint m = 0; m < M; m++) {{
            const uint d = m * 32 + lane;
            for (uint c = 0; c < C4; c++) {{
                const uint col = row * C4 + c;
                s[m][c] = (d < DK && col < DV) ? st[(ulong)d * DV + col] : 0.0f;
            }}
        }}
        for (uint i = 0; i < clen; i++) {{
            const uint t = t0 + i;
            float kk[M];
            float al[M];
            for (uint m = 0; m < M; m++) {{
                const uint d = m * 32 + lane;
                kk[m] = (d < DK) ? float(kp[t * DK + d]) : 0.0f;
                al[m] = (d < DK) ? exp(float(gp[t * DK + d])) : 0.0f;
            }}
            const float beta = float(bp[t]);
            for (uint m = 0; m < M; m++)
                for (uint c = 0; c < C4; c++) s[m][c] *= al[m];
            device float* ws = wss + i * rowElems;
            for (uint c = 0; c < C4; c++) {{
                const uint col = row * C4 + c;
                float kv = 0.0f;
                for (uint m = 0; m < M; m++) kv += s[m][c] * kk[m];
                const float kvm = simd_sum(kv);
                // Raw delta (no beta): the adjoint formulas consume it
                // unscaled; beta enters the state update separately.
                const float vv = (col < DV) ? float(vp[t * DV + col]) : 0.0f;
                const float delta = vv - kvm;
                for (uint m = 0; m < M; m++) s[m][c] += kk[m] * (beta * delta);
                if (col < DV) wsd[i * DV + col] = delta;
            }}
            for (uint m = 0; m < M; m++) {{
                const uint d = m * 32 + lane;
                if (d >= DK) continue;
                for (uint c = 0; c < C4; c++) {{
                    const uint col = row * C4 + c;
                    if (col < DV) ws[(ulong)d * DV + col] = s[m][c];
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_device);

        // Reverse walk of the chunk's tokens.
        for (uint idown = 0; idown < clen; idown++) {{
            const uint i = clen - 1 - idown;
            const uint t = t0 + i;
            const float beta = float(bp[t]);
            float kk[M];
            float al[M];
            float qq[M];
            float gg[C4];
            float dd[C4];
            for (uint m = 0; m < M; m++) {{
                const uint d = m * 32 + lane;
                const bool dok = d < DK;
                kk[m] = dok ? float(kp[t * DK + d]) : 0.0f;
                al[m] = dok ? exp(float(gp[t * DK + d])) : 0.0f;
                qq[m] = dok ? SCALE * float(qp[t * DK + d]) : 0.0f;
            }}
            for (uint c = 0; c < C4; c++) {{
                const uint col = row * C4 + c;
                gg[c] = (col < DV) ? float(dop[t * DV + col]) : 0.0f;
                dd[c] = (col < DV) ? wsd[i * DV + col] : 0.0f;
            }}
            device const float* ws = wss + i * rowElems;
            device const float* wp = (i == 0) ? (starts + ci * rowElems) : (wss + (i - 1) * rowElems);
            // L += scale · q · do^T
            for (uint m = 0; m < M; m++)
                for (uint c = 0; c < C4; c++) lam[m][c] += qq[m] * gg[c];
            // lamk[c] = sum_dk L[dk, dv_c] · k[dk]
            float lamk[C4];
            for (uint c = 0; c < C4; c++) {{
                float acc = 0.0f;
                for (uint m = 0; m < M; m++) acc += lam[m][c] * kk[m];
                lamk[c] = simd_sum(acc);
            }}
            if (lane == 0) {{
                for (uint c = 0; c < C4; c++) {{
                    const uint col = row * C4 + c;
                    if (col < DV)
                        dV[(ulong)bh * steps * DV + t * DV + col] = T(beta * lamk[c]);
                }}
            }}
            // Per-dk partial sums over this row's C4 value columns.
            for (uint m = 0; m < M; m++) {{
                const uint d = m * 32 + lane;
                float sdq = 0.0f;
                float sldel = 0.0f;
                float ssdec = 0.0f;
                float sdga = 0.0f;
                for (uint c = 0; c < C4; c++) {{
                    const uint col = row * C4 + c;
                    const bool ok = (d < DK) && (col < DV);
                    const float sprev = ok ? wp[(ulong)d * DV + col] : 0.0f;
                    const float stt = ok ? ws[(ulong)d * DV + col] : 0.0f;
                    const float mm = lam[m][c] - beta * kk[m] * lamk[c];
                    sdq += stt * gg[c];
                    sldel += lam[m][c] * dd[c];
                    ssdec += al[m] * sprev * lamk[c];
                    sdga += sprev * mm;
                    lam[m][c] = al[m] * mm;
                }}
                if (d < DK) {{
                    const uint base = d * 4 + row;
                    p_dq[base] = sdq;
                    p_ldel[base] = sldel;
                    p_sdec[base] = ssdec;
                    p_dga[base] = sdga;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (row == 0) {{
                for (uint m = 0; m < M; m++) {{
                    const uint d = m * 32 + lane;
                    if (d >= DK) continue;
                    const uint base = d * 4;
                    r_dq[d] = p_dq[base] + p_dq[base + 1] + p_dq[base + 2] + p_dq[base + 3];
                    r_ldel[d] = p_ldel[base] + p_ldel[base + 1] + p_ldel[base + 2] + p_ldel[base + 3];
                    r_sdec[d] = p_sdec[base] + p_sdec[base + 1] + p_sdec[base + 2] + p_sdec[base + 3];
                    r_dga[d] = p_dga[base] + p_dga[base + 1] + p_dga[base + 2] + p_dga[base + 3];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float dbp = 0.0f;
            for (uint m = 0; m < M; m++) {{
                const uint d = m * 32 + lane;
                if (d < DK) dbp += kk[m] * r_ldel[d];
            }}
            const float dbv = simd_sum(dbp);
            if (row == 0) {{
                for (uint m = 0; m < M; m++) {{
                    const uint d = m * 32 + lane;
                    if (d >= DK) continue;
                    dQ[(ulong)bh * steps * DK + t * DK + d] = T(SCALE * r_dq[d]);
                    dK[(ulong)bh * steps * DK + t * DK + d] = T(beta * (r_ldel[d] - r_sdec[d]));
                    dG[(ulong)bh * steps * DK + t * DK + d] = T(al[m] * r_dga[d]);
                }}
                if (lane == 0) dB[(ulong)bh * steps + t] = T(dbv);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
    }}
}}
"#,
            ty = ty,
            dk = dk,
            dv = dv,
            m = dk.div_ceil(32),
            c4 = dv.div_ceil(4),
            chunk = 64,
            scale = scale,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_into(
        q: &MetalTensor,
        k: &MetalTensor,
        v: &MetalTensor,
        g: &MetalTensor,
        beta: &MetalTensor,
        state: &MetalTensor,
        scale: f64,
        output: &MetalTensor,
        state_next: &MetalTensor,
        resources: IntoResources<'_>,
    ) -> crate::err::Res<()> {
        let dtype = q.dtype;
        let (h, dk) = (q.layout.shape()[0], q.layout.shape()[1]);
        let dv = v.layout.shape()[1];
        if !resources.staging.is_empty()
            || !resources.status.is_empty()
            || !resources.scratch.is_empty()
        {
            return Err("kda decode: staging, status, and scratch views must be empty".to_string());
        }
        for input in [q, k, v, g, beta, state] {
            if !input.layout.is_contiguous() {
                return Err("kda decode: inputs must be contiguous before decode_into".to_string());
            }
        }
        validate_destination(output, &[h, dv], dtype, "kda decode output")?;
        validate_destination(
            state_next,
            &[h, dk, dv],
            DType::F32,
            "kda decode state-next",
        )?;
        let pipe = cached_pipeline(0xDA01, dtype, dk, dv, scale, "decode")?;
        let elem = |off: usize| off * dtype.size_in_bytes();
        MetalDevice::get().with_encoder(|e| {
            e.setComputePipelineState(pipe.as_raw());
            set_buffer(e, 0, &q.buffer, elem(q.layout.offset()));
            set_buffer(e, 1, &k.buffer, elem(k.layout.offset()));
            set_buffer(e, 2, &v.buffer, elem(v.layout.offset()));
            set_buffer(e, 3, &g.buffer, elem(g.layout.offset()));
            set_buffer(e, 4, &beta.buffer, elem(beta.layout.offset()));
            set_buffer(
                e,
                5,
                &state.buffer,
                state.layout.offset() * DType::F32.size_in_bytes(),
            );
            set_buffer(
                e,
                6,
                &state_next.buffer,
                state_next.layout.offset() * DType::F32.size_in_bytes(),
            );
            set_buffer(
                e,
                7,
                &output.buffer,
                output.layout.offset() * dtype.size_in_bytes(),
            );
            e.dispatchThreadgroups_threadsPerThreadgroup(
                objc2_metal::MTLSize {
                    width: 1,
                    height: dv.div_ceil(4),
                    depth: h,
                },
                objc2_metal::MTLSize {
                    width: 32,
                    height: 4,
                    depth: 1,
                },
            );
        });
        Ok(())
    }

    pub fn decode(
        q: &MetalTensor,
        k: &MetalTensor,
        v: &MetalTensor,
        g: &MetalTensor,
        beta: &MetalTensor,
        state: &MetalTensor,
        scale: f64,
    ) -> crate::err::Res<MetalTensor> {
        let (h, dv) = (q.layout.shape()[0], v.layout.shape()[1]);
        let (q, k, v, g, beta) = (
            wrap_contig(q)?,
            wrap_contig(k)?,
            wrap_contig(v)?,
            wrap_contig(g)?,
            wrap_contig(beta)?,
        );
        if !state.layout.is_contiguous() {
            return Err("kda decode: state must be contiguous".to_string());
        }
        warm_decode(q.dtype, q.layout.shape()[1], v.layout.shape()[1], scale)?;
        let output = MetalTensor::empty(MetalDevice::get(), vec![h, dv], q.dtype);
        decode_into(
            &q,
            &k,
            &v,
            &g,
            &beta,
            state,
            scale,
            &output,
            state,
            IntoResources::empty(),
        )?;
        Ok(output)
    }

    fn fwd_pipeline(dtype: DType, dk: usize, dv: usize, scale: f64) -> crate::err::Res<Pipeline> {
        MetalDevice::get().compile_lazy(
            pipeline_key(0xDA02, dtype, dk, dv, scale),
            "et_kda_forward",
            || fwd_source(dtype, dk, dv, scale),
        )
    }

    fn bwd_pipeline(dtype: DType, dk: usize, dv: usize, scale: f64) -> crate::err::Res<Pipeline> {
        MetalDevice::get().compile_lazy(
            pipeline_key(0xDA03, dtype, dk, dv, scale),
            "et_kda_backward",
            || bwd_source(dtype, dk, dv, scale),
        )
    }

    pub fn warm_forward(dtype: DType, dk: usize, dv: usize, scale: f64) -> crate::err::Res<()> {
        fwd_pipeline(dtype, dk, dv, scale)?;
        Ok(())
    }

    pub fn warm_forward_exact(requirements: &super::ForwardRequirements) -> crate::err::Res<()> {
        warm_forward(
            requirements.dtype,
            requirements.dk,
            requirements.dv,
            f64::from_bits(requirements.scale_bits),
        )
    }

    pub fn warm_backward(dtype: DType, dk: usize, dv: usize, scale: f64) -> crate::err::Res<()> {
        bwd_pipeline(dtype, dk, dv, scale)?;
        Ok(())
    }

    pub fn warm_backward_exact(requirements: &super::BackwardRequirements) -> crate::err::Res<()> {
        warm_backward(
            requirements.dtype,
            requirements.dk,
            requirements.dv,
            f64::from_bits(requirements.scale_bits),
        )
    }

    // Fused forward scan: q/k/g [BH, T, Dk], v [BH, T, Dv], beta [BH, T];
    // optional fp32 initial state [BH, Dk, Dv] and final-state writeback.
    // Returns (output [BH, T, Dv] in the input dtype, final state when
    // requested).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_into(
        q: &MetalTensor,
        k: &MetalTensor,
        v: &MetalTensor,
        g: &MetalTensor,
        beta: &MetalTensor,
        scale: f64,
        initial_state: Option<&MetalTensor>,
        output: &MetalTensor,
        state_next: Option<&MetalTensor>,
        resources: IntoResources<'_>,
    ) -> crate::err::Res<()> {
        let dtype = q.dtype;
        let (bh, steps, dk) = (
            q.layout.shape()[0],
            q.layout.shape()[1],
            q.layout.shape()[2],
        );
        let dv = v.layout.shape()[2];
        if !resources.staging.is_empty()
            || !resources.status.is_empty()
            || !resources.scratch.is_empty()
        {
            return Err(
                "kda forward: staging, status, and scratch views must be empty".to_string(),
            );
        }
        require_contiguous(&[q, k, v, g, beta], "kda forward")?;
        if let Some(state) = initial_state {
            require_contiguous(&[state], "kda forward initial state")?;
            validate_destination(
                state,
                &[bh, dk, dv],
                DType::F32,
                "kda forward initial state",
            )?;
        }
        validate_destination(output, &[bh, steps, dv], dtype, "kda forward output")?;
        if let Some(state) = state_next {
            validate_destination(state, &[bh, dk, dv], DType::F32, "kda forward state-next")?;
        }
        let pipe = cached_pipeline(0xDA02, dtype, dk, dv, scale, "forward")?;
        let dev = MetalDevice::get();
        let mut flags = 0u32;
        if initial_state.is_some() {
            flags |= 1;
        }
        if state_next.is_some() {
            flags |= 2;
        }
        let elem = |off: usize| off * dtype.size_in_bytes();
        dev.with_encoder(|e| {
            e.setComputePipelineState(pipe.as_raw());
            set_buffer(e, 0, &q.buffer, elem(q.layout.offset()));
            set_buffer(e, 1, &k.buffer, elem(k.layout.offset()));
            set_buffer(e, 2, &v.buffer, elem(v.layout.offset()));
            set_buffer(e, 3, &g.buffer, elem(g.layout.offset()));
            set_buffer(e, 4, &beta.buffer, elem(beta.layout.offset()));
            let s0 = initial_state.unwrap_or(output);
            let s1 = state_next.unwrap_or(output);
            set_buffer(
                e,
                5,
                &s0.buffer,
                s0.layout.offset() * s0.dtype.size_in_bytes(),
            );
            set_buffer(
                e,
                6,
                &s1.buffer,
                s1.layout.offset() * s1.dtype.size_in_bytes(),
            );
            set_buffer(
                e,
                7,
                &output.buffer,
                output.layout.offset() * dtype.size_in_bytes(),
            );
            set_bytes(e, 8, &(steps as u32));
            set_bytes(e, 9, &flags);
            e.dispatchThreadgroups_threadsPerThreadgroup(
                objc2_metal::MTLSize {
                    width: 1,
                    height: dv.div_ceil(4),
                    depth: bh,
                },
                objc2_metal::MTLSize {
                    width: 32,
                    height: 4,
                    depth: 1,
                },
            );
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        q: &MetalTensor,
        k: &MetalTensor,
        v: &MetalTensor,
        g: &MetalTensor,
        beta: &MetalTensor,
        scale: f64,
        initial_state: Option<&MetalTensor>,
        write_final_state: bool,
    ) -> crate::err::Res<(MetalTensor, Option<MetalTensor>)> {
        let dtype = q.dtype;
        let (bh, steps, dk) = (
            q.layout.shape()[0],
            q.layout.shape()[1],
            q.layout.shape()[2],
        );
        let dv = v.layout.shape()[2];
        let (q, k, v, g, beta) = (
            wrap_contig(q)?,
            wrap_contig(k)?,
            wrap_contig(v)?,
            wrap_contig(g)?,
            wrap_contig(beta)?,
        );
        let initial_state = initial_state.map(wrap_contig).transpose()?;
        warm_forward(dtype, dk, dv, scale)?;
        let output = MetalTensor::empty(MetalDevice::get(), vec![bh, steps, dv], dtype);
        let state_next = write_final_state
            .then(|| MetalTensor::empty(MetalDevice::get(), vec![bh, dk, dv], DType::F32));
        forward_into(
            &q,
            &k,
            &v,
            &g,
            &beta,
            scale,
            initial_state.as_ref(),
            &output,
            state_next.as_ref(),
            IntoResources::empty(),
        )?;
        Ok((output, state_next))
    }

    // Fused closed-form backward: same operand contract as the composed
    // reference, g the output cotangent [BH, T, Dv]. Returns (dq, dk, dv,
    // dlog_decay, dbeta) in the input dtype.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_into(
        q: &MetalTensor,
        k: &MetalTensor,
        v: &MetalTensor,
        g: &MetalTensor,
        beta: &MetalTensor,
        dout: &MetalTensor,
        scale: f64,
        dq: &MetalTensor,
        dk_out: &MetalTensor,
        dv_out: &MetalTensor,
        dg: &MetalTensor,
        db: &MetalTensor,
        resources: IntoResources<'_>,
    ) -> crate::err::Res<()> {
        let dtype = q.dtype;
        let (bh, steps, dk) = (
            q.layout.shape()[0],
            q.layout.shape()[1],
            q.layout.shape()[2],
        );
        let dv = v.layout.shape()[2];
        if !resources.staging.is_empty() || !resources.status.is_empty() {
            return Err("kda backward: staging and status views must be empty".to_string());
        }
        if resources.scratch.len() != 1 {
            return Err("kda backward: exactly one scratch view is required".to_string());
        }
        require_contiguous(&[q, k, v, g, beta, dout], "kda backward")?;
        validate_destination(dq, &[bh, steps, dk], dtype, "kda backward dq")?;
        validate_destination(dk_out, &[bh, steps, dk], dtype, "kda backward dk")?;
        validate_destination(dv_out, &[bh, steps, dv], dtype, "kda backward dv")?;
        validate_destination(dg, &[bh, steps, dk], dtype, "kda backward dg")?;
        validate_destination(db, &[bh, steps, 1], dtype, "kda backward db")?;
        let pipe = cached_pipeline(0xDA03, dtype, dk, dv, scale, "backward")?;
        let dev = MetalDevice::get();
        let nchunks = steps.div_ceil(64);
        let requirements = super::backward_requirements(dtype, bh, steps, dk, dv, scale)?;
        let scratch = &resources.scratch[0];
        if scratch.dtype != DType::F32
            || !scratch.layout.is_contiguous()
            || scratch.numel().checked_mul(DType::F32.size_in_bytes())
                != Some(requirements.scratch_bytes)
        {
            return Err(format!(
                "kda backward: scratch must be contiguous f32 storage of {} bytes",
                requirements.scratch_bytes
            ));
        }
        let scratch_end = scratch
            .layout
            .checked_max_index()
            .and_then(|elements| elements.checked_mul(DType::F32.size_in_bytes()))
            .ok_or_else(|| "kda backward: scratch byte size overflow".to_string())?;
        if scratch_end > scratch.buffer.size {
            return Err("kda backward: scratch view exceeds its buffer".to_string());
        }
        let elem = |off: usize| off * dtype.size_in_bytes();
        dev.with_encoder(|e| {
            e.setComputePipelineState(pipe.as_raw());
            set_buffer(e, 0, &q.buffer, elem(q.layout.offset()));
            set_buffer(e, 1, &k.buffer, elem(k.layout.offset()));
            set_buffer(e, 2, &v.buffer, elem(v.layout.offset()));
            set_buffer(e, 3, &g.buffer, elem(g.layout.offset()));
            set_buffer(e, 4, &beta.buffer, elem(beta.layout.offset()));
            set_buffer(e, 5, &dout.buffer, elem(dout.layout.offset()));
            set_buffer(
                e,
                6,
                &scratch.buffer,
                scratch.layout.offset() * DType::F32.size_in_bytes(),
            );
            set_buffer(e, 7, &dq.buffer, elem(dq.layout.offset()));
            set_buffer(e, 8, &dk_out.buffer, elem(dk_out.layout.offset()));
            set_buffer(e, 9, &dv_out.buffer, elem(dv_out.layout.offset()));
            set_buffer(e, 10, &dg.buffer, elem(dg.layout.offset()));
            set_buffer(e, 11, &db.buffer, elem(db.layout.offset()));
            set_bytes(e, 12, &(steps as u32));
            set_bytes(e, 13, &(nchunks as u32));
            e.dispatchThreadgroups_threadsPerThreadgroup(
                objc2_metal::MTLSize {
                    width: 1,
                    height: 1,
                    depth: bh,
                },
                objc2_metal::MTLSize {
                    width: 32,
                    height: 4,
                    depth: 1,
                },
            );
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        q: &MetalTensor,
        k: &MetalTensor,
        v: &MetalTensor,
        g: &MetalTensor,
        beta: &MetalTensor,
        dout: &MetalTensor,
        scale: f64,
    ) -> crate::err::Res<(
        MetalTensor,
        MetalTensor,
        MetalTensor,
        MetalTensor,
        MetalTensor,
    )> {
        let dtype = q.dtype;
        let (bh, steps, dk) = (
            q.layout.shape()[0],
            q.layout.shape()[1],
            q.layout.shape()[2],
        );
        let dv = v.layout.shape()[2];
        let (q, k, v, g, beta, dout) = (
            wrap_contig(q)?,
            wrap_contig(k)?,
            wrap_contig(v)?,
            wrap_contig(g)?,
            wrap_contig(beta)?,
            wrap_contig(dout)?,
        );
        warm_backward(dtype, dk, dv, scale)?;
        let dev = MetalDevice::get();
        let dq = MetalTensor::empty(dev, vec![bh, steps, dk], dtype);
        let dk_out = MetalTensor::empty(dev, vec![bh, steps, dk], dtype);
        let dv_out = MetalTensor::empty(dev, vec![bh, steps, dv], dtype);
        let dg = MetalTensor::empty(dev, vec![bh, steps, dk], dtype);
        let db = MetalTensor::empty(dev, vec![bh, steps, 1], dtype);
        let requirements = super::backward_requirements(dtype, bh, steps, dk, dv, scale)?;
        let scratch = MetalTensor::empty(
            dev,
            vec![requirements.scratch_bytes / DType::F32.size_in_bytes()],
            DType::F32,
        );
        backward_into(
            &q,
            &k,
            &v,
            &g,
            &beta,
            &dout,
            scale,
            &dq,
            &dk_out,
            &dv_out,
            &dg,
            &db,
            IntoResources {
                staging: &[],
                status: &[],
                scratch: std::slice::from_ref(&scratch),
            },
        )?;
        Ok((dq, dk_out, dv_out, dg, db))
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

    fn warm_cat_tree(mut shapes: Vec<Vec<usize>>, dim: usize) {
        while shapes.len() > 1 {
            let mut next = Vec::with_capacity(shapes.len().div_ceil(2));
            for pair in shapes.chunks(2) {
                if pair.len() == 1 {
                    next.push(pair[0].clone());
                    continue;
                }
                crate::runtime::metal::indexing::warm_cat(
                    &[pair[0].as_slice(), pair[1].as_slice()],
                    crate::runtime::dtype::DType::F32,
                    dim,
                )
                .unwrap();
                let mut output = pair[0].clone();
                output[dim] += pair[1][dim];
                next.push(output);
            }
            shapes = next;
        }
    }

    // The fused single-token kernel against the composed chunked path
    // with a carried state (the recurrence oracle).
    #[test]
    fn decode_kernel_matches_composed() {
        let dev = MetalDevice::get();
        for (h, dk, dv) in [(2usize, 64usize, 64usize), (3, 128, 128)] {
            let scale = 1.0 / (dk as f64).sqrt();
            let q3 = MT::from_f32(dev, prand(h * dk, 11), vec![h, 1, dk]);
            let k3 = MT::from_f32(dev, prand(h * dk, 12), vec![h, 1, dk]);
            let v3 = MT::from_f32(dev, prand(h * dv, 13), vec![h, 1, dv]);
            let g3 = MT::from_f32(
                dev,
                prand(h * dk, 14)
                    .into_iter()
                    .map(|x| (x.abs() + 0.05) * -1.5)
                    .collect(),
                vec![h, 1, dk],
            );
            let b3 = MT::from_f32(
                dev,
                prand(h, 15)
                    .into_iter()
                    .map(|x| x.abs() * 0.9 + 0.05)
                    .collect(),
                vec![h, 1, 1],
            );
            let flat = |t: &MT, w: usize| MT {
                buffer: t.buffer.clone(),
                layout: crate::runtime::layout::Layout::contiguous(vec![h, w]),
                dtype: t.dtype,
            };
            let (q, k, v, g, b) = (
                flat(&q3, dk),
                flat(&k3, dk),
                flat(&v3, dv),
                flat(&g3, dk),
                flat(&b3, 1),
            );
            let state = MT::from_f32(dev, prand(h * dk * dv, 16), vec![h, dk, dv]);

            let (composed_out, composed_state) =
                crate::runtime::metal::composed::kda_chunk_with_state(
                    &q3, &k3, &v3, &g3, &b3, scale, &state,
                )
                .unwrap();
            let fused_out = super::metal::decode(&q, &k, &v, &g, &b, &state, scale).unwrap();
            dev.synchronize().unwrap();

            let a = composed_out.read_f32().unwrap();
            let c = fused_out.read_f32().unwrap();
            let max_diff = |x: &[f32], y: &[f32]| {
                x.iter()
                    .zip(y.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max)
            };
            let out_diff = max_diff(&a, &c);
            let state_diff = max_diff(
                &composed_state.read_f32().unwrap(),
                &state.read_f32().unwrap(),
            );
            assert!(
                out_diff < 1e-4 && state_diff < 1e-4,
                "({h}, {dk}, {dv}): out diff {out_diff}, state diff {state_diff}"
            );
        }
    }

    // The fused sequential forward/backward against the composed chunked
    // reference (itself finite-difference-verified), crossing a chunk boundary.
    #[test]
    fn forward_backward_match_composed() {
        let dev = MetalDevice::get();
        let (bh, t, dk, dv) = (2usize, 70usize, 64usize, 64usize);
        let scale = 1.0 / (dk as f64).sqrt();
        let q = MT::from_f32(dev, prand(bh * t * dk, 21), vec![bh, t, dk]);
        let k = MT::from_f32(dev, prand(bh * t * dk, 22), vec![bh, t, dk]);
        let v = MT::from_f32(dev, prand(bh * t * dv, 23), vec![bh, t, dv]);
        let g = MT::from_f32(
            dev,
            prand(bh * t * dk, 24)
                .into_iter()
                .map(|x| (x.abs() + 0.05) * -1.5)
                .collect(),
            vec![bh, t, dk],
        );
        let b = MT::from_f32(
            dev,
            prand(bh * t, 25)
                .into_iter()
                .map(|x| x.abs() * 0.9 + 0.05)
                .collect(),
            vec![bh, t, 1],
        );
        let w = MT::from_f32(dev, prand(bh * t * dv, 26), vec![bh, t, dv]);

        let chunk_lengths = [64usize, t - 64];
        for chunk in chunk_lengths {
            let block_lengths = (0..chunk.div_ceil(16))
                .map(|block| 16usize.min(chunk - block * 16))
                .collect::<Vec<_>>();
            for &rows in &block_lengths {
                warm_cat_tree(
                    block_lengths
                        .iter()
                        .map(|&columns| vec![bh, rows, columns])
                        .collect(),
                    2,
                );
            }
            warm_cat_tree(
                block_lengths
                    .iter()
                    .map(|&rows| vec![bh, rows, chunk])
                    .collect(),
                1,
            );
        }
        warm_cat_tree(
            chunk_lengths
                .iter()
                .map(|&chunk| vec![bh, chunk, dv])
                .collect(),
            1,
        );
        for width in [dk, dv, 1] {
            warm_cat_tree((0..t).map(|_| vec![bh, 1, width]).collect(), 1);
        }

        let max_diff = |x: &[f32], y: &[f32]| {
            x.iter()
                .zip(y.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max)
        };

        // Forward (zero initial state).
        let composed_fwd =
            crate::runtime::metal::composed::kda_chunk_forward(&q, &k, &v, &g, &b, scale).unwrap();
        let (fused_fwd, _final_state) =
            super::metal::forward(&q, &k, &v, &g, &b, scale, None, true).unwrap();
        dev.synchronize().unwrap();
        let fwd_diff = max_diff(
            &composed_fwd.read_f32().unwrap(),
            &fused_fwd.read_f32().unwrap(),
        );
        assert!(fwd_diff < 1e-3, "forward diff {fwd_diff}");

        // Stateful forward: the second half from the first half's final
        // state must equal the full-sequence output tail.
        let half = t / 2;
        let narrow = |x: &MT, start: usize, len: usize| MT {
            buffer: x.buffer.clone(),
            layout: x.layout.narrow(1, start, len),
            dtype: x.dtype,
        };
        let (_, half_state) = super::metal::forward(
            &narrow(&q, 0, half),
            &narrow(&k, 0, half),
            &narrow(&v, 0, half),
            &narrow(&g, 0, half),
            &narrow(&b, 0, half),
            scale,
            None,
            true,
        )
        .unwrap();
        let (tail_out, _) = super::metal::forward(
            &narrow(&q, half, t - half),
            &narrow(&k, half, t - half),
            &narrow(&v, half, t - half),
            &narrow(&g, half, t - half),
            &narrow(&b, half, t - half),
            scale,
            half_state.as_ref(),
            false,
        )
        .unwrap();
        let composed_tail = crate::runtime::metal::ops::contiguous(&MT {
            buffer: composed_fwd.buffer.clone(),
            layout: composed_fwd.layout.narrow(1, half, t - half),
            dtype: composed_fwd.dtype,
        })
        .unwrap();
        dev.synchronize().unwrap();
        let tail_diff = max_diff(
            &composed_tail.read_f32().unwrap(),
            &tail_out.read_f32().unwrap(),
        );
        assert!(tail_diff < 1e-3, "stateful tail diff {tail_diff}");

        // Backward.
        let composed_bwd =
            crate::runtime::metal::composed::kda_chunk_backward(&q, &k, &v, &g, &b, &w, scale)
                .unwrap();
        let fused_bwd = super::metal::backward(&q, &k, &v, &g, &b, &w, scale).unwrap();
        dev.synchronize().unwrap();
        let pairs = [
            ("dq", &composed_bwd.0, &fused_bwd.0),
            ("dk", &composed_bwd.1, &fused_bwd.1),
            ("dv", &composed_bwd.2, &fused_bwd.2),
            ("dg", &composed_bwd.3, &fused_bwd.3),
            ("db", &composed_bwd.4, &fused_bwd.4),
        ];
        let mut report = String::new();
        let mut worst = 0f32;
        for (name, c, f) in pairs {
            let diff = max_diff(&c.read_f32().unwrap(), &f.read_f32().unwrap());
            report.push_str(&format!("{name} {diff:.5}  "));
            worst = worst.max(diff);
        }
        assert!(worst < 1e-3, "backward diffs: {report}");
    }

    #[test]
    fn into_matches_wrappers_and_uses_only_supplied_views() {
        let dev = MetalDevice::get();
        let (bh, t, dk, dv) = (1usize, 5usize, 16usize, 12usize);
        let scale = 1.0 / (dk as f64).sqrt();
        let q = MT::from_f32(dev, prand(bh * t * dk, 61), vec![bh, t, dk]);
        let k = MT::from_f32(dev, prand(bh * t * dk, 62), vec![bh, t, dk]);
        let v = MT::from_f32(dev, prand(bh * t * dv, 63), vec![bh, t, dv]);
        let decay = MT::from_f32(
            dev,
            prand(bh * t * dk, 64)
                .into_iter()
                .map(|value| -(value.abs() + 0.2))
                .collect(),
            vec![bh, t, dk],
        );
        let beta = MT::from_f32(
            dev,
            prand(bh * t, 65)
                .into_iter()
                .map(|value| value.abs() * 0.5)
                .collect(),
            vec![bh, t, 1],
        );
        let dout = MT::from_f32(dev, prand(bh * t * dv, 66), vec![bh, t, dv]);

        let (expected_output, expected_state) =
            super::metal::forward(&q, &k, &v, &decay, &beta, scale, None, true).unwrap();
        let expected_backward =
            super::metal::backward(&q, &k, &v, &decay, &beta, &dout, scale).unwrap();

        let qd = MT::from_f32(dev, prand(dk, 71), vec![1, dk]);
        let kd = MT::from_f32(dev, prand(dk, 72), vec![1, dk]);
        let vd = MT::from_f32(dev, prand(dv, 73), vec![1, dv]);
        let gd = MT::from_f32(
            dev,
            prand(dk, 74)
                .into_iter()
                .map(|value| -(value.abs() + 0.2))
                .collect(),
            vec![1, dk],
        );
        let bd = MT::from_f32(dev, vec![0.4], vec![1, 1]);
        let state_values = prand(dk * dv, 75);
        let expected_decode_state = MT::from_f32(dev, state_values.clone(), vec![1, dk, dv]);
        let decode_state = MT::from_f32(dev, state_values, vec![1, dk, dv]);
        let expected_decode =
            super::metal::decode(&qd, &kd, &vd, &gd, &bd, &expected_decode_state, scale).unwrap();

        let output = MT::empty(dev, vec![bh, t, dv], q.dtype);
        let state_next = MT::empty(dev, vec![bh, dk, dv], crate::runtime::dtype::DType::F32);
        let dq = MT::empty(dev, vec![bh, t, dk], q.dtype);
        let dk_out = MT::empty(dev, vec![bh, t, dk], q.dtype);
        let dv_out = MT::empty(dev, vec![bh, t, dv], q.dtype);
        let dg_out = MT::empty(dev, vec![bh, t, dk], q.dtype);
        let db_out = MT::empty(dev, vec![bh, t, 1], q.dtype);
        let requirements = super::backward_requirements(q.dtype, bh, t, dk, dv, scale).unwrap();
        let scratch = MT::empty(
            dev,
            vec![requirements.scratch_bytes / 4],
            crate::runtime::dtype::DType::F32,
        );
        let decode_output = MT::empty(dev, vec![1, dv], q.dtype);
        let decode_state_next = MT::empty(dev, vec![1, dk, dv], crate::runtime::dtype::DType::F32);
        let forward_requirements =
            super::forward_requirements(q.dtype, bh, t, dk, dv, scale, true).unwrap();
        let decode_requirements = super::decode_requirements(q.dtype, 1, dk, dv, scale).unwrap();
        super::metal::warm_forward_exact(&forward_requirements).unwrap();
        super::metal::warm_backward_exact(&requirements).unwrap();
        super::metal::warm_decode_exact(&decode_requirements).unwrap();
        dev.synchronize().unwrap();

        let _dispatch_guard = dev.begin_executable_dispatch().unwrap();
        super::metal::forward_into(
            &q,
            &k,
            &v,
            &decay,
            &beta,
            scale,
            None,
            &output,
            Some(&state_next),
            super::metal::IntoResources::empty(),
        )
        .unwrap();
        super::metal::backward_into(
            &q,
            &k,
            &v,
            &decay,
            &beta,
            &dout,
            scale,
            &dq,
            &dk_out,
            &dv_out,
            &dg_out,
            &db_out,
            super::metal::IntoResources {
                staging: &[],
                status: &[],
                scratch: std::slice::from_ref(&scratch),
            },
        )
        .unwrap();
        super::metal::decode_into(
            &qd,
            &kd,
            &vd,
            &gd,
            &bd,
            &decode_state,
            scale,
            &decode_output,
            &decode_state_next,
            super::metal::IntoResources::empty(),
        )
        .unwrap();
        dev.synchronize().unwrap();

        let pairs = [
            (&expected_output, &output),
            (expected_state.as_ref().unwrap(), &state_next),
            (&expected_backward.0, &dq),
            (&expected_backward.1, &dk_out),
            (&expected_backward.2, &dv_out),
            (&expected_backward.3, &dg_out),
            (&expected_backward.4, &db_out),
            (&expected_decode, &decode_output),
            (&expected_decode_state, &decode_state_next),
        ];
        for (expected, actual) in pairs {
            let diff = expected
                .read_f32()
                .unwrap()
                .iter()
                .zip(actual.read_f32().unwrap())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(diff < 1e-6, "into parity diff {diff}");
        }
    }
}

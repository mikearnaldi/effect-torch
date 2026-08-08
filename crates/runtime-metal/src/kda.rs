//! Fused KDA recurrent decode on Metal (RFC 0018, phase 3): one kernel
//! launch per (sequence slot, layer) advances the gated delta-rule state
//! by one token — S = Diag(alpha) S + beta k (v - (Diag(alpha) S)^T k)^T,
//! o = scale * S^T q — with the [Dk, Dv] fp32 state distributed across
//! threadgroup registers (each of 32 lanes holds Dk/32 rows for one of 4
//! value columns), simd_sum reductions for the k^T S and S^T q
//! contractions, and the state read and written in place. This replaces
//! the ~45-launch composed chunk path for the T=1 decode step; chunked
//! prefill keeps the composed reference. Head dims must satisfy
//! Dk % 32 == 0 and Dv % 4 == 0 (both 64/128 in practice).

use crate::runtime::dtype::DType;

pub fn is_supported(dtype: DType, head_dim: usize, value_dim: usize) -> bool {
    matches!(dtype, DType::F32 | DType::BF16)
        && head_dim % 32 == 0
        && head_dim <= 128
        && value_dim % 4 == 0
        && value_dim <= 128
}

#[cfg(target_os = "macos")]
pub use metal::{backward, decode, forward};

#[cfg(target_os = "macos")]
mod metal {
    use crate::runtime::dtype::DType;
    use crate::runtime::metal::device::{set_buffer, set_bytes, MetalDevice, Pipeline};
    use crate::runtime::metal::run::MetalTensor;

    use objc2_metal::MTLComputeCommandEncoder;

    fn wrap_contig(t: &MetalTensor) -> crate::err::Res<MetalTensor> {
        if t.layout.is_contiguous() && t.layout.offset() == 0 {
            Ok(t.clone())
        } else {
            crate::runtime::metal::kernels::strided_copy(MetalDevice::get(), t)
        }
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
    device float* S [[buffer(5)]],
    device T* O [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {{
    const uint lane = tpitg.x;
    const uint dv = tgid.y * 4 + tpitg.y;
    const uint h = tgid.z;
    device const T* qp = Q + h * DK;
    device const T* kp = K + h * DK;
    device const T* gp = G + h * DK;
    device float* sp = S + (ulong)h * DK * DV;
    float s[M];
    float kk[M];
    float kv = 0.0f;
    for (uint m = 0; m < M; m++) {{
        const uint d = m * 32 + lane;
        const float km = float(kp[d]);
        float sm = sp[(ulong)d * DV + dv] * exp(float(gp[d]));
        kv += sm * km;
        s[m] = sm;
        kk[m] = km;
    }}
    const float kvm = simd_sum(kv);
    const float delta = (float(V[h * DV + dv]) - kvm) * float(B[h]);
    float qo = 0.0f;
    for (uint m = 0; m < M; m++) {{
        const uint d = m * 32 + lane;
        s[m] += kk[m] * delta;
        qo += s[m] * float(qp[d]);
    }}
    const float o = simd_sum(qo) * SCALE;
    if (lane == 0) {{
        O[h * DV + dv] = T(o);
    }}
    for (uint m = 0; m < M; m++) {{
        sp[(ulong)(m * 32 + lane) * DV + dv] = s[m];
    }}
}}
"#,
            ty = ty,
            dk = dk,
            dv = dv,
            m = dk / 32,
            scale = scale,
        )
    }

    fn pipeline(dtype: DType, dk: usize, dv: usize, scale: f64) -> crate::err::Res<Pipeline> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        (0xDA01u32, dtype, dk, dv, scale.to_bits()).hash(&mut hasher);
        MetalDevice::get().compile_lazy(hasher.finish(), "et_kda_decode", || {
            kernel_source(dtype, dk, dv, scale)
        })
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
        for (uint m = 0; m < M; m++) s[m] = s0p[(ulong)(m * 32 + lane) * DV + dv];
    }} else {{
        for (uint m = 0; m < M; m++) s[m] = 0.0f;
    }}
    for (uint t = 0; t < steps; t++) {{
        float kk[M];
        float kv = 0.0f;
        for (uint m = 0; m < M; m++) {{
            const uint d = m * 32 + lane;
            const float km = float(kp[t * DK + d]);
            float sm = s[m] * exp(float(gp[t * DK + d]));
            kv += sm * km;
            s[m] = sm;
            kk[m] = km;
        }}
        const float kvm = simd_sum(kv);
        const float delta = (float(vp[t * DV + dv]) - kvm) * float(bp[t]);
        float qo = 0.0f;
        for (uint m = 0; m < M; m++) {{
            s[m] += kk[m] * delta;
            qo += s[m] * float(qp[t * DK + m * 32 + lane]);
        }}
        const float o = simd_sum(qo) * SCALE;
        if (lane == 0) op[t * DV + dv] = T(o);
    }}
    if (write_s1) {{
        device float* s1p = S1 + (ulong)bh * DK * DV;
        for (uint m = 0; m < M; m++) s1p[(ulong)(m * 32 + lane) * DV + dv] = s[m];
    }}
}}
"#,
            ty = ty,
            dk = dk,
            dv = dv,
            m = dk / 32,
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
            for (uint m = 0; m < M; m++)
                for (uint c = 0; c < C4; c++)
                    dst[(ulong)(m * 32 + lane) * DV + row * C4 + c] = s[m][c];
        }}
        float kk[M];
        float al[M];
        for (uint m = 0; m < M; m++) {{
            const uint d = m * 32 + lane;
            kk[m] = float(kp[t * DK + d]);
            al[m] = exp(float(gp[t * DK + d]));
        }}
        const float beta = float(bp[t]);
        for (uint m = 0; m < M; m++)
            for (uint c = 0; c < C4; c++) s[m][c] *= al[m];
        for (uint c = 0; c < C4; c++) {{
            float kv = 0.0f;
            for (uint m = 0; m < M; m++) kv += s[m][c] * kk[m];
            const float kvm = simd_sum(kv);
            const float delta = (float(vp[t * DV + row * C4 + c]) - kvm) * beta;
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
        for (uint m = 0; m < M; m++)
            for (uint c = 0; c < C4; c++)
                s[m][c] = st[(ulong)(m * 32 + lane) * DV + row * C4 + c];
        for (uint i = 0; i < clen; i++) {{
            const uint t = t0 + i;
            float kk[M];
            float al[M];
            for (uint m = 0; m < M; m++) {{
                const uint d = m * 32 + lane;
                kk[m] = float(kp[t * DK + d]);
                al[m] = exp(float(gp[t * DK + d]));
            }}
            const float beta = float(bp[t]);
            for (uint m = 0; m < M; m++)
                for (uint c = 0; c < C4; c++) s[m][c] *= al[m];
            device float* ws = wss + i * rowElems;
            for (uint c = 0; c < C4; c++) {{
                float kv = 0.0f;
                for (uint m = 0; m < M; m++) kv += s[m][c] * kk[m];
                const float kvm = simd_sum(kv);
                // Raw delta (no beta): the adjoint formulas consume it
                // unscaled; beta enters the state update separately.
                const float delta = float(vp[t * DV + row * C4 + c]) - kvm;
                for (uint m = 0; m < M; m++) s[m][c] += kk[m] * (beta * delta);
                wsd[i * DV + row * C4 + c] = delta;
            }}
            for (uint m = 0; m < M; m++)
                for (uint c = 0; c < C4; c++)
                    ws[(ulong)(m * 32 + lane) * DV + row * C4 + c] = s[m][c];
        }}
        threadgroup_barrier(mem_flags::mem_device);

        // Reverse walk of the chunk's tokens.
        for (uint idown = 0; idown < clen; idown++) {{
            const uint i = clen - 1 - idown;
            const uint t = t0 + i;
            const float beta = float(bp[t]);
            float kk[M];
            float al[M];
            float gg[C4];
            float dd[C4];
            for (uint m = 0; m < M; m++) {{
                const uint d = m * 32 + lane;
                kk[m] = float(kp[t * DK + d]);
                al[m] = exp(float(gp[t * DK + d]));
            }}
            for (uint c = 0; c < C4; c++) {{
                gg[c] = float(dop[t * DV + row * C4 + c]);
                dd[c] = wsd[i * DV + row * C4 + c];
            }}
            device const float* ws = wss + i * rowElems;
            device const float* wp = (i == 0) ? (starts + ci * rowElems) : (wss + (i - 1) * rowElems);
            // L += scale · q · do^T
            for (uint m = 0; m < M; m++) {{
                const float qv = SCALE * float(qp[t * DK + m * 32 + lane]);
                for (uint c = 0; c < C4; c++) lam[m][c] += qv * gg[c];
            }}
            // lamk[c] = sum_dk L[dk, dv_c] · k[dk]
            float lamk[C4];
            for (uint c = 0; c < C4; c++) {{
                float acc = 0.0f;
                for (uint m = 0; m < M; m++) acc += lam[m][c] * kk[m];
                lamk[c] = simd_sum(acc);
            }}
            if (lane == 0) {{
                for (uint c = 0; c < C4; c++)
                    dV[(ulong)bh * steps * DV + t * DV + row * C4 + c] = T(beta * lamk[c]);
            }}
            // Per-dk partial sums over this row's C4 value columns.
            for (uint m = 0; m < M; m++) {{
                const uint d = m * 32 + lane;
                float sdq = 0.0f;
                float sldel = 0.0f;
                float ssdec = 0.0f;
                float sdga = 0.0f;
                for (uint c = 0; c < C4; c++) {{
                    const float sprev = wp[(ulong)d * DV + row * C4 + c];
                    const float stt = ws[(ulong)d * DV + row * C4 + c];
                    const float mm = lam[m][c] - beta * kk[m] * lamk[c];
                    sdq += stt * gg[c];
                    sldel += lam[m][c] * dd[c];
                    ssdec += al[m] * sprev * lamk[c];
                    sdga += sprev * mm;
                    lam[m][c] = al[m] * mm;
                }}
                const uint base = d * 4 + row;
                p_dq[base] = sdq;
                p_ldel[base] = sldel;
                p_sdec[base] = ssdec;
                p_dga[base] = sdga;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (row == 0) {{
                for (uint m = 0; m < M; m++) {{
                    const uint d = m * 32 + lane;
                    const uint base = d * 4;
                    r_dq[d] = p_dq[base] + p_dq[base + 1] + p_dq[base + 2] + p_dq[base + 3];
                    r_ldel[d] = p_ldel[base] + p_ldel[base + 1] + p_ldel[base + 2] + p_ldel[base + 3];
                    r_sdec[d] = p_sdec[base] + p_sdec[base + 1] + p_sdec[base + 2] + p_sdec[base + 3];
                    r_dga[d] = p_dga[base] + p_dga[base + 1] + p_dga[base + 2] + p_dga[base + 3];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float dbp = 0.0f;
            for (uint m = 0; m < M; m++) dbp += kk[m] * r_ldel[m * 32 + lane];
            const float dbv = simd_sum(dbp);
            if (row == 0) {{
                for (uint m = 0; m < M; m++) {{
                    const uint d = m * 32 + lane;
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
            m = dk / 32,
            c4 = dv / 4,
            chunk = 64,
            scale = scale,
        )
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
        let dtype = q.dtype;
        let (h, dk) = (q.layout.shape()[0], q.layout.shape()[1]);
        let dv = v.layout.shape()[1];
        let (q, k, v, g, beta) = (
            wrap_contig(q)?,
            wrap_contig(k)?,
            wrap_contig(v)?,
            wrap_contig(g)?,
            wrap_contig(beta)?,
        );
        let pipe = pipeline(dtype, dk, dv, scale)?;
        let out_buf = MetalDevice::get().alloc(h * dv, dtype);
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
            set_buffer(e, 6, &out_buf, 0);
            e.dispatchThreadgroups_threadsPerThreadgroup(
                objc2_metal::MTLSize {
                    width: 1,
                    height: dv / 4,
                    depth: h,
                },
                objc2_metal::MTLSize {
                    width: 32,
                    height: 4,
                    depth: 1,
                },
            );
        });
        Ok(MetalTensor {
            buffer: out_buf,
            layout: crate::runtime::layout::Layout::contiguous(vec![h, dv]),
            dtype,
        })
    }

    fn fwd_pipeline(dtype: DType, dk: usize, dv: usize, scale: f64) -> crate::err::Res<Pipeline> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        (0xDA02u32, dtype, dk, dv, scale.to_bits()).hash(&mut hasher);
        MetalDevice::get().compile_lazy(hasher.finish(), "et_kda_forward", || {
            fwd_source(dtype, dk, dv, scale)
        })
    }

    fn bwd_pipeline(dtype: DType, dk: usize, dv: usize, scale: f64) -> crate::err::Res<Pipeline> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        (0xDA03u32, dtype, dk, dv, scale.to_bits()).hash(&mut hasher);
        MetalDevice::get().compile_lazy(hasher.finish(), "et_kda_backward", || {
            bwd_source(dtype, dk, dv, scale)
        })
    }

    // Fused forward scan: q/k/g [BH, T, Dk], v [BH, T, Dv], beta [BH, T];
    // optional fp32 initial state [BH, Dk, Dv] and final-state writeback.
    // Returns (output [BH, T, Dv] in the input dtype, final state when
    // requested).
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
        let pipe = fwd_pipeline(dtype, dk, dv, scale)?;
        let out_buf = MetalDevice::get().alloc(bh * steps * dv, dtype);
        let dev = MetalDevice::get();
        let dummy = dev.alloc(1, DType::F32);
        let s0_buf = match initial_state {
            Some(s) => wrap_contig(s)?.buffer.clone(),
            None => dummy.clone(),
        };
        let s1_buf = if write_final_state {
            dev.alloc(bh * dk * dv, DType::F32)
        } else {
            dummy.clone()
        };
        let mut flags = 0u32;
        if initial_state.is_some() {
            flags |= 1;
        }
        if write_final_state {
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
            set_buffer(e, 5, &s0_buf, 0);
            set_buffer(e, 6, &s1_buf, 0);
            set_buffer(e, 7, &out_buf, 0);
            set_bytes(e, 8, &(steps as u32));
            set_bytes(e, 9, &flags);
            e.dispatchThreadgroups_threadsPerThreadgroup(
                objc2_metal::MTLSize {
                    width: 1,
                    height: dv / 4,
                    depth: bh,
                },
                objc2_metal::MTLSize {
                    width: 32,
                    height: 4,
                    depth: 1,
                },
            );
        });
        let out = MetalTensor {
            buffer: out_buf,
            layout: crate::runtime::layout::Layout::contiguous(vec![bh, steps, dv]),
            dtype,
        };
        let final_state = if write_final_state {
            Some(MetalTensor {
                buffer: s1_buf,
                layout: crate::runtime::layout::Layout::contiguous(vec![bh, dk, dv]),
                dtype: DType::F32,
            })
        } else {
            None
        };
        Ok((out, final_state))
    }

    // Fused closed-form backward: same operand contract as the composed
    // reference, g the output cotangent [BH, T, Dv]. Returns (dq, dk, dv,
    // dlog_decay, dbeta) in the input dtype.
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
        let pipe = bwd_pipeline(dtype, dk, dv, scale)?;
        let dev = MetalDevice::get();
        let nchunks = steps.div_ceil(64);
        let ws_floats = bh * ((nchunks + 64) * dk * dv + 64 * dv);
        let ws = dev.alloc(ws_floats.max(1), DType::F32);
        let dq_buf = dev.alloc(bh * steps * dk, dtype);
        let dk_buf = dev.alloc(bh * steps * dk, dtype);
        let dv_buf = dev.alloc(bh * steps * dv, dtype);
        let dg_buf = dev.alloc(bh * steps * dk, dtype);
        let db_buf = dev.alloc(bh * steps, dtype);
        let elem = |off: usize| off * dtype.size_in_bytes();
        dev.with_encoder(|e| {
            e.setComputePipelineState(pipe.as_raw());
            set_buffer(e, 0, &q.buffer, elem(q.layout.offset()));
            set_buffer(e, 1, &k.buffer, elem(k.layout.offset()));
            set_buffer(e, 2, &v.buffer, elem(v.layout.offset()));
            set_buffer(e, 3, &g.buffer, elem(g.layout.offset()));
            set_buffer(e, 4, &beta.buffer, elem(beta.layout.offset()));
            set_buffer(e, 5, &dout.buffer, elem(dout.layout.offset()));
            set_buffer(e, 6, &ws, 0);
            set_buffer(e, 7, &dq_buf, 0);
            set_buffer(e, 8, &dk_buf, 0);
            set_buffer(e, 9, &dv_buf, 0);
            set_buffer(e, 10, &dg_buf, 0);
            set_buffer(e, 11, &db_buf, 0);
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
        let wrap = |buffer, shape: Vec<usize>| MetalTensor {
            buffer,
            layout: crate::runtime::layout::Layout::contiguous(shape),
            dtype,
        };
        Ok((
            wrap(dq_buf, vec![bh, steps, dk]),
            wrap(dk_buf, vec![bh, steps, dk]),
            wrap(dv_buf, vec![bh, steps, dv]),
            wrap(dg_buf, vec![bh, steps, dk]),
            wrap(db_buf, vec![bh, steps, 1]),
        ))
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
    // reference (itself finite-difference-verified), T crossing a chunk
    // boundary.
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

        let max_diff = |x: &[f32], y: &[f32]| {
            x.iter()
                .zip(y.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max)
        };

        // Forward (zero initial state).
        let composed_fwd =
            crate::runtime::metal::composed::kda_chunk_forward(&q, &k, &v, &g, &b, scale).unwrap();
        let (fused_fwd, final_state) =
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
}

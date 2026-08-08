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
pub use metal::decode;

#[cfg(target_os = "macos")]
mod metal {
    use crate::runtime::dtype::DType;
    use crate::runtime::metal::device::{set_buffer, MetalDevice, Pipeline};
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

    // One slot's single-token step: q/k/g [H, Dk], v [H, Dv], beta [H],
    // state [H, Dk, Dv] fp32 read and written in place. Returns o as
    // [H, Dv] in the input dtype.
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
}

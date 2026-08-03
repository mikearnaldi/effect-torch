//! Flash attention on Metal: a single-kernel forward (tiled, online
//! softmax, the score matrix never materializes) and a chunked-recompute
//! backward (flash-2 structure orchestrated with candle ops, so memory
//! stays bounded by the chunk instead of the full [T, S] score matrix).
//! Both are f32/Metal-only execution strategies for the semantic
//! `Tensor.scaledDotProductAttention` node; the composed candle path in
//! lib.rs remains the reference and the fallback for other device/dtype
//! pairs.

use candle_core::{D, DType, Device, Tensor};

const TILE_Q: usize = 16;
const TILE_K: usize = 32;
const THREADS: usize = 128;
const BACKWARD_CHUNK: usize = 64;

/// Whether the flash path can run this forward: Metal f32, any shapes.
pub fn is_supported(q: &Tensor) -> bool {
    matches!(q.device(), Device::Metal(_)) && q.dtype() == DType::F32
}

/// Whether the fused two-kernel backward can run: Metal f32, head dims
/// within the threadgroup tile budget (D ≤ 64 covers the nano regime;
/// larger falls back to the composed recompute).
pub fn backward_supported(q: &Tensor, v: &Tensor) -> bool {
    let rank = q.rank();
    let d = q.dim(rank - 1).unwrap_or(usize::MAX);
    let dv = v.dim(rank - 1).unwrap_or(usize::MAX);
    is_supported(q) && d == dv && d <= 64
}

#[cfg(target_os = "macos")]
mod metal {
    use super::{THREADS, TILE_K, TILE_Q};
    use crate::bridge;
    use crate::runtime::metal::device::{set_buffer, MetalDevice, Pipeline};
    use crate::runtime::metal::run::MetalTensor;
    use candle_core::{DType, Tensor};
    use objc2_metal::MTLComputeCommandEncoder;
    use std::sync::Arc;

    fn wrap_contig(t: &Tensor) -> candle_core::Result<MetalTensor> {
        let w = bridge::metal::wrap(t)?;
        if w.layout.is_contiguous() {
            Ok(w)
        } else {
            crate::runtime::metal::kernels::strided_copy(MetalDevice::get(), &w)
                .map_err(candle_core::Error::Msg)
        }
    }

    fn alloc_f32(n: usize) -> Arc<crate::runtime::metal::device::Buffer> {
        MetalDevice::get().alloc(n.max(1), crate::runtime::dtype::DType::F32)
    }

    // The forward kernel: one threadgroup per (query tile, batch*head).
    // Scores for a key tile are computed into threadgroup memory, folded
    // into the running (max, sum) online softmax, and consumed by the
    // P·V accumulation in place — the [T, S] matrix never exists.
    // Everything shape-dependent is baked in as #defines (keying the
    // pipeline cache): T, S, D, DV, the scale, causal.
    fn kernel_source(
        t: usize,
        s: usize,
        d: usize,
        dv: usize,
        scale: f64,
        causal: bool,
    ) -> String {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;

#define T {t}
#define S {s}
#define D {d}
#define DV {dv}
#define TQ {tq}
#define TK {tk}
#define NT {nt}
#define SCALE {scale:?}f
#define CAUSAL {causal}
// Right-aligned causal window: query q attends to keys k <= q + OFFSET.
#define OFFSET {offset}

kernel void et_sdpa_fwd(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    device float* L [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {{
    const int q0 = tgid.x * TQ;
    const long bh = tgid.y;
    const uint tid = tpitg.x;
    device const float* Qb = Q + bh * (long)T * D;
    device const float* Kb = K + bh * (long)S * D;
    device const float* Vb = V + bh * (long)S * DV;
    device float* Ob = O + bh * (long)T * DV;

    threadgroup float St[TQ][TK];
    threadgroup float Ot[TQ][DV];
    threadgroup float corr[TQ];
    threadgroup float m[TQ];
    threadgroup float l[TQ];

    for (int i = tid; i < TQ; i += NT) {{ m[i] = -INFINITY; l[i] = 0.0f; }}
    for (int i = tid; i < TQ * DV; i += NT) {{ Ot[i / DV][i % DV] = 0.0f; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int kt = 0; kt < S; kt += TK) {{
        // Tiles fully past the causal diagonal contribute nothing.
        if (CAUSAL && kt > q0 + TQ - 1 + OFFSET) {{ break; }}
        for (int i = tid; i < TQ * TK; i += NT) {{
            int qi = i / TK;
            int kj = i % TK;
            int q = q0 + qi;
            int k = kt + kj;
            float acc;
            if (q < T && k < S && (!CAUSAL || k <= q + OFFSET)) {{
                acc = 0.0f;
                for (int d = 0; d < D; d++) {{ acc += Qb[q * D + d] * Kb[k * D + d]; }}
                acc *= SCALE;
            }} else {{
                acc = -INFINITY;
            }}
            St[qi][kj] = acc;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Online softmax update, one thread per query row. Dead rows
        // (q >= T) keep mn == -inf: corr and p collapse to 0 and the
        // epilogue skips them.
        for (int i = tid; i < TQ; i += NT) {{
            float mx = -INFINITY;
            for (int kj = 0; kj < TK; kj++) {{ mx = max(mx, St[i][kj]); }}
            float mn = max(m[i], mx);
            float c = (mn == -INFINITY) ? 0.0f : exp(m[i] - mn);
            float s = 0.0f;
            for (int kj = 0; kj < TK; kj++) {{
                float p = (mn == -INFINITY) ? 0.0f : exp(St[i][kj] - mn);
                St[i][kj] = p;
                s += p;
            }}
            l[i] = l[i] * c + s;
            m[i] = mn;
            corr[i] = c;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = O * corr + P · V (P is St, already exponentiated).
        for (int i = tid; i < TQ * DV; i += NT) {{
            int qi = i / DV;
            int dv = i % DV;
            float acc = Ot[qi][dv] * corr[qi];
            for (int kj = 0; kj < TK; kj++) {{
                int k = kt + kj;
                if (k < S) {{ acc += St[qi][kj] * Vb[k * DV + dv]; }}
            }}
            Ot[qi][dv] = acc;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    for (int i = tid; i < TQ * DV; i += NT) {{
        int qi = i / DV;
        int dv = i % DV;
        int q = q0 + qi;
        if (q < T) {{ Ob[q * DV + dv] = Ot[qi][dv] / l[qi]; }}
    }}
    // L = logsumexp(scores) per row, for the backward's P recomputation.
    for (int i = tid; i < TQ; i += NT) {{
        int q = q0 + i;
        if (q < T) {{ L[bh * (long)T + q] = m[i] + log(l[i]); }}
    }}
}}
"#,
            t = t,
            s = s,
            d = d,
            dv = dv,
            tq = TILE_Q,
            tk = TILE_K,
            nt = THREADS,
            scale = scale as f32,
            causal = if causal { 1 } else { 0 },
            offset = s.saturating_sub(t),
        )
    }

    fn pipeline(
        _mdev: &candle_core::MetalDevice,
        t: usize,
        s: usize,
        d: usize,
        dv: usize,
        scale: f64,
        causal: bool,
    ) -> candle_core::Result<Pipeline> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        (t, s, d, dv, scale.to_bits(), causal).hash(&mut hasher);
        let key = hasher.finish();
        let src = kernel_source(t, s, d, dv, scale, causal);
        MetalDevice::get()
            .compile(key, &src, "et_sdpa_fwd")
            .map_err(candle_core::Error::Msg)
    }

    pub fn forward(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f64,
        causal: bool,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let rank = q.rank();
        let (t, d) = (q.dim(rank - 2)?, q.dim(rank - 1)?);
        let (s, dv) = (v.dim(rank - 2)?, v.dim(rank - 1)?);
        let bh: usize = q.dims()[..rank - 2].iter().product();
        let out_shape: Vec<usize> = q.dims()[..rank - 1].iter().copied().chain([dv]).collect();
        let l_shape: Vec<usize> = q.dims()[..rank - 1].to_vec();
        let device = q.device();
        let mdev = device.as_metal_device()?;
        device.synchronize()?;
        // Flatten leading dims: [BH, T, D] — free on contiguous tensors.
        let q = wrap_contig(&q.reshape((bh, t, d))?)?;
        let k = wrap_contig(&k.reshape((bh, s, d))?)?;
        let v = wrap_contig(&v.reshape((bh, s, dv))?)?;
        let pipe = pipeline(mdev, t, s, d, dv, scale, causal)?;
        let o_buf = alloc_f32(bh * t * dv);
        let l_buf = alloc_f32(bh * t);
        let byte_offset = |off: usize| off * DType::F32.size_in_bytes();
        MetalDevice::get().with_encoder(|e| {
            e.setComputePipelineState(pipe.as_raw());
            set_buffer(e, 0, &q.buffer, byte_offset(q.layout.offset()));
            set_buffer(e, 1, &k.buffer, byte_offset(k.layout.offset()));
            set_buffer(e, 2, &v.buffer, byte_offset(v.layout.offset()));
            set_buffer(e, 3, &o_buf, 0);
            set_buffer(e, 4, &l_buf, 0);
            e.dispatchThreadgroups_threadsPerThreadgroup(
                objc2_metal::MTLSize {
                    width: t.div_ceil(TILE_Q),
                    height: bh,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: THREADS,
                    height: 1,
                    depth: 1,
                },
            );
        });
        MetalDevice::get().synchronize();
        let o = bridge::metal::unwrap(&o_buf, vec![bh, t, dv], DType::F32, mdev)?.reshape(out_shape)?;
        let l = bridge::metal::unwrap(&l_buf, vec![bh, t], DType::F32, mdev)?.reshape(l_shape)?;
        Ok((o, l))
    }
    // The fused backward: two kernels, no atomics. Pass A (key-tiled)
    // accumulates dk/dv — thread (kj-cell) owns its accumulator in
    // registers across the whole query sweep. Pass B (query-tiled)
    // accumulates dq. The score tile is recomputed once per tile pair
    // in threadgroup memory and consumed by all four gradients — the
    // shared-data win over the composed recompute (~12 DRAM round
    // trips per tile).
    fn bwd_source(t: usize, s: usize, d: usize, dv: usize, scale: f64, causal: bool) -> String {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;

#define T {t}
#define S {s}
#define D {d}
#define DV {dv}
#define TQ {tq}
#define TK {tk}
#define NT {nt}
#define SCALE {scale:?}f
#define CAUSAL {causal}
#define OFFSET {offset}
#define CELLS_DK ((TK * D + NT - 1) / NT)
#define CELLS_DV ((TK * DV + NT - 1) / NT)
#define CELLS_DQ ((TQ * D + NT - 1) / NT)

kernel void et_sdpa_bwd_kv(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device const float* G [[buffer(3)]],
    device const float* L [[buffer(4)]],
    device const float* Dvec [[buffer(5)]],
    device float* DK [[buffer(6)]],
    device float* DVout [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {{
    const int j0 = tgid.x * TK;
    const long bh = tgid.y;
    const uint tid = tpitg.x;
    device const float* Qb = Q + bh * (long)T * D;
    device const float* Kb = K + bh * (long)S * D;
    device const float* Vb = V + bh * (long)S * DV;
    device const float* Gb = G + bh * (long)T * DV;
    device const float* Lb = L + bh * (long)T;
    device const float* Db = Dvec + bh * (long)T;
    device float* DKb = DK + bh * (long)S * D;
    device float* DVb = DVout + bh * (long)S * DV;

    threadgroup float Kt[TK][D];
    threadgroup float Vt[TK][DV];
    for (int i = tid; i < TK * D; i += NT) {{
        int kj = i / D, dd = i % D;
        Kt[kj][dd] = (j0 + kj < S) ? Kb[(j0 + kj) * D + dd] : 0.0f;
    }}
    for (int i = tid; i < TK * DV; i += NT) {{
        int kj = i / DV, dv = i % DV;
        Vt[kj][dv] = (j0 + kj < S) ? Vb[(j0 + kj) * DV + dv] : 0.0f;
    }}
    float acc_dk[CELLS_DK];
    float acc_dv[CELLS_DV];
    for (int w = 0; w < CELLS_DK; w++) {{ acc_dk[w] = 0.0f; }}
    for (int w = 0; w < CELLS_DV; w++) {{ acc_dv[w] = 0.0f; }}
    threadgroup float St[TQ][TK];
    threadgroup float Pt[TQ][TK];
    threadgroup float Qt[TQ][D];
    threadgroup float Gt[TQ][DV];
    threadgroup float lt[TQ];
    threadgroup float dt[TQ];

    for (int i0 = 0; i0 < T; i0 += TQ) {{
        // No causal early-out here: later query tiles attend key tiles
        // that earlier ones do not (the per-cell mask handles it).
        for (int i = tid; i < TQ * D; i += NT) {{
            int t = i / D, dd = i % D;
            Qt[t][dd] = (i0 + t < T) ? Qb[(i0 + t) * D + dd] : 0.0f;
        }}
        for (int i = tid; i < TQ * DV; i += NT) {{
            int t = i / DV, dv = i % DV;
            Gt[t][dv] = (i0 + t < T) ? Gb[(i0 + t) * DV + dv] : 0.0f;
        }}
        for (int i = tid; i < TQ; i += NT) {{
            lt[i] = (i0 + i < T) ? Lb[i0 + i] : 0.0f;
            dt[i] = (i0 + i < T) ? Db[i0 + i] : 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < TQ * TK; i += NT) {{
            int t = i / TK, kj = i % TK;
            int q = i0 + t, k = j0 + kj;
            float p;
            if (q < T && k < S && (!CAUSAL || k <= q + OFFSET)) {{
                float acc = 0.0f;
                for (int dd = 0; dd < D; dd++) {{ acc += Qt[t][dd] * Kt[kj][dd]; }}
                p = exp(acc * SCALE - lt[t]);
            }} else {{
                p = 0.0f;
            }}
            Pt[t][kj] = p;
            float dp = 0.0f;
            for (int dv = 0; dv < DV; dv++) {{ dp += Gt[t][dv] * Vt[kj][dv]; }}
            St[t][kj] = p * (dp - dt[t]) * SCALE;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int w = 0; w < CELLS_DK; w++) {{
            int c = tid + w * NT;
            if (c >= TK * D) {{ break; }}
            int kj = c / D, dd = c % D;
            float acc = 0.0f;
            for (int t = 0; t < TQ; t++) {{ acc += St[t][kj] * Qt[t][dd]; }}
            acc_dk[w] += acc;
        }}
        for (int w = 0; w < CELLS_DV; w++) {{
            int c = tid + w * NT;
            if (c >= TK * DV) {{ break; }}
            int kj = c / DV, dv = c % DV;
            float acc = 0.0f;
            for (int t = 0; t < TQ; t++) {{ acc += Pt[t][kj] * Gt[t][dv]; }}
            acc_dv[w] += acc;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    for (int w = 0; w < CELLS_DK; w++) {{
        int c = tid + w * NT;
        if (c >= TK * D) {{ break; }}
        int kj = c / D, dd = c % D;
        if (j0 + kj < S) {{ DKb[(j0 + kj) * D + dd] = acc_dk[w]; }}
    }}
    for (int w = 0; w < CELLS_DV; w++) {{
        int c = tid + w * NT;
        if (c >= TK * DV) {{ break; }}
        int kj = c / DV, dv = c % DV;
        if (j0 + kj < S) {{ DVb[(j0 + kj) * DV + dv] = acc_dv[w]; }}
    }}
}}

kernel void et_sdpa_bwd_q(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device const float* G [[buffer(3)]],
    device const float* L [[buffer(4)]],
    device const float* Dvec [[buffer(5)]],
    device float* DQ [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {{
    const int i0 = tgid.x * TQ;
    const long bh = tgid.y;
    const uint tid = tpitg.x;
    device const float* Qb = Q + bh * (long)T * D;
    device const float* Kb = K + bh * (long)S * D;
    device const float* Vb = V + bh * (long)S * DV;
    device const float* Gb = G + bh * (long)T * DV;
    device const float* Lb = L + bh * (long)T;
    device const float* Db = Dvec + bh * (long)T;
    device float* DQb = DQ + bh * (long)T * D;

    threadgroup float Qt[TQ][D];
    threadgroup float Gt[TQ][DV];
    threadgroup float lt[TQ];
    threadgroup float dt[TQ];
    for (int i = tid; i < TQ * D; i += NT) {{
        int t = i / D, dd = i % D;
        Qt[t][dd] = (i0 + t < T) ? Qb[(i0 + t) * D + dd] : 0.0f;
    }}
    for (int i = tid; i < TQ * DV; i += NT) {{
        int t = i / DV, dv = i % DV;
        Gt[t][dv] = (i0 + t < T) ? Gb[(i0 + t) * DV + dv] : 0.0f;
    }}
    for (int i = tid; i < TQ; i += NT) {{
        lt[i] = (i0 + i < T) ? Lb[i0 + i] : 0.0f;
        dt[i] = (i0 + i < T) ? Db[i0 + i] : 0.0f;
    }}
    float acc_dq[CELLS_DQ];
    for (int w = 0; w < CELLS_DQ; w++) {{ acc_dq[w] = 0.0f; }}
    threadgroup float St[TQ][TK];
    threadgroup float Pt[TQ][TK];
    threadgroup float Kt[TK][D];
    threadgroup float Vt[TK][DV];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int j0 = 0; j0 < S; j0 += TK) {{
        if (CAUSAL && j0 > i0 + TQ - 1 + OFFSET) {{ break; }}
        for (int i = tid; i < TK * D; i += NT) {{
            int kj = i / D, dd = i % D;
            Kt[kj][dd] = (j0 + kj < S) ? Kb[(j0 + kj) * D + dd] : 0.0f;
        }}
        for (int i = tid; i < TK * DV; i += NT) {{
            int kj = i / DV, dv = i % DV;
            Vt[kj][dv] = (j0 + kj < S) ? Vb[(j0 + kj) * DV + dv] : 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < TQ * TK; i += NT) {{
            int t = i / TK, kj = i % TK;
            int q = i0 + t, k = j0 + kj;
            float p;
            if (q < T && k < S && (!CAUSAL || k <= q + OFFSET)) {{
                float acc = 0.0f;
                for (int dd = 0; dd < D; dd++) {{ acc += Qt[t][dd] * Kt[kj][dd]; }}
                p = exp(acc * SCALE - lt[t]);
            }} else {{
                p = 0.0f;
            }}
            Pt[t][kj] = p;
            float dp = 0.0f;
            for (int dv = 0; dv < DV; dv++) {{ dp += Gt[t][dv] * Vt[kj][dv]; }}
            St[t][kj] = p * (dp - dt[t]) * SCALE;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int w = 0; w < CELLS_DQ; w++) {{
            int c = tid + w * NT;
            if (c >= TQ * D) {{ break; }}
            int t = c / D, dd = c % D;
            float acc = 0.0f;
            for (int kj = 0; kj < TK; kj++) {{ acc += St[t][kj] * Kt[kj][dd]; }}
            acc_dq[w] += acc;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    for (int w = 0; w < CELLS_DQ; w++) {{
        int c = tid + w * NT;
        if (c >= TQ * D) {{ break; }}
        int t = c / D, dd = c % D;
        if (i0 + t < T) {{ DQb[(i0 + t) * D + dd] = acc_dq[w]; }}
    }}
}}
"#,
            t = t,
            s = s,
            d = d,
            dv = dv,
            tq = TILE_Q,
            tk = TILE_K,
            nt = THREADS,
            scale = scale as f32,
            causal = if causal { 1 } else { 0 },
            offset = s.saturating_sub(t),
        )
    }

    fn bwd_pipeline(
        _mdev: &candle_core::MetalDevice,
        name: &'static str,
        t: usize,
        s: usize,
        d: usize,
        dv: usize,
        scale: f64,
        causal: bool,
    ) -> candle_core::Result<Pipeline> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        (name, t, s, d, dv, scale.to_bits(), causal).hash(&mut hasher);
        let key = hasher.finish();
        let src = bwd_source(t, s, d, dv, scale, causal);
        MetalDevice::get()
            .compile(key, &src, name)
            .map_err(candle_core::Error::Msg)
    }

    // The fused backward: d_vec via candle (one small op), then two
    // kernel launches. Returns (dq, dk, dv) [BH, T/S, D].
    #[allow(clippy::too_many_arguments)]
    pub fn backward_fused(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        o: &Tensor,
        l: &Tensor,
        g: &Tensor,
        scale: f64,
        causal: bool,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let rank = q.rank();
        let (t, s, d, dv) = (q.dim(rank - 2)?, k.dim(rank - 2)?, q.dim(rank - 1)?, v.dim(rank - 1)?);
        let bh: usize = q.dims()[..rank - 2].iter().product();
        let (q_shape, k_shape, v_shape) = (q.dims().to_vec(), k.dims().to_vec(), v.dims().to_vec());
        let device = q.device();
        let mdev = device.as_metal_device()?;
        let qr = q.reshape((bh, t, d))?;
        let kr = k.reshape((bh, s, d))?;
        let vr = v.reshape((bh, s, dv))?;
        let or_ = o.reshape((bh, t, dv))?;
        let gr = g.reshape((bh, t, dv))?;
        // d_vec = rowsum(g ∘ o): one small op through the migrated arms.
        let d_vec = (&gr * &or_)?.sum(candle_core::D::Minus1)?;
        device.synchronize()?;
        let q = wrap_contig(&qr)?;
        let k = wrap_contig(&kr)?;
        let v = wrap_contig(&vr)?;
        let o = wrap_contig(&or_)?;
        let g = wrap_contig(&gr)?;
        let l = wrap_contig(&l.reshape(bh * t)?)?;
        let d_vec = wrap_contig(&d_vec)?;
        let dq_buf = alloc_f32(bh * t * d);
        let dk_buf = alloc_f32(bh * s * d);
        let dv_buf = alloc_f32(bh * s * dv);
        let off = |o: usize| o * DType::F32.size_in_bytes();
        {
            let pipe = bwd_pipeline(mdev, "et_sdpa_bwd_kv", t, s, d, dv, scale, causal)?;
            MetalDevice::get().with_encoder(|e| {
                e.setComputePipelineState(pipe.as_raw());
                set_buffer(e, 0, &q.buffer, off(q.layout.offset()));
                set_buffer(e, 1, &k.buffer, off(k.layout.offset()));
                set_buffer(e, 2, &v.buffer, off(v.layout.offset()));
                set_buffer(e, 3, &g.buffer, off(g.layout.offset()));
                set_buffer(e, 4, &l.buffer, off(l.layout.offset()));
                set_buffer(e, 5, &d_vec.buffer, off(d_vec.layout.offset()));
                set_buffer(e, 6, &dk_buf, 0);
                set_buffer(e, 7, &dv_buf, 0);
                e.dispatchThreadgroups_threadsPerThreadgroup(
                    objc2_metal::MTLSize {
                        width: s.div_ceil(TILE_K),
                        height: bh,
                        depth: 1,
                    },
                    objc2_metal::MTLSize {
                        width: THREADS,
                        height: 1,
                        depth: 1,
                    },
                );
            });
        }
        {
            let pipe = bwd_pipeline(mdev, "et_sdpa_bwd_q", t, s, d, dv, scale, causal)?;
            MetalDevice::get().with_encoder(|e| {
                e.setComputePipelineState(pipe.as_raw());
                set_buffer(e, 0, &q.buffer, off(q.layout.offset()));
                set_buffer(e, 1, &k.buffer, off(k.layout.offset()));
                set_buffer(e, 2, &v.buffer, off(v.layout.offset()));
                set_buffer(e, 3, &g.buffer, off(g.layout.offset()));
                set_buffer(e, 4, &l.buffer, off(l.layout.offset()));
                set_buffer(e, 5, &d_vec.buffer, off(d_vec.layout.offset()));
                set_buffer(e, 6, &dq_buf, 0);
                e.dispatchThreadgroups_threadsPerThreadgroup(
                    objc2_metal::MTLSize {
                        width: t.div_ceil(TILE_Q),
                        height: bh,
                        depth: 1,
                    },
                    objc2_metal::MTLSize {
                        width: THREADS,
                        height: 1,
                        depth: 1,
                    },
                );
            });
        }
        MetalDevice::get().synchronize();
        let dq = bridge::metal::unwrap(&dq_buf, vec![bh, t, d], DType::F32, mdev)?.reshape(q_shape)?;
        let dk = bridge::metal::unwrap(&dk_buf, vec![bh, s, d], DType::F32, mdev)?.reshape(k_shape)?;
        let dv = bridge::metal::unwrap(&dv_buf, vec![bh, s, dv], DType::F32, mdev)?.reshape(v_shape)?;
        Ok((dq, dk, dv))
    }
}

#[cfg(not(target_os = "macos"))]
mod metal {
    use candle_core::Tensor;
    pub fn forward(
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _scale: f64,
        _causal: bool,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        Err(candle_core::Error::Msg(
            "sdpa flash: Metal is only available on macOS".to_string(),
        ))
    }
    #[allow(clippy::too_many_arguments)]
    pub fn backward_fused(
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _o: &Tensor,
        _l: &Tensor,
        _g: &Tensor,
        _scale: f64,
        _causal: bool,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        unreachable!("fused sdpa backward is Metal-only")
    }
}

/// The flash forward: O and the per-row logsumexp L (consumed by the
/// chunked backward). Caller checks `is_supported` first.
pub fn forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    causal: bool,
) -> candle_core::Result<(Tensor, Tensor)> {
    metal::forward(q, k, v, scale, causal)
}

// The additive causal mask [T, C] for the key chunk starting at `jt`:
// 0 where jt + j <= i + (S - T) (the right-aligned window the forward
// kernel uses), -inf elsewhere. Applied before exp, so masked
// probabilities are exactly 0 (never inf × 0 = NaN).
fn chunk_additive_mask(
    t: usize,
    jt: usize,
    c: usize,
    offset: usize,
    dtype: DType,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let i = (Tensor::arange(0u32, t as u32, device)? + offset as f64)?.reshape((t, 1))?;
    let j = Tensor::arange(jt as u32, (jt + c) as u32, device)?.reshape((1, c))?;
    let allowed = j.broadcast_le(&i)?;
    let zeros = Tensor::zeros((t, c), dtype, device)?;
    let neg = match dtype {
        DType::F32 => Tensor::full(f32::NEG_INFINITY, (t, c), device)?,
        DType::F64 => Tensor::full(f64::NEG_INFINITY, (t, c), device)?,
        dtype => return Err(candle_core::Error::UnsupportedDTypeForOp(dtype, "sdpa").bt()),
    };
    allowed.where_cond(&zeros, &neg)
}

/// The chunked backward (flash-2 recomputation structure): with L from
/// the forward, P = exp(S − L) is rebuilt one key chunk at a time, so
/// the full [T, S] matrix never materializes. dV and dK are direct
/// per-chunk writes (concatenated at the end); dQ accumulates across
/// chunks. D = rowsum(dO ∘ O) is computed once.
pub fn backward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    o: &Tensor,
    l: &Tensor,
    g: &Tensor,
    scale: f64,
    causal: bool,
) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
    if backward_supported(q, v) {
        return metal::backward_fused(q, k, v, o, l, g, scale, causal);
    }
    let rank = q.rank();
    let (t, s) = (q.dim(rank - 2)?, k.dim(rank - 2)?);
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let d_vec = ((&g * o)?).sum(D::Minus1)?.unsqueeze(D::Minus1)?;
    let l_col = l.unsqueeze(D::Minus1)?;
    let mut dq: Option<Tensor> = None;
    let mut dvs = Vec::new();
    let mut dks = Vec::new();
    let mut jt = 0;
    while jt < s {
        let c = BACKWARD_CHUNK.min(s - jt);
        let kj = k.narrow(rank - 2, jt, c)?.contiguous()?;
        let vj = v.narrow(rank - 2, jt, c)?.contiguous()?;
        let mut sj = (q.broadcast_matmul(&kj.transpose(rank - 2, rank - 1)?.contiguous()?)? * scale)?;
        if causal {
            sj = sj.broadcast_add(&chunk_additive_mask(
                t,
                jt,
                c,
                s.saturating_sub(t),
                sj.dtype(),
                sj.device(),
            )?)?;
        }
        let pj = sj.broadcast_sub(&l_col)?.exp()?;
        dvs.push(
            pj.transpose(rank - 2, rank - 1)?
                .contiguous()?
                .broadcast_matmul(&g)?,
        );
        let dpj = g.broadcast_matmul(&vj.transpose(rank - 2, rank - 1)?.contiguous()?)?;
        let dsj = ((&pj * &dpj.broadcast_sub(&d_vec)?)? * scale)?;
        dks.push(
            dsj.transpose(rank - 2, rank - 1)?
                .contiguous()?
                .broadcast_matmul(&q)?,
        );
        let dqj = dsj.broadcast_matmul(&kj)?;
        dq = Some(match dq {
            None => dqj,
            Some(acc) => (acc + dqj)?,
        });
        jt += c;
    }
    let dv = Tensor::cat(&dvs, rank - 2)?;
    let dk = Tensor::cat(&dks, rank - 2)?;
    Ok((dq.expect("sdpa backward: empty key sequence"), dk, dv))
}

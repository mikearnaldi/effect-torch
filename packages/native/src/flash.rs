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

#[cfg(target_os = "macos")]
mod metal {
    use super::{THREADS, TILE_K, TILE_Q};
    use candle_core::{DType, MetalStorage, Storage, Tensor};
    use candle_metal_kernels::metal::ComputePipeline;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    fn pipelines() -> &'static Mutex<HashMap<u64, ComputePipeline>> {
        static CACHE: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
        CACHE.get_or_init(|| Mutex::new(HashMap::new()))
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
        mdev: &candle_core::MetalDevice,
        t: usize,
        s: usize,
        d: usize,
        dv: usize,
        scale: f64,
        causal: bool,
    ) -> candle_core::Result<ComputePipeline> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        (t, s, d, dv, scale.to_bits(), causal).hash(&mut hasher);
        let key = hasher.finish();
        let mut cache = pipelines().lock().unwrap();
        if let Some(p) = cache.get(&key) {
            return Ok(p.clone());
        }
        let src = kernel_source(t, s, d, dv, scale, causal);
        #[allow(deprecated)]
        let opts = {
            let o = objc2_metal::MTLCompileOptions::new();
            o.setFastMathEnabled(false);
            o
        };
        let lib = mdev
            .device()
            .new_library_with_source(&src, Some(&opts))
            .map_err(|e| candle_core::Error::Msg(format!("sdpa flash: {e}")))?;
        let func = lib
            .get_function("et_sdpa_fwd", None)
            .map_err(|e| candle_core::Error::Msg(format!("sdpa flash: {e}")))?;
        let p = mdev
            .device()
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| candle_core::Error::Msg(format!("sdpa flash: {e}")))?;
        if std::env::var_os("EFFECT_TORCH_FUSION_DEBUG").is_some() {
            eprintln!("[sdpa] compiled flash kernel #{} (key {key:x})", cache.len() + 1);
        }
        cache.insert(key, p.clone());
        Ok(p)
    }

    fn buffer_of(t: &Tensor) -> candle_core::Result<(candle_metal_kernels::metal::Buffer, usize)> {
        let (storage, layout) = t.storage_and_layout();
        match &*storage {
            Storage::Metal(m) => Ok((m.buffer().clone(), layout.start_offset())),
            _ => Err(candle_core::Error::Msg(
                "sdpa flash: expected Metal storage".to_string(),
            )),
        }
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
        // Flatten leading dims: [BH, T, D] — free on contiguous tensors.
        let q = q.reshape((bh, t, d))?.contiguous()?;
        let k = k.reshape((bh, s, d))?.contiguous()?;
        let v = v.reshape((bh, s, dv))?.contiguous()?;
        let pipe = pipeline(mdev, t, s, d, dv, scale, causal)?;
        let o_buf = mdev.new_buffer(bh * t * dv, DType::F32, "sdpa_o")?;
        let l_buf = mdev.new_buffer(bh * t, DType::F32, "sdpa_l")?;
        let encoder = mdev.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipe);
        let byte_offset = |off: usize| off * DType::F32.size_in_bytes();
        let (qb, qo) = buffer_of(&q)?;
        let (kb, ko) = buffer_of(&k)?;
        let (vb, vo) = buffer_of(&v)?;
        encoder.set_buffer(0, Some(&qb), byte_offset(qo));
        encoder.set_buffer(1, Some(&kb), byte_offset(ko));
        encoder.set_buffer(2, Some(&vb), byte_offset(vo));
        encoder.set_buffer(3, Some(&o_buf), 0);
        encoder.set_buffer(4, Some(&l_buf), 0);
        let q_tiles = t.div_ceil(TILE_Q);
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: q_tiles,
                height: bh,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: THREADS,
                height: 1,
                depth: 1,
            },
        );
        // no end_encoding: candle's Commands owns the encoder lifecycle
        let mk = |buf, n, shape: Vec<usize>| {
            Tensor::from_storage(
                Storage::Metal(MetalStorage::new(buf, mdev.clone(), n, DType::F32)),
                shape,
                candle_core::op::BackpropOp::none(),
                false,
            )
        };
        let o = mk(o_buf, bh * t * dv, vec![bh, t, dv]).reshape(out_shape)?;
        let l = mk(l_buf, bh * t, vec![bh, t]).reshape(l_shape)?;
        Ok((o, l))
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

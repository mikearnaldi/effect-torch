//! Paged decode attention on Metal (RFC 0013, stage 2): one kernel
//! launch attends q [B, H, 1, D] over pool slabs IN PLACE — K/V rows
//! are read through the block table (pages), never gathered into a
//! contiguous copy. One threadgroup per (sequence slot, head) streams
//! its slot's blocks with online-softmax accumulation, so the context
//! length is a runtime value (unlike the training flash pipeline,
//! nothing shape-dependent is baked). Slab dtypes f16/bf16 load
//! natively; int8 slabs dequantize in registers with the per-(token,
//! head) scale slab (RFC 0012). The composed scatter+gather path in
//! lib.rs remains the reference and the CPU fallback.

use candle_core::{DType, Device, Tensor};

/// Whether the paged kernel can run this decode step: Metal, f32
/// compute, head dim within the kernel's register budget, slabs in a
/// supported storage dtype.
pub fn is_supported(q: &Tensor, slab_dtype: DType, head_dim: usize) -> bool {
    matches!(q.device(), Device::Metal(_))
        && q.dtype() == DType::F32
        && head_dim <= 128
        && matches!(slab_dtype, DType::F32 | DType::F16 | DType::BF16 | DType::U8)
}

#[cfg(target_os = "macos")]
pub use metal::{decode, scatter};

#[cfg(target_os = "macos")]
mod metal {
    use candle_core::{DType, MetalStorage, Storage, Tensor};
    use candle_metal_kernels::metal::ComputePipeline;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    const THREADS: usize = 128;

    fn pipelines() -> &'static Mutex<HashMap<u64, ComputePipeline>> {
        static CACHE: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
        CACHE.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn compile(
        mdev: &candle_core::MetalDevice,
        key: u64,
        src: &str,
        name: &'static str,
    ) -> candle_core::Result<ComputePipeline> {
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
            .new_library_with_source(src, Some(&opts))
            .map_err(|e| candle_core::Error::Msg(format!("paged {name}: {e}")))?;
        let func = lib
            .get_function(name, None)
            .map_err(|e| candle_core::Error::Msg(format!("paged {name}: {e}")))?;
        let p = mdev
            .device()
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| candle_core::Error::Msg(format!("paged {name}: {e}")))?;
        cache.insert(key, p.clone());
        Ok(p)
    }

    fn buffer_of(t: &Tensor) -> candle_core::Result<(candle_metal_kernels::metal::Buffer, usize)> {
        let (storage, layout) = t.storage_and_layout();
        match &*storage {
            Storage::Metal(m) => Ok((m.buffer().clone(), layout.start_offset())),
            _ => Err(candle_core::Error::Msg(
                "paged decode: expected Metal storage".to_string(),
            )),
        }
    }

    // Writes the new-token row of every slot into the slabs in one
    // launch (per layer): one threadgroup per (slot, head) computes
    // its row's physical address and stores D values, quantizing with
    // an in-threadgroup absmax for int8 (same grid as the composed
    // scatter: absmax/127 + eps, round, offset 128).
    fn scatter_source(d: usize, slab_dtype: DType) -> String {
        let (kv_ty, int8) = match slab_dtype {
            DType::F32 => ("float", 0),
            DType::F16 => ("half", 0),
            DType::BF16 => ("bfloat", 0),
            DType::U8 => ("uchar", 1),
            other => unreachable!("paged scatter: unsupported slab dtype {other:?}"),
        };
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;

#define D {d}
#define NT 32
#define T_KV {kv_ty}
#define INT8 {int8}

kernel void et_paged_scatter(
    device const float* Kn [[buffer(0)]],
    device const float* Vn [[buffer(1)]],
    device T_KV* K [[buffer(2)]],
    device T_KV* V [[buffer(3)]],
    device float* kscales [[buffer(4)]],
    device float* vscales [[buffer(5)]],
    device const uint* tables [[buffer(6)]],
    device const uint* ctxlens [[buffer(7)]],
    constant uint& blockSize [[buffer(8)]],
    constant uint& maxBlocks [[buffer(9)]],
    constant uint& H [[buffer(10)]],
    constant uint& advance [[buffer(11)]],
    uint3 gridDim [[threadgroups_per_grid]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {{
    const uint b = tgid.y;
    const uint h = tgid.x;
    const uint tid = tpitg.x;
    const uint C = gridDim.z;
    const uint cursor = ctxlens[b] - advance;
    // Rows cursor..needed, one per new token; D-wide within each row.
    for (uint p = 0; p < advance; p++) {{
        const uint pos = cursor + p;
        const uint phys = tables[(ulong)b * maxBlocks + pos / blockSize] * blockSize + (pos % blockSize);
        const ulong dst = (ulong)phys * H * D + h * D;
        device const float* krow = Kn + ((ulong)b * H * C + h * C + p) * D;
        device const float* vrow = Vn + ((ulong)b * H * C + h * C + p) * D;
#if INT8
        float amax_k = 0.0f;
        float amax_v = 0.0f;
        for (int d = tid; d < D; d += NT) {{
            amax_k = max(amax_k, fabs(krow[d]));
            amax_v = max(amax_v, fabs(vrow[d]));
        }}
        amax_k = simd_max(amax_k);
        amax_v = simd_max(amax_v);
        const float sk = amax_k / 127.0f + 1e-12f;
        const float sv = amax_v / 127.0f + 1e-12f;
        if (tid == 0) {{
            kscales[(ulong)phys * H + h] = sk;
            vscales[(ulong)phys * H + h] = sv;
        }}
        for (int d = tid; d < D; d += NT) {{
            K[dst + d] = (T_KV)(clamp(rint(krow[d] / sk), -127.0f, 127.0f) + 128.0f);
            V[dst + d] = (T_KV)(clamp(rint(vrow[d] / sv), -127.0f, 127.0f) + 128.0f);
        }}
#else
        for (int d = tid; d < D; d += NT) {{
            K[dst + d] = T_KV(krow[d]);
            V[dst + d] = T_KV(vrow[d]);
        }}
#endif
    }}
}}
"#,
            d = d,
            kv_ty = kv_ty,
            int8 = int8,
        )
    }

    // Fused batched scatter: k_new/v_new [B, H, C, D] f32 (C = 1 for
    // decode, the chunk for prefill); write rows per slot are
    // ctxlens[b] - advance .. ctxlens[b].
    #[allow(clippy::too_many_arguments)]
    pub fn scatter(
        k_new: &Tensor,
        v_new: &Tensor,
        k_slab: &Tensor,
        v_slab: &Tensor,
        k_scales: Option<&Tensor>,
        v_scales: Option<&Tensor>,
        tables: &Tensor,
        ctxlens: &Tensor,
        block_size: usize,
        advance: usize,
    ) -> candle_core::Result<()> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let (b, h, c, d) = k_new.dims4()?;
        let device = k_new.device();
        let mdev = device.as_metal_device()?;
        let slab_dtype = k_slab.dtype();
        let mut hasher = DefaultHasher::new();
        (0x5CA7u32, d, slab_dtype).hash(&mut hasher);
        let src = scatter_source(d, slab_dtype);
        let pipe = compile(mdev, hasher.finish(), &src, "et_paged_scatter")?;
        let k_new = k_new.contiguous()?;
        let v_new = v_new.contiguous()?;
        let encoder = mdev.command_encoder()?;
        let encoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipe);
        let (knb, kno) = buffer_of(&k_new)?;
        let (vnb, vno) = buffer_of(&v_new)?;
        let (kb, ko) = buffer_of(k_slab)?;
        let (vb, vo) = buffer_of(v_slab)?;
        let (tb, to) = buffer_of(tables)?;
        let (cb, co) = buffer_of(ctxlens)?;
        let f32_off = |off: usize| off * DType::F32.size_in_bytes();
        let u32_off = |off: usize| off * DType::U32.size_in_bytes();
        let elem_off = |off: usize| off * slab_dtype.size_in_bytes();
        encoder.set_input_buffer(0, Some(&knb), f32_off(kno));
        encoder.set_input_buffer(1, Some(&vnb), f32_off(vno));
        encoder.set_output_buffer(2, Some(&kb), elem_off(ko));
        encoder.set_output_buffer(3, Some(&vb), elem_off(vo));
        let (ksb, kso) = match k_scales {
            Some(t) => buffer_of(t)?,
            None => (kb.clone(), ko),
        };
        let (vsb, vso) = match v_scales {
            Some(t) => buffer_of(t)?,
            None => (vb.clone(), vo),
        };
        encoder.set_output_buffer(4, Some(&ksb), f32_off(kso));
        encoder.set_output_buffer(5, Some(&vsb), f32_off(vso));
        encoder.set_input_buffer(6, Some(&tb), u32_off(to));
        encoder.set_input_buffer(7, Some(&cb), u32_off(co));
        encoder.set_bytes(8, &(block_size as u32));
        encoder.set_bytes(9, &(tables.dim(1)? as u32));
        encoder.set_bytes(10, &(h as u32));
        encoder.set_bytes(11, &(advance as u32));
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: h,
                height: b,
                depth: c,
            },
            objc2_metal::MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    // One threadgroup per (slot, head): q is staged once into
    // threadgroup memory, K/V rows stream through the block table
    // with 128-bit vector loads (D % 4 == 0) — each thread keeps a
    // local online softmax (m, l, acc[D]) over strided rows; partials
    // fold within simd groups (32 lanes) via shuffles, then four group
    // partials combine in shared memory — no NT x D scratch, so D up
    // to 128 stays in the 32KB budget.
    fn kernel_source(d: usize, slab_dtype: DType, scale: f64) -> String {
        let (kv_ty, int8) = match slab_dtype {
            DType::F32 => ("float", 0),
            DType::F16 => ("half", 0),
            DType::BF16 => ("bfloat", 0),
            DType::U8 => ("uchar", 1),
            other => unreachable!("paged decode: unsupported slab dtype {other:?}"),
        };
        let vec4 = usize::from(d % 4 == 0);
        // One 128-bit slab-row load, dequantized to float4.
        let load4 = match slab_dtype {
            DType::F32 => "float4(*(device const packed_float4*)base) * s".to_string(),
            DType::F16 => "float4(*(device const packed_half4*)base) * s".to_string(),
            DType::BF16 => "float4(*(device const packed_bfloat4*)base) * s".to_string(),
            DType::U8 => "((float4(float(packed & 0xFF), float((packed >> 8) & 0xFF), float((packed >> 16) & 0xFF), float((packed >> 24) & 0xFF))) - 128.0f) * s".to_string(),
            other => unreachable!("paged decode: {other:?}"),
        };
        let load4_prelude = match slab_dtype {
            DType::U8 => "const uint packed = *(device const uint*)(base);",
            _ => "",
        };
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;

#define D {d}
#define NT {nt}
#define SCALE {scale:?}f
#define T_KV {kv_ty}
#define INT8 {int8}
#define VEC4 {vec4}

inline float4 kv_load4(device const T_KV* base, float s) {{
    {load4_prelude}
    return {load4};
}}

kernel void et_paged_decode(
    device const float* Q [[buffer(0)]],
    device const T_KV* K [[buffer(1)]],
    device const T_KV* V [[buffer(2)]],
    device const uint* tables [[buffer(3)]],
    device const uint* ctxlens [[buffer(4)]],
    device float* O [[buffer(5)]],
    device const float* kscales [[buffer(6)]],
    device const float* vscales [[buffer(7)]],
    constant uint& blockSize [[buffer(8)]],
    constant uint& maxBlocks [[buffer(9)]],
    constant uint& window [[buffer(10)]],
    constant uint& H [[buffer(11)]],
    constant uint& advance [[buffer(12)]],
    uint3 gridDim [[threadgroups_per_grid]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {{
    const uint b = tgid.y;
    const uint h = tgid.x;
    const uint p = tgid.z;
    const uint C = gridDim.z;
    const uint tid = tpitg.x;
    const uint needed = ctxlens[b];
    // Causal per q row: row p of the new chunk attends through
    // cursor + p (pads clamp to the real frontier; their outputs are
    // discarded downstream).
    const uint cursor = needed - advance;
    const uint ctx = min(cursor + p + 1, needed);
    const uint start = (window > 0 && ctx > window) ? ctx - window : 0;
    device const uint* table = tables + (ulong)b * maxBlocks;

    // Stage q once: the whole threadgroup reads threadgroup memory
    // from here on, never the device.
    threadgroup float qg[D];
    for (int d = tid; d < D; d += NT) {{ qg[d] = Q[((ulong)b * H * C + h * C + p) * D + d]; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY;
    float l = 0.0f;
    float acc[D];
    for (int d = 0; d < D; d++) {{ acc[d] = 0.0f; }}

    for (uint row = start + tid; row < ctx; row += NT) {{
        const uint phys = table[row / blockSize] * blockSize + (row % blockSize);
        const ulong krow = (ulong)phys * H * D + h * D;
        float score = 0.0f;
#if INT8
        const float ks = kscales[(ulong)phys * H + h];
#else
        const float ks = 1.0f;
#endif
#if VEC4
        for (int j = 0; j < D / 4; j++) {{
            const float4 kv = kv_load4(K + krow + j * 4, ks);
            const float4 q4 = float4(qg[j * 4], qg[j * 4 + 1], qg[j * 4 + 2], qg[j * 4 + 3]);
            score += dot(q4, kv);
        }}
#else
        for (int d = 0; d < D; d++) {{
#if INT8
            const float kv = (float(K[krow + d]) - 128.0f) * ks;
#else
            const float kv = float(K[krow + d]);
#endif
            score += qg[d] * kv;
        }}
#endif
        score *= SCALE;
        const float mn = max(m, score);
        const float c = (mn == -INFINITY) ? 0.0f : exp(m - mn);
        const float p = (mn == -INFINITY) ? 0.0f : exp(score - mn);
#if INT8
        const float vs = vscales[(ulong)phys * H + h];
#else
        const float vs = 1.0f;
#endif
#if VEC4
        for (int j = 0; j < D / 4; j++) {{
            const float4 vv = kv_load4(V + krow + j * 4, vs);
            const float4 prev = float4(acc[j * 4], acc[j * 4 + 1], acc[j * 4 + 2], acc[j * 4 + 3]);
            const float4 next = prev * c + p * vv;
            acc[j * 4] = next.x;
            acc[j * 4 + 1] = next.y;
            acc[j * 4 + 2] = next.z;
            acc[j * 4 + 3] = next.w;
        }}
#else
        for (int d = 0; d < D; d++) {{
#if INT8
            const float vv = (float(V[krow + d]) - 128.0f) * vs;
#else
            const float vv = float(V[krow + d]);
#endif
            acc[d] = acc[d] * c + p * vv;
        }}
#endif
        l = l * c + p;
        m = mn;
    }}

    // Fold within each simd group: group max, rescale, group sums.
    const float gm = simd_max(m);
    const float rc = (gm == -INFINITY) ? 0.0f : exp(m - gm);
    l *= rc;
    l = simd_sum(l);
    for (int d = 0; d < D; d++) {{ acc[d] = simd_sum(acc[d] * rc); }}
    // Four group partials (NT = 128 lanes) combine in shared memory.
    threadgroup float pm[NT / 32];
    threadgroup float pl[NT / 32];
    threadgroup float pacc[NT / 32][D];
    const uint lane = tid % 32;
    const uint grp = tid / 32;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {{
        pm[grp] = gm;
        pl[grp] = l;
        for (int d = 0; d < D; d++) {{ pacc[grp][d] = acc[d]; }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {{
        float fm = -INFINITY;
        for (uint g = 0; g < NT / 32; g++) {{ fm = max(fm, pm[g]); }}
        float fl = 0.0f;
        float facc[D];
        for (int d = 0; d < D; d++) {{ facc[d] = 0.0f; }}
        for (uint g = 0; g < NT / 32; g++) {{
            const float c = (fm == -INFINITY) ? 0.0f : exp(pm[g] - fm);
            fl += pl[g] * c;
            for (int d = 0; d < D; d++) {{ facc[d] += pacc[g][d] * c; }}
        }}
        device float* o = O + ((ulong)b * H * C + h * C + p) * D;
        const float inv = (fl == 0.0f) ? 0.0f : 1.0f / fl;
        for (int d = 0; d < D; d++) {{ o[d] = facc[d] * inv; }}
    }}
}}
"#,
            d = d,
            nt = THREADS,
            scale = scale as f32,
            kv_ty = kv_ty,
            int8 = int8,
            vec4 = vec4,
            load4_prelude = load4_prelude,
            load4 = load4,
        )
    }

    fn pipeline(
        mdev: &candle_core::MetalDevice,
        d: usize,
        slab_dtype: DType,
        scale: f64,
    ) -> candle_core::Result<ComputePipeline> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        (d, slab_dtype, scale.to_bits()).hash(&mut hasher);
        let src = kernel_source(d, slab_dtype, scale);
        compile(mdev, hasher.finish(), &src, "et_paged_decode")
    }

    // One launch for the whole batch: q [B, H, C, D] contiguous
    // (C = 1 for decode, the chunk for prefill), tables
    // [B, maxBlocks] u32, ctxlens [B] u32 (post-run frontier; the
    // kernel derives per-row causal lengths from `advance`). Returns
    // [B, H, C, D] f32.
    #[allow(clippy::too_many_arguments)]
    pub fn decode(
        q: &Tensor,
        k_slab: &Tensor,
        v_slab: &Tensor,
        k_scales: Option<&Tensor>,
        v_scales: Option<&Tensor>,
        tables: &Tensor,
        ctxlens: &Tensor,
        window: Option<usize>,
        scale: f64,
        block_size: usize,
        advance: usize,
    ) -> candle_core::Result<Tensor> {
        let (b, h, c, d) = q.dims4()?;
        let device = q.device();
        let mdev = device.as_metal_device()?;
        let slab_dtype = k_slab.dtype();
        let q = q.contiguous()?;
        let pipe = pipeline(mdev, d, slab_dtype, scale)?;
        let out_buf = mdev.new_buffer(b * h * c * d, DType::F32, "paged_o")?;
        let max_blocks = tables.dim(1)?;
        let encoder = mdev.command_encoder()?;
        let encoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipe);
        let (qb, qo) = buffer_of(&q)?;
        let (kb, ko) = buffer_of(k_slab)?;
        let (vb, vo) = buffer_of(v_slab)?;
        let (tb, to) = buffer_of(tables)?;
        let (cb, co) = buffer_of(ctxlens)?;
        let f32_off = |off: usize| off * DType::F32.size_in_bytes();
        let u32_off = |off: usize| off * DType::U32.size_in_bytes();

        let elem_off = |off: usize| off * slab_dtype.size_in_bytes();
        encoder.set_input_buffer(0, Some(&qb), f32_off(qo));
        encoder.set_input_buffer(1, Some(&kb), elem_off(ko));
        encoder.set_input_buffer(2, Some(&vb), elem_off(vo));
        encoder.set_input_buffer(3, Some(&tb), u32_off(to));
        encoder.set_input_buffer(4, Some(&cb), u32_off(co));
        encoder.set_output_buffer(5, Some(&out_buf), 0);
        // Scale buffers are always bound; point at K when unused.
        let (ksb, kso) = match k_scales {
            Some(t) => buffer_of(t)?,
            None => (kb.clone(), ko),
        };
        let (vsb, vso) = match v_scales {
            Some(t) => buffer_of(t)?,
            None => (vb.clone(), vo),
        };

        encoder.set_input_buffer(6, Some(&ksb), f32_off(kso));
        encoder.set_input_buffer(7, Some(&vsb), f32_off(vso));
        encoder.set_bytes(8, &(block_size as u32));
        encoder.set_bytes(9, &(max_blocks as u32));
        encoder.set_bytes(10, &(window.unwrap_or(0) as u32));
        encoder.set_bytes(11, &(h as u32));
        encoder.set_bytes(12, &(advance as u32));
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: h,
                height: b,
                depth: c,
            },
            objc2_metal::MTLSize {
                width: THREADS,
                height: 1,
                depth: 1,
            },
        );
        // no end_encoding: candle's Commands owns the encoder lifecycle
        Ok(Tensor::from_storage(
            Storage::Metal(MetalStorage::new(out_buf, mdev.clone(), b * h * c * d, DType::F32)),
            (b, h, c, d),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }
}

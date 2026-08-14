//! Paged decode attention on Metal (RFC 0013, stage 2): one kernel
//! launch attends q [B, H, 1, D] over pool slabs IN PLACE — K/V rows
//! are read through the block table (pages), never gathered into a
//! contiguous copy. One threadgroup per (sequence slot, head) streams
//! its slot's blocks with online-softmax accumulation, so the context
//! length is a runtime value (unlike the training flash pipeline,
//! nothing shape-dependent is baked). Slab dtypes f16/bf16 load
//! natively; int8 slabs dequantize in registers with the per-(token,
//! head) scale slab (RFC 0012). The primitive scatter+gather reference in
//! lib.rs remains the reference and the CPU fallback.
//!
//! ## Cache invariants
//!
//! - The pool is a pair of slabs `[pool_rows, H_kv, D]` (or
//!   `[pool_rows, H_kv * D]`) in `slab_dtype`; row addresses are
//!   computed device-side from the per-slot block table
//!   (`tables [B, maxBlocks] u32`), `ctxlens [B] u32` (post-run
//!   frontier), `block_bases [B] u32` (first visible table index), and
//!   `block_size`.
//! - **Scatter** (`et_paged_scatter`): one threadgroup of one simdgroup
//!   (32 threads) per (slot, head) writes rows `ctxlens[b] - advance
//!   .. ctxlens[b]` of the new-token chunk into the slabs. Int8 slabs
//!   quantize with an in-threadgroup absmax scale (`absmax/127 + eps`,
//!   round, +128 offset) stored per (physical row, head).
//! - **Attention** (`et_paged_decode`): one threadgroup of 128 threads
//!   per (slot, head, chunk row); q is staged in threadgroup memory,
//!   K/V rows stream through the table with 128-bit vector loads
//!   (requires `D % 4 == 0` for the vector path; scalar fallback
//!   otherwise), and each thread keeps a local online softmax
//!   `(m, l, acc[D])` folded across simdgroups via shuffles and a
//!   shared-memory combine. Causality is per chunk row: row `p`
//!   attends through `cursor + p` (pads clamp to the real frontier).
//! - Grouped-query attention maps query head `h` to K/V head
//!   `h / (H_q / H_kv)`; `H_q` must be a multiple of `H_kv`.
//! - A sliding `window` (0 = disabled) restricts the attended range to
//!   `[ctx - window, ctx)`.
//!
//! The `*_into` entry points allocate nothing, require empty resource
//! views, and need their exact pipelines pre-warmed.

use crate::runtime::dtype::DType;

/// Planner-facing requirements of the paged scatter launch (writes
/// new-token rows into the slabs in place; no outputs of its own).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScatterRequirements {
    /// Slab storage dtype (f32/f16/bf16, or u8 for int8-quantized).
    pub slab_dtype: DType,
    /// Head dimension `D`.
    pub head_dim: usize,
    /// Always 0: the scatter writes the slabs in place.
    pub output_bytes: usize,
    /// Always 0: slab writes ARE the state transaction.
    pub state_next_bytes: usize,
    /// Always 0: no staging views.
    pub staging_bytes: usize,
    /// Always 0: no runtime status.
    pub status_bytes: usize,
    /// Always 0: no scratch workspace.
    pub scratch_bytes: usize,
    /// Always 1: a single scatter kernel per launch.
    pub pipeline_count: usize,
}

/// Planner-facing requirements of a paged attention (decode) launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionRequirements {
    /// Slab storage dtype (f32/f16/bf16, or u8 for int8-quantized).
    pub slab_dtype: DType,
    /// Head dimension `D` (≤ 128 for the kernel's register budget).
    pub head_dim: usize,
    /// Query heads `H_q`.
    pub query_heads: usize,
    /// K/V heads `H_kv` (must divide `H_q`).
    pub kv_heads: usize,
    /// Softmax scale, stored as `f64::to_bits` so requirements stay
    /// `Eq` and hashable for pipeline keys.
    pub scale_bits: u64,
    /// Bytes of the f32 `[B, H_q, C, D]` output.
    pub output_bytes: usize,
    /// Always 0: attention does not mutate the cache.
    pub state_next_bytes: usize,
    /// Always 0: no staging views.
    pub staging_bytes: usize,
    /// Always 0: no runtime status.
    pub status_bytes: usize,
    /// Always 0: no scratch workspace (accumulators live in registers
    /// and threadgroup memory).
    pub scratch_bytes: usize,
    /// Always 1: a single decode kernel per launch.
    pub pipeline_count: usize,
}

/// Requirements of a scatter of `batch × heads × chunk × head_dim`
/// new-token rows into slabs of `slab_dtype`.
pub fn scatter_requirements(
    slab_dtype: DType,
    batch: usize,
    heads: usize,
    chunk: usize,
    head_dim: usize,
) -> crate::err::Res<ScatterRequirements> {
    if !matches!(
        slab_dtype,
        DType::F32 | DType::F16 | DType::BF16 | DType::U8
    ) {
        return Err(format!(
            "paged scatter: unsupported slab dtype {slab_dtype:?}"
        ));
    }
    batch
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(chunk))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| "paged scatter: requirement element count overflow".to_string())?;
    Ok(ScatterRequirements {
        slab_dtype,
        head_dim,
        output_bytes: 0,
        state_next_bytes: 0,
        staging_bytes: 0,
        status_bytes: 0,
        scratch_bytes: 0,
        pipeline_count: 1,
    })
}

/// Requirements of a paged attention launch over `batch` slots,
/// `chunk` query rows each, with the given head geometry and softmax
/// `scale`. Validates GQA divisibility and computes the f32 output
/// size with overflow checks.
pub fn attention_requirements(
    slab_dtype: DType,
    batch: usize,
    query_heads: usize,
    kv_heads: usize,
    chunk: usize,
    head_dim: usize,
    scale: f64,
) -> crate::err::Res<AttentionRequirements> {
    if !matches!(
        slab_dtype,
        DType::F32 | DType::F16 | DType::BF16 | DType::U8
    ) {
        return Err(format!(
            "paged attention: unsupported slab dtype {slab_dtype:?}"
        ));
    }
    if kv_heads == 0 || !query_heads.is_multiple_of(kv_heads) {
        return Err("paged attention: query heads must be divisible by K/V heads".to_string());
    }
    let output_bytes = batch
        .checked_mul(query_heads)
        .and_then(|value| value.checked_mul(chunk))
        .and_then(|value| value.checked_mul(head_dim))
        .and_then(|value| value.checked_mul(DType::F32.size_in_bytes()))
        .ok_or_else(|| "paged attention: requirement byte size overflow".to_string())?;
    Ok(AttentionRequirements {
        slab_dtype,
        head_dim,
        query_heads,
        kv_heads,
        scale_bits: scale.to_bits(),
        output_bytes,
        state_next_bytes: 0,
        staging_bytes: 0,
        status_bytes: 0,
        scratch_bytes: 0,
        pipeline_count: 1,
    })
}

/// Alias of [`attention_requirements`] for the decode-step call site
/// (chunk = 1 in decode; the same kernel serves chunked prefill).
pub fn decode_requirements(
    slab_dtype: DType,
    batch: usize,
    query_heads: usize,
    kv_heads: usize,
    chunk: usize,
    head_dim: usize,
    scale: f64,
) -> crate::err::Res<AttentionRequirements> {
    attention_requirements(
        slab_dtype,
        batch,
        query_heads,
        kv_heads,
        chunk,
        head_dim,
        scale,
    )
}

#[cfg(test)]
mod requirement_tests {
    use super::*;

    #[test]
    fn requirements_are_exact_and_pipeline_specific() {
        let scatter = scatter_requirements(DType::U8, 2, 4, 3, 16).unwrap();
        assert_eq!(
            (
                scatter.output_bytes,
                scatter.state_next_bytes,
                scatter.staging_bytes,
                scatter.status_bytes,
                scatter.scratch_bytes,
            ),
            (0, 0, 0, 0, 0)
        );
        assert_eq!(scatter.pipeline_count, 1);

        let attention = decode_requirements(DType::F16, 2, 4, 4, 3, 16, 0.25).unwrap();
        assert_eq!(attention.output_bytes, 2 * 4 * 3 * 16 * 4);
        assert_eq!(attention.pipeline_count, 1);
        assert!(scatter_requirements(DType::I64, 1, 1, 1, 1).is_err());
        assert!(attention_requirements(DType::F32, usize::MAX, 2, 2, 1, 1, 1.0).is_err());
    }
}

/// Whether the paged kernel can run this decode step: Metal, f32
/// compute, head dim within the kernel's register budget, slabs in a
/// supported storage dtype.
pub fn is_supported(
    q: &crate::runtime::metal::run::MetalTensor,
    slab_dtype: DType,
    head_dim: usize,
) -> bool {
    q.dtype == crate::runtime::dtype::DType::F32
        && head_dim <= 128
        && matches!(
            slab_dtype,
            DType::F32 | DType::F16 | DType::BF16 | DType::U8
        )
}

#[cfg(target_os = "macos")]
pub use metal::{
    attention_into, decode, decode_into, scatter, scatter_into, warm_all, warm_attention,
    warm_attention_exact, warm_scatter, warm_scatter_exact, IntoResources,
};

#[cfg(target_os = "macos")]
mod metal {
    use crate::runtime::dtype::DType;
    use crate::runtime::metal::device::{set_buffer, set_bytes, MetalDevice, Pipeline};
    use crate::runtime::metal::run::MetalTensor;

    use objc2_metal::MTLComputeCommandEncoder;

    /// Borrowed resource views supplied by the executable planner. Both
    /// paged kernels operate entirely on their explicit arguments, so
    /// every slice must be empty; see [`IntoResources::empty`].
    #[derive(Clone, Copy)]
    pub struct IntoResources<'a> {
        /// Must be empty.
        pub staging: &'a [MetalTensor],
        /// Must be empty.
        pub status: &'a [MetalTensor],
        /// Must be empty.
        pub scratch: &'a [MetalTensor],
    }

    impl IntoResources<'_> {
        /// The (only) valid resource set for paged dispatches.
        pub const fn empty() -> Self {
            Self {
                staging: &[],
                status: &[],
                scratch: &[],
            }
        }
    }

    const THREADS: usize = 128;

    fn wrap_contig(t: &MetalTensor) -> crate::err::Res<MetalTensor> {
        if t.layout.is_contiguous() {
            Ok(t.clone())
        } else {
            crate::runtime::metal::kernels::strided_copy(MetalDevice::get(), t)
        }
    }

    fn slab_dtype(t: &MetalTensor) -> DType {
        t.dtype
    }

    fn slab_heads(tensor: &MetalTensor, head_dim: usize) -> crate::err::Res<usize> {
        let shape = tensor.layout.shape();
        match shape {
            [_, heads, dimension] if *dimension == head_dim => Ok(*heads),
            [_, row_width] if head_dim != 0 && row_width.is_multiple_of(head_dim) => {
                Ok(row_width / head_dim)
            }
            _ => {
                Err("paged attention: K/V slab row width must be divisible by head dim".to_string())
            }
        }
    }

    fn validate_empty_resources(
        resources: IntoResources<'_>,
        operation: &str,
    ) -> crate::err::Res<()> {
        if !resources.staging.is_empty()
            || !resources.status.is_empty()
            || !resources.scratch.is_empty()
        {
            return Err(format!(
                "{operation}: staging, status, and scratch views must be empty"
            ));
        }
        Ok(())
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
    device const uint* blockBases [[buffer(12)]],
    uint3 gridDim [[threadgroups_per_grid]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {{
    const uint b = tgid.y;
    const uint h = tgid.x;
    const uint tid = tpitg.x;
    const uint C = gridDim.z;
    if (ctxlens[b] == 0) {{ return; }}
    const uint cursor = ctxlens[b] - advance;
    // Rows cursor..needed, one per new token; D-wide within each row.
    for (uint p = 0; p < advance; p++) {{
        const uint pos = cursor + p;
        const uint tableIndex = pos / blockSize - blockBases[b];
        const uint phys = tables[(ulong)b * maxBlocks + tableIndex] * blockSize + (pos % blockSize);
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

    fn scatter_key(d: usize, slab_dtype: DType) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        (0x5CA7u32, d, slab_dtype).hash(&mut hasher);
        hasher.finish()
    }

    /// Warms the scatter and attention pipelines for every supported
    /// slab dtype; returns the number of pipelines compiled or fetched.
    #[allow(clippy::too_many_arguments)]
    pub fn warm_all(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        scale: f64,
    ) -> crate::err::Res<usize> {
        let mut count = 0;
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::U8] {
            MetalDevice::get().compile_lazy(scatter_key(d, dtype), "et_paged_scatter", || {
                scatter_source(d, dtype)
            })?;
            pipeline(d, query_heads, kv_heads, dtype, scale)?;
            count += 2;
        }
        Ok(count)
    }

    /// Warms the scatter pipeline for (`head_dim`, `slab_dtype`).
    pub fn warm_scatter(d: usize, slab_dtype: DType) -> crate::err::Res<()> {
        MetalDevice::get().compile_lazy(scatter_key(d, slab_dtype), "et_paged_scatter", || {
            scatter_source(d, slab_dtype)
        })?;
        Ok(())
    }

    /// Warms exactly the scatter pipeline described by `requirements`.
    pub fn warm_scatter_exact(requirements: &super::ScatterRequirements) -> crate::err::Res<()> {
        warm_scatter(requirements.head_dim, requirements.slab_dtype)
    }

    /// Warms the attention pipeline for the given head geometry, slab
    /// dtype, and softmax scale.
    pub fn warm_attention(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        slab_dtype: DType,
        scale: f64,
    ) -> crate::err::Res<()> {
        pipeline(d, query_heads, kv_heads, slab_dtype, scale)?;
        Ok(())
    }

    /// Warms exactly the attention pipeline described by `requirements`.
    pub fn warm_attention_exact(
        requirements: &super::AttentionRequirements,
    ) -> crate::err::Res<()> {
        warm_attention(
            requirements.head_dim,
            requirements.query_heads,
            requirements.kv_heads,
            requirements.slab_dtype,
            f64::from_bits(requirements.scale_bits),
        )
    }

    /// Non-allocating fused batched scatter: `k_new`/`v_new [B, H, C,
    /// D]` f32 (C = 1 for decode, the chunk for prefill) are written
    /// into the slabs at rows `ctxlens[b] - advance .. ctxlens[b]` per
    /// slot. `k_scales`/`v_scales` are required iff the slabs are int8.
    /// Allocates nothing; requires the exact scatter pipeline to be
    /// warm.
    #[allow(clippy::too_many_arguments)]
    pub fn scatter_into(
        k_new: &MetalTensor,
        v_new: &MetalTensor,
        k_slab: &MetalTensor,
        v_slab: &MetalTensor,
        k_scales: Option<&MetalTensor>,
        v_scales: Option<&MetalTensor>,
        tables: &MetalTensor,
        ctxlens: &MetalTensor,
        block_bases: &MetalTensor,
        block_size: usize,
        advance: usize,
        resources: IntoResources<'_>,
    ) -> crate::err::Res<()> {
        let (b, h, c, d) = (
            k_new.layout.shape()[0],
            k_new.layout.shape()[1],
            k_new.layout.shape()[2],
            k_new.layout.shape()[3],
        );
        let slab_dtype = slab_dtype(k_slab);
        validate_empty_resources(resources, "paged scatter")?;
        if !k_new.layout.is_contiguous() || !v_new.layout.is_contiguous() {
            return Err(
                "paged scatter: new K/V inputs must be contiguous before scatter_into".to_string(),
            );
        }
        let pipe = MetalDevice::get()
            .pipeline_cached(scatter_key(d, slab_dtype))
            .ok_or_else(|| {
                "paged scatter: exact pipeline is not warm; call warm_scatter".to_string()
            })?;
        let f32_off = |off: usize| off * 4;
        let u32_off = |off: usize| off * 4;
        let elem_off = |off: usize| off * slab_dtype.size_in_bytes();
        let max_blocks = tables.layout.shape()[1] as u32;
        MetalDevice::get().with_encoder(|e| {
            e.setComputePipelineState(pipe.as_raw());
            set_buffer(e, 0, &k_new.buffer, f32_off(k_new.layout.offset()));
            set_buffer(e, 1, &v_new.buffer, f32_off(v_new.layout.offset()));
            set_buffer(e, 2, &k_slab.buffer, elem_off(k_slab.layout.offset()));
            set_buffer(e, 3, &v_slab.buffer, elem_off(v_slab.layout.offset()));
            let (ksb, kso) = match k_scales {
                Some(t) => (&t.buffer, t.layout.offset()),
                None => (&k_slab.buffer, k_slab.layout.offset()),
            };
            let (vsb, vso) = match v_scales {
                Some(t) => (&t.buffer, t.layout.offset()),
                None => (&v_slab.buffer, v_slab.layout.offset()),
            };
            set_buffer(e, 4, ksb, f32_off(kso));
            set_buffer(e, 5, vsb, f32_off(vso));
            set_buffer(e, 6, &tables.buffer, u32_off(tables.layout.offset()));
            set_buffer(e, 7, &ctxlens.buffer, u32_off(ctxlens.layout.offset()));
            set_bytes(e, 8, &(block_size as u32));
            set_bytes(e, 9, &max_blocks);
            set_bytes(e, 10, &(h as u32));
            set_bytes(e, 11, &(advance as u32));
            set_buffer(
                e,
                12,
                &block_bases.buffer,
                u32_off(block_bases.layout.offset()),
            );
            e.dispatchThreadgroups_threadsPerThreadgroup(
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
        });
        Ok(())
    }

    /// Allocating convenience wrapper around [`scatter_into`] (makes
    /// inputs contiguous and warms the pipeline first).
    #[allow(clippy::too_many_arguments)]
    pub fn scatter(
        k_new: &MetalTensor,
        v_new: &MetalTensor,
        k_slab: &MetalTensor,
        v_slab: &MetalTensor,
        k_scales: Option<&MetalTensor>,
        v_scales: Option<&MetalTensor>,
        tables: &MetalTensor,
        ctxlens: &MetalTensor,
        block_bases: &MetalTensor,
        block_size: usize,
        advance: usize,
    ) -> crate::err::Res<()> {
        let k_new = wrap_contig(k_new)?;
        let v_new = wrap_contig(v_new)?;
        warm_scatter(k_new.layout.shape()[3], k_slab.dtype)?;
        scatter_into(
            &k_new,
            &v_new,
            k_slab,
            v_slab,
            k_scales,
            v_scales,
            tables,
            ctxlens,
            block_bases,
            block_size,
            advance,
            IntoResources::empty(),
        )
    }

    // One threadgroup per (slot, head): q is staged once into
    // threadgroup memory, K/V rows stream through the block table
    // with 128-bit vector loads (D % 4 == 0) — each thread keeps a
    // local online softmax (m, l, acc[D]) over strided rows; partials
    // fold within simd groups (32 lanes) via shuffles, then four group
    // partials combine in shared memory — no NT x D scratch, so D up
    // to 128 stays in the 32KB budget.
    fn kernel_source(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        slab_dtype: DType,
        scale: f64,
    ) -> String {
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
#define QH {query_heads}
#define KVH {kv_heads}
#define GROUP {head_group_size}

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
    device const uint* blockBases [[buffer(13)]],
    uint3 gridDim [[threadgroups_per_grid]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {{
    const uint b = tgid.y;
    const uint h = tgid.x;
    const uint kvh = h / GROUP;
    const uint p = tgid.z;
    const uint C = gridDim.z;
    const uint tid = tpitg.x;
    const uint needed = ctxlens[b];
    if (needed == 0) {{
        if (tid == 0) {{
            device float* o = O + ((ulong)b * QH * C + h * C + p) * D;
            for (int d = 0; d < D; d++) {{ o[d] = 0.0f; }}
        }}
        return;
    }}
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
    for (int d = tid; d < D; d += NT) {{ qg[d] = Q[((ulong)b * QH * C + h * C + p) * D + d]; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY;
    float l = 0.0f;
    float acc[D];
    for (int d = 0; d < D; d++) {{ acc[d] = 0.0f; }}

    for (uint row = start + tid; row < ctx; row += NT) {{
        const uint phys = table[row / blockSize - blockBases[b]] * blockSize + (row % blockSize);
        const ulong krow = (ulong)phys * KVH * D + kvh * D;
        float score = 0.0f;
#if INT8
        const float ks = kscales[(ulong)phys * KVH + kvh];
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
        const float vs = vscales[(ulong)phys * KVH + kvh];
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
        device float* o = O + ((ulong)b * QH * C + h * C + p) * D;
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
            query_heads = query_heads,
            kv_heads = kv_heads,
            head_group_size = query_heads / kv_heads,
            load4_prelude = load4_prelude,
            load4 = load4,
        )
    }

    fn attention_key(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        slab_dtype: DType,
        scale: f64,
    ) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        (d, query_heads, kv_heads, slab_dtype, scale.to_bits()).hash(&mut hasher);
        hasher.finish()
    }

    fn pipeline(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        slab_dtype: DType,
        scale: f64,
    ) -> crate::err::Res<Pipeline> {
        if kv_heads == 0 || !query_heads.is_multiple_of(kv_heads) {
            return Err("paged attention: query heads must be divisible by K/V heads".to_string());
        }
        MetalDevice::get().compile_lazy(
            attention_key(d, query_heads, kv_heads, slab_dtype, scale),
            "et_paged_decode",
            || kernel_source(d, query_heads, kv_heads, slab_dtype, scale),
        )
    }

    /// Non-allocating paged attention: one launch for the whole batch.
    /// `q [B, H, C, D]` f32 contiguous (C = 1 for decode, the chunk for
    /// prefill), `tables [B, maxBlocks]` u32, `ctxlens [B]` u32
    /// (post-run frontier; the kernel derives per-row causal lengths
    /// from `advance`), `block_bases [B]` u32. `output` must be
    /// contiguous f32 `[B, H, C, D]`. Allocates nothing; requires the
    /// exact attention pipeline to be warm.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_into(
        q: &MetalTensor,
        k_slab: &MetalTensor,
        v_slab: &MetalTensor,
        k_scales: Option<&MetalTensor>,
        v_scales: Option<&MetalTensor>,
        tables: &MetalTensor,
        ctxlens: &MetalTensor,
        block_bases: &MetalTensor,
        window: Option<usize>,
        scale: f64,
        block_size: usize,
        advance: usize,
        output: &MetalTensor,
        resources: IntoResources<'_>,
    ) -> crate::err::Res<()> {
        let (b, h, c, d) = (
            q.layout.shape()[0],
            q.layout.shape()[1],
            q.layout.shape()[2],
            q.layout.shape()[3],
        );
        let kv_heads = slab_heads(k_slab, d)?;
        if kv_heads == 0 || !h.is_multiple_of(kv_heads) {
            return Err("paged attention: query heads must be divisible by K/V heads".to_string());
        }
        let slab_dtype = slab_dtype(k_slab);
        validate_empty_resources(resources, "paged attention")?;
        if !q.layout.is_contiguous() {
            return Err(
                "paged attention: query must be contiguous before attention_into".to_string(),
            );
        }
        if output.dtype != DType::F32
            || output.layout.shape() != [b, h, c, d]
            || !output.layout.is_contiguous()
        {
            return Err(format!(
                "paged attention: output must be contiguous [{b}, {h}, {c}, {d}]:F32"
            ));
        }
        MetalDevice::get().mark_buffer_write(&output.buffer)?;
        let pipe = MetalDevice::get()
            .pipeline_cached(attention_key(d, h, kv_heads, slab_dtype, scale))
            .ok_or_else(|| {
                "paged attention: exact pipeline is not warm; call warm_attention".to_string()
            })?;
        let max_blocks = tables.layout.shape()[1];
        let f32_off = |off: usize| off * DType::F32.size_in_bytes();
        let u32_off = |off: usize| off * DType::U32.size_in_bytes();
        let elem_off = |off: usize| off * slab_dtype.size_in_bytes();
        MetalDevice::get().with_encoder(|e| {
            e.setComputePipelineState(pipe.as_raw());
            set_buffer(e, 0, &q.buffer, f32_off(q.layout.offset()));
            set_buffer(e, 1, &k_slab.buffer, elem_off(k_slab.layout.offset()));
            set_buffer(e, 2, &v_slab.buffer, elem_off(v_slab.layout.offset()));
            set_buffer(e, 3, &tables.buffer, u32_off(tables.layout.offset()));
            set_buffer(e, 4, &ctxlens.buffer, u32_off(ctxlens.layout.offset()));
            set_buffer(e, 5, &output.buffer, f32_off(output.layout.offset()));
            let (ksb, kso) = match k_scales {
                Some(t) => (&t.buffer, t.layout.offset()),
                None => (&k_slab.buffer, k_slab.layout.offset()),
            };
            let (vsb, vso) = match v_scales {
                Some(t) => (&t.buffer, t.layout.offset()),
                None => (&v_slab.buffer, v_slab.layout.offset()),
            };
            set_buffer(e, 6, ksb, f32_off(kso));
            set_buffer(e, 7, vsb, f32_off(vso));
            set_bytes(e, 8, &(block_size as u32));
            set_bytes(e, 9, &(max_blocks as u32));
            set_bytes(e, 10, &(window.unwrap_or(0) as u32));
            set_bytes(e, 11, &(h as u32));
            set_bytes(e, 12, &(advance as u32));
            set_buffer(
                e,
                13,
                &block_bases.buffer,
                u32_off(block_bases.layout.offset()),
            );
            e.dispatchThreadgroups_threadsPerThreadgroup(
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
        });
        Ok(())
    }

    /// Decode-step alias of [`attention_into`]; identical kernel and
    /// contract, named for the decode call site.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_into(
        q: &MetalTensor,
        k_slab: &MetalTensor,
        v_slab: &MetalTensor,
        k_scales: Option<&MetalTensor>,
        v_scales: Option<&MetalTensor>,
        tables: &MetalTensor,
        ctxlens: &MetalTensor,
        block_bases: &MetalTensor,
        window: Option<usize>,
        scale: f64,
        block_size: usize,
        advance: usize,
        output: &MetalTensor,
        resources: IntoResources<'_>,
    ) -> crate::err::Res<()> {
        attention_into(
            q,
            k_slab,
            v_slab,
            k_scales,
            v_scales,
            tables,
            ctxlens,
            block_bases,
            window,
            scale,
            block_size,
            advance,
            output,
            resources,
        )
    }

    /// Allocating convenience wrapper around [`decode_into`]; returns
    /// the f32 `[B, H, C, D]` attention output.
    #[allow(clippy::too_many_arguments)]
    pub fn decode(
        q: &MetalTensor,
        k_slab: &MetalTensor,
        v_slab: &MetalTensor,
        k_scales: Option<&MetalTensor>,
        v_scales: Option<&MetalTensor>,
        tables: &MetalTensor,
        ctxlens: &MetalTensor,
        block_bases: &MetalTensor,
        window: Option<usize>,
        scale: f64,
        block_size: usize,
        advance: usize,
    ) -> crate::err::Res<MetalTensor> {
        let shape = q.layout.shape().to_vec();
        let q = wrap_contig(q)?;
        let d = q.layout.shape()[3];
        let kv_heads = slab_heads(k_slab, d)?;
        warm_attention(d, q.layout.shape()[1], kv_heads, k_slab.dtype, scale)?;
        let output = MetalTensor::empty(MetalDevice::get(), shape, DType::F32);
        decode_into(
            &q,
            k_slab,
            v_slab,
            k_scales,
            v_scales,
            tables,
            ctxlens,
            block_bases,
            window,
            scale,
            block_size,
            advance,
            &output,
            IntoResources::empty(),
        )?;
        Ok(output)
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use crate::runtime::dtype::DType;
    use crate::runtime::layout::Layout;
    use crate::runtime::metal::device::MetalDevice;
    use crate::runtime::metal::run::MetalTensor as MT;

    fn u32_tensor(dev: &MetalDevice, values: &[u32], shape: Vec<usize>) -> MT {
        MT {
            buffer: dev.alloc_with_data_u32(values),
            layout: Layout::contiguous(shape),
            dtype: DType::U32,
        }
    }

    fn max_diff(a: &MT, b: &MT) -> f32 {
        a.read_f32()
            .unwrap()
            .iter()
            .zip(b.read_f32().unwrap())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn scatter_attention_into_match_wrappers_and_use_only_supplied_views() {
        let dev = MetalDevice::get();
        let (b, h, c, d) = (1usize, 2usize, 1usize, 8usize);
        let q = MT::from_f32(
            dev,
            (0..b * h * c * d)
                .map(|index| index as f32 / 13.0)
                .collect(),
            vec![b, h, c, d],
        );
        let k_new = MT::from_f32(
            dev,
            (0..b * h * c * d)
                .map(|index| (index as f32 + 1.0) / 11.0)
                .collect(),
            vec![b, h, c, d],
        );
        let v_new = MT::from_f32(
            dev,
            (0..b * h * c * d)
                .map(|index| (index as f32 - 3.0) / 7.0)
                .collect(),
            vec![b, h, c, d],
        );
        let expected_k = MT::from_f32(dev, vec![0.0; 2 * h * d], vec![2, h, d]);
        let expected_v = MT::from_f32(dev, vec![0.0; 2 * h * d], vec![2, h, d]);
        let actual_k = MT::from_f32(dev, vec![0.0; 2 * h * d], vec![2, h, d]);
        let actual_v = MT::from_f32(dev, vec![0.0; 2 * h * d], vec![2, h, d]);
        let tables = u32_tensor(dev, &[0], vec![b, 1]);
        let ctxlens = u32_tensor(dev, &[1], vec![b]);
        let block_bases = u32_tensor(dev, &[0], vec![b]);
        let scale = 1.0 / (d as f64).sqrt();

        super::metal::scatter(
            &k_new,
            &v_new,
            &expected_k,
            &expected_v,
            None,
            None,
            &tables,
            &ctxlens,
            &block_bases,
            2,
            1,
        )
        .unwrap();
        let expected_output = super::metal::decode(
            &q,
            &expected_k,
            &expected_v,
            None,
            None,
            &tables,
            &ctxlens,
            &block_bases,
            None,
            scale,
            2,
            1,
        )
        .unwrap();
        let actual_output = MT::empty(dev, vec![b, h, c, d], DType::F32);
        let scatter_requirements = super::scatter_requirements(DType::F32, b, h, c, d).unwrap();
        let attention_requirements =
            super::attention_requirements(DType::F32, b, h, h, c, d, scale).unwrap();
        super::metal::warm_scatter_exact(&scatter_requirements).unwrap();
        super::metal::warm_attention_exact(&attention_requirements).unwrap();
        dev.synchronize().unwrap();

        let _dispatch_guard = dev.begin_executable_dispatch().unwrap();
        super::metal::scatter_into(
            &k_new,
            &v_new,
            &actual_k,
            &actual_v,
            None,
            None,
            &tables,
            &ctxlens,
            &block_bases,
            2,
            1,
            super::metal::IntoResources::empty(),
        )
        .unwrap();
        super::metal::attention_into(
            &q,
            &actual_k,
            &actual_v,
            None,
            None,
            &tables,
            &ctxlens,
            &block_bases,
            None,
            scale,
            2,
            1,
            &actual_output,
            super::metal::IntoResources::empty(),
        )
        .unwrap();
        dev.synchronize().unwrap();
        assert!(max_diff(&expected_k, &actual_k) < 1e-6);
        assert!(max_diff(&expected_v, &actual_v) < 1e-6);
        assert!(max_diff(&expected_output, &actual_output) < 1e-6);
    }

    #[test]
    fn grouped_queries_directly_index_kv_heads() {
        let dev = MetalDevice::get();
        let (batch, query_heads, kv_heads, chunk, head_dim) =
            (1usize, 4usize, 2usize, 1usize, 4usize);
        let q = MT::from_f32(
            dev,
            vec![0.0; batch * query_heads * chunk * head_dim],
            vec![batch, query_heads, chunk, head_dim],
        );
        let k_new = MT::from_f32(
            dev,
            vec![0.0; batch * kv_heads * chunk * head_dim],
            vec![batch, kv_heads, chunk, head_dim],
        );
        let v_new = MT::from_f32(
            dev,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![batch, kv_heads, chunk, head_dim],
        );
        let k_slab = MT::from_f32(
            dev,
            vec![0.0; 2 * kv_heads * head_dim],
            vec![2, kv_heads, head_dim],
        );
        let v_slab = MT::from_f32(
            dev,
            vec![0.0; 2 * kv_heads * head_dim],
            vec![2, kv_heads, head_dim],
        );
        let tables = u32_tensor(dev, &[0], vec![batch, 1]);
        let ctxlens = u32_tensor(dev, &[1], vec![batch]);
        let block_bases = u32_tensor(dev, &[0], vec![batch]);
        super::metal::scatter(
            &k_new,
            &v_new,
            &k_slab,
            &v_slab,
            None,
            None,
            &tables,
            &ctxlens,
            &block_bases,
            2,
            1,
        )
        .unwrap();
        let output = super::metal::decode(
            &q,
            &k_slab,
            &v_slab,
            None,
            None,
            &tables,
            &ctxlens,
            &block_bases,
            None,
            1.0,
            2,
            1,
        )
        .unwrap();
        dev.synchronize().unwrap();
        assert_eq!(
            output.read_f32().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0,]
        );
    }
}

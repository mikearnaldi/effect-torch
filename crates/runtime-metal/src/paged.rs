//! Paged decode attention on Metal for RFC 0013 stage 2. One kernel attends
//! q [B, H, 1, D] over pool slabs in place. It reads K/V rows through the block
//! table without gathering a contiguous copy. One threadgroup per sequence
//! slot and head streams that slot's blocks and accumulates an online softmax.
//! Context length is a runtime value; unlike the training flash pipeline, the
//! kernel bakes in no context-dependent shape. f16/bf16 slabs load natively.
//! int8 slabs dequantize in registers with the per-token, per-head scale slab
//! from RFC 0012. The primitive scatter and gather implementation in `lib.rs`
//! remains the reference and CPU fallback.
//!
//! ## Cache invariants
//!
//! - The pool is a pair of slabs `[pool_rows, H_kv, D]` (or
//!   `[pool_rows, H_kv * D]`) in `slab_dtype`; row addresses are
//!   computed device-side from the per-slot block table
//!   (`tables [B, maxBlocks] u32`), `ctxlens [B] u32` (post-run
//!   frontier), `block_bases [B] u32` (first visible table index),
//!   `advances [B] u32`, and `block_size`.
//! - `et_paged_scatter` uses one 32-thread simdgroup per slot and head. It
//!   writes rows `ctxlens[b] - advances[b] .. ctxlens[b]` of the new-token
//!   chunk into the slabs. Int8 slabs use an in-threadgroup absmax scale of
//!   `absmax/127 + eps`, then round with a +128 offset. The scale is stored per
//!   physical row and head.
//! - `et_paged_decode` uses one 256-thread threadgroup with eight simdgroups
//!   per slot, head, and chunk row. It stages q in threadgroup memory and
//!   streams K/V rows through the table. Each simdgroup owns one context row at
//!   a time. Each lane owns a `D/32`-wide head slice, using one 128-bit float4
//!   load when `D == 128` and a scalar loop otherwise. Per-lane online-softmax
//!   state is `(m, l, acc[LANE_D])`. Group partials fold in threadgroup memory,
//!   and lane 0 of group 0 writes the normalized output. Causality applies per
//!   chunk row: row `p` attends through `cursor + p`, with pads clamped to
//!   the real frontier.
//! - `et_paged_prefill_mma` and `et_paged_prefill` handle causal chunks of
//!   `PREFILL_MIN_CHUNK` or more rows with a query-tiled kernel, so each K/V
//!   row is streamed
//!   from DRAM once per tile instead of once per query row. f16 slabs
//!   with `D % 8 == 0` use the MMA kernel: one 128-thread threadgroup
//!   per (slot, head, 32-row tile); each simdgroup owns 8 query rows
//!   for the whole context walk, computing 8x8 score/PV tiles with
//!   `simdgroup_half8x8` MMAs straight off the slabs and running the
//!   per-row online softmax in registers (per-lane rows via the simd
//!   fragment layout, quad reductions via `simd_shuffle_xor`, running-
//!   max rescales skipped when the max does not move). Other slab
//!   dtypes (f32 reference precision, bf16/int8 conversion) use
//!   `et_paged_prefill`: one 256-thread threadgroup per (slot, head,
//!   tile of `PREFILL_QT` rows), each simdgroup owning `QT/8` rows and
//!   scoring via simd reductions. Same causal, window, GQA, and int8
//!   semantics as `et_paged_decode`; decode (`chunk == 1`), small
//!   chunks, and the block-bidirectional path keep the row-parallel
//!   kernel.
//! - Grouped-query attention maps query head `h` to K/V head
//!   `h / (H_q / H_kv)`; `H_q` must be a multiple of `H_kv`.
//! - A sliding `window` (0 = disabled) restricts the attended range to
//!   `[ctx - window, ctx)`.
//!
//! The `*_into` entry points allocate nothing, require empty resource
//! views, and need their exact pipelines pre-warmed.

use crate::runtime::dtype::DType;

/// Requirements for the paged scatter launch (writes
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

/// Requirements for a paged attention (decode) launch.
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
    /// Number of input vectors in the fixed query chunk.
    pub chunk: usize,
    /// Flash-decoding split count: 1 for the row-parallel kernel, > 1
    /// (causal `chunk == 1` only) for the split partial + combine pair.
    pub splits: usize,
    /// Bytes of the f32 `[B, H_q, C, D]` output.
    pub output_bytes: usize,
    /// Always 0: attention does not mutate the cache.
    pub state_next_bytes: usize,
    /// Always 0: no staging views.
    pub staging_bytes: usize,
    /// Always 0: no runtime status.
    pub status_bytes: usize,
    /// 0 when `splits == 1` (accumulators live in registers and
    /// threadgroup memory); otherwise the f32 `[B, H_q, S, D + 2]`
    /// partial-records scratch.
    pub scratch_bytes: usize,
    /// 1 for the single decode kernel, 2 for split partial + combine.
    pub pipeline_count: usize,
}

/// Visibility semantics for query rows in the current attention chunk.
///
/// This is independent of the processor architecture; Metal uses it to
/// specialize the paged-attention pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionMode {
    /// Query row `p` sees the committed prefix and current rows through `p`.
    Causal,
    /// Every query row sees the committed prefix and the entire current block.
    BidirectionalBlock,
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
        chunk,
        splits: 1,
        output_bytes,
        state_next_bytes: 0,
        staging_bytes: 0,
        status_bytes: 0,
        scratch_bytes: 0,
        pipeline_count: 1,
    })
}

/// Requirements of a flash-decoding split paged attention launch over
/// one causal query row per slot (`chunk == 1`): a partial kernel over
/// `splits` context subranges writing unnormalized f32 `[B, H_q, S,
/// D + 2]` records (`m`, `l`, `acc[D]`) into scratch, then a combine
/// kernel producing the normalized output.
pub fn attention_split_requirements(
    slab_dtype: DType,
    batch: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f64,
    splits: usize,
) -> crate::err::Res<AttentionRequirements> {
    if splits < 2 {
        return Err(format!(
            "paged attention split: splits must be at least 2, got {splits}"
        ));
    }
    let mut requirements =
        attention_requirements(slab_dtype, batch, query_heads, kv_heads, 1, head_dim, scale)?;
    requirements.splits = splits;
    requirements.scratch_bytes = batch
        .checked_mul(query_heads)
        .and_then(|value| value.checked_mul(splits))
        .and_then(|value| value.checked_mul(head_dim + 2))
        .and_then(|value| value.checked_mul(DType::F32.size_in_bytes()))
        .ok_or_else(|| "paged attention split: scratch byte size overflow".to_string())?;
    requirements.pipeline_count = 2;
    Ok(requirements)
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
        assert_eq!(attention.splits, 1);
        let split = attention_split_requirements(DType::F16, 2, 4, 4, 16, 0.25, 8).unwrap();
        assert_eq!(split.chunk, 1);
        assert_eq!(split.scratch_bytes, 2 * 4 * 8 * (16 + 2) * 4);
        assert_eq!(split.pipeline_count, 2);
        assert!(attention_split_requirements(DType::F16, 2, 4, 4, 16, 0.25, 1).is_err());
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
    attention_block, attention_block_into, attention_into, attention_prefill_into,
    attention_split_into, decode, decode_into, scatter, scatter_into, warm_all, warm_all_split,
    warm_attention, warm_attention_block, warm_attention_block_exact, warm_attention_exact,
    warm_attention_prefill, warm_attention_split, warm_attention_split_exact, warm_scatter,
    warm_scatter_exact, IntoResources,
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

    const ATTENTION_THREADS: usize = 256;

    /// Query rows per threadgroup tile in the chunked-prefill kernel
    /// (`et_paged_prefill`). Must be a multiple of the 8 simdgroups per
    /// threadgroup; 16 (two rows per simdgroup) is the measured sweet
    /// spot between DRAM-traffic reduction and register pressure.
    pub(crate) const PREFILL_QT: usize = 16;

    /// Query rows per threadgroup tile in the MMA prefill kernel
    /// (`et_paged_prefill_mma`); fixed by the kernel's 8-rows-per-
    /// simdgroup structure (4 simdgroups x 8 rows, 128 threads).
    pub(crate) const PREFILL_MMA_QT: usize = 32;

    /// Minimum causal chunk length routed to the query-tiled prefill
    /// kernel; smaller chunks keep the row-parallel kernel (tiling
    /// overhead does not pay off there).
    pub(crate) const PREFILL_MIN_CHUNK: usize = 8;

    /// Whether a paged attention launch with this mode and chunk runs
    /// the query-tiled prefill kernel instead of the row-parallel one.
    pub(crate) fn uses_tiled_prefill(mode: super::AttentionMode, chunk: usize) -> bool {
        mode == super::AttentionMode::Causal && chunk >= PREFILL_MIN_CHUNK
    }

    /// Whether a launch runs the MMA prefill kernel: causal chunked
    /// prefill on f16 slabs whose head dim fits the kernel's PV tiling
    /// and threadgroup budget. Other slab dtypes stay on the
    /// simd-reduction tiled kernel: the MMA path loads slab rows as
    /// half directly, which would break the f32 tier's reference
    /// precision and cannot convert bf16/int8 rows.
    pub(crate) fn uses_mma_prefill(
        mode: super::AttentionMode,
        chunk: usize,
        d: usize,
        slab_dtype: DType,
    ) -> bool {
        uses_tiled_prefill(mode, chunk) && d % 8 == 0 && d <= 128 && slab_dtype == DType::F16
    }

    /// Which kernel serves a paged attention dispatch.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum PrefillKernel {
        /// `et_paged_decode` row-parallel (decode, small chunks,
        /// block-bidirectional, exotic head dims).
        Rows,
        /// `et_paged_prefill` query-tiled with simd reductions
        /// (causal chunks, head dims the MMA kernel cannot tile).
        Tiled(usize),
        /// `et_paged_prefill_mma` query-tiled flash attention.
        Mma,
    }

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

    fn validate_advances(
        advances: &MetalTensor,
        ctxlens: &MetalTensor,
        batch: usize,
        chunk: usize,
        operation: &str,
    ) -> crate::err::Res<()> {
        advances.validate_destination(operation, &[batch], DType::U32)?;
        ctxlens.validate_destination(operation, &[batch], DType::U32)?;
        for lane in 0..batch {
            // Both tensors are validated contiguous shared u32 staging buffers.
            let advance = unsafe {
                *advances
                    .buffer
                    .contents_ptr()
                    .cast::<u32>()
                    .add(advances.layout.offset() + lane)
            } as usize;
            let context = unsafe {
                *ctxlens
                    .buffer
                    .contents_ptr()
                    .cast::<u32>()
                    .add(ctxlens.layout.offset() + lane)
            } as usize;
            if advance > chunk || advance > context {
                return Err(format!(
                    "{operation}: lane {lane} advance {advance} exceeds chunk {chunk} or context {context}"
                ));
            }
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
    device const uint* advances [[buffer(11)]],
    device const uint* blockBases [[buffer(12)]],
    uint3 gridDim [[threadgroups_per_grid]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {{
    const uint b = tgid.y;
    const uint h = tgid.x;
    const uint tid = tpitg.x;
    const uint C = gridDim.z;
    const uint advance = advances[b];
    if (advance == 0 || ctxlens[b] == 0) {{ return; }}
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
        chunk: usize,
    ) -> crate::err::Res<usize> {
        let mut count = 0;
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::U8] {
            MetalDevice::get().compile_lazy(scatter_key(d, dtype), "et_paged_scatter", || {
                scatter_source(d, dtype)
            })?;
            pipeline(
                d,
                query_heads,
                kv_heads,
                dtype,
                scale,
                super::AttentionMode::Causal,
                chunk,
            )?;
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
        chunk: usize,
    ) -> crate::err::Res<()> {
        pipeline(
            d,
            query_heads,
            kv_heads,
            slab_dtype,
            scale,
            super::AttentionMode::Causal,
            chunk,
        )?;
        Ok(())
    }

    /// Warms the row-parallel causal pipeline explicitly, bypassing the
    /// tiled-prefill routing (test and probe reference support).
    #[allow(dead_code)]
    pub(crate) fn warm_attention_rows(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        slab_dtype: DType,
        scale: f64,
        chunk: usize,
    ) -> crate::err::Res<()> {
        if kv_heads == 0 || !query_heads.is_multiple_of(kv_heads) {
            return Err("paged attention: query heads must be divisible by K/V heads".to_string());
        }
        MetalDevice::get().compile_lazy(
            attention_key(
                d,
                query_heads,
                kv_heads,
                slab_dtype,
                scale,
                super::AttentionMode::Causal,
                chunk,
            ),
            "et_paged_decode",
            || {
                kernel_source(
                    d,
                    query_heads,
                    kv_heads,
                    slab_dtype,
                    scale,
                    super::AttentionMode::Causal,
                )
            },
        )?;
        Ok(())
    }

    /// Warms block-bidirectional attention for the given geometry and scale.
    pub fn warm_attention_block(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        slab_dtype: DType,
        scale: f64,
        chunk: usize,
    ) -> crate::err::Res<()> {
        pipeline(
            d,
            query_heads,
            kv_heads,
            slab_dtype,
            scale,
            super::AttentionMode::BidirectionalBlock,
            chunk,
        )?;
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
            requirements.chunk,
        )
    }

    /// Warms block-bidirectional attention described by `requirements`.
    pub fn warm_attention_block_exact(
        requirements: &super::AttentionRequirements,
    ) -> crate::err::Res<()> {
        warm_attention_block(
            requirements.head_dim,
            requirements.query_heads,
            requirements.kv_heads,
            requirements.slab_dtype,
            f64::from_bits(requirements.scale_bits),
            requirements.chunk,
        )
    }

    /// Non-allocating fused batched scatter: `k_new`/`v_new [B, H, C,
    /// D]` f32 (C = 1 for decode, the chunk for prefill) are written
    /// into the slabs at rows `ctxlens[b] - advances[b] .. ctxlens[b]` per
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
        advances: &MetalTensor,
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
        validate_advances(advances, ctxlens, b, c, "paged scatter")?;
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
            set_buffer(e, 11, &advances.buffer, u32_off(advances.layout.offset()));
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

    /// Makes inputs contiguous, warms the pipeline, allocates outputs, and calls
    /// [`scatter_into`].
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
        advances: &MetalTensor,
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
            advances,
            IntoResources::empty(),
        )
    }

    /// Row-parallel paged attention: 256 threads per threadgroup (8
    /// simdgroups), one simdgroup per context row at a time, each lane
    /// owning a `LANE_D`-wide slice of the head dimension (`D / 4` lanes
    /// with a float4 load when `D == 128`). Group partials combine in
    /// threadgroup memory at the end.
    fn kernel_source(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        slab_dtype: DType,
        scale: f64,
        mode: super::AttentionMode,
    ) -> String {
        let (kv_ty, int8) = match slab_dtype {
            DType::F32 => ("float", 0),
            DType::F16 => ("half", 0),
            DType::BF16 => ("bfloat", 0),
            DType::U8 => ("uchar", 1),
            other => unreachable!("paged decode: unsupported slab dtype {other:?}"),
        };
        let bidirectional_block = usize::from(mode == super::AttentionMode::BidirectionalBlock);
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
#define ROW_GROUPS (NT / 32)
#define LANE_D ((D + 31) / 32)
#define SCALE {scale:?}f
#define T_KV {kv_ty}
#define INT8 {int8}
#define QH {query_heads}
#define KVH {kv_heads}
#define GROUP {head_group_size}
#define BIDIRECTIONAL_BLOCK {bidirectional_block}

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
    device const uint* advances [[buffer(12)]],
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
    const uint group = tid / 32;
    const uint lane = tid % 32;
    const uint needed = ctxlens[b];
    const uint advance = advances[b];
    if (advance == 0 || needed == 0) {{
        if (tid == 0) {{
            device float* o = O + ((ulong)b * QH * C + h * C + p) * D;
            for (int d = 0; d < D; d++) {{ o[d] = 0.0f; }}
        }}
        return;
    }}
    const uint cursor = needed - advance;
#if BIDIRECTIONAL_BLOCK
    const uint ctx = needed;
    const uint start = (window > 0 && cursor > window) ? cursor - window : 0;
#else
    const uint ctx = min(cursor + p + 1, needed);
    const uint start = (window > 0 && ctx > window) ? ctx - window : 0;
#endif
    device const uint* table = tables + (ulong)b * maxBlocks;

    threadgroup float qg[D];
    for (int d = tid; d < D; d += NT) {{ qg[d] = Q[((ulong)b * QH * C + h * C + p) * D + d]; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY;
    float l = 0.0f;
    float acc[LANE_D];
    for (int j = 0; j < LANE_D; j++) {{ acc[j] = 0.0f; }}

    for (uint row = start + group; row < ctx; row += ROW_GROUPS) {{
        const uint phys = table[row / blockSize - blockBases[b]] * blockSize + (row % blockSize);
        const ulong krow = (ulong)phys * KVH * D + kvh * D;
#if INT8
        const float ks = kscales[(ulong)phys * KVH + kvh];
#else
        const float ks = 1.0f;
#endif
        float score = 0.0f;
#if D == 128
        const uint dim = lane * 4;
        const float4 kv = kv_load4(K + krow + dim, ks);
        const float4 qv = float4(qg[dim], qg[dim + 1], qg[dim + 2], qg[dim + 3]);
        score = dot(qv, kv);
#else
        for (int j = 0; j < LANE_D; j++) {{
            const uint dim = lane * LANE_D + j;
            if (dim < D) {{
#if INT8
                const float kv = (float(K[krow + dim]) - 128.0f) * ks;
#else
                const float kv = float(K[krow + dim]);
#endif
                score += qg[dim] * kv;
            }}
        }}
#endif
        score = simd_sum(score) * SCALE;
        const float mn = max(m, score);
        const float rescale = (mn == -INFINITY) ? 0.0f : exp(m - mn);
        const float probability = (mn == -INFINITY) ? 0.0f : exp(score - mn);
#if INT8
        const float vs = vscales[(ulong)phys * KVH + kvh];
#else
        const float vs = 1.0f;
#endif
#if D == 128
        const float4 vv = kv_load4(V + krow + dim, vs);
        acc[0] = acc[0] * rescale + probability * vv.x;
        acc[1] = acc[1] * rescale + probability * vv.y;
        acc[2] = acc[2] * rescale + probability * vv.z;
        acc[3] = acc[3] * rescale + probability * vv.w;
#else
        for (int j = 0; j < LANE_D; j++) {{
            const uint dim = lane * LANE_D + j;
            if (dim < D) {{
#if INT8
                const float vv = (float(V[krow + dim]) - 128.0f) * vs;
#else
                const float vv = float(V[krow + dim]);
#endif
                acc[j] = acc[j] * rescale + probability * vv;
            }}
        }}
#endif
        l = l * rescale + probability;
        m = mn;
    }}

    threadgroup float pm[ROW_GROUPS];
    threadgroup float pl[ROW_GROUPS];
    threadgroup float pacc[ROW_GROUPS][D];
    if (lane == 0) {{
        pm[group] = m;
        pl[group] = l;
    }}
    for (int j = 0; j < LANE_D; j++) {{
        const uint dim = lane * LANE_D + j;
        if (dim < D) {{ pacc[group][dim] = acc[j]; }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {{
        float fm = -INFINITY;
        for (uint group = 0; group < ROW_GROUPS; group++) {{ fm = max(fm, pm[group]); }}
        float fl = 0.0f;
        device float* o = O + ((ulong)b * QH * C + h * C + p) * D;
        for (int d = 0; d < D; d++) {{ o[d] = 0.0f; }}
        for (uint group = 0; group < ROW_GROUPS; group++) {{
            const float rescale = (fm == -INFINITY) ? 0.0f : exp(pm[group] - fm);
            fl += pl[group] * rescale;
            for (int d = 0; d < D; d++) {{ o[d] += pacc[group][d] * rescale; }}
        }}
        const float inv = (fl == 0.0f) ? 0.0f : 1.0f / fl;
        for (int d = 0; d < D; d++) {{ o[d] *= inv; }}
    }}
}}
"#,
            d = d,
            nt = ATTENTION_THREADS,
            scale = scale as f32,
            kv_ty = kv_ty,
            int8 = int8,
            query_heads = query_heads,
            kv_heads = kv_heads,
            head_group_size = query_heads / kv_heads,
            bidirectional_block = bidirectional_block,
            load4_prelude = load4_prelude,
            load4 = load4,
        )
    }

    /// Query-tiled causal chunked prefill: grid `(query_heads, batch,
    /// ceil(chunk / QT))` of 256 threads (8 simdgroups). Each simdgroup
    /// owns `RPS = QT / 8` query rows of the tile and streams the whole
    /// visible context once, scoring its rows against every K/V row via
    /// per-lane dot products over its `LANE_D`-wide head slice plus a
    /// simd reduction, and running an independent online softmax per
    /// owned row entirely in registers. Every simdgroup reads the same
    /// K/V rows, so each row misses DRAM at most once per tile (the rest
    /// hit L1/L2). This reduces traffic by a factor of `QT` compared with the
    /// row-parallel kernel, which re-reads the context for every query row.
    /// Per-row causal visibility matches `et_paged_decode` exactly: absolute row
    /// `p` attends `[start_p, min(cursor + p + 1, needed))` with
    /// `start_p = ctx_p - window` when a window is set.
    fn prefill_kernel_source(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        slab_dtype: DType,
        scale: f64,
        qt: usize,
    ) -> String {
        let (kv_ty, int8) = match slab_dtype {
            DType::F32 => ("float", 0),
            DType::F16 => ("half", 0),
            DType::BF16 => ("bfloat", 0),
            DType::U8 => ("uchar", 1),
            other => unreachable!("paged prefill: unsupported slab dtype {other:?}"),
        };
        let load4 = match slab_dtype {
            DType::F32 => "float4(*(device const packed_float4*)base) * s".to_string(),
            DType::F16 => "float4(*(device const packed_half4*)base) * s".to_string(),
            DType::BF16 => "float4(*(device const packed_bfloat4*)base) * s".to_string(),
            DType::U8 => "((float4(float(packed & 0xFF), float((packed >> 8) & 0xFF), float((packed >> 16) & 0xFF), float((packed >> 24) & 0xFF))) - 128.0f) * s".to_string(),
            other => unreachable!("paged prefill: {other:?}"),
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
#define ROW_GROUPS (NT / 32)
#define QT {qt}
#define RPS (QT / ROW_GROUPS)
#define LANE_D ((D + 31) / 32)
#define SCALE {scale:?}f
#define T_KV {kv_ty}
#define INT8 {int8}
#define QH {query_heads}
#define KVH {kv_heads}
#define GROUP {head_group_size}

inline float4 kv_load4(device const T_KV* base, float s) {{
    {load4_prelude}
    return {load4};
}}

kernel void et_paged_prefill(
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
    device const uint* advances [[buffer(12)]],
    device const uint* blockBases [[buffer(13)]],
    constant uint& C [[buffer(14)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {{
    const uint b = tgid.y;
    const uint h = tgid.x;
    const uint kvh = h / GROUP;
    const uint p0 = tgid.z * QT;
    const uint tid = tpitg.x;
    const uint group = tid / 32;
    const uint lane = tid % 32;
    const uint needed = ctxlens[b];
    const uint advance = advances[b];
    const uint rows = min((uint)QT, C - p0);

    if (advance == 0 || needed == 0) {{
        for (uint i = 0; i < RPS; i++) {{
            const uint rr = group * RPS + i;
            if (rr < rows) {{
                device float* o = O + ((ulong)b * QH * C + h * C + p0 + rr) * D;
                for (int j = 0; j < LANE_D; j++) {{
                    const uint dim = lane * LANE_D + j;
                    if (dim < D) {{ o[dim] = 0.0f; }}
                }}
            }}
        }}
        return;
    }}
    const uint cursor = needed - advance;
    // Per-row causal visibility (identical to et_paged_decode): tile row
    // rr (absolute row p0 + rr) attends the committed prefix plus
    // in-chunk positions <= rr; pads clamp to the real frontier.
    uint ctx_r[RPS];
    uint start_r[RPS];
    bool live[RPS];
    uint lo = 0xFFFFFFFFu;
    uint hi = 0u;
#pragma unroll
    for (uint i = 0; i < RPS; i++) {{
        const uint rr = group * RPS + i;
        live[i] = rr < rows;
        const uint ctx = live[i] ? min(cursor + p0 + rr + 1, needed) : 0u;
        ctx_r[i] = ctx;
        start_r[i] = (window > 0 && ctx > window) ? ctx - window : 0u;
        if (live[i]) {{
            lo = min(lo, start_r[i]);
            hi = max(hi, ctx);
        }}
    }}
    device const uint* table = tables + (ulong)b * maxBlocks;

    float m[RPS];
    float l[RPS];
    float acc[RPS][LANE_D];
    float qr[RPS][LANE_D];
#pragma unroll
    for (uint i = 0; i < RPS; i++) {{
        m[i] = -INFINITY;
        l[i] = 0.0f;
        const uint rr = min(group * RPS + i, rows - 1);
        device const float* qrow = Q + ((ulong)b * QH * C + h * C + p0 + rr) * D;
        for (int j = 0; j < LANE_D; j++) {{
            acc[i][j] = 0.0f;
            const uint dim = lane * LANE_D + j;
            qr[i][j] = (live[i] && dim < D) ? qrow[dim] : 0.0f;
        }}
    }}

    for (uint row = lo; row < hi; row++) {{
        const uint phys = table[row / blockSize - blockBases[b]] * blockSize + (row % blockSize);
        const ulong kvrow = (ulong)phys * KVH * D + kvh * D;
#if INT8
        const float ks = kscales[(ulong)phys * KVH + kvh];
        const float vs = vscales[(ulong)phys * KVH + kvh];
#else
        const float ks = 1.0f;
        const float vs = 1.0f;
#endif
#if D == 128
        const uint dim = lane * 4;
        const float4 kv = kv_load4(K + kvrow + dim, ks);
        const float4 vv = kv_load4(V + kvrow + dim, vs);
#else
        float kvr[LANE_D];
        float vvr[LANE_D];
        for (int j = 0; j < LANE_D; j++) {{
            const uint dim = lane * LANE_D + j;
            if (dim < D) {{
#if INT8
                kvr[j] = (float(K[kvrow + dim]) - 128.0f) * ks;
                vvr[j] = (float(V[kvrow + dim]) - 128.0f) * vs;
#else
                kvr[j] = float(K[kvrow + dim]);
                vvr[j] = float(V[kvrow + dim]);
#endif
            }} else {{
                kvr[j] = 0.0f;
                vvr[j] = 0.0f;
            }}
        }}
#endif
#pragma unroll
        for (uint i = 0; i < RPS; i++) {{
#if D == 128
            const float partial = dot(float4(qr[i][0], qr[i][1], qr[i][2], qr[i][3]), kv);
#else
            float partial = 0.0f;
            for (int j = 0; j < LANE_D; j++) {{ partial += qr[i][j] * kvr[j]; }}
#endif
            float score = simd_sum(partial) * SCALE;
            if (row < start_r[i] || row >= ctx_r[i]) {{ score = -INFINITY; }}
            const float mn = max(m[i], score);
            const float rescale = (mn == -INFINITY) ? 0.0f : exp(m[i] - mn);
            const float probability = (mn == -INFINITY) ? 0.0f : exp(score - mn);
#if D == 128
            acc[i][0] = acc[i][0] * rescale + probability * vv.x;
            acc[i][1] = acc[i][1] * rescale + probability * vv.y;
            acc[i][2] = acc[i][2] * rescale + probability * vv.z;
            acc[i][3] = acc[i][3] * rescale + probability * vv.w;
#else
            for (int j = 0; j < LANE_D; j++) {{
                acc[i][j] = acc[i][j] * rescale + probability * vvr[j];
            }}
#endif
            l[i] = l[i] * rescale + probability;
            m[i] = mn;
        }}
    }}

    for (uint i = 0; i < RPS; i++) {{
        if (!live[i]) {{ continue; }}
        const uint rr = group * RPS + i;
        device float* o = O + ((ulong)b * QH * C + h * C + p0 + rr) * D;
        const float inv = (l[i] == 0.0f) ? 0.0f : 1.0f / l[i];
        for (int j = 0; j < LANE_D; j++) {{
            const uint dim = lane * LANE_D + j;
            if (dim < D) {{ o[dim] = acc[i][j] * inv; }}
        }}
    }}
}}
"#,
            d = d,
            nt = ATTENTION_THREADS,
            qt = qt,
            scale = scale as f32,
            kv_ty = kv_ty,
            int8 = int8,
            query_heads = query_heads,
            kv_heads = kv_heads,
            head_group_size = query_heads / kv_heads,
            load4_prelude = load4_prelude,
            load4 = load4,
        )
    }

    /// Query-tiled causal chunked prefill via simdgroup matrix
    /// multiply-accumulate (flash-attention structure, same shape as
    /// llama.cpp's `kernel_flash_attn_ext`): grid `(query_heads, batch,
    /// ceil(chunk / 32))` of 128 threads (4 simdgroups). f16 slabs
    /// only. Each simdgroup owns 8 query rows of the tile for the
    /// whole walk: it stages its q rows in its own threadgroup-memory
    /// slice, then for every 8 context rows loads the K^T and V 8x8
    /// tiles straight from the slab (8-row runs aligned to 8 never
    /// cross a page when `blockSize % 8 == 0`, a dispatch-time gate),
    /// computes the `8 x 8` score tile with `simdgroup_half8x8` MMAs
    /// (f32 accumulate), runs the per-row online softmax entirely in
    /// registers (per-lane rows via the fragment layout, quad
    /// reductions via `simd_shuffle_xor`), converts the probabilities
    /// to a half fragment in place, and accumulates `P x V` into
    /// persistent per-row accumulator tiles. No K/V staging, no
    /// threadgroup barriers in the walk, and no cross-simdgroup reduction. Each
    /// K/V row misses DRAM at most once per eight query rows, and score
    /// computation is matrix-unit bound instead of shuffle bound. Per-row
    /// causal visibility matches `et_paged_decode` exactly. Requires `D % 8 == 0`, `D <= 128`
    /// (accumulator register budget). f32/bf16/u8 slabs route to
    /// `et_paged_prefill` instead (half loads would break f32
    /// reference precision and cannot convert bf16/int8 rows).
    fn prefill_mma_kernel_source(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        scale: f64,
    ) -> String {
        format!(
            r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define D {d}
#define NT 128
#define QT 32
#define SCALE {scale:?}f
#define QH {query_heads}
#define KVH {kv_heads}
#define GROUP {head_group_size}

kernel void et_paged_prefill_mma(
    device const float* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device const uint* tables [[buffer(3)]],
    device const uint* ctxlens [[buffer(4)]],
    device float* O [[buffer(5)]],
    device const float* kscales [[buffer(6)]],
    device const float* vscales [[buffer(7)]],
    constant uint& blockSize [[buffer(8)]],
    constant uint& maxBlocks [[buffer(9)]],
    constant uint& window [[buffer(10)]],
    constant uint& H [[buffer(11)]],
    device const uint* advances [[buffer(12)]],
    device const uint* blockBases [[buffer(13)]],
    constant uint& C [[buffer(14)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]
) {{
    const uint b = tgid.y;
    const uint h = tgid.x;
    const uint kvh = h / GROUP;
    const uint p0 = tgid.z * QT;
    const uint needed = ctxlens[b];
    const uint advance = advances[b];
    const uint rows = min((uint)QT, C - p0);

    threadgroup half q_s[QT * D];

    if (advance == 0 || needed == 0) {{
        for (uint e = tid; e < rows * D; e += NT) {{
            O[((ulong)b * QH * C + h * C + p0 + e / D) * D + e % D] = 0.0f;
        }}
        return;
    }}
    const uint cursor = needed - advance;
    device const uint* table = tables + (ulong)b * maxBlocks;

    // Stage this simdgroup's 8 query rows in its own slice (zero-padded
    // beyond the live rows); only a simdgroup-scope barrier is needed.
    {{
        const uint live = (rows > sg * 8) ? min(8u, rows - sg * 8) : 0u;
        device const float* qbase = Q + ((ulong)b * QH * C + h * C + p0 + sg * 8) * D;
        threadgroup half* qs = q_s + sg * 8 * D;
        for (uint e = lane; e < 8 * D; e += 32) {{
            qs[e] = (e / D < live) ? half(qbase[e]) : half(0.0h);
        }}
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Fragment layout (verified by simdgroup_float8x8_layout_probe):
    // lane t owns row (t/16)*4 + (t%8)/2 and columns ((t%16)/8)*4 +
    // (t%2)*2 + {{0, 1}} of each 8x8 tile. Each lane owns one row, so its
    // online-softmax state is local and row reductions are quad
    // shuffles (lanes of a row are closed under xor masks 1 and 8).
    const uint fr = (lane / 16) * 4 + ((lane % 8) / 2);
    const uint fc = ((lane % 16) / 8) * 4 + (lane % 2) * 2;
    const uint i = sg * 8 + fr;
    const uint ctx_i = min(cursor + p0 + i + 1, needed);
    const uint start_i = (window > 0 && ctx_i > window) ? ctx_i - window : 0u;
    // Group-local walk bounds: its 8 rows share ctx within 8 of each
    // other. Aligned down to 8 so direct 8-row slab loads never cross
    // a page; per-element masking re-applies exact visibility.
    const uint ctx_lo = min(cursor + p0 + sg * 8 + 1, needed);
    const uint hi = min(cursor + p0 + min(sg * 8 + 8u, rows), needed);
    const uint start_lo = (window > 0 && ctx_lo > window) ? ctx_lo - window : 0u;
    const uint lo = start_lo & ~7u;

    float m_run = -INFINITY;
    float l_run = 0.0f;
    simdgroup_float8x8 acc[D / 8];
#pragma unroll
    for (int j = 0; j < D / 8; j++) {{ acc[j] = simdgroup_float8x8(0.0f); }}
    // The q A-fragments are loop-invariant: hold them in registers.
    simdgroup_half8x8 qa[D / 8];
#pragma unroll
    for (int j = 0; j < D / 8; j++) {{
        simdgroup_load(qa[j], &q_s[sg * 8 * D + j * 8], D);
    }}

    for (uint kv0 = lo; kv0 < hi; kv0 += 16) {{
        // Two 8-row blocks in flight per iteration for ILP.
        const ulong kvrow0 = (kv0 < needed)
            ? (ulong)(table[kv0 / blockSize - blockBases[b]] * blockSize + (kv0 % blockSize))
                * KVH * D + kvh * D
            : 0u;
        const ulong kvrow1 = (kv0 + 8 < needed)
            ? (ulong)(table[(kv0 + 8) / blockSize - blockBases[b]] * blockSize
                    + ((kv0 + 8) % blockSize))
                * KVH * D + kvh * D
            : 0u;
        simdgroup_float8x8 sa(0.0f);
        simdgroup_float8x8 sb(0.0f);
#pragma unroll
        for (int j = 0; j < D / 8; j++) {{
            simdgroup_half8x8 b0;
            simdgroup_half8x8 b1;
            if (kv0 < needed) {{
                simdgroup_load(b0, K + kvrow0 + j * 8, KVH * D, 0, true);
            }} else {{
                b0 = simdgroup_half8x8(0.0h);
            }}
            if (kv0 + 8 < needed) {{
                simdgroup_load(b1, K + kvrow1 + j * 8, KVH * D, 0, true);
            }} else {{
                b1 = simdgroup_half8x8(0.0h);
            }}
            simdgroup_multiply_accumulate(sa, qa[j], b0, sa);
            simdgroup_multiply_accumulate(sb, qa[j], b1, sb);
        }}
        // Online softmax for block A then block B (sequential order is
        // part of the semantics: rescale A applies before PV-A, etc.).
        float pa0;
        float pa1;
        float pb0;
        float pb1;
        {{
            float s0 = sa.thread_elements()[0] * SCALE;
            float s1 = sa.thread_elements()[1] * SCALE;
            if (kv0 + fc < start_i || kv0 + fc >= ctx_i) {{ s0 = -INFINITY; }}
            if (kv0 + fc + 1 < start_i || kv0 + fc + 1 >= ctx_i) {{ s1 = -INFINITY; }}
            float m_tile = max(s0, s1);
            m_tile = max(m_tile, simd_shuffle_xor(m_tile, 1));
            m_tile = max(m_tile, simd_shuffle_xor(m_tile, 8));
            const float m_new = max(m_run, m_tile);
            // The running max saturates quickly; only touch the
            // accumulators when it actually moves.
            if (m_new > m_run) {{
                const float rescale = (m_run == -INFINITY) ? 0.0f : exp(m_run - m_new);
#pragma unroll
                for (int j = 0; j < D / 8; j++) {{
                    acc[j].thread_elements()[0] *= rescale;
                    acc[j].thread_elements()[1] *= rescale;
                }}
                l_run *= rescale;
                m_run = m_new;
            }}
            pa0 = (s0 == -INFINITY) ? 0.0f : exp(s0 - m_run);
            pa1 = (s1 == -INFINITY) ? 0.0f : exp(s1 - m_run);
            float l_blk = pa0 + pa1;
            l_blk += simd_shuffle_xor(l_blk, 1);
            l_blk += simd_shuffle_xor(l_blk, 8);
            l_run += l_blk;
            simdgroup_half8x8 a;
            a.thread_elements()[0] = half(pa0);
            a.thread_elements()[1] = half(pa1);
#pragma unroll
            for (int j = 0; j < D / 8; j++) {{
                simdgroup_half8x8 bmat;
                if (kv0 < needed) {{
                    simdgroup_load(bmat, V + kvrow0 + j * 8, KVH * D);
                }} else {{
                    bmat = simdgroup_half8x8(0.0h);
                }}
                simdgroup_multiply_accumulate(acc[j], a, bmat, acc[j]);
            }}
        }}
        {{
            float s0 = sb.thread_elements()[0] * SCALE;
            float s1 = sb.thread_elements()[1] * SCALE;
            const uint kb = kv0 + 8;
            if (kb + fc < start_i || kb + fc >= ctx_i) {{ s0 = -INFINITY; }}
            if (kb + fc + 1 < start_i || kb + fc + 1 >= ctx_i) {{ s1 = -INFINITY; }}
            float m_tile = max(s0, s1);
            m_tile = max(m_tile, simd_shuffle_xor(m_tile, 1));
            m_tile = max(m_tile, simd_shuffle_xor(m_tile, 8));
            const float m_new = max(m_run, m_tile);
            if (m_new > m_run) {{
                const float rescale = (m_run == -INFINITY) ? 0.0f : exp(m_run - m_new);
#pragma unroll
                for (int j = 0; j < D / 8; j++) {{
                    acc[j].thread_elements()[0] *= rescale;
                    acc[j].thread_elements()[1] *= rescale;
                }}
                l_run *= rescale;
                m_run = m_new;
            }}
            pb0 = (s0 == -INFINITY) ? 0.0f : exp(s0 - m_run);
            pb1 = (s1 == -INFINITY) ? 0.0f : exp(s1 - m_run);
            float l_blk = pb0 + pb1;
            l_blk += simd_shuffle_xor(l_blk, 1);
            l_blk += simd_shuffle_xor(l_blk, 8);
            l_run += l_blk;
            simdgroup_half8x8 a;
            a.thread_elements()[0] = half(pb0);
            a.thread_elements()[1] = half(pb1);
#pragma unroll
            for (int j = 0; j < D / 8; j++) {{
                simdgroup_half8x8 bmat;
                if (kb < needed) {{
                    simdgroup_load(bmat, V + kvrow1 + j * 8, KVH * D);
                }} else {{
                    bmat = simdgroup_half8x8(0.0h);
                }}
                simdgroup_multiply_accumulate(acc[j], a, bmat, acc[j]);
            }}
        }}
    }}

    // Normalize and write the group's rows (same fragment layout).
    if (i < rows) {{
        const float inv = (l_run == 0.0f) ? 0.0f : 1.0f / l_run;
        device float* o = O + ((ulong)b * QH * C + h * C + p0 + i) * D;
#pragma unroll
        for (int j = 0; j < D / 8; j++) {{
            o[j * 8 + fc] = acc[j].thread_elements()[0] * inv;
            o[j * 8 + fc + 1] = acc[j].thread_elements()[1] * inv;
        }}
    }}
}}
"#,
            d = d,
            scale = scale as f32,
            query_heads = query_heads,
            kv_heads = kv_heads,
            head_group_size = query_heads / kv_heads,
        )
    }

    /// Flash-decoding split partial kernel (causal one-token decode
    /// only): grid `(query_heads, batch, splits)` of 256 threads; split
    /// `s` owns the deterministic subrange `[start + len*s/S, start +
    /// len*(s+1)/S)` of `[start, ctx)` and runs the same row-parallel
    /// per-simdgroup online-softmax accumulation and threadgroup fold
    /// as `et_paged_decode`, but writes the raw unnormalized f32
    /// partial record `(m, l, acc[D])` to scratch `[B, H, S, D + 2]`.
    /// Every split writes, even when its subrange (or the slot) is
    /// empty: `(-inf, 0, zeros)`.
    fn split_kernel_source(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        slab_dtype: DType,
        scale: f64,
        splits: usize,
    ) -> String {
        let (kv_ty, int8) = match slab_dtype {
            DType::F32 => ("float", 0),
            DType::F16 => ("half", 0),
            DType::BF16 => ("bfloat", 0),
            DType::U8 => ("uchar", 1),
            other => unreachable!("paged decode split: unsupported slab dtype {other:?}"),
        };
        let load4 = match slab_dtype {
            DType::F32 => "float4(*(device const packed_float4*)base) * s".to_string(),
            DType::F16 => "float4(*(device const packed_half4*)base) * s".to_string(),
            DType::BF16 => "float4(*(device const packed_bfloat4*)base) * s".to_string(),
            DType::U8 => "((float4(float(packed & 0xFF), float((packed >> 8) & 0xFF), float((packed >> 16) & 0xFF), float((packed >> 24) & 0xFF))) - 128.0f) * s".to_string(),
            other => unreachable!("paged decode split: {other:?}"),
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
#define ROW_GROUPS (NT / 32)
#define LANE_D ((D + 31) / 32)
#define SCALE {scale:?}f
#define T_KV {kv_ty}
#define INT8 {int8}
#define QH {query_heads}
#define KVH {kv_heads}
#define GROUP {head_group_size}
#define SPLITS {splits}

inline float4 kv_load4(device const T_KV* base, float s) {{
    {load4_prelude}
    return {load4};
}}

kernel void et_paged_decode_split(
    device const float* Q [[buffer(0)]],
    device const T_KV* K [[buffer(1)]],
    device const T_KV* V [[buffer(2)]],
    device const uint* tables [[buffer(3)]],
    device const uint* ctxlens [[buffer(4)]],
    device float* P [[buffer(5)]],
    device const float* kscales [[buffer(6)]],
    device const float* vscales [[buffer(7)]],
    constant uint& blockSize [[buffer(8)]],
    constant uint& maxBlocks [[buffer(9)]],
    constant uint& window [[buffer(10)]],
    constant uint& H [[buffer(11)]],
    device const uint* advances [[buffer(12)]],
    device const uint* blockBases [[buffer(13)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {{
    const uint b = tgid.y;
    const uint h = tgid.x;
    const uint kvh = h / GROUP;
    const uint s = tgid.z;
    const uint tid = tpitg.x;
    const uint group = tid / 32;
    const uint lane = tid % 32;
    device float* rec = P + (((ulong)b * QH + h) * SPLITS + s) * (D + 2);
    const uint needed = ctxlens[b];
    const uint advance = advances[b];
    if (advance == 0 || needed == 0) {{
        if (tid == 0) {{
            rec[0] = -INFINITY;
            rec[1] = 0.0f;
        }}
        for (uint d = tid; d < D; d += NT) {{ rec[2 + d] = 0.0f; }}
        return;
    }}
    const uint cursor = needed - advance;
    const uint ctx = min(cursor + 1, needed);
    const uint start = (window > 0 && ctx > window) ? ctx - window : 0;
    const uint len = ctx - start;
    const uint lo = start + len * s / SPLITS;
    const uint hi = start + len * (s + 1) / SPLITS;
    device const uint* table = tables + (ulong)b * maxBlocks;

    threadgroup float qg[D];
    for (int d = tid; d < D; d += NT) {{ qg[d] = Q[((ulong)b * QH + h) * D + d]; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY;
    float l = 0.0f;
    float acc[LANE_D];
    for (int j = 0; j < LANE_D; j++) {{ acc[j] = 0.0f; }}

    for (uint row = lo + group; row < hi; row += ROW_GROUPS) {{
        const uint phys = table[row / blockSize - blockBases[b]] * blockSize + (row % blockSize);
        const ulong krow = (ulong)phys * KVH * D + kvh * D;
#if INT8
        const float ks = kscales[(ulong)phys * KVH + kvh];
#else
        const float ks = 1.0f;
#endif
        float score = 0.0f;
#if D == 128
        const uint dim = lane * 4;
        const float4 kv = kv_load4(K + krow + dim, ks);
        const float4 qv = float4(qg[dim], qg[dim + 1], qg[dim + 2], qg[dim + 3]);
        score = dot(qv, kv);
#else
        for (int j = 0; j < LANE_D; j++) {{
            const uint dim = lane * LANE_D + j;
            if (dim < D) {{
#if INT8
                const float kv = (float(K[krow + dim]) - 128.0f) * ks;
#else
                const float kv = float(K[krow + dim]);
#endif
                score += qg[dim] * kv;
            }}
        }}
#endif
        score = simd_sum(score) * SCALE;
        const float mn = max(m, score);
        const float rescale = (mn == -INFINITY) ? 0.0f : exp(m - mn);
        const float probability = (mn == -INFINITY) ? 0.0f : exp(score - mn);
#if INT8
        const float vs = vscales[(ulong)phys * KVH + kvh];
#else
        const float vs = 1.0f;
#endif
#if D == 128
        const float4 vv = kv_load4(V + krow + dim, vs);
        acc[0] = acc[0] * rescale + probability * vv.x;
        acc[1] = acc[1] * rescale + probability * vv.y;
        acc[2] = acc[2] * rescale + probability * vv.z;
        acc[3] = acc[3] * rescale + probability * vv.w;
#else
        for (int j = 0; j < LANE_D; j++) {{
            const uint dim = lane * LANE_D + j;
            if (dim < D) {{
#if INT8
                const float vv = (float(V[krow + dim]) - 128.0f) * vs;
#else
                const float vv = float(V[krow + dim]);
#endif
                acc[j] = acc[j] * rescale + probability * vv;
            }}
        }}
#endif
        l = l * rescale + probability;
        m = mn;
    }}

    threadgroup float pm[ROW_GROUPS];
    threadgroup float pl[ROW_GROUPS];
    threadgroup float pacc[ROW_GROUPS][D];
    if (lane == 0) {{
        pm[group] = m;
        pl[group] = l;
    }}
    for (int j = 0; j < LANE_D; j++) {{
        const uint dim = lane * LANE_D + j;
        if (dim < D) {{ pacc[group][dim] = acc[j]; }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {{
        float fm = -INFINITY;
        for (uint group = 0; group < ROW_GROUPS; group++) {{ fm = max(fm, pm[group]); }}
        float fl = 0.0f;
        for (int d = 0; d < D; d++) {{ rec[2 + d] = 0.0f; }}
        for (uint group = 0; group < ROW_GROUPS; group++) {{
            const float rescale = (fm == -INFINITY) ? 0.0f : exp(pm[group] - fm);
            fl += pl[group] * rescale;
            for (int d = 0; d < D; d++) {{ rec[2 + d] += pacc[group][d] * rescale; }}
        }}
        rec[0] = fm;
        rec[1] = fl;
    }}
}}
"#,
            d = d,
            nt = ATTENTION_THREADS,
            scale = scale as f32,
            kv_ty = kv_ty,
            int8 = int8,
            query_heads = query_heads,
            kv_heads = kv_heads,
            head_group_size = query_heads / kv_heads,
            splits = splits,
            load4_prelude = load4_prelude,
            load4 = load4,
        )
    }

    /// Split combine kernel: one threadgroup of 32 threads per
    /// `(head, slot)` reduces the `SPLITS` partial records in scratch
    /// `[B, H, S, D + 2]`. It reduces max `m`, then weights `l` and `acc`
    /// by `exp(m_s - M)` in fixed split order, and writes the normalized
    /// `[B, H, 1, D]` output row. An all-empty slot writes zeros.
    fn combine_kernel_source(d: usize, query_heads: usize, splits: usize) -> String {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;

#define D {d}
#define NT 32
#define QH {query_heads}
#define SPLITS {splits}

kernel void et_paged_decode_combine(
    device const float* P [[buffer(0)]],
    device float* O [[buffer(1)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {{
    const uint b = tgid.y;
    const uint h = tgid.x;
    const uint tid = tpitg.x;
    device const float* base = P + ((ulong)b * QH + h) * SPLITS * (D + 2);
    float maximum = -INFINITY;
    for (uint s = 0; s < SPLITS; s++) {{ maximum = max(maximum, base[s * (D + 2)]); }}
    float weights[SPLITS];
    float l = 0.0f;
    for (uint s = 0; s < SPLITS; s++) {{
        const float w = (maximum == -INFINITY) ? 0.0f : exp(base[s * (D + 2)] - maximum);
        weights[s] = w;
        l += base[s * (D + 2) + 1] * w;
    }}
    const float inv = (l == 0.0f) ? 0.0f : 1.0f / l;
    device float* o = O + ((ulong)b * QH + h) * D;
    for (uint d = tid; d < D; d += NT) {{
        float acc = 0.0f;
        for (uint s = 0; s < SPLITS; s++) {{ acc += base[s * (D + 2) + 2 + d] * weights[s]; }}
        o[d] = acc * inv;
    }}
}}
"#,
            d = d,
            query_heads = query_heads,
            splits = splits,
        )
    }

    fn split_partial_key(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        slab_dtype: DType,
        scale: f64,
        splits: usize,
    ) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        (
            0x5D17u32,
            d,
            query_heads,
            kv_heads,
            slab_dtype,
            scale.to_bits(),
            splits,
        )
            .hash(&mut hasher);
        hasher.finish()
    }

    fn split_combine_key(d: usize, query_heads: usize, splits: usize) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        (0xC0B2u32, d, query_heads, splits).hash(&mut hasher);
        hasher.finish()
    }

    fn split_pipelines(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        slab_dtype: DType,
        scale: f64,
        splits: usize,
    ) -> crate::err::Res<(Pipeline, Pipeline)> {
        if splits < 2 {
            return Err(format!(
                "paged attention split: splits must be at least 2, got {splits}"
            ));
        }
        if kv_heads == 0 || !query_heads.is_multiple_of(kv_heads) {
            return Err("paged attention: query heads must be divisible by K/V heads".to_string());
        }
        let device = MetalDevice::get();
        let partial = device.compile_lazy(
            split_partial_key(d, query_heads, kv_heads, slab_dtype, scale, splits),
            "et_paged_decode_split",
            || split_kernel_source(d, query_heads, kv_heads, slab_dtype, scale, splits),
        )?;
        let combine = device.compile_lazy(
            split_combine_key(d, query_heads, splits),
            "et_paged_decode_combine",
            || combine_kernel_source(d, query_heads, splits),
        )?;
        Ok((partial, combine))
    }

    /// Warms the split partial and combine pipelines for the given
    /// head geometry, slab dtype, softmax scale, and split count.
    pub fn warm_attention_split(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        slab_dtype: DType,
        scale: f64,
        splits: usize,
    ) -> crate::err::Res<()> {
        split_pipelines(d, query_heads, kv_heads, slab_dtype, scale, splits)?;
        Ok(())
    }

    /// Warms the split partial and combine pipelines for every
    /// supported slab dtype; returns the number of pipelines compiled
    /// or fetched.
    pub fn warm_all_split(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        scale: f64,
        splits: usize,
    ) -> crate::err::Res<usize> {
        let mut count = 0;
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::U8] {
            warm_attention_split(d, query_heads, kv_heads, dtype, scale, splits)?;
            count += 2;
        }
        Ok(count)
    }

    /// Warms exactly the split pipelines described by `requirements`.
    pub fn warm_attention_split_exact(
        requirements: &super::AttentionRequirements,
    ) -> crate::err::Res<()> {
        warm_attention_split(
            requirements.head_dim,
            requirements.query_heads,
            requirements.kv_heads,
            requirements.slab_dtype,
            f64::from_bits(requirements.scale_bits),
            requirements.splits,
        )
    }

    fn attention_key(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        slab_dtype: DType,
        scale: f64,
        mode: super::AttentionMode,
        chunk: usize,
    ) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        (
            d,
            query_heads,
            kv_heads,
            slab_dtype,
            scale.to_bits(),
            mode,
            chunk,
        )
            .hash(&mut hasher);
        hasher.finish()
    }

    fn prefill_key(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        slab_dtype: DType,
        scale: f64,
        chunk: usize,
        qt: usize,
    ) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        (
            0x9FE1u32,
            d,
            query_heads,
            kv_heads,
            slab_dtype,
            scale.to_bits(),
            chunk,
            qt,
        )
            .hash(&mut hasher);
        hasher.finish()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_pipeline(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        slab_dtype: DType,
        scale: f64,
        chunk: usize,
        qt: usize,
    ) -> crate::err::Res<Pipeline> {
        if kv_heads == 0 || !query_heads.is_multiple_of(kv_heads) {
            return Err("paged attention: query heads must be divisible by K/V heads".to_string());
        }
        if qt == 0 || !qt.is_multiple_of(ATTENTION_THREADS / 32) {
            return Err(format!(
                "paged prefill: query tile {qt} must be a positive multiple of {}",
                ATTENTION_THREADS / 32
            ));
        }
        MetalDevice::get().compile_lazy(
            prefill_key(d, query_heads, kv_heads, slab_dtype, scale, chunk, qt),
            "et_paged_prefill",
            || prefill_kernel_source(d, query_heads, kv_heads, slab_dtype, scale, qt),
        )
    }

    fn prefill_mma_key(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        slab_dtype: DType,
        scale: f64,
        chunk: usize,
    ) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        (
            0xFFAAu32,
            d,
            query_heads,
            kv_heads,
            slab_dtype,
            scale.to_bits(),
            chunk,
        )
            .hash(&mut hasher);
        hasher.finish()
    }

    pub(crate) fn prefill_mma_pipeline(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        slab_dtype: DType,
        scale: f64,
        chunk: usize,
    ) -> crate::err::Res<Pipeline> {
        if kv_heads == 0 || !query_heads.is_multiple_of(kv_heads) {
            return Err("paged attention: query heads must be divisible by K/V heads".to_string());
        }
        if d % 8 != 0 || d > 128 {
            return Err(format!(
                "paged prefill mma: head dim {d} must be a multiple of 8 and at most 128"
            ));
        }
        if slab_dtype != DType::F16 {
            return Err(format!(
                "paged prefill mma: slab dtype {slab_dtype:?} must be f16 (direct half loads)"
            ));
        }
        MetalDevice::get().compile_lazy(
            prefill_mma_key(d, query_heads, kv_heads, slab_dtype, scale, chunk),
            "et_paged_prefill_mma",
            || prefill_mma_kernel_source(d, query_heads, kv_heads, scale),
        )
    }

    fn pipeline(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        slab_dtype: DType,
        scale: f64,
        mode: super::AttentionMode,
        chunk: usize,
    ) -> crate::err::Res<Pipeline> {
        if kv_heads == 0 || !query_heads.is_multiple_of(kv_heads) {
            return Err("paged attention: query heads must be divisible by K/V heads".to_string());
        }
        if uses_mma_prefill(mode, chunk, d, slab_dtype) {
            // Warm the simd-reduction tiled kernel as well: dispatches
            // whose block size would let an 8-row run cross a page fall
            // back to it.
            prefill_pipeline(
                d,
                query_heads,
                kv_heads,
                slab_dtype,
                scale,
                chunk,
                PREFILL_QT,
            )?;
            return prefill_mma_pipeline(d, query_heads, kv_heads, slab_dtype, scale, chunk);
        }
        if uses_tiled_prefill(mode, chunk) {
            return prefill_pipeline(
                d,
                query_heads,
                kv_heads,
                slab_dtype,
                scale,
                chunk,
                PREFILL_QT,
            );
        }
        MetalDevice::get().compile_lazy(
            attention_key(d, query_heads, kv_heads, slab_dtype, scale, mode, chunk),
            "et_paged_decode",
            || kernel_source(d, query_heads, kv_heads, slab_dtype, scale, mode),
        )
    }

    /// Shared non-allocating paged attention dispatch: validates the
    /// inputs, binds buffers 0..=13, and launches either the
    /// row-parallel kernel ([`PrefillKernel::Rows`], grid depth =
    /// chunk) or a query-tiled causal prefill kernel
    /// ([`PrefillKernel::Tiled`]/[`PrefillKernel::Mma`], grid depth =
    /// ceil(chunk / tile), chunk passed on buffer 14).
    #[allow(clippy::too_many_arguments)]
    fn attention_dispatch(
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
        advances: &MetalTensor,
        mode: super::AttentionMode,
        kernel: PrefillKernel,
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
        validate_advances(advances, ctxlens, b, c, "paged attention")?;
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
            .pipeline_cached(match kernel {
                PrefillKernel::Rows => attention_key(d, h, kv_heads, slab_dtype, scale, mode, c),
                PrefillKernel::Tiled(qt) => prefill_key(d, h, kv_heads, slab_dtype, scale, c, qt),
                PrefillKernel::Mma => prefill_mma_key(d, h, kv_heads, slab_dtype, scale, c),
            })
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
            set_buffer(e, 12, &advances.buffer, u32_off(advances.layout.offset()));
            set_buffer(
                e,
                13,
                &block_bases.buffer,
                u32_off(block_bases.layout.offset()),
            );
            let (depth, threads) = match kernel {
                PrefillKernel::Rows => (c, ATTENTION_THREADS),
                PrefillKernel::Tiled(qt) => (c.div_ceil(qt), ATTENTION_THREADS),
                PrefillKernel::Mma => (c.div_ceil(PREFILL_MMA_QT), 128),
            };
            if kernel != PrefillKernel::Rows {
                set_bytes(e, 14, &(c as u32));
            }
            e.dispatchThreadgroups_threadsPerThreadgroup(
                objc2_metal::MTLSize {
                    width: h,
                    height: b,
                    depth,
                },
                objc2_metal::MTLSize {
                    width: threads,
                    height: 1,
                    depth: 1,
                },
            );
        });
        Ok(())
    }

    /// Non-allocating paged attention: one launch for the whole batch.
    /// `q [B, H, C, D]` f32 contiguous (C = 1 for decode, the chunk for
    /// prefill), `tables [B, maxBlocks]` u32, `ctxlens [B]` u32
    /// (post-run frontier; the kernel derives per-row causal lengths
    /// from `advances [B]`), `block_bases [B]` u32. `output` must be
    /// contiguous f32 `[B, H, C, D]`. Allocates nothing; requires the
    /// exact attention pipeline to be warm.
    #[allow(clippy::too_many_arguments)]
    fn attention_into_mode(
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
        advances: &MetalTensor,
        mode: super::AttentionMode,
        output: &MetalTensor,
        resources: IntoResources<'_>,
    ) -> crate::err::Res<()> {
        let (c, d) = (q.layout.shape()[2], q.layout.shape()[3]);
        let kernel = if uses_mma_prefill(mode, c, d, k_slab.dtype) && block_size % 8 == 0 {
            PrefillKernel::Mma
        } else if uses_tiled_prefill(mode, c) {
            PrefillKernel::Tiled(PREFILL_QT)
        } else {
            PrefillKernel::Rows
        };
        attention_dispatch(
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
            advances,
            mode,
            kernel,
            output,
            resources,
        )
    }

    /// Non-allocating row-parallel paged attention with no prefill
    /// routing (test and probe reference for the tiled kernel).
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_rows_into(
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
        advances: &MetalTensor,
        output: &MetalTensor,
        resources: IntoResources<'_>,
    ) -> crate::err::Res<()> {
        attention_dispatch(
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
            advances,
            super::AttentionMode::Causal,
            PrefillKernel::Rows,
            output,
            resources,
        )
    }

    /// Non-allocating query-tiled causal prefill attention with an
    /// explicit tile size: identical contract to [`attention_into`],
    /// but always dispatches `et_paged_prefill` with `qt` query rows
    /// per threadgroup tile (any chunk, including small ones, so tests
    /// and probes can exercise the tiled path directly).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_prefill_qt_into(
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
        advances: &MetalTensor,
        qt: usize,
        output: &MetalTensor,
        resources: IntoResources<'_>,
    ) -> crate::err::Res<()> {
        attention_dispatch(
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
            advances,
            super::AttentionMode::Causal,
            PrefillKernel::Tiled(qt),
            output,
            resources,
        )
    }

    /// Non-allocating MMA query-tiled causal prefill attention:
    /// identical contract to [`attention_into`], always dispatched
    /// through `et_paged_prefill_mma` regardless of chunk length
    /// (probe and test support).
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_prefill_mma_into(
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
        advances: &MetalTensor,
        output: &MetalTensor,
        resources: IntoResources<'_>,
    ) -> crate::err::Res<()> {
        attention_dispatch(
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
            advances,
            super::AttentionMode::Causal,
            PrefillKernel::Mma,
            output,
            resources,
        )
    }

    /// Non-allocating query-tiled causal prefill attention using the default
    /// tile size `PREFILL_QT`. It follows the [`attention_into`] contract and
    /// always dispatches through `et_paged_prefill`, regardless of chunk length.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill_into(
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
        advances: &MetalTensor,
        output: &MetalTensor,
        resources: IntoResources<'_>,
    ) -> crate::err::Res<()> {
        attention_prefill_qt_into(
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
            advances,
            PREFILL_QT,
            output,
            resources,
        )
    }

    /// Warms the query-tiled causal prefill pipeline with the default
    /// tile size for the given head geometry, slab dtype, softmax
    /// scale, and chunk.
    pub fn warm_attention_prefill(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        slab_dtype: DType,
        scale: f64,
        chunk: usize,
    ) -> crate::err::Res<()> {
        prefill_pipeline(
            d,
            query_heads,
            kv_heads,
            slab_dtype,
            scale,
            chunk,
            PREFILL_QT,
        )?;
        Ok(())
    }

    /// Warms the query-tiled causal prefill pipeline with an explicit
    /// tile size (probe and test support).
    #[allow(dead_code)]
    pub(crate) fn warm_attention_prefill_qt(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        slab_dtype: DType,
        scale: f64,
        chunk: usize,
        qt: usize,
    ) -> crate::err::Res<()> {
        prefill_pipeline(d, query_heads, kv_heads, slab_dtype, scale, chunk, qt)?;
        Ok(())
    }

    /// Warms the MMA query-tiled causal prefill pipeline (probe and
    /// test support).
    #[allow(dead_code)]
    pub(crate) fn warm_attention_prefill_mma(
        d: usize,
        query_heads: usize,
        kv_heads: usize,
        slab_dtype: DType,
        scale: f64,
        chunk: usize,
    ) -> crate::err::Res<()> {
        prefill_mma_pipeline(d, query_heads, kv_heads, slab_dtype, scale, chunk)?;
        Ok(())
    }

    /// Non-allocating causal paged attention. This preserves the original
    /// per-row visibility semantics used by decode and prefill callers.
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
        advances: &MetalTensor,
        output: &MetalTensor,
        resources: IntoResources<'_>,
    ) -> crate::err::Res<()> {
        attention_into_mode(
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
            advances,
            super::AttentionMode::Causal,
            output,
            resources,
        )
    }

    /// Non-allocating block-bidirectional paged attention. `ctxlens` is the
    /// post-scatter frontier and `advances` is the real current-block length;
    /// every query sees all `advances` current rows. If `window` is set, it
    /// retains that many committed rows plus the entire current block.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_block_into(
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
        advances: &MetalTensor,
        output: &MetalTensor,
        resources: IntoResources<'_>,
    ) -> crate::err::Res<()> {
        attention_into_mode(
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
            advances,
            super::AttentionMode::BidirectionalBlock,
            output,
            resources,
        )
    }

    /// Non-allocating flash-decoding split paged attention for causal
    /// one-token decode (`q [B, H, 1, D]`): a partial launch over
    /// `splits` deterministic context subranges writes unnormalized
    /// records into `resources.scratch[0]` (exactly `[B, H, splits,
    /// D + 2]` f32 contiguous), then a combine launch writes the
    /// normalized `output [B, H, 1, D]`. Allocates nothing; requires
    /// the exact split pipelines to be warm. The two dispatches run on
    /// the same stream, whose per-dispatch buffer barrier orders the
    /// combine after the partial.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_split_into(
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
        advances: &MetalTensor,
        splits: usize,
        output: &MetalTensor,
        resources: IntoResources<'_>,
    ) -> crate::err::Res<()> {
        let (b, h, c, d) = (
            q.layout.shape()[0],
            q.layout.shape()[1],
            q.layout.shape()[2],
            q.layout.shape()[3],
        );
        if splits < 2 {
            return Err(format!(
                "paged attention split: splits must be at least 2, got {splits}"
            ));
        }
        if c != 1 {
            return Err(format!(
                "paged attention split: only one-token decode is supported, got chunk {c}"
            ));
        }
        let kv_heads = slab_heads(k_slab, d)?;
        if kv_heads == 0 || !h.is_multiple_of(kv_heads) {
            return Err("paged attention: query heads must be divisible by K/V heads".to_string());
        }
        let slab_dtype = slab_dtype(k_slab);
        if !resources.staging.is_empty()
            || !resources.status.is_empty()
            || resources.scratch.len() != 1
        {
            return Err(
                "paged attention split: staging and status must be empty, scratch must hold exactly one tensor"
                    .to_string(),
            );
        }
        validate_advances(advances, ctxlens, b, c, "paged attention split")?;
        if !q.layout.is_contiguous() {
            return Err(
                "paged attention split: query must be contiguous before attention_split_into"
                    .to_string(),
            );
        }
        if output.dtype != DType::F32
            || output.layout.shape() != [b, h, c, d]
            || !output.layout.is_contiguous()
        {
            return Err(format!(
                "paged attention split: output must be contiguous [{b}, {h}, {c}, {d}]:F32"
            ));
        }
        let scratch = &resources.scratch[0];
        scratch.validate_destination(
            "paged attention split scratch",
            &[b, h, splits, d + 2],
            DType::F32,
        )?;
        MetalDevice::get().mark_buffer_write(&output.buffer)?;
        let device = MetalDevice::get();
        let partial = device
            .pipeline_cached(split_partial_key(d, h, kv_heads, slab_dtype, scale, splits))
            .ok_or_else(|| {
                "paged attention split: exact pipeline is not warm; call warm_attention_split"
                    .to_string()
            })?;
        let combine = device
            .pipeline_cached(split_combine_key(d, h, splits))
            .ok_or_else(|| {
                "paged attention split: exact pipeline is not warm; call warm_attention_split"
                    .to_string()
            })?;
        let max_blocks = tables.layout.shape()[1];
        let f32_off = |off: usize| off * DType::F32.size_in_bytes();
        let u32_off = |off: usize| off * DType::U32.size_in_bytes();
        let elem_off = |off: usize| off * slab_dtype.size_in_bytes();
        device.with_encoder(|e| {
            e.setComputePipelineState(partial.as_raw());
            set_buffer(e, 0, &q.buffer, f32_off(q.layout.offset()));
            set_buffer(e, 1, &k_slab.buffer, elem_off(k_slab.layout.offset()));
            set_buffer(e, 2, &v_slab.buffer, elem_off(v_slab.layout.offset()));
            set_buffer(e, 3, &tables.buffer, u32_off(tables.layout.offset()));
            set_buffer(e, 4, &ctxlens.buffer, u32_off(ctxlens.layout.offset()));
            set_buffer(e, 5, &scratch.buffer, f32_off(scratch.layout.offset()));
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
            set_buffer(e, 12, &advances.buffer, u32_off(advances.layout.offset()));
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
                    depth: splits,
                },
                objc2_metal::MTLSize {
                    width: ATTENTION_THREADS,
                    height: 1,
                    depth: 1,
                },
            );
        });
        device.with_encoder(|e| {
            e.setComputePipelineState(combine.as_raw());
            set_buffer(e, 0, &scratch.buffer, f32_off(scratch.layout.offset()));
            set_buffer(e, 1, &output.buffer, f32_off(output.layout.offset()));
            e.dispatchThreadgroups_threadsPerThreadgroup(
                objc2_metal::MTLSize {
                    width: h,
                    height: b,
                    depth: 1,
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
        advances: &MetalTensor,
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
            advances,
            output,
            resources,
        )
    }

    /// Allocates output, calls [`decode_into`], and returns the f32
    /// `[B, H, C, D]` attention result.
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
        advances: &MetalTensor,
    ) -> crate::err::Res<MetalTensor> {
        let shape = q.layout.shape().to_vec();
        let q = wrap_contig(q)?;
        let d = q.layout.shape()[3];
        let kv_heads = slab_heads(k_slab, d)?;
        warm_attention(
            d,
            q.layout.shape()[1],
            kv_heads,
            k_slab.dtype,
            scale,
            q.layout.shape()[2],
        )?;
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
            advances,
            &output,
            IntoResources::empty(),
        )?;
        Ok(output)
    }

    /// Allocates outputs and calls [`attention_block_into`].
    #[allow(clippy::too_many_arguments)]
    pub fn attention_block(
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
        advances: &MetalTensor,
    ) -> crate::err::Res<MetalTensor> {
        let shape = q.layout.shape().to_vec();
        let q = wrap_contig(q)?;
        let d = q.layout.shape()[3];
        let kv_heads = slab_heads(k_slab, d)?;
        warm_attention_block(
            d,
            q.layout.shape()[1],
            kv_heads,
            k_slab.dtype,
            scale,
            q.layout.shape()[2],
        )?;
        let output = MetalTensor::empty(MetalDevice::get(), shape, DType::F32);
        attention_block_into(
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
            advances,
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
    use objc2_metal::MTLComputeCommandEncoder;

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

    fn dense_bidirectional(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        dimension: usize,
        scale: f32,
        start: usize,
    ) -> Vec<f32> {
        let rows = k.len() / dimension;
        let mut output = Vec::with_capacity(q.len());
        for query in q.chunks_exact(dimension) {
            let scores: Vec<f32> = (start..rows)
                .map(|row| {
                    query
                        .iter()
                        .zip(&k[row * dimension..(row + 1) * dimension])
                        .map(|(q, k)| q * k)
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|score| (score - max).exp()).collect();
            let denominator: f32 = weights.iter().sum();
            for column in 0..dimension {
                output.push(
                    weights
                        .iter()
                        .enumerate()
                        .map(|(index, weight)| weight * v[(start + index) * dimension + column])
                        .sum::<f32>()
                        / denominator,
                );
            }
        }
        output
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
        let advances = u32_tensor(dev, &[1], vec![b]);
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
            &advances,
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
            &advances,
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
            &advances,
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
            &advances,
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
        let advances = u32_tensor(dev, &[1], vec![batch]);
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
            &advances,
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
            &advances,
        )
        .unwrap();
        dev.synchronize().unwrap();
        assert_eq!(
            output.read_f32().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0,]
        );
    }

    #[test]
    fn paged_kernels_apply_each_physical_lanes_advance() {
        let dev = MetalDevice::get();
        let (batch, heads, chunk, dimension) = (2usize, 1usize, 2usize, 4usize);
        let q = MT::from_f32(
            dev,
            vec![0.0; batch * heads * chunk * dimension],
            vec![batch, heads, chunk, dimension],
        );
        let k_new = MT::from_f32(
            dev,
            vec![0.0; batch * heads * chunk * dimension],
            vec![batch, heads, chunk, dimension],
        );
        let v_new = MT::from_f32(
            dev,
            [1.0, 99.0, 2.0, 4.0]
                .into_iter()
                .flat_map(|value| [value; 4])
                .collect(),
            vec![batch, heads, chunk, dimension],
        );
        let k_slab = MT::from_f32(dev, vec![0.0; 8 * dimension], vec![8, heads, dimension]);
        let v_slab = MT::from_f32(dev, vec![0.0; 8 * dimension], vec![8, heads, dimension]);
        let tables = u32_tensor(dev, &[0, 1], vec![batch, 1]);
        let ctxlens = u32_tensor(dev, &[1, 2], vec![batch]);
        let block_bases = u32_tensor(dev, &[0, 0], vec![batch]);
        let advances = u32_tensor(dev, &[1, 2], vec![batch]);

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
            4,
            &advances,
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
            4,
            &advances,
        )
        .unwrap();
        dev.synchronize().unwrap();
        assert_eq!(
            output.read_f32().unwrap(),
            [1.0, 1.0, 2.0, 3.0]
                .into_iter()
                .flat_map(|value| [value; 4])
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn current_block_is_bidirectional_and_window_keeps_committed_rows() {
        let dev = MetalDevice::get();
        let (batch, heads, chunk, dimension) = (1usize, 1usize, 3usize, 4usize);
        let scale = 0.5f64;
        let q_values = vec![
            1.0, 0.0, 0.5, -0.25, 0.0, 1.0, -0.5, 0.25, 0.3, -0.2, 0.7, 1.0,
        ];
        let committed_k = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let current_k = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5];
        let committed_v = vec![1.0, 2.0, 3.0, 4.0, -2.0, 1.0, 0.5, 3.0];
        let current_v = vec![
            4.0, -1.0, 2.0, 0.0, 0.5, 3.0, -2.0, 1.0, 5.0, 2.0, 1.0, -3.0,
        ];
        let q = MT::from_f32(dev, q_values.clone(), vec![batch, heads, chunk, dimension]);
        let k_new = MT::from_f32(dev, current_k.clone(), vec![batch, heads, chunk, dimension]);
        let v_new = MT::from_f32(dev, current_v.clone(), vec![batch, heads, chunk, dimension]);
        let mut initial_k = committed_k.clone();
        initial_k.resize(8 * dimension, 0.0);
        let mut initial_v = committed_v.clone();
        initial_v.resize(8 * dimension, 0.0);
        let k_slab = MT::from_f32(dev, initial_k, vec![8, heads, dimension]);
        let v_slab = MT::from_f32(dev, initial_v, vec![8, heads, dimension]);
        let tables = u32_tensor(dev, &[0, 1], vec![batch, 2]);
        let ctxlens = u32_tensor(dev, &[5], vec![batch]);
        let block_bases = u32_tensor(dev, &[0], vec![batch]);
        let advances = u32_tensor(dev, &[3], vec![batch]);

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
            4,
            &advances,
        )
        .unwrap();
        let requirements =
            super::attention_requirements(DType::F32, batch, heads, heads, chunk, dimension, scale)
                .unwrap();
        super::metal::warm_attention_block_exact(&requirements).unwrap();
        let block_output = MT::empty(dev, vec![batch, heads, chunk, dimension], DType::F32);
        super::metal::attention_block_into(
            &q,
            &k_slab,
            &v_slab,
            None,
            None,
            &tables,
            &ctxlens,
            &block_bases,
            None,
            scale,
            4,
            &advances,
            &block_output,
            super::metal::IntoResources::empty(),
        )
        .unwrap();
        let causal_output = super::metal::decode(
            &q,
            &k_slab,
            &v_slab,
            None,
            None,
            &tables,
            &ctxlens,
            &block_bases,
            None,
            scale,
            4,
            &advances,
        )
        .unwrap();
        let window_output = super::metal::attention_block(
            &q,
            &k_slab,
            &v_slab,
            None,
            None,
            &tables,
            &ctxlens,
            &block_bases,
            Some(1),
            scale,
            4,
            &advances,
        )
        .unwrap();
        dev.synchronize().unwrap();

        let logical_k = [committed_k, current_k].concat();
        let logical_v = [committed_v, current_v].concat();
        let dense = dense_bidirectional(
            &q_values,
            &logical_k,
            &logical_v,
            dimension,
            scale as f32,
            0,
        );
        let dense_window = dense_bidirectional(
            &q_values,
            &logical_k,
            &logical_v,
            dimension,
            scale as f32,
            1,
        );
        let actual = block_output.read_f32().unwrap();
        let actual_window = window_output.read_f32().unwrap();
        assert!(actual
            .iter()
            .zip(&dense)
            .all(|(actual, expected)| (actual - expected).abs() < 2e-5));
        assert!(actual_window
            .iter()
            .zip(&dense_window)
            .all(|(actual, expected)| (actual - expected).abs() < 2e-5));
        assert!(max_diff(&block_output, &causal_output) > 1e-2);
    }

    #[test]
    fn split_decode_matches_row_parallel_decode() {
        let dev = MetalDevice::get();
        let (batch, query_heads, kv_heads, dimension) = (2usize, 4usize, 2usize, 64usize);
        let block_size = 16usize;
        let capacity = 520usize;
        let max_blocks = 40usize;
        let pool_rows = batch * max_blocks * block_size;
        let scale = 1.0 / (dimension as f64).sqrt();
        let mut seed = 0x1234_5678u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((seed >> 8) as f32 / (1 << 24) as f32) - 0.5
        };
        let q = MT::from_f32(
            dev,
            (0..batch * query_heads * dimension)
                .map(|_| next())
                .collect(),
            vec![batch, query_heads, 1, dimension],
        );
        let k_chunk = MT::from_f32(
            dev,
            (0..batch * kv_heads * capacity * dimension)
                .map(|_| next())
                .collect(),
            vec![batch, kv_heads, capacity, dimension],
        );
        let v_chunk = MT::from_f32(
            dev,
            (0..batch * kv_heads * capacity * dimension)
                .map(|_| next())
                .collect(),
            vec![batch, kv_heads, capacity, dimension],
        );
        let tables = u32_tensor(
            dev,
            &(0..batch * max_blocks)
                .map(|index| index as u32)
                .collect::<Vec<_>>(),
            vec![batch, max_blocks],
        );
        let block_bases = u32_tensor(dev, &[0, 0], vec![batch]);
        let full_ctxlens = u32_tensor(dev, &[capacity as u32, capacity as u32], vec![batch]);
        let full_advances = u32_tensor(dev, &[capacity as u32, capacity as u32], vec![batch]);

        for slab_dtype in [DType::F32, DType::F16, DType::BF16, DType::U8] {
            let k_slab = MT::empty(dev, vec![pool_rows, kv_heads, dimension], slab_dtype);
            let v_slab = MT::empty(dev, vec![pool_rows, kv_heads, dimension], slab_dtype);
            let (k_scales, v_scales) = if slab_dtype == DType::U8 {
                (
                    Some(MT::empty(dev, vec![pool_rows, kv_heads], DType::F32)),
                    Some(MT::empty(dev, vec![pool_rows, kv_heads], DType::F32)),
                )
            } else {
                (None, None)
            };
            super::metal::scatter(
                &k_chunk,
                &v_chunk,
                &k_slab,
                &v_slab,
                k_scales.as_ref(),
                v_scales.as_ref(),
                &tables,
                &full_ctxlens,
                &block_bases,
                block_size,
                &full_advances,
            )
            .unwrap();
            let tolerance = match slab_dtype {
                DType::F32 => 2e-5,
                DType::F16 => 2e-3,
                _ => 2e-2,
            };
            for (first, second) in [
                (0usize, 0usize),
                (1, 0),
                (3, 1),
                (17, 9),
                (100, 37),
                (513, 260),
            ] {
                let ctxlens = u32_tensor(dev, &[first as u32, second as u32], vec![batch]);
                let advances = u32_tensor(
                    dev,
                    &[
                        u32::from(first > 0).min(first as u32),
                        u32::from(second > 0).min(second as u32),
                    ],
                    vec![batch],
                );
                for window in [None, Some(30)] {
                    let reference = super::metal::decode(
                        &q,
                        &k_slab,
                        &v_slab,
                        k_scales.as_ref(),
                        v_scales.as_ref(),
                        &tables,
                        &ctxlens,
                        &block_bases,
                        window,
                        scale,
                        block_size,
                        &advances,
                    )
                    .unwrap();
                    for splits in [2usize, 8] {
                        super::metal::warm_attention_split(
                            dimension,
                            query_heads,
                            kv_heads,
                            slab_dtype,
                            scale,
                            splits,
                        )
                        .unwrap();
                        let scratch = MT::empty(
                            dev,
                            vec![batch, query_heads, splits, dimension + 2],
                            DType::F32,
                        );
                        let actual =
                            MT::empty(dev, vec![batch, query_heads, 1, dimension], DType::F32);
                        super::metal::attention_split_into(
                            &q,
                            &k_slab,
                            &v_slab,
                            k_scales.as_ref(),
                            v_scales.as_ref(),
                            &tables,
                            &ctxlens,
                            &block_bases,
                            window,
                            scale,
                            block_size,
                            &advances,
                            splits,
                            &actual,
                            super::metal::IntoResources {
                                staging: &[],
                                status: &[],
                                scratch: std::slice::from_ref(&scratch),
                            },
                        )
                        .unwrap();
                        dev.synchronize().unwrap();
                        let diff = max_diff(&reference, &actual);
                        assert!(
                            diff < tolerance,
                            "split decode mismatch: dtype {slab_dtype:?} ctx [{first}, {second}] window {window:?} splits {splits}: {diff}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn split_decode_matches_row_parallel_decode_d128() {
        let dev = MetalDevice::get();
        let (batch, query_heads, kv_heads, dimension) = (1usize, 4usize, 2usize, 128usize);
        let block_size = 16usize;
        let capacity = 320usize;
        let max_blocks = 20usize;
        let pool_rows = batch * max_blocks * block_size;
        let scale = 1.0 / (dimension as f64).sqrt();
        let mut seed = 0x8765_4321u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((seed >> 8) as f32 / (1 << 24) as f32) - 0.5
        };
        let q = MT::from_f32(
            dev,
            (0..batch * query_heads * dimension)
                .map(|_| next())
                .collect(),
            vec![batch, query_heads, 1, dimension],
        );
        let k_chunk = MT::from_f32(
            dev,
            (0..batch * kv_heads * capacity * dimension)
                .map(|_| next())
                .collect(),
            vec![batch, kv_heads, capacity, dimension],
        );
        let v_chunk = MT::from_f32(
            dev,
            (0..batch * kv_heads * capacity * dimension)
                .map(|_| next())
                .collect(),
            vec![batch, kv_heads, capacity, dimension],
        );
        let tables = u32_tensor(
            dev,
            &(0..batch * max_blocks)
                .map(|index| index as u32)
                .collect::<Vec<_>>(),
            vec![batch, max_blocks],
        );
        let block_bases = u32_tensor(dev, &[0], vec![batch]);
        let full_ctxlens = u32_tensor(dev, &[capacity as u32], vec![batch]);
        let full_advances = u32_tensor(dev, &[capacity as u32], vec![batch]);
        let k_slab = MT::empty(dev, vec![pool_rows, kv_heads, dimension], DType::F16);
        let v_slab = MT::empty(dev, vec![pool_rows, kv_heads, dimension], DType::F16);
        super::metal::scatter(
            &k_chunk,
            &v_chunk,
            &k_slab,
            &v_slab,
            None,
            None,
            &tables,
            &full_ctxlens,
            &block_bases,
            block_size,
            &full_advances,
        )
        .unwrap();
        let splits = 8usize;
        super::metal::warm_attention_split(
            dimension,
            query_heads,
            kv_heads,
            DType::F16,
            scale,
            splits,
        )
        .unwrap();
        for context in [0usize, 5, 300] {
            let ctxlens = u32_tensor(dev, &[context as u32], vec![batch]);
            let advances = u32_tensor(
                dev,
                &[u32::from(context > 0).min(context as u32)],
                vec![batch],
            );
            let reference = super::metal::decode(
                &q,
                &k_slab,
                &v_slab,
                None,
                None,
                &tables,
                &ctxlens,
                &block_bases,
                None,
                scale,
                block_size,
                &advances,
            )
            .unwrap();
            let scratch = MT::empty(
                dev,
                vec![batch, query_heads, splits, dimension + 2],
                DType::F32,
            );
            let actual = MT::empty(dev, vec![batch, query_heads, 1, dimension], DType::F32);
            super::metal::attention_split_into(
                &q,
                &k_slab,
                &v_slab,
                None,
                None,
                &tables,
                &ctxlens,
                &block_bases,
                None,
                scale,
                block_size,
                &advances,
                splits,
                &actual,
                super::metal::IntoResources {
                    staging: &[],
                    status: &[],
                    scratch: std::slice::from_ref(&scratch),
                },
            )
            .unwrap();
            dev.synchronize().unwrap();
            let diff = max_diff(&reference, &actual);
            assert!(diff < 2e-3, "split decode d=128 ctx {context}: {diff}");
        }
    }

    #[test]
    #[ignore = "manual long-context paged attention probe"]
    fn paged_decode_bandwidth_probe() {
        let dev = MetalDevice::get();
        let (batch, query_heads, kv_heads, chunk, dimension) =
            (1usize, 32usize, 2usize, 1usize, 128usize);
        let rows = 4096usize;
        let block_size = 16usize;
        let q = MT::from_f32(
            dev,
            vec![0.01; batch * query_heads * chunk * dimension],
            vec![batch, query_heads, chunk, dimension],
        );
        let k_slab = MT::from_f32(
            dev,
            vec![0.001; rows * kv_heads * dimension],
            vec![rows, kv_heads, dimension],
        );
        let v_slab = MT::from_f32(
            dev,
            vec![0.002; rows * kv_heads * dimension],
            vec![rows, kv_heads, dimension],
        );
        let tables = u32_tensor(
            dev,
            &(0..(rows / block_size) as u32).collect::<Vec<_>>(),
            vec![batch, rows / block_size],
        );
        let block_bases = u32_tensor(dev, &[0], vec![batch]);
        let advances = u32_tensor(dev, &[1], vec![batch]);
        let output = MT::empty(dev, vec![batch, query_heads, chunk, dimension], DType::F32);
        let requirements = super::attention_requirements(
            DType::F32,
            batch,
            query_heads,
            kv_heads,
            chunk,
            dimension,
            0.08838834764831845,
        )
        .unwrap();
        super::metal::warm_attention_exact(&requirements).unwrap();
        let splits = 8usize;
        super::metal::warm_attention_split(
            dimension,
            query_heads,
            kv_heads,
            DType::F32,
            0.08838834764831845,
            splits,
        )
        .unwrap();
        let scratch = MT::empty(
            dev,
            vec![batch, query_heads, splits, dimension + 2],
            DType::F32,
        );
        for context in [64usize, 512, 1024, 2048, 3072, 4096] {
            let ctxlens = u32_tensor(dev, &[context as u32], vec![batch]);
            for _ in 0..5 {
                super::metal::decode_into(
                    &q,
                    &k_slab,
                    &v_slab,
                    None,
                    None,
                    &tables,
                    &ctxlens,
                    &block_bases,
                    None,
                    0.08838834764831845,
                    block_size,
                    &advances,
                    &output,
                    super::metal::IntoResources::empty(),
                )
                .unwrap();
            }
            dev.synchronize().unwrap();
            let started = std::time::Instant::now();
            for _ in 0..50 {
                super::metal::decode_into(
                    &q,
                    &k_slab,
                    &v_slab,
                    None,
                    None,
                    &tables,
                    &ctxlens,
                    &block_bases,
                    None,
                    0.08838834764831845,
                    block_size,
                    &advances,
                    &output,
                    super::metal::IntoResources::empty(),
                )
                .unwrap();
            }
            dev.synchronize().unwrap();
            let ms = started.elapsed().as_secs_f64() * 1000.0 / 50.0;
            let bytes = context * kv_heads * dimension * 2 * DType::F32.size_in_bytes();
            eprintln!(
                "paged decode ctx={context}: {ms:.3} ms/layer, {:.1} GB/s logical KV",
                bytes as f64 / (ms / 1000.0) / 1e9
            );
            for _ in 0..5 {
                super::metal::attention_split_into(
                    &q,
                    &k_slab,
                    &v_slab,
                    None,
                    None,
                    &tables,
                    &ctxlens,
                    &block_bases,
                    None,
                    0.08838834764831845,
                    block_size,
                    &advances,
                    splits,
                    &output,
                    super::metal::IntoResources {
                        staging: &[],
                        status: &[],
                        scratch: std::slice::from_ref(&scratch),
                    },
                )
                .unwrap();
            }
            dev.synchronize().unwrap();
            let started = std::time::Instant::now();
            for _ in 0..50 {
                super::metal::attention_split_into(
                    &q,
                    &k_slab,
                    &v_slab,
                    None,
                    None,
                    &tables,
                    &ctxlens,
                    &block_bases,
                    None,
                    0.08838834764831845,
                    block_size,
                    &advances,
                    splits,
                    &output,
                    super::metal::IntoResources {
                        staging: &[],
                        status: &[],
                        scratch: std::slice::from_ref(&scratch),
                    },
                )
                .unwrap();
            }
            dev.synchronize().unwrap();
            let ms = started.elapsed().as_secs_f64() * 1000.0 / 50.0;
            eprintln!(
                "paged decode split(8) ctx={context}: {ms:.3} ms/layer, {:.1} GB/s logical KV",
                bytes as f64 / (ms / 1000.0) / 1e9
            );
        }
    }

    #[test]
    fn simdgroup_float8x8_layout_probe() {
        // The MMA prefill kernel rescales and stores accumulator tiles
        // per element, relying on the 8x8 simdgroup fragment layout.
        // Verify lane t owns row (t/16)*4 + (t%8)/2 and columns
        // ((t%16)/8)*4 + (t%2)*2 + {0,1} (both slots in one row).
        let dev = MetalDevice::get();
        let source = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void et_layout_probe(
    device float* out [[buffer(0)]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    simdgroup_float8x8 m;
    m.thread_elements()[0] = float(lane * 10 + 0);
    m.thread_elements()[1] = float(lane * 10 + 1);
    simdgroup_store(m, out, 8);
}
"#;
        dev.compile_lazy(0x1A907u64, "et_layout_probe", || source.to_string())
            .unwrap();
        let pipe = dev.pipeline_cached(0x1A907u64).unwrap();
        let output = MT::empty(dev, vec![64], DType::F32);
        dev.with_encoder(|e| {
            e.setComputePipelineState(pipe.as_raw());
            crate::runtime::metal::device::set_buffer(e, 0, &output.buffer, 0);
            e.dispatchThreadgroups_threadsPerThreadgroup(
                objc2_metal::MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
        });
        dev.synchronize().unwrap();
        let values = output.read_f32().unwrap();
        // The 8x8 fragment layout this kernel family relies on: lane t
        // owns row (t/16)*4 + (t%8)/2 and columns ((t%16)/8)*4 +
        // (t%2)*2 + {0,1} (both slots in the same row, so per-row
        // online-softmax rescales are lane-uniform).
        for r in 0..8usize {
            for c in 0..8usize {
                let slot = c % 2;
                let lane = (r % 4) * 2 + (r / 4) * 16 + ((c % 4) / 2) + (c / 4) * 8;
                assert_eq!(
                    values[r * 8 + c],
                    (lane * 10 + slot) as f32,
                    "fragment layout mismatch at ({r}, {c})"
                );
            }
        }
    }

    #[test]
    fn tiled_prefill_matches_row_parallel() {
        let dev = MetalDevice::get();
        let (batch, block_size, capacity, max_blocks) = (2usize, 16usize, 300usize, 19usize);
        let pool_rows = batch * max_blocks * block_size;
        let mut seed = 0x0F0F_0F0Fu32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((seed >> 8) as f32 / (1 << 24) as f32) - 0.5
        };
        for (query_heads, kv_heads, dimension) in [
            (4usize, 4usize, 64usize),
            (4, 2, 64),
            (8, 1, 64),
            (4, 2, 128),
        ] {
            let scale = 1.0 / (dimension as f64).sqrt();
            let k_chunk = MT::from_f32(
                dev,
                (0..batch * kv_heads * capacity * dimension)
                    .map(|_| next())
                    .collect(),
                vec![batch, kv_heads, capacity, dimension],
            );
            let v_chunk = MT::from_f32(
                dev,
                (0..batch * kv_heads * capacity * dimension)
                    .map(|_| next())
                    .collect(),
                vec![batch, kv_heads, capacity, dimension],
            );
            let tables = u32_tensor(
                dev,
                &(0..batch * max_blocks)
                    .map(|index| index as u32)
                    .collect::<Vec<_>>(),
                vec![batch, max_blocks],
            );
            let block_bases = u32_tensor(dev, &[0, 0], vec![batch]);
            let full_ctxlens = u32_tensor(dev, &[capacity as u32, capacity as u32], vec![batch]);
            let full_advances = u32_tensor(dev, &[capacity as u32, capacity as u32], vec![batch]);
            for slab_dtype in [DType::F32, DType::F16, DType::BF16, DType::U8] {
                if dimension == 128 && slab_dtype != DType::F16 {
                    continue; // keep the d=128 sweep small; it exercises the load4 path
                }
                let k_slab = MT::empty(dev, vec![pool_rows, kv_heads, dimension], slab_dtype);
                let v_slab = MT::empty(dev, vec![pool_rows, kv_heads, dimension], slab_dtype);
                let (k_scales, v_scales) = if slab_dtype == DType::U8 {
                    (
                        Some(MT::empty(dev, vec![pool_rows, kv_heads], DType::F32)),
                        Some(MT::empty(dev, vec![pool_rows, kv_heads], DType::F32)),
                    )
                } else {
                    (None, None)
                };
                super::metal::scatter(
                    &k_chunk,
                    &v_chunk,
                    &k_slab,
                    &v_slab,
                    k_scales.as_ref(),
                    v_scales.as_ref(),
                    &tables,
                    &full_ctxlens,
                    &block_bases,
                    block_size,
                    &full_advances,
                )
                .unwrap();
                let tolerance = match slab_dtype {
                    DType::F32 => 2e-5,
                    DType::F16 => 2e-3,
                    _ => 2e-2,
                };
                for chunk in [2usize, 5, 8, 16, 17, 33] {
                    let q = MT::from_f32(
                        dev,
                        (0..batch * query_heads * chunk * dimension)
                            .map(|_| next())
                            .collect(),
                        vec![batch, query_heads, chunk, dimension],
                    );
                    super::metal::warm_attention_rows(
                        dimension,
                        query_heads,
                        kv_heads,
                        slab_dtype,
                        scale,
                        chunk,
                    )
                    .unwrap();
                    // Auto-routed warm + dispatch: row-parallel below the
                    // tiling threshold, tiled at/above it.
                    super::metal::warm_attention(
                        dimension,
                        query_heads,
                        kv_heads,
                        slab_dtype,
                        scale,
                        chunk,
                    )
                    .unwrap();
                    super::metal::warm_attention_prefill(
                        dimension,
                        query_heads,
                        kv_heads,
                        slab_dtype,
                        scale,
                        chunk,
                    )
                    .unwrap();
                    if slab_dtype == DType::F16 {
                        super::metal::warm_attention_prefill_mma(
                            dimension,
                            query_heads,
                            kv_heads,
                            slab_dtype,
                            scale,
                            chunk,
                        )
                        .unwrap();
                    }
                    // (prefix lane 0, prefix lane 1, advance); advance
                    // below the chunk exercises pad-row masking, zero
                    // advance the zero-fill path.
                    for (first, second, advance) in [
                        (0usize, 0usize, chunk),
                        (1, 5, chunk),
                        (37, 260, chunk),
                        (260, 41, chunk),
                        (9, 100, 1),
                        (17, 60, (chunk / 3).max(1)),
                        (3, 40, 0),
                    ] {
                        let ctxlens = u32_tensor(
                            dev,
                            &[(first + advance) as u32, (second + advance) as u32],
                            vec![batch],
                        );
                        let advances =
                            u32_tensor(dev, &[advance as u32, advance as u32], vec![batch]);
                        for window in [None, Some(30)] {
                            let reference = MT::empty(
                                dev,
                                vec![batch, query_heads, chunk, dimension],
                                DType::F32,
                            );
                            super::metal::attention_rows_into(
                                &q,
                                &k_slab,
                                &v_slab,
                                k_scales.as_ref(),
                                v_scales.as_ref(),
                                &tables,
                                &ctxlens,
                                &block_bases,
                                window,
                                scale,
                                block_size,
                                &advances,
                                &reference,
                                super::metal::IntoResources::empty(),
                            )
                            .unwrap();
                            let tiled = MT::empty(
                                dev,
                                vec![batch, query_heads, chunk, dimension],
                                DType::F32,
                            );
                            super::metal::attention_prefill_into(
                                &q,
                                &k_slab,
                                &v_slab,
                                k_scales.as_ref(),
                                v_scales.as_ref(),
                                &tables,
                                &ctxlens,
                                &block_bases,
                                window,
                                scale,
                                block_size,
                                &advances,
                                &tiled,
                                super::metal::IntoResources::empty(),
                            )
                            .unwrap();
                            let routed = MT::empty(
                                dev,
                                vec![batch, query_heads, chunk, dimension],
                                DType::F32,
                            );
                            super::metal::attention_into(
                                &q,
                                &k_slab,
                                &v_slab,
                                k_scales.as_ref(),
                                v_scales.as_ref(),
                                &tables,
                                &ctxlens,
                                &block_bases,
                                window,
                                scale,
                                block_size,
                                &advances,
                                &routed,
                                super::metal::IntoResources::empty(),
                            )
                            .unwrap();
                            let mma = MT::empty(
                                dev,
                                vec![batch, query_heads, chunk, dimension],
                                DType::F32,
                            );
                            if slab_dtype == DType::F16 {
                                super::metal::attention_prefill_mma_into(
                                    &q,
                                    &k_slab,
                                    &v_slab,
                                    k_scales.as_ref(),
                                    v_scales.as_ref(),
                                    &tables,
                                    &ctxlens,
                                    &block_bases,
                                    window,
                                    scale,
                                    block_size,
                                    &advances,
                                    &mma,
                                    super::metal::IntoResources::empty(),
                                )
                                .unwrap();
                            }
                            dev.synchronize().unwrap();
                            let tiled_diff = max_diff(&reference, &tiled);
                            let routed_diff = max_diff(&reference, &routed);
                            let mma_diff = if slab_dtype == DType::F16 {
                                max_diff(&reference, &mma)
                            } else {
                                0.0
                            };
                            assert!(
                                tiled_diff < tolerance,
                                "tiled prefill mismatch: heads {query_heads}/{kv_heads} d {dimension} dtype {slab_dtype:?} chunk {chunk} ctx [{first}, {second}] window {window:?}: {tiled_diff}"
                            );
                            assert!(
                                routed_diff < tolerance,
                                "routed prefill mismatch: heads {query_heads}/{kv_heads} d {dimension} dtype {slab_dtype:?} chunk {chunk} ctx [{first}, {second}] window {window:?}: {routed_diff}"
                            );
                            assert!(
                                mma_diff < tolerance,
                                "mma prefill mismatch: heads {query_heads}/{kv_heads} d {dimension} dtype {slab_dtype:?} chunk {chunk} ctx [{first}, {second}] window {window:?}: {mma_diff}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "manual query-tiled prefill attention probe"]
    fn paged_prefill_tiled_probe() {
        let dev = MetalDevice::get();
        let (batch, query_heads, kv_heads, chunk, dimension) =
            (1usize, 32usize, 2usize, 256usize, 128usize);
        let block_size = 16usize;
        let capacity = 8192usize;
        let max_blocks = capacity / block_size;
        let pool_rows = batch * max_blocks * block_size;
        let slab_dtype = DType::F16;
        let scale = 0.08838834764831845f64;
        let q = MT::from_f32(
            dev,
            vec![0.01; batch * query_heads * chunk * dimension],
            vec![batch, query_heads, chunk, dimension],
        );
        let k_new = MT::from_f32(
            dev,
            vec![0.001; batch * kv_heads * capacity * dimension],
            vec![batch, kv_heads, capacity, dimension],
        );
        let v_new = MT::from_f32(
            dev,
            vec![0.002; batch * kv_heads * capacity * dimension],
            vec![batch, kv_heads, capacity, dimension],
        );
        let k_slab = MT::empty(dev, vec![pool_rows, kv_heads, dimension], slab_dtype);
        let v_slab = MT::empty(dev, vec![pool_rows, kv_heads, dimension], slab_dtype);
        let tables = u32_tensor(
            dev,
            &(0..max_blocks as u32).collect::<Vec<_>>(),
            vec![batch, max_blocks],
        );
        let block_bases = u32_tensor(dev, &[0], vec![batch]);
        let full_ctxlens = u32_tensor(dev, &[capacity as u32], vec![batch]);
        let full_advances = u32_tensor(dev, &[capacity as u32], vec![batch]);
        super::metal::scatter(
            &k_new,
            &v_new,
            &k_slab,
            &v_slab,
            None,
            None,
            &tables,
            &full_ctxlens,
            &block_bases,
            block_size,
            &full_advances,
        )
        .unwrap();
        super::metal::warm_attention_rows(
            dimension,
            query_heads,
            kv_heads,
            slab_dtype,
            scale,
            chunk,
        )
        .unwrap();
        for qt in [8usize, 16, 32] {
            super::metal::warm_attention_prefill_qt(
                dimension,
                query_heads,
                kv_heads,
                slab_dtype,
                scale,
                chunk,
                qt,
            )
            .unwrap();
        }
        super::metal::warm_attention_prefill_mma(
            dimension,
            query_heads,
            kv_heads,
            slab_dtype,
            scale,
            chunk,
        )
        .unwrap();
        let advances = u32_tensor(dev, &[chunk as u32], vec![batch]);
        let output = MT::empty(dev, vec![batch, query_heads, chunk, dimension], DType::F32);
        let bytes = |context: usize| {
            (context * kv_heads * dimension * 2 * slab_dtype.size_in_bytes()) as f64
        };
        for context in [1024usize, 3072, 8192] {
            let ctxlens = u32_tensor(dev, &[context as u32], vec![batch]);
            let mut report = format!("prefill ctx={context} chunk={chunk}:");
            let variants: [(&str, usize); 5] = [
                ("rows", 0),
                ("qt8", 8),
                ("qt16", 16),
                ("qt32", 32),
                ("mma", usize::MAX),
            ];
            for (label, variant) in variants {
                let run = |q: &MT, output: &MT| {
                    if variant == 0 {
                        super::metal::attention_rows_into(
                            q,
                            &k_slab,
                            &v_slab,
                            None,
                            None,
                            &tables,
                            &ctxlens,
                            &block_bases,
                            None,
                            scale,
                            block_size,
                            &advances,
                            output,
                            super::metal::IntoResources::empty(),
                        )
                        .unwrap();
                    } else if variant == usize::MAX {
                        super::metal::attention_prefill_mma_into(
                            q,
                            &k_slab,
                            &v_slab,
                            None,
                            None,
                            &tables,
                            &ctxlens,
                            &block_bases,
                            None,
                            scale,
                            block_size,
                            &advances,
                            output,
                            super::metal::IntoResources::empty(),
                        )
                        .unwrap();
                    } else {
                        super::metal::attention_prefill_qt_into(
                            q,
                            &k_slab,
                            &v_slab,
                            None,
                            None,
                            &tables,
                            &ctxlens,
                            &block_bases,
                            None,
                            scale,
                            block_size,
                            &advances,
                            variant,
                            output,
                            super::metal::IntoResources::empty(),
                        )
                        .unwrap();
                    }
                };
                for _ in 0..3 {
                    run(&q, &output);
                }
                dev.synchronize().unwrap();
                let iterations = if variant == 0 { 5 } else { 20 };
                let started = std::time::Instant::now();
                for _ in 0..iterations {
                    run(&q, &output);
                }
                dev.synchronize().unwrap();
                let ms = started.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
                let logical_reads = match variant {
                    0 => chunk * query_heads,
                    usize::MAX => chunk.div_ceil(super::metal::PREFILL_MMA_QT) * query_heads,
                    qt => chunk.div_ceil(qt) * query_heads,
                };
                report += &format!(
                    " {label}={ms:.3} ms/layer ({:.1} GB/s logical)",
                    bytes(context) * logical_reads as f64 / (ms / 1000.0) / 1e9
                );
            }
            eprintln!("{report}");
        }
    }
}

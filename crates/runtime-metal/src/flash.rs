//! Flash attention on Metal: a single-kernel forward (tiled, online
//! softmax, the score matrix never materializes) and a chunked-recompute
//! backward (three native passes in a flash-2 structure, so memory stays
//! bounded by one row-dot scratch vector instead of the full [T, S] score matrix).
//! Both are f32/Metal-only execution strategies for the semantic
//! `Tensor.scaledDotProductAttention` node; CPU references remain the
//! numerical oracle in tests.
//!
//! ## Kernel contracts
//!
//! - **Forward** (`et_sdpa_fwd`): one threadgroup of `THREADS` threads
//!   per (query tile of `TILE_Q` rows, batch·head). Score tiles of
//!   `TILE_Q × TILE_K` are computed into threadgroup memory, folded
//!   into the running (max, sum) online softmax, and consumed by the
//!   P·V accumulation in place. Everything shape-dependent (T, S, D,
//!   DV, scale, causal, window, GQA group) is baked in as `#define`s
//!   and keys the pipeline cache. Also writes the per-row f32
//!   logsumexp `L` for the backward's P recomputation. Causal masking
//!   is right-aligned (`OFFSET = S - T`); `window` restricts attention
//!   to `k > q + OFFSET - WINDOW`.
//! - **Backward** (three dispatches, no atomics):
//!   `et_sdpa_bwd_d` computes the row dots `D[row] = ⟨G[row], O[row]⟩`
//!   into f32 scratch; `et_sdpa_bwd_kv` (key-tiled) accumulates dk/dv
//!   in registers across the full query sweep; `et_sdpa_bwd_q`
//!   (query-tiled) accumulates dq. Score tiles are recomputed from
//!   `L` once per tile pair in threadgroup memory and shared by all
//!   four gradients. Grouped-query attention is **not** differentiable
//!   and is rejected at plan time.
//!
//! The `*_into` entry points validate contiguity/shape/dtype, mark
//! destinations written, allocate nothing, and require pipelines
//! pre-warmed via `warm_*_exact`.

/// Query rows per forward threadgroup tile.
const TILE_Q: usize = 16;
/// Key rows per tile streamed through the score buffer.
const TILE_K: usize = 32;
/// Threads per threadgroup for all flash kernels.
const THREADS: usize = 128;

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

/// Planner-facing description of one device buffer a launch needs
/// (shape, dtype, and derived element/byte counts, overflow-checked).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferRequirement {
    /// Logical shape of the buffer.
    pub shape: Vec<usize>,
    /// Element dtype.
    pub dtype: crate::runtime::dtype::DType,
    /// Total element count (product of `shape`).
    pub elements: usize,
    /// Total byte count (`elements.max(1) * dtype.size_in_bytes()`).
    pub bytes: usize,
}

impl BufferRequirement {
    fn new(
        shape: Vec<usize>,
        dtype: crate::runtime::dtype::DType,
        label: &str,
    ) -> crate::err::Res<Self> {
        let elements = shape
            .iter()
            .try_fold(1usize, |total, dimension| total.checked_mul(*dimension));
        let elements = elements.ok_or_else(|| format!("flash: {label} element count overflow"))?;
        let bytes = elements
            .max(1)
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| format!("flash: {label} byte size overflow"))?;
        Ok(Self {
            shape,
            dtype,
            elements,
            bytes,
        })
    }
}

/// Forward dispatch topology: the single tiled online-softmax kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdpaForwardTopology {
    /// One launch, threadgroups over (query tiles, batch·heads).
    Flash {
        query_tile: usize,
        key_tile: usize,
        threads: usize,
        dispatches: usize,
    },
}

/// Planner-facing requirements of a flash-attention forward.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SdpaForwardRequirements {
    /// Attention output (`q_shape` with trailing dim = `value_depth`).
    pub output: BufferRequirement,
    /// Per-row f32 logsumexp (`q_shape` minus the trailing dim),
    /// consumed by the backward.
    pub logsumexp: BufferRequirement,
    /// Selected dispatch topology.
    pub topology: SdpaForwardTopology,
    /// `batch * heads` folded into one grid dim.
    pub batch_heads: usize,
    /// GQA group size (`query_heads / kv_heads`; 1 for MHA).
    pub head_group_size: usize,
    /// Query length `T`.
    pub query_len: usize,
    /// Key/value length `S`.
    pub key_len: usize,
    /// Query/key depth `D`.
    pub query_depth: usize,
    /// Value depth `DV`.
    pub value_depth: usize,
    /// Storage dtype (f32/f16/bf16).
    pub dtype: crate::runtime::dtype::DType,
}

/// Backward dispatch topology: the three-pass flash-2 recompute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdpaBackwardTopology {
    /// `et_sdpa_bwd_d`, then `et_sdpa_bwd_kv`, then `et_sdpa_bwd_q`.
    FlashRecompute {
        query_tile: usize,
        key_tile: usize,
        threads: usize,
        dispatches: usize,
    },
}

/// Planner-facing requirements of a flash-attention backward.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SdpaBackwardRequirements {
    /// `dq` output (same shape/dtype as `q`).
    pub dq: BufferRequirement,
    /// `dk` output (same shape/dtype as `k`).
    pub dk: BufferRequirement,
    /// `dv` output (same shape/dtype as `v`).
    pub dv: BufferRequirement,
    /// f32 row-dot scratch `D[row] = ⟨G[row], O[row]⟩`
    /// (`q_shape` minus the trailing dim), shared by all three passes.
    pub d_vec_scratch: BufferRequirement,
    /// Selected dispatch topology.
    pub topology: SdpaBackwardTopology,
    /// `batch * heads` folded into one grid dim.
    pub batch_heads: usize,
    /// Query length `T`.
    pub query_len: usize,
    /// Key/value length `S`.
    pub key_len: usize,
    /// Query/key depth `D`.
    pub query_depth: usize,
    /// Value depth `DV`.
    pub value_depth: usize,
    /// Storage dtype (f32/f16/bf16).
    pub dtype: crate::runtime::dtype::DType,
}

#[cfg(target_os = "macos")]
pub use metal::{
    backward_fused, backward_fused_into, backward_requirements, forward, forward_into,
    forward_requirements, warm_backward, warm_backward_exact, warm_forward, warm_forward_exact,
};

#[cfg(target_os = "macos")]
mod metal {
    use super::{
        BufferRequirement, SdpaBackwardRequirements, SdpaBackwardTopology, SdpaForwardRequirements,
        SdpaForwardTopology, THREADS, TILE_K, TILE_Q,
    };
    use crate::runtime::metal::device::{set_buffer, MetalDevice, Pipeline};
    use crate::runtime::metal::run::MetalTensor;

    use objc2_metal::MTLComputeCommandEncoder;
    use std::sync::Arc;

    fn contig(t: &MetalTensor) -> crate::err::Res<MetalTensor> {
        if t.layout.is_contiguous() {
            Ok(t.clone())
        } else {
            crate::runtime::metal::kernels::strided_copy(MetalDevice::get(), t)
        }
    }

    fn alloc(
        elements: usize,
        dtype: crate::runtime::dtype::DType,
    ) -> Arc<crate::runtime::metal::device::Buffer> {
        #[cfg(test)]
        super::record_device_allocation();
        MetalDevice::get().alloc(elements.max(1), dtype)
    }

    fn supported_dtype(dtype: crate::runtime::dtype::DType) -> crate::err::Res<()> {
        if matches!(
            dtype,
            crate::runtime::dtype::DType::F32
                | crate::runtime::dtype::DType::F16
                | crate::runtime::dtype::DType::BF16
        ) {
            Ok(())
        } else {
            Err(format!("flash: unsupported dtype {dtype:?}"))
        }
    }

    fn geometry(
        q_shape: &[usize],
        k_shape: &[usize],
        v_shape: &[usize],
        dtype: crate::runtime::dtype::DType,
    ) -> crate::err::Res<(usize, usize, usize, usize, usize, usize)> {
        supported_dtype(dtype)?;
        let rank = q_shape.len();
        if rank < 2 || k_shape.len() != rank || v_shape.len() != rank {
            return Err(format!(
                "flash: q/k/v must have the same rank >= 2, got {q_shape:?}, {k_shape:?}, {v_shape:?}"
            ));
        }
        let leading = if rank < 3 { rank - 2 } else { rank - 3 };
        if q_shape[..leading] != k_shape[..leading] || q_shape[..leading] != v_shape[..leading] {
            return Err("flash: q/k/v leading dimensions must match".to_string());
        }
        let group = if rank < 3 {
            1
        } else {
            let query_heads = q_shape[rank - 3];
            let kv_heads = k_shape[rank - 3];
            if v_shape[rank - 3] != kv_heads
                || kv_heads == 0
                || !query_heads.is_multiple_of(kv_heads)
            {
                return Err(
                    "flash: query heads must be divisible by matching K/V heads".to_string()
                );
            }
            query_heads / kv_heads
        };
        let (t, s, d, kd, vs, dv) = (
            q_shape[rank - 2],
            k_shape[rank - 2],
            q_shape[rank - 1],
            k_shape[rank - 1],
            v_shape[rank - 2],
            v_shape[rank - 1],
        );
        if d != kd || s != vs {
            return Err(format!(
                "flash: incompatible q/k/v dimensions ({d} vs {kd}, {s} vs {vs})"
            ));
        }
        let bh = q_shape[..rank - 2]
            .iter()
            .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
            .ok_or_else(|| "flash: batch/head count overflow".to_string())?;
        if bh == 0 || t == 0 || s == 0 || d == 0 || dv == 0 {
            return Err("flash: zero-sized dimensions are unsupported".to_string());
        }
        Ok((bh, t, s, d, dv, group))
    }

    /// Plans a flash forward: validates the q/k/v geometry (shared
    /// leading dims, GQA divisibility, matching depths) and sizes the
    /// output and logsumexp buffers.
    pub fn forward_requirements(
        q_shape: &[usize],
        k_shape: &[usize],
        v_shape: &[usize],
        dtype: crate::runtime::dtype::DType,
    ) -> crate::err::Res<SdpaForwardRequirements> {
        let (batch_heads, query_len, key_len, query_depth, value_depth, head_group_size) =
            geometry(q_shape, k_shape, v_shape, dtype)?;
        let rank = q_shape.len();
        let mut output_shape = q_shape[..rank - 1].to_vec();
        output_shape.push(value_depth);
        let logsumexp_shape = q_shape[..rank - 1].to_vec();
        Ok(SdpaForwardRequirements {
            output: BufferRequirement::new(output_shape, dtype, "output")?,
            logsumexp: BufferRequirement::new(
                logsumexp_shape,
                crate::runtime::dtype::DType::F32,
                "logsumexp",
            )?,
            topology: SdpaForwardTopology::Flash {
                query_tile: TILE_Q,
                key_tile: TILE_K,
                threads: THREADS,
                dispatches: 1,
            },
            batch_heads,
            head_group_size,
            query_len,
            key_len,
            query_depth,
            value_depth,
            dtype,
        })
    }

    /// Plans a flash backward; rejects grouped-query attention (not
    /// differentiable in this implementation).
    pub fn backward_requirements(
        q_shape: &[usize],
        k_shape: &[usize],
        v_shape: &[usize],
        dtype: crate::runtime::dtype::DType,
    ) -> crate::err::Res<SdpaBackwardRequirements> {
        let (batch_heads, query_len, key_len, query_depth, value_depth, head_group_size) =
            geometry(q_shape, k_shape, v_shape, dtype)?;
        if head_group_size != 1 {
            return Err(
                "flash backward: grouped-query attention is not differentiable".to_string(),
            );
        }
        Ok(SdpaBackwardRequirements {
            dq: BufferRequirement::new(q_shape.to_vec(), dtype, "dq")?,
            dk: BufferRequirement::new(k_shape.to_vec(), dtype, "dk")?,
            dv: BufferRequirement::new(v_shape.to_vec(), dtype, "dv")?,
            d_vec_scratch: BufferRequirement::new(
                q_shape[..q_shape.len() - 1].to_vec(),
                crate::runtime::dtype::DType::F32,
                "d_vec scratch",
            )?,
            topology: SdpaBackwardTopology::FlashRecompute {
                query_tile: TILE_Q,
                key_tile: TILE_K,
                threads: THREADS,
                dispatches: 3,
            },
            batch_heads,
            query_len,
            key_len,
            query_depth,
            value_depth,
            dtype,
        })
    }

    fn require_contiguous(tensor: &MetalTensor, label: &str) -> crate::err::Res<()> {
        if !tensor.layout.is_contiguous() {
            return Err(format!("flash: {label} must be contiguous"));
        }
        let end = tensor
            .layout
            .offset()
            .checked_add(tensor.numel())
            .and_then(|elements| elements.checked_mul(tensor.dtype.size_in_bytes()))
            .ok_or_else(|| format!("flash: {label} storage range overflow"))?;
        if end > tensor.buffer.size {
            return Err(format!("flash: {label} storage is too small"));
        }
        Ok(())
    }

    fn require_exact(
        tensor: &MetalTensor,
        shape: &[usize],
        dtype: crate::runtime::dtype::DType,
        label: &str,
    ) -> crate::err::Res<()> {
        if tensor.layout.shape() != shape || tensor.dtype != dtype {
            return Err(format!(
                "flash: {label} must be {shape:?}:{dtype:?}, got {:?}:{:?}",
                tensor.layout.shape(),
                tensor.dtype
            ));
        }
        require_contiguous(tensor, label)
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
        window: Option<usize>,
        head_group_size: usize,
        ty: &str,
    ) -> String {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;

#define STOR {ty}

#define T {t}
#define S {s}
#define D {d}
#define DV {dv}
#define TQ {tq}
#define TK {tk}
#define NT {nt}
#define SCALE {scale:?}f
#define CAUSAL {causal}
#define WINDOW {window}
#define GROUP {head_group_size}
// Right-aligned causal window: query q attends to keys k <= q + OFFSET.
#define OFFSET {offset}

kernel void et_sdpa_fwd(
    device const STOR* Q [[buffer(0)]],
    device const STOR* K [[buffer(1)]],
    device const STOR* V [[buffer(2)]],
    device STOR* O [[buffer(3)]],
    device float* L [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {{
    const int q0 = tgid.x * TQ;
    const long bh = tgid.y;
    const uint tid = tpitg.x;
    device const STOR* Qb = Q + bh * (long)T * D;
    device const STOR* Kb = K + (bh / GROUP) * (long)S * D;
    device const STOR* Vb = V + (bh / GROUP) * (long)S * DV;
    device STOR* Ob = O + bh * (long)T * DV;

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
            if (q < T && k < S && (!CAUSAL || (k <= q + OFFSET && (WINDOW == 0 || k + WINDOW > q + OFFSET)))) {{
                acc = 0.0f;
                for (int d = 0; d < D; d++) {{ acc += float(Qb[q * D + d]) * float(Kb[k * D + d]); }}
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
                if (k < S) {{ acc += St[qi][kj] * float(Vb[k * DV + dv]); }}
            }}
            Ot[qi][dv] = acc;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    for (int i = tid; i < TQ * DV; i += NT) {{
        int qi = i / DV;
        int dv = i % DV;
        int q = q0 + qi;
        if (q < T) {{ Ob[q * DV + dv] = STOR(Ot[qi][dv] / l[qi]); }}
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
            window = window.unwrap_or(0),
            head_group_size = head_group_size,
            offset = s.saturating_sub(t),
            ty = ty,
        )
    }

    fn forward_key(
        t: usize,
        s: usize,
        d: usize,
        dv: usize,
        scale: f64,
        causal: bool,
        window: Option<usize>,
        head_group_size: usize,
        dtype: crate::runtime::dtype::DType,
    ) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        (
            t,
            s,
            d,
            dv,
            scale.to_bits(),
            causal,
            window,
            head_group_size,
            dtype,
        )
            .hash(&mut hasher);
        hasher.finish()
    }

    fn pipeline(
        t: usize,
        s: usize,
        d: usize,
        dv: usize,
        scale: f64,
        causal: bool,
        window: Option<usize>,
        head_group_size: usize,
        dtype: crate::runtime::dtype::DType,
    ) -> crate::err::Res<Pipeline> {
        #[cfg(test)]
        super::record_pipeline_request();
        let ty = match dtype {
            crate::runtime::dtype::DType::F32 => "float",
            crate::runtime::dtype::DType::F16 => "half",
            crate::runtime::dtype::DType::BF16 => "bfloat",
            other => return Err(format!("flash: unsupported dtype {other:?}")),
        };
        MetalDevice::get().compile_lazy(
            forward_key(t, s, d, dv, scale, causal, window, head_group_size, dtype),
            "et_sdpa_fwd",
            || kernel_source(t, s, d, dv, scale, causal, window, head_group_size, ty),
        )
    }

    fn cached_forward_pipeline(
        t: usize,
        s: usize,
        d: usize,
        dv: usize,
        scale: f64,
        causal: bool,
        window: Option<usize>,
        head_group_size: usize,
        dtype: crate::runtime::dtype::DType,
    ) -> crate::err::Res<Pipeline> {
        MetalDevice::get()
            .pipeline_cached(forward_key(
                t,
                s,
                d,
                dv,
                scale,
                causal,
                window,
                head_group_size,
                dtype,
            ))
            .ok_or_else(|| {
                "flash: forward pipeline is not warm; call warm_forward_exact first".to_string()
            })
    }

    /// Warms the forward pipeline for the given shapes, scale, masking,
    /// and dtype.
    pub fn warm_forward(
        q_shape: &[usize],
        k_shape: &[usize],
        v_shape: &[usize],
        scale: f64,
        causal: bool,
        window: Option<usize>,
        dtype: crate::runtime::dtype::DType,
    ) -> crate::err::Res<()> {
        let requirements = forward_requirements(q_shape, k_shape, v_shape, dtype)?;
        warm_forward_exact(&requirements, scale, causal, window)
    }

    /// Warms exactly the forward pipeline described by `requirements`
    /// plus the runtime (`scale`, `causal`, `window`) parameters, all
    /// of which are baked into the pipeline key.
    pub fn warm_forward_exact(
        requirements: &SdpaForwardRequirements,
        scale: f64,
        causal: bool,
        window: Option<usize>,
    ) -> crate::err::Res<()> {
        pipeline(
            requirements.query_len,
            requirements.key_len,
            requirements.query_depth,
            requirements.value_depth,
            scale,
            causal,
            window,
            requirements.head_group_size,
            requirements.dtype,
        )?;
        Ok(())
    }

    /// Non-allocating flash forward dispatch: writes the attention
    /// output and the per-row f32 `logsumexp`. All tensors contiguous;
    /// pipeline must be warm for the exact (shape, scale, causal,
    /// window, dtype) combination.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_into(
        q: &MetalTensor,
        k: &MetalTensor,
        v: &MetalTensor,
        scale: f64,
        causal: bool,
        window: Option<usize>,
        output: &MetalTensor,
        logsumexp: &MetalTensor,
    ) -> crate::err::Res<()> {
        if q.dtype != k.dtype || q.dtype != v.dtype {
            return Err("flash: q/k/v dtypes must match".to_string());
        }
        let (bh, t, s, d, dv, head_group_size) = geometry(
            q.layout.shape(),
            k.layout.shape(),
            v.layout.shape(),
            q.dtype,
        )?;
        for (tensor, label) in [(q, "q"), (k, "k"), (v, "v")] {
            require_contiguous(tensor, label)?;
        }
        let rank = q.layout.shape().len();
        let output_shape = output.layout.shape();
        if output.dtype != q.dtype
            || output_shape.len() != rank
            || output_shape[..rank - 1] != q.layout.shape()[..rank - 1]
            || output_shape[rank - 1] != dv
        {
            return Err("flash: output shape or dtype mismatch".to_string());
        }
        require_contiguous(output, "output")?;
        require_exact(
            logsumexp,
            &q.layout.shape()[..rank - 1],
            crate::runtime::dtype::DType::F32,
            "logsumexp",
        )?;
        MetalDevice::get().mark_buffer_write(&output.buffer)?;
        MetalDevice::get().mark_buffer_write(&logsumexp.buffer)?;

        let pipe =
            cached_forward_pipeline(t, s, d, dv, scale, causal, window, head_group_size, q.dtype)?;
        let sz = q.dtype.size_in_bytes();
        MetalDevice::get().with_encoder(|e| {
            e.setComputePipelineState(pipe.as_raw());
            set_buffer(e, 0, &q.buffer, q.layout.offset() * sz);
            set_buffer(e, 1, &k.buffer, k.layout.offset() * sz);
            set_buffer(e, 2, &v.buffer, v.layout.offset() * sz);
            set_buffer(e, 3, &output.buffer, output.layout.offset() * sz);
            set_buffer(
                e,
                4,
                &logsumexp.buffer,
                logsumexp.layout.offset() * crate::runtime::dtype::DType::F32.size_in_bytes(),
            );
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
        Ok(())
    }

    /// Allocating convenience wrapper around [`forward_into`]; returns
    /// `(output, logsumexp)`.
    pub fn forward(
        q: &MetalTensor,
        k: &MetalTensor,
        v: &MetalTensor,
        scale: f64,
        causal: bool,
        window: Option<usize>,
    ) -> crate::err::Res<(MetalTensor, MetalTensor)> {
        if q.dtype != k.dtype || q.dtype != v.dtype {
            return Err("flash: q/k/v dtypes must match".to_string());
        }
        let requirements = forward_requirements(
            q.layout.shape(),
            k.layout.shape(),
            v.layout.shape(),
            q.dtype,
        )?;
        warm_forward_exact(&requirements, scale, causal, window)?;
        let q = contig(q)?;
        let k = contig(k)?;
        let v = contig(v)?;
        let o = MetalTensor {
            buffer: alloc(requirements.output.elements, q.dtype),
            layout: crate::runtime::layout::Layout::contiguous(requirements.output.shape.clone()),
            dtype: q.dtype,
        };
        let l = MetalTensor {
            buffer: alloc(
                requirements.logsumexp.elements.max(1),
                crate::runtime::dtype::DType::F32,
            ),
            layout: crate::runtime::layout::Layout::contiguous(
                requirements.logsumexp.shape.clone(),
            ),
            dtype: crate::runtime::dtype::DType::F32,
        };
        forward_into(&q, &k, &v, scale, causal, window, &o, &l)?;
        Ok((o, l))
    }
    // The fused gradient passes use no atomics. Pass A (key-tiled)
    // accumulates dk/dv — thread (kj-cell) owns its accumulator in
    // registers across the whole query sweep. Pass B (query-tiled)
    // accumulates dq. The score tile is recomputed once per tile pair
    // in threadgroup memory and consumed by all four gradients — the
    // shared-data win over the composed recompute (~12 DRAM round
    // trips per tile).
    fn bwd_source(
        t: usize,
        s: usize,
        d: usize,
        dv: usize,
        scale: f64,
        causal: bool,
        window: Option<usize>,
        ty: &str,
    ) -> String {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;

#define STOR {ty}

#define T {t}
#define S {s}
#define D {d}
#define DV {dv}
#define TQ {tq}
#define TK {tk}
#define NT {nt}
#define SCALE {scale:?}f
#define CAUSAL {causal}
#define WINDOW {window}
#define OFFSET {offset}
#define CELLS_DK ((TK * D + NT - 1) / NT)
#define CELLS_DV ((TK * DV + NT - 1) / NT)
#define CELLS_DQ ((TQ * D + NT - 1) / NT)

// D[row] = dot(G[row], O[row]). This replaces the allocating composed
// multiply/reduce prelude and writes directly into declared f32 scratch.
kernel void et_sdpa_bwd_d(
    device const STOR* O [[buffer(0)]],
    device const STOR* G [[buffer(1)]],
    device float* Dvec [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    const uint row = gid / 32;
    const uint lane = gid % 32;
    float acc = 0.0f;
    const ulong base = (ulong)row * DV;
    for (uint j = lane; j < DV; j += 32) {{
        acc += float(O[base + j]) * float(G[base + j]);
    }}
    acc = simd_sum(acc);
    if (lane == 0) {{ Dvec[row] = acc; }}
}}

kernel void et_sdpa_bwd_kv(
    device const STOR* Q [[buffer(0)]],
    device const STOR* K [[buffer(1)]],
    device const STOR* V [[buffer(2)]],
    device const STOR* G [[buffer(3)]],
    device const float* L [[buffer(4)]],
    device const float* Dvec [[buffer(5)]],
    device STOR* DK [[buffer(6)]],
    device STOR* DVout [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {{
    const int j0 = tgid.x * TK;
    const long bh = tgid.y;
    const uint tid = tpitg.x;
    device const STOR* Qb = Q + bh * (long)T * D;
    device const STOR* Kb = K + bh * (long)S * D;
    device const STOR* Vb = V + bh * (long)S * DV;
    device const STOR* Gb = G + bh * (long)T * DV;
    device const float* Lb = L + bh * (long)T;
    device const float* Db = Dvec + bh * (long)T;
    device STOR* DKb = DK + bh * (long)S * D;
    device STOR* DVb = DVout + bh * (long)S * DV;

    threadgroup float Kt[TK][D];
    threadgroup float Vt[TK][DV];
    for (int i = tid; i < TK * D; i += NT) {{
        int kj = i / D, dd = i % D;
        Kt[kj][dd] = (j0 + kj < S) ? float(Kb[(j0 + kj) * D + dd]) : 0.0f;
    }}
    for (int i = tid; i < TK * DV; i += NT) {{
        int kj = i / DV, dv = i % DV;
        Vt[kj][dv] = (j0 + kj < S) ? float(Vb[(j0 + kj) * DV + dv]) : 0.0f;
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
            Qt[t][dd] = (i0 + t < T) ? float(Qb[(i0 + t) * D + dd]) : 0.0f;
        }}
        for (int i = tid; i < TQ * DV; i += NT) {{
            int t = i / DV, dv = i % DV;
            Gt[t][dv] = (i0 + t < T) ? float(Gb[(i0 + t) * DV + dv]) : 0.0f;
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
            if (q < T && k < S && (!CAUSAL || (k <= q + OFFSET && (WINDOW == 0 || k + WINDOW > q + OFFSET)))) {{
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
        if (j0 + kj < S) {{ DKb[(j0 + kj) * D + dd] = STOR(acc_dk[w]); }}
    }}
    for (int w = 0; w < CELLS_DV; w++) {{
        int c = tid + w * NT;
        if (c >= TK * DV) {{ break; }}
        int kj = c / DV, dv = c % DV;
        if (j0 + kj < S) {{ DVb[(j0 + kj) * DV + dv] = STOR(acc_dv[w]); }}
    }}
}}

kernel void et_sdpa_bwd_q(
    device const STOR* Q [[buffer(0)]],
    device const STOR* K [[buffer(1)]],
    device const STOR* V [[buffer(2)]],
    device const STOR* G [[buffer(3)]],
    device const float* L [[buffer(4)]],
    device const float* Dvec [[buffer(5)]],
    device STOR* DQ [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {{
    const int i0 = tgid.x * TQ;
    const long bh = tgid.y;
    const uint tid = tpitg.x;
    device const STOR* Qb = Q + bh * (long)T * D;
    device const STOR* Kb = K + bh * (long)S * D;
    device const STOR* Vb = V + bh * (long)S * DV;
    device const STOR* Gb = G + bh * (long)T * DV;
    device const float* Lb = L + bh * (long)T;
    device const float* Db = Dvec + bh * (long)T;
    device STOR* DQb = DQ + bh * (long)T * D;

    threadgroup float Qt[TQ][D];
    threadgroup float Gt[TQ][DV];
    threadgroup float lt[TQ];
    threadgroup float dt[TQ];
    for (int i = tid; i < TQ * D; i += NT) {{
        int t = i / D, dd = i % D;
        Qt[t][dd] = (i0 + t < T) ? float(Qb[(i0 + t) * D + dd]) : 0.0f;
    }}
    for (int i = tid; i < TQ * DV; i += NT) {{
        int t = i / DV, dv = i % DV;
        Gt[t][dv] = (i0 + t < T) ? float(Gb[(i0 + t) * DV + dv]) : 0.0f;
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
            Kt[kj][dd] = (j0 + kj < S) ? float(Kb[(j0 + kj) * D + dd]) : 0.0f;
        }}
        for (int i = tid; i < TK * DV; i += NT) {{
            int kj = i / DV, dv = i % DV;
            Vt[kj][dv] = (j0 + kj < S) ? float(Vb[(j0 + kj) * DV + dv]) : 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < TQ * TK; i += NT) {{
            int t = i / TK, kj = i % TK;
            int q = i0 + t, k = j0 + kj;
            float p;
            if (q < T && k < S && (!CAUSAL || (k <= q + OFFSET && (WINDOW == 0 || k + WINDOW > q + OFFSET)))) {{
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
        if (i0 + t < T) {{ DQb[(i0 + t) * D + dd] = STOR(acc_dq[w]); }}
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
            window = window.unwrap_or(0),
            offset = s.saturating_sub(t),
            ty = ty,
        )
    }

    fn backward_key(
        name: &'static str,
        t: usize,
        s: usize,
        d: usize,
        dv: usize,
        scale: f64,
        causal: bool,
        window: Option<usize>,
        dtype: crate::runtime::dtype::DType,
    ) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        (name, t, s, d, dv, scale.to_bits(), causal, window, dtype).hash(&mut hasher);
        hasher.finish()
    }

    fn bwd_pipeline(
        name: &'static str,
        t: usize,
        s: usize,
        d: usize,
        dv: usize,
        scale: f64,
        causal: bool,
        window: Option<usize>,
        dtype: crate::runtime::dtype::DType,
    ) -> crate::err::Res<Pipeline> {
        #[cfg(test)]
        super::record_pipeline_request();
        let ty = match dtype {
            crate::runtime::dtype::DType::F32 => "float",
            crate::runtime::dtype::DType::F16 => "half",
            crate::runtime::dtype::DType::BF16 => "bfloat",
            other => return Err(format!("flash: unsupported dtype {other:?}")),
        };
        MetalDevice::get().compile_lazy(
            backward_key(name, t, s, d, dv, scale, causal, window, dtype),
            name,
            || bwd_source(t, s, d, dv, scale, causal, window, ty),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn cached_backward_pipeline(
        name: &'static str,
        t: usize,
        s: usize,
        d: usize,
        dv: usize,
        scale: f64,
        causal: bool,
        window: Option<usize>,
        dtype: crate::runtime::dtype::DType,
    ) -> crate::err::Res<Pipeline> {
        MetalDevice::get()
            .pipeline_cached(backward_key(
                name, t, s, d, dv, scale, causal, window, dtype,
            ))
            .ok_or_else(|| {
                format!("flash: {name} pipeline is not warm; call warm_backward_exact first")
            })
    }

    /// Warms the three backward pipelines for the given shapes, scale,
    /// masking, and dtype.
    pub fn warm_backward(
        q_shape: &[usize],
        k_shape: &[usize],
        v_shape: &[usize],
        scale: f64,
        causal: bool,
        window: Option<usize>,
        dtype: crate::runtime::dtype::DType,
    ) -> crate::err::Res<()> {
        let requirements = backward_requirements(q_shape, k_shape, v_shape, dtype)?;
        warm_backward_exact(&requirements, scale, causal, window)
    }

    /// Warms exactly the three backward pipelines described by
    /// `requirements` plus the runtime (`scale`, `causal`, `window`)
    /// parameters.
    pub fn warm_backward_exact(
        requirements: &SdpaBackwardRequirements,
        scale: f64,
        causal: bool,
        window: Option<usize>,
    ) -> crate::err::Res<()> {
        let (t, s, d, dv, dtype) = (
            requirements.query_len,
            requirements.key_len,
            requirements.query_depth,
            requirements.value_depth,
            requirements.dtype,
        );
        bwd_pipeline("et_sdpa_bwd_d", t, s, d, dv, scale, causal, window, dtype)?;
        bwd_pipeline("et_sdpa_bwd_kv", t, s, d, dv, scale, causal, window, dtype)?;
        bwd_pipeline("et_sdpa_bwd_q", t, s, d, dv, scale, causal, window, dtype)?;
        Ok(())
    }

    /// Non-allocating fused backward dispatch: given the forward's
    /// output `o`, logsumexp `l`, and cotangent `g`, writes `dq`, `dk`,
    /// `dv_out` and the `d_vec_scratch` row dots. All three pipelines
    /// are resolved before any work is encoded, so a partial warmup
    /// fails without a partially encoded operation. Allocates nothing.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_fused_into(
        q: &MetalTensor,
        k: &MetalTensor,
        v: &MetalTensor,
        o: &MetalTensor,
        l: &MetalTensor,
        g: &MetalTensor,
        scale: f64,
        causal: bool,
        window: Option<usize>,
        dq: &MetalTensor,
        dk: &MetalTensor,
        dv_out: &MetalTensor,
        d_vec_scratch: &MetalTensor,
    ) -> crate::err::Res<()> {
        if q.dtype != k.dtype || q.dtype != v.dtype || q.dtype != o.dtype || q.dtype != g.dtype {
            return Err("flash: backward storage dtypes must match".to_string());
        }
        let (bh, t, s, d, dv, head_group_size) = geometry(
            q.layout.shape(),
            k.layout.shape(),
            v.layout.shape(),
            q.dtype,
        )?;
        if head_group_size != 1 {
            return Err(
                "flash backward: grouped-query attention is not differentiable".to_string(),
            );
        }
        let rank = q.layout.shape().len();
        let output_shape = o.layout.shape();
        if output_shape.len() != rank
            || output_shape[..rank - 1] != q.layout.shape()[..rank - 1]
            || output_shape[rank - 1] != dv
            || g.layout.shape() != output_shape
        {
            return Err("flash: backward output/gradient shape mismatch".to_string());
        }
        require_exact(
            l,
            &q.layout.shape()[..rank - 1],
            crate::runtime::dtype::DType::F32,
            "logsumexp",
        )?;
        require_exact(dq, q.layout.shape(), q.dtype, "dq")?;
        require_exact(dk, k.layout.shape(), q.dtype, "dk")?;
        require_exact(dv_out, v.layout.shape(), q.dtype, "dv")?;
        require_exact(
            d_vec_scratch,
            &q.layout.shape()[..rank - 1],
            crate::runtime::dtype::DType::F32,
            "d_vec scratch",
        )?;
        for (tensor, label) in [(q, "q"), (k, "k"), (v, "v"), (o, "output"), (g, "gradient")] {
            require_contiguous(tensor, label)?;
        }
        for tensor in [dq, dk, dv_out, d_vec_scratch] {
            MetalDevice::get().mark_buffer_write(&tensor.buffer)?;
        }

        // Resolve every pipeline before encoding any work, so a partial
        // warmup cannot leave a partially encoded backward operation.
        let d_pipe =
            cached_backward_pipeline("et_sdpa_bwd_d", t, s, d, dv, scale, causal, window, q.dtype)?;
        let kv_pipe = cached_backward_pipeline(
            "et_sdpa_bwd_kv",
            t,
            s,
            d,
            dv,
            scale,
            causal,
            window,
            q.dtype,
        )?;
        let q_pipe =
            cached_backward_pipeline("et_sdpa_bwd_q", t, s, d, dv, scale, causal, window, q.dtype)?;
        let sz = q.dtype.size_in_bytes();
        let f32_size = crate::runtime::dtype::DType::F32.size_in_bytes();

        MetalDevice::get().with_encoder(|e| {
            e.setComputePipelineState(d_pipe.as_raw());
            set_buffer(e, 0, &o.buffer, o.layout.offset() * sz);
            set_buffer(e, 1, &g.buffer, g.layout.offset() * sz);
            set_buffer(
                e,
                2,
                &d_vec_scratch.buffer,
                d_vec_scratch.layout.offset() * f32_size,
            );
            e.dispatchThreads_threadsPerThreadgroup(
                objc2_metal::MTLSize {
                    width: bh * t * 32,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: THREADS,
                    height: 1,
                    depth: 1,
                },
            );
        });
        MetalDevice::get().with_encoder(|e| {
            e.setComputePipelineState(kv_pipe.as_raw());
            set_buffer(e, 0, &q.buffer, q.layout.offset() * sz);
            set_buffer(e, 1, &k.buffer, k.layout.offset() * sz);
            set_buffer(e, 2, &v.buffer, v.layout.offset() * sz);
            set_buffer(e, 3, &g.buffer, g.layout.offset() * sz);
            set_buffer(e, 4, &l.buffer, l.layout.offset() * f32_size);
            set_buffer(
                e,
                5,
                &d_vec_scratch.buffer,
                d_vec_scratch.layout.offset() * f32_size,
            );
            set_buffer(e, 6, &dk.buffer, dk.layout.offset() * sz);
            set_buffer(e, 7, &dv_out.buffer, dv_out.layout.offset() * sz);
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
        MetalDevice::get().with_encoder(|e| {
            e.setComputePipelineState(q_pipe.as_raw());
            set_buffer(e, 0, &q.buffer, q.layout.offset() * sz);
            set_buffer(e, 1, &k.buffer, k.layout.offset() * sz);
            set_buffer(e, 2, &v.buffer, v.layout.offset() * sz);
            set_buffer(e, 3, &g.buffer, g.layout.offset() * sz);
            set_buffer(e, 4, &l.buffer, l.layout.offset() * f32_size);
            set_buffer(
                e,
                5,
                &d_vec_scratch.buffer,
                d_vec_scratch.layout.offset() * f32_size,
            );
            set_buffer(e, 6, &dq.buffer, dq.layout.offset() * sz);
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
        Ok(())
    }

    /// Allocating convenience wrapper around [`backward_fused_into`];
    /// returns `(dq, dk, dv)`.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_fused(
        q: &MetalTensor,
        k: &MetalTensor,
        v: &MetalTensor,
        o: &MetalTensor,
        l: &MetalTensor,
        g: &MetalTensor,
        scale: f64,
        causal: bool,
        window: Option<usize>,
    ) -> crate::err::Res<(MetalTensor, MetalTensor, MetalTensor)> {
        if q.dtype != k.dtype || q.dtype != v.dtype || q.dtype != o.dtype || q.dtype != g.dtype {
            return Err("flash: backward storage dtypes must match".to_string());
        }
        let requirements = backward_requirements(
            q.layout.shape(),
            k.layout.shape(),
            v.layout.shape(),
            q.dtype,
        );
        let requirements = requirements?;
        warm_backward_exact(&requirements, scale, causal, window)?;
        let q = contig(q)?;
        let k = contig(k)?;
        let v = contig(v)?;
        let o = contig(o)?;
        let l = contig(l)?;
        let g = contig(g)?;
        let dq = MetalTensor {
            buffer: alloc(requirements.dq.elements, q.dtype),
            layout: crate::runtime::layout::Layout::contiguous(requirements.dq.shape.clone()),
            dtype: q.dtype,
        };
        let dk = MetalTensor {
            buffer: alloc(requirements.dk.elements, q.dtype),
            layout: crate::runtime::layout::Layout::contiguous(requirements.dk.shape.clone()),
            dtype: q.dtype,
        };
        let dv_out = MetalTensor {
            buffer: alloc(requirements.dv.elements, q.dtype),
            layout: crate::runtime::layout::Layout::contiguous(requirements.dv.shape.clone()),
            dtype: q.dtype,
        };
        let d_vec_scratch = MetalTensor {
            buffer: alloc(
                requirements.d_vec_scratch.elements.max(1),
                crate::runtime::dtype::DType::F32,
            ),
            layout: crate::runtime::layout::Layout::contiguous(
                requirements.d_vec_scratch.shape.clone(),
            ),
            dtype: crate::runtime::dtype::DType::F32,
        };
        backward_fused_into(
            &q,
            &k,
            &v,
            &o,
            &l,
            &g,
            scale,
            causal,
            window,
            &dq,
            &dk,
            &dv_out,
            &d_vec_scratch,
        )?;
        Ok((dq, dk, dv_out))
    }
}

#[cfg(test)]
mod attn_probe {
    use crate::runtime::dtype::DType;
    use crate::runtime::metal::device::MetalDevice;
    use crate::runtime::metal::run::MetalTensor as MT;

    fn pattern(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 7 + 3) % 13) as f32 / 4.0 - 1.5)
            .collect()
    }

    #[test]
    fn requirements_include_auxiliary_scratch_and_topology() {
        let forward = crate::flash::forward_requirements(
            &[2, 3, 5, 7],
            &[2, 3, 11, 7],
            &[2, 3, 11, 13],
            DType::F16,
        )
        .unwrap();
        assert_eq!(forward.output.shape, [2, 3, 5, 13]);
        assert_eq!(forward.logsumexp.shape, [2, 3, 5]);
        assert_eq!(forward.logsumexp.dtype, DType::F32);
        assert_eq!(forward.logsumexp.bytes, 2 * 3 * 5 * 4);
        assert_eq!(
            forward.topology,
            super::SdpaForwardTopology::Flash {
                query_tile: 16,
                key_tile: 32,
                threads: 128,
                dispatches: 1,
            }
        );

        let backward = crate::flash::backward_requirements(
            &[2, 3, 5, 7],
            &[2, 3, 11, 7],
            &[2, 3, 11, 13],
            DType::F16,
        )
        .unwrap();
        assert_eq!(backward.dq.shape, [2, 3, 5, 7]);
        assert_eq!(backward.dk.shape, [2, 3, 11, 7]);
        assert_eq!(backward.dv.shape, [2, 3, 11, 13]);
        assert_eq!(backward.d_vec_scratch.shape, [2, 3, 5]);
        assert_eq!(backward.d_vec_scratch.dtype, DType::F32);
        assert_eq!(
            backward.topology,
            super::SdpaBackwardTopology::FlashRecompute {
                query_tile: 16,
                key_tile: 32,
                threads: 128,
                dispatches: 3,
            }
        );
    }

    #[test]
    fn grouped_query_ratios_and_local_windows_match_explicit_values() {
        let dev = MetalDevice::get();
        for (query_heads, kv_heads) in [(4usize, 2usize), (16, 1)] {
            let q_values = pattern(query_heads * 2 * 2);
            let k_values = pattern(kv_heads * 3 * 2);
            let v_values = pattern(kv_heads * 3 * 3);
            let q = MT::from_f32(dev, q_values, vec![1, query_heads, 2, 2]);
            let k = MT::from_f32(dev, k_values.clone(), vec![1, kv_heads, 3, 2]);
            let v = MT::from_f32(dev, v_values.clone(), vec![1, kv_heads, 3, 3]);
            let repeat = |values: &[f32], width: usize| {
                let mut repeated = Vec::new();
                let per_head = 3 * width;
                for head in 0..kv_heads {
                    for _ in 0..query_heads / kv_heads {
                        repeated.extend_from_slice(&values[head * per_head..(head + 1) * per_head]);
                    }
                }
                repeated
            };
            let repeated_k = MT::from_f32(dev, repeat(&k_values, 2), vec![1, query_heads, 3, 2]);
            let repeated_v = MT::from_f32(dev, repeat(&v_values, 3), vec![1, query_heads, 3, 3]);
            let (actual, _) = crate::flash::forward(&q, &k, &v, 0.5, false, None).unwrap();
            let (expected, _) =
                crate::flash::forward(&q, &repeated_k, &repeated_v, 0.5, false, None).unwrap();
            dev.synchronize().unwrap();
            for (actual, expected) in actual
                .read_f32()
                .unwrap()
                .iter()
                .zip(expected.read_f32().unwrap())
            {
                assert!((actual - expected).abs() < 2e-5);
            }
        }

        let q = MT::from_f32(dev, vec![0.0; 2], vec![1, 1, 2, 1]);
        let k = MT::from_f32(dev, vec![0.0; 5], vec![1, 1, 5, 1]);
        let v = MT::from_f32(dev, vec![1.0, 2.0, 4.0, 8.0, 16.0], vec![1, 1, 5, 1]);
        let (local, _) = crate::flash::forward(&q, &k, &v, 1.0, true, Some(2)).unwrap();
        let (full, _) = crate::flash::forward(&q, &k, &v, 1.0, true, None).unwrap();
        dev.synchronize().unwrap();
        assert_eq!(local.read_f32().unwrap(), vec![6.0, 12.0]);
        assert_eq!(full.read_f32().unwrap(), vec![3.75, 6.2]);
    }

    #[test]
    fn into_matches_allocating_forward_and_cpu_backward_without_allocating() {
        let dev = MetalDevice::get();
        let q_values = pattern(8);
        let k_values: Vec<f32> = pattern(20).iter().map(|x| x * 0.7).collect();
        let v_values: Vec<f32> = pattern(20).iter().map(|x| x * -0.5).collect();
        let g_values: Vec<f32> = pattern(8).iter().map(|x| x * 0.3).collect();
        let q = MT::from_f32(dev, q_values.clone(), vec![1, 1, 2, 4]);
        let k = MT::from_f32(dev, k_values.clone(), vec![1, 1, 5, 4]);
        let v = MT::from_f32(dev, v_values.clone(), vec![1, 1, 5, 4]);
        let g = MT::from_f32(dev, g_values.clone(), vec![1, 1, 2, 4]);
        let scale = 0.5;
        let forward_req = crate::flash::forward_requirements(
            q.layout.shape(),
            k.layout.shape(),
            v.layout.shape(),
            DType::F32,
        )
        .unwrap();
        crate::flash::warm_forward_exact(&forward_req, scale, true, None).unwrap();
        let o_into = MT::empty(dev, forward_req.output.shape.clone(), DType::F32);
        let l_into = MT::empty(dev, forward_req.logsumexp.shape.clone(), DType::F32);
        super::reset_test_counts();
        crate::flash::forward_into(&q, &k, &v, scale, true, None, &o_into, &l_into).unwrap();
        assert_eq!(super::test_counts(), (0, 0));

        let (o, l) = crate::flash::forward(&q, &k, &v, scale, true, None).unwrap();
        let backward_req = crate::flash::backward_requirements(
            q.layout.shape(),
            k.layout.shape(),
            v.layout.shape(),
            DType::F32,
        )
        .unwrap();
        crate::flash::warm_backward_exact(&backward_req, scale, true, None).unwrap();
        let dq = MT::empty(dev, backward_req.dq.shape.clone(), DType::F32);
        let dk = MT::empty(dev, backward_req.dk.shape.clone(), DType::F32);
        let dv = MT::empty(dev, backward_req.dv.shape.clone(), DType::F32);
        let d_vec = MT::empty(dev, backward_req.d_vec_scratch.shape.clone(), DType::F32);
        super::reset_test_counts();
        crate::flash::backward_fused_into(
            &q, &k, &v, &o, &l, &g, scale, true, None, &dq, &dk, &dv, &d_vec,
        )
        .unwrap();
        assert_eq!(super::test_counts(), (0, 0));

        dev.synchronize().unwrap();
        let qc = crate::runtime::cpu::Tensor::from_vec(q_values, vec![1, 1, 2, 4]);
        let kc = crate::runtime::cpu::Tensor::from_vec(k_values, vec![1, 1, 5, 4]);
        let vc = crate::runtime::cpu::Tensor::from_vec(v_values, vec![1, 1, 5, 4]);
        let gc = crate::runtime::cpu::Tensor::from_vec(g_values, vec![1, 1, 2, 4]);
        let oc = crate::runtime::cpu::composed::sdpa_forward(&qc, &kc, &vc, scale, true);
        let gpu = o.read_f32().unwrap();
        let gpu_into = o_into.read_f32().unwrap();
        let oc = oc.contiguous();
        let crate::runtime::cpu::CpuBuffer::F32(cpu) = &oc.buffer else {
            panic!("expected f32 output")
        };
        for ((g, gi), c) in gpu.iter().zip(&gpu_into).zip(cpu.iter()) {
            assert!((g - c).abs() < 1e-5, "flash {g} vs composed {c}");
            assert!((gi - g).abs() < 1e-6, "into {gi} vs allocating {g}");
        }

        let (dqc, dkc, dvc) =
            crate::runtime::cpu::composed::sdpa_backward(&qc, &kc, &vc, &gc, scale, true);
        for (actual, expected) in [
            (dq.read_f32().unwrap(), dqc),
            (dk.read_f32().unwrap(), dkc),
            (dv.read_f32().unwrap(), dvc),
        ] {
            let expected = expected.contiguous();
            let crate::runtime::cpu::CpuBuffer::F32(expected) = &expected.buffer else {
                panic!("expected f32 gradient")
            };
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert!(
                    (actual - expected).abs() < 2e-5,
                    "flash backward {actual} vs composed {expected}"
                );
            }
        }
    }
}

//! Tiled matmul family: naive tiled gemm, simdgroup-MMA gemm, and split-K
//! gemm, all with fused bias/residual/gelu epilogues.
//!
//! # Algorithms
//!
//! - [`GemmAlgorithm::Tiled`] — one threadgroup per 16×16 output tile,
//!   threadgroup-memory staged; the reference path, used for small shapes
//!   and when MMA is disabled (`EFFECT_TORCH_NO_MMA`).
//! - [`GemmAlgorithm::SimdgroupMma`] — 8×8 simdgroup matrix
//!   multiply-accumulate, one threadgroup per 64×64 (or 32×32 on smaller
//!   threadgroup-memory devices) output tile. bf16/f16 stage and multiply
//!   natively with an f32 accumulator; f32 stages as f32.
//! - [`GemmAlgorithm::SplitK`] — for long-K narrow-output gemms (backward
//!   dX/dW): K is partitioned across threadgroups writing f32 partials,
//!   reduced by a second kernel in a fixed order (deterministic). Plain
//!   epilogues only.
//!
//! # Restrictions
//!
//! - dtypes: f32, f16, bf16 only; M, N, K and batch strides must fit u32;
//!   inputs and destinations must be contiguous (strides are the batch
//!   strides `stride_a`/`stride_b`, row-major within a matrix).
//! - The epilogue contract is fixed: `v = acc + bias + residual` (each
//!   optional), stored plainly, through gelu, or — dual — both (plain
//!   pre-activation for backward, gelu output for the next op).
//!
//! # Planning contract
//!
//! [`gemm_requirements`]/[`matmul_requirements`] compute the exact
//! algorithm, output sizes, and split-K scratch for a shape; planning
//! snapshots the MMA environment so later changes cannot alter a compiled
//! executable. `precompile_*` caches exactly those pipelines; the
//! `*_into` entry points validate every buffer against the requirements
//! and dispatch without allocating.

use super::device::{set_buffer, set_bytes, MetalDevice};
use super::emit::ACT_FNS;
use super::run::MetalTensor;
use crate::runtime::dtype::DType;
use objc2_metal::MTLComputeCommandEncoder;
use objc2_metal::MTLDevice as _;

const TILE: usize = 16;

/// RFC 0016 phase 3: gemm epilogues. The accumulator is finalized as
/// `v = acc + bias + residual` (each term optional), then stored either
/// plainly, through gelu, or — `dual` — both (the plain pre-activation
/// feeds backward, the gelu output feeds the next op, one gemm launch
/// writes both).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Epilogue {
    /// Store `acc + bias` unchanged.
    None,
    /// Store `acc + bias + R` (residual add, full-size extra input).
    Residual,
    /// Store `gelu_erf(acc + bias)`.
    GeluErf,
    /// Store `gelu_tanh(acc + bias)`.
    GeluTanh,
    /// Dual store: plain pre-activation to `D`, `gelu_erf(v)` to `D2`.
    GeluErfDual,
    /// Dual store: plain pre-activation to `D`, `gelu_tanh(v)` to `D2`.
    GeluTanhDual,
}

impl Epilogue {
    fn gelu_fn(self) -> Option<&'static str> {
        match self {
            Epilogue::GeluErf | Epilogue::GeluErfDual => Some("gelu_as"),
            Epilogue::GeluTanh | Epilogue::GeluTanhDual => Some("gelu_tanh_as"),
            _ => None,
        }
    }

    fn dual(self) -> bool {
        matches!(self, Epilogue::GeluErfDual | Epilogue::GeluTanhDual)
    }

    fn code(self) -> u64 {
        match self {
            Epilogue::None => 0,
            Epilogue::Residual => 1,
            Epilogue::GeluErf => 2,
            Epilogue::GeluTanh => 3,
            Epilogue::GeluErfDual => 4,
            Epilogue::GeluTanhDual => 5,
        }
    }
}

fn gemm_source(bias: bool, epilogue: Epilogue, ty: &str) -> String {
    let bias_decl = if bias {
        format!("    device const {ty}* bias [[buffer(3)]],\n")
    } else {
        String::new()
    };
    let bias_add = if bias { " + float(bias[j])" } else { "" };
    let res_decl = if epilogue == Epilogue::Residual {
        format!("    device const {ty}* R [[buffer(9)]],\n")
    } else {
        String::new()
    };
    let res_add = if epilogue == Epilogue::Residual {
        " + float(R[d_idx])"
    } else {
        ""
    };
    let dual_decl = if epilogue.dual() {
        format!("    device {ty}* D2 [[buffer(10)]],\n")
    } else {
        String::new()
    };
    let act_fns = if epilogue.gelu_fn().is_some() {
        ACT_FNS
    } else {
        ""
    };
    let store = match (epilogue.gelu_fn(), epilogue.dual()) {
        (Some(g), true) => format!("D[d_idx] = {ty}(v);\n        D2[d_idx] = {ty}({g}(v));"),
        (Some(g), false) => format!("D[d_idx] = {ty}({g}(v));"),
        (None, _) => format!("D[d_idx] = {ty}(v);"),
    };
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;
{act_fns}
kernel void et_gemm(
    device const {ty}* A [[buffer(0)]],
    device const {ty}* B [[buffer(1)]],
    device {ty}* D [[buffer(2)]],
{bias_decl}{res_decl}{dual_decl}    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& strideA [[buffer(7)]],
    constant uint& strideB [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {{
    const uint i = tgid.y * {TILE} + tpitg.y;
    const uint j = tgid.x * {TILE} + tpitg.x;
    const uint batch = tgid.z;
    threadgroup float As[{TILE}][{TILE}];
    threadgroup float Bs[{TILE}][{TILE}];
    const ulong a_batch = (ulong)batch * strideA;
    const ulong b_batch = (ulong)batch * strideB;
    const ulong d_batch = (ulong)batch * M * N;
    float acc = 0.0f;
    for (uint t = 0; t < K; t += {TILE}) {{
        const uint ak = t + tpitg.x;
        const uint bk = t + tpitg.y;
        As[tpitg.y][tpitg.x] = (i < M && ak < K) ? float(A[a_batch + (ulong)i * K + ak]) : 0.0f;
        Bs[tpitg.y][tpitg.x] = (bk < K && j < N) ? float(B[b_batch + (ulong)bk * N + j]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = 0; p < {TILE}; ++p) {{
            acc += As[tpitg.y][p] * Bs[p][tpitg.x];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (i < M && j < N) {{
        const ulong d_idx = d_batch + (ulong)i * N + j;
        const float v = acc{bias_add}{res_add};
        {store}
    }}
}}
"#,
        TILE = TILE,
        ty = ty,
        act_fns = act_fns,
        bias_decl = bias_decl,
        bias_add = bias_add,
        res_decl = res_decl,
        res_add = res_add,
        dual_decl = dual_decl,
        store = store,
    )
}

fn key_for(bias: bool, epilogue: Epilogue, dtype: DType) -> u64 {
    let base = if bias { 0x6E11_B1A5 } else { 0x6E11_0000 };
    base ^ (dtype as u64) ^ (epilogue.code() << 32)
}

// simdgroup-MMA gemm: each threadgroup produces a T×T output tile
// with T/16 × (threads/32 ÷ T/16) simdgroups, each accumulating a
// 16×(T/2 ÷ ...) quadrant of 8×8 simdgroup matrices. The geometry is
// derived from the device's threadgroup memory, not hardcoded:
// 64×64 (20 KB, 8 simdgroups) where it fits, 32×32 (6 KB, 4
// simdgroups) on smaller chips. Inputs convert to f32 during the
// cooperative threadgroup load, so one template covers f32/bf16/f16;
// the epilogue (bias/residual/gelu/dual-store) runs from a threadgroup
// staging tile. The naive kernel stays for EFFECT_TORCH_NO_MMA A/B,
// small shapes, and reference.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct MmaConfig {
    tile: usize,
    threads: usize,
}

fn mma_config(dev: &MetalDevice) -> MmaConfig {
    static CONFIG: std::sync::OnceLock<MmaConfig> = std::sync::OnceLock::new();
    *CONFIG.get_or_init(|| {
        if dev.raw().maxThreadgroupMemoryLength() >= 20 * 1024 {
            MmaConfig {
                tile: 64,
                threads: 256,
            }
        } else {
            MmaConfig {
                tile: 32,
                threads: 128,
            }
        }
    })
}

/// Batched gemm problem size: `batch` independent `m×k · k×n` products.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GemmShape {
    pub batch: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

/// The kernel selected for a gemm shape (see the module docs).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GemmAlgorithm {
    /// 16×16 threadgroup-tiled reference kernel.
    Tiled,
    /// Simdgroup-MMA kernel with the given tile/threads geometry.
    SimdgroupMma { tile: usize, threads: usize },
    /// Split-K: MMA partials over `splits` K-slices plus a deterministic
    /// reduce kernel.
    SplitK {
        tile: usize,
        threads: usize,
        splits: usize,
    },
}

impl GemmAlgorithm {
    fn mma_config(self) -> Option<MmaConfig> {
        match self {
            GemmAlgorithm::Tiled => None,
            GemmAlgorithm::SimdgroupMma { tile, threads }
            | GemmAlgorithm::SplitK { tile, threads, .. } => Some(MmaConfig { tile, threads }),
        }
    }
}

/// Exact f32 partials buffer a split-K gemm needs: `[splits, batch, m, n]`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SplitKScratchRequirement {
    pub dtype: DType,
    pub shape: [usize; 4],
    pub elements: usize,
    pub bytes: usize,
}

/// Exact buffers and algorithm selected for one GEMM shape. Execution consumes
/// this value directly, so changing the process MMA environment after planning
/// cannot change the algorithm or its scratch requirement.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GemmRequirements {
    /// The batched problem shape.
    pub shape: GemmShape,
    /// Element type of every operand and output (f32/f16/bf16).
    pub dtype: DType,
    /// Whether a per-column bias vector is bound.
    pub has_bias: bool,
    /// The fused epilogue.
    pub epilogue: Epilogue,
    /// Whether MMA selection was enabled at planning time.
    pub mma: bool,
    /// The selected kernel and its geometry.
    pub algorithm: GemmAlgorithm,
    /// `batch * m * n`.
    pub output_elements: usize,
    /// `output_elements * dtype size`.
    pub output_bytes: usize,
    /// 1, or 2 for dual epilogues (plain + gelu outputs).
    pub output_count: usize,
    /// f32 partials buffer required by the split-K algorithm.
    pub split_k_scratch: Option<SplitKScratchRequirement>,
}

fn checked_product(values: &[usize], what: &str) -> Result<usize, String> {
    values.iter().try_fold(1usize, |product, &value| {
        product
            .checked_mul(value)
            .ok_or_else(|| format!("gemm: {what} size overflow"))
    })
}

fn dtype_name(dtype: DType) -> Result<&'static str, String> {
    match dtype {
        DType::F32 => Ok("float"),
        DType::F16 => Ok("half"),
        DType::BF16 => Ok("bfloat"),
        other => Err(format!("gemm: unsupported dtype {other:?}")),
    }
}

fn mma_candidate(
    dev: &MetalDevice,
    shape: GemmShape,
) -> Result<Option<(MmaConfig, usize)>, String> {
    const MIN_GROUPS: usize = 64;
    let big = mma_config(dev);
    let groups = |tile: usize| -> Result<usize, String> {
        checked_product(
            &[shape.m.div_ceil(tile), shape.n.div_ceil(tile), shape.batch],
            "threadgroup grid",
        )
    };
    let big_groups = groups(big.tile)?;
    if shape.m >= big.tile && shape.n >= big.tile && shape.k >= 32 && big_groups >= MIN_GROUPS {
        return Ok(Some((big, big_groups)));
    }
    if big.tile > 32 && shape.m >= 32 && shape.n >= 32 && shape.k >= 16 {
        let medium_groups = groups(32)?;
        if medium_groups >= MIN_GROUPS {
            return Ok(Some((
                MmaConfig {
                    tile: 32,
                    threads: 128,
                },
                medium_groups,
            )));
        }
    }
    Ok(None)
}

/// Plans a gemm: validates dtype/dimension limits, selects the algorithm
/// (MMA only when `mma` and the shape saturates the GPU; split-K only for
/// plain long-K narrow-output cases), and computes exact output/scratch
/// sizes. Pure — no allocation or compilation.
#[allow(clippy::too_many_arguments)]
pub fn gemm_requirements(
    dev: &MetalDevice,
    dtype: DType,
    has_bias: bool,
    epilogue: Epilogue,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    mma: bool,
) -> Result<GemmRequirements, String> {
    dtype_name(dtype)?;
    if m > u32::MAX as usize || n > u32::MAX as usize || k > u32::MAX as usize {
        return Err("gemm: M, N, and K must fit in u32".to_string());
    }
    let shape = GemmShape { batch, m, n, k };
    let output_elements = checked_product(&[batch, m, n], "output")?;
    let output_bytes = output_elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| "gemm: output byte size overflow".to_string())?;
    let candidate = if mma {
        mma_candidate(dev, shape)?
    } else {
        None
    };
    let algorithm = match candidate {
        Some((cfg, group_count))
            if !has_bias && epilogue == Epilogue::None && k >= 2048 && group_count < 256 =>
        {
            let splits = 2048usize
                .div_ceil(group_count)
                .clamp(1, 32)
                .min((k / 128).max(1));
            if splits > 1 {
                GemmAlgorithm::SplitK {
                    tile: cfg.tile,
                    threads: cfg.threads,
                    splits,
                }
            } else {
                GemmAlgorithm::SimdgroupMma {
                    tile: cfg.tile,
                    threads: cfg.threads,
                }
            }
        }
        Some((cfg, _)) => GemmAlgorithm::SimdgroupMma {
            tile: cfg.tile,
            threads: cfg.threads,
        },
        None => GemmAlgorithm::Tiled,
    };
    let split_k_scratch = match algorithm {
        GemmAlgorithm::SplitK { splits, .. } => {
            let elements = output_elements
                .checked_mul(splits)
                .ok_or_else(|| "gemm: split-K scratch size overflow".to_string())?;
            let bytes = elements
                .checked_mul(DType::F32.size_in_bytes())
                .ok_or_else(|| "gemm: split-K scratch byte size overflow".to_string())?;
            Some(SplitKScratchRequirement {
                dtype: DType::F32,
                shape: [splits, batch, m, n],
                elements,
                bytes,
            })
        }
        _ => None,
    };
    Ok(GemmRequirements {
        shape,
        dtype,
        has_bias,
        epilogue,
        mma,
        algorithm,
        output_elements,
        output_bytes,
        output_count: if epilogue.dual() { 2 } else { 1 },
        split_k_scratch,
    })
}

/// Plans a plain batched matmul from operand shapes (rank ≥ 2, inner dims
/// equal, batch dims broadcast-compatible).
pub fn matmul_requirements(
    dev: &MetalDevice,
    a_shape: &[usize],
    b_shape: &[usize],
    dtype: DType,
    mma: bool,
) -> Result<GemmRequirements, String> {
    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err("matmul needs rank >= 2".to_string());
    }
    let ar = a_shape.len();
    let br = b_shape.len();
    let (m, k) = (a_shape[ar - 2], a_shape[ar - 1]);
    let (k2, n) = (b_shape[br - 2], b_shape[br - 1]);
    if k != k2 {
        return Err(format!("matmul inner dim mismatch: {k} vs {k2}"));
    }
    let batch_a = checked_product(&a_shape[..ar - 2], "left batch")?;
    let batch_b = checked_product(&b_shape[..br - 2], "right batch")?;
    if batch_a != batch_b && batch_a != 1 && batch_b != 1 {
        return Err(format!("matmul batch mismatch: {batch_a} vs {batch_b}"));
    }
    gemm_requirements(
        dev,
        dtype,
        false,
        Epilogue::None,
        batch_a.max(batch_b),
        m,
        n,
        k,
        mma,
    )
}

fn gemm_mma_source(bias: bool, epilogue: Epilogue, ty: &str, cfg: MmaConfig) -> String {
    let bias_decl = if bias {
        format!("    device const {ty}* bias [[buffer(3)]],\n")
    } else {
        String::new()
    };
    let bias_add = if bias { " + float(bias[j])" } else { "" };
    let res_decl = if epilogue == Epilogue::Residual {
        format!("    device const {ty}* R [[buffer(9)]],\n")
    } else {
        String::new()
    };
    let res_add = if epilogue == Epilogue::Residual {
        " + float(R[d_idx])"
    } else {
        ""
    };
    let dual_decl = if epilogue.dual() {
        format!("    device {ty}* D2 [[buffer(10)]],\n")
    } else {
        String::new()
    };
    let act_fns = if epilogue.gelu_fn().is_some() {
        ACT_FNS
    } else {
        ""
    };
    let store = match (epilogue.gelu_fn(), epilogue.dual()) {
        (Some(g), true) => format!("D[d_idx] = {ty}(v);\n            D2[d_idx] = {ty}({g}(v));"),
        (Some(g), false) => format!("D[d_idx] = {ty}({g}(v));"),
        (None, _) => format!("D[d_idx] = {ty}(v);"),
    };
    let t = cfg.tile;
    let threads = cfg.threads;
    let sg_per_col = t / 16; // simdgroups stacked vertically
    let qw = t / (threads / 32 / sg_per_col); // quadrant width
    let dj = qw / 8;
    let load_n = t * 8;
    let store_n = t * t;
    // bf16/f16 stage and multiply natively (matrix units take the
    // reduced-precision inputs with an f32 accumulator); f32 stages as
    // f32. Native staging halves threadgroup traffic and skips the
    // conversion.
    let (stage_ty, sg_ty, zero) = match ty {
        "bfloat" => ("bfloat", "bfloat", "bfloat(0.0f)"),
        "half" => ("half", "half", "half(0.0h)"),
        _ => ("float", "float", "0.0f"),
    };
    let a_expr = "A[a_batch + (ulong)(m0 + r) * K + k0 + c]";
    let b_expr = "B[b_batch + (ulong)(k0 + r) * N + n0 + c]";
    let (a_load, b_load) = if sg_ty == "float" {
        (format!("float({a_expr})"), format!("float({b_expr})"))
    } else {
        (a_expr.to_string(), b_expr.to_string())
    };
    format!(
        r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
{act_fns}
kernel void et_gemm_mma(
    device const {ty}* A [[buffer(0)]],
    device const {ty}* B [[buffer(1)]],
    device {ty}* D [[buffer(2)]],
{bias_decl}{res_decl}{dual_decl}    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& strideA [[buffer(7)]],
    constant uint& strideB [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {{
    const uint m0 = tgid.y * {T}u;
    const uint n0 = tgid.x * {T}u;
    const uint batch = tgid.z;
    const ulong a_batch = (ulong)batch * strideA;
    const ulong b_batch = (ulong)batch * strideB;
    const ulong d_batch = (ulong)batch * M * N;
    threadgroup {STAGE_TY} As[{T}][8];
    threadgroup {STAGE_TY} Bs[8][{T}];
    const uint sg = tid / 32u;
    const uint qm = (sg % {SG_PER_COL}u) * 16u;
    const uint qn = (sg / {SG_PER_COL}u) * {QW}u;
    simdgroup_float8x8 acc[2][{DJ}];
    for (uint di = 0; di < 2u; di++)
        for (uint dj = 0; dj < {DJ}u; dj++)
            acc[di][dj] = simdgroup_float8x8(0.0f);
    for (uint k0 = 0; k0 < K; k0 += 8u) {{
        for (uint e = tid; e < {LOAD_N}u; e += {THREADS}u) {{
            const uint r = e / 8u, c = e % 8u;
            As[r][c] = (m0 + r < M && k0 + c < K) ? {A_LOAD} : {ZERO};
        }}
        for (uint e = tid; e < {LOAD_N}u; e += {THREADS}u) {{
            const uint r = e / {T}u, c = e % {T}u;
            Bs[r][c] = (k0 + r < K && n0 + c < N) ? {B_LOAD} : {ZERO};
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint di = 0; di < 2u; di++) {{
            simdgroup_{SG_TY}8x8 af;
            simdgroup_load(af, &As[qm + 8u * di][0], 8);
            for (uint dj = 0; dj < {DJ}u; dj++) {{
                simdgroup_{SG_TY}8x8 bf;
                simdgroup_load(bf, &Bs[0][qn + 8u * dj], {T});
                simdgroup_multiply_accumulate(acc[di][dj], af, bf, acc[di][dj]);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    threadgroup float Cs[{T}][{T}];
    for (uint di = 0; di < 2u; di++)
        for (uint dj = 0; dj < {DJ}u; dj++)
            simdgroup_store(acc[di][dj], &Cs[qm + 8u * di][qn + 8u * dj], {T});
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < {STORE_N}u; e += {THREADS}u) {{
        const uint r = e / {T}u, c = e % {T}u;
        const uint i = m0 + r, j = n0 + c;
        if (i < M && j < N) {{
            const ulong d_idx = d_batch + (ulong)i * N + j;
            const float v = Cs[r][c]{bias_add}{res_add};
            {store}
        }}
    }}
}}
"#,
        act_fns = act_fns,
        ty = ty,
        bias_decl = bias_decl,
        res_decl = res_decl,
        dual_decl = dual_decl,
        bias_add = bias_add,
        res_add = res_add,
        store = store,
        T = t,
        THREADS = threads,
        SG_PER_COL = sg_per_col,
        QW = qw,
        DJ = dj,
        LOAD_N = load_n,
        STORE_N = store_n,
        STAGE_TY = stage_ty,
        SG_TY = sg_ty,
        ZERO = zero,
        A_LOAD = a_load,
        B_LOAD = b_load,
    )
}

fn mma_key_for(bias: bool, epilogue: Epilogue, dtype: DType, cfg: MmaConfig) -> u64 {
    key_for(bias, epilogue, dtype) ^ 0xA11A_0000_0000 ^ ((cfg.tile as u64) << 48)
}

fn splitk_key(
    name: &'static str,
    dtype: DType,
    cfg: MmaConfig,
    splits: usize,
    total: usize,
) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    (
        name,
        dtype,
        cfg.tile,
        cfg.threads,
        splits,
        total,
        MetalDevice::WIDE,
    )
        .hash(&mut hasher);
    hasher.finish()
}

// Split-K for long-K narrow-output gemms (head-dX, trunk-dW): the
// output grid alone starves the GPU and every threadgroup re-reads all
// of A and B, so K is partitioned across threadgroups; each element
// is read once, writing f32 partials that a second kernel reduces in
// a fixed order (deterministic). Biases and epilogues are unsupported;
// only plain backward gemms take this path. Staging intentionally stays
// single-buffered: double buffering wins the head-dX microbench but
// loses 6.5% over 400 FineWeb steps on M4 Max due to thermal throttling.
fn gemm_splitk_source(
    ty: &str,
    sg_ty: &str,
    cfg: MmaConfig,
    splits: usize,
    total: usize,
) -> String {
    let t = cfg.tile;
    let threads = cfg.threads;
    let sg_per_col = t / 16;
    let qw = t / (threads / 32 / sg_per_col);
    let dj = qw / 8;
    let load_n = t * 8;
    let store_n = t * t;
    let zero = if sg_ty == "float" {
        "0.0f"
    } else if sg_ty == "bfloat" {
        "bfloat(0.0f)"
    } else {
        "half(0.0h)"
    };
    let a_expr = "A[a_batch + (ulong)(m0 + r) * K + kk + c]";
    let b_expr = "B[b_batch + (ulong)(kk + r) * N + n0 + c]";
    let (a_load, b_load) = if sg_ty == "float" {
        (format!("float({a_expr})"), format!("float({b_expr})"))
    } else {
        (a_expr.to_string(), b_expr.to_string())
    };
    format!(
        r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void et_gemm_splitk(
    device const {ty}* A [[buffer(0)]],
    device const {ty}* B [[buffer(1)]],
    device float* P [[buffer(2)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& strideA [[buffer(7)]],
    constant uint& strideB [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {{
    const uint slice = tgid.z % {SPLITS}u;
    const uint batch = tgid.z / {SPLITS}u;
    const uint m0 = tgid.y * {T}u;
    const uint n0 = tgid.x * {T}u;
    const ulong a_batch = (ulong)batch * strideA;
    const ulong b_batch = (ulong)batch * strideB;
    threadgroup {sg_ty} As[{T}][8];
    threadgroup {sg_ty} Bs[8][{T}];
    const uint sg = tid / 32u;
    const uint qm = (sg % {SG_PER_COL}u) * 16u;
    const uint qn = (sg / {SG_PER_COL}u) * {QW}u;
    const uint klen = (K + {SPLITS}u - 1u) / {SPLITS}u;
    const uint k_start = slice * klen;
    const uint k_end = min(K, k_start + klen);
    simdgroup_float8x8 acc[2][{DJ}];
    for (uint di = 0; di < 2u; di++)
        for (uint dj = 0; dj < {DJ}u; dj++)
            acc[di][dj] = simdgroup_float8x8(0.0f);
    for (uint k0 = k_start; k0 < k_end; k0 += 8u) {{
        const uint kk = k0;
        for (uint e = tid; e < {LOAD_N}u; e += {THREADS}u) {{
            const uint r = e / 8u, c = e % 8u;
            As[r][c] = (m0 + r < M && kk + c < k_end) ? {A_LOAD} : {ZERO};
        }}
        for (uint e = tid; e < {LOAD_N}u; e += {THREADS}u) {{
            const uint r = e / {T}u, c = e % {T}u;
            Bs[r][c] = (kk + r < k_end && n0 + c < N) ? {B_LOAD} : {ZERO};
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint di = 0; di < 2u; di++) {{
            simdgroup_{sg_ty}8x8 af;
            simdgroup_load(af, &As[qm + 8u * di][0], 8);
            for (uint dj = 0; dj < {DJ}u; dj++) {{
                simdgroup_{sg_ty}8x8 bf;
                simdgroup_load(bf, &Bs[0][qn + 8u * dj], {T});
                simdgroup_multiply_accumulate(acc[di][dj], af, bf, acc[di][dj]);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    threadgroup float Cs[{T}][{T}];
    for (uint di = 0; di < 2u; di++)
        for (uint dj = 0; dj < {DJ}u; dj++)
            simdgroup_store(acc[di][dj], &Cs[qm + 8u * di][qn + 8u * dj], {T});
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong p_base = (ulong)slice * {TOTAL}ul + (ulong)batch * M * N;
    for (uint e = tid; e < {STORE_N}u; e += {THREADS}u) {{
        const uint r = e / {T}u, c = e % {T}u;
        const uint i = m0 + r, j = n0 + c;
        if (i < M && j < N) {{
            P[p_base + (ulong)i * N + j] = Cs[r][c];
        }}
    }}
}}

kernel void et_gemm_splitk_reduce(
    device const float* P [[buffer(0)]],
    device {ty}* D [[buffer(1)]],
    uint2 gid2 [[thread_position_in_grid]]
) {{
    const ulong i = ulong(gid2.y) * {WIDE}ul + ulong(gid2.x);
    if (i < {TOTAL}ul) {{
        float acc = 0.0f;
        for (uint s = 0u; s < {SPLITS}u; s++) {{
            acc += P[(ulong)s * {TOTAL}ul + i];
        }}
        D[i] = {ty}(acc);
    }}
}}
"#,
        ty = ty,
        sg_ty = sg_ty,
        T = t,
        THREADS = threads,
        SG_PER_COL = sg_per_col,
        QW = qw,
        DJ = dj,
        LOAD_N = load_n,
        STORE_N = store_n,
        ZERO = zero,
        A_LOAD = a_load,
        B_LOAD = b_load,
        SPLITS = splits,
        WIDE = MetalDevice::WIDE,
        TOTAL = total,
    )
}

#[allow(clippy::too_many_arguments)]
fn gemm_splitk_into(
    dev: &MetalDevice,
    a: &MetalTensor,
    b: &MetalTensor,
    out: &MetalTensor,
    scratch: &MetalTensor,
    splits: usize,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    stride_a: usize,
    stride_b: usize,
    cfg: MmaConfig,
) -> Result<(), String> {
    let esz = a.dtype.size_in_bytes();
    let total = batch * m * n;
    let key = splitk_key("et_gemm_splitk", a.dtype, cfg, splits, total);
    let pipeline = dev.pipeline_cached(key).ok_or_else(|| {
        format!("gemm: split-K pipeline {key:#x} was not precompiled for exact requirements")
    })?;
    let rkey = splitk_key("et_gemm_splitk_reduce", a.dtype, cfg, splits, total);
    let rpipeline = dev.pipeline_cached(rkey).ok_or_else(|| {
        format!(
            "gemm: split-K reduce pipeline {rkey:#x} was not precompiled for exact requirements"
        )
    })?;
    let (mu, nu, ku) = (m as u32, n as u32, k as u32);
    let (sa, sb) = (stride_a as u32, stride_b as u32);
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &a.buffer, a.layout.offset() * esz);
        set_buffer(e, 1, &b.buffer, b.layout.offset() * esz);
        set_buffer(
            e,
            2,
            &scratch.buffer,
            scratch.layout.offset() * DType::F32.size_in_bytes(),
        );
        set_bytes(e, 4, &mu);
        set_bytes(e, 5, &nu);
        set_bytes(e, 6, &ku);
        set_bytes(e, 7, &sa);
        set_bytes(e, 8, &sb);
        e.dispatchThreadgroups_threadsPerThreadgroup(
            MetalDevice::grid(n.div_ceil(cfg.tile), m.div_ceil(cfg.tile), batch * splits),
            MetalDevice::grid(cfg.threads, 1, 1),
        );
    });
    let padded = total.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(rpipeline.as_raw());
        set_buffer(
            e,
            0,
            &scratch.buffer,
            scratch.layout.offset() * DType::F32.size_in_bytes(),
        );
        set_buffer(e, 1, &out.buffer, out.layout.offset() * esz);
        let (g, tg) = MetalDevice::grid_flat(padded);
        e.dispatchThreads_threadsPerThreadgroup(g, tg);
    });
    Ok(())
}

/// Compiles only the pipeline(s) selected in `requirements` and allocates no
/// tensor or scratch buffers.
pub fn precompile_gemm_fused(
    dev: &MetalDevice,
    requirements: &GemmRequirements,
) -> Result<usize, String> {
    if requirements.output_elements == 0 {
        return Ok(0);
    }
    let ty = dtype_name(requirements.dtype)?;
    match requirements.algorithm {
        GemmAlgorithm::Tiled => {
            dev.compile_lazy(
                key_for(
                    requirements.has_bias,
                    requirements.epilogue,
                    requirements.dtype,
                ),
                "et_gemm",
                || gemm_source(requirements.has_bias, requirements.epilogue, ty),
            )?;
            Ok(1)
        }
        GemmAlgorithm::SimdgroupMma { tile, threads } => {
            let cfg = MmaConfig { tile, threads };
            dev.compile_lazy(
                mma_key_for(
                    requirements.has_bias,
                    requirements.epilogue,
                    requirements.dtype,
                    cfg,
                ),
                "et_gemm_mma",
                || gemm_mma_source(requirements.has_bias, requirements.epilogue, ty, cfg),
            )?;
            Ok(1)
        }
        GemmAlgorithm::SplitK {
            tile,
            threads,
            splits,
        } => {
            let cfg = MmaConfig { tile, threads };
            let total = requirements.output_elements;
            let sg_ty = if requirements.dtype == DType::F32 {
                "float"
            } else {
                ty
            };
            let source = gemm_splitk_source(ty, sg_ty, cfg, splits, total);
            dev.compile_lazy(
                splitk_key("et_gemm_splitk", requirements.dtype, cfg, splits, total),
                "et_gemm_splitk",
                || source.clone(),
            )?;
            dev.compile_lazy(
                splitk_key(
                    "et_gemm_splitk_reduce",
                    requirements.dtype,
                    cfg,
                    splits,
                    total,
                ),
                "et_gemm_splitk_reduce",
                || source,
            )?;
            Ok(2)
        }
    }
}

/// [`precompile_gemm_fused`] under its former name.
pub fn precompile_gemm(
    dev: &MetalDevice,
    requirements: &GemmRequirements,
) -> Result<usize, String> {
    precompile_gemm_fused(dev, requirements)
}

/// [`precompile_gemm_fused`] for matmul requirements.
pub fn precompile_matmul(
    dev: &MetalDevice,
    requirements: &GemmRequirements,
) -> Result<usize, String> {
    precompile_gemm_fused(dev, requirements)
}

/// Plans and precompiles in one call; returns the pipeline count.
#[allow(clippy::too_many_arguments)]
pub fn warm_gemm_fused(
    dev: &MetalDevice,
    dtype: DType,
    has_bias: bool,
    epilogue: Epilogue,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    mma: bool,
) -> Result<usize, String> {
    let requirements = gemm_requirements(dev, dtype, has_bias, epilogue, batch, m, n, k, mma)?;
    precompile_gemm_fused(dev, &requirements)
}

fn matrix_span(batch: usize, stride: usize, elements: usize) -> Result<usize, String> {
    if batch == 0 || elements == 0 {
        return Ok(0);
    }
    (batch - 1)
        .checked_mul(stride)
        .and_then(|offset| offset.checked_add(elements))
        .ok_or_else(|| "gemm: input span overflow".to_string())
}

fn validate_contiguous(
    tensor: &MetalTensor,
    dtype: DType,
    elements: usize,
    exact: bool,
    name: &str,
) -> Result<(), String> {
    if tensor.dtype != dtype {
        return Err(format!(
            "gemm: {name} dtype mismatch, expected {dtype:?}, got {:?}",
            tensor.dtype
        ));
    }
    if !tensor.layout.is_contiguous() {
        return Err(format!("gemm: {name} must be contiguous"));
    }
    let actual = tensor
        .layout
        .checked_numel()
        .ok_or_else(|| format!("gemm: {name} element count overflow"))?;
    if (exact && actual != elements) || (!exact && actual < elements) {
        let relation = if exact { "exactly" } else { "at least" };
        return Err(format!(
            "gemm: {name} needs {relation} {elements} elements, got {actual}"
        ));
    }
    let bytes = elements
        .checked_mul(dtype.size_in_bytes())
        .and_then(|size| {
            tensor
                .layout
                .offset()
                .checked_mul(dtype.size_in_bytes())?
                .checked_add(size)
        })
        .ok_or_else(|| format!("gemm: {name} byte range overflow"))?;
    if bytes > tensor.buffer.size {
        return Err(format!(
            "gemm: {name} byte range {bytes} exceeds buffer size {}",
            tensor.buffer.size
        ));
    }
    Ok(())
}

/// Dispatches a fused gemm into caller-provided destinations, validating
/// every operand, output, and scratch buffer against `requirements` (which
/// must have been planned and precompiled for exactly these arguments).
/// Buffer bindings: A=0, B=1, D=2, bias=3, dims/strides=4–8, residual=9,
/// dual output=10. Performs no allocation.
#[allow(clippy::too_many_arguments)]
pub fn gemm_fused_into(
    dev: &MetalDevice,
    a: &MetalTensor,
    b: &MetalTensor,
    bias: Option<&MetalTensor>,
    residual: Option<&MetalTensor>,
    out: &MetalTensor,
    out2: Option<&MetalTensor>,
    split_k_scratch: Option<&MetalTensor>,
    stride_a: usize,
    stride_b: usize,
    requirements: &GemmRequirements,
) -> Result<(), String> {
    let GemmShape { batch, m, n, k } = requirements.shape;
    if a.dtype != requirements.dtype || b.dtype != requirements.dtype {
        return Err(format!(
            "gemm: input dtype mismatch for {:?} requirements",
            requirements.dtype
        ));
    }
    if bias.is_some() != requirements.has_bias {
        return Err("gemm: bias presence does not match requirements".to_string());
    }
    if residual.is_some() != (requirements.epilogue == Epilogue::Residual) {
        return Err("gemm: residual presence does not match epilogue requirements".to_string());
    }
    if out2.is_some() != (requirements.output_count == 2) {
        return Err("gemm: secondary output presence does not match requirements".to_string());
    }
    if stride_a > u32::MAX as usize || stride_b > u32::MAX as usize {
        return Err("gemm: batch strides must fit in u32".to_string());
    }
    match (requirements.split_k_scratch, split_k_scratch) {
        (Some(expected), Some(actual)) => validate_contiguous(
            actual,
            expected.dtype,
            expected.elements,
            true,
            "split-K scratch",
        )?,
        (Some(_), None) => return Err("gemm: exact split-K scratch buffer is required".to_string()),
        (None, Some(_)) => {
            return Err("gemm: scratch supplied for an algorithm that needs none".to_string())
        }
        (None, None) => {}
    }
    let a_matrix = m
        .checked_mul(k)
        .ok_or_else(|| "gemm: left matrix size overflow".to_string())?;
    let b_matrix = k
        .checked_mul(n)
        .ok_or_else(|| "gemm: right matrix size overflow".to_string())?;
    validate_contiguous(
        a,
        requirements.dtype,
        matrix_span(batch, stride_a, a_matrix)?,
        false,
        "left input",
    )?;
    validate_contiguous(
        b,
        requirements.dtype,
        matrix_span(batch, stride_b, b_matrix)?,
        false,
        "right input",
    )?;
    if let Some(bias) = bias {
        validate_contiguous(bias, requirements.dtype, n, true, "bias")?;
    }
    if let Some(residual) = residual {
        validate_contiguous(
            residual,
            requirements.dtype,
            requirements.output_elements,
            true,
            "residual",
        )?;
    }
    validate_contiguous(
        out,
        requirements.dtype,
        requirements.output_elements,
        true,
        "destination",
    )?;
    if let Some(out2) = out2 {
        validate_contiguous(
            out2,
            requirements.dtype,
            requirements.output_elements,
            true,
            "secondary destination",
        )?;
    }
    dev.mark_buffer_write(&out.buffer)?;
    if let Some(out2) = out2 {
        dev.mark_buffer_write(&out2.buffer)?;
    }
    if requirements.output_elements == 0 {
        return Ok(());
    }

    if let GemmAlgorithm::SplitK { splits, .. } = requirements.algorithm {
        return gemm_splitk_into(
            dev,
            a,
            b,
            out,
            split_k_scratch.expect("validated split-K scratch"),
            splits,
            batch,
            m,
            n,
            k,
            stride_a,
            stride_b,
            requirements
                .algorithm
                .mma_config()
                .expect("split-K has MMA config"),
        );
    }

    let cfg = requirements.algorithm.mma_config();
    let (key, name) = match cfg {
        Some(cfg) => (
            mma_key_for(
                requirements.has_bias,
                requirements.epilogue,
                requirements.dtype,
                cfg,
            ),
            "et_gemm_mma",
        ),
        None => (
            key_for(
                requirements.has_bias,
                requirements.epilogue,
                requirements.dtype,
            ),
            "et_gemm",
        ),
    };
    let pipeline = dev.pipeline_cached(key).ok_or_else(|| {
        format!("gemm: {name} pipeline {key:#x} was not precompiled for exact requirements")
    })?;
    let esz = requirements.dtype.size_in_bytes();
    let (mu, nu, ku) = (m as u32, n as u32, k as u32);
    let (sa, sb) = (stride_a as u32, stride_b as u32);
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &a.buffer, a.layout.offset() * esz);
        set_buffer(e, 1, &b.buffer, b.layout.offset() * esz);
        set_buffer(e, 2, &out.buffer, out.layout.offset() * esz);
        if let Some(bias) = bias {
            set_buffer(e, 3, &bias.buffer, bias.layout.offset() * esz);
        }
        set_bytes(e, 4, &mu);
        set_bytes(e, 5, &nu);
        set_bytes(e, 6, &ku);
        set_bytes(e, 7, &sa);
        set_bytes(e, 8, &sb);
        if let Some(residual) = residual {
            set_buffer(e, 9, &residual.buffer, residual.layout.offset() * esz);
        }
        if let Some(out2) = out2 {
            set_buffer(e, 10, &out2.buffer, out2.layout.offset() * esz);
        }
        if let Some(cfg) = cfg {
            e.dispatchThreadgroups_threadsPerThreadgroup(
                MetalDevice::grid(n.div_ceil(cfg.tile), m.div_ceil(cfg.tile), batch),
                MetalDevice::grid(cfg.threads, 1, 1),
            );
        } else {
            e.dispatchThreadgroups_threadsPerThreadgroup(
                MetalDevice::grid(n.div_ceil(TILE), m.div_ceil(TILE), batch),
                MetalDevice::grid(TILE, TILE, 1),
            );
        }
    });
    Ok(())
}

/// [`gemm_fused_into`] restricted to the plain single-output epilogue.
#[allow(clippy::too_many_arguments)]
pub fn gemm_into(
    dev: &MetalDevice,
    a: &MetalTensor,
    b: &MetalTensor,
    bias: Option<&MetalTensor>,
    out: &MetalTensor,
    split_k_scratch: Option<&MetalTensor>,
    stride_a: usize,
    stride_b: usize,
    requirements: &GemmRequirements,
) -> Result<(), String> {
    if requirements.epilogue != Epilogue::None || requirements.output_count != 1 {
        return Err("gemm_into requires a plain, single-output epilogue".to_string());
    }
    gemm_fused_into(
        dev,
        a,
        b,
        bias,
        None,
        out,
        None,
        split_k_scratch,
        stride_a,
        stride_b,
        requirements,
    )
}

/// Allocating fused gemm: plans, precompiles, allocates output(s) and any
/// split-K scratch, and dispatches.
#[allow(clippy::too_many_arguments)]
pub fn gemm_fused(
    dev: &MetalDevice,
    a: &MetalTensor,
    b: &MetalTensor,
    bias: Option<&MetalTensor>,
    residual: Option<&MetalTensor>,
    epilogue: Epilogue,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    stride_a: usize,
    stride_b: usize,
) -> Result<(MetalTensor, Option<MetalTensor>), String> {
    let requirements = gemm_requirements(
        dev,
        a.dtype,
        bias.is_some(),
        epilogue,
        batch,
        m,
        n,
        k,
        super::device::mma_enabled(),
    )?;
    precompile_gemm_fused(dev, &requirements)?;
    let shape = vec![batch, m, n];
    let out = MetalTensor::empty(dev, shape.clone(), a.dtype);
    let out2 = (requirements.output_count == 2).then(|| MetalTensor::empty(dev, shape, a.dtype));
    let scratch = requirements
        .split_k_scratch
        .map(|scratch| MetalTensor::empty(dev, scratch.shape.to_vec(), scratch.dtype));
    gemm_fused_into(
        dev,
        a,
        b,
        bias,
        residual,
        &out,
        out2.as_ref(),
        scratch.as_ref(),
        stride_a,
        stride_b,
        &requirements,
    )?;
    Ok((out, out2))
}

/// Allocating plain gemm with optional bias.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    dev: &MetalDevice,
    a: &MetalTensor,
    b: &MetalTensor,
    bias: Option<&MetalTensor>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    stride_a: usize,
    stride_b: usize,
) -> Result<MetalTensor, String> {
    gemm_fused(
        dev,
        a,
        b,
        bias,
        None,
        Epilogue::None,
        batch,
        m,
        n,
        k,
        stride_a,
        stride_b,
    )
    .map(|(out, _)| out)
}

/// Batched matmul into a caller-provided destination: validates the exact
/// requirements, derives batch strides (0 when that operand broadcasts),
/// and dispatches the plain epilogue.
pub fn matmul_into(
    dev: &MetalDevice,
    a: &MetalTensor,
    b: &MetalTensor,
    out: &MetalTensor,
    split_k_scratch: Option<&MetalTensor>,
    requirements: &GemmRequirements,
) -> Result<(), String> {
    let a_shape = a.layout.shape();
    let b_shape = b.layout.shape();
    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err("matmul needs rank >= 2".to_string());
    }
    let ar = a_shape.len();
    let br = b_shape.len();
    let batch_a = checked_product(&a_shape[..ar - 2], "left batch")?;
    let batch_b = checked_product(&b_shape[..br - 2], "right batch")?;
    let actual_shape = GemmShape {
        batch: batch_a.max(batch_b),
        m: a_shape[ar - 2],
        n: b_shape[br - 1],
        k: a_shape[ar - 1],
    };
    if a.dtype != b.dtype
        || requirements.dtype != a.dtype
        || requirements.shape != actual_shape
        || requirements.has_bias
        || requirements.epilogue != Epilogue::None
    {
        return Err("matmul: inputs do not match exact requirements".to_string());
    }
    if a_shape[ar - 1] != b_shape[br - 2] || (batch_a != batch_b && batch_a != 1 && batch_b != 1) {
        return Err("matmul: input shapes are incompatible".to_string());
    }
    let output_batch_shape = if batch_a >= batch_b {
        &a_shape[..ar - 2]
    } else {
        &b_shape[..br - 2]
    };
    let out_shape = out.layout.shape();
    if out_shape.len() != output_batch_shape.len() + 2
        || &out_shape[..output_batch_shape.len()] != output_batch_shape
        || out_shape[out_shape.len() - 2] != actual_shape.m
        || out_shape[out_shape.len() - 1] != actual_shape.n
    {
        return Err("matmul: destination shape does not match broadcast output".to_string());
    }
    let stride_a = if batch_a == 1 {
        0
    } else {
        actual_shape
            .m
            .checked_mul(actual_shape.k)
            .ok_or_else(|| "matmul: left stride overflow".to_string())?
    };
    let stride_b = if batch_b == 1 {
        0
    } else {
        actual_shape
            .k
            .checked_mul(actual_shape.n)
            .ok_or_else(|| "matmul: right stride overflow".to_string())?
    };
    gemm_into(
        dev,
        a,
        b,
        None,
        out,
        split_k_scratch,
        stride_a,
        stride_b,
        requirements,
    )
}

/// Allocating batched matmul with batch-dimension broadcasting.
pub fn matmul(dev: &MetalDevice, a: &MetalTensor, b: &MetalTensor) -> Result<MetalTensor, String> {
    let requirements = matmul_requirements(
        dev,
        a.layout.shape(),
        b.layout.shape(),
        a.dtype,
        super::device::mma_enabled(),
    )?;
    precompile_matmul(dev, &requirements)?;
    let ar = a.layout.shape().len();
    let br = b.layout.shape().len();
    let batch_a = checked_product(&a.layout.shape()[..ar - 2], "left batch")?;
    let batch_b = checked_product(&b.layout.shape()[..br - 2], "right batch")?;
    let mut out_shape = if batch_a >= batch_b {
        a.layout.shape()[..ar - 2].to_vec()
    } else {
        b.layout.shape()[..br - 2].to_vec()
    };
    out_shape.extend([requirements.shape.m, requirements.shape.n]);
    let out = MetalTensor::empty(dev, out_shape, a.dtype);
    let scratch = requirements
        .split_k_scratch
        .map(|scratch| MetalTensor::empty(dev, scratch.shape.to_vec(), scratch.dtype));
    matmul_into(dev, a, b, &out, scratch.as_ref(), &requirements)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_matches_cpu() {
        let dev = MetalDevice::get();
        let m = 37usize;
        let n = 53usize;
        let k = 29usize;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.19).cos()).collect();
        let ta = MetalTensor::from_f32(dev, a.clone(), vec![m, k]);
        let tb = MetalTensor::from_f32(dev, b.clone(), vec![k, n]);
        let out = gemm(dev, &ta, &tb, None, 1, m, n, k, m * k, k * n).unwrap();
        dev.synchronize().unwrap();
        let got = out.read_f32().unwrap();
        let mut want = vec![0f32; m * n];
        // SAFETY: `a`, `b`, and `want` are live vectors of exactly the
        // m*k, k*n, and m*n elements the sgemm signature reads/writes with
        // the given row-major leading dimensions.
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                n as isize,
                1,
                0.0,
                want.as_mut_ptr(),
                n as isize,
                1,
            );
        }
        for (x, y) in got.iter().zip(&want) {
            assert!((x - y).abs() / y.abs().max(1.0) < 1e-4, "{x} vs {y}");
        }
    }

    #[test]
    fn gemm_requirements_report_exact_splitk_scratch() {
        let dev = MetalDevice::get();
        let requirements = gemm_requirements(
            dev,
            DType::BF16,
            false,
            Epilogue::None,
            2,
            128,
            256,
            2051,
            true,
        )
        .unwrap();
        assert_eq!(
            requirements.algorithm,
            GemmAlgorithm::SplitK {
                tile: 32,
                threads: 128,
                splits: 16,
            }
        );
        let scratch = requirements.split_k_scratch.unwrap();
        assert_eq!(scratch.dtype, DType::F32);
        assert_eq!(scratch.shape, [16, 2, 128, 256]);
        assert_eq!(scratch.elements, 16 * 2 * 128 * 256);
        assert_eq!(scratch.bytes, scratch.elements * 4);
        assert_eq!(
            requirements,
            gemm_requirements(
                dev,
                DType::BF16,
                false,
                Epilogue::None,
                2,
                128,
                256,
                2051,
                true,
            )
            .unwrap()
        );

        for (has_bias, epilogue, outputs) in [
            (true, Epilogue::None, 1),
            (true, Epilogue::Residual, 1),
            (true, Epilogue::GeluErf, 1),
            (true, Epilogue::GeluTanhDual, 2),
        ] {
            let requirements = gemm_requirements(
                dev,
                DType::BF16,
                has_bias,
                epilogue,
                2,
                128,
                256,
                2051,
                true,
            )
            .unwrap();
            assert!(requirements.split_k_scratch.is_none());
            assert_eq!(requirements.output_count, outputs);
        }

        let f32_requirements = gemm_requirements(
            dev,
            DType::F32,
            false,
            Epilogue::None,
            2,
            128,
            256,
            2051,
            true,
        )
        .unwrap();
        let too_small = MetalTensor::empty(dev, vec![1], DType::F32);
        let error = gemm_into(
            dev,
            &too_small,
            &too_small,
            None,
            &too_small,
            Some(&too_small),
            0,
            0,
            &f32_requirements,
        )
        .unwrap_err();
        assert!(error.contains("split-K scratch needs exactly"), "{error}");
        let error = gemm_into(
            dev,
            &too_small,
            &too_small,
            None,
            &too_small,
            None,
            0,
            0,
            &f32_requirements,
        )
        .unwrap_err();
        assert!(error.contains("exact split-K scratch"), "{error}");
    }

    #[test]
    fn plain_matmul_into_matches_allocating_wrapper() {
        let dev = MetalDevice::get();
        let (m, n, k) = (13usize, 17usize, 11usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.13).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.29).cos()).collect();
        let ta = MetalTensor::from_f32(dev, a, vec![m, k]);
        let tb = MetalTensor::from_f32(dev, b, vec![k, n]);
        let requirements = matmul_requirements(dev, &[m, k], &[k, n], DType::F32, false).unwrap();
        precompile_matmul(dev, &requirements).unwrap();
        let destination = MetalTensor::empty(dev, vec![m, n], DType::F32);
        matmul_into(dev, &ta, &tb, &destination, None, &requirements).unwrap();
        let allocating = matmul(dev, &ta, &tb).unwrap();
        dev.synchronize().unwrap();
        let got = destination.read_f32().unwrap();
        let want = allocating.read_f32().unwrap();
        for (got, want) in got.iter().zip(&want) {
            assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        }
    }

    #[test]
    fn gemm_into_requires_precompile_and_allocates_no_device_buffer() {
        let dev = MetalDevice::new(0).unwrap();
        let (m, n, k) = (8usize, 9usize, 7usize);
        let a = MetalTensor::from_f32(&dev, vec![0.25; m * k], vec![m, k]);
        let b = MetalTensor::from_f32(&dev, vec![0.5; k * n], vec![k, n]);
        let out = MetalTensor::empty(&dev, vec![1, m, n], DType::F32);
        let requirements =
            gemm_requirements(&dev, DType::F32, false, Epilogue::None, 1, m, n, k, false).unwrap();
        let error =
            gemm_into(&dev, &a, &b, None, &out, None, m * k, k * n, &requirements).unwrap_err();
        assert!(error.contains("not precompiled"), "{error}");

        precompile_gemm(&dev, &requirements).unwrap();
        let _dispatch_guard = dev.begin_executable_dispatch().unwrap();
        let result = gemm_into(&dev, &a, &b, None, &out, None, m * k, k * n, &requirements);
        result.unwrap();
        dev.synchronize().unwrap();
    }

    #[test]
    fn gemm_bias_and_batch() {
        let dev = MetalDevice::get();
        let (batch, m, n, k) = (3usize, 8usize, 8usize, 8usize);
        let a = vec![1f32; batch * m * k];
        let b = vec![0.5f32; k * n];
        let bias: Vec<f32> = (0..n).map(|j| j as f32).collect();
        let ta = MetalTensor::from_f32(dev, a, vec![batch, m, k]);
        let tb = MetalTensor::from_f32(dev, b, vec![k, n]);
        let tbias = MetalTensor::from_f32(dev, bias, vec![n]);
        let out = gemm(dev, &ta, &tb, Some(&tbias), batch, m, n, k, m * k, 0).unwrap();
        dev.synchronize().unwrap();
        let got = out.read_f32().unwrap();
        for (i, v) in got.iter().enumerate() {
            let j = i % n;
            assert_eq!(*v, 8.0 * 0.5 + j as f32, "index {i}");
        }
    }

    fn erf_as(x: f32) -> f32 {
        let ax = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * ax);
        let mut p = 1.061405429f32;
        p = p * t - 1.453152027;
        p = p * t + 1.421413741;
        p = p * t - 0.284496736;
        p = p * t + 0.254829592;
        let tail = 1.0 - p * t * (-x * x).exp();
        x.signum() * tail
    }

    fn gelu_erf(x: f32) -> f32 {
        0.5 * x * (1.0 + erf_as(x * std::f32::consts::FRAC_1_SQRT_2))
    }

    fn gelu_tanh(x: f32) -> f32 {
        let u = 0.7978845608028654f32 * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + u.tanh())
    }

    #[test]
    fn gemm_residual_epilogue() {
        let dev = MetalDevice::get();
        let (batch, m, n, k) = (2usize, 17usize, 23usize, 11usize);
        let a: Vec<f32> = (0..batch * m * k)
            .map(|i| (i as f32 * 0.31).sin())
            .collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.17).cos()).collect();
        let bias: Vec<f32> = (0..n).map(|j| j as f32 * 0.5).collect();
        let r: Vec<f32> = (0..batch * m * n)
            .map(|i| (i as f32 * 0.07).sin() * 2.0)
            .collect();
        let ta = MetalTensor::from_f32(dev, a.clone(), vec![batch, m, k]);
        let tb = MetalTensor::from_f32(dev, b.clone(), vec![k, n]);
        let tbias = MetalTensor::from_f32(dev, bias.clone(), vec![n]);
        let tr = MetalTensor::from_f32(dev, r.clone(), vec![batch, m, n]);
        let (out, extra) = gemm_fused(
            dev,
            &ta,
            &tb,
            Some(&tbias),
            Some(&tr),
            Epilogue::Residual,
            batch,
            m,
            n,
            k,
            m * k,
            0,
        )
        .unwrap();
        assert!(extra.is_none());
        dev.synchronize().unwrap();
        let got = out.read_f32().unwrap();
        for bi in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut acc = bias[j];
                    for p in 0..k {
                        acc += a[bi * m * k + i * k + p] * b[p * n + j];
                    }
                    let want = acc + r[bi * m * n + i * n + j];
                    let got = got[bi * m * n + i * n + j];
                    assert!(
                        (got - want).abs() / want.abs().max(1.0) < 1e-4,
                        "{got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn gemm_mma_bf16_matches_f32() {
        let dev = MetalDevice::get();
        // Exactly 64 32x32 threadgroups: enough to force the normal MMA path.
        let (batch, m, n, k) = (16usize, 37usize, 53usize, 37usize);
        let a: Vec<f32> = (0..batch * m * k)
            .map(|i| (i as f32 * 0.37).sin())
            .collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.19).cos()).collect();
        let to_bf16 = |v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|x| half::bf16::from_f32(*x).to_bits().to_le_bytes())
                .collect()
        };
        let from_bytes = |bytes: Vec<u8>, shape: Vec<usize>| MetalTensor {
            buffer: dev.upload_bytes(&bytes),
            layout: crate::runtime::layout::Layout::contiguous(shape),
            dtype: DType::BF16,
        };
        let ta32 = MetalTensor::from_f32(dev, a.clone(), vec![batch, m, k]);
        let tb32 = MetalTensor::from_f32(dev, b.clone(), vec![k, n]);
        let tab = from_bytes(to_bf16(&a), vec![batch, m, k]);
        let tbb = from_bytes(to_bf16(&b), vec![k, n]);
        let out32 = gemm(dev, &ta32, &tb32, None, batch, m, n, k, m * k, 0).unwrap();
        let outb = gemm(dev, &tab, &tbb, None, batch, m, n, k, m * k, 0).unwrap();
        let cfg = MmaConfig {
            tile: 32,
            threads: 128,
        };
        assert_eq!(
            dev.pipeline_cached(mma_key_for(false, Epilogue::None, DType::BF16, cfg))
                .is_some(),
            std::env::var_os("EFFECT_TORCH_NO_MMA").is_none()
        );
        dev.synchronize().unwrap();
        let gotb: Vec<f32> = {
            let n = outb.numel();
            let ptr = outb.buffer.contents_ptr().cast::<u16>();
            // SAFETY: synchronized above; the bf16 buffer holds `n` u16
            // elements and shared-mode contents are host-readable.
            let bits = unsafe { std::slice::from_raw_parts(ptr, n) };
            bits.iter()
                .map(|b| half::bf16::from_bits(*b).to_f32())
                .collect()
        };
        // Reference: bf16-rounded inputs, f32 dot, bf16-rounded output.
        let r = |x: f32| half::bf16::from_f32(x).to_f32();
        for bi in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let dot: f32 = (0..k)
                        .map(|p| r(a[bi * m * k + i * k + p]) * r(b[p * n + j]))
                        .sum();
                    let want = r(dot);
                    let got = gotb[bi * m * n + i * n + j];
                    assert!(
                        (got - want).abs() / want.abs().max(1.0) < 1e-2,
                        "bf16 at [{bi},{i},{j}]: {got} vs {want}"
                    );
                }
            }
        }
        let _ = out32;
    }

    #[test]
    fn gemm_splitk_and_bias_variant_match_cpu() {
        // The 32x32 MMA grid has exactly 64 threadgroups, enough to
        // select MMA but still narrow enough to select split-K.
        let dev = MetalDevice::get();
        let (batch, m, n, k) = (2usize, 128usize, 256usize, 2051usize);
        let a: Vec<f32> = (0..batch * m * k)
            .map(|i| (i as f32 * 0.001).sin() * 0.5)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| (i as f32 * 0.0017).cos() * 0.5)
            .collect();
        let bias: Vec<f32> = (0..n).map(|j| j as f32 * 0.01).collect();
        let ta = MetalTensor::from_f32(dev, a.clone(), vec![batch, m, k]);
        let tb = MetalTensor::from_f32(dev, b.clone(), vec![k, n]);
        let tbias = MetalTensor::from_f32(dev, bias.clone(), vec![n]);
        let out = gemm(dev, &ta, &tb, None, batch, m, n, k, m * k, 0).unwrap();
        let cfg = MmaConfig {
            tile: 32,
            threads: 128,
        };
        let use_mma = std::env::var_os("EFFECT_TORCH_NO_MMA").is_none();
        assert_eq!(
            dev.pipeline_cached(splitk_key(
                "et_gemm_splitk",
                DType::F32,
                cfg,
                16,
                batch * m * n
            ))
            .is_some(),
            use_mma
        );
        let biased = gemm(dev, &ta, &tb, Some(&tbias), batch, m, n, k, m * k, 0).unwrap();
        assert_eq!(
            dev.pipeline_cached(mma_key_for(true, Epilogue::None, DType::F32, cfg))
                .is_some(),
            use_mma
        );
        dev.synchronize().unwrap();
        let got = out.read_f32().unwrap();
        let got_biased = biased.read_f32().unwrap();
        for (bi, i, j) in [
            (0, 0, 0),
            (0, 63, 255),
            (1, 0, 255),
            (1, 127, 0),
            (1, 127, 255),
        ] {
            let want: f32 = (0..k)
                .map(|p| a[bi * m * k + i * k + p] * b[p * n + j])
                .sum();
            let index = bi * m * n + i * n + j;
            let actual = got[index];
            assert!(
                (actual - want).abs() / want.abs().max(1.0) < 1e-3,
                "splitk[{bi},{i},{j}]: {actual} vs {want}"
            );
            let actual_biased = got_biased[index];
            let want_biased = want + bias[j];
            assert!(
                (actual_biased - want_biased).abs() / want_biased.abs().max(1.0) < 1e-3,
                "biased[{bi},{i},{j}]: {actual_biased} vs {want_biased}"
            );
        }
    }

    #[test]
    fn gemm_splitk_keys_separate_pipeline_sources() {
        let cfg = MmaConfig {
            tile: 32,
            threads: 128,
        };
        let f32_main = splitk_key("et_gemm_splitk", DType::F32, cfg, 16, 65_536);
        let f16_main = splitk_key("et_gemm_splitk", DType::F16, cfg, 16, 65_536);
        let f32_reduce = splitk_key("et_gemm_splitk_reduce", DType::F32, cfg, 16, 65_536);
        assert_ne!(f32_main, f16_main);
        assert_ne!(f32_main, f32_reduce);
    }

    #[test]
    fn gemm_gelu_epilogues() {
        let dev = MetalDevice::get();
        let (m, n, k) = (19usize, 29usize, 13usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.23).sin() * 1.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.11).cos()).collect();
        let bias: Vec<f32> = (0..n).map(|j| j as f32 * 0.25 - 3.0).collect();
        let ta = MetalTensor::from_f32(dev, a.clone(), vec![m, k]);
        let tb = MetalTensor::from_f32(dev, b.clone(), vec![k, n]);
        let tbias = MetalTensor::from_f32(dev, bias.clone(), vec![n]);
        let pre: Vec<f32> = (0..m * n)
            .map(|idx| {
                let (i, j) = (idx / n, idx % n);
                bias[j] + (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum::<f32>()
            })
            .collect();
        for (epilogue, gelu, dual) in [
            (Epilogue::GeluErf, gelu_erf as fn(f32) -> f32, false),
            (Epilogue::GeluTanh, gelu_tanh as fn(f32) -> f32, false),
            (Epilogue::GeluErfDual, gelu_erf as fn(f32) -> f32, true),
            (Epilogue::GeluTanhDual, gelu_tanh as fn(f32) -> f32, true),
        ] {
            let (out, out2) = gemm_fused(
                dev,
                &ta,
                &tb,
                Some(&tbias),
                None,
                epilogue,
                1,
                m,
                n,
                k,
                m * k,
                0,
            )
            .unwrap();
            dev.synchronize().unwrap();
            let (pre_buf, gelu_buf) = if dual {
                (
                    Some(out.read_f32().unwrap()),
                    out2.unwrap().read_f32().unwrap(),
                )
            } else {
                assert!(out2.is_none());
                (None, out.read_f32().unwrap())
            };
            for idx in 0..m * n {
                let want_g = gelu(pre[idx]);
                let got_g = gelu_buf[idx];
                assert!(
                    (got_g - want_g).abs() / want_g.abs().max(1.0) < 1e-3,
                    "{epilogue:?} gelu: {got_g} vs {want_g}"
                );
                if let Some(pre_buf) = &pre_buf {
                    let got_m = pre_buf[idx];
                    assert!(
                        (got_m - pre[idx]).abs() / pre[idx].abs().max(1.0) < 1e-4,
                        "{epilogue:?} pre: {got_m} vs {}",
                        pre[idx]
                    );
                }
            }
        }
    }
}

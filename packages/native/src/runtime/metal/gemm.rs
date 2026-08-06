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
    None,
    Residual,
    GeluErf,
    GeluTanh,
    GeluErfDual,
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
    let act_fns = if epilogue.gelu_fn().is_some() { ACT_FNS } else { "" };
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
#[derive(Clone, Copy, PartialEq, Eq)]
struct MmaConfig {
    tile: usize,
    threads: usize,
}

fn mma_config(dev: &MetalDevice) -> MmaConfig {
    static CONFIG: std::sync::OnceLock<MmaConfig> = std::sync::OnceLock::new();
    *CONFIG.get_or_init(|| {
        if dev.raw().maxThreadgroupMemoryLength() >= 20 * 1024 {
            MmaConfig { tile: 64, threads: 256 }
        } else {
            MmaConfig { tile: 32, threads: 128 }
        }
    })
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
    let act_fns = if epilogue.gelu_fn().is_some() { ACT_FNS } else { "" };
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
    threadgroup float As[{T}][8];
    threadgroup float Bs[8][{T}];
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
            As[r][c] = (m0 + r < M && k0 + c < K) ? float(A[a_batch + (ulong)(m0 + r) * K + k0 + c]) : 0.0f;
        }}
        for (uint e = tid; e < {LOAD_N}u; e += {THREADS}u) {{
            const uint r = e / {T}u, c = e % {T}u;
            Bs[r][c] = (k0 + r < K && n0 + c < N) ? float(B[b_batch + (ulong)(k0 + r) * N + n0 + c]) : 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint di = 0; di < 2u; di++) {{
            simdgroup_float8x8 af;
            simdgroup_load(af, &As[qm + 8u * di][0], 8);
            for (uint dj = 0; dj < {DJ}u; dj++) {{
                simdgroup_float8x8 bf;
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
    )
}

fn mma_key_for(bias: bool, epilogue: Epilogue, dtype: DType, cfg: MmaConfig) -> u64 {
    key_for(bias, epilogue, dtype) ^ 0xA11A_0000_0000 ^ ((cfg.tile as u64) << 48)
}

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
    assert_eq!(a.dtype, b.dtype, "gemm: dtype mismatch");
    assert!(
        matches!(a.dtype, DType::F32 | DType::F16 | DType::BF16),
        "gemm: unsupported dtype {:?}",
        a.dtype
    );
    if let Some(bias) = bias {
        assert_eq!(bias.dtype, a.dtype, "gemm: bias dtype mismatch");
    }
    if let Some(r) = residual {
        assert_eq!(r.dtype, a.dtype, "gemm: residual dtype mismatch");
        assert_eq!(epilogue, Epilogue::Residual, "gemm: residual without residual epilogue");
    }
    assert_eq!(
        (epilogue == Epilogue::Residual),
        residual.is_some(),
        "gemm: residual epilogue needs a residual tensor"
    );
    let esz = a.dtype.size_in_bytes();
    let ty = match a.dtype {
        DType::F32 => "float",
        DType::F16 => "half",
        DType::BF16 => "bfloat",
        _ => unreachable!(),
    };
    let has_bias = bias.is_some();
    // MMA pays when the grid fills the GPU AND K is long enough to
    // amortize the cooperative loads — measured on M4 Max: below ~64
    // threadgroups the naive kernel wins (N=128/256 starve at 4–16
    // groups), above it MMA pulls ahead (N=512 at 64 groups, 1.3x;
    // N=4096, 3x). Two bands: the device's preferred geometry
    // (mma_config) for full-size gemms, 32x32 for medium ones; truly
    // small gemms run the naive kernel.
    const MIN_GROUPS: usize = 64;
    let big = mma_config(dev);
    let groups = |tile: usize| (m.div_ceil(tile)) * (n.div_ceil(tile)) * batch;
    let medium = MmaConfig { tile: 32, threads: 128 };
    let cfg = if m >= big.tile && n >= big.tile && k >= 32 && groups(big.tile) >= MIN_GROUPS {
        Some(big)
    } else if big.tile > 32 && m >= 32 && n >= 32 && k >= 16 && groups(32) >= MIN_GROUPS {
        Some(medium)
    } else {
        None
    };
    let use_mma = std::env::var_os("EFFECT_TORCH_NO_MMA").is_none() && cfg.is_some();
    let pipeline = if let Some(cfg) = cfg.filter(|_| use_mma) {
        dev.compile_lazy(mma_key_for(has_bias, epilogue, a.dtype, cfg), "et_gemm_mma", || {
            gemm_mma_source(has_bias, epilogue, ty, cfg)
        })?
    } else {
        dev.compile_lazy(key_for(has_bias, epilogue, a.dtype), "et_gemm", || {
            gemm_source(has_bias, epilogue, ty)
        })?
    };
    let out = MetalTensor {
        buffer: dev.alloc(batch * m * n, a.dtype),
        layout: crate::runtime::layout::Layout::contiguous(vec![batch, m, n]),
        dtype: a.dtype,
    };
    let out2 = if epilogue.dual() {
        Some(MetalTensor {
            buffer: dev.alloc(batch * m * n, a.dtype),
            layout: crate::runtime::layout::Layout::contiguous(vec![batch, m, n]),
            dtype: a.dtype,
        })
    } else {
        None
    };
    let (mu, nu, ku) = (m as u32, n as u32, k as u32);
    let (sa, sb) = (stride_a as u32, stride_b as u32);
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &a.buffer, a.layout.offset() * esz);
        set_buffer(e, 1, &b.buffer, b.layout.offset() * esz);
        set_buffer(e, 2, &out.buffer, 0);
        if let Some(bias) = bias {
            set_buffer(e, 3, &bias.buffer, bias.layout.offset() * esz);
        }
        set_bytes(e, 4, &mu);
        set_bytes(e, 5, &nu);
        set_bytes(e, 6, &ku);
        set_bytes(e, 7, &sa);
        set_bytes(e, 8, &sb);
        if let Some(r) = residual {
            set_buffer(e, 9, &r.buffer, r.layout.offset() * esz);
        }
        if let Some(out2) = &out2 {
            set_buffer(e, 10, &out2.buffer, 0);
        }
        if let Some(cfg) = cfg.filter(|_| use_mma) {
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
    Ok((out, out2))
}

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
    gemm_fused(dev, a, b, bias, None, Epilogue::None, batch, m, n, k, stride_a, stride_b)
        .map(|(out, _)| out)
}

pub fn matmul(dev: &MetalDevice, a: &MetalTensor, b: &MetalTensor) -> Result<MetalTensor, String> {
    let ar = a.layout.shape().len();
    let br = b.layout.shape().len();
    assert!(ar >= 2 && br >= 2, "matmul needs rank >= 2");
    let m = a.layout.shape()[ar - 2];
    let k = a.layout.shape()[ar - 1];
    let k2 = b.layout.shape()[br - 2];
    let n = b.layout.shape()[br - 1];
    assert_eq!(k, k2, "matmul inner dim mismatch");
    let batch_a: usize = a.layout.shape()[..ar - 2].iter().product();
    let batch_b: usize = b.layout.shape()[..br - 2].iter().product();
    assert!(
        batch_a == batch_b || batch_a == 1 || batch_b == 1,
        "matmul batch mismatch: {batch_a} vs {batch_b}"
    );
    let batch = batch_a.max(batch_b);
    let stride_a = if batch_a == 1 { 0 } else { m * k };
    let stride_b = if batch_b == 1 { 0 } else { k * n };
    let out = gemm(dev, a, b, None, batch, m, n, k, stride_a, stride_b)?;
    let mut out_shape = if batch_a >= batch_b {
        a.layout.shape()[..ar - 2].to_vec()
    } else {
        b.layout.shape()[..br - 2].to_vec()
    };
    out_shape.extend([m, n]);
    Ok(MetalTensor {
        buffer: out.buffer,
        layout: crate::runtime::layout::Layout::contiguous(out_shape),
        dtype: a.dtype,
    })
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
        dev.synchronize();
        let got = out.read_f32().unwrap();
        let mut want = vec![0f32; m * n];
        unsafe {
            matrixmultiply::sgemm(m, k, n, 1.0, a.as_ptr(), k as isize, 1, b.as_ptr(), n as isize, 1, 0.0, want.as_mut_ptr(), n as isize, 1);
        }
        for (x, y) in got.iter().zip(&want) {
            assert!((x - y).abs() / y.abs().max(1.0) < 1e-4, "{x} vs {y}");
        }
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
        dev.synchronize();
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
        let a: Vec<f32> = (0..batch * m * k).map(|i| (i as f32 * 0.31).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.17).cos()).collect();
        let bias: Vec<f32> = (0..n).map(|j| j as f32 * 0.5).collect();
        let r: Vec<f32> = (0..batch * m * n).map(|i| (i as f32 * 0.07).sin() * 2.0).collect();
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
        dev.synchronize();
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
                    assert!((got - want).abs() / want.abs().max(1.0) < 1e-4, "{got} vs {want}");
                }
            }
        }
    }

    #[test]
    fn gemm_mma_bf16_matches_f32() {
        let dev = MetalDevice::get();
        let (batch, m, n, k) = (2usize, 37usize, 53usize, 29usize);
        let a: Vec<f32> = (0..batch * m * k).map(|i| (i as f32 * 0.37).sin()).collect();
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
        dev.synchronize();
        let gotb: Vec<f32> = {
            let n = outb.numel();
            let ptr = unsafe { outb.buffer.contents_ptr().cast::<u16>() };
            let bits = unsafe { std::slice::from_raw_parts(ptr, n) };
            bits.iter().map(|b| half::bf16::from_bits(*b).to_f32()).collect()
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
            dev.synchronize();
            let (pre_buf, gelu_buf) = if dual {
                (Some(out.read_f32().unwrap()), out2.unwrap().read_f32().unwrap())
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

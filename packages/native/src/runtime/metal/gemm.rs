use super::device::{set_buffer, set_bytes, MetalDevice};
use super::emit::ACT_FNS;
use super::run::MetalTensor;
use crate::runtime::dtype::DType;
use objc2_metal::MTLComputeCommandEncoder;

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
    let pipeline = dev.compile_lazy(key_for(has_bias, epilogue, a.dtype), "et_gemm", || {
        gemm_source(has_bias, epilogue, ty)
    })?;
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
        e.dispatchThreadgroups_threadsPerThreadgroup(
            MetalDevice::grid(n.div_ceil(TILE), m.div_ceil(TILE), batch),
            MetalDevice::grid(TILE, TILE, 1),
        );
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

use super::device::{set_buffer, set_bytes, MetalDevice};
use objc2_metal::MTLComputeCommandEncoder;
use super::run::MetalTensor;
use crate::runtime::dtype::DType;

fn msl_type(d: DType) -> &'static str {
    match d {
        DType::F32 => "float",
        DType::F64 => "double",
        DType::F16 => "half",
        DType::BF16 => "bfloat",
        DType::U8 => "uchar",
        DType::U32 => "uint",
        DType::I64 => "long",
    }
}

fn key(parts: &[u64]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    for p in parts {
        p.hash(&mut h);
    }
    h.finish()
}

pub fn fill(dev: &MetalDevice, out: &MetalTensor, value: f64) -> Result<(), String> {
    let n = out.numel();
    if n == 0 {
        return Ok(());
    }
    let ty = msl_type(out.dtype);
    let make_src = || format!(
        r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_fill(device {ty}* out [[buffer(0)]], constant float& v [[buffer(1)]], uint i [[thread_position_in_grid]]) {{
    if (i < {n}u) out[i] = ({ty})v;
}}
"#
    );
    let pipeline = dev.compile_lazy(key(&[0xF111, out.dtype as u64, n as u64]), "et_fill", make_src)?;
    let v = value as f32;
    let padded = n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &out.buffer, out.layout.offset() * out.dtype.size_in_bytes());
        set_bytes(e, 1, &v);
        e.dispatchThreads_threadsPerThreadgroup(MetalDevice::grid(padded, 1, 1), MetalDevice::grid(256, 1, 1));
    });
    Ok(())
}

pub fn relu_i64(dev: &MetalDevice, x: &MetalTensor) -> Result<MetalTensor, String> {
    assert_eq!(x.dtype, DType::I64);
    let n = x.numel();
    let out = MetalTensor::empty(dev, x.layout.shape().to_vec(), DType::I64);
    if n == 0 {
        return Ok(out);
    }
    let make_src = || format!(
        r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_relu_i64(device const long* a [[buffer(0)]], device long* out [[buffer(1)]], uint i [[thread_position_in_grid]]) {{
    if (i < {n}u) out[i] = max(a[i], 0L);
}}
"#
    );
    let pipeline = dev.compile_lazy(key(&[0x8E10, n as u64]), "et_relu_i64", make_src)?;
    let padded = n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &x.buffer, x.layout.offset() * 8);
        set_buffer(e, 1, &out.buffer, 0);
        e.dispatchThreads_threadsPerThreadgroup(MetalDevice::grid(padded, 1, 1), MetalDevice::grid(256, 1, 1));
    });
    Ok(out)
}

pub fn cast(dev: &MetalDevice, x: &MetalTensor, dtype: DType) -> Result<MetalTensor, String> {
    if x.dtype == dtype {
        return Ok(MetalTensor {
            buffer: x.buffer.clone(),
            layout: x.layout.clone(),
            dtype,
        });
    }
    let n = x.numel();
    let out = MetalTensor::empty(dev, x.layout.shape().to_vec(), dtype);
    if n == 0 {
        return Ok(out);
    }
    let (src_ty, dst_ty) = (msl_type(x.dtype), msl_type(dtype));
    let make_src = || format!(
        r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_cast(device const {src_ty}* a [[buffer(0)]], device {dst_ty}* out [[buffer(1)]], uint i [[thread_position_in_grid]]) {{
    if (i < {n}u) out[i] = ({dst_ty})a[i];
}}
"#
    );
    let pipeline = dev.compile_lazy(key(&[0xCA57, x.dtype as u64, dtype as u64, n as u64]), "et_cast", make_src)?;
    let padded = n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &x.buffer, x.layout.offset() * x.dtype.size_in_bytes());
        set_buffer(e, 1, &out.buffer, 0);
        e.dispatchThreads_threadsPerThreadgroup(MetalDevice::grid(padded, 1, 1), MetalDevice::grid(256, 1, 1));
    });
    Ok(out)
}

pub fn strided_copy(dev: &MetalDevice, x: &MetalTensor) -> Result<MetalTensor, String> {
    if x.layout.is_contiguous() && x.layout.offset() == 0 {
        return Ok(x.clone());
    }
    let n = x.numel();
    let out = MetalTensor::empty(dev, x.layout.shape().to_vec(), x.dtype);
    if n == 0 {
        return Ok(out);
    }
    let shape = x.layout.shape();
    let strides = x.layout.strides();
    let ty = msl_type(x.dtype);
    let rank = shape.len();
    let contig = crate::runtime::layout::Layout::contiguous(shape.to_vec());
    let cs = contig.strides().to_vec();
    let mut decompose = String::new();
    for d in 0..rank {
        if strides[d] == 0 || shape[d] == 1 {
            continue;
        }
        let coord = if d == rank - 1 {
            format!("(i % {})", shape[d])
        } else {
            format!("((i / {}) % {})", cs[d], shape[d])
        };
        if strides[d] == 1 {
            decompose.push_str(&format!("            src_off += {coord};\n"));
        } else {
            decompose.push_str(&format!("            src_off += {coord} * {};\n", strides[d]));
        }
    }
    let make_src = || format!(
        r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_scopy(device const {ty}* a [[buffer(0)]], device {ty}* out [[buffer(1)]], uint i [[thread_position_in_grid]]) {{
    if (i < {n}u) {{
        uint src_off = 0u;
{decompose}        out[i] = a[src_off];
    }}
}}
"#
    );
    let shape_key = key(&shape.iter().map(|&v| v as u64).chain(strides.iter().map(|&v| v as u64)).collect::<Vec<_>>());
    let pipeline = dev.compile_lazy(key(&[0x5C09, x.dtype as u64, shape_key]), "et_scopy", make_src)?;
    let padded = n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &x.buffer, x.layout.offset() * x.dtype.size_in_bytes());
        set_buffer(e, 1, &out.buffer, 0);
        e.dispatchThreads_threadsPerThreadgroup(MetalDevice::grid(padded, 1, 1), MetalDevice::grid(256, 1, 1));
    });
    Ok(out)
}

const RNG_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline ulong xoro_next(thread ulong& s0, thread ulong& s1) {
    ulong r = s0 + s1;
    s1 ^= s0;
    s0 = (s0 << 55 | s0 >> 9) ^ s1 ^ (s1 << 14);
    s1 = (s1 << 36 | s1 >> 28);
    return r;
}

inline void xoro_seed(thread ulong& s0, thread ulong& s1, ulong seed) {
    ulong s = seed + 0x9E3779B97F4A7C15ul;
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    s0 = s;
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    s1 = s;
}

inline float xoro_f32(thread ulong& s0, thread ulong& s1) {
    return (float)(xoro_next(s0, s1) >> 40) * (1.0f / 16777216.0f);
}
"#;

pub fn randn(dev: &MetalDevice, shape: &[usize], seed: u64) -> Result<MetalTensor, String> {
    let n: usize = shape.iter().product();
    let out = MetalTensor::empty(dev, shape.to_vec(), DType::F32);
    if n == 0 {
        return Ok(out);
    }
    let make_src = || format!(
        r#"{RNG_SRC}
kernel void et_randn(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {{
    if (i < {n}u) {{
        ulong s0, s1;
        xoro_seed(s0, s1, {seed}ul + (ulong)i * 0x9E3779B97F4A7C15ul);
        float u1 = max(xoro_f32(s0, s1), 1e-12f);
        float u2 = xoro_f32(s0, s1);
        out[i] = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI_F * u2);
    }}
}}
"#
    );
    let pipeline = dev.compile_lazy(key(&[0x8A11, seed, n as u64]), "et_randn", make_src)?;
    let padded = n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &out.buffer, 0);
        e.dispatchThreads_threadsPerThreadgroup(MetalDevice::grid(padded, 1, 1), MetalDevice::grid(256, 1, 1));
    });
    Ok(out)
}

pub fn uniform(dev: &MetalDevice, lo: f64, hi: f64, shape: &[usize], seed: u64) -> Result<MetalTensor, String> {
    let n: usize = shape.iter().product();
    let out = MetalTensor::empty(dev, shape.to_vec(), DType::F32);
    if n == 0 {
        return Ok(out);
    }
    let make_src = || format!(
        r#"{RNG_SRC}
kernel void et_uniform(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {{
    if (i < {n}u) {{
        ulong s0, s1;
        xoro_seed(s0, s1, {seed}ul + (ulong)i * 0x9E3779B97F4A7C15ul);
        out[i] = {:?}f + ({:?}f - {:?}f) * xoro_f32(s0, s1);
    }}
}}
"#,
        lo as f32, hi as f32, lo as f32
    );
    let pipeline = dev.compile_lazy(key(&[0x0B1F, seed, n as u64, lo.to_bits() as u64, hi.to_bits() as u64]), "et_uniform", make_src)?;
    let padded = n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &out.buffer, 0);
        e.dispatchThreads_threadsPerThreadgroup(MetalDevice::grid(padded, 1, 1), MetalDevice::grid(256, 1, 1));
    });
    Ok(out)
}

pub fn arange(dev: &MetalDevice, start: f64, end: f64, step: f64, dtype: DType) -> Result<MetalTensor, String> {
    let n = ((end - start) / step).ceil().max(0.0) as usize;
    let out = MetalTensor::empty(dev, vec![n], dtype);
    if n == 0 {
        return Ok(out);
    }
    let ty = msl_type(dtype);
    let make_src = || format!(
        r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_arange(device {ty}* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {{
    if (i < {n}u) out[i] = ({ty})((float)i * {:?}f + {:?}f);
}}
"#,
        step, start
    );
    let pipeline = dev.compile_lazy(key(&[0xA26E, dtype as u64, start.to_bits(), step.to_bits(), n as u64]), "et_arange", make_src)?;
    let padded = n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &out.buffer, 0);
        e.dispatchThreads_threadsPerThreadgroup(MetalDevice::grid(padded, 1, 1), MetalDevice::grid(256, 1, 1));
    });
    Ok(out)
}

pub fn eye(dev: &MetalDevice, n: usize, dtype: DType) -> Result<MetalTensor, String> {
    let out = MetalTensor::empty(dev, vec![n, n], dtype);
    fill(dev, &out, 0.0)?;
    if n == 0 {
        return Ok(out);
    }
    let ty = msl_type(dtype);
    let make_src = || format!(
        r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_eye(device {ty}* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {{
    if (i < {n}u) out[i * {n}u + i] = ({ty})1;
}}
"#
    );
    let pipeline = dev.compile_lazy(key(&[0xE7E, dtype as u64, n as u64]), "et_eye", make_src)?;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &out.buffer, 0);
        e.dispatchThreads_threadsPerThreadgroup(MetalDevice::grid(n, 1, 1), MetalDevice::grid(n.min(256), 1, 1));
    });
    Ok(out)
}

pub fn argreduce(dev: &MetalDevice, x: &MetalTensor, dim: usize, pick_max: bool) -> Result<MetalTensor, String> {
    let shape = x.layout.shape();
    let rank = shape.len();
    let n = shape[dim];
    let dstride = x.layout.strides()[dim];
    let kept: Vec<usize> = (0..rank).filter(|&d| d != dim).collect();
    let kept_dims: Vec<usize> = kept.iter().map(|&d| shape[d]).collect();
    let kept_strides: Vec<usize> = kept.iter().map(|&d| x.layout.strides()[d]).collect();
    let kept_n: usize = kept_dims.iter().product();
    let mut out_shape = shape.to_vec();
    out_shape[dim] = 1;
    let out = MetalTensor::empty(dev, out_shape, DType::U32);
    if kept_n == 0 {
        return Ok(out);
    }
    let ty = msl_type(x.dtype);
    let cmp = if pick_max { ">" } else { "<" };
    let kept_rank = kept.len();
    let mut decompose = String::new();
    for k in (0..kept_rank).rev() {
        let c = kept_dims[k];
        let s = kept_strides[k];
        if k == kept_rank - 1 {
            decompose.push_str(&format!("        base += (gid % {c}u) * {s}u;\n"));
        } else {
            let div: usize = kept_dims[k + 1..].iter().product();
            decompose.push_str(&format!("        base += ((gid / {div}u) % {c}u) * {s}u;\n"));
        }
    }
    let make_src = || format!(
        r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_argred(
    device const {ty}* x [[buffer(0)]],
    device uint* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= {kept_n}u) return;
    uint base = 0u;
{decompose}    uint best = 0u;
    {ty} best_v = x[base];
    for (uint i = 1u; i < {n}u; ++i) {{
        {ty} v = x[base + i * {dstride}u];
        if (v {cmp} best_v) {{ best_v = v; best = i; }}
    }}
    out[gid] = best;
}}
"#
    );
    let pipeline = dev.compile_lazy(key(&[0xA26D, x.dtype as u64, dim as u64, pick_max as u64, n as u64, key(&kept_dims.iter().map(|&v| v as u64).collect::<Vec<_>>())]), "et_argred", make_src)?;
    let padded = kept_n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &x.buffer, x.layout.offset() * x.dtype.size_in_bytes());
        set_buffer(e, 1, &out.buffer, 0);
        e.dispatchThreads_threadsPerThreadgroup(MetalDevice::grid(padded, 1, 1), MetalDevice::grid(256, 1, 1));
    });
    Ok(out)
}

pub fn cumsum(dev: &MetalDevice, x: &MetalTensor, dim: usize) -> Result<MetalTensor, String> {
    let shape = x.layout.shape();
    let rank = shape.len();
    let n = shape[dim];
    let dstride = x.layout.strides()[dim];
    let kept: Vec<usize> = (0..rank).filter(|&d| d != dim).collect();
    let kept_dims: Vec<usize> = kept.iter().map(|&d| shape[d]).collect();
    let kept_strides: Vec<usize> = kept.iter().map(|&d| x.layout.strides()[d]).collect();
    let kept_n: usize = kept_dims.iter().product();
    let out = MetalTensor::empty(dev, shape.to_vec(), x.dtype);
    if kept_n == 0 {
        return Ok(out);
    }
    let ty = msl_type(x.dtype);
    let out_strides = crate::runtime::layout::Layout::contiguous(shape.to_vec());
    let os = out_strides.strides().to_vec();
    let kept_rank = kept.len();
    let mut decompose = String::new();
    for k in (0..kept_rank).rev() {
        let c = kept_dims[k];
        let s = kept_strides[k];
        let o = os[kept[k]];
        if k == kept_rank - 1 {
            decompose.push_str(&format!("        base += (gid % {c}u) * {s}u;\n        obase += (gid % {c}u) * {o}u;\n"));
        } else {
            let div: usize = kept_dims[k + 1..].iter().product();
            decompose.push_str(&format!("        base += ((gid / {div}u) % {c}u) * {s}u;\n        obase += ((gid / {div}u) % {c}u) * {o}u;\n"));
        }
    }
    let make_src = || format!(
        r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_cumsum(
    device const {ty}* x [[buffer(0)]],
    device {ty}* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= {kept_n}u) return;
    uint base = 0u;
    uint obase = 0u;
{decompose}    {ty} acc = ({ty})0;
    for (uint i = 0u; i < {n}u; ++i) {{
        acc += x[base + i * {dstride}u];
        out[obase + i * {os_dim}u] = acc;
    }}
}}
"#,
        os_dim = os[dim]
    );
    let pipeline = dev.compile_lazy(key(&[0xC50A, x.dtype as u64, dim as u64, n as u64, key(&kept_dims.iter().map(|&v| v as u64).collect::<Vec<_>>())]), "et_cumsum", make_src)?;
    let padded = kept_n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &x.buffer, x.layout.offset() * x.dtype.size_in_bytes());
        set_buffer(e, 1, &out.buffer, 0);
        e.dispatchThreads_threadsPerThreadgroup(MetalDevice::grid(padded, 1, 1), MetalDevice::grid(256, 1, 1));
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cast_f16_roundtrip() {
        let dev = MetalDevice::get();
        let x = MetalTensor::from_f32(dev, vec![1.5, -2.25, 100.0], vec![3]);
        let h = cast(dev, &x, DType::F16).unwrap();
        let back = cast(dev, &h, DType::F32).unwrap();
        dev.synchronize();
        assert_eq!(back.read_f32(), vec![1.5, -2.25, 100.0]);
    }

    #[test]
    fn strided_copy_permuted() {
        let dev = MetalDevice::get();
        let x = MetalTensor::from_f32(dev, (0..6).map(|v| v as f32).collect(), vec![2, 3]);
        let p = MetalTensor {
            buffer: x.buffer.clone(),
            layout: x.layout.permute(&[1, 0]),
            dtype: x.dtype,
        };
        let c = strided_copy(dev, &p).unwrap();
        dev.synchronize();
        assert_eq!(c.read_f32(), vec![0., 3., 1., 4., 2., 5.]);
    }

    #[test]
    fn randn_deterministic_per_seed() {
        let dev = MetalDevice::get();
        let a = randn(dev, &[8], 42).unwrap();
        let b = randn(dev, &[8], 42).unwrap();
        dev.synchronize();
        assert_eq!(a.read_f32(), b.read_f32());
        let m: f32 = a.read_f32().iter().sum::<f32>() / 8.0;
        assert!(m.abs() < 2.0);
    }

    #[test]
    fn arange_eye_fill() {
        let dev = MetalDevice::get();
        let a = arange(dev, 0.0, 5.0, 2.0, DType::F32).unwrap();
        dev.synchronize();
        assert_eq!(a.read_f32(), vec![0., 2., 4.]);
        let e = eye(dev, 2, DType::F32).unwrap();
        dev.synchronize();
        assert_eq!(e.read_f32(), vec![1., 0., 0., 1.]);
    }
}

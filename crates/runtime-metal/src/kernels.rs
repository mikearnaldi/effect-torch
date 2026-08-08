use super::device::{set_buffer, MetalDevice};
use super::run::MetalTensor;
use crate::runtime::dtype::DType;
use objc2_metal::MTLComputeCommandEncoder;

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
    let wide = MetalDevice::WIDE;
    let n = out.numel();
    if n == 0 {
        return Ok(());
    }
    let ty = msl_type(out.dtype);
    // The scalar is baked as a literal of the target type: routing it
    // through a float constant would silently round integer and f64
    // fills above 2^24 (a checkpointed u32 sampler length hit exactly
    // this).
    let literal = match out.dtype {
        DType::U8 => format!("(uchar){}", value as u8),
        DType::U32 => format!("(uint){}u", value as u32),
        DType::I64 => format!("(long){}ll", value as i64),
        DType::F64 => format!("{value:?}"),
        _ => format!("{:?}f", value as f32),
    };
    let make_src = || {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_fill(device {ty}* out [[buffer(0)]], uint2 gid2 [[thread_position_in_grid]]) {{
    const ulong i = ulong(gid2.y) * {wide}ul + ulong(gid2.x);
    if (i < {n}ul) out[i] = ({ty})({literal});
}}
"#
        )
    };
    let pipeline = dev.compile_lazy(
        key(&[0xF111, out.dtype as u64, n as u64, value.to_bits()]),
        "et_fill",
        make_src,
    )?;
    let padded = n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(
            e,
            0,
            &out.buffer,
            out.layout.offset() * out.dtype.size_in_bytes(),
        );
        {
            let (g, tg) = MetalDevice::grid_flat(padded);
            e.dispatchThreads_threadsPerThreadgroup(g, tg);
        }
    });
    Ok(())
}

pub fn relu_i64(dev: &MetalDevice, x: &MetalTensor) -> Result<MetalTensor, String> {
    let wide = MetalDevice::WIDE;
    assert_eq!(x.dtype, DType::I64);
    let n = x.numel();
    let out = MetalTensor::empty(dev, x.layout.shape().to_vec(), DType::I64);
    if n == 0 {
        return Ok(out);
    }
    let make_src = || {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_relu_i64(device const long* a [[buffer(0)]], device long* out [[buffer(1)]], uint2 gid2 [[thread_position_in_grid]]) {{
    const ulong i = ulong(gid2.y) * {wide}ul + ulong(gid2.x);
    if (i < {n}ul) out[i] = max(a[i], 0L);
}}
"#
        )
    };
    let pipeline = dev.compile_lazy(key(&[0x8E10, n as u64]), "et_relu_i64", make_src)?;
    let padded = n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &x.buffer, x.layout.offset() * 8);
        set_buffer(e, 1, &out.buffer, 0);
        {
            let (g, tg) = MetalDevice::grid_flat(padded);
            e.dispatchThreads_threadsPerThreadgroup(g, tg);
        }
    });
    Ok(out)
}

pub fn cast(dev: &MetalDevice, x: &MetalTensor, dtype: DType) -> Result<MetalTensor, String> {
    let wide = MetalDevice::WIDE;
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
    let make_src = || {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_cast(device const {src_ty}* a [[buffer(0)]], device {dst_ty}* out [[buffer(1)]], uint2 gid2 [[thread_position_in_grid]]) {{
    const ulong i = ulong(gid2.y) * {wide}ul + ulong(gid2.x);
    if (i < {n}ul) out[i] = ({dst_ty})a[i];
}}
"#
        )
    };
    let pipeline = dev.compile_lazy(
        key(&[0xCA57, x.dtype as u64, dtype as u64, n as u64]),
        "et_cast",
        make_src,
    )?;
    let padded = n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &x.buffer, x.layout.offset() * x.dtype.size_in_bytes());
        set_buffer(e, 1, &out.buffer, 0);
        {
            let (g, tg) = MetalDevice::grid_flat(padded);
            e.dispatchThreads_threadsPerThreadgroup(g, tg);
        }
    });
    Ok(out)
}

pub fn strided_copy(dev: &MetalDevice, x: &MetalTensor) -> Result<MetalTensor, String> {
    let wide = MetalDevice::WIDE;
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
            decompose.push_str(&format!(
                "            src_off += {coord} * {};\n",
                strides[d]
            ));
        }
    }
    let make_src = || {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_scopy(device const {ty}* a [[buffer(0)]], device {ty}* out [[buffer(1)]], uint2 gid2 [[thread_position_in_grid]]) {{
    const ulong i = ulong(gid2.y) * {wide}ul + ulong(gid2.x);
    if (i < {n}ul) {{
        uint src_off = 0u;
{decompose}        out[i] = a[src_off];
    }}
}}
"#
        )
    };
    let shape_key = key(&shape
        .iter()
        .map(|&v| v as u64)
        .chain(strides.iter().map(|&v| v as u64))
        .collect::<Vec<_>>());
    let pipeline = dev.compile_lazy(
        key(&[0x5C09, x.dtype as u64, shape_key]),
        "et_scopy",
        make_src,
    )?;
    let padded = n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &x.buffer, x.layout.offset() * x.dtype.size_in_bytes());
        set_buffer(e, 1, &out.buffer, 0);
        {
            let (g, tg) = MetalDevice::grid_flat(padded);
            e.dispatchThreads_threadsPerThreadgroup(g, tg);
        }
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
    let wide = MetalDevice::WIDE;
    let n: usize = shape.iter().product();
    let out = MetalTensor::empty(dev, shape.to_vec(), DType::F32);
    if n == 0 {
        return Ok(out);
    }
    let make_src = || {
        format!(
            r#"{RNG_SRC}
kernel void et_randn(device float* out [[buffer(0)]], uint2 gid2 [[thread_position_in_grid]]) {{
    const ulong i = ulong(gid2.y) * {wide}ul + ulong(gid2.x);
    if (i < {n}ul) {{
        ulong s0, s1;
        xoro_seed(s0, s1, {seed}ul + (ulong)i * 0x9E3779B97F4A7C15ul);
        float u1 = max(xoro_f32(s0, s1), 1e-12f);
        float u2 = xoro_f32(s0, s1);
        out[i] = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI_F * u2);
    }}
}}
"#
        )
    };
    let pipeline = dev.compile_lazy(key(&[0x8A11, seed, n as u64]), "et_randn", make_src)?;
    let padded = n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &out.buffer, 0);
        {
            let (g, tg) = MetalDevice::grid_flat(padded);
            e.dispatchThreads_threadsPerThreadgroup(g, tg);
        }
    });
    Ok(out)
}

pub fn uniform(
    dev: &MetalDevice,
    lo: f64,
    hi: f64,
    shape: &[usize],
    seed: u64,
) -> Result<MetalTensor, String> {
    let wide = MetalDevice::WIDE;
    let n: usize = shape.iter().product();
    let out = MetalTensor::empty(dev, shape.to_vec(), DType::F32);
    if n == 0 {
        return Ok(out);
    }
    let make_src = || {
        format!(
            r#"{RNG_SRC}
kernel void et_uniform(device float* out [[buffer(0)]], uint2 gid2 [[thread_position_in_grid]]) {{
    const ulong i = ulong(gid2.y) * {wide}ul + ulong(gid2.x);
    if (i < {n}ul) {{
        ulong s0, s1;
        xoro_seed(s0, s1, {seed}ul + (ulong)i * 0x9E3779B97F4A7C15ul);
        out[i] = {:?}f + ({:?}f - {:?}f) * xoro_f32(s0, s1);
    }}
}}
"#,
            lo as f32, hi as f32, lo as f32
        )
    };
    let pipeline = dev.compile_lazy(
        key(&[
            0x0B1F,
            seed,
            n as u64,
            lo.to_bits() as u64,
            hi.to_bits() as u64,
        ]),
        "et_uniform",
        make_src,
    )?;
    let padded = n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &out.buffer, 0);
        {
            let (g, tg) = MetalDevice::grid_flat(padded);
            e.dispatchThreads_threadsPerThreadgroup(g, tg);
        }
    });
    Ok(out)
}

pub fn arange(
    dev: &MetalDevice,
    start: f64,
    end: f64,
    step: f64,
    dtype: DType,
) -> Result<MetalTensor, String> {
    let wide = MetalDevice::WIDE;
    let n = ((end - start) / step).ceil().max(0.0) as usize;
    let out = MetalTensor::empty(dev, vec![n], dtype);
    if n == 0 {
        return Ok(out);
    }
    let ty = msl_type(dtype);
    // Integer arange computes in 64-bit integer arithmetic: the float
    // form rounds positions above 2^24 (token ids and position grids can
    // exceed that). Integral starts/steps are exact; fractional ones
    // truncate toward zero, matching the final integer cast.
    let element = match dtype {
        DType::I64 => format!("(long)i * {}ll + {}ll", step as i64, start as i64),
        DType::U32 => format!("(uint)((ulong)i * {}ul + {}ul)", step as u32, start as u32),
        _ => format!("(float)i * {:?}f + {:?}f", step, start),
    };
    let make_src = || {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_arange(device {ty}* out [[buffer(0)]], uint2 gid2 [[thread_position_in_grid]]) {{
    const ulong i = ulong(gid2.y) * {wide}ul + ulong(gid2.x);
    if (i < {n}ul) out[i] = ({ty})({element});
}}
"#,
        )
    };
    let pipeline = dev.compile_lazy(
        key(&[
            0xA26E,
            dtype as u64,
            start.to_bits(),
            step.to_bits(),
            n as u64,
        ]),
        "et_arange",
        make_src,
    )?;
    let padded = n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &out.buffer, 0);
        {
            let (g, tg) = MetalDevice::grid_flat(padded);
            e.dispatchThreads_threadsPerThreadgroup(g, tg);
        }
    });
    Ok(out)
}

pub fn eye(dev: &MetalDevice, n: usize, dtype: DType) -> Result<MetalTensor, String> {
    let wide = MetalDevice::WIDE;
    let out = MetalTensor::empty(dev, vec![n, n], dtype);
    fill(dev, &out, 0.0)?;
    if n == 0 {
        return Ok(out);
    }
    let ty = msl_type(dtype);
    let make_src = || {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_eye(device {ty}* out [[buffer(0)]], uint2 gid2 [[thread_position_in_grid]]) {{
    const ulong i = ulong(gid2.y) * {wide}ul + ulong(gid2.x);
    if (i < {n}ul) out[i * {n}u + i] = ({ty})1;
}}
"#
        )
    };
    let pipeline = dev.compile_lazy(key(&[0xE7E, dtype as u64, n as u64]), "et_eye", make_src)?;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &out.buffer, 0);
        e.dispatchThreads_threadsPerThreadgroup(
            MetalDevice::grid(n, 1, 1),
            MetalDevice::grid(n.min(256), 1, 1),
        );
    });
    Ok(out)
}

pub fn argreduce(
    dev: &MetalDevice,
    x: &MetalTensor,
    dim: usize,
    pick_max: bool,
) -> Result<MetalTensor, String> {
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
            decompose.push_str(&format!("        base += (gid % {c}u) * {s}ul;\n"));
        } else {
            let div: usize = kept_dims[k + 1..].iter().product();
            decompose.push_str(&format!(
                "        base += ((gid / {div}u) % {c}u) * {s}ul;\n"
            ));
        }
    }
    let make_src = || {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_argred(
    device const {ty}* x [[buffer(0)]],
    device uint* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= {kept_n}u) return;
    ulong base = 0ul;
{decompose}    uint best = 0u;
    {ty} best_v = x[base];
    for (uint i = 1u; i < {n}u; ++i) {{
        {ty} v = x[base + ulong(i) * {dstride}ul];
        if (v {cmp} best_v) {{ best_v = v; best = i; }}
    }}
    out[gid] = best;
}}
"#
        )
    };
    let pipeline = dev.compile_lazy(
        key(&[
            0xA26D,
            x.dtype as u64,
            dim as u64,
            pick_max as u64,
            n as u64,
            key(&kept_dims.iter().map(|&v| v as u64).collect::<Vec<_>>()),
        ]),
        "et_argred",
        make_src,
    )?;
    let padded = kept_n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &x.buffer, x.layout.offset() * x.dtype.size_in_bytes());
        set_buffer(e, 1, &out.buffer, 0);
        {
            let (g, tg) = MetalDevice::grid_flat(padded);
            e.dispatchThreads_threadsPerThreadgroup(g, tg);
        }
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
            decompose.push_str(&format!(
                "        base += (gid % {c}u) * {s}ul;\n        obase += (gid % {c}u) * {o}ul;\n"
            ));
        } else {
            let div: usize = kept_dims[k + 1..].iter().product();
            decompose.push_str(&format!("        base += ((gid / {div}u) % {c}u) * {s}ul;\n        obase += ((gid / {div}u) % {c}u) * {o}ul;\n"));
        }
    }
    let make_src = || {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_cumsum(
    device const {ty}* x [[buffer(0)]],
    device {ty}* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= {kept_n}u) return;
    ulong base = 0ul;
    ulong obase = 0ul;
{decompose}    {ty} acc = ({ty})0;
    for (uint i = 0u; i < {n}u; ++i) {{
        acc += x[base + ulong(i) * {dstride}ul];
        out[obase + ulong(i) * {os_dim}ul] = acc;
    }}
}}
"#,
            os_dim = os[dim]
        )
    };
    let pipeline = dev.compile_lazy(
        key(&[
            0xC50A,
            x.dtype as u64,
            dim as u64,
            n as u64,
            key(&kept_dims.iter().map(|&v| v as u64).collect::<Vec<_>>()),
        ]),
        "et_cumsum",
        make_src,
    )?;
    let padded = kept_n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &x.buffer, x.layout.offset() * x.dtype.size_in_bytes());
        set_buffer(e, 1, &out.buffer, 0);
        {
            let (g, tg) = MetalDevice::grid_flat(padded);
            e.dispatchThreads_threadsPerThreadgroup(g, tg);
        }
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
        dev.synchronize().unwrap();
        assert_eq!(back.read_f32().unwrap(), vec![1.5, -2.25, 100.0]);
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
        dev.synchronize().unwrap();
        assert_eq!(c.read_f32().unwrap(), vec![0., 3., 1., 4., 2., 5.]);
    }

    #[test]
    fn randn_deterministic_per_seed() {
        let dev = MetalDevice::get();
        let a = randn(dev, &[8], 42).unwrap();
        let b = randn(dev, &[8], 42).unwrap();
        dev.synchronize().unwrap();
        assert_eq!(a.read_f32().unwrap(), b.read_f32().unwrap());
        let m: f32 = a.read_f32().unwrap().iter().sum::<f32>() / 8.0;
        assert!(m.abs() < 2.0);
    }

    #[test]
    fn arange_eye_fill() {
        let dev = MetalDevice::get();
        let a = arange(dev, 0.0, 5.0, 2.0, DType::F32).unwrap();
        dev.synchronize().unwrap();
        assert_eq!(a.read_f32().unwrap(), vec![0., 2., 4.]);
        let e = eye(dev, 2, DType::F32).unwrap();
        dev.synchronize().unwrap();
        assert_eq!(e.read_f32().unwrap(), vec![1., 0., 0., 1.]);
    }

    // Integer scalars must not round-trip through f32: values above
    // 2^24 have no exact f32 form (a checkpointed u32 sampler length
    // regressed this way).
    #[test]
    fn fill_and_arange_are_exact_for_large_integers() {
        let dev = MetalDevice::get();
        let out = MetalTensor::empty(dev, vec![2], DType::U32);
        fill(dev, &out, 744_841_714.0).unwrap();
        let a = arange(dev, 16_777_214.0, 16_777_220.0, 1.0, DType::I64).unwrap();
        dev.synchronize().unwrap();
        let raw = &out.buffer;
        let words = unsafe { std::slice::from_raw_parts(raw.contents_ptr().cast::<u32>(), 2) };
        assert_eq!(words, &[744_841_714, 744_841_714]);
        let a_raw = &a.buffer;
        let longs = unsafe { std::slice::from_raw_parts(a_raw.contents_ptr().cast::<i64>(), 6) };
        assert_eq!(
            longs,
            &[16_777_214, 16_777_215, 16_777_216, 16_777_217, 16_777_218, 16_777_219]
        );
    }
}

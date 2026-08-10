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

fn layout_key(layout: &crate::runtime::layout::Layout) -> u64 {
    key(&layout
        .shape()
        .iter()
        .map(|&value| value as u64)
        .chain(layout.strides().iter().map(|&value| value as u64))
        .collect::<Vec<_>>())
}

fn source_offset(layout: &crate::runtime::layout::Layout, index: &str, indent: &str) -> String {
    if layout.is_contiguous() {
        return format!("{indent}const ulong src_off = {index};\n");
    }
    let shape = layout.shape();
    let strides = layout.strides();
    let contiguous = crate::runtime::layout::Layout::contiguous(shape.to_vec());
    let contiguous_strides = contiguous.strides();
    let mut source = format!("{indent}ulong src_off = 0ul;\n");
    for dimension in 0..shape.len() {
        if strides[dimension] == 0 || shape[dimension] == 1 {
            continue;
        }
        let coordinate = if dimension == shape.len() - 1 {
            format!("({index} % {})", shape[dimension])
        } else {
            format!(
                "(({index} / {}) % {})",
                contiguous_strides[dimension], shape[dimension]
            )
        };
        if strides[dimension] == 1 {
            source.push_str(&format!("{indent}src_off += {coordinate};\n"));
        } else {
            source.push_str(&format!(
                "{indent}src_off += {coordinate} * {};\n",
                strides[dimension]
            ));
        }
    }
    source
}

fn precompiled_pipeline(
    dev: &MetalDevice,
    pipeline_key: u64,
    name: &str,
) -> Result<super::device::Pipeline, String> {
    dev.pipeline_cached(pipeline_key).ok_or_else(|| {
        format!(
            "metal kernel {name} pipeline {pipeline_key:#x} was not precompiled for the exact layout"
        )
    })
}

fn fill_pipeline(
    dev: &MetalDevice,
    dtype: DType,
    n: usize,
) -> Result<super::device::Pipeline, String> {
    let wide = MetalDevice::WIDE;
    let ty = msl_type(dtype);
    let value = match dtype {
        DType::U8 | DType::U32 | DType::I64 => format!("({ty})raw"),
        DType::F16 | DType::BF16 | DType::F32 => {
            format!("({ty})as_type<float>((uint)raw)")
        }
        DType::F64 => "(double)as_type<double>(raw)".to_string(),
    };
    let make_src = || {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_fill(device {ty}* out [[buffer(0)]], constant ulong& raw [[buffer(1)]], uint2 gid2 [[thread_position_in_grid]]) {{
    const ulong i = ulong(gid2.y) * {wide}ul + ulong(gid2.x);
    if (i < {n}ul) out[i] = {value};
}}
"#
        )
    };
    dev.compile_lazy(key(&[0xF111, dtype as u64, n as u64]), "et_fill", make_src)
}

pub fn compile_fill(
    dev: &MetalDevice,
    shape: &[usize],
    _value: f64,
    dtype: DType,
) -> Result<(), String> {
    let n = shape.iter().product::<usize>();
    if n != 0 {
        fill_pipeline(dev, dtype, n)?;
    }
    Ok(())
}

pub fn warm_fill(shape: &[usize], value: f64, dtype: DType) -> Result<(), String> {
    compile_fill(MetalDevice::get(), shape, value, dtype)
}

pub fn fill_into(dev: &MetalDevice, out: &MetalTensor, value: f64) -> Result<(), String> {
    out.validate_destination("fill", out.layout.shape(), out.dtype)?;
    let n = out.numel();
    if n == 0 {
        return Ok(());
    }
    let pipeline =
        precompiled_pipeline(dev, key(&[0xF111, out.dtype as u64, n as u64]), "et_fill")?;
    let raw = match out.dtype {
        DType::U8 => value as u8 as u64,
        DType::U32 => value as u32 as u64,
        DType::I64 => value as i64 as u64,
        DType::F16 | DType::BF16 | DType::F32 => (value as f32).to_bits() as u64,
        DType::F64 => value.to_bits(),
    };
    let padded = n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(
            e,
            0,
            &out.buffer,
            out.layout.offset() * out.dtype.size_in_bytes(),
        );
        super::device::set_bytes(e, 1, &raw);
        {
            let (g, tg) = MetalDevice::grid_flat(padded);
            e.dispatchThreads_threadsPerThreadgroup(g, tg);
        }
    });
    Ok(())
}

pub fn fill(dev: &MetalDevice, out: &MetalTensor, value: f64) -> Result<(), String> {
    compile_fill(dev, out.layout.shape(), value, out.dtype)?;
    fill_into(dev, out, value)
}

fn relu_i64_pipeline(
    dev: &MetalDevice,
    layout: &crate::runtime::layout::Layout,
) -> Result<super::device::Pipeline, String> {
    let wide = MetalDevice::WIDE;
    let n = layout.numel();
    let offset = source_offset(layout, "i", "        ");
    let make_src = || {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_relu_i64(device const long* a [[buffer(0)]], device long* out [[buffer(1)]], uint2 gid2 [[thread_position_in_grid]]) {{
    const ulong i = ulong(gid2.y) * {wide}ul + ulong(gid2.x);
    if (i < {n}ul) {{
{offset}        out[i] = max(a[src_off], 0L);
    }}
}}
"#
        )
    };
    dev.compile_lazy(key(&[0x8E10, layout_key(layout)]), "et_relu_i64", make_src)
}

pub fn compile_relu_i64_layout(
    dev: &MetalDevice,
    layout: &crate::runtime::layout::Layout,
) -> Result<(), String> {
    if layout.numel() != 0 {
        relu_i64_pipeline(dev, layout)?;
    }
    Ok(())
}

pub fn compile_relu_i64(dev: &MetalDevice, shape: &[usize]) -> Result<(), String> {
    compile_relu_i64_layout(
        dev,
        &crate::runtime::layout::Layout::contiguous(shape.to_vec()),
    )
}

pub fn warm_relu_i64(shape: &[usize]) -> Result<(), String> {
    compile_relu_i64(MetalDevice::get(), shape)
}

pub fn relu_i64(dev: &MetalDevice, x: &MetalTensor) -> Result<MetalTensor, String> {
    compile_relu_i64_layout(dev, &x.layout)?;
    let out = MetalTensor::empty(dev, x.layout.shape().to_vec(), DType::I64);
    relu_i64_into(dev, x, &out)?;
    Ok(out)
}

pub fn relu_i64_into(dev: &MetalDevice, x: &MetalTensor, out: &MetalTensor) -> Result<(), String> {
    if x.dtype != DType::I64 {
        return Err(format!("relu_i64 input must be i64, got {:?}", x.dtype));
    }
    out.validate_destination("relu_i64", x.layout.shape(), DType::I64)?;
    let n = x.numel();
    if n == 0 {
        return Ok(());
    }
    let pipeline = precompiled_pipeline(dev, key(&[0x8E10, layout_key(&x.layout)]), "et_relu_i64")?;
    let padded = n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &x.buffer, x.layout.offset() * 8);
        set_buffer(e, 1, &out.buffer, out.layout.offset() * 8);
        {
            let (g, tg) = MetalDevice::grid_flat(padded);
            e.dispatchThreads_threadsPerThreadgroup(g, tg);
        }
    });
    Ok(())
}

fn cast_pipeline(
    dev: &MetalDevice,
    layout: &crate::runtime::layout::Layout,
    source: DType,
    destination: DType,
) -> Result<super::device::Pipeline, String> {
    let wide = MetalDevice::WIDE;
    let n = layout.numel();
    let (src_ty, dst_ty) = (msl_type(source), msl_type(destination));
    let offset = source_offset(layout, "i", "        ");
    let make_src = || {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_cast(device const {src_ty}* a [[buffer(0)]], device {dst_ty}* out [[buffer(1)]], uint2 gid2 [[thread_position_in_grid]]) {{
    const ulong i = ulong(gid2.y) * {wide}ul + ulong(gid2.x);
    if (i < {n}ul) {{
{offset}        out[i] = ({dst_ty})a[src_off];
    }}
}}
"#
        )
    };
    dev.compile_lazy(
        key(&[
            0xCA57,
            source as u64,
            destination as u64,
            layout_key(layout),
        ]),
        "et_cast",
        make_src,
    )
}

pub fn compile_cast_layout(
    dev: &MetalDevice,
    layout: &crate::runtime::layout::Layout,
    source: DType,
    destination: DType,
) -> Result<(), String> {
    if layout.numel() != 0 {
        if source == destination {
            copy_pipeline(dev, layout, source)?;
        } else {
            cast_pipeline(dev, layout, source, destination)?;
        }
    }
    Ok(())
}

pub fn compile_cast(
    dev: &MetalDevice,
    shape: &[usize],
    source: DType,
    destination: DType,
) -> Result<(), String> {
    compile_cast_layout(
        dev,
        &crate::runtime::layout::Layout::contiguous(shape.to_vec()),
        source,
        destination,
    )
}

pub fn warm_cast(shape: &[usize], source: DType, destination: DType) -> Result<(), String> {
    compile_cast(MetalDevice::get(), shape, source, destination)
}

pub fn cast(dev: &MetalDevice, x: &MetalTensor, dtype: DType) -> Result<MetalTensor, String> {
    if x.dtype == dtype {
        return Ok(MetalTensor {
            buffer: x.buffer.clone(),
            layout: x.layout.clone(),
            dtype,
        });
    }
    compile_cast_layout(dev, &x.layout, x.dtype, dtype)?;
    let out = MetalTensor::empty(dev, x.layout.shape().to_vec(), dtype);
    cast_into(dev, x, &out)?;
    Ok(out)
}

pub fn cast_into(dev: &MetalDevice, x: &MetalTensor, out: &MetalTensor) -> Result<(), String> {
    out.validate_destination("cast", x.layout.shape(), out.dtype)?;
    if x.dtype == out.dtype {
        return copy_into(dev, x, out);
    }
    let n = x.numel();
    if n == 0 {
        return Ok(());
    }
    let pipeline = precompiled_pipeline(
        dev,
        key(&[
            0xCA57,
            x.dtype as u64,
            out.dtype as u64,
            layout_key(&x.layout),
        ]),
        "et_cast",
    )?;
    let padded = n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &x.buffer, x.layout.offset() * x.dtype.size_in_bytes());
        set_buffer(
            e,
            1,
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

pub fn strided_copy(dev: &MetalDevice, x: &MetalTensor) -> Result<MetalTensor, String> {
    if x.layout.is_contiguous() && x.layout.offset() == 0 {
        return Ok(x.clone());
    }
    compile_copy_layout(dev, &x.layout, x.dtype)?;
    let out = MetalTensor::empty(dev, x.layout.shape().to_vec(), x.dtype);
    copy_into(dev, x, &out)?;
    Ok(out)
}

fn copy_pipeline(
    dev: &MetalDevice,
    layout: &crate::runtime::layout::Layout,
    dtype: DType,
) -> Result<super::device::Pipeline, String> {
    let wide = MetalDevice::WIDE;
    let n = layout.numel();
    let ty = msl_type(dtype);
    let offset = source_offset(layout, "i", "        ");
    let make_src = || {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_scopy(device const {ty}* a [[buffer(0)]], device {ty}* out [[buffer(1)]], uint2 gid2 [[thread_position_in_grid]]) {{
    const ulong i = ulong(gid2.y) * {wide}ul + ulong(gid2.x);
    if (i < {n}ul) {{
{offset}        out[i] = a[src_off];
    }}
}}
"#
        )
    };
    dev.compile_lazy(
        key(&[0x5C09, dtype as u64, layout_key(layout)]),
        "et_scopy",
        make_src,
    )
}

pub fn compile_copy(dev: &MetalDevice, shape: &[usize], dtype: DType) -> Result<(), String> {
    let layout = crate::runtime::layout::Layout::contiguous(shape.to_vec());
    compile_copy_layout(dev, &layout, dtype)
}

pub fn compile_copy_layout(
    dev: &MetalDevice,
    layout: &crate::runtime::layout::Layout,
    dtype: DType,
) -> Result<(), String> {
    if layout.numel() != 0 {
        copy_pipeline(dev, layout, dtype)?;
    }
    Ok(())
}

pub fn warm_copy(shape: &[usize], dtype: DType) -> Result<(), String> {
    compile_copy(MetalDevice::get(), shape, dtype)
}

pub fn warm_copy_layout(
    layout: &crate::runtime::layout::Layout,
    dtype: DType,
) -> Result<(), String> {
    compile_copy_layout(MetalDevice::get(), layout, dtype)
}

pub fn copy_into(
    dev: &MetalDevice,
    source: &MetalTensor,
    destination: &MetalTensor,
) -> Result<(), String> {
    if source.layout.shape() != destination.layout.shape() || source.dtype != destination.dtype {
        return Err(format!(
            "metal copy destination mismatch: source {:?}:{:?}, destination {:?}:{:?}",
            source.layout.shape(),
            source.dtype,
            destination.layout.shape(),
            destination.dtype
        ));
    }
    destination.validate_destination("copy", source.layout.shape(), source.dtype)?;
    let n = source.numel();
    if n == 0 {
        return Ok(());
    }
    let pipeline = precompiled_pipeline(
        dev,
        key(&[0x5C09, source.dtype as u64, layout_key(&source.layout)]),
        "et_scopy",
    )?;
    let padded = n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(
            e,
            0,
            &source.buffer,
            source.layout.offset() * source.dtype.size_in_bytes(),
        );
        set_buffer(
            e,
            1,
            &destination.buffer,
            destination.layout.offset() * destination.dtype.size_in_bytes(),
        );
        {
            let (g, tg) = MetalDevice::grid_flat(padded);
            e.dispatchThreads_threadsPerThreadgroup(g, tg);
        }
    });
    Ok(())
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

fn randn_pipeline(dev: &MetalDevice, n: usize) -> Result<super::device::Pipeline, String> {
    let wide = MetalDevice::WIDE;
    let make_src = || {
        format!(
            r#"{RNG_SRC}
kernel void et_randn(device float* out [[buffer(0)]], constant ulong& seed [[buffer(1)]], uint2 gid2 [[thread_position_in_grid]]) {{
    const ulong i = ulong(gid2.y) * {wide}ul + ulong(gid2.x);
    if (i < {n}ul) {{
        ulong s0, s1;
        xoro_seed(s0, s1, seed + (ulong)i * 0x9E3779B97F4A7C15ul);
        float u1 = max(xoro_f32(s0, s1), 1e-12f);
        float u2 = xoro_f32(s0, s1);
        out[i] = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI_F * u2);
    }}
}}
"#
        )
    };
    dev.compile_lazy(key(&[0x8A11, n as u64]), "et_randn", make_src)
}

pub fn compile_randn(dev: &MetalDevice, shape: &[usize]) -> Result<(), String> {
    if shape.iter().product::<usize>() != 0 {
        randn_pipeline(dev, shape.iter().product())?;
    }
    Ok(())
}

pub fn warm_randn(shape: &[usize]) -> Result<(), String> {
    compile_randn(MetalDevice::get(), shape)
}

pub fn randn(dev: &MetalDevice, shape: &[usize], seed: u64) -> Result<MetalTensor, String> {
    compile_randn(dev, shape)?;
    let out = MetalTensor::empty(dev, shape.to_vec(), DType::F32);
    randn_into(dev, &out, seed)?;
    Ok(out)
}

pub fn randn_into(dev: &MetalDevice, out: &MetalTensor, seed: u64) -> Result<(), String> {
    out.validate_destination("randn", out.layout.shape(), DType::F32)?;
    let n = out.numel();
    if n == 0 {
        return Ok(());
    }
    let pipeline = precompiled_pipeline(dev, key(&[0x8A11, n as u64]), "et_randn")?;
    let padded = n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(
            e,
            0,
            &out.buffer,
            out.layout.offset() * out.dtype.size_in_bytes(),
        );
        super::device::set_bytes(e, 1, &seed);
        {
            let (g, tg) = MetalDevice::grid_flat(padded);
            e.dispatchThreads_threadsPerThreadgroup(g, tg);
        }
    });
    Ok(())
}

pub fn uniform(
    dev: &MetalDevice,
    lo: f64,
    hi: f64,
    shape: &[usize],
    seed: u64,
) -> Result<MetalTensor, String> {
    compile_uniform(dev, lo, hi, shape)?;
    let out = MetalTensor::empty(dev, shape.to_vec(), DType::F32);
    uniform_into(dev, lo, hi, &out, seed)?;
    Ok(out)
}

pub fn uniform_into(
    dev: &MetalDevice,
    lo: f64,
    hi: f64,
    out: &MetalTensor,
    seed: u64,
) -> Result<(), String> {
    out.validate_destination("uniform", out.layout.shape(), DType::F32)?;
    let n = out.numel();
    if n == 0 {
        return Ok(());
    }
    let pipeline = precompiled_pipeline(
        dev,
        key(&[0x0B1F, n as u64, lo.to_bits() as u64, hi.to_bits() as u64]),
        "et_uniform",
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
        super::device::set_bytes(e, 1, &seed);
        {
            let (g, tg) = MetalDevice::grid_flat(padded);
            e.dispatchThreads_threadsPerThreadgroup(g, tg);
        }
    });
    Ok(())
}

fn uniform_pipeline(
    dev: &MetalDevice,
    lo: f64,
    hi: f64,
    n: usize,
) -> Result<super::device::Pipeline, String> {
    let wide = MetalDevice::WIDE;
    let make_src = || {
        format!(
            r#"{RNG_SRC}
kernel void et_uniform(device float* out [[buffer(0)]], constant ulong& seed [[buffer(1)]], uint2 gid2 [[thread_position_in_grid]]) {{
    const ulong i = ulong(gid2.y) * {wide}ul + ulong(gid2.x);
    if (i < {n}ul) {{
        ulong s0, s1;
        xoro_seed(s0, s1, seed + (ulong)i * 0x9E3779B97F4A7C15ul);
        out[i] = {:?}f + ({:?}f - {:?}f) * xoro_f32(s0, s1);
    }}
}}
"#,
            lo as f32, hi as f32, lo as f32
        )
    };
    dev.compile_lazy(
        key(&[0x0B1F, n as u64, lo.to_bits() as u64, hi.to_bits() as u64]),
        "et_uniform",
        make_src,
    )
}

pub fn compile_uniform(dev: &MetalDevice, lo: f64, hi: f64, shape: &[usize]) -> Result<(), String> {
    if shape.iter().product::<usize>() != 0 {
        uniform_pipeline(dev, lo, hi, shape.iter().product())?;
    }
    Ok(())
}

pub fn warm_uniform(lo: f64, hi: f64, shape: &[usize]) -> Result<(), String> {
    compile_uniform(MetalDevice::get(), lo, hi, shape)
}

pub fn arange_len(start: f64, end: f64, step: f64) -> usize {
    ((end - start) / step).ceil().max(0.0) as usize
}

fn arange_pipeline(
    dev: &MetalDevice,
    start: f64,
    step: f64,
    dtype: DType,
    n: usize,
) -> Result<super::device::Pipeline, String> {
    let wide = MetalDevice::WIDE;
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
    dev.compile_lazy(
        key(&[
            0xA26E,
            dtype as u64,
            start.to_bits(),
            step.to_bits(),
            n as u64,
        ]),
        "et_arange",
        make_src,
    )
}

pub fn compile_arange(
    dev: &MetalDevice,
    start: f64,
    end: f64,
    step: f64,
    dtype: DType,
) -> Result<(), String> {
    let n = arange_len(start, end, step);
    if n != 0 {
        arange_pipeline(dev, start, step, dtype, n)?;
    }
    Ok(())
}

pub fn warm_arange(start: f64, end: f64, step: f64, dtype: DType) -> Result<(), String> {
    compile_arange(MetalDevice::get(), start, end, step, dtype)
}

pub fn arange(
    dev: &MetalDevice,
    start: f64,
    end: f64,
    step: f64,
    dtype: DType,
) -> Result<MetalTensor, String> {
    compile_arange(dev, start, end, step, dtype)?;
    let out = MetalTensor::empty(dev, vec![arange_len(start, end, step)], dtype);
    arange_into(dev, start, end, step, &out)?;
    Ok(out)
}

pub fn arange_into(
    dev: &MetalDevice,
    start: f64,
    end: f64,
    step: f64,
    out: &MetalTensor,
) -> Result<(), String> {
    let n = arange_len(start, end, step);
    out.validate_destination("arange", &[n], out.dtype)?;
    if n == 0 {
        return Ok(());
    }
    let pipeline = precompiled_pipeline(
        dev,
        key(&[
            0xA26E,
            out.dtype as u64,
            start.to_bits(),
            step.to_bits(),
            n as u64,
        ]),
        "et_arange",
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

fn eye_pipeline(
    dev: &MetalDevice,
    n: usize,
    dtype: DType,
) -> Result<super::device::Pipeline, String> {
    let wide = MetalDevice::WIDE;
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
    dev.compile_lazy(key(&[0xE7E, dtype as u64, n as u64]), "et_eye", make_src)
}

pub fn compile_eye(dev: &MetalDevice, n: usize, dtype: DType) -> Result<(), String> {
    compile_fill(dev, &[n, n], 0.0, dtype)?;
    if n != 0 {
        eye_pipeline(dev, n, dtype)?;
    }
    Ok(())
}

pub fn warm_eye(n: usize, dtype: DType) -> Result<(), String> {
    compile_eye(MetalDevice::get(), n, dtype)
}

pub fn eye(dev: &MetalDevice, n: usize, dtype: DType) -> Result<MetalTensor, String> {
    compile_eye(dev, n, dtype)?;
    let out = MetalTensor::empty(dev, vec![n, n], dtype);
    eye_into(dev, &out)?;
    Ok(out)
}

pub fn eye_into(dev: &MetalDevice, out: &MetalTensor) -> Result<(), String> {
    let shape = out.layout.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(format!(
            "eye destination must be square rank-2 storage, got {shape:?}"
        ));
    }
    out.validate_destination("eye", shape, out.dtype)?;
    let n = shape[0];
    fill_into(dev, out, 0.0)?;
    if n == 0 {
        return Ok(());
    }
    let pipeline = precompiled_pipeline(dev, key(&[0xE7E, out.dtype as u64, n as u64]), "et_eye")?;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(
            e,
            0,
            &out.buffer,
            out.layout.offset() * out.dtype.size_in_bytes(),
        );
        e.dispatchThreads_threadsPerThreadgroup(
            MetalDevice::grid(n, 1, 1),
            MetalDevice::grid(n.min(256), 1, 1),
        );
    });
    Ok(())
}

fn argreduce_pipeline(
    dev: &MetalDevice,
    layout: &crate::runtime::layout::Layout,
    dtype: DType,
    dim: usize,
    pick_max: bool,
) -> Result<super::device::Pipeline, String> {
    let shape = layout.shape();
    let rank = shape.len();
    let n = shape[dim];
    let dstride = layout.strides()[dim];
    let kept: Vec<usize> = (0..rank).filter(|&d| d != dim).collect();
    let kept_dims: Vec<usize> = kept.iter().map(|&d| shape[d]).collect();
    let kept_strides: Vec<usize> = kept.iter().map(|&d| layout.strides()[d]).collect();
    let kept_n: usize = kept_dims.iter().product();
    let ty = msl_type(dtype);
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
    dev.compile_lazy(
        key(&[
            0xA26D,
            dtype as u64,
            dim as u64,
            pick_max as u64,
            layout_key(layout),
        ]),
        "et_argred",
        make_src,
    )
}

pub fn compile_argreduce_layout(
    dev: &MetalDevice,
    layout: &crate::runtime::layout::Layout,
    dtype: DType,
    dim: usize,
    pick_max: bool,
) -> Result<(), String> {
    if dim >= layout.shape().len() {
        return Err(format!(
            "argreduce dimension {dim} is out of range for shape {:?}",
            layout.shape()
        ));
    }
    if layout.shape()[dim] == 0 {
        return Err("argreduce cannot reduce an empty dimension".to_string());
    }
    let kept_n = layout
        .shape()
        .iter()
        .enumerate()
        .filter_map(|(dimension, &size)| (dimension != dim).then_some(size))
        .product::<usize>();
    if kept_n != 0 {
        argreduce_pipeline(dev, layout, dtype, dim, pick_max)?;
    }
    Ok(())
}

pub fn compile_argreduce(
    dev: &MetalDevice,
    shape: &[usize],
    dtype: DType,
    dim: usize,
    pick_max: bool,
) -> Result<(), String> {
    compile_argreduce_layout(
        dev,
        &crate::runtime::layout::Layout::contiguous(shape.to_vec()),
        dtype,
        dim,
        pick_max,
    )
}

pub fn warm_argreduce(
    shape: &[usize],
    dtype: DType,
    dim: usize,
    pick_max: bool,
) -> Result<(), String> {
    compile_argreduce(MetalDevice::get(), shape, dtype, dim, pick_max)
}

pub fn argreduce(
    dev: &MetalDevice,
    x: &MetalTensor,
    dim: usize,
    pick_max: bool,
) -> Result<MetalTensor, String> {
    if dim >= x.layout.shape().len() {
        return Err(format!(
            "argreduce dimension {dim} is out of range for shape {:?}",
            x.layout.shape()
        ));
    }
    let mut out_shape = x.layout.shape().to_vec();
    out_shape[dim] = 1;
    compile_argreduce_layout(dev, &x.layout, x.dtype, dim, pick_max)?;
    let out = MetalTensor::empty(dev, out_shape, DType::U32);
    argreduce_into(dev, x, dim, pick_max, &out)?;
    Ok(out)
}

pub fn argreduce_into(
    dev: &MetalDevice,
    x: &MetalTensor,
    dim: usize,
    pick_max: bool,
    out: &MetalTensor,
) -> Result<(), String> {
    if dim >= x.layout.shape().len() {
        return Err(format!(
            "argreduce dimension {dim} is out of range for shape {:?}",
            x.layout.shape()
        ));
    }
    if x.layout.shape()[dim] == 0 {
        return Err("argreduce cannot reduce an empty dimension".to_string());
    }
    let mut out_shape = x.layout.shape().to_vec();
    out_shape[dim] = 1;
    out.validate_destination("argreduce", &out_shape, DType::U32)?;
    let kept_n = out.numel();
    if kept_n == 0 {
        return Ok(());
    }
    let pipeline = precompiled_pipeline(
        dev,
        key(&[
            0xA26D,
            x.dtype as u64,
            dim as u64,
            pick_max as u64,
            layout_key(&x.layout),
        ]),
        "et_argred",
    )?;
    let padded = kept_n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &x.buffer, x.layout.offset() * x.dtype.size_in_bytes());
        set_buffer(e, 1, &out.buffer, out.layout.offset() * 4);
        {
            let (g, tg) = MetalDevice::grid_flat(padded);
            e.dispatchThreads_threadsPerThreadgroup(g, tg);
        }
    });
    Ok(())
}

fn cumsum_pipeline(
    dev: &MetalDevice,
    layout: &crate::runtime::layout::Layout,
    dtype: DType,
    dim: usize,
) -> Result<super::device::Pipeline, String> {
    let shape = layout.shape();
    let rank = shape.len();
    let n = shape[dim];
    let dstride = layout.strides()[dim];
    let kept: Vec<usize> = (0..rank).filter(|&d| d != dim).collect();
    let kept_dims: Vec<usize> = kept.iter().map(|&d| shape[d]).collect();
    let kept_strides: Vec<usize> = kept.iter().map(|&d| layout.strides()[d]).collect();
    let kept_n: usize = kept_dims.iter().product();
    let ty = msl_type(dtype);
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
    dev.compile_lazy(
        key(&[0xC50A, dtype as u64, dim as u64, layout_key(layout)]),
        "et_cumsum",
        make_src,
    )
}

pub fn compile_cumsum_layout(
    dev: &MetalDevice,
    layout: &crate::runtime::layout::Layout,
    dtype: DType,
    dim: usize,
) -> Result<(), String> {
    if dim >= layout.shape().len() {
        return Err(format!(
            "cumsum dimension {dim} is out of range for shape {:?}",
            layout.shape()
        ));
    }
    let kept_n = layout
        .shape()
        .iter()
        .enumerate()
        .filter_map(|(dimension, &size)| (dimension != dim).then_some(size))
        .product::<usize>();
    if kept_n != 0 {
        cumsum_pipeline(dev, layout, dtype, dim)?;
    }
    Ok(())
}

pub fn compile_cumsum(
    dev: &MetalDevice,
    shape: &[usize],
    dtype: DType,
    dim: usize,
) -> Result<(), String> {
    compile_cumsum_layout(
        dev,
        &crate::runtime::layout::Layout::contiguous(shape.to_vec()),
        dtype,
        dim,
    )
}

pub fn warm_cumsum(shape: &[usize], dtype: DType, dim: usize) -> Result<(), String> {
    compile_cumsum(MetalDevice::get(), shape, dtype, dim)
}

pub fn cumsum(dev: &MetalDevice, x: &MetalTensor, dim: usize) -> Result<MetalTensor, String> {
    compile_cumsum_layout(dev, &x.layout, x.dtype, dim)?;
    let out = MetalTensor::empty(dev, x.layout.shape().to_vec(), x.dtype);
    cumsum_into(dev, x, dim, &out)?;
    Ok(out)
}

pub fn cumsum_into(
    dev: &MetalDevice,
    x: &MetalTensor,
    dim: usize,
    out: &MetalTensor,
) -> Result<(), String> {
    if dim >= x.layout.shape().len() {
        return Err(format!(
            "cumsum dimension {dim} is out of range for shape {:?}",
            x.layout.shape()
        ));
    }
    out.validate_destination("cumsum", x.layout.shape(), x.dtype)?;
    let kept_n = x
        .layout
        .shape()
        .iter()
        .enumerate()
        .filter_map(|(dimension, &size)| (dimension != dim).then_some(size))
        .product::<usize>();
    if kept_n == 0 {
        return Ok(());
    }
    let pipeline = precompiled_pipeline(
        dev,
        key(&[0xC50A, x.dtype as u64, dim as u64, layout_key(&x.layout)]),
        "et_cumsum",
    )?;
    let padded = kept_n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &x.buffer, x.layout.offset() * x.dtype.size_in_bytes());
        set_buffer(
            e,
            1,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(tensor: &MetalTensor) -> Vec<u8> {
        let size = tensor.dtype.size_in_bytes();
        let offset = tensor.layout.offset() * size;
        unsafe {
            std::slice::from_raw_parts(
                tensor.buffer.contents_ptr().cast::<u8>().add(offset),
                tensor.numel() * size,
            )
            .to_vec()
        }
    }

    fn from_i64(dev: &MetalDevice, values: &[i64], shape: Vec<usize>) -> MetalTensor {
        let tensor = MetalTensor::empty(dev, shape, DType::I64);
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                tensor.buffer.contents_ptr().cast::<i64>(),
                values.len(),
            );
        }
        tensor
    }

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

    #[test]
    fn allocating_wrappers_match_into_primitives() {
        let dev = MetalDevice::new(0).unwrap();
        let x = MetalTensor::from_f32(&dev, vec![3.0, -2.0, 5.0, 1.0, 4.0, -1.0], vec![2, 3]);
        let permuted = MetalTensor {
            buffer: x.buffer.clone(),
            layout: x.layout.permute(&[1, 0]),
            dtype: x.dtype,
        };

        let cast_wrapped = cast(&dev, &x, DType::F16).unwrap();
        let cast_destination = MetalTensor::empty(&dev, vec![2, 3], DType::F16);
        cast_into(&dev, &x, &cast_destination).unwrap();

        let copy_wrapped = strided_copy(&dev, &permuted).unwrap();
        let copy_destination = MetalTensor::empty(&dev, vec![3, 2], DType::F32);
        copy_into(&dev, &permuted, &copy_destination).unwrap();

        let randn_wrapped = randn(&dev, &[8], 42).unwrap();
        let randn_destination = MetalTensor::empty(&dev, vec![8], DType::F32);
        randn_into(&dev, &randn_destination, 42).unwrap();

        let uniform_wrapped = uniform(&dev, -2.0, 3.0, &[8], 9).unwrap();
        let uniform_destination = MetalTensor::empty(&dev, vec![8], DType::F32);
        uniform_into(&dev, -2.0, 3.0, &uniform_destination, 9).unwrap();

        let arg_wrapped = argreduce(&dev, &x, 1, true).unwrap();
        let arg_destination = MetalTensor::empty(&dev, vec![2, 1], DType::U32);
        argreduce_into(&dev, &x, 1, true, &arg_destination).unwrap();

        let sum_wrapped = cumsum(&dev, &x, 1).unwrap();
        let sum_destination = MetalTensor::empty(&dev, vec![2, 3], DType::F32);
        cumsum_into(&dev, &x, 1, &sum_destination).unwrap();

        dev.synchronize().unwrap();
        for (wrapped, destination) in [
            (&cast_wrapped, &cast_destination),
            (&copy_wrapped, &copy_destination),
            (&randn_wrapped, &randn_destination),
            (&uniform_wrapped, &uniform_destination),
            (&arg_wrapped, &arg_destination),
            (&sum_wrapped, &sum_destination),
        ] {
            assert_eq!(bytes(wrapped), bytes(destination));
        }
    }

    #[test]
    fn into_primitives_use_no_planned_allocations_or_uploads() {
        let dev = MetalDevice::new(0).unwrap();
        let x = MetalTensor::from_f32(&dev, vec![3.0, -2.0, 5.0, 1.0, 4.0, -1.0], vec![2, 3]);
        let permuted = MetalTensor {
            buffer: x.buffer.clone(),
            layout: x.layout.permute(&[1, 0]),
            dtype: x.dtype,
        };
        let integers = from_i64(&dev, &[3, -2, 5, -1], vec![4]);

        let fill_buffer = dev.alloc(5, DType::F32);
        let fill_destination = MetalTensor {
            buffer: fill_buffer,
            layout: crate::runtime::layout::Layout::new(vec![3], vec![1], 1),
            dtype: DType::F32,
        };
        let relu_destination = MetalTensor::empty(&dev, vec![4], DType::I64);
        let cast_destination = MetalTensor::empty(&dev, vec![2, 3], DType::F16);
        let copy_destination = MetalTensor::empty(&dev, vec![3, 2], DType::F32);
        let randn_destination = MetalTensor::empty(&dev, vec![8], DType::F32);
        let uniform_destination = MetalTensor::empty(&dev, vec![8], DType::F32);
        let arange_destination = MetalTensor::empty(&dev, vec![3], DType::F32);
        let eye_destination = MetalTensor::empty(&dev, vec![3, 3], DType::F32);
        let arg_destination = MetalTensor::empty(&dev, vec![2, 1], DType::U32);
        let sum_destination = MetalTensor::empty(&dev, vec![2, 3], DType::F32);

        compile_fill(&dev, &[3], 7.0, DType::F32).unwrap();
        compile_relu_i64_layout(&dev, &integers.layout).unwrap();
        compile_cast_layout(&dev, &x.layout, DType::F32, DType::F16).unwrap();
        compile_copy_layout(&dev, &permuted.layout, DType::F32).unwrap();
        compile_randn(&dev, &[8]).unwrap();
        compile_uniform(&dev, -2.0, 3.0, &[8]).unwrap();
        compile_arange(&dev, 0.0, 5.0, 2.0, DType::F32).unwrap();
        compile_eye(&dev, 3, DType::F32).unwrap();
        compile_argreduce_layout(&dev, &x.layout, DType::F32, 1, true).unwrap();
        compile_cumsum_layout(&dev, &x.layout, DType::F32, 1).unwrap();

        let _dispatch_guard = dev.begin_executable_dispatch().unwrap();

        fill_into(&dev, &fill_destination, 7.0).unwrap();
        relu_i64_into(&dev, &integers, &relu_destination).unwrap();
        cast_into(&dev, &x, &cast_destination).unwrap();
        copy_into(&dev, &permuted, &copy_destination).unwrap();
        randn_into(&dev, &randn_destination, 42).unwrap();
        uniform_into(&dev, -2.0, 3.0, &uniform_destination, 9).unwrap();
        arange_into(&dev, 0.0, 5.0, 2.0, &arange_destination).unwrap();
        eye_into(&dev, &eye_destination).unwrap();
        argreduce_into(&dev, &x, 1, true, &arg_destination).unwrap();
        cumsum_into(&dev, &x, 1, &sum_destination).unwrap();

        let result = dev.synchronize();
        result.unwrap();
        assert_eq!(fill_destination.buffer.read_f32(1, 3), vec![7.0; 3]);
    }

    #[test]
    fn into_requires_the_exact_precompiled_pipeline() {
        let dev = MetalDevice::new(0).unwrap();
        let out = MetalTensor::empty(&dev, vec![17], DType::F32);
        let error = fill_into(&dev, &out, 2.0).unwrap_err();
        assert!(error.contains("not precompiled"), "{error}");

        compile_fill(&dev, &[16], 2.0, DType::F32).unwrap();
        let error = fill_into(&dev, &out, 2.0).unwrap_err();
        assert!(error.contains("not precompiled"), "{error}");

        compile_fill(&dev, &[17], 2.0, DType::F32).unwrap();
        fill_into(&dev, &out, 2.0).unwrap();
        dev.synchronize().unwrap();
        assert_eq!(out.buffer.read_f32(0, 17), vec![2.0; 17]);
    }
}

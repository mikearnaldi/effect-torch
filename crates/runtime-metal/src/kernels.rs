//! Primitive kernels: fill, relu-i64, cast, strided copy, random
//! (randn/uniform), arange, eye, argreduce, cumsum.
//!
//! # Conventions
//!
//! - Each operation has a private `*_pipeline` builder for its layout, dtype,
//!   and sizes. A public `compile_*` or `warm_*` function precompiles it.
//!   Allocating wrappers create destinations, while `*_into` functions
//!   validate and dispatch without allocating.
//! - Strided sources bake stride decomposition into the emitted source. The
//!   pipeline key hashes shape and strides, so each layout gets a different
//!   kernel. Destinations are contiguous.
//! - These kernels support f32, f16, bf16, u8, u32, and i64. The fusion emitter
//!   supports f32, f16, and bf16. f64 has MSL syntax here but the value boundary
//!   rejects it because Metal does not support it. Integer fill and arange use
//!   64-bit arithmetic because values above 2^24 have no exact f32 form.
//! - Dispatch uses one thread per output element over a padded flat grid
//!   ([`MetalDevice::grid_flat`]) and widens to 64-bit indexing past
//!   `u32::MAX`. Argreduce and cumsum use one thread per kept slice and loop
//!   serially over the reduced dimension.
//! - randn and uniform use per-thread xoroshiro128+ seeded from the global seed
//!   and element index. Results are deterministic for a seed regardless of
//!   dispatch shape.

use super::device::{set_buffer, set_bytes, Buffer, MetalDevice};
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

fn select_key(
    layouts: &[crate::runtime::layout::Layout; 3],
    condition: DType,
    dtype: DType,
) -> u64 {
    key(&[
        0x5E1EC7,
        condition as u64,
        dtype as u64,
        layout_key(&layouts[0]),
        layout_key(&layouts[1]),
        layout_key(&layouts[2]),
    ])
}

/// Precompile a typed select. Conditions retain their own dtype so nonzero
/// values cannot become zero through a narrowing conversion.
pub(crate) fn compile_select(
    dev: &MetalDevice,
    layouts: &[crate::runtime::layout::Layout; 3],
    condition: DType,
    dtype: DType,
) -> Result<(), String> {
    let n = layouts[0].numel();
    if n == 0 {
        return Ok(());
    }
    let cty = msl_type(condition);
    let ty = msl_type(dtype);
    let wide = MetalDevice::WIDE;
    dev.compile_lazy(select_key(layouts, condition, dtype), "et_select", || {
        let c = source_offset(&layouts[0], "i", "        ").replace("src_off", "c_off");
        let a = source_offset(&layouts[1], "i", "        ").replace("src_off", "a_off");
        let b = source_offset(&layouts[2], "i", "        ").replace("src_off", "b_off");
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_select(device const {cty}* cond [[buffer(0)]],
                      device const {ty}* a [[buffer(1)]],
                      device const {ty}* b [[buffer(2)]],
                      device {ty}* out [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {{
    const ulong i = ulong(gid.y) * {wide}ul + ulong(gid.x);
    if (i < {n}ul) {{
{c}{a}{b}        out[i] = cond[c_off] != {cty}(0) ? a[a_off] : b[b_off];
    }}
}}
"#
        )
    })?;
    Ok(())
}

pub(crate) fn select_into(
    dev: &MetalDevice,
    inputs: [&MetalTensor; 3],
    out: &MetalTensor,
) -> Result<(), String> {
    let layouts = inputs.map(|input| input.layout.broadcast_to(out.layout.shape()));
    let n = out.numel();
    if n == 0 {
        return Ok(());
    }
    let pipeline = precompiled_pipeline(
        dev,
        select_key(&layouts, inputs[0].dtype, out.dtype),
        "et_select",
    )?;
    dev.with_encoder(|encoder| {
        encoder.setComputePipelineState(pipeline.as_raw());
        for (index, input) in inputs.iter().enumerate() {
            set_buffer(
                encoder,
                index,
                &input.buffer,
                input.layout.offset() * input.dtype.size_in_bytes(),
            );
        }
        set_buffer(
            encoder,
            3,
            &out.buffer,
            out.layout.offset() * out.dtype.size_in_bytes(),
        );
        let (grid, group) = MetalDevice::grid_flat(n.div_ceil(256) * 256);
        encoder.dispatchThreads_threadsPerThreadgroup(grid, group);
    });
    Ok(())
}

fn integer_binary_key(
    layouts: &[crate::runtime::layout::Layout; 2],
    dtype: DType,
    op: crate::ops::BinOp,
) -> u64 {
    key(&[
        0x1ADD,
        op as u64,
        dtype as u64,
        layout_key(&layouts[0]),
        layout_key(&layouts[1]),
    ])
}

/// Exact integer comparisons and wrapping arithmetic, with no F32 intermediates.
pub(crate) fn compile_integer_binary(
    dev: &MetalDevice,
    layouts: &[crate::runtime::layout::Layout; 2],
    dtype: DType,
    op: crate::ops::BinOp,
) -> Result<(), String> {
    if dtype.is_float() {
        return Err("integer binary requires an integer dtype".to_string());
    }
    let n = layouts[0].numel();
    if n == 0 {
        return Ok(());
    }
    let ty = msl_type(dtype);
    let carrier = if dtype == DType::I64 { "ulong" } else { "uint" };
    use crate::ops::BinOp;
    let expression = match op {
        BinOp::Add => format!("{ty}({carrier}(av) + {carrier}(bv))"),
        BinOp::Sub => format!("{ty}({carrier}(av) - {carrier}(bv))"),
        BinOp::Mul => format!("{ty}({carrier}(av) * {carrier}(bv))"),
        BinOp::Min => "min(av, bv)".to_string(),
        BinOp::Max => "max(av, bv)".to_string(),
        BinOp::Eq => "av == bv".to_string(),
        BinOp::Ne => "av != bv".to_string(),
        BinOp::Lt => "av < bv".to_string(),
        BinOp::Le => "av <= bv".to_string(),
        BinOp::Gt => "av > bv".to_string(),
        BinOp::Ge => "av >= bv".to_string(),
        BinOp::Div => return Err("integer division is unsupported on Metal".to_string()),
    };
    let out_ty = if op.is_comparison() { "uchar" } else { ty };
    let wide = MetalDevice::WIDE;
    dev.compile_lazy(integer_binary_key(layouts, dtype, op), "et_integer_binary", || {
        let a = source_offset(&layouts[0], "i", "        ").replace("src_off", "a_off");
        let b = source_offset(&layouts[1], "i", "        ").replace("src_off", "b_off");
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_integer_binary(device const {ty}* a [[buffer(0)]],
                           device const {ty}* b [[buffer(1)]],
                           device {out_ty}* out [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {{
    const ulong i = ulong(gid.y) * {wide}ul + ulong(gid.x);
    if (i < {n}ul) {{
{a}{b}        const {ty} av = a[a_off], bv = b[b_off];
        out[i] = {expression};
    }}
}}
"#
        )
    })?;
    Ok(())
}

pub(crate) fn integer_binary_into(
    dev: &MetalDevice,
    a: &MetalTensor,
    b: &MetalTensor,
    op: crate::ops::BinOp,
    out: &MetalTensor,
) -> Result<(), String> {
    if a.dtype != b.dtype || a.dtype.is_float() {
        return Err("integer binary requires matching integer dtypes".to_string());
    }
    let layouts = [a, b].map(|input| input.layout.broadcast_to(out.layout.shape()));
    let n = out.numel();
    if n == 0 {
        return Ok(());
    }
    let pipeline = precompiled_pipeline(
        dev,
        integer_binary_key(&layouts, a.dtype, op),
        "et_integer_binary",
    )?;
    dev.with_encoder(|encoder| {
        encoder.setComputePipelineState(pipeline.as_raw());
        for (index, input) in [a, b, out].iter().enumerate() {
            set_buffer(
                encoder,
                index,
                &input.buffer,
                input.layout.offset() * input.dtype.size_in_bytes(),
            );
        }
        let (grid, group) = MetalDevice::grid_flat(n.div_ceil(256) * 256);
        encoder.dispatchThreads_threadsPerThreadgroup(grid, group);
    });
    Ok(())
}

fn integer_reduce_layouts(
    layout: &crate::runtime::layout::Layout,
    dims: &[usize],
) -> [crate::runtime::layout::Layout; 2] {
    [false, true].map(|reduced| {
        let axes = (0..layout.rank())
            .filter(|axis| dims.contains(axis) == reduced)
            .collect::<Vec<_>>();
        crate::runtime::layout::Layout::new(
            axes.iter().map(|&axis| layout.shape()[axis]).collect(),
            axes.iter().map(|&axis| layout.strides()[axis]).collect(),
            0,
        )
    })
}

fn integer_reduce_key(
    layouts: &[crate::runtime::layout::Layout; 2],
    dtype: DType,
    op: crate::fusion::ReduceOp,
) -> u64 {
    key(&[
        0x1ED0CE,
        dtype as u64,
        op as u64,
        layout_key(&layouts[0]),
        layout_key(&layouts[1]),
    ])
}

/// Exact min/max reduction with signed comparisons and strided input access.
pub(crate) fn compile_integer_reduce(
    dev: &MetalDevice,
    layout: &crate::runtime::layout::Layout,
    dtype: DType,
    dims: &[usize],
    op: crate::fusion::ReduceOp,
) -> Result<(), String> {
    use crate::fusion::ReduceOp;
    if dtype.is_float() || !matches!(op, ReduceOp::Min | ReduceOp::Max) {
        return Err("integer reduction requires min or max on integer storage".to_string());
    }
    let layouts = integer_reduce_layouts(layout, dims);
    let n = layouts[0].numel();
    let reduced = layouts[1].numel();
    if n == 0 {
        return Ok(());
    }
    let ty = msl_type(dtype);
    let pick_max = matches!(op, ReduceOp::Max);
    let identity = match (dtype, pick_max) {
        (DType::I64, true) => "long(0x8000000000000000ul)",
        (DType::I64, false) => "long(0x7ffffffffffffffful)",
        (DType::U32, false) => "0xffffffffu",
        (DType::U8, false) => "uchar(255)",
        _ => "0",
    };
    let combine = if pick_max { "max" } else { "min" };
    let wide = MetalDevice::WIDE;
    dev.compile_lazy(integer_reduce_key(&layouts, dtype, op), "et_integer_reduce", || {
        let outer = source_offset(&layouts[0], "i", "        ").replace("src_off", "outer_off");
        let inner = if reduced == 0 { String::new() } else {
            source_offset(&layouts[1], "r", "            ").replace("src_off", "inner_off")
                + &format!("            acc = {combine}(acc, src[outer_off + inner_off]);\n")
        };
        format!(r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_integer_reduce(device const {ty}* src [[buffer(0)]],
                              device {ty}* out [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {{
    const ulong i = ulong(gid.y) * {wide}ul + ulong(gid.x);
    if (i < {n}ul) {{
{outer}        {ty} acc = {identity};
        for (ulong r = 0; r < {reduced}ul; ++r) {{
{inner}        }}
        out[i] = acc;
    }}
}}
"#)
    })?;
    Ok(())
}

pub(crate) fn integer_reduce_into(
    dev: &MetalDevice,
    input: &MetalTensor,
    dims: &[usize],
    op: crate::fusion::ReduceOp,
    out: &MetalTensor,
) -> Result<(), String> {
    let layouts = integer_reduce_layouts(&input.layout, dims);
    let n = out.numel();
    if n == 0 {
        return Ok(());
    }
    let pipeline = precompiled_pipeline(
        dev,
        integer_reduce_key(&layouts, input.dtype, op),
        "et_integer_reduce",
    )?;
    dev.with_encoder(|encoder| {
        encoder.setComputePipelineState(pipeline.as_raw());
        for (index, tensor) in [input, out].iter().enumerate() {
            set_buffer(
                encoder,
                index,
                &tensor.buffer,
                tensor.layout.offset() * tensor.dtype.size_in_bytes(),
            );
        }
        let (grid, group) = MetalDevice::grid_flat(n.div_ceil(256) * 256);
        encoder.dispatchThreads_threadsPerThreadgroup(grid, group);
    });
    Ok(())
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

/// Precompiles the fill pipeline for `shape`/`dtype` (no-op when empty).
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

/// [`compile_fill`] against the process-wide device.
pub fn warm_fill(shape: &[usize], value: f64, dtype: DType) -> Result<(), String> {
    compile_fill(MetalDevice::get(), shape, value, dtype)
}

/// Fills `out` with `value`. Integer dtypes cast from the f64 bits
/// exactly; the fill value never round-trips through f32 for them.
/// Requires the precompiled pipeline for the exact element count/dtype.
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

/// [`fill_into`] that precompiles first.
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

/// Precompiles the i64 relu pipeline for an exact (possibly strided)
/// layout.
pub fn compile_relu_i64_layout(
    dev: &MetalDevice,
    layout: &crate::runtime::layout::Layout,
) -> Result<(), String> {
    if layout.numel() != 0 {
        relu_i64_pipeline(dev, layout)?;
    }
    Ok(())
}

/// [`compile_relu_i64_layout`] for a contiguous layout of `shape`.
pub fn compile_relu_i64(dev: &MetalDevice, shape: &[usize]) -> Result<(), String> {
    compile_relu_i64_layout(
        dev,
        &crate::runtime::layout::Layout::contiguous(shape.to_vec()),
    )
}

/// [`compile_relu_i64`] against the process-wide device.
pub fn warm_relu_i64(shape: &[usize]) -> Result<(), String> {
    compile_relu_i64(MetalDevice::get(), shape)
}

/// Allocating i64 relu: clamps each element at zero.
pub fn relu_i64(dev: &MetalDevice, x: &MetalTensor) -> Result<MetalTensor, String> {
    compile_relu_i64_layout(dev, &x.layout)?;
    let out = MetalTensor::empty(dev, x.layout.shape().to_vec(), DType::I64);
    relu_i64_into(dev, x, &out)?;
    Ok(out)
}

/// Destination form of [`relu_i64`]; requires the precompiled pipeline.
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

// Shared by materialized casts and fused Cast/RoundTo boundaries. Native
// bfloat construction flushes F32 subnormals even with fast math disabled.
pub(crate) const BF16_CONVERSION_MSL: &str = r#"
inline bfloat et_bf16_from_float(float value) {
    const uint bits = as_type<uint>(value);
    const uint magnitude = bits & 0x7fffffffu;
    const ushort rounded = magnitude > 0x7f800000u
        ? ushort((bits >> 16) | 0x0040u)
        : ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
    return as_type<bfloat>(rounded);
}
inline float et_bf16_to_float(bfloat value) {
    return as_type<float>(uint(as_type<ushort>(value)) << 16);
}
"#;

fn cast_pipeline(
    dev: &MetalDevice,
    layout: &crate::runtime::layout::Layout,
    source: DType,
    destination: DType,
) -> Result<super::device::Pipeline, String> {
    let wide = MetalDevice::WIDE;
    let n = layout.numel();
    let (src_ty, dst_ty) = (msl_type(source), msl_type(destination));
    let conversion = if source == DType::F32 && destination == DType::BF16 {
        "out[i] = et_bf16_from_float(a[src_off]);".to_string()
    } else if source == DType::BF16 && destination == DType::F32 {
        "out[i] = et_bf16_to_float(a[src_off]);".to_string()
    } else if source == DType::I64 && destination == DType::F16 {
        // MSL long -> half overflows immediately above 65504. RNE instead
        // overflows at 65520. Every integer in the finite range is exact in
        // F32, so this bounded conversion introduces no intermediate rounding.
        "const long value = a[src_off];
        out[i] = value >= 65520l ? half(INFINITY) : value <= -65520l ? half(-INFINITY) : half(float(value));".to_string()
    } else if source.is_float() && !destination.is_float() {
        // F16/BF16 widen exactly to F32. Compare against powers of two rather
        // than integer maxima rounded up to an unrepresentable conversion.
        let (minimum, maximum, lower, upper) = match destination {
            DType::I64 => (
                "long(0x8000000000000000ul)",
                "long(0x7ffffffffffffffful)",
                "-9223372036854775808.0f",
                "9223372036854775808.0f",
            ),
            DType::U32 => ("0u", "0xffffffffu", "0.0f", "4294967296.0f"),
            DType::U8 => ("uchar(0)", "uchar(255)", "0.0f", "256.0f"),
            _ => unreachable!("integer destination"),
        };
        format!("const float value = float(a[src_off]);
        out[i] = isnan(value) ? {dst_ty}(0) : value <= {lower} ? {minimum} : value >= {upper} ? {maximum} : {dst_ty}(value);")
    } else {
        format!("out[i] = ({dst_ty})a[src_off];")
    };
    let offset = source_offset(layout, "i", "        ");
    let make_src = || {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;
{BF16_CONVERSION_MSL}
kernel void et_cast(device const {src_ty}* a [[buffer(0)]], device {dst_ty}* out [[buffer(1)]], uint2 gid2 [[thread_position_in_grid]]) {{
    const ulong i = ulong(gid2.y) * {wide}ul + ulong(gid2.x);
    if (i < {n}ul) {{
{offset}        {conversion}
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

/// Precompiles the cast (or copy, when source == destination) pipeline
/// for an exact layout and dtype pair.
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

/// [`compile_cast_layout`] for a contiguous layout of `shape`.
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

/// [`compile_cast`] against the process-wide device.
pub fn warm_cast(shape: &[usize], source: DType, destination: DType) -> Result<(), String> {
    compile_cast(MetalDevice::get(), shape, source, destination)
}

/// Allocating cast; a same-dtype cast aliases the input (no copy).
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

/// Destination form of [`cast`]; same-dtype degenerates to
/// [`copy_into`]. Requires the precompiled pipeline.
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

/// Materializes any (possibly strided/offset) tensor as a fresh
/// contiguous tensor; contiguous offset-zero inputs are returned as-is.
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

/// [`compile_copy_layout`] for a contiguous layout of `shape`.
pub fn compile_copy(dev: &MetalDevice, shape: &[usize], dtype: DType) -> Result<(), String> {
    let layout = crate::runtime::layout::Layout::contiguous(shape.to_vec());
    compile_copy_layout(dev, &layout, dtype)
}

/// Precompiles the strided-copy pipeline for an exact source layout.
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

/// [`compile_copy`] against the process-wide device.
pub fn warm_copy(shape: &[usize], dtype: DType) -> Result<(), String> {
    compile_copy(MetalDevice::get(), shape, dtype)
}

/// [`compile_copy_layout`] against the process-wide device.
pub fn warm_copy_layout(
    layout: &crate::runtime::layout::Layout,
    dtype: DType,
) -> Result<(), String> {
    compile_copy_layout(MetalDevice::get(), layout, dtype)
}

/// Copies `source` into `destination` (same shape/dtype; the source may
/// be strided, the destination must be contiguous). Requires the
/// precompiled pipeline.
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

/// Copies `bytes` from `source` at `source_offset` into `destination` at
/// `destination_offset` with a flat device kernel on the current stream.
/// Used for GPU-ordered state-transaction copies between deferred
/// invocations: the stream's per-dispatch barriers order it after the
/// kernels that produced `source` and before later readers of
/// `destination`, so no host fence is needed. Both buffers must be large
/// enough; the caller (executable state commits) validates the ranges.
pub fn copy_bytes_into(
    dev: &MetalDevice,
    source: &Buffer,
    source_offset: usize,
    destination: &Buffer,
    destination_offset: usize,
    bytes: usize,
) -> Result<(), String> {
    if bytes == 0 {
        return Ok(());
    }
    if source_offset.saturating_add(bytes) > source.size
        || destination_offset.saturating_add(bytes) > destination.size
    {
        return Err("metal byte copy exceeds its buffer".to_string());
    }
    let wide = MetalDevice::WIDE;
    let pipeline = dev.compile_lazy(
        key(&[0xBC09]),
        "et_bcopy",
        || {
            format!(
                r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_bcopy(device const uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]], constant ulong& n [[buffer(2)]], uint2 gid2 [[thread_position_in_grid]]) {{
    const ulong i = ulong(gid2.y) * {wide}ul + ulong(gid2.x);
    const ulong base = i * 4ul;
    if (base < n) {{
        const ulong end = min(base + 4ul, n);
        for (ulong j = base; j < end; j++) {{
            dst[j] = src[j];
        }}
    }}
}}
"#
            )
        },
    )?;
    let words = bytes.div_ceil(4);
    let padded = words.div_ceil(256) * 256;
    let n = bytes as u64;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, source, source_offset);
        set_buffer(e, 1, destination, destination_offset);
        set_bytes(e, 2, &n);
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

/// Precompiles the deterministic randn pipeline for a given element count.
pub fn compile_randn(dev: &MetalDevice, shape: &[usize]) -> Result<(), String> {
    if shape.iter().product::<usize>() != 0 {
        randn_pipeline(dev, shape.iter().product())?;
    }
    Ok(())
}

/// [`compile_randn`] against the process-wide device.
pub fn warm_randn(shape: &[usize]) -> Result<(), String> {
    compile_randn(MetalDevice::get(), shape)
}

/// Allocates an f32 tensor of `shape` filled with standard-normal samples
/// from `seed` using Box-Muller over per-element seeded xoroshiro128+.
pub fn randn(dev: &MetalDevice, shape: &[usize], seed: u64) -> Result<MetalTensor, String> {
    compile_randn(dev, shape)?;
    let out = MetalTensor::empty(dev, shape.to_vec(), DType::F32);
    randn_into(dev, &out, seed)?;
    Ok(out)
}

/// Destination form of [`randn`]; requires the precompiled pipeline.
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

/// Allocates an f32 tensor of `shape` filled with uniform samples from
/// `[lo, hi)`.
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

/// Destination form of [`uniform`]; requires the precompiled pipeline for
/// the exact (n, lo, hi) triple.
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

/// Precompiles the uniform pipeline for (n, lo, hi).
pub fn compile_uniform(dev: &MetalDevice, lo: f64, hi: f64, shape: &[usize]) -> Result<(), String> {
    if shape.iter().product::<usize>() != 0 {
        uniform_pipeline(dev, lo, hi, shape.iter().product())?;
    }
    Ok(())
}

/// [`compile_uniform`] against the process-wide device.
pub fn warm_uniform(lo: f64, hi: f64, shape: &[usize]) -> Result<(), String> {
    compile_uniform(MetalDevice::get(), lo, hi, shape)
}

/// Number of elements `arange(start, end, step)` produces
/// (`ceil((end - start) / step)`, clamped at zero).
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

/// Precompiles the arange pipeline for (start, step, dtype, n).
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

/// [`compile_arange`] against the process-wide device.
pub fn warm_arange(start: f64, end: f64, step: f64, dtype: DType) -> Result<(), String> {
    compile_arange(MetalDevice::get(), start, end, step, dtype)
}

/// Allocates the `start, start + step, ...` sequence (integer dtypes
/// compute in exact 64-bit integer arithmetic).
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

/// Destination form of [`arange`]; requires the precompiled pipeline.
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

/// Precompiles the eye pipeline (and the zero fill that precedes it).
pub fn compile_eye(dev: &MetalDevice, n: usize, dtype: DType) -> Result<(), String> {
    compile_fill(dev, &[n, n], 0.0, dtype)?;
    if n != 0 {
        eye_pipeline(dev, n, dtype)?;
    }
    Ok(())
}

/// [`compile_eye`] against the process-wide device.
pub fn warm_eye(n: usize, dtype: DType) -> Result<(), String> {
    compile_eye(MetalDevice::get(), n, dtype)
}

/// Allocates the n×n identity matrix.
pub fn eye(dev: &MetalDevice, n: usize, dtype: DType) -> Result<MetalTensor, String> {
    compile_eye(dev, n, dtype)?;
    let out = MetalTensor::empty(dev, vec![n, n], dtype);
    eye_into(dev, &out)?;
    Ok(out)
}

/// Destination form of [`eye`]: zero-fills `out` then writes the diagonal.
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
    let parallel = dtype == DType::F32 && dstride == 1 && n >= 1024;
    let make_src = || {
        if parallel {
            return format!(
                r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_argred(
    device const float* x [[buffer(0)]],
    device uint* out [[buffer(1)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {{
    if (gid >= {kept_n}u) return;
    ulong base = 0ul;
{decompose}    uint best = tid;
    float best_v = x[base + tid];
    for (uint i = tid + 256u; i < {n}u; i += 256u) {{
        float v = x[base + i];
        if (v {cmp} best_v) {{ best_v = v; best = i; }}
    }}
    threadgroup float values[256];
    threadgroup uint indexes[256];
    values[tid] = best_v;
    indexes[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1u) {{
        if (tid < offset) {{
            float v = values[tid + offset];
            uint i = indexes[tid + offset];
            if (v {cmp} values[tid] || (v == values[tid] && i < indexes[tid])) {{
                values[tid] = v;
                indexes[tid] = i;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (tid == 0u) out[gid] = indexes[0];
}}
"#
            );
        }
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

/// Precompiles the argreduce pipeline for an exact layout/dim/direction.
/// Errors on an out-of-range or empty reduced dimension.
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

/// [`compile_argreduce_layout`] for a contiguous layout of `shape`.
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

/// [`compile_argreduce`] against the process-wide device.
pub fn warm_argreduce(
    shape: &[usize],
    dtype: DType,
    dim: usize,
    pick_max: bool,
) -> Result<(), String> {
    compile_argreduce(MetalDevice::get(), shape, dtype, dim, pick_max)
}

/// Allocating argreduce: per kept slice, the index of the max (or min,
/// when `pick_max` is false) element along `dim`, as u32 with `keepdim`
/// shape.
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

/// Destination form of [`argreduce`]; requires the precompiled pipeline.
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
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &x.buffer, x.layout.offset() * x.dtype.size_in_bytes());
        set_buffer(e, 1, &out.buffer, out.layout.offset() * 4);
        if x.dtype == DType::F32 && x.layout.strides()[dim] == 1 && x.layout.shape()[dim] >= 1024 {
            e.dispatchThreadgroups_threadsPerThreadgroup(
                MetalDevice::grid(kept_n, 1, 1),
                MetalDevice::grid(256, 1, 1),
            );
        } else {
            let padded = kept_n.div_ceil(256) * 256;
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
    let accumulator = if dtype.is_float() { "float" } else { ty };
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
{decompose}    {accumulator} acc = ({accumulator})0;
    for (uint i = 0u; i < {n}u; ++i) {{
        acc += {accumulator}(x[base + ulong(i) * {dstride}ul]);
        out[obase + ulong(i) * {os_dim}ul] = {ty}(acc);
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

/// Precompiles the cumsum pipeline for an exact layout/dim. Errors on an
/// out-of-range dimension.
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

/// [`compile_cumsum_layout`] for a contiguous layout of `shape`.
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

/// [`compile_cumsum`] against the process-wide device.
pub fn warm_cumsum(shape: &[usize], dtype: DType, dim: usize) -> Result<(), String> {
    compile_cumsum(MetalDevice::get(), shape, dtype, dim)
}

/// Allocating inclusive prefix sum along `dim`. Each slice uses one serial
/// thread, making the operation deterministic and O(n) per slice.
pub fn cumsum(dev: &MetalDevice, x: &MetalTensor, dim: usize) -> Result<MetalTensor, String> {
    compile_cumsum_layout(dev, &x.layout, x.dtype, dim)?;
    let out = MetalTensor::empty(dev, x.layout.shape().to_vec(), x.dtype);
    cumsum_into(dev, x, dim, &out)?;
    Ok(out)
}

/// Destination form of [`cumsum`]; requires the precompiled pipeline.
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
        // SAFETY: tests call this only after `dev.synchronize()`, so the
        // GPU is done writing; the buffer is shared-mode and the layout
        // range fits the allocation.
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
        // SAFETY: fresh unique buffer with no GPU work encoded against it
        // yet; `values.len() == numel` by construction, and shared-mode
        // contents are host-writable.
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
    fn integer_cast_to_f16_rounds_at_the_overflow_boundary() {
        let dev = MetalDevice::get();
        let values = [
            65504,
            65505,
            65519,
            65520,
            65521,
            -65504,
            -65505,
            -65519,
            -65520,
            -65521,
            i64::MAX,
            i64::MIN,
        ];
        let source = from_i64(dev, &values, vec![values.len()]);
        let half = cast(dev, &source, DType::F16).unwrap();
        let unsigned = MetalTensor {
            buffer: dev.alloc_with_data_u32(&[
                65504,
                65505,
                65519,
                65520,
                65521,
                1 << 31,
                u32::MAX,
            ]),
            layout: crate::runtime::layout::Layout::contiguous(vec![7]),
            dtype: DType::U32,
        };
        let unsigned_half = cast(dev, &unsigned, DType::F16).unwrap();
        dev.synchronize().unwrap();
        let bits = |tensor: &MetalTensor| {
            bytes(tensor)
                .chunks_exact(2)
                .map(|pair| u16::from_ne_bytes([pair[0], pair[1]]))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            bits(&half),
            [
                0x7bff, 0x7bff, 0x7bff, 0x7c00, 0x7c00, 0xfbff, 0xfbff, 0xfbff, 0xfc00, 0xfc00,
                0x7c00, 0xfc00
            ]
        );
        assert_eq!(
            bits(&unsigned_half),
            [0x7bff, 0x7bff, 0x7bff, 0x7c00, 0x7c00, 0x7c00, 0x7c00]
        );
    }

    #[test]
    fn dense_cast_modes_preserve_bits_wrap_saturate_and_round_once() {
        let dev = MetalDevice::get();
        let upload = |data: &[u8], dtype, count| MetalTensor {
            buffer: dev.upload_bytes(data),
            layout: crate::runtime::layout::Layout::contiguous(vec![count]),
            dtype,
        };
        // Identity is a bit copy even for signaling NaNs and signed zero.
        for (dtype, bits) in [
            (DType::F16, [0x7c01u16, 0xfe55, 0x8000, 1]),
            (DType::BF16, [0x7f81, 0xffa5, 0x8000, 1]),
        ] {
            let data = bits
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>();
            let source = upload(&data, dtype, bits.len());
            let output = MetalTensor::empty(dev, vec![bits.len()], dtype);
            compile_cast_layout(dev, &source.layout, dtype, dtype).unwrap();
            cast_into(dev, &source, &output).unwrap();
            dev.synchronize().unwrap();
            assert_eq!(bytes(&output), data);
        }
        let integer = from_i64(
            dev,
            &[i64::MIN, -1, 256, (1i64 << 62) + 257, i64::MAX],
            vec![5],
        );
        for dtype in [DType::U8, DType::U32] {
            let result = cast(dev, &integer, dtype).unwrap();
            dev.synchronize().unwrap();
            let expected = if dtype == DType::U8 {
                vec![0, 255, 0, 1, 255]
            } else {
                [0u32, u32::MAX, 256, 257, u32::MAX]
                    .iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect()
            };
            assert_eq!(bytes(&result), expected);
        }
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let floats = MetalTensor::from_f32(
                dev,
                vec![
                    f32::NEG_INFINITY,
                    -1.75,
                    -0.0,
                    0.0,
                    1.75,
                    f32::INFINITY,
                    f32::NAN,
                ],
                vec![7],
            );
            let source = cast(dev, &floats, dtype).unwrap();
            for destination in [DType::U8, DType::U32, DType::I64] {
                let result = cast(dev, &source, destination).unwrap();
                dev.synchronize().unwrap();
                let expected: Vec<u8> = match destination {
                    DType::U8 => vec![0, 0, 0, 0, 1, u8::MAX, 0],
                    DType::U32 => [0u32, 0, 0, 0, 1, u32::MAX, 0]
                        .iter()
                        .flat_map(|x| x.to_ne_bytes())
                        .collect(),
                    _ => [i64::MIN, -1, 0, 0, 1, i64::MAX, 0]
                        .iter()
                        .flat_map(|x| x.to_ne_bytes())
                        .collect(),
                };
                assert_eq!(bytes(&result), expected, "{dtype:?} -> {destination:?}");
            }
        }
        // Midpoint + sticky bit distinguishes one rounding from an F32 detour.
        for (dtype, shift, step) in [(DType::BF16, 23, 24), (DType::F32, 7, 8)] {
            let values = [(1u32 << 31) + (1 << shift), (1u32 << 31) + (1 << shift) + 1];
            let source = upload(
                &values
                    .iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect::<Vec<_>>(),
                DType::U32,
                2,
            );
            let rounded = cast(dev, &source, dtype).unwrap();
            let wide = cast(dev, &rounded, DType::F32).unwrap();
            dev.synchronize().unwrap();
            assert_eq!(
                wide.read_f32().unwrap(),
                [(1u64 << 31) as f32, ((1u64 << 31) + (1 << step)) as f32]
            );
        }
    }

    #[test]
    fn float_to_integer_casts_saturate_at_exact_destination_bounds() {
        let dev = MetalDevice::get();
        let values = [
            f32::from_bits(0xdf000001),
            -9223372036854775808.0,
            f32::from_bits(0xdeffffff),
            f32::from_bits(0x5effffff),
            9223372036854775808.0,
            f32::from_bits(0x5f000001),
            f32::from_bits(0x4f7fffff),
            4294967296.0,
            f32::from_bits(0x4f800001),
            -256.75,
            -1.75,
            -0.75,
            0.75,
            254.75,
            255.75,
            256.0,
            f32::MAX,
            -f32::MAX,
        ];
        let source = MetalTensor::from_f32(dev, values.to_vec(), vec![values.len()]);
        for destination in [DType::I64, DType::U32, DType::U8] {
            let output = cast(dev, &source, destination).unwrap();
            dev.synchronize().unwrap();
            // Rust's float-to-integer casts specify truncate-and-saturate.
            let expected: Vec<u8> = values
                .iter()
                .flat_map(|&value| match destination {
                    DType::I64 => (value as i64).to_ne_bytes().to_vec(),
                    DType::U32 => (value as u32).to_ne_bytes().to_vec(),
                    _ => vec![value as u8],
                })
                .collect();
            assert_eq!(bytes(&output), expected, "F32 -> {destination:?}");
        }
    }

    #[test]
    fn float_casts_preserve_gradual_underflow_and_ties_to_even() {
        let dev = MetalDevice::get();
        for (destination, input_bits, expected) in [
            (
                DType::F16,
                vec![
                    0u32, 0x80000000, 0x33000000, 0x33000001, 0x33800000, 0xb3000000, 0xb3000001,
                ],
                vec![0u16, 0x8000, 0, 1, 1, 0x8000, 0x8001],
            ),
            (
                DType::BF16,
                vec![
                    0u32, 0x80000000, 0x00008000, 0x00008001, 0x00018000, 0x80008000, 0x80008001,
                ],
                vec![0u16, 0x8000, 0, 1, 2, 0x8000, 0x8001],
            ),
        ] {
            let input = MetalTensor::from_f32(
                dev,
                input_bits.into_iter().map(f32::from_bits).collect(),
                vec![expected.len()],
            );
            let output = cast(dev, &input, destination).unwrap();
            dev.synchronize().unwrap();
            assert_eq!(
                bytes(&output),
                expected
                    .iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect::<Vec<_>>(),
                "F32 -> {destination:?}"
            );
        }
        for (source, bits) in [
            (
                DType::F16,
                vec![0u16, 0x8000, 1, 0x8001, 0x7bff, 0x7c00, 0xfc00],
            ),
            (
                DType::BF16,
                vec![
                    0u16, 0x8000, 1, 0x8001, 0x3300, 0x3301, 0x3380, 0x477f, 0x4780, 0x7f80, 0xff80,
                ],
            ),
        ] {
            let data = bits
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>();
            let input = MetalTensor {
                buffer: dev.upload_bytes(&data),
                layout: crate::runtime::layout::Layout::contiguous(vec![bits.len()]),
                dtype: source,
            };
            let values = bits
                .iter()
                .map(|&bits| {
                    if source == DType::F16 {
                        half::f16::from_bits(bits).to_f32()
                    } else {
                        half::bf16::from_bits(bits).to_f32()
                    }
                })
                .collect::<Vec<_>>();
            for destination in [
                DType::F32,
                if source == DType::F16 {
                    DType::BF16
                } else {
                    DType::F16
                },
            ] {
                let output = cast(dev, &input, destination).unwrap();
                dev.synchronize().unwrap();
                let expected: Vec<u8> = values
                    .iter()
                    .flat_map(|&value| match destination {
                        DType::F32 => value.to_bits().to_ne_bytes().to_vec(),
                        DType::BF16 => half::bf16::from_f32(value).to_bits().to_ne_bytes().to_vec(),
                        _ => half::f16::from_f32(value).to_bits().to_ne_bytes().to_vec(),
                    })
                    .collect();
                assert_eq!(bytes(&output), expected, "{source:?} -> {destination:?}");
            }
        }
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

    #[test]
    fn parallel_argmax_reduces_large_rows_and_keeps_first_tie() {
        let dev = MetalDevice::get();
        let width = 2048usize;
        let mut values = vec![-1.0f32; 3 * width];
        values[17] = 4.0;
        values[width + 1023] = 7.0;
        values[2 * width + 511] = 9.0;
        values[2 * width + 1535] = 9.0;
        let input = MetalTensor::from_f32(dev, values, vec![3, width]);
        let output = argreduce(dev, &input, 1, true).unwrap();
        dev.synchronize().unwrap();

        // SAFETY: synchronization completed and the output contains three u32 indexes.
        let indexes =
            unsafe { std::slice::from_raw_parts(output.buffer.contents_ptr().cast::<u32>(), 3) };
        assert_eq!(indexes, &[17, 1023, 511]);
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
        // SAFETY: synchronized above; the u32 buffer holds 2 elements.
        let words = unsafe { std::slice::from_raw_parts(raw.contents_ptr().cast::<u32>(), 2) };
        assert_eq!(words, &[744_841_714, 744_841_714]);
        let a_raw = &a.buffer;
        // SAFETY: synchronized above; the i64 buffer holds 6 elements.
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

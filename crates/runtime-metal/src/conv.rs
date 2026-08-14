//! Convolution kernels: conv1d/conv2d, transposed conv1d/conv2d, and
//! weight-gradient (backward-w) conv1d/conv2d.
//!
//! # Design
//!
//! Direct convolutions — no im2col, no workspace: one thread per output
//! element loops over the receptive field, accumulating in f32 regardless
//! of storage dtype. Input and weight/gradient strides are baked into the
//! emitted MSL as constant arithmetic, so arbitrary layouts (including
//! strided views and nonzero offsets) are consumed directly; destinations
//! must be contiguous. Grouped convolutions decompose channel indices at
//! emission time.
//!
//! # Restrictions
//!
//! - dtypes: f16, bf16, and f32 only, and both operands must match.
//! - 32-bit flat output indexing (one `uint gid` per output element).
//! - Ranks are fixed: conv1d is [N, C, L] · [O, C/groups, K], conv2d is
//!   [N, C, H, W] · [O, C/groups, KH, KW]; transposed convs take the
//!   [C_in, C_out/groups, K...] weight convention.
//!
//! # Requirements contract
//!
//! `*_requirements` compute the exact output shape/bytes and pipeline
//! count (convolutions need no scratch or staging); `compile_*_layouts` /
//! `warm_*_layouts` precompile; the `*_into` entry points validate the
//! destination, require the warm pipeline, and allocate nothing.

use super::device::{set_buffer, MetalDevice, Pipeline};
use super::run::MetalTensor;
use crate::runtime::dtype::DType;
use crate::runtime::layout::Layout;
use objc2_metal::MTLComputeCommandEncoder;

const HEADER: &str = "#include <metal_stdlib>\nusing namespace metal;\n";

/// An exact tensor (contiguous shape + dtype) a convolution requires as
/// its destination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRequirement {
    pub shape: Vec<usize>,
    pub dtype: DType,
}

/// Alias kept for the executable planner's vocabulary.
pub type BufferRequirement = TensorRequirement;

impl TensorRequirement {
    /// Byte size of the described tensor, checked for overflow.
    pub fn bytes(&self) -> Result<usize, String> {
        checked_product(&self.shape)?
            .checked_mul(self.dtype.size_in_bytes())
            .ok_or_else(|| {
                format!(
                    "convolution tensor byte size overflow for {:?}:{:?}",
                    self.shape, self.dtype
                )
            })
    }
}

/// Convolution kernels consume arbitrary input layouts directly and need no
/// workspace or host staging. Only the destination has to be provided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvRequirements {
    /// The contiguous destination tensor.
    pub output: TensorRequirement,
    /// `prod(output.shape)`.
    pub output_elements: usize,
    /// `output_elements * dtype size`.
    pub output_bytes: usize,
    /// Always zero: direct convolutions need no workspace.
    pub scratch_bytes: usize,
    /// Always zero: no host staging.
    pub staging_bytes: usize,
    /// Always zero: no per-invocation status buffer.
    pub status_bytes: usize,
    /// 1 for a non-empty output, else 0.
    pub pipeline_count: usize,
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

fn layout_key(layout: &Layout) -> u64 {
    let mut parts = Vec::with_capacity(layout.rank() * 2);
    parts.extend(layout.shape().iter().map(|&value| value as u64));
    parts.extend(layout.strides().iter().map(|&value| value as u64));
    key(&parts)
}

fn cached_pipeline(dev: &MetalDevice, pipeline_key: u64, name: &str) -> Result<Pipeline, String> {
    dev.pipeline_cached(pipeline_key).ok_or_else(|| {
        format!("{name}: exact pipeline is not warm; call the matching warm_*_layouts function")
    })
}

fn conv1d_key(
    x: &Layout,
    w: &Layout,
    dtype: DType,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> u64 {
    key(&[
        0xC0A1,
        dtype as u64,
        layout_key(x),
        layout_key(w),
        stride as u64,
        padding as u64,
        dilation as u64,
        groups as u64,
    ])
}

fn conv2d_key(
    x: &Layout,
    w: &Layout,
    dtype: DType,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> u64 {
    key(&[
        0xC0A2,
        dtype as u64,
        layout_key(x),
        layout_key(w),
        stride as u64,
        padding as u64,
        dilation as u64,
        groups as u64,
    ])
}

#[allow(clippy::too_many_arguments)]
fn conv_transpose1d_key(
    x: &Layout,
    w: &Layout,
    dtype: DType,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> u64 {
    key(&[
        0xC0B1,
        dtype as u64,
        layout_key(x),
        layout_key(w),
        stride as u64,
        padding as u64,
        output_padding as u64,
        dilation as u64,
        groups as u64,
    ])
}

#[allow(clippy::too_many_arguments)]
fn conv_transpose2d_key(
    x: &Layout,
    w: &Layout,
    dtype: DType,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> u64 {
    key(&[
        0xC0B2,
        dtype as u64,
        layout_key(x),
        layout_key(w),
        stride as u64,
        padding as u64,
        output_padding as u64,
        dilation as u64,
        groups as u64,
    ])
}

#[allow(clippy::too_many_arguments)]
fn conv1d_backward_w_key(
    x: &Layout,
    gradient: &Layout,
    dtype: DType,
    kernel: usize,
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> u64 {
    key(&[
        0xC0C1,
        dtype as u64,
        layout_key(x),
        layout_key(gradient),
        kernel as u64,
        out_channels as u64,
        stride as u64,
        padding as u64,
        dilation as u64,
        groups as u64,
    ])
}

#[allow(clippy::too_many_arguments)]
fn conv2d_backward_w_key(
    x: &Layout,
    gradient: &Layout,
    dtype: DType,
    kernel: [usize; 2],
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> u64 {
    key(&[
        0xC0C2,
        dtype as u64,
        layout_key(x),
        layout_key(gradient),
        kernel[0] as u64,
        kernel[1] as u64,
        out_channels as u64,
        stride as u64,
        padding as u64,
        dilation as u64,
        groups as u64,
    ])
}

fn checked_product(values: &[usize]) -> Result<usize, String> {
    values.iter().try_fold(1usize, |total, value| {
        total
            .checked_mul(*value)
            .ok_or_else(|| "convolution element count overflow".to_string())
    })
}

fn msl_type(dtype: DType) -> Result<&'static str, String> {
    match dtype {
        DType::F32 => Ok("float"),
        DType::F16 => Ok("half"),
        DType::BF16 => Ok("bfloat"),
        _ => Err(format!(
            "metal convolution supports f16, bf16, and f32, got {dtype:?}"
        )),
    }
}

fn same_dtype(a: DType, b: DType) -> Result<DType, String> {
    msl_type(a)?;
    if a != b {
        return Err(format!(
            "metal convolution dtype mismatch: input {a:?}, second operand {b:?}"
        ));
    }
    Ok(a)
}

fn require_rank(layout: &Layout, rank: usize, name: &str) -> Result<(), String> {
    if layout.rank() != rank {
        return Err(format!(
            "{name}: expected rank {rank}, got {:?}",
            layout.shape()
        ));
    }
    Ok(())
}

fn conv_output_dim(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    name: &str,
) -> Result<usize, String> {
    if stride == 0 || dilation == 0 || kernel == 0 {
        return Err(format!(
            "{name}: stride, dilation, and kernel size must be non-zero"
        ));
    }
    let padded = input
        .checked_add(
            padding
                .checked_mul(2)
                .ok_or_else(|| format!("{name}: padding overflow"))?,
        )
        .ok_or_else(|| format!("{name}: padded input size overflow"))?;
    let receptive = dilation
        .checked_mul(kernel - 1)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| format!("{name}: receptive field overflow"))?;
    if padded < receptive {
        return Err(format!(
            "{name}: kernel receptive field {receptive} exceeds padded input {padded}"
        ));
    }
    Ok((padded - receptive) / stride + 1)
}

fn conv_transpose_output_dim(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    name: &str,
) -> Result<usize, String> {
    if input == 0 || stride == 0 || dilation == 0 || kernel == 0 {
        return Err(format!(
            "{name}: input, stride, dilation, and kernel size must be non-zero"
        ));
    }
    let expanded = (input - 1)
        .checked_mul(stride)
        .and_then(|value| value.checked_add(dilation.checked_mul(kernel - 1)?))
        .and_then(|value| value.checked_add(output_padding))
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| format!("{name}: output size overflow"))?;
    let trim = padding
        .checked_mul(2)
        .ok_or_else(|| format!("{name}: padding overflow"))?;
    expanded
        .checked_sub(trim)
        .ok_or_else(|| format!("{name}: padding exceeds expanded output"))
}

fn forward_channels(x: &Layout, w: &Layout, groups: usize, name: &str) -> Result<(), String> {
    if groups == 0 || x.shape()[1] % groups != 0 || w.shape()[0] % groups != 0 {
        return Err(format!(
            "{name}: input and output channels must be divisible by non-zero groups"
        ));
    }
    if w.shape()[1]
        .checked_mul(groups)
        .ok_or_else(|| format!("{name}: grouped channel count overflow"))?
        != x.shape()[1]
    {
        return Err(format!(
            "{name}: weight channels {:?} do not match input channels {} with {groups} groups",
            w.shape(),
            x.shape()[1]
        ));
    }
    Ok(())
}

fn transpose_channels(x: &Layout, w: &Layout, groups: usize, name: &str) -> Result<(), String> {
    if groups == 0 || x.shape()[1] % groups != 0 || w.shape()[0] != x.shape()[1] {
        return Err(format!(
            "{name}: weight dim 0 must equal input channels and channels must be divisible by non-zero groups"
        ));
    }
    Ok(())
}

fn backward_channels(
    x: &Layout,
    gradient: &Layout,
    out_channels: usize,
    groups: usize,
    name: &str,
) -> Result<(), String> {
    if groups == 0
        || x.shape()[1] % groups != 0
        || out_channels % groups != 0
        || gradient.shape()[0] != x.shape()[0]
        || gradient.shape()[1] != out_channels
    {
        return Err(format!(
            "{name}: incompatible batch/channel dimensions for {out_channels} output channels and {groups} groups"
        ));
    }
    Ok(())
}

fn requirements(shape: Vec<usize>, dtype: DType) -> Result<ConvRequirements, String> {
    let output_elements = checked_product(&shape)?;
    let output_bytes = output_elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| "convolution output byte size overflow".to_string())?;
    Ok(ConvRequirements {
        output: TensorRequirement { shape, dtype },
        output_elements,
        output_bytes,
        scratch_bytes: 0,
        staging_bytes: 0,
        status_bytes: 0,
        pipeline_count: usize::from(output_elements != 0),
    })
}

/// Validates a conv1d problem ([N, C, L] × [O, C/groups, K]) and returns
/// the exact output requirement.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_requirements(
    x_layout: &Layout,
    x_dtype: DType,
    w_layout: &Layout,
    w_dtype: DType,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<ConvRequirements, String> {
    require_rank(x_layout, 3, "conv1d")?;
    require_rank(w_layout, 3, "conv1d")?;
    let dtype = same_dtype(x_dtype, w_dtype)?;
    forward_channels(x_layout, w_layout, groups, "conv1d")?;
    let length = conv_output_dim(
        x_layout.shape()[2],
        w_layout.shape()[2],
        stride,
        padding,
        dilation,
        "conv1d",
    )?;
    requirements(
        vec![x_layout.shape()[0], w_layout.shape()[0], length],
        dtype,
    )
}

/// Validates a conv2d problem ([N, C, H, W] × [O, C/groups, KH, KW]) and
/// returns the exact output requirement.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_requirements(
    x_layout: &Layout,
    x_dtype: DType,
    w_layout: &Layout,
    w_dtype: DType,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<ConvRequirements, String> {
    require_rank(x_layout, 4, "conv2d")?;
    require_rank(w_layout, 4, "conv2d")?;
    let dtype = same_dtype(x_dtype, w_dtype)?;
    forward_channels(x_layout, w_layout, groups, "conv2d")?;
    let height = conv_output_dim(
        x_layout.shape()[2],
        w_layout.shape()[2],
        stride,
        padding,
        dilation,
        "conv2d",
    )?;
    let width = conv_output_dim(
        x_layout.shape()[3],
        w_layout.shape()[3],
        stride,
        padding,
        dilation,
        "conv2d",
    )?;
    requirements(
        vec![x_layout.shape()[0], w_layout.shape()[0], height, width],
        dtype,
    )
}

/// Validates a transposed conv1d problem (weights are [C_in,
/// C_out/groups, K]) and returns the exact output requirement.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d_requirements(
    x_layout: &Layout,
    x_dtype: DType,
    w_layout: &Layout,
    w_dtype: DType,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<ConvRequirements, String> {
    require_rank(x_layout, 3, "conv_transpose1d")?;
    require_rank(w_layout, 3, "conv_transpose1d")?;
    let dtype = same_dtype(x_dtype, w_dtype)?;
    transpose_channels(x_layout, w_layout, groups, "conv_transpose1d")?;
    let length = conv_transpose_output_dim(
        x_layout.shape()[2],
        w_layout.shape()[2],
        stride,
        padding,
        output_padding,
        dilation,
        "conv_transpose1d",
    )?;
    let out_channels = w_layout.shape()[1]
        .checked_mul(groups)
        .ok_or_else(|| "conv_transpose1d: output channel count overflow".to_string())?;
    requirements(vec![x_layout.shape()[0], out_channels, length], dtype)
}

/// Validates a transposed conv2d problem and returns the exact output
/// requirement.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d_requirements(
    x_layout: &Layout,
    x_dtype: DType,
    w_layout: &Layout,
    w_dtype: DType,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<ConvRequirements, String> {
    require_rank(x_layout, 4, "conv_transpose2d")?;
    require_rank(w_layout, 4, "conv_transpose2d")?;
    let dtype = same_dtype(x_dtype, w_dtype)?;
    transpose_channels(x_layout, w_layout, groups, "conv_transpose2d")?;
    let height = conv_transpose_output_dim(
        x_layout.shape()[2],
        w_layout.shape()[2],
        stride,
        padding,
        output_padding,
        dilation,
        "conv_transpose2d",
    )?;
    let width = conv_transpose_output_dim(
        x_layout.shape()[3],
        w_layout.shape()[3],
        stride,
        padding,
        output_padding,
        dilation,
        "conv_transpose2d",
    )?;
    let out_channels = w_layout.shape()[1]
        .checked_mul(groups)
        .ok_or_else(|| "conv_transpose2d: output channel count overflow".to_string())?;
    requirements(
        vec![x_layout.shape()[0], out_channels, height, width],
        dtype,
    )
}

/// Validates a conv1d weight-gradient problem and returns the exact
/// [O, C/groups, K] output requirement.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_backward_w_requirements(
    x_layout: &Layout,
    x_dtype: DType,
    gradient_layout: &Layout,
    gradient_dtype: DType,
    kernel: usize,
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<ConvRequirements, String> {
    require_rank(x_layout, 3, "conv1d_backward_w")?;
    require_rank(gradient_layout, 3, "conv1d_backward_w")?;
    let dtype = same_dtype(x_dtype, gradient_dtype)?;
    backward_channels(
        x_layout,
        gradient_layout,
        out_channels,
        groups,
        "conv1d_backward_w",
    )?;
    let expected = conv_output_dim(
        x_layout.shape()[2],
        kernel,
        stride,
        padding,
        dilation,
        "conv1d_backward_w",
    )?;
    if gradient_layout.shape()[2] != expected {
        return Err(format!(
            "conv1d_backward_w: gradient length {}, expected {expected}",
            gradient_layout.shape()[2]
        ));
    }
    requirements(
        vec![out_channels, x_layout.shape()[1] / groups, kernel],
        dtype,
    )
}

/// Validates a conv2d weight-gradient problem and returns the exact
/// [O, C/groups, KH, KW] output requirement.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_backward_w_requirements(
    x_layout: &Layout,
    x_dtype: DType,
    gradient_layout: &Layout,
    gradient_dtype: DType,
    kernel: [usize; 2],
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<ConvRequirements, String> {
    require_rank(x_layout, 4, "conv2d_backward_w")?;
    require_rank(gradient_layout, 4, "conv2d_backward_w")?;
    let dtype = same_dtype(x_dtype, gradient_dtype)?;
    backward_channels(
        x_layout,
        gradient_layout,
        out_channels,
        groups,
        "conv2d_backward_w",
    )?;
    let expected_height = conv_output_dim(
        x_layout.shape()[2],
        kernel[0],
        stride,
        padding,
        dilation,
        "conv2d_backward_w",
    )?;
    let expected_width = conv_output_dim(
        x_layout.shape()[3],
        kernel[1],
        stride,
        padding,
        dilation,
        "conv2d_backward_w",
    )?;
    if gradient_layout.shape()[2..] != [expected_height, expected_width] {
        return Err(format!(
            "conv2d_backward_w: gradient spatial shape {:?}, expected [{expected_height}, {expected_width}]",
            &gradient_layout.shape()[2..]
        ));
    }
    requirements(
        vec![
            out_channels,
            x_layout.shape()[1] / groups,
            kernel[0],
            kernel[1],
        ],
        dtype,
    )
}

fn validate_destination(
    destination: &MetalTensor,
    requirement: &TensorRequirement,
    name: &str,
) -> Result<(), String> {
    if destination.layout.shape() != requirement.shape || destination.dtype != requirement.dtype {
        return Err(format!(
            "{name}: destination mismatch, expected {:?}:{:?}, got {:?}:{:?}",
            requirement.shape,
            requirement.dtype,
            destination.layout.shape(),
            destination.dtype
        ));
    }
    if !destination.layout.is_contiguous() {
        return Err(format!("{name}: destination must be contiguous"));
    }
    let required = requirement.bytes()?;
    let offset = destination
        .layout
        .offset()
        .checked_mul(destination.dtype.size_in_bytes())
        .ok_or_else(|| format!("{name}: destination offset overflow"))?;
    if offset
        .checked_add(required)
        .is_none_or(|end| end > destination.buffer.size)
    {
        return Err(format!(
            "{name}: destination buffer has {} bytes, needs {required} bytes at offset {offset}",
            destination.buffer.size
        ));
    }
    MetalDevice::get().mark_buffer_write(&destination.buffer)?;
    Ok(())
}

fn grid_for(
    dev: &MetalDevice,
    pipeline: &Pipeline,
    n: usize,
    binds: &[(usize, &super::device::Buffer, usize)],
) {
    let padded = n.div_ceil(256) * 256;
    dev.with_encoder(|encoder| {
        encoder.setComputePipelineState(pipeline.as_raw());
        for &(index, buffer, offset) in binds {
            set_buffer(encoder, index, buffer, offset);
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MetalDevice::grid(padded, 1, 1),
            MetalDevice::grid(256, 1, 1),
        );
    });
}

fn conv1d_pipeline(
    dev: &MetalDevice,
    x: &Layout,
    w: &Layout,
    dtype: DType,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Pipeline, String> {
    let ty = msl_type(dtype)?;
    let (n, length) = (x.shape()[0], x.shape()[2]);
    let (c_out, c_per, kernel) = (w.shape()[0], w.shape()[1], w.shape()[2]);
    let out_length = conv_output_dim(length, kernel, stride, padding, dilation, "conv1d")?;
    let total = n * c_out * out_length;
    let (xs, ws) = (x.strides(), w.strides());
    let cout_per = c_out / groups;
    let source = || {
        format!(
            r#"{HEADER}
kernel void et_conv1d(device const {ty}* x [[buffer(0)]], device const {ty}* w [[buffer(1)]], device {ty}* out [[buffer(2)]], uint gid [[thread_position_in_grid]]) {{
    if (gid >= {total}u) return;
    const uint i = gid % {out_length}u;
    const uint oc = (gid / {out_length}u) % {c_out}u;
    const uint b = gid / ({out_length}u * {c_out}u);
    const uint group = oc / {cout_per}u;
    float acc = 0.0f;
    for (uint ci = 0u; ci < {c_per}u; ++ci) {{
        const uint ic = group * {c_per}u + ci;
        for (uint k = 0u; k < {kernel}u; ++k) {{
            const long pos = (long)(i * {stride}u + k * {dilation}u) - {padding}l;
            if (pos >= 0 && pos < {length}l) acc += float(x[b * {x0}ul + ic * {x1}ul + (uint)pos * {x2}ul]) * float(w[oc * {w0}ul + ci * {w1}ul + k * {w2}ul]);
        }}
    }}
    out[gid] = ({ty})acc;
}}
"#,
            x0 = xs[0],
            x1 = xs[1],
            x2 = xs[2],
            w0 = ws[0],
            w1 = ws[1],
            w2 = ws[2]
        )
    };
    dev.compile_lazy(
        conv1d_key(x, w, dtype, stride, padding, dilation, groups),
        "et_conv1d",
        source,
    )
}

fn conv2d_pipeline(
    dev: &MetalDevice,
    x: &Layout,
    w: &Layout,
    dtype: DType,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Pipeline, String> {
    let ty = msl_type(dtype)?;
    let (n, height, width) = (x.shape()[0], x.shape()[2], x.shape()[3]);
    let (c_out, c_per, kh, kw) = (w.shape()[0], w.shape()[1], w.shape()[2], w.shape()[3]);
    let oh = conv_output_dim(height, kh, stride, padding, dilation, "conv2d")?;
    let ow = conv_output_dim(width, kw, stride, padding, dilation, "conv2d")?;
    let total = n * c_out * oh * ow;
    let (xs, ws) = (x.strides(), w.strides());
    let cout_per = c_out / groups;
    let source = || {
        format!(
            r#"{HEADER}
kernel void et_conv2d(device const {ty}* x [[buffer(0)]], device const {ty}* w [[buffer(1)]], device {ty}* out [[buffer(2)]], uint gid [[thread_position_in_grid]]) {{
    if (gid >= {total}u) return;
    const uint j = gid % {ow}u; const uint i = (gid / {ow}u) % {oh}u;
    const uint oc = (gid / ({ow}u * {oh}u)) % {c_out}u; const uint b = gid / ({ow}u * {oh}u * {c_out}u); const uint group = oc / {cout_per}u;
    float acc = 0.0f;
    for (uint ci = 0u; ci < {c_per}u; ++ci) {{ const uint ic = group * {c_per}u + ci;
        for (uint ky = 0u; ky < {kh}u; ++ky) {{ const long py = (long)(i * {stride}u + ky * {dilation}u) - {padding}l; if (py < 0 || py >= {height}l) continue;
            for (uint kx = 0u; kx < {kw}u; ++kx) {{ const long px = (long)(j * {stride}u + kx * {dilation}u) - {padding}l; if (px < 0 || px >= {width}l) continue;
                acc += float(x[b * {x0}ul + ic * {x1}ul + (uint)py * {x2}ul + (uint)px * {x3}ul]) * float(w[oc * {w0}ul + ci * {w1}ul + ky * {w2}ul + kx * {w3}ul]); }} }} }}
    out[gid] = ({ty})acc;
}}
"#,
            x0 = xs[0],
            x1 = xs[1],
            x2 = xs[2],
            x3 = xs[3],
            w0 = ws[0],
            w1 = ws[1],
            w2 = ws[2],
            w3 = ws[3]
        )
    };
    dev.compile_lazy(
        conv2d_key(x, w, dtype, stride, padding, dilation, groups),
        "et_conv2d",
        source,
    )
}

#[allow(clippy::too_many_arguments)]
fn conv_transpose1d_pipeline(
    dev: &MetalDevice,
    x: &Layout,
    w: &Layout,
    dtype: DType,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Pipeline, String> {
    let ty = msl_type(dtype)?;
    let (n, c_in, length) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let (c_out_per_group, kernel) = (w.shape()[1], w.shape()[2]);
    let c_out = c_out_per_group * groups;
    let out_length = conv_transpose_output_dim(
        length,
        kernel,
        stride,
        padding,
        output_padding,
        dilation,
        "conv_transpose1d",
    )?;
    let total = n * c_out * out_length;
    let cin_per = c_in / groups;
    let (xs, ws) = (x.strides(), w.strides());
    let source = || {
        format!(
            r#"{HEADER}
kernel void et_convt1d(device const {ty}* x [[buffer(0)]], device const {ty}* w [[buffer(1)]], device {ty}* out [[buffer(2)]], uint gid [[thread_position_in_grid]]) {{
    if (gid >= {total}u) return; const uint o = gid % {out_length}u; const uint oc = (gid / {out_length}u) % {c_out}u; const uint b = gid / ({out_length}u * {c_out}u); const uint group = oc / {c_out_per_group}u; float acc = 0.0f;
    for (uint ci = 0u; ci < {cin_per}u; ++ci) {{ const uint ic = group * {cin_per}u + ci; for (uint k = 0u; k < {kernel}u; ++k) {{ const long num = (long)(o + {padding}u) - (long)(k * {dilation}u); if (num < 0 || (uint)num % {stride}u != 0u) continue; const uint i = (uint)num / {stride}u; if (i < {length}u) acc += float(x[b * {x0}ul + ic * {x1}ul + i * {x2}ul]) * float(w[ic * {w0}ul + (oc % {c_out_per_group}u) * {w1}ul + k * {w2}ul]); }} }} out[gid] = ({ty})acc;
}}
"#,
            x0 = xs[0],
            x1 = xs[1],
            x2 = xs[2],
            w0 = ws[0],
            w1 = ws[1],
            w2 = ws[2]
        )
    };
    dev.compile_lazy(
        conv_transpose1d_key(
            x,
            w,
            dtype,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        ),
        "et_convt1d",
        source,
    )
}

#[allow(clippy::too_many_arguments)]
fn conv_transpose2d_pipeline(
    dev: &MetalDevice,
    x: &Layout,
    w: &Layout,
    dtype: DType,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Pipeline, String> {
    let ty = msl_type(dtype)?;
    let (n, c_in, height, width) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
    let (c_out_per_group, kh, kw) = (w.shape()[1], w.shape()[2], w.shape()[3]);
    let c_out = c_out_per_group * groups;
    let oh = conv_transpose_output_dim(
        height,
        kh,
        stride,
        padding,
        output_padding,
        dilation,
        "conv_transpose2d",
    )?;
    let ow = conv_transpose_output_dim(
        width,
        kw,
        stride,
        padding,
        output_padding,
        dilation,
        "conv_transpose2d",
    )?;
    let total = n * c_out * oh * ow;
    let cin_per = c_in / groups;
    let (xs, ws) = (x.strides(), w.strides());
    let source = || {
        format!(
            r#"{HEADER}
kernel void et_convt2d(device const {ty}* x [[buffer(0)]], device const {ty}* w [[buffer(1)]], device {ty}* out [[buffer(2)]], uint gid [[thread_position_in_grid]]) {{
    if (gid >= {total}u) return; const uint oj = gid % {ow}u; const uint oi = (gid / {ow}u) % {oh}u; const uint oc = (gid / ({ow}u * {oh}u)) % {c_out}u; const uint b = gid / ({ow}u * {oh}u * {c_out}u); const uint group = oc / {c_out_per_group}u; float acc = 0.0f;
    for (uint ci = 0u; ci < {cin_per}u; ++ci) {{ const uint ic = group * {cin_per}u + ci; for (uint ky = 0u; ky < {kh}u; ++ky) {{ const long numy = (long)(oi + {padding}u) - (long)(ky * {dilation}u); if (numy < 0 || (uint)numy % {stride}u != 0u) continue; const uint i = (uint)numy / {stride}u; if (i >= {height}u) continue; for (uint kx = 0u; kx < {kw}u; ++kx) {{ const long numx = (long)(oj + {padding}u) - (long)(kx * {dilation}u); if (numx < 0 || (uint)numx % {stride}u != 0u) continue; const uint j = (uint)numx / {stride}u; if (j < {width}u) acc += float(x[b * {x0}ul + ic * {x1}ul + i * {x2}ul + j * {x3}ul]) * float(w[ic * {w0}ul + (oc % {c_out_per_group}u) * {w1}ul + ky * {w2}ul + kx * {w3}ul]); }} }} }} out[gid] = ({ty})acc;
}}
"#,
            x0 = xs[0],
            x1 = xs[1],
            x2 = xs[2],
            x3 = xs[3],
            w0 = ws[0],
            w1 = ws[1],
            w2 = ws[2],
            w3 = ws[3]
        )
    };
    dev.compile_lazy(
        conv_transpose2d_key(
            x,
            w,
            dtype,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        ),
        "et_convt2d",
        source,
    )
}

#[allow(clippy::too_many_arguments)]
fn conv1d_backward_w_pipeline(
    dev: &MetalDevice,
    x: &Layout,
    gradient: &Layout,
    dtype: DType,
    kernel: usize,
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Pipeline, String> {
    let ty = msl_type(dtype)?;
    let (n, c_in, length) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let out_length = gradient.shape()[2];
    let c_per = c_in / groups;
    let cout_per = out_channels / groups;
    let total = out_channels * c_per * kernel;
    let (xs, gs) = (x.strides(), gradient.strides());
    let source = || {
        format!(
            r#"{HEADER}
kernel void et_conv1d_bw(device const {ty}* x [[buffer(0)]], device const {ty}* g [[buffer(1)]], device {ty}* out [[buffer(2)]], uint gid [[thread_position_in_grid]]) {{
    if (gid >= {total}u) return; const uint k = gid % {kernel}u; const uint ci = (gid / {kernel}u) % {c_per}u; const uint oc = gid / ({kernel}u * {c_per}u); const uint group = oc / {cout_per}u; const uint ic = group * {c_per}u + ci; float acc = 0.0f;
    for (uint b = 0u; b < {n}u; ++b) for (uint i = 0u; i < {out_length}u; ++i) {{ const long pos = (long)(i * {stride}u + k * {dilation}u) - {padding}l; if (pos >= 0 && pos < {length}l) acc += float(g[b * {g0}ul + oc * {g1}ul + i * {g2}ul]) * float(x[b * {x0}ul + ic * {x1}ul + (uint)pos * {x2}ul]); }} out[gid] = ({ty})acc;
}}
"#,
            x0 = xs[0],
            x1 = xs[1],
            x2 = xs[2],
            g0 = gs[0],
            g1 = gs[1],
            g2 = gs[2]
        )
    };
    dev.compile_lazy(
        conv1d_backward_w_key(
            x,
            gradient,
            dtype,
            kernel,
            out_channels,
            stride,
            padding,
            dilation,
            groups,
        ),
        "et_conv1d_bw",
        source,
    )
}

#[allow(clippy::too_many_arguments)]
fn conv2d_backward_w_pipeline(
    dev: &MetalDevice,
    x: &Layout,
    gradient: &Layout,
    dtype: DType,
    kernel: [usize; 2],
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Pipeline, String> {
    let ty = msl_type(dtype)?;
    let (n, c_in, height, width) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
    let (oh, ow) = (gradient.shape()[2], gradient.shape()[3]);
    let (kh, kw) = (kernel[0], kernel[1]);
    let c_per = c_in / groups;
    let cout_per = out_channels / groups;
    let total = out_channels * c_per * kh * kw;
    let (xs, gs) = (x.strides(), gradient.strides());
    let source = || {
        format!(
            r#"{HEADER}
kernel void et_conv2d_bw(device const {ty}* x [[buffer(0)]], device const {ty}* g [[buffer(1)]], device {ty}* out [[buffer(2)]], uint gid [[thread_position_in_grid]]) {{
    if (gid >= {total}u) return; const uint kx = gid % {kw}u; const uint ky = (gid / {kw}u) % {kh}u; const uint ci = (gid / ({kw}u * {kh}u)) % {c_per}u; const uint oc = gid / ({kw}u * {kh}u * {c_per}u); const uint group = oc / {cout_per}u; const uint ic = group * {c_per}u + ci; float acc = 0.0f;
    for (uint b = 0u; b < {n}u; ++b) for (uint i = 0u; i < {oh}u; ++i) {{ const long py = (long)(i * {stride}u + ky * {dilation}u) - {padding}l; if (py < 0 || py >= {height}l) continue; for (uint j = 0u; j < {ow}u; ++j) {{ const long px = (long)(j * {stride}u + kx * {dilation}u) - {padding}l; if (px >= 0 && px < {width}l) acc += float(g[b * {g0}ul + oc * {g1}ul + i * {g2}ul + j * {g3}ul]) * float(x[b * {x0}ul + ic * {x1}ul + (uint)py * {x2}ul + (uint)px * {x3}ul]); }} }} out[gid] = ({ty})acc;
}}
"#,
            x0 = xs[0],
            x1 = xs[1],
            x2 = xs[2],
            x3 = xs[3],
            g0 = gs[0],
            g1 = gs[1],
            g2 = gs[2],
            g3 = gs[3]
        )
    };
    dev.compile_lazy(
        conv2d_backward_w_key(
            x,
            gradient,
            dtype,
            kernel,
            out_channels,
            stride,
            padding,
            dilation,
            groups,
        ),
        "et_conv2d_bw",
        source,
    )
}

fn dispatch(
    dev: &MetalDevice,
    pipeline: &Pipeline,
    total: usize,
    first: &MetalTensor,
    second: &MetalTensor,
    destination: &MetalTensor,
) {
    if total == 0 {
        return;
    }
    grid_for(
        dev,
        pipeline,
        total,
        &[
            (
                0,
                &first.buffer,
                first.layout.offset() * first.dtype.size_in_bytes(),
            ),
            (
                1,
                &second.buffer,
                second.layout.offset() * second.dtype.size_in_bytes(),
            ),
            (
                2,
                &destination.buffer,
                destination.layout.offset() * destination.dtype.size_in_bytes(),
            ),
        ],
    );
}

/// Dispatches conv1d into `destination`; requires the precompiled pipeline
/// for the exact layouts and hyperparameters.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_into(
    dev: &MetalDevice,
    x: &MetalTensor,
    w: &MetalTensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    destination: &MetalTensor,
) -> Result<(), String> {
    let requirement = conv1d_requirements(
        &x.layout, x.dtype, &w.layout, w.dtype, stride, padding, dilation, groups,
    )?;
    validate_destination(destination, &requirement.output, "conv1d")?;
    let total = checked_product(&requirement.output.shape)?;
    if total != 0 {
        let pipeline = cached_pipeline(
            dev,
            conv1d_key(
                &x.layout, &w.layout, x.dtype, stride, padding, dilation, groups,
            ),
            "conv1d",
        )?;
        dispatch(dev, &pipeline, total, x, w, destination);
    }
    Ok(())
}

/// Destination form of [`conv2d`].
#[allow(clippy::too_many_arguments)]
pub fn conv2d_into(
    dev: &MetalDevice,
    x: &MetalTensor,
    w: &MetalTensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    destination: &MetalTensor,
) -> Result<(), String> {
    let requirement = conv2d_requirements(
        &x.layout, x.dtype, &w.layout, w.dtype, stride, padding, dilation, groups,
    )?;
    validate_destination(destination, &requirement.output, "conv2d")?;
    let total = checked_product(&requirement.output.shape)?;
    if total != 0 {
        let pipeline = cached_pipeline(
            dev,
            conv2d_key(
                &x.layout, &w.layout, x.dtype, stride, padding, dilation, groups,
            ),
            "conv2d",
        )?;
        dispatch(dev, &pipeline, total, x, w, destination);
    }
    Ok(())
}

/// Destination form of [`conv_transpose1d`].
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d_into(
    dev: &MetalDevice,
    x: &MetalTensor,
    w: &MetalTensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
    destination: &MetalTensor,
) -> Result<(), String> {
    let requirement = conv_transpose1d_requirements(
        &x.layout,
        x.dtype,
        &w.layout,
        w.dtype,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    )?;
    validate_destination(destination, &requirement.output, "conv_transpose1d")?;
    let total = checked_product(&requirement.output.shape)?;
    if total != 0 {
        let pipeline = cached_pipeline(
            dev,
            conv_transpose1d_key(
                &x.layout,
                &w.layout,
                x.dtype,
                stride,
                padding,
                output_padding,
                dilation,
                groups,
            ),
            "conv_transpose1d",
        )?;
        dispatch(dev, &pipeline, total, x, w, destination);
    }
    Ok(())
}

/// Destination form of [`conv_transpose2d`].
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d_into(
    dev: &MetalDevice,
    x: &MetalTensor,
    w: &MetalTensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
    destination: &MetalTensor,
) -> Result<(), String> {
    let requirement = conv_transpose2d_requirements(
        &x.layout,
        x.dtype,
        &w.layout,
        w.dtype,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    )?;
    validate_destination(destination, &requirement.output, "conv_transpose2d")?;
    let total = checked_product(&requirement.output.shape)?;
    if total != 0 {
        let pipeline = cached_pipeline(
            dev,
            conv_transpose2d_key(
                &x.layout,
                &w.layout,
                x.dtype,
                stride,
                padding,
                output_padding,
                dilation,
                groups,
            ),
            "conv_transpose2d",
        )?;
        dispatch(dev, &pipeline, total, x, w, destination);
    }
    Ok(())
}

/// Destination form of [`conv1d_backward_w`].
#[allow(clippy::too_many_arguments)]
pub fn conv1d_backward_w_into(
    dev: &MetalDevice,
    x: &MetalTensor,
    gradient: &MetalTensor,
    kernel: usize,
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    destination: &MetalTensor,
) -> Result<(), String> {
    let requirement = conv1d_backward_w_requirements(
        &x.layout,
        x.dtype,
        &gradient.layout,
        gradient.dtype,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )?;
    validate_destination(destination, &requirement.output, "conv1d_backward_w")?;
    let total = checked_product(&requirement.output.shape)?;
    if total != 0 {
        let pipeline = cached_pipeline(
            dev,
            conv1d_backward_w_key(
                &x.layout,
                &gradient.layout,
                x.dtype,
                kernel,
                out_channels,
                stride,
                padding,
                dilation,
                groups,
            ),
            "conv1d_backward_w",
        )?;
        dispatch(dev, &pipeline, total, x, gradient, destination);
    }
    Ok(())
}

/// Destination form of [`conv2d_backward_w`].
#[allow(clippy::too_many_arguments)]
pub fn conv2d_backward_w_into(
    dev: &MetalDevice,
    x: &MetalTensor,
    gradient: &MetalTensor,
    kernel: [usize; 2],
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    destination: &MetalTensor,
) -> Result<(), String> {
    let requirement = conv2d_backward_w_requirements(
        &x.layout,
        x.dtype,
        &gradient.layout,
        gradient.dtype,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )?;
    validate_destination(destination, &requirement.output, "conv2d_backward_w")?;
    let total = checked_product(&requirement.output.shape)?;
    if total != 0 {
        let pipeline = cached_pipeline(
            dev,
            conv2d_backward_w_key(
                &x.layout,
                &gradient.layout,
                x.dtype,
                kernel,
                out_channels,
                stride,
                padding,
                dilation,
                groups,
            ),
            "conv2d_backward_w",
        )?;
        dispatch(dev, &pipeline, total, x, gradient, destination);
    }
    Ok(())
}

/// Precompiles the conv1d pipeline for exact layouts/hyperparameters.
#[allow(clippy::too_many_arguments)]
pub fn compile_conv1d_layouts(
    dev: &MetalDevice,
    x_layout: &Layout,
    x_dtype: DType,
    w_layout: &Layout,
    w_dtype: DType,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<(), String> {
    let requirement = conv1d_requirements(
        x_layout, x_dtype, w_layout, w_dtype, stride, padding, dilation, groups,
    )?;
    if checked_product(&requirement.output.shape)? != 0 {
        conv1d_pipeline(
            dev, x_layout, w_layout, x_dtype, stride, padding, dilation, groups,
        )?;
    }
    Ok(())
}

/// Precompiles the conv2d pipeline for exact layouts/hyperparameters.
#[allow(clippy::too_many_arguments)]
pub fn compile_conv2d_layouts(
    dev: &MetalDevice,
    x_layout: &Layout,
    x_dtype: DType,
    w_layout: &Layout,
    w_dtype: DType,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<(), String> {
    let requirement = conv2d_requirements(
        x_layout, x_dtype, w_layout, w_dtype, stride, padding, dilation, groups,
    )?;
    if checked_product(&requirement.output.shape)? != 0 {
        conv2d_pipeline(
            dev, x_layout, w_layout, x_dtype, stride, padding, dilation, groups,
        )?;
    }
    Ok(())
}

/// Precompiles the transposed conv1d pipeline for exact layouts.
#[allow(clippy::too_many_arguments)]
pub fn compile_conv_transpose1d_layouts(
    dev: &MetalDevice,
    x_layout: &Layout,
    x_dtype: DType,
    w_layout: &Layout,
    w_dtype: DType,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<(), String> {
    let requirement = conv_transpose1d_requirements(
        x_layout,
        x_dtype,
        w_layout,
        w_dtype,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    )?;
    if checked_product(&requirement.output.shape)? != 0 {
        conv_transpose1d_pipeline(
            dev,
            x_layout,
            w_layout,
            x_dtype,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        )?;
    }
    Ok(())
}

/// Precompiles the transposed conv2d pipeline for exact layouts.
#[allow(clippy::too_many_arguments)]
pub fn compile_conv_transpose2d_layouts(
    dev: &MetalDevice,
    x_layout: &Layout,
    x_dtype: DType,
    w_layout: &Layout,
    w_dtype: DType,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<(), String> {
    let requirement = conv_transpose2d_requirements(
        x_layout,
        x_dtype,
        w_layout,
        w_dtype,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    )?;
    if checked_product(&requirement.output.shape)? != 0 {
        conv_transpose2d_pipeline(
            dev,
            x_layout,
            w_layout,
            x_dtype,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        )?;
    }
    Ok(())
}

/// Precompiles the conv1d weight-gradient pipeline for exact layouts.
#[allow(clippy::too_many_arguments)]
pub fn compile_conv1d_backward_w_layouts(
    dev: &MetalDevice,
    x_layout: &Layout,
    x_dtype: DType,
    gradient_layout: &Layout,
    gradient_dtype: DType,
    kernel: usize,
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<(), String> {
    let requirement = conv1d_backward_w_requirements(
        x_layout,
        x_dtype,
        gradient_layout,
        gradient_dtype,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )?;
    if checked_product(&requirement.output.shape)? != 0 {
        conv1d_backward_w_pipeline(
            dev,
            x_layout,
            gradient_layout,
            x_dtype,
            kernel,
            out_channels,
            stride,
            padding,
            dilation,
            groups,
        )?;
    }
    Ok(())
}

/// Precompiles the conv2d weight-gradient pipeline for exact layouts.
#[allow(clippy::too_many_arguments)]
pub fn compile_conv2d_backward_w_layouts(
    dev: &MetalDevice,
    x_layout: &Layout,
    x_dtype: DType,
    gradient_layout: &Layout,
    gradient_dtype: DType,
    kernel: [usize; 2],
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<(), String> {
    let requirement = conv2d_backward_w_requirements(
        x_layout,
        x_dtype,
        gradient_layout,
        gradient_dtype,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )?;
    if checked_product(&requirement.output.shape)? != 0 {
        conv2d_backward_w_pipeline(
            dev,
            x_layout,
            gradient_layout,
            x_dtype,
            kernel,
            out_channels,
            stride,
            padding,
            dilation,
            groups,
        )?;
    }
    Ok(())
}

/// [`compile_conv1d_layouts`] against the process-wide device.
#[allow(clippy::too_many_arguments)]
pub fn warm_conv1d_layouts(
    x_layout: &Layout,
    x_dtype: DType,
    w_layout: &Layout,
    w_dtype: DType,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<(), String> {
    compile_conv1d_layouts(
        MetalDevice::get(),
        x_layout,
        x_dtype,
        w_layout,
        w_dtype,
        stride,
        padding,
        dilation,
        groups,
    )
}

/// [`compile_conv2d_layouts`] against the process-wide device.
#[allow(clippy::too_many_arguments)]
pub fn warm_conv2d_layouts(
    x_layout: &Layout,
    x_dtype: DType,
    w_layout: &Layout,
    w_dtype: DType,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<(), String> {
    compile_conv2d_layouts(
        MetalDevice::get(),
        x_layout,
        x_dtype,
        w_layout,
        w_dtype,
        stride,
        padding,
        dilation,
        groups,
    )
}

/// [`compile_conv_transpose1d_layouts`] against the process-wide device.
#[allow(clippy::too_many_arguments)]
pub fn warm_conv_transpose1d_layouts(
    x_layout: &Layout,
    x_dtype: DType,
    w_layout: &Layout,
    w_dtype: DType,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<(), String> {
    compile_conv_transpose1d_layouts(
        MetalDevice::get(),
        x_layout,
        x_dtype,
        w_layout,
        w_dtype,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    )
}

/// [`compile_conv_transpose2d_layouts`] against the process-wide device.
#[allow(clippy::too_many_arguments)]
pub fn warm_conv_transpose2d_layouts(
    x_layout: &Layout,
    x_dtype: DType,
    w_layout: &Layout,
    w_dtype: DType,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<(), String> {
    compile_conv_transpose2d_layouts(
        MetalDevice::get(),
        x_layout,
        x_dtype,
        w_layout,
        w_dtype,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    )
}

/// [`compile_conv1d_backward_w_layouts`] against the process-wide device.
#[allow(clippy::too_many_arguments)]
pub fn warm_conv1d_backward_w_layouts(
    x_layout: &Layout,
    x_dtype: DType,
    gradient_layout: &Layout,
    gradient_dtype: DType,
    kernel: usize,
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<(), String> {
    compile_conv1d_backward_w_layouts(
        MetalDevice::get(),
        x_layout,
        x_dtype,
        gradient_layout,
        gradient_dtype,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )
}

/// [`compile_conv2d_backward_w_layouts`] against the process-wide device.
#[allow(clippy::too_many_arguments)]
pub fn warm_conv2d_backward_w_layouts(
    x_layout: &Layout,
    x_dtype: DType,
    gradient_layout: &Layout,
    gradient_dtype: DType,
    kernel: [usize; 2],
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<(), String> {
    compile_conv2d_backward_w_layouts(
        MetalDevice::get(),
        x_layout,
        x_dtype,
        gradient_layout,
        gradient_dtype,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )
}

/// [`warm_conv1d_layouts`] for contiguous f32 shapes.
pub fn warm_conv1d(
    x_shape: &[usize],
    w_shape: &[usize],
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<(), String> {
    warm_conv1d_layouts(
        &Layout::contiguous(x_shape.to_vec()),
        DType::F32,
        &Layout::contiguous(w_shape.to_vec()),
        DType::F32,
        stride,
        padding,
        dilation,
        groups,
    )
}

/// [`warm_conv2d_layouts`] for contiguous f32 shapes.
pub fn warm_conv2d(
    x_shape: &[usize],
    w_shape: &[usize],
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<(), String> {
    warm_conv2d_layouts(
        &Layout::contiguous(x_shape.to_vec()),
        DType::F32,
        &Layout::contiguous(w_shape.to_vec()),
        DType::F32,
        stride,
        padding,
        dilation,
        groups,
    )
}

/// [`warm_conv_transpose1d_layouts`] for contiguous f32 shapes.
#[allow(clippy::too_many_arguments)]
pub fn warm_conv_transpose1d(
    x_shape: &[usize],
    w_shape: &[usize],
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<(), String> {
    warm_conv_transpose1d_layouts(
        &Layout::contiguous(x_shape.to_vec()),
        DType::F32,
        &Layout::contiguous(w_shape.to_vec()),
        DType::F32,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    )
}

/// [`warm_conv_transpose2d_layouts`] for contiguous f32 shapes.
#[allow(clippy::too_many_arguments)]
pub fn warm_conv_transpose2d(
    x_shape: &[usize],
    w_shape: &[usize],
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<(), String> {
    warm_conv_transpose2d_layouts(
        &Layout::contiguous(x_shape.to_vec()),
        DType::F32,
        &Layout::contiguous(w_shape.to_vec()),
        DType::F32,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    )
}

/// [`warm_conv1d_backward_w_layouts`] for contiguous f32 shapes.
#[allow(clippy::too_many_arguments)]
pub fn warm_conv1d_backward_w(
    x_shape: &[usize],
    gradient_shape: &[usize],
    kernel: usize,
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<(), String> {
    warm_conv1d_backward_w_layouts(
        &Layout::contiguous(x_shape.to_vec()),
        DType::F32,
        &Layout::contiguous(gradient_shape.to_vec()),
        DType::F32,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )
}

/// [`warm_conv2d_backward_w_layouts`] for contiguous f32 shapes.
#[allow(clippy::too_many_arguments)]
pub fn warm_conv2d_backward_w(
    x_shape: &[usize],
    gradient_shape: &[usize],
    kernel: [usize; 2],
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<(), String> {
    warm_conv2d_backward_w_layouts(
        &Layout::contiguous(x_shape.to_vec()),
        DType::F32,
        &Layout::contiguous(gradient_shape.to_vec()),
        DType::F32,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )
}

/// Allocating conv1d: plans, precompiles, allocates the output, and
/// dispatches.
pub fn conv1d(
    dev: &MetalDevice,
    x: &MetalTensor,
    w: &MetalTensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<MetalTensor, String> {
    let requirement = conv1d_requirements(
        &x.layout, x.dtype, &w.layout, w.dtype, stride, padding, dilation, groups,
    )?;
    compile_conv1d_layouts(
        dev, &x.layout, x.dtype, &w.layout, w.dtype, stride, padding, dilation, groups,
    )?;
    let output = MetalTensor::empty(
        dev,
        requirement.output.shape.clone(),
        requirement.output.dtype,
    );
    conv1d_into(dev, x, w, stride, padding, dilation, groups, &output)?;
    Ok(output)
}

/// Allocating conv2d.
pub fn conv2d(
    dev: &MetalDevice,
    x: &MetalTensor,
    w: &MetalTensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<MetalTensor, String> {
    let requirement = conv2d_requirements(
        &x.layout, x.dtype, &w.layout, w.dtype, stride, padding, dilation, groups,
    )?;
    compile_conv2d_layouts(
        dev, &x.layout, x.dtype, &w.layout, w.dtype, stride, padding, dilation, groups,
    )?;
    let output = MetalTensor::empty(
        dev,
        requirement.output.shape.clone(),
        requirement.output.dtype,
    );
    conv2d_into(dev, x, w, stride, padding, dilation, groups, &output)?;
    Ok(output)
}

/// Allocating transposed conv1d.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d(
    dev: &MetalDevice,
    x: &MetalTensor,
    w: &MetalTensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<MetalTensor, String> {
    let requirement = conv_transpose1d_requirements(
        &x.layout,
        x.dtype,
        &w.layout,
        w.dtype,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    )?;
    compile_conv_transpose1d_layouts(
        dev,
        &x.layout,
        x.dtype,
        &w.layout,
        w.dtype,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    )?;
    let output = MetalTensor::empty(
        dev,
        requirement.output.shape.clone(),
        requirement.output.dtype,
    );
    conv_transpose1d_into(
        dev,
        x,
        w,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        &output,
    )?;
    Ok(output)
}

/// Allocating transposed conv2d.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d(
    dev: &MetalDevice,
    x: &MetalTensor,
    w: &MetalTensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<MetalTensor, String> {
    let requirement = conv_transpose2d_requirements(
        &x.layout,
        x.dtype,
        &w.layout,
        w.dtype,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    )?;
    compile_conv_transpose2d_layouts(
        dev,
        &x.layout,
        x.dtype,
        &w.layout,
        w.dtype,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    )?;
    let output = MetalTensor::empty(
        dev,
        requirement.output.shape.clone(),
        requirement.output.dtype,
    );
    conv_transpose2d_into(
        dev,
        x,
        w,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        &output,
    )?;
    Ok(output)
}

/// Allocating conv1d weight gradient.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_backward_w(
    dev: &MetalDevice,
    x: &MetalTensor,
    gradient: &MetalTensor,
    kernel: usize,
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<MetalTensor, String> {
    let requirement = conv1d_backward_w_requirements(
        &x.layout,
        x.dtype,
        &gradient.layout,
        gradient.dtype,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )?;
    compile_conv1d_backward_w_layouts(
        dev,
        &x.layout,
        x.dtype,
        &gradient.layout,
        gradient.dtype,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )?;
    let output = MetalTensor::empty(
        dev,
        requirement.output.shape.clone(),
        requirement.output.dtype,
    );
    conv1d_backward_w_into(
        dev,
        x,
        gradient,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
        &output,
    )?;
    Ok(output)
}

/// Allocating conv2d weight gradient.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_backward_w(
    dev: &MetalDevice,
    x: &MetalTensor,
    gradient: &MetalTensor,
    kernel: [usize; 2],
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<MetalTensor, String> {
    let requirement = conv2d_backward_w_requirements(
        &x.layout,
        x.dtype,
        &gradient.layout,
        gradient.dtype,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )?;
    compile_conv2d_backward_w_layouts(
        dev,
        &x.layout,
        x.dtype,
        &gradient.layout,
        gradient.dtype,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )?;
    let output = MetalTensor::empty(
        dev,
        requirement.output.shape.clone(),
        requirement.output.dtype,
    );
    conv2d_backward_w_into(
        dev,
        x,
        gradient,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
        &output,
    )?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn output(dev: &MetalDevice, requirement: &ConvRequirements) -> MetalTensor {
        MetalTensor::empty(
            dev,
            requirement.output.shape.clone(),
            requirement.output.dtype,
        )
    }

    #[test]
    fn destination_apis_match_allocating_wrappers() {
        let dev = MetalDevice::get();
        let x1 = MetalTensor::from_f32(dev, vec![1., 2., 3., 4.], vec![1, 1, 4]);
        let w1 = MetalTensor::from_f32(dev, vec![1., 1.], vec![1, 1, 2]);
        let y1 = conv1d(dev, &x1, &w1, 1, 0, 1, 1).unwrap();
        let r1 =
            conv1d_requirements(&x1.layout, x1.dtype, &w1.layout, w1.dtype, 1, 0, 1, 1).unwrap();
        let d1 = output(dev, &r1);
        conv1d_into(dev, &x1, &w1, 1, 0, 1, 1, &d1).unwrap();

        let x2 = MetalTensor::from_f32(dev, (1..=9).map(|v| v as f32).collect(), vec![1, 1, 3, 3]);
        let w2 = MetalTensor::from_f32(dev, vec![1., 0., 0., 1.], vec![1, 1, 2, 2]);
        let y2 = conv2d(dev, &x2, &w2, 1, 0, 1, 1).unwrap();
        let r2 =
            conv2d_requirements(&x2.layout, x2.dtype, &w2.layout, w2.dtype, 1, 0, 1, 1).unwrap();
        let d2 = output(dev, &r2);
        conv2d_into(dev, &x2, &w2, 1, 0, 1, 1, &d2).unwrap();

        let t1 = conv_transpose1d(dev, &x1, &w1, 2, 0, 0, 1, 1).unwrap();
        let tr1 = conv_transpose1d_requirements(
            &x1.layout, x1.dtype, &w1.layout, w1.dtype, 2, 0, 0, 1, 1,
        )
        .unwrap();
        let td1 = output(dev, &tr1);
        conv_transpose1d_into(dev, &x1, &w1, 2, 0, 0, 1, 1, &td1).unwrap();

        let t2 = conv_transpose2d(dev, &x2, &w2, 2, 0, 0, 1, 1).unwrap();
        let tr2 = conv_transpose2d_requirements(
            &x2.layout, x2.dtype, &w2.layout, w2.dtype, 2, 0, 0, 1, 1,
        )
        .unwrap();
        let td2 = output(dev, &tr2);
        conv_transpose2d_into(dev, &x2, &w2, 2, 0, 0, 1, 1, &td2).unwrap();

        let g1 = MetalTensor::from_f32(dev, vec![1., 1., 1.], vec![1, 1, 3]);
        let b1 = conv1d_backward_w(dev, &x1, &g1, 2, 1, 1, 0, 1, 1).unwrap();
        let br1 = conv1d_backward_w_requirements(
            &x1.layout, x1.dtype, &g1.layout, g1.dtype, 2, 1, 1, 0, 1, 1,
        )
        .unwrap();
        let bd1 = output(dev, &br1);
        conv1d_backward_w_into(dev, &x1, &g1, 2, 1, 1, 0, 1, 1, &bd1).unwrap();

        let g2 = MetalTensor::from_f32(dev, vec![1.; 4], vec![1, 1, 2, 2]);
        let b2 = conv2d_backward_w(dev, &x2, &g2, [2, 2], 1, 1, 0, 1, 1).unwrap();
        let br2 = conv2d_backward_w_requirements(
            &x2.layout,
            x2.dtype,
            &g2.layout,
            g2.dtype,
            [2, 2],
            1,
            1,
            0,
            1,
            1,
        )
        .unwrap();
        let bd2 = output(dev, &br2);
        conv2d_backward_w_into(dev, &x2, &g2, [2, 2], 1, 1, 0, 1, 1, &bd2).unwrap();

        dev.synchronize().unwrap();
        assert_eq!(y1.read_f32().unwrap(), d1.read_f32().unwrap());
        assert_eq!(y2.read_f32().unwrap(), d2.read_f32().unwrap());
        assert_eq!(t1.read_f32().unwrap(), td1.read_f32().unwrap());
        assert_eq!(t2.read_f32().unwrap(), td2.read_f32().unwrap());
        assert_eq!(b1.read_f32().unwrap(), bd1.read_f32().unwrap());
        assert_eq!(b2.read_f32().unwrap(), bd2.read_f32().unwrap());
    }

    #[test]
    fn into_uses_no_device_allocation_after_warm() {
        let dev = MetalDevice::new(0).unwrap();
        let x = MetalTensor::from_f32(&dev, vec![1., 2., 3., 4.], vec![1, 1, 4]);
        let w = MetalTensor::from_f32(&dev, vec![1., 1.], vec![1, 1, 2]);
        let requirement =
            conv1d_requirements(&x.layout, x.dtype, &w.layout, w.dtype, 1, 0, 1, 1).unwrap();
        let destination = output(&dev, &requirement);
        let error = conv1d_into(&dev, &x, &w, 1, 0, 1, 1, &destination).unwrap_err();
        assert!(error.contains("not warm"), "{error}");
        compile_conv1d_layouts(&dev, &x.layout, x.dtype, &w.layout, w.dtype, 1, 0, 1, 1).unwrap();
        let _dispatch_guard = dev.begin_executable_dispatch().unwrap();
        let result = conv1d_into(&dev, &x, &w, 1, 0, 1, 1, &destination);
        result.unwrap();
        dev.synchronize().unwrap();
    }

    #[test]
    fn requirements_are_exact_for_layouts_and_dtypes() {
        let input = Layout::new(vec![2, 3, 8], vec![32, 1, 4], 5);
        let weight = Layout::new(vec![6, 3, 3], vec![1, 18, 6], 7);
        let requirements =
            conv1d_requirements(&input, DType::BF16, &weight, DType::BF16, 2, 1, 1, 1).unwrap();
        assert_eq!(requirements.output.shape, [2, 6, 4]);
        assert_eq!(requirements.output_elements, 48);
        assert_eq!(requirements.output_bytes, 96);
        assert_eq!(requirements.scratch_bytes, 0);
        assert_eq!(requirements.staging_bytes, 0);
        assert_eq!(requirements.pipeline_count, 1);
    }

    #[test]
    fn conv2d_backward_w_basic() {
        let dev = MetalDevice::get();
        let x = MetalTensor::from_f32(dev, (1..=9).map(|v| v as f32).collect(), vec![1, 1, 3, 3]);
        let gradient = MetalTensor::from_f32(dev, vec![1.; 4], vec![1, 1, 2, 2]);
        let output = conv2d_backward_w(dev, &x, &gradient, [2, 2], 1, 1, 0, 1, 1).unwrap();
        dev.synchronize().unwrap();
        assert_eq!(output.read_f32().unwrap(), vec![12., 16., 24., 28.]);
    }
}

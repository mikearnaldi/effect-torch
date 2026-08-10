use super::tensor::{CpuBuffer, CpuDestination, CpuTensorRequirement, Elem, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvAlgorithm {
    DirectF64Accumulator,
}

/// Exact output, scratch, and selected algorithm for one convolution invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvRequirements {
    pub output: CpuTensorRequirement,
    pub scratch: Vec<CpuTensorRequirement>,
    pub algorithm: ConvAlgorithm,
    pub output_elements: usize,
}

impl ConvRequirements {
    pub fn scratch_bytes(&self) -> usize {
        self.scratch
            .iter()
            .try_fold(0usize, |total, requirement| {
                total.checked_add(requirement.bytes)
            })
            .expect("convolution scratch byte size overflow")
    }
}

fn allocate_output(
    requirement: CpuTensorRequirement,
    write: impl FnOnce(&mut CpuDestination<'_>) -> Result<(), String>,
) -> Tensor {
    let mut output = Tensor::empty(&requirement.shape, requirement.dtype);
    {
        let mut destination = output
            .destination()
            .expect("new CPU tensor storage must be unique");
        write(&mut destination).expect("new CPU tensor must satisfy convolution requirements");
    }
    output
}

fn require_rank(tensor: &Tensor, rank: usize, operation: &str) -> Result<(), String> {
    if tensor.shape().len() != rank {
        return Err(format!(
            "{operation}: expected rank {rank}, got {:?}",
            tensor.shape()
        ));
    }
    Ok(())
}

fn same_dtype(first: &Tensor, second: &Tensor, operation: &str) -> Result<(), String> {
    if first.dtype() != second.dtype() {
        return Err(format!(
            "{operation}: input dtype {} does not match second operand dtype {}",
            first.dtype(),
            second.dtype()
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
    operation: &str,
) -> Result<usize, String> {
    if stride == 0 || dilation == 0 || kernel == 0 {
        return Err(format!(
            "{operation}: stride, dilation, and kernel size must be non-zero"
        ));
    }
    let padded = input
        .checked_add(
            padding
                .checked_mul(2)
                .ok_or_else(|| format!("{operation}: padding overflow"))?,
        )
        .ok_or_else(|| format!("{operation}: padded input size overflow"))?;
    let receptive = dilation
        .checked_mul(kernel - 1)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| format!("{operation}: receptive field overflow"))?;
    if padded < receptive {
        return Err(format!(
            "{operation}: kernel receptive field {receptive} exceeds padded input {padded}"
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
    operation: &str,
) -> Result<usize, String> {
    if input == 0 || stride == 0 || dilation == 0 || kernel == 0 {
        return Err(format!(
            "{operation}: input, stride, dilation, and kernel size must be non-zero"
        ));
    }
    let expanded = (input - 1)
        .checked_mul(stride)
        .and_then(|value| value.checked_add(dilation.checked_mul(kernel - 1)?))
        .and_then(|value| value.checked_add(output_padding))
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| format!("{operation}: output size overflow"))?;
    let trim = padding
        .checked_mul(2)
        .ok_or_else(|| format!("{operation}: padding overflow"))?;
    expanded
        .checked_sub(trim)
        .ok_or_else(|| format!("{operation}: padding exceeds expanded output"))
}

fn forward_channels(x: &Tensor, w: &Tensor, groups: usize, operation: &str) -> Result<(), String> {
    if groups == 0 || x.shape()[1] % groups != 0 || w.shape()[0] % groups != 0 {
        return Err(format!(
            "{operation}: input and output channels must be divisible by non-zero groups"
        ));
    }
    if w.shape()[1]
        .checked_mul(groups)
        .ok_or_else(|| format!("{operation}: grouped channel count overflow"))?
        != x.shape()[1]
    {
        return Err(format!(
            "{operation}: weight channels {:?} do not match input channels {} with {groups} groups",
            w.shape(),
            x.shape()[1]
        ));
    }
    Ok(())
}

fn transpose_channels(
    x: &Tensor,
    w: &Tensor,
    groups: usize,
    operation: &str,
) -> Result<(), String> {
    if groups == 0 || x.shape()[1] % groups != 0 || w.shape()[0] != x.shape()[1] {
        return Err(format!(
            "{operation}: weight dim 0 must equal input channels and channels must be divisible by non-zero groups"
        ));
    }
    Ok(())
}

fn backward_channels(
    x: &Tensor,
    gradient: &Tensor,
    out_channels: usize,
    groups: usize,
    operation: &str,
) -> Result<(), String> {
    if groups == 0
        || x.shape()[1] % groups != 0
        || out_channels % groups != 0
        || gradient.shape()[0] != x.shape()[0]
        || gradient.shape()[1] != out_channels
    {
        return Err(format!(
            "{operation}: incompatible batch/channel dimensions for {out_channels} output channels and {groups} groups"
        ));
    }
    Ok(())
}

fn conv1d_shape(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<[usize; 3], String> {
    require_rank(x, 3, "conv1d")?;
    require_rank(w, 3, "conv1d")?;
    same_dtype(x, w, "conv1d")?;
    forward_channels(x, w, groups, "conv1d")?;
    let length = conv_output_dim(
        x.shape()[2],
        w.shape()[2],
        stride,
        padding,
        dilation,
        "conv1d",
    )?;
    Ok([x.shape()[0], w.shape()[0], length])
}

fn conv2d_shape(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<[usize; 4], String> {
    require_rank(x, 4, "conv2d")?;
    require_rank(w, 4, "conv2d")?;
    same_dtype(x, w, "conv2d")?;
    forward_channels(x, w, groups, "conv2d")?;
    let height = conv_output_dim(
        x.shape()[2],
        w.shape()[2],
        stride,
        padding,
        dilation,
        "conv2d",
    )?;
    let width = conv_output_dim(
        x.shape()[3],
        w.shape()[3],
        stride,
        padding,
        dilation,
        "conv2d",
    )?;
    Ok([x.shape()[0], w.shape()[0], height, width])
}

#[allow(clippy::too_many_arguments)]
fn conv_transpose1d_shape(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<[usize; 3], String> {
    require_rank(x, 3, "conv_transpose1d")?;
    require_rank(w, 3, "conv_transpose1d")?;
    same_dtype(x, w, "conv_transpose1d")?;
    transpose_channels(x, w, groups, "conv_transpose1d")?;
    let length = conv_transpose_output_dim(
        x.shape()[2],
        w.shape()[2],
        stride,
        padding,
        output_padding,
        dilation,
        "conv_transpose1d",
    )?;
    let out_channels = w.shape()[1]
        .checked_mul(groups)
        .ok_or_else(|| "conv_transpose1d: output channel count overflow".to_string())?;
    Ok([x.shape()[0], out_channels, length])
}

#[allow(clippy::too_many_arguments)]
fn conv_transpose2d_shape(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<[usize; 4], String> {
    require_rank(x, 4, "conv_transpose2d")?;
    require_rank(w, 4, "conv_transpose2d")?;
    same_dtype(x, w, "conv_transpose2d")?;
    transpose_channels(x, w, groups, "conv_transpose2d")?;
    let height = conv_transpose_output_dim(
        x.shape()[2],
        w.shape()[2],
        stride,
        padding,
        output_padding,
        dilation,
        "conv_transpose2d",
    )?;
    let width = conv_transpose_output_dim(
        x.shape()[3],
        w.shape()[3],
        stride,
        padding,
        output_padding,
        dilation,
        "conv_transpose2d",
    )?;
    let out_channels = w.shape()[1]
        .checked_mul(groups)
        .ok_or_else(|| "conv_transpose2d: output channel count overflow".to_string())?;
    Ok([x.shape()[0], out_channels, height, width])
}

#[allow(clippy::too_many_arguments)]
fn conv1d_backward_w_shape(
    x: &Tensor,
    gradient: &Tensor,
    kernel: usize,
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<[usize; 3], String> {
    require_rank(x, 3, "conv1d_backward_w")?;
    require_rank(gradient, 3, "conv1d_backward_w")?;
    same_dtype(x, gradient, "conv1d_backward_w")?;
    backward_channels(x, gradient, out_channels, groups, "conv1d_backward_w")?;
    let expected = conv_output_dim(
        x.shape()[2],
        kernel,
        stride,
        padding,
        dilation,
        "conv1d_backward_w",
    )?;
    if gradient.shape()[2] != expected {
        return Err(format!(
            "conv1d_backward_w: gradient length {}, expected {expected}",
            gradient.shape()[2]
        ));
    }
    Ok([out_channels, x.shape()[1] / groups, kernel])
}

#[allow(clippy::too_many_arguments)]
fn conv2d_backward_w_shape(
    x: &Tensor,
    gradient: &Tensor,
    kernel: [usize; 2],
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<[usize; 4], String> {
    require_rank(x, 4, "conv2d_backward_w")?;
    require_rank(gradient, 4, "conv2d_backward_w")?;
    same_dtype(x, gradient, "conv2d_backward_w")?;
    backward_channels(x, gradient, out_channels, groups, "conv2d_backward_w")?;
    let expected_height = conv_output_dim(
        x.shape()[2],
        kernel[0],
        stride,
        padding,
        dilation,
        "conv2d_backward_w",
    )?;
    let expected_width = conv_output_dim(
        x.shape()[3],
        kernel[1],
        stride,
        padding,
        dilation,
        "conv2d_backward_w",
    )?;
    if gradient.shape()[2..] != [expected_height, expected_width] {
        return Err(format!(
            "conv2d_backward_w: gradient spatial shape {:?}, expected [{expected_height}, {expected_width}]",
            &gradient.shape()[2..]
        ));
    }
    Ok([out_channels, x.shape()[1] / groups, kernel[0], kernel[1]])
}

fn requirement(shape: &[usize], tensor: &Tensor) -> Result<ConvRequirements, String> {
    let output_elements = shape
        .iter()
        .try_fold(1usize, |total, &dimension| total.checked_mul(dimension))
        .ok_or_else(|| "convolution output element count overflow".to_string())?;
    output_elements
        .checked_mul(tensor.dtype().size_in_bytes())
        .ok_or_else(|| "convolution output byte size overflow".to_string())?;
    Ok(ConvRequirements {
        output: CpuTensorRequirement::new(shape, tensor.dtype()),
        scratch: Vec::new(),
        algorithm: ConvAlgorithm::DirectF64Accumulator,
        output_elements,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn conv1d_requirements(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<ConvRequirements, String> {
    let shape = conv1d_shape(x, w, stride, padding, dilation, groups)?;
    requirement(&shape, x)
}

#[allow(clippy::too_many_arguments)]
pub fn conv1d_output_requirements(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<CpuTensorRequirement, String> {
    Ok(conv1d_requirements(x, w, stride, padding, dilation, groups)?.output)
}

#[allow(clippy::too_many_arguments)]
pub fn conv1d_scratch_requirements(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Vec<CpuTensorRequirement>, String> {
    Ok(conv1d_requirements(x, w, stride, padding, dilation, groups)?.scratch)
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_requirements(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<ConvRequirements, String> {
    let shape = conv2d_shape(x, w, stride, padding, dilation, groups)?;
    requirement(&shape, x)
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_output_requirements(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<CpuTensorRequirement, String> {
    Ok(conv2d_requirements(x, w, stride, padding, dilation, groups)?.output)
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_scratch_requirements(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Vec<CpuTensorRequirement>, String> {
    Ok(conv2d_requirements(x, w, stride, padding, dilation, groups)?.scratch)
}

#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d_requirements(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<ConvRequirements, String> {
    let shape = conv_transpose1d_shape(x, w, stride, padding, output_padding, dilation, groups)?;
    requirement(&shape, x)
}

#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d_output_requirements(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<CpuTensorRequirement, String> {
    Ok(
        conv_transpose1d_requirements(x, w, stride, padding, output_padding, dilation, groups)?
            .output,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d_scratch_requirements(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Vec<CpuTensorRequirement>, String> {
    Ok(
        conv_transpose1d_requirements(x, w, stride, padding, output_padding, dilation, groups)?
            .scratch,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d_requirements(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<ConvRequirements, String> {
    let shape = conv_transpose2d_shape(x, w, stride, padding, output_padding, dilation, groups)?;
    requirement(&shape, x)
}

#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d_output_requirements(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<CpuTensorRequirement, String> {
    Ok(
        conv_transpose2d_requirements(x, w, stride, padding, output_padding, dilation, groups)?
            .output,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d_scratch_requirements(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Vec<CpuTensorRequirement>, String> {
    Ok(
        conv_transpose2d_requirements(x, w, stride, padding, output_padding, dilation, groups)?
            .scratch,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv1d_backward_w_requirements(
    x: &Tensor,
    gradient: &Tensor,
    kernel: usize,
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<ConvRequirements, String> {
    let shape = conv1d_backward_w_shape(
        x,
        gradient,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )?;
    requirement(&shape, x)
}

#[allow(clippy::too_many_arguments)]
pub fn conv1d_backward_w_output_requirements(
    x: &Tensor,
    gradient: &Tensor,
    kernel: usize,
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<CpuTensorRequirement, String> {
    Ok(conv1d_backward_w_requirements(
        x,
        gradient,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )?
    .output)
}

#[allow(clippy::too_many_arguments)]
pub fn conv1d_backward_w_scratch_requirements(
    x: &Tensor,
    gradient: &Tensor,
    kernel: usize,
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Vec<CpuTensorRequirement>, String> {
    Ok(conv1d_backward_w_requirements(
        x,
        gradient,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )?
    .scratch)
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_backward_w_requirements(
    x: &Tensor,
    gradient: &Tensor,
    kernel: [usize; 2],
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<ConvRequirements, String> {
    let shape = conv2d_backward_w_shape(
        x,
        gradient,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )?;
    requirement(&shape, x)
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_backward_w_output_requirements(
    x: &Tensor,
    gradient: &Tensor,
    kernel: [usize; 2],
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<CpuTensorRequirement, String> {
    Ok(conv2d_backward_w_requirements(
        x,
        gradient,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )?
    .output)
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_backward_w_scratch_requirements(
    x: &Tensor,
    gradient: &Tensor,
    kernel: [usize; 2],
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Vec<CpuTensorRequirement>, String> {
    Ok(conv2d_backward_w_requirements(
        x,
        gradient,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )?
    .scratch)
}

fn conv1d_into_impl<T: Elem>(
    x: &Tensor,
    x_values: &[T],
    w: &Tensor,
    w_values: &[T],
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    shape: [usize; 3],
    destination: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let [n, c_out, length_out] = shape;
    let channels_per_group = w.shape()[1];
    let kernel = w.shape()[2];
    let output_channels_per_group = c_out / groups;
    destination.write::<T, _>("conv1d", &shape, |output| {
        for batch in 0..n {
            for output_channel in 0..c_out {
                let group = output_channel / output_channels_per_group;
                for output_position in 0..length_out {
                    let mut accumulator = 0.0f64;
                    for input_channel_in_group in 0..channels_per_group {
                        let input_channel = group * channels_per_group + input_channel_in_group;
                        for kernel_position in 0..kernel {
                            let padded_position =
                                output_position * stride + kernel_position * dilation;
                            if padded_position < padding {
                                continue;
                            }
                            let input_position = padded_position - padding;
                            if input_position >= x.shape()[2] {
                                continue;
                            }
                            let x_index = x.layout.offset()
                                + batch * x.layout.strides()[0]
                                + input_channel * x.layout.strides()[1]
                                + input_position * x.layout.strides()[2];
                            let w_index = w.layout.offset()
                                + output_channel * w.layout.strides()[0]
                                + input_channel_in_group * w.layout.strides()[1]
                                + kernel_position * w.layout.strides()[2];
                            accumulator += x_values[x_index].to_f64() * w_values[w_index].to_f64();
                        }
                    }
                    output[(batch * c_out + output_channel) * length_out + output_position] =
                        T::from_f64(accumulator);
                }
            }
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn conv2d_into_impl<T: Elem>(
    x: &Tensor,
    x_values: &[T],
    w: &Tensor,
    w_values: &[T],
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    shape: [usize; 4],
    destination: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let [n, c_out, height_out, width_out] = shape;
    let channels_per_group = w.shape()[1];
    let (kernel_height, kernel_width) = (w.shape()[2], w.shape()[3]);
    let output_channels_per_group = c_out / groups;
    destination.write::<T, _>("conv2d", &shape, |output| {
        for batch in 0..n {
            for output_channel in 0..c_out {
                let group = output_channel / output_channels_per_group;
                for output_y in 0..height_out {
                    for output_x in 0..width_out {
                        let mut accumulator = 0.0f64;
                        for input_channel_in_group in 0..channels_per_group {
                            let input_channel = group * channels_per_group + input_channel_in_group;
                            for kernel_y in 0..kernel_height {
                                let padded_y = output_y * stride + kernel_y * dilation;
                                if padded_y < padding {
                                    continue;
                                }
                                let input_y = padded_y - padding;
                                if input_y >= x.shape()[2] {
                                    continue;
                                }
                                for kernel_x in 0..kernel_width {
                                    let padded_x = output_x * stride + kernel_x * dilation;
                                    if padded_x < padding {
                                        continue;
                                    }
                                    let input_x = padded_x - padding;
                                    if input_x >= x.shape()[3] {
                                        continue;
                                    }
                                    let x_index = x.layout.offset()
                                        + batch * x.layout.strides()[0]
                                        + input_channel * x.layout.strides()[1]
                                        + input_y * x.layout.strides()[2]
                                        + input_x * x.layout.strides()[3];
                                    let w_index = w.layout.offset()
                                        + output_channel * w.layout.strides()[0]
                                        + input_channel_in_group * w.layout.strides()[1]
                                        + kernel_y * w.layout.strides()[2]
                                        + kernel_x * w.layout.strides()[3];
                                    accumulator +=
                                        x_values[x_index].to_f64() * w_values[w_index].to_f64();
                                }
                            }
                        }
                        output[((batch * c_out + output_channel) * height_out + output_y)
                            * width_out
                            + output_x] = T::from_f64(accumulator);
                    }
                }
            }
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn conv_transpose1d_into_impl<T: Elem>(
    x: &Tensor,
    x_values: &[T],
    w: &Tensor,
    w_values: &[T],
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    shape: [usize; 3],
    destination: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let [n, c_out, length_out] = shape;
    let input_channels_per_group = x.shape()[1] / groups;
    let output_channels_per_group = w.shape()[1];
    let kernel = w.shape()[2];
    destination.write::<T, _>("conv_transpose1d", &shape, |output| {
        for batch in 0..n {
            for output_channel in 0..c_out {
                let group = output_channel / output_channels_per_group;
                let output_channel_in_group = output_channel % output_channels_per_group;
                for output_position in 0..length_out {
                    let mut accumulator = 0.0f64;
                    for input_channel_in_group in 0..input_channels_per_group {
                        let input_channel =
                            group * input_channels_per_group + input_channel_in_group;
                        for input_position in 0..x.shape()[2] {
                            for kernel_position in 0..kernel {
                                let expanded_position =
                                    input_position * stride + kernel_position * dilation;
                                if expanded_position < padding
                                    || expanded_position - padding != output_position
                                {
                                    continue;
                                }
                                let x_index = x.layout.offset()
                                    + batch * x.layout.strides()[0]
                                    + input_channel * x.layout.strides()[1]
                                    + input_position * x.layout.strides()[2];
                                let w_index = w.layout.offset()
                                    + input_channel * w.layout.strides()[0]
                                    + output_channel_in_group * w.layout.strides()[1]
                                    + kernel_position * w.layout.strides()[2];
                                accumulator +=
                                    x_values[x_index].to_f64() * w_values[w_index].to_f64();
                            }
                        }
                    }
                    output[(batch * c_out + output_channel) * length_out + output_position] =
                        T::from_f64(accumulator);
                }
            }
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn conv_transpose2d_into_impl<T: Elem>(
    x: &Tensor,
    x_values: &[T],
    w: &Tensor,
    w_values: &[T],
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    shape: [usize; 4],
    destination: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let [n, c_out, height_out, width_out] = shape;
    let input_channels_per_group = x.shape()[1] / groups;
    let output_channels_per_group = w.shape()[1];
    let (kernel_height, kernel_width) = (w.shape()[2], w.shape()[3]);
    destination.write::<T, _>("conv_transpose2d", &shape, |output| {
        for batch in 0..n {
            for output_channel in 0..c_out {
                let group = output_channel / output_channels_per_group;
                let output_channel_in_group = output_channel % output_channels_per_group;
                for output_y in 0..height_out {
                    for output_x in 0..width_out {
                        let mut accumulator = 0.0f64;
                        for input_channel_in_group in 0..input_channels_per_group {
                            let input_channel =
                                group * input_channels_per_group + input_channel_in_group;
                            for input_y in 0..x.shape()[2] {
                                for input_x in 0..x.shape()[3] {
                                    for kernel_y in 0..kernel_height {
                                        let expanded_y = input_y * stride + kernel_y * dilation;
                                        if expanded_y < padding || expanded_y - padding != output_y
                                        {
                                            continue;
                                        }
                                        for kernel_x in 0..kernel_width {
                                            let expanded_x = input_x * stride + kernel_x * dilation;
                                            if expanded_x < padding
                                                || expanded_x - padding != output_x
                                            {
                                                continue;
                                            }
                                            let x_index = x.layout.offset()
                                                + batch * x.layout.strides()[0]
                                                + input_channel * x.layout.strides()[1]
                                                + input_y * x.layout.strides()[2]
                                                + input_x * x.layout.strides()[3];
                                            let w_index = w.layout.offset()
                                                + input_channel * w.layout.strides()[0]
                                                + output_channel_in_group * w.layout.strides()[1]
                                                + kernel_y * w.layout.strides()[2]
                                                + kernel_x * w.layout.strides()[3];
                                            accumulator += x_values[x_index].to_f64()
                                                * w_values[w_index].to_f64();
                                        }
                                    }
                                }
                            }
                        }
                        output[((batch * c_out + output_channel) * height_out + output_y)
                            * width_out
                            + output_x] = T::from_f64(accumulator);
                    }
                }
            }
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn conv1d_backward_w_into_impl<T: Elem>(
    x: &Tensor,
    x_values: &[T],
    gradient: &Tensor,
    gradient_values: &[T],
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    shape: [usize; 3],
    destination: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let [out_channels, channels_per_group, kernel] = shape;
    let output_channels_per_group = out_channels / groups;
    destination.write::<T, _>("conv1d_backward_w", &shape, |output| {
        for output_channel in 0..out_channels {
            let group = output_channel / output_channels_per_group;
            for input_channel_in_group in 0..channels_per_group {
                let input_channel = group * channels_per_group + input_channel_in_group;
                for kernel_position in 0..kernel {
                    let mut accumulator = 0.0f64;
                    for batch in 0..x.shape()[0] {
                        for output_position in 0..gradient.shape()[2] {
                            let padded_position =
                                output_position * stride + kernel_position * dilation;
                            if padded_position < padding {
                                continue;
                            }
                            let input_position = padded_position - padding;
                            if input_position >= x.shape()[2] {
                                continue;
                            }
                            let x_index = x.layout.offset()
                                + batch * x.layout.strides()[0]
                                + input_channel * x.layout.strides()[1]
                                + input_position * x.layout.strides()[2];
                            let gradient_index = gradient.layout.offset()
                                + batch * gradient.layout.strides()[0]
                                + output_channel * gradient.layout.strides()[1]
                                + output_position * gradient.layout.strides()[2];
                            accumulator += x_values[x_index].to_f64()
                                * gradient_values[gradient_index].to_f64();
                        }
                    }
                    output[(output_channel * channels_per_group + input_channel_in_group)
                        * kernel
                        + kernel_position] = T::from_f64(accumulator);
                }
            }
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn conv2d_backward_w_into_impl<T: Elem>(
    x: &Tensor,
    x_values: &[T],
    gradient: &Tensor,
    gradient_values: &[T],
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    shape: [usize; 4],
    destination: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let [out_channels, channels_per_group, kernel_height, kernel_width] = shape;
    let output_channels_per_group = out_channels / groups;
    destination.write::<T, _>("conv2d_backward_w", &shape, |output| {
        for output_channel in 0..out_channels {
            let group = output_channel / output_channels_per_group;
            for input_channel_in_group in 0..channels_per_group {
                let input_channel = group * channels_per_group + input_channel_in_group;
                for kernel_y in 0..kernel_height {
                    for kernel_x in 0..kernel_width {
                        let mut accumulator = 0.0f64;
                        for batch in 0..x.shape()[0] {
                            for output_y in 0..gradient.shape()[2] {
                                let padded_y = output_y * stride + kernel_y * dilation;
                                if padded_y < padding {
                                    continue;
                                }
                                let input_y = padded_y - padding;
                                if input_y >= x.shape()[2] {
                                    continue;
                                }
                                for output_x in 0..gradient.shape()[3] {
                                    let padded_x = output_x * stride + kernel_x * dilation;
                                    if padded_x < padding {
                                        continue;
                                    }
                                    let input_x = padded_x - padding;
                                    if input_x >= x.shape()[3] {
                                        continue;
                                    }
                                    let x_index = x.layout.offset()
                                        + batch * x.layout.strides()[0]
                                        + input_channel * x.layout.strides()[1]
                                        + input_y * x.layout.strides()[2]
                                        + input_x * x.layout.strides()[3];
                                    let gradient_index = gradient.layout.offset()
                                        + batch * gradient.layout.strides()[0]
                                        + output_channel * gradient.layout.strides()[1]
                                        + output_y * gradient.layout.strides()[2]
                                        + output_x * gradient.layout.strides()[3];
                                    accumulator += x_values[x_index].to_f64()
                                        * gradient_values[gradient_index].to_f64();
                                }
                            }
                        }
                        output[((output_channel * channels_per_group + input_channel_in_group)
                            * kernel_height
                            + kernel_y)
                            * kernel_width
                            + kernel_x] = T::from_f64(accumulator);
                    }
                }
            }
        }
    })
}

#[allow(clippy::too_many_arguments)]
pub fn conv1d_into(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    destination: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let shape = conv1d_shape(x, w, stride, padding, dilation, groups)?;
    macro_rules! dispatch {
        ($x:expr, $w:expr) => {
            conv1d_into_impl(
                x,
                $x,
                w,
                $w,
                stride,
                padding,
                dilation,
                groups,
                shape,
                destination,
            )
        };
    }
    match (&x.buffer, &w.buffer) {
        (CpuBuffer::F32(x), CpuBuffer::F32(w)) => dispatch!(x, w),
        (CpuBuffer::F64(x), CpuBuffer::F64(w)) => dispatch!(x, w),
        (CpuBuffer::F16(x), CpuBuffer::F16(w)) => dispatch!(x, w),
        (CpuBuffer::BF16(x), CpuBuffer::BF16(w)) => dispatch!(x, w),
        (CpuBuffer::U8(x), CpuBuffer::U8(w)) => dispatch!(x, w),
        (CpuBuffer::U32(x), CpuBuffer::U32(w)) => dispatch!(x, w),
        (CpuBuffer::I64(x), CpuBuffer::I64(w)) => dispatch!(x, w),
        _ => unreachable!("dtype checked before dispatch"),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Tensor {
    let requirements = conv1d_requirements(x, w, stride, padding, dilation, groups)
        .unwrap_or_else(|message| panic!("{message}"));
    allocate_output(requirements.output, |destination| {
        conv1d_into(x, w, stride, padding, dilation, groups, destination)
    })
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_into(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    destination: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let shape = conv2d_shape(x, w, stride, padding, dilation, groups)?;
    macro_rules! dispatch {
        ($x:expr, $w:expr) => {
            conv2d_into_impl(
                x,
                $x,
                w,
                $w,
                stride,
                padding,
                dilation,
                groups,
                shape,
                destination,
            )
        };
    }
    match (&x.buffer, &w.buffer) {
        (CpuBuffer::F32(x), CpuBuffer::F32(w)) => dispatch!(x, w),
        (CpuBuffer::F64(x), CpuBuffer::F64(w)) => dispatch!(x, w),
        (CpuBuffer::F16(x), CpuBuffer::F16(w)) => dispatch!(x, w),
        (CpuBuffer::BF16(x), CpuBuffer::BF16(w)) => dispatch!(x, w),
        (CpuBuffer::U8(x), CpuBuffer::U8(w)) => dispatch!(x, w),
        (CpuBuffer::U32(x), CpuBuffer::U32(w)) => dispatch!(x, w),
        (CpuBuffer::I64(x), CpuBuffer::I64(w)) => dispatch!(x, w),
        _ => unreachable!("dtype checked before dispatch"),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Tensor {
    let requirements = conv2d_requirements(x, w, stride, padding, dilation, groups)
        .unwrap_or_else(|message| panic!("{message}"));
    allocate_output(requirements.output, |destination| {
        conv2d_into(x, w, stride, padding, dilation, groups, destination)
    })
}

#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d_into(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
    destination: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let shape = conv_transpose1d_shape(x, w, stride, padding, output_padding, dilation, groups)?;
    macro_rules! dispatch {
        ($x:expr, $w:expr) => {
            conv_transpose1d_into_impl(
                x,
                $x,
                w,
                $w,
                stride,
                padding,
                dilation,
                groups,
                shape,
                destination,
            )
        };
    }
    match (&x.buffer, &w.buffer) {
        (CpuBuffer::F32(x), CpuBuffer::F32(w)) => dispatch!(x, w),
        (CpuBuffer::F64(x), CpuBuffer::F64(w)) => dispatch!(x, w),
        (CpuBuffer::F16(x), CpuBuffer::F16(w)) => dispatch!(x, w),
        (CpuBuffer::BF16(x), CpuBuffer::BF16(w)) => dispatch!(x, w),
        (CpuBuffer::U8(x), CpuBuffer::U8(w)) => dispatch!(x, w),
        (CpuBuffer::U32(x), CpuBuffer::U32(w)) => dispatch!(x, w),
        (CpuBuffer::I64(x), CpuBuffer::I64(w)) => dispatch!(x, w),
        _ => unreachable!("dtype checked before dispatch"),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Tensor {
    let requirements =
        conv_transpose1d_requirements(x, w, stride, padding, output_padding, dilation, groups)
            .unwrap_or_else(|message| panic!("{message}"));
    allocate_output(requirements.output, |destination| {
        conv_transpose1d_into(
            x,
            w,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            destination,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d_into(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
    destination: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let shape = conv_transpose2d_shape(x, w, stride, padding, output_padding, dilation, groups)?;
    macro_rules! dispatch {
        ($x:expr, $w:expr) => {
            conv_transpose2d_into_impl(
                x,
                $x,
                w,
                $w,
                stride,
                padding,
                dilation,
                groups,
                shape,
                destination,
            )
        };
    }
    match (&x.buffer, &w.buffer) {
        (CpuBuffer::F32(x), CpuBuffer::F32(w)) => dispatch!(x, w),
        (CpuBuffer::F64(x), CpuBuffer::F64(w)) => dispatch!(x, w),
        (CpuBuffer::F16(x), CpuBuffer::F16(w)) => dispatch!(x, w),
        (CpuBuffer::BF16(x), CpuBuffer::BF16(w)) => dispatch!(x, w),
        (CpuBuffer::U8(x), CpuBuffer::U8(w)) => dispatch!(x, w),
        (CpuBuffer::U32(x), CpuBuffer::U32(w)) => dispatch!(x, w),
        (CpuBuffer::I64(x), CpuBuffer::I64(w)) => dispatch!(x, w),
        _ => unreachable!("dtype checked before dispatch"),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Tensor {
    let requirements =
        conv_transpose2d_requirements(x, w, stride, padding, output_padding, dilation, groups)
            .unwrap_or_else(|message| panic!("{message}"));
    allocate_output(requirements.output, |destination| {
        conv_transpose2d_into(
            x,
            w,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            destination,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn conv1d_backward_w_into(
    x: &Tensor,
    gradient: &Tensor,
    kernel: usize,
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    destination: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let shape = conv1d_backward_w_shape(
        x,
        gradient,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )?;
    macro_rules! dispatch {
        ($x:expr, $gradient:expr) => {
            conv1d_backward_w_into_impl(
                x,
                $x,
                gradient,
                $gradient,
                stride,
                padding,
                dilation,
                groups,
                shape,
                destination,
            )
        };
    }
    match (&x.buffer, &gradient.buffer) {
        (CpuBuffer::F32(x), CpuBuffer::F32(gradient)) => dispatch!(x, gradient),
        (CpuBuffer::F64(x), CpuBuffer::F64(gradient)) => dispatch!(x, gradient),
        (CpuBuffer::F16(x), CpuBuffer::F16(gradient)) => dispatch!(x, gradient),
        (CpuBuffer::BF16(x), CpuBuffer::BF16(gradient)) => dispatch!(x, gradient),
        (CpuBuffer::U8(x), CpuBuffer::U8(gradient)) => dispatch!(x, gradient),
        (CpuBuffer::U32(x), CpuBuffer::U32(gradient)) => dispatch!(x, gradient),
        (CpuBuffer::I64(x), CpuBuffer::I64(gradient)) => dispatch!(x, gradient),
        _ => unreachable!("dtype checked before dispatch"),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn conv1d_backward_w(
    x: &Tensor,
    gradient: &Tensor,
    kernel: usize,
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Tensor {
    let requirements = conv1d_backward_w_requirements(
        x,
        gradient,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )
    .unwrap_or_else(|message| panic!("{message}"));
    allocate_output(requirements.output, |destination| {
        conv1d_backward_w_into(
            x,
            gradient,
            kernel,
            out_channels,
            stride,
            padding,
            dilation,
            groups,
            destination,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_backward_w_into(
    x: &Tensor,
    gradient: &Tensor,
    kernel: [usize; 2],
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    destination: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let shape = conv2d_backward_w_shape(
        x,
        gradient,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )?;
    macro_rules! dispatch {
        ($x:expr, $gradient:expr) => {
            conv2d_backward_w_into_impl(
                x,
                $x,
                gradient,
                $gradient,
                stride,
                padding,
                dilation,
                groups,
                shape,
                destination,
            )
        };
    }
    match (&x.buffer, &gradient.buffer) {
        (CpuBuffer::F32(x), CpuBuffer::F32(gradient)) => dispatch!(x, gradient),
        (CpuBuffer::F64(x), CpuBuffer::F64(gradient)) => dispatch!(x, gradient),
        (CpuBuffer::F16(x), CpuBuffer::F16(gradient)) => dispatch!(x, gradient),
        (CpuBuffer::BF16(x), CpuBuffer::BF16(gradient)) => dispatch!(x, gradient),
        (CpuBuffer::U8(x), CpuBuffer::U8(gradient)) => dispatch!(x, gradient),
        (CpuBuffer::U32(x), CpuBuffer::U32(gradient)) => dispatch!(x, gradient),
        (CpuBuffer::I64(x), CpuBuffer::I64(gradient)) => dispatch!(x, gradient),
        _ => unreachable!("dtype checked before dispatch"),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_backward_w(
    x: &Tensor,
    gradient: &Tensor,
    kernel: [usize; 2],
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Tensor {
    let requirements = conv2d_backward_w_requirements(
        x,
        gradient,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )
    .unwrap_or_else(|message| panic!("{message}"));
    allocate_output(requirements.output, |destination| {
        conv2d_backward_w_into(
            x,
            gradient,
            kernel,
            out_channels,
            stride,
            padding,
            dilation,
            groups,
            destination,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuBuffer, ExecutableAllocationGuard, CPU_STORAGE_ALIGNMENT};
    use effect_torch_runtime::{DType, Layout};

    fn f32_data(tensor: &Tensor) -> Vec<f32> {
        let CpuBuffer::F32(values) = &tensor.buffer else {
            panic!()
        };
        values.as_slice().to_vec()
    }

    fn output(requirements: &ConvRequirements) -> Tensor {
        Tensor::empty(&requirements.output.shape, requirements.output.dtype)
    }

    #[test]
    fn conv1d_basic() {
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], vec![1, 1, 4]);
        let w = Tensor::from_vec(vec![1f32, 1.], vec![1, 1, 2]);
        let y = conv1d(&x, &w, 1, 0, 1, 1);
        assert_eq!(y.shape(), &[1, 1, 3]);
        assert_eq!(f32_data(&y), vec![3., 5., 7.]);
    }

    #[test]
    fn conv2d_basic() {
        let x = Tensor::from_vec(
            (1..=9).map(|value| value as f32).collect(),
            vec![1, 1, 3, 3],
        );
        let w = Tensor::from_vec(vec![1f32, 0., 0., 1.], vec![1, 1, 2, 2]);
        let y = conv2d(&x, &w, 1, 0, 1, 1);
        assert_eq!(y.shape(), &[1, 1, 2, 2]);
        assert_eq!(f32_data(&y), vec![6., 8., 12., 14.]);
    }

    #[test]
    fn conv_transpose1d_basic() {
        let x = Tensor::from_vec(vec![1f32, 2.], vec![1, 1, 2]);
        let w = Tensor::from_vec(vec![1f32, 1.], vec![1, 1, 2]);
        let y = conv_transpose1d(&x, &w, 2, 0, 0, 1, 1);
        assert_eq!(y.shape(), &[1, 1, 4]);
        assert_eq!(f32_data(&y), vec![1., 1., 2., 2.]);
    }

    #[test]
    fn backward_w_basic() {
        let x1 = Tensor::from_vec(vec![1f32, 2., 3., 4.], vec![1, 1, 4]);
        let g1 = Tensor::from_vec(vec![1f32, 1., 1.], vec![1, 1, 3]);
        assert_eq!(
            f32_data(&conv1d_backward_w(&x1, &g1, 2, 1, 1, 0, 1, 1)),
            vec![6., 9.]
        );

        let x2 = Tensor::from_vec(
            (1..=9).map(|value| value as f32).collect(),
            vec![1, 1, 3, 3],
        );
        let g2 = Tensor::from_vec(vec![1f32, 1., 1., 1.], vec![1, 1, 2, 2]);
        let dw = conv2d_backward_w(&x2, &g2, [2, 2], 1, 1, 0, 1, 1);
        assert_eq!(f32_data(&dw), vec![12., 16., 24., 28.]);
    }

    #[test]
    fn every_into_variant_matches_its_wrapper() {
        let x1 = Tensor::from_vec(vec![1f32, 2., 3., 4.], vec![1, 1, 4]);
        let w1 = Tensor::from_vec(vec![1f32, -1.], vec![1, 1, 2]);
        let forward1 = conv1d_requirements(&x1, &w1, 1, 1, 1, 1).unwrap();
        let mut forward1_output = output(&forward1);
        conv1d_into(
            &x1,
            &w1,
            1,
            1,
            1,
            1,
            &mut forward1_output.destination().unwrap(),
        )
        .unwrap();
        assert_eq!(
            f32_data(&forward1_output),
            f32_data(&conv1d(&x1, &w1, 1, 1, 1, 1))
        );

        let x2 = Tensor::from_vec(
            (1..=9).map(|value| value as f32).collect(),
            vec![1, 1, 3, 3],
        );
        let w2 = Tensor::from_vec(vec![1f32, 0., 0., 1.], vec![1, 1, 2, 2]);
        let forward2 = conv2d_requirements(&x2, &w2, 1, 1, 1, 1).unwrap();
        let mut forward2_output = output(&forward2);
        conv2d_into(
            &x2,
            &w2,
            1,
            1,
            1,
            1,
            &mut forward2_output.destination().unwrap(),
        )
        .unwrap();
        assert_eq!(
            f32_data(&forward2_output),
            f32_data(&conv2d(&x2, &w2, 1, 1, 1, 1))
        );

        let transpose1 = conv_transpose1d_requirements(&x1, &w1, 2, 1, 1, 1, 1).unwrap();
        let mut transpose1_output = output(&transpose1);
        conv_transpose1d_into(
            &x1,
            &w1,
            2,
            1,
            1,
            1,
            1,
            &mut transpose1_output.destination().unwrap(),
        )
        .unwrap();
        assert_eq!(
            f32_data(&transpose1_output),
            f32_data(&conv_transpose1d(&x1, &w1, 2, 1, 1, 1, 1))
        );

        let transpose2 = conv_transpose2d_requirements(&x2, &w2, 2, 1, 1, 1, 1).unwrap();
        let mut transpose2_output = output(&transpose2);
        conv_transpose2d_into(
            &x2,
            &w2,
            2,
            1,
            1,
            1,
            1,
            &mut transpose2_output.destination().unwrap(),
        )
        .unwrap();
        assert_eq!(
            f32_data(&transpose2_output),
            f32_data(&conv_transpose2d(&x2, &w2, 2, 1, 1, 1, 1))
        );

        let g1 = Tensor::from_vec(vec![1f32; 5], vec![1, 1, 5]);
        let backward1 = conv1d_backward_w_requirements(&x1, &g1, 2, 1, 1, 1, 1, 1).unwrap();
        let mut backward1_output = output(&backward1);
        conv1d_backward_w_into(
            &x1,
            &g1,
            2,
            1,
            1,
            1,
            1,
            1,
            &mut backward1_output.destination().unwrap(),
        )
        .unwrap();
        assert_eq!(
            f32_data(&backward1_output),
            f32_data(&conv1d_backward_w(&x1, &g1, 2, 1, 1, 1, 1, 1))
        );

        let g2 = Tensor::from_vec(vec![1f32; 16], vec![1, 1, 4, 4]);
        let backward2 = conv2d_backward_w_requirements(&x2, &g2, [2, 2], 1, 1, 1, 1, 1).unwrap();
        let mut backward2_output = output(&backward2);
        conv2d_backward_w_into(
            &x2,
            &g2,
            [2, 2],
            1,
            1,
            1,
            1,
            1,
            &mut backward2_output.destination().unwrap(),
        )
        .unwrap();
        assert_eq!(
            f32_data(&backward2_output),
            f32_data(&conv2d_backward_w(&x2, &g2, [2, 2], 1, 1, 1, 1, 1))
        );
    }

    #[test]
    fn requirements_have_exact_size_alignment_and_no_scratch() {
        let x = Tensor::from_vec(vec![1f32; 2 * 3 * 5], vec![2, 3, 5]);
        let w = Tensor::from_vec(vec![1f32; 4 * 3 * 3], vec![4, 3, 3]);
        let requirements = conv1d_requirements(&x, &w, 2, 1, 1, 1).unwrap();
        assert_eq!(requirements.output.shape, vec![2, 4, 3]);
        assert_eq!(
            requirements.output.bytes,
            2 * 4 * 3 * std::mem::size_of::<f32>()
        );
        assert_eq!(requirements.output.alignment, CPU_STORAGE_ALIGNMENT);
        assert_eq!(requirements.algorithm, ConvAlgorithm::DirectF64Accumulator);
        assert_eq!(requirements.output_elements, 2 * 4 * 3);
        assert!(requirements.scratch.is_empty());
        assert_eq!(requirements.scratch_bytes(), 0);

        let transpose_w = Tensor::from_vec(vec![1f32; 3 * 3 * 3], vec![3, 3, 3]);
        let transposed = conv_transpose1d_requirements(&x, &transpose_w, 2, 1, 1, 1, 1).unwrap();
        assert_eq!(transposed.output.shape, vec![2, 3, 10]);
        assert_eq!(
            transposed.output.bytes,
            2 * 3 * 10 * std::mem::size_of::<f32>()
        );
        assert_eq!(transposed.output.alignment, CPU_STORAGE_ALIGNMENT);
    }

    #[test]
    fn into_rejects_invalid_destinations() {
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], vec![1, 1, 4]);
        let w = Tensor::from_vec(vec![1f32, 1.], vec![1, 1, 2]);
        let mut wrong_shape = Tensor::zeros(&[3], DType::F32);
        assert!(conv1d_into(&x, &w, 1, 0, 1, 1, &mut wrong_shape.destination().unwrap()).is_err());

        let mut wrong_dtype = Tensor::zeros(&[1, 1, 3], DType::F64);
        assert!(conv1d_into(&x, &w, 1, 0, 1, 1, &mut wrong_dtype.destination().unwrap()).is_err());

        let short = Tensor::from_vec(vec![0.0f32; 2], vec![2]);
        let mut insufficient = Tensor {
            buffer: short.buffer,
            layout: Layout::contiguous(vec![1, 1, 3]),
        };
        assert!(conv1d_into(&x, &w, 1, 0, 1, 1, &mut insufficient.destination().unwrap()).is_err());
    }

    #[test]
    fn all_destinations_execute_under_allocation_guard() {
        let x1_storage = Tensor::from_vec(vec![1f32, 0., 2., 0., 3., 0., 4., 0.], vec![8]);
        let x1 = x1_storage.view(Layout::new(vec![1, 1, 4], vec![8, 8, 2], 0));
        let w1_storage = Tensor::from_vec(vec![1f32, 0., 1., 0.], vec![4]);
        let w1 = w1_storage.view(Layout::new(vec![1, 1, 2], vec![4, 4, 2], 0));
        let g1 = Tensor::from_vec(vec![1f32; 3], vec![1, 1, 3]);
        let x2_storage = Tensor::from_vec(
            (1..=9).map(|value| value as f32).collect(),
            vec![1, 1, 3, 3],
        );
        let x2 = x2_storage.view(x2_storage.layout.permute(&[0, 1, 3, 2]));
        let w2_storage = Tensor::from_vec(vec![1f32; 4], vec![1, 1, 2, 2]);
        let w2 = w2_storage.view(w2_storage.layout.permute(&[0, 1, 3, 2]));
        let g2 = Tensor::from_vec(vec![1f32; 4], vec![1, 1, 2, 2]);

        let mut forward1 = output(&conv1d_requirements(&x1, &w1, 1, 0, 1, 1).unwrap());
        let mut forward2 = output(&conv2d_requirements(&x2, &w2, 1, 0, 1, 1).unwrap());
        let mut transpose1 =
            output(&conv_transpose1d_requirements(&x1, &w1, 2, 0, 0, 1, 1).unwrap());
        let mut transpose2 =
            output(&conv_transpose2d_requirements(&x2, &w2, 2, 0, 0, 1, 1).unwrap());
        let mut backward1 =
            output(&conv1d_backward_w_requirements(&x1, &g1, 2, 1, 1, 0, 1, 1).unwrap());
        let mut backward2 =
            output(&conv2d_backward_w_requirements(&x2, &g2, [2, 2], 1, 1, 0, 1, 1).unwrap());

        let _guard = ExecutableAllocationGuard::enter();
        conv1d_into(&x1, &w1, 1, 0, 1, 1, &mut forward1.destination().unwrap()).unwrap();
        conv2d_into(&x2, &w2, 1, 0, 1, 1, &mut forward2.destination().unwrap()).unwrap();
        conv_transpose1d_into(
            &x1,
            &w1,
            2,
            0,
            0,
            1,
            1,
            &mut transpose1.destination().unwrap(),
        )
        .unwrap();
        conv_transpose2d_into(
            &x2,
            &w2,
            2,
            0,
            0,
            1,
            1,
            &mut transpose2.destination().unwrap(),
        )
        .unwrap();
        conv1d_backward_w_into(
            &x1,
            &g1,
            2,
            1,
            1,
            0,
            1,
            1,
            &mut backward1.destination().unwrap(),
        )
        .unwrap();
        conv2d_backward_w_into(
            &x2,
            &g2,
            [2, 2],
            1,
            1,
            0,
            1,
            1,
            &mut backward2.destination().unwrap(),
        )
        .unwrap();
    }
}

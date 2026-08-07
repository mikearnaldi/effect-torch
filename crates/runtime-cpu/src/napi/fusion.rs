use super::err::Res;
use super::value::Value;
use crate::{Elem, Tensor};
use effect_torch_compiler::{interpret_core, interpret_reduce_core, Expr, ReduceOp, Scalar};
use effect_torch_graph::Device;
use effect_torch_runtime::DType;

pub use effect_torch_compiler::{adamw_exprs, sgd_exprs};

pub fn is_supported(device: &Device, dtype: DType) -> bool {
    device.is_cpu() && matches!(dtype, DType::F32 | DType::F64)
}

pub fn run(
    exprs: &[Expr],
    inputs: &[Value],
    strides: Option<&[Vec<usize>]>,
    scalars: &[Value],
    n: usize,
    shape: &[usize],
    dtype: DType,
    device: &Device,
) -> Res<Vec<Value>> {
    if !device.is_cpu() {
        return Err("fusion: unsupported device".to_string());
    }
    if let Some(strides) = strides {
        if strides.len() != inputs.len() {
            return Err(format!(
                "fusion: got {} stride entries for {} inputs",
                strides.len(),
                inputs.len()
            ));
        }
    }
    for scalar in scalars {
        if scalar.numel() != 1 {
            return Err(format!(
                "fusion: scalar lanes must have exactly one element, got {}",
                scalar.numel()
            ));
        }
    }
    if n == 0 {
        return Ok(exprs
            .iter()
            .map(|_| Value(Tensor::zeros(shape, dtype)))
            .collect());
    }
    match dtype {
        DType::F32 => bridge_elementwise::<f32>(exprs, inputs, strides, scalars, n, shape),
        DType::F64 => bridge_elementwise::<f64>(exprs, inputs, strides, scalars, n, shape),
        _ => Err(format!("fusion: unsupported dtype {dtype:?}")),
    }
}

fn bridge_elementwise<T: Scalar + Elem>(
    exprs: &[Expr],
    inputs: &[Value],
    strides: Option<&[Vec<usize>]>,
    scalars: &[Value],
    n: usize,
    shape: &[usize],
) -> Res<Vec<Value>> {
    let owned: Vec<Tensor> = inputs
        .iter()
        .chain(scalars)
        .map(|value| value.tensor().contiguous())
        .collect();
    let mut slices = Vec::with_capacity(inputs.len());
    for tensor in &owned[..inputs.len()] {
        slices.push(native_slice::<T>(tensor)?);
    }
    let mut scalar_values = Vec::with_capacity(scalars.len());
    for tensor in &owned[inputs.len()..] {
        scalar_values.push(native_slice::<T>(tensor)?[0]);
    }
    Ok(
        interpret_core::<T>(exprs, &slices, strides, &scalar_values, n, shape)
            .into_iter()
            .map(|output| Value(Tensor::from_vec(output, shape.to_vec())))
            .collect(),
    )
}

fn native_slice<T: Elem>(tensor: &Tensor) -> Res<&[T]> {
    T::slice_of(tensor).ok_or_else(|| {
        "fusion: native bridge expects contiguous inputs of matching dtype".to_string()
    })
}

#[allow(clippy::too_many_arguments)]
pub fn run_reduce(
    op: ReduceOp,
    expr: &Expr,
    inputs: &[Value],
    strides: &[Vec<usize>],
    in_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    out_shape: &[usize],
    dtype: DType,
    device: &Device,
) -> Res<Value> {
    if !device.is_cpu() {
        return Err("fusion: unsupported device".to_string());
    }
    if strides.len() != inputs.len() {
        return Err(format!(
            "fusion: got {} stride entries for {} inputs",
            strides.len(),
            inputs.len()
        ));
    }
    if out_shape.iter().product::<usize>() == 0 {
        return Ok(Value(Tensor::zeros(out_shape, dtype)));
    }
    match dtype {
        DType::F32 => bridge_reduce::<f32>(
            op, expr, inputs, strides, in_shape, dims, keepdims, out_shape,
        ),
        DType::F64 => bridge_reduce::<f64>(
            op, expr, inputs, strides, in_shape, dims, keepdims, out_shape,
        ),
        _ => Err(format!("fusion: unsupported dtype {dtype:?}")),
    }
}

#[allow(clippy::too_many_arguments)]
fn bridge_reduce<T: Scalar + Elem>(
    op: ReduceOp,
    expr: &Expr,
    inputs: &[Value],
    strides: &[Vec<usize>],
    in_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    out_shape: &[usize],
) -> Res<Value> {
    let owned: Vec<Tensor> = inputs
        .iter()
        .map(|value| value.tensor().contiguous())
        .collect();
    let slices = owned
        .iter()
        .map(native_slice::<T>)
        .collect::<Res<Vec<_>>>()?;
    let output = interpret_reduce_core::<T>(
        op, expr, &slices, strides, in_shape, dims, keepdims, out_shape,
    );
    Ok(Value(Tensor::from_vec(output, out_shape.to_vec())))
}

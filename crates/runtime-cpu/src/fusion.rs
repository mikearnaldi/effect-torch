//! Bridge that executes compiler-fused expression trees on CPU tensors.
//!
//! The compiler flattens fusible elementwise (and single-output reduction)
//! subgraphs into expression trees and hands them here as a
//! [`CpuFusionProgram`]. [`prepare`] compiles the trees once; the resulting
//! program is immutable and reused for every execution.
//!
//! Execution ([`run_elementwise_into`], [`run_elementwise_multi_into`],
//! [`run_reduce_into`]) is allocation-free: it writes into planned
//! destinations and uses a caller-provided scratch tensor sized by
//! [`scratch_requirement`]. The scratch holds a per-input lane cache followed
//! by the program's value stack, both in the command's native dtype.
//!
//! Inputs are read through explicit per-lane strides when any lane is
//! non-contiguous (broadcasting is expressed as stride-0 entries); contiguous
//! inputs may omit strides and are read linearly. Every read offset is
//! validated against the lane's storage before execution, so a mismatched
//! plan fails cleanly instead of reading out of bounds. Only `f32` and `f64`
//! are supported on this backend.

use crate::value::Value;
use crate::{CpuDestination, CpuTensorRequirement, Elem, Tensor};
use effect_torch_compiler::{Expr, ReduceOp, Scalar};
use effect_torch_graph::Device;
use effect_torch_runtime::DType;

type Res<T> = Result<T, String>;

pub use effect_torch_compiler::{adamw_exprs, sgd_exprs, CpuFusionProgram};

/// Whether this backend can execute a fused program for `device`/`dtype`.
pub fn is_supported(device: &Device, dtype: DType) -> bool {
    device.is_cpu() && matches!(dtype, DType::F32 | DType::F64)
}

/// Flattens expression trees once. A compiled CPU command must retain the
/// returned immutable program and reuse it for every execution.
pub fn prepare(exprs: &[Expr]) -> CpuFusionProgram {
    CpuFusionProgram::new(exprs)
}

/// Exact evaluation scratch retained in planned CPU storage. The lane cache
/// and value stack have the command's native dtype.
pub fn scratch_requirement(program: &CpuFusionProgram, dtype: DType) -> Res<CpuTensorRequirement> {
    if !matches!(dtype, DType::F32 | DType::F64) {
        return Err(format!("fusion: unsupported dtype {dtype:?}"));
    }
    Ok(CpuTensorRequirement::new(
        &[program.scratch_elements()],
        dtype,
    ))
}

fn checked_numel(shape: &[usize], operation: &str) -> Res<usize> {
    shape
        .iter()
        .try_fold(1usize, |total, &dimension| total.checked_mul(dimension))
        .ok_or_else(|| format!("{operation}: element count overflow"))
}

fn strided_offset(index: usize, shape: &[usize], strides: &[usize]) -> usize {
    let mut remainder = index;
    let mut offset = 0usize;
    for dimension in (0..shape.len()).rev() {
        let width = shape[dimension].max(1);
        offset += (remainder % width) * strides[dimension];
        remainder /= width;
    }
    offset
}

fn max_strided_offset(shape: &[usize], strides: &[usize]) -> Option<usize> {
    shape
        .iter()
        .zip(strides)
        .try_fold(0usize, |offset, (&width, &stride)| {
            width
                .saturating_sub(1)
                .checked_mul(stride)
                .and_then(|increment| offset.checked_add(increment))
        })
}

fn native_slice<T: Elem>(tensor: &Tensor) -> Res<&[T]> {
    let storage = T::storage_of(&tensor.buffer)
        .ok_or_else(|| "fusion: native bridge expects inputs of matching dtype".to_string())?;
    storage
        .as_slice()
        .get(tensor.layout.offset()..)
        .ok_or_else(|| "fusion: input layout offset exceeds storage".to_string())
}

fn validate_scratch<T: Elem>(program: &CpuFusionProgram, scratch: &CpuDestination<'_>) -> Res<()> {
    scratch.validate_current::<T>("fusion scratch")?;
    let expected = [program.scratch_elements()];
    if scratch.shape() != expected {
        return Err(format!(
            "fusion scratch shape mismatch: expected {expected:?}, got {:?}",
            scratch.shape()
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_elementwise<T: Elem>(
    program: &CpuFusionProgram,
    inputs: &[Value],
    strides: Option<&[Vec<usize>]>,
    scalars: &[Value],
    n: usize,
    shape: &[usize],
    destinations: &[CpuDestination<'_>],
    scratch: &CpuDestination<'_>,
) -> Res<()> {
    if program.input_count() > inputs.len() {
        return Err(format!(
            "fusion: program references {} input lanes, got {}",
            program.input_count(),
            inputs.len()
        ));
    }
    if program.scalar_count() > scalars.len() {
        return Err(format!(
            "fusion: program references {} scalar lanes, got {}",
            program.scalar_count(),
            scalars.len()
        ));
    }
    if destinations.len() != program.output_count() {
        return Err(format!(
            "fusion: program has {} outputs, got {} destinations",
            program.output_count(),
            destinations.len()
        ));
    }
    let shape_n = checked_numel(shape, "fusion")?;
    if n != shape_n {
        return Err(format!(
            "fusion: element count mismatch: shape has {shape_n}, got {n}"
        ));
    }
    if let Some(strides) = strides {
        if strides.len() != inputs.len() {
            return Err(format!(
                "fusion: got {} stride entries for {} inputs",
                strides.len(),
                inputs.len()
            ));
        }
        if let Some((lane, strides)) = strides
            .iter()
            .enumerate()
            .find(|(_, strides)| strides.len() != shape.len())
        {
            return Err(format!(
                "fusion: lane {lane} has {} strides for rank {}",
                strides.len(),
                shape.len()
            ));
        }
    } else if let Some((lane, _)) = inputs
        .iter()
        .enumerate()
        .find(|(_, input)| !input.tensor().layout.is_contiguous())
    {
        return Err(format!(
            "fusion: lane {lane} requires explicit strides for a non-contiguous layout"
        ));
    }
    for (lane, input) in inputs.iter().enumerate() {
        let values = native_slice::<T>(input.tensor())?;
        if n == 0 {
            continue;
        }
        let last = match strides {
            Some(strides) => max_strided_offset(shape, &strides[lane])
                .ok_or_else(|| "fusion: input offset overflow".to_string())?,
            None => n - 1,
        };
        if last >= values.len() {
            return Err(format!(
                "fusion: lane {lane} reads element {last}, but has {} elements",
                values.len()
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
        native_slice::<T>(scalar.tensor())?;
    }
    for destination in destinations {
        destination.validate_current::<T>("fusion")?;
        if destination.shape() != shape {
            return Err(format!(
                "fusion destination shape mismatch: expected {shape:?}, got {:?}",
                destination.shape()
            ));
        }
    }
    validate_scratch::<T>(program, scratch)
}

#[allow(clippy::too_many_arguments)]
fn bridge_elementwise_into<T: Scalar + Elem>(
    program: &CpuFusionProgram,
    inputs: &[Value],
    strides: Option<&[Vec<usize>]>,
    scalars: &[Value],
    n: usize,
    shape: &[usize],
    destinations: &mut [CpuDestination<'_>],
    scratch: &mut CpuDestination<'_>,
) -> Res<()> {
    validate_elementwise::<T>(
        program,
        inputs,
        strides,
        scalars,
        n,
        shape,
        destinations,
        scratch,
    )?;
    scratch.write_current::<T, _>("fusion scratch", |scratch_values| -> Res<()> {
        let (lane_values, values) = scratch_values.split_at_mut(program.input_count());
        for (output_index, destination) in destinations.iter_mut().enumerate() {
            destination.write::<T, _>("fusion", shape, |output| {
                for index in 0..n {
                    for lane in 0..program.input_count() {
                        let offset = strides.map_or(index, |strides| {
                            strided_offset(index, shape, &strides[lane])
                        });
                        lane_values[lane] = native_slice::<T>(inputs[lane].tensor())
                            .expect("fusion inputs were validated")[offset];
                    }
                    output[index] = program.evaluate(
                        output_index,
                        |lane| lane_values[lane as usize],
                        |scalar| {
                            native_slice::<T>(scalars[scalar as usize].tensor())
                                .expect("fusion scalars were validated")[0]
                        },
                        values,
                    );
                }
            })?;
        }
        Ok(())
    })?
}

/// Executes a multi-output fused elementwise program into `destinations`,
/// one per program output, reading `inputs` under optional explicit
/// `strides` and single-element `scalars`. `n` must equal the element count
/// of `shape`.
#[allow(clippy::too_many_arguments)]
pub fn run_elementwise_multi_into(
    program: &CpuFusionProgram,
    inputs: &[Value],
    strides: Option<&[Vec<usize>]>,
    scalars: &[Value],
    n: usize,
    shape: &[usize],
    dtype: DType,
    device: &Device,
    destinations: &mut [CpuDestination<'_>],
    scratch: &mut CpuDestination<'_>,
) -> Res<()> {
    if !device.is_cpu() {
        return Err("fusion: unsupported device".to_string());
    }
    match dtype {
        DType::F32 => bridge_elementwise_into::<f32>(
            program,
            inputs,
            strides,
            scalars,
            n,
            shape,
            destinations,
            scratch,
        ),
        DType::F64 => bridge_elementwise_into::<f64>(
            program,
            inputs,
            strides,
            scalars,
            n,
            shape,
            destinations,
            scratch,
        ),
        _ => Err(format!("fusion: unsupported dtype {dtype:?}")),
    }
}

/// Single-output form of [`run_elementwise_multi_into`]; the program must
/// have exactly one output.
#[allow(clippy::too_many_arguments)]
pub fn run_elementwise_into(
    program: &CpuFusionProgram,
    inputs: &[Value],
    strides: Option<&[Vec<usize>]>,
    scalars: &[Value],
    n: usize,
    shape: &[usize],
    dtype: DType,
    device: &Device,
    destination: &mut CpuDestination<'_>,
    scratch: &mut CpuDestination<'_>,
) -> Res<()> {
    if program.output_count() != 1 {
        return Err(format!(
            "fusion: single-output execution requires one plan, got {}",
            program.output_count()
        ));
    }
    run_elementwise_multi_into(
        program,
        inputs,
        strides,
        scalars,
        n,
        shape,
        dtype,
        device,
        std::slice::from_mut(destination),
        scratch,
    )
}

#[cfg(test)]
fn run(
    exprs: &[Expr],
    inputs: &[Value],
    strides: Option<&[Vec<usize>]>,
    scalars: &[Value],
    n: usize,
    shape: &[usize],
    dtype: DType,
    device: &Device,
) -> Res<Vec<Value>> {
    let program = prepare(exprs);
    let prepared_inputs = inputs
        .iter()
        .map(|value| Value(value.tensor().contiguous()))
        .collect::<Vec<_>>();
    let prepared_scalars = scalars
        .iter()
        .map(|value| Value(value.tensor().contiguous()))
        .collect::<Vec<_>>();
    let mut outputs = exprs
        .iter()
        .map(|_| Tensor::empty(shape, dtype))
        .collect::<Vec<_>>();
    let requirement = scratch_requirement(&program, dtype)?;
    let mut scratch = Tensor::empty(&requirement.shape, requirement.dtype);
    {
        let mut destinations = outputs
            .iter_mut()
            .map(Tensor::destination)
            .collect::<Result<Vec<_>, _>>()?;
        run_elementwise_multi_into(
            &program,
            &prepared_inputs,
            strides,
            &prepared_scalars,
            n,
            shape,
            dtype,
            device,
            &mut destinations,
            &mut scratch.destination()?,
        )?;
    }
    Ok(outputs.into_iter().map(Value).collect())
}

fn validate_reduce_shape(
    in_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    out_shape: &[usize],
) -> Res<()> {
    for (index, &dimension) in dims.iter().enumerate() {
        if dimension >= in_shape.len() {
            return Err(format!(
                "fusion reduce: dimension {dimension} is out of bounds for rank {}",
                in_shape.len()
            ));
        }
        if dims[..index].contains(&dimension) {
            return Err(format!(
                "fusion reduce: dimension {dimension} appears more than once"
            ));
        }
    }
    let expected_rank = if keepdims {
        in_shape.len()
    } else {
        in_shape.len() - dims.len()
    };
    if out_shape.len() != expected_rank {
        return Err(format!(
            "fusion reduce: output rank mismatch: expected {expected_rank}, got {}",
            out_shape.len()
        ));
    }
    let mut output_dimension = 0usize;
    for (dimension, &width) in in_shape.iter().enumerate() {
        if dims.contains(&dimension) {
            if keepdims {
                if out_shape[output_dimension] != 1 {
                    return Err(format!(
                        "fusion reduce: reduced output dimension {output_dimension} must be 1"
                    ));
                }
                output_dimension += 1;
            }
        } else {
            if out_shape[output_dimension] != width {
                return Err(format!(
                    "fusion reduce: output dimension {output_dimension} must be {width}, got {}",
                    out_shape[output_dimension]
                ));
            }
            output_dimension += 1;
        }
    }
    Ok(())
}

fn reduce_output_offset(
    index: usize,
    in_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    out_shape: &[usize],
) -> usize {
    let mut remainder = index;
    let mut output_dimension = out_shape.len();
    let mut output_stride = 1usize;
    let mut output_offset = 0usize;
    for dimension in (0..in_shape.len()).rev() {
        let width = in_shape[dimension].max(1);
        let coordinate = remainder % width;
        remainder /= width;
        if !dims.contains(&dimension) {
            output_dimension -= 1;
            output_offset += coordinate * output_stride;
            output_stride *= out_shape[output_dimension];
        } else if keepdims {
            output_dimension -= 1;
            output_stride *= out_shape[output_dimension];
        }
    }
    output_offset
}

fn reduce_init<T: Scalar>(op: ReduceOp) -> T {
    T::from_f64(match op {
        ReduceOp::Sum | ReduceOp::Mean => 0.0,
        ReduceOp::Prod => 1.0,
        ReduceOp::Max => f64::NEG_INFINITY,
        ReduceOp::Min => f64::INFINITY,
    })
}

fn reduce_fold<T: Scalar>(op: ReduceOp, accumulator: T, value: T) -> T {
    match op {
        ReduceOp::Sum | ReduceOp::Mean => accumulator.add(value),
        ReduceOp::Prod => accumulator.mul(value),
        ReduceOp::Max => accumulator.max(value),
        ReduceOp::Min => accumulator.min(value),
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_reduce<T: Elem>(
    program: &CpuFusionProgram,
    inputs: &[Value],
    strides: &[Vec<usize>],
    in_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    out_shape: &[usize],
    destination: &CpuDestination<'_>,
    scratch: &CpuDestination<'_>,
) -> Res<()> {
    if program.output_count() != 1 {
        return Err(format!(
            "fusion reduce: requires one plan, got {}",
            program.output_count()
        ));
    }
    if program.scalar_count() != 0 {
        return Err("fusion reduce: scalar lanes are unsupported".to_string());
    }
    if program.input_count() > inputs.len() {
        return Err(format!(
            "fusion reduce: program references {} input lanes, got {}",
            program.input_count(),
            inputs.len()
        ));
    }
    if strides.len() != inputs.len() {
        return Err(format!(
            "fusion: got {} stride entries for {} inputs",
            strides.len(),
            inputs.len()
        ));
    }
    if let Some((lane, strides)) = strides
        .iter()
        .enumerate()
        .find(|(_, strides)| strides.len() != in_shape.len())
    {
        return Err(format!(
            "fusion reduce: lane {lane} has {} strides for rank {}",
            strides.len(),
            in_shape.len()
        ));
    }
    validate_reduce_shape(in_shape, dims, keepdims, out_shape)?;
    let input_elements = checked_numel(in_shape, "fusion reduce")?;
    checked_numel(out_shape, "fusion reduce")?;
    for (lane, input) in inputs.iter().enumerate() {
        let values = native_slice::<T>(input.tensor())?;
        if input_elements == 0 {
            continue;
        }
        let last = max_strided_offset(in_shape, &strides[lane])
            .ok_or_else(|| "fusion reduce: input offset overflow".to_string())?;
        if last >= values.len() {
            return Err(format!(
                "fusion reduce: lane {lane} reads element {last}, but has {} elements",
                values.len()
            ));
        }
    }
    destination.validate_current::<T>("fusion reduce")?;
    if destination.shape() != out_shape {
        return Err(format!(
            "fusion reduce destination shape mismatch: expected {out_shape:?}, got {:?}",
            destination.shape()
        ));
    }
    validate_scratch::<T>(program, scratch)
}

#[allow(clippy::too_many_arguments)]
fn bridge_reduce_into<T: Scalar + Elem>(
    op: ReduceOp,
    program: &CpuFusionProgram,
    inputs: &[Value],
    strides: &[Vec<usize>],
    in_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    out_shape: &[usize],
    destination: &mut CpuDestination<'_>,
    scratch: &mut CpuDestination<'_>,
) -> Res<()> {
    validate_reduce::<T>(
        program,
        inputs,
        strides,
        in_shape,
        dims,
        keepdims,
        out_shape,
        destination,
        scratch,
    )?;
    let input_elements = checked_numel(in_shape, "fusion reduce")?;
    scratch.write_current::<T, _>("fusion scratch", |scratch_values| -> Res<()> {
        let (lane_values, values) = scratch_values.split_at_mut(program.input_count());
        destination.write::<T, _>("fusion reduce", out_shape, |output| {
            output.fill(reduce_init::<T>(op));
            for index in 0..input_elements {
                for lane in 0..program.input_count() {
                    let offset = strided_offset(index, in_shape, &strides[lane]);
                    lane_values[lane] = native_slice::<T>(inputs[lane].tensor())
                        .expect("fusion reduce inputs were validated")[offset];
                }
                let value = program.evaluate(
                    0,
                    |lane| lane_values[lane as usize],
                    |_| unreachable!("fused reduce scalar lane"),
                    values,
                );
                let output_index = reduce_output_offset(index, in_shape, dims, keepdims, out_shape);
                output[output_index] = reduce_fold(op, output[output_index], value);
            }
            if op == ReduceOp::Mean {
                let extent = dims
                    .iter()
                    .fold(1usize, |total, &dimension| total * in_shape[dimension]);
                let extent = <T as Scalar>::from_f64(extent as f64);
                for value in output {
                    *value = value.div(extent);
                }
            }
        })?;
        Ok(())
    })?
}

/// Executes a fused map-then-reduce: evaluates `program` at every input
/// element (addressed through per-lane `strides` over `in_shape`) and folds
/// the results with `op` into the `out_shape` destination. `Mean` divides by
/// the reduced extent after the fold. Scalar lanes are unsupported.
#[allow(clippy::too_many_arguments)]
pub fn run_reduce_into(
    op: ReduceOp,
    program: &CpuFusionProgram,
    inputs: &[Value],
    strides: &[Vec<usize>],
    in_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    out_shape: &[usize],
    dtype: DType,
    device: &Device,
    destination: &mut CpuDestination<'_>,
    scratch: &mut CpuDestination<'_>,
) -> Res<()> {
    if !device.is_cpu() {
        return Err("fusion: unsupported device".to_string());
    }
    match dtype {
        DType::F32 => bridge_reduce_into::<f32>(
            op,
            program,
            inputs,
            strides,
            in_shape,
            dims,
            keepdims,
            out_shape,
            destination,
            scratch,
        ),
        DType::F64 => bridge_reduce_into::<f64>(
            op,
            program,
            inputs,
            strides,
            in_shape,
            dims,
            keepdims,
            out_shape,
            destination,
            scratch,
        ),
        _ => Err(format!("fusion: unsupported dtype {dtype:?}")),
    }
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn run_reduce(
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
    let program = prepare(std::slice::from_ref(expr));
    let prepared_inputs = inputs
        .iter()
        .map(|value| Value(value.tensor().contiguous()))
        .collect::<Vec<_>>();
    let mut output = Tensor::empty(out_shape, dtype);
    let requirement = scratch_requirement(&program, dtype)?;
    let mut scratch = Tensor::empty(&requirement.shape, requirement.dtype);
    run_reduce_into(
        op,
        &program,
        &prepared_inputs,
        strides,
        in_shape,
        dims,
        keepdims,
        out_shape,
        dtype,
        device,
        &mut output.destination()?,
        &mut scratch.destination()?,
    )?;
    Ok(Value(output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuBuffer, CpuSegment, ExecutableAllocationGuard};
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use std::sync::Arc;

    struct CountingAllocator;

    thread_local! {
        static COUNT_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
        static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
    }

    fn record_allocation() {
        let enabled = COUNT_ALLOCATIONS.try_with(Cell::get).unwrap_or(false);
        if enabled {
            let _ = ALLOCATION_COUNT.try_with(|count| count.set(count.get() + 1));
        }
    }

    // Test-only allocator that counts allocations to prove planned execution
    // is allocation-free.
    // SAFETY: every method forwards to `System` with the unchanged layout and
    // arguments supplied by the global allocator protocol; the only added
    // behavior is a thread-local counter increment.
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            record_allocation();
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            record_allocation();
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            unsafe { System.dealloc(pointer, layout) }
        }

        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            record_allocation();
            unsafe { System.realloc(pointer, layout, new_size) }
        }
    }

    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    struct AllocationCountGuard;

    impl Drop for AllocationCountGuard {
        fn drop(&mut self) {
            COUNT_ALLOCATIONS.with(|enabled| enabled.set(false));
        }
    }

    fn allocations_during(run: impl FnOnce()) -> usize {
        ALLOCATION_COUNT.with(|count| count.set(0));
        COUNT_ALLOCATIONS.with(|enabled| enabled.set(true));
        let guard = AllocationCountGuard;
        run();
        drop(guard);
        ALLOCATION_COUNT.with(Cell::get)
    }

    fn f32_values(tensor: &Tensor) -> &[f32] {
        let CpuBuffer::F32(values) = &tensor.buffer else {
            panic!("expected f32 tensor")
        };
        values.as_slice()
    }

    fn f64_values(tensor: &Tensor) -> &[f64] {
        let CpuBuffer::F64(values) = &tensor.buffer else {
            panic!("expected f64 tensor")
        };
        values.as_slice()
    }

    #[test]
    fn program_reports_exact_native_scratch() {
        let exprs = [
            Expr::Add(
                Box::new(Expr::Input(0)),
                Box::new(Expr::Mul(
                    Box::new(Expr::Input(1)),
                    Box::new(Expr::cst(2.0)),
                )),
            ),
            Expr::Add(Box::new(Expr::Input(0)), Box::new(Expr::Scalar(0))),
        ];
        let program = prepare(&exprs);
        assert_eq!(program.output_count(), 2);
        assert_eq!(program.input_count(), 2);
        assert_eq!(program.scalar_count(), 1);
        assert_eq!(program.value_scratch_len(), 3);
        assert_eq!(program.scratch_elements(), 5);
        let f32_requirement = scratch_requirement(&program, DType::F32).unwrap();
        assert_eq!(f32_requirement.shape, vec![5]);
        assert_eq!(f32_requirement.bytes, 20);
        assert_eq!(f32_requirement.dtype, DType::F32);
        let f64_requirement = scratch_requirement(&program, DType::F64).unwrap();
        assert_eq!(f64_requirement.shape, vec![5]);
        assert_eq!(f64_requirement.bytes, 40);
    }

    #[test]
    fn planned_multi_output_into_is_allocation_free_and_matches_wrapper() {
        let a = Value(Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        ));
        let b = Value(Tensor::from_vec(vec![10.0f32, 20.0, 30.0], vec![1, 3]));
        let scalar = Value(Tensor::from_vec(vec![0.5f32], vec![]));
        let exprs = [
            Expr::Add(Box::new(Expr::Input(0)), Box::new(Expr::Input(1))),
            Expr::Mul(Box::new(Expr::Input(0)), Box::new(Expr::Scalar(0))),
        ];
        let strides = [vec![3, 1], vec![0, 1]];
        let wrapped = run(
            &exprs,
            &[a.clone(), b.clone()],
            Some(&strides),
            std::slice::from_ref(&scalar),
            6,
            &[2, 3],
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let program = prepare(&exprs);
        let requirement = scratch_requirement(&program, DType::F32).unwrap();
        let segment = CpuSegment::allocate(192, 64).unwrap();
        let first = Tensor::from_segment(Arc::clone(&segment), 0, vec![2, 3], DType::F32).unwrap();
        let second =
            Tensor::from_segment(Arc::clone(&segment), 64, vec![2, 3], DType::F32).unwrap();
        let scratch =
            Tensor::from_segment(segment, 128, requirement.shape, requirement.dtype).unwrap();
        // SAFETY: the fixed test schedule assigns disjoint planned ranges.
        let first_destination = unsafe { CpuDestination::from_planned(&first) };
        // SAFETY: the fixed test schedule assigns disjoint planned ranges.
        let second_destination = unsafe { CpuDestination::from_planned(&second) };
        // SAFETY: the fixed test schedule assigns disjoint planned ranges.
        let mut scratch_destination = unsafe { CpuDestination::from_planned(&scratch) };
        let mut destinations = [first_destination, second_destination];
        let allocations = allocations_during(|| {
            let _guard = ExecutableAllocationGuard::enter();
            run_elementwise_multi_into(
                &program,
                &[a, b],
                Some(&strides),
                std::slice::from_ref(&scalar),
                6,
                &[2, 3],
                DType::F32,
                &Device::Cpu,
                &mut destinations,
                &mut scratch_destination,
            )
            .unwrap();
        });
        assert_eq!(allocations, 0);
        assert_eq!(f32_values(&first), f32_values(wrapped[0].tensor()));
        assert_eq!(f32_values(&second), f32_values(wrapped[1].tensor()));
    }

    #[test]
    fn single_output_into_matches_multi_output_path() {
        let input = Value(Tensor::from_vec(vec![1.0f32, -2.0, 3.0], vec![3]));
        let program = prepare(&[Expr::Neg(Box::new(Expr::Input(0)))]);
        let requirement = scratch_requirement(&program, DType::F32).unwrap();
        let mut output = Tensor::empty(&[3], DType::F32);
        let mut scratch = Tensor::empty(&requirement.shape, requirement.dtype);
        let allocations = allocations_during(|| {
            let _guard = ExecutableAllocationGuard::enter();
            run_elementwise_into(
                &program,
                std::slice::from_ref(&input),
                None,
                &[],
                3,
                &[3],
                DType::F32,
                &Device::Cpu,
                &mut output.destination().unwrap(),
                &mut scratch.destination().unwrap(),
            )
            .unwrap();
        });
        assert_eq!(allocations, 0);
        assert_eq!(f32_values(&output), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn planned_reduce_into_is_allocation_free_and_matches_wrapper() {
        let input = Value(Tensor::from_vec(
            vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        ));
        let expr = Expr::Mul(Box::new(Expr::Input(0)), Box::new(Expr::Input(0)));
        let strides = [vec![3, 1]];
        let wrapped = run_reduce(
            ReduceOp::Mean,
            &expr,
            std::slice::from_ref(&input),
            &strides,
            &[2, 3],
            &[1],
            false,
            &[2],
            DType::F64,
            &Device::Cpu,
        )
        .unwrap();
        let program = prepare(std::slice::from_ref(&expr));
        let requirement = scratch_requirement(&program, DType::F64).unwrap();
        assert_eq!(requirement.shape, vec![3]);
        assert_eq!(requirement.bytes, 24);
        let segment = CpuSegment::allocate(128, 64).unwrap();
        let output = Tensor::from_segment(Arc::clone(&segment), 0, vec![2], DType::F64).unwrap();
        let scratch =
            Tensor::from_segment(segment, 64, requirement.shape, requirement.dtype).unwrap();
        // SAFETY: the fixed test schedule assigns disjoint planned ranges.
        let mut destination = unsafe { CpuDestination::from_planned(&output) };
        // SAFETY: the fixed test schedule assigns disjoint planned ranges.
        let mut scratch_destination = unsafe { CpuDestination::from_planned(&scratch) };
        let allocations = allocations_during(|| {
            let _guard = ExecutableAllocationGuard::enter();
            run_reduce_into(
                ReduceOp::Mean,
                &program,
                std::slice::from_ref(&input),
                &strides,
                &[2, 3],
                &[1],
                false,
                &[2],
                DType::F64,
                &Device::Cpu,
                &mut destination,
                &mut scratch_destination,
            )
            .unwrap();
        });
        assert_eq!(allocations, 0);
        assert_eq!(f64_values(&output), f64_values(wrapped.tensor()));
        assert_eq!(f64_values(&output), &[14.0 / 3.0, 77.0 / 3.0]);
    }
}

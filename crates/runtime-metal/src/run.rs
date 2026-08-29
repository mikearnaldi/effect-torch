//! [`MetalTensor`] and the fused elementwise/reduction runners.
//!
//! A `MetalTensor` is a buffer view with a logical layout and dtype. The
//! `emit` module synthesizes elementwise and reduction kernels for each
//! expression. It compiles the fusion IR to MSL once and caches it under a
//! structural hash. Runners only bind buffers and dispatch.
//!
//! # API shape
//!
//! Every operation comes in three layers:
//!
//! - `compile_*` and `warm_*` emit and cache a pipeline for an expression
//!   and layout without allocating or dispatching.
//! - `run_*` allocates destination tensors and dispatches. It compiles on a
//!   cache miss.
//! - `run_*_into` dispatches into caller-owned destinations. The pipeline
//!   must be precompiled, and the function does not allocate, so executable
//!   dispatch can call it.
//!
//! The `*_prekeyed` variants take a precomputed pipeline key so hot paths
//! skip both expression hashing and source emission.
//!
//! # Dispatch conventions
//!
//! Grids are flat: one thread per output element, padded to the
//! [`emit::BLOCK`] threadgroup size, widening to a 2-D grid with 64-bit
//! indexing past `u32::MAX` elements. Buffer binding order is inputs,
//! then the packed scalar buffer (if any), then outputs. All inputs must
//! share one dtype (f32 or bf16) and all destinations must be contiguous.

use super::device::{set_buffer, MetalDevice};
use super::emit;
use crate::fusion::{Expr, ReduceOp};
use crate::runtime::dtype::DType;
use objc2_metal::MTLComputeCommandEncoder;
use std::sync::Arc;

/// Shared Metal buffer storage with a logical layout and dtype. Cloned views
/// share the physical allocation.
#[derive(Clone)]
pub struct MetalTensor {
    pub buffer: Arc<super::device::Buffer>,
    pub layout: crate::runtime::layout::Layout,
    pub dtype: DType,
}

impl MetalTensor {
    /// Uploads f32 host data as a contiguous tensor.
    pub fn from_f32(dev: &MetalDevice, data: Vec<f32>, shape: Vec<usize>) -> Self {
        MetalTensor {
            buffer: dev.alloc_with_data(&data),
            layout: crate::runtime::layout::Layout::contiguous(shape),
            dtype: DType::F32,
        }
    }

    /// Allocates a contiguous tensor and asynchronously fills it with
    /// zeros on the current submission stream.
    pub fn zeros(dev: &MetalDevice, shape: Vec<usize>, dtype: DType) -> Self {
        let n: usize = shape.iter().product();
        let buffer = dev.alloc(n.max(1), dtype);
        let out = MetalTensor {
            buffer,
            layout: crate::runtime::layout::Layout::contiguous(shape),
            dtype,
        };
        if n > 0 {
            // Async fill on the caller's submission: ordered before any later
            // consumer in that context, with no host fence required.
            let _ = super::kernels::fill(dev, &out, 0.0);
        }
        out
    }

    // Alloc without the zero-fill dispatch: only for outputs the kernel
    // provably overwrites in full.
    pub fn empty(dev: &MetalDevice, shape: Vec<usize>, dtype: DType) -> Self {
        let n: usize = shape.iter().product();
        let buffer = dev.alloc(n.max(1), dtype);
        MetalTensor {
            buffer,
            layout: crate::runtime::layout::Layout::contiguous(shape),
            dtype,
        }
    }

    /// Number of logical elements in the layout.
    pub fn numel(&self) -> usize {
        self.layout.numel()
    }

    /// Validates the destination's shape, dtype, contiguous layout, and byte
    /// range. It then registers the current stream as the next writer for
    /// cross-stream synchronization.
    pub(crate) fn validate_destination(
        &self,
        operation: &str,
        shape: &[usize],
        dtype: DType,
    ) -> Result<(), String> {
        if self.layout.shape() != shape || self.dtype != dtype {
            return Err(format!(
                "{operation} destination mismatch: expected {shape:?}:{dtype:?}, got {:?}:{:?}",
                self.layout.shape(),
                self.dtype
            ));
        }
        if !self.layout.is_contiguous() {
            return Err(format!(
                "{operation} destination must have contiguous layout, got {:?}",
                self.layout
            ));
        }
        let end = self
            .layout
            .offset()
            .checked_add(self.numel())
            .and_then(|elements| elements.checked_mul(dtype.size_in_bytes()))
            .ok_or_else(|| format!("{operation} destination byte range overflow"))?;
        if end > self.buffer.size {
            return Err(format!(
                "{operation} destination requires {end} bytes, buffer has {}",
                self.buffer.size
            ));
        }
        MetalDevice::get().mark_buffer_write(&self.buffer)?;
        Ok(())
    }

    /// Synchronizes the producer stream and reads the logical contents
    /// back to the host as f32.
    pub fn read_f32(&self) -> crate::err::Res<Vec<f32>> {
        crate::runtime::metal::device::MetalDevice::get().synchronize_buffer(&self.buffer)?;
        Ok(self.buffer.read_f32(self.layout.offset(), self.numel()))
    }

    /// Synchronizes the producer stream and reads the contents back as
    /// u32, widening from u8 or truncating from i64. Other dtypes are
    /// rejected.
    pub fn to_u32_vec(&self) -> crate::err::Res<Vec<u32>> {
        crate::runtime::metal::device::MetalDevice::get().synchronize_buffer(&self.buffer)?;
        let n = self.numel();
        let size = self.dtype.size_in_bytes();
        // SAFETY: synchronize_buffer above guarantees the GPU is done
        // writing; the buffer is shared-mode so contents_ptr is a valid
        // host mapping, and `offset * size + n * size` is within the
        // logical tensor, which validate/alloc guarantees fits the buffer.
        let ptr = unsafe {
            self.buffer
                .contents_ptr()
                .cast::<u8>()
                .add(self.layout.offset() * size)
        };
        // SAFETY: `ptr` is valid for `n * size` bytes per the argument
        // above.
        let bytes = unsafe { std::slice::from_raw_parts(ptr, n * size) };
        let mut out = Vec::with_capacity(n);
        match self.dtype {
            DType::U32 => out.extend(
                bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            ),
            DType::U8 => out.extend(bytes.iter().map(|&b| b as u32)),
            DType::I64 => out.extend(bytes.chunks_exact(8).map(|c| {
                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as u32
            })),
            _ => return crate::err::err("to_u32_vec: dtype must be u8/u32/i64"),
        }
        Ok(out)
    }
}

/// Structural pipeline cache key for a fused elementwise kernel: the
/// expressions, per-input lane strides, output shape, element count,
/// scalar count, and dtype.
pub fn elementwise_key(
    exprs: &[Expr],
    lane_strides: &[Vec<usize>],
    shape: &[usize],
    n: usize,
    num_scalars: usize,
    dtype: DType,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    exprs.hash(&mut hasher);
    lane_strides.hash(&mut hasher);
    shape.hash(&mut hasher);
    n.hash(&mut hasher);
    num_scalars.hash(&mut hasher);
    (dtype as u8).hash(&mut hasher);
    hasher.finish()
}

/// Structural pipeline cache key for a fused reduction kernel.
#[allow(clippy::too_many_arguments)]
pub fn reduce_key(
    op: ReduceOp,
    expr: &Expr,
    lane_strides: &[Vec<usize>],
    in_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    out_shape: &[usize],
    dtype: DType,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (op as u8).hash(&mut hasher);
    expr.hash(&mut hasher);
    lane_strides.hash(&mut hasher);
    in_shape.hash(&mut hasher);
    dims.hash(&mut hasher);
    keepdims.hash(&mut hasher);
    out_shape.hash(&mut hasher);
    (dtype as u8).hash(&mut hasher);
    hasher.finish()
}

/// Compiles the exact elementwise pipeline key without allocating buffers.
pub fn compile_elementwise(
    dev: &MetalDevice,
    exprs: &[Expr],
    lane_strides: &[Vec<usize>],
    shape: &[usize],
    n: usize,
    num_scalars: usize,
    dtype: DType,
) -> Result<(), String> {
    if n == 0 {
        return Ok(());
    }
    let key = elementwise_key(exprs, lane_strides, shape, n, num_scalars, dtype);
    if dev.pipeline_cached(key).is_none() {
        let source = emit::emit_elementwise(
            exprs,
            lane_strides,
            shape,
            n,
            num_scalars,
            "et_fused",
            dtype,
        );
        dev.compile_slow(key, &source, "et_fused")?;
    }
    Ok(())
}

/// [`compile_elementwise`] against the process-wide device.
pub fn warm_elementwise(
    dev: &MetalDevice,
    exprs: &[Expr],
    lane_strides: &[Vec<usize>],
    shape: &[usize],
    n: usize,
    num_scalars: usize,
    dtype: DType,
) -> Result<(), String> {
    compile_elementwise(dev, exprs, lane_strides, shape, n, num_scalars, dtype)
}

/// Compiles the exact fused-reduce pipeline key without allocating buffers.
#[allow(clippy::too_many_arguments)]
pub fn compile_reduce(
    dev: &MetalDevice,
    op: ReduceOp,
    expr: &Expr,
    lane_strides: &[Vec<usize>],
    in_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    out_shape: &[usize],
    dtype: DType,
) -> Result<(), String> {
    if out_shape.iter().product::<usize>() == 0 {
        return Ok(());
    }
    let key = reduce_key(
        op,
        expr,
        lane_strides,
        in_shape,
        dims,
        keepdims,
        out_shape,
        dtype,
    );
    if dev.pipeline_cached(key).is_none() {
        let source = emit::emit_reduce(
            op,
            expr,
            lane_strides,
            in_shape,
            dims,
            keepdims,
            out_shape,
            "et_fused_reduce",
            dtype,
        );
        dev.compile_slow(key, &source, "et_fused_reduce")?;
    }
    Ok(())
}

/// [`compile_reduce`] against the process-wide device.
#[allow(clippy::too_many_arguments)]
pub fn warm_reduce(
    dev: &MetalDevice,
    op: ReduceOp,
    expr: &Expr,
    lane_strides: &[Vec<usize>],
    in_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    out_shape: &[usize],
    dtype: DType,
) -> Result<(), String> {
    compile_reduce(
        dev,
        op,
        expr,
        lane_strides,
        in_shape,
        dims,
        keepdims,
        out_shape,
        dtype,
    )
}

/// Allocates one destination per expression and runs the fused
/// elementwise kernel, uploading `scalars` as the packed scalar buffer.
#[allow(clippy::too_many_arguments)]
pub fn run_elementwise(
    dev: &MetalDevice,
    exprs: &[Expr],
    inputs: &[&MetalTensor],
    lane_strides: &[Vec<usize>],
    scalars: &[f32],
    n: usize,
    shape: &[usize],
) -> Result<Vec<MetalTensor>, String> {
    let scalar_buf = if scalars.is_empty() {
        None
    } else {
        Some(dev.alloc_with_data(scalars))
    };
    let key = elementwise_key(
        exprs,
        lane_strides,
        shape,
        n,
        scalars.len(),
        inputs[0].dtype,
    );
    run_elementwise_prekeyed(
        dev,
        key,
        exprs,
        inputs,
        lane_strides,
        scalar_buf.as_deref(),
        scalars.len(),
        n,
        shape,
    )
}

// Same kernel as run_elementwise, but the packed scalar buffer is supplied
// directly because it is already device-resident. This needs no host readback.
/// Allocates destinations and dispatches the fused kernel with a
/// device-resident scalar buffer (no host upload).
#[allow(clippy::too_many_arguments)]
pub fn run_elementwise_scalar_buf(
    dev: &MetalDevice,
    exprs: &[Expr],
    inputs: &[&MetalTensor],
    lane_strides: &[Vec<usize>],
    scalar_buf: Option<&super::device::Buffer>,
    num_scalars: usize,
    n: usize,
    shape: &[usize],
) -> Result<Vec<MetalTensor>, String> {
    let key = elementwise_key(exprs, lane_strides, shape, n, num_scalars, inputs[0].dtype);
    run_elementwise_prekeyed(
        dev,
        key,
        exprs,
        inputs,
        lane_strides,
        scalar_buf,
        num_scalars,
        n,
        shape,
    )
}

// Fully prekeyed: no expr hashing, no source emission unless the pipeline
// cache actually misses.
/// Allocates destinations and dispatches the fused kernel for a
/// precomputed pipeline `key` (compiling only on an actual cache miss).
#[allow(clippy::too_many_arguments)]
pub fn run_elementwise_prekeyed(
    dev: &MetalDevice,
    key: u64,
    exprs: &[Expr],
    inputs: &[&MetalTensor],
    lane_strides: &[Vec<usize>],
    scalar_buf: Option<&super::device::Buffer>,
    num_scalars: usize,
    n: usize,
    shape: &[usize],
) -> Result<Vec<MetalTensor>, String> {
    let dtype = inputs[0].dtype;
    if n != 0 && dev.pipeline_cached(key).is_none() {
        let source = emit::emit_elementwise(
            exprs,
            lane_strides,
            shape,
            n,
            num_scalars,
            "et_fused",
            dtype,
        );
        dev.compile_slow(key, &source, "et_fused")?;
    }
    let mut out_bufs = Vec::with_capacity(exprs.len());
    for _ in 0..exprs.len() {
        out_bufs.push(MetalTensor::empty(dev, shape.to_vec(), dtype));
    }
    let output_refs = out_bufs.iter().collect::<Vec<_>>();
    run_elementwise_into_prekeyed(
        dev,
        key,
        exprs,
        inputs,
        lane_strides,
        scalar_buf,
        num_scalars,
        n,
        shape,
        &output_refs,
    )?;
    Ok(out_bufs)
}

/// Dispatches the fused elementwise kernel into caller-provided
/// destinations, computing the pipeline key from the arguments. The exact
/// pipeline must be precompiled; no allocation happens here.
#[allow(clippy::too_many_arguments)]
pub fn run_elementwise_into(
    dev: &MetalDevice,
    exprs: &[Expr],
    inputs: &[&MetalTensor],
    lane_strides: &[Vec<usize>],
    scalar_buf: Option<&super::device::Buffer>,
    num_scalars: usize,
    n: usize,
    shape: &[usize],
    outputs: &[&MetalTensor],
) -> Result<(), String> {
    let dtype = inputs
        .first()
        .ok_or_else(|| "elementwise requires at least one input".to_string())?
        .dtype;
    let key = elementwise_key(exprs, lane_strides, shape, n, num_scalars, dtype);
    run_elementwise_into_prekeyed(
        dev,
        key,
        exprs,
        inputs,
        lane_strides,
        scalar_buf,
        num_scalars,
        n,
        shape,
        outputs,
    )
}

/// [`run_elementwise_into`] with a precomputed pipeline key. Validates
/// the input/stride/expression/destination counts and every destination,
/// then binds inputs, scalars, and outputs in declaration order and
/// dispatches one thread per (padded) output element.
#[allow(clippy::too_many_arguments)]
pub fn run_elementwise_into_prekeyed(
    dev: &MetalDevice,
    key: u64,
    exprs: &[Expr],
    inputs: &[&MetalTensor],
    lane_strides: &[Vec<usize>],
    scalar_buf: Option<&super::device::Buffer>,
    num_scalars: usize,
    n: usize,
    shape: &[usize],
    outputs: &[&MetalTensor],
) -> Result<(), String> {
    let dtype = inputs
        .first()
        .ok_or_else(|| "elementwise requires at least one input".to_string())?
        .dtype;
    if inputs.len() != lane_strides.len() {
        return Err(format!(
            "elementwise received {} inputs and {} stride entries",
            inputs.len(),
            lane_strides.len()
        ));
    }
    if exprs.len() != outputs.len() {
        return Err(format!(
            "elementwise received {} expressions and {} destinations",
            exprs.len(),
            outputs.len()
        ));
    }
    if shape.iter().product::<usize>() != n {
        return Err(format!(
            "elementwise shape {shape:?} has a different element count than {n}"
        ));
    }
    if num_scalars == 0 && scalar_buf.is_some() || num_scalars != 0 && scalar_buf.is_none() {
        return Err(format!(
            "elementwise scalar buffer mismatch for {num_scalars} scalar(s)"
        ));
    }
    for (index, input) in inputs.iter().enumerate() {
        if input.dtype != dtype {
            return Err(format!(
                "elementwise input {index} has dtype {:?}, expected {dtype:?}",
                input.dtype
            ));
        }
        if lane_strides[index].len() != shape.len() {
            return Err(format!(
                "elementwise input {index} has rank-{} strides for rank-{} output",
                lane_strides[index].len(),
                shape.len()
            ));
        }
    }
    for output in outputs {
        output.validate_destination("elementwise", shape, dtype)?;
    }
    if n == 0 {
        return Ok(());
    }
    let pipeline = dev.pipeline_cached(key).ok_or_else(|| {
        format!("elementwise pipeline {key:#x} was not precompiled for the exact expression")
    })?;
    let padded = n.div_ceil(emit::BLOCK) * emit::BLOCK;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        let mut idx = 0usize;
        for t in inputs {
            set_buffer(
                e,
                idx,
                &t.buffer,
                t.layout.offset() * t.dtype.size_in_bytes(),
            );
            idx += 1;
        }
        if let Some(buf) = scalar_buf {
            set_buffer(e, idx, buf, 0);
            idx += 1;
        }
        for t in outputs {
            set_buffer(
                e,
                idx,
                &t.buffer,
                t.layout.offset() * t.dtype.size_in_bytes(),
            );
            idx += 1;
        }
        e.dispatchThreads_threadsPerThreadgroup(
            MetalDevice::grid_flat(padded).0,
            MetalDevice::grid_flat(padded).1,
        );
    });
    Ok(())
}

/// Allocates the reduction output and dispatches the fused reduce kernel
/// (compiling on a cache miss).
#[allow(clippy::too_many_arguments)]
pub fn run_reduce(
    dev: &MetalDevice,
    op: ReduceOp,
    expr: &Expr,
    inputs: &[&MetalTensor],
    lane_strides: &[Vec<usize>],
    in_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    out_shape: &[usize],
) -> Result<MetalTensor, String> {
    let dtype = inputs[0].dtype;
    compile_reduce(
        dev,
        op,
        expr,
        lane_strides,
        in_shape,
        dims,
        keepdims,
        out_shape,
        dtype,
    )?;
    let out = MetalTensor::empty(dev, out_shape.to_vec(), dtype);
    run_reduce_into(
        dev,
        op,
        expr,
        inputs,
        lane_strides,
        in_shape,
        dims,
        keepdims,
        out_shape,
        &out,
    )?;
    Ok(out)
}

/// Dispatches the fused reduce kernel into a caller-provided contiguous
/// destination. The exact pipeline must be precompiled via
/// [`compile_reduce`]; one thread per output element loops over the
/// (flattened) reduced dimensions.
#[allow(clippy::too_many_arguments)]
pub fn run_reduce_into(
    dev: &MetalDevice,
    op: ReduceOp,
    expr: &Expr,
    inputs: &[&MetalTensor],
    lane_strides: &[Vec<usize>],
    in_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    out_shape: &[usize],
    out: &MetalTensor,
) -> Result<(), String> {
    let dtype = inputs
        .first()
        .ok_or_else(|| "reduce requires at least one input".to_string())?
        .dtype;
    if inputs.len() != lane_strides.len() {
        return Err(format!(
            "reduce received {} inputs and {} stride entries",
            inputs.len(),
            lane_strides.len()
        ));
    }
    for (index, input) in inputs.iter().enumerate() {
        if input.dtype != dtype {
            return Err(format!(
                "reduce input {index} has dtype {:?}, expected {dtype:?}",
                input.dtype
            ));
        }
        if lane_strides[index].len() != in_shape.len() {
            return Err(format!(
                "reduce input {index} has rank-{} strides for rank-{} input",
                lane_strides[index].len(),
                in_shape.len()
            ));
        }
    }
    out.validate_destination("reduce", out_shape, dtype)?;
    let out_n: usize = out_shape.iter().product();
    if out_n == 0 {
        return Ok(());
    }
    let key = reduce_key(
        op,
        expr,
        lane_strides,
        in_shape,
        dims,
        keepdims,
        out_shape,
        dtype,
    );
    let pipeline = dev.pipeline_cached(key).ok_or_else(|| {
        format!("reduce pipeline {key:#x} was not precompiled for the exact expression")
    })?;
    let padded = out_n.div_ceil(emit::BLOCK) * emit::BLOCK;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        let mut idx = 0usize;
        for t in inputs {
            set_buffer(
                e,
                idx,
                &t.buffer,
                t.layout.offset() * t.dtype.size_in_bytes(),
            );
            idx += 1;
        }
        set_buffer(
            e,
            idx,
            &out.buffer,
            out.layout.offset() * out.dtype.size_in_bytes(),
        );
        e.dispatchThreads_threadsPerThreadgroup(
            MetalDevice::grid_flat(padded).0,
            MetalDevice::grid_flat(padded).1,
        );
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::{interpret_core, interpret_reduce_core};

    fn dev() -> &'static MetalDevice {
        MetalDevice::get()
    }

    #[test]
    fn elementwise_matches_interpreter() {
        let dev = dev();
        let a: Vec<f32> = (0..24).map(|i| (i as f32) * 0.25 - 3.0).collect();
        let b: Vec<f32> = (0..24).map(|i| (i as f32) * 0.125 + 0.5).collect();
        let ta = MetalTensor::from_f32(dev, a.clone(), vec![4, 6]);
        let tb = MetalTensor::from_f32(dev, b.clone(), vec![4, 6]);
        let exprs = vec![
            Expr::Add(
                Box::new(Expr::Input(0)),
                Box::new(Expr::Mul(
                    Box::new(Expr::Input(1)),
                    Box::new(Expr::Const(2.0f64.to_bits())),
                )),
            ),
            Expr::Tanh(Box::new(Expr::Input(0))),
            Expr::Max(
                Box::new(Expr::Sqrt(Box::new(Expr::Abs(Box::new(Expr::Input(1)))))),
                Box::new(Expr::Const(0.25f64.to_bits())),
            ),
        ];
        let outs = run_elementwise(
            dev,
            &exprs,
            &[&ta, &tb],
            &[vec![6, 1], vec![6, 1]],
            &[],
            24,
            &[4, 6],
        )
        .unwrap();
        let expected = interpret_core::<f32>(&exprs, &[&a, &b], None, &[], 24, &[4, 6]);
        for (got, want) in outs.iter().zip(&expected) {
            let g = got.read_f32().unwrap();
            for (x, y) in g.iter().zip(want) {
                assert!((x - y).abs() < 1e-5, "{x} vs {y}");
            }
        }
    }

    #[test]
    fn broadcast_lane_matches_interpreter() {
        let dev = dev();
        let a: Vec<f32> = (0..6).map(|i| i as f32 + 1.0).collect();
        let b: Vec<f32> = vec![10.0, 20.0];
        let ta = MetalTensor::from_f32(dev, a.clone(), vec![2, 3]);
        let tb = MetalTensor::from_f32(dev, b.clone(), vec![2, 1]);
        let exprs = vec![Expr::Mul(
            Box::new(Expr::Input(0)),
            Box::new(Expr::Input(1)),
        )];
        let strides = vec![vec![3, 1], vec![1, 0]];
        let outs = run_elementwise(dev, &exprs, &[&ta, &tb], &strides, &[], 6, &[2, 3]).unwrap();
        let expected = interpret_core::<f32>(&exprs, &[&a, &b], Some(&strides), &[], 6, &[2, 3]);
        let g = outs[0].read_f32().unwrap();
        for (x, y) in g.iter().zip(&expected[0]) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
        assert_eq!(g, vec![10.0, 20.0, 30.0, 80.0, 100.0, 120.0]);
    }

    #[test]
    fn reduce_matches_interpreter() {
        let dev = dev();
        let a: Vec<f32> = (0..24).map(|i| (i as f32) * 0.5 - 2.0).collect();
        let ta = MetalTensor::from_f32(dev, a.clone(), vec![4, 6]);
        let expr = Expr::Mul(Box::new(Expr::Input(0)), Box::new(Expr::Input(0)));
        let out = run_reduce(
            dev,
            ReduceOp::Sum,
            &expr,
            &[&ta],
            &[vec![6, 1]],
            &[4, 6],
            &[1],
            false,
            &[4],
        )
        .unwrap();
        let want = interpret_reduce_core::<f32>(
            ReduceOp::Sum,
            &expr,
            &[&a],
            &[vec![6, 1]],
            &[4, 6],
            &[1],
            false,
            &[4],
        );
        let g = out.read_f32().unwrap();
        for (x, y) in g.iter().zip(&want) {
            assert!((x - y).abs() / y.abs().max(1.0) < 1e-4, "{x} vs {y}");
        }
    }

    #[test]
    fn fused_wrappers_match_into_destinations() {
        let dev = MetalDevice::new(0).unwrap();
        let a = MetalTensor::from_f32(&dev, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = MetalTensor::from_f32(&dev, vec![10.0, 20.0, 30.0], vec![1, 3]);
        let exprs = [Expr::Add(
            Box::new(Expr::Input(0)),
            Box::new(Expr::Input(1)),
        )];
        let strides = [vec![3, 1], vec![0, 1]];
        let wrapped = run_elementwise(&dev, &exprs, &[&a, &b], &strides, &[], 6, &[2, 3]).unwrap();
        let destination = MetalTensor::empty(&dev, vec![2, 3], DType::F32);
        run_elementwise_into(
            &dev,
            &exprs,
            &[&a, &b],
            &strides,
            None,
            0,
            6,
            &[2, 3],
            &[&destination],
        )
        .unwrap();

        let reduce_wrapped = run_reduce(
            &dev,
            ReduceOp::Sum,
            &Expr::Input(0),
            &[&a],
            &[vec![3, 1]],
            &[2, 3],
            &[1],
            false,
            &[2],
        )
        .unwrap();
        let reduce_destination = MetalTensor::empty(&dev, vec![2], DType::F32);
        run_reduce_into(
            &dev,
            ReduceOp::Sum,
            &Expr::Input(0),
            &[&a],
            &[vec![3, 1]],
            &[2, 3],
            &[1],
            false,
            &[2],
            &reduce_destination,
        )
        .unwrap();

        dev.synchronize().unwrap();
        assert_eq!(
            wrapped[0].buffer.read_f32(0, 6),
            destination.buffer.read_f32(0, 6)
        );
        assert_eq!(
            reduce_wrapped.buffer.read_f32(0, 2),
            reduce_destination.buffer.read_f32(0, 2)
        );
    }

    #[test]
    fn fused_into_paths_use_no_planned_allocations_or_uploads() {
        let dev = MetalDevice::new(0).unwrap();
        let input = MetalTensor::from_f32(&dev, vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let elementwise_output = MetalTensor::empty(&dev, vec![2, 2], DType::F32);
        let reduce_output = MetalTensor::empty(&dev, vec![2], DType::F32);
        let expr = Expr::Mul(Box::new(Expr::Input(0)), Box::new(Expr::Input(0)));
        compile_elementwise(
            &dev,
            std::slice::from_ref(&expr),
            &[vec![2, 1]],
            &[2, 2],
            4,
            0,
            DType::F32,
        )
        .unwrap();
        compile_reduce(
            &dev,
            ReduceOp::Sum,
            &Expr::Input(0),
            &[vec![2, 1]],
            &[2, 2],
            &[1],
            false,
            &[2],
            DType::F32,
        )
        .unwrap();

        let _dispatch_guard = dev.begin_executable_dispatch().unwrap();
        run_elementwise_into(
            &dev,
            std::slice::from_ref(&expr),
            &[&input],
            &[vec![2, 1]],
            None,
            0,
            4,
            &[2, 2],
            &[&elementwise_output],
        )
        .unwrap();
        run_reduce_into(
            &dev,
            ReduceOp::Sum,
            &Expr::Input(0),
            &[&input],
            &[vec![2, 1]],
            &[2, 2],
            &[1],
            false,
            &[2],
            &reduce_output,
        )
        .unwrap();
        let result = dev.synchronize();
        result.unwrap();
        assert_eq!(reduce_output.buffer.read_f32(0, 2), vec![3.0, 7.0]);
    }

    #[test]
    fn fused_into_requires_the_exact_precompiled_expression() {
        let dev = MetalDevice::new(0).unwrap();
        let input = MetalTensor::from_f32(&dev, vec![1.0, -2.0, 3.0, -4.0], vec![4]);
        let output = MetalTensor::empty(&dev, vec![4], DType::F32);
        let expression = Expr::Abs(Box::new(Expr::Input(0)));
        let error = run_elementwise_into(
            &dev,
            std::slice::from_ref(&expression),
            &[&input],
            &[vec![1]],
            None,
            0,
            4,
            &[4],
            &[&output],
        )
        .unwrap_err();
        assert!(error.contains("not precompiled"), "{error}");

        compile_elementwise(
            &dev,
            &[Expr::Neg(Box::new(Expr::Input(0)))],
            &[vec![1]],
            &[4],
            4,
            0,
            DType::F32,
        )
        .unwrap();
        let error = run_elementwise_into(
            &dev,
            std::slice::from_ref(&expression),
            &[&input],
            &[vec![1]],
            None,
            0,
            4,
            &[4],
            &[&output],
        )
        .unwrap_err();
        assert!(error.contains("not precompiled"), "{error}");

        compile_elementwise(
            &dev,
            std::slice::from_ref(&expression),
            &[vec![1]],
            &[4],
            4,
            0,
            DType::F32,
        )
        .unwrap();
        run_elementwise_into(
            &dev,
            std::slice::from_ref(&expression),
            &[&input],
            &[vec![1]],
            None,
            0,
            4,
            &[4],
            &[&output],
        )
        .unwrap();
        dev.synchronize().unwrap();
        assert_eq!(output.buffer.read_f32(0, 4), vec![1.0, 2.0, 3.0, 4.0]);
    }
}

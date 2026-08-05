use super::device::{set_buffer, MetalDevice};
use objc2_metal::MTLComputeCommandEncoder;
use super::emit;
use crate::fusion::{Expr, ReduceOp};
use crate::runtime::dtype::DType;
use std::sync::Arc;

#[derive(Clone)]
pub struct MetalTensor {
    pub buffer: Arc<super::device::Buffer>,
    pub layout: crate::runtime::layout::Layout,
    pub dtype: DType,
}

impl MetalTensor {
    pub fn from_f32(dev: &MetalDevice, data: Vec<f32>, shape: Vec<usize>) -> Self {
        MetalTensor {
            buffer: dev.alloc_with_data(&data),
            layout: crate::runtime::layout::Layout::contiguous(shape),
            dtype: DType::F32,
        }
    }

    pub fn zeros(dev: &MetalDevice, shape: Vec<usize>, dtype: DType) -> Self {
        let n: usize = shape.iter().product();
        let buffer = dev.alloc(n.max(1), dtype);
        let out = MetalTensor {
            buffer,
            layout: crate::runtime::layout::Layout::contiguous(shape),
            dtype,
        };
        if n > 0 {
            // Async fill on the serial encoder: ordered before any consumer
            // dispatch, no host fence required.
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

    pub fn numel(&self) -> usize {
        self.layout.numel()
    }

    pub fn read_f32(&self) -> crate::err::Res<Vec<f32>> {
        crate::runtime::metal::device::MetalDevice::get().synchronize()?;
        Ok(self.buffer.read_f32(self.layout.offset(), self.numel()))
    }

    pub fn to_u32_vec(&self) -> crate::err::Res<Vec<u32>> {
        crate::runtime::metal::device::MetalDevice::get().synchronize()?;
        let n = self.numel();
        let size = self.dtype.size_in_bytes();
        let ptr = unsafe { self.buffer.contents_ptr().cast::<u8>().add(self.layout.offset() * size) };
        let bytes = unsafe { std::slice::from_raw_parts(ptr, n * size) };
        let mut out = Vec::with_capacity(n);
        match self.dtype {
            DType::U32 => out.extend(bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))),
            DType::U8 => out.extend(bytes.iter().map(|&b| b as u32)),
            DType::I64 => out.extend(bytes.chunks_exact(8).map(|c| {
                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as u32
            })),
            _ => return crate::err::err("to_u32_vec: dtype must be u8/u32/i64"),
        }
        Ok(out)
    }
}

fn hash_exprs(parts: &[u64]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    for p in parts {
        p.hash(&mut h);
    }
    h.finish()
}

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

pub fn run_elementwise(
    dev: &MetalDevice,
    exprs: &[Expr],
    inputs: &[&MetalTensor],
    lane_strides: &[Vec<usize>],
    scalars: &[f32],
    n: usize,
    shape: &[usize],
) -> Result<Vec<MetalTensor>, String> {
    let scalar_buf = if scalars.is_empty() { None } else { Some(dev.alloc_with_data(scalars)) };
    let key = elementwise_key(exprs, lane_strides, shape, n, scalars.len(), inputs[0].dtype);
    run_elementwise_prekeyed(dev, key, exprs, inputs, lane_strides, scalar_buf.as_deref(), scalars.len(), n, shape)
}

// Same kernel as run_elementwise, but the packed scalar buffer is supplied
// directly (already device-resident) — no host readback anywhere.
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
    run_elementwise_prekeyed(dev, key, exprs, inputs, lane_strides, scalar_buf, num_scalars, n, shape)
}

// Fully prekeyed: no expr hashing, no source emission unless the pipeline
// cache actually misses.
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
    let name = "et_fused";
    let pipeline = match dev.pipeline_cached(key) {
        Some(p) => p,
        None => {
            let src = emit::emit_elementwise(exprs, lane_strides, shape, n, num_scalars, name, dtype);
            dev.compile_slow(key, &src, name)?
        }
    };

    let mut out_bufs = Vec::with_capacity(exprs.len());
    for _ in 0..exprs.len() {
        out_bufs.push(MetalTensor::empty(dev, shape.to_vec(), dtype));
    }
    let padded = n.div_ceil(emit::BLOCK) * emit::BLOCK;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        let mut idx = 0usize;
        for t in inputs {
            set_buffer(e, idx, &t.buffer, t.layout.offset() * t.dtype.size_in_bytes());
            idx += 1;
        }
        if let Some(buf) = scalar_buf {
            set_buffer(e, idx, buf, 0);
            idx += 1;
        }
        for t in &out_bufs {
            set_buffer(e, idx, &t.buffer, 0);
            idx += 1;
        }
        e.dispatchThreads_threadsPerThreadgroup(
            MetalDevice::grid_flat(padded).0,
            MetalDevice::grid_flat(padded).1,
        );
    });
    Ok(out_bufs)
}

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
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (op as u8).hash(&mut hasher);
    expr.hash(&mut hasher);
    lane_strides.hash(&mut hasher);
    in_shape.hash(&mut hasher);
    dims.hash(&mut hasher);
    keepdims.hash(&mut hasher);
    out_shape.hash(&mut hasher);
    let dtype = inputs[0].dtype;
    (dtype as u8).hash(&mut hasher);
    let key = hasher.finish();
    let name = "et_fused_reduce";
    let pipeline = match dev.pipeline_cached(key) {
        Some(p) => p,
        None => {
            let src = emit::emit_reduce(op, expr, lane_strides, in_shape, dims, keepdims, out_shape, name, dtype);
            dev.compile_slow(key, &src, name)?
        }
    };

    let out_n: usize = out_shape.iter().product();
    let out = MetalTensor::empty(dev, out_shape.to_vec(), dtype);
    let padded = out_n.div_ceil(emit::BLOCK) * emit::BLOCK;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        let mut idx = 0usize;
        for t in inputs {
            set_buffer(e, idx, &t.buffer, t.layout.offset() * t.dtype.size_in_bytes());
            idx += 1;
        }
        set_buffer(e, idx, &out.buffer, 0);
        e.dispatchThreads_threadsPerThreadgroup(
            MetalDevice::grid_flat(padded).0,
            MetalDevice::grid_flat(padded).1,
        );
    });
    Ok(out)
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
            Expr::Add(Box::new(Expr::Input(0)), Box::new(Expr::Mul(Box::new(Expr::Input(1)), Box::new(Expr::Const(2.0f64.to_bits()))))),
            Expr::Tanh(Box::new(Expr::Input(0))),
            Expr::Max(
                Box::new(Expr::Sqrt(Box::new(Expr::Abs(Box::new(Expr::Input(1)))))),
                Box::new(Expr::Const(0.25f64.to_bits())),
            ),
        ];
        let outs = run_elementwise(dev, &exprs, &[&ta, &tb], &[vec![6, 1], vec![6, 1]], &[], 24, &[4, 6]).unwrap();
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
        let exprs = vec![Expr::Mul(Box::new(Expr::Input(0)), Box::new(Expr::Input(1)))];
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
        let out = run_reduce(dev, ReduceOp::Sum, &expr, &[&ta], &[vec![6, 1]], &[4, 6], &[1], false, &[4]).unwrap();
        let want = interpret_reduce_core::<f32>(ReduceOp::Sum, &expr, &[&a], &[vec![6, 1]], &[4, 6], &[1], false, &[4]);
        let g = out.read_f32().unwrap();
        for (x, y) in g.iter().zip(&want) {
            assert!((x - y).abs() / y.abs().max(1.0) < 1e-4, "{x} vs {y}");
        }
    }
}

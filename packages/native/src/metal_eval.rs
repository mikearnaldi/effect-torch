use crate::bridge;
use crate::fusion::{Expr, ReduceOp};
use crate::runtime::metal::device::MetalDevice;
use crate::runtime::metal::run::MetalTensor;
use crate::runtime::metal::{gemm, indexing, kernels};
use candle_core::{DType, Device, Tensor};

pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

fn bin_expr(op: &BinOp, a: Expr, b: Expr) -> Expr {
    match op {
        BinOp::Add => Expr::Add(Box::new(a), Box::new(b)),
        BinOp::Sub => Expr::Sub(Box::new(a), Box::new(b)),
        BinOp::Mul => Expr::Mul(Box::new(a), Box::new(b)),
        BinOp::Div => Expr::Div(Box::new(a), Box::new(b)),
        BinOp::Min => Expr::Min(Box::new(a), Box::new(b)),
        BinOp::Max => Expr::Max(Box::new(a), Box::new(b)),
        BinOp::Lt => Expr::Lt(Box::new(a), Box::new(b)),
        BinOp::Le => Expr::Le(Box::new(a), Box::new(b)),
        BinOp::Gt => Expr::Gt(Box::new(a), Box::new(b)),
        BinOp::Ge => Expr::Ge(Box::new(a), Box::new(b)),
        BinOp::Eq => Expr::Eq(Box::new(a), Box::new(b)),
        BinOp::Ne => Expr::Ne(Box::new(a), Box::new(b)),
    }
}

pub enum UnOp {
    Neg,
    Sqrt,
    Exp,
    Sin,
    Cos,
    Tanh,
    Abs,
    Log,
    Floor,
    Ceil,
    Round,
    Erf,
    Sign,
}

fn un_expr(op: &UnOp, a: Expr) -> Expr {
    let zero = || Expr::Const(0.0f64.to_bits());
    match op {
        UnOp::Neg => Expr::Neg(Box::new(a)),
        UnOp::Sqrt => Expr::Sqrt(Box::new(a)),
        UnOp::Exp => Expr::Exp(Box::new(a)),
        UnOp::Sin => Expr::Sin(Box::new(a)),
        UnOp::Cos => Expr::Cos(Box::new(a)),
        UnOp::Tanh => Expr::Tanh(Box::new(a)),
        UnOp::Abs => Expr::Abs(Box::new(a)),
        UnOp::Log => Expr::Log(Box::new(a)),
        UnOp::Floor => Expr::Floor(Box::new(a)),
        UnOp::Ceil => Expr::Ceil(Box::new(a)),
        UnOp::Round => Expr::Round(Box::new(a)),
        UnOp::Erf => Expr::Erf(Box::new(a)),
        UnOp::Sign => {
            let pos = Expr::Gt(Box::new(a.clone()), Box::new(zero()));
            let neg = Expr::Lt(Box::new(a), Box::new(zero()));
            Expr::Sub(
                Box::new(Expr::Select(Box::new(pos), Box::new(Expr::Const(1.0f64.to_bits())), Box::new(zero()))),
                Box::new(Expr::Select(Box::new(neg), Box::new(Expr::Const(1.0f64.to_bits())), Box::new(zero()))),
            )
        }
    }
}

fn mdev(t: &Tensor) -> candle_core::Result<candle_core::MetalDevice> {
    Ok(t.device().as_metal_device()?.clone())
}

fn wrap_contig(t: &Tensor) -> candle_core::Result<MetalTensor> {
    let wrapped = bridge::metal::wrap(t)?;
    if wrapped.layout.is_contiguous() {
        Ok(wrapped)
    } else {
        kernels::strided_copy(MetalDevice::get(), &wrapped).map_err(candle_core::Error::Msg)
    }
}

fn require_f32(t: &Tensor) -> candle_core::Result<()> {
    if t.dtype() != DType::F32 {
        return Err(candle_core::Error::Msg(
            "metal_eval: emitter is f32-only".to_string(),
        ));
    }
    Ok(())
}

pub fn compare(a: &Tensor, b: &Tensor, op: BinOp) -> candle_core::Result<Tensor> {
    let f = binary(a, b, op)?;
    cast(&f, crate::runtime::dtype::DType::U8)
}

pub fn binary(a: &Tensor, b: &Tensor, op: BinOp) -> candle_core::Result<Tensor> {
    require_f32(a)?;
    require_f32(b)?;
    a.device().synchronize()?;
    let shape = broadcast_shape_pub(a.shape().dims(), b.shape().dims())?;
    let an = wrap_contig(a)?;
    let bn = wrap_contig(b)?;
    let sa = bridge_strides(a, &shape)?;
    let sb = bridge_strides(b, &shape)?;
    let n: usize = shape.iter().product();
    let exprs = vec![bin_expr(&op, Expr::Input(0), Expr::Input(1))];
    let outs = crate::runtime::metal::run::run_elementwise(
        MetalDevice::get(),
        &exprs,
        &[&an, &bn],
        &[sa, sb],
        &[],
        n,
        &shape,
    )
    .map_err(candle_core::Error::Msg)?;
    bridge::metal::unwrap(&outs[0].buffer, shape, DType::F32, &mdev(a)?)
}

fn bridge_strides(t: &Tensor, shape: &[usize]) -> candle_core::Result<Vec<usize>> {
    let t_shape = t.shape().dims();
    let rank = shape.len();
    let extra = rank - t_shape.len();
    let mut out = vec![0usize; rank];
    let contig = crate::runtime::layout::Layout::contiguous(t_shape.to_vec());
    let cs = contig.strides().to_vec();
    for d in 0..t_shape.len() {
        let src = t_shape[d];
        let dst = shape[extra + d];
        if src == dst {
            out[extra + d] = cs[d];
        } else if src == 1 {
            out[extra + d] = 0;
        } else {
            return Err(candle_core::Error::Msg(format!(
                "broadcast: cannot map {t_shape:?} to {shape:?}"
            )));
        }
    }
    Ok(out)
}

fn broadcast_shape_pub(a: &[usize], b: &[usize]) -> candle_core::Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![1usize; rank];
    for d in 0..rank {
        let ad = if d < rank - a.len() { 1 } else { a[d - (rank - a.len())] };
        let bd = if d < rank - b.len() { 1 } else { b[d - (rank - b.len())] };
        if ad != bd && ad != 1 && bd != 1 {
            return Err(candle_core::Error::Msg(format!(
                "shape mismatch: {a:?} vs {b:?}"
            )));
        }
        out[d] = ad.max(bd);
    }
    Ok(out)
}

pub fn unary(a: &Tensor, op: UnOp) -> candle_core::Result<Tensor> {
    a.device().synchronize()?;
    let an = wrap_contig(a)?;
    let shape = a.shape().dims().to_vec();
    let n: usize = shape.iter().product();
    let contig = crate::runtime::layout::Layout::contiguous(shape.clone());
    let exprs = vec![un_expr(&op, Expr::Input(0))];
    let outs = crate::runtime::metal::run::run_elementwise(
        MetalDevice::get(),
        &exprs,
        &[&an],
        &[contig.strides().to_vec()],
        &[],
        n,
        &shape,
    )
    .map_err(candle_core::Error::Msg)?;
    bridge::metal::unwrap(&outs[0].buffer, shape, DType::F32, &mdev(a)?)
}

pub fn powf(a: &Tensor, e: f64) -> candle_core::Result<Tensor> {
    a.device().synchronize()?;
    let an = wrap_contig(a)?;
    let shape = a.shape().dims().to_vec();
    let n: usize = shape.iter().product();
    let contig = crate::runtime::layout::Layout::contiguous(shape.clone());
    let exprs = vec![Expr::Powf(Box::new(Expr::Input(0)), e.to_bits())];
    let outs = crate::runtime::metal::run::run_elementwise(
        MetalDevice::get(),
        &exprs,
        &[&an],
        &[contig.strides().to_vec()],
        &[],
        n,
        &shape,
    )
    .map_err(candle_core::Error::Msg)?;
    bridge::metal::unwrap(&outs[0].buffer, shape, DType::F32, &mdev(a)?)
}

pub fn reduce(a: &Tensor, dims: &[usize], keepdims: bool, op: ReduceOp) -> candle_core::Result<Tensor> {
    a.device().synchronize()?;
    let an = wrap_contig(a)?;
    let in_shape = a.shape().dims().to_vec();
    let out_shape: Vec<usize> = if keepdims {
        let mut s = in_shape.clone();
        for &d in dims {
            s[d] = 1;
        }
        s
    } else {
        in_shape
            .iter()
            .enumerate()
            .filter(|(d, _)| !dims.contains(d))
            .map(|(_, &s)| s)
            .collect()
    };
    let contig = crate::runtime::layout::Layout::contiguous(in_shape.clone());
    let out = crate::runtime::metal::run::run_reduce(
        MetalDevice::get(),
        op,
        &Expr::Input(0),
        &[&an],
        &[contig.strides().to_vec()],
        &in_shape,
        dims,
        keepdims,
        &out_shape,
    )
    .map_err(candle_core::Error::Msg)?;
    bridge::metal::unwrap(&out.buffer, out_shape, DType::F32, &mdev(a)?)
}

pub fn matmul(a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    a.device().synchronize()?;
    let an = wrap_contig(a)?;
    let bn = wrap_contig(b)?;
    let out = gemm::matmul(MetalDevice::get(), &an, &bn).map_err(candle_core::Error::Msg)?;
    bridge::metal::unwrap(&out.buffer, out.layout.shape().to_vec(), DType::F32, &mdev(a)?)
}

pub fn cast(a: &Tensor, dtype: crate::runtime::dtype::DType) -> candle_core::Result<Tensor> {
    a.device().synchronize()?;
    let an = wrap_contig(a)?;
    let out = kernels::cast(MetalDevice::get(), &an, dtype).map_err(candle_core::Error::Msg)?;
    bridge::metal::unwrap(
        &out.buffer,
        out.layout.shape().to_vec(),
        bridge::dtype_from_native(dtype),
        &mdev(a)?,
    )
}

pub fn index_select(a: &Tensor, dim: usize, ids: &Tensor) -> candle_core::Result<Tensor> {
    a.device().synchronize()?;
    let an = wrap_contig(a)?;
    let ids_vec = ids_u32(ids)?;
    let out = indexing::index_select(MetalDevice::get(), &an, dim, &ids_vec)
        .map_err(candle_core::Error::Msg)?;
    bridge::metal::unwrap(&out.buffer, out.layout.shape().to_vec(), a.dtype(), &mdev(a)?)
}

pub fn gather(a: &Tensor, dim: usize, ids: &Tensor) -> candle_core::Result<Tensor> {
    a.device().synchronize()?;
    let an = wrap_contig(a)?;
    let ids_vec = ids_u32(ids)?;
    let out = indexing::gather(MetalDevice::get(), &an, dim, &ids_vec, ids.shape().dims())
        .map_err(candle_core::Error::Msg)?;
    bridge::metal::unwrap(&out.buffer, out.layout.shape().to_vec(), a.dtype(), &mdev(a)?)
}

pub fn scatter_add(a: &Tensor, dim: usize, ids: &Tensor, src: &Tensor) -> candle_core::Result<Tensor> {
    a.device().synchronize()?;
    let an = wrap_contig(a)?;
    let sn = wrap_contig(src)?;
    let ids_vec = ids_u32(ids)?;
    let out = indexing::scatter_add(MetalDevice::get(), &an, dim, &ids_vec, &sn)
        .map_err(candle_core::Error::Msg)?;
    bridge::metal::unwrap(&out.buffer, out.layout.shape().to_vec(), a.dtype(), &mdev(a)?)
}

pub fn cat(a: &Tensor, b: &Tensor, dim: usize) -> candle_core::Result<Tensor> {
    a.device().synchronize()?;
    let an = wrap_contig(a)?;
    let bn = wrap_contig(b)?;
    let out = indexing::cat(MetalDevice::get(), &[&an, &bn], dim).map_err(candle_core::Error::Msg)?;
    bridge::metal::unwrap(&out.buffer, out.layout.shape().to_vec(), a.dtype(), &mdev(a)?)
}

fn ids_u32(ids: &Tensor) -> candle_core::Result<Vec<u32>> {
    let flat = ids.flatten_all()?;
    match ids.dtype() {
        DType::U32 => flat.to_vec1::<u32>(),
        DType::I64 => Ok(flat.to_vec1::<i64>()?.into_iter().map(|v| v as u32).collect()),
        DType::U8 => Ok(flat.to_vec1::<u8>()?.into_iter().map(|v| v as u32).collect()),
        d => Err(candle_core::Error::Msg(format!("indices must be u8/u32/i64, got {d:?}"))),
    }
}

pub fn fill(shape: &[usize], value: f64, dtype: crate::runtime::dtype::DType, device: &Device) -> candle_core::Result<Tensor> {
    let out = MetalTensor::zeros(MetalDevice::get(), shape.to_vec(), dtype);
    kernels::fill(MetalDevice::get(), &out, value).map_err(candle_core::Error::Msg)?;
    bridge::metal::unwrap(
        &out.buffer,
        shape.to_vec(),
        bridge::dtype_from_native(dtype),
        device.as_metal_device()?,
    )
}

pub fn arange(start: f64, end: f64, step: f64, dtype: crate::runtime::dtype::DType, device: &Device) -> candle_core::Result<Tensor> {
    let out = kernels::arange(MetalDevice::get(), start, end, step, dtype)
        .map_err(candle_core::Error::Msg)?;
    bridge::metal::unwrap(
        &out.buffer,
        out.layout.shape().to_vec(),
        bridge::dtype_from_native(dtype),
        device.as_metal_device()?,
    )
}

pub fn eye(n: usize, dtype: crate::runtime::dtype::DType, device: &Device) -> candle_core::Result<Tensor> {
    let out = kernels::eye(MetalDevice::get(), n, dtype).map_err(candle_core::Error::Msg)?;
    bridge::metal::unwrap(
        &out.buffer,
        vec![n, n],
        bridge::dtype_from_native(dtype),
        device.as_metal_device()?,
    )
}

pub fn randn(shape: &[usize], seed: u64, device: &Device) -> candle_core::Result<Tensor> {
    let out = kernels::randn(MetalDevice::get(), shape, seed).map_err(candle_core::Error::Msg)?;
    bridge::metal::unwrap(&out.buffer, shape.to_vec(), DType::F32, device.as_metal_device()?)
}

pub fn uniform(lo: f64, hi: f64, shape: &[usize], seed: u64, device: &Device) -> candle_core::Result<Tensor> {
    let out = kernels::uniform(MetalDevice::get(), lo, hi, shape, seed).map_err(candle_core::Error::Msg)?;
    bridge::metal::unwrap(&out.buffer, shape.to_vec(), DType::F32, device.as_metal_device()?)
}

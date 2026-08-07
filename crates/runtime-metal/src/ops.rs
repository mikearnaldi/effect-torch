use crate::fusion::{Expr, ReduceOp};
use crate::runtime::dtype::DType;
use crate::runtime::layout::Layout;
use crate::runtime::metal::device::MetalDevice;
use crate::runtime::metal::run::MetalTensor;
use crate::runtime::metal::{conv, gemm, indexing, kernels};

#[derive(Clone, Copy)]
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

#[derive(Clone, Copy)]
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
    Gelu,
    GeluTanh,
    Sign,
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
        UnOp::Gelu => Expr::Gelu(Box::new(a)),
        UnOp::GeluTanh => Expr::GeluTanh(Box::new(a)),
        UnOp::Sign => {
            let pos = Expr::Gt(Box::new(a.clone()), Box::new(zero()));
            let neg = Expr::Lt(Box::new(a), Box::new(zero()));
            Expr::Sub(
                Box::new(Expr::Select(
                    Box::new(pos),
                    Box::new(Expr::Const(1.0f64.to_bits())),
                    Box::new(zero()),
                )),
                Box::new(Expr::Select(
                    Box::new(neg),
                    Box::new(Expr::Const(1.0f64.to_bits())),
                    Box::new(zero()),
                )),
            )
        }
    }
}

fn contig(t: &MetalTensor) -> crate::err::Res<MetalTensor> {
    if t.layout.is_contiguous() && t.layout.offset() == 0 {
        Ok(t.clone())
    } else {
        kernels::strided_copy(MetalDevice::get(), t)
    }
}

fn require_f32(t: &MetalTensor) -> crate::err::Res<()> {
    if !matches!(t.dtype, DType::F32 | DType::BF16) {
        return Err(format!(
            "metal_native: emitter supports f32 and bf16, got {:?}",
            t.dtype
        ));
    }
    Ok(())
}

fn broadcast_shape(a: &[usize], b: &[usize]) -> crate::err::Res<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![1usize; rank];
    for d in 0..rank {
        let ad = if d < rank - a.len() {
            1
        } else {
            a[d - (rank - a.len())]
        };
        let bd = if d < rank - b.len() {
            1
        } else {
            b[d - (rank - b.len())]
        };
        if ad != bd && ad != 1 && bd != 1 {
            return Err(format!("shape mismatch: {a:?} vs {b:?}"));
        }
        out[d] = ad.max(bd);
    }
    Ok(out)
}

fn lane_strides(t_shape: &[usize], shape: &[usize]) -> crate::err::Res<Vec<usize>> {
    let rank = shape.len();
    let extra = rank - t_shape.len();
    let mut out = vec![0usize; rank];
    let contig = Layout::contiguous(t_shape.to_vec());
    let cs = contig.strides().to_vec();
    for d in 0..t_shape.len() {
        let src = t_shape[d];
        let dst = shape[extra + d];
        if src == dst {
            out[extra + d] = cs[d];
        } else if src == 1 {
            out[extra + d] = 0;
        } else {
            return Err(format!("broadcast: cannot map {t_shape:?} to {shape:?}"));
        }
    }
    Ok(out)
}

fn elementwise(
    exprs: &[Expr],
    inputs: &[&MetalTensor],
    strides: Vec<Vec<usize>>,
    shape: &[usize],
) -> crate::err::Res<MetalTensor> {
    let n: usize = shape.iter().product();
    let outs = crate::runtime::metal::run::run_elementwise(
        MetalDevice::get(),
        exprs,
        inputs,
        &strides,
        &[],
        n,
        shape,
    )?;
    Ok(outs.into_iter().next().expect("one output"))
}

pub fn to_f32(t: &MetalTensor) -> crate::err::Res<MetalTensor> {
    if t.dtype == DType::F32 {
        return Ok(t.clone());
    }
    let f32t = kernels::cast(MetalDevice::get(), t, DType::F32)?;
    Ok(f32t)
}

pub fn from_f32(t: &MetalTensor, dtype: DType) -> crate::err::Res<MetalTensor> {
    if dtype == DType::F32 {
        return Ok(t.clone());
    }
    kernels::cast(MetalDevice::get(), t, dtype)
}

pub fn binary(a: &MetalTensor, b: &MetalTensor, op: BinOp) -> crate::err::Res<MetalTensor> {
    require_f32(a)?;
    require_f32(b)?;
    if a.dtype != b.dtype {
        return Err(format!(
            "binary: dtype mismatch, got {:?} and {:?}; cast explicitly",
            a.dtype, b.dtype
        ));
    }
    let shape = broadcast_shape(a.layout.shape(), b.layout.shape())?;
    let an = contig(a)?;
    let bn = contig(b)?;
    let sa = lane_strides(a.layout.shape(), &shape)?;
    let sb = lane_strides(b.layout.shape(), &shape)?;
    let exprs = vec![bin_expr(&op, Expr::Input(0), Expr::Input(1))];
    elementwise(&exprs, &[&an, &bn], vec![sa, sb], &shape)
}

pub fn binary_promote(a: &MetalTensor, b: &MetalTensor, op: BinOp) -> crate::err::Res<MetalTensor> {
    // A 0-d float operand never promotes a float tensor's dtype: the
    // scalar is cast into it (mirrors scalar_aware_binary_dtype in
    // lib.rs), so an f32 scalar gradient scaling a bf16 tensor stays
    // bf16 instead of materializing an f32 copy of the tensor.
    let ac;
    let bc;
    let (a, b) = if a.dtype != b.dtype
        && a.dtype.is_float()
        && b.dtype.is_float()
        && a.layout.shape().is_empty()
        && !b.layout.shape().is_empty()
    {
        ac = cast(a, b.dtype)?;
        (&ac, b)
    } else if a.dtype != b.dtype
        && a.dtype.is_float()
        && b.dtype.is_float()
        && b.layout.shape().is_empty()
        && !a.layout.shape().is_empty()
    {
        bc = cast(b, a.dtype)?;
        (a, &bc)
    } else {
        (a, b)
    };
    if a.dtype == b.dtype && matches!(a.dtype, DType::F32 | DType::BF16) {
        return binary(a, b, op);
    }
    let dt = a.dtype;
    let a32 = to_f32(a)?;
    let b32 = to_f32(b)?;
    let out = binary(&a32, &b32, op)?;
    from_f32(&out, dt)
}

pub fn compare(a: &MetalTensor, b: &MetalTensor, op: BinOp) -> crate::err::Res<MetalTensor> {
    let a32 = to_f32(a)?;
    let b32 = to_f32(b)?;
    let f = binary(&a32, &b32, op)?;
    kernels::cast(MetalDevice::get(), &f, DType::U8)
}

pub fn unary_promote(a: &MetalTensor, op: UnOp) -> crate::err::Res<MetalTensor> {
    let dt = a.dtype;
    let a32 = to_f32(a)?;
    let out = unary(&a32, op)?;
    from_f32(&out, dt)
}

pub fn unary(a: &MetalTensor, op: UnOp) -> crate::err::Res<MetalTensor> {
    require_f32(a)?;
    let an = contig(a)?;
    let shape = a.layout.shape().to_vec();
    let contig_l = Layout::contiguous(shape.clone());
    let exprs = vec![un_expr(&op, Expr::Input(0))];
    elementwise(&exprs, &[&an], vec![contig_l.strides().to_vec()], &shape)
}

pub fn relu(a: &MetalTensor) -> crate::err::Res<MetalTensor> {
    match a.dtype {
        DType::F32 | DType::F16 | DType::BF16 => {
            let dt = a.dtype;
            let a32 = to_f32(a)?;
            let an = contig(&a32)?;
            let shape = an.layout.shape().to_vec();
            let contig_l = Layout::contiguous(shape.clone());
            let exprs = vec![Expr::Max(
                Box::new(Expr::Input(0)),
                Box::new(Expr::Const(0.0f64.to_bits())),
            )];
            let out = elementwise(&exprs, &[&an], vec![contig_l.strides().to_vec()], &shape)?;
            from_f32(&out, dt)
        }
        DType::U8 | DType::U32 => contig(a),
        DType::I64 => {
            let an = contig(a)?;
            kernels::relu_i64(MetalDevice::get(), &an)
        }
        other => Err(format!("relu: unsupported dtype {other:?}")),
    }
}

pub fn powf(a: &MetalTensor, e: f64) -> crate::err::Res<MetalTensor> {
    require_f32(a)?;
    let an = contig(a)?;
    let shape = a.layout.shape().to_vec();
    let contig_l = Layout::contiguous(shape.clone());
    let exprs = vec![Expr::Powf(Box::new(Expr::Input(0)), e.to_bits())];
    elementwise(&exprs, &[&an], vec![contig_l.strides().to_vec()], &shape)
}

pub fn where_(
    cond: &MetalTensor,
    a: &MetalTensor,
    b: &MetalTensor,
) -> crate::err::Res<MetalTensor> {
    require_f32(a)?;
    require_f32(b)?;
    if a.dtype != b.dtype {
        return Err(format!(
            "where: branch dtype mismatch, got {:?} and {:?}; cast explicitly",
            a.dtype, b.dtype
        ));
    }
    // The emitter types every lane uniformly; the condition rides the
    // branch dtype (u8/f32 masks cast cleanly into either).
    let cond32 = kernels::cast(MetalDevice::get(), cond, a.dtype)?;
    let shape = broadcast_shape(
        &broadcast_shape(cond.layout.shape(), a.layout.shape())?,
        b.layout.shape(),
    )?;
    let cn = contig(&cond32)?;
    let an = contig(a)?;
    let bn = contig(b)?;
    let sc = lane_strides(cond.layout.shape(), &shape)?;
    let sa = lane_strides(a.layout.shape(), &shape)?;
    let sb = lane_strides(b.layout.shape(), &shape)?;
    let exprs = vec![Expr::Select(
        Box::new(Expr::Input(0)),
        Box::new(Expr::Input(1)),
        Box::new(Expr::Input(2)),
    )];
    elementwise(&exprs, &[&cn, &an, &bn], vec![sc, sa, sb], &shape)
}

pub fn reduce(
    a: &MetalTensor,
    dims: &[usize],
    keepdims: bool,
    op: ReduceOp,
) -> crate::err::Res<MetalTensor> {
    if !matches!(a.dtype, DType::F32 | DType::BF16) {
        return Err(format!(
            "reduce: unsupported dtype {:?} on Metal (f32 or bf16)",
            a.dtype
        ));
    }
    let an = contig(a)?;
    let in_shape = a.layout.shape().to_vec();
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
    let contig_l = Layout::contiguous(in_shape.clone());
    crate::runtime::metal::run::run_reduce(
        MetalDevice::get(),
        op,
        &Expr::Input(0),
        &[&an],
        &[contig_l.strides().to_vec()],
        &in_shape,
        dims,
        keepdims,
        &out_shape,
    )
}

pub fn matmul(a: &MetalTensor, b: &MetalTensor) -> crate::err::Res<MetalTensor> {
    if a.dtype != b.dtype {
        return Err(format!(
            "matmul: dtype mismatch, got {:?} and {:?}",
            a.dtype, b.dtype
        ));
    }
    if !matches!(a.dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(format!("matmul: unsupported dtype {:?} on Metal", a.dtype));
    }
    let an = contig(a)?;
    let bn = contig(b)?;
    gemm::matmul(MetalDevice::get(), &an, &bn)
}

pub fn cast(a: &MetalTensor, dtype: DType) -> crate::err::Res<MetalTensor> {
    let an = contig(a)?;
    kernels::cast(MetalDevice::get(), &an, dtype)
}

pub fn contiguous(t: &MetalTensor) -> crate::err::Res<MetalTensor> {
    contig(t)
}

pub fn permute(t: &MetalTensor, dims: &[usize]) -> crate::err::Res<MetalTensor> {
    let p = MetalTensor {
        buffer: t.buffer.clone(),
        layout: t.layout.permute(dims),
        dtype: t.dtype,
    };
    contig(&p)
}

pub fn broadcast_to(t: &MetalTensor, shape: &[usize]) -> crate::err::Res<MetalTensor> {
    let b = MetalTensor {
        buffer: t.buffer.clone(),
        layout: t.layout.broadcast_to(shape),
        dtype: t.dtype,
    };
    contig(&b)
}

pub fn index_select(
    a: &MetalTensor,
    dim: usize,
    ids: &MetalTensor,
) -> crate::err::Res<MetalTensor> {
    let an = contig(a)?;
    let idn = contig(ids)?;
    indexing::index_select(MetalDevice::get(), &an, dim, &idn)
}

pub fn gather(
    a: &MetalTensor,
    dim: usize,
    ids: &MetalTensor,
    ids_shape: &[usize],
) -> crate::err::Res<MetalTensor> {
    let an = contig(a)?;
    let idn = contig(ids)?;
    indexing::gather(MetalDevice::get(), &an, dim, &idn, ids_shape)
}

pub fn scatter_add(
    a: &MetalTensor,
    dim: usize,
    ids: &MetalTensor,
    src: &MetalTensor,
) -> crate::err::Res<MetalTensor> {
    let an = contig(a)?;
    let sn = contig(src)?;
    let idn = contig(ids)?;
    indexing::scatter_add(MetalDevice::get(), &an, dim, &idn, &sn)
}

pub fn cat(a: &MetalTensor, b: &MetalTensor, dim: usize) -> crate::err::Res<MetalTensor> {
    let an = contig(a)?;
    let bn = contig(b)?;
    indexing::cat(MetalDevice::get(), &[&an, &bn], dim)
}

pub fn argreduce(a: &MetalTensor, dim: usize, pick_max: bool) -> crate::err::Res<MetalTensor> {
    let an = contig(a)?;
    kernels::argreduce(MetalDevice::get(), &an, dim, pick_max)
}

pub fn cumsum(a: &MetalTensor, dim: usize) -> crate::err::Res<MetalTensor> {
    let an = contig(a)?;
    kernels::cumsum(MetalDevice::get(), &an, dim)
}

pub fn fill(shape: &[usize], value: f64, dtype: DType) -> crate::err::Res<MetalTensor> {
    let out = MetalTensor::empty(MetalDevice::get(), shape.to_vec(), dtype);
    kernels::fill(MetalDevice::get(), &out, value)?;
    Ok(out)
}

pub fn arange(start: f64, end: f64, step: f64, dtype: DType) -> crate::err::Res<MetalTensor> {
    kernels::arange(MetalDevice::get(), start, end, step, dtype)
}

pub fn eye(n: usize, dtype: DType) -> crate::err::Res<MetalTensor> {
    kernels::eye(MetalDevice::get(), n, dtype)
}

pub fn randn(shape: &[usize], seed: u64) -> crate::err::Res<MetalTensor> {
    kernels::randn(MetalDevice::get(), shape, seed)
}

pub fn uniform(lo: f64, hi: f64, shape: &[usize], seed: u64) -> crate::err::Res<MetalTensor> {
    kernels::uniform(MetalDevice::get(), lo, hi, shape, seed)
}

pub fn conv1d(
    x: &MetalTensor,
    w: &MetalTensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> crate::err::Res<MetalTensor> {
    let xn = contig(x)?;
    let wn = contig(w)?;
    conv::conv1d(
        MetalDevice::get(),
        &xn,
        &wn,
        stride,
        padding,
        dilation,
        groups,
    )
}

pub fn conv2d(
    x: &MetalTensor,
    w: &MetalTensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> crate::err::Res<MetalTensor> {
    let xn = contig(x)?;
    let wn = contig(w)?;
    conv::conv2d(
        MetalDevice::get(),
        &xn,
        &wn,
        stride,
        padding,
        dilation,
        groups,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d(
    x: &MetalTensor,
    w: &MetalTensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> crate::err::Res<MetalTensor> {
    let xn = contig(x)?;
    let wn = contig(w)?;
    conv::conv_transpose1d(
        MetalDevice::get(),
        &xn,
        &wn,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d(
    x: &MetalTensor,
    w: &MetalTensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> crate::err::Res<MetalTensor> {
    let xn = contig(x)?;
    let wn = contig(w)?;
    conv::conv_transpose2d(
        MetalDevice::get(),
        &xn,
        &wn,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_backward_w(
    x: &MetalTensor,
    g: &MetalTensor,
    kernel: [usize; 2],
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> crate::err::Res<MetalTensor> {
    let xn = contig(x)?;
    let gn = contig(g)?;
    conv::conv2d_backward_w(
        MetalDevice::get(),
        &xn,
        &gn,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
    )
}

pub fn gemm_bias(
    x: &MetalTensor,
    w: &MetalTensor,
    bias: &MetalTensor,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> crate::err::Res<MetalTensor> {
    let xn = contig(x)?;
    let wn = contig(w)?;
    gemm::gemm(
        MetalDevice::get(),
        &xn,
        &wn,
        Some(bias),
        batch,
        m,
        n,
        k,
        m * k,
        0,
    )
}

pub fn linear(
    x: &MetalTensor,
    w: &MetalTensor,
    bias: &MetalTensor,
) -> crate::err::Res<MetalTensor> {
    require_f32(x)?;
    require_f32(w)?;
    let dims = x.layout.shape();
    let rank = dims.len();
    let (k, n) = (w.layout.shape()[0], w.layout.shape()[1]);
    let m = dims[rank - 2];
    let b: usize = dims[..rank - 2].iter().product();
    let x_flat = MetalTensor {
        buffer: x.buffer.clone(),
        layout: Layout::contiguous(vec![b, m, k]),
        dtype: x.dtype,
    };
    let xn = contig(&x_flat)?;
    let out = gemm_bias(&xn, w, bias, b, m, n, k)?;
    let mut out_shape = dims.to_vec();
    out_shape[rank - 1] = n;
    Ok(MetalTensor {
        buffer: out.buffer,
        layout: Layout::contiguous(out_shape),
        dtype: out.dtype,
    })
}

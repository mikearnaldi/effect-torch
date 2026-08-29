//! Dispatch helpers used by executable commands for ordinary operations:
//! binary/unary/compare/cast/matmul/contiguous/views plus the creation and
//! conv families forwarded to [`crate::kernels`], [`crate::indexing`],
//! [`crate::gemm`], and [`crate::conv`].
//!
//! # Dtype and layout rules
//!
//! - The fused emitter computes in f32 and stores f32/bf16. Elementwise entry
//!   points accept those types directly. The `*_promote` family casts other
//!   types to f32, runs the fused kernel, and casts back. Comparisons use an f32
//!   intermediate and produce u8.
//! - Broadcasting follows NumPy rules. A lane stride of 0 encodes a broadcast
//!   dimension in the emitted kernel.
//! - Allocating paths use `kernels::strided_copy` to materialize
//!   non-contiguous or offset inputs. The `*_into` forms keep caller layouts
//!   where the kernel supports them.
//!
//! # Scratch contract
//!
//! `*_scratch_requirements` report intermediate tensors in consumption order.
//! The `*_into` variants validate caller-owned scratch and allocate nothing.
//! Allocating wrappers derive the requirements and delegate to `*_into`.

use crate::fusion::{Expr, ReduceOp};
use crate::runtime::dtype::DType;
use crate::runtime::layout::Layout;
use crate::runtime::metal::device::MetalDevice;
use crate::runtime::metal::run::MetalTensor;
use crate::runtime::metal::{conv, gemm, indexing, kernels};

/// Elementwise binary operators (comparisons produce 1.0/0.0 in the fused
/// kernel; the `compare` wrappers then cast to u8).
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

/// Elementwise unary operators. `Sign` is lowered to a select expression;
/// there is no dedicated IR node.
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

/// One exact intermediate tensor an operation needs (contiguous, offset 0).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScratchRequirement {
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl ScratchRequirement {
    /// Byte size of the described tensor, checked for overflow.
    pub fn bytes(&self) -> crate::err::Res<usize> {
        self.shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .and_then(|count| count.checked_mul(self.dtype.size_in_bytes()))
            .ok_or_else(|| "metal operation scratch byte size overflow".to_string())
    }

    fn validate(&self, tensor: &MetalTensor, name: &str) -> crate::err::Res<()> {
        tensor.validate_destination(name, &self.shape, self.dtype)
    }
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

fn tensor_lane_strides(t: &MetalTensor, shape: &[usize]) -> crate::err::Res<Vec<usize>> {
    layout_lane_strides(&t.layout, shape)
}

fn layout_lane_strides(layout: &Layout, shape: &[usize]) -> crate::err::Res<Vec<usize>> {
    let t_shape = layout.shape();
    let rank = shape.len();
    let extra = rank
        .checked_sub(t_shape.len())
        .ok_or_else(|| format!("broadcast: cannot map {t_shape:?} to {shape:?}"))?;
    let mut out = vec![0usize; rank];
    for dimension in 0..t_shape.len() {
        let source = t_shape[dimension];
        let destination = shape[extra + dimension];
        if source == destination {
            out[extra + dimension] = layout.strides()[dimension];
        } else if source != 1 {
            return Err(format!("broadcast: cannot map {t_shape:?} to {shape:?}"));
        }
    }
    Ok(out)
}

fn compile_elementwise_exact(
    exprs: &[Expr],
    inputs: &[(&Layout, DType)],
    shape: &[usize],
) -> crate::err::Res<()> {
    let dtype = inputs
        .first()
        .ok_or_else(|| "elementwise requires at least one input".to_string())?
        .1;
    if !matches!(dtype, DType::F32 | DType::BF16) {
        return Err(format!(
            "metal_native: emitter supports f32 and bf16, got {dtype:?}"
        ));
    }
    if inputs.iter().any(|(_, input_dtype)| *input_dtype != dtype) {
        return Err("elementwise precompile received mixed input dtypes".to_string());
    }
    let strides = inputs
        .iter()
        .map(|(layout, _)| layout_lane_strides(layout, shape))
        .collect::<Result<Vec<_>, _>>()?;
    crate::run::compile_elementwise(
        MetalDevice::get(),
        exprs,
        &strides,
        shape,
        shape.iter().product(),
        0,
        dtype,
    )
}

fn elementwise_into(
    exprs: &[Expr],
    inputs: &[&MetalTensor],
    strides: Vec<Vec<usize>>,
    shape: &[usize],
    out: &MetalTensor,
) -> crate::err::Res<()> {
    if exprs.len() != 1 {
        return Err(format!(
            "single-output operation received {} expressions",
            exprs.len()
        ));
    }
    crate::runtime::metal::run::run_elementwise_into(
        MetalDevice::get(),
        exprs,
        inputs,
        &strides,
        None,
        0,
        shape.iter().product(),
        shape,
        &[out],
    )
}

/// Precompiles the fused elementwise kernel for a unary op on `shape`
/// (f32 math; non-f32 storage is converted before the op).
pub fn warm_unary(shape: &[usize], _dtype: DType, op: UnOp) -> crate::err::Res<()> {
    let expr = un_expr(&op, Expr::Input(0));
    crate::run::warm_elementwise(
        MetalDevice::get(),
        &[expr],
        &[Layout::contiguous(shape.to_vec()).strides().to_vec()],
        shape,
        shape.iter().product(),
        0,
        // unary_promote converts non-f32 storage before the fused operation.
        DType::F32,
    )
}

/// Precompiles relu for `shape`; integer dtypes need no fused kernel
/// (copy for u8/u32, a dedicated kernel for i64).
pub fn warm_relu(shape: &[usize], dtype: DType) -> crate::err::Res<()> {
    if matches!(dtype, DType::U8 | DType::U32) {
        return Ok(());
    }
    if dtype == DType::I64 {
        return Ok(());
    }
    crate::run::warm_elementwise(
        MetalDevice::get(),
        &[Expr::Max(
            Box::new(Expr::Input(0)),
            Box::new(Expr::Const(0.0f64.to_bits())),
        )],
        &[Layout::contiguous(shape.to_vec()).strides().to_vec()],
        shape,
        shape.iter().product(),
        0,
        DType::F32,
    )
}

/// Precompiles `x.powf(e)` for `shape`; the exponent is the f64 bits.
pub fn warm_pow(shape: &[usize], exponent_bits: u64) -> crate::err::Res<()> {
    crate::run::warm_elementwise(
        MetalDevice::get(),
        &[Expr::Powf(Box::new(Expr::Input(0)), exponent_bits)],
        &[Layout::contiguous(shape.to_vec()).strides().to_vec()],
        shape,
        shape.iter().product(),
        0,
        DType::F32,
    )
}

/// Precompiles a broadcast binary op, applying the same dtype-selection
/// rules as [`binary_promote`]/[`compare`].
pub fn warm_binary(
    a_shape: &[usize],
    a_dtype: DType,
    b_shape: &[usize],
    b_dtype: DType,
    op: BinOp,
    compare: bool,
) -> crate::err::Res<()> {
    let shape = broadcast_shape(a_shape, b_shape)?;
    let dtype = if compare {
        DType::F32
    } else if a_dtype == b_dtype && matches!(a_dtype, DType::F32 | DType::BF16) {
        a_dtype
    } else if a_shape.is_empty() && !b_shape.is_empty() && a_dtype.is_float() && b_dtype.is_float()
    {
        b_dtype
    } else if b_shape.is_empty() && !a_shape.is_empty() && a_dtype.is_float() && b_dtype.is_float()
    {
        a_dtype
    } else {
        DType::F32
    };
    crate::run::warm_elementwise(
        MetalDevice::get(),
        &[bin_expr(&op, Expr::Input(0), Expr::Input(1))],
        &[
            lane_strides(a_shape, &shape)?,
            lane_strides(b_shape, &shape)?,
        ],
        &shape,
        shape.iter().product(),
        0,
        dtype,
    )
}

/// Precompiles a three-way broadcast select (`where`).
pub fn warm_where(
    condition_shape: &[usize],
    a_shape: &[usize],
    b_shape: &[usize],
    dtype: DType,
) -> crate::err::Res<()> {
    let shape = broadcast_shape(&broadcast_shape(condition_shape, a_shape)?, b_shape)?;
    crate::run::warm_elementwise(
        MetalDevice::get(),
        &[Expr::Select(
            Box::new(Expr::Input(0)),
            Box::new(Expr::Input(1)),
            Box::new(Expr::Input(2)),
        )],
        &[
            lane_strides(condition_shape, &shape)?,
            lane_strides(a_shape, &shape)?,
            lane_strides(b_shape, &shape)?,
        ],
        &shape,
        shape.iter().product(),
        0,
        dtype,
    )
}

/// Precompiles a plain (unfused expression) reduction over `dims`.
pub fn warm_reduce(
    in_shape: &[usize],
    dtype: DType,
    dims: &[usize],
    keepdims: bool,
    op: ReduceOp,
) -> crate::err::Res<()> {
    let out_shape = if keepdims {
        let mut shape = in_shape.to_vec();
        for &dimension in dims {
            shape[dimension] = 1;
        }
        shape
    } else {
        in_shape
            .iter()
            .enumerate()
            .filter_map(|(dimension, size)| (!dims.contains(&dimension)).then_some(*size))
            .collect()
    };
    crate::run::warm_reduce(
        MetalDevice::get(),
        op,
        &Expr::Input(0),
        &[Layout::contiguous(in_shape.to_vec()).strides().to_vec()],
        in_shape,
        dims,
        keepdims,
        &out_shape,
        if matches!(dtype, DType::F32 | DType::BF16) {
            dtype
        } else {
            DType::F32
        },
    )
}

/// Casts to f32; an f32 input is returned as-is (aliased).
pub fn to_f32(t: &MetalTensor) -> crate::err::Res<MetalTensor> {
    if t.dtype == DType::F32 {
        return Ok(t.clone());
    }
    let f32t = kernels::cast(MetalDevice::get(), t, DType::F32)?;
    Ok(f32t)
}

/// Casts an f32 tensor to `dtype`; f32 targets alias the input.
pub fn from_f32(t: &MetalTensor, dtype: DType) -> crate::err::Res<MetalTensor> {
    if dtype == DType::F32 {
        return Ok(t.clone());
    }
    kernels::cast(MetalDevice::get(), t, dtype)
}

fn validate_scratch(
    operation: &str,
    requirements: &[ScratchRequirement],
    scratch: &[&MetalTensor],
) -> crate::err::Res<()> {
    if requirements.len() != scratch.len() {
        return Err(format!(
            "{operation}: expected {} scratch tensor(s), got {}",
            requirements.len(),
            scratch.len()
        ));
    }
    for (index, (requirement, tensor)) in requirements.iter().zip(scratch).enumerate() {
        requirement.validate(tensor, &format!("{operation} scratch {index}"))?;
    }
    Ok(())
}

fn allocate_scratch(requirements: &[ScratchRequirement]) -> Vec<MetalTensor> {
    requirements
        .iter()
        .map(|requirement| {
            MetalTensor::empty(
                MetalDevice::get(),
                requirement.shape.clone(),
                requirement.dtype,
            )
        })
        .collect()
}

/// Precompiles every pipeline [`binary_promote_into`] will need: scalar
/// dtype promotion casts, the fused broadcast kernel, and the result cast.
pub fn precompile_binary_promote(
    a: &MetalTensor,
    b: &MetalTensor,
    op: BinOp,
) -> crate::err::Res<()> {
    let mut a_layout = a.layout.clone();
    let mut b_layout = b.layout.clone();
    let mut a_dtype = a.dtype;
    let mut b_dtype = b.dtype;
    if a_dtype != b_dtype
        && a_dtype.is_float()
        && b_dtype.is_float()
        && a.layout.shape().is_empty()
        && !b.layout.shape().is_empty()
    {
        kernels::compile_cast_layout(MetalDevice::get(), &a_layout, a_dtype, b_dtype)?;
        a_layout = Layout::contiguous(a.layout.shape().to_vec());
        a_dtype = b_dtype;
    } else if a_dtype != b_dtype
        && a_dtype.is_float()
        && b_dtype.is_float()
        && b.layout.shape().is_empty()
        && !a.layout.shape().is_empty()
    {
        kernels::compile_cast_layout(MetalDevice::get(), &b_layout, b_dtype, a_dtype)?;
        b_layout = Layout::contiguous(b.layout.shape().to_vec());
        b_dtype = a_dtype;
    }
    let shape = broadcast_shape(a.layout.shape(), b.layout.shape())?;
    if a_dtype == b_dtype && matches!(a_dtype, DType::F32 | DType::BF16) {
        return compile_elementwise_exact(
            &[bin_expr(&op, Expr::Input(0), Expr::Input(1))],
            &[(&a_layout, a_dtype), (&b_layout, b_dtype)],
            &shape,
        );
    }
    if a_dtype != DType::F32 {
        kernels::compile_cast_layout(MetalDevice::get(), &a_layout, a_dtype, DType::F32)?;
        a_layout = Layout::contiguous(a.layout.shape().to_vec());
        a_dtype = DType::F32;
    }
    if b_dtype != DType::F32 {
        kernels::compile_cast_layout(MetalDevice::get(), &b_layout, b_dtype, DType::F32)?;
        b_layout = Layout::contiguous(b.layout.shape().to_vec());
        b_dtype = DType::F32;
    }
    compile_elementwise_exact(
        &[bin_expr(&op, Expr::Input(0), Expr::Input(1))],
        &[(&a_layout, a_dtype), (&b_layout, b_dtype)],
        &shape,
    )?;
    let output_dtype = binary_promote_output_dtype(a, b);
    if output_dtype != DType::F32 {
        kernels::compile_cast(MetalDevice::get(), &shape, DType::F32, output_dtype)?;
    }
    Ok(())
}

/// Precompiles the compare pipeline set: f32 casts for both operands, the
/// fused compare kernel, and the f32→u8 result cast.
pub fn precompile_compare(a: &MetalTensor, b: &MetalTensor, op: BinOp) -> crate::err::Res<()> {
    let mut a_layout = a.layout.clone();
    let mut b_layout = b.layout.clone();
    if a.dtype != DType::F32 {
        kernels::compile_cast_layout(MetalDevice::get(), &a_layout, a.dtype, DType::F32)?;
        a_layout = Layout::contiguous(a.layout.shape().to_vec());
    }
    if b.dtype != DType::F32 {
        kernels::compile_cast_layout(MetalDevice::get(), &b_layout, b.dtype, DType::F32)?;
        b_layout = Layout::contiguous(b.layout.shape().to_vec());
    }
    let shape = broadcast_shape(a.layout.shape(), b.layout.shape())?;
    compile_elementwise_exact(
        &[bin_expr(&op, Expr::Input(0), Expr::Input(1))],
        &[(&a_layout, DType::F32), (&b_layout, DType::F32)],
        &shape,
    )?;
    kernels::compile_cast(MetalDevice::get(), &shape, DType::F32, DType::U8)
}

/// Precompiles the promote-cast, fused unary kernel, and cast-back
/// pipeline set for a non-f32 (or directly for an f32) operand.
pub fn precompile_unary_promote(a: &MetalTensor, op: UnOp) -> crate::err::Res<()> {
    if a.dtype == DType::F32 {
        return compile_elementwise_exact(
            &[un_expr(&op, Expr::Input(0))],
            &[(&a.layout, DType::F32)],
            a.layout.shape(),
        );
    }
    kernels::compile_cast_layout(MetalDevice::get(), &a.layout, a.dtype, DType::F32)?;
    let layout = Layout::contiguous(a.layout.shape().to_vec());
    compile_elementwise_exact(
        &[un_expr(&op, Expr::Input(0))],
        &[(&layout, DType::F32)],
        a.layout.shape(),
    )?;
    kernels::compile_cast_layout(MetalDevice::get(), &layout, DType::F32, a.dtype)
}

fn relu_expr() -> Expr {
    Expr::Max(
        Box::new(Expr::Input(0)),
        Box::new(Expr::Const(0.0f64.to_bits())),
    )
}

/// Precompiles the dtype-appropriate relu path (fused f32, cast round
/// trip for f16/bf16, copy for u8/u32, dedicated kernel for i64).
pub fn precompile_relu(a: &MetalTensor) -> crate::err::Res<()> {
    match a.dtype {
        DType::F32 => {
            compile_elementwise_exact(&[relu_expr()], &[(&a.layout, DType::F32)], a.layout.shape())
        }
        DType::F16 | DType::BF16 => {
            kernels::compile_cast_layout(MetalDevice::get(), &a.layout, a.dtype, DType::F32)?;
            let layout = Layout::contiguous(a.layout.shape().to_vec());
            compile_elementwise_exact(&[relu_expr()], &[(&layout, DType::F32)], a.layout.shape())?;
            kernels::compile_cast_layout(MetalDevice::get(), &layout, DType::F32, a.dtype)
        }
        DType::U8 | DType::U32 => {
            kernels::compile_copy_layout(MetalDevice::get(), &a.layout, a.dtype)
        }
        DType::I64 => kernels::compile_relu_i64_layout(MetalDevice::get(), &a.layout),
        other => Err(format!("relu: unsupported dtype {other:?}")),
    }
}

/// Precompiles the select kernel (and the condition cast when the
/// condition dtype differs from the branch dtype).
pub fn precompile_where(
    cond: &MetalTensor,
    a: &MetalTensor,
    b: &MetalTensor,
) -> crate::err::Res<()> {
    let _ = where_scratch_requirements(cond, a, b)?;
    let condition_layout = if cond.dtype == a.dtype {
        cond.layout.clone()
    } else {
        kernels::compile_cast_layout(MetalDevice::get(), &cond.layout, cond.dtype, a.dtype)?;
        Layout::contiguous(cond.layout.shape().to_vec())
    };
    let shape = broadcast_shape(
        &broadcast_shape(cond.layout.shape(), a.layout.shape())?,
        b.layout.shape(),
    )?;
    compile_elementwise_exact(
        &[Expr::Select(
            Box::new(Expr::Input(0)),
            Box::new(Expr::Input(1)),
            Box::new(Expr::Input(2)),
        )],
        &[
            (&condition_layout, a.dtype),
            (&a.layout, a.dtype),
            (&b.layout, b.dtype),
        ],
        &shape,
    )
}

/// Precompiles the fused broadcast binary kernel for exact input
/// layouts/dtypes.
pub fn precompile_binary(a: &MetalTensor, b: &MetalTensor, op: BinOp) -> crate::err::Res<()> {
    let shape = broadcast_shape(a.layout.shape(), b.layout.shape())?;
    let exprs = [bin_expr(&op, Expr::Input(0), Expr::Input(1))];
    compile_elementwise_exact(
        &exprs,
        &[(&a.layout, a.dtype), (&b.layout, b.dtype)],
        &shape,
    )
}

/// Allocating broadcast binary op; both operands must share one f32/bf16
/// dtype.
pub fn binary(a: &MetalTensor, b: &MetalTensor, op: BinOp) -> crate::err::Res<MetalTensor> {
    precompile_binary(a, b, op)?;
    let shape = broadcast_shape(a.layout.shape(), b.layout.shape())?;
    let out = MetalTensor::empty(MetalDevice::get(), shape, a.dtype);
    binary_into(a, b, op, &out)?;
    Ok(out)
}

/// Destination form of [`binary`]; requires the precompiled pipeline.
pub fn binary_into(
    a: &MetalTensor,
    b: &MetalTensor,
    op: BinOp,
    out: &MetalTensor,
) -> crate::err::Res<()> {
    require_f32(a)?;
    require_f32(b)?;
    if a.dtype != b.dtype {
        return Err(format!(
            "binary: dtype mismatch, got {:?} and {:?}; cast explicitly",
            a.dtype, b.dtype
        ));
    }
    let shape = broadcast_shape(a.layout.shape(), b.layout.shape())?;
    out.validate_destination("binary", &shape, a.dtype)?;
    let sa = tensor_lane_strides(a, &shape)?;
    let sb = tensor_lane_strides(b, &shape)?;
    let exprs = vec![bin_expr(&op, Expr::Input(0), Expr::Input(1))];
    elementwise_into(&exprs, &[a, b], vec![sa, sb], &shape, out)
}

/// Allocating binary op with dtype promotion: mismatched float dtypes
/// promote a scalar operand to the tensor's dtype; non-f32/bf16 operands
/// round-trip through f32.
pub fn binary_promote(a: &MetalTensor, b: &MetalTensor, op: BinOp) -> crate::err::Res<MetalTensor> {
    precompile_binary_promote(a, b, op)?;
    let shape = broadcast_shape(a.layout.shape(), b.layout.shape())?;
    let dtype = binary_promote_output_dtype(a, b);
    let out = MetalTensor::empty(MetalDevice::get(), shape, dtype);
    let requirements = binary_promote_scratch_requirements(a, b)?;
    let owned = allocate_scratch(&requirements);
    let scratch = owned.iter().collect::<Vec<_>>();
    binary_promote_into(a, b, op, &out, &scratch)?;
    Ok(out)
}

fn binary_promote_output_dtype(a: &MetalTensor, b: &MetalTensor) -> DType {
    if a.dtype != b.dtype
        && a.dtype.is_float()
        && b.dtype.is_float()
        && a.layout.shape().is_empty()
        && !b.layout.shape().is_empty()
    {
        b.dtype
    } else {
        a.dtype
    }
}

/// The exact intermediate tensors [`binary_promote_into`] consumes, in
/// order: optional scalar-promotion cast, optional f32 casts, optional
/// f32 result buffer.
pub fn binary_promote_scratch_requirements(
    a: &MetalTensor,
    b: &MetalTensor,
) -> crate::err::Res<Vec<ScratchRequirement>> {
    let mut requirements = Vec::new();
    let mut a_dtype = a.dtype;
    let mut b_dtype = b.dtype;
    if a_dtype != b_dtype
        && a_dtype.is_float()
        && b_dtype.is_float()
        && a.layout.shape().is_empty()
        && !b.layout.shape().is_empty()
    {
        requirements.push(ScratchRequirement {
            shape: a.layout.shape().to_vec(),
            dtype: b_dtype,
        });
        a_dtype = b_dtype;
    } else if a_dtype != b_dtype
        && a_dtype.is_float()
        && b_dtype.is_float()
        && b.layout.shape().is_empty()
        && !a.layout.shape().is_empty()
    {
        requirements.push(ScratchRequirement {
            shape: b.layout.shape().to_vec(),
            dtype: a_dtype,
        });
        b_dtype = a_dtype;
    }
    if a_dtype == b_dtype && matches!(a_dtype, DType::F32 | DType::BF16) {
        return Ok(requirements);
    }
    if a_dtype != DType::F32 {
        requirements.push(ScratchRequirement {
            shape: a.layout.shape().to_vec(),
            dtype: DType::F32,
        });
    }
    if b_dtype != DType::F32 {
        requirements.push(ScratchRequirement {
            shape: b.layout.shape().to_vec(),
            dtype: DType::F32,
        });
    }
    if binary_promote_output_dtype(a, b) != DType::F32 {
        requirements.push(ScratchRequirement {
            shape: broadcast_shape(a.layout.shape(), b.layout.shape())?,
            dtype: DType::F32,
        });
    }
    Ok(requirements)
}

/// Destination form of [`binary_promote`]; `scratch` must match
/// [`binary_promote_scratch_requirements`] exactly.
pub fn binary_promote_into(
    a: &MetalTensor,
    b: &MetalTensor,
    op: BinOp,
    out: &MetalTensor,
    scratch: &[&MetalTensor],
) -> crate::err::Res<()> {
    let requirements = binary_promote_scratch_requirements(a, b)?;
    validate_scratch("binary promote", &requirements, scratch)?;
    let shape = broadcast_shape(a.layout.shape(), b.layout.shape())?;
    let output_dtype = binary_promote_output_dtype(a, b);
    out.validate_destination("binary promote", &shape, output_dtype)?;

    let mut index = 0usize;
    let mut a_value = a;
    let mut b_value = b;
    if a.dtype != b.dtype
        && a.dtype.is_float()
        && b.dtype.is_float()
        && a.layout.shape().is_empty()
        && !b.layout.shape().is_empty()
    {
        kernels::cast_into(MetalDevice::get(), a, scratch[index])?;
        a_value = scratch[index];
        index += 1;
    } else if a.dtype != b.dtype
        && a.dtype.is_float()
        && b.dtype.is_float()
        && b.layout.shape().is_empty()
        && !a.layout.shape().is_empty()
    {
        kernels::cast_into(MetalDevice::get(), b, scratch[index])?;
        b_value = scratch[index];
        index += 1;
    }
    if a_value.dtype == b_value.dtype && matches!(a_value.dtype, DType::F32 | DType::BF16) {
        return binary_into(a_value, b_value, op, out);
    }

    let a32 = if a_value.dtype == DType::F32 {
        a_value
    } else {
        kernels::cast_into(MetalDevice::get(), a_value, scratch[index])?;
        let value = scratch[index];
        index += 1;
        value
    };
    let b32 = if b_value.dtype == DType::F32 {
        b_value
    } else {
        kernels::cast_into(MetalDevice::get(), b_value, scratch[index])?;
        let value = scratch[index];
        index += 1;
        value
    };
    if output_dtype == DType::F32 {
        binary_into(a32, b32, op, out)
    } else {
        let result = scratch[index];
        binary_into(a32, b32, op, result)?;
        kernels::cast_into(MetalDevice::get(), result, out)
    }
}

/// Allocating comparison producing u8 (1/0) via an f32 fused kernel.
pub fn compare(a: &MetalTensor, b: &MetalTensor, op: BinOp) -> crate::err::Res<MetalTensor> {
    precompile_compare(a, b, op)?;
    let shape = broadcast_shape(a.layout.shape(), b.layout.shape())?;
    let out = MetalTensor::empty(MetalDevice::get(), shape, DType::U8);
    let requirements = compare_scratch_requirements(a, b)?;
    let owned = allocate_scratch(&requirements);
    let scratch = owned.iter().collect::<Vec<_>>();
    compare_into(a, b, op, &out, &scratch)?;
    Ok(out)
}

/// The exact intermediates [`compare_into`] consumes: optional f32 casts
/// of each operand, then the f32 broadcast result.
pub fn compare_scratch_requirements(
    a: &MetalTensor,
    b: &MetalTensor,
) -> crate::err::Res<Vec<ScratchRequirement>> {
    let mut requirements = Vec::new();
    if a.dtype != DType::F32 {
        requirements.push(ScratchRequirement {
            shape: a.layout.shape().to_vec(),
            dtype: DType::F32,
        });
    }
    if b.dtype != DType::F32 {
        requirements.push(ScratchRequirement {
            shape: b.layout.shape().to_vec(),
            dtype: DType::F32,
        });
    }
    requirements.push(ScratchRequirement {
        shape: broadcast_shape(a.layout.shape(), b.layout.shape())?,
        dtype: DType::F32,
    });
    Ok(requirements)
}

/// Destination form of [`compare`]; `scratch` must match
/// [`compare_scratch_requirements`] exactly.
pub fn compare_into(
    a: &MetalTensor,
    b: &MetalTensor,
    op: BinOp,
    out: &MetalTensor,
    scratch: &[&MetalTensor],
) -> crate::err::Res<()> {
    let requirements = compare_scratch_requirements(a, b)?;
    validate_scratch("compare", &requirements, scratch)?;
    let shape = broadcast_shape(a.layout.shape(), b.layout.shape())?;
    out.validate_destination("compare", &shape, DType::U8)?;
    let mut index = 0usize;
    let a32 = if a.dtype == DType::F32 {
        a
    } else {
        kernels::cast_into(MetalDevice::get(), a, scratch[index])?;
        let value = scratch[index];
        index += 1;
        value
    };
    let b32 = if b.dtype == DType::F32 {
        b
    } else {
        kernels::cast_into(MetalDevice::get(), b, scratch[index])?;
        let value = scratch[index];
        index += 1;
        value
    };
    let result = scratch[index];
    binary_into(a32, b32, op, result)?;
    kernels::cast_into(MetalDevice::get(), result, out)
}

/// Allocating unary op with f32 promotion for non-f32 storage dtypes.
pub fn unary_promote(a: &MetalTensor, op: UnOp) -> crate::err::Res<MetalTensor> {
    precompile_unary_promote(a, op)?;
    let out = MetalTensor::empty(MetalDevice::get(), a.layout.shape().to_vec(), a.dtype);
    let requirements = unary_promote_scratch_requirements(a);
    let owned = allocate_scratch(&requirements);
    let scratch = owned.iter().collect::<Vec<_>>();
    unary_promote_into(a, op, &out, &scratch)?;
    Ok(out)
}

/// The single f32 round-trip buffer [`unary_promote_into`] needs for
/// non-f32 inputs (empty for f32).
pub fn unary_promote_scratch_requirements(a: &MetalTensor) -> Vec<ScratchRequirement> {
    if a.dtype == DType::F32 {
        Vec::new()
    } else {
        vec![ScratchRequirement {
            shape: a.layout.shape().to_vec(),
            dtype: DType::F32,
        }]
    }
}

/// Destination form of [`unary_promote`]; for non-f32 inputs the fused
/// kernel runs in-place on the f32 scratch buffer between the two casts.
pub fn unary_promote_into(
    a: &MetalTensor,
    op: UnOp,
    out: &MetalTensor,
    scratch: &[&MetalTensor],
) -> crate::err::Res<()> {
    let requirements = unary_promote_scratch_requirements(a);
    validate_scratch("unary promote", &requirements, scratch)?;
    out.validate_destination("unary promote", a.layout.shape(), a.dtype)?;
    if requirements.is_empty() {
        return unary_into(a, op, out);
    }
    let value = scratch[0];
    kernels::cast_into(MetalDevice::get(), a, value)?;
    unary_into(value, op, value)?;
    kernels::cast_into(MetalDevice::get(), value, out)
}

/// Precompiles the fused unary kernel for an exact layout/dtype.
pub fn precompile_unary(a: &MetalTensor, op: UnOp) -> crate::err::Res<()> {
    compile_elementwise_exact(
        &[un_expr(&op, Expr::Input(0))],
        &[(&a.layout, a.dtype)],
        a.layout.shape(),
    )
}

/// Allocating fused unary op (f32/bf16 storage only).
pub fn unary(a: &MetalTensor, op: UnOp) -> crate::err::Res<MetalTensor> {
    precompile_unary(a, op)?;
    let out = MetalTensor::empty(MetalDevice::get(), a.layout.shape().to_vec(), a.dtype);
    unary_into(a, op, &out)?;
    Ok(out)
}

/// Destination form of [`unary`]; requires the precompiled pipeline.
pub fn unary_into(a: &MetalTensor, op: UnOp, out: &MetalTensor) -> crate::err::Res<()> {
    require_f32(a)?;
    let shape = a.layout.shape().to_vec();
    out.validate_destination("unary", &shape, a.dtype)?;
    let exprs = vec![un_expr(&op, Expr::Input(0))];
    elementwise_into(&exprs, &[a], vec![a.layout.strides().to_vec()], &shape, out)
}

/// Allocating relu with the dtype-appropriate path.
pub fn relu(a: &MetalTensor) -> crate::err::Res<MetalTensor> {
    precompile_relu(a)?;
    let out = MetalTensor::empty(MetalDevice::get(), a.layout.shape().to_vec(), a.dtype);
    let requirements = relu_scratch_requirements(a)?;
    let owned = allocate_scratch(&requirements);
    let scratch = owned.iter().collect::<Vec<_>>();
    relu_into(a, &out, &scratch)?;
    Ok(out)
}

/// The f32 round-trip buffer relu needs for f16/bf16 inputs (none for
/// f32 or integer dtypes).
pub fn relu_scratch_requirements(a: &MetalTensor) -> crate::err::Res<Vec<ScratchRequirement>> {
    match a.dtype {
        DType::F16 | DType::BF16 => Ok(vec![ScratchRequirement {
            shape: a.layout.shape().to_vec(),
            dtype: DType::F32,
        }]),
        DType::F32 | DType::U8 | DType::U32 | DType::I64 => Ok(Vec::new()),
        other => Err(format!("relu: unsupported dtype {other:?}")),
    }
}

/// Destination form of [`relu`]; integer dtypes pass through (u8/u32) or
/// use the dedicated i64 kernel.
pub fn relu_into(
    a: &MetalTensor,
    out: &MetalTensor,
    scratch: &[&MetalTensor],
) -> crate::err::Res<()> {
    let requirements = relu_scratch_requirements(a)?;
    validate_scratch("relu", &requirements, scratch)?;
    out.validate_destination("relu", a.layout.shape(), a.dtype)?;
    match a.dtype {
        DType::F32 => {
            let exprs = vec![Expr::Max(
                Box::new(Expr::Input(0)),
                Box::new(Expr::Const(0.0f64.to_bits())),
            )];
            elementwise_into(
                &exprs,
                &[a],
                vec![a.layout.strides().to_vec()],
                a.layout.shape(),
                out,
            )
        }
        DType::F16 | DType::BF16 => {
            let value = scratch[0];
            kernels::cast_into(MetalDevice::get(), a, value)?;
            let exprs = vec![Expr::Max(
                Box::new(Expr::Input(0)),
                Box::new(Expr::Const(0.0f64.to_bits())),
            )];
            elementwise_into(
                &exprs,
                &[value],
                vec![value.layout.strides().to_vec()],
                value.layout.shape(),
                value,
            )?;
            kernels::cast_into(MetalDevice::get(), value, out)
        }
        DType::U8 | DType::U32 => kernels::copy_into(MetalDevice::get(), a, out),
        DType::I64 => kernels::relu_i64_into(MetalDevice::get(), a, out),
        other => Err(format!("relu: unsupported dtype {other:?}")),
    }
}

/// Precompiles the fused powf kernel for an exact layout/dtype.
pub fn precompile_powf(a: &MetalTensor, e: f64) -> crate::err::Res<()> {
    compile_elementwise_exact(
        &[Expr::Powf(Box::new(Expr::Input(0)), e.to_bits())],
        &[(&a.layout, a.dtype)],
        a.layout.shape(),
    )
}

/// Allocating `a.powf(e)`.
pub fn powf(a: &MetalTensor, e: f64) -> crate::err::Res<MetalTensor> {
    precompile_powf(a, e)?;
    let out = MetalTensor::empty(MetalDevice::get(), a.layout.shape().to_vec(), a.dtype);
    powf_into(a, e, &out)?;
    Ok(out)
}

/// Destination form of [`powf`]; requires the precompiled pipeline.
pub fn powf_into(a: &MetalTensor, e: f64, out: &MetalTensor) -> crate::err::Res<()> {
    require_f32(a)?;
    let shape = a.layout.shape().to_vec();
    out.validate_destination("pow", &shape, a.dtype)?;
    let exprs = vec![Expr::Powf(Box::new(Expr::Input(0)), e.to_bits())];
    elementwise_into(&exprs, &[a], vec![a.layout.strides().to_vec()], &shape, out)
}

/// Allocating three-way broadcast select: `cond != 0 ? a : b` per lane.
pub fn where_(
    cond: &MetalTensor,
    a: &MetalTensor,
    b: &MetalTensor,
) -> crate::err::Res<MetalTensor> {
    precompile_where(cond, a, b)?;
    let shape = broadcast_shape(
        &broadcast_shape(cond.layout.shape(), a.layout.shape())?,
        b.layout.shape(),
    )?;
    let out = MetalTensor::empty(MetalDevice::get(), shape, a.dtype);
    let requirements = where_scratch_requirements(cond, a, b)?;
    let owned = allocate_scratch(&requirements);
    let scratch = owned.iter().collect::<Vec<_>>();
    where_into(cond, a, b, &out, &scratch)?;
    Ok(out)
}

/// The single cast buffer [`where_into`] needs when the condition dtype
/// differs from the (f32/bf16) branch dtype.
pub fn where_scratch_requirements(
    cond: &MetalTensor,
    a: &MetalTensor,
    b: &MetalTensor,
) -> crate::err::Res<Vec<ScratchRequirement>> {
    require_f32(a)?;
    require_f32(b)?;
    if a.dtype != b.dtype {
        return Err(format!(
            "where: branch dtype mismatch, got {:?} and {:?}; cast explicitly",
            a.dtype, b.dtype
        ));
    }
    Ok((cond.dtype != a.dtype)
        .then(|| ScratchRequirement {
            shape: cond.layout.shape().to_vec(),
            dtype: a.dtype,
        })
        .into_iter()
        .collect())
}

/// Destination form of [`where_`].
pub fn where_into(
    cond: &MetalTensor,
    a: &MetalTensor,
    b: &MetalTensor,
    out: &MetalTensor,
    scratch: &[&MetalTensor],
) -> crate::err::Res<()> {
    let requirements = where_scratch_requirements(cond, a, b)?;
    validate_scratch("where", &requirements, scratch)?;
    let shape = broadcast_shape(
        &broadcast_shape(cond.layout.shape(), a.layout.shape())?,
        b.layout.shape(),
    )?;
    out.validate_destination("where", &shape, a.dtype)?;
    let condition = if requirements.is_empty() {
        cond
    } else {
        kernels::cast_into(MetalDevice::get(), cond, scratch[0])?;
        scratch[0]
    };
    let sc = tensor_lane_strides(condition, &shape)?;
    let sa = tensor_lane_strides(a, &shape)?;
    let sb = tensor_lane_strides(b, &shape)?;
    let exprs = vec![Expr::Select(
        Box::new(Expr::Input(0)),
        Box::new(Expr::Input(1)),
        Box::new(Expr::Input(2)),
    )];
    elementwise_into(&exprs, &[condition, a, b], vec![sc, sa, sb], &shape, out)
}

/// Allocating reduction over `dims` (f32/bf16 only; deterministic serial
/// per-output accumulation).
pub fn reduce(
    a: &MetalTensor,
    dims: &[usize],
    keepdims: bool,
    op: ReduceOp,
) -> crate::err::Res<MetalTensor> {
    precompile_reduce(a, dims, keepdims, op)?;
    let out_shape = reduce_output_shape(a.layout.shape(), dims, keepdims);
    let out = MetalTensor::empty(MetalDevice::get(), out_shape, a.dtype);
    reduce_into(a, dims, keepdims, op, &out)?;
    Ok(out)
}

/// Precompiles the reduction pipeline for an exact input layout and
/// dims/keepdims/op combination.
pub fn precompile_reduce(
    a: &MetalTensor,
    dims: &[usize],
    keepdims: bool,
    op: ReduceOp,
) -> crate::err::Res<()> {
    if !matches!(a.dtype, DType::F32 | DType::BF16) {
        return Err(format!(
            "reduce: unsupported dtype {:?} on Metal (f32 or bf16)",
            a.dtype
        ));
    }
    let in_shape = a.layout.shape().to_vec();
    let out_shape = reduce_output_shape(&in_shape, dims, keepdims);
    crate::run::compile_reduce(
        MetalDevice::get(),
        op,
        &Expr::Input(0),
        &[a.layout.strides().to_vec()],
        &in_shape,
        dims,
        keepdims,
        &out_shape,
        a.dtype,
    )
}

fn reduce_output_shape(in_shape: &[usize], dims: &[usize], keepdims: bool) -> Vec<usize> {
    if keepdims {
        let mut shape = in_shape.to_vec();
        for &dimension in dims {
            shape[dimension] = 1;
        }
        shape
    } else {
        in_shape
            .iter()
            .enumerate()
            .filter_map(|(dimension, &size)| (!dims.contains(&dimension)).then_some(size))
            .collect()
    }
}

/// Destination form of [`reduce`]; requires the precompiled pipeline.
pub fn reduce_into(
    a: &MetalTensor,
    dims: &[usize],
    keepdims: bool,
    op: ReduceOp,
    out: &MetalTensor,
) -> crate::err::Res<()> {
    if !matches!(a.dtype, DType::F32 | DType::BF16) {
        return Err(format!(
            "reduce: unsupported dtype {:?} on Metal (f32 or bf16)",
            a.dtype
        ));
    }
    let in_shape = a.layout.shape().to_vec();
    let out_shape = reduce_output_shape(&in_shape, dims, keepdims);
    out.validate_destination("reduce", &out_shape, a.dtype)?;
    crate::runtime::metal::run::run_reduce_into(
        MetalDevice::get(),
        op,
        &Expr::Input(0),
        &[a],
        &[a.layout.strides().to_vec()],
        &in_shape,
        dims,
        keepdims,
        &out_shape,
        out,
    )
}

/// Allocating matmul (f32/f16/bf16); non-contiguous inputs are
/// materialized first.
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

/// Destination form of [`matmul`]; forwards to [`gemm::matmul_into`] with
/// caller-planned requirements.
pub fn matmul_into(
    a: &MetalTensor,
    b: &MetalTensor,
    out: &MetalTensor,
    split_k_scratch: Option<&MetalTensor>,
    requirements: &gemm::GemmRequirements,
) -> crate::err::Res<()> {
    gemm::matmul_into(MetalDevice::get(), a, b, out, split_k_scratch, requirements)
}

/// Forwards to [`kernels::cast`].
pub fn cast(a: &MetalTensor, dtype: DType) -> crate::err::Res<MetalTensor> {
    kernels::cast(MetalDevice::get(), a, dtype)
}

/// Forwards to [`kernels::cast_into`].
pub fn cast_into(a: &MetalTensor, out: &MetalTensor) -> crate::err::Res<()> {
    kernels::cast_into(MetalDevice::get(), a, out)
}

/// Materializes a contiguous offset-zero copy (aliases the input when it
/// already is one).
pub fn contiguous(t: &MetalTensor) -> crate::err::Res<MetalTensor> {
    contig(t)
}

/// Destination form of [`contiguous`]; forwards to [`kernels::copy_into`].
pub fn contiguous_into(t: &MetalTensor, out: &MetalTensor) -> crate::err::Res<()> {
    kernels::copy_into(MetalDevice::get(), t, out)
}

/// Permutes dimensions, materializing a contiguous result.
pub fn permute(t: &MetalTensor, dims: &[usize]) -> crate::err::Res<MetalTensor> {
    let p = MetalTensor {
        buffer: t.buffer.clone(),
        layout: t.layout.permute(dims),
        dtype: t.dtype,
    };
    contig(&p)
}

/// Destination form of [`permute`]: strided-copies the permuted view.
pub fn permute_into(t: &MetalTensor, dims: &[usize], out: &MetalTensor) -> crate::err::Res<()> {
    let permuted = MetalTensor {
        buffer: t.buffer.clone(),
        layout: t.layout.permute(dims),
        dtype: t.dtype,
    };
    kernels::copy_into(MetalDevice::get(), &permuted, out)
}

/// Broadcasts to `shape`, materializing a contiguous result.
pub fn broadcast_to(t: &MetalTensor, shape: &[usize]) -> crate::err::Res<MetalTensor> {
    let b = MetalTensor {
        buffer: t.buffer.clone(),
        layout: t.layout.broadcast_to(shape),
        dtype: t.dtype,
    };
    contig(&b)
}

/// Destination form of [`broadcast_to`]: strided-copies the broadcast
/// view (zero-stride reads).
pub fn broadcast_to_into(
    t: &MetalTensor,
    shape: &[usize],
    out: &MetalTensor,
) -> crate::err::Res<()> {
    let broadcast = MetalTensor {
        buffer: t.buffer.clone(),
        layout: t.layout.broadcast_to(shape),
        dtype: t.dtype,
    };
    kernels::copy_into(MetalDevice::get(), &broadcast, out)
}

/// Forwards to [`indexing::index_select`] after materializing inputs.
pub fn index_select(
    a: &MetalTensor,
    dim: usize,
    ids: &MetalTensor,
) -> crate::err::Res<MetalTensor> {
    let an = contig(a)?;
    let idn = contig(ids)?;
    indexing::index_select(MetalDevice::get(), &an, dim, &idn)
}

/// Forwards to [`indexing::index_select_into`].
pub fn index_select_into(
    a: &MetalTensor,
    dim: usize,
    ids: &MetalTensor,
    out: &MetalTensor,
    ids_scratch: Option<&MetalTensor>,
) -> crate::err::Res<()> {
    indexing::index_select_into(MetalDevice::get(), a, dim, ids, out, ids_scratch)
}

/// Forwards to [`indexing::gather`] after materializing inputs.
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

/// Forwards to [`indexing::gather_into`].
pub fn gather_into(
    a: &MetalTensor,
    dim: usize,
    ids: &MetalTensor,
    ids_shape: &[usize],
    out: &MetalTensor,
    ids_scratch: Option<&MetalTensor>,
) -> crate::err::Res<()> {
    indexing::gather_into(MetalDevice::get(), a, dim, ids, ids_shape, out, ids_scratch)
}

/// Forwards to [`indexing::scatter_add`] after materializing inputs.
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

/// Forwards to [`indexing::scatter_add_into`].
#[allow(clippy::too_many_arguments)]
pub fn scatter_add_into(
    a: &MetalTensor,
    dim: usize,
    ids: &MetalTensor,
    source: &MetalTensor,
    out: &MetalTensor,
    ids_scratch: Option<&MetalTensor>,
    accumulator: Option<&MetalTensor>,
    source_cast: Option<&MetalTensor>,
) -> crate::err::Res<()> {
    indexing::scatter_add_into(
        MetalDevice::get(),
        a,
        dim,
        ids,
        source,
        out,
        ids_scratch,
        accumulator,
        source_cast,
    )
}

/// Concatenates two tensors along `dim` (materializes inputs first).
pub fn cat(a: &MetalTensor, b: &MetalTensor, dim: usize) -> crate::err::Res<MetalTensor> {
    let an = contig(a)?;
    let bn = contig(b)?;
    indexing::cat(MetalDevice::get(), &[&an, &bn], dim)
}

/// Forwards to [`indexing::cat_into`].
pub fn cat_into(tensors: &[&MetalTensor], dim: usize, out: &MetalTensor) -> crate::err::Res<()> {
    indexing::cat_into(MetalDevice::get(), tensors, dim, out)
}

/// Forwards to [`kernels::argreduce`].
pub fn argreduce(a: &MetalTensor, dim: usize, pick_max: bool) -> crate::err::Res<MetalTensor> {
    kernels::argreduce(MetalDevice::get(), a, dim, pick_max)
}

/// Forwards to [`kernels::argreduce_into`].
pub fn argreduce_into(
    a: &MetalTensor,
    dim: usize,
    pick_max: bool,
    out: &MetalTensor,
) -> crate::err::Res<()> {
    kernels::argreduce_into(MetalDevice::get(), a, dim, pick_max, out)
}

/// Forwards to [`kernels::cumsum`].
pub fn cumsum(a: &MetalTensor, dim: usize) -> crate::err::Res<MetalTensor> {
    kernels::cumsum(MetalDevice::get(), a, dim)
}

/// Forwards to [`kernels::cumsum_into`].
pub fn cumsum_into(a: &MetalTensor, dim: usize, out: &MetalTensor) -> crate::err::Res<()> {
    kernels::cumsum_into(MetalDevice::get(), a, dim, out)
}

/// Allocates a tensor of `shape` filled with `value`.
pub fn fill(shape: &[usize], value: f64, dtype: DType) -> crate::err::Res<MetalTensor> {
    kernels::compile_fill(MetalDevice::get(), shape, value, dtype)?;
    let out = MetalTensor::empty(MetalDevice::get(), shape.to_vec(), dtype);
    fill_into(value, &out)?;
    Ok(out)
}

/// Forwards to [`kernels::fill_into`].
pub fn fill_into(value: f64, out: &MetalTensor) -> crate::err::Res<()> {
    kernels::fill_into(MetalDevice::get(), out, value)
}

/// Forwards to [`kernels::arange`].
pub fn arange(start: f64, end: f64, step: f64, dtype: DType) -> crate::err::Res<MetalTensor> {
    kernels::arange(MetalDevice::get(), start, end, step, dtype)
}

/// Forwards to [`kernels::arange_into`].
pub fn arange_into(start: f64, end: f64, step: f64, out: &MetalTensor) -> crate::err::Res<()> {
    kernels::arange_into(MetalDevice::get(), start, end, step, out)
}

/// Forwards to [`kernels::eye`].
pub fn eye(n: usize, dtype: DType) -> crate::err::Res<MetalTensor> {
    kernels::eye(MetalDevice::get(), n, dtype)
}

/// Forwards to [`kernels::eye_into`].
pub fn eye_into(out: &MetalTensor) -> crate::err::Res<()> {
    kernels::eye_into(MetalDevice::get(), out)
}

/// Forwards to [`kernels::randn`].
pub fn randn(shape: &[usize], seed: u64) -> crate::err::Res<MetalTensor> {
    kernels::randn(MetalDevice::get(), shape, seed)
}

/// Forwards to [`kernels::randn_into`].
pub fn randn_into(out: &MetalTensor, seed: u64) -> crate::err::Res<()> {
    kernels::randn_into(MetalDevice::get(), out, seed)
}

/// Forwards to [`kernels::uniform`].
pub fn uniform(lo: f64, hi: f64, shape: &[usize], seed: u64) -> crate::err::Res<MetalTensor> {
    kernels::uniform(MetalDevice::get(), lo, hi, shape, seed)
}

/// Forwards to [`kernels::uniform_into`].
pub fn uniform_into(lo: f64, hi: f64, out: &MetalTensor, seed: u64) -> crate::err::Res<()> {
    kernels::uniform_into(MetalDevice::get(), lo, hi, out, seed)
}

/// Forwards to [`conv::conv1d`] after materializing inputs.
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

/// Forwards to [`conv::conv1d_into`].
#[allow(clippy::too_many_arguments)]
pub fn conv1d_into(
    x: &MetalTensor,
    w: &MetalTensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    out: &MetalTensor,
) -> crate::err::Res<()> {
    conv::conv1d_into(
        MetalDevice::get(),
        x,
        w,
        stride,
        padding,
        dilation,
        groups,
        out,
    )
}

/// Forwards to [`conv::conv2d`] after materializing inputs.
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

/// Forwards to [`conv::conv2d_into`].
#[allow(clippy::too_many_arguments)]
pub fn conv2d_into(
    x: &MetalTensor,
    w: &MetalTensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    out: &MetalTensor,
) -> crate::err::Res<()> {
    conv::conv2d_into(
        MetalDevice::get(),
        x,
        w,
        stride,
        padding,
        dilation,
        groups,
        out,
    )
}

/// Forwards to [`conv::conv_transpose1d`] after materializing inputs.
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

/// Forwards to [`conv::conv_transpose1d_into`].
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d_into(
    x: &MetalTensor,
    w: &MetalTensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
    out: &MetalTensor,
) -> crate::err::Res<()> {
    conv::conv_transpose1d_into(
        MetalDevice::get(),
        x,
        w,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        out,
    )
}

/// Forwards to [`conv::conv_transpose2d`] after materializing inputs.
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

/// Forwards to [`conv::conv_transpose2d_into`].
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d_into(
    x: &MetalTensor,
    w: &MetalTensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
    out: &MetalTensor,
) -> crate::err::Res<()> {
    conv::conv_transpose2d_into(
        MetalDevice::get(),
        x,
        w,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        out,
    )
}

/// Forwards to [`conv::conv2d_backward_w`] after materializing inputs.
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

/// Forwards to [`conv::conv2d_backward_w_into`].
#[allow(clippy::too_many_arguments)]
pub fn conv2d_backward_w_into(
    x: &MetalTensor,
    gradient: &MetalTensor,
    kernel: [usize; 2],
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    out: &MetalTensor,
) -> crate::err::Res<()> {
    conv::conv2d_backward_w_into(
        MetalDevice::get(),
        x,
        gradient,
        kernel,
        out_channels,
        stride,
        padding,
        dilation,
        groups,
        out,
    )
}

/// Batched gemm with a broadcast bias (`w` is the [k, n] weight).
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

/// Destination form of [`gemm_bias`]; `requirements` must match the exact
/// dimensions and carry `has_bias`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bias_into(
    x: &MetalTensor,
    w: &MetalTensor,
    bias: &MetalTensor,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    out: &MetalTensor,
    split_k_scratch: Option<&MetalTensor>,
    requirements: &gemm::GemmRequirements,
) -> crate::err::Res<()> {
    if requirements.shape.batch != batch
        || requirements.shape.m != m
        || requirements.shape.n != n
        || requirements.shape.k != k
        || !requirements.has_bias
    {
        return Err("gemm_bias: supplied requirements do not match exact dimensions".to_string());
    }
    gemm::gemm_into(
        MetalDevice::get(),
        x,
        w,
        Some(bias),
        out,
        split_k_scratch,
        m * k,
        0,
        requirements,
    )
}

/// `x @ w + bias` over the last two dimensions of `x` (leading dims
/// flatten to the gemm batch), restoring the original leading shape.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(tensor: &MetalTensor) -> Vec<u8> {
        let size = tensor.dtype.size_in_bytes();
        let offset = tensor.layout.offset() * size;
        // SAFETY: tests call this only after `dev.synchronize()`; the
        // layout's byte range fits the shared-mode buffer.
        unsafe {
            std::slice::from_raw_parts(
                tensor.buffer.contents_ptr().cast::<u8>().add(offset),
                tensor.numel() * size,
            )
            .to_vec()
        }
    }

    #[test]
    fn operation_wrappers_match_into_apis_and_exact_scratch() {
        let dev = MetalDevice::get();
        let a = MetalTensor::from_f32(dev, vec![1.0, -2.0, 3.0, -4.0], vec![2, 2]);
        let b = MetalTensor::from_f32(dev, vec![10.0, 20.0], vec![1, 2]);

        let binary_wrapped = binary(&a, &b, BinOp::Add).unwrap();
        let binary_destination = MetalTensor::empty(dev, vec![2, 2], DType::F32);
        precompile_binary(&a, &b, BinOp::Add).unwrap();
        binary_into(&a, &b, BinOp::Add, &binary_destination).unwrap();

        let half = kernels::cast(dev, &a, DType::F16).unwrap();
        let unary_wrapped = unary_promote(&half, UnOp::Abs).unwrap();
        let unary_requirements = unary_promote_scratch_requirements(&half);
        assert_eq!(unary_requirements.len(), 1);
        assert_eq!(unary_requirements[0].bytes().unwrap(), 4 * 4);
        let unary_scratch = allocate_scratch(&unary_requirements);
        let unary_scratch_refs = unary_scratch.iter().collect::<Vec<_>>();
        let unary_destination = MetalTensor::empty(dev, vec![2, 2], DType::F16);
        precompile_unary_promote(&half, UnOp::Abs).unwrap();
        unary_promote_into(&half, UnOp::Abs, &unary_destination, &unary_scratch_refs).unwrap();

        let compare_wrapped = compare(&half, &a, BinOp::Gt).unwrap();
        let compare_requirements = compare_scratch_requirements(&half, &a).unwrap();
        let compare_scratch = allocate_scratch(&compare_requirements);
        let compare_scratch_refs = compare_scratch.iter().collect::<Vec<_>>();
        let compare_destination = MetalTensor::empty(dev, vec![2, 2], DType::U8);
        precompile_compare(&half, &a, BinOp::Gt).unwrap();
        compare_into(
            &half,
            &a,
            BinOp::Gt,
            &compare_destination,
            &compare_scratch_refs,
        )
        .unwrap();

        dev.synchronize().unwrap();
        assert_eq!(bytes(&binary_wrapped), bytes(&binary_destination));
        assert_eq!(bytes(&unary_wrapped), bytes(&unary_destination));
        assert_eq!(bytes(&compare_wrapped), bytes(&compare_destination));
    }
}

//! Elementwise expression IR and single-kernel fusion (RFC 0007).
//!
//! A fused region is a small scalar expression DAG over named input lanes.
//! On Metal it is lowered to a `ug` SSA kernel, compiled once per distinct
//! expression (cached) and launched over the flattened input buffers; on
//! CPU the same IR is interpreted per element in a single pass. GPU fusion
//! is f32-only (the `ug` SSA has no f64 constants and Metal has no f64);
//! the CPU interpreter covers f32 and f64.

use candle_core::{DType, Device, Storage, Tensor};
#[cfg(any(target_os = "macos", feature = "cuda"))]
use std::collections::HashMap;

const BLOCK: usize = 256;

/// Row-major contiguous strides of `shape`, in elements.
pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// Strides with which a lane of shape `lane` is read when the region's
/// output has shape `out`: right-aligned broadcasting, where a lane dim of
/// 1 (or a missing leading dim) gets stride 0. `None` when the shapes are
/// not broadcast-compatible.
pub fn lane_strides(lane: &[usize], out: &[usize]) -> Option<Vec<usize>> {
    if lane.len() > out.len() {
        return None;
    }
    let offset = out.len() - lane.len();
    let own = contiguous_strides(lane);
    let mut strides = vec![0usize; out.len()];
    for (i, &ld) in lane.iter().enumerate() {
        let od = out[offset + i];
        if ld == od {
            strides[offset + i] = own[i];
        } else if ld == 1 {
            strides[offset + i] = 0;
        } else {
            return None;
        }
    }
    Some(strides)
}

/// Whether `lane` broadcasts into `out` (right-aligned, dims equal or 1).
pub fn broadcast_compatible(lane: &[usize], out: &[usize]) -> bool {
    lane_strides(lane, out).is_some()
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Expr {
    // per-element input lane k
    Input(u32),
    // scalar input k: a one-element tensor read at offset 0. Used for
    // values that vary between launches (step counts, scheduled learning
    // rates) so they do not poison the compiled-kernel cache key.
    Scalar(u32),
    // f64 bits, so the IR is Eq + Hash and can key the pipeline cache
    Const(u64),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Min(Box<Expr>, Box<Expr>),
    Max(Box<Expr>, Box<Expr>),
    // Comparisons yield 1.0 / 0.0; they exist to feed Select.
    Lt(Box<Expr>, Box<Expr>),
    Le(Box<Expr>, Box<Expr>),
    Gt(Box<Expr>, Box<Expr>),
    Ge(Box<Expr>, Box<Expr>),
    Eq(Box<Expr>, Box<Expr>),
    Ne(Box<Expr>, Box<Expr>),
    // cond != 0 ? lhs : rhs — a true select that does not propagate NaN
    // from the unselected side (unlike an arithmetic mask).
    Select(Box<Expr>, Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Sqrt(Box<Expr>),
    Exp(Box<Expr>),
    Sin(Box<Expr>),
    Cos(Box<Expr>),
    Tanh(Box<Expr>),
    Abs(Box<Expr>),
    Log(Box<Expr>),
    Floor(Box<Expr>),
    Ceil(Box<Expr>),
    Round(Box<Expr>),
    // constant exponent (f64 bits, keeping the IR Eq + Hash). Common
    // exponents lower to multiplies/sqrt; the rest lower to the platform
    // pow.
    Powf(Box<Expr>, u64),
    // Exact in the CPU interpreter; lowered to a stable expansion in ug
    // ops for GPU kernels (Metal has no erf).
    Erf(Box<Expr>),
}

/// `x^e` with special cases for the common exponents: exact multiplies
/// and sqrt are faster and more accurate than the platform pow.
pub fn pow_expr(x: Expr, e: f64) -> Expr {
    match e {
        0.0 => Expr::cst(1.0),
        1.0 => x,
        -1.0 => Expr::Div(Box::new(Expr::cst(1.0)), Box::new(x)),
        2.0 => Expr::Mul(Box::new(x.clone()), Box::new(x)),
        3.0 => Expr::Mul(
            Box::new(Expr::Mul(Box::new(x.clone()), Box::new(x.clone()))),
            Box::new(x),
        ),
        0.5 => Expr::Sqrt(Box::new(x)),
        -0.5 => Expr::Div(Box::new(Expr::cst(1.0)), Box::new(Expr::Sqrt(Box::new(x)))),
        _ => Expr::Powf(Box::new(x), e.to_bits()),
    }
}

/// The reduce that terminates a fused-reduce region (RFC 0007 phase 3a):
/// the elementwise expression is evaluated per input element inside the
/// reduce loop and folded into an accumulator, so the chain's
/// intermediate never materializes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
}

impl ReduceOp {
    fn init(&self) -> f64 {
        match self {
            ReduceOp::Sum | ReduceOp::Mean => 0.0,
            ReduceOp::Max => f64::NEG_INFINITY,
            ReduceOp::Min => f64::INFINITY,
        }
    }

    fn fold<T: Scalar>(&self, acc: T, v: T) -> T {
        match self {
            ReduceOp::Sum | ReduceOp::Mean => acc.add(v),
            ReduceOp::Max => acc.max(v),
            ReduceOp::Min => acc.min(v),
        }
    }
}

impl Expr {
    pub fn cst(v: f64) -> Self {
        Expr::Const(v.to_bits())
    }

    /// Remaps per-element lane indices through `remap` (which must cover
    /// every lane the expression references).
    pub fn remap_lanes(&self, remap: &std::collections::HashMap<u32, u32>) -> Self {
        self.remap_inputs(&mut |k| remap[&k])
    }

    /// Number of nodes in the expression tree (shared subtrees count per
    /// occurrence: this bounds emitted kernel size, not SSA values).
    pub fn ops(&self) -> usize {
        match self {
            Expr::Input(_) | Expr::Scalar(_) | Expr::Const(_) => 1,
            Expr::Select(c, a, b) => 1 + c.ops() + a.ops() + b.ops(),
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Min(a, b)
            | Expr::Max(a, b)
            | Expr::Lt(a, b)
            | Expr::Le(a, b)
            | Expr::Gt(a, b)
            | Expr::Ge(a, b)
            | Expr::Eq(a, b)
            | Expr::Ne(a, b) => 1 + a.ops() + b.ops(),
            Expr::Neg(a)
            | Expr::Sqrt(a)
            | Expr::Exp(a)
            | Expr::Sin(a)
            | Expr::Cos(a)
            | Expr::Tanh(a)
            | Expr::Abs(a)
            | Expr::Log(a)
            | Expr::Floor(a)
            | Expr::Ceil(a)
            | Expr::Round(a)
            | Expr::Powf(a, _)
            | Expr::Erf(a) => 1 + a.ops(),
        }
    }

    fn remap_inputs(&self, f: &mut dyn FnMut(u32) -> u32) -> Self {
        match self {
            Expr::Input(k) => Expr::Input(f(*k)),
            Expr::Scalar(k) => Expr::Scalar(*k),
            Expr::Const(b) => Expr::Const(*b),
            Expr::Add(a, b) => Expr::Add(Box::new(a.remap_inputs(f)), Box::new(b.remap_inputs(f))),
            Expr::Sub(a, b) => Expr::Sub(Box::new(a.remap_inputs(f)), Box::new(b.remap_inputs(f))),
            Expr::Mul(a, b) => Expr::Mul(Box::new(a.remap_inputs(f)), Box::new(b.remap_inputs(f))),
            Expr::Div(a, b) => Expr::Div(Box::new(a.remap_inputs(f)), Box::new(b.remap_inputs(f))),
            Expr::Min(a, b) => Expr::Min(Box::new(a.remap_inputs(f)), Box::new(b.remap_inputs(f))),
            Expr::Max(a, b) => Expr::Max(Box::new(a.remap_inputs(f)), Box::new(b.remap_inputs(f))),
            Expr::Lt(a, b) => Expr::Lt(Box::new(a.remap_inputs(f)), Box::new(b.remap_inputs(f))),
            Expr::Le(a, b) => Expr::Le(Box::new(a.remap_inputs(f)), Box::new(b.remap_inputs(f))),
            Expr::Gt(a, b) => Expr::Gt(Box::new(a.remap_inputs(f)), Box::new(b.remap_inputs(f))),
            Expr::Ge(a, b) => Expr::Ge(Box::new(a.remap_inputs(f)), Box::new(b.remap_inputs(f))),
            Expr::Eq(a, b) => Expr::Eq(Box::new(a.remap_inputs(f)), Box::new(b.remap_inputs(f))),
            Expr::Ne(a, b) => Expr::Ne(Box::new(a.remap_inputs(f)), Box::new(b.remap_inputs(f))),
            Expr::Select(c, a, b) => Expr::Select(
                Box::new(c.remap_inputs(f)),
                Box::new(a.remap_inputs(f)),
                Box::new(b.remap_inputs(f)),
            ),
            Expr::Neg(a) => Expr::Neg(Box::new(a.remap_inputs(f))),
            Expr::Sqrt(a) => Expr::Sqrt(Box::new(a.remap_inputs(f))),
            Expr::Exp(a) => Expr::Exp(Box::new(a.remap_inputs(f))),
            Expr::Sin(a) => Expr::Sin(Box::new(a.remap_inputs(f))),
            Expr::Cos(a) => Expr::Cos(Box::new(a.remap_inputs(f))),
            Expr::Tanh(a) => Expr::Tanh(Box::new(a.remap_inputs(f))),
            Expr::Abs(a) => Expr::Abs(Box::new(a.remap_inputs(f))),
            Expr::Log(a) => Expr::Log(Box::new(a.remap_inputs(f))),
            Expr::Floor(a) => Expr::Floor(Box::new(a.remap_inputs(f))),
            Expr::Ceil(a) => Expr::Ceil(Box::new(a.remap_inputs(f))),
            Expr::Round(a) => Expr::Round(Box::new(a.remap_inputs(f))),
            Expr::Powf(a, e) => Expr::Powf(Box::new(a.remap_inputs(f)), *e),
            Expr::Erf(a) => Expr::Erf(Box::new(a.remap_inputs(f))),
        }
    }

    /// Inlines `replacement` for `lane` and remaps the remaining lanes
    /// through `remap` (used by the multi-output merge to absorb a shared
    /// prefix into its continuations). Each occurrence is decided by its
    /// original index in a single pass, so a remapped index that collides
    /// with `lane` cannot be mistaken for the inlined one. The
    /// replacement's own indices must already be in the merged namespace.
    pub fn merge_lane(
        &self,
        lane: u32,
        replacement: &Expr,
        remap: &std::collections::HashMap<u32, u32>,
    ) -> Self {
        fn go(e: &Expr, lane: u32, r: &Expr, remap: &std::collections::HashMap<u32, u32>) -> Expr {
            match e {
                Expr::Input(k) if *k == lane => r.clone(),
                Expr::Input(k) => Expr::Input(remap[k]),
                Expr::Scalar(k) => Expr::Scalar(*k),
                Expr::Const(b) => Expr::Const(*b),
                Expr::Add(a, b) => Expr::Add(Box::new(go(a, lane, r, remap)), Box::new(go(b, lane, r, remap))),
                Expr::Sub(a, b) => Expr::Sub(Box::new(go(a, lane, r, remap)), Box::new(go(b, lane, r, remap))),
                Expr::Mul(a, b) => Expr::Mul(Box::new(go(a, lane, r, remap)), Box::new(go(b, lane, r, remap))),
                Expr::Div(a, b) => Expr::Div(Box::new(go(a, lane, r, remap)), Box::new(go(b, lane, r, remap))),
                Expr::Min(a, b) => Expr::Min(Box::new(go(a, lane, r, remap)), Box::new(go(b, lane, r, remap))),
                Expr::Max(a, b) => Expr::Max(Box::new(go(a, lane, r, remap)), Box::new(go(b, lane, r, remap))),
                Expr::Lt(a, b) => Expr::Lt(Box::new(go(a, lane, r, remap)), Box::new(go(b, lane, r, remap))),
                Expr::Le(a, b) => Expr::Le(Box::new(go(a, lane, r, remap)), Box::new(go(b, lane, r, remap))),
                Expr::Gt(a, b) => Expr::Gt(Box::new(go(a, lane, r, remap)), Box::new(go(b, lane, r, remap))),
                Expr::Ge(a, b) => Expr::Ge(Box::new(go(a, lane, r, remap)), Box::new(go(b, lane, r, remap))),
                Expr::Eq(a, b) => Expr::Eq(Box::new(go(a, lane, r, remap)), Box::new(go(b, lane, r, remap))),
                Expr::Ne(a, b) => Expr::Ne(Box::new(go(a, lane, r, remap)), Box::new(go(b, lane, r, remap))),
                Expr::Select(c, a, b) => Expr::Select(
                    Box::new(go(c, lane, r, remap)),
                    Box::new(go(a, lane, r, remap)),
                    Box::new(go(b, lane, r, remap)),
                ),
                Expr::Neg(a) => Expr::Neg(Box::new(go(a, lane, r, remap))),
                Expr::Sqrt(a) => Expr::Sqrt(Box::new(go(a, lane, r, remap))),
                Expr::Exp(a) => Expr::Exp(Box::new(go(a, lane, r, remap))),
                Expr::Sin(a) => Expr::Sin(Box::new(go(a, lane, r, remap))),
                Expr::Cos(a) => Expr::Cos(Box::new(go(a, lane, r, remap))),
                Expr::Tanh(a) => Expr::Tanh(Box::new(go(a, lane, r, remap))),
                Expr::Abs(a) => Expr::Abs(Box::new(go(a, lane, r, remap))),
                Expr::Log(a) => Expr::Log(Box::new(go(a, lane, r, remap))),
                Expr::Floor(a) => Expr::Floor(Box::new(go(a, lane, r, remap))),
                Expr::Ceil(a) => Expr::Ceil(Box::new(go(a, lane, r, remap))),
                Expr::Round(a) => Expr::Round(Box::new(go(a, lane, r, remap))),
                Expr::Powf(a, e) => Expr::Powf(Box::new(go(a, lane, r, remap)), *e),
                Expr::Erf(a) => Expr::Erf(Box::new(go(a, lane, r, remap))),
            }
        }
        go(self, lane, replacement, remap)
    }

}

pub trait Scalar: Copy {
    fn from_f64(v: f64) -> Self;
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul(self, o: Self) -> Self;
    fn div(self, o: Self) -> Self;
    fn min(self, o: Self) -> Self;
    fn max(self, o: Self) -> Self;
    fn neg(self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tanh(self) -> Self;
    fn abs(self) -> Self;
    fn log(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn powf(self, e: f64) -> Self;
    fn erf(self) -> Self;
    fn pick(cond: Self, lhs: Self, rhs: Self) -> Self;
    fn lt(self, o: Self) -> Self;
    fn le(self, o: Self) -> Self;
    fn gt(self, o: Self) -> Self;
    fn ge(self, o: Self) -> Self;
    fn eq(self, o: Self) -> Self;
    fn ne(self, o: Self) -> Self;
}

macro_rules! impl_scalar {
    ($ty:ty, $erf:path) => {
        impl Scalar for $ty {
            fn from_f64(v: f64) -> Self {
                v as $ty
            }
            fn add(self, o: Self) -> Self {
                self + o
            }
            fn sub(self, o: Self) -> Self {
                self - o
            }
            fn mul(self, o: Self) -> Self {
                self * o
            }
            fn div(self, o: Self) -> Self {
                self / o
            }
            fn min(self, o: Self) -> Self {
                self.min(o)
            }
            fn max(self, o: Self) -> Self {
                self.max(o)
            }
            fn neg(self) -> Self {
                -self
            }
            fn sqrt(self) -> Self {
                self.sqrt()
            }
            fn exp(self) -> Self {
                self.exp()
            }
            fn sin(self) -> Self {
                self.sin()
            }
            fn cos(self) -> Self {
                self.cos()
            }
            fn tanh(self) -> Self {
                self.tanh()
            }
            fn abs(self) -> Self {
                self.abs()
            }
            fn log(self) -> Self {
                self.ln()
            }
            fn floor(self) -> Self {
                self.floor()
            }
            fn ceil(self) -> Self {
                self.ceil()
            }
            fn round(self) -> Self {
                self.round()
            }
            fn powf(self, e: f64) -> Self {
                self.powf(e as $ty)
            }
            fn erf(self) -> Self {
                $erf(self)
            }
            fn pick(cond: Self, lhs: Self, rhs: Self) -> Self {
                if cond != 0.0 as $ty { lhs } else { rhs }
            }
            fn lt(self, o: Self) -> Self {
                if self < o { 1.0 as $ty } else { 0.0 as $ty }
            }
            fn le(self, o: Self) -> Self {
                if self <= o { 1.0 as $ty } else { 0.0 as $ty }
            }
            fn gt(self, o: Self) -> Self {
                if self > o { 1.0 as $ty } else { 0.0 as $ty }
            }
            fn ge(self, o: Self) -> Self {
                if self >= o { 1.0 as $ty } else { 0.0 as $ty }
            }
            fn eq(self, o: Self) -> Self {
                if self == o { 1.0 as $ty } else { 0.0 as $ty }
            }
            fn ne(self, o: Self) -> Self {
                if self != o { 1.0 as $ty } else { 0.0 as $ty }
            }
        }
    };
}
impl_scalar!(f32, libm::erff);
impl_scalar!(f64, libm::erf);

fn eval_at<T: Scalar>(e: &Expr, i: usize, inputs: &[&[T]], scalars: &[T]) -> T {
    eval(e, &|k| inputs[k as usize][i], scalars)
}

// The scalar evaluator shared by the contiguous path (lane accessor reads
// lane[k][i]) and the strided path (lane values pre-gathered per element).
fn eval<T: Scalar, F: Fn(u32) -> T>(e: &Expr, lane: &F, scalars: &[T]) -> T {
    match e {
        Expr::Input(k) => lane(*k),
        Expr::Scalar(k) => scalars[*k as usize],
        Expr::Const(bits) => T::from_f64(f64::from_bits(*bits)),
        Expr::Add(a, b) => eval(a, lane, scalars).add(eval(b, lane, scalars)),
        Expr::Sub(a, b) => eval(a, lane, scalars).sub(eval(b, lane, scalars)),
        Expr::Mul(a, b) => eval(a, lane, scalars).mul(eval(b, lane, scalars)),
        Expr::Div(a, b) => eval(a, lane, scalars).div(eval(b, lane, scalars)),
        Expr::Min(a, b) => eval(a, lane, scalars).min(eval(b, lane, scalars)),
        Expr::Max(a, b) => eval(a, lane, scalars).max(eval(b, lane, scalars)),
        Expr::Lt(a, b) => eval(a, lane, scalars).lt(eval(b, lane, scalars)),
        Expr::Le(a, b) => eval(a, lane, scalars).le(eval(b, lane, scalars)),
        Expr::Gt(a, b) => eval(a, lane, scalars).gt(eval(b, lane, scalars)),
        Expr::Ge(a, b) => eval(a, lane, scalars).ge(eval(b, lane, scalars)),
        Expr::Eq(a, b) => eval(a, lane, scalars).eq(eval(b, lane, scalars)),
        Expr::Ne(a, b) => eval(a, lane, scalars).ne(eval(b, lane, scalars)),
        Expr::Select(c, a, b) => T::pick(
            eval(c, lane, scalars),
            eval(a, lane, scalars),
            eval(b, lane, scalars),
        ),
        Expr::Neg(a) => eval(a, lane, scalars).neg(),
        Expr::Sqrt(a) => eval(a, lane, scalars).sqrt(),
        Expr::Exp(a) => eval(a, lane, scalars).exp(),
        Expr::Sin(a) => eval(a, lane, scalars).sin(),
        Expr::Cos(a) => eval(a, lane, scalars).cos(),
        Expr::Tanh(a) => eval(a, lane, scalars).tanh(),
        Expr::Abs(a) => eval(a, lane, scalars).abs(),
        Expr::Log(a) => eval(a, lane, scalars).log(),
        Expr::Floor(a) => eval(a, lane, scalars).floor(),
        Expr::Ceil(a) => eval(a, lane, scalars).ceil(),
        Expr::Round(a) => eval(a, lane, scalars).round(),
        Expr::Powf(a, e) => eval(a, lane, scalars).powf(f64::from_bits(*e)),
        Expr::Erf(a) => eval(a, lane, scalars).erf(),
    }
}

fn cpu_input_slices<'a, T: candle_core::WithDType>(
    inputs: &'a [Tensor],
) -> candle_core::Result<Vec<(std::sync::RwLockReadGuard<'a, Storage>, usize)>> {
    inputs
        .iter()
        .map(|t| {
            let (storage, layout) = t.storage_and_layout();
            Ok((storage, layout.start_offset()))
        })
        .collect()
}

fn interpret_cpu<T: candle_core::WithDType + Scalar>(
    exprs: &[Expr],
    inputs: &[Tensor],
    strides: Option<&[Vec<usize>]>,
    scalars: &[Tensor],
    n: usize,
    shape: &[usize],
) -> candle_core::Result<Vec<Tensor>> {
    let guards = cpu_input_slices::<T>(inputs)?;
    let mut slices: Vec<&[T]> = Vec::with_capacity(guards.len());
    for ((storage, offset), input) in guards.iter().zip(inputs.iter()) {
        let cpu = match &**storage {
            Storage::Cpu(cpu) => cpu,
            _ => {
                return Err(candle_core::Error::Msg(
                    "fusion: expected CPU storage".to_string(),
                ))
            }
        };
        let data = cpu.as_slice::<T>()?;
        slices.push(&data[*offset..*offset + input.elem_count()]);
    }
    let scalar_guards = cpu_input_slices::<T>(scalars)?;
    let mut scalar_values: Vec<T> = Vec::with_capacity(scalar_guards.len());
    for (storage, offset) in &scalar_guards {
        let cpu = match &**storage {
            Storage::Cpu(cpu) => cpu,
            _ => {
                return Err(candle_core::Error::Msg(
                    "fusion: expected CPU storage".to_string(),
                ))
            }
        };
        scalar_values.push(cpu.as_slice::<T>()?[*offset]);
    }
    let contig = contiguous_strides(shape);
    let strided = match strides {
        Some(ss) if ss.iter().any(|s| s != &contig) => Some(ss),
        _ => None,
    };
    let mut outs: Vec<Vec<T>> = exprs
        .iter()
        .map(|_| vec![<T as Scalar>::from_f64(0.0); n])
        .collect();
    match strided {
        None => {
            for i in 0..n {
                for (out, expr) in outs.iter_mut().zip(exprs.iter()) {
                    out[i] = eval_at(expr, i, &slices, &scalar_values);
                }
            }
        }
        // Broadcast lanes: walk the output coordinates with an odometer so
        // each lane's element offset is incremental (no per-element div/mod).
        Some(ss) => {
            let rank = shape.len();
            let mut coord = vec![0usize; rank];
            let mut offs = vec![0usize; inputs.len()];
            let mut lane_vals = vec![<T as Scalar>::from_f64(0.0); inputs.len()];
            for i in 0..n {
                for (v, (slice, off)) in lane_vals.iter_mut().zip(slices.iter().zip(offs.iter())) {
                    *v = slice[*off];
                }
                for (out, expr) in outs.iter_mut().zip(exprs.iter()) {
                    out[i] = eval(expr, &|k| lane_vals[k as usize], &scalar_values);
                }
                for d in (0..rank).rev() {
                    coord[d] += 1;
                    for (off, s) in offs.iter_mut().zip(ss.iter()) {
                        *off += s[d];
                    }
                    if coord[d] < shape[d] {
                        break;
                    }
                    coord[d] = 0;
                    for (off, s) in offs.iter_mut().zip(ss.iter()) {
                        *off -= shape[d] * s[d];
                    }
                }
            }
        }
    }
    outs.into_iter()
        .map(|out| Tensor::from_vec(out, shape, &Device::Cpu))
        .collect()
}

// The fused-reduce interpreter: one pass over the input with an odometer
// walk, evaluating the expression once per input element and folding into
// the accumulator of its output element. Lane offsets and the output
// offset are maintained incrementally, so reduced dims (output stride 0)
// and broadcast lanes (lane stride 0) cost nothing.
#[allow(clippy::too_many_arguments)]
fn interpret_reduce_cpu<T: candle_core::WithDType + Scalar>(
    op: ReduceOp,
    expr: &Expr,
    inputs: &[Tensor],
    strides: &[Vec<usize>],
    in_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    out_shape: &[usize],
) -> candle_core::Result<Tensor> {
    let guards = cpu_input_slices::<T>(inputs)?;
    let mut slices: Vec<&[T]> = Vec::with_capacity(guards.len());
    for ((storage, offset), input) in guards.iter().zip(inputs.iter()) {
        let cpu = match &**storage {
            Storage::Cpu(cpu) => cpu,
            _ => {
                return Err(candle_core::Error::Msg(
                    "fusion: expected CPU storage".to_string(),
                ))
            }
        };
        let data = cpu.as_slice::<T>()?;
        slices.push(&data[*offset..*offset + input.elem_count()]);
    }
    let in_n: usize = in_shape.iter().product();
    let out_n: usize = out_shape.iter().product();
    // Output strides in input-dim space: reduced dims get stride 0;
    // non-reduced dims get their contiguous output stride (the out-dim
    // index is d itself with keepdims, or the compacted index without).
    let rank = in_shape.len();
    let contig_out = contiguous_strides(out_shape);
    let mut out_strides = vec![0usize; rank];
    let mut o = 0;
    for d in 0..rank {
        if dims.contains(&d) {
            continue;
        }
        let out_d = if keepdims { d } else { o };
        out_strides[d] = contig_out[out_d];
        o += 1;
    }
    let init = <T as Scalar>::from_f64(op.init());
    let mut acc = vec![init; out_n];
    let mut coord = vec![0usize; rank];
    let mut offs = vec![0usize; inputs.len()];
    let mut lane_vals = vec![init; inputs.len()];
    let mut out_off = 0usize;
    for _ in 0..in_n {
        for (v, (slice, off)) in lane_vals.iter_mut().zip(slices.iter().zip(offs.iter())) {
            *v = slice[*off];
        }
        let v = eval(expr, &|k| lane_vals[k as usize], &[]);
        acc[out_off] = op.fold(acc[out_off], v);
        for d in (0..rank).rev() {
            coord[d] += 1;
            for (off, s) in offs.iter_mut().zip(strides.iter()) {
                *off += s[d];
            }
            out_off += out_strides[d];
            if coord[d] < in_shape[d] {
                break;
            }
            coord[d] = 0;
            for (off, s) in offs.iter_mut().zip(strides.iter()) {
                *off -= in_shape[d] * s[d];
            }
            out_off -= in_shape[d] * out_strides[d];
        }
    }
    if op == ReduceOp::Mean {
        let extent: usize = dims.iter().map(|&d| in_shape[d]).product();
        let e = <T as Scalar>::from_f64(extent as f64);
        for a in acc.iter_mut() {
            *a = Scalar::div(*a, e);
        }
    }
    Tensor::from_vec(acc, out_shape, &Device::Cpu)
}

#[cfg(any(target_os = "macos", feature = "cuda"))]
fn f32_cst(v: f32) -> candle_core::Result<ug::Const> {
    v.try_into()
        .map_err(|e| candle_core::Error::Msg(format!("fusion: {e}")))
}

#[cfg(any(target_os = "macos", feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
fn lower_expr(
    e: &Expr,
    b: &mut ug::block::Block,
    lanes: &[ug::block::Id],
    num_inputs: usize,
    lowered_lanes: &mut HashMap<u32, ug::block::Id>,
    lane_offsets: &[ug::block::Id],
    zero: ug::block::Id,
    dtype: ug::DType,
) -> candle_core::Result<ug::block::Id> {
    use ug::lang::ssa::{BinaryOp, Instr as I, UnaryOp};
    Ok(match e {
        Expr::Input(k) => {
            if let Some(id) = lowered_lanes.get(k) {
                *id
            } else {
                let id = b.push(I::Load {
                    src: lanes[*k as usize].to_varid(),
                    offset: lane_offsets[*k as usize].to_a(),
                    dtype,
                });
                lowered_lanes.insert(*k, id);
                id
            }
        }
        Expr::Scalar(k) => b.push(I::Load {
            src: lanes[num_inputs + *k as usize].to_varid(),
            offset: zero.to_a(),
            dtype,
        }),
        Expr::Const(bits) => {
            let v = f64::from_bits(*bits) as f32;
            let cst: ug::Const = v
                .try_into()
                .map_err(|e| candle_core::Error::Msg(format!("fusion: {e}")))?;
            b.push(I::Const(cst))
        }
        Expr::Add(a, x) => {
            let a = lower_expr(a, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            let x = lower_expr(x, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            b.binary(BinaryOp::Add, a, x, dtype)
        }
        Expr::Sub(a, x) => {
            let a = lower_expr(a, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            let x = lower_expr(x, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            b.binary(BinaryOp::Sub, a, x, dtype)
        }
        Expr::Mul(a, x) => {
            let a = lower_expr(a, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            let x = lower_expr(x, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            b.binary(BinaryOp::Mul, a, x, dtype)
        }
        Expr::Div(a, x) => {
            let a = lower_expr(a, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            let x = lower_expr(x, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            b.binary(BinaryOp::Div, a, x, dtype)
        }
        Expr::Min(a, x) => {
            let a = lower_expr(a, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            let x = lower_expr(x, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            b.binary(BinaryOp::Min, a, x, dtype)
        }
        Expr::Max(a, x) => {
            let a = lower_expr(a, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            let x = lower_expr(x, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            b.binary(BinaryOp::Max, a, x, dtype)
        }
        Expr::Lt(a, x) | Expr::Le(a, x) | Expr::Gt(a, x) | Expr::Ge(a, x) | Expr::Eq(a, x)
        | Expr::Ne(a, x) => {
            let op = match e {
                Expr::Lt(..) => BinaryOp::Lt,
                Expr::Le(..) => BinaryOp::Le,
                Expr::Gt(..) => BinaryOp::Gt,
                Expr::Ge(..) => BinaryOp::Ge,
                Expr::Eq(..) => BinaryOp::Eq,
                _ => BinaryOp::Ne,
            };
            let a = lower_expr(a, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            let x = lower_expr(x, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            b.binary(op, a, x, ug::DType::I32)
        }
        Expr::Select(c, a, x) => {
            let c = lower_expr(c, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            let a = lower_expr(a, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            let x = lower_expr(x, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            b.select(c, a, x, dtype)
        }
        Expr::Neg(a) => {
            let a = lower_expr(a, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            b.unary(UnaryOp::Neg, a, dtype)
        }
        Expr::Sqrt(a) => {
            let a = lower_expr(a, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            b.unary(UnaryOp::Sqrt, a, dtype)
        }
        Expr::Exp(a) => {
            let a = lower_expr(a, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            b.unary(UnaryOp::Exp, a, dtype)
        }
        Expr::Sin(a) => {
            let a = lower_expr(a, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            b.unary(UnaryOp::Sin, a, dtype)
        }
        Expr::Cos(a) => {
            let a = lower_expr(a, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            b.unary(UnaryOp::Cos, a, dtype)
        }
        Expr::Tanh(a) => {
            let x = lower_expr(a, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            b.unary(UnaryOp::Tanh, x, dtype)
        }
        Expr::Abs(a) => {
            let x = lower_expr(a, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            b.unary(UnaryOp::Abs, x, dtype)
        }
        Expr::Log(a) => {
            let x = lower_expr(a, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            b.unary(UnaryOp::Log, x, dtype)
        }
        Expr::Floor(a) => {
            let x = lower_expr(a, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            b.unary(UnaryOp::Floor, x, dtype)
        }
        Expr::Ceil(a) => {
            let x = lower_expr(a, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            b.unary(UnaryOp::Ceil, x, dtype)
        }
        Expr::Round(a) => {
            let x = lower_expr(a, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            b.unary(UnaryOp::Round, x, dtype)
        }
        Expr::Powf(a, e) => {
            let x = lower_expr(a, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            let exp = b.push(I::Const(f32_cst(f64::from_bits(*e) as f32)?));
            b.binary(BinaryOp::Pow, x, exp, dtype)
        }
        // Abramowitz & Stegun 7.1.26 (max error ~1.5e-7) with
        // sign(x) = x / max(|x|, 1e-30); the CPU interpreter uses the
        // exact libm erf instead
        Expr::Erf(a) => {
            let x = lower_expr(a, b, lanes, num_inputs, lowered_lanes, lane_offsets, zero, dtype)?;
            let c = |v: f64, b: &mut ug::block::Block| -> candle_core::Result<ug::block::Id> {
                Ok(b.push(I::Const(f32_cst(v as f32)?)))
            };
            let one = c(1.0, b)?;
            let ax = {
                let neg = b.unary(UnaryOp::Neg, x, dtype);
                b.binary(BinaryOp::Max, x, neg, dtype)
            };
            let t = {
                let k = c(0.3275911, b)?;
                let kx = b.binary(BinaryOp::Mul, k, ax, dtype);
                let denom = b.binary(BinaryOp::Add, one, kx, dtype);
                b.binary(BinaryOp::Div, one, denom, dtype)
            };
            let poly = {
                let a1 = c(0.254829592, b)?;
                let a2 = c(-0.284496736, b)?;
                let a3 = c(1.421413741, b)?;
                let a4 = c(-1.453152027, b)?;
                let a5 = c(1.061405429, b)?;
                let p = b.binary(BinaryOp::Mul, a5, t, dtype);
                let p = b.binary(BinaryOp::Add, p, a4, dtype);
                let p = b.binary(BinaryOp::Mul, p, t, dtype);
                let p = b.binary(BinaryOp::Add, p, a3, dtype);
                let p = b.binary(BinaryOp::Mul, p, t, dtype);
                let p = b.binary(BinaryOp::Add, p, a2, dtype);
                let p = b.binary(BinaryOp::Mul, p, t, dtype);
                b.binary(BinaryOp::Add, p, a1, dtype)
            };
            let tail = {
                let x2 = b.binary(BinaryOp::Mul, x, x, dtype);
                let nx2 = b.unary(UnaryOp::Neg, x2, dtype);
                let e = b.unary(UnaryOp::Exp, nx2, dtype);
                let pt = b.binary(BinaryOp::Mul, poly, t, dtype);
                let pte = b.binary(BinaryOp::Mul, pt, e, dtype);
                b.binary(BinaryOp::Sub, one, pte, dtype)
            };
            let sign = {
                let eps = c(1e-30, b)?;
                let denom = b.binary(BinaryOp::Max, ax, eps, dtype);
                b.binary(BinaryOp::Div, x, denom, dtype)
            };
            b.binary(BinaryOp::Mul, sign, tail, dtype)
        }
    })
}

// Lowers the IR to a ug SSA kernel with one store per output expression.
// Lane k reads from pointer argument k at an offset computed from the
// flattened output index and the lane's strides (broadcast lanes re-read
// elements; contiguous lanes read at the index itself). The element count
// and lane strides are baked in as constants (the pipeline cache is keyed
// by them): Metal kernel functions cannot take plain scalar arguments, and
// baking enables constant folding. Loads are clamped to n-1 so the
// trailing partial block recomputes the last element harmlessly instead of
// reading out of bounds.
#[cfg(any(target_os = "macos", feature = "cuda"))]
fn build_kernel(
    exprs: &[Expr],
    lane_strides: &[Vec<usize>],
    out_shape: &[usize],
    num_scalars: usize,
    n: usize,
    dtype: ug::DType,
) -> candle_core::Result<ug::lang::ssa::Kernel> {
    use ug::lang::ssa::{BinaryOp, Instr as I, Special};

    let num_inputs = lane_strides.len();
    let mut b = ug::block::Block::empty();
    let mut lanes = Vec::with_capacity(num_inputs + num_scalars);
    for k in 0..(num_inputs + num_scalars) {
        lanes.push(b.push(I::DefineGlobal { index: k, dtype }));
    }
    let num_outputs = exprs.len();
    let mut outs = Vec::with_capacity(num_outputs);
    for j in 0..num_outputs {
        outs.push(b.push(I::DefineGlobal {
            index: num_inputs + num_scalars + j,
            dtype,
        }));
    }

    let gi = b.push(I::Special(Special::BlockIdx));
    let ti = b.push(I::Special(Special::ThreadIdx));
    let off = b.mul(gi, BLOCK as i32);
    let off = b.binary(BinaryOp::Add, off, ti, ug::DType::I32);
    let last = b.push(I::Const(ug::Const::I32(n as i32 - 1)));
    let clamped = b.binary(BinaryOp::Min, off.to_a(), last.to_a(), ug::DType::I32);
    let zero = b.push(I::Const(ug::Const::I32(0)));

    // Per-lane read offsets. Contiguous lanes read at the clamped index;
    // broadcast lanes decompose it into coordinates and recombine with
    // their strides (stride 0 dims and size-1 dims contribute nothing).
    let i32c = |v: usize, b: &mut ug::block::Block| b.push(I::Const(ug::Const::I32(v as i32)));
    let contig = contiguous_strides(out_shape);
    let rank = out_shape.len();
    let mut lane_offsets: Vec<ug::block::Id> = Vec::with_capacity(num_inputs);
    for strides in lane_strides {
        if strides == &contig {
            lane_offsets.push(clamped);
            continue;
        }
        let mut acc: Option<ug::block::Id> = None;
        for d in 0..rank {
            if out_shape[d] == 1 || strides[d] == 0 {
                continue;
            }
            let coord = if d == rank - 1 {
                let dim = i32c(out_shape[d], &mut b);
                b.binary(BinaryOp::Mod, clamped, dim.to_a(), ug::DType::I32)
            } else {
                let stride = i32c(contig[d], &mut b);
                let q = b.binary(BinaryOp::Div, clamped, stride.to_a(), ug::DType::I32);
                let dim = i32c(out_shape[d], &mut b);
                b.binary(BinaryOp::Mod, q.to_a(), dim.to_a(), ug::DType::I32)
            };
            let term = if strides[d] == 1 {
                coord
            } else {
                let s = i32c(strides[d], &mut b);
                b.binary(BinaryOp::Mul, coord.to_a(), s.to_a(), ug::DType::I32)
            };
            acc = Some(match acc {
                None => term,
                Some(prev) => b.binary(BinaryOp::Add, prev.to_a(), term.to_a(), ug::DType::I32),
            });
        }
        // A lane with no contributing dims (a broadcast scalar) reads offset 0.
        lane_offsets.push(acc.unwrap_or(zero));
    }

    let mut lowered_lanes: HashMap<u32, ug::block::Id> = HashMap::new();

    for (j, expr) in exprs.iter().enumerate() {
        let value = lower_expr(expr, &mut b, &lanes, num_inputs, &mut lowered_lanes, &lane_offsets, zero, dtype)?;
        b.push(I::Store {
            dst: outs[j].to_varid(),
            offset: off.to_a(),
            value: value.to_a(),
            dtype,
        });
    }
    let instrs = b
        .relocate()
        .map_err(|e| candle_core::Error::Msg(format!("fusion: {e}")))?;
    ug::lang::ssa::Kernel::from_instrs(instrs)
        .map_err(|e| candle_core::Error::Msg(format!("fusion: {e}")))
}

// Lowers a fused-reduce region to a ug SSA kernel: one thread per output
// element decomposes its flat index into the non-reduced coordinates
// (the loop-invariant base offset of each lane), then walks a Range loop
// over the reduce extent, recomputing per-lane offsets from the reduce
// coordinates and folding the expression value into an accumulator.
// Everything shape-dependent is baked in as constants (keying the
// pipeline cache): dim sizes, lane strides, the extent.
#[cfg(any(target_os = "macos", feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
fn build_reduce_kernel(
    op: ReduceOp,
    expr: &Expr,
    lane_strides: &[Vec<usize>],
    in_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    out_shape: &[usize],
    out_n: usize,
    dtype: ug::DType,
) -> candle_core::Result<ug::lang::ssa::Kernel> {
    use ug::lang::ssa::{BinaryOp, Instr as I, Special};

    let rank = in_shape.len();
    let num_inputs = lane_strides.len();
    let mut b = ug::block::Block::empty();
    let mut lanes = Vec::with_capacity(num_inputs + 1);
    for k in 0..=num_inputs {
        lanes.push(b.push(I::DefineGlobal { index: k, dtype }));
    }
    let out = lanes[num_inputs];

    let gi = b.push(I::Special(Special::BlockIdx));
    let ti = b.push(I::Special(Special::ThreadIdx));
    let off = b.mul(gi, BLOCK as i32);
    let off = b.binary(BinaryOp::Add, off, ti, ug::DType::I32);
    let last = b.push(I::Const(ug::Const::I32(out_n as i32 - 1)));
    let clamped = b.binary(BinaryOp::Min, off.to_a(), last.to_a(), ug::DType::I32);
    let zero = b.push(I::Const(ug::Const::I32(0)));

    let i32c = |v: usize, b: &mut ug::block::Block| b.push(I::Const(ug::Const::I32(v as i32)));
    // Loop-invariant per-lane base offsets from the non-reduced
    // coordinates of the (clamped) flat output index.
    let contig_out = contiguous_strides(out_shape);
    let mut base_offsets: Vec<ug::block::Id> = Vec::with_capacity(num_inputs);
    for strides in lane_strides {
        let mut acc: Option<ug::block::Id> = None;
        let mut o = 0;
        for d in 0..rank {
            if dims.contains(&d) {
                continue;
            }
            let out_d = if keepdims { d } else { o };
            o += 1;
            if out_shape[out_d] == 1 || strides[d] == 0 {
                continue;
            }
            let coord = if out_d == out_shape.len() - 1 {
                let dim = i32c(out_shape[out_d], &mut b);
                b.binary(BinaryOp::Mod, clamped, dim.to_a(), ug::DType::I32)
            } else {
                let stride = i32c(contig_out[out_d], &mut b);
                let q = b.binary(BinaryOp::Div, clamped, stride.to_a(), ug::DType::I32);
                let dim = i32c(out_shape[out_d], &mut b);
                b.binary(BinaryOp::Mod, q.to_a(), dim.to_a(), ug::DType::I32)
            };
            let term = if strides[d] == 1 {
                coord
            } else {
                let s = i32c(strides[d], &mut b);
                b.binary(BinaryOp::Mul, coord.to_a(), s.to_a(), ug::DType::I32)
            };
            acc = Some(match acc {
                None => term,
                Some(prev) => b.binary(BinaryOp::Add, prev.to_a(), term.to_a(), ug::DType::I32),
            });
        }
        base_offsets.push(acc.unwrap_or(zero));
    }

    // The reduce loop: r walks the extent; its decomposition into reduce
    // coordinates (row-major over the reduced dims) advances each lane's
    // offset by coord * stride along the reduced dims.
    let red_sizes: Vec<usize> = dims.iter().map(|&d| in_shape[d]).collect();
    let red_contig = contiguous_strides(&red_sizes);
    let extent: usize = red_sizes.iter().product();
    let init = match op {
        ReduceOp::Sum | ReduceOp::Mean => 0.0f32,
        ReduceOp::Max => f32::NEG_INFINITY,
        ReduceOp::Min => f32::INFINITY,
    };
    let fold_op = match op {
        ReduceOp::Sum | ReduceOp::Mean => BinaryOp::Add,
        ReduceOp::Max => BinaryOp::Max,
        ReduceOp::Min => BinaryOp::Min,
    };
    let acc = b.push(I::DefineAcc(f32_cst(init)?));
    let range = b.range(0, extent as i32, 1);
    let r = range.id();
    let mut lane_offsets = base_offsets.clone();
    for (j, &d) in dims.iter().enumerate() {
        if red_sizes[j] == 1 {
            continue;
        }
        let rcoord = if j == dims.len() - 1 {
            let dim = i32c(red_sizes[j], &mut b);
            b.binary(BinaryOp::Mod, r, dim.to_a(), ug::DType::I32)
        } else {
            let stride = i32c(red_contig[j], &mut b);
            let q = b.binary(BinaryOp::Div, r, stride.to_a(), ug::DType::I32);
            let dim = i32c(red_sizes[j], &mut b);
            b.binary(BinaryOp::Mod, q.to_a(), dim.to_a(), ug::DType::I32)
        };
        for (k, strides) in lane_strides.iter().enumerate() {
            if strides[d] == 0 {
                continue;
            }
            let term = if strides[d] == 1 {
                rcoord
            } else {
                let s = i32c(strides[d], &mut b);
                b.binary(BinaryOp::Mul, rcoord.to_a(), s.to_a(), ug::DType::I32)
            };
            lane_offsets[k] =
                b.binary(BinaryOp::Add, lane_offsets[k].to_a(), term.to_a(), ug::DType::I32);
        }
    }
    let mut lowered_lanes: HashMap<u32, ug::block::Id> = HashMap::new();
    let v = lower_expr(
        expr,
        &mut b,
        &lanes,
        num_inputs,
        &mut lowered_lanes,
        &lane_offsets,
        zero,
        dtype,
    )?;
    let folded = b.binary(fold_op, acc.to_a(), v.to_a(), dtype);
    b.push(I::Assign {
        dst: acc.to_varid(),
        src: folded.to_a(),
    });
    b.end_range(range)
        .map_err(|e| candle_core::Error::Msg(format!("fusion: {e}")))?;

    let value = if op == ReduceOp::Mean {
        let e = b.push(I::Const(f32_cst(extent as f32)?));
        b.binary(BinaryOp::Div, acc.to_a(), e.to_a(), dtype)
    } else {
        acc
    };
    b.push(I::Store {
        dst: out.to_varid(),
        offset: off.to_a(),
        value: value.to_a(),
        dtype,
    });
    let instrs = b
        .relocate()
        .map_err(|e| candle_core::Error::Msg(format!("fusion: {e}")))?;
    ug::lang::ssa::Kernel::from_instrs(instrs)
        .map_err(|e| candle_core::Error::Msg(format!("fusion: {e}")))
}

// Env-gated phase accounting for fusion::run (EFFECT_TORCH_FUSION_TIMING):
// [hash, pipeline-cache+compile, buffers+encoder, bind+dispatch]. Prints
// aggregates every 2048 calls.
#[cfg(target_os = "macos")]
fn fusion_phase_nanos(phase: usize, nanos: u64) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static PHASES: [AtomicU64; 4] = [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ];
    static CALLS: AtomicU64 = AtomicU64::new(0);
    const NAMES: [&str; 4] = ["hash", "pipeline", "buffers", "dispatch"];
    PHASES[phase].fetch_add(nanos, Ordering::Relaxed);
    let calls = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if phase == 3 && calls % 2048 == 0 {
        let totals: Vec<u64> = PHASES.iter().map(|p| p.load(Ordering::Relaxed)).collect();
        eprintln!(
            "[fusion-timing] {calls} calls: {}",
            NAMES
                .iter()
                .zip(totals.iter())
                .map(|(n, t)| format!("{n} {:.1}ms", *t as f64 / 1e6))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn fusion_phase_nanos(_phase: usize, _nanos: u64) {}

#[cfg(target_os = "macos")]
mod metal {
    use super::{build_kernel, build_reduce_kernel, fusion_phase_nanos, Expr, ReduceOp, BLOCK};
    use candle_core::{DType, Device, MetalStorage, Storage, Tensor};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    fn pipelines() -> &'static Mutex<HashMap<u64, candle_metal_kernels::metal::ComputePipeline>> {
        static CACHE: OnceLock<Mutex<HashMap<u64, candle_metal_kernels::metal::ComputePipeline>>> =
            OnceLock::new();
        CACHE.get_or_init(|| Mutex::new(HashMap::new()))
    }

    pub fn run(
        exprs: &[Expr],
        inputs: &[Tensor],
        lane_strides: &[Vec<usize>],
        scalars: &[Tensor],
        n: usize,
        shape: &[usize],
        device: &Device,
    ) -> candle_core::Result<Vec<Tensor>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mdev = device.as_metal_device()?;
        let phase_timing = std::env::var_os("EFFECT_TORCH_FUSION_TIMING").is_some();
        let t_start = std::time::Instant::now();
        // Metal exposes at most 31 buffer argument slots per kernel.
        if inputs.len() + scalars.len() + exprs.len() > 31 {
            return Err(candle_core::Error::Msg(format!(
                "fusion: {} buffer arguments exceed Metal's limit of 31",
                inputs.len() + scalars.len() + exprs.len()
            )));
        }
        let mut hasher = DefaultHasher::new();
        exprs.hash(&mut hasher);
        inputs.len().hash(&mut hasher);
        scalars.len().hash(&mut hasher);
        n.hash(&mut hasher);
        // Strides are baked into the kernel as constants, so they must key
        // the pipeline cache.
        shape.hash(&mut hasher);
        lane_strides.hash(&mut hasher);
        let key = hasher.finish();
        if phase_timing {
            fusion_phase_nanos(0, t_start.elapsed().as_nanos() as u64);
        }
        let t_phase = std::time::Instant::now();
        let pipeline = {
            let mut cache = pipelines().lock().unwrap();
            match cache.get(&key) {
                Some(p) => p.clone(),
                None => {
                    let kernel =
                        build_kernel(exprs, lane_strides, shape, scalars.len(), n, ug::DType::F32)?;
                    let p = mdev.compile("effect_torch_fused", kernel)?;
                    if std::env::var_os("EFFECT_TORCH_FUSION_DEBUG").is_some() {
                        eprintln!("[fusion] compiled kernel #{} (key {key:x})", cache.len() + 1);
                    }
                    cache.insert(key, p.clone());
                    p
                }
            }
        };
        if phase_timing {
            fusion_phase_nanos(1, t_phase.elapsed().as_nanos() as u64);
        }
        let t_phase = std::time::Instant::now();
        let padded = n.div_ceil(BLOCK) * BLOCK;
        let mut out_bufs = Vec::with_capacity(exprs.len());
        for _ in exprs {
            out_bufs.push(mdev.new_buffer(padded.max(1), DType::F32, "fused")?);
        }
        let encoder = mdev.command_encoder()?;
        if phase_timing {
            fusion_phase_nanos(2, t_phase.elapsed().as_nanos() as u64);
        }
        let t_phase = std::time::Instant::now();
        encoder.set_compute_pipeline_state(&pipeline);
        for (i, t) in inputs.iter().chain(scalars.iter()).enumerate() {
            let (storage, layout) = t.storage_and_layout();
            let metal = match &*storage {
                Storage::Metal(m) => m,
                _ => {
                    return Err(candle_core::Error::Msg(
                        "fusion: expected Metal storage".to_string(),
                    ))
                }
            };
            encoder.set_buffer(
                i,
                Some(metal.buffer()),
                layout.start_offset() * DType::F32.size_in_bytes(),
            );
        }
        for (j, buf) in out_bufs.iter().enumerate() {
            encoder.set_buffer(inputs.len() + scalars.len() + j, Some(buf), 0);
        }
        encoder.dispatch_threads(
            objc2_metal::MTLSize {
                width: padded,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: BLOCK,
                height: 1,
                depth: 1,
            },
        );
        // no end_encoding: candle's Commands owns the encoder lifecycle
        if phase_timing {
            fusion_phase_nanos(3, t_phase.elapsed().as_nanos() as u64);
        }
        out_bufs
            .into_iter()
            .map(|buf| {
                Ok(Tensor::from_storage(
                    Storage::Metal(MetalStorage::new(buf, mdev.clone(), n, DType::F32)),
                    shape,
                    candle_core::op::BackpropOp::none(),
                    false,
                ))
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_reduce(
        op: ReduceOp,
        expr: &Expr,
        inputs: &[Tensor],
        lane_strides: &[Vec<usize>],
        in_shape: &[usize],
        dims: &[usize],
        keepdims: bool,
        out_shape: &[usize],
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mdev = device.as_metal_device()?;
        // inputs + the single output must fit Metal's 31 buffer slots.
        if inputs.len() + 1 > 31 {
            return Err(candle_core::Error::Msg(format!(
                "fusion: {} buffer arguments exceed Metal's limit of 31",
                inputs.len() + 1
            )));
        }
        let out_n: usize = out_shape.iter().product();
        let phase_timing = std::env::var_os("EFFECT_TORCH_FUSION_TIMING").is_some();
        let t_start = std::time::Instant::now();
        let mut hasher = DefaultHasher::new();
        op.hash(&mut hasher);
        expr.hash(&mut hasher);
        inputs.len().hash(&mut hasher);
        in_shape.hash(&mut hasher);
        dims.hash(&mut hasher);
        keepdims.hash(&mut hasher);
        out_shape.hash(&mut hasher);
        lane_strides.hash(&mut hasher);
        let key = hasher.finish();
        let pipeline = {
            let mut cache = pipelines().lock().unwrap();
            match cache.get(&key) {
                Some(p) => p.clone(),
                None => {
                    let kernel = build_reduce_kernel(
                        op,
                        expr,
                        lane_strides,
                        in_shape,
                        dims,
                        keepdims,
                        out_shape,
                        out_n,
                        ug::DType::F32,
                    )?;
                    let p = mdev.compile("effect_torch_fused_reduce", kernel)?;
                    if std::env::var_os("EFFECT_TORCH_FUSION_DEBUG").is_some() {
                        eprintln!("[fusion] compiled reduce kernel #{} (key {key:x})", cache.len() + 1);
                    }
                    cache.insert(key, p.clone());
                    p
                }
            }
        };
        if phase_timing {
            fusion_phase_nanos(0, t_start.elapsed().as_nanos() as u64);
        }
        let t_phase = std::time::Instant::now();
        let padded = out_n.div_ceil(BLOCK) * BLOCK;
        let out_buf = mdev.new_buffer(padded.max(1), DType::F32, "fused_reduce")?;
        let encoder = mdev.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        for (i, t) in inputs.iter().enumerate() {
            let (storage, layout) = t.storage_and_layout();
            let metal = match &*storage {
                Storage::Metal(m) => m,
                _ => {
                    return Err(candle_core::Error::Msg(
                        "fusion: expected Metal storage".to_string(),
                    ))
                }
            };
            encoder.set_buffer(
                i,
                Some(metal.buffer()),
                layout.start_offset() * DType::F32.size_in_bytes(),
            );
        }
        encoder.set_buffer(inputs.len(), Some(&out_buf), 0);
        encoder.dispatch_threads(
            objc2_metal::MTLSize {
                width: padded,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: BLOCK,
                height: 1,
                depth: 1,
            },
        );
        // no end_encoding: candle's Commands owns the encoder lifecycle
        if phase_timing {
            fusion_phase_nanos(2, t_phase.elapsed().as_nanos() as u64);
            fusion_phase_nanos(3, 0);
        }
        Ok(Tensor::from_storage(
            Storage::Metal(MetalStorage::new(out_buf, mdev.clone(), out_n, DType::F32)),
            out_shape,
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }
}

/// Whether `run` can execute directly on this device/dtype pair. Callers
/// keep their composed candle-op path as the fallback when this is false.
pub fn is_supported(device: &Device, dtype: DType) -> bool {
    match device {
        Device::Cpu => matches!(dtype, DType::F32 | DType::F64),
        #[cfg(target_os = "macos")]
        Device::Metal(_) => dtype == DType::F32,
        #[cfg(not(target_os = "macos"))]
        Device::Metal(_) => false,
        // CUDA is disabled until the ug-cuda path can be tested on real
        // hardware; the region pass treats these nodes as unfusable and
        // they keep their composed candle-op eval.
        Device::Cuda(_) => false,
    }
}

// The fused AdamW update as three expressions over lanes [param, grad, m,
// v] with scalar lanes [lr, 1 - beta1^t, 1 - beta2^t], mirroring the
// composed update's operation order exactly. Step-dependent values are
// scalar lanes so the compiled kernel is stable across steps.
pub fn adamw_exprs(beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> [Expr; 3] {
    let (p, g, m, v) = (Expr::Input(0), Expr::Input(1), Expr::Input(2), Expr::Input(3));
    let (lr, c1, c2) = (Expr::Scalar(0), Expr::Scalar(1), Expr::Scalar(2));
    let next_m = Expr::Add(
        Box::new(Expr::Mul(Box::new(m), Box::new(Expr::cst(beta1)))),
        Box::new(Expr::Mul(Box::new(g.clone()), Box::new(Expr::cst(1.0 - beta1)))),
    );
    let next_v = Expr::Add(
        Box::new(Expr::Mul(Box::new(v), Box::new(Expr::cst(beta2)))),
        Box::new(Expr::Mul(
            Box::new(Expr::Mul(Box::new(g.clone()), Box::new(g))),
            Box::new(Expr::cst(1.0 - beta2)),
        )),
    );
    let m_hat = Expr::Div(Box::new(next_m.clone()), Box::new(c1));
    let v_hat = Expr::Div(Box::new(next_v.clone()), Box::new(c2));
    let adjusted = Expr::Mul(
        Box::new(Expr::Div(
            Box::new(m_hat),
            Box::new(Expr::Add(
                Box::new(Expr::Sqrt(Box::new(v_hat))),
                Box::new(Expr::cst(eps)),
            )),
        )),
        Box::new(lr.clone()),
    );
    let base = if weight_decay == 0.0 {
        p
    } else {
        Expr::Mul(
            Box::new(p),
            Box::new(Expr::Sub(
                Box::new(Expr::cst(1.0)),
                Box::new(Expr::Mul(Box::new(lr), Box::new(Expr::cst(weight_decay)))),
            )),
        )
    };
    [Expr::Sub(Box::new(base), Box::new(adjusted)), next_m, next_v]
}

// The fused momentum-SGD update over lanes [param, grad, velocity] with
// scalar lanes [lr, first], mirroring the composed update including the
// first-step v = g initialization as a select on the 0-d `first` flag.
pub fn sgd_exprs(momentum: f64, dampening: f64, nesterov: bool, weight_decay: f64) -> [Expr; 2] {
    let (p, g, v) = (Expr::Input(0), Expr::Input(1), Expr::Input(2));
    let (lr, first) = (Expr::Scalar(0), Expr::Scalar(1));
    let gp = if weight_decay == 0.0 {
        g
    } else {
        Expr::Add(Box::new(g), Box::new(Expr::Mul(Box::new(p.clone()), Box::new(Expr::cst(weight_decay)))))
    };
    let continued = Expr::Add(
        Box::new(Expr::Mul(Box::new(v), Box::new(Expr::cst(momentum)))),
        Box::new(Expr::Mul(Box::new(gp.clone()), Box::new(Expr::cst(1.0 - dampening)))),
    );
    let next_v = Expr::Select(
        Box::new(Expr::Gt(Box::new(first), Box::new(Expr::cst(0.5)))),
        Box::new(gp.clone()),
        Box::new(continued),
    );
    let used = if nesterov {
        Expr::Add(
            Box::new(gp),
            Box::new(Expr::Mul(Box::new(next_v.clone()), Box::new(Expr::cst(momentum)))),
        )
    } else {
        next_v.clone()
    };
    [
        Expr::Sub(Box::new(p), Box::new(Expr::Mul(Box::new(used), Box::new(lr)))),
        next_v,
    ]
}

/// Evaluates each expression over the flattened inputs (broadcast to the
/// output element count) and returns one tensor per expression. `strides`
/// gives each input lane's strides in output-dim space (`None` means every
/// lane is contiguous and output-shaped). `scalars` are one-element tensors
/// read at offset 0.
pub fn run(
    exprs: &[Expr],
    inputs: &[Tensor],
    strides: Option<&[Vec<usize>]>,
    scalars: &[Tensor],
    n: usize,
    shape: &[usize],
    dtype: DType,
    device: &Device,
) -> candle_core::Result<Vec<Tensor>> {
    if let Some(ss) = strides {
        if ss.len() != inputs.len() {
            return Err(candle_core::Error::Msg(format!(
                "fusion: got {} stride entries for {} inputs",
                ss.len(),
                inputs.len()
            )));
        }
    }
    let mut owned = Vec::with_capacity(inputs.len() + scalars.len());
    for t in inputs.iter().chain(scalars.iter()) {
        owned.push(if t.is_contiguous() { t.clone() } else { t.contiguous()? });
    }
    for s in scalars.iter() {
        if s.elem_count() != 1 {
            return Err(candle_core::Error::Msg(format!(
                "fusion: scalar lanes must have exactly one element, got {}",
                s.elem_count()
            )));
        }
    }
    let inputs = &owned[..inputs.len()];
    let scalars = &owned[inputs.len()..];
    if n == 0 {
        return exprs
            .iter()
            .map(|_| Tensor::zeros(shape, dtype, device))
            .collect();
    }
    let contig;
    let lane_strides = match strides {
        Some(ss) => ss,
        None => {
            contig = vec![contiguous_strides(shape); inputs.len()];
            &contig
        }
    };
    match (device, dtype) {
        (Device::Cpu, DType::F32) => interpret_cpu::<f32>(exprs, inputs, strides, scalars, n, shape),
        (Device::Cpu, DType::F64) => interpret_cpu::<f64>(exprs, inputs, strides, scalars, n, shape),
        #[cfg(target_os = "macos")]
        (Device::Metal(_), DType::F32) => {
            metal::run(exprs, inputs, lane_strides, scalars, n, shape, device)
        }
        _ => Err(candle_core::Error::Msg(format!(
            "fusion: unsupported device/dtype {device:?} {dtype:?}"
        ))),
    }
}

/// Evaluates a fused-reduce region: `expr` is computed per input element
/// (lanes read through `strides` in input-dim space) and folded over
/// `dims` into one output element per non-reduced coordinate. The
/// elementwise intermediate is never materialized. `dims` must be sorted
/// ascending and non-empty; `out_shape` is the reduced shape with
/// keepdims applied.
#[allow(clippy::too_many_arguments)]
pub fn run_reduce(
    op: ReduceOp,
    expr: &Expr,
    inputs: &[Tensor],
    strides: &[Vec<usize>],
    in_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    out_shape: &[usize],
    dtype: DType,
    device: &Device,
) -> candle_core::Result<Tensor> {
    if strides.len() != inputs.len() {
        return Err(candle_core::Error::Msg(format!(
            "fusion: got {} stride entries for {} inputs",
            strides.len(),
            inputs.len()
        )));
    }
    let mut owned = Vec::with_capacity(inputs.len());
    for t in inputs {
        owned.push(if t.is_contiguous() { t.clone() } else { t.contiguous()? });
    }
    let inputs = &owned[..];
    let out_n: usize = out_shape.iter().product();
    if out_n == 0 {
        return Tensor::zeros(out_shape, dtype, device);
    }
    match (device, dtype) {
        (Device::Cpu, DType::F32) => {
            interpret_reduce_cpu::<f32>(op, expr, inputs, strides, in_shape, dims, keepdims, out_shape)
        }
        (Device::Cpu, DType::F64) => {
            interpret_reduce_cpu::<f64>(op, expr, inputs, strides, in_shape, dims, keepdims, out_shape)
        }
        #[cfg(target_os = "macos")]
        (Device::Metal(_), DType::F32) => {
            metal::run_reduce(op, expr, inputs, strides, in_shape, dims, keepdims, out_shape, device)
        }
        _ => Err(candle_core::Error::Msg(format!(
            "fusion: unsupported device/dtype {device:?} {dtype:?}"
        ))),
    }
}

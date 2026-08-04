//! Elementwise expression IR and single-kernel fusion (RFC 0007).
//!
//! A fused region is a small scalar expression DAG over named input lanes.
//! On Metal it is lowered to an SSA-form MSL kernel by the first-party
//! emitter (`runtime::metal::emit`), compiled once per distinct expression
//! (cached) and launched over the flattened input buffers; on CPU the same
//! IR is interpreted per element in a single pass. GPU fusion is f32-only;
//! the CPU interpreter covers f32 and f64.

use crate::dev::Device;
use crate::runtime::dtype::DType;
#[cfg(target_os = "macos")]
use crate::runtime::metal::device::MetalDevice;
#[cfg(target_os = "macos")]
use crate::runtime::metal::run::MetalTensor;
#[cfg(any(target_os = "macos", feature = "cuda"))]

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
    Prod,
}

impl ReduceOp {
    fn init(&self) -> f64 {
        match self {
            ReduceOp::Sum | ReduceOp::Mean => 0.0,
            ReduceOp::Prod => 1.0,
            ReduceOp::Max => f64::NEG_INFINITY,
            ReduceOp::Min => f64::INFINITY,
        }
    }

    fn fold<T: Scalar>(&self, acc: T, v: T) -> T {
        match self {
            ReduceOp::Sum | ReduceOp::Mean => acc.add(v),
            ReduceOp::Prod => acc.mul(v),
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

pub(crate) fn interpret_core<T: Scalar>(
    exprs: &[Expr],
    slices: &[&[T]],
    strides: Option<&[Vec<usize>]>,
    scalar_values: &[T],
    n: usize,
    shape: &[usize],
) -> Vec<Vec<T>> {
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
                    out[i] = eval_at(expr, i, slices, scalar_values);
                }
            }
        }
        // Broadcast lanes: walk the output coordinates with an odometer so
        // each lane's element offset is incremental (no per-element div/mod).
        Some(ss) => {
            let rank = shape.len();
            let mut coord = vec![0usize; rank];
            let mut offs = vec![0usize; slices.len()];
            let mut lane_vals = vec![<T as Scalar>::from_f64(0.0); slices.len()];
            for i in 0..n {
                for (v, (slice, off)) in lane_vals.iter_mut().zip(slices.iter().zip(offs.iter())) {
                    *v = slice[*off];
                }
                for (out, expr) in outs.iter_mut().zip(exprs.iter()) {
                    out[i] = eval(expr, &|k| lane_vals[k as usize], scalar_values);
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
    outs
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn interpret_reduce_core<T: Scalar>(
    op: ReduceOp,
    expr: &Expr,
    slices: &[&[T]],
    strides: &[Vec<usize>],
    in_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    out_shape: &[usize],
) -> Vec<T> {
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
    let mut offs = vec![0usize; slices.len()];
    let mut lane_vals = vec![init; slices.len()];
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
    acc
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
    use super::{fusion_phase_nanos, Expr, ReduceOp};
    use crate::dev::Device;
    use crate::runtime::dtype::DType;
    use crate::runtime::metal::device::MetalDevice;
    use crate::runtime::metal::run::MetalTensor;

    pub fn run(
        exprs: &[Expr],
        inputs: &[crate::val::Val],
        lane_strides: &[Vec<usize>],
        scalars: &[crate::val::Val],
        n: usize,
        shape: &[usize],
        device: &Device,
    ) -> crate::err::Res<Vec<crate::val::Val>> {
        let _ = device;
        // Metal exposes at most 31 buffer argument slots per kernel
        // (scalars share one packed slot).
        if inputs.len() + exprs.len() + 1 > 31 {
            return Err(format!(
                "fusion: {} buffer arguments exceed Metal's limit of 31",
                inputs.len() + exprs.len() + 1
            ));
        }
        let phase_timing = std::env::var_os("EFFECT_TORCH_FUSION_TIMING").is_some();
        let t_start = std::time::Instant::now();
        let native: Vec<MetalTensor> = inputs
            .iter()
            .map(|v| v.as_metal().cloned())
            .collect::<crate::err::Res<_>>()?;
        let refs: Vec<&MetalTensor> = native.iter().collect();
        // Pack the scalar lanes without any host readback: when every
        // scalar is already a Metal f32 tensor, cat the 0-d values into one
        // device buffer and bind it directly. Otherwise fall back to a
        // single fence for all host readbacks.
        let dev = MetalDevice::get();
        let metal_scalars: Option<Vec<MetalTensor>> = scalars
            .iter()
            .map(|v| match v {
                crate::val::Val::Metal(t) if t.dtype == DType::F32 => Some(t.clone()),
                _ => None,
            })
            .collect();
        let (scalar_buf, num_scalars) = if scalars.is_empty() {
            (None, 0)
        } else if let Some(ts) = metal_scalars {
            let views: Vec<MetalTensor> = ts
                .iter()
                .map(|t| MetalTensor {
                    buffer: t.buffer.clone(),
                    layout: crate::runtime::layout::Layout::contiguous(vec![1]),
                    dtype: DType::F32,
                })
                .collect();
            let refs: Vec<&MetalTensor> = views.iter().collect();
            let packed = crate::runtime::metal::indexing::cat(dev, &refs, 0)?;
            (Some(packed.buffer.clone()), scalars.len())
        } else {
            dev.synchronize();
            let vals: Vec<f32> = scalars
                .iter()
                .map(|v| v.to_f32_vec().map(|x| x[0]))
                .collect::<crate::err::Res<_>>()?;
            (Some(dev.alloc_with_data(&vals)), vals.len())
        };
        if phase_timing {
            fusion_phase_nanos(0, t_start.elapsed().as_nanos() as u64);
        }
        let outs = crate::runtime::metal::run::run_elementwise_scalar_buf(
            MetalDevice::get(),
            exprs,
            &refs,
            lane_strides,
            scalar_buf.as_deref(),
            num_scalars,
            n,
            shape,
        )?;
        outs.into_iter()
            .map(crate::val::Val::Metal)
            .map(Ok)
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_reduce(
        op: ReduceOp,
        expr: &Expr,
        inputs: &[crate::val::Val],
        lane_strides: &[Vec<usize>],
        in_shape: &[usize],
        dims: &[usize],
        keepdims: bool,
        out_shape: &[usize],
        device: &Device,
    ) -> crate::err::Res<crate::val::Val> {
        let _ = device;
        if inputs.len() + 1 > 31 {
            return Err(format!(
                "fusion: {} buffer arguments exceed Metal's limit of 31",
                inputs.len() + 1
            ));
        }
        let native: Vec<MetalTensor> = inputs
            .iter()
            .map(|v| v.as_metal().cloned())
            .collect::<crate::err::Res<_>>()?;
        let refs: Vec<&MetalTensor> = native.iter().collect();
        let out = crate::runtime::metal::run::run_reduce(
            MetalDevice::get(),
            op,
            expr,
            &refs,
            lane_strides,
            in_shape,
            dims,
            keepdims,
            out_shape,
        )?;
        Ok(crate::val::Val::Metal(out))
    }
}

/// Whether `run` can execute directly on this device/dtype pair. Callers
/// keep their composed candle-op path as the fallback when this is false.
pub fn is_supported(device: &Device, dtype: DType) -> bool {
    match device {
        Device::Cpu => matches!(dtype, DType::F32 | DType::F64),
        #[cfg(target_os = "macos")]
        Device::Metal => dtype == DType::F32,
        #[cfg(not(target_os = "macos"))]
        Device::Metal => false,
    }
}

// The fused AdamW update as three expressions over lanes [param, grad, m,
// v] with scalar lanes [lr, 1 - beta1^t, 1 - beta2^t], mirroring the
// composed update's operation order exactly. Step-dependent values are
// scalar lanes so the compiled kernel is stable across steps.
pub fn adamw_exprs(beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> [Expr; 3] {
    adamw_exprs_with(
        beta1,
        beta2,
        eps,
        weight_decay,
        Expr::Scalar(0),
        Expr::Scalar(1),
        Expr::Scalar(2),
    )
}

fn adamw_exprs_with(beta1: f64, beta2: f64, eps: f64, weight_decay: f64, lr: Expr, c1: Expr, c2: Expr) -> [Expr; 3] {
    let (p, g, m, v) = (Expr::Input(0), Expr::Input(1), Expr::Input(2), Expr::Input(3));
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
    sgd_exprs_with(momentum, dampening, nesterov, weight_decay, Expr::Scalar(0), Expr::Scalar(1))
}

fn sgd_exprs_with(momentum: f64, dampening: f64, nesterov: bool, weight_decay: f64, lr: Expr, first: Expr) -> [Expr; 2] {
    let (p, g, v) = (Expr::Input(0), Expr::Input(1), Expr::Input(2));
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

// A pre-built, process-cached AdamW group plan: the exprs, lane strides and
// pipeline key depend only on the parameter count, shape and
// hyperparameters, so groups rebuild and re-hash none of it per step.
pub struct GroupPlan {
    pub exprs: Vec<Expr>,
    pub strides: Vec<Vec<usize>>,
    pub key: u64,
    pub num_scalars: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn adamw_group_plan(
    params_len: usize,
    shape: &[usize],
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
) -> std::sync::Arc<GroupPlan> {
    type Key = (usize, Vec<usize>, u64, u64, u64, u64);
    static PLANS: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<Key, std::sync::Arc<GroupPlan>>>> =
        std::sync::OnceLock::new();
    let key: Key = (
        params_len,
        shape.to_vec(),
        beta1.to_bits(),
        beta2.to_bits(),
        eps.to_bits(),
        weight_decay.to_bits(),
    );
    let cache = PLANS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Some(p) = cache.lock().unwrap().get(&key) {
        return p.clone();
    }
    let base = adamw_exprs(beta1, beta2, eps, weight_decay);
    let mut exprs = Vec::with_capacity(params_len * 3);
    for i in 0..params_len {
        let remap: std::collections::HashMap<u32, u32> = (0u32..4)
            .map(|k| (k, (i * 4) as u32 + k))
            .collect();
        for expr in &base {
            exprs.push(expr.remap_lanes(&remap));
        }
    }
    let contig = contiguous_strides(shape);
    let strides = vec![contig; params_len * 4];
    let n: usize = shape.iter().product();
    let key_hash = crate::runtime::metal::run::elementwise_key(&exprs, &strides, shape, n, 3);
    let plan = std::sync::Arc::new(GroupPlan {
        exprs,
        strides,
        key: key_hash,
        num_scalars: 3,
    });
    cache.lock().unwrap().insert(key, plan.clone());
    plan
}

// Cat 0-d f32 scalars into one packed device buffer (no host readback for
// device-resident scalars; CPU scalars upload as f32). Callers memoize the
// result per walk.
pub fn pack_scalars_metal(scalars: &[crate::val::Val]) -> crate::err::Res<crate::val::Val> {
    let dev = MetalDevice::get();
    let mut views = Vec::with_capacity(scalars.len());
    for v in scalars {
        let t = match v {
            crate::val::Val::Metal(t) => {
                let t = if t.layout.is_contiguous() && t.layout.offset() == 0 {
                    t.clone()
                } else {
                    crate::runtime::metal::kernels::strided_copy(dev, t)?
                };
                if t.dtype == DType::F32 {
                    t
                } else {
                    crate::runtime::metal::kernels::cast(dev, &t, DType::F32)?
                }
            }
            crate::val::Val::Cpu(t) => {
                let t = t.cast(DType::F32).contiguous();
                let crate::runtime::cpu::CpuBuffer::F32(v) = &t.buffer else { unreachable!() };
                MetalTensor::from_f32(dev, v.as_slice().to_vec(), vec![1])
            }
        };
        views.push(MetalTensor {
            buffer: t.buffer.clone(),
            layout: crate::runtime::layout::Layout::contiguous(vec![1]),
            dtype: DType::F32,
        });
    }
    let refs: Vec<&MetalTensor> = views.iter().collect();
    Ok(crate::val::Val::Metal(crate::runtime::metal::indexing::cat(dev, &refs, 0)?))
}

// Runs a cached group plan on Metal with a pre-packed scalar buffer.
pub fn run_group_metal(
    plan: &GroupPlan,
    inputs: &[crate::val::Val],
    packed_scalars: &crate::val::Val,
    shape: &[usize],
) -> crate::err::Res<Vec<crate::val::Val>> {
    if inputs.len() + plan.exprs.len() + 1 > 31 {
        return Err(format!(
            "fusion: {} buffer arguments exceed Metal's limit of 31",
            inputs.len() + plan.exprs.len() + 1
        ));
    }
    let dev = MetalDevice::get();
    let mut owned = Vec::with_capacity(inputs.len());
    for v in inputs {
        let t = v.as_metal()?;
        owned.push(if t.layout.is_contiguous() {
            t.clone()
        } else {
            crate::runtime::metal::kernels::strided_copy(dev, t)?
        });
    }
    let refs: Vec<&MetalTensor> = owned.iter().collect();
    let packed = packed_scalars.as_metal()?;
    let n: usize = shape.iter().product();
    let outs = crate::runtime::metal::run::run_elementwise_prekeyed(
        dev,
        plan.key,
        &plan.exprs,
        &refs,
        &plan.strides,
        Some(&packed.buffer),
        plan.num_scalars,
        n,
        shape,
    )?;
    Ok(outs.into_iter().map(crate::val::Val::Metal).collect())
}

/// Evaluates each expression over the flattened inputs (broadcast to the
/// output element count) and returns one tensor per expression. `strides`
/// gives each input lane's strides in output-dim space (`None` means every
/// lane is contiguous and output-shaped). `scalars` are one-element tensors
/// read at offset 0.
pub fn run(
    exprs: &[Expr],
    inputs: &[crate::val::Val],
    strides: Option<&[Vec<usize>]>,
    scalars: &[crate::val::Val],
    n: usize,
    shape: &[usize],
    dtype: DType,
    device: &Device,
) -> crate::err::Res<Vec<crate::val::Val>> {
    if let Some(ss) = strides {
        if ss.len() != inputs.len() {
            return Err(format!(
                "fusion: got {} stride entries for {} inputs",
                ss.len(),
                inputs.len()
            ));
        }
    }
    let mut owned = Vec::with_capacity(inputs.len() + scalars.len());
    for v in inputs.iter().chain(scalars.iter()) {
        owned.push(match v {
            crate::val::Val::Cpu(t) => crate::val::Val::Cpu(t.contiguous()),
            crate::val::Val::Metal(t) => {
                if t.layout.is_contiguous() {
                    v.clone()
                } else {
                    crate::val::Val::Metal(
                        crate::runtime::metal::kernels::strided_copy(
                            crate::runtime::metal::device::MetalDevice::get(),
                            t,
                        )?,
                    )
                }
            }
        });
    }
    for s in scalars.iter() {
        if s.numel() != 1 {
            return Err(format!(
                "fusion: scalar lanes must have exactly one element, got {}",
                s.numel()
            ));
        }
    }
    let inputs = &owned[..inputs.len()];
    let scalars = &owned[inputs.len()..];
    if n == 0 {
        return exprs
            .iter()
            .map(|_| match device {
                Device::Cpu => Ok(crate::val::Val::Cpu(crate::runtime::cpu::Tensor::zeros(shape, dtype))),
                Device::Metal => Ok(crate::val::Val::Metal(crate::runtime::metal::run::MetalTensor::zeros(
                    crate::runtime::metal::device::MetalDevice::get(),
                    shape.to_vec(),
                    dtype,
                ))),
            })
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
        (Device::Cpu, DType::F32) => cpu_bridge_elementwise::<f32>(exprs, inputs, strides, scalars, n, shape),
        (Device::Cpu, DType::F64) => cpu_bridge_elementwise::<f64>(exprs, inputs, strides, scalars, n, shape),
        #[cfg(target_os = "macos")]
        (Device::Metal, DType::F32) => {
            metal::run(exprs, inputs, lane_strides, scalars, n, shape, device)
        }
        _ => Err(format!("fusion: unsupported device/dtype {device:?} {dtype:?}")),
    }
}

fn cpu_bridge_elementwise<T: Scalar + crate::runtime::cpu::Elem>(
    exprs: &[Expr],
    inputs: &[crate::val::Val],
    strides: Option<&[Vec<usize>]>,
    scalars: &[crate::val::Val],
    n: usize,
    shape: &[usize],
) -> crate::err::Res<Vec<crate::val::Val>> {
    let mut slices: Vec<&[T]> = Vec::with_capacity(inputs.len());
    for v in inputs {
        slices.push(native_slice::<T>(v)?);
    }
    let mut scalar_values: Vec<T> = Vec::with_capacity(scalars.len());
    for v in scalars {
        scalar_values.push(native_slice::<T>(v)?[0]);
    }
    let outs = interpret_core::<T>(exprs, &slices, strides, &scalar_values, n, shape);
    outs.into_iter()
        .map(|out| {
            Ok(crate::val::Val::Cpu(crate::runtime::cpu::Tensor::from_vec(out, shape.to_vec())))
        })
        .collect()
}

fn native_slice<'a, T: crate::runtime::cpu::Elem>(v: &'a crate::val::Val) -> crate::err::Res<&'a [T]> {
    let crate::val::Val::Cpu(t) = v else {
        return Err("fusion: expected a CPU value".to_string());
    };
    T::slice_of(t).ok_or_else(|| "fusion: native bridge expects contiguous inputs of matching dtype".to_string())
}

#[allow(clippy::too_many_arguments)]
fn cpu_bridge_reduce<T: Scalar + crate::runtime::cpu::Elem>(
    op: ReduceOp,
    expr: &Expr,
    inputs: &[crate::val::Val],
    strides: &[Vec<usize>],
    in_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    out_shape: &[usize],
) -> crate::err::Res<crate::val::Val> {
    let mut slices: Vec<&[T]> = Vec::with_capacity(inputs.len());
    for v in inputs {
        slices.push(native_slice::<T>(v)?);
    }
    let acc = interpret_reduce_core::<T>(op, expr, &slices, strides, in_shape, dims, keepdims, out_shape);
    Ok(crate::val::Val::Cpu(crate::runtime::cpu::Tensor::from_vec(acc, out_shape.to_vec())))
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
    inputs: &[crate::val::Val],
    strides: &[Vec<usize>],
    in_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    out_shape: &[usize],
    dtype: DType,
    device: &Device,
) -> crate::err::Res<crate::val::Val> {
    if strides.len() != inputs.len() {
        return Err(format!(
            "fusion: got {} stride entries for {} inputs",
            strides.len(),
            inputs.len()
        ));
    }
    let mut owned = Vec::with_capacity(inputs.len());
    for v in inputs {
        owned.push(match v {
            crate::val::Val::Cpu(t) => crate::val::Val::Cpu(t.contiguous()),
            crate::val::Val::Metal(t) => {
                if t.layout.is_contiguous() {
                    v.clone()
                } else {
                    crate::val::Val::Metal(
                        crate::runtime::metal::kernels::strided_copy(
                            crate::runtime::metal::device::MetalDevice::get(),
                            t,
                        )?,
                    )
                }
            }
        });
    }
    let inputs = &owned[..];
    let out_n: usize = out_shape.iter().product();
    if out_n == 0 {
        return match device {
            Device::Cpu => Ok(crate::val::Val::Cpu(crate::runtime::cpu::Tensor::zeros(out_shape, dtype))),
            Device::Metal => Ok(crate::val::Val::Metal(crate::runtime::metal::run::MetalTensor::zeros(
                crate::runtime::metal::device::MetalDevice::get(),
                out_shape.to_vec(),
                dtype,
            ))),
        };
    }
    match (device, dtype) {
        (Device::Cpu, DType::F32) => {
            cpu_bridge_reduce::<f32>(op, expr, inputs, strides, in_shape, dims, keepdims, out_shape)
        }
        (Device::Cpu, DType::F64) => {
            cpu_bridge_reduce::<f64>(op, expr, inputs, strides, in_shape, dims, keepdims, out_shape)
        }
        #[cfg(target_os = "macos")]
        (Device::Metal, DType::F32) => {
            metal::run_reduce(op, expr, inputs, strides, in_shape, dims, keepdims, out_shape, device)
        }
        _ => Err(format!("fusion: unsupported device/dtype {device:?} {dtype:?}")),
    }
}

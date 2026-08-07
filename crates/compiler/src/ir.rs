//! Elementwise expression IR and single-kernel fusion (RFC 0007).
//!
//! A fused region is a small scalar expression DAG over named input lanes.
//! On Metal it is lowered to an SSA-form MSL kernel by the first-party
//! emitter (`runtime::metal::emit`), compiled once per distinct expression
//! (cached) and launched over the flattened input buffers; on CPU the same
//! IR is interpreted per element in a single pass. GPU fusion is f32-only;
//! the CPU interpreter covers f32 and f64.

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
    // gelu with the exact erf form / the tanh approximation; emitted as
    // one helper so gemm epilogues and elementwise regions share it.
    Gelu(Box<Expr>),
    GeluTanh(Box<Expr>),
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
            | Expr::Erf(a)
            | Expr::Gelu(a)
            | Expr::GeluTanh(a) => 1 + a.ops(),
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
            Expr::Gelu(a) => Expr::Gelu(Box::new(a.remap_inputs(f))),
            Expr::GeluTanh(a) => Expr::GeluTanh(Box::new(a.remap_inputs(f))),
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
                Expr::Add(a, b) => Expr::Add(
                    Box::new(go(a, lane, r, remap)),
                    Box::new(go(b, lane, r, remap)),
                ),
                Expr::Sub(a, b) => Expr::Sub(
                    Box::new(go(a, lane, r, remap)),
                    Box::new(go(b, lane, r, remap)),
                ),
                Expr::Mul(a, b) => Expr::Mul(
                    Box::new(go(a, lane, r, remap)),
                    Box::new(go(b, lane, r, remap)),
                ),
                Expr::Div(a, b) => Expr::Div(
                    Box::new(go(a, lane, r, remap)),
                    Box::new(go(b, lane, r, remap)),
                ),
                Expr::Min(a, b) => Expr::Min(
                    Box::new(go(a, lane, r, remap)),
                    Box::new(go(b, lane, r, remap)),
                ),
                Expr::Max(a, b) => Expr::Max(
                    Box::new(go(a, lane, r, remap)),
                    Box::new(go(b, lane, r, remap)),
                ),
                Expr::Lt(a, b) => Expr::Lt(
                    Box::new(go(a, lane, r, remap)),
                    Box::new(go(b, lane, r, remap)),
                ),
                Expr::Le(a, b) => Expr::Le(
                    Box::new(go(a, lane, r, remap)),
                    Box::new(go(b, lane, r, remap)),
                ),
                Expr::Gt(a, b) => Expr::Gt(
                    Box::new(go(a, lane, r, remap)),
                    Box::new(go(b, lane, r, remap)),
                ),
                Expr::Ge(a, b) => Expr::Ge(
                    Box::new(go(a, lane, r, remap)),
                    Box::new(go(b, lane, r, remap)),
                ),
                Expr::Eq(a, b) => Expr::Eq(
                    Box::new(go(a, lane, r, remap)),
                    Box::new(go(b, lane, r, remap)),
                ),
                Expr::Ne(a, b) => Expr::Ne(
                    Box::new(go(a, lane, r, remap)),
                    Box::new(go(b, lane, r, remap)),
                ),
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
                Expr::Gelu(a) => Expr::Gelu(Box::new(go(a, lane, r, remap))),
                Expr::GeluTanh(a) => Expr::GeluTanh(Box::new(go(a, lane, r, remap))),
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
                if cond != 0.0 as $ty {
                    lhs
                } else {
                    rhs
                }
            }
            fn lt(self, o: Self) -> Self {
                if self < o {
                    1.0 as $ty
                } else {
                    0.0 as $ty
                }
            }
            fn le(self, o: Self) -> Self {
                if self <= o {
                    1.0 as $ty
                } else {
                    0.0 as $ty
                }
            }
            fn gt(self, o: Self) -> Self {
                if self > o {
                    1.0 as $ty
                } else {
                    0.0 as $ty
                }
            }
            fn ge(self, o: Self) -> Self {
                if self >= o {
                    1.0 as $ty
                } else {
                    0.0 as $ty
                }
            }
            fn eq(self, o: Self) -> Self {
                if self == o {
                    1.0 as $ty
                } else {
                    0.0 as $ty
                }
            }
            fn ne(self, o: Self) -> Self {
                if self != o {
                    1.0 as $ty
                } else {
                    0.0 as $ty
                }
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
        Expr::Gelu(a) => {
            let x = eval(a, lane, scalars);
            let inner = x.mul(T::from_f64(std::f64::consts::FRAC_1_SQRT_2)).erf();
            x.mul(T::from_f64(0.5)).mul(T::from_f64(1.0).add(inner))
        }
        Expr::GeluTanh(a) => {
            let x = eval(a, lane, scalars);
            let u = x
                .add(x.mul(x).mul(x).mul(T::from_f64(0.044715)))
                .mul(T::from_f64(0.7978845608028654));
            x.mul(T::from_f64(0.5)).mul(T::from_f64(1.0).add(u.tanh()))
        }
    }
}

pub fn interpret_core<T: Scalar>(
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
pub fn interpret_reduce_core<T: Scalar>(
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

fn adamw_exprs_with(
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    lr: Expr,
    c1: Expr,
    c2: Expr,
) -> [Expr; 3] {
    let (p, g, m, v) = (
        Expr::Input(0),
        Expr::Input(1),
        Expr::Input(2),
        Expr::Input(3),
    );
    let next_m = Expr::Add(
        Box::new(Expr::Mul(Box::new(m), Box::new(Expr::cst(beta1)))),
        Box::new(Expr::Mul(
            Box::new(g.clone()),
            Box::new(Expr::cst(1.0 - beta1)),
        )),
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
    [
        Expr::Sub(Box::new(base), Box::new(adjusted)),
        next_m,
        next_v,
    ]
}

// The fused momentum-SGD update over lanes [param, grad, velocity] with
// scalar lanes [lr, first], mirroring the composed update including the
// first-step v = g initialization as a select on the 0-d `first` flag.
pub fn sgd_exprs(momentum: f64, dampening: f64, nesterov: bool, weight_decay: f64) -> [Expr; 2] {
    sgd_exprs_with(
        momentum,
        dampening,
        nesterov,
        weight_decay,
        Expr::Scalar(0),
        Expr::Scalar(1),
    )
}

fn sgd_exprs_with(
    momentum: f64,
    dampening: f64,
    nesterov: bool,
    weight_decay: f64,
    lr: Expr,
    first: Expr,
) -> [Expr; 2] {
    let (p, g, v) = (Expr::Input(0), Expr::Input(1), Expr::Input(2));
    let gp = if weight_decay == 0.0 {
        g
    } else {
        Expr::Add(
            Box::new(g),
            Box::new(Expr::Mul(
                Box::new(p.clone()),
                Box::new(Expr::cst(weight_decay)),
            )),
        )
    };
    let continued = Expr::Add(
        Box::new(Expr::Mul(Box::new(v), Box::new(Expr::cst(momentum)))),
        Box::new(Expr::Mul(
            Box::new(gp.clone()),
            Box::new(Expr::cst(1.0 - dampening)),
        )),
    );
    let next_v = Expr::Select(
        Box::new(Expr::Gt(Box::new(first), Box::new(Expr::cst(0.5)))),
        Box::new(gp.clone()),
        Box::new(continued),
    );
    let used = if nesterov {
        Expr::Add(
            Box::new(gp),
            Box::new(Expr::Mul(
                Box::new(next_v.clone()),
                Box::new(Expr::cst(momentum)),
            )),
        )
    } else {
        next_v.clone()
    };
    [
        Expr::Sub(
            Box::new(p),
            Box::new(Expr::Mul(Box::new(used), Box::new(lr))),
        ),
        next_v,
    ]
}

impl effect_torch_graph::FusionExpression for Expr {
    type ReduceOp = ReduceOp;

    fn lane_strides(lane: &[usize], out: &[usize]) -> Option<Vec<usize>> {
        lane_strides(lane, out)
    }
}

pub fn is_supported(
    device: &effect_torch_graph::Device,
    dtype: effect_torch_runtime::DType,
) -> bool {
    match device {
        effect_torch_graph::Device::Cpu => matches!(
            dtype,
            effect_torch_runtime::DType::F32 | effect_torch_runtime::DType::F64
        ),
        effect_torch_graph::Device::Metal => matches!(
            dtype,
            effect_torch_runtime::DType::F32 | effect_torch_runtime::DType::BF16
        ),
    }
}

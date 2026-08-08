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

#[derive(Debug)]
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

impl Expr {
    // Moves child boxes into the worklist, leaving cheap leaves behind;
    // used by Drop so destructor glue never recurses.
    fn drain_children(&mut self, worklist: &mut Vec<Box<Expr>>) {
        fn dummy() -> Box<Expr> {
            Box::new(Expr::Const(0))
        }
        match self {
            Expr::Input(_) | Expr::Scalar(_) | Expr::Const(_) => {}
            Expr::Select(c, a, b) => {
                worklist.push(std::mem::replace(c, dummy()));
                worklist.push(std::mem::replace(a, dummy()));
                worklist.push(std::mem::replace(b, dummy()));
            }
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
            | Expr::Ne(a, b) => {
                worklist.push(std::mem::replace(a, dummy()));
                worklist.push(std::mem::replace(b, dummy()));
            }
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
            | Expr::GeluTanh(a) => worklist.push(std::mem::replace(a, dummy())),
        }
    }
}

impl Drop for Expr {
    fn drop(&mut self) {
        // A long elementwise chain fuses into one deep Expr; default
        // destructor glue recurses over the Box chain and overflows the
        // (worker-thread) stack, so descendants drain into a worklist.
        // Worklist entries drop with dummy leaves in place, so their own
        // Drop returns immediately.
        let mut worklist = Vec::new();
        self.drain_children(&mut worklist);
        while let Some(mut node) = worklist.pop() {
            node.drain_children(&mut worklist);
        }
    }
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

    // Child references in left-to-right order.
    fn children(&self) -> Vec<&Expr> {
        match self {
            Expr::Input(_) | Expr::Scalar(_) | Expr::Const(_) => Vec::new(),
            Expr::Select(c, a, b) => vec![c.as_ref(), a.as_ref(), b.as_ref()],
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
            | Expr::Ne(a, b) => vec![a.as_ref(), b.as_ref()],
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
            | Expr::GeluTanh(a) => vec![a.as_ref()],
        }
    }

    // Rebuilds the same variant with new children (left-to-right).
    fn rebuild(&self, mut children: Vec<Expr>) -> Expr {
        let mut next = || Box::new(children.remove(0));
        match self {
            Expr::Input(k) => Expr::Input(*k),
            Expr::Scalar(k) => Expr::Scalar(*k),
            Expr::Const(b) => Expr::Const(*b),
            Expr::Select(..) => Expr::Select(next(), next(), next()),
            Expr::Add(..) => Expr::Add(next(), next()),
            Expr::Sub(..) => Expr::Sub(next(), next()),
            Expr::Mul(..) => Expr::Mul(next(), next()),
            Expr::Div(..) => Expr::Div(next(), next()),
            Expr::Min(..) => Expr::Min(next(), next()),
            Expr::Max(..) => Expr::Max(next(), next()),
            Expr::Lt(..) => Expr::Lt(next(), next()),
            Expr::Le(..) => Expr::Le(next(), next()),
            Expr::Gt(..) => Expr::Gt(next(), next()),
            Expr::Ge(..) => Expr::Ge(next(), next()),
            Expr::Eq(..) => Expr::Eq(next(), next()),
            Expr::Ne(..) => Expr::Ne(next(), next()),
            Expr::Neg(..) => Expr::Neg(next()),
            Expr::Sqrt(..) => Expr::Sqrt(next()),
            Expr::Exp(..) => Expr::Exp(next()),
            Expr::Sin(..) => Expr::Sin(next()),
            Expr::Cos(..) => Expr::Cos(next()),
            Expr::Tanh(..) => Expr::Tanh(next()),
            Expr::Abs(..) => Expr::Abs(next()),
            Expr::Log(..) => Expr::Log(next()),
            Expr::Floor(..) => Expr::Floor(next()),
            Expr::Ceil(..) => Expr::Ceil(next()),
            Expr::Round(..) => Expr::Round(next()),
            Expr::Powf(_, e) => Expr::Powf(next(), *e),
            Expr::Erf(..) => Expr::Erf(next()),
            Expr::Gelu(..) => Expr::Gelu(next()),
            Expr::GeluTanh(..) => Expr::GeluTanh(next()),
        }
    }

    // Iterative post-order rebuild: `f` may substitute a leaf (Some) or
    // keep it (None); internal nodes rebuild from already-transformed
    // children. Tree transforms must never recurse — deep fused regions
    // are bounded by heap, not the call stack.
    fn transform(&self, f: &mut dyn FnMut(&Expr) -> Option<Expr>) -> Expr {
        let mut stack: Vec<(&Expr, bool)> = vec![(self, false)];
        let mut out: Vec<Expr> = Vec::new();
        while let Some((node, processed)) = stack.pop() {
            let children = node.children();
            if processed {
                let rebuilt = out.split_off(out.len() - children.len());
                out.push(node.rebuild(rebuilt));
                continue;
            }
            if children.is_empty() {
                out.push(f(node).unwrap_or_else(|| node.rebuild(Vec::new())));
                continue;
            }
            stack.push((node, true));
            for child in children.into_iter().rev() {
                stack.push((child, false));
            }
        }
        debug_assert_eq!(out.len(), 1);
        out.pop().expect("transform result")
    }

    /// Remaps per-element lane indices through `remap` (which must cover
    /// every lane the expression references).
    pub fn remap_lanes(&self, remap: &std::collections::HashMap<u32, u32>) -> Self {
        self.remap_inputs(&mut |k| remap[&k])
    }

    /// Number of nodes in the expression tree (shared subtrees count per
    /// occurrence: this bounds emitted kernel size, not SSA values).
    pub fn ops(&self) -> usize {
        let mut count = 0usize;
        let mut stack = vec![self];
        while let Some(node) = stack.pop() {
            count += 1;
            stack.extend(node.children());
        }
        count
    }

    fn remap_inputs(&self, f: &mut dyn FnMut(u32) -> u32) -> Self {
        self.transform(&mut |e| match e {
            Expr::Input(k) => Some(Expr::Input(f(*k))),
            _ => None,
        })
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
        self.transform(&mut |e| match e {
            Expr::Input(k) if *k == lane => Some(replacement.clone()),
            Expr::Input(k) => Some(Expr::Input(remap[k])),
            _ => None,
        })
    }
}

// Clone, equality and hashing are manual and iterative (via the
// post-order plan): derived glue would recurse over deep Box chains.
impl Clone for Expr {
    fn clone(&self) -> Self {
        self.transform(&mut |_| None)
    }
}

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        flatten(self) == flatten(other)
    }
}

impl Eq for Expr {}

impl std::hash::Hash for Expr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for op in flatten(self) {
            op.hash(state);
        }
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

// A fused expression flattened to a post-order plan. A long elementwise
// chain fuses into one deep Expr, and the interpreter runs once per
// element — a recursive tree walk would multiply the chain depth by the
// call stack of every element. The plan evaluates with an explicit value
// stack: depth is bounded by heap, never the call stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Flat {
    Input(u32),
    Scalar(u32),
    Const(u64),
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
    Select,
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
    Powf(u64),
    Erf,
    Gelu,
    GeluTanh,
}

// Iterative post-order flattening (the traversal is stack-safe like the
// evaluator below).
fn flatten(e: &Expr) -> Vec<Flat> {
    let mut out = Vec::new();
    let mut stack: Vec<(&Expr, bool)> = vec![(e, false)];
    while let Some((node, processed)) = stack.pop() {
        if processed {
            out.push(match node {
                Expr::Input(k) => Flat::Input(*k),
                Expr::Scalar(k) => Flat::Scalar(*k),
                Expr::Const(bits) => Flat::Const(*bits),
                Expr::Add(..) => Flat::Add,
                Expr::Sub(..) => Flat::Sub,
                Expr::Mul(..) => Flat::Mul,
                Expr::Div(..) => Flat::Div,
                Expr::Min(..) => Flat::Min,
                Expr::Max(..) => Flat::Max,
                Expr::Lt(..) => Flat::Lt,
                Expr::Le(..) => Flat::Le,
                Expr::Gt(..) => Flat::Gt,
                Expr::Ge(..) => Flat::Ge,
                Expr::Eq(..) => Flat::Eq,
                Expr::Ne(..) => Flat::Ne,
                Expr::Select(..) => Flat::Select,
                Expr::Neg(..) => Flat::Neg,
                Expr::Sqrt(..) => Flat::Sqrt,
                Expr::Exp(..) => Flat::Exp,
                Expr::Sin(..) => Flat::Sin,
                Expr::Cos(..) => Flat::Cos,
                Expr::Tanh(..) => Flat::Tanh,
                Expr::Abs(..) => Flat::Abs,
                Expr::Log(..) => Flat::Log,
                Expr::Floor(..) => Flat::Floor,
                Expr::Ceil(..) => Flat::Ceil,
                Expr::Round(..) => Flat::Round,
                Expr::Powf(_, e) => Flat::Powf(*e),
                Expr::Erf(..) => Flat::Erf,
                Expr::Gelu(..) => Flat::Gelu,
                Expr::GeluTanh(..) => Flat::GeluTanh,
            });
            continue;
        }
        stack.push((node, true));
        match node {
            Expr::Input(_) | Expr::Scalar(_) | Expr::Const(_) => {}
            Expr::Select(c, a, b) => {
                stack.push((b, false));
                stack.push((a, false));
                stack.push((c, false));
            }
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
            | Expr::Ne(a, b) => {
                stack.push((b, false));
                stack.push((a, false));
            }
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
            | Expr::GeluTanh(a) => stack.push((a, false)),
        }
    }
    out
}

// The scalar evaluator shared by the contiguous path (lane accessor reads
// lane[k][i]) and the strided path (lane values pre-gathered per element).
// `values` is caller-owned scratch, reused across elements.
fn eval_plan<T: Scalar, F: Fn(u32) -> T>(
    plan: &[Flat],
    lane: &F,
    scalars: &[T],
    values: &mut Vec<T>,
) -> T {
    macro_rules! binary {
        ($m:ident) => {{
            let b = values.pop().expect("plan operand");
            let a = values.pop().expect("plan operand");
            values.push(a.$m(b));
        }};
    }
    macro_rules! unary {
        ($m:ident) => {{
            let a = values.pop().expect("plan operand");
            values.push(a.$m());
        }};
    }
    values.clear();
    for op in plan {
        match op {
            Flat::Input(k) => values.push(lane(*k)),
            Flat::Scalar(k) => values.push(scalars[*k as usize]),
            Flat::Const(bits) => values.push(T::from_f64(f64::from_bits(*bits))),
            Flat::Add => binary!(add),
            Flat::Sub => binary!(sub),
            Flat::Mul => binary!(mul),
            Flat::Div => binary!(div),
            Flat::Min => binary!(min),
            Flat::Max => binary!(max),
            Flat::Lt => binary!(lt),
            Flat::Le => binary!(le),
            Flat::Gt => binary!(gt),
            Flat::Ge => binary!(ge),
            Flat::Eq => binary!(eq),
            Flat::Ne => binary!(ne),
            Flat::Select => {
                let b = values.pop().expect("plan operand");
                let a = values.pop().expect("plan operand");
                let c = values.pop().expect("plan operand");
                values.push(T::pick(c, a, b));
            }
            Flat::Neg => unary!(neg),
            Flat::Sqrt => unary!(sqrt),
            Flat::Exp => unary!(exp),
            Flat::Sin => unary!(sin),
            Flat::Cos => unary!(cos),
            Flat::Tanh => unary!(tanh),
            Flat::Abs => unary!(abs),
            Flat::Log => unary!(log),
            Flat::Floor => unary!(floor),
            Flat::Ceil => unary!(ceil),
            Flat::Round => unary!(round),
            Flat::Powf(e) => {
                let a = values.pop().expect("plan operand");
                values.push(a.powf(f64::from_bits(*e)));
            }
            Flat::Erf => unary!(erf),
            Flat::Gelu => {
                let x = values.pop().expect("plan operand");
                let inner = x.mul(T::from_f64(std::f64::consts::FRAC_1_SQRT_2)).erf();
                values.push(x.mul(T::from_f64(0.5)).mul(T::from_f64(1.0).add(inner)));
            }
            Flat::GeluTanh => {
                let x = values.pop().expect("plan operand");
                let u = x
                    .add(x.mul(x).mul(x).mul(T::from_f64(0.044715)))
                    .mul(T::from_f64(0.7978845608028654));
                values.push(x.mul(T::from_f64(0.5)).mul(T::from_f64(1.0).add(u.tanh())));
            }
        }
    }
    debug_assert_eq!(values.len(), 1);
    values.pop().expect("plan result")
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
    let init = <T as Scalar>::from_f64(0.0);
    let mut outs: Vec<Vec<T>> = exprs.iter().map(|_| vec![init; n]).collect();
    let plans: Vec<Vec<Flat>> = exprs.iter().map(flatten).collect();
    let mut values: Vec<T> = Vec::new();
    match strided {
        None => {
            for i in 0..n {
                for (out, plan) in outs.iter_mut().zip(plans.iter()) {
                    out[i] =
                        eval_plan(plan, &|k| slices[k as usize][i], scalar_values, &mut values);
                }
            }
        }
        // Broadcast lanes: walk the output coordinates with an odometer so
        // each lane's element offset is incremental (no per-element div/mod).
        Some(ss) => {
            let rank = shape.len();
            let mut coord = vec![0usize; rank];
            let mut offs = vec![0usize; slices.len()];
            let mut lane_vals = vec![init; slices.len()];
            for i in 0..n {
                for (v, (slice, off)) in lane_vals.iter_mut().zip(slices.iter().zip(offs.iter())) {
                    *v = slice[*off];
                }
                for (out, plan) in outs.iter_mut().zip(plans.iter()) {
                    out[i] =
                        eval_plan(plan, &|k| lane_vals[k as usize], scalar_values, &mut values);
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
    let plan = flatten(expr);
    let mut values: Vec<T> = Vec::new();
    for _ in 0..in_n {
        for (v, (slice, off)) in lane_vals.iter_mut().zip(slices.iter().zip(offs.iter())) {
            *v = slice[*off];
        }
        let v = eval_plan(&plan, &|k| lane_vals[k as usize], &[], &mut values);
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

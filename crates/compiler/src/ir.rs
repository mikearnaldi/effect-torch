//! Elementwise expression IR and single-kernel fusion from RFC 0007.
//!
//! A fused region is a scalar expression DAG over named input lanes. On Metal,
//! `runtime::metal::emit` emits an SSA-form MSL kernel. The runtime caches one
//! compiled kernel per expression and launches it over flattened input buffers.
//! On CPU, the interpreter evaluates the same IR once per element. Metal uses
//! f32 for fused math and supports f32 or bf16 lane storage. The CPU interpreter
//! supports f32 and f64.

/// Row-major contiguous strides of `shape`, in elements.
pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// Computes strides for reading a `lane` shape into output shape `out` with
/// right-aligned broadcasting. A lane dimension of 1 or a missing leading
/// dimension gets stride 0. Returns `None` for incompatible shapes.
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

/// Whether `lane` broadcasts right-aligned into `out` with equal or unit dimensions.
pub fn broadcast_compatible(lane: &[usize], out: &[usize]) -> bool {
    lane_strides(lane, out).is_some()
}

/// Scalar expression tree for one fused value.
///
/// The IR implements `Eq + Hash` by storing floating-point constants as bit
/// patterns. An expression can therefore key the compiled-kernel cache.
/// Iterative cloning, equality, hashing, transformation, and destruction keep
/// deep fused chains off the call stack.
#[derive(Debug)]
pub enum KernelExpr {
    // Per-element input lane k.
    Input(u32),
    // Scalar input k reads a one-element tensor at offset 0. Launch-varying
    // values such as step counts and scheduled learning rates remain outside
    // the compiled-kernel cache key.
    Scalar(u32),
    // f64 bits let the Eq + Hash IR key the pipeline cache.
    Const(u64),
    Add(Box<KernelExpr>, Box<KernelExpr>),
    Sub(Box<KernelExpr>, Box<KernelExpr>),
    Mul(Box<KernelExpr>, Box<KernelExpr>),
    Div(Box<KernelExpr>, Box<KernelExpr>),
    Min(Box<KernelExpr>, Box<KernelExpr>),
    Max(Box<KernelExpr>, Box<KernelExpr>),
    // Comparisons return 1.0 or 0.0 for Select.
    Lt(Box<KernelExpr>, Box<KernelExpr>),
    Le(Box<KernelExpr>, Box<KernelExpr>),
    Gt(Box<KernelExpr>, Box<KernelExpr>),
    Ge(Box<KernelExpr>, Box<KernelExpr>),
    Eq(Box<KernelExpr>, Box<KernelExpr>),
    Ne(Box<KernelExpr>, Box<KernelExpr>),
    // `cond != 0 ? lhs : rhs` is a true select. Unlike an arithmetic mask,
    // it does not propagate NaN from the unselected side.
    Select(Box<KernelExpr>, Box<KernelExpr>, Box<KernelExpr>),
    Neg(Box<KernelExpr>),
    Sqrt(Box<KernelExpr>),
    Exp(Box<KernelExpr>),
    Sin(Box<KernelExpr>),
    Cos(Box<KernelExpr>),
    Tanh(Box<KernelExpr>),
    Abs(Box<KernelExpr>),
    Log(Box<KernelExpr>),
    Floor(Box<KernelExpr>),
    Ceil(Box<KernelExpr>),
    Round(Box<KernelExpr>),
    // The exponent uses f64 bits to keep the IR Eq + Hash. Common exponents
    // lower to multiplies or sqrt. The rest use the platform pow.
    Powf(Box<KernelExpr>, u64),
    // The CPU interpreter computes erf directly. GPU kernels use a stable ug
    // expansion because Metal has no erf.
    Erf(Box<KernelExpr>),
    // GELU uses either the exact erf form or the tanh approximation. One
    // emitted helper serves GEMM epilogues and elementwise regions.
    Gelu(Box<KernelExpr>),
    GeluTanh(Box<KernelExpr>),
}

pub type Expr = KernelExpr;

impl KernelExpr {
    // Moves child boxes into the worklist and leaves cheap leaves behind.
    // Drop uses this to avoid recursive destructor glue.
    fn drain_children(&mut self, worklist: &mut Vec<Box<KernelExpr>>) {
        fn dummy() -> Box<KernelExpr> {
            Box::new(KernelExpr::Const(0))
        }
        match self {
            KernelExpr::Input(_) | KernelExpr::Scalar(_) | KernelExpr::Const(_) => {}
            KernelExpr::Select(c, a, b) => {
                worklist.push(std::mem::replace(c, dummy()));
                worklist.push(std::mem::replace(a, dummy()));
                worklist.push(std::mem::replace(b, dummy()));
            }
            KernelExpr::Add(a, b)
            | KernelExpr::Sub(a, b)
            | KernelExpr::Mul(a, b)
            | KernelExpr::Div(a, b)
            | KernelExpr::Min(a, b)
            | KernelExpr::Max(a, b)
            | KernelExpr::Lt(a, b)
            | KernelExpr::Le(a, b)
            | KernelExpr::Gt(a, b)
            | KernelExpr::Ge(a, b)
            | KernelExpr::Eq(a, b)
            | KernelExpr::Ne(a, b) => {
                worklist.push(std::mem::replace(a, dummy()));
                worklist.push(std::mem::replace(b, dummy()));
            }
            KernelExpr::Neg(a)
            | KernelExpr::Sqrt(a)
            | KernelExpr::Exp(a)
            | KernelExpr::Sin(a)
            | KernelExpr::Cos(a)
            | KernelExpr::Tanh(a)
            | KernelExpr::Abs(a)
            | KernelExpr::Log(a)
            | KernelExpr::Floor(a)
            | KernelExpr::Ceil(a)
            | KernelExpr::Round(a)
            | KernelExpr::Powf(a, _)
            | KernelExpr::Erf(a)
            | KernelExpr::Gelu(a)
            | KernelExpr::GeluTanh(a) => worklist.push(std::mem::replace(a, dummy())),
        }
    }
}

impl Drop for KernelExpr {
    fn drop(&mut self) {
        // Default destructor glue recurses through a deep KernelExpr box chain
        // and can overflow a worker thread's stack. Drain descendants into a
        // worklist instead. Each entry has dummy leaves when it drops, so its
        // own Drop returns immediately.
        let mut worklist = Vec::new();
        self.drain_children(&mut worklist);
        while let Some(mut node) = worklist.pop() {
            node.drain_children(&mut worklist);
        }
    }
}

/// Builds `x^e`, using exact multiplies and sqrt for common exponents because
/// they are faster and more accurate than the platform pow.
pub fn pow_expr(x: KernelExpr, e: f64) -> KernelExpr {
    match e {
        0.0 => KernelExpr::cst(1.0),
        1.0 => x,
        -1.0 => KernelExpr::Div(Box::new(KernelExpr::cst(1.0)), Box::new(x)),
        2.0 => KernelExpr::Mul(Box::new(x.clone()), Box::new(x)),
        3.0 => KernelExpr::Mul(
            Box::new(KernelExpr::Mul(Box::new(x.clone()), Box::new(x.clone()))),
            Box::new(x),
        ),
        0.5 => KernelExpr::Sqrt(Box::new(x)),
        -0.5 => KernelExpr::Div(
            Box::new(KernelExpr::cst(1.0)),
            Box::new(KernelExpr::Sqrt(Box::new(x))),
        ),
        _ => KernelExpr::Powf(Box::new(x), e.to_bits()),
    }
}

/// The reduction that terminates a fused-reduce region from RFC 0007 phase 3a.
/// The reduce loop evaluates the expression for each input element and folds
/// it into an accumulator without materializing the chain's intermediate.
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

impl KernelExpr {
    pub fn cst(v: f64) -> Self {
        KernelExpr::Const(v.to_bits())
    }

    // Child references in left-to-right order.
    fn children(&self) -> Vec<&KernelExpr> {
        match self {
            KernelExpr::Input(_) | KernelExpr::Scalar(_) | KernelExpr::Const(_) => Vec::new(),
            KernelExpr::Select(c, a, b) => vec![c.as_ref(), a.as_ref(), b.as_ref()],
            KernelExpr::Add(a, b)
            | KernelExpr::Sub(a, b)
            | KernelExpr::Mul(a, b)
            | KernelExpr::Div(a, b)
            | KernelExpr::Min(a, b)
            | KernelExpr::Max(a, b)
            | KernelExpr::Lt(a, b)
            | KernelExpr::Le(a, b)
            | KernelExpr::Gt(a, b)
            | KernelExpr::Ge(a, b)
            | KernelExpr::Eq(a, b)
            | KernelExpr::Ne(a, b) => vec![a.as_ref(), b.as_ref()],
            KernelExpr::Neg(a)
            | KernelExpr::Sqrt(a)
            | KernelExpr::Exp(a)
            | KernelExpr::Sin(a)
            | KernelExpr::Cos(a)
            | KernelExpr::Tanh(a)
            | KernelExpr::Abs(a)
            | KernelExpr::Log(a)
            | KernelExpr::Floor(a)
            | KernelExpr::Ceil(a)
            | KernelExpr::Round(a)
            | KernelExpr::Powf(a, _)
            | KernelExpr::Erf(a)
            | KernelExpr::Gelu(a)
            | KernelExpr::GeluTanh(a) => vec![a.as_ref()],
        }
    }

    // Rebuilds the same variant with new children in left-to-right order.
    fn rebuild(&self, mut children: Vec<KernelExpr>) -> KernelExpr {
        let mut next = || Box::new(children.remove(0));
        match self {
            KernelExpr::Input(k) => KernelExpr::Input(*k),
            KernelExpr::Scalar(k) => KernelExpr::Scalar(*k),
            KernelExpr::Const(b) => KernelExpr::Const(*b),
            KernelExpr::Select(..) => KernelExpr::Select(next(), next(), next()),
            KernelExpr::Add(..) => KernelExpr::Add(next(), next()),
            KernelExpr::Sub(..) => KernelExpr::Sub(next(), next()),
            KernelExpr::Mul(..) => KernelExpr::Mul(next(), next()),
            KernelExpr::Div(..) => KernelExpr::Div(next(), next()),
            KernelExpr::Min(..) => KernelExpr::Min(next(), next()),
            KernelExpr::Max(..) => KernelExpr::Max(next(), next()),
            KernelExpr::Lt(..) => KernelExpr::Lt(next(), next()),
            KernelExpr::Le(..) => KernelExpr::Le(next(), next()),
            KernelExpr::Gt(..) => KernelExpr::Gt(next(), next()),
            KernelExpr::Ge(..) => KernelExpr::Ge(next(), next()),
            KernelExpr::Eq(..) => KernelExpr::Eq(next(), next()),
            KernelExpr::Ne(..) => KernelExpr::Ne(next(), next()),
            KernelExpr::Neg(..) => KernelExpr::Neg(next()),
            KernelExpr::Sqrt(..) => KernelExpr::Sqrt(next()),
            KernelExpr::Exp(..) => KernelExpr::Exp(next()),
            KernelExpr::Sin(..) => KernelExpr::Sin(next()),
            KernelExpr::Cos(..) => KernelExpr::Cos(next()),
            KernelExpr::Tanh(..) => KernelExpr::Tanh(next()),
            KernelExpr::Abs(..) => KernelExpr::Abs(next()),
            KernelExpr::Log(..) => KernelExpr::Log(next()),
            KernelExpr::Floor(..) => KernelExpr::Floor(next()),
            KernelExpr::Ceil(..) => KernelExpr::Ceil(next()),
            KernelExpr::Round(..) => KernelExpr::Round(next()),
            KernelExpr::Powf(_, e) => KernelExpr::Powf(next(), *e),
            KernelExpr::Erf(..) => KernelExpr::Erf(next()),
            KernelExpr::Gelu(..) => KernelExpr::Gelu(next()),
            KernelExpr::GeluTanh(..) => KernelExpr::GeluTanh(next()),
        }
    }

    // Iterative postorder rebuild. `f` may replace a leaf with Some or keep
    // it with None. Internal nodes rebuild from transformed children. Tree
    // transforms stay off the call stack for deep fused regions.
    fn transform(&self, f: &mut dyn FnMut(&KernelExpr) -> Option<KernelExpr>) -> KernelExpr {
        let mut stack: Vec<(&KernelExpr, bool)> = vec![(self, false)];
        let mut out: Vec<KernelExpr> = Vec::new();
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

    /// Remaps per-element lane indices through `remap`. The map must cover
    /// every referenced lane.
    pub fn remap_lanes(&self, remap: &std::collections::HashMap<u32, u32>) -> Self {
        self.remap_inputs(&mut |k| remap[&k])
    }

    /// Counts expression-tree nodes. Shared subtrees count once per occurrence.
    /// This bounds emitted kernel size rather than SSA values.
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
            KernelExpr::Input(k) => Some(KernelExpr::Input(f(*k))),
            _ => None,
        })
    }

    /// Inlines `replacement` for `lane` and remaps other lanes through
    /// `remap`. Multi-output merging uses this to absorb a shared prefix into
    /// its continuations. Each occurrence uses its original index, so a
    /// remapped index that equals `lane` is not mistaken for the inlined lane.
    /// The replacement indices must already use the merged namespace.
    pub fn merge_lane(
        &self,
        lane: u32,
        replacement: &KernelExpr,
        remap: &std::collections::HashMap<u32, u32>,
    ) -> Self {
        self.transform(&mut |e| match e {
            KernelExpr::Input(k) if *k == lane => Some(replacement.clone()),
            KernelExpr::Input(k) => Some(KernelExpr::Input(remap[k])),
            _ => None,
        })
    }
}

// Clone, equality, and hashing use a manual iterative postorder plan because
// derived implementations would recurse through deep box chains.
impl Clone for KernelExpr {
    fn clone(&self) -> Self {
        self.transform(&mut |_| None)
    }
}

impl PartialEq for KernelExpr {
    fn eq(&self, other: &Self) -> bool {
        flatten(self) == flatten(other)
    }
}

impl Eq for KernelExpr {}

impl std::hash::Hash for KernelExpr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for op in flatten(self) {
            op.hash(state);
        }
    }
}

/// Scalar arithmetic interface for fusion IR evaluators, including the CPU
/// interpreter. Comparisons return 1.0 or 0.0, `pick` selects a value, and
/// `from_f64` narrows constants.
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

// A fused expression flattened into a postorder plan. The interpreter runs a
// deep KernelExpr once per element, so a recursive walk would consume stack
// space for every element. Prepared CPU programs retain this plan and evaluate
// it with caller-owned scratch.
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

// Iterative postorder flattening keeps the traversal off the call stack, as
// does the evaluator below.
fn flatten_into(e: &KernelExpr, out: &mut Vec<Flat>) {
    let mut stack: Vec<(&KernelExpr, bool)> = vec![(e, false)];
    while let Some((node, processed)) = stack.pop() {
        if processed {
            out.push(match node {
                KernelExpr::Input(k) => Flat::Input(*k),
                KernelExpr::Scalar(k) => Flat::Scalar(*k),
                KernelExpr::Const(bits) => Flat::Const(*bits),
                KernelExpr::Add(..) => Flat::Add,
                KernelExpr::Sub(..) => Flat::Sub,
                KernelExpr::Mul(..) => Flat::Mul,
                KernelExpr::Div(..) => Flat::Div,
                KernelExpr::Min(..) => Flat::Min,
                KernelExpr::Max(..) => Flat::Max,
                KernelExpr::Lt(..) => Flat::Lt,
                KernelExpr::Le(..) => Flat::Le,
                KernelExpr::Gt(..) => Flat::Gt,
                KernelExpr::Ge(..) => Flat::Ge,
                KernelExpr::Eq(..) => Flat::Eq,
                KernelExpr::Ne(..) => Flat::Ne,
                KernelExpr::Select(..) => Flat::Select,
                KernelExpr::Neg(..) => Flat::Neg,
                KernelExpr::Sqrt(..) => Flat::Sqrt,
                KernelExpr::Exp(..) => Flat::Exp,
                KernelExpr::Sin(..) => Flat::Sin,
                KernelExpr::Cos(..) => Flat::Cos,
                KernelExpr::Tanh(..) => Flat::Tanh,
                KernelExpr::Abs(..) => Flat::Abs,
                KernelExpr::Log(..) => Flat::Log,
                KernelExpr::Floor(..) => Flat::Floor,
                KernelExpr::Ceil(..) => Flat::Ceil,
                KernelExpr::Round(..) => Flat::Round,
                KernelExpr::Powf(_, e) => Flat::Powf(*e),
                KernelExpr::Erf(..) => Flat::Erf,
                KernelExpr::Gelu(..) => Flat::Gelu,
                KernelExpr::GeluTanh(..) => Flat::GeluTanh,
            });
            continue;
        }
        stack.push((node, true));
        match node {
            KernelExpr::Input(_) | KernelExpr::Scalar(_) | KernelExpr::Const(_) => {}
            KernelExpr::Select(c, a, b) => {
                stack.push((b, false));
                stack.push((a, false));
                stack.push((c, false));
            }
            KernelExpr::Add(a, b)
            | KernelExpr::Sub(a, b)
            | KernelExpr::Mul(a, b)
            | KernelExpr::Div(a, b)
            | KernelExpr::Min(a, b)
            | KernelExpr::Max(a, b)
            | KernelExpr::Lt(a, b)
            | KernelExpr::Le(a, b)
            | KernelExpr::Gt(a, b)
            | KernelExpr::Ge(a, b)
            | KernelExpr::Eq(a, b)
            | KernelExpr::Ne(a, b) => {
                stack.push((b, false));
                stack.push((a, false));
            }
            KernelExpr::Neg(a)
            | KernelExpr::Sqrt(a)
            | KernelExpr::Exp(a)
            | KernelExpr::Sin(a)
            | KernelExpr::Cos(a)
            | KernelExpr::Tanh(a)
            | KernelExpr::Abs(a)
            | KernelExpr::Log(a)
            | KernelExpr::Floor(a)
            | KernelExpr::Ceil(a)
            | KernelExpr::Round(a)
            | KernelExpr::Powf(a, _)
            | KernelExpr::Erf(a)
            | KernelExpr::Gelu(a)
            | KernelExpr::GeluTanh(a) => stack.push((a, false)),
        }
    }
}

fn flatten(e: &KernelExpr) -> Vec<Flat> {
    let mut out = Vec::new();
    flatten_into(e, &mut out);
    out
}

/// Immutable flattened expression plans retained by a compiled CPU command.
///
/// `scratch_elements` is the exact native scalar count needed to cache input
/// lanes and evaluate the deepest plan. Preparation allocates this storage.
/// Evaluation does not allocate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuFusionProgram {
    ops: Box<[Flat]>,
    plan_ends: Box<[usize]>,
    input_count: usize,
    scalar_count: usize,
    value_scratch_len: usize,
}

impl CpuFusionProgram {
    /// Flattens every expression into one contiguous postorder operation list.
    /// Preparation computes exact lane counts and maximum value-stack depth,
    /// so evaluation neither allocates nor measures them again.
    pub fn new(exprs: &[KernelExpr]) -> Self {
        let mut ops = Vec::new();
        let mut plan_ends = Vec::with_capacity(exprs.len());
        let mut input_count = 0usize;
        let mut scalar_count = 0usize;
        let mut value_scratch_len = 0usize;
        for expr in exprs {
            let start = ops.len();
            flatten_into(expr, &mut ops);
            let mut depth = 0usize;
            for op in &ops[start..] {
                match op {
                    Flat::Input(k) => {
                        input_count = input_count.max(*k as usize + 1);
                        depth += 1;
                    }
                    Flat::Scalar(k) => {
                        scalar_count = scalar_count.max(*k as usize + 1);
                        depth += 1;
                    }
                    Flat::Const(_) => depth += 1,
                    Flat::Add
                    | Flat::Sub
                    | Flat::Mul
                    | Flat::Div
                    | Flat::Min
                    | Flat::Max
                    | Flat::Lt
                    | Flat::Le
                    | Flat::Gt
                    | Flat::Ge
                    | Flat::Eq
                    | Flat::Ne => {
                        debug_assert!(depth >= 2);
                        depth -= 1;
                    }
                    Flat::Select => {
                        debug_assert!(depth >= 3);
                        depth -= 2;
                    }
                    Flat::Neg
                    | Flat::Sqrt
                    | Flat::Exp
                    | Flat::Sin
                    | Flat::Cos
                    | Flat::Tanh
                    | Flat::Abs
                    | Flat::Log
                    | Flat::Floor
                    | Flat::Ceil
                    | Flat::Round
                    | Flat::Powf(_)
                    | Flat::Erf
                    | Flat::Gelu
                    | Flat::GeluTanh => debug_assert!(depth >= 1),
                }
                value_scratch_len = value_scratch_len.max(depth);
            }
            debug_assert_eq!(depth, 1);
            plan_ends.push(ops.len());
        }
        Self {
            ops: ops.into_boxed_slice(),
            plan_ends: plan_ends.into_boxed_slice(),
            input_count,
            scalar_count,
            value_scratch_len,
        }
    }

    /// Number of fused outputs, with one plan per expression.
    pub fn output_count(&self) -> usize {
        self.plan_ends.len()
    }

    /// Number of per-element input lanes referenced across all plans.
    pub fn input_count(&self) -> usize {
        self.input_count
    }

    /// Number of launch-varying scalar lanes referenced across all plans.
    pub fn scalar_count(&self) -> usize {
        self.scalar_count
    }

    /// Maximum value-stack depth any plan needs during evaluation.
    pub fn value_scratch_len(&self) -> usize {
        self.value_scratch_len
    }

    /// Total caller-owned scratch elements for cached input lanes and the value stack.
    pub fn scratch_elements(&self) -> usize {
        self.input_count
            .checked_add(self.value_scratch_len)
            .expect("CPU fusion scratch element count overflow")
    }

    fn plan(&self, output: usize) -> &[Flat] {
        let end = self.plan_ends[output];
        let start = output
            .checked_sub(1)
            .map_or(0, |previous| self.plan_ends[previous]);
        &self.ops[start..end]
    }

    /// Evaluates one output plan with caller-supplied lane accessors and value
    /// stack. Callers must provide at least `value_scratch_len()` elements.
    pub fn evaluate<T: Scalar>(
        &self,
        output: usize,
        mut lane: impl FnMut(u32) -> T,
        mut scalar: impl FnMut(u32) -> T,
        values: &mut [T],
    ) -> T {
        assert!(values.len() >= self.value_scratch_len);
        eval_plan(self.plan(output), &mut lane, &mut scalar, values)
    }
}

fn eval_plan<T: Scalar>(
    plan: &[Flat],
    lane: &mut impl FnMut(u32) -> T,
    scalar: &mut impl FnMut(u32) -> T,
    values: &mut [T],
) -> T {
    macro_rules! binary {
        ($m:ident, $depth:ident) => {{
            debug_assert!($depth >= 2);
            values[$depth - 2] = values[$depth - 2].$m(values[$depth - 1]);
            $depth -= 1;
        }};
    }
    macro_rules! unary {
        ($m:ident, $depth:ident) => {{
            debug_assert!($depth >= 1);
            values[$depth - 1] = values[$depth - 1].$m();
        }};
    }
    let mut depth = 0usize;
    for op in plan {
        match op {
            Flat::Input(k) => {
                values[depth] = lane(*k);
                depth += 1;
            }
            Flat::Scalar(k) => {
                values[depth] = scalar(*k);
                depth += 1;
            }
            Flat::Const(bits) => {
                values[depth] = T::from_f64(f64::from_bits(*bits));
                depth += 1;
            }
            Flat::Add => binary!(add, depth),
            Flat::Sub => binary!(sub, depth),
            Flat::Mul => binary!(mul, depth),
            Flat::Div => binary!(div, depth),
            Flat::Min => binary!(min, depth),
            Flat::Max => binary!(max, depth),
            Flat::Lt => binary!(lt, depth),
            Flat::Le => binary!(le, depth),
            Flat::Gt => binary!(gt, depth),
            Flat::Ge => binary!(ge, depth),
            Flat::Eq => binary!(eq, depth),
            Flat::Ne => binary!(ne, depth),
            Flat::Select => {
                debug_assert!(depth >= 3);
                values[depth - 3] =
                    T::pick(values[depth - 3], values[depth - 2], values[depth - 1]);
                depth -= 2;
            }
            Flat::Neg => unary!(neg, depth),
            Flat::Sqrt => unary!(sqrt, depth),
            Flat::Exp => unary!(exp, depth),
            Flat::Sin => unary!(sin, depth),
            Flat::Cos => unary!(cos, depth),
            Flat::Tanh => unary!(tanh, depth),
            Flat::Abs => unary!(abs, depth),
            Flat::Log => unary!(log, depth),
            Flat::Floor => unary!(floor, depth),
            Flat::Ceil => unary!(ceil, depth),
            Flat::Round => unary!(round, depth),
            Flat::Powf(e) => {
                debug_assert!(depth >= 1);
                values[depth - 1] = values[depth - 1].powf(f64::from_bits(*e));
            }
            Flat::Erf => unary!(erf, depth),
            Flat::Gelu => {
                debug_assert!(depth >= 1);
                let x = values[depth - 1];
                let inner = x.mul(T::from_f64(std::f64::consts::FRAC_1_SQRT_2)).erf();
                values[depth - 1] = x.mul(T::from_f64(0.5)).mul(T::from_f64(1.0).add(inner));
            }
            Flat::GeluTanh => {
                debug_assert!(depth >= 1);
                let x = values[depth - 1];
                let u = x
                    .add(x.mul(x).mul(x).mul(T::from_f64(0.044715)))
                    .mul(T::from_f64(0.7978845608028654));
                values[depth - 1] = x.mul(T::from_f64(0.5)).mul(T::from_f64(1.0).add(u.tanh()));
            }
        }
    }
    debug_assert_eq!(depth, 1);
    values[0]
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

/// Interprets a prepared program over `n` elements and writes every output.
/// `slices` contains per-element input lanes. `strides` optionally supplies
/// each lane's broadcast strides against `shape`. `None` means contiguous.
/// `scalar_values` contains launch-varying scalar lanes. The caller owns all
/// scratch. [`CpuFusionProgram::scratch_elements`] gives its exact size.
pub fn interpret_core_into<T: Scalar>(
    program: &CpuFusionProgram,
    slices: &[&[T]],
    strides: Option<&[Vec<usize>]>,
    scalar_values: &[T],
    n: usize,
    shape: &[usize],
    outputs: &mut [&mut [T]],
    scratch: &mut [T],
) {
    assert_eq!(outputs.len(), program.output_count());
    assert!(slices.len() >= program.input_count());
    assert!(scalar_values.len() >= program.scalar_count());
    assert!(outputs.iter().all(|output| output.len() >= n));
    assert!(scratch.len() >= program.scratch_elements());
    if let Some(strides) = strides {
        assert_eq!(strides.len(), slices.len());
        assert!(strides.iter().all(|lane| lane.len() == shape.len()));
    }
    let (lane_values, values) = scratch.split_at_mut(program.input_count());
    for index in 0..n {
        for lane in 0..program.input_count() {
            let offset = strides.map_or(index, |strides| {
                strided_offset(index, shape, &strides[lane])
            });
            lane_values[lane] = slices[lane][offset];
        }
        for (output_index, output) in outputs.iter_mut().enumerate() {
            output[index] = program.evaluate(
                output_index,
                |lane| lane_values[lane as usize],
                |scalar| scalar_values[scalar as usize],
                values,
            );
        }
    }
}

/// Prepares a program, allocates outputs and scratch, and calls
/// [`interpret_core_into`]. Returns one vector per expression.
pub fn interpret_core<T: Scalar>(
    exprs: &[KernelExpr],
    slices: &[&[T]],
    strides: Option<&[Vec<usize>]>,
    scalar_values: &[T],
    n: usize,
    shape: &[usize],
) -> Vec<Vec<T>> {
    let program = CpuFusionProgram::new(exprs);
    let init = <T as Scalar>::from_f64(0.0);
    let mut outs: Vec<Vec<T>> = exprs.iter().map(|_| vec![init; n]).collect();
    let mut output_slices = outs.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>();
    let mut scratch = vec![init; program.scratch_elements()];
    interpret_core_into(
        &program,
        slices,
        strides,
        scalar_values,
        n,
        shape,
        &mut output_slices,
        &mut scratch,
    );
    outs
}

fn reduce_output_offset(
    index: usize,
    in_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    out_shape: &[usize],
) -> usize {
    let mut remainder = index;
    let mut out_dimension = out_shape.len();
    let mut out_stride = 1usize;
    let mut out_offset = 0usize;
    for dimension in (0..in_shape.len()).rev() {
        let width = in_shape[dimension].max(1);
        let coordinate = remainder % width;
        remainder /= width;
        if !dims.contains(&dimension) {
            out_dimension -= 1;
            out_offset += coordinate * out_stride;
            out_stride *= out_shape[out_dimension];
        } else if keepdims {
            out_dimension -= 1;
            out_stride *= out_shape[out_dimension];
        }
    }
    debug_assert_eq!(out_dimension, 0);
    out_offset
}

/// Interprets a fused-reduce program. The reduce loop evaluates the expression
/// per input element and folds it into the output cell's accumulator without
/// materializing an elementwise intermediate. `dims` contains sorted reduced
/// dimensions. `keepdims` keeps them as size 1 in `out_shape`. `Mean`
/// divides by the reduced extent after the fold.
#[allow(clippy::too_many_arguments)]
pub fn interpret_reduce_core_into<T: Scalar>(
    op: ReduceOp,
    program: &CpuFusionProgram,
    slices: &[&[T]],
    strides: &[Vec<usize>],
    in_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    out_shape: &[usize],
    output: &mut [T],
    scratch: &mut [T],
) {
    assert_eq!(program.output_count(), 1);
    assert_eq!(program.scalar_count(), 0);
    assert!(slices.len() >= program.input_count());
    assert_eq!(strides.len(), slices.len());
    assert!(strides.iter().all(|lane| lane.len() == in_shape.len()));
    assert_eq!(output.len(), out_shape.iter().product::<usize>());
    assert!(scratch.len() >= program.scratch_elements());
    let init = <T as Scalar>::from_f64(op.init());
    output.fill(init);
    let (lane_values, values) = scratch.split_at_mut(program.input_count());
    for index in 0..in_shape.iter().product() {
        for lane in 0..program.input_count() {
            lane_values[lane] = slices[lane][strided_offset(index, in_shape, &strides[lane])];
        }
        let value = program.evaluate(
            0,
            |lane| lane_values[lane as usize],
            |_| unreachable!("fused reduce scalar lane"),
            values,
        );
        let output_index = reduce_output_offset(index, in_shape, dims, keepdims, out_shape);
        output[output_index] = op.fold(output[output_index], value);
    }
    if op == ReduceOp::Mean {
        let extent: usize = dims.iter().map(|&dimension| in_shape[dimension]).product();
        let extent = <T as Scalar>::from_f64(extent as f64);
        for value in output {
            *value = Scalar::div(*value, extent);
        }
    }
}

/// Owning wrapper around [`interpret_reduce_core_into`].
#[allow(clippy::too_many_arguments)]
pub fn interpret_reduce_core<T: Scalar>(
    op: ReduceOp,
    expr: &KernelExpr,
    slices: &[&[T]],
    strides: &[Vec<usize>],
    in_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    out_shape: &[usize],
) -> Vec<T> {
    let program = CpuFusionProgram::new(std::slice::from_ref(expr));
    let init = <T as Scalar>::from_f64(op.init());
    let mut output = vec![init; out_shape.iter().product()];
    let mut scratch = vec![init; program.scratch_elements()];
    interpret_reduce_core_into(
        op,
        &program,
        slices,
        strides,
        in_shape,
        dims,
        keepdims,
        out_shape,
        &mut output,
        &mut scratch,
    );
    output
}
// The fused AdamW update uses three expressions over lanes [param, grad, m, v]
// and scalar lanes [lr, 1 - beta1^t, 1 - beta2^t]. It matches the composed
// update's operation order. Keeping step-dependent values in scalar lanes
// makes the compiled kernel stable across steps.
/// Returns the fused AdamW expressions `[param', m', v']` with the lane layout
/// described above.
pub fn adamw_exprs(beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> [KernelExpr; 3] {
    adamw_exprs_with(
        beta1,
        beta2,
        eps,
        weight_decay,
        KernelExpr::Scalar(0),
        KernelExpr::Scalar(1),
        KernelExpr::Scalar(2),
    )
}

fn adamw_exprs_with(
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    lr: KernelExpr,
    c1: KernelExpr,
    c2: KernelExpr,
) -> [KernelExpr; 3] {
    let (p, g, m, v) = (
        KernelExpr::Input(0),
        KernelExpr::Input(1),
        KernelExpr::Input(2),
        KernelExpr::Input(3),
    );
    let next_m = KernelExpr::Add(
        Box::new(KernelExpr::Mul(
            Box::new(m),
            Box::new(KernelExpr::cst(beta1)),
        )),
        Box::new(KernelExpr::Mul(
            Box::new(g.clone()),
            Box::new(KernelExpr::cst(1.0 - beta1)),
        )),
    );
    let next_v = KernelExpr::Add(
        Box::new(KernelExpr::Mul(
            Box::new(v),
            Box::new(KernelExpr::cst(beta2)),
        )),
        Box::new(KernelExpr::Mul(
            Box::new(KernelExpr::Mul(Box::new(g.clone()), Box::new(g))),
            Box::new(KernelExpr::cst(1.0 - beta2)),
        )),
    );
    let m_hat = KernelExpr::Div(Box::new(next_m.clone()), Box::new(c1));
    let v_hat = KernelExpr::Div(Box::new(next_v.clone()), Box::new(c2));
    let adjusted = KernelExpr::Mul(
        Box::new(KernelExpr::Div(
            Box::new(m_hat),
            Box::new(KernelExpr::Add(
                Box::new(KernelExpr::Sqrt(Box::new(v_hat))),
                Box::new(KernelExpr::cst(eps)),
            )),
        )),
        Box::new(lr.clone()),
    );
    let base = if weight_decay == 0.0 {
        p
    } else {
        KernelExpr::Mul(
            Box::new(p),
            Box::new(KernelExpr::Sub(
                Box::new(KernelExpr::cst(1.0)),
                Box::new(KernelExpr::Mul(
                    Box::new(lr),
                    Box::new(KernelExpr::cst(weight_decay)),
                )),
            )),
        )
    };
    [
        KernelExpr::Sub(Box::new(base), Box::new(adjusted)),
        next_m,
        next_v,
    ]
}

// The fused momentum-SGD update uses lanes [param, grad, velocity] and scalar
// lanes [lr, first]. It matches the composed update, including first-step
// v = g initialization selected by the zero-dimensional `first` flag.
/// Returns fused momentum-SGD expressions `[param', velocity']` with tensor
/// lanes [param, grad, velocity] and scalar lanes [lr, first].
pub fn sgd_exprs(
    momentum: f64,
    dampening: f64,
    nesterov: bool,
    weight_decay: f64,
) -> [KernelExpr; 2] {
    sgd_exprs_with(
        momentum,
        dampening,
        nesterov,
        weight_decay,
        KernelExpr::Scalar(0),
        KernelExpr::Scalar(1),
    )
}

fn sgd_exprs_with(
    momentum: f64,
    dampening: f64,
    nesterov: bool,
    weight_decay: f64,
    lr: KernelExpr,
    first: KernelExpr,
) -> [KernelExpr; 2] {
    let (p, g, v) = (
        KernelExpr::Input(0),
        KernelExpr::Input(1),
        KernelExpr::Input(2),
    );
    let gp = if weight_decay == 0.0 {
        g
    } else {
        KernelExpr::Add(
            Box::new(g),
            Box::new(KernelExpr::Mul(
                Box::new(p.clone()),
                Box::new(KernelExpr::cst(weight_decay)),
            )),
        )
    };
    let continued = KernelExpr::Add(
        Box::new(KernelExpr::Mul(
            Box::new(v),
            Box::new(KernelExpr::cst(momentum)),
        )),
        Box::new(KernelExpr::Mul(
            Box::new(gp.clone()),
            Box::new(KernelExpr::cst(1.0 - dampening)),
        )),
    );
    let next_v = KernelExpr::Select(
        Box::new(KernelExpr::Gt(
            Box::new(first),
            Box::new(KernelExpr::cst(0.5)),
        )),
        Box::new(gp.clone()),
        Box::new(continued),
    );
    let used = if nesterov {
        KernelExpr::Add(
            Box::new(gp),
            Box::new(KernelExpr::Mul(
                Box::new(next_v.clone()),
                Box::new(KernelExpr::cst(momentum)),
            )),
        )
    } else {
        next_v.clone()
    };
    [
        KernelExpr::Sub(
            Box::new(p),
            Box::new(KernelExpr::Mul(Box::new(used), Box::new(lr))),
        ),
        next_v,
    ]
}

/// Whether fusion supports a device and dtype pair. CPU supports f32 and f64.
/// Metal supports f32 and bf16. CUDA fusion is not lowered yet.
pub fn is_fusion_supported(
    device: &effect_torch_graph::Device,
    dtype: effect_torch_runtime::DType,
) -> bool {
    match device {
        effect_torch_graph::Device::Cpu(_) => matches!(
            dtype,
            effect_torch_runtime::DType::F32 | effect_torch_runtime::DType::F64
        ),
        effect_torch_graph::Device::Metal(_) => matches!(
            dtype,
            effect_torch_runtime::DType::F32 | effect_torch_runtime::DType::BF16
        ),
        effect_torch_graph::Device::Cuda(_) => false,
    }
}

#[cfg(test)]
mod interpreter_tests {
    use super::*;

    #[test]
    fn prepared_into_paths_match_wrappers_with_exact_scratch() {
        let exprs = [
            KernelExpr::Add(
                Box::new(KernelExpr::Input(0)),
                Box::new(KernelExpr::Input(1)),
            ),
            KernelExpr::Mul(
                Box::new(KernelExpr::Input(0)),
                Box::new(KernelExpr::Scalar(0)),
            ),
        ];
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [10.0f32, 20.0, 30.0];
        let strides = [vec![3, 1], vec![0, 1]];
        let wrapped = interpret_core(&exprs, &[&a, &b], Some(&strides), &[0.5], 6, &[2, 3]);
        let program = CpuFusionProgram::new(&exprs);
        assert_eq!(program.input_count(), 2);
        assert_eq!(program.value_scratch_len(), 2);
        assert_eq!(program.scratch_elements(), 4);
        let mut first = [0.0f32; 6];
        let mut second = [0.0f32; 6];
        let mut outputs: [&mut [f32]; 2] = [&mut first, &mut second];
        let mut scratch = [0.0f32; 4];
        interpret_core_into(
            &program,
            &[&a, &b],
            Some(&strides),
            &[0.5],
            6,
            &[2, 3],
            &mut outputs,
            &mut scratch,
        );
        assert_eq!(first.as_slice(), wrapped[0]);
        assert_eq!(second.as_slice(), wrapped[1]);

        let reduce_expr = KernelExpr::Mul(
            Box::new(KernelExpr::Input(0)),
            Box::new(KernelExpr::Input(0)),
        );
        let reduce_program = CpuFusionProgram::new(std::slice::from_ref(&reduce_expr));
        assert_eq!(reduce_program.scratch_elements(), 3);
        let wrapped_reduce = interpret_reduce_core(
            ReduceOp::Mean,
            &reduce_expr,
            &[&a],
            &[vec![3, 1]],
            &[2, 3],
            &[1],
            false,
            &[2],
        );
        let mut reduced = [0.0f32; 2];
        let mut reduce_scratch = [0.0f32; 3];
        interpret_reduce_core_into(
            ReduceOp::Mean,
            &reduce_program,
            &[&a],
            &[vec![3, 1]],
            &[2, 3],
            &[1],
            false,
            &[2],
            &mut reduced,
            &mut reduce_scratch,
        );
        assert_eq!(reduced.as_slice(), wrapped_reduce);
    }
}

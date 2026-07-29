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

impl Expr {
    pub fn cst(v: f64) -> Self {
        Expr::Const(v.to_bits())
    }

    /// Adds `base` to every per-element lane index (used when merging two
    /// regions whose lane lists are concatenated).
    pub fn shift_inputs(&self, base: u32) -> Self {
        match self {
            Expr::Input(k) => Expr::Input(k + base),
            Expr::Scalar(k) => Expr::Scalar(*k),
            Expr::Const(b) => Expr::Const(*b),
            Expr::Add(a, b) => Expr::Add(Box::new(a.shift_inputs(base)), Box::new(b.shift_inputs(base))),
            Expr::Sub(a, b) => Expr::Sub(Box::new(a.shift_inputs(base)), Box::new(b.shift_inputs(base))),
            Expr::Mul(a, b) => Expr::Mul(Box::new(a.shift_inputs(base)), Box::new(b.shift_inputs(base))),
            Expr::Div(a, b) => Expr::Div(Box::new(a.shift_inputs(base)), Box::new(b.shift_inputs(base))),
            Expr::Min(a, b) => Expr::Min(Box::new(a.shift_inputs(base)), Box::new(b.shift_inputs(base))),
            Expr::Max(a, b) => Expr::Max(Box::new(a.shift_inputs(base)), Box::new(b.shift_inputs(base))),
            Expr::Neg(a) => Expr::Neg(Box::new(a.shift_inputs(base))),
            Expr::Sqrt(a) => Expr::Sqrt(Box::new(a.shift_inputs(base))),
            Expr::Exp(a) => Expr::Exp(Box::new(a.shift_inputs(base))),
            Expr::Sin(a) => Expr::Sin(Box::new(a.shift_inputs(base))),
            Expr::Cos(a) => Expr::Cos(Box::new(a.shift_inputs(base))),
            Expr::Tanh(a) => Expr::Tanh(Box::new(a.shift_inputs(base))),
            Expr::Abs(a) => Expr::Abs(Box::new(a.shift_inputs(base))),
            Expr::Log(a) => Expr::Log(Box::new(a.shift_inputs(base))),
            Expr::Floor(a) => Expr::Floor(Box::new(a.shift_inputs(base))),
            Expr::Ceil(a) => Expr::Ceil(Box::new(a.shift_inputs(base))),
            Expr::Round(a) => Expr::Round(Box::new(a.shift_inputs(base))),
            Expr::Powf(a, e) => Expr::Powf(Box::new(a.shift_inputs(base)), *e),
            Expr::Erf(a) => Expr::Erf(Box::new(a.shift_inputs(base))),
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
        }
    };
}
impl_scalar!(f32, libm::erff);
impl_scalar!(f64, libm::erf);

fn eval_at<T: Scalar>(e: &Expr, i: usize, inputs: &[&[T]], scalars: &[T]) -> T {
    match e {
        Expr::Input(k) => inputs[*k as usize][i],
        Expr::Scalar(k) => scalars[*k as usize],
        Expr::Const(bits) => T::from_f64(f64::from_bits(*bits)),
        Expr::Add(a, b) => eval_at(a, i, inputs, scalars).add(eval_at(b, i, inputs, scalars)),
        Expr::Sub(a, b) => eval_at(a, i, inputs, scalars).sub(eval_at(b, i, inputs, scalars)),
        Expr::Mul(a, b) => eval_at(a, i, inputs, scalars).mul(eval_at(b, i, inputs, scalars)),
        Expr::Div(a, b) => eval_at(a, i, inputs, scalars).div(eval_at(b, i, inputs, scalars)),
        Expr::Min(a, b) => eval_at(a, i, inputs, scalars).min(eval_at(b, i, inputs, scalars)),
        Expr::Max(a, b) => eval_at(a, i, inputs, scalars).max(eval_at(b, i, inputs, scalars)),
        Expr::Neg(a) => eval_at(a, i, inputs, scalars).neg(),
        Expr::Sqrt(a) => eval_at(a, i, inputs, scalars).sqrt(),
        Expr::Exp(a) => eval_at(a, i, inputs, scalars).exp(),
        Expr::Sin(a) => eval_at(a, i, inputs, scalars).sin(),
        Expr::Cos(a) => eval_at(a, i, inputs, scalars).cos(),
        Expr::Tanh(a) => eval_at(a, i, inputs, scalars).tanh(),
        Expr::Abs(a) => eval_at(a, i, inputs, scalars).abs(),
        Expr::Log(a) => eval_at(a, i, inputs, scalars).log(),
        Expr::Floor(a) => eval_at(a, i, inputs, scalars).floor(),
        Expr::Ceil(a) => eval_at(a, i, inputs, scalars).ceil(),
        Expr::Round(a) => eval_at(a, i, inputs, scalars).round(),
        Expr::Powf(a, e) => eval_at(a, i, inputs, scalars).powf(f64::from_bits(*e)),
        Expr::Erf(a) => eval_at(a, i, inputs, scalars).erf(),
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
    scalars: &[Tensor],
    n: usize,
    shape: &[usize],
) -> candle_core::Result<Vec<Tensor>> {
    let guards = cpu_input_slices::<T>(inputs)?;
    let mut slices: Vec<&[T]> = Vec::with_capacity(guards.len());
    for (storage, offset) in &guards {
        let cpu = match &**storage {
            Storage::Cpu(cpu) => cpu,
            _ => {
                return Err(candle_core::Error::Msg(
                    "fusion: expected CPU storage".to_string(),
                ))
            }
        };
        let data = cpu.as_slice::<T>()?;
        slices.push(&data[*offset..*offset + n]);
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
    let mut outs = Vec::with_capacity(exprs.len());
    for expr in exprs {
        let mut out = vec![<T as Scalar>::from_f64(0.0); n];
        for (i, o) in out.iter_mut().enumerate() {
            *o = eval_at(expr, i, &slices, &scalar_values);
        }
        outs.push(Tensor::from_vec(out, shape, &Device::Cpu)?);
    }
    Ok(outs)
}

#[cfg(any(target_os = "macos", feature = "cuda"))]
fn f32_cst(v: f32) -> candle_core::Result<ug::Const> {
    v.try_into()
        .map_err(|e| candle_core::Error::Msg(format!("fusion: {e}")))
}

// Lowers the IR to a ug SSA kernel with one store per output expression.
// Lane k reads from pointer argument k. The element count is baked in as a
// constant (the pipeline cache is keyed by it): Metal kernel functions
// cannot take plain scalar arguments, and baking enables constant folding.
// Loads are clamped to n-1 so the trailing partial block recomputes the
// last element harmlessly instead of reading out of bounds.
#[cfg(any(target_os = "macos", feature = "cuda"))]
fn build_kernel(
    exprs: &[Expr],
    num_inputs: usize,
    num_scalars: usize,
    n: usize,
    dtype: ug::DType,
) -> candle_core::Result<ug::lang::ssa::Kernel> {
    use ug::lang::ssa::{BinaryOp, Instr as I, Special};

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

    let mut lowered_lanes: HashMap<u32, ug::block::Id> = HashMap::new();
    fn lower(
        e: &Expr,
        b: &mut ug::block::Block,
        lanes: &[ug::block::Id],
        num_inputs: usize,
        lowered_lanes: &mut HashMap<u32, ug::block::Id>,
        offset: ug::block::Id,
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
                        offset: offset.to_a(),
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
                let a = lower(a, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                let x = lower(x, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                b.binary(BinaryOp::Add, a, x, dtype)
            }
            Expr::Sub(a, x) => {
                let a = lower(a, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                let x = lower(x, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                b.binary(BinaryOp::Sub, a, x, dtype)
            }
            Expr::Mul(a, x) => {
                let a = lower(a, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                let x = lower(x, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                b.binary(BinaryOp::Mul, a, x, dtype)
            }
            Expr::Div(a, x) => {
                let a = lower(a, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                let x = lower(x, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                b.binary(BinaryOp::Div, a, x, dtype)
            }
            Expr::Min(a, x) => {
                let a = lower(a, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                let x = lower(x, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                b.binary(BinaryOp::Min, a, x, dtype)
            }
            Expr::Max(a, x) => {
                let a = lower(a, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                let x = lower(x, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                b.binary(BinaryOp::Max, a, x, dtype)
            }
            Expr::Neg(a) => {
                let a = lower(a, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                b.unary(UnaryOp::Neg, a, dtype)
            }
            Expr::Sqrt(a) => {
                let a = lower(a, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                b.unary(UnaryOp::Sqrt, a, dtype)
            }
            Expr::Exp(a) => {
                let a = lower(a, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                b.unary(UnaryOp::Exp, a, dtype)
            }
            Expr::Sin(a) => {
                let a = lower(a, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                b.unary(UnaryOp::Sin, a, dtype)
            }
            Expr::Cos(a) => {
                let a = lower(a, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                b.unary(UnaryOp::Cos, a, dtype)
            }
            Expr::Tanh(a) => {
                let x = lower(a, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                b.unary(UnaryOp::Tanh, x, dtype)
            }
            Expr::Abs(a) => {
                let x = lower(a, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                b.unary(UnaryOp::Abs, x, dtype)
            }
            Expr::Log(a) => {
                let x = lower(a, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                b.unary(UnaryOp::Log, x, dtype)
            }
            Expr::Floor(a) => {
                let x = lower(a, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                b.unary(UnaryOp::Floor, x, dtype)
            }
            Expr::Ceil(a) => {
                let x = lower(a, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                b.unary(UnaryOp::Ceil, x, dtype)
            }
            Expr::Round(a) => {
                let x = lower(a, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                b.unary(UnaryOp::Round, x, dtype)
            }
            Expr::Powf(a, e) => {
                let x = lower(a, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
                let exp = b.push(I::Const(f32_cst(f64::from_bits(*e) as f32)?));
                b.binary(BinaryOp::Pow, x, exp, dtype)
            }
            // Abramowitz & Stegun 7.1.26 (max error ~1.5e-7) with
            // sign(x) = x / max(|x|, 1e-30); the CPU interpreter uses the
            // exact libm erf instead
            Expr::Erf(a) => {
                let x = lower(a, b, lanes, num_inputs, lowered_lanes, offset, zero, dtype)?;
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

    for (j, expr) in exprs.iter().enumerate() {
        let value = lower(expr, &mut b, &lanes, num_inputs, &mut lowered_lanes, clamped, zero, dtype)?;
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

#[cfg(target_os = "macos")]
mod metal {
    use super::{build_kernel, Expr, BLOCK};
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
        scalars: &[Tensor],
        n: usize,
        shape: &[usize],
        device: &Device,
    ) -> candle_core::Result<Vec<Tensor>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mdev = device.as_metal_device()?;
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
        let key = hasher.finish();
        let pipeline = {
            let mut cache = pipelines().lock().unwrap();
            match cache.get(&key) {
                Some(p) => p.clone(),
                None => {
                    let kernel =
                        build_kernel(exprs, inputs.len(), scalars.len(), n, ug::DType::F32)?;
                    let p = mdev.compile("effect_torch_fused", kernel)?;
                    if std::env::var_os("EFFECT_TORCH_FUSION_DEBUG").is_some() {
                        eprintln!("[fusion] compiled kernel #{} (key {key:x})", cache.len() + 1);
                    }
                    cache.insert(key, p.clone());
                    p
                }
            }
        };
        let padded = n.div_ceil(BLOCK) * BLOCK;
        let mut out_bufs = Vec::with_capacity(exprs.len());
        for _ in exprs {
            out_bufs.push(mdev.new_buffer(padded.max(1), DType::F32, "fused")?);
        }
        let encoder = mdev.command_encoder()?;
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
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => dtype == DType::F32,
        #[cfg(not(feature = "cuda"))]
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
// scalar lane [lr], mirroring the composed update including the first-step
// v = g initialization.
pub fn sgd_exprs(momentum: f64, dampening: f64, nesterov: bool, weight_decay: f64, first_step: bool) -> [Expr; 2] {
    let (p, g, v) = (Expr::Input(0), Expr::Input(1), Expr::Input(2));
    let lr = Expr::Scalar(0);
    let gp = if weight_decay == 0.0 {
        g
    } else {
        Expr::Add(Box::new(g), Box::new(Expr::Mul(Box::new(p.clone()), Box::new(Expr::cst(weight_decay)))))
    };
    let next_v = if first_step {
        gp.clone()
    } else {
        Expr::Add(
            Box::new(Expr::Mul(Box::new(v), Box::new(Expr::cst(momentum)))),
            Box::new(Expr::Mul(Box::new(gp.clone()), Box::new(Expr::cst(1.0 - dampening)))),
        )
    };
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

/// Evaluates each expression over the flattened inputs (all the same
/// element count) and returns one tensor per expression. `scalars` are
/// one-element tensors read at offset 0.
pub fn run(
    exprs: &[Expr],
    inputs: &[Tensor],
    scalars: &[Tensor],
    n: usize,
    shape: &[usize],
    dtype: DType,
    device: &Device,
) -> candle_core::Result<Vec<Tensor>> {
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
    match (device, dtype) {
        (Device::Cpu, DType::F32) => interpret_cpu::<f32>(exprs, inputs, scalars, n, shape),
        (Device::Cpu, DType::F64) => interpret_cpu::<f64>(exprs, inputs, scalars, n, shape),
        #[cfg(target_os = "macos")]
        (Device::Metal(_), DType::F32) => metal::run(exprs, inputs, scalars, n, shape, device),
        _ => Err(candle_core::Error::Msg(format!(
            "fusion: unsupported device/dtype {device:?} {dtype:?}"
        ))),
    }
}

/// A one-element tensor holding a scalar lane value.
pub fn scalar_tensor(v: f64, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
    match dtype {
        DType::F32 => Tensor::full(v as f32, (), device),
        DType::F64 => Tensor::full(v, (), device),
        dtype => Err(candle_core::Error::UnsupportedDTypeForOp(dtype, "fusion").bt()),
    }
}

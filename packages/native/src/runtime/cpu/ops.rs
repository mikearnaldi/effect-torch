use super::tensor::{CpuBuffer, Elem, Tensor};
use crate::runtime::layout::{broadcast_shape, Layout};
use half::{bf16, f16};

fn strided_offsets(la: &Layout, lb: &Layout, out_i: usize) -> (usize, usize) {
    let shape = la.shape();
    let rank = shape.len();
    let mut ia = la.offset();
    let mut ib = lb.offset();
    let mut rem = out_i;
    for d in (0..rank).rev() {
        let c = shape[d].max(1);
        let i = rem % c;
        rem /= c;
        ia += i * la.strides()[d];
        ib += i * lb.strides()[d];
    }
    (ia, ib)
}

fn binary_impl<T: Elem>(a: &[T], la: &Layout, b: &[T], lb: &Layout, f: impl Fn(T, T) -> T) -> Tensor {
    let shape = broadcast_shape(la.shape(), lb.shape());
    let la = la.broadcast_to(&shape);
    let lb = lb.broadcast_to(&shape);
    let n: usize = shape.iter().product();
    let mut out = vec![T::default(); n];
    if la.is_contiguous() && lb.is_contiguous() {
        let (oa, ob) = (la.offset(), lb.offset());
        for i in 0..n {
            out[i] = f(a[oa + i], b[ob + i]);
        }
    } else {
        for out_i in 0..n {
            let (ia, ib) = strided_offsets(&la, &lb, out_i);
            out[out_i] = f(a[ia], b[ib]);
        }
    }
    Tensor::from_vec(out, shape)
}

fn binary_u8_impl<T: Elem>(a: &[T], la: &Layout, b: &[T], lb: &Layout, f: impl Fn(T, T) -> u8) -> Tensor {
    let shape = broadcast_shape(la.shape(), lb.shape());
    let la = la.broadcast_to(&shape);
    let lb = lb.broadcast_to(&shape);
    let n: usize = shape.iter().product();
    let mut out = vec![0u8; n];
    for out_i in 0..n {
        let (ia, ib) = strided_offsets(&la, &lb, out_i);
        out[out_i] = f(a[ia], b[ib]);
    }
    Tensor::from_vec(out, shape)
}

fn unary_impl<T: Elem>(a: &[T], la: &Layout, f: impl Fn(T) -> T) -> Tensor {
    let n = la.numel();
    let mut out = vec![T::default(); n];
    if la.is_contiguous() {
        let o = la.offset();
        for i in 0..n {
            out[i] = f(a[o + i]);
        }
    } else {
        for out_i in 0..n {
            let (ia, _) = strided_offsets(la, la, out_i);
            out[out_i] = f(a[ia]);
        }
    }
    Tensor::from_vec(out, la.shape().to_vec())
}

macro_rules! dispatch_binary {
    ($self:expr, $rhs:expr, $impl:ident, $f:expr) => {{
        assert_eq!($self.dtype(), $rhs.dtype(), "mixed dtypes");
        match (&$self.buffer, &$rhs.buffer) {
            (CpuBuffer::F32(a), CpuBuffer::F32(b)) => $impl(a, &$self.layout, b, &$rhs.layout, $f),
            (CpuBuffer::F64(a), CpuBuffer::F64(b)) => $impl(a, &$self.layout, b, &$rhs.layout, $f),
            (CpuBuffer::F16(a), CpuBuffer::F16(b)) => $impl(a, &$self.layout, b, &$rhs.layout, $f),
            (CpuBuffer::BF16(a), CpuBuffer::BF16(b)) => $impl(a, &$self.layout, b, &$rhs.layout, $f),
            (CpuBuffer::U8(a), CpuBuffer::U8(b)) => $impl(a, &$self.layout, b, &$rhs.layout, $f),
            (CpuBuffer::U32(a), CpuBuffer::U32(b)) => $impl(a, &$self.layout, b, &$rhs.layout, $f),
            (CpuBuffer::I64(a), CpuBuffer::I64(b)) => $impl(a, &$self.layout, b, &$rhs.layout, $f),
            _ => unreachable!("dtype checked above"),
        }
    }};
}

macro_rules! dispatch_float_unary {
    ($self:expr, $f32f:expr, $f64f:expr, $f16f:expr, $bf16f:expr) => {{
        assert!($self.dtype().is_float(), "requires a float dtype");
        match &$self.buffer {
            CpuBuffer::F32(v) => unary_impl(v, &$self.layout, $f32f),
            CpuBuffer::F64(v) => unary_impl(v, &$self.layout, $f64f),
            CpuBuffer::F16(v) => unary_impl(v, &$self.layout, $f16f),
            CpuBuffer::BF16(v) => unary_impl(v, &$self.layout, $bf16f),
            _ => unreachable!("float checked above"),
        }
    }};
}

macro_rules! float_ops {
    ($name:ident, $f32f:expr, $f64f:expr) => {
        pub fn $name(&self) -> Tensor {
            dispatch_float_unary!(
                self,
                $f32f,
                $f64f,
                |x: f16| f16::from_f32(($f32f)(x.to_f32())),
                |x: bf16| bf16::from_f32(($f32f)(x.to_f32()))
            )
        }
    };
}

impl Tensor {
    pub fn add(&self, rhs: &Tensor) -> Tensor {
        dispatch_binary!(self, rhs, binary_impl, |a, b| a + b)
    }
    pub fn sub(&self, rhs: &Tensor) -> Tensor {
        dispatch_binary!(self, rhs, binary_impl, |a, b| a - b)
    }
    pub fn mul(&self, rhs: &Tensor) -> Tensor {
        dispatch_binary!(self, rhs, binary_impl, |a, b| a * b)
    }
    pub fn div(&self, rhs: &Tensor) -> Tensor {
        dispatch_binary!(self, rhs, binary_impl, |a, b| a / b)
    }
    pub fn maximum(&self, rhs: &Tensor) -> Tensor {
        dispatch_binary!(self, rhs, binary_impl, |a, b| if a >= b { a } else { b })
    }
    pub fn minimum(&self, rhs: &Tensor) -> Tensor {
        dispatch_binary!(self, rhs, binary_impl, |a, b| if a <= b { a } else { b })
    }
    pub fn pow(&self, rhs: &Tensor) -> Tensor {
        assert!(self.dtype().is_float(), "pow requires float dtypes");
        match (&self.buffer, &rhs.buffer) {
            (CpuBuffer::F32(a), CpuBuffer::F32(b)) => binary_impl(a, &self.layout, b, &rhs.layout, |a: f32, b: f32| a.powf(b)),
            (CpuBuffer::F64(a), CpuBuffer::F64(b)) => binary_impl(a, &self.layout, b, &rhs.layout, |a: f64, b: f64| a.powf(b)),
            (CpuBuffer::F16(a), CpuBuffer::F16(b)) => binary_impl(a, &self.layout, b, &rhs.layout, |a: f16, b: f16| f16::from_f32(a.to_f32().powf(b.to_f32()))),
            (CpuBuffer::BF16(a), CpuBuffer::BF16(b)) => binary_impl(a, &self.layout, b, &rhs.layout, |a: bf16, b: bf16| bf16::from_f32(a.to_f32().powf(b.to_f32()))),
            _ => unreachable!("float checked above"),
        }
    }

    pub fn eq(&self, rhs: &Tensor) -> Tensor {
        dispatch_binary!(self, rhs, binary_u8_impl, |a, b| (a == b) as u8)
    }
    pub fn gt(&self, rhs: &Tensor) -> Tensor {
        dispatch_binary!(self, rhs, binary_u8_impl, |a, b| (a > b) as u8)
    }
    pub fn lt(&self, rhs: &Tensor) -> Tensor {
        dispatch_binary!(self, rhs, binary_u8_impl, |a, b| (a < b) as u8)
    }
    pub fn ge(&self, rhs: &Tensor) -> Tensor {
        dispatch_binary!(self, rhs, binary_u8_impl, |a, b| (a >= b) as u8)
    }
    pub fn le(&self, rhs: &Tensor) -> Tensor {
        dispatch_binary!(self, rhs, binary_u8_impl, |a, b| (a <= b) as u8)
    }

    pub fn where_(&self, cond: &Tensor, other: &Tensor) -> Tensor {
        assert_eq!(cond.dtype(), crate::runtime::dtype::DType::U8, "where condition must be u8");
        assert_eq!(self.dtype(), other.dtype(), "mixed dtypes");
        let CpuBuffer::U8(c) = &cond.buffer else { unreachable!() };
        let shape = broadcast_shape(&broadcast_shape(cond.shape(), self.shape()), other.shape());
        let lc = cond.layout.broadcast_to(&shape);
        let la = self.layout.broadcast_to(&shape);
        let lb = other.layout.broadcast_to(&shape);
        let n: usize = shape.iter().product();
        macro_rules! go {
            ($a:expr, $b:expr, $t:ty) => {{
                let mut out = vec![<$t>::default(); n];
                for i in 0..n {
                    let (ic, rest) = strided_offsets(&lc, &la, i);
                    let (_, ib) = strided_offsets(&lc, &lb, i);
                    out[i] = if c[ic] != 0 { $a[rest] } else { $b[ib] };
                }
                Tensor::from_vec(out, shape.clone())
            }};
        }
        match (&self.buffer, &other.buffer) {
            (CpuBuffer::F32(a), CpuBuffer::F32(b)) => go!(a, b, f32),
            (CpuBuffer::F64(a), CpuBuffer::F64(b)) => go!(a, b, f64),
            (CpuBuffer::F16(a), CpuBuffer::F16(b)) => go!(a, b, f16),
            (CpuBuffer::BF16(a), CpuBuffer::BF16(b)) => go!(a, b, bf16),
            (CpuBuffer::U8(a), CpuBuffer::U8(b)) => go!(a, b, u8),
            (CpuBuffer::U32(a), CpuBuffer::U32(b)) => go!(a, b, u32),
            (CpuBuffer::I64(a), CpuBuffer::I64(b)) => go!(a, b, i64),
            _ => unreachable!("dtype checked above"),
        }
    }

    pub fn neg(&self) -> Tensor {
        assert!(self.dtype() != crate::runtime::dtype::DType::U8, "neg on u8");
        match &self.buffer {
            CpuBuffer::F32(v) => unary_impl(v, &self.layout, |x: f32| -x),
            CpuBuffer::F64(v) => unary_impl(v, &self.layout, |x: f64| -x),
            CpuBuffer::F16(v) => unary_impl(v, &self.layout, |x: f16| -x),
            CpuBuffer::BF16(v) => unary_impl(v, &self.layout, |x: bf16| -x),
            CpuBuffer::U32(v) => unary_impl(v, &self.layout, |x: u32| x.wrapping_neg()),
            CpuBuffer::I64(v) => unary_impl(v, &self.layout, |x: i64| -x),
            _ => unreachable!(),
        }
    }

    pub fn powf(&self, e: f64) -> Tensor {
        dispatch_float_unary!(
            self,
            |x: f32| x.powf(e as f32),
            |x: f64| x.powf(e),
            |x: f16| f16::from_f32(x.to_f32().powf(e as f32)),
            |x: bf16| bf16::from_f32(x.to_f32().powf(e as f32))
        )
    }

    pub fn squeeze_dims(&self, dims: &[usize]) -> Tensor {
        let shape: Vec<usize> = self
            .shape()
            .iter()
            .enumerate()
            .filter(|(d, &s)| !dims.contains(d) || s != 1)
            .map(|(_, &s)| s)
            .collect();
        self.contiguous().view(Layout::contiguous(shape))
    }

    float_ops!(sqrt, |x: f32| x.sqrt(), |x: f64| x.sqrt());
    float_ops!(exp, |x: f32| x.exp(), |x: f64| x.exp());
    float_ops!(log, |x: f32| x.ln(), |x: f64| x.ln());
    float_ops!(sin, |x: f32| x.sin(), |x: f64| x.sin());
    float_ops!(cos, |x: f32| x.cos(), |x: f64| x.cos());
    float_ops!(tanh, |x: f32| x.tanh(), |x: f64| x.tanh());
    float_ops!(erf, |x: f32| libm::erff(x), |x: f64| libm::erf(x));
    float_ops!(floor, |x: f32| x.floor(), |x: f64| x.floor());
    float_ops!(ceil, |x: f32| x.ceil(), |x: f64| x.ceil());
    float_ops!(round, |x: f32| x.round(), |x: f64| x.round());
    float_ops!(abs, |x: f32| x.abs(), |x: f64| x.abs());
    float_ops!(sign, |x: f32| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }, |x: f64| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 });
    pub fn relu(&self) -> Tensor {
        match &self.buffer {
            CpuBuffer::F32(v) => unary_impl(v, &self.layout, |x: f32| x.max(0.0)),
            CpuBuffer::F64(v) => unary_impl(v, &self.layout, |x: f64| x.max(0.0)),
            CpuBuffer::F16(v) => unary_impl(v, &self.layout, |x: f16| f16::from_f32(x.to_f32().max(0.0))),
            CpuBuffer::BF16(v) => unary_impl(v, &self.layout, |x: bf16| bf16::from_f32(x.to_f32().max(0.0))),
            CpuBuffer::U8(v) => unary_impl(v, &self.layout, |x: u8| x),
            CpuBuffer::U32(v) => unary_impl(v, &self.layout, |x: u32| x),
            CpuBuffer::I64(v) => unary_impl(v, &self.layout, |x: i64| x.max(0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_data(t: &Tensor) -> Vec<f32> {
        let CpuBuffer::F32(v) = &t.buffer else { panic!() };
        v.as_slice().to_vec()
    }

    #[test]
    fn binary_broadcast() {
        let a = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], vec![2, 3]);
        let b = Tensor::from_vec(vec![10f32, 20., 30.], vec![3]);
        let c = a.add(&b);
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(f32_data(&c), vec![11., 22., 33., 14., 25., 36.]);
    }

    #[test]
    fn binary_strided() {
        let a = Tensor::from_vec((0..6).map(|x| x as f32).collect(), vec![2, 3]);
        let at = a.view(a.layout.permute(&[1, 0]));
        let c = at.add(&at);
        assert_eq!(c.shape(), &[3, 2]);
        assert_eq!(f32_data(&c), vec![0., 6., 2., 8., 4., 10.]);
    }

    #[test]
    fn comparisons_and_where() {
        let a = Tensor::from_vec(vec![1f32, 5., 3.], vec![3]);
        let b = Tensor::from_vec(vec![2f32, 4., 3.], vec![3]);
        let m = a.gt(&b);
        let CpuBuffer::U8(v) = &m.buffer else { panic!() };
        assert_eq!(v.as_slice(), &[0, 1, 0]);
        let w = a.where_(&m, &b);
        assert_eq!(f32_data(&w), vec![2., 5., 3.]);
    }

    #[test]
    fn unary_float() {
        let a = Tensor::from_vec(vec![0f32, 1.0], vec![2]);
        let e = a.exp();
        assert!((f32_data(&e)[1] - std::f32::consts::E).abs() < 1e-6);
        let h = Tensor::from_vec(vec![f16::ONE], vec![1]).exp();
        assert_eq!(h.dtype(), crate::runtime::dtype::DType::F16);
    }
}

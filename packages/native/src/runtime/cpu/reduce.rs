use super::tensor::{CpuBuffer, Elem, Tensor};
use crate::runtime::layout::Layout;
use half::{bf16, f16};

fn kept_shape(shape: &[usize], dims: &[usize]) -> Vec<usize> {
    let mut out = shape.to_vec();
    for &d in dims {
        out[d] = 1;
    }
    out
}

fn reduce_impl<T: Elem>(
    a: &[T],
    la: &Layout,
    dims: &[usize],
    init: T,
    f: impl Fn(T, T) -> T,
) -> Tensor {
    let shape = la.shape();
    let out_shape = kept_shape(shape, dims);
    let out_n: usize = out_shape.iter().product();
    let red_n: usize = dims.iter().map(|&d| shape[d]).product();
    let mut out = vec![init; out_n];

    let rank = shape.len();
    let mut red_strides = Vec::with_capacity(dims.len());
    for &d in dims {
        red_strides.push(la.strides()[d]);
    }
    let kept: Vec<usize> = (0..rank).filter(|d| !dims.contains(d)).collect();
    let kept_strides: Vec<usize> = kept.iter().map(|&d| la.strides()[d]).collect();
    let kept_dims: Vec<usize> = kept.iter().map(|&d| shape[d]).collect();

    for out_i in 0..out_n {
        let mut base = la.offset();
        let mut rem = out_i;
        for k in (0..kept.len()).rev() {
            let c = kept_dims[k].max(1);
            let i = rem % c;
            rem /= c;
            base += i * kept_strides[k];
        }
        let mut acc = init;
        for red_i in 0..red_n {
            let mut src = base;
            let mut r = red_i;
            for k in (0..dims.len()).rev() {
                let c = shape[dims[k]].max(1);
                let i = r % c;
                r /= c;
                src += i * red_strides[k];
            }
            acc = f(acc, a[src]);
        }
        out[out_i] = acc;
    }
    Tensor::from_vec(out, out_shape)
}

fn argreduce_impl<T: Elem + PartialOrd>(
    a: &[T],
    la: &Layout,
    dim: usize,
    pick_max: bool,
) -> Tensor {
    let shape = la.shape();
    let out_shape = kept_shape(shape, &[dim]);
    let out_n: usize = out_shape.iter().product();
    let n = shape[dim];
    let mut out = vec![0u32; out_n];
    let rank = shape.len();
    let kept: Vec<usize> = (0..rank).filter(|&d| d != dim).collect();
    let kept_strides: Vec<usize> = kept.iter().map(|&d| la.strides()[d]).collect();
    let kept_dims: Vec<usize> = kept.iter().map(|&d| shape[d]).collect();
    let dstride = la.strides()[dim];

    for out_i in 0..out_n {
        let mut base = la.offset();
        let mut rem = out_i;
        for k in (0..kept.len()).rev() {
            let c = kept_dims[k].max(1);
            let i = rem % c;
            rem /= c;
            base += i * kept_strides[k];
        }
        let mut best = 0usize;
        let mut best_v = a[base];
        for i in 1..n {
            let v = a[base + i * dstride];
            let better = if pick_max { v > best_v } else { v < best_v };
            if better {
                best = i;
                best_v = v;
            }
        }
        out[out_i] = best as u32;
    }
    Tensor::from_vec(out, out_shape)
}

fn cumsum_impl<T: Elem + std::ops::Add<Output = T>>(a: &[T], la: &Layout, dim: usize) -> Tensor {
    let shape = la.shape();
    let n = shape[dim];
    let total = la.numel();
    let mut out = vec![T::default(); total];
    let dstride = la.strides()[dim];
    let rank = shape.len();
    let kept: Vec<usize> = (0..rank).filter(|&d| d != dim).collect();
    let kept_strides: Vec<usize> = kept.iter().map(|&d| la.strides()[d]).collect();
    let kept_dims: Vec<usize> = kept.iter().map(|&d| shape[d]).collect();
    let kept_n: usize = kept_dims.iter().product();

    let out_strides = Layout::contiguous(shape.to_vec());
    let os = out_strides.strides().to_vec();

    for kept_i in 0..kept_n {
        let mut base = la.offset();
        let mut obase = 0usize;
        let mut rem = kept_i;
        for k in (0..kept.len()).rev() {
            let c = kept_dims[k].max(1);
            let i = rem % c;
            rem /= c;
            base += i * kept_strides[k];
            obase += i * os[kept[k]];
        }
        let mut acc = T::default();
        for i in 0..n {
            acc = acc + a[base + i * dstride];
            out[obase + i * os[dim]] = acc;
        }
    }
    Tensor::from_vec(out, shape.to_vec())
}

macro_rules! dispatch_reduce {
    ($self:expr, $dims:expr, $init:expr, $f:expr) => {
        match &$self.buffer {
            CpuBuffer::F32(v) => reduce_impl(v, &$self.layout, $dims, $init, $f),
            CpuBuffer::F64(v) => reduce_impl(v, &$self.layout, $dims, $init, $f),
            CpuBuffer::F16(v) => reduce_impl(v, &$self.layout, $dims, $init, $f),
            CpuBuffer::BF16(v) => reduce_impl(v, &$self.layout, $dims, $init, $f),
            CpuBuffer::U8(v) => reduce_impl(v, &$self.layout, $dims, $init, $f),
            CpuBuffer::U32(v) => reduce_impl(v, &$self.layout, $dims, $init, $f),
            CpuBuffer::I64(v) => reduce_impl(v, &$self.layout, $dims, $init, $f),
        }
    };
}

macro_rules! dispatch_argreduce {
    ($self:expr, $dim:expr, $pick_max:expr) => {
        match &$self.buffer {
            CpuBuffer::F32(v) => argreduce_impl(v, &$self.layout, $dim, $pick_max),
            CpuBuffer::F64(v) => argreduce_impl(v, &$self.layout, $dim, $pick_max),
            CpuBuffer::F16(v) => argreduce_impl(v, &$self.layout, $dim, $pick_max),
            CpuBuffer::BF16(v) => argreduce_impl(v, &$self.layout, $dim, $pick_max),
            CpuBuffer::U8(v) => argreduce_impl(v, &$self.layout, $dim, $pick_max),
            CpuBuffer::U32(v) => argreduce_impl(v, &$self.layout, $dim, $pick_max),
            CpuBuffer::I64(v) => argreduce_impl(v, &$self.layout, $dim, $pick_max),
        }
    };
}

impl Tensor {
    pub fn sum(&self, dims: &[usize]) -> Tensor {
        dispatch_reduce!(self, dims, Default::default(), |a, b| a + b)
    }

    pub fn prod(&self, dims: &[usize]) -> Tensor {
        macro_rules! one {
            (f32) => { 1f32 };
            (f64) => { 1f64 };
            (f16) => { f16::ONE };
            (bf16) => { bf16::ONE };
            (u8) => { 1u8 };
            (u32) => { 1u32 };
            (i64) => { 1i64 };
        }
        match &self.buffer {
            CpuBuffer::F32(v) => reduce_impl(v, &self.layout, dims, one!(f32), |a, b| a * b),
            CpuBuffer::F64(v) => reduce_impl(v, &self.layout, dims, one!(f64), |a, b| a * b),
            CpuBuffer::F16(v) => reduce_impl(v, &self.layout, dims, one!(f16), |a, b| a * b),
            CpuBuffer::BF16(v) => reduce_impl(v, &self.layout, dims, one!(bf16), |a, b| a * b),
            CpuBuffer::U8(v) => reduce_impl(v, &self.layout, dims, one!(u8), |a, b| a * b),
            CpuBuffer::U32(v) => reduce_impl(v, &self.layout, dims, one!(u32), |a, b| a * b),
            CpuBuffer::I64(v) => reduce_impl(v, &self.layout, dims, one!(i64), |a, b| a * b),
        }
    }

    pub fn max(&self, dims: &[usize]) -> Tensor {
        match &self.buffer {
            CpuBuffer::F32(v) => reduce_impl(v, &self.layout, dims, f32::NEG_INFINITY, |a: f32, b: f32| if a >= b { a } else { b }),
            CpuBuffer::F64(v) => reduce_impl(v, &self.layout, dims, f64::NEG_INFINITY, |a: f64, b: f64| if a >= b { a } else { b }),
            CpuBuffer::F16(v) => reduce_impl(v, &self.layout, dims, f16::NEG_INFINITY, |a: f16, b: f16| if a >= b { a } else { b }),
            CpuBuffer::BF16(v) => reduce_impl(v, &self.layout, dims, bf16::NEG_INFINITY, |a: bf16, b: bf16| if a >= b { a } else { b }),
            CpuBuffer::U8(v) => reduce_impl(v, &self.layout, dims, u8::MIN, |a: u8, b: u8| a.max(b)),
            CpuBuffer::U32(v) => reduce_impl(v, &self.layout, dims, u32::MIN, |a: u32, b: u32| a.max(b)),
            CpuBuffer::I64(v) => reduce_impl(v, &self.layout, dims, i64::MIN, |a: i64, b: i64| a.max(b)),
        }
    }

    pub fn min(&self, dims: &[usize]) -> Tensor {
        match &self.buffer {
            CpuBuffer::F32(v) => reduce_impl(v, &self.layout, dims, f32::INFINITY, |a: f32, b: f32| if a <= b { a } else { b }),
            CpuBuffer::F64(v) => reduce_impl(v, &self.layout, dims, f64::INFINITY, |a: f64, b: f64| if a <= b { a } else { b }),
            CpuBuffer::F16(v) => reduce_impl(v, &self.layout, dims, f16::INFINITY, |a: f16, b: f16| if a <= b { a } else { b }),
            CpuBuffer::BF16(v) => reduce_impl(v, &self.layout, dims, bf16::INFINITY, |a: bf16, b: bf16| if a <= b { a } else { b }),
            CpuBuffer::U8(v) => reduce_impl(v, &self.layout, dims, u8::MAX, |a: u8, b: u8| a.min(b)),
            CpuBuffer::U32(v) => reduce_impl(v, &self.layout, dims, u32::MAX, |a: u32, b: u32| a.min(b)),
            CpuBuffer::I64(v) => reduce_impl(v, &self.layout, dims, i64::MAX, |a: i64, b: i64| a.min(b)),
        }
    }

    pub fn mean(&self, dims: &[usize]) -> Tensor {
        assert!(self.dtype().is_float(), "mean requires a float dtype");
        let count: usize = dims.iter().map(|&d| self.shape()[d]).product();
        let s = self.sum(dims);
        let c = Tensor::full(s.shape(), count as f64, s.dtype());
        s.div(&c)
    }

    pub fn argmax(&self, dim: usize) -> Tensor {
        dispatch_argreduce!(self, dim, true)
    }

    pub fn argmin(&self, dim: usize) -> Tensor {
        dispatch_argreduce!(self, dim, false)
    }

    pub fn cumsum(&self, dim: usize) -> Tensor {
        match &self.buffer {
            CpuBuffer::F32(v) => cumsum_impl(v, &self.layout, dim),
            CpuBuffer::F64(v) => cumsum_impl(v, &self.layout, dim),
            CpuBuffer::F16(v) => cumsum_impl(v, &self.layout, dim),
            CpuBuffer::BF16(v) => cumsum_impl(v, &self.layout, dim),
            CpuBuffer::U8(v) => cumsum_impl(v, &self.layout, dim),
            CpuBuffer::U32(v) => cumsum_impl(v, &self.layout, dim),
            CpuBuffer::I64(v) => cumsum_impl(v, &self.layout, dim),
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
    fn sum_dims() {
        let t = Tensor::from_vec((0..6).map(|x| x as f32).collect(), vec![2, 3]);
        let s = t.sum(&[1]);
        assert_eq!(s.shape(), &[2, 1]);
        assert_eq!(f32_data(&s), vec![3., 12.]);
        let s0 = t.sum(&[0]);
        assert_eq!(f32_data(&s0), vec![3., 5., 7.]);
        let m = t.mean(&[0, 1]);
        assert_eq!(f32_data(&m), vec![2.5]);
    }

    #[test]
    fn sum_strided() {
        let t = Tensor::from_vec((0..6).map(|x| x as f32).collect(), vec![2, 3]);
        let tt = t.view(t.layout.permute(&[1, 0]));
        let s = tt.sum(&[0]);
        assert_eq!(f32_data(&s), vec![3., 12.]);
    }

    #[test]
    fn argmax_cumsum() {
        let t = Tensor::from_vec(vec![1f32, 5., 2., 0., 4., 4.], vec![2, 3]);
        let am = t.argmax(1);
        let CpuBuffer::U32(v) = &am.buffer else { panic!() };
        assert_eq!(v.as_slice(), &[1, 1]);
        let c = t.cumsum(1);
        assert_eq!(f32_data(&c), vec![1., 6., 8., 0., 4., 8.]);
    }
}

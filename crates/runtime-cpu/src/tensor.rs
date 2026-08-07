use effect_torch_runtime::{DType, Layout};
use half::{bf16, f16};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum CpuBuffer {
    F32(Arc<Vec<f32>>),
    F64(Arc<Vec<f64>>),
    F16(Arc<Vec<f16>>),
    BF16(Arc<Vec<bf16>>),
    U8(Arc<Vec<u8>>),
    U32(Arc<Vec<u32>>),
    I64(Arc<Vec<i64>>),
}

impl CpuBuffer {
    pub fn dtype(&self) -> DType {
        match self {
            CpuBuffer::F32(_) => DType::F32,
            CpuBuffer::F64(_) => DType::F64,
            CpuBuffer::F16(_) => DType::F16,
            CpuBuffer::BF16(_) => DType::BF16,
            CpuBuffer::U8(_) => DType::U8,
            CpuBuffer::U32(_) => DType::U32,
            CpuBuffer::I64(_) => DType::I64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            CpuBuffer::F32(v) => v.len(),
            CpuBuffer::F64(v) => v.len(),
            CpuBuffer::F16(v) => v.len(),
            CpuBuffer::BF16(v) => v.len(),
            CpuBuffer::U8(v) => v.len(),
            CpuBuffer::U32(v) => v.len(),
            CpuBuffer::I64(v) => v.len(),
        }
    }
}

pub trait Elem: Copy + Default + 'static {
    fn buffer_of(v: Vec<Self>) -> CpuBuffer;
    fn dtype() -> DType;
    fn to_f64(self) -> f64;
    fn from_f64(x: f64) -> Self;
    fn slice_of(t: &Tensor) -> Option<&[Self]>;
}

macro_rules! impl_elem {
    ($t:ty, $variant:ident, $dtype:expr, $to:expr, $from:expr) => {
        impl Elem for $t {
            fn buffer_of(v: Vec<Self>) -> CpuBuffer {
                CpuBuffer::$variant(Arc::new(v))
            }
            fn dtype() -> DType {
                $dtype
            }
            fn to_f64(self) -> f64 {
                #[allow(clippy::redundant_closure_call)]
                let f: fn($t) -> f64 = $to;
                f(self)
            }
            fn from_f64(x: f64) -> Self {
                #[allow(clippy::redundant_closure_call)]
                let f: fn(f64) -> $t = $from;
                f(x)
            }
            fn slice_of(t: &Tensor) -> Option<&[Self]> {
                let CpuBuffer::$variant(v) = &t.buffer else {
                    return None;
                };
                if !t.layout.is_contiguous() {
                    return None;
                }
                let start = t.layout.offset();
                Some(&v[start..start + t.numel()])
            }
        }
    };
}

impl_elem!(f32, F32, DType::F32, |x| x as f64, |x| x as f32);
impl_elem!(f64, F64, DType::F64, |x| x, |x| x);
impl_elem!(f16, F16, DType::F16, |x: f16| x.to_f64(), f16::from_f64);
impl_elem!(
    bf16,
    BF16,
    DType::BF16,
    |x: bf16| x.to_f64(),
    bf16::from_f64
);
impl_elem!(u8, U8, DType::U8, |x| x as f64, |x| x as u8);
impl_elem!(u32, U32, DType::U32, |x| x as f64, |x| x as u32);
impl_elem!(i64, I64, DType::I64, |x| x as f64, |x| x as i64);

#[derive(Debug, Clone)]
pub struct Tensor {
    pub buffer: CpuBuffer,
    pub layout: Layout,
}

impl Tensor {
    pub fn new(buffer: CpuBuffer, layout: Layout) -> Self {
        assert!(layout.max_index() <= buffer.len(), "layout exceeds buffer");
        Tensor { buffer, layout }
    }

    pub fn dtype(&self) -> DType {
        self.buffer.dtype()
    }

    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    pub fn numel(&self) -> usize {
        self.layout.numel()
    }

    pub fn from_vec<T: Elem>(data: Vec<T>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Tensor::new(T::buffer_of(data), Layout::contiguous(shape))
    }

    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        Self::full(shape, 0.0, dtype)
    }

    pub fn ones(shape: &[usize], dtype: DType) -> Self {
        Self::full(shape, 1.0, dtype)
    }

    pub fn full(shape: &[usize], value: f64, dtype: DType) -> Self {
        let n = shape.iter().product();
        let buffer = match dtype {
            DType::F32 => CpuBuffer::F32(Arc::new(vec![value as f32; n])),
            DType::F64 => CpuBuffer::F64(Arc::new(vec![value; n])),
            DType::F16 => CpuBuffer::F16(Arc::new(vec![f16::from_f64(value); n])),
            DType::BF16 => CpuBuffer::BF16(Arc::new(vec![bf16::from_f64(value); n])),
            DType::U8 => CpuBuffer::U8(Arc::new(vec![value as u8; n])),
            DType::U32 => CpuBuffer::U32(Arc::new(vec![value as u32; n])),
            DType::I64 => CpuBuffer::I64(Arc::new(vec![value as i64; n])),
        };
        Tensor::new(buffer, Layout::contiguous(shape.to_vec()))
    }

    pub fn view(&self, layout: Layout) -> Self {
        assert!(layout.max_index() <= self.buffer.len());
        Tensor {
            buffer: self.buffer.clone(),
            layout,
        }
    }

    pub fn contiguous(&self) -> Self {
        if self.layout.is_contiguous()
            && self.layout.offset() == 0
            && self.buffer.len() == self.layout.numel()
        {
            return self.clone();
        }
        match &self.buffer {
            CpuBuffer::F32(v) => contiguous_impl(v, &self.layout),
            CpuBuffer::F64(v) => contiguous_impl(v, &self.layout),
            CpuBuffer::F16(v) => contiguous_impl(v, &self.layout),
            CpuBuffer::BF16(v) => contiguous_impl(v, &self.layout),
            CpuBuffer::U8(v) => contiguous_impl(v, &self.layout),
            CpuBuffer::U32(v) => contiguous_impl(v, &self.layout),
            CpuBuffer::I64(v) => contiguous_impl(v, &self.layout),
        }
    }

    pub fn cast(&self, dtype: DType) -> Self {
        if self.dtype() == dtype {
            return self.clone();
        }
        let src = self.contiguous();
        macro_rules! go {
            ($v:expr, $s:ty) => {
                cast_impl::<$s>($v, src.shape(), dtype)
            };
        }
        match &src.buffer {
            CpuBuffer::F32(v) => go!(v, f32),
            CpuBuffer::F64(v) => go!(v, f64),
            CpuBuffer::F16(v) => go!(v, f16),
            CpuBuffer::BF16(v) => go!(v, bf16),
            CpuBuffer::U8(v) => go!(v, u8),
            CpuBuffer::U32(v) => go!(v, u32),
            CpuBuffer::I64(v) => go!(v, i64),
        }
    }
}

fn contiguous_impl<T: Elem>(src: &[T], layout: &Layout) -> Tensor {
    let mut out = vec![T::default(); layout.numel()];
    copy_strided(src, layout, &mut out);
    Tensor::new(
        T::buffer_of(out),
        Layout::contiguous(layout.shape().to_vec()),
    )
}

fn cast_impl<T: Elem>(src: &[T], shape: &[usize], dtype: DType) -> Tensor {
    macro_rules! to {
        ($t:ty) => {
            Tensor::from_vec(
                src.iter()
                    .map(|&x| <$t>::from_f64(x.to_f64()))
                    .collect::<Vec<$t>>(),
                shape.to_vec(),
            )
        };
    }
    match dtype {
        DType::F32 => to!(f32),
        DType::F64 => to!(f64),
        DType::F16 => to!(f16),
        DType::BF16 => to!(bf16),
        DType::U8 => to!(u8),
        DType::U32 => to!(u32),
        DType::I64 => to!(i64),
    }
}

pub fn copy_strided<T: Copy>(src: &[T], layout: &Layout, dst: &mut [T]) {
    if layout.is_contiguous() {
        let start = layout.offset();
        dst.copy_from_slice(&src[start..start + layout.numel()]);
        return;
    }
    let shape = layout.shape();
    let strides = layout.strides();
    let rank = shape.len();
    let n = layout.numel();
    for out_i in 0..n {
        let mut src_i = layout.offset();
        let mut rem = out_i;
        for d in (0..rank).rev() {
            let c = shape[d].max(1);
            let i = rem % c;
            rem /= c;
            src_i += i * strides[d];
        }
        dst[out_i] = src[src_i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_gathers_strided() {
        let t = Tensor::from_vec((0..12).map(|x| x as f32).collect(), vec![3, 4]);
        let v = t.view(t.layout.permute(&[1, 0]));
        let c = v.contiguous();
        assert_eq!(c.shape(), &[4, 3]);
        let CpuBuffer::F32(data) = &c.buffer else {
            panic!()
        };
        assert_eq!(
            data.as_slice(),
            &[0., 4., 8., 1., 5., 9., 2., 6., 10., 3., 7., 11.]
        );
    }

    #[test]
    fn cast_roundtrip_f16() {
        let t = Tensor::from_vec(vec![1.5f32, -2.25, 100.0], vec![3]);
        let h = t.cast(DType::F16);
        assert_eq!(h.dtype(), DType::F16);
        let back = h.cast(DType::F32);
        let CpuBuffer::F32(data) = &back.buffer else {
            panic!()
        };
        assert_eq!(data.as_slice(), &[1.5, -2.25, 100.0]);
    }

    #[test]
    fn cast_int_float() {
        let t = Tensor::from_vec(vec![1u32, 2, 255], vec![3]);
        let f = t.cast(DType::F32);
        let CpuBuffer::F32(data) = &f.buffer else {
            panic!()
        };
        assert_eq!(data.as_slice(), &[1.0, 2.0, 255.0]);
        let b = f.cast(DType::U8);
        let CpuBuffer::U8(data) = &b.buffer else {
            panic!()
        };
        assert_eq!(data.as_slice(), &[1, 2, 255]);
    }
}

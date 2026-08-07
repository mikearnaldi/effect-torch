use super::tensor::{CpuBuffer, Tensor};
use effect_torch_runtime::{DType, Layout};

fn indices_vec(t: &Tensor) -> Vec<usize> {
    let c = t.contiguous();
    match &c.buffer {
        CpuBuffer::U32(v) => v.iter().map(|&x| x as usize).collect(),
        CpuBuffer::I64(v) => v.iter().map(|&x| x as usize).collect(),
        CpuBuffer::U8(v) => v.iter().map(|&x| x as usize).collect(),
        _ => panic!("indices must be u8/u32/i64"),
    }
}

fn gather_impl<T: super::tensor::Elem>(
    x: &[T],
    xl: &Layout,
    ids: &[usize],
    ids_shape: &[usize],
    dim: usize,
) -> Tensor {
    let shape = xl.shape();
    let rank = shape.len();
    assert!(dim < rank);
    assert_eq!(ids_shape.len(), rank);
    let total: usize = ids_shape.iter().product();
    let mut out = vec![T::default(); total];
    for (lin, &id) in ids.iter().enumerate() {
        assert!(id < shape[dim], "gather index out of bounds");
        let mut rem = lin;
        let mut base = xl.offset();
        for d in (0..rank).rev() {
            let c = ids_shape[d].max(1);
            let i = rem % c;
            rem /= c;
            let coord = if d == dim { id } else { i };
            base += coord * xl.strides()[d];
        }
        out[lin] = x[base];
    }
    Tensor::from_vec(out, ids_shape.to_vec())
}

fn scatter_add_impl<T: super::tensor::Elem + std::ops::Add<Output = T>>(
    x: &[T],
    xl: &Layout,
    ids: &[usize],
    ids_shape: &[usize],
    src: &[T],
    dim: usize,
) -> Tensor {
    let shape = xl.shape();
    let rank = shape.len();
    let total: usize = ids_shape.iter().product();
    let mut out = vec![T::default(); xl.numel()];
    super::tensor::copy_strided(x, xl, &mut out);
    let out_strides = Layout::contiguous(shape.to_vec());
    let os = out_strides.strides().to_vec();
    for lin in 0..total {
        let id = ids[lin];
        assert!(id < shape[dim], "scatter_add index out of bounds");
        let mut rem = lin;
        let mut base = 0usize;
        for d in (0..rank).rev() {
            let c = ids_shape[d].max(1);
            let i = rem % c;
            rem /= c;
            let coord = if d == dim { id } else { i };
            base += coord * os[d];
        }
        out[base] = out[base] + src[lin];
    }
    Tensor::from_vec(out, shape.to_vec())
}

fn index_select_impl<T: super::tensor::Elem>(
    x: &[T],
    xl: &Layout,
    ids: &[usize],
    dim: usize,
) -> Tensor {
    let shape = xl.shape();
    let rank = shape.len();
    let dstride = xl.strides()[dim];
    let l = ids.len();
    let mut out_shape = shape.to_vec();
    out_shape[dim] = l;
    let total: usize = out_shape.iter().product();
    let mut out = vec![T::default(); total];
    for lin in 0..total {
        let mut rem = lin;
        let mut src = xl.offset();
        for d in (0..rank).rev() {
            let c = out_shape[d].max(1);
            let i = rem % c;
            rem /= c;
            src += if d == dim {
                let id = ids[i];
                assert!(id < shape[dim], "index_select index out of bounds");
                id * dstride
            } else {
                i * xl.strides()[d]
            };
        }
        out[lin] = x[src];
    }
    Tensor::from_vec(out, out_shape)
}

impl Tensor {
    pub fn gather(&self, dim: usize, ids: &Tensor) -> Tensor {
        assert_eq!(
            ids.shape().len(),
            self.shape().len(),
            "gather: rank mismatch"
        );
        let flat = indices_vec(ids);
        let ids_shape = ids.shape().to_vec();
        let xc = self.contiguous();
        macro_rules! go {
            ($v:expr) => {
                gather_impl($v, &self.layout, &flat, &ids_shape, dim)
            };
        }
        match &xc.buffer {
            CpuBuffer::F32(v) => go!(v),
            CpuBuffer::F64(v) => go!(v),
            CpuBuffer::F16(v) => go!(v),
            CpuBuffer::BF16(v) => go!(v),
            CpuBuffer::U8(v) => go!(v),
            CpuBuffer::U32(v) => go!(v),
            CpuBuffer::I64(v) => go!(v),
        }
    }

    pub fn index_select(&self, dim: usize, ids: &Tensor) -> Tensor {
        let flat = indices_vec(ids);
        assert_eq!(flat.len(), ids.numel(), "index_select ids must be 1-D");
        let xc = self.contiguous();
        macro_rules! go {
            ($v:expr) => {
                index_select_impl($v, &self.layout, &flat, dim)
            };
        }
        match &xc.buffer {
            CpuBuffer::F32(v) => go!(v),
            CpuBuffer::F64(v) => go!(v),
            CpuBuffer::F16(v) => go!(v),
            CpuBuffer::BF16(v) => go!(v),
            CpuBuffer::U8(v) => go!(v),
            CpuBuffer::U32(v) => go!(v),
            CpuBuffer::I64(v) => go!(v),
        }
    }

    pub fn scatter_add(&self, dim: usize, ids: &Tensor, src: &Tensor) -> Tensor {
        assert_eq!(self.dtype(), src.dtype(), "mixed dtypes");
        assert_eq!(
            ids.shape(),
            src.shape(),
            "scatter_add: indexes shape must match src shape"
        );
        let flat = indices_vec(ids);
        let ids_shape = ids.shape().to_vec();
        let src_c = src.contiguous();
        macro_rules! go {
            ($x:expr, $s:expr) => {
                scatter_add_impl($x, &self.layout, &flat, &ids_shape, $s, dim)
            };
        }
        match (&self.buffer, &src_c.buffer) {
            (CpuBuffer::F32(x), CpuBuffer::F32(s)) => go!(x, s),
            (CpuBuffer::F64(x), CpuBuffer::F64(s)) => go!(x, s),
            (CpuBuffer::F16(x), CpuBuffer::F16(s)) => go!(x, s),
            (CpuBuffer::BF16(x), CpuBuffer::BF16(s)) => go!(x, s),
            (CpuBuffer::U8(x), CpuBuffer::U8(s)) => go!(x, s),
            (CpuBuffer::U32(x), CpuBuffer::U32(s)) => go!(x, s),
            (CpuBuffer::I64(x), CpuBuffer::I64(s)) => go!(x, s),
            _ => unreachable!("dtype checked above"),
        }
    }

    pub fn cat(tensors: &[&Tensor], dim: usize) -> Tensor {
        assert!(!tensors.is_empty());
        let dtype = tensors[0].dtype();
        let rank = tensors[0].shape().len();
        assert!(dim < rank);
        let mut out_shape = tensors[0].shape().to_vec();
        for t in tensors {
            assert_eq!(t.dtype(), dtype, "mixed dtypes");
            assert_eq!(t.shape().len(), rank, "cat rank mismatch");
            for d in 0..rank {
                if d != dim {
                    assert_eq!(t.shape()[d], out_shape[d], "cat shape mismatch");
                }
            }
        }
        out_shape[dim] = tensors.iter().map(|t| t.shape()[dim]).sum();
        let out_n: usize = out_shape.iter().product();
        let inner: usize = out_shape[dim + 1..].iter().product();
        let outer: usize = out_shape[..dim].iter().product();

        macro_rules! go {
            ($variant:ident, $t:ty) => {{
                let mut out = vec![<$t>::default(); out_n];
                let mut dim_off = 0usize;
                for t in tensors {
                    let tc = t.contiguous();
                    let CpuBuffer::$variant(data) = &tc.buffer else {
                        unreachable!()
                    };
                    let tdim = t.shape()[dim];
                    for o in 0..outer {
                        let s = o * tdim * inner;
                        let d = o * out_shape[dim] * inner + dim_off * inner;
                        out[d..d + tdim * inner].copy_from_slice(&data[s..s + tdim * inner]);
                    }
                    dim_off += tdim;
                }
                Tensor::from_vec(out, out_shape.clone())
            }};
        }
        match dtype {
            DType::F32 => go!(F32, f32),
            DType::F64 => go!(F64, f64),
            DType::F16 => go!(F16, half::f16),
            DType::BF16 => go!(BF16, half::bf16),
            DType::U8 => go!(U8, u8),
            DType::U32 => go!(U32, u32),
            DType::I64 => go!(I64, i64),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_data(t: &Tensor) -> Vec<f32> {
        let CpuBuffer::F32(v) = &t.buffer else {
            panic!()
        };
        v.as_slice().to_vec()
    }

    #[test]
    fn gather_rows() {
        let x = Tensor::from_vec((0..6).map(|v| v as f32).collect(), vec![2, 3]);
        let ids = Tensor::from_vec(vec![1u32, 1, 1, 0, 0, 0, 1, 1, 1], vec![3, 3]);
        let g = x.gather(0, &ids);
        assert_eq!(g.shape(), &[3, 3]);
        assert_eq!(f32_data(&g), vec![3., 4., 5., 0., 1., 2., 3., 4., 5.]);
    }

    #[test]
    fn index_select_dim1() {
        let x = Tensor::from_vec((0..6).map(|v| v as f32).collect(), vec![2, 3]);
        let ids = Tensor::from_vec(vec![2u32, 0], vec![2]);
        let g = x.index_select(1, &ids);
        assert_eq!(g.shape(), &[2, 2]);
        assert_eq!(f32_data(&g), vec![2., 0., 5., 3.]);
    }

    #[test]
    fn scatter_add_embedding() {
        let table = Tensor::zeros(&[4, 2], DType::F32);
        let ids = Tensor::from_vec(vec![1u32, 1, 3, 3], vec![2, 2]);
        let src = Tensor::from_vec(vec![1f32, 2., 3., 4.], vec![2, 2]);
        let out = table.scatter_add(0, &ids, &src);
        assert_eq!(f32_data(&out), vec![0., 0., 1., 2., 0., 0., 3., 4.]);
    }

    #[test]
    fn cat_dim0_dim1() {
        let a = Tensor::from_vec(vec![1f32, 2.], vec![1, 2]);
        let b = Tensor::from_vec(vec![3f32, 4.], vec![1, 2]);
        let c = Tensor::cat(&[&a, &b], 0);
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(f32_data(&c), vec![1., 2., 3., 4.]);
        let d = Tensor::cat(&[&a, &a], 1);
        assert_eq!(d.shape(), &[1, 4]);
        assert_eq!(f32_data(&d), vec![1., 2., 1., 2.]);
    }
}

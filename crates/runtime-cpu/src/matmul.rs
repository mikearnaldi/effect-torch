use super::tensor::{CpuBuffer, Elem, Tensor};
use effect_torch_runtime::broadcast_shape;

fn naive_gemm<T: Elem + std::ops::Add<Output = T> + std::ops::Mul<Output = T>>(
    a: &[T],
    b: &[T],
    out: &mut [T],
    m: usize,
    n: usize,
    k: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = out[i * n + j];
            for p in 0..k {
                acc = acc + a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
}

fn sgemm(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

fn dgemm(a: &[f64], b: &[f64], out: &mut [f64], m: usize, n: usize, k: usize) {
    unsafe {
        matrixmultiply::dgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

fn batched_matmul<T: Elem + Copy>(
    a: &[T],
    b: &[T],
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    gemm: impl Fn(&[T], &[T], &mut [T], usize, usize, usize),
) -> Vec<T> {
    let mut out = vec![T::default(); batch * m * n];
    for bi in 0..batch {
        gemm(
            &a[bi * m * k..(bi + 1) * m * k],
            &b[bi * k * n..(bi + 1) * k * n],
            &mut out[bi * m * n..(bi + 1) * m * n],
            m,
            n,
            k,
        );
    }
    out
}

impl Tensor {
    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        self.try_matmul(rhs)
            .unwrap_or_else(|message| panic!("{message}"))
    }

    pub fn try_matmul(&self, rhs: &Tensor) -> Result<Tensor, &'static str> {
        assert_eq!(self.dtype(), rhs.dtype(), "mixed dtypes");
        let a_rank = self.shape().len();
        let b_rank = rhs.shape().len();
        assert!(a_rank >= 2 && b_rank >= 2, "matmul needs rank >= 2");
        let m = self.shape()[a_rank - 2];
        let k = self.shape()[a_rank - 1];
        let k2 = rhs.shape()[b_rank - 2];
        let n = rhs.shape()[b_rank - 1];
        assert_eq!(k, k2, "matmul inner dim mismatch");
        let batch_shape = broadcast_shape(&self.shape()[..a_rank - 2], &rhs.shape()[..b_rank - 2]);
        let batch: usize = batch_shape.iter().product();

        let a_bc = self.view(
            self.layout.broadcast_to(
                &[
                    batch_shape.as_slice(),
                    self.shape()[a_rank - 2..].to_vec().as_slice(),
                ]
                .concat(),
            ),
        );
        let b_bc = rhs.view(
            rhs.layout.broadcast_to(
                &[
                    batch_shape.as_slice(),
                    rhs.shape()[b_rank - 2..].to_vec().as_slice(),
                ]
                .concat(),
            ),
        );
        let a_c = a_bc.contiguous();
        let b_c = b_bc.contiguous();

        let mut out_shape = batch_shape;
        out_shape.extend([m, n]);

        macro_rules! go {
            ($av:expr, $bv:expr, $t:ty, $gemm:expr) => {{
                let out = batched_matmul::<$t>($av, $bv, batch, m, n, k, $gemm);
                Tensor::from_vec(out, out_shape.clone())
            }};
        }
        match (&a_c.buffer, &b_c.buffer) {
            (CpuBuffer::F32(a), CpuBuffer::F32(b)) => {
                Ok(go!(a, b, f32, |a, b, o, m, n, k| sgemm(a, b, o, m, n, k)))
            }
            (CpuBuffer::F64(a), CpuBuffer::F64(b)) => {
                Ok(go!(a, b, f64, |a, b, o, m, n, k| dgemm(a, b, o, m, n, k)))
            }
            (CpuBuffer::F16(_), CpuBuffer::F16(_)) => {
                Err("f16 matmul is not supported on the CPU backend")
            }
            (CpuBuffer::BF16(_), CpuBuffer::BF16(_)) => {
                Err("bf16 matmul is not supported on the CPU backend")
            }
            (CpuBuffer::U8(a), CpuBuffer::U8(b)) => Ok(go!(a, b, u8, naive_gemm)),
            (CpuBuffer::U32(a), CpuBuffer::U32(b)) => Ok(go!(a, b, u32, naive_gemm)),
            (CpuBuffer::I64(a), CpuBuffer::I64(b)) => Ok(go!(a, b, i64, naive_gemm)),
            _ => unreachable!("dtype checked above"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_2d() {
        let a = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], vec![2, 3]);
        let b = Tensor::from_vec(vec![1f32, 0., 0., 1., 1., 1.], vec![3, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[2, 2]);
        let CpuBuffer::F32(v) = &c.buffer else {
            panic!()
        };
        assert_eq!(v.as_slice(), &[4., 5., 10., 11.]);
    }

    #[test]
    fn matmul_batched() {
        let a = Tensor::from_vec(vec![1f32; 8], vec![2, 2, 2]);
        let b = Tensor::from_vec(vec![2f32; 4], vec![2, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[2, 2, 2]);
        let CpuBuffer::F32(v) = &c.buffer else {
            panic!()
        };
        assert!(v.iter().all(|&x| x == 4.0));
    }

    #[test]
    fn unsupported_half_matmul_is_fallible() {
        let a = Tensor::from_vec(vec![half::f16::ZERO; 4], vec![2, 2]);
        let b = Tensor::from_vec(vec![half::f16::ZERO; 4], vec![2, 2]);
        assert_eq!(
            a.try_matmul(&b).err(),
            Some("f16 matmul is not supported on the CPU backend")
        );

        let a = Tensor::from_vec(vec![half::bf16::ZERO; 4], vec![2, 2]);
        let b = Tensor::from_vec(vec![half::bf16::ZERO; 4], vec![2, 2]);
        assert_eq!(
            a.try_matmul(&b).err(),
            Some("bf16 matmul is not supported on the CPU backend")
        );
    }
}

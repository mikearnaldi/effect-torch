use super::tensor::{CpuBuffer, Tensor};
use effect_torch_runtime::DType;

fn lu_decompose(a: &[f64], n: usize) -> Option<(Vec<f64>, Vec<usize>, i32)> {
    let mut lu = a.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();
    let mut sign = 1i32;
    for col in 0..n {
        let mut pivot = col;
        let mut best = lu[col * n + col].abs();
        for r in col + 1..n {
            let v = lu[r * n + col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < 1e-300 {
            return None;
        }
        if pivot != col {
            for j in 0..n {
                lu.swap(col * n + j, pivot * n + j);
            }
            perm.swap(col, pivot);
            sign = -sign;
        }
        let d = lu[col * n + col];
        for r in col + 1..n {
            let factor = lu[r * n + col] / d;
            lu[r * n + col] = factor;
            for j in col + 1..n {
                lu[r * n + j] -= factor * lu[col * n + j];
            }
        }
    }
    Some((lu, perm, sign))
}

fn lu_solve(lu: &[f64], perm: &[usize], b: &[f64], n: usize) -> Vec<f64> {
    let mut x: Vec<f64> = perm.iter().map(|&p| b[p]).collect();
    for i in 1..n {
        for j in 0..i {
            x[i] -= lu[i * n + j] * x[j];
        }
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            x[i] -= lu[i * n + j] * x[j];
        }
        x[i] /= lu[i * n + i];
    }
    x
}

impl Tensor {
    fn batched_square_f64(&self) -> (Vec<f64>, usize, usize) {
        let shape = self.shape();
        assert!(shape.len() >= 2);
        let n = shape[shape.len() - 1];
        assert_eq!(shape[shape.len() - 2], n, "linalg requires square matrices");
        let batch: usize = shape[..shape.len() - 2].iter().product();
        let c = self.cast(DType::F64).contiguous();
        let CpuBuffer::F64(v) = &c.buffer else {
            unreachable!()
        };
        (v.as_slice().to_vec(), n, batch)
    }

    pub fn det(&self) -> Tensor {
        let (a, n, batch) = self.batched_square_f64();
        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            let m = &a[b * n * n..(b + 1) * n * n];
            match lu_decompose(m, n) {
                Some((lu, _, sign)) => {
                    let mut d = sign as f64;
                    for i in 0..n {
                        d *= lu[i * n + i];
                    }
                    out.push(d);
                }
                None => out.push(0.0),
            }
        }
        let mut shape = self.shape()[..self.shape().len() - 2].to_vec();
        if shape.is_empty() {
            shape.push(1);
        }
        Tensor::from_vec(out, shape).cast(self.dtype())
    }

    pub fn inverse(&self) -> Tensor {
        self.try_inverse()
            .unwrap_or_else(|message| panic!("{message}"))
    }

    pub fn try_inverse(&self) -> Result<Tensor, &'static str> {
        let (a, n, batch) = self.batched_square_f64();
        let mut out = vec![0f64; batch * n * n];
        for b in 0..batch {
            let m = &a[b * n * n..(b + 1) * n * n];
            let (lu, perm, _) = lu_decompose(m, n).ok_or("matrix is singular")?;
            for col in 0..n {
                let mut e = vec![0f64; n];
                e[col] = 1.0;
                let x = lu_solve(&lu, &perm, &e, n);
                for r in 0..n {
                    out[b * n * n + r * n + col] = x[r];
                }
            }
        }
        Ok(Tensor::from_vec(out, self.shape().to_vec()).cast(self.dtype()))
    }

    pub fn solve(&self, rhs: &Tensor) -> Tensor {
        self.try_solve(rhs)
            .unwrap_or_else(|message| panic!("{message}"))
    }

    pub fn try_solve(&self, rhs: &Tensor) -> Result<Tensor, &'static str> {
        let (a, n, batch) = self.batched_square_f64();
        let rshape = rhs.shape();
        assert!(rshape.len() >= 2);
        assert_eq!(rshape[rshape.len() - 2], n);
        let nrhs = rshape[rshape.len() - 1];
        let rbatch: usize = rshape[..rshape.len() - 2].iter().product();
        assert_eq!(rbatch, batch, "solve batch mismatch");
        let rc = rhs.cast(DType::F64).contiguous();
        let CpuBuffer::F64(bv) = &rc.buffer else {
            unreachable!()
        };
        let mut out = vec![0f64; batch * n * nrhs];
        for b in 0..batch {
            let m = &a[b * n * n..(b + 1) * n * n];
            let (lu, perm, _) = lu_decompose(m, n).ok_or("matrix is singular")?;
            for col in 0..nrhs {
                let colv: Vec<f64> = (0..n).map(|r| bv[b * n * nrhs + r * nrhs + col]).collect();
                let x = lu_solve(&lu, &perm, &colv, n);
                for r in 0..n {
                    out[b * n * nrhs + r * nrhs + col] = x[r];
                }
            }
        }
        Ok(Tensor::from_vec(out, rshape.to_vec()).cast(self.dtype()))
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
    fn inverse_and_det() {
        let a = Tensor::from_vec(vec![2f32, 0., 0., 4.], vec![2, 2]);
        let d = a.det();
        assert!((f32_data(&d)[0] - 8.0).abs() < 1e-5);
        let inv = a.inverse();
        let got = f32_data(&inv);
        assert!((got[0] - 0.5).abs() < 1e-6 && (got[3] - 0.25).abs() < 1e-6);
        let prod = a.matmul(&inv);
        let p = f32_data(&prod);
        assert!((p[0] - 1.0).abs() < 1e-5 && p[1].abs() < 1e-5 && (p[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn solve_system() {
        let a = Tensor::from_vec(vec![3f32, 1., 1., 2.], vec![2, 2]);
        let b = Tensor::from_vec(vec![9f32, 8.], vec![2, 1]);
        let x = a.solve(&b);
        let got = f32_data(&x);
        assert!((got[0] - 2.0).abs() < 1e-5 && (got[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn singular_inverse_and_solve_are_fallible() {
        let a = Tensor::from_vec(vec![1f32, 2., 2., 4.], vec![2, 2]);
        let b = Tensor::from_vec(vec![1f32, 2.], vec![2, 1]);
        assert_eq!(a.try_inverse().err(), Some("matrix is singular"));
        assert_eq!(a.try_solve(&b).err(), Some("matrix is singular"));
    }
}

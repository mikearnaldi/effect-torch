use super::cpu::{CpuBuffer, Tensor};
use super::dtype::DType;

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
    fn square_f64(&self) -> (Vec<f64>, usize) {
        let shape = self.shape();
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[0], shape[1], "linalg requires a square matrix");
        let n = shape[0];
        let c = self.cast(DType::F64).contiguous();
        let CpuBuffer::F64(v) = &c.buffer else { unreachable!() };
        (v.as_slice().to_vec(), n)
    }

    pub fn det(&self) -> Tensor {
        let (a, n) = self.square_f64();
        let Some((lu, _, sign)) = lu_decompose(&a, n) else {
            return Tensor::full(&[1], 0.0, self.dtype());
        };
        let mut d = sign as f64;
        for i in 0..n {
            d *= lu[i * n + i];
        }
        Tensor::full(&[1], d, self.dtype())
    }

    pub fn inverse(&self) -> Tensor {
        let (a, n) = self.square_f64();
        let (lu, perm, _) = lu_decompose(&a, n).expect("matrix is singular");
        let mut inv = vec![0f64; n * n];
        for col in 0..n {
            let mut e = vec![0f64; n];
            e[col] = 1.0;
            let x = lu_solve(&lu, &perm, &e, n);
            for r in 0..n {
                inv[r * n + col] = x[r];
            }
        }
        let out = Tensor::from_vec(inv, vec![n, n]);
        out.cast(self.dtype())
    }

    pub fn solve(&self, rhs: &Tensor) -> Tensor {
        let (a, n) = self.square_f64();
        let shape = rhs.shape();
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[0], n);
        let nrhs = shape[1];
        let rc = rhs.cast(DType::F64).contiguous();
        let CpuBuffer::F64(bv) = &rc.buffer else { unreachable!() };
        let (lu, perm, _) = lu_decompose(&a, n).expect("matrix is singular");
        let mut out = vec![0f64; n * nrhs];
        for col in 0..nrhs {
            let b: Vec<f64> = (0..n).map(|r| bv[r * nrhs + col]).collect();
            let x = lu_solve(&lu, &perm, &b, n);
            for r in 0..n {
                out[r * nrhs + col] = x[r];
            }
        }
        Tensor::from_vec(out, shape.to_vec()).cast(self.dtype())
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
}

use super::tensor::Tensor;
use effect_torch_runtime::DType;
use std::sync::Mutex;

struct Xoroshiro128Plus {
    s0: u64,
    s1: u64,
}

impl Xoroshiro128Plus {
    fn new(seed: u64) -> Self {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        Xoroshiro128Plus {
            s0: next(),
            s1: next(),
        }
    }

    fn next_u64(&mut self) -> u64 {
        let s0 = self.s0;
        let mut s1 = self.s1;
        let r = s0.wrapping_add(s1);
        s1 ^= s0;
        self.s0 = s0.rotate_left(55) ^ s1 ^ (s1 << 14);
        self.s1 = s1.rotate_left(36);
        r
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
}

static RNG: Mutex<Option<Xoroshiro128Plus>> = Mutex::new(None);

fn with_rng<T>(f: impl FnOnce(&mut Xoroshiro128Plus) -> T) -> T {
    let mut guard = RNG.lock().unwrap();
    let rng = guard.get_or_insert_with(|| Xoroshiro128Plus::new(299792458));
    f(rng)
}

pub fn reseed(seed: u64) {
    let mut guard = RNG.lock().unwrap();
    *guard = Some(Xoroshiro128Plus::new(seed));
}

impl Tensor {
    pub fn arange(start: f64, end: f64, step: f64, dtype: DType) -> Self {
        assert!(step != 0.0);
        let n = ((end - start) / step).ceil().max(0.0) as usize;
        let data: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
        Tensor::full(&[n], 0.0, dtype).with_values(&data)
    }

    pub fn eye(n: usize, dtype: DType) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Tensor::full(&[n, n], 0.0, dtype).with_values(&data)
    }

    pub fn randn(shape: &[usize], dtype: DType) -> Self {
        let n: usize = shape.iter().product();
        let data: Vec<f64> = with_rng(|rng| {
            (0..n)
                .map(|_| {
                    let u1 = rng.next_f64().max(1e-12);
                    let u2 = rng.next_f64();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                })
                .collect()
        });
        Tensor::full(shape, 0.0, dtype).with_values(&data)
    }

    pub fn uniform(lo: f64, hi: f64, shape: &[usize], dtype: DType) -> Self {
        let n: usize = shape.iter().product();
        let data: Vec<f64> =
            with_rng(|rng| (0..n).map(|_| lo + (hi - lo) * rng.next_f64()).collect());
        Tensor::full(shape, 0.0, dtype).with_values(&data)
    }

    fn with_values(&self, data: &[f64]) -> Self {
        assert_eq!(data.len(), self.numel());
        macro_rules! go {
            ($variant:ident, $t:ty) => {
                Tensor::from_vec(
                    data.iter()
                        .map(|&x| <$t as super::tensor::Elem>::from_f64(x))
                        .collect::<Vec<$t>>(),
                    self.shape().to_vec(),
                )
            };
        }
        match self.dtype() {
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
    use crate::CpuBuffer;

    #[test]
    fn arange_eye() {
        let a = Tensor::arange(0.0, 5.0, 2.0, DType::F32);
        let CpuBuffer::F32(v) = &a.buffer else {
            panic!()
        };
        assert_eq!(v.as_slice(), &[0.0, 2.0, 4.0]);
        let e = Tensor::eye(2, DType::F32);
        let CpuBuffer::F32(v) = &e.buffer else {
            panic!()
        };
        assert_eq!(v.as_slice(), &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn randn_deterministic() {
        reseed(42);
        let a = Tensor::randn(&[4], DType::F32);
        reseed(42);
        let b = Tensor::randn(&[4], DType::F32);
        let CpuBuffer::F32(x) = &a.buffer else {
            panic!()
        };
        let CpuBuffer::F32(y) = &b.buffer else {
            panic!()
        };
        assert_eq!(x, y);
    }
}

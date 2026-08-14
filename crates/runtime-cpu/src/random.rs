//! Tensor factories and pseudo-random generation.
//!
//! Deterministic factories (`arange`, `eye`) and random factories (`randn`,
//! `uniform`) share the requirements/into/wrapper contract of the rest of the
//! crate. Random values come from a process-global `xoroshiro128+` generator
//! behind a mutex; [`reseed`] replaces its state. `randn` uses the
//! Box–Muller transform with `u1` clamped away from zero.
//!
//! The executor does not use the global stream: it calls the seeded
//! `*_seeded_into` variants with a per-invocation, per-node seed so compiled
//! programs are reproducible regardless of global RNG state.

use super::tensor::{CpuDestination, CpuOperationRequirements, CpuTensorRequirement, Elem, Tensor};
use effect_torch_runtime::DType;
use half::{bf16, f16};
use std::sync::Mutex;

/// xoroshiro128+ PRNG. The two 64-bit state words are seeded from a single
/// `u64` by running an LCG-style bit mixer (xor-shift steps over a
/// golden-ratio offset) twice.
struct Xoroshiro128Plus {
    s0: u64,
    s1: u64,
}

impl Xoroshiro128Plus {
    fn new(seed: u64) -> Self {
        let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        Self {
            s0: next(),
            s1: next(),
        }
    }

    fn next_u64(&mut self) -> u64 {
        let s0 = self.s0;
        let mut s1 = self.s1;
        let result = s0.wrapping_add(s1);
        s1 ^= s0;
        self.s0 = s0.rotate_left(55) ^ s1 ^ (s1 << 14);
        self.s1 = s1.rotate_left(36);
        result
    }

    /// Next uniform double in `[0, 1)` (53-bit mantissa).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
}

static RNG: Mutex<Option<Xoroshiro128Plus>> = Mutex::new(None);

/// Borrows the global generator, lazily seeding it with a fixed default
/// (299792458) so unseeded runs are still reproducible within a process.
fn with_rng<T>(function: impl FnOnce(&mut Xoroshiro128Plus) -> T) -> T {
    let mut guard = RNG.lock().unwrap();
    let rng = guard.get_or_insert_with(|| Xoroshiro128Plus::new(299792458));
    function(rng)
}

/// Replaces the global generator state with a fresh stream seeded by `seed`.
pub fn reseed(seed: u64) {
    let mut guard = RNG.lock().unwrap();
    *guard = Some(Xoroshiro128Plus::new(seed));
}

fn arange_len(start: f64, end: f64, step: f64) -> usize {
    assert!(step != 0.0);
    ((end - start) / step).ceil().max(0.0) as usize
}

fn allocate_output(
    requirements: CpuOperationRequirements,
    write: impl FnOnce(&mut CpuDestination<'_>) -> Result<(), String>,
) -> Tensor {
    let mut output = Tensor::empty(&requirements.output.shape, requirements.output.dtype);
    {
        let mut destination = output
            .destination()
            .expect("new CPU tensor storage must be unique");
        write(&mut destination).expect("new CPU tensor must satisfy random output requirements");
    }
    output
}

fn write_generated<T: Elem>(
    operation: &str,
    destination: &mut CpuDestination<'_>,
    mut generate: impl FnMut(usize) -> f64,
) -> Result<(), String> {
    destination.write_current::<T, _>(operation, |output| {
        for (index, value) in output.iter_mut().enumerate() {
            *value = T::from_f64(generate(index));
        }
    })
}

fn write_for_dtype(
    operation: &str,
    destination: &mut CpuDestination<'_>,
    generate: impl FnMut(usize) -> f64,
) -> Result<(), String> {
    macro_rules! write {
        ($type:ty) => {
            write_generated::<$type>(operation, destination, generate)
        };
    }
    match destination.dtype() {
        DType::F32 => write!(f32),
        DType::F64 => write!(f64),
        DType::F16 => write!(f16),
        DType::BF16 => write!(bf16),
        DType::U8 => write!(u8),
        DType::U32 => write!(u32),
        DType::I64 => write!(i64),
    }
}

fn randn_with_rng_into(
    destination: &mut CpuDestination<'_>,
    rng: &mut Xoroshiro128Plus,
) -> Result<(), String> {
    macro_rules! write {
        ($type:ty) => {
            destination.write_current::<$type, _>("randn", |output| {
                for value in output {
                    let u1 = rng.next_f64().max(1e-12);
                    let u2 = rng.next_f64();
                    let sample = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    *value = <$type as Elem>::from_f64(sample);
                }
            })
        };
    }
    match destination.dtype() {
        DType::F32 => write!(f32),
        DType::F64 => write!(f64),
        DType::F16 => write!(f16),
        DType::BF16 => write!(bf16),
        DType::U8 => write!(u8),
        DType::U32 => write!(u32),
        DType::I64 => write!(i64),
    }
}

fn uniform_with_rng_into(
    lo: f64,
    hi: f64,
    destination: &mut CpuDestination<'_>,
    rng: &mut Xoroshiro128Plus,
) -> Result<(), String> {
    macro_rules! write {
        ($type:ty) => {
            destination.write_current::<$type, _>("uniform", |output| {
                for value in output {
                    *value = <$type as Elem>::from_f64(lo + (hi - lo) * rng.next_f64());
                }
            })
        };
    }
    match destination.dtype() {
        DType::F32 => write!(f32),
        DType::F64 => write!(f64),
        DType::F16 => write!(f16),
        DType::BF16 => write!(bf16),
        DType::U8 => write!(u8),
        DType::U32 => write!(u32),
        DType::I64 => write!(i64),
    }
}

impl Tensor {
    pub fn arange_requirements(
        start: f64,
        end: f64,
        step: f64,
        dtype: DType,
    ) -> CpuOperationRequirements {
        CpuOperationRequirements::without_scratch(&[arange_len(start, end, step)], dtype)
    }

    pub fn arange_output_requirements(
        start: f64,
        end: f64,
        step: f64,
        dtype: DType,
    ) -> CpuTensorRequirement {
        Self::arange_requirements(start, end, step, dtype).output
    }

    pub fn arange_scratch_requirements(
        _start: f64,
        _end: f64,
        _step: f64,
        _dtype: DType,
    ) -> &'static [CpuTensorRequirement] {
        &[]
    }

    /// Writes `start + i * step` into `destination`, whose length must equal
    /// `ceil((end - start) / step)` (clamped at zero; `step` must be non-zero).
    pub fn arange_into(
        start: f64,
        end: f64,
        step: f64,
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        let len = arange_len(start, end, step);
        if destination.shape() != [len] {
            return Err(format!(
                "arange destination shape mismatch: expected [{len}], got {:?}",
                destination.shape()
            ));
        }
        write_for_dtype("arange", destination, |index| start + index as f64 * step)
    }

    pub fn arange(start: f64, end: f64, step: f64, dtype: DType) -> Self {
        allocate_output(
            Self::arange_requirements(start, end, step, dtype),
            |destination| Self::arange_into(start, end, step, destination),
        )
    }

    pub fn eye_requirements(n: usize, dtype: DType) -> CpuOperationRequirements {
        CpuOperationRequirements::without_scratch(&[n, n], dtype)
    }

    pub fn eye_output_requirements(n: usize, dtype: DType) -> CpuTensorRequirement {
        Self::eye_requirements(n, dtype).output
    }

    pub fn eye_scratch_requirements(_n: usize, _dtype: DType) -> &'static [CpuTensorRequirement] {
        &[]
    }

    /// Writes the `n × n` identity matrix into a square rank-2 destination.
    pub fn eye_into(destination: &mut CpuDestination<'_>) -> Result<(), String> {
        if destination.shape().len() != 2 || destination.shape()[0] != destination.shape()[1] {
            return Err(format!(
                "eye destination must be square rank-2, got {:?}",
                destination.shape()
            ));
        }
        let n = destination.shape()[0];
        write_for_dtype("eye", destination, |index| {
            if index / n == index % n {
                1.0
            } else {
                0.0
            }
        })
    }

    pub fn eye(n: usize, dtype: DType) -> Self {
        allocate_output(Self::eye_requirements(n, dtype), Self::eye_into)
    }

    pub fn randn_requirements(shape: &[usize], dtype: DType) -> CpuOperationRequirements {
        CpuOperationRequirements::without_scratch(shape, dtype)
    }

    pub fn randn_output_requirements(shape: &[usize], dtype: DType) -> CpuTensorRequirement {
        Self::randn_requirements(shape, dtype).output
    }

    pub fn randn_scratch_requirements(
        _shape: &[usize],
        _dtype: DType,
    ) -> &'static [CpuTensorRequirement] {
        &[]
    }

    /// Fills `destination` with standard-normal samples from the global
    /// stream (Box–Muller on `f64`, narrowed to the destination dtype).
    pub fn randn_into(destination: &mut CpuDestination<'_>) -> Result<(), String> {
        with_rng(|rng| randn_with_rng_into(destination, rng))
    }

    pub(crate) fn randn_seeded_into(
        seed: u64,
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        randn_with_rng_into(destination, &mut Xoroshiro128Plus::new(seed))
    }

    pub fn randn(shape: &[usize], dtype: DType) -> Self {
        allocate_output(Self::randn_requirements(shape, dtype), Self::randn_into)
    }

    pub fn uniform_requirements(
        _lo: f64,
        _hi: f64,
        shape: &[usize],
        dtype: DType,
    ) -> CpuOperationRequirements {
        CpuOperationRequirements::without_scratch(shape, dtype)
    }

    pub fn uniform_output_requirements(
        lo: f64,
        hi: f64,
        shape: &[usize],
        dtype: DType,
    ) -> CpuTensorRequirement {
        Self::uniform_requirements(lo, hi, shape, dtype).output
    }

    pub fn uniform_scratch_requirements(
        _lo: f64,
        _hi: f64,
        _shape: &[usize],
        _dtype: DType,
    ) -> &'static [CpuTensorRequirement] {
        &[]
    }

    /// Fills `destination` with samples uniform in `[lo, hi)` from the
    /// global stream.
    pub fn uniform_into(
        lo: f64,
        hi: f64,
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        with_rng(|rng| uniform_with_rng_into(lo, hi, destination, rng))
    }

    pub(crate) fn uniform_seeded_into(
        lo: f64,
        hi: f64,
        seed: u64,
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        uniform_with_rng_into(lo, hi, destination, &mut Xoroshiro128Plus::new(seed))
    }

    pub fn uniform(lo: f64, hi: f64, shape: &[usize], dtype: DType) -> Self {
        allocate_output(
            Self::uniform_requirements(lo, hi, shape, dtype),
            |destination| Self::uniform_into(lo, hi, destination),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::ExecutableAllocationGuard;
    use crate::CpuBuffer;

    #[test]
    fn arange_eye() {
        let arange = Tensor::arange(0.0, 5.0, 2.0, DType::F32);
        let CpuBuffer::F32(values) = &arange.buffer else {
            panic!()
        };
        assert_eq!(values.as_slice(), &[0.0, 2.0, 4.0]);
        let eye = Tensor::eye(2, DType::F32);
        let CpuBuffer::F32(values) = &eye.buffer else {
            panic!()
        };
        assert_eq!(values.as_slice(), &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn randn_deterministic() {
        let mut a = Tensor::empty(&[4], DType::F32);
        let mut b = Tensor::empty(&[4], DType::F32);
        randn_with_rng_into(
            &mut a.destination().unwrap(),
            &mut Xoroshiro128Plus::new(42),
        )
        .unwrap();
        randn_with_rng_into(
            &mut b.destination().unwrap(),
            &mut Xoroshiro128Plus::new(42),
        )
        .unwrap();
        let CpuBuffer::F32(x) = &a.buffer else {
            panic!()
        };
        let CpuBuffer::F32(y) = &b.buffer else {
            panic!()
        };
        assert_eq!(x, y);
    }

    #[test]
    fn random_into_is_allocation_free_and_matches_wrapper_stream() {
        let requirements = Tensor::randn_requirements(&[8], DType::F32);
        assert_eq!(requirements.output.bytes, 32);
        assert_eq!(requirements.scratch_bytes(), 0);
        let mut output = Tensor::empty(&requirements.output.shape, requirements.output.dtype);
        let mut output_rng = Xoroshiro128Plus::new(7);
        {
            let _guard = ExecutableAllocationGuard::enter();
            randn_with_rng_into(&mut output.destination().unwrap(), &mut output_rng).unwrap();
        }
        let mut wrapped_rng = Xoroshiro128Plus::new(7);
        let wrapped = allocate_output(requirements, |destination| {
            randn_with_rng_into(destination, &mut wrapped_rng)
        });
        let CpuBuffer::F32(output) = &output.buffer else {
            panic!()
        };
        let CpuBuffer::F32(wrapped) = &wrapped.buffer else {
            panic!()
        };
        assert_eq!(output, wrapped);
    }
}

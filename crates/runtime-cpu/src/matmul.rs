//! Matrix multiplication with NumPy-style batch broadcasting.
//!
//! Shapes of rank ≥ 2 are treated as `batch..., m, k` × `batch..., k, n`;
//! batch dimensions broadcast against each other. Both operands may carry
//! arbitrary strided layouts — the kernel indexes them through their strides
//! rather than materializing contiguous copies.
//!
//! The only algorithm currently selected is [`MatmulAlgorithm::Naive`]: a
//! row-major `m × k × n` loop that accumulates directly into the destination.
//! Floating-point dtypes accumulate with fused multiply-add; integer dtypes
//! (`u8`, `u32`, `i64`) use wrapping-free plain `c + a * b` semantics in
//! their own type. `f16`/`bf16` matmul is deliberately rejected at planning
//! time.

use super::tensor::{CpuBuffer, CpuDestination, CpuTensorRequirement, Elem, Tensor};
use effect_torch_runtime::{DType, Layout};

/// Kernel selected for a matmul invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatmulAlgorithm {
    /// Direct triple loop over the (possibly strided) operand layouts.
    Naive,
}

/// Exact resources and algorithm selected for one matmul invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulRequirements {
    pub output: CpuTensorRequirement,
    pub scratch: Vec<CpuTensorRequirement>,
    pub algorithm: MatmulAlgorithm,
    pub left_layout: Layout,
    pub right_layout: Layout,
    pub batch: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl MatmulRequirements {
    pub fn scratch_bytes(&self) -> usize {
        self.scratch
            .iter()
            .try_fold(0usize, |total, requirement| {
                total.checked_add(requirement.bytes)
            })
            .expect("matmul scratch byte size overflow")
    }
}

fn checked_product(values: &[usize]) -> Result<usize, &'static str> {
    values
        .iter()
        .try_fold(1usize, |total, &value| total.checked_mul(value))
        .ok_or("matmul element count overflow")
}

fn checked_requirement(
    shape: &[usize],
    dtype: DType,
) -> Result<CpuTensorRequirement, &'static str> {
    checked_product(shape)?
        .checked_mul(dtype.size_in_bytes())
        .ok_or("matmul byte size overflow")?;
    Ok(CpuTensorRequirement::new(shape, dtype))
}

fn matmul_output_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, &'static str> {
    if a.len() < 2 || b.len() < 2 {
        return Err("matmul needs rank >= 2");
    }
    let a_batch_rank = a.len() - 2;
    let b_batch_rank = b.len() - 2;
    let batch_rank = a_batch_rank.max(b_batch_rank);
    let mut output = Vec::with_capacity(batch_rank + 2);
    for dimension in 0..batch_rank {
        let left = if dimension < batch_rank - a_batch_rank {
            1
        } else {
            a[dimension - (batch_rank - a_batch_rank)]
        };
        let right = if dimension < batch_rank - b_batch_rank {
            1
        } else {
            b[dimension - (batch_rank - b_batch_rank)]
        };
        if left != right && left != 1 && right != 1 {
            return Err("matmul batch dimensions are not broadcast-compatible");
        }
        output.push(left.max(right));
    }
    let m = a[a.len() - 2];
    let k = a[a.len() - 1];
    if k != b[b.len() - 2] {
        return Err("matmul inner dim mismatch");
    }
    output.extend([m, b[b.len() - 1]]);
    Ok(output)
}

fn batch_dimension(layout: &Layout, batch_rank: usize, dimension: usize) -> usize {
    let source_batch_rank = layout.rank() - 2;
    if dimension < batch_rank - source_batch_rank {
        1
    } else {
        layout.shape()[dimension - (batch_rank - source_batch_rank)]
    }
}

fn batch_offset(layout: &Layout, output_batch_shape: &[usize], mut batch_index: usize) -> usize {
    let source_batch_rank = layout.rank() - 2;
    let mut offset = layout.offset();
    for dimension in (0..output_batch_shape.len()).rev() {
        let width = output_batch_shape[dimension].max(1);
        let coordinate = batch_index % width;
        batch_index /= width;
        if dimension >= output_batch_shape.len() - source_batch_rank {
            let source_dimension = dimension - (output_batch_shape.len() - source_batch_rank);
            if layout.shape()[source_dimension] != 1 {
                offset += coordinate * layout.strides()[source_dimension];
            }
        }
    }
    offset
}

fn validate_matmul(
    a: &Tensor,
    b: &Tensor,
    destination: &CpuDestination<'_>,
    scratch: &[CpuDestination<'_>],
    requirements: &MatmulRequirements,
) -> Result<(), String> {
    if a.dtype() != b.dtype() || requirements.output.dtype != a.dtype() {
        return Err("matmul: inputs and exact requirements must have one dtype".into());
    }
    if a.layout != requirements.left_layout || b.layout != requirements.right_layout {
        return Err("matmul: input layouts do not match exact requirements".into());
    }
    if a.shape().len() < 2 || b.shape().len() < 2 {
        return Err("matmul needs rank >= 2".into());
    }
    let batch_rank = (a.shape().len() - 2).max(b.shape().len() - 2);
    if requirements.output.shape.len() != batch_rank + 2 {
        return Err("matmul: exact output rank is invalid".into());
    }
    for dimension in 0..batch_rank {
        let left = batch_dimension(&a.layout, batch_rank, dimension);
        let right = batch_dimension(&b.layout, batch_rank, dimension);
        if left != right && left != 1 && right != 1 {
            return Err("matmul: input batch dimensions are incompatible".into());
        }
        if requirements.output.shape[dimension] != left.max(right) {
            return Err("matmul: exact output batch shape is invalid".into());
        }
    }
    let m = a.shape()[a.shape().len() - 2];
    let k = a.shape()[a.shape().len() - 1];
    let k2 = b.shape()[b.shape().len() - 2];
    let n = b.shape()[b.shape().len() - 1];
    if k != k2 || (requirements.m, requirements.n, requirements.k) != (m, n, k) {
        return Err("matmul: matrix dimensions do not match exact requirements".into());
    }
    let batch =
        checked_product(&requirements.output.shape[..batch_rank]).map_err(str::to_string)?;
    if requirements.batch != batch
        || requirements.output.shape[batch_rank] != m
        || requirements.output.shape[batch_rank + 1] != n
    {
        return Err("matmul: output dimensions do not match exact requirements".into());
    }
    if destination.shape() != requirements.output.shape
        || destination.dtype() != requirements.output.dtype
    {
        return Err("matmul: destination does not match exact requirements".into());
    }
    if scratch.len() != requirements.scratch.len() {
        return Err("matmul: scratch does not match exact requirements".into());
    }
    for (actual, expected) in scratch.iter().zip(&requirements.scratch) {
        if actual.shape() != expected.shape || actual.dtype() != expected.dtype {
            return Err("matmul: scratch does not match exact requirements".into());
        }
    }
    match (requirements.algorithm, a.dtype()) {
        (MatmulAlgorithm::Naive, DType::F32 | DType::F64 | DType::U8 | DType::U32 | DType::I64) => {
            Ok(())
        }
        _ => Err("matmul: algorithm and dtype do not match exact requirements".into()),
    }
}

fn naive_into<T>(
    a: &[T],
    a_layout: &Layout,
    b: &[T],
    b_layout: &Layout,
    destination: &mut CpuDestination<'_>,
    requirements: &MatmulRequirements,
    multiply_add: impl Fn(T, T, T) -> T,
) -> Result<(), String>
where
    T: Elem,
{
    let a_row = a_layout.strides()[a_layout.rank() - 2];
    let a_col = a_layout.strides()[a_layout.rank() - 1];
    let b_row = b_layout.strides()[b_layout.rank() - 2];
    let b_col = b_layout.strides()[b_layout.rank() - 1];
    let batch_rank = requirements.output.shape.len() - 2;
    destination.write_current::<T, _>("matmul", |output| {
        for batch in 0..requirements.batch {
            let a_offset = batch_offset(a_layout, &requirements.output.shape[..batch_rank], batch);
            let b_offset = batch_offset(b_layout, &requirements.output.shape[..batch_rank], batch);
            let output_offset = batch * requirements.m * requirements.n;
            for row in 0..requirements.m {
                let output_row = output_offset + row * requirements.n;
                output[output_row..output_row + requirements.n].fill(T::default());
                for inner in 0..requirements.k {
                    let left = a[a_offset + row * a_row + inner * a_col];
                    for column in 0..requirements.n {
                        let destination = &mut output[output_row + column];
                        *destination = multiply_add(
                            left,
                            b[b_offset + inner * b_row + column * b_col],
                            *destination,
                        );
                    }
                }
            }
        }
    })
}

impl Tensor {
    /// Plans a matmul: validates dtypes/ranks/broadcasting, freezes the
    /// operand layouts and algorithm, and computes the exact output
    /// requirement. The returned plan is immutable; [`Tensor::matmul_into`]
    /// rejects any operand or destination that deviates from it.
    pub fn matmul_requirements(&self, rhs: &Tensor) -> Result<MatmulRequirements, &'static str> {
        if self.dtype() != rhs.dtype() {
            return Err("mixed dtypes");
        }
        let output_shape = matmul_output_shape(self.shape(), rhs.shape())?;
        let rank = output_shape.len();
        let batch = checked_product(&output_shape[..rank - 2])?;
        let algorithm = match self.dtype() {
            DType::F32 | DType::F64 | DType::U8 | DType::U32 | DType::I64 => MatmulAlgorithm::Naive,
            DType::F16 => return Err("f16 matmul is not supported on the CPU backend"),
            DType::BF16 => return Err("bf16 matmul is not supported on the CPU backend"),
        };
        Ok(MatmulRequirements {
            output: checked_requirement(&output_shape, self.dtype())?,
            scratch: Vec::new(),
            algorithm,
            left_layout: self.layout.clone(),
            right_layout: rhs.layout.clone(),
            batch,
            m: output_shape[rank - 2],
            n: output_shape[rank - 1],
            k: self.shape()[self.shape().len() - 1],
        })
    }

    pub fn matmul_output_requirements(
        &self,
        rhs: &Tensor,
    ) -> Result<CpuTensorRequirement, &'static str> {
        Ok(self.matmul_requirements(rhs)?.output)
    }

    pub fn matmul_scratch_requirements(
        &self,
        rhs: &Tensor,
    ) -> Result<Vec<CpuTensorRequirement>, &'static str> {
        Ok(self.matmul_requirements(rhs)?.scratch)
    }

    /// Executes a previously planned matmul into `destination` without
    /// allocating. Fails if inputs, destination, scratch, or the frozen plan
    /// disagree.
    pub fn matmul_into(
        &self,
        rhs: &Tensor,
        destination: &mut CpuDestination<'_>,
        scratch: &mut [CpuDestination<'_>],
        requirements: &MatmulRequirements,
    ) -> Result<(), String> {
        validate_matmul(self, rhs, destination, scratch, requirements)?;
        match (requirements.algorithm, &self.buffer, &rhs.buffer) {
            (MatmulAlgorithm::Naive, CpuBuffer::F32(a), CpuBuffer::F32(b)) => naive_into(
                a,
                &self.layout,
                b,
                &rhs.layout,
                destination,
                requirements,
                f32::mul_add,
            ),
            (MatmulAlgorithm::Naive, CpuBuffer::F64(a), CpuBuffer::F64(b)) => naive_into(
                a,
                &self.layout,
                b,
                &rhs.layout,
                destination,
                requirements,
                f64::mul_add,
            ),
            (MatmulAlgorithm::Naive, CpuBuffer::U8(a), CpuBuffer::U8(b)) => naive_into(
                a,
                &self.layout,
                b,
                &rhs.layout,
                destination,
                requirements,
                |a, b, c| c + a * b,
            ),
            (MatmulAlgorithm::Naive, CpuBuffer::U32(a), CpuBuffer::U32(b)) => naive_into(
                a,
                &self.layout,
                b,
                &rhs.layout,
                destination,
                requirements,
                |a, b, c| c + a * b,
            ),
            (MatmulAlgorithm::Naive, CpuBuffer::I64(a), CpuBuffer::I64(b)) => naive_into(
                a,
                &self.layout,
                b,
                &rhs.layout,
                destination,
                requirements,
                |a, b, c| c + a * b,
            ),
            _ => Err("matmul: buffers do not match exact requirements".into()),
        }
    }

    /// Allocating wrapper: plan, allocate, execute. Panics on invalid
    /// operands.
    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        self.try_matmul(rhs)
            .unwrap_or_else(|message| panic!("{message}"))
    }

    /// Fallible allocating wrapper around [`Tensor::matmul`].
    pub fn try_matmul(&self, rhs: &Tensor) -> Result<Tensor, &'static str> {
        let requirements = self.matmul_requirements(rhs)?;
        let mut output = Tensor::empty(&requirements.output.shape, requirements.output.dtype);
        let result = self.matmul_into(
            rhs,
            &mut output
                .destination()
                .expect("new CPU tensor storage must be unique"),
            &mut [],
            &requirements,
        );
        if let Err(error) = result {
            panic!("matmul exact execution failed: {error}");
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::ExecutableAllocationGuard;

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
    fn exact_requirements_freeze_algorithm_and_into_matches_wrapper() {
        let a = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], vec![2, 3]);
        let b = Tensor::from_vec(vec![1f32, 0., 0., 1., 1., 1.], vec![3, 2]);
        let requirements = a.matmul_requirements(&b).unwrap();
        assert_eq!(requirements.algorithm, MatmulAlgorithm::Naive);
        assert_eq!(requirements.output.shape, vec![2, 2]);
        assert_eq!(requirements.output.dtype, DType::F32);
        assert_eq!(requirements.output.bytes, 4 * size_of::<f32>());
        assert_eq!(requirements.batch, 1);
        assert_eq!((requirements.m, requirements.n, requirements.k), (2, 2, 3));
        assert!(requirements.scratch.is_empty());
        assert_eq!(requirements.scratch_bytes(), 0);

        let mut output = Tensor::empty(&requirements.output.shape, requirements.output.dtype);
        a.matmul_into(
            &b,
            &mut output.destination().unwrap(),
            &mut [],
            &requirements,
        )
        .unwrap();
        let wrapped = a.matmul(&b);
        let (CpuBuffer::F32(actual), CpuBuffer::F32(expected)) = (&output.buffer, &wrapped.buffer)
        else {
            panic!()
        };
        assert_eq!(actual, expected);
    }

    #[test]
    fn allocation_free_batched_broadcast_into_under_guard() {
        let a = Tensor::from_vec(
            (1..=12).map(|value| value as f32).collect(),
            vec![2, 1, 2, 3],
        );
        let b = Tensor::from_vec(
            vec![1f32, 0., 0., 1., 1., 1., 2., 0., 0., 2., 2., 2.],
            vec![1, 2, 3, 2],
        );
        let requirements = a.matmul_requirements(&b).unwrap();
        assert_eq!(requirements.output.shape, vec![2, 2, 2, 2]);
        assert_eq!(requirements.batch, 4);
        let mut output = Tensor::empty(&requirements.output.shape, requirements.output.dtype);
        let mut destination = output.destination().unwrap();
        {
            let _guard = ExecutableAllocationGuard::enter();
            a.matmul_into(&b, &mut destination, &mut [], &requirements)
                .unwrap();
        }
        let wrapped = a.matmul(&b);
        let (CpuBuffer::F32(actual), CpuBuffer::F32(expected)) = (&output.buffer, &wrapped.buffer)
        else {
            panic!()
        };
        assert_eq!(actual, expected);
    }

    #[test]
    fn matmul_consumes_strided_layouts_and_preserves_integer_dtype() {
        let a = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6], vec![2, 3]);
        let a = a.view(a.layout.permute(&[1, 0]));
        let b = Tensor::from_vec(vec![1u32, 2, 3, 4], vec![2, 2]);
        let requirements = a.matmul_requirements(&b).unwrap();
        assert_eq!(requirements.algorithm, MatmulAlgorithm::Naive);
        assert_eq!(requirements.output.dtype, DType::U32);
        let output = a.matmul(&b);
        let CpuBuffer::U32(values) = &output.buffer else {
            panic!()
        };
        assert_eq!(values.as_slice(), &[13, 18, 17, 24, 21, 30]);
    }

    #[test]
    fn into_rejects_layout_dtype_and_requirement_mismatches() {
        let a = Tensor::from_vec(vec![1f32, 2., 3., 4.], vec![2, 2]);
        let b = Tensor::from_vec(vec![1f32, 0., 0., 1.], vec![2, 2]);
        let requirements = a.matmul_requirements(&b).unwrap();
        let mut wrong_dtype = Tensor::empty(&requirements.output.shape, DType::F64);
        assert!(a
            .matmul_into(
                &b,
                &mut wrong_dtype.destination().unwrap(),
                &mut [],
                &requirements,
            )
            .is_err());

        let transposed = a.view(a.layout.permute(&[1, 0]));
        let mut output = Tensor::empty(&requirements.output.shape, requirements.output.dtype);
        assert!(transposed
            .matmul_into(
                &b,
                &mut output.destination().unwrap(),
                &mut [],
                &requirements,
            )
            .is_err());
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

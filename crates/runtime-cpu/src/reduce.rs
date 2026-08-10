use super::tensor::{
    CpuBuffer, CpuDestination, CpuOperationRequirements, CpuTensorRequirement, Elem, Tensor,
};
use effect_torch_runtime::{DType, Layout};
use half::{bf16, f16};

fn validate_dims(shape: &[usize], dims: &[usize], operation: &str) {
    for &dimension in dims {
        assert!(
            dimension < shape.len(),
            "{operation}: reduction dimension {dimension} is out of bounds for rank {}",
            shape.len()
        );
    }
}

fn kept_shape(shape: &[usize], dims: &[usize]) -> Vec<usize> {
    validate_dims(shape, dims, "reduce");
    let mut output = shape.to_vec();
    for &dimension in dims {
        output[dimension] = 1;
    }
    output
}

fn validate_reduce_destination<T: Elem>(
    operation: &str,
    layout: &Layout,
    dims: &[usize],
    destination: &CpuDestination<'_>,
) -> Result<(), String> {
    validate_dims(layout.shape(), dims, operation);
    destination.validate_current::<T>(operation)?;
    if destination.shape().len() != layout.shape().len() {
        return Err(format!(
            "{operation} destination rank mismatch: expected {}, got {}",
            layout.shape().len(),
            destination.shape().len()
        ));
    }
    for (dimension, (&input, &output)) in layout.shape().iter().zip(destination.shape()).enumerate()
    {
        let expected = if dims.contains(&dimension) { 1 } else { input };
        if output != expected {
            return Err(format!(
                "{operation} destination shape mismatch at dimension {dimension}: expected {expected}, got {output}"
            ));
        }
    }
    Ok(())
}

fn reduce_into_impl<T: Elem>(
    operation: &str,
    source: &[T],
    layout: &Layout,
    dims: &[usize],
    init: T,
    reduce: impl Fn(T, T) -> T,
    finish: impl Fn(T) -> T,
    destination: &mut CpuDestination<'_>,
) -> Result<(), String> {
    validate_reduce_destination::<T>(operation, layout, dims, destination)?;
    let reduced_elements = dims
        .iter()
        .try_fold(1usize, |total, &dimension| {
            total.checked_mul(layout.shape()[dimension])
        })
        .ok_or_else(|| format!("{operation} reduction element count overflow"))?;
    destination.write_current::<T, _>(operation, |output| {
        for (output_index, output_value) in output.iter_mut().enumerate() {
            let mut base = layout.offset();
            let mut remainder = output_index;
            for dimension in (0..layout.shape().len()).rev() {
                let width = if dims.contains(&dimension) {
                    1
                } else {
                    layout.shape()[dimension].max(1)
                };
                let coordinate = remainder % width;
                remainder /= width;
                base += coordinate * layout.strides()[dimension];
            }

            let mut accumulator = init;
            for reduced_index in 0..reduced_elements {
                let mut source_index = base;
                let mut reduced_remainder = reduced_index;
                for &dimension in dims.iter().rev() {
                    let width = layout.shape()[dimension].max(1);
                    let coordinate = reduced_remainder % width;
                    reduced_remainder /= width;
                    source_index += coordinate * layout.strides()[dimension];
                }
                accumulator = reduce(accumulator, source[source_index]);
            }
            *output_value = finish(accumulator);
        }
    })
}

fn argreduce_into_impl<T: Elem + PartialOrd>(
    operation: &str,
    source: &[T],
    layout: &Layout,
    dimension: usize,
    pick_max: bool,
    destination: &mut CpuDestination<'_>,
) -> Result<(), String> {
    validate_reduce_destination::<u32>(operation, layout, &[dimension], destination)?;
    let reduced_len = layout.shape()[dimension];
    let reduced_stride = layout.strides()[dimension];
    destination.write_current::<u32, _>(operation, |output| {
        for (output_index, output_value) in output.iter_mut().enumerate() {
            let mut base = layout.offset();
            let mut remainder = output_index;
            for current in (0..layout.shape().len()).rev() {
                let width = if current == dimension {
                    1
                } else {
                    layout.shape()[current].max(1)
                };
                let coordinate = remainder % width;
                remainder /= width;
                base += coordinate * layout.strides()[current];
            }
            let mut best_index = 0usize;
            let mut best_value = source[base];
            for index in 1..reduced_len {
                let value = source[base + index * reduced_stride];
                let better = if pick_max {
                    value > best_value
                } else {
                    value < best_value
                };
                if better {
                    best_index = index;
                    best_value = value;
                }
            }
            *output_value = best_index as u32;
        }
    })
}

fn contiguous_stride(shape: &[usize], dimension: usize) -> usize {
    shape[dimension + 1..].iter().product()
}

fn cumsum_into_impl<T: Elem + std::ops::Add<Output = T>>(
    source: &[T],
    layout: &Layout,
    dimension: usize,
    destination: &mut CpuDestination<'_>,
) -> Result<(), String> {
    if dimension >= layout.shape().len() {
        return Err(format!(
            "cumsum dimension {dimension} is out of bounds for rank {}",
            layout.shape().len()
        ));
    }
    if destination.shape() != layout.shape() {
        return Err(format!(
            "cumsum destination shape mismatch: expected {:?}, got {:?}",
            layout.shape(),
            destination.shape()
        ));
    }
    destination.validate_current::<T>("cumsum")?;
    let reduced_len = layout.shape()[dimension];
    let groups = layout
        .shape()
        .iter()
        .enumerate()
        .filter(|(current, _)| *current != dimension)
        .try_fold(1usize, |total, (_, &width)| total.checked_mul(width))
        .ok_or_else(|| "cumsum group count overflow".to_string())?;
    destination.write_current::<T, _>("cumsum", |output| {
        for group in 0..groups {
            let mut source_base = layout.offset();
            let mut output_base = 0usize;
            let mut remainder = group;
            for current in (0..layout.shape().len()).rev() {
                if current == dimension {
                    continue;
                }
                let width = layout.shape()[current].max(1);
                let coordinate = remainder % width;
                remainder /= width;
                source_base += coordinate * layout.strides()[current];
                output_base += coordinate * contiguous_stride(layout.shape(), current);
            }
            let mut accumulator = T::default();
            let output_stride = contiguous_stride(layout.shape(), dimension);
            for index in 0..reduced_len {
                accumulator =
                    accumulator + source[source_base + index * layout.strides()[dimension]];
                output[output_base + index * output_stride] = accumulator;
            }
        }
    })
}

fn requirements(tensor: &Tensor, dims: &[usize], dtype: DType) -> CpuOperationRequirements {
    CpuOperationRequirements::without_scratch(&kept_shape(tensor.shape(), dims), dtype)
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
        write(&mut destination).expect("new CPU tensor must satisfy reduction requirements");
    }
    output
}

macro_rules! basic_requirements {
    ($requirements:ident, $output:ident, $scratch:ident) => {
        pub fn $requirements(&self, dims: &[usize]) -> CpuOperationRequirements {
            requirements(self, dims, self.dtype())
        }

        pub fn $output(&self, dims: &[usize]) -> CpuTensorRequirement {
            self.$requirements(dims).output
        }

        pub fn $scratch(&self, _dims: &[usize]) -> &'static [CpuTensorRequirement] {
            &[]
        }
    };
}

impl Tensor {
    basic_requirements!(
        sum_requirements,
        sum_output_requirements,
        sum_scratch_requirements
    );

    pub fn sum_into(
        &self,
        dims: &[usize],
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        macro_rules! sum {
            ($values:expr, $type:ty) => {
                reduce_into_impl(
                    "sum",
                    $values,
                    &self.layout,
                    dims,
                    <$type>::default(),
                    |a, b| a + b,
                    |value| value,
                    destination,
                )
            };
        }
        match &self.buffer {
            CpuBuffer::F32(values) => sum!(values, f32),
            CpuBuffer::F64(values) => sum!(values, f64),
            CpuBuffer::F16(values) => sum!(values, f16),
            CpuBuffer::BF16(values) => sum!(values, bf16),
            CpuBuffer::U8(values) => sum!(values, u8),
            CpuBuffer::U32(values) => sum!(values, u32),
            CpuBuffer::I64(values) => sum!(values, i64),
        }
    }

    pub fn sum(&self, dims: &[usize]) -> Tensor {
        allocate_output(self.sum_requirements(dims), |destination| {
            self.sum_into(dims, destination)
        })
    }

    basic_requirements!(
        prod_requirements,
        prod_output_requirements,
        prod_scratch_requirements
    );

    pub fn prod_into(
        &self,
        dims: &[usize],
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        macro_rules! product {
            ($values:expr, $one:expr) => {
                reduce_into_impl(
                    "prod",
                    $values,
                    &self.layout,
                    dims,
                    $one,
                    |a, b| a * b,
                    |value| value,
                    destination,
                )
            };
        }
        match &self.buffer {
            CpuBuffer::F32(values) => product!(values, 1f32),
            CpuBuffer::F64(values) => product!(values, 1f64),
            CpuBuffer::F16(values) => product!(values, f16::ONE),
            CpuBuffer::BF16(values) => product!(values, bf16::ONE),
            CpuBuffer::U8(values) => product!(values, 1u8),
            CpuBuffer::U32(values) => product!(values, 1u32),
            CpuBuffer::I64(values) => product!(values, 1i64),
        }
    }

    pub fn prod(&self, dims: &[usize]) -> Tensor {
        allocate_output(self.prod_requirements(dims), |destination| {
            self.prod_into(dims, destination)
        })
    }

    basic_requirements!(
        max_requirements,
        max_output_requirements,
        max_scratch_requirements
    );

    pub fn max_into(
        &self,
        dims: &[usize],
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        macro_rules! maximum {
            ($values:expr, $init:expr, $type:ty) => {
                reduce_into_impl(
                    "max",
                    $values,
                    &self.layout,
                    dims,
                    $init,
                    |a: $type, b: $type| if a >= b { a } else { b },
                    |value| value,
                    destination,
                )
            };
        }
        match &self.buffer {
            CpuBuffer::F32(values) => maximum!(values, f32::NEG_INFINITY, f32),
            CpuBuffer::F64(values) => maximum!(values, f64::NEG_INFINITY, f64),
            CpuBuffer::F16(values) => maximum!(values, f16::NEG_INFINITY, f16),
            CpuBuffer::BF16(values) => maximum!(values, bf16::NEG_INFINITY, bf16),
            CpuBuffer::U8(values) => maximum!(values, u8::MIN, u8),
            CpuBuffer::U32(values) => maximum!(values, u32::MIN, u32),
            CpuBuffer::I64(values) => maximum!(values, i64::MIN, i64),
        }
    }

    pub fn max(&self, dims: &[usize]) -> Tensor {
        allocate_output(self.max_requirements(dims), |destination| {
            self.max_into(dims, destination)
        })
    }

    basic_requirements!(
        min_requirements,
        min_output_requirements,
        min_scratch_requirements
    );

    pub fn min_into(
        &self,
        dims: &[usize],
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        macro_rules! minimum {
            ($values:expr, $init:expr, $type:ty) => {
                reduce_into_impl(
                    "min",
                    $values,
                    &self.layout,
                    dims,
                    $init,
                    |a: $type, b: $type| if a <= b { a } else { b },
                    |value| value,
                    destination,
                )
            };
        }
        match &self.buffer {
            CpuBuffer::F32(values) => minimum!(values, f32::INFINITY, f32),
            CpuBuffer::F64(values) => minimum!(values, f64::INFINITY, f64),
            CpuBuffer::F16(values) => minimum!(values, f16::INFINITY, f16),
            CpuBuffer::BF16(values) => minimum!(values, bf16::INFINITY, bf16),
            CpuBuffer::U8(values) => minimum!(values, u8::MAX, u8),
            CpuBuffer::U32(values) => minimum!(values, u32::MAX, u32),
            CpuBuffer::I64(values) => minimum!(values, i64::MAX, i64),
        }
    }

    pub fn min(&self, dims: &[usize]) -> Tensor {
        allocate_output(self.min_requirements(dims), |destination| {
            self.min_into(dims, destination)
        })
    }

    pub fn mean_requirements(&self, dims: &[usize]) -> CpuOperationRequirements {
        assert!(self.dtype().is_float(), "mean requires a float dtype");
        requirements(self, dims, self.dtype())
    }

    pub fn mean_output_requirements(&self, dims: &[usize]) -> CpuTensorRequirement {
        self.mean_requirements(dims).output
    }

    pub fn mean_scratch_requirements(&self, _dims: &[usize]) -> &'static [CpuTensorRequirement] {
        &[]
    }

    pub fn mean_into(
        &self,
        dims: &[usize],
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        assert!(self.dtype().is_float(), "mean requires a float dtype");
        let count = dims.iter().try_fold(1usize, |total, &dimension| {
            self.shape()
                .get(dimension)
                .and_then(|width| total.checked_mul(*width))
        });
        let count = count.ok_or_else(|| "mean reduction element count overflow".to_string())?;
        match &self.buffer {
            CpuBuffer::F32(values) => reduce_into_impl(
                "mean",
                values,
                &self.layout,
                dims,
                0f32,
                |a, b| a + b,
                |value| value / count as f32,
                destination,
            ),
            CpuBuffer::F64(values) => reduce_into_impl(
                "mean",
                values,
                &self.layout,
                dims,
                0f64,
                |a, b| a + b,
                |value| value / count as f64,
                destination,
            ),
            CpuBuffer::F16(values) => reduce_into_impl(
                "mean",
                values,
                &self.layout,
                dims,
                f16::ZERO,
                |a, b| a + b,
                |value| value / f16::from_f64(count as f64),
                destination,
            ),
            CpuBuffer::BF16(values) => reduce_into_impl(
                "mean",
                values,
                &self.layout,
                dims,
                bf16::ZERO,
                |a, b| a + b,
                |value| value / bf16::from_f64(count as f64),
                destination,
            ),
            _ => unreachable!("float dtype checked before dispatch"),
        }
    }

    pub fn mean(&self, dims: &[usize]) -> Tensor {
        allocate_output(self.mean_requirements(dims), |destination| {
            self.mean_into(dims, destination)
        })
    }

    pub fn argmax_requirements(&self, dimension: usize) -> CpuOperationRequirements {
        requirements(self, &[dimension], DType::U32)
    }

    pub fn argmax_output_requirements(&self, dimension: usize) -> CpuTensorRequirement {
        self.argmax_requirements(dimension).output
    }

    pub fn argmax_scratch_requirements(
        &self,
        _dimension: usize,
    ) -> &'static [CpuTensorRequirement] {
        &[]
    }

    pub fn argmax_into(
        &self,
        dimension: usize,
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        macro_rules! argmax {
            ($values:expr) => {
                argreduce_into_impl(
                    "argmax",
                    $values,
                    &self.layout,
                    dimension,
                    true,
                    destination,
                )
            };
        }
        match &self.buffer {
            CpuBuffer::F32(values) => argmax!(values),
            CpuBuffer::F64(values) => argmax!(values),
            CpuBuffer::F16(values) => argmax!(values),
            CpuBuffer::BF16(values) => argmax!(values),
            CpuBuffer::U8(values) => argmax!(values),
            CpuBuffer::U32(values) => argmax!(values),
            CpuBuffer::I64(values) => argmax!(values),
        }
    }

    pub fn argmax(&self, dimension: usize) -> Tensor {
        allocate_output(self.argmax_requirements(dimension), |destination| {
            self.argmax_into(dimension, destination)
        })
    }

    pub fn argmin_requirements(&self, dimension: usize) -> CpuOperationRequirements {
        requirements(self, &[dimension], DType::U32)
    }

    pub fn argmin_output_requirements(&self, dimension: usize) -> CpuTensorRequirement {
        self.argmin_requirements(dimension).output
    }

    pub fn argmin_scratch_requirements(
        &self,
        _dimension: usize,
    ) -> &'static [CpuTensorRequirement] {
        &[]
    }

    pub fn argmin_into(
        &self,
        dimension: usize,
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        macro_rules! argmin {
            ($values:expr) => {
                argreduce_into_impl(
                    "argmin",
                    $values,
                    &self.layout,
                    dimension,
                    false,
                    destination,
                )
            };
        }
        match &self.buffer {
            CpuBuffer::F32(values) => argmin!(values),
            CpuBuffer::F64(values) => argmin!(values),
            CpuBuffer::F16(values) => argmin!(values),
            CpuBuffer::BF16(values) => argmin!(values),
            CpuBuffer::U8(values) => argmin!(values),
            CpuBuffer::U32(values) => argmin!(values),
            CpuBuffer::I64(values) => argmin!(values),
        }
    }

    pub fn argmin(&self, dimension: usize) -> Tensor {
        allocate_output(self.argmin_requirements(dimension), |destination| {
            self.argmin_into(dimension, destination)
        })
    }

    pub fn cumsum_requirements(&self, dimension: usize) -> CpuOperationRequirements {
        assert!(
            dimension < self.shape().len(),
            "cumsum dimension out of bounds"
        );
        CpuOperationRequirements::without_scratch(self.shape(), self.dtype())
    }

    pub fn cumsum_output_requirements(&self, dimension: usize) -> CpuTensorRequirement {
        self.cumsum_requirements(dimension).output
    }

    pub fn cumsum_scratch_requirements(
        &self,
        _dimension: usize,
    ) -> &'static [CpuTensorRequirement] {
        &[]
    }

    pub fn cumsum_into(
        &self,
        dimension: usize,
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        match &self.buffer {
            CpuBuffer::F32(values) => {
                cumsum_into_impl(values, &self.layout, dimension, destination)
            }
            CpuBuffer::F64(values) => {
                cumsum_into_impl(values, &self.layout, dimension, destination)
            }
            CpuBuffer::F16(values) => {
                cumsum_into_impl(values, &self.layout, dimension, destination)
            }
            CpuBuffer::BF16(values) => {
                cumsum_into_impl(values, &self.layout, dimension, destination)
            }
            CpuBuffer::U8(values) => cumsum_into_impl(values, &self.layout, dimension, destination),
            CpuBuffer::U32(values) => {
                cumsum_into_impl(values, &self.layout, dimension, destination)
            }
            CpuBuffer::I64(values) => {
                cumsum_into_impl(values, &self.layout, dimension, destination)
            }
        }
    }

    pub fn cumsum(&self, dimension: usize) -> Tensor {
        allocate_output(self.cumsum_requirements(dimension), |destination| {
            self.cumsum_into(dimension, destination)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::ExecutableAllocationGuard;

    fn f32_data(tensor: &Tensor) -> Vec<f32> {
        let CpuBuffer::F32(values) = &tensor.buffer else {
            panic!()
        };
        values.as_slice().to_vec()
    }

    #[test]
    fn sum_dims() {
        let tensor = Tensor::from_vec((0..6).map(|x| x as f32).collect(), vec![2, 3]);
        let sum = tensor.sum(&[1]);
        assert_eq!(sum.shape(), &[2, 1]);
        assert_eq!(f32_data(&sum), vec![3., 12.]);
        let sum_zero = tensor.sum(&[0]);
        assert_eq!(f32_data(&sum_zero), vec![3., 5., 7.]);
        let mean = tensor.mean(&[0, 1]);
        assert_eq!(f32_data(&mean), vec![2.5]);
    }

    #[test]
    fn sum_strided() {
        let tensor = Tensor::from_vec((0..6).map(|x| x as f32).collect(), vec![2, 3]);
        let transposed = tensor.view(tensor.layout.permute(&[1, 0]));
        let sum = transposed.sum(&[0]);
        assert_eq!(f32_data(&sum), vec![3., 12.]);
    }

    #[test]
    fn argmax_cumsum() {
        let tensor = Tensor::from_vec(vec![1f32, 5., 2., 0., 4., 4.], vec![2, 3]);
        let argmax = tensor.argmax(1);
        let CpuBuffer::U32(values) = &argmax.buffer else {
            panic!()
        };
        assert_eq!(values.as_slice(), &[1, 1]);
        let cumulative = tensor.cumsum(1);
        assert_eq!(f32_data(&cumulative), vec![1., 6., 8., 0., 4., 8.]);
    }

    #[test]
    fn exact_requirements_and_allocation_free_into_parity() {
        let tensor = Tensor::from_vec((0..12).map(|value| value as f32).collect(), vec![3, 4]);
        let requirements = tensor.sum_requirements(&[1]);
        assert_eq!(requirements.output.shape, vec![3, 1]);
        assert_eq!(requirements.output.bytes, 12);
        assert_eq!(requirements.scratch_bytes(), 0);
        let mut output = Tensor::empty(&requirements.output.shape, requirements.output.dtype);
        {
            let _guard = ExecutableAllocationGuard::enter();
            tensor
                .sum_into(&[1], &mut output.destination().unwrap())
                .unwrap();
        }
        assert_eq!(f32_data(&output), f32_data(&tensor.sum(&[1])));
    }
}

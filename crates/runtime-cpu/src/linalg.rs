use super::tensor::{source_index, CpuBuffer, CpuDestination, CpuTensorRequirement, Elem, Tensor};
use effect_torch_runtime::{DType, Layout};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinalgAlgorithm {
    LuPartialPivotF64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeterminantRequirements {
    pub output: CpuTensorRequirement,
    pub scratch: Vec<CpuTensorRequirement>,
    pub algorithm: LinalgAlgorithm,
    pub matrix_layout: Layout,
    pub batch: usize,
    pub n: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InverseRequirements {
    pub output: CpuTensorRequirement,
    pub scratch: Vec<CpuTensorRequirement>,
    pub algorithm: LinalgAlgorithm,
    pub matrix_layout: Layout,
    pub batch: usize,
    pub n: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolveRequirements {
    pub output: CpuTensorRequirement,
    pub scratch: Vec<CpuTensorRequirement>,
    pub algorithm: LinalgAlgorithm,
    pub matrix_layout: Layout,
    pub rhs_layout: Layout,
    pub batch: usize,
    pub n: usize,
    pub rhs_columns: usize,
}

macro_rules! scratch_bytes {
    ($type:ty) => {
        impl $type {
            pub fn scratch_bytes(&self) -> usize {
                self.scratch
                    .iter()
                    .try_fold(0usize, |total, requirement| {
                        total.checked_add(requirement.bytes)
                    })
                    .expect("linalg scratch byte size overflow")
            }
        }
    };
}

scratch_bytes!(DeterminantRequirements);
scratch_bytes!(InverseRequirements);
scratch_bytes!(SolveRequirements);

fn checked_product(values: &[usize]) -> Result<usize, &'static str> {
    values
        .iter()
        .try_fold(1usize, |total, &value| total.checked_mul(value))
        .ok_or("linalg element count overflow")
}

fn checked_requirement(
    shape: &[usize],
    dtype: DType,
) -> Result<CpuTensorRequirement, &'static str> {
    checked_product(shape)?
        .checked_mul(dtype.size_in_bytes())
        .ok_or("linalg byte size overflow")?;
    Ok(CpuTensorRequirement::new(shape, dtype))
}

fn square_dimensions(tensor: &Tensor) -> Result<(usize, usize), &'static str> {
    if tensor.shape().len() < 2 {
        return Err("linalg requires rank >= 2");
    }
    let rank = tensor.shape().len();
    let n = tensor.shape()[rank - 1];
    if tensor.shape()[rank - 2] != n {
        return Err("linalg requires square matrices");
    }
    Ok((n, checked_product(&tensor.shape()[..rank - 2])?))
}

fn lu_decompose_in_place(
    lu: &mut [f64],
    n: usize,
    work: &mut [f64],
    work_columns: usize,
) -> Option<i32> {
    let mut sign = 1i32;
    for column in 0..n {
        let mut pivot = column;
        let mut best = lu[column * n + column].abs();
        for row in column + 1..n {
            let value = lu[row * n + column].abs();
            if value > best {
                best = value;
                pivot = row;
            }
        }
        if best < 1e-300 {
            return None;
        }
        if pivot != column {
            for index in 0..n {
                lu.swap(column * n + index, pivot * n + index);
            }
            for index in 0..work_columns {
                work.swap(column * work_columns + index, pivot * work_columns + index);
            }
            sign = -sign;
        }
        let diagonal = lu[column * n + column];
        for row in column + 1..n {
            let factor = lu[row * n + column] / diagonal;
            lu[row * n + column] = factor;
            for index in column + 1..n {
                lu[row * n + index] -= factor * lu[column * n + index];
            }
        }
    }
    Some(sign)
}

fn lu_solve_in_place(lu: &[f64], work: &mut [f64], n: usize, columns: usize) {
    for row in 1..n {
        for previous in 0..row {
            let factor = lu[row * n + previous];
            for column in 0..columns {
                work[row * columns + column] -= factor * work[previous * columns + column];
            }
        }
    }
    for row in (0..n).rev() {
        for following in row + 1..n {
            let factor = lu[row * n + following];
            for column in 0..columns {
                work[row * columns + column] -= factor * work[following * columns + column];
            }
        }
        let diagonal = lu[row * n + row];
        for column in 0..columns {
            work[row * columns + column] /= diagonal;
        }
    }
}

fn copy_matrix<T: Elem>(source: &[T], layout: &Layout, batch: usize, n: usize, lu: &mut [f64]) {
    let base = batch * n * n;
    for (index, value) in lu.iter_mut().enumerate() {
        *value = source[source_index(layout, base + index)].to_f64();
    }
}

fn validate_scratch(
    operation: &str,
    expected: &[CpuTensorRequirement],
    actual: &[CpuDestination<'_>],
) -> Result<(), String> {
    if expected.len() != actual.len() {
        return Err(format!(
            "{operation}: expected {} scratch tensors, got {}",
            expected.len(),
            actual.len()
        ));
    }
    for (requirement, destination) in expected.iter().zip(actual) {
        if destination.shape() != requirement.shape || destination.dtype() != requirement.dtype {
            return Err(format!(
                "{operation}: scratch does not match exact requirements"
            ));
        }
        destination.validate_current::<f64>(operation)?;
    }
    Ok(())
}

fn validate_matrix_exact(
    operation: &str,
    matrix: &Tensor,
    expected_layout: &Layout,
    expected_n: usize,
    expected_batch: usize,
) -> Result<(), String> {
    if &matrix.layout != expected_layout {
        return Err(format!(
            "{operation}: matrix layout does not match exact requirements"
        ));
    }
    let (n, batch) = square_dimensions(matrix).map_err(str::to_string)?;
    if n != expected_n || batch != expected_batch {
        return Err(format!(
            "{operation}: matrix dimensions do not match exact requirements"
        ));
    }
    Ok(())
}

fn determinant_typed<T: Elem>(
    source: &[T],
    layout: &Layout,
    destination: &mut CpuDestination<'_>,
    scratch: &mut [CpuDestination<'_>],
    requirements: &DeterminantRequirements,
) -> Result<(), String> {
    let lu_destination = &mut scratch[0];
    lu_destination.write::<f64, _>(
        "det",
        &requirements.scratch[0].shape,
        |lu| -> Result<(), String> {
            destination.write::<T, _>(
                "det",
                &requirements.output.shape,
                |output| -> Result<(), String> {
                    for (batch, output_value) in output.iter_mut().enumerate() {
                        copy_matrix(source, layout, batch, requirements.n, lu);
                        let determinant =
                            match lu_decompose_in_place(lu, requirements.n, &mut [], 0) {
                                Some(sign) => {
                                    let mut value = sign as f64;
                                    for index in 0..requirements.n {
                                        value *= lu[index * requirements.n + index];
                                    }
                                    value
                                }
                                None => 0.0,
                            };
                        *output_value = T::from_f64(determinant);
                    }
                    Ok(())
                },
            )?
        },
    )?
}

fn inverse_typed<T: Elem>(
    source: &[T],
    layout: &Layout,
    destination: &mut CpuDestination<'_>,
    scratch: &mut [CpuDestination<'_>],
    requirements: &InverseRequirements,
) -> Result<(), String> {
    let (lu_destination, work_destination) = scratch.split_at_mut(1);
    lu_destination[0].write::<f64, _>(
        "inverse",
        &requirements.scratch[0].shape,
        |lu| -> Result<(), String> {
            work_destination[0].write::<f64, _>(
                "inverse",
                &requirements.scratch[1].shape,
                |work| -> Result<(), String> {
                    destination.write::<T, _>(
                        "inverse",
                        &requirements.output.shape,
                        |output| -> Result<(), String> {
                            for batch in 0..requirements.batch {
                                copy_matrix(source, layout, batch, requirements.n, lu);
                                work.fill(0.0);
                                for index in 0..requirements.n {
                                    work[index * requirements.n + index] = 1.0;
                                }
                                lu_decompose_in_place(lu, requirements.n, work, requirements.n)
                                    .ok_or_else(|| "matrix is singular".to_string())?;
                                lu_solve_in_place(lu, work, requirements.n, requirements.n);
                                let output_base = batch * requirements.n * requirements.n;
                                for (index, &value) in work.iter().enumerate() {
                                    output[output_base + index] = T::from_f64(value);
                                }
                            }
                            Ok(())
                        },
                    )?
                },
            )?
        },
    )?
}

#[allow(clippy::too_many_arguments)]
fn solve_typed<A: Elem, R: Elem>(
    matrix: &[A],
    matrix_layout: &Layout,
    rhs: &[R],
    rhs_layout: &Layout,
    destination: &mut CpuDestination<'_>,
    scratch: &mut [CpuDestination<'_>],
    requirements: &SolveRequirements,
) -> Result<(), String> {
    let (lu_destination, work_destination) = scratch.split_at_mut(1);
    lu_destination[0].write::<f64, _>(
        "solve",
        &requirements.scratch[0].shape,
        |lu| -> Result<(), String> {
            work_destination[0].write::<f64, _>(
                "solve",
                &requirements.scratch[1].shape,
                |work| -> Result<(), String> {
                    destination.write::<A, _>(
                        "solve",
                        &requirements.output.shape,
                        |output| -> Result<(), String> {
                            for batch in 0..requirements.batch {
                                copy_matrix(matrix, matrix_layout, batch, requirements.n, lu);
                                let rhs_base = batch * requirements.n * requirements.rhs_columns;
                                for (index, value) in work.iter_mut().enumerate() {
                                    *value =
                                        rhs[source_index(rhs_layout, rhs_base + index)].to_f64();
                                }
                                lu_decompose_in_place(
                                    lu,
                                    requirements.n,
                                    work,
                                    requirements.rhs_columns,
                                )
                                .ok_or_else(|| "matrix is singular".to_string())?;
                                lu_solve_in_place(
                                    lu,
                                    work,
                                    requirements.n,
                                    requirements.rhs_columns,
                                );
                                for (index, &value) in work.iter().enumerate() {
                                    output[rhs_base + index] = A::from_f64(value);
                                }
                            }
                            Ok(())
                        },
                    )?
                },
            )?
        },
    )?
}

fn solve_rhs_typed<A: Elem>(
    matrix: &[A],
    matrix_layout: &Layout,
    rhs: &Tensor,
    destination: &mut CpuDestination<'_>,
    scratch: &mut [CpuDestination<'_>],
    requirements: &SolveRequirements,
) -> Result<(), String> {
    macro_rules! solve_rhs {
        ($values:expr, $type:ty) => {
            solve_typed::<A, $type>(
                matrix,
                matrix_layout,
                $values,
                &rhs.layout,
                destination,
                scratch,
                requirements,
            )
        };
    }
    match &rhs.buffer {
        CpuBuffer::F32(values) => solve_rhs!(values, f32),
        CpuBuffer::F64(values) => solve_rhs!(values, f64),
        CpuBuffer::F16(values) => solve_rhs!(values, half::f16),
        CpuBuffer::BF16(values) => solve_rhs!(values, half::bf16),
        CpuBuffer::U8(values) => solve_rhs!(values, u8),
        CpuBuffer::U32(values) => solve_rhs!(values, u32),
        CpuBuffer::I64(values) => solve_rhs!(values, i64),
    }
}

impl Tensor {
    pub fn det_requirements(&self) -> Result<DeterminantRequirements, &'static str> {
        let (n, batch) = square_dimensions(self)?;
        let rank = self.shape().len();
        let output_shape = if rank == 2 {
            vec![1]
        } else {
            self.shape()[..rank - 2].to_vec()
        };
        Ok(DeterminantRequirements {
            output: checked_requirement(&output_shape, self.dtype())?,
            scratch: vec![checked_requirement(&[n, n], DType::F64)?],
            algorithm: LinalgAlgorithm::LuPartialPivotF64,
            matrix_layout: self.layout.clone(),
            batch,
            n,
        })
    }

    pub fn det_output_requirements(&self) -> Result<CpuTensorRequirement, &'static str> {
        Ok(self.det_requirements()?.output)
    }

    pub fn det_scratch_requirements(&self) -> Result<Vec<CpuTensorRequirement>, &'static str> {
        Ok(self.det_requirements()?.scratch)
    }

    pub fn det_into(
        &self,
        destination: &mut CpuDestination<'_>,
        scratch: &mut [CpuDestination<'_>],
        requirements: &DeterminantRequirements,
    ) -> Result<(), String> {
        validate_matrix_exact(
            "det",
            self,
            &requirements.matrix_layout,
            requirements.n,
            requirements.batch,
        )?;
        if requirements.algorithm != LinalgAlgorithm::LuPartialPivotF64 {
            return Err("det: unsupported exact algorithm".into());
        }
        let rank = self.shape().len();
        let output_shape_matches = if rank == 2 {
            requirements.output.shape.as_slice() == [1]
        } else {
            requirements.output.shape == self.shape()[..rank - 2]
        };
        if !output_shape_matches
            || requirements.output.dtype != self.dtype()
            || destination.shape() != requirements.output.shape
            || destination.dtype() != requirements.output.dtype
        {
            return Err("det: output does not match exact requirements".into());
        }
        validate_scratch("det", &requirements.scratch, scratch)?;
        if requirements.scratch.len() != 1
            || requirements.scratch[0].shape.as_slice() != [requirements.n, requirements.n]
            || requirements.scratch[0].dtype != DType::F64
        {
            return Err("det: invalid exact scratch requirements".into());
        }
        macro_rules! determinant {
            ($values:expr, $type:ty) => {
                determinant_typed::<$type>(
                    $values,
                    &self.layout,
                    destination,
                    scratch,
                    requirements,
                )
            };
        }
        match &self.buffer {
            CpuBuffer::F32(values) => determinant!(values, f32),
            CpuBuffer::F64(values) => determinant!(values, f64),
            CpuBuffer::F16(values) => determinant!(values, half::f16),
            CpuBuffer::BF16(values) => determinant!(values, half::bf16),
            CpuBuffer::U8(values) => determinant!(values, u8),
            CpuBuffer::U32(values) => determinant!(values, u32),
            CpuBuffer::I64(values) => determinant!(values, i64),
        }
    }

    pub fn det(&self) -> Tensor {
        let requirements = self
            .det_requirements()
            .unwrap_or_else(|message| panic!("{message}"));
        let mut output = Tensor::empty(&requirements.output.shape, requirements.output.dtype);
        let mut lu = Tensor::empty(
            &requirements.scratch[0].shape,
            requirements.scratch[0].dtype,
        );
        self.det_into(
            &mut output
                .destination()
                .expect("new CPU tensor storage must be unique"),
            &mut [lu
                .destination()
                .expect("new CPU scratch storage must be unique")],
            &requirements,
        )
        .unwrap_or_else(|message| panic!("{message}"));
        output
    }

    pub fn inverse_requirements(&self) -> Result<InverseRequirements, &'static str> {
        let (n, batch) = square_dimensions(self)?;
        Ok(InverseRequirements {
            output: checked_requirement(self.shape(), self.dtype())?,
            scratch: vec![
                checked_requirement(&[n, n], DType::F64)?,
                checked_requirement(&[n, n], DType::F64)?,
            ],
            algorithm: LinalgAlgorithm::LuPartialPivotF64,
            matrix_layout: self.layout.clone(),
            batch,
            n,
        })
    }

    pub fn inverse_output_requirements(&self) -> Result<CpuTensorRequirement, &'static str> {
        Ok(self.inverse_requirements()?.output)
    }

    pub fn inverse_scratch_requirements(&self) -> Result<Vec<CpuTensorRequirement>, &'static str> {
        Ok(self.inverse_requirements()?.scratch)
    }

    pub fn inverse_into(
        &self,
        destination: &mut CpuDestination<'_>,
        scratch: &mut [CpuDestination<'_>],
        requirements: &InverseRequirements,
    ) -> Result<(), String> {
        validate_matrix_exact(
            "inverse",
            self,
            &requirements.matrix_layout,
            requirements.n,
            requirements.batch,
        )?;
        if requirements.algorithm != LinalgAlgorithm::LuPartialPivotF64
            || requirements.output.shape != self.shape()
            || requirements.output.dtype != self.dtype()
            || destination.shape() != requirements.output.shape
            || destination.dtype() != requirements.output.dtype
        {
            return Err("inverse: output or algorithm does not match exact requirements".into());
        }
        validate_scratch("inverse", &requirements.scratch, scratch)?;
        if requirements.scratch.len() != 2
            || requirements.scratch[0].shape.as_slice() != [requirements.n, requirements.n]
            || requirements.scratch[1].shape.as_slice() != [requirements.n, requirements.n]
            || requirements
                .scratch
                .iter()
                .any(|requirement| requirement.dtype != DType::F64)
        {
            return Err("inverse: invalid exact scratch requirements".into());
        }
        macro_rules! inverse {
            ($values:expr, $type:ty) => {
                inverse_typed::<$type>($values, &self.layout, destination, scratch, requirements)
            };
        }
        match &self.buffer {
            CpuBuffer::F32(values) => inverse!(values, f32),
            CpuBuffer::F64(values) => inverse!(values, f64),
            CpuBuffer::F16(values) => inverse!(values, half::f16),
            CpuBuffer::BF16(values) => inverse!(values, half::bf16),
            CpuBuffer::U8(values) => inverse!(values, u8),
            CpuBuffer::U32(values) => inverse!(values, u32),
            CpuBuffer::I64(values) => inverse!(values, i64),
        }
    }

    pub fn inverse(&self) -> Tensor {
        self.try_inverse()
            .unwrap_or_else(|message| panic!("{message}"))
    }

    pub fn try_inverse(&self) -> Result<Tensor, &'static str> {
        let requirements = self
            .inverse_requirements()
            .unwrap_or_else(|message| panic!("{message}"));
        let mut output = Tensor::empty(&requirements.output.shape, requirements.output.dtype);
        let mut lu = Tensor::empty(
            &requirements.scratch[0].shape,
            requirements.scratch[0].dtype,
        );
        let mut work = Tensor::empty(
            &requirements.scratch[1].shape,
            requirements.scratch[1].dtype,
        );
        let result = self.inverse_into(
            &mut output
                .destination()
                .expect("new CPU tensor storage must be unique"),
            &mut [
                lu.destination()
                    .expect("new CPU scratch storage must be unique"),
                work.destination()
                    .expect("new CPU scratch storage must be unique"),
            ],
            &requirements,
        );
        match result {
            Ok(()) => Ok(output),
            Err(message) if message == "matrix is singular" => Err("matrix is singular"),
            Err(message) => panic!("inverse exact execution failed: {message}"),
        }
    }

    pub fn solve_requirements(&self, rhs: &Tensor) -> Result<SolveRequirements, &'static str> {
        let (n, batch) = square_dimensions(self)?;
        if rhs.shape().len() < 2 {
            return Err("solve rhs requires rank >= 2");
        }
        let rank = rhs.shape().len();
        if rhs.shape()[rank - 2] != n {
            return Err("solve rhs row count mismatch");
        }
        if checked_product(&rhs.shape()[..rank - 2])? != batch {
            return Err("solve batch mismatch");
        }
        let rhs_columns = rhs.shape()[rank - 1];
        Ok(SolveRequirements {
            output: checked_requirement(rhs.shape(), self.dtype())?,
            scratch: vec![
                checked_requirement(&[n, n], DType::F64)?,
                checked_requirement(&[n, rhs_columns], DType::F64)?,
            ],
            algorithm: LinalgAlgorithm::LuPartialPivotF64,
            matrix_layout: self.layout.clone(),
            rhs_layout: rhs.layout.clone(),
            batch,
            n,
            rhs_columns,
        })
    }

    pub fn solve_output_requirements(
        &self,
        rhs: &Tensor,
    ) -> Result<CpuTensorRequirement, &'static str> {
        Ok(self.solve_requirements(rhs)?.output)
    }

    pub fn solve_scratch_requirements(
        &self,
        rhs: &Tensor,
    ) -> Result<Vec<CpuTensorRequirement>, &'static str> {
        Ok(self.solve_requirements(rhs)?.scratch)
    }

    pub fn solve_into(
        &self,
        rhs: &Tensor,
        destination: &mut CpuDestination<'_>,
        scratch: &mut [CpuDestination<'_>],
        requirements: &SolveRequirements,
    ) -> Result<(), String> {
        validate_matrix_exact(
            "solve",
            self,
            &requirements.matrix_layout,
            requirements.n,
            requirements.batch,
        )?;
        if rhs.layout != requirements.rhs_layout {
            return Err("solve: rhs layout does not match exact requirements".into());
        }
        let rhs_rank = rhs.shape().len();
        if rhs_rank < 2
            || rhs.shape()[rhs_rank - 2] != requirements.n
            || rhs.shape()[rhs_rank - 1] != requirements.rhs_columns
            || checked_product(&rhs.shape()[..rhs_rank - 2]).map_err(str::to_string)?
                != requirements.batch
            || requirements.algorithm != LinalgAlgorithm::LuPartialPivotF64
            || requirements.output.shape != rhs.shape()
            || requirements.output.dtype != self.dtype()
            || destination.shape() != requirements.output.shape
            || destination.dtype() != requirements.output.dtype
        {
            return Err(
                "solve: inputs, output, or algorithm do not match exact requirements".into(),
            );
        }
        validate_scratch("solve", &requirements.scratch, scratch)?;
        if requirements.scratch.len() != 2
            || requirements.scratch[0].shape.as_slice() != [requirements.n, requirements.n]
            || requirements.scratch[1].shape.as_slice()
                != [requirements.n, requirements.rhs_columns]
            || requirements
                .scratch
                .iter()
                .any(|requirement| requirement.dtype != DType::F64)
        {
            return Err("solve: invalid exact scratch requirements".into());
        }
        macro_rules! solve_matrix {
            ($values:expr, $type:ty) => {
                solve_rhs_typed::<$type>(
                    $values,
                    &self.layout,
                    rhs,
                    destination,
                    scratch,
                    requirements,
                )
            };
        }
        match &self.buffer {
            CpuBuffer::F32(values) => solve_matrix!(values, f32),
            CpuBuffer::F64(values) => solve_matrix!(values, f64),
            CpuBuffer::F16(values) => solve_matrix!(values, half::f16),
            CpuBuffer::BF16(values) => solve_matrix!(values, half::bf16),
            CpuBuffer::U8(values) => solve_matrix!(values, u8),
            CpuBuffer::U32(values) => solve_matrix!(values, u32),
            CpuBuffer::I64(values) => solve_matrix!(values, i64),
        }
    }

    pub fn solve(&self, rhs: &Tensor) -> Tensor {
        self.try_solve(rhs)
            .unwrap_or_else(|message| panic!("{message}"))
    }

    pub fn try_solve(&self, rhs: &Tensor) -> Result<Tensor, &'static str> {
        let requirements = self
            .solve_requirements(rhs)
            .unwrap_or_else(|message| panic!("{message}"));
        let mut output = Tensor::empty(&requirements.output.shape, requirements.output.dtype);
        let mut lu = Tensor::empty(
            &requirements.scratch[0].shape,
            requirements.scratch[0].dtype,
        );
        let mut work = Tensor::empty(
            &requirements.scratch[1].shape,
            requirements.scratch[1].dtype,
        );
        let result = self.solve_into(
            rhs,
            &mut output
                .destination()
                .expect("new CPU tensor storage must be unique"),
            &mut [
                lu.destination()
                    .expect("new CPU scratch storage must be unique"),
                work.destination()
                    .expect("new CPU scratch storage must be unique"),
            ],
            &requirements,
        );
        match result {
            Ok(()) => Ok(output),
            Err(message) if message == "matrix is singular" => Err("matrix is singular"),
            Err(message) => panic!("solve exact execution failed: {message}"),
        }
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
    fn inverse_det_and_solve_wrappers_preserve_results() {
        let a = Tensor::from_vec(vec![3f32, 1., 1., 2.], vec![2, 2]);
        let determinant = a.det();
        assert!((f32_data(&determinant)[0] - 5.0).abs() < 1e-5);

        let inverse = a.inverse();
        let product = a.matmul(&inverse);
        let product = f32_data(&product);
        assert!((product[0] - 1.0).abs() < 1e-5);
        assert!(product[1].abs() < 1e-5);
        assert!(product[2].abs() < 1e-5);
        assert!((product[3] - 1.0).abs() < 1e-5);

        let rhs = Tensor::from_vec(vec![9f32, 8.], vec![2, 1]);
        let solution = a.solve(&rhs);
        let solution = f32_data(&solution);
        assert!((solution[0] - 2.0).abs() < 1e-5);
        assert!((solution[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn requirements_are_exact_and_algorithm_is_frozen() {
        let a = Tensor::from_vec(vec![2f32, 0., 0., 4., 3., 1., 1., 2.], vec![2, 2, 2]);
        let rhs = Tensor::from_vec(vec![2f64, 4., 9., 8.], vec![2, 2, 1]);

        let determinant = a.det_requirements().unwrap();
        assert_eq!(determinant.algorithm, LinalgAlgorithm::LuPartialPivotF64);
        assert_eq!(determinant.output.shape, vec![2]);
        assert_eq!(determinant.output.dtype, DType::F32);
        assert_eq!(determinant.scratch.len(), 1);
        assert_eq!(determinant.scratch[0].shape, vec![2, 2]);
        assert_eq!(determinant.scratch[0].dtype, DType::F64);
        assert_eq!(determinant.scratch_bytes(), 4 * size_of::<f64>());

        let inverse = a.inverse_requirements().unwrap();
        assert_eq!(inverse.output.shape, vec![2, 2, 2]);
        assert_eq!(inverse.scratch.len(), 2);
        assert_eq!(inverse.scratch_bytes(), 8 * size_of::<f64>());

        let solve = a.solve_requirements(&rhs).unwrap();
        assert_eq!(solve.output.shape, vec![2, 2, 1]);
        assert_eq!(solve.output.dtype, DType::F32);
        assert_eq!(solve.rhs_columns, 1);
        assert_eq!(solve.scratch[1].shape, vec![2, 1]);
        assert_eq!(solve.scratch_bytes(), 6 * size_of::<f64>());
    }

    #[test]
    fn all_linalg_into_paths_are_allocation_free_under_guard_and_match_wrappers() {
        let a = Tensor::from_vec(vec![3f32, 1., 1., 2.], vec![2, 2]);
        let rhs = Tensor::from_vec(vec![9f32, 8.], vec![2, 1]);

        let det_requirements = a.det_requirements().unwrap();
        let mut det_output = Tensor::empty(
            &det_requirements.output.shape,
            det_requirements.output.dtype,
        );
        let mut det_lu = Tensor::empty(
            &det_requirements.scratch[0].shape,
            det_requirements.scratch[0].dtype,
        );
        let mut det_destination = det_output.destination().unwrap();
        let mut det_scratch = [det_lu.destination().unwrap()];

        let inverse_requirements = a.inverse_requirements().unwrap();
        let mut inverse_output = Tensor::empty(
            &inverse_requirements.output.shape,
            inverse_requirements.output.dtype,
        );
        let mut inverse_lu = Tensor::empty(
            &inverse_requirements.scratch[0].shape,
            inverse_requirements.scratch[0].dtype,
        );
        let mut inverse_work = Tensor::empty(
            &inverse_requirements.scratch[1].shape,
            inverse_requirements.scratch[1].dtype,
        );
        let mut inverse_destination = inverse_output.destination().unwrap();
        let mut inverse_scratch = [
            inverse_lu.destination().unwrap(),
            inverse_work.destination().unwrap(),
        ];

        let solve_requirements = a.solve_requirements(&rhs).unwrap();
        let mut solve_output = Tensor::empty(
            &solve_requirements.output.shape,
            solve_requirements.output.dtype,
        );
        let mut solve_lu = Tensor::empty(
            &solve_requirements.scratch[0].shape,
            solve_requirements.scratch[0].dtype,
        );
        let mut solve_work = Tensor::empty(
            &solve_requirements.scratch[1].shape,
            solve_requirements.scratch[1].dtype,
        );
        let mut solve_destination = solve_output.destination().unwrap();
        let mut solve_scratch = [
            solve_lu.destination().unwrap(),
            solve_work.destination().unwrap(),
        ];

        {
            let _guard = ExecutableAllocationGuard::enter();
            a.det_into(&mut det_destination, &mut det_scratch, &det_requirements)
                .unwrap();
            a.inverse_into(
                &mut inverse_destination,
                &mut inverse_scratch,
                &inverse_requirements,
            )
            .unwrap();
            a.solve_into(
                &rhs,
                &mut solve_destination,
                &mut solve_scratch,
                &solve_requirements,
            )
            .unwrap();
        }

        assert_eq!(f32_data(&det_output), f32_data(&a.det()));
        assert_eq!(f32_data(&inverse_output), f32_data(&a.inverse()));
        assert_eq!(f32_data(&solve_output), f32_data(&a.solve(&rhs)));
    }

    #[test]
    fn strided_f64_inputs_and_mixed_rhs_dtype_are_supported() {
        let matrix = Tensor::from_vec(vec![4f64, 2., 1., 3.], vec![2, 2]);
        let matrix = matrix.view(matrix.layout.permute(&[1, 0]));
        let inverse = matrix.inverse();
        let CpuBuffer::F64(values) = &inverse.buffer else {
            panic!()
        };
        let expected = [0.3, -0.1, -0.2, 0.4];
        for (&actual, expected) in values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }

        let rhs = Tensor::from_vec(vec![1f32, 3., 2., 4.], vec![2, 2]);
        let rhs = rhs.view(rhs.layout.permute(&[1, 0]));
        let solution = matrix.solve(&rhs);
        assert_eq!(solution.dtype(), DType::F64);
        let reconstructed = matrix.matmul(&solution);
        let CpuBuffer::F64(actual) = &reconstructed.buffer else {
            panic!()
        };
        let expected = rhs.cast(DType::F64).contiguous();
        let CpuBuffer::F64(expected) = &expected.buffer else {
            panic!()
        };
        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn batched_linalg_reuses_exact_scratch_across_pivoted_matrices() {
        let matrix = Tensor::from_vec(vec![2f64, 0., 0., 4., 0., 1., 2., 3.], vec![2, 2, 2]);
        let rhs = Tensor::from_vec(vec![2f64, 4., 8., 12., 1., 2., 5., 8.], vec![2, 2, 2]);

        let determinant = matrix.det();
        let CpuBuffer::F64(determinant) = &determinant.buffer else {
            panic!()
        };
        assert_eq!(determinant.as_slice(), &[8.0, -2.0]);

        let inverse = matrix.inverse();
        let identity = matrix.matmul(&inverse);
        let CpuBuffer::F64(identity) = &identity.buffer else {
            panic!()
        };
        for (index, &value) in identity.iter().enumerate() {
            let expected = if index % 4 == 0 || index % 4 == 3 {
                1.0
            } else {
                0.0
            };
            assert!((value - expected).abs() < 1e-12);
        }

        let solution = matrix.solve(&rhs);
        let reconstructed = matrix.matmul(&solution);
        let (CpuBuffer::F64(actual), CpuBuffer::F64(expected)) =
            (&reconstructed.buffer, &rhs.buffer)
        else {
            panic!()
        };
        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn into_rejects_wrong_scratch_dtype_and_stale_layout_requirements() {
        let matrix = Tensor::from_vec(vec![2f32, 0., 0., 4.], vec![2, 2]);
        let requirements = matrix.inverse_requirements().unwrap();
        let mut output = Tensor::empty(&requirements.output.shape, requirements.output.dtype);
        let mut wrong_lu = Tensor::empty(&requirements.scratch[0].shape, DType::F32);
        let mut work = Tensor::empty(
            &requirements.scratch[1].shape,
            requirements.scratch[1].dtype,
        );
        assert!(matrix
            .inverse_into(
                &mut output.destination().unwrap(),
                &mut [wrong_lu.destination().unwrap(), work.destination().unwrap(),],
                &requirements,
            )
            .is_err());

        let transposed = matrix.view(matrix.layout.permute(&[1, 0]));
        let mut lu = Tensor::empty(
            &requirements.scratch[0].shape,
            requirements.scratch[0].dtype,
        );
        assert!(transposed
            .inverse_into(
                &mut output.destination().unwrap(),
                &mut [lu.destination().unwrap(), work.destination().unwrap()],
                &requirements,
            )
            .is_err());
    }

    #[test]
    fn singular_inverse_and_solve_are_fallible() {
        let a = Tensor::from_vec(vec![1f32, 2., 2., 4.], vec![2, 2]);
        let b = Tensor::from_vec(vec![1f32, 2.], vec![2, 1]);
        assert_eq!(a.try_inverse().err(), Some("matrix is singular"));
        assert_eq!(a.try_solve(&b).err(), Some("matrix is singular"));
    }
}

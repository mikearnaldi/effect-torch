//! Index-space operations: `gather`, `index_select`, `scatter_add`, `cat`.
//!
//! Index tensors may be `u8`, `u32`, or `i64`. The kernels reject negative
//! `i64` indexes and check every index against the addressed extent during
//! execution. They address sources through their layouts, so strided inputs do
//! not need a contiguous copy.
//!
//! - `gather` reads `input[coord..., ids[..], coord...]` where the ids
//!   tensor has the same rank as the input and the output takes the ids
//!   shape.
//! - `index_select` replaces dimension `dim` of the input with `ids.numel()`
//!   selected slices.
//! - `scatter_add` copies the input to the output, then accumulates `src`
//!   into positions addressed by `ids` along `dim`.
//! - `cat` concatenates same-rank, same-dtype tensors along `dim`, copying
//!   each input in one outer × dim × inner blocked pass.
//!
//! [`IndexingRequirements`] records an invocation's output requirement, input
//! layouts, and [`IndexingTopology`] pass count for the executor.

use super::tensor::{source_index, CpuBuffer, CpuDestination, CpuTensorRequirement, Elem, Tensor};
use effect_torch_runtime::{DType, Layout};

/// Execution strategy recorded in an [`IndexingRequirements`] plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexingTopology {
    /// Direct indexing. `passes` counts output sweeps. `scatter_add` uses
    /// two, one to copy and one to accumulate. `cat` uses one per input.
    Direct { passes: usize },
}

/// Exact caller-owned resources for one layout-aware indexing invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexingRequirements {
    pub output: CpuTensorRequirement,
    /// Optional converted IDs. This is `None` because direct CPU indexing
    /// consumes IDs indirectly.
    pub ids: Option<CpuTensorRequirement>,
    pub scratch: Vec<CpuTensorRequirement>,
    pub topology: IndexingTopology,
    pub input_layouts: Vec<Layout>,
    pub dim: usize,
}

impl IndexingRequirements {
    pub fn scratch_bytes(&self) -> usize {
        self.ids
            .iter()
            .chain(&self.scratch)
            .try_fold(0usize, |total, requirement| {
                total.checked_add(requirement.bytes)
            })
            .expect("indexing scratch byte size overflow")
    }
}

fn checked_requirement(
    shape: &[usize],
    dtype: DType,
    operation: &str,
) -> Result<CpuTensorRequirement, String> {
    shape
        .iter()
        .try_fold(1usize, |total, &dimension| total.checked_mul(dimension))
        .and_then(|elements| elements.checked_mul(dtype.size_in_bytes()))
        .ok_or_else(|| format!("{operation}: output byte size overflow"))?;
    Ok(CpuTensorRequirement::new(shape, dtype))
}

fn allocate_output(
    requirement: CpuTensorRequirement,
    write: impl FnOnce(&mut CpuDestination<'_>) -> Result<(), String>,
) -> Tensor {
    let mut output = Tensor::empty(&requirement.shape, requirement.dtype);
    {
        let mut destination = output
            .destination()
            .expect("new CPU tensor storage must be unique");
        write(&mut destination).expect("new CPU tensor must satisfy indexing requirements");
    }
    output
}

fn validate_indices(ids: &Tensor, operation: &str) -> Result<(), String> {
    match ids.dtype() {
        DType::U8 | DType::U32 | DType::I64 => Ok(()),
        dtype => Err(format!(
            "{operation}: indices must be u8/u32/i64, got {dtype}"
        )),
    }
}

fn index_at(ids: &Tensor, linear: usize, operation: &str) -> Result<usize, String> {
    let index = source_index(&ids.layout, linear);
    match &ids.buffer {
        CpuBuffer::U8(values) => Ok(values[index] as usize),
        CpuBuffer::U32(values) => Ok(values[index] as usize),
        CpuBuffer::I64(values) => usize::try_from(values[index])
            .map_err(|_| format!("{operation}: negative index {}", values[index])),
        _ => Err(format!(
            "{operation}: indices must be u8/u32/i64, got {}",
            ids.dtype()
        )),
    }
}

fn validate_dimension(tensor: &Tensor, dim: usize, operation: &str) -> Result<(), String> {
    if dim >= tensor.shape().len() {
        return Err(format!(
            "{operation}: dimension {dim} is out of bounds for rank {}",
            tensor.shape().len()
        ));
    }
    Ok(())
}

fn gather_into_impl<T: Elem>(
    input: &[T],
    tensor: &Tensor,
    dim: usize,
    ids: &Tensor,
    destination: &mut CpuDestination<'_>,
) -> Result<(), String> {
    destination.write::<T, _>("gather", ids.shape(), |output| -> Result<(), String> {
        for (linear, value) in output.iter_mut().enumerate() {
            let id = index_at(ids, linear, "gather")?;
            if id >= tensor.shape()[dim] {
                return Err(format!(
                    "gather: index {id} is out of bounds for extent {}",
                    tensor.shape()[dim]
                ));
            }
            let mut remainder = linear;
            let mut input_index = tensor.layout.offset();
            for dimension in (0..tensor.shape().len()).rev() {
                let width = ids.shape()[dimension].max(1);
                let coordinate = remainder % width;
                remainder /= width;
                input_index += if dimension == dim {
                    id * tensor.layout.strides()[dimension]
                } else {
                    coordinate * tensor.layout.strides()[dimension]
                };
            }
            *value = input[input_index];
        }
        Ok(())
    })?
}

fn index_select_into_impl<T: Elem>(
    input: &[T],
    tensor: &Tensor,
    dim: usize,
    ids: &Tensor,
    destination: &mut CpuDestination<'_>,
) -> Result<(), String> {
    destination.write_current_shaped::<T, _>(
        "index_select",
        |output_shape, output| -> Result<(), String> {
            for (linear, value) in output.iter_mut().enumerate() {
                let mut remainder = linear;
                let mut input_index = tensor.layout.offset();
                for dimension in (0..tensor.shape().len()).rev() {
                    let width = output_shape[dimension].max(1);
                    let coordinate = remainder % width;
                    remainder /= width;
                    if dimension == dim {
                        let id = index_at(ids, coordinate, "index_select")?;
                        if id >= tensor.shape()[dim] {
                            return Err(format!(
                                "index_select: index {id} is out of bounds for extent {}",
                                tensor.shape()[dim]
                            ));
                        }
                        input_index += id * tensor.layout.strides()[dimension];
                    } else {
                        input_index += coordinate * tensor.layout.strides()[dimension];
                    }
                }
                *value = input[input_index];
            }
            Ok(())
        },
    )?
}

fn scatter_add_into_impl<T: Elem + std::ops::Add<Output = T>>(
    input: &[T],
    tensor: &Tensor,
    dim: usize,
    ids: &Tensor,
    source: &[T],
    source_tensor: &Tensor,
    destination: &mut CpuDestination<'_>,
) -> Result<(), String> {
    destination.write::<T, _>(
        "scatter_add",
        tensor.shape(),
        |output| -> Result<(), String> {
            super::tensor::copy_strided(input, &tensor.layout, output);
            for linear in 0..source_tensor.numel() {
                let id = index_at(ids, linear, "scatter_add")?;
                if id >= tensor.shape()[dim] {
                    return Err(format!(
                        "scatter_add: index {id} is out of bounds for extent {}",
                        tensor.shape()[dim]
                    ));
                }
                let mut remainder = linear;
                let mut output_index = 0usize;
                let mut output_stride = 1usize;
                for dimension in (0..tensor.shape().len()).rev() {
                    let width = source_tensor.shape()[dimension].max(1);
                    let coordinate = remainder % width;
                    remainder /= width;
                    output_index += if dimension == dim {
                        id * output_stride
                    } else {
                        coordinate * output_stride
                    };
                    output_stride *= tensor.shape()[dimension];
                }
                output[output_index] =
                    output[output_index] + source[source_index(&source_tensor.layout, linear)];
            }
            Ok(())
        },
    )?
}

fn cat_strided_into_impl<T: Elem>(
    tensors: &[&Tensor],
    dim: usize,
    destination: &mut CpuDestination<'_>,
) -> Result<(), String> {
    destination.write_current_shaped::<T, _>("cat", |output_shape, output| {
        let inner = output_shape[dim + 1..].iter().product::<usize>();
        let outer = output_shape[..dim].iter().product::<usize>();
        let mut dimension_offset = 0usize;
        for tensor in tensors {
            let tensor_dimension = tensor.shape()[dim];
            let input = T::storage_of(&tensor.buffer)
                .expect("cat dtype validated before dispatch")
                .as_slice();
            for outer_index in 0..outer {
                for dimension_index in 0..tensor_dimension {
                    for inner_index in 0..inner {
                        let input_linear = (outer_index * tensor_dimension + dimension_index)
                            * inner
                            + inner_index;
                        let output_linear =
                            (outer_index * output_shape[dim] + dimension_offset + dimension_index)
                                * inner
                                + inner_index;
                        output[output_linear] = input[source_index(&tensor.layout, input_linear)];
                    }
                }
            }
            dimension_offset += tensor_dimension;
        }
    })
}

impl Tensor {
    pub fn gather_requirements(
        &self,
        dim: usize,
        ids: &Tensor,
    ) -> Result<IndexingRequirements, String> {
        validate_dimension(self, dim, "gather")?;
        validate_indices(ids, "gather")?;
        if ids.shape().len() != self.shape().len() {
            return Err(format!(
                "gather: index rank {} does not match input rank {}",
                ids.shape().len(),
                self.shape().len()
            ));
        }
        for dimension in 0..self.shape().len() {
            if dimension != dim && ids.shape()[dimension] > self.shape()[dimension] {
                return Err(format!(
                    "gather: index extent exceeds input extent at dimension {dimension}"
                ));
            }
        }
        Ok(IndexingRequirements {
            output: checked_requirement(ids.shape(), self.dtype(), "gather")?,
            ids: None,
            scratch: Vec::new(),
            topology: IndexingTopology::Direct { passes: 1 },
            input_layouts: vec![self.layout.clone(), ids.layout.clone()],
            dim,
        })
    }

    pub fn gather_output_requirements(
        &self,
        dim: usize,
        ids: &Tensor,
    ) -> Result<CpuTensorRequirement, String> {
        Ok(self.gather_requirements(dim, ids)?.output)
    }

    pub fn gather_scratch_requirements(
        &self,
        dim: usize,
        ids: &Tensor,
    ) -> Result<Vec<CpuTensorRequirement>, String> {
        Ok(self.gather_requirements(dim, ids)?.scratch)
    }

    pub fn gather_into(
        &self,
        dim: usize,
        ids: &Tensor,
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        validate_dimension(self, dim, "gather")?;
        validate_indices(ids, "gather")?;
        if ids.shape().len() != self.shape().len() {
            return Err(format!(
                "gather: index rank {} does not match input rank {}",
                ids.shape().len(),
                self.shape().len()
            ));
        }
        for dimension in 0..self.shape().len() {
            if dimension != dim && ids.shape()[dimension] > self.shape()[dimension] {
                return Err(format!(
                    "gather: index extent exceeds input extent at dimension {dimension}"
                ));
            }
        }
        macro_rules! gather {
            ($values:expr) => {
                gather_into_impl($values, self, dim, ids, destination)
            };
        }
        match &self.buffer {
            CpuBuffer::F32(values) => gather!(values),
            CpuBuffer::F64(values) => gather!(values),
            CpuBuffer::F16(values) => gather!(values),
            CpuBuffer::BF16(values) => gather!(values),
            CpuBuffer::U8(values) => gather!(values),
            CpuBuffer::U32(values) => gather!(values),
            CpuBuffer::I64(values) => gather!(values),
        }
    }

    pub fn gather(&self, dim: usize, ids: &Tensor) -> Tensor {
        let requirements = self
            .gather_requirements(dim, ids)
            .unwrap_or_else(|message| panic!("{message}"));
        allocate_output(requirements.output, |destination| {
            self.gather_into(dim, ids, destination)
        })
    }

    pub fn index_select_requirements(
        &self,
        dim: usize,
        ids: &Tensor,
    ) -> Result<IndexingRequirements, String> {
        validate_dimension(self, dim, "index_select")?;
        validate_indices(ids, "index_select")?;
        let mut output_shape = self.shape().to_vec();
        output_shape[dim] = ids.numel();
        Ok(IndexingRequirements {
            output: checked_requirement(&output_shape, self.dtype(), "index_select")?,
            ids: None,
            scratch: Vec::new(),
            topology: IndexingTopology::Direct { passes: 1 },
            input_layouts: vec![self.layout.clone(), ids.layout.clone()],
            dim,
        })
    }

    pub fn index_select_output_requirements(
        &self,
        dim: usize,
        ids: &Tensor,
    ) -> Result<CpuTensorRequirement, String> {
        Ok(self.index_select_requirements(dim, ids)?.output)
    }

    pub fn index_select_scratch_requirements(
        &self,
        dim: usize,
        ids: &Tensor,
    ) -> Result<Vec<CpuTensorRequirement>, String> {
        Ok(self.index_select_requirements(dim, ids)?.scratch)
    }

    pub fn index_select_into(
        &self,
        dim: usize,
        ids: &Tensor,
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        validate_dimension(self, dim, "index_select")?;
        validate_indices(ids, "index_select")?;
        if destination.shape().len() != self.shape().len() {
            return Err(format!(
                "index_select destination rank mismatch: expected {}, got {}",
                self.shape().len(),
                destination.shape().len()
            ));
        }
        for dimension in 0..self.shape().len() {
            let expected = if dimension == dim {
                ids.numel()
            } else {
                self.shape()[dimension]
            };
            if destination.shape()[dimension] != expected {
                return Err(format!(
                    "index_select destination shape mismatch at dimension {dimension}: expected {expected}, got {}",
                    destination.shape()[dimension]
                ));
            }
        }
        macro_rules! index_select {
            ($values:expr) => {
                index_select_into_impl($values, self, dim, ids, destination)
            };
        }
        match &self.buffer {
            CpuBuffer::F32(values) => index_select!(values),
            CpuBuffer::F64(values) => index_select!(values),
            CpuBuffer::F16(values) => index_select!(values),
            CpuBuffer::BF16(values) => index_select!(values),
            CpuBuffer::U8(values) => index_select!(values),
            CpuBuffer::U32(values) => index_select!(values),
            CpuBuffer::I64(values) => index_select!(values),
        }
    }

    pub fn index_select(&self, dim: usize, ids: &Tensor) -> Tensor {
        let requirements = self
            .index_select_requirements(dim, ids)
            .unwrap_or_else(|message| panic!("{message}"));
        allocate_output(requirements.output, |destination| {
            self.index_select_into(dim, ids, destination)
        })
    }

    pub fn scatter_add_requirements(
        &self,
        dim: usize,
        ids: &Tensor,
        src: &Tensor,
    ) -> Result<IndexingRequirements, String> {
        validate_dimension(self, dim, "scatter_add")?;
        validate_indices(ids, "scatter_add")?;
        if self.dtype() != src.dtype() {
            return Err(format!(
                "scatter_add: source dtype {} does not match input dtype {}",
                src.dtype(),
                self.dtype()
            ));
        }
        if ids.shape() != src.shape() {
            return Err(format!(
                "scatter_add: indexes shape {:?} does not match source shape {:?}",
                ids.shape(),
                src.shape()
            ));
        }
        if src.shape().len() != self.shape().len() {
            return Err(format!(
                "scatter_add: source rank {} does not match input rank {}",
                src.shape().len(),
                self.shape().len()
            ));
        }
        for dimension in 0..self.shape().len() {
            if dimension != dim && src.shape()[dimension] > self.shape()[dimension] {
                return Err(format!(
                    "scatter_add: source extent exceeds input at dimension {dimension}"
                ));
            }
        }
        Ok(IndexingRequirements {
            output: checked_requirement(self.shape(), self.dtype(), "scatter_add")?,
            ids: None,
            scratch: Vec::new(),
            topology: IndexingTopology::Direct { passes: 2 },
            input_layouts: vec![self.layout.clone(), ids.layout.clone(), src.layout.clone()],
            dim,
        })
    }

    pub fn scatter_add_output_requirements(
        &self,
        dim: usize,
        ids: &Tensor,
        src: &Tensor,
    ) -> Result<CpuTensorRequirement, String> {
        Ok(self.scatter_add_requirements(dim, ids, src)?.output)
    }

    pub fn scatter_add_scratch_requirements(
        &self,
        dim: usize,
        ids: &Tensor,
        src: &Tensor,
    ) -> Result<Vec<CpuTensorRequirement>, String> {
        Ok(self.scatter_add_requirements(dim, ids, src)?.scratch)
    }

    pub fn scatter_add_into(
        &self,
        dim: usize,
        ids: &Tensor,
        src: &Tensor,
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        validate_dimension(self, dim, "scatter_add")?;
        validate_indices(ids, "scatter_add")?;
        if self.dtype() != src.dtype() {
            return Err(format!(
                "scatter_add: source dtype {} does not match input dtype {}",
                src.dtype(),
                self.dtype()
            ));
        }
        if ids.shape() != src.shape() {
            return Err(format!(
                "scatter_add: indexes shape {:?} does not match source shape {:?}",
                ids.shape(),
                src.shape()
            ));
        }
        if src.shape().len() != self.shape().len() {
            return Err(format!(
                "scatter_add: source rank {} does not match input rank {}",
                src.shape().len(),
                self.shape().len()
            ));
        }
        for dimension in 0..self.shape().len() {
            if dimension != dim && src.shape()[dimension] > self.shape()[dimension] {
                return Err(format!(
                    "scatter_add: source extent exceeds input at dimension {dimension}"
                ));
            }
        }
        macro_rules! scatter_add {
            ($input:expr, $source:expr) => {
                scatter_add_into_impl($input, self, dim, ids, $source, src, destination)
            };
        }
        match (&self.buffer, &src.buffer) {
            (CpuBuffer::F32(input), CpuBuffer::F32(source)) => scatter_add!(input, source),
            (CpuBuffer::F64(input), CpuBuffer::F64(source)) => scatter_add!(input, source),
            (CpuBuffer::F16(input), CpuBuffer::F16(source)) => scatter_add!(input, source),
            (CpuBuffer::BF16(input), CpuBuffer::BF16(source)) => scatter_add!(input, source),
            (CpuBuffer::U8(input), CpuBuffer::U8(source)) => scatter_add!(input, source),
            (CpuBuffer::U32(input), CpuBuffer::U32(source)) => scatter_add!(input, source),
            (CpuBuffer::I64(input), CpuBuffer::I64(source)) => scatter_add!(input, source),
            _ => unreachable!("dtype checked before dispatch"),
        }
    }

    pub fn scatter_add(&self, dim: usize, ids: &Tensor, src: &Tensor) -> Tensor {
        let requirements = self
            .scatter_add_requirements(dim, ids, src)
            .unwrap_or_else(|message| panic!("{message}"));
        allocate_output(requirements.output, |destination| {
            self.scatter_add_into(dim, ids, src, destination)
        })
    }

    pub fn cat_requirements(
        tensors: &[&Tensor],
        dim: usize,
    ) -> Result<IndexingRequirements, String> {
        let Some(first) = tensors.first() else {
            return Err("cat: tensors must not be empty".to_string());
        };
        let dtype = first.dtype();
        let rank = first.shape().len();
        if dim >= rank {
            return Err(format!(
                "cat: dimension {dim} is out of bounds for rank {rank}"
            ));
        }
        let mut output_shape = first.shape().to_vec();
        let mut concatenated = 0usize;
        for tensor in tensors {
            if tensor.dtype() != dtype || tensor.shape().len() != rank {
                return Err("cat: all inputs must have the same dtype and rank".to_string());
            }
            for dimension in 0..rank {
                if dimension != dim && tensor.shape()[dimension] != output_shape[dimension] {
                    return Err(format!("cat: shape mismatch at dimension {dimension}"));
                }
            }
            concatenated = concatenated
                .checked_add(tensor.shape()[dim])
                .ok_or_else(|| "cat: concatenated dimension overflow".to_string())?;
        }
        output_shape[dim] = concatenated;
        Ok(IndexingRequirements {
            output: checked_requirement(&output_shape, dtype, "cat")?,
            ids: None,
            scratch: Vec::new(),
            topology: IndexingTopology::Direct {
                passes: tensors.len(),
            },
            input_layouts: tensors.iter().map(|tensor| tensor.layout.clone()).collect(),
            dim,
        })
    }

    pub fn cat_output_requirements(
        tensors: &[&Tensor],
        dim: usize,
    ) -> Result<CpuTensorRequirement, String> {
        Ok(Self::cat_requirements(tensors, dim)?.output)
    }

    pub fn cat_scratch_requirements(
        tensors: &[&Tensor],
        dim: usize,
    ) -> Result<Vec<CpuTensorRequirement>, String> {
        Ok(Self::cat_requirements(tensors, dim)?.scratch)
    }

    pub fn cat_into(
        tensors: &[&Tensor],
        dim: usize,
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        let Some(first) = tensors.first() else {
            return Err("cat: tensors must not be empty".to_string());
        };
        let rank = first.shape().len();
        if dim >= rank {
            return Err(format!(
                "cat: dimension {dim} is out of bounds for rank {rank}"
            ));
        }
        if destination.shape().len() != rank {
            return Err(format!(
                "cat destination rank mismatch: expected {rank}, got {}",
                destination.shape().len()
            ));
        }
        let mut concatenated = 0usize;
        for tensor in tensors {
            if tensor.dtype() != first.dtype() {
                return Err(format!(
                    "cat: input dtype {} does not match {}",
                    tensor.dtype(),
                    first.dtype()
                ));
            }
            if tensor.shape().len() != rank {
                return Err(format!(
                    "cat: input rank {} does not match {rank}",
                    tensor.shape().len()
                ));
            }
            for dimension in 0..rank {
                if dimension != dim && tensor.shape()[dimension] != first.shape()[dimension] {
                    return Err(format!("cat: shape mismatch at dimension {dimension}"));
                }
            }
            concatenated = concatenated
                .checked_add(tensor.shape()[dim])
                .ok_or_else(|| "cat: concatenated dimension overflow".to_string())?;
        }
        for dimension in 0..rank {
            let expected = if dimension == dim {
                concatenated
            } else {
                first.shape()[dimension]
            };
            if destination.shape()[dimension] != expected {
                return Err(format!(
                    "cat destination shape mismatch at dimension {dimension}: expected {expected}, got {}",
                    destination.shape()[dimension]
                ));
            }
        }
        match first.dtype() {
            DType::F32 => cat_strided_into_impl::<f32>(tensors, dim, destination),
            DType::F64 => cat_strided_into_impl::<f64>(tensors, dim, destination),
            DType::F16 => cat_strided_into_impl::<half::f16>(tensors, dim, destination),
            DType::BF16 => cat_strided_into_impl::<half::bf16>(tensors, dim, destination),
            DType::U8 => cat_strided_into_impl::<u8>(tensors, dim, destination),
            DType::U32 => cat_strided_into_impl::<u32>(tensors, dim, destination),
            DType::I64 => cat_strided_into_impl::<i64>(tensors, dim, destination),
        }
    }

    pub fn cat(tensors: &[&Tensor], dim: usize) -> Tensor {
        let requirements =
            Self::cat_requirements(tensors, dim).unwrap_or_else(|message| panic!("{message}"));
        allocate_output(requirements.output, |destination| {
            Self::cat_into(tensors, dim, destination)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExecutableAllocationGuard, CPU_STORAGE_ALIGNMENT};
    use effect_torch_runtime::Layout;

    fn f32_data(tensor: &Tensor) -> Vec<f32> {
        let CpuBuffer::F32(values) = &tensor.buffer else {
            panic!()
        };
        values.as_slice().to_vec()
    }

    #[test]
    fn gather_rows() {
        let x = Tensor::from_vec((0..6).map(|value| value as f32).collect(), vec![2, 3]);
        let ids = Tensor::from_vec(vec![1u32, 1, 1, 0, 0, 0, 1, 1, 1], vec![3, 3]);
        let gathered = x.gather(0, &ids);
        assert_eq!(gathered.shape(), &[3, 3]);
        assert_eq!(
            f32_data(&gathered),
            vec![3., 4., 5., 0., 1., 2., 3., 4., 5.]
        );
    }

    #[test]
    fn index_select_dim1() {
        let x = Tensor::from_vec((0..6).map(|value| value as f32).collect(), vec![2, 3]);
        let ids = Tensor::from_vec(vec![2u32, 0], vec![2]);
        let selected = x.index_select(1, &ids);
        assert_eq!(selected.shape(), &[2, 2]);
        assert_eq!(f32_data(&selected), vec![2., 0., 5., 3.]);
    }

    #[test]
    fn scatter_add_embedding() {
        let table = Tensor::zeros(&[4, 2], DType::F32);
        let ids = Tensor::from_vec(vec![1u32, 1, 3, 3], vec![2, 2]);
        let source = Tensor::from_vec(vec![1f32, 2., 3., 4.], vec![2, 2]);
        let output = table.scatter_add(0, &ids, &source);
        assert_eq!(f32_data(&output), vec![0., 0., 1., 2., 0., 0., 3., 4.]);
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

    #[test]
    fn exact_requirements_and_into_match_wrappers() {
        let input = Tensor::from_vec((0..6).map(|value| value as f32).collect(), vec![2, 3]);
        let ids = Tensor::from_vec(vec![2i64, 0], vec![2]);
        let requirements = input.index_select_requirements(1, &ids).unwrap();
        assert_eq!(requirements.output.shape, vec![2, 2]);
        assert_eq!(requirements.output.dtype, DType::F32);
        assert_eq!(requirements.output.bytes, 4 * std::mem::size_of::<f32>());
        assert_eq!(requirements.output.alignment, CPU_STORAGE_ALIGNMENT);
        assert!(requirements.ids.is_none());
        assert!(requirements.scratch.is_empty());
        assert_eq!(requirements.scratch_bytes(), 0);

        let mut output = Tensor::empty(&requirements.output.shape, requirements.output.dtype);
        input
            .index_select_into(1, &ids, &mut output.destination().unwrap())
            .unwrap();
        assert_eq!(f32_data(&output), f32_data(&input.index_select(1, &ids)));

        let cat = Tensor::cat_requirements(&[&input, &input], 0).unwrap();
        assert_eq!(cat.output.shape, vec![4, 3]);
        assert_eq!(cat.output.bytes, 12 * std::mem::size_of::<f32>());
        assert_eq!(cat.output.alignment, CPU_STORAGE_ALIGNMENT);

        let gather_ids = Tensor::from_vec(vec![2u8, 0, 1, 1], vec![2, 2]);
        let gather_requirements = input.gather_requirements(1, &gather_ids).unwrap();
        let mut gathered = Tensor::empty(
            &gather_requirements.output.shape,
            gather_requirements.output.dtype,
        );
        input
            .gather_into(1, &gather_ids, &mut gathered.destination().unwrap())
            .unwrap();
        assert_eq!(f32_data(&gathered), f32_data(&input.gather(1, &gather_ids)));

        let source = Tensor::from_vec(vec![1f32, 2., 3., 4.], vec![2, 2]);
        let scatter_requirements = input
            .scatter_add_requirements(1, &gather_ids, &source)
            .unwrap();
        let mut scattered = Tensor::empty(
            &scatter_requirements.output.shape,
            scatter_requirements.output.dtype,
        );
        input
            .scatter_add_into(
                1,
                &gather_ids,
                &source,
                &mut scattered.destination().unwrap(),
            )
            .unwrap();
        assert_eq!(
            f32_data(&scattered),
            f32_data(&input.scatter_add(1, &gather_ids, &source))
        );

        let mut concatenated = Tensor::empty(&cat.output.shape, cat.output.dtype);
        Tensor::cat_into(
            &[&input, &input],
            0,
            &mut concatenated.destination().unwrap(),
        )
        .unwrap();
        assert_eq!(
            f32_data(&concatenated),
            f32_data(&Tensor::cat(&[&input, &input], 0))
        );
    }

    #[test]
    fn into_rejects_invalid_destinations() {
        let input = Tensor::from_vec((0..6).map(|value| value as f32).collect(), vec![2, 3]);
        let ids = Tensor::from_vec(vec![2u32, 0], vec![2]);
        let mut wrong_shape = Tensor::zeros(&[4], DType::F32);
        assert!(input
            .index_select_into(1, &ids, &mut wrong_shape.destination().unwrap())
            .is_err());

        let mut wrong_dtype = Tensor::zeros(&[2, 2], DType::F64);
        assert!(input
            .index_select_into(1, &ids, &mut wrong_dtype.destination().unwrap())
            .is_err());

        let short = Tensor::from_vec(vec![0.0f32; 3], vec![3]);
        let mut insufficient = Tensor {
            buffer: short.buffer,
            layout: Layout::contiguous(vec![2, 2]),
        };
        assert!(input
            .index_select_into(1, &ids, &mut insufficient.destination().unwrap())
            .is_err());
    }

    #[test]
    fn caller_destinations_run_under_execution_guard_without_materialization() {
        let input = Tensor::from_vec((0..6).map(|value| value as f32).collect(), vec![2, 3]);
        let input = input.view(input.layout.permute(&[1, 0]));
        let ids = Tensor::from_vec(vec![1i64, 0], vec![2]);
        let mut selected = Tensor::empty(&[3, 2], DType::F32);
        let mut gathered = Tensor::empty(&[2, 2], DType::F32);
        let gather_ids = Tensor::from_vec(vec![2u8, 0, 1, 1], vec![2, 2]);
        let source = Tensor::from_vec(vec![1f32, 2., 3., 4.], vec![2, 2]);
        let base = Tensor::zeros(&[3, 2], DType::F32);
        let mut scattered = Tensor::empty(&[3, 2], DType::F32);
        let mut concatenated = Tensor::empty(&[6, 2], DType::F32);

        let _guard = ExecutableAllocationGuard::enter();
        input
            .index_select_into(1, &ids, &mut selected.destination().unwrap())
            .unwrap();
        input
            .gather_into(0, &gather_ids, &mut gathered.destination().unwrap())
            .unwrap();
        base.scatter_add_into(
            0,
            &gather_ids,
            &source,
            &mut scattered.destination().unwrap(),
        )
        .unwrap();
        Tensor::cat_into(
            &[&selected, &selected],
            0,
            &mut concatenated.destination().unwrap(),
        )
        .unwrap();
    }
}

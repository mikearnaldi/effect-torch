//! Stable expert grouping followed by the ordinary CPU matmul implementation.
//! Scratch contains activations and results only; expert weights stay borrowed.

use crate::{CpuBuffer, CpuDestination, CpuTensorRequirement, Elem, Tensor};
use effect_torch_runtime::{DType, Layout};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct Plan {
    layouts: [Layout; 3],
    dtype: DType,
    pub scratch: [CpuTensorRequirement; 2],
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(x: &Tensor, weight: &Tensor, ids: &Tensor) -> Result<Tensor, String> {
        let plan = Plan::new([x, weight, ids])?;
        let scratch = plan
            .scratch
            .each_ref()
            .map(|s| Tensor::empty(&s.shape, s.dtype));
        let mut output = Tensor::full(&[x.shape()[0], weight.shape()[1]], -99.0, x.dtype());
        let _guard = crate::storage::ExecutableAllocationGuard::enter();
        plan.execute_into(
            [x, weight, ids],
            [&scratch[0], &scratch[1]],
            &mut output.destination()?,
        )?;
        Ok(output)
    }

    fn check<T: Elem + PartialEq + std::fmt::Debug>(x: &Tensor, weight: &Tensor, ids: &Tensor) {
        let actual = run(x, weight, ids).unwrap();
        let rows = x.shape()[0];
        let inner = x.shape()[1];
        let columns = weight.shape()[1];
        let CpuBuffer::U32(routes) = &ids.buffer else {
            unreachable!()
        };
        let xs = T::storage_of(&x.buffer).unwrap();
        let ws = T::storage_of(&weight.buffer).unwrap();
        let actual = T::slice_of(&actual).unwrap();
        for expert in 0..weight.shape()[0] {
            let selected = (0..rows)
                .filter(|&row| {
                    routes[ids.layout.offset() + row * ids.layout.strides()[0]] as usize == expert
                })
                .collect::<Vec<_>>();
            if selected.is_empty() {
                continue;
            }
            let a = Tensor::from_vec(
                selected
                    .iter()
                    .flat_map(|&row| {
                        (0..inner).map(move |i| {
                            xs[x.layout.offset()
                                + row * x.layout.strides()[0]
                                + i * x.layout.strides()[1]]
                        })
                    })
                    .collect::<Vec<_>>(),
                vec![selected.len(), inner],
            );
            let b = Tensor::from_vec(
                (0..inner)
                    .flat_map(|i| {
                        (0..columns).map(move |o| {
                            ws[weight.layout.offset()
                                + expert * weight.layout.strides()[0]
                                + o * weight.layout.strides()[1]
                                + i * weight.layout.strides()[2]]
                        })
                    })
                    .collect::<Vec<_>>(),
                vec![inner, columns],
            );
            let expected = a.matmul(&b);
            let expected = T::slice_of(&expected).unwrap();
            for (group_row, &row) in selected.iter().enumerate() {
                assert_eq!(
                    &actual[row * columns..(row + 1) * columns],
                    &expected[group_row * columns..(group_row + 1) * columns]
                );
            }
        }
    }

    #[test]
    fn grouped_matches_exact_matmul_with_dense_groups_and_duplicate_routes() {
        for rows in [1, 7, 33, 96] {
            for dtype in [DType::F32, DType::BF16] {
                let x = Tensor::from_vec(
                    (0..rows * 65).map(|i| (i as f32 * 0.173).sin()).collect(),
                    vec![rows, 65],
                )
                .cast(dtype);
                let weight = Tensor::from_vec(
                    (0..3 * 17 * 65).map(|i| (i as f32 * 0.317).cos()).collect(),
                    vec![3, 17, 65],
                )
                .cast(dtype);
                for route in [0, 1] {
                    let ids = Tensor::from_vec(
                        (0..rows)
                            .map(|i| {
                                if route == 0 {
                                    2
                                } else {
                                    ((i * 5 + 1) % 3) as u32
                                }
                            })
                            .collect(),
                        vec![rows],
                    );
                    if dtype == DType::F32 {
                        check::<f32>(&x, &weight, &ids);
                    } else {
                        check::<half::bf16>(&x, &weight, &ids);
                    }
                }
            }
        }
    }

    #[test]
    fn grouped_consumes_offset_transposed_and_broadcast_views() {
        for dtype in [DType::F32, DType::BF16] {
            let xb =
                Tensor::from_vec((0..81).map(|i| i as f32 / 7.0).collect(), vec![81]).cast(dtype);
            let wb = Tensor::from_vec((0..81).map(|i| (i as f32 * 0.2).cos()).collect(), vec![81])
                .cast(dtype);
            let ids = Tensor::from_vec(vec![99u32, 2, 99, 0, 99, 2, 99, 1, 99, 0], vec![10])
                .view(Layout::new(vec![5], vec![2], 1));
            for strides in [vec![1, 7], vec![0, 1], vec![1, 0]] {
                let x = xb.view(Layout::new(vec![5, 4], strides, 2));
                for strides in [vec![25, 1, 5], vec![0, 4, 1], vec![16, 0, 1]] {
                    let w = wb.view(Layout::new(vec![3, 3, 4], strides, 1));
                    if dtype == DType::F32 {
                        check::<f32>(&x, &w, &ids);
                    } else {
                        check::<half::bf16>(&x, &w, &ids);
                    }
                }
            }
            let ids = ids.view(Layout::new(vec![5], vec![0], 1));
            if dtype == DType::F32 {
                check::<f32>(
                    &xb.view(Layout::contiguous(vec![5, 4])),
                    &wb.view(Layout::contiguous(vec![3, 3, 4])),
                    &ids,
                );
            } else {
                check::<half::bf16>(
                    &xb.view(Layout::contiguous(vec![5, 4])),
                    &wb.view(Layout::contiguous(vec![3, 3, 4])),
                    &ids,
                );
            }
        }
    }

    #[test]
    fn grouped_empty_axes_and_invalid_routes() {
        for dtype in [DType::F32, DType::BF16] {
            for (rows, inner, columns) in [(0, 4, 3), (5, 0, 3), (5, 4, 0), (0, 0, 0)] {
                let x = Tensor::zeros(&[rows, inner], dtype);
                let w = Tensor::zeros(&[2, columns, inner], dtype);
                let ids = Tensor::zeros(&[rows], DType::U32);
                let actual = run(&x, &w, &ids).unwrap();
                assert_eq!(actual.shape(), &[rows, columns]);
                if dtype == DType::F32 {
                    assert!(f32::slice_of(&actual).unwrap().iter().all(|&v| v == 0.0));
                } else {
                    assert!(half::bf16::slice_of(&actual)
                        .unwrap()
                        .iter()
                        .all(|v| v.to_f32() == 0.0));
                }
                if rows != 0 {
                    let bad = Tensor::full(&[rows], 2.0, DType::U32);
                    assert!(run(&x, &w, &bad).unwrap_err().contains("out of range"));
                }
            }
        }
    }
}

impl Plan {
    pub(crate) fn new(inputs: [&Tensor; 3]) -> Result<Self, String> {
        let [x, weight, indexes] = inputs;
        if x.shape().len() != 2
            || weight.shape().len() != 3
            || indexes.shape() != [x.shape()[0]]
            || weight.shape()[2] != x.shape()[1]
            || weight.shape()[0] == 0
            || weight.shape()[0] > u32::MAX as usize
            || indexes.dtype() != DType::U32
            || !matches!(x.dtype(), DType::F32 | DType::BF16)
            || weight.dtype() != x.dtype()
        {
            return Err("groupedExpertLinearRows: invalid input shapes or dtypes".into());
        }
        Ok(Self {
            layouts: inputs.map(|input| input.layout.clone()),
            dtype: x.dtype(),
            scratch: [
                CpuTensorRequirement::new(x.shape(), x.dtype()),
                CpuTensorRequirement::new(&[x.shape()[0], weight.shape()[1]], x.dtype()),
            ],
        })
    }

    /// The executor supplies exclusive planned scratch and output ranges.
    pub(crate) fn execute_into(
        &self,
        inputs: [&Tensor; 3],
        scratch: [&Tensor; 2],
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        for (index, input) in inputs.iter().enumerate() {
            if input.shape() != self.layouts[index].shape()
                || input.layout.strides() != self.layouts[index].strides()
                || input.dtype() != if index == 2 { DType::U32 } else { self.dtype }
            {
                return Err("groupedExpertLinearRows: input differs from compiled metadata".into());
            }
        }
        for (tensor, requirement) in scratch.iter().zip(&self.scratch) {
            if tensor.shape() != requirement.shape || tensor.dtype() != requirement.dtype {
                return Err(
                    "groupedExpertLinearRows: scratch differs from compiled metadata".into(),
                );
            }
        }
        let [x, weight, indexes] = inputs;
        let CpuBuffer::U32(ids) = &indexes.buffer else {
            unreachable!()
        };
        let mut groups = BTreeMap::<usize, Vec<usize>>::new();
        // Validate even when O=0 or I=0, before writing the destination.
        for row in 0..x.shape()[0] {
            let expert = ids[indexes.layout.offset() + row * indexes.layout.strides()[0]] as usize;
            if expert >= weight.shape()[0] {
                return Err("groupedExpertLinearRows: expert index is out of range".into());
            }
            groups.entry(expert).or_default().push(row);
        }
        match self.dtype {
            DType::F32 => self.grouped_into::<f32>(inputs, scratch, destination, &groups),
            DType::BF16 => self.grouped_into::<half::bf16>(inputs, scratch, destination, &groups),
            _ => unreachable!(),
        }
    }

    fn grouped_into<T: Elem>(
        &self,
        inputs: [&Tensor; 3],
        scratch: [&Tensor; 2],
        destination: &mut CpuDestination<'_>,
        groups: &BTreeMap<usize, Vec<usize>>,
    ) -> Result<(), String> {
        let [x, weight, _] = inputs;
        let (rows, inner, columns) = (x.shape()[0], x.shape()[1], weight.shape()[1]);
        let xs = T::storage_of(&x.buffer).ok_or("groupedExpertLinearRows: invalid storage")?;
        let [packed_x, packed_y] = scratch;
        // SAFETY: the command owns its planned scratch ranges exclusively.
        unsafe { CpuDestination::from_planned(packed_x) }.write::<T, _>(
            "groupedExpertLinearRows activations",
            &[rows, inner],
            |out| {
                for (packed, &row) in groups.values().flatten().enumerate() {
                    for i in 0..inner {
                        out[packed * inner + i] = xs[x.layout.offset()
                            + row * x.layout.strides()[0]
                            + i * x.layout.strides()[1]];
                    }
                }
            },
        )?;
        let mut start = 0;
        for (&expert, group) in groups {
            let a = packed_x.view(packed_x.layout.narrow(0, start, group.len()));
            let b = weight.view(Layout::new(
                vec![inner, columns],
                vec![weight.layout.strides()[2], weight.layout.strides()[1]],
                weight.layout.offset() + expert * weight.layout.strides()[0],
            ));
            let y = packed_y.view(packed_y.layout.narrow(0, start, group.len()));
            // CPU matmul policy depends only on dtype. Plan the exact active M,
            // including runtime storage offsets, then consume that immutable plan.
            let requirements = a.matmul_requirements(&b).map_err(str::to_string)?;
            // SAFETY: this is a subset of the exclusive planned result scratch.
            a.matmul_into(
                &b,
                &mut unsafe { CpuDestination::from_planned(&y) },
                &mut [],
                &requirements,
            )?;
            start += group.len();
        }
        let values =
            T::slice_of(packed_y).ok_or("groupedExpertLinearRows: invalid result scratch")?;
        destination.write::<T, _>("groupedExpertLinearRows", &[rows, columns], |out| {
            for (packed, &row) in groups.values().flatten().enumerate() {
                out[row * columns..(row + 1) * columns]
                    .copy_from_slice(&values[packed * columns..(packed + 1) * columns]);
            }
        })
    }
}

//! Exact-size expert matmuls on Metal. The only host-visible data is routing:
//! one N-word readback and one N-word stable permutation upload per invocation.
//! Every floating-point operation uses ordinary Metal matmul, including its
//! shape-dependent tiled/MMA/split-K choice. One expert matrix is packed at a
//! time; the full weight bank is never gathered or replicated.

use crate::device::{set_buffer, MetalDevice};
use crate::gemm::{self, GemmRequirements};
use crate::run::MetalTensor;
use effect_torch_runtime::{DType, Layout};
use objc2_metal::MTLComputeCommandEncoder;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Clone)]
pub(crate) struct Plan {
    layouts: [Layout; 3],
    dtype: DType,
    gather_key: u64,
    restore_key: u64,
    // Cheap O(N) metadata, with no M padding. Pipelines are shared by algorithm,
    // tile and split count, rather than compiled once per possible group size.
    variants: Vec<GemmRequirements>,
    pub(crate) split_k_elements: usize,
}

impl Plan {
    pub(crate) fn new(layouts: [Layout; 3], dtype: DType, mma: bool) -> Result<Self, String> {
        let [x, weight, ids] = &layouts;
        if !matches!(dtype, DType::F32 | DType::BF16)
            || x.rank() != 2
            || weight.rank() != 3
            || ids.shape() != [x.shape()[0]]
            || x.shape()[1] != weight.shape()[2]
            || weight.shape()[0] == 0
            || weight.shape()[0] > u32::MAX as usize
            || x.shape()[0] > u32::MAX as usize
        {
            return Err("groupedExpertLinearRows: invalid shapes or dtype".into());
        }
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        "groupedExpertLinearRows-v1".hash(&mut hash);
        dtype.hash(&mut hash);
        for layout in &layouts {
            layout.shape().hash(&mut hash);
            layout.strides().hash(&mut hash);
        }
        let gather_key = hash.finish();
        "restore".hash(&mut hash);
        let restore_key = hash.finish();
        let variants = (0..=x.shape()[0])
            .map(|rows| {
                gemm::matmul_requirements(
                    MetalDevice::get(),
                    &[rows, x.shape()[1]],
                    &[x.shape()[1], weight.shape()[1]],
                    dtype,
                    mma,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let split_k_elements = variants
            .iter()
            .filter_map(|plan| plan.split_k_scratch)
            .map(|scratch| scratch.elements)
            .max()
            .unwrap_or(0);
        Ok(Self {
            layouts,
            dtype,
            gather_key,
            restore_key,
            variants,
            split_k_elements,
        })
    }

    fn weight_layout(&self, offset: usize) -> Layout {
        let weight = &self.layouts[1];
        Layout::new(
            vec![weight.shape()[2], weight.shape()[1]],
            vec![weight.strides()[2], weight.strides()[1]],
            offset,
        )
    }

    pub(crate) fn warm(&self) -> Result<usize, String> {
        let dev = MetalDevice::get();
        let mut count = 0;
        let mut algorithms = Vec::new();
        for plan in &self.variants {
            if plan.output_elements != 0 && !algorithms.contains(&plan.algorithm) {
                count += gemm::precompile_matmul(dev, plan)?;
                algorithms.push(plan.algorithm);
            }
        }
        if self.layouts[0].shape()[0] == 0 {
            return Ok(count);
        }
        let weight = self.weight_layout(0);
        crate::kernels::warm_copy_layout(&weight, self.dtype)?;
        count += usize::from(weight.numel() != 0);
        let x = &self.layouts[0];
        let (rows, inner, columns) = (x.shape()[0], x.shape()[1], self.layouts[1].shape()[1]);
        let ty = if self.dtype == DType::BF16 {
            "ushort"
        } else {
            "uint"
        };
        let wide = MetalDevice::WIDE;
        if rows * inner != 0 {
            let (s0, s1) = (x.strides()[0], x.strides()[1]);
            dev.compile_lazy(self.gather_key, "et_grouped_expert_gather", || {
                format!(
                    r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_grouped_expert_gather(
    device const {ty}* x [[buffer(0)]], device const uint* permutation [[buffer(1)]],
    device {ty}* packed [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {{
    ulong i = ulong(gid.y) * {wide}ul + ulong(gid.x);
    if (i < {rows}ul * {inner}ul) {{
        ulong row = i / {inner}ul, column = i % {inner}ul;
        packed[i] = x[ulong(permutation[row]) * {s0}ul + column * {s1}ul];
    }}
}}
"#
                )
            })?;
            count += 1;
        }
        if rows * columns != 0 {
            dev.compile_lazy(self.restore_key, "et_grouped_expert_restore", || {
                format!(
                    r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_grouped_expert_restore(
    device const {ty}* packed [[buffer(0)]], device const uint* permutation [[buffer(1)]],
    device {ty}* output [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {{
    ulong i = ulong(gid.y) * {wide}ul + ulong(gid.x);
    if (i < {rows}ul * {columns}ul) {{
        ulong row = i / {columns}ul, column = i % {columns}ul;
        output[ulong(permutation[row]) * {columns}ul + column] = packed[i];
    }}
}}
"#
                )
            })?;
            count += 1;
        }
        Ok(count)
    }

    pub(crate) fn execute_into(
        &self,
        inputs: [&MetalTensor; 3],
        output: &MetalTensor,
        scratch: &[&MetalTensor],
        permutation: &MetalTensor,
        cancelled: &AtomicBool,
    ) -> Result<(), String> {
        for (index, input) in inputs.iter().enumerate() {
            if input.layout.shape() != self.layouts[index].shape()
                || input.layout.strides() != self.layouts[index].strides()
                || input.dtype != if index == 2 { DType::U32 } else { self.dtype }
                || input
                    .layout
                    .max_index()
                    .checked_mul(input.dtype.size_in_bytes())
                    .is_none_or(|bytes| bytes > input.buffer.size)
            {
                return Err("groupedExpertLinearRows: input differs from compiled metadata".into());
            }
        }
        let [x, weight, ids] = inputs;
        let (rows, inner, columns) = (
            x.layout.shape()[0],
            x.layout.shape()[1],
            weight.layout.shape()[1],
        );
        output.validate_destination("groupedExpertLinearRows", &[rows, columns], self.dtype)?;
        permutation.validate_destination(
            "groupedExpertLinearRows permutation",
            &[rows],
            DType::U32,
        )?;
        if scratch.len() != 3 + usize::from(self.split_k_elements != 0) {
            return Err("groupedExpertLinearRows: invalid scratch count".into());
        }
        scratch[0].validate_destination(
            "groupedExpertLinearRows activations",
            &[rows, inner],
            self.dtype,
        )?;
        scratch[1].validate_destination(
            "groupedExpertLinearRows results",
            &[rows, columns],
            self.dtype,
        )?;
        scratch[2].validate_destination(
            "groupedExpertLinearRows weight",
            &[inner, columns],
            self.dtype,
        )?;
        if rows == 0 {
            return Ok(());
        }
        check_cancelled(cancelled)?;
        let dev = MetalDevice::get();
        // This is an explicit control-data fence, not a capturable GPU-only
        // command. It also drains previous users of the reused staging range.
        dev.synchronize_buffer(&ids.buffer)?;
        check_cancelled(cancelled)?;
        let mut groups = BTreeMap::<usize, Vec<u32>>::new();
        // SAFETY: the fence completed the producer; input byte bounds were
        // checked above. Read the logical strided route vector only.
        let id_ptr = ids.buffer.contents_ptr().cast::<u32>();
        for row in 0..rows {
            let expert = unsafe { *id_ptr.add(ids.layout.offset() + row * ids.layout.strides()[0]) }
                as usize;
            if expert >= weight.layout.shape()[0] {
                return Err("groupedExpertLinearRows: expert index is out of range".into());
            }
            groups.entry(expert).or_default().push(row as u32);
        }
        // SAFETY: planned invocation staging is exclusively owned, contiguous,
        // validated above and no GPU consumer remains after the full fence.
        let perm = unsafe {
            std::slice::from_raw_parts_mut(
                permutation
                    .buffer
                    .contents_ptr()
                    .cast::<u32>()
                    .add(permutation.layout.offset()),
                rows,
            )
        };
        for (slot, &row) in groups.values().flatten().enumerate() {
            perm[slot] = row;
        }
        check_cancelled(cancelled)?;
        self.copy_rows(self.gather_key, x, permutation, scratch[0], rows * inner)?;
        let mut start = 0;
        for (&expert, group) in &groups {
            check_cancelled(cancelled)?;
            let a = view(scratch[0], vec![group.len(), inner], start * inner);
            let y = view(scratch[1], vec![group.len(), columns], start * columns);
            let b = MetalTensor {
                buffer: weight.buffer.clone(),
                dtype: self.dtype,
                layout: self
                    .weight_layout(weight.layout.offset() + expert * weight.layout.strides()[0]),
            };
            crate::kernels::copy_into(dev, &b, scratch[2])?;
            let requirements = &self.variants[group.len()];
            let partials = requirements
                .split_k_scratch
                .map(|requirement| view(scratch[3], requirement.shape.to_vec(), 0));
            crate::ops::matmul_into(&a, scratch[2], &y, partials.as_ref(), requirements)?;
            start += group.len();
        }
        self.copy_rows(
            self.restore_key,
            scratch[1],
            permutation,
            output,
            rows * columns,
        )
    }

    fn copy_rows(
        &self,
        key: u64,
        source: &MetalTensor,
        permutation: &MetalTensor,
        output: &MetalTensor,
        elements: usize,
    ) -> Result<(), String> {
        if elements == 0 {
            return Ok(());
        }
        let dev = MetalDevice::get();
        output.validate_destination(
            "groupedExpertLinearRows copy",
            output.layout.shape(),
            self.dtype,
        )?;
        let pipeline = dev
            .pipeline_cached(key)
            .ok_or("groupedExpertLinearRows: pipeline is not warm")?;
        dev.with_encoder(|encoder| {
            encoder.setComputePipelineState(pipeline.as_raw());
            set_buffer(
                encoder,
                0,
                &source.buffer,
                source.layout.offset() * self.dtype.size_in_bytes(),
            );
            set_buffer(
                encoder,
                1,
                &permutation.buffer,
                permutation.layout.offset() * 4,
            );
            set_buffer(
                encoder,
                2,
                &output.buffer,
                output.layout.offset() * self.dtype.size_in_bytes(),
            );
            let (grid, threads) = MetalDevice::grid_flat(elements.div_ceil(256) * 256);
            encoder.dispatchThreads_threadsPerThreadgroup(grid, threads);
        });
        Ok(())
    }
}

fn view(source: &MetalTensor, shape: Vec<usize>, offset: usize) -> MetalTensor {
    let layout = Layout::contiguous(shape);
    MetalTensor {
        buffer: source.buffer.clone(),
        dtype: source.dtype,
        layout: Layout::new(
            layout.shape().to_vec(),
            layout.strides().to_vec(),
            source.layout.offset() + offset,
        ),
    }
}

fn check_cancelled(cancelled: &AtomicBool) -> Result<(), String> {
    if cancelled.load(Ordering::Relaxed) {
        Err("operation aborted".into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(data: Vec<f32>, shape: Vec<usize>, dtype: DType) -> MetalTensor {
        let source = MetalTensor::from_f32(MetalDevice::get(), data, shape);
        crate::ops::cast(&source, dtype).unwrap()
    }

    fn ids(data: &[u32]) -> MetalTensor {
        MetalTensor {
            buffer: MetalDevice::get().alloc_with_data_u32(data),
            layout: Layout::contiguous(vec![data.len()]),
            dtype: DType::U32,
        }
    }

    fn execute(
        x: &MetalTensor,
        w: &MetalTensor,
        ids: &MetalTensor,
        cancelled: &AtomicBool,
    ) -> Result<MetalTensor, String> {
        let dev = MetalDevice::get();
        let (rows, inner, columns) = (
            x.layout.shape()[0],
            x.layout.shape()[1],
            w.layout.shape()[1],
        );
        let plan = Plan::new(
            [x.layout.clone(), w.layout.clone(), ids.layout.clone()],
            x.dtype,
            true,
        )?;
        plan.warm()?;
        let output = MetalTensor::empty(dev, vec![rows, columns], x.dtype);
        let permutation = MetalTensor::empty(dev, vec![rows], DType::U32);
        let mut scratch = vec![
            MetalTensor::empty(dev, vec![rows, inner], x.dtype),
            MetalTensor::empty(dev, vec![rows, columns], x.dtype),
            MetalTensor::empty(dev, vec![inner, columns], x.dtype),
        ];
        if plan.split_k_elements != 0 {
            scratch.push(MetalTensor::empty(
                dev,
                vec![plan.split_k_elements],
                DType::F32,
            ));
        }
        let submission = dev.begin_submission()?;
        let dispatch = dev.begin_executable_dispatch()?;
        let result = plan.execute_into(
            [x, w, ids],
            &output,
            &scratch.iter().collect::<Vec<_>>(),
            &permutation,
            cancelled,
        );
        drop(dispatch);
        // Scratch stays alive until all GPU users have completed on success,
        // invalid routes and cancellation alike.
        dev.synchronize()?;
        drop(submission);
        result?;
        Ok(output)
    }

    fn host(t: &MetalTensor) -> Vec<f32> {
        crate::ops::cast(&crate::ops::contiguous(t).unwrap(), DType::F32)
            .unwrap()
            .read_f32()
            .unwrap()
    }

    fn check(x: &MetalTensor, weight: &MetalTensor, indexes: &MetalTensor) {
        let xs = host(x);
        let ws = host(weight);
        let routes = crate::ops::contiguous(indexes)
            .unwrap()
            .to_u32_vec()
            .unwrap();
        let (rows, inner, columns) = (
            x.layout.shape()[0],
            x.layout.shape()[1],
            weight.layout.shape()[1],
        );
        let actual = host(&execute(x, weight, indexes, &AtomicBool::new(false)).unwrap());
        let mut expected = vec![0.0f32; rows * columns];
        for expert in 0..weight.layout.shape()[0] {
            let selected = (0..rows)
                .filter(|&row| routes[row] as usize == expert)
                .collect::<Vec<_>>();
            if selected.is_empty() {
                continue;
            }
            let a = tensor(
                selected
                    .iter()
                    .flat_map(|&row| xs[row * inner..(row + 1) * inner].iter().copied())
                    .collect(),
                vec![selected.len(), inner],
                x.dtype,
            );
            let mut transposed = vec![0.0f32; inner * columns];
            for i in 0..inner {
                for o in 0..columns {
                    transposed[i * columns + o] = ws[(expert * columns + o) * inner + i];
                }
            }
            let b = tensor(transposed, vec![inner, columns], x.dtype);
            let result = host(&crate::ops::matmul(&a, &b).unwrap());
            for (group_row, &row) in selected.iter().enumerate() {
                expected[row * columns..(row + 1) * columns]
                    .copy_from_slice(&result[group_row * columns..(group_row + 1) * columns]);
            }
        }
        assert_eq!(actual, expected, "exact same-backend grouped matmul parity");
    }

    #[test]
    fn grouped_matches_ordinary_matmul_at_exact_algorithm_transitions() {
        crate::device::with_execution_environment(false, true, || {
            for dtype in [DType::F32, DType::BF16] {
                for (group_rows, inner) in [
                    (1, 65),
                    (32, 65),
                    (33, 65),
                    (193, 65),
                    (33, 2049),
                    (34, 2049),
                    (128, 2049),
                    (129, 2049),
                ] {
                    let rows = group_rows + 7;
                    let x = tensor(
                        (0..rows * inner)
                            .map(|i| (i as f32 * 0.173).sin())
                            .collect(),
                        vec![rows, inner],
                        dtype,
                    );
                    let w = tensor(
                        (0..2 * 1024 * inner)
                            .map(|i| (i as f32 * 0.317).cos())
                            .collect(),
                        vec![2, 1024, inner],
                        dtype,
                    );
                    let mut routes = vec![1u32; rows];
                    for index in 0..7 {
                        routes[index * rows / 7] = 0;
                    }
                    assert_eq!(
                        routes.iter().filter(|&&expert| expert == 1).count(),
                        group_rows
                    );
                    check(&x, &w, &ids(&routes));
                }
            }
        });
    }

    #[test]
    fn grouped_plans_bound_pipeline_variants_for_all_2224_rows() {
        let plan = Plan::new(
            [
                Layout::contiguous(vec![2224, 2049]),
                Layout::contiguous(vec![16, 1024, 2049]),
                Layout::contiguous(vec![2224]),
            ],
            DType::BF16,
            true,
        )
        .unwrap();
        assert_eq!(plan.variants.len(), 2225);
        let mut algorithms = Vec::new();
        for (rows, variant) in plan.variants.iter().enumerate() {
            assert_eq!(variant.shape.m, rows);
            assert_eq!(
                *variant,
                gemm::matmul_requirements(
                    MetalDevice::get(),
                    &[rows, 2049],
                    &[2049, 1024],
                    DType::BF16,
                    true
                )
                .unwrap()
            );
            if !algorithms.contains(&variant.algorithm) {
                algorithms.push(variant.algorithm);
            }
        }
        assert!(algorithms.len() <= 16, "{:?}", algorithms);
        assert!(plan.warm().unwrap() <= 35);
        assert_ne!(plan.variants[32].algorithm, plan.variants[33].algorithm);
        assert_eq!(plan.variants[33].algorithm, plan.variants[34].algorithm);
        assert_ne!(plan.variants[128].algorithm, plan.variants[129].algorithm);
    }

    #[test]
    fn grouped_transposed_offset_and_broadcast_views() {
        for dtype in [DType::F32, DType::BF16] {
            let xb = tensor((0..81).map(|i| i as f32 / 7.0).collect(), vec![81], dtype);
            let wb = tensor(
                (0..81).map(|i| (i as f32 * 0.2).cos()).collect(),
                vec![81],
                dtype,
            );
            let mut routes = ids(&[99, 2, 99, 0, 99, 2, 99, 1, 99, 0]);
            routes.layout = Layout::new(vec![5], vec![2], 1);
            for strides in [vec![1, 7], vec![0, 1], vec![1, 0]] {
                let x = MetalTensor {
                    layout: Layout::new(vec![5, 4], strides, 2),
                    ..xb.clone()
                };
                for strides in [vec![25, 1, 5], vec![0, 4, 1], vec![16, 0, 1]] {
                    let w = MetalTensor {
                        layout: Layout::new(vec![3, 3, 4], strides, 1),
                        ..wb.clone()
                    };
                    check(&x, &w, &routes);
                }
            }
            routes.layout = Layout::new(vec![5], vec![0], 1);
            check(
                &view(&xb, vec![5, 4], 0),
                &view(&wb, vec![3, 3, 4], 0),
                &routes,
            );
        }
    }

    #[test]
    fn grouped_empty_axes_invalid_routes_and_cancellation() {
        for dtype in [DType::F32, DType::BF16] {
            for (rows, inner, columns) in [(0, 4, 3), (5, 0, 3), (5, 4, 0), (0, 0, 0)] {
                let x = tensor(vec![0.0; rows * inner], vec![rows, inner], dtype);
                let w = tensor(
                    vec![0.0; 2 * columns * inner],
                    vec![2, columns, inner],
                    dtype,
                );
                let routes = ids(&vec![0; rows]);
                let result = execute(&x, &w, &routes, &AtomicBool::new(false)).unwrap();
                assert_eq!(host(&result), vec![0.0; rows * columns]);
                if rows != 0 {
                    let bad = ids(&vec![2; rows]);
                    assert!(execute(&x, &w, &bad, &AtomicBool::new(false))
                        .err()
                        .unwrap()
                        .contains("out of range"));
                    assert_eq!(
                        execute(&x, &w, &routes, &AtomicBool::new(true))
                            .err()
                            .unwrap(),
                        "operation aborted"
                    );
                }
            }
        }
    }
}

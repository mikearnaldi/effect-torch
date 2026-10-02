//! Row-selected expert dots over native F32/BF16 storage, with one SIMD group
//! per output and a planned one-word invalid-route status.

use crate::device::{set_buffer, MetalDevice};
use crate::run::MetalTensor;
use effect_torch_runtime::{DType, Layout};
use objc2_metal::MTLComputeCommandEncoder;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
pub(crate) struct Plan {
    layouts: [Layout; 3],
    dtype: DType,
    key: u64,
    work: usize,
    threads: usize,
}

impl Plan {
    pub(crate) fn new(layouts: [Layout; 3], dtype: DType) -> Result<Self, String> {
        let [x, weight, ids] = &layouts;
        if !matches!(dtype, DType::F32 | DType::BF16)
            || x.shape().len() != 2
            || weight.shape().len() != 3
            || ids.shape() != [x.shape()[0]]
            || x.shape()[1] != weight.shape()[2]
            || weight.shape()[0] == 0
            || weight.shape()[0] > u32::MAX as usize
        {
            return Err("expertLinearRows: invalid shapes or dtype".into());
        }
        let work = x.shape()[0]
            .checked_mul(weight.shape()[1].max(1))
            .ok_or("expertLinearRows: work size overflow")?;
        let threads = work.min(8192).div_ceil(8) * 256;
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        "expertLinearRows-v1".hash(&mut hash);
        dtype.hash(&mut hash);
        for layout in &layouts {
            layout.shape().hash(&mut hash);
            layout.strides().hash(&mut hash);
        }
        Ok(Self {
            layouts,
            dtype,
            key: hash.finish(),
            work,
            threads,
        })
    }

    pub(crate) fn warm(&self) -> Result<(), String> {
        crate::kernels::warm_fill(&[1], 0.0, DType::U32)?;
        if self.work == 0 {
            return Ok(());
        }
        let [x, weight, ids] = &self.layouts;
        let (columns, inner, experts) = (weight.shape()[1], x.shape()[1], weight.shape()[0]);
        let (xs0, xs1, ws0, ws1, ws2, is0) = (
            x.strides()[0],
            x.strides()[1],
            weight.strides()[0],
            weight.strides()[1],
            weight.strides()[2],
            ids.strides()[0],
        );
        let width = columns.max(1);
        let groups = self.threads / 32;
        let work = self.work;
        let (ty, load_x, load_w, store) = if self.dtype == DType::BF16 {
            (
                "bfloat",
                "et_bf16_to_float(x[xi])",
                "et_bf16_to_float(w[wi])",
                "et_bf16_from_float(sum)",
            )
        } else {
            ("float", "x[xi]", "w[wi]", "sum")
        };
        let conversion = crate::kernels::BF16_CONVERSION_MSL;
        MetalDevice::get().compile_lazy(self.key, "et_expert_linear_rows", || {
            format!(
                r#"
#include <metal_stdlib>
using namespace metal;
{conversion}
kernel void et_expert_linear_rows(
    device const {ty}* x [[buffer(0)]], device const {ty}* w [[buffer(1)]],
    device const uint* ids [[buffer(2)]], device {ty}* output [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]],
    uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {{
    for (ulong out = ulong(gid) / 32ul; out < {work}ul; out += {groups}ul) {{
        ulong row = out / {width}ul, column = out % {width}ul;
        uint expert = ids[row * {is0}ul];
        if (ulong(expert) >= {experts}ul) {{
            if (lane == 0u) {{
                atomic_store_explicit(status, 1u, memory_order_relaxed);
                if ({columns}ul != 0ul) output[out] = {ty}(0);
            }}
            continue;
        }}
        if ({columns}ul == 0ul) continue;
        float sum = 0.0f;
        for (ulong i = lane; i < {inner}ul; i += 32ul) {{
            ulong xi = row * {xs0}ul + i * {xs1}ul;
            ulong wi = ulong(expert) * {ws0}ul + column * {ws1}ul + i * {ws2}ul;
            sum = fma({load_x}, {load_w}, sum);
        }}
        sum = simd_sum(sum);
        if (lane == 0u) output[out] = {store};
    }}
}}
"#
            )
        })?;
        Ok(())
    }

    pub(crate) fn execute_into(
        &self,
        inputs: [&MetalTensor; 3],
        output: &MetalTensor,
        status: &MetalTensor,
    ) -> Result<(), String> {
        for (index, input) in inputs.iter().enumerate() {
            if input.layout.shape() != self.layouts[index].shape()
                || input.layout.strides() != self.layouts[index].strides()
                || input.dtype != if index == 2 { DType::U32 } else { self.dtype }
            {
                return Err("expertLinearRows: input differs from compiled layout or dtype".into());
            }
        }
        output.validate_destination(
            "expertLinearRows",
            &[self.layouts[0].shape()[0], self.layouts[1].shape()[1]],
            self.dtype,
        )?;
        status.validate_destination("expertLinearRows status", &[1], DType::U32)?;
        let dev = MetalDevice::get();
        crate::kernels::fill_into(dev, status, 0.0)?;
        if self.work == 0 {
            return Ok(());
        }
        let pipeline = dev
            .pipeline_cached(self.key)
            .ok_or("expertLinearRows: pipeline is not warm")?;
        dev.with_encoder(|encoder| {
            encoder.setComputePipelineState(pipeline.as_raw());
            for (index, input) in inputs.iter().enumerate() {
                set_buffer(
                    encoder,
                    index,
                    &input.buffer,
                    input.layout.offset() * input.dtype.size_in_bytes(),
                );
            }
            set_buffer(
                encoder,
                3,
                &output.buffer,
                output.layout.offset() * self.dtype.size_in_bytes(),
            );
            set_buffer(encoder, 4, &status.buffer, status.layout.offset() * 4);
            encoder.dispatchThreads_threadsPerThreadgroup(
                MetalDevice::grid(self.threads, 1, 1),
                MetalDevice::grid(256, 1, 1),
            );
        });
        Ok(())
    }
}

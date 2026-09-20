//! Indexed expert dot products over borrowed F32/BF16 weight storage.
//! No gathered weight matrices or conversion buffers are materialized.

use crate::tensor::{CpuBuffer, CpuDestination, Elem, Tensor};
use half::bf16;

impl Tensor {
    pub fn expert_linear_rows_into(
        &self,
        weight: &Tensor,
        indexes: &Tensor,
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        if self.shape().len() != 2
            || weight.shape().len() != 3
            || indexes.shape() != [self.shape()[0]]
            || weight.shape()[2] != self.shape()[1]
            || weight.shape()[0] == 0
            || weight.shape()[0] > u32::MAX as usize
        {
            return Err("expertLinearRows: invalid input shapes".into());
        }
        let CpuBuffer::U32(ids) = &indexes.buffer else {
            return Err("expertLinearRows: indices must be U32".into());
        };
        // Validate every route even when O=0, before writing any destination.
        for row in 0..self.shape()[0] {
            let expert = ids[indexes.layout.offset() + row * indexes.layout.strides()[0]];
            if expert as usize >= weight.shape()[0] {
                return Err("expertLinearRows: expert index is out of range".into());
            }
        }
        match (&self.buffer, &weight.buffer) {
            (CpuBuffer::F32(x), CpuBuffer::F32(w)) => {
                dot_into(self, weight, indexes, x, w, |v| v, |v| v, destination)
            }
            (CpuBuffer::BF16(x), CpuBuffer::BF16(w)) => dot_into(
                self,
                weight,
                indexes,
                x,
                w,
                bf16::to_f32,
                bf16::from_f32,
                destination,
            ),
            _ => Err("expertLinearRows: expected matching F32 or BF16 storage".into()),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dot_into<T: Elem>(
    x: &Tensor,
    weight: &Tensor,
    indexes: &Tensor,
    xs: &[T],
    ws: &[T],
    widen: impl Fn(T) -> f32,
    narrow: impl Fn(f32) -> T,
    destination: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let CpuBuffer::U32(ids) = &indexes.buffer else {
        unreachable!()
    };
    let (rows, inner, columns) = (x.shape()[0], x.shape()[1], weight.shape()[1]);
    destination.write::<T, _>("expertLinearRows", &[rows, columns], |output| {
        for row in 0..rows {
            let expert = ids[indexes.layout.offset() + row * indexes.layout.strides()[0]] as usize;
            let xb = x.layout.offset() + row * x.layout.strides()[0];
            let wb = weight.layout.offset() + expert * weight.layout.strides()[0];
            for column in 0..columns {
                let mut sum = 0.0f32;
                for i in 0..inner {
                    let a = widen(xs[xb + i * x.layout.strides()[1]]);
                    let b = widen(
                        ws[wb
                            + column * weight.layout.strides()[1]
                            + i * weight.layout.strides()[2]],
                    );
                    sum = a.mul_add(b, sum);
                }
                output[row * columns + column] = narrow(sum);
            }
        }
    })
}

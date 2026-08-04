//! Fused rotary embedding on Metal: one kernel per tensor instead of
//! ~15 composed ops (table build, narrows, muls, cat). Angles are
//! computed in-register (powf per element is cheaper than the table's
//! launches at these sizes); the backward is the same kernel with
//! negated angles (Rᵀ = R(−θ) for orthogonal rotations). CPU keeps the
//! composed reference path in lib.rs.

use crate::runtime::metal::run::MetalTensor;

/// Whether the fused rotary path can run: Metal, f32, even head dim.
pub fn is_supported(x: &MetalTensor) -> bool {
    x.dtype == crate::runtime::dtype::DType::F32 && x.layout.shape().last().unwrap_or(&1) % 2 == 0
}

#[cfg(target_os = "macos")]
pub use metal::rotary;

#[cfg(target_os = "macos")]
mod metal {
    use crate::runtime::metal::device::{set_buffer, set_bytes, MetalDevice, Pipeline};
    use crate::runtime::metal::run::MetalTensor;

    use objc2_metal::MTLComputeCommandEncoder;

    const NT: usize = 256;

    // One thread per (row, t, j): angle = (offset + t) * theta^(-2j/D);
    // GPT-NeoX half-split rotation. sign = -1 gives the transpose
    // rotation (the backward).
    fn source() -> &'static str {
        r#"
#include <metal_stdlib>
using namespace metal;

kernel void et_rotary(
    device const float* X [[buffer(0)]],
    device float* O [[buffer(1)]],
    device const float* offsets [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& rowsPerBatch [[buffer(5)]],
    constant float& theta [[buffer(6)]],
    constant float& sign [[buffer(7)]],
    constant uint& rows [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= rows * T * (D / 2)) { return; }
    const uint hd = D / 2;
    const uint j = tid % hd;
    const uint t = (tid / hd) % T;
    const uint row = tid / (hd * T);
    const uint batch = row / rowsPerBatch;
    const float angle = sign * (offsets[batch] + float(t)) * pow(theta, -2.0f * float(j) / float(D));
    const float c = cos(angle);
    const float s = sin(angle);
    const ulong base = ((ulong)row * T + t) * D;
    const float f = X[base + j];
    const float g = X[base + hd + j];
    O[base + j] = f * c - g * s;
    O[base + hd + j] = g * c + f * s;
}
"#
    }

    fn pipeline() -> crate::err::Res<Pipeline> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        "et_rotary".hash(&mut hasher);
        let key = hasher.finish();
        MetalDevice::get().compile_lazy(key, "et_rotary", || source().to_string())
    }

    /// x [.., T, D] -> R(sign·angles) x. `offsets` is one position
    /// offset per leading-dim-0 batch element (a single [0] for
    /// absolute positions).
    pub fn rotary(x: &MetalTensor, offsets: &[usize], theta: f64, sign: f32) -> crate::err::Res<MetalTensor> {
        let dims = x.layout.shape();
        let rank = dims.len();
        let (t, d) = (dims[rank - 2], dims[rank - 1]);
        let batch = dims[0];
        let rows: usize = dims[..rank - 2].iter().product();
        let rows_per_batch = rows / batch;
        // A single offset broadcasts over the batch.
        let offsets_vec: Vec<f32> = if offsets.len() == 1 {
            vec![offsets[0] as f32; batch]
        } else {
            if offsets.len() != batch {
                return Err(format!(
                    "rotary: {} offsets for batch {batch}",
                    offsets.len()
                ));
            }
            offsets.iter().map(|&o| o as f32).collect()
        };
        let xn = if x.layout.is_contiguous() {
            x.clone()
        } else {
            crate::runtime::metal::kernels::strided_copy(MetalDevice::get(), x)?
        };
        let offsets_buf = MetalDevice::get().alloc_with_data(&offsets_vec);
        let out_buf = MetalDevice::get().alloc(xn.numel().max(1), crate::runtime::dtype::DType::F32);
        let pipe = pipeline()?;
        MetalDevice::get().with_encoder(|e| {
            e.setComputePipelineState(pipe.as_raw());
            set_buffer(e, 0, &xn.buffer, xn.layout.offset() * 4);
            set_buffer(e, 2, &offsets_buf, 0);
            set_buffer(e, 1, &out_buf, 0);
            set_bytes(e, 3, &(t as u32));
            set_bytes(e, 4, &(d as u32));
            set_bytes(e, 5, &(rows_per_batch as u32));
            set_bytes(e, 6, &(theta as f32));
            set_bytes(e, 7, &sign);
            set_bytes(e, 8, &(rows as u32));
            let total = rows * t * (d / 2);
            e.dispatchThreads_threadsPerThreadgroup(
                objc2_metal::MTLSize {
                    width: total.div_ceil(NT) * NT,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: NT,
                    height: 1,
                    depth: 1,
                },
            );
        });
        Ok(MetalTensor {
            buffer: out_buf,
            layout: crate::runtime::layout::Layout::contiguous(dims.to_vec()),
            dtype: crate::runtime::dtype::DType::F32,
        })
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use crate::runtime::metal::device::MetalDevice;
    use crate::runtime::metal::run::MetalTensor as MT;

    #[test]
    fn kernel_matches_composed() {
        let dev = MetalDevice::get();
        for shape in [(1usize, 1usize, 2usize, 8usize), (1, 2, 4, 4), (2, 3, 5, 16)] {
            let (b, h, t, d) = shape;
            let data: Vec<f32> = (0..b * h * t * d).map(|i| i as f32).collect();
            let x = MT::from_f32(dev, data, vec![b, h, t, d]);
            let composed = crate::runtime::metal::composed::rotary_forward(&x, &[0], 10000.0, 1.0).unwrap();
            let fused = super::metal::rotary(&x, &[0], 10000.0, 1.0).unwrap();
            dev.synchronize();
            let a = composed.read_f32();
            let bb = fused.read_f32();
            let max_diff = a.iter().zip(bb.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(max_diff < 1e-3, "shape {shape:?} max diff {max_diff}");
            // Backward: transpose rotation inverts the forward.
            let back = super::metal::rotary(&fused, &[0], 10000.0, -1.0).unwrap();
            dev.synchronize();
            let x_vals = x.read_f32();
            let rt = back.read_f32();
            let max_diff = x_vals.iter().zip(rt.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(max_diff < 1e-3, "shape {shape:?} roundtrip max diff {max_diff}");
        }
    }
}

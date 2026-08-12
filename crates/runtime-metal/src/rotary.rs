//! Fused rotary embedding on Metal: one kernel per tensor instead of
//! ~15 composed ops (table build, narrows, muls, cat). Angles are
//! computed in-register (powf per element is cheaper than the table's
//! launches at these sizes); the backward is the same kernel with
//! negated angles (Rᵀ = R(−θ) for orthogonal rotations). CPU keeps the
//! composed reference path in lib.rs.

use crate::runtime::metal::run::MetalTensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RotaryRequirements {
    pub dtype: crate::runtime::dtype::DType,
    pub output_bytes: usize,
    pub state_next_bytes: usize,
    pub staging_bytes: usize,
    pub status_bytes: usize,
    pub scratch_bytes: usize,
    pub pipeline_count: usize,
}

pub fn requirements(
    dtype: crate::runtime::dtype::DType,
    shape: &[usize],
) -> crate::err::Res<RotaryRequirements> {
    if shape.len() < 2 || shape[shape.len() - 1] % 2 != 0 {
        return Err("rotary: expected rank >= 2 and an even head dimension".to_string());
    }
    if !matches!(
        dtype,
        crate::runtime::dtype::DType::F32
            | crate::runtime::dtype::DType::F16
            | crate::runtime::dtype::DType::BF16
    ) {
        return Err(format!("rotary: unsupported dtype {dtype:?}"));
    }
    let elements = shape
        .iter()
        .try_fold(1usize, |count, value| count.checked_mul(*value))
        .ok_or_else(|| "rotary: requirement element count overflow".to_string())?;
    Ok(RotaryRequirements {
        dtype,
        output_bytes: elements
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| "rotary: requirement byte size overflow".to_string())?,
        state_next_bytes: 0,
        staging_bytes: (if shape.len() == 2 { 1 } else { shape[0] })
            .checked_mul(crate::runtime::dtype::DType::F32.size_in_bytes())
            .ok_or_else(|| "rotary: requirement staging size overflow".to_string())?,
        status_bytes: 0,
        scratch_bytes: 0,
        pipeline_count: 1,
    })
}

pub fn rotary_requirements(
    dtype: crate::runtime::dtype::DType,
    shape: &[usize],
) -> crate::err::Res<RotaryRequirements> {
    requirements(dtype, shape)
}

#[cfg(test)]
mod requirement_tests {
    use super::*;
    use crate::runtime::dtype::DType;

    #[test]
    fn requirements_include_exact_offsets_staging() {
        let requirement = rotary_requirements(DType::F16, &[3, 2, 5, 16]).unwrap();
        assert_eq!(requirement.output_bytes, 3 * 2 * 5 * 16 * 2);
        assert_eq!(requirement.staging_bytes, 3 * 4);
        assert_eq!(
            (
                requirement.state_next_bytes,
                requirement.status_bytes,
                requirement.scratch_bytes,
            ),
            (0, 0, 0)
        );
        assert_eq!(requirement.pipeline_count, 1);
        assert!(rotary_requirements(DType::F32, &[1, 2, 3]).is_err());
        assert!(rotary_requirements(DType::F32, &[usize::MAX, 2, 2]).is_err());
    }
}

/// Whether the fused rotary path can run: Metal, f32, even head dim.
pub fn is_supported(x: &MetalTensor) -> bool {
    matches!(
        x.dtype,
        crate::runtime::dtype::DType::F32
            | crate::runtime::dtype::DType::F16
            | crate::runtime::dtype::DType::BF16
    ) && x.layout.shape().last().unwrap_or(&1) % 2 == 0
}

#[cfg(target_os = "macos")]
pub use metal::{rotary, rotary_into, warm, warm_exact, IntoResources};

#[cfg(target_os = "macos")]
mod metal {
    use crate::runtime::metal::device::{set_buffer, set_bytes, MetalDevice, Pipeline};
    use crate::runtime::metal::run::MetalTensor;
    use effect_torch_graph::RotaryLayout;

    use objc2_metal::MTLComputeCommandEncoder;

    #[derive(Clone, Copy)]
    pub struct IntoResources<'a> {
        pub staging: &'a [MetalTensor],
        pub status: &'a [MetalTensor],
        pub scratch: &'a [MetalTensor],
    }

    const NT: usize = 256;

    // One thread per (row, t, j): angle = (offset + t) * theta^(-2j/D);
    // GPT-NeoX half-split rotation. sign = -1 gives the transpose
    // rotation (the backward).
    fn source(ty: &str, layout: RotaryLayout) -> String {
        r#"
#include <metal_stdlib>
using namespace metal;

#define STOR {ty}
#define INTERLEAVED {interleaved}

kernel void et_rotary(
    device const STOR* X [[buffer(0)]],
    device STOR* O [[buffer(1)]],
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
    const uint first = INTERLEAVED ? 2 * j : j;
    const uint second = INTERLEAVED ? 2 * j + 1 : hd + j;
    const float f = float(X[base + first]);
    const float g = float(X[base + second]);
    O[base + first] = STOR(f * c - g * s);
    O[base + second] = STOR(g * c + f * s);
}
"#
        .replace("{ty}", ty)
        .replace(
            "{interleaved}",
            if layout == RotaryLayout::InterleavedPairs {
                "1"
            } else {
                "0"
            },
        )
    }

    fn pipeline_key(dtype: crate::runtime::dtype::DType, layout: RotaryLayout) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        ("et_rotary", dtype, layout).hash(&mut hasher);
        hasher.finish()
    }

    fn pipeline(
        dtype: crate::runtime::dtype::DType,
        layout: RotaryLayout,
    ) -> crate::err::Res<Pipeline> {
        let ty = match dtype {
            crate::runtime::dtype::DType::F32 => "float",
            crate::runtime::dtype::DType::F16 => "half",
            crate::runtime::dtype::DType::BF16 => "bfloat",
            other => return Err(format!("rotary: unsupported dtype {other:?}")),
        };
        MetalDevice::get().compile_lazy(pipeline_key(dtype, layout), "et_rotary", || {
            source(ty, layout)
        })
    }

    fn cached_pipeline(
        dtype: crate::runtime::dtype::DType,
        layout: RotaryLayout,
    ) -> crate::err::Res<Pipeline> {
        MetalDevice::get()
            .pipeline_cached(pipeline_key(dtype, layout))
            .ok_or_else(|| "rotary: exact pipeline is not warm; call warm".to_string())
    }

    pub fn warm(dtype: crate::runtime::dtype::DType, layout: RotaryLayout) -> crate::err::Res<()> {
        pipeline(dtype, layout)?;
        Ok(())
    }

    pub fn warm_exact(
        requirements: &super::RotaryRequirements,
        layout: RotaryLayout,
    ) -> crate::err::Res<()> {
        warm(requirements.dtype, layout)
    }

    /// x [.., T, D] -> R(sign·angles) x. `offsets` is one position
    /// offset per leading-dim-0 batch element (a single [0] for
    /// absolute positions).
    pub fn rotary_into(
        x: &MetalTensor,
        offsets: &[usize],
        theta: f64,
        sign: f32,
        layout: RotaryLayout,
        output: &MetalTensor,
        resources: IntoResources<'_>,
    ) -> crate::err::Res<()> {
        let dims = x.layout.shape();
        let rank = dims.len();
        if rank < 2 || dims[rank - 1] % 2 != 0 {
            return Err("rotary: expected rank >= 2 and an even head dimension".to_string());
        }
        let (t, d) = (dims[rank - 2], dims[rank - 1]);
        let batch = if rank == 2 { 1 } else { dims[0] };
        let rows: usize = dims[..rank - 2].iter().product();
        let rows_per_batch = rows / batch;
        if offsets.len() != 1 && offsets.len() != batch {
            return Err(format!(
                "rotary: {} offsets for batch {batch}",
                offsets.len()
            ));
        }
        if !x.layout.is_contiguous() {
            return Err("rotary: input must be contiguous before rotary_into".to_string());
        }
        if output.dtype != x.dtype
            || output.layout.shape() != dims
            || !output.layout.is_contiguous()
        {
            return Err(format!(
                "rotary: output must be contiguous {:?}:{:?}",
                dims, x.dtype
            ));
        }
        let output_end = output
            .layout
            .checked_max_index()
            .and_then(|elements| elements.checked_mul(output.dtype.size_in_bytes()))
            .ok_or_else(|| "rotary: output byte size overflow".to_string())?;
        if output_end > output.buffer.size {
            return Err("rotary: output view exceeds its buffer".to_string());
        }
        MetalDevice::get().mark_buffer_write(&output.buffer)?;
        if resources.staging.len() != 1
            || !resources.status.is_empty()
            || !resources.scratch.is_empty()
        {
            return Err(
                "rotary: exactly one offsets staging view and no status/scratch views are required"
                    .to_string(),
            );
        }
        let offsets_staging = &resources.staging[0];
        if offsets_staging.dtype != crate::runtime::dtype::DType::F32
            || offsets_staging.layout.shape() != [batch]
            || !offsets_staging.layout.is_contiguous()
        {
            return Err(format!(
                "rotary: offsets staging must be contiguous [{batch}]:F32"
            ));
        }
        let staging_end = offsets_staging
            .layout
            .checked_max_index()
            .and_then(|elements| {
                elements.checked_mul(crate::runtime::dtype::DType::F32.size_in_bytes())
            })
            .ok_or_else(|| "rotary: offsets staging byte size overflow".to_string())?;
        if staging_end > offsets_staging.buffer.size {
            return Err("rotary: offsets staging view exceeds its buffer".to_string());
        }
        let offset_ptr = unsafe {
            offsets_staging
                .buffer
                .contents_ptr()
                .cast::<f32>()
                .add(offsets_staging.layout.offset())
        };
        for index in 0..batch {
            let value = if offsets.len() == 1 {
                offsets[0]
            } else {
                offsets[index]
            };
            unsafe { offset_ptr.add(index).write(value as f32) };
        }
        let pipe = cached_pipeline(x.dtype, layout)?;
        MetalDevice::get().with_encoder(|e| {
            e.setComputePipelineState(pipe.as_raw());
            set_buffer(e, 0, &x.buffer, x.layout.offset() * x.dtype.size_in_bytes());
            set_buffer(
                e,
                2,
                &offsets_staging.buffer,
                offsets_staging.layout.offset() * crate::runtime::dtype::DType::F32.size_in_bytes(),
            );
            set_buffer(
                e,
                1,
                &output.buffer,
                output.layout.offset() * output.dtype.size_in_bytes(),
            );
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
        Ok(())
    }

    pub fn rotary(
        x: &MetalTensor,
        offsets: &[usize],
        theta: f64,
        sign: f32,
        layout: RotaryLayout,
    ) -> crate::err::Res<MetalTensor> {
        let dims = x.layout.shape().to_vec();
        let batch = if dims.len() == 2 { 1 } else { dims[0] };
        let x = if x.layout.is_contiguous() {
            x.clone()
        } else {
            crate::runtime::metal::kernels::strided_copy(MetalDevice::get(), x)?
        };
        warm(x.dtype, layout)?;
        let output = MetalTensor::empty(MetalDevice::get(), dims, x.dtype);
        let offsets_staging = MetalTensor::empty(
            MetalDevice::get(),
            vec![batch],
            crate::runtime::dtype::DType::F32,
        );
        rotary_into(
            &x,
            offsets,
            theta,
            sign,
            layout,
            &output,
            IntoResources {
                staging: std::slice::from_ref(&offsets_staging),
                status: &[],
                scratch: &[],
            },
        )?;
        Ok(output)
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use crate::runtime::metal::device::MetalDevice;
    use crate::runtime::metal::run::MetalTensor as MT;

    #[test]
    fn kernel_matches_composed() {
        let dev = MetalDevice::get();
        for shape in [
            (1usize, 1usize, 2usize, 8usize),
            (1, 2, 4, 4),
            (2, 3, 5, 16),
        ] {
            let (b, h, t, d) = shape;
            let data: Vec<f32> = (0..b * h * t * d).map(|i| i as f32).collect();
            let x = MT::from_f32(dev, data, vec![b, h, t, d]);
            crate::runtime::metal::indexing::warm_cat(
                &[&[b, h, t, d / 2], &[b, h, t, d / 2]],
                crate::runtime::dtype::DType::F32,
                3,
            )
            .unwrap();
            let composed =
                crate::runtime::metal::composed::rotary_forward(&x, &[0], 10000.0, 1.0).unwrap();
            let fused = super::metal::rotary(
                &x,
                &[0],
                10000.0,
                1.0,
                effect_torch_graph::RotaryLayout::HalfSplit,
            )
            .unwrap();
            dev.synchronize().unwrap();
            let a = composed.read_f32().unwrap();
            let bb = fused.read_f32().unwrap();
            let max_diff = a
                .iter()
                .zip(bb.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(max_diff < 1e-3, "shape {shape:?} max diff {max_diff}");
            // Backward: transpose rotation inverts the forward.
            let back = super::metal::rotary(
                &fused,
                &[0],
                10000.0,
                -1.0,
                effect_torch_graph::RotaryLayout::HalfSplit,
            )
            .unwrap();
            dev.synchronize().unwrap();
            let x_vals = x.read_f32().unwrap();
            let rt = back.read_f32().unwrap();
            let max_diff = x_vals
                .iter()
                .zip(rt.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_diff < 1e-3,
                "shape {shape:?} roundtrip max diff {max_diff}"
            );
        }
    }

    #[test]
    fn interleaved_pairs_use_the_pair_frequency_and_rank_two_batching() {
        let dev = MetalDevice::get();
        let x = MT::from_f32(
            dev,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 4],
        );
        let half = super::metal::rotary(
            &x,
            &[0],
            10_000.0,
            1.0,
            effect_torch_graph::RotaryLayout::HalfSplit,
        )
        .unwrap();
        let interleaved = super::metal::rotary(
            &x,
            &[0],
            10_000.0,
            1.0,
            effect_torch_graph::RotaryLayout::InterleavedPairs,
        )
        .unwrap();
        let roundtrip = super::metal::rotary(
            &interleaved,
            &[0],
            10_000.0,
            -1.0,
            effect_torch_graph::RotaryLayout::InterleavedPairs,
        )
        .unwrap();
        dev.synchronize().unwrap();

        let (s0, c0) = 1.0f32.sin_cos();
        let (s1, c1) = 0.01f32.sin_cos();
        let expected = vec![
            1.0,
            2.0,
            3.0,
            4.0,
            5.0 * c0 - 6.0 * s0,
            6.0 * c0 + 5.0 * s0,
            7.0 * c1 - 8.0 * s1,
            8.0 * c1 + 7.0 * s1,
        ];
        for (actual, expected) in interleaved.read_f32().unwrap().iter().zip(expected) {
            assert!((actual - expected).abs() < 2e-6);
        }
        assert_ne!(half.read_f32().unwrap(), interleaved.read_f32().unwrap());
        for (actual, expected) in roundtrip
            .read_f32()
            .unwrap()
            .iter()
            .zip(x.read_f32().unwrap())
        {
            assert!((actual - expected).abs() < 2e-6);
        }
    }

    #[test]
    fn into_matches_wrapper_and_uses_only_supplied_views() {
        let dev = MetalDevice::get();
        let shape = vec![2, 2, 3, 8];
        let data = (0..shape.iter().product::<usize>())
            .map(|index| index as f32 / 17.0)
            .collect();
        let x = MT::from_f32(dev, data, shape.clone());
        let expected = super::metal::rotary(
            &x,
            &[1, 4],
            10_000.0,
            1.0,
            effect_torch_graph::RotaryLayout::HalfSplit,
        )
        .unwrap();
        let requirements = super::rotary_requirements(x.dtype, &shape).unwrap();
        let output = MT::empty(dev, shape, x.dtype);
        let offsets = MT::empty(dev, vec![2], crate::runtime::dtype::DType::F32);
        super::metal::warm_exact(&requirements, effect_torch_graph::RotaryLayout::HalfSplit)
            .unwrap();
        dev.synchronize().unwrap();

        let _dispatch_guard = dev.begin_executable_dispatch().unwrap();
        super::metal::rotary_into(
            &x,
            &[1, 4],
            10_000.0,
            1.0,
            effect_torch_graph::RotaryLayout::HalfSplit,
            &output,
            super::metal::IntoResources {
                staging: std::slice::from_ref(&offsets),
                status: &[],
                scratch: &[],
            },
        )
        .unwrap();
        dev.synchronize().unwrap();
        let max_diff = expected
            .read_f32()
            .unwrap()
            .iter()
            .zip(output.read_f32().unwrap())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-6, "into parity diff {max_diff}");
    }
}

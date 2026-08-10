//! Fused linear layer on Metal: y = x·W + b in one gemm launch (the
//! addmm epilogue — bias rides the kernel's C source with ldc = 0,
//! broadcast over rows and batch) instead of matmul + broadcast-add.
//! CPU falls back to composed ops at the call site.

use crate::runtime::metal::run::MetalTensor;

/// Whether the fused linear path can run: Metal, f32, 2-D weight.
pub fn is_supported(x: &MetalTensor, weight: &MetalTensor) -> bool {
    matches!(
        x.dtype,
        crate::runtime::dtype::DType::F32 | crate::runtime::dtype::DType::BF16
    ) && weight.dtype == x.dtype
        && weight.layout.shape().len() == 2
}

#[cfg(target_os = "macos")]
pub use metal::{
    linear_forward, linear_forward_fused, linear_forward_fused_into,
    linear_forward_fused_requirements, linear_forward_into, linear_forward_requirements,
    linear_requirements, precompile_linear_forward, precompile_linear_forward_fused,
    LinearRequirements,
};

#[cfg(target_os = "macos")]
mod metal {
    use crate::runtime::dtype::DType;
    use crate::runtime::metal::device::MetalDevice;
    use crate::runtime::metal::gemm::{self, Epilogue, GemmRequirements, SplitKScratchRequirement};
    use crate::runtime::metal::run::MetalTensor;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct LinearRequirements {
        pub dtype: DType,
        pub rank: usize,
        pub batch: usize,
        pub m: usize,
        pub n: usize,
        pub k: usize,
        pub epilogue: Epilogue,
        pub mma: bool,
        pub output_elements: usize,
        pub output_bytes: usize,
        pub output_count: usize,
        pub split_k_scratch: Option<SplitKScratchRequirement>,
        pub gemm: GemmRequirements,
    }

    pub fn linear_requirements(
        dev: &MetalDevice,
        x_shape: &[usize],
        weight_shape: &[usize],
        dtype: DType,
        epilogue: Epilogue,
        mma: bool,
    ) -> Result<LinearRequirements, String> {
        if x_shape.len() < 2 {
            return Err("linear: input rank must be at least 2".to_string());
        }
        if weight_shape.len() != 2 {
            return Err("linear: weight must have rank 2".to_string());
        }
        let rank = x_shape.len();
        let (k, n) = (weight_shape[0], weight_shape[1]);
        if x_shape[rank - 1] != k {
            return Err(format!(
                "linear: input width {} does not match weight width {k}",
                x_shape[rank - 1]
            ));
        }
        let batch = x_shape[..rank - 2]
            .iter()
            .try_fold(1usize, |product, &dim| product.checked_mul(dim))
            .ok_or_else(|| "linear: batch size overflow".to_string())?;
        let m = x_shape[rank - 2];
        let gemm = gemm::gemm_requirements(dev, dtype, true, epilogue, batch, m, n, k, mma)?;
        Ok(LinearRequirements {
            dtype,
            rank,
            batch,
            m,
            n,
            k,
            epilogue,
            mma,
            output_elements: gemm.output_elements,
            output_bytes: gemm.output_bytes,
            output_count: gemm.output_count,
            split_k_scratch: gemm.split_k_scratch,
            gemm,
        })
    }

    pub fn linear_forward_requirements(
        dev: &MetalDevice,
        x_shape: &[usize],
        weight_shape: &[usize],
        dtype: DType,
        mma: bool,
    ) -> Result<LinearRequirements, String> {
        linear_requirements(dev, x_shape, weight_shape, dtype, Epilogue::None, mma)
    }

    fn epilogue(residual: bool, gelu: Option<(bool, bool)>) -> Result<Epilogue, String> {
        match (residual, gelu) {
            (true, None) => Ok(Epilogue::Residual),
            (false, Some((false, false))) => Ok(Epilogue::GeluErf),
            (false, Some((true, false))) => Ok(Epilogue::GeluTanh),
            (false, Some((false, true))) => Ok(Epilogue::GeluErfDual),
            (false, Some((true, true))) => Ok(Epilogue::GeluTanhDual),
            (true, Some(_)) => {
                Err("linear: residual and gelu epilogues cannot combine".to_string())
            }
            (false, None) => Ok(Epilogue::None),
        }
    }

    pub fn linear_forward_fused_requirements(
        dev: &MetalDevice,
        x_shape: &[usize],
        weight_shape: &[usize],
        dtype: DType,
        residual: bool,
        gelu: Option<(bool, bool)>,
        mma: bool,
    ) -> Result<LinearRequirements, String> {
        linear_requirements(
            dev,
            x_shape,
            weight_shape,
            dtype,
            epilogue(residual, gelu)?,
            mma,
        )
    }

    pub fn precompile_linear_forward_fused(
        dev: &MetalDevice,
        requirements: &LinearRequirements,
    ) -> Result<usize, String> {
        gemm::precompile_gemm_fused(dev, &requirements.gemm)
    }

    pub fn precompile_linear_forward(
        dev: &MetalDevice,
        requirements: &LinearRequirements,
    ) -> Result<usize, String> {
        if requirements.epilogue != Epilogue::None || requirements.output_count != 1 {
            return Err("linear: plain precompile received fused requirements".to_string());
        }
        precompile_linear_forward_fused(dev, requirements)
    }

    fn validate_output_shape(
        x: &MetalTensor,
        tensor: &MetalTensor,
        n: usize,
        name: &str,
    ) -> Result<(), String> {
        let x_shape = x.layout.shape();
        let shape = tensor.layout.shape();
        if shape.len() != x_shape.len()
            || shape[..shape.len() - 1] != x_shape[..x_shape.len() - 1]
            || shape[shape.len() - 1] != n
        {
            return Err(format!("linear: {name} shape does not match output shape"));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn linear_forward_fused_into(
        dev: &MetalDevice,
        x: &MetalTensor,
        weight: &MetalTensor,
        bias: &MetalTensor,
        residual: Option<&MetalTensor>,
        out: &MetalTensor,
        out2: Option<&MetalTensor>,
        split_k_scratch: Option<&MetalTensor>,
        requirements: &LinearRequirements,
    ) -> Result<(), String> {
        let x_shape = x.layout.shape();
        let weight_shape = weight.layout.shape();
        if x.dtype != requirements.dtype
            || x_shape.len() != requirements.rank
            || x_shape[x_shape.len() - 2] != requirements.m
            || x_shape[x_shape.len() - 1] != requirements.k
            || weight.dtype != requirements.dtype
            || weight_shape != [requirements.k, requirements.n]
        {
            return Err("linear: inputs do not match exact requirements".to_string());
        }
        let batch = x_shape[..x_shape.len() - 2]
            .iter()
            .try_fold(1usize, |product, &dim| product.checked_mul(dim))
            .ok_or_else(|| "linear: batch size overflow".to_string())?;
        if batch != requirements.batch {
            return Err("linear: input batch does not match exact requirements".to_string());
        }
        if bias.layout.shape() != [requirements.n] {
            return Err("linear: bias shape must exactly match output width".to_string());
        }
        validate_output_shape(x, out, requirements.n, "destination")?;
        if let Some(residual) = residual {
            validate_output_shape(x, residual, requirements.n, "residual")?;
        }
        if let Some(out2) = out2 {
            validate_output_shape(x, out2, requirements.n, "secondary destination")?;
        }
        let stride_a = requirements
            .m
            .checked_mul(requirements.k)
            .ok_or_else(|| "linear: input batch stride overflow".to_string())?;
        gemm::gemm_fused_into(
            dev,
            x,
            weight,
            Some(bias),
            residual,
            out,
            out2,
            split_k_scratch,
            stride_a,
            0,
            &requirements.gemm,
        )
    }

    pub fn linear_forward_into(
        dev: &MetalDevice,
        x: &MetalTensor,
        weight: &MetalTensor,
        bias: &MetalTensor,
        out: &MetalTensor,
        split_k_scratch: Option<&MetalTensor>,
        requirements: &LinearRequirements,
    ) -> Result<(), String> {
        if requirements.epilogue != Epilogue::None || requirements.output_count != 1 {
            return Err("linear_forward_into requires plain requirements".to_string());
        }
        linear_forward_fused_into(
            dev,
            x,
            weight,
            bias,
            None,
            out,
            None,
            split_k_scratch,
            requirements,
        )
    }

    pub fn linear_forward(
        x: &MetalTensor,
        weight: &MetalTensor,
        bias: &MetalTensor,
    ) -> crate::err::Res<MetalTensor> {
        let (out, extra) = linear_forward_fused(x, weight, bias, None, None)?;
        debug_assert!(extra.is_none());
        Ok(out)
    }

    /// y = x·W + b with an optional epilogue: a residual add (same-shape
    /// C source) or gelu (optionally dual-storing the pre-activation as
    /// the second return). One gemm launch either way.
    pub fn linear_forward_fused(
        x: &MetalTensor,
        weight: &MetalTensor,
        bias: &MetalTensor,
        residual: Option<&MetalTensor>,
        gelu: Option<(bool, bool)>,
    ) -> crate::err::Res<(MetalTensor, Option<MetalTensor>)> {
        let dims = x.layout.shape();
        let rank = dims.len();
        let dev = MetalDevice::get();
        let selected_epilogue = epilogue(residual.is_some(), gelu)?;
        let requirements = linear_requirements(
            dev,
            dims,
            weight.layout.shape(),
            x.dtype,
            selected_epilogue,
            crate::runtime::metal::device::mma_enabled(),
        )?;
        precompile_linear_forward_fused(dev, &requirements)?;
        let xn = if x.layout.is_contiguous() {
            x.clone()
        } else {
            crate::runtime::metal::kernels::strided_copy(dev, x)?
        };
        let wn = if weight.layout.is_contiguous() {
            weight.clone()
        } else {
            crate::runtime::metal::kernels::strided_copy(dev, weight)?
        };
        let bn = if bias.layout.is_contiguous() {
            bias.clone()
        } else {
            crate::runtime::metal::kernels::strided_copy(dev, bias)?
        };
        let rn = match residual {
            Some(r) if r.layout.is_contiguous() => Some(r.clone()),
            Some(r) => Some(crate::runtime::metal::kernels::strided_copy(dev, r)?),
            None => None,
        };
        let mut out_shape = dims.to_vec();
        out_shape[rank - 1] = requirements.n;
        let out = MetalTensor::empty(dev, out_shape.clone(), x.dtype);
        let out2 =
            (requirements.output_count == 2).then(|| MetalTensor::empty(dev, out_shape, x.dtype));
        let scratch = requirements
            .split_k_scratch
            .map(|scratch| MetalTensor::empty(dev, scratch.shape.to_vec(), scratch.dtype));
        linear_forward_fused_into(
            dev,
            &xn,
            &wn,
            &bn,
            rn.as_ref(),
            &out,
            out2.as_ref(),
            scratch.as_ref(),
            &requirements,
        )?;
        Ok((out, out2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::dtype::DType;
    use crate::runtime::metal::device::MetalDevice;
    use crate::runtime::metal::run::MetalTensor;

    #[test]
    fn linear_into_matches_allocating_plain_residual_and_gelu_outputs() {
        let dev = MetalDevice::get();
        let (batch, m, k, n) = (2usize, 5usize, 7usize, 11usize);
        let x: Vec<f32> = (0..batch * m * k)
            .map(|i| (i as f32 * 0.17).sin())
            .collect();
        let weight: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.31).cos()).collect();
        let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.2 - 0.7).collect();
        let residual: Vec<f32> = (0..batch * m * n)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let tx = MetalTensor::from_f32(dev, x, vec![batch, m, k]);
        let tw = MetalTensor::from_f32(dev, weight, vec![k, n]);
        let tb = MetalTensor::from_f32(dev, bias, vec![n]);
        let tr = MetalTensor::from_f32(dev, residual, vec![batch, m, n]);

        for (has_residual, gelu) in [
            (false, None),
            (true, None),
            (false, Some((false, false))),
            (false, Some((true, true))),
        ] {
            let requirements = linear_forward_fused_requirements(
                dev,
                tx.layout.shape(),
                tw.layout.shape(),
                DType::F32,
                has_residual,
                gelu,
                false,
            )
            .unwrap();
            precompile_linear_forward_fused(dev, &requirements).unwrap();
            let destination = MetalTensor::empty(dev, vec![batch, m, n], DType::F32);
            let destination2 = (requirements.output_count == 2)
                .then(|| MetalTensor::empty(dev, vec![batch, m, n], DType::F32));
            let residual = has_residual.then_some(&tr);
            linear_forward_fused_into(
                dev,
                &tx,
                &tw,
                &tb,
                residual,
                &destination,
                destination2.as_ref(),
                None,
                &requirements,
            )
            .unwrap();
            let (allocating, allocating2) =
                linear_forward_fused(&tx, &tw, &tb, residual, gelu).unwrap();
            dev.synchronize().unwrap();
            let got = destination.read_f32().unwrap();
            let want = allocating.read_f32().unwrap();
            for (got, want) in got.iter().zip(&want) {
                assert!((got - want).abs() < 1e-5, "{got} vs {want}");
            }
            match (destination2, allocating2) {
                (Some(got), Some(want)) => {
                    let got = got.read_f32().unwrap();
                    let want = want.read_f32().unwrap();
                    for (got, want) in got.iter().zip(&want) {
                        assert!((got - want).abs() < 1e-5, "{got} vs {want}");
                    }
                }
                (None, None) => {}
                _ => panic!("linear dual output mismatch"),
            }
        }
    }

    #[test]
    fn linear_into_allocates_no_device_buffer() {
        let dev = MetalDevice::new(0).unwrap();
        let (m, k, n) = (4usize, 6usize, 5usize);
        let x = MetalTensor::from_f32(&dev, vec![0.25; m * k], vec![m, k]);
        let weight = MetalTensor::from_f32(&dev, vec![0.5; k * n], vec![k, n]);
        let bias = MetalTensor::from_f32(&dev, vec![0.1; n], vec![n]);
        let output = MetalTensor::empty(&dev, vec![m, n], DType::F32);
        let requirements = linear_forward_requirements(
            &dev,
            x.layout.shape(),
            weight.layout.shape(),
            DType::F32,
            false,
        )
        .unwrap();
        precompile_linear_forward(&dev, &requirements).unwrap();

        let _dispatch_guard = dev.begin_executable_dispatch().unwrap();
        let result = linear_forward_into(&dev, &x, &weight, &bias, &output, None, &requirements);
        result.unwrap();
        dev.synchronize().unwrap();
    }
}

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
pub use metal::linear_forward;

#[cfg(target_os = "macos")]
pub use metal::linear_forward_fused;

#[cfg(target_os = "macos")]
mod metal {
    use crate::runtime::metal::device::MetalDevice;
    use crate::runtime::metal::gemm::Epilogue;
    use crate::runtime::metal::run::MetalTensor;

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
        let (k, n) = (weight.layout.shape()[0], weight.layout.shape()[1]);
        let m = dims[rank - 2];
        let b: usize = dims[..rank - 2].iter().product();
        let flat = |t: &MetalTensor, shape: Vec<usize>| MetalTensor {
            buffer: t.buffer.clone(),
            layout: crate::runtime::layout::Layout::contiguous(shape),
            dtype: t.dtype,
        };
        let xr = flat(x, vec![b, m, k]);
        let xn = if xr.layout.is_contiguous() {
            xr
        } else {
            crate::runtime::metal::kernels::strided_copy(MetalDevice::get(), &xr)?
        };
        let wn = if weight.layout.is_contiguous() {
            weight.clone()
        } else {
            crate::runtime::metal::kernels::strided_copy(MetalDevice::get(), weight)?
        };
        let rn = match residual {
            Some(r) => {
                let rr = flat(r, vec![b, m, n]);
                Some(if rr.layout.is_contiguous() {
                    rr
                } else {
                    crate::runtime::metal::kernels::strided_copy(MetalDevice::get(), &rr)?
                })
            }
            None => None,
        };
        let epilogue = match (rn.is_some(), gelu) {
            (true, None) => Epilogue::Residual,
            (false, Some((false, false))) => Epilogue::GeluErf,
            (false, Some((true, false))) => Epilogue::GeluTanh,
            (false, Some((false, true))) => Epilogue::GeluErfDual,
            (false, Some((true, true))) => Epilogue::GeluTanhDual,
            (true, Some(_)) => {
                return Err("linear: residual and gelu epilogues cannot combine".to_string())
            }
            (false, None) => Epilogue::None,
        };
        let (out, out2) = crate::runtime::metal::gemm::gemm_fused(
            MetalDevice::get(),
            &xn,
            &wn,
            Some(bias),
            rn.as_ref(),
            epilogue,
            b,
            m,
            n,
            k,
            m * k,
            0,
        )?;
        let mut out_shape = dims.to_vec();
        out_shape[rank - 1] = n;
        let wrap = |t: MetalTensor| MetalTensor {
            buffer: t.buffer,
            layout: crate::runtime::layout::Layout::contiguous(out_shape.clone()),
            dtype: t.dtype,
        };
        Ok((wrap(out), out2.map(wrap)))
    }
}

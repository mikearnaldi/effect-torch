//! Fused linear layer on Metal: y = x·W + b in one gemm launch (the
//! addmm epilogue — bias rides the kernel's C source with ldc = 0,
//! broadcast over rows and batch) instead of matmul + broadcast-add.
//! CPU falls back to composed ops at the call site.

use crate::runtime::metal::run::MetalTensor;

/// Whether the fused linear path can run: Metal, f32, 2-D weight.
pub fn is_supported(x: &MetalTensor, weight: &MetalTensor) -> bool {
    x.dtype == crate::runtime::dtype::DType::F32
        && weight.dtype == crate::runtime::dtype::DType::F32
        && weight.layout.shape().len() == 2
}

#[cfg(target_os = "macos")]
pub use metal::linear_forward;

#[cfg(target_os = "macos")]
mod metal {
    use crate::runtime::metal::device::MetalDevice;
    use crate::runtime::metal::run::MetalTensor;


    pub fn linear_forward(x: &MetalTensor, weight: &MetalTensor, bias: &MetalTensor) -> crate::err::Res<MetalTensor> {
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
        let out = crate::runtime::metal::gemm::gemm(
            MetalDevice::get(),
            &xn,
            &wn,
            Some(bias),
            b,
            m,
            n,
            k,
            m * k,
            0,
        )?;
        let mut out_shape = dims.to_vec();
        out_shape[rank - 1] = n;
        Ok(MetalTensor {
            buffer: out.buffer,
            layout: crate::runtime::layout::Layout::contiguous(out_shape),
            dtype: out.dtype,
        })
    }
}


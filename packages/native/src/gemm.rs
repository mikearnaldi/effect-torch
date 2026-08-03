//! Fused linear layer on Metal: y = x·W + b in one gemm launch (the
//! addmm epilogue — bias rides the kernel's C source with ldc = 0,
//! broadcast over rows and batch) instead of matmul + broadcast-add.
//! CPU falls back to composed ops at the call site.

use candle_core::{DType, Device, Tensor};

/// Whether the fused linear path can run: Metal, f32, 2-D weight.
pub fn is_supported(x: &Tensor, weight: &Tensor) -> bool {
    matches!(x.device(), Device::Metal(_))
        && x.dtype() == DType::F32
        && weight.dtype() == DType::F32
        && weight.rank() == 2
}

#[cfg(target_os = "macos")]
pub use metal::linear_forward;

#[cfg(target_os = "macos")]
mod metal {
    use crate::bridge;
    use crate::runtime::metal::device::MetalDevice;
    use candle_core::{DType, Tensor};

    pub fn linear_forward(x: &Tensor, weight: &Tensor, bias: &Tensor) -> candle_core::Result<Tensor> {
        let dims = x.dims();
        let rank = dims.len();
        let (k, n) = weight.dims2()?;
        let m = dims[rank - 2];
        let b: usize = dims[..rank - 2].iter().product();
        let device = x.device();
        let mdev = device.as_metal_device()?;
        device.synchronize()?;
        let xr = x.reshape((b, m, k))?;
        let xn = bridge::metal::wrap(&xr)?;
        let xn = if xn.layout.is_contiguous() {
            xn
        } else {
            crate::runtime::metal::kernels::strided_copy(MetalDevice::get(), &xn)
                .map_err(candle_core::Error::Msg)?
        };
        let wn = bridge::metal::wrap(weight)?;
        let wn = if wn.layout.is_contiguous() {
            wn
        } else {
            crate::runtime::metal::kernels::strided_copy(MetalDevice::get(), &wn)
                .map_err(candle_core::Error::Msg)?
        };
        let bn = bridge::metal::wrap(bias)?;
        let out = crate::runtime::metal::gemm::gemm(
            MetalDevice::get(),
            &xn,
            &wn,
            Some(&bn),
            b,
            m,
            n,
            k,
            m * k,
            0,
        )
        .map_err(candle_core::Error::Msg)?;
        MetalDevice::get().synchronize();
        let mut out_shape = dims.to_vec();
        out_shape[rank - 1] = n;
        bridge::metal::unwrap(&out.buffer, out_shape, DType::F32, mdev)
    }
}

#[cfg(not(target_os = "macos"))]
mod metal {
    use candle_core::Tensor;
    pub fn linear_forward(_x: &Tensor, _w: &Tensor, _b: &Tensor) -> candle_core::Result<Tensor> {
        unreachable!("fused linear is Metal-only")
    }
}

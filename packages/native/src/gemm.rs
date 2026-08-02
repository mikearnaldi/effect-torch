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
    use candle_core::{DType, MetalStorage, Storage, Tensor};

    pub fn linear_forward(x: &Tensor, weight: &Tensor, bias: &Tensor) -> candle_core::Result<Tensor> {
        let dims = x.dims();
        let rank = dims.len();
        let (k, n) = weight.dims2()?;
        let m = dims[rank - 2];
        let b: usize = dims[..rank - 2].iter().product();
        let device = x.device();
        let mdev = device.as_metal_device()?;
        // Flatten leading dims: the gemm's batch path is single-stride.
        let x = x.contiguous()?.reshape((b, m, k))?;
        let weight = weight.contiguous()?;
        let bias = bias.contiguous()?;
        let out_buf = mdev.new_buffer(b * m * n, DType::F32, "linear")?;
        {
            let (x_storage, x_layout) = x.storage_and_layout();
            let (w_storage, w_layout) = weight.storage_and_layout();
            let (b_storage, b_layout) = bias.storage_and_layout();
            let (xm, wm, bm) = match (&*x_storage, &*w_storage, &*b_storage) {
                (Storage::Metal(xm), Storage::Metal(wm), Storage::Metal(bm)) => (xm, wm, bm),
                _ => {
                    return Err(candle_core::Error::Msg(
                        "linear: expected Metal storage".to_string(),
                    ))
                }
            };
            let encoder = mdev.command_encoder()?;
            candle_metal_kernels::call_mlx_gemm_bias(
                mdev.device(),
                &encoder,
                mdev.kernels(),
                candle_metal_kernels::GemmDType::F32,
                (b, m, n, k),
                x_layout.stride(),
                x_layout.start_offset() * DType::F32.size_in_bytes(),
                xm.buffer(),
                w_layout.stride(),
                w_layout.start_offset() * DType::F32.size_in_bytes(),
                wm.buffer(),
                bm.buffer(),
                &out_buf,
            )
            .map_err(|e| candle_core::Error::Msg(format!("linear: {e}")))?;
            let _ = b_layout;
        }
        let mut out_shape = dims.to_vec();
        out_shape[rank - 1] = n;
        Ok(Tensor::from_storage(
            Storage::Metal(MetalStorage::new(out_buf, mdev.clone(), b * m * n, DType::F32)),
            out_shape,
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(not(target_os = "macos"))]
mod metal {
    use candle_core::Tensor;
    pub fn linear_forward(_x: &Tensor, _w: &Tensor, _b: &Tensor) -> candle_core::Result<Tensor> {
        unreachable!("fused linear is Metal-only")
    }
}

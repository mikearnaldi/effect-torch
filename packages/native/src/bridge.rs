use crate::runtime;
use candle_core::{DType, Tensor};

#[cfg(target_os = "macos")]
pub mod metal {
    use candle_core::{DType, MetalStorage, Storage, Tensor};
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::MTLBuffer;
    use std::sync::Arc;

    pub fn wrap(t: &Tensor) -> candle_core::Result<crate::runtime::metal::run::MetalTensor> {
        let (storage, layout) = t.storage_and_layout();
        let Storage::Metal(m) = &*storage else {
            return Err(candle_core::Error::Msg(
                "bridge::metal: expected Metal storage".to_string(),
            ));
        };
        let buf = m.buffer();
        let raw: &ProtocolObject<dyn MTLBuffer> = buf.as_ref();
        let retained = unsafe { Retained::retain(raw as *const _ as *mut _) }
            .ok_or_else(|| candle_core::Error::Msg("bridge::metal: null buffer".to_string()))?;
        let dtype = crate::bridge::dtype_to_native(t.dtype());
        let buffer = crate::runtime::metal::device::Buffer::from_raw(retained, buf.length());
        Ok(crate::runtime::metal::run::MetalTensor {
            buffer: Arc::new(buffer),
            layout: crate::runtime::layout::Layout::new(
                t.shape().dims().to_vec(),
                layout.stride().to_vec(),
                layout.start_offset(),
            ),
            dtype,
        })
    }

    pub fn unwrap(
        buffer: &Arc<crate::runtime::metal::device::Buffer>,
        shape: Vec<usize>,
        dtype: DType,
        device: &candle_core::MetalDevice,
    ) -> candle_core::Result<Tensor> {
        let raw: &ProtocolObject<dyn MTLBuffer> = buffer.as_raw();
        let retained = unsafe { Retained::retain(raw as *const _ as *mut _) }
            .ok_or_else(|| candle_core::Error::Msg("bridge::metal: null buffer".to_string()))?;
        let buf = candle_metal_kernels::metal::Buffer::new(retained);
        let count: usize = shape.iter().product();
        let storage = MetalStorage::new(std::sync::Arc::new(buf), device.clone(), count, dtype);
        Ok(Tensor::from_storage(
            Storage::Metal(storage),
            shape,
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }

    fn raw_of(t: &Tensor) -> candle_core::Result<(Retained<ProtocolObject<dyn MTLBuffer>>, usize, usize)> {
        let (storage, layout) = t.storage_and_layout();
        let Storage::Metal(m) = &*storage else {
            return Err(candle_core::Error::Msg(
                "bridge::metal: expected Metal storage".to_string(),
            ));
        };
        let buf = m.buffer();
        let raw: &ProtocolObject<dyn MTLBuffer> = buf.as_ref();
        let retained = unsafe { Retained::retain(raw as *const _ as *mut _) }
            .ok_or_else(|| candle_core::Error::Msg("bridge::metal: null buffer".to_string()))?;
        Ok((retained, layout.start_offset(), buf.length()))
    }
}

pub fn dtype_to_native(d: DType) -> runtime::dtype::DType {
    match d {
        DType::F32 => runtime::dtype::DType::F32,
        DType::F64 => runtime::dtype::DType::F64,
        DType::F16 => runtime::dtype::DType::F16,
        DType::BF16 => runtime::dtype::DType::BF16,
        DType::U8 => runtime::dtype::DType::U8,
        DType::U32 => runtime::dtype::DType::U32,
        DType::I64 => runtime::dtype::DType::I64,
        other => panic!("unsupported candle dtype: {other:?}"),
    }
}

pub fn dtype_from_native(d: runtime::dtype::DType) -> DType {
    match d {
        runtime::dtype::DType::F32 => DType::F32,
        runtime::dtype::DType::F64 => DType::F64,
        runtime::dtype::DType::F16 => DType::F16,
        runtime::dtype::DType::BF16 => DType::BF16,
        runtime::dtype::DType::U8 => DType::U8,
        runtime::dtype::DType::U32 => DType::U32,
        runtime::dtype::DType::I64 => DType::I64,
    }
}

pub fn from_candle(t: &Tensor) -> candle_core::Result<runtime::cpu::Tensor> {
    if !t.device().is_cpu() {
        return Err(candle_core::Error::Msg(
            "bridge::from_candle: expected a CPU tensor".to_string(),
        ));
    }
    let t = t.contiguous()?;
    let shape = t.shape().dims().to_vec();
    let out = match t.dtype() {
        DType::F32 => runtime::cpu::Tensor::from_vec(t.flatten_all()?.to_vec1::<f32>()?, shape),
        DType::F64 => runtime::cpu::Tensor::from_vec(t.flatten_all()?.to_vec1::<f64>()?, shape),
        DType::F16 => runtime::cpu::Tensor::from_vec(t.flatten_all()?.to_vec1::<half::f16>()?, shape),
        DType::BF16 => runtime::cpu::Tensor::from_vec(t.flatten_all()?.to_vec1::<half::bf16>()?, shape),
        DType::U8 => runtime::cpu::Tensor::from_vec(t.flatten_all()?.to_vec1::<u8>()?, shape),
        DType::U32 => runtime::cpu::Tensor::from_vec(t.flatten_all()?.to_vec1::<u32>()?, shape),
        DType::I64 => runtime::cpu::Tensor::from_vec(t.flatten_all()?.to_vec1::<i64>()?, shape),
        other => {
            return Err(candle_core::Error::Msg(format!(
                "bridge::from_candle: unsupported dtype {other:?}"
            )))
        }
    };
    Ok(out)
}

pub fn to_candle(t: &runtime::cpu::Tensor) -> candle_core::Result<Tensor> {
    let shape = t.shape().to_vec();
    let out = match &t.buffer {
        runtime::cpu::CpuBuffer::F32(v) => Tensor::from_vec(v.as_slice().to_vec(), shape, &candle_core::Device::Cpu)?,
        runtime::cpu::CpuBuffer::F64(v) => Tensor::from_vec(v.as_slice().to_vec(), shape, &candle_core::Device::Cpu)?,
        runtime::cpu::CpuBuffer::F16(v) => Tensor::from_vec(v.as_slice().to_vec(), shape, &candle_core::Device::Cpu)?,
        runtime::cpu::CpuBuffer::BF16(v) => Tensor::from_vec(v.as_slice().to_vec(), shape, &candle_core::Device::Cpu)?,
        runtime::cpu::CpuBuffer::U8(v) => Tensor::from_vec(v.as_slice().to_vec(), shape, &candle_core::Device::Cpu)?,
        runtime::cpu::CpuBuffer::U32(v) => Tensor::from_vec(v.as_slice().to_vec(), shape, &candle_core::Device::Cpu)?,
        runtime::cpu::CpuBuffer::I64(v) => Tensor::from_vec(v.as_slice().to_vec(), shape, &candle_core::Device::Cpu)?,
    };
    Ok(out)
}

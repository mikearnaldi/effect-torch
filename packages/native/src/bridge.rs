use crate::runtime;
use candle_core::{DType, Tensor};

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

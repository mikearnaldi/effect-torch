use crate::dev::Device;
use crate::err::{self, Res};
use crate::runtime;

#[derive(Clone)]
pub enum Val {
    Cpu(runtime::cpu::Tensor),
    Metal(runtime::metal::run::MetalTensor),
}

impl Val {
    pub fn is_cpu(&self) -> bool {
        matches!(self, Val::Cpu(_))
    }

    pub fn is_metal(&self) -> bool {
        matches!(self, Val::Metal(_))
    }

    pub fn device(&self) -> Device {
        match self {
            Val::Cpu(_) => Device::Cpu,
            Val::Metal(_) => Device::Metal,
        }
    }

    pub fn as_cpu(&self) -> Res<&runtime::cpu::Tensor> {
        match self {
            Val::Cpu(t) => Ok(t),
            Val::Metal(_) => err::err("device mismatch: expected CPU value"),
        }
    }

    pub fn as_metal(&self) -> Res<&runtime::metal::run::MetalTensor> {
        match self {
            Val::Metal(t) => Ok(t),
            Val::Cpu(_) => err::err("device mismatch: expected Metal value"),
        }
    }

    pub fn dtype(&self) -> runtime::dtype::DType {
        match self {
            Val::Cpu(t) => t.dtype(),
            Val::Metal(t) => t.dtype,
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        match self {
            Val::Cpu(t) => t.shape().to_vec(),
            Val::Metal(t) => t.layout.shape().to_vec(),
        }
    }

    pub fn numel(&self) -> usize {
        match self {
            Val::Cpu(t) => t.numel(),
            Val::Metal(t) => t.numel(),
        }
    }

    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    pub fn byte_size(&self) -> usize {
        self.numel() * self.dtype().size_in_bytes()
    }

    pub fn synchronize(&self) {
        if self.is_metal() {
            runtime::metal::device::MetalDevice::get().synchronize();
        }
    }

    pub fn to_f32_vec(&self) -> Res<Vec<f32>> {
        match self {
            Val::Cpu(t) => {
                let t = t.cast(runtime::dtype::DType::F32).contiguous();
                let runtime::cpu::CpuBuffer::F32(v) = &t.buffer else { unreachable!() };
                Ok(v.as_slice().to_vec())
            }
            Val::Metal(t) => {
                let dev = runtime::metal::device::MetalDevice::get();
                let t = &runtime::metal::kernels::strided_copy(dev, t)?;
                let t = if t.dtype == runtime::dtype::DType::F32 {
                    t.clone()
                } else {
                    runtime::metal::kernels::cast(dev, t, runtime::dtype::DType::F32)?
                };
                dev.synchronize();
                Ok(t.read_f32())
            }
        }
    }

    pub fn to_f64_vec(&self) -> Res<Vec<f64>> {
        match self {
            Val::Cpu(t) => {
                let t = t.cast(runtime::dtype::DType::F64).contiguous();
                let runtime::cpu::CpuBuffer::F64(v) = &t.buffer else { unreachable!() };
                Ok(v.as_slice().to_vec())
            }
            Val::Metal(_) => {
                let f32s = self.to_f32_vec()?;
                Ok(f32s.into_iter().map(|v| v as f64).collect())
            }
        }
    }

    pub fn to_u32_vec(&self) -> Res<Vec<u32>> {
        match self {
            Val::Cpu(t) => {
                let t = t.cast(runtime::dtype::DType::U32).contiguous();
                let runtime::cpu::CpuBuffer::U32(v) = &t.buffer else { unreachable!() };
                Ok(v.as_slice().to_vec())
            }
            Val::Metal(t) => {
                let dev = runtime::metal::device::MetalDevice::get();
                let t = &runtime::metal::kernels::strided_copy(dev, t)?;
                dev.synchronize();
                let n = t.numel();
                let size = t.dtype.size_in_bytes();
                let mut out = Vec::with_capacity(n);
                let ptr = unsafe { t.buffer.contents_ptr().cast::<u8>().add(t.layout.offset() * size) };
                let bytes = unsafe { std::slice::from_raw_parts(ptr, n * size) };
                match t.dtype {
                    runtime::dtype::DType::U32 => {
                        out.extend(bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])));
                    }
                    runtime::dtype::DType::U8 => {
                        out.extend(bytes.iter().map(|&b| b as u32));
                    }
                    runtime::dtype::DType::I64 => {
                        out.extend(bytes.chunks_exact(8).map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as u32));
                    }
                    _ => return err::err("to_u32_vec: dtype must be u8/u32/i64"),
                }
                Ok(out)
            }
        }
    }

    pub fn to_i64_vec(&self) -> Res<Vec<i64>> {
        match self {
            Val::Cpu(t) => {
                let t = t.cast(runtime::dtype::DType::I64).contiguous();
                let runtime::cpu::CpuBuffer::I64(v) = &t.buffer else { unreachable!() };
                Ok(v.as_slice().to_vec())
            }
            Val::Metal(_) => Ok(self.to_u32_vec()?.into_iter().map(|v| v as i64).collect()),
        }
    }

    pub fn to_u8_vec(&self) -> Res<Vec<u8>> {
        match self {
            Val::Cpu(t) => {
                let t = t.cast(runtime::dtype::DType::U8).contiguous();
                let runtime::cpu::CpuBuffer::U8(v) = &t.buffer else { unreachable!() };
                Ok(v.as_slice().to_vec())
            }
            Val::Metal(t) => {
                let dev = runtime::metal::device::MetalDevice::get();
                let t = &runtime::metal::kernels::strided_copy(dev, t)?;
                dev.synchronize();
                let n = t.numel();
                let ptr = unsafe { t.buffer.contents_ptr().cast::<u8>().add(t.layout.offset() * t.dtype.size_in_bytes()) };
                let bytes = unsafe { std::slice::from_raw_parts(ptr, n * t.dtype.size_in_bytes()) };
                match t.dtype {
                    runtime::dtype::DType::U8 => Ok(bytes.to_vec()),
                    runtime::dtype::DType::U32 => Ok(bytes.chunks_exact(4).map(|c| c[0]).collect()),
                    _ => err::err("to_u8_vec: dtype must be u8/u32"),
                }
            }
        }
    }
}

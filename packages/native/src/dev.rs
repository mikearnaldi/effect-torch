#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Metal,
}

impl Device {
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    pub fn is_metal(&self) -> bool {
        matches!(self, Device::Metal)
    }

    pub fn same_device(&self, other: &Device) -> bool {
        self == other
    }

    pub fn name(&self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Metal => "metal",
        }
    }
}

use crate::runtime::dtype::DType as ND;

pub fn dtype_name(d: ND) -> &'static str {
    d.name()
}

pub fn dtype_of_name(name: &str) -> Option<ND> {
    match name {
        "f32" => Some(ND::F32),
        "f64" => Some(ND::F64),
        "f16" => Some(ND::F16),
        "bf16" => Some(ND::BF16),
        "u8" => Some(ND::U8),
        "u32" => Some(ND::U32),
        "i64" => Some(ND::I64),
        _ => None,
    }
}

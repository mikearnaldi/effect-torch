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

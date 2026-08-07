use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    F16,
    BF16,
    U8,
    U32,
    I64,
}

impl DType {
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::U8 => 1,
            DType::U32 => 4,
            DType::I64 => 8,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::U8 => "u8",
            DType::U32 => "u32",
            DType::I64 => "i64",
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F64 | DType::F16 | DType::BF16)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

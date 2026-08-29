//! Element types shared by every backend.
//!
//! [`DType`] is a closed enumeration, so backends report their supported
//! types through [`Capabilities`](crate::Capabilities). Quantized formats
//! such as [`GgmlKQuant`](crate::GgmlKQuant) are storage encodings, not
//! compute dtypes, and do not appear here.

use std::fmt;

/// Scalar element type of a tensor or buffer.
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
    /// Size of one element in bytes.
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

    /// Stable lowercase name of the dtype (e.g. `"f32"`, `"bf16"`).
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

    /// Returns `true` for `F32`, `F64`, `F16`, and `BF16`.
    pub fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F64 | DType::F16 | DType::BF16)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

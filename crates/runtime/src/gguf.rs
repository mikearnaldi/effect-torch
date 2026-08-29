//! Strict, resource-limited, cancellable parsing of GGUF v3 model files.
//!
//! [`parse_gguf`] reads the magic, version, metadata table, and tensor
//! catalog. It returns a [`GgufFile`] with each tensor's logical shape,
//! on-disk format, and absolute byte range without reading tensor payloads.
//! [`read_gguf_tensor_into`] uses positioned reads to load one payload into
//! a caller-owned buffer. Multiple threads can therefore read the same file.
//!
//! # Supported subset
//!
//! - GGUF version 3 only (`GGUF_VERSION`) with the spec's little-endian
//!   encoding.
//! - Tensor formats: `F32` and the K-quants `Q2_K`/`Q3_K`/`Q4_K`/`Q5_K`/
//!   `Q6_K` ([`GgufTensorFormat`]). The parser rejects other GGML type codes.
//!   If any tensor is quantized, `general.quantization_version` must be
//!   present, be `u32`, and equal 2.
//! - `general.architecture` must be a non-empty string.
//!   `general.alignment` is optional. When present, it must be a `u32` power
//!   of two. It defaults to [`DEFAULT_ALIGNMENT`] (32).
//! - Every GGUF scalar metadata type and homogeneous arrays. Nested arrays
//!   are rejected.
//!
//! # Tensor geometry
//!
//! GGUF stores dimensions least-major first. Descriptors reverse them into
//! the runtime's row-major [`GgufTensorDescriptor::logical_shape`]. For a
//! K-quant tensor, the logical column count in GGUF dimension 0 must be
//! divisible by the 256-element block. Its
//! [`GgufTensorDescriptor::physical_shape`] is
//! `[rows, encoded_row_bytes]`, the packed byte layout. For F32, the physical
//! and logical shapes match, and `byte_len` is `numel * 4`.
//!
//! # Resource limits and validation
//!
//! The header cannot exceed [`MAX_HEADER_BYTES`]. `MAX_*` constants cap
//! metadata and tensor counts, key and name lengths, string lengths, and array
//! lengths. Before allocating, the parser also checks each value against the
//! remaining file size. It checks every length, offset, and product for
//! overflow.
//! 64-bit metadata integers and tensor dimensions must fit the JavaScript
//! safe-integer range ([`JS_SAFE_INTEGER`], 2^53 − 1) because the Node.js
//! embedding receives them. Tensor data offsets must use the file's
//! alignment. Tensor ranges must stay inside the file and cannot overlap.
//! Metadata keys and tensor names must be unique and non-empty. Tensor rank
//! must be in `1..=4`.
//!
//! # Cancellation
//!
//! Both entry points accept an optional [`CancellationFlag`]. They poll it
//! before the first I/O and between header or payload chunks. A set flag
//! returns [`GgufParseError::Cancelled`]. Cancellation may leave part of the
//! caller's output buffer written.

use crate::CancellationFlag;
use std::collections::HashSet;
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const GGML_QUANTIZATION_VERSION: u32 = 2;
const DEFAULT_ALIGNMENT: u64 = 32;
const JS_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
const READ_CHUNK_BYTES: usize = 8 * 1024 * 1024;
const HEADER_READ_CHUNK_BYTES: usize = 64 * 1024;
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
const MAX_METADATA_COUNT: u64 = 4_096;
const MAX_TENSOR_COUNT: u64 = 65_536;
const MAX_METADATA_KEY_BYTES: u64 = 65_535;
const MAX_TENSOR_NAME_BYTES: u64 = 64;
const MAX_METADATA_STRING_BYTES: u64 = 16 * 1024 * 1024;
const MAX_METADATA_ARRAY_ELEMENTS: u64 = 1_048_576;

/// Why parsing or tensor loading failed.
#[derive(Debug)]
pub enum GgufParseError {
    /// The file violates the GGUF format or this parser's constraints.
    Invalid(String),
    /// An underlying I/O operation failed.
    Io(String),
    /// A cancellation check found the flag set.
    Cancelled,
}

impl GgufParseError {
    fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }

    /// Returns `true` for cancellation errors.
    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

impl fmt::Display for GgufParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "gguf: {message}"),
            Self::Io(message) => write!(f, "gguf: I/O failed: {message}"),
            Self::Cancelled => f.write_str("gguf: operation aborted"),
        }
    }
}

impl std::error::Error for GgufParseError {}

/// Typed GGUF metadata value. Numeric tags use the spec's type codes
/// (`0..=12`). Arrays are homogeneous [`GgufMetadataArray`] values.
#[derive(Clone, Debug, PartialEq)]
pub enum GgufMetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(GgufMetadataArray),
}

impl GgufMetadataValue {
    /// Returns the type tag name, such as `"u32"` or `"string"`. Arrays
    /// return their element type.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::U8(_) => "u8",
            Self::I8(_) => "i8",
            Self::U16(_) => "u16",
            Self::I16(_) => "i16",
            Self::U32(_) => "u32",
            Self::I32(_) => "i32",
            Self::F32(_) => "f32",
            Self::Bool(_) => "bool",
            Self::String(_) => "string",
            Self::U64(_) => "u64",
            Self::I64(_) => "i64",
            Self::F64(_) => "f64",
            Self::Array(array) => array.kind(),
        }
    }
}

/// A homogeneous GGUF metadata array (all elements share one type).
#[derive(Clone, Debug, PartialEq)]
pub enum GgufMetadataArray {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
}

impl GgufMetadataArray {
    /// Returns the element type tag name, such as `"f32"`.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::U8(_) => "u8",
            Self::I8(_) => "i8",
            Self::U16(_) => "u16",
            Self::I16(_) => "i16",
            Self::U32(_) => "u32",
            Self::I32(_) => "i32",
            Self::F32(_) => "f32",
            Self::Bool(_) => "bool",
            Self::String(_) => "string",
            Self::U64(_) => "u64",
            Self::I64(_) => "i64",
            Self::F64(_) => "f64",
        }
    }
}

/// One key/value pair of the metadata table, in file order.
#[derive(Clone, Debug, PartialEq)]
pub struct GgufMetadataEntry {
    pub key: String,
    pub value: GgufMetadataValue,
}

/// A GGML K-quant block encoding (256 values per block).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlKQuant {
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl GgmlKQuant {
    /// Returns the canonical GGML encoding name, such as `"Q4_K"`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
        }
    }

    /// Parses a canonical GGML name, or returns `None` if it is unknown.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Q2_K" => Some(Self::Q2K),
            "Q3_K" => Some(Self::Q3K),
            "Q4_K" => Some(Self::Q4K),
            "Q5_K" => Some(Self::Q5K),
            "Q6_K" => Some(Self::Q6K),
            _ => None,
        }
    }

    // GGML K-quant layouts pack this many bytes per 256-element block.
    fn block_bytes(self) -> usize {
        match self {
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
        }
    }

    /// Returns the packed byte length of a `columns`-element logical row.
    /// Returns `None` if `columns` is not a multiple of 256 or the result
    /// overflows.
    pub fn encoded_row_bytes(self, columns: usize) -> Option<usize> {
        columns
            .is_multiple_of(256)
            .then(|| (columns / 256).checked_mul(self.block_bytes()))
            .flatten()
    }
}

/// On-disk encoding of a tensor: dense little-endian `F32` or one of the
/// supported K-quants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufTensorFormat {
    F32,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl GgufTensorFormat {
    /// Returns the canonical name, such as `"F32"` or `"Q2_K"`.
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
        }
    }

    // GGML type code 0 is F32. Codes 10..=14 are Q2_K..Q6_K.
    fn from_code(code: u32) -> Result<Self, GgufParseError> {
        match code {
            0 => Ok(Self::F32),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            _ => Err(GgufParseError::invalid(format!(
                "unsupported GGML tensor type code {code}"
            ))),
        }
    }

    fn block_bytes(self) -> Option<usize> {
        self.quantization().map(GgmlKQuant::block_bytes)
    }

    /// The K-quant encoding of this format, or `None` for `F32`.
    pub fn quantization(self) -> Option<GgmlKQuant> {
        match self {
            Self::F32 => None,
            Self::Q2K => Some(GgmlKQuant::Q2K),
            Self::Q3K => Some(GgmlKQuant::Q3K),
            Self::Q4K => Some(GgmlKQuant::Q4K),
            Self::Q5K => Some(GgmlKQuant::Q5K),
            Self::Q6K => Some(GgmlKQuant::Q6K),
        }
    }
}

/// One tensor's name, format, shapes, and absolute byte range in the file.
///
/// `logical_shape` reverses GGUF's least-major-first dimensions into
/// row-major order. `physical_shape` describes the stored payload. It equals
/// `logical_shape` for `F32` and `[rows, encoded_row_bytes]` for K-quants.
/// During parsing, `data_offset` and `byte_len` must fit inside the file and
/// cannot overlap another tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufTensorDescriptor {
    pub name: String,
    pub format: GgufTensorFormat,
    pub logical_shape: Vec<usize>,
    pub physical_shape: Vec<usize>,
    data_offset: u64,
    byte_len: usize,
}

impl GgufTensorDescriptor {
    /// Absolute file offset of the tensor's payload.
    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }

    /// Exact byte length of the tensor's payload.
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }
}

/// Parsed architecture name, metadata table in file order, and tensor catalog.
#[derive(Clone, Debug, PartialEq)]
pub struct GgufFile {
    pub architecture: String,
    pub metadata: Vec<GgufMetadataEntry>,
    pub tensors: Vec<GgufTensorDescriptor>,
}

// Bounds-checked cursor over the header. `position` cannot exceed `file_len`
// or `MAX_HEADER_BYTES`. Chunking larger reads at
// `HEADER_READ_CHUNK_BYTES` bounds cancellation latency for large strings.
struct HeaderReader<'a> {
    file: &'a mut File,
    file_len: u64,
    position: u64,
    cancellation: Option<&'a CancellationFlag>,
}

impl HeaderReader<'_> {
    fn check_cancelled(&self) -> Result<(), GgufParseError> {
        if self
            .cancellation
            .is_some_and(CancellationFlag::is_cancelled)
        {
            Err(GgufParseError::Cancelled)
        } else {
            Ok(())
        }
    }

    fn read_exact(&mut self, output: &mut [u8]) -> Result<(), GgufParseError> {
        self.check_cancelled()?;
        let length = u64::try_from(output.len())
            .map_err(|_| GgufParseError::invalid("header read length is out of range"))?;
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| GgufParseError::invalid("header position overflows"))?;
        if end > self.file_len {
            return Err(GgufParseError::invalid("unexpected end of file in header"));
        }
        if end > MAX_HEADER_BYTES {
            return Err(GgufParseError::invalid(format!(
                "header exceeds the {MAX_HEADER_BYTES}-byte resource limit"
            )));
        }
        for chunk in output.chunks_mut(HEADER_READ_CHUNK_BYTES) {
            self.check_cancelled()?;
            self.file
                .read_exact(chunk)
                .map_err(|error| GgufParseError::Io(error.to_string()))?;
            self.position += chunk.len() as u64;
        }
        Ok(())
    }

    fn read_u8(&mut self) -> Result<u8, GgufParseError> {
        let mut bytes = [0; 1];
        self.read_exact(&mut bytes)?;
        Ok(bytes[0])
    }

    fn read_i8(&mut self) -> Result<i8, GgufParseError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, GgufParseError> {
        let mut bytes = [0; 2];
        self.read_exact(&mut bytes)?;
        Ok(u16::from_le_bytes(bytes))
    }

    fn read_i16(&mut self) -> Result<i16, GgufParseError> {
        let mut bytes = [0; 2];
        self.read_exact(&mut bytes)?;
        Ok(i16::from_le_bytes(bytes))
    }

    fn read_u32(&mut self) -> Result<u32, GgufParseError> {
        let mut bytes = [0; 4];
        self.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_i32(&mut self) -> Result<i32, GgufParseError> {
        let mut bytes = [0; 4];
        self.read_exact(&mut bytes)?;
        Ok(i32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> Result<u64, GgufParseError> {
        let mut bytes = [0; 8];
        self.read_exact(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_i64(&mut self) -> Result<i64, GgufParseError> {
        let mut bytes = [0; 8];
        self.read_exact(&mut bytes)?;
        Ok(i64::from_le_bytes(bytes))
    }

    fn read_f32(&mut self) -> Result<f32, GgufParseError> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_f64(&mut self) -> Result<f64, GgufParseError> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    fn read_bool(&mut self) -> Result<bool, GgufParseError> {
        match self.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(GgufParseError::invalid(format!(
                "invalid boolean metadata byte {value}"
            ))),
        }
    }

    // Before allocation, check a count against `maximum` and the remaining
    // file bytes. No element can use fewer than `minimum_bytes`, so the count
    // cannot fit when `value > remaining / minimum_bytes`.
    fn count(
        &self,
        value: u64,
        what: &str,
        minimum_bytes: u64,
        maximum: u64,
    ) -> Result<usize, GgufParseError> {
        if value > maximum {
            return Err(GgufParseError::invalid(format!(
                "{what} count {value} exceeds the resource limit {maximum}"
            )));
        }
        let remaining = self
            .file_len
            .min(MAX_HEADER_BYTES)
            .saturating_sub(self.position);
        if minimum_bytes > 0 && value > remaining / minimum_bytes {
            return Err(GgufParseError::invalid(format!(
                "{what} count {value} exceeds the remaining file"
            )));
        }
        usize::try_from(value)
            .map_err(|_| GgufParseError::invalid(format!("{what} count {value} is out of range")))
    }

    // Read a length-prefixed UTF-8 string of at most `maximum` bytes.
    // `try_reserve_exact` returns allocation failure as an error.
    fn read_string(&mut self, what: &str, maximum: u64) -> Result<String, GgufParseError> {
        let byte_len = self.read_u64()?;
        let length = self.count(byte_len, what, 1, maximum)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| GgufParseError::invalid(format!("{what} is too large")))?;
        bytes.resize(length, 0);
        self.read_exact(&mut bytes)?;
        String::from_utf8(bytes)
            .map_err(|_| GgufParseError::invalid(format!("{what} is not valid UTF-8")))
    }
}

// Reject metadata outside ±2^53−1 to prevent precision loss in Node.js.
fn checked_u64_metadata(value: u64, what: &str) -> Result<u64, GgufParseError> {
    if value > JS_SAFE_INTEGER {
        Err(GgufParseError::invalid(format!(
            "{what} value {value} is outside the JavaScript safe integer range"
        )))
    } else {
        Ok(value)
    }
}

fn checked_i64_metadata(value: i64, what: &str) -> Result<i64, GgufParseError> {
    if value < -(JS_SAFE_INTEGER as i64) || value > JS_SAFE_INTEGER as i64 {
        Err(GgufParseError::invalid(format!(
            "{what} value {value} is outside the JavaScript safe integer range"
        )))
    } else {
        Ok(value)
    }
}

fn reserve<T>(values: &mut Vec<T>, count: usize, what: &str) -> Result<(), GgufParseError> {
    values
        .try_reserve_exact(count)
        .map_err(|_| GgufParseError::invalid(format!("{what} is too large")))
}

fn read_array(reader: &mut HeaderReader<'_>) -> Result<GgufMetadataArray, GgufParseError> {
    let element_type = reader.read_u32()?;
    if element_type == 9 {
        return Err(GgufParseError::invalid(
            "nested metadata arrays are not supported by this parser",
        ));
    }
    let count_u64 = reader.read_u64()?;
    let minimum_bytes = match element_type {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4 | 5 | 6 => 4,
        8 | 10 | 11 | 12 => 8,
        _ => {
            return Err(GgufParseError::invalid(format!(
                "unsupported metadata array element type {element_type}"
            )))
        }
    };
    let count = reader.count(
        count_u64,
        "metadata array",
        minimum_bytes,
        MAX_METADATA_ARRAY_ELEMENTS,
    )?;
    macro_rules! numeric_array {
        ($variant:ident, $read:ident) => {{
            let mut values = Vec::new();
            reserve(&mut values, count, "metadata array")?;
            for _ in 0..count {
                values.push(reader.$read()?);
            }
            GgufMetadataArray::$variant(values)
        }};
    }
    Ok(match element_type {
        0 => numeric_array!(U8, read_u8),
        1 => numeric_array!(I8, read_i8),
        2 => numeric_array!(U16, read_u16),
        3 => numeric_array!(I16, read_i16),
        4 => numeric_array!(U32, read_u32),
        5 => numeric_array!(I32, read_i32),
        6 => numeric_array!(F32, read_f32),
        7 => numeric_array!(Bool, read_bool),
        8 => {
            let mut values = Vec::new();
            reserve(&mut values, count, "metadata string array")?;
            for _ in 0..count {
                values
                    .push(reader.read_string("metadata array string", MAX_METADATA_STRING_BYTES)?);
            }
            GgufMetadataArray::String(values)
        }
        10 => {
            let mut values = Vec::new();
            reserve(&mut values, count, "metadata u64 array")?;
            for _ in 0..count {
                values.push(checked_u64_metadata(
                    reader.read_u64()?,
                    "metadata u64 array",
                )?);
            }
            GgufMetadataArray::U64(values)
        }
        11 => {
            let mut values = Vec::new();
            reserve(&mut values, count, "metadata i64 array")?;
            for _ in 0..count {
                values.push(checked_i64_metadata(
                    reader.read_i64()?,
                    "metadata i64 array",
                )?);
            }
            GgufMetadataArray::I64(values)
        }
        12 => numeric_array!(F64, read_f64),
        _ => unreachable!("metadata array type was validated"),
    })
}

fn read_metadata_value(
    reader: &mut HeaderReader<'_>,
    value_type: u32,
) -> Result<GgufMetadataValue, GgufParseError> {
    Ok(match value_type {
        0 => GgufMetadataValue::U8(reader.read_u8()?),
        1 => GgufMetadataValue::I8(reader.read_i8()?),
        2 => GgufMetadataValue::U16(reader.read_u16()?),
        3 => GgufMetadataValue::I16(reader.read_i16()?),
        4 => GgufMetadataValue::U32(reader.read_u32()?),
        5 => GgufMetadataValue::I32(reader.read_i32()?),
        6 => GgufMetadataValue::F32(reader.read_f32()?),
        7 => GgufMetadataValue::Bool(reader.read_bool()?),
        8 => GgufMetadataValue::String(
            reader.read_string("metadata string", MAX_METADATA_STRING_BYTES)?,
        ),
        9 => GgufMetadataValue::Array(read_array(reader)?),
        10 => GgufMetadataValue::U64(checked_u64_metadata(reader.read_u64()?, "metadata u64")?),
        11 => GgufMetadataValue::I64(checked_i64_metadata(reader.read_i64()?, "metadata i64")?),
        12 => GgufMetadataValue::F64(reader.read_f64()?),
        _ => {
            return Err(GgufParseError::invalid(format!(
                "unsupported metadata value type {value_type}"
            )))
        }
    })
}

// Raw tensor entry with least-major-first GGUF dimensions and an offset from
// the aligned tensor-data section. Parsing converts it to an absolute,
// validated `GgufTensorDescriptor`.
#[derive(Debug)]
struct RawTensorDescriptor {
    name: String,
    dimensions: Vec<usize>,
    format: GgufTensorFormat,
    relative_offset: u64,
}

fn checked_product(values: &[usize], what: &str) -> Result<usize, GgufParseError> {
    values.iter().try_fold(1usize, |total, &value| {
        total
            .checked_mul(value)
            .ok_or_else(|| GgufParseError::invalid(format!("{what} size overflows")))
    })
}

// The caller checks that `alignment` is a non-zero power of two. The mask
// round-up is exact. Only the addition can overflow.
fn align_up(value: u64, alignment: u64) -> Result<u64, GgufParseError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| GgufParseError::invalid("aligned tensor-data offset overflows"))
}

/// Parses a GGUF v3 header and returns its metadata table and tensor catalog.
/// It checks tensor payload ranges and overlap without loading the payloads.
///
/// Seeks the file to offset 0 first. When the caller supplies `cancellation`,
/// the function polls it between read chunks. A set flag returns
/// [`GgufParseError::Cancelled`]. The module documentation lists all
/// validation and resource limits.
pub fn parse_gguf(
    file: &mut File,
    cancellation: Option<&CancellationFlag>,
) -> Result<GgufFile, GgufParseError> {
    if cancellation.is_some_and(CancellationFlag::is_cancelled) {
        return Err(GgufParseError::Cancelled);
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| GgufParseError::Io(error.to_string()))?;
    let file_len = file
        .metadata()
        .map_err(|error| GgufParseError::Io(error.to_string()))?
        .len();
    if cancellation.is_some_and(CancellationFlag::is_cancelled) {
        return Err(GgufParseError::Cancelled);
    }
    let mut reader = HeaderReader {
        file,
        file_len,
        position: 0,
        cancellation,
    };

    let mut magic = [0; 4];
    reader.read_exact(&mut magic)?;
    if &magic != GGUF_MAGIC {
        return Err(GgufParseError::invalid("invalid magic"));
    }
    let version = reader.read_u32()?;
    if version != GGUF_VERSION {
        return Err(GgufParseError::invalid(format!(
            "unsupported version {version}; expected version {GGUF_VERSION}"
        )));
    }
    let tensor_count_u64 = reader.read_u64()?;
    let metadata_count_u64 = reader.read_u64()?;
    let metadata_count = reader.count(metadata_count_u64, "metadata", 13, MAX_METADATA_COUNT)?;
    let tensor_count = reader.count(tensor_count_u64, "tensor", 32, MAX_TENSOR_COUNT)?;

    let mut metadata = Vec::new();
    reserve(&mut metadata, metadata_count, "metadata table")?;
    let mut metadata_keys = HashSet::new();
    metadata_keys
        .try_reserve(metadata_count)
        .map_err(|_| GgufParseError::invalid("metadata key set is too large"))?;
    for _ in 0..metadata_count {
        let key = reader.read_string("metadata key", MAX_METADATA_KEY_BYTES)?;
        if key.is_empty() {
            return Err(GgufParseError::invalid("metadata key must not be empty"));
        }
        if !metadata_keys.insert(key.clone()) {
            return Err(GgufParseError::invalid(format!(
                "duplicate metadata key {key:?}"
            )));
        }
        let value_type = reader.read_u32()?;
        let value = read_metadata_value(&mut reader, value_type)?;
        metadata.push(GgufMetadataEntry { key, value });
    }

    let architecture = metadata
        .iter()
        .find(|entry| entry.key == "general.architecture")
        .ok_or_else(|| GgufParseError::invalid("missing general.architecture metadata"))?;
    let GgufMetadataValue::String(architecture) = &architecture.value else {
        return Err(GgufParseError::invalid(
            "general.architecture metadata must be a string",
        ));
    };
    if architecture.is_empty() {
        return Err(GgufParseError::invalid(
            "general.architecture metadata must not be empty",
        ));
    }
    let architecture = architecture.clone();

    let alignment = match metadata
        .iter()
        .find(|entry| entry.key == "general.alignment")
    {
        None => DEFAULT_ALIGNMENT,
        Some(GgufMetadataEntry {
            value: GgufMetadataValue::U32(value),
            ..
        }) => u64::from(*value),
        Some(_) => {
            return Err(GgufParseError::invalid(
                "general.alignment metadata must be u32",
            ))
        }
    };
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(GgufParseError::invalid(format!(
            "general.alignment must be a positive power of two, got {alignment}"
        )));
    }

    let mut raw_tensors = Vec::new();
    reserve(&mut raw_tensors, tensor_count, "tensor catalog")?;
    let mut tensor_names = HashSet::new();
    tensor_names
        .try_reserve(tensor_count)
        .map_err(|_| GgufParseError::invalid("tensor name set is too large"))?;
    let mut has_quantized_tensors = false;
    for _ in 0..tensor_count {
        let name = reader.read_string("tensor name", MAX_TENSOR_NAME_BYTES)?;
        if name.is_empty() {
            return Err(GgufParseError::invalid("tensor name must not be empty"));
        }
        if !tensor_names.insert(name.clone()) {
            return Err(GgufParseError::invalid(format!(
                "duplicate tensor name {name:?}"
            )));
        }
        let rank = reader.read_u32()?;
        if !(1..=4).contains(&rank) {
            return Err(GgufParseError::invalid(format!(
                "tensor {name:?} rank must be in 1..=4, got {rank}"
            )));
        }
        let mut dimensions = Vec::new();
        reserve(&mut dimensions, rank as usize, "tensor dimensions")?;
        for _ in 0..rank {
            let dimension = reader.read_u64()?;
            if dimension == 0 || dimension > JS_SAFE_INTEGER {
                return Err(GgufParseError::invalid(format!(
                    "tensor {name:?} has invalid dimension {dimension}"
                )));
            }
            dimensions.push(usize::try_from(dimension).map_err(|_| {
                GgufParseError::invalid(format!(
                    "tensor {name:?} dimension {dimension} is out of range"
                ))
            })?);
        }
        let format = GgufTensorFormat::from_code(reader.read_u32()?)?;
        has_quantized_tensors |= format.quantization().is_some();
        let relative_offset = reader.read_u64()?;
        if relative_offset % alignment != 0 {
            return Err(GgufParseError::invalid(format!(
                "tensor {name:?} offset {relative_offset} is not aligned to {alignment}"
            )));
        }
        raw_tensors.push(RawTensorDescriptor {
            name,
            dimensions,
            format,
            relative_offset,
        });
    }

    if has_quantized_tensors {
        let quantization_version = metadata
            .iter()
            .find(|entry| entry.key == "general.quantization_version")
            .ok_or_else(|| {
                GgufParseError::invalid(
                    "missing general.quantization_version metadata for quantized tensors",
                )
            })?;
        let GgufMetadataValue::U32(quantization_version) = &quantization_version.value else {
            return Err(GgufParseError::invalid(
                "general.quantization_version metadata must be u32",
            ));
        };
        if *quantization_version != GGML_QUANTIZATION_VERSION {
            return Err(GgufParseError::invalid(format!(
                "unsupported general.quantization_version {quantization_version}; expected {GGML_QUANTIZATION_VERSION}"
            )));
        }
    }

    let data_start = align_up(reader.position, alignment)?;
    if data_start > file_len {
        return Err(GgufParseError::invalid(
            "aligned tensor-data section starts beyond the file",
        ));
    }

    let mut tensors = Vec::new();
    reserve(&mut tensors, tensor_count, "tensor descriptors")?;
    for raw in raw_tensors {
        let mut logical_shape = raw.dimensions.clone();
        logical_shape.reverse();
        let (physical_shape, byte_len) = match raw.format.block_bytes() {
            None => {
                let elements = checked_product(&logical_shape, &format!("tensor {:?}", raw.name))?;
                let byte_len = elements.checked_mul(4).ok_or_else(|| {
                    GgufParseError::invalid(format!("tensor {:?} byte length overflows", raw.name))
                })?;
                (logical_shape.clone(), byte_len)
            }
            Some(block_bytes) => {
                let columns = raw.dimensions[0];
                if columns % 256 != 0 {
                    return Err(GgufParseError::invalid(format!(
                        "tensor {:?} least-major dimension {columns} is not divisible by 256 for {}",
                        raw.name,
                        raw.format.name()
                    )));
                }
                let rows = checked_product(
                    &raw.dimensions[1..],
                    &format!("tensor {:?} row count", raw.name),
                )?;
                let encoded_row_bytes =
                    (columns / 256).checked_mul(block_bytes).ok_or_else(|| {
                        GgufParseError::invalid(format!(
                            "tensor {:?} encoded row length overflows",
                            raw.name
                        ))
                    })?;
                let byte_len = rows.checked_mul(encoded_row_bytes).ok_or_else(|| {
                    GgufParseError::invalid(format!("tensor {:?} byte length overflows", raw.name))
                })?;
                (vec![rows, encoded_row_bytes], byte_len)
            }
        };
        let data_offset = data_start
            .checked_add(raw.relative_offset)
            .ok_or_else(|| GgufParseError::invalid("absolute tensor offset overflows"))?;
        let byte_len_u64 = u64::try_from(byte_len)
            .map_err(|_| GgufParseError::invalid("tensor byte length is out of range"))?;
        let end = data_offset
            .checked_add(byte_len_u64)
            .ok_or_else(|| GgufParseError::invalid("tensor range end overflows"))?;
        if end > file_len {
            return Err(GgufParseError::invalid(format!(
                "tensor {:?} range [{data_offset}, {end}) exceeds file size {file_len}",
                raw.name
            )));
        }
        tensors.push(GgufTensorDescriptor {
            name: raw.name,
            format: raw.format,
            logical_shape,
            physical_shape,
            data_offset,
            byte_len,
        });
    }

    let mut ranges = Vec::new();
    reserve(&mut ranges, tensors.len(), "tensor ranges")?;
    ranges.extend(tensors.iter().map(|tensor| {
        (
            tensor.data_offset,
            tensor.data_offset + tensor.byte_len as u64,
            tensor.name.as_str(),
        )
    }));
    ranges.sort_unstable_by_key(|range| range.0);
    for pair in ranges.windows(2) {
        if pair[1].0 < pair[0].1 {
            return Err(GgufParseError::invalid(format!(
                "tensor ranges for {:?} and {:?} overlap",
                pair[0].2, pair[1].2
            )));
        }
    }
    reader.check_cancelled()?;
    Ok(GgufFile {
        architecture,
        metadata,
        tensors,
    })
}

/// Loads one tensor payload into `output` with positioned `read_exact_at`
/// calls. They leave the file cursor unchanged, which permits concurrent reads.
///
/// `output.len()` must equal `tensor.byte_len()`. The function reads chunks
/// of at most `READ_CHUNK_BYTES` and polls for cancellation before each one.
/// Cancellation leaves the written prefix unchanged. The function copies raw,
/// possibly quantized bytes without dequantizing them.
pub fn read_gguf_tensor_into(
    file: &File,
    tensor: &GgufTensorDescriptor,
    output: &mut [u8],
    cancellation: Option<&CancellationFlag>,
) -> Result<(), GgufParseError> {
    if output.len() != tensor.byte_len {
        return Err(GgufParseError::invalid(format!(
            "tensor {:?} requires {} destination bytes, got {}",
            tensor.name,
            tensor.byte_len,
            output.len()
        )));
    }
    for (index, chunk) in output.chunks_mut(READ_CHUNK_BYTES).enumerate() {
        if cancellation.is_some_and(CancellationFlag::is_cancelled) {
            return Err(GgufParseError::Cancelled);
        }
        let relative = index
            .checked_mul(READ_CHUNK_BYTES)
            .ok_or_else(|| GgufParseError::invalid("tensor read offset overflows"))?;
        let offset = tensor
            .data_offset
            .checked_add(relative as u64)
            .ok_or_else(|| GgufParseError::invalid("tensor read offset overflows"))?;
        read_exact_at(file, chunk, offset)?;
    }
    if cancellation.is_some_and(CancellationFlag::is_cancelled) {
        return Err(GgufParseError::Cancelled);
    }
    Ok(())
}

// Read tensor payloads by position. Unix `read_exact_at` loops internally.
// Windows `seek_read` may return fewer bytes, so check its result and report
// a short read as a truncated payload.
#[cfg(unix)]
fn read_exact_at(file: &File, output: &mut [u8], offset: u64) -> Result<(), GgufParseError> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(output, offset)
        .map_err(|error| GgufParseError::Io(error.to_string()))
}

#[cfg(windows)]
fn read_exact_at(file: &File, output: &mut [u8], offset: u64) -> Result<(), GgufParseError> {
    use std::os::windows::fs::FileExt;
    file.seek_read(output, offset)
        .map_err(|error| GgufParseError::Io(error.to_string()))
        .and_then(|read| {
            if read == output.len() {
                Ok(())
            } else {
                Err(GgufParseError::invalid("unexpected end of tensor payload"))
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    #[derive(Clone)]
    struct TensorFixture {
        name: &'static str,
        dimensions: Vec<u64>,
        format: u32,
        offset: u64,
        bytes: usize,
    }

    fn string(output: &mut Vec<u8>, value: &str) {
        output.extend_from_slice(&(value.len() as u64).to_le_bytes());
        output.extend_from_slice(value.as_bytes());
    }

    fn metadata(output: &mut Vec<u8>, key: &str, kind: u32, value: &[u8]) {
        string(output, key);
        output.extend_from_slice(&kind.to_le_bytes());
        output.extend_from_slice(value);
    }

    fn string_value(value: &str) -> Vec<u8> {
        let mut output = Vec::new();
        string(&mut output, value);
        output
    }

    fn array_value(kind: u32, count: u64, values: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        output.extend_from_slice(&kind.to_le_bytes());
        output.extend_from_slice(&count.to_le_bytes());
        output.extend_from_slice(values);
        output
    }

    fn build(
        extra_metadata: &[(&str, u32, Vec<u8>)],
        tensors: &[TensorFixture],
        alignment: u32,
    ) -> Vec<u8> {
        let mut output = Vec::new();
        output.extend_from_slice(GGUF_MAGIC);
        output.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        output.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        output.extend_from_slice(&((extra_metadata.len() + 2) as u64).to_le_bytes());
        metadata(
            &mut output,
            "general.architecture",
            8,
            &string_value("test"),
        );
        metadata(
            &mut output,
            "general.alignment",
            4,
            &alignment.to_le_bytes(),
        );
        for (key, kind, value) in extra_metadata {
            metadata(&mut output, key, *kind, value);
        }
        for tensor in tensors {
            string(&mut output, tensor.name);
            output.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
            for dimension in &tensor.dimensions {
                output.extend_from_slice(&dimension.to_le_bytes());
            }
            output.extend_from_slice(&tensor.format.to_le_bytes());
            output.extend_from_slice(&tensor.offset.to_le_bytes());
        }
        let aligned = output.len().next_multiple_of(alignment as usize);
        output.resize(aligned, 0);
        let length = tensors
            .iter()
            .map(|tensor| tensor.offset as usize + tensor.bytes)
            .max()
            .unwrap_or(0);
        output.resize(aligned + length, 0);
        output
    }

    fn parse_bytes(bytes: &[u8]) -> Result<GgufFile, GgufParseError> {
        let path = std::env::temp_dir().join(format!(
            "effect-torch-gguf-test-{}-{}.gguf",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).unwrap();
        let mut file = File::open(&path).unwrap();
        let parsed = parse_gguf(&mut file, None);
        std::fs::remove_file(path).ok();
        parsed
    }

    #[test]
    fn parses_f32_and_every_supported_k_quant_shape() {
        let tensors = [
            TensorFixture {
                name: "dense",
                dimensions: vec![3, 2],
                format: 0,
                offset: 0,
                bytes: 24,
            },
            TensorFixture {
                name: "q2",
                dimensions: vec![256, 2],
                format: 10,
                offset: 32,
                bytes: 168,
            },
            TensorFixture {
                name: "q3",
                dimensions: vec![256, 2],
                format: 11,
                offset: 224,
                bytes: 220,
            },
            TensorFixture {
                name: "q4",
                dimensions: vec![512, 2],
                format: 12,
                offset: 448,
                bytes: 576,
            },
            TensorFixture {
                name: "q5",
                dimensions: vec![256, 2],
                format: 13,
                offset: 1024,
                bytes: 352,
            },
            TensorFixture {
                name: "q6",
                dimensions: vec![256, 2],
                format: 14,
                offset: 1376,
                bytes: 420,
            },
        ];
        let parsed = parse_bytes(&build(
            &[(
                "general.quantization_version",
                4,
                GGML_QUANTIZATION_VERSION.to_le_bytes().to_vec(),
            )],
            &tensors,
            32,
        ))
        .unwrap();
        assert_eq!(parsed.architecture, "test");
        assert_eq!(parsed.tensors[0].logical_shape, [2, 3]);
        assert_eq!(parsed.tensors[0].physical_shape, [2, 3]);
        assert_eq!(parsed.tensors[1].physical_shape, [2, 84]);
        assert_eq!(parsed.tensors[2].physical_shape, [2, 110]);
        assert_eq!(parsed.tensors[3].physical_shape, [2, 288]);
        assert_eq!(parsed.tensors[4].physical_shape, [2, 176]);
        assert_eq!(parsed.tensors[5].physical_shape, [2, 210]);
    }

    #[test]
    fn parses_every_metadata_scalar_and_homogeneous_array_type() {
        let mut string_array = Vec::new();
        string(&mut string_array, "a");
        string(&mut string_array, "b");
        let entries = vec![
            ("u8", 0, vec![7]),
            ("i8", 1, vec![(-7i8) as u8]),
            ("u16", 2, 17u16.to_le_bytes().to_vec()),
            ("i16", 3, (-17i16).to_le_bytes().to_vec()),
            ("u32", 4, 27u32.to_le_bytes().to_vec()),
            ("i32", 5, (-27i32).to_le_bytes().to_vec()),
            ("f32", 6, 1.5f32.to_le_bytes().to_vec()),
            ("bool", 7, vec![1]),
            ("string", 8, string_value("value")),
            ("u64", 10, 37u64.to_le_bytes().to_vec()),
            ("i64", 11, (-37i64).to_le_bytes().to_vec()),
            ("f64", 12, 2.5f64.to_le_bytes().to_vec()),
            ("a_u8", 9, array_value(0, 2, &[1, 2])),
            ("a_i8", 9, array_value(1, 2, &[255, 2])),
            ("a_u16", 9, array_value(2, 1, &3u16.to_le_bytes())),
            ("a_i16", 9, array_value(3, 1, &(-3i16).to_le_bytes())),
            ("a_u32", 9, array_value(4, 1, &4u32.to_le_bytes())),
            ("a_i32", 9, array_value(5, 1, &(-4i32).to_le_bytes())),
            ("a_f32", 9, array_value(6, 1, &3.5f32.to_le_bytes())),
            ("a_bool", 9, array_value(7, 2, &[0, 1])),
            ("a_string", 9, array_value(8, 2, &string_array)),
            ("a_u64", 9, array_value(10, 1, &5u64.to_le_bytes())),
            ("a_i64", 9, array_value(11, 1, &(-5i64).to_le_bytes())),
            ("a_f64", 9, array_value(12, 1, &4.5f64.to_le_bytes())),
        ];
        let tensor = TensorFixture {
            name: "dense",
            dimensions: vec![1],
            format: 0,
            offset: 0,
            bytes: 4,
        };
        let parsed = parse_bytes(&build(&entries, &[tensor], 32)).unwrap();
        assert_eq!(parsed.metadata.len(), entries.len() + 2);
        assert!(matches!(
            parsed.metadata.last().unwrap().value,
            GgufMetadataValue::Array(GgufMetadataArray::F64(ref values)) if values == &[4.5]
        ));
    }

    #[test]
    fn rejects_malformed_headers_catalogs_ranges_and_metadata() {
        let tensor = TensorFixture {
            name: "weight",
            dimensions: vec![256],
            format: 10,
            offset: 0,
            bytes: 84,
        };
        let quantization_version = [(
            "general.quantization_version",
            4,
            GGML_QUANTIZATION_VERSION.to_le_bytes().to_vec(),
        )];
        let valid = build(&quantization_version, std::slice::from_ref(&tensor), 32);

        let mut bad_magic = valid.clone();
        bad_magic[0] = b'X';
        assert!(parse_bytes(&bad_magic).is_err());

        let mut bad_version = valid.clone();
        bad_version[4..8].copy_from_slice(&2u32.to_le_bytes());
        assert!(parse_bytes(&bad_version).is_err());

        assert!(parse_bytes(&build(
            &[("general.architecture", 8, string_value("other"))],
            std::slice::from_ref(&tensor),
            32,
        ))
        .is_err());
        assert!(parse_bytes(&build(
            &[("unsafe", 10, u64::MAX.to_le_bytes().to_vec())],
            std::slice::from_ref(&tensor),
            32,
        ))
        .is_err());
        assert!(parse_bytes(&build(
            &quantization_version,
            std::slice::from_ref(&tensor),
            3,
        ))
        .is_err());

        let duplicate = [tensor.clone(), tensor.clone()];
        assert!(parse_bytes(&build(&quantization_version, &duplicate, 32)).is_err());
        let overlap = [
            tensor.clone(),
            TensorFixture {
                name: "other",
                offset: 32,
                ..tensor.clone()
            },
        ];
        assert!(parse_bytes(&build(&quantization_version, &overlap, 32)).is_err());
        let partial = TensorFixture {
            dimensions: vec![255],
            ..tensor.clone()
        };
        assert!(parse_bytes(&build(&quantization_version, &[partial], 32)).is_err());
        let unsupported = TensorFixture {
            format: 1,
            ..tensor.clone()
        };
        assert!(parse_bytes(&build(&quantization_version, &[unsupported], 32)).is_err());
        let zero = TensorFixture {
            dimensions: vec![0],
            ..tensor.clone()
        };
        assert!(parse_bytes(&build(&quantization_version, &[zero], 32)).is_err());

        let truncated = &valid[..valid.len() - 1];
        assert!(parse_bytes(truncated).is_err());
    }

    #[test]
    fn requires_quantization_version_two_for_k_quants() {
        let tensor = TensorFixture {
            name: "weight",
            dimensions: vec![256],
            format: 10,
            offset: 0,
            bytes: 84,
        };
        let error = parse_bytes(&build(&[], std::slice::from_ref(&tensor), 32)).unwrap_err();
        assert!(error
            .to_string()
            .contains("missing general.quantization_version"));

        let wrong_type = [(
            "general.quantization_version",
            10,
            2u64.to_le_bytes().to_vec(),
        )];
        assert!(
            parse_bytes(&build(&wrong_type, std::slice::from_ref(&tensor), 32))
                .unwrap_err()
                .to_string()
                .contains("must be u32")
        );

        let wrong_version = [(
            "general.quantization_version",
            4,
            1u32.to_le_bytes().to_vec(),
        )];
        assert!(
            parse_bytes(&build(&wrong_version, std::slice::from_ref(&tensor), 32,))
                .unwrap_err()
                .to_string()
                .contains("expected 2")
        );

        let supported = [(
            "general.quantization_version",
            4,
            2u32.to_le_bytes().to_vec(),
        )];
        assert!(parse_bytes(&build(&supported, &[tensor], 32)).is_ok());
    }

    #[test]
    fn allows_f32_without_quantization_version() {
        let tensor = TensorFixture {
            name: "weight",
            dimensions: vec![1],
            format: 0,
            offset: 0,
            bytes: 4,
        };
        assert!(parse_bytes(&build(&[], &[tensor], 32)).is_ok());
    }

    #[test]
    fn rejects_counts_and_lengths_over_resource_limits_before_allocation() {
        let mut metadata_count = Vec::new();
        metadata_count.extend_from_slice(GGUF_MAGIC);
        metadata_count.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        metadata_count.extend_from_slice(&0u64.to_le_bytes());
        metadata_count.extend_from_slice(&(MAX_METADATA_COUNT + 1).to_le_bytes());
        assert!(parse_bytes(&metadata_count)
            .unwrap_err()
            .to_string()
            .contains("resource limit"));

        let mut key_length = Vec::new();
        key_length.extend_from_slice(GGUF_MAGIC);
        key_length.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        key_length.extend_from_slice(&0u64.to_le_bytes());
        key_length.extend_from_slice(&1u64.to_le_bytes());
        key_length.extend_from_slice(&(MAX_METADATA_KEY_BYTES + 1).to_le_bytes());
        key_length.extend_from_slice(&[0; 13]);
        assert!(parse_bytes(&key_length)
            .unwrap_err()
            .to_string()
            .contains("resource limit"));
    }

    #[test]
    fn pre_cancelled_parse_stops_before_header_io() {
        let path = std::env::temp_dir().join(format!(
            "effect-torch-gguf-cancelled-{}-{}.gguf",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, b"not a gguf").unwrap();
        let mut file = File::open(&path).unwrap();
        let cancellation = CancellationFlag::new();
        cancellation.cancel();
        assert!(matches!(
            parse_gguf(&mut file, Some(&cancellation)),
            Err(GgufParseError::Cancelled)
        ));
        drop(file);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn validates_sparse_large_file_ranges_without_reading_payloads() {
        let tensor = TensorFixture {
            name: "sparse",
            dimensions: vec![1],
            format: 0,
            offset: 1 << 32,
            bytes: 4,
        };
        let mut header = Vec::new();
        header.extend_from_slice(GGUF_MAGIC);
        header.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        header.extend_from_slice(&1u64.to_le_bytes());
        header.extend_from_slice(&2u64.to_le_bytes());
        metadata(
            &mut header,
            "general.architecture",
            8,
            &string_value("test"),
        );
        metadata(&mut header, "general.alignment", 4, &32u32.to_le_bytes());
        string(&mut header, tensor.name);
        header.extend_from_slice(&1u32.to_le_bytes());
        header.extend_from_slice(&1u64.to_le_bytes());
        header.extend_from_slice(&tensor.format.to_le_bytes());
        header.extend_from_slice(&tensor.offset.to_le_bytes());
        header.resize(header.len().next_multiple_of(32), 0);
        let data_start = header.len() as u64;
        let path = std::env::temp_dir().join(format!(
            "effect-torch-gguf-sparse-{}-{}.gguf",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.write_all(&header).unwrap();
        file.set_len(data_start + tensor.offset + tensor.bytes as u64)
            .unwrap();
        let parsed = parse_gguf(&mut file, None).unwrap();
        assert_eq!(parsed.tensors[0].data_offset(), data_start + tensor.offset);
        drop(file);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn fills_a_caller_owned_tensor_destination_exactly() {
        let tensor = TensorFixture {
            name: "dense",
            dimensions: vec![2],
            format: 0,
            offset: 0,
            bytes: 8,
        };
        let mut bytes = build(&[], &[tensor], 32);
        let payload = [1.5f32.to_le_bytes(), (-2.25f32).to_le_bytes()].concat();
        let start = bytes.len() - payload.len();
        bytes[start..].copy_from_slice(&payload);
        let path = std::env::temp_dir().join(format!(
            "effect-torch-gguf-fill-{}-{}.gguf",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).unwrap();
        let mut file = File::open(&path).unwrap();
        let parsed = parse_gguf(&mut file, None).unwrap();
        let mut output = [0u8; 8];
        read_gguf_tensor_into(&file, &parsed.tensors[0], &mut output, None).unwrap();
        assert_eq!(output, payload.as_slice());
        assert!(read_gguf_tensor_into(&file, &parsed.tensors[0], &mut output[..7], None).is_err());
        drop(file);
        std::fs::remove_file(path).ok();
    }
}

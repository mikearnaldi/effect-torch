//! Safetensors archive IO for the napi boundary.
//!
//! Serialization gathers each (possibly strided) tensor into dense
//! little-endian bytes; deserialization validates the byte length against
//! the declared shape and dtype before decoding. Writes are atomic: the
//! archive is serialized to a sibling temporary file (PID + sequence
//! numbered) and renamed over the destination, with the temporary removed on
//! failure.

use super::err::{err, Res};
use super::value::Value;
use crate::{CpuBuffer, Tensor};
use effect_torch_runtime::DType as RuntimeDType;
use safetensors::tensor::{serialize, Dtype, SafeTensors, TensorView};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP_FILE: AtomicU64 = AtomicU64::new(0);

fn archive_dtype(dtype: RuntimeDType) -> Dtype {
    match dtype {
        RuntimeDType::F32 => Dtype::F32,
        RuntimeDType::F64 => Dtype::F64,
        RuntimeDType::F16 => Dtype::F16,
        RuntimeDType::BF16 => Dtype::BF16,
        RuntimeDType::U8 => Dtype::U8,
        RuntimeDType::U32 => Dtype::U32,
        RuntimeDType::I64 => Dtype::I64,
    }
}

fn runtime_dtype(dtype: Dtype) -> Res<RuntimeDType> {
    match dtype {
        Dtype::F32 => Ok(RuntimeDType::F32),
        Dtype::F64 => Ok(RuntimeDType::F64),
        Dtype::F16 => Ok(RuntimeDType::F16),
        Dtype::BF16 => Ok(RuntimeDType::BF16),
        Dtype::U8 => Ok(RuntimeDType::U8),
        Dtype::U32 => Ok(RuntimeDType::U32),
        Dtype::I64 => Ok(RuntimeDType::I64),
        other => err(format!("safetensors: unsupported dtype {other:?}")),
    }
}

/// Gathers a value's logical elements into dense little-endian bytes.
pub fn tensor_bytes(value: &Value) -> Res<Vec<u8>> {
    let tensor = value.tensor();
    let mut output = Vec::with_capacity(tensor.numel() * tensor.dtype().size_in_bytes());
    let logical_offset = |mut linear: usize| {
        let mut offset = tensor.layout.offset();
        for dimension in (0..tensor.layout.rank()).rev() {
            let width = tensor.layout.shape()[dimension].max(1);
            offset += (linear % width) * tensor.layout.strides()[dimension];
            linear /= width;
        }
        offset
    };
    macro_rules! write_le {
        ($values:expr) => {
            for index in 0..tensor.numel() {
                output.extend_from_slice(&$values[logical_offset(index)].to_le_bytes());
            }
        };
    }
    match &tensor.buffer {
        CpuBuffer::F32(values) => write_le!(values),
        CpuBuffer::F64(values) => write_le!(values),
        CpuBuffer::F16(values) => write_le!(values),
        CpuBuffer::BF16(values) => write_le!(values),
        CpuBuffer::U8(values) => {
            for index in 0..tensor.numel() {
                output.push(values[logical_offset(index)]);
            }
        }
        CpuBuffer::U32(values) => write_le!(values),
        CpuBuffer::I64(values) => write_le!(values),
    }
    Ok(output)
}

/// Decodes dense little-endian bytes into a contiguous tensor of `shape`
/// and `dtype`; the byte length must match exactly.
pub fn value_from_bytes(bytes: &[u8], shape: &[usize], dtype: RuntimeDType) -> Res<Value> {
    let elements = shape
        .iter()
        .try_fold(1usize, |total, &dimension| total.checked_mul(dimension))
        .ok_or_else(|| "safetensors: tensor element count overflows".to_string())?;
    let expected = elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| "safetensors: tensor byte length overflows".to_string())?;
    if bytes.len() != expected {
        return err(format!(
            "safetensors: expected {expected} bytes for {dtype} tensor with shape {shape:?}, got {}",
            bytes.len()
        ));
    }
    let tensor = match dtype {
        RuntimeDType::F32 => Tensor::from_vec(
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("validated chunk")))
                .collect(),
            shape.to_vec(),
        ),
        RuntimeDType::F64 => Tensor::from_vec(
            bytes
                .chunks_exact(8)
                .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("validated chunk")))
                .collect(),
            shape.to_vec(),
        ),
        RuntimeDType::F16 => Tensor::from_vec(
            bytes
                .chunks_exact(2)
                .map(|chunk| half::f16::from_le_bytes(chunk.try_into().expect("validated chunk")))
                .collect(),
            shape.to_vec(),
        ),
        RuntimeDType::BF16 => Tensor::from_vec(
            bytes
                .chunks_exact(2)
                .map(|chunk| half::bf16::from_le_bytes(chunk.try_into().expect("validated chunk")))
                .collect(),
            shape.to_vec(),
        ),
        RuntimeDType::U8 => Tensor::from_vec(bytes.to_vec(), shape.to_vec()),
        RuntimeDType::U32 => Tensor::from_vec(
            bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("validated chunk")))
                .collect(),
            shape.to_vec(),
        ),
        RuntimeDType::I64 => Tensor::from_vec(
            bytes
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("validated chunk")))
                .collect(),
            shape.to_vec(),
        ),
    };
    Ok(Value(tensor))
}

fn temporary_path(path: &Path) -> Res<PathBuf> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "safetensors: output path has no valid file name".to_string())?;
    for _ in 0..100 {
        let sequence = NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(".{name}.{}.{}.tmp", std::process::id(), sequence));
        if !candidate.exists() {
            return Ok(candidate);
        }
    }
    err("safetensors: could not allocate a temporary output path")
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Res<()> {
    let temporary = temporary_path(path)?;
    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| error.to_string())?;
        file.write_all(bytes).map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
        drop(file);
        std::fs::rename(&temporary, path).map_err(|error| error.to_string())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

/// Serializes `tensors` and `metadata` into a safetensors archive at
/// `path`, atomically replacing any existing file.
pub fn save(
    tensors: &HashMap<String, Value>,
    metadata: &HashMap<String, String>,
    path: &str,
) -> Res<()> {
    let mut owned = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors {
        owned.push((
            name.clone(),
            archive_dtype(tensor.dtype()),
            tensor.shape(),
            tensor_bytes(tensor)?,
        ));
    }
    let views = owned
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            TensorView::new(*dtype, shape.to_vec(), bytes).map(|view| (name.as_str(), view))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    let encoded = serialize(views, Some(metadata.clone())).map_err(|error| error.to_string())?;
    atomic_write(Path::new(path), &encoded)
}

/// A loaded archive: name-sorted entries plus archive metadata.
pub struct LoadedArchive {
    pub entries: Vec<(String, Value)>,
    pub metadata: HashMap<String, String>,
}

/// Loads the archive at `path`, decoding every tensor and returning entries
/// sorted by name.
pub fn load(path: &str) -> Res<LoadedArchive> {
    let raw = std::fs::read(path).map_err(|error| error.to_string())?;
    let (_, parsed_metadata) =
        SafeTensors::read_metadata(&raw).map_err(|error| error.to_string())?;
    let metadata = parsed_metadata.metadata().clone().unwrap_or_default();
    let tensors = SafeTensors::deserialize(&raw).map_err(|error| error.to_string())?;
    let mut entries = Vec::with_capacity(tensors.len());
    for name in tensors.names() {
        let view = tensors.tensor(name).map_err(|error| error.to_string())?;
        let dtype = runtime_dtype(view.dtype())?;
        entries.push((
            name.to_string(),
            value_from_bytes(view.data(), view.shape(), dtype)?,
        ));
    }
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(LoadedArchive { entries, metadata })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_with_metadata_and_escaped_name() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "quoted\"\\\nname".to_string(),
            Value(Tensor::from_vec(
                vec![half::f16::from_bits(0x3e00), half::f16::from_bits(0xc080)],
                vec![2],
            )),
        );
        let metadata = HashMap::from([("framework".to_string(), "effect-torch".to_string())]);
        let path = std::env::temp_dir().join(format!(
            "effect-torch-safetensors-{}-{}.safetensors",
            std::process::id(),
            NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        save(&tensors, &metadata, path.to_str().unwrap()).unwrap();
        let archive = load(path.to_str().unwrap()).unwrap();
        assert_eq!(archive.metadata, metadata);
        assert_eq!(archive.entries[0].0, "quoted\"\\\nname");
        assert_eq!(
            tensor_bytes(&archive.entries[0].1).unwrap(),
            vec![0x00, 0x3e, 0x80, 0xc0]
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn exact_bf16_import_export() {
        let bytes = [0x80, 0x3f, 0x00, 0xc0];
        let value = value_from_bytes(&bytes, &[2], RuntimeDType::BF16).unwrap();
        assert_eq!(tensor_bytes(&value).unwrap(), bytes);
    }

    #[test]
    fn malformed_archive_is_rejected() {
        let path = std::env::temp_dir().join(format!(
            "effect-torch-safetensors-bad-{}-{}.safetensors",
            std::process::id(),
            NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, b"not a safetensors archive").unwrap();
        assert!(load(path.to_str().unwrap()).is_err());
        std::fs::remove_file(path).ok();
    }
}

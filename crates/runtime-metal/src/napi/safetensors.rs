use super::err::{err, Res};
use super::value::Value;
use crate::device::MetalDevice;
use effect_torch_runtime::DType;
use safetensors::tensor::{serialize, Dtype, SafeTensors, TensorView};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP_FILE: AtomicU64 = AtomicU64::new(0);

fn safetensors_dtype(dtype: DType) -> Dtype {
    match dtype {
        DType::F32 => Dtype::F32,
        DType::F64 => Dtype::F64,
        DType::F16 => Dtype::F16,
        DType::BF16 => Dtype::BF16,
        DType::U8 => Dtype::U8,
        DType::U32 => Dtype::U32,
        DType::I64 => Dtype::I64,
    }
}

fn runtime_dtype(dtype: Dtype) -> Res<DType> {
    match dtype {
        Dtype::F32 => Ok(DType::F32),
        Dtype::F64 => Ok(DType::F64),
        Dtype::F16 => Ok(DType::F16),
        Dtype::BF16 => Ok(DType::BF16),
        Dtype::U8 => Ok(DType::U8),
        Dtype::U32 => Ok(DType::U32),
        Dtype::I64 => Ok(DType::I64),
        other => err(format!("safetensors: unsupported dtype {other:?}")),
    }
}

pub fn tensor_bytes(value: &Value) -> Res<Vec<u8>> {
    let tensor = &value.0;
    MetalDevice::get().synchronize()?;
    let count = tensor.numel();
    let base = tensor.buffer.contents_ptr().cast::<u8>();
    let logical_offset = |mut linear: usize| {
        let mut offset = tensor.layout.offset();
        for dimension in (0..tensor.layout.rank()).rev() {
            let width = tensor.layout.shape()[dimension].max(1);
            offset += (linear % width) * tensor.layout.strides()[dimension];
            linear /= width;
        }
        offset
    };
    let mut output = Vec::with_capacity(count * tensor.dtype.size_in_bytes());
    macro_rules! write_le {
        ($type:ty) => {{
            let source = base.cast::<$type>();
            for index in 0..count {
                output.extend_from_slice(
                    &unsafe { *source.add(logical_offset(index)) }.to_le_bytes(),
                );
            }
        }};
    }
    match tensor.dtype {
        DType::F32 => write_le!(f32),
        DType::F64 => write_le!(f64),
        DType::F16 | DType::BF16 => write_le!(u16),
        DType::U8 => {
            for index in 0..count {
                output.push(unsafe { *base.add(logical_offset(index)) });
            }
        }
        DType::U32 => write_le!(u32),
        DType::I64 => write_le!(i64),
    }
    Ok(output)
}

pub fn value_from_bytes(bytes: &[u8], shape: &[usize], dtype: DType) -> Res<Value> {
    crate::value::value_from_bytes(bytes, shape, dtype)
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

pub fn save(
    tensors: &HashMap<String, Value>,
    metadata: &HashMap<String, String>,
    path: &str,
) -> Res<()> {
    let mut owned = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors {
        owned.push((
            name.clone(),
            safetensors_dtype(tensor.dtype()),
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

pub struct LoadedArchive {
    pub entries: Vec<(String, Value)>,
    pub metadata: HashMap<String, String>,
}

pub fn load(path: &str) -> Res<LoadedArchive> {
    let raw = std::fs::read(path).map_err(|error| error.to_string())?;
    let (_, parsed_metadata) =
        SafeTensors::read_metadata(&raw).map_err(|error| error.to_string())?;
    let metadata = parsed_metadata.metadata().clone().unwrap_or_default();
    let tensors = SafeTensors::deserialize(&raw).map_err(|error| error.to_string())?;
    let mut entries = Vec::with_capacity(tensors.len());
    for name in tensors.names() {
        let view = tensors.tensor(name).map_err(|error| error.to_string())?;
        entries.push((
            name.to_string(),
            value_from_bytes(view.data(), view.shape(), runtime_dtype(view.dtype())?)?,
        ));
    }
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(LoadedArchive { entries, metadata })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn archive_path(kind: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "effect-torch-metal-safetensors-{kind}-{}-{}.safetensors",
            std::process::id(),
            NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn roundtrip_with_metadata_and_escaped_name() {
        let name = "quoted\"\\\nname";
        let value = value_from_bytes(&[0x00, 0x3e, 0x80, 0xc0], &[2], DType::F16).unwrap();
        let tensors = HashMap::from([(name.to_string(), value)]);
        let metadata = HashMap::from([("framework".to_string(), "effect-torch".to_string())]);
        let path = archive_path("roundtrip");
        save(&tensors, &metadata, path.to_str().unwrap()).unwrap();
        let archive = load(path.to_str().unwrap()).unwrap();
        assert_eq!(archive.metadata, metadata);
        assert_eq!(archive.entries[0].0, name);
        assert_eq!(
            tensor_bytes(&archive.entries[0].1).unwrap(),
            [0x00, 0x3e, 0x80, 0xc0]
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn exact_bf16_import_export() {
        let bytes = [0x80, 0x3f, 0x00, 0xc0];
        let value = value_from_bytes(&bytes, &[2], DType::BF16).unwrap();
        assert_eq!(tensor_bytes(&value).unwrap(), bytes);
    }

    #[test]
    fn malformed_archive_is_rejected() {
        let path = archive_path("bad");
        std::fs::write(&path, b"not a safetensors archive").unwrap();
        assert!(load(path.to_str().unwrap()).is_err());
        std::fs::remove_file(path).ok();
    }
}

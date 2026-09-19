use crate::{CudaDevice, CudaValue};
use effect_torch_napi::safetensors::{self as shared, Error as SharedError};
use effect_torch_runtime::{DType, StorageRepresentation};
#[cfg(test)]
use safetensors::tensor::SafeTensors;
use safetensors::tensor::{serialize, Dtype, TensorView};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_TEMP_FILE: AtomicU64 = AtomicU64::new(0);

fn archive_dtype(dtype: DType) -> Dtype {
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

fn runtime_dtype(dtype: Dtype) -> Result<DType, String> {
    match dtype {
        Dtype::F32 => Ok(DType::F32),
        Dtype::F64 => Ok(DType::F64),
        Dtype::F16 => Ok(DType::F16),
        Dtype::BF16 => Ok(DType::BF16),
        Dtype::U8 => Ok(DType::U8),
        Dtype::U32 => Ok(DType::U32),
        Dtype::I64 => Ok(DType::I64),
        other => Err(format!("safetensors: unsupported dtype {other:?}")),
    }
}

fn tensor_bytes(value: &CudaValue) -> Result<Vec<u8>, String> {
    if value.spec().storage.representation != StorageRepresentation::Dense {
        return Err(
            "safetensors: packed tensors require an explicit representation-aware archive"
                .to_string(),
        );
    }
    value.read_storage_bytes()
}

fn temporary_path(path: &Path) -> Result<PathBuf, String> {
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
    Err("safetensors: could not allocate a temporary output path".to_string())
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
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
    tensors: &HashMap<String, CudaValue>,
    metadata: &HashMap<String, String>,
    path: &str,
) -> Result<(), String> {
    let mut owned = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors {
        owned.push((
            name.clone(),
            archive_dtype(tensor.dtype()),
            tensor.shape().to_vec(),
            tensor_bytes(tensor)?,
        ));
    }
    let encoded = encode_archive(&owned, metadata)?;
    atomic_write(Path::new(path), &encoded)
}

fn encode_archive(
    owned: &[(String, Dtype, Vec<usize>, Vec<u8>)],
    metadata: &HashMap<String, String>,
) -> Result<Vec<u8>, String> {
    let views = owned
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            TensorView::new(*dtype, shape.clone(), bytes).map(|view| (name.as_str(), view))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    serialize(views, Some(metadata.clone())).map_err(|error| error.to_string())
}

#[cfg(test)]
fn decode_archive(raw: &[u8]) -> Result<(SafeTensors<'_>, HashMap<String, String>), String> {
    let (_, parsed_metadata) =
        SafeTensors::read_metadata(raw).map_err(|error| error.to_string())?;
    let metadata = parsed_metadata.metadata().clone().unwrap_or_default();
    let tensors = SafeTensors::deserialize(raw).map_err(|error| error.to_string())?;
    for name in tensors.names() {
        let view = tensors.tensor(name).map_err(|error| error.to_string())?;
        crate::value::validate_storage_bytes(
            effect_torch_runtime::ValueSpec::dense(runtime_dtype(view.dtype())?, view.shape()),
            view.data().len(),
        )?;
    }
    Ok((tensors, metadata))
}

pub struct LoadedArchive {
    pub entries: Vec<(String, CudaValue)>,
    pub metadata: HashMap<String, String>,
}

/// Plans and loads the archive at `path` onto `device`. `names` selects
/// tensors; `None` loads every tensor and `Some([])` loads none. Every
/// selected dtype is validated before the first device allocation, so an
/// unsupported dtype cannot leave a partially uploaded archive behind.
pub fn load(
    path: &str,
    names: Option<&[String]>,
    device: Arc<CudaDevice>,
    cancelled: &dyn Fn() -> bool,
) -> Result<LoadedArchive, SharedError> {
    let plan = shared::plan(path, names, cancelled)?;
    let metadata = plan.metadata().clone();
    for meta in plan.entries() {
        let dtype = runtime_dtype(meta.dtype).map_err(SharedError::message)?;
        let shape = meta
            .shape
            .iter()
            .map(|&dimension| dimension as usize)
            .collect::<Vec<_>>();
        let bytes = usize::try_from(meta.byte_length)
            .map_err(|_| SharedError::message("safetensors: payload exceeds host address range"))?;
        crate::value::validate_storage_bytes(
            effect_torch_runtime::ValueSpec::dense(dtype, &shape),
            bytes,
        )
        .map_err(SharedError::message)?;
        if cancelled() {
            return Err(SharedError::Cancelled);
        }
    }
    let entries = plan.load(cancelled, |meta, bytes| {
        let dtype = runtime_dtype(meta.dtype).map_err(SharedError::message)?;
        let shape = meta
            .shape
            .iter()
            .map(|&dimension| dimension as usize)
            .collect::<Vec<_>>();
        CudaValue::from_dense_bytes(device.clone(), shape, dtype, &bytes)
            .map_err(SharedError::message)
    })?;
    Ok(LoadedArchive { entries, metadata })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_inspection_needs_no_cuda_device_or_u32_element_count() {
        let path = std::env::temp_dir().join(format!(
            "cuda-inspection-{}-{}.safetensors",
            std::process::id(),
            NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        let header = r#"{"unused":{"dtype":"U8","shape":[2,2147483648],"data_offsets":[0,4294967296]},"selected":{"dtype":"BF16","shape":[1],"data_offsets":[4294967296,4294967298]}}"#;
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        file.set_len(8 + header.len() as u64 + 4_294_967_298)
            .unwrap();
        drop(file);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let inspection = runtime
            .block_on(super::super::inspect_safetensors(
                path.to_str().unwrap().to_string(),
                None,
            ))
            .unwrap();
        assert_eq!(inspection.entries.len(), 2);
        assert_eq!(inspection.entries[0].name, "selected");
        assert_eq!(inspection.entries[0].dtype, "bf16");
        assert_eq!(inspection.entries[0].byte_length, 2.0);
        assert_eq!(inspection.entries[1].shape, vec![2, 2_147_483_648]);
        assert_eq!(inspection.entries[1].byte_length, 4_294_967_296.0);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn archive_preserves_integer_and_float_bits() {
        let integers = [
            i64::MIN,
            i64::MIN + 1,
            -9_007_199_254_740_993,
            9_007_199_254_740_993,
            i64::MAX,
        ];
        let i64_bytes = integers
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let half_bytes = [0x0000u16, 0x8000, 0x0001, 0x7c01, 0x7e42]
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect::<Vec<_>>();
        let entries = vec![
            (
                "i64".to_string(),
                Dtype::I64,
                vec![integers.len()],
                i64_bytes,
            ),
            ("f16".to_string(), Dtype::F16, vec![5], half_bytes.clone()),
            ("bf16".to_string(), Dtype::BF16, vec![5], half_bytes),
            (
                "u32".to_string(),
                Dtype::U32,
                vec![1],
                u32::MAX.to_le_bytes().to_vec(),
            ),
            ("u8".to_string(), Dtype::U8, vec![2], vec![0, 255]),
            (
                "f32".to_string(),
                Dtype::F32,
                vec![1],
                0x7f800123u32.to_le_bytes().to_vec(),
            ),
            (
                "f64".to_string(),
                Dtype::F64,
                vec![1],
                0x7ff0000000000123u64.to_le_bytes().to_vec(),
            ),
        ];
        let metadata = HashMap::from([("test".to_string(), "exact-bits".to_string())]);
        let encoded = encode_archive(&entries, &metadata).unwrap();
        let (decoded, actual_metadata) = decode_archive(&encoded).unwrap();
        assert_eq!(actual_metadata, metadata);
        for (name, dtype, shape, bytes) in entries {
            let view = decoded.tensor(&name).unwrap();
            assert_eq!(view.dtype(), dtype);
            assert_eq!(view.shape(), shape);
            assert_eq!(view.data(), bytes);
        }
    }

    #[test]
    fn archive_rejects_unsupported_scalar_dtype() {
        let encoded = encode_archive(
            &[("i32".to_string(), Dtype::I32, vec![1], vec![0; 4])],
            &HashMap::new(),
        )
        .unwrap();
        assert!(decode_archive(&encoded).is_err());
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn device_archive_preserves_i64_and_half_payloads() {
        let device = CudaDevice::get(0).unwrap();
        let integers = [
            i64::MIN + 1,
            -9_007_199_254_740_993,
            9_007_199_254_740_993,
            i64::MAX,
        ];
        let half = [1u16, 0x8000, 0x7c01, 0x7e42]
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect::<Vec<_>>();
        let values = HashMap::from([
            (
                "i64".to_string(),
                CudaValue::from_dense_bytes(
                    device.clone(),
                    vec![4],
                    DType::I64,
                    &integers
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap(),
            ),
            (
                "f16".to_string(),
                CudaValue::from_dense_bytes(device.clone(), vec![4], DType::F16, &half).unwrap(),
            ),
            (
                "bf16".to_string(),
                CudaValue::from_dense_bytes(device.clone(), vec![4], DType::BF16, &half).unwrap(),
            ),
        ]);
        let path =
            std::env::temp_dir().join(format!("cuda-storage-{}.safetensors", std::process::id()));
        save(&values, &HashMap::new(), path.to_str().unwrap()).unwrap();
        let loaded = load(path.to_str().unwrap(), None, device, &|| false).unwrap();
        std::fs::remove_file(path).unwrap();
        for (name, value) in loaded.entries {
            assert_eq!(
                value.read_storage_bytes().unwrap(),
                values[&name].read_storage_bytes().unwrap()
            );
            if name == "i64" {
                assert_eq!(value.readback_i64().unwrap(), integers);
            }
        }
    }
}

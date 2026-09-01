use crate::{CudaDevice, CudaValue};
use effect_torch_runtime::DType;
use safetensors::tensor::{serialize, Dtype, SafeTensors, TensorView};
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
    if value.dtype() == DType::I64 {
        let values = value.readback_i64()?;
        let mut output = Vec::with_capacity(values.len() * 8);
        for value in values {
            output.extend_from_slice(&value.to_le_bytes());
        }
        return Ok(output);
    }
    let values = value.readback()?;
    let dtype = value.dtype();
    let mut output = Vec::with_capacity(values.len() * dtype.size_in_bytes());
    for value in values {
        match dtype {
            DType::F64 => output.extend_from_slice(&value.to_le_bytes()),
            DType::F32 => output.extend_from_slice(&(value as f32).to_le_bytes()),
            DType::F16 => output.extend_from_slice(&half::f16::from_f64(value).to_le_bytes()),
            DType::BF16 => output.extend_from_slice(&half::bf16::from_f64(value).to_le_bytes()),
            DType::I64 => output.extend_from_slice(&(value as i64).to_le_bytes()),
            DType::U32 => output.extend_from_slice(&(value as u32).to_le_bytes()),
            DType::U8 => output.push(value as u8),
        }
    }
    Ok(output)
}

fn values_from_bytes(bytes: &[u8], shape: &[usize], dtype: DType) -> Result<Vec<f64>, String> {
    let elements = shape
        .iter()
        .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
        .ok_or_else(|| "safetensors: tensor element count overflows".to_string())?;
    let expected = elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| "safetensors: tensor byte length overflows".to_string())?;
    if bytes.len() != expected {
        return Err(format!(
            "safetensors: expected {expected} bytes for {dtype} tensor with shape {shape:?}, got {}",
            bytes.len()
        ));
    }
    let width = dtype.size_in_bytes();
    Ok(bytes
        .chunks_exact(width)
        .map(|chunk| match dtype {
            DType::F64 => f64::from_le_bytes(chunk.try_into().expect("validated chunk")),
            DType::F32 => f32::from_le_bytes(chunk.try_into().expect("validated chunk")) as f64,
            DType::F16 => {
                half::f16::from_le_bytes(chunk.try_into().expect("validated chunk")).to_f64()
            }
            DType::BF16 => {
                half::bf16::from_le_bytes(chunk.try_into().expect("validated chunk")).to_f64()
            }
            DType::I64 => i64::from_le_bytes(chunk.try_into().expect("validated chunk")) as f64,
            DType::U32 => u32::from_le_bytes(chunk.try_into().expect("validated chunk")) as f64,
            DType::U8 => chunk[0] as f64,
        })
        .collect())
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
    let views = owned
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            TensorView::new(*dtype, shape.clone(), bytes).map(|view| (name.as_str(), view))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    let encoded = serialize(views, Some(metadata.clone())).map_err(|error| error.to_string())?;
    atomic_write(Path::new(path), &encoded)
}

pub struct LoadedArchive {
    pub entries: Vec<(String, CudaValue)>,
    pub metadata: HashMap<String, String>,
}

pub fn load(path: &str, device: Arc<CudaDevice>) -> Result<LoadedArchive, String> {
    let raw = std::fs::read(path).map_err(|error| error.to_string())?;
    let (_, parsed_metadata) =
        SafeTensors::read_metadata(&raw).map_err(|error| error.to_string())?;
    let metadata = parsed_metadata.metadata().clone().unwrap_or_default();
    let tensors = SafeTensors::deserialize(&raw).map_err(|error| error.to_string())?;
    let mut entries = Vec::with_capacity(tensors.len());
    for name in tensors.names() {
        let view = tensors.tensor(name).map_err(|error| error.to_string())?;
        let dtype = runtime_dtype(view.dtype())?;
        let shape = view.shape().to_vec();
        let value = if dtype == DType::I64 {
            let values = view
                .data()
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("validated chunk")))
                .collect::<Vec<_>>();
            CudaValue::from_i64_host(device.clone(), shape, &values)?
        } else {
            let values = values_from_bytes(view.data(), &shape, dtype)?;
            CudaValue::from_host(device.clone(), shape, dtype, &values)?
        };
        entries.push((name.to_string(), value));
    }
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(LoadedArchive { entries, metadata })
}

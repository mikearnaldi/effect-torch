use super::{failure, run_compute, CancellationState, CancellationToken, NativeTensor};
use crate::{CudaDevice, CudaValue};
use effect_torch_runtime::{
    parse_gguf, read_gguf_tensor_into, DType, GgufMetadataArray, GgufMetadataEntry,
    GgufMetadataValue, GgufParseError, GgufTensorDescriptor,
};
use napi::{Error, Result, Status};
use napi_derive::napi;
use std::fs::File;
use std::sync::Arc;

#[napi(object, object_from_js = false)]
pub struct NativeGgufMetadataEntry {
    pub key: String,
    pub kind: String,
    pub number_value: Option<f64>,
    pub string_value: Option<String>,
    pub boolean_value: Option<bool>,
    pub number_array: Option<Vec<f64>>,
    pub string_array: Option<Vec<String>>,
    pub boolean_array: Option<Vec<bool>>,
}

#[napi(object, object_from_js = false)]
pub struct NativeGgufTensorDescriptor {
    pub name: String,
    pub format: String,
    pub logical_shape: Vec<f64>,
    pub logical_dtype: String,
    pub physical_shape: Vec<f64>,
    pub physical_dtype: String,
}

#[napi(object, object_from_js = false)]
pub struct NativeGgufInspection {
    pub metadata: Vec<NativeGgufMetadataEntry>,
    pub tensors: Vec<NativeGgufTensorDescriptor>,
}

#[napi(object, object_from_js = false)]
pub struct NativeGgufLoadedEntry {
    pub descriptor: NativeGgufTensorDescriptor,
    pub tensor: NativeTensor,
}

#[napi(object, object_from_js = false)]
pub struct NativeGgufArchive {
    pub entries: Vec<NativeGgufLoadedEntry>,
}

fn metadata_entry(entry: GgufMetadataEntry) -> NativeGgufMetadataEntry {
    let mut output = NativeGgufMetadataEntry {
        key: entry.key,
        kind: entry.value.kind().to_string(),
        number_value: None,
        string_value: None,
        boolean_value: None,
        number_array: None,
        string_array: None,
        boolean_array: None,
    };
    macro_rules! number {
        ($value:expr) => {
            output.number_value = Some($value as f64)
        };
    }
    macro_rules! numbers {
        ($values:expr) => {
            output.number_array = Some($values.into_iter().map(|value| value as f64).collect())
        };
    }
    match entry.value {
        GgufMetadataValue::U8(value) => number!(value),
        GgufMetadataValue::I8(value) => number!(value),
        GgufMetadataValue::U16(value) => number!(value),
        GgufMetadataValue::I16(value) => number!(value),
        GgufMetadataValue::U32(value) => number!(value),
        GgufMetadataValue::I32(value) => number!(value),
        GgufMetadataValue::F32(value) => number!(value),
        GgufMetadataValue::Bool(value) => output.boolean_value = Some(value),
        GgufMetadataValue::String(value) => output.string_value = Some(value),
        GgufMetadataValue::U64(value) => number!(value),
        GgufMetadataValue::I64(value) => number!(value),
        GgufMetadataValue::F64(value) => number!(value),
        GgufMetadataValue::Array(array) => match array {
            GgufMetadataArray::U8(values) => numbers!(values),
            GgufMetadataArray::I8(values) => numbers!(values),
            GgufMetadataArray::U16(values) => numbers!(values),
            GgufMetadataArray::I16(values) => numbers!(values),
            GgufMetadataArray::U32(values) => numbers!(values),
            GgufMetadataArray::I32(values) => numbers!(values),
            GgufMetadataArray::F32(values) => numbers!(values),
            GgufMetadataArray::Bool(values) => output.boolean_array = Some(values),
            GgufMetadataArray::String(values) => output.string_array = Some(values),
            GgufMetadataArray::U64(values) => numbers!(values),
            GgufMetadataArray::I64(values) => numbers!(values),
            GgufMetadataArray::F64(values) => numbers!(values),
        },
    }
    output
}

fn descriptor(value: &GgufTensorDescriptor) -> NativeGgufTensorDescriptor {
    NativeGgufTensorDescriptor {
        name: value.name.clone(),
        format: value.format.name().to_string(),
        logical_shape: value
            .logical_shape
            .iter()
            .map(|&value| value as f64)
            .collect(),
        logical_dtype: "f32".to_string(),
        physical_shape: value
            .physical_shape
            .iter()
            .map(|&value| value as f64)
            .collect(),
        physical_dtype: if value.format.name() == "F32" {
            "f32".to_string()
        } else {
            "u8".to_string()
        },
    }
}

fn gguf_error(error: GgufParseError) -> Error {
    Error::new(
        if error.is_cancelled() {
            Status::Cancelled
        } else {
            Status::GenericFailure
        },
        error.to_string(),
    )
}

fn open(path: &str) -> Result<File> {
    File::open(path).map_err(|error| failure(format!("gguf: failed to open {path:?}: {error}")))
}

#[napi]
pub async fn inspect_gguf(
    path: String,
    token: Option<&CancellationToken>,
) -> Result<NativeGgufInspection> {
    let state = token
        .map(|token| token.state.clone())
        .unwrap_or_else(|| Arc::new(CancellationState::new()));
    let notify = token.map(|token| token.notify.clone());
    run_compute(state, notify, move |cancelled, _| {
        let mut file = open(&path)?;
        let parsed = parse_gguf(&mut file, Some(cancelled)).map_err(gguf_error)?;
        Ok(NativeGgufInspection {
            metadata: parsed.metadata.into_iter().map(metadata_entry).collect(),
            tensors: parsed.tensors.iter().map(descriptor).collect(),
        })
    })
    .await
}

#[napi]
pub async fn load_gguf_for_device(
    path: String,
    device_ordinal: u32,
    token: Option<&CancellationToken>,
) -> Result<NativeGgufArchive> {
    let device = CudaDevice::get(device_ordinal).map_err(failure)?;
    let state = token
        .map(|token| token.state.clone())
        .unwrap_or_else(|| Arc::new(CancellationState::new()));
    let notify = token.map(|token| token.notify.clone());
    run_compute(state, notify, move |cancelled, _| {
        let mut file = open(&path)?;
        let parsed = parse_gguf(&mut file, Some(cancelled)).map_err(gguf_error)?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(parsed.tensors.len())
            .map_err(|_| failure("gguf: tensor catalog is too large"))?;
        for tensor in parsed.tensors {
            let dtype = if tensor.format.name() == "F32" {
                DType::F32
            } else {
                DType::U8
            };
            let width = if dtype == DType::F32 { 4 } else { 1 };
            let elements = tensor
                .physical_shape
                .iter()
                .try_fold(1usize, |total, dimension| {
                    total
                        .checked_mul(*dimension)
                        .ok_or_else(|| failure("gguf: tensor shape overflowed"))
                })?;
            let mut bytes = vec![0u8; elements * width];
            read_gguf_tensor_into(&file, &tensor, &mut bytes, Some(cancelled))
                .map_err(gguf_error)?;
            let value = if dtype == DType::F32 {
                let values = bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
                    .collect::<Vec<_>>();
                CudaValue::from_f32_host(device.clone(), tensor.physical_shape.clone(), &values)
            } else {
                CudaValue::from_packed_host(device.clone(), tensor.physical_shape.clone(), &bytes)
            }
            .map_err(failure)?;
            entries.push(NativeGgufLoadedEntry {
                descriptor: descriptor(&tensor),
                tensor: NativeTensor::wrap(value),
            });
        }
        Ok(NativeGgufArchive { entries })
    })
    .await
}

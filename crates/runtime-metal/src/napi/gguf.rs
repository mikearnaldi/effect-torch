//! Cancellable GGUF inspection and direct-to-Metal loading for Node.
//!
//! Inspection converts the strict backend-neutral GGUF catalog into
//! JavaScript-safe object records. Loading parses the same catalog, allocates
//! one shared Metal destination per tensor, and streams bytes directly into
//! that destination on a blocking worker. Quantized tensors remain opaque u8
//! storage with separate logical f32 metadata; F32 tensors retain f32 storage.
//! Cancellation is polled by the parser and tensor reader, and partially built
//! archives are dropped rather than published on failure.

use super::{run_compute, value, CancellationToken, NativeTensor};
use effect_torch_runtime::{
    parse_gguf, read_gguf_tensor_into, DType, GgufMetadataArray, GgufMetadataEntry,
    GgufMetadataValue, GgufParseError, GgufTensorDescriptor,
};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::fs::File;

/// JavaScript representation of one scalar or homogeneous-array metadata item.
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

/// JavaScript-safe logical and physical description of one GGUF tensor.
#[napi(object, object_from_js = false)]
pub struct NativeGgufTensorDescriptor {
    pub name: String,
    pub format: String,
    pub logical_shape: Vec<f64>,
    pub logical_dtype: String,
    pub physical_shape: Vec<f64>,
    pub physical_dtype: String,
}

/// Header/catalog inspection result without loaded tensor data.
#[napi(object, object_from_js = false)]
pub struct NativeGgufInspection {
    pub metadata: Vec<NativeGgufMetadataEntry>,
    pub tensors: Vec<NativeGgufTensorDescriptor>,
}

/// One loaded tensor paired with the descriptor used to validate it.
#[napi(object, object_from_js = false)]
pub struct NativeGgufLoadedEntry {
    pub descriptor: NativeGgufTensorDescriptor,
    pub tensor: NativeTensor,
}

/// Complete loaded archive returned atomically to JavaScript.
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
    File::open(path).map_err(|error| {
        Error::new(
            Status::GenericFailure,
            format!("gguf: failed to open {path:?}: {error}"),
        )
    })
}

/// Parses and validates a GGUF file without loading tensor payloads.
#[napi]
pub async fn inspect_gguf(
    path: String,
    token: Option<&CancellationToken>,
) -> Result<NativeGgufInspection> {
    run_compute(token, move |cancelled, _state| {
        let mut file = open(&path)?;
        let parsed = parse_gguf(&mut file, Some(cancelled)).map_err(gguf_error)?;
        Ok(NativeGgufInspection {
            metadata: parsed.metadata.into_iter().map(metadata_entry).collect(),
            tensors: parsed.tensors.iter().map(descriptor).collect(),
        })
    })
    .await
}

/// Parses a GGUF file and loads every supported tensor into Metal storage.
#[napi]
pub async fn load_gguf(
    path: String,
    token: Option<&CancellationToken>,
) -> Result<NativeGgufArchive> {
    run_compute(token, move |cancelled, _state| {
        let mut file = open(&path)?;
        let parsed = parse_gguf(&mut file, Some(cancelled)).map_err(gguf_error)?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(parsed.tensors.len())
            .map_err(|_| Error::new(Status::GenericFailure, "gguf: tensor catalog is too large"))?;
        for tensor in parsed.tensors {
            let dtype = if tensor.format.name() == "F32" {
                DType::F32
            } else {
                DType::U8
            };
            if dtype == DType::F32 && cfg!(target_endian = "big") {
                return Err(Error::new(
                    Status::GenericFailure,
                    "gguf: direct F32 loading is not supported on big-endian targets",
                ));
            }
            let mut loaded = value::empty_shared_value(&tensor.physical_shape, dtype)
                .map_err(|error| Error::new(Status::GenericFailure, format!("gguf: {error}")))?;
            value::write_value_bytes(&mut loaded, |bytes| {
                read_gguf_tensor_into(&file, &tensor, bytes, Some(cancelled))
            })
            .map_err(|error| Error::new(Status::GenericFailure, format!("gguf: {error}")))?
            .map_err(gguf_error)?;
            entries.push(NativeGgufLoadedEntry {
                descriptor: descriptor(&tensor),
                tensor: NativeTensor::wrap(loaded),
            });
        }
        Ok(NativeGgufArchive { entries })
    })
    .await
}

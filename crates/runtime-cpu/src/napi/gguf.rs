//! GGUF archive inspection and loading over the napi boundary.
//!
//! [`inspect_gguf`] parses only the header, which contains metadata and the
//! tensor catalog. [`load_gguf`] reads selected tensor payloads directly
//! into new CPU tensor storage. F32 tensors use f32 storage. Quantized and
//! other formats use packed `u8` storage with the physical shape from the
//! descriptor. Both functions are async and check the cancellation token
//! while parsing and reading.

use super::{run_compute, CancellationToken, NativeTensor};
use crate::{Tensor, Value};
use effect_torch_runtime::{
    parse_gguf, read_gguf_tensor_into, DType, GgufMetadataArray, GgufMetadataEntry,
    GgufMetadataValue, GgufParseError, GgufTensorDescriptor,
};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::fs::File;

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
    File::open(path).map_err(|error| {
        Error::new(
            Status::GenericFailure,
            format!("gguf: failed to open {path:?}: {error}"),
        )
    })
}

/// Parses the GGUF header at `path` and returns its metadata and tensor
/// catalog without reading payloads.
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

/// Loads selected tensors of the GGUF archive at `path` into CPU tensors.
/// Omitted names load all tensors; an empty list loads none.
/// Direct f32 loading requires a little-endian target. Quantized formats use
/// packed `u8` payloads.
#[napi]
pub async fn load_gguf(
    path: String,
    token: Option<&CancellationToken>,
    names: Option<Vec<String>>,
) -> Result<NativeGgufArchive> {
    run_compute(token, move |cancelled, _state| {
        let mut file = open(&path)?;
        let parsed = parse_gguf(&mut file, Some(cancelled)).map_err(gguf_error)?;
        load_selected(&file, parsed, names.as_deref(), cancelled)
    })
    .await
}

fn load_selected(
    file: &File,
    parsed: effect_torch_runtime::GgufFile,
    names: Option<&[String]>,
    cancelled: &effect_torch_runtime::CancellationFlag,
) -> Result<NativeGgufArchive> {
    let tensors = parsed
        .select_tensors(names, Some(cancelled))
        .map_err(gguf_error)?;
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(tensors.len())
        .map_err(|_| Error::new(Status::GenericFailure, "gguf: tensor catalog is too large"))?;
    for tensor in tensors {
        if cancelled.is_cancelled() {
            return Err(gguf_error(GgufParseError::Cancelled));
        }
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
        let mut value = Tensor::empty(&tensor.physical_shape, dtype);
        let mut destination = value
            .destination()
            .map_err(|error| Error::new(Status::GenericFailure, format!("gguf: {error}")))?;
        destination
            .write_bytes("gguf", |bytes| {
                read_gguf_tensor_into(file, &tensor, bytes, Some(cancelled))
            })
            .map_err(|error| Error::new(Status::GenericFailure, format!("gguf: {error}")))?
            .map_err(gguf_error)?;
        drop(destination);
        let mut value = Value::dense(value);
        if let effect_torch_runtime::StorageRepresentation::Packed(
            effect_torch_runtime::PackedFormat::GgmlKQuant(codec),
        ) = tensor.value_spec().storage.representation
        {
            value = value
                .with_packed_storage(codec, tensor.logical_shape.clone())
                .map_err(super::err::to_napi_err)?;
        }
        entries.push(NativeGgufLoadedEntry {
            descriptor: descriptor(&tensor),
            tensor: NativeTensor::wrap(value),
        });
        #[cfg(test)]
        tests::after_load(&entries.last().unwrap().tensor);
    }
    Ok(NativeGgufArchive { entries })
}

#[cfg(test)]
#[path = "../../../runtime/src/gguf/test_fixture.rs"]
mod test_fixture;

#[cfg(test)]
mod tests {
    use super::*;
    use effect_torch_runtime::{CancellationFlag, GgmlKQuant, PackedFormat, StorageRepresentation};
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;

    thread_local! {
        static AFTER_LOAD: RefCell<Option<Box<dyn FnMut(&NativeTensor)>>> = RefCell::new(None);
    }

    // Observe native ownership and interrupt at an exact payload boundary.
    pub(super) fn after_load(tensor: &NativeTensor) {
        AFTER_LOAD.with_borrow_mut(|hook| {
            if let Some(hook) = hook {
                hook(tensor);
            }
        });
    }

    struct HookGuard;

    impl Drop for HookGuard {
        fn drop(&mut self) {
            AFTER_LOAD.with_borrow_mut(|hook| *hook = None);
        }
    }

    fn bytes(tensor: &NativeTensor) -> Vec<u8> {
        let value = tensor.value_cloned().unwrap();
        super::super::safetensors::tensor_bytes(&Value::dense(value.tensor().clone())).unwrap()
    }

    #[test]
    fn selected_load_skips_unused_8_gib_and_preserves_packed_storage() {
        let mut fixture = test_fixture::SparseGguf::new(1 << 33);
        let names = vec!["packed".into(), "dense".into()];
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let archive = runtime
            .block_on(load_gguf(
                fixture.path.to_str().unwrap().into(),
                None,
                Some(names.clone()),
            ))
            .unwrap();
        let empty = runtime
            .block_on(load_gguf(
                fixture.path.to_str().unwrap().into(),
                None,
                Some(vec![]),
            ))
            .unwrap();
        assert!(empty.entries.is_empty());
        assert_eq!(archive.entries.len(), 2);
        assert_eq!(archive.entries[0].descriptor.name, "dense");
        assert_eq!(bytes(&archive.entries[0].tensor), fixture.dense);
        assert_eq!(bytes(&archive.entries[1].tensor), fixture.packed);
        let packed = archive.entries[1].tensor.value_cloned().unwrap();
        assert_eq!(packed.dtype(), DType::F32);
        assert_eq!(packed.shape(), [2, 256]);
        assert_eq!(packed.tensor().dtype(), DType::U8);
        assert_eq!(
            packed.storage().representation,
            StorageRepresentation::Packed(PackedFormat::GgmlKQuant(GgmlKQuant::Q4K))
        );
        assert_eq!(archive.entries[1].descriptor.logical_dtype, "f32");
        assert_eq!(archive.entries[1].descriptor.physical_dtype, "u8");
        assert_eq!(archive.entries[1].descriptor.physical_shape, [2.0, 144.0]);
        let parsed = parse_gguf(&mut fixture.file, None).unwrap();
        fixture.truncate_unused();
        let archive = load_selected(
            &fixture.file,
            parsed,
            Some(&names),
            &CancellationFlag::new(),
        )
        .unwrap();
        assert_eq!(bytes(&archive.entries[0].tensor), fixture.dense);
        assert_eq!(bytes(&archive.entries[1].tensor), fixture.packed);
    }

    #[test]
    fn selection_preflight_and_cancellation_precede_cpu_storage_allocation() {
        let mut fixture = test_fixture::SparseGguf::new(1 << 33);
        let parsed = parse_gguf(&mut fixture.file, None).unwrap();
        fixture.file.set_len(0).unwrap();
        let cancelled = CancellationFlag::new();
        let _guard = crate::storage::ExecutableAllocationGuard::enter();
        for (names, expected) in [
            (vec!["dense".into(), "missing".into()], "unknown selected"),
            (vec!["dense".into(), "dense".into()], "duplicate selected"),
            (vec!["dense".into(), "".into()], "must not be empty"),
        ] {
            let error = load_selected(&fixture.file, parsed.clone(), Some(&names), &cancelled)
                .err()
                .unwrap();
            assert!(error.reason.contains(expected), "{error}");
        }
        assert!(
            load_selected(&fixture.file, parsed.clone(), Some(&[]), &cancelled)
                .unwrap()
                .entries
                .is_empty()
        );
        cancelled.cancel();
        let error = load_selected(&fixture.file, parsed, None, &cancelled)
            .err()
            .unwrap();
        assert_eq!(error.status, Status::Cancelled);
    }

    #[test]
    fn partial_archive_is_dropped_on_read_failure_and_cancellation() {
        for cancel in [false, true] {
            let mut fixture = test_fixture::SparseGguf::new(4);
            let parsed = parse_gguf(&mut fixture.file, None).unwrap();
            let cancelled = Arc::new(CancellationFlag::new());
            let flag = cancelled.clone();
            let file = fixture.file.try_clone().unwrap();
            let slots = Rc::new(RefCell::new(Vec::new()));
            let observed = slots.clone();
            AFTER_LOAD.with_borrow_mut(|hook| {
                *hook = Some(Box::new(move |tensor| {
                    observed.borrow_mut().push(Arc::downgrade(&tensor.slot));
                    if cancel {
                        flag.cancel();
                    } else {
                        file.set_len(0).unwrap();
                    }
                }));
            });
            let _guard = HookGuard;
            let error = load_selected(
                &fixture.file,
                parsed,
                Some(&["dense".into(), "packed".into()]),
                &cancelled,
            )
            .err()
            .unwrap();
            assert_eq!(
                error.status,
                if cancel {
                    Status::Cancelled
                } else {
                    Status::GenericFailure
                }
            );
            assert_eq!(slots.borrow().len(), 1);
            assert!(
                slots.borrow()[0].upgrade().is_none(),
                "partial native handle leaked"
            );
        }
    }

    #[test]
    fn omitted_names_load_all_and_empty_names_still_validate_the_full_header() {
        let mut fixture = test_fixture::SparseGguf::new(4);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let path = fixture.path.to_str().unwrap().to_string();
        let archive = runtime
            .block_on(load_gguf(path.clone(), None, None))
            .unwrap();
        assert_eq!(archive.entries.len(), 3);
        for (format, expected) in [(1, "unsupported GGML"), (12, "256-value block")] {
            fixture.set_unused_format(format);
            for names in [Some(vec![]), Some(vec!["dense".into()])] {
                let error = runtime
                    .block_on(load_gguf(path.clone(), None, names))
                    .err()
                    .unwrap();
                assert!(error.reason.contains(expected), "{error}");
            }
        }
        fixture.set_unused_format(0);
        fixture.truncate_unused();
        for names in [Some(vec![]), Some(vec!["dense".into()])] {
            let error = runtime
                .block_on(load_gguf(path.clone(), None, names))
                .err()
                .unwrap();
            assert!(error.reason.contains("exceeds file size"), "{error}");
        }
    }
}

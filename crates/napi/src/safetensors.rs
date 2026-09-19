//! Backend-neutral safetensors header indexing and selective payload reading.
//!
//! Every native backend shares this reader so header parsing, range validation,
//! multi-shard indexing, and cancellation behave identically. The reader never
//! maps or copies a whole archive: it reads the 8-byte header length, the JSON
//! header, and only the payload byte ranges selected by the caller.
//!
//! A path is either a standalone .safetensors archive or a Hugging Face
//! *.safetensors.index.json weight map. Index shard names are resolved
//! relative to the index file without lexical parent traversal. Shard symlinks
//! may point outside that directory, as in Hugging Face snapshots. Inspecting an
//! index reads every shard header but no payload bytes; planning a selective
//! load opens only the shards that contain a selected tensor, so a missing
//! unselected shard never fails the load.
//!
//! Header validation is strict about format geometry and lenient everywhere
//! else. It rejects truncated files, headers whose declared length exceeds the
//! file, duplicate JSON keys, unknown dtypes, duplicate or overlapping byte
//! ranges, payloads that run past the data section, and shapes whose element
//! count or byte length would overflow. It does not impose a device or element
//! count cap, because inspection must describe large tensors that the caller
//! never loads.
//!
//! Shard __metadata__ maps are merged deterministically. Keys with equal
//! values are deduplicated; a key with conflicting values is an error. A load
//! merges only the shards it opens, which for a selective load is the minimal
//! set required by the selection. Index metadata, including numeric total_size,
//! is bookkeeping and is not included in the string archive metadata.

use safetensors::tensor::Dtype;
use serde::de::{self, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

/// Largest integer representable exactly by a JavaScript number. The native
/// inspection ABI carries byte lengths as f64, so lengths above this boundary
/// are rejected instead of returning a silently rounded value.
pub const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

/// Payload read granularity. Cancellation is polled between chunks so a large
/// tensor cannot block cooperative interruption.
const READ_CHUNK_BYTES: usize = 1 << 20;
// Same header limit as the safetensors crate. Bound malformed sparse headers
// before reserving host memory.
const MAX_HEADER_BYTES: u64 = 100_000_000;

/// Failure raised by the shared safetensors reader.
#[derive(Debug)]
pub enum Error {
    /// Cancellation won before the operation published a result.
    Cancelled,
    /// Format, validation, or I/O failure.
    Message(String),
}

impl Error {
    /// Builds a message failure from any displayable value.
    pub fn message(message: impl Into<String>) -> Self {
        Error::Message(message.into())
    }

    /// Maps the failure onto the N-API boundary. Cancellation keeps its
    /// dedicated status so callers observe an aborted operation.
    pub fn into_napi(self) -> napi::Error {
        match self {
            Error::Cancelled => napi::Error::new(napi::Status::Cancelled, "operation aborted"),
            Error::Message(message) => napi::Error::new(napi::Status::GenericFailure, message),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Cancelled => formatter.write_str("operation aborted"),
            Error::Message(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for Error {}

impl From<String> for Error {
    fn from(message: String) -> Self {
        Error::Message(message)
    }
}

impl From<&str> for Error {
    fn from(message: &str) -> Self {
        Error::Message(message.to_string())
    }
}

/// Metadata for one tensor from a standalone header or index shard header.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    /// Tensor name.
    pub name: String,
    /// Safetensors scalar type.
    pub dtype: Dtype,
    /// Shape with each dimension validated to fit in the u32 ABI shape.
    pub shape: Vec<u32>,
    /// Exact little-endian payload length in bytes.
    pub byte_length: u64,
}

/// Header-only view of an archive.
#[derive(Debug, Clone)]
pub struct Inspection {
    /// Every tensor entry, sorted by name.
    pub entries: Vec<TensorMeta>,
    /// Merged archive metadata.
    pub metadata: HashMap<String, String>,
}

/// A validated, ready-to-read selection of tensor payloads. Planning performs
/// all index resolution, header parsing, and geometry validation without
/// reading payload bytes, so callers can preflight dtypes before allocating.
pub struct LoadPlan {
    sources: Vec<ShardSource>,
    entries: Vec<SelectedEntry>,
    metadata: HashMap<String, String>,
}

struct ShardSource {
    path: PathBuf,
    data_start: u64,
    file: File,
}

struct SelectedEntry {
    meta: TensorMeta,
    source: usize,
    begin: u64,
    end: u64,
}

impl LoadPlan {
    /// Selected tensor descriptors in name-sorted order.
    pub fn entries(&self) -> Vec<TensorMeta> {
        self.entries
            .iter()
            .map(|entry| entry.meta.clone())
            .collect()
    }

    /// Merged metadata of the shards this plan opened.
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// Returns true when the selection contains no tensors.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Reads each selected payload, bounded to one tensor at a time, and hands
    /// it to on_tensor. Cancellation is polled before every tensor and between
    /// payload chunks. If on_tensor or a read fails, the values already
    /// produced are dropped, releasing their handles exactly once.
    pub fn load<T, F>(
        self,
        cancelled: &dyn Fn() -> bool,
        mut on_tensor: F,
    ) -> Result<Vec<(String, T)>, Error>
    where
        F: FnMut(&TensorMeta, Vec<u8>) -> Result<T, Error>,
    {
        let LoadPlan {
            mut sources,
            entries,
            ..
        } = self;
        let mut loaded: Vec<(String, T)> = Vec::new();
        loaded.try_reserve_exact(entries.len()).map_err(|_| {
            Error::message("safetensors: selected tensor count is too large to stage")
        })?;
        for selected in entries {
            if cancelled() {
                return Err(Error::Cancelled);
            }
            let source = &mut sources[selected.source];
            let offset = source
                .data_start
                .checked_add(selected.begin)
                .ok_or_else(|| Error::message("safetensors: tensor byte offset overflows"))?;
            let length = selected.end - selected.begin;
            let bytes = read_payload(&mut source.file, &source.path, offset, length, cancelled)?;
            if cancelled() {
                return Err(Error::Cancelled);
            }
            let value = on_tensor(&selected.meta, bytes)?;
            loaded.push((selected.meta.name, value));
        }
        if cancelled() {
            return Err(Error::Cancelled);
        }
        Ok(loaded)
    }
}

/// Reads header geometry without loading any payload.
pub fn inspect(path: &str, cancelled: &dyn Fn() -> bool) -> Result<Inspection, Error> {
    if cancelled() {
        return Err(Error::Cancelled);
    }
    let path = Path::new(path);
    if is_index(path) {
        inspect_index(path, cancelled)
    } else {
        inspect_standalone(path, cancelled)
    }
}

/// Resolves a selection and validates every selected tensor without reading a
/// payload. Missing, duplicate, and reserved names are rejected.
pub fn plan(
    path: &str,
    names: Option<&[String]>,
    cancelled: &dyn Fn() -> bool,
) -> Result<LoadPlan, Error> {
    if cancelled() {
        return Err(Error::Cancelled);
    }
    let path = Path::new(path);
    if is_index(path) {
        plan_index(path, names, cancelled)
    } else {
        plan_standalone(path, names, cancelled)
    }
}

/// Lowercase stable dtype name for the inspection ABI.
pub fn dtype_name(dtype: Dtype) -> String {
    format!("{dtype:?}").to_lowercase()
}

/// Converts a validated byte length to the f64 ABI value, rejecting lengths
/// outside the JavaScript safe-integer range.
pub fn byte_length_f64(byte_length: u64) -> Result<f64, Error> {
    if byte_length > MAX_SAFE_INTEGER {
        return Err(Error::message(format!(
            "safetensors: tensor byte length {byte_length} exceeds the JavaScript safe integer range"
        )));
    }
    Ok(byte_length as f64)
}

fn is_index(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".safetensors.index.json"))
}

fn inspect_standalone(path: &Path, cancelled: &dyn Fn() -> bool) -> Result<Inspection, Error> {
    if cancelled() {
        return Err(Error::Cancelled);
    }
    let mut file = open(path)?;
    let header = parse_header(&mut file, path, cancelled)?;
    if cancelled() {
        return Err(Error::Cancelled);
    }
    let mut entries: Vec<TensorMeta> = header
        .entries
        .into_iter()
        .map(|(name, entry)| meta_from(name, &entry))
        .collect();
    entries.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(Inspection {
        entries,
        metadata: header.metadata,
    })
}

fn plan_standalone(
    path: &Path,
    names: Option<&[String]>,
    cancelled: &dyn Fn() -> bool,
) -> Result<LoadPlan, Error> {
    if cancelled() {
        return Err(Error::Cancelled);
    }
    let mut file = open(path)?;
    let header = parse_header(&mut file, path, cancelled)?;
    if cancelled() {
        return Err(Error::Cancelled);
    }
    let selected = select_names(header.entries.iter().map(|(name, _)| name.as_str()), names)?;
    let by_name: HashMap<&str, &HeaderEntry> = header
        .entries
        .iter()
        .map(|(name, entry)| (name.as_str(), entry))
        .collect();
    let mut entries = Vec::with_capacity(selected.len());
    for name in selected {
        let entry = by_name
            .get(name.as_str())
            .expect("selected names come from the parsed header");
        entries.push(SelectedEntry {
            meta: meta_from(name, entry),
            source: 0,
            begin: entry.begin,
            end: entry.end,
        });
    }
    Ok(LoadPlan {
        sources: vec![ShardSource {
            path: path.to_path_buf(),
            data_start: header.data_start,
            file,
        }],
        entries,
        metadata: header.metadata,
    })
}

fn inspect_index(path: &Path, cancelled: &dyn Fn() -> bool) -> Result<Inspection, Error> {
    let index = parse_index(path, cancelled)?;
    let mut entries = Vec::new();
    let mut metadata = HashMap::new();
    for (shard_index, shard_path) in index.shards.iter().enumerate() {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        let mut file = open(shard_path)?;
        let header = parse_header(&mut file, shard_path, cancelled)?;
        validate_shard(&index, shard_index, &header)?;
        merge_metadata(&mut metadata, &header.metadata)?;
        for (name, entry) in header.entries {
            entries.push(meta_from(name, &entry));
        }
    }
    entries.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(Inspection { entries, metadata })
}

fn plan_index(
    path: &Path,
    names: Option<&[String]>,
    cancelled: &dyn Fn() -> bool,
) -> Result<LoadPlan, Error> {
    let index = parse_index(path, cancelled)?;
    let selected = select_names(index.name_to_shard.keys().map(|name| name.as_str()), names)?;
    let mut selection: Vec<Vec<String>> = index.shards.iter().map(|_| Vec::new()).collect();
    for name in selected {
        let shard = index.name_to_shard[name.as_str()];
        selection[shard].push(name);
    }
    let mut sources: Vec<ShardSource> = Vec::new();
    let mut entries = Vec::new();
    let mut metadata = HashMap::new();
    for (shard_index, shard_names) in selection.into_iter().enumerate() {
        if shard_names.is_empty() {
            continue;
        }
        if cancelled() {
            return Err(Error::Cancelled);
        }
        let shard_path = &index.shards[shard_index];
        let mut file = open(shard_path)?;
        let header = parse_header(&mut file, shard_path, cancelled)?;
        validate_shard(&index, shard_index, &header)?;
        merge_metadata(&mut metadata, &header.metadata)?;
        let by_name: HashMap<&str, &HeaderEntry> = header
            .entries
            .iter()
            .map(|(name, entry)| (name.as_str(), entry))
            .collect();
        let source = sources.len();
        for name in &shard_names {
            let entry = by_name.get(name.as_str()).ok_or_else(|| {
                Error::message(format!(
                    "safetensors: index maps {name:?} to shard {:?} but the shard header does not contain it",
                    shard_path.display()
                ))
            })?;
            entries.push(SelectedEntry {
                meta: meta_from(name.clone(), entry),
                source,
                begin: entry.begin,
                end: entry.end,
            });
        }
        sources.push(ShardSource {
            path: shard_path.clone(),
            data_start: header.data_start,
            file,
        });
    }
    entries.sort_by(|left, right| left.meta.name.cmp(&right.meta.name));
    Ok(LoadPlan {
        sources,
        entries,
        metadata,
    })
}

fn select_names<'a>(
    available: impl Iterator<Item = &'a str>,
    names: Option<&[String]>,
) -> Result<Vec<String>, Error> {
    let available: HashSet<&str> = available.collect();
    match names {
        None => {
            let mut selected: Vec<String> =
                available.iter().map(|name| (*name).to_string()).collect();
            selected.sort();
            Ok(selected)
        }
        Some(names) => {
            let mut seen = HashSet::with_capacity(names.len());
            for name in names {
                if name == "__metadata__" {
                    return Err(Error::message(
                        "safetensors: __metadata__ is reserved and cannot be selected as a tensor",
                    ));
                }
                if !seen.insert(name.as_str()) {
                    return Err(Error::message(format!(
                        "safetensors: selected tensor name {name:?} is duplicated"
                    )));
                }
            }
            let mut selected = Vec::with_capacity(names.len());
            for name in names {
                if !available.contains(name.as_str()) {
                    return Err(Error::message(format!(
                        "safetensors: tensor {name:?} is not present in the archive"
                    )));
                }
                selected.push(name.clone());
            }
            selected.sort();
            Ok(selected)
        }
    }
}

fn meta_from(name: String, entry: &HeaderEntry) -> TensorMeta {
    TensorMeta {
        name,
        dtype: entry.dtype,
        shape: entry.shape.clone(),
        byte_length: entry.end - entry.begin,
    }
}

struct HeaderEntry {
    dtype: Dtype,
    shape: Vec<u32>,
    begin: u64,
    end: u64,
}

struct ParsedHeader {
    data_start: u64,
    entries: Vec<(String, HeaderEntry)>,
    metadata: HashMap<String, String>,
}

fn open(path: &Path) -> Result<File, Error> {
    File::open(path).map_err(|error| {
        Error::message(format!(
            "safetensors: failed to open {:?}: {error}",
            path.display()
        ))
    })
}

fn parse_header(
    file: &mut File,
    path: &Path,
    cancelled: &dyn Fn() -> bool,
) -> Result<ParsedHeader, Error> {
    let file_len = file
        .metadata()
        .map_err(|error| {
            Error::message(format!(
                "safetensors: failed to stat {:?}: {error}",
                path.display()
            ))
        })?
        .len();
    if file_len < 8 {
        return Err(Error::message(format!(
            "safetensors: {:?} is shorter than the 8-byte header length",
            path.display()
        )));
    }
    let mut length_bytes = [0u8; 8];
    file.seek(SeekFrom::Start(0))
        .and_then(|_| file.read_exact(&mut length_bytes))
        .map_err(|error| {
            Error::message(format!(
                "safetensors: failed to read the header length of {:?}: {error}",
                path.display()
            ))
        })?;
    record_read(8);
    let header_len = u64::from_le_bytes(length_bytes);
    if header_len > MAX_HEADER_BYTES {
        return Err(Error::message(
            "safetensors: header exceeds the 100 MB format limit",
        ));
    }
    let data_start = header_len
        .checked_add(8)
        .ok_or_else(|| Error::message("safetensors: header length overflows"))?;
    if data_start > file_len {
        return Err(Error::message(format!(
            "safetensors: header length {header_len} exceeds the {file_len}-byte file {:?}",
            path.display()
        )));
    }
    let header_bytes = read_payload(file, path, 8, header_len, cancelled)?;
    let object = parse_unique_object(&header_bytes)?;
    let mut metadata_value = None;
    let mut entries = Vec::with_capacity(object.len());
    for (key, value) in object {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        if key == "__metadata__" {
            metadata_value = Some(value);
        } else {
            entries.push((key, parse_tensor_info(&value)?));
        }
    }
    let metadata = parse_metadata(metadata_value)?;
    let data_len = file_len - data_start;
    validate_entries(&entries, data_len, path)?;
    Ok(ParsedHeader {
        data_start,
        entries,
        metadata,
    })
}

fn parse_metadata(value: Option<serde_json::Value>) -> Result<HashMap<String, String>, Error> {
    let Some(value) = value else {
        return Ok(HashMap::new());
    };
    let serde_json::Value::Object(object) = value else {
        return Err(Error::message(
            "safetensors: __metadata__ must be a JSON object",
        ));
    };
    let mut metadata = HashMap::with_capacity(object.len());
    for (key, value) in object {
        let value = value.as_str().ok_or_else(|| {
            Error::message(format!(
                "safetensors: metadata value for {key:?} must be a string"
            ))
        })?;
        metadata.insert(key, value.to_string());
    }
    Ok(metadata)
}

fn parse_tensor_info(value: &serde_json::Value) -> Result<HeaderEntry, Error> {
    let object = value
        .as_object()
        .ok_or_else(|| Error::message("safetensors: tensor entry must be a JSON object"))?;
    let dtype_value = object
        .get("dtype")
        .ok_or_else(|| Error::message("safetensors: tensor entry has no dtype"))?;
    let dtype: Dtype = serde_json::from_value(dtype_value.clone()).map_err(|error| {
        Error::message(format!("safetensors: unsupported tensor dtype: {error}"))
    })?;
    let shape_value = object
        .get("shape")
        .ok_or_else(|| Error::message("safetensors: tensor entry has no shape"))?;
    let shape_array = shape_value
        .as_array()
        .ok_or_else(|| Error::message("safetensors: tensor shape must be an array"))?;
    let mut shape = Vec::with_capacity(shape_array.len());
    for dimension in shape_array {
        let dimension = dimension.as_u64().ok_or_else(|| {
            Error::message("safetensors: tensor shape dimensions must be integers")
        })?;
        let dimension = u32::try_from(dimension).map_err(|_| {
            Error::message("safetensors: tensor shape dimension exceeds the u32 ABI range")
        })?;
        shape.push(dimension);
    }
    let offsets = object
        .get("data_offsets")
        .and_then(|value| value.as_array())
        .ok_or_else(|| Error::message("safetensors: tensor entry has no data_offsets array"))?;
    if offsets.len() != 2 {
        return Err(Error::message(
            "safetensors: data_offsets must contain exactly two integers",
        ));
    }
    let begin = offsets[0]
        .as_u64()
        .ok_or_else(|| Error::message("safetensors: data_offsets must be non-negative integers"))?;
    let end = offsets[1]
        .as_u64()
        .ok_or_else(|| Error::message("safetensors: data_offsets must be non-negative integers"))?;
    if begin > end {
        return Err(Error::message(
            "safetensors: data_offsets begin must not exceed end",
        ));
    }
    Ok(HeaderEntry {
        dtype,
        shape,
        begin,
        end,
    })
}

fn validate_entries(
    entries: &[(String, HeaderEntry)],
    data_len: u64,
    path: &Path,
) -> Result<(), Error> {
    let mut ranges: Vec<(u64, u64, &str)> = Vec::with_capacity(entries.len());
    for (name, entry) in entries {
        let expected = payload_byte_length(entry.dtype, &entry.shape, name)?;
        let declared = entry.end - entry.begin;
        if declared != expected {
            return Err(Error::message(format!(
                "safetensors: tensor {name:?} declares {declared} bytes but its {} shape {:?} requires {expected}",
                dtype_name(entry.dtype),
                entry.shape
            )));
        }
        if entry.end > data_len {
            return Err(Error::message(format!(
                "safetensors: tensor {name:?} ends at {} beyond the {data_len}-byte data section of {:?}",
                entry.end,
                path.display()
            )));
        }
        ranges.push((entry.begin, entry.end, name.as_str()));
    }
    ranges.sort_by_key(|(begin, end, _)| (*begin, *end));
    let mut previous_end = 0;
    for (begin, end, name) in ranges {
        if begin != previous_end {
            return Err(Error::message(format!(
                "safetensors: tensor {name:?} has a gap or overlap in its byte range"
            )));
        }
        previous_end = end;
    }
    if previous_end != data_len {
        return Err(Error::message(
            "safetensors: trailing bytes outside tensor payloads",
        ));
    }
    Ok(())
}

fn payload_byte_length(dtype: Dtype, shape: &[u32], name: &str) -> Result<u64, Error> {
    if shape.contains(&0) {
        return Ok(0);
    }
    let elements = shape
        .iter()
        .try_fold(1u64, |total, &dimension| {
            total.checked_mul(u64::from(dimension))
        })
        .ok_or_else(|| {
            Error::message(format!(
                "safetensors: tensor {name:?} element count overflows"
            ))
        })?;
    let bits = elements
        .checked_mul(dtype.bitsize() as u64)
        .ok_or_else(|| {
            Error::message(format!(
                "safetensors: tensor {name:?} byte length overflows"
            ))
        })?;
    if bits % 8 != 0 {
        return Err(Error::message(format!(
            "safetensors: tensor {name:?} has a sub-byte payload that is not byte aligned"
        )));
    }
    Ok(bits / 8)
}

fn read_payload(
    file: &mut File,
    path: &Path,
    offset: u64,
    length: u64,
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<u8>, Error> {
    if cancelled() {
        return Err(Error::Cancelled);
    }
    let length = usize::try_from(length)
        .map_err(|_| Error::message("safetensors: tensor payload does not fit in host memory"))?;
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(length).map_err(|_| {
        Error::message(format!(
            "safetensors: could not stage {length} payload bytes from {:?}",
            path.display()
        ))
    })?;
    file.seek(SeekFrom::Start(offset)).map_err(|error| {
        Error::message(format!(
            "safetensors: seek to {offset} in {:?} failed: {error}",
            path.display()
        ))
    })?;
    let mut start = 0;
    while start < length {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        let end = (start + READ_CHUNK_BYTES).min(length);
        buffer.resize(end, 0);
        file.read_exact(&mut buffer[start..end]).map_err(|error| {
            Error::message(format!(
                "safetensors: payload read from {:?} failed: {error}",
                path.display()
            ))
        })?;
        record_read(end - start);
        start = end;
    }
    if cancelled() {
        return Err(Error::Cancelled);
    }
    Ok(buffer)
}

fn merge_metadata(
    target: &mut HashMap<String, String>,
    incoming: &HashMap<String, String>,
) -> Result<(), Error> {
    for (key, value) in incoming {
        if let Some(existing) = target.get(key) {
            if existing != value {
                return Err(Error::message(format!(
                    "safetensors: conflicting archive metadata for {key:?}"
                )));
            }
        } else {
            target.insert(key.clone(), value.clone());
        }
    }
    Ok(())
}

struct Index {
    shards: Vec<PathBuf>,
    name_to_shard: HashMap<String, usize>,
    names_by_shard: Vec<HashSet<String>>,
}

impl Index {
    fn shard_of(&self, name: &str) -> Option<usize> {
        self.name_to_shard.get(name).copied()
    }
}

fn validate_shard(index: &Index, shard: usize, header: &ParsedHeader) -> Result<(), Error> {
    let path = &index.shards[shard];
    for (name, _) in &header.entries {
        if index.shard_of(name) != Some(shard) {
            return Err(Error::message(format!(
                "safetensors: shard {:?} contains {name:?}, which the index does not map to it",
                path.display()
            )));
        }
    }
    let present: HashSet<&str> = header
        .entries
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    for name in &index.names_by_shard[shard] {
        if !present.contains(name.as_str()) {
            return Err(Error::message(format!(
                "safetensors: index maps {name:?} to shard {:?} but the shard header does not contain it",
                path.display()
            )));
        }
    }
    Ok(())
}

fn parse_index(path: &Path, cancelled: &dyn Fn() -> bool) -> Result<Index, Error> {
    let mut file = open(path)?;
    let length = file
        .metadata()
        .map_err(|error| Error::message(error.to_string()))?
        .len();
    if length > MAX_HEADER_BYTES {
        return Err(Error::message(
            "safetensors: index exceeds the 100 MB header limit",
        ));
    }
    let bytes = read_payload(&mut file, path, 0, length, cancelled)?;
    let document = parse_index_document(&bytes)?;
    let weight_map = document.weight_map.ok_or_else(|| {
        Error::message(format!(
            "safetensors: index {:?} has no weight_map",
            path.display()
        ))
    })?;
    let mut sorted: BTreeMap<String, String> = BTreeMap::new();
    for (name, value) in weight_map {
        if name == "__metadata__" {
            return Err(Error::message(
                "safetensors: __metadata__ is a reserved tensor name",
            ));
        }
        if cancelled() {
            return Err(Error::Cancelled);
        }
        let shard = value.as_str().ok_or_else(|| {
            Error::message(format!(
                "safetensors: index weight_map value for {name:?} must be a string"
            ))
        })?;
        sorted.insert(name, shard.to_string());
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut shards: Vec<PathBuf> = Vec::new();
    let mut shard_names: HashMap<String, usize> = HashMap::new();
    let mut name_to_shard = HashMap::with_capacity(sorted.len());
    let mut names_by_shard: Vec<HashSet<String>> = Vec::new();
    for (name, shard_name) in sorted {
        let shard_index = match shard_names.get(&shard_name) {
            Some(&index) => index,
            None => {
                let resolved = resolve_shard(parent, &shard_name)?;
                let index = shards.len();
                shards.push(resolved);
                shard_names.insert(shard_name, index);
                names_by_shard.push(HashSet::new());
                index
            }
        };
        names_by_shard[shard_index].insert(name.clone());
        name_to_shard.insert(name, shard_index);
    }
    Ok(Index {
        shards,
        name_to_shard,
        names_by_shard,
    })
}

fn resolve_shard(parent: &Path, name: &str) -> Result<PathBuf, Error> {
    if name.is_empty() {
        return Err(Error::message(
            "safetensors: index shard name must not be empty",
        ));
    }
    let candidate = Path::new(name);
    if candidate.is_absolute() {
        return Err(Error::message(format!(
            "safetensors: index shard {name:?} must be relative to the index"
        )));
    }
    let mut resolved = parent.to_path_buf();
    for component in candidate.components() {
        match component {
            Component::Normal(part) => resolved.push(part),
            Component::CurDir => {}
            Component::ParentDir => {
                return Err(Error::message(format!(
                    "safetensors: index shard {name:?} escapes the index directory"
                )))
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(Error::message(format!(
                    "safetensors: index shard {name:?} must be relative to the index"
                )))
            }
        }
    }
    if resolved == parent {
        return Err(Error::message(format!(
            "safetensors: index shard {name:?} does not name a file"
        )));
    }
    Ok(resolved)
}

struct IndexDocument {
    weight_map: Option<serde_json::Map<String, serde_json::Value>>,
}

impl<'de> Deserialize<'de> for IndexDocument {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct IndexVisitor;
        impl<'de> Visitor<'de> for IndexVisitor {
            type Value = IndexDocument;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a Hugging Face safetensors index object")
            }

            fn visit_map<A>(self, mut access: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut weight_map = None;
                let mut seen: HashSet<String> = HashSet::new();
                while let Some(key) = access.next_key::<String>()? {
                    if !seen.insert(key.clone()) {
                        return Err(de::Error::custom(format!(
                            "safetensors: duplicate index key {key:?}"
                        )));
                    }
                    match key.as_str() {
                        "weight_map" => {
                            weight_map = Some(access.next_value::<UniqueObject>()?.0);
                        }
                        "metadata" => {
                            let value = access.next_value::<UniqueValue>()?.0;
                            if !value.is_object() {
                                return Err(de::Error::custom(
                                    "safetensors: index metadata must be an object",
                                ));
                            }
                        }
                        _ => {
                            let _ = access.next_value::<IgnoredAny>()?;
                        }
                    }
                }
                Ok(IndexDocument { weight_map })
            }
        }
        deserializer.deserialize_map(IndexVisitor)
    }
}

fn parse_index_document(bytes: &[u8]) -> Result<IndexDocument, Error> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let document = IndexDocument::deserialize(&mut deserializer)
        .map_err(|error| Error::message(format!("safetensors: invalid index document: {error}")))?;
    deserializer.end().map_err(|error| {
        Error::message(format!(
            "safetensors: trailing data in index document: {error}"
        ))
    })?;
    Ok(document)
}

/// JSON object parsed with duplicate-key rejection. Safetensors headers and
/// Hugging Face weight maps are maps whose duplicate keys collapse silently
/// under the default serde behavior, which would hide malformed archives.
struct UniqueObject(serde_json::Map<String, serde_json::Value>);

impl<'de> Deserialize<'de> for UniqueObject {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct UniqueVisitor;
        impl<'de> Visitor<'de> for UniqueVisitor {
            type Value = serde_json::Map<String, serde_json::Value>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON object with unique keys")
            }

            fn visit_map<A>(self, mut access: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut object = serde_json::Map::new();
                while let Some(key) = access.next_key::<String>()? {
                    if object.contains_key(&key) {
                        return Err(de::Error::custom(format!(
                            "safetensors: duplicate JSON key {key:?}"
                        )));
                    }
                    let value = access.next_value::<UniqueValue>()?.0;
                    object.insert(key, value);
                }
                Ok(object)
            }
        }
        deserializer
            .deserialize_map(UniqueVisitor)
            .map(UniqueObject)
    }
}

// Recursively apply duplicate-key rejection to descriptors and metadata too.
struct UniqueValue(serde_json::Value);

impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ValueVisitor;
        impl<'de> Visitor<'de> for ValueVisitor {
            type Value = serde_json::Value;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON value with unique object keys")
            }

            fn visit_map<A: MapAccess<'de>>(self, access: A) -> Result<Self::Value, A::Error> {
                UniqueObject::deserialize(de::value::MapAccessDeserializer::new(access))
                    .map(|object| serde_json::Value::Object(object.0))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = access.next_element::<UniqueValue>()? {
                    values.push(value.0);
                }
                Ok(serde_json::Value::Array(values))
            }

            fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(value.into())
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(value.into())
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(value.into())
            }

            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(value)
                    .map(serde_json::Value::Number)
                    .ok_or_else(|| E::custom("non-finite JSON number"))
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(value.into())
            }

            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(serde_json::Value::Null)
            }
        }
        deserializer.deserialize_any(ValueVisitor).map(UniqueValue)
    }
}

fn parse_unique_object(bytes: &[u8]) -> Result<serde_json::Map<String, serde_json::Value>, Error> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let object = UniqueObject::deserialize(&mut deserializer)
        .map_err(|error| Error::message(format!("safetensors: invalid header JSON: {error}")))?;
    deserializer.end().map_err(|error| {
        Error::message(format!(
            "safetensors: trailing data in header JSON: {error}"
        ))
    })?;
    Ok(object.0)
}

#[cfg(test)]
thread_local! {
    static READ_BYTES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[inline]
fn record_read(_bytes: usize) {
    #[cfg(test)]
    READ_BYTES.with(|count| count.set(count.get() + _bytes as u64));
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize, TensorView};
    use std::io::Write;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    fn never() -> bool {
        false
    }

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "effect-torch-napi-safetensors-{}-{name}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_archive(
        path: &Path,
        tensors: Vec<(&str, Dtype, Vec<usize>, Vec<u8>)>,
        metadata: Option<HashMap<String, String>>,
    ) {
        let owned = tensors
            .into_iter()
            .map(|(name, dtype, shape, bytes)| (name.to_string(), dtype, shape, bytes))
            .collect::<Vec<_>>();
        let views = owned
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                (
                    name.clone(),
                    TensorView::new(*dtype, shape.clone(), bytes).expect("valid test tensor"),
                )
            })
            .collect::<Vec<_>>();
        let encoded = serialize(views, metadata).expect("test archive serializes");
        std::fs::write(path, encoded).unwrap();
    }

    fn write_raw(path: &Path, header: &str, data_len: u64) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        std::fs::write(path, &bytes).unwrap();
        let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        file.set_len(bytes.len() as u64 + data_len).unwrap();
    }

    fn write_index(path: &Path, body: &str) {
        std::fs::write(path, body.as_bytes()).unwrap();
    }

    fn shard_file(
        dir: &Path,
        name: &str,
        tensors: Vec<(&str, Dtype, Vec<usize>, Vec<u8>)>,
    ) -> PathBuf {
        let path = dir.join(name);
        write_archive(&path, tensors, None);
        path
    }

    #[test]
    fn inspection_reports_sorted_geometry_and_metadata() {
        let dir = test_dir("inspect");
        let path = dir.join("model.safetensors");
        write_archive(
            &path,
            vec![
                ("b", Dtype::BF16, vec![2], vec![0x80, 0x3f, 0x00, 0xc0]),
                ("a", Dtype::F32, vec![1, 2], vec![0; 8]),
            ],
            Some(HashMap::from([(
                "framework".to_string(),
                "effect-torch".to_string(),
            )])),
        );
        let inspection = inspect(path.to_str().unwrap(), &never).unwrap();
        assert_eq!(
            inspection.metadata.get("framework").unwrap(),
            "effect-torch"
        );
        assert_eq!(inspection.entries.len(), 2);
        assert_eq!(inspection.entries[0].name, "a");
        assert_eq!(inspection.entries[0].dtype, Dtype::F32);
        assert_eq!(inspection.entries[0].shape, vec![1, 2]);
        assert_eq!(inspection.entries[0].byte_length, 8);
        assert_eq!(inspection.entries[1].name, "b");
        assert_eq!(inspection.entries[1].dtype, Dtype::BF16);
        assert_eq!(inspection.entries[1].byte_length, 4);
        assert_eq!(dtype_name(inspection.entries[1].dtype), "bf16");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn selective_load_reads_only_chosen_payloads() {
        let dir = test_dir("selective");
        let path = dir.join("model.safetensors");
        write_archive(
            &path,
            vec![
                ("a", Dtype::BF16, vec![1], vec![0x80, 0x3f]),
                ("b", Dtype::U8, vec![3], vec![1, 2, 3]),
            ],
            None,
        );
        let selected = vec!["a".to_string()];
        let load_plan = plan(path.to_str().unwrap(), Some(&selected), &never).unwrap();
        assert_eq!(load_plan.entries().len(), 1);
        let loaded = load_plan.load(&never, |_meta, bytes| Ok(bytes)).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "a");
        assert_eq!(loaded[0].1, vec![0x80, 0x3f]);

        let none = plan(path.to_str().unwrap(), Some(&[]), &never).unwrap();
        assert!(none.is_empty());
        assert_eq!(
            none.load(&never, |_meta, bytes| Ok(bytes)).unwrap().len(),
            0
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn selection_rejects_missing_duplicate_and_reserved_names() {
        let dir = test_dir("selection");
        let path = dir.join("model.safetensors");
        write_archive(&path, vec![("a", Dtype::U8, vec![1], vec![1])], None);
        let missing = vec!["missing".to_string()];
        assert!(plan(path.to_str().unwrap(), Some(&missing), &never).is_err());
        let duplicate = vec!["a".to_string(), "a".to_string()];
        assert!(plan(path.to_str().unwrap(), Some(&duplicate), &never).is_err());
        let reserved = vec!["__metadata__".to_string()];
        assert!(plan(path.to_str().unwrap(), Some(&reserved), &never).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_truncated_files() {
        let dir = test_dir("truncated");
        let short = dir.join("short.safetensors");
        std::fs::write(&short, b"abc").unwrap();
        assert!(inspect(short.to_str().unwrap(), &never).is_err());

        let over = dir.join("over.safetensors");
        let mut bytes = 1000u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{\"a\":{}}");
        std::fs::write(&over, &bytes).unwrap();
        assert!(inspect(over.to_str().unwrap(), &never).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_malformed_header_geometry() {
        let dir = test_dir("geometry");
        let cases = [
            (
                "invalid-json.safetensors",
                "not json",
                0u64,
            ),
            (
                "duplicate-keys.safetensors",
                "{\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}",
                1,
            ),
            (
                "unknown-dtype.safetensors",
                "{\"a\":{\"dtype\":\"NOPE\",\"shape\":[1],\"data_offsets\":[0,4]}}",
                4,
            ),
            (
                "size-mismatch.safetensors",
                "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}",
                8,
            ),
            (
                "out-of-bounds.safetensors",
                "{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}}",
                2,
            ),
            (
                "element-overflow.safetensors",
                "{\"a\":{\"dtype\":\"U8\",\"shape\":[4294967295,4294967295,4294967295],\"data_offsets\":[0,0]}}",
                0,
            ),
            (
                "dimension-overflow.safetensors",
                "{\"a\":{\"dtype\":\"U8\",\"shape\":[4294967296],\"data_offsets\":[0,0]}}",
                0,
            ),
            (
                "misaligned.safetensors",
                "{\"a\":{\"dtype\":\"F4\",\"shape\":[1],\"data_offsets\":[0,0]}}",
                0,
            ),
            (
                "reversed-offsets.safetensors",
                "{\"a\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[2,1]}}",
                2,
            ),
            (
                "overlap.safetensors",
                "{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]},\"b\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[2,6]}}",
                6,
            ),
            (
                "bad-metadata.safetensors",
                "{\"__metadata__\":5,\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}",
                1,
            ),
        ];
        for (name, header, data_len) in cases {
            let path = dir.join(name);
            write_raw(&path, header, data_len);
            assert!(
                inspect(path.to_str().unwrap(), &never).is_err(),
                "expected {name} to be rejected"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sparse_archive_reads_only_selected_payload() {
        let dir = test_dir("sparse");
        let path = dir.join("sparse.safetensors");
        let header = r#"{"small":{"dtype":"BF16","shape":[1],"data_offsets":[4294967296,4294967298]},"huge":{"dtype":"U8","shape":[2,2147483648],"data_offsets":[0,4294967296]}}"#;
        let huge_len: u64 = 4_294_967_296;
        write_raw(&path, header, huge_len + 2);
        let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(8 + header.len() as u64 + huge_len))
            .unwrap();
        file.write_all(&[0x81, 0x7f]).unwrap();
        drop(file);

        READ_BYTES.set(0);
        let inspection = inspect(path.to_str().unwrap(), &never).unwrap();
        assert_eq!(READ_BYTES.get(), 8 + header.len() as u64);
        assert_eq!(inspection.entries.len(), 2);
        let huge = inspection
            .entries
            .iter()
            .find(|entry| entry.name == "huge")
            .unwrap();
        assert_eq!(huge.shape, vec![2, 2_147_483_648]);
        assert_eq!(huge.byte_length, huge_len);

        READ_BYTES.set(0);
        let selected = vec!["small".to_string()];
        let plan = plan(path.to_str().unwrap(), Some(&selected), &never).unwrap();
        let loaded = plan.load(&never, |_meta, bytes| Ok(bytes)).unwrap();
        assert_eq!(loaded[0].1, vec![0x81, 0x7f]);
        assert_eq!(READ_BYTES.get(), 8 + header.len() as u64 + 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn index_selective_load_skips_missing_unselected_shard() {
        let dir = test_dir("index-missing");
        shard_file(
            &dir,
            "model-00001-of-00002.safetensors",
            vec![("a", Dtype::BF16, vec![1], vec![0x80, 0x3f])],
        );
        let index = dir.join("model.safetensors.index.json");
        write_index(
            &index,
            "{\"metadata\":{\"total_size\":2},\"weight_map\":{\"a\":\"model-00001-of-00002.safetensors\",\"b\":\"model-00002-of-00002.safetensors\"}}",
        );
        assert!(inspect(index.to_str().unwrap(), &never).is_err());
        assert!(plan(index.to_str().unwrap(), None, &never).is_err());

        let selected = vec!["a".to_string()];
        let load_plan = plan(index.to_str().unwrap(), Some(&selected), &never).unwrap();
        assert!(load_plan.metadata().is_empty());
        let loaded = load_plan.load(&never, |_meta, bytes| Ok(bytes)).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "a");
        assert_eq!(loaded[0].1, vec![0x80, 0x3f]);

        let missing_shard = vec!["b".to_string()];
        assert!(plan(index.to_str().unwrap(), Some(&missing_shard), &never).is_err());
        assert!(plan(index.to_str().unwrap(), Some(&[]), &never)
            .unwrap()
            .is_empty());
        assert!(plan(
            index.to_str().unwrap(),
            Some(&["missing".to_string()]),
            &never
        )
        .is_err());
        assert!(plan(
            index.to_str().unwrap(),
            Some(&["a".to_string(), "a".to_string()]),
            &never
        )
        .is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn index_inspection_merges_metadata_and_rejects_conflicts() {
        let dir = test_dir("index-metadata");
        write_archive(
            &dir.join("shard-a.safetensors"),
            vec![("a", Dtype::U8, vec![1], vec![1])],
            Some(HashMap::from([
                ("shared".to_string(), "x".to_string()),
                ("a".to_string(), "1".to_string()),
            ])),
        );
        write_archive(
            &dir.join("shard-b.safetensors"),
            vec![("b", Dtype::U8, vec![1], vec![2])],
            Some(HashMap::from([
                ("shared".to_string(), "x".to_string()),
                ("b".to_string(), "2".to_string()),
            ])),
        );
        let index = dir.join("model.safetensors.index.json");
        write_index(
            &index,
            "{\"metadata\":{\"total_size\":2,\"index_only\":\"bookkeeping\"},\"weight_map\":{\"a\":\"shard-a.safetensors\",\"b\":\"shard-b.safetensors\"}}",
        );
        let inspection = inspect(index.to_str().unwrap(), &never).unwrap();
        assert_eq!(inspection.entries.len(), 2);
        assert_eq!(inspection.metadata.len(), 3);
        assert_eq!(inspection.metadata.get("shared").unwrap(), "x");
        let load_plan = plan(index.to_str().unwrap(), None, &never).unwrap();
        assert_eq!(load_plan.metadata(), &inspection.metadata);
        let selected = plan(index.to_str().unwrap(), Some(&["a".to_string()]), &never).unwrap();
        assert_eq!(
            selected.metadata(),
            &HashMap::from([
                ("shared".to_string(), "x".to_string()),
                ("a".to_string(), "1".to_string()),
            ])
        );

        write_archive(
            &dir.join("shard-b.safetensors"),
            vec![("b", Dtype::U8, vec![1], vec![2])],
            Some(HashMap::from([("shared".to_string(), "y".to_string())])),
        );
        assert!(inspect(index.to_str().unwrap(), &never).is_err());
        assert!(plan(index.to_str().unwrap(), None, &never).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn index_rejects_wrong_keys_paths_and_absent_mappings() {
        let dir = test_dir("index-invalid");
        write_index(
            &dir.join("no-weight-map.safetensors.index.json"),
            "{\"metadata\":{}}",
        );
        assert!(inspect(
            dir.join("no-weight-map.safetensors.index.json")
                .to_str()
                .unwrap(),
            &never
        )
        .is_err());

        write_index(
            &dir.join("non-object.safetensors.index.json"),
            "{\"weight_map\":[1]}",
        );
        assert!(inspect(
            dir.join("non-object.safetensors.index.json")
                .to_str()
                .unwrap(),
            &never
        )
        .is_err());

        write_index(
            &dir.join("non-string.safetensors.index.json"),
            "{\"weight_map\":{\"a\":5}}",
        );
        assert!(inspect(
            dir.join("non-string.safetensors.index.json")
                .to_str()
                .unwrap(),
            &never
        )
        .is_err());

        write_index(
            &dir.join("duplicate.safetensors.index.json"),
            "{\"weight_map\":{\"a\":\"x.safetensors\",\"a\":\"y.safetensors\"}}",
        );
        assert!(inspect(
            dir.join("duplicate.safetensors.index.json")
                .to_str()
                .unwrap(),
            &never
        )
        .is_err());

        write_index(
            &dir.join("bad-metadata.safetensors.index.json"),
            "{\"weight_map\":{},\"metadata\":5}",
        );
        assert!(inspect(
            dir.join("bad-metadata.safetensors.index.json")
                .to_str()
                .unwrap(),
            &never
        )
        .is_err());

        write_index(
            &dir.join("escape.safetensors.index.json"),
            "{\"weight_map\":{\"a\":\"../escape.safetensors\"}}",
        );
        assert!(inspect(
            dir.join("escape.safetensors.index.json").to_str().unwrap(),
            &never
        )
        .is_err());

        shard_file(
            &dir,
            "present.safetensors",
            vec![("a", Dtype::U8, vec![1], vec![1])],
        );
        write_index(
            &dir.join("absent.safetensors.index.json"),
            "{\"weight_map\":{\"x\":\"present.safetensors\"}}",
        );
        assert!(inspect(
            dir.join("absent.safetensors.index.json").to_str().unwrap(),
            &never
        )
        .is_err());
        let selected = vec!["x".to_string()];
        assert!(plan(
            dir.join("absent.safetensors.index.json").to_str().unwrap(),
            Some(&selected),
            &never
        )
        .is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    struct Tracked(Arc<AtomicUsize>);

    impl Drop for Tracked {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn cancellation_between_tensors_releases_partial_values() {
        let dir = test_dir("cancel");
        let path = dir.join("cancel.safetensors");
        write_archive(
            &path,
            vec![
                ("a", Dtype::U8, vec![1], vec![7]),
                ("b", Dtype::U8, vec![1], vec![8]),
            ],
            None,
        );
        let plan = plan(path.to_str().unwrap(), None, &never).unwrap();
        let flag = Arc::new(AtomicBool::new(false));
        let callbacks = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let result = plan.load(&|| flag.load(Ordering::SeqCst), {
            let flag = Arc::clone(&flag);
            let callbacks = Arc::clone(&callbacks);
            let drops = Arc::clone(&drops);
            move |_meta, _bytes| {
                callbacks.fetch_add(1, Ordering::SeqCst);
                flag.store(true, Ordering::SeqCst);
                Ok(Tracked(Arc::clone(&drops)))
            }
        });
        assert!(matches!(result, Err(Error::Cancelled)));
        assert_eq!(callbacks.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn byte_length_rejects_unsafe_integers() {
        assert!(byte_length_f64(MAX_SAFE_INTEGER).is_ok());
        assert!(byte_length_f64(MAX_SAFE_INTEGER + 1).is_err());
    }

    #[test]
    fn rejects_nested_duplicates_gaps_and_byte_length_overflow() {
        let dir = test_dir("nested-invalid");
        for (index, header, length) in [
            (
                0,
                r#"{"a":{"dtype":"U8","dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
                1,
            ),
            (1, r#"{"__metadata__":{"a":"x","a":"y"}}"#, 0),
            (
                2,
                r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#,
                2,
            ),
            (
                3,
                r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
                2,
            ),
            (
                4,
                r#"{"a":{"dtype":"F64","shape":[4294967295,4294967295],"data_offsets":[0,0]}}"#,
                0,
            ),
            (
                5,
                r#"{"a":{"dtype":"U8","shape":[1.5],"data_offsets":[0,1]}}"#,
                1,
            ),
        ] {
            let path = dir.join(format!("{index}.safetensors"));
            write_raw(&path, header, length);
            assert!(
                inspect(path.to_str().unwrap(), &never).is_err(),
                "case {index}"
            );
            assert!(
                plan(path.to_str().unwrap(), Some(&[]), &never).is_err(),
                "case {index}"
            );
        }
        let path = dir.join("huge-header.safetensors");
        std::fs::write(&path, (MAX_HEADER_BYTES + 1).to_le_bytes()).unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(MAX_HEADER_BYTES + 9)
            .unwrap();
        READ_BYTES.set(0);
        assert!(inspect(path.to_str().unwrap(), &never).is_err());
        assert_eq!(READ_BYTES.get(), 8);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn cancellation_interrupts_payload_and_header_chunks() {
        let dir = test_dir("chunk-cancellation");
        let path = dir.join("model.safetensors");
        let header = r#"{"a":{"dtype":"U8","shape":[3145728],"data_offsets":[0,3145728]}}"#;
        write_raw(&path, header, (READ_CHUNK_BYTES * 3) as u64);
        let plan = plan(path.to_str().unwrap(), None, &never).unwrap();
        READ_BYTES.set(0);
        let result = plan.load::<(), _>(&|| READ_BYTES.get() >= READ_CHUNK_BYTES as u64, |_, _| {
            panic!("cancelled payload must not allocate a tensor");
        });
        assert!(matches!(result, Err(Error::Cancelled)));
        assert_eq!(READ_BYTES.get(), READ_CHUNK_BYTES as u64);

        let header = format!("{{}}{}", " ".repeat(READ_CHUNK_BYTES * 3));
        write_raw(&path, &header, 0);
        READ_BYTES.set(0);
        assert!(matches!(
            inspect(path.to_str().unwrap(), &|| READ_BYTES.get()
                > READ_CHUNK_BYTES as u64),
            Err(Error::Cancelled)
        ));
        assert_eq!(READ_BYTES.get(), 8 + READ_CHUNK_BYTES as u64);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn later_read_failure_releases_partial_values() {
        let dir = test_dir("partial-read-failure");
        let path = dir.join("model.safetensors");
        write_archive(
            &path,
            vec![
                ("a", Dtype::U8, vec![1], vec![1]),
                ("b", Dtype::U8, vec![1], vec![2]),
            ],
            None,
        );
        let plan = plan(path.to_str().unwrap(), None, &never).unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        let result = plan.load(&never, |_, _| {
            // Simulate a file truncated after its header was validated.
            std::fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .unwrap()
                .set_len(0)
                .unwrap();
            Ok(Tracked(drops.clone()))
        });
        assert!(matches!(result, Err(Error::Message(_))));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn callback_failure_and_last_tensor_cancellation_release_values() {
        let dir = test_dir("callback-failure");
        let path = dir.join("model.safetensors");
        write_archive(
            &path,
            vec![
                ("a", Dtype::U8, vec![1], vec![1]),
                ("b", Dtype::U8, vec![1], vec![2]),
            ],
            None,
        );
        let drops = Arc::new(AtomicUsize::new(0));
        let plan = plan(path.to_str().unwrap(), None, &never).unwrap();
        let result = plan.load(&never, |meta, _| {
            if meta.name == "b" {
                return Err(Error::message("upload failed"));
            }
            Ok(Tracked(drops.clone()))
        });
        assert!(result.is_err());
        assert_eq!(drops.load(Ordering::SeqCst), 1);

        let plan = super::plan(path.to_str().unwrap(), Some(&["a".to_string()]), &never).unwrap();
        let flag = AtomicBool::new(false);
        let result = plan.load(&|| flag.load(Ordering::SeqCst), |_, _| {
            flag.store(true, Ordering::SeqCst);
            Ok(Tracked(drops.clone()))
        });
        assert!(matches!(result, Err(Error::Cancelled)));
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn selected_shards_validate_every_header_mapping() {
        let dir = test_dir("shard-mappings");
        shard_file(
            &dir,
            "a.safetensors",
            vec![("a", Dtype::U8, vec![1], vec![1])],
        );
        shard_file(
            &dir,
            "b.safetensors",
            vec![
                ("a", Dtype::U8, vec![1], vec![1]),
                ("b", Dtype::U8, vec![1], vec![2]),
            ],
        );
        let path = dir.join("model.safetensors.index.json");
        write_index(
            &path,
            r#"{"weight_map":{"a":"a.safetensors","b":"b.safetensors"}}"#,
        );
        assert!(inspect(path.to_str().unwrap(), &never).is_err());
        assert!(plan(path.to_str().unwrap(), Some(&["b".to_string()]), &never).is_err());
        assert!(plan(path.to_str().unwrap(), Some(&["a".to_string()]), &never).is_ok());
        for shard in [
            "../outside.safetensors",
            "/absolute.safetensors",
            "nested/../outside.safetensors",
        ] {
            write_index(
                &path,
                &serde_json::json!({"weight_map": {"a": shard}}).to_string(),
            );
            assert!(plan(path.to_str().unwrap(), Some(&[]), &never).is_err());
        }
        write_index(&path, r#"{"weight_map":{"__metadata__":"a.safetensors"}}"#);
        assert!(plan(path.to_str().unwrap(), None, &never).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn cancelled_worker_drops_late_loaded_result() {
        let dir = test_dir("late-result");
        let path = dir.join("model.safetensors");
        write_archive(&path, vec![("a", Dtype::U8, vec![1], vec![1])], None);
        let plan = plan(path.to_str().unwrap(), None, &never).unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        let worker_drops = drops.clone();
        let state = Arc::new(crate::CancellationState::new());
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let result = runtime.block_on(crate::run_compute(state, None, move |_, state| {
            let loaded = plan
                .load(&never, |_, _| Ok(Tracked(worker_drops.clone())))
                .map_err(Error::into_napi)?;
            // Cancellation races publication after loading the last value.
            state.cancel();
            Ok(loaded)
        }));
        assert!(matches!(result, Err(error) if error.status == napi::Status::Cancelled));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn index_follows_relative_symlink_shards() {
        use std::os::unix::fs::symlink;

        let dir = test_dir("index-symlink");
        let blobs = dir.join("blobs");
        let snapshot = dir.join("snapshots").join("revision");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&snapshot).unwrap();
        let blob = blobs.join("shard-blob.safetensors");
        write_archive(&blob, vec![("a", Dtype::U8, vec![1], vec![9])], None);
        // The index-relative shard name is a plain relative file name, but its
        // symlink target lives outside the index directory. Hugging Face cache
        // snapshots use exactly this layout and must keep working.
        symlink(
            "../../blobs/shard-blob.safetensors",
            snapshot.join("model-00001-of-00001.safetensors"),
        )
        .unwrap();
        let index_blob = blobs.join("index-blob");
        write_index(
            &index_blob,
            "{\"weight_map\":{\"a\":\"model-00001-of-00001.safetensors\"}}",
        );
        let index = snapshot.join("model.safetensors.index.json");
        symlink("../../blobs/index-blob", &index).unwrap();
        let inspection = inspect(index.to_str().unwrap(), &never).unwrap();
        assert_eq!(inspection.entries.len(), 1);
        let selected = vec!["a".to_string()];
        let load_plan = plan(index.to_str().unwrap(), Some(&selected), &never).unwrap();
        let loaded = load_plan.load(&never, |_meta, bytes| Ok(bytes)).unwrap();
        assert_eq!(loaded[0].1, vec![9]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

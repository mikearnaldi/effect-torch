use crate::device::CUDA_TOP_K_LIMIT;
use crate::{
    compile_stateful_with_options, compile_with_options, CudaDevice, CudaExecutable,
    CudaSequenceState, CudaStateInvocation, CudaValue,
};
use cudarc::driver::CudaSlice;
use effect_torch_compiler::{
    specialize_decode_layout_outputs_with_attention, CompileOptions, CurrentBlockAttention,
    DecodeGeometry, DecodeLayout, DecodeOutputSelection, InferenceOptions,
};
use effect_torch_graph::{
    AttentionWindow, CrossEntropyReduction, Device, LeafSlot, Node, NodeKind, PositionOffset,
    RotaryLayout,
};
use effect_torch_napi::{run_compute, CancellationState};
use effect_torch_runtime::{
    effective_probabilities, purpose_counter, random_unit, sample_logits, sample_probabilities,
    DType, GgmlKQuant, SamplingOptions, SamplingPurpose,
};
use napi::bindgen_prelude::{Buffer, Uint8Array};
use napi::{Error, Result, Status};
use napi_derive::napi;
use serde_json::Value as JsonValue;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[path = "napi/gguf.rs"]
mod gguf;
#[path = "napi/safetensors.rs"]
mod safetensors_io;
#[allow(unused_imports)]
pub use gguf::{inspect_gguf, load_gguf_for_device};

fn invalid(message: impl Into<String>) -> Error {
    Error::new(Status::InvalidArg, message.into())
}

fn failure(message: impl Into<String>) -> Error {
    Error::new(Status::GenericFailure, message.into())
}

fn parse_dtype(dtype: &str) -> Result<DType> {
    match dtype {
        "f64" => Ok(DType::F64),
        "f32" => Ok(DType::F32),
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        "i64" => Ok(DType::I64),
        "u32" => Ok(DType::U32),
        "u8" => Ok(DType::U8),
        _ => Err(invalid(format!("unsupported CUDA dtype {dtype}"))),
    }
}

fn parse_kquant(encoding: &str) -> Result<GgmlKQuant> {
    match encoding {
        "Q2_K" => Ok(GgmlKQuant::Q2K),
        "Q3_K" => Ok(GgmlKQuant::Q3K),
        "Q4_K" => Ok(GgmlKQuant::Q4K),
        "Q5_K" => Ok(GgmlKQuant::Q5K),
        "Q6_K" => Ok(GgmlKQuant::Q6K),
        _ => Err(invalid(format!("unsupported CUDA encoding {encoding}"))),
    }
}

fn encode(values: Vec<f64>, dtype: DType) -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in values {
        match dtype {
            DType::F64 => bytes.extend_from_slice(&value.to_le_bytes()),
            DType::F32 | DType::F16 | DType::BF16 => {
                bytes.extend_from_slice(&(value as f32).to_le_bytes());
            }
            DType::I64 => bytes.extend_from_slice(&(value as i64).to_le_bytes()),
            DType::U32 => bytes.extend_from_slice(&(value as u32).to_le_bytes()),
            DType::U8 => bytes.push(value as u8),
        }
    }
    bytes
}

fn shape(shape: Vec<u32>) -> Vec<usize> {
    shape
        .into_iter()
        .map(|dimension| dimension as usize)
        .collect()
}

fn non_negative_safe_integer(value: f64, name: &str) -> Result<u64> {
    const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > MAX_SAFE_INTEGER {
        return Err(invalid(format!(
            "sample: {name} must be a non-negative safe integer, got {value}"
        )));
    }
    Ok(value as u64)
}

fn sampling_options(
    temperature: f64,
    top_k: f64,
    top_p: f64,
    seed: f64,
    counter: f64,
) -> Result<SamplingOptions> {
    let top_k = non_negative_safe_integer(top_k, "topK")?;
    Ok(SamplingOptions {
        temperature,
        top_k: if top_k == 0 {
            None
        } else {
            Some(usize::try_from(top_k).map_err(|_| invalid("sample: topK is out of range"))?)
        },
        top_p,
        seed: non_negative_safe_integer(seed, "seed")?,
        counter: non_negative_safe_integer(counter, "counter")?,
    })
}

fn sample_cuda_value(
    value: &CudaValue,
    options: SamplingOptions,
    mut cancelled: impl FnMut() -> bool,
) -> std::result::Result<u32, String> {
    if value.dtype() == DType::F32
        && options.temperature == 0.0
        && options.top_k.is_none()
        && options.top_p == 1.0
    {
        if cancelled() {
            return Err("operation aborted".to_string());
        }
        let token = value.greedy_argmax()?;
        if cancelled() {
            return Err("operation aborted".to_string());
        }
        return Ok(token);
    }
    if value.dtype() == DType::F32
        && options.temperature.is_finite()
        && options.temperature > 0.0
        && options.top_p.is_finite()
        && options.top_p > 0.0
        && options.top_p <= 1.0
        && matches!(options.top_k, Some(1..=CUDA_TOP_K_LIMIT))
    {
        if cancelled() {
            return Err("operation aborted".to_string());
        }
        let top_k = options.top_k.expect("CUDA top-k path requires topK");
        let (values, tokens) = value.topk(top_k)?;
        if cancelled() {
            return Err("operation aborted".to_string());
        }
        let selected = sample_logits(values.len(), |index| values[index], options, cancelled)?;
        return Ok(tokens[selected as usize]);
    }
    let values = value.readback()?;
    sample_logits(values.len(), |index| values[index], options, cancelled)
}

#[napi(object)]
pub struct NativePackedCausalChainsLayout {
    pub rows_per_sequence: u32,
}

#[napi(string_enum)]
#[derive(Clone, Copy)]
pub enum NativeCurrentBlockAttention {
    Causal,
    Bidirectional,
}

impl From<NativeCurrentBlockAttention> for CurrentBlockAttention {
    fn from(value: NativeCurrentBlockAttention) -> Self {
        match value {
            NativeCurrentBlockAttention::Causal => Self::Causal,
            NativeCurrentBlockAttention::Bidirectional => Self::Bidirectional,
        }
    }
}

#[napi(string_enum)]
#[derive(Clone, Copy)]
pub enum NativeDecodeOutputSelection {
    AllRows,
    SplitLastTokenRow,
    BatchedLastTokenRow,
}

impl From<NativeDecodeOutputSelection> for DecodeOutputSelection {
    fn from(value: NativeDecodeOutputSelection) -> Self {
        match value {
            NativeDecodeOutputSelection::AllRows => Self::AllRows,
            NativeDecodeOutputSelection::SplitLastTokenRow => Self::SplitLastTokenRow,
            NativeDecodeOutputSelection::BatchedLastTokenRow => Self::BatchedLastTokenRow,
        }
    }
}

#[napi(object)]
pub struct NativeKvStateSchema {
    pub max_tokens: u32,
    pub block_size: u32,
    pub kv_dtype: String,
    pub window: Option<u32>,
    pub current_block_attention: Option<NativeCurrentBlockAttention>,
    pub batch: u32,
    pub packed_causal_chains: Option<NativePackedCausalChainsLayout>,
    pub last_token_row: Option<bool>,
    pub output_selections: Option<Vec<NativeDecodeOutputSelection>>,
}

#[napi(object)]
pub struct NativeRecurrentStateSchema {
    pub kda_layers: u32,
    pub kda_heads: u32,
    pub kda_head_dim: u32,
    pub kda_value_dim: u32,
    pub conv_layers: u32,
    pub conv_channels: u32,
    pub conv_kernel: u32,
}

#[napi(object)]
pub struct NativeCompileOptions {
    pub optimize: Option<bool>,
    pub constant_weights: Option<bool>,
}

#[napi(object)]
pub struct NativeInstructionDiagnostics {
    pub kind: String,
    pub count: f64,
}

#[napi(object)]
pub struct NativeCompilePhaseDiagnostics {
    pub phase: String,
    pub nanoseconds: f64,
}

#[napi(object)]
pub struct NativeMemoryDiagnostics {
    pub external_bytes: f64,
    pub persistent_bytes: f64,
    pub state_bytes: f64,
    pub output_bytes: f64,
    pub workspace_bytes: f64,
    pub transaction_bytes: f64,
    pub peak_live_bytes: f64,
    pub packing_overhead_bytes: f64,
}

#[napi(object)]
pub struct NativeExecutableDiagnostics {
    pub semantic_nodes_before_optimization: f64,
    pub semantic_nodes_after_optimization: f64,
    pub instructions: Vec<NativeInstructionDiagnostics>,
    pub pipeline_count: f64,
    pub command_count: f64,
    pub synchronization_count: f64,
    pub memory: NativeMemoryDiagnostics,
    pub compile_phases: Vec<NativeCompilePhaseDiagnostics>,
}

fn executable_diagnostics(
    diagnostics: &effect_torch_runtime::ExecutableDiagnostics,
) -> NativeExecutableDiagnostics {
    let memory = &diagnostics.memory;
    NativeExecutableDiagnostics {
        semantic_nodes_before_optimization: diagnostics.semantic_nodes_before_optimization as f64,
        semantic_nodes_after_optimization: diagnostics.semantic_nodes_after_optimization as f64,
        instructions: diagnostics
            .instructions
            .iter()
            .map(|instruction| NativeInstructionDiagnostics {
                kind: instruction.kind.clone(),
                count: instruction.count as f64,
            })
            .collect(),
        pipeline_count: diagnostics.pipeline_count as f64,
        command_count: diagnostics.command_count as f64,
        synchronization_count: diagnostics.synchronization_count as f64,
        memory: NativeMemoryDiagnostics {
            external_bytes: memory.external_bytes as f64,
            persistent_bytes: memory.persistent_bytes as f64,
            state_bytes: memory.state_bytes as f64,
            output_bytes: memory.output_bytes as f64,
            workspace_bytes: memory.workspace_bytes as f64,
            transaction_bytes: memory.transaction_bytes as f64,
            peak_live_bytes: memory.peak_live_bytes as f64,
            packing_overhead_bytes: memory.packing_overhead_bytes as f64,
        },
        compile_phases: diagnostics
            .compile_phases
            .iter()
            .map(|timing| NativeCompilePhaseDiagnostics {
                phase: timing.phase.clone(),
                nanoseconds: timing.nanoseconds as f64,
            })
            .collect(),
    }
}

fn compile_options(explicit: Option<NativeCompileOptions>, stateful: bool) -> CompileOptions {
    let mut options = CompileOptions::from_environment();
    if let Some(explicit) = explicit {
        if let Some(optimize) = explicit.optimize {
            options.optimize = optimize;
        }
        if stateful || explicit.constant_weights.is_some() {
            options.inference = Some(InferenceOptions {
                constant_weights: explicit.constant_weights.unwrap_or(false),
            });
        }
    } else if stateful {
        options.inference = Some(InferenceOptions::default());
    }
    options
}

#[napi(object)]
pub struct NativeSamplingOptions {
    pub temperature: f64,
    pub top_k: f64,
    pub top_p: f64,
    pub seed: f64,
    pub counter: f64,
}

#[napi]
pub struct NativeTargetMatchingOutput {
    pages: Vec<Vec<u32>>,
    accepted: Vec<u32>,
    outputs: Vec<NativeTensor>,
}

#[napi]
impl NativeTargetMatchingOutput {
    #[napi(getter)]
    pub fn pages(&self) -> Vec<Vec<u32>> {
        self.pages.clone()
    }

    #[napi(getter)]
    pub fn accepted(&self) -> Vec<u32> {
        self.accepted.clone()
    }

    #[napi(getter)]
    pub fn outputs(&self) -> Vec<NativeTensor> {
        self.outputs.clone()
    }
}

#[derive(Clone)]
struct CudaStateSchema {
    max_tokens: u32,
    block_size: u32,
    kv_dtype: DType,
    window: Option<u32>,
    batch: u32,
    packed_rows_per_sequence: Option<u32>,
    geometry: DecodeGeometry,
}

struct PoolInner {
    ordinal: u32,
    layers: u32,
    kv_heads: u32,
    head_dim: u32,
    max_tokens: u32,
    block_size: u32,
    dtype: DType,
    recurrent: NativeRecurrentStateSchema,
    usage: Mutex<PoolUsage>,
}

#[derive(Clone, Default)]
struct PoolUsage {
    blocks: HashMap<BlockKey, u32>,
    snapshots: HashMap<BlockKey, Arc<SequenceState>>,
}

#[derive(Clone)]
struct BlockKey {
    hash: u64,
    value: Arc<str>,
}

impl BlockKey {
    fn new(value: String) -> Self {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        Self {
            hash: hasher.finish(),
            value: value.into(),
        }
    }
}

impl PartialEq for BlockKey {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
            && (Arc::ptr_eq(&self.value, &other.value) || self.value == other.value)
    }
}

impl Eq for BlockKey {}

impl Hash for BlockKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

#[derive(Clone, Default)]
struct SequenceState {
    cursor: u32,
    tokens: Vec<u32>,
    keys: Vec<Vec<f64>>,
    values: Vec<Vec<f64>>,
    kda_states: Vec<Vec<f64>>,
    conv_states: Vec<Vec<f64>>,
    block_keys: Vec<BlockKey>,
}

struct SequenceInner {
    pool: Arc<PoolInner>,
    state: Mutex<SequenceState>,
    device_cache: Mutex<Option<SequenceDeviceCache>>,
    released: AtomicBool,
    running: AtomicBool,
}

struct SequenceDeviceCache {
    keys: Arc<CudaSlice<f32>>,
    values: Arc<CudaSlice<f32>>,
    layer_size: usize,
    cursor: Option<Arc<CudaSlice<u32>>>,
    valid: Option<Arc<CudaSlice<u32>>>,
    decode_graph: Option<crate::executable::CudaDecodeGraph>,
}

struct SequenceLease {
    inner: Arc<SequenceInner>,
}

impl Drop for SequenceLease {
    fn drop(&mut self) {
        self.inner.running.store(false, Ordering::Release);
    }
}

#[napi]
pub struct NativeKvPool {
    inner: Arc<PoolInner>,
}

#[napi]
impl NativeKvPool {
    #[napi(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: u32,
        layers: u32,
        kv_heads: u32,
        head_dim: u32,
        max_tokens: u32,
        block_size: Option<u32>,
        dtype: Option<String>,
        recurrent: Option<NativeRecurrentStateSchema>,
    ) -> Result<Self> {
        CudaDevice::get(device).map_err(failure)?;
        let block_size = block_size.unwrap_or(16);
        let dtype = parse_dtype(dtype.as_deref().unwrap_or("f32"))?;
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16 | DType::U8) {
            return Err(invalid("kv pool: dtype must be f32, f16, bf16, or u8"));
        }
        if max_tokens == 0 || block_size == 0 || !max_tokens.is_multiple_of(block_size) {
            return Err(invalid(
                "kv pool: capacity must be a positive multiple of block size",
            ));
        }
        if (layers == 0) != (kv_heads == 0 || head_dim == 0) {
            return Err(invalid(
                "kv pool: attention geometry must be entirely zero or positive",
            ));
        }
        let recurrent = recurrent.unwrap_or(NativeRecurrentStateSchema {
            kda_layers: 0,
            kda_heads: 0,
            kda_head_dim: 0,
            kda_value_dim: 0,
            conv_layers: 0,
            conv_channels: 0,
            conv_kernel: 0,
        });
        Ok(Self {
            inner: Arc::new(PoolInner {
                ordinal: device,
                layers,
                kv_heads,
                head_dim,
                max_tokens,
                block_size,
                dtype,
                recurrent,
                usage: Mutex::new(PoolUsage::default()),
            }),
        })
    }

    #[napi(getter)]
    pub fn capacity(&self) -> u32 {
        self.inner.max_tokens
    }

    #[napi(getter)]
    pub fn free_blocks(&self) -> u32 {
        let used = self
            .inner
            .usage
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .blocks
            .len() as u32;
        self.inner.max_tokens / self.inner.block_size - used
    }

    #[napi(getter)]
    pub fn cached_blocks(&self) -> u32 {
        self.inner
            .usage
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .blocks
            .values()
            .filter(|&&references| references == 0)
            .count() as u32
    }

    #[napi]
    pub fn make_sequence(&self) -> NativeKvSequence {
        let kda_state_size = self.inner.recurrent.kda_heads as usize
            * self.inner.recurrent.kda_head_dim as usize
            * self.inner.recurrent.kda_value_dim as usize;
        let conv_state_size = self.inner.recurrent.conv_kernel.saturating_sub(1) as usize
            * self.inner.recurrent.conv_channels as usize;
        NativeKvSequence {
            inner: Arc::new(SequenceInner {
                pool: self.inner.clone(),
                state: Mutex::new(SequenceState {
                    keys: vec![Vec::new(); self.inner.layers as usize],
                    values: vec![Vec::new(); self.inner.layers as usize],
                    kda_states: vec![
                        vec![0.0; kda_state_size];
                        self.inner.recurrent.kda_layers as usize
                    ],
                    conv_states: vec![
                        vec![0.0; conv_state_size];
                        self.inner.recurrent.conv_layers as usize
                    ],
                    ..SequenceState::default()
                }),
                device_cache: Mutex::new(None),
                released: AtomicBool::new(false),
                running: AtomicBool::new(false),
            }),
        }
    }
}

#[napi]
pub struct NativeKvSequence {
    inner: Arc<SequenceInner>,
}

impl NativeKvSequence {
    fn lease(&self) -> Result<SequenceLease> {
        if self.inner.released.load(Ordering::Acquire) {
            return Err(invalid("kv sequence was released"));
        }
        if self
            .inner
            .running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(invalid("kv sequence is already in use"));
        }
        Ok(SequenceLease {
            inner: self.inner.clone(),
        })
    }
}

#[napi]
impl NativeKvSequence {
    #[napi(getter)]
    pub fn cursor(&self) -> u32 {
        self.inner
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .cursor
    }

    #[napi]
    pub fn fork(&self) -> Result<Self> {
        let _lease = self.lease()?;
        let state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone();
        let device_cache = {
            let cache = self
                .inner
                .device_cache
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            cache
                .as_ref()
                .map(|cache| {
                    let device = CudaDevice::get(self.inner.pool.ordinal).map_err(failure)?;
                    let mut keys = unsafe { device.stream.alloc::<f32>(cache.keys.len()) }
                        .map_err(|error| failure(error.to_string()))?;
                    let mut values = unsafe { device.stream.alloc::<f32>(cache.values.len()) }
                        .map_err(|error| failure(error.to_string()))?;
                    device
                        .stream
                        .memcpy_dtod(cache.keys.as_ref(), &mut keys)
                        .map_err(|error| failure(error.to_string()))?;
                    device
                        .stream
                        .memcpy_dtod(cache.values.as_ref(), &mut values)
                        .map_err(|error| failure(error.to_string()))?;
                    Ok::<_, Error>(SequenceDeviceCache {
                        keys: Arc::new(keys),
                        values: Arc::new(values),
                        layer_size: cache.layer_size,
                        cursor: None,
                        valid: None,
                        decode_graph: None,
                    })
                })
                .transpose()?
        };
        let mut usage = self
            .inner
            .pool
            .usage
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        for key in &state.block_keys {
            let references = usage
                .blocks
                .get_mut(key)
                .ok_or_else(|| failure("kv sequence references a missing pool block"))?;
            *references += 1;
        }
        Ok(Self {
            inner: Arc::new(SequenceInner {
                pool: self.inner.pool.clone(),
                state: Mutex::new(state),
                device_cache: Mutex::new(device_cache),
                released: AtomicBool::new(false),
                running: AtomicBool::new(false),
            }),
        })
    }

    #[napi]
    pub fn release(&self) {
        if !self.inner.released.swap(true, Ordering::AcqRel) {
            let mut usage = self
                .inner
                .pool
                .usage
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let mut state = self
                .inner
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            state.cursor = 0;
            state.tokens.clear();
            for key in state.block_keys.drain(..) {
                if let Some(references) = usage.blocks.get_mut(&key) {
                    *references = references.saturating_sub(1);
                }
            }
            for layer in &mut state.keys {
                layer.fill(0.0);
            }
            for layer in &mut state.values {
                layer.fill(0.0);
            }
            for layer in &mut state.kda_states {
                layer.fill(0.0);
            }
            for layer in &mut state.conv_states {
                layer.fill(0.0);
            }
            *self
                .inner
                .device_cache
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = None;
        }
    }

    #[napi]
    pub fn prefill_match(&self, tokens: Vec<u32>) -> Result<u32> {
        if self.inner.released.load(Ordering::Acquire) {
            return Err(invalid("kv sequence was released"));
        }
        if self.inner.running.load(Ordering::Acquire) {
            return Err(invalid("kv sequence is already in use"));
        }
        let pool = &self.inner.pool;
        let recurrent = pool.recurrent.kda_layers > 0 || pool.recurrent.conv_layers > 0;
        if recurrent && pool.layers == 0 {
            return Ok(0);
        }
        let mut usage = pool.usage.lock().unwrap_or_else(|error| error.into_inner());
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if state.cursor != 0 || !state.block_keys.is_empty() {
            return Err(invalid("prefill match: sequence already holds tokens"));
        }
        let block_size = pool.block_size as usize;
        let matchable = tokens.len().saturating_sub(1) / block_size;
        for blocks in (1..=matchable).rev() {
            let end = blocks * block_size;
            let key = BlockKey::new(format!(
                "full:{end}:{}",
                tokens[..end]
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            ));
            let Some(snapshot) = usage.snapshots.get(&key).cloned() else {
                continue;
            };
            if snapshot
                .block_keys
                .iter()
                .any(|block| !usage.blocks.contains_key(block))
            {
                continue;
            }
            for block in &snapshot.block_keys {
                *usage
                    .blocks
                    .get_mut(block)
                    .expect("snapshot block was checked") += 1;
            }
            *state = (*snapshot).clone();
            *self
                .inner
                .device_cache
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = None;
            return Ok(end as u32);
        }
        Ok(0)
    }
}

fn attributes(value: &str) -> Result<JsonValue> {
    serde_json::from_str(value)
        .map_err(|error| invalid(format!("invalid node attributes: {error}")))
}

fn attribute<'a>(attributes: &'a JsonValue, name: &str) -> Result<&'a JsonValue> {
    attributes
        .get(name)
        .ok_or_else(|| invalid(format!("missing node attribute {name}")))
}

fn number(attributes: &JsonValue, name: &str) -> Result<f64> {
    attribute(attributes, name)?
        .as_f64()
        .ok_or_else(|| invalid(format!("node attribute {name} must be a number")))
}

fn integer(attributes: &JsonValue, name: &str) -> Result<usize> {
    let value = attribute(attributes, name)?
        .as_u64()
        .ok_or_else(|| invalid(format!("node attribute {name} must be an unsigned integer")))?;
    usize::try_from(value).map_err(|_| invalid(format!("node attribute {name} exceeds usize")))
}

fn signed_integer(attributes: &JsonValue, name: &str) -> Result<i64> {
    attribute(attributes, name)?
        .as_i64()
        .ok_or_else(|| invalid(format!("node attribute {name} must be an integer")))
}

fn boolean(attributes: &JsonValue, name: &str) -> Result<bool> {
    attribute(attributes, name)?
        .as_bool()
        .ok_or_else(|| invalid(format!("node attribute {name} must be a boolean")))
}

fn string<'a>(attributes: &'a JsonValue, name: &str) -> Result<&'a str> {
    attribute(attributes, name)?
        .as_str()
        .ok_or_else(|| invalid(format!("node attribute {name} must be a string")))
}

fn dimensions(attributes: &JsonValue, name: &str) -> Result<Vec<usize>> {
    attribute(attributes, name)?
        .as_array()
        .ok_or_else(|| invalid(format!("node attribute {name} must be an array")))?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    invalid(format!(
                        "node attribute {name} must contain unsigned integers"
                    ))
                })
        })
        .collect()
}

fn ranges(attributes: &JsonValue) -> Result<Vec<(usize, usize, usize)>> {
    attribute(attributes, "ranges")?
        .as_array()
        .ok_or_else(|| invalid("node attribute ranges must be an array"))?
        .iter()
        .map(|range| {
            let range = range
                .as_array()
                .ok_or_else(|| invalid("each slice range must be an array"))?;
            if range.len() != 3 {
                return Err(invalid("each slice range must have three entries"));
            }
            let value = |index: usize| {
                range[index]
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| invalid("slice ranges must contain unsigned integers"))
            };
            Ok((value(0)?, value(1)?, value(2)?))
        })
        .collect()
}

fn input(inputs: &[Arc<Node>], index: usize, operation: &str) -> Result<Arc<Node>> {
    inputs
        .get(index)
        .cloned()
        .ok_or_else(|| invalid(format!("{operation}: missing tensor input {index}")))
}

fn lazy(node: std::result::Result<Arc<Node>, String>) -> Result<LazyTensor> {
    node.map(|node| LazyTensor { node }).map_err(invalid)
}

/// Whether a usable CUDA device and `compute_120` NVRTC compiler are present.
#[napi]
pub fn is_available() -> bool {
    CudaDevice::get(0).is_ok()
}

/// Number of CUDA devices visible to this process.
#[napi]
pub fn device_count() -> Result<u32> {
    CudaDevice::count().map_err(failure)
}

#[napi]
pub struct CancellationToken {
    state: Arc<CancellationState>,
    notify: Arc<tokio::sync::Notify>,
}

#[napi]
impl CancellationToken {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            state: Arc::new(CancellationState::new()),
            notify: Arc::new(tokio::sync::Notify::new()),
        }
    }

    #[napi]
    pub fn cancel(&self) {
        if self.state.cancel() {
            self.notify.notify_one();
        }
    }

    #[napi(getter)]
    pub fn cancelled(&self) -> bool {
        self.state.flag().is_cancelled()
    }
}

#[napi]
pub struct LazyTensor {
    node: Arc<Node>,
}

#[napi]
pub struct NativeExposure {
    name: String,
    tensor: LazyTensor,
}

#[napi]
impl NativeExposure {
    #[napi(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    #[napi(getter)]
    pub fn tensor(&self) -> LazyTensor {
        LazyTensor {
            node: self.tensor.node.clone(),
        }
    }
}

#[napi]
impl LazyTensor {
    #[napi(getter)]
    pub fn shape(&self) -> Vec<u32> {
        self.node
            .shape
            .iter()
            .map(|dimension| *dimension as u32)
            .collect()
    }

    #[napi(getter)]
    pub fn dtype(&self) -> String {
        self.node.dtype.name().to_string()
    }

    #[napi(getter)]
    pub fn device(&self) -> String {
        match self.node.device {
            Device::Cuda(ordinal) => format!("cuda:{ordinal}"),
            _ => self.node.device.name().to_string(),
        }
    }

    #[napi]
    pub fn exposures(&self) -> Result<Vec<NativeExposure>> {
        let mut seen = HashSet::new();
        let mut names = HashSet::new();
        let mut found = Vec::new();
        let mut stack = vec![self.node.clone()];
        while let Some(node) = stack.pop() {
            if !seen.insert(node.id) {
                continue;
            }
            if let NodeKind::Expose { a, name } = &node.kind {
                if !names.insert(name.clone()) {
                    return Err(invalid(format!(
                        "expose: duplicate exposure name \"{name}\""
                    )));
                }
                found.push(NativeExposure {
                    name: name.clone(),
                    tensor: LazyTensor { node: a.clone() },
                });
            }
            stack.extend(effect_torch_graph::node_children(&node.kind));
        }
        Ok(found)
    }
}

#[derive(Clone)]
#[napi]
pub struct NativeTensor {
    slot: Arc<LeafSlot>,
    ordinal: u32,
}

impl NativeTensor {
    fn wrap(value: CudaValue) -> Self {
        let ordinal = value.ordinal();
        Self {
            slot: Arc::new(LeafSlot::new(value)),
            ordinal,
        }
    }

    fn value(&self) -> Result<CudaValue> {
        self.slot
            .get::<CudaValue>()
            .map_err(|error| failure(error.to_string()))
    }
}

#[napi]
impl NativeTensor {
    #[napi]
    pub fn clear(&self) {
        self.slot.clear();
    }

    #[napi(js_name = "writeBytes")]
    pub fn write_bytes(&self, data: Uint8Array) -> Result<()> {
        let value = self.value()?;
        if value.dtype() == DType::I64 {
            if !data.len().is_multiple_of(8) {
                return Err(invalid(format!(
                    "CUDA i64 byte length {} is not divisible by 8",
                    data.len()
                )));
            }
            return value
                .write_i64_host(
                    &data
                        .chunks_exact(8)
                        .map(|bytes| {
                            i64::from_le_bytes(bytes.try_into().expect("eight-byte chunk"))
                        })
                        .collect::<Vec<_>>(),
                )
                .map_err(failure);
        }
        value
            .write_host(&crate::executable::decode(&data, value.dtype()).map_err(failure)?)
            .map_err(failure)
    }

    #[napi(getter)]
    pub fn shape(&self) -> Result<Vec<u32>> {
        Ok(self
            .value()?
            .shape()
            .iter()
            .map(|dimension| *dimension as u32)
            .collect())
    }

    #[napi(getter)]
    pub fn dtype(&self) -> Result<String> {
        Ok(self.value()?.dtype().name().to_string())
    }

    #[napi(getter)]
    pub fn device(&self) -> Result<String> {
        self.value()?;
        Ok(format!("cuda:{}", self.ordinal))
    }

    #[napi]
    pub async fn readback(&self, token: Option<&CancellationToken>) -> Result<Buffer> {
        let value = self.value()?;
        let state = token
            .map(|token| token.state.clone())
            .unwrap_or_else(|| Arc::new(CancellationState::new()));
        let notify = token.map(|token| token.notify.clone());
        run_compute(state, notify, move |cancelled, _| {
            if cancelled.is_cancelled() {
                return Err(Error::new(Status::Cancelled, "operation aborted"));
            }
            let dtype = value.dtype();
            if dtype == DType::I64 {
                let values = value.readback_i64().map_err(failure)?;
                let mut bytes = Vec::with_capacity(values.len() * 8);
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                return Ok(Buffer::from(bytes));
            }
            let values = value.readback().map_err(failure)?;
            if cancelled.is_cancelled() {
                return Err(Error::new(Status::Cancelled, "operation aborted"));
            }
            Ok(Buffer::from(encode(values, dtype)))
        })
        .await
    }

    #[napi]
    #[allow(clippy::too_many_arguments)]
    pub async fn sample(
        &self,
        temperature: f64,
        top_k: f64,
        top_p: f64,
        seed: f64,
        counter: f64,
        token: Option<&CancellationToken>,
    ) -> Result<u32> {
        let value = self.value()?;
        if value.shape().len() != 1 {
            return Err(invalid(format!(
                "sample: logits must be rank 1, got rank {}",
                value.shape().len()
            )));
        }
        if !matches!(
            value.dtype(),
            DType::F64 | DType::F32 | DType::F16 | DType::BF16
        ) {
            return Err(invalid(format!(
                "sample: logits must have a floating-point dtype, got {}",
                value.dtype().name()
            )));
        }
        let options = sampling_options(temperature, top_k, top_p, seed, counter)?;
        let state = token
            .map(|token| token.state.clone())
            .unwrap_or_else(|| Arc::new(CancellationState::new()));
        let notify = token.map(|token| token.notify.clone());
        run_compute(state, notify, move |cancelled, _| {
            sample_cuda_value(&value, options, || cancelled.is_cancelled()).map_err(|message| {
                if message == "operation aborted" {
                    Error::new(Status::Cancelled, message)
                } else {
                    invalid(message)
                }
            })
        })
        .await
    }
}

#[napi]
pub struct Executable {
    inner: Arc<CudaExecutable>,
    state: Option<CudaStateSchema>,
}

impl Executable {
    fn sequence_leases(
        &self,
        sequences: Vec<&NativeKvSequence>,
        slots: &[u32],
        active_mask: &[bool],
        valid_lengths: &[u32],
        advances: &[u32],
        tokens: &[Vec<u32>],
    ) -> Result<Vec<SequenceLease>> {
        let schema = self
            .state
            .as_ref()
            .ok_or_else(|| invalid("execute: state invocation requires a stateful executable"))?;
        let batch = schema.batch as usize;
        if sequences.is_empty()
            || sequences.len() > batch
            || sequences.len() != slots.len()
            || sequences.len() != tokens.len()
            || active_mask.len() != batch
            || valid_lengths.len() != batch
            || advances.len() != batch
        {
            return Err(invalid("execute: invalid fixed-lane state metadata"));
        }
        let mut seen = vec![false; batch];
        let mut leases = Vec::with_capacity(sequences.len());
        for (request, sequence) in sequences.into_iter().enumerate() {
            let slot = slots[request] as usize;
            if slot >= batch
                || seen[slot]
                || tokens[request].is_empty()
                || tokens[request].len() != advances[slot] as usize
                || valid_lengths[slot] != advances[slot]
            {
                return Err(invalid("execute: invalid sequence slot or token row"));
            }
            seen[slot] = true;
            let pool = &sequence.inner.pool;
            if leases
                .first()
                .is_some_and(|lease: &SequenceLease| !Arc::ptr_eq(&lease.inner.pool, pool))
            {
                return Err(invalid(
                    "execute: every sequence must use the same state pool",
                ));
            }
            let recurrent = &pool.recurrent;
            if pool.ordinal != self.inner.ordinal()
                || pool.layers != schema.geometry.layers as u32
                || pool.kv_heads != schema.geometry.kv_heads as u32
                || pool.head_dim != schema.geometry.head_dim as u32
                || pool.max_tokens != schema.max_tokens
                || pool.block_size != schema.block_size
                || pool.dtype != schema.kv_dtype
                || recurrent.kda_layers != schema.geometry.kda.layers as u32
                || recurrent.kda_heads != schema.geometry.kda.heads as u32
                || recurrent.kda_head_dim != schema.geometry.kda.head_dim as u32
                || recurrent.kda_value_dim != schema.geometry.kda.value_dim as u32
                || recurrent.conv_layers != schema.geometry.conv.layers as u32
                || recurrent.conv_channels != schema.geometry.conv.channels as u32
                || recurrent.conv_kernel != schema.geometry.conv.kernel as u32
            {
                return Err(invalid(
                    "execute: pool geometry does not match executable state",
                ));
            }
            let cursor = sequence
                .inner
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .cursor;
            if schema.window.is_none() && cursor.saturating_add(advances[slot]) > schema.max_tokens
            {
                return Err(invalid(format!(
                    "execute: sequence context exceeds pool capacity {}",
                    schema.max_tokens
                )));
            }
            leases.push(sequence.lease()?);
        }
        for lane in 0..batch {
            if active_mask[lane] != seen[lane]
                || (!seen[lane] && (valid_lengths[lane] != 0 || advances[lane] != 0))
            {
                return Err(invalid("execute: inconsistent fixed-lane state metadata"));
            }
        }
        Ok(leases)
    }

    fn block_plan(
        schema: &CudaStateSchema,
        leases: &[SequenceLease],
        tokens: &[Vec<u32>],
        usage: &PoolUsage,
    ) -> Result<(PoolUsage, Vec<Vec<BlockKey>>)> {
        let block_size = schema.block_size as usize;
        let capacity = (schema.max_tokens / schema.block_size) as usize;
        let mut planned = usage.clone();
        let mut desired = Vec::with_capacity(leases.len());
        for (request, lease) in leases.iter().enumerate() {
            let state = lease
                .inner
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            for key in &state.block_keys {
                if let Some(references) = planned.blocks.get_mut(key) {
                    *references = references.saturating_sub(1);
                }
            }
            let mut combined = state.tokens.clone();
            combined.extend_from_slice(&tokens[request]);
            let previous_blocks = state.tokens.len().div_ceil(block_size);
            let previous_retained = schema.window.map_or(previous_blocks, |window| {
                previous_blocks.min((window as usize).div_ceil(block_size))
            });
            let previous_start = previous_blocks.saturating_sub(previous_retained);
            let unchanged = state.tokens.len() / block_size;
            let blocks = combined.len().div_ceil(block_size);
            let retained = schema.window.map_or(blocks, |window| {
                blocks.min((window as usize).div_ceil(block_size))
            });
            let start = blocks.saturating_sub(retained);
            let identity = Arc::as_ptr(&lease.inner) as usize;
            desired.push(
                (start..blocks)
                    .map(|block| {
                        if block < unchanged && block >= previous_start {
                            return state.block_keys[block - previous_start].clone();
                        }
                        let end = ((block + 1) * block_size).min(combined.len());
                        if end % block_size == 0 {
                            BlockKey::new(format!(
                                "full:{end}:{}",
                                combined[..end]
                                    .iter()
                                    .map(u32::to_string)
                                    .collect::<Vec<_>>()
                                    .join(",")
                            ))
                        } else {
                            BlockKey::new(format!("partial:{identity}:{block}"))
                        }
                    })
                    .collect::<Vec<_>>(),
            );
        }
        let retained = desired.iter().flatten().cloned().collect::<HashSet<_>>();
        for keys in &desired {
            for key in keys {
                if let Some(references) = planned.blocks.get_mut(key) {
                    *references += 1;
                    continue;
                }
                while planned.blocks.len() >= capacity {
                    let evict = planned.blocks.iter().find_map(|(key, &references)| {
                        (references == 0 && !retained.contains(key)).then(|| key.clone())
                    });
                    let Some(evict) = evict else {
                        return Err(invalid("execute: KV pool exhausted"));
                    };
                    planned.blocks.remove(&evict);
                    planned.snapshots.remove(&evict);
                }
                planned.blocks.insert(key.clone(), 1);
            }
        }
        Ok((planned, desired))
    }

    fn decode_state_with_schema(
        schema: &CudaStateSchema,
        leases: &[SequenceLease],
        slots: &[u32],
        valid_lengths: &[u32],
    ) -> Result<CudaStateInvocation> {
        let device_cache = if leases.len() == 1 && slots == [0] && valid_lengths.len() == 1 {
            leases[0]
                .inner
                .device_cache
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .take()
        } else {
            None
        };
        let retained_kv = device_cache.is_some();
        let (key_cache, value_cache, cache_layer_size, cursor_device, valid_device, decode_graph) =
            device_cache.map_or((None, None, None, None, None, None), |cache| {
                (
                    Some(cache.keys),
                    Some(cache.values),
                    Some(cache.layer_size),
                    cache.cursor,
                    cache.valid,
                    cache.decode_graph,
                )
            });
        Ok(CudaStateInvocation {
            sequences: leases
                .iter()
                .map(|lease| {
                    let state = lease
                        .inner
                        .state
                        .lock()
                        .unwrap_or_else(|error| error.into_inner());
                    CudaSequenceState {
                        cursor: state.cursor,
                        keys: if retained_kv {
                            Vec::new()
                        } else {
                            state.keys.clone()
                        },
                        values: if retained_kv {
                            Vec::new()
                        } else {
                            state.values.clone()
                        },
                        kda_states: state.kda_states.clone(),
                        conv_states: state.conv_states.clone(),
                    }
                })
                .collect(),
            slots: slots.to_vec(),
            valid_lengths: valid_lengths.to_vec(),
            capacity: leases[0].inner.pool.max_tokens,
            cache_dtype: leases[0].inner.pool.dtype,
            packed_rows_per_sequence: schema.packed_rows_per_sequence,
            key_cache,
            value_cache,
            cache_layer_size,
            cursor_device,
            valid_device,
            decode_graph,
            prepared_kv: None,
        })
    }

    fn should_readback_kv(leases: &[SequenceLease], slots: &[u32], advances: &[u32]) -> bool {
        if leases.len() != 1 || slots != [0] || advances.len() != 1 {
            return true;
        }
        let pool = &leases[0].inner.pool;
        if pool.recurrent.kda_layers > 0 || pool.recurrent.conv_layers > 0 {
            let cursor = leases[0]
                .inner
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .cursor;
            return cursor
                .saturating_add(advances[0])
                .is_multiple_of(pool.block_size);
        }
        false
    }

    fn retain_device_cache(
        leases: &[SequenceLease],
        slots: &[u32],
        decoded: &mut CudaStateInvocation,
    ) {
        if leases.len() != 1 || slots != [0] || decoded.valid_lengths.len() != 1 {
            return;
        }
        let (Some(keys), Some(values), Some(layer_size)) = (
            decoded.key_cache.take(),
            decoded.value_cache.take(),
            decoded.cache_layer_size,
        ) else {
            return;
        };
        *leases[0]
            .inner
            .device_cache
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(SequenceDeviceCache {
            keys,
            values,
            layer_size,
            cursor: decoded.cursor_device.take(),
            valid: decoded.valid_device.take(),
            decode_graph: decoded.decode_graph.take(),
        });
    }

    fn with_retained_device_cache<A>(
        leases: &[SequenceLease],
        slots: &[u32],
        decoded: &mut CudaStateInvocation,
        execute: impl FnOnce(&mut CudaStateInvocation) -> Result<A>,
    ) -> Result<A> {
        let result = execute(decoded);
        Self::retain_device_cache(leases, slots, decoded);
        result
    }

    fn commit_sequences(
        leases: &[SequenceLease],
        slots: &[u32],
        advances: &[u32],
        tokens: &[Vec<u32>],
        decoded: &CudaStateInvocation,
        commit_kv: bool,
        block_keys: Vec<Vec<BlockKey>>,
        usage: &mut PoolUsage,
    ) {
        for (request, lease) in leases.iter().enumerate() {
            let mut state = lease
                .inner
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let advance = advances[slots[request] as usize];
            state.cursor = state.cursor.saturating_add(advance);
            state.tokens.extend_from_slice(&tokens[request]);
            if commit_kv {
                state.keys = decoded.sequences[request].keys.clone();
                state.values = decoded.sequences[request].values.clone();
            }
            state.kda_states = decoded.sequences[request].kda_states.clone();
            state.conv_states = decoded.sequences[request].conv_states.clone();
            state.block_keys = block_keys[request].clone();
            if (commit_kv || lease.inner.pool.layers == 0)
                && state.cursor.is_multiple_of(lease.inner.pool.block_size)
            {
                if let Some(key) = state.block_keys.last().cloned() {
                    usage.snapshots.insert(key, Arc::new(state.clone()));
                }
            }
        }
    }

    fn advance_sequences(
        leases: &[SequenceLease],
        slots: &[u32],
        advances: &[u32],
        tokens: &[Vec<u32>],
        block_keys: Vec<Vec<BlockKey>>,
    ) {
        for (request, lease) in leases.iter().enumerate() {
            let mut state = lease
                .inner
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            state.cursor = state
                .cursor
                .saturating_add(advances[slots[request] as usize]);
            state.tokens.extend_from_slice(&tokens[request]);
            state.block_keys = block_keys[request].clone();
        }
    }

    fn commit_sequence_caches(
        leases: &[SequenceLease],
        decoded: &CudaStateInvocation,
        commit_kv: bool,
        usage: &mut PoolUsage,
    ) {
        for (request, lease) in leases.iter().enumerate() {
            let mut state = lease
                .inner
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if commit_kv {
                state.keys = decoded.sequences[request].keys.clone();
                state.values = decoded.sequences[request].values.clone();
            }
            state.kda_states = decoded.sequences[request].kda_states.clone();
            state.conv_states = decoded.sequences[request].conv_states.clone();
            if (commit_kv || lease.inner.pool.layers == 0)
                && state.cursor.is_multiple_of(lease.inner.pool.block_size)
            {
                if let Some(key) = state.block_keys.last().cloned() {
                    usage.snapshots.insert(key, Arc::new(state.clone()));
                }
            }
        }
    }
}

#[napi]
impl Executable {
    #[napi(getter)]
    pub fn diagnostics(&self) -> NativeExecutableDiagnostics {
        executable_diagnostics(self.inner.diagnostics())
    }

    #[napi(getter)]
    pub fn stateful(&self) -> bool {
        self.state.is_some()
    }

    #[napi(getter)]
    pub fn batch(&self) -> u32 {
        self.state.as_ref().map_or(0, |state| state.batch)
    }

    #[napi(getter)]
    pub fn allows_window_eviction(&self) -> bool {
        self.state
            .as_ref()
            .is_some_and(|state| state.geometry.allows_window_eviction)
    }

    #[napi(getter)]
    pub fn layers(&self) -> u32 {
        self.state
            .as_ref()
            .map_or(0, |state| state.geometry.layers as u32)
    }

    #[napi(getter)]
    pub fn kv_heads(&self) -> u32 {
        self.state
            .as_ref()
            .map_or(0, |state| state.geometry.kv_heads as u32)
    }

    #[napi(getter)]
    pub fn head_dim(&self) -> u32 {
        self.state
            .as_ref()
            .map_or(0, |state| state.geometry.head_dim as u32)
    }

    #[napi(getter)]
    pub fn kda_layers(&self) -> u32 {
        self.state
            .as_ref()
            .map_or(0, |state| state.geometry.kda.layers as u32)
    }

    #[napi(getter)]
    pub fn kda_heads(&self) -> u32 {
        self.state
            .as_ref()
            .map_or(0, |state| state.geometry.kda.heads as u32)
    }

    #[napi(getter)]
    pub fn kda_head_dim(&self) -> u32 {
        self.state
            .as_ref()
            .map_or(0, |state| state.geometry.kda.head_dim as u32)
    }

    #[napi(getter)]
    pub fn kda_value_dim(&self) -> u32 {
        self.state
            .as_ref()
            .map_or(0, |state| state.geometry.kda.value_dim as u32)
    }

    #[napi(getter)]
    pub fn conv_layers(&self) -> u32 {
        self.state
            .as_ref()
            .map_or(0, |state| state.geometry.conv.layers as u32)
    }

    #[napi(getter)]
    pub fn conv_channels(&self) -> u32 {
        self.state
            .as_ref()
            .map_or(0, |state| state.geometry.conv.channels as u32)
    }

    #[napi(getter)]
    pub fn conv_kernel(&self) -> u32 {
        self.state
            .as_ref()
            .map_or(0, |state| state.geometry.conv.kernel as u32)
    }

    #[napi(getter)]
    pub fn device(&self) -> String {
        format!("cuda:{}", self.inner.ordinal())
    }

    #[napi(getter)]
    pub fn instruction_count(&self) -> u32 {
        self.inner.instruction_count() as u32
    }

    #[napi]
    pub async fn execute(
        &self,
        bindings: Vec<&NativeTensor>,
        scalars: Vec<f64>,
        token: Option<&CancellationToken>,
    ) -> Result<Vec<NativeTensor>> {
        let bindings = bindings
            .into_iter()
            .map(NativeTensor::value)
            .collect::<Result<Vec<_>>>()?;
        let executable = self.inner.clone();
        let state = token
            .map(|token| token.state.clone())
            .unwrap_or_else(|| Arc::new(CancellationState::new()));
        let notify = token.map(|token| token.notify.clone());
        run_compute(state, notify, move |cancelled, _| {
            executable
                .execute(&bindings, &scalars, cancelled)
                .map(|values| values.into_iter().map(NativeTensor::wrap).collect())
                .map_err(|message| {
                    if cancelled.is_cancelled() {
                        Error::new(Status::Cancelled, "operation aborted")
                    } else {
                        failure(message)
                    }
                })
        })
        .await
    }

    #[napi]
    #[allow(clippy::too_many_arguments)]
    pub async fn execute_stateful(
        &self,
        bindings: Vec<&NativeTensor>,
        sequences: Vec<&NativeKvSequence>,
        slots: Vec<u32>,
        active_mask: Vec<bool>,
        valid_lengths: Vec<u32>,
        advances: Vec<u32>,
        tokens: Vec<Vec<u32>>,
        token: Option<&CancellationToken>,
    ) -> Result<Vec<NativeTensor>> {
        let leases = self.sequence_leases(
            sequences,
            &slots,
            &active_mask,
            &valid_lengths,
            &advances,
            &tokens,
        )?;
        let bindings = bindings
            .into_iter()
            .map(NativeTensor::value)
            .collect::<Result<Vec<_>>>()?;
        let executable = self.inner.clone();
        let schema = self.state.clone().expect("state invocation was validated");
        let state = token
            .map(|token| token.state.clone())
            .unwrap_or_else(|| Arc::new(CancellationState::new()));
        let notify = token.map(|token| token.notify.clone());
        run_compute(state, notify, move |cancelled, _| {
            let mut decode_state =
                Self::decode_state_with_schema(&schema, &leases, &slots, &valid_lengths)?;
            Self::with_retained_device_cache(&leases, &slots, &mut decode_state, |decode_state| {
                let pool = leases[0].inner.pool.clone();
                let mut usage = pool.usage.lock().unwrap_or_else(|error| error.into_inner());
                let (mut planned, block_keys) =
                    Self::block_plan(&schema, &leases, &tokens, &usage)?;
                let values = executable
                    .execute_stateful(&bindings, &[], decode_state, cancelled)
                    .map_err(failure)?;
                let readback_kv = Self::should_readback_kv(&leases, &slots, &advances);
                if readback_kv {
                    executable.readback_state(decode_state).map_err(failure)?;
                }
                Self::commit_sequences(
                    &leases,
                    &slots,
                    &advances,
                    &tokens,
                    decode_state,
                    readback_kv,
                    block_keys,
                    &mut planned,
                );
                *usage = planned;
                Ok(values.into_iter().map(NativeTensor::wrap).collect())
            })
        })
        .await
    }

    #[napi]
    #[allow(clippy::too_many_arguments)]
    pub async fn execute_sampled(
        &self,
        bindings: Vec<&NativeTensor>,
        sequences: Vec<&NativeKvSequence>,
        slots: Vec<u32>,
        active_mask: Vec<bool>,
        valid_lengths: Vec<u32>,
        advances: Vec<u32>,
        tokens: Vec<Vec<u32>>,
        sampling: Vec<NativeSamplingOptions>,
        token: Option<&CancellationToken>,
    ) -> Result<Vec<u32>> {
        if sampling.len() != sequences.len() {
            return Err(invalid(
                "executeSampled: expected one sampling policy per sequence",
            ));
        }
        let options = sampling
            .into_iter()
            .map(|options| {
                sampling_options(
                    options.temperature,
                    options.top_k,
                    options.top_p,
                    options.seed,
                    options.counter,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let leases = self.sequence_leases(
            sequences,
            &slots,
            &active_mask,
            &valid_lengths,
            &advances,
            &tokens,
        )?;
        let bindings = bindings
            .into_iter()
            .map(NativeTensor::value)
            .collect::<Result<Vec<_>>>()?;
        let executable = self.inner.clone();
        let schema = self.state.clone().expect("state invocation was validated");
        let state = token
            .map(|token| token.state.clone())
            .unwrap_or_else(|| Arc::new(CancellationState::new()));
        let notify = token.map(|token| token.notify.clone());
        run_compute(state, notify, move |cancelled, _| {
            let mut decode_state =
                Self::decode_state_with_schema(&schema, &leases, &slots, &valid_lengths)?;
            Self::with_retained_device_cache(&leases, &slots, &mut decode_state, |decode_state| {
                let pool = leases[0].inner.pool.clone();
                let mut usage = pool.usage.lock().unwrap_or_else(|error| error.into_inner());
                let (mut planned, block_keys) =
                    Self::block_plan(&schema, &leases, &tokens, &usage)?;
                let graphed = if options.len() == 1 {
                    executable
                        .execute_stateful_graphed(&bindings, &tokens[0], decode_state, cancelled)
                        .map_err(failure)?
                } else {
                    None
                };
                let sampled = if let Some(logits) = graphed {
                    vec![
                        sample_cuda_value(&logits, options[0], || cancelled.is_cancelled())
                            .map_err(invalid)?,
                    ]
                } else {
                    let values = executable
                        .execute_stateful(&bindings, &[], decode_state, cancelled)
                        .map_err(failure)?;
                    let mut sampled = Vec::with_capacity(slots.len());
                    for (request, slot) in slots.iter().enumerate() {
                        let logits = values
                            .get(*slot as usize)
                            .ok_or_else(|| failure("executeSampled: missing lane output"))?;
                        sampled.push(
                            sample_cuda_value(logits, options[request], || {
                                cancelled.is_cancelled()
                            })
                            .map_err(invalid)?,
                        );
                    }
                    sampled
                };
                let readback_kv = Self::should_readback_kv(&leases, &slots, &advances);
                if readback_kv {
                    executable.readback_state(decode_state).map_err(failure)?;
                }
                Self::commit_sequences(
                    &leases,
                    &slots,
                    &advances,
                    &tokens,
                    decode_state,
                    readback_kv,
                    block_keys,
                    &mut planned,
                );
                *usage = planned;
                Ok(sampled)
            })
        })
        .await
    }

    #[napi]
    #[allow(clippy::too_many_arguments)]
    pub async fn execute_sampled_steps(
        &self,
        bindings: Vec<&NativeTensor>,
        sequences: Vec<&NativeKvSequence>,
        slots: Vec<u32>,
        active_mask: Vec<bool>,
        valid_lengths: Vec<u32>,
        advances: Vec<u32>,
        tokens: Vec<Vec<u32>>,
        sampling: Vec<Vec<NativeSamplingOptions>>,
        token: Option<&CancellationToken>,
    ) -> Result<Vec<Vec<u32>>> {
        if sampling.is_empty() || sampling.iter().any(|step| step.len() != sequences.len()) {
            return Err(invalid(
                "executeSampledSteps: expected one sampling policy per sequence and step",
            ));
        }
        if tokens.iter().any(|row| row.len() != 1)
            || slots.iter().any(|&slot| {
                valid_lengths.get(slot as usize) != Some(&1)
                    || advances.get(slot as usize) != Some(&1)
            })
        {
            return Err(invalid(
                "executeSampledSteps: every active lane must contain one token",
            ));
        }
        let options = sampling
            .into_iter()
            .map(|step| {
                step.into_iter()
                    .map(|options| {
                        sampling_options(
                            options.temperature,
                            options.top_k,
                            options.top_p,
                            options.seed,
                            options.counter,
                        )
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let leases = self.sequence_leases(
            sequences,
            &slots,
            &active_mask,
            &valid_lengths,
            &advances,
            &tokens,
        )?;
        let mut bindings = bindings
            .into_iter()
            .map(NativeTensor::value)
            .collect::<Result<Vec<_>>>()?;
        let first = bindings
            .first()
            .ok_or_else(|| invalid("executeSampledSteps: missing token binding"))?;
        let binding_shape = first.shape().to_vec();
        let binding_dtype = first.dtype();
        let binding_device = first.device.clone();
        let binding_elements = binding_shape.iter().try_fold(1usize, |total, &dimension| {
            total
                .checked_mul(dimension)
                .ok_or_else(|| invalid("executeSampledSteps: token binding size overflowed"))
        })?;
        let batch = self
            .state
            .as_ref()
            .expect("state invocation was validated")
            .batch as usize;
        if binding_elements % batch != 0 {
            return Err(invalid(
                "executeSampledSteps: token binding does not match the decode batch",
            ));
        }
        let lane_width = binding_elements / batch;
        let executable = self.inner.clone();
        let schema = self.state.clone().expect("state invocation was validated");
        let state = token
            .map(|token| token.state.clone())
            .unwrap_or_else(|| Arc::new(CancellationState::new()));
        let notify = token.map(|token| token.notify.clone());
        run_compute(state, notify, move |cancelled, _| {
            let mut decode_state =
                Self::decode_state_with_schema(&schema, &leases, &slots, &valid_lengths)?;
            Self::with_retained_device_cache(&leases, &slots, &mut decode_state, |decode_state| {
                let pool = leases[0].inner.pool.clone();
                let mut usage = pool.usage.lock().unwrap_or_else(|error| error.into_inner());
                let mut current_tokens = tokens;
                let mut sampled = vec![Vec::with_capacity(options.len()); slots.len()];
                let step_count = options.len();
                for (step_index, step) in options.into_iter().enumerate() {
                    let (planned, block_keys) =
                        Self::block_plan(&schema, &leases, &current_tokens, &usage)?;
                    let values = executable
                        .execute_stateful(&bindings, &[], decode_state, cancelled)
                        .map_err(failure)?;
                    let mut next = Vec::with_capacity(slots.len());
                    for (request, slot) in slots.iter().enumerate() {
                        let logits = values
                            .get(*slot as usize)
                            .ok_or_else(|| failure("executeSampledSteps: missing lane output"))?;
                        let token =
                            sample_cuda_value(logits, step[request], || cancelled.is_cancelled())
                                .map_err(invalid)?;
                        sampled[request].push(token);
                        next.push(token);
                    }
                    drop(values);
                    Self::advance_sequences(
                        &leases,
                        &slots,
                        &advances,
                        &current_tokens,
                        block_keys,
                    );
                    *usage = planned;
                    for (request, &slot) in slots.iter().enumerate() {
                        decode_state.sequences[request].cursor = decode_state.sequences[request]
                            .cursor
                            .saturating_add(advances[slot as usize]);
                    }
                    if step_index + 1 < step_count {
                        current_tokens = next.iter().map(|&token| vec![token]).collect();
                        let mut host = vec![0.0; binding_elements];
                        for (request, &slot) in slots.iter().enumerate() {
                            host[slot as usize * lane_width] = f64::from(next[request]);
                        }
                        bindings[0] = CudaValue::from_host(
                            binding_device.clone(),
                            binding_shape.clone(),
                            binding_dtype,
                            &host,
                        )
                        .map_err(failure)?;
                    }
                }
                let readback_kv = Self::should_readback_kv(&leases, &slots, &[0]);
                if readback_kv {
                    executable.readback_state(decode_state).map_err(failure)?;
                }
                Self::commit_sequence_caches(&leases, decode_state, readback_kv, &mut usage);
                Ok(sampled)
            })
        })
        .await
    }

    #[napi]
    #[allow(clippy::too_many_arguments)]
    pub async fn execute_target_matching(
        &self,
        sequences: Vec<&NativeKvSequence>,
        slots: Vec<u32>,
        tokens: Vec<Vec<u32>>,
        sampling: Vec<Vec<NativeSamplingOptions>>,
        page_limits: Vec<u32>,
        eos_tokens: Vec<Vec<u32>>,
        proposal_probabilities: Option<&NativeTensor>,
        token: Option<&CancellationToken>,
    ) -> Result<NativeTargetMatchingOutput> {
        let schema = self
            .state
            .clone()
            .ok_or_else(|| invalid("executeTargetMatching: verifier must be stateful"))?;
        let rows = schema
            .packed_rows_per_sequence
            .ok_or_else(|| invalid("executeTargetMatching: verifier must use packed causal rows"))?
            as usize;
        if sequences.is_empty()
            || sequences.len() != slots.len()
            || sequences.len() != tokens.len()
            || sequences.len() != page_limits.len()
            || sequences.len() != eos_tokens.len()
            || sampling.len() < rows
            || sampling.iter().any(|step| step.len() != sequences.len())
            || page_limits.iter().enumerate().any(|(request, &limit)| {
                limit == 0
                    || limit as usize > rows
                    || if proposal_probabilities.is_some() {
                        tokens[request].len() != 1
                    } else {
                        tokens[request].len() != limit as usize
                    }
            })
        {
            return Err(invalid(
                "executeTargetMatching: inconsistent sequence, sampling, or page metadata",
            ));
        }
        let options = sampling
            .into_iter()
            .map(|step| {
                step.into_iter()
                    .map(|options| {
                        sampling_options(
                            options.temperature,
                            options.top_k,
                            options.top_p,
                            options.seed,
                            options.counter,
                        )
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let batch = schema.batch as usize;
        let mut active_mask = vec![false; batch];
        let mut valid_lengths = vec![0; batch];
        let mut advances = vec![0; batch];
        for (request, &slot) in slots.iter().enumerate() {
            let lane = slot as usize;
            if lane >= batch || active_mask[lane] {
                return Err(invalid(
                    "executeTargetMatching: sequence slots must be unique and in range",
                ));
            }
            active_mask[lane] = true;
            valid_lengths[lane] = page_limits[request];
            advances[lane] = page_limits[request];
        }
        let lease_tokens = if proposal_probabilities.is_some() {
            tokens
                .iter()
                .zip(&page_limits)
                .map(|(row, &limit)| {
                    let mut expanded = Vec::with_capacity(limit as usize);
                    expanded.push(row[0]);
                    expanded.resize(limit as usize, 0);
                    expanded
                })
                .collect::<Vec<_>>()
        } else {
            tokens.clone()
        };
        let leases = self.sequence_leases(
            sequences,
            &slots,
            &active_mask,
            &valid_lengths,
            &advances,
            &lease_tokens,
        )?;
        let (input_shape, input_dtype) = self
            .inner
            .tensor_input(0)
            .ok_or_else(|| invalid("executeTargetMatching: verifier has no token input"))?;
        let elements = input_shape.iter().try_fold(1usize, |total, &dimension| {
            total
                .checked_mul(dimension)
                .ok_or_else(|| invalid("executeTargetMatching: token input size overflowed"))
        })?;
        if elements != batch * rows {
            return Err(invalid(
                "executeTargetMatching: verifier token input does not match packed rows",
            ));
        }
        let device = CudaDevice::get(leases[0].inner.pool.ordinal).map_err(failure)?;
        let proposal_probabilities = proposal_probabilities
            .map(NativeTensor::value)
            .transpose()?;
        let executable = self.inner.clone();
        let state = token
            .map(|token| token.state.clone())
            .unwrap_or_else(|| Arc::new(CancellationState::new()));
        let notify = token.map(|token| token.notify.clone());
        run_compute(state, notify, move |cancelled, _| {
            let mut decode_state =
                Self::decode_state_with_schema(&schema, &leases, &slots, &valid_lengths)?;
            Self::with_retained_device_cache(&leases, &slots, &mut decode_state, |decode_state| {
            let proposal = proposal_probabilities
                .as_ref()
                .map(|value| {
                    let shape = value.shape();
                    if shape.len() != 3
                        || shape[0] != batch
                        || shape[1] < rows.saturating_sub(1)
                    {
                        return Err(failure(
                            "executeTargetMatching: proposal probabilities have invalid geometry",
                        ));
                    }
                    let values = value.readback().map_err(failure)?;
                    if values.len() != shape[0] * shape[1] * shape[2] {
                        return Err(failure(
                            "executeTargetMatching: proposal probabilities have invalid storage",
                        ));
                    }
                    Ok((values, shape[1], shape[2]))
                })
                .transpose()?;
            let mut token_rows = tokens;
            if let Some((probabilities, proposal_rows, proposal_vocab)) = &proposal {
                for (request, &slot) in slots.iter().enumerate() {
                    let limit = page_limits[request] as usize;
                    let mut row = Vec::with_capacity(limit);
                    row.push(token_rows[request][0]);
                    for step in 0..limit.saturating_sub(1) {
                        let offset = (slot as usize * *proposal_rows + step) * *proposal_vocab;
                        row.push(
                            sample_probabilities(
                                &probabilities[offset..offset + *proposal_vocab],
                                options[step][request].seed,
                                purpose_counter(
                                    options[step][request].counter,
                                    SamplingPurpose::Proposal,
                                    0,
                                ),
                                || cancelled.is_cancelled(),
                            )
                            .map_err(invalid)?,
                        );
                    }
                    token_rows[request] = row;
                }
            }
            let mut host = vec![0.0; elements];
            for (request, &slot) in slots.iter().enumerate() {
                let start = slot as usize * rows;
                for (step, &token) in token_rows[request].iter().enumerate() {
                    host[start + step] = f64::from(token);
                }
            }
            let binding = CudaValue::from_host(device, input_shape, input_dtype, &host)
                .map_err(failure)?;
            let mut values = executable
                .execute_stateful(std::slice::from_ref(&binding), &[], decode_state, cancelled)
                .map_err(failure)?;
            let logits = values
                .first()
                .ok_or_else(|| failure("executeTargetMatching: verifier logits are missing"))?;
            let vocab = *logits
                .shape()
                .last()
                .ok_or_else(|| failure("executeTargetMatching: verifier logits are scalar"))?;
            let logits = logits.readback().map_err(failure)?;
            if vocab == 0 || logits.len() != batch * rows * vocab {
                return Err(failure(
                    "executeTargetMatching: verifier logits have invalid geometry",
                ));
            }
            let mut pages = Vec::with_capacity(slots.len());
            let mut accepted = Vec::with_capacity(slots.len());
            let mut consumed = Vec::with_capacity(slots.len());
            let mut actual_advances = vec![0; batch];
            for (request, &slot) in slots.iter().enumerate() {
                let limit = page_limits[request] as usize;
                let mut page = Vec::with_capacity(limit);
                let candidate_count = limit.saturating_sub(1);
                let mut accepted_count = 0u32;
                let mut rejected = false;
                for step in 0..candidate_count {
                    let row = slot as usize * rows + step;
                    let offset = row * vocab;
                    let candidate = token_rows[request][step + 1];
                    let sampled = if let Some((probabilities, proposal_rows, proposal_vocab)) = &proposal {
                        if *proposal_vocab != vocab {
                            return Err(failure(
                                "executeTargetMatching: target and proposal vocabularies differ",
                            ));
                        }
                        let target = effective_probabilities(
                            vocab,
                            |index| logits[offset + index],
                            options[step][request],
                            || cancelled.is_cancelled(),
                        )
                        .map_err(invalid)?;
                        let proposal_offset =
                            (slot as usize * *proposal_rows + step) * *proposal_vocab;
                        let proposal =
                            &probabilities[proposal_offset..proposal_offset + *proposal_vocab];
                        let candidate_index = candidate as usize;
                        let (&p, &q) = target
                            .get(candidate_index)
                            .zip(proposal.get(candidate_index))
                            .ok_or_else(|| {
                                invalid(
                                    "executeTargetMatching: proposal candidate is outside the vocabulary",
                                )
                            })?;
                        let accepted = if options[step][request].temperature == 0.0 {
                            p == 1.0
                        } else if !q.is_finite() || q <= 0.0 || !p.is_finite() || p < 0.0 {
                            return Err(invalid(
                                "executeTargetMatching: proposal candidate has invalid probability",
                            ));
                        } else {
                            p >= q
                                || random_unit(
                                    options[step][request].seed,
                                    purpose_counter(
                                        options[step][request].counter,
                                        SamplingPurpose::Accept,
                                        0,
                                    ),
                                ) < p / q
                        };
                        if accepted {
                            candidate
                        } else {
                            let residual = target
                                .iter()
                                .zip(proposal)
                                .map(|(&p, &q)| (p - q).max(0.0))
                                .collect::<Vec<_>>();
                            rejected = true;
                            sample_probabilities(
                                &residual,
                                options[step][request].seed,
                                purpose_counter(
                                    options[step][request].counter,
                                    SamplingPurpose::Residual,
                                    0,
                                ),
                                || cancelled.is_cancelled(),
                            )
                            .map_err(invalid)?
                        }
                    } else {
                        sample_logits(
                            vocab,
                            |index| logits[offset + index],
                            options[step][request],
                            || cancelled.is_cancelled(),
                        )
                        .map_err(invalid)?
                    };
                    page.push(sampled);
                    if proposal.is_none() && sampled != candidate {
                        rejected = true;
                    }
                    if !rejected && sampled == candidate {
                        accepted_count += 1;
                    }
                    if eos_tokens[request].contains(&sampled)
                        || rejected
                        || sampled != candidate
                    {
                        break;
                    }
                }
                if !rejected && page.len() < limit && !page.last().is_some_and(|token| {
                    eos_tokens[request].contains(token)
                }) {
                    let step = page.len();
                    let row = slot as usize * rows + step;
                    let offset = row * vocab;
                    let sampled = sample_logits(
                        vocab,
                        |index| logits[offset + index],
                        if proposal.is_some() {
                            SamplingOptions {
                                counter: purpose_counter(
                                    options[step][request].counter,
                                    SamplingPurpose::Target,
                                    0,
                                ),
                                ..options[step][request]
                            }
                        } else {
                            options[step][request]
                        },
                        || cancelled.is_cancelled(),
                    )
                    .map_err(invalid)?;
                    page.push(sampled);
                }
                actual_advances[slot as usize] = page.len() as u32;
                consumed.push(token_rows[request][..page.len()].to_vec());
                accepted.push(accepted_count);
                pages.push(page);
            }
            if slots
                .iter()
                .enumerate()
                .any(|(request, &slot)| actual_advances[slot as usize] != page_limits[request])
            {
                let mut committed = Self::decode_state_with_schema(
                    &schema,
                    &leases,
                    &slots,
                    &actual_advances,
                )?;
                committed.key_cache = decode_state.key_cache.take();
                committed.value_cache = decode_state.value_cache.take();
                committed.cache_layer_size = decode_state.cache_layer_size;
                *decode_state = committed;
                drop(values);
                values = executable
                    .execute_stateful(
                        std::slice::from_ref(&binding),
                        &[],
                        decode_state,
                        cancelled,
                    )
                    .map_err(failure)?;
            }
            let pool = leases[0].inner.pool.clone();
            let mut usage = pool.usage.lock().unwrap_or_else(|error| error.into_inner());
            let (mut planned, block_keys) = Self::block_plan(&schema, &leases, &consumed, &usage)?;
            let readback_kv = Self::should_readback_kv(&leases, &slots, &actual_advances);
            if readback_kv {
                executable.readback_state(decode_state).map_err(failure)?;
            }
            Self::commit_sequences(
                &leases,
                &slots,
                &actual_advances,
                &consumed,
                decode_state,
                readback_kv,
                block_keys,
                &mut planned,
            );
            *usage = planned;
            Ok(NativeTargetMatchingOutput {
                pages,
                accepted,
                outputs: values.into_iter().map(NativeTensor::wrap).collect(),
            })
            })
        })
        .await
    }
}

#[napi]
pub struct CudaRuntime {
    ordinal: u32,
    _device: Arc<CudaDevice>,
}

#[napi]
impl CudaRuntime {
    #[napi(constructor)]
    pub fn new(device: Option<u32>) -> Result<Self> {
        let ordinal = device.unwrap_or(0);
        let device = CudaDevice::get(ordinal).map_err(failure)?;
        Ok(Self {
            ordinal,
            _device: device,
        })
    }

    #[napi(getter)]
    pub fn device(&self) -> String {
        format!("cuda:{}", self.ordinal)
    }

    #[napi]
    pub fn constant(&self, value: f64, dtype: String) -> Result<LazyTensor> {
        lazy(Node::new(NodeKind::Full {
            shape: Vec::new(),
            value,
            dtype: parse_dtype(&dtype)?,
            device: Device::Cuda(self.ordinal),
        }))
    }

    #[napi]
    pub fn zeros(&self, dimensions: Vec<u32>, dtype: String) -> Result<LazyTensor> {
        lazy(Node::new(NodeKind::Zeros {
            shape: shape(dimensions),
            dtype: parse_dtype(&dtype)?,
            device: Device::Cuda(self.ordinal),
        }))
    }

    #[napi]
    pub fn ones(&self, dimensions: Vec<u32>, dtype: String) -> Result<LazyTensor> {
        lazy(Node::new(NodeKind::Ones {
            shape: shape(dimensions),
            dtype: parse_dtype(&dtype)?,
            device: Device::Cuda(self.ordinal),
        }))
    }

    #[napi]
    pub fn full(&self, dimensions: Vec<u32>, value: f64, dtype: String) -> Result<LazyTensor> {
        lazy(Node::new(NodeKind::Full {
            shape: shape(dimensions),
            value,
            dtype: parse_dtype(&dtype)?,
            device: Device::Cuda(self.ordinal),
        }))
    }

    #[napi(js_name = "fromBytes")]
    pub fn upload_bytes(
        &self,
        data: Uint8Array,
        dimensions: Vec<u32>,
        dtype: String,
    ) -> Result<LazyTensor> {
        lazy(Node::new(NodeKind::FromBytes {
            data: data.to_vec(),
            shape: shape(dimensions),
            dtype: parse_dtype(&dtype)?,
            device: Device::Cuda(self.ordinal),
        }))
    }

    #[napi(js_name = "uploadBytes")]
    pub fn materialize_bytes(
        &self,
        data: Uint8Array,
        dimensions: Vec<u32>,
        dtype: String,
    ) -> Result<NativeTensor> {
        let dtype = parse_dtype(&dtype)?;
        let shape = shape(dimensions);
        let value = if dtype == DType::I64 {
            if !data.len().is_multiple_of(8) {
                return Err(invalid(format!(
                    "CUDA i64 byte length {} is not divisible by 8",
                    data.len()
                )));
            }
            CudaValue::from_i64_host(
                self._device.clone(),
                shape,
                &data
                    .chunks_exact(8)
                    .map(|bytes| i64::from_le_bytes(bytes.try_into().expect("eight-byte chunk")))
                    .collect::<Vec<_>>(),
            )
        } else {
            CudaValue::from_host(
                self._device.clone(),
                shape,
                dtype,
                &crate::executable::decode(&data, dtype).map_err(failure)?,
            )
        }
        .map_err(failure)?;
        Ok(NativeTensor::wrap(value))
    }

    #[napi(js_name = "fromMaterialized")]
    pub fn materialized(&self, tensor: &NativeTensor) -> Result<LazyTensor> {
        if tensor.ordinal != self.ordinal {
            return Err(invalid(format!(
                "tensor uses CUDA device {}, expected {}",
                tensor.ordinal, self.ordinal
            )));
        }
        tensor.value()?;
        lazy(Node::new(NodeKind::Leaf(tensor.slot.clone())))
    }

    #[napi]
    pub fn graph_node(
        &self,
        operation: String,
        inputs: Vec<&LazyTensor>,
        encoded_attributes: String,
    ) -> Result<LazyTensor> {
        let expected = Device::Cuda(self.ordinal);
        if inputs.iter().any(|input| input.node.device != expected) {
            return Err(invalid(format!(
                "{operation}: every input must use CUDA device {}",
                self.ordinal
            )));
        }
        let inputs = inputs
            .into_iter()
            .map(|input| input.node.clone())
            .collect::<Vec<_>>();
        let attributes = attributes(&encoded_attributes)?;
        if operation == "vmap" {
            let result = effect_torch_autodiff::vmap(
                &input(&inputs, 0, &operation)?,
                &input(&inputs, 1, &operation)?,
                &input(&inputs, 2, &operation)?,
                integer(&attributes, "dim")?,
            );
            return lazy(result);
        }
        let unary = |kind: fn(Arc<Node>) -> NodeKind| -> Result<NodeKind> {
            Ok(kind(input(&inputs, 0, &operation)?))
        };
        let binary = |kind: fn(Arc<Node>, Arc<Node>) -> NodeKind| -> Result<NodeKind> {
            Ok(kind(
                input(&inputs, 0, &operation)?,
                input(&inputs, 1, &operation)?,
            ))
        };
        let kind = match operation.as_str() {
            "randn" => NodeKind::Randn {
                shape: dimensions(&attributes, "shape")?,
                dtype: parse_dtype(string(&attributes, "dtype")?)?,
                device: expected,
            },
            "uniform" => NodeKind::Uniform {
                shape: dimensions(&attributes, "shape")?,
                lo: number(&attributes, "lo")?,
                hi: number(&attributes, "hi")?,
                dtype: parse_dtype(string(&attributes, "dtype")?)?,
                device: expected,
            },
            "arange" => NodeKind::Arange {
                start: number(&attributes, "start")?,
                end: number(&attributes, "end")?,
                step: number(&attributes, "step")?,
                dtype: parse_dtype(string(&attributes, "dtype")?)?,
                device: expected,
            },
            "eye" => NodeKind::Eye {
                n: integer(&attributes, "n")?,
                dtype: parse_dtype(string(&attributes, "dtype")?)?,
                device: expected,
            },
            "input" => NodeKind::Input {
                slot: u32::try_from(integer(&attributes, "slot")?)
                    .map_err(|_| invalid("input slot exceeds u32"))?,
                shape: dimensions(&attributes, "shape")?,
                dtype: parse_dtype(string(&attributes, "dtype")?)?,
                device: expected,
            },
            "scalarInput" => NodeKind::ScalarInput {
                slot: u32::try_from(integer(&attributes, "slot")?)
                    .map_err(|_| invalid("scalar input slot exceeds u32"))?,
                dtype: parse_dtype(string(&attributes, "dtype")?)?,
                device: expected,
            },
            "add" => binary(|a, b| NodeKind::Add { a, b })?,
            "sub" => binary(|a, b| NodeKind::Sub { a, b })?,
            "mul" => binary(|a, b| NodeKind::Mul { a, b })?,
            "div" => binary(|a, b| NodeKind::Div { a, b })?,
            "maximum" => binary(|a, b| NodeKind::Maximum { a, b })?,
            "minimum" => binary(|a, b| NodeKind::Minimum { a, b })?,
            "eq" => binary(|a, b| NodeKind::Eq { a, b })?,
            "gt" => binary(|a, b| NodeKind::Gt { a, b })?,
            "lt" => binary(|a, b| NodeKind::Lt { a, b })?,
            "ge" => binary(|a, b| NodeKind::Ge { a, b })?,
            "le" => binary(|a, b| NodeKind::Le { a, b })?,
            "neg" => unary(|a| NodeKind::Neg { a })?,
            "abs" => unary(|a| NodeKind::Abs { a })?,
            "sqrt" => unary(|a| NodeKind::Sqrt { a })?,
            "exp" => unary(|a| NodeKind::Exp { a })?,
            "log" => unary(|a| NodeKind::Log { a })?,
            "sin" => unary(|a| NodeKind::Sin { a })?,
            "cos" => unary(|a| NodeKind::Cos { a })?,
            "tanh" => unary(|a| NodeKind::Tanh { a })?,
            "relu" => unary(|a| NodeKind::Relu { a })?,
            "erf" => unary(|a| NodeKind::Erf { a })?,
            "floor" => unary(|a| NodeKind::Floor { a })?,
            "ceil" => unary(|a| NodeKind::Ceil { a })?,
            "round" => unary(|a| NodeKind::Round { a })?,
            "sign" => unary(|a| NodeKind::Sign { a })?,
            "inverse" => unary(|a| NodeKind::Inverse { a })?,
            "det" => unary(|a| NodeKind::Det { a })?,
            "stopGradient" => unary(|a| NodeKind::StopGradient { a })?,
            "checkpoint" => unary(|a| NodeKind::Checkpoint { a })?,
            "expose" => NodeKind::Expose {
                a: input(&inputs, 0, &operation)?,
                name: string(&attributes, "name")?.to_string(),
            },
            "gelu" => NodeKind::Gelu {
                a: input(&inputs, 0, &operation)?,
                approximate: attributes
                    .get("approximate")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false),
            },
            "pow" => NodeKind::Pow {
                a: input(&inputs, 0, &operation)?,
                exp: number(&attributes, "exponent")?,
            },
            "cast" => NodeKind::Cast {
                a: input(&inputs, 0, &operation)?,
                dtype: parse_dtype(string(&attributes, "dtype")?)?,
            },
            "whereCond" => NodeKind::Where {
                cond: input(&inputs, 0, &operation)?,
                a: input(&inputs, 1, &operation)?,
                b: input(&inputs, 2, &operation)?,
            },
            "sum" | "prod" | "mean" | "max" | "min" => {
                let a = input(&inputs, 0, &operation)?;
                let dims = dimensions(&attributes, "dims")?;
                let keepdims = boolean(&attributes, "keepdims")?;
                match operation.as_str() {
                    "sum" => NodeKind::Sum { a, dims, keepdims },
                    "prod" => NodeKind::Prod { a, dims, keepdims },
                    "mean" => NodeKind::Mean { a, dims, keepdims },
                    "max" => NodeKind::Max { a, dims, keepdims },
                    "min" => NodeKind::Min { a, dims, keepdims },
                    _ => unreachable!(),
                }
            }
            "argmax" => NodeKind::Argmax {
                a: input(&inputs, 0, &operation)?,
                dim: integer(&attributes, "dim")?,
            },
            "argmin" => NodeKind::Argmin {
                a: input(&inputs, 0, &operation)?,
                dim: integer(&attributes, "dim")?,
            },
            "cumsum" => NodeKind::Cumsum {
                a: input(&inputs, 0, &operation)?,
                dim: integer(&attributes, "dim")?,
            },
            "indexSelect" => NodeKind::IndexSelect {
                a: input(&inputs, 0, &operation)?,
                indexes: input(&inputs, 1, &operation)?,
                dim: integer(&attributes, "dim")?,
            },
            "scatterAdd" => NodeKind::ScatterAdd {
                a: input(&inputs, 0, &operation)?,
                indexes: input(&inputs, 1, &operation)?,
                src: input(&inputs, 2, &operation)?,
                dim: integer(&attributes, "dim")?,
            },
            "gather" => NodeKind::Gather {
                a: input(&inputs, 0, &operation)?,
                indexes: input(&inputs, 1, &operation)?,
                dim: integer(&attributes, "dim")?,
            },
            "crossEntropy" => NodeKind::CrossEntropy {
                logits: input(&inputs, 0, &operation)?,
                target: input(&inputs, 1, &operation)?,
                ignore_index: signed_integer(&attributes, "ignoreIndex")?,
                reduction: CrossEntropyReduction::Mean,
            },
            "scaledDotProductAttention" => {
                let window = match attributes.get("window") {
                    None => AttentionWindow::Inherit,
                    Some(JsonValue::Null) => AttentionWindow::Full,
                    Some(value) => AttentionWindow::Local(
                        value
                            .as_u64()
                            .and_then(|value| usize::try_from(value).ok())
                            .ok_or_else(|| {
                                invalid("attention window must be a positive integer or null")
                            })?,
                    ),
                };
                NodeKind::Sdpa {
                    q: input(&inputs, 0, &operation)?,
                    k: input(&inputs, 1, &operation)?,
                    v: input(&inputs, 2, &operation)?,
                    scale: number(&attributes, "scale")?,
                    causal: boolean(&attributes, "causal")?,
                    window,
                }
            }
            "kdaChunk" => NodeKind::KdaChunk {
                q: input(&inputs, 0, &operation)?,
                k: input(&inputs, 1, &operation)?,
                v: input(&inputs, 2, &operation)?,
                log_decay: input(&inputs, 3, &operation)?,
                beta: input(&inputs, 4, &operation)?,
                scale: number(&attributes, "scale")?,
            },
            "shortConv1d" => NodeKind::ShortConv1d {
                x: input(&inputs, 0, &operation)?,
                weight: input(&inputs, 1, &operation)?,
            },
            "positionEmbedding" => NodeKind::PositionEmbedding {
                weight: input(&inputs, 0, &operation)?,
                seq_len: integer(&attributes, "seqLen")?,
            },
            "rotaryEmbedding" => NodeKind::RotaryEmbedding {
                x: input(&inputs, 0, &operation)?,
                seq_len: integer(&attributes, "seqLen")?,
                theta: number(&attributes, "theta")?,
                offset: PositionOffset::Absolute,
                layout: match string(&attributes, "layout")? {
                    "HalfSplit" => RotaryLayout::HalfSplit,
                    "InterleavedPairs" => RotaryLayout::InterleavedPairs,
                    layout => return Err(invalid(format!("unsupported rotary layout {layout}"))),
                },
            },
            "layerNorm" => NodeKind::LayerNorm {
                x: input(&inputs, 0, &operation)?,
                weight: input(&inputs, 1, &operation)?,
                bias: input(&inputs, 2, &operation)?,
                eps: number(&attributes, "eps")?,
            },
            "rmsNorm" => NodeKind::RmsNorm {
                x: input(&inputs, 0, &operation)?,
                weight: inputs.get(1).cloned(),
                eps: number(&attributes, "eps")?,
            },
            "linear" => NodeKind::Linear {
                x: input(&inputs, 0, &operation)?,
                weight: input(&inputs, 1, &operation)?,
                bias: input(&inputs, 2, &operation)?,
            },
            "quantizedLinear" => {
                let logical = dimensions(&attributes, "logicalShape")?;
                if logical.len() != 2 {
                    return Err(invalid(
                        "quantizedLinear: logical shape must be [rows, columns]",
                    ));
                }
                let codec = parse_kquant(string(&attributes, "encoding")?)?;
                NodeKind::QuantizedLinear {
                    x: input(&inputs, 0, &operation)?,
                    weight: input(&inputs, 1, &operation)?,
                    bias: inputs.get(2).cloned(),
                    codec,
                    weight_shape: [logical[0], logical[1]],
                }
            }
            "quantizedEmbedding" => {
                let logical = dimensions(&attributes, "logicalShape")?;
                if logical.len() != 2 {
                    return Err(invalid(
                        "quantizedEmbedding: logical shape must be [rows, columns]",
                    ));
                }
                let padding_index = match attributes.get("paddingIndex") {
                    None | Some(JsonValue::Null) => None,
                    Some(value) => Some(
                        value
                            .as_u64()
                            .and_then(|value| usize::try_from(value).ok())
                            .ok_or_else(|| {
                                invalid("quantizedEmbedding: paddingIndex must be an integer")
                            })?,
                    ),
                };
                NodeKind::QuantizedEmbedding {
                    indexes: input(&inputs, 0, &operation)?,
                    weight: input(&inputs, 1, &operation)?,
                    codec: parse_kquant(string(&attributes, "encoding")?)?,
                    weight_shape: [logical[0], logical[1]],
                    padding_index,
                }
            }
            "conv1d" => NodeKind::Conv1d {
                x: input(&inputs, 0, &operation)?,
                w: input(&inputs, 1, &operation)?,
                stride: integer(&attributes, "stride")?,
                padding: integer(&attributes, "padding")?,
                dilation: integer(&attributes, "dilation")?,
                groups: integer(&attributes, "groups")?,
            },
            "conv2d" => NodeKind::Conv2d {
                x: input(&inputs, 0, &operation)?,
                w: input(&inputs, 1, &operation)?,
                stride: integer(&attributes, "stride")?,
                padding: integer(&attributes, "padding")?,
                dilation: integer(&attributes, "dilation")?,
                groups: integer(&attributes, "groups")?,
            },
            "reshape" => NodeKind::Reshape {
                a: input(&inputs, 0, &operation)?,
                shape: dimensions(&attributes, "shape")?,
            },
            "permute" => NodeKind::Permute {
                a: input(&inputs, 0, &operation)?,
                dims: dimensions(&attributes, "dims")?,
            },
            "slice" => NodeKind::Slice {
                a: input(&inputs, 0, &operation)?,
                ranges: ranges(&attributes)?,
            },
            "concat" => NodeKind::Concat {
                a: input(&inputs, 0, &operation)?,
                b: input(&inputs, 1, &operation)?,
                dim: integer(&attributes, "dim")?,
            },
            "broadcastTo" => NodeKind::BroadcastTo {
                a: input(&inputs, 0, &operation)?,
                shape: dimensions(&attributes, "shape")?,
            },
            "matmul" => binary(|a, b| NodeKind::Matmul { a, b })?,
            "solve" => binary(|a, b| NodeKind::Solve { a, b })?,
            "adamwStep" => NodeKind::AdamWStep {
                param: input(&inputs, 0, &operation)?,
                grad: input(&inputs, 1, &operation)?,
                m: input(&inputs, 2, &operation)?,
                v: input(&inputs, 3, &operation)?,
                lr: input(&inputs, 4, &operation)?,
                c1: input(&inputs, 5, &operation)?,
                c2: input(&inputs, 6, &operation)?,
                beta1: number(&attributes, "beta1")?,
                beta2: number(&attributes, "beta2")?,
                eps: number(&attributes, "eps")?,
                weight_decay: number(&attributes, "weightDecay")?,
            },
            "adamwOut" => NodeKind::AdamWOut {
                step: input(&inputs, 0, &operation)?,
                index: u8::try_from(integer(&attributes, "index")?)
                    .map_err(|_| invalid("adamw output index exceeds u8"))?,
            },
            "sgdStep" => NodeKind::SgdStep {
                param: input(&inputs, 0, &operation)?,
                grad: input(&inputs, 1, &operation)?,
                velocity: input(&inputs, 2, &operation)?,
                first: input(&inputs, 3, &operation)?,
                lr: input(&inputs, 4, &operation)?,
                momentum: number(&attributes, "momentum")?,
                dampening: number(&attributes, "dampening")?,
                nesterov: boolean(&attributes, "nesterov")?,
                weight_decay: number(&attributes, "weightDecay")?,
            },
            "sgdOut" => NodeKind::SgdOut {
                step: input(&inputs, 0, &operation)?,
                index: u8::try_from(integer(&attributes, "index")?)
                    .map_err(|_| invalid("sgd output index exceeds u8"))?,
            },
            _ => {
                return Err(invalid(format!(
                    "unsupported CUDA graph operation {operation}"
                )))
            }
        };
        lazy(Node::new(kind))
    }

    #[napi]
    pub fn add(&self, a: &LazyTensor, b: &LazyTensor) -> Result<LazyTensor> {
        let expected = Device::Cuda(self.ordinal);
        if a.node.device != expected || b.node.device != expected {
            return Err(invalid(format!(
                "add: operands must use CUDA device {}",
                self.ordinal
            )));
        }
        lazy(Node::new(NodeKind::Add {
            a: a.node.clone(),
            b: b.node.clone(),
        }))
    }

    #[napi]
    pub fn compile(
        &self,
        roots: Vec<&LazyTensor>,
        options: Option<NativeCompileOptions>,
        state: Option<NativeKvStateSchema>,
    ) -> Result<Executable> {
        let mut roots = roots
            .iter()
            .map(|root| root.node.clone())
            .collect::<Vec<_>>();
        if roots.is_empty() {
            return Err(invalid("compile: expected at least one root"));
        }
        let state = match state {
            None => None,
            Some(native) => {
                if native.batch == 0
                    || native.max_tokens == 0
                    || native.block_size == 0
                    || !native.max_tokens.is_multiple_of(native.block_size)
                {
                    return Err(invalid(
                        "compile: batch, maxTokens, and blockSize must be positive and blockSize must divide maxTokens",
                    ));
                }
                if native
                    .window
                    .is_some_and(|window| window == 0 || window > native.max_tokens)
                {
                    return Err(invalid("compile: window must be in 1..=maxTokens"));
                }
                if native.last_token_row.is_some() && native.output_selections.is_some() {
                    return Err(invalid(
                        "compile: lastTokenRow and outputSelections are mutually exclusive",
                    ));
                }
                let batch = native.batch as usize;
                let output_selections = native.output_selections.map_or_else(
                    || {
                        vec![
                            if native.last_token_row.unwrap_or(false) {
                                DecodeOutputSelection::SplitLastTokenRow
                            } else {
                                DecodeOutputSelection::AllRows
                            };
                            roots.len()
                        ]
                    },
                    |values| values.into_iter().map(Into::into).collect(),
                );
                let packed_rows_per_sequence = native
                    .packed_causal_chains
                    .as_ref()
                    .map(|packed| packed.rows_per_sequence);
                let layout = packed_rows_per_sequence.map_or(DecodeLayout::dense(batch), |rows| {
                    DecodeLayout::packed_causal_chains(batch, rows as usize)
                });
                let (rewritten, geometry) = specialize_decode_layout_outputs_with_attention(
                    &roots,
                    native.window.map(|window| window as usize),
                    layout,
                    &output_selections,
                    native
                        .current_block_attention
                        .map(Into::into)
                        .unwrap_or_default(),
                )
                .map_err(failure)?;
                roots = rewritten;
                Some(CudaStateSchema {
                    max_tokens: native.max_tokens,
                    block_size: native.block_size,
                    kv_dtype: parse_dtype(&native.kv_dtype)?,
                    window: if geometry.allows_window_eviction {
                        native.window
                    } else {
                        None
                    },
                    batch: native.batch,
                    packed_rows_per_sequence,
                    geometry,
                })
            }
        };
        let options = compile_options(options, state.is_some());
        let compiled = match &state {
            Some(state) => compile_stateful_with_options(
                roots,
                self.ordinal,
                state.geometry.cursor_slot,
                state.geometry.cursor_tensor,
                options,
            ),
            None => compile_with_options(roots, self.ordinal, options),
        };
        compiled
            .map(|inner| Executable {
                inner: Arc::new(inner),
                state,
            })
            .map_err(failure)
    }
}

/// Reverse-mode gradients of `loss` with respect to each tensor in `wrt`.
#[napi]
pub fn grad(loss: &LazyTensor, wrt: Vec<&LazyTensor>) -> Result<Vec<LazyTensor>> {
    let targets = wrt
        .iter()
        .map(|tensor| tensor.node.clone())
        .collect::<Vec<_>>();
    effect_torch_autodiff::grad(&loss.node, &targets)
        .map(|gradients| {
            gradients
                .into_iter()
                .map(|node| LazyTensor { node })
                .collect()
        })
        .map_err(failure)
}

#[napi(object, object_from_js = false)]
pub struct NativeSafetensorsEntry {
    pub name: String,
    pub tensor: NativeTensor,
}

#[napi(object, object_from_js = false)]
pub struct NativeSafetensorsArchive {
    pub entries: Vec<NativeSafetensorsEntry>,
    pub metadata: HashMap<String, String>,
}

#[napi]
pub async fn save_tensors(
    path: String,
    names: Vec<String>,
    tensors: Vec<&NativeTensor>,
    metadata: HashMap<String, String>,
    token: Option<&CancellationToken>,
) -> Result<()> {
    if names.len() != tensors.len() || names.is_empty() {
        return Err(invalid(
            "save_tensors: names and tensors must have the same nonzero length",
        ));
    }
    let unique = names.iter().collect::<HashSet<_>>();
    if unique.len() != names.len() || names.iter().any(|name| name == "__metadata__") {
        return Err(invalid(
            "save_tensors: tensor names must be unique and cannot be __metadata__",
        ));
    }
    let tensors = tensors
        .into_iter()
        .map(NativeTensor::value)
        .collect::<Result<Vec<_>>>()?;
    let state = token
        .map(|token| token.state.clone())
        .unwrap_or_else(|| Arc::new(CancellationState::new()));
    let notify = token.map(|token| token.notify.clone());
    run_compute(state, notify, move |cancelled, _| {
        if cancelled.is_cancelled() {
            return Err(Error::new(Status::Cancelled, "operation aborted"));
        }
        let values = names.into_iter().zip(tensors).collect::<HashMap<_, _>>();
        safetensors_io::save(&values, &metadata, &path).map_err(failure)?;
        if cancelled.is_cancelled() {
            return Err(Error::new(Status::Cancelled, "operation aborted"));
        }
        Ok(())
    })
    .await
}

#[napi]
pub async fn load_tensors(
    path: String,
    device: u32,
    token: Option<&CancellationToken>,
) -> Result<NativeSafetensorsArchive> {
    let device = CudaDevice::get(device).map_err(failure)?;
    let state = token
        .map(|token| token.state.clone())
        .unwrap_or_else(|| Arc::new(CancellationState::new()));
    let notify = token.map(|token| token.notify.clone());
    run_compute(state, notify, move |cancelled, _| {
        if cancelled.is_cancelled() {
            return Err(Error::new(Status::Cancelled, "operation aborted"));
        }
        let archive = safetensors_io::load(&path, device).map_err(failure)?;
        if cancelled.is_cancelled() {
            return Err(Error::new(Status::Cancelled, "operation aborted"));
        }
        Ok(NativeSafetensorsArchive {
            entries: archive
                .entries
                .into_iter()
                .map(|(name, value)| NativeSafetensorsEntry {
                    name,
                    tensor: NativeTensor::wrap(value),
                })
                .collect(),
            metadata: archive.metadata,
        })
    })
    .await
}

use crate::device::CUDA_TOP_K_LIMIT;
use crate::executable::{compile_stateful_with_layout, CudaKvSnapshot, CudaStateLayout};
use crate::{
    compile_with_options, CudaDevice, CudaExecutable, CudaSequenceState, CudaStateInvocation,
    CudaValue,
};
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
    DType, KvLayerDescriptor, PackedFormat, SamplingOptions, SamplingPurpose, StateAccessMode,
    StorageMetadata, StorageRepresentation, ValueSpec,
};
use napi::bindgen_prelude::{Buffer, Uint32Array, Uint8Array};
use napi::{Error, Result, Status};
use napi_derive::napi;
use serde_json::Value as JsonValue;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

#[path = "napi/chain96.rs"]
mod chain96;
#[path = "napi/chain97.rs"]
mod chain97;
#[path = "napi/gguf.rs"]
mod gguf;
#[path = "napi/literal89.rs"]
mod literal89;
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

#[napi(object)]
pub struct NativeStorageMetadata {
    pub representation: String,
    pub format: Option<String>,
}

fn native_storage(representation: StorageRepresentation) -> NativeStorageMetadata {
    match representation {
        StorageRepresentation::Dense => NativeStorageMetadata {
            representation: "dense".to_string(),
            format: None,
        },
        StorageRepresentation::Packed(format) => NativeStorageMetadata {
            representation: "packed".to_string(),
            format: Some(format.name().to_string()),
        },
    }
}

fn input_storage(attributes: &JsonValue) -> Result<StorageMetadata> {
    let Some(storage) = attributes.get("storage") else {
        return Ok(StorageMetadata::dense());
    };
    let object = storage
        .as_object()
        .ok_or_else(|| invalid("input storage must be an object"))?;
    if object
        .keys()
        .any(|key| key != "representation" && key != "format")
    {
        return Err(invalid("input storage contains an unknown field"));
    }
    match string(storage, "representation")? {
        "dense" if !object.contains_key("format") => Ok(StorageMetadata::dense()),
        "packed" => {
            let PackedFormat::GgmlKQuant(codec) =
                PackedFormat::from_name(string(storage, "format")?).map_err(invalid)?;
            Ok(StorageMetadata::packed(codec))
        }
        other => Err(invalid(format!(
            "invalid input storage representation {other:?}"
        ))),
    }
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
#[derive(Clone)]
pub struct NativeKvLayerDescriptor {
    pub layer_id: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub dtype: String,
    pub retention_window: Option<u32>,
}

impl From<KvLayerDescriptor> for NativeKvLayerDescriptor {
    fn from(layer: KvLayerDescriptor) -> Self {
        Self {
            layer_id: layer.layer_id,
            kv_heads: layer.kv_heads as u32,
            head_dim: layer.head_dim as u32,
            dtype: layer.dtype.name().to_string(),
            retention_window: layer.retention.map(|value| value as u32),
        }
    }
}

fn state_access(access: Option<&str>) -> Result<StateAccessMode> {
    match access {
        None | Some("Append") | Some("append") => Ok(StateAccessMode::Append),
        Some("ReadOnly") | Some("readOnly") | Some("read-only") => Ok(StateAccessMode::ReadOnly),
        _ => Err(invalid("compile: invalid state access")),
    }
}

#[napi(object)]
pub struct NativeKvStateSchema {
    pub access: Option<String>,
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
    pub random_seed: Option<u32>,
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
pub struct NativeDTypeLegalizationDiagnostics {
    pub target_backend: String,
    pub target_architecture: String,
    pub lowering_abi_revision: f64,
    pub policy_revision: f64,
    pub capability_queries: f64,
    pub native_lowering_units: f64,
    pub legalized_lowering_units: f64,
    pub kernel_local_legalizations: f64,
    pub materialized_conversions: f64,
    pub materialized_conversion_bytes: f64,
    pub decompositions: f64,
    pub rejected_region_candidates: f64,
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
    pub legalization: NativeDTypeLegalizationDiagnostics,
    pub compile_phases: Vec<NativeCompilePhaseDiagnostics>,
}

fn executable_diagnostics(
    diagnostics: &effect_torch_runtime::ExecutableDiagnostics,
) -> NativeExecutableDiagnostics {
    let memory = &diagnostics.memory;
    let legalization = &diagnostics.legalization;
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
        legalization: NativeDTypeLegalizationDiagnostics {
            target_backend: legalization.target_backend.clone(),
            target_architecture: legalization.target_architecture.clone(),
            lowering_abi_revision: legalization.lowering_abi_revision as f64,
            policy_revision: legalization.policy_revision as f64,
            capability_queries: legalization.capability_queries as f64,
            native_lowering_units: legalization.native_lowering_units as f64,
            legalized_lowering_units: legalization.legalized_lowering_units as f64,
            kernel_local_legalizations: legalization.kernel_local_legalizations as f64,
            materialized_conversions: legalization.materialized_conversions as f64,
            materialized_conversion_bytes: legalization.materialized_conversion_bytes as f64,
            decompositions: legalization.decompositions as f64,
            rejected_region_candidates: legalization.rejected_region_candidates as f64,
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
        options.random_seed = explicit.random_seed.map(u64::from);
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

#[napi(object, object_from_js = false)]
pub struct NativeChain97Output {
    pub feedback: NativeTensor,
    pub statistics: Buffer,
}

#[derive(Clone)]
struct CudaStateSchema {
    max_tokens: u32,
    block_size: u32,
    kv_dtype: DType,
    batch: u32,
    packed_rows_per_sequence: Option<u32>,
    geometry: DecodeGeometry,
    access: StateAccessMode,
}

struct PoolInner {
    kv_layers: Vec<KvLayerDescriptor>,
    explicit_layers: bool,
    ordinal: u32,
    max_tokens: u32,
    block_size: u32,
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
    kv_storage: Option<CudaKvSnapshot>,
    kda_states: Vec<Vec<f32>>,
    conv_states: Vec<Vec<f32>>,
    block_keys: Vec<BlockKey>,
}

struct SequenceInner {
    pool: Arc<PoolInner>,
    state: Mutex<SequenceState>,
    released: AtomicBool,
    running: AtomicBool,
}

struct SequenceLease {
    inner: Arc<SequenceInner>,
}

impl SequenceInner {
    fn release_if_idle(&self) {
        if self.released.load(Ordering::Acquire)
            && self
                .running
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        {
            self.clear_released();
            self.running.store(false, Ordering::Release);
        }
    }

    fn clear_released(&self) {
        let mut usage = self
            .pool
            .usage
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        for key in state.block_keys.drain(..) {
            if let Some(count) = usage.blocks.get_mut(&key) {
                *count = count.saturating_sub(1);
            }
        }
        *state = SequenceState::default();
    }
}

impl Drop for SequenceInner {
    fn drop(&mut self) {
        self.clear_released();
    }
}

impl Drop for SequenceLease {
    fn drop(&mut self) {
        self.inner.running.store(false, Ordering::Release);
        self.inner.release_if_idle();
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
        kv_layers: Option<Vec<NativeKvLayerDescriptor>>,
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
        if kv_layers.is_none() && (layers == 0) != (kv_heads == 0 || head_dim == 0) {
            return Err(invalid(
                "kv pool: attention geometry must be entirely zero or positive",
            ));
        }
        let explicit_layers = kv_layers.is_some();
        let mut kv_layers = match kv_layers {
            Some(layers) => layers
                .into_iter()
                .map(|layer| {
                    Ok(KvLayerDescriptor {
                        layer_id: layer.layer_id,
                        kv_heads: layer.kv_heads as usize,
                        head_dim: layer.head_dim as usize,
                        dtype: parse_dtype(&layer.dtype)?,
                        retention: layer.retention_window.map(|n| n as usize),
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            None => (0..layers)
                .map(|layer_id| KvLayerDescriptor {
                    layer_id,
                    kv_heads: kv_heads as usize,
                    head_dim: head_dim as usize,
                    dtype,
                    retention: None,
                })
                .collect(),
        };
        kv_layers.sort_by_key(|layer| layer.layer_id);
        for (id, layer) in kv_layers.iter().enumerate() {
            if layer.layer_id as usize != id
                || layer.kv_heads == 0
                || layer.head_dim == 0
                || !matches!(
                    layer.dtype,
                    DType::F32 | DType::F16 | DType::BF16 | DType::U8
                )
                || layer.row_bytes().is_none()
            {
                return Err(invalid("kv pool: invalid layer descriptor"));
            }
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
                kv_layers,
                explicit_layers,
                max_tokens,
                block_size,
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
            .values()
            .filter(|&&references| references > 0)
            .count() as u32;
        (self.inner.max_tokens / self.inner.block_size).saturating_sub(used)
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
                released: AtomicBool::new(false),
                running: AtomicBool::new(false),
            }),
        }
    }
}

#[napi(object, object_from_js = false)]
pub struct NativeKvLayerSnapshot {
    pub layer_id: u32,
    pub start_position: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub dtype: String,
    pub keys: Vec<f64>,
    pub values: Vec<f64>,
}

#[napi(object, object_from_js = false)]
pub struct NativeKvSnapshotInspection {
    pub cursor: u32,
    pub retained_bytes: f64,
    pub shared_bytes: f64,
    pub copied_bytes: f64,
    pub layers: Vec<NativeKvLayerSnapshot>,
}

struct PrefixData {
    pool: Arc<PoolInner>,
    state: SequenceState,
}

fn retain_block_keys(pool: &PoolInner, keys: &[BlockKey]) -> Result<()> {
    let mut usage = pool.usage.lock().unwrap_or_else(|error| error.into_inner());
    if keys
        .iter()
        .any(|key| usage.blocks.get(key).is_none_or(|count| *count == u32::MAX))
    {
        return Err(failure(
            "kv prefix references a missing or saturated pool block",
        ));
    }
    for key in keys {
        *usage.blocks.get_mut(key).expect("block reference checked") += 1;
    }
    Ok(())
}

impl Drop for PrefixData {
    fn drop(&mut self) {
        let mut usage = self
            .pool
            .usage
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        for key in &self.state.block_keys {
            if let Some(count) = usage.blocks.get_mut(key) {
                *count = count.saturating_sub(1);
            }
        }
    }
}

impl PrefixData {
    fn bytes(&self, shared_only: bool) -> usize {
        let mut allocations = HashMap::new();
        if let Some(storage) = &self.state.kv_storage {
            for layer in &storage.layers {
                for page in &layer.pages {
                    if shared_only && Arc::strong_count(page) <= 1 {
                        continue;
                    }
                    for (identity, bytes) in [
                        Some(page.keys.allocation()),
                        Some(page.values.allocation()),
                        page.key_scales.as_ref().map(|buffer| buffer.allocation()),
                        page.value_scales.as_ref().map(|buffer| buffer.allocation()),
                    ]
                    .into_iter()
                    .flatten()
                    {
                        allocations.insert(identity, bytes);
                    }
                }
            }
        }
        allocations.values().sum()
    }
}

#[napi]
pub struct NativeKvPrefix {
    inner: Mutex<Option<Arc<PrefixData>>>,
}

impl NativeKvPrefix {
    fn borrowed(&self) -> Result<Arc<PrefixData>> {
        self.inner
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
            .ok_or_else(|| invalid("kv prefix was released"))
    }
}

#[napi]
impl NativeKvPrefix {
    #[napi]
    pub fn release(&self) {
        self.inner
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
    }

    #[napi(getter)]
    pub fn cursor(&self) -> Result<u32> {
        Ok(self.borrowed()?.state.cursor)
    }

    #[napi(getter)]
    pub fn retained_bytes(&self) -> Result<f64> {
        Ok(self.borrowed()?.bytes(false) as f64)
    }

    #[napi(getter)]
    pub fn shared_bytes(&self) -> Result<f64> {
        Ok(self.borrowed()?.bytes(true) as f64)
    }

    #[napi(getter)]
    pub fn copied_bytes(&self) -> Result<f64> {
        self.borrowed()?;
        Ok(0.0)
    }

    #[napi]
    pub fn fork(&self) -> Result<NativeKvSequence> {
        let prefix = self.borrowed()?;
        retain_block_keys(&prefix.pool, &prefix.state.block_keys)?;
        Ok(NativeKvSequence {
            inner: Arc::new(SequenceInner {
                pool: prefix.pool.clone(),
                state: Mutex::new(prefix.state.clone()),
                released: AtomicBool::new(false),
                running: AtomicBool::new(false),
            }),
        })
    }

    #[napi]
    pub fn inspect(&self) -> Result<NativeKvSnapshotInspection> {
        let prefix = self.borrowed()?;
        let mut layers = Vec::with_capacity(prefix.pool.kv_layers.len());
        for descriptor in &prefix.pool.kv_layers {
            let storage = prefix.state.kv_storage.as_ref().and_then(|storage| {
                storage
                    .layers
                    .iter()
                    .find(|layer| layer.descriptor.layer_id == descriptor.layer_id)
            });
            let start = storage.map_or(prefix.state.cursor, |layer| layer.start_position);
            let mut keys = Vec::new();
            let mut values = Vec::new();
            if let Some(layer) = storage {
                let device = if layer.pages.is_empty() {
                    None
                } else {
                    Some(CudaDevice::get(prefix.pool.ordinal).map_err(failure)?)
                };
                let mut expected = start;
                for page in &layer.pages {
                    let begin = page.start.max(start);
                    let end = page
                        .start
                        .checked_add(page.count)
                        .ok_or_else(|| failure("kv inspect: page position overflow"))?
                        .min(prefix.state.cursor);
                    if begin >= end {
                        continue;
                    }
                    if begin != expected {
                        return Err(failure("kv inspect: missing or overlapping prefix rows"));
                    }
                    let device = device.as_ref().expect("nonempty pages have a device");
                    let export = |buffer: &crate::buffer::CudaBuffer<u8>,
                                  scales: Option<&crate::buffer::CudaBuffer<f32>>|
                     -> Result<Vec<f64>> {
                        let bytes = device
                            .stream
                            .clone_dtoh(buffer)
                            .map_err(|error| failure(error.to_string()))?;
                        let scales = scales
                            .map(|values| {
                                device
                                    .stream
                                    .clone_dtoh(values)
                                    .map_err(|error| failure(error.to_string()))
                            })
                            .transpose()?;
                        let row_width = descriptor.kv_heads * descriptor.head_dim;
                        let mut result = Vec::with_capacity((end - begin) as usize * row_width);
                        for token in begin..end {
                            for head in 0..descriptor.kv_heads {
                                for column in 0..descriptor.head_dim {
                                    let row = (token - page.start) as usize;
                                    let element = (row * descriptor.kv_heads + head)
                                        * descriptor.head_dim
                                        + column;
                                    let offset = element * descriptor.dtype.size_in_bytes();
                                    let raw = bytes
                                        .get(offset..offset + descriptor.dtype.size_in_bytes())
                                        .ok_or_else(|| {
                                            failure("kv inspect: truncated page storage")
                                        })?;
                                    let value = match descriptor.dtype {
                                        DType::F32 => {
                                            f32::from_le_bytes(raw.try_into().expect("f32 bytes"))
                                                as f64
                                        }
                                        DType::F16 => half::f16::from_bits(u16::from_le_bytes(
                                            raw.try_into().expect("f16 bytes"),
                                        ))
                                        .to_f64(),
                                        DType::BF16 => half::bf16::from_bits(u16::from_le_bytes(
                                            raw.try_into().expect("bf16 bytes"),
                                        ))
                                        .to_f64(),
                                        DType::U8 => {
                                            let scale = scales
                                                .as_ref()
                                                .and_then(|values| {
                                                    values.get(row * descriptor.kv_heads + head)
                                                })
                                                .ok_or_else(|| {
                                                    failure(
                                                        "kv inspect: missing quantization scale",
                                                    )
                                                })?;
                                            ((raw[0] as f32 - 128.0) * scale) as f64
                                        }
                                        _ => {
                                            return Err(failure(
                                                "kv inspect: unsupported storage dtype",
                                            ));
                                        }
                                    };
                                    result.push(value);
                                }
                            }
                        }
                        Ok(result)
                    };
                    keys.extend(export(&page.keys, page.key_scales.as_ref())?);
                    values.extend(export(&page.values, page.value_scales.as_ref())?);
                    expected = end;
                }
                if expected != prefix.state.cursor {
                    return Err(failure("kv inspect: prefix is missing its tail"));
                }
            }
            layers.push(NativeKvLayerSnapshot {
                layer_id: descriptor.layer_id,
                start_position: start,
                kv_heads: descriptor.kv_heads as u32,
                head_dim: descriptor.head_dim as u32,
                dtype: descriptor.dtype.name().to_string(),
                keys,
                values,
            });
        }
        Ok(NativeKvSnapshotInspection {
            cursor: prefix.state.cursor,
            retained_bytes: prefix.bytes(false) as f64,
            shared_bytes: prefix.bytes(true) as f64,
            copied_bytes: 0.0,
            layers,
        })
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
        if self.inner.released.load(Ordering::Acquire) {
            self.inner.clear_released();
            self.inner.running.store(false, Ordering::Release);
            return Err(invalid("kv sequence was released"));
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
    pub fn snapshot(&self) -> Result<NativeKvPrefix> {
        let _lease = self.lease()?;
        let state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone();
        retain_block_keys(&self.inner.pool, &state.block_keys)?;
        Ok(NativeKvPrefix {
            inner: Mutex::new(Some(Arc::new(PrefixData {
                pool: self.inner.pool.clone(),
                state,
            }))),
        })
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
                released: AtomicBool::new(false),
                running: AtomicBool::new(false),
            }),
        })
    }

    #[napi]
    pub fn release(&self) {
        self.inner.released.store(true, Ordering::Release);
        self.inner.release_if_idle();
    }

    #[napi]
    pub fn prefill_match(&self, tokens: Vec<u32>) -> Result<u32> {
        let _lease = self.lease()?;
        let pool = &self.inner.pool;
        let recurrent = pool.recurrent.kda_layers > 0 || pool.recurrent.conv_layers > 0;
        if recurrent && pool.kv_layers.is_empty() {
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
    pub fn storage(&self) -> NativeStorageMetadata {
        native_storage(self.node.value_spec().storage.representation)
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

#[derive(Default)]
struct ExportedTensorMemory {
    allocations: HashMap<usize, (usize, usize)>,
    bytes: usize,
}

static EXPORTED_TENSOR_MEMORY: LazyLock<Mutex<ExportedTensorMemory>> =
    LazyLock::new(|| Mutex::new(ExportedTensorMemory::default()));

struct ExportedTensorAccounting {
    allocation: usize,
    active: AtomicBool,
}

impl ExportedTensorAccounting {
    fn new((allocation, bytes): (usize, usize)) -> Self {
        let mut memory = EXPORTED_TENSOR_MEMORY
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some((_, references)) = memory.allocations.get_mut(&allocation) {
            *references += 1;
        } else {
            memory.allocations.insert(allocation, (bytes, 1));
            memory.bytes += bytes;
        }
        Self {
            allocation,
            active: AtomicBool::new(true),
        }
    }

    fn release(&self) {
        if !self.active.swap(false, Ordering::AcqRel) {
            return;
        }
        let mut memory = EXPORTED_TENSOR_MEMORY
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let Some((bytes, references)) = memory.allocations.get_mut(&self.allocation) else {
            return;
        };
        *references -= 1;
        if *references == 0 {
            let bytes = *bytes;
            memory.allocations.remove(&self.allocation);
            memory.bytes -= bytes;
        }
    }
}

impl Drop for ExportedTensorAccounting {
    fn drop(&mut self) {
        self.release();
    }
}

/// Backing bytes retained by live exported tensor slots, deduplicated across aliases.
#[napi]
pub fn external_memory_bytes() -> f64 {
    EXPORTED_TENSOR_MEMORY
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .bytes as f64
}

#[derive(Clone)]
#[napi]
pub struct NativeTensor {
    accounting: Arc<ExportedTensorAccounting>,
    slot: Arc<LeafSlot>,
    ordinal: u32,
}

impl NativeTensor {
    fn wrap(value: CudaValue) -> Self {
        let ordinal = value.ordinal();
        let accounting = Arc::new(ExportedTensorAccounting::new(value.allocation()));
        Self {
            accounting,
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
        // Remove the accounting entry before the allocation can be freed and its identity reused.
        self.accounting.release();
        self.slot.clear();
    }

    /// Returns an independently clearable slot retaining the same device allocation.
    #[napi]
    pub fn retain(&self) -> Result<NativeTensor> {
        Ok(Self::wrap(self.value()?))
    }

    #[napi(js_name = "writeBytes")]
    pub fn write_bytes(&self, data: Uint8Array) -> Result<()> {
        self.value()?.write_storage_bytes(&data).map_err(failure)
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
    pub fn storage(&self) -> Result<NativeStorageMetadata> {
        Ok(native_storage(self.value()?.spec().storage.representation))
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
            if value.spec().storage.representation != StorageRepresentation::Dense {
                return Err(invalid(
                    "packed tensors require explicit dequantization before readback",
                ));
            }
            let bytes = if matches!(value.dtype(), DType::F16 | DType::BF16) {
                value
                    .readback()
                    .map_err(failure)?
                    .into_iter()
                    .flat_map(|value| (value as f32).to_le_bytes())
                    .collect()
            } else {
                value.read_storage_bytes().map_err(failure)?
            };
            if cancelled.is_cancelled() {
                return Err(Error::new(Status::Cancelled, "operation aborted"));
            }
            Ok(Buffer::from(bytes))
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
    request_rng99: Option<Arc<crate::executable::RequestRng99>>,
}

impl PoolInner {
    fn matches_layers(&self, layers: &[KvLayerDescriptor]) -> bool {
        self.kv_layers.len() == layers.len()
            && self.kv_layers.iter().zip(layers).all(|(pool, layer)| {
                let mut expected = *layer;
                if !self.explicit_layers {
                    expected.retention = pool.retention;
                }
                *pool == expected
            })
    }
}

impl CudaStateSchema {
    fn retention_start(&self, cursor: u32) -> u32 {
        self.geometry
            .kv_layers
            .iter()
            .map(|layer| {
                layer
                    .retention
                    .map_or(0, |retention| cursor.saturating_sub(retention as u32))
            })
            .min()
            .unwrap_or(0)
    }

    fn validate_pool(&self, pool: &PoolInner, ordinal: u32) -> Result<()> {
        let recurrent = &pool.recurrent;
        if pool.ordinal != ordinal
            || !pool.matches_layers(&self.geometry.kv_layers)
            || pool.max_tokens != self.max_tokens
            || pool.block_size != self.block_size
            || recurrent.kda_layers != self.geometry.kda.layers as u32
            || recurrent.kda_heads != self.geometry.kda.heads as u32
            || recurrent.kda_head_dim != self.geometry.kda.head_dim as u32
            || recurrent.kda_value_dim != self.geometry.kda.value_dim as u32
            || recurrent.conv_layers != self.geometry.conv.layers as u32
            || recurrent.conv_channels != self.geometry.conv.channels as u32
            || recurrent.conv_kernel != self.geometry.conv.kernel as u32
        {
            return Err(invalid(
                "execute: pool geometry does not match executable state",
            ));
        }
        Ok(())
    }
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
        if schema.access != StateAccessMode::Append {
            return Err(invalid("execute: ReadOnly state requires executeReadOnly"));
        }
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
            schema.validate_pool(pool, self.inner.ordinal())?;
            let cursor = sequence
                .inner
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .cursor;
            if cursor.checked_add(advances[slot]).is_none()
                || cursor
                    .saturating_add(advances[slot])
                    .saturating_sub(schema.retention_start(cursor))
                    > schema.max_tokens
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
            let previous_start = schema.retention_start(state.cursor) as usize / block_size;
            let unchanged = state.tokens.len() / block_size;
            let blocks = combined.len().div_ceil(block_size);
            let start = schema.retention_start(combined.len() as u32) as usize / block_size;
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
        _executable: &CudaExecutable,
        schema: &CudaStateSchema,
        leases: &[SequenceLease],
        slots: &[u32],
        valid_lengths: &[u32],
    ) -> Result<CudaStateInvocation> {
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
                        keys: Vec::new(),
                        values: Vec::new(),
                        kv_storage: state.kv_storage.clone(),
                        kda_states: state.kda_states.clone(),
                        conv_states: state.conv_states.clone(),
                    }
                })
                .collect(),
            slots: slots.to_vec(),
            valid_lengths: valid_lengths.to_vec(),
            capacity: schema.max_tokens,
            cache_dtype: schema.kv_dtype,
            packed_rows_per_sequence: schema.packed_rows_per_sequence,
            kv_layers: schema.geometry.kv_layers.clone(),
            access: schema.access,
            cache: None,
        })
    }

    fn with_state_transaction<A>(
        leases: &[SequenceLease],
        _slots: &[u32],
        decoded: &mut CudaStateInvocation,
        execute: impl FnOnce(&mut CudaStateInvocation) -> Result<A>,
    ) -> Result<A> {
        let checkpoints = leases
            .iter()
            .map(|lease| {
                lease
                    .inner
                    .state
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .clone()
            })
            .collect::<Vec<_>>();
        let result = execute(decoded);
        if result.is_err() {
            let mut usage = leases[0]
                .inner
                .pool
                .usage
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            for (lease, previous) in leases.iter().zip(checkpoints) {
                let mut current = lease
                    .inner
                    .state
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                for key in &current.block_keys {
                    if let Some(references) = usage.blocks.get_mut(key) {
                        *references = references.saturating_sub(1);
                    }
                }
                for key in &previous.block_keys {
                    *usage.blocks.entry(key.clone()).or_default() += 1;
                }
                *current = previous;
            }
        }
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
                state.kv_storage = decoded.sequences[request].kv_storage.clone();
            }
            state.kda_states = decoded.sequences[request].kda_states.clone();
            state.conv_states = decoded.sequences[request].conv_states.clone();
            state.block_keys = block_keys[request].clone();
            if (commit_kv || lease.inner.pool.kv_layers.is_empty())
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
                state.kv_storage = decoded.sequences[request].kv_storage.clone();
            }
            state.kda_states = decoded.sequences[request].kda_states.clone();
            state.conv_states = decoded.sequences[request].conv_states.clone();
            if (commit_kv || lease.inner.pool.kv_layers.is_empty())
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
    pub fn kv_layers(&self) -> Vec<NativeKvLayerDescriptor> {
        self.state.as_ref().map_or_else(Vec::new, |state| {
            state
                .geometry
                .kv_layers
                .iter()
                .copied()
                .map(Into::into)
                .collect()
        })
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

    /// Private diagnostic fork: two immutable templates, one request RNG stream.
    #[napi]
    pub fn fork_request_rng99(&self, peer: &Executable, seed: u32) -> Result<Vec<Executable>> {
        if self.request_rng99.is_some()
            || peer.request_rng99.is_some()
            || self.inner.ordinal() != peer.inner.ordinal()
            || !crate::executable::RequestRng99::graphs_disabled()
            || [self, peer].iter().any(|program| {
                !program.inner.request_rng99_admitted()
                    || !program
                        .state
                        .as_ref()
                        .is_some_and(|state| state.access == StateAccessMode::ReadOnly)
            })
        {
            return Err(invalid(
                "forkRequestRng99: unforked read-only seed0 single-source same-device templates and graphs disabled required",
            ));
        }
        let rng = Arc::new(crate::executable::RequestRng99::new(seed));
        Ok([self, peer]
            .into_iter()
            .map(|program| Executable {
                inner: program.inner.clone(),
                state: program.state.clone(),
                request_rng99: Some(rng.clone()),
            })
            .collect())
    }

    #[napi]
    pub async fn execute_read_only(
        &self,
        bindings: Vec<&NativeTensor>,
        prefixes: Vec<&NativeKvPrefix>,
        slots: Vec<u32>,
        active_mask: Vec<bool>,
        valid_lengths: Vec<u32>,
        token: Option<&CancellationToken>,
    ) -> Result<Vec<NativeTensor>> {
        let schema = self
            .state
            .as_ref()
            .ok_or_else(|| invalid("executeReadOnly: state schema required"))?;
        if schema.access != StateAccessMode::ReadOnly
            || schema.geometry.kda.layers != 0
            || schema.geometry.conv.layers != 0
            || schema.packed_rows_per_sequence.is_some()
        {
            return Err(invalid(
                "executeReadOnly: requires a dense read-only KV executable",
            ));
        }
        let batch = schema.batch as usize;
        if prefixes.is_empty()
            || prefixes.len() > batch
            || slots.len() != prefixes.len()
            || active_mask.len() != batch
            || valid_lengths.len() != batch
        {
            return Err(invalid("executeReadOnly: invalid fixed-lane metadata"));
        }
        let prefixes = prefixes
            .iter()
            .map(|prefix| prefix.borrowed())
            .collect::<Result<Vec<_>>>()?;
        let mut seen = vec![false; batch];
        for (prefix, &slot) in prefixes.iter().zip(&slots) {
            let slot = slot as usize;
            if slot >= batch || seen[slot] || valid_lengths[slot] == 0 {
                return Err(invalid("executeReadOnly: invalid or duplicate active slot"));
            }
            seen[slot] = true;
            if !Arc::ptr_eq(&prefixes[0].pool, &prefix.pool) {
                return Err(invalid("executeReadOnly: prefixes must share a pool"));
            }
            schema.validate_pool(&prefix.pool, self.inner.ordinal())?;
        }
        if (0..batch).any(|slot| {
            active_mask[slot] != seen[slot] || (!seen[slot] && valid_lengths[slot] != 0)
        }) {
            return Err(invalid(
                "executeReadOnly: inconsistent inactive lane metadata",
            ));
        }
        let bindings = bindings
            .into_iter()
            .map(NativeTensor::value)
            .collect::<Result<Vec<_>>>()?;
        let schema = schema.clone();
        let executable = self.inner.clone();
        let request_rng99 = self.request_rng99.clone();
        let state = token
            .map(|token| token.state.clone())
            .unwrap_or_else(|| Arc::new(CancellationState::new()));
        let notify = token.map(|token| token.notify.clone());
        run_compute(state, notify, move |cancelled, _| {
            let mut invocation = CudaStateInvocation {
                sequences: prefixes
                    .iter()
                    .map(|prefix| CudaSequenceState {
                        cursor: prefix.state.cursor,
                        keys: Vec::new(),
                        values: Vec::new(),
                        kda_states: Vec::new(),
                        conv_states: Vec::new(),
                        kv_storage: prefix.state.kv_storage.clone(),
                    })
                    .collect(),
                slots,
                valid_lengths,
                capacity: schema.max_tokens,
                cache_dtype: schema.kv_dtype,
                packed_rows_per_sequence: None,
                kv_layers: schema.geometry.kv_layers,
                access: StateAccessMode::ReadOnly,
                cache: None,
            };
            match request_rng99.as_deref() {
                Some(rng) => executable.execute_stateful_request99(
                    &bindings,
                    &mut invocation,
                    cancelled,
                    rng,
                ),
                None => executable.execute_stateful(&bindings, &[], &mut invocation, cancelled),
            }
            .map(|outputs| outputs.into_iter().map(NativeTensor::wrap).collect())
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
    pub async fn execute_chain96(
        &self,
        head: Option<&Executable>,
        sampler: &Executable,
        bindings: Vec<&NativeTensor>,
        prefixes: Vec<&NativeKvPrefix>,
        slots: Vec<u32>,
        active_mask: Vec<bool>,
        valid_lengths: Vec<u32>,
        temperature: f64,
        token: Option<&CancellationToken>,
    ) -> Result<Vec<NativeTensor>> {
        if self.request_rng99.is_some()
            || sampler.request_rng99.is_some()
            || head.is_some_and(|head| head.request_rng99.is_some())
        {
            return Err(invalid("requestRng99 forks must use executeReadOnly"));
        }
        if head.is_some_and(|head| {
            head.state.is_some() || head.inner.ordinal() != self.inner.ordinal()
        }) || sampler.state.is_some()
            || sampler.inner.ordinal() != self.inner.ordinal()
            || !temperature.is_finite()
        {
            return Err(invalid(
                "executeChain96: stateless same-device successors and finite temperature required",
            ));
        }
        if !self.inner.chain96_dense_outputs()
            || head.is_some_and(|head| !head.inner.chain96_successor(&[]))
            || !sampler.inner.chain96_successor(&[DType::F32])
        {
            return Err(invalid(
                "executeChain96: dense successors require zero head scalars and one F32 sampler scalar",
            ));
        }
        let body_outputs = self.inner.outputs();
        let sampler_input = head.map_or(body_outputs, |head| head.inner.outputs());
        if body_outputs.len() != 1
            || sampler_input.len() != 1
            || sampler.inner.outputs().is_empty()
            || sampler.inner.outputs().len() > 8
            || head.is_some_and(|head| {
                head.inner.tensor_input(0).as_ref() != body_outputs.first()
                    || head.inner.tensor_input(1).is_some()
            })
            || sampler.inner.tensor_input(0).as_ref() != sampler_input.first()
            || sampler.inner.tensor_input(1).is_some()
        {
            return Err(invalid(
                "executeChain96: incompatible bounded one-output stage metadata",
            ));
        }
        let head = head.map(|head| head.inner.clone());
        let sampler = sampler.inner.clone();
        let schema = self
            .state
            .as_ref()
            .ok_or_else(|| invalid("executeChain96: state schema required"))?;
        if schema.access != StateAccessMode::ReadOnly
            || schema.geometry.kda.layers != 0
            || schema.geometry.conv.layers != 0
            || schema.packed_rows_per_sequence.is_some()
        {
            return Err(invalid(
                "executeChain96: requires a dense read-only KV executable",
            ));
        }
        let batch = schema.batch as usize;
        if prefixes.is_empty()
            || prefixes.len() > batch
            || slots.len() != prefixes.len()
            || active_mask.len() != batch
            || valid_lengths.len() != batch
        {
            return Err(invalid("executeChain96: invalid fixed-lane metadata"));
        }
        let prefixes = prefixes
            .iter()
            .map(|prefix| prefix.borrowed())
            .collect::<Result<Vec<_>>>()?;
        let mut seen = vec![false; batch];
        for (prefix, &slot) in prefixes.iter().zip(&slots) {
            let slot = slot as usize;
            if slot >= batch || seen[slot] || valid_lengths[slot] == 0 {
                return Err(invalid("executeChain96: invalid or duplicate active slot"));
            }
            seen[slot] = true;
            if !Arc::ptr_eq(&prefixes[0].pool, &prefix.pool) {
                return Err(invalid("executeChain96: prefixes must share a pool"));
            }
            schema.validate_pool(&prefix.pool, self.inner.ordinal())?;
        }
        if (0..batch).any(|slot| {
            active_mask[slot] != seen[slot] || (!seen[slot] && valid_lengths[slot] != 0)
        }) {
            return Err(invalid(
                "executeChain96: inconsistent inactive lane metadata",
            ));
        }
        let bindings = bindings
            .into_iter()
            .map(NativeTensor::value)
            .collect::<Result<Vec<_>>>()?;
        let schema = schema.clone();
        let executable = self.inner.clone();
        let state = token
            .map(|token| token.state.clone())
            .unwrap_or_else(|| Arc::new(CancellationState::new()));
        let notify = token.map(|token| token.notify.clone());
        run_compute(state, notify, move |cancelled, _| {
            let mut invocation = CudaStateInvocation {
                sequences: prefixes
                    .iter()
                    .map(|prefix| CudaSequenceState {
                        cursor: prefix.state.cursor,
                        keys: Vec::new(),
                        values: Vec::new(),
                        kda_states: Vec::new(),
                        conv_states: Vec::new(),
                        kv_storage: prefix.state.kv_storage.clone(),
                    })
                    .collect(),
                slots,
                valid_lengths,
                capacity: schema.max_tokens,
                cache_dtype: schema.kv_dtype,
                packed_rows_per_sequence: None,
                kv_layers: schema.geometry.kv_layers,
                access: StateAccessMode::ReadOnly,
                cache: None,
            };
            chain96::execute(
                || executable.execute_stateful(&bindings, &[], &mut invocation, cancelled),
                head.as_deref(),
                &sampler,
                temperature,
                cancelled,
            )
            .map(|outputs| outputs.into_iter().map(NativeTensor::wrap).collect())
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

    /// Pure immutable admission for the host-input / processed-output pipeline.
    #[napi]
    pub fn supports_chain97(
        &self,
        head: Option<&Executable>,
        sampler: &Executable,
        width: u32,
    ) -> bool {
        let width = width as usize;
        let Some(schema) = self.state.as_ref() else {
            return false;
        };
        let body_outputs = self.inner.outputs();
        let sampler_input = head.map_or(body_outputs, |head| head.inner.outputs());
        self.request_rng99.is_none()
            && sampler.request_rng99.is_none()
            && head.is_none_or(|head| head.request_rng99.is_none())
            && schema.access == StateAccessMode::ReadOnly
            && schema.batch == 1
            && schema.geometry.kda.layers == 0
            && schema.geometry.conv.layers == 0
            && schema.packed_rows_per_sequence.is_none()
            && (1..=256).contains(&width)
            && self.inner.tensor_input(0) == Some((vec![1, width], DType::U32))
            && self.inner.chain96_dense_outputs()
            && body_outputs.len() == 1
            && sampler_input.len() == 1
            && head.is_none_or(|head| {
                head.state.is_none()
                    && head.inner.ordinal() == self.inner.ordinal()
                    && head.inner.chain96_successor(&[])
                    && head.inner.tensor_input(0).as_ref() == body_outputs.first()
                    && head.inner.tensor_input(1).is_none()
            })
            && sampler.state.is_none()
            && sampler.inner.ordinal() == self.inner.ordinal()
            && sampler.inner.chain96_successor(&[DType::F32])
            && sampler.inner.tensor_input(0).as_ref() == sampler_input.first()
            && sampler.inner.tensor_input(1).is_none()
            && sampler.inner.outputs().len() == 2
            && sampler.inner.outputs()[1] == (vec![width * 4 + 1], DType::F32)
    }

    #[napi]
    pub async fn execute_chain97(
        &self,
        head: Option<&Executable>,
        sampler: &Executable,
        canvas: Uint32Array,
        bindings: Vec<&NativeTensor>,
        prefixes: Vec<&NativeKvPrefix>,
        slots: Vec<u32>,
        active_mask: Vec<bool>,
        valid_lengths: Vec<u32>,
        temperature: f64,
        token: Option<&CancellationToken>,
    ) -> Result<NativeChain97Output> {
        if self.request_rng99.is_some()
            || sampler.request_rng99.is_some()
            || head.is_some_and(|head| head.request_rng99.is_some())
        {
            return Err(invalid("requestRng99 forks must use executeReadOnly"));
        }
        if head.is_some_and(|head| {
            head.state.is_some() || head.inner.ordinal() != self.inner.ordinal()
        }) || sampler.state.is_some()
            || sampler.inner.ordinal() != self.inner.ordinal()
            || !temperature.is_finite()
        {
            return Err(invalid(
                "executeChain97: stateless same-device successors and finite temperature required",
            ));
        }
        if !self.inner.chain96_dense_outputs()
            || head.is_some_and(|head| !head.inner.chain96_successor(&[]))
            || !sampler.inner.chain96_successor(&[DType::F32])
        {
            return Err(invalid(
                "executeChain97: dense successors require zero head scalars and one F32 sampler scalar",
            ));
        }
        let body_outputs = self.inner.outputs();
        let sampler_input = head.map_or(body_outputs, |head| head.inner.outputs());
        if body_outputs.len() != 1
            || sampler_input.len() != 1
            || sampler.inner.outputs().is_empty()
            || sampler.inner.outputs().len() > 8
            || head.is_some_and(|head| {
                head.inner.tensor_input(0).as_ref() != body_outputs.first()
                    || head.inner.tensor_input(1).is_some()
            })
            || sampler.inner.tensor_input(0).as_ref() != sampler_input.first()
            || sampler.inner.tensor_input(1).is_some()
        {
            return Err(invalid(
                "executeChain97: incompatible bounded one-output stage metadata",
            ));
        }
        let canvas = canvas.to_vec();
        if canvas.is_empty()
            || canvas.len() > 256
            || self.inner.tensor_input(0) != Some((vec![1, canvas.len()], DType::U32))
            || sampler.inner.outputs().len() != 2
            || sampler.inner.outputs()[1] != (vec![canvas.len() * 4 + 1], DType::F32)
        {
            return Err(invalid(
                "executeChain97: expected bounded U32 canvas and packed F32 statistics output",
            ));
        }
        let upload = Node::new(NodeKind::FromBytes {
            data: canvas
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            shape: vec![1, canvas.len()],
            dtype: DType::U32,
            device: Device::Cuda(self.inner.ordinal()),
        })
        .map_err(invalid)?;
        let device = CudaDevice::get(self.inner.ordinal()).map_err(failure)?;
        let head = head.map(|head| head.inner.clone());
        let sampler = sampler.inner.clone();
        let schema = self
            .state
            .as_ref()
            .ok_or_else(|| invalid("executeChain97: state schema required"))?;
        if schema.access != StateAccessMode::ReadOnly
            || schema.geometry.kda.layers != 0
            || schema.geometry.conv.layers != 0
            || schema.packed_rows_per_sequence.is_some()
        {
            return Err(invalid(
                "executeChain97: requires a dense read-only KV executable",
            ));
        }
        let batch = schema.batch as usize;
        if prefixes.is_empty()
            || prefixes.len() > batch
            || slots.len() != prefixes.len()
            || active_mask.len() != batch
            || valid_lengths.len() != batch
        {
            return Err(invalid("executeChain97: invalid fixed-lane metadata"));
        }
        let prefixes = prefixes
            .iter()
            .map(|prefix| prefix.borrowed())
            .collect::<Result<Vec<_>>>()?;
        let mut seen = vec![false; batch];
        for (prefix, &slot) in prefixes.iter().zip(&slots) {
            let slot = slot as usize;
            if slot >= batch || seen[slot] || valid_lengths[slot] == 0 {
                return Err(invalid("executeChain97: invalid or duplicate active slot"));
            }
            seen[slot] = true;
            if !Arc::ptr_eq(&prefixes[0].pool, &prefix.pool) {
                return Err(invalid("executeChain97: prefixes must share a pool"));
            }
            schema.validate_pool(&prefix.pool, self.inner.ordinal())?;
        }
        if (0..batch).any(|slot| {
            active_mask[slot] != seen[slot] || (!seen[slot] && valid_lengths[slot] != 0)
        }) {
            return Err(invalid(
                "executeChain97: inconsistent inactive lane metadata",
            ));
        }
        let bindings = bindings
            .into_iter()
            .map(NativeTensor::value)
            .collect::<Result<Vec<_>>>()?;
        let schema = schema.clone();
        let executable = self.inner.clone();
        let state = token
            .map(|token| token.state.clone())
            .unwrap_or_else(|| Arc::new(CancellationState::new()));
        let notify = token.map(|token| token.notify.clone());
        run_compute(state, notify, move |cancelled, _| {
            let mut invocation = CudaStateInvocation {
                sequences: prefixes
                    .iter()
                    .map(|prefix| CudaSequenceState {
                        cursor: prefix.state.cursor,
                        keys: Vec::new(),
                        values: Vec::new(),
                        kda_states: Vec::new(),
                        conv_states: Vec::new(),
                        kv_storage: prefix.state.kv_storage.clone(),
                    })
                    .collect(),
                slots,
                valid_lengths,
                capacity: schema.max_tokens,
                cache_dtype: schema.kv_dtype,
                packed_rows_per_sequence: None,
                kv_layers: schema.geometry.kv_layers,
                access: StateAccessMode::ReadOnly,
                cache: None,
            };
            chain97::execute(
                device,
                upload,
                bindings,
                |bindings| executable.execute_stateful(bindings, &[], &mut invocation, cancelled),
                head.as_deref(),
                &sampler,
                temperature,
                cancelled,
            )
            .map(|(feedback, statistics)| NativeChain97Output {
                feedback: NativeTensor::wrap(feedback),
                statistics: Buffer::from(statistics),
            })
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
    pub async fn execute(
        &self,
        bindings: Vec<&NativeTensor>,
        scalars: Vec<f64>,
        token: Option<&CancellationToken>,
    ) -> Result<Vec<NativeTensor>> {
        if self.state.is_some() {
            return Err(invalid(
                "execute: stateful executable requires executeStateful or executeReadOnly",
            ));
        }
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
        run_compute(state, notify, move |cancelled, cancellation| {
            let mut decode_state = Self::decode_state_with_schema(
                &executable,
                &schema,
                &leases,
                &slots,
                &valid_lengths,
            )?;
            Self::with_state_transaction(&leases, &slots, &mut decode_state, |decode_state| {
                let pool = leases[0].inner.pool.clone();
                let mut usage = pool.usage.lock().unwrap_or_else(|error| error.into_inner());
                let (mut planned, block_keys) =
                    Self::block_plan(&schema, &leases, &tokens, &usage)?;
                let values = executable
                    .execute_stateful(&bindings, &[], decode_state, cancelled)
                    .map_err(failure)?;
                let readback_kv = true;
                if readback_kv {
                    executable.readback_state(decode_state).map_err(failure)?;
                }
                if !cancellation.complete() {
                    return Err(Error::new(Status::Cancelled, "operation aborted"));
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
        run_compute(state, notify, move |cancelled, cancellation| {
            let mut decode_state = Self::decode_state_with_schema(
                &executable,
                &schema,
                &leases,
                &slots,
                &valid_lengths,
            )?;
            Self::with_state_transaction(&leases, &slots, &mut decode_state, |decode_state| {
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
                let readback_kv = true;
                if readback_kv {
                    executable.readback_state(decode_state).map_err(failure)?;
                }
                if !cancellation.complete() {
                    return Err(Error::new(Status::Cancelled, "operation aborted"));
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
        run_compute(state, notify, move |cancelled, cancellation| {
            let mut decode_state = Self::decode_state_with_schema(
                &executable,
                &schema,
                &leases,
                &slots,
                &valid_lengths,
            )?;
            Self::with_state_transaction(&leases, &slots, &mut decode_state, |decode_state| {
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
                    executable.readback_state(decode_state).map_err(failure)?;
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
                let readback_kv = true;
                if !cancellation.complete() {
                    return Err(Error::new(Status::Cancelled, "operation aborted"));
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
        run_compute(state, notify, move |cancelled, cancellation| {
            let mut decode_state =
                Self::decode_state_with_schema(&executable, &schema, &leases, &slots, &valid_lengths)?;
            Self::with_state_transaction(&leases, &slots, &mut decode_state, |decode_state| {
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
                let committed = Self::decode_state_with_schema(
                    &executable,
                    &schema,
                    &leases,
                    &slots,
                    &actual_advances,
                )?;
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
            let readback_kv = true;
            if readback_kv {
                executable.readback_state(decode_state).map_err(failure)?;
            }
            if !cancellation.complete() { return Err(Error::new(Status::Cancelled, "operation aborted")); }
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
        let value = CudaValue::from_dense_bytes(self._device.clone(), shape, dtype, &data)
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

    /// Materializes a bounded set of existing scalar F32/U32 byte literals.
    #[napi]
    pub async fn materialize_literals89(
        &self,
        roots: Vec<&LazyTensor>,
        token: Option<&CancellationToken>,
    ) -> Result<Vec<NativeTensor>> {
        literal89::execute(
            self._device.clone(),
            roots.into_iter().map(|root| root.node.clone()).collect(),
            token,
        )
        .await
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
            "input" => {
                let shape = dimensions(&attributes, "shape")?;
                let dtype = parse_dtype(string(&attributes, "dtype")?)?;
                let storage = input_storage(&attributes)?;
                ValueSpec {
                    semantic_dtype: dtype,
                    logical_shape: &shape,
                    storage: storage.as_spec(),
                }
                .validate()
                .map_err(invalid)?;
                NodeKind::Input {
                    slot: u32::try_from(integer(&attributes, "slot")?)
                        .map_err(|_| invalid("input slot exceeds u32"))?,
                    shape,
                    dtype,
                    storage,
                    device: expected,
                }
            }
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
            "topKIndices" => NodeKind::TopKIndices {
                a: input(&inputs, 0, &operation)?,
                k: integer(&attributes, "k")?,
            },
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
            "scaledDotProductAttentionConfigured" => {
                let parse_window = |name: &str| -> Result<AttentionWindow> {
                    match attributes.get(name) {
                        None => Ok(AttentionWindow::Inherit),
                        Some(JsonValue::Null) => Ok(AttentionWindow::Full),
                        Some(value) => value
                            .as_u64()
                            .and_then(|v| usize::try_from(v).ok())
                            .map(AttentionWindow::Local)
                            .ok_or_else(|| {
                                invalid(format!("{name} must be a non-negative integer or null"))
                            }),
                    }
                };
                NodeKind::SdpaConfigured {
                    q: input(&inputs, 0, &operation)?,
                    k: input(&inputs, 1, &operation)?,
                    v: input(&inputs, 2, &operation)?,
                    scale: number(&attributes, "scale")?,
                    causal: boolean(&attributes, "causal")?,
                    window: parse_window("window")?,
                    rounding: match string(&attributes, "rounding")? {
                        "fused" => effect_torch_graph::AttentionRounding::Fused,
                        "stepwise" => effect_torch_graph::AttentionRounding::Stepwise,
                        _ => return Err(invalid("unsupported attention rounding")),
                    },
                    layer_id: attributes
                        .get("layerId")
                        .map(|value| {
                            value
                                .as_u64()
                                .and_then(|v| u32::try_from(v).ok())
                                .ok_or_else(|| invalid("layerId must be a U32 integer"))
                        })
                        .transpose()?,
                    retention: parse_window("retentionWindow")?,
                }
            }
            "rotaryEmbeddingExplicit" => NodeKind::RotaryEmbeddingExplicit {
                x: input(&inputs, 0, &operation)?,
                positions: input(&inputs, 1, &operation)?,
                inverse_frequencies: input(&inputs, 2, &operation)?,
                layout: match string(&attributes, "layout")? {
                    "HalfSplit" => RotaryLayout::HalfSplit,
                    "InterleavedPairs" => RotaryLayout::InterleavedPairs,
                    layout => return Err(invalid(format!("unsupported rotary layout {layout}"))),
                },
            },
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
            "expertLinearRows" => NodeKind::ExpertLinearRows {
                x: input(&inputs, 0, &operation)?,
                weight: input(&inputs, 1, &operation)?,
                indexes: input(&inputs, 2, &operation)?,
            },
            "groupedExpertLinearRows" => NodeKind::GroupedExpertLinearRows {
                x: input(&inputs, 0, &operation)?,
                weight: input(&inputs, 1, &operation)?,
                indexes: input(&inputs, 2, &operation)?,
            },
            "linear" => NodeKind::Linear {
                x: input(&inputs, 0, &operation)?,
                weight: input(&inputs, 1, &operation)?,
                bias: input(&inputs, 2, &operation)?,
            },
            "quantizedLinear" => NodeKind::QuantizedLinear {
                x: input(&inputs, 0, &operation)?,
                weight: input(&inputs, 1, &operation)?,
                bias: inputs.get(2).cloned(),
            },
            "quantizedEmbedding" => {
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
                )));
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
                let kv_dtype = parse_dtype(&native.kv_dtype)?;
                if !matches!(kv_dtype, DType::F32 | DType::F16 | DType::BF16 | DType::U8) {
                    return Err(invalid("compile: KV dtype must be f32, f16, bf16, or u8"));
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
                let (rewritten, mut geometry) = specialize_decode_layout_outputs_with_attention(
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
                for layer in &mut geometry.kv_layers {
                    layer.dtype = kv_dtype;
                }
                Some(CudaStateSchema {
                    access: state_access(native.access.as_deref())?,
                    max_tokens: native.max_tokens,
                    block_size: native.block_size,
                    kv_dtype,
                    batch: native.batch,
                    packed_rows_per_sequence,
                    geometry,
                })
            }
        };
        let options = compile_options(options, state.is_some());
        let compiled = match &state {
            Some(state) => compile_stateful_with_layout(
                roots,
                self.ordinal,
                state.geometry.cursor_slot,
                state.geometry.cursor_tensor,
                options,
                CudaStateLayout {
                    capacity: state.max_tokens,
                    dtype: state.kv_dtype,
                    slots: state.batch,
                    packed_rows_per_sequence: state.packed_rows_per_sequence,
                    kv_layers: state.geometry.kv_layers.clone(),
                    access: state.access,
                },
            ),
            None => compile_with_options(roots, self.ordinal, options),
        };
        compiled
            .map(|inner| Executable {
                inner: Arc::new(inner),
                state,
                request_rng99: None,
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

#[napi(object, object_from_js = false)]
pub struct NativeSafetensorsTensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u32>,
    pub byte_length: f64,
}

#[napi(object, object_from_js = false)]
pub struct NativeSafetensorsInspection {
    pub entries: Vec<NativeSafetensorsTensorInfo>,
    pub metadata: HashMap<String, String>,
}

fn native_safetensors_inspection(
    inspection: effect_torch_napi::safetensors::Inspection,
) -> Result<NativeSafetensorsInspection> {
    let mut entries = Vec::with_capacity(inspection.entries.len());
    for meta in inspection.entries {
        entries.push(NativeSafetensorsTensorInfo {
            name: meta.name,
            dtype: effect_torch_napi::safetensors::dtype_name(meta.dtype),
            shape: meta.shape,
            byte_length: effect_torch_napi::safetensors::byte_length_f64(meta.byte_length)
                .map_err(effect_torch_napi::safetensors::Error::into_napi)?,
        });
    }
    Ok(NativeSafetensorsInspection {
        entries,
        metadata: inspection.metadata,
    })
}

/// Reads a standalone safetensors header or a Hugging Face index and every
/// shard header it references without acquiring a CUDA device or allocating.
#[napi]
pub async fn inspect_safetensors(
    path: String,
    token: Option<&CancellationToken>,
) -> Result<NativeSafetensorsInspection> {
    let state = token
        .map(|token| token.state.clone())
        .unwrap_or_else(|| Arc::new(CancellationState::new()));
    let notify = token.map(|token| token.notify.clone());
    run_compute(state, notify, move |cancelled, _| {
        if cancelled.is_cancelled() {
            return Err(Error::new(Status::Cancelled, "operation aborted"));
        }
        let inspection =
            effect_torch_napi::safetensors::inspect(&path, &|| cancelled.is_cancelled())
                .map_err(effect_torch_napi::safetensors::Error::into_napi)?;
        native_safetensors_inspection(inspection)
    })
    .await
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
    names: Option<Vec<String>>,
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
        let archive = safetensors_io::load(&path, names.as_deref(), device, &|| {
            cancelled.is_cancelled()
        })
        .map_err(effect_torch_napi::safetensors::Error::into_napi)?;
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

#[cfg(test)]
#[path = "napi_state_storage_tests.rs"]
mod state_storage_tests;

#[cfg(test)]
mod storage_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn input_metadata_accepts_only_closed_representations() {
        for invalid_storage in [
            json!(null),
            json!("Q4_K"),
            json!({}),
            json!({"representation": "opaque"}),
            json!({"representation": "packed"}),
            json!({"representation": "dense", "format": "Q4_K"}),
            json!({"representation": "packed", "format": "GGML"}),
            json!({"representation": "packed", "format": "Q4_K", "blockSize": 256}),
        ] {
            assert!(input_storage(&json!({"storage": invalid_storage})).is_err());
        }
        assert_eq!(input_storage(&json!({})).unwrap(), StorageMetadata::dense());
        assert_eq!(
            input_storage(&json!({"storage": {"representation": "dense"}})).unwrap(),
            StorageMetadata::dense()
        );
        for format in ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"] {
            let storage =
                input_storage(&json!({"storage": {"representation": "packed", "format": format}}))
                    .unwrap();
            let spec = ValueSpec {
                semantic_dtype: DType::F32,
                logical_shape: &[2, 256],
                storage: storage.as_spec(),
            };
            assert!(spec.validate().is_ok());
            assert!(ValueSpec {
                semantic_dtype: DType::U8,
                ..spec
            }
            .validate()
            .is_err());
            assert!(ValueSpec {
                logical_shape: &[2, 257],
                ..spec
            }
            .validate()
            .is_err());
            let published = native_storage(storage.representation);
            assert_eq!(published.representation, "packed");
            assert_eq!(published.format.as_deref(), Some(format));
        }
    }
}

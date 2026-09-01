//! Node.js (napi-rs) bindings for the CPU runtime.
//!
//! The bindings export:
//!
//! - [`NativeTensor`] is a materialized CPU tensor held in a graph leaf slot.
//!   `EXTERNAL_MEMORY_BYTES` tracks its byte size and mirrors it to V8's
//!   external-memory accounting. [`NativeTensor::readback`] exports a JS
//!   `ArrayBuffer`. Large, contiguous buffers not previously exported use
//!   zero-copy readback. Other buffers use an owned copy.
//! - [`LazyTensor`] is a lazy graph node whose methods build graph structure
//!   without running it.
//! - [`compile`] and [`Executable`] compile roots into a cached
//!   `CpuExecutable` and run it asynchronously on the tokio worker pool with
//!   optional scalar bindings and cancellation.
//! - [`CancellationToken`] provides cooperative cancellation shared with the
//!   runtime [`CancellationFlag`]. Compute tasks poll it and abort with a
//!   `Cancelled` status.
//! - [`NativeKvPool`] and [`NativeKvSequence`] manage paged KV caches for
//!   stateful decoding. The pool context implements `executable::CpuState` to
//!   stage and commit cache updates as a transaction.
//! - `save_tensors` and `load_tensors` handle safetensors archives.
//!   [`inspect_gguf`] and [`load_gguf`] handle GGUF archives.
//!
//! Exported buffers either deep-copy into an owned allocation with
//! `FinalizeHint::Owned`, or use `FinalizeHint::ZeroCopy` to keep the source
//! tensor alive and register its address. Registration prevents exporting the
//! same range twice. The N-API finalizer releases either kind exactly once.

mod err;
mod gguf;
mod safetensors;
mod value;

pub use gguf::{inspect_gguf, load_gguf};

use self::err::to_napi_err;
use self::value::Value;
use crate::{composed, executable, pool, CpuBuffer, CpuDestination, Elem, Tensor};
use effect_torch_compiler::{
    specialize_decode_layout_outputs_with_attention, CompileOptions, CurrentBlockAttention,
    DecodeGeometry, DecodeLayout, DecodeOutputSelection, InferenceOptions, PreparedProgram,
    ProgramRequest, ProgramSlot, StateCursorSlot,
};
use effect_torch_graph::CrossEntropyReduction as CeReduction;
use effect_torch_graph::{AttentionWindow, Device, KvAttentionMode, PositionOffset, RotaryLayout};
use effect_torch_napi::{try_register_export, unregister_export, vec_to_bytes, CancellationState};
use effect_torch_runtime::{
    effective_probabilities, purpose_counter, random_unit_at, sample_logits, sample_probabilities,
    sampling_coordinate, Buffer, CancellationFlag, DType, GgmlKQuant, Layout, SamplingOptions,
    SamplingPurpose,
};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

pub type LeafSlot = effect_torch_graph::LeafSlot;
pub(crate) type Node = effect_torch_graph::Node;
type NodeKind = effect_torch_graph::NodeKind;
type KdaGeometry = effect_torch_compiler::KdaGeometry;
type ConvGeometry = effect_torch_compiler::ConvGeometry;

fn cpu_device() -> Device {
    Device::Cpu(0)
}

fn attention_window(value: i64) -> Result<AttentionWindow> {
    match value {
        -1 => Ok(AttentionWindow::Inherit),
        0 => Ok(AttentionWindow::Full),
        value if value > 0 => usize::try_from(value)
            .map(AttentionWindow::Local)
            .map_err(|_| Error::new(Status::InvalidArg, "attention window is out of range")),
        _ => Err(Error::new(
            Status::InvalidArg,
            "attention window must encode inherit, full, or a positive size",
        )),
    }
}

fn rotary_layout(value: &str) -> Result<RotaryLayout> {
    match value {
        "HalfSplit" => Ok(RotaryLayout::HalfSplit),
        "InterleavedPairs" => Ok(RotaryLayout::InterleavedPairs),
        _ => Err(Error::new(
            Status::InvalidArg,
            format!("unsupported rotary layout {value}"),
        )),
    }
}

fn ggml_k_quant(value: &str) -> Result<GgmlKQuant> {
    GgmlKQuant::from_name(value).ok_or_else(|| {
        Error::new(
            Status::InvalidArg,
            format!("unsupported GGML K-quant encoding {value}"),
        )
    })
}

/// Describes how to release an exported readback buffer when V8 finalizes its
/// external `ArrayBuffer`.
enum FinalizeHint {
    /// The buffer aliases live tensor storage. Dropping the clone releases the
    /// tensor reference and unregisters the address, allowing another export.
    ZeroCopy { value: Value, addr: usize },
    /// The buffer owns a leaked `Vec<u8>` copy that the finalizer reconstructs
    /// and drops.
    Owned {
        ptr: *mut u8,
        len: usize,
        cap: usize,
    },
}

/// N-API finalizer for external array buffers.
///
/// # Safety
/// Node must call this exactly once per external buffer. `hint` must be the
/// pointer produced by `Box::into_raw` in `to_napi_value`.
unsafe extern "C" fn finalize_readback(
    _env: napi::sys::napi_env,
    _data: *mut std::ffi::c_void,
    hint: *mut std::ffi::c_void,
) {
    // SAFETY: `hint` came from `Box::into_raw`. The call-site guard makes
    // this its only reclamation point.
    let hint = unsafe { Box::from_raw(hint as *mut FinalizeHint) };
    release_readback(*hint);
}

fn release_readback(hint: FinalizeHint) {
    match hint {
        FinalizeHint::ZeroCopy { value, addr } => {
            drop(value);
            unregister_export(addr);
        }
        FinalizeHint::Owned { ptr, len, cap } => {
            // SAFETY: `ptr/len/cap` came from a leaked `Vec<u8>` via
            // `vec_to_bytes`. The finalizer reconstructs them exactly once.
            drop(unsafe { Vec::from_raw_parts(ptr, len, cap) });
        }
    }
}

/// Byte buffer returned to JS as an external `ArrayBuffer`.
///
/// Owns the byte release plan in `hint`. `Drop` releases it immediately if
/// the value never reaches N-API.
pub struct Readback {
    data: *mut u8,
    byte_len: usize,
    hint: Option<FinalizeHint>,
}

// SAFETY: Node dereferences the raw pointer only while the buffer is alive.
// The hint keeps its tensor clone or owned Vec alive until the finalizer runs.
// The N-API runtime's env model synchronizes all access.
unsafe impl Send for Readback {}

/// Releases the hint unless a successful N-API call nulls the pointer. This
/// prevents a failed `to_napi_value` from leaking or freeing twice.
struct FinalizeHintGuard(*mut std::ffi::c_void);

impl Drop for FinalizeHintGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: a non-null pointer means N-API never took ownership, so
            // this is the only reclamation.
            let hint = unsafe { Box::from_raw(self.0 as *mut FinalizeHint) };
            release_readback(*hint);
        }
    }
}

impl Drop for Readback {
    fn drop(&mut self) {
        if let Some(hint) = self.hint.take() {
            release_readback(hint);
        }
    }
}

impl ToNapiValue for Readback {
    /// Converts the readback into an external `ArrayBuffer` whose finalizer
    /// owns the release hint.
    ///
    /// # Safety
    /// The returned value keeps `value.data` valid by moving the release hint
    /// into the finalizer. The finalizer releases the hint once on success.
    /// `hint_guard` releases it once on failure.
    unsafe fn to_napi_value(
        env: napi::sys::napi_env,
        mut value: Self,
    ) -> Result<napi::sys::napi_value> {
        let hint = Box::into_raw(Box::new(
            value
                .hint
                .take()
                .expect("readback ownership transferred once"),
        )) as *mut std::ffi::c_void;
        let mut hint_guard = FinalizeHintGuard(hint);
        let mut result = std::ptr::null_mut();
        napi::check_status!(
            // SAFETY: `env` is live for this conversion. `value.data` points
            // to `value.byte_len` bytes kept alive by the hint. The finalizer
            // and hint form valid heap state.
            unsafe {
                napi::sys::napi_create_external_arraybuffer(
                    env,
                    value.data.cast(),
                    value.byte_len,
                    Some(finalize_readback),
                    hint,
                    &mut result,
                )
            },
            "failed to create external arraybuffer"
        )?;
        hint_guard.0 = std::ptr::null_mut();
        Ok(result)
    }
}

#[napi(string_enum)]
#[derive(Clone, Copy)]
pub enum NativeDType {
    #[napi(value = "f32")]
    F32,
    #[napi(value = "f64")]
    F64,
    #[napi(value = "i64")]
    I64,
    #[napi(value = "u8")]
    U8,
    #[napi(value = "u32")]
    U32,
    #[napi(value = "f16")]
    F16,
    #[napi(value = "bf16")]
    BF16,
}

impl From<NativeDType> for DType {
    fn from(dtype: NativeDType) -> Self {
        match dtype {
            NativeDType::F32 => DType::F32,
            NativeDType::F64 => DType::F64,
            NativeDType::I64 => DType::I64,
            NativeDType::U8 => DType::U8,
            NativeDType::U32 => DType::U32,
            NativeDType::F16 => DType::F16,
            NativeDType::BF16 => DType::BF16,
        }
    }
}

#[napi(object)]
pub struct NativeCompileOptions {
    pub optimize: Option<bool>,
    pub constant_weights: Option<bool>,
}

#[napi(object)]
pub struct NativeSamplingOptions {
    pub temperature: f64,
    pub top_k: f64,
    pub top_p: f64,
    pub seed: f64,
    pub counter: f64,
}

fn non_negative_safe_integer(value: f64, name: &str) -> Result<u64> {
    const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > MAX_SAFE_INTEGER {
        return Err(Error::new(
            Status::InvalidArg,
            format!("sample: {name} must be a non-negative safe integer, got {value}"),
        ));
    }
    Ok(value as u64)
}

fn sampling_options(options: NativeSamplingOptions) -> Result<SamplingOptions> {
    let top_k = non_negative_safe_integer(options.top_k, "topK")?;
    Ok(SamplingOptions {
        temperature: options.temperature,
        top_k: if top_k == 0 {
            None
        } else {
            Some(
                usize::try_from(top_k)
                    .map_err(|_| Error::new(Status::InvalidArg, "sample: topK is out of range"))?,
            )
        },
        top_p: options.top_p,
        seed: non_negative_safe_integer(options.seed, "seed")?,
        counter: non_negative_safe_integer(options.counter, "counter")?,
    })
}

#[napi(object)]
pub struct NativePackedCausalChainsLayout {
    pub rows_per_sequence: u32,
}

#[napi(object)]
pub struct NativeKvStateSchema {
    pub max_tokens: u32,
    pub block_size: u32,
    pub kv_dtype: NativeDType,
    pub window: Option<u32>,
    pub batch: u32,
    pub packed_causal_chains: Option<NativePackedCausalChainsLayout>,
    pub last_token_row: Option<bool>,
    pub output_selections: Option<Vec<NativeDecodeOutputSelection>>,
    pub current_block_attention: Option<NativeCurrentBlockAttention>,
}

#[napi(string_enum)]
#[derive(Clone, Copy)]
pub enum NativeCurrentBlockAttention {
    Causal,
    Bidirectional,
}

impl From<NativeCurrentBlockAttention> for CurrentBlockAttention {
    fn from(mode: NativeCurrentBlockAttention) -> Self {
        match mode {
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
    fn from(selection: NativeDecodeOutputSelection) -> Self {
        match selection {
            NativeDecodeOutputSelection::AllRows => Self::AllRows,
            NativeDecodeOutputSelection::SplitLastTokenRow => Self::SplitLastTokenRow,
            NativeDecodeOutputSelection::BatchedLastTokenRow => Self::BatchedLastTokenRow,
        }
    }
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

/// A materialized CPU tensor exported to JavaScript.
///
/// Wraps the value in a graph [`LeafSlot`] so it can feed compiled programs
/// as a generated binding. Wrap, clear, and finalize mirror the tracked byte
/// size to V8's external memory accounting.
#[napi(custom_finalize)]
pub struct NativeTensor {
    pub(crate) slot: Arc<LeafSlot>,
    bytes: i64,
}

impl NativeTensor {
    fn wrap(value: Value) -> Self {
        let bytes = value.byte_size().max(4096) as i64;
        EXTERNAL_MEMORY_BYTES.fetch_add(bytes, Ordering::Relaxed);
        Self {
            slot: Arc::new(LeafSlot::new(value)),
            bytes,
        }
    }

    fn value_cloned(&self) -> Result<Value> {
        self.slot
            .get()
            .map_err(|error| Error::new(Status::GenericFailure, error.to_string()))
    }

    fn release_accounting(&mut self) {
        if self.bytes != 0 {
            EXTERNAL_MEMORY_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
            self.bytes = 0;
        }
    }
}

impl Drop for NativeTensor {
    fn drop(&mut self) {
        self.release_accounting();
    }
}

static EXTERNAL_MEMORY_BYTES: AtomicI64 = AtomicI64::new(0);
static V8_REPORTED: AtomicI64 = AtomicI64::new(0);

fn sync_v8(env: &Env) {
    let accounted = EXTERNAL_MEMORY_BYTES.load(Ordering::Relaxed);
    let reported = V8_REPORTED.swap(accounted, Ordering::Relaxed);
    let delta = accounted - reported;
    if delta != 0 {
        let _ = env.adjust_external_memory(delta);
    }
}

impl ObjectFinalize for NativeTensor {
    fn finalize(mut self, env: Env) -> Result<()> {
        self.release_accounting();
        sync_v8(&env);
        Ok(())
    }
}

/// Cooperative cancellation handle shared with async compute tasks.
///
/// `cancel()` sets the flag and wakes the tokio notifier so a blocked
/// executor can abort. Kernels also poll the flag inside their loops
/// and return `Status::Cancelled` with "operation aborted".
#[napi]
pub struct CancellationToken {
    state: Arc<CancellationState>,
    notify: Arc<tokio::sync::Notify>,
}

#[napi]
impl CancellationToken {
    #[napi(constructor)]
    pub fn new(env: Env) -> Self {
        sync_v8(&env);
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
impl NativeTensor {
    #[napi]
    pub fn clear(&mut self, env: Env) -> Result<()> {
        if self.slot.clear() {
            EXTERNAL_MEMORY_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
            self.bytes = 0;
            sync_v8(&env);
        }
        Ok(())
    }

    #[napi(getter)]
    pub fn shape(&self) -> Result<Vec<u32>> {
        Ok(self
            .value_cloned()?
            .shape()
            .iter()
            .map(|&dimension| dimension as u32)
            .collect())
    }

    #[napi(getter)]
    pub fn dtype(&self) -> Result<String> {
        Ok(self.value_cloned()?.dtype().name().to_string())
    }

    #[napi(getter)]
    pub fn device(&self) -> Result<String> {
        self.value_cloned()?;
        Ok("cpu".to_string())
    }

    #[napi(ts_return_type = "Promise<ArrayBuffer>")]
    pub async fn readback(&self, token: Option<&CancellationToken>) -> Result<Readback> {
        let value = self.value_cloned()?;
        run_compute(token, move |cancelled, _state| {
            if cancelled.load(Ordering::Acquire) {
                return Err(Error::new(Status::Cancelled, "operation aborted"));
            }
            let readback = readback_blocking(&value)?;
            if cancelled.load(Ordering::Acquire) {
                return Err(Error::new(Status::Cancelled, "operation aborted"));
            }
            Ok(readback)
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
        cancellation_token: Option<&CancellationToken>,
    ) -> Result<u32> {
        let value = self.value_cloned()?;
        let options = sampling_options(NativeSamplingOptions {
            temperature,
            top_k,
            top_p,
            seed,
            counter,
        })?;
        run_compute(cancellation_token, move |cancelled, _state| {
            sample_blocking(&value, options, || cancelled.is_cancelled())
        })
        .await
    }
}

fn sample_blocking(
    value: &Value,
    options: SamplingOptions,
    cancelled: impl FnMut() -> bool,
) -> Result<u32> {
    let tensor = value.tensor();
    if tensor.layout.rank() != 1 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "sample: logits must be rank 1, got rank {}",
                tensor.layout.rank()
            ),
        ));
    }
    let length = tensor.numel();
    let offset = tensor.layout.offset();
    let stride = tensor.layout.strides()[0];
    macro_rules! sample {
        ($values:expr) => {
            sample_logits(
                length,
                |index| $values[offset + index * stride].to_f64(),
                options,
                cancelled,
            )
        };
    }
    let result = match &tensor.buffer {
        CpuBuffer::F16(values) => sample!(values),
        CpuBuffer::BF16(values) => sample!(values),
        CpuBuffer::F32(values) => sample!(values),
        CpuBuffer::F64(values) => sample!(values),
        _ => {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "sample: logits must have a floating-point dtype, got {}",
                    tensor.dtype()
                ),
            ))
        }
    };
    result.map_err(|message| {
        let status = if message == "operation aborted" {
            Status::Cancelled
        } else {
            Status::InvalidArg
        };
        Error::new(status, message)
    })
}

fn readback_blocking(value: &Value) -> Result<Readback> {
    let tensor = value.tensor();
    let element_size = tensor.dtype().size_in_bytes();
    let base = match &tensor.buffer {
        CpuBuffer::U8(values) => values.as_ptr().cast::<u8>(),
        CpuBuffer::U32(values) => values.as_ptr().cast::<u8>(),
        CpuBuffer::I64(values) => values.as_ptr().cast::<u8>(),
        CpuBuffer::BF16(values) => values.as_ptr().cast::<u8>(),
        CpuBuffer::F16(values) => values.as_ptr().cast::<u8>(),
        CpuBuffer::F32(values) => values.as_ptr().cast::<u8>(),
        CpuBuffer::F64(values) => values.as_ptr().cast::<u8>(),
    };
    let count = tensor.numel();
    let logical_offset = |mut linear: usize| {
        let mut offset = tensor.layout.offset();
        for dimension in (0..tensor.layout.rank()).rev() {
            let width = tensor.layout.shape()[dimension].max(1);
            offset += (linear % width) * tensor.layout.strides()[dimension];
            linear /= width;
        }
        offset
    };
    macro_rules! gather {
        ($values:expr, $type:ty) => {{
            let values = $values;
            (0..count)
                .map(|index| values[logical_offset(index)])
                .collect::<Vec<$type>>()
        }};
    }
    let owned = match &tensor.buffer {
        CpuBuffer::F16(values) => Some(vec_to_bytes(
            gather!(values, half::f16)
                .into_iter()
                .map(f32::from)
                .collect::<Vec<_>>(),
        )),
        CpuBuffer::BF16(values) => Some(vec_to_bytes(
            gather!(values, half::bf16)
                .into_iter()
                .map(f32::from)
                .collect::<Vec<_>>(),
        )),
        CpuBuffer::F32(values) if !tensor.layout.is_contiguous() => {
            Some(vec_to_bytes(gather!(values, f32)))
        }
        CpuBuffer::F64(values) if !tensor.layout.is_contiguous() => {
            Some(vec_to_bytes(gather!(values, f64)))
        }
        CpuBuffer::I64(values) if !tensor.layout.is_contiguous() => {
            Some(vec_to_bytes(gather!(values, i64)))
        }
        CpuBuffer::U8(values) if !tensor.layout.is_contiguous() => {
            Some(vec_to_bytes(gather!(values, u8)))
        }
        CpuBuffer::U32(values) if !tensor.layout.is_contiguous() => {
            Some(vec_to_bytes(gather!(values, u32)))
        }
        _ => None,
    };
    if let Some((_, ptr, len, cap)) = owned {
        return Ok(Readback {
            data: ptr,
            byte_len: len,
            hint: Some(FinalizeHint::Owned { ptr, len, cap }),
        });
    }
    let offset = tensor.layout.offset() * element_size;
    let byte_len = count * element_size;
    if !base.is_null() && byte_len <= 4096 {
        // SAFETY: the tensor view keeps the segment alive and covers
        // `offset..offset + byte_len` initialized bytes. Small tensors are
        // always copied and never aliased into JS.
        let bytes = unsafe { std::slice::from_raw_parts(base.add(offset), byte_len) }.to_vec();
        let (_, ptr, len, cap) = vec_to_bytes(bytes);
        return Ok(Readback {
            data: ptr,
            byte_len: len,
            hint: Some(FinalizeHint::Owned { ptr, len, cap }),
        });
    }
    if !base.is_null() {
        let addr = base as usize + offset;
        if try_register_export(addr) {
            return Ok(Readback {
                data: addr as *mut u8,
                byte_len,
                hint: Some(FinalizeHint::ZeroCopy {
                    value: value.clone(),
                    addr,
                }),
            });
        }
    }
    // SAFETY: fallback copy after zero-copy registration failed. The retained
    // tensor view covers `offset..offset + byte_len` initialized bytes.
    let bytes = unsafe { std::slice::from_raw_parts(base.add(offset), byte_len) }.to_vec();
    let (_, ptr, len, cap) = vec_to_bytes(bytes);
    Ok(Readback {
        data: ptr,
        byte_len: len,
        hint: Some(FinalizeHint::Owned { ptr, len, cap }),
    })
}

struct ConstantCache {
    map: HashMap<(u64, DType), Arc<Node>>,
    order: VecDeque<(u64, DType)>,
}

static CONSTANT_CACHE: LazyLock<Mutex<ConstantCache>> = LazyLock::new(|| {
    Mutex::new(ConstantCache {
        map: HashMap::new(),
        order: VecDeque::new(),
    })
});

const CONSTANT_CACHE_LIMIT: usize = 4096;

fn cached_constant(value: f64, dtype: DType) -> std::result::Result<Arc<Node>, String> {
    let key = (value.to_bits(), dtype);
    let mut cache = CONSTANT_CACHE.lock().unwrap();
    if let Some(node) = cache.map.get(&key) {
        return Ok(node.clone());
    }
    let node = Node::new(NodeKind::Full {
        shape: vec![],
        value,
        dtype,
        device: cpu_device(),
    })?;
    if cache.order.len() >= CONSTANT_CACHE_LIMIT {
        if let Some(oldest) = cache.order.pop_front() {
            cache.map.remove(&oldest);
        }
    }
    cache.map.insert(key, node.clone());
    cache.order.push_back(key);
    Ok(node)
}

const CHUNKED_CE_MIN_LOGITS: usize = 1 << 28;
const CHUNKED_CE_CHUNK_LOGITS: usize = 1 << 26;
const CHUNKED_CE_MAX_CHUNKS: usize = 64;

fn chunked_ce_limits() -> (usize, usize) {
    let read = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(default)
    };
    (
        read("EFFECT_TORCH_CE_CHUNK_MIN", CHUNKED_CE_MIN_LOGITS),
        read("EFFECT_TORCH_CE_CHUNK_SIZE", CHUNKED_CE_CHUNK_LOGITS),
    )
}

fn chunked_head_ce(
    logits: &Arc<Node>,
    target: &Arc<Node>,
    ignore_index: i64,
) -> std::result::Result<Arc<Node>, String> {
    let (minimum, chunk_size) = chunked_ce_limits();
    chunked_head_ce_with(logits, target, ignore_index, minimum, chunk_size)
}

fn chunked_head_ce_with(
    logits: &Arc<Node>,
    target: &Arc<Node>,
    ignore_index: i64,
    minimum: usize,
    chunk_size: usize,
) -> std::result::Result<Arc<Node>, String> {
    let plain = Node::new(NodeKind::CrossEntropy {
        logits: logits.clone(),
        target: target.clone(),
        ignore_index,
        reduction: CeReduction::Mean,
    })?;
    let NodeKind::Linear { x, weight, bias } = &logits.kind else {
        return Ok(plain);
    };
    let (_inner, vocabulary) = (weight.shape[0], weight.shape[1]);
    let rank = x.shape.len();
    if rank < 2 {
        return Ok(plain);
    }
    let rows: usize = x.shape[..rank - 1].iter().product();
    let elements = rows.saturating_mul(vocabulary);
    if rows < 2 || elements < minimum {
        return Ok(plain);
    }
    let chunks = (elements / chunk_size)
        .clamp(2, CHUNKED_CE_MAX_CHUNKS)
        .min(rows);
    if chunks < 2 {
        return Ok(plain);
    }
    // Evaluation runs the chunk loop directly, so it never materializes the
    // full [rows, vocab] logits. The closed-form backward keeps one chunk of
    // grad-logits workspace at a time. The graph-chain version kept every
    // chunk's workspace until the head-parameter roots ran.
    Node::new(NodeKind::ChunkedHeadCe {
        x: x.clone(),
        weight: weight.clone(),
        bias: bias.clone(),
        target: target.clone(),
        ignore_index,
    })
}

/// A lazy CPU graph node whose methods build graph structure without running
/// it. Materialize it with `compile` and `execute`. Use `grad` for
/// reverse-mode gradients.
#[napi]
pub struct LazyTensor {
    node: Arc<Node>,
}

/// One named exposure in a lazy graph. Contains the name and wrapped tensor
/// handle from the exposure node's child.
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

macro_rules! lazy_ctor {
    ($body:expr) => {
        match $body {
            Ok(node) => Ok(Self { node }),
            Err(message) => Err(Error::new(Status::InvalidArg, message)),
        }
    };
}

#[napi]
impl LazyTensor {
    #[napi(getter)]
    pub fn shape(&self) -> Vec<u32> {
        self.node
            .shape
            .iter()
            .map(|&dimension| dimension as u32)
            .collect()
    }

    #[napi(getter)]
    pub fn dtype(&self) -> String {
        self.node.dtype.name().to_string()
    }

    #[napi]
    pub fn metadata(&self) -> (Vec<u32>, String) {
        (self.shape(), self.dtype())
    }

    #[napi(factory)]
    pub fn zeros(shape: Vec<u32>, dtype: Option<NativeDType>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Zeros {
            shape: shape
                .into_iter()
                .map(|dimension| dimension as usize)
                .collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: cpu_device(),
        }))
    }

    #[napi(factory)]
    pub fn ones(shape: Vec<u32>, dtype: Option<NativeDType>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Ones {
            shape: shape
                .into_iter()
                .map(|dimension| dimension as usize)
                .collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: cpu_device(),
        }))
    }

    #[napi(factory)]
    pub fn full(shape: Vec<u32>, value: f64, dtype: Option<NativeDType>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Full {
            shape: shape
                .into_iter()
                .map(|dimension| dimension as usize)
                .collect(),
            value,
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: cpu_device(),
        }))
    }

    #[napi(factory)]
    pub fn randn(shape: Vec<u32>, dtype: Option<NativeDType>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Randn {
            shape: shape
                .into_iter()
                .map(|dimension| dimension as usize)
                .collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: cpu_device(),
        }))
    }

    #[napi(factory)]
    pub fn uniform(shape: Vec<u32>, lo: f64, hi: f64, dtype: Option<NativeDType>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Uniform {
            lo,
            hi,
            shape: shape
                .into_iter()
                .map(|dimension| dimension as usize)
                .collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: cpu_device(),
        }))
    }

    #[napi(factory)]
    pub fn arange(start: f64, end: f64, step: f64, dtype: Option<NativeDType>) -> Result<Self> {
        if step == 0.0 {
            return Err(Error::new(
                Status::InvalidArg,
                "arange: step must be non-zero",
            ));
        }
        lazy_ctor!(Node::new(NodeKind::Arange {
            start,
            end,
            step,
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: cpu_device(),
        }))
    }

    #[napi(factory)]
    pub fn eye(n: u32, dtype: Option<NativeDType>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Eye {
            n: n as usize,
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: cpu_device(),
        }))
    }

    #[napi(factory)]
    pub fn constant(value: f64, dtype: Option<NativeDType>) -> Result<Self> {
        let dtype = dtype.unwrap_or(NativeDType::F32).into();
        lazy_ctor!(cached_constant(value, dtype))
    }

    #[napi(factory)]
    pub fn from_bytes(
        data: Uint8Array,
        shape: Vec<u32>,
        dtype: Option<NativeDType>,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::FromBytes {
            data: data.to_vec(),
            shape: shape
                .into_iter()
                .map(|dimension| dimension as usize)
                .collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: cpu_device(),
        }))
    }

    #[napi(factory)]
    pub fn from_materialized(tensor: &NativeTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Leaf(tensor.slot.clone())))
    }

    #[napi(factory)]
    pub fn input(slot: u32, shape: Vec<u32>, dtype: Option<NativeDType>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Input {
            slot,
            shape: shape
                .into_iter()
                .map(|dimension| dimension as usize)
                .collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: cpu_device(),
        }))
    }

    #[napi(factory)]
    pub fn scalar_input(slot: u32, dtype: Option<NativeDType>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::ScalarInput {
            slot,
            dtype: dtype.unwrap_or(NativeDType::F64).into(),
            device: cpu_device(),
        }))
    }

    #[napi]
    pub fn add(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Add {
            a: self.node.clone(),
            b: other.node.clone()
        }))
    }

    #[napi]
    pub fn sub(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Sub {
            a: self.node.clone(),
            b: other.node.clone()
        }))
    }

    #[napi]
    pub fn mul(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Mul {
            a: self.node.clone(),
            b: other.node.clone()
        }))
    }

    #[napi]
    pub fn div(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Div {
            a: self.node.clone(),
            b: other.node.clone()
        }))
    }

    #[napi]
    pub fn maximum(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Maximum {
            a: self.node.clone(),
            b: other.node.clone()
        }))
    }

    #[napi]
    pub fn minimum(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Minimum {
            a: self.node.clone(),
            b: other.node.clone()
        }))
    }

    #[napi]
    pub fn eq(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Eq {
            a: self.node.clone(),
            b: other.node.clone()
        }))
    }

    #[napi]
    pub fn gt(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Gt {
            a: self.node.clone(),
            b: other.node.clone()
        }))
    }

    #[napi]
    pub fn lt(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Lt {
            a: self.node.clone(),
            b: other.node.clone()
        }))
    }

    #[napi]
    pub fn ge(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Ge {
            a: self.node.clone(),
            b: other.node.clone()
        }))
    }

    #[napi]
    pub fn le(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Le {
            a: self.node.clone(),
            b: other.node.clone()
        }))
    }

    #[napi]
    pub fn matmul(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Matmul {
            a: self.node.clone(),
            b: other.node.clone()
        }))
    }

    #[napi]
    pub fn inverse(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Inverse {
            a: self.node.clone()
        }))
    }

    #[napi]
    pub fn det(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Det {
            a: self.node.clone()
        }))
    }

    #[napi]
    pub fn solve(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Solve {
            a: self.node.clone(),
            b: other.node.clone()
        }))
    }

    #[napi]
    pub fn neg(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Neg {
            a: self.node.clone()
        }))
    }

    #[napi]
    pub fn abs(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Abs {
            a: self.node.clone()
        }))
    }

    #[napi]
    pub fn sqrt(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Sqrt {
            a: self.node.clone()
        }))
    }

    #[napi]
    pub fn exp(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Exp {
            a: self.node.clone()
        }))
    }

    #[napi]
    pub fn tanh(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Tanh {
            a: self.node.clone()
        }))
    }

    #[napi]
    pub fn gelu(&self, approximate: Option<bool>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Gelu {
            a: self.node.clone(),
            approximate: approximate.unwrap_or(false),
        }))
    }

    #[napi]
    pub fn relu(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Relu {
            a: self.node.clone()
        }))
    }

    #[napi]
    pub fn erf(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Erf {
            a: self.node.clone()
        }))
    }

    #[napi]
    pub fn floor(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Floor {
            a: self.node.clone()
        }))
    }

    #[napi]
    pub fn ceil(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Ceil {
            a: self.node.clone()
        }))
    }

    #[napi]
    pub fn round(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Round {
            a: self.node.clone()
        }))
    }

    #[napi]
    pub fn sign(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Sign {
            a: self.node.clone()
        }))
    }

    #[napi]
    pub fn where_cond(&self, a: &LazyTensor, b: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Where {
            cond: self.node.clone(),
            a: a.node.clone(),
            b: b.node.clone(),
        }))
    }

    #[napi]
    pub fn argmax(&self, dim: u32) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Argmax {
            a: self.node.clone(),
            dim: dim as usize
        }))
    }

    #[napi]
    pub fn argmin(&self, dim: u32) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Argmin {
            a: self.node.clone(),
            dim: dim as usize
        }))
    }

    #[napi]
    pub fn cumsum(&self, dim: u32) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Cumsum {
            a: self.node.clone(),
            dim: dim as usize
        }))
    }

    #[napi]
    pub fn index_select(&self, dim: u32, indexes: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::IndexSelect {
            a: self.node.clone(),
            dim: dim as usize,
            indexes: indexes.node.clone(),
        }))
    }

    #[napi]
    pub fn scatter_add(&self, dim: u32, indexes: &LazyTensor, src: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::ScatterAdd {
            a: self.node.clone(),
            dim: dim as usize,
            indexes: indexes.node.clone(),
            src: src.node.clone(),
        }))
    }

    #[napi]
    pub fn gather(&self, dim: u32, indexes: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Gather {
            a: self.node.clone(),
            dim: dim as usize,
            indexes: indexes.node.clone(),
        }))
    }

    #[napi]
    pub fn cross_entropy(&self, target: &LazyTensor, ignore_index: i64) -> Result<Self> {
        lazy_ctor!(chunked_head_ce(&self.node, &target.node, ignore_index))
    }

    #[napi]
    pub fn scaled_dot_product_attention(
        &self,
        k: &LazyTensor,
        v: &LazyTensor,
        scale: f64,
        causal: bool,
        window: i64,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Sdpa {
            q: self.node.clone(),
            k: k.node.clone(),
            v: v.node.clone(),
            scale,
            causal,
            window: attention_window(window)?,
        }))
    }

    #[napi]
    pub fn kda_chunk(
        &self,
        k: &LazyTensor,
        v: &LazyTensor,
        log_decay: &LazyTensor,
        beta: &LazyTensor,
        scale: f64,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::KdaChunk {
            q: self.node.clone(),
            k: k.node.clone(),
            v: v.node.clone(),
            log_decay: log_decay.node.clone(),
            beta: beta.node.clone(),
            scale,
        }))
    }

    #[napi(js_name = "shortConv1d")]
    pub fn short_conv1d(&self, weight: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::ShortConv1d {
            x: self.node.clone(),
            weight: weight.node.clone(),
        }))
    }

    #[napi]
    pub fn position_embedding(&self, seq_len: u32) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::PositionEmbedding {
            weight: self.node.clone(),
            seq_len: seq_len as usize,
        }))
    }

    #[napi]
    pub fn rotary_embedding(&self, seq_len: u32, theta: f64, layout: String) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::RotaryEmbedding {
            x: self.node.clone(),
            seq_len: seq_len as usize,
            theta,
            offset: PositionOffset::Absolute,
            layout: rotary_layout(&layout)?,
        }))
    }

    #[napi]
    pub fn layer_norm(&self, weight: &LazyTensor, bias: &LazyTensor, eps: f64) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::LayerNorm {
            x: self.node.clone(),
            weight: weight.node.clone(),
            bias: bias.node.clone(),
            eps,
        }))
    }

    #[napi]
    pub fn rms_norm(&self, weight: Option<&LazyTensor>, eps: f64) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::RmsNorm {
            x: self.node.clone(),
            weight: weight.map(|value| value.node.clone()),
            eps,
        }))
    }

    #[napi]
    pub fn linear(&self, weight: &LazyTensor, bias: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Linear {
            x: self.node.clone(),
            weight: weight.node.clone(),
            bias: bias.node.clone(),
        }))
    }

    #[napi]
    pub fn quantized_linear(
        &self,
        weight: &LazyTensor,
        bias: Option<&LazyTensor>,
        encoding: String,
        rows: u32,
        columns: u32,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::QuantizedLinear {
            x: self.node.clone(),
            weight: weight.node.clone(),
            bias: bias.map(|value| value.node.clone()),
            codec: ggml_k_quant(&encoding)?,
            weight_shape: [rows as usize, columns as usize],
        }))
    }

    #[napi]
    pub fn quantized_embedding(
        &self,
        weight: &LazyTensor,
        encoding: String,
        rows: u32,
        columns: u32,
        padding_index: Option<u32>,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::QuantizedEmbedding {
            indexes: self.node.clone(),
            weight: weight.node.clone(),
            codec: ggml_k_quant(&encoding)?,
            weight_shape: [rows as usize, columns as usize],
            padding_index: padding_index.map(|value| value as usize),
        }))
    }

    #[napi(js_name = "conv1d")]
    pub fn conv_1d(
        &self,
        weight: &LazyTensor,
        stride: u32,
        padding: u32,
        dilation: u32,
        groups: u32,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Conv1d {
            x: self.node.clone(),
            w: weight.node.clone(),
            stride: stride as usize,
            padding: padding as usize,
            dilation: dilation as usize,
            groups: groups as usize,
        }))
    }

    #[napi(js_name = "conv2d")]
    pub fn conv_2d(
        &self,
        weight: &LazyTensor,
        stride: u32,
        padding: u32,
        dilation: u32,
        groups: u32,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Conv2d {
            x: self.node.clone(),
            w: weight.node.clone(),
            stride: stride as usize,
            padding: padding as usize,
            dilation: dilation as usize,
            groups: groups as usize,
        }))
    }

    #[napi]
    pub fn log(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Log {
            a: self.node.clone()
        }))
    }

    #[napi]
    pub fn sin(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Sin {
            a: self.node.clone()
        }))
    }

    #[napi]
    pub fn cos(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Cos {
            a: self.node.clone()
        }))
    }

    #[napi]
    pub fn pow(&self, exp: f64) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Pow {
            a: self.node.clone(),
            exp
        }))
    }

    #[napi]
    pub fn cast(&self, dtype: NativeDType) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Cast {
            a: self.node.clone(),
            dtype: dtype.into()
        }))
    }

    #[napi]
    pub fn sum(&self, dims: Vec<u32>, keepdims: bool) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Sum {
            a: self.node.clone(),
            dims: dims.into_iter().map(|dim| dim as usize).collect(),
            keepdims,
        }))
    }

    #[napi]
    pub fn prod(&self, dims: Vec<u32>, keepdims: bool) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Prod {
            a: self.node.clone(),
            dims: dims.into_iter().map(|dim| dim as usize).collect(),
            keepdims,
        }))
    }

    #[napi]
    pub fn mean(&self, dims: Vec<u32>, keepdims: bool) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Mean {
            a: self.node.clone(),
            dims: dims.into_iter().map(|dim| dim as usize).collect(),
            keepdims,
        }))
    }

    #[napi]
    pub fn max(&self, dims: Vec<u32>, keepdims: bool) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Max {
            a: self.node.clone(),
            dims: dims.into_iter().map(|dim| dim as usize).collect(),
            keepdims,
        }))
    }

    #[napi]
    pub fn min(&self, dims: Vec<u32>, keepdims: bool) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Min {
            a: self.node.clone(),
            dims: dims.into_iter().map(|dim| dim as usize).collect(),
            keepdims,
        }))
    }

    #[napi]
    pub fn reshape(&self, shape: Vec<u32>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Reshape {
            a: self.node.clone(),
            shape: shape
                .into_iter()
                .map(|dimension| dimension as usize)
                .collect(),
        }))
    }

    #[napi]
    pub fn permute(&self, dims: Vec<u32>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Permute {
            a: self.node.clone(),
            dims: dims.into_iter().map(|dim| dim as usize).collect(),
        }))
    }

    #[napi]
    pub fn slice(&self, ranges: Vec<Vec<u32>>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Slice {
            a: self.node.clone(),
            ranges: ranges
                .iter()
                .map(|range| (range[0] as usize, range[1] as usize, range[2] as usize))
                .collect(),
        }))
    }

    #[napi]
    pub fn concat(&self, other: &LazyTensor, dim: u32) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Concat {
            a: self.node.clone(),
            b: other.node.clone(),
            dim: dim as usize,
        }))
    }

    #[napi]
    pub fn broadcast_to(&self, shape: Vec<u32>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::BroadcastTo {
            a: self.node.clone(),
            shape: shape
                .into_iter()
                .map(|dimension| dimension as usize)
                .collect(),
        }))
    }

    #[napi]
    pub fn stop_gradient(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::StopGradient {
            a: self.node.clone()
        }))
    }

    #[napi]
    pub fn expose(&self, name: String) -> Result<Self> {
        if name.is_empty() {
            return Err(Error::new(
                Status::InvalidArg,
                "expose: name must be nonempty".to_string(),
            ));
        }
        lazy_ctor!(Node::new(NodeKind::Expose {
            a: self.node.clone(),
            name
        }))
    }

    /// Walks the lazy graph reachable from this node and returns each exposure's
    /// name and wrapped tensor in deterministic first-visit order. Duplicate
    /// names are a caller error.
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
                    return Err(Error::new(
                        Status::InvalidArg,
                        format!("expose: duplicate exposure name \"{name}\""),
                    ));
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

    #[napi]
    pub fn checkpoint(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Checkpoint {
            a: self.node.clone()
        }))
    }

    #[napi]
    pub fn vmap(&self, x: &LazyTensor, batched_x: &LazyTensor, dim: u32) -> Result<Self> {
        lazy_ctor!(effect_torch_autodiff::vmap(
            &self.node,
            &x.node,
            &batched_x.node,
            dim as usize,
        ))
    }

    #[napi]
    #[allow(clippy::too_many_arguments)]
    pub fn adamw_step(
        &self,
        grad: &LazyTensor,
        m: &LazyTensor,
        v: &LazyTensor,
        lr: &LazyTensor,
        c1: &LazyTensor,
        c2: &LazyTensor,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::AdamWStep {
            param: self.node.clone(),
            grad: grad.node.clone(),
            m: m.node.clone(),
            v: v.node.clone(),
            lr: lr.node.clone(),
            c1: c1.node.clone(),
            c2: c2.node.clone(),
            beta1,
            beta2,
            eps,
            weight_decay,
        }))
    }

    #[napi]
    pub fn adamw_out(&self, index: u8) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::AdamWOut {
            step: self.node.clone(),
            index
        }))
    }

    #[napi]
    #[allow(clippy::too_many_arguments)]
    pub fn sgd_step(
        &self,
        grad: &LazyTensor,
        velocity: &LazyTensor,
        first: &LazyTensor,
        lr: &LazyTensor,
        momentum: f64,
        dampening: f64,
        nesterov: bool,
        weight_decay: f64,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::SgdStep {
            param: self.node.clone(),
            grad: grad.node.clone(),
            velocity: velocity.node.clone(),
            first: first.node.clone(),
            lr: lr.node.clone(),
            momentum,
            dampening,
            nesterov,
            weight_decay,
        }))
    }

    #[napi]
    pub fn sgd_out(&self, index: u8) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::SgdOut {
            step: self.node.clone(),
            index
        }))
    }
}

/// Reverse-mode gradients of `loss` with respect to each tensor in `wrt`,
/// in the same order.
#[napi]
pub fn grad(loss: &LazyTensor, wrt: Vec<&LazyTensor>) -> Result<Vec<LazyTensor>> {
    let targets = wrt
        .iter()
        .map(|tensor| tensor.node.clone())
        .collect::<Vec<_>>();
    let gradients = effect_torch_autodiff::grad(&loss.node, &targets)
        .map_err(|message| Error::new(Status::GenericFailure, message))?;
    Ok(gradients
        .into_iter()
        .map(|node| LazyTensor { node })
        .collect())
}

/// Returns `true` because the CPU backend is available on every target.
#[napi]
pub fn is_available() -> bool {
    true
}

/// Runs a blocking compute closure on the N-API worker pool. Passes it the
/// token's cancellation state, or a new state, and the notify handle.
async fn run_compute<T: Send + 'static>(
    token: Option<&CancellationToken>,
    compute: impl FnOnce(&CancellationFlag, &CancellationState) -> Result<T> + Send + 'static,
) -> Result<T> {
    let state = token
        .map(|token| token.state.clone())
        .unwrap_or_else(|| Arc::new(CancellationState::new()));
    let notify = token.map(|token| token.notify.clone());
    effect_torch_napi::run_compute(state, notify, compute).await
}

/// Runs one stage of a larger transaction without completing the caller's
/// one-shot cancellation arbitration. The transaction completes arbitration
/// only after all stages are ready to publish.
async fn run_compute_pending<T: Send + 'static>(
    token: Option<&CancellationToken>,
    compute: impl FnOnce(&CancellationFlag, &CancellationState) -> Result<T> + Send + 'static,
) -> Result<T> {
    let state = token
        .map(|token| token.state.clone())
        .unwrap_or_else(|| Arc::new(CancellationState::new()));
    let worker_state = state.clone();
    let result = tokio::task::spawn_blocking(move || compute(worker_state.flag(), &worker_state))
        .await
        .map_err(effect_torch_napi::to_join_error)?;
    if state.flag().is_cancelled() {
        drop(result);
        Err(Error::new(Status::Cancelled, "operation aborted"))
    } else {
        result
    }
}

#[cfg(test)]
thread_local! {
    static CPU_PREPARATION_COUNTS: std::cell::Cell<(usize, usize, usize)> =
        const { std::cell::Cell::new((0, 0, 0)) };
}

#[cfg(test)]
fn cpu_preparation_counts() -> (usize, usize, usize) {
    CPU_PREPARATION_COUNTS.with(std::cell::Cell::get)
}

fn prepare_cpu_program(
    roots: &[Arc<Node>],
    options: CompileOptions,
    state_cursor: Option<StateCursorSlot>,
) -> err::Res<(PreparedProgram, Vec<executable::CpuGeneratedValue>)> {
    let mut request = ProgramRequest::from_roots(roots.to_vec(), options);
    if let Some(state_cursor) = state_cursor {
        request = request.with_state_cursor(state_cursor);
    }
    let program = request.prepare()?;
    #[cfg(test)]
    CPU_PREPARATION_COUNTS.with(|counts| {
        let (indexes, collections, gets) = counts.get();
        counts.set((indexes + 1, collections, gets));
    });
    let generated = executable::load_generated_values(&program.index)?;
    #[cfg(test)]
    CPU_PREPARATION_COUNTS.with(|counts| {
        let (indexes, collections, gets) = counts.get();
        counts.set((indexes, collections + 1, gets + generated.len()));
    });
    Ok((program, generated))
}

#[derive(Clone)]
struct ProgramInner {
    executable: Arc<executable::CpuExecutable>,
    slots: Vec<ProgramSlot>,
    generated_bindings: Vec<Value>,
}

#[derive(Clone)]
struct GeneratedBindingSignature {
    shape: Vec<usize>,
    dtype: DType,
    layout: Layout,
}

#[derive(Clone)]
struct CachedProgram {
    executable: Arc<executable::CpuExecutable>,
    slots: Vec<ProgramSlot>,
    generated: Vec<GeneratedBindingSignature>,
    generated_order: Vec<usize>,
}

#[derive(Default)]
struct ProgramCache {
    entries: HashMap<String, CachedProgram>,
    order: VecDeque<String>,
}

impl ProgramCache {
    fn get(&mut self, key: &str) -> Option<CachedProgram> {
        let entry = self.entries.get(key)?.clone();
        if let Some(index) = self.order.iter().position(|existing| existing == key) {
            self.order.remove(index);
        }
        self.order.push_back(key.to_string());
        Some(entry)
    }

    fn insert(&mut self, key: String, entry: CachedProgram) {
        const CAPACITY: usize = 64;
        if self.entries.contains_key(&key) {
            self.order.retain(|existing| existing != &key);
        }
        self.entries.insert(key.clone(), entry);
        self.order.push_back(key);
        while self.entries.len() > CAPACITY {
            if let Some(evicted) = self.order.pop_front() {
                self.entries.remove(&evicted);
            }
        }
    }
}

fn program_cache() -> &'static Mutex<ProgramCache> {
    static CACHE: LazyLock<Mutex<ProgramCache>> = LazyLock::new(Default::default);
    &CACHE
}

fn ordered_generated_bindings(
    program: &PreparedProgram,
    generated: &[executable::CpuGeneratedValue],
    order: &[usize],
) -> Option<Vec<Value>> {
    if generated.len() != program.index.leaves.len() || order.len() != generated.len() {
        return None;
    }
    order
        .iter()
        .map(|index| generated.get(*index).map(|value| value.value().clone()))
        .collect()
}

fn generated_signatures(values: &[Value]) -> Vec<GeneratedBindingSignature> {
    values
        .iter()
        .map(|value| GeneratedBindingSignature {
            shape: value.shape().to_vec(),
            dtype: value.dtype(),
            layout: value.tensor().layout.clone(),
        })
        .collect()
}

fn generated_match(values: &[Value], expected: &[GeneratedBindingSignature]) -> bool {
    values.len() == expected.len()
        && values.iter().zip(expected).all(|(value, expected)| {
            value.shape() == expected.shape
                && value.dtype() == expected.dtype
                && value.tensor().layout == expected.layout
        })
}

/// A compiled CPU program exported to JavaScript. `ProgramCache` stores up to
/// 64 executables in an LRU cache keyed by structural hash. Compiling the same
/// graph reuses the cached artifact. Each hit revalidates generated bindings
/// against the cached signature.
#[napi]
#[derive(Clone)]
pub struct Executable {
    inner: ProgramInner,
    state: Option<KvStateSchema>,
}

#[derive(Clone, Copy)]
struct KvStateSchema {
    max_tokens: usize,
    block_size: usize,
    kv_dtype: DType,
    window: Option<usize>,
    allows_window_eviction: bool,
    batch: usize,
    packed_rows_per_sequence: Option<usize>,
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    kda: KdaGeometry,
    conv: ConvGeometry,
    cursor_slot: u32,
    cursor_tensor: bool,
}

impl KvStateSchema {
    fn from_native(schema: NativeKvStateSchema, geometry: DecodeGeometry) -> Result<Self> {
        let max_tokens = schema.max_tokens as usize;
        let block_size = schema.block_size as usize;
        let batch = schema.batch as usize;
        let packed_rows_per_sequence = schema
            .packed_causal_chains
            .map(|layout| layout.rows_per_sequence as usize);
        if max_tokens == 0 || block_size == 0 || max_tokens % block_size != 0 {
            return Err(Error::new(
                Status::InvalidArg,
                "compile: KV max_tokens must be positive and divisible by block_size",
            ));
        }
        if batch == 0 {
            return Err(Error::new(
                Status::InvalidArg,
                "compile: KV batch must be positive",
            ));
        }
        if packed_rows_per_sequence == Some(0)
            || packed_rows_per_sequence.is_some_and(|rows| batch.checked_mul(rows).is_none())
        {
            return Err(Error::new(
                Status::InvalidArg,
                "compile: packed causal-chain rowsPerSequence must be positive and graph rows must not overflow",
            ));
        }
        let kv_dtype = schema.kv_dtype.into();
        if !matches!(kv_dtype, DType::F32 | DType::F16 | DType::BF16 | DType::U8) {
            return Err(Error::new(
                Status::InvalidArg,
                format!("compile: unsupported KV state dtype {}", kv_dtype.name()),
            ));
        }
        let requested_window = schema.window.map(|window| window as usize);
        if requested_window.is_some_and(|window| window == 0 || window > max_tokens) {
            return Err(Error::new(
                Status::InvalidArg,
                "compile: KV window must be in 1..=max_tokens",
            ));
        }
        let allows_window_eviction = geometry.allows_window_eviction;
        let window = if allows_window_eviction {
            requested_window
        } else {
            None
        };
        Ok(Self {
            max_tokens,
            block_size,
            kv_dtype,
            window,
            allows_window_eviction,
            batch,
            packed_rows_per_sequence,
            layers: geometry.layers,
            kv_heads: geometry.kv_heads,
            head_dim: geometry.head_dim,
            kda: geometry.kda,
            conv: geometry.conv,
            cursor_slot: geometry.cursor_slot,
            cursor_tensor: geometry.cursor_tensor,
        })
    }

    fn referenced_state_bytes(&self) -> std::result::Result<usize, String> {
        let checked = |values: &[usize], label: &str| {
            values
                .iter()
                .try_fold(1usize, |total, value| total.checked_mul(*value))
                .ok_or_else(|| format!("compile: {label} byte size overflow"))
        };
        let kv_elements = checked(
            &[
                self.layers,
                self.max_tokens,
                self.kv_heads,
                self.head_dim,
                2,
            ],
            "KV slab",
        )?;
        let mut bytes = kv_elements
            .checked_mul(self.kv_dtype.size_in_bytes())
            .ok_or_else(|| "compile: KV slab byte size overflow".to_string())?;
        if self.kv_dtype == DType::U8 {
            bytes = bytes
                .checked_add(
                    checked(
                        &[self.layers, self.max_tokens, self.kv_heads, 2],
                        "KV scale",
                    )?
                    .checked_mul(DType::F32.size_in_bytes())
                    .ok_or_else(|| "compile: KV scale byte size overflow".to_string())?,
                )
                .ok_or_else(|| "compile: KV state byte size overflow".to_string())?;
        }
        let kda_bytes = checked(
            &[
                self.batch,
                self.kda.layers,
                self.kda.heads,
                self.kda.head_dim,
                self.kda.value_dim,
            ],
            "KDA state",
        )?
        .checked_mul(self.kda.dtype.size_in_bytes())
        .ok_or_else(|| "compile: KDA state byte size overflow".to_string())?;
        let conv_bytes = checked(
            &[
                self.batch,
                self.conv.layers,
                self.conv.kernel.saturating_sub(1),
                self.conv.channels,
            ],
            "convolution state",
        )?
        .checked_mul(DType::F32.size_in_bytes())
        .ok_or_else(|| "compile: convolution state byte size overflow".to_string())?;
        bytes
            .checked_add(kda_bytes)
            .and_then(|bytes| bytes.checked_add(conv_bytes))
            .ok_or_else(|| "compile: total state byte size overflow".to_string())
    }
}

fn validate_tensor_input(input: &NativeTensor, slot: usize, declared: &ProgramSlot) -> Result<()> {
    let got = input.value_cloned()?;
    if got.shape() != declared.shape.as_slice()
        || got.dtype() != declared.dtype
        || !declared.device.is_cpu()
    {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "input slot {slot}: expected {}, got {}:{}@cpu",
                declared.signature(),
                got.shape()
                    .iter()
                    .map(|dimension| dimension.to_string())
                    .collect::<Vec<_>>()
                    .join("x"),
                got.dtype().name(),
            ),
        ));
    }
    Ok(())
}

fn validate_stateful_tensor_input(
    input: &NativeTensor,
    slot: usize,
    declared: &ProgramSlot,
) -> Result<()> {
    validate_tensor_input(input, slot, declared)
}

impl Executable {
    async fn execute_stateless(
        &self,
        inputs: Vec<&NativeTensor>,
        scalars: Vec<f64>,
        token: Option<&CancellationToken>,
    ) -> Result<Vec<NativeTensor>> {
        let signature = &self.inner.executable.signature;
        signature
            .validate_invocation_counts(inputs.len(), scalars.len(), 0, None)
            .map_err(|error| Error::new(Status::InvalidArg, error.to_string()))?;
        let mut tensors = inputs.iter();
        for (slot, declared) in self.inner.slots.iter().enumerate() {
            if !declared.scalar {
                validate_tensor_input(
                    tensors.next().expect("tensor count checked"),
                    slot,
                    declared,
                )?;
            }
        }
        let executable = self.inner.executable.clone();
        let generated = self.inner.generated_bindings.clone();
        let tensor_inputs = inputs
            .iter()
            .map(|input| input.value_cloned())
            .collect::<Result<Vec<_>>>()?;
        for (index, value) in tensor_inputs.iter().enumerate() {
            signature
                .validate_binding_metadata(
                    index,
                    value.dtype(),
                    value.tensor().placement(),
                    &value.tensor().layout,
                )
                .map_err(|error| Error::new(Status::InvalidArg, error.to_string()))?;
        }
        run_compute(token, move |cancelled, _state| {
            Ok(executable::execute_with_scalars(
                &executable,
                &tensor_inputs,
                &generated,
                &scalars,
                cancelled,
            )
            .map_err(to_napi_err)?
            .into_iter()
            .map(NativeTensor::wrap)
            .collect())
        })
        .await
    }
}

fn resolve_compile_options(native: Option<NativeCompileOptions>, stateful: bool) -> CompileOptions {
    let mut options = CompileOptions::from_environment();
    if let Some(native) = native {
        if let Some(optimize) = native.optimize {
            options.optimize = optimize;
        }
        if stateful || native.constant_weights.is_some() {
            options.inference = Some(InferenceOptions {
                constant_weights: native.constant_weights.unwrap_or(false),
            });
        }
    } else if stateful {
        options.inference = Some(InferenceOptions::default());
    }
    options
}

/// Compiles lazy roots into an [`Executable`]. A KV `state` schema first
/// specializes the graph for paged KV attention and a state cursor, then
/// includes that state plan in the executable. `cache_key` enables the
/// process-wide executable cache.
#[napi]
pub fn compile(
    roots: Vec<&LazyTensor>,
    options: Option<NativeCompileOptions>,
    state: Option<NativeKvStateSchema>,
    cache_key: Option<String>,
) -> Result<Executable> {
    let mut roots = roots
        .iter()
        .map(|tensor| tensor.node.clone())
        .collect::<Vec<_>>();
    if roots.is_empty() {
        return Err(Error::new(
            Status::InvalidArg,
            "compile: expected at least one root",
        ));
    }
    let compile_options = resolve_compile_options(options, state.is_some());
    let mut state_cursor = None;
    let state = match state {
        Some(native) => {
            if native.batch == 0
                || native.max_tokens == 0
                || native.block_size == 0
                || native.max_tokens % native.block_size != 0
            {
                return Err(Error::new(
                    Status::InvalidArg,
                    "compile: KV batch/max_tokens/block_size must be positive and max_tokens must be divisible by block_size",
                ));
            }
            if native
                .window
                .is_some_and(|window| window == 0 || window > native.max_tokens)
            {
                return Err(Error::new(
                    Status::InvalidArg,
                    "compile: KV window must be in 1..=max_tokens",
                ));
            }
            let batch = native.batch as usize;
            let window = native.window.map(|window| window as usize);
            if native.last_token_row.is_some() && native.output_selections.is_some() {
                return Err(Error::new(
                    Status::InvalidArg,
                    "compile: last_token_row and output_selections are mutually exclusive",
                ));
            }
            let output_selections = native.output_selections.clone().map_or_else(
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
                |selections| selections.into_iter().map(Into::into).collect(),
            );
            let layout =
                native
                    .packed_causal_chains
                    .as_ref()
                    .map_or(DecodeLayout::dense(batch), |packed| {
                        DecodeLayout::packed_causal_chains(batch, packed.rows_per_sequence as usize)
                    });
            let (rewritten, geometry) = specialize_decode_layout_outputs_with_attention(
                &roots,
                window,
                layout,
                &output_selections,
                native
                    .current_block_attention
                    .map(Into::into)
                    .unwrap_or_default(),
            )
            .map_err(|error| Error::new(Status::GenericFailure, error))?;
            roots = rewritten;
            state_cursor = Some(geometry.state_cursor());
            Some(KvStateSchema::from_native(native, geometry)?)
        }
        None => None,
    };
    let (program, generated_values) =
        prepare_cpu_program(&roots, compile_options, state_cursor).map_err(to_napi_err)?;
    let state_bytes = state
        .as_ref()
        .map(KvStateSchema::referenced_state_bytes)
        .transpose()
        .map_err(to_napi_err)?;
    let state_plan = state
        .zip(state_bytes)
        .map(|(state, bytes)| executable::CpuStatePlan {
            bytes,
            cursor_slot: state.cursor_slot,
            cursor_tensor: state.cursor_tensor,
            batch: state
                .packed_rows_per_sequence
                .map_or(state.batch, |rows| state.batch * rows),
        });
    let effective_cache_key = cache_key
        .filter(|_| {
            std::env::var_os("EFFECT_TORCH_NO_EXECUTABLE_CACHE").is_none()
                && !program
                    .options
                    .inference
                    .as_ref()
                    .is_some_and(|inference| inference.constant_weights)
        })
        .map(|key| format!("{key}|{:?}", program.options));
    if let Some(key) = effective_cache_key.as_deref() {
        let cached = program_cache()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(key);
        if let Some(cached) = cached {
            if cached.executable.signature == program.signature {
                if let Some(current) = (generated_values.len() == cached.generated_order.len())
                    .then(|| {
                        ordered_generated_bindings(
                            &program,
                            &generated_values,
                            &cached.generated_order,
                        )
                    })
                    .flatten()
                    .filter(|current| generated_match(current, &cached.generated))
                {
                    return Ok(Executable {
                        inner: ProgramInner {
                            executable: Arc::clone(&cached.executable),
                            slots: cached.slots,
                            generated_bindings: current,
                        },
                        state,
                    });
                }
            }
        }
    }
    let compilation = executable::compile_prepared(&program, &generated_values, state_plan)
        .map_err(to_napi_err)?;
    let slots = compilation.slots;
    if slots.iter().any(|slot| !slot.device.is_cpu()) {
        return Err(Error::new(
            Status::InvalidArg,
            "compile: graph contains an unsupported device",
        ));
    }
    if let Some(key) = effective_cache_key {
        let generated_positions = generated_values
            .iter()
            .enumerate()
            .map(|(index, generated)| (generated.slot_identity(), index))
            .collect::<HashMap<_, _>>();
        let generated_order = compilation
            .generated_slots
            .iter()
            .map(|slot| generated_positions.get(slot).copied())
            .collect::<Option<Vec<_>>>();
        let ordered = generated_order
            .as_deref()
            .and_then(|order| ordered_generated_bindings(&program, &generated_values, order));
        if let Some((generated_order, ordered)) = generated_order.zip(ordered) {
            let generated = generated_signatures(&ordered);
            if !generated_match(&compilation.generated_bindings, &generated) {
                return Err(Error::new(
                    Status::GenericFailure,
                    "compile: prepared generated binding order changed during lowering",
                ));
            }
            program_cache()
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .insert(
                    key,
                    CachedProgram {
                        executable: Arc::clone(&compilation.executable),
                        slots: slots.clone(),
                        generated,
                        generated_order,
                    },
                );
        }
    }
    Ok(Executable {
        inner: ProgramInner {
            executable: compilation.executable,
            slots,
            generated_bindings: compilation.generated_bindings,
        },
        state,
    })
}

/// Serializes tensors to a safetensors archive. Writes a temporary file and
/// renames it atomically after validating that names are unique.
#[napi]
pub async fn save_tensors(
    path: String,
    names: Vec<String>,
    tensors: Vec<&NativeTensor>,
    metadata: HashMap<String, String>,
    token: Option<&CancellationToken>,
) -> Result<()> {
    if names.len() != tensors.len() {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "save_tensors: got {} names for {} tensors",
                names.len(),
                tensors.len()
            ),
        ));
    }
    if names.is_empty() {
        return Err(Error::new(
            Status::InvalidArg,
            "save_tensors: expected at least one tensor",
        ));
    }
    let unique = names.iter().collect::<HashSet<_>>();
    if unique.len() != names.len() || names.iter().any(|name| name == "__metadata__") {
        return Err(Error::new(
            Status::InvalidArg,
            "save_tensors: tensor names must be unique and cannot be __metadata__",
        ));
    }
    let tensors = tensors
        .iter()
        .map(|tensor| tensor.value_cloned())
        .collect::<Result<Vec<_>>>()?;
    run_compute(token, move |cancelled, _state| {
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::new(Status::Cancelled, "operation aborted"));
        }
        let mut values = HashMap::with_capacity(names.len());
        for (name, tensor) in names.iter().zip(tensors) {
            values.insert(name.clone(), tensor);
        }
        safetensors::save(&values, &metadata, &path).map_err(to_napi_err)?;
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::new(Status::Cancelled, "operation aborted"));
        }
        Ok(())
    })
    .await
}

/// One named tensor of a loaded safetensors archive.
#[napi(object, object_from_js = false)]
pub struct NativeSafetensorsEntry {
    pub name: String,
    pub tensor: NativeTensor,
}

/// A loaded safetensors archive with entries sorted by name and metadata.
#[napi(object, object_from_js = false)]
pub struct NativeSafetensorsArchive {
    pub entries: Vec<NativeSafetensorsEntry>,
    pub metadata: HashMap<String, String>,
}

/// Loads a safetensors archive, rejecting unsupported dtypes and malformed
/// byte lengths.
#[napi]
pub async fn load_tensors(
    path: String,
    token: Option<&CancellationToken>,
) -> Result<NativeSafetensorsArchive> {
    run_compute(token, move |cancelled, _state| {
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::new(Status::Cancelled, "operation aborted"));
        }
        let archive = safetensors::load(&path).map_err(to_napi_err)?;
        if cancelled.load(Ordering::Acquire) {
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

/// Total bytes attributed to live [`NativeTensor`] values. V8's external
/// memory accounting mirrors this value.
#[napi]
pub fn external_memory_bytes() -> i64 {
    EXTERNAL_MEMORY_BYTES.load(Ordering::Relaxed)
}

const HASH_SEED: u64 = 0xcbf2_9ce4_8422_2325;
const HASH_PRIME: u64 = 0x0000_0100_0000_01b3;

fn chain_hash(previous: u64, tokens: &[u32]) -> u64 {
    let mut hash = previous;
    for token in tokens {
        for byte in token.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(HASH_PRIME);
        }
    }
    hash
}

struct RecurrentSnapshot {
    kda: Vec<Vec<f32>>,
    conv: Vec<Vec<f32>>,
}

fn capture_recurrent_snapshot(state: &SeqState) -> Option<RecurrentSnapshot> {
    let kda = state
        .kda_states
        .iter()
        .map(|tensor| Value(tensor.clone()).to_f32_vec().ok())
        .collect::<Option<Vec<_>>>()?;
    let conv = state
        .conv_states
        .iter()
        .map(|tensor| Value(tensor.clone()).to_f32_vec().ok())
        .collect::<Option<Vec<_>>>()?;
    Some(RecurrentSnapshot { kda, conv })
}

fn restore_recurrent_snapshot(state: &mut SeqState, snapshot: &RecurrentSnapshot) -> bool {
    if snapshot.kda.len() != state.kda_states.len()
        || snapshot.conv.len() != state.conv_states.len()
        || snapshot
            .kda
            .iter()
            .zip(&state.kda_states)
            .any(|(data, tensor)| data.len() != tensor.numel())
        || snapshot
            .conv
            .iter()
            .zip(&state.conv_states)
            .any(|(data, tensor)| data.len() != tensor.numel())
    {
        return false;
    }
    let kda = snapshot
        .kda
        .iter()
        .zip(&state.kda_states)
        .map(|(data, tensor)| Tensor::from_vec(data.clone(), tensor.shape().to_vec()))
        .collect();
    let conv = snapshot
        .conv
        .iter()
        .zip(&state.conv_states)
        .map(|(data, tensor)| Tensor::from_vec(data.clone(), tensor.shape().to_vec()))
        .collect();
    state.kda_states = kda;
    state.conv_states = conv;
    true
}

struct BlockStore {
    free: Vec<u32>,
    refcounts: Vec<u32>,
    hashes: Vec<Option<u64>>,
    by_hash: HashMap<u64, Vec<u32>>,
    snapshots: HashMap<u64, Arc<RecurrentSnapshot>>,
    lru: VecDeque<u32>,
}

#[derive(Default)]
struct DeferredPrefixMetadata {
    hashes: Vec<(u32, u64)>,
    snapshots: Vec<(u64, RecurrentSnapshot)>,
}

impl BlockStore {
    fn new(num_blocks: usize) -> Self {
        Self {
            free: (0..num_blocks as u32).rev().collect(),
            refcounts: vec![0; num_blocks],
            hashes: vec![None; num_blocks],
            by_hash: HashMap::new(),
            snapshots: HashMap::new(),
            lru: VecDeque::new(),
        }
    }

    fn is_cached(&self, block: u32) -> bool {
        self.refcounts[block as usize] == 0
            && match self.hashes[block as usize] {
                Some(hash) => self
                    .by_hash
                    .get(&hash)
                    .is_some_and(|blocks| blocks.contains(&block)),
                None => false,
            }
    }

    fn uncache(&mut self, block: u32, hash: u64) {
        if let Some(blocks) = self.by_hash.get_mut(&hash) {
            if let Some(index) = blocks.iter().position(|&candidate| candidate == block) {
                blocks.swap_remove(index);
            }
            if blocks.is_empty() {
                self.by_hash.remove(&hash);
                self.snapshots.remove(&hash);
            }
        }
    }

    fn cached(&self) -> usize {
        self.by_hash
            .values()
            .map(|blocks| {
                blocks
                    .iter()
                    .filter(|&&block| self.refcounts[block as usize] == 0)
                    .count()
            })
            .sum()
    }
}

struct PoolInner {
    k: Vec<pool::Slab>,
    v: Vec<pool::Slab>,
    scales: Vec<pool::Slab>,
    kv_dtype: DType,
    kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_tokens: usize,
    kda: KdaGeometry,
    conv: ConvGeometry,
    blocks: Mutex<BlockStore>,
}

impl PoolInner {
    fn ref_block(&self, block: u32) -> bool {
        let Ok(mut store) = self.blocks.lock() else {
            return false;
        };
        let Some(count) = store.refcounts.get_mut(block as usize) else {
            return false;
        };
        *count += 1;
        true
    }

    fn alloc_block(&self) -> Option<u32> {
        let mut store = self.blocks.lock().ok()?;
        if let Some(block) = store.free.pop() {
            store.refcounts[block as usize] = 1;
            store.hashes[block as usize] = None;
            return Some(block);
        }
        while let Some(candidate) = store.lru.pop_front() {
            if !store.is_cached(candidate) {
                continue;
            }
            let hash = store.hashes[candidate as usize].expect("cached block has a hash");
            store.uncache(candidate, hash);
            store.hashes[candidate as usize] = None;
            store.refcounts[candidate as usize] = 1;
            return Some(candidate);
        }
        None
    }

    fn take_block(&self, hash: u64) -> Option<u32> {
        let mut store = self.blocks.lock().ok()?;
        let block = *store.by_hash.get(&hash)?.first()?;
        store.refcounts[block as usize] += 1;
        Some(block)
    }

    fn unref_block(&self, block: u32) {
        if let Ok(mut store) = self.blocks.lock() {
            let count = &mut store.refcounts[block as usize];
            *count = count.saturating_sub(1);
            if *count == 0 {
                match store.hashes[block as usize] {
                    Some(_) => store.lru.push_back(block),
                    None => store.free.push(block),
                }
            }
        }
    }

    #[cfg(test)]
    fn discard_block(&self, block: u32) {
        if let Ok(mut store) = self.blocks.lock() {
            if let Some(hash) = store.hashes[block as usize] {
                store.uncache(block, hash);
                store.hashes[block as usize] = None;
            }
            let count = &mut store.refcounts[block as usize];
            *count = count.saturating_sub(1);
            if *count == 0 && !store.free.contains(&block) {
                store.free.push(block);
            }
        }
    }

    fn set_hash(&self, block: u32, hash: u64) {
        if let Ok(mut store) = self.blocks.lock() {
            store.hashes[block as usize] = Some(hash);
            store.by_hash.entry(hash).or_default().push(block);
        }
    }

    fn publish_snapshot(&self, hash: u64, snapshot: RecurrentSnapshot) {
        if let Ok(mut store) = self.blocks.lock() {
            if store.by_hash.contains_key(&hash) {
                store.snapshots.insert(hash, Arc::new(snapshot));
            }
        }
    }

    fn take_snapshot_prefix(
        &self,
        hashes: &[u64],
    ) -> Option<(Vec<u32>, u64, Arc<RecurrentSnapshot>)> {
        let mut store = self.blocks.lock().ok()?;
        let mut deepest = 0;
        for (index, hash) in hashes.iter().enumerate() {
            if !store.by_hash.contains_key(hash) {
                break;
            }
            if store.snapshots.contains_key(hash) {
                deepest = index + 1;
            }
        }
        if deepest == 0 {
            return None;
        }
        let mut blocks = Vec::with_capacity(deepest);
        for hash in &hashes[..deepest] {
            let block = *store.by_hash.get(hash)?.first()?;
            store.refcounts[block as usize] += 1;
            blocks.push(block);
        }
        let boundary_hash = hashes[deepest - 1];
        let snapshot = store.snapshots.get(&boundary_hash)?.clone();
        Some((blocks, boundary_hash, snapshot))
    }

    fn maybe_publish_recurrent_snapshot(&self, state: &SeqState) {
        if self.k.is_empty() || (self.kda.layers == 0 && self.conv.layers == 0) {
            return;
        }
        if state.cursor == 0 || state.cursor % self.block_size != 0 {
            return;
        }
        if let Some(snapshot) = capture_recurrent_snapshot(state) {
            self.publish_snapshot(state.last_hash, snapshot);
        }
    }

    fn available(&self) -> usize {
        self.blocks
            .lock()
            .map(|store| store.free.len() + store.cached())
            .unwrap_or(0)
    }

    fn cached_count(&self) -> usize {
        self.blocks.lock().map(|store| store.cached()).unwrap_or(0)
    }

    fn stage_recurrent_snapshot(&self, state: &SeqState, metadata: &mut DeferredPrefixMetadata) {
        if self.k.is_empty() || (self.kda.layers == 0 && self.conv.layers == 0) {
            return;
        }
        if state.cursor == 0 || state.cursor % self.block_size != 0 {
            return;
        }
        if let Some(snapshot) = capture_recurrent_snapshot(state) {
            metadata.snapshots.push((state.last_hash, snapshot));
        }
    }
}

struct SeqState {
    blocks: Vec<u32>,
    head: usize,
    cursor: usize,
    advance: usize,
    last_hash: u64,
    pending: Vec<u32>,
    // Sequence creation allocates per-layer recurrent state: [H, Dk, Dv] f32
    // per KDA layer and [K-1, C] f32 per short-conv layer.
    kda_states: Vec<Tensor>,
    conv_states: Vec<Tensor>,
}

impl SeqState {
    fn note_tokens(&mut self, pool: &PoolInner, tokens: &[u32]) {
        for (block, hash) in self.note_tokens_deferred(pool, tokens) {
            pool.set_hash(block, hash);
        }
    }

    fn note_tokens_deferred(&mut self, pool: &PoolInner, tokens: &[u32]) -> Vec<(u32, u64)> {
        let mut hashes = Vec::new();
        for (index, &token) in tokens.iter().enumerate() {
            self.pending.push(token);
            if self.pending.len() == pool.block_size {
                let hash = chain_hash(self.last_hash, &self.pending);
                self.last_hash = hash;
                self.pending.clear();
                let block_index = (self.cursor + index) / pool.block_size;
                if let Some(&block) = self.blocks.get(block_index) {
                    hashes.push((block, hash));
                }
            }
        }
        hashes
    }
}

/// Per-execution decode context implementing [`executable::CpuState`]. It
/// stages KV, KDA, and convolution updates during `run_command`, publishes
/// them on `commit`, and drops staged work on `rollback`.
struct KvContext {
    pool: Arc<PoolInner>,
    slots: Vec<Option<Arc<Mutex<SeqState>>>>,
    advances: Vec<usize>,
    packed: Option<PackedCausalRows>,
    window: Option<usize>,
    kda: KdaGeometry,
    conv: ConvGeometry,
    transaction: Mutex<Option<CpuStateTransaction>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PackedCausalRows {
    row_offsets: Vec<usize>,
    logical_rows: usize,
    row_to_physical: Vec<Option<usize>>,
    positions: Vec<usize>,
}

impl PackedCausalRows {
    fn build(
        graph_rows: usize,
        physical_batch: usize,
        physical_slots: &[usize],
        cursors: &[usize],
        lengths: &[usize],
    ) -> std::result::Result<Self, String> {
        if physical_slots.len() != cursors.len() || cursors.len() != lengths.len() {
            return Err(
                "executeSpeculative: inconsistent packed causal-chain metadata".to_string(),
            );
        }
        let mut row_offsets = Vec::with_capacity(lengths.len() + 1);
        let mut row_to_physical = Vec::with_capacity(graph_rows);
        let mut positions = Vec::with_capacity(graph_rows);
        row_offsets.push(0);
        for ((&physical, &cursor), &length) in physical_slots.iter().zip(cursors).zip(lengths) {
            if physical >= physical_batch || length == 0 {
                return Err(
                    "executeSpeculative: invalid packed causal-chain lane or length".to_string(),
                );
            }
            let end = row_to_physical.len().checked_add(length).ok_or_else(|| {
                "executeSpeculative: packed causal-chain row overflow".to_string()
            })?;
            if end > graph_rows || cursor.checked_add(length).is_none() {
                return Err(
                    "executeSpeculative: packed causal-chain layout exceeds its graph".to_string(),
                );
            }
            for offset in 0..length {
                row_to_physical.push(Some(physical));
                positions.push(cursor + offset);
            }
            row_offsets.push(end);
        }
        let logical_rows = row_to_physical.len();
        row_to_physical.resize(graph_rows, None);
        positions.resize(graph_rows, 0);
        Ok(Self {
            row_offsets,
            logical_rows,
            row_to_physical,
            positions,
        })
    }
}

impl KvContext {
    fn graph_rows(&self) -> usize {
        self.packed
            .as_ref()
            .map_or(self.slots.len(), |packed| packed.row_to_physical.len())
    }

    fn graph_row(&self, row: usize) -> Option<(usize, usize)> {
        match &self.packed {
            Some(packed) => {
                packed.row_to_physical[row].map(|physical| (physical, packed.positions[row]))
            }
            None => self.slots[row].as_ref().map(|_| (row, 0)),
        }
    }
}

struct CpuStateTransaction {
    frontiers: Vec<Option<usize>>,
    advances: Vec<usize>,
    cursors: Vec<usize>,
    eviction_starts: Vec<usize>,
}

impl executable::CpuState for Arc<KvContext> {
    fn begin(
        &self,
        executable: &executable::CpuExecutable,
        values: &[Value],
    ) -> std::result::Result<(), String> {
        let mut frontiers = Vec::with_capacity(self.slots.len());
        let mut advances = Vec::with_capacity(self.slots.len());
        let mut cursors = Vec::with_capacity(self.slots.len());
        let mut active_index = 0;
        for (lane, slot) in self.slots.iter().enumerate() {
            let Some(slot) = slot else {
                frontiers.push(None);
                advances.push(0);
                cursors.push(0);
                continue;
            };
            let mut state = slot
                .lock()
                .map_err(|error| format!("decode state lock poisoned: {error}"))?;
            let additional = self
                .pool
                .max_tokens
                .div_ceil(self.pool.block_size)
                .saturating_sub(state.blocks.len());
            state.blocks.reserve(additional);
            frontiers.push(Some(state.blocks.len()));
            let advance = if self.advances.len() == self.slots.len() {
                self.advances[lane]
            } else {
                self.advances[active_index]
            };
            advances.push(advance);
            cursors.push(state.cursor);
            active_index += 1;
        }
        let graph_cursors = self
            .packed
            .as_ref()
            .map_or_else(|| cursors.clone(), |packed| packed.positions.clone());
        if let Some(cursor) = executable.state_cursor {
            let value = values
                .get(cursor.index())
                .ok_or_else(|| "decode cursor staging value is unresolved".to_string())?;
            if value.dtype() != DType::I64 {
                return Err("decode cursor staging must use i64".to_string());
            }
            // SAFETY: the cursor staging value belongs to this invocation's
            // exclusive planned staging range. `begin` runs before any command
            // reads it.
            let mut destination = unsafe { CpuDestination::from_planned(value.tensor()) };
            destination.write::<i64, _>("decode cursor", &value.shape(), |output| {
                if output.len() == 1 && graph_cursors.len() == 1 {
                    output[0] = graph_cursors[0] as i64;
                } else {
                    assert_eq!(output.len(), graph_cursors.len());
                    for (output, cursor) in output.iter_mut().zip(&graph_cursors) {
                        *output = *cursor as i64;
                    }
                }
            })?;
        }
        for physical in &executable.physical {
            let executable::CpuPhysicalCommand::Encode(id) = *physical;
            let command = executable
                .instruction(id)
                .ok_or_else(|| format!("decode physical instruction {id} is unresolved"))?;
            let (op, _) = command
                .kind
                .operation()
                .ok_or_else(|| format!("decode physical instruction {id} references a boundary"))?;
            if !matches!(
                op,
                executable::CpuOp::KdaRecurrence { .. }
                    | executable::CpuOp::ConvState { .. }
                    | executable::CpuOp::KvAttention { .. }
            ) {
                continue;
            }
            let input = &values[command.inputs[0].value.index()];
            let shape = input.shape();
            let steps = shape
                .get(shape.len().saturating_sub(2))
                .copied()
                .ok_or_else(|| "decode state input has no token dimension".to_string())?;
            let invalid_advance = if let Some(packed) = &self.packed {
                steps != 1
                    || packed.logical_rows == 0
                    || packed.row_offsets.last().copied() != Some(packed.logical_rows)
            } else {
                self.slots
                    .iter()
                    .zip(&advances)
                    .any(|(slot, advance)| slot.is_some() && (*advance == 0 || *advance > steps))
            };
            if invalid_advance {
                return Err(format!(
                    "decode token advance must be in 1..={steps} for {}",
                    op.name()
                ));
            }
        }
        *self
            .transaction
            .lock()
            .map_err(|error| format!("decode transaction lock poisoned: {error}"))? =
            Some(CpuStateTransaction {
                frontiers,
                advances,
                cursors: graph_cursors,
                eviction_starts: vec![usize::MAX; self.slots.len()],
            });
        Ok(())
    }

    fn run_command(
        &self,
        command: &executable::CpuCommand,
        inputs: &[Value],
        staging: &[Value],
        outputs: &mut [CpuDestination<'_>],
        _scratch: &mut [CpuDestination<'_>],
        state_outputs: &mut [CpuDestination<'_>],
    ) -> std::result::Result<(), String> {
        let mut transaction = self
            .transaction
            .lock()
            .map_err(|error| format!("decode transaction lock poisoned: {error}"))?;
        let transaction = transaction
            .as_mut()
            .ok_or_else(|| "decode state command ran outside a transaction".to_string())?;
        let (op, _) = command
            .kind
            .operation()
            .ok_or_else(|| "state adapter received a boundary instruction".to_string())?;
        match op {
            executable::CpuOp::KdaRecurrence { scale, layer } => {
                prepare_kda_staging(self, transaction, *layer, inputs, staging)?;
                composed::kda_chunk_forward_into(
                    inputs[0].tensor(),
                    inputs[1].tensor(),
                    inputs[2].tensor(),
                    staging[1].tensor(),
                    staging[2].tensor(),
                    *scale,
                    Some(staging[0].tensor()),
                    &mut outputs[0],
                    Some(&mut state_outputs[0]),
                    None,
                )
            }
            executable::CpuOp::ConvState { layer } => {
                prepare_conv_staging(self, *layer, &staging[0])?;
                conv_state_into(
                    &inputs[0],
                    &inputs[1],
                    &staging[0],
                    &transaction.advances,
                    &mut outputs[0],
                    &mut state_outputs[0],
                )
            }
            executable::CpuOp::LastTokenRow { lane } => {
                let lane = *lane;
                if lane >= self.slots.len() {
                    return Err(format!(
                        "last token row: physical lane {lane} is out of range"
                    ));
                }
                if self.slots[lane].is_some() {
                    last_token_row_into(&inputs[0], transaction.advances[lane], &mut outputs[0])
                } else {
                    zero_last_token_row(&inputs[0], &mut outputs[0])
                }
            }
            executable::CpuOp::KvAttention {
                scale,
                layer,
                window,
                mode,
            } => kv_attention_into(
                self,
                *layer,
                &inputs[0],
                &inputs[1],
                &inputs[2],
                *scale,
                *window,
                *mode,
                &mut outputs[0],
                &mut transaction.eviction_starts,
            ),
            executable::CpuOp::RotaryEmbedding { theta, layout, .. } => {
                composed::rotary_forward_into(
                    inputs[0].tensor(),
                    &transaction.cursors,
                    *theta,
                    1.0,
                    *layout,
                    &mut outputs[0],
                )
            }
            _ => Err("state adapter received a non-state CPU command".to_string()),
        }
    }

    fn commit(
        &self,
        executable: &executable::CpuExecutable,
        values: &[Value],
    ) -> std::result::Result<(), String> {
        commit_recurrent_state(self, executable, values)?;
        {
            let transaction = self
                .transaction
                .lock()
                .map_err(|error| format!("decode transaction lock poisoned: {error}"))?;
            let transaction = transaction
                .as_ref()
                .ok_or_else(|| "decode commit has no active transaction".to_string())?;
            for (slot, start) in self.slots.iter().zip(&transaction.eviction_starts) {
                let Some(slot) = slot else { continue };
                let mut state = slot
                    .lock()
                    .map_err(|error| format!("decode state lock poisoned: {error}"))?;
                if *start != usize::MAX {
                    kv_evict(&self.pool, &mut state, *start);
                }
            }
        }
        *self
            .transaction
            .lock()
            .map_err(|error| format!("decode transaction lock poisoned: {error}"))? = None;
        Ok(())
    }

    fn rollback(&self) {
        let transaction = self
            .transaction
            .lock()
            .ok()
            .and_then(|mut transaction| transaction.take());
        let Some(transaction) = transaction else {
            return;
        };
        for (slot, frontier) in self.slots.iter().zip(transaction.frontiers) {
            if let (Some(slot), Some(frontier)) = (slot, frontier) {
                if let Ok(mut state) = slot.lock() {
                    for block in state.blocks.split_off(frontier) {
                        self.pool.unref_block(block);
                    }
                }
            }
        }
    }
}

fn tensor_element<T: Elem>(tensor: &Tensor, index: usize, operation: &str) -> err::Res<T> {
    let storage = T::storage_of(&tensor.buffer)
        .ok_or_else(|| format!("{operation}: tensor dtype mismatch"))?;
    Ok(storage[crate::tensor::source_index(&tensor.layout, index)])
}

fn write_masked_state_input<T: Elem>(
    source: &Tensor,
    advances: &[usize],
    destination: &mut CpuDestination<'_>,
) -> err::Res<()> {
    let shape = source.shape();
    let batch = shape[..shape.len().saturating_sub(3)]
        .iter()
        .product::<usize>();
    if shape.len() < 3 || batch != advances.len() || destination.shape() != shape {
        return Err("kda recurrence: state mask geometry does not match decode slots".to_string());
    }
    let steps = shape[shape.len() - 2];
    let width = shape[shape.len() - 1];
    let per_batch = source.numel() / advances.len();
    destination.write::<T, _>("kda masked input", shape, |output| {
        for (index, value) in output.iter_mut().enumerate() {
            let batch = index / per_batch;
            let step = (index / width) % steps;
            *value = if step < advances[batch] {
                tensor_element::<T>(source, index, "kda masked input")
                    .expect("source dtype and layout were validated")
            } else {
                T::default()
            };
        }
    })
}

fn write_kda_initial<T: Elem>(
    context: &KvContext,
    layer: usize,
    destination: &mut CpuDestination<'_>,
) -> err::Res<()> {
    let geometry = context.kda;
    let expected = [
        context.slots.len() * geometry.heads,
        geometry.head_dim,
        geometry.value_dim,
    ];
    if destination.shape() != expected || destination.dtype() != T::dtype() {
        return Err("kda recurrence: planned initial-state geometry is invalid".to_string());
    }
    let per_slot = geometry.heads * geometry.head_dim * geometry.value_dim;
    destination.write::<T, _>("kda initial state", &expected, |output| {
        output.fill(T::default());
        for (batch, slot) in context.slots.iter().enumerate() {
            let Some(slot) = slot else { continue };
            let state = slot.lock().expect("decode run owns an unpoisoned sequence");
            let Some(source) = state.kda_states.get(layer) else {
                continue;
            };
            assert_eq!(
                source.shape(),
                [geometry.heads, geometry.head_dim, geometry.value_dim]
            );
            assert_eq!(source.dtype(), T::dtype());
            for index in 0..per_slot {
                output[batch * per_slot + index] =
                    tensor_element::<T>(source, index, "kda initial state")
                        .expect("persistent KDA state was validated");
            }
        }
    })
}

fn prepare_kda_staging(
    context: &KvContext,
    transaction: &CpuStateTransaction,
    layer: u32,
    inputs: &[Value],
    staging: &[Value],
) -> err::Res<()> {
    if layer as usize >= context.kda.layers || staging.len() != 3 || inputs.len() != 5 {
        return Err("kda recurrence: invalid state command plan".to_string());
    }
    // SAFETY: `staging[0]` is this state command's planned staging range,
    // exclusively owned for the duration of `begin`/`run_command`.
    let mut initial = unsafe { CpuDestination::from_planned(staging[0].tensor()) };
    match initial.dtype() {
        DType::F32 => write_kda_initial::<f32>(context, layer as usize, &mut initial)?,
        DType::F64 => write_kda_initial::<f64>(context, layer as usize, &mut initial)?,
        dtype => {
            return Err(format!(
                "kda recurrence: unsupported planned state dtype {dtype}"
            ))
        }
    }
    for (source, target) in inputs[3..5].iter().zip(&staging[1..]) {
        // SAFETY: each staging tensor occupies a distinct planned range owned
        // by this state command for the duration of the write.
        let mut destination = unsafe { CpuDestination::from_planned(target.tensor()) };
        match source.dtype() {
            DType::F32 => write_masked_state_input::<f32>(
                source.tensor(),
                &transaction.advances,
                &mut destination,
            )?,
            DType::F64 => write_masked_state_input::<f64>(
                source.tensor(),
                &transaction.advances,
                &mut destination,
            )?,
            DType::F16 => write_masked_state_input::<half::f16>(
                source.tensor(),
                &transaction.advances,
                &mut destination,
            )?,
            DType::BF16 => write_masked_state_input::<half::bf16>(
                source.tensor(),
                &transaction.advances,
                &mut destination,
            )?,
            dtype => return Err(format!("kda recurrence: unsupported input dtype {dtype}")),
        }
    }
    Ok(())
}

fn prepare_conv_staging(context: &KvContext, layer: u32, staging: &Value) -> err::Res<()> {
    let geometry = context.conv;
    if layer as usize >= geometry.layers
        || staging.dtype() != DType::F32
        || staging.shape() != [context.slots.len(), geometry.kernel - 1, geometry.channels]
    {
        return Err("conv state: invalid state command plan".to_string());
    }
    let per_slot = (geometry.kernel - 1) * geometry.channels;
    // SAFETY: `staging` is this state command's planned staging range,
    // exclusively owned for the duration of the write.
    let mut destination = unsafe { CpuDestination::from_planned(staging.tensor()) };
    destination.write::<f32, _>("conv initial state", staging.tensor().shape(), |output| {
        output.fill(0.0);
        for (batch, slot) in context.slots.iter().enumerate() {
            let Some(slot) = slot else { continue };
            let state = slot.lock().expect("decode run owns an unpoisoned sequence");
            let Some(source) = state.conv_states.get(layer as usize) else {
                continue;
            };
            assert_eq!(source.shape(), [geometry.kernel - 1, geometry.channels]);
            assert_eq!(source.dtype(), DType::F32);
            for index in 0..per_slot {
                output[batch * per_slot + index] =
                    tensor_element::<f32>(source, index, "conv initial state")
                        .expect("persistent convolution state was validated");
            }
        }
    })?;
    Ok(())
}

fn last_token_row_into_impl<T: Elem>(
    source: &Tensor,
    advance: usize,
    destination: &mut CpuDestination<'_>,
) -> err::Res<()> {
    let shape = source.shape();
    if shape.len() != 3 || shape[0] != 1 || destination.shape() != [shape[2]] {
        return Err("last token row: state command geometry must be [1, T, V] -> [V]".to_string());
    }
    let (steps, width) = (shape[1], shape[2]);
    if advance == 0 || advance > steps {
        return Err(format!(
            "last token row: token advance must be in 1..={steps}, got {advance}"
        ));
    }
    let row = advance - 1;
    destination.write::<T, _>("last token row", &[width], |output| {
        for (column, value) in output.iter_mut().enumerate() {
            *value = tensor_element::<T>(source, row * width + column, "last token row")
                .expect("source dtype and layout were validated");
        }
    })
}

fn last_token_row_into(
    source: &Value,
    advance: usize,
    destination: &mut CpuDestination<'_>,
) -> err::Res<()> {
    match source.tensor().dtype() {
        DType::F32 => last_token_row_into_impl::<f32>(source.tensor(), advance, destination),
        DType::F64 => last_token_row_into_impl::<f64>(source.tensor(), advance, destination),
        DType::F16 => last_token_row_into_impl::<half::f16>(source.tensor(), advance, destination),
        DType::BF16 => {
            last_token_row_into_impl::<half::bf16>(source.tensor(), advance, destination)
        }
        DType::U8 => last_token_row_into_impl::<u8>(source.tensor(), advance, destination),
        DType::U32 => last_token_row_into_impl::<u32>(source.tensor(), advance, destination),
        DType::I64 => last_token_row_into_impl::<i64>(source.tensor(), advance, destination),
    }
}

fn zero_last_token_row_impl<T: Elem>(
    source: &Tensor,
    destination: &mut CpuDestination<'_>,
) -> err::Res<()> {
    let shape = source.shape();
    if shape.len() != 3 || shape[0] != 1 || destination.shape() != [shape[2]] {
        return Err("last token row: state command geometry must be [1, T, V] -> [V]".to_string());
    }
    destination.write::<T, _>("inactive last token row", &[shape[2]], |output| {
        output.fill(T::default());
    })
}

fn zero_last_token_row(source: &Value, destination: &mut CpuDestination<'_>) -> err::Res<()> {
    match source.dtype() {
        DType::F32 => zero_last_token_row_impl::<f32>(source.tensor(), destination),
        DType::F64 => zero_last_token_row_impl::<f64>(source.tensor(), destination),
        DType::F16 => zero_last_token_row_impl::<half::f16>(source.tensor(), destination),
        DType::BF16 => zero_last_token_row_impl::<half::bf16>(source.tensor(), destination),
        DType::U8 => zero_last_token_row_impl::<u8>(source.tensor(), destination),
        DType::U32 => zero_last_token_row_impl::<u32>(source.tensor(), destination),
        DType::I64 => zero_last_token_row_impl::<i64>(source.tensor(), destination),
    }
}

fn conv_state_into_impl<T: Elem>(
    x: &Tensor,
    weight: &Tensor,
    state: &Tensor,
    advances: &[usize],
    output: &mut CpuDestination<'_>,
    state_next: &mut CpuDestination<'_>,
) -> err::Res<()> {
    let shape = x.shape();
    let rank = shape.len();
    let batch = shape[..rank - 2].iter().product::<usize>();
    let steps = shape[rank - 2];
    let channels = shape[rank - 1];
    let kernel = weight.shape()[1];
    if batch != advances.len()
        || state.shape() != [batch, kernel - 1, channels]
        || state.dtype() != DType::F32
        || output.shape() != shape
        || output.dtype() != x.dtype()
        || state_next.shape() != state.shape()
        || state_next.dtype() != DType::F32
    {
        return Err("conv state: inconsistent planned destinations".to_string());
    }
    state_next.write::<f32, _>("conv next state", state.shape(), |next| {
        output.write::<T, _>("conv state output", shape, |out| {
            for batch_index in 0..batch {
                for step in 0..steps {
                    for channel in 0..channels {
                        let mut value = 0.0f64;
                        for tap in 0..kernel {
                            let window_index = step + tap;
                            let source = if window_index < kernel - 1 {
                                tensor_element::<f32>(
                                    state,
                                    (batch_index * (kernel - 1) + window_index) * channels
                                        + channel,
                                    "conv state",
                                )
                                .expect("convolution state dtype was validated")
                                    as f64
                            } else {
                                tensor_element::<T>(
                                    x,
                                    (batch_index * steps + window_index - (kernel - 1)) * channels
                                        + channel,
                                    "conv state input",
                                )
                                .expect("convolution input dtype was validated")
                                .to_f64() as f32 as f64
                            };
                            let coefficient = tensor_element::<T>(
                                weight,
                                channel * kernel + tap,
                                "conv state weight",
                            )
                            .expect("convolution weight dtype was validated")
                            .to_f64() as f32 as f64;
                            value += source * coefficient;
                        }
                        out[(batch_index * steps + step) * channels + channel] = T::from_f64(value);
                    }
                }
                for index in 0..kernel - 1 {
                    let window_index = advances[batch_index] + index;
                    for channel in 0..channels {
                        next[(batch_index * (kernel - 1) + index) * channels + channel] =
                            if window_index < kernel - 1 {
                                tensor_element::<f32>(
                                    state,
                                    (batch_index * (kernel - 1) + window_index) * channels
                                        + channel,
                                    "conv state",
                                )
                                .expect("convolution state dtype was validated")
                            } else {
                                tensor_element::<T>(
                                    x,
                                    (batch_index * steps + window_index - (kernel - 1)) * channels
                                        + channel,
                                    "conv state input",
                                )
                                .expect("convolution input dtype was validated")
                                .to_f64() as f32
                            };
                    }
                }
            }
        })
    })??;
    Ok(())
}

fn conv_state_into(
    x: &Value,
    weight: &Value,
    state: &Value,
    advances: &[usize],
    output: &mut CpuDestination<'_>,
    state_next: &mut CpuDestination<'_>,
) -> err::Res<()> {
    match x.dtype() {
        DType::F32 => conv_state_into_impl::<f32>(
            x.tensor(),
            weight.tensor(),
            state.tensor(),
            advances,
            output,
            state_next,
        ),
        DType::F64 => conv_state_into_impl::<f64>(
            x.tensor(),
            weight.tensor(),
            state.tensor(),
            advances,
            output,
            state_next,
        ),
        DType::F16 => conv_state_into_impl::<half::f16>(
            x.tensor(),
            weight.tensor(),
            state.tensor(),
            advances,
            output,
            state_next,
        ),
        DType::BF16 => conv_state_into_impl::<half::bf16>(
            x.tensor(),
            weight.tensor(),
            state.tensor(),
            advances,
            output,
            state_next,
        ),
        dtype => Err(format!("conv state: unsupported input dtype {dtype}")),
    }
}

fn recurrent_state_slices(
    value: &Value,
    batch: usize,
    per_batch: usize,
    squeeze_batch: bool,
) -> Vec<Tensor> {
    (0..batch)
        .map(|index| {
            let mut layout = value
                .tensor()
                .layout
                .narrow(0, index * per_batch, per_batch);
            if squeeze_batch {
                let mut shape = layout.shape().to_vec();
                let mut strides = layout.strides().to_vec();
                shape.remove(0);
                strides.remove(0);
                layout = Layout::new(shape, strides, layout.offset());
            }
            value.tensor().view(layout)
        })
        .collect()
}

fn commit_recurrent_state(
    context: &KvContext,
    executable: &executable::CpuExecutable,
    values: &[Value],
) -> err::Res<()> {
    let batch = context.slots.len();
    for physical in &executable.physical {
        let executable::CpuPhysicalCommand::Encode(id) = *physical;
        let command = executable
            .instruction(id)
            .ok_or_else(|| format!("decode physical instruction {id} is unresolved"))?;
        let Some(state_output) = command.state.first() else {
            continue;
        };
        let value = &values[state_output.value.index()];
        let (op, _) = command
            .kind
            .operation()
            .ok_or_else(|| "decode physical instruction references a boundary".to_string())?;
        match op {
            executable::CpuOp::KdaRecurrence { layer, .. } => {
                let per_batch = context.kda.heads;
                let updates = recurrent_state_slices(value, batch, per_batch, false);
                for (slot, update) in context.slots.iter().zip(updates) {
                    let Some(slot) = slot else { continue };
                    let mut state = slot
                        .lock()
                        .map_err(|error| format!("decode state lock poisoned: {error}"))?;
                    let layer = *layer as usize;
                    let target = state.kda_states.get_mut(layer).ok_or_else(|| {
                        format!("KDA state commit destination {layer} is missing")
                    })?;
                    let mut destination = target.destination().map_err(|_| {
                        format!("KDA state commit destination {layer} is still borrowed")
                    })?;
                    update.copy_into(&mut destination)?;
                }
            }
            executable::CpuOp::ConvState { layer } => {
                let updates = recurrent_state_slices(value, batch, 1, true);
                for (slot, update) in context.slots.iter().zip(updates) {
                    let Some(slot) = slot else { continue };
                    let mut state = slot
                        .lock()
                        .map_err(|error| format!("decode state lock poisoned: {error}"))?;
                    let layer = *layer as usize;
                    let target = state.conv_states.get_mut(layer).ok_or_else(|| {
                        format!("convolution state commit destination {layer} is missing")
                    })?;
                    let mut destination = target.destination().map_err(|_| {
                        format!("convolution state commit destination {layer} is still borrowed")
                    })?;
                    update.copy_into(&mut destination)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn kv_attention_into(
    context: &KvContext,
    layer: u32,
    q: &Value,
    k: &Value,
    v: &Value,
    scale: f64,
    window: Option<usize>,
    mode: KvAttentionMode,
    output: &mut CpuDestination<'_>,
    eviction_starts: &mut [usize],
) -> err::Res<()> {
    let dimensions = q.shape();
    let rank = dimensions.len();
    if rank < 4
        || q.dtype() != DType::F32
        || k.dtype() != DType::F32
        || v.dtype() != DType::F32
        || k.shape().len() != rank
        || v.shape().len() != rank
        || output.shape() != dimensions
        || output.dtype() != DType::F32
    {
        return Err("kv attention: expected matching rank-4-or-higher f32 tensors".to_string());
    }
    let batch = dimensions[..rank - 3].iter().product::<usize>();
    let (query_heads, tokens, width) = (
        dimensions[rank - 3],
        dimensions[rank - 2],
        dimensions[rank - 1],
    );
    let kv_heads = k.shape()[rank - 3];
    if k.shape()[..rank - 3] != dimensions[..rank - 3]
        || v.shape()[..rank - 3] != dimensions[..rank - 3]
        || v.shape()[rank - 3] != kv_heads
        || kv_heads == 0
        || !query_heads.is_multiple_of(kv_heads)
        || k.shape()[rank - 2] != tokens
        || v.shape()[rank - 2] != tokens
        || k.shape()[rank - 1] != width
        || v.shape()[rank - 1] != width
    {
        return Err("kv attention: incompatible grouped-query q/k/v shapes".to_string());
    }
    if batch != context.graph_rows() {
        return Err(format!(
            "kv attention: batch {batch} does not match {} graph rows",
            context.graph_rows()
        ));
    }
    let layer_index = layer as usize;
    output.write::<f32, _>("kv attention output", &dimensions, |out| -> err::Res<()> {
        out.fill(0.0);
        for batch_index in 0..batch {
            let Some((physical_lane, explicit_position)) = context.graph_row(batch_index) else {
                continue;
            };
            let Some(slot) = &context.slots[physical_lane] else {
                return Err(
                    "kv attention: packed row maps to an inactive physical lane".to_string()
                );
            };
            let mut state = slot
                .lock()
                .map_err(|error| format!("kv attention: sequence lock poisoned: {error}"))?;
            let planned_tokens = if context.packed.is_some() {
                state.advance
            } else {
                tokens
            };
            let (cursor, needed, start) = kv_prepare(
                &context.pool,
                &mut state,
                layer_index,
                context.window,
                mode,
                kv_heads,
                width,
                planned_tokens,
            )?;
            let advance = state.advance;
            let row_advance = if context.packed.is_some() { 1 } else { advance };
            let row_position = if context.packed.is_some() {
                if explicit_position < cursor || explicit_position >= needed {
                    return Err(
                        "kv attention: packed row position is outside its sequence advance"
                            .to_string(),
                    );
                }
                explicit_position
            } else {
                cursor
            };
            let physical = |position: usize| -> u32 {
                state.blocks[position / context.pool.block_size] * context.pool.block_size as u32
                    + (position % context.pool.block_size) as u32
            };

            let input_value =
                |value: &Value, token: usize, head: usize, column: usize| -> err::Res<f32> {
                    let logical =
                        ((batch_index * kv_heads + head) * tokens + token) * width + column;
                    tensor_element::<f32>(value.tensor(), logical, "kv attention input")
                };
            if context.pool.k[layer_index].dtype == DType::U8 {
                for (value, slab, scales) in [
                    (
                        k,
                        &context.pool.k[layer_index],
                        &context.pool.scales[2 * layer_index],
                    ),
                    (
                        v,
                        &context.pool.v[layer_index],
                        &context.pool.scales[2 * layer_index + 1],
                    ),
                ] {
                    scales.write(|mut scale_values| {
                        slab.write(|mut quantized| -> err::Res<()> {
                            for token in 0..row_advance {
                                let row = physical(row_position + token) as usize;
                                for head in 0..kv_heads {
                                    let mut maximum = 0.0f32;
                                    for column in 0..width {
                                        maximum = maximum
                                            .max(input_value(value, token, head, column)?.abs());
                                    }
                                    let scale = maximum / 127.0 + 1e-12;
                                    scale_values.set_f32(row * kv_heads + head, scale);
                                    for column in 0..width {
                                        let value = (input_value(value, token, head, column)?
                                            / scale)
                                            .round()
                                            .clamp(-127.0, 127.0)
                                            + 128.0;
                                        quantized.set_u8(
                                            (row * kv_heads + head) * width + column,
                                            value as u8,
                                        );
                                    }
                                }
                            }
                            Ok(())
                        })
                    })?;
                }
            } else {
                for (value, slab) in [
                    (k, &context.pool.k[layer_index]),
                    (v, &context.pool.v[layer_index]),
                ] {
                    slab.write(|mut destination| -> err::Res<()> {
                        for token in 0..row_advance {
                            let row = physical(row_position + token) as usize;
                            for head in 0..kv_heads {
                                for column in 0..width {
                                    destination.set_f32(
                                        (row * kv_heads + head) * width + column,
                                        input_value(value, token, head, column)?,
                                    );
                                }
                            }
                        }
                        Ok(())
                    })?;
                }
            }

            let mut attend = |keys: &pool::SlabReader<'_>,
                              values: &pool::SlabReader<'_>,
                              key_scales: Option<&pool::SlabReader<'_>>,
                              value_scales: Option<&pool::SlabReader<'_>>|
             -> err::Res<()> {
                let cached = |reader: &pool::SlabReader<'_>,
                              scales: Option<&pool::SlabReader<'_>>,
                              position: usize,
                              head: usize,
                              column: usize|
                 -> f32 {
                    let row = physical(position) as usize;
                    let raw = reader.get_f32((row * kv_heads + head) * width + column);
                    scales.map_or(raw, |scales| {
                        (raw - 128.0) * scales.get_f32(row * kv_heads + head)
                    })
                };
                for head in 0..query_heads {
                    let kv_head = head / (query_heads / kv_heads);
                    for query in 0..row_advance {
                        let end = if mode == KvAttentionMode::BidirectionalBlock {
                            needed
                        } else {
                            (row_position + query + 1).min(needed)
                        };
                        let begin = window.map_or(start, |window| {
                            if mode == KvAttentionMode::BidirectionalBlock {
                                row_position.saturating_sub(window).max(start)
                            } else {
                                end.saturating_sub(window).max(start)
                            }
                        });
                        let mut maximum = f64::NEG_INFINITY;
                        for position in begin..end {
                            let mut score = 0.0f64;
                            for column in 0..width {
                                let q_index = ((batch_index * query_heads + head) * tokens + query)
                                    * width
                                    + column;
                                score += tensor_element::<f32>(
                                    q.tensor(),
                                    q_index,
                                    "kv attention query",
                                )? as f64
                                    * cached(keys, key_scales, position, kv_head, column) as f64;
                            }
                            maximum = maximum.max(score * scale);
                        }
                        let mut denominator = 0.0f64;
                        for position in begin..end {
                            let mut score = 0.0f64;
                            for column in 0..width {
                                let q_index = ((batch_index * query_heads + head) * tokens + query)
                                    * width
                                    + column;
                                score += tensor_element::<f32>(
                                    q.tensor(),
                                    q_index,
                                    "kv attention query",
                                )? as f64
                                    * cached(keys, key_scales, position, kv_head, column) as f64;
                            }
                            denominator += (score * scale - maximum).exp();
                        }
                        for column in 0..width {
                            let mut result = 0.0f64;
                            for position in begin..end {
                                let mut score = 0.0f64;
                                for depth in 0..width {
                                    let q_index = ((batch_index * query_heads + head) * tokens
                                        + query)
                                        * width
                                        + depth;
                                    score += tensor_element::<f32>(
                                        q.tensor(),
                                        q_index,
                                        "kv attention query",
                                    )? as f64
                                        * cached(keys, key_scales, position, kv_head, depth) as f64;
                                }
                                result += (score * scale - maximum).exp()
                                    * cached(values, value_scales, position, kv_head, column)
                                        as f64;
                            }
                            out[((batch_index * query_heads + head) * tokens + query) * width
                                + column] = (result / denominator) as f32;
                        }
                    }
                }
                Ok(())
            };
            context.pool.k[layer_index].read(|keys| {
                context.pool.v[layer_index].read(|values| {
                    if context.pool.k[layer_index].dtype == DType::U8 {
                        context.pool.scales[2 * layer_index].read(|key_scales| {
                            context.pool.scales[2 * layer_index + 1].read(|value_scales| {
                                attend(&keys, &values, Some(&key_scales), Some(&value_scales))
                            })
                        })
                    } else {
                        attend(&keys, &values, None, None)
                    }
                })
            })?;
            eviction_starts[physical_lane] = if eviction_starts[physical_lane] == usize::MAX {
                start
            } else {
                eviction_starts[physical_lane].max(start)
            };
        }
        Ok(())
    })??;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn kv_prepare(
    pool: &Arc<PoolInner>,
    state: &mut SeqState,
    layer: usize,
    window: Option<usize>,
    mode: KvAttentionMode,
    heads: usize,
    width: usize,
    tokens: usize,
) -> err::Res<(usize, usize, usize)> {
    if layer >= pool.k.len() {
        return Err(format!(
            "kv attention: layer {layer} out of range for {} pool layers",
            pool.k.len()
        ));
    }
    if heads != pool.kv_heads || width != pool.head_dim {
        return Err(format!(
            "kv attention: layer {layer} shape [{heads}, {width}] does not match pool geometry [{}, {}]",
            pool.kv_heads, pool.head_dim
        ));
    }
    let cursor = state.cursor;
    let advance = state.advance;
    if advance == 0 || advance > tokens {
        return Err(format!(
            "kv attention: advance {advance} out of range for chunk length {tokens}"
        ));
    }
    let needed = cursor + advance;
    let start = window.map_or(0, |window| {
        if mode == KvAttentionMode::BidirectionalBlock {
            cursor.saturating_sub(window)
        } else {
            needed.saturating_sub(window)
        }
    });
    if needed.saturating_sub(start) > pool.max_tokens {
        return Err(format!(
            "kv attention: live context {} exceeds pool capacity {}",
            needed.saturating_sub(start),
            pool.max_tokens
        ));
    }
    while state.blocks.len() * pool.block_size < needed {
        let block = pool.alloc_block().ok_or_else(|| {
            format!(
                "kv attention: pool exhausted ({} tokens across live sequences)",
                pool.max_tokens
            )
        })?;
        state.blocks.push(block);
    }
    Ok((cursor, needed, start))
}

fn kv_evict(pool: &PoolInner, state: &mut SeqState, start: usize) {
    while state.head < state.blocks.len() && (state.head + 1) * pool.block_size <= start {
        pool.unref_block(state.blocks[state.head]);
        state.head += 1;
    }
}

/// Paged KV-cache pool for stateful decoding.
///
/// The pool owns per-layer key and value slabs of `max_tokens` positions in
/// `f32`, `f16`, `bf16`, or int8-quantized `u8`. Sequences lease
/// `block_size` pages. Identical token prefixes share cached blocks. The pool
/// also validates and allocates optional recurrent KDA and convolution state.
#[napi]
pub struct NativeKvPool {
    inner: Arc<PoolInner>,
}

#[napi]
impl NativeKvPool {
    /// Creates a pool. Geometries must be all zero or all positive.
    /// `max_tokens` must be a positive multiple of `block_size`, whose
    /// default is 16.
    #[napi(constructor)]
    pub fn new(
        layers: u32,
        kv_heads: u32,
        head_dim: u32,
        max_tokens: u32,
        block_size: Option<u32>,
        dtype: Option<NativeDType>,
        recurrent: Option<NativeRecurrentStateSchema>,
    ) -> Result<Self> {
        let dtype: DType = dtype.unwrap_or(NativeDType::F32).into();
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16 | DType::U8) {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "kv pool: dtype must be f32, f16, bf16 or u8 (int8-quantized), got {}",
                    dtype.name()
                ),
            ));
        }
        let (layers, kv_heads, head_dim, max_tokens) = (
            layers as usize,
            kv_heads as usize,
            head_dim as usize,
            max_tokens as usize,
        );
        let block_size = block_size.unwrap_or(16) as usize;
        let recurrent = recurrent.unwrap_or(NativeRecurrentStateSchema {
            kda_layers: 0,
            kda_heads: 0,
            kda_head_dim: 0,
            kda_value_dim: 0,
            conv_layers: 0,
            conv_channels: 0,
            conv_kernel: 0,
        });
        let kda = KdaGeometry {
            layers: recurrent.kda_layers as usize,
            heads: recurrent.kda_heads as usize,
            head_dim: recurrent.kda_head_dim as usize,
            value_dim: recurrent.kda_value_dim as usize,
            dtype: DType::F32,
        };
        let conv = ConvGeometry {
            layers: recurrent.conv_layers as usize,
            channels: recurrent.conv_channels as usize,
            kernel: recurrent.conv_kernel as usize,
        };
        if layers == 0 && (kv_heads != 0 || head_dim != 0) {
            return Err(Error::new(
                Status::InvalidArg,
                "kv pool: heads and head dim must be zero when layers is zero",
            ));
        }
        if layers > 0 && (kv_heads == 0 || head_dim == 0) {
            return Err(Error::new(
                Status::InvalidArg,
                "kv pool: layers, kv heads and head dim must be positive",
            ));
        }
        if block_size == 0 || max_tokens == 0 || max_tokens % block_size != 0 {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "kv pool: capacity {max_tokens} must be a positive multiple of block size {block_size}"
                ),
            ));
        }
        if (kda.layers == 0 && (kda.heads != 0 || kda.head_dim != 0 || kda.value_dim != 0))
            || (kda.layers > 0 && (kda.heads == 0 || kda.head_dim == 0 || kda.value_dim == 0))
        {
            return Err(Error::new(
                Status::InvalidArg,
                "kv pool: KDA geometry must be entirely zero or entirely positive",
            ));
        }
        if (conv.layers == 0 && (conv.channels != 0 || conv.kernel != 0))
            || (conv.layers > 0 && (conv.channels == 0 || conv.kernel == 0))
        {
            return Err(Error::new(
                Status::InvalidArg,
                "kv pool: convolution geometry must be entirely zero or entirely positive",
            ));
        }
        let mut k = Vec::with_capacity(layers);
        let mut v = Vec::with_capacity(layers);
        let mut scales = Vec::with_capacity(layers * 2);
        for _ in 0..layers {
            k.push(pool::Slab::new(max_tokens, kv_heads * head_dim, dtype));
            v.push(pool::Slab::new(max_tokens, kv_heads * head_dim, dtype));
            if dtype == DType::U8 {
                scales.push(pool::Slab::new(max_tokens, kv_heads, DType::F32));
                scales.push(pool::Slab::new(max_tokens, kv_heads, DType::F32));
            }
        }
        Ok(Self {
            inner: Arc::new(PoolInner {
                k,
                v,
                scales,
                kv_dtype: dtype,
                kv_heads,
                head_dim,
                block_size,
                max_tokens,
                kda,
                conv,
                blocks: Mutex::new(BlockStore::new(max_tokens / block_size)),
            }),
        })
    }

    #[napi(getter)]
    pub fn capacity(&self) -> u32 {
        self.inner.max_tokens as u32
    }

    #[napi(getter)]
    pub fn free_blocks(&self) -> u32 {
        self.inner.available() as u32
    }

    #[napi(getter)]
    pub fn cached_blocks(&self) -> u32 {
        self.inner.cached_count() as u32
    }

    #[napi]
    pub fn make_sequence(&self) -> NativeKvSequence {
        NativeKvSequence::new(self.inner.clone())
    }
}

/// One decode sequence's handle into a [`NativeKvPool`]. It holds leased
/// blocks, the committed cursor, pending tokens, and recurrent or convolution
/// state. `release`, drop, or JS finalization returns blocks to the pool
/// exactly once. `run_lock` serializes execution against release.
#[napi(custom_finalize)]
pub struct NativeKvSequence {
    pool: Arc<PoolInner>,
    state: Arc<Mutex<SeqState>>,
    run_lock: Arc<Mutex<()>>,
    released: Arc<AtomicBool>,
    finalize_releases: bool,
}

impl Clone for NativeKvSequence {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            state: self.state.clone(),
            run_lock: self.run_lock.clone(),
            released: self.released.clone(),
            finalize_releases: false,
        }
    }
}

impl NativeKvSequence {
    fn new(pool: Arc<PoolInner>) -> Self {
        let kda_states = (0..pool.kda.layers)
            .map(|_| {
                Tensor::zeros(
                    &[pool.kda.heads, pool.kda.head_dim, pool.kda.value_dim],
                    pool.kda.dtype,
                )
            })
            .collect();
        let conv_states = (0..pool.conv.layers)
            .map(|_| {
                Tensor::zeros(
                    &[pool.conv.kernel.saturating_sub(1), pool.conv.channels],
                    DType::F32,
                )
            })
            .collect();
        Self {
            state: Arc::new(Mutex::new(SeqState {
                blocks: Vec::with_capacity(pool.max_tokens / pool.block_size),
                head: 0,
                cursor: 0,
                advance: 0,
                last_hash: HASH_SEED,
                pending: Vec::with_capacity(pool.block_size),
                kda_states,
                conv_states,
            })),
            pool,
            run_lock: Arc::new(Mutex::new(())),
            released: Arc::new(AtomicBool::new(false)),
            finalize_releases: true,
        }
    }

    fn return_blocks(&self) {
        if self.released.swap(true, Ordering::SeqCst) {
            return;
        }
        let _run_guard = self
            .run_lock
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let head = state.head;
        for block in state.blocks.split_off(head) {
            self.pool.unref_block(block);
        }
        state.cursor = 0;
        state.advance = 0;
        state.last_hash = HASH_SEED;
        state.pending.clear();
    }
}

impl ObjectFinalize for NativeKvSequence {
    fn finalize(self, _env: Env) -> Result<()> {
        if self.finalize_releases {
            self.return_blocks();
        }
        Ok(())
    }
}

impl Drop for NativeKvSequence {
    fn drop(&mut self) {
        if self.finalize_releases {
            self.return_blocks();
        }
    }
}

#[napi]
impl NativeKvSequence {
    #[napi(getter)]
    pub fn cursor(&self) -> u32 {
        self.state
            .lock()
            .map(|state| state.cursor as u32)
            .unwrap_or(0)
    }

    #[napi]
    pub fn release(&self) {
        self.return_blocks();
    }

    #[napi]
    pub fn prefill_match(&self, tokens: Vec<u32>) -> Result<u32> {
        let _run_guard = self.run_lock.lock().map_err(|error| {
            Error::new(
                Status::GenericFailure,
                format!("kv sequence lock poisoned: {error}"),
            )
        })?;
        if self.released.load(Ordering::SeqCst) {
            return Err(Error::new(
                Status::GenericFailure,
                "kv sequence is released",
            ));
        }
        let mut state = self.state.lock().map_err(|error| {
            Error::new(
                Status::GenericFailure,
                format!("kv sequence lock poisoned: {error}"),
            )
        })?;
        if state.cursor > 0 || !state.blocks.is_empty() {
            return Err(Error::new(
                Status::GenericFailure,
                "prefill match: sequence already holds tokens",
            ));
        }
        let recurrent = self.pool.kda.layers > 0 || self.pool.conv.layers > 0;
        if recurrent && self.pool.k.is_empty() {
            return Ok(0);
        }
        let matchable = tokens.len().saturating_sub(1) / self.pool.block_size;
        let mut hash = HASH_SEED;
        if recurrent {
            let mut hashes = Vec::with_capacity(matchable);
            for index in 0..matchable {
                hash = chain_hash(
                    hash,
                    &tokens[index * self.pool.block_size..(index + 1) * self.pool.block_size],
                );
                hashes.push(hash);
            }
            let Some((blocks, boundary_hash, snapshot)) = self.pool.take_snapshot_prefix(&hashes)
            else {
                return Ok(0);
            };
            let matched = blocks.len();
            if !restore_recurrent_snapshot(&mut state, &snapshot) {
                for block in blocks {
                    self.pool.unref_block(block);
                }
                return Ok(0);
            }
            state.blocks = blocks;
            state.last_hash = boundary_hash;
            state.cursor = matched * self.pool.block_size;
            return Ok(state.cursor as u32);
        }
        for index in 0..matchable {
            let next = chain_hash(
                hash,
                &tokens[index * self.pool.block_size..(index + 1) * self.pool.block_size],
            );
            match self.pool.take_block(next) {
                Some(block) => {
                    state.blocks.push(block);
                    hash = next;
                }
                None => break,
            }
        }
        state.last_hash = hash;
        state.cursor = state.blocks.len() * self.pool.block_size;
        Ok(state.cursor as u32)
    }
}

fn validate_execution_mode(
    stateful: bool,
    scalar_count: usize,
    sequence_count: Option<usize>,
    slots: Option<&[u32]>,
    active_mask: Option<&[bool]>,
    valid_lengths: Option<&[u32]>,
    advances: Option<&[u32]>,
    token_count: Option<usize>,
) -> Result<()> {
    if stateful {
        if scalar_count != 0 {
            return Err(Error::new(
                Status::InvalidArg,
                "execute: stateful executables do not accept scalar inputs",
            ));
        }
        if sequence_count.is_none()
            || slots.is_none()
            || active_mask.is_none()
            || valid_lengths.is_none()
            || advances.is_none()
            || token_count.is_none()
        {
            Err(Error::new(
                Status::InvalidArg,
                "execute: stateful executables require sequences, slots, activeMask, validLengths, advances, and tokens",
            ))
        } else {
            Ok(())
        }
    } else if sequence_count.is_some()
        || slots.is_some()
        || active_mask.is_some()
        || valid_lengths.is_some()
        || advances.is_some()
        || token_count.is_some()
    {
        Err(Error::new(
            Status::InvalidArg,
            "execute: stateless executables do not accept state",
        ))
    } else {
        Ok(())
    }
}

fn validate_sampled_execution_mode(
    stateful: bool,
    sequence_count: usize,
    token_count: usize,
    sampling_count: usize,
) -> Result<()> {
    if !stateful {
        return Err(Error::new(
            Status::InvalidArg,
            "executeSampled: requires a stateful executable",
        ));
    }
    if sequence_count != token_count {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "executeSampled: expected one token list per sequence, got {token_count} for {sequence_count} sequences"
            ),
        ));
    }
    if sampling_count != sequence_count {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "executeSampled: expected one sampling options object per active sequence/output, got {sampling_count} for {sequence_count} sequences"
            ),
        ));
    }
    Ok(())
}

fn validate_pool_schema(schema: &KvStateSchema, pool: &PoolInner) -> Result<()> {
    if schema
        .window
        .is_some_and(|window| window == 0 || window > schema.max_tokens)
        || pool.max_tokens != schema.max_tokens
        || pool.block_size != schema.block_size
        || pool.kv_dtype != schema.kv_dtype
        || pool.k.len() != schema.layers
        || pool.kv_heads != schema.kv_heads
        || pool.head_dim != schema.head_dim
        || pool.kda != schema.kda
        || pool.conv != schema.conv
    {
        return Err(Error::new(
            Status::InvalidArg,
            "execute: pool capacity/block size/dtype/geometry does not match the compiled state schema",
        ));
    }
    Ok(())
}

fn validate_recurrent_state_schema(schema: &KvStateSchema, state: &SeqState) -> Result<()> {
    if (!state.kda_states.is_empty() && state.kda_states.len() != schema.kda.layers)
        || state.kda_states.iter().any(|tensor| {
            tensor.shape() != [schema.kda.heads, schema.kda.head_dim, schema.kda.value_dim]
                || tensor.dtype() != schema.kda.dtype
        })
        || (!state.conv_states.is_empty() && state.conv_states.len() != schema.conv.layers)
        || state.conv_states.iter().any(|tensor| {
            tensor.shape() != [schema.conv.kernel.saturating_sub(1), schema.conv.channels]
                || tensor.dtype() != DType::F32
        })
    {
        return Err(Error::new(
            Status::InvalidArg,
            "execute: sequence recurrent geometry/dtype does not match the compiled state schema",
        ));
    }
    Ok(())
}

fn validate_fixed_lanes(
    schema: &KvStateSchema,
    sequences: usize,
    slots: &[u32],
    active_mask: &[bool],
    valid_lengths: &[u32],
    advances: &[u32],
    tokens: &[Vec<u32>],
) -> Result<()> {
    if sequences == 0
        || sequences > schema.batch
        || slots.len() != sequences
        || tokens.len() != sequences
    {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "execute: executable accepts 1..={} compact sequences with matching slots and token rows, got {sequences}",
                schema.batch
            ),
        ));
    }
    if active_mask.len() != schema.batch
        || valid_lengths.len() != schema.batch
        || advances.len() != schema.batch
    {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "execute: activeMask, validLengths, and advances must have fixed width {}",
                schema.batch
            ),
        ));
    }
    let mut seen = vec![false; schema.batch];
    for (request, &slot) in slots.iter().enumerate() {
        let slot = slot as usize;
        if slot >= schema.batch || seen[slot] {
            return Err(Error::new(
                Status::InvalidArg,
                "execute: slots must be unique physical lane IDs in range",
            ));
        }
        seen[slot] = true;
        if tokens[request].is_empty() || tokens[request].len() != advances[slot] as usize {
            return Err(Error::new(Status::InvalidArg, format!("execute: token row {request} must contain exactly the valid tokens for slot {slot}")));
        }
    }
    for lane in 0..schema.batch {
        if active_mask[lane] != seen[lane]
            || valid_lengths[lane] != advances[lane]
            || (seen[lane] && advances[lane] == 0)
            || (!seen[lane] && advances[lane] != 0)
        {
            return Err(Error::new(
                Status::InvalidArg,
                format!("execute: inconsistent fixed-lane metadata at slot {lane}"),
            ));
        }
    }
    Ok(())
}

fn speculative_probabilities(
    value: &Value,
    row: Option<(usize, usize)>,
    options: SamplingOptions,
    cancelled: &AtomicBool,
) -> std::result::Result<Vec<f64>, String> {
    let tensor = value.tensor();
    let (length, offset, stride) = match row {
        None if tensor.layout.rank() == 1 => (
            tensor.shape()[0],
            tensor.layout.offset(),
            tensor.layout.strides()[0],
        ),
        Some((lane, token)) if tensor.layout.rank() == 3 => {
            let shape = tensor.shape();
            if lane >= shape[0] || token >= shape[1] {
                return Err("executeSpeculative: logits row is out of range".to_string());
            }
            (
                shape[2],
                tensor.layout.offset()
                    + lane * tensor.layout.strides()[0]
                    + token * tensor.layout.strides()[1],
                tensor.layout.strides()[2],
            )
        }
        _ => {
            return Err(
                "executeSpeculative: proposer logits must be [V] and target logits [B,T,V]"
                    .to_string(),
            )
        }
    };
    macro_rules! probabilities {
        ($values:expr) => {
            effective_probabilities(
                length,
                |index| $values[offset + index * stride].to_f64(),
                options,
                || cancelled.load(Ordering::Acquire),
            )
        };
    }
    match &tensor.buffer {
        CpuBuffer::F16(values) => probabilities!(values),
        CpuBuffer::BF16(values) => probabilities!(values),
        CpuBuffer::F32(values) => probabilities!(values),
        CpuBuffer::F64(values) => probabilities!(values),
        _ => Err(format!(
            "executeSpeculative: logits must be floating point, got {}",
            tensor.dtype()
        )),
    }
}

fn speculative_normalized_row(
    value: &Value,
    row: (usize, usize),
) -> std::result::Result<Vec<f64>, String> {
    let tensor = value.tensor();
    let shape = tensor.shape();
    if tensor.layout.rank() != 3 || row.0 >= shape[0] || row.1 >= shape[1] {
        return Err("executeSpeculative: probability row is out of range".to_string());
    }
    let length = shape[2];
    let offset = tensor.layout.offset()
        + row.0 * tensor.layout.strides()[0]
        + row.1 * tensor.layout.strides()[1];
    let stride = tensor.layout.strides()[2];
    macro_rules! values {
        ($values:expr) => {
            Ok((0..length)
                .map(|index| $values[offset + index * stride].to_f64())
                .collect())
        };
    }
    match &tensor.buffer {
        CpuBuffer::F16(values) => values!(values),
        CpuBuffer::BF16(values) => values!(values),
        CpuBuffer::F32(values) => values!(values),
        CpuBuffer::F64(values) => values!(values),
        _ => Err(format!(
            "executeSpeculative: probabilities must be floating point, got {}",
            tensor.dtype()
        )),
    }
}

fn speculative_sample_logits(
    value: &Value,
    row: (usize, usize),
    options: SamplingOptions,
    cancelled: &AtomicBool,
) -> std::result::Result<u32, String> {
    let tensor = value.tensor();
    let shape = tensor.shape();
    if tensor.layout.rank() != 3 || row.0 >= shape[0] || row.1 >= shape[1] {
        return Err("executeSpeculative: target logits row is out of range".to_string());
    }
    let length = shape[2];
    let offset = tensor.layout.offset()
        + row.0 * tensor.layout.strides()[0]
        + row.1 * tensor.layout.strides()[1];
    let stride = tensor.layout.strides()[2];
    macro_rules! sample {
        ($values:expr) => {
            sample_logits(
                length,
                |index| $values[offset + index * stride].to_f64(),
                options,
                || cancelled.load(Ordering::Acquire),
            )
        };
    }
    match &tensor.buffer {
        CpuBuffer::F16(values) => sample!(values),
        CpuBuffer::BF16(values) => sample!(values),
        CpuBuffer::F32(values) => sample!(values),
        CpuBuffer::F64(values) => sample!(values),
        _ => Err(format!(
            "executeSpeculative: logits must be floating point, got {}",
            tensor.dtype()
        )),
    }
}

fn speculative_token_input(
    dtype: DType,
    batch: usize,
    tokens: &[u32],
    steps: usize,
) -> std::result::Result<Value, String> {
    if tokens.len() != batch * steps {
        return Err("executeSpeculative: internal token input width mismatch".to_string());
    }
    let tensor = match dtype {
        DType::I64 => Tensor::from_vec(
            tokens.iter().map(|&token| token as i64).collect(),
            vec![batch, steps],
        ),
        DType::U32 => Tensor::from_vec(tokens.to_vec(), vec![batch, steps]),
        dtype => {
            return Err(format!(
                "executeSpeculative: token input must be i64 or u32, got {}",
                dtype.name()
            ))
        }
    };
    Ok(Value(tensor))
}

fn clone_speculative_state(
    pool: &Arc<PoolInner>,
    sequence_state: &Arc<Mutex<SeqState>>,
) -> std::result::Result<Arc<Mutex<SeqState>>, String> {
    let state = sequence_state
        .lock()
        .map_err(|error| format!("kv sequence lock poisoned: {error}"))?;
    let mut blocks = Vec::with_capacity(state.blocks.len() + 1);
    for &block in &state.blocks {
        if !pool.ref_block(block) {
            for block in blocks {
                pool.unref_block(block);
            }
            return Err("executeSpeculative: failed to retain a KV block".to_string());
        }
        blocks.push(block);
    }
    Ok(Arc::new(Mutex::new(SeqState {
        blocks,
        head: state.head,
        cursor: state.cursor,
        advance: 0,
        last_hash: state.last_hash,
        pending: state.pending.clone(),
        kda_states: state.kda_states.clone(),
        conv_states: state.conv_states.clone(),
    })))
}

fn discard_speculative_states(pool: &PoolInner, states: &[Arc<Mutex<SeqState>>]) {
    for state in states {
        if let Ok(mut state) = state.lock() {
            for block in state.blocks.drain(..) {
                pool.unref_block(block);
            }
        }
    }
}

fn speculative_note_tokens(
    state: &mut SeqState,
    pool: &PoolInner,
    tokens: &[u32],
) -> Vec<(u32, u64)> {
    state.note_tokens_deferred(pool, tokens)
}

#[napi]
impl Executable {
    #[napi(getter)]
    pub fn stateful(&self) -> bool {
        self.state.is_some()
    }

    #[napi(getter)]
    pub fn batch(&self) -> u32 {
        self.state.map_or(0, |state| state.batch as u32)
    }

    #[napi(getter)]
    pub fn allows_window_eviction(&self) -> bool {
        self.state.is_some_and(|state| state.allows_window_eviction)
    }

    #[napi(getter)]
    pub fn diagnostics(&self) -> NativeExecutableDiagnostics {
        executable_diagnostics(&self.inner.executable.diagnostics)
    }

    #[napi(getter)]
    pub fn layers(&self) -> u32 {
        self.state.map_or(0, |state| state.layers as u32)
    }

    #[napi(getter)]
    pub fn kv_heads(&self) -> u32 {
        self.state.map_or(0, |state| state.kv_heads as u32)
    }

    #[napi(getter)]
    pub fn head_dim(&self) -> u32 {
        self.state.map_or(0, |state| state.head_dim as u32)
    }

    #[napi(getter)]
    pub fn kda_layers(&self) -> u32 {
        self.state.map_or(0, |state| state.kda.layers as u32)
    }

    #[napi(getter)]
    pub fn kda_heads(&self) -> u32 {
        self.state.map_or(0, |state| state.kda.heads as u32)
    }

    #[napi(getter)]
    pub fn kda_head_dim(&self) -> u32 {
        self.state.map_or(0, |state| state.kda.head_dim as u32)
    }

    #[napi(getter)]
    pub fn kda_value_dim(&self) -> u32 {
        self.state.map_or(0, |state| state.kda.value_dim as u32)
    }

    #[napi(getter)]
    pub fn conv_layers(&self) -> u32 {
        self.state.map_or(0, |state| state.conv.layers as u32)
    }

    #[napi(getter)]
    pub fn conv_channels(&self) -> u32 {
        self.state.map_or(0, |state| state.conv.channels as u32)
    }

    #[napi(getter)]
    pub fn conv_kernel(&self) -> u32 {
        self.state.map_or(0, |state| state.conv.kernel as u32)
    }

    /// Runs the program asynchronously on the worker pool.
    ///
    /// Stateless executables take `inputs` and optional `scalars`. Stateful
    /// decode executables also take one [`NativeKvSequence`] and its new tokens
    /// for each batch lane. They run under a `CpuState` transaction and commit
    /// cache updates only if not cancelled. Cancellation returns
    /// `Status::Cancelled` and leaves all sequence state uncommitted.
    #[napi]
    pub async fn execute(
        &self,
        inputs: Vec<&NativeTensor>,
        scalars: Vec<f64>,
        sequences: Option<Vec<&NativeKvSequence>>,
        slots: Option<Vec<u32>>,
        active_mask: Option<Vec<bool>>,
        valid_lengths: Option<Vec<u32>>,
        advances: Option<Vec<u32>>,
        tokens: Option<Vec<Vec<u32>>>,
        cancellation_token: Option<&CancellationToken>,
    ) -> Result<Vec<NativeTensor>> {
        validate_execution_mode(
            self.state.is_some(),
            scalars.len(),
            sequences.as_ref().map(Vec::len),
            slots.as_deref(),
            active_mask.as_deref(),
            valid_lengths.as_deref(),
            advances.as_deref(),
            tokens.as_ref().map(Vec::len),
        )?;
        let Some(schema) = self.state else {
            return self
                .execute_stateless(inputs, scalars, cancellation_token)
                .await;
        };
        let sequences = sequences.expect("state invocation was validated");
        let slots = slots.expect("state invocation was validated");
        let active_mask = active_mask.expect("state invocation was validated");
        let valid_lengths = valid_lengths.expect("state invocation was validated");
        let advances = advances.expect("state invocation was validated");
        let tokens = tokens.expect("state invocation was validated");
        validate_fixed_lanes(
            &schema,
            sequences.len(),
            &slots,
            &active_mask,
            &valid_lengths,
            &advances,
            &tokens,
        )?;
        match self
            .execute_stateful(
                inputs,
                sequences,
                slots,
                advances,
                tokens,
                StatefulInvocation::Tensors,
                cancellation_token,
            )
            .await?
        {
            StatefulExecutionOutput::Tensors(outputs) => Ok(outputs),
            StatefulExecutionOutput::Samples(_) => {
                unreachable!("ordinary stateful execution returned samples")
            }
        }
    }

    /// Runs a stateful program and samples its active outputs before committing
    /// the sequence transaction. Does not publish output tensor wrappers.
    #[napi]
    pub async fn execute_sampled(
        &self,
        inputs: Vec<&NativeTensor>,
        sequences: Vec<&NativeKvSequence>,
        slots: Vec<u32>,
        active_mask: Vec<bool>,
        valid_lengths: Vec<u32>,
        advances: Vec<u32>,
        tokens: Vec<Vec<u32>>,
        sampling: Vec<NativeSamplingOptions>,
        cancellation_token: Option<&CancellationToken>,
    ) -> Result<Vec<u32>> {
        validate_sampled_execution_mode(
            self.state.is_some(),
            sequences.len(),
            tokens.len(),
            sampling.len(),
        )?;
        let sampling = sampling
            .into_iter()
            .map(sampling_options)
            .collect::<Result<Vec<_>>>()?;
        let schema = self.state.expect("sampled state invocation was validated");
        validate_fixed_lanes(
            &schema,
            sequences.len(),
            &slots,
            &active_mask,
            &valid_lengths,
            &advances,
            &tokens,
        )?;
        match self
            .execute_stateful(
                inputs,
                sequences,
                slots,
                advances,
                tokens,
                StatefulInvocation::Sampled(sampling),
                cancellation_token,
            )
            .await?
        {
            StatefulExecutionOutput::Samples(tokens) => Ok(tokens),
            StatefulExecutionOutput::Tensors(_) => {
                unreachable!("sampled stateful execution returned tensors")
            }
        }
    }

    #[napi]
    #[allow(clippy::too_many_arguments)]
    pub async fn execute_speculative(
        &self,
        proposer: &Executable,
        target_sequences: Vec<&NativeKvSequence>,
        proposer_sequences: Vec<&NativeKvSequence>,
        slots: Vec<u32>,
        pending_tokens: Vec<u32>,
        sampling: Vec<NativeInferenceSamplingOptions>,
        sequence_ids: Vec<NativeU64>,
        absolute_positions: Vec<NativeU64>,
        max_draft_tokens: u32,
        page_limits: Vec<u32>,
        eos_tokens: Vec<Vec<u32>>,
        cancellation_token: Option<&CancellationToken>,
    ) -> Result<Vec<Vec<u32>>> {
        self.execute_speculative_detailed(
            proposer,
            target_sequences,
            proposer_sequences,
            slots,
            pending_tokens,
            sampling,
            sequence_ids,
            absolute_positions,
            max_draft_tokens,
            page_limits,
            eos_tokens,
            cancellation_token,
        )
        .await
        .map(|execution| execution.pages)
    }
}

struct SpeculativeExecution {
    pages: Vec<Vec<u32>>,
    proposed: u64,
    accepted: Vec<usize>,
    provisional_blocks: u64,
    rolled_back_blocks: u64,
    draft_nanos: u64,
    verification_nanos: u64,
}

impl Executable {
    #[allow(clippy::too_many_arguments)]
    async fn execute_speculative_detailed(
        &self,
        proposer: &Executable,
        target_sequences: Vec<&NativeKvSequence>,
        proposer_sequences: Vec<&NativeKvSequence>,
        slots: Vec<u32>,
        pending_tokens: Vec<u32>,
        sampling: Vec<NativeInferenceSamplingOptions>,
        sequence_ids: Vec<NativeU64>,
        absolute_positions: Vec<NativeU64>,
        max_draft_tokens: u32,
        page_limits: Vec<u32>,
        eos_tokens: Vec<Vec<u32>>,
        cancellation_token: Option<&CancellationToken>,
    ) -> Result<SpeculativeExecution> {
        let target_schema = self.state.ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "executeSpeculative: target must be stateful",
            )
        })?;
        let proposer_schema = proposer.state.ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "executeSpeculative: proposer must be stateful",
            )
        })?;
        let count = target_sequences.len();
        if count == 0
            || proposer_sequences.len() != count
            || slots.len() != count
            || pending_tokens.len() != count
            || sampling.len() != count
            || sequence_ids.len() != count
            || absolute_positions.len() != count
            || page_limits.len() != count
            || eos_tokens.len() != count
            || max_draft_tokens == 0
            || target_schema.batch != proposer_schema.batch
            || count > target_schema.batch
        {
            return Err(Error::new(
                Status::InvalidArg,
                "executeSpeculative: inconsistent batch arrays, programs, or maxDraftTokens",
            ));
        }
        if target_schema.window.is_some()
            || proposer_schema.window.is_some()
            || target_schema.kda.layers != 0
            || target_schema.conv.layers != 0
            || proposer_schema.kda.layers != 0
            || proposer_schema.conv.layers != 0
        {
            return Err(Error::new(
                Status::InvalidArg,
                "executeSpeculative: window, KDA, and convolution state are unsupported",
            ));
        }
        let max_draft = max_draft_tokens as usize;
        let target_steps = max_draft + 1;
        if target_schema.packed_rows_per_sequence != Some(target_steps)
            || proposer_schema.packed_rows_per_sequence.is_some()
        {
            return Err(Error::new(
                Status::InvalidArg,
                "executeSpeculative: target must use matching packed causal chains and proposer must use dense rows",
            ));
        }
        let target_graph_rows = target_schema
            .batch
            .checked_mul(target_steps)
            .ok_or_else(|| {
                Error::new(
                    Status::InvalidArg,
                    "executeSpeculative: target graph rows overflow",
                )
            })?;
        let token_slot = |executable: &Executable, schema: KvStateSchema, shape: [usize; 2]| {
            let slots = executable
                .inner
                .slots
                .iter()
                .enumerate()
                .filter(|(index, slot)| {
                    !slot.scalar && !(schema.cursor_tensor && *index as u32 == schema.cursor_slot)
                })
                .map(|(_, slot)| slot)
                .collect::<Vec<_>>();
            if slots.len() != 1 || slots[0].shape != shape {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!(
                        "executeSpeculative: expected one token input [{},{}]",
                        shape[0], shape[1]
                    ),
                ));
            }
            if !matches!(slots[0].dtype, DType::I64 | DType::U32) {
                return Err(Error::new(
                    Status::InvalidArg,
                    "executeSpeculative: token input must be i64 or u32",
                ));
            }
            Ok(slots[0].dtype)
        };
        let target_token_dtype = token_slot(self, target_schema, [target_graph_rows, 1])?;
        let proposer_token_dtype =
            token_slot(proposer, proposer_schema, [proposer_schema.batch, 1])?;
        if self.inner.executable.program.outputs.len() != 1
            || proposer.inner.executable.program.outputs.len() != proposer_schema.batch
        {
            return Err(Error::new(
                Status::InvalidArg,
                "executeSpeculative: target must have one all-row output and proposer one row per lane",
            ));
        }
        let target_program = &self.inner.executable.program;
        let target_declaration = &target_program.values[target_program.outputs[0].index()];
        let proposer_program = &proposer.inner.executable.program;
        let proposer_declarations = proposer_program
            .outputs
            .iter()
            .map(|output| &proposer_program.values[output.index()])
            .collect::<Vec<_>>();
        let vocabulary = target_declaration.shape.get(2).copied().unwrap_or(0);
        if target_declaration.shape.as_ref() != [target_graph_rows, 1, vocabulary]
            || vocabulary == 0
            || !matches!(
                target_declaration.dtype,
                DType::F16 | DType::BF16 | DType::F32 | DType::F64
            )
            || proposer_declarations.iter().any(|declaration| {
                declaration.shape.as_ref() != [vocabulary]
                    || declaration.dtype != proposer_declarations[0].dtype
                    || !matches!(
                        declaration.dtype,
                        DType::F16 | DType::BF16 | DType::F32 | DType::F64
                    )
            })
            || pending_tokens
                .iter()
                .any(|token| *token as usize >= vocabulary)
            || eos_tokens
                .iter()
                .flatten()
                .any(|token| *token as usize >= vocabulary)
        {
            return Err(Error::new(
                Status::InvalidArg,
                "executeSpeculative: malformed logits geometry, dtype, or token metadata",
            ));
        }
        let mut seen_slots = HashSet::new();
        let mut seen_sequences = HashSet::new();
        for index in 0..count {
            let slot = slots[index] as usize;
            if slot >= target_schema.batch || !seen_slots.insert(slot) || page_limits[index] == 0 {
                return Err(Error::new(
                    Status::InvalidArg,
                    "executeSpeculative: slots must be unique and pageLimits positive",
                ));
            }
            for sequence in [target_sequences[index], proposer_sequences[index]] {
                if sequence.released.load(Ordering::SeqCst)
                    || !seen_sequences.insert(Arc::as_ptr(&sequence.state) as usize)
                {
                    return Err(Error::new(
                        Status::InvalidArg,
                        "executeSpeculative: sequences must be live and distinct",
                    ));
                }
            }
            validate_pool_schema(&target_schema, &target_sequences[index].pool)?;
            validate_pool_schema(&proposer_schema, &proposer_sequences[index].pool)?;
            if index > 0
                && (!Arc::ptr_eq(&target_sequences[0].pool, &target_sequences[index].pool)
                    || !Arc::ptr_eq(&proposer_sequences[0].pool, &proposer_sequences[index].pool))
            {
                return Err(Error::new(
                    Status::InvalidArg,
                    "executeSpeculative: each sequence set must share one pool",
                ));
            }
        }
        let sampling = sampling
            .iter()
            .map(|options| inference_sampling(options, "proposer"))
            .collect::<Result<Vec<_>>>()?;
        let sequence_ids = sequence_ids
            .into_iter()
            .map(|value| value.0)
            .collect::<Vec<_>>();
        let absolute_positions = absolute_positions
            .into_iter()
            .map(|value| value.0)
            .collect::<Vec<_>>();
        let target_executable = self.inner.executable.clone();
        let target_generated = self.inner.generated_bindings.clone();
        let proposer_executable = proposer.inner.executable.clone();
        let proposer_generated = proposer.inner.generated_bindings.clone();
        let target_pool = target_sequences[0].pool.clone();
        let proposer_pool = proposer_sequences[0].pool.clone();
        let target_states = target_sequences
            .iter()
            .map(|sequence| sequence.state.clone())
            .collect::<Vec<_>>();
        let proposer_states = proposer_sequences
            .iter()
            .map(|sequence| sequence.state.clone())
            .collect::<Vec<_>>();
        let released = target_sequences
            .iter()
            .chain(&proposer_sequences)
            .map(|sequence| sequence.released.clone())
            .collect::<Vec<_>>();
        let mut lock_entries = target_sequences
            .iter()
            .chain(&proposer_sequences)
            .map(|sequence| {
                (
                    Arc::as_ptr(&sequence.run_lock) as usize,
                    sequence.run_lock.clone(),
                )
            })
            .collect::<Vec<_>>();
        lock_entries.sort_by_key(|entry| entry.0);
        let run_locks = lock_entries
            .into_iter()
            .map(|entry| entry.1)
            .collect::<Vec<_>>();
        run_compute(cancellation_token, move |cancelled, cancellation| {
            let _guards = run_locks
                .iter()
                .map(|lock| {
                    lock.lock().map_err(|error| {
                        Error::new(
                            Status::GenericFailure,
                            format!("kv sequence lock poisoned: {error}"),
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if released
                .iter()
                .any(|released| released.load(Ordering::SeqCst))
            {
                return Err(Error::new(
                    Status::GenericFailure,
                    "kv sequence is released",
                ));
            }
            let mut draft_lengths = page_limits
                .iter()
                .map(|limit| max_draft.min((*limit as usize).saturating_sub(1)))
                .collect::<Vec<_>>();
            for index in 0..count {
                let target = target_states[index]
                    .lock()
                    .map_err(|error| Error::new(Status::GenericFailure, error.to_string()))?;
                let proposer = proposer_states[index]
                    .lock()
                    .map_err(|error| Error::new(Status::GenericFailure, error.to_string()))?;
                if target.head != 0
                    || proposer.head != 0
                    || target.cursor != proposer.cursor
                    || target.cursor >= target_schema.max_tokens
                    || proposer.cursor >= proposer_schema.max_tokens
                {
                    return Err(Error::new(
                        Status::InvalidArg,
                        "executeSpeculative: sequence exceeds state capacity or has evicted KV blocks",
                    ));
                }
                let remaining = (target_schema.max_tokens - target.cursor)
                    .min(proposer_schema.max_tokens - proposer.cursor);
                draft_lengths[index] = draft_lengths[index].min(remaining - 1);
            }
            let mut target_shadow = Vec::with_capacity(count);
            for state in &target_states {
                match clone_speculative_state(&target_pool, state) {
                    Ok(state) => target_shadow.push(state),
                    Err(error) => {
                        discard_speculative_states(&target_pool, &target_shadow);
                        return Err(to_napi_err(error));
                    }
                }
            }
            let mut proposer_shadow = Vec::with_capacity(count);
            for state in &proposer_states {
                match clone_speculative_state(&proposer_pool, state) {
                    Ok(state) => proposer_shadow.push(state),
                    Err(error) => {
                        discard_speculative_states(&target_pool, &target_shadow);
                        discard_speculative_states(&proposer_pool, &proposer_shadow);
                        return Err(to_napi_err(error));
                    }
                }
            }
            let result = (|| -> std::result::Result<_, String> {
                let draft_started = std::time::Instant::now();
                let initial_target = target_shadow
                    .iter()
                    .map(|state| {
                        let state = state.lock().map_err(|error| error.to_string())?;
                        Ok((
                            state.cursor,
                            state.last_hash,
                            state.pending.clone(),
                            state.blocks.len(),
                        ))
                    })
                    .collect::<std::result::Result<Vec<_>, String>>()?;
                let initial_proposer = proposer_shadow
                    .iter()
                    .map(|state| {
                        let state = state.lock().map_err(|error| error.to_string())?;
                        Ok((
                            state.cursor,
                            state.last_hash,
                            state.pending.clone(),
                            state.blocks.len(),
                        ))
                    })
                    .collect::<std::result::Result<Vec<_>, String>>()?;
                let mut candidates = vec![Vec::<u32>::new(); count];
                let mut proposal_probabilities = vec![Vec::<Vec<f64>>::new(); count];
                for step in 0..max_draft {
                    let active = (0..count)
                        .filter(|&index| step < draft_lengths[index])
                        .collect::<Vec<_>>();
                    if active.is_empty() {
                        break;
                    }
                    let mut input_tokens = vec![0; proposer_schema.batch];
                    let mut lane_states = vec![None; proposer_schema.batch];
                    let mut advances = vec![0; proposer_schema.batch];
                    for &index in &active {
                        let slot = slots[index] as usize;
                        input_tokens[slot] = if step == 0 {
                            pending_tokens[index]
                        } else {
                            candidates[index][step - 1]
                        };
                        lane_states[slot] = Some(proposer_shadow[index].clone());
                        advances[slot] = 1;
                        proposer_shadow[index]
                            .lock()
                            .map_err(|error| error.to_string())?
                            .advance = 1;
                    }
                    let input = speculative_token_input(
                        proposer_token_dtype,
                        proposer_schema.batch,
                        &input_tokens,
                        1,
                    )?;
                    let context = Arc::new(KvContext {
                        pool: proposer_pool.clone(),
                        slots: lane_states,
                        advances,
                        packed: None,
                        window: None,
                        kda: proposer_schema.kda,
                        conv: proposer_schema.conv,
                        transaction: Mutex::new(None),
                    });
                    let outputs = executable::execute_stateful(
                        &proposer_executable,
                        &[input],
                        &proposer_generated,
                        cancelled,
                        &context,
                        &|| true,
                    )?;
                    for &index in &active {
                        let probabilities = speculative_probabilities(
                            &outputs[slots[index] as usize],
                            None,
                            sampling[index],
                            cancelled,
                        )?;
                        let candidate = sample_probabilities(
                            &probabilities,
                            coordinate_seed(
                                sampling[index].seed,
                                sequence_ids[index],
                                absolute_positions[index] + step as u64,
                                SamplingPurpose::Proposal,
                                0,
                            ),
                            0,
                            || cancelled.load(Ordering::Acquire),
                        )?;
                        candidates[index].push(candidate);
                        proposal_probabilities[index].push(probabilities);
                        proposer_shadow[index]
                            .lock()
                            .map_err(|error| error.to_string())?
                            .cursor += 1;
                    }
                }
                let draft_nanos = u64::try_from(draft_started.elapsed().as_nanos())
                    .unwrap_or(u64::MAX)
                    .max(1);
                let chain_lengths = candidates
                    .iter()
                    .map(|tokens| tokens.len() + 1)
                    .collect::<Vec<_>>();
                let physical_slots = slots.iter().map(|slot| *slot as usize).collect::<Vec<_>>();
                let target_cursors = initial_target.iter().map(|initial| initial.0).collect::<Vec<_>>();
                let packed = PackedCausalRows::build(
                    target_graph_rows,
                    target_schema.batch,
                    &physical_slots,
                    &target_cursors,
                    &chain_lengths,
                )?;
                let mut verify_tokens = vec![0; target_graph_rows];
                let mut target_lanes = vec![None; target_schema.batch];
                let mut target_advances = vec![0; target_schema.batch];
                for index in 0..count {
                    let slot = slots[index] as usize;
                    let base = packed.row_offsets[index];
                    verify_tokens[base] = pending_tokens[index];
                    verify_tokens[base + 1..base + 1 + candidates[index].len()]
                        .copy_from_slice(&candidates[index]);
                    target_lanes[slot] = Some(target_shadow[index].clone());
                    target_advances[slot] = candidates[index].len() + 1;
                    target_shadow[index]
                        .lock()
                        .map_err(|error| error.to_string())?
                        .advance = candidates[index].len() + 1;
                }
                let verify_input = speculative_token_input(
                    target_token_dtype,
                    target_graph_rows,
                    &verify_tokens,
                    1,
                )?;
                let target_context = Arc::new(KvContext {
                    pool: target_pool.clone(),
                    slots: target_lanes,
                    advances: target_advances,
                    packed: Some(packed.clone()),
                    window: None,
                    kda: target_schema.kda,
                    conv: target_schema.conv,
                    transaction: Mutex::new(None),
                });
                let verification_started = std::time::Instant::now();
                let target_outputs = executable::execute_stateful(
                    &target_executable,
                    &[verify_input],
                    &target_generated,
                    cancelled,
                    &target_context,
                    &|| true,
                )?;
                let verification_nanos = u64::try_from(verification_started.elapsed().as_nanos())
                    .unwrap_or(u64::MAX)
                    .max(1);
                let baseline_blocks = initial_target
                    .iter()
                    .chain(&initial_proposer)
                    .map(|initial| initial.3 as u64)
                    .sum::<u64>();
                let peak_blocks = target_shadow
                    .iter()
                    .chain(&proposer_shadow)
                    .map(|state| state.lock().map(|state| state.blocks.len() as u64).unwrap_or(0))
                    .sum::<u64>();
                let provisional_blocks = peak_blocks.saturating_sub(baseline_blocks);
                if target_outputs.len() != 1 {
                    return Err("executeSpeculative: target did not produce one output".to_string());
                }
                let target_output = &target_outputs[0];
                if target_output.shape().len() != 3
                    || target_output.shape()[0] != target_graph_rows
                    || target_output.shape()[1] != 1
                {
                    return Err("executeSpeculative: target output must be [graphRows,1,V]".to_string());
                }
                let vocabulary = target_output.shape()[2];
                let mut pages = Vec::with_capacity(count);
                for index in 0..count {
                    let row_offset = packed.row_offsets[index];
                    let mut page = Vec::with_capacity(candidates[index].len() + 1);
                    let mut rejected = false;
                    for step in 0..candidates[index].len() {
                        let p = speculative_probabilities(
                            target_output,
                            Some((row_offset + step, 0)),
                            sampling[index],
                            cancelled,
                        )?;
                        if p.len() != proposal_probabilities[index][step].len()
                            || p.len() != vocabulary
                        {
                            return Err("executeSpeculative: target/proposer vocabulary mismatch"
                                .to_string());
                        }
                        let candidate = candidates[index][step];
                        let q = &proposal_probabilities[index][step];
                        let accepted = if sampling[index].temperature == 0.0 {
                            p[candidate as usize] == 1.0
                        } else {
                            random_unit_at(sampling_coordinate(
                                sampling[index].seed,
                                sequence_ids[index],
                                absolute_positions[index] + step as u64,
                                SamplingPurpose::Accept,
                                0,
                            )) < (p[candidate as usize] / q[candidate as usize]).min(1.0)
                        };
                        if accepted {
                            page.push(candidate);
                            continue;
                        }
                        let residual = p
                            .iter()
                            .zip(q)
                            .map(|(&p, &q)| (p - q).max(0.0))
                            .collect::<Vec<_>>();
                        page.push(sample_probabilities(
                            &residual,
                            coordinate_seed(
                                sampling[index].seed,
                                sequence_ids[index],
                                absolute_positions[index] + step as u64,
                                SamplingPurpose::Residual,
                                0,
                            ),
                            0,
                            || cancelled.load(Ordering::Acquire),
                        )?);
                        rejected = true;
                        break;
                    }
                    if !rejected {
                        let row = candidates[index].len();
                        let p = speculative_probabilities(
                            target_output,
                            Some((row_offset + row, 0)),
                            sampling[index],
                            cancelled,
                        )?;
                        page.push(sample_probabilities(
                            &p,
                            coordinate_seed(
                                sampling[index].seed,
                                sequence_ids[index],
                                absolute_positions[index] + row as u64,
                                SamplingPurpose::Target,
                                0,
                            ),
                            0,
                            || cancelled.load(Ordering::Acquire),
                        )?);
                    }
                    if let Some(position) = page
                        .iter()
                        .position(|token| eos_tokens[index].contains(token))
                    {
                        page.truncate(position + 1);
                    }
                    page.truncate(page_limits[index] as usize);
                    pages.push(page);
                }
                let catchup = (0..count)
                    .filter_map(|index| {
                        if candidates[index].is_empty() {
                            Some((index, pending_tokens[index]))
                        } else if pages[index].len() == candidates[index].len() + 1
                            && pages[index][..candidates[index].len()] == candidates[index]
                        {
                            Some((index, *candidates[index].last().unwrap()))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                if !catchup.is_empty() {
                    let mut inputs = vec![0; proposer_schema.batch];
                    let mut lanes = vec![None; proposer_schema.batch];
                    let mut advances = vec![0; proposer_schema.batch];
                    for &(index, token) in &catchup {
                        let slot = slots[index] as usize;
                        inputs[slot] = token;
                        lanes[slot] = Some(proposer_shadow[index].clone());
                        advances[slot] = 1;
                        proposer_shadow[index]
                            .lock()
                            .map_err(|error| error.to_string())?
                            .advance = 1;
                    }
                    let input = speculative_token_input(
                        proposer_token_dtype,
                        proposer_schema.batch,
                        &inputs,
                        1,
                    )?;
                    let context = Arc::new(KvContext {
                        pool: proposer_pool.clone(),
                        slots: lanes,
                        advances,
                        packed: None,
                        window: None,
                        kda: proposer_schema.kda,
                        conv: proposer_schema.conv,
                        transaction: Mutex::new(None),
                    });
                    executable::execute_stateful(
                        &proposer_executable,
                        &[input],
                        &proposer_generated,
                        cancelled,
                        &context,
                        &|| true,
                    )?;
                }
                let mut target_hashes = Vec::new();
                let mut proposer_hashes = Vec::new();
                for index in 0..count {
                    let consumed = pages[index].len();
                    let retained = std::iter::once(pending_tokens[index])
                        .chain(
                            pages[index]
                                .iter()
                                .take(consumed.saturating_sub(1))
                                .copied(),
                        )
                        .collect::<Vec<_>>();
                    for (shadow, initial, pool, hashes) in [
                        (
                            &target_shadow[index],
                            &initial_target[index],
                            &target_pool,
                            &mut target_hashes,
                        ),
                        (
                            &proposer_shadow[index],
                            &initial_proposer[index],
                            &proposer_pool,
                            &mut proposer_hashes,
                        ),
                    ] {
                        let mut state = shadow.lock().map_err(|error| error.to_string())?;
                        let needed = (initial.0 + consumed).div_ceil(pool.block_size);
                        for block in state.blocks.drain(needed..) {
                            pool.unref_block(block);
                        }
                        state.cursor = initial.0;
                        state.last_hash = initial.1;
                        state.pending = initial.2.clone();
                        state.advance = 0;
                        hashes.extend(speculative_note_tokens(&mut state, pool, &retained));
                        state.cursor += consumed;
                    }
                }
                if cancelled.load(Ordering::Acquire) || !cancellation.complete() {
                    return Err("operation aborted".to_string());
                }
                let retained_blocks = target_shadow
                    .iter()
                    .chain(&proposer_shadow)
                    .map(|state| state.lock().map(|state| state.blocks.len() as u64).unwrap_or(0))
                    .sum::<u64>();
                let rolled_back_blocks = provisional_blocks
                    .saturating_sub(retained_blocks.saturating_sub(baseline_blocks));
                let mut canonical = target_states
                    .iter()
                    .chain(&proposer_states)
                    .map(|state| state.lock().map_err(|error| error.to_string()))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                for index in 0..count * 2 {
                    let shadow = if index < count {
                        &target_shadow[index]
                    } else {
                        &proposer_shadow[index - count]
                    };
                    let mut shadow = shadow.lock().map_err(|error| error.to_string())?;
                    let replacement = SeqState {
                        blocks: std::mem::take(&mut shadow.blocks),
                        head: shadow.head,
                        cursor: shadow.cursor,
                        advance: 0,
                        last_hash: shadow.last_hash,
                        pending: std::mem::take(&mut shadow.pending),
                        kda_states: std::mem::take(&mut shadow.kda_states),
                        conv_states: std::mem::take(&mut shadow.conv_states),
                    };
                    let old = std::mem::replace(&mut *canonical[index], replacement);
                    let pool = if index < count {
                        &target_pool
                    } else {
                        &proposer_pool
                    };
                    for block in old.blocks {
                        pool.unref_block(block);
                    }
                }
                drop(canonical);
                for (block, hash) in target_hashes {
                    target_pool.set_hash(block, hash);
                }
                for (block, hash) in proposer_hashes {
                    proposer_pool.set_hash(block, hash);
                }
                let proposed = candidates.iter().map(Vec::len).sum::<usize>() as u64;
                let accepted = candidates
                    .iter()
                    .zip(&pages)
                    .map(|(candidates, page)| {
                        candidates
                            .iter()
                            .zip(page)
                            .take_while(|(candidate, emitted)| candidate == emitted)
                            .count()
                    })
                    .collect::<Vec<_>>();
                Ok(SpeculativeExecution {
                    pages,
                    proposed,
                    accepted,
                    provisional_blocks,
                    rolled_back_blocks,
                    draft_nanos,
                    verification_nanos,
                })
            })();
            discard_speculative_states(&target_pool, &target_shadow);
            discard_speculative_states(&proposer_pool, &proposer_shadow);
            result.map_err(to_napi_err)
        })
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_history_lookup_detailed(
        &self,
        target_sequences: Vec<&NativeKvSequence>,
        slots: Vec<u32>,
        pending_tokens: Vec<u32>,
        histories: Vec<Vec<u32>>,
        sampling: Vec<SamplingOptions>,
        sequence_ids: Vec<u64>,
        absolute_positions: Vec<u64>,
        config: HistoryLookupConfig,
        max_draft_tokens: usize,
        page_limits: Vec<u32>,
        eos_tokens: Vec<Vec<u32>>,
        cancellation_token: Option<&CancellationToken>,
    ) -> Result<SpeculativeExecution> {
        let schema = self.state.ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "executeHistoryLookup: target must be stateful",
            )
        })?;
        let count = target_sequences.len();
        if count == 0
            || slots.len() != count
            || pending_tokens.len() != count
            || histories.len() != count
            || sampling.len() != count
            || sequence_ids.len() != count
            || absolute_positions.len() != count
            || page_limits.len() != count
            || eos_tokens.len() != count
            || max_draft_tokens == 0
            || count > schema.batch
            || schema.window.is_some()
            || schema.kda.layers != 0
            || schema.conv.layers != 0
        {
            return Err(Error::new(
                Status::InvalidArg,
                "executeHistoryLookup: inconsistent batch arrays or unsupported target state",
            ));
        }
        let target_steps = max_draft_tokens + 1;
        if schema.packed_rows_per_sequence != Some(target_steps) {
            return Err(Error::new(
                Status::InvalidArg,
                "executeHistoryLookup: target packed causal-chain width differs",
            ));
        }
        let graph_rows = schema.batch.checked_mul(target_steps).ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "executeHistoryLookup: graph rows overflow",
            )
        })?;
        let token_slots = self
            .inner
            .slots
            .iter()
            .enumerate()
            .filter(|(index, slot)| {
                !slot.scalar && !(schema.cursor_tensor && *index as u32 == schema.cursor_slot)
            })
            .map(|(_, slot)| slot)
            .collect::<Vec<_>>();
        if token_slots.len() != 1
            || token_slots[0].shape != [graph_rows, 1]
            || !matches!(token_slots[0].dtype, DType::I64 | DType::U32)
            || self.inner.executable.program.outputs.len() != 1
        {
            return Err(Error::new(
                Status::InvalidArg,
                "executeHistoryLookup: verifier token or output contract differs",
            ));
        }
        let token_dtype = token_slots[0].dtype;
        let program = &self.inner.executable.program;
        let declaration = &program.values[program.outputs[0].index()];
        let vocabulary = declaration.shape.get(2).copied().unwrap_or(0);
        if declaration.shape.as_ref() != [graph_rows, 1, vocabulary]
            || vocabulary == 0
            || !matches!(
                declaration.dtype,
                DType::F16 | DType::BF16 | DType::F32 | DType::F64
            )
            || pending_tokens
                .iter()
                .chain(eos_tokens.iter().flatten())
                .any(|token| *token as usize >= vocabulary)
        {
            return Err(Error::new(
                Status::InvalidArg,
                "executeHistoryLookup: malformed verifier logits or token metadata",
            ));
        }
        let mut seen_slots = HashSet::new();
        let mut seen_sequences = HashSet::new();
        for (index, sequence) in target_sequences.iter().enumerate() {
            if slots[index] as usize >= schema.batch
                || !seen_slots.insert(slots[index])
                || page_limits[index] == 0
                || sequence.released.load(Ordering::SeqCst)
                || !seen_sequences.insert(Arc::as_ptr(&sequence.state) as usize)
            {
                return Err(Error::new(
                    Status::InvalidArg,
                    "executeHistoryLookup: slots and sequences must be live and unique",
                ));
            }
            validate_pool_schema(&schema, &sequence.pool)?;
            if index > 0 && !Arc::ptr_eq(&target_sequences[0].pool, &sequence.pool) {
                return Err(Error::new(
                    Status::InvalidArg,
                    "executeHistoryLookup: target sequences must share one pool",
                ));
            }
        }

        let executable = self.inner.executable.clone();
        let generated = self.inner.generated_bindings.clone();
        let pool = target_sequences[0].pool.clone();
        let states = target_sequences
            .iter()
            .map(|sequence| sequence.state.clone())
            .collect::<Vec<_>>();
        let released = target_sequences
            .iter()
            .map(|sequence| sequence.released.clone())
            .collect::<Vec<_>>();
        let mut lock_entries = target_sequences
            .iter()
            .map(|sequence| {
                (
                    Arc::as_ptr(&sequence.run_lock) as usize,
                    sequence.run_lock.clone(),
                )
            })
            .collect::<Vec<_>>();
        lock_entries.sort_by_key(|entry| entry.0);
        let run_locks = lock_entries
            .into_iter()
            .map(|entry| entry.1)
            .collect::<Vec<_>>();
        run_compute(cancellation_token, move |cancelled, cancellation| {
            let _guards = run_locks
                .iter()
                .map(|lock| {
                    lock.lock().map_err(|error| {
                        Error::new(
                            Status::GenericFailure,
                            format!("kv sequence lock poisoned: {error}"),
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if released.iter().any(|value| value.load(Ordering::SeqCst)) {
                return Err(Error::new(
                    Status::GenericFailure,
                    "kv sequence is released",
                ));
            }
            let initial = states
                .iter()
                .zip(&histories)
                .zip(&pending_tokens)
                .map(|((state, history), pending)| {
                    let state = state.lock().map_err(|error| error.to_string())?;
                    if state.head != 0
                        || state.cursor >= schema.max_tokens
                        || history.len() != state.cursor + 1
                        || history.last() != Some(pending)
                    {
                        return Err(
                            "executeHistoryLookup: visible history does not match target cursor"
                                .to_string(),
                        );
                    }
                    Ok((
                        state.cursor,
                        state.last_hash,
                        state.pending.clone(),
                        state.blocks.len(),
                    ))
                })
                .collect::<std::result::Result<Vec<_>, String>>()
                .map_err(to_napi_err)?;
            let mut shadow = Vec::with_capacity(count);
            for state in &states {
                match clone_speculative_state(&pool, state) {
                    Ok(state) => shadow.push(state),
                    Err(error) => {
                        discard_speculative_states(&pool, &shadow);
                        return Err(to_napi_err(error));
                    }
                }
            }
            let result = (|| -> std::result::Result<_, String> {
                let draft_started = std::time::Instant::now();
                let candidates = histories
                    .iter()
                    .enumerate()
                    .map(|(index, history)| {
                        let limit = max_draft_tokens
                            .min((page_limits[index] as usize).saturating_sub(1))
                            .min(schema.max_tokens - initial[index].0 - 1);
                        history_lookup(
                            history,
                            config.min_match_tokens,
                            config.max_match_tokens,
                            limit,
                        )
                    })
                    .collect::<Vec<_>>();
                let draft_nanos = u64::try_from(draft_started.elapsed().as_nanos())
                    .unwrap_or(u64::MAX)
                    .max(1);
                let chain_lengths = candidates
                    .iter()
                    .map(|tokens| tokens.len() + 1)
                    .collect::<Vec<_>>();
                let physical_slots = slots.iter().map(|slot| *slot as usize).collect::<Vec<_>>();
                let cursors = initial.iter().map(|state| state.0).collect::<Vec<_>>();
                let packed = PackedCausalRows::build(
                    graph_rows,
                    schema.batch,
                    &physical_slots,
                    &cursors,
                    &chain_lengths,
                )?;
                let mut verify_tokens = vec![0; graph_rows];
                let mut lanes = vec![None; schema.batch];
                let mut advances = vec![0; schema.batch];
                for index in 0..count {
                    let slot = slots[index] as usize;
                    let base = packed.row_offsets[index];
                    verify_tokens[base] = pending_tokens[index];
                    verify_tokens[base + 1..base + 1 + candidates[index].len()]
                        .copy_from_slice(&candidates[index]);
                    lanes[slot] = Some(shadow[index].clone());
                    advances[slot] = candidates[index].len() + 1;
                    shadow[index]
                        .lock()
                        .map_err(|error| error.to_string())?
                        .advance = advances[slot];
                }
                let input = speculative_token_input(token_dtype, graph_rows, &verify_tokens, 1)?;
                let context = Arc::new(KvContext {
                    pool: pool.clone(),
                    slots: lanes,
                    advances,
                    packed: Some(packed.clone()),
                    window: None,
                    kda: schema.kda,
                    conv: schema.conv,
                    transaction: Mutex::new(None),
                });
                let verification_started = std::time::Instant::now();
                let outputs = executable::execute_stateful(
                    &executable,
                    &[input],
                    &generated,
                    cancelled,
                    &context,
                    &|| true,
                )?;
                let verification_nanos = u64::try_from(verification_started.elapsed().as_nanos())
                    .unwrap_or(u64::MAX)
                    .max(1);
                if outputs.len() != 1 {
                    return Err(
                        "executeHistoryLookup: verifier did not produce one output".to_string()
                    );
                }
                let output = &outputs[0];
                let baseline_blocks = initial.iter().map(|state| state.3 as u64).sum::<u64>();
                let peak_blocks = shadow
                    .iter()
                    .map(|state| {
                        state
                            .lock()
                            .map(|state| state.blocks.len() as u64)
                            .unwrap_or(0)
                    })
                    .sum::<u64>();
                let provisional_blocks = peak_blocks.saturating_sub(baseline_blocks);
                let mut pages = Vec::with_capacity(count);
                let mut accepted = Vec::with_capacity(count);
                let mut proposed = 0u64;
                for index in 0..count {
                    let row_offset = packed.row_offsets[index];
                    let mut page = Vec::with_capacity(candidates[index].len() + 1);
                    let mut accepted_count = 0;
                    let mut rejected = false;
                    for (step, candidate) in candidates[index].iter().copied().enumerate() {
                        let target = speculative_sample_logits(
                            output,
                            (row_offset + step, 0),
                            SamplingOptions {
                                seed: coordinate_seed(
                                    sampling[index].seed,
                                    sequence_ids[index],
                                    absolute_positions[index] + step as u64,
                                    SamplingPurpose::Target,
                                    0,
                                ),
                                counter: 0,
                                ..sampling[index]
                            },
                            cancelled,
                        )?;
                        if target == candidate {
                            page.push(candidate);
                            accepted_count += 1;
                        } else {
                            page.push(target);
                            rejected = true;
                            break;
                        }
                    }
                    if !rejected {
                        let step = candidates[index].len();
                        page.push(speculative_sample_logits(
                            output,
                            (row_offset + step, 0),
                            SamplingOptions {
                                seed: coordinate_seed(
                                    sampling[index].seed,
                                    sequence_ids[index],
                                    absolute_positions[index] + step as u64,
                                    SamplingPurpose::Target,
                                    0,
                                ),
                                counter: 0,
                                ..sampling[index]
                            },
                            cancelled,
                        )?);
                    }
                    if let Some(position) = page
                        .iter()
                        .position(|token| eos_tokens[index].contains(token))
                    {
                        page.truncate(position + 1);
                    }
                    page.truncate(page_limits[index] as usize);
                    proposed += candidates[index].len().min(page.len()) as u64;
                    accepted.push(accepted_count.min(page.len()));
                    pages.push(page);
                }
                let mut hashes = Vec::new();
                for index in 0..count {
                    let consumed = pages[index].len();
                    let retained = std::iter::once(pending_tokens[index])
                        .chain(
                            pages[index]
                                .iter()
                                .take(consumed.saturating_sub(1))
                                .copied(),
                        )
                        .collect::<Vec<_>>();
                    let mut state = shadow[index].lock().map_err(|error| error.to_string())?;
                    let needed = (initial[index].0 + consumed).div_ceil(pool.block_size);
                    for block in state.blocks.drain(needed..) {
                        pool.unref_block(block);
                    }
                    state.cursor = initial[index].0;
                    state.last_hash = initial[index].1;
                    state.pending = initial[index].2.clone();
                    state.advance = 0;
                    hashes.extend(speculative_note_tokens(&mut state, &pool, &retained));
                    state.cursor += consumed;
                }
                if cancelled.load(Ordering::Acquire) || !cancellation.complete() {
                    return Err("operation aborted".to_string());
                }
                let retained_blocks = shadow
                    .iter()
                    .map(|state| {
                        state
                            .lock()
                            .map(|state| state.blocks.len() as u64)
                            .unwrap_or(0)
                    })
                    .sum::<u64>();
                let rolled_back_blocks = provisional_blocks
                    .saturating_sub(retained_blocks.saturating_sub(baseline_blocks));
                let mut canonical = states
                    .iter()
                    .map(|state| state.lock().map_err(|error| error.to_string()))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                for index in 0..count {
                    let mut staged = shadow[index].lock().map_err(|error| error.to_string())?;
                    let replacement = SeqState {
                        blocks: std::mem::take(&mut staged.blocks),
                        head: staged.head,
                        cursor: staged.cursor,
                        advance: 0,
                        last_hash: staged.last_hash,
                        pending: std::mem::take(&mut staged.pending),
                        kda_states: std::mem::take(&mut staged.kda_states),
                        conv_states: std::mem::take(&mut staged.conv_states),
                    };
                    let old = std::mem::replace(&mut *canonical[index], replacement);
                    for block in old.blocks {
                        pool.unref_block(block);
                    }
                }
                drop(canonical);
                for (block, hash) in hashes {
                    pool.set_hash(block, hash);
                }
                Ok(SpeculativeExecution {
                    pages,
                    proposed,
                    accepted,
                    provisional_blocks,
                    rolled_back_blocks,
                    draft_nanos,
                    verification_nanos,
                })
            })();
            discard_speculative_states(&pool, &shadow);
            result.map_err(to_napi_err)
        })
        .await
    }
}

fn history_lookup(
    history: &[u32],
    min_match_tokens: usize,
    max_match_tokens: usize,
    max_draft_tokens: usize,
) -> Vec<u32> {
    if max_draft_tokens == 0 || history.len() <= min_match_tokens {
        return Vec::new();
    }
    let max_match_tokens = max_match_tokens.min(history.len() - 1);
    for width in (min_match_tokens..=max_match_tokens).rev() {
        let suffix = history.len() - width;
        for start in (0..suffix).rev() {
            if history[start..start + width] == history[suffix..] {
                return history[start + width..]
                    .iter()
                    .copied()
                    .take(max_draft_tokens)
                    .collect();
            }
        }
    }
    Vec::new()
}

enum StatefulInvocation {
    Tensors,
    Sampled(Vec<SamplingOptions>),
}

enum StatefulExecutionOutput {
    Tensors(Vec<NativeTensor>),
    Samples(Vec<u32>),
}

impl Executable {
    async fn execute_stateful(
        &self,
        inputs: Vec<&NativeTensor>,
        sequences: Vec<&NativeKvSequence>,
        physical_slots: Vec<u32>,
        advances: Vec<u32>,
        tokens: Vec<Vec<u32>>,
        invocation: StatefulInvocation,
        token: Option<&CancellationToken>,
    ) -> Result<StatefulExecutionOutput> {
        self.execute_stateful_with_prefix_metadata(
            inputs,
            sequences,
            physical_slots,
            advances,
            tokens,
            invocation,
            token,
            None,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_stateful_with_prefix_metadata(
        &self,
        inputs: Vec<&NativeTensor>,
        sequences: Vec<&NativeKvSequence>,
        physical_slots: Vec<u32>,
        advances: Vec<u32>,
        tokens: Vec<Vec<u32>>,
        invocation: StatefulInvocation,
        token: Option<&CancellationToken>,
        prefix_metadata: Option<Arc<Mutex<DeferredPrefixMetadata>>>,
    ) -> Result<StatefulExecutionOutput> {
        let schema = self.state.expect("stateful execution was validated");
        let batch = schema.batch;
        for (index, sequence) in sequences.iter().enumerate() {
            if sequence.released.load(Ordering::SeqCst) {
                return Err(Error::new(
                    Status::GenericFailure,
                    format!("kv sequence {index} is released"),
                ));
            }
            if !Arc::ptr_eq(&sequence.pool, &sequences[0].pool) {
                return Err(Error::new(
                    Status::InvalidArg,
                    "kv run: batched sequences must share one pool",
                ));
            }
            if sequences[..index]
                .iter()
                .any(|other| Arc::ptr_eq(&other.state, &sequence.state))
            {
                return Err(Error::new(
                    Status::InvalidArg,
                    "kv run: duplicate sequence in batch",
                ));
            }
        }
        let pool = &sequences[0].pool;
        validate_pool_schema(&schema, pool)?;
        let runtime_values =
            usize::from(self.inner.slots.get(schema.cursor_slot as usize).is_some());
        self.inner
            .executable
            .signature
            .validate_invocation_counts(inputs.len(), 0, runtime_values, None)
            .map_err(|error| Error::new(Status::InvalidArg, error.to_string()))?;
        for (slot, declared) in self.inner.slots.iter().enumerate() {
            if declared.scalar || (schema.cursor_tensor && slot as u32 == schema.cursor_slot) {
                continue;
            }
            let input_index = slot
                - self
                    .inner
                    .slots
                    .iter()
                    .take(slot)
                    .filter(|declared| declared.scalar)
                    .count()
                - usize::from(schema.cursor_tensor && slot as u32 > schema.cursor_slot);
            validate_stateful_tensor_input(inputs[input_index], slot, declared)?;
        }
        let executable = self.inner.executable.clone();
        let generated = self.inner.generated_bindings.clone();
        let inputs = inputs
            .iter()
            .map(|input| input.value_cloned())
            .collect::<Result<Vec<_>>>()?;
        let mut lane_states = vec![None; batch];
        for (sequence, &slot) in sequences.iter().zip(&physical_slots) {
            lane_states[slot as usize] = Some(sequence.state.clone());
        }
        if advances.len() != batch && advances.len() != physical_slots.len() {
            return Err(Error::new(
                Status::InvalidArg,
                "kv run: advances must be physical-batch or compact-sequence aligned",
            ));
        }
        let context = Arc::new(KvContext {
            pool: sequences[0].pool.clone(),
            slots: lane_states,
            advances: {
                let mut lanes = vec![0; batch];
                for (index, &slot) in physical_slots.iter().enumerate() {
                    lanes[slot as usize] = if advances.len() == batch {
                        advances[slot as usize]
                    } else {
                        advances[index]
                    } as usize;
                }
                lanes
            },
            packed: None,
            window: schema.window,
            kda: schema.kda,
            conv: schema.conv,
            transaction: Mutex::new(None),
        });
        let mut ordered = sequences.clone();
        ordered.sort_by_key(|sequence| Arc::as_ptr(&sequence.run_lock) as usize);
        let run_locks = ordered
            .iter()
            .map(|sequence| sequence.run_lock.clone())
            .collect::<Vec<_>>();
        let released = sequences
            .iter()
            .map(|sequence| sequence.released.clone())
            .collect::<Vec<_>>();
        let states = sequences
            .iter()
            .map(|sequence| sequence.state.clone())
            .collect::<Vec<_>>();
        let max_tokens = schema.max_tokens;
        let pending_transaction = prefix_metadata.is_some();
        let compute = move |cancelled: &CancellationFlag, cancellation: &CancellationState| {
            let _guards = run_locks
                .iter()
                .map(|lock| {
                    lock.lock().map_err(|error| {
                        Error::new(
                            Status::GenericFailure,
                            format!("kv sequence lock poisoned: {error}"),
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            for (index, released) in released.iter().enumerate() {
                if released.load(Ordering::SeqCst) {
                    return Err(Error::new(
                        Status::GenericFailure,
                        format!("kv sequence {index} is released"),
                    ));
                }
            }
            for (index, state) in states.iter().enumerate() {
                let state = state.lock().map_err(|error| {
                    Error::new(
                        Status::GenericFailure,
                        format!("kv sequence lock poisoned: {error}"),
                    )
                })?;
                validate_recurrent_state_schema(&schema, &state)?;
                let frontier = state.cursor.checked_add(tokens[index].len());
                if frontier.is_none()
                    || (schema.window.is_none() && frontier.is_some_and(|value| value > max_tokens))
                {
                    return Err(Error::new(
                        Status::InvalidArg,
                        format!(
                            "execute: sequence {index} exceeds compiled max_tokens {max_tokens}"
                        ),
                    ));
                }
            }
            for (index, state) in states.iter().enumerate() {
                state
                    .lock()
                    .map_err(|error| {
                        Error::new(
                            Status::GenericFailure,
                            format!("kv sequence lock poisoned: {error}"),
                        )
                    })?
                    .advance = tokens[index].len();
            }
            let frontiers = states
                .iter()
                .map(|state| state.lock().map(|state| state.blocks.len()).unwrap_or(0))
                .collect::<Vec<_>>();
            let rollback = || {
                for (index, state) in states.iter().enumerate() {
                    if let Ok(mut state) = state.lock() {
                        for block in state.blocks.split_off(frontiers[index]) {
                            context.pool.unref_block(block);
                        }
                        state.advance = 0;
                    }
                }
            };
            let output = match invocation {
                StatefulInvocation::Tensors => executable::execute_stateful(
                    &executable,
                    &inputs,
                    &generated,
                    cancelled,
                    &context,
                    &|| cancellation.complete(),
                )
                .map(|outputs| {
                    StatefulExecutionOutput::Tensors(
                        outputs.into_iter().map(NativeTensor::wrap).collect(),
                    )
                }),
                StatefulInvocation::Sampled(sampling) => {
                    let mut sampled = None;
                    let mut sample_outputs = |outputs: &[Value]| {
                        if sampling.len() > outputs.len() {
                            return Err(format!(
                                "executeSampled: executable has {} outputs for {} active sequences",
                                outputs.len(),
                                sampling.len()
                            ));
                        }
                        sampled = Some(
                            physical_slots
                                .iter()
                                .zip(&sampling)
                                .map(|(&slot, options)| {
                                    let output = outputs.get(slot as usize).ok_or_else(|| format!(
                                        "executeSampled: executable has no output for physical slot {slot}"
                                    ))?;
                                    sample_blocking(output, *options, || cancelled.is_cancelled())
                                        .map_err(|error| error.reason)
                                })
                                .collect::<std::result::Result<Vec<_>, _>>()?,
                        );
                        Ok(())
                    };
                    let result = executable::execute_stateful_before_commit(
                        &executable,
                        &inputs,
                        &generated,
                        cancelled,
                        &context,
                        &|| cancellation.complete(),
                        &mut sample_outputs,
                    );
                    drop(sample_outputs);
                    result.map(|()| {
                        StatefulExecutionOutput::Samples(
                            sampled.expect("successful sampled execution produced tokens"),
                        )
                    })
                }
            };
            let output = match output {
                Ok(output) => output,
                Err(error) => {
                    rollback();
                    return Err(to_napi_err(error));
                }
            };
            if let Some(prefix_metadata) = &prefix_metadata {
                let mut metadata = prefix_metadata.lock().map_err(|error| {
                    Error::new(
                        Status::GenericFailure,
                        format!("deferred prefix metadata lock poisoned: {error}"),
                    )
                })?;
                for (index, state) in states.iter().enumerate() {
                    if let Ok(mut state) = state.lock() {
                        metadata
                            .hashes
                            .extend(state.note_tokens_deferred(&context.pool, &tokens[index]));
                        state.cursor += state.advance;
                        state.advance = 0;
                        context.pool.stage_recurrent_snapshot(&state, &mut metadata);
                    }
                }
            } else {
                for (index, state) in states.iter().enumerate() {
                    if let Ok(mut state) = state.lock() {
                        state.note_tokens(&context.pool, &tokens[index]);
                        state.cursor += state.advance;
                        state.advance = 0;
                        context.pool.maybe_publish_recurrent_snapshot(&state);
                    }
                }
            }
            Ok(output)
        };
        if pending_transaction {
            run_compute_pending(token, compute).await
        } else {
            run_compute(token, compute).await
        }
    }
}

fn inference_error(phase: &str, message: impl std::fmt::Display) -> Error {
    Error::new(
        Status::GenericFailure,
        format!("inference[{phase}]: {message}"),
    )
}

#[derive(Clone, Copy)]
pub struct NativeU64(u64);

unsafe extern "C" {
    fn napi_get_value_bigint_uint64(
        env: sys::napi_env,
        value: sys::napi_value,
        result: *mut u64,
        lossless: *mut bool,
    ) -> sys::napi_status;
    fn napi_create_bigint_uint64(
        env: sys::napi_env,
        value: u64,
        result: *mut sys::napi_value,
    ) -> sys::napi_status;
}

impl TypeName for NativeU64 {
    fn type_name() -> &'static str {
        "BigInt"
    }

    fn value_type() -> ValueType {
        ValueType::Unknown
    }
}

impl ValidateNapiValue for NativeU64 {}

impl FromNapiValue for NativeU64 {
    unsafe fn from_napi_value(env: sys::napi_env, value: sys::napi_value) -> Result<Self> {
        let mut result = 0u64;
        let mut lossless = false;
        napi::check_status!(unsafe {
            napi_get_value_bigint_uint64(env, value, &mut result, &mut lossless)
        })?;
        if !lossless {
            return Err(Error::new(
                Status::InvalidArg,
                "unsigned 64-bit BigInt is out of range",
            ));
        }
        Ok(Self(result))
    }
}

impl ToNapiValue for NativeU64 {
    unsafe fn to_napi_value(env: sys::napi_env, value: Self) -> Result<sys::napi_value> {
        let mut result = std::ptr::null_mut();
        napi::check_status!(unsafe { napi_create_bigint_uint64(env, value.0, &mut result) })?;
        Ok(result)
    }
}

fn bigint_u64(value: &NativeU64, _name: &str) -> Result<u64> {
    Ok(value.0)
}

fn u64_bigint(value: u64) -> NativeU64 {
    NativeU64(value)
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeInferenceSamplingOptions {
    pub temperature: f64,
    pub top_k: u32,
    pub top_p: f64,
    pub seed: NativeU64,
}

#[napi(object)]
#[derive(Clone, Default)]
pub struct NativeInferenceSamplingOverrides {
    pub temperature: Option<f64>,
    pub top_k: Option<u32>,
    pub top_p: Option<f64>,
    pub seed: Option<NativeU64>,
}

fn inference_sampling(
    value: &NativeInferenceSamplingOptions,
    phase: &str,
) -> Result<SamplingOptions> {
    if !value.temperature.is_finite() || value.temperature < 0.0 {
        return Err(inference_error(
            phase,
            "temperature must be finite and non-negative",
        ));
    }
    if !value.top_p.is_finite() || value.top_p <= 0.0 || value.top_p > 1.0 {
        return Err(inference_error(phase, "topP must be in (0, 1]"));
    }
    Ok(SamplingOptions {
        temperature: value.temperature,
        top_k: (value.top_k != 0).then_some(value.top_k as usize),
        top_p: value.top_p,
        seed: bigint_u64(&value.seed, "seed")?,
        counter: 0,
    })
}

fn inference_sampling_override(
    value: &NativeInferenceSamplingOverrides,
    mut base: SamplingOptions,
    phase: &str,
) -> Result<SamplingOptions> {
    if let Some(temperature) = value.temperature {
        base.temperature = temperature;
    }
    if let Some(top_k) = value.top_k {
        base.top_k = (top_k != 0).then_some(top_k as usize);
    }
    if let Some(top_p) = value.top_p {
        base.top_p = top_p;
    }
    if let Some(seed) = &value.seed {
        base.seed = bigint_u64(seed, "seed")?;
    }
    inference_sampling(
        &NativeInferenceSamplingOptions {
            temperature: base.temperature,
            top_k: base.top_k.unwrap_or(0) as u32,
            top_p: base.top_p,
            seed: u64_bigint(base.seed),
        },
        phase,
    )
}

fn coordinate_seed(
    seed: u64,
    sequence_id: u64,
    absolute_position: u64,
    purpose: SamplingPurpose,
    subcounter: u64,
) -> u64 {
    let mut key = seed;
    for component in [sequence_id, absolute_position, purpose as u64, subcounter] {
        key = purpose_counter(key, SamplingPurpose::Target, component);
    }
    key
}

fn sample_value_at(
    value: &Value,
    mut options: SamplingOptions,
    sequence_id: u64,
    position: u64,
    purpose: SamplingPurpose,
    cancelled: impl FnMut() -> bool,
) -> Result<u32> {
    options.seed = coordinate_seed(options.seed, sequence_id, position, purpose, 0);
    options.counter = 0;
    sample_blocking(value, options, cancelled)
}

#[derive(Clone)]
struct InferencePrograms {
    target_prefill: Executable,
    target_decode: Executable,
    target_verify: Option<Executable>,
    target_pool: Arc<PoolInner>,
    proposer_prefill: Option<Executable>,
    proposer_decode: Option<Executable>,
    proposer_pool: Option<Arc<PoolInner>>,
    replay_prefill: Option<Executable>,
    #[allow(dead_code)]
    replay_decode: Option<Executable>,
    replay_verify: Option<Executable>,
    replay_pool: Option<Arc<PoolInner>>,
    max_draft_tokens: usize,
    batch_size: usize,
    token_dtype: DType,
    #[allow(dead_code)]
    default_sampling: SamplingOptions,
    // Sessions do not orchestrate this stage plan yet. Storing it here keeps
    // validation and retention atomic with the exact-chain bundle.
    #[allow(dead_code)]
    proposer_plan: Option<ValidatedProposerPlan>,
    history_lookup: Option<HistoryLookupConfig>,
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeInferenceValueMetadata {
    pub dtype: NativeDType,
    pub shape: Vec<u32>,
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeInferenceHiddenTap {
    pub name: String,
    /// Root index in each target executable. Root zero remains target logits.
    pub output_root: u32,
    pub value: NativeInferenceValueMetadata,
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeInferenceSharedTensor {
    /// `TokenEmbedding` or `LmHead`.
    pub kind: String,
    pub name: String,
    pub value: NativeInferenceValueMetadata,
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeInferenceValueRef {
    /// PendingTokens, CandidatePrefix, CommittedHistory, TargetHidden,
    /// SharedTokenEmbedding, SharedLmHead, or StageOutput.
    pub kind: String,
    pub target_output: Option<u32>,
    pub stage: Option<u32>,
    pub output: Option<u32>,
    /// Required concrete metadata for external token/history values.
    pub value: Option<NativeInferenceValueMetadata>,
    /// Select `[physicalLane, ..]` as a one-row view. Valid only for TargetHidden.
    pub select_target_row: Option<bool>,
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeInferenceStageInput {
    pub slot: u32,
    pub value: NativeInferenceValueRef,
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeInferenceStage {
    /// Stable operation identifier, for example `ParallelBlock`.
    pub operation_id: String,
    /// Stable layout/search identifier when the operation has one.
    pub layout_id: Option<String>,
    pub history_lookup: Option<NativeHistoryLookupLayout>,
    pub inputs: Vec<NativeInferenceStageInput>,
    pub outputs: Vec<NativeInferenceValueMetadata>,
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeHistoryLookupLayout {
    pub id: String,
    pub min_match_tokens: u32,
    pub max_match_tokens: u32,
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeInferenceStatePlan {
    /// `None` or `Kv`.
    pub kind: String,
    pub schema_id: Option<String>,
    /// `None`, `AutoregressiveChain`, or `Replay`.
    pub commit_kind: String,
    pub commit_stages: Vec<u32>,
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeInferenceOutputPlan {
    pub topology: String,
    pub probabilities: String,
    pub token_ids: NativeInferenceValueRef,
    pub probability_rows: Option<NativeInferenceValueRef>,
    pub parents: Option<NativeInferenceValueRef>,
    pub confidence: Option<NativeInferenceValueRef>,
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeInferenceTokenMap {
    /// `Identity` or `Table`.
    pub kind: String,
    pub fingerprint: String,
    pub proposer_vocabulary: Option<u32>,
    pub target_ids: Option<Vec<u32>>,
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeInferenceProposerPlan {
    pub vocabulary: u32,
    pub token_map_fingerprint: String,
    pub hidden_taps: Vec<NativeInferenceHiddenTap>,
    pub prefill_hidden_taps: Option<Vec<NativeInferenceHiddenTap>>,
    pub verify_hidden_taps: Option<Vec<NativeInferenceHiddenTap>>,
    pub shared_tensors: Vec<NativeInferenceSharedTensor>,
    pub stages: Vec<NativeInferenceStage>,
    pub state: NativeInferenceStatePlan,
    pub output: NativeInferenceOutputPlan,
    pub token_map: NativeInferenceTokenMap,
    pub trained_max_rows: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ValueMetadata {
    dtype: DType,
    shape: Vec<usize>,
}

impl ValueMetadata {
    fn native(value: &NativeInferenceValueMetadata) -> Self {
        Self {
            dtype: value.dtype.into(),
            shape: value
                .shape
                .iter()
                .map(|&dimension| dimension as usize)
                .collect(),
        }
    }

    fn value(value: &Value) -> Self {
        Self {
            dtype: value.dtype(),
            shape: value.shape().to_vec(),
        }
    }
}

#[derive(Clone)]
#[allow(dead_code)]
struct ValidatedStage {
    executable: Executable,
    inputs: Vec<(usize, NativeInferenceValueRef)>,
}

#[derive(Clone)]
#[allow(dead_code)]
struct ValidatedProposerPlan {
    schema: Option<Arc<NativeInferenceProposerPlan>>,
    target_hidden: HashMap<usize, ValueMetadata>,
    shared: HashMap<String, Value>,
    stages: Vec<ValidatedStage>,
    prefill_taps: Vec<(usize, usize)>,
    verify_taps: Vec<(usize, usize)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HistoryLookupConfig {
    min_match_tokens: usize,
    max_match_tokens: usize,
}

fn history_lookup_config(
    plan: &NativeInferenceProposerPlan,
    shared_tensors: &[&NativeTensor],
    stage_executables: &[&Executable],
) -> Result<Option<HistoryLookupConfig>> {
    let is_history = plan.stages.len() == 1 && plan.stages[0].operation_id == "HistoryLookup";
    if !is_history {
        if plan
            .stages
            .iter()
            .any(|stage| stage.operation_id == "HistoryLookup")
        {
            return Err(inference_error(
                "compile",
                "HistoryLookup must be the only proposer stage",
            ));
        }
        return Ok(None);
    }
    let stage = &plan.stages[0];
    let layout = stage.history_lookup.as_ref();
    let token_ids = &plan.output.token_ids;
    if layout.is_none_or(|layout| {
        layout.id != "suffix-ngram-v1"
            || layout.min_match_tokens == 0
            || layout.min_match_tokens > layout.max_match_tokens
    }) || stage.layout_id.as_deref() != Some("suffix-ngram-v1")
        || !stage.inputs.is_empty()
        || stage.outputs.len() != 1
        || stage.outputs[0].shape.len() != 1
        || !matches!(DType::from(stage.outputs[0].dtype), DType::U32 | DType::I64)
        || !stage_executables.is_empty()
        || !shared_tensors.is_empty()
        || !plan.hidden_taps.is_empty()
        || !plan.shared_tensors.is_empty()
        || plan.state.kind != "None"
        || plan.state.schema_id.is_some()
        || plan.state.commit_kind != "None"
        || !plan.state.commit_stages.is_empty()
        || plan.output.topology != "Chains"
        || plan.output.probabilities != "Deterministic"
        || token_ids.kind != "StageOutput"
        || token_ids.stage != Some(0)
        || token_ids.output != Some(0)
        || plan.output.probability_rows.is_some()
        || plan.output.parents.is_some()
        || plan.output.confidence.is_some()
        || plan.token_map.kind != "Identity"
        || plan.token_map.proposer_vocabulary.is_some()
        || plan.token_map.target_ids.is_some()
        || plan.token_map.fingerprint != plan.token_map_fingerprint
    {
        return Err(inference_error(
            "compile",
            "HistoryLookup requires one native suffix-ngram-v1 deterministic identity chain",
        ));
    }
    let layout = layout.expect("history layout was validated");
    Ok(Some(HistoryLookupConfig {
        min_match_tokens: layout.min_match_tokens as usize,
        max_match_tokens: layout.max_match_tokens as usize,
    }))
}

fn output_metadata(executable: &Executable, output: usize) -> Option<ValueMetadata> {
    executable
        .inner
        .executable
        .signature
        .outputs
        .get(output)
        .map(|value| ValueMetadata {
            dtype: value.dtype,
            shape: value.shape.clone(),
        })
}

fn dense_tap_output(batch: usize, semantic_root: usize) -> usize {
    batch + semantic_root - 1
}

#[allow(dead_code)]
fn routed_target_row(value: &Value, physical_lane: usize) -> std::result::Result<Value, String> {
    let layout = &value.tensor().layout;
    if layout.rank() == 0 || physical_lane >= layout.shape()[0] {
        return Err(format!(
            "inference[proposer]: target hidden row {physical_lane} is outside shape {:?}",
            layout.shape()
        ));
    }
    Ok(Value(value.tensor().view(layout.narrow(
        0,
        physical_lane,
        1,
    ))))
}

fn resolved_route_metadata(
    route: &NativeInferenceValueRef,
    hidden: &HashMap<usize, ValueMetadata>,
    shared: &HashMap<String, ValueMetadata>,
    stage_outputs: &[Vec<ValueMetadata>],
    current_stage: usize,
) -> Result<Option<ValueMetadata>> {
    let no_indexes =
        || route.target_output.is_none() && route.stage.is_none() && route.output.is_none();
    let selected = route.select_target_row.unwrap_or(false);
    let metadata = match route.kind.as_str() {
        "PendingTokens" | "CandidatePrefix" | "CommittedHistory" => {
            if !no_indexes() || selected {
                return Err(inference_error(
                    "compile",
                    "external ValueRef carries invalid indices",
                ));
            }
            return route
                .value
                .as_ref()
                .map(ValueMetadata::native)
                .map(Some)
                .ok_or_else(|| {
                    inference_error("compile", "external ValueRef is missing metadata")
                });
        }
        "TargetHidden" => {
            if route.stage.is_some() || route.output.is_some() || route.value.is_some() {
                return Err(inference_error(
                    "compile",
                    "TargetHidden carries stage indices",
                ));
            }
            let root = route
                .target_output
                .ok_or_else(|| inference_error("compile", "TargetHidden is missing targetOutput"))?
                as usize;
            hidden.get(&root).cloned().ok_or_else(|| {
                inference_error(
                    "compile",
                    "TargetHidden references an undeclared output root",
                )
            })?
        }
        "SharedTokenEmbedding" | "SharedLmHead" => {
            if !no_indexes() || selected || route.value.is_some() {
                return Err(inference_error(
                    "compile",
                    "shared ValueRef carries invalid indices",
                ));
            }
            let kind = if route.kind == "SharedTokenEmbedding" {
                "TokenEmbedding"
            } else {
                "LmHead"
            };
            shared.get(kind).cloned().ok_or_else(|| {
                inference_error("compile", "ValueRef references an undeclared shared tensor")
            })?
        }
        "StageOutput" => {
            if route.target_output.is_some() || selected || route.value.is_some() {
                return Err(inference_error(
                    "compile",
                    "StageOutput carries invalid routing metadata",
                ));
            }
            let stage = route
                .stage
                .ok_or_else(|| inference_error("compile", "StageOutput is missing stage"))?
                as usize;
            let output = route
                .output
                .ok_or_else(|| inference_error("compile", "StageOutput is missing output"))?
                as usize;
            if stage >= current_stage {
                return Err(inference_error(
                    "compile",
                    "StageOutput routes must point backward",
                ));
            }
            stage_outputs
                .get(stage)
                .and_then(|outputs| outputs.get(output))
                .cloned()
                .ok_or_else(|| inference_error("compile", "StageOutput index is out of range"))?
        }
        _ => return Err(inference_error("compile", "unsupported ValueRef kind")),
    };
    if selected {
        if route.kind != "TargetHidden" || metadata.shape.is_empty() || metadata.shape[0] == 0 {
            return Err(inference_error(
                "compile",
                "row selection requires a nonempty TargetHidden",
            ));
        }
        let mut shape = metadata.shape;
        shape[0] = 1;
        return Ok(Some(ValueMetadata {
            dtype: metadata.dtype,
            shape,
        }));
    }
    Ok(Some(metadata))
}

#[derive(Default)]
struct InferenceIds {
    next_sequence_id: u64,
    next_round_id: u64,
}

fn reserve_inference_ids(ids: &mut InferenceIds, sequence_count: usize) -> Result<(u64, Vec<u64>)> {
    let next_round_id = ids
        .next_round_id
        .checked_add(1)
        .ok_or_else(|| inference_error("admission", "round ID space exhausted"))?;
    let count = u64::try_from(sequence_count)
        .map_err(|_| inference_error("admission", "sequence ID space exhausted"))?;
    let next_sequence_id = ids
        .next_sequence_id
        .checked_add(count)
        .ok_or_else(|| inference_error("admission", "sequence ID space exhausted"))?;
    let round_id = ids.next_round_id;
    let sequence_ids = (ids.next_sequence_id..next_sequence_id).collect();
    ids.next_round_id = next_round_id;
    ids.next_sequence_id = next_sequence_id;
    Ok((round_id, sequence_ids))
}

#[napi]
pub struct NativeInferenceArtifact {
    programs: Arc<InferencePrograms>,
    diagnostics: Arc<Mutex<NativeInferenceDiagnosticsState>>,
    ids: Arc<Mutex<InferenceIds>>,
}

#[derive(Default)]
struct NativeInferenceDiagnosticsState {
    rounds_started: u64,
    rounds_completed: u64,
    rounds_recovered: u64,
    last_round_id: Option<u64>,
    last_failure_phase: Option<String>,
    ordinary_rounds: u64,
    speculative_rounds: u64,
    proposed_tokens: u64,
    accepted_tokens: u64,
    emitted_tokens: u64,
    provisional_blocks: u64,
    rolled_back_blocks: u64,
    draft_nanos: u64,
    verification_nanos: u64,
    accepted_length_histogram: Vec<u64>,
    target_pool_high_water_blocks: u64,
    proposer_pool_high_water_blocks: u64,
}

#[napi(object)]
pub struct NativeInferenceDiagnostics {
    pub rounds_started: NativeU64,
    pub rounds_completed: NativeU64,
    pub rounds_recovered: NativeU64,
    pub last_round_id: Option<NativeU64>,
    pub last_failure_phase: Option<String>,
    pub ordinary_rounds: NativeU64,
    pub speculative_rounds: NativeU64,
    pub proposed_tokens: NativeU64,
    pub accepted_tokens: NativeU64,
    pub emitted_tokens: NativeU64,
    pub provisional_blocks: NativeU64,
    pub rolled_back_blocks: NativeU64,
    pub draft_nanos: NativeU64,
    pub verification_nanos: NativeU64,
    pub accepted_length_histogram: Vec<NativeU64>,
    pub target_pool_high_water_blocks: NativeU64,
    pub proposer_pool_high_water_blocks: Option<NativeU64>,
}

fn same_state(left: KvStateSchema, right: KvStateSchema) -> bool {
    left.max_tokens == right.max_tokens
        && left.block_size == right.block_size
        && left.kv_dtype == right.kv_dtype
        && left.window == right.window
        && left.batch == right.batch
        && left.layers == right.layers
        && left.kv_heads == right.kv_heads
        && left.head_dim == right.head_dim
        && left.kda == right.kda
        && left.conv == right.conv
}

fn inference_token_shape(executable: &Executable) -> Result<(usize, usize, DType)> {
    let schema = executable
        .state
        .ok_or_else(|| inference_error("compile", "inference programs must be stateful"))?;
    let slots = executable
        .inner
        .slots
        .iter()
        .enumerate()
        .filter(|(index, slot)| {
            !slot.scalar && !(schema.cursor_tensor && *index as u32 == schema.cursor_slot)
        })
        .map(|(_, slot)| slot)
        .collect::<Vec<_>>();
    if slots.len() != 1
        || slots[0].shape.len() != 2
        || slots[0].shape[0] == 0
        || slots[0].shape[1] == 0
    {
        return Err(inference_error(
            "compile",
            "each inference program must have exactly one rank-two token input",
        ));
    }
    if !matches!(slots[0].dtype, DType::U32 | DType::I64) {
        return Err(inference_error(
            "compile",
            "token inputs must be u32 or i64",
        ));
    }
    Ok((slots[0].shape[0], slots[0].shape[1], slots[0].dtype))
}

fn validate_inference_program(
    executable: &Executable,
    pool: &NativeKvPool,
    batch_size: usize,
    steps: usize,
    token_dtype: DType,
    packed: Option<usize>,
) -> Result<KvStateSchema> {
    let schema = executable
        .state
        .ok_or_else(|| inference_error("compile", "inference program is stateless"))?;
    validate_pool_schema(&schema, &pool.inner)
        .map_err(|error| inference_error("compile", error.reason))?;
    let (rows, actual_steps, actual_dtype) = inference_token_shape(executable)?;
    let expected_rows = packed.map_or(batch_size, |width| batch_size * width);
    if schema.batch != batch_size
        || rows != expected_rows
        || actual_steps != steps
        || actual_dtype != token_dtype
        || schema.packed_rows_per_sequence != packed
    {
        return Err(inference_error(
            "compile",
            "program shape/state does not match inference bundle",
        ));
    }
    Ok(schema)
}

fn validate_replay_program(
    executable: &Executable,
    pool: &NativeKvPool,
    batch: usize,
    taps: &[(usize, usize)],
    target: &Executable,
) -> Result<KvStateSchema> {
    let schema = executable
        .state
        .ok_or_else(|| inference_error("compile", "replay program is stateless"))?;
    validate_pool_schema(&schema, &pool.inner)
        .map_err(|error| inference_error("compile", error.reason))?;
    let slots = executable
        .inner
        .slots
        .iter()
        .enumerate()
        .filter(|(index, slot)| {
            !slot.scalar && !(schema.cursor_tensor && *index as u32 == schema.cursor_slot)
        })
        .map(|(_, slot)| slot)
        .collect::<Vec<_>>();
    if schema.batch != batch
        || schema.packed_rows_per_sequence.is_some()
        || slots.len() != taps.len()
        || executable.inner.slots.iter().any(|slot| slot.scalar)
    {
        return Err(inference_error(
            "compile",
            "replay state or input count does not match its tap phase",
        ));
    }
    for (slot, &(_, physical)) in slots.iter().zip(taps) {
        let source = output_metadata(target, physical)
            .ok_or_else(|| inference_error("compile", "replay tap output is missing"))?;
        if slot.dtype != source.dtype || slot.shape != source.shape {
            return Err(inference_error(
                "compile",
                "replay input metadata does not match its target tap",
            ));
        }
    }
    Ok(schema)
}

fn validate_proposer_plan(
    plan: NativeInferenceProposerPlan,
    shared_tensors: Vec<&NativeTensor>,
    stage_executables: Vec<&Executable>,
    target_prefill: &Executable,
    target_decode: &Executable,
    target_verify: Option<&Executable>,
) -> Result<ValidatedProposerPlan> {
    if plan.vocabulary == 0 || plan.trained_max_rows == 0 || plan.token_map_fingerprint.is_empty() {
        return Err(inference_error(
            "compile",
            "proposer plan has invalid target metadata",
        ));
    }
    if plan.shared_tensors.len() != shared_tensors.len()
        || plan.stages.len() != stage_executables.len()
        || plan.stages.is_empty()
    {
        return Err(inference_error(
            "compile",
            "proposer plan handle arrays do not match their descriptors",
        ));
    }

    let mut target_hidden = HashMap::new();
    let mut hidden_layers = HashSet::new();
    for tap in &plan.hidden_taps {
        let root = tap.output_root as usize;
        if root == 0 || !hidden_layers.insert(tap.name.clone()) || target_hidden.contains_key(&root)
        {
            return Err(inference_error(
                "compile",
                "target hidden taps must have unique names and output roots after logits",
            ));
        }
        let declared = ValueMetadata::native(&tap.value);
        let decode_output = target_decode
            .state
            .map_or(root, |schema| dense_tap_output(schema.batch, root));
        if output_metadata(target_decode, decode_output).as_ref() != Some(&declared) {
            return Err(inference_error(
                "compile",
                "decode hidden tap metadata does not match its split physical output",
            ));
        }
        target_hidden.insert(root, declared);
    }

    let validate_phase_taps = |taps: &[NativeInferenceHiddenTap],
                               executable: &Executable,
                               split: bool|
     -> Result<Vec<(usize, usize)>> {
        let mut names = HashSet::new();
        let mut roots = HashSet::new();
        taps.iter()
            .map(|tap| {
                let root = tap.output_root as usize;
                if root == 0 || !names.insert(tap.name.clone()) || !roots.insert(root) {
                    return Err(inference_error(
                        "compile",
                        "phase hidden taps must have unique names and nonzero roots",
                    ));
                }
                let physical = if split {
                    dense_tap_output(batch_size_from(executable)?, root)
                } else {
                    root
                };
                if output_metadata(executable, physical).as_ref()
                    != Some(&ValueMetadata::native(&tap.value))
                {
                    return Err(inference_error(
                        "compile",
                        "phase hidden tap metadata does not match its physical output",
                    ));
                }
                Ok((root, physical))
            })
            .collect()
    };
    let prefill_taps = validate_phase_taps(
        plan.prefill_hidden_taps
            .as_deref()
            .unwrap_or(&plan.hidden_taps),
        target_prefill,
        true,
    )?;
    let verify_taps = if let Some(verify) = target_verify {
        validate_phase_taps(
            plan.verify_hidden_taps
                .as_deref()
                .unwrap_or(&plan.hidden_taps),
            verify,
            false,
        )?
    } else {
        Vec::new()
    };

    let mut shared = HashMap::new();
    let mut shared_metadata = HashMap::new();
    let mut shared_names = HashSet::new();
    for (descriptor, tensor) in plan.shared_tensors.iter().zip(shared_tensors) {
        if !matches!(descriptor.kind.as_str(), "TokenEmbedding" | "LmHead")
            || descriptor.name.is_empty()
            || !shared_names.insert(descriptor.name.as_str())
            || shared.contains_key(&descriptor.kind)
        {
            return Err(inference_error(
                "compile",
                "shared tensor keys and names must be unique",
            ));
        }
        let value = tensor.value_cloned()?;
        let declared = ValueMetadata::native(&descriptor.value);
        shared_metadata.insert(descriptor.kind.clone(), declared);
        shared.insert(descriptor.kind.clone(), value);
    }

    let mut outputs = Vec::with_capacity(plan.stages.len());
    let mut stages = Vec::with_capacity(plan.stages.len());
    for (stage_index, (stage, executable)) in plan.stages.iter().zip(stage_executables).enumerate()
    {
        let layout_required = matches!(
            stage.operation_id.as_str(),
            "ParallelBlock" | "TreeExpand" | "HistoryLookup"
        );
        if !matches!(
            stage.operation_id.as_str(),
            "Autoregressive"
                | "ParallelBlock"
                | "SequentialHead"
                | "TreeExpand"
                | "TargetPath"
                | "HistoryLookup"
        ) || stage.layout_id.as_ref().is_some_and(|id| id.is_empty())
            || layout_required != stage.layout_id.is_some()
            || stage.outputs.is_empty()
            || stage.outputs.len() != executable.inner.executable.signature.outputs.len()
        {
            return Err(inference_error(
                "compile",
                "stage operation or output metadata is invalid",
            ));
        }
        let declared_outputs = stage
            .outputs
            .iter()
            .map(ValueMetadata::native)
            .collect::<Vec<_>>();
        if declared_outputs
            .iter()
            .enumerate()
            .any(|(output, declared)| {
                output_metadata(executable, output).as_ref() != Some(declared)
            })
        {
            return Err(inference_error(
                "compile",
                "stage output metadata does not match its executable",
            ));
        }

        let schema = executable.state;
        let tensor_slots = executable
            .inner
            .slots
            .iter()
            .enumerate()
            .filter(|(slot, value)| {
                !value.scalar
                    && !schema.is_some_and(|state| {
                        state.cursor_tensor && *slot as u32 == state.cursor_slot
                    })
            })
            .map(|(slot, value)| (slot, value))
            .collect::<Vec<_>>();
        if executable.inner.slots.iter().any(|slot| slot.scalar)
            || tensor_slots.len() != stage.inputs.len()
        {
            return Err(inference_error(
                "compile",
                "stage inputs do not cover its executable tensor slots",
            ));
        }
        let mut seen = HashSet::new();
        let mut validated_inputs = Vec::with_capacity(stage.inputs.len());
        for input in &stage.inputs {
            let slot = input.slot as usize;
            if !seen.insert(slot) {
                return Err(inference_error(
                    "compile",
                    "stage input slots must be unique",
                ));
            }
            let declared_slot = tensor_slots
                .iter()
                .find_map(|(index, value)| (*index == slot).then_some(*value))
                .ok_or_else(|| {
                    inference_error("compile", "stage input slot is not a tensor slot")
                })?;
            let routed = resolved_route_metadata(
                &input.value,
                &target_hidden,
                &shared_metadata,
                &outputs,
                stage_index,
            )?
            .expect("all native routes carry concrete metadata");
            let shared = matches!(
                input.value.kind.as_str(),
                "SharedTokenEmbedding" | "SharedLmHead"
            );
            if !shared
                && (routed.dtype != declared_slot.dtype || routed.shape != declared_slot.shape)
            {
                return Err(inference_error(
                    "compile",
                    "stage input route metadata does not match its executable slot",
                ));
            }
            validated_inputs.push((slot, input.value.clone()));
        }
        validated_inputs.sort_by_key(|(slot, _)| *slot);
        outputs.push(declared_outputs);
        stages.push(ValidatedStage {
            executable: (*executable).clone(),
            inputs: validated_inputs,
        });
    }

    let committed = plan
        .state
        .commit_stages
        .iter()
        .map(|&stage| stage as usize)
        .collect::<HashSet<_>>();
    let stateful = stages
        .iter()
        .enumerate()
        .filter_map(|(index, stage)| stage.executable.state.is_some().then_some(index))
        .collect::<HashSet<_>>();
    match plan.state.kind.as_str() {
        "None"
            if plan.state.schema_id.is_none()
                && plan.state.commit_kind == "None"
                && plan.state.commit_stages.is_empty()
                && stages.iter().all(|stage| stage.executable.state.is_none()) => {}
        "Kv" if plan
            .state
            .schema_id
            .as_ref()
            .is_some_and(|id| !id.is_empty())
            && matches!(
                plan.state.commit_kind.as_str(),
                "AutoregressiveChain" | "Replay"
            )
            && !plan.state.commit_stages.is_empty()
            && committed.len() == plan.state.commit_stages.len()
            && (plan.state.commit_kind != "AutoregressiveChain" || committed.len() == 1)
            && stateful.iter().all(|stage| committed.contains(stage))
            && plan
                .state
                .commit_stages
                .iter()
                .all(|&stage| (stage as usize) < stages.len()) => {}
        _ => {
            return Err(inference_error(
                "compile",
                "proposer state/commit metadata is invalid",
            ))
        }
    }

    let mut output_metadata = Vec::new();
    for route in [
        Some(&plan.output.token_ids),
        plan.output.probability_rows.as_ref(),
        plan.output.parents.as_ref(),
        plan.output.confidence.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        output_metadata.push(
            resolved_route_metadata(
                route,
                &target_hidden,
                &shared_metadata,
                &outputs,
                stages.len(),
            )?
            .expect("all native routes carry concrete metadata"),
        );
    }
    if !matches!(plan.output.topology.as_str(), "Chains" | "Trees")
        || !matches!(
            plan.output.probabilities.as_str(),
            "CausalNormalized" | "Deterministic" | "Unavailable"
        )
        || (plan.output.probabilities == "CausalNormalized")
            != plan.output.probability_rows.is_some()
        || (plan.output.topology == "Trees" && plan.output.parents.is_none())
    {
        return Err(inference_error(
            "compile",
            "proposer output metadata is invalid",
        ));
    }
    let token_ids = &output_metadata[0];
    let parallel = plan.stages.len() == 1 && plan.stages[0].operation_id == "ParallelBlock";
    if (!parallel && token_ids.shape.len() != 1)
        || (parallel && token_ids.shape.len() != 2)
        || !matches!(token_ids.dtype, DType::U32 | DType::I64)
    {
        return Err(inference_error(
            "compile",
            "proposer tokenIds must be a rank-one integer value",
        ));
    }
    if plan.token_map.fingerprint != plan.token_map_fingerprint {
        return Err(inference_error(
            "compile",
            "token-map fingerprint does not match target contract",
        ));
    }
    match plan.token_map.kind.as_str() {
        "Identity"
            if plan.token_map.proposer_vocabulary.is_none()
                && plan.token_map.target_ids.is_none() => {}
        "Table" => {
            let vocabulary = plan.token_map.proposer_vocabulary.ok_or_else(|| {
                inference_error("compile", "table token map is missing proposerVocabulary")
            })? as usize;
            let ids = plan.token_map.target_ids.as_ref().ok_or_else(|| {
                inference_error("compile", "table token map is missing targetIds")
            })?;
            if vocabulary == 0
                || ids.len() != vocabulary
                || ids.iter().any(|&token| token >= plan.vocabulary)
            {
                return Err(inference_error(
                    "compile",
                    "table token map is out of range",
                ));
            }
        }
        _ => return Err(inference_error("compile", "token-map metadata is invalid")),
    }

    Ok(ValidatedProposerPlan {
        schema: Some(Arc::new(plan)),
        target_hidden,
        shared,
        stages,
        prefill_taps,
        verify_taps,
    })
}

fn batch_size_from(executable: &Executable) -> Result<usize> {
    executable
        .state
        .map(|schema| schema.batch)
        .ok_or_else(|| inference_error("compile", "tap executable must be stateful"))
}

/// Executes the stateless subset of a validated stage DAG with direct `Value`
/// bindings. ParallelBlock and SequentialHead orchestration can call this
/// function. Sessions still handle stateful and external token or history
/// routes.
#[allow(dead_code)]
fn execute_stateless_stage_dag(
    plan: &ValidatedProposerPlan,
    target_outputs: &[Value],
    physical_lane: usize,
    cancelled: &CancellationFlag,
) -> std::result::Result<Vec<Vec<Value>>, String> {
    if plan
        .schema
        .as_ref()
        .is_some_and(|schema| schema.state.kind != "None")
    {
        return Err(
            "inference[proposer]: stateless DAG helper received a stateful plan".to_string(),
        );
    }
    let mut stage_outputs: Vec<Vec<Value>> = Vec::with_capacity(plan.stages.len());
    for stage in &plan.stages {
        if stage.executable.state.is_some() {
            return Err(
                "inference[proposer]: stateless DAG helper received a stateful stage".to_string(),
            );
        }
        let mut inputs = Vec::with_capacity(stage.inputs.len());
        for (_, route) in &stage.inputs {
            let value = match route.kind.as_str() {
                "TargetHidden" => {
                    let root = route.target_output.expect("validated target output") as usize;
                    let value = target_outputs.get(root).ok_or_else(|| {
                        "inference[proposer]: target output is unavailable".to_string()
                    })?;
                    if plan.target_hidden.get(&root) != Some(&ValueMetadata::value(value)) {
                        return Err(
                            "inference[proposer]: target output metadata changed after compilation"
                                .to_string(),
                        );
                    }
                    if route.select_target_row.unwrap_or(false) {
                        routed_target_row(value, physical_lane)?
                    } else {
                        value.clone()
                    }
                }
                "SharedTokenEmbedding" => {
                    plan.shared.get("TokenEmbedding").cloned().ok_or_else(|| {
                        "inference[proposer]: shared embedding is unavailable".to_string()
                    })?
                }
                "SharedLmHead" => plan.shared.get("LmHead").cloned().ok_or_else(|| {
                    "inference[proposer]: shared LM head is unavailable".to_string()
                })?,
                "StageOutput" => stage_outputs
                    .get(route.stage.expect("validated stage") as usize)
                    .and_then(|values| values.get(route.output.expect("validated output") as usize))
                    .cloned()
                    .ok_or_else(|| {
                        "inference[proposer]: stage output is unavailable".to_string()
                    })?,
                _ => {
                    return Err(
                        "inference[proposer]: external route requires session orchestration"
                            .to_string(),
                    )
                }
            };
            inputs.push(value);
        }
        let outputs = executable::execute(
            &stage.executable.inner.executable,
            &inputs,
            &stage.executable.inner.generated_bindings,
            cancelled,
            None,
        )?;
        stage_outputs.push(outputs);
    }
    Ok(stage_outputs)
}

#[napi]
#[allow(clippy::too_many_arguments)]
pub fn compile_inference(
    target_prefill: &Executable,
    target_decode: &Executable,
    target_verify: Option<&Executable>,
    target_pool: &NativeKvPool,
    proposer_prefill: Option<&Executable>,
    proposer_decode: Option<&Executable>,
    proposer_pool: Option<&NativeKvPool>,
    max_draft_tokens: u32,
    batch_size: u32,
    token_dtype: NativeDType,
    sampling: NativeInferenceSamplingOptions,
    proposer_plan: Option<NativeInferenceProposerPlan>,
    shared_tensors: Option<Vec<&NativeTensor>>,
    stage_executables: Option<Vec<&Executable>>,
    replay_prefill: Option<&Executable>,
    replay_decode: Option<&Executable>,
    replay_verify: Option<&Executable>,
    replay_pool: Option<&NativeKvPool>,
) -> Result<NativeInferenceArtifact> {
    let batch_size = batch_size as usize;
    if batch_size == 0 {
        return Err(inference_error("compile", "batchSize must be positive"));
    }
    let token_dtype: DType = token_dtype.into();
    if !matches!(token_dtype, DType::U32 | DType::I64) {
        return Err(inference_error("compile", "tokenDtype must be u32 or i64"));
    }
    let default_sampling = inference_sampling(&sampling, "compile")?;
    let (_, prefill_steps, _) = inference_token_shape(target_prefill)?;
    let target_schema = validate_inference_program(
        target_prefill,
        target_pool,
        batch_size,
        prefill_steps,
        token_dtype,
        None,
    )?;
    let decode_schema =
        validate_inference_program(target_decode, target_pool, batch_size, 1, token_dtype, None)?;
    if !same_state(target_schema, decode_schema) {
        return Err(inference_error(
            "compile",
            "target prefill/decode state schemas differ",
        ));
    }
    let proposer_present =
        proposer_prefill.is_some() || proposer_decode.is_some() || proposer_pool.is_some();
    let generalized_present =
        proposer_plan.is_some() || shared_tensors.is_some() || stage_executables.is_some();
    let replay_present = replay_prefill.is_some()
        || replay_decode.is_some()
        || replay_verify.is_some()
        || replay_pool.is_some();
    if generalized_present
        != (proposer_plan.is_some() && shared_tensors.is_some() && stage_executables.is_some())
    {
        return Err(inference_error(
            "compile",
            "incomplete generalized proposer plan arguments",
        ));
    }
    if proposer_present
        != (proposer_prefill.is_some() && proposer_decode.is_some() && proposer_pool.is_some())
        || proposer_present && generalized_present
        || (proposer_present || generalized_present) != target_verify.is_some()
        || (proposer_present || generalized_present) != (max_draft_tokens > 0)
        || replay_present
            != (replay_prefill.is_some()
                && replay_decode.is_some()
                && replay_verify.is_some()
                && replay_pool.is_some())
    {
        return Err(inference_error(
            "compile",
            "incomplete target/proposer inference bundle",
        ));
    }
    if let Some(verify) = target_verify {
        let width = max_draft_tokens as usize + 1;
        let verify_schema = validate_inference_program(
            verify,
            target_pool,
            batch_size,
            1,
            token_dtype,
            Some(width),
        )?;
        if !same_state(target_schema, verify_schema) {
            return Err(inference_error(
                "compile",
                "target verifier state schema differs",
            ));
        }
    }
    if let (Some(prefill), Some(decode), Some(pool)) =
        (proposer_prefill, proposer_decode, proposer_pool)
    {
        let (_, proposer_steps, _) = inference_token_shape(prefill)?;
        let proposer_schema = validate_inference_program(
            prefill,
            pool,
            batch_size,
            proposer_steps,
            token_dtype,
            None,
        )?;
        let proposer_decode_schema =
            validate_inference_program(decode, pool, batch_size, 1, token_dtype, None)?;
        if !same_state(proposer_schema, proposer_decode_schema)
            || target_schema.max_tokens != proposer_schema.max_tokens
        {
            return Err(inference_error(
                "compile",
                "proposer prefill/decode state schemas differ",
            ));
        }
    }
    let (proposer_plan, history_lookup) = match (proposer_plan, shared_tensors, stage_executables) {
        (Some(plan), Some(shared), Some(stages)) => {
            let history = history_lookup_config(&plan, &shared, &stages)?;
            if let Some(config) = history {
                let target_program = &target_verify
                    .expect("generalized bundle has a verifier")
                    .inner
                    .executable
                    .program;
                let output = target_program
                    .outputs
                    .first()
                    .map(|output| &target_program.values[output.index()]);
                let expected_rows = batch_size
                    .checked_mul(max_draft_tokens as usize + 1)
                    .ok_or_else(|| inference_error("compile", "HistoryLookup rows overflow"))?;
                let expected_proposals = (batch_size as u32)
                    .checked_mul(max_draft_tokens)
                    .ok_or_else(|| inference_error("compile", "HistoryLookup rows overflow"))?;
                if plan.vocabulary == 0
                    || plan.trained_max_rows < max_draft_tokens
                    || DType::from(plan.stages[0].outputs[0].dtype) != token_dtype
                    || plan.stages[0].outputs[0].shape != [expected_proposals]
                    || output.is_none_or(|output| {
                        output.shape.as_ref() != [expected_rows, 1, plan.vocabulary as usize]
                    })
                {
                    return Err(inference_error(
                        "compile",
                        "HistoryLookup metadata does not match verifier geometry",
                    ));
                }
                (None, Some(config))
            } else {
                let validated = validate_proposer_plan(
                    plan,
                    shared,
                    stages,
                    target_prefill,
                    target_decode,
                    target_verify,
                )?;
                (Some(validated), None)
            }
        }
        (None, None, None) => (None, None),
        _ => unreachable!("generalized argument completeness checked"),
    };
    if let Some(plan) = &proposer_plan {
        let schema = plan.schema.as_ref().expect("validated plan retains schema");
        let has_probability_rows = schema.output.probability_rows.is_some();
        let valid_probability_route = if has_probability_rows {
            schema.output.probabilities == "CausalNormalized"
                && schema.stages[0].outputs.len() == 2
                && schema
                    .output
                    .probability_rows
                    .as_ref()
                    .is_some_and(|route| {
                        route.kind == "StageOutput"
                            && route.stage == Some(0)
                            && route.output == Some(1)
                    })
        } else {
            schema.output.probabilities == "Unavailable" && schema.stages[0].outputs.len() == 1
        };
        let parallel = schema.stages.len() == 1
            && schema.stages[0].operation_id == "ParallelBlock"
            && schema.stages[0].layout_id.as_deref() == Some("parallel-block")
            && schema.state.kind == "Kv"
            && schema.state.commit_kind == "Replay"
            && schema.state.commit_stages == [0]
            && schema.output.topology == "Chains"
            && valid_probability_route
            && schema.output.token_ids.kind == "StageOutput"
            && schema.output.token_ids.stage == Some(0)
            && schema.output.token_ids.output == Some(0)
            && schema.output.parents.is_none()
            && schema.output.confidence.is_none()
            && schema.token_map.kind == "Identity"
            && schema.trained_max_rows >= max_draft_tokens
            && schema.stages[0].history_lookup.is_none()
            && schema.stages[0]
                .inputs
                .iter()
                .filter(|input| input.value.kind == "PendingTokens")
                .count()
                == 1
            && schema.stages[0].inputs.iter().all(|input| {
                matches!(
                    input.value.kind.as_str(),
                    "PendingTokens" | "SharedTokenEmbedding" | "SharedLmHead"
                )
            });
        if !parallel || !replay_present {
            return Err(inference_error(
                "compile",
                "generalized execution requires one ParallelBlock stage and a complete replay bundle",
            ));
        }
        let replay_pool = replay_pool.expect("complete replay bundle");
        let replay_prefill = replay_prefill.expect("complete replay bundle");
        let replay_decode = replay_decode.expect("complete replay bundle");
        let replay_verify = replay_verify.expect("complete replay bundle");
        let prefill_schema = validate_replay_program(
            replay_prefill,
            replay_pool,
            batch_size,
            &plan.prefill_taps,
            target_prefill,
        )?;
        let decode_replay_schema = validate_replay_program(
            replay_decode,
            replay_pool,
            batch_size,
            &plan
                .schema
                .as_ref()
                .expect("schema")
                .hidden_taps
                .iter()
                .map(|tap| {
                    (
                        tap.output_root as usize,
                        dense_tap_output(batch_size, tap.output_root as usize),
                    )
                })
                .collect::<Vec<_>>(),
            target_decode,
        )?;
        let verify_replay_schema = validate_replay_program(
            replay_verify,
            replay_pool,
            batch_size,
            &plan.verify_taps,
            target_verify.expect("parallel bundle has verifier"),
        )?;
        let stage_schema = plan.stages[0]
            .executable
            .state
            .ok_or_else(|| inference_error("compile", "ParallelBlock stage must be stateful"))?;
        let stage_output = output_metadata(&plan.stages[0].executable, 0);
        let probability_output = output_metadata(&plan.stages[0].executable, 1);
        let expected_probability_output = has_probability_rows.then(|| ValueMetadata {
            dtype: DType::F32,
            shape: vec![
                batch_size,
                schema.trained_max_rows as usize,
                schema.vocabulary as usize,
            ],
        });
        if !same_state(prefill_schema, decode_replay_schema)
            || !same_state(prefill_schema, verify_replay_schema)
            || !same_state(prefill_schema, stage_schema)
            || stage_schema.packed_rows_per_sequence.is_some()
            || stage_output
                != Some(ValueMetadata {
                    dtype: DType::U32,
                    shape: vec![batch_size, schema.trained_max_rows as usize],
                })
            || probability_output != expected_probability_output
            || target_schema.max_tokens != prefill_schema.max_tokens
        {
            return Err(inference_error(
                "compile",
                "ParallelBlock stage/replay geometry or candidate output is invalid",
            ));
        }
    } else if replay_present {
        return Err(inference_error(
            "compile",
            "replay bundle requires ParallelBlock",
        ));
    }
    Ok(NativeInferenceArtifact {
        programs: Arc::new(InferencePrograms {
            target_prefill: target_prefill.clone(),
            target_decode: target_decode.clone(),
            target_verify: target_verify.cloned(),
            target_pool: target_pool.inner.clone(),
            proposer_prefill: proposer_prefill.cloned(),
            proposer_decode: proposer_decode.cloned(),
            proposer_pool: proposer_pool.map(|pool| pool.inner.clone()),
            replay_prefill: replay_prefill.cloned(),
            replay_decode: replay_decode.cloned(),
            replay_verify: replay_verify.cloned(),
            replay_pool: replay_pool.map(|pool| pool.inner.clone()),
            max_draft_tokens: max_draft_tokens as usize,
            batch_size,
            token_dtype,
            default_sampling,
            proposer_plan,
            history_lookup,
        }),
        diagnostics: Arc::new(Mutex::new(NativeInferenceDiagnosticsState {
            accepted_length_histogram: vec![0; max_draft_tokens as usize + 1],
            ..NativeInferenceDiagnosticsState::default()
        })),
        ids: Arc::new(Mutex::new(InferenceIds::default())),
    })
}

static NEXT_INFERENCE_SESSION_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
struct InferenceSequenceState {
    id: u64,
    slot: usize,
    target: NativeKvSequence,
    proposer: Option<NativeKvSequence>,
    pending: u32,
    history: Vec<u32>,
    generated: u64,
    budget: Option<u64>,
    eos: Vec<u32>,
    terminal: Option<String>,
    sampling: SamplingOptions,
}

#[derive(Clone)]
struct ReceiptPage {
    sequence_id: u64,
    tokens: Vec<u32>,
    stop_reason: Option<String>,
}

#[derive(Clone)]
struct Receipt {
    round_id: u64,
    request: ReceiptRequest,
    pages: Vec<ReceiptPage>,
}

#[derive(Clone, PartialEq, Eq)]
enum ReceiptRequest {
    Add(Vec<AddRequestIdentity>),
    Round(Vec<RoundRequestIdentity>),
}

#[derive(Clone, PartialEq, Eq)]
struct SamplingIdentity {
    temperature: u64,
    top_k: Option<usize>,
    top_p: u64,
    seed: u64,
}

impl From<SamplingOptions> for SamplingIdentity {
    fn from(value: SamplingOptions) -> Self {
        Self {
            temperature: value.temperature.to_bits(),
            top_k: value.top_k,
            top_p: value.top_p.to_bits(),
            seed: value.seed,
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
struct AddRequestIdentity {
    prompt: Vec<u32>,
    sampling: SamplingIdentity,
    max_tokens: u32,
    eos_tokens: Vec<u32>,
}

#[derive(Clone, PartialEq, Eq)]
struct RoundRequestIdentity {
    sequence_id: u64,
    sampling: SamplingOverrideIdentity,
}

#[derive(Clone, PartialEq, Eq)]
struct SamplingOverrideIdentity {
    temperature: Option<u64>,
    top_k: Option<u32>,
    top_p: Option<u64>,
    seed: Option<u64>,
}

impl From<&NativeInferenceSamplingOverrides> for SamplingOverrideIdentity {
    fn from(value: &NativeInferenceSamplingOverrides) -> Self {
        Self {
            temperature: value.temperature.map(f64::to_bits),
            top_k: value.top_k,
            top_p: value.top_p.map(f64::to_bits),
            seed: value.seed.as_ref().map(|seed| seed.0),
        }
    }
}

struct InferenceSessionState {
    closed: bool,
    sequences: HashMap<u64, InferenceSequenceState>,
    receipt: Option<Receipt>,
}

struct InferenceSessionInner {
    id: u64,
    programs: Arc<InferencePrograms>,
    diagnostics: Arc<Mutex<NativeInferenceDiagnosticsState>>,
    ids: Arc<Mutex<InferenceIds>>,
    operation: Arc<tokio::sync::Mutex<()>>,
    state: Mutex<InferenceSessionState>,
}

impl Drop for InferenceSessionInner {
    fn drop(&mut self) {
        if let Ok(state) = self.state.get_mut() {
            for (_, sequence) in state.sequences.drain() {
                sequence.target.return_blocks();
                if let Some(proposer) = sequence.proposer {
                    proposer.return_blocks();
                }
            }
        }
    }
}

#[napi]
#[derive(Clone)]
pub struct NativeInferenceSession {
    inner: Arc<InferenceSessionInner>,
}

#[napi]
#[derive(Clone)]
pub struct NativeInferenceSequence {
    session_id: u64,
    sequence_id: u64,
}

#[napi]
impl NativeInferenceSequence {
    #[napi(getter)]
    pub fn sequence_id(&self) -> NativeU64 {
        u64_bigint(self.sequence_id)
    }
}

#[napi(object)]
pub struct NativeInferenceTokenPage {
    pub sequence_id: NativeU64,
    pub tokens: Vec<u32>,
    pub stop_reason: Option<String>,
}

#[napi(object)]
pub struct NativeInferenceRoundResult {
    pub round_id: NativeU64,
    pub recovered: bool,
    pub pages: Vec<NativeInferenceTokenPage>,
}

#[napi(object)]
pub struct NativeInferenceInspection {
    pub sequence_id: NativeU64,
    pub cursor: NativeU64,
    pub terminal: Option<String>,
}

fn inference_sequence_ids(
    session_id: u64,
    sequences: &[&NativeInferenceSequence],
) -> Result<Vec<u64>> {
    let ids = sequences
        .iter()
        .map(|sequence| sequence.sequence_id)
        .collect::<Vec<_>>();
    if sequences
        .iter()
        .any(|sequence| sequence.session_id != session_id)
        || ids.iter().collect::<HashSet<_>>().len() != ids.len()
    {
        return Err(inference_error(
            "admission",
            "foreign or duplicate inference sequence",
        ));
    }
    Ok(ids)
}

fn receipt_result(
    _session_id: u64,
    receipt: &Receipt,
    recovered: bool,
) -> NativeInferenceRoundResult {
    NativeInferenceRoundResult {
        round_id: u64_bigint(receipt.round_id),
        recovered,
        pages: receipt
            .pages
            .iter()
            .map(|page| NativeInferenceTokenPage {
                sequence_id: u64_bigint(page.sequence_id),
                tokens: page.tokens.clone(),
                stop_reason: page.stop_reason.clone(),
            })
            .collect(),
    }
}

fn validate_finish_lifecycle(state: &InferenceSessionState) -> Result<()> {
    if state.closed {
        return Err(inference_error("finish", "session is closed"));
    }
    if state.receipt.is_some() {
        return Err(inference_error(
            "finish",
            "round receipt must be acknowledged before finishing sequences",
        ));
    }
    Ok(())
}

fn try_lifecycle_operation(
    operation: &Arc<tokio::sync::Mutex<()>>,
    phase: &str,
) -> Result<tokio::sync::OwnedMutexGuard<()>> {
    operation
        .clone()
        .try_lock_owned()
        .map_err(|_| inference_error(phase, "an inference operation is active"))
}

fn prompt_tokens(prompt: &NativeTensor, dtype: DType) -> Result<Vec<u32>> {
    let value = prompt.value_cloned()?;
    if value.shape().len() != 2
        || value.shape()[0] != 1
        || value.shape()[1] == 0
        || value.dtype() != dtype
    {
        return Err(inference_error(
            "admission",
            "prompt must be a nonempty [1,T] token tensor",
        ));
    }
    match dtype {
        DType::U32 => value
            .to_u32_vec()
            .map_err(|error| inference_error("admission", error)),
        DType::I64 => value
            .to_i64_vec()
            .map_err(|error| inference_error("admission", error))?
            .into_iter()
            .map(|token| {
                u32::try_from(token)
                    .map_err(|_| inference_error("admission", "prompt token does not fit u32"))
            })
            .collect(),
        _ => unreachable!("inference token dtype was validated"),
    }
}

fn token_value(dtype: DType, values: Vec<u32>, shape: Vec<usize>) -> Value {
    match dtype {
        DType::U32 => Value(Tensor::from_vec(values, shape)),
        DType::I64 => Value(Tensor::from_vec(
            values.into_iter().map(i64::from).collect(),
            shape,
        )),
        _ => unreachable!("inference token dtype was validated"),
    }
}

fn shadow_sequence(pool: Arc<PoolInner>, state: Arc<Mutex<SeqState>>) -> NativeKvSequence {
    NativeKvSequence {
        pool,
        state,
        run_lock: Arc::new(Mutex::new(())),
        released: Arc::new(AtomicBool::new(false)),
        finalize_releases: false,
    }
}

fn install_prefix_metadata(store: &mut BlockStore, metadata: DeferredPrefixMetadata) {
    for (block, hash) in metadata.hashes {
        store.hashes[block as usize] = Some(hash);
        store.by_hash.entry(hash).or_default().push(block);
    }
    for (hash, snapshot) in metadata.snapshots {
        if store.by_hash.contains_key(&hash) {
            store.snapshots.insert(hash, Arc::new(snapshot));
        }
    }
}

fn publish_deferred_prefixes<R>(
    target_pool: &Arc<PoolInner>,
    target_metadata: &Arc<Mutex<DeferredPrefixMetadata>>,
    proposer: Option<(&Arc<PoolInner>, &Arc<Mutex<DeferredPrefixMetadata>>)>,
    commit: impl FnOnce() -> R,
) -> Result<R> {
    let target_metadata = std::mem::take(
        &mut *target_metadata
            .lock()
            .map_err(|error| inference_error("publish", error))?,
    );
    let proposer = proposer
        .map(|(pool, metadata)| -> Result<_> {
            let metadata = std::mem::take(
                &mut *metadata
                    .lock()
                    .map_err(|error| inference_error("publish", error))?,
            );
            Ok((pool, metadata))
        })
        .transpose()?;

    let Some((proposer_pool, proposer_metadata)) = proposer else {
        let mut target_store = target_pool
            .blocks
            .lock()
            .map_err(|error| inference_error("publish", error))?;
        install_prefix_metadata(&mut target_store, target_metadata);
        let result = commit();
        drop(target_store);
        return Ok(result);
    };

    if Arc::ptr_eq(target_pool, proposer_pool) {
        let mut store = target_pool
            .blocks
            .lock()
            .map_err(|error| inference_error("publish", error))?;
        install_prefix_metadata(&mut store, target_metadata);
        install_prefix_metadata(&mut store, proposer_metadata);
        let result = commit();
        drop(store);
        return Ok(result);
    }

    if Arc::as_ptr(target_pool) as usize <= Arc::as_ptr(proposer_pool) as usize {
        let mut target_store = target_pool
            .blocks
            .lock()
            .map_err(|error| inference_error("publish", error))?;
        let mut proposer_store = proposer_pool
            .blocks
            .lock()
            .map_err(|error| inference_error("publish", error))?;
        install_prefix_metadata(&mut target_store, target_metadata);
        install_prefix_metadata(&mut proposer_store, proposer_metadata);
        let result = commit();
        drop(proposer_store);
        drop(target_store);
        Ok(result)
    } else {
        let mut proposer_store = proposer_pool
            .blocks
            .lock()
            .map_err(|error| inference_error("publish", error))?;
        let mut target_store = target_pool
            .blocks
            .lock()
            .map_err(|error| inference_error("publish", error))?;
        install_prefix_metadata(&mut target_store, target_metadata);
        install_prefix_metadata(&mut proposer_store, proposer_metadata);
        let result = commit();
        drop(target_store);
        drop(proposer_store);
        Ok(result)
    }
}

async fn prefill_sequences(
    executable: &Executable,
    sequences: &[NativeKvSequence],
    slots: &[u32],
    prompts: &[Vec<u32>],
    token_dtype: DType,
    sample: Option<&[(SamplingOptions, u64)]>,
    cancellation_token: Option<&CancellationToken>,
    prefix_metadata: Arc<Mutex<DeferredPrefixMetadata>>,
    replay: Option<PrefillReplay<'_>>,
) -> Result<Vec<Option<u32>>> {
    let schema = executable.state.expect("inference prefill was validated");
    let (_, chunk, _) = inference_token_shape(executable)?;
    let mut offsets = sequences
        .iter()
        .map(|sequence| {
            sequence
                .state
                .lock()
                .map(|state| state.cursor)
                .map_err(|error| inference_error("prefill", error))
        })
        .collect::<Result<Vec<_>>>()?;
    if offsets
        .iter()
        .zip(prompts)
        .any(|(offset, prompt)| *offset >= prompt.len())
    {
        return Err(inference_error(
            "prefill",
            "matched prefix must leave the final prompt token executable",
        ));
    }
    let mut sampled = vec![None; prompts.len()];
    while offsets
        .iter()
        .zip(prompts)
        .any(|(offset, prompt)| *offset < prompt.len())
    {
        if cancellation_token.is_some_and(|token| token.cancelled()) {
            return Err(inference_error("prefill", "operation aborted"));
        }
        let active = offsets
            .iter()
            .zip(prompts)
            .enumerate()
            .filter_map(|(index, (offset, prompt))| (*offset < prompt.len()).then_some(index))
            .collect::<Vec<_>>();
        let mut input = vec![0u32; schema.batch * chunk];
        let mut rows = Vec::with_capacity(active.len());
        let mut active_sequences = Vec::with_capacity(active.len());
        let mut active_slots = Vec::with_capacity(active.len());
        let mut advances = Vec::with_capacity(active.len());
        let mut finals = Vec::with_capacity(active.len());
        for &index in &active {
            let real = chunk.min(prompts[index].len() - offsets[index]);
            let row = prompts[index][offsets[index]..offsets[index] + real].to_vec();
            let slot = slots[index] as usize;
            input[slot * chunk..slot * chunk + real].copy_from_slice(&row);
            rows.push(row);
            active_sequences.push(&sequences[index]);
            active_slots.push(slots[index]);
            advances.push(real as u32);
            finals.push(offsets[index] + real == prompts[index].len());
        }
        let input = NativeTensor::wrap(token_value(token_dtype, input, vec![schema.batch, chunk]));
        let output = executable
            .execute_stateful_with_prefix_metadata(
                vec![&input],
                active_sequences,
                active_slots.clone(),
                advances,
                rows,
                StatefulInvocation::Tensors,
                cancellation_token,
                Some(prefix_metadata.clone()),
            )
            .await
            .map_err(|error| inference_error("prefill", error.reason))?;
        let StatefulExecutionOutput::Tensors(outputs) = output else {
            unreachable!("prefill tensor execution returned samples")
        };
        if let Some(replay) = &replay {
            let replay_inputs = replay
                .taps
                .iter()
                .map(|&(_, physical)| {
                    outputs.get(physical).ok_or_else(|| {
                        inference_error("proposer", "prefill hidden tap output is missing")
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            replay
                .executable
                .execute_stateful_with_prefix_metadata(
                    replay_inputs,
                    active
                        .iter()
                        .map(|&index| &replay.sequences[index])
                        .collect(),
                    active_slots.clone(),
                    active
                        .iter()
                        .map(|&index| chunk.min(prompts[index].len() - offsets[index]) as u32)
                        .collect(),
                    active
                        .iter()
                        .map(|&index| {
                            let real = chunk.min(prompts[index].len() - offsets[index]);
                            prompts[index][offsets[index]..offsets[index] + real].to_vec()
                        })
                        .collect(),
                    StatefulInvocation::Tensors,
                    cancellation_token,
                    Some(replay.prefix_metadata.clone()),
                )
                .await
                .map_err(|error| inference_error("proposer", error.reason))?;
        }
        if let Some(sampling) = sample {
            for (active_index, &index) in active.iter().enumerate() {
                if finals[active_index] {
                    let output = outputs
                        .get(active_slots[active_index] as usize)
                        .ok_or_else(|| inference_error("sample", "prefill output lane is missing"))?
                        .value_cloned()?;
                    sampled[index] = Some(sample_value_at(
                        &output,
                        sampling[index].0,
                        sampling[index].1,
                        0,
                        SamplingPurpose::Target,
                        || false,
                    )?);
                }
            }
        }
        for &index in &active {
            offsets[index] += chunk.min(prompts[index].len() - offsets[index]);
        }
    }
    Ok(sampled)
}

struct PrefillReplay<'a> {
    executable: &'a Executable,
    sequences: &'a [NativeKvSequence],
    taps: &'a [(usize, usize)],
    prefix_metadata: Arc<Mutex<DeferredPrefixMetadata>>,
}

#[allow(clippy::too_many_arguments)]
async fn execute_packed_shadow(
    executable: &Executable,
    pool: Arc<PoolInner>,
    states: Vec<Arc<Mutex<SeqState>>>,
    slots: Vec<u32>,
    advances: Vec<usize>,
    tokens: Vec<Vec<u32>>,
    packed: PackedCausalRows,
    input: Value,
    cancellation_token: Option<&CancellationToken>,
) -> Result<(Vec<Value>, DeferredPrefixMetadata)> {
    let schema = executable.state.expect("packed verifier was validated");
    let program = executable.inner.executable.clone();
    let generated_bindings = executable.inner.generated_bindings.clone();
    let context = Arc::new(KvContext {
        pool: pool.clone(),
        slots: {
            let mut lanes = vec![None; schema.batch];
            for (state, &slot) in states.iter().zip(&slots) {
                lanes[slot as usize] = Some(state.clone());
            }
            lanes
        },
        advances: {
            let mut lanes = vec![0; schema.batch];
            for (&slot, &advance) in slots.iter().zip(&advances) {
                lanes[slot as usize] = advance;
            }
            lanes
        },
        packed: Some(packed),
        window: schema.window,
        kda: schema.kda,
        conv: schema.conv,
        transaction: Mutex::new(None),
    });
    run_compute_pending(cancellation_token, move |cancelled, _| {
        let frontiers = states
            .iter()
            .map(|state| state.lock().map(|state| state.blocks.len()).unwrap_or(0))
            .collect::<Vec<_>>();
        for (state, &advance) in states.iter().zip(&advances) {
            state
                .lock()
                .map_err(|error| to_napi_err(error.to_string()))?
                .advance = advance;
        }
        let outputs = match executable::execute_stateful(
            &program,
            &[input],
            &generated_bindings,
            cancelled,
            &context,
            &|| true,
        ) {
            Ok(outputs) => outputs,
            Err(error) => {
                for (index, state) in states.iter().enumerate() {
                    if let Ok(mut state) = state.lock() {
                        for block in state.blocks.split_off(frontiers[index]) {
                            pool.unref_block(block);
                        }
                        state.advance = 0;
                    }
                }
                return Err(to_napi_err(error));
            }
        };
        let mut metadata = DeferredPrefixMetadata::default();
        for ((state, row), &advance) in states.iter().zip(&tokens).zip(&advances) {
            let mut state = state
                .lock()
                .map_err(|error| to_napi_err(error.to_string()))?;
            metadata
                .hashes
                .extend(state.note_tokens_deferred(&pool, row));
            state.cursor += advance;
            state.advance = 0;
            pool.stage_recurrent_snapshot(&state, &mut metadata);
        }
        Ok((outputs, metadata))
    })
    .await
}

#[allow(clippy::too_many_arguments)]
async fn execute_parallel_block_detailed(
    programs: &InferencePrograms,
    target_sequences: Vec<&NativeKvSequence>,
    proposer_sequences: Vec<&NativeKvSequence>,
    slots: Vec<u32>,
    pending_tokens: Vec<u32>,
    sampling: Vec<SamplingOptions>,
    sequence_ids: Vec<u64>,
    absolute_positions: Vec<u64>,
    page_limits: Vec<u32>,
    eos_tokens: Vec<Vec<u32>>,
    cancellation_token: Option<&CancellationToken>,
) -> Result<SpeculativeExecution> {
    let count = target_sequences.len();
    let plan = programs
        .proposer_plan
        .as_ref()
        .ok_or_else(|| inference_error("proposer", "ParallelBlock plan is missing"))?;
    let stage = &plan.stages[0];
    let replay = programs
        .replay_verify
        .as_ref()
        .ok_or_else(|| inference_error("proposer", "verify replay is missing"))?;
    let verifier = programs
        .target_verify
        .as_ref()
        .ok_or_else(|| inference_error("verify", "target verifier is missing"))?;
    let target_pool = programs.target_pool.clone();
    let proposer_pool = programs
        .replay_pool
        .as_ref()
        .expect("parallel validation retained replay pool")
        .clone();
    if count == 0
        || proposer_sequences.len() != count
        || slots.len() != count
        || pending_tokens.len() != count
        || sampling.len() != count
        || sequence_ids.len() != count
        || absolute_positions.len() != count
        || page_limits.len() != count
        || eos_tokens.len() != count
    {
        return Err(inference_error(
            "admission",
            "inconsistent ParallelBlock round arrays",
        ));
    }

    let initial_target = target_sequences
        .iter()
        .map(|sequence| {
            let state = sequence
                .state
                .lock()
                .map_err(|error| inference_error("verify", error))?;
            Ok((
                state.cursor,
                state.last_hash,
                state.pending.clone(),
                state.blocks.len(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let initial_proposer = proposer_sequences
        .iter()
        .map(|sequence| {
            let state = sequence
                .state
                .lock()
                .map_err(|error| inference_error("proposer", error))?;
            Ok((
                state.cursor,
                state.last_hash,
                state.pending.clone(),
                state.blocks.len(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    if initial_target
        .iter()
        .zip(&initial_proposer)
        .any(|(target, proposer)| target.0 != proposer.0)
    {
        return Err(inference_error(
            "proposer",
            "target and replay cursors differ",
        ));
    }
    let mut target_shadow = Vec::with_capacity(count);
    let mut proposer_shadow = Vec::with_capacity(count);
    for index in 0..count {
        match clone_speculative_state(&target_pool, &target_sequences[index].state) {
            Ok(state) => target_shadow.push(state),
            Err(error) => {
                discard_speculative_states(&target_pool, &target_shadow);
                discard_speculative_states(&proposer_pool, &proposer_shadow);
                return Err(inference_error("verify", error));
            }
        }
        match clone_speculative_state(&proposer_pool, &proposer_sequences[index].state) {
            Ok(state) => proposer_shadow.push(state),
            Err(error) => {
                discard_speculative_states(&target_pool, &target_shadow);
                discard_speculative_states(&proposer_pool, &proposer_shadow);
                return Err(inference_error("proposer", error));
            }
        }
    }

    let result = async {
        let draft_lengths = (0..count)
            .map(|index| {
                programs
                    .max_draft_tokens
                    .min((page_limits[index] as usize).saturating_sub(1))
                    .min(
                        target_pool
                            .max_tokens
                            .saturating_sub(initial_target[index].0 + 1),
                    )
                    .min(
                        proposer_pool
                            .max_tokens
                            .saturating_sub(initial_proposer[index].0 + 1),
                    )
            })
            .collect::<Vec<_>>();
        let draft_started = std::time::Instant::now();
        let trained_draft_tokens = plan
            .schema
            .as_ref()
            .expect("parallel plan retains schema")
            .trained_max_rows as usize;
        let mut candidates = vec![Vec::new(); count];
        let mut proposal_probabilities = vec![Vec::<Vec<f64>>::new(); count];
        let mut draft_provisional_blocks = 0u64;
        let proposal_active = (0..count)
            .filter(|&index| draft_lengths[index] != 0)
            .collect::<Vec<_>>();
        if !proposal_active.is_empty() {
            let mut draft_states = Vec::with_capacity(proposal_active.len());
            for &index in &proposal_active {
                match clone_speculative_state(&proposer_pool, &proposer_shadow[index]) {
                    Ok(state) => draft_states.push(state),
                    Err(error) => {
                        discard_speculative_states(&proposer_pool, &draft_states);
                        return Err(inference_error("proposer", error));
                    }
                }
            }
            let draft_sequences = draft_states
                .iter()
                .map(|state| shadow_sequence(proposer_pool.clone(), state.clone()))
                .collect::<Vec<_>>();
            let pending = NativeTensor::wrap(token_value(
                programs.token_dtype,
                {
                    let mut values = vec![0; programs.batch_size];
                    for index in 0..count {
                        values[slots[index] as usize] = pending_tokens[index];
                    }
                    values
                },
                vec![programs.batch_size],
            ));
            let mut owned_inputs = Vec::new();
            for (_, route) in &stage.inputs {
                let value =
                    match route.kind.as_str() {
                        "PendingTokens" => pending.value_cloned()?,
                        "SharedTokenEmbedding" => {
                            plan.shared.get("TokenEmbedding").cloned().ok_or_else(|| {
                                inference_error("proposer", "shared token embedding is missing")
                            })?
                        }
                        "SharedLmHead" => plan.shared.get("LmHead").cloned().ok_or_else(|| {
                            inference_error("proposer", "shared LM head is missing")
                        })?,
                        _ => {
                            return Err(inference_error(
                                "proposer",
                                "unsupported ParallelBlock input route",
                            ))
                        }
                    };
                owned_inputs.push(NativeTensor::wrap(value));
            }
            let proposal_metadata = Arc::new(Mutex::new(DeferredPrefixMetadata::default()));
            let output_result = stage
                .executable
                .execute_stateful_with_prefix_metadata(
                    owned_inputs.iter().collect(),
                    draft_sequences.iter().collect(),
                    proposal_active.iter().map(|&index| slots[index]).collect(),
                    proposal_active
                        .iter()
                        .map(|_| (trained_draft_tokens + 1) as u32)
                        .collect(),
                    proposal_active
                        .iter()
                        .map(|&index| vec![pending_tokens[index]; trained_draft_tokens + 1])
                        .collect(),
                    StatefulInvocation::Tensors,
                    cancellation_token,
                    Some(proposal_metadata),
                )
                .await;
            draft_provisional_blocks = draft_states
                .iter()
                .zip(&proposal_active)
                .map(|(state, &index)| {
                    state
                        .lock()
                        .map(|state| {
                            state.blocks.len().saturating_sub(initial_proposer[index].3) as u64
                        })
                        .unwrap_or(0)
                })
                .sum();
            discard_speculative_states(&proposer_pool, &draft_states);
            let output =
                output_result.map_err(|error| inference_error("proposer", error.reason))?;
            let StatefulExecutionOutput::Tensors(outputs) = output else {
                unreachable!("ParallelBlock returned sampled output")
            };
            if outputs.len() == 2 {
                let probabilities = outputs[1]
                    .value_cloned()
                    .map_err(|error| inference_error("proposer", error))?;
                for index in 0..count {
                    for step in 0..draft_lengths[index] {
                        if cancellation_token.is_some_and(|token| token.cancelled()) {
                            return Err(inference_error("proposer", "operation aborted"));
                        }
                        let q = speculative_normalized_row(
                            &probabilities,
                            (slots[index] as usize, step),
                        )
                        .map_err(|error| inference_error("proposer", error))?;
                        let candidate = sample_probabilities(
                            &q,
                            coordinate_seed(
                                sampling[index].seed,
                                sequence_ids[index],
                                absolute_positions[index] + step as u64,
                                SamplingPurpose::Proposal,
                                0,
                            ),
                            0,
                            || cancellation_token.is_some_and(|token| token.cancelled()),
                        )
                        .map_err(|error| inference_error("proposer", error))?;
                        candidates[index].push(candidate);
                        proposal_probabilities[index].push(q);
                    }
                }
            } else {
                let values = outputs
                    .first()
                    .ok_or_else(|| inference_error("proposer", "candidate output is missing"))?
                    .value_cloned()?
                    .to_u32_vec()
                    .map_err(|error| inference_error("proposer", error))?;
                for index in 0..count {
                    let start = slots[index] as usize * trained_draft_tokens;
                    candidates[index]
                        .extend_from_slice(&values[start..start + draft_lengths[index]]);
                }
            }
        }
        let draft_nanos = u64::try_from(draft_started.elapsed().as_nanos())
            .unwrap_or(u64::MAX)
            .max(1);

        let width = programs.max_draft_tokens + 1;
        let graph_rows = programs.batch_size * width;
        let chain_lengths = candidates
            .iter()
            .map(|tokens| tokens.len() + 1)
            .collect::<Vec<_>>();
        let packed = PackedCausalRows::build(
            graph_rows,
            programs.batch_size,
            &slots.iter().map(|slot| *slot as usize).collect::<Vec<_>>(),
            &initial_target
                .iter()
                .map(|state| state.0)
                .collect::<Vec<_>>(),
            &chain_lengths,
        )
        .map_err(|error| inference_error("verify", error))?;
        let mut verify_tokens = vec![0; graph_rows];
        let mut token_rows = Vec::with_capacity(count);
        for index in 0..count {
            let mut row = Vec::with_capacity(chain_lengths[index]);
            row.push(pending_tokens[index]);
            row.extend_from_slice(&candidates[index]);
            verify_tokens[packed.row_offsets[index]..packed.row_offsets[index] + row.len()]
                .copy_from_slice(&row);
            token_rows.push(row);
        }
        let verification_started = std::time::Instant::now();
        let (target_outputs, _) = execute_packed_shadow(
            verifier,
            target_pool.clone(),
            target_shadow.clone(),
            slots.clone(),
            chain_lengths.clone(),
            token_rows,
            packed.clone(),
            speculative_token_input(programs.token_dtype, graph_rows, &verify_tokens, 1)
                .map_err(|error| inference_error("verify", error))?,
            cancellation_token,
        )
        .await?;
        let target_peak_blocks = target_shadow
            .iter()
            .zip(&initial_target)
            .map(|(state, initial)| {
                state
                    .lock()
                    .map(|state| state.blocks.len().saturating_sub(initial.3) as u64)
                    .unwrap_or(0)
            })
            .sum::<u64>();
        let logits = target_outputs
            .first()
            .ok_or_else(|| inference_error("verify", "target logits are missing"))?;
        let mut pages = Vec::with_capacity(count);
        let mut accepted = Vec::with_capacity(count);
        for index in 0..count {
            let mut page = Vec::with_capacity(candidates[index].len() + 1);
            let mut accepted_count = 0;
            let mut rejected = false;
            for (step, &candidate) in candidates[index].iter().enumerate() {
                if cancellation_token.is_some_and(|token| token.cancelled()) {
                    return Err(inference_error("verify", "operation aborted"));
                }
                if !proposal_probabilities[index].is_empty() {
                    let p = speculative_probabilities(
                        logits,
                        Some((packed.row_offsets[index] + step, 0)),
                        sampling[index],
                        &AtomicBool::new(false),
                    )
                    .map_err(|error| inference_error("verify", error))?;
                    let q = &proposal_probabilities[index][step];
                    let accepted = if sampling[index].temperature == 0.0 {
                        p[candidate as usize] == 1.0
                    } else {
                        random_unit_at(sampling_coordinate(
                            sampling[index].seed,
                            sequence_ids[index],
                            absolute_positions[index] + step as u64,
                            SamplingPurpose::Accept,
                            0,
                        )) < (p[candidate as usize] / q[candidate as usize]).min(1.0)
                    };
                    if accepted {
                        page.push(candidate);
                        accepted_count += 1;
                        continue;
                    }
                    let residual = p
                        .iter()
                        .zip(q)
                        .map(|(&p, &q)| (p - q).max(0.0))
                        .collect::<Vec<_>>();
                    page.push(
                        sample_probabilities(
                            &residual,
                            coordinate_seed(
                                sampling[index].seed,
                                sequence_ids[index],
                                absolute_positions[index] + step as u64,
                                SamplingPurpose::Residual,
                                0,
                            ),
                            0,
                            || cancellation_token.is_some_and(|token| token.cancelled()),
                        )
                        .map_err(|error| inference_error("verify", error))?,
                    );
                    rejected = true;
                    break;
                }
                let target = speculative_sample_logits(
                    logits,
                    (packed.row_offsets[index] + step, 0),
                    SamplingOptions {
                        seed: coordinate_seed(
                            sampling[index].seed,
                            sequence_ids[index],
                            absolute_positions[index] + step as u64,
                            SamplingPurpose::Target,
                            0,
                        ),
                        counter: 0,
                        ..sampling[index]
                    },
                    &AtomicBool::new(false),
                )
                .map_err(|error| inference_error("verify", error))?;
                page.push(target);
                if target == candidate {
                    accepted_count += 1;
                } else {
                    rejected = true;
                    break;
                }
            }
            if !rejected {
                if cancellation_token.is_some_and(|token| token.cancelled()) {
                    return Err(inference_error("verify", "operation aborted"));
                }
                let step = candidates[index].len();
                page.push(if !proposal_probabilities[index].is_empty() {
                    let p = speculative_probabilities(
                        logits,
                        Some((packed.row_offsets[index] + step, 0)),
                        sampling[index],
                        &AtomicBool::new(false),
                    )
                    .map_err(|error| inference_error("verify", error))?;
                    sample_probabilities(
                        &p,
                        coordinate_seed(
                            sampling[index].seed,
                            sequence_ids[index],
                            absolute_positions[index] + step as u64,
                            SamplingPurpose::Target,
                            0,
                        ),
                        0,
                        || cancellation_token.is_some_and(|token| token.cancelled()),
                    )
                    .map_err(|error| inference_error("verify", error))?
                } else {
                    speculative_sample_logits(
                        logits,
                        (packed.row_offsets[index] + step, 0),
                        SamplingOptions {
                            seed: coordinate_seed(
                                sampling[index].seed,
                                sequence_ids[index],
                                absolute_positions[index] + step as u64,
                                SamplingPurpose::Target,
                                0,
                            ),
                            counter: 0,
                            ..sampling[index]
                        },
                        &AtomicBool::new(false),
                    )
                    .map_err(|error| inference_error("verify", error))?
                });
            }
            if let Some(position) = page
                .iter()
                .position(|token| eos_tokens[index].contains(token))
            {
                page.truncate(position + 1);
            }
            page.truncate(page_limits[index] as usize);
            accepted.push(accepted_count.min(page.len()));
            pages.push(page);
        }

        let consumed = pages.iter().map(Vec::len).collect::<Vec<_>>();
        let mut target_metadata = DeferredPrefixMetadata::default();
        let mut retained_rows = Vec::with_capacity(count);
        for index in 0..count {
            let retained = std::iter::once(pending_tokens[index])
                .chain(
                    pages[index]
                        .iter()
                        .take(consumed[index].saturating_sub(1))
                        .copied(),
                )
                .collect::<Vec<_>>();
            retained_rows.push(retained.clone());
            let mut state = target_shadow[index]
                .lock()
                .map_err(|error| inference_error("verify", error))?;
            let needed =
                (initial_target[index].0 + consumed[index]).div_ceil(target_pool.block_size);
            for block in state.blocks.drain(needed..) {
                target_pool.unref_block(block);
            }
            state.cursor = initial_target[index].0;
            state.last_hash = initial_target[index].1;
            state.pending = initial_target[index].2.clone();
            state.advance = 0;
            target_metadata.hashes.extend(speculative_note_tokens(
                &mut state,
                &target_pool,
                &retained,
            ));
            state.cursor += consumed[index];
        }

        let replay_inputs = plan
            .verify_taps
            .iter()
            .map(|&(_, physical)| {
                target_outputs
                    .get(physical)
                    .cloned()
                    .map(NativeTensor::wrap)
                    .ok_or_else(|| {
                        inference_error("proposer", "verify hidden tap output is missing")
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let replay_sequences = proposer_shadow
            .iter()
            .map(|state| shadow_sequence(proposer_pool.clone(), state.clone()))
            .collect::<Vec<_>>();
        let replay_metadata = Arc::new(Mutex::new(DeferredPrefixMetadata::default()));
        replay
            .execute_stateful_with_prefix_metadata(
                replay_inputs.iter().collect(),
                replay_sequences.iter().collect(),
                slots.clone(),
                consumed.iter().map(|&length| length as u32).collect(),
                retained_rows,
                StatefulInvocation::Tensors,
                cancellation_token,
                Some(replay_metadata.clone()),
            )
            .await
            .map_err(|error| inference_error("proposer", error.reason))?;
        let paired_peak_blocks = target_shadow
            .iter()
            .zip(&initial_target)
            .map(|(state, initial)| {
                state
                    .lock()
                    .map(|state| state.blocks.len().saturating_sub(initial.3) as u64)
                    .unwrap_or(0)
            })
            .sum::<u64>()
            + proposer_shadow
                .iter()
                .zip(&initial_proposer)
                .map(|(state, initial)| {
                    state
                        .lock()
                        .map(|state| state.blocks.len().saturating_sub(initial.3) as u64)
                        .unwrap_or(0)
                })
                .sum::<u64>();
        if cancellation_token.is_some_and(|token| !token.state.complete()) {
            return Err(inference_error("publish", "operation aborted"));
        }
        let target_metadata = Arc::new(Mutex::new(target_metadata));
        let mut old_target_blocks = Vec::new();
        let mut old_proposer_blocks = Vec::new();
        publish_deferred_prefixes(
            &target_pool,
            &target_metadata,
            Some((&proposer_pool, &replay_metadata)),
            || {
                for index in 0..count {
                    for (canonical, shadow, old_blocks) in [
                        (
                            &target_sequences[index].state,
                            &target_shadow[index],
                            &mut old_target_blocks,
                        ),
                        (
                            &proposer_sequences[index].state,
                            &proposer_shadow[index],
                            &mut old_proposer_blocks,
                        ),
                    ] {
                        let mut canonical =
                            canonical.lock().unwrap_or_else(|error| error.into_inner());
                        let mut shadow = shadow.lock().unwrap_or_else(|error| error.into_inner());
                        let replacement = SeqState {
                            blocks: std::mem::take(&mut shadow.blocks),
                            head: shadow.head,
                            cursor: shadow.cursor,
                            advance: 0,
                            last_hash: shadow.last_hash,
                            pending: std::mem::take(&mut shadow.pending),
                            kda_states: std::mem::take(&mut shadow.kda_states),
                            conv_states: std::mem::take(&mut shadow.conv_states),
                        };
                        let old = std::mem::replace(&mut *canonical, replacement);
                        old_blocks.extend(old.blocks);
                    }
                }
            },
        )?;
        for block in old_target_blocks {
            target_pool.unref_block(block);
        }
        for block in old_proposer_blocks {
            proposer_pool.unref_block(block);
        }
        let verification_nanos = u64::try_from(verification_started.elapsed().as_nanos())
            .unwrap_or(u64::MAX)
            .max(1);
        let baseline_blocks = initial_target
            .iter()
            .map(|state| state.3 as u64)
            .sum::<u64>()
            + initial_proposer
                .iter()
                .map(|state| state.3 as u64)
                .sum::<u64>();
        let retained_blocks = target_sequences
            .iter()
            .chain(&proposer_sequences)
            .map(|sequence| {
                sequence
                    .state
                    .lock()
                    .map(|state| state.blocks.len() as u64)
                    .unwrap_or(0)
            })
            .sum::<u64>();
        let retained_growth = retained_blocks.saturating_sub(baseline_blocks);
        let provisional_blocks = draft_provisional_blocks
            .max(target_peak_blocks)
            .max(paired_peak_blocks);
        Ok(SpeculativeExecution {
            proposed: candidates
                .iter()
                .zip(&pages)
                .map(|(candidates, page)| candidates.len().min(page.len()))
                .sum::<usize>() as u64,
            pages,
            accepted,
            provisional_blocks,
            rolled_back_blocks: provisional_blocks.saturating_sub(retained_growth),
            draft_nanos,
            verification_nanos,
        })
    }
    .await;
    discard_speculative_states(&target_pool, &target_shadow);
    discard_speculative_states(&proposer_pool, &proposer_shadow);
    result
}

#[napi]
impl NativeInferenceArtifact {
    #[napi]
    pub fn open(&self) -> NativeInferenceSession {
        let id = NEXT_INFERENCE_SESSION_ID.fetch_add(1, Ordering::Relaxed);
        NativeInferenceSession {
            inner: Arc::new(InferenceSessionInner {
                id,
                programs: self.programs.clone(),
                diagnostics: self.diagnostics.clone(),
                ids: self.ids.clone(),
                operation: Arc::new(tokio::sync::Mutex::new(())),
                state: Mutex::new(InferenceSessionState {
                    closed: false,
                    sequences: HashMap::new(),
                    receipt: None,
                }),
            }),
        }
    }

    #[napi]
    pub fn diagnostics(&self) -> Result<NativeInferenceDiagnostics> {
        let diagnostics = self
            .diagnostics
            .lock()
            .map_err(|error| inference_error("inspect", error))?;
        Ok(NativeInferenceDiagnostics {
            rounds_started: u64_bigint(diagnostics.rounds_started),
            rounds_completed: u64_bigint(diagnostics.rounds_completed),
            rounds_recovered: u64_bigint(diagnostics.rounds_recovered),
            last_round_id: diagnostics.last_round_id.map(u64_bigint),
            last_failure_phase: diagnostics.last_failure_phase.clone(),
            ordinary_rounds: u64_bigint(diagnostics.ordinary_rounds),
            speculative_rounds: u64_bigint(diagnostics.speculative_rounds),
            proposed_tokens: u64_bigint(diagnostics.proposed_tokens),
            accepted_tokens: u64_bigint(diagnostics.accepted_tokens),
            emitted_tokens: u64_bigint(diagnostics.emitted_tokens),
            provisional_blocks: u64_bigint(diagnostics.provisional_blocks),
            rolled_back_blocks: u64_bigint(diagnostics.rolled_back_blocks),
            draft_nanos: u64_bigint(diagnostics.draft_nanos),
            verification_nanos: u64_bigint(diagnostics.verification_nanos),
            accepted_length_histogram: diagnostics
                .accepted_length_histogram
                .iter()
                .copied()
                .map(u64_bigint)
                .collect(),
            target_pool_high_water_blocks: u64_bigint(diagnostics.target_pool_high_water_blocks),
            proposer_pool_high_water_blocks: self
                .programs
                .proposer_pool
                .as_ref()
                .or(self.programs.replay_pool.as_ref())
                .map(|_| u64_bigint(diagnostics.proposer_pool_high_water_blocks)),
        })
    }
}

impl NativeInferenceSession {
    fn note_failure(&self, phase: &str) {
        self.inner
            .diagnostics
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .last_failure_phase = Some(phase.to_string());
    }

    fn recovered(&self, request: &ReceiptRequest) -> Result<Option<NativeInferenceRoundResult>> {
        let state = self
            .inner
            .state
            .lock()
            .map_err(|error| inference_error("publish", error))?;
        if state.closed {
            return Err(inference_error("publish", "session is closed"));
        }
        let Some(receipt) = &state.receipt else {
            return Ok(None);
        };
        if &receipt.request != request {
            drop(state);
            self.note_failure("publish");
            return Err(inference_error(
                "publish",
                "pending receipt belongs to a different operation or request",
            ));
        }
        self.inner
            .diagnostics
            .lock()
            .map_err(|error| inference_error("publish", error))?
            .rounds_recovered += 1;
        Ok(Some(receipt_result(self.inner.id, receipt, true)))
    }

    fn reserve_ids(&self, sequence_count: usize) -> Result<(u64, Vec<u64>)> {
        let mut ids = self
            .inner
            .ids
            .lock()
            .map_err(|error| inference_error("admission", error))?;
        reserve_inference_ids(&mut ids, sequence_count)
    }

    fn lifecycle_operation(&self, phase: &str) -> Result<tokio::sync::OwnedMutexGuard<()>> {
        try_lifecycle_operation(&self.inner.operation, phase)
    }

    fn publish_receipt(
        &self,
        state: &mut InferenceSessionState,
        round_id: u64,
        request: ReceiptRequest,
        pages: Vec<ReceiptPage>,
    ) -> NativeInferenceRoundResult {
        let receipt = Receipt {
            round_id,
            request,
            pages,
        };
        state.receipt = Some(receipt.clone());
        let mut diagnostics = self
            .inner
            .diagnostics
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        diagnostics.rounds_completed += 1;
        diagnostics.last_round_id = Some(round_id);
        receipt_result(self.inner.id, &receipt, false)
    }
}

#[napi]
impl NativeInferenceSession {
    #[napi]
    pub fn sequence(&self, sequence_id: NativeU64) -> Result<NativeInferenceSequence> {
        let state = self
            .inner
            .state
            .lock()
            .map_err(|error| inference_error("inspect", error))?;
        if state.closed || !state.sequences.contains_key(&sequence_id.0) {
            return Err(inference_error("inspect", "sequence is not live"));
        }
        Ok(NativeInferenceSequence {
            session_id: self.inner.id,
            sequence_id: sequence_id.0,
        })
    }

    #[napi]
    pub async fn add(
        &self,
        prompts: Vec<&NativeTensor>,
        sampling: Vec<NativeInferenceSamplingOptions>,
        max_tokens: Vec<u32>,
        eos_tokens: Vec<Vec<u32>>,
        cancellation_token: Option<&CancellationToken>,
    ) -> Result<NativeInferenceRoundResult> {
        let _operation = self.inner.operation.clone().lock_owned().await;
        let count = prompts.len();
        if count == 0
            || sampling.len() != count
            || max_tokens.len() != count
            || eos_tokens.len() != count
        {
            return Err(inference_error(
                "admission",
                "add arrays must be nonempty and have equal length",
            ));
        }
        let prompts = prompts
            .into_iter()
            .map(|prompt| prompt_tokens(prompt, self.inner.programs.token_dtype))
            .collect::<Result<Vec<_>>>()?;
        let sampling = sampling
            .iter()
            .map(|sampling| inference_sampling(sampling, "admission"))
            .collect::<Result<Vec<_>>>()?;
        let request = ReceiptRequest::Add(
            (0..count)
                .map(|index| AddRequestIdentity {
                    prompt: prompts[index].clone(),
                    sampling: sampling[index].into(),
                    max_tokens: max_tokens[index],
                    eos_tokens: eos_tokens[index].clone(),
                })
                .collect(),
        );
        if let Some(receipt) = self.recovered(&request)? {
            return Ok(receipt);
        }
        let slots = {
            let state = self
                .inner
                .state
                .lock()
                .map_err(|error| inference_error("admission", error))?;
            let slots = (0..self.inner.programs.batch_size)
                .filter(|slot| {
                    state
                        .sequences
                        .values()
                        .all(|sequence| sequence.slot != *slot)
                })
                .take(count)
                .collect::<Vec<_>>();
            if slots.len() != count {
                return Err(inference_error(
                    "admission",
                    "session has insufficient free slots",
                ));
            }
            slots
        };
        let (round_id, ids) = self.reserve_ids(count)?;
        let mut target = (0..count)
            .map(|_| NativeKvSequence::new(self.inner.programs.target_pool.clone()))
            .collect::<Vec<_>>();
        let proposer_state_pool = self.inner.programs.proposer_pool.as_ref().or(self
            .inner
            .programs
            .replay_pool
            .as_ref());
        let mut proposer = proposer_state_pool.map(|pool| {
            (0..count)
                .map(|_| NativeKvSequence::new(pool.clone()))
                .collect::<Vec<_>>()
        });
        for (sequence, prompt) in target.iter().zip(&prompts) {
            if self.inner.programs.replay_pool.is_some() {
                continue;
            }
            if let Err(error) = sequence.prefill_match(prompt.clone()) {
                for sequence in &target {
                    sequence.return_blocks();
                }
                if let Some(proposer) = &proposer {
                    for sequence in proposer {
                        sequence.return_blocks();
                    }
                }
                self.note_failure("prefill");
                return Err(inference_error("prefill", error.reason));
            }
        }
        if self.inner.programs.proposer_prefill.is_some() {
            let proposer = proposer.as_ref().expect("exact proposer has state");
            for (sequence, prompt) in proposer.iter().zip(&prompts) {
                if let Err(error) = sequence.prefill_match(prompt.clone()) {
                    for sequence in &target {
                        sequence.return_blocks();
                    }
                    for sequence in proposer {
                        sequence.return_blocks();
                    }
                    self.note_failure("proposer");
                    return Err(inference_error("proposer", error.reason));
                }
            }
        }
        self.inner
            .diagnostics
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .rounds_started += 1;
        let slot_u32 = slots.iter().map(|slot| *slot as u32).collect::<Vec<_>>();
        let keyed = sampling
            .iter()
            .copied()
            .zip(ids.iter().copied())
            .collect::<Vec<_>>();
        let target_prefix_metadata = Arc::new(Mutex::new(DeferredPrefixMetadata::default()));
        let proposer_prefix_metadata = proposer
            .as_ref()
            .map(|_| Arc::new(Mutex::new(DeferredPrefixMetadata::default())));
        let sampled = match prefill_sequences(
            &self.inner.programs.target_prefill,
            &target,
            &slot_u32,
            &prompts,
            self.inner.programs.token_dtype,
            Some(&keyed),
            cancellation_token,
            target_prefix_metadata.clone(),
            self.inner
                .programs
                .replay_prefill
                .as_ref()
                .map(|executable| {
                    let plan = self
                        .inner
                        .programs
                        .proposer_plan
                        .as_ref()
                        .expect("replay bundle has proposer plan");
                    PrefillReplay {
                        executable,
                        sequences: proposer.as_ref().expect("replay bundle has proposer state"),
                        taps: &plan.prefill_taps,
                        prefix_metadata: proposer_prefix_metadata
                            .as_ref()
                            .expect("replay admission has proposer metadata")
                            .clone(),
                    }
                }),
        )
        .await
        {
            Ok(sampled) => sampled,
            Err(error) => {
                if let Some(token) = cancellation_token {
                    token.state.complete();
                }
                self.note_failure("prefill");
                for sequence in &target {
                    sequence.return_blocks();
                }
                if let Some(proposer) = &proposer {
                    for sequence in proposer {
                        sequence.return_blocks();
                    }
                }
                return Err(error);
            }
        };
        if let (Some(prefill), Some(proposer)) = (&self.inner.programs.proposer_prefill, &proposer)
        {
            if let Err(error) = prefill_sequences(
                prefill,
                proposer,
                &slot_u32,
                &prompts,
                self.inner.programs.token_dtype,
                None,
                cancellation_token,
                proposer_prefix_metadata
                    .as_ref()
                    .expect("paired admission has proposer prefix metadata")
                    .clone(),
                None,
            )
            .await
            {
                if let Some(token) = cancellation_token {
                    token.state.complete();
                }
                self.note_failure("proposer");
                for sequence in &target {
                    sequence.return_blocks();
                }
                for sequence in proposer {
                    sequence.return_blocks();
                }
                return Err(inference_error("proposer", error.reason));
            }
        }
        let mut pages = Vec::with_capacity(count);
        let mut sequence_states = Vec::with_capacity(count);
        for index in 0..count {
            let token = sampled[index]
                .ok_or_else(|| inference_error("sample", "prefill did not sample a token"))?;
            let budget = (max_tokens[index] != 0).then_some(max_tokens[index] as u64);
            let terminal = if eos_tokens[index].contains(&token) {
                Some("eos".to_string())
            } else if budget == Some(1) {
                Some("maxTokens".to_string())
            } else {
                None
            };
            sequence_states.push(InferenceSequenceState {
                id: ids[index],
                slot: slots[index],
                target: target[index].clone(),
                proposer: proposer.as_ref().map(|values| values[index].clone()),
                pending: token,
                history: prompts[index]
                    .iter()
                    .copied()
                    .chain(std::iter::once(token))
                    .collect(),
                generated: 1,
                budget,
                eos: eos_tokens[index].clone(),
                terminal: terminal.clone(),
                // Admission overrides apply only to the first page. Later
                // rounds resolve sparse overrides from artifact defaults.
                sampling: self.inner.programs.default_sampling,
            });
            pages.push(ReceiptPage {
                sequence_id: ids[index],
                tokens: vec![token],
                stop_reason: terminal,
            });
        }
        if cancellation_token.is_some_and(|token| !token.state.complete()) {
            self.note_failure("publish");
            for sequence in &target {
                sequence.return_blocks();
            }
            if let Some(proposer) = &proposer {
                for sequence in proposer {
                    sequence.return_blocks();
                }
            }
            return Err(inference_error("publish", "operation aborted"));
        }
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let proposer_publication = self
            .inner
            .programs
            .proposer_pool
            .as_ref()
            .or(self.inner.programs.replay_pool.as_ref())
            .zip(proposer_prefix_metadata.as_ref());
        let result = publish_deferred_prefixes(
            &self.inner.programs.target_pool,
            &target_prefix_metadata,
            proposer_publication,
            || {
                for (index, mut sequence) in sequence_states.into_iter().enumerate() {
                    sequence.target.finalize_releases = true;
                    if let Some(proposer) = &mut sequence.proposer {
                        proposer.finalize_releases = true;
                    }
                    state.sequences.insert(ids[index], sequence);
                }
                for sequence in &mut target {
                    sequence.finalize_releases = false;
                }
                if let Some(proposer) = &mut proposer {
                    for sequence in proposer {
                        sequence.finalize_releases = false;
                    }
                }
                self.publish_receipt(&mut state, round_id, request, pages)
            },
        )?;
        {
            let mut diagnostics = self
                .inner
                .diagnostics
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            diagnostics.emitted_tokens += count as u64;
            let target_capacity = self.inner.programs.target_pool.max_tokens
                / self.inner.programs.target_pool.block_size;
            diagnostics.target_pool_high_water_blocks = diagnostics
                .target_pool_high_water_blocks
                .max((target_capacity - self.inner.programs.target_pool.available()) as u64);
            if let Some(pool) = self.inner.programs.proposer_pool.as_ref().or(self
                .inner
                .programs
                .replay_pool
                .as_ref())
            {
                let capacity = pool.max_tokens / pool.block_size;
                diagnostics.proposer_pool_high_water_blocks = diagnostics
                    .proposer_pool_high_water_blocks
                    .max((capacity - pool.available()) as u64);
            }
        }
        Ok(result)
    }

    #[napi]
    pub async fn run_round(
        &self,
        sequences: Vec<&NativeInferenceSequence>,
        sampling: Vec<NativeInferenceSamplingOverrides>,
        cancellation_token: Option<&CancellationToken>,
    ) -> Result<NativeInferenceRoundResult> {
        let _operation = self.inner.operation.clone().lock_owned().await;
        if sequences.is_empty() || sequences.len() != sampling.len() {
            return Err(inference_error(
                "admission",
                "runRound arrays must be nonempty and have equal length",
            ));
        }
        let ids = inference_sequence_ids(self.inner.id, &sequences)?;
        let selected = {
            let state = self
                .inner
                .state
                .lock()
                .map_err(|error| inference_error("admission", error))?;
            ids.iter()
                .map(|id| {
                    let sequence = state
                        .sequences
                        .get(id)
                        .ok_or_else(|| inference_error("admission", "sequence is not live"))?;
                    Ok(sequence.clone())
                })
                .collect::<Result<Vec<_>>>()?
        };
        let request_sampling = sampling
            .iter()
            .map(SamplingOverrideIdentity::from)
            .collect::<Vec<_>>();
        let sampling = sampling
            .iter()
            .zip(&selected)
            .map(|(sampling, sequence)| {
                inference_sampling_override(sampling, sequence.sampling, "sample")
            })
            .collect::<Result<Vec<_>>>()?;
        let request = ReceiptRequest::Round(
            ids.iter()
                .copied()
                .zip(request_sampling)
                .map(|(sequence_id, sampling)| RoundRequestIdentity {
                    sequence_id,
                    sampling,
                })
                .collect(),
        );
        if let Some(receipt) = self.recovered(&request)? {
            return Ok(receipt);
        }
        if selected.iter().any(|sequence| sequence.terminal.is_some()) {
            return Err(inference_error("admission", "sequence is terminal"));
        }
        let (round_id, reserved_sequences) = self.reserve_ids(0)?;
        debug_assert!(reserved_sequences.is_empty());
        self.inner
            .diagnostics
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .rounds_started += 1;
        let slots = selected
            .iter()
            .map(|sequence| sequence.slot as u32)
            .collect::<Vec<_>>();
        let target = selected
            .iter()
            .map(|sequence| &sequence.target)
            .collect::<Vec<_>>();
        let (token_pages, speculative_metrics) = if self.inner.programs.replay_verify.is_some() {
            let proposer = selected
                .iter()
                .map(|sequence| {
                    sequence
                        .proposer
                        .as_ref()
                        .expect("parallel artifact has replay sequence state")
                })
                .collect::<Vec<_>>();
            let page_limits = selected
                .iter()
                .map(|sequence| {
                    sequence
                        .budget
                        .map_or(u64::MAX, |budget| budget.saturating_sub(sequence.generated))
                        .min(self.inner.programs.max_draft_tokens as u64 + 1)
                        .max(1) as u32
                })
                .collect();
            let execution = execute_parallel_block_detailed(
                &self.inner.programs,
                target,
                proposer,
                slots.clone(),
                selected.iter().map(|sequence| sequence.pending).collect(),
                sampling.clone(),
                selected.iter().map(|sequence| sequence.id).collect(),
                selected.iter().map(|sequence| sequence.generated).collect(),
                page_limits,
                selected
                    .iter()
                    .map(|sequence| sequence.eos.clone())
                    .collect(),
                cancellation_token,
            )
            .await
            .map_err(|error| {
                self.note_failure("verify");
                error
            })?;
            let metrics = (
                execution.proposed,
                execution.accepted,
                execution.provisional_blocks,
                execution.rolled_back_blocks,
                execution.draft_nanos,
                execution.verification_nanos,
            );
            (execution.pages, Some(metrics))
        } else if let (Some(verify), Some(config)) = (
            &self.inner.programs.target_verify,
            self.inner.programs.history_lookup,
        ) {
            let page_limits = selected
                .iter()
                .map(|sequence| {
                    let remaining = sequence
                        .budget
                        .map_or(u64::MAX, |budget| budget.saturating_sub(sequence.generated));
                    remaining
                        .min(self.inner.programs.max_draft_tokens as u64 + 1)
                        .max(1) as u32
                })
                .collect::<Vec<_>>();
            let execution = verify
                .execute_history_lookup_detailed(
                    target,
                    slots.clone(),
                    selected.iter().map(|sequence| sequence.pending).collect(),
                    selected
                        .iter()
                        .map(|sequence| sequence.history.clone())
                        .collect(),
                    sampling.clone(),
                    selected.iter().map(|sequence| sequence.id).collect(),
                    selected.iter().map(|sequence| sequence.generated).collect(),
                    config,
                    self.inner.programs.max_draft_tokens,
                    page_limits,
                    selected
                        .iter()
                        .map(|sequence| sequence.eos.clone())
                        .collect(),
                    cancellation_token,
                )
                .await
                .map_err(|error| {
                    self.note_failure("verify");
                    inference_error("verify", error.reason)
                })?;
            let metrics = (
                execution.proposed,
                execution.accepted,
                execution.provisional_blocks,
                execution.rolled_back_blocks,
                execution.draft_nanos,
                execution.verification_nanos,
            );
            (execution.pages, Some(metrics))
        } else if let (Some(verify), Some(proposer_decode)) = (
            &self.inner.programs.target_verify,
            &self.inner.programs.proposer_decode,
        ) {
            let proposer = selected
                .iter()
                .map(|sequence| {
                    sequence
                        .proposer
                        .as_ref()
                        .expect("speculative artifact has paired sequence state")
                })
                .collect::<Vec<_>>();
            let native_sampling = sampling
                .iter()
                .map(|options| NativeInferenceSamplingOptions {
                    temperature: options.temperature,
                    top_k: options.top_k.unwrap_or(0) as u32,
                    top_p: options.top_p,
                    seed: u64_bigint(options.seed),
                })
                .collect::<Vec<_>>();
            let page_limits = selected
                .iter()
                .map(|sequence| {
                    let remaining = sequence
                        .budget
                        .map_or(u64::MAX, |budget| budget.saturating_sub(sequence.generated));
                    remaining
                        .min(self.inner.programs.max_draft_tokens as u64 + 1)
                        .max(1) as u32
                })
                .collect::<Vec<_>>();
            let execution = verify
                .execute_speculative_detailed(
                    proposer_decode,
                    target,
                    proposer,
                    slots.clone(),
                    selected.iter().map(|sequence| sequence.pending).collect(),
                    native_sampling,
                    selected
                        .iter()
                        .map(|sequence| u64_bigint(sequence.id))
                        .collect(),
                    selected
                        .iter()
                        .map(|sequence| u64_bigint(sequence.generated))
                        .collect(),
                    self.inner.programs.max_draft_tokens as u32,
                    page_limits,
                    selected
                        .iter()
                        .map(|sequence| sequence.eos.clone())
                        .collect(),
                    cancellation_token,
                )
                .await
                .map_err(|error| {
                    self.note_failure("verify");
                    inference_error("verify", error.reason)
                })?;
            let metrics = (
                execution.proposed,
                execution.accepted,
                execution.provisional_blocks,
                execution.rolled_back_blocks,
                execution.draft_nanos,
                execution.verification_nanos,
            );
            (execution.pages, Some(metrics))
        } else {
            let options = sampling
                .iter()
                .zip(&selected)
                .map(|(options, sequence)| SamplingOptions {
                    seed: coordinate_seed(
                        options.seed,
                        sequence.id,
                        sequence.generated,
                        SamplingPurpose::Target,
                        0,
                    ),
                    counter: 0,
                    ..*options
                })
                .collect::<Vec<_>>();
            let input = NativeTensor::wrap(token_value(
                self.inner.programs.token_dtype,
                {
                    let mut input = vec![0; self.inner.programs.batch_size];
                    for sequence in &selected {
                        input[sequence.slot] = sequence.pending;
                    }
                    input
                },
                vec![self.inner.programs.batch_size, 1],
            ));
            let output = self
                .inner
                .programs
                .target_decode
                .execute_stateful(
                    vec![&input],
                    target,
                    slots.clone(),
                    vec![1; selected.len()],
                    selected
                        .iter()
                        .map(|sequence| vec![sequence.pending])
                        .collect(),
                    StatefulInvocation::Sampled(options),
                    cancellation_token,
                )
                .await
                .map_err(|error| {
                    self.note_failure("sample");
                    inference_error("sample", error.reason)
                })?;
            let StatefulExecutionOutput::Samples(tokens) = output else {
                unreachable!("sampled decode returned tensors")
            };
            (tokens.into_iter().map(|token| vec![token]).collect(), None)
        };
        {
            let mut diagnostics = self
                .inner
                .diagnostics
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let emitted = token_pages.iter().map(Vec::len).sum::<usize>() as u64;
            diagnostics.emitted_tokens += emitted;
            if let Some((proposed, accepted, provisional, rolled_back, draft, verification)) =
                &speculative_metrics
            {
                diagnostics.speculative_rounds += 1;
                diagnostics.proposed_tokens += proposed;
                diagnostics.accepted_tokens += accepted.iter().sum::<usize>() as u64;
                diagnostics.provisional_blocks += provisional;
                diagnostics.rolled_back_blocks += rolled_back;
                diagnostics.draft_nanos = diagnostics.draft_nanos.saturating_add(*draft);
                diagnostics.verification_nanos =
                    diagnostics.verification_nanos.saturating_add(*verification);
                for &length in accepted {
                    if let Some(bucket) = diagnostics.accepted_length_histogram.get_mut(length) {
                        *bucket += 1;
                    }
                }
            } else {
                diagnostics.ordinary_rounds += 1;
            }
            let target_capacity = self.inner.programs.target_pool.max_tokens
                / self.inner.programs.target_pool.block_size;
            diagnostics.target_pool_high_water_blocks = diagnostics
                .target_pool_high_water_blocks
                .max((target_capacity - self.inner.programs.target_pool.available()) as u64);
            if let Some(pool) = self.inner.programs.proposer_pool.as_ref().or(self
                .inner
                .programs
                .replay_pool
                .as_ref())
            {
                let capacity = pool.max_tokens / pool.block_size;
                diagnostics.proposer_pool_high_water_blocks = diagnostics
                    .proposer_pool_high_water_blocks
                    .max((capacity - pool.available()) as u64);
            }
        }
        let mut pages = Vec::with_capacity(selected.len());
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        for (index, tokens) in token_pages.into_iter().enumerate() {
            let sequence = state
                .sequences
                .get_mut(&ids[index])
                .expect("selected sequence remained live");
            let token = *tokens.last().expect("native round pages are nonempty");
            sequence.pending = token;
            sequence.history.extend(tokens.iter().copied());
            sequence.generated += tokens.len() as u64;
            let terminal = if sequence.eos.contains(&token) {
                Some("eos".to_string())
            } else if sequence
                .budget
                .is_some_and(|budget| sequence.generated >= budget)
            {
                Some("maxTokens".to_string())
            } else {
                None
            };
            sequence.terminal = terminal.clone();
            pages.push(ReceiptPage {
                sequence_id: sequence.id,
                tokens,
                stop_reason: terminal,
            });
        }
        Ok(self.publish_receipt(&mut state, round_id, request, pages))
    }

    #[napi]
    pub fn acknowledge(&self, round_id: NativeU64) -> Result<()> {
        let round_id = bigint_u64(&round_id, "roundId")?;
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|error| inference_error("publish", error))?;
        if state.receipt.as_ref().map(|receipt| receipt.round_id) != Some(round_id) {
            return Err(inference_error("publish", "round receipt is not pending"));
        }
        state.receipt = None;
        Ok(())
    }

    #[napi]
    pub fn finish(&self, sequences: Vec<&NativeInferenceSequence>) -> Result<()> {
        let _operation = self.lifecycle_operation("finish")?;
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|error| inference_error("finish", error))?;
        validate_finish_lifecycle(&state)?;
        let mut ids = HashSet::new();
        for sequence in &sequences {
            if sequence.session_id != self.inner.id
                || !ids.insert(sequence.sequence_id)
                || !state.sequences.contains_key(&sequence.sequence_id)
            {
                return Err(inference_error(
                    "finish",
                    "foreign, duplicate, or dead sequence",
                ));
            }
        }
        for id in ids {
            if let Some(sequence) = state.sequences.remove(&id) {
                sequence.target.return_blocks();
                if let Some(proposer) = sequence.proposer {
                    proposer.return_blocks();
                }
            }
        }
        Ok(())
    }

    #[napi]
    pub fn inspect(&self, sequence: &NativeInferenceSequence) -> Result<NativeInferenceInspection> {
        let state = self
            .inner
            .state
            .lock()
            .map_err(|error| inference_error("inspect", error))?;
        if state.closed || sequence.session_id != self.inner.id {
            return Err(inference_error(
                "inspect",
                "foreign sequence or closed session",
            ));
        }
        let sequence = state
            .sequences
            .get(&sequence.sequence_id)
            .ok_or_else(|| inference_error("inspect", "sequence is not live"))?;
        let cursor = sequence
            .target
            .state
            .lock()
            .map_err(|error| inference_error("inspect", error))?
            .cursor as u64;
        Ok(NativeInferenceInspection {
            sequence_id: u64_bigint(sequence.id),
            cursor: u64_bigint(cursor),
            terminal: sequence.terminal.clone(),
        })
    }

    #[napi]
    pub fn close(&self) -> Result<()> {
        {
            let state = self
                .inner
                .state
                .lock()
                .map_err(|error| inference_error("close", error))?;
            if state.closed {
                return Ok(());
            }
        }
        let _operation = self.lifecycle_operation("close")?;
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|error| inference_error("close", error))?;
        if state.closed {
            return Ok(());
        }
        for (_, sequence) in state.sequences.drain() {
            sequence.target.return_blocks();
            if let Some(proposer) = sequence.proposer {
                proposer.return_blocks();
            }
        }
        state.receipt = None;
        state.closed = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(tensor: Tensor) -> Arc<Node> {
        Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(Value(tensor))))).unwrap()
    }

    fn sampling_options() -> SamplingOptions {
        SamplingOptions {
            temperature: 0.0,
            top_k: None,
            top_p: 1.0,
            seed: 7,
            counter: 3,
        }
    }

    fn stateless_input_executable(shape: &[usize]) -> Executable {
        let root = LazyTensor {
            node: Node::new(NodeKind::Input {
                slot: 0,
                shape: shape.to_vec(),
                dtype: DType::F32,
                device: cpu_device(),
            })
            .unwrap(),
        };
        compile(vec![&root], None, None, None).unwrap()
    }

    fn value_ref(kind: &str) -> NativeInferenceValueRef {
        NativeInferenceValueRef {
            kind: kind.to_string(),
            target_output: None,
            stage: None,
            output: None,
            value: None,
            select_target_row: None,
        }
    }

    #[test]
    fn history_lookup_prefers_longest_suffix_and_caps_draft() {
        assert_eq!(history_lookup(&[1, 2, 3, 8, 1, 2, 3], 1, 4, 2), [8, 1]);
        assert!(history_lookup(&[1, 2, 3], 2, 3, 4).is_empty());
        assert!(history_lookup(&[1, 2, 1, 2], 1, 3, 0).is_empty());
    }

    #[test]
    fn history_lookup_uses_most_recent_equal_width_match() {
        assert_eq!(
            history_lookup(&[1, 2, 9, 1, 2, 8, 1, 2], 2, 2, 3),
            [8, 1, 2]
        );
    }

    #[test]
    fn split_logits_translate_semantic_tap_roots_after_lane_outputs() {
        assert_eq!(dense_tap_output(3, 1), 3);
        assert_eq!(dense_tap_output(3, 2), 4);
        assert_eq!(dense_tap_output(1, 4), 4);
    }

    #[test]
    fn target_hidden_row_selection_is_a_direct_value_view() {
        let source = Value(Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
        ));
        let selected = routed_target_row(&source, 1).unwrap();

        assert_eq!(selected.shape(), &[1, 2]);
        assert_eq!(selected.tensor().layout.offset(), 2);
        assert_eq!(selected.to_f32_vec().unwrap(), [3.0, 4.0]);
        assert_eq!(source.to_f32_vec().unwrap(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn stateless_dag_routes_values_between_executables_without_wrappers() {
        let executable = stateless_input_executable(&[2]);
        let mut hidden_route = value_ref("TargetHidden");
        hidden_route.target_output = Some(1);
        let mut stage_route = value_ref("StageOutput");
        stage_route.stage = Some(0);
        stage_route.output = Some(0);
        let plan = ValidatedProposerPlan {
            schema: None,
            target_hidden: HashMap::from([(
                1,
                ValueMetadata {
                    dtype: DType::F32,
                    shape: vec![2],
                },
            )]),
            shared: HashMap::new(),
            stages: vec![
                ValidatedStage {
                    executable: executable.clone(),
                    inputs: vec![(0, hidden_route)],
                },
                ValidatedStage {
                    executable,
                    inputs: vec![(0, stage_route)],
                },
            ],
            prefill_taps: Vec::new(),
            verify_taps: Vec::new(),
        };
        let target = vec![
            Value(Tensor::from_vec(vec![0.0f32], vec![1])),
            Value(Tensor::from_vec(vec![7.0f32, 11.0], vec![2])),
        ];

        let outputs =
            execute_stateless_stage_dag(&plan, &target, 0, &CancellationFlag::new()).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0][0].to_f32_vec().unwrap(), [7.0, 11.0]);
        assert_eq!(outputs[1][0].to_f32_vec().unwrap(), [7.0, 11.0]);
    }

    fn assert_strided_float_sampling<T: Elem>() {
        let tensor = Tensor::from_vec(
            [99.0, 1.0, 99.0, 5.0, 99.0, 3.0]
                .into_iter()
                .map(T::from_f64)
                .collect(),
            vec![6],
        )
        .view(Layout::new(vec![3], vec![2], 1));
        assert_eq!(
            sample_blocking(&Value(tensor), sampling_options(), || false).unwrap(),
            1
        );
    }

    #[test]
    fn sampling_borrows_strided_float_logits() {
        assert_strided_float_sampling::<half::f16>();
        assert_strided_float_sampling::<half::bf16>();
        assert_strided_float_sampling::<f32>();
        assert_strided_float_sampling::<f64>();
    }

    #[test]
    fn sampling_rejects_invalid_tensor_inputs_and_reports_cancellation() {
        let integer = Value(Tensor::from_vec(vec![1u32, 2, 3], vec![3]));
        let error = sample_blocking(&integer, sampling_options(), || false).unwrap_err();
        assert_eq!(error.status, Status::InvalidArg);
        assert!(error.reason.contains("floating-point dtype"));

        let matrix = Value(Tensor::from_vec(vec![1.0f32, 2.0], vec![1, 2]));
        let error = sample_blocking(&matrix, sampling_options(), || false).unwrap_err();
        assert_eq!(error.status, Status::InvalidArg);
        assert!(error.reason.contains("rank 1"));

        let empty = Value(Tensor::from_vec(Vec::<f32>::new(), vec![0]));
        let error = sample_blocking(&empty, sampling_options(), || false).unwrap_err();
        assert_eq!(error.status, Status::InvalidArg);
        assert!(error.reason.contains("non-empty"));

        let logits = Value(Tensor::from_vec(vec![1.0f32, 2.0], vec![2]));
        let error = sample_blocking(&logits, sampling_options(), || true).unwrap_err();
        assert_eq!(error.status, Status::Cancelled);
        assert_eq!(error.reason, "operation aborted");
    }

    #[test]
    fn executable_diagnostics_exposes_compile_phases() {
        let root = leaf(Tensor::from_vec(vec![1.0f32], vec![1]));
        let compilation = executable::compile(
            std::slice::from_ref(&root),
            CompileOptions::default(),
            CHUNKED_CE_CHUNK_LOGITS,
        )
        .unwrap();
        let diagnostics = executable_diagnostics(&compilation.executable.diagnostics);

        assert_eq!(
            diagnostics
                .compile_phases
                .iter()
                .map(|timing| timing.phase.as_str())
                .collect::<Vec<_>>(),
            [
                "graph_index",
                "optimization",
                "lowering",
                "lowered_program_validation",
                "memory_planning",
                "physical_planning",
                "artifact_assembly",
                "publication",
            ]
        );
        assert!(diagnostics
            .compile_phases
            .iter()
            .all(|timing| timing.nanoseconds.is_finite() && timing.nanoseconds >= 0.0));
    }

    fn eval_f32(node: &Arc<Node>) -> Vec<f32> {
        let cancelled = CancellationFlag::new();
        let compilation = executable::compile(
            std::slice::from_ref(node),
            CompileOptions::default(),
            CHUNKED_CE_CHUNK_LOGITS,
        )
        .unwrap();
        executable::execute(
            &compilation.executable,
            &[],
            &compilation.generated_bindings,
            &cancelled,
            None,
        )
        .unwrap()
        .remove(0)
        .to_f32_vec()
        .unwrap()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, name: &str) {
        assert_eq!(actual.len(), expected.len(), "{name}: length");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{name}[{index}]: {actual} vs {expected}"
            );
        }
    }

    fn test_state_schema() -> KvStateSchema {
        KvStateSchema {
            max_tokens: 8,
            block_size: 4,
            kv_dtype: DType::F32,
            window: Some(8),
            allows_window_eviction: true,
            batch: 2,
            packed_rows_per_sequence: None,
            layers: 1,
            kv_heads: 2,
            head_dim: 4,
            kda: KdaGeometry::default(),
            conv: ConvGeometry::default(),
            cursor_slot: 1,
            cursor_tensor: true,
        }
    }

    fn test_schema_pool() -> PoolInner {
        PoolInner {
            k: vec![pool::Slab::new(8, 8, DType::F32)],
            v: vec![pool::Slab::new(8, 8, DType::F32)],
            scales: Vec::new(),
            kv_dtype: DType::F32,
            kv_heads: 2,
            head_dim: 4,
            block_size: 4,
            max_tokens: 8,
            kda: KdaGeometry::default(),
            conv: ConvGeometry::default(),
            blocks: Mutex::new(BlockStore::new(2)),
        }
    }

    #[test]
    fn structural_cache_shares_the_plan_and_rebinds_leaves() {
        let before = cpu_preparation_counts();
        let graph = |left: Vec<f32>, right: Vec<f32>| LazyTensor {
            node: Node::new(NodeKind::Add {
                a: leaf(Tensor::from_vec(left, vec![2])),
                b: leaf(Tensor::from_vec(right, vec![2])),
            })
            .unwrap(),
        };
        let first_root = graph(vec![1.0, 2.0], vec![3.0, 4.0]);
        let second_root = graph(vec![10.0, 20.0], vec![30.0, 40.0]);
        let key = Some("cpu-structural-cache-rebind-test".to_string());
        let first = compile(vec![&first_root], None, None, key.clone()).unwrap();
        let after_miss = cpu_preparation_counts();
        assert_eq!(after_miss.0 - before.0, 1, "cache miss index builds");
        assert_eq!(after_miss.1 - before.1, 1, "cache miss leaf collections");
        assert_eq!(after_miss.2 - before.2, 2, "cache miss leaf gets");
        let second = compile(vec![&second_root], None, None, key).unwrap();
        let after_hit = cpu_preparation_counts();
        assert_eq!(after_hit.0 - after_miss.0, 1, "cache hit index builds");
        assert_eq!(after_hit.1 - after_miss.1, 1, "cache hit leaf collections");
        assert_eq!(after_hit.2 - after_miss.2, 2, "cache hit leaf gets");
        assert!(Arc::ptr_eq(
            &first.inner.executable,
            &second.inner.executable
        ));
        assert_eq!(
            first.inner.executable.signature,
            second.inner.executable.signature
        );
        assert_eq!(first.inner.executable.signature.outputs.len(), 1);
        assert_eq!(
            first.inner.executable.memory,
            second.inner.executable.memory
        );
        assert_eq!(
            first.inner.executable.diagnostics,
            second.inner.executable.diagnostics
        );
        let mut first_outputs = Vec::new();
        let mut second_outputs = Vec::new();
        for _ in 0..8 {
            first_outputs.push(
                executable::execute(
                    &first.inner.executable,
                    &[],
                    &first.inner.generated_bindings,
                    &CancellationFlag::new(),
                    None,
                )
                .unwrap()
                .remove(0),
            );
            second_outputs.push(
                executable::execute(
                    &second.inner.executable,
                    &[],
                    &second.inner.generated_bindings,
                    &CancellationFlag::new(),
                    None,
                )
                .unwrap()
                .remove(0),
            );
        }
        for output in first_outputs {
            assert_eq!(output.to_f32_vec().unwrap(), [4.0, 6.0]);
        }
        for output in second_outputs {
            assert_eq!(output.to_f32_vec().unwrap(), [40.0, 60.0]);
        }
    }

    #[test]
    fn constant_weights_bypass_structural_cache() {
        let graph = |values: Vec<f32>| LazyTensor {
            node: leaf(Tensor::from_vec(values, vec![2])),
        };
        let first_root = graph(vec![1.0, 2.0]);
        let second_root = graph(vec![3.0, 4.0]);
        let constant_options = || NativeCompileOptions {
            optimize: None,
            constant_weights: Some(true),
        };
        let key = Some("cpu-constant-weight-cache-suppression-test".to_string());
        let first = compile(
            vec![&first_root],
            Some(constant_options()),
            None,
            key.clone(),
        )
        .unwrap();
        let second = compile(vec![&second_root], Some(constant_options()), None, key).unwrap();

        assert!(!Arc::ptr_eq(
            &first.inner.executable,
            &second.inner.executable
        ));
        assert!(first.inner.generated_bindings.is_empty());
        assert!(second.inner.generated_bindings.is_empty());
        assert_eq!(first.inner.executable.memory.report.persistent_bytes, 8);
        assert_eq!(second.inner.executable.memory.report.persistent_bytes, 8);
        assert_eq!(
            executable::execute(
                &first.inner.executable,
                &[],
                &[],
                &CancellationFlag::new(),
                None,
            )
            .unwrap()[0]
                .to_f32_vec()
                .unwrap(),
            [1.0, 2.0]
        );
        assert_eq!(
            executable::execute(
                &second.inner.executable,
                &[],
                &[],
                &CancellationFlag::new(),
                None,
            )
            .unwrap()[0]
                .to_f32_vec()
                .unwrap(),
            [3.0, 4.0]
        );
    }

    #[test]
    fn recurrent_state_slices_share_transaction_storage_without_backend_allocation() {
        let kda = Value(Tensor::from_vec(
            (0..24).map(|value| value as f32).collect(),
            vec![4, 2, 3],
        ));
        let conv = Value(Tensor::from_vec(
            (0..12).map(|value| value as f32).collect(),
            vec![2, 2, 3],
        ));

        let (kda_slices, conv_slices) = {
            let _guard = crate::ExecutableAllocationGuard::enter();
            (
                recurrent_state_slices(&kda, 2, 2, false),
                recurrent_state_slices(&conv, 2, 1, true),
            )
        };

        assert_eq!(kda_slices[0].shape(), &[2, 2, 3]);
        assert_eq!(kda_slices[1].shape(), &[2, 2, 3]);
        assert_eq!(
            Value(kda_slices[1].clone()).to_f32_vec().unwrap(),
            (12..24).map(|value| value as f32).collect::<Vec<_>>()
        );
        assert_eq!(conv_slices[0].shape(), &[2, 3]);
        assert_eq!(conv_slices[1].shape(), &[2, 3]);
        assert_eq!(
            Value(conv_slices[1].clone()).to_f32_vec().unwrap(),
            (6..12).map(|value| value as f32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn invocation_mode_rejects_stateless_and_stateful_mismatches() {
        assert!(validate_execution_mode(false, 1, None, None, None, None, None, None).is_ok());
        assert!(validate_execution_mode(
            false,
            0,
            Some(1),
            Some(&[0]),
            Some(&[true]),
            Some(&[1]),
            Some(&[1]),
            Some(1)
        )
        .is_err());
        assert!(validate_execution_mode(
            true,
            1,
            Some(1),
            Some(&[0]),
            Some(&[true]),
            Some(&[1]),
            Some(&[1]),
            Some(1)
        )
        .is_err());
        assert!(validate_execution_mode(true, 0, None, None, None, None, None, None).is_err());
        assert!(validate_execution_mode(
            true,
            0,
            Some(1),
            Some(&[0]),
            Some(&[true]),
            Some(&[1]),
            Some(&[1]),
            Some(1)
        )
        .is_ok());
        let schema = test_state_schema();
        assert!(validate_fixed_lanes(
            &schema,
            1,
            &[1],
            &[false, true],
            &[0, 2],
            &[0, 2],
            &[vec![4, 5]]
        )
        .is_ok());
        assert!(validate_fixed_lanes(
            &schema,
            1,
            &[1],
            &[true, false],
            &[0, 2],
            &[0, 2],
            &[vec![4, 5]]
        )
        .is_err());
        assert!(validate_fixed_lanes(
            &schema,
            1,
            &[1],
            &[false, true],
            &[0, 2],
            &[0, 1],
            &[vec![4]]
        )
        .is_err());
    }

    #[test]
    fn execute_rejects_every_schema_pool_mismatch() {
        let pool = test_schema_pool();
        let schema = test_state_schema();
        validate_pool_schema(&schema, &pool).unwrap();

        let mismatches = [
            KvStateSchema {
                max_tokens: 4,
                ..schema
            },
            KvStateSchema {
                block_size: 2,
                ..schema
            },
            KvStateSchema {
                kv_dtype: DType::F16,
                ..schema
            },
            KvStateSchema {
                layers: 2,
                ..schema
            },
            KvStateSchema {
                kv_heads: 1,
                ..schema
            },
            KvStateSchema {
                head_dim: 2,
                ..schema
            },
            KvStateSchema {
                window: Some(9),
                ..schema
            },
        ];
        for mismatch in mismatches {
            assert!(validate_pool_schema(&mismatch, &pool).is_err());
        }

        let mut state = SeqState {
            blocks: Vec::new(),
            head: 0,
            cursor: 0,
            advance: 0,
            last_hash: HASH_SEED,
            pending: Vec::new(),
            kda_states: Vec::new(),
            conv_states: Vec::new(),
        };
        validate_recurrent_state_schema(&schema, &state).unwrap();
        state.kda_states.push(Tensor::zeros(&[1, 2, 3], DType::F32));
        assert!(validate_recurrent_state_schema(&schema, &state).is_err());
    }

    #[test]
    fn state_capacity_is_exact_and_propagated_to_the_plan() {
        let schema = KvStateSchema {
            kv_dtype: DType::U8,
            kda: KdaGeometry {
                layers: 1,
                heads: 3,
                head_dim: 4,
                value_dim: 5,
                dtype: DType::F64,
            },
            conv: ConvGeometry {
                layers: 2,
                channels: 7,
                kernel: 3,
            },
            ..test_state_schema()
        };
        let state_bytes = schema.referenced_state_bytes().unwrap();
        assert_eq!(state_bytes, 1_440);

        let root = leaf(Tensor::ones(&[2], DType::F32));
        let compilation = executable::compile_with_state_bytes(
            &[root],
            CompileOptions::default(),
            CHUNKED_CE_CHUNK_LOGITS,
            Some(state_bytes),
        )
        .unwrap();
        assert_eq!(compilation.executable.memory.report.state_bytes, 1_440);
        assert_eq!(compilation.executable.diagnostics.memory.state_bytes, 1_440);
    }

    #[test]
    fn kv_slab_scatter_gather_roundtrip() {
        let slab = pool::Slab::new(8, 6, DType::F32);
        let source = (0..12).map(|value| value as f32).collect::<Vec<_>>();
        slab.write_rows_f32(&[4, 5], &source);
        assert_eq!(slab.read_rows_f32(&[4, 5]), source);
    }

    #[test]
    fn packed_causal_rows_compact_ragged_chains_and_pad_the_graph() {
        let packed = PackedCausalRows::build(8, 2, &[1, 0], &[7, 19], &[3, 2]).unwrap();
        assert_eq!(packed.row_offsets, [0, 3, 5]);
        assert_eq!(packed.logical_rows, 5);
        assert_eq!(
            packed.row_to_physical,
            [
                Some(1),
                Some(1),
                Some(1),
                Some(0),
                Some(0),
                None,
                None,
                None
            ]
        );
        assert_eq!(packed.positions, [7, 8, 9, 19, 20, 0, 0, 0]);

        assert!(PackedCausalRows::build(4, 2, &[0, 1], &[0, 0], &[3, 2]).is_err());
        assert!(PackedCausalRows::build(4, 2, &[2], &[0], &[1]).is_err());
        assert!(PackedCausalRows::build(4, 2, &[0], &[0], &[0]).is_err());
    }

    #[test]
    fn packed_kv_scatter_uses_distinct_positions_and_padding_is_inactive() {
        let pool = Arc::new(PoolInner {
            k: vec![pool::Slab::new(4, 1, DType::F32)],
            v: vec![pool::Slab::new(4, 1, DType::F32)],
            scales: Vec::new(),
            kv_dtype: DType::F32,
            kv_heads: 1,
            head_dim: 1,
            block_size: 4,
            max_tokens: 4,
            kda: KdaGeometry::default(),
            conv: ConvGeometry::default(),
            blocks: Mutex::new(BlockStore::new(1)),
        });
        let state = Arc::new(Mutex::new(SeqState {
            blocks: Vec::new(),
            head: 0,
            cursor: 0,
            advance: 2,
            last_hash: HASH_SEED,
            pending: Vec::new(),
            kda_states: Vec::new(),
            conv_states: Vec::new(),
        }));
        let packed = PackedCausalRows::build(3, 1, &[0], &[0], &[2]).unwrap();
        let context = KvContext {
            pool: pool.clone(),
            slots: vec![Some(state.clone())],
            advances: vec![2],
            packed: Some(packed),
            window: None,
            kda: KdaGeometry::default(),
            conv: ConvGeometry::default(),
            transaction: Mutex::new(None),
        };
        let q = Value(Tensor::from_vec(vec![1.0f32, 1.0, 99.0], vec![3, 1, 1, 1]));
        let k = Value(Tensor::from_vec(vec![1.0f32, 2.0, 99.0], vec![3, 1, 1, 1]));
        let v = Value(Tensor::from_vec(
            vec![10.0f32, 20.0, 99.0],
            vec![3, 1, 1, 1],
        ));
        let mut actual = Tensor::zeros(&q.shape(), DType::F32);
        let mut destination = CpuDestination::new(&mut actual).unwrap();
        let mut eviction_starts = vec![usize::MAX];
        kv_attention_into(
            &context,
            0,
            &q,
            &k,
            &v,
            1.0,
            None,
            KvAttentionMode::Causal,
            &mut destination,
            &mut eviction_starts,
        )
        .unwrap();
        drop(destination);

        assert_eq!(pool.k[0].read_rows_f32(&[0, 1]), [1.0, 2.0]);
        assert_eq!(pool.v[0].read_rows_f32(&[0, 1]), [10.0, 20.0]);
        let output = Value(actual).to_f32_vec().unwrap();
        assert_eq!(output[0], 10.0);
        assert!(output[1] > 10.0 && output[1] < 20.0);
        assert_eq!(output[2], 0.0);
        assert_eq!(state.lock().unwrap().advance, 2);
    }

    #[test]
    fn kv_attention_matches_sdpa() {
        let pool = Arc::new(PoolInner {
            k: vec![pool::Slab::new(8, 8, DType::F32)],
            v: vec![pool::Slab::new(8, 8, DType::F32)],
            scales: Vec::new(),
            kv_dtype: DType::F32,
            kv_heads: 2,
            head_dim: 4,
            block_size: 4,
            max_tokens: 8,
            kda: KdaGeometry::default(),
            conv: ConvGeometry::default(),
            blocks: Mutex::new(BlockStore::new(2)),
        });
        let state = Arc::new(Mutex::new(SeqState {
            blocks: Vec::new(),
            head: 0,
            cursor: 0,
            advance: 3,
            last_hash: HASH_SEED,
            pending: Vec::new(),
            kda_states: Vec::new(),
            conv_states: Vec::new(),
        }));
        let context = KvContext {
            pool,
            slots: vec![Some(state.clone())],
            advances: vec![3],
            packed: None,
            window: None,
            kda: KdaGeometry::default(),
            conv: ConvGeometry::default(),
            transaction: Mutex::new(None),
        };
        let q = Value(Tensor::from_vec(
            (0..48).map(|value| value as f32).collect(),
            vec![1, 4, 3, 4],
        ));
        let k = Value(Tensor::from_vec(
            (24..48).map(|value| value as f32 * 0.01).collect(),
            vec![1, 2, 3, 4],
        ));
        let v = Value(Tensor::from_vec(
            (48..72).map(|value| value as f32 * 0.01).collect(),
            vec![1, 2, 3, 4],
        ));
        let repeat_heads = |tensor: &Tensor| {
            let values = Value(tensor.clone()).to_f32_vec().unwrap();
            let per_head = 3 * 4;
            let mut repeated = Vec::new();
            for head in 0..2 {
                for _ in 0..2 {
                    repeated.extend_from_slice(&values[head * per_head..(head + 1) * per_head]);
                }
            }
            Tensor::from_vec(repeated, vec![1, 4, 3, 4])
        };
        let expected = Value(composed::sdpa_forward(
            q.tensor(),
            &repeat_heads(k.tensor()),
            &repeat_heads(v.tensor()),
            0.5,
            true,
        ))
        .to_f32_vec()
        .unwrap();
        let mut actual = Tensor::zeros(&q.shape(), DType::F32);
        let mut destination = CpuDestination::new(&mut actual).unwrap();
        let mut eviction_starts = vec![usize::MAX];
        {
            let _guard = crate::ExecutableAllocationGuard::enter();
            kv_attention_into(
                &context,
                0,
                &q,
                &k,
                &v,
                0.5,
                None,
                KvAttentionMode::Causal,
                &mut destination,
                &mut eviction_starts,
            )
            .unwrap();
        }
        drop(destination);
        let actual = Value(actual).to_f32_vec().unwrap();
        assert_close(&actual, &expected, 1e-6, "kv attention");

        let expected_block = Value(composed::sdpa_forward(
            q.tensor(),
            &repeat_heads(k.tensor()),
            &repeat_heads(v.tensor()),
            0.5,
            false,
        ))
        .to_f32_vec()
        .unwrap();
        let mut block = Tensor::zeros(&q.shape(), DType::F32);
        let mut block_destination = CpuDestination::new(&mut block).unwrap();
        kv_attention_into(
            &context,
            0,
            &q,
            &k,
            &v,
            0.5,
            None,
            KvAttentionMode::BidirectionalBlock,
            &mut block_destination,
            &mut eviction_starts,
        )
        .unwrap();
        drop(block_destination);
        let block = Value(block).to_f32_vec().unwrap();
        assert_close(&block, &expected_block, 1e-6, "block kv attention");
        assert_ne!(block, actual);
        assert_eq!(state.lock().unwrap().advance, 3);
    }

    #[test]
    fn block_attention_window_retains_committed_rows_plus_current_block() {
        let pool = Arc::new(PoolInner {
            k: vec![pool::Slab::new(8, 1, DType::F32)],
            v: vec![pool::Slab::new(8, 1, DType::F32)],
            scales: Vec::new(),
            kv_dtype: DType::F32,
            kv_heads: 1,
            head_dim: 1,
            block_size: 4,
            max_tokens: 8,
            kda: KdaGeometry::default(),
            conv: ConvGeometry::default(),
            blocks: Mutex::new(BlockStore::new(2)),
        });
        let mut state = SeqState {
            blocks: Vec::new(),
            head: 0,
            cursor: 6,
            advance: 2,
            last_hash: HASH_SEED,
            pending: Vec::new(),
            kda_states: Vec::new(),
            conv_states: Vec::new(),
        };
        let (cursor, needed, start) = kv_prepare(
            &pool,
            &mut state,
            0,
            Some(3),
            KvAttentionMode::BidirectionalBlock,
            1,
            1,
            2,
        )
        .unwrap();
        assert_eq!((cursor, needed, start), (6, 8, 3));
    }

    #[test]
    fn compiled_kda_commits_planned_next_state() {
        let pool = Arc::new(PoolInner {
            k: Vec::new(),
            v: Vec::new(),
            scales: Vec::new(),
            kv_dtype: DType::F32,
            kv_heads: 0,
            head_dim: 0,
            block_size: 4,
            max_tokens: 8,
            kda: KdaGeometry {
                layers: 1,
                heads: 1,
                head_dim: 2,
                value_dim: 2,
                dtype: DType::F32,
            },
            conv: ConvGeometry::default(),
            blocks: Mutex::new(BlockStore::new(2)),
        });
        let state = Arc::new(Mutex::new(SeqState {
            blocks: Vec::new(),
            head: 0,
            cursor: 0,
            advance: 2,
            last_hash: HASH_SEED,
            pending: Vec::new(),
            kda_states: vec![Tensor::zeros(&[1, 2, 2], DType::F32)],
            conv_states: Vec::new(),
        }));
        let context = Arc::new(KvContext {
            pool,
            slots: vec![Some(state.clone())],
            advances: vec![2],
            packed: None,
            window: None,
            kda: KdaGeometry {
                layers: 1,
                heads: 1,
                head_dim: 2,
                value_dim: 2,
                dtype: DType::F32,
            },
            conv: ConvGeometry::default(),
            transaction: Mutex::new(None),
        });
        let leaf = |values: Vec<f32>, shape: Vec<usize>| {
            Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(Value(
                Tensor::from_vec(values, shape),
            )))))
            .unwrap()
        };
        let q = leaf(vec![0.2, 0.4, 0.1, 0.3], vec![1, 1, 2, 2]);
        let k = leaf(vec![0.3, 0.1, 0.4, 0.2], vec![1, 1, 2, 2]);
        let v = leaf(vec![0.5, 0.7, 0.6, 0.8], vec![1, 1, 2, 2]);
        let decay = leaf(vec![-0.1, -0.2, -0.3, -0.4], vec![1, 1, 2, 2]);
        let beta = leaf(vec![0.5, 0.25], vec![1, 1, 2, 1]);
        let recurrence = Node::new(NodeKind::KdaRecurrence {
            q: q.clone(),
            k: k.clone(),
            v: v.clone(),
            log_decay: decay.clone(),
            beta: beta.clone(),
            scale: 0.5,
            layer: 0,
        })
        .unwrap();
        let compilation = executable::compile(
            &[recurrence],
            CompileOptions {
                optimize: false,
                ..CompileOptions::default()
            },
            1024,
        )
        .unwrap();
        assert!(compilation.executable.memory.report.transaction_bytes > 0);

        let zero = Tensor::zeros(&[1, 2, 2], DType::F32);
        let leaf_value = |node: &Arc<Node>| -> Value {
            match &node.kind {
                NodeKind::Leaf(slot) => slot.get().unwrap(),
                _ => unreachable!("test nodes are leaves"),
            }
        };
        let q_value = leaf_value(&q);
        let k_value = leaf_value(&k);
        let v_value = leaf_value(&v);
        let decay_value = leaf_value(&decay);
        let beta_value = leaf_value(&beta);
        let (expected_output, expected_state) = composed::kda_chunk_with_state(
            q_value.tensor(),
            k_value.tensor(),
            v_value.tensor(),
            decay_value.tensor(),
            beta_value.tensor(),
            0.5,
            &zero,
        );
        let outputs = executable::execute(
            &compilation.executable,
            &[],
            &compilation.generated_bindings,
            &CancellationFlag::new(),
            Some(&context),
        )
        .unwrap();
        assert_close(
            &outputs[0].to_f32_vec().unwrap(),
            &Value(expected_output).to_f32_vec().unwrap(),
            1e-6,
            "compiled KDA output",
        );
        let committed = state.lock().unwrap().kda_states[0].clone();
        assert_close(
            &Value(committed).to_f32_vec().unwrap(),
            &Value(expected_state).to_f32_vec().unwrap(),
            1e-6,
            "compiled KDA state",
        );
    }

    fn last_token_row_context(advance: usize) -> Arc<KvContext> {
        Arc::new(KvContext {
            pool: Arc::new(PoolInner {
                k: Vec::new(),
                v: Vec::new(),
                scales: Vec::new(),
                kv_dtype: DType::F32,
                kv_heads: 0,
                head_dim: 0,
                block_size: 4,
                max_tokens: 8,
                kda: KdaGeometry::default(),
                conv: ConvGeometry::default(),
                blocks: Mutex::new(BlockStore::new(2)),
            }),
            slots: vec![Some(Arc::new(Mutex::new(SeqState {
                blocks: Vec::new(),
                head: 0,
                cursor: 0,
                advance,
                last_hash: HASH_SEED,
                pending: Vec::new(),
                kda_states: Vec::new(),
                conv_states: Vec::new(),
            })))],
            advances: vec![advance],
            packed: None,
            window: None,
            kda: KdaGeometry::default(),
            conv: ConvGeometry::default(),
            transaction: Mutex::new(None),
        })
    }

    fn compile_last_token_row() -> executable::CpuCompilation {
        let logits = Node::new(NodeKind::LastTokenRow {
            a: Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(Value(
                Tensor::from_vec((0..12).map(|value| value as f32).collect(), vec![1, 3, 4]),
            )))))
            .unwrap(),
        })
        .unwrap();
        executable::compile(
            &[logits],
            CompileOptions {
                optimize: false,
                ..CompileOptions::default()
            },
            1024,
        )
        .unwrap()
    }

    #[test]
    fn last_token_row_copies_the_advanced_row() {
        let compilation = compile_last_token_row();
        let context = last_token_row_context(2);
        let outputs = executable::execute(
            &compilation.executable,
            &[],
            &compilation.generated_bindings,
            &CancellationFlag::new(),
            Some(&context),
        )
        .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape(), &[4]);
        assert_eq!(outputs[0].to_f32_vec().unwrap(), [4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn last_token_rows_use_physical_lane_advances_and_skip_inactive_lanes() {
        let source = leaf(Tensor::from_vec(
            (0..24).map(|value| value as f32).collect(),
            vec![2, 3, 4],
        ));
        let row = |lane| {
            Node::new(NodeKind::LastTokenRow {
                a: Node::new(NodeKind::Slice {
                    a: source.clone(),
                    ranges: vec![(lane, lane + 1, 1), (0, 3, 1), (0, 4, 1)],
                })
                .unwrap(),
            })
            .unwrap()
        };
        let compilation = executable::compile(
            &[row(0), row(1)],
            CompileOptions {
                optimize: false,
                ..CompileOptions::default()
            },
            1024,
        )
        .unwrap();
        let active = Arc::new(Mutex::new(SeqState {
            blocks: Vec::new(),
            head: 0,
            cursor: 7,
            advance: 2,
            last_hash: HASH_SEED,
            pending: Vec::new(),
            kda_states: Vec::new(),
            conv_states: Vec::new(),
        }));
        let context = Arc::new(KvContext {
            pool: Arc::new(block_store_pool(2)),
            slots: vec![None, Some(active)],
            advances: vec![0, 2],
            packed: None,
            window: None,
            kda: KdaGeometry::default(),
            conv: ConvGeometry::default(),
            transaction: Mutex::new(None),
        });

        let outputs = executable::execute(
            &compilation.executable,
            &[],
            &compilation.generated_bindings,
            &CancellationFlag::new(),
            Some(&context),
        )
        .unwrap();

        assert_eq!(outputs[0].to_f32_vec().unwrap(), [0.0; 4]);
        assert_eq!(outputs[1].to_f32_vec().unwrap(), [16.0, 17.0, 18.0, 19.0]);
    }

    #[test]
    fn last_token_row_requires_a_state_context_and_a_valid_advance() {
        let compilation = compile_last_token_row();
        let error = executable::execute(
            &compilation.executable,
            &[],
            &compilation.generated_bindings,
            &CancellationFlag::new(),
            None,
        )
        .err()
        .unwrap();
        assert_eq!(error, "last_token_row: operation requires a state context");

        for advance in [0, 4] {
            let context = last_token_row_context(advance);
            let error = executable::execute(
                &compilation.executable,
                &[],
                &compilation.generated_bindings,
                &CancellationFlag::new(),
                Some(&context),
            )
            .err()
            .unwrap();
            assert_eq!(
                error,
                format!("last token row: token advance must be in 1..=3, got {advance}")
            );
        }
    }

    #[test]
    fn last_token_row_compile_flag_rewrites_decode_outputs() {
        let root = LazyTensor {
            node: Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(Value(
                Tensor::from_vec((0..24).map(|value| value as f32).collect(), vec![1, 3, 8]),
            )))))
            .unwrap(),
        };
        let schema = NativeKvStateSchema {
            max_tokens: 8,
            block_size: 4,
            kv_dtype: NativeDType::F32,
            window: None,
            batch: 1,
            packed_causal_chains: None,
            last_token_row: Some(true),
            output_selections: None,
            current_block_attention: None,
        };
        let executable = compile(vec![&root], None, Some(schema), None).unwrap();
        let outputs = &executable.inner.executable.signature.outputs;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape, vec![8]);
    }

    #[test]
    fn per_root_decode_outputs_split_logits_and_batch_hidden_rows() {
        let logits = LazyTensor {
            node: Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(Value(
                Tensor::zeros(&[2, 3, 8], DType::F32),
            )))))
            .unwrap(),
        };
        let hidden = LazyTensor {
            node: Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(Value(
                Tensor::zeros(&[2, 3, 6], DType::F32),
            )))))
            .unwrap(),
        };
        let executable = compile(
            vec![&logits, &hidden],
            None,
            Some(NativeKvStateSchema {
                max_tokens: 8,
                block_size: 4,
                kv_dtype: NativeDType::F32,
                window: None,
                batch: 2,
                packed_causal_chains: None,
                last_token_row: None,
                output_selections: Some(vec![
                    NativeDecodeOutputSelection::SplitLastTokenRow,
                    NativeDecodeOutputSelection::BatchedLastTokenRow,
                ]),
                current_block_attention: None,
            }),
            None,
        )
        .unwrap();
        let outputs = &executable.inner.executable.signature.outputs;
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].shape, vec![8]);
        assert_eq!(outputs[1].shape, vec![8]);
        assert_eq!(outputs[2].shape, vec![2, 6]);
    }

    #[test]
    fn packed_compile_preserves_all_graph_rows_and_rejects_malformed_layouts() {
        let root = LazyTensor {
            node: Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(Value(
                Tensor::zeros(&[6, 1, 8], DType::F32),
            )))))
            .unwrap(),
        };
        let schema = |rows_per_sequence| NativeKvStateSchema {
            max_tokens: 8,
            block_size: 4,
            kv_dtype: NativeDType::F32,
            window: None,
            batch: 2,
            packed_causal_chains: Some(NativePackedCausalChainsLayout { rows_per_sequence }),
            last_token_row: None,
            output_selections: None,
            current_block_attention: None,
        };
        let executable = compile(vec![&root], None, Some(schema(3)), None).unwrap();
        assert_eq!(
            executable.inner.executable.signature.outputs[0].shape,
            vec![6, 1, 8]
        );
        assert_eq!(executable.state.unwrap().packed_rows_per_sequence, Some(3));

        assert!(compile(vec![&root], None, Some(schema(0)), None).is_err());
    }

    #[test]
    fn compiled_short_conv_commits_planned_next_state() {
        let pool = Arc::new(PoolInner {
            k: Vec::new(),
            v: Vec::new(),
            scales: Vec::new(),
            kv_dtype: DType::F32,
            kv_heads: 0,
            head_dim: 0,
            block_size: 4,
            max_tokens: 8,
            kda: KdaGeometry::default(),
            conv: ConvGeometry {
                layers: 1,
                channels: 2,
                kernel: 3,
            },
            blocks: Mutex::new(BlockStore::new(2)),
        });
        let state = Arc::new(Mutex::new(SeqState {
            blocks: Vec::new(),
            head: 0,
            cursor: 0,
            advance: 2,
            last_hash: HASH_SEED,
            pending: Vec::new(),
            kda_states: Vec::new(),
            conv_states: vec![Tensor::zeros(&[2, 2], DType::F32)],
        }));
        let context = Arc::new(KvContext {
            pool,
            slots: vec![Some(state.clone())],
            advances: vec![2],
            packed: None,
            window: None,
            kda: KdaGeometry::default(),
            conv: ConvGeometry {
                layers: 1,
                channels: 2,
                kernel: 3,
            },
            transaction: Mutex::new(None),
        });
        let x_value = Value(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]));
        let weight_value = Value(Tensor::from_vec(
            vec![0.5, 1.0, -0.5, 0.25, 0.75, 1.25],
            vec![2, 3],
        ));
        let x = Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(x_value.clone())))).unwrap();
        let weight = Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(
            weight_value.clone(),
        ))))
        .unwrap();
        let operation = Node::new(NodeKind::ConvState {
            x,
            weight,
            layer: 0,
        })
        .unwrap();
        let compilation = executable::compile(
            &[operation],
            CompileOptions {
                optimize: false,
                ..CompileOptions::default()
            },
            1024,
        )
        .unwrap();
        let outputs = executable::execute(
            &compilation.executable,
            &[],
            &compilation.generated_bindings,
            &CancellationFlag::new(),
            Some(&context),
        )
        .unwrap();
        assert_close(
            &outputs[0].to_f32_vec().unwrap(),
            &[-0.5, 2.5, -0.5, 6.5],
            1e-6,
            "compiled convolution output",
        );
        let committed = state.lock().unwrap().conv_states[0].clone();
        assert_close(
            &Value(committed).to_f32_vec().unwrap(),
            &[1.0, 2.0, 3.0, 4.0],
            1e-6,
            "compiled convolution state",
        );
    }

    fn block_store_pool(blocks: usize) -> PoolInner {
        PoolInner {
            k: Vec::new(),
            v: Vec::new(),
            scales: Vec::new(),
            kv_dtype: DType::F32,
            kv_heads: 1,
            head_dim: 1,
            block_size: 2,
            max_tokens: blocks * 2,
            kda: KdaGeometry::default(),
            conv: ConvGeometry::default(),
            blocks: Mutex::new(BlockStore::new(blocks)),
        }
    }

    #[test]
    fn prefix_cache_take_and_reclaim() {
        let pool = block_store_pool(2);
        let a = pool.alloc_block().unwrap();
        let b = pool.alloc_block().unwrap();
        assert!(pool.alloc_block().is_none());
        pool.set_hash(a, 42);
        pool.unref_block(a);
        assert_eq!(pool.cached_count(), 1);
        assert_eq!(pool.take_block(42), Some(a));
        assert_eq!(pool.cached_count(), 0);
        pool.unref_block(a);
        assert_eq!(pool.alloc_block(), Some(a));
        pool.unref_block(a);
        pool.unref_block(b);
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn note_tokens_hashes_completed_blocks() {
        let pool = block_store_pool(2);
        let mut state = SeqState {
            blocks: vec![pool.alloc_block().unwrap(), pool.alloc_block().unwrap()],
            head: 0,
            cursor: 0,
            advance: 0,
            last_hash: HASH_SEED,
            pending: Vec::new(),
            kda_states: Vec::new(),
            conv_states: Vec::new(),
        };
        state.note_tokens(&pool, &[7, 8, 9]);
        let first = chain_hash(HASH_SEED, &[7, 8]);
        let second = chain_hash(first, &[9, 5]);
        assert_eq!(
            pool.blocks.lock().unwrap().hashes[state.blocks[0] as usize],
            Some(first)
        );
        state.cursor = 3;
        state.note_tokens(&pool, &[5]);
        assert_eq!(
            pool.blocks.lock().unwrap().hashes[state.blocks[1] as usize],
            Some(second)
        );
    }

    #[test]
    fn speculative_hashes_are_deferred_until_publication() {
        let pool = block_store_pool(1);
        let block = pool.alloc_block().unwrap();
        let mut state = SeqState {
            blocks: vec![block],
            head: 0,
            cursor: 0,
            advance: 0,
            last_hash: HASH_SEED,
            pending: Vec::new(),
            kda_states: Vec::new(),
            conv_states: Vec::new(),
        };
        let hashes = speculative_note_tokens(&mut state, &pool, &[3, 4]);
        assert_eq!(hashes, vec![(block, chain_hash(HASH_SEED, &[3, 4]))]);
        assert_eq!(pool.blocks.lock().unwrap().hashes[block as usize], None);
        for (block, hash) in hashes {
            pool.set_hash(block, hash);
        }
        assert_eq!(
            pool.blocks.lock().unwrap().hashes[block as usize],
            Some(chain_hash(HASH_SEED, &[3, 4]))
        );
    }

    #[test]
    fn cohesive_target_prefill_is_invisible_until_paired_publication() {
        let target_pool = Arc::new(block_store_pool(1));
        let proposer_pool = Arc::new(block_store_pool(1));
        let target_metadata = Arc::new(Mutex::new(DeferredPrefixMetadata::default()));
        let proposer_metadata = Arc::new(Mutex::new(DeferredPrefixMetadata::default()));
        let target_hash = chain_hash(HASH_SEED, &[3, 4]);
        let proposer_hash = chain_hash(HASH_SEED, &[3, 4]);
        let paused = Arc::new(std::sync::Barrier::new(2));
        let resume = Arc::new(std::sync::Barrier::new(2));

        let worker = {
            let target_pool = target_pool.clone();
            let proposer_pool = proposer_pool.clone();
            let target_metadata = target_metadata.clone();
            let proposer_metadata = proposer_metadata.clone();
            let paused = paused.clone();
            let resume = resume.clone();
            std::thread::spawn(move || {
                let target_block = target_pool.alloc_block().unwrap();
                let mut target = SeqState {
                    blocks: vec![target_block],
                    head: 0,
                    cursor: 0,
                    advance: 0,
                    last_hash: HASH_SEED,
                    pending: Vec::new(),
                    kda_states: Vec::new(),
                    conv_states: Vec::new(),
                };
                target_metadata
                    .lock()
                    .unwrap()
                    .hashes
                    .extend(target.note_tokens_deferred(&target_pool, &[3, 4]));
                paused.wait();
                resume.wait();

                let proposer_block = proposer_pool.alloc_block().unwrap();
                let mut proposer = SeqState {
                    blocks: vec![proposer_block],
                    head: 0,
                    cursor: 0,
                    advance: 0,
                    last_hash: HASH_SEED,
                    pending: Vec::new(),
                    kda_states: Vec::new(),
                    conv_states: Vec::new(),
                };
                proposer_metadata
                    .lock()
                    .unwrap()
                    .hashes
                    .extend(proposer.note_tokens_deferred(&proposer_pool, &[3, 4]));
                publish_deferred_prefixes(
                    &target_pool,
                    &target_metadata,
                    Some((&proposer_pool, &proposer_metadata)),
                    || (),
                )
                .unwrap();
            })
        };

        paused.wait();
        assert_eq!(target_pool.cached_count(), 0);
        assert_eq!(target_pool.take_block(target_hash), None);
        assert!(target_pool.blocks.lock().unwrap().by_hash.is_empty());
        resume.wait();
        worker.join().unwrap();

        assert!(target_pool.take_block(target_hash).is_some());
        assert!(proposer_pool.take_block(proposer_hash).is_some());
    }

    #[test]
    fn cohesive_proposer_exhaustion_rolls_back_without_changing_target_cache() {
        let target_pool = Arc::new(block_store_pool(2));
        let cached = target_pool.alloc_block().unwrap();
        target_pool.set_hash(cached, 17);
        target_pool.unref_block(cached);
        let baseline = target_pool.cached_count();

        let provisional = target_pool.alloc_block().unwrap();
        let mut target = SeqState {
            blocks: vec![provisional],
            head: 0,
            cursor: 0,
            advance: 0,
            last_hash: HASH_SEED,
            pending: Vec::new(),
            kda_states: Vec::new(),
            conv_states: Vec::new(),
        };
        let staged_hash = chain_hash(HASH_SEED, &[8, 9]);
        let mut metadata = DeferredPrefixMetadata::default();
        metadata
            .hashes
            .extend(target.note_tokens_deferred(&target_pool, &[8, 9]));

        let proposer_pool = block_store_pool(1);
        let occupied = proposer_pool.alloc_block().unwrap();
        assert_eq!(
            proposer_pool.alloc_block(),
            None,
            "forced proposer exhaustion"
        );
        drop(metadata);
        target_pool.discard_block(provisional);

        assert_eq!(target_pool.cached_count(), baseline);
        assert_eq!(target_pool.take_block(staged_hash), None);
        assert_eq!(target_pool.take_block(17), Some(cached));
        target_pool.unref_block(cached);
        assert_eq!(target_pool.cached_count(), baseline);
        proposer_pool.unref_block(occupied);
    }

    #[test]
    fn cohesive_prefix_claims_are_independent_and_rollback_preserves_caches() {
        let prompt = vec![1, 2, 3, 4, 5, 6];
        let first_hash = chain_hash(HASH_SEED, &[1, 2]);
        let second_hash = chain_hash(first_hash, &[3, 4]);
        let target_pool = Arc::new(block_store_pool(4));
        let proposer_pool = Arc::new(block_store_pool(4));

        for hash in [first_hash, second_hash] {
            let block = target_pool.alloc_block().unwrap();
            target_pool.set_hash(block, hash);
            target_pool.unref_block(block);
        }
        let proposer_block = proposer_pool.alloc_block().unwrap();
        proposer_pool.set_hash(proposer_block, first_hash);
        proposer_pool.unref_block(proposer_block);

        let target = NativeKvSequence::new(target_pool.clone());
        let proposer = NativeKvSequence::new(proposer_pool.clone());
        assert_eq!(target.prefill_match(prompt.clone()).unwrap(), 4);
        assert_eq!(proposer.prefill_match(prompt.clone()).unwrap(), 2);
        assert_eq!(prompt.len() - target.cursor() as usize, 2);
        assert_eq!(prompt.len() - proposer.cursor() as usize, 4);
        assert!(target.cursor() as usize + 1 < prompt.len());
        assert!(proposer.cursor() as usize + 1 < prompt.len());

        let target_suffix = target_pool.alloc_block().unwrap();
        target.state.lock().unwrap().blocks.push(target_suffix);
        let proposer_suffix = proposer_pool.alloc_block().unwrap();
        proposer.state.lock().unwrap().blocks.push(proposer_suffix);
        target.return_blocks();
        proposer.return_blocks();

        let target_store = target_pool.blocks.lock().unwrap();
        assert_eq!(target_store.refcounts, vec![0; 4]);
        assert_eq!(target_store.hashes.iter().flatten().count(), 2);
        assert_eq!(target_store.by_hash.len(), 2);
        assert!(target_store.by_hash.contains_key(&first_hash));
        assert!(target_store.by_hash.contains_key(&second_hash));
        drop(target_store);
        let proposer_store = proposer_pool.blocks.lock().unwrap();
        assert_eq!(proposer_store.refcounts, vec![0; 4]);
        assert_eq!(proposer_store.hashes.iter().flatten().count(), 1);
        assert_eq!(proposer_store.by_hash.len(), 1);
        assert!(proposer_store.by_hash.contains_key(&first_hash));
    }

    #[test]
    fn speculative_shadow_shares_the_unreachable_partial_tail_and_rolls_back() {
        let pool = Arc::new(block_store_pool(2));
        let original = pool.alloc_block().unwrap();
        let state = Arc::new(Mutex::new(SeqState {
            blocks: vec![original],
            head: 0,
            cursor: 1,
            advance: 0,
            last_hash: HASH_SEED,
            pending: vec![7],
            kda_states: Vec::new(),
            conv_states: Vec::new(),
        }));
        let shadow = clone_speculative_state(&pool, &state).unwrap();
        assert_eq!(shadow.lock().unwrap().blocks[0], original);
        assert_eq!(state.lock().unwrap().blocks, vec![original]);
        discard_speculative_states(&pool, &[shadow]);
        assert_eq!(state.lock().unwrap().blocks, vec![original]);
        assert_eq!(pool.available(), 1);
    }

    #[test]
    fn speculative_rng_coordinates_ignore_slot_and_request_order() {
        let proposal = purpose_counter(91, SamplingPurpose::Proposal, 0);
        assert_eq!(proposal, purpose_counter(91, SamplingPurpose::Proposal, 0));
        assert_ne!(proposal, purpose_counter(91, SamplingPurpose::Accept, 0));
        assert_ne!(proposal, purpose_counter(92, SamplingPurpose::Proposal, 0));
        assert_ne!(
            purpose_counter(95, SamplingPurpose::Residual, 0),
            purpose_counter(95, SamplingPurpose::Target, 0)
        );
    }

    fn hybrid_pool(blocks: usize) -> PoolInner {
        PoolInner {
            k: vec![pool::Slab::new(blocks * 2, 1, DType::F32)],
            v: vec![pool::Slab::new(blocks * 2, 1, DType::F32)],
            scales: Vec::new(),
            kv_dtype: DType::F32,
            kv_heads: 1,
            head_dim: 1,
            block_size: 2,
            max_tokens: blocks * 2,
            kda: KdaGeometry {
                layers: 1,
                heads: 1,
                head_dim: 2,
                value_dim: 2,
                dtype: DType::F32,
            },
            conv: ConvGeometry {
                layers: 1,
                channels: 2,
                kernel: 3,
            },
            blocks: Mutex::new(BlockStore::new(blocks)),
        }
    }

    fn hybrid_state(kda_fill: f32, conv_fill: f32) -> SeqState {
        SeqState {
            blocks: Vec::new(),
            head: 0,
            cursor: 0,
            advance: 0,
            last_hash: HASH_SEED,
            pending: Vec::new(),
            kda_states: vec![Tensor::from_vec(vec![kda_fill; 4], vec![1, 2, 2])],
            conv_states: vec![Tensor::from_vec(vec![conv_fill; 4], vec![2, 2])],
        }
    }

    fn snapshot_of(state: &SeqState) -> RecurrentSnapshot {
        capture_recurrent_snapshot(state).expect("hybrid state captures")
    }

    #[test]
    fn snapshot_publishes_only_at_resident_block_boundaries() {
        let pool = hybrid_pool(4);
        let mut state = hybrid_state(1.0, 2.0);
        state.cursor = 3;
        state.last_hash = 42;
        pool.maybe_publish_recurrent_snapshot(&state);
        assert!(pool.blocks.lock().unwrap().snapshots.is_empty());

        state.cursor = 4;
        pool.maybe_publish_recurrent_snapshot(&state);
        assert!(
            pool.blocks.lock().unwrap().snapshots.is_empty(),
            "non-resident hashes must not gain snapshots"
        );

        let intermediate = pool.alloc_block().unwrap();
        pool.set_hash(intermediate, 7);
        let last = pool.alloc_block().unwrap();
        pool.set_hash(last, 42);
        pool.maybe_publish_recurrent_snapshot(&state);
        let store = pool.blocks.lock().unwrap();
        assert!(store.snapshots.contains_key(&42));
        assert_eq!(store.snapshots.len(), 1, "intermediate hashes stay KV-only");
        drop(store);

        let kv_only = PoolInner {
            kda: KdaGeometry::default(),
            conv: ConvGeometry::default(),
            ..hybrid_pool(2)
        };
        kv_only.set_hash(kv_only.alloc_block().unwrap(), 42);
        kv_only.maybe_publish_recurrent_snapshot(&state);
        assert!(kv_only.blocks.lock().unwrap().snapshots.is_empty());

        let recurrent_only = PoolInner {
            k: Vec::new(),
            v: Vec::new(),
            ..hybrid_pool(2)
        };
        recurrent_only.set_hash(recurrent_only.alloc_block().unwrap(), 42);
        recurrent_only.maybe_publish_recurrent_snapshot(&state);
        assert!(recurrent_only.blocks.lock().unwrap().snapshots.is_empty());
    }

    #[test]
    fn snapshot_prefix_truncates_to_the_deepest_snapshot_boundary() {
        let pool = hybrid_pool(4);
        let (h1, h2, h3) = (11, 22, 33);
        for hash in [h1, h2, h3] {
            let block = pool.alloc_block().unwrap();
            pool.set_hash(block, hash);
            pool.unref_block(block);
        }
        pool.publish_snapshot(h1, snapshot_of(&hybrid_state(1.0, 1.0)));
        let (blocks, boundary, _) = pool.take_snapshot_prefix(&[h1, h2, h3]).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(boundary, h1);
        for block in blocks {
            pool.unref_block(block);
        }

        pool.publish_snapshot(h3, snapshot_of(&hybrid_state(3.0, 3.0)));
        let (blocks, boundary, _) = pool.take_snapshot_prefix(&[h1, h2, h3]).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(boundary, h3);
        for block in blocks {
            pool.unref_block(block);
        }

        let (blocks, boundary, _) = pool.take_snapshot_prefix(&[h1, 99, h3]).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(boundary, h1);
        for block in blocks {
            pool.unref_block(block);
        }

        assert!(pool.take_snapshot_prefix(&[h2]).is_none());
        assert!(pool.take_snapshot_prefix(&[]).is_none());
    }

    #[test]
    fn snapshot_restore_deep_copies_and_validates_geometry() {
        let snapshot = snapshot_of(&hybrid_state(3.0, 4.0));
        let mut state = hybrid_state(0.0, 0.0);
        assert!(restore_recurrent_snapshot(&mut state, &snapshot));
        assert_eq!(
            Value(state.kda_states[0].clone()).to_f32_vec().unwrap(),
            vec![3.0; 4]
        );
        assert_eq!(
            Value(state.conv_states[0].clone()).to_f32_vec().unwrap(),
            vec![4.0; 4]
        );

        state.kda_states[0] = Tensor::from_vec(vec![9.0; 4], vec![1, 2, 2]);
        state.conv_states[0] = Tensor::from_vec(vec![8.0; 4], vec![2, 2]);
        assert_eq!(snapshot.kda[0], vec![3.0; 4]);
        assert_eq!(snapshot.conv[0], vec![4.0; 4]);

        let mut mismatched = SeqState {
            kda_states: Vec::new(),
            ..hybrid_state(0.0, 0.0)
        };
        assert!(!restore_recurrent_snapshot(&mut mismatched, &snapshot));
        assert!(mismatched.kda_states.is_empty());
        assert_eq!(
            Value(mismatched.conv_states[0].clone())
                .to_f32_vec()
                .unwrap(),
            vec![0.0; 4]
        );
    }

    #[test]
    fn eviction_removes_the_snapshot_with_the_last_block() {
        let pool = hybrid_pool(1);
        let block = pool.alloc_block().unwrap();
        pool.set_hash(block, 7);
        pool.publish_snapshot(7, snapshot_of(&hybrid_state(1.0, 1.0)));
        pool.unref_block(block);
        assert!(pool.blocks.lock().unwrap().snapshots.contains_key(&7));

        assert_eq!(pool.alloc_block(), Some(block));
        assert!(pool.blocks.lock().unwrap().snapshots.is_empty());
        assert!(pool.take_snapshot_prefix(&[7]).is_none());
    }

    #[test]
    fn prefill_match_hybrid_requires_and_restores_snapshots() {
        let pool = NativeKvPool::new(
            1,
            1,
            1,
            8,
            Some(2),
            None,
            Some(NativeRecurrentStateSchema {
                kda_layers: 1,
                kda_heads: 1,
                kda_head_dim: 2,
                kda_value_dim: 2,
                conv_layers: 1,
                conv_channels: 2,
                conv_kernel: 3,
            }),
        )
        .unwrap();
        let tokens = vec![10, 11, 12, 13, 14];
        let h1 = chain_hash(HASH_SEED, &[10, 11]);
        let h2 = chain_hash(h1, &[12, 13]);
        for hash in [h1, h2] {
            let block = pool.inner.alloc_block().unwrap();
            pool.inner.set_hash(block, hash);
            pool.inner.unref_block(block);
        }

        let unmatched = pool.make_sequence();
        assert_eq!(unmatched.prefill_match(tokens.clone()).unwrap(), 0);
        assert_eq!(unmatched.cursor(), 0);
        assert!(unmatched.state.lock().unwrap().blocks.is_empty());

        pool.inner
            .publish_snapshot(h2, snapshot_of(&hybrid_state(5.0, 6.0)));
        let matched = pool.make_sequence();
        assert_eq!(matched.prefill_match(tokens).unwrap(), 4);
        let state = matched.state.lock().unwrap();
        assert_eq!(state.cursor, 4);
        assert_eq!(state.last_hash, h2);
        assert_eq!(state.blocks.len(), 2);
        assert_eq!(
            Value(state.kda_states[0].clone()).to_f32_vec().unwrap(),
            vec![5.0; 4]
        );
        assert_eq!(
            Value(state.conv_states[0].clone()).to_f32_vec().unwrap(),
            vec![6.0; 4]
        );
        drop(state);

        let pure = NativeKvPool::new(
            0,
            0,
            0,
            8,
            Some(2),
            None,
            Some(NativeRecurrentStateSchema {
                kda_layers: 1,
                kda_heads: 1,
                kda_head_dim: 2,
                kda_value_dim: 2,
                conv_layers: 0,
                conv_channels: 0,
                conv_kernel: 0,
            }),
        )
        .unwrap();
        assert_eq!(
            pure.make_sequence().prefill_match(vec![1, 2, 3]).unwrap(),
            0
        );
    }

    fn linear_head(
        targets: Vec<i64>,
        dtype: DType,
    ) -> (Arc<Node>, Arc<Node>, Arc<Node>, Arc<Node>, Arc<Node>) {
        let x_leaf = leaf(
            Tensor::from_vec(
                (0..24).map(|index| (index as f32 * 0.37).sin()).collect(),
                vec![2, 3, 4],
            )
            .cast(dtype),
        );
        let x = Node::new(NodeKind::Tanh { a: x_leaf.clone() }).unwrap();
        let weight = leaf(
            Tensor::from_vec(
                (0..32)
                    .map(|index| (index as f32 * 0.11).cos() * 0.5)
                    .collect(),
                vec![4, 8],
            )
            .cast(dtype),
        );
        let bias = leaf(
            Tensor::from_vec(
                (0..8).map(|index| index as f32 * 0.05 - 0.2).collect(),
                vec![8],
            )
            .cast(dtype),
        );
        let logits = Node::new(NodeKind::Linear {
            x,
            weight: weight.clone(),
            bias: bias.clone(),
        })
        .unwrap();
        let target = leaf(Tensor::from_vec(targets, vec![2, 3]));
        (logits, target, x_leaf, weight, bias)
    }

    #[test]
    fn chunked_cross_entropy_matches_plain_loss_and_gradients() {
        let (logits, target, x, weight, bias) = linear_head(vec![0, 1, 2, 3, 4, 5], DType::F32);
        let plain = chunked_head_ce_with(&logits, &target, -100, usize::MAX, 1).unwrap();
        let chunked = chunked_head_ce_with(&logits, &target, -100, 0, 1).unwrap();
        assert_close(&eval_f32(&plain), &eval_f32(&chunked), 1e-5, "loss");
        let plain_gradients =
            effect_torch_autodiff::grad(&plain, &[x.clone(), weight.clone(), bias.clone()])
                .unwrap();
        let chunked_gradients = effect_torch_autodiff::grad(&chunked, &[x, weight, bias]).unwrap();
        for ((name, plain), chunked) in ["dx", "dw", "db"]
            .iter()
            .zip(&plain_gradients)
            .zip(&chunked_gradients)
        {
            assert_close(&eval_f32(plain), &eval_f32(chunked), 1e-4, name);
        }
    }

    #[test]
    fn chunked_cross_entropy_handles_ignored_chunks() {
        let (logits, target, ..) = linear_head(vec![-100, 1, 2, 3, -100, 5], DType::F32);
        let plain = chunked_head_ce_with(&logits, &target, -100, usize::MAX, 1).unwrap();
        let chunked = chunked_head_ce_with(&logits, &target, -100, 0, 1).unwrap();
        assert_close(&eval_f32(&plain), &eval_f32(&chunked), 1e-5, "loss");
    }

    #[test]
    fn chunked_cross_entropy_preserves_bf16_dtype() {
        let (logits, target, ..) = linear_head(vec![0, 1, 2, 3, 4, 5], DType::BF16);
        let chunked = chunked_head_ce_with(&logits, &target, -100, 0, 1).unwrap();
        assert_eq!(chunked.dtype, DType::BF16);
    }

    #[test]
    fn gelu_gradient_matches_finite_difference() {
        for approximate in [false, true] {
            let data = (0..12)
                .map(|index| (index as f32 * 0.43).sin() * 2.0)
                .collect::<Vec<_>>();
            let x = leaf(Tensor::from_vec(data.clone(), vec![3, 4]));
            let output = Node::new(NodeKind::Gelu {
                a: x.clone(),
                approximate,
            })
            .unwrap();
            let loss = Node::new(NodeKind::Sum {
                a: output,
                dims: vec![0, 1],
                keepdims: false,
            })
            .unwrap();
            let gradient = effect_torch_autodiff::grad(&loss, std::slice::from_ref(&x)).unwrap();
            let actual = eval_f32(&gradient[0]);
            let epsilon = 1e-3;
            for index in 0..data.len() {
                let evaluate = |mut values: Vec<f32>, delta: f32| {
                    values[index] += delta;
                    let x = leaf(Tensor::from_vec(values, vec![3, 4]));
                    let output = Node::new(NodeKind::Gelu { a: x, approximate }).unwrap();
                    let loss = Node::new(NodeKind::Sum {
                        a: output,
                        dims: vec![0, 1],
                        keepdims: false,
                    })
                    .unwrap();
                    eval_f32(&loss)[0]
                };
                let expected = (evaluate(data.clone(), epsilon) - evaluate(data.clone(), -epsilon))
                    / (2.0 * epsilon);
                assert!(
                    (actual[index] - expected).abs() / expected.abs().max(1.0) < 1e-2,
                    "gradient {index}: {} vs {expected}",
                    actual[index]
                );
            }
        }
    }

    #[test]
    fn inference_id_exhaustion_does_not_partially_advance_allocators() {
        let mut round_exhausted = InferenceIds {
            next_sequence_id: 7,
            next_round_id: u64::MAX,
        };
        assert!(reserve_inference_ids(&mut round_exhausted, 2).is_err());
        assert_eq!(round_exhausted.next_sequence_id, 7);
        assert_eq!(round_exhausted.next_round_id, u64::MAX);

        let mut sequence_exhausted = InferenceIds {
            next_sequence_id: u64::MAX,
            next_round_id: 9,
        };
        assert!(reserve_inference_ids(&mut sequence_exhausted, 1).is_err());
        assert_eq!(sequence_exhausted.next_sequence_id, u64::MAX);
        assert_eq!(sequence_exhausted.next_round_id, 9);
    }

    #[test]
    fn inference_round_override_preserves_unspecified_artifact_sampling() {
        let base = SamplingOptions {
            temperature: 0.25,
            top_k: Some(8),
            top_p: 0.75,
            seed: 91,
            counter: 0,
        };
        let effective = inference_sampling_override(
            &NativeInferenceSamplingOverrides {
                top_k: Some(4),
                ..NativeInferenceSamplingOverrides::default()
            },
            base,
            "sample",
        )
        .unwrap();
        assert_eq!(effective.temperature, 0.25);
        assert_eq!(effective.top_k, Some(4));
        assert_eq!(effective.top_p, 0.75);
        assert_eq!(effective.seed, 91);
    }

    #[test]
    fn durable_receipt_identity_distinguishes_operation_order_and_sampling() {
        let sampling = SamplingIdentity::from(sampling_options());
        let add = ReceiptRequest::Add(vec![AddRequestIdentity {
            prompt: vec![1, 2],
            sampling: sampling.clone(),
            max_tokens: 4,
            eos_tokens: vec![3],
        }]);
        let sampling = SamplingOverrideIdentity::from(&NativeInferenceSamplingOverrides {
            top_k: Some(4),
            ..NativeInferenceSamplingOverrides::default()
        });
        let round = ReceiptRequest::Round(vec![RoundRequestIdentity {
            sequence_id: 1,
            sampling: sampling.clone(),
        }]);
        let reordered = ReceiptRequest::Round(vec![
            RoundRequestIdentity {
                sequence_id: 2,
                sampling: sampling.clone(),
            },
            RoundRequestIdentity {
                sequence_id: 1,
                sampling,
            },
        ]);
        assert!(add != round);
        assert!(round != reordered);
    }

    #[test]
    fn cohesive_sequence_validation_rejects_foreign_and_duplicate_handles() {
        let first = NativeInferenceSequence {
            session_id: 1,
            sequence_id: 7,
        };
        let duplicate = first.clone();
        let foreign = NativeInferenceSequence {
            session_id: 2,
            sequence_id: 8,
        };
        assert!(inference_sequence_ids(1, &[&first, &duplicate]).is_err());
        assert!(inference_sequence_ids(1, &[&first, &foreign]).is_err());
        assert_eq!(inference_sequence_ids(1, &[&first]).unwrap(), vec![7]);
    }

    #[test]
    fn direct_lifecycle_mutation_rejects_active_operations() {
        let operation = Arc::new(tokio::sync::Mutex::new(()));
        let active = operation.clone().try_lock_owned().unwrap();
        let finish = try_lifecycle_operation(&operation, "finish").unwrap_err();
        let close = try_lifecycle_operation(&operation, "close").unwrap_err();
        assert_eq!(
            finish.reason,
            "inference[finish]: an inference operation is active"
        );
        assert_eq!(
            close.reason,
            "inference[close]: an inference operation is active"
        );
        drop(active);
        assert!(try_lifecycle_operation(&operation, "close").is_ok());
    }

    #[test]
    fn direct_finish_rejects_unacknowledged_receipts() {
        let mut state = InferenceSessionState {
            closed: false,
            sequences: HashMap::new(),
            receipt: Some(Receipt {
                round_id: 1,
                request: ReceiptRequest::Add(Vec::new()),
                pages: Vec::new(),
            }),
        };
        let error = validate_finish_lifecycle(&state).unwrap_err();
        assert_eq!(
            error.reason,
            "inference[finish]: round receipt must be acknowledged before finishing sequences"
        );
        state.receipt = None;
        assert!(validate_finish_lifecycle(&state).is_ok());
        state.closed = true;
        assert_eq!(
            validate_finish_lifecycle(&state).unwrap_err().reason,
            "inference[finish]: session is closed"
        );
    }

    #[test]
    fn transferred_kv_sequence_clone_releases_blocks_exactly_once() {
        let pool = Arc::new(hybrid_pool(2));
        let capacity = pool.available();
        let mut original = NativeKvSequence::new(pool.clone());
        original
            .state
            .lock()
            .unwrap()
            .blocks
            .push(pool.alloc_block().unwrap());
        let mut owner = original.clone();
        owner.finalize_releases = true;
        original.finalize_releases = false;
        drop(original);
        assert_eq!(pool.available(), capacity - 1);
        drop(owner);
        assert_eq!(pool.available(), capacity);
    }
}

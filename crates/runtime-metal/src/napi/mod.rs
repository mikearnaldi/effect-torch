//! Node-API bindings for the Metal backend.
//!
//! This module connects JavaScript handles to the graph, compiler, and runtime.
//! [`LazyTensor`] owns immutable graph nodes. [`NativeTensor`] owns concrete
//! Metal leaf slots and reports their external memory to V8. Compiled
//! executables keep frozen bindings and pipelines. KV pool and sequence types
//! own paged decode and recurrent state. Compile, execute, GGUF, and readback
//! work runs on blocking workers through
//! [`effect_torch_napi::CancellationState`]. Cancellation or completion wins
//! exactly once, so the runtime never publishes failed or cancelled work.
//!
//! # Readback ownership
//!
//! Small or strided readbacks copy into a Rust `Vec` and transfer its
//! allocation to an external `ArrayBuffer`. Large contiguous readbacks avoid
//! a copy. The runtime registers the buffer address, retains the Metal
//! allocation through a cloned `value::Value`, and unregisters and drops it in
//! the JS finalizer. `FinalizeHintGuard` restores Rust ownership if ArrayBuffer
//! creation fails. Every transferred allocation or retained tensor is released
//! exactly once.
//!
//! # Decode state
//!
//! KV blocks use reference counts and content hashes for prefix reuse. Before
//! dispatch, the runtime writes sequence cursors, block tables, KDA matrices,
//! and short-convolution windows into planned staging buffers. It commits them
//! only after successful execution. Snapshot copies synchronize through their
//! execution path. Callers must not overlap direct state preparation with GPU
//! writes to the same sequence.

mod err;
mod gguf;
mod runtime;
pub(crate) mod safetensors;
pub(crate) mod value;

pub use gguf::{inspect_gguf, load_gguf, load_gguf_for_device};

use crate::executable;
use crate::executable::{ConvGeometry, KdaGeometry, KvStateSchema, MetalDecodeContext, SeqState};
use crate::run::MetalTensor;

use effect_torch_compiler::{
    specialize_decode_layout_outputs_with_attention, CompileOptions,
    ConvGeometry as DecodeConvGeometry, CurrentBlockAttention, DecodeLayout, DecodeOutputSelection,
    InferenceOptions, KdaGeometry as DecodeKdaGeometry, PreparedProgram, ProgramRequest,
    ProgramSlot, StateCursorSlot,
};
use effect_torch_graph::CrossEntropyReduction as CeReduction;
use effect_torch_graph::{
    node_children, AttentionWindow, Device, KvAttentionMode, PositionOffset, RotaryLayout,
};
use effect_torch_napi::{try_register_export, unregister_export, vec_to_bytes, CancellationState};
use effect_torch_runtime::{
    effective_probabilities, purpose_counter, random_unit, random_unit_at, sample_probabilities,
    sampling_coordinate, Buffer, GgmlKQuant, SamplingOptions, SamplingPurpose,
    MAX_SAMPLING_VOCABULARY,
};
use runtime::dtype::DType;
pub type LeafSlot = effect_torch_graph::LeafSlot;
pub(crate) type Node = effect_torch_graph::Node;
pub(crate) type NodeKind = effect_torch_graph::NodeKind;

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

impl From<DecodeKdaGeometry> for KdaGeometry {
    fn from(geometry: DecodeKdaGeometry) -> Self {
        Self {
            layers: geometry.layers,
            heads: geometry.heads,
            head_dim: geometry.head_dim,
            value_dim: geometry.value_dim,
        }
    }
}

impl From<DecodeConvGeometry> for ConvGeometry {
    fn from(geometry: DecodeConvGeometry) -> Self {
        Self {
            layers: geometry.layers,
            channels: geometry.channels,
            kernel: geometry.kernel,
        }
    }
}

use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use err::to_napi_err;

use crate::paged;
use runtime::metal::device;

enum FinalizeHint {
    ZeroCopy {
        tensor: value::Value,
        addr: usize,
    },
    Owned {
        ptr: *mut u8,
        len: usize,
        cap: usize,
    },
}

unsafe extern "C" fn finalize_readback(
    _env: napi::sys::napi_env,
    _data: *mut std::ffi::c_void,
    hint: *mut std::ffi::c_void,
) {
    // SAFETY: `hint` was created by `Box::into_raw` in `to_napi_value`, is
    // handed to exactly one external ArrayBuffer, and Node invokes this
    // finalizer at most once. Reconstructing the box transfers ownership back
    // to Rust for the single release below.
    let hint = unsafe { Box::from_raw(hint as *mut FinalizeHint) };
    release_readback(*hint);
}

fn release_readback(hint: FinalizeHint) {
    match hint {
        FinalizeHint::ZeroCopy { tensor, addr } => {
            drop(tensor);
            unregister_export(addr);
        }
        FinalizeHint::Owned { ptr, len, cap } => {
            // SAFETY: `ptr`, `len`, and `cap` come unchanged from
            // `vec_to_bytes`, which forgot exactly one Vec allocation. This
            // branch is the unique owner and reconstructs it once for drop.
            drop(unsafe { Vec::from_raw_parts(ptr, len, cap) });
        }
    }
}

pub struct Readback {
    data: *mut u8,
    byte_len: usize,
    hint: Option<FinalizeHint>,
}

// SAFETY: `Readback` contains either uniquely owned Vec allocation metadata or
// a raw pointer retained by an owned `Value`. No thread dereferences `data`
// before conversion to an ArrayBuffer, and both ownership hints are Send.
unsafe impl Send for Readback {}

struct FinalizeHintGuard(*mut std::ffi::c_void);

impl Drop for FinalizeHintGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: a non-null guard owns the Box produced by
            // `Box::into_raw`; the pointer is nulled only after NAPI accepts
            // ownership, so this failure path reconstructs it exactly once.
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
    /// Transfers this readback's backing ownership to a Node ArrayBuffer.
    ///
    /// # Safety
    ///
    /// Called by NAPI with a live environment. `self.data` is valid for
    /// `self.byte_len` bytes and retained by `self.hint`; the finalizer takes
    /// over that hint exactly once if creation succeeds.
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
            // SAFETY: `data` is retained by `hint` for the ArrayBuffer's full
            // lifetime, `byte_len` is the validated logical byte extent, and
            // `finalize_readback` receives the same Box pointer for one-time
            // release.
            unsafe {
                napi::sys::napi_create_external_arraybuffer(
                    env,
                    value.data as *mut std::ffi::c_void,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
pub struct NativePackedCausalChainsLayout {
    pub rows_per_sequence: u32,
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
    let top_k = if top_k == 0 {
        None
    } else {
        Some(
            usize::try_from(top_k)
                .map_err(|_| Error::new(Status::InvalidArg, "sample: topK is out of range"))?,
        )
    };
    if !options.temperature.is_finite() || options.temperature < 0.0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "sample: temperature must be finite and non-negative, got {}",
                options.temperature
            ),
        ));
    }
    if !options.top_p.is_finite() || options.top_p <= 0.0 || options.top_p > 1.0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "sample: topP must be finite and in (0, 1], got {}",
                options.top_p
            ),
        ));
    }
    if options.temperature > 0.0 {
        if top_k.is_none() && options.top_p < 1.0 {
            return Err(Error::new(
                Status::InvalidArg,
                "sample: topP < 1 requires topK in [1, 64] for positive-temperature Metal sampling",
            ));
        }
        if let Some(top_k) = top_k.filter(|top_k| *top_k > 64) {
            return Err(Error::new(
                Status::InvalidArg,
                format!("sample: topK {top_k} exceeds Metal positive-temperature limit 64"),
            ));
        }
    }
    Ok(SamplingOptions {
        temperature: options.temperature,
        top_k,
        top_p: options.top_p,
        seed: non_negative_safe_integer(options.seed, "seed")?,
        counter: non_negative_safe_integer(options.counter, "counter")?,
    })
}

fn dtype_name(dtype: DType) -> &'static str {
    dtype.name()
}

fn get_device(device_ordinal: Option<u32>) -> Device {
    Device::Metal(device_ordinal.unwrap_or(0))
}

#[napi(custom_finalize)]
pub struct NativeTensor {
    pub(crate) slot: std::sync::Arc<LeafSlot>,
    bytes: i64,
    device_ordinal: usize,
}

impl NativeTensor {
    fn wrap(inner: value::Value) -> Self {
        let device_ordinal = inner.device().ordinal() as usize;
        // Buffers cost at least one memory page regardless of logical tensor size.
        // Metal allocates in 4 KB units, and malloc behaves similarly. Without
        // this floor, tiny tensors look free to V8, which delays collection. The
        // backend then cannot reuse pooled buffers because the pool requires
        // strong_count == 1, so memory and per-allocation costs keep growing.
        let bytes = inner.byte_size().max(4096) as i64;
        // Accounting is native-only: every handle that reaches JS is counted
        // here at creation and subtracted in the finalizer/dispose. V8 is
        // told the delta at the next main-thread touchpoint (see sync_v8);
        // no JS-side involvement, so no missed sites and no drift.
        EXTERNAL_MEMORY_BYTES.fetch_add(bytes, Ordering::Relaxed);
        *EXTERNAL_MEMORY_BY_DEVICE
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .entry(device_ordinal)
            .or_default() += bytes;
        Self {
            slot: std::sync::Arc::new(LeafSlot::new(inner)),
            bytes,
            device_ordinal,
        }
    }

    fn val_cloned(&self) -> Result<value::Value> {
        self.slot
            .get()
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))
    }

    fn release_accounting(&mut self) {
        if self.bytes != 0 {
            EXTERNAL_MEMORY_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
            let mut by_device = EXTERNAL_MEMORY_BY_DEVICE
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if let Some(bytes) = by_device.get_mut(&self.device_ordinal) {
                *bytes -= self.bytes;
                if *bytes == 0 {
                    by_device.remove(&self.device_ordinal);
                }
            }
            self.bytes = 0;
        }
    }
}

impl Drop for NativeTensor {
    fn drop(&mut self) {
        self.release_accounting();
    }
}

// Native bytes currently retained by JS-reachable tensors.
static EXTERNAL_MEMORY_BYTES: AtomicI64 = AtomicI64::new(0);
static EXTERNAL_MEMORY_BY_DEVICE: LazyLock<Mutex<HashMap<usize, i64>>> =
    LazyLock::new(Default::default);
// What V8 has been told so far (adjust_external_memory is main-thread only).
static V8_REPORTED: AtomicI64 = AtomicI64::new(0);

fn sync_v8(env: &Env) {
    let accounted = EXTERNAL_MEMORY_BYTES.load(Ordering::Relaxed);
    let reported = V8_REPORTED.swap(accounted, Ordering::Relaxed);
    let delta = accounted - reported;
    if delta != 0 {
        let _ = env.adjust_external_memory(delta);
    }
}

// V8's GC only sees the small JS handle; report the native buffer size so
// collection is scheduled with knowledge of native memory pressure.
impl ObjectFinalize for NativeTensor {
    fn finalize(mut self, env: Env) -> Result<()> {
        self.release_accounting();
        sync_v8(&env);
        Ok(())
    }
}

#[napi]
pub struct CancellationToken {
    state: Arc<CancellationState>,
    notify: Arc<tokio::sync::Notify>,
}

#[napi]
impl CancellationToken {
    #[napi(constructor)]
    pub fn new(env: Env) -> Self {
        // Every async evaluation allocates a token on the main thread just
        // before spawning; syncing V8's external-memory view here keeps the
        // GC's pressure signal within one evaluation of reality.
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
    /// Releases the tensor's buffer before garbage collection. Later use of the
    /// handle or a lazy graph built from it returns a typed error.
    #[napi]
    pub fn clear(&mut self, env: Env) -> Result<()> {
        if self.slot.clear() {
            self.release_accounting();
            sync_v8(&env);
        }
        Ok(())
    }

    #[napi(getter)]
    pub fn shape(&self) -> Result<Vec<u32>> {
        Ok(self
            .val_cloned()?
            .shape()
            .iter()
            .map(|&d| d as u32)
            .collect())
    }

    #[napi(getter)]
    pub fn dtype(&self) -> Result<String> {
        Ok(self.val_cloned()?.dtype().name().to_string())
    }

    #[napi(getter)]
    pub fn device(&self) -> Result<String> {
        Ok(self.val_cloned()?.device().name().to_string())
    }

    #[napi(ts_return_type = "Promise<ArrayBuffer>")]
    pub async fn readback(&self, token: Option<&CancellationToken>) -> Result<Readback> {
        let inner = self.val_cloned()?;
        run_compute_on(self.device_ordinal, token, move |cancelled, _state| {
            if cancelled.load(Ordering::Acquire) {
                return Err(Error::new(
                    Status::Cancelled,
                    "operation aborted".to_string(),
                ));
            }
            let value = readback_blocking(&inner)?;
            if cancelled.load(Ordering::Acquire) {
                return Err(Error::new(
                    Status::Cancelled,
                    "operation aborted".to_string(),
                ));
            }
            Ok(value)
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
        let options = sampling_options(NativeSamplingOptions {
            temperature,
            top_k,
            top_p,
            seed,
            counter,
        })?;
        let inner = self.val_cloned()?;
        run_compute_on(
            self.device_ordinal,
            cancellation_token,
            move |cancelled, _state| sample_blocking(&inner, options, cancelled),
        )
        .await
    }
}

fn sample_blocking(
    inner: &value::Value,
    options: SamplingOptions,
    cancelled: &effect_torch_runtime::CancellationFlag,
) -> Result<u32> {
    let tensor = inner.as_metal().map_err(to_napi_err)?;
    if cancelled.is_cancelled() {
        return Err(Error::new(Status::Cancelled, "operation aborted"));
    }
    let result = crate::sampling::sample(tensor, options).map_err(to_napi_err)?;
    runtime::metal::device::MetalDevice::get()
        .synchronize_buffer(&result.buffer)
        .map_err(to_napi_err)?;
    if cancelled.is_cancelled() {
        return Err(Error::new(Status::Cancelled, "operation aborted"));
    }
    let offset = result.layout.offset();
    let values = result.buffer.contents_ptr().cast::<u32>();
    // SAFETY: `sampling::sample` returns an eight-byte shared u32 allocation,
    // and synchronization above completed the only GPU writer.
    let (status, token) = unsafe { (*values.add(offset), *values.add(offset + 1)) };
    match status {
        0 => Ok(token),
        crate::sampling::STATUS_NONFINITE => Err(Error::new(
            Status::InvalidArg,
            format!("sample: logit {token} is not finite"),
        )),
        _ => Err(Error::new(
            Status::GenericFailure,
            format!("sample: GPU sampler returned unknown status {status}"),
        )),
    }
}

fn readback_blocking(inner: &value::Value) -> Result<Readback> {
    let tensor = inner.as_metal().map_err(to_napi_err)?;
    runtime::metal::device::MetalDevice::get()
        .synchronize_buffer(&tensor.buffer)
        .map_err(to_napi_err)?;
    let base = tensor.buffer.contents_ptr() as *const u8;
    let elem_size = tensor.dtype.size_in_bytes();
    let count = tensor.numel();
    let byte_len = count * elem_size;
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
        ($type:ty) => {{
            let source = base.cast::<$type>();
            (0..count)
                // SAFETY: the producing command stream was synchronized above;
                // every logical offset comes from the tensor's validated
                // shape/strides and indexes initialized shared-buffer storage.
                .map(|index| unsafe { *source.add(logical_offset(index)) })
                .collect::<Vec<$type>>()
        }};
    }
    let owned = match tensor.dtype {
        runtime::dtype::DType::F16 => Some(vec_to_bytes(
            gather!(u16)
                .into_iter()
                .map(|bits| half::f16::from_bits(bits).to_f32())
                .collect::<Vec<_>>(),
        )),
        runtime::dtype::DType::BF16 => Some(vec_to_bytes(
            gather!(u16)
                .into_iter()
                .map(|bits| half::bf16::from_bits(bits).to_f32())
                .collect::<Vec<_>>(),
        )),
        _ if !tensor.layout.is_contiguous() => Some(match tensor.dtype {
            runtime::dtype::DType::F32 => vec_to_bytes(gather!(f32)),
            runtime::dtype::DType::F64 => vec_to_bytes(gather!(f64)),
            runtime::dtype::DType::I64 => vec_to_bytes(gather!(i64)),
            runtime::dtype::DType::U8 => vec_to_bytes(gather!(u8)),
            runtime::dtype::DType::U32 => vec_to_bytes(gather!(u32)),
            dtype => {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!("readback not implemented for dtype: {}", dtype.name()),
                ))
            }
        }),
        _ => None,
    };
    if let Some((_, ptr, len, cap)) = owned {
        return Ok(Readback {
            data: ptr,
            byte_len: len,
            hint: Some(FinalizeHint::Owned { ptr, len, cap }),
        });
    }
    let offset = tensor.layout.offset() * elem_size;
    // Small readbacks are copied so a short-lived scalar/metadata ArrayBuffer
    // cannot pin an entire pooled output segment until GC.
    if !base.is_null() && byte_len <= 4096 {
        // SAFETY: synchronization completed all GPU writes, and the contiguous
        // tensor view guarantees `offset..offset + byte_len` lies inside the
        // retained shared buffer. A zero length does not dereference `base`.
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
                    tensor: inner.clone(),
                    addr,
                }),
            });
        }
    }
    // SAFETY: fallback copy after zero-copy registration failed; the same
    // synchronized contiguous-view bounds as the small-copy branch apply.
    let bytes = unsafe { std::slice::from_raw_parts(base.add(offset), byte_len) }.to_vec();
    let (_, ptr, len, cap) = vec_to_bytes(bytes);
    Ok(Readback {
        data: ptr,
        byte_len: len,
        hint: Some(FinalizeHint::Owned { ptr, len, cap }),
    })
}

struct ConstantCache {
    map: HashMap<(u64, DType, u32), Arc<Node>>,
    order: std::collections::VecDeque<(u64, DType, u32)>,
}

static CONSTANT_CACHE: LazyLock<Mutex<ConstantCache>> = LazyLock::new(|| {
    Mutex::new(ConstantCache {
        map: HashMap::new(),
        order: std::collections::VecDeque::new(),
    })
});

const CONSTANT_CACHE_LIMIT: usize = 4096;

fn cached_constant(
    value: f64,
    dtype: DType,
    device: Device,
) -> std::result::Result<Arc<Node>, String> {
    let key = (value.to_bits(), dtype, device.ordinal());
    let mut cache = CONSTANT_CACHE.lock().unwrap();
    if let Some(node) = cache.map.get(&key) {
        return Ok(node.clone());
    }
    let node = Node::new(NodeKind::Full {
        shape: vec![],
        value,
        dtype,
        device,
    })?;
    if cache.order.len() >= CONSTANT_CACHE_LIMIT {
        if let Some(old) = cache.order.pop_front() {
            cache.map.remove(&old);
        }
    }
    cache.map.insert(key, node.clone());
    cache.order.push_back(key);
    Ok(node)
}

// RFC 0016 phase 2 chunks cross_entropy(Linear(x, w, b)) when an LM head
// produces a large logits tensor. Graph construction rewrites it into Sum
// cross-entropies wrapped in Checkpoints. Each chunk's logits live only until
// use and are recomputed during backward, instead of retaining the full
// [rows, vocab] tensor. The runtime combines chunk sums in f32 and divides by
// the active count, matching Mean reduction closely. Model code does not
// change, and the rewrite works across backends.
const CHUNKED_CE_MIN_LOGITS: usize = 1 << 28;
const CHUNKED_CE_CHUNK_LOGITS: usize = 1 << 26;
const CHUNKED_CE_MAX_CHUNKS: usize = 64;

fn chunked_ce_limits() -> (usize, usize) {
    let read = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
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
    let (min_logits, chunk_logits) = chunked_ce_limits();
    chunked_head_ce_with(logits, target, ignore_index, min_logits, chunk_logits)
}

fn chunked_head_ce_with(
    logits: &Arc<Node>,
    target: &Arc<Node>,
    ignore_index: i64,
    min_logits: usize,
    chunk_logits: usize,
) -> std::result::Result<Arc<Node>, String> {
    // Validate with the exact unchunked semantics first so error messages
    // are identical whether or not the rewrite fires.
    let plain = Node::new(NodeKind::CrossEntropy {
        logits: logits.clone(),
        target: target.clone(),
        ignore_index,
        reduction: CeReduction::Mean,
    })?;
    let NodeKind::Linear { x, weight, bias } = &logits.kind else {
        return Ok(plain);
    };
    let (_k_dim, vocab) = (weight.shape[0], weight.shape[1]);
    let rank = x.shape.len();
    if rank < 2 {
        return Ok(plain);
    }
    let rows: usize = x.shape[..rank - 1].iter().product();
    if rows < 2 {
        return Ok(plain);
    }
    let numel = rows.saturating_mul(vocab);
    if numel < min_logits {
        return Ok(plain);
    }
    let chunks = (numel / chunk_logits)
        .clamp(2, CHUNKED_CE_MAX_CHUNKS)
        .min(rows);
    if chunks < 2 {
        return Ok(plain);
    }
    // The semantic node: evaluation runs the chunk loop natively, so the
    // [rows, vocab] logits never materialize whole, and the closed-form
    // backward holds one chunk of grad-logits workspace at a time (the
    // graph-chain version retained every chunk's workspace until the
    // head-parameter roots ran).
    Node::new(NodeKind::ChunkedHeadCe {
        x: x.clone(),
        weight: weight.clone(),
        bias: bias.clone(),
        target: target.clone(),
        ignore_index,
    })
}

#[napi]
pub struct LazyTensor {
    node: Arc<Node>,
}

/// One named exposure discovered in a lazy graph: the name and the wrapped
/// tensor handle (the exposure node's child).
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
        self.node.shape.iter().map(|&d| d as u32).collect()
    }

    #[napi(getter)]
    pub fn dtype(&self) -> String {
        dtype_name(self.node.dtype).to_string()
    }

    #[napi]
    pub fn metadata(&self) -> (Vec<u32>, String) {
        (self.shape(), self.dtype())
    }

    #[napi(factory)]
    pub fn zeros(
        shape: Vec<u32>,
        dtype: Option<NativeDType>,
        device_ordinal: Option<u32>,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Zeros {
            shape: shape.iter().map(|&d| d as usize).collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(device_ordinal),
        }))
    }

    #[napi(factory)]
    pub fn ones(
        shape: Vec<u32>,
        dtype: Option<NativeDType>,
        device_ordinal: Option<u32>,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Ones {
            shape: shape.iter().map(|&d| d as usize).collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(device_ordinal),
        }))
    }

    #[napi(factory)]
    pub fn full(
        shape: Vec<u32>,
        value: f64,
        dtype: Option<NativeDType>,
        device_ordinal: Option<u32>,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Full {
            shape: shape.iter().map(|&d| d as usize).collect(),
            value,
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(device_ordinal),
        }))
    }

    #[napi(factory)]
    pub fn randn(
        shape: Vec<u32>,
        dtype: Option<NativeDType>,
        device_ordinal: Option<u32>,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Randn {
            shape: shape.iter().map(|&d| d as usize).collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(device_ordinal),
        }))
    }

    #[napi(factory)]
    pub fn uniform(
        shape: Vec<u32>,
        lo: f64,
        hi: f64,
        dtype: Option<NativeDType>,
        device_ordinal: Option<u32>,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Uniform {
            lo,
            hi,
            shape: shape.iter().map(|&d| d as usize).collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(device_ordinal),
        }))
    }

    #[napi(factory)]
    pub fn arange(
        start: f64,
        end: f64,
        step: f64,
        dtype: Option<NativeDType>,
        device_ordinal: Option<u32>,
    ) -> Result<Self> {
        if step == 0.0 {
            return Err(Error::new(
                Status::InvalidArg,
                "arange: step must be non-zero".to_string(),
            ));
        }
        lazy_ctor!(Node::new(NodeKind::Arange {
            start,
            end,
            step,
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(device_ordinal),
        }))
    }

    #[napi(factory)]
    pub fn eye(n: u32, dtype: Option<NativeDType>, device_ordinal: Option<u32>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Eye {
            n: n as usize,
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(device_ordinal),
        }))
    }

    // A shared 0-d constant: the same (value, dtype, device) triple maps to
    // one graph node forever instead of allocating a fresh node per use.
    // Nodes hold no buffers, so the cache is cheap; it is size-bounded so
    // cold values rotate through. Devices are process singletons, so the
    // device kind is the whole key.
    #[napi(factory)]
    pub fn constant(
        value: f64,
        dtype: Option<NativeDType>,
        device_ordinal: Option<u32>,
    ) -> Result<Self> {
        let device = get_device(device_ordinal);
        let dtype: DType = dtype.unwrap_or(NativeDType::F32).into();
        match cached_constant(value, dtype, device) {
            Ok(node) => Ok(Self { node }),
            Err(message) => Err(Error::new(Status::InvalidArg, message)),
        }
    }

    #[napi(factory)]
    pub fn from_bytes(
        data: Uint8Array,
        shape: Vec<u32>,
        dtype: Option<NativeDType>,
        device_ordinal: Option<u32>,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::FromBytes {
            data: data.to_vec(),
            shape: shape.iter().map(|&d| d as usize).collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(device_ordinal),
        }))
    }

    #[napi(factory)]
    pub fn from_materialized(tensor: &NativeTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Leaf(tensor.slot.clone())))
    }

    // RFC 0008: placeholder leaves. `input` declares one tensor argument of a
    // compiled program; `scalar_input` declares one 0-d runtime scalar (lr,
    // step counts, ...). Both carry their declared signature so the rest of
    // the graph validates shapes at trace time.
    #[napi(factory)]
    pub fn input(
        slot: u32,
        shape: Vec<u32>,
        dtype: Option<NativeDType>,
        device_ordinal: Option<u32>,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Input {
            slot,
            shape: shape.iter().map(|&d| d as usize).collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(device_ordinal),
        }))
    }

    #[napi(factory)]
    pub fn scalar_input(
        slot: u32,
        dtype: Option<NativeDType>,
        device_ordinal: Option<u32>,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::ScalarInput {
            slot,
            dtype: dtype.unwrap_or(NativeDType::F64).into(),
            device: get_device(device_ordinal),
        }))
    }

    #[napi]
    pub fn add(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Add {
            a: self.node.clone(),
            b: other.node.clone(),
        }))
    }

    #[napi]
    pub fn sub(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Sub {
            a: self.node.clone(),
            b: other.node.clone(),
        }))
    }

    #[napi]
    pub fn mul(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Mul {
            a: self.node.clone(),
            b: other.node.clone(),
        }))
    }

    #[napi]
    pub fn div(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Div {
            a: self.node.clone(),
            b: other.node.clone(),
        }))
    }

    #[napi]
    pub fn maximum(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Maximum {
            a: self.node.clone(),
            b: other.node.clone(),
        }))
    }

    #[napi]
    pub fn minimum(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Minimum {
            a: self.node.clone(),
            b: other.node.clone(),
        }))
    }

    #[napi]
    pub fn eq(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Eq {
            a: self.node.clone(),
            b: other.node.clone(),
        }))
    }

    #[napi]
    pub fn gt(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Gt {
            a: self.node.clone(),
            b: other.node.clone(),
        }))
    }

    #[napi]
    pub fn lt(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Lt {
            a: self.node.clone(),
            b: other.node.clone(),
        }))
    }

    #[napi]
    pub fn ge(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Ge {
            a: self.node.clone(),
            b: other.node.clone(),
        }))
    }

    #[napi]
    pub fn le(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Le {
            a: self.node.clone(),
            b: other.node.clone(),
        }))
    }

    #[napi]
    pub fn matmul(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Matmul {
            a: self.node.clone(),
            b: other.node.clone(),
        }))
    }

    #[napi]
    pub fn inverse(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Inverse {
            a: self.node.clone(),
        }))
    }

    #[napi]
    pub fn det(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Det {
            a: self.node.clone(),
        }))
    }

    #[napi]
    pub fn solve(&self, other: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Solve {
            a: self.node.clone(),
            b: other.node.clone(),
        }))
    }

    #[napi]
    pub fn neg(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Neg {
            a: self.node.clone(),
        }))
    }

    #[napi]
    pub fn abs(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Abs {
            a: self.node.clone(),
        }))
    }

    #[napi]
    pub fn sqrt(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Sqrt {
            a: self.node.clone(),
        }))
    }

    #[napi]
    pub fn exp(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Exp {
            a: self.node.clone(),
        }))
    }

    #[napi]
    pub fn tanh(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Tanh {
            a: self.node.clone(),
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
            a: self.node.clone(),
        }))
    }

    #[napi]
    pub fn erf(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Erf {
            a: self.node.clone(),
        }))
    }

    #[napi]
    pub fn floor(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Floor {
            a: self.node.clone(),
        }))
    }

    #[napi]
    pub fn ceil(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Ceil {
            a: self.node.clone(),
        }))
    }

    #[napi]
    pub fn round(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Round {
            a: self.node.clone(),
        }))
    }

    #[napi]
    pub fn sign(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Sign {
            a: self.node.clone(),
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
            dim: dim as usize,
        }))
    }

    #[napi]
    pub fn argmin(&self, dim: u32) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Argmin {
            a: self.node.clone(),
            dim: dim as usize,
        }))
    }

    #[napi]
    pub fn cumsum(&self, dim: u32) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Cumsum {
            a: self.node.clone(),
            dim: dim as usize,
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
        w: &LazyTensor,
        stride: u32,
        padding: u32,
        dilation: u32,
        groups: u32,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Conv1d {
            x: self.node.clone(),
            w: w.node.clone(),
            stride: stride as usize,
            padding: padding as usize,
            dilation: dilation as usize,
            groups: groups as usize,
        }))
    }

    #[napi(js_name = "conv2d")]
    pub fn conv_2d(
        &self,
        w: &LazyTensor,
        stride: u32,
        padding: u32,
        dilation: u32,
        groups: u32,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Conv2d {
            x: self.node.clone(),
            w: w.node.clone(),
            stride: stride as usize,
            padding: padding as usize,
            dilation: dilation as usize,
            groups: groups as usize,
        }))
    }

    #[napi]
    pub fn log(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Log {
            a: self.node.clone(),
        }))
    }

    #[napi]
    pub fn sin(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Sin {
            a: self.node.clone(),
        }))
    }

    #[napi]
    pub fn cos(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Cos {
            a: self.node.clone(),
        }))
    }

    #[napi]
    pub fn pow(&self, exp: f64) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Pow {
            a: self.node.clone(),
            exp,
        }))
    }

    #[napi]
    pub fn cast(&self, dtype: NativeDType) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Cast {
            a: self.node.clone(),
            dtype: dtype.into(),
        }))
    }

    #[napi]
    pub fn sum(&self, dims: Vec<u32>, keepdims: bool) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Sum {
            a: self.node.clone(),
            dims: dims.iter().map(|&d| d as usize).collect(),
            keepdims,
        }))
    }

    #[napi]
    pub fn prod(&self, dims: Vec<u32>, keepdims: bool) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Prod {
            a: self.node.clone(),
            dims: dims.iter().map(|&d| d as usize).collect(),
            keepdims,
        }))
    }

    #[napi]
    pub fn mean(&self, dims: Vec<u32>, keepdims: bool) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Mean {
            a: self.node.clone(),
            dims: dims.iter().map(|&d| d as usize).collect(),
            keepdims,
        }))
    }

    #[napi]
    pub fn max(&self, dims: Vec<u32>, keepdims: bool) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Max {
            a: self.node.clone(),
            dims: dims.iter().map(|&d| d as usize).collect(),
            keepdims,
        }))
    }

    #[napi]
    pub fn min(&self, dims: Vec<u32>, keepdims: bool) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Min {
            a: self.node.clone(),
            dims: dims.iter().map(|&d| d as usize).collect(),
            keepdims,
        }))
    }

    #[napi]
    pub fn reshape(&self, shape: Vec<u32>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Reshape {
            a: self.node.clone(),
            shape: shape.iter().map(|&d| d as usize).collect(),
        }))
    }

    #[napi]
    pub fn permute(&self, dims: Vec<u32>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Permute {
            a: self.node.clone(),
            dims: dims.iter().map(|&d| d as usize).collect(),
        }))
    }

    #[napi]
    pub fn slice(&self, ranges: Vec<Vec<u32>>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Slice {
            a: self.node.clone(),
            ranges: ranges
                .iter()
                .map(|r| (r[0] as usize, r[1] as usize, r[2] as usize))
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
            shape: shape.iter().map(|&d| d as usize).collect(),
        }))
    }

    #[napi]
    pub fn stop_gradient(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::StopGradient {
            a: self.node.clone(),
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
            name,
        }))
    }

    /// Walks the lazy graph reachable from this node and returns every
    /// exposure (name plus the wrapped tensor) in deterministic first-visit
    /// order; duplicate names are a caller error.
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
            stack.extend(node_children(&node.kind));
        }
        Ok(found)
    }

    #[napi]
    pub fn checkpoint(&self) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Checkpoint {
            a: self.node.clone(),
        }))
    }

    #[napi]
    pub fn vmap(&self, x: &LazyTensor, batched_x: &LazyTensor, dim: u32) -> Result<Self> {
        lazy_ctor!(effect_torch_autodiff::vmap(
            &self.node,
            &x.node,
            &batched_x.node,
            dim as usize
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
            index,
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
            index,
        }))
    }
}

#[napi]
pub fn grad(loss: &LazyTensor, wrt: Vec<&LazyTensor>) -> Result<Vec<LazyTensor>> {
    let targets: Vec<Arc<Node>> = wrt.iter().map(|t| t.node.clone()).collect();
    let grads = effect_torch_autodiff::grad(&loss.node, &targets)
        .map_err(|message| Error::new(Status::GenericFailure, message))?;
    Ok(grads.into_iter().map(|node| LazyTensor { node }).collect())
}

#[napi]
pub fn is_available() -> bool {
    objc2::rc::autoreleasepool(|_| runtime::metal::device::is_available())
}

#[napi]
pub fn is_device_available(device_ordinal: u32) -> bool {
    objc2::rc::autoreleasepool(|_| {
        runtime::metal::device::is_ordinal_available(device_ordinal as usize)
    })
}

fn with_device<T>(device_ordinal: usize, operation: impl FnOnce() -> Result<T>) -> Result<T> {
    runtime::metal::device::MetalDevice::with_ordinal(device_ordinal, operation)
        .map_err(to_napi_err)?
}

async fn run_compute<T: Send + 'static>(
    token: Option<&CancellationToken>,
    compute: impl FnOnce(&effect_torch_runtime::CancellationFlag, &CancellationState) -> Result<T>
        + Send
        + 'static,
) -> Result<T> {
    let state = token
        .map(|token| token.state.clone())
        .unwrap_or_else(|| Arc::new(CancellationState::new()));
    let notify = token.map(|token| token.notify.clone());
    let compute = move |flag: &effect_torch_runtime::CancellationFlag,
                        state: &CancellationState| {
        objc2::rc::autoreleasepool(|_| compute(flag, state))
    };
    effect_torch_napi::run_compute(state, notify, compute).await
}

async fn run_compute_on<T: Send + 'static>(
    device_ordinal: usize,
    token: Option<&CancellationToken>,
    compute: impl FnOnce(&effect_torch_runtime::CancellationFlag, &CancellationState) -> Result<T>
        + Send
        + 'static,
) -> Result<T> {
    run_compute(token, move |cancelled, state| {
        with_device(device_ordinal, || compute(cancelled, state))
    })
    .await
}

#[cfg(test)]
fn compile_metal_roots(roots: &[Arc<Node>]) -> err::Res<executable::MetalCompilation> {
    compile_metal_roots_with_options(roots, CompileOptions::from_environment(), None)
}

#[cfg(test)]
fn compile_metal_roots_with_options(
    roots: &[Arc<Node>],
    options: CompileOptions,
    state_schema: Option<KvStateSchema>,
) -> err::Res<executable::MetalCompilation> {
    let mut request = ProgramRequest::from_roots(roots.to_vec(), options);
    if let Some(schema) = state_schema {
        request = request.with_state_cursor(StateCursorSlot::new(
            schema.cursor_slot,
            schema.cursor_tensor,
        ));
    }
    let program = request.prepare()?;
    let generated_bindings = executable::load_generated_bindings(&program.index)?;
    compile_prepared_metal(&program, &generated_bindings, state_schema)
}

fn compile_prepared_metal(
    program: &PreparedProgram,
    generated_bindings: &[value::Value],
    state_schema: Option<KvStateSchema>,
) -> err::Res<executable::MetalCompilation> {
    executable::compile_prepared_with_state(program, generated_bindings, state_schema)
}

// RFC 0010 defines paged KV inference. A `NativeKvPool` is a fixed-capacity
// store of key and value rows for each attention layer, allocated once per
// inference artifact. A `NativeKvSequence` is a block table and cursor over
// the pool, analogous to a process over OS pages. Stateful `compile` rewrites
// a traced forward graph for generation. Causal Sdpa becomes KvAttention, which
// scatters new tokens into the pool and attends over cached context.
// PositionEmbedding becomes a cursor-offset gather. Compilation then freezes
// the result. The frozen graph remains a pure function of its inputs. The run's
// KV context carries the pool and sequence. Parallel runs write disjoint blocks,
// and a sequence's run lock serializes its runs.

// Chained FNV-1a over token blocks. The hash of block i covers the prefix
// through block i. Equal hashes therefore mean equal tokens at equal absolute
// positions, so RoPE produces cached rows identical to a recompute.
const HASH_SEED: u64 = 0xcbf2_9ce4_8422_2325;
const HASH_PRIME: u64 = 0x0000_0100_0000_01b3;

fn chain_hash(prev: u64, tokens: &[u32]) -> u64 {
    let mut hash = prev;
    for token in tokens {
        for byte in token.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(HASH_PRIME);
        }
    }
    hash
}

// Blocks have a reference count and gain a chained content hash once fully
// written. Content hashes allow sharing across live sequences. A prompt can
// reference a resident prefix held by another sequence or the cache instead of
// recomputing it. Unreferenced hashed blocks form an LRU cache reclaimed under
// pressure.
struct BlockStore {
    free: Vec<u32>,
    refcounts: Vec<u32>,
    // Content hash of each completed block; a block is hashed when its
    // last row is written, so partial tail blocks are unhashable.
    hashes: Vec<Option<u64>>,
    // Every completed block by content hash, owned or not (duplicates
    // arise when two sequences compute the same prefix concurrently).
    // The cached subset is exactly the entries with refcount 0.
    by_hash: HashMap<u64, Vec<u32>>,
    // LRU order of cached (unreferenced) blocks, most recent at the
    // back; entries go stale when the block is taken or evicted and
    // are skipped.
    lru: VecDeque<u32>,
    snapshots: HashMap<u64, Arc<RecurrentSnapshot>>,
}

impl BlockStore {
    fn new(num_blocks: usize) -> Self {
        Self {
            free: (0..num_blocks as u32).rev().collect(),
            refcounts: vec![0; num_blocks],
            hashes: vec![None; num_blocks],
            by_hash: HashMap::new(),
            lru: VecDeque::new(),
            snapshots: HashMap::new(),
        }
    }

    // A block is reclaimable cache content only while unreferenced,
    // hashed, and still listed under its hash.
    fn is_cached(&self, block: u32) -> bool {
        self.refcounts[block as usize] == 0
            && match self.hashes[block as usize] {
                Some(hash) => self
                    .by_hash
                    .get(&hash)
                    .is_some_and(|ids| ids.contains(&block)),
                None => false,
            }
    }

    fn uncache(&mut self, block: u32, hash: u64) {
        if let Some(ids) = self.by_hash.get_mut(&hash) {
            if let Some(at) = ids.iter().position(|&id| id == block) {
                ids.swap_remove(at);
            }
            if ids.is_empty() {
                self.by_hash.remove(&hash);
                self.snapshots.remove(&hash);
            }
        }
    }

    fn unref(&mut self, block: u32) {
        let count = &mut self.refcounts[block as usize];
        *count = count.saturating_sub(1);
        if *count == 0 {
            match self.hashes[block as usize] {
                Some(_) => self.lru.push_back(block),
                None => self.free.push(block),
            }
        }
    }

    fn cached(&self) -> usize {
        self.by_hash
            .values()
            .map(|ids| {
                ids.iter()
                    .filter(|&&id| self.refcounts[id as usize] == 0)
                    .count()
            })
            .sum()
    }
}

struct RecurrentSnapshot {
    kda: Vec<Box<[f32]>>,
    conv: Vec<Box<[f32]>>,
}

fn copy_f32_state(tensor: &runtime::metal::run::MetalTensor) -> Option<Box<[f32]>> {
    if tensor.dtype != DType::F32 || !tensor.layout.is_contiguous() {
        return None;
    }
    // SAFETY: dtype/contiguity were checked above; the layout offset addresses
    // the first f32 of a retained shared buffer and `numel` is its logical
    // contiguous extent. State capture is called only after producing work is
    // synchronized by the decode execution path.
    let ptr = unsafe {
        tensor
            .buffer
            .contents_ptr()
            .cast::<f32>()
            .add(tensor.layout.offset())
    };
    // SAFETY: `ptr` and the `numel` extent were established above.
    Some(unsafe { std::slice::from_raw_parts(ptr, tensor.numel()) }.into())
}

fn write_f32_state(tensor: &runtime::metal::run::MetalTensor, data: &[f32]) -> err::Res<()> {
    if tensor.dtype != DType::F32 || !tensor.layout.is_contiguous() || tensor.numel() != data.len()
    {
        return Err("recurrent state destination does not match its snapshot".to_string());
    }
    // SAFETY: validation proves equal f32 element counts and contiguous
    // destination storage. Snapshot storage is distinct host memory, so the
    // source and destination ranges cannot overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            tensor
                .buffer
                .contents_ptr()
                .cast::<f32>()
                .add(tensor.layout.offset()),
            data.len(),
        );
    }
    Ok(())
}

impl RecurrentSnapshot {
    fn capture(state: &SeqState) -> Option<Self> {
        Some(Self {
            kda: state
                .kda_states
                .iter()
                .map(copy_f32_state)
                .collect::<Option<Vec<_>>>()?,
            conv: state
                .conv_states
                .iter()
                .map(copy_f32_state)
                .collect::<Option<Vec<_>>>()?,
        })
    }

    fn restore_into(&self, state: &mut SeqState) -> err::Res<()> {
        if self.kda.len() != state.kda_states.len() || self.conv.len() != state.conv_states.len() {
            return Err("recurrent snapshot geometry does not match the sequence".to_string());
        }
        for (tensor, data) in state.kda_states.iter().zip(&self.kda) {
            write_f32_state(tensor, data)?;
        }
        for (tensor, data) in state.conv_states.iter().zip(&self.conv) {
            write_f32_state(tensor, data)?;
        }
        Ok(())
    }
}

enum PoolSlab {
    NativeMetal(runtime::metal::run::MetalTensor),
}

impl PoolSlab {
    fn dtype(&self) -> DType {
        match self {
            PoolSlab::NativeMetal(t) => t.dtype,
        }
    }

    fn metal(&self) -> err::Res<&runtime::metal::run::MetalTensor> {
        match self {
            PoolSlab::NativeMetal(t) => Ok(t),
        }
    }
}

struct PoolInner {
    device_ordinal: usize,
    // Per layer, flat [max_tokens, kv_heads, head_dim] slabs; block b
    // occupies rows b*block_size..(b+1)*block_size. Slab dtype u8 means
    // int8-quantized storage (RFC 0012 storage tier): rows are
    // symmetric-quantized with a per-(token, head) absmax scale held in
    // `scales` contains two slabs per layer, k then v, when data slabs are
    // u8. It is empty otherwise.
    k: Vec<PoolSlab>,
    v: Vec<PoolSlab>,
    scales: Vec<PoolSlab>,
    kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_tokens: usize,
    dtype: DType,
    kda: KdaGeometry,
    conv: ConvGeometry,
    padding_kda_states: Vec<runtime::metal::run::MetalTensor>,
    padding_conv_states: Vec<runtime::metal::run::MetalTensor>,
    blocks: Mutex<BlockStore>,
}

impl PoolInner {
    fn ref_blocks(&self, blocks: &[u32]) -> err::Res<()> {
        let mut store = self
            .blocks
            .lock()
            .map_err(|error| format!("kv block store lock poisoned: {error}"))?;
        for (index, &block) in blocks.iter().enumerate() {
            let result = store
                .refcounts
                .get_mut(block as usize)
                .ok_or_else(|| "kv block reference is out of range".to_string())
                .and_then(|count| {
                    *count = count
                        .checked_add(1)
                        .ok_or_else(|| "kv block reference count exhausted".to_string())?;
                    Ok(())
                });
            if let Err(error) = result {
                for &retained in &blocks[..index] {
                    store.unref(retained);
                }
                return Err(error);
            }
        }
        Ok(())
    }

    fn live_blocks(&self) -> u64 {
        self.blocks
            .lock()
            .map(|store| store.refcounts.iter().filter(|count| **count > 0).count() as u64)
            .unwrap_or(0)
    }

    // Takes a fresh block with refcount 1: free list first, then LRU
    // eviction of unreferenced cached blocks.
    #[cfg(test)]
    fn alloc_block(&self) -> Option<u32> {
        self.alloc_block_with_cache_eviction(true)
    }

    fn alloc_block_with_cache_eviction(&self, evict_cache: bool) -> Option<u32> {
        let mut store = self.blocks.lock().ok()?;
        if let Some(block) = store.free.pop() {
            store.refcounts[block as usize] = 1;
            store.hashes[block as usize] = None;
            return Some(block);
        }
        if !evict_cache {
            return None;
        }
        while let Some(candidate) = store.lru.pop_front() {
            if !store.is_cached(candidate) {
                continue;
            }
            let hash = store.hashes[candidate as usize].expect("cached implies hashed");
            store.uncache(candidate, hash);
            store.hashes[candidate as usize] = None;
            store.refcounts[candidate as usize] = 1;
            return Some(candidate);
        }
        None
    }

    // Takes a reference to a resident block by content hash (a
    // prefix-cache hit), whether it is held by a live sequence or
    // unreferenced in the cache.
    fn take_block(&self, hash: u64) -> Option<u32> {
        let mut store = self.blocks.lock().ok()?;
        let block = *store.by_hash.get(&hash)?.first()?;
        store.refcounts[block as usize] += 1;
        Some(block)
    }

    // Drops a reference: the last one makes the block cache content
    // (hashed, reclaimable) or returns it to the free list.
    fn unref_block(&self, block: u32) {
        if let Ok(mut store) = self.blocks.lock() {
            store.unref(block);
        }
    }

    fn unref_blocks(&self, blocks: &[u32]) {
        if let Ok(mut store) = self.blocks.lock() {
            for &block in blocks {
                store.unref(block);
            }
        }
    }

    fn take_recurrent_prefix(
        &self,
        boundary_hashes: &[u64],
    ) -> Option<(Vec<u32>, u64, Arc<RecurrentSnapshot>)> {
        let mut store = self.blocks.lock().ok()?;
        let mut taken: Vec<u32> = Vec::new();
        let mut deepest: Option<(usize, Arc<RecurrentSnapshot>)> = None;
        for (index, &hash) in boundary_hashes.iter().enumerate() {
            let Some(&block) = store.by_hash.get(&hash).and_then(|ids| ids.first()) else {
                break;
            };
            store.refcounts[block as usize] += 1;
            taken.push(block);
            if let Some(snapshot) = store.snapshots.get(&hash) {
                deepest = Some((index, Arc::clone(snapshot)));
            }
        }
        let Some((index, snapshot)) = deepest else {
            for block in taken.drain(..) {
                store.unref(block);
            }
            return None;
        };
        for block in taken.drain(index + 1..) {
            store.unref(block);
        }
        Some((taken, boundary_hashes[index], snapshot))
    }

    fn publish_recurrent_snapshot(&self, state: &SeqState) {
        if self.k.is_empty() || (self.kda.layers == 0 && self.conv.layers == 0) {
            return;
        }
        if state.cursor == 0 || state.cursor % self.block_size != 0 {
            return;
        }
        if state.kda_states.len() != self.kda.layers || state.conv_states.len() != self.conv.layers
        {
            return;
        }
        let Some(snapshot) = RecurrentSnapshot::capture(state) else {
            return;
        };
        if let Ok(mut store) = self.blocks.lock() {
            if store.by_hash.contains_key(&state.last_hash) {
                store.snapshots.insert(state.last_hash, Arc::new(snapshot));
            }
        }
    }

    fn set_hash(&self, block: u32, hash: u64) {
        if let Ok(mut store) = self.blocks.lock() {
            store.hashes[block as usize] = Some(hash);
            store.by_hash.entry(hash).or_default().push(block);
        }
    }

    // Blocks available for new content: free plus reclaimable cached.
    fn available(&self) -> usize {
        self.blocks
            .lock()
            .map(|store| store.free.len() + store.cached())
            .unwrap_or(0)
    }

    fn cached_count(&self) -> usize {
        self.blocks.lock().map(|store| store.cached()).unwrap_or(0)
    }
}

impl SeqState {
    // Records a run's real tokens, hashing each block whose final row
    // they complete. Runs only append, so a single rolling hash chains
    // correctly across prefill chunks and decode steps. Called with the
    // cursor still at its pre-run value.
    fn note_tokens(&mut self, pool: &PoolInner, tokens: &[u32]) {
        self.note_tokens_with(tokens, pool.block_size, |block, hash| {
            pool.set_hash(block, hash)
        });
    }

    // Speculative roots update their rolling token metadata but do not make
    // provisional or rejected blocks visible through the prefix cache.
    fn note_tokens_provisional(&mut self, tokens: &[u32], block_size: usize) {
        self.note_tokens_with(tokens, block_size, |_, _| {});
        debug_assert!(self.pending.len() < block_size);
    }

    fn note_tokens_with(
        &mut self,
        tokens: &[u32],
        block_size: usize,
        mut publish: impl FnMut(u32, u64),
    ) {
        for (i, &token) in tokens.iter().enumerate() {
            self.pending.push(token);
            if self.pending.len() == block_size {
                let hash = chain_hash(self.last_hash, &self.pending);
                self.last_hash = hash;
                self.pending.clear();
                // The block holding this token completed; it was
                // allocated by the run that wrote its first row.
                let block_index = (self.cursor + i) / block_size;
                if let Some(&block) = block_index
                    .checked_sub(self.head)
                    .and_then(|index| self.blocks.get(index))
                {
                    publish(block, hash);
                }
            }
        }
    }
}

// RFC 0018: uniform KDA layer geometry of a decode program.
fn persistent_f32_zeros(shape: Vec<usize>) -> err::Res<runtime::metal::run::MetalTensor> {
    let elements = shape
        .iter()
        .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
        .ok_or_else(|| "decode recurrent state size overflow".to_string())?;
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "decode recurrent state byte size overflow".to_string())?;
    let buffer = device::MetalDevice::get().alloc_raw_checked(bytes)?;
    // SAFETY: `alloc_raw_checked(bytes)` returned writable shared storage of
    // at least `bytes`; no tensor view has been published yet, so zeroing the
    // full allocation through its host pointer is exclusive.
    unsafe {
        std::ptr::write_bytes(buffer.contents_ptr().cast::<u8>(), 0, bytes);
    }
    Ok(runtime::metal::run::MetalTensor {
        buffer,
        layout: runtime::layout::Layout::contiguous(shape),
        dtype: DType::F32,
    })
}

fn prepare_recurrent_states(
    state: &mut SeqState,
    kda: KdaGeometry,
    conv: ConvGeometry,
) -> err::Res<()> {
    while state.kda_states.len() < kda.layers {
        state.kda_states.push(persistent_f32_zeros(vec![
            kda.heads,
            kda.head_dim,
            kda.value_dim,
        ])?);
    }
    while state.conv_states.len() < conv.layers {
        state.conv_states.push(persistent_f32_zeros(vec![
            conv.kernel.saturating_sub(1),
            conv.channels,
        ])?);
    }
    Ok(())
}

fn sequence_state(pool: &PoolInner, padding: bool) -> err::Res<SeqState> {
    let mut state = SeqState {
        blocks: Vec::with_capacity(pool.max_tokens / pool.block_size),
        head: 0,
        cursor: 0,
        advance: 0,
        last_hash: HASH_SEED,
        pending: Vec::new(),
        kda_states: if padding {
            pool.padding_kda_states.clone()
        } else {
            Vec::new()
        },
        conv_states: if padding {
            pool.padding_conv_states.clone()
        } else {
            Vec::new()
        },
    };
    if !padding {
        prepare_recurrent_states(&mut state, pool.kda, pool.conv)?;
    }
    Ok(state)
}

#[derive(Clone, Copy)]
struct SequenceCheckpoint {
    blocks: usize,
    advance: usize,
    kda_states: usize,
    conv_states: usize,
}

fn rollback_sequence_setup(
    pool: &PoolInner,
    slots: &[Arc<Mutex<SeqState>>],
    checkpoints: &[SequenceCheckpoint],
) -> err::Res<()> {
    if slots.len() != checkpoints.len() {
        return Err("kv rollback: sequence checkpoint count mismatch".to_string());
    }
    for (slot, checkpoint) in slots.iter().zip(checkpoints) {
        let blocks = {
            let mut state = slot
                .lock()
                .map_err(|error| format!("kv rollback: sequence lock poisoned: {error}"))?;
            state.advance = checkpoint.advance;
            state.kda_states.truncate(checkpoint.kda_states);
            state.conv_states.truncate(checkpoint.conv_states);
            state.blocks.split_off(checkpoint.blocks)
        };
        for block in blocks {
            pool.unref_block(block);
        }
    }
    Ok(())
}

pub(crate) struct KvContext {
    pool: Arc<PoolInner>,
    // Compact request-order sequence state. `lanes` maps each entry to
    // its fixed physical batch row; inactive rows have no sequence.
    pub(crate) slots: Vec<Arc<Mutex<SeqState>>>,
    schema: KvStateSchema,
    tokens: Vec<Vec<u32>>,
    lanes: Vec<usize>,
    /// Packed graph row to compact request index. `None` rows are inert pads.
    packed_rows: Option<Vec<Option<usize>>>,
    /// Explicit absolute position for each packed graph row.
    packed_positions: Option<Vec<usize>>,
    publish_hashes: bool,
    state_only: bool,
}

impl MetalDecodeContext for KvContext {
    fn schema(&self) -> &KvStateSchema {
        &self.schema
    }

    fn slots(&self) -> &[Arc<Mutex<SeqState>>] {
        &self.slots
    }

    fn active_lane(&self, request_index: usize) -> usize {
        self.lanes[request_index]
    }

    fn physical_slot(&self, lane: usize) -> Option<&Arc<Mutex<SeqState>>> {
        self.lanes
            .iter()
            .position(|physical| *physical == lane)
            .and_then(|request| self.slots.get(request))
    }

    fn active_batch(&self) -> usize {
        self.slots.len()
    }

    fn position_offsets(&self) -> err::Res<Vec<usize>> {
        if let Some(positions) = &self.packed_positions {
            return Ok(positions.clone());
        }
        let mut offsets = vec![0; self.schema.graph_batch];
        for (request, slot) in self.slots.iter().enumerate() {
            offsets[self.lanes[request]] = slot
                .lock()
                .map_err(|error| format!("decode position: sequence lock poisoned: {error}"))?
                .cursor;
        }
        Ok(offsets)
    }

    fn prepare_state(&self, cursor: &runtime::metal::run::MetalTensor) -> err::Res<()> {
        let shape = if self.schema.cursor_tensor {
            vec![self.schema.graph_batch]
        } else {
            Vec::new()
        };
        cursor.validate_destination("decode cursor", &shape, DType::I64)?;
        let count = if self.schema.cursor_tensor {
            self.schema.graph_batch
        } else {
            1
        };
        // Inactive physical lanes have a zero cursor.
        unsafe {
            std::ptr::write_bytes(
                cursor
                    .buffer
                    .contents_ptr()
                    .cast::<u8>()
                    .add(cursor.layout.offset() * DType::I64.size_in_bytes()),
                0,
                count * DType::I64.size_in_bytes(),
            );
        }
        let positions = self.position_offsets()?;
        for (index, value) in positions.into_iter().enumerate() {
            // SAFETY: `validate_destination` proved an i64 contiguous cursor
            // buffer with enough elements for the fixed schema; each index is
            // below `count` and slots are written once.
            unsafe {
                cursor
                    .buffer
                    .contents_ptr()
                    .cast::<i64>()
                    .add(cursor.layout.offset() + index)
                    .write(value as i64);
            }
        }
        Ok(())
    }

    fn prepare_kv_attention(
        &self,
        layer: u32,
        plan: &executable::KvAttentionPlan,
        staging: &[runtime::metal::run::MetalTensor],
    ) -> err::Res<()> {
        prepare_kv_attention(self, layer, plan, staging)
    }

    fn kv_attention_into(
        &self,
        layer: u32,
        q: &runtime::metal::run::MetalTensor,
        k: &runtime::metal::run::MetalTensor,
        v: &runtime::metal::run::MetalTensor,
        scale: f64,
        window: Option<usize>,
        mode: KvAttentionMode,
        output: &runtime::metal::run::MetalTensor,
        staging: &[runtime::metal::run::MetalTensor],
        scratch: &[runtime::metal::run::MetalTensor],
    ) -> err::Res<()> {
        kv_attention_into(
            self, layer, q, k, v, scale, window, mode, output, staging, scratch,
        )
    }

    fn evict_before(&self, state: &mut SeqState, start: usize) {
        kv_evict(&self.pool, state, start);
    }

    fn commit_slot(&self, index: usize, state: &mut SeqState) {
        if self.publish_hashes {
            state.note_tokens(&self.pool, &self.tokens[index]);
        } else {
            state.note_tokens_provisional(&self.tokens[index], self.pool.block_size);
        }
        state.cursor += state.advance;
        state.advance = 0;
        if self.publish_hashes {
            self.pool.publish_recurrent_snapshot(state);
        }
    }
}

pub(crate) fn prepare_kv_attention(
    kv: &KvContext,
    layer: u32,
    plan: &executable::KvAttentionPlan,
    staging: &[runtime::metal::run::MetalTensor],
) -> err::Res<()> {
    if staging.len() != 5 {
        return Err(format!(
            "kv attention: expected five planned staging buffers, got {}",
            staging.len()
        ));
    }
    let schema = kv.schema;
    if plan.batch != schema.graph_batch
        || plan.kv_heads != schema.kv_heads
        || plan.head_dim != schema.head_dim
    {
        return Err(format!(
            "kv attention: fixed plan does not match the bound state schema"
        ));
    }
    let expected_shapes = [
        vec![schema.graph_batch, schema.max_tokens / schema.block_size],
        vec![schema.graph_batch],
        vec![schema.graph_batch],
        vec![schema.graph_batch],
        vec![schema.graph_batch],
    ];
    for (tensor, shape) in staging.iter().zip(&expected_shapes) {
        tensor.validate_destination("kv staging", shape, DType::U32)?;
    }
    let table = &staging[0];
    let max_blocks = schema.max_tokens / schema.block_size;
    let table_bytes = schema
        .graph_batch
        .checked_mul(max_blocks)
        .and_then(|elements| elements.checked_mul(DType::U32.size_in_bytes()))
        .ok_or_else(|| "kv attention: block table byte size overflow".to_string())?;
    // SAFETY: destination validation above proves every staging tensor is a
    // contiguous u32 buffer with the listed shape. The computed byte extents
    // cover exactly those shapes, and preparation runs before GPU dispatch, so
    // host writes are exclusive.
    unsafe {
        std::ptr::write_bytes(
            table
                .buffer
                .contents_ptr()
                .cast::<u8>()
                .add(table.layout.offset() * DType::U32.size_in_bytes()),
            0,
            table_bytes,
        );
        for tensor in &staging[1..] {
            std::ptr::write_bytes(
                tensor
                    .buffer
                    .contents_ptr()
                    .cast::<u8>()
                    .add(tensor.layout.offset() * DType::U32.size_in_bytes()),
                0,
                schema.graph_batch * DType::U32.size_in_bytes(),
            );
        }
    }
    let layer = layer as usize;
    if let Some(rows) = &kv.packed_rows {
        if rows.len() != schema.graph_batch
            || kv
                .packed_positions
                .as_ref()
                .is_none_or(|positions| positions.len() != rows.len())
        {
            return Err("kv attention: malformed packed row metadata".to_string());
        }
        // Reserve each physical sequence frontier once. Every graph row then
        // references that same block table with its own context and position.
        for slot in &kv.slots {
            let mut state = slot
                .lock()
                .map_err(|error| format!("kv attention: sequence lock poisoned: {error}"))?;
            let advance = state.advance;
            kv_prepare(
                &kv.pool,
                &mut state,
                layer,
                schema.window,
                plan.mode,
                plan.kv_heads,
                plan.head_dim,
                advance,
            )?;
        }
        let mut local_rows = vec![0usize; kv.slots.len()];
        for (batch_index, request) in rows.iter().enumerate() {
            let Some(request) = request else { continue };
            let slot = kv.slots.get(*request).ok_or_else(|| {
                "kv attention: packed row references an unknown sequence".to_string()
            })?;
            let state = slot
                .lock()
                .map_err(|error| format!("kv attention: sequence lock poisoned: {error}"))?;
            if state.blocks.len() > max_blocks || local_rows[*request] >= state.advance {
                return Err(
                    "kv attention: packed row exceeds its logical sequence rows".to_string()
                );
            }
            unsafe {
                let table_row = table
                    .buffer
                    .contents_ptr()
                    .cast::<u32>()
                    .add(table.layout.offset() + batch_index * max_blocks);
                std::ptr::copy_nonoverlapping(state.blocks.as_ptr(), table_row, state.blocks.len());
            }
            let context = state.cursor + local_rows[*request] + 1;
            local_rows[*request] += 1;
            for (tensor, value) in
                staging[1..]
                    .iter()
                    .zip([context as u32, state.head as u32, 1, 0])
            {
                unsafe {
                    tensor
                        .buffer
                        .contents_ptr()
                        .cast::<u32>()
                        .add(tensor.layout.offset() + batch_index)
                        .write(value);
                }
            }
        }
        if local_rows
            .iter()
            .zip(&kv.slots)
            .any(|(rows, slot)| slot.lock().map_or(true, |state| *rows != state.advance))
        {
            return Err("kv attention: packed logical row counts are inconsistent".to_string());
        }
        return Ok(());
    }
    for (request_index, slot) in kv.slots.iter().enumerate() {
        let batch_index = kv.lanes[request_index];
        let mut state = slot
            .lock()
            .map_err(|error| format!("kv attention: sequence lock poisoned: {error}"))?;
        let (_, needed, _) = kv_prepare(
            &kv.pool,
            &mut state,
            layer,
            schema.window,
            plan.mode,
            plan.kv_heads,
            plan.head_dim,
            plan.time,
        )?;
        if state.blocks.len() > max_blocks {
            return Err("kv attention: block table exceeds its schema capacity".to_string());
        }
        // SAFETY: the table row has `max_blocks` u32 entries and the preceding
        // capacity check proves `state.blocks.len() <= max_blocks`; source and
        // destination are distinct allocations.
        unsafe {
            let table_row = table
                .buffer
                .contents_ptr()
                .cast::<u32>()
                .add(table.layout.offset() + batch_index * max_blocks);
            std::ptr::copy_nonoverlapping(state.blocks.as_ptr(), table_row, state.blocks.len());
        }
        for (tensor, value) in staging[1..].iter().zip([
            needed as u32,
            state.head as u32,
            state.advance as u32,
            plan.time.saturating_sub(state.advance) as u32,
        ]) {
            // SAFETY: each scalar staging tensor was validated as `[batch]`
            // contiguous u32 storage and `batch_index < active_batch <= batch`.
            unsafe {
                tensor
                    .buffer
                    .contents_ptr()
                    .cast::<u32>()
                    .add(tensor.layout.offset() + batch_index)
                    .write(value);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn kv_attention_into(
    kv: &KvContext,
    layer: u32,
    q: &runtime::metal::run::MetalTensor,
    k: &runtime::metal::run::MetalTensor,
    v: &runtime::metal::run::MetalTensor,
    scale: f64,
    window: Option<usize>,
    mode: KvAttentionMode,
    output: &runtime::metal::run::MetalTensor,
    staging: &[runtime::metal::run::MetalTensor],
    scratch: &[runtime::metal::run::MetalTensor],
) -> err::Res<()> {
    if q.dtype != DType::F32 || k.dtype != DType::F32 || v.dtype != DType::F32 {
        return Err("kv attention: paged destination path requires f32 q/k/v".to_string());
    }
    let rank = q.layout.shape().len();
    let (batch, time, query_heads, head_dim) = (
        q.layout.shape()[..rank - 3].iter().product::<usize>(),
        q.layout.shape()[rank - 2],
        q.layout.shape()[rank - 3],
        q.layout.shape()[rank - 1],
    );
    let kv_heads = k.layout.shape()[rank - 3];
    if k.layout.shape()[..rank - 3] != q.layout.shape()[..rank - 3]
        || v.layout.shape()[..rank - 3] != q.layout.shape()[..rank - 3]
        || v.layout.shape()[rank - 3] != kv_heads
        || kv_heads == 0
        || !query_heads.is_multiple_of(kv_heads)
        || k.layout.shape()[rank - 2] != time
        || v.layout.shape()[rank - 2] != time
        || k.layout.shape()[rank - 1] != head_dim
        || v.layout.shape()[rank - 1] != head_dim
    {
        return Err("kv attention: incompatible grouped-query q/k/v shapes".to_string());
    }
    if batch != kv.schema.graph_batch || output.layout.shape() != q.layout.shape() {
        return Err("kv attention: destination shape or decode batch is inconsistent".to_string());
    }
    if staging.len() != 5 {
        return Err("kv attention: planned invocation staging is missing".to_string());
    }
    let (tables, context_lengths, block_bases, advances) =
        (&staging[0], &staging[1], &staging[2], &staging[3]);
    let layer = layer as usize;
    let slab_dtype = kv.pool.k[layer].dtype();
    if !paged::is_supported(q, slab_dtype, head_dim) {
        return Err(format!(
            "kv attention: paged destination path does not support f32/{slab_dtype:?}/D={head_dim}"
        ));
    }
    let (k_scales, v_scales) = match slab_dtype {
        DType::U8 => (
            Some(kv.pool.scales[2 * layer].metal()?),
            Some(kv.pool.scales[2 * layer + 1].metal()?),
        ),
        _ => (None, None),
    };
    paged::scatter_into(
        k,
        v,
        kv.pool.k[layer].metal()?,
        kv.pool.v[layer].metal()?,
        k_scales,
        v_scales,
        tables,
        context_lengths,
        block_bases,
        kv.pool.block_size,
        advances,
        paged::IntoResources::empty(),
    )?;
    if kv.state_only {
        return Ok(());
    }
    let attend = if mode == KvAttentionMode::BidirectionalBlock {
        paged::attention_block_into
    } else {
        paged::attention_into
    };
    let use_split = if scratch.is_empty() {
        false
    } else {
        // Splitting pays off once the context is long enough to saturate the
        // GPU; below that, the extra partial/combine dispatch is pure overhead.
        let lengths = unsafe {
            std::slice::from_raw_parts(
                context_lengths
                    .buffer
                    .contents_ptr()
                    .cast::<u32>()
                    .add(context_lengths.layout.offset()),
                context_lengths.layout.shape()[0],
            )
        };
        lengths.iter().any(|&context| context >= 512)
    };
    if !use_split {
        attend(
            q,
            kv.pool.k[layer].metal()?,
            kv.pool.v[layer].metal()?,
            k_scales,
            v_scales,
            tables,
            context_lengths,
            block_bases,
            window,
            scale,
            kv.pool.block_size,
            advances,
            output,
            paged::IntoResources::empty(),
        )?;
    } else {
        // Split flash-decoding path: planned only for causal one-token
        // decode; the scratch shape fixes the split count.
        if mode == KvAttentionMode::BidirectionalBlock {
            return Err("kv attention: split decode scratch is causal-only".to_string());
        }
        if scratch.len() != 1 || time != 1 {
            return Err(
                "kv attention: split decode requires one scratch tensor and one-token decode"
                    .to_string(),
            );
        }
        let splits = scratch[0].layout.shape().get(2).copied().unwrap_or(0);
        paged::attention_split_into(
            q,
            kv.pool.k[layer].metal()?,
            kv.pool.v[layer].metal()?,
            k_scales,
            v_scales,
            tables,
            context_lengths,
            block_bases,
            window,
            scale,
            kv.pool.block_size,
            advances,
            splits,
            output,
            paged::IntoResources {
                staging: &[],
                status: &[],
                scratch,
            },
        )?;
    }
    Ok(())
}

// Validates the slot and allocates blocks up to the new frontier.
// Returns (cursor, needed, start): the pre-run cursor, the post-run
// frontier, and the attention window's start. The executable's fused
// paged scatter consumes the resulting planned block-table staging.
#[allow(clippy::too_many_arguments)]
fn kv_prepare(
    pool: &Arc<PoolInner>,
    state: &mut SeqState,
    layer: usize,
    window: Option<usize>,
    mode: KvAttentionMode,
    h: usize,
    d: usize,
    t: usize,
) -> err::Res<(usize, usize, usize)> {
    if layer >= pool.k.len() {
        return Err(format!(
            "kv attention: layer {layer} out of range for {} pool layers",
            pool.k.len()
        ));
    }
    if h != pool.kv_heads || d != pool.head_dim {
        return Err(format!(
            "kv attention: layer {layer} shape [{h}, {d}] does not match pool geometry [{}, {}]",
            pool.kv_heads, pool.head_dim
        ));
    }
    let cursor = state.cursor;
    // Chunked prefill: q carries the chunk length t, only `advance`
    // rows are real; the rest are pads whose outputs the caller
    // discards (causality keeps real rows from ever attending to them).
    let advance = state.advance;
    if advance == 0 || advance > t {
        return Err(format!(
            "kv attention: advance {advance} out of range for chunk length {t}"
        ));
    }
    let needed = cursor
        .checked_add(advance)
        .ok_or_else(|| "kv attention: token frontier overflow".to_string())?;
    // Live rows after this step: everything from the attention window
    // frontier on. Blocks fully below the frontier are dead and their
    // capacity is reclaimed, so a windowed sequence's footprint is
    // O(window) however long it generates.
    let start = window.map_or(0, |w| {
        if mode == KvAttentionMode::BidirectionalBlock {
            cursor.saturating_sub(w)
        } else {
            needed.saturating_sub(w)
        }
    });
    if needed - start > pool.max_tokens {
        return Err(format!(
            "kv attention: live context {} exceeds pool capacity {}",
            needed - start,
            pool.max_tokens
        ));
    }
    let needed_blocks = needed.div_ceil(pool.block_size);
    while state.head + state.blocks.len() < needed_blocks {
        // Provisional and durable executions may both reclaim unreferenced
        // prefix-cache blocks under pressure. Live blocks are never eligible.
        let block = pool.alloc_block_with_cache_eviction(true).ok_or_else(|| {
            err::err_str(format!(
                "kv attention: pool exhausted ({} tokens across live sequences)",
                pool.max_tokens
            ))
        })?;
        state.blocks.push(block);
    }
    Ok((cursor, needed, start))
}

// Remove blocks before `start`. When the sequence holds the last reference,
// `unref_block` moves the block to the prefix cache for matching prompts.
fn kv_evict(pool: &PoolInner, state: &mut SeqState, start: usize) {
    while !state.blocks.is_empty() && (state.head + 1) * pool.block_size <= start {
        let dead = state.blocks.remove(0);
        pool.unref_block(dead);
        state.head += 1;
    }
}

#[napi]
pub struct NativeKvPool {
    inner: Arc<PoolInner>,
}

#[napi]
impl NativeKvPool {
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
        Self::for_ordinal(
            layers, kv_heads, head_dim, max_tokens, block_size, dtype, recurrent, 0,
        )
    }

    #[napi(factory)]
    #[allow(clippy::too_many_arguments)]
    pub fn for_device(
        layers: u32,
        kv_heads: u32,
        head_dim: u32,
        max_tokens: u32,
        block_size: Option<u32>,
        dtype: Option<NativeDType>,
        recurrent: Option<NativeRecurrentStateSchema>,
        device_ordinal: u32,
    ) -> Result<Self> {
        Self::for_ordinal(
            layers,
            kv_heads,
            head_dim,
            max_tokens,
            block_size,
            dtype,
            recurrent,
            device_ordinal as usize,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn for_ordinal(
        layers: u32,
        kv_heads: u32,
        head_dim: u32,
        max_tokens: u32,
        block_size: Option<u32>,
        dtype: Option<NativeDType>,
        recurrent: Option<NativeRecurrentStateSchema>,
        device_ordinal: usize,
    ) -> Result<Self> {
        with_device(device_ordinal, || {
            Self::create(
                layers,
                kv_heads,
                head_dim,
                max_tokens,
                block_size,
                dtype,
                recurrent,
                device_ordinal,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn create(
        layers: u32,
        kv_heads: u32,
        head_dim: u32,
        max_tokens: u32,
        block_size: Option<u32>,
        dtype: Option<NativeDType>,
        recurrent: Option<NativeRecurrentStateSchema>,
        device_ordinal: usize,
    ) -> Result<Self> {
        let dtype: DType = dtype.unwrap_or(NativeDType::F32).into();
        // u8 slabs are the int8-quantized storage tier: bytes plus a
        // per-(token, head) f32 scale, not an arithmetic dtype.
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16 | DType::U8) {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "kv pool: dtype must be f32, f16, bf16 or u8 (int8-quantized), got {}",
                    dtype_name(dtype)
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
        };
        let conv = ConvGeometry {
            layers: recurrent.conv_layers as usize,
            channels: recurrent.conv_channels as usize,
            kernel: recurrent.conv_kernel as usize,
        };
        // Zero attention layers needs no KV slabs; recurrent state, sequence
        // cursors, and block hashing still work independently.
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
        if max_tokens == 0 || max_tokens % block_size != 0 {
            return Err(Error::new(
                Status::InvalidArg,
                format!("kv pool: capacity {max_tokens} must be a positive multiple of block size {block_size}"),
            ));
        }
        if (kda.layers == 0 && (kda.heads != 0 || kda.head_dim != 0 || kda.value_dim != 0))
            || (kda.layers > 0 && (kda.heads == 0 || kda.head_dim == 0 || kda.value_dim == 0))
        {
            return Err(Error::new(
                Status::InvalidArg,
                "kv pool: KDA geometry must be entirely zero or entirely positive".to_string(),
            ));
        }
        if (conv.layers == 0 && (conv.channels != 0 || conv.kernel != 0))
            || (conv.layers > 0 && (conv.channels == 0 || conv.kernel == 0))
        {
            return Err(Error::new(
                Status::InvalidArg,
                "kv pool: convolution geometry must be entirely zero or entirely positive"
                    .to_string(),
            ));
        }
        let num_blocks = max_tokens / block_size;
        let mut k = Vec::with_capacity(layers);
        let mut v = Vec::with_capacity(layers);
        let mut scales = Vec::with_capacity(layers);
        for _ in 0..layers {
            let mslab = |row_width: usize, dtype: runtime::dtype::DType| {
                PoolSlab::NativeMetal(runtime::metal::run::MetalTensor {
                    buffer: runtime::metal::device::MetalDevice::get()
                        .alloc((max_tokens * row_width).max(1), dtype),
                    layout: runtime::layout::Layout::contiguous(vec![max_tokens, row_width]),
                    dtype,
                })
            };
            k.push(mslab(kv_heads * head_dim, dtype));
            v.push(mslab(kv_heads * head_dim, dtype));
            if dtype == DType::U8 {
                for _ in 0..2 {
                    scales.push(mslab(kv_heads, runtime::dtype::DType::F32));
                }
            }
        }
        let mut padding_state = SeqState {
            blocks: Vec::new(),
            head: 0,
            cursor: 0,
            advance: 0,
            last_hash: HASH_SEED,
            pending: Vec::new(),
            kda_states: Vec::new(),
            conv_states: Vec::new(),
        };
        prepare_recurrent_states(&mut padding_state, kda, conv).map_err(to_napi_err)?;
        Ok(Self {
            inner: Arc::new(PoolInner {
                device_ordinal,
                k,
                v,
                scales,
                kv_heads,
                head_dim,
                block_size,
                max_tokens,
                dtype,
                kda,
                conv,
                padding_kda_states: padding_state.kda_states,
                padding_conv_states: padding_state.conv_states,
                blocks: Mutex::new(BlockStore::new(num_blocks)),
            }),
        })
    }

    #[napi(getter)]
    pub fn capacity(&self) -> u32 {
        self.inner.max_tokens as u32
    }

    // Blocks available for new content: free plus reclaimable cached.
    #[napi(getter)]
    pub fn free_blocks(&self) -> u32 {
        self.inner.available() as u32
    }

    // Unreferenced blocks held by the prefix cache, reusable by a
    // prompt with a matching prefix and evictable under pressure.
    #[napi(getter)]
    pub fn cached_blocks(&self) -> u32 {
        self.inner.cached_count() as u32
    }

    #[napi]
    pub fn make_sequence(&self) -> Result<NativeKvSequence> {
        with_device(self.inner.device_ordinal, || {
            Ok(NativeKvSequence {
                pool: self.inner.clone(),
                state: Arc::new(Mutex::new(
                    sequence_state(&self.inner, false).map_err(to_napi_err)?,
                )),
                run_lock: Arc::new(Mutex::new(())),
                released: Arc::new(AtomicBool::new(false)),
            })
        })
    }
}

#[napi(custom_finalize)]
pub struct NativeKvSequence {
    pool: Arc<PoolInner>,
    state: Arc<Mutex<SeqState>>,
    // Serializes runs of this sequence; other sequences run concurrently
    // (their blocks are disjoint by allocation).
    run_lock: Arc<Mutex<()>>,
    released: Arc<AtomicBool>,
}

// Garbage collection returns sequence blocks to the pool. `release()` returns
// them earlier but is not required for cleanup.
impl ObjectFinalize for NativeKvSequence {
    fn finalize(self, _env: Env) -> Result<()> {
        self.return_blocks();
        Ok(())
    }
}

impl NativeKvSequence {
    // A fresh empty sequence on this sequence's pool (used for internal
    // pad slots in ragged batched runs).
    #[cfg(test)]
    fn new_sequence_like(&self) -> Self {
        NativeKvSequence {
            pool: self.pool.clone(),
            state: Arc::new(Mutex::new(
                sequence_state(&self.pool, true)
                    .expect("pool padding state was allocated during construction"),
            )),
            run_lock: Arc::new(Mutex::new(())),
            released: Arc::new(AtomicBool::new(false)),
        }
    }

    fn return_blocks(&self) {
        if self.released.swap(true, Ordering::SeqCst) {
            return;
        }
        // Drain under the run lock: a run holds the sequence's blocks
        // for its whole duration, so releasing must wait for an
        // in-flight run rather than unref blocks it still scatters
        // into. Lock order stays run_lock -> state -> pool blocks.
        let _run_guard = self
            .run_lock
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        for block in state.blocks.drain(..) {
            self.pool.unref_block(block);
        }
        state.head = 0;
        state.cursor = 0;
        state.advance = 0;
        state.last_hash = HASH_SEED;
        state.pending.clear();
    }

    fn prefill_match_recurrent(&self, state: &mut SeqState, tokens: &[u32]) -> Result<u32> {
        let block_size = self.pool.block_size;
        let matchable = tokens.len().saturating_sub(1) / block_size;
        let mut boundary_hashes = Vec::with_capacity(matchable);
        let mut hash = HASH_SEED;
        for i in 0..matchable {
            hash = chain_hash(hash, &tokens[i * block_size..(i + 1) * block_size]);
            boundary_hashes.push(hash);
        }
        let Some((blocks, hash, snapshot)) = self.pool.take_recurrent_prefix(&boundary_hashes)
        else {
            return Ok(0);
        };
        if let Err(message) = snapshot.restore_into(state) {
            for block in blocks {
                self.pool.unref_block(block);
            }
            state.head = 0;
            state.cursor = 0;
            state.advance = 0;
            state.last_hash = HASH_SEED;
            state.pending.clear();
            return Err(Error::new(
                Status::GenericFailure,
                format!("prefill match: recurrent state restore failed: {message}"),
            ));
        }
        let matched = blocks.len() * block_size;
        state.blocks = blocks;
        state.last_hash = hash;
        state.cursor = matched;
        Ok(matched as u32)
    }
}

impl Drop for NativeKvSequence {
    fn drop(&mut self) {
        self.return_blocks();
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

    // Returns the sequence's blocks to the pool. Running a released
    // sequence is an error; releasing twice is a no-op.
    #[napi]
    pub fn release(&self) {
        self.return_blocks();
    }

    // Claims the longest resident prompt prefix and returns its token length.
    // The caller prefills only the remaining suffix. Only whole blocks match
    // because a partial tail block is not final. The block containing the last
    // prompt token is always computed because its logits are the prefill
    // result. Hybrid pools with KV blocks and recurrent state match only at
    // boundaries with a published recurrent snapshot, then restore that state
    // into the sequence. Content addressing lets prompts share a common prefix
    // without exposing the match to callers.
    #[napi]
    pub fn prefill_match(&self, tokens: Vec<u32>) -> Result<u32> {
        let _run_guard = self.run_lock.lock().map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("kv sequence lock poisoned: {e}"),
            )
        })?;
        if self.released.load(Ordering::SeqCst) {
            return Err(Error::new(
                Status::GenericFailure,
                "kv sequence is released".to_string(),
            ));
        }
        let mut state = self.state.lock().map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("kv sequence lock poisoned: {e}"),
            )
        })?;
        if state.cursor > 0 || !state.blocks.is_empty() {
            return Err(Error::new(
                Status::GenericFailure,
                "prefill match: sequence already holds tokens".to_string(),
            ));
        }
        if self.pool.kda.layers > 0 || self.pool.conv.layers > 0 {
            if self.pool.k.is_empty() {
                return Ok(0);
            }
            return self.prefill_match_recurrent(&mut state, &tokens);
        }
        let block_size = self.pool.block_size;
        let matchable = tokens.len().saturating_sub(1) / block_size;
        let mut hash = HASH_SEED;
        for i in 0..matchable {
            let next = chain_hash(hash, &tokens[i * block_size..(i + 1) * block_size]);
            match self.pool.take_block(next) {
                Some(block) => {
                    state.blocks.push(block);
                    hash = next;
                }
                None => break,
            }
        }
        state.last_hash = hash;
        state.cursor = (state.head + state.blocks.len()) * block_size;
        Ok(state.cursor as u32)
    }
}

struct StatefulExecutable {
    cursor_slot: u32,
    cursor_tensor: bool,
    allows_window_eviction: bool,
    schema: KvStateSchema,
    packed_rows_per_sequence: Option<usize>,
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
                "execute: stateful executables do not accept scalar inputs".to_string(),
            ));
        }
        match (
            sequence_count,
            slots,
            active_mask,
            valid_lengths,
            advances,
            token_count,
        ) {
            (Some(sequences), Some(slots), Some(_), Some(_), Some(_), Some(tokens))
                if sequences == tokens && sequences == slots.len() =>
            {
                Ok(())
            }
            (Some(sequences), _, _, _, _, Some(tokens)) => Err(Error::new(
                Status::InvalidArg,
                format!(
                    "execute: expected one token list per sequence, got {tokens} for {sequences} sequences"
                ),
            )),
            _ => Err(Error::new(
                Status::InvalidArg,
                "execute: stateful executables require sequences, slots, activeMask, validLengths, advances, and tokens".to_string(),
            )),
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
            "execute: stateless executables do not accept state".to_string(),
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

fn validate_sampled_outputs(
    executable: &executable::MetalExecutable,
    sampling: &[SamplingOptions],
) -> Result<()> {
    if sampling.len() > executable.program.outputs.len() {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "executeSampled: executable has {} outputs for {} active sequences",
                executable.program.outputs.len(),
                sampling.len()
            ),
        ));
    }
    for (index, options) in sampling.iter().enumerate() {
        let output = executable.program.outputs[index];
        let declaration = &executable.program.values[output.index()];
        if declaration.shape.len() != 1 {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "sample: logits must be rank 1, got rank {}",
                    declaration.shape.len()
                ),
            ));
        }
        if !matches!(declaration.dtype, DType::F16 | DType::BF16 | DType::F32) {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "sample: logits dtype must be f16, bf16, or f32, got {}",
                    declaration.dtype.name()
                ),
            ));
        }
        let vocabulary = declaration.shape[0];
        if vocabulary == 0 {
            return Err(Error::new(
                Status::InvalidArg,
                "sample: logits must be non-empty",
            ));
        }
        if vocabulary > MAX_SAMPLING_VOCABULARY {
            return Err(Error::new(
                Status::InvalidArg,
                format!("sample: vocabulary {vocabulary} exceeds limit {MAX_SAMPLING_VOCABULARY}"),
            ));
        }
        if let Some(top_k) = options.top_k {
            if top_k > vocabulary {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!("sample: topK must be in [1, {vocabulary}], got {top_k}"),
                ));
            }
        }
    }
    Ok(())
}

fn validate_fixed_lanes(
    batch: usize,
    sequence_count: usize,
    slots: &[u32],
    active_mask: &[bool],
    valid_lengths: &[u32],
    advances: &[u32],
    tokens: &[Vec<u32>],
) -> Result<Vec<usize>> {
    if slots.len() != sequence_count
        || tokens.len() != sequence_count
        || active_mask.len() != batch
        || valid_lengths.len() != batch
        || advances.len() != batch
    {
        return Err(Error::new(
            Status::InvalidArg,
            format!("kv run: slots and tokens must be compact and activeMask/validLengths/advances must have fixed batch length {batch}"),
        ));
    }
    let lanes = slots.iter().map(|slot| *slot as usize).collect::<Vec<_>>();
    if lanes.iter().any(|lane| *lane >= batch)
        || lanes
            .iter()
            .enumerate()
            .any(|(i, lane)| lanes[..i].contains(lane))
        || active_mask.iter().filter(|active| **active).count() != sequence_count
        || lanes.iter().any(|lane| !active_mask[*lane])
        || active_mask.iter().enumerate().any(|(lane, active)| {
            *active != lanes.contains(&lane)
                || valid_lengths[lane] != advances[lane]
                || (!*active && advances[lane] != 0)
        })
        || lanes.iter().enumerate().any(|(request, lane)| {
            advances[*lane] == 0 || tokens[request].len() != advances[*lane] as usize
        })
    {
        return Err(Error::new(
            Status::InvalidArg,
            "kv run: invalid fixed-lane mask, lengths, advances, slots, or token rows".to_string(),
        ));
    }
    Ok(lanes)
}

fn validate_stateful_tensor_input(
    input: &NativeTensor,
    slot: usize,
    declared: &ProgramSlot,
) -> Result<()> {
    let got = input.val_cloned()?;
    let shape = got.shape();
    let device = got.device();
    if shape != declared.shape
        || got.dtype() != declared.dtype
        || !device.same_device(&declared.device)
    {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "input slot {slot}: expected bounded {}, got {:?}:{}@{}",
                declared.signature(),
                shape,
                got.dtype().name(),
                device
            ),
        ));
    }
    Ok(())
}

#[napi]
pub struct Executable {
    inner: ProgramInner,
    state: Option<StatefulExecutable>,
    device_ordinal: usize,
}

#[napi]
impl Executable {
    #[napi(getter)]
    pub fn diagnostics(&self) -> NativeExecutableDiagnostics {
        executable_diagnostics(&self.inner.executable.diagnostics)
    }

    #[napi(getter)]
    pub fn stateful(&self) -> bool {
        self.state.is_some()
    }

    #[napi(getter)]
    pub fn batch(&self) -> u32 {
        self.state
            .as_ref()
            .map_or(0, |state| state.schema.batch as u32)
    }

    #[napi(getter)]
    pub fn packed_rows_per_sequence(&self) -> Option<u32> {
        self.state
            .as_ref()
            .and_then(|state| state.packed_rows_per_sequence)
            .map(|rows| rows as u32)
    }

    #[napi(getter)]
    pub fn allows_window_eviction(&self) -> bool {
        self.state
            .as_ref()
            .is_some_and(|state| state.allows_window_eviction)
    }

    #[napi(getter)]
    pub fn layers(&self) -> u32 {
        self.state
            .as_ref()
            .map_or(0, |state| state.schema.layers as u32)
    }

    #[napi(getter)]
    pub fn kv_heads(&self) -> u32 {
        self.state
            .as_ref()
            .map_or(0, |state| state.schema.kv_heads as u32)
    }

    #[napi(getter)]
    pub fn head_dim(&self) -> u32 {
        self.state
            .as_ref()
            .map_or(0, |state| state.schema.head_dim as u32)
    }

    #[napi(getter)]
    pub fn kda_layers(&self) -> u32 {
        self.state
            .as_ref()
            .map_or(0, |state| state.schema.kda.layers as u32)
    }

    #[napi(getter)]
    pub fn kda_heads(&self) -> u32 {
        self.state
            .as_ref()
            .map_or(0, |state| state.schema.kda.heads as u32)
    }

    #[napi(getter)]
    pub fn kda_head_dim(&self) -> u32 {
        self.state
            .as_ref()
            .map_or(0, |state| state.schema.kda.head_dim as u32)
    }

    #[napi(getter)]
    pub fn kda_value_dim(&self) -> u32 {
        self.state
            .as_ref()
            .map_or(0, |state| state.schema.kda.value_dim as u32)
    }

    #[napi(getter)]
    pub fn conv_layers(&self) -> u32 {
        self.state
            .as_ref()
            .map_or(0, |state| state.schema.conv.layers as u32)
    }

    #[napi(getter)]
    pub fn conv_channels(&self) -> u32 {
        self.state
            .as_ref()
            .map_or(0, |state| state.schema.conv.channels as u32)
    }

    #[napi(getter)]
    pub fn conv_kernel(&self) -> u32 {
        self.state
            .as_ref()
            .map_or(0, |state| state.schema.conv.kernel as u32)
    }

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
        token: Option<&CancellationToken>,
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
        if self.state.is_none() {
            return self.execute_stateless(inputs, scalars, token).await;
        }

        match self
            .execute_stateful(
                inputs,
                sequences.expect("state invocation was validated"),
                slots.expect("state invocation was validated"),
                active_mask.expect("state invocation was validated"),
                valid_lengths.expect("state invocation was validated"),
                advances.expect("state invocation was validated"),
                tokens.expect("state invocation was validated"),
                StatefulInvocation::Tensors,
                token,
            )
            .await?
        {
            StatefulExecutionOutput::Tensors(outputs) => Ok(outputs),
            StatefulExecutionOutput::Samples(_) => {
                unreachable!("ordinary stateful execution returned samples")
            }
        }
    }

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
        validate_sampled_outputs(&self.inner.executable, &sampling)?;

        match self
            .execute_stateful(
                inputs,
                sequences,
                slots,
                active_mask,
                valid_lengths,
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
        sampling: Vec<NativeSamplingOptions>,
        max_draft_tokens: u32,
        page_limits: Vec<u32>,
        eos_tokens: Vec<Vec<u32>>,
        cancellation_token: Option<&CancellationToken>,
    ) -> Result<Vec<Vec<u32>>> {
        if proposer.device_ordinal != self.device_ordinal
            || target_sequences
                .iter()
                .any(|sequence| sequence.pool.device_ordinal != self.device_ordinal)
            || proposer_sequences
                .iter()
                .any(|sequence| sequence.pool.device_ordinal != self.device_ordinal)
        {
            return Err(Error::new(
                Status::InvalidArg,
                "executeSpeculative: executables and sequences must use the same Metal device",
            ));
        }
        let sampling = sampling
            .into_iter()
            .map(sampling_options)
            .collect::<Result<Vec<_>>>()?;
        let target = validate_speculative_program(self, false, false)?;
        let proposer = validate_speculative_program(proposer, true, false)?;
        validate_speculative_request(
            &target,
            &proposer,
            &target_sequences,
            &proposer_sequences,
            &slots,
            &pending_tokens,
            &sampling,
            max_draft_tokens as usize,
            &page_limits,
            &eos_tokens,
        )?;

        let target_states = target_sequences
            .iter()
            .map(|sequence| sequence.state.clone())
            .collect::<Vec<_>>();
        let proposer_states = proposer_sequences
            .iter()
            .map(|sequence| sequence.state.clone())
            .collect::<Vec<_>>();
        let target_pool = target_sequences[0].pool.clone();
        let proposer_pool = proposer_sequences[0].pool.clone();
        let mut locks = target_sequences
            .iter()
            .chain(&proposer_sequences)
            .map(|sequence| sequence.run_lock.clone())
            .collect::<Vec<_>>();
        locks.sort_by_key(|lock| Arc::as_ptr(lock) as usize);
        locks.dedup_by(|left, right| Arc::ptr_eq(left, right));
        let released = target_sequences
            .iter()
            .chain(&proposer_sequences)
            .map(|sequence| sequence.released.clone())
            .collect::<Vec<_>>();
        let slots = slots
            .into_iter()
            .map(|slot| slot as usize)
            .collect::<Vec<_>>();

        let compute = move |cancelled: &effect_torch_runtime::CancellationFlag,
                            cancellation: &CancellationState| {
            let _guards = locks
                .iter()
                .map(|lock| {
                    lock.lock().map_err(|error| {
                        Error::new(
                            Status::GenericFailure,
                            format!("speculative sequence lock poisoned: {error}"),
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
                    "speculative sequence is released".to_string(),
                ));
            }
            let mut target_shadow =
                ShadowSequences::new(target_pool, &target_states).map_err(to_napi_err)?;
            let mut proposer_shadow =
                ShadowSequences::new(proposer_pool, &proposer_states).map_err(to_napi_err)?;
            let pages = execute_speculative_blocking(
                &target,
                &proposer,
                &mut target_shadow,
                &mut proposer_shadow,
                &slots,
                &pending_tokens,
                &sampling,
                max_draft_tokens as usize,
                &page_limits,
                &eos_tokens,
                cancelled,
                None,
                None,
            )
            .map_err(to_napi_err)?;
            if !cancellation.complete() {
                return Err(Error::new(
                    Status::GenericFailure,
                    "operation aborted".to_string(),
                ));
            }
            // Check every destination before the first mutation. The run locks
            // exclude all other sequence operations through publication.
            for state in target_states.iter().chain(&proposer_states) {
                drop(state.lock().map_err(|error| {
                    Error::new(
                        Status::GenericFailure,
                        format!("speculative state lock poisoned: {error}"),
                    )
                })?);
            }
            publish_speculative_states(&mut target_shadow, &target_states, &pending_tokens, &pages);
            publish_speculative_states(
                &mut proposer_shadow,
                &proposer_states,
                &pending_tokens,
                &pages,
            );
            Ok(pages)
        };
        run_compute_on(self.device_ordinal, cancellation_token, compute).await
    }
}

#[derive(Clone)]
struct SpeculativeProgram {
    executable: Arc<executable::MetalExecutable>,
    generated: Vec<value::Value>,
    schema: KvStateSchema,
    batch: usize,
    time: usize,
    vocabulary: usize,
    token_dtype: DType,
    packed_rows_per_sequence: Option<usize>,
}

fn validate_speculative_program(
    program: &Executable,
    proposer: bool,
    allow_ephemeral_outputs: bool,
) -> Result<SpeculativeProgram> {
    let phase = if proposer { "proposer" } else { "target" };
    let state = program.state.as_ref().ok_or_else(|| {
        Error::new(
            Status::InvalidArg,
            format!("executeSpeculative: {phase} must be stateful"),
        )
    })?;
    if proposer == state.packed_rows_per_sequence.is_some() {
        return Err(Error::new(
            Status::InvalidArg,
            format!("executeSpeculative: {phase} has the wrong dense/packed decode layout"),
        ));
    }
    if state.schema.window.is_some()
        || state.allows_window_eviction
        || state.schema.kda.layers != 0
        || state.schema.conv.layers != 0
    {
        return Err(Error::new(
            Status::InvalidArg,
            format!("executeSpeculative: {phase} window/KDA/convolution state is unsupported"),
        ));
    }
    let tensor_slots = program
        .inner
        .slots
        .iter()
        .enumerate()
        .filter(|(index, slot)| {
            !slot.scalar && !(state.cursor_tensor && *index as u32 == state.cursor_slot)
        })
        .map(|(_, slot)| slot)
        .collect::<Vec<_>>();
    let expected_rows = state.schema.graph_batch;
    if tensor_slots.len() != 1
        || !matches!(tensor_slots[0].dtype, DType::U32 | DType::I64)
        || tensor_slots[0].shape.len() != 2
        || tensor_slots[0].shape[0] != expected_rows
        || tensor_slots[0].shape[1] != 1
    {
        return Err(Error::new(
            Status::InvalidArg,
            format!("executeSpeculative: {phase} must have one u32/i64 [B, T] token input"),
        ));
    }
    let time = tensor_slots[0].shape[1];
    if time == 0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!("executeSpeculative: {phase} token width must be positive"),
        ));
    }
    let outputs = &program.inner.executable.program.outputs;
    let values = &program.inner.executable.program.values;
    let vocabulary = if proposer {
        if outputs.len() != state.schema.batch {
            return Err(Error::new(
                Status::InvalidArg,
                "executeSpeculative: proposer must expose one last-row logits vector per fixed lane",
            ));
        }
        let first = &values[outputs[0].index()];
        if first.shape.len() != 1
            || outputs.iter().any(|output| {
                let declaration = &values[output.index()];
                declaration.shape != first.shape || declaration.dtype != first.dtype
            })
        {
            return Err(Error::new(
                Status::InvalidArg,
                "executeSpeculative: proposer outputs must be uniform rank-one logits",
            ));
        }
        first.shape[0]
    } else {
        if outputs.is_empty() || (!allow_ephemeral_outputs && outputs.len() != 1) {
            return Err(Error::new(
                Status::InvalidArg,
                "executeSpeculative: target must expose one [graphRows, 1, V] logits tensor",
            ));
        }
        let output = &values[outputs[0].index()];
        if output.shape.len() != 3
            || output.shape[0] != state.schema.graph_batch
            || output.shape[1] != 1
        {
            return Err(Error::new(
                Status::InvalidArg,
                "executeSpeculative: target output must be [graphRows, 1, V]",
            ));
        }
        output.shape[2]
    };
    let valid_dtype = outputs
        .iter()
        .take(if proposer { outputs.len() } else { 1 })
        .all(|output| {
            matches!(
                values[output.index()].dtype,
                DType::F16 | DType::BF16 | DType::F32
            )
        });
    if !valid_dtype || vocabulary == 0 || vocabulary > MAX_SAMPLING_VOCABULARY {
        return Err(Error::new(
            Status::InvalidArg,
            format!("executeSpeculative: {phase} has invalid logits dtype or vocabulary"),
        ));
    }
    Ok(SpeculativeProgram {
        executable: program.inner.executable.clone(),
        generated: program.inner.generated_bindings.clone(),
        schema: state.schema,
        batch: state.schema.batch,
        time,
        vocabulary,
        token_dtype: tensor_slots[0].dtype,
        packed_rows_per_sequence: state.packed_rows_per_sequence,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_speculative_request(
    target: &SpeculativeProgram,
    proposer: &SpeculativeProgram,
    target_sequences: &[&NativeKvSequence],
    proposer_sequences: &[&NativeKvSequence],
    slots: &[u32],
    pending: &[u32],
    sampling: &[SamplingOptions],
    max_draft_tokens: usize,
    page_limits: &[u32],
    eos_tokens: &[Vec<u32>],
) -> Result<()> {
    let count = target_sequences.len();
    if count == 0
        || count > target.batch
        || proposer.batch != target.batch
        || proposer.vocabulary != target.vocabulary
        || proposer_sequences.len() != count
        || slots.len() != count
        || pending.len() != count
        || sampling.len() != count
        || page_limits.len() != count
        || eos_tokens.len() != count
        || target
            .packed_rows_per_sequence
            .is_none_or(|rows| max_draft_tokens + 1 > rows)
    {
        return Err(Error::new(
            Status::InvalidArg,
            "executeSpeculative: incompatible batch, vocabulary, arrays, or draft width",
        ));
    }
    let mut sorted_slots = slots.to_vec();
    sorted_slots.sort_unstable();
    if sorted_slots.windows(2).any(|pair| pair[0] == pair[1])
        || sorted_slots
            .last()
            .is_some_and(|slot| *slot as usize >= target.batch)
        || pending
            .iter()
            .any(|token| *token as usize >= target.vocabulary)
        || page_limits.iter().any(|limit| *limit == 0)
    {
        return Err(Error::new(
            Status::InvalidArg,
            "executeSpeculative: invalid fixed slots, token, EOS token, or page limit",
        ));
    }
    for options in sampling {
        if options.top_k.is_some_and(|top_k| top_k > target.vocabulary) {
            return Err(Error::new(
                Status::InvalidArg,
                "executeSpeculative: topK exceeds the shared vocabulary",
            ));
        }
    }
    let mut all_states: Vec<*const Mutex<SeqState>> = Vec::with_capacity(count * 2);
    for (phase, sequences, schema) in [
        ("target", target_sequences, target.schema),
        ("proposer", proposer_sequences, proposer.schema),
    ] {
        for (index, sequence) in sequences.iter().enumerate() {
            if sequence.released.load(Ordering::SeqCst)
                || !Arc::ptr_eq(&sequence.pool, &sequences[0].pool)
                || sequence.pool.max_tokens != schema.max_tokens
                || sequence.pool.block_size != schema.block_size
                || sequence.pool.dtype != schema.kv_dtype
                || sequence.pool.k.len() != schema.layers
                || sequence.pool.kv_heads != schema.kv_heads
                || sequence.pool.head_dim != schema.head_dim
                || sequence.pool.kda != schema.kda
                || sequence.pool.conv != schema.conv
            {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!("executeSpeculative: {phase} sequence {index} is incompatible"),
                ));
            }
            let pointer = Arc::as_ptr(&sequence.state);
            if all_states.contains(&pointer) {
                return Err(Error::new(
                    Status::InvalidArg,
                    "executeSpeculative: duplicate sequence across target/proposer batches",
                ));
            }
            all_states.push(pointer);
        }
    }
    Ok(())
}

struct ShadowSequences {
    pool: Arc<PoolInner>,
    states: Vec<Arc<Mutex<SeqState>>>,
}

impl ShadowSequences {
    fn new(pool: Arc<PoolInner>, canonical: &[Arc<Mutex<SeqState>>]) -> err::Res<Self> {
        let mut shadow = Self {
            pool,
            states: Vec::with_capacity(canonical.len()),
        };
        for state in canonical {
            let state = state
                .lock()
                .map_err(|error| format!("speculative state lock poisoned: {error}"))?;
            shadow.pool.ref_blocks(&state.blocks)?;
            let retained = state.blocks.clone();
            let mut provisional = match sequence_state(&shadow.pool, false) {
                Ok(provisional) => provisional,
                Err(error) => {
                    shadow.pool.unref_blocks(&retained);
                    return Err(error);
                }
            };
            if shadow.pool.kda.layers > 0 || shadow.pool.conv.layers > 0 {
                let snapshot = RecurrentSnapshot::capture(&state).ok_or_else(|| {
                    shadow.pool.unref_blocks(&retained);
                    "speculative recurrent state cannot be copied".to_string()
                })?;
                if let Err(error) = snapshot.restore_into(&mut provisional) {
                    shadow.pool.unref_blocks(&retained);
                    return Err(error);
                }
            }
            provisional.blocks = retained;
            provisional.head = state.head;
            provisional.cursor = state.cursor;
            provisional.last_hash = state.last_hash;
            provisional.pending = state.pending.clone();
            shadow.states.push(Arc::new(Mutex::new(provisional)));
        }
        Ok(shadow)
    }

    fn provisional_blocks(&self, canonical: &[Arc<Mutex<SeqState>>]) -> u64 {
        self.states
            .iter()
            .zip(canonical)
            .map(|(provisional, canonical)| {
                let provisional = provisional
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                let canonical = canonical.lock().unwrap_or_else(|error| error.into_inner());
                provisional
                    .blocks
                    .iter()
                    .filter(|block| !canonical.blocks.contains(block))
                    .count() as u64
            })
            .sum()
    }
}

impl Drop for ShadowSequences {
    fn drop(&mut self) {
        let mut blocks = Vec::new();
        for state in &self.states {
            let mut state = state.lock().unwrap_or_else(|error| error.into_inner());
            blocks.append(&mut state.blocks);
        }
        self.pool.unref_blocks(&blocks);
    }
}

fn contiguous_f32_tensor(value: &value::Value, operation: &str) -> err::Res<MetalTensor> {
    let tensor = value.as_metal()?;
    if tensor.dtype != DType::F32 || !tensor.layout.is_contiguous() {
        return Err(format!("{operation}: expected a contiguous f32 tensor"));
    }
    Ok(tensor.clone())
}

fn f32_values(tensor: &MetalTensor) -> &[f32] {
    // SAFETY: `contiguous_f32_tensor` validated dtype/layout, stateful execution
    // synchronized the buffer before returning outputs, and the tensor owns the
    // underlying Metal allocation for the lifetime of the returned slice.
    unsafe {
        std::slice::from_raw_parts(
            tensor
                .buffer
                .contents_ptr()
                .cast::<f32>()
                .add(tensor.layout.offset()),
            tensor.numel(),
        )
    }
}

fn probabilities_f32(
    logits: &[f32],
    options: SamplingOptions,
    cancelled: &effect_torch_runtime::CancellationFlag,
) -> err::Res<Vec<f64>> {
    effective_probabilities(
        logits.len(),
        |index| logits[index] as f64,
        options,
        || cancelled.load(Ordering::Relaxed),
    )
}

fn sample_f32_at(
    probabilities: &[f32],
    sampling: InferenceSampling,
    sequence_id: u64,
    position: u64,
    purpose: SamplingPurpose,
    subcounter: u64,
) -> u32 {
    if sampling.temperature == 0.0 {
        return probabilities
            .iter()
            .position(|probability| *probability > 0.0)
            .unwrap_or(0) as u32;
    }
    let total: f64 = probabilities
        .iter()
        .map(|probability| *probability as f64)
        .sum();
    let mut draw = random_unit_at(sampling_coordinate(
        sampling.seed,
        sequence_id,
        position,
        purpose,
        subcounter,
    )) * total;
    for (token, probability) in probabilities.iter().enumerate() {
        draw -= *probability as f64;
        if draw < 0.0 {
            return token as u32;
        }
    }
    probabilities.len().saturating_sub(1) as u32
}

fn read_float_tensor(value: &value::Value) -> err::Res<Vec<f64>> {
    let tensor = value.as_metal()?;
    if !tensor.layout.is_contiguous() {
        return Err("executeSpeculative: logits output must be contiguous".to_string());
    }
    let offset = tensor.layout.offset();
    let elements = tensor.numel();
    let end = offset
        .checked_add(elements)
        .and_then(|count| count.checked_mul(tensor.dtype.size_in_bytes()))
        .ok_or_else(|| "executeSpeculative: logits layout overflow".to_string())?;
    if end > tensor.buffer.size {
        return Err("executeSpeculative: logits output exceeds its buffer".to_string());
    }
    let pointer = tensor.buffer.contents_ptr();
    let values = (0..elements)
        .map(|index| unsafe {
            match tensor.dtype {
                DType::F32 => *pointer.cast::<f32>().add(offset + index) as f64,
                DType::F16 => {
                    half::f16::from_bits(*pointer.cast::<u16>().add(offset + index)).to_f64()
                }
                DType::BF16 => {
                    half::bf16::from_bits(*pointer.cast::<u16>().add(offset + index)).to_f64()
                }
                _ => unreachable!("validated logits dtype"),
            }
        })
        .collect();
    Ok(values)
}

fn token_input(
    tokens: &[Vec<u32>],
    lanes: &[usize],
    batch: usize,
    time: usize,
    dtype: DType,
) -> err::Res<value::Value> {
    let mut dense = vec![0u32; batch * time];
    for (tokens, lane) in tokens.iter().zip(lanes) {
        if tokens.is_empty() || tokens.len() > time {
            return Err("executeSpeculative: invalid logical token row".to_string());
        }
        dense[*lane * time..*lane * time + tokens.len()].copy_from_slice(tokens);
    }
    let bytes = match dtype {
        DType::U32 => dense
            .iter()
            .flat_map(|token| token.to_ne_bytes())
            .collect::<Vec<_>>(),
        DType::I64 => dense
            .iter()
            .flat_map(|token| (*token as i64).to_ne_bytes())
            .collect::<Vec<_>>(),
        _ => return Err("executeSpeculative: token input must be u32 or i64".to_string()),
    };
    value::value_from_bytes(&bytes, &[batch, time], dtype)
}

fn token_vector(
    tokens: &[u32],
    lanes: &[usize],
    batch: usize,
    dtype: DType,
) -> err::Res<value::Value> {
    let mut dense = vec![0u32; batch];
    for (token, lane) in tokens.iter().zip(lanes) {
        dense[*lane] = *token;
    }
    let bytes = match dtype {
        DType::U32 => dense
            .iter()
            .flat_map(|token| token.to_ne_bytes())
            .collect::<Vec<_>>(),
        DType::I64 => dense
            .iter()
            .flat_map(|token| (*token as i64).to_ne_bytes())
            .collect::<Vec<_>>(),
        _ => return Err("ParallelBlock anchor input must be u32 or i64".to_string()),
    };
    value::value_from_bytes(&bytes, &[batch], dtype)
}

#[derive(Debug, PartialEq, Eq)]
struct PackedVerificationPlan {
    row_offsets: Vec<usize>,
    logical_rows: Vec<usize>,
    row_to_request: Vec<Option<usize>>,
    positions: Vec<usize>,
    tokens: Vec<Vec<u32>>,
}

fn packed_verification_plan(
    graph_rows: usize,
    rows_per_sequence: usize,
    cursors: &[usize],
    tokens: &[Vec<u32>],
) -> err::Res<PackedVerificationPlan> {
    if cursors.len() != tokens.len() || tokens.is_empty() {
        return Err("executeSpeculative: malformed packed sequence metadata".to_string());
    }
    let logical_rows = tokens.iter().map(Vec::len).collect::<Vec<_>>();
    if logical_rows
        .iter()
        .any(|rows| *rows == 0 || *rows > rows_per_sequence)
    {
        return Err(
            "executeSpeculative: packed logical rows exceed the compiled layout".to_string(),
        );
    }
    let mut row_offsets = Vec::with_capacity(tokens.len() + 1);
    row_offsets.push(0usize);
    for rows in &logical_rows {
        row_offsets.push(
            row_offsets
                .last()
                .copied()
                .unwrap()
                .checked_add(*rows)
                .ok_or_else(|| "executeSpeculative: packed row offset overflow".to_string())?,
        );
    }
    if row_offsets.last().copied().unwrap_or(0) > graph_rows {
        return Err("executeSpeculative: packed rows exceed graphRows".to_string());
    }
    let mut dense_tokens = vec![vec![0]; graph_rows];
    let mut row_to_request = vec![None; graph_rows];
    let mut positions = vec![0; graph_rows];
    for (request, row) in tokens.iter().enumerate() {
        let start = row_offsets[request];
        for (local, token) in row.iter().enumerate() {
            dense_tokens[start + local][0] = *token;
            row_to_request[start + local] = Some(request);
            positions[start + local] = cursors[request]
                .checked_add(local)
                .ok_or_else(|| "executeSpeculative: packed position overflow".to_string())?;
        }
    }
    Ok(PackedVerificationPlan {
        row_offsets,
        logical_rows,
        row_to_request,
        positions,
        tokens: dense_tokens,
    })
}

fn speculative_invocation(
    program: &SpeculativeProgram,
    shadow: &ShadowSequences,
    request_indices: &[usize],
    slots: &[usize],
    tokens: Vec<Vec<u32>>,
    packed: Option<&PackedVerificationPlan>,
) -> err::Res<(KvContext, value::Value)> {
    let states = request_indices
        .iter()
        .map(|index| shadow.states[*index].clone())
        .collect::<Vec<_>>();
    let lanes = request_indices
        .iter()
        .map(|index| slots[*index])
        .collect::<Vec<_>>();
    for (state, tokens) in states.iter().zip(&tokens) {
        state
            .lock()
            .map_err(|error| format!("speculative shadow lock poisoned: {error}"))?
            .advance = tokens.len();
    }
    let (input_tokens, input_lanes, packed_rows, packed_positions) = if let Some(packed) = packed {
        (
            packed.tokens.as_slice(),
            (0..program.schema.graph_batch).collect::<Vec<_>>(),
            Some(packed.row_to_request.clone()),
            Some(packed.positions.clone()),
        )
    } else {
        (tokens.as_slice(), lanes.clone(), None, None)
    };
    let input = token_input(
        input_tokens,
        &input_lanes,
        program.schema.graph_batch,
        program.time,
        program.token_dtype,
    )?;
    let context = KvContext {
        pool: shadow.pool.clone(),
        slots: states,
        schema: program.schema,
        tokens,
        lanes,
        packed_rows,
        packed_positions,
        publish_hashes: false,
        state_only: false,
    };
    Ok((context, input))
}

fn run_speculative_program(
    program: &SpeculativeProgram,
    shadow: &ShadowSequences,
    request_indices: &[usize],
    slots: &[usize],
    tokens: Vec<Vec<u32>>,
    packed: Option<&PackedVerificationPlan>,
    cancelled: &effect_torch_runtime::CancellationFlag,
    headless: bool,
) -> err::Res<Vec<value::Value>> {
    let (context, input) =
        speculative_invocation(program, shadow, request_indices, slots, tokens, packed)?;
    executable::execute_stateful(
        &program.executable,
        &[input],
        &program.generated,
        cancelled,
        &context,
        &|| true,
        headless,
    )
}

/// Deferred twin of [`run_speculative_program`]: encodes the chunk onto the
/// caller's submission stream, publishes its state transactions as
/// GPU-ordered device copies, advances the CPU-side cursors immediately,
/// and returns a pending whose drain performs the single host fence.
#[allow(clippy::too_many_arguments)]
fn run_speculative_program_deferred(
    program: &SpeculativeProgram,
    shadow: &ShadowSequences,
    request_indices: &[usize],
    slots: &[usize],
    tokens: Vec<Vec<u32>>,
    packed: Option<&PackedVerificationPlan>,
    cancelled: &effect_torch_runtime::CancellationFlag,
    submission: &device::MetalSubmissionGuard<'_>,
    headless: bool,
) -> err::Res<executable::PendingExecution> {
    let (context, input) =
        speculative_invocation(program, shadow, request_indices, slots, tokens, packed)?;
    executable::execute_stateful_deferred(
        &program.executable,
        &[input],
        &program.generated,
        cancelled,
        &context,
        &|| true,
        submission,
        headless,
    )
}

fn stateful_invocation(
    schema: KvStateSchema,
    shadow: &ShadowSequences,
    request_indices: &[usize],
    lanes: &[usize],
    tokens: Vec<Vec<u32>>,
    state_only: bool,
) -> err::Res<KvContext> {
    let states = request_indices
        .iter()
        .map(|index| shadow.states[*index].clone())
        .collect::<Vec<_>>();
    for (state, tokens) in states.iter().zip(&tokens) {
        state
            .lock()
            .map_err(|error| format!("parallel replay shadow lock poisoned: {error}"))?
            .advance = tokens.len();
    }
    Ok(KvContext {
        pool: shadow.pool.clone(),
        slots: states,
        schema,
        tokens,
        lanes: lanes.to_vec(),
        packed_rows: None,
        packed_positions: None,
        publish_hashes: false,
        state_only,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_stateful_values(
    executable: &Arc<executable::MetalExecutable>,
    generated: &[value::Value],
    schema: KvStateSchema,
    shadow: &ShadowSequences,
    request_indices: &[usize],
    lanes: &[usize],
    bindings: &[value::Value],
    tokens: Vec<Vec<u32>>,
    cancelled: &effect_torch_runtime::CancellationFlag,
    state_only: bool,
) -> err::Res<Vec<value::Value>> {
    let context = stateful_invocation(schema, shadow, request_indices, lanes, tokens, state_only)?;
    executable::execute_stateful(
        executable,
        bindings,
        generated,
        cancelled,
        &context,
        &|| true,
        false,
    )
}

/// Deferred twin of [`run_stateful_values`]; see
/// [`run_speculative_program_deferred`].
#[allow(clippy::too_many_arguments)]
fn run_stateful_values_deferred(
    executable: &Arc<executable::MetalExecutable>,
    generated: &[value::Value],
    schema: KvStateSchema,
    shadow: &ShadowSequences,
    request_indices: &[usize],
    lanes: &[usize],
    bindings: &[value::Value],
    tokens: Vec<Vec<u32>>,
    cancelled: &effect_torch_runtime::CancellationFlag,
    submission: &device::MetalSubmissionGuard<'_>,
    state_only: bool,
) -> err::Res<executable::PendingExecution> {
    let context = stateful_invocation(schema, shadow, request_indices, lanes, tokens, state_only)?;
    executable::execute_stateful_deferred(
        executable,
        bindings,
        generated,
        cancelled,
        &context,
        &|| true,
        submission,
        false,
    )
}

fn replay_outputs(
    replay: &ReplayProgram,
    shadow: &ShadowSequences,
    source_outputs: &[value::Value],
    tap_outputs: &[usize],
    request_indices: &[usize],
    lanes: &[usize],
    tokens: Vec<Vec<u32>>,
    cancelled: &effect_torch_runtime::CancellationFlag,
) -> err::Res<()> {
    let bindings = tap_bindings(source_outputs, tap_outputs)?;
    run_stateful_values(
        &replay.executable,
        &replay.generated,
        replay.schema,
        shadow,
        request_indices,
        lanes,
        &bindings,
        tokens,
        cancelled,
        true,
    )?;
    Ok(())
}

fn tap_bindings(
    source_outputs: &[value::Value],
    tap_outputs: &[usize],
) -> err::Res<Vec<value::Value>> {
    tap_outputs
        .iter()
        .map(|output| {
            source_outputs
                .get(*output)
                .cloned()
                .ok_or_else(|| "parallel replay target tap is unavailable".to_string())
        })
        .collect()
}

/// Deferred twin of [`replay_outputs`]: the target tap outputs are GPU
/// buffers produced earlier on the same stream, so the replay chains after
/// them without a host fence.
#[allow(clippy::too_many_arguments)]
fn replay_outputs_deferred(
    replay: &ReplayProgram,
    shadow: &ShadowSequences,
    source_outputs: &[value::Value],
    tap_outputs: &[usize],
    request_indices: &[usize],
    lanes: &[usize],
    tokens: Vec<Vec<u32>>,
    cancelled: &effect_torch_runtime::CancellationFlag,
    submission: &device::MetalSubmissionGuard<'_>,
) -> err::Res<executable::PendingExecution> {
    let bindings = tap_bindings(source_outputs, tap_outputs)?;
    run_stateful_values_deferred(
        &replay.executable,
        &replay.generated,
        replay.schema,
        shadow,
        request_indices,
        lanes,
        &bindings,
        tokens,
        cancelled,
        submission,
        true,
    )
}

/// Maximum number of deferred prefill invocations allowed in flight before
/// a single shared drain; bounds the workspace leases held by pendings
/// (each deferred chunk pins its own workspace until the fence).
const PREFILL_IN_FLIGHT_LIMIT: usize = 8;

fn run_parallel_prefill(
    buckets: &[PrefillBucket],
    plan: &RetainedProposerPlan,
    target_shadow: &ShadowSequences,
    proposer_shadow: &ShadowSequences,
    slots: &[usize],
    prompts: &[Vec<u32>],
    cancelled: &effect_torch_runtime::CancellationFlag,
) -> err::Res<Vec<value::Value>> {
    let mut offsets = vec![0usize; prompts.len()];
    let mut final_outputs = vec![None; prompts.len()];
    let mut pendings: Vec<executable::PendingExecution> = Vec::new();
    let mut submission: Option<device::MetalSubmissionGuard<'_>> = None;
    loop {
        let active = (0..prompts.len())
            .filter(|index| offsets[*index] < prompts[*index].len())
            .collect::<Vec<_>>();
        if active.is_empty() {
            break;
        }
        // Buckets are chosen conservatively from the longest active lane;
        // shorter lanes zero-pad to the same compiled chunk.
        let remaining = active
            .iter()
            .map(|index| prompts[*index].len() - offsets[*index])
            .max()
            .expect("active lanes are nonempty");
        let bucket = prefill_bucket(buckets, remaining);
        let time = bucket.prefill.time;
        if std::env::var_os("EFFECT_TORCH_DEBUG_PREFILL").is_some() {
            eprintln!(
                "[prefill-debug] buckets={:?} remaining={remaining} chosen={time}",
                buckets
                    .iter()
                    .map(|bucket| bucket.prefill.time)
                    .collect::<Vec<_>>()
            );
        }
        let replay = bucket
            .replay
            .as_ref()
            .expect("parallel prefill bucket has a replay program");
        let mut finishes = false;
        let tokens = active
            .iter()
            .map(|index| {
                let end = (offsets[*index] + time).min(prompts[*index].len());
                let chunk = prompts[*index][offsets[*index]..end].to_vec();
                offsets[*index] = end;
                // Mixed-length lanes share one invocation, so a chunk that
                // finishes any prompt conservatively computes the head.
                finishes |= end == prompts[*index].len();
                chunk
            })
            .collect::<Vec<_>>();
        let headless = !finishes;
        if headless {
            // Deferred: the target chunk and its proposer replay chain on
            // one submission stream; the replay binds the chunk's tap
            // outputs as GPU buffers, so no host fence is needed between
            // them.
            if submission.is_none() {
                submission = Some(device::MetalDevice::get().begin_submission()?);
            }
            let guard = submission.as_ref().expect("prefill submission opened");
            let pending = run_speculative_program_deferred(
                &bucket.prefill,
                target_shadow,
                &active,
                slots,
                tokens.clone(),
                None,
                cancelled,
                guard,
                true,
            )?;
            let outputs = pending.outputs().to_vec();
            pendings.push(pending);
            let replay_pending = replay_outputs_deferred(
                replay,
                proposer_shadow,
                &outputs,
                &plan.prefill_tap_outputs,
                &active,
                &active.iter().map(|index| slots[*index]).collect::<Vec<_>>(),
                tokens,
                cancelled,
                guard,
            )?;
            pendings.push(replay_pending);
            if pendings.len() >= PREFILL_IN_FLIGHT_LIMIT {
                executable::PendingExecution::drain_batch(std::mem::take(&mut pendings))?;
            }
            continue;
        }
        // A headed chunk forces a drain of every pending deferred chunk
        // with one fence, then runs on the synchronous path as before.
        executable::PendingExecution::drain_batch(std::mem::take(&mut pendings))?;
        drop(submission.take());
        let outputs = run_speculative_program(
            &bucket.prefill,
            target_shadow,
            &active,
            slots,
            tokens.clone(),
            None,
            cancelled,
            false,
        )?;
        replay_outputs(
            replay,
            proposer_shadow,
            &outputs,
            &plan.prefill_tap_outputs,
            &active,
            &active.iter().map(|index| slots[*index]).collect::<Vec<_>>(),
            tokens,
            cancelled,
        )?;
        for index in active {
            final_outputs[index] = Some(outputs[slots[index]].clone());
        }
    }
    // The loop always ends on a headed chunk (every prompt's final chunk
    // finishes its lane), but drain defensively if it did not.
    executable::PendingExecution::drain_batch(std::mem::take(&mut pendings))?;
    drop(submission.take());
    for (target, proposer) in target_shadow.states.iter().zip(&proposer_shadow.states) {
        let target_cursor = target
            .lock()
            .map_err(|error| format!("parallel target prefill shadow lock poisoned: {error}"))?
            .cursor;
        let proposer_cursor = proposer
            .lock()
            .map_err(|error| format!("parallel replay prefill shadow lock poisoned: {error}"))?
            .cursor;
        if target_cursor != proposer_cursor {
            return Err("parallel prefill target/proposer cursors diverged".to_string());
        }
    }
    final_outputs
        .into_iter()
        .map(|output| output.ok_or_else(|| "parallel prefill produced no logits".to_string()))
        .collect()
}

fn read_u32_tensor(value: &value::Value) -> err::Res<Vec<u32>> {
    let tensor = value.as_metal()?;
    if tensor.dtype != DType::U32 || !tensor.layout.is_contiguous() {
        return Err("ParallelBlock candidate output must be contiguous u32".to_string());
    }
    let offset = tensor.layout.offset();
    if offset + tensor.numel() > tensor.buffer.size / DType::U32.size_in_bytes() {
        return Err("ParallelBlock candidate output exceeds its buffer".to_string());
    }
    Ok((0..tensor.numel())
        .map(|index| unsafe {
            *tensor
                .buffer
                .contents_ptr()
                .cast::<u32>()
                .add(offset + index)
        })
        .collect())
}

fn sample_verification_rows(
    outputs: &[value::Value],
    packed: &PackedVerificationPlan,
    sampling: &[SamplingOptions],
    sequence_ids: &[u64],
    positions: &[u64],
    cancelled: &effect_torch_runtime::CancellationFlag,
) -> err::Res<(Vec<usize>, Vec<u32>)> {
    if sampling.iter().all(|options| options.temperature == 0.0) {
        let result = crate::sampling::sample_greedy_rows(outputs[0].as_metal()?)?;
        device::MetalDevice::get().synchronize_buffer(&result.buffer)?;
        if cancelled.is_cancelled() {
            return Err("parallel greedy sampling was cancelled".to_string());
        }
        let values = result.buffer.contents_ptr().cast::<u32>();
        let rows = result.numel() / 2;
        let mut tokens = Vec::with_capacity(rows);
        for row in 0..rows {
            let status = unsafe { *values.add(row * 2) };
            let token = unsafe { *values.add(row * 2 + 1) };
            if status == crate::sampling::STATUS_NONFINITE {
                return Err(format!("sample: logit {token} is not finite"));
            }
            if status != crate::sampling::STATUS_OK {
                return Err(format!(
                    "sample: GPU sampler returned unknown status {status}"
                ));
            }
            tokens.push(token);
        }
        return Ok((packed.row_offsets.clone(), tokens));
    }

    let logits = outputs[0].as_metal()?;
    if !logits.layout.is_contiguous() || logits.layout.rank() < 2 {
        return Err("parallel target logits must be contiguous with rank at least two".to_string());
    }
    let vocabulary = *logits
        .layout
        .shape()
        .last()
        .ok_or_else(|| "parallel target logits shape is empty".to_string())?;
    let rows = logits.numel() / vocabulary;
    if rows != packed.row_to_request.len()
        || sampling.len() != sequence_ids.len()
        || sampling.len() != positions.len()
    {
        return Err("parallel target sampling geometry is inconsistent".to_string());
    }
    let result_elements = rows
        .checked_mul(2)
        .ok_or_else(|| "parallel target sampling result size overflows".to_string())?;
    let result = runtime::metal::run::MetalTensor {
        buffer: device::MetalDevice::get().alloc_raw_checked(
            crate::sampling::required_result_allocation_bytes(result_elements)?,
        )?,
        layout: runtime::layout::Layout::contiguous(vec![result_elements]),
        dtype: DType::U32,
    };
    for (row, request) in packed.row_to_request.iter().enumerate() {
        let Some(request) = request else {
            continue;
        };
        let candidate_index = row - packed.row_offsets[*request];
        let options = SamplingOptions {
            seed: coordinate_seed(
                sampling[*request].seed,
                sequence_ids[*request],
                positions[*request] + candidate_index as u64,
                SamplingPurpose::Target,
                0,
            ),
            counter: 0,
            ..sampling[*request]
        };
        let row_logits = runtime::metal::run::MetalTensor {
            buffer: logits.buffer.clone(),
            layout: runtime::layout::Layout::new(
                vec![vocabulary],
                vec![1],
                logits.layout.offset() + row * vocabulary,
            ),
            dtype: logits.dtype,
        };
        if candidate_index == 0 {
            crate::sampling::warm_exact(&row_logits, options)?;
        }
        crate::sampling::sample_into(&row_logits, &result, row * 2, options)?;
    }
    device::MetalDevice::get().synchronize_buffer(&result.buffer)?;
    if cancelled.is_cancelled() {
        return Err("parallel target sampling was cancelled".to_string());
    }
    let values = result.buffer.contents_ptr().cast::<u32>();
    let mut tokens = vec![0; rows];
    for (row, request) in packed.row_to_request.iter().enumerate() {
        if request.is_none() {
            continue;
        }
        let status = unsafe { *values.add(row * 2) };
        let token = unsafe { *values.add(row * 2 + 1) };
        if status == crate::sampling::STATUS_NONFINITE {
            return Err(format!("sample: logit {token} is not finite"));
        }
        if status != crate::sampling::STATUS_OK {
            return Err(format!(
                "sample: GPU sampler returned unknown status {status}"
            ));
        }
        tokens[row] = token;
    }
    Ok((packed.row_offsets.clone(), tokens))
}

fn run_prefill_program(
    buckets: &[PrefillBucket],
    shadow: &ShadowSequences,
    slots: &[usize],
    prompts: &[Vec<u32>],
    cancelled: &effect_torch_runtime::CancellationFlag,
) -> err::Res<Vec<value::Value>> {
    let mut offsets = shadow
        .states
        .iter()
        .zip(prompts)
        .map(|(state, prompt)| {
            let cursor = state
                .lock()
                .map_err(|error| format!("inference prefill shadow lock: {error}"))?
                .cursor;
            if cursor >= prompt.len() {
                return Err("inference[prefill]: matched prefix includes final token".to_string());
            }
            Ok(cursor)
        })
        .collect::<err::Res<Vec<_>>>()?;
    let mut final_outputs = vec![None; prompts.len()];
    let mut pendings: Vec<executable::PendingExecution> = Vec::new();
    let mut submission: Option<device::MetalSubmissionGuard<'_>> = None;
    loop {
        let active = (0..prompts.len())
            .filter(|index| offsets[*index] < prompts[*index].len())
            .collect::<Vec<_>>();
        if active.is_empty() {
            break;
        }
        // Buckets are chosen conservatively from the longest active lane;
        // shorter lanes zero-pad to the same compiled chunk.
        let remaining = active
            .iter()
            .map(|index| prompts[*index].len() - offsets[*index])
            .max()
            .expect("active lanes are nonempty");
        let bucket = prefill_bucket(buckets, remaining);
        let time = bucket.prefill.time;
        if std::env::var_os("EFFECT_TORCH_DEBUG_PREFILL").is_some() {
            eprintln!(
                "[prefill-debug] buckets={:?} remaining={remaining} chosen={time}",
                buckets
                    .iter()
                    .map(|bucket| bucket.prefill.time)
                    .collect::<Vec<_>>()
            );
        }
        let mut finishes = false;
        let tokens = active
            .iter()
            .map(|index| {
                let end = (offsets[*index] + time).min(prompts[*index].len());
                let chunk = prompts[*index][offsets[*index]..end].to_vec();
                offsets[*index] = end;
                // Mixed-length lanes share one invocation, so a chunk that
                // finishes any prompt conservatively computes the head.
                finishes |= end == prompts[*index].len();
                chunk
            })
            .collect::<Vec<_>>();
        // Non-final chunks run headless: the compiler-marked LM-head chain
        // is skipped at dispatch, only state and tap outputs are computed.
        let headless = !finishes;
        if headless {
            // Deferred: consecutive headless chunks pipeline on one
            // submission stream with GPU-ordered state transactions; the
            // single host fence happens at the next headed chunk (or when
            // the in-flight limit bounds the leased workspaces).
            if submission.is_none() {
                submission = Some(device::MetalDevice::get().begin_submission()?);
            }
            let pending = run_speculative_program_deferred(
                &bucket.prefill,
                shadow,
                &active,
                slots,
                tokens,
                None,
                cancelled,
                submission.as_ref().expect("prefill submission opened"),
                true,
            )?;
            pendings.push(pending);
            if pendings.len() >= PREFILL_IN_FLIGHT_LIMIT {
                executable::PendingExecution::drain_batch(std::mem::take(&mut pendings))?;
            }
            continue;
        }
        // A headed chunk forces a drain of every pending deferred chunk
        // with one fence, then runs on the synchronous path as before.
        executable::PendingExecution::drain_batch(std::mem::take(&mut pendings))?;
        drop(submission.take());
        let outputs = run_speculative_program(
            &bucket.prefill,
            shadow,
            &active,
            slots,
            tokens,
            None,
            cancelled,
            false,
        )?;
        for index in active {
            final_outputs[index] = Some(outputs[slots[index]].clone());
        }
    }
    // The loop always ends on a headed chunk (every prompt's final chunk
    // finishes its lane), but drain defensively if it did not.
    executable::PendingExecution::drain_batch(std::mem::take(&mut pendings))?;
    drop(submission.take());
    final_outputs
        .into_iter()
        .map(|output| {
            output.ok_or_else(|| "inference[prefill]: prompt produced no output".to_string())
        })
        .collect()
}

fn coordinate_seed(
    seed: u64,
    sequence_id: u64,
    position: u64,
    purpose: SamplingPurpose,
    subcounter: u64,
) -> u64 {
    let mut key = seed;
    for component in [sequence_id, position, purpose as u64, subcounter] {
        key = purpose_counter(key, SamplingPurpose::Target, component);
    }
    key
}

fn run_sampled_program(
    program: &SpeculativeProgram,
    shadow: &ShadowSequences,
    slots: &[usize],
    tokens: Vec<Vec<u32>>,
    sampling: &[SamplingOptions],
    cancelled: &effect_torch_runtime::CancellationFlag,
) -> err::Res<Vec<u32>> {
    let states = shadow.states.clone();
    for state in &states {
        state
            .lock()
            .map_err(|error| format!("inference decode shadow lock: {error}"))?
            .advance = 1;
    }
    let input = token_input(
        &tokens,
        slots,
        program.schema.graph_batch,
        program.time,
        program.token_dtype,
    )?;
    let context = KvContext {
        pool: shadow.pool.clone(),
        slots: states,
        schema: program.schema,
        tokens,
        lanes: slots.to_vec(),
        packed_rows: None,
        packed_positions: None,
        publish_hashes: false,
        state_only: false,
    };
    executable::execute_stateful_sampled(
        &program.executable,
        &[input],
        &program.generated,
        cancelled,
        &context,
        &|| true,
        sampling,
    )
}

fn probabilities(
    logits: &[f64],
    options: SamplingOptions,
    cancelled: &effect_torch_runtime::CancellationFlag,
) -> err::Res<Vec<f64>> {
    effective_probabilities(
        logits.len(),
        |index| logits[index],
        options,
        || cancelled.load(Ordering::Relaxed),
    )
}

fn cut_speculative_page(page: &mut Vec<u32>, eos_tokens: &[u32], page_limit: u32) {
    if let Some(index) = page.iter().position(|token| eos_tokens.contains(token)) {
        page.truncate(index + 1);
    }
    page.truncate(page_limit as usize);
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

#[allow(clippy::too_many_arguments)]
fn execute_history_lookup_blocking(
    target: &SpeculativeProgram,
    target_shadow: &mut ShadowSequences,
    slots: &[usize],
    pending: &[u32],
    histories: &[Vec<u32>],
    sampling: &[SamplingOptions],
    config: HistoryLookupConfig,
    max_draft_tokens: usize,
    page_limits: &[u32],
    eos_tokens: &[Vec<u32>],
    cancelled: &effect_torch_runtime::CancellationFlag,
    sequence_ids: &[u64],
    positions: &[u64],
    stats: &mut SpeculativeStats,
) -> err::Res<Vec<Vec<u32>>> {
    let proposal_limits = target_shadow
        .states
        .iter()
        .zip(page_limits)
        .map(|(state, limit)| {
            let cursor = state
                .lock()
                .map_err(|error| format!("speculative shadow lock poisoned: {error}"))?
                .cursor;
            if cursor >= target.schema.max_tokens {
                return Err("executeSpeculative: exhausted target sequence cursor".to_string());
            }
            Ok(max_draft_tokens
                .min(*limit as usize - 1)
                .min(target.schema.max_tokens - cursor - 1))
        })
        .collect::<err::Res<Vec<_>>>()?;
    let draft_started = std::time::Instant::now();
    let candidates = histories
        .iter()
        .zip(&proposal_limits)
        .map(|(history, limit)| {
            history_lookup(
                history,
                config.min_match_tokens,
                config.max_match_tokens,
                *limit,
            )
        })
        .collect::<Vec<_>>();
    stats.proposed = candidates.iter().map(Vec::len).sum();
    stats.draft_nanos = draft_started.elapsed().as_nanos().max(1) as u64;

    let verify_tokens = pending
        .iter()
        .zip(&candidates)
        .map(|(pending, candidates)| {
            std::iter::once(*pending)
                .chain(candidates.iter().copied())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let cursors = target_shadow
        .states
        .iter()
        .map(|state| {
            state
                .lock()
                .map(|state| state.cursor)
                .map_err(|error| format!("speculative shadow lock poisoned: {error}"))
        })
        .collect::<err::Res<Vec<_>>>()?;
    let packed = packed_verification_plan(
        target.schema.graph_batch,
        target
            .packed_rows_per_sequence
            .ok_or_else(|| "executeSpeculative: target packed layout is missing".to_string())?,
        &cursors,
        &verify_tokens,
    )?;
    let verify_started = std::time::Instant::now();
    let outputs = run_speculative_program(
        target,
        target_shadow,
        &(0..pending.len()).collect::<Vec<_>>(),
        slots,
        verify_tokens,
        Some(&packed),
        cancelled,
        false,
    )?;
    stats.verification_nanos = verify_started.elapsed().as_nanos().max(1) as u64;
    let mut pages = Vec::with_capacity(pending.len());
    for lane in 0..pending.len() {
        let mut page = Vec::with_capacity(candidates[lane].len() + 1);
        let mut accepted = 0;
        let mut rejected = false;
        for (candidate_index, &candidate) in candidates[lane].iter().enumerate() {
            let (matrix, _) = executable::route_leading_row(
                &outputs[0],
                packed.row_offsets[lane] + candidate_index,
            )?;
            let (logits, _) = executable::route_leading_row(&matrix, 0)?;
            let target = sample_blocking(
                &logits,
                SamplingOptions {
                    seed: coordinate_seed(
                        sampling[lane].seed,
                        sequence_ids[lane],
                        positions[lane] + candidate_index as u64,
                        SamplingPurpose::Target,
                        0,
                    ),
                    counter: 0,
                    ..sampling[lane]
                },
                cancelled,
            )
            .map_err(|error| error.reason)?;
            if target == candidate {
                page.push(candidate);
                accepted += 1;
            } else {
                page.push(target);
                rejected = true;
                break;
            }
        }
        if !rejected {
            let bonus_index = candidates[lane].len();
            let (matrix, _) =
                executable::route_leading_row(&outputs[0], packed.row_offsets[lane] + bonus_index)?;
            let (logits, _) = executable::route_leading_row(&matrix, 0)?;
            page.push(
                sample_blocking(
                    &logits,
                    SamplingOptions {
                        seed: coordinate_seed(
                            sampling[lane].seed,
                            sequence_ids[lane],
                            positions[lane] + bonus_index as u64,
                            SamplingPurpose::Target,
                            0,
                        ),
                        counter: 0,
                        ..sampling[lane]
                    },
                    cancelled,
                )
                .map_err(|error| error.reason)?,
            );
        }
        cut_speculative_page(&mut page, &eos_tokens[lane], page_limits[lane]);
        stats.accepted[lane] = accepted.min(page.len());
        pages.push(page);
    }
    if cancelled.load(Ordering::Relaxed) {
        return Err("operation aborted".to_string());
    }
    Ok(pages)
}

#[allow(clippy::too_many_arguments)]
fn execute_parallel_blocking(
    target: &SpeculativeProgram,
    replay: &ReplayProgram,
    plan: &RetainedProposerPlan,
    target_shadow: &mut ShadowSequences,
    proposal_shadow: &ShadowSequences,
    replay_shadow: &ShadowSequences,
    slots: &[usize],
    pending: &[u32],
    sampling: &[SamplingOptions],
    max_draft_tokens: usize,
    page_limits: &[u32],
    eos_tokens: &[Vec<u32>],
    cancelled: &effect_torch_runtime::CancellationFlag,
    sequence_ids: &[u64],
    positions: &[u64],
    stats: &mut SpeculativeStats,
) -> err::Res<Vec<Vec<u32>>> {
    let stage = &plan.stages[0];
    let stage_schema = stage
        .schema
        .ok_or_else(|| "ParallelBlock stage state is unavailable".to_string())?;
    let proposal_limits = target_shadow
        .states
        .iter()
        .zip(page_limits)
        .map(|(state, limit)| {
            let cursor = state
                .lock()
                .map_err(|error| format!("parallel target shadow lock poisoned: {error}"))?
                .cursor;
            if cursor >= target.schema.max_tokens {
                return Err("ParallelBlock target cursor is exhausted".to_string());
            }
            Ok(max_draft_tokens
                .min(*limit as usize - 1)
                .min(target.schema.max_tokens - cursor - 1))
        })
        .collect::<err::Res<Vec<_>>>()?;
    let pending_value = token_vector(pending, slots, stage_schema.graph_batch, target.token_dtype)?;
    let mut dynamic = HashMap::new();
    dynamic.insert("PendingTokens".to_string(), pending_value);
    let mut bindings = plan.schema.stages[0]
        .inputs
        .iter()
        .map(|binding| {
            resolve_routed_value(plan, &binding.value, &[], &dynamic, &[], 0)
                .map(|value| (binding.slot, value))
        })
        .collect::<err::Res<Vec<_>>>()?;
    bindings.sort_by_key(|(slot, _)| *slot);
    let bindings = bindings
        .into_iter()
        .map(|(_, value)| value)
        .collect::<Vec<_>>();
    let proposal_active = (0..pending.len())
        .filter(|index| proposal_limits[*index] > 0)
        .collect::<Vec<_>>();
    let trained_draft_tokens = plan.schema.trained_max_rows as usize;
    let proposal_tokens = proposal_active
        .iter()
        .map(|index| vec![pending[*index]; trained_draft_tokens + 1])
        .collect::<Vec<_>>();
    let started = std::time::Instant::now();
    let (dense_candidates, draft_probabilities) = if proposal_active.is_empty() {
        (vec![0; stage_schema.batch * trained_draft_tokens], None)
    } else {
        let proposal_outputs = run_stateful_values(
            &stage.executable,
            &stage.generated,
            stage_schema,
            proposal_shadow,
            &proposal_active,
            &proposal_active
                .iter()
                .map(|index| slots[*index])
                .collect::<Vec<_>>(),
            &bindings,
            proposal_tokens,
            cancelled,
            false,
        )?;
        if proposal_outputs.len() == 2 {
            let tensor = contiguous_f32_tensor(
                &proposal_outputs[1],
                "ParallelBlock proposal probabilities",
            )?;
            let probabilities = f32_values(&tensor);
            let mut sampled = vec![0; stage_schema.batch * trained_draft_tokens];
            for lane in &proposal_active {
                for row in 0..proposal_limits[*lane] {
                    let start = (slots[*lane] * trained_draft_tokens + row) * target.vocabulary;
                    let q = &probabilities[start..start + target.vocabulary];
                    sampled[slots[*lane] * trained_draft_tokens + row] = sample_f32_at(
                        q,
                        InferenceSampling {
                            temperature: sampling[*lane].temperature,
                            top_k: sampling[*lane].top_k,
                            top_p: sampling[*lane].top_p,
                            seed: sampling[*lane].seed,
                        },
                        sequence_ids[*lane],
                        positions[*lane] + row as u64,
                        SamplingPurpose::Proposal,
                        0,
                    );
                }
            }
            (sampled, Some(tensor))
        } else {
            (read_u32_tensor(&proposal_outputs[0])?, None)
        }
    };
    let candidates = slots
        .iter()
        .zip(&proposal_limits)
        .map(|(slot, limit)| {
            dense_candidates[*slot * trained_draft_tokens..*slot * trained_draft_tokens + *limit]
                .to_vec()
        })
        .collect::<Vec<_>>();
    stats.proposed = candidates.iter().map(Vec::len).sum();
    stats.draft_nanos = started.elapsed().as_nanos().max(1) as u64;

    let verify_tokens = pending
        .iter()
        .zip(&candidates)
        .map(|(pending, candidates)| {
            std::iter::once(*pending)
                .chain(candidates.iter().copied())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let cursors = target_shadow
        .states
        .iter()
        .map(|state| {
            state
                .lock()
                .map(|state| state.cursor)
                .map_err(|error| format!("parallel target shadow lock poisoned: {error}"))
        })
        .collect::<err::Res<Vec<_>>>()?;
    let packed = packed_verification_plan(
        target.schema.graph_batch,
        target
            .packed_rows_per_sequence
            .ok_or_else(|| "ParallelBlock verifier packed layout is missing".to_string())?,
        &cursors,
        &verify_tokens,
    )?;
    let verify_started = std::time::Instant::now();
    let outputs = run_speculative_program(
        target,
        target_shadow,
        &(0..pending.len()).collect::<Vec<_>>(),
        slots,
        verify_tokens,
        Some(&packed),
        cancelled,
        false,
    )?;
    stats.verification_nanos = verify_started.elapsed().as_nanos().max(1) as u64;
    let target_samples = if draft_probabilities.is_none() {
        Some(sample_verification_rows(
            &outputs,
            &packed,
            sampling,
            sequence_ids,
            positions,
            cancelled,
        )?)
    } else {
        None
    };
    let target_logits = if draft_probabilities.is_some() {
        Some(contiguous_f32_tensor(
            &outputs[0],
            "ParallelBlock target logits",
        )?)
    } else {
        None
    };
    let mut pages = Vec::with_capacity(pending.len());
    for lane in 0..pending.len() {
        let mut page = Vec::with_capacity(candidates[lane].len() + 1);
        let mut rejected = false;
        let mut accepted = 0;
        for (candidate_index, &candidate) in candidates[lane].iter().enumerate() {
            let sampled = if let Some(proposal_probabilities) = &draft_probabilities {
                let target_start = (packed.row_offsets[lane] + candidate_index) * target.vocabulary;
                let target_values = f32_values(
                    target_logits
                        .as_ref()
                        .expect("probability-routed ParallelBlock retains target logits"),
                );
                let p = probabilities_f32(
                    &target_values[target_start..target_start + target.vocabulary],
                    sampling[lane],
                    cancelled,
                )?;
                let proposal_start =
                    (slots[lane] * trained_draft_tokens + candidate_index) * target.vocabulary;
                let proposal_values = f32_values(proposal_probabilities);
                let q = &proposal_values[proposal_start..proposal_start + target.vocabulary];
                let accept = if sampling[lane].temperature == 0.0 {
                    p[candidate as usize] > 0.0
                } else {
                    let probability =
                        (p[candidate as usize] / q[candidate as usize] as f64).min(1.0);
                    random_unit_at(sampling_coordinate(
                        sampling[lane].seed,
                        sequence_ids[lane],
                        positions[lane] + candidate_index as u64,
                        SamplingPurpose::Accept,
                        0,
                    )) < probability
                };
                if accept {
                    candidate
                } else {
                    let residual = p
                        .iter()
                        .zip(q)
                        .map(|(p, q)| (p - *q as f64).max(0.0))
                        .collect::<Vec<_>>();
                    sample_at(
                        &residual,
                        InferenceSampling {
                            temperature: sampling[lane].temperature,
                            top_k: sampling[lane].top_k,
                            top_p: sampling[lane].top_p,
                            seed: sampling[lane].seed,
                        },
                        sequence_ids[lane],
                        positions[lane] + candidate_index as u64,
                        SamplingPurpose::Residual,
                        0,
                    )
                }
            } else if let Some((offsets, tokens)) = &target_samples {
                tokens[offsets[lane] + candidate_index]
            } else {
                let (matrix, _) = executable::route_leading_row(
                    &outputs[0],
                    packed.row_offsets[lane] + candidate_index,
                )?;
                let (logits, _) = executable::route_leading_row(&matrix, 0)?;
                sample_blocking(
                    &logits,
                    SamplingOptions {
                        seed: coordinate_seed(
                            sampling[lane].seed,
                            sequence_ids[lane],
                            positions[lane] + candidate_index as u64,
                            SamplingPurpose::Target,
                            0,
                        ),
                        counter: 0,
                        ..sampling[lane]
                    },
                    cancelled,
                )
                .map_err(|error| error.reason)?
            };
            if sampled == candidate {
                page.push(candidate);
                accepted += 1;
            } else {
                page.push(sampled);
                rejected = true;
                break;
            }
        }
        if !rejected {
            let bonus = candidates[lane].len();
            page.push(if draft_probabilities.is_some() {
                let target_start = (packed.row_offsets[lane] + bonus) * target.vocabulary;
                let target_values = f32_values(
                    target_logits
                        .as_ref()
                        .expect("probability-routed ParallelBlock retains target logits"),
                );
                let p = probabilities_f32(
                    &target_values[target_start..target_start + target.vocabulary],
                    sampling[lane],
                    cancelled,
                )?;
                sample_at(
                    &p,
                    InferenceSampling {
                        temperature: sampling[lane].temperature,
                        top_k: sampling[lane].top_k,
                        top_p: sampling[lane].top_p,
                        seed: sampling[lane].seed,
                    },
                    sequence_ids[lane],
                    positions[lane] + bonus as u64,
                    SamplingPurpose::Target,
                    0,
                )
            } else if let Some((offsets, tokens)) = &target_samples {
                tokens[offsets[lane] + bonus]
            } else {
                let (matrix, _) =
                    executable::route_leading_row(&outputs[0], packed.row_offsets[lane] + bonus)?;
                let (logits, _) = executable::route_leading_row(&matrix, 0)?;
                sample_blocking(
                    &logits,
                    SamplingOptions {
                        seed: coordinate_seed(
                            sampling[lane].seed,
                            sequence_ids[lane],
                            positions[lane] + bonus as u64,
                            SamplingPurpose::Target,
                            0,
                        ),
                        counter: 0,
                        ..sampling[lane]
                    },
                    cancelled,
                )
                .map_err(|error| error.reason)?
            });
        }
        cut_speculative_page(&mut page, &eos_tokens[lane], page_limits[lane]);
        stats.accepted[lane] = accepted.min(page.len());
        pages.push(page);
    }
    let consumed = pending
        .iter()
        .zip(&pages)
        .map(|(pending, page)| {
            std::iter::once(*pending)
                .chain(page.iter().copied())
                .take(page.len())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    replay_outputs(
        replay,
        replay_shadow,
        &outputs,
        &plan.verify_tap_outputs,
        &(0..pending.len()).collect::<Vec<_>>(),
        &(0..pending.len()).collect::<Vec<_>>(),
        consumed,
        cancelled,
    )?;
    for (lane, state) in replay_shadow.states.iter().enumerate() {
        let cursor = state
            .lock()
            .map_err(|error| format!("parallel replay shadow lock poisoned: {error}"))?
            .cursor;
        if cursor != cursors[lane] + pages[lane].len() {
            return Err("ParallelBlock replay did not preserve target cursor parity".to_string());
        }
    }
    if cancelled.load(Ordering::Relaxed) {
        return Err("operation aborted".to_string());
    }
    Ok(pages)
}

#[allow(clippy::too_many_arguments)]
fn execute_speculative_blocking(
    target: &SpeculativeProgram,
    proposer: &SpeculativeProgram,
    target_shadow: &mut ShadowSequences,
    proposer_shadow: &mut ShadowSequences,
    slots: &[usize],
    pending: &[u32],
    sampling: &[SamplingOptions],
    max_draft_tokens: usize,
    page_limits: &[u32],
    eos_tokens: &[Vec<u32>],
    cancelled: &effect_torch_runtime::CancellationFlag,
    coordinates: Option<(&[u64], &[u64])>,
    mut stats: Option<&mut SpeculativeStats>,
) -> err::Res<Vec<Vec<u32>>> {
    let count = pending.len();
    let proposal_limits = page_limits
        .iter()
        .enumerate()
        .map(|(lane, limit)| {
            let target_cursor = target_shadow.states[lane]
                .lock()
                .map_err(|error| format!("speculative shadow lock poisoned: {error}"))?
                .cursor;
            let proposer_cursor = proposer_shadow.states[lane]
                .lock()
                .map_err(|error| format!("speculative shadow lock poisoned: {error}"))?
                .cursor;
            if target_cursor != proposer_cursor
                || target_cursor >= target.schema.max_tokens
                || proposer_cursor >= proposer.schema.max_tokens
            {
                return Err(
                    "executeSpeculative: incompatible or exhausted sequence cursors".to_string(),
                );
            }
            let remaining = (target.schema.max_tokens - target_cursor)
                .min(proposer.schema.max_tokens - proposer_cursor);
            Ok(max_draft_tokens.min(*limit as usize - 1).min(remaining - 1))
        })
        .collect::<err::Res<Vec<_>>>()?;
    let mut candidates = vec![Vec::<u32>::new(); count];
    let mut proposal_probabilities = vec![Vec::<Vec<f64>>::new(); count];
    let draft_started = std::time::Instant::now();
    for candidate_index in 0..proposal_limits.iter().copied().max().unwrap_or(0) {
        let active = (0..count)
            .filter(|lane| candidate_index < proposal_limits[*lane])
            .collect::<Vec<_>>();
        let tokens = active
            .iter()
            .map(|lane| {
                vec![if candidate_index == 0 {
                    pending[*lane]
                } else {
                    candidates[*lane][candidate_index - 1]
                }]
            })
            .collect::<Vec<_>>();
        let outputs = run_speculative_program(
            proposer,
            proposer_shadow,
            &active,
            slots,
            tokens,
            None,
            cancelled,
            false,
        )?;
        for lane in active {
            let logits = read_float_tensor(&outputs[slots[lane]])?;
            let mut options = sampling[lane];
            options.counter = purpose_counter(
                options.counter + candidate_index as u64,
                SamplingPurpose::Proposal,
                0,
            );
            let q = probabilities(&logits, options, cancelled)?;
            let token = if let Some((sequence_ids, positions)) = coordinates {
                sample_at(
                    &q,
                    InferenceSampling {
                        temperature: options.temperature,
                        top_k: options.top_k,
                        top_p: options.top_p,
                        seed: options.seed,
                    },
                    sequence_ids[lane],
                    positions[lane] + candidate_index as u64,
                    SamplingPurpose::Proposal,
                    0,
                )
            } else {
                sample_probabilities(&q, options.seed, options.counter, || {
                    cancelled.load(Ordering::Relaxed)
                })?
            };
            candidates[lane].push(token);
            proposal_probabilities[lane].push(q);
        }
    }
    if let Some(stats) = stats.as_deref_mut() {
        stats.proposed = proposal_limits.iter().sum();
        stats.draft_nanos = draft_started.elapsed().as_nanos().max(1) as u64;
    }

    let all = (0..count).collect::<Vec<_>>();
    let verify_tokens = (0..count)
        .map(|lane| {
            let mut row = Vec::with_capacity(candidates[lane].len() + 1);
            row.push(pending[lane]);
            row.extend_from_slice(&candidates[lane]);
            row
        })
        .collect::<Vec<_>>();
    let target_cursors = target_shadow
        .states
        .iter()
        .map(|state| {
            state
                .lock()
                .map(|state| state.cursor)
                .map_err(|error| format!("speculative shadow lock poisoned: {error}"))
        })
        .collect::<err::Res<Vec<_>>>()?;
    let packed = packed_verification_plan(
        target.schema.graph_batch,
        target
            .packed_rows_per_sequence
            .ok_or_else(|| "executeSpeculative: target packed layout is missing".to_string())?,
        &target_cursors,
        &verify_tokens,
    )?;
    let verify_started = std::time::Instant::now();
    let target_outputs = run_speculative_program(
        target,
        target_shadow,
        &all,
        slots,
        verify_tokens,
        Some(&packed),
        cancelled,
        false,
    )?;
    if let Some(stats) = stats.as_deref_mut() {
        stats.verification_nanos = verify_started.elapsed().as_nanos().max(1) as u64;
    }
    let target_logits = read_float_tensor(&target_outputs[0])?;
    let row = target.vocabulary;
    let mut pages = Vec::with_capacity(count);
    for lane in 0..count {
        let mut page = Vec::new();
        let mut rejected = false;
        let mut accepted = 0;
        for candidate_index in 0..candidates[lane].len() {
            let start = (packed.row_offsets[lane] + candidate_index) * row;
            let p = probabilities(
                &target_logits[start..start + row],
                sampling[lane],
                cancelled,
            )?;
            let q = &proposal_probabilities[lane][candidate_index];
            let token = candidates[lane][candidate_index];
            let accept = if sampling[lane].temperature == 0.0 {
                p[token as usize] > 0.0
            } else {
                let probability = (p[token as usize] / q[token as usize]).min(1.0);
                if let Some((sequence_ids, positions)) = coordinates {
                    random_unit_at(sampling_coordinate(
                        sampling[lane].seed,
                        sequence_ids[lane],
                        positions[lane] + candidate_index as u64,
                        SamplingPurpose::Accept,
                        0,
                    )) < probability
                } else {
                    random_unit(
                        sampling[lane].seed,
                        purpose_counter(
                            sampling[lane].counter + candidate_index as u64,
                            SamplingPurpose::Accept,
                            0,
                        ),
                    ) < probability
                }
            };
            if accept {
                page.push(token);
                accepted += 1;
                continue;
            }
            let residual = p
                .iter()
                .zip(q)
                .map(|(p, q)| (p - q).max(0.0))
                .collect::<Vec<_>>();
            page.push(if let Some((sequence_ids, positions)) = coordinates {
                sample_at(
                    &residual,
                    InferenceSampling {
                        temperature: sampling[lane].temperature,
                        top_k: sampling[lane].top_k,
                        top_p: sampling[lane].top_p,
                        seed: sampling[lane].seed,
                    },
                    sequence_ids[lane],
                    positions[lane] + candidate_index as u64,
                    SamplingPurpose::Residual,
                    0,
                )
            } else {
                sample_probabilities(
                    &residual,
                    sampling[lane].seed,
                    purpose_counter(
                        sampling[lane].counter + candidate_index as u64,
                        SamplingPurpose::Residual,
                        0,
                    ),
                    || cancelled.load(Ordering::Relaxed),
                )?
            });
            rejected = true;
            break;
        }
        if !rejected {
            let bonus_index = candidates[lane].len();
            let start = (packed.row_offsets[lane] + bonus_index) * row;
            let p = probabilities(
                &target_logits[start..start + row],
                sampling[lane],
                cancelled,
            )?;
            page.push(if let Some((sequence_ids, positions)) = coordinates {
                sample_at(
                    &p,
                    InferenceSampling {
                        temperature: sampling[lane].temperature,
                        top_k: sampling[lane].top_k,
                        top_p: sampling[lane].top_p,
                        seed: sampling[lane].seed,
                    },
                    sequence_ids[lane],
                    positions[lane] + bonus_index as u64,
                    SamplingPurpose::Target,
                    0,
                )
            } else {
                sample_probabilities(
                    &p,
                    sampling[lane].seed,
                    purpose_counter(
                        sampling[lane].counter + bonus_index as u64,
                        SamplingPurpose::Target,
                        0,
                    ),
                    || cancelled.load(Ordering::Relaxed),
                )?
            });
        }
        cut_speculative_page(&mut page, &eos_tokens[lane], page_limits[lane]);
        if let Some(stats) = stats.as_deref_mut() {
            stats.accepted[lane] = accepted.min(page.len());
        }
        pages.push(page);
    }

    // A full chain has not consumed its final candidate in the proposer. The
    // zero-draft case similarly has not consumed the old pending token.
    let mut catchup = Vec::new();
    let mut catchup_tokens = Vec::new();
    for lane in 0..count {
        let old_cursor = proposer_shadow.states[lane]
            .lock()
            .map_err(|error| format!("speculative shadow lock poisoned: {error}"))?
            .cursor
            .saturating_sub(proposal_limits[lane]);
        let desired = old_cursor + pages[lane].len();
        let current = proposer_shadow.states[lane]
            .lock()
            .map_err(|error| format!("speculative shadow lock poisoned: {error}"))?
            .cursor;
        if desired > current {
            let consumed = std::iter::once(pending[lane])
                .chain(pages[lane].iter().copied())
                .take(pages[lane].len())
                .collect::<Vec<_>>();
            catchup.push(lane);
            catchup_tokens.push(vec![consumed[current - old_cursor]]);
        }
    }
    if !catchup.is_empty() {
        run_speculative_program(
            proposer,
            proposer_shadow,
            &catchup,
            slots,
            catchup_tokens,
            None,
            cancelled,
            false,
        )?;
    }
    if cancelled.load(Ordering::Relaxed) {
        return Err("operation aborted".to_string());
    }
    Ok(pages)
}

#[derive(Default)]
struct SpeculativeStats {
    proposed: usize,
    accepted: Vec<usize>,
    draft_nanos: u64,
    verification_nanos: u64,
}

fn publish_speculative_states(
    shadow: &mut ShadowSequences,
    canonical: &[Arc<Mutex<SeqState>>],
    pending: &[u32],
    pages: &[Vec<u32>],
) -> u64 {
    let mut rolled_back = 0;
    for (index, (shadow_state, canonical_state)) in shadow.states.iter().zip(canonical).enumerate()
    {
        let mut provisional = shadow_state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut canonical = canonical_state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let consumed = std::iter::once(pending[index])
            .chain(pages[index].iter().copied())
            .take(pages[index].len())
            .collect::<Vec<_>>();
        let desired_cursor = canonical.cursor + consumed.len();
        let retained = desired_cursor
            .div_ceil(shadow.pool.block_size)
            .saturating_sub(provisional.head);
        for block in provisional.blocks.drain(retained..) {
            if !canonical.blocks.contains(&block) {
                rolled_back += 1;
            }
            shadow.pool.unref_block(block);
        }
        let old_blocks = std::mem::replace(
            &mut canonical.blocks,
            std::mem::take(&mut provisional.blocks),
        );
        canonical.head = provisional.head;
        canonical.advance = 0;
        canonical.note_tokens(&shadow.pool, &consumed);
        canonical.cursor = desired_cursor;
        for block in old_blocks {
            shadow.pool.unref_block(block);
        }
    }
    rolled_back
}

// Cohesive inference owns lane assignment, paired state, policy and receipts in
// one native object. The lower-level stateful execution API above remains a
// separate primitive for callers that need logits or direct KV control.
#[derive(Clone, Copy)]
struct InferenceSampling {
    temperature: f64,
    top_k: Option<usize>,
    top_p: f64,
    seed: u64,
}

#[napi(object)]
pub struct NativeInferenceSamplingOptions {
    pub temperature: f64,
    pub top_k: u32,
    pub top_p: f64,
    pub seed: BigInt,
}

#[napi(object)]
pub struct NativeInferenceSamplingOverride {
    pub temperature: Option<f64>,
    pub top_k: Option<u32>,
    pub top_p: Option<f64>,
    pub seed: Option<BigInt>,
}

fn inference_sampling(value: NativeInferenceSamplingOptions) -> Result<InferenceSampling> {
    let (negative, seed, lossless) = value.seed.get_u64();
    if negative || !lossless {
        return Err(Error::new(
            Status::InvalidArg,
            "inference: seed must be an unsigned 64-bit integer",
        ));
    }
    let options = sampling_options(NativeSamplingOptions {
        temperature: value.temperature,
        top_k: value.top_k as f64,
        top_p: value.top_p,
        seed: 0.0,
        counter: 0.0,
    })?;
    Ok(InferenceSampling {
        temperature: options.temperature,
        top_k: options.top_k,
        top_p: options.top_p,
        seed,
    })
}

fn inference_sampling_override(
    base: InferenceSampling,
    value: NativeInferenceSamplingOverride,
) -> Result<InferenceSampling> {
    inference_sampling(NativeInferenceSamplingOptions {
        temperature: value.temperature.unwrap_or(base.temperature),
        top_k: value.top_k.unwrap_or(base.top_k.unwrap_or(0) as u32),
        top_p: value.top_p.unwrap_or(base.top_p),
        seed: value.seed.unwrap_or_else(|| bigint(base.seed)),
    })
}

fn fingerprint_word(hash: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *hash ^= byte as u64;
        *hash = hash.wrapping_mul(HASH_PRIME);
    }
}

fn fingerprint_sampling(hash: &mut u64, value: InferenceSampling) {
    fingerprint_word(hash, value.temperature.to_bits());
    fingerprint_word(hash, value.top_k.map_or(u64::MAX, |value| value as u64));
    fingerprint_word(hash, value.top_p.to_bits());
    fingerprint_word(hash, value.seed);
}

fn add_fingerprint(
    prompts: &[Vec<u32>],
    sampling: &[InferenceSampling],
    max_tokens: &[Option<u32>],
    eos_tokens: &[Vec<u32>],
) -> u64 {
    let mut hash = HASH_SEED;
    fingerprint_word(&mut hash, 1);
    fingerprint_word(&mut hash, prompts.len() as u64);
    for index in 0..prompts.len() {
        fingerprint_word(&mut hash, prompts[index].len() as u64);
        for &token in &prompts[index] {
            fingerprint_word(&mut hash, token as u64);
        }
        fingerprint_sampling(&mut hash, sampling[index]);
        fingerprint_word(&mut hash, max_tokens[index].map_or(u64::MAX, u64::from));
        fingerprint_word(&mut hash, eos_tokens[index].len() as u64);
        for &token in &eos_tokens[index] {
            fingerprint_word(&mut hash, token as u64);
        }
    }
    hash
}

fn round_fingerprint(ids: &[u64], sampling: &[InferenceSampling]) -> u64 {
    let mut hash = HASH_SEED;
    fingerprint_word(&mut hash, 2);
    fingerprint_word(&mut hash, ids.len() as u64);
    for (&id, &sampling) in ids.iter().zip(sampling) {
        fingerprint_word(&mut hash, id);
        fingerprint_sampling(&mut hash, sampling);
    }
    hash
}

fn record_inference_failure(
    diagnostics: &InferenceDiagnosticsState,
    error: &Error,
    fallback: &str,
) {
    let reason = &error.reason;
    let phase = reason
        .split_once("inference[")
        .and_then(|(_, rest)| rest.split_once(']'))
        .map_or(fallback, |(phase, _)| phase);
    *diagnostics
        .last_failure_phase
        .lock()
        .unwrap_or_else(|error| error.into_inner()) = Some(phase.to_string());
}

fn inference_options(value: InferenceSampling) -> SamplingOptions {
    SamplingOptions {
        temperature: value.temperature,
        top_k: value.top_k,
        top_p: value.top_p,
        seed: value.seed,
        counter: 0,
    }
}

fn bigint(value: u64) -> BigInt {
    BigInt {
        sign_bit: false,
        words: vec![value],
    }
}

fn inference_program(
    program: &Executable,
    phase: &str,
    time_one: bool,
    allow_ephemeral_outputs: bool,
) -> Result<SpeculativeProgram> {
    let state = program.state.as_ref().ok_or_else(|| {
        Error::new(
            Status::InvalidArg,
            format!("inference[{phase}]: executable must be stateful"),
        )
    })?;
    if state.packed_rows_per_sequence.is_some() {
        return Err(Error::new(
            Status::InvalidArg,
            format!("inference[{phase}]: executable must use dense lanes"),
        ));
    }
    let tensor_slots = program
        .inner
        .slots
        .iter()
        .enumerate()
        .filter(|(index, slot)| {
            !slot.scalar && !(state.cursor_tensor && *index as u32 == state.cursor_slot)
        })
        .map(|(_, slot)| slot)
        .collect::<Vec<_>>();
    if tensor_slots.len() != 1
        || !matches!(tensor_slots[0].dtype, DType::U32 | DType::I64)
        || tensor_slots[0].shape.len() != 2
        || tensor_slots[0].shape[0] != state.schema.graph_batch
        || tensor_slots[0].shape[1] == 0
        || (time_one && tensor_slots[0].shape[1] != 1)
    {
        return Err(Error::new(
            Status::InvalidArg,
            format!("inference[{phase}]: expected one compatible [B, T] token input"),
        ));
    }
    let outputs = &program.inner.executable.program.outputs;
    let values = &program.inner.executable.program.values;
    if outputs.len() < state.schema.batch
        || (!allow_ephemeral_outputs && outputs.len() != state.schema.batch)
    {
        return Err(Error::new(
            Status::InvalidArg,
            format!("inference[{phase}]: expected one last-token logits row per lane"),
        ));
    }
    let first = &values[outputs[0].index()];
    if first.shape.len() != 1
        || first.shape[0] == 0
        || first.shape[0] > MAX_SAMPLING_VOCABULARY
        || !matches!(first.dtype, DType::F16 | DType::BF16 | DType::F32)
        || outputs.iter().take(state.schema.batch).any(|output| {
            let declaration = &values[output.index()];
            declaration.shape != first.shape || declaration.dtype != first.dtype
        })
    {
        return Err(Error::new(
            Status::InvalidArg,
            format!("inference[{phase}]: invalid last-token logits outputs"),
        ));
    }
    Ok(SpeculativeProgram {
        executable: program.inner.executable.clone(),
        generated: program.inner.generated_bindings.clone(),
        schema: state.schema,
        batch: state.schema.batch,
        time: tensor_slots[0].shape[1],
        vocabulary: first.shape[0],
        token_dtype: tensor_slots[0].dtype,
        packed_rows_per_sequence: None,
    })
}

fn compatible_pool(program: &SpeculativeProgram, pool: &Arc<PoolInner>) -> bool {
    pool.max_tokens == program.schema.max_tokens
        && pool.block_size == program.schema.block_size
        && pool.dtype == program.schema.kv_dtype
        && pool.k.len() == program.schema.layers
        && pool.kv_heads == program.schema.kv_heads
        && pool.head_dim == program.schema.head_dim
        && pool.kda == program.schema.kda
        && pool.conv == program.schema.conv
}

fn same_inference_state(left: KvStateSchema, right: KvStateSchema) -> bool {
    left.max_tokens == right.max_tokens
        && left.block_size == right.block_size
        && left.kv_dtype == right.kv_dtype
        && left.window == right.window
        && left.batch == right.batch
        && left.graph_batch == right.graph_batch
        && left.layers == right.layers
        && left.kv_heads == right.kv_heads
        && left.head_dim == right.head_dim
        && left.kda == right.kda
        && left.conv == right.conv
}

fn validate_replay_program(
    replay: &Executable,
    pool: &Arc<PoolInner>,
    batch: usize,
    target: &Executable,
    tap_outputs: &[usize],
) -> Result<ReplayProgram> {
    let state = replay
        .state
        .as_ref()
        .ok_or_else(|| plan_error("ParallelBlock replay executable must be stateful"))?;
    let slots = replay
        .inner
        .slots
        .iter()
        .enumerate()
        .filter(|(index, slot)| {
            !slot.scalar && !(state.cursor_tensor && *index as u32 == state.cursor_slot)
        })
        .map(|(_, slot)| slot)
        .collect::<Vec<_>>();
    let outputs = &target.inner.executable.program.outputs;
    let values = &target.inner.executable.program.values;
    if state.schema.batch != batch
        || state.packed_rows_per_sequence.is_some()
        || !compatible_pool_schema(state.schema, pool)
        || slots.len() != tap_outputs.len()
        || replay.inner.slots.iter().any(|slot| slot.scalar)
    {
        return Err(plan_error(
            "ParallelBlock replay state or input count is invalid",
        ));
    }
    for (slot, &physical) in slots.iter().zip(tap_outputs) {
        let source = outputs
            .get(physical)
            .and_then(|output| values.get(output.index()))
            .ok_or_else(|| plan_error("ParallelBlock replay tap output is missing"))?;
        if slot.dtype != source.dtype || slot.shape.as_slice() != source.shape.as_ref() {
            return Err(plan_error(
                "ParallelBlock replay input does not match its physical target tap",
            ));
        }
    }
    Ok(ReplayProgram {
        executable: replay.inner.executable.clone(),
        generated: replay.inner.generated_bindings.clone(),
        schema: state.schema,
    })
}

fn compatible_pool_schema(schema: KvStateSchema, pool: &Arc<PoolInner>) -> bool {
    pool.max_tokens == schema.max_tokens
        && pool.block_size == schema.block_size
        && pool.dtype == schema.kv_dtype
        && pool.k.len() == schema.layers
        && pool.kv_heads == schema.kv_heads
        && pool.head_dim == schema.head_dim
        && pool.kda == schema.kda
        && pool.conv == schema.conv
}

fn read_prompt(tensor: &NativeTensor, dtype: DType, max_time: usize) -> Result<Vec<u32>> {
    let value = tensor.val_cloned()?;
    if value.dtype() != dtype || value.shape().len() != 2 || value.shape()[0] != 1 {
        return Err(Error::new(
            Status::InvalidArg,
            "inference[prefill]: prompt must be one token row with the artifact dtype",
        ));
    }
    let length = value.shape()[1];
    if length == 0 || length > max_time {
        return Err(Error::new(
            Status::InvalidArg,
            format!("inference[prefill]: prompt length must be in 1..={max_time}"),
        ));
    }
    let tensor = value.as_metal().map_err(to_napi_err)?;
    if !tensor.layout.is_contiguous() || tensor.layout.offset() != 0 {
        return Err(Error::new(
            Status::InvalidArg,
            "inference[prefill]: prompt must be contiguous",
        ));
    }
    let pointer = tensor.buffer.contents_ptr();
    let tokens = (0..length)
        .map(|index| unsafe {
            match dtype {
                DType::U32 => Ok(*pointer.cast::<u32>().add(index)),
                DType::I64 => u32::try_from(*pointer.cast::<i64>().add(index)).map_err(|_| {
                    Error::new(
                        Status::InvalidArg,
                        "inference[prefill]: token is outside u32",
                    )
                }),
                _ => unreachable!("inference token dtype was validated"),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(tokens)
}

fn make_managed_sequence(pool: &Arc<PoolInner>) -> Result<NativeKvSequence> {
    Ok(NativeKvSequence {
        pool: pool.clone(),
        state: Arc::new(Mutex::new(
            sequence_state(pool, false).map_err(to_napi_err)?,
        )),
        run_lock: Arc::new(Mutex::new(())),
        released: Arc::new(AtomicBool::new(false)),
    })
}

fn publish_shadow(shadow: &mut ShadowSequences, canonical: &[Arc<Mutex<SeqState>>]) {
    for (provisional, canonical) in shadow.states.iter().zip(canonical) {
        let mut provisional = provisional
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut canonical = canonical.lock().unwrap_or_else(|error| error.into_inner());
        let old_blocks = std::mem::replace(
            &mut canonical.blocks,
            std::mem::take(&mut provisional.blocks),
        );
        canonical.head = provisional.head;
        canonical.cursor = provisional.cursor;
        canonical.advance = 0;
        canonical.last_hash = provisional.last_hash;
        canonical.pending = std::mem::take(&mut provisional.pending);
        canonical.kda_states = std::mem::take(&mut provisional.kda_states);
        canonical.conv_states = std::mem::take(&mut provisional.conv_states);
        for block in old_blocks {
            shadow.pool.unref_block(block);
        }
    }
}

struct StagedCacheMetadata {
    pool: Arc<PoolInner>,
    blocks: Vec<(u32, u64)>,
    snapshots: Vec<(u64, Arc<RecurrentSnapshot>)>,
}

fn stage_cache_metadata(shadow: &ShadowSequences, prompts: &[Vec<u32>]) -> StagedCacheMetadata {
    let mut blocks = Vec::new();
    let mut snapshots = Vec::new();
    for (state, prompt) in shadow.states.iter().zip(prompts) {
        let state = state.lock().unwrap_or_else(|error| error.into_inner());
        let mut hash = HASH_SEED;
        for (absolute, tokens) in prompt.chunks_exact(shadow.pool.block_size).enumerate() {
            hash = chain_hash(hash, tokens);
            if let Some(&block) = absolute
                .checked_sub(state.head)
                .and_then(|index| state.blocks.get(index))
            {
                blocks.push((block, hash));
            }
        }
        if !shadow.pool.k.is_empty()
            && (shadow.pool.kda.layers > 0 || shadow.pool.conv.layers > 0)
            && state.cursor > 0
            && state.cursor % shadow.pool.block_size == 0
            && state.kda_states.len() == shadow.pool.kda.layers
            && state.conv_states.len() == shadow.pool.conv.layers
        {
            if let Some(captured) = RecurrentSnapshot::capture(&state) {
                snapshots.push((state.last_hash, Arc::new(captured)));
            }
        }
    }
    StagedCacheMetadata {
        pool: shadow.pool.clone(),
        blocks,
        snapshots,
    }
}

fn publish_cache_metadata(store: &mut BlockStore, metadata: &StagedCacheMetadata) {
    for &(block, hash) in &metadata.blocks {
        if store.hashes[block as usize].is_none() {
            store.hashes[block as usize] = Some(hash);
            store.by_hash.entry(hash).or_default().push(block);
        }
    }
    for (hash, snapshot) in &metadata.snapshots {
        if store.by_hash.contains_key(hash) {
            store.snapshots.insert(*hash, snapshot.clone());
        }
    }
}

fn publish_paired_cache_metadata(
    target: &StagedCacheMetadata,
    proposer: Option<&StagedCacheMetadata>,
) {
    let Some(proposer) = proposer else {
        let mut store = target
            .pool
            .blocks
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        publish_cache_metadata(&mut store, target);
        return;
    };
    if Arc::ptr_eq(&target.pool, &proposer.pool) {
        let mut store = target
            .pool
            .blocks
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        publish_cache_metadata(&mut store, target);
        publish_cache_metadata(&mut store, proposer);
        return;
    }

    let target_first = Arc::as_ptr(&target.pool) < Arc::as_ptr(&proposer.pool);
    let (first, second) = if target_first {
        (target, proposer)
    } else {
        (proposer, target)
    };
    let mut first_store = first
        .pool
        .blocks
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let mut second_store = second
        .pool
        .blocks
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    publish_cache_metadata(&mut first_store, first);
    publish_cache_metadata(&mut second_store, second);
}

fn sample_at(
    probabilities: &[f64],
    sampling: InferenceSampling,
    sequence_id: u64,
    position: u64,
    purpose: SamplingPurpose,
    subcounter: u64,
) -> u32 {
    if sampling.temperature == 0.0 {
        return probabilities
            .iter()
            .position(|probability| *probability > 0.0)
            .unwrap_or(0) as u32;
    }
    let total: f64 = probabilities.iter().sum();
    let mut draw = random_unit_at(sampling_coordinate(
        sampling.seed,
        sequence_id,
        position,
        purpose,
        subcounter,
    )) * total;
    for (token, probability) in probabilities.iter().enumerate() {
        draw -= probability;
        if draw < 0.0 {
            return token as u32;
        }
    }
    probabilities.len().saturating_sub(1) as u32
}

/// One compiled prefill shape bucket: a headed program plus its optional
/// headless body, sharing the decode program's state geometry and pool.
#[derive(Clone)]
struct PrefillBucket {
    prefill: SpeculativeProgram,
    /// This bucket's proposer replay program (parallel-block configs).
    replay: Option<ReplayProgram>,
}

// Largest compiled bucket covering the remaining tokens of the longest
// active lane; shorter remainders fall back to the smallest bucket, so every
// chunk keeps one fixed compiled shape and buckets only bound padding waste.
fn prefill_bucket(buckets: &[PrefillBucket], remaining: usize) -> &PrefillBucket {
    buckets
        .iter()
        .rev()
        .find(|bucket| bucket.prefill.time <= remaining)
        .unwrap_or(&buckets[0])
}

#[derive(Clone)]
struct InferencePrograms {
    device_ordinal: usize,
    target_prefill: SpeculativeProgram,
    /// Every compiled prefill shape bucket sorted by time ascending; the last
    /// entry always mirrors the primary prefill program and its optional
    /// headless body.
    prefill_buckets: Vec<PrefillBucket>,
    target_decode: SpeculativeProgram,
    /// Verify programs per packed rows-per-sequence width, ascending.
    target_verify: Vec<SpeculativeProgram>,
    target_pool: Arc<PoolInner>,
    proposer_prefill: Option<SpeculativeProgram>,
    proposer_decode: Option<SpeculativeProgram>,
    proposer_pool: Option<Arc<PoolInner>>,
    replay_prefill: Option<ReplayProgram>,
    #[allow(dead_code)]
    replay_decode: Option<ReplayProgram>,
    /// Replay programs per verify width, aligned with target_verify.
    replay_verify: Vec<ReplayProgram>,
    replay_pool: Option<Arc<PoolInner>>,
    max_draft_tokens: usize,
    batch: usize,
    sampling: InferenceSampling,
    proposer_plan: Option<Arc<RetainedProposerPlan>>,
    history_lookup: Option<HistoryLookupConfig>,
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeValueRef {
    pub kind: String,
    pub name: Option<String>,
    pub binding: Option<u32>,
    pub stage: Option<u32>,
    pub output: Option<u32>,
    pub row: Option<u32>,
    pub select_row: Option<bool>,
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeProposerValueSchema {
    pub shape: Vec<u32>,
    pub dtype: NativeDType,
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeTargetHiddenTap {
    pub name: String,
    pub output: u32,
    pub shape: Vec<u32>,
    pub dtype: NativeDType,
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeSharedTargetBinding {
    pub kind: String,
    pub name: String,
    pub tensor: u32,
    pub shape: Vec<u32>,
    pub dtype: NativeDType,
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeStageInputBinding {
    pub slot: u32,
    pub value: NativeValueRef,
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeProposerStage {
    pub executable: Option<u32>,
    pub operation_id: String,
    pub layout_id: Option<String>,
    pub history_lookup: Option<NativeHistoryLookupLayout>,
    pub inputs: Vec<NativeStageInputBinding>,
    pub outputs: Vec<NativeProposerValueSchema>,
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
pub struct NativeProposerStatePlan {
    pub kind: String,
    pub schema_id: Option<String>,
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeProposerCommitPlan {
    pub kind: String,
    pub stage: Option<u32>,
    pub stages: Option<Vec<u32>>,
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeProposerOutputPlan {
    pub topology: String,
    pub probabilities: String,
    pub token_ids: NativeValueRef,
    pub probability_rows: Option<NativeValueRef>,
    pub parents: Option<NativeValueRef>,
    pub confidence: Option<NativeValueRef>,
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeTokenMapPlan {
    pub kind: String,
    pub fingerprint: String,
    pub proposer_vocabulary: Option<u32>,
    pub target_ids: Option<Vec<u32>>,
}

#[napi(object)]
#[derive(Clone)]
pub struct NativeProposerPlan {
    pub target_prefill_taps: Vec<NativeTargetHiddenTap>,
    pub target_decode_taps: Vec<NativeTargetHiddenTap>,
    pub target_verify_taps: Vec<NativeTargetHiddenTap>,
    pub shared_target_bindings: Vec<NativeSharedTargetBinding>,
    pub stages: Vec<NativeProposerStage>,
    pub state: NativeProposerStatePlan,
    pub commit: Option<NativeProposerCommitPlan>,
    pub output: NativeProposerOutputPlan,
    pub token_map: NativeTokenMapPlan,
    pub trained_max_rows: u32,
}

#[derive(Clone)]
#[allow(dead_code)]
struct RetainedStageProgram {
    executable: Arc<executable::MetalExecutable>,
    generated: Vec<value::Value>,
    schema: Option<KvStateSchema>,
    packed_rows_per_sequence: Option<usize>,
}

#[allow(dead_code)]
struct RetainedProposerPlan {
    schema: NativeProposerPlan,
    stages: Vec<RetainedStageProgram>,
    shared: Vec<value::Value>,
    target_decode_prefix_outputs: usize,
    prefill_tap_outputs: Vec<usize>,
    verify_tap_outputs: Vec<usize>,
}

#[derive(Clone)]
struct ReplayProgram {
    executable: Arc<executable::MetalExecutable>,
    generated: Vec<value::Value>,
    schema: KvStateSchema,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HistoryLookupConfig {
    min_match_tokens: usize,
    max_match_tokens: usize,
}

fn history_lookup_config(
    plan: &NativeProposerPlan,
    stage_executables: &[&Executable],
    shared_target_tensors: &[&NativeTensor],
) -> Result<Option<HistoryLookupConfig>> {
    let is_history = plan.stages.len() == 1 && plan.stages[0].operation_id == "HistoryLookup";
    if !is_history {
        return Ok(None);
    }
    let stage = &plan.stages[0];
    let layout = stage.history_lookup.as_ref();
    let token_output = &plan.output.token_ids;
    if layout.is_none_or(|layout| {
        layout.id != "suffix-ngram-v1"
            || layout.min_match_tokens == 0
            || layout.min_match_tokens > layout.max_match_tokens
    }) || stage.layout_id.as_deref() != Some("suffix-ngram-v1")
        || stage.executable.is_some()
        || !stage.inputs.is_empty()
        || stage.outputs.len() != 1
        || !matches!(DType::from(stage.outputs[0].dtype), DType::U32 | DType::I64)
        || !stage_executables.is_empty()
        || !shared_target_tensors.is_empty()
        || !plan.shared_target_bindings.is_empty()
        || !plan.target_prefill_taps.is_empty()
        || !plan.target_decode_taps.is_empty()
        || !plan.target_verify_taps.is_empty()
        || plan.state.kind != "None"
        || plan.state.schema_id.is_some()
        || plan.commit.is_some()
        || plan.output.topology != "Chains"
        || plan.output.probabilities != "Deterministic"
        || token_output.kind != "StageOutput"
        || token_output.stage != Some(0)
        || token_output.output != Some(0)
        || plan.output.probability_rows.is_some()
        || plan.output.parents.is_some()
        || plan.output.confidence.is_some()
        || plan.token_map.kind != "Identity"
        || plan.token_map.proposer_vocabulary.is_some()
        || plan.token_map.target_ids.is_some()
    {
        return Err(plan_error(
            "HistoryLookup requires one native suffix-ngram-v1 deterministic identity chain",
        ));
    }
    let layout = layout.expect("history layout was validated");
    Ok(Some(HistoryLookupConfig {
        min_match_tokens: layout.min_match_tokens as usize,
        max_match_tokens: layout.max_match_tokens as usize,
    }))
}

fn plan_error(message: impl Into<String>) -> Error {
    Error::new(
        Status::InvalidArg,
        format!("inference[compile]: proposer plan {}", message.into()),
    )
}

fn schema_matches(schema: &NativeProposerValueSchema, shape: &[usize], dtype: DType) -> bool {
    DType::from(schema.dtype) == dtype
        && schema.shape.len() == shape.len()
        && schema
            .shape
            .iter()
            .zip(shape)
            .all(|(declared, actual)| *declared as usize == *actual)
}

fn validate_plan_ref<'a>(
    reference: &NativeValueRef,
    before_stage: usize,
    plan: &'a NativeProposerPlan,
) -> Result<Option<NativeProposerValueSchema>> {
    let metadata = match reference.kind.as_str() {
        "PendingTokens" | "CandidatePrefix" | "CommittedHistory" => None,
        "TargetHidden" => {
            let name = reference
                .name
                .as_deref()
                .ok_or_else(|| plan_error("TargetHidden route is missing name"))?;
            let tap = plan
                .target_decode_taps
                .iter()
                .find(|tap| tap.name == name)
                .ok_or_else(|| {
                    plan_error(format!(
                        "references undeclared target hidden exposure {name}"
                    ))
                })?;
            Some(NativeProposerValueSchema {
                shape: tap.shape.clone(),
                dtype: tap.dtype,
            })
        }
        "SharedBinding" => {
            let binding = plan
                .shared_target_bindings
                .get(
                    reference
                        .binding
                        .ok_or_else(|| plan_error("SharedBinding route is missing binding"))?
                        as usize,
                )
                .ok_or_else(|| plan_error("references an undeclared shared binding"))?;
            Some(NativeProposerValueSchema {
                shape: binding.shape.clone(),
                dtype: binding.dtype,
            })
        }
        "StageOutput" => {
            let stage = reference
                .stage
                .ok_or_else(|| plan_error("StageOutput route is missing stage"))?
                as usize;
            let output = reference
                .output
                .ok_or_else(|| plan_error("StageOutput route is missing output"))?
                as usize;
            if stage >= before_stage {
                return Err(plan_error(format!(
                    "stage output {stage}:{output} is not a backward reference"
                )));
            }
            Some(
                plan.stages
                    .get(stage)
                    .and_then(|stage| stage.outputs.get(output))
                    .cloned()
                    .ok_or_else(|| {
                        plan_error(format!("references missing stage output {stage}:{output}"))
                    })?,
            )
        }
        kind => return Err(plan_error(format!("has unknown ValueRef kind {kind}"))),
    };
    if reference.kind != "TargetHidden" && reference.name.is_some()
        || reference.kind != "SharedBinding" && reference.binding.is_some()
        || reference.kind != "StageOutput"
            && (reference.stage.is_some() || reference.output.is_some())
        || reference.kind != "TargetHidden" && reference.select_row == Some(true)
    {
        return Err(plan_error(
            "ValueRef has fields that do not belong to its kind",
        ));
    }
    let Some(mut metadata) = metadata else {
        if reference.row.is_some() || reference.select_row == Some(true) {
            return Err(plan_error(
                "cannot select a row from a dynamic token ValueRef",
            ));
        }
        return Ok(None);
    };
    if reference.row.is_some() && reference.select_row == Some(true) {
        return Err(plan_error(
            "ValueRef cannot select both a fixed row and the active lane",
        ));
    }
    if let Some(row) = reference
        .row
        .or(reference.select_row.filter(|selected| *selected).map(|_| 0))
    {
        let rows = metadata
            .shape
            .first()
            .copied()
            .ok_or_else(|| plan_error("cannot select a row from a scalar value"))?;
        if row >= rows {
            return Err(plan_error(format!(
                "row {row} is outside leading dimension {rows}"
            )));
        }
        metadata.shape.remove(0);
    }
    Ok(Some(metadata))
}

fn validate_and_retain_proposer_plan(
    plan: NativeProposerPlan,
    target_prefill: &Executable,
    target_decode: &Executable,
    target_verify: Option<&Executable>,
    stage_executables: Vec<&Executable>,
    shared_target_tensors: Vec<&NativeTensor>,
) -> Result<RetainedProposerPlan> {
    if plan.trained_max_rows == 0 || plan.token_map.fingerprint.is_empty() {
        return Err(plan_error(
            "has invalid trainedMaxRows or token-map fingerprint",
        ));
    }
    match plan.token_map.kind.as_str() {
        "Identity" if plan.token_map.target_ids.is_none() => {}
        "Table" => {
            let vocabulary = plan
                .token_map
                .proposer_vocabulary
                .ok_or_else(|| plan_error("table token map is missing proposerVocabulary"))?
                as usize;
            let ids = plan
                .token_map
                .target_ids
                .as_ref()
                .ok_or_else(|| plan_error("table token map is missing targetIds"))?;
            let target_vocabulary = target_decode
                .inner
                .executable
                .program
                .outputs
                .first()
                .map(|output| {
                    target_decode.inner.executable.program.values[output.index()].shape[0]
                })
                .unwrap_or(0);
            if ids.len() != vocabulary || ids.iter().any(|id| *id as usize >= target_vocabulary) {
                return Err(plan_error("table token map has invalid target IDs"));
            }
        }
        kind => return Err(plan_error(format!("has invalid token-map kind {kind}"))),
    }
    let validate_taps = |label: &str,
                         taps: &[NativeTargetHiddenTap],
                         executable: &Executable,
                         prefix_outputs: usize|
     -> Result<()> {
        let outputs = &executable.inner.executable.program.outputs;
        let values = &executable.inner.executable.program.values;
        let mut names = HashSet::new();
        let mut roots = HashSet::new();
        for tap in taps {
            if !names.insert(tap.name.clone()) || !roots.insert(tap.output) {
                return Err(plan_error(format!(
                    "declares duplicate {label} hidden exposure or output root"
                )));
            }
            let physical_output = (tap.output as usize)
                .checked_sub(1)
                .and_then(|root| prefix_outputs.checked_add(root))
                .ok_or_else(|| {
                    plan_error(format!(
                        "{label} hidden tap must reference a non-logits output root"
                    ))
                })?;
            let value = outputs
                .get(physical_output)
                .and_then(|output| values.get(output.index()))
                .ok_or_else(|| {
                    plan_error(format!("{label} hidden output {} is missing", tap.output))
                })?;
            if !schema_matches(
                &NativeProposerValueSchema {
                    shape: tap.shape.clone(),
                    dtype: tap.dtype,
                },
                &value.shape,
                value.dtype,
            ) {
                return Err(plan_error(format!(
                    "{label} hidden output {} metadata does not match",
                    tap.output
                )));
            }
        }
        Ok(())
    };
    let prefill_prefix = target_prefill
        .state
        .as_ref()
        .map_or(1, |state| state.schema.batch);
    let decode_prefix = target_decode
        .state
        .as_ref()
        .map_or(1, |state| state.schema.batch);
    validate_taps(
        "prefill",
        &plan.target_prefill_taps,
        target_prefill,
        prefill_prefix,
    )?;
    validate_taps(
        "decode",
        &plan.target_decode_taps,
        target_decode,
        decode_prefix,
    )?;
    if let Some(target_verify) = target_verify {
        validate_taps("verify", &plan.target_verify_taps, target_verify, 1)?;
    } else if !plan.target_verify_taps.is_empty() {
        return Err(plan_error("declares verify taps without a target verifier"));
    }
    let mut shared_kinds = HashSet::new();
    for binding in &plan.shared_target_bindings {
        if !matches!(binding.kind.as_str(), "TokenEmbedding" | "LmHead")
            || binding.name.is_empty()
            || !shared_kinds.insert(binding.kind.clone())
        {
            return Err(plan_error(
                "has an invalid or duplicate shared target binding",
            ));
        }
        let tensor = shared_target_tensors
            .get(binding.tensor as usize)
            .ok_or_else(|| {
                plan_error(format!("shared tensor index {} is missing", binding.tensor))
            })?;
        let value = tensor.val_cloned()?;
        // GGUF leaves expose encoded physical storage here. The logical schema
        // is validated in TypeScript and the executable validates the physical
        // binding before execution.
        drop(value);
    }
    if plan.stages.is_empty() || stage_executables.len() != plan.stages.len() {
        return Err(plan_error("stage executable count does not match stages"));
    }
    for (stage_index, stage) in plan.stages.iter().enumerate() {
        if stage
            .executable
            .is_none_or(|executable| executable as usize >= stage_executables.len())
            || stage.operation_id.is_empty()
            || stage.layout_id.as_ref().is_some_and(String::is_empty)
            || stage.history_lookup.is_some()
        {
            return Err(plan_error(format!(
                "stage {stage_index} has invalid executable or IDs"
            )));
        }
        let executable = stage_executables[stage.executable.unwrap() as usize];
        let state = executable.state.as_ref();
        let tensor_slots = executable
            .inner
            .slots
            .iter()
            .enumerate()
            .filter(|(index, slot)| {
                !slot.scalar
                    && !state.is_some_and(|state| {
                        state.cursor_tensor && *index as u32 == state.cursor_slot
                    })
            })
            .collect::<Vec<_>>();
        if executable.inner.slots.iter().any(|slot| slot.scalar)
            || stage.inputs.len() != tensor_slots.len()
        {
            return Err(plan_error(format!(
                "stage {stage_index} input count does not match executable"
            )));
        }
        let mut slots = HashSet::new();
        for binding in &stage.inputs {
            if !slots.insert(binding.slot) {
                return Err(plan_error(format!(
                    "stage {stage_index} binds slot {} twice",
                    binding.slot
                )));
            }
            let declared = executable
                .inner
                .slots
                .get(binding.slot as usize)
                .filter(|slot| !slot.scalar)
                .ok_or_else(|| {
                    plan_error(format!(
                        "stage {stage_index} binds missing tensor slot {}",
                        binding.slot
                    ))
                })?;
            if let Some(source) = validate_plan_ref(&binding.value, stage_index, &plan)? {
                if binding.value.kind != "SharedBinding"
                    && !schema_matches(&source, &declared.shape, declared.dtype)
                {
                    return Err(plan_error(format!(
                        "stage {stage_index} slot {} route metadata does not match",
                        binding.slot
                    )));
                }
            }
        }
        let outputs = &executable.inner.executable.program.outputs;
        if stage.outputs.len() != outputs.len() {
            return Err(plan_error(format!(
                "stage {stage_index} output count does not match executable"
            )));
        }
        for (schema, output) in stage.outputs.iter().zip(outputs) {
            let value = &executable.inner.executable.program.values[output.index()];
            if !schema_matches(schema, &value.shape, value.dtype) {
                return Err(plan_error(format!(
                    "stage {stage_index} output metadata does not match executable"
                )));
            }
        }
    }
    for reference in [
        Some(&plan.output.token_ids),
        plan.output.probability_rows.as_ref(),
        plan.output.parents.as_ref(),
        plan.output.confidence.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        validate_plan_ref(reference, plan.stages.len(), &plan)?;
    }
    if !matches!(plan.output.topology.as_str(), "Chains" | "Trees")
        || !matches!(
            plan.output.probabilities.as_str(),
            "CausalNormalized" | "Deterministic" | "Unavailable"
        )
    {
        return Err(plan_error(
            "has invalid output topology or probability contract",
        ));
    }
    match plan.state.kind.as_str() {
        "None" if plan.state.schema_id.is_none() && plan.commit.is_none() => {}
        "Kv" => match plan.commit.as_ref().map(|commit| commit.kind.as_str()) {
            Some("AutoregressiveChain") => {
                let stage = plan
                    .commit
                    .as_ref()
                    .and_then(|commit| commit.stage)
                    .ok_or_else(|| plan_error("KV chain commit is missing stage"))?;
                if stage as usize >= plan.stages.len() {
                    return Err(plan_error("KV chain commit stage is out of range"));
                }
            }
            Some("Replay") => {
                let stages = plan
                    .commit
                    .as_ref()
                    .and_then(|commit| commit.stages.as_ref())
                    .ok_or_else(|| plan_error("KV replay commit is missing stages"))?;
                if stages.is_empty()
                    || stages
                        .iter()
                        .any(|stage| *stage as usize >= plan.stages.len())
                {
                    return Err(plan_error("KV replay commit stages are invalid"));
                }
            }
            _ => return Err(plan_error("KV state has an invalid commit plan")),
        },
        _ => return Err(plan_error("has invalid state metadata")),
    }

    // Validation above is side-effect free. Clone every retained native owner only now,
    // so a failed constructor cannot leave a partially retained plan.
    let stages = plan
        .stages
        .iter()
        .map(|stage| {
            let executable = stage_executables[stage.executable.unwrap() as usize];
            RetainedStageProgram {
                executable: executable.inner.executable.clone(),
                generated: executable.inner.generated_bindings.clone(),
                schema: executable.state.as_ref().map(|state| state.schema),
                packed_rows_per_sequence: executable
                    .state
                    .as_ref()
                    .and_then(|state| state.packed_rows_per_sequence),
            }
        })
        .collect();
    let shared = plan
        .shared_target_bindings
        .iter()
        .map(|binding| shared_target_tensors[binding.tensor as usize].val_cloned())
        .collect::<Result<Vec<_>>>()?;
    let prefill_tap_outputs = plan
        .target_prefill_taps
        .iter()
        .map(|tap| prefill_prefix + tap.output as usize - 1)
        .collect();
    let verify_tap_outputs = plan
        .target_verify_taps
        .iter()
        .map(|tap| tap.output as usize)
        .collect();
    Ok(RetainedProposerPlan {
        schema: plan,
        stages,
        shared,
        target_decode_prefix_outputs: decode_prefix,
        prefill_tap_outputs,
        verify_tap_outputs,
    })
}

#[allow(dead_code)]
fn resolve_routed_value(
    plan: &RetainedProposerPlan,
    reference: &NativeValueRef,
    target_outputs: &[value::Value],
    dynamic: &HashMap<String, value::Value>,
    stage_outputs: &[Vec<value::Value>],
    target_row: usize,
) -> std::result::Result<value::Value, String> {
    let value = match reference.kind.as_str() {
        "PendingTokens" | "CandidatePrefix" | "CommittedHistory" => dynamic
            .get(&reference.kind)
            .cloned()
            .ok_or_else(|| format!("route: {} is unbound", reference.kind))?,
        "TargetHidden" => {
            let name = reference
                .name
                .as_deref()
                .ok_or_else(|| "route: target hidden name is missing".to_string())?;
            let tap = plan
                .schema
                .target_decode_taps
                .iter()
                .find(|tap| tap.name == name)
                .ok_or_else(|| format!("route: target hidden exposure {name} is undeclared"))?;
            target_outputs
                .get(
                    (tap.output as usize)
                        .checked_sub(1)
                        .and_then(|root| plan.target_decode_prefix_outputs.checked_add(root))
                        .ok_or_else(|| "route: target output index overflows".to_string())?,
                )
                .cloned()
                .ok_or_else(|| format!("route: target output {} is unavailable", tap.output))?
        }
        "SharedBinding" => {
            let index = reference
                .binding
                .ok_or_else(|| "route: shared binding index is missing".to_string())?
                as usize;
            plan.shared[index].clone()
        }
        "StageOutput" => stage_outputs
            .get(
                reference
                    .stage
                    .ok_or_else(|| "route: stage is missing".to_string())? as usize,
            )
            .and_then(|outputs| {
                reference
                    .output
                    .and_then(|output| outputs.get(output as usize))
            })
            .cloned()
            .ok_or_else(|| "route: stage output is unavailable".to_string())?,
        kind => return Err(format!("route: unsupported ValueRef kind {kind}")),
    };
    match reference.row.map(|row| row as usize).or_else(|| {
        reference
            .select_row
            .is_some_and(|selected| selected)
            .then_some(target_row)
    }) {
        Some(row) => executable::route_leading_row(&value, row).map(|(value, _copied)| value),
        None => Ok(value),
    }
}

/// Executes the stateless subset of a retained proposer DAG. Intermediate
/// values stay as native Metal Values and only the explicitly selected result
/// is returned to its native caller; this helper never creates JS tensor
/// wrappers or reads device storage back to the host.
#[allow(dead_code)]
fn execute_stateless_proposer_dag(
    plan: &RetainedProposerPlan,
    target_outputs: &[value::Value],
    dynamic: &HashMap<String, value::Value>,
    target_row: usize,
    cancelled: &effect_torch_runtime::CancellationFlag,
) -> std::result::Result<Vec<Vec<value::Value>>, String> {
    if plan.schema.state.kind != "None" {
        return Err(
            "route: stateless DAG helper cannot execute a stateful proposer plan".to_string(),
        );
    }
    let mut outputs = Vec::with_capacity(plan.stages.len());
    for (index, (stage, retained)) in plan.schema.stages.iter().zip(&plan.stages).enumerate() {
        if retained.executable.state_schema.is_some() {
            return Err(format!("route: stateless stage {index} has decode state"));
        }
        let mut bindings = stage
            .inputs
            .iter()
            .map(|binding| {
                resolve_routed_value(
                    plan,
                    &binding.value,
                    target_outputs,
                    dynamic,
                    &outputs,
                    target_row,
                )
                .map(|value| (binding.slot, value))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        bindings.sort_by_key(|(slot, _)| *slot);
        let bindings = bindings
            .into_iter()
            .map(|(_, value)| value)
            .collect::<Vec<_>>();
        outputs.push(executable::execute_with_scalars(
            &retained.executable,
            &bindings,
            &retained.generated,
            &[],
            cancelled,
        )?);
    }
    Ok(outputs)
}

#[derive(Default)]
struct InferenceDiagnosticsState {
    rounds_started: AtomicU64,
    rounds_completed: AtomicU64,
    rounds_recovered: AtomicU64,
    ordinary_rounds: AtomicU64,
    speculative_rounds: AtomicU64,
    proposed_tokens: AtomicU64,
    accepted_tokens: AtomicU64,
    emitted_tokens: AtomicU64,
    provisional_blocks: AtomicU64,
    rolled_back_blocks: AtomicU64,
    draft_nanos: AtomicU64,
    verification_nanos: AtomicU64,
    accepted_length_histogram: Mutex<Vec<u64>>,
    target_pool_high_water_blocks: AtomicU64,
    proposer_pool_high_water_blocks: AtomicU64,
    last_round_id: AtomicU64,
    has_round: AtomicBool,
    last_failure_phase: Mutex<Option<String>>,
}

#[napi]
pub struct NativeInferenceArtifact {
    programs: Arc<InferencePrograms>,
    next_sequence_id: Arc<AtomicU64>,
    next_round_id: Arc<AtomicU64>,
    diagnostics: Arc<InferenceDiagnosticsState>,
}

#[napi(object)]
pub struct NativeInferenceDiagnostics {
    pub rounds_started: BigInt,
    pub rounds_completed: BigInt,
    pub rounds_recovered: BigInt,
    pub ordinary_rounds: BigInt,
    pub speculative_rounds: BigInt,
    pub proposed_tokens: BigInt,
    pub accepted_tokens: BigInt,
    pub emitted_tokens: BigInt,
    pub provisional_blocks: BigInt,
    pub rolled_back_blocks: BigInt,
    pub draft_nanos: BigInt,
    pub verification_nanos: BigInt,
    pub accepted_length_histogram: Vec<BigInt>,
    pub target_pool_high_water_blocks: BigInt,
    pub proposer_pool_high_water_blocks: Option<BigInt>,
    pub last_round_id: Option<BigInt>,
    pub last_failure_phase: Option<String>,
}

#[napi]
impl NativeInferenceArtifact {
    #[napi(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        target_prefill: &Executable,
        target_decode: &Executable,
        target_verify: Vec<&Executable>,
        target_pool: &NativeKvPool,
        proposer_prefill: Option<&Executable>,
        proposer_decode: Option<&Executable>,
        proposer_pool: Option<&NativeKvPool>,
        max_draft_tokens: Option<u32>,
        batch_size: u32,
        token_dtype: NativeDType,
        sampling: NativeInferenceSamplingOptions,
        proposer_plan: Option<NativeProposerPlan>,
        stage_executables: Option<Vec<&Executable>>,
        shared_target_tensors: Option<Vec<&NativeTensor>>,
        replay_prefill: Option<&Executable>,
        replay_decode: Option<&Executable>,
        replay_verify: Vec<&Executable>,
        replay_pool: Option<&NativeKvPool>,
        prefill_buckets: Option<Vec<&Executable>>,
        replay_prefill_buckets: Option<Vec<&Executable>>,
    ) -> Result<Self> {
        let device_ordinal = target_prefill.device_ordinal;
        let executables_match = target_decode.device_ordinal == device_ordinal
            && target_verify
                .iter()
                .all(|executable| executable.device_ordinal == device_ordinal)
            && proposer_prefill
                .is_none_or(|executable| executable.device_ordinal == device_ordinal)
            && proposer_decode.is_none_or(|executable| executable.device_ordinal == device_ordinal)
            && stage_executables.as_ref().is_none_or(|executables| {
                executables
                    .iter()
                    .all(|executable| executable.device_ordinal == device_ordinal)
            })
            && replay_prefill.is_none_or(|executable| executable.device_ordinal == device_ordinal)
            && replay_decode.is_none_or(|executable| executable.device_ordinal == device_ordinal)
            && replay_verify
                .iter()
                .all(|executable| executable.device_ordinal == device_ordinal)
            && prefill_buckets.as_ref().is_none_or(|executables| {
                executables
                    .iter()
                    .all(|executable| executable.device_ordinal == device_ordinal)
            })
            && replay_prefill_buckets.as_ref().is_none_or(|executables| {
                executables
                    .iter()
                    .all(|executable| executable.device_ordinal == device_ordinal)
            });
        let pools_match = target_pool.inner.device_ordinal == device_ordinal
            && proposer_pool.is_none_or(|pool| pool.inner.device_ordinal == device_ordinal)
            && replay_pool.is_none_or(|pool| pool.inner.device_ordinal == device_ordinal);
        let tensors_match = shared_target_tensors.as_ref().is_none_or(|tensors| {
            tensors
                .iter()
                .all(|tensor| tensor.device_ordinal == device_ordinal)
        });
        if !executables_match || !pools_match || !tensors_match {
            return Err(Error::new(
                Status::InvalidArg,
                "inference[compile]: all resources must use the same Metal device",
            ));
        }
        let generalized = proposer_plan.is_some();
        if generalized
            && (proposer_prefill.is_some()
                || proposer_decode.is_some()
                || proposer_pool.is_some()
                || target_verify.is_empty()
                || !max_draft_tokens.is_some_and(|value| value > 0))
        {
            return Err(plan_error(
                "generalized proposer requires only target verify and maxDraftTokens",
            ));
        }
        let target_prefill_program = target_prefill;
        let target_decode_program = target_decode;
        let target_verify_program = target_verify.last().copied();
        let target_verify_executables = target_verify.clone();
        let target_prefill = inference_program(target_prefill, "prefill", false, generalized)?;
        let target_decode = inference_program(target_decode, "decode", true, generalized)?;
        let target_verify = target_verify
            .iter()
            .map(|program| validate_speculative_program(program, false, generalized))
            .collect::<Result<Vec<_>>>()?;
        // Widths ascend by packed rows per sequence; duplicates or a
        // missing packed layout make the runtime width selection ambiguous.
        let mut verify_rows = Vec::new();
        for program in &target_verify {
            let rows = program.packed_rows_per_sequence.ok_or_else(|| {
                plan_error("speculative verify program requires a packed rows layout")
            })?;
            if verify_rows.last().is_some_and(|last| *last >= rows) {
                return Err(plan_error(
                    "speculative verify widths must be strictly ascending",
                ));
            }
            verify_rows.push(rows);
        }
        let proposer_prefill = if generalized {
            None
        } else {
            proposer_prefill
                .map(|program| inference_program(program, "proposer", false, false))
                .transpose()?
        };
        let proposer_decode = if generalized {
            None
        } else {
            proposer_decode
                .map(|program| inference_program(program, "proposer", true, false))
                .transpose()?
        };
        let proposer_pool = if generalized {
            None
        } else {
            proposer_pool.map(|pool| pool.inner.clone())
        };
        let proposer_complete = proposer_prefill.is_some()
            && proposer_decode.is_some()
            && proposer_pool.is_some()
            && !target_verify.is_empty()
            && max_draft_tokens.is_some_and(|value| value > 0);
        let proposer_empty = proposer_prefill.is_none()
            && proposer_decode.is_none()
            && proposer_pool.is_none()
            && target_verify.is_empty()
            && max_draft_tokens.is_none();
        if !generalized && !proposer_complete && !proposer_empty {
            return Err(Error::new(
                Status::InvalidArg,
                "inference[compile]: proposer requires prefill, decode, pool, target verify and maxDraftTokens",
            ));
        }
        let batch = batch_size as usize;
        let dtype: DType = token_dtype.into();
        if batch == 0
            || target_prefill.batch != batch
            || target_decode.batch != batch
            || target_prefill.token_dtype != dtype
            || target_decode.token_dtype != dtype
            || target_prefill.vocabulary != target_decode.vocabulary
            || !compatible_pool(&target_prefill, &target_pool.inner)
            || !compatible_pool(&target_decode, &target_pool.inner)
        {
            return Err(Error::new(
                Status::InvalidArg,
                "inference[compile]: incompatible target programs, pool, batch or token dtype",
            ));
        }
        // Smaller prefill shape buckets share the primary prefill's state
        // schema, pool, batch, and token dtype; only the compiled time
        // dimension differs. They are stored ascending with the primary
        // (largest) bucket last.
        let prefill_bucket_programs = prefill_buckets.unwrap_or_default();
        let mut bucket_pairs = Vec::new();
        for bucket in &prefill_bucket_programs {
            let bucket_prefill = inference_program(bucket, "prefill", false, generalized)?;
            if bucket_prefill.batch != batch
                || bucket_prefill.token_dtype != dtype
                || bucket_prefill.vocabulary != target_decode.vocabulary
                || bucket_prefill.time >= target_prefill.time
                || !same_inference_state(bucket_prefill.schema, target_prefill.schema)
                || !compatible_pool(&bucket_prefill, &target_pool.inner)
            {
                return Err(Error::new(
                    Status::InvalidArg,
                    "inference[compile]: incompatible prefill bucket program, pool, batch or token dtype",
                ));
            }
            bucket_pairs.push((*bucket, bucket_prefill));
        }
        bucket_pairs.sort_by_key(|(_, program)| program.time);
        if bucket_pairs
            .windows(2)
            .any(|pair| pair[0].1.time == pair[1].1.time)
        {
            return Err(Error::new(
                Status::InvalidArg,
                "inference[compile]: prefill bucket times must be distinct",
            ));
        }
        let mut prefill_buckets: Vec<PrefillBucket> = bucket_pairs
            .iter()
            .map(|(_, program)| PrefillBucket {
                prefill: program.clone(),
                replay: None,
            })
            .collect();
        prefill_buckets.push(PrefillBucket {
            prefill: target_prefill.clone(),
            replay: None,
        });
        if proposer_complete {
            let draft = max_draft_tokens.unwrap() as usize;
            let prefill = proposer_prefill.as_ref().unwrap();
            let decode = proposer_decode.as_ref().unwrap();
            let verify = target_verify.last().unwrap();
            let pool = proposer_pool.as_ref().unwrap();
            if prefill.batch != batch
                || decode.batch != batch
                || verify.batch != batch
                || prefill.token_dtype != dtype
                || decode.token_dtype != dtype
                || prefill.vocabulary != target_decode.vocabulary
                || decode.vocabulary != target_decode.vocabulary
                || verify.vocabulary != target_decode.vocabulary
                || verify
                    .packed_rows_per_sequence
                    .is_none_or(|rows| draft + 1 > rows)
                || !compatible_pool(prefill, pool)
                || !compatible_pool(decode, pool)
                || !compatible_pool(verify, &target_pool.inner)
            {
                return Err(Error::new(
                    Status::InvalidArg,
                    "inference[compile]: incompatible proposer/verify artifact bundle",
                ));
            }
        }
        if generalized {
            let draft = max_draft_tokens.unwrap() as usize;
            for (index, verify) in target_verify.iter().enumerate() {
                if verify.batch != batch
                    || verify.token_dtype != dtype
                    || verify.vocabulary != target_decode.vocabulary
                    || (index == target_verify.len() - 1 && draft + 1 > verify_rows[index])
                    || !compatible_pool(verify, &target_pool.inner)
                {
                    return Err(Error::new(
                        Status::InvalidArg,
                        "inference[compile]: incompatible generalized target verifier",
                    ));
                }
            }
        }
        let sampling = inference_sampling(sampling)?;
        let replay_present = replay_prefill.is_some()
            || replay_decode.is_some()
            || !replay_verify.is_empty()
            || replay_pool.is_some()
            || replay_prefill_buckets
                .as_ref()
                .is_some_and(|buckets| !buckets.is_empty());
        if replay_present
            != (replay_prefill.is_some()
                && replay_decode.is_some()
                && !replay_verify.is_empty()
                && replay_pool.is_some())
        {
            return Err(plan_error(
                "ParallelBlock requires a complete replay bundle",
            ));
        }
        let unexpected_plan_handles = proposer_plan.is_none()
            && (stage_executables
                .as_ref()
                .is_some_and(|values| !values.is_empty())
                || shared_target_tensors
                    .as_ref()
                    .is_some_and(|values| !values.is_empty()));
        let stage_executables = stage_executables.unwrap_or_default();
        let shared_target_tensors = shared_target_tensors.unwrap_or_default();
        let history_lookup = proposer_plan
            .as_ref()
            .map(|plan| history_lookup_config(plan, &stage_executables, &shared_target_tensors))
            .transpose()?
            .flatten();
        if history_lookup.is_some_and(|_| {
            proposer_plan
                .as_ref()
                .is_some_and(|plan| plan.trained_max_rows < max_draft_tokens.unwrap_or(0))
        }) {
            return Err(plan_error(
                "HistoryLookup maxDraftTokens exceeds trainedMaxRows",
            ));
        }
        let proposer_plan = proposer_plan
            .map(|plan| {
                if history_lookup.is_some() {
                    Ok(RetainedProposerPlan {
                        schema: plan,
                        stages: Vec::new(),
                        shared: Vec::new(),
                        target_decode_prefix_outputs: target_decode_program
                            .state
                            .as_ref()
                            .map_or(1, |state| state.schema.batch),
                        prefill_tap_outputs: Vec::new(),
                        verify_tap_outputs: Vec::new(),
                    })
                } else {
                    validate_and_retain_proposer_plan(
                        plan,
                        target_prefill_program,
                        target_decode_program,
                        target_verify_program,
                        stage_executables,
                        shared_target_tensors,
                    )
                }
                .map(Arc::new)
            })
            .transpose()?;
        if unexpected_plan_handles {
            return Err(plan_error("handle arrays require a proposerPlan"));
        }
        let (replay_prefill, replay_decode, replay_verify, replay_pool) = if let Some(plan) =
            proposer_plan.as_ref().filter(|_| history_lookup.is_none())
        {
            let schema = &plan.schema;
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
                                && route.row.is_none()
                                && route.select_row != Some(true)
                        })
            } else {
                schema.output.probabilities == "Unavailable" && schema.stages[0].outputs.len() == 1
            };
            let parallel = schema.stages.len() == 1
                && schema.stages[0].operation_id == "ParallelBlock"
                && schema.stages[0].layout_id.as_deref() == Some("parallel-block")
                && valid_probability_route
                && schema.state.kind == "Kv"
                && schema
                    .state
                    .schema_id
                    .as_ref()
                    .is_some_and(|id| !id.is_empty())
                && schema.commit.as_ref().is_some_and(|commit| {
                    commit.kind == "Replay"
                        && commit.stage.is_none()
                        && commit.stages.as_deref() == Some(&[0])
                })
                && schema.output.topology == "Chains"
                && schema.output.parents.is_none()
                && schema.output.confidence.is_none()
                && schema.output.token_ids.kind == "StageOutput"
                && schema.output.token_ids.stage == Some(0)
                && schema.output.token_ids.output == Some(0)
                && schema.output.token_ids.row.is_none()
                && schema.output.token_ids.select_row != Some(true)
                && schema.token_map.kind == "Identity"
                && schema.token_map.proposer_vocabulary.is_none()
                && schema.token_map.target_ids.is_none()
                && schema.trained_max_rows >= max_draft_tokens.unwrap_or(0);
            let stage_routes = &schema.stages[0].inputs;
            let valid_routes = stage_routes
                .iter()
                .filter(|binding| binding.value.kind == "PendingTokens")
                .count()
                == 1
                && stage_routes.iter().all(|binding| {
                    matches!(
                        binding.value.kind.as_str(),
                        "PendingTokens" | "SharedBinding"
                    )
                });
            if !parallel || !valid_routes || !replay_present {
                return Err(plan_error(
                    "generalized execution requires one replayable ParallelBlock stage",
                ));
            }
            let stage = &plan.stages[0];
            let stage_schema = stage
                .schema
                .ok_or_else(|| plan_error("ParallelBlock stage must be stateful"))?;
            let stage_outputs = &stage.executable.program.outputs;
            let stage_values = &stage.executable.program.values;
            if stage.packed_rows_per_sequence.is_some()
                || stage_outputs.len() != if has_probability_rows { 2 } else { 1 }
                || stage_outputs
                    .first()
                    .and_then(|output| stage_values.get(output.index()))
                    .is_none_or(|output| {
                        output.dtype != DType::U32
                            || output.shape.as_ref() != [batch, schema.trained_max_rows as usize]
                    })
                || (has_probability_rows
                    && stage_outputs
                        .get(1)
                        .and_then(|output| stage_values.get(output.index()))
                        .is_none_or(|output| {
                            output.dtype != DType::F32
                                || output.shape.as_ref()
                                    != schema.stages[0].outputs[1]
                                        .shape
                                        .iter()
                                        .map(|dimension| *dimension as usize)
                                        .collect::<Vec<_>>()
                        }))
            {
                return Err(plan_error(
                    "ParallelBlock stage must output one u32 [batch, trainedMaxRows] tensor",
                ));
            }
            let pool = &replay_pool.expect("complete replay bundle").inner;
            let prefill = validate_replay_program(
                replay_prefill.expect("complete replay bundle"),
                pool,
                batch,
                target_prefill_program,
                &plan.prefill_tap_outputs,
            )?;
            // Every prefill shape bucket replays through its own program,
            // compiled against that bucket's tap shapes and aligned with
            // the sorted (ascending) bucket order.
            let replay_bucket_programs = replay_prefill_buckets.unwrap_or_default();
            if replay_bucket_programs.len() != bucket_pairs.len() {
                return Err(plan_error(
                    "ParallelBlock replay prefill buckets must align with the prefill buckets",
                ));
            }
            let mut bucket_replays = Vec::with_capacity(bucket_pairs.len());
            for (replay, (target, _)) in replay_bucket_programs.iter().zip(&bucket_pairs) {
                let bucket_replay = validate_replay_program(
                    replay,
                    pool,
                    batch,
                    target,
                    &plan.prefill_tap_outputs,
                )?;
                if !same_inference_state(bucket_replay.schema, prefill.schema) {
                    return Err(plan_error(
                        "ParallelBlock replay bucket state geometry is incompatible",
                    ));
                }
                bucket_replays.push(bucket_replay);
            }
            for (bucket, replay) in prefill_buckets.iter_mut().zip(
                bucket_replays
                    .into_iter()
                    .chain(std::iter::once(prefill.clone())),
            ) {
                bucket.replay = Some(replay);
            }
            let decode_taps = schema
                .target_decode_taps
                .iter()
                .map(|tap| batch + tap.output as usize - 1)
                .collect::<Vec<_>>();
            let decode = validate_replay_program(
                replay_decode.expect("complete replay bundle"),
                pool,
                batch,
                target_decode_program,
                &decode_taps,
            )?;
            if replay_verify.len() != target_verify.len() {
                return Err(plan_error(
                    "ParallelBlock replay requires one program per verify width",
                ));
            }
            let verify = replay_verify
                .iter()
                .zip(target_verify_executables.iter())
                .map(|(replay, target)| {
                    validate_replay_program(replay, pool, batch, target, &plan.verify_tap_outputs)
                })
                .collect::<Result<Vec<_>, _>>()?;
            if !same_inference_state(prefill.schema, decode.schema)
                || verify
                    .iter()
                    .any(|program| !same_inference_state(prefill.schema, program.schema))
                || !same_inference_state(prefill.schema, stage_schema)
                || prefill.schema.max_tokens != target_prefill.schema.max_tokens
            {
                return Err(plan_error(
                    "ParallelBlock proposal and replay state geometry is incompatible",
                ));
            }
            (Some(prefill), Some(decode), verify, Some(pool.clone()))
        } else {
            if replay_present {
                return Err(plan_error("replay bundle requires ParallelBlock"));
            }
            (None, None, Vec::new(), None)
        };
        Ok(Self {
            programs: Arc::new(InferencePrograms {
                device_ordinal,
                target_prefill,
                prefill_buckets,
                target_decode,
                target_verify,
                target_pool: target_pool.inner.clone(),
                proposer_prefill,
                proposer_decode,
                proposer_pool,
                replay_prefill,
                replay_decode,
                replay_verify,
                replay_pool,
                max_draft_tokens: max_draft_tokens.unwrap_or(0) as usize,
                batch,
                sampling,
                proposer_plan,
                history_lookup,
            }),
            next_sequence_id: Arc::new(AtomicU64::new(0)),
            next_round_id: Arc::new(AtomicU64::new(0)),
            diagnostics: Arc::new(InferenceDiagnosticsState {
                accepted_length_histogram: Mutex::new(vec![
                    0;
                    max_draft_tokens.unwrap_or(0) as usize
                        + 1
                ]),
                ..InferenceDiagnosticsState::default()
            }),
        })
    }

    #[napi]
    pub fn open(&self) -> NativeInferenceSession {
        static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);
        NativeInferenceSession {
            programs: self.programs.clone(),
            next_sequence_id: self.next_sequence_id.clone(),
            next_round_id: self.next_round_id.clone(),
            diagnostics: self.diagnostics.clone(),
            state: Arc::new(Mutex::new(InferenceSessionState {
                id: NEXT_SESSION.fetch_add(1, Ordering::Relaxed),
                closed: false,
                lanes: (0..self.programs.batch).map(|_| None).collect(),
                receipt: None,
                spec_proposed: 0,
                spec_accepted: 0,
            })),
        }
    }

    #[napi(getter)]
    pub fn proposer_plan(&self) -> Option<NativeProposerPlan> {
        self.programs
            .proposer_plan
            .as_ref()
            .map(|plan| plan.schema.clone())
    }

    #[napi(getter)]
    pub fn inference_diagnostics(&self) -> NativeInferenceDiagnostics {
        NativeInferenceDiagnostics {
            rounds_started: bigint(self.diagnostics.rounds_started.load(Ordering::Relaxed)),
            rounds_completed: bigint(self.diagnostics.rounds_completed.load(Ordering::Relaxed)),
            rounds_recovered: bigint(self.diagnostics.rounds_recovered.load(Ordering::Relaxed)),
            ordinary_rounds: bigint(self.diagnostics.ordinary_rounds.load(Ordering::Relaxed)),
            speculative_rounds: bigint(self.diagnostics.speculative_rounds.load(Ordering::Relaxed)),
            proposed_tokens: bigint(self.diagnostics.proposed_tokens.load(Ordering::Relaxed)),
            accepted_tokens: bigint(self.diagnostics.accepted_tokens.load(Ordering::Relaxed)),
            emitted_tokens: bigint(self.diagnostics.emitted_tokens.load(Ordering::Relaxed)),
            provisional_blocks: bigint(self.diagnostics.provisional_blocks.load(Ordering::Relaxed)),
            rolled_back_blocks: bigint(self.diagnostics.rolled_back_blocks.load(Ordering::Relaxed)),
            draft_nanos: bigint(self.diagnostics.draft_nanos.load(Ordering::Relaxed)),
            verification_nanos: bigint(self.diagnostics.verification_nanos.load(Ordering::Relaxed)),
            accepted_length_histogram: self
                .diagnostics
                .accepted_length_histogram
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .iter()
                .copied()
                .map(bigint)
                .collect(),
            target_pool_high_water_blocks: bigint(
                self.diagnostics
                    .target_pool_high_water_blocks
                    .load(Ordering::Relaxed),
            ),
            proposer_pool_high_water_blocks: self
                .programs
                .proposer_pool
                .as_ref()
                .or(self.programs.replay_pool.as_ref())
                .map(|_| {
                    bigint(
                        self.diagnostics
                            .proposer_pool_high_water_blocks
                            .load(Ordering::Relaxed),
                    )
                }),
            last_round_id: self
                .diagnostics
                .has_round
                .load(Ordering::Relaxed)
                .then(|| bigint(self.diagnostics.last_round_id.load(Ordering::Relaxed))),
            last_failure_phase: self
                .diagnostics
                .last_failure_phase
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clone(),
        }
    }
}

struct InferenceLane {
    sequence_id: u64,
    target: NativeKvSequence,
    proposer: Option<NativeKvSequence>,
    pending: u32,
    generated: u64,
    max_tokens: Option<u64>,
    eos: Vec<u32>,
    terminal: Option<String>,
    sampling: InferenceSampling,
    history: Vec<u32>,
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
    fingerprint: u64,
    pages: Vec<ReceiptPage>,
}

fn recover_receipt(receipt: &Option<Receipt>, fingerprint: u64) -> Result<Option<&Receipt>> {
    match receipt {
        Some(receipt) if receipt.fingerprint == fingerprint => Ok(Some(receipt)),
        Some(_) => Err(Error::new(
            Status::InvalidArg,
            "inference[admission]: pending receipt belongs to a different request",
        )),
        None => Ok(None),
    }
}

fn acknowledge_receipt(receipt: &mut Option<Receipt>, round_id: u64) -> Result<()> {
    match receipt {
        Some(pending) if pending.round_id == round_id => *receipt = None,
        _ => {
            return Err(Error::new(
                Status::InvalidArg,
                "inference[publish]: receipt is not pending",
            ))
        }
    }
    Ok(())
}

struct InferenceSessionState {
    id: u64,
    closed: bool,
    lanes: Vec<Option<InferenceLane>>,
    receipt: Option<Receipt>,
    /// Cumulative proposal/acceptance counts driving the adaptive verify
    /// width. Deterministic per session (no wall-clock inputs), so seeded
    /// runs replay identically.
    spec_proposed: u64,
    spec_accepted: u64,
}

/// One more verified row costs a fraction of an ordinary decode step
/// (measured ~0.2 on the reference machine); a draft row is worth its rows
/// while the chain-acceptance probability of reaching it clears that ratio.
const MARGINAL_ROW_ACCEPTANCE: f64 = 0.2;

/// Deterministic adaptive verify width: start on the efficient M=8 arm, then
/// use the widest width whose marginal draft rows clear the cumulative
/// acceptance rate. Deterministic given the session's token stream.
fn select_verify_rows(proposed: u64, accepted: u64, max_rows: usize) -> usize {
    if proposed == 0 {
        return max_rows.min(8);
    }
    let acceptance = accepted as f64 / proposed as f64;
    let mut rows = 2;
    while rows < max_rows && acceptance.powi((rows - 1) as i32) >= MARGINAL_ROW_ACCEPTANCE {
        rows += 1;
    }
    rows
}

#[napi]
pub struct NativeInferenceSequence {
    session_id: u64,
    sequence_id: u64,
}

#[napi]
impl NativeInferenceSequence {
    #[napi(getter)]
    pub fn sequence_id(&self) -> BigInt {
        bigint(self.sequence_id)
    }
}

#[napi(object)]
pub struct NativeInferencePage {
    pub sequence_id: BigInt,
    pub tokens: Vec<u32>,
    pub stop_reason: Option<String>,
}

#[napi(object)]
pub struct NativeInferenceRoundResult {
    pub round_id: BigInt,
    pub recovered: bool,
    pub pages: Vec<NativeInferencePage>,
}

#[napi(object)]
pub struct NativeInferenceInspection {
    pub sequence_id: BigInt,
    pub cursor: BigInt,
    pub terminal: Option<String>,
}

fn native_receipt(
    _session_id: u64,
    receipt: &Receipt,
    recovered: bool,
) -> NativeInferenceRoundResult {
    NativeInferenceRoundResult {
        round_id: bigint(receipt.round_id),
        recovered,
        pages: receipt
            .pages
            .iter()
            .map(|page| NativeInferencePage {
                sequence_id: bigint(page.sequence_id),
                tokens: page.tokens.clone(),
                stop_reason: page.stop_reason.clone(),
            })
            .collect(),
    }
}

fn next_monotonic(counter: &AtomicU64, name: &str) -> Result<u64> {
    counter
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |value| {
            value.checked_add(1)
        })
        .map_err(|_| {
            Error::new(
                Status::GenericFailure,
                format!("inference: {name} exhausted"),
            )
        })
}

#[napi]
pub struct NativeInferenceSession {
    programs: Arc<InferencePrograms>,
    next_sequence_id: Arc<AtomicU64>,
    next_round_id: Arc<AtomicU64>,
    diagnostics: Arc<InferenceDiagnosticsState>,
    state: Arc<Mutex<InferenceSessionState>>,
}

#[napi]
impl NativeInferenceSession {
    #[napi]
    pub async fn add(
        &self,
        prompts: Vec<&NativeTensor>,
        sampling: Vec<NativeInferenceSamplingOverride>,
        max_tokens: Vec<Option<u32>>,
        eos_tokens: Vec<Vec<u32>>,
        cancellation_token: Option<&CancellationToken>,
    ) -> Result<NativeInferenceRoundResult> {
        if prompts.is_empty()
            || prompts.len() != sampling.len()
            || prompts.len() != max_tokens.len()
            || prompts.len() != eos_tokens.len()
            || max_tokens.iter().flatten().any(|value| *value == 0)
        {
            return Err(Error::new(
                Status::InvalidArg,
                "inference[admission]: add arrays must be nonempty and equal length with positive maxTokens",
            ));
        }
        if prompts
            .iter()
            .any(|prompt| prompt.device_ordinal != self.programs.device_ordinal)
        {
            return Err(Error::new(
                Status::InvalidArg,
                "inference[admission]: prompts and artifact must use the same Metal device",
            ));
        }
        {
            let state = self.state.lock().map_err(|error| {
                Error::new(
                    Status::GenericFailure,
                    format!("inference session lock: {error}"),
                )
            })?;
            if state.closed {
                return Err(Error::new(
                    Status::GenericFailure,
                    "inference[admission]: session is closed",
                ));
            }
            if state.receipt.is_none()
                && prompts.len() > state.lanes.iter().filter(|lane| lane.is_none()).count()
            {
                return Err(Error::new(
                    Status::InvalidArg,
                    "inference[admission]: insufficient free lanes",
                ));
            }
        }
        let prompts = prompts
            .iter()
            .map(|prompt| {
                read_prompt(
                    prompt,
                    self.programs.target_prefill.token_dtype,
                    self.programs.target_prefill.schema.max_tokens,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let sampling = sampling
            .into_iter()
            .map(|value| inference_sampling_override(self.programs.sampling, value))
            .collect::<Result<Vec<_>>>()?;
        let fingerprint = add_fingerprint(&prompts, &sampling, &max_tokens, &eos_tokens);
        let programs = self.programs.clone();
        let state = self.state.clone();
        let diagnostics = self.diagnostics.clone();
        let sequence_counter = self.next_sequence_id.clone();
        let round_counter = self.next_round_id.clone();
        let failure_diagnostics = diagnostics.clone();
        let device_ordinal = programs.device_ordinal;
        let compute = move |cancelled: &effect_torch_runtime::CancellationFlag,
                            cancellation: &CancellationState| {
            let mut session = state.lock().map_err(|error| {
                Error::new(
                    Status::GenericFailure,
                    format!("inference session lock: {error}"),
                )
            })?;
            if session.closed {
                return Err(Error::new(
                    Status::GenericFailure,
                    "inference[admission]: session is closed",
                ));
            }
            if let Some(receipt) = recover_receipt(&session.receipt, fingerprint)? {
                diagnostics.rounds_recovered.fetch_add(1, Ordering::Relaxed);
                return Ok(native_receipt(session.id, receipt, true));
            }
            diagnostics.rounds_started.fetch_add(1, Ordering::Relaxed);
            let slots = session
                .lanes
                .iter()
                .enumerate()
                .filter_map(|(slot, lane)| lane.is_none().then_some(slot))
                .take(prompts.len())
                .collect::<Vec<_>>();
            if slots.len() != prompts.len() {
                return Err(Error::new(
                    Status::InvalidArg,
                    "inference[admission]: insufficient free lanes",
                ));
            }
            let ids = (0..prompts.len())
                .map(|_| next_monotonic(&sequence_counter, "sequence IDs"))
                .collect::<Result<Vec<_>>>()?;
            let target = (0..prompts.len())
                .map(|_| make_managed_sequence(&programs.target_pool))
                .collect::<Result<Vec<_>>>()?;
            if programs.replay_prefill.is_none() {
                for (sequence, prompt) in target.iter().zip(&prompts) {
                    sequence.prefill_match(prompt.clone())?;
                }
            }
            let target_states = target
                .iter()
                .map(|sequence| sequence.state.clone())
                .collect::<Vec<_>>();
            let mut target_shadow =
                ShadowSequences::new(programs.target_pool.clone(), &target_states)
                    .map_err(to_napi_err)?;
            let (proposer, mut proposer_shadow, proposer_states) =
                if let (Some(prefill), Some(pool)) =
                    (&programs.proposer_prefill, &programs.proposer_pool)
                {
                    let sequences = (0..prompts.len())
                        .map(|_| make_managed_sequence(pool))
                        .collect::<Result<Vec<_>>>()?;
                    for (sequence, prompt) in sequences.iter().zip(&prompts) {
                        sequence.prefill_match(prompt.clone())?;
                    }
                    let states = sequences
                        .iter()
                        .map(|sequence| sequence.state.clone())
                        .collect::<Vec<_>>();
                    let shadow =
                        ShadowSequences::new(pool.clone(), &states).map_err(to_napi_err)?;
                    // Proposer programs are never bucketed.
                    let proposer_bucket = PrefillBucket {
                        prefill: prefill.clone(),
                        replay: None,
                    };
                    run_prefill_program(
                        std::slice::from_ref(&proposer_bucket),
                        &shadow,
                        &slots,
                        &prompts,
                        cancelled,
                    )
                    .map_err(to_napi_err)?;
                    (Some(sequences), Some(shadow), states)
                } else if let Some(pool) = &programs.replay_pool {
                    let sequences = (0..prompts.len())
                        .map(|_| make_managed_sequence(pool))
                        .collect::<Result<Vec<_>>>()?;
                    let states = sequences
                        .iter()
                        .map(|sequence| sequence.state.clone())
                        .collect::<Vec<_>>();
                    let shadow =
                        ShadowSequences::new(pool.clone(), &states).map_err(to_napi_err)?;
                    (Some(sequences), Some(shadow), states)
                } else {
                    (None, None, Vec::new())
                };
            let outputs = if let (Some(_), Some(plan), Some(proposer_shadow)) = (
                &programs.replay_prefill,
                &programs.proposer_plan,
                proposer_shadow.as_ref(),
            ) {
                run_parallel_prefill(
                    &programs.prefill_buckets,
                    plan,
                    &target_shadow,
                    proposer_shadow,
                    &slots,
                    &prompts,
                    cancelled,
                )
            } else {
                run_prefill_program(
                    &programs.prefill_buckets,
                    &target_shadow,
                    &slots,
                    &prompts,
                    cancelled,
                )
            }
            .map_err(to_napi_err)?;
            let mut pages = Vec::with_capacity(prompts.len());
            let mut policies = Vec::with_capacity(prompts.len());
            for index in 0..prompts.len() {
                let logits = read_float_tensor(&outputs[index]).map_err(to_napi_err)?;
                let probabilities =
                    probabilities(&logits, inference_options(sampling[index]), cancelled)
                        .map_err(to_napi_err)?;
                let token = sample_at(
                    &probabilities,
                    sampling[index],
                    ids[index],
                    0,
                    SamplingPurpose::Target,
                    0,
                );
                let terminal = if eos_tokens[index].contains(&token) {
                    Some("eos".to_string())
                } else if max_tokens[index] == Some(1) {
                    Some("maxTokens".to_string())
                } else {
                    None
                };
                pages.push(ReceiptPage {
                    sequence_id: ids[index],
                    tokens: vec![token],
                    stop_reason: terminal.clone(),
                });
                policies.push((token, terminal));
            }
            if !cancellation.complete() {
                return Err(Error::new(Status::GenericFailure, "operation aborted"));
            }
            let round_id = next_monotonic(&round_counter, "round IDs")?;
            let mut provisional_blocks = target_shadow.provisional_blocks(&target_states);
            if let Some(shadow) = proposer_shadow.as_ref() {
                provisional_blocks += shadow.provisional_blocks(&proposer_states);
            }
            let target_cache = stage_cache_metadata(&target_shadow, &prompts);
            let proposer_cache = proposer_shadow
                .as_ref()
                .map(|shadow| stage_cache_metadata(shadow, &prompts));
            publish_shadow(&mut target_shadow, &target_states);
            if let Some(shadow) = proposer_shadow.as_mut() {
                publish_shadow(shadow, &proposer_states);
            }
            let mut target = target.into_iter();
            let mut proposer = proposer.map(Vec::into_iter);
            for index in 0..prompts.len() {
                session.lanes[slots[index]] = Some(InferenceLane {
                    sequence_id: ids[index],
                    target: target.next().expect("target sequence exists"),
                    proposer: proposer
                        .as_mut()
                        .map(|values| values.next().expect("proposer sequence exists")),
                    pending: policies[index].0,
                    generated: 1,
                    max_tokens: max_tokens[index].map(u64::from),
                    eos: eos_tokens[index].clone(),
                    terminal: policies[index].1.clone(),
                    // Admission overrides apply only to the first page; later
                    // rounds resolve sparse overrides from artifact defaults.
                    sampling: programs.sampling,
                    history: prompts[index]
                        .iter()
                        .copied()
                        .chain(std::iter::once(policies[index].0))
                        .collect(),
                });
            }
            let receipt = Receipt {
                round_id,
                fingerprint,
                pages,
            };
            diagnostics.rounds_completed.fetch_add(1, Ordering::Relaxed);
            diagnostics
                .emitted_tokens
                .fetch_add(prompts.len() as u64, Ordering::Relaxed);
            diagnostics
                .provisional_blocks
                .fetch_add(provisional_blocks, Ordering::Relaxed);
            diagnostics
                .target_pool_high_water_blocks
                .fetch_max(programs.target_pool.live_blocks(), Ordering::Relaxed);
            if let Some(pool) = programs
                .proposer_pool
                .as_ref()
                .or(programs.replay_pool.as_ref())
            {
                diagnostics
                    .proposer_pool_high_water_blocks
                    .fetch_max(pool.live_blocks(), Ordering::Relaxed);
            }
            diagnostics.last_round_id.store(round_id, Ordering::Relaxed);
            diagnostics.has_round.store(true, Ordering::Relaxed);
            session.receipt = Some(receipt.clone());
            publish_paired_cache_metadata(&target_cache, proposer_cache.as_ref());
            Ok(native_receipt(session.id, &receipt, false))
        };
        let result = run_compute_on(device_ordinal, cancellation_token, compute).await;
        if let Err(error) = &result {
            record_inference_failure(&failure_diagnostics, error, "prefill");
        }
        result
    }

    #[napi]
    pub async fn run_round(
        &self,
        sequences: Vec<&NativeInferenceSequence>,
        sampling: Vec<NativeInferenceSamplingOverride>,
        cancellation_token: Option<&CancellationToken>,
    ) -> Result<NativeInferenceRoundResult> {
        if sequences.is_empty() || sequences.len() != sampling.len() {
            return Err(Error::new(
                Status::InvalidArg,
                "inference[admission]: round arrays must be nonempty and equal",
            ));
        }
        let identities = sequences
            .iter()
            .map(|sequence| (sequence.session_id, sequence.sequence_id))
            .collect::<Vec<_>>();
        let programs = self.programs.clone();
        let state = self.state.clone();
        let diagnostics = self.diagnostics.clone();
        let round_counter = self.next_round_id.clone();
        let failure_diagnostics = diagnostics.clone();
        let device_ordinal = programs.device_ordinal;
        let compute = move |cancelled: &effect_torch_runtime::CancellationFlag,
                            cancellation: &CancellationState| {
            let mut session = state.lock().map_err(|error| {
                Error::new(
                    Status::GenericFailure,
                    format!("inference session lock: {error}"),
                )
            })?;
            if session.closed {
                return Err(Error::new(
                    Status::GenericFailure,
                    "inference[admission]: session is closed",
                ));
            }
            let mut slots = Vec::with_capacity(identities.len());
            for (owner, id) in &identities {
                if *owner != session.id
                    || slots.iter().any(|slot: &usize| {
                        session.lanes[*slot]
                            .as_ref()
                            .is_some_and(|lane| lane.sequence_id == *id)
                    })
                {
                    return Err(Error::new(
                        Status::InvalidArg,
                        "inference[admission]: foreign or duplicate sequence",
                    ));
                }
                let slot = session
                    .lanes
                    .iter()
                    .position(|lane| lane.as_ref().is_some_and(|lane| lane.sequence_id == *id))
                    .ok_or_else(|| {
                        Error::new(
                            Status::InvalidArg,
                            "inference[admission]: sequence is not live",
                        )
                    })?;
                if session.lanes[slot].as_ref().unwrap().terminal.is_some() {
                    return Err(Error::new(
                        Status::InvalidArg,
                        "inference[admission]: sequence is terminal",
                    ));
                }
                slots.push(slot);
            }
            let selected = slots
                .iter()
                .map(|slot| session.lanes[*slot].as_ref().unwrap())
                .collect::<Vec<_>>();
            let sampling = sampling
                .into_iter()
                .zip(&selected)
                .map(|(value, lane)| inference_sampling_override(lane.sampling, value))
                .collect::<Result<Vec<_>>>()?;
            let ids = selected
                .iter()
                .map(|lane| lane.sequence_id)
                .collect::<Vec<_>>();
            let fingerprint = round_fingerprint(&ids, &sampling);
            if let Some(receipt) = recover_receipt(&session.receipt, fingerprint)? {
                diagnostics.rounds_recovered.fetch_add(1, Ordering::Relaxed);
                return Ok(native_receipt(session.id, receipt, true));
            }
            diagnostics.rounds_started.fetch_add(1, Ordering::Relaxed);
            let target_states = selected
                .iter()
                .map(|lane| lane.target.state.clone())
                .collect::<Vec<_>>();
            let mut target_shadow =
                ShadowSequences::new(programs.target_pool.clone(), &target_states)
                    .map_err(to_napi_err)?;
            let pending = selected.iter().map(|lane| lane.pending).collect::<Vec<_>>();
            let positions = selected
                .iter()
                .map(|lane| lane.generated)
                .collect::<Vec<_>>();
            let page_limits = selected
                .iter()
                .map(|lane| {
                    lane.max_tokens
                        .map_or(programs.max_draft_tokens as u64 + 1, |limit| {
                            limit - lane.generated
                        })
                        .min(programs.max_draft_tokens as u64 + 1)
                        .max(1) as u32
                })
                .collect::<Vec<_>>();
            let eos = selected
                .iter()
                .map(|lane| lane.eos.clone())
                .collect::<Vec<_>>();
            let options = sampling
                .iter()
                .copied()
                .map(inference_options)
                .collect::<Vec<_>>();
            let parallel = (
                (!programs.target_verify.is_empty()).then_some(&programs.target_verify),
                (!programs.replay_verify.is_empty()).then_some(&programs.replay_verify),
                programs.replay_pool.as_ref(),
                programs.proposer_plan.as_ref(),
            );
            let (pages, mut proposer_shadow, proposer_states) =
                if let (Some(verifies), Some(replays), Some(pool), Some(plan)) = parallel {
                    // Adaptive verify width from the session's cumulative
                    // acceptance, rounded down to a compiled width.
                    let wanted = select_verify_rows(
                        session.spec_proposed,
                        session.spec_accepted,
                        verifies
                            .last()
                            .and_then(|verify| verify.packed_rows_per_sequence)
                            .unwrap_or(1),
                    );
                    let arm = verifies
                        .iter()
                        .rposition(|verify| {
                            verify
                                .packed_rows_per_sequence
                                .is_some_and(|rows| rows <= wanted)
                        })
                        .unwrap_or(0);
                    let verify = &verifies[arm];
                    let arm_drafts = verify
                        .packed_rows_per_sequence
                        .expect("verified packed layout")
                        - 1;
                    let proposer_states = selected
                        .iter()
                        .map(|lane| {
                            lane.proposer
                                .as_ref()
                                .expect("validated ParallelBlock proposer")
                                .state
                                .clone()
                        })
                        .collect::<Vec<_>>();
                    for (target, proposer) in target_states.iter().zip(&proposer_states) {
                        let target_cursor = target
                            .lock()
                            .map_err(|error| Error::new(Status::GenericFailure, error.to_string()))?
                            .cursor;
                        let proposer_cursor = proposer
                            .lock()
                            .map_err(|error| Error::new(Status::GenericFailure, error.to_string()))?
                            .cursor;
                        if target_cursor != proposer_cursor {
                            return Err(Error::new(
                                Status::GenericFailure,
                                "inference[proposal]: target/proposer cursors diverged",
                            ));
                        }
                    }
                    let proposal_shadow = ShadowSequences::new(pool.clone(), &proposer_states)
                        .map_err(to_napi_err)?;
                    let replay_shadow = ShadowSequences::new(pool.clone(), &proposer_states)
                        .map_err(to_napi_err)?;
                    let mut stats = SpeculativeStats {
                        accepted: vec![0; selected.len()],
                        ..SpeculativeStats::default()
                    };
                    let pages = execute_parallel_blocking(
                        verify,
                        &replays[arm],
                        plan,
                        &mut target_shadow,
                        &proposal_shadow,
                        &replay_shadow,
                        &slots,
                        &pending,
                        &options,
                        programs.max_draft_tokens.min(arm_drafts),
                        &page_limits,
                        &eos,
                        cancelled,
                        &ids,
                        &positions,
                        &mut stats,
                    )
                    .map_err(to_napi_err)?;
                    session.spec_proposed += stats.proposed as u64;
                    session.spec_accepted += stats.accepted.iter().sum::<usize>() as u64;
                    let accepted_total = stats.accepted.iter().sum::<usize>() as u64;
                    diagnostics
                        .speculative_rounds
                        .fetch_add(1, Ordering::Relaxed);
                    diagnostics
                        .proposed_tokens
                        .fetch_add(stats.proposed as u64, Ordering::Relaxed);
                    diagnostics
                        .accepted_tokens
                        .fetch_add(accepted_total, Ordering::Relaxed);
                    diagnostics
                        .draft_nanos
                        .fetch_add(stats.draft_nanos, Ordering::Relaxed);
                    diagnostics
                        .verification_nanos
                        .fetch_add(stats.verification_nanos, Ordering::Relaxed);
                    if let Ok(mut histogram) = diagnostics.accepted_length_histogram.lock() {
                        for accepted in stats.accepted {
                            if let Some(count) = histogram.get_mut(accepted) {
                                *count += 1;
                            }
                        }
                    }
                    (pages, Some(replay_shadow), proposer_states)
                } else if let (Some(verify), Some(config)) =
                    (&programs.target_verify.last(), programs.history_lookup)
                {
                    let histories = selected
                        .iter()
                        .map(|lane| lane.history.clone())
                        .collect::<Vec<_>>();
                    let mut stats = SpeculativeStats {
                        accepted: vec![0; selected.len()],
                        ..SpeculativeStats::default()
                    };
                    let pages = execute_history_lookup_blocking(
                        verify,
                        &mut target_shadow,
                        &slots,
                        &pending,
                        &histories,
                        &options,
                        config,
                        programs.max_draft_tokens,
                        &page_limits,
                        &eos,
                        cancelled,
                        &ids,
                        &positions,
                        &mut stats,
                    )
                    .map_err(to_napi_err)?;
                    diagnostics
                        .speculative_rounds
                        .fetch_add(1, Ordering::Relaxed);
                    diagnostics
                        .proposed_tokens
                        .fetch_add(stats.proposed as u64, Ordering::Relaxed);
                    diagnostics.accepted_tokens.fetch_add(
                        stats.accepted.iter().sum::<usize>() as u64,
                        Ordering::Relaxed,
                    );
                    diagnostics
                        .draft_nanos
                        .fetch_add(stats.draft_nanos, Ordering::Relaxed);
                    diagnostics
                        .verification_nanos
                        .fetch_add(stats.verification_nanos, Ordering::Relaxed);
                    if let Ok(mut histogram) = diagnostics.accepted_length_histogram.lock() {
                        for accepted in stats.accepted {
                            if let Some(count) = histogram.get_mut(accepted) {
                                *count += 1;
                            }
                        }
                    }
                    (pages, None, Vec::new())
                } else if let (Some(verify), Some(decode), Some(pool)) = (
                    &programs.target_verify.last(),
                    &programs.proposer_decode,
                    &programs.proposer_pool,
                ) {
                    let proposer_states = selected
                        .iter()
                        .map(|lane| {
                            lane.proposer
                                .as_ref()
                                .expect("validated proposer")
                                .state
                                .clone()
                        })
                        .collect::<Vec<_>>();
                    let mut proposer_shadow = ShadowSequences::new(pool.clone(), &proposer_states)
                        .map_err(to_napi_err)?;
                    let mut stats = SpeculativeStats {
                        accepted: vec![0; selected.len()],
                        ..SpeculativeStats::default()
                    };
                    let pages = execute_speculative_blocking(
                        verify,
                        decode,
                        &mut target_shadow,
                        &mut proposer_shadow,
                        &slots,
                        &pending,
                        &options,
                        programs.max_draft_tokens,
                        &page_limits,
                        &eos,
                        cancelled,
                        Some((&ids, &positions)),
                        Some(&mut stats),
                    )
                    .map_err(to_napi_err)?;
                    diagnostics
                        .speculative_rounds
                        .fetch_add(1, Ordering::Relaxed);
                    diagnostics
                        .proposed_tokens
                        .fetch_add(stats.proposed as u64, Ordering::Relaxed);
                    diagnostics.accepted_tokens.fetch_add(
                        stats.accepted.iter().sum::<usize>() as u64,
                        Ordering::Relaxed,
                    );
                    diagnostics
                        .draft_nanos
                        .fetch_add(stats.draft_nanos, Ordering::Relaxed);
                    diagnostics
                        .verification_nanos
                        .fetch_add(stats.verification_nanos, Ordering::Relaxed);
                    if let Ok(mut histogram) = diagnostics.accepted_length_histogram.lock() {
                        for accepted in stats.accepted {
                            if let Some(count) = histogram.get_mut(accepted) {
                                *count += 1;
                            }
                        }
                    }
                    (pages, Some(proposer_shadow), proposer_states)
                } else {
                    diagnostics.ordinary_rounds.fetch_add(1, Ordering::Relaxed);
                    let fused_sampling = sampling
                        .iter()
                        .enumerate()
                        .map(|(index, sampling)| SamplingOptions {
                            seed: coordinate_seed(
                                sampling.seed,
                                ids[index],
                                positions[index],
                                SamplingPurpose::Target,
                                0,
                            ),
                            counter: 0,
                            ..inference_options(*sampling)
                        })
                        .collect::<Vec<_>>();
                    let sampled = run_sampled_program(
                        &programs.target_decode,
                        &target_shadow,
                        &slots,
                        pending.iter().map(|token| vec![*token]).collect(),
                        &fused_sampling,
                        cancelled,
                    )
                    .map_err(to_napi_err)?;
                    (
                        sampled.into_iter().map(|token| vec![token]).collect(),
                        None,
                        Vec::new(),
                    )
                };
            if !cancellation.complete() {
                return Err(Error::new(Status::GenericFailure, "operation aborted"));
            }
            let round_id = next_monotonic(&round_counter, "round IDs")?;
            let mut provisional_blocks = target_shadow.provisional_blocks(&target_states);
            if let Some(shadow) = proposer_shadow.as_ref() {
                provisional_blocks += shadow.provisional_blocks(&proposer_states);
            }
            let mut rolled_back =
                publish_speculative_states(&mut target_shadow, &target_states, &pending, &pages);
            if let Some(shadow) = proposer_shadow.as_mut() {
                rolled_back +=
                    publish_speculative_states(shadow, &proposer_states, &pending, &pages);
            }
            let mut receipt_pages = Vec::with_capacity(pages.len());
            for (index, page) in pages.into_iter().enumerate() {
                let lane = session.lanes[slots[index]].as_mut().unwrap();
                lane.pending = *page.last().expect("native pages are nonempty");
                lane.generated += page.len() as u64;
                lane.history.extend_from_slice(&page);
                let terminal = if page.iter().any(|token| lane.eos.contains(token)) {
                    Some("eos".to_string())
                } else if lane.max_tokens.is_some_and(|limit| lane.generated >= limit) {
                    Some("maxTokens".to_string())
                } else {
                    None
                };
                lane.terminal = terminal.clone();
                receipt_pages.push(ReceiptPage {
                    sequence_id: lane.sequence_id,
                    tokens: page,
                    stop_reason: terminal,
                });
            }
            let receipt = Receipt {
                round_id,
                fingerprint,
                pages: receipt_pages,
            };
            diagnostics.rounds_completed.fetch_add(1, Ordering::Relaxed);
            diagnostics.emitted_tokens.fetch_add(
                receipt
                    .pages
                    .iter()
                    .map(|page| page.tokens.len() as u64)
                    .sum(),
                Ordering::Relaxed,
            );
            diagnostics
                .provisional_blocks
                .fetch_add(provisional_blocks, Ordering::Relaxed);
            diagnostics
                .rolled_back_blocks
                .fetch_add(rolled_back, Ordering::Relaxed);
            diagnostics
                .target_pool_high_water_blocks
                .fetch_max(programs.target_pool.live_blocks(), Ordering::Relaxed);
            if let Some(pool) = programs
                .proposer_pool
                .as_ref()
                .or(programs.replay_pool.as_ref())
            {
                diagnostics
                    .proposer_pool_high_water_blocks
                    .fetch_max(pool.live_blocks(), Ordering::Relaxed);
            }
            diagnostics.last_round_id.store(round_id, Ordering::Relaxed);
            diagnostics.has_round.store(true, Ordering::Relaxed);
            session.receipt = Some(receipt.clone());
            Ok(native_receipt(session.id, &receipt, false))
        };
        let result = run_compute_on(device_ordinal, cancellation_token, compute).await;
        if let Err(error) = &result {
            record_inference_failure(&failure_diagnostics, error, "verify");
        }
        result
    }

    #[napi]
    pub fn acknowledge(&self, round_id: BigInt) -> Result<()> {
        let (negative, round_id, lossless) = round_id.get_u64();
        if negative || !lossless {
            return Err(Error::new(
                Status::InvalidArg,
                "inference[publish]: invalid round ID",
            ));
        }
        let mut state = self.state.lock().map_err(|error| {
            Error::new(
                Status::GenericFailure,
                format!("inference session lock: {error}"),
            )
        })?;
        acknowledge_receipt(&mut state.receipt, round_id)
    }

    #[napi]
    pub fn sequence(&self, sequence_id: BigInt) -> Result<NativeInferenceSequence> {
        let (negative, sequence_id, lossless) = sequence_id.get_u64();
        if negative || !lossless {
            return Err(Error::new(
                Status::InvalidArg,
                "inference[inspect]: invalid sequence ID",
            ));
        }
        let state = self.state.lock().map_err(|error| {
            Error::new(
                Status::GenericFailure,
                format!("inference session lock: {error}"),
            )
        })?;
        if state.closed
            || !state
                .lanes
                .iter()
                .flatten()
                .any(|lane| lane.sequence_id == sequence_id)
        {
            return Err(Error::new(
                Status::InvalidArg,
                "inference[inspect]: sequence is not live",
            ));
        }
        Ok(NativeInferenceSequence {
            session_id: state.id,
            sequence_id,
        })
    }

    #[napi]
    pub fn finish(&self, sequences: Vec<&NativeInferenceSequence>) -> Result<()> {
        let mut state = self.state.lock().map_err(|error| {
            Error::new(
                Status::GenericFailure,
                format!("inference session lock: {error}"),
            )
        })?;
        if state.closed {
            return Err(Error::new(
                Status::GenericFailure,
                "inference[finish]: session is closed",
            ));
        }
        if state.receipt.is_some() {
            return Err(Error::new(
                Status::GenericFailure,
                "inference[finish]: unacknowledged receipt",
            ));
        }
        let mut slots = Vec::with_capacity(sequences.len());
        for sequence in sequences {
            if sequence.session_id != state.id {
                return Err(Error::new(
                    Status::InvalidArg,
                    "inference[finish]: foreign sequence",
                ));
            }
            let slot = state
                .lanes
                .iter()
                .position(|lane| {
                    lane.as_ref()
                        .is_some_and(|lane| lane.sequence_id == sequence.sequence_id)
                })
                .ok_or_else(|| {
                    Error::new(
                        Status::InvalidArg,
                        "inference[finish]: sequence is not live",
                    )
                })?;
            if slots.contains(&slot) {
                return Err(Error::new(
                    Status::InvalidArg,
                    "inference[finish]: duplicate sequence",
                ));
            }
            slots.push(slot);
        }
        for slot in slots {
            state.lanes[slot] = None;
        }
        Ok(())
    }

    #[napi]
    pub fn inspect(&self, sequence: &NativeInferenceSequence) -> Result<NativeInferenceInspection> {
        let state = self.state.lock().map_err(|error| {
            Error::new(
                Status::GenericFailure,
                format!("inference session lock: {error}"),
            )
        })?;
        if state.closed {
            return Err(Error::new(
                Status::GenericFailure,
                "inference[inspect]: session is closed",
            ));
        }
        if sequence.session_id != state.id {
            return Err(Error::new(
                Status::InvalidArg,
                "inference[inspect]: foreign sequence",
            ));
        }
        let lane = state
            .lanes
            .iter()
            .flatten()
            .find(|lane| lane.sequence_id == sequence.sequence_id)
            .ok_or_else(|| {
                Error::new(
                    Status::InvalidArg,
                    "inference[inspect]: sequence is not live",
                )
            })?;
        Ok(NativeInferenceInspection {
            sequence_id: bigint(lane.sequence_id),
            cursor: bigint(lane.target.cursor() as u64),
            terminal: lane.terminal.clone(),
        })
    }

    #[napi]
    pub fn close(&self) -> Result<()> {
        let mut state = self.state.lock().map_err(|error| {
            Error::new(
                Status::GenericFailure,
                format!("inference session lock: {error}"),
            )
        })?;
        state.lanes.fill_with(|| None);
        state.receipt = None;
        state.closed = true;
        Ok(())
    }
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
        seqs: Vec<&NativeKvSequence>,
        slots: Vec<u32>,
        active_mask: Vec<bool>,
        valid_lengths: Vec<u32>,
        advances: Vec<u32>,
        tokens: Vec<Vec<u32>>,
        invocation: StatefulInvocation,
        token: Option<&CancellationToken>,
    ) -> Result<StatefulExecutionOutput> {
        let batch = self.state.as_ref().expect("state checked").schema.batch;
        if inputs
            .iter()
            .any(|input| input.device_ordinal != self.device_ordinal)
            || seqs
                .iter()
                .any(|sequence| sequence.pool.device_ordinal != self.device_ordinal)
        {
            return Err(Error::new(
                Status::InvalidArg,
                "execute: inputs, state, and executable must use the same Metal device",
            ));
        }
        if seqs.is_empty() || seqs.len() > batch {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "kv run: program accepts 1..={} sequences, got {}",
                    batch,
                    seqs.len()
                ),
            ));
        }
        self.execute_stateful_inner(
            inputs,
            seqs,
            slots,
            active_mask,
            valid_lengths,
            advances,
            tokens,
            invocation,
            token,
        )
        .await
    }
}

impl Executable {
    async fn execute_stateful_inner(
        &self,
        inputs: Vec<&NativeTensor>,
        seqs: Vec<&NativeKvSequence>,
        slots: Vec<u32>,
        active_mask: Vec<bool>,
        valid_lengths: Vec<u32>,
        advances: Vec<u32>,
        tokens: Vec<Vec<u32>>,
        invocation: StatefulInvocation,
        token: Option<&CancellationToken>,
    ) -> Result<StatefulExecutionOutput> {
        let batch = self.state.as_ref().expect("state checked").schema.batch;
        if tokens.len() != seqs.len() || tokens.iter().any(|t| t.is_empty()) {
            return Err(Error::new(
                Status::InvalidArg,
                "kv run: expected one non-empty token list per sequence".to_string(),
            ));
        }
        let lanes = validate_fixed_lanes(
            batch,
            seqs.len(),
            &slots,
            &active_mask,
            &valid_lengths,
            &advances,
            &tokens,
        )?;
        for (i, seq) in seqs.iter().enumerate() {
            if seq.released.load(Ordering::SeqCst) {
                return Err(Error::new(
                    Status::GenericFailure,
                    format!("kv sequence {i} is released"),
                ));
            }
            if !Arc::ptr_eq(&seq.pool, &seqs[0].pool) {
                return Err(Error::new(
                    Status::InvalidArg,
                    "kv run: batched sequences must share one pool".to_string(),
                ));
            }
            if seqs[..i]
                .iter()
                .any(|other| Arc::ptr_eq(&other.state, &seq.state))
            {
                return Err(Error::new(
                    Status::InvalidArg,
                    "kv run: duplicate sequence in batch".to_string(),
                ));
            }
        }
        let stateful = self.state.as_ref().expect("state checked");
        let schema = stateful.schema;
        let pool = &seqs[0].pool;
        if pool.max_tokens != schema.max_tokens
            || pool.block_size != schema.block_size
            || pool.dtype != schema.kv_dtype
            || pool.k.len() != schema.layers
            || pool.kv_heads != schema.kv_heads
            || pool.head_dim != schema.head_dim
            || pool.kda != schema.kda
            || pool.conv != schema.conv
        {
            return Err(Error::new(
                Status::InvalidArg,
                "kv run: pool geometry/dtype does not match the compiled state schema".to_string(),
            ));
        }
        let inner = &self.inner;
        let runtime_values = usize::from(inner.slots.get(stateful.cursor_slot as usize).is_some());
        inner
            .executable
            .signature
            .validate_invocation_counts(inputs.len(), 0, runtime_values, None)
            .map_err(|error| Error::new(Status::InvalidArg, error.to_string()))?;
        for (slot, declared) in inner.slots.iter().enumerate() {
            if declared.scalar || (stateful.cursor_tensor && slot as u32 == stateful.cursor_slot) {
                continue;
            }
            let input_index = slot
                - inner.slots.iter().take(slot).filter(|s| s.scalar).count()
                - usize::from(stateful.cursor_tensor && (slot as u32) > stateful.cursor_slot);
            validate_stateful_tensor_input(inputs[input_index], slot, declared)?;
        }
        let executable = inner.executable.clone();
        let generated = inner.generated_bindings.clone();
        let inputs: Vec<value::Value> = inputs
            .iter()
            .map(|input| input.val_cloned())
            .collect::<Result<Vec<_>>>()?;
        let kv = Arc::new(KvContext {
            pool: seqs[0].pool.clone(),
            slots: seqs.iter().map(|seq| seq.state.clone()).collect(),
            schema,
            tokens: tokens.clone(),
            lanes: lanes.clone(),
            packed_rows: None,
            packed_positions: None,
            publish_hashes: true,
            state_only: false,
        });
        // Lock every sequence in address order; overlapping batches
        // acquire the same locks in the same order, so no deadlock.
        let mut ordered: Vec<&NativeKvSequence> = seqs.clone();
        ordered.sort_by_key(|seq| Arc::as_ptr(&seq.run_lock) as usize);
        let run_locks: Vec<Arc<Mutex<()>>> =
            ordered.iter().map(|seq| seq.run_lock.clone()).collect();
        let released: Vec<Arc<AtomicBool>> = seqs.iter().map(|seq| seq.released.clone()).collect();
        let slot_states: Vec<Arc<Mutex<SeqState>>> =
            seqs.iter().map(|seq| seq.state.clone()).collect();
        let device_ordinal = self.device_ordinal;
        run_compute_on(device_ordinal, token, move |cancelled, cancellation| {
            let _run_guards: Vec<_> = run_locks
                .iter()
                .map(|lock| {
                    lock.lock().map_err(|e| {
                        Error::new(
                            Status::GenericFailure,
                            format!("kv sequence lock poisoned: {e}"),
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
            for (index, state) in slot_states.iter().enumerate() {
                let cursor = state
                    .lock()
                    .map_err(|e| {
                        Error::new(
                            Status::GenericFailure,
                            format!("kv sequence lock poisoned: {e}"),
                        )
                    })?
                    .cursor;
                let frontier = cursor.checked_add(tokens[index].len());
                if frontier.is_none()
                    || (kv.schema.window.is_none()
                        && frontier.is_some_and(|value| value > kv.schema.max_tokens))
                {
                    return Err(Error::new(
                        Status::InvalidArg,
                        format!(
                            "kv run: sequence {index} exceeds compiled max_tokens {}",
                            kv.schema.max_tokens
                        ),
                    ));
                }
            }
            let checkpoints: Vec<SequenceCheckpoint> = slot_states
                .iter()
                .map(|state| {
                    let state = state.lock().map_err(|e| {
                        Error::new(
                            Status::GenericFailure,
                            format!("kv sequence lock poisoned: {e}"),
                        )
                    })?;
                    Ok(SequenceCheckpoint {
                        blocks: state.blocks.len(),
                        advance: state.advance,
                        kda_states: state.kda_states.len(),
                        conv_states: state.conv_states.len(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let rollback = || {
                rollback_sequence_setup(&kv.pool, &slot_states, &checkpoints).map_err(to_napi_err)
            };
            for (index, state) in slot_states.iter().enumerate() {
                state
                    .lock()
                    .map_err(|e| {
                        Error::new(
                            Status::GenericFailure,
                            format!("kv sequence lock poisoned: {e}"),
                        )
                    })?
                    .advance = advances[lanes[index]] as usize;
            }
            let output = {
                let commit = || cancellation.complete();
                let result = match invocation {
                    StatefulInvocation::Tensors => executable::execute_stateful(
                        &executable,
                        &inputs,
                        &generated,
                        cancelled,
                        kv.as_ref(),
                        &commit,
                        false,
                    )
                    .map(|outputs| {
                        StatefulExecutionOutput::Tensors(
                            outputs
                                .into_iter()
                                .map(|output| NativeTensor::wrap(output))
                                .collect(),
                        )
                    }),
                    StatefulInvocation::Sampled(sampling) => executable::execute_stateful_sampled(
                        &executable,
                        &inputs,
                        &generated,
                        cancelled,
                        kv.as_ref(),
                        &commit,
                        &sampling,
                    )
                    .map(StatefulExecutionOutput::Samples),
                };
                match result {
                    Ok(output) => output,
                    Err(error) => {
                        rollback()?;
                        return Err(to_napi_err(error));
                    }
                }
            };
            Ok(output)
        })
        .await
    }
}

struct ProgramInner {
    executable: Arc<executable::MetalExecutable>,
    slots: Vec<ProgramSlot>,
    generated_bindings: Vec<value::Value>,
}

#[derive(Clone)]
struct GeneratedBindingSignature {
    shape: Vec<usize>,
    dtype: DType,
}

#[derive(Clone)]
struct CachedProgram {
    executable: Arc<executable::MetalExecutable>,
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
    semantic: &[value::Value],
    order: &[usize],
) -> Option<Vec<value::Value>> {
    if semantic.len() != program.index.leaves.len() || order.len() != semantic.len() {
        return None;
    }
    order
        .iter()
        .map(|position| semantic.get(*position).cloned())
        .collect()
}

fn generated_signatures(values: &[value::Value]) -> Vec<GeneratedBindingSignature> {
    values
        .iter()
        .map(|value| GeneratedBindingSignature {
            shape: value.shape().to_vec(),
            dtype: value.dtype(),
        })
        .collect()
}

fn generated_match(values: &[value::Value], expected: &[GeneratedBindingSignature]) -> bool {
    values.len() == expected.len()
        && values.iter().zip(expected).all(|(value, expected)| {
            value.shape() == expected.shape
                && value.dtype() == expected.dtype
                && value.device().is_metal()
                && value.as_metal().is_ok_and(|tensor| {
                    tensor.layout.is_contiguous() && tensor.layout.offset() == 0
                })
        })
}

impl Executable {
    async fn execute_stateless(
        &self,
        inputs: Vec<&NativeTensor>,
        scalars: Vec<f64>,
        token: Option<&CancellationToken>,
    ) -> Result<Vec<NativeTensor>> {
        let inner = &self.inner;
        if inputs
            .iter()
            .any(|input| input.device_ordinal != self.device_ordinal)
        {
            return Err(Error::new(
                Status::InvalidArg,
                "execute: inputs and executable must use the same Metal device",
            ));
        }
        let signature = &inner.executable.signature;
        signature
            .validate_invocation_counts(inputs.len(), scalars.len(), 0, None)
            .map_err(|error| Error::new(Status::InvalidArg, error.to_string()))?;
        let mut tensors = inputs.iter();
        // Slots are indexed by declaration order; tensor and scalar
        // arguments arrive as separate vectors in slot order.
        for (slot, declared) in inner.slots.iter().enumerate() {
            if declared.scalar {
                continue;
            }
            let input = tensors.next().expect("tensor count checked");
            let got = input.val_cloned()?;
            let device = got.device();
            if got.shape() != declared.shape.as_slice()
                || got.dtype() != declared.dtype
                || !device.same_device(&declared.device)
            {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!(
                        "input slot {slot}: expected {}, got {}:{}@{}",
                        declared.signature(),
                        got.shape()
                            .iter()
                            .map(|d| d.to_string())
                            .collect::<Vec<_>>()
                            .join("x"),
                        got.dtype().name(),
                        device
                    ),
                ));
            }
        }
        let executable = inner.executable.clone();
        let generated = inner.generated_bindings.clone();
        let inputs: Vec<value::Value> = inputs
            .iter()
            .map(|input| input.val_cloned())
            .collect::<Result<Vec<_>>>()?;
        for (index, value) in inputs.iter().enumerate() {
            let tensor = value.as_metal().map_err(to_napi_err)?;
            signature
                .validate_binding_metadata(index, value.dtype(), tensor.placement(), &tensor.layout)
                .map_err(|error| Error::new(Status::InvalidArg, error.to_string()))?;
        }
        let device_ordinal = self.device_ordinal;
        run_compute_on(device_ordinal, token, move |cancelled, _state| {
            Ok(executable::execute_with_scalars(
                &executable,
                &inputs,
                &generated,
                &scalars,
                cancelled,
            )
            .map_err(to_napi_err)?
            .into_iter()
            .map(|output| NativeTensor::wrap(output))
            .collect())
        })
        .await
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

#[napi]
pub fn compile(
    roots: Vec<&LazyTensor>,
    options: Option<NativeCompileOptions>,
    state: Option<NativeKvStateSchema>,
    cache_key: Option<String>,
) -> Result<Executable> {
    compile_for_ordinal(roots, options, state, cache_key, 0)
}

#[napi]
pub fn compile_for_device(
    roots: Vec<&LazyTensor>,
    options: Option<NativeCompileOptions>,
    state: Option<NativeKvStateSchema>,
    cache_key: Option<String>,
    device_ordinal: u32,
) -> Result<Executable> {
    compile_for_ordinal(roots, options, state, cache_key, device_ordinal as usize)
}

fn compile_for_ordinal(
    roots: Vec<&LazyTensor>,
    options: Option<NativeCompileOptions>,
    state: Option<NativeKvStateSchema>,
    cache_key: Option<String>,
    device_ordinal: usize,
) -> Result<Executable> {
    with_device(device_ordinal, || {
        compile_inner(roots, options, state, cache_key, device_ordinal)
    })
}

fn compile_inner(
    roots: Vec<&LazyTensor>,
    options: Option<NativeCompileOptions>,
    state: Option<NativeKvStateSchema>,
    cache_key: Option<String>,
    device_ordinal: usize,
) -> Result<Executable> {
    let mut nodes: Vec<Arc<Node>> = roots.iter().map(|tensor| tensor.node.clone()).collect();
    if nodes.is_empty() {
        return Err(Error::new(
            Status::InvalidArg,
            "compile: expected at least one root".to_string(),
        ));
    }
    let device_ordinal = u32::try_from(device_ordinal)
        .map_err(|_| Error::new(Status::InvalidArg, "Metal device ordinal exceeds u32::MAX"))?;
    let expected_device = Device::Metal(device_ordinal);
    if nodes
        .iter()
        .any(|node| !node.device.same_device(&expected_device))
    {
        return Err(Error::new(
            Status::InvalidArg,
            format!("compile: every root must use Metal device {device_ordinal}"),
        ));
    }
    let device_ordinal = device_ordinal as usize;

    let options = compile_options(options, state.is_some());
    let mut executable_state = None;
    let state_schema = if let Some(state) = state {
        if state.batch == 0 || state.max_tokens == 0 || state.block_size == 0 {
            return Err(Error::new(
                Status::InvalidArg,
                "compile: state batch, max_tokens, and block_size must be positive".to_string(),
            ));
        }
        let requested_window = state.window.map(|value| value as usize);
        if requested_window.is_some_and(|window| window == 0 || window > state.max_tokens as usize)
        {
            return Err(Error::new(
                Status::InvalidArg,
                "compile: KV window must be in 1..=max_tokens".to_string(),
            ));
        }
        let packed_rows_per_sequence = state
            .packed_causal_chains
            .as_ref()
            .map(|layout| layout.rows_per_sequence as usize);
        if packed_rows_per_sequence == Some(0) {
            return Err(Error::new(
                Status::InvalidArg,
                "compile: packed causal-chain rows must be positive and use all-row outputs",
            ));
        }
        let layout = packed_rows_per_sequence.map_or_else(
            || DecodeLayout::dense(state.batch as usize),
            |rows| DecodeLayout::packed_causal_chains(state.batch as usize, rows),
        );
        let graph_batch = layout
            .graph_rows()
            .map_err(|error| Error::new(Status::InvalidArg, error))?;
        if state.last_token_row.is_some() && state.output_selections.is_some() {
            return Err(Error::new(
                Status::InvalidArg,
                "compile: last_token_row and output_selections are mutually exclusive",
            ));
        }
        let output_selections = state.output_selections.clone().map_or_else(
            || {
                vec![
                    if state.last_token_row.unwrap_or(false) {
                        DecodeOutputSelection::SplitLastTokenRow
                    } else {
                        DecodeOutputSelection::AllRows
                    };
                    nodes.len()
                ]
            },
            |selections| selections.into_iter().map(Into::into).collect(),
        );
        let (rewritten, geometry) = specialize_decode_layout_outputs_with_attention(
            &nodes,
            requested_window,
            layout,
            &output_selections,
            state
                .current_block_attention
                .map(Into::into)
                .unwrap_or_default(),
        )
        .map_err(|error| Error::new(Status::GenericFailure, error))?;
        nodes = rewritten;
        let window = if geometry.allows_window_eviction {
            requested_window
        } else {
            None
        };
        let kda = geometry.kda.into();
        let conv = geometry.conv.into();
        let schema = KvStateSchema {
            max_tokens: state.max_tokens as usize,
            block_size: state.block_size as usize,
            kv_dtype: state.kv_dtype.into(),
            window,
            batch: state.batch as usize,
            graph_batch,
            layers: geometry.layers,
            kv_heads: geometry.kv_heads,
            head_dim: geometry.head_dim,
            kda,
            conv,
            cursor_slot: geometry.cursor_slot,
            cursor_tensor: geometry.cursor_tensor,
        };
        executable_state = Some(StatefulExecutable {
            cursor_slot: geometry.cursor_slot,
            cursor_tensor: geometry.cursor_tensor,
            allows_window_eviction: geometry.allows_window_eviction,
            schema,
            packed_rows_per_sequence,
        });
        Some(schema)
    } else {
        None
    };
    let mut request = ProgramRequest::from_roots(nodes, options);
    if let Some(state) = &executable_state {
        request =
            request.with_state_cursor(StateCursorSlot::new(state.cursor_slot, state.cursor_tensor));
    }
    let program = request.prepare().map_err(to_napi_err)?;
    let semantic_generated =
        executable::load_generated_bindings(&program.index).map_err(to_napi_err)?;
    let effective_cache_key = cache_key
        .filter(|_| {
            std::env::var_os("EFFECT_TORCH_NO_EXECUTABLE_CACHE").is_none()
                && !program
                    .options
                    .inference
                    .as_ref()
                    .is_some_and(|inference| inference.constant_weights)
        })
        .map(|key| format!("metal:{device_ordinal}|{key}|{:?}", program.options));
    if let Some(key) = effective_cache_key.as_deref() {
        let cached = program_cache()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(key);
        if let Some(cached) = cached {
            if cached.executable.signature == program.signature {
                if let Some(current) = ordered_generated_bindings(
                    &program,
                    &semantic_generated,
                    &cached.generated_order,
                )
                .filter(|current| generated_match(current, &cached.generated))
                {
                    return Ok(Executable {
                        inner: ProgramInner {
                            executable: Arc::clone(&cached.executable),
                            slots: cached.slots,
                            generated_bindings: current,
                        },
                        state: executable_state,
                        device_ordinal,
                    });
                }
            }
        }
    }
    let compilation =
        compile_prepared_metal(&program, &semantic_generated, state_schema).map_err(to_napi_err)?;
    let slots = compilation.slots;
    if let Some(key) = effective_cache_key {
        let generated_order = &compilation.generated_order;
        let generated = generated_signatures(&compilation.generated_bindings);
        if ordered_generated_bindings(&program, &semantic_generated, generated_order)
            .is_some_and(|values| generated_match(&values, &generated))
        {
            program_cache()
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .insert(
                    key,
                    CachedProgram {
                        executable: Arc::clone(&compilation.executable),
                        slots: slots.clone(),
                        generated,
                        generated_order: generated_order.clone(),
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
        state: executable_state,
        device_ordinal,
    })
}

// Saves already materialized tensors without introducing a graph execution path.
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
            "save_tensors: expected at least one tensor".to_string(),
        ));
    }
    let unique = names.iter().collect::<HashSet<_>>();
    if unique.len() != names.len() || names.iter().any(|name| name == "__metadata__") {
        return Err(Error::new(
            Status::InvalidArg,
            "save_tensors: tensor names must be unique and cannot be __metadata__".to_string(),
        ));
    }
    let device_ordinal = tensors[0].device_ordinal;
    if tensors
        .iter()
        .any(|tensor| tensor.device_ordinal != device_ordinal)
    {
        return Err(Error::new(
            Status::InvalidArg,
            "save_tensors: all tensors must use the same Metal device",
        ));
    }
    let tensors = tensors
        .iter()
        .map(|tensor| tensor.val_cloned())
        .collect::<Result<Vec<_>>>()?;
    run_compute_on(device_ordinal, token, move |cancelled, _state| {
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::new(
                Status::Cancelled,
                "operation aborted".to_string(),
            ));
        }
        let mut map = std::collections::HashMap::with_capacity(names.len());
        for (name, tensor) in names.iter().zip(tensors) {
            map.insert(name.clone(), tensor);
        }
        safetensors::save(&map, &metadata, &path).map_err(to_napi_err)?;
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::new(
                Status::Cancelled,
                "operation aborted".to_string(),
            ));
        }
        Ok(())
    })
    .await
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

// Loads a safetensors file straight into native tensors on the given device;
// JS only receives opaque handles and names. Entries are sorted by name so
// the result is deterministic.
#[napi]
pub async fn load_tensors(
    path: String,
    token: Option<&CancellationToken>,
) -> Result<NativeSafetensorsArchive> {
    load_tensors_on(path, token, 0).await
}

#[napi]
pub async fn load_tensors_for_device(
    path: String,
    device_ordinal: u32,
    token: Option<&CancellationToken>,
) -> Result<NativeSafetensorsArchive> {
    load_tensors_on(path, token, device_ordinal as usize).await
}

async fn load_tensors_on(
    path: String,
    token: Option<&CancellationToken>,
    device_ordinal: usize,
) -> Result<NativeSafetensorsArchive> {
    run_compute_on(device_ordinal, token, move |cancelled, _state| {
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::new(
                Status::Cancelled,
                "operation aborted".to_string(),
            ));
        }
        let archive = safetensors::load(&path).map_err(to_napi_err)?;
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::new(
                Status::Cancelled,
                "operation aborted".to_string(),
            ));
        }
        Ok(NativeSafetensorsArchive {
            entries: archive
                .entries
                .into_iter()
                .map(|(name, tensor)| NativeSafetensorsEntry {
                    name,
                    tensor: NativeTensor::wrap(tensor),
                })
                .collect(),
            metadata: archive.metadata,
        })
    })
    .await
}

// Native bytes currently retained by JS-reachable tensors. Exposed so tests
// can verify the accounting returns to baseline: Node's
// `process.memoryUsage().external` does not reflect
// `napi_adjust_external_memory`.
#[napi]
pub fn external_memory_bytes() -> i64 {
    EXTERNAL_MEMORY_BYTES.load(Ordering::Relaxed)
}

#[napi]
pub fn external_memory_bytes_for_device(device_ordinal: u32) -> i64 {
    EXTERNAL_MEMORY_BY_DEVICE
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&(device_ordinal as usize))
        .copied()
        .unwrap_or(0)
}

#[cfg(test)]
mod epilogue_tests {
    use super::*;
    use runtime::metal::device::MetalDevice;
    use runtime::metal::run::MetalTensor;

    fn route(kind: &str) -> NativeValueRef {
        NativeValueRef {
            kind: kind.to_string(),
            name: None,
            binding: None,
            stage: None,
            output: None,
            row: None,
            select_row: None,
        }
    }

    fn routing_plan() -> NativeProposerPlan {
        NativeProposerPlan {
            target_prefill_taps: vec![],
            target_decode_taps: vec![NativeTargetHiddenTap {
                name: "layers.7.hidden".to_string(),
                output: 1,
                shape: vec![2, 3],
                dtype: NativeDType::F32,
            }],
            target_verify_taps: vec![],
            shared_target_bindings: vec![],
            stages: vec![NativeProposerStage {
                executable: Some(0),
                operation_id: "ParallelBlock".to_string(),
                layout_id: Some("block-v1".to_string()),
                history_lookup: None,
                inputs: vec![],
                outputs: vec![NativeProposerValueSchema {
                    shape: vec![3],
                    dtype: NativeDType::F32,
                }],
            }],
            state: NativeProposerStatePlan {
                kind: "None".to_string(),
                schema_id: None,
            },
            commit: None,
            output: NativeProposerOutputPlan {
                topology: "Chains".to_string(),
                probabilities: "Deterministic".to_string(),
                token_ids: route("PendingTokens"),
                probability_rows: None,
                parents: None,
                confidence: None,
            },
            token_map: NativeTokenMapPlan {
                kind: "Identity".to_string(),
                fingerprint: "identity".to_string(),
                proposer_vocabulary: None,
                target_ids: None,
            },
            trained_max_rows: 2,
        }
    }

    #[test]
    fn proposer_routes_preserve_metadata_and_reject_forward_references() {
        let plan = routing_plan();
        let mut hidden = route("TargetHidden");
        hidden.name = Some("layers.7.hidden".to_string());
        hidden.select_row = Some(true);
        let metadata = validate_plan_ref(&hidden, 0, &plan).unwrap().unwrap();
        assert_eq!(metadata.shape, vec![3]);
        assert_eq!(metadata.dtype, NativeDType::F32);

        let mut prior = route("StageOutput");
        prior.stage = Some(0);
        prior.output = Some(0);
        assert!(validate_plan_ref(&prior, 1, &plan).is_ok());
        assert!(validate_plan_ref(&prior, 0, &plan).is_err());
    }

    #[test]
    fn only_exact_native_history_lookup_plan_is_recognized() {
        let mut plan = routing_plan();
        plan.target_decode_taps.clear();
        plan.stages[0] = NativeProposerStage {
            executable: None,
            operation_id: "HistoryLookup".to_string(),
            layout_id: Some("suffix-ngram-v1".to_string()),
            history_lookup: Some(NativeHistoryLookupLayout {
                id: "suffix-ngram-v1".to_string(),
                min_match_tokens: 2,
                max_match_tokens: 8,
            }),
            inputs: vec![],
            outputs: vec![NativeProposerValueSchema {
                shape: vec![8],
                dtype: NativeDType::U32,
            }],
        };
        plan.output.token_ids = NativeValueRef {
            kind: "StageOutput".to_string(),
            name: None,
            binding: None,
            stage: Some(0),
            output: Some(0),
            row: None,
            select_row: None,
        };
        assert_eq!(
            history_lookup_config(&plan, &[], &[]).unwrap(),
            Some(HistoryLookupConfig {
                min_match_tokens: 2,
                max_match_tokens: 8,
            })
        );

        plan.output.topology = "Trees".to_string();
        assert!(history_lookup_config(&plan, &[], &[]).is_err());
    }

    #[test]
    fn stateless_invocation_rejects_state() {
        assert!(
            validate_execution_mode(false, 0, Some(1), None, None, None, None, Some(1)).is_err()
        );
    }

    #[test]
    fn stateful_invocation_rejects_scalars_and_mismatched_state() {
        assert!(
            validate_execution_mode(true, 1, Some(1), None, None, None, None, Some(1)).is_err()
        );
        assert!(
            validate_execution_mode(true, 0, Some(2), None, None, None, None, Some(1)).is_err()
        );
        assert!(validate_execution_mode(true, 0, Some(1), None, None, None, None, None).is_err());
    }

    #[test]
    fn fixed_lane_validation_accepts_ragged_request_order() {
        let lanes = validate_fixed_lanes(
            4,
            2,
            &[3, 1],
            &[false, true, false, true],
            &[0, 2, 0, 1],
            &[0, 2, 0, 1],
            &[vec![7], vec![8, 9]],
        )
        .unwrap();
        assert_eq!(lanes, [3, 1]);
        assert!(validate_fixed_lanes(
            4,
            2,
            &[3, 1],
            &[false, true, false, true],
            &[0, 1, 0, 1],
            &[0, 2, 0, 1],
            &[vec![7], vec![8, 9]],
        )
        .is_err());
    }

    #[test]
    fn sampled_invocation_requires_state_and_one_option_per_active_output() {
        assert!(validate_sampled_execution_mode(false, 1, 1, 1).is_err());
        assert!(validate_sampled_execution_mode(true, 2, 1, 2).is_err());
        assert!(validate_sampled_execution_mode(true, 2, 2, 1).is_err());
        assert!(validate_sampled_execution_mode(true, 2, 2, 3).is_err());
        validate_sampled_execution_mode(true, 2, 2, 2).unwrap();
    }

    #[test]
    fn native_sampling_options_require_safe_integer_controls() {
        let options = |top_k, seed, counter| NativeSamplingOptions {
            temperature: 0.0,
            top_k,
            top_p: 1.0,
            seed,
            counter,
        };
        assert!(sampling_options(options(-1.0, 0.0, 0.0)).is_err());
        assert!(sampling_options(options(1.5, 0.0, 0.0)).is_err());
        assert!(sampling_options(options(0.0, f64::INFINITY, 0.0)).is_err());
        assert!(sampling_options(options(0.0, 0.0, 9_007_199_254_740_992.0)).is_err());
        assert_eq!(
            sampling_options(options(0.0, 7.0, 3.0)).unwrap(),
            greedy_sampling_options()
        );
    }

    #[test]
    fn speculative_rng_domains_are_candidate_keyed_and_slot_free() {
        let base = 91;
        let first = purpose_counter(base, SamplingPurpose::Proposal, 0);
        assert_eq!(first, purpose_counter(base, SamplingPurpose::Proposal, 0));
        assert_ne!(
            first,
            purpose_counter(base + 1, SamplingPurpose::Proposal, 0)
        );
        assert_ne!(first, purpose_counter(base, SamplingPurpose::Accept, 0));
        assert_ne!(
            purpose_counter(base + 3, SamplingPurpose::Residual, 0),
            purpose_counter(base + 3, SamplingPurpose::Target, 0)
        );
    }

    #[test]
    fn adaptive_verification_starts_at_eight_and_widens_for_acceptance() {
        assert_eq!(select_verify_rows(0, 0, 16), 8);
        assert_eq!(select_verify_rows(0, 0, 4), 4);
        assert_eq!(select_verify_rows(100, 95, 16), 16);
        assert!(select_verify_rows(100, 50, 16) < 8);
    }

    #[test]
    fn packed_verification_metadata_handles_ragged_rows_positions_and_padding() {
        let plan = packed_verification_plan(8, 4, &[11, 29], &[vec![7, 8, 9], vec![10]]).unwrap();
        assert_eq!(plan.row_offsets, [0, 3, 4]);
        assert_eq!(plan.logical_rows, [3, 1]);
        assert_eq!(
            plan.row_to_request,
            [Some(0), Some(0), Some(0), Some(1), None, None, None, None]
        );
        assert_eq!(plan.positions, [11, 12, 13, 29, 0, 0, 0, 0]);
        assert_eq!(
            plan.tokens,
            [
                vec![7],
                vec![8],
                vec![9],
                vec![10],
                vec![0],
                vec![0],
                vec![0],
                vec![0]
            ]
        );
    }

    #[test]
    fn packed_verification_metadata_rejects_malformed_layouts() {
        assert!(packed_verification_plan(4, 2, &[0, 0], &[vec![1, 2], vec![3, 4, 5]]).is_err());
        assert!(packed_verification_plan(3, 2, &[0, 0], &[vec![1, 2], vec![3, 4]]).is_err());
        assert!(packed_verification_plan(4, 2, &[0], &[vec![1], vec![2]]).is_err());
    }

    #[test]
    fn history_lookup_prefers_longest_suffix_then_most_recent_continuation() {
        assert_eq!(
            history_lookup(&[1, 2, 3, 7, 1, 2, 3, 8, 1, 2, 3], 2, 3, 4),
            [8, 1, 2, 3]
        );
        assert_eq!(
            history_lookup(&[4, 5, 6, 4, 5, 7, 4, 5], 2, 2, 3),
            [7, 4, 5]
        );
        assert!(history_lookup(&[1, 2, 3], 2, 4, 3).is_empty());
    }

    #[test]
    fn history_lookup_applies_independent_ragged_lane_caps() {
        let histories = [vec![1, 2, 9, 1, 2], vec![3, 4, 5, 3, 4]];
        let limits = [1, 3];
        let candidates = histories
            .iter()
            .zip(limits)
            .map(|(history, limit)| history_lookup(history, 2, 2, limit))
            .collect::<Vec<_>>();
        assert_eq!(candidates, [vec![9], vec![5, 3, 4]]);
    }

    #[test]
    fn history_lookup_target_shadow_is_rollback_safe_and_publishes_only_cut_tokens() {
        let pool = NativeKvPool::new(0, 0, 0, 16, Some(4), Some(NativeDType::F32), None).unwrap();
        let sequence = seed_sequence(&pool, &[1, 2, 3]);
        let states = vec![sequence.state.clone()];
        let visible = vec![1, 2, 3, 10];
        let before = canonical_snapshot(&sequence.state);
        {
            let shadow = ShadowSequences::new(pool.inner.clone(), &states).unwrap();
            stage_speculative_rows(&shadow, &[10, 20, 21, 22]);
        }
        assert_eq!(canonical_snapshot(&sequence.state), before);
        assert_eq!(visible, [1, 2, 3, 10]);

        let mut shadow = ShadowSequences::new(pool.inner.clone(), &states).unwrap();
        stage_speculative_rows(&shadow, &[10, 20, 21, 22]);
        let mut page = vec![20, 21, 22, 99];
        cut_speculative_page(&mut page, &[21], 3);
        publish_speculative_states(&mut shadow, &states, &[10], std::slice::from_ref(&page));
        let mut committed_history = visible;
        committed_history.extend_from_slice(&page);
        assert_eq!(page, [20, 21]);
        assert_eq!(committed_history, [1, 2, 3, 10, 20, 21]);
        assert_eq!(sequence.state.lock().unwrap().cursor, 5);
    }

    #[test]
    fn speculative_effective_distributions_cover_greedy_and_residual_sampling() {
        let mut options = greedy_sampling_options();
        let greedy =
            effective_probabilities(3, |index| [2.0, 2.0, 1.0][index], options, || false).unwrap();
        assert_eq!(greedy, [1.0, 0.0, 0.0]);

        options.temperature = 1.0;
        options.top_k = None;
        options.top_p = 1.0;
        let target = effective_probabilities(
            2,
            |index| [0.7_f64.ln(), 0.3_f64.ln()][index],
            options,
            || false,
        )
        .unwrap();
        let proposal = effective_probabilities(
            2,
            |index| [0.2_f64.ln(), 0.8_f64.ln()][index],
            options,
            || false,
        )
        .unwrap();
        let residual = target
            .iter()
            .zip(proposal)
            .map(|(target, proposal)| (target - proposal).max(0.0))
            .collect::<Vec<_>>();
        assert_eq!(sample_probabilities(&residual, 7, 11, || false).unwrap(), 0);
    }

    #[test]
    fn speculative_shadow_failure_releases_only_new_blocks() {
        let pool = NativeKvPool::new(0, 0, 0, 16, Some(4), Some(NativeDType::F32), None).unwrap();
        let sequence = pool.make_sequence().unwrap();
        let before = pool.free_blocks();
        {
            let shadow =
                ShadowSequences::new(pool.inner.clone(), &[sequence.state.clone()]).unwrap();
            shadow.states[0]
                .lock()
                .unwrap()
                .blocks
                .push(pool.inner.alloc_block().unwrap());
            assert_eq!(pool.free_blocks(), before - 1);
        }
        assert_eq!(pool.free_blocks(), before);
        assert!(sequence.state.lock().unwrap().blocks.is_empty());
    }

    #[test]
    fn shadow_window_eviction_does_not_release_canonical_ownership() {
        let pool = NativeKvPool::new(0, 0, 0, 16, Some(4), Some(NativeDType::F32), None).unwrap();
        let sequence = pool.make_sequence().unwrap();
        let first = pool.inner.alloc_block().unwrap();
        let second = pool.inner.alloc_block().unwrap();
        {
            let mut canonical = sequence.state.lock().unwrap();
            canonical.blocks = vec![first, second];
            canonical.cursor = 8;
        }
        let before = pool.free_blocks();
        {
            let shadow =
                ShadowSequences::new(pool.inner.clone(), &[sequence.state.clone()]).unwrap();
            {
                let mut provisional = shadow.states[0].lock().unwrap();
                kv_evict(&pool.inner, &mut provisional, 4);
                assert_eq!(provisional.blocks, [second]);
                assert_eq!(provisional.head, 1);
            }
            let canonical = sequence.state.lock().unwrap();
            assert_eq!(canonical.blocks, [first, second]);
            assert_eq!(canonical.head, 0);
            assert_eq!(pool.free_blocks(), before);
        }
        assert_eq!(sequence.state.lock().unwrap().blocks, [first, second]);
        assert_eq!(pool.free_blocks(), before);
    }

    #[test]
    fn inference_fingerprints_bind_operation_order_and_effective_policy() {
        let greedy = InferenceSampling {
            temperature: 0.0,
            top_k: None,
            top_p: 1.0,
            seed: 7,
        };
        let changed = InferenceSampling { seed: 8, ..greedy };
        let add = add_fingerprint(&[vec![1, 2]], &[greedy], &[Some(3)], &[vec![9]]);
        assert_eq!(
            add,
            add_fingerprint(&[vec![1, 2]], &[greedy], &[Some(3)], &[vec![9]])
        );
        assert_ne!(
            add,
            add_fingerprint(&[vec![1, 3]], &[greedy], &[Some(3)], &[vec![9]])
        );
        assert_ne!(
            add,
            add_fingerprint(&[vec![1, 2]], &[changed], &[Some(3)], &[vec![9]])
        );
        assert_ne!(
            round_fingerprint(&[1, 2], &[greedy, changed]),
            round_fingerprint(&[2, 1], &[changed, greedy])
        );
        assert_ne!(add, round_fingerprint(&[1], &[greedy]));
    }

    #[test]
    fn exhausted_round_counter_does_not_wrap() {
        let counter = AtomicU64::new(u64::MAX);
        assert!(next_monotonic(&counter, "round IDs").is_err());
        assert_eq!(counter.load(Ordering::SeqCst), u64::MAX);
    }

    #[test]
    fn speculative_publication_keeps_final_token_pending_and_hashes_only_consumed_rows() {
        let pool = NativeKvPool::new(0, 0, 0, 16, Some(4), Some(NativeDType::F32), None).unwrap();
        let sequence = pool.make_sequence().unwrap();
        let states = vec![sequence.state.clone()];

        let mut first = ShadowSequences::new(pool.inner.clone(), &states).unwrap();
        first.states[0]
            .lock()
            .unwrap()
            .blocks
            .push(pool.inner.alloc_block().unwrap());
        publish_speculative_states(&mut first, &states, &[9], &[vec![10, 11, 12]]);
        {
            let state = sequence.state.lock().unwrap();
            assert_eq!(state.cursor, 3);
            assert_eq!(state.pending, [9, 10, 11]);
            assert_eq!(state.last_hash, HASH_SEED);
        }
        assert!(pool.inner.blocks.lock().unwrap().by_hash.is_empty());

        let mut second = ShadowSequences::new(pool.inner.clone(), &states).unwrap();
        publish_speculative_states(&mut second, &states, &[12], &[vec![13]]);
        let state = sequence.state.lock().unwrap();
        assert_eq!(state.cursor, 4);
        assert!(state.pending.is_empty());
        let expected = chain_hash(HASH_SEED, &[9, 10, 11, 12]);
        assert_eq!(state.last_hash, expected);
        let block = state.blocks[0];
        assert_eq!(
            pool.inner.blocks.lock().unwrap().hashes[block as usize],
            Some(expected)
        );
    }

    #[test]
    fn provisional_token_metadata_never_publishes_prefix_hashes() {
        let pool = NativeKvPool::new(0, 0, 0, 16, Some(4), Some(NativeDType::F32), None).unwrap();
        let sequence = pool.make_sequence().unwrap();
        let block = pool.inner.alloc_block().unwrap();
        let mut state = sequence.state.lock().unwrap();
        state.blocks.push(block);
        state.note_tokens_provisional(&[1, 2, 3, 4], pool.inner.block_size);
        state.cursor = 4;
        assert_ne!(state.last_hash, HASH_SEED);
        assert!(pool.inner.blocks.lock().unwrap().by_hash.is_empty());
    }

    fn seed_sequence(pool: &NativeKvPool, tokens: &[u32]) -> NativeKvSequence {
        let sequence = pool.make_sequence().unwrap();
        let mut state = sequence.state.lock().unwrap();
        while state.blocks.len() < tokens.len().div_ceil(pool.inner.block_size) {
            state.blocks.push(pool.inner.alloc_block().unwrap());
        }
        state.note_tokens(&pool.inner, tokens);
        state.cursor = tokens.len();
        drop(state);
        sequence
    }

    fn stage_speculative_rows(shadow: &ShadowSequences, tokens: &[u32]) {
        let mut state = shadow.states[0].lock().unwrap();
        let desired = (state.cursor + tokens.len())
            .div_ceil(shadow.pool.block_size)
            .saturating_sub(state.head);
        while state.blocks.len() < desired {
            state.blocks.push(shadow.pool.alloc_block().unwrap());
        }
        state.note_tokens_provisional(tokens, shadow.pool.block_size);
        state.cursor += tokens.len();
    }

    fn assert_committed_store(pool: &NativeKvPool, state: &SeqState) {
        let store = pool.inner.blocks.lock().unwrap();
        for ids in store.by_hash.values() {
            assert!(ids.iter().all(|block| state.blocks.contains(block)));
        }
        for (block, count) in store.refcounts.iter().enumerate() {
            if state.blocks.contains(&(block as u32)) {
                assert_eq!(*count, 1, "canonical block {block} must have one owner");
            } else {
                assert_eq!(*count, 0, "provisional block {block} retained an owner");
                assert!(
                    !store.is_cached(block as u32),
                    "provisional block {block} became cache-visible"
                );
            }
        }
    }

    #[test]
    fn every_acceptance_cut_synchronizes_target_and_proposer_publication() {
        const CANDIDATES: [u32; 3] = [20, 21, 22];
        const ANCHOR: u32 = 10;

        for cut in 0..=CANDIDATES.len() {
            let target_pool =
                NativeKvPool::new(0, 0, 0, 16, Some(4), Some(NativeDType::F32), None).unwrap();
            let proposer_pool =
                NativeKvPool::new(0, 0, 0, 16, Some(4), Some(NativeDType::F32), None).unwrap();
            let target = seed_sequence(&target_pool, &[1, 2, 3]);
            let proposer = seed_sequence(&proposer_pool, &[1, 2, 3]);
            let target_states = vec![target.state.clone()];
            let proposer_states = vec![proposer.state.clone()];
            let mut target_shadow =
                ShadowSequences::new(target_pool.inner.clone(), &target_states).unwrap();
            let mut proposer_shadow =
                ShadowSequences::new(proposer_pool.inner.clone(), &proposer_states).unwrap();

            stage_speculative_rows(&target_shadow, &[ANCHOR, 20, 21, 22]);
            stage_speculative_rows(&proposer_shadow, &[ANCHOR, 20, 21]);
            let page = if cut == CANDIDATES.len() {
                stage_speculative_rows(&proposer_shadow, &[22]);
                vec![20, 21, 22, 99]
            } else {
                let mut page = CANDIDATES[..cut].to_vec();
                page.push(90 + cut as u32);
                page
            };

            publish_speculative_states(
                &mut target_shadow,
                &target_states,
                &[ANCHOR],
                std::slice::from_ref(&page),
            );
            publish_speculative_states(
                &mut proposer_shadow,
                &proposer_states,
                &[ANCHOR],
                std::slice::from_ref(&page),
            );

            let target_state = target.state.lock().unwrap();
            let proposer_state = proposer.state.lock().unwrap();
            let consumed = std::iter::once(ANCHOR)
                .chain(page.iter().copied())
                .take(page.len())
                .collect::<Vec<_>>();
            let all_consumed = [vec![1, 2, 3], consumed].concat();
            let complete = all_consumed.len() / target_pool.inner.block_size;
            let mut expected_hash = HASH_SEED;
            for block in all_consumed.chunks_exact(target_pool.inner.block_size) {
                expected_hash = chain_hash(expected_hash, block);
            }

            assert_eq!(target_state.cursor, 3 + page.len(), "acceptance cut {cut}");
            assert_eq!(
                proposer_state.cursor, target_state.cursor,
                "acceptance cut {cut}"
            );
            assert_eq!(
                target_state.last_hash, expected_hash,
                "acceptance cut {cut}"
            );
            assert_eq!(
                proposer_state.last_hash, expected_hash,
                "acceptance cut {cut}"
            );
            assert_eq!(target_state.pending, all_consumed[complete * 4..]);
            assert_eq!(proposer_state.pending, target_state.pending);
            assert_eq!(target_state.blocks.len(), (3 + page.len()).div_ceil(4));
            assert_eq!(proposer_state.blocks.len(), target_state.blocks.len());
            assert_eq!(
                page.last(),
                Some(&(if cut == 3 { 99 } else { 90 + cut as u32 }))
            );
            assert_committed_store(&target_pool, &target_state);
            assert_committed_store(&proposer_pool, &proposer_state);
        }
    }

    #[test]
    fn eos_and_budget_cut_each_candidate_correction_and_bonus_boundary() {
        let full = vec![20, 21, 22, 99];
        for (index, eos) in full.iter().copied().enumerate() {
            let mut page = full.clone();
            cut_speculative_page(&mut page, &[eos], u32::MAX);
            assert_eq!(page, full[..=index], "EOS at full-chain index {index}");
        }
        for cut in 0..3 {
            let mut raw = [20, 21, 22][..cut].to_vec();
            let correction = 90 + cut as u32;
            raw.push(correction);
            let mut page = raw.clone();
            cut_speculative_page(&mut page, &[correction], u32::MAX);
            assert_eq!(page, raw, "EOS at rejection correction {cut}");
        }
        for budget in 1..=full.len() {
            let mut page = full.clone();
            cut_speculative_page(&mut page, &[], budget as u32);
            assert_eq!(page, full[..budget], "budget cut {budget}");
        }
        let mut eos_before_budget = full.clone();
        cut_speculative_page(&mut eos_before_budget, &[21], 4);
        assert_eq!(eos_before_budget, [20, 21]);
        let mut budget_before_eos = full;
        cut_speculative_page(&mut budget_before_eos, &[22], 2);
        assert_eq!(budget_before_eos, [20, 21]);
    }

    fn canonical_snapshot(state: &Arc<Mutex<SeqState>>) -> (Vec<u32>, usize, usize, u64, Vec<u32>) {
        let state = state.lock().unwrap();
        (
            state.blocks.clone(),
            state.head,
            state.cursor,
            state.last_hash,
            state.pending.clone(),
        )
    }

    fn assert_failed_speculation_is_invisible(phase: &str, cancel: bool) {
        let target_pool =
            NativeKvPool::new(0, 0, 0, 16, Some(4), Some(NativeDType::F32), None).unwrap();
        let proposer_pool =
            NativeKvPool::new(0, 0, 0, 16, Some(4), Some(NativeDType::F32), None).unwrap();
        let target = seed_sequence(&target_pool, &[1, 2, 3, 4, 5]);
        let proposer = seed_sequence(&proposer_pool, &[1, 2, 3, 4, 5]);
        let target_before = canonical_snapshot(&target.state);
        let proposer_before = canonical_snapshot(&proposer.state);
        let target_available = target_pool.inner.available();
        let proposer_available = proposer_pool.inner.available();
        let target_hashes = target_pool.inner.blocks.lock().unwrap().by_hash.clone();
        let proposer_hashes = proposer_pool.inner.blocks.lock().unwrap().by_hash.clone();
        {
            let target_shadow =
                ShadowSequences::new(target_pool.inner.clone(), &[target.state.clone()]).unwrap();
            let proposer_shadow =
                ShadowSequences::new(proposer_pool.inner.clone(), &[proposer.state.clone()])
                    .unwrap();
            if cancel {
                stage_speculative_rows(&proposer_shadow, &[10, 20, 21]);
                stage_speculative_rows(&target_shadow, &[10, 20, 21, 22]);
                let cancellation = CancellationState::new();
                assert!(cancellation.cancel());
                assert!(cancellation.flag().is_cancelled());
                assert!(!cancellation.complete());
                // The same gate used by add/runRound rejects publication, so
                // dropping both shadows is the complete cancellation path.
            } else {
                let failed = (|| -> err::Res<()> {
                    stage_speculative_rows(&proposer_shadow, &[10, 20, 21]);
                    if phase != "proposer" {
                        stage_speculative_rows(&target_shadow, &[10, 20, 21, 22]);
                    }
                    Err(format!("forced {phase} failure"))
                })();
                assert_eq!(failed.unwrap_err(), format!("forced {phase} failure"));
            }
        }
        assert_eq!(canonical_snapshot(&target.state), target_before, "{phase}");
        assert_eq!(
            canonical_snapshot(&proposer.state),
            proposer_before,
            "{phase}"
        );
        assert_eq!(target_pool.inner.available(), target_available, "{phase}");
        assert_eq!(
            proposer_pool.inner.available(),
            proposer_available,
            "{phase}"
        );
        assert_eq!(
            target_pool.inner.blocks.lock().unwrap().by_hash,
            target_hashes
        );
        assert_eq!(
            proposer_pool.inner.blocks.lock().unwrap().by_hash,
            proposer_hashes
        );
    }

    #[test]
    fn forced_phase_failures_leave_canonical_state_and_cache_unchanged() {
        for phase in ["proposer", "verifier", "sampler", "acceptance"] {
            assert_failed_speculation_is_invisible(phase, false);
        }
    }

    #[test]
    fn prepublication_cancellation_discards_both_provisional_roots() {
        assert_failed_speculation_is_invisible("publication", true);
    }

    #[test]
    fn cohesive_prefill_commit_matches_each_pool_at_its_own_cursor() {
        let target_pool =
            NativeKvPool::new(0, 0, 0, 16, Some(4), Some(NativeDType::F32), None).unwrap();
        let proposer_pool =
            NativeKvPool::new(0, 0, 0, 16, Some(4), Some(NativeDType::F32), None).unwrap();
        let prompt = (1..=9).collect::<Vec<_>>();

        let target = target_pool.make_sequence().unwrap();
        let target_states = vec![target.state.clone()];
        let mut target_shadow =
            ShadowSequences::new(target_pool.inner.clone(), &target_states).unwrap();
        stage_speculative_rows(&target_shadow, &prompt);
        let target_cache = stage_cache_metadata(&target_shadow, std::slice::from_ref(&prompt));

        let proposer = proposer_pool.make_sequence().unwrap();
        let proposer_states = vec![proposer.state.clone()];
        let mut proposer_shadow =
            ShadowSequences::new(proposer_pool.inner.clone(), &proposer_states).unwrap();
        stage_speculative_rows(&proposer_shadow, &prompt[..4]);
        let proposer_cache = stage_cache_metadata(&proposer_shadow, &[prompt[..4].to_vec()]);

        assert!(target_pool.inner.blocks.lock().unwrap().by_hash.is_empty());
        assert!(proposer_pool
            .inner
            .blocks
            .lock()
            .unwrap()
            .by_hash
            .is_empty());
        let boundary_hash = chain_hash(HASH_SEED, &prompt[..4]);
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let (target_lookup, proposer_lookup) = std::thread::scope(|scope| {
            let target_barrier = barrier.clone();
            let target_pool = target_pool.inner.clone();
            let target = scope.spawn(move || {
                target_barrier.wait();
                target_pool.take_block(boundary_hash)
            });
            let proposer_barrier = barrier.clone();
            let proposer_pool = proposer_pool.inner.clone();
            let proposer = scope.spawn(move || {
                proposer_barrier.wait();
                proposer_pool.take_block(boundary_hash)
            });
            barrier.wait();
            (target.join().unwrap(), proposer.join().unwrap())
        });
        assert_eq!((target_lookup, proposer_lookup), (None, None));
        publish_shadow(&mut target_shadow, &target_states);
        publish_shadow(&mut proposer_shadow, &proposer_states);
        publish_paired_cache_metadata(&target_cache, Some(&proposer_cache));

        let later_target = target_pool.make_sequence().unwrap();
        let later_proposer = proposer_pool.make_sequence().unwrap();
        assert_eq!(later_target.prefill_match(prompt.clone()).unwrap(), 8);
        assert_eq!(later_proposer.prefill_match(prompt.clone()).unwrap(), 4);
        assert_eq!(prompt.len() - later_target.cursor() as usize, 1);
        assert_eq!(prompt.len() - later_proposer.cursor() as usize, 5);
    }

    #[test]
    fn provisional_pool_exhaustion_does_not_evict_committed_cache() {
        let pool = NativeKvPool::new(0, 0, 0, 8, Some(4), Some(NativeDType::F32), None).unwrap();
        let cached = seed_sequence(&pool, &[1, 2, 3, 4]);
        cached.release();
        let before = pool.inner.blocks.lock().unwrap().by_hash.clone();
        let live = pool.inner.alloc_block().unwrap();

        assert!(pool.inner.alloc_block_with_cache_eviction(false).is_none());
        let store = pool.inner.blocks.lock().unwrap();
        assert_eq!(store.by_hash, before);
        assert_eq!(store.refcounts[live as usize], 1);
        assert_eq!(store.cached(), 1);
    }

    #[test]
    fn exact_receipt_recovery_requires_matching_request_and_explicit_ack() {
        let mut pending = Some(Receipt {
            round_id: 17,
            fingerprint: 23,
            pages: vec![ReceiptPage {
                sequence_id: 5,
                tokens: vec![7, 8],
                stop_reason: None,
            }],
        });

        assert!(recover_receipt(&pending, 24).is_err());
        assert_eq!(pending.as_ref().unwrap().round_id, 17);
        let recovered = recover_receipt(&pending, 23).unwrap().unwrap();
        assert_eq!(recovered.pages[0].tokens, [7, 8]);
        assert!(
            pending.is_some(),
            "recovery must retain the receipt until ack"
        );
        assert!(acknowledge_receipt(&mut pending, 18).is_err());
        assert_eq!(pending.as_ref().unwrap().round_id, 17);
        acknowledge_receipt(&mut pending, 17).unwrap();
        assert!(pending.is_none());
        assert!(acknowledge_receipt(&mut pending, 17).is_err());
    }

    fn mleaf(data: Vec<f32>, shape: Vec<usize>) -> Arc<Node> {
        let t = MetalTensor::from_f32(MetalDevice::get(), data, shape);
        Node::new(NodeKind::Leaf(std::sync::Arc::new(LeafSlot::new(
            value::Value(t),
        ))))
        .unwrap()
    }

    fn greedy_sampling_options() -> SamplingOptions {
        SamplingOptions {
            temperature: 0.0,
            top_k: None,
            top_p: 1.0,
            seed: 7,
            counter: 3,
        }
    }

    #[test]
    fn sampling_reads_strided_native_float_storage() {
        let values = [100.0_f32, 1.0, -100.0, 5.0, -100.0, 3.0];
        let cases = [
            (
                DType::F32,
                values
                    .iter()
                    .flat_map(|value| value.to_ne_bytes())
                    .collect::<Vec<_>>(),
            ),
            (
                DType::F16,
                values
                    .iter()
                    .flat_map(|value| half::f16::from_f32(*value).to_bits().to_ne_bytes())
                    .collect::<Vec<_>>(),
            ),
            (
                DType::BF16,
                values
                    .iter()
                    .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_ne_bytes())
                    .collect::<Vec<_>>(),
            ),
        ];

        for (dtype, bytes) in cases {
            let logits = value::Value(MetalTensor {
                buffer: MetalDevice::get().upload_bytes(&bytes),
                layout: runtime::layout::Layout::new(vec![3], vec![2], 1),
                dtype,
            });
            assert_eq!(
                sample_blocking(
                    &logits,
                    greedy_sampling_options(),
                    &effect_torch_runtime::CancellationFlag::new(),
                )
                .unwrap(),
                1,
                "{dtype}"
            );
        }
    }

    #[test]
    fn sampling_rejects_non_vector_empty_and_nonfloat_logits() {
        let device = MetalDevice::get();
        let cancelled = effect_torch_runtime::CancellationFlag::new();
        let matrix = value::Value(MetalTensor::from_f32(device, vec![1.0, 2.0], vec![1, 2]));
        let empty = value::Value(MetalTensor::empty(device, vec![0], DType::F32));
        let integers = value::Value(MetalTensor {
            buffer: device.alloc_with_data_u32(&[1, 2]),
            layout: runtime::layout::Layout::contiguous(vec![2]),
            dtype: DType::U32,
        });

        assert!(sample_blocking(&matrix, greedy_sampling_options(), &cancelled).is_err());
        assert!(sample_blocking(&empty, greedy_sampling_options(), &cancelled).is_err());
        assert!(sample_blocking(&integers, greedy_sampling_options(), &cancelled).is_err());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn sampled_execution_failure_rolls_back_sequence_setup() {
        let root = LazyTensor {
            node: mleaf(
                vec![
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    4.0,
                    f32::NAN,
                    6.0,
                    7.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                vec![1, 3, 4],
            ),
        };
        let executable = compile(
            vec![&root],
            None,
            Some(NativeKvStateSchema {
                max_tokens: 32,
                block_size: 4,
                kv_dtype: NativeDType::F32,
                window: None,
                batch: 1,
                packed_causal_chains: None,
                last_token_row: Some(true),
                output_selections: None,
                current_block_attention: None,
            }),
            None,
        )
        .unwrap();
        let pool = NativeKvPool::new(0, 0, 0, 32, Some(4), Some(NativeDType::F32), None).unwrap();
        let sequence = pool.make_sequence().unwrap();
        let free_blocks = pool.free_blocks();

        let error = executable
            .execute_sampled(
                Vec::new(),
                vec![&sequence],
                vec![0],
                vec![true],
                vec![2],
                vec![2],
                vec![vec![1, 2]],
                vec![NativeSamplingOptions {
                    temperature: 0.0,
                    top_k: 0.0,
                    top_p: 1.0,
                    seed: 7.0,
                    counter: 3.0,
                }],
                None,
            )
            .await
            .unwrap_err();

        assert!(error.to_string().contains("sample: logit 1 is not finite"));
        let state = sequence.state.lock().unwrap();
        assert_eq!(state.cursor, 0);
        assert_eq!(state.advance, 0);
        assert!(state.blocks.is_empty());
        assert_eq!(pool.free_blocks(), free_blocks);
    }

    #[test]
    fn per_root_decode_outputs_split_logits_and_batch_hidden_rows() {
        let logits = LazyTensor {
            node: mleaf(vec![0.0; 2 * 3 * 8], vec![2, 3, 8]),
        };
        let hidden = LazyTensor {
            node: mleaf(vec![0.0; 2 * 3 * 6], vec![2, 3, 6]),
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
    fn executable_diagnostics_exposes_compile_phases() {
        let root = mleaf(vec![1.0], vec![1]);
        let compilation = compile_metal_roots(std::slice::from_ref(&root)).unwrap();
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
                "pipeline_preparation",
                "artifact_assembly",
                "compile_submission",
                "publication",
            ]
        );
        assert!(diagnostics
            .compile_phases
            .iter()
            .all(|timing| timing.nanoseconds.is_finite() && timing.nanoseconds >= 0.0));
    }

    fn mleaf_u32(data: Vec<u32>, shape: Vec<usize>) -> Arc<Node> {
        let t = MetalTensor {
            buffer: MetalDevice::get().alloc_with_data_u32(&data),
            layout: runtime::layout::Layout::contiguous(shape),
            dtype: DType::U32,
        };
        Node::new(NodeKind::Leaf(std::sync::Arc::new(LeafSlot::new(
            value::Value(t),
        ))))
        .unwrap()
    }

    fn eval_f32(node: &Arc<Node>) -> Vec<f32> {
        let compilation = compile_metal_roots(std::slice::from_ref(node)).unwrap();
        executable::execute(
            &compilation.executable,
            &[],
            &compilation.generated_bindings,
            &effect_torch_runtime::CancellationFlag::new(),
            None,
        )
        .unwrap()[0]
            .to_f32_vec()
            .unwrap()
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let d = (x - y).abs() / y.abs().max(1.0);
            assert!(d <= tol, "{what}[{i}]: {x} vs {y}");
        }
    }

    fn linear(x: &Arc<Node>, weight: &Arc<Node>, bias: &Arc<Node>) -> Arc<Node> {
        Node::new(NodeKind::Linear {
            x: x.clone(),
            weight: weight.clone(),
            bias: bias.clone(),
        })
        .unwrap()
    }

    #[test]
    fn structural_cache_shares_the_plan_and_rebinds_leaves() {
        let graph = |left: Vec<f32>, right: Vec<f32>| LazyTensor {
            node: Node::new(NodeKind::Add {
                a: mleaf(left, vec![2]),
                b: mleaf(right, vec![2]),
            })
            .unwrap(),
        };
        let first_root = graph(vec![1.0, 2.0], vec![3.0, 4.0]);
        let second_root = graph(vec![10.0, 20.0], vec![30.0, 40.0]);
        let key = Some("metal-structural-cache-rebind-test".to_string());
        let first = compile(vec![&first_root], None, None, key.clone()).unwrap();
        let second = compile(vec![&second_root], None, None, key).unwrap();
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
                    &effect_torch_runtime::CancellationFlag::new(),
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
                    &effect_torch_runtime::CancellationFlag::new(),
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
            node: mleaf(values, vec![2]),
        };
        let first_root = graph(vec![1.0, 2.0]);
        let second_root = graph(vec![3.0, 4.0]);
        let constant_options = || NativeCompileOptions {
            optimize: None,
            constant_weights: Some(true),
        };
        let key = Some("metal-constant-weight-cache-suppression-test".to_string());
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
                &effect_torch_runtime::CancellationFlag::new(),
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
                &effect_torch_runtime::CancellationFlag::new(),
                None,
            )
            .unwrap()[0]
                .to_f32_vec()
                .unwrap(),
            [3.0, 4.0]
        );
    }

    #[test]
    fn zero_copy_readback_retains_pooled_output_after_tensor_clear() {
        let root = LazyTensor {
            node: Node::new(NodeKind::Randn {
                shape: vec![2048],
                dtype: DType::F32,
                device: Device::Metal(0),
            })
            .unwrap(),
        };
        let program = compile(vec![&root], None, None, None).unwrap();
        let run = || {
            executable::execute(
                &program.inner.executable,
                &[],
                &program.inner.generated_bindings,
                &effect_torch_runtime::CancellationFlag::new(),
                None,
            )
            .unwrap()
            .remove(0)
        };
        let mut tensor = NativeTensor::wrap(run());
        let inner = tensor.val_cloned().unwrap();
        let readback = readback_blocking(&inner).unwrap();
        drop(inner);
        assert!(matches!(
            readback.hint.as_ref(),
            Some(FinalizeHint::ZeroCopy { .. })
        ));
        // SAFETY: `readback` owns/retains `data` for `byte_len` bytes and the
        // f32-producing tensor guarantees alignment and a multiple-of-four
        // length for the duration of this copy.
        let expected = unsafe {
            std::slice::from_raw_parts(readback.data.cast::<f32>(), readback.byte_len / 4).to_vec()
        };

        assert!(tensor.slot.clear());
        tensor.release_accounting();
        for _ in 0..4 {
            drop(run());
        }
        // SAFETY: the zero-copy hint still retains the same Metal allocation
        // after the tensor handle was cleared; extent/alignment are unchanged.
        let retained = unsafe {
            std::slice::from_raw_parts(readback.data.cast::<f32>(), readback.byte_len / 4).to_vec()
        };
        assert_eq!(retained, expected);
    }

    #[test]
    fn fused_ce_invalid_label_reports_class_count() {
        let logits = mleaf(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], vec![2, 3]);
        let target = mleaf_u32(vec![0, 3], vec![2]);
        let ce = Node::new(NodeKind::CrossEntropy {
            logits,
            target,
            ignore_index: -1,
            reduction: CeReduction::Mean,
        })
        .unwrap();
        let compilation = compile_metal_roots(std::slice::from_ref(&ce)).unwrap();
        let error = match executable::execute(
            &compilation.executable,
            &[],
            &compilation.generated_bindings,
            &effect_torch_runtime::CancellationFlag::new(),
            None,
        ) {
            Ok(_) => panic!("invalid CE label unexpectedly succeeded"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            "cross_entropy: target out of range [0, 3) at an active position"
        );
    }

    #[test]
    fn layer_norm_side_outputs_are_explicit_executable_outputs() {
        let x = mleaf(
            (0..6).map(|index| index as f32 * 0.2 - 0.5).collect(),
            vec![2, 3],
        );
        let weight = mleaf(vec![1.0, 0.8, 1.2], vec![3]);
        let g = mleaf(vec![0.2, -0.1, 0.4, 0.3, 0.5, -0.2], vec![2, 3]);
        let backward = Node::new(NodeKind::LayerNormBackward {
            x,
            weight,
            g,
            eps: 1e-5,
        })
        .unwrap();
        let dw = Node::new(NodeKind::LayerNormBackwardOut {
            of: backward.clone(),
            index: 1,
        })
        .unwrap();
        let db = Node::new(NodeKind::LayerNormBackwardOut {
            of: backward.clone(),
            index: 2,
        })
        .unwrap();
        let compilation = compile_metal_roots(&[dw, db]).unwrap();
        assert_eq!(compilation.executable.program.outputs.len(), 2);
        assert_eq!(compilation.executable.commands().len(), 1);
        assert!(matches!(
            compilation.executable.commands()[0].kind.operation(),
            Some((executable::MetalOp::LayerNormBackward { .. }, _))
        ));
    }

    #[test]
    fn kv_invocation_metadata_fills_only_planned_staging() {
        let pool = NativeKvPool::new(1, 1, 2, 32, Some(16), Some(NativeDType::F32), None).unwrap();
        let sequence = pool.make_sequence().unwrap();
        sequence.state.lock().unwrap().advance = 3;
        let schema = KvStateSchema {
            max_tokens: 32,
            block_size: 16,
            kv_dtype: DType::F32,
            window: Some(16),
            batch: 1,
            graph_batch: 1,
            layers: 1,
            kv_heads: 1,
            head_dim: 2,
            kda: KdaGeometry::default(),
            conv: ConvGeometry::default(),
            cursor_slot: u32::MAX,
            cursor_tensor: false,
        };
        let context = KvContext {
            pool: pool.inner.clone(),
            slots: vec![sequence.state.clone()],
            schema,
            tokens: vec![vec![1, 2, 3]],
            lanes: vec![0],
            packed_rows: None,
            packed_positions: None,
            publish_hashes: true,
            state_only: false,
        };
        let staging = [[1, 2].as_slice(), &[1], &[1], &[1], &[1]]
            .into_iter()
            .map(|shape| {
                let elements = shape.iter().product::<usize>();
                MetalTensor {
                    buffer: MetalDevice::get().alloc_raw(elements * DType::U32.size_in_bytes()),
                    layout: runtime::layout::Layout::contiguous(shape.to_vec()),
                    dtype: DType::U32,
                }
            })
            .collect::<Vec<_>>();
        let plan = executable::KvAttentionPlan {
            batch: 1,
            query_heads: 1,
            kv_heads: 1,
            time: 4,
            head_dim: 2,
            mode: KvAttentionMode::Causal,
            splits: 1,
        };
        let allocation_attempts = device::EXECUTABLE_ALLOCATION_ATTEMPTS.load(Ordering::Relaxed);
        {
            let _guard = MetalDevice::get().begin_executable_dispatch().unwrap();
            context.prepare_kv_attention(0, &plan, &staging).unwrap();
        }
        assert_eq!(
            device::EXECUTABLE_ALLOCATION_ATTEMPTS.load(Ordering::Relaxed),
            allocation_attempts
        );
        assert_eq!(staging[1].to_u32_vec().unwrap(), vec![3]);
        assert_eq!(staging[2].to_u32_vec().unwrap(), vec![0]);
        assert_eq!(staging[3].to_u32_vec().unwrap(), vec![3]);
        assert_eq!(staging[4].to_u32_vec().unwrap(), vec![1]);
        assert_eq!(
            staging[0].to_u32_vec().unwrap()[0],
            context.slots[0].lock().unwrap().blocks[0]
        );
    }

    #[test]
    fn packed_kv_staging_duplicates_sequences_and_keeps_padding_inactive() {
        let pool = NativeKvPool::new(1, 1, 2, 32, Some(4), Some(NativeDType::F32), None).unwrap();
        let first = pool.make_sequence().unwrap();
        let second = pool.make_sequence().unwrap();
        {
            let mut state = first.state.lock().unwrap();
            state.cursor = 4;
            state.advance = 3;
        }
        {
            let mut state = second.state.lock().unwrap();
            state.cursor = 9;
            state.advance = 1;
        }
        let context = KvContext {
            pool: pool.inner.clone(),
            slots: vec![first.state.clone(), second.state.clone()],
            schema: KvStateSchema {
                max_tokens: 32,
                block_size: 4,
                kv_dtype: DType::F32,
                window: None,
                batch: 2,
                graph_batch: 8,
                layers: 1,
                kv_heads: 1,
                head_dim: 2,
                kda: KdaGeometry::default(),
                conv: ConvGeometry::default(),
                cursor_slot: u32::MAX,
                cursor_tensor: true,
            },
            tokens: vec![vec![1, 2, 3], vec![4]],
            lanes: vec![1, 0],
            packed_rows: Some(vec![
                Some(0),
                Some(0),
                Some(0),
                Some(1),
                None,
                None,
                None,
                None,
            ]),
            packed_positions: Some(vec![4, 5, 6, 9, 0, 0, 0, 0]),
            publish_hashes: false,
            state_only: false,
        };
        assert_eq!(
            context.position_offsets().unwrap(),
            [4, 5, 6, 9, 0, 0, 0, 0]
        );
        let staging = [[8, 8].as_slice(), &[8], &[8], &[8], &[8]]
            .into_iter()
            .map(|shape| MetalTensor {
                buffer: MetalDevice::get()
                    .alloc_raw(shape.iter().product::<usize>() * DType::U32.size_in_bytes()),
                layout: runtime::layout::Layout::contiguous(shape.to_vec()),
                dtype: DType::U32,
            })
            .collect::<Vec<_>>();
        let plan = executable::KvAttentionPlan {
            batch: 8,
            query_heads: 1,
            kv_heads: 1,
            time: 1,
            head_dim: 2,
            mode: KvAttentionMode::Causal,
            splits: 1,
        };
        {
            let _guard = MetalDevice::get().begin_executable_dispatch().unwrap();
            context.prepare_kv_attention(0, &plan, &staging).unwrap();
        }
        assert_eq!(staging[1].to_u32_vec().unwrap(), [5, 6, 7, 10, 0, 0, 0, 0]);
        assert_eq!(staging[3].to_u32_vec().unwrap(), [1, 1, 1, 1, 0, 0, 0, 0]);
        let tables = staging[0].to_u32_vec().unwrap();
        assert_eq!(&tables[0..8], &tables[8..16]);
        assert_eq!(&tables[8..16], &tables[16..24]);
        assert!(tables[32..].iter().all(|value| *value == 0));
        assert_eq!(first.state.lock().unwrap().blocks.len(), 2);
        assert_eq!(second.state.lock().unwrap().blocks.len(), 3);
    }

    #[test]
    fn windowed_kv_staging_rebases_the_compact_live_block_table() {
        let pool = NativeKvPool::new(1, 1, 2, 16, Some(4), Some(NativeDType::F32), None).unwrap();
        let sequence = pool.make_sequence().unwrap();
        {
            let mut state = sequence.state.lock().unwrap();
            for _ in 0..4 {
                state.blocks.push(pool.inner.alloc_block().unwrap());
            }
            state.cursor = 16;
            state.advance = 1;
            kv_evict(&pool.inner, &mut state, 9);
            assert_eq!(state.head, 2);
            assert_eq!(state.blocks.len(), 2);
        }
        let schema = KvStateSchema {
            max_tokens: 16,
            block_size: 4,
            kv_dtype: DType::F32,
            window: Some(8),
            batch: 1,
            graph_batch: 1,
            layers: 1,
            kv_heads: 1,
            head_dim: 2,
            kda: KdaGeometry::default(),
            conv: ConvGeometry::default(),
            cursor_slot: u32::MAX,
            cursor_tensor: false,
        };
        let context = KvContext {
            pool: pool.inner.clone(),
            slots: vec![sequence.state.clone()],
            schema,
            tokens: vec![vec![7]],
            lanes: vec![0],
            packed_rows: None,
            packed_positions: None,
            publish_hashes: true,
            state_only: false,
        };
        let staging = [[1, 4].as_slice(), &[1], &[1], &[1], &[1]]
            .into_iter()
            .map(|shape| {
                let elements = shape.iter().product::<usize>();
                MetalTensor {
                    buffer: MetalDevice::get().alloc_raw(elements * DType::U32.size_in_bytes()),
                    layout: runtime::layout::Layout::contiguous(shape.to_vec()),
                    dtype: DType::U32,
                }
            })
            .collect::<Vec<_>>();
        let plan = executable::KvAttentionPlan {
            batch: 1,
            query_heads: 1,
            kv_heads: 1,
            time: 1,
            head_dim: 2,
            mode: KvAttentionMode::Causal,
            splits: 1,
        };
        {
            let _guard = MetalDevice::get().begin_executable_dispatch().unwrap();
            context.prepare_kv_attention(0, &plan, &staging).unwrap();
        }
        let state = sequence.state.lock().unwrap();
        assert_eq!(state.head, 2);
        assert_eq!(state.blocks.len(), 3);
        assert_eq!(staging[1].to_u32_vec().unwrap(), [17]);
        assert_eq!(staging[2].to_u32_vec().unwrap(), [2]);
        assert_eq!(&staging[0].to_u32_vec().unwrap()[..3], state.blocks);
    }

    #[test]
    fn inactive_physical_lanes_have_no_sequence_and_consume_no_blocks() {
        let pool = NativeKvPool::new(1, 1, 2, 8, Some(4), Some(NativeDType::F32), None).unwrap();
        let slots = (0..2)
            .map(|_| {
                let mut state = sequence_state(&pool.inner, false).unwrap();
                state.advance = 1;
                Arc::new(Mutex::new(state))
            })
            .collect::<Vec<_>>();
        let schema = KvStateSchema {
            max_tokens: 8,
            block_size: 4,
            kv_dtype: DType::F32,
            window: None,
            batch: 8,
            graph_batch: 8,
            layers: 1,
            kv_heads: 1,
            head_dim: 2,
            kda: KdaGeometry::default(),
            conv: ConvGeometry::default(),
            cursor_slot: u32::MAX,
            cursor_tensor: false,
        };
        let context = KvContext {
            pool: pool.inner.clone(),
            slots: slots.clone(),
            schema,
            tokens: vec![vec![1]; 2],
            lanes: vec![1, 6],
            packed_rows: None,
            packed_positions: None,
            publish_hashes: true,
            state_only: false,
        };
        let staging = [[8, 2].as_slice(), &[8], &[8], &[8], &[8]]
            .into_iter()
            .map(|shape| {
                let elements = shape.iter().product::<usize>();
                MetalTensor {
                    buffer: MetalDevice::get().alloc_raw(elements * DType::U32.size_in_bytes()),
                    layout: runtime::layout::Layout::contiguous(shape.to_vec()),
                    dtype: DType::U32,
                }
            })
            .collect::<Vec<_>>();
        let plan = executable::KvAttentionPlan {
            batch: 8,
            query_heads: 1,
            kv_heads: 1,
            time: 1,
            head_dim: 2,
            mode: KvAttentionMode::Causal,
            splits: 1,
        };
        {
            let _guard = MetalDevice::get().begin_executable_dispatch().unwrap();
            context.prepare_kv_attention(0, &plan, &staging).unwrap();
        }
        assert_eq!(pool.free_blocks(), 0);
        assert!(slots
            .iter()
            .all(|state| state.lock().unwrap().blocks.len() == 1));
        assert_eq!(staging[1].to_u32_vec().unwrap(), [0, 1, 0, 0, 0, 0, 1, 0]);
    }

    #[test]
    fn planned_stateful_attention_matches_partial_prefill_reference() {
        let q = mleaf(vec![0.0; 8], vec![1, 1, 4, 2]);
        let k = mleaf(vec![0.0; 8], vec![1, 1, 4, 2]);
        let v = mleaf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0, 100.0],
            vec![1, 1, 4, 2],
        );
        let root = LazyTensor {
            node: Node::new(NodeKind::Sdpa {
                q,
                k,
                v,
                scale: 1.0,
                causal: true,
                window: AttentionWindow::Inherit,
            })
            .unwrap(),
        };
        let program = compile(
            vec![&root],
            None,
            Some(NativeKvStateSchema {
                max_tokens: 32,
                block_size: 4,
                kv_dtype: NativeDType::F32,
                window: None,
                batch: 1,
                packed_causal_chains: None,
                last_token_row: None,
                output_selections: None,
                current_block_attention: None,
            }),
            None,
        )
        .unwrap();
        let pool = NativeKvPool::new(1, 1, 2, 32, Some(4), Some(NativeDType::F32), None).unwrap();
        let sequence = pool.make_sequence().unwrap();
        sequence.state.lock().unwrap().advance = 3;
        let context = KvContext {
            pool: pool.inner.clone(),
            slots: vec![sequence.state.clone()],
            schema: program.state.as_ref().unwrap().schema,
            tokens: vec![vec![1, 2, 3]],
            lanes: vec![0],
            packed_rows: None,
            packed_positions: None,
            publish_hashes: true,
            state_only: false,
        };
        let output = executable::execute_stateful(
            &program.inner.executable,
            &[],
            &program.inner.generated_bindings,
            &effect_torch_runtime::CancellationFlag::new(),
            &context,
            &|| true,
            false,
        )
        .unwrap();

        assert_close(
            &output[0].to_f32_vec().unwrap(),
            &[1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 3.0, 4.0],
            1e-6,
            "partial prefill",
        );
    }

    #[test]
    fn planned_stateful_attention_matches_projected_partial_prefill() {
        let input_data = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let mut projection = vec![0.0; 4 * 12];
        for input_index in 0..4 {
            for section in 0..3 {
                projection[input_index * 12 + section * 4 + input_index] = 1.0;
            }
        }
        let build = |input: Arc<Node>| {
            let qkv = linear(
                &input,
                &mleaf(projection.clone(), vec![4, 12]),
                &mleaf(vec![0.0; 12], vec![12]),
            );
            let split = |start: usize| {
                let sliced = Node::new(NodeKind::Slice {
                    a: qkv.clone(),
                    ranges: vec![(0, 1, 1), (0, 4, 1), (start, start + 4, 1)],
                })
                .unwrap();
                let reshaped = Node::new(NodeKind::Reshape {
                    a: sliced,
                    shape: vec![1, 4, 2, 2],
                })
                .unwrap();
                Node::new(NodeKind::Permute {
                    a: reshaped,
                    dims: vec![0, 2, 1, 3],
                })
                .unwrap()
            };
            let attended = Node::new(NodeKind::Sdpa {
                q: split(0),
                k: split(4),
                v: split(8),
                scale: 1.0 / 2.0f64.sqrt(),
                causal: true,
                window: AttentionWindow::Inherit,
            })
            .unwrap();
            Node::new(NodeKind::Reshape {
                a: Node::new(NodeKind::Permute {
                    a: attended,
                    dims: vec![0, 2, 1, 3],
                })
                .unwrap(),
                shape: vec![1, 4, 4],
            })
            .unwrap()
        };
        let reference = eval_f32(&build(mleaf(input_data.clone(), vec![1, 4, 4])));
        let declared_input = Node::new(NodeKind::Input {
            slot: 0,
            shape: vec![1, 4, 4],
            dtype: DType::F32,
            device: get_device(None),
        })
        .unwrap();
        let root = LazyTensor {
            node: build(declared_input),
        };
        let program = compile(
            vec![&root],
            None,
            Some(NativeKvStateSchema {
                max_tokens: 32,
                block_size: 4,
                kv_dtype: NativeDType::F32,
                window: None,
                batch: 1,
                packed_causal_chains: None,
                last_token_row: None,
                output_selections: None,
                current_block_attention: None,
            }),
            None,
        )
        .unwrap();
        let pool = NativeKvPool::new(1, 2, 2, 32, Some(4), Some(NativeDType::F32), None).unwrap();
        let sequence = pool.make_sequence().unwrap();
        sequence.state.lock().unwrap().advance = 3;
        let context = KvContext {
            pool: pool.inner.clone(),
            slots: vec![sequence.state.clone()],
            schema: program.state.as_ref().unwrap().schema,
            tokens: vec![vec![1, 2, 3]],
            lanes: vec![0],
            packed_rows: None,
            packed_positions: None,
            publish_hashes: true,
            state_only: false,
        };
        let output = executable::execute_stateful(
            &program.inner.executable,
            &[value::Value(MetalTensor::from_f32(
                MetalDevice::get(),
                input_data,
                vec![1, 4, 4],
            ))],
            &program.inner.generated_bindings,
            &effect_torch_runtime::CancellationFlag::new(),
            &context,
            &|| true,
            false,
        )
        .unwrap()[0]
            .to_f32_vec()
            .unwrap();

        assert_close(&output[..12], &reference[..12], 1e-6, "projected prefill");
    }

    #[test]
    fn planned_stateful_one_token_decode_uses_split_attention() {
        let q = mleaf(vec![0.0; 2], vec![1, 1, 1, 2]);
        let k = mleaf(vec![0.0; 2], vec![1, 1, 1, 2]);
        let v = mleaf(vec![5.0, 7.0], vec![1, 1, 1, 2]);
        let root = LazyTensor {
            node: Node::new(NodeKind::Sdpa {
                q,
                k,
                v,
                scale: 1.0,
                causal: true,
                window: AttentionWindow::Inherit,
            })
            .unwrap(),
        };
        let program = compile(
            vec![&root],
            None,
            Some(NativeKvStateSchema {
                max_tokens: 32,
                block_size: 4,
                kv_dtype: NativeDType::F32,
                window: None,
                batch: 1,
                packed_causal_chains: None,
                last_token_row: None,
                output_selections: None,
                current_block_attention: None,
            }),
            None,
        )
        .unwrap();
        let plan = program
            .inner
            .executable
            .commands()
            .iter()
            .filter_map(|command| command.kind.operation())
            .find_map(|(op, plan)| match (op, plan) {
                (
                    executable::MetalOp::KvAttention { .. },
                    executable::MetalCommandPlan::KvAttention(plan),
                ) => Some(*plan),
                _ => None,
            })
            .expect("compiled KV attention plan");
        assert_eq!(plan.splits, 8);
        let pool = NativeKvPool::new(1, 1, 2, 32, Some(4), Some(NativeDType::F32), None).unwrap();
        let sequence = pool.make_sequence().unwrap();
        sequence.state.lock().unwrap().advance = 1;
        let context = KvContext {
            pool: pool.inner.clone(),
            slots: vec![sequence.state.clone()],
            schema: program.state.as_ref().unwrap().schema,
            tokens: vec![vec![7]],
            lanes: vec![0],
            packed_rows: None,
            packed_positions: None,
            publish_hashes: true,
            state_only: false,
        };
        let output = executable::execute_stateful(
            &program.inner.executable,
            &[],
            &program.inner.generated_bindings,
            &effect_torch_runtime::CancellationFlag::new(),
            &context,
            &|| true,
            false,
        )
        .unwrap();

        assert_close(
            &output[0].to_f32_vec().unwrap(),
            &[5.0, 7.0],
            1e-6,
            "split decode",
        );
    }

    #[test]
    fn failed_decode_setup_restores_all_sequence_metadata() {
        let pool = NativeKvPool::new(1, 1, 2, 32, Some(16), Some(NativeDType::F32), None).unwrap();
        let sequence = pool.make_sequence().unwrap();
        sequence.state.lock().unwrap().advance = 7;
        let available = pool.free_blocks();
        let checkpoint = {
            let state = sequence.state.lock().unwrap();
            SequenceCheckpoint {
                blocks: state.blocks.len(),
                advance: state.advance,
                kda_states: state.kda_states.len(),
                conv_states: state.conv_states.len(),
            }
        };
        {
            let mut state = sequence.state.lock().unwrap();
            state.advance = 3;
            state.blocks.push(pool.inner.alloc_block().unwrap());
            prepare_recurrent_states(
                &mut state,
                KdaGeometry {
                    layers: 1,
                    heads: 1,
                    head_dim: 2,
                    value_dim: 2,
                },
                ConvGeometry {
                    layers: 1,
                    kernel: 3,
                    channels: 2,
                },
            )
            .unwrap();
        }

        rollback_sequence_setup(
            &pool.inner,
            std::slice::from_ref(&sequence.state),
            &[checkpoint],
        )
        .unwrap();
        let state = sequence.state.lock().unwrap();
        assert!(state.blocks.is_empty());
        assert_eq!(state.advance, 7);
        assert!(state.kda_states.is_empty());
        assert!(state.conv_states.is_empty());
        assert_eq!(pool.free_blocks(), available);
    }

    #[test]
    fn prefill_match_rechecks_release_after_acquiring_the_run_lock() {
        let pool = NativeKvPool::new(1, 1, 2, 32, Some(16), Some(NativeDType::F32), None).unwrap();
        let sequence = Arc::new(pool.make_sequence().unwrap());
        let run_guard = sequence.run_lock.lock().unwrap();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let worker = {
            let sequence = sequence.clone();
            std::thread::spawn(move || {
                started_tx.send(()).unwrap();
                sequence.prefill_match((0..17).collect())
            })
        };
        started_rx.recv().unwrap();
        sequence.released.store(true, Ordering::SeqCst);
        drop(run_guard);

        let error = worker.join().unwrap().unwrap_err();
        assert!(error.to_string().contains("kv sequence is released"));
        let state = sequence.state.lock().unwrap();
        assert!(state.blocks.is_empty());
        assert_eq!(state.cursor, 0);
    }

    #[test]
    fn recurrent_state_is_prepared_before_execution_and_padding_reuses_pool_zeros() {
        let pool = NativeKvPool::new(
            0,
            0,
            0,
            32,
            Some(16),
            Some(NativeDType::F32),
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
        let sequence = pool.make_sequence().unwrap();
        let state = sequence.state.lock().unwrap();
        assert_eq!(state.kda_states.len(), 1);
        assert_eq!(state.conv_states.len(), 1);
        assert!(!Arc::ptr_eq(
            &state.kda_states[0].buffer,
            &pool.inner.padding_kda_states[0].buffer
        ));
        drop(state);

        let allocations = device::EXECUTABLE_ALLOCATION_ATTEMPTS.load(Ordering::Relaxed);
        let padding = {
            let _guard = MetalDevice::get().begin_executable_dispatch().unwrap();
            sequence.new_sequence_like()
        };
        assert_eq!(
            device::EXECUTABLE_ALLOCATION_ATTEMPTS.load(Ordering::Relaxed),
            allocations
        );
        let state = padding.state.lock().unwrap();
        assert!(Arc::ptr_eq(
            &state.kda_states[0].buffer,
            &pool.inner.padding_kda_states[0].buffer
        ));
        assert!(Arc::ptr_eq(
            &state.conv_states[0].buffer,
            &pool.inner.padding_conv_states[0].buffer
        ));
    }

    #[test]
    fn stateful_compile_requires_and_retains_the_complete_kv_schema() {
        let q = mleaf(vec![1.0, 0.0], vec![1, 1, 1, 2]);
        let root = LazyTensor {
            node: Node::new(NodeKind::Sdpa {
                q: q.clone(),
                k: q.clone(),
                v: q,
                scale: 1.0,
                causal: true,
                window: AttentionWindow::Inherit,
            })
            .unwrap(),
        };
        let program = compile(
            vec![&root],
            None,
            Some(NativeKvStateSchema {
                max_tokens: 64,
                block_size: 16,
                kv_dtype: NativeDType::F16,
                window: Some(32),
                batch: 1,
                packed_causal_chains: None,
                last_token_row: None,
                output_selections: None,
                current_block_attention: None,
            }),
            None,
        )
        .unwrap();
        assert!(program.stateful());
        assert!(program.allows_window_eviction());
        assert_eq!(program.batch(), 1);
        assert_eq!(program.layers(), 1);
        assert_eq!(
            program.inner.executable.options.inference,
            Some(InferenceOptions {
                constant_weights: false
            })
        );
        assert_eq!(
            program.inner.executable.state_schema,
            Some(KvStateSchema {
                max_tokens: 64,
                block_size: 16,
                kv_dtype: DType::F16,
                window: Some(32),
                batch: 1,
                graph_batch: 1,
                layers: 1,
                kv_heads: 1,
                head_dim: 2,
                kda: KdaGeometry::default(),
                conv: ConvGeometry::default(),
                cursor_slot: 0,
                cursor_tensor: false,
            })
        );
        let commands = program.inner.executable.commands();
        let command = commands
            .iter()
            .find(|command| {
                matches!(
                    command.kind.operation(),
                    Some((executable::MetalOp::KvAttention { .. }, _))
                )
            })
            .unwrap();
        assert_eq!(command.staging.len(), 5);
    }

    fn hybrid_pool(max_tokens: u32, block_size: u32) -> NativeKvPool {
        NativeKvPool::new(
            1,
            1,
            2,
            max_tokens,
            Some(block_size),
            Some(NativeDType::F32),
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
        .unwrap()
    }

    fn hybrid_schema(max_tokens: usize, block_size: usize) -> KvStateSchema {
        KvStateSchema {
            max_tokens,
            block_size,
            kv_dtype: DType::F32,
            window: None,
            batch: 1,
            graph_batch: 1,
            layers: 1,
            kv_heads: 1,
            head_dim: 2,
            kda: KdaGeometry {
                layers: 1,
                heads: 1,
                head_dim: 2,
                value_dim: 2,
            },
            conv: ConvGeometry {
                layers: 1,
                channels: 2,
                kernel: 3,
            },
            cursor_slot: u32::MAX,
            cursor_tensor: false,
        }
    }

    fn state_f32(tensor: &MetalTensor) -> Vec<f32> {
        // SAFETY: test fixtures pass contiguous f32 state tensors; their
        // retained buffers contain `numel` initialized values from `offset`.
        unsafe {
            std::slice::from_raw_parts(
                tensor
                    .buffer
                    .contents_ptr()
                    .cast::<f32>()
                    .add(tensor.layout.offset()),
                tensor.numel(),
            )
            .to_vec()
        }
    }

    fn set_state_f32(tensor: &MetalTensor, data: &[f32]) {
        assert_eq!(tensor.numel(), data.len());
        // SAFETY: the assertion proves equal extents, fixtures use contiguous
        // f32 state storage, and the host slice cannot overlap Metal memory.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                tensor
                    .buffer
                    .contents_ptr()
                    .cast::<f32>()
                    .add(tensor.layout.offset()),
                data.len(),
            );
        }
    }

    fn commit_tokens(pool: &NativeKvPool, sequence: &NativeKvSequence, tokens: &[u32]) {
        let context = KvContext {
            pool: pool.inner.clone(),
            slots: vec![sequence.state.clone()],
            schema: hybrid_schema(pool.inner.max_tokens, pool.inner.block_size),
            tokens: vec![tokens.to_vec()],
            lanes: vec![0],
            packed_rows: None,
            packed_positions: None,
            publish_hashes: true,
            state_only: false,
        };
        let mut state = sequence.state.lock().unwrap();
        let needed = (state.cursor + tokens.len()).div_ceil(pool.inner.block_size);
        while state.head + state.blocks.len() < needed {
            state.blocks.push(pool.inner.alloc_block().unwrap());
        }
        state.advance = tokens.len();
        context.commit_slot(0, &mut state);
    }

    #[test]
    fn recurrent_snapshots_publish_only_at_the_final_block_boundary() {
        let pool = hybrid_pool(32, 4);
        let sequence = pool.make_sequence().unwrap();
        let tokens: Vec<u32> = (0..8).collect();
        {
            let state = sequence.state.lock().unwrap();
            set_state_f32(&state.kda_states[0], &[1.0, 2.0, 3.0, 4.0]);
            set_state_f32(&state.conv_states[0], &[5.0, 6.0, 7.0, 8.0]);
        }
        commit_tokens(&pool, &sequence, &tokens);
        assert_eq!(sequence.cursor(), 8);
        let first = chain_hash(HASH_SEED, &tokens[..4]);
        let second = chain_hash(first, &tokens[4..]);
        let store = pool.inner.blocks.lock().unwrap();
        assert_eq!(store.snapshots.len(), 1);
        assert!(!store.snapshots.contains_key(&first));
        let snapshot = store.snapshots.get(&second).unwrap();
        assert_eq!(&*snapshot.kda[0], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&*snapshot.conv[0], &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn recurrent_snapshots_skip_runs_ending_mid_block() {
        let pool = hybrid_pool(32, 4);
        let sequence = pool.make_sequence().unwrap();
        commit_tokens(&pool, &sequence, &[1, 2, 3]);
        assert_eq!(sequence.cursor(), 3);
        assert!(pool.inner.blocks.lock().unwrap().snapshots.is_empty());
    }

    #[test]
    fn recurrent_snapshots_are_not_published_for_padding_slots() {
        let pool = hybrid_pool(32, 4);
        let sequence = pool.make_sequence().unwrap();
        let padding = sequence.new_sequence_like();
        let context = KvContext {
            pool: pool.inner.clone(),
            slots: vec![sequence.state.clone(), padding.state.clone()],
            schema: hybrid_schema(32, 4),
            tokens: vec![vec![1, 2, 3, 4], vec![1, 2, 3, 4]],
            lanes: vec![0],
            packed_rows: None,
            packed_positions: None,
            publish_hashes: true,
            state_only: false,
        };
        let mut states = context
            .slots
            .iter()
            .map(|slot| slot.lock().unwrap())
            .collect::<Vec<_>>();
        {
            let state = &mut states[0];
            state.blocks.push(pool.inner.alloc_block().unwrap());
            state.advance = 4;
            set_state_f32(&state.kda_states[0], &[1.0, 2.0, 3.0, 4.0]);
        }
        for (index, state) in states.iter_mut().take(context.active_batch()).enumerate() {
            context.commit_slot(index, state);
        }
        assert_eq!(states[1].cursor, 0);
        let store = pool.inner.blocks.lock().unwrap();
        assert_eq!(store.snapshots.len(), 1);
        let snapshot = store
            .snapshots
            .get(&chain_hash(HASH_SEED, &[1, 2, 3, 4]))
            .unwrap();
        assert_eq!(&*snapshot.kda[0], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn pure_recurrent_pools_never_publish_or_match() {
        let pool = NativeKvPool::new(
            0,
            0,
            0,
            32,
            Some(4),
            Some(NativeDType::F32),
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
        let sequence = pool.make_sequence().unwrap();
        commit_tokens(&pool, &sequence, &[1, 2, 3, 4]);
        assert!(pool.inner.blocks.lock().unwrap().snapshots.is_empty());
        let other = pool.make_sequence().unwrap();
        assert_eq!(other.prefill_match(vec![1, 2, 3, 4, 5]).unwrap(), 0);
        assert!(other.state.lock().unwrap().blocks.is_empty());
    }

    #[test]
    fn prefill_match_restores_deep_copies_of_the_boundary_snapshot() {
        let pool = hybrid_pool(32, 4);
        let sequence = pool.make_sequence().unwrap();
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        {
            let state = sequence.state.lock().unwrap();
            set_state_f32(&state.kda_states[0], &[1.0, 2.0, 3.0, 4.0]);
            set_state_f32(&state.conv_states[0], &[5.0, 6.0, 7.0, 8.0]);
        }
        commit_tokens(&pool, &sequence, &tokens);
        {
            let state = sequence.state.lock().unwrap();
            set_state_f32(&state.kda_states[0], &[9.0; 4]);
            set_state_f32(&state.conv_states[0], &[9.0; 4]);
        }

        let resumed = pool.make_sequence().unwrap();
        assert_eq!(resumed.prefill_match(vec![1, 2, 3, 4, 5]).unwrap(), 4);
        {
            let state = resumed.state.lock().unwrap();
            assert_eq!(state.cursor, 4);
            assert_eq!(state.last_hash, chain_hash(HASH_SEED, &tokens));
            assert_eq!(state.blocks.len(), 1);
            assert_eq!(state_f32(&state.kda_states[0]), vec![1.0, 2.0, 3.0, 4.0]);
            assert_eq!(state_f32(&state.conv_states[0]), vec![5.0, 6.0, 7.0, 8.0]);
            set_state_f32(&state.kda_states[0], &[7.0; 4]);
            set_state_f32(&state.conv_states[0], &[7.0; 4]);
        }

        let third = pool.make_sequence().unwrap();
        assert_eq!(third.prefill_match(vec![1, 2, 3, 4, 5]).unwrap(), 4);
        let state = third.state.lock().unwrap();
        assert_eq!(state_f32(&state.kda_states[0]), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(state_f32(&state.conv_states[0]), vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn prefill_match_truncates_to_the_deepest_snapshot_boundary() {
        let pool = hybrid_pool(32, 4);
        let sequence = pool.make_sequence().unwrap();
        commit_tokens(&pool, &sequence, &[0, 1, 2, 3]);
        let first = chain_hash(HASH_SEED, &[0, 1, 2, 3]);
        let extra = pool.inner.alloc_block().unwrap();
        pool.inner.set_hash(extra, chain_hash(first, &[4, 5, 6, 7]));

        let resumed = pool.make_sequence().unwrap();
        assert_eq!(resumed.prefill_match((0..9).collect()).unwrap(), 4);
        {
            let state = resumed.state.lock().unwrap();
            assert_eq!(state.cursor, 4);
            assert_eq!(state.last_hash, first);
            assert_eq!(state.blocks.len(), 1);
        }
        assert_eq!(
            pool.inner.blocks.lock().unwrap().refcounts[extra as usize],
            1
        );
    }

    #[test]
    fn prefill_match_without_a_snapshot_boundary_holds_nothing() {
        let pool = hybrid_pool(32, 4);
        let block = pool.inner.alloc_block().unwrap();
        pool.inner
            .set_hash(block, chain_hash(HASH_SEED, &[1, 2, 3, 4]));
        let sequence = pool.make_sequence().unwrap();
        assert_eq!(sequence.prefill_match(vec![1, 2, 3, 4, 5]).unwrap(), 0);
        assert!(sequence.state.lock().unwrap().blocks.is_empty());
        assert_eq!(
            pool.inner.blocks.lock().unwrap().refcounts[block as usize],
            1
        );
    }

    #[test]
    fn prefill_match_unrefs_taken_blocks_when_the_restore_fails() {
        let pool = hybrid_pool(32, 4);
        let sequence = pool.make_sequence().unwrap();
        commit_tokens(&pool, &sequence, &[1, 2, 3, 4]);
        let block = sequence.state.lock().unwrap().blocks[0];
        let hash = chain_hash(HASH_SEED, &[1, 2, 3, 4]);
        pool.inner.blocks.lock().unwrap().snapshots.insert(
            hash,
            Arc::new(RecurrentSnapshot {
                kda: Vec::new(),
                conv: Vec::new(),
            }),
        );
        let resumed = pool.make_sequence().unwrap();
        let error = resumed.prefill_match(vec![1, 2, 3, 4, 5]).unwrap_err();
        assert!(error.to_string().contains("recurrent state restore failed"));
        {
            let state = resumed.state.lock().unwrap();
            assert!(state.blocks.is_empty());
            assert_eq!(state.cursor, 0);
            assert_eq!(state.last_hash, HASH_SEED);
        }
        assert_eq!(
            pool.inner.blocks.lock().unwrap().refcounts[block as usize],
            1
        );
    }

    #[test]
    fn recurrent_snapshots_are_removed_with_their_last_resident_block() {
        let pool = hybrid_pool(8, 4);
        let sequence = pool.make_sequence().unwrap();
        commit_tokens(&pool, &sequence, &[1, 2, 3, 4]);
        let block = sequence.state.lock().unwrap().blocks[0];
        let hash = chain_hash(HASH_SEED, &[1, 2, 3, 4]);
        assert!(pool
            .inner
            .blocks
            .lock()
            .unwrap()
            .snapshots
            .contains_key(&hash));
        sequence.release();
        assert!(pool
            .inner
            .blocks
            .lock()
            .unwrap()
            .snapshots
            .contains_key(&hash));
        let _other = pool.inner.alloc_block().unwrap();
        let evicted = pool.inner.alloc_block().unwrap();
        assert_eq!(evicted, block);
        assert!(pool.inner.blocks.lock().unwrap().snapshots.is_empty());
    }
}

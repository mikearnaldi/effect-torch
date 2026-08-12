mod err;
mod gguf;
mod runtime;
pub(crate) mod safetensors;
pub(crate) mod value;

use crate::executable;
use crate::executable::{ConvGeometry, KdaGeometry, KvStateSchema, MetalDecodeContext, SeqState};

use effect_torch_compiler::{
    specialize_decode, CompileOptions, ConvGeometry as DecodeConvGeometry, InferenceOptions,
    KdaGeometry as DecodeKdaGeometry, PreparedProgram, ProgramRequest, ProgramSlot,
    StateCursorSlot,
};
use effect_torch_graph::CrossEntropyReduction as CeReduction;
use effect_torch_graph::{AttentionWindow, Device, PositionOffset, RotaryLayout};
use effect_torch_napi::{try_register_export, unregister_export, vec_to_bytes, CancellationState};
use effect_torch_runtime::{Buffer, GgmlKQuant};
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
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
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
            drop(unsafe { Vec::from_raw_parts(ptr, len, cap) });
        }
    }
}

pub struct Readback {
    data: *mut u8,
    byte_len: usize,
    hint: Option<FinalizeHint>,
}

unsafe impl Send for Readback {}

struct FinalizeHintGuard(*mut std::ffi::c_void);

impl Drop for FinalizeHintGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
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
    pub last_token_row: Option<bool>,
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

fn dtype_name(dtype: DType) -> &'static str {
    dtype.name()
}

fn get_device() -> Device {
    Device::Metal
}

#[napi(custom_finalize)]
pub struct NativeTensor {
    pub(crate) slot: std::sync::Arc<LeafSlot>,
    bytes: i64,
}

impl NativeTensor {
    fn wrap(inner: value::Value) -> Self {
        // Buffers cost at least a memory page regardless of the tensor's
        // logical size (Metal allocates 4KB-granular, malloc similar). Without
        // reporting that floor, a stream of tiny tensors looks free to V8 and
        // collection is deferred indefinitely — the backend allocator then
        // can't reuse the pooled buffers (the pool requires
        // strong_count == 1) and both memory and per-allocation cost grow
        // without bound.
        let bytes = inner.byte_size().max(4096) as i64;
        // Accounting is native-only: every handle that reaches JS is counted
        // here at creation and subtracted in the finalizer/dispose. V8 is
        // told the delta at the next main-thread touchpoint (see sync_v8);
        // no JS-side involvement, so no missed sites and no drift.
        EXTERNAL_MEMORY_BYTES.fetch_add(bytes, Ordering::Relaxed);
        Self {
            slot: std::sync::Arc::new(LeafSlot::new(inner)),
            bytes,
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
    /// Releases the tensor's buffer early instead of waiting for the
    /// garbage collector. Using the handle — or any lazy graph built
    /// from it — afterwards is a typed error.
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
        run_compute(token, move |cancelled, _state| {
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
    let bytes = unsafe { std::slice::from_raw_parts(base.add(offset), byte_len) }.to_vec();
    let (_, ptr, len, cap) = vec_to_bytes(bytes);
    Ok(Readback {
        data: ptr,
        byte_len: len,
        hint: Some(FinalizeHint::Owned { ptr, len, cap }),
    })
}

struct ConstantCache {
    map: HashMap<(u64, DType, &'static str), Arc<Node>>,
    order: std::collections::VecDeque<(u64, DType, &'static str)>,
}

static CONSTANT_CACHE: LazyLock<Mutex<ConstantCache>> = LazyLock::new(|| {
    Mutex::new(ConstantCache {
        map: HashMap::new(),
        order: std::collections::VecDeque::new(),
    })
});

const CONSTANT_CACHE_LIMIT: usize = 4096;

fn device_key(device: &Device) -> &'static str {
    debug_assert!(device.is_metal());
    "metal"
}

fn cached_constant(
    value: f64,
    dtype: DType,
    device: Device,
) -> std::result::Result<Arc<Node>, String> {
    let key = (value.to_bits(), dtype, device_key(&device));
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

// RFC 0016 phase 2 — chunked head. cross_entropy(Linear(x, w, b)) with a
// huge logits tensor (LM vocab heads) is rewritten at graph construction
// into per-chunk Sum cross-entropies, each wrapped in a Checkpoint so the
// chunk logits live one chunk at a time (recomputed in backward) instead
// of the full [rows, vocab] tensor being retained for the whole walk.
// The chunk sums are combined in f32 and divided by the exact active
// count, reproducing the Mean reduction bit-closely; model code is
// untouched and the rewrite is backend-agnostic.
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
    pub fn zeros(shape: Vec<u32>, dtype: Option<NativeDType>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Zeros {
            shape: shape.iter().map(|&d| d as usize).collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(),
        }))
    }

    #[napi(factory)]
    pub fn ones(shape: Vec<u32>, dtype: Option<NativeDType>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Ones {
            shape: shape.iter().map(|&d| d as usize).collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(),
        }))
    }

    #[napi(factory)]
    pub fn full(shape: Vec<u32>, value: f64, dtype: Option<NativeDType>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Full {
            shape: shape.iter().map(|&d| d as usize).collect(),
            value,
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(),
        }))
    }

    #[napi(factory)]
    pub fn randn(shape: Vec<u32>, dtype: Option<NativeDType>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Randn {
            shape: shape.iter().map(|&d| d as usize).collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(),
        }))
    }

    #[napi(factory)]
    pub fn uniform(shape: Vec<u32>, lo: f64, hi: f64, dtype: Option<NativeDType>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Uniform {
            lo,
            hi,
            shape: shape.iter().map(|&d| d as usize).collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(),
        }))
    }

    #[napi(factory)]
    pub fn arange(start: f64, end: f64, step: f64, dtype: Option<NativeDType>) -> Result<Self> {
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
            device: get_device(),
        }))
    }

    #[napi(factory)]
    pub fn eye(n: u32, dtype: Option<NativeDType>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Eye {
            n: n as usize,
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(),
        }))
    }

    // A shared 0-d constant: the same (value, dtype, device) triple maps to
    // one graph node forever instead of allocating a fresh node per use.
    // Nodes hold no buffers, so the cache is cheap; it is size-bounded so
    // cold values rotate through. Devices are process singletons, so the
    // device kind is the whole key.
    #[napi(factory)]
    pub fn constant(value: f64, dtype: Option<NativeDType>) -> Result<Self> {
        let device = get_device();
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
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::FromBytes {
            data: data.to_vec(),
            shape: shape.iter().map(|&d| d as usize).collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(),
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
    pub fn input(slot: u32, shape: Vec<u32>, dtype: Option<NativeDType>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Input {
            slot,
            shape: shape.iter().map(|&d| d as usize).collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(),
        }))
    }

    #[napi(factory)]
    pub fn scalar_input(slot: u32, dtype: Option<NativeDType>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::ScalarInput {
            slot,
            dtype: dtype.unwrap_or(NativeDType::F64).into(),
            device: get_device(),
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

// RFC 0010: paged KV inference. A `NativeKvPool` is a fixed-capacity
// store of key/value rows per attention layer, allocated once per
// inference artifact; a `NativeKvSequence` is a block table and cursor
// over the pool (the OS paging model: blocks are pages, sequences are
// processes). Stateful `compile` rewrites a traced forward graph for
// generation — causal Sdpa becomes KvAttention (scatter the new tokens
// into the pool, attend over the cached context), PositionEmbedding
// becomes a cursor-offset gather — and freezes the result like
// `compile`. The frozen graph stays a pure function of its inputs: the
// pool and sequence travel through the run's kv context, parallel runs
// of one program write disjoint blocks, and per-sequence runs serialize
// on the sequence's run lock.

// Chained FNV-1a over a token block: the hash of block i covers the
// whole prefix through block i, so equal hashes imply equal tokens at
// equal absolute positions — with RoPE that makes the cached rows
// bit-identical to a recompute.
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

// Block ownership and the prefix cache. Blocks carry a refcount and,
// once fully written, a chained content hash. Sharing is
// content-addressed and works across LIVE sequences: a prompt whose
// prefix is resident — held by a running sequence or unreferenced in
// the cache — takes a reference instead of recomputing. Unreferenced
// hashed blocks form the LRU cache, reclaimed under pressure.
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
    let ptr = unsafe {
        tensor
            .buffer
            .contents_ptr()
            .cast::<f32>()
            .add(tensor.layout.offset())
    };
    Some(unsafe { std::slice::from_raw_parts(ptr, tensor.numel()) }.into())
}

fn write_f32_state(tensor: &runtime::metal::run::MetalTensor, data: &[f32]) -> err::Res<()> {
    if tensor.dtype != DType::F32 || !tensor.layout.is_contiguous() || tensor.numel() != data.len()
    {
        return Err("recurrent state destination does not match its snapshot".to_string());
    }
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
    // Per layer, flat [max_tokens, kv_heads, head_dim] slabs; block b
    // occupies rows b*block_size..(b+1)*block_size. Slab dtype u8 means
    // int8-quantized storage (RFC 0012 storage tier): rows are
    // symmetric-quantized with a per-(token, head) absmax scale held in
    // `scales` — two slabs per layer (k then v) when the data slabs are
    // u8, empty otherwise.
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
    // Takes a fresh block with refcount 1: free list first, then LRU
    // eviction of unreferenced cached blocks.
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
        for (i, &token) in tokens.iter().enumerate() {
            self.pending.push(token);
            if self.pending.len() == pool.block_size {
                let hash = chain_hash(self.last_hash, &self.pending);
                self.last_hash = hash;
                self.pending.clear();
                // The block holding this token completed; it was
                // allocated by the run that wrote its first row.
                let block_index = (self.cursor + i) / pool.block_size;
                if let Some(&block) = block_index
                    .checked_sub(self.head)
                    .and_then(|index| self.blocks.get(index))
                {
                    pool.set_hash(block, hash);
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
    // One slot per leading batch element of the program's signature:
    // slot b owns batch row b of every KvAttention's q/k/v — its block
    // table, cursor, window, advance. Single-sequence runs have one
    // slot (RFC 0013).
    pub(crate) slots: Vec<Arc<Mutex<SeqState>>>,
    schema: KvStateSchema,
    tokens: Vec<Vec<u32>>,
    active_batch: usize,
}

impl MetalDecodeContext for KvContext {
    fn schema(&self) -> &KvStateSchema {
        &self.schema
    }

    fn slots(&self) -> &[Arc<Mutex<SeqState>>] {
        &self.slots
    }

    fn active_batch(&self) -> usize {
        self.active_batch
    }

    fn prepare_state(&self, cursor: &runtime::metal::run::MetalTensor) -> err::Res<()> {
        let shape = if self.schema.cursor_tensor {
            vec![self.schema.batch]
        } else {
            Vec::new()
        };
        cursor.validate_destination("decode cursor", &shape, DType::I64)?;
        let count = if self.schema.cursor_tensor {
            self.slots.len()
        } else {
            1
        };
        for (index, slot) in self.slots.iter().take(count).enumerate() {
            let value = slot
                .lock()
                .map_err(|error| format!("decode cursor: sequence lock poisoned: {error}"))?
                .cursor as i64;
            unsafe {
                cursor
                    .buffer
                    .contents_ptr()
                    .cast::<i64>()
                    .add(cursor.layout.offset() + index)
                    .write(value);
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
        output: &runtime::metal::run::MetalTensor,
        staging: &[runtime::metal::run::MetalTensor],
    ) -> err::Res<()> {
        kv_attention_into(self, layer, q, k, v, scale, window, output, staging)
    }

    fn evict_before(&self, state: &mut SeqState, start: usize) {
        kv_evict(&self.pool, state, start);
    }

    fn commit_slot(&self, index: usize, state: &mut SeqState) {
        state.note_tokens(&self.pool, &self.tokens[index]);
        state.cursor += state.advance;
        state.advance = 0;
        self.pool.publish_recurrent_snapshot(state);
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
    if plan.batch != schema.batch
        || plan.kv_heads != schema.kv_heads
        || plan.head_dim != schema.head_dim
        || plan.batch != kv.slots.len()
    {
        return Err(format!(
            "kv attention: fixed plan does not match the bound state schema"
        ));
    }
    let expected_shapes = [
        vec![schema.batch, schema.max_tokens / schema.block_size],
        vec![schema.batch],
        vec![schema.batch],
        vec![schema.batch],
        vec![schema.batch],
    ];
    for (tensor, shape) in staging.iter().zip(&expected_shapes) {
        tensor.validate_destination("kv staging", shape, DType::U32)?;
    }
    let table = &staging[0];
    let max_blocks = schema.max_tokens / schema.block_size;
    let table_bytes = schema
        .batch
        .checked_mul(max_blocks)
        .and_then(|elements| elements.checked_mul(DType::U32.size_in_bytes()))
        .ok_or_else(|| "kv attention: block table byte size overflow".to_string())?;
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
                schema.batch * DType::U32.size_in_bytes(),
            );
        }
    }
    let layer = layer as usize;
    for (batch_index, slot) in kv.slots.iter().take(kv.active_batch).enumerate() {
        let mut state = slot
            .lock()
            .map_err(|error| format!("kv attention: sequence lock poisoned: {error}"))?;
        let (_, needed, _) = kv_prepare(
            &kv.pool,
            &mut state,
            layer,
            schema.window,
            plan.kv_heads,
            plan.head_dim,
            plan.time,
        )?;
        if state.blocks.len() > max_blocks {
            return Err("kv attention: block table exceeds its schema capacity".to_string());
        }
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
    output: &runtime::metal::run::MetalTensor,
    staging: &[runtime::metal::run::MetalTensor],
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
    if batch != kv.slots.len() || output.layout.shape() != q.layout.shape() {
        return Err("kv attention: destination shape or decode batch is inconsistent".to_string());
    }
    let advances = kv
        .slots
        .iter()
        .take(kv.active_batch)
        .map(|slot| {
            slot.lock()
                .map(|state| state.advance)
                .map_err(|error| format!("kv attention: sequence lock poisoned: {error}"))
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let advance = advances.first().copied().unwrap_or(0);
    if advance == 0 || advance > time || advances.iter().any(|value| *value != advance) {
        return Err(format!(
            "kv attention: paged batch requires one non-zero uniform advance, got {advances:?}"
        ));
    }
    if staging.len() != 5 {
        return Err("kv attention: planned invocation staging is missing".to_string());
    }
    let (tables, context_lengths, block_bases) = (&staging[0], &staging[1], &staging[2]);
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
        advance,
        paged::IntoResources::empty(),
    )?;
    paged::attention_into(
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
        advance,
        output,
        paged::IntoResources::empty(),
    )?;
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
    let start = window.map_or(0, |w| needed.saturating_sub(w));
    if needed - start > pool.max_tokens {
        return Err(format!(
            "kv attention: live context {} exceeds pool capacity {}",
            needed - start,
            pool.max_tokens
        ));
    }
    let needed_blocks = needed.div_ceil(pool.block_size);
    while state.head + state.blocks.len() < needed_blocks {
        let block = pool.alloc_block().ok_or_else(|| {
            err::err_str(format!(
                "kv attention: pool exhausted ({} tokens across live sequences)",
                pool.max_tokens
            ))
        })?;
        state.blocks.push(block);
    }
    Ok((cursor, needed, start))
}

// again. The last reference lands them in the prefix cache — their
// content is still valid for a matching prompt.
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
        Ok(NativeKvSequence {
            pool: self.inner.clone(),
            state: Arc::new(Mutex::new(
                sequence_state(&self.inner, false).map_err(to_napi_err)?,
            )),
            run_lock: Arc::new(Mutex::new(())),
            released: Arc::new(AtomicBool::new(false)),
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

// Blocks return to the pool when the sequence is collected — GC alone is
// sufficient for lifecycle; `release()` only returns them early.
impl ObjectFinalize for NativeKvSequence {
    fn finalize(self, _env: Env) -> Result<()> {
        self.return_blocks();
        Ok(())
    }
}

impl NativeKvSequence {
    // A fresh empty sequence on this sequence's pool (used for internal
    // pad slots in ragged batched runs).
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

    // Claims the longest resident prefix of the prompt from the pool's
    // prefix cache and returns its token length; the caller prefills
    // only the remaining suffix. Only whole blocks match (a partial
    // tail block's content is not final), and the block holding the
    // last prompt token is always computed — its logits are prefill's
    // result. Hybrid pools (KV blocks plus recurrent state) match only
    // boundaries with a published recurrent snapshot and restore that
    // state into the sequence. Sharing is content-addressed: two
    // prompts that merely begin alike share; nothing about the match
    // is visible to callers.
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
}

fn validate_execution_mode(
    stateful: bool,
    scalar_count: usize,
    sequence_count: Option<usize>,
    token_count: Option<usize>,
) -> Result<()> {
    if stateful {
        if scalar_count != 0 {
            return Err(Error::new(
                Status::InvalidArg,
                "execute: stateful executables do not accept scalar inputs".to_string(),
            ));
        }
        match (sequence_count, token_count) {
            (Some(sequences), Some(tokens)) if sequences == tokens => Ok(()),
            (Some(sequences), Some(tokens)) => Err(Error::new(
                Status::InvalidArg,
                format!(
                    "execute: expected one token list per sequence, got {tokens} for {sequences} sequences"
                ),
            )),
            _ => Err(Error::new(
                Status::InvalidArg,
                "execute: stateful executables require sequence and token arrays".to_string(),
            )),
        }
    } else if sequence_count.is_some() || token_count.is_some() {
        Err(Error::new(
            Status::InvalidArg,
            "execute: stateless executables do not accept state".to_string(),
        ))
    } else {
        Ok(())
    }
}

fn validate_stateful_tensor_input(
    input: &NativeTensor,
    slot: usize,
    declared: &ProgramSlot,
    active_batch: usize,
    compiled_batch: usize,
    bounded_batch: bool,
) -> Result<()> {
    let got = input.val_cloned()?;
    let shape = got.shape();
    let shape_matches = if bounded_batch {
        declared.shape.first() == Some(&compiled_batch)
            && shape.first() == Some(&active_batch)
            && shape.len() == declared.shape.len()
            && shape[1..] == declared.shape[1..]
    } else {
        shape == declared.shape
    };
    if !shape_matches
        || got.dtype() != declared.dtype
        || device_key(&got.device()) != device_key(&declared.device)
    {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "input slot {slot}: expected bounded {}, got {:?}:{}@{}",
                declared.signature(),
                shape,
                got.dtype().name(),
                device_key(&got.device())
            ),
        ));
    }
    Ok(())
}

#[napi]
pub struct Executable {
    inner: ProgramInner,
    state: Option<StatefulExecutable>,
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
        tokens: Option<Vec<Vec<u32>>>,
        token: Option<&CancellationToken>,
    ) -> Result<Vec<NativeTensor>> {
        validate_execution_mode(
            self.state.is_some(),
            scalars.len(),
            sequences.as_ref().map(Vec::len),
            tokens.as_ref().map(Vec::len),
        )?;
        if self.state.is_none() {
            return self.execute_stateless(inputs, scalars, token).await;
        }

        self.execute_stateful(
            inputs,
            sequences.expect("state invocation was validated"),
            tokens.expect("state invocation was validated"),
            token,
        )
        .await
    }
}

impl Executable {
    // Ragged stateful batches pad to the executable's fixed width with
    // throwaway sequences and zero token rows.
    async fn execute_stateful(
        &self,
        inputs: Vec<&NativeTensor>,
        seqs: Vec<&NativeKvSequence>,
        tokens: Vec<Vec<u32>>,
        token: Option<&CancellationToken>,
    ) -> Result<Vec<NativeTensor>> {
        let batch = self.state.as_ref().expect("state checked").schema.batch;
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
        let active_batch = seqs.len();
        let pad: Vec<NativeKvSequence> = (active_batch..batch)
            .map(|_| seqs[0].new_sequence_like())
            .collect();
        let mut all: Vec<&NativeKvSequence> = seqs;
        all.extend(pad.iter());
        let mut all_tokens = tokens;
        let advance = all_tokens.first().map(|t| t.len()).unwrap_or(1);
        all_tokens.extend(std::iter::repeat_n(vec![0u32; advance], pad.len()));
        let out = self
            .execute_stateful_inner(inputs, all, all_tokens, active_batch, token)
            .await;
        for p in &pad {
            p.release();
        }
        out
    }
}

impl Executable {
    async fn execute_stateful_inner(
        &self,
        inputs: Vec<&NativeTensor>,
        seqs: Vec<&NativeKvSequence>,
        tokens: Vec<Vec<u32>>,
        active_batch: usize,
        token: Option<&CancellationToken>,
    ) -> Result<Vec<NativeTensor>> {
        let batch = seqs.len();
        if tokens.len() != batch || tokens.iter().any(|t| t.is_empty()) {
            return Err(Error::new(
                Status::InvalidArg,
                "kv run: expected one non-empty token list per sequence".to_string(),
            ));
        }
        let advance = tokens[0].len();
        if tokens.iter().any(|t| t.len() != advance) {
            return Err(Error::new(
                Status::InvalidArg,
                "kv run: batched runs advance every sequence by the same count".to_string(),
            ));
        }
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
        let bounded_slot = inner
            .slots
            .iter()
            .enumerate()
            .rev()
            .find(|(slot, declared)| !declared.scalar && *slot as u32 != stateful.cursor_slot)
            .map(|(slot, _)| slot);
        for (slot, declared) in inner.slots.iter().enumerate() {
            if declared.scalar || (stateful.cursor_tensor && slot as u32 == stateful.cursor_slot) {
                continue;
            }
            let input_index = slot
                - inner.slots.iter().take(slot).filter(|s| s.scalar).count()
                - usize::from(stateful.cursor_tensor && (slot as u32) > stateful.cursor_slot);
            validate_stateful_tensor_input(
                inputs[input_index],
                slot,
                declared,
                active_batch,
                schema.batch,
                bounded_slot == Some(slot),
            )?;
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
            active_batch,
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
        run_compute(token, move |cancelled, cancellation| {
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
            for (index, state) in slot_states.iter().take(active_batch).enumerate() {
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
                    .advance = if index < active_batch {
                    tokens[index].len()
                } else {
                    0
                };
            }
            let outputs = {
                let commit = || cancellation.complete();
                match executable::execute_stateful(
                    &executable,
                    &inputs,
                    &generated,
                    cancelled,
                    kv.as_ref(),
                    &commit,
                ) {
                    Ok(outputs) => outputs.into_iter().map(NativeTensor::wrap).collect(),
                    Err(error) => {
                        rollback()?;
                        return Err(to_napi_err(error));
                    }
                }
            };
            Ok(outputs)
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
            if got.shape() != declared.shape.as_slice()
                || got.dtype() != declared.dtype
                || device_key(&got.device()) != device_key(&declared.device)
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
                        device_key(&got.device())
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
        run_compute(token, move |cancelled, _state| {
            Ok(executable::execute_with_scalars(
                &executable,
                &inputs,
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
    let mut nodes: Vec<Arc<Node>> = roots.iter().map(|tensor| tensor.node.clone()).collect();
    if nodes.is_empty() {
        return Err(Error::new(
            Status::InvalidArg,
            "compile: expected at least one root".to_string(),
        ));
    }

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
        let (rewritten, geometry) = specialize_decode(
            &nodes,
            requested_window,
            state.batch as usize,
            state.last_token_row.unwrap_or(false),
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
        .map(|key| format!("{key}|{:?}", program.options));
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
    let tensors = tensors
        .iter()
        .map(|tensor| tensor.val_cloned())
        .collect::<Result<Vec<_>>>()?;
    run_compute(token, move |cancelled, _state| {
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
    run_compute(token, move |cancelled, _state| {
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

#[cfg(test)]
mod epilogue_tests {
    use super::*;
    use runtime::metal::device::MetalDevice;
    use runtime::metal::run::MetalTensor;

    #[test]
    fn stateless_invocation_rejects_state() {
        assert!(validate_execution_mode(false, 0, Some(1), Some(1)).is_err());
    }

    #[test]
    fn stateful_invocation_rejects_scalars_and_mismatched_state() {
        assert!(validate_execution_mode(true, 1, Some(1), Some(1)).is_err());
        assert!(validate_execution_mode(true, 0, Some(2), Some(1)).is_err());
        assert!(validate_execution_mode(true, 0, Some(1), None).is_err());
    }

    fn mleaf(data: Vec<f32>, shape: Vec<usize>) -> Arc<Node> {
        let t = MetalTensor::from_f32(MetalDevice::get(), data, shape);
        Node::new(NodeKind::Leaf(std::sync::Arc::new(LeafSlot::new(
            value::Value(t),
        ))))
        .unwrap()
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
                device: Device::Metal,
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
        let expected = unsafe {
            std::slice::from_raw_parts(readback.data.cast::<f32>(), readback.byte_len / 4).to_vec()
        };

        assert!(tensor.slot.clear());
        EXTERNAL_MEMORY_BYTES.fetch_sub(tensor.bytes, Ordering::Relaxed);
        tensor.bytes = 0;
        for _ in 0..4 {
            drop(run());
        }
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
            active_batch: 1,
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
            active_batch: 1,
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
    fn inactive_padding_slots_do_not_consume_kv_blocks() {
        let pool = NativeKvPool::new(1, 1, 2, 8, Some(4), Some(NativeDType::F32), None).unwrap();
        let slots = (0..8)
            .map(|index| {
                let mut state = sequence_state(&pool.inner, index >= 2).unwrap();
                state.advance = usize::from(index < 2);
                Arc::new(Mutex::new(state))
            })
            .collect::<Vec<_>>();
        let schema = KvStateSchema {
            max_tokens: 8,
            block_size: 4,
            kv_dtype: DType::F32,
            window: None,
            batch: 8,
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
            tokens: vec![vec![1]; 8],
            active_batch: 2,
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
        };
        {
            let _guard = MetalDevice::get().begin_executable_dispatch().unwrap();
            context.prepare_kv_attention(0, &plan, &staging).unwrap();
        }
        assert_eq!(pool.free_blocks(), 0);
        assert!(slots[..2]
            .iter()
            .all(|state| state.lock().unwrap().blocks.len() == 1));
        assert!(slots[2..]
            .iter()
            .all(|state| state.lock().unwrap().blocks.is_empty()));
        assert_eq!(&staging[1].to_u32_vec().unwrap()[..2], &[1, 1]);
        assert!(staging[1].to_u32_vec().unwrap()[2..]
            .iter()
            .all(|value| *value == 0));
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
                last_token_row: None,
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
            active_batch: 1,
        };
        let output = executable::execute_stateful(
            &program.inner.executable,
            &[],
            &program.inner.generated_bindings,
            &effect_torch_runtime::CancellationFlag::new(),
            &context,
            &|| true,
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
            device: get_device(),
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
                last_token_row: None,
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
            active_batch: 1,
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
        )
        .unwrap()[0]
            .to_f32_vec()
            .unwrap();

        assert_close(&output[..12], &reference[..12], 1e-6, "projected prefill");
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
                last_token_row: None,
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
            active_batch: 1,
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
            active_batch: 1,
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
        for (index, state) in states.iter_mut().take(context.active_batch).enumerate() {
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

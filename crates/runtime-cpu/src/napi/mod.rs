mod err;
mod gguf;
mod safetensors;
mod value;

use self::err::to_napi_err;
use self::value::Value;
use crate::{composed, executable, pool, CpuBuffer, CpuDestination, Elem, Tensor};
use effect_torch_compiler::{
    specialize_decode, CompileOptions, DecodeGeometry, InferenceOptions, PreparedProgram,
    ProgramRequest, ProgramSlot, StateCursorSlot,
};
use effect_torch_graph::CrossEntropyReduction as CeReduction;
use effect_torch_graph::{AttentionWindow, Device, PositionOffset, RotaryLayout};
use effect_torch_napi::{try_register_export, unregister_export, vec_to_bytes, CancellationState};
use effect_torch_runtime::{Buffer, CancellationFlag, DType, GgmlKQuant, Layout};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

pub type LeafSlot = effect_torch_graph::LeafSlot;
pub(crate) type Node = effect_torch_graph::Node;
type NodeKind = effect_torch_graph::NodeKind;
type KdaGeometry = effect_torch_compiler::KdaGeometry;
type ConvGeometry = effect_torch_compiler::ConvGeometry;

fn cpu_device() -> Device {
    Device::Cpu
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

enum FinalizeHint {
    ZeroCopy {
        value: Value,
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
        FinalizeHint::ZeroCopy { value, addr } => {
            drop(value);
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

#[napi]
pub fn is_available() -> bool {
    true
}

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

#[napi]
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
    active_batch: usize,
    compiled_batch: usize,
    bounded_batch: bool,
) -> Result<()> {
    let got = input.value_cloned()?;
    let shape_matches = if bounded_batch {
        declared.shape.first() == Some(&compiled_batch)
            && got.shape().first() == Some(&active_batch)
            && got.shape().len() == declared.shape.len()
            && got.shape()[1..] == declared.shape[1..]
    } else {
        got.shape() == declared.shape
    };
    if !shape_matches || got.dtype() != declared.dtype || !declared.device.is_cpu() {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "input slot {slot}: expected bounded {}, got {:?}:{}@cpu",
                declared.signature(),
                got.shape(),
                got.dtype().name(),
            ),
        ));
    }
    Ok(())
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
            let last_token_row = native.last_token_row.unwrap_or(false);
            let (rewritten, geometry) = specialize_decode(&roots, window, batch, last_token_row)
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
            batch: state.batch,
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
}

struct SeqState {
    blocks: Vec<u32>,
    head: usize,
    cursor: usize,
    advance: usize,
    last_hash: u64,
    pending: Vec<u32>,
    // Per-layer recurrent state is allocated when the sequence is created:
    // [H, Dk, Dv] f32 per KDA layer and [K-1, C] f32 per short-conv layer.
    kda_states: Vec<Tensor>,
    conv_states: Vec<Tensor>,
}

impl SeqState {
    fn note_tokens(&mut self, pool: &PoolInner, tokens: &[u32]) {
        for (index, &token) in tokens.iter().enumerate() {
            self.pending.push(token);
            if self.pending.len() == pool.block_size {
                let hash = chain_hash(self.last_hash, &self.pending);
                self.last_hash = hash;
                self.pending.clear();
                let block_index = (self.cursor + index) / pool.block_size;
                if let Some(&block) = self.blocks.get(block_index) {
                    pool.set_hash(block, hash);
                }
            }
        }
    }
}

struct KvContext {
    pool: Arc<PoolInner>,
    slots: Vec<Arc<Mutex<SeqState>>>,
    active_batch: usize,
    window: Option<usize>,
    kda: KdaGeometry,
    conv: ConvGeometry,
    transaction: Mutex<Option<CpuStateTransaction>>,
}

struct CpuStateTransaction {
    frontiers: Vec<usize>,
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
        for slot in &self.slots {
            let mut state = slot
                .lock()
                .map_err(|error| format!("decode state lock poisoned: {error}"))?;
            let additional = self
                .pool
                .max_tokens
                .div_ceil(self.pool.block_size)
                .saturating_sub(state.blocks.len());
            state.blocks.reserve(additional);
            frontiers.push(state.blocks.len());
            advances.push(state.advance);
            cursors.push(state.cursor);
        }
        if let Some(cursor) = executable.state_cursor {
            let value = values
                .get(cursor.index())
                .ok_or_else(|| "decode cursor staging value is unresolved".to_string())?;
            if value.dtype() != DType::I64 {
                return Err("decode cursor staging must use i64".to_string());
            }
            let mut destination = unsafe { CpuDestination::from_planned(value.tensor()) };
            destination.write::<i64, _>("decode cursor", &value.shape(), |output| {
                if output.len() == 1 && cursors.len() == 1 {
                    output[0] = cursors[0] as i64;
                } else {
                    assert_eq!(output.len(), cursors.len());
                    for (output, cursor) in output.iter_mut().zip(&cursors) {
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
            if advances
                .iter()
                .take(self.active_batch)
                .any(|advance| *advance == 0 || *advance > steps)
            {
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
                cursors,
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
            executable::CpuOp::LastTokenRow => {
                last_token_row_into(&inputs[0], transaction.advances[0], &mut outputs[0])
            }
            executable::CpuOp::KvAttention {
                scale,
                layer,
                window,
            } => kv_attention_into(
                self,
                *layer,
                &inputs[0],
                &inputs[1],
                &inputs[2],
                *scale,
                *window,
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
            let mut states = self
                .slots
                .iter()
                .map(|slot| {
                    slot.lock()
                        .map_err(|error| format!("decode state lock poisoned: {error}"))
                })
                .collect::<err::Res<Vec<_>>>()?;
            for (state, start) in states.iter_mut().zip(&transaction.eviction_starts) {
                if *start != usize::MAX {
                    kv_evict(&self.pool, state, *start);
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
            if let Ok(mut state) = slot.lock() {
                for block in state.blocks.split_off(frontier) {
                    self.pool.unref_block(block);
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
    let mut destination = unsafe { CpuDestination::from_planned(staging.tensor()) };
    destination.write::<f32, _>("conv initial state", staging.tensor().shape(), |output| {
        output.fill(0.0);
        for (batch, slot) in context.slots.iter().enumerate() {
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
    let mut states = context
        .slots
        .iter()
        .map(|slot| {
            slot.lock()
                .map_err(|error| format!("decode state lock poisoned: {error}"))
        })
        .collect::<err::Res<Vec<_>>>()?;

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
                for (state, update) in states.iter_mut().zip(updates) {
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
                for (state, update) in states.iter_mut().zip(updates) {
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
    if batch != context.slots.len() {
        return Err(format!(
            "kv attention: batch {batch} does not match {} kv slots",
            context.slots.len()
        ));
    }
    let layer_index = layer as usize;
    output.write::<f32, _>("kv attention output", &dimensions, |out| -> err::Res<()> {
        out.fill(0.0);
        for (batch_index, slot) in context.slots.iter().take(context.active_batch).enumerate() {
            let mut state = slot
                .lock()
                .map_err(|error| format!("kv attention: sequence lock poisoned: {error}"))?;
            let (cursor, needed, start) = kv_prepare(
                &context.pool,
                &mut state,
                layer_index,
                context.window,
                kv_heads,
                width,
                tokens,
            )?;
            let advance = state.advance;
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
                            for token in 0..advance {
                                let row = physical(cursor + token) as usize;
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
                        for token in 0..advance {
                            let row = physical(cursor + token) as usize;
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
                    for query in 0..tokens {
                        let end = (cursor + query + 1).min(needed);
                        let begin =
                            window.map_or(start, |window| end.saturating_sub(window).max(start));
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
            eviction_starts[batch_index] = if eviction_starts[batch_index] == usize::MAX {
                start
            } else {
                eviction_starts[batch_index].max(start)
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
    let start = window.map_or(0, |window| needed.saturating_sub(window));
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

#[napi(custom_finalize)]
pub struct NativeKvSequence {
    pool: Arc<PoolInner>,
    state: Arc<Mutex<SeqState>>,
    run_lock: Arc<Mutex<()>>,
    released: Arc<AtomicBool>,
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
        }
    }

    fn new_sequence_like(&self) -> Self {
        Self::new(self.pool.clone())
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
        self.return_blocks();
        Ok(())
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
    token_count: Option<usize>,
) -> Result<()> {
    if stateful {
        if scalar_count != 0 {
            return Err(Error::new(
                Status::InvalidArg,
                "execute: stateful executables do not accept scalar inputs",
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
                "execute: stateful executables require sequence and token arrays",
            )),
        }
    } else if sequence_count.is_some() || token_count.is_some() {
        Err(Error::new(
            Status::InvalidArg,
            "execute: stateless executables do not accept state",
        ))
    } else {
        Ok(())
    }
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

fn validate_active_batch(schema: &KvStateSchema, sequences: usize) -> Result<()> {
    if sequences == 0 || sequences > schema.batch {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "execute: executable accepts 1..={} sequences, got {sequences}",
                schema.batch
            ),
        ));
    }
    Ok(())
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
        let Some(schema) = self.state else {
            return self.execute_stateless(inputs, scalars, token).await;
        };
        let mut sequences = sequences.expect("state invocation was validated");
        let mut tokens = tokens.expect("state invocation was validated");
        let active_batch = sequences.len();
        validate_active_batch(&schema, active_batch)?;
        let padding = (sequences.len()..schema.batch)
            .map(|_| sequences[0].new_sequence_like())
            .collect::<Vec<_>>();
        sequences.extend(padding.iter());
        let advance = tokens.first().map(Vec::len).unwrap_or(1);
        tokens.extend(std::iter::repeat(vec![0; advance]).take(padding.len()));
        let output = self
            .execute_stateful(inputs, sequences, tokens, active_batch, token)
            .await;
        for sequence in &padding {
            sequence.release();
        }
        output
    }
}

impl Executable {
    async fn execute_stateful(
        &self,
        inputs: Vec<&NativeTensor>,
        sequences: Vec<&NativeKvSequence>,
        tokens: Vec<Vec<u32>>,
        active_batch: usize,
        token: Option<&CancellationToken>,
    ) -> Result<Vec<NativeTensor>> {
        let schema = self.state.expect("stateful execution was validated");
        let batch = sequences.len();
        if tokens.len() != batch || tokens.iter().any(Vec::is_empty) {
            return Err(Error::new(
                Status::InvalidArg,
                "kv run: expected one non-empty token list per sequence",
            ));
        }
        let advance = tokens[0].len();
        if tokens.iter().any(|row| row.len() != advance) {
            return Err(Error::new(
                Status::InvalidArg,
                "kv run: batched runs advance every sequence by the same count",
            ));
        }
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
        let bounded_slot = self
            .inner
            .slots
            .iter()
            .enumerate()
            .rev()
            .find(|(slot, declared)| !declared.scalar && *slot as u32 != schema.cursor_slot)
            .map(|(slot, _)| slot);
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
            validate_stateful_tensor_input(
                inputs[input_index],
                slot,
                declared,
                active_batch,
                schema.batch,
                bounded_slot == Some(slot),
            )?;
        }
        let executable = self.inner.executable.clone();
        let generated = self.inner.generated_bindings.clone();
        let inputs = inputs
            .iter()
            .map(|input| input.value_cloned())
            .collect::<Result<Vec<_>>>()?;
        let context = Arc::new(KvContext {
            pool: sequences[0].pool.clone(),
            slots: sequences
                .iter()
                .map(|sequence| sequence.state.clone())
                .collect(),
            active_batch,
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
        run_compute(token, move |cancelled, cancellation| {
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
            for (index, state) in states.iter().take(active_batch).enumerate() {
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
                    .advance = if index < active_batch {
                    tokens[index].len()
                } else {
                    0
                };
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
            let outputs = match executable::execute_stateful(
                &executable,
                &inputs,
                &generated,
                cancelled,
                &context,
                &|| cancellation.complete(),
            ) {
                Ok(outputs) => outputs.into_iter().map(NativeTensor::wrap).collect(),
                Err(error) => {
                    rollback();
                    return Err(to_napi_err(error));
                }
            };
            for (index, state) in states.iter().take(active_batch).enumerate() {
                if let Ok(mut state) = state.lock() {
                    state.note_tokens(&context.pool, &tokens[index]);
                    state.cursor += state.advance;
                    state.advance = 0;
                    context.pool.maybe_publish_recurrent_snapshot(&state);
                }
            }
            Ok(outputs)
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(tensor: Tensor) -> Arc<Node> {
        Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(Value(tensor))))).unwrap()
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
        assert!(validate_execution_mode(false, 1, None, None).is_ok());
        assert!(validate_execution_mode(false, 0, Some(1), Some(1)).is_err());
        assert!(validate_execution_mode(true, 1, Some(1), Some(1)).is_err());
        assert!(validate_execution_mode(true, 0, None, None).is_err());
        assert!(validate_execution_mode(true, 0, Some(2), Some(1)).is_err());
        assert!(validate_execution_mode(true, 0, Some(2), Some(2)).is_ok());
        let schema = test_state_schema();
        assert!(validate_active_batch(&schema, 0).is_err());
        assert!(validate_active_batch(&schema, 1).is_ok());
        assert!(validate_active_batch(&schema, 2).is_ok());
        assert!(validate_active_batch(&schema, 3).is_err());
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
            slots: vec![state.clone()],
            active_batch: 1,
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
                &mut destination,
                &mut eviction_starts,
            )
            .unwrap();
        }
        drop(destination);
        let actual = Value(actual).to_f32_vec().unwrap();
        assert_close(&actual, &expected, 1e-6, "kv attention");
        assert_eq!(state.lock().unwrap().advance, 3);
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
            slots: vec![state.clone()],
            active_batch: 1,
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
            slots: vec![Arc::new(Mutex::new(SeqState {
                blocks: Vec::new(),
                head: 0,
                cursor: 0,
                advance,
                last_hash: HASH_SEED,
                pending: Vec::new(),
                kda_states: Vec::new(),
                conv_states: Vec::new(),
            }))],
            active_batch: 1,
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
            last_token_row: Some(true),
        };
        let executable = compile(vec![&root], None, Some(schema), None).unwrap();
        let outputs = &executable.inner.executable.signature.outputs;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape, vec![8]);
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
            slots: vec![state.clone()],
            active_batch: 1,
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
}

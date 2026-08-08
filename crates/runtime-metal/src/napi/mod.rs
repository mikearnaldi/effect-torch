mod err;
mod fusion;
mod runtime;
mod safetensors;
mod value;

#[cfg(test)]
use effect_torch_compiler::gemm_epilogue_pass;
use effect_torch_compiler::{collect_program_slots, fuse_roots, ProgramSlot};
use effect_torch_graph::CrossEntropyReduction as CeReduction;
use effect_torch_graph::Device;
use effect_torch_graph::{node_children, remap_children, PositionOffset};
use effect_torch_napi::{try_register_export, unregister_export, vec_to_bytes, CancellationState};
use runtime::dtype::DType;
pub type LeafSlot = effect_torch_graph::LeafSlot;
pub(crate) type Node = effect_torch_graph::Node<effect_torch_compiler::Expr>;
type NodeKind = effect_torch_graph::NodeKind<effect_torch_compiler::Expr>;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use err::to_napi_err;

use runtime::metal::ops as metal_ops;

use runtime::metal::{
    composed, device, flash, kda, kernels, layer_norm, linear, loss, paged, rotary,
};

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
    // f16/bf16 read back as f32: JS has no half typed arrays we can
    // rely on, and the conversion keeps the destructor surface small.
    let flat = if matches!(
        inner.dtype(),
        runtime::dtype::DType::F16 | runtime::dtype::DType::BF16
    ) {
        match inner {
            value::Value(t) => value::Value(
                runtime::metal::kernels::cast(
                    runtime::metal::device::MetalDevice::get(),
                    t,
                    runtime::dtype::DType::F32,
                )
                .map_err(to_napi_err)?,
            ),
        }
    } else {
        inner.clone()
    };
    // Contiguous layout for the flat read; Metal reads synchronize first.
    let (base, offset, keep, elem_count) = match &flat {
        value::Value(t) => {
            let t = if t.layout.is_contiguous() {
                t.clone()
            } else {
                value::Value(
                    runtime::metal::kernels::strided_copy(
                        runtime::metal::device::MetalDevice::get(),
                        t,
                    )
                    .map_err(to_napi_err)?,
                )
                .as_metal()
                .map_err(to_napi_err)?
                .clone()
            };
            runtime::metal::device::MetalDevice::get()
                .synchronize()
                .map_err(to_napi_err)?;
            let base = t.buffer.contents_ptr() as *const u8;
            let offset = t.layout.offset() * t.dtype.size_in_bytes();
            let keep = value::Value(t);
            (base, offset, keep, 0usize)
        }
    };
    let _ = elem_count;
    let elem_size = flat.dtype().size_in_bytes();
    let count = flat.numel();
    let byte_len = count * elem_size;
    if !base.is_null() {
        let addr = base as usize + offset;
        if try_register_export(addr) {
            return Ok(Readback {
                data: addr as *mut u8,
                byte_len,
                hint: Some(FinalizeHint::ZeroCopy { tensor: keep, addr }),
            });
        }
    }
    let (_, ptr, len, cap) = match flat.dtype() {
        runtime::dtype::DType::F32 => vec_to_bytes(flat.to_f32_vec().map_err(to_napi_err)?),
        runtime::dtype::DType::F64 => vec_to_bytes(flat.to_f64_vec().map_err(to_napi_err)?),
        runtime::dtype::DType::I64 => vec_to_bytes(flat.to_i64_vec().map_err(to_napi_err)?),
        runtime::dtype::DType::U8 => vec_to_bytes(flat.to_u8_vec().map_err(to_napi_err)?),
        runtime::dtype::DType::U32 => vec_to_bytes(flat.to_u32_vec().map_err(to_napi_err)?),
        dtype => {
            return Err(Error::new(
                Status::InvalidArg,
                format!("readback not implemented for dtype: {}", dtype.name()),
            ))
        }
    };
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
    let (k_dim, vocab) = (weight.shape[0], weight.shape[1]);
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
    let device = logits.device.clone();
    let x2 = if rank == 2 {
        x.clone()
    } else {
        Node::new(NodeKind::Reshape {
            a: x.clone(),
            shape: vec![rows, k_dim],
        })?
    };
    let t1 = if target.shape.as_slice() == [rows] {
        target.clone()
    } else {
        Node::new(NodeKind::Reshape {
            a: target.clone(),
            shape: vec![rows],
        })?
    };
    // Exact active count in f32: rows - #{t == ignore_index}. Counts are
    // integers far below 2^24, so f32 is exact. A u32 target can never
    // hold a negative (or huge) ignore_index, matching ce_ignored_mask.
    let ignored_count =
        if target.dtype == DType::U32 && (ignore_index < 0 || ignore_index > u32::MAX as i64) {
            cached_constant(0.0, DType::F32, device.clone())?
        } else {
            let ignore = cached_constant(ignore_index as f64, target.dtype, device.clone())?;
            let ignored = Node::new(NodeKind::Eq {
                a: t1.clone(),
                b: ignore,
            })?;
            let ignored = Node::new(NodeKind::Cast {
                a: ignored,
                dtype: DType::F32,
            })?;
            Node::new(NodeKind::Sum {
                a: ignored,
                dims: vec![0],
                keepdims: false,
            })?
        };
    let rows_f32 = cached_constant(rows as f64, DType::F32, device)?;
    let active = Node::new(NodeKind::Sub {
        a: rows_f32,
        b: ignored_count,
    })?;
    let chunk_len = rows.div_ceil(chunks);
    let mut total: Option<Arc<Node>> = None;
    let mut off = 0;
    while off < rows {
        let end = (off + chunk_len).min(rows);
        let xk = Node::new(NodeKind::Slice {
            a: x2.clone(),
            ranges: vec![(off, end, 1), (0, k_dim, 1)],
        })?;
        let tk = Node::new(NodeKind::Slice {
            a: t1.clone(),
            ranges: vec![(off, end, 1)],
        })?;
        let lk = Node::new(NodeKind::Linear {
            x: xk,
            weight: weight.clone(),
            bias: bias.clone(),
        })?;
        let cek = Node::new(NodeKind::CrossEntropy {
            logits: lk,
            target: tk,
            ignore_index,
            reduction: CeReduction::Sum,
        })?;
        // The checkpoint makes backward recompute this chunk's logits
        // (one extra head gemm per chunk) instead of retaining every
        // chunk's logits from forward to backward.
        let ck = Node::new(NodeKind::Checkpoint { a: cek })?;
        let ck32 = Node::new(NodeKind::Cast {
            a: ck,
            dtype: DType::F32,
        })?;
        total = Some(match total {
            None => ck32,
            Some(t) => Node::new(NodeKind::Add { a: t, b: ck32 })?,
        });
        off = end;
    }
    let total = total.expect("at least one chunk");
    let mean = Node::new(NodeKind::Div {
        a: total,
        b: active,
    })?;
    if mean.dtype == logits.dtype {
        Ok(mean)
    } else {
        Node::new(NodeKind::Cast {
            a: mean,
            dtype: logits.dtype,
        })
    }
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
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Sdpa {
            q: self.node.clone(),
            k: k.node.clone(),
            v: v.node.clone(),
            scale,
            causal,
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
    pub fn rotary_embedding(&self, seq_len: u32, theta: f64) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::RotaryEmbedding {
            x: self.node.clone(),
            seq_len: seq_len as usize,
            theta,
            offset: PositionOffset::Absolute,
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
    pub fn linear(&self, weight: &LazyTensor, bias: &LazyTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Linear {
            x: self.node.clone(),
            weight: weight.node.clone(),
            bias: bias.node.clone(),
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

// Evaluation holds intermediate results in a cache keyed by node id. A node
// is freed from the cache as soon as its last in-graph consumer has been
// evaluated (roots are pinned: they are returned to the caller), so peak
// memory tracks the maximum live range rather than the whole graph. Dropping
// a cache entry only releases the evaluator's reference: `Leaf` tensors
// owned by JS handles stay alive through candle's refcounting.
struct Evaluator {
    cache: std::collections::HashMap<u64, value::Value>,
    // AdamW step id -> (next m, next v); the step node's own value is the
    // updated parameter, stored in the regular cache
    adamw: std::collections::HashMap<u64, [value::Value; 2]>,
    sgd: std::collections::HashMap<u64, value::Value>,
    // FusedElementwiseMulti id -> all outputs; the node's own cache entry
    // holds output 0 so the evaluator's single-value invariant holds
    multi: std::collections::HashMap<u64, Vec<value::Value>>,
    // LayerNormBackward id -> (dw, db); the node's own cache entry is dx.
    ln: std::collections::HashMap<u64, [value::Value; 2]>,
    // Optimizer step scalars (lr, bias corrections) cast to a given
    // dtype/device, memoized per walk: identical for every parameter,
    // and the naive path copies each scalar per parameter per step.
    step_scalars: std::collections::HashMap<(u64, DType, u8), value::Value>,
    // Device-packed fusion scalar buffers (cat of the 0-d step scalars),
    // memoized per walk: every AdamW group in a step packs the same triple.
    scalar_packs: std::collections::HashMap<(u64, u64, u64), value::Value>,
    consumers: std::collections::HashMap<u64, usize>,
    roots: HashSet<u64>,
    // RFC 0008: Input/ScalarInput node id -> argument buffer, populated by
    // CompiledProgram::run. Empty for ordinary eval_lazy walks.
    slots: std::collections::HashMap<u64, value::Value>,
    // RFC 0010: the pool + sequence a kv program runs against. None for
    // ordinary and non-kv compiled walks; KvAttention nodes error without
    // it.
    kv: Option<Arc<KvContext>>,
    // Deferred cross-entropy status checks (fused CE): (buffer, kind,
    // classes). Reading the status requires a device sync, which would
    // split the walk's encode/execute pipeline mid-graph — so the fused
    // kernels record their status here and the walk validates them after
    // its final synchronize, preserving the exact error semantics.
    ce_checks: Vec<(value::Value, CeCheck, usize)>,
}

// Which deferred status check a fused cross-entropy call needs. Sum
// reductions never divide by the active count, so their forward only
// validates labels and their backward needs no check at all.
#[derive(Clone, Copy, PartialEq, Eq)]

enum CeCheck {
    ForwardMean,
    ForwardSum,
    BackwardMean,
}

impl Evaluator {
    fn new(roots: &[Arc<Node>]) -> Self {
        Self::with_slots(roots, std::collections::HashMap::new())
    }

    fn with_slots(
        roots: &[Arc<Node>],
        slots: std::collections::HashMap<u64, value::Value>,
    ) -> Self {
        Self::with_kv(roots, slots, None)
    }

    fn with_kv(
        roots: &[Arc<Node>],
        slots: std::collections::HashMap<u64, value::Value>,
        kv: Option<Arc<KvContext>>,
    ) -> Self {
        let mut consumers = std::collections::HashMap::new();
        // One visited set across all roots: a node shared between roots
        // is visited once, so each parent edge is counted exactly once —
        // matching the single decrement it gets when that parent is
        // evaluated. Counting per root would multiply the counts of
        // shared nodes by the number of roots reaching them, and their
        // buffers would never be released mid-walk.
        let mut visited = HashSet::new();
        for root in roots {
            count_consumers(root, &mut consumers, &mut visited);
        }
        Self {
            cache: std::collections::HashMap::new(),
            adamw: std::collections::HashMap::new(),
            sgd: std::collections::HashMap::new(),
            multi: std::collections::HashMap::new(),
            ln: std::collections::HashMap::new(),
            step_scalars: std::collections::HashMap::new(),

            scalar_packs: std::collections::HashMap::new(),
            consumers,
            roots: roots.iter().map(|root| root.id).collect(),
            slots,
            kv,

            ce_checks: Vec::new(),
        }
    }

    // RFC 0016: buffer identities of every Metal tensor the walk still
    // references — the arena capture's liveness set at a node boundary.

    fn live_buffer_keys(&self) -> HashSet<usize> {
        let mut out = HashSet::new();
        let mut add = |v: &value::Value| {
            if let Some(key) = v.buffer_key() {
                out.insert(key);
            }
        };
        for v in self.cache.values() {
            add(v);
        }
        for pair in self.adamw.values() {
            add(&pair[0]);
            add(&pair[1]);
        }
        for v in self.sgd.values() {
            add(v);
        }
        for vs in self.multi.values() {
            for v in vs {
                add(v);
            }
        }
        for pair in self.ln.values() {
            add(&pair[0]);
            add(&pair[1]);
        }
        for v in self.step_scalars.values() {
            add(v);
        }
        for v in self.scalar_packs.values() {
            add(v);
        }
        for v in self.slots.values() {
            add(v);
        }
        for (v, _, _) in &self.ce_checks {
            add(v);
        }
        out
    }

    // Runs the deferred fused-CE status checks after the walk's final
    // synchronize: forward statuses are [loss, active, invalid],
    // backward counts are [active]. Errors are exactly the composed
    // path's, raised from the same eval call.

    fn run_ce_checks(&self) -> err::Res<()> {
        for (buffer, kind, classes) in &self.ce_checks {
            let values = buffer.to_f32_vec()?;
            match kind {
                CeCheck::ForwardMean | CeCheck::ForwardSum => {
                    let (active, invalid) = (values[1] as usize, values[2] as usize);
                    if *kind == CeCheck::ForwardMean && active == 0 {
                        return Err(
                            "cross_entropy: no active targets (all positions are ignored)"
                                .to_string(),
                        );
                    }
                    if invalid > 0 {
                        return Err(format!(
                            "cross_entropy: target out of range [0, {classes}) at an active position"
                        ));
                    }
                }
                CeCheck::BackwardMean => {
                    if values[0] == 0.0 {
                        return Err(
                            "cross_entropy: no active targets (all positions are ignored)"
                                .to_string(),
                        );
                    }
                }
            }
        }
        Ok(())
    }

    fn value(&self, id: u64) -> err::Res<value::Value> {
        self.cache
            .get(&id)
            .cloned()
            .ok_or_else(|| format!("internal error: unevaluated node {id}"))
    }

    // A step scalar cast to the target dtype/device, memoized per walk
    // (one copy per distinct dtype instead of one per parameter).
    fn step_scalar(&mut self, id: u64, dtype: DType, _device: &Device) -> err::Res<value::Value> {
        let key = (id, dtype, 2);
        if let Some(cached) = self.step_scalars.get(&key) {
            return Ok(cached.clone());
        }
        let cast = match self.value(id)? {
            value::Value(t) => value::Value(metal_ops::cast(&t, dtype)?),
        };
        self.step_scalars.insert(key, cast.clone());
        Ok(cast)
    }

    fn release_children(&mut self, node: &Arc<Node>) {
        for child in node_children(&node.kind) {
            if let Some(count) = self.consumers.get_mut(&child.id) {
                *count -= 1;
                if *count == 0 && !self.roots.contains(&child.id) {
                    self.cache.remove(&child.id);
                    self.adamw.remove(&child.id);
                    self.sgd.remove(&child.id);
                    self.multi.remove(&child.id);
                }
            }
        }
    }
}

fn count_consumers(
    root: &Arc<Node>,
    consumers: &mut std::collections::HashMap<u64, usize>,
    visited: &mut HashSet<u64>,
) {
    let mut stack = vec![root.clone()];
    while let Some(node) = stack.pop() {
        if !visited.insert(node.id) {
            continue;
        }
        for child in node_children(&node.kind) {
            *consumers.entry(child.id).or_insert(0) += 1;
            stack.push(child);
        }
    }
}

// depth, so chains of arbitrary length evaluate on a fixed stack. Children
// are always computed before their parents, and `eval_uncached` reads their
// values straight from the cache.
// Device-side index path: indices already on Metal stay on the device
// (cast to u32 there); reading them back would synchronize the whole
// command queue mid-step — the wte gather/scatter pair was costing
// ~530ms/step of host-blocked time at batch 32 before this.

fn metal_ids_u32(indexes: &value::Value) -> err::Res<runtime::metal::run::MetalTensor> {
    match indexes {
        value::Value(t) => {
            let t = metal_ops::contiguous(t)?;
            if t.dtype == DType::U32 {
                Ok(t)
            } else {
                metal_ops::cast(&t, DType::U32)
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn adamw_native(
    p: &runtime::metal::run::MetalTensor,
    g: &runtime::metal::run::MetalTensor,
    m: &runtime::metal::run::MetalTensor,
    v: &runtime::metal::run::MetalTensor,
    lr: &runtime::metal::run::MetalTensor,
    c1: &runtime::metal::run::MetalTensor,
    c2: &runtime::metal::run::MetalTensor,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
) -> err::Res<(
    runtime::metal::run::MetalTensor,
    runtime::metal::run::MetalTensor,
    runtime::metal::run::MetalTensor,
)> {
    let b1 = metal_ops::fill(m.layout.shape(), beta1, m.dtype)?;
    let ob1 = metal_ops::fill(g.layout.shape(), 1.0 - beta1, g.dtype)?;
    let next_m = metal_ops::binary(
        &metal_ops::binary(m, &b1, metal_ops::BinOp::Mul)?,
        &metal_ops::binary(g, &ob1, metal_ops::BinOp::Mul)?,
        metal_ops::BinOp::Add,
    )?;
    let b2 = metal_ops::fill(v.layout.shape(), beta2, v.dtype)?;
    let ob2 = metal_ops::fill(g.layout.shape(), 1.0 - beta2, g.dtype)?;
    let gg = metal_ops::binary(g, g, metal_ops::BinOp::Mul)?;
    let next_v = metal_ops::binary(
        &metal_ops::binary(v, &b2, metal_ops::BinOp::Mul)?,
        &metal_ops::binary(&gg, &ob2, metal_ops::BinOp::Mul)?,
        metal_ops::BinOp::Add,
    )?;
    let m_hat = metal_ops::binary(&next_m, c1, metal_ops::BinOp::Div)?;
    let v_hat = metal_ops::binary(&next_v, c2, metal_ops::BinOp::Div)?;
    let denom = metal_ops::binary(
        &metal_ops::unary(&v_hat, metal_ops::UnOp::Sqrt)?,
        &metal_ops::fill(v_hat.layout.shape(), eps, v_hat.dtype)?,
        metal_ops::BinOp::Add,
    )?;
    let adjusted = metal_ops::binary(
        &metal_ops::binary(&m_hat, &denom, metal_ops::BinOp::Div)?,
        lr,
        metal_ops::BinOp::Mul,
    )?;
    // p * (1 - lr * weight_decay) - adjusted, factored as
    // p - p * (lr * weight_decay) - adjusted.
    let next_p = if weight_decay == 0.0 {
        metal_ops::binary(p, &adjusted, metal_ops::BinOp::Sub)?
    } else {
        let wd = metal_ops::fill(lr.layout.shape(), weight_decay, lr.dtype)?;
        let decay = metal_ops::binary(
            p,
            &metal_ops::binary(lr, &wd, metal_ops::BinOp::Mul)?,
            metal_ops::BinOp::Mul,
        )?;
        metal_ops::binary(
            &metal_ops::binary(p, &decay, metal_ops::BinOp::Sub)?,
            &adjusted,
            metal_ops::BinOp::Sub,
        )?
    };
    Ok((next_p, next_m, next_v))
}

// A 0-d float operand never promotes a float tensor's dtype: the scalar
// is cast into the tensor's dtype before dispatch (mirrors
// scalar_aware_binary_dtype), so e.g. an f32 scalar gradient scaling a
// bf16 tensor stays bf16.
fn coerce_scalar_vals(a: value::Value, b: value::Value) -> err::Res<(value::Value, value::Value)> {
    if a.dtype() == b.dtype() || !a.dtype().is_float() || !b.dtype().is_float() {
        return Ok((a, b));
    }
    let a_scalar = a.shape().is_empty();
    let b_scalar = b.shape().is_empty();
    if a_scalar == b_scalar {
        return Ok((a, b));
    }
    let cast_val = |v: &value::Value, dtype: runtime::dtype::DType| -> err::Res<value::Value> {
        Ok(match v {
            value::Value(t) => value::Value(metal_ops::cast(t, dtype)?),
        })
    };
    if a_scalar {
        let target = b.dtype();
        Ok((cast_val(&a, target)?, b))
    } else {
        let target = a.dtype();
        Ok((a, cast_val(&b, target)?))
    }
}

fn normalize_node_output(node: &Node, mut output: value::Value) -> err::Res<value::Value> {
    if output.device() != node.device {
        return Err(format!(
            "{} returned device {}, expected {}",
            node_kind_name(&node.kind),
            output.device().name(),
            node.device.name()
        ));
    }
    if output.dtype() != node.dtype {
        output = match output {
            value::Value(tensor) => value::Value(runtime::metal::kernels::cast(
                runtime::metal::device::MetalDevice::get(),
                &tensor,
                node.dtype,
            )?),
        };
    }
    if output.shape() != node.shape {
        let expected: usize = node.shape.iter().product();
        if output.numel() != expected {
            return Err(format!(
                "{} returned shape {:?}, expected {:?}",
                node_kind_name(&node.kind),
                output.shape(),
                node.shape
            ));
        }
        output = match output {
            value::Value(tensor) => {
                let contiguous = if tensor.layout.is_contiguous() && tensor.layout.offset() == 0 {
                    tensor
                } else {
                    runtime::metal::kernels::strided_copy(
                        runtime::metal::device::MetalDevice::get(),
                        &tensor,
                    )?
                };
                value::Value(runtime::metal::run::MetalTensor {
                    buffer: contiguous.buffer,
                    layout: runtime::layout::Layout::contiguous(node.shape.clone()),
                    dtype: contiguous.dtype,
                })
            }
        };
    }
    Ok(output)
}

fn eval_node(
    root: &Arc<Node>,
    cancelled: &AtomicBool,
    ev: &mut Evaluator,
) -> err::Res<value::Value> {
    let mut stack: Vec<(Arc<Node>, bool)> = vec![(root.clone(), false)];
    while let Some((node, processed)) = stack.pop() {
        if cancelled.load(Ordering::Relaxed) {
            return Err("operation aborted".to_string());
        }
        if ev.cache.contains_key(&node.id) {
            continue;
        }
        if processed {
            let kind_timing = std::env::var_os("EFFECT_TORCH_KIND_TIMING").is_some();
            let t0 = kind_timing.then(std::time::Instant::now);

            if runtime::metal::arena::capture_active() {
                runtime::metal::arena::set_current_kind(node_kind_name(&node.kind));
            }
            let output = normalize_node_output(&node, eval_uncached(&node, ev)?)?;
            if let Some(t0) = t0 {
                kind_timing_nanos(node_kind_name(&node.kind), t0.elapsed().as_nanos() as u64);
            }
            ev.cache.insert(node.id, output);
            ev.release_children(&node);

            if runtime::metal::arena::capture_active() {
                runtime::metal::arena::capture_checkpoint(&ev.live_buffer_keys());
            }
            continue;
        }
        stack.push((node.clone(), true));
        for child in node_children(&node.kind) {
            if !ev.cache.contains_key(&child.id) {
                stack.push((child, false));
            }
        }
    }
    Ok(ev
        .cache
        .get(&root.id)
        .expect("root is evaluated before its consumers")
        .clone())
}

// Each package enables exactly one backend, making cross-backend mismatch
// arms unreachable after feature selection while keeping the shared evaluator explicit.
#[allow(unreachable_patterns)]
fn eval_uncached(node: &Arc<Node>, ev: &mut Evaluator) -> err::Res<value::Value> {
    match &node.device {
        Device::Metal => {}
        device => {
            return Err(format!(
                "device {} is unsupported by this addon",
                device.name()
            ))
        }
    }
    let output = match &node.kind {
        NodeKind::Leaf(slot) => slot.get().map_err(|e| e.to_string())?,
        NodeKind::Input { slot, .. } | NodeKind::ScalarInput { slot, .. } => {
            ev.slots.get(&node.id).cloned().ok_or_else(|| {
                format!(
                    "input slot {slot} is unbound: placeholder leaves evaluate only inside a compiled program run"
                )
            })?
        }
        NodeKind::FromBytes {
            data,
            shape,
            dtype,
            device: _,
        } => safetensors::value_from_bytes(data, shape, *dtype)?,
        NodeKind::Zeros {
            shape,
            dtype,
            device,
        } => match device {


            Device::Metal => value::Value(metal_ops::fill(shape, 0.0, *dtype)?),
            _ => return Err(format!("device {} is unsupported by this addon", device.name())),
        },
        NodeKind::Ones {
            shape,
            dtype,
            device,
        } => match device {


            Device::Metal => value::Value(metal_ops::fill(shape, 1.0, *dtype)?),
            _ => return Err(format!("device {} is unsupported by this addon", device.name())),
        },
        NodeKind::Full {
            shape,
            value,
            dtype,
            device,
        } => match device {


            Device::Metal => value::Value(metal_ops::fill(shape, *value, *dtype)?),
            _ => return Err(format!("device {} is unsupported by this addon", device.name())),
        },
        NodeKind::Randn {
            shape,
            dtype,
            device,
        } => match device {


            Device::Metal => {
                static SEED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(299792458);
                let seed = SEED.fetch_add(1, Ordering::Relaxed);
                let t = metal_ops::randn(shape, seed)?;
                value::Value(metal_ops::cast(&t, *dtype)?)
            }
            _ => return Err(format!("device {} is unsupported by this addon", device.name())),
        },
        NodeKind::Uniform {
            lo,
            hi,
            shape,
            dtype,
            device,
        } => match device {


            Device::Metal => {
                static SEED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(78778899);
                let seed = SEED.fetch_add(1, Ordering::Relaxed);
                let t = metal_ops::uniform(*lo, *hi, shape, seed)?;
                value::Value(metal_ops::cast(&t, *dtype)?)
            }
            _ => return Err(format!("device {} is unsupported by this addon", device.name())),
        },
        NodeKind::Arange {
            start,
            end,
            step,
            dtype,
            device,
        } => match device {


            Device::Metal => value::Value(metal_ops::arange(*start, *end, *step, *dtype)?),
            _ => return Err(format!("device {} is unsupported by this addon", device.name())),
        },
        NodeKind::Eye { n, dtype, device } => match device {


            Device::Metal => value::Value(metal_ops::eye(*n, *dtype)?),
            _ => return Err(format!("device {} is unsupported by this addon", device.name())),
        },
        NodeKind::Add { a, b } => {
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            let (a, b) = coerce_scalar_vals(a, b)?;
            match (&a, &b) {


                (value::Value(x), value::Value(y)) => {
                    value::Value(metal_ops::binary_promote(x, y, metal_ops::BinOp::Add)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Sub { a, b } => {
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            let (a, b) = coerce_scalar_vals(a, b)?;
            match (&a, &b) {


                (value::Value(x), value::Value(y)) => {
                    value::Value(metal_ops::binary_promote(x, y, metal_ops::BinOp::Sub)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Mul { a, b } => {
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            let (a, b) = coerce_scalar_vals(a, b)?;
            match (&a, &b) {


                (value::Value(x), value::Value(y)) => {
                    value::Value(metal_ops::binary_promote(x, y, metal_ops::BinOp::Mul)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Div { a, b } => {
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            let (a, b) = coerce_scalar_vals(a, b)?;
            match (&a, &b) {


                (value::Value(x), value::Value(y)) => {
                    value::Value(metal_ops::binary_promote(x, y, metal_ops::BinOp::Div)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Eq { a, b } => {
            let (x, y) = (ev.value(a.id)?, ev.value(b.id)?);
            match (&x, &y) {


                (value::Value(a), value::Value(b)) => {
                    value::Value(metal_ops::compare(a, b, metal_ops::BinOp::Eq)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Gt { a, b } => {
            let (x, y) = (ev.value(a.id)?, ev.value(b.id)?);
            match (&x, &y) {


                (value::Value(a), value::Value(b)) => {
                    value::Value(metal_ops::compare(a, b, metal_ops::BinOp::Gt)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Lt { a, b } => {
            let (x, y) = (ev.value(a.id)?, ev.value(b.id)?);
            match (&x, &y) {


                (value::Value(a), value::Value(b)) => {
                    value::Value(metal_ops::compare(a, b, metal_ops::BinOp::Lt)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Ge { a, b } => {
            let (x, y) = (ev.value(a.id)?, ev.value(b.id)?);
            match (&x, &y) {


                (value::Value(a), value::Value(b)) => {
                    value::Value(metal_ops::compare(a, b, metal_ops::BinOp::Ge)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Le { a, b } => {
            let (x, y) = (ev.value(a.id)?, ev.value(b.id)?);
            match (&x, &y) {


                (value::Value(a), value::Value(b)) => {
                    value::Value(metal_ops::compare(a, b, metal_ops::BinOp::Le)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Maximum { a, b } => {
            let (x, y) = (ev.value(a.id)?, ev.value(b.id)?);
            let (x, y) = coerce_scalar_vals(x, y)?;
            match (&x, &y) {


                (value::Value(a), value::Value(b)) => {
                    value::Value(metal_ops::binary_promote(a, b, metal_ops::BinOp::Max)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Minimum { a, b } => {
            let (x, y) = (ev.value(a.id)?, ev.value(b.id)?);
            let (x, y) = coerce_scalar_vals(x, y)?;
            match (&x, &y) {


                (value::Value(a), value::Value(b)) => {
                    value::Value(metal_ops::binary_promote(a, b, metal_ops::BinOp::Min)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Neg { a } => {
            let x = ev.value(a.id)?;
            match &x {


                value::Value(t) => value::Value(metal_ops::unary_promote(t, metal_ops::UnOp::Neg)?),
            }
        }
        NodeKind::Abs { a } => {
            let x = ev.value(a.id)?;
            match &x {


                value::Value(t) => value::Value(metal_ops::unary_promote(t, metal_ops::UnOp::Abs)?),
            }
        }
        NodeKind::Sqrt { a } => {
            let x = ev.value(a.id)?;
            match &x {


                value::Value(t) => value::Value(metal_ops::unary_promote(t, metal_ops::UnOp::Sqrt)?),
            }
        }
        NodeKind::Exp { a } => {
            let x = ev.value(a.id)?;
            match &x {


                value::Value(t) => value::Value(metal_ops::unary_promote(t, metal_ops::UnOp::Exp)?),
            }
        }
        NodeKind::Log { a } => {
            let x = ev.value(a.id)?;
            match &x {


                value::Value(t) => value::Value(metal_ops::unary_promote(t, metal_ops::UnOp::Log)?),
            }
        }
        NodeKind::Sin { a } => {
            let x = ev.value(a.id)?;
            match &x {


                value::Value(t) => value::Value(metal_ops::unary_promote(t, metal_ops::UnOp::Sin)?),
            }
        }
        NodeKind::Cos { a } => {
            let x = ev.value(a.id)?;
            match &x {


                value::Value(t) => value::Value(metal_ops::unary_promote(t, metal_ops::UnOp::Cos)?),
            }
        }
        NodeKind::Tanh { a } => {
            let x = ev.value(a.id)?;
            match &x {


                value::Value(t) => value::Value(metal_ops::unary_promote(t, metal_ops::UnOp::Tanh)?),
            }
        }
        NodeKind::Relu { a } => {
            let a = ev.value(a.id)?;
            match &a {


                value::Value(t) => value::Value(metal_ops::relu(t)?),
            }
        }
        NodeKind::Erf { a } => {
            let x = ev.value(a.id)?;
            match &x {


                value::Value(t) => value::Value(metal_ops::unary_promote(t, metal_ops::UnOp::Erf)?),
            }
        }
        NodeKind::Gelu { a, approximate } => {
            let x = ev.value(a.id)?;
            match &x {


                value::Value(t) => {
                    let op = if *approximate {
                        metal_ops::UnOp::GeluTanh
                    } else {
                        metal_ops::UnOp::Gelu
                    };
                    value::Value(metal_ops::unary_promote(t, op)?)
                }
            }
        }
        NodeKind::Floor { a } => {
            let x = ev.value(a.id)?;
            match &x {


                value::Value(t) => value::Value(metal_ops::unary_promote(t, metal_ops::UnOp::Floor)?),
            }
        }
        NodeKind::Ceil { a } => {
            let x = ev.value(a.id)?;
            match &x {


                value::Value(t) => value::Value(metal_ops::unary_promote(t, metal_ops::UnOp::Ceil)?),
            }
        }
        NodeKind::Round { a } => {
            let x = ev.value(a.id)?;
            match &x {


                value::Value(t) => value::Value(metal_ops::unary_promote(t, metal_ops::UnOp::Round)?),
            }
        }
        NodeKind::Sign { a } => {
            let x = ev.value(a.id)?;
            match &x {


                value::Value(t) => value::Value(metal_ops::unary_promote(t, metal_ops::UnOp::Sign)?),
            }
        }
        NodeKind::Where { cond, a, b } => {
            let cond = ev.value(cond.id)?;
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            match (&a, &b, &cond) {

                (value::Value(x), value::Value(y), value::Value(c)) => {
                    value::Value(metal_ops::where_(c, x, y)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Argmax { a, dim } => {
            let x = ev.value(a.id)?;
            match &x {


                value::Value(t) => {
                    let r = metal_ops::argreduce(t, *dim, true)?;
                    value::Value(metal_ops::cast(&r, runtime::dtype::DType::I64)?)
                }
            }
        }
        NodeKind::Argmin { a, dim } => {
            let x = ev.value(a.id)?;
            match &x {


                value::Value(t) => {
                    let r = metal_ops::argreduce(t, *dim, false)?;
                    value::Value(metal_ops::cast(&r, runtime::dtype::DType::I64)?)
                }
            }
        }
        NodeKind::Cumsum { a, dim } => {
            let x = ev.value(a.id)?;
            match &x {


                value::Value(t) => value::Value(metal_ops::cumsum(t, *dim)?),
            }
        }
        NodeKind::ScatterAdd {
            a,
            dim,
            indexes,
            src,
        } => {
            let a = ev.value(a.id)?;
            let indexes = ev.value(indexes.id)?;
            let src = ev.value(src.id)?;
            match (&a, &src) {

                (value::Value(x), value::Value(s)) => {
                    let ids = metal_ids_u32(&indexes)?;
                    value::Value(metal_ops::scatter_add(x, *dim, &ids, s)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Gather { a, dim, indexes } => {
            let a = ev.value(a.id)?;
            let indexes = ev.value(indexes.id)?;
            match &a {

                value::Value(x) => {
                    let ids = metal_ids_u32(&indexes)?;
                    value::Value(metal_ops::gather(x, *dim, &ids, &indexes.shape())?)
                }
            }
        }
        NodeKind::IndexSelect { a, dim, indexes } => {
            let a = ev.value(a.id)?;
            let indexes = ev.value(indexes.id)?;
            match &a {

                value::Value(x) => {
                    let ids = metal_ids_u32(&indexes)?;
                    value::Value(metal_ops::index_select(x, *dim, &ids)?)
                }
            }
        }
        NodeKind::CrossEntropy {
            logits,
            target,
            ignore_index,
            reduction,
        } => {
            let logits_t = ev.value(logits.id)?;
            let target_t = ev.value(target.id)?;
            match (&logits_t, &target_t) {

                (value::Value(l), value::Value(t)) => {
                    if loss::is_supported(l, t) {
                        let (loss_t, status) = loss::ce_forward(l, t, *ignore_index, *reduction)?;
                        let classes = l.layout.shape()[l.layout.shape().len() - 1];
                        let check = match reduction {
                            CeReduction::Mean => CeCheck::ForwardMean,
                            CeReduction::Sum => CeCheck::ForwardSum,
                        };
                        ev.ce_checks.push((value::Value(status), check, l.numel() / classes));
                        value::Value(loss_t)
                    } else {
                        let l32 = metal_ops::to_f32(l)?;
                        let r = composed::cross_entropy_forward(&l32, t, *ignore_index, *reduction)?;
                        value::Value(r)
                    }
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::CrossEntropyBackward {
            logits,
            target,
            ignore_index,
            reduction,
        } => {
            let logits_t = ev.value(logits.id)?;
            let target_t = ev.value(target.id)?;
            match (&logits_t, &target_t) {

                (value::Value(l), value::Value(t)) => {
                    if loss::is_supported(l, t) {
                        let (grad, count) = loss::ce_backward(l, t, *ignore_index, *reduction)?;
                        // Sum backward does not divide by the active
                        // count, so the zero-active check is a forward
                        // (Mean) concern only.
                        if *reduction == CeReduction::Mean {
                            ev.ce_checks.push((value::Value(count), CeCheck::BackwardMean, 0));
                        }
                        value::Value(grad)
                    } else {
                        let l32 = metal_ops::to_f32(l)?;
                        let r = composed::cross_entropy_backward(&l32, t, *ignore_index, *reduction)?;
                        value::Value(r)
                    }
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Sdpa {
            q,
            k,
            v,
            scale,
            causal,
        } => {
            let q = ev.value(q.id)?;
            let k = ev.value(k.id)?;
            let v = ev.value(v.id)?;
            match (&q, &k, &v) {

                (value::Value(q), value::Value(k), value::Value(v)) => {
                    if matches!(q.dtype, runtime::dtype::DType::F32 | runtime::dtype::DType::BF16) {
                        let (o, l) = flash::forward(q, k, v, *scale, *causal)?;
                        // L rides the evaluator for the chunked backward; the
                        // node's own cache entry holds O.
                        ev.multi.insert(node.id, vec![value::Value(o.clone()), value::Value(l)]);
                        value::Value(o)
                    } else {
                        let q32 = metal_ops::to_f32(q)?;
                        let k32 = metal_ops::to_f32(k)?;
                        let v32 = metal_ops::to_f32(v)?;
                        let r = composed::sdpa_forward(&q32, &k32, &v32, *scale, *causal)?;
                        value::Value(metal_ops::from_f32(&r, q.dtype)?)
                    }
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::SdpaBackward {
            q,
            k,
            v,
            g,
            fwd: _fwd,
            scale,
            causal,
        } => {
            let q = ev.value(q.id)?;
            let k = ev.value(k.id)?;
            let v = ev.value(v.id)?;
            let g = ev.value(g.id)?;

            let o = ev.value(_fwd.id)?;

            let l = ev.multi.get(&_fwd.id).and_then(|outs| outs.get(1)).cloned();
            let (dq, dk, dv) = match (&q, &k, &v, &g) {

                (value::Value(q), value::Value(k), value::Value(v), value::Value(g)) => {
                    if let (Some(value::Value(l)), true) = (&l, matches!(q.dtype, runtime::dtype::DType::F32 | runtime::dtype::DType::BF16)) {
                        let o = o.as_metal()?;
                        let (dq, dk, dv) = flash::backward_fused(q, k, v, o, l, g, *scale, *causal)?;
                        (
                            value::Value(dq),
                            value::Value(dk),
                            value::Value(dv),
                        )
                    } else {
                        let q32 = metal_ops::to_f32(q)?;
                        let k32 = metal_ops::to_f32(k)?;
                        let v32 = metal_ops::to_f32(v)?;
                        let g32 = metal_ops::to_f32(g)?;
                        let (dq, dk, dv) =
                            composed::sdpa_backward(&q32, &k32, &v32, &g32, *scale, *causal)?;
                        (
                            value::Value(metal_ops::from_f32(&dq, q.dtype)?),
                            value::Value(metal_ops::from_f32(&dk, q.dtype)?),
                            value::Value(metal_ops::from_f32(&dv, q.dtype)?),
                        )
                    }
                }
                _ => return Err("device mismatch".to_string()),
            };
            ev.multi.insert(node.id, vec![dq.clone(), dk, dv]);
            dq
        }
        NodeKind::SdpaBackwardOut { of, index } => ev
            .multi
            .get(&of.id)
            .and_then(|outs| outs.get(*index as usize))
            .cloned()
            .ok_or_else(|| {
                err::err_str("sdpa backward out: outputs missing".to_string())
            })?,
        NodeKind::KdaChunk {
            q,
            k,
            v,
            log_decay,
            beta,
            scale,
        } => {
            let q = ev.value(q.id)?;
            let k = ev.value(k.id)?;
            let v = ev.value(v.id)?;
            let log_decay = ev.value(log_decay.id)?;
            let beta = ev.value(beta.id)?;
            match (&q, &k, &v, &log_decay, &beta) {

                (
                    value::Value(q),
                    value::Value(k),
                    value::Value(v),
                    value::Value(g),
                    value::Value(b),
                ) => value::Value(composed::kda_chunk_forward(q, k, v, g, b, *scale)?),
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::KdaRecurrence {
            q,
            k,
            v,
            log_decay,
            beta,
            scale,
            layer,
        } => {
            let context = ev.kv.clone().ok_or_else(|| {
                err::err_str(
                    "kda recurrence: node evaluates only inside a decode program run".to_string(),
                )
            })?;
            kda_recurrence(
                &context,
                *layer,
                &ev.value(q.id)?,
                &ev.value(k.id)?,
                &ev.value(v.id)?,
                &ev.value(log_decay.id)?,
                &ev.value(beta.id)?,
                *scale,
            )?
        }
        NodeKind::ShortConv1d { x, weight } => {
            let x = ev.value(x.id)?;
            let weight = ev.value(weight.id)?;
            match (&x, &weight) {

                (value::Value(x), value::Value(w)) => {
                    value::Value(composed::short_conv1d_forward(x, w)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::KdaBackward {
            q,
            k,
            v,
            log_decay,
            beta,
            g,
            scale,
        } => {
            let q = ev.value(q.id)?;
            let k = ev.value(k.id)?;
            let v = ev.value(v.id)?;
            let log_decay = ev.value(log_decay.id)?;
            let beta = ev.value(beta.id)?;
            let g = ev.value(g.id)?;
            match (&q, &k, &v, &log_decay, &beta, &g) {

                (
                    value::Value(q),
                    value::Value(k),
                    value::Value(v),
                    value::Value(ld),
                    value::Value(b),
                    value::Value(g),
                ) => {
                    let (dq, dk, dv, dg, db) =
                        composed::kda_chunk_backward(q, k, v, ld, b, g, *scale)?;
                    let values = vec![
                        value::Value(dq),
                        value::Value(dk),
                        value::Value(dv),
                        value::Value(dg),
                        value::Value(db),
                    ];
                    let head = values[0].clone();
                    ev.multi.insert(node.id, values);
                    head
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::KdaBackwardOut { of, index } => ev
            .multi
            .get(&of.id)
            .and_then(|outs| outs.get(*index as usize))
            .cloned()
            .ok_or_else(|| err::err_str("kda backward out: outputs missing".to_string()))?,
        NodeKind::ShortConv1dBackwardX { x, weight, g } => {
            let x = ev.value(x.id)?;
            let weight = ev.value(weight.id)?;
            let g = ev.value(g.id)?;
            match (&x, &weight, &g) {

                (value::Value(x), value::Value(w), value::Value(g)) => {
                    value::Value(composed::short_conv1d_backward_x(x, w, g)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::ShortConv1dBackwardW { x, weight, g } => {
            let x = ev.value(x.id)?;
            let weight = ev.value(weight.id)?;
            let g = ev.value(g.id)?;
            match (&x, &weight, &g) {

                (value::Value(x), value::Value(w), value::Value(g)) => {
                    value::Value(composed::short_conv1d_backward_w(x, w, g)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::ConvState { x, weight, layer } => {
            let context = ev.kv.clone().ok_or_else(|| {
                err::err_str(
                    "conv state: node evaluates only inside a decode program run".to_string(),
                )
            })?;
            conv_state(&context, *layer, &ev.value(x.id)?, &ev.value(weight.id)?)?
        }
        NodeKind::PositionEmbedding { weight, seq_len } => {
            let w = ev.value(weight.id)?;
            match &w {

                value::Value(w) => {
                    let n = runtime::metal::run::MetalTensor {
                        buffer: w.buffer.clone(),
                        layout: w.layout.narrow(0, 0, *seq_len),
                        dtype: w.dtype,
                    };
                    value::Value(metal_ops::contiguous(&n)?)
                }
            }
        },
        NodeKind::KvAttention {
            q,
            k,
            v,
            scale,
            layer,
            window,
        } => {
            let kv = ev.kv.clone().ok_or_else(|| {
                err::err_str(
                    "kv attention: node evaluates only inside a kv program run".to_string(),
                )
            })?;
            kv_attention(
                &kv,
                *layer,
                &ev.value(q.id)?,
                &ev.value(k.id)?,
                &ev.value(v.id)?,
                *scale,
                *window,
            )?
        }
        NodeKind::RotaryEmbedding { x, theta, offset, .. } => {
            let x = ev.value(x.id)?;
            let offsets = match offset {
                PositionOffset::Absolute => vec![0usize],
                PositionOffset::Cursor => {
                    let kv = ev.kv.as_ref().ok_or_else(|| {
                        err::err_str(
                            "rotary embedding: cursor offset outside a kv program run".to_string(),
                        )
                    })?;
                    // One cursor per batch slot (RFC 0013).
                    let mut offsets = Vec::with_capacity(kv.slots.len());
                    for slot in &kv.slots {
                        offsets.push(slot.lock().map_err(|e| {
                            err::err_str(format!(
                                "rotary embedding: sequence lock poisoned: {e}"
                            ))
                        })?.cursor);
                    }
                    offsets
                }
            };
            match &x {

                value::Value(x) => {
                    if matches!(x.dtype, runtime::dtype::DType::F32 | runtime::dtype::DType::BF16) {
                        value::Value(rotary::rotary(x, &offsets, *theta, 1.0)?)
                    } else {
                        let x32 = metal_ops::to_f32(x)?;
                        let r = composed::rotary_forward(&x32, &offsets, *theta, 1.0)?;
                        value::Value(metal_ops::from_f32(&r, x.dtype)?)
                    }
                }
            }
        }
        NodeKind::RotaryEmbeddingBackward { g, theta, .. } => {
            let g = ev.value(g.id)?;
            match &g {

                value::Value(g) => {
                    if matches!(g.dtype, runtime::dtype::DType::F32 | runtime::dtype::DType::BF16) {
                        // Transpose rotation == forward with negated angles.
                        value::Value(rotary::rotary(g, &[0usize], *theta, -1.0)?)
                    } else {
                        let g32 = metal_ops::to_f32(g)?;
                        let r = composed::rotary_forward(&g32, &[0usize], *theta, -1.0)?;
                        value::Value(metal_ops::from_f32(&r, g.dtype)?)
                    }
                }
            }
        }
        NodeKind::Linear { x, weight, bias } => {
            let x = ev.value(x.id)?;
            let weight = ev.value(weight.id)?;
            let bias = ev.value(bias.id)?;
            match (&x, &weight, &bias) {

                (value::Value(x), value::Value(w), value::Value(b)) => {
                    if linear::is_supported(x, w) {
                        value::Value(linear::linear_forward(x, w, b)?)
                    } else {
                        let x32 = metal_ops::to_f32(x)?;
                        let w32 = metal_ops::to_f32(w)?;
                        let b32 = metal_ops::to_f32(b)?;
                        let r = metal_ops::binary(
                            &metal_ops::matmul(&x32, &w32)?,
                            &b32,
                            metal_ops::BinOp::Add,
                        )?;
                        value::Value(metal_ops::from_f32(&r, x.dtype)?)
                    }
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::LinearResidual {
            x,
            weight,
            bias,
            residual,
        } => {
            let x = ev.value(x.id)?;
            let weight = ev.value(weight.id)?;
            let bias = ev.value(bias.id)?;
            let residual = ev.value(residual.id)?;
            match (&x, &weight, &bias, &residual) {

                (value::Value(x), value::Value(w), value::Value(b), value::Value(r)) => {
                    if linear::is_supported(x, w) {
                        let (out, extra) = linear::linear_forward_fused(x, w, b, Some(r), None)?;
                        debug_assert!(extra.is_none());
                        value::Value(out)
                    } else {
                        let x32 = metal_ops::to_f32(x)?;
                        let w32 = metal_ops::to_f32(w)?;
                        let b32 = metal_ops::to_f32(b)?;
                        let r32 = metal_ops::to_f32(r)?;
                        let out = metal_ops::binary(
                            &metal_ops::binary(&metal_ops::matmul(&x32, &w32)?, &b32, metal_ops::BinOp::Add)?,
                            &r32,
                            metal_ops::BinOp::Add,
                        )?;
                        value::Value(metal_ops::from_f32(&out, x.dtype)?)
                    }
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::LinearGelu {
            x,
            weight,
            bias,
            approximate,
            dual,
        } => {
            let x = ev.value(x.id)?;
            let weight = ev.value(weight.id)?;
            let bias = ev.value(bias.id)?;
            let finish = |m: value::Value, g: value::Value, ev: &mut Evaluator| {
                if *dual {
                    ev.multi.insert(node.id, vec![m.clone(), g]);
                    m
                } else {
                    g
                }
            };
            match (&x, &weight, &bias) {

                (value::Value(x), value::Value(w), value::Value(b)) => {
                    if linear::is_supported(x, w) {
                        let (out, out2) = linear::linear_forward_fused(
                            x,
                            w,
                            b,
                            None,
                            Some((*approximate, *dual)),
                        )?;
                        if *dual {
                            let g = out2.expect("dual gelu gemm writes two outputs");
                            finish(value::Value(out), value::Value(g), ev)
                        } else {
                            value::Value(out)
                        }
                    } else {
                        let x32 = metal_ops::to_f32(x)?;
                        let w32 = metal_ops::to_f32(w)?;
                        let b32 = metal_ops::to_f32(b)?;
                        let m32 = metal_ops::binary(
                            &metal_ops::matmul(&x32, &w32)?,
                            &b32,
                            metal_ops::BinOp::Add,
                        )?;
                        let m = metal_ops::from_f32(&m32, x.dtype)?;
                        let op = if *approximate {
                            metal_ops::UnOp::GeluTanh
                        } else {
                            metal_ops::UnOp::Gelu
                        };
                        let g = metal_ops::unary_promote(&m, op)?;
                        finish(value::Value(m), value::Value(g), ev)
                    }
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::LayerNorm { x, weight, bias, eps } => {
            let x = ev.value(x.id)?;
            let weight = ev.value(weight.id)?;
            let bias = ev.value(bias.id)?;
            match (&x, &weight, &bias) {

                (value::Value(x), value::Value(w), value::Value(b)) => {
                    if layer_norm::is_supported(x, w) {
                        value::Value(layer_norm::ln_forward(x, w, b, *eps)?)
                    } else {
                        let x32 = metal_ops::to_f32(x)?;
                        let w32 = metal_ops::to_f32(w)?;
                        let b32 = metal_ops::to_f32(b)?;
                        let r = composed::layer_norm_forward(&x32, &w32, &b32, *eps)?;
                        value::Value(metal_ops::from_f32(&r, x.dtype)?)
                    }
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::LayerNormBackward { x, weight, g, eps } => {
            let x = ev.value(x.id)?;
            let weight = ev.value(weight.id)?;
            let g = ev.value(g.id)?;
            match (&x, &weight, &g) {

                (value::Value(x), value::Value(w), value::Value(g)) => {
                    if layer_norm::is_supported(x, w) {
                        let (dx, xh) = layer_norm::ln_backward(x, w, g, *eps)?;
                        let dw = metal_ops::reduce(
                            &metal_ops::binary(g, &xh, metal_ops::BinOp::Mul)?,
                            &(0..x.layout.shape().len() - w.layout.shape().len()).collect::<Vec<_>>(),
                            false,
                            fusion::ReduceOp::Sum,
                        )?;
                        let db = metal_ops::reduce(
                            g,
                            &(0..x.layout.shape().len() - w.layout.shape().len()).collect::<Vec<_>>(),
                            false,
                            fusion::ReduceOp::Sum,
                        )?;
                        ev.ln.insert(node.id, [value::Value(dw), value::Value(db)]);
                        value::Value(dx)
                    } else {
                        let x32 = metal_ops::to_f32(x)?;
                        let w32 = metal_ops::to_f32(w)?;
                        let g32 = metal_ops::to_f32(g)?;
                        let (dx, dw, db) =
                            composed::layer_norm_backward(&x32, &w32, &g32, *eps)?;
                        ev.ln.insert(node.id, [
                            value::Value(metal_ops::from_f32(&dw, w.dtype)?),
                            value::Value(metal_ops::from_f32(&db, w.dtype)?),
                        ]);
                        value::Value(metal_ops::from_f32(&dx, x.dtype)?)
                    }
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::LayerNormBackwardOut { of, index } => {
            let _ = ev.value(of.id)?;
            ev.ln
                .get(&of.id)
                .and_then(|outs| outs.get(*index as usize - 1))
                .cloned()
                .ok_or_else(|| {
                    err::err_str(
                        "layer_norm_backward_out: backward node has no stored outputs".to_string(),
                    )
                })?
        }
        NodeKind::Conv1d {
            x,
            w,
            stride,
            padding,
            dilation,
            groups,
        } => {
            let x = ev.value(x.id)?;
            let w = ev.value(w.id)?;
            match (&x, &w) {

                (value::Value(x), value::Value(w)) => {
                    let xn = metal_ops::contiguous(x)?;
                    let wn = metal_ops::contiguous(w)?;
                    value::Value(runtime::metal::conv::conv1d(
                        runtime::metal::device::MetalDevice::get(),
                        &xn,
                        &wn,
                        *stride,
                        *padding,
                        *dilation,
                        *groups,
                    )?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Conv2d {
            x,
            w,
            stride,
            padding,
            dilation,
            groups,
        } => {
            let x = ev.value(x.id)?;
            let w = ev.value(w.id)?;
            match (&x, &w) {

                (value::Value(x), value::Value(w)) => {
                    let xn = metal_ops::contiguous(x)?;
                    let wn = metal_ops::contiguous(w)?;
                    value::Value(runtime::metal::conv::conv2d(
                        runtime::metal::device::MetalDevice::get(),
                        &xn,
                        &wn,
                        *stride,
                        *padding,
                        *dilation,
                        *groups,
                    )?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::ConvTranspose1d {
            x,
            w,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        } => {
            let x = ev.value(x.id)?;
            let w = ev.value(w.id)?;
            match (&x, &w) {

                (value::Value(x), value::Value(w)) => {
                    let xn = metal_ops::contiguous(x)?;
                    let wn = metal_ops::contiguous(w)?;
                    value::Value(runtime::metal::conv::conv_transpose1d(
                        runtime::metal::device::MetalDevice::get(),
                        &xn,
                        &wn,
                        *stride,
                        *padding,
                        *output_padding,
                        *dilation,
                        *groups,
                    )?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::ConvTranspose2d {
            x,
            w,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        } => {
            let x = ev.value(x.id)?;
            let w = ev.value(w.id)?;
            match (&x, &w) {

                (value::Value(x), value::Value(w)) => {
                    let xn = metal_ops::contiguous(x)?;
                    let wn = metal_ops::contiguous(w)?;
                    value::Value(runtime::metal::conv::conv_transpose2d(
                        runtime::metal::device::MetalDevice::get(),
                        &xn,
                        &wn,
                        *stride,
                        *padding,
                        *output_padding,
                        *dilation,
                        *groups,
                    )?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Conv1dBackwardW {
            x,
            g,
            kernel,
            out_channels,
            stride,
            padding,
            dilation,
            groups,
        } => {
            let x = ev.value(x.id)?;
            let g = ev.value(g.id)?;

            let squeeze4 = |t: &runtime::metal::run::MetalTensor| -> runtime::metal::run::MetalTensor {
                let s = t.layout.shape();
                runtime::metal::run::MetalTensor {
                    buffer: t.buffer.clone(),
                    layout: runtime::layout::Layout::contiguous(vec![s[0], s[1], s[2], 1]),
                    dtype: t.dtype,
                }
            };

            let squeeze3 = |t: &runtime::metal::run::MetalTensor| -> runtime::metal::run::MetalTensor {
                let s = t.layout.shape();
                runtime::metal::run::MetalTensor {
                    buffer: t.buffer.clone(),
                    layout: runtime::layout::Layout::contiguous(vec![s[0], s[1], s[2]]),
                    dtype: t.dtype,
                }
            };
            match (&x, &g) {

                (value::Value(x), value::Value(g)) => {
                    let xn = metal_ops::contiguous(&squeeze4(x))?;
                    let gn = metal_ops::contiguous(&squeeze4(g))?;
                    let dw = runtime::metal::conv::conv2d_backward_w(
                        runtime::metal::device::MetalDevice::get(),
                        &xn,
                        &gn,
                        [*kernel, 1],
                        *out_channels,
                        *stride,
                        *padding,
                        *dilation,
                        *groups,
                    )?;
                    value::Value(squeeze3(&dw))
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Conv2dBackwardW {
            x,
            g,
            kernel,
            out_channels,
            stride,
            padding,
            dilation,
            groups,
        } => {
            let x = ev.value(x.id)?;
            let g = ev.value(g.id)?;
            match (&x, &g) {

                (value::Value(x), value::Value(g)) => {
                    let xn = metal_ops::contiguous(x)?;
                    let gn = metal_ops::contiguous(g)?;
                    value::Value(runtime::metal::conv::conv2d_backward_w(
                        runtime::metal::device::MetalDevice::get(),
                        &xn,
                        &gn,
                        *kernel,
                        *out_channels,
                        *stride,
                        *padding,
                        *dilation,
                        *groups,
                    )?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Pow { a, exp } => {
            let x = ev.value(a.id)?;
            match &x {


                value::Value(t) => value::Value(metal_ops::powf(&metal_ops::to_f32(t)?, *exp)?),
            }
        }
        NodeKind::Cast { a, dtype } => {
            let x = ev.value(a.id)?;
            match &x {


                value::Value(t) => {
                    value::Value(metal_ops::cast(t, *dtype)?)
                }
            }
        }
        NodeKind::Sum { a, dims, keepdims } => {
            let t = ev.value(a.id)?;
            match &t {

                value::Value(x) => {
                    value::Value({
                        let x32;
                        let x = if matches!(x.dtype, runtime::dtype::DType::F32 | runtime::dtype::DType::BF16) {
                            x
                        } else {
                            x32 = metal_ops::to_f32(x)?;
                            &x32
                        };
                        metal_ops::reduce(x, dims, *keepdims, fusion::ReduceOp::Sum)?
                    })
                }
            }
        }
        NodeKind::Mean { a, dims, keepdims } => {
            let t = ev.value(a.id)?;
            match &t {

                value::Value(x) => {
                    value::Value({
                        let x32;
                        let x = if matches!(x.dtype, runtime::dtype::DType::F32 | runtime::dtype::DType::BF16) {
                            x
                        } else {
                            x32 = metal_ops::to_f32(x)?;
                            &x32
                        };
                        metal_ops::reduce(x, dims, *keepdims, fusion::ReduceOp::Mean)?
                    })
                }
            }
        }
        NodeKind::Max { a, dims, keepdims } => {
            let t = ev.value(a.id)?;
            match &t {

                value::Value(x) => {
                    value::Value({
                        let x32;
                        let x = if matches!(x.dtype, runtime::dtype::DType::F32 | runtime::dtype::DType::BF16) {
                            x
                        } else {
                            x32 = metal_ops::to_f32(x)?;
                            &x32
                        };
                        metal_ops::reduce(x, dims, *keepdims, fusion::ReduceOp::Max)?
                    })
                }
            }
        }
        NodeKind::Min { a, dims, keepdims } => {
            let t = ev.value(a.id)?;
            match &t {

                value::Value(x) => {
                    value::Value({
                        let x32;
                        let x = if matches!(x.dtype, runtime::dtype::DType::F32 | runtime::dtype::DType::BF16) {
                            x
                        } else {
                            x32 = metal_ops::to_f32(x)?;
                            &x32
                        };
                        metal_ops::reduce(x, dims, *keepdims, fusion::ReduceOp::Min)?
                    })
                }
            }
        }
        NodeKind::Prod { a, dims, keepdims } => {
            let t = ev.value(a.id)?;
            match &t {

                value::Value(x) => {
                    value::Value({
                        let x32;
                        let x = if matches!(x.dtype, runtime::dtype::DType::F32 | runtime::dtype::DType::BF16) {
                            x
                        } else {
                            x32 = metal_ops::to_f32(x)?;
                            &x32
                        };
                        metal_ops::reduce(x, dims, *keepdims, fusion::ReduceOp::Prod)?
                    })
                }
            }
        }
        NodeKind::Reshape { a, shape } => {
            let x = ev.value(a.id)?;
            match &x {

                value::Value(t) => {
                    let r = metal_ops::contiguous(t)?;
                    value::Value(runtime::metal::run::MetalTensor {
                        buffer: r.buffer,
                        layout: runtime::layout::Layout::contiguous(shape.clone()),
                        dtype: r.dtype,
                    })
                }
            }
        }
        NodeKind::Permute { a, dims } => {
            let x = ev.value(a.id)?;
            match &x {

                value::Value(t) => {
                    value::Value(metal_ops::permute(t, dims)?)
                }
            }
        }
        NodeKind::Slice { a, ranges } => {
            let t = ev.value(a.id)?;
            match &t {

                value::Value(x) => {
                    let mut r = x.clone();
                    for (dim, &(start, stop, stride)) in ranges.iter().enumerate() {
                        let len = stop.saturating_sub(start).div_ceil(stride);
                        if len == 0 {
                            let mut shape = r.layout.shape().to_vec();
                            shape[dim] = 0;
                            r = runtime::metal::run::MetalTensor::zeros(
                                runtime::metal::device::MetalDevice::get(),
                                shape,
                                r.dtype,
                            );
                            continue;
                        }
                        r = runtime::metal::run::MetalTensor {
                            buffer: r.buffer.clone(),
                            layout: r.layout.narrow(dim, start, (len - 1) * stride + 1),
                            dtype: r.dtype,
                        };
                        r = metal_ops::contiguous(&r)?;
                        if stride > 1 {
                            let idx: Vec<u32> = (0..len as u32).map(|i| i * stride as u32).collect();
                            let ids = runtime::metal::indexing::ids_from_host(
                                runtime::metal::device::MetalDevice::get(),
                                &idx,
                            );
                            r = metal_ops::index_select(&r, dim, &ids)?;
                        }
                    }
                    value::Value(r)
                }
            }
        }
        NodeKind::Concat { a, b, dim } => {
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            match (&a, &b) {

                (value::Value(x), value::Value(y)) => {
                    value::Value(metal_ops::cat(x, y, *dim)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::BroadcastTo { a, shape } => {
            let x = ev.value(a.id)?;
            match &x {

                value::Value(t) => {
                    value::Value(metal_ops::broadcast_to(t, shape)?)
                }
            }
        }
        NodeKind::Matmul { a, b } => {
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            match (&a, &b) {


                (value::Value(x), value::Value(y)) => {
                    value::Value(metal_ops::matmul(x, y)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Inverse { a } => {
            let t = ev.value(a.id)?;
            match &t {


                value::Value(_) => return Err("inverse is not supported on Metal".to_string()),
            }
        }
        NodeKind::Det { a } => {
            let t = ev.value(a.id)?;
            match &t {


                value::Value(_) => return Err("det is not supported on Metal".to_string()),
            }
        }
        NodeKind::Solve { a, b } => {
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            match (&a, &b) {


                (value::Value(_), value::Value(_)) => {
                    return Err("solve is not supported on Metal".to_string())
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::StopGradient { a } => ev.value(a.id)?,
        NodeKind::Checkpoint { a } => ev.value(a.id)?,
        NodeKind::AdamWStep {
            param,
            grad,
            m,
            v,
            lr,
            c1,
            c2,
            beta1,
            beta2,
            eps,
            weight_decay,
        } => {
            let p = ev.value(param.id)?;
            let g = ev.value(grad.id)?;
            let m_t = ev.value(m.id)?;
            let v_t = ev.value(v.id)?;
            // The step-varying scalars arrive as 0-d tensors; cast to the
            // parameter dtype and copy to its device. The evaluator memoizes
            // these scalar conversions across every parameter in the step.
            let lr_t = ev.step_scalar(lr.id, p.dtype(), &p.device())?;
            let c1_t = ev.step_scalar(c1.id, p.dtype(), &p.device())?;
            let c2_t = ev.step_scalar(c2.id, p.dtype(), &p.device())?;
            let fused = if fusion::is_supported(&p.device(), p.dtype()) {
                let plan = fusion::adamw_group_plan(
                    1,
                    &p.shape(),
                    *beta1,
                    *beta2,
                    *eps,
                    *weight_decay,
                );
                let pack = match ev.scalar_packs.entry((lr.id, c1.id, c2.id)) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(fusion::pack_scalars_metal(&[
                            lr_t.clone(),
                            c1_t.clone(),
                            c2_t.clone(),
                        ])?)
                    }
                };
                fusion::run_group_metal(
                    &plan,
                    &[p.clone(), g.clone(), m_t.clone(), v_t.clone()],
                    pack,
                    &p.shape(),
                )
                .ok()
            } else {
                None
            };
            match fused {
                Some(outs) => {
                    let mut it = outs.into_iter();
                    let next_p = it.next().unwrap();
                    ev.adamw
                        .insert(node.id, [it.next().unwrap(), it.next().unwrap()]);
                    next_p
                }
                None => match &p {

                    value::Value(p) => {
                        let g32 = metal_ops::to_f32(g.as_metal()?)?;
                        let m32 = metal_ops::to_f32(m_t.as_metal()?)?;
                        let v32 = metal_ops::to_f32(v_t.as_metal()?)?;
                        let (np, nm, nv) = adamw_native(
                            p,
                            &g32,
                            &m32,
                            &v32,
                            &metal_ops::to_f32(lr_t.as_metal()?)?,
                            &metal_ops::to_f32(c1_t.as_metal()?)?,
                            &metal_ops::to_f32(c2_t.as_metal()?)?,
                            *beta1,
                            *beta2,
                            *eps,
                            *weight_decay,
                        )?;
                        ev.adamw.insert(node.id, [value::Value(nm), value::Value(nv)]);
                        value::Value(np)
                    }
                },
            }
        }
        NodeKind::AdamWOut { step, index } => {
            // the step is evaluated before its projections; make sure of it
            let _ = ev.value(step.id)?;
            let outputs = ev.adamw.get(&step.id).ok_or_else(|| {
                err::err_str("adamw_out: step has no stored moments".to_string())
            })?;
            match index {
                0 => ev.value(step.id)?,
                1 => outputs[0].clone(),
                _ => outputs[1].clone(),
            }
        }
        NodeKind::AdamWStepGroup {
            params,
            grads,
            ms,
            vs,
            lr,
            c1,
            c2,
            beta1,
            beta2,
            eps,
            weight_decay,
        } => {
            let first_p = ev.value(params[0].id)?;
            let lr_t = ev.step_scalar(lr.id, first_p.dtype(), &first_p.device())?;
            let c1_t = ev.step_scalar(c1.id, first_p.dtype(), &first_p.device())?;
            let c2_t = ev.step_scalar(c2.id, first_p.dtype(), &first_p.device())?;
            let mut inputs = Vec::with_capacity(params.len() * 4);
            for i in 0..params.len() {
                inputs.push(ev.value(params[i].id)?);
                inputs.push(ev.value(grads[i].id)?);
                inputs.push(ev.value(ms[i].id)?);
                inputs.push(ev.value(vs[i].id)?);
            }
            let outs = if fusion::is_supported(&first_p.device(), first_p.dtype()) {


                {
                    let plan = fusion::adamw_group_plan(
                        params.len(),
                        &first_p.shape(),
                        *beta1,
                        *beta2,
                        *eps,
                        *weight_decay,
                    );
                    let pack = match ev.scalar_packs.entry((lr.id, c1.id, c2.id)) {
                        std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                        std::collections::hash_map::Entry::Vacant(e) => {
                            e.insert(fusion::pack_scalars_metal(&[lr_t, c1_t, c2_t])?)
                        }
                    };
                    fusion::run_group_metal(&plan, &inputs, pack, &first_p.shape())?
                }
            } else {
                let base = fusion::adamw_exprs(*beta1, *beta2, *eps, *weight_decay);
                let mut exprs = Vec::with_capacity(params.len() * 3);
                for i in 0..params.len() {
                    let remap: std::collections::HashMap<u32, u32> = (0u32..4)
                        .map(|k| (k, (i * 4) as u32 + k))
                        .collect();
                    for expr in &base {
                        exprs.push(expr.remap_lanes(&remap));
                    }
                }
                fusion::run(
                    &exprs,
                    &inputs,
                    None,
                    &[lr_t, c1_t, c2_t],
                    first_p.numel(),
                    &first_p.shape(),
                    first_p.dtype(),
                    &first_p.device(),
                )?
            };
            let head = outs[0].clone();
            ev.multi.insert(node.id, outs);
            head
        }
        NodeKind::AdamWGroupOut { of, param, index } => {
            let _ = ev.value(of.id)?;
            ev.multi
                .get(&of.id)
                .and_then(|outs| outs.get(*param as usize * 3 + *index as usize))
                .cloned()
                .ok_or_else(|| {
                    err::err_str("adamw_group_out: group has no stored outputs".to_string())
                })?
        }
        NodeKind::SgdStep {
            param,
            grad,
            velocity,
            first,
            lr,
            momentum,
            dampening,
            nesterov,
            weight_decay,
        } => {
            let p = ev.value(param.id)?;
            let g = ev.value(grad.id)?;
            let v_t = ev.value(velocity.id)?;
            let first_t = ev.step_scalar(first.id, p.dtype(), &p.device())?;
            let lr_t = ev.step_scalar(lr.id, p.dtype(), &p.device())?;
            let fused = if fusion::is_supported(&p.device(), p.dtype()) {
                let exprs = fusion::sgd_exprs(*momentum, *dampening, *nesterov, *weight_decay);
                fusion::run(
                    &exprs,
                    &[p.clone(), g.clone(), v_t.clone()],
                    None,
                    &[lr_t.clone(), first_t.clone()],
                    p.numel(),
                    &p.shape(),
                    p.dtype(),
                    &p.device(),
                )
                .ok()
            } else {
                None
            };
            match fused {
                Some(outs) => {
                    let mut it = outs.into_iter();
                    let next_p = it.next().unwrap();
                    ev.sgd.insert(node.id, it.next().unwrap());
                    next_p
                }
                None => match &p {

                    value::Value(p) => {
                        let g = if *weight_decay == 0.0 {
                            g.as_metal()?.clone()
                        } else {
                            let wd = metal_ops::fill(p.layout.shape(), *weight_decay, p.dtype)?;
                            metal_ops::binary(
                                g.as_metal()?,
                                &metal_ops::binary(p, &wd, metal_ops::BinOp::Mul)?,
                                metal_ops::BinOp::Add,
                            )?
                        };
                        // next_v = first ? g : momentum * v + (1 - dampening) * g,
                        // as tensor arithmetic: velocity is zeros on the first
                        // step, so the (1 - first) branch contributes nothing.
                        let mom = metal_ops::fill(p.layout.shape(), *momentum, p.dtype)?;
                        let damp = metal_ops::fill(p.layout.shape(), 1.0 - dampening, p.dtype)?;
                        let continued = metal_ops::binary(
                            &metal_ops::binary(v_t.as_metal()?, &mom, metal_ops::BinOp::Mul)?,
                            &metal_ops::binary(&g, &damp, metal_ops::BinOp::Mul)?,
                            metal_ops::BinOp::Add,
                        )?;
                        let first32 = metal_ops::to_f32(first_t.as_metal()?)?;
                        let not_first = metal_ops::binary(
                            &metal_ops::fill(first32.layout.shape(), 1.0, first32.dtype)?,
                            &first32,
                            metal_ops::BinOp::Sub,
                        )?;
                        let next_v = metal_ops::binary(
                            &metal_ops::binary(&first32, &g, metal_ops::BinOp::Mul)?,
                            &metal_ops::binary(&not_first, &continued, metal_ops::BinOp::Mul)?,
                            metal_ops::BinOp::Add,
                        )?;
                        let used = if *nesterov {
                            metal_ops::binary(
                                &g,
                                &metal_ops::binary(&next_v, &mom, metal_ops::BinOp::Mul)?,
                                metal_ops::BinOp::Add,
                            )?
                        } else {
                            next_v.clone()
                        };
                        let next_p = metal_ops::binary(
                            p,
                            &metal_ops::binary(&used, &metal_ops::to_f32(lr_t.as_metal()?)?, metal_ops::BinOp::Mul)?,
                            metal_ops::BinOp::Sub,
                        )?;
                        ev.sgd.insert(node.id, value::Value(next_v));
                        value::Value(next_p)
                    }
                },
            }
        }
        NodeKind::SgdOut { step, index } => {
            let _ = ev.value(step.id)?;
            match index {
                0 => ev.value(step.id)?,
                _ => ev.sgd.get(&step.id).cloned().ok_or_else(|| {
                    err::err_str("sgd_out: step has no stored velocity".to_string())
                })?,
            }
        }
        NodeKind::FusedElementwise {
            inputs,
            strides,
            shape,
            expr,
        } => {
            let ts: Vec<value::Value> = inputs
                .iter()
                .map(|i| ev.value(i.id))
                .collect::<err::Res<Vec<_>>>()?;
            let strides: Vec<Vec<usize>> = strides.to_vec();
            let first = &ts[0];
            let outs = fusion::run(
                std::slice::from_ref(expr),
                &ts,
                Some(&strides),
                &[],
                shape.iter().product(),
                shape,
                first.dtype(),
                &first.device(),
            )?;
            outs.into_iter().next().unwrap()
        }
        NodeKind::FusedElementwiseMulti {
            inputs,
            strides,
            shape,
            exprs,
        } => {
            let ts: Vec<value::Value> = inputs
                .iter()
                .map(|i| ev.value(i.id))
                .collect::<err::Res<Vec<_>>>()?;
            let strides: Vec<Vec<usize>> = strides.to_vec();
            let first = &ts[0];
            let outs = fusion::run(
                exprs,
                &ts,
                Some(&strides),
                &[],
                shape.iter().product(),
                shape,
                first.dtype(),
                &first.device(),
            )?;
            let head = outs[0].clone();
            ev.multi.insert(node.id, outs);
            head
        }
        NodeKind::FusedPick { of, index } => ev
            .multi
            .get(&of.id)
            .and_then(|outs| outs.get(*index as usize))
            .cloned()
            .ok_or_else(|| {
                err::err_str("fused pick: multi output missing".to_string())
            })?,
        NodeKind::FusedReduce {
            inputs,
            strides,
            in_shape,
            expr,
            op,
            dims,
            keepdims,
            shape,
        } => {
            let fine = std::env::var_os("EFFECT_TORCH_FUSION_TIMING").is_some();
            let t0 = std::time::Instant::now();
            let ts: Vec<value::Value> = inputs
                .iter()
                .map(|i| ev.value(i.id))
                .collect::<err::Res<Vec<_>>>()?;
            if fine {
                eprintln!("[fine] reduce collect {:.1}us ({} inputs)", t0.elapsed().as_micros(), ts.len());
            }
            let first = &ts[0];
            let strides: Vec<Vec<usize>> = strides.to_vec();
            fusion::run_reduce(
                *op,
                expr,
                &ts,
                &strides,
                in_shape,
                dims,
                *keepdims,
                shape,
                first.dtype(),
                &first.device(),
            )?
        }
    };
    Ok(output)
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

// The vendored Metal backend keeps one shared "current" command buffer per
// device (candle-metal-kernels `Commands`): concurrent evaluation walks on
// the same device interleave their kernels into each other's buffers, and a
// walk can read back outputs whose producing kernels were committed — and
// only awaited — by another walk. Serialize Metal walks process-wide; GPU
// work serializes on the command queue anyway, so this costs nothing.

static METAL_EVAL_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

fn metal_eval_guard(_nodes: &[Arc<Node>]) -> std::sync::MutexGuard<'static, ()> {
    METAL_EVAL_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

#[napi]
pub async fn eval_lazy(
    tensors: Vec<&LazyTensor>,
    token: Option<&CancellationToken>,
) -> Result<Vec<NativeTensor>> {
    let nodes: Vec<Arc<Node>> = tensors.iter().map(|t| t.node.clone()).collect();
    let walk_timing = std::env::var_os("EFFECT_TORCH_WALK_TIMING").is_some();
    let t0 = std::time::Instant::now();
    let nodes = fuse_roots_cached(&nodes)?;
    if walk_timing {
        eprintln!(
            "[walk] fuse_roots {:.1}us ({} nodes)",
            t0.elapsed().as_micros(),
            nodes.len()
        );
    }
    if std::env::var_os("EFFECT_TORCH_EVAL_STATS").is_some() {
        let mut counts: HashMap<&'static str, usize> = HashMap::new();
        let mut seen = std::collections::HashSet::new();
        let mut stack: Vec<Arc<Node>> = nodes.clone();
        while let Some(node) = stack.pop() {
            if !seen.insert(node.id) {
                continue;
            }
            *counts.entry(node_kind_name(&node.kind)).or_insert(0) += 1;
            stack.extend(node_children(&node.kind));
        }
        let mut entries: Vec<_> = counts.into_iter().collect();
        entries.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        eprintln!(
            "[eval-stats] {} nodes: {}",
            seen.len(),
            entries
                .iter()
                .map(|(k, n)| format!("{k}×{n}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    run_compute(token, move |cancelled, _state| {
        let t1 = std::time::Instant::now();
        let _guard = metal_eval_guard(&nodes);
        let mut ev = Evaluator::new(&nodes);
        let mut outputs = Vec::with_capacity(nodes.len());
        for node in &nodes {
            let output = eval_node(node, cancelled, &mut ev).map_err(to_napi_err)?;
            outputs.push(NativeTensor::wrap(output));
        }
        // Synchronize once after all roots have been encoded.
        // The sync is device-global; one call covers every output.
        if let Some(first) = outputs.first() {
            first.val_cloned()?.synchronize().map_err(to_napi_err)?;
        }
        // Deferred fused-CE status checks (would have split the
        // pipeline mid-walk).
        ev.run_ce_checks().map_err(to_napi_err)?;
        if walk_timing {
            eprintln!(
                "[walk] eval {:.1}us ({} nodes)",
                t1.elapsed().as_micros(),
                nodes.len()
            );
        }
        Ok(outputs)
    })
    .await
}

// Fusion is a deterministic pure function of the graph, and graphs are
// immutable (stable node ids), so eval-time fusion is memoized by root
// ids — re-walking the same graph (training loops, repeated computes)
// pays fuse_roots once, not 2.4ms per eval. Bounded: 32 entries, evict
// oldest.
fn fuse_roots_cached(roots: &[Arc<Node>]) -> Result<Vec<Arc<Node>>> {
    if std::env::var_os("EFFECT_TORCH_NO_FUSION").is_some() {
        return Ok(roots.to_vec());
    }
    type FusionCache = (u64, HashMap<Vec<u64>, (u64, Vec<Arc<Node>>)>);
    static CACHE: LazyLock<Mutex<FusionCache>> = LazyLock::new(|| Mutex::new((0, HashMap::new())));
    let key: Vec<u64> = roots.iter().map(|r| r.id).collect();
    {
        let mut cache = CACHE.lock().map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("fusion cache lock poisoned: {e}"),
            )
        })?;
        cache.0 += 1;
        let tick = cache.0;
        if let Some((entry_tick, fused)) = cache.1.get_mut(&key) {
            *entry_tick = tick;
            return Ok(fused.clone());
        }
    }
    let fused = fuse_roots(roots).map_err(|e| Error::new(Status::GenericFailure, e))?;
    let mut cache = CACHE.lock().map_err(|e| {
        Error::new(
            Status::GenericFailure,
            format!("fusion cache lock poisoned: {e}"),
        )
    })?;
    cache.0 += 1;
    let tick = cache.0;
    if cache.1.len() >= 32 {
        if let Some(oldest) = cache
            .1
            .iter()
            .min_by_key(|(_, (tick, _))| *tick)
            .map(|(k, _)| k.clone())
        {
            cache.1.remove(&oldest);
        }
    }
    cache.1.insert(key, (tick, fused.clone()));
    Ok(fused)
}

// Per-kind self-time accounting for EFFECT_TORCH_KIND_TIMING: prints
// aggregates every 20000 node evals.
fn kind_timing_nanos(kind: &'static str, nanos: u64) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static LOCK: LazyLock<Mutex<HashMap<&'static str, (u64, u64)>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    static CALLS: AtomicU64 = AtomicU64::new(0);
    if let Ok(mut map) = LOCK.lock() {
        let entry = map.entry(kind).or_insert((0, 0));
        entry.0 += nanos;
        entry.1 += 1;
    }
    let calls = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if calls.is_multiple_of(4096) {
        if let Ok(map) = LOCK.lock() {
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by_key(|(_, (nanos, _))| std::cmp::Reverse(*nanos));
            eprintln!(
                "[kind-timing] {}",
                entries
                    .iter()
                    .map(|(k, (nanos, count))| format!("{k} {:.1}ms/{count}", *nanos as f64 / 1e6))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }
}

// Coarse node-kind names for EFFECT_TORCH_EVAL_STATS.
fn node_kind_name(kind: &NodeKind) -> &'static str {
    match kind {
        NodeKind::FusedElementwise { .. } => "Fused",
        NodeKind::FusedElementwiseMulti { .. } => "FusedMulti",
        NodeKind::FusedReduce { .. } => "FusedReduce",
        NodeKind::FusedPick { .. } => "FusedPick",
        NodeKind::Add { .. } => "Add",
        NodeKind::Sub { .. } => "Sub",
        NodeKind::Mul { .. } => "Mul",
        NodeKind::Div { .. } => "Div",
        NodeKind::Matmul { .. } => "Matmul",
        NodeKind::Linear { .. } => "Linear",
        NodeKind::LinearGelu { .. } => "LinearGelu",
        NodeKind::LinearResidual { .. } => "LinearResidual",
        NodeKind::Gelu { .. } => "Gelu",
        NodeKind::Sdpa { .. } => "Sdpa",
        NodeKind::SdpaBackward { .. } | NodeKind::SdpaBackwardOut { .. } => "SdpaBwd",
        NodeKind::Concat { .. } => "Concat",
        NodeKind::Slice { .. } => "Slice",
        NodeKind::Permute { .. } => "Permute",
        NodeKind::Reshape { .. } => "Reshape",
        NodeKind::BroadcastTo { .. } => "Broadcast",
        NodeKind::Cast { .. } => "Cast",
        NodeKind::Gather { .. } | NodeKind::IndexSelect { .. } => "Gather",
        NodeKind::ScatterAdd { .. } => "ScatterAdd",
        NodeKind::RotaryEmbedding { .. } => "Rope",
        NodeKind::RotaryEmbeddingBackward { .. } => "RopeBwd",
        NodeKind::LayerNorm { .. } => "LayerNorm",
        NodeKind::LayerNormBackward { .. } => "LayerNormBwd",
        NodeKind::LayerNormBackwardOut { .. } => "LayerNormOut",
        NodeKind::PositionEmbedding { .. } => "PosEmb",
        NodeKind::KvAttention { .. } => "KvAttention",
        NodeKind::KdaChunk { .. } => "KdaChunk",
        NodeKind::KdaRecurrence { .. } => "KdaRecurrence",
        NodeKind::KdaBackward { .. } | NodeKind::KdaBackwardOut { .. } => "KdaBwd",
        NodeKind::ShortConv1d { .. } => "ShortConv",
        NodeKind::ShortConv1dBackwardX { .. } | NodeKind::ShortConv1dBackwardW { .. } => {
            "ShortConvBwd"
        }
        NodeKind::ConvState { .. } => "ConvState",
        NodeKind::Sum { .. }
        | NodeKind::Mean { .. }
        | NodeKind::Max { .. }
        | NodeKind::Min { .. }
        | NodeKind::Prod { .. } => "Reduce",
        NodeKind::CrossEntropy { .. } | NodeKind::CrossEntropyBackward { .. } => "CE",
        NodeKind::AdamWStep { .. } | NodeKind::AdamWOut { .. } => "AdamW",
        NodeKind::AdamWStepGroup { .. } => "AdamWGroup",
        NodeKind::AdamWGroupOut { .. } => "AdamWGroupOut",
        NodeKind::SgdStep { .. } | NodeKind::SgdOut { .. } => "Sgd",
        NodeKind::Exp { .. }
        | NodeKind::Log { .. }
        | NodeKind::Sin { .. }
        | NodeKind::Cos { .. }
        | NodeKind::Tanh { .. }
        | NodeKind::Erf { .. }
        | NodeKind::Sqrt { .. }
        | NodeKind::Abs { .. }
        | NodeKind::Sign { .. }
        | NodeKind::Neg { .. }
        | NodeKind::Relu { .. }
        | NodeKind::Pow { .. }
        | NodeKind::Floor { .. }
        | NodeKind::Ceil { .. }
        | NodeKind::Round { .. } => "Unary",
        NodeKind::Maximum { .. } | NodeKind::Minimum { .. } => "MaxMin",
        NodeKind::Eq { .. }
        | NodeKind::Lt { .. }
        | NodeKind::Le { .. }
        | NodeKind::Gt { .. }
        | NodeKind::Ge { .. } => "Cmp",
        NodeKind::Where { .. } => "Where",
        NodeKind::Checkpoint { .. } | NodeKind::StopGradient { .. } => "Passthrough",
        NodeKind::Argmax { .. } | NodeKind::Argmin { .. } | NodeKind::Cumsum { .. } => "Scan",
        NodeKind::Inverse { .. } | NodeKind::Det { .. } | NodeKind::Solve { .. } => "Linalg",
        NodeKind::Leaf(_) | NodeKind::Input { .. } | NodeKind::ScalarInput { .. } => "Input",
        NodeKind::FromBytes { .. }
        | NodeKind::Zeros { .. }
        | NodeKind::Ones { .. }
        | NodeKind::Full { .. }
        | NodeKind::Randn { .. }
        | NodeKind::Uniform { .. }
        | NodeKind::Arange { .. }
        | NodeKind::Eye { .. } => "Const",
        _ => "Other",
    }
}

// RFC 0010: paged KV inference. A `NativeKvPool` is a fixed-capacity// arena of key/value rows per attention layer, allocated once per
// inference artifact; a `NativeKvSequence` is a block table and cursor
// over the pool (the OS paging model: blocks are pages, sequences are
// processes). `compile_decode` rewrites a traced forward graph for
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
}

impl BlockStore {
    fn new(num_blocks: usize) -> Self {
        Self {
            free: (0..num_blocks as u32).rev().collect(),
            refcounts: vec![0; num_blocks],
            hashes: vec![None; num_blocks],
            by_hash: HashMap::new(),
            lru: VecDeque::new(),
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

struct SeqState {
    // Absolute block table: blocks[i] holds positions [i*bs, (i+1)*bs).
    blocks: Vec<u32>,
    // Blocks below `head` are dead (fully below the attention window
    // frontier) and already returned to the pool.
    head: usize,
    cursor: usize,
    // Real new-token count of the current run (chunked prefill passes
    // padded chunks: q carries the chunk length, only `advance` rows
    // are real). Set by the run, consumed by KvAttention, added to the
    // cursor on completion.
    advance: usize,
    // Rolling chained hash of the last completed block, and the tokens
    // of the incomplete tail block accumulating toward the next one.
    last_hash: u64,
    pending: Vec<u32>,
    // RFC 0018: per-layer recurrent state, allocated lazily from the
    // decode geometry on first use — [H, Dk, Dv] f32 per KDA layer and
    // [K-1, C] f32 per short-conv layer.
    kda_states: Vec<runtime::metal::run::MetalTensor>,
    conv_states: Vec<runtime::metal::run::MetalTensor>,
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
                if let Some(&block) = self.blocks.get(block_index) {
                    pool.set_hash(block, hash);
                }
            }
        }
    }
}

// RFC 0018: uniform KDA layer geometry of a decode program.
#[derive(Clone, Copy, Default)]
struct KdaGeometry {
    layers: usize,
    heads: usize,
    head_dim: usize,
    value_dim: usize,
}

// RFC 0018: uniform short-conv layer geometry of a decode program.
#[derive(Clone, Copy, Default)]
struct ConvGeometry {
    layers: usize,
    channels: usize,
    kernel: usize,
}

struct KvContext {
    pool: Arc<PoolInner>,
    // One slot per leading batch element of the program's signature:
    // slot b owns batch row b of every KvAttention's q/k/v — its block
    // table, cursor, window, advance. Single-sequence runs have one
    // slot (RFC 0013).
    slots: Vec<Arc<Mutex<SeqState>>>,
    // Block tables + context lengths for the paged kernel, built once
    // per run (identical across layers: blocks settle at layer 0's
    // prepare, the cursor advances only at run end).
    paged_tables: Mutex<
        Option<(
            runtime::metal::run::MetalTensor,
            runtime::metal::run::MetalTensor,
        )>,
    >,
    kda: KdaGeometry,
    conv: ConvGeometry,
}

// RFC 0018: stateful KDA evaluation, one sequence slot per leading
// batch row. Each slot's [H, Dk, Dv] f32 state drives the chunked
// recurrence and is replaced by the final state.
#[allow(clippy::too_many_arguments)]
fn kda_recurrence(
    kv: &KvContext,
    layer: u32,
    q: &value::Value,
    k: &value::Value,
    v: &value::Value,
    log_decay: &value::Value,
    beta: &value::Value,
    scale: f64,
) -> err::Res<value::Value> {
    let geometry = kv.kda;
    if geometry.layers == 0 || (layer as usize) >= geometry.layers {
        return Err("kda recurrence: layer out of range for the decode geometry".to_string());
    }
    let dims = q.shape();
    let rank = dims.len();
    let t = dims[rank - 2];
    let batch: usize = dims[..rank - 3].iter().product();
    if batch != kv.slots.len() {
        return Err(format!(
            "kda recurrence: batch {batch} does not match {} decode slots",
            kv.slots.len()
        ));
    }
    let mut outs: Vec<runtime::metal::run::MetalTensor> = Vec::with_capacity(batch);
    for (b, slot) in kv.slots.iter().enumerate() {
        let narrow =
            |x: &value::Value, width: usize| -> err::Res<runtime::metal::run::MetalTensor> {
                let x = x.as_metal()?;
                let narrowed = metal_ops::contiguous(&runtime::metal::run::MetalTensor {
                    buffer: x.buffer.clone(),
                    layout: x.layout.narrow(0, b, 1),
                    dtype: x.dtype,
                })?;
                Ok(runtime::metal::run::MetalTensor {
                    buffer: narrowed.buffer.clone(),
                    layout: runtime::layout::Layout::contiguous(vec![geometry.heads, t, width]),
                    dtype: narrowed.dtype,
                })
            };
        let mut state = slot
            .lock()
            .map_err(|e| err::err_str(format!("kda recurrence: sequence lock poisoned: {e}")))?;
        while state.kda_states.len() < geometry.layers {
            state.kda_states.push(metal_ops::fill(
                &[geometry.heads, geometry.head_dim, geometry.value_dim],
                0.0,
                runtime::dtype::DType::F32,
            )?);
        }
        let qs = narrow(q, geometry.head_dim)?;
        let ks = narrow(k, geometry.head_dim)?;
        let vs = narrow(v, geometry.value_dim)?;
        let gs = narrow(log_decay, geometry.head_dim)?;
        let bs = narrow(beta, 1)?;
        // Chunked prefill right-pads: pad rows must contribute identity
        // updates (beta 0, log-decay 0) so the running state only
        // absorbs real tokens.
        let (gs, bs) = if state.advance < t {
            let mut mask = vec![0f32; t];
            mask[..state.advance].fill(1.0);
            let mask = runtime::metal::run::MetalTensor::from_f32(
                device::MetalDevice::get(),
                mask,
                vec![1, t, 1],
            );
            (
                metal_ops::binary(&gs, &mask, metal_ops::BinOp::Mul)?,
                metal_ops::binary(&bs, &mask, metal_ops::BinOp::Mul)?,
            )
        } else {
            (gs, bs)
        };
        // The T=1 decode step runs the fused register-resident
        // recurrence (RFC 0018 phase 3); chunked prefill keeps the
        // composed reference path.
        let in_dtype = q.as_metal()?.dtype;
        if t == 1
            && state.advance == 1
            && std::env::var_os("EFFECT_TORCH_NO_KDA_FUSED").is_none()
            && kda::is_supported(in_dtype, geometry.head_dim, geometry.value_dim)
        {
            let flat =
                |x: &runtime::metal::run::MetalTensor, w: usize| runtime::metal::run::MetalTensor {
                    buffer: x.buffer.clone(),
                    layout: runtime::layout::Layout::contiguous(vec![geometry.heads, w]),
                    dtype: x.dtype,
                };
            let out = kda::decode(
                &flat(&qs, geometry.head_dim),
                &flat(&ks, geometry.head_dim),
                &flat(&vs, geometry.value_dim),
                &flat(&gs, geometry.head_dim),
                &flat(&bs, 1),
                &state.kda_states[layer as usize],
                scale,
            )?;
            outs.push(metal_ops::contiguous(&runtime::metal::run::MetalTensor {
                buffer: out.buffer.clone(),
                layout: runtime::layout::Layout::contiguous(vec![
                    1,
                    geometry.heads,
                    1,
                    geometry.value_dim,
                ]),
                dtype: out.dtype,
            })?);
            continue;
        }
        let (out, final_state) = composed::kda_chunk_with_state(
            &qs,
            &ks,
            &vs,
            &gs,
            &bs,
            scale,
            &state.kda_states[layer as usize],
        )?;
        state.kda_states[layer as usize] = final_state;
        outs.push(metal_ops::contiguous(&runtime::metal::run::MetalTensor {
            buffer: out.buffer.clone(),
            layout: runtime::layout::Layout::contiguous(vec![
                1,
                geometry.heads,
                t,
                geometry.value_dim,
            ]),
            dtype: out.dtype,
        })?);
    }
    let mut acc = outs
        .first()
        .ok_or_else(|| "kda recurrence: empty batch".to_string())?
        .clone();
    for o in &outs[1..] {
        acc = metal_ops::cat(&acc, o, 0)?;
    }
    Ok(value::Value(acc))
}

// RFC 0018: stateful short-conv evaluation, one sequence slot per
// leading batch row. Each slot's [K-1, C] f32 window is shifted by the
// new tokens and written back.
fn conv_state(
    kv: &KvContext,
    layer: u32,
    x: &value::Value,
    weight: &value::Value,
) -> err::Res<value::Value> {
    let geometry = kv.conv;
    if geometry.layers == 0 || (layer as usize) >= geometry.layers {
        return Err("conv state: layer out of range for the decode geometry".to_string());
    }
    let dims = x.shape();
    let rank = dims.len();
    let t = dims[rank - 2];
    let batch: usize = dims[..rank - 2].iter().product();
    if batch != kv.slots.len() {
        return Err(format!(
            "conv state: batch {batch} does not match {} decode slots",
            kv.slots.len()
        ));
    }
    let in_dtype = x.as_metal()?.dtype;
    let w32 = metal_ops::to_f32(weight.as_metal()?)?;
    let mut outs: Vec<runtime::metal::run::MetalTensor> = Vec::with_capacity(batch);
    for (b, slot) in kv.slots.iter().enumerate() {
        let xs = {
            let x = x.as_metal()?;
            let narrowed = metal_ops::contiguous(&runtime::metal::run::MetalTensor {
                buffer: x.buffer.clone(),
                layout: x.layout.narrow(0, b, 1),
                dtype: x.dtype,
            })?;
            metal_ops::to_f32(&runtime::metal::run::MetalTensor {
                buffer: narrowed.buffer.clone(),
                layout: runtime::layout::Layout::contiguous(vec![t, geometry.channels]),
                dtype: narrowed.dtype,
            })?
        };
        let mut state = slot
            .lock()
            .map_err(|e| err::err_str(format!("conv state: sequence lock poisoned: {e}")))?;
        while state.conv_states.len() < geometry.layers {
            state.conv_states.push(metal_ops::fill(
                &[geometry.kernel - 1, geometry.channels],
                0.0,
                runtime::dtype::DType::F32,
            )?);
        }
        let (out, new_state) = composed::short_conv1d_with_state(
            &xs,
            &w32,
            &state.conv_states[layer as usize],
            state.advance,
        )?;
        state.conv_states[layer as usize] = new_state;
        let out = metal_ops::from_f32(&out, in_dtype)?;
        outs.push(metal_ops::contiguous(&runtime::metal::run::MetalTensor {
            buffer: out.buffer.clone(),
            layout: runtime::layout::Layout::contiguous(vec![1, t, geometry.channels]),
            dtype: out.dtype,
        })?);
    }
    let mut acc = outs
        .first()
        .ok_or_else(|| "conv state: empty batch".to_string())?
        .clone();
    for o in &outs[1..] {
        acc = metal_ops::cat(&acc, o, 0)?;
    }
    Ok(value::Value(acc))
}

// RoPE forward: x [.., T, D] (D even), one position offset per leading
// batch element (a single offset for unbatched graphs, one per slot in
// batched kv runs, RFC 0013). GPT-NeoX half-split rotation with
// theta^(-2j/D).

// The KvAttention eval: scatter q's companion k/v at [cursor, cursor+t)
// (allocating blocks as the frontier crosses them), then attend q
// causally over the last `window` positions of the gathered context
// (None: the whole context). The gather feeds the composed sdpa path —
// shape-polymorphic kernels, so no pipeline recompile as the context
// grows (a dedicated paged kernel is the throughput follow-up).
// Dispatches over the batch dim (RFC 0013): slot b of the kv context
// owns batch row b. The per-slot work (scatter, attend, evict) is
// identical; the rest of the graph runs batched in one walk.
fn kv_attention(
    kv: &KvContext,
    layer: u32,
    q: &value::Value,
    k: &value::Value,
    v: &value::Value,
    scale: f64,
    window: Option<usize>,
) -> err::Res<value::Value> {
    let dims = q.shape();
    let batch: usize = dims[..dims.len() - 3].iter().product();
    if batch != kv.slots.len() {
        return Err(format!(
            "kv attention: batch {batch} does not match {} kv slots",
            kv.slots.len()
        ));
    }
    // Metal: one fused scatter + one kernel launch attend all slots
    // over the pool slabs in place — no gather copy, decode and
    // chunked prefill alike (RFC 0013, stage 2). Everything else
    // falls back to the composed per-slot path.

    {
        let rank = dims.len();
        let (t, h, d) = (dims[rank - 2], dims[rank - 3], dims[rank - 1]);
        if paged::is_supported(q.as_metal()?, kv.pool.k[layer as usize].dtype(), d) {
            return kv_attention_paged(
                kv,
                layer,
                q.as_metal()?,
                k.as_metal()?,
                v.as_metal()?,
                scale,
                window,
                batch,
                t,
                h,
                d,
            );
        }
    }
    if batch == 1 {
        let mut state = kv.slots[0]
            .lock()
            .map_err(|e| err::err_str(format!("kv attention: sequence lock poisoned: {e}")))?;
        return kv_attention_slot(&kv.pool, &mut state, layer, q, k, v, scale, window);
    }
    let mut outs = Vec::with_capacity(batch);
    for (b, slot) in kv.slots.iter().enumerate() {
        let mut state = slot
            .lock()
            .map_err(|e| err::err_str(format!("kv attention: sequence lock poisoned: {e}")))?;
        let narrow = |x: &value::Value| -> err::Res<value::Value> {
            match x {
                value::Value(x) => {
                    let n = runtime::metal::run::MetalTensor {
                        buffer: x.buffer.clone(),
                        layout: x.layout.narrow(0, b, 1),
                        dtype: x.dtype,
                    };
                    Ok(value::Value(metal_ops::contiguous(&n)?))
                }
            }
        };
        outs.push(kv_attention_slot(
            &kv.pool,
            &mut state,
            layer,
            &narrow(q)?,
            &narrow(k)?,
            &narrow(v)?,
            scale,
            window,
        )?);
    }
    match &outs[0] {
        value::Value(_) => {
            let refs: Vec<&runtime::metal::run::MetalTensor> =
                outs.iter().map(|o| o.as_metal().unwrap()).collect();
            metal_ops::cat(refs[0], refs[1], 0).map(value::Value)
        }
    }
}

// Metal paged attention (RFC 0013, stage 2): per-slot prepare
// (validate, allocate), one fused scatter of every slot's new rows,
// one kernel launch attending over the slabs in place (per-row causal
// lengths cover decode and chunked prefill), then per-slot eviction.
#[allow(clippy::too_many_arguments)]
fn kv_attention_paged(
    kv: &KvContext,
    layer: u32,
    q: &runtime::metal::run::MetalTensor,
    k: &runtime::metal::run::MetalTensor,
    v: &runtime::metal::run::MetalTensor,
    scale: f64,
    window: Option<usize>,
    batch: usize,
    t: usize,
    h: usize,
    d: usize,
) -> err::Res<value::Value> {
    let pool = &kv.pool;
    let layer = layer as usize;
    let mut ctxlens = Vec::with_capacity(batch);
    let mut starts = Vec::with_capacity(batch);
    let mut advance = 0usize;
    for slot in kv.slots.iter() {
        let mut state = slot
            .lock()
            .map_err(|e| err::err_str(format!("kv attention: sequence lock poisoned: {e}")))?;
        advance = state.advance;
        let (_cursor, needed, start) = kv_prepare(pool, &mut state, layer, window, h, d, t)?;
        ctxlens.push(needed as u32);
        starts.push(start);
    }
    // Tables/ctxlens settle at layer 0; later layers reuse them.
    let mut cache = kv
        .paged_tables
        .lock()
        .map_err(|e| err::err_str(format!("kv attention: table cache lock poisoned: {e}")))?;
    if cache.is_none() {
        let mut tables: Vec<u32> = Vec::new();
        let mut max_blocks = 0usize;
        for slot in &kv.slots {
            let state = slot
                .lock()
                .map_err(|e| err::err_str(format!("kv attention: sequence lock poisoned: {e}")))?;
            max_blocks = max_blocks.max(state.blocks.len());
        }
        for slot in &kv.slots {
            let state = slot
                .lock()
                .map_err(|e| err::err_str(format!("kv attention: sequence lock poisoned: {e}")))?;
            tables.extend_from_slice(&state.blocks);
            tables.resize(tables.len() + (max_blocks - state.blocks.len()), 0);
        }
        *cache = Some((
            runtime::metal::run::MetalTensor {
                buffer: runtime::metal::device::MetalDevice::get().alloc_with_data_u32(&tables),
                layout: runtime::layout::Layout::contiguous(vec![batch, max_blocks]),
                dtype: runtime::dtype::DType::U32,
            },
            runtime::metal::run::MetalTensor {
                buffer: runtime::metal::device::MetalDevice::get().alloc_with_data_u32(&ctxlens),
                layout: runtime::layout::Layout::contiguous(vec![batch]),
                dtype: runtime::dtype::DType::U32,
            },
        ));
    }
    let (tables, ctxlens) = cache.as_ref().expect("populated above");
    let slab_dtype = pool.k[layer].dtype();
    let (k_scales, v_scales) = match slab_dtype {
        DType::U8 => (
            Some(pool.scales[2 * layer].metal()?),
            Some(pool.scales[2 * layer + 1].metal()?),
        ),
        _ => (None, None),
    };
    // One fused scatter for all slots, one attention kernel launch.
    paged::scatter(
        k,
        v,
        pool.k[layer].metal()?,
        pool.v[layer].metal()?,
        k_scales,
        v_scales,
        tables,
        ctxlens,
        pool.block_size,
        advance,
    )?;
    let out = paged::decode(
        q,
        pool.k[layer].metal()?,
        pool.v[layer].metal()?,
        k_scales,
        v_scales,
        tables,
        ctxlens,
        window,
        scale,
        pool.block_size,
        advance,
    )?;
    for (b, slot) in kv.slots.iter().enumerate() {
        let mut state = slot
            .lock()
            .map_err(|e| err::err_str(format!("kv attention: sequence lock poisoned: {e}")))?;
        kv_evict(pool, &mut state, starts[b]);
    }
    Ok(value::Value(out))
}

#[allow(clippy::too_many_arguments)]
fn kv_attention_slot(
    pool: &Arc<PoolInner>,
    state: &mut SeqState,
    layer: u32,
    q: &value::Value,
    k: &value::Value,
    v: &value::Value,
    scale: f64,
    window: Option<usize>,
) -> err::Res<value::Value> {
    if q.dtype() != runtime::dtype::DType::F32 {
        return Err(format!(
            "kv attention: dtype must be f32, got {:?}",
            q.dtype()
        ));
    }
    let layer = layer as usize;
    let dims = q.shape();
    let rank = dims.len();
    let (t, h, d) = (dims[rank - 2], dims[rank - 3], dims[rank - 1]);
    let (cursor, needed, start) = kv_prepare(pool, state, layer, window, h, d, t)?;
    kv_scatter_rows(pool, state, layer, k, v, h, d)?;
    let slab_dtype = pool.k[layer].dtype();
    let full = cursor + t;
    // Logical position -> physical row, through the sequence's block
    // table. The table is append-only: entries below `head` are dead
    // (already freed) and never queried, since gathers start at the
    // window frontier.
    let physical = |p: usize| -> u32 {
        state.blocks[p / pool.block_size] * pool.block_size as u32 + (p % pool.block_size) as u32
    };
    // Gather the attended context through the block table: positions
    // [start, needed) are real; rows past the real frontier are
    // zero-padded (only pad q rows, whose outputs are discarded, can
    // attend to them). The gather spans [start, cursor+t) so the
    // causal mask aligns for every q row.
    let ctx_rows: Vec<u32> = (start..needed).map(&physical).collect();
    let ctx = full - start;
    // Gather the context rows through the block table, dequantize when
    // needed, and attend through the composed fallback.
    let ctx_rows_t = runtime::metal::indexing::ids_from_host(
        runtime::metal::device::MetalDevice::get(),
        &ctx_rows,
    );
    let gather_rows =
        |slab: &PoolSlab, scale: Option<&PoolSlab>| -> err::Res<runtime::metal::run::MetalTensor> {
            let gathered = runtime::metal::indexing::index_select(
                runtime::metal::device::MetalDevice::get(),
                slab.metal()?,
                0,
                &ctx_rows_t,
            )?;
            let real = match scale {
                Some(scale) => {
                    let g32 = metal_ops::cast(&gathered, runtime::dtype::DType::F32)?;
                    let off = metal_ops::fill(g32.layout.shape(), 128.0, g32.dtype)?;
                    let centered = metal_ops::binary(&g32, &off, metal_ops::BinOp::Sub)?;
                    let scales = runtime::metal::indexing::index_select(
                        runtime::metal::device::MetalDevice::get(),
                        scale.metal()?,
                        0,
                        &ctx_rows_t,
                    )?;
                    let scales = runtime::metal::run::MetalTensor {
                        buffer: scales.buffer.clone(),
                        layout: runtime::layout::Layout::contiguous(vec![ctx_rows.len(), h, 1]),
                        dtype: scales.dtype,
                    };
                    metal_ops::binary(&centered, &scales, metal_ops::BinOp::Mul)?
                }
                None => metal_ops::cast(&gathered, runtime::dtype::DType::F32)?,
            };
            let pad = ctx - ctx_rows.len();
            let full = if pad > 0 {
                let zeros = metal_ops::fill(&[pad, h, d], 0.0, real.dtype)?;
                metal_ops::cat(&real, &zeros, 0)?
            } else {
                real
            };
            let permuted = metal_ops::permute(&full, &[1, 0, 2])?;
            let expanded = runtime::metal::run::MetalTensor {
                buffer: permuted.buffer.clone(),
                layout: runtime::layout::Layout::contiguous(vec![1, h, ctx, d]),
                dtype: permuted.dtype,
            };
            metal_ops::contiguous(&expanded)
        };
    let (k_scale, v_scale) = match slab_dtype {
        DType::U8 => (
            Some(&pool.scales[2 * layer]),
            Some(&pool.scales[2 * layer + 1]),
        ),
        _ => (None, None),
    };
    let kn = gather_rows(&pool.k[layer], k_scale)?;
    let vn = gather_rows(&pool.v[layer], v_scale)?;
    let q32 = metal_ops::to_f32(q.as_metal()?)?;
    let output = composed::sdpa_forward(&q32, &kn, &vn, scale, true)?;
    kv_evict(pool, state, start);
    Ok(value::Value(output))
}

// Validates the slot and allocates blocks up to the new frontier.
// Returns (cursor, needed, start): the pre-run cursor, the post-run
// frontier, and the attention window's start. Scatter of the new rows
// is `kv_scatter_rows` (composed path) or the fused kernel (paged).
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
    let needed = cursor + advance;
    // Live rows after this step: everything from the attention window
    // frontier on. Blocks fully below the frontier are dead and their
    // capacity is reclaimed, so a windowed sequence's footprint is
    // O(window) however long it generates.
    let full = cursor + t;
    let start = window.map_or(0, |w| full.saturating_sub(w));
    if needed - start > pool.max_tokens {
        return Err(format!(
            "kv attention: live context {} exceeds pool capacity {}",
            needed - start,
            pool.max_tokens
        ));
    }
    while state.blocks.len() * pool.block_size < needed {
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

// Composed scatter of the run's real new rows into the pool slabs
// (quantized when the slabs are int8). The paged path uses the fused
// scatter kernel instead.
fn kv_scatter_rows(
    pool: &Arc<PoolInner>,
    state: &mut SeqState,
    layer: usize,
    k: &value::Value,
    v: &value::Value,
    h: usize,
    d: usize,
) -> err::Res<()> {
    let slab_dtype = pool.k[layer].dtype();
    let cursor = state.cursor;
    let advance = state.advance;
    let needed = cursor + advance;
    // Logical position -> physical row, through the sequence's block
    // table (append-only; entries below `head` are dead and never
    // written).
    let physical = |p: usize| -> u32 {
        state.blocks[p / pool.block_size] * pool.block_size as u32 + (p % pool.block_size) as u32
    };
    let write_rows: Vec<u32> = (cursor..needed).map(&physical).collect();
    // The paged fallback computes rows on-device and scatters into slabs
    // using the physical block-table indices.
    let new_rows = |x: &value::Value| -> err::Res<runtime::metal::run::MetalTensor> {
        let x = x.as_metal()?;
        let p = metal_ops::permute(x, &[0, 2, 1, 3])?;
        let n = runtime::metal::run::MetalTensor {
            buffer: p.buffer.clone(),
            layout: p.layout.narrow(1, 0, advance),
            dtype: p.dtype,
        };
        let r = metal_ops::contiguous(&n)?;
        Ok(runtime::metal::run::MetalTensor {
            buffer: r.buffer,
            layout: runtime::layout::Layout::contiguous(vec![advance, h, d]),
            dtype: r.dtype,
        })
    };
    if slab_dtype == DType::U8 {
        let quantize = |x: &runtime::metal::run::MetalTensor| -> err::Res<(
            runtime::metal::run::MetalTensor,
            runtime::metal::run::MetalTensor,
        )> {
            let abs = metal_ops::unary(x, metal_ops::UnOp::Abs)?;
            let amax = metal_ops::reduce(&abs, &[2], true, fusion::ReduceOp::Max)?;
            let scale = metal_ops::binary(
                &amax,
                &metal_ops::fill(amax.layout.shape(), 127.0, amax.dtype)?,
                metal_ops::BinOp::Div,
            )?;
            let scale = metal_ops::binary(
                &scale,
                &metal_ops::fill(scale.layout.shape(), 1e-12, scale.dtype)?,
                metal_ops::BinOp::Add,
            )?;
            let q = metal_ops::binary(x, &scale, metal_ops::BinOp::Div)?;
            let q = metal_ops::binary(
                &q,
                &metal_ops::fill(q.layout.shape(), 128.0, q.dtype)?,
                metal_ops::BinOp::Add,
            )?;
            let q = metal_ops::unary(&q, metal_ops::UnOp::Round)?;
            let q = metal_ops::cast(&q, runtime::dtype::DType::U8)?;
            Ok((q, scale))
        };
        let (qk, sk) = quantize(&new_rows(k)?)?;
        let (qv, sv) = quantize(&new_rows(v)?)?;
        runtime::metal::indexing::scatter_set(
            runtime::metal::device::MetalDevice::get(),
            pool.k[layer].metal()?,
            0,
            &write_rows,
            &qk,
        )?;
        runtime::metal::indexing::scatter_set(
            runtime::metal::device::MetalDevice::get(),
            pool.v[layer].metal()?,
            0,
            &write_rows,
            &qv,
        )?;
        runtime::metal::indexing::scatter_set(
            runtime::metal::device::MetalDevice::get(),
            pool.scales[2 * layer].metal()?,
            0,
            &write_rows,
            &sk,
        )?;
        runtime::metal::indexing::scatter_set(
            runtime::metal::device::MetalDevice::get(),
            pool.scales[2 * layer + 1].metal()?,
            0,
            &write_rows,
            &sv,
        )?;
    } else {
        let nd = slab_dtype;
        let kr = metal_ops::cast(&new_rows(k)?, nd)?;
        let vr = metal_ops::cast(&new_rows(v)?, nd)?;
        runtime::metal::indexing::scatter_set(
            runtime::metal::device::MetalDevice::get(),
            pool.k[layer].metal()?,
            0,
            &write_rows,
            &kr,
        )?;
        runtime::metal::indexing::scatter_set(
            runtime::metal::device::MetalDevice::get(),
            pool.v[layer].metal()?,
            0,
            &write_rows,
            &vr,
        )?;
    }
    Ok(())
}

// again. The last reference lands them in the prefix cache — their
// content is still valid for a matching prompt.
fn kv_evict(pool: &PoolInner, state: &mut SeqState, start: usize) {
    while (state.head + 1) * pool.block_size <= start {
        let dead = state.blocks[state.head];
        pool.unref_block(dead);
        state.head += 1;
    }
}

struct DecodeGeometry {
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    kda: KdaGeometry,
    conv: ConvGeometry,
    cursor_slot: u32,
    // Batched programs only: whether a cursor [batch] tensor slot
    // exists — created when a learned PositionEmbedding is rewritten
    // (RoPE reads cursors from the kv context instead, RFC 0013).
    cursor_tensor: bool,
}

// The decode rewrite: same traced forward graph, cache-relevant nodes
// reinterpreted. Deterministic — the layer ordinal of each Sdpa is its
// order of first encounter in a post-order walk from the roots, so the
// prefill and decode traces of one model agree. `batch` is the
// graph's leading dim (RFC 0013): 1 for prefill/single decode (cursor
// binds as a scalar), N for batched decode (cursor becomes a [batch]
// tensor slot and position machinery is built per slot).
fn decode_rewrite(
    roots: &[Arc<Node>],
    window: Option<usize>,
    batch: usize,
) -> std::result::Result<(Vec<Arc<Node>>, DecodeGeometry), String> {
    let mut max_slot: Option<u32> = None;
    {
        let mut visited = HashSet::new();
        let mut stack: Vec<Arc<Node>> = roots.to_vec();
        while let Some(node) = stack.pop() {
            if !visited.insert(node.id) {
                continue;
            }
            match &node.kind {
                NodeKind::Input { slot, .. } => {
                    max_slot = Some(max_slot.map_or(*slot, |m: u32| m.max(*slot)))
                }
                NodeKind::ScalarInput { .. } => {
                    return Err(
                        "decode: runtime scalar inputs are not supported in inference graphs"
                            .to_string(),
                    )
                }
                _ => {}
            }
            stack.extend(node_children(&node.kind));
        }
    }
    let cursor_slot = max_slot.map_or(0, |m| m + 1);
    let mut visited = HashSet::new();
    let mut order = Vec::new();
    let mut stack: Vec<(Arc<Node>, bool)> = roots.iter().map(|r| (r.clone(), false)).collect();
    while let Some((node, processed)) = stack.pop() {
        if processed {
            order.push(node);
            continue;
        }
        if !visited.insert(node.id) {
            continue;
        }
        stack.push((node.clone(), true));
        for child in node_children(&node.kind) {
            stack.push((child, false));
        }
    }
    let mut map: HashMap<u64, Arc<Node>> = HashMap::new();
    let mut layers = 0usize;
    let mut kda_layers = 0usize;
    let mut conv_layers = 0usize;
    let mut cursor_tensor = false;
    let mut geometry: Option<(usize, usize)> = None;
    let mut kda_geometry: Option<(usize, usize, usize)> = None;
    let mut conv_geometry: Option<(usize, usize)> = None;
    for node in &order {
        let remap =
            |child: &Arc<Node>| map.get(&child.id).cloned().unwrap_or_else(|| child.clone());
        let rebuilt = match &node.kind {
            NodeKind::Sdpa {
                q,
                k,
                v,
                scale,
                causal,
            } => {
                if !causal {
                    return Err(
                        "decode: only causal attention is cacheable, found a non-causal sdpa"
                            .to_string(),
                    );
                }
                let rank = k.shape.len();
                if rank != 4 || k.shape[..rank - 3].iter().product::<usize>() != batch {
                    return Err(format!(
                        "decode: kv caching expects attention of shape [{batch}, H, T, D], got {:?}",
                        k.shape
                    ));
                }
                let (heads, dim) = (k.shape[rank - 3], k.shape[rank - 1]);
                match geometry {
                    Some((h0, d0)) if h0 != heads || d0 != dim => {
                        return Err(format!(
                            "decode: attention layers disagree on head geometry ([{h0}, {d0}] vs [{heads}, {dim}])"
                        ));
                    }
                    None => geometry = Some((heads, dim)),
                    _ => {}
                }
                let layer = layers;
                layers += 1;
                NodeKind::KvAttention {
                    q: remap(q),
                    k: remap(k),
                    v: remap(v),
                    scale: *scale,
                    layer: layer as u32,
                    window,
                }
            }
            NodeKind::KdaChunk {
                q,
                k,
                v,
                log_decay,
                beta,
                scale,
            } => {
                let rank = q.shape.len();
                if rank != 4 || q.shape[..rank - 3].iter().product::<usize>() != batch {
                    return Err(format!(
                        "decode: kda state caching expects layers of shape [{batch}, H, T, D], got {:?}",
                        q.shape
                    ));
                }
                let current = (q.shape[rank - 3], q.shape[rank - 1], v.shape[rank - 1]);
                match kda_geometry {
                    Some(previous) if previous != current => {
                        return Err(format!(
                            "decode: kda layers disagree on head geometry ({previous:?} vs {current:?})"
                        ));
                    }
                    None => kda_geometry = Some(current),
                    _ => {}
                }
                let layer = kda_layers;
                kda_layers += 1;
                NodeKind::KdaRecurrence {
                    q: remap(q),
                    k: remap(k),
                    v: remap(v),
                    log_decay: remap(log_decay),
                    beta: remap(beta),
                    scale: *scale,
                    layer: layer as u32,
                }
            }
            NodeKind::ShortConv1d { x, weight } => {
                let rank = x.shape.len();
                if rank != 3 || x.shape[..rank - 2].iter().product::<usize>() != batch {
                    return Err(format!(
                        "decode: conv state caching expects layers of shape [{batch}, T, C], got {:?}",
                        x.shape
                    ));
                }
                let current = (x.shape[rank - 1], weight.shape[1]);
                match conv_geometry {
                    Some(previous) if previous != current => {
                        return Err(format!(
                            "decode: short conv layers disagree on geometry ({previous:?} vs {current:?})"
                        ));
                    }
                    None => conv_geometry = Some(current),
                    _ => {}
                }
                let layer = conv_layers;
                conv_layers += 1;
                NodeKind::ConvState {
                    x: remap(x),
                    weight: remap(weight),
                    layer: layer as u32,
                }
            }
            NodeKind::RotaryEmbedding {
                x,
                seq_len,
                theta,
                offset,
            } => NodeKind::RotaryEmbedding {
                x: remap(x),
                seq_len: *seq_len,
                theta: *theta,
                offset: match offset {
                    PositionOffset::Absolute => PositionOffset::Cursor,
                    PositionOffset::Cursor => PositionOffset::Cursor,
                },
            },
            NodeKind::PositionEmbedding { weight, seq_len } => {
                let t = *seq_len;
                let e = weight.shape[1];
                let device = weight.device.clone();
                if batch > 1 {
                    // Batched (RFC 0013): per-slot cursors arrive as a
                    // [batch] tensor; flat gather indices cursors[b] + p
                    // over the [maxPositions, E] table, reshaped back.
                    cursor_tensor = true;
                    let cursors = Node::new(NodeKind::Input {
                        slot: cursor_slot,
                        shape: vec![batch],
                        dtype: DType::I64,
                        device: device.clone(),
                    })?;
                    let positions = Node::new(NodeKind::Add {
                        a: Node::new(NodeKind::Reshape {
                            a: cursors,
                            shape: vec![batch, 1],
                        })?,
                        b: Node::new(NodeKind::BroadcastTo {
                            a: Node::new(NodeKind::Reshape {
                                a: Node::new(NodeKind::Arange {
                                    start: 0.0,
                                    end: t as f64,
                                    step: 1.0,
                                    dtype: DType::I64,
                                    device: device.clone(),
                                })?,
                                shape: vec![1, t],
                            })?,
                            shape: vec![batch, t],
                        })?,
                    })?;
                    let indexes = Node::new(NodeKind::BroadcastTo {
                        a: Node::new(NodeKind::Reshape {
                            a: positions,
                            shape: vec![batch * t, 1],
                        })?,
                        shape: vec![batch * t, e],
                    })?;
                    NodeKind::Reshape {
                        a: Node::new(NodeKind::Gather {
                            a: remap(weight),
                            dim: 0,
                            indexes,
                        })?,
                        shape: vec![batch, t, e],
                    }
                } else {
                    let positions = Node::new(NodeKind::Add {
                        a: Node::new(NodeKind::Arange {
                            start: 0.0,
                            end: t as f64,
                            step: 1.0,
                            dtype: DType::I64,
                            device: device.clone(),
                        })?,
                        b: Node::new(NodeKind::ScalarInput {
                            slot: cursor_slot,
                            dtype: DType::I64,
                            device: device.clone(),
                        })?,
                    })?;
                    let indexes = Node::new(NodeKind::BroadcastTo {
                        a: Node::new(NodeKind::Reshape {
                            a: positions,
                            shape: vec![t, 1],
                        })?,
                        shape: vec![t, e],
                    })?;
                    NodeKind::Gather {
                        a: remap(weight),
                        dim: 0,
                        indexes,
                    }
                }
            }
            kind => remap_children(kind, &remap),
        };
        map.insert(node.id, Node::new(rebuilt)?);
    }
    if layers == 0 && kda_layers == 0 {
        return Err(
            "decode: model has no cacheable attention or recurrent layers (no causal sdpa or kda chunk node in the forward graph)"
                .to_string(),
        );
    }
    let (kv_heads, head_dim) = geometry.unwrap_or((0, 0));
    let kda = kda_geometry
        .map(|(heads, head_dim, value_dim)| KdaGeometry {
            layers: kda_layers,
            heads,
            head_dim,
            value_dim,
        })
        .unwrap_or_default();
    let conv = conv_geometry
        .map(|(channels, kernel)| ConvGeometry {
            layers: conv_layers,
            channels,
            kernel,
        })
        .unwrap_or_default();
    let roots = roots
        .iter()
        .map(|r| map.get(&r.id).cloned().unwrap_or_else(|| r.clone()))
        .collect();
    Ok((
        roots,
        DecodeGeometry {
            layers,
            kv_heads,
            head_dim,
            kda,
            conv,
            cursor_slot,
            cursor_tensor,
        },
    ))
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
        // Zero layers is the pure-recurrent (KDA-only) stack: no KV
        // slabs, but sequences and block hashing still work.
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
        Ok(Self {
            inner: Arc::new(PoolInner {
                k,
                v,
                scales,
                kv_heads,
                head_dim,
                block_size,
                max_tokens,
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
    pub fn make_sequence(&self) -> NativeKvSequence {
        NativeKvSequence {
            pool: self.inner.clone(),
            state: Arc::new(Mutex::new(SeqState {
                blocks: Vec::new(),
                head: 0,
                cursor: 0,
                advance: 0,
                last_hash: HASH_SEED,
                pending: Vec::new(),
                kda_states: Vec::new(),
                conv_states: Vec::new(),
            })),
            run_lock: Arc::new(Mutex::new(())),
            released: AtomicBool::new(false),
        }
    }
}

#[napi(custom_finalize)]
pub struct NativeKvSequence {
    pool: Arc<PoolInner>,
    state: Arc<Mutex<SeqState>>,
    // Serializes runs of this sequence; other sequences run concurrently
    // (their blocks are disjoint by allocation).
    run_lock: Arc<Mutex<()>>,
    released: AtomicBool,
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
            state: Arc::new(Mutex::new(SeqState {
                blocks: Vec::new(),
                head: 0,
                cursor: 0,
                advance: 0,
                last_hash: HASH_SEED,
                pending: Vec::new(),
                kda_states: Vec::new(),
                conv_states: Vec::new(),
            })),
            run_lock: Arc::new(Mutex::new(())),
            released: AtomicBool::new(false),
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
        let Ok(_run_guard) = self.run_lock.lock() else {
            return;
        };
        if let Ok(mut state) = self.state.lock() {
            // Blocks below head were evicted already; draining them
            // again would double-free.
            let head = state.head;
            let blocks = state.blocks.split_off(head);
            for block in blocks {
                self.pool.unref_block(block);
            }
            state.cursor = 0;
            state.advance = 0;
            state.last_hash = HASH_SEED;
            state.pending.clear();
        }
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
    // result. Sharing is content-addressed: two prompts that merely
    // begin alike share; nothing about the match is visible to callers.
    #[napi]
    pub fn prefill_match(&self, tokens: Vec<u32>) -> Result<u32> {
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
        state.cursor = state.blocks.len() * block_size;
        Ok(state.cursor as u32)
    }
}

#[napi]
pub struct DecodeProgram {
    inner: ProgramInner,
    cursor_slot: u32,
    layers: u32,
    kv_heads: u32,
    head_dim: u32,
    kda: KdaGeometry,
    conv: ConvGeometry,
    batch: u32,
    cursor_tensor: bool,
}

#[napi]
impl DecodeProgram {
    #[napi(getter)]
    pub fn batch(&self) -> u32 {
        self.batch
    }
    #[napi(getter)]
    pub fn signature(&self) -> Result<String> {
        Ok(self.inner.signature.clone())
    }

    #[napi(getter)]
    pub fn layers(&self) -> u32 {
        self.layers
    }

    #[napi(getter)]
    pub fn kv_heads(&self) -> u32 {
        self.kv_heads
    }

    #[napi(getter)]
    pub fn head_dim(&self) -> u32 {
        self.head_dim
    }

    #[napi(getter)]
    pub fn kda_layers(&self) -> u32 {
        self.kda.layers as u32
    }

    #[napi(getter)]
    pub fn kda_heads(&self) -> u32 {
        self.kda.heads as u32
    }

    #[napi(getter)]
    pub fn kda_head_dim(&self) -> u32 {
        self.kda.head_dim as u32
    }

    #[napi(getter)]
    pub fn kda_value_dim(&self) -> u32 {
        self.kda.value_dim as u32
    }

    #[napi(getter)]
    pub fn conv_layers(&self) -> u32 {
        self.conv.layers as u32
    }

    #[napi(getter)]
    pub fn conv_channels(&self) -> u32 {
        self.conv.channels as u32
    }

    #[napi(getter)]
    pub fn conv_kernel(&self) -> u32 {
        self.conv.kernel as u32
    }

    // Runs the frozen decode/prefill graph against a sequence: the
    // cursor scalar binds from the sequence state, the kv context
    // carries the pool and block table, and the cursor advances by the
    // real token count on completion — `tokens` carries the REAL new
    // tokens of this run (1 for decode; the un-padded tokens of each
    // chunk for prefill, whose inputs are padded to the fixed chunk
    // shape). The tokens also feed the prefix cache's block hashes.
    #[napi]
    pub async fn run(
        &self,
        inputs: Vec<&NativeTensor>,
        seq: &NativeKvSequence,
        tokens: Vec<u32>,
        token: Option<&CancellationToken>,
    ) -> Result<Vec<NativeTensor>> {
        if self.batch != 1 {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "kv run: this program is batched (batch {}), use run_batched",
                    self.batch
                ),
            ));
        }
        self.run_inner(inputs, vec![seq], vec![tokens], token).await
    }

    // RFC 0013: one run stepping `batch` sequences one chunk each.
    // Slot b of the kv context owns batch row b — its block table,
    // cursor, advance — and the cursors bind as a [batch] tensor slot
    // (batched programs carry no scalar). Run locks are taken in
    // address order, so overlapping batches cannot deadlock; duplicate
    // sequences in one batch are an error.
    #[napi]
    pub async fn run_batched(
        &self,
        inputs: Vec<&NativeTensor>,
        seqs: Vec<&NativeKvSequence>,
        tokens: Vec<Vec<u32>>,
        token: Option<&CancellationToken>,
    ) -> Result<Vec<NativeTensor>> {
        // Ragged batches pad internally: throwaway sequences (token 0,
        // outputs discarded) fill the program's fixed width, and their
        // blocks are released before the call returns — callers never see
        // the padding.
        if seqs.is_empty() || seqs.len() > self.batch as usize {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "kv run: program accepts 1..={} sequences, got {}",
                    self.batch,
                    seqs.len()
                ),
            ));
        }
        let pad: Vec<NativeKvSequence> = (seqs.len()..self.batch as usize)
            .map(|_| seqs[0].new_sequence_like())
            .collect();
        let mut all: Vec<&NativeKvSequence> = seqs;
        all.extend(pad.iter());
        let mut all_tokens = tokens;
        let advance = all_tokens.first().map(|t| t.len()).unwrap_or(1);
        all_tokens.extend(std::iter::repeat_n(vec![0u32; advance], pad.len()));
        // The ids input is fixed-width too: pad the caller's last input
        // ([n, T] decode ids) with zero rows up to the program width.
        let mut owned_inputs: Vec<NativeTensor> = inputs
            .iter()
            .map(|t| t.val_cloned().map(NativeTensor::wrap))
            .collect::<Result<Vec<_>>>()?;
        if let Some(last) = owned_inputs.last_mut() {
            let last_val = last.val_cloned()?;
            let shape = last_val.shape();
            if shape.len() == 2 && shape[0] < self.batch as usize {
                let pad_rows = self.batch as usize - shape[0];
                let zeros = match &last_val {
                    value::Value(_) => value::Value(
                        metal_ops::fill(&[pad_rows, shape[1]], 0.0, last_val.dtype())
                            .map_err(to_napi_err)?,
                    ),
                };
                #[allow(unreachable_patterns)]
                let padded = match (&last_val, &zeros) {
                    (value::Value(a), value::Value(b)) => {
                        value::Value(metal_ops::cat(a, b, 0).map_err(to_napi_err)?)
                    }
                    _ => {
                        return Err(Error::new(
                            Status::GenericFailure,
                            "decode: padding tensors are on different devices".to_string(),
                        ))
                    }
                };
                last.slot = std::sync::Arc::new(LeafSlot::new(padded));
            }
        }
        let refs: Vec<&NativeTensor> = owned_inputs.iter().collect();
        let out = self.run_inner(refs, all, all_tokens, token).await;
        for p in &pad {
            p.release();
        }
        out
    }
}

impl DecodeProgram {
    async fn run_inner(
        &self,
        inputs: Vec<&NativeTensor>,
        seqs: Vec<&NativeKvSequence>,
        tokens: Vec<Vec<u32>>,
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
        let inner = &self.inner;
        let tensor_count = inner.slots.iter().filter(|s| !s.scalar).count();
        let caller_inputs = tensor_count - usize::from(self.cursor_tensor);
        if inputs.len() != caller_inputs {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "program expected {caller_inputs} tensor inputs, got {}",
                    inputs.len()
                ),
            ));
        }
        for (slot, declared) in inner.slots.iter().enumerate() {
            if declared.scalar || (self.cursor_tensor && slot as u32 == self.cursor_slot) {
                continue;
            }
            let input_index = slot
                - inner.slots.iter().take(slot).filter(|s| s.scalar).count()
                - usize::from(self.cursor_tensor && (slot as u32) > self.cursor_slot);
            let got = inputs[input_index].val_cloned()?;
            if got.shape() != declared.shape.as_slice()
                || got.dtype() != declared.dtype
                || device_key(&got.device()) != device_key(&declared.device)
            {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!(
                        "input slot {slot}: expected {}, got {:?}",
                        declared.signature(),
                        got.shape()
                    ),
                ));
            }
        }
        let slots = inner.slots.clone();
        let roots = inner.roots.clone();
        let leaves = inner.leaves.clone();
        let inputs: Vec<value::Value> = inputs
            .iter()
            .map(|input| input.val_cloned())
            .collect::<Result<Vec<_>>>()?;
        let kv = Arc::new(KvContext {
            pool: seqs[0].pool.clone(),
            slots: seqs.iter().map(|seq| seq.state.clone()).collect(),

            paged_tables: Mutex::new(None),
            kda: self.kda,
            conv: self.conv,
        });
        // Lock every sequence in address order; overlapping batches
        // acquire the same locks in the same order, so no deadlock.
        let mut ordered: Vec<&NativeKvSequence> = seqs.clone();
        ordered.sort_by_key(|seq| Arc::as_ptr(&seq.run_lock) as usize);
        let run_locks: Vec<Arc<Mutex<()>>> =
            ordered.iter().map(|seq| seq.run_lock.clone()).collect();
        let slot_states: Vec<Arc<Mutex<SeqState>>> =
            seqs.iter().map(|seq| seq.state.clone()).collect();
        let cursor_slot = self.cursor_slot;
        let batched = self.batch > 1;
        let cursor_tensor = self.cursor_tensor;
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
            let _guard = metal_eval_guard(&roots);
            for (i, state) in slot_states.iter().enumerate() {
                state
                    .lock()
                    .map_err(|e| {
                        Error::new(
                            Status::GenericFailure,
                            format!("kv sequence lock poisoned: {e}"),
                        )
                    })?
                    .advance = tokens[i].len();
            }
            // Bindings are built inside the eval guard: scalar_binding
            // allocates on the device, which is not safe to do
            // concurrently with another walk on Metal.
            let mut bindings = std::collections::HashMap::new();
            let mut tensors = inputs.iter();
            for (slot, declared) in slots.iter().enumerate() {
                let binding = if declared.scalar {
                    if batched || slot as u32 != cursor_slot {
                        return Err(Error::new(
                            Status::GenericFailure,
                            format!("decode: unexpected scalar slot {slot}"),
                        ));
                    }
                    let cursor = slot_states[0]
                        .lock()
                        .map_err(|e| {
                            Error::new(
                                Status::GenericFailure,
                                format!("kv sequence lock poisoned: {e}"),
                            )
                        })?
                        .cursor;
                    scalar_binding(cursor as f64, declared.dtype, &declared.device)
                        .map_err(to_napi_err)?
                } else if cursor_tensor && slot as u32 == cursor_slot {
                    let mut cursors = Vec::with_capacity(batch);
                    for state in &slot_states {
                        cursors.push(
                            state
                                .lock()
                                .map_err(|e| {
                                    Error::new(
                                        Status::GenericFailure,
                                        format!("kv sequence lock poisoned: {e}"),
                                    )
                                })?
                                .cursor as i64,
                        );
                    }
                    let data: Vec<u8> = cursors
                        .iter()
                        .flat_map(|cursor| cursor.to_le_bytes())
                        .collect();
                    value::Value(runtime::metal::run::MetalTensor {
                        buffer: runtime::metal::device::MetalDevice::get().upload_bytes(&data),
                        layout: runtime::layout::Layout::contiguous(vec![batch]),
                        dtype: runtime::dtype::DType::I64,
                    })
                } else {
                    tensors.next().expect("tensor count checked").clone()
                };
                bindings.insert(slot as u64, binding);
            }
            let by_id: std::collections::HashMap<u64, value::Value> = leaves
                .iter()
                .map(|(id, slot)| Ok((*id, bindings[&(*slot as u64)].clone())))
                .collect::<err::Res<_>>()
                .map_err(to_napi_err)?;
            // Blocks allocated by a failed run roll back: the cursor did
            // not advance, so every block beyond the pre-run frontier is
            // unreferenced and returns to the pool (a poisoned sequence
            // must not take the pool down with it).
            let frontiers: Vec<usize> = slot_states
                .iter()
                .map(|state| state.lock().map(|s| s.blocks.len()).unwrap_or(0))
                .collect();
            // RFC 0018: recurrent state mutates in place during the
            // walk, so a failed run restores a deep pre-run snapshot
            // (KV blocks roll back by refcount instead).
            let snapshot = |states: &[runtime::metal::run::MetalTensor]| -> err::Res<
                Vec<runtime::metal::run::MetalTensor>,
            > {
                states
                    .iter()
                    .map(|t| kernels::strided_copy(device::MetalDevice::get(), t))
                    .collect()
            };
            let mut kda_snapshots = Vec::with_capacity(slot_states.len());
            let mut conv_snapshots = Vec::with_capacity(slot_states.len());
            for state in &slot_states {
                let state = state.lock().map_err(|e| {
                    Error::new(
                        Status::GenericFailure,
                        format!("kv sequence lock poisoned: {e}"),
                    )
                })?;
                kda_snapshots.push(snapshot(&state.kda_states).map_err(to_napi_err)?);
                conv_snapshots.push(snapshot(&state.conv_states).map_err(to_napi_err)?);
            }
            let rollback = || {
                for (i, state) in slot_states.iter().enumerate() {
                    if let Ok(mut state) = state.lock() {
                        for block in state.blocks.split_off(frontiers[i]) {
                            kv.pool.unref_block(block);
                        }
                        state.advance = 0;
                        state.kda_states = kda_snapshots[i].clone();
                        state.conv_states = conv_snapshots[i].clone();
                    }
                }
            };
            let mut ev = Evaluator::with_kv(&roots, by_id, Some(kv.clone()));
            let mut outputs = Vec::with_capacity(roots.len());
            for node in &roots {
                let output = match eval_node(node, cancelled, &mut ev) {
                    Ok(output) => output,
                    Err(error) => {
                        rollback();
                        return Err(to_napi_err(error));
                    }
                };
                outputs.push(NativeTensor::wrap(output));
            }
            // Synchronize once after all roots have been encoded.
            if let Some(first) = outputs.first() {
                let synchronized = first
                    .val_cloned()
                    .and_then(|value| value.synchronize().map_err(to_napi_err));
                if let Err(error) = synchronized {
                    rollback();
                    return Err(error);
                }
            }
            if let Err(error) = ev.run_ce_checks() {
                rollback();
                return Err(to_napi_err(error));
            }
            if !cancellation.complete() {
                rollback();
                return Err(Error::new(
                    Status::Cancelled,
                    "operation aborted".to_string(),
                ));
            }
            for (i, state) in slot_states.iter().enumerate() {
                if let Ok(mut state) = state.lock() {
                    state.note_tokens(&kv.pool, &tokens[i]);
                    state.cursor += state.advance;
                    state.advance = 0;
                }
            }
            Ok(outputs)
        })
        .await
    }
}

// RFC 0010: compiles a traced forward graph for generation: rewrites
// causal attention into paged kv attention (optionally sliding-window
// over the last `window` positions) and position embeddings into
// cursor-offset gathers (adding one cursor slot), fuses, and freezes.
// Fails when the graph has no causal sdpa node (nothing to cache) or
// uses runtime scalars (unsupported in inference graphs). `batch`
// (RFC 0013, default 1) traces a batched decode program: the graph's
// leading dim is the batch, the cursor becomes a [batch] tensor slot,
// and position machinery is built per slot.
#[napi]
pub fn compile_decode(
    roots: Vec<&LazyTensor>,
    window: Option<u32>,
    batch: Option<u32>,
) -> Result<DecodeProgram> {
    let nodes: Vec<Arc<Node>> = roots.iter().map(|t| t.node.clone()).collect();
    if nodes.is_empty() {
        return Err(Error::new(
            Status::InvalidArg,
            "compile_decode: expected at least one root".to_string(),
        ));
    }
    let batch = batch.unwrap_or(1);
    if batch == 0 {
        return Err(Error::new(
            Status::InvalidArg,
            "compile_decode: batch must be positive".to_string(),
        ));
    }
    let (nodes, geometry) = decode_rewrite(&nodes, window.map(|w| w as usize), batch as usize)
        .map_err(|e| Error::new(Status::GenericFailure, e))?;
    let nodes = fuse_roots(&nodes).map_err(|e| Error::new(Status::GenericFailure, e))?;
    let (slots, leaves) =
        collect_program_slots(&nodes).map_err(|e| Error::new(Status::InvalidArg, e))?;
    let signature = slots
        .iter()
        .map(|slot| slot.signature())
        .collect::<Vec<_>>()
        .join(",");
    Ok(DecodeProgram {
        inner: ProgramInner {
            roots: nodes,
            slots,
            leaves,
            signature,

            arena: Arc::new(Mutex::new(ArenaState::default())),
        },
        cursor_slot: geometry.cursor_slot,
        layers: geometry.layers as u32,
        kv_heads: geometry.kv_heads as u32,
        head_dim: geometry.head_dim as u32,
        kda: geometry.kda,
        conv: geometry.conv,
        batch,
        cursor_tensor: geometry.cursor_tensor,
    })
}

// RFC 0008: a frozen, reusable graph executable. `compile` traces slot
// declarations out of the root DAG, fuses once, and stores the immutable
// post-fusion roots; `run` rebinds Input/ScalarInput leaves to call
// arguments and evaluates with the same per-call Evaluator as eval_lazy.
// A program holds no device buffers beyond the constant/parameter leaves
// the traced graph already referenced.

struct ProgramInner {
    roots: Vec<Arc<Node>>,
    slots: Vec<ProgramSlot>,
    // Placeholder node id -> slot index, collected once at freeze time.
    leaves: Vec<(u64, u32)>,
    signature: String,
    // RFC 0016: the arena plan learned from the first run's allocation
    // sequence and replayed by later runs. `captured` suppresses repeat
    // capture after a divergence.
    arena: Arc<Mutex<ArenaState>>,
}

#[derive(Default)]
struct ArenaState {
    captured: bool,
    plan: Option<Arc<runtime::metal::arena::Plan>>,
}

#[napi]
pub struct CompiledProgram {
    inner: ProgramInner,
}

fn scalar_binding(value: f64, dtype: DType, _device: &Device) -> err::Res<value::Value> {
    Ok(value::Value(metal_ops::fill(&[], value, dtype)?))
}

#[napi]
impl CompiledProgram {
    #[napi(getter)]
    pub fn signature(&self) -> Result<String> {
        Ok(self.inner.signature.clone())
    }

    // Drops the frozen graphs; constant/parameter leaf buffers stay alive
    // through any NativeTensor handles that share them. Running a disposed
    // program is an error.
    #[napi]
    pub async fn run(
        &self,
        inputs: Vec<&NativeTensor>,
        scalars: Vec<f64>,
        token: Option<&CancellationToken>,
    ) -> Result<Vec<NativeTensor>> {
        let inner = &self.inner;
        let tensor_count = inner.slots.iter().filter(|s| !s.scalar).count();
        let scalar_count = inner.slots.len() - tensor_count;
        if inputs.len() != tensor_count {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "program expected {tensor_count} tensor inputs, got {}",
                    inputs.len()
                ),
            ));
        }
        if scalars.len() != scalar_count {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "program expected {scalar_count} scalar inputs, got {}",
                    scalars.len()
                ),
            ));
        }
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
        let slots = inner.slots.clone();
        let roots = inner.roots.clone();
        let leaves = inner.leaves.clone();

        let arena_state = inner.arena.clone();

        let arena_eligible = true;
        let inputs: Vec<value::Value> = inputs
            .iter()
            .map(|input| input.val_cloned())
            .collect::<Result<Vec<_>>>()?;
        run_compute(token, move |cancelled, _state| {
        let _guard = metal_eval_guard(&roots);
        // RFC 0016: replay the planned arena if one exists; otherwise
        // capture this run's allocation sequence to build one.

        let (plan, capturing) = {
            let state = arena_state.lock().unwrap();
            (state.plan.clone(), state.plan.is_none() && !state.captured && arena_eligible)
        };

        let replaying = plan.is_some();

        if let Some(plan) = plan {
            runtime::metal::arena::replay_begin(plan);
        }

        if capturing {
            runtime::metal::arena::capture_begin();
        }
        // Clears the arena session on error paths that skip the explicit
        // end below.

        let _arena_session = runtime::metal::arena::SessionGuard::new();
        // Bindings are built inside the eval guard: scalar_binding
        // allocates on the device, which is not safe to do concurrently
        // with another walk on Metal.
        let mut bindings = std::collections::HashMap::new();
        let mut tensors = inputs.iter();
        let mut scalars_iter = scalars.iter();
        for (slot, declared) in slots.iter().enumerate() {
            let binding = if declared.scalar {
                let value = scalars_iter.next().expect("scalar count checked");
                scalar_binding(*value, declared.dtype, &declared.device).map_err(to_napi_err)?
            } else {
                tensors.next().expect("tensor count checked").clone()
            };
            bindings.insert(slot as u64, binding);
        }
        let by_id: std::collections::HashMap<u64, value::Value> = leaves
            .iter()
            .map(|(id, slot)| {
                Ok((*id, bindings[&(*slot as u64)].clone()))
            })
            .collect::<err::Res<_>>().map_err(to_napi_err)?;
        let mut ev = Evaluator::with_slots(&roots, by_id);
            if std::env::var_os("EFFECT_TORCH_EVAL_STATS").is_some() {
                let mut counts: HashMap<&'static str, usize> = HashMap::new();
                let mut seen = std::collections::HashSet::new();
                let mut stack: Vec<Arc<Node>> = roots.clone();
                while let Some(node) = stack.pop() {
                    if !seen.insert(node.id) {
                        continue;
                    }
                    *counts.entry(node_kind_name(&node.kind)).or_insert(0) += 1;
                    stack.extend(node_children(&node.kind));
                }
                let mut entries: Vec<_> = counts.into_iter().collect();
                entries.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
                eprintln!(
                    "[program-stats] {} nodes: {}",
                    seen.len(),
                    entries.iter().map(|(k, n)| format!("{k}×{n}")).collect::<Vec<_>>().join(" ")
                );
            }
            let walk_timing = std::env::var_os("EFFECT_TORCH_WALK_TIMING").is_some();
            let t1 = std::time::Instant::now();
            let mut outputs = Vec::with_capacity(roots.len());
            for node in &roots {
                let output = eval_node(node, cancelled, &mut ev).map_err(to_napi_err)?;
                outputs.push(NativeTensor::wrap(output));
            }
            let t_encode = t1.elapsed();
            // Synchronize once after all roots have been encoded. Consumers that need values on
            // the host synchronize at readback; device-side reuse needs
            // no host round-trip. Device-global: one call.
            if let Some(first) = outputs.first() {
                first.val_cloned()?.synchronize().map_err(to_napi_err)?;
            }
            let t_sync = t1.elapsed() - t_encode;
            ev.run_ce_checks().map_err(to_napi_err)?;

            {
                if replaying {
                    if runtime::metal::arena::replay_end() {
                        let mut state = arena_state.lock().unwrap();
                        state.plan = None;
                        state.captured = true;
                        if std::env::var_os("EFFECT_TORCH_ARENA_DEBUG").is_some() {
                            eprintln!("[arena] replay diverged from plan; arena disabled for this program");
                        }
                    }
                } else if capturing {
                    let live = ev.live_buffer_keys();
                    let captured = runtime::metal::arena::capture_end(&live);
                    let plan = runtime::metal::arena::plan(&captured, &|size| {
                        runtime::metal::device::MetalDevice::get().alloc_raw(size)
                    });
                    if std::env::var_os("EFFECT_TORCH_ARENA_DEBUG").is_some() {
                        eprintln!("[arena] {}", runtime::metal::arena::report(&captured, plan.total));
                    }
                    let mut state = arena_state.lock().unwrap();
                    state.plan = Some(std::sync::Arc::new(plan));
                    state.captured = true;
                }
            }
            if walk_timing {

                {
                let (d, s, n) = runtime::metal::device::dispatch_stats_reset();
                eprintln!("[walk] program eval {:.1}us ({} roots) encode {:.1}us sync {:.1}us dispatches {} syncs {} sync_wait {:.1}us", t1.elapsed().as_micros(), roots.len(), t_encode.as_micros(), t_sync.as_micros(), d, s, n as f64 / 1000.0);
                }

            }
            Ok(outputs)
        })
        .await
    }
}

#[napi]
pub fn compile(roots: Vec<&LazyTensor>) -> Result<CompiledProgram> {
    let nodes: Vec<Arc<Node>> = roots.iter().map(|t| t.node.clone()).collect();
    if nodes.is_empty() {
        return Err(Error::new(
            Status::InvalidArg,
            "compile: expected at least one root".to_string(),
        ));
    }
    // Slots are collected from the fused DAG: the fusion rewrite rebuilds
    // nodes with fresh ids, so declarations taken from the unfused graph
    // would bind against node ids the program no longer contains.
    let nodes = fuse_roots(&nodes).map_err(|e| Error::new(Status::GenericFailure, e))?;
    let (slots, leaves) =
        collect_program_slots(&nodes).map_err(|e| Error::new(Status::InvalidArg, e))?;
    let signature = slots
        .iter()
        .map(|slot| slot.signature())
        .collect::<Vec<_>>()
        .join(",");
    Ok(CompiledProgram {
        inner: ProgramInner {
            roots: nodes,
            slots,
            leaves,
            signature,

            arena: Arc::new(Mutex::new(ArenaState::default())),
        },
    })
}

// Saves tensors to a safetensors file without the data ever touching the JS
// thread: all entries are evaluated in one shared walk (subgraphs shared
// between entries are computed once) and serialized in Rust.
#[napi]
pub async fn save_tensors(
    path: String,
    names: Vec<String>,
    tensors: Vec<&LazyTensor>,
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
    let nodes: Vec<Arc<Node>> = tensors.iter().map(|t| t.node.clone()).collect();
    run_compute(token, move |cancelled, _state| {
        let _guard = metal_eval_guard(&nodes);
        let mut ev = Evaluator::new(&nodes);
        let mut map = std::collections::HashMap::with_capacity(names.len());
        for (name, node) in names.iter().zip(nodes.iter()) {
            let output = eval_node(node, cancelled, &mut ev).map_err(to_napi_err)?;
            output.synchronize().map_err(to_napi_err)?;
            map.insert(name.clone(), output);
        }
        safetensors::save(&map, &metadata, &path).map_err(to_napi_err)
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

    fn mleaf(data: Vec<f32>, shape: Vec<usize>) -> Arc<Node> {
        let t = MetalTensor::from_f32(MetalDevice::get(), data, shape);
        Node::new(NodeKind::Leaf(std::sync::Arc::new(LeafSlot::new(
            value::Value(t),
        ))))
        .unwrap()
    }

    fn eval_f32(node: &Arc<Node>) -> Vec<f32> {
        let cancelled = AtomicBool::new(false);
        let mut ev = Evaluator::new(std::slice::from_ref(node));
        let v = eval_node(node, &cancelled, &mut ev).unwrap();
        v.to_f32_vec().unwrap()
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let d = (x - y).abs() / y.abs().max(1.0);
            assert!(d <= tol, "{what}[{i}]: {x} vs {y}");
        }
    }

    fn kind_counts(root: &Arc<Node>) -> HashMap<&'static str, usize> {
        kind_counts_all(std::slice::from_ref(root))
    }

    fn kind_counts_all(roots: &[Arc<Node>]) -> HashMap<&'static str, usize> {
        let mut counts: HashMap<&'static str, usize> = HashMap::new();
        let mut seen = HashSet::new();
        let mut stack = roots.to_vec();
        while let Some(n) = stack.pop() {
            if !seen.insert(n.id) {
                continue;
            }
            *counts.entry(node_kind_name(&n.kind)).or_insert(0) += 1;
            stack.extend(node_children(&n.kind));
        }
        counts
    }

    fn fixture() -> (Arc<Node>, Arc<Node>, Arc<Node>, Arc<Node>) {
        let x = mleaf(
            (0..24).map(|i| (i as f32 * 0.37).sin()).collect(),
            vec![2, 3, 4],
        );
        let w = mleaf(
            (0..32).map(|i| (i as f32 * 0.11).cos() * 0.5).collect(),
            vec![4, 8],
        );
        let b = mleaf((0..8).map(|i| i as f32 * 0.05 - 0.2).collect(), vec![8]);
        let r = mleaf(
            (0..48).map(|i| (i as f32 * 0.19).sin() * 0.3).collect(),
            vec![2, 3, 8],
        );
        (x, w, b, r)
    }

    fn linear(x: &Arc<Node>, w: &Arc<Node>, b: &Arc<Node>) -> Arc<Node> {
        Node::new(NodeKind::Linear {
            x: x.clone(),
            weight: w.clone(),
            bias: b.clone(),
        })
        .unwrap()
    }

    fn sum_all(a: &Arc<Node>) -> Arc<Node> {
        Node::new(NodeKind::Sum {
            a: a.clone(),
            dims: vec![0, 1, 2],
            keepdims: false,
        })
        .unwrap()
    }

    #[test]
    fn residual_epilogue_matches_unfused() {
        let (x, w, b, r) = fixture();
        let out = Node::new(NodeKind::Add {
            a: linear(&x, &w, &b),
            b: r.clone(),
        })
        .unwrap();
        let loss = sum_all(&out);
        let grads =
            effect_torch_autodiff::grad(&loss, &[x.clone(), w.clone(), b.clone(), r.clone()])
                .unwrap();
        let mut roots = vec![loss.clone()];
        roots.extend(grads.iter().cloned());
        let fused = gemm_epilogue_pass(&roots).unwrap();
        let counts = kind_counts(&fused[0].clone());
        assert_eq!(counts.get("LinearResidual"), Some(&1), "{counts:?}");
        assert_eq!(counts.get("Linear"), None, "{counts:?}");
        assert_eq!(counts.get("Add"), None, "{counts:?}");
        for ((name, p), f) in ["loss", "dx", "dw", "db", "dr"]
            .iter()
            .zip(roots.iter())
            .zip(fused.iter())
        {
            assert_close(&eval_f32(p), &eval_f32(f), 1e-4, name);
        }
    }

    #[test]
    fn residual_epilogue_keeps_shared_linear() {
        // The Linear output feeding anything besides the Add (here a
        // second residual consumer) blocks the rewrite.
        let (x, w, b, r) = fixture();
        let lin = linear(&x, &w, &b);
        let out1 = Node::new(NodeKind::Add {
            a: lin.clone(),
            b: r.clone(),
        })
        .unwrap();
        let out2 = Node::new(NodeKind::Add {
            a: lin,
            b: r.clone(),
        })
        .unwrap();
        let fused = gemm_epilogue_pass(&[out1, out2]).unwrap();
        let counts = kind_counts(
            &Node::new(NodeKind::Add {
                a: fused[0].clone(),
                b: fused[1].clone(),
            })
            .unwrap(),
        );
        assert_eq!(counts.get("LinearResidual"), None, "{counts:?}");
    }

    #[test]
    fn gelu_epilogue_matches_unfused() {
        for approximate in [false, true] {
            let (x, w, b, _) = fixture();
            let g = Node::new(NodeKind::Gelu {
                a: linear(&x, &w, &b),
                approximate,
            })
            .unwrap();
            let loss = sum_all(&g);
            let grads =
                effect_torch_autodiff::grad(&loss, &[x.clone(), w.clone(), b.clone()]).unwrap();
            let mut roots = vec![loss.clone()];
            roots.extend(grads.iter().cloned());
            let fused = gemm_epilogue_pass(&roots).unwrap();
            let counts = kind_counts_all(&fused);
            // Backward reads the pre-activation, so the dual-store
            // variant must be present behind two FusedPick nodes.
            assert_eq!(counts.get("LinearGelu"), Some(&1), "{counts:?}");
            assert_eq!(counts.get("FusedPick"), Some(&2), "{counts:?}");
            assert_eq!(counts.get("Linear"), None, "{counts:?}");
            for ((name, p), f) in ["loss", "dx", "dw", "db"]
                .iter()
                .zip(roots.iter())
                .zip(fused.iter())
            {
                assert_close(&eval_f32(p), &eval_f32(f), 1e-3, name);
            }
        }
    }

    #[test]
    fn gelu_epilogue_drops_preact_without_backward() {
        let (x, w, b, _) = fixture();
        let g = Node::new(NodeKind::Gelu {
            a: linear(&x, &w, &b),
            approximate: false,
        })
        .unwrap();
        let fused = gemm_epilogue_pass(std::slice::from_ref(&g)).unwrap();
        let counts = kind_counts(&fused[0]);
        assert_eq!(counts.get("LinearGelu"), Some(&1), "{counts:?}");
        assert_eq!(counts.get("FusedPick"), None, "{counts:?}");
        assert_close(&eval_f32(&g), &eval_f32(&fused[0]), 1e-4, "gelu");
    }
}

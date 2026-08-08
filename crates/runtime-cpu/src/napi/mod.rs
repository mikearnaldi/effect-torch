mod err;
mod fusion;
mod safetensors;
mod value;

use self::err::to_napi_err;
use self::value::Value;
use crate::{composed, conv, pool, CpuBuffer, Tensor};
use effect_torch_compiler::{collect_program_slots, fuse_roots, ProgramSlot};
use effect_torch_graph::CrossEntropyReduction as CeReduction;
use effect_torch_graph::{node_children, remap_children, Device, PositionOffset};
use effect_torch_napi::{try_register_export, unregister_export, vec_to_bytes, CancellationState};
use effect_torch_runtime::{CancellationFlag, DType, Layout};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

pub type LeafSlot = effect_torch_graph::LeafSlot;
pub(crate) type Node = effect_torch_graph::Node<effect_torch_compiler::Expr>;
type NodeKind = effect_torch_graph::NodeKind<effect_torch_compiler::Expr>;

fn cpu_device() -> Device {
    Device::Cpu
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
            .into_iter()
            .map(|dimension| dimension as u32)
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
    let flat = if matches!(value.dtype(), DType::F16 | DType::BF16) {
        Value(value.tensor().cast(DType::F32))
    } else {
        value.clone()
    };
    let tensor = flat.tensor().contiguous();
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
    let offset = tensor.layout.offset() * element_size;
    let byte_len = tensor.numel() * element_size;
    if !base.is_null() {
        let addr = base as usize + offset;
        if try_register_export(addr) {
            return Ok(Readback {
                data: addr as *mut u8,
                byte_len,
                hint: Some(FinalizeHint::ZeroCopy {
                    value: Value(tensor),
                    addr,
                }),
            });
        }
    }
    let (_, ptr, len, cap) = match flat.dtype() {
        DType::F32 => vec_to_bytes(flat.to_f32_vec().map_err(to_napi_err)?),
        DType::F64 => vec_to_bytes(flat.to_f64_vec().map_err(to_napi_err)?),
        DType::I64 => vec_to_bytes(flat.to_i64_vec().map_err(to_napi_err)?),
        DType::U8 => vec_to_bytes(flat.to_u8_vec().map_err(to_napi_err)?),
        DType::U32 => vec_to_bytes(flat.to_u32_vec().map_err(to_napi_err)?),
        dtype => {
            return Err(Error::new(
                Status::InvalidArg,
                format!("readback not implemented for dtype: {}", dtype.name()),
            ));
        }
    };
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
    let (inner, vocabulary) = (weight.shape[0], weight.shape[1]);
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
    let x = if rank == 2 {
        x.clone()
    } else {
        Node::new(NodeKind::Reshape {
            a: x.clone(),
            shape: vec![rows, inner],
        })?
    };
    let target = if target.shape.as_slice() == [rows] {
        target.clone()
    } else {
        Node::new(NodeKind::Reshape {
            a: target.clone(),
            shape: vec![rows],
        })?
    };
    let ignored_count =
        if target.dtype == DType::U32 && (ignore_index < 0 || ignore_index > u32::MAX as i64) {
            cached_constant(0.0, DType::F32)?
        } else {
            let ignore = cached_constant(ignore_index as f64, target.dtype)?;
            let ignored = Node::new(NodeKind::Eq {
                a: target.clone(),
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
    let active = Node::new(NodeKind::Sub {
        a: cached_constant(rows as f64, DType::F32)?,
        b: ignored_count,
    })?;
    let chunk_length = rows.div_ceil(chunks);
    let mut total = None;
    let mut offset = 0;
    while offset < rows {
        let end = (offset + chunk_length).min(rows);
        let x_chunk = Node::new(NodeKind::Slice {
            a: x.clone(),
            ranges: vec![(offset, end, 1), (0, inner, 1)],
        })?;
        let target_chunk = Node::new(NodeKind::Slice {
            a: target.clone(),
            ranges: vec![(offset, end, 1)],
        })?;
        let logits_chunk = Node::new(NodeKind::Linear {
            x: x_chunk,
            weight: weight.clone(),
            bias: bias.clone(),
        })?;
        let loss = Node::new(NodeKind::CrossEntropy {
            logits: logits_chunk,
            target: target_chunk,
            ignore_index,
            reduction: CeReduction::Sum,
        })?;
        let loss = Node::new(NodeKind::Checkpoint { a: loss })?;
        let loss = Node::new(NodeKind::Cast {
            a: loss,
            dtype: DType::F32,
        })?;
        total = Some(match total {
            None => loss,
            Some(previous) => Node::new(NodeKind::Add {
                a: previous,
                b: loss,
            })?,
        });
        offset = end;
    }
    let mean = Node::new(NodeKind::Div {
        a: total.expect("at least one chunk"),
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

fn gelu(tensor: &Tensor, approximate: bool) -> Tensor {
    let dtype = tensor.dtype();
    let full = |value: f64| Tensor::full(&[], value, dtype);
    if approximate {
        let inner = tensor
            .add(&tensor.mul(tensor).mul(tensor).mul(&full(0.044715)))
            .mul(&full(0.7978845608028654));
        tensor.mul(&full(0.5)).mul(&full(1.0).add(&inner.tanh()))
    } else {
        let inner = tensor.mul(&full(std::f64::consts::FRAC_1_SQRT_2)).erf();
        tensor.mul(&full(0.5)).mul(&full(1.0).add(&inner))
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

struct Evaluator {
    cache: HashMap<u64, Value>,
    adamw: HashMap<u64, [Value; 2]>,
    sgd: HashMap<u64, Value>,
    multi: HashMap<u64, Vec<Value>>,
    layer_norm: HashMap<u64, [Value; 2]>,
    step_scalars: HashMap<(u64, DType), Value>,
    consumers: HashMap<u64, usize>,
    roots: HashSet<u64>,
    slots: HashMap<u64, Value>,
    kv: Option<Arc<KvContext>>,
}

impl Evaluator {
    fn new(roots: &[Arc<Node>]) -> Self {
        Self::with_slots(roots, HashMap::new())
    }

    fn with_slots(roots: &[Arc<Node>], slots: HashMap<u64, Value>) -> Self {
        Self::with_kv(roots, slots, None)
    }

    fn with_kv(
        roots: &[Arc<Node>],
        slots: HashMap<u64, Value>,
        kv: Option<Arc<KvContext>>,
    ) -> Self {
        let mut consumers = HashMap::new();
        let mut visited = HashSet::new();
        for root in roots {
            count_consumers(root, &mut consumers, &mut visited);
        }
        Self {
            cache: HashMap::new(),
            adamw: HashMap::new(),
            sgd: HashMap::new(),
            multi: HashMap::new(),
            layer_norm: HashMap::new(),
            step_scalars: HashMap::new(),
            consumers,
            roots: roots.iter().map(|root| root.id).collect(),
            slots,
            kv,
        }
    }

    fn value(&self, id: u64) -> err::Res<Value> {
        self.cache
            .get(&id)
            .cloned()
            .ok_or_else(|| format!("internal error: unevaluated node {id}"))
    }

    fn step_scalar(&mut self, id: u64, dtype: DType) -> err::Res<Value> {
        let key = (id, dtype);
        if let Some(value) = self.step_scalars.get(&key) {
            return Ok(value.clone());
        }
        let value = Value(self.value(id)?.tensor().cast(dtype));
        self.step_scalars.insert(key, value.clone());
        Ok(value)
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
                    self.layer_norm.remove(&child.id);
                }
            }
        }
    }
}

fn count_consumers(
    root: &Arc<Node>,
    consumers: &mut HashMap<u64, usize>,
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

fn coerce_scalar_values(a: Value, b: Value) -> (Value, Value) {
    if a.dtype() == b.dtype() || !a.dtype().is_float() || !b.dtype().is_float() {
        return (a, b);
    }
    let a_scalar = a.shape().is_empty();
    let b_scalar = b.shape().is_empty();
    if a_scalar == b_scalar {
        return (a, b);
    }
    if a_scalar {
        (Value(a.tensor().cast(b.dtype())), b)
    } else {
        (a.clone(), Value(b.tensor().cast(a.dtype())))
    }
}

fn normalize_node_output(node: &Node, mut output: Value) -> err::Res<Value> {
    if !node.device.is_cpu() {
        return Err(format!(
            "{} requested an unsupported device",
            node_kind_name(&node.kind)
        ));
    }
    if output.dtype() != node.dtype {
        output = Value(output.tensor().cast(node.dtype));
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
        output = Value(
            output
                .tensor()
                .contiguous()
                .view(Layout::contiguous(node.shape.clone())),
        );
    }
    Ok(output)
}

fn eval_node(
    root: &Arc<Node>,
    cancelled: &CancellationFlag,
    evaluator: &mut Evaluator,
) -> err::Res<Value> {
    let mut stack = vec![(root.clone(), false)];
    while let Some((node, processed)) = stack.pop() {
        if cancelled.load(Ordering::Relaxed) {
            return Err("operation aborted".to_string());
        }
        if evaluator.cache.contains_key(&node.id) {
            continue;
        }
        if processed {
            let output = normalize_node_output(&node, eval_uncached(&node, evaluator)?)?;
            evaluator.cache.insert(node.id, output);
            evaluator.release_children(&node);
            continue;
        }
        stack.push((node.clone(), true));
        for child in node_children(&node.kind) {
            if !evaluator.cache.contains_key(&child.id) {
                stack.push((child, false));
            }
        }
    }
    Ok(evaluator
        .cache
        .get(&root.id)
        .expect("root is evaluated before its consumers")
        .clone())
}

fn eval_uncached(node: &Arc<Node>, evaluator: &mut Evaluator) -> err::Res<Value> {
    if !node.device.is_cpu() {
        return Err("unsupported device".to_string());
    }
    let value = match &node.kind {
        NodeKind::Leaf(slot) => slot.get().map_err(|error| error.to_string())?,
        NodeKind::Input { slot, .. } | NodeKind::ScalarInput { slot, .. } => evaluator
            .slots
            .get(&node.id)
            .cloned()
            .ok_or_else(|| {
                format!(
                    "input slot {slot} is unbound: placeholder leaves evaluate only inside a compiled program run"
                )
            })?,
        NodeKind::FromBytes {
            data,
            shape,
            dtype,
            ..
        } => safetensors::value_from_bytes(data, shape, *dtype)?,
        NodeKind::Zeros { shape, dtype, .. } => Value(Tensor::zeros(shape, *dtype)),
        NodeKind::Ones { shape, dtype, .. } => Value(Tensor::ones(shape, *dtype)),
        NodeKind::Full {
            shape,
            value,
            dtype,
            ..
        } => Value(Tensor::full(shape, *value, *dtype)),
        NodeKind::Randn { shape, dtype, .. } => Value(Tensor::randn(shape, *dtype)),
        NodeKind::Uniform {
            lo,
            hi,
            shape,
            dtype,
            ..
        } => Value(Tensor::uniform(*lo, *hi, shape, *dtype)),
        NodeKind::Arange {
            start,
            end,
            step,
            dtype,
            ..
        } => Value(Tensor::arange(*start, *end, *step, *dtype)),
        NodeKind::Eye { n, dtype, .. } => Value(Tensor::eye(*n, *dtype)),
        NodeKind::Add { a, b } => {
            let (a, b) = coerce_scalar_values(evaluator.value(a.id)?, evaluator.value(b.id)?);
            Value(a.tensor().add(b.tensor()))
        }
        NodeKind::Sub { a, b } => {
            let (a, b) = coerce_scalar_values(evaluator.value(a.id)?, evaluator.value(b.id)?);
            Value(a.tensor().sub(b.tensor()))
        }
        NodeKind::Mul { a, b } => {
            let (a, b) = coerce_scalar_values(evaluator.value(a.id)?, evaluator.value(b.id)?);
            Value(a.tensor().mul(b.tensor()))
        }
        NodeKind::Div { a, b } => {
            let (a, b) = coerce_scalar_values(evaluator.value(a.id)?, evaluator.value(b.id)?);
            Value(a.tensor().div(b.tensor()))
        }
        NodeKind::Eq { a, b } => Value(
            evaluator
                .value(a.id)?
                .tensor()
                .eq(evaluator.value(b.id)?.tensor()),
        ),
        NodeKind::Gt { a, b } => Value(
            evaluator
                .value(a.id)?
                .tensor()
                .gt(evaluator.value(b.id)?.tensor()),
        ),
        NodeKind::Lt { a, b } => Value(
            evaluator
                .value(a.id)?
                .tensor()
                .lt(evaluator.value(b.id)?.tensor()),
        ),
        NodeKind::Ge { a, b } => Value(
            evaluator
                .value(a.id)?
                .tensor()
                .ge(evaluator.value(b.id)?.tensor()),
        ),
        NodeKind::Le { a, b } => Value(
            evaluator
                .value(a.id)?
                .tensor()
                .le(evaluator.value(b.id)?.tensor()),
        ),
        NodeKind::Maximum { a, b } => {
            let (a, b) = coerce_scalar_values(evaluator.value(a.id)?, evaluator.value(b.id)?);
            Value(a.tensor().maximum(b.tensor()))
        }
        NodeKind::Minimum { a, b } => {
            let (a, b) = coerce_scalar_values(evaluator.value(a.id)?, evaluator.value(b.id)?);
            Value(a.tensor().minimum(b.tensor()))
        }
        NodeKind::Neg { a } => Value(evaluator.value(a.id)?.tensor().neg()),
        NodeKind::Abs { a } => Value(evaluator.value(a.id)?.tensor().abs()),
        NodeKind::Sqrt { a } => Value(evaluator.value(a.id)?.tensor().sqrt()),
        NodeKind::Exp { a } => Value(evaluator.value(a.id)?.tensor().exp()),
        NodeKind::Log { a } => Value(evaluator.value(a.id)?.tensor().log()),
        NodeKind::Sin { a } => Value(evaluator.value(a.id)?.tensor().sin()),
        NodeKind::Cos { a } => Value(evaluator.value(a.id)?.tensor().cos()),
        NodeKind::Tanh { a } => Value(evaluator.value(a.id)?.tensor().tanh()),
        NodeKind::Relu { a } => Value(evaluator.value(a.id)?.tensor().relu()),
        NodeKind::Erf { a } => Value(evaluator.value(a.id)?.tensor().erf()),
        NodeKind::Gelu { a, approximate } => Value(gelu(
            evaluator.value(a.id)?.tensor(),
            *approximate,
        )),
        NodeKind::Floor { a } => Value(evaluator.value(a.id)?.tensor().floor()),
        NodeKind::Ceil { a } => Value(evaluator.value(a.id)?.tensor().ceil()),
        NodeKind::Round { a } => Value(evaluator.value(a.id)?.tensor().round()),
        NodeKind::Sign { a } => Value(evaluator.value(a.id)?.tensor().sign()),
        NodeKind::Where { cond, a, b } => Value(
            evaluator
                .value(a.id)?
                .tensor()
                .where_(
                    evaluator.value(cond.id)?.tensor(),
                    evaluator.value(b.id)?.tensor(),
                ),
        ),
        NodeKind::Argmax { a, dim } => Value(
            evaluator
                .value(a.id)?
                .tensor()
                .argmax(*dim)
                .cast(DType::I64),
        ),
        NodeKind::Argmin { a, dim } => Value(
            evaluator
                .value(a.id)?
                .tensor()
                .argmin(*dim)
                .cast(DType::I64),
        ),
        NodeKind::Cumsum { a, dim } => Value(evaluator.value(a.id)?.tensor().cumsum(*dim)),
        NodeKind::ScatterAdd {
            a,
            dim,
            indexes,
            src,
        } => Value(evaluator.value(a.id)?.tensor().scatter_add(
            *dim,
            evaluator.value(indexes.id)?.tensor(),
            evaluator.value(src.id)?.tensor(),
        )),
        NodeKind::Gather { a, dim, indexes } => Value(
            evaluator
                .value(a.id)?
                .tensor()
                .gather(*dim, evaluator.value(indexes.id)?.tensor()),
        ),
        NodeKind::IndexSelect { a, dim, indexes } => Value(
            evaluator
                .value(a.id)?
                .tensor()
                .index_select(*dim, evaluator.value(indexes.id)?.tensor()),
        ),
        NodeKind::CrossEntropy {
            logits,
            target,
            ignore_index,
            reduction,
        } => Value(composed::cross_entropy_forward(
            evaluator.value(logits.id)?.tensor(),
            evaluator.value(target.id)?.tensor(),
            *ignore_index,
            *reduction,
        )?),
        NodeKind::CrossEntropyBackward {
            logits,
            target,
            ignore_index,
            reduction,
        } => Value(composed::cross_entropy_backward(
            evaluator.value(logits.id)?.tensor(),
            evaluator.value(target.id)?.tensor(),
            *ignore_index,
            *reduction,
        )?),
        NodeKind::Sdpa {
            q,
            k,
            v,
            scale,
            causal,
        } => Value(composed::sdpa_forward(
            evaluator.value(q.id)?.tensor(),
            evaluator.value(k.id)?.tensor(),
            evaluator.value(v.id)?.tensor(),
            *scale,
            *causal,
        )),
        NodeKind::SdpaBackward {
            q,
            k,
            v,
            g,
            scale,
            causal,
            ..
        } => {
            let (dq, dk, dv) = composed::sdpa_backward(
                evaluator.value(q.id)?.tensor(),
                evaluator.value(k.id)?.tensor(),
                evaluator.value(v.id)?.tensor(),
                evaluator.value(g.id)?.tensor(),
                *scale,
                *causal,
            );
            let values = vec![Value(dq), Value(dk), Value(dv)];
            let head = values[0].clone();
            evaluator.multi.insert(node.id, values);
            head
        }
        NodeKind::SdpaBackwardOut { of, index } => evaluator
            .multi
            .get(&of.id)
            .and_then(|values| values.get(*index as usize))
            .cloned()
            .ok_or_else(|| "sdpa backward out: outputs missing".to_string())?,
        NodeKind::KdaChunk {
            q,
            k,
            v,
            log_decay,
            beta,
            scale,
        } => Value(composed::kda_chunk_forward(
            evaluator.value(q.id)?.tensor(),
            evaluator.value(k.id)?.tensor(),
            evaluator.value(v.id)?.tensor(),
            evaluator.value(log_decay.id)?.tensor(),
            evaluator.value(beta.id)?.tensor(),
            *scale,
        )),
        NodeKind::KdaRecurrence {
            q,
            k,
            v,
            log_decay,
            beta,
            scale,
            layer,
        } => {
            let context = evaluator.kv.clone().ok_or_else(|| {
                "kda recurrence: node evaluates only inside a decode program run".to_string()
            })?;
            kda_recurrence(
                &context,
                *layer,
                &evaluator.value(q.id)?,
                &evaluator.value(k.id)?,
                &evaluator.value(v.id)?,
                &evaluator.value(log_decay.id)?,
                &evaluator.value(beta.id)?,
                *scale,
            )?
        }
        NodeKind::ShortConv1d { x, weight } => Value(composed::short_conv1d_forward(
            evaluator.value(x.id)?.tensor(),
            evaluator.value(weight.id)?.tensor(),
        )),
        NodeKind::KdaBackward {
            q,
            k,
            v,
            log_decay,
            beta,
            g,
            scale,
        } => {
            let (dq, dk, dv, dg, db) = composed::kda_chunk_backward(
                evaluator.value(q.id)?.tensor(),
                evaluator.value(k.id)?.tensor(),
                evaluator.value(v.id)?.tensor(),
                evaluator.value(log_decay.id)?.tensor(),
                evaluator.value(beta.id)?.tensor(),
                evaluator.value(g.id)?.tensor(),
                *scale,
            );
            let values = vec![
                Value(dq),
                Value(dk),
                Value(dv),
                Value(dg),
                Value(db),
            ];
            let head = values[0].clone();
            evaluator.multi.insert(node.id, values);
            head
        }
        NodeKind::KdaBackwardOut { of, index } => evaluator
            .multi
            .get(&of.id)
            .and_then(|values| values.get(*index as usize))
            .cloned()
            .ok_or_else(|| "kda backward out: outputs missing".to_string())?,
        NodeKind::ShortConv1dBackwardX { x, weight, g } => {
            Value(composed::short_conv1d_backward_x(
                evaluator.value(x.id)?.tensor(),
                evaluator.value(weight.id)?.tensor(),
                evaluator.value(g.id)?.tensor(),
            ))
        }
        NodeKind::ShortConv1dBackwardW { x, weight, g } => {
            Value(composed::short_conv1d_backward_w(
                evaluator.value(x.id)?.tensor(),
                evaluator.value(weight.id)?.tensor(),
                evaluator.value(g.id)?.tensor(),
            ))
        }
        NodeKind::ConvState { x, weight, layer } => {
            let context = evaluator.kv.clone().ok_or_else(|| {
                "conv state: node evaluates only inside a decode program run".to_string()
            })?;
            conv_state(
                &context,
                *layer,
                &evaluator.value(x.id)?,
                &evaluator.value(weight.id)?,
            )?
        }
        NodeKind::PositionEmbedding { weight, seq_len } => {
            let weight = evaluator.value(weight.id)?;
            Value(
                weight
                    .tensor()
                    .view(weight.tensor().layout.narrow(0, 0, *seq_len))
                    .contiguous(),
            )
        }
        NodeKind::KvAttention {
            q,
            k,
            v,
            scale,
            layer,
            window,
        } => {
            let context = evaluator.kv.clone().ok_or_else(|| {
                "kv attention: node evaluates only inside a kv program run".to_string()
            })?;
            kv_attention(
                &context,
                *layer,
                &evaluator.value(q.id)?,
                &evaluator.value(k.id)?,
                &evaluator.value(v.id)?,
                *scale,
                *window,
            )?
        }
        NodeKind::RotaryEmbedding {
            x, theta, offset, ..
        } => {
            let offsets = match offset {
                PositionOffset::Absolute => vec![0],
                PositionOffset::Cursor => {
                    let context = evaluator.kv.as_ref().ok_or_else(|| {
                        "rotary embedding: cursor offset outside a kv program run".to_string()
                    })?;
                    context
                        .slots
                        .iter()
                        .map(|slot| {
                            slot.lock()
                                .map(|state| state.cursor)
                                .map_err(|error| {
                                    format!("rotary embedding: sequence lock poisoned: {error}")
                                })
                        })
                        .collect::<err::Res<Vec<_>>>()?
                }
            };
            Value(composed::rotary_forward(
                evaluator.value(x.id)?.tensor(),
                &offsets,
                *theta,
                1.0,
            )?)
        }
        NodeKind::RotaryEmbeddingBackward { g, theta, .. } => Value(
            composed::rotary_forward(
                evaluator.value(g.id)?.tensor(),
                &[0],
                *theta,
                -1.0,
            )?,
        ),
        NodeKind::Linear { x, weight, bias } => {
            let activation = evaluator
                .value(x.id)?
                .tensor()
                .try_matmul(evaluator.value(weight.id)?.tensor())?;
            Value(activation.add(evaluator.value(bias.id)?.tensor()))
        }
        NodeKind::LinearResidual {
            x,
            weight,
            bias,
            residual,
        } => {
            let activation = evaluator
                .value(x.id)?
                .tensor()
                .try_matmul(evaluator.value(weight.id)?.tensor())?;
            Value(
                activation
                .add(evaluator.value(bias.id)?.tensor())
                .add(evaluator.value(residual.id)?.tensor()),
            )
        }
        NodeKind::LinearGelu {
            x,
            weight,
            bias,
            approximate,
            dual,
        } => {
            let activation = evaluator
                .value(x.id)?
                .tensor()
                .try_matmul(evaluator.value(weight.id)?.tensor())?
                .add(evaluator.value(bias.id)?.tensor());
            let output = gelu(&activation, *approximate);
            if *dual {
                let values = vec![Value(activation), Value(output)];
                let head = values[0].clone();
                evaluator.multi.insert(node.id, values);
                head
            } else {
                Value(output)
            }
        }
        NodeKind::LayerNorm {
            x,
            weight,
            bias,
            eps,
        } => Value(composed::layer_norm_forward(
            evaluator.value(x.id)?.tensor(),
            evaluator.value(weight.id)?.tensor(),
            evaluator.value(bias.id)?.tensor(),
            *eps,
        )),
        NodeKind::LayerNormBackward { x, weight, g, eps } => {
            let (dx, dw, db) = composed::layer_norm_backward(
                evaluator.value(x.id)?.tensor(),
                evaluator.value(weight.id)?.tensor(),
                evaluator.value(g.id)?.tensor(),
                *eps,
            );
            evaluator
                .layer_norm
                .insert(node.id, [Value(dw), Value(db)]);
            Value(dx)
        }
        NodeKind::LayerNormBackwardOut { of, index } => evaluator
            .layer_norm
            .get(&of.id)
            .and_then(|values| values.get(*index as usize - 1))
            .cloned()
            .ok_or_else(|| {
                "layer_norm_backward_out: backward node has no stored outputs".to_string()
            })?,
        NodeKind::Conv1d {
            x,
            w,
            stride,
            padding,
            dilation,
            groups,
        } => Value(conv::conv1d(
            evaluator.value(x.id)?.tensor(),
            evaluator.value(w.id)?.tensor(),
            *stride,
            *padding,
            *dilation,
            *groups,
        )),
        NodeKind::Conv2d {
            x,
            w,
            stride,
            padding,
            dilation,
            groups,
        } => Value(conv::conv2d(
            evaluator.value(x.id)?.tensor(),
            evaluator.value(w.id)?.tensor(),
            *stride,
            *padding,
            *dilation,
            *groups,
        )),
        NodeKind::ConvTranspose1d {
            x,
            w,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        } => Value(conv::conv_transpose1d(
            evaluator.value(x.id)?.tensor(),
            evaluator.value(w.id)?.tensor(),
            *stride,
            *padding,
            *output_padding,
            *dilation,
            *groups,
        )),
        NodeKind::ConvTranspose2d {
            x,
            w,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        } => Value(conv::conv_transpose2d(
            evaluator.value(x.id)?.tensor(),
            evaluator.value(w.id)?.tensor(),
            *stride,
            *padding,
            *output_padding,
            *dilation,
            *groups,
        )),
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
            let x = evaluator.value(x.id)?;
            let g = evaluator.value(g.id)?;
            let expand = |tensor: &Tensor| {
                let shape = tensor.shape();
                tensor.contiguous().view(Layout::contiguous(vec![
                    shape[0], shape[1], shape[2], 1,
                ]))
            };
            let output = conv::conv2d_backward_w(
                &expand(x.tensor()),
                &expand(g.tensor()),
                [*kernel, 1],
                *out_channels,
                *stride,
                *padding,
                *dilation,
                *groups,
            );
            let shape = output.shape();
            Value(output.contiguous().view(Layout::contiguous(vec![
                shape[0], shape[1], shape[2],
            ])))
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
        } => Value(conv::conv2d_backward_w(
            evaluator.value(x.id)?.tensor(),
            evaluator.value(g.id)?.tensor(),
            *kernel,
            *out_channels,
            *stride,
            *padding,
            *dilation,
            *groups,
        )),
        NodeKind::Pow { a, exp } => Value(evaluator.value(a.id)?.tensor().powf(*exp)),
        NodeKind::Cast { a, dtype } => Value(evaluator.value(a.id)?.tensor().cast(*dtype)),
        NodeKind::Sum { a, dims, keepdims } => {
            let output = evaluator.value(a.id)?.tensor().sum(dims);
            Value(if *keepdims {
                output
            } else {
                output.squeeze_dims(dims)
            })
        }
        NodeKind::Mean { a, dims, keepdims } => {
            let output = evaluator.value(a.id)?.tensor().mean(dims);
            Value(if *keepdims {
                output
            } else {
                output.squeeze_dims(dims)
            })
        }
        NodeKind::Max { a, dims, keepdims } => {
            let output = evaluator.value(a.id)?.tensor().max(dims);
            Value(if *keepdims {
                output
            } else {
                output.squeeze_dims(dims)
            })
        }
        NodeKind::Min { a, dims, keepdims } => {
            let output = evaluator.value(a.id)?.tensor().min(dims);
            Value(if *keepdims {
                output
            } else {
                output.squeeze_dims(dims)
            })
        }
        NodeKind::Prod { a, dims, keepdims } => {
            let output = evaluator.value(a.id)?.tensor().prod(dims);
            Value(if *keepdims {
                output
            } else {
                output.squeeze_dims(dims)
            })
        }
        NodeKind::Reshape { a, shape } => Value(
            evaluator
                .value(a.id)?
                .tensor()
                .contiguous()
                .view(Layout::contiguous(shape.clone())),
        ),
        NodeKind::Permute { a, dims } => {
            let value = evaluator.value(a.id)?;
            Value(value.tensor().view(value.tensor().layout.permute(dims)).contiguous())
        }
        NodeKind::Slice { a, ranges } => {
            let mut output = evaluator.value(a.id)?.into_tensor();
            for (dimension, &(start, stop, stride)) in ranges.iter().enumerate() {
                let length = stop.saturating_sub(start).div_ceil(stride);
                if length == 0 {
                    let mut shape = output.shape().to_vec();
                    shape[dimension] = 0;
                    output = Tensor::zeros(&shape, output.dtype());
                    continue;
                }
                output = output
                    .view(output.layout.narrow(
                        dimension,
                        start,
                        (length - 1) * stride + 1,
                    ))
                    .contiguous();
                if stride > 1 {
                    let indexes: Vec<u32> = (0..length as u32)
                        .map(|index| index * stride as u32)
                        .collect();
                    output = output.index_select(
                        dimension,
                        &Tensor::from_vec(indexes, vec![length]),
                    );
                }
            }
            Value(output)
        }
        NodeKind::Concat { a, b, dim } => Value(Tensor::cat(
            &[
                evaluator.value(a.id)?.tensor(),
                evaluator.value(b.id)?.tensor(),
            ],
            *dim,
        )),
        NodeKind::BroadcastTo { a, shape } => {
            let value = evaluator.value(a.id)?;
            Value(
                value
                    .tensor()
                    .view(value.tensor().layout.broadcast_to(shape))
                    .contiguous(),
            )
        }
        NodeKind::Matmul { a, b } => {
            let output = evaluator
                .value(a.id)?
                .tensor()
                .try_matmul(evaluator.value(b.id)?.tensor())?;
            Value(output)
        }
        NodeKind::Inverse { a } => Value(evaluator.value(a.id)?.tensor().try_inverse()?),
        NodeKind::Det { a } => Value(evaluator.value(a.id)?.tensor().det()),
        NodeKind::Solve { a, b } => {
            let output = evaluator
                .value(a.id)?
                .tensor()
                .try_solve(evaluator.value(b.id)?.tensor())?;
            Value(output)
        }
        NodeKind::StopGradient { a } | NodeKind::Checkpoint { a } => evaluator.value(a.id)?,
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
            let param_value = evaluator.value(param.id)?;
            let grad_value = evaluator.value(grad.id)?;
            let m_value = evaluator.value(m.id)?;
            let v_value = evaluator.value(v.id)?;
            let lr_value = evaluator.step_scalar(lr.id, param_value.dtype())?;
            let c1_value = evaluator.step_scalar(c1.id, param_value.dtype())?;
            let c2_value = evaluator.step_scalar(c2.id, param_value.dtype())?;
            let fused = if fusion::is_supported(&param_value.device(), param_value.dtype()) {
                fusion::run(
                    &fusion::adamw_exprs(*beta1, *beta2, *eps, *weight_decay),
                    &[
                        param_value.clone(),
                        grad_value.clone(),
                        m_value.clone(),
                        v_value.clone(),
                    ],
                    None,
                    &[lr_value.clone(), c1_value.clone(), c2_value.clone()],
                    param_value.numel(),
                    &param_value.shape(),
                    param_value.dtype(),
                    &cpu_device(),
                )
                .ok()
            } else {
                None
            };
            if let Some(mut outputs) = fused {
                let next_param = outputs.remove(0);
                evaluator
                    .adamw
                    .insert(node.id, [outputs.remove(0), outputs.remove(0)]);
                next_param
            } else {
                let (next_param, next_m, next_v) = composed::adamw_step(
                    param_value.tensor(),
                    grad_value.tensor(),
                    m_value.tensor(),
                    v_value.tensor(),
                    lr_value.tensor(),
                    c1_value.tensor(),
                    c2_value.tensor(),
                    *beta1,
                    *beta2,
                    *eps,
                    *weight_decay,
                );
                evaluator
                    .adamw
                    .insert(node.id, [Value(next_m), Value(next_v)]);
                Value(next_param)
            }
        }
        NodeKind::AdamWOut { step, index } => {
            let _ = evaluator.value(step.id)?;
            match index {
                0 => evaluator.value(step.id)?,
                1 => evaluator.adamw[&step.id][0].clone(),
                _ => evaluator.adamw[&step.id][1].clone(),
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
            let first = evaluator.value(params[0].id)?;
            let lr = evaluator.step_scalar(lr.id, first.dtype())?;
            let c1 = evaluator.step_scalar(c1.id, first.dtype())?;
            let c2 = evaluator.step_scalar(c2.id, first.dtype())?;
            let mut inputs = Vec::with_capacity(params.len() * 4);
            for index in 0..params.len() {
                inputs.push(evaluator.value(params[index].id)?);
                inputs.push(evaluator.value(grads[index].id)?);
                inputs.push(evaluator.value(ms[index].id)?);
                inputs.push(evaluator.value(vs[index].id)?);
            }
            let base = fusion::adamw_exprs(*beta1, *beta2, *eps, *weight_decay);
            let mut expressions = Vec::with_capacity(params.len() * 3);
            for index in 0..params.len() {
                let remap: HashMap<u32, u32> =
                    (0..4).map(|lane| (lane, index as u32 * 4 + lane)).collect();
                expressions.extend(base.iter().map(|expr| expr.remap_lanes(&remap)));
            }
            let outputs = fusion::run(
                &expressions,
                &inputs,
                None,
                &[lr, c1, c2],
                first.numel(),
                &first.shape(),
                first.dtype(),
                &cpu_device(),
            )?;
            let head = outputs[0].clone();
            evaluator.multi.insert(node.id, outputs);
            head
        }
        NodeKind::AdamWGroupOut { of, param, index } => {
            let _ = evaluator.value(of.id)?;
            evaluator
                .multi
                .get(&of.id)
                .and_then(|outputs| outputs.get(*param as usize * 3 + *index as usize))
                .cloned()
                .ok_or_else(|| "adamw_group_out: group has no stored outputs".to_string())?
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
            let param_value = evaluator.value(param.id)?;
            let grad_value = evaluator.value(grad.id)?;
            let velocity_value = evaluator.value(velocity.id)?;
            let first_value = evaluator.step_scalar(first.id, param_value.dtype())?;
            let lr_value = evaluator.step_scalar(lr.id, param_value.dtype())?;
            let fused = if fusion::is_supported(&param_value.device(), param_value.dtype()) {
                fusion::run(
                    &fusion::sgd_exprs(*momentum, *dampening, *nesterov, *weight_decay),
                    &[
                        param_value.clone(),
                        grad_value.clone(),
                        velocity_value.clone(),
                    ],
                    None,
                    &[lr_value.clone(), first_value.clone()],
                    param_value.numel(),
                    &param_value.shape(),
                    param_value.dtype(),
                    &cpu_device(),
                )
                .ok()
            } else {
                None
            };
            if let Some(mut outputs) = fused {
                let next_param = outputs.remove(0);
                evaluator.sgd.insert(node.id, outputs.remove(0));
                next_param
            } else {
                let (next_param, next_velocity) = composed::sgd_step(
                    param_value.tensor(),
                    grad_value.tensor(),
                    velocity_value.tensor(),
                    lr_value.tensor(),
                    first_value.tensor(),
                    *momentum,
                    *dampening,
                    *nesterov,
                    *weight_decay,
                );
                evaluator.sgd.insert(node.id, Value(next_velocity));
                Value(next_param)
            }
        }
        NodeKind::SgdOut { step, index } => {
            let _ = evaluator.value(step.id)?;
            if *index == 0 {
                evaluator.value(step.id)?
            } else {
                evaluator
                    .sgd
                    .get(&step.id)
                    .cloned()
                    .ok_or_else(|| "sgd_out: step has no stored velocity".to_string())?
            }
        }
        NodeKind::FusedElementwise {
            inputs,
            strides,
            shape,
            expr,
        } => {
            let values = inputs
                .iter()
                .map(|input| evaluator.value(input.id))
                .collect::<err::Res<Vec<_>>>()?;
            let first = &values[0];
            fusion::run(
                std::slice::from_ref(expr),
                &values,
                Some(strides),
                &[],
                shape.iter().product(),
                shape,
                first.dtype(),
                &cpu_device(),
            )?
            .remove(0)
        }
        NodeKind::FusedElementwiseMulti {
            inputs,
            strides,
            shape,
            exprs,
        } => {
            let values = inputs
                .iter()
                .map(|input| evaluator.value(input.id))
                .collect::<err::Res<Vec<_>>>()?;
            let first = &values[0];
            let outputs = fusion::run(
                exprs,
                &values,
                Some(strides),
                &[],
                shape.iter().product(),
                shape,
                first.dtype(),
                &cpu_device(),
            )?;
            let head = outputs[0].clone();
            evaluator.multi.insert(node.id, outputs);
            head
        }
        NodeKind::FusedPick { of, index } => evaluator
            .multi
            .get(&of.id)
            .and_then(|outputs| outputs.get(*index as usize))
            .cloned()
            .ok_or_else(|| "fused pick: multi output missing".to_string())?,
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
            let values = inputs
                .iter()
                .map(|input| evaluator.value(input.id))
                .collect::<err::Res<Vec<_>>>()?;
            let first = &values[0];
            fusion::run_reduce(
                *op,
                expr,
                &values,
                strides,
                in_shape,
                dims,
                *keepdims,
                shape,
                first.dtype(),
                &cpu_device(),
            )?
        }
    };
    Ok(value)
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

#[napi]
pub async fn eval_lazy(
    tensors: Vec<&LazyTensor>,
    token: Option<&CancellationToken>,
) -> Result<Vec<NativeTensor>> {
    let roots = tensors
        .iter()
        .map(|tensor| tensor.node.clone())
        .collect::<Vec<_>>();
    let roots = fuse_roots_cached(&roots)?;
    run_compute(token, move |cancelled, _state| {
        let mut evaluator = Evaluator::new(&roots);
        roots
            .iter()
            .map(|root| {
                eval_node(root, cancelled, &mut evaluator)
                    .map(NativeTensor::wrap)
                    .map_err(to_napi_err)
            })
            .collect()
    })
    .await
}

fn fuse_roots_cached(roots: &[Arc<Node>]) -> Result<Vec<Arc<Node>>> {
    if std::env::var_os("EFFECT_TORCH_NO_FUSION").is_some() {
        return Ok(roots.to_vec());
    }
    type FusionCache = (u64, HashMap<Vec<u64>, (u64, Vec<Arc<Node>>)>);
    static CACHE: LazyLock<Mutex<FusionCache>> = LazyLock::new(|| Mutex::new((0, HashMap::new())));
    let key = roots.iter().map(|root| root.id).collect::<Vec<_>>();
    {
        let mut cache = CACHE.lock().map_err(|error| {
            Error::new(
                Status::GenericFailure,
                format!("fusion cache lock poisoned: {error}"),
            )
        })?;
        cache.0 += 1;
        let tick = cache.0;
        if let Some((entry_tick, fused)) = cache.1.get_mut(&key) {
            *entry_tick = tick;
            return Ok(fused.clone());
        }
    }
    let fused = fuse_roots(roots).map_err(|error| Error::new(Status::GenericFailure, error))?;
    let mut cache = CACHE.lock().map_err(|error| {
        Error::new(
            Status::GenericFailure,
            format!("fusion cache lock poisoned: {error}"),
        )
    })?;
    cache.0 += 1;
    let tick = cache.0;
    if cache.1.len() >= 32 {
        if let Some(oldest) = cache
            .1
            .iter()
            .min_by_key(|(_, (entry_tick, _))| *entry_tick)
            .map(|(key, _)| key.clone())
        {
            cache.1.remove(&oldest);
        }
    }
    cache.1.insert(key, (tick, fused.clone()));
    Ok(fused)
}

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

struct ProgramInner {
    roots: Vec<Arc<Node>>,
    slots: Vec<ProgramSlot>,
    leaves: Vec<(u64, u32)>,
    signature: String,
}

#[napi]
pub struct CompiledProgram {
    inner: ProgramInner,
}

fn scalar_binding(value: f64, dtype: DType) -> Value {
    Value(Tensor::full(&[], value, dtype))
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

#[napi]
impl CompiledProgram {
    #[napi(getter)]
    pub fn signature(&self) -> Result<String> {
        Ok(self.inner.signature.clone())
    }

    #[napi]
    pub async fn run(
        &self,
        inputs: Vec<&NativeTensor>,
        scalars: Vec<f64>,
        token: Option<&CancellationToken>,
    ) -> Result<Vec<NativeTensor>> {
        let tensor_count = self.inner.slots.iter().filter(|slot| !slot.scalar).count();
        let scalar_count = self.inner.slots.len() - tensor_count;
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
        for (slot, declared) in self.inner.slots.iter().enumerate() {
            if !declared.scalar {
                validate_tensor_input(
                    tensors.next().expect("tensor count checked"),
                    slot,
                    declared,
                )?;
            }
        }
        let slots = self.inner.slots.clone();
        let roots = self.inner.roots.clone();
        let leaves = self.inner.leaves.clone();
        let inputs = inputs
            .iter()
            .map(|input| input.value_cloned())
            .collect::<Result<Vec<_>>>()?;
        run_compute(token, move |cancelled, _state| {
            let mut bindings = HashMap::new();
            let mut tensors = inputs.iter();
            let mut scalars = scalars.iter();
            for (slot, declared) in slots.iter().enumerate() {
                let binding = if declared.scalar {
                    scalar_binding(
                        *scalars.next().expect("scalar count checked"),
                        declared.dtype,
                    )
                } else {
                    tensors.next().expect("tensor count checked").clone()
                };
                bindings.insert(slot as u64, binding);
            }
            let by_id = leaves
                .iter()
                .map(|(id, slot)| (*id, bindings[&(*slot as u64)].clone()))
                .collect::<HashMap<_, _>>();
            let mut evaluator = Evaluator::with_slots(&roots, by_id);
            roots
                .iter()
                .map(|root| {
                    eval_node(root, cancelled, &mut evaluator)
                        .map(NativeTensor::wrap)
                        .map_err(to_napi_err)
                })
                .collect()
        })
        .await
    }
}

#[napi]
pub fn compile(roots: Vec<&LazyTensor>) -> Result<CompiledProgram> {
    let roots = roots
        .iter()
        .map(|tensor| tensor.node.clone())
        .collect::<Vec<_>>();
    if roots.is_empty() {
        return Err(Error::new(
            Status::InvalidArg,
            "compile: expected at least one root",
        ));
    }
    let roots = fuse_roots(&roots).map_err(|error| Error::new(Status::GenericFailure, error))?;
    let (slots, leaves) =
        collect_program_slots(&roots).map_err(|error| Error::new(Status::InvalidArg, error))?;
    if slots.iter().any(|slot| !slot.device.is_cpu()) {
        return Err(Error::new(
            Status::InvalidArg,
            "compile: graph contains an unsupported device",
        ));
    }
    let signature = slots
        .iter()
        .map(ProgramSlot::signature)
        .collect::<Vec<_>>()
        .join(",");
    Ok(CompiledProgram {
        inner: ProgramInner {
            roots,
            slots,
            leaves,
            signature,
        },
    })
}

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
    let roots = tensors
        .iter()
        .map(|tensor| tensor.node.clone())
        .collect::<Vec<_>>();
    run_compute(token, move |cancelled, _state| {
        let mut evaluator = Evaluator::new(&roots);
        let mut values = HashMap::with_capacity(names.len());
        for (name, root) in names.iter().zip(&roots) {
            values.insert(
                name.clone(),
                eval_node(root, cancelled, &mut evaluator).map_err(to_napi_err)?,
            );
        }
        safetensors::save(&values, &metadata, &path).map_err(to_napi_err)
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

struct BlockStore {
    free: Vec<u32>,
    refcounts: Vec<u32>,
    hashes: Vec<Option<u64>>,
    by_hash: HashMap<u64, Vec<u32>>,
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
    kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_tokens: usize,
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
    // RFC 0018: per-layer recurrent state, allocated lazily from the
    // decode geometry on first use — [H, Dk, Dv] f32 per KDA layer and
    // [K-1, C] f32 per short-conv layer.
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
    slots: Vec<Arc<Mutex<SeqState>>>,
    kda: KdaGeometry,
    conv: ConvGeometry,
}

// RFC 0018: stateful KDA evaluation, one sequence slot per leading
// batch row. Each slot's [H, Dk, Dv] f32 state drives the chunked
// recurrence and is replaced by the final state.
#[allow(clippy::too_many_arguments)]
fn kda_recurrence(
    context: &KvContext,
    layer: u32,
    q: &Value,
    k: &Value,
    v: &Value,
    log_decay: &Value,
    beta: &Value,
    scale: f64,
) -> err::Res<Value> {
    let geometry = context.kda;
    if geometry.layers == 0 || (layer as usize) >= geometry.layers {
        return Err("kda recurrence: layer out of range for the decode geometry".to_string());
    }
    let dimensions = q.shape();
    let batch = dimensions[..dimensions.len() - 3].iter().product::<usize>();
    if batch != context.slots.len() {
        return Err(format!(
            "kda recurrence: batch {batch} does not match {} decode slots",
            context.slots.len()
        ));
    }
    let mut outputs = Vec::with_capacity(batch);
    for (index, slot) in context.slots.iter().enumerate() {
        let narrow = |value: &Value, width: usize| {
            value
                .tensor()
                .view(value.tensor().layout.narrow(0, index, 1))
                .contiguous()
                .view(Layout::contiguous(vec![
                    geometry.heads,
                    dimensions[dimensions.len() - 2],
                    width,
                ]))
        };
        let mut state = slot
            .lock()
            .map_err(|error| format!("kda recurrence: sequence lock poisoned: {error}"))?;
        while state.kda_states.len() < geometry.layers {
            state.kda_states.push(Tensor::zeros(
                &[geometry.heads, geometry.head_dim, geometry.value_dim],
                DType::F32,
            ));
        }
        let tokens = dimensions[dimensions.len() - 2];
        let qs = narrow(q, geometry.head_dim);
        let ks = narrow(k, geometry.head_dim);
        let vs = narrow(v, geometry.value_dim);
        let gs = narrow(log_decay, geometry.head_dim);
        let bs = narrow(beta, 1);
        // Chunked prefill right-pads: pad rows must contribute identity
        // updates (beta 0, log-decay 0) so the running state only
        // absorbs real tokens.
        let (gs, bs) = if state.advance < tokens {
            let mut mask = vec![0f32; tokens];
            mask[..state.advance].fill(1.0);
            let mask = Tensor::from_vec(mask, vec![1, tokens, 1]);
            (gs.mul(&mask), bs.mul(&mask))
        } else {
            (gs, bs)
        };
        let (out, final_state) = composed::kda_chunk_with_state(
            &qs,
            &ks,
            &vs,
            &gs,
            &bs,
            scale,
            &state.kda_states[layer as usize].clone(),
        );
        state.kda_states[layer as usize] = final_state;
        let mut out_shape = out.shape().to_vec();
        out_shape.insert(0, 1);
        outputs.push(Value(out.view(Layout::contiguous(out_shape))));
    }
    let tensors = outputs.iter().map(Value::tensor).collect::<Vec<_>>();
    Ok(Value(Tensor::cat(&tensors, 0)))
}

// RFC 0018: stateful short-conv evaluation, one sequence slot per
// leading batch row. Each slot's [K-1, C] f32 window is shifted by the
// new tokens and written back.
fn conv_state(context: &KvContext, layer: u32, x: &Value, weight: &Value) -> err::Res<Value> {
    let geometry = context.conv;
    if geometry.layers == 0 || (layer as usize) >= geometry.layers {
        return Err("conv state: layer out of range for the decode geometry".to_string());
    }
    let dimensions = x.shape();
    let batch = dimensions[..dimensions.len() - 2].iter().product::<usize>();
    if batch != context.slots.len() {
        return Err(format!(
            "conv state: batch {batch} does not match {} decode slots",
            context.slots.len()
        ));
    }
    let t = dimensions[dimensions.len() - 2];
    let in_dtype = x.dtype();
    let w32 = weight.tensor().cast(DType::F32).contiguous();
    let mut outputs = Vec::with_capacity(batch);
    for (index, slot) in context.slots.iter().enumerate() {
        let xs = x
            .tensor()
            .view(x.tensor().layout.narrow(0, index, 1))
            .contiguous()
            .view(Layout::contiguous(vec![t, geometry.channels]))
            .cast(DType::F32);
        let mut state = slot
            .lock()
            .map_err(|error| format!("conv state: sequence lock poisoned: {error}"))?;
        while state.conv_states.len() < geometry.layers {
            state.conv_states.push(Tensor::zeros(
                &[geometry.kernel - 1, geometry.channels],
                DType::F32,
            ));
        }
        let (out, new_state) = composed::short_conv1d_with_state(
            &xs,
            &w32,
            &state.conv_states[layer as usize],
            state.advance,
        );
        state.conv_states[layer as usize] = new_state;
        outputs.push(Value(out.cast(in_dtype).view(Layout::contiguous(vec![
            1,
            t,
            geometry.channels,
        ]))));
    }
    let tensors = outputs.iter().map(Value::tensor).collect::<Vec<_>>();
    Ok(Value(Tensor::cat(&tensors, 0)))
}

fn kv_attention(
    context: &KvContext,
    layer: u32,
    q: &Value,
    k: &Value,
    v: &Value,
    scale: f64,
    window: Option<usize>,
) -> err::Res<Value> {
    let dimensions = q.shape();
    let batch = dimensions[..dimensions.len() - 3].iter().product::<usize>();
    if batch != context.slots.len() {
        return Err(format!(
            "kv attention: batch {batch} does not match {} kv slots",
            context.slots.len()
        ));
    }
    if batch == 1 {
        let mut state = context.slots[0]
            .lock()
            .map_err(|error| format!("kv attention: sequence lock poisoned: {error}"))?;
        return kv_attention_slot(&context.pool, &mut state, layer, q, k, v, scale, window);
    }
    let mut outputs = Vec::with_capacity(batch);
    for (index, slot) in context.slots.iter().enumerate() {
        let narrow = |value: &Value| {
            Value(
                value
                    .tensor()
                    .view(value.tensor().layout.narrow(0, index, 1))
                    .contiguous(),
            )
        };
        let mut state = slot
            .lock()
            .map_err(|error| format!("kv attention: sequence lock poisoned: {error}"))?;
        outputs.push(kv_attention_slot(
            &context.pool,
            &mut state,
            layer,
            &narrow(q),
            &narrow(k),
            &narrow(v),
            scale,
            window,
        )?);
    }
    let tensors = outputs.iter().map(Value::tensor).collect::<Vec<_>>();
    Ok(Value(Tensor::cat(&tensors, 0)))
}

#[allow(clippy::too_many_arguments)]
fn kv_attention_slot(
    pool: &Arc<PoolInner>,
    state: &mut SeqState,
    layer: u32,
    q: &Value,
    k: &Value,
    v: &Value,
    scale: f64,
    window: Option<usize>,
) -> err::Res<Value> {
    if q.dtype() != DType::F32 {
        return Err(format!(
            "kv attention: dtype must be f32, got {:?}",
            q.dtype()
        ));
    }
    let layer = layer as usize;
    let dimensions = q.shape();
    let rank = dimensions.len();
    let (tokens, heads, width) = (
        dimensions[rank - 2],
        dimensions[rank - 3],
        dimensions[rank - 1],
    );
    let (cursor, needed, start) = kv_prepare(pool, state, layer, window, heads, width, tokens)?;
    kv_scatter_rows(pool, state, layer, k, v, heads, width)?;
    let full = cursor + tokens;
    let physical = |position: usize| -> u32 {
        state.blocks[position / pool.block_size] * pool.block_size as u32
            + (position % pool.block_size) as u32
    };
    let rows = (start..needed).map(physical).collect::<Vec<_>>();
    let context_length = full - start;
    let gather = |slab: &pool::Slab, scale: Option<&pool::Slab>| -> Tensor {
        let raw = slab.read_rows_f32(&rows);
        let mut values = match scale {
            Some(scale) => {
                pool::dequantize_int8(&raw, &scale.read_rows_f32(&rows), rows.len(), heads, width)
            }
            None => raw,
        };
        values.resize(context_length * heads * width, 0.0);
        let mut permuted = vec![0.0; context_length * heads * width];
        for row in 0..context_length {
            for head in 0..heads {
                for column in 0..width {
                    permuted[(head * context_length + row) * width + column] =
                        values[(row * heads + head) * width + column];
                }
            }
        }
        Tensor::from_vec(permuted, vec![1, heads, context_length, width])
    };
    let (k_scale, v_scale) = if pool.k[layer].dtype == DType::U8 {
        (
            Some(&pool.scales[2 * layer]),
            Some(&pool.scales[2 * layer + 1]),
        )
    } else {
        (None, None)
    };
    let output = composed::sdpa_forward(
        q.tensor(),
        &gather(&pool.k[layer], k_scale),
        &gather(&pool.v[layer], v_scale),
        scale,
        true,
    );
    kv_evict(pool, state, start);
    Ok(Value(output))
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
    let full = cursor + tokens;
    let start = window.map_or(0, |window| full.saturating_sub(window));
    if needed - start > pool.max_tokens {
        return Err(format!(
            "kv attention: live context {} exceeds pool capacity {}",
            needed - start,
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

fn kv_scatter_rows(
    pool: &Arc<PoolInner>,
    state: &SeqState,
    layer: usize,
    k: &Value,
    v: &Value,
    heads: usize,
    width: usize,
) -> err::Res<()> {
    let cursor = state.cursor;
    let advance = state.advance;
    let physical = |position: usize| -> u32 {
        state.blocks[position / pool.block_size] * pool.block_size as u32
            + (position % pool.block_size) as u32
    };
    let rows = (cursor..cursor + advance).map(physical).collect::<Vec<_>>();
    let new_rows = |value: &Value| -> Vec<f32> {
        let tensor = value.tensor();
        let permuted = tensor
            .view(tensor.layout.permute(&[0, 2, 1, 3]))
            .contiguous();
        let narrowed = permuted
            .view(permuted.layout.narrow(1, 0, advance))
            .contiguous();
        let narrowed = narrowed
            .view(Layout::contiguous(vec![advance, heads, width]))
            .cast(DType::F32)
            .contiguous();
        let CpuBuffer::F32(values) = &narrowed.buffer else {
            unreachable!()
        };
        values.as_slice().to_vec()
    };
    let k_rows = new_rows(k);
    let v_rows = new_rows(v);
    if pool.k[layer].dtype == DType::U8 {
        let (quantized_k, scales_k) = pool::quantize_int8(&k_rows, advance, heads, width);
        let (quantized_v, scales_v) = pool::quantize_int8(&v_rows, advance, heads, width);
        pool.k[layer].write_rows_u8(&rows, &quantized_k);
        pool.v[layer].write_rows_u8(&rows, &quantized_v);
        pool.scales[2 * layer].write_rows_f32(&rows, &scales_k);
        pool.scales[2 * layer + 1].write_rows_f32(&rows, &scales_v);
    } else {
        pool.k[layer].write_rows_f32(&rows, &k_rows);
        pool.v[layer].write_rows_f32(&rows, &v_rows);
    }
    Ok(())
}

fn kv_evict(pool: &PoolInner, state: &mut SeqState, start: usize) {
    while (state.head + 1) * pool.block_size <= start {
        pool.unref_block(state.blocks[state.head]);
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
    cursor_tensor: bool,
}

fn decode_rewrite(
    roots: &[Arc<Node>],
    window: Option<usize>,
    batch: usize,
) -> std::result::Result<(Vec<Arc<Node>>, DecodeGeometry), String> {
    let mut maximum_slot = None;
    let mut visited = HashSet::new();
    let mut stack = roots.to_vec();
    while let Some(node) = stack.pop() {
        if !visited.insert(node.id) {
            continue;
        }
        match &node.kind {
            NodeKind::Input { slot, .. } => {
                maximum_slot = Some(maximum_slot.map_or(*slot, |current: u32| current.max(*slot)));
            }
            NodeKind::ScalarInput { .. } => {
                return Err(
                    "decode: runtime scalar inputs are not supported in inference graphs"
                        .to_string(),
                );
            }
            _ => {}
        }
        stack.extend(node_children(&node.kind));
    }
    let cursor_slot = maximum_slot.map_or(0, |slot| slot + 1);
    let mut visited = HashSet::new();
    let mut order = Vec::new();
    let mut stack = roots
        .iter()
        .map(|root| (root.clone(), false))
        .collect::<Vec<_>>();
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
    let mut remapped = HashMap::new();
    let mut layers = 0;
    let mut kda_layers = 0;
    let mut conv_layers = 0;
    let mut cursor_tensor = false;
    let mut geometry = None;
    let mut kda_geometry = None;
    let mut conv_geometry = None;
    for node in order {
        let remap = |child: &Arc<Node>| {
            remapped
                .get(&child.id)
                .cloned()
                .unwrap_or_else(|| child.clone())
        };
        let kind = match &node.kind {
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
                let current = (k.shape[rank - 3], k.shape[rank - 1]);
                if let Some(previous) = geometry {
                    if previous != current {
                        return Err(format!(
                            "decode: attention layers disagree on head geometry ({previous:?} vs {current:?})"
                        ));
                    }
                } else {
                    geometry = Some(current);
                }
                let layer = layers;
                layers += 1;
                NodeKind::KvAttention {
                    q: remap(q),
                    k: remap(k),
                    v: remap(v),
                    scale: *scale,
                    layer,
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
                if let Some(previous) = kda_geometry {
                    if previous != current {
                        return Err(format!(
                            "decode: kda layers disagree on head geometry ({previous:?} vs {current:?})"
                        ));
                    }
                } else {
                    kda_geometry = Some(current);
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
                    layer,
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
                if let Some(previous) = conv_geometry {
                    if previous != current {
                        return Err(format!(
                            "decode: short conv layers disagree on geometry ({previous:?} vs {current:?})"
                        ));
                    }
                } else {
                    conv_geometry = Some(current);
                }
                let layer = conv_layers;
                conv_layers += 1;
                NodeKind::ConvState {
                    x: remap(x),
                    weight: remap(weight),
                    layer,
                }
            }
            NodeKind::RotaryEmbedding {
                x, seq_len, theta, ..
            } => NodeKind::RotaryEmbedding {
                x: remap(x),
                seq_len: *seq_len,
                theta: *theta,
                offset: PositionOffset::Cursor,
            },
            NodeKind::PositionEmbedding { weight, seq_len } => {
                let tokens = *seq_len;
                let width = weight.shape[1];
                if batch > 1 {
                    cursor_tensor = true;
                    let cursors = Node::new(NodeKind::Input {
                        slot: cursor_slot,
                        shape: vec![batch],
                        dtype: DType::I64,
                        device: cpu_device(),
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
                                    end: tokens as f64,
                                    step: 1.0,
                                    dtype: DType::I64,
                                    device: cpu_device(),
                                })?,
                                shape: vec![1, tokens],
                            })?,
                            shape: vec![batch, tokens],
                        })?,
                    })?;
                    let indexes = Node::new(NodeKind::BroadcastTo {
                        a: Node::new(NodeKind::Reshape {
                            a: positions,
                            shape: vec![batch * tokens, 1],
                        })?,
                        shape: vec![batch * tokens, width],
                    })?;
                    NodeKind::Reshape {
                        a: Node::new(NodeKind::Gather {
                            a: remap(weight),
                            dim: 0,
                            indexes,
                        })?,
                        shape: vec![batch, tokens, width],
                    }
                } else {
                    let positions = Node::new(NodeKind::Add {
                        a: Node::new(NodeKind::Arange {
                            start: 0.0,
                            end: tokens as f64,
                            step: 1.0,
                            dtype: DType::I64,
                            device: cpu_device(),
                        })?,
                        b: Node::new(NodeKind::ScalarInput {
                            slot: cursor_slot,
                            dtype: DType::I64,
                            device: cpu_device(),
                        })?,
                    })?;
                    let indexes = Node::new(NodeKind::BroadcastTo {
                        a: Node::new(NodeKind::Reshape {
                            a: positions,
                            shape: vec![tokens, 1],
                        })?,
                        shape: vec![tokens, width],
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
        remapped.insert(node.id, Node::new(kind)?);
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
            layers: kda_layers as usize,
            heads,
            head_dim,
            value_dim,
        })
        .unwrap_or_default();
    let conv = conv_geometry
        .map(|(channels, kernel)| ConvGeometry {
            layers: conv_layers as usize,
            channels,
            kernel,
        })
        .unwrap_or_default();
    let roots = roots
        .iter()
        .map(|root| {
            remapped
                .get(&root.id)
                .cloned()
                .unwrap_or_else(|| root.clone())
        })
        .collect();
    Ok((
        roots,
        DecodeGeometry {
            layers: layers as usize,
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
                kv_heads,
                head_dim,
                block_size,
                max_tokens,
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
    released: AtomicBool,
}

impl NativeKvSequence {
    fn new(pool: Arc<PoolInner>) -> Self {
        Self {
            pool,
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

    fn new_sequence_like(&self) -> Self {
        Self::new(self.pool.clone())
    }

    fn return_blocks(&self) {
        if self.released.swap(true, Ordering::SeqCst) {
            return;
        }
        let Ok(_run_guard) = self.run_lock.lock() else {
            return;
        };
        if let Ok(mut state) = self.state.lock() {
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
        let matchable = tokens.len().saturating_sub(1) / self.pool.block_size;
        let mut hash = HASH_SEED;
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

    #[napi]
    pub async fn run(
        &self,
        inputs: Vec<&NativeTensor>,
        sequence: &NativeKvSequence,
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
        self.run_inner(inputs, vec![sequence], vec![tokens], token)
            .await
    }

    #[napi]
    pub async fn run_batched(
        &self,
        inputs: Vec<&NativeTensor>,
        sequences: Vec<&NativeKvSequence>,
        tokens: Vec<Vec<u32>>,
        token: Option<&CancellationToken>,
    ) -> Result<Vec<NativeTensor>> {
        if sequences.is_empty() || sequences.len() > self.batch as usize {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "kv run: program accepts 1..={} sequences, got {}",
                    self.batch,
                    sequences.len()
                ),
            ));
        }
        let padding = (sequences.len()..self.batch as usize)
            .map(|_| sequences[0].new_sequence_like())
            .collect::<Vec<_>>();
        let mut all_sequences = sequences;
        all_sequences.extend(padding.iter());
        let mut all_tokens = tokens;
        let advance = all_tokens.first().map(Vec::len).unwrap_or(1);
        all_tokens.extend(std::iter::repeat(vec![0; advance]).take(padding.len()));
        let mut owned_inputs = inputs
            .iter()
            .map(|input| input.value_cloned().map(NativeTensor::wrap))
            .collect::<Result<Vec<_>>>()?;
        if let Some(last) = owned_inputs.last_mut() {
            let value = last.value_cloned()?;
            let shape = value.shape();
            if shape.len() == 2 && shape[0] < self.batch as usize {
                let zeros =
                    Tensor::zeros(&[self.batch as usize - shape[0], shape[1]], value.dtype());
                let padded = Value(Tensor::cat(&[value.tensor(), &zeros], 0));
                last.slot = Arc::new(LeafSlot::new(padded));
            }
        }
        let input_refs = owned_inputs.iter().collect::<Vec<_>>();
        let output = self
            .run_inner(input_refs, all_sequences, all_tokens, token)
            .await;
        for sequence in &padding {
            sequence.release();
        }
        output
    }
}

impl DecodeProgram {
    async fn run_inner(
        &self,
        inputs: Vec<&NativeTensor>,
        sequences: Vec<&NativeKvSequence>,
        tokens: Vec<Vec<u32>>,
        token: Option<&CancellationToken>,
    ) -> Result<Vec<NativeTensor>> {
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
        if pool.k.len() != self.layers as usize
            || pool.kv_heads != self.kv_heads as usize
            || pool.head_dim != self.head_dim as usize
        {
            return Err(Error::new(
                Status::InvalidArg,
                "kv run: pool geometry does not match the decode program",
            ));
        }
        let tensor_count = self.inner.slots.iter().filter(|slot| !slot.scalar).count();
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
        for (slot, declared) in self.inner.slots.iter().enumerate() {
            if declared.scalar || (self.cursor_tensor && slot as u32 == self.cursor_slot) {
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
                - usize::from(self.cursor_tensor && slot as u32 > self.cursor_slot);
            validate_tensor_input(inputs[input_index], slot, declared)?;
        }
        let slots = self.inner.slots.clone();
        let roots = self.inner.roots.clone();
        let leaves = self.inner.leaves.clone();
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
            kda: self.kda,
            conv: self.conv,
        });
        let mut ordered = sequences.clone();
        ordered.sort_by_key(|sequence| Arc::as_ptr(&sequence.run_lock) as usize);
        let run_locks = ordered
            .iter()
            .map(|sequence| sequence.run_lock.clone())
            .collect::<Vec<_>>();
        let states = sequences
            .iter()
            .map(|sequence| sequence.state.clone())
            .collect::<Vec<_>>();
        let cursor_slot = self.cursor_slot;
        let batched = self.batch > 1;
        let cursor_tensor = self.cursor_tensor;
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
            let mut bindings = HashMap::new();
            let mut tensors = inputs.iter();
            for (slot, declared) in slots.iter().enumerate() {
                let binding = if declared.scalar {
                    if batched || slot as u32 != cursor_slot {
                        return Err(Error::new(
                            Status::GenericFailure,
                            format!("decode: unexpected scalar slot {slot}"),
                        ));
                    }
                    let cursor = states[0]
                        .lock()
                        .map_err(|error| {
                            Error::new(
                                Status::GenericFailure,
                                format!("kv sequence lock poisoned: {error}"),
                            )
                        })?
                        .cursor;
                    scalar_binding(cursor as f64, declared.dtype)
                } else if cursor_tensor && slot as u32 == cursor_slot {
                    let cursors = states
                        .iter()
                        .map(|state| {
                            state
                                .lock()
                                .map(|state| state.cursor as i64)
                                .map_err(|error| {
                                    Error::new(
                                        Status::GenericFailure,
                                        format!("kv sequence lock poisoned: {error}"),
                                    )
                                })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Value(Tensor::from_vec(cursors, vec![batch]))
                } else {
                    tensors.next().expect("tensor count checked").clone()
                };
                bindings.insert(slot as u64, binding);
            }
            let by_id = leaves
                .iter()
                .map(|(id, slot)| (*id, bindings[&(*slot as u64)].clone()))
                .collect::<HashMap<_, _>>();
            let frontiers = states
                .iter()
                .map(|state| state.lock().map(|state| state.blocks.len()).unwrap_or(0))
                .collect::<Vec<_>>();
            // RFC 0018: recurrent state mutates in place during the
            // walk, so a failed run restores the pre-run snapshot (KV
            // blocks roll back by refcount instead).
            let kda_snapshots = states
                .iter()
                .map(|state| {
                    state
                        .lock()
                        .map(|state| state.kda_states.clone())
                        .unwrap_or_default()
                })
                .collect::<Vec<_>>();
            let conv_snapshots = states
                .iter()
                .map(|state| {
                    state
                        .lock()
                        .map(|state| state.conv_states.clone())
                        .unwrap_or_default()
                })
                .collect::<Vec<_>>();
            let rollback = || {
                for (index, state) in states.iter().enumerate() {
                    if let Ok(mut state) = state.lock() {
                        for block in state.blocks.split_off(frontiers[index]) {
                            context.pool.unref_block(block);
                        }
                        state.advance = 0;
                        state.kda_states = kda_snapshots[index].clone();
                        state.conv_states = conv_snapshots[index].clone();
                    }
                }
            };
            let mut evaluator = Evaluator::with_kv(&roots, by_id, Some(context.clone()));
            let mut outputs = Vec::with_capacity(roots.len());
            for root in &roots {
                match eval_node(root, cancelled, &mut evaluator) {
                    Ok(value) => outputs.push(NativeTensor::wrap(value)),
                    Err(error) => {
                        rollback();
                        return Err(to_napi_err(error));
                    }
                }
            }
            if !cancellation.complete() {
                rollback();
                return Err(Error::new(Status::Cancelled, "operation aborted"));
            }
            for (index, state) in states.iter().enumerate() {
                if let Ok(mut state) = state.lock() {
                    state.note_tokens(&context.pool, &tokens[index]);
                    state.cursor += state.advance;
                    state.advance = 0;
                }
            }
            Ok(outputs)
        })
        .await
    }
}

#[napi]
pub fn compile_decode(
    roots: Vec<&LazyTensor>,
    window: Option<u32>,
    batch: Option<u32>,
) -> Result<DecodeProgram> {
    let roots = roots
        .iter()
        .map(|tensor| tensor.node.clone())
        .collect::<Vec<_>>();
    if roots.is_empty() {
        return Err(Error::new(
            Status::InvalidArg,
            "compile_decode: expected at least one root",
        ));
    }
    let batch = batch.unwrap_or(1);
    if batch == 0 {
        return Err(Error::new(
            Status::InvalidArg,
            "compile_decode: batch must be positive",
        ));
    }
    let (roots, geometry) =
        decode_rewrite(&roots, window.map(|window| window as usize), batch as usize)
            .map_err(|error| Error::new(Status::GenericFailure, error))?;
    let roots = fuse_roots(&roots).map_err(|error| Error::new(Status::GenericFailure, error))?;
    let (slots, leaves) =
        collect_program_slots(&roots).map_err(|error| Error::new(Status::InvalidArg, error))?;
    let signature = slots
        .iter()
        .map(ProgramSlot::signature)
        .collect::<Vec<_>>()
        .join(",");
    Ok(DecodeProgram {
        inner: ProgramInner {
            roots,
            slots,
            leaves,
            signature,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(tensor: Tensor) -> Arc<Node> {
        Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(Value(tensor))))).unwrap()
    }

    fn eval_f32(node: &Arc<Node>) -> Vec<f32> {
        let cancelled = CancellationFlag::new();
        let mut evaluator = Evaluator::new(std::slice::from_ref(node));
        eval_node(node, &cancelled, &mut evaluator)
            .unwrap()
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
            kv_heads: 2,
            head_dim: 4,
            block_size: 4,
            max_tokens: 8,
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
            kda: KdaGeometry::default(),
            conv: ConvGeometry::default(),
        };
        let q = Value(Tensor::from_vec(
            (0..24).map(|value| value as f32).collect(),
            vec![1, 2, 3, 4],
        ));
        let k = Value(Tensor::from_vec(
            (24..48).map(|value| value as f32 * 0.01).collect(),
            vec![1, 2, 3, 4],
        ));
        let v = Value(Tensor::from_vec(
            (48..72).map(|value| value as f32 * 0.01).collect(),
            vec![1, 2, 3, 4],
        ));
        let actual = kv_attention(&context, 0, &q, &k, &v, 0.5, None)
            .unwrap()
            .to_f32_vec()
            .unwrap();
        let expected = Value(composed::sdpa_forward(
            q.tensor(),
            k.tensor(),
            v.tensor(),
            0.5,
            true,
        ))
        .to_f32_vec()
        .unwrap();
        assert_close(&actual, &expected, 1e-6, "kv attention");
        assert_eq!(state.lock().unwrap().advance, 3);
    }

    fn block_store_pool(blocks: usize) -> PoolInner {
        PoolInner {
            k: Vec::new(),
            v: Vec::new(),
            scales: Vec::new(),
            kv_heads: 1,
            head_dim: 1,
            block_size: 2,
            max_tokens: blocks * 2,
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

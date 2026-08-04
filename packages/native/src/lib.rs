use runtime::dtype::DType;
use dev::Device;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

mod dev;
mod safetensors;
mod err;
mod val;
mod fusion;
mod runtime;
mod tokenizer;

use runtime::metal::{composed, flash, layer_norm, linear, loss, paged, rotary};
use runtime::metal::ops as metal_ops;
use err::to_napi_err;

fn to_join_err(err: tokio::task::JoinError) -> Error {
    Error::new(Status::GenericFailure, err.to_string())
}

fn conv_check(
    op: &str,
    x: &Node,
    w: &Node,
    stride: usize,
    _padding: usize,
    dilation: usize,
    groups: usize,
) -> std::result::Result<(), String> {
    if stride < 1 || dilation < 1 || groups < 1 {
        return Err(format!(
            "{op}: stride, dilation and groups must be >= 1, got {stride}, {dilation}, {groups}"
        ));
    }
    let c_in = x.shape[1];
    let c_out = w.shape[0];
    if c_in % groups != 0 || c_out % groups != 0 {
        return Err(format!(
            "{op}: channels [{c_in}, {c_out}] are not divisible into {groups} groups"
        ));
    }
    if w.shape[1] != c_in / groups {
        return Err(format!(
            "{op}: weight has {} input channels per group, expected {}",
            w.shape[1],
            c_in / groups
        ));
    }
    if !x.dtype.is_float() || x.dtype != w.dtype {
        return Err(format!(
            "{op}: dtypes must be floating point and match, got {:?} and {:?}",
            x.dtype, w.dtype
        ));
    }
    if !x.device.same_device(&w.device) {
        return Err(format!("{op}: input and weight must be on the same device"));
    }
    Ok(())
}

// Validates q [.., T, D], k [.., S, D], v [.., S, Dv] and returns the
// attention output shape [.., T, Dv]. Leading dims must match exactly.
fn sdpa_check(op: &str, q: &Node, k: &Node, v: &Node) -> std::result::Result<Vec<usize>, String> {
    let rank = q.shape.len();
    if rank < 2 || k.shape.len() != rank || v.shape.len() != rank {
        return Err(format!(
            "{op}: q, k and v must share a rank >= 2, got {:?}, {:?} and {:?}",
            q.shape, k.shape, v.shape
        ));
    }
    if q.shape[..rank - 2] != k.shape[..rank - 2] || q.shape[..rank - 2] != v.shape[..rank - 2] {
        return Err(format!(
            "{op}: leading dims must match, got {:?}, {:?} and {:?}",
            q.shape, k.shape, v.shape
        ));
    }
    if q.shape[rank - 1] != k.shape[rank - 1] {
        return Err(format!(
            "{op}: q and k head dims mismatch, got {:?} and {:?}",
            q.shape, k.shape
        ));
    }
    if k.shape[rank - 2] != v.shape[rank - 2] {
        return Err(format!(
            "{op}: k and v sequence lengths mismatch, got {:?} and {:?}",
            k.shape, v.shape
        ));
    }
    if !matches!(q.dtype, DType::F32 | DType::F64) {
        return Err(format!("{op}: dtype must be f32 or f64, got {:?}", q.dtype));
    }
    if k.dtype != q.dtype || v.dtype != q.dtype {
        return Err(format!("{op}: q, k and v must share a dtype, got {:?}, {:?} and {:?}", q.dtype, k.dtype, v.dtype));
    }
    if !k.device.same_device(&q.device) || !v.device.same_device(&q.device) {
        return Err(format!("{op}: q, k and v must be on the same device"));
    }
    let mut out = q.shape[..rank - 1].to_vec();
    out.push(v.shape[rank - 1]);
    Ok(out)
}

fn conv_out_dim(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> std::result::Result<usize, String> {
    let effective = dilation * (kernel - 1) + 1;
    if input + 2 * padding < effective {
        return Err(format!(
            "conv: kernel of effective size {effective} exceeds the padded input size {}",
            input + 2 * padding
        ));
    }
    Ok((input + 2 * padding - effective) / stride + 1)
}

#[cfg(debug_assertions)]
fn exported_buffers() -> &'static Mutex<std::collections::HashSet<usize>> {
    static BUFFERS: OnceLock<Mutex<std::collections::HashSet<usize>>> = OnceLock::new();
    BUFFERS.get_or_init(|| Mutex::new(std::collections::HashSet::new()))
}

#[cfg(debug_assertions)]
fn try_register_export(addr: usize) -> bool {
    exported_buffers().lock().unwrap().insert(addr)
}

#[cfg(not(debug_assertions))]
fn try_register_export(_addr: usize) -> bool {
    true
}

fn unregister_export(#[allow(unused)] addr: usize) {
    #[cfg(debug_assertions)]
    exported_buffers().lock().unwrap().remove(&addr);
}

enum FinalizeHint {
    ZeroCopy {
        tensor: val::Val,
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
    match *hint {
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
    hint: FinalizeHint,
}

unsafe impl Send for Readback {}

impl ToNapiValue for Readback {
    unsafe fn to_napi_value(
        env: napi::sys::napi_env,
        value: Self,
    ) -> Result<napi::sys::napi_value> {
        let hint = Box::into_raw(Box::new(value.hint)) as *mut std::ffi::c_void;
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
        Ok(result)
    }
}

fn vec_to_bytes<T>(mut vec: Vec<T>) -> (usize, *mut u8, usize, usize) {
    let ptr = vec.as_mut_ptr() as *mut u8;
    let len = vec.len() * std::mem::size_of::<T>();
    let cap = vec.capacity() * std::mem::size_of::<T>();
    let addr = ptr as usize;
    std::mem::forget(vec);
    (addr, ptr, len, cap)
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

fn get_device(device: Option<String>) -> Result<Device> {
    match device.as_deref().unwrap_or("cpu") {
        "cpu" => Ok(Device::Cpu),
        "metal" => {
            #[cfg(target_os = "macos")]
            {
                static METAL: OnceLock<dev::Device> = OnceLock::new();
                Ok(METAL.get_or_init(|| dev::Device::Metal).clone())
            }
            #[cfg(not(target_os = "macos"))]
            {
                Err(Error::new(
                    Status::InvalidArg,
                    "metal device is only available on macOS builds".to_string(),
                ))
            }
        }
        "cuda" => Err(Error::new(
            Status::InvalidArg,
            "cuda is not a supported device (supported: cpu, metal)".to_string(),
        )),
        other => Err(Error::new(
            Status::InvalidArg,
            format!("unsupported device: {other}"),
        )),
    }
}

#[napi(custom_finalize)]
pub struct NativeTensor {
    pub(crate) inner: val::Val,
    bytes: i64,
}

impl NativeTensor {
    fn wrap(inner: val::Val) -> Self {
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
        Self { inner, bytes }
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
    fn finalize(self, env: Env) -> Result<()> {
        EXTERNAL_MEMORY_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
        sync_v8(&env);
        Ok(())
    }
}

#[napi]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
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
            cancelled: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(tokio::sync::Notify::new()),
        }
    }

    #[napi]
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
        self.notify.notify_waiters();
    }

    #[napi(getter)]
    pub fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

#[napi]
impl NativeTensor {
    #[napi(getter)]
    pub fn shape(&self) -> Vec<u32> {
        self.inner.shape().iter().map(|&d| d as u32).collect()
    }

    #[napi(getter)]
    pub fn dtype(&self) -> String {
        self.inner.dtype().name().to_string()
    }

    #[napi(getter)]
    pub fn device(&self) -> String {
        self.inner.device().name().to_string()
    }

    #[napi(ts_return_type = "Promise<ArrayBuffer>")]
    pub async fn readback(&self, token: Option<&CancellationToken>) -> Result<Readback> {
        let inner = self.inner.clone();
        let compute = tokio::task::spawn_blocking(move || readback_blocking(&inner));
        match token {
            Some(token) => {
                if token.cancelled.load(Ordering::Relaxed) {
                    return Err(Error::new(
                        Status::Cancelled,
                        "operation aborted".to_string(),
                    ));
                }
                let notify = token.notify.clone();
                tokio::select! {
                    result = compute => result.map_err(to_join_err)?,
                    _ = notify.notified() => Err(Error::new(
                        Status::Cancelled,
                        "operation aborted".to_string(),
                    )),
                }
            }
            None => compute.await.map_err(to_join_err)?,
        }
    }
}

fn readback_blocking(inner: &val::Val) -> Result<Readback> {
    // f16/bf16 read back as f32: JS has no half typed arrays we can
    // rely on, and the conversion keeps the destructor surface small.
    let flat = if matches!(inner.dtype(), runtime::dtype::DType::F16 | runtime::dtype::DType::BF16) {
        match inner {
            val::Val::Cpu(t) => val::Val::Cpu(t.cast(runtime::dtype::DType::F32)),
            val::Val::Metal(t) => val::Val::Metal(
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
    let (base, offset, _dtype, elem_count) = match &flat {
        val::Val::Cpu(t) => {
            let t = t.contiguous();
            let elem_size = t.dtype().size_in_bytes();
            let base = match &t.buffer {
                runtime::cpu::CpuBuffer::U8(v) => v.as_ptr() as *const u8,
                runtime::cpu::CpuBuffer::U32(v) => v.as_ptr() as *const u8,
                runtime::cpu::CpuBuffer::I64(v) => v.as_ptr() as *const u8,
                runtime::cpu::CpuBuffer::BF16(v) => v.as_ptr() as *const u8,
                runtime::cpu::CpuBuffer::F16(v) => v.as_ptr() as *const u8,
                runtime::cpu::CpuBuffer::F32(v) => v.as_ptr() as *const u8,
                runtime::cpu::CpuBuffer::F64(v) => v.as_ptr() as *const u8,
            };
            let offset = t.layout.offset() * elem_size;
            let keep = val::Val::Cpu(t);
            (base, offset, keep, 0usize)
        }
        val::Val::Metal(t) => {
            let t = if t.layout.is_contiguous() {
                t.clone()
            } else {
                val::Val::Metal(
                    runtime::metal::kernels::strided_copy(runtime::metal::device::MetalDevice::get(), t)
                        .map_err(to_napi_err)?,
                )
                .as_metal()
                .map_err(to_napi_err)?
                .clone()
            };
            runtime::metal::device::MetalDevice::get().synchronize();
            let base = t.buffer.contents_ptr() as *const u8;
            let offset = t.layout.offset() * t.dtype.size_in_bytes();
            let keep = val::Val::Metal(t);
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
                hint: FinalizeHint::ZeroCopy {
                    tensor: flat.clone(),
                    addr,
                },
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
        hint: FinalizeHint::Owned { ptr, len, cap },
    })
}

// Where a position-indexed semantic node reads its base position:
// zero in user graphs, the sequence cursor in decode-rewritten ones.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PositionOffset {
    Absolute,
    Cursor,
}

enum NodeKind {
    Leaf(val::Val),
    // RFC 0008: placeholder leaves for compiled programs. An Input carries
    // the declared signature of one call argument; it evaluates only inside
    // CompiledProgram::run, which binds the slot to an argument buffer.
    Input {
        slot: u32,
        shape: Vec<usize>,
        dtype: runtime::dtype::DType,
        device: dev::Device,
    },
    ScalarInput {
        slot: u32,
        dtype: runtime::dtype::DType,
        device: dev::Device,
    },
    FromBytes {
        data: Vec<u8>,
        shape: Vec<usize>,
        dtype: runtime::dtype::DType,
        device: dev::Device,
    },
    Zeros {
        shape: Vec<usize>,
        dtype: runtime::dtype::DType,
        device: dev::Device,
    },
    Ones {
        shape: Vec<usize>,
        dtype: runtime::dtype::DType,
        device: dev::Device,
    },
    Full {
        shape: Vec<usize>,
        value: f64,
        dtype: runtime::dtype::DType,
        device: dev::Device,
    },
    Randn {
        shape: Vec<usize>,
        dtype: runtime::dtype::DType,
        device: dev::Device,
    },
    Uniform {
        lo: f64,
        hi: f64,
        shape: Vec<usize>,
        dtype: runtime::dtype::DType,
        device: dev::Device,
    },
    Arange {
        start: f64,
        end: f64,
        step: f64,
        dtype: runtime::dtype::DType,
        device: dev::Device,
    },
    Eye {
        n: usize,
        dtype: runtime::dtype::DType,
        device: dev::Device,
    },
    Add {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Sub {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Mul {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Div {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Eq {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Gt {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Lt {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Ge {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Le {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Maximum {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Minimum {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Neg {
        a: Arc<Node>,
    },
    Abs {
        a: Arc<Node>,
    },
    Sqrt {
        a: Arc<Node>,
    },
    Exp {
        a: Arc<Node>,
    },
    Log {
        a: Arc<Node>,
    },
    Sin {
        a: Arc<Node>,
    },
    Cos {
        a: Arc<Node>,
    },
    Tanh {
        a: Arc<Node>,
    },
    Relu {
        a: Arc<Node>,
    },
    Erf {
        a: Arc<Node>,
    },
    Floor {
        a: Arc<Node>,
    },
    Ceil {
        a: Arc<Node>,
    },
    Round {
        a: Arc<Node>,
    },
    Sign {
        a: Arc<Node>,
    },
    Where {
        cond: Arc<Node>,
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Pow {
        a: Arc<Node>,
        exp: f64,
    },
    Cast {
        a: Arc<Node>,
        dtype: runtime::dtype::DType,
    },
    Sum {
        a: Arc<Node>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    Mean {
        a: Arc<Node>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    Max {
        a: Arc<Node>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    Min {
        a: Arc<Node>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    Prod {
        a: Arc<Node>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    Argmax {
        a: Arc<Node>,
        dim: usize,
    },
    Argmin {
        a: Arc<Node>,
        dim: usize,
    },
    Cumsum {
        a: Arc<Node>,
        dim: usize,
    },
    IndexSelect {
        a: Arc<Node>,
        dim: usize,
        indexes: Arc<Node>,
    },
    ScatterAdd {
        a: Arc<Node>,
        dim: usize,
        indexes: Arc<Node>,
        src: Arc<Node>,
    },
    Gather {
        a: Arc<Node>,
        dim: usize,
        indexes: Arc<Node>,
    },
    CrossEntropy {
        logits: Arc<Node>,
        target: Arc<Node>,
        ignore_index: i64,
    },
    CrossEntropyBackward {
        logits: Arc<Node>,
        target: Arc<Node>,
        ignore_index: i64,
    },
    // Scaled dot-product attention as one semantic node (the SgdStep
    // precedent: semantics in the graph, execution strategy native). The
    // eval arms compose candle ops as the reference implementation; a
    // fused flash kernel can replace them without touching the graph or
    // its adjoints. Shapes: q [.., T, D], k [.., S, D], v [.., S, Dv]
    // with equal leading dims; the output is [.., T, Dv].
    Sdpa {
        q: Arc<Node>,
        k: Arc<Node>,
        v: Arc<Node>,
        scale: f64,
        causal: bool,
    },
    // Closed-form backward: recomputes P = softmax(scores) from q/k and
    // produces (dq, dk, dv) in one eval; consumers read them through
    // SdpaBackwardOut. On Metal the forward's flash kernel stashes the
    // per-row logsumexp in the evaluator, and this node runs the
    // chunked-recompute backward against it (bounded memory); elsewhere
    // it recomputes P with composed candle ops. `fwd` is the Sdpa node
    // this is the adjoint of (reads its output and stashed L). Not
    // differentiable (no second-order).
    SdpaBackward {
        q: Arc<Node>,
        k: Arc<Node>,
        v: Arc<Node>,
        g: Arc<Node>,
        fwd: Arc<Node>,
        scale: f64,
        causal: bool,
    },
    SdpaBackwardOut {
        of: Arc<Node>,
        index: u8,
    },
    // Absolute position embedding as one semantic node: rows 0..seq_len
    // of the [max_positions, E] weight table (the Sdpa precedent —
    // semantics in the graph, execution strategy native). Semantic so
    // the RFC 0010 decode rewrite can offset the positions by the
    // runtime cursor instead of re-deriving "this gather is a position
    // embedding" from composed ops.
    PositionEmbedding {
        weight: Arc<Node>,
        seq_len: usize,
    },
    // RFC 0010: paged KV attention, the decode/prefill semantic node
    // produced by the decode rewrite (never written by user code —
    // `compile_decode` turns each causal Sdpa into one). Scatters the
    // new tokens' k/v into the sequence's pool blocks at the cursor,
    // then attends q causally over the last `window` cached positions
    // (None: the whole context). q, k and v are [1, H, T, D]/[1, H, T, Dv]
    // with a shared T (the new tokens); the pool, block table and
    // cursor arrive via the run's kv context, keeping the graph a pure
    // function of its inputs. Not differentiable.
    KvAttention {
        q: Arc<Node>,
        k: Arc<Node>,
        v: Arc<Node>,
        scale: f64,
        layer: u32,
        window: Option<usize>,
    },
    // RoPE (GPT-NeoX half-split rotary) as one semantic node: x is
    // [.., T, D] with D even; the last dim rotates in half pairs by
    // (offset + position) * theta^(-2j/D). Absolute positions ride the
    // tensor, attention sees only offsets — so cached K/V stay valid as
    // the context grows, which learned absolute embeddings cannot do.
    // `offset` is Absolute in user graphs; the decode rewrite flips it
    // to Cursor (the run's kv cursor) instead of re-deriving "this
    // arange is a position" from composed ops.
    RotaryEmbedding {
        x: Arc<Node>,
        seq_len: usize,
        theta: f64,
        offset: PositionOffset,
    },
    // Backward of RotaryEmbedding (absolute positions only): the
    // transpose rotation, evaluated by the same fused kernel with
    // negated angles. Carries the input's shape/seq_len for metadata.
    RotaryEmbeddingBackward {
        g: Arc<Node>,
        shape: Vec<usize>,
        seq_len: usize,
        theta: f64,
    },
    // Layer normalization over the last dim: y = (x − μ)/√(σ² + eps) ·
    // weight + bias. Semantic node (like RotaryEmbedding) so the fused
    // Metal kernel handles it as one launch and decode compilation can
    // pass it through.
    LayerNorm {
        x: Arc<Node>,
        weight: Arc<Node>,
        bias: Arc<Node>,
        eps: f64,
    },
    // Backward of LayerNorm: evaluates dx (its own value) and stores
    // (dw, db) for LayerNormBackwardOut, like the optimizer steps.
    LayerNormBackward {
        x: Arc<Node>,
        weight: Arc<Node>,
        g: Arc<Node>,
        eps: f64,
    },
    // Reads one weight-side output of a LayerNormBackward (1 = dw,
    // 2 = db).
    LayerNormBackwardOut {
        of: Arc<Node>,
        index: u8,
    },
    // Fused linear layer: y = x·W + b in one gemm launch (addmm
    // epilogue on Metal). Semantic node — Model.linear and attention
    // projections build it directly.
    Linear {
        x: Arc<Node>,
        weight: Arc<Node>,
        bias: Arc<Node>,
    },
    Conv1d {
        x: Arc<Node>,
        w: Arc<Node>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    },
    Conv2d {
        x: Arc<Node>,
        w: Arc<Node>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    },
    ConvTranspose1d {
        x: Arc<Node>,
        w: Arc<Node>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    },
    ConvTranspose2d {
        x: Arc<Node>,
        w: Arc<Node>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    },
    Conv1dBackwardW {
        x: Arc<Node>,
        g: Arc<Node>,
        kernel: usize,
        out_channels: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    },
    Conv2dBackwardW {
        x: Arc<Node>,
        g: Arc<Node>,
        kernel: [usize; 2],
        out_channels: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    },
    Reshape {
        a: Arc<Node>,
        shape: Vec<usize>,
    },
    Permute {
        a: Arc<Node>,
        dims: Vec<usize>,
    },
    Slice {
        a: Arc<Node>,
        ranges: Vec<(usize, usize, usize)>,
    },
    Concat {
        a: Arc<Node>,
        b: Arc<Node>,
        dim: usize,
    },
    BroadcastTo {
        a: Arc<Node>,
        shape: Vec<usize>,
    },
    Matmul {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Inverse {
        a: Arc<Node>,
    },
    Det {
        a: Arc<Node>,
    },
    Solve {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    // lr, c1 (1 - beta1^t) and c2 (1 - beta2^t) are 0-d tensor children:
    // step-varying values flow through the graph so a frozen graph (RFC
    // 0008) never replays a stale step count or learning rate.
    AdamWStep {
        param: Arc<Node>,
        grad: Arc<Node>,
        m: Arc<Node>,
        v: Arc<Node>,
        lr: Arc<Node>,
        c1: Arc<Node>,
        c2: Arc<Node>,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    },
    AdamWOut {
        step: Arc<Node>,
        index: u8,
    },
    // Freeze-time grouping of same-shape AdamW steps (the endgame for
    // the optimizer: one fused launch per group instead of one per
    // parameter — ≤4 params, Metal's 31-buffer limit: 4 lanes + 3
    // outputs each plus 3 scalars).
    AdamWStepGroup {
        params: Vec<Arc<Node>>,
        grads: Vec<Arc<Node>>,
        ms: Vec<Arc<Node>>,
        vs: Vec<Arc<Node>>,
        lr: Arc<Node>,
        c1: Arc<Node>,
        c2: Arc<Node>,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    },
    // One output of a grouped step: `param`-th parameter's updated
    // param (0), m (1), or v (2).
    AdamWGroupOut {
        of: Arc<Node>,
        param: u32,
        index: u8,
    },
    // `first` is a 0-d flag (1.0 on the first step, 0.0 after) selecting
    // v = g over v = momentum * v + (1 - dampening) * g; velocity is always
    // a real buffer (zeros at init), so no placeholder is needed.
    SgdStep {
        param: Arc<Node>,
        grad: Arc<Node>,
        velocity: Arc<Node>,
        first: Arc<Node>,
        lr: Arc<Node>,
        momentum: f64,
        dampening: f64,
        nesterov: bool,
        weight_decay: f64,
    },
    SgdOut {
        step: Arc<Node>,
        index: u8,
    },
    // Created only by the evaluation-time fusion rewrite (RFC 0007 phase
    // 2): a maximal chain of elementwise ops compiled to one kernel. Never
    // appears in user graphs, so autodiff and vmap reject it. Input lanes
    // may be broadcast-smaller than the output: `strides` gives each
    // lane's strides in output-dim space (0 = broadcast along that dim).
    FusedElementwise {
        inputs: Vec<Arc<Node>>,
        strides: Vec<Vec<usize>>,
        shape: Vec<usize>,
        expr: fusion::Expr,
    },
    // Created by the multi-output post-pass (RFC 0007): a shared fused
    // prefix and its fused continuations compiled to one kernel with one
    // store per output. Consumers are FusedPick nodes.
    FusedElementwiseMulti {
        inputs: Vec<Arc<Node>>,
        strides: Vec<Vec<usize>>,
        shape: Vec<usize>,
        exprs: Vec<fusion::Expr>,
    },
    // Reads one output of a FusedElementwiseMulti.
    FusedPick {
        of: Arc<Node>,
        index: u8,
    },
    // Created only by the evaluation-time fusion rewrite (RFC 0007 phase
    // 3a): an elementwise chain terminated by a single reduce, compiled to
    // one kernel that evaluates the chain inside the reduce loop — the
    // chain's intermediate never materializes. `strides` are per-lane in
    // input-dim space; `dims` sorted ascending; `shape` is the reduced
    // shape with keepdims applied.
    FusedReduce {
        inputs: Vec<Arc<Node>>,
        strides: Vec<Vec<usize>>,
        in_shape: Vec<usize>,
        expr: fusion::Expr,
        op: fusion::ReduceOp,
        dims: Vec<usize>,
        keepdims: bool,
        shape: Vec<usize>,
    },
    StopGradient {
        a: Arc<Node>,
    },
    Checkpoint {
        a: Arc<Node>,
    },
}

pub struct Node {
    id: u64,
    shape: Vec<usize>,
    dtype: runtime::dtype::DType,
    device: dev::Device,
    kind: NodeKind,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

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

fn device_key(device: &dev::Device) -> &'static str {
    match device {
        dev::Device::Cpu => "cpu",
        dev::Device::Metal => "metal",
    }
}

fn cached_constant(value: f64, dtype: DType, device: Device) -> std::result::Result<Arc<Node>, String> {
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

fn broadcast_shapes(a: &[usize], b: &[usize]) -> std::result::Result<Vec<usize>, String> {
    let rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let da = if i < rank - a.len() { 1 } else { a[i - (rank - a.len())] };
        let db = if i < rank - b.len() { 1 } else { b[i - (rank - b.len())] };
        if da != db && da != 1 && db != 1 {
            return Err(format!("shapes {a:?} and {b:?} are not broadcastable"));
        }
        out.push(da.max(db));
    }
    Ok(out)
}

fn reduced_shape(shape: &[usize], dims: &[usize], keepdims: bool) -> Vec<usize> {
    if keepdims {
        shape
            .iter()
            .enumerate()
            .map(|(i, &d)| if dims.contains(&i) { 1 } else { d })
            .collect()
    } else {
        shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !dims.contains(i))
            .map(|(_, &d)| d)
            .collect()
    }
}

impl Node {
    fn new(kind: NodeKind) -> std::result::Result<Arc<Node>, String> {
        let (shape, dtype, device) = match &kind {
            NodeKind::Leaf(tensor) => (
                tensor.shape(),
                tensor.dtype(),
                match tensor.device() {
                    dev::Device::Cpu => dev::Device::Cpu,
                    dev::Device::Metal => dev::Device::Metal,
                },
            ),
            NodeKind::Input {
                shape,
                dtype,
                device,
                ..
            } => (shape.clone(), *dtype, device.clone()),
            NodeKind::ScalarInput { dtype, device, .. } => (vec![], *dtype, device.clone()),
            NodeKind::FromBytes {
                shape,
                dtype,
                device,
                ..
            }
            | NodeKind::Zeros {
                shape,
                dtype,
                device,
            }
            | NodeKind::Ones {
                shape,
                dtype,
                device,
            }
            |             NodeKind::Randn {
                shape,
                dtype,
                device,
            } => (shape.clone(), *dtype, device.clone()),
            NodeKind::Uniform {
                lo,
                hi,
                shape,
                dtype,
                device,
            } => {
                if !dtype.is_float() {
                    return Err(format!("uniform: dtype must be floating point, got {dtype:?}"));
                }
                if hi <= lo {
                    return Err(format!("uniform: expected lo < hi, got lo={lo} hi={hi}"));
                }
                (shape.clone(), *dtype, device.clone())
            }
            NodeKind::Full {
                shape,
                dtype,
                device,
                ..
            } => (shape.clone(), *dtype, device.clone()),
            NodeKind::Arange {
                start,
                end,
                step,
                dtype,
                device,
            } => {
                let n = ((end - start) / step).ceil().max(0.0) as usize;
                (vec![n], *dtype, device.clone())
            }
            NodeKind::Eye { n, dtype, device } => (vec![*n, *n], *dtype, device.clone()),
            NodeKind::Add { a, b }
            | NodeKind::Sub { a, b }
            | NodeKind::Mul { a, b }
            | NodeKind::Div { a, b }
            | NodeKind::Maximum { a, b }
            | NodeKind::Minimum { a, b } => (
                broadcast_shapes(&a.shape, &b.shape)?,
                a.dtype,
                a.device.clone(),
            ),
            NodeKind::Eq { a, b }
            | NodeKind::Gt { a, b }
            | NodeKind::Lt { a, b }
            | NodeKind::Ge { a, b }
            | NodeKind::Le { a, b } => (
                broadcast_shapes(&a.shape, &b.shape)?,
                DType::U8,
                a.device.clone(),
            ),
            NodeKind::Neg { a }
            | NodeKind::Abs { a }
            | NodeKind::Sqrt { a }
            | NodeKind::Exp { a }
            | NodeKind::Log { a }
            | NodeKind::Sin { a }
            | NodeKind::Cos { a }
            | NodeKind::Tanh { a }
            | NodeKind::Relu { a }
            | NodeKind::Erf { a }
            | NodeKind::Floor { a }
            | NodeKind::Ceil { a }
            | NodeKind::Round { a }
            | NodeKind::Sign { a }
            | NodeKind::Checkpoint { a }
            | NodeKind::StopGradient { a } => (a.shape.clone(), a.dtype, a.device.clone()),
            NodeKind::Pow { a, .. } => (a.shape.clone(), a.dtype, a.device.clone()),
            NodeKind::Where { cond, a, b } => {
                if cond.dtype != DType::U8 {
                    return Err(format!("where: condition must be u8, got {:?}", cond.dtype));
                }
                if a.dtype != b.dtype {
                    return Err(format!(
                        "where: dtype mismatch, got {:?} and {:?}",
                        a.dtype, b.dtype
                    ));
                }
                let shape = broadcast_shapes(&cond.shape, &a.shape)?;
                let shape = broadcast_shapes(&shape, &b.shape)?;
                (shape, a.dtype, a.device.clone())
            }
            NodeKind::Argmax { a, dim } | NodeKind::Argmin { a, dim } => {
                if a.shape.is_empty() || *dim >= a.shape.len() {
                    return Err(format!(
                        "argmax/argmin: dim {dim} out of range for rank {}",
                        a.shape.len()
                    ));
                }
                let mut shape = a.shape.clone();
                shape.remove(*dim);
                (shape, DType::I64, a.device.clone())
            }
            NodeKind::Cumsum { a, dim } => {
                if a.shape.is_empty() || *dim >= a.shape.len() {
                    return Err(format!(
                        "cumsum: dim {dim} out of range for rank {}",
                        a.shape.len()
                    ));
                }
                (a.shape.clone(), a.dtype, a.device.clone())
            }
            NodeKind::IndexSelect { a, dim, indexes } => {
                if a.shape.is_empty() || *dim >= a.shape.len() {
                    return Err(format!(
                        "index_select: dim {dim} out of range for rank {}",
                        a.shape.len()
                    ));
                }
                if !matches!(indexes.dtype, DType::I64 | DType::U32) {
                    return Err(format!(
                        "index_select: indexes must be i64 or u32, got {:?}",
                        indexes.dtype
                    ));
                }
                if indexes.shape.len() != 1 {
                    return Err(format!(
                        "index_select: indexes must be 1-D, got shape {:?}",
                        indexes.shape
                    ));
                }
                let mut shape = a.shape.clone();
                shape[*dim] = indexes.shape[0];
                (shape, a.dtype, a.device.clone())
            }
            NodeKind::ScatterAdd {
                a,
                dim,
                indexes,
                src,
            } => {
                if a.shape.is_empty() || *dim >= a.shape.len() {
                    return Err(format!(
                        "scatter_add: dim {dim} out of range for rank {}",
                        a.shape.len()
                    ));
                }
                if !matches!(indexes.dtype, DType::I64 | DType::U32) {
                    return Err(format!(
                        "scatter_add: indexes must be i64 or u32, got {:?}",
                        indexes.dtype
                    ));
                }
                if indexes.shape != src.shape {
                    return Err(format!(
                        "scatter_add: indexes shape {:?} must match src shape {:?}",
                        indexes.shape, src.shape
                    ));
                }
                if src.dtype != a.dtype {
                    return Err(format!(
                        "scatter_add: src dtype {:?} does not match target dtype {:?}",
                        src.dtype, a.dtype
                    ));
                }
                (a.shape.clone(), a.dtype, a.device.clone())
            }
            NodeKind::Gather { a, dim, indexes } => {
                if a.shape.is_empty() || *dim >= a.shape.len() {
                    return Err(format!(
                        "gather: dim {dim} out of range for rank {}",
                        a.shape.len()
                    ));
                }
                if !matches!(indexes.dtype, DType::I64 | DType::U32) {
                    return Err(format!("gather: indexes must be i64 or u32, got {:?}", indexes.dtype));
                }
                if indexes.shape.len() != a.shape.len() {
                    return Err(format!(
                        "gather: indexes rank {} must match input rank {}",
                        indexes.shape.len(),
                        a.shape.len()
                    ));
                }
                for i in 0..a.shape.len() {
                    if i != *dim && indexes.shape[i] > a.shape[i] {
                        return Err(format!(
                            "gather: indexes shape {:?} exceeds input shape {:?} at dim {i}",
                            indexes.shape, a.shape
                        ));
                    }
                }
                (indexes.shape.clone(), a.dtype, a.device.clone())
            }
            NodeKind::CrossEntropy {
                logits,
                target,
                ignore_index: _,
            } => {
                let rank = logits.shape.len();
                if rank < 1 {
                    return Err("cross_entropy: logits must have rank >= 1".to_string());
                }
                if logits.shape[rank - 1] == 0 {
                    return Err("cross_entropy: class dimension must be non-empty".to_string());
                }
                if !matches!(logits.dtype, DType::F32 | DType::F64) {
                    return Err(format!(
                        "cross_entropy: logits must be f32 or f64, got {:?}",
                        logits.dtype
                    ));
                }
                if !matches!(target.dtype, DType::I64 | DType::U32) {
                    return Err(format!(
                        "cross_entropy: targets must be i64 or u32, got {:?}",
                        target.dtype
                    ));
                }
                if target.shape != logits.shape[..rank - 1] {
                    return Err(format!(
                        "cross_entropy: targets shape {:?} does not match logits leading shape {:?}",
                        target.shape,
                        &logits.shape[..rank - 1]
                    ));
                }
                if !target.device.same_device(&logits.device) {
                    return Err("cross_entropy: logits and targets must be on the same device".to_string());
                }
                (Vec::new(), logits.dtype, logits.device.clone())
            }
            NodeKind::CrossEntropyBackward { logits, .. } => {
                (logits.shape.clone(), logits.dtype, logits.device.clone())
            }
            NodeKind::Sdpa { q, k, v, .. } => {
                let out = sdpa_check("sdpa", q, k, v)?;
                (out, q.dtype, q.device.clone())
            }
            NodeKind::SdpaBackward { q, k, v, g, fwd, .. } => {
                let out = sdpa_check("sdpa", q, k, v)?;
                if !matches!(&fwd.kind, NodeKind::Sdpa { .. }) {
                    return Err("sdpa backward: fwd must be an sdpa node".to_string());
                }
                if g.shape != out {
                    return Err(format!(
                        "sdpa backward: grad shape {:?} does not match the attention output shape {out:?}",
                        g.shape
                    ));
                }
                if g.dtype != q.dtype || !g.device.same_device(&q.device) {
                    return Err("sdpa backward: grad must share dtype and device with q".to_string());
                }
                (q.shape.clone(), q.dtype, q.device.clone())
            }
            NodeKind::SdpaBackwardOut { of, index } => {
                let NodeKind::SdpaBackward { q, k, v, .. } = &of.kind else {
                    return Err("sdpa backward out: source must be an sdpa backward node".to_string());
                };
                let source = match index {
                    0 => q,
                    1 => k,
                    2 => v,
                    i => return Err(format!("sdpa backward out: index must be 0..=2, got {i}")),
                };
                (source.shape.clone(), source.dtype, source.device.clone())
            }
            NodeKind::PositionEmbedding { weight, seq_len } => {
                if weight.shape.len() != 2 {
                    return Err(format!(
                        "position_embedding: weight must be [maxPositions, E], got {:?}",
                        weight.shape
                    ));
                }
                if *seq_len == 0 || *seq_len > weight.shape[0] {
                    return Err(format!(
                        "position_embedding: seq_len {seq_len} out of range for {} positions",
                        weight.shape[0]
                    ));
                }
                if !weight.dtype.is_float() {
                    return Err(format!(
                        "position_embedding: weight must be a float dtype, got {:?}",
                        weight.dtype
                    ));
                }
                (
                    vec![*seq_len, weight.shape[1]],
                    weight.dtype,
                    weight.device.clone(),
                )
            }
            NodeKind::KvAttention { q, k, v, .. } => {
                let out = sdpa_check("kv_attention", q, k, v)?;
                let rank = q.shape.len();
                if q.shape[rank - 2] != k.shape[rank - 2] {
                    return Err(format!(
                        "kv_attention: q, k and v must share the new-token length, got {:?}, {:?} and {:?}",
                        q.shape, k.shape, v.shape
                    ));
                }
                if rank < 3 {
                    // Leading dims are the batch (one per kv slot);
                    // slot counts are checked at run time (RFC 0013).
                    return Err(format!(
                        "kv_attention: expected shape [B..., H, T, D], got {:?}",
                        q.shape
                    ));
                }
                (out, q.dtype, q.device.clone())
            }
            NodeKind::RotaryEmbedding { x, seq_len, .. } => {
                let rank = x.shape.len();
                if rank < 2 {
                    return Err(format!(
                        "rotary_embedding: expected [.., T, D], got {:?}",
                        x.shape
                    ));
                }
                let (t, d) = (x.shape[rank - 2], x.shape[rank - 1]);
                if *seq_len != t {
                    return Err(format!(
                        "rotary_embedding: seq_len {seq_len} does not match the input's T {t}"
                    ));
                }
                if d % 2 != 0 {
                    return Err(format!("rotary_embedding: head dim must be even, got {d}"));
                }
                if x.dtype != DType::F32 {
                    return Err(format!(
                        "rotary_embedding: dtype must be f32, got {:?}",
                        x.dtype
                    ));
                }
                (x.shape.clone(), x.dtype, x.device.clone())
            }
            NodeKind::RotaryEmbeddingBackward { g, shape, seq_len, .. } => {
                let rank = shape.len();
                if rank < 2 || shape[rank - 2] != *seq_len || g.shape != *shape {
                    return Err(format!(
                        "rotary_embedding_backward: expected grad of shape {shape:?}, got {:?}",
                        g.shape
                    ));
                }
                (shape.clone(), g.dtype, g.device.clone())
            }
            NodeKind::Linear { x, weight, bias } => {
                let rank = x.shape.len();
                if rank < 2
                    || weight.shape.len() != 2
                    || x.shape[rank - 1] != weight.shape[0]
                    || bias.shape != [weight.shape[1]]
                {
                    return Err(format!(
                        "linear: expected x [.., K], weight [K, N], bias [N], got {:?} x {:?} + {:?}",
                        x.shape, weight.shape, bias.shape
                    ));
                }
                let mut out = x.shape.clone();
                out[rank - 1] = weight.shape[1];
                (out, x.dtype, x.device.clone())
            }
            NodeKind::LayerNorm { x, weight, bias, .. } => {
                let rank = x.shape.len();
                let k = weight.shape.len();
                if rank < k
                    || x.shape[rank - k..] != weight.shape[..]
                    || bias.shape != weight.shape
                {
                    return Err(format!(
                        "layer_norm: weight and bias must match the input's trailing dims {:?}, got {:?} and {:?}",
                        x.shape, weight.shape, bias.shape
                    ));
                }
                (x.shape.clone(), x.dtype, x.device.clone())
            }
            NodeKind::LayerNormBackward { x, weight, g, .. } => {
                let rank = x.shape.len();
                let k = weight.shape.len();
                if rank < k || x.shape[rank - k..] != weight.shape[..] || g.shape != x.shape {
                    return Err(format!(
                        "layer_norm_backward: expected grad of shape {:?}, got {:?}",
                        x.shape, g.shape
                    ));
                }
                (x.shape.clone(), x.dtype, x.device.clone())
            }
            NodeKind::LayerNormBackwardOut { of, index } => {
                let NodeKind::LayerNormBackward { weight, .. } = &of.kind else {
                    return Err("layer_norm_backward_out: parent is not a backward node".to_string());
                };
                if *index == 0 || *index > 2 {
                    return Err(format!(
                        "layer_norm_backward_out: index must be 1 (dw) or 2 (db), got {index}"
                    ));
                }
                (weight.shape.clone(), weight.dtype, weight.device.clone())
            }
            NodeKind::Conv1d {
                x,
                w,
                stride,
                padding,
                dilation,
                groups,
            } => {
                if x.shape.len() != 3 || w.shape.len() != 3 {
                    return Err(format!(
                        "conv1d: expected rank-3 input and weight, got ranks {} and {}",
                        x.shape.len(),
                        w.shape.len()
                    ));
                }
                conv_check("conv1d", x, w, *stride, *padding, *dilation, *groups)?;
                let out = conv_out_dim(x.shape[2], w.shape[2], *stride, *padding, *dilation)?;
                (vec![x.shape[0], w.shape[0], out], x.dtype, x.device.clone())
            }
            NodeKind::Conv2d {
                x,
                w,
                stride,
                padding,
                dilation,
                groups,
            } => {
                if x.shape.len() != 4 || w.shape.len() != 4 {
                    return Err(format!(
                        "conv2d: expected rank-4 input and weight, got ranks {} and {}",
                        x.shape.len(),
                        w.shape.len()
                    ));
                }
                conv_check("conv2d", x, w, *stride, *padding, *dilation, *groups)?;
                let oh = conv_out_dim(x.shape[2], w.shape[2], *stride, *padding, *dilation)?;
                let ow = conv_out_dim(x.shape[3], w.shape[3], *stride, *padding, *dilation)?;
                (vec![x.shape[0], w.shape[0], oh, ow], x.dtype, x.device.clone())
            }
            NodeKind::ConvTranspose1d {
                x,
                w,
                stride,
                padding,
                output_padding,
                dilation,
                ..
            } => {
                if x.shape.len() != 3 || w.shape.len() != 3 {
                    return Err("conv_transpose1d: expected rank-3 input and weight".to_string());
                }
                let out = (x.shape[2] - 1) * stride + dilation * (w.shape[2] - 1) + output_padding
                    + 1
                    - 2 * padding;
                (vec![x.shape[0], w.shape[1], out], x.dtype, x.device.clone())
            }
            NodeKind::ConvTranspose2d {
                x,
                w,
                stride,
                padding,
                output_padding,
                dilation,
                ..
            } => {
                if x.shape.len() != 4 || w.shape.len() != 4 {
                    return Err("conv_transpose2d: expected rank-4 input and weight".to_string());
                }
                let oh = (x.shape[2] - 1) * stride + dilation * (w.shape[2] - 1) + output_padding
                    + 1
                    - 2 * padding;
                let ow = (x.shape[3] - 1) * stride + dilation * (w.shape[3] - 1) + output_padding
                    + 1
                    - 2 * padding;
                (vec![x.shape[0], w.shape[1], oh, ow], x.dtype, x.device.clone())
            }
            NodeKind::Conv1dBackwardW {
                x,
                kernel,
                out_channels,
                groups,
                ..
            } => (
                vec![*out_channels, x.shape[1] / groups, *kernel],
                x.dtype,
                x.device.clone(),
            ),
            NodeKind::Conv2dBackwardW {
                x,
                kernel,
                out_channels,
                groups,
                ..
            } => (
                vec![*out_channels, x.shape[1] / groups, kernel[0], kernel[1]],
                x.dtype,
                x.device.clone(),
            ),
            NodeKind::Cast { a, dtype } => (a.shape.clone(), *dtype, a.device.clone()),
            NodeKind::Sum { a, dims, keepdims }
            | NodeKind::Mean { a, dims, keepdims }
            | NodeKind::Max { a, dims, keepdims }
            | NodeKind::Min { a, dims, keepdims }
            | NodeKind::Prod { a, dims, keepdims } => (
                reduced_shape(&a.shape, dims, *keepdims),
                a.dtype,
                a.device.clone(),
            ),
            NodeKind::Reshape { a, shape } => {
                let before: usize = a.shape.iter().product();
                let after: usize = shape.iter().product();
                if before != after {
                    return Err(format!(
                        "reshape: cannot reshape {:?} ({before} elements) to {shape:?} ({after} elements)",
                        a.shape
                    ));
                }
                (shape.clone(), a.dtype, a.device.clone())
            }
            NodeKind::Permute { a, dims } => {
                if dims.len() != a.shape.len()
                    || dims.iter().any(|&d| d >= a.shape.len())
                    || (1..dims.len()).any(|i| dims[..i].contains(&dims[i]))
                {
                    return Err(format!(
                        "permute: dims {dims:?} are not a permutation of rank {}",
                        a.shape.len()
                    ));
                }
                (
                    dims.iter().map(|&d| a.shape[d]).collect(),
                    a.dtype,
                    a.device.clone(),
                )
            }
            NodeKind::Slice { a, ranges } => {
                if ranges.len() != a.shape.len() {
                    return Err(format!(
                        "slice: expected {} ranges, got {}",
                        a.shape.len(),
                        ranges.len()
                    ));
                }
                let shape = ranges
                    .iter()
                    .map(|&(start, stop, stride)| stop.saturating_sub(start).div_ceil(stride))
                    .collect();
                (shape, a.dtype, a.device.clone())
            }
            NodeKind::Concat { a, b, dim } => {
                if a.shape.len() != b.shape.len() || *dim >= a.shape.len() {
                    return Err(format!(
                        "concat: rank/dim mismatch, {:?} vs {:?} along dim {dim}",
                        a.shape, b.shape
                    ));
                }
                let mut shape = a.shape.clone();
                for i in 0..shape.len() {
                    if i == *dim {
                        shape[i] += b.shape[i];
                    } else if a.shape[i] != b.shape[i] {
                        return Err(format!(
                            "concat: shape mismatch at dim {i}, {:?} vs {:?}",
                            a.shape, b.shape
                        ));
                    }
                }
                (shape, a.dtype, a.device.clone())
            }
            NodeKind::BroadcastTo { a, shape } => {
                if shape.len() < a.shape.len() {
                    return Err(format!(
                        "broadcast_to: cannot broadcast {:?} to lower rank {shape:?}",
                        a.shape
                    ));
                }
                let offset = shape.len() - a.shape.len();
                for (i, &d) in a.shape.iter().enumerate() {
                    if d != shape[offset + i] && d != 1 {
                        return Err(format!(
                            "broadcast_to: cannot broadcast {:?} to {shape:?}",
                            a.shape
                        ));
                    }
                }
                (shape.clone(), a.dtype, a.device.clone())
            }
            NodeKind::Matmul { a, b } => {
                if a.shape.len() < 2 || b.shape.len() < 2 {
                    return Err(format!(
                        "matmul: expected tensors of rank >= 2, got {:?} and {:?}",
                        a.shape, b.shape
                    ));
                }
                let ar = a.shape.len();
                let br = b.shape.len();
                if a.shape[ar - 1] != b.shape[br - 2] {
                    return Err(format!(
                        "matmul: inner dimensions mismatch, got {:?} and {:?}",
                        a.shape, b.shape
                    ));
                }
                let mut shape = broadcast_shapes(&a.shape[..ar - 2], &b.shape[..br - 2])?;
                shape.push(a.shape[ar - 2]);
                shape.push(b.shape[br - 1]);
                (shape, a.dtype, a.device.clone())
            }
            NodeKind::Inverse { a } | NodeKind::Det { a } => {
                let rank = a.shape.len();
                if rank < 2 || a.shape[rank - 2] != a.shape[rank - 1] {
                    return Err(format!(
                        "linalg: expected a tensor square on its last two dimensions, got shape {:?}",
                        a.shape
                    ));
                }
                if !a.dtype.is_float() {
                    return Err(format!("linalg: dtype must be floating point, got {:?}", a.dtype));
                }
                if matches!(&kind, NodeKind::Det { .. }) {
                    (a.shape[..rank - 2].to_vec(), a.dtype, a.device.clone())
                } else {
                    (a.shape.clone(), a.dtype, a.device.clone())
                }
            }
            NodeKind::Solve { a, b } => {
                let rank = a.shape.len();
                if rank < 2 || a.shape[rank - 2] != a.shape[rank - 1] {
                    return Err(format!(
                        "solve: expected a coefficient tensor square on its last two dimensions, got shape {:?}",
                        a.shape
                    ));
                }
                if b.shape.len() != rank
                    || b.shape[..rank - 2] != a.shape[..rank - 2]
                    || b.shape[rank - 2] != a.shape[rank - 1]
                {
                    return Err(format!(
                        "solve: expected a right-hand side of shape {:?} with {} rows, got shape {:?}",
                        &a.shape[..rank - 1],
                        a.shape[rank - 1],
                        b.shape
                    ));
                }
                if !a.dtype.is_float() || a.dtype != b.dtype {
                    return Err(format!(
                        "solve: dtypes must be floating point and match, got {:?} and {:?}",
                        a.dtype, b.dtype
                    ));
                }
                (b.shape.clone(), a.dtype, a.device.clone())
            }
            NodeKind::AdamWStep {
                param,
                grad,
                m,
                v,
                lr,
                c1,
                c2,
                ..
            } => {
                if !param.dtype.is_float() {
                    return Err(format!(
                        "adamw_step: dtype must be floating point, got {:?}",
                        param.dtype
                    ));
                }
                for (name, t) in [("grad", grad), ("m", m), ("v", v)] {
                    if t.shape != param.shape || t.dtype != param.dtype {
                        return Err(format!(
                            "adamw_step: {name} must match the parameter shape and dtype"
                        ));
                    }
                }
                for (name, t) in [("lr", lr), ("c1", c1), ("c2", c2)] {
                    if !t.shape.is_empty() {
                        return Err(format!("adamw_step: {name} must be a scalar (0-d) tensor"));
                    }
                }
                (param.shape.clone(), param.dtype, param.device.clone())
            }
            NodeKind::AdamWOut { step, index } => {
                if *index > 2 {
                    return Err(format!("adamw_out: index must be 0, 1 or 2, got {index}"));
                }
                (step.shape.clone(), step.dtype, step.device.clone())
            }
            NodeKind::AdamWStepGroup {
                params,
                grads,
                ms,
                vs,
                lr,
                c1,
                c2,
                ..
            } => {
                let n = params.len();
                if n == 0 || n > 4 {
                    return Err(format!(
                        "adamw_step_group: groups hold 1..=4 params (the 31-buffer limit: 4 lanes + 3 outputs each plus 3 scalars), got {n}"
                    ));
                }
                if grads.len() != n || ms.len() != n || vs.len() != n {
                    return Err("adamw_step_group: params, grads, ms and vs must be equally long".to_string());
                }
                let first = &params[0];
                if !first.dtype.is_float() {
                    return Err(format!(
                        "adamw_step_group: dtype must be floating point, got {:?}",
                        first.dtype
                    ));
                }
                for (name, tensors) in [("param", params), ("grad", grads), ("m", ms), ("v", vs)] {
                    for t in tensors {
                        if t.shape != first.shape || t.dtype != first.dtype {
                            return Err(format!(
                                "adamw_step_group: {name} must share the group's shape and dtype"
                            ));
                        }
                    }
                }
                for (name, t) in [("lr", lr), ("c1", c1), ("c2", c2)] {
                    if !t.shape.is_empty() {
                        return Err(format!("adamw_step_group: {name} must be a scalar (0-d) tensor"));
                    }
                }
                (first.shape.clone(), first.dtype, first.device.clone())
            }
            NodeKind::AdamWGroupOut { of, param, index } => {
                let NodeKind::AdamWStepGroup { params, .. } = &of.kind else {
                    return Err("adamw_group_out: parent is not a step group".to_string());
                };
                if *index > 2 || *param as usize >= params.len() {
                    return Err(format!(
                        "adamw_group_out: param {param} or index {index} out of range"
                    ));
                }
                (of.shape.clone(), of.dtype, of.device.clone())
            }
            NodeKind::SgdStep {
                param,
                grad,
                velocity,
                first,
                lr,
                ..
            } => {
                if !param.dtype.is_float() {
                    return Err(format!(
                        "sgd_step: dtype must be floating point, got {:?}",
                        param.dtype
                    ));
                }
                for (name, t) in [("grad", grad), ("velocity", velocity)] {
                    if t.shape != param.shape || t.dtype != param.dtype {
                        return Err(format!(
                            "sgd_step: {name} must match the parameter shape and dtype"
                        ));
                    }
                }
                for (name, t) in [("first", first), ("lr", lr)] {
                    if !t.shape.is_empty() {
                        return Err(format!("sgd_step: {name} must be a scalar (0-d) tensor"));
                    }
                }
                (param.shape.clone(), param.dtype, param.device.clone())
            }
            NodeKind::SgdOut { step, index } => {
                if *index > 1 {
                    return Err(format!("sgd_out: index must be 0 or 1, got {index}"));
                }
                (step.shape.clone(), step.dtype, step.device.clone())
            }
            NodeKind::FusedElementwise {
                inputs,
                strides,
                shape,
                ..
            } => {
                if inputs.is_empty() {
                    return Err("fused: at least one input lane is required".to_string());
                }
                if strides.len() != inputs.len() {
                    return Err(format!(
                        "fused: got {} stride entries for {} inputs",
                        strides.len(),
                        inputs.len()
                    ));
                }
                let first = &inputs[0];
                for (input, stride) in inputs.iter().zip(strides.iter()) {
                    if input.dtype != first.dtype || !input.device.same_device(&first.device) {
                        return Err(
                            "fused: all inputs must share dtype and device".to_string()
                        );
                    }
                    if fusion::lane_strides(&input.shape, shape).as_ref() != Some(stride) {
                        return Err(format!(
                            "fused: input shape {:?} does not broadcast to {:?} with strides {stride:?}",
                            input.shape, shape
                        ));
                    }
                }
                if !first.dtype.is_float() {
                    return Err(format!(
                        "fused: dtype must be floating point, got {:?}",
                        first.dtype
                    ));
                }
                (shape.clone(), first.dtype, first.device.clone())
            }
            NodeKind::FusedElementwiseMulti {
                inputs,
                strides,
                shape,
                exprs,
            } => {
                if exprs.is_empty() {
                    return Err("fused multi: at least one output is required".to_string());
                }
                if inputs.is_empty() {
                    return Err("fused multi: at least one input lane is required".to_string());
                }
                if strides.len() != inputs.len() {
                    return Err(format!(
                        "fused multi: got {} stride entries for {} inputs",
                        strides.len(),
                        inputs.len()
                    ));
                }
                let first = &inputs[0];
                for (input, stride) in inputs.iter().zip(strides.iter()) {
                    if input.dtype != first.dtype || !input.device.same_device(&first.device) {
                        return Err("fused multi: all inputs must share dtype and device".to_string());
                    }
                    if fusion::lane_strides(&input.shape, shape).as_ref() != Some(stride) {
                        return Err(format!(
                            "fused multi: input shape {:?} does not broadcast to {:?} with strides {stride:?}",
                            input.shape, shape
                        ));
                    }
                }
                if !first.dtype.is_float() {
                    return Err(format!(
                        "fused multi: dtype must be floating point, got {:?}",
                        first.dtype
                    ));
                }
                (shape.clone(), first.dtype, first.device.clone())
            }
            NodeKind::FusedPick { of, .. } => (of.shape.clone(), of.dtype, of.device.clone()),
            NodeKind::FusedReduce {
                inputs,
                strides,
                in_shape,
                dims,
                keepdims,
                shape,
                ..
            } => {
                if inputs.is_empty() {
                    return Err("fused reduce: at least one input lane is required".to_string());
                }
                if strides.len() != inputs.len() {
                    return Err(format!(
                        "fused reduce: got {} stride entries for {} inputs",
                        strides.len(),
                        inputs.len()
                    ));
                }
                if dims.is_empty()
                    || dims.iter().any(|&d| d >= in_shape.len())
                    || !dims.windows(2).all(|w| w[0] < w[1])
                {
                    return Err(format!(
                        "fused reduce: dims {dims:?} are not sorted unique dims of {in_shape:?}"
                    ));
                }
                if &reduced_shape(in_shape, dims, *keepdims) != shape {
                    return Err(format!(
                        "fused reduce: shape {shape:?} is not {in_shape:?} reduced over {dims:?} (keepdims {keepdims})"
                    ));
                }
                let first = &inputs[0];
                for (input, stride) in inputs.iter().zip(strides.iter()) {
                    if input.dtype != first.dtype || !input.device.same_device(&first.device) {
                        return Err(
                            "fused reduce: all inputs must share dtype and device".to_string()
                        );
                    }
                    if fusion::lane_strides(&input.shape, in_shape).as_ref() != Some(stride) {
                        return Err(format!(
                            "fused reduce: input shape {:?} does not broadcast to {:?} with strides {stride:?}",
                            input.shape, in_shape
                        ));
                    }
                }
                if !first.dtype.is_float() {
                    return Err(format!(
                        "fused reduce: dtype must be floating point, got {:?}",
                        first.dtype
                    ));
                }
                (shape.clone(), first.dtype, first.device.clone())
            }
        };
        check_dtype_device(dtype, &device)?;
        Ok(Arc::new(Node {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            shape,
            dtype,
            device,
            kind,
        }))
    }
}

// RFC 0012: device dtype capabilities, enforced at graph construction
// (Node::new is the single choke point — every lazy node, including
// from-bytes leaves and nodes rebuilt by compile/fuse rewrites, passes
// through here). Metal's shading language has no f64. Never a silent
// downcast, never deferred to compute time.
fn check_dtype_device(dtype: DType, device: &Device) -> std::result::Result<(), String> {
    if matches!(dtype, DType::F64) && matches!(device, Device::Metal) {
        return Err(
            "dtype f64 is not supported on device metal (supported: f32, f16, bf16, i64, u32, u8); cast explicitly or use device cpu"
                .to_string(),
        );
    }
    Ok(())
}

#[napi]
pub struct LazyTensor {
    node: Arc<Node>,
}

impl LazyTensor {
    pub(crate) fn from_node(node: Arc<Node>) -> Self {
        Self { node }
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

    #[napi(factory)]
    pub fn zeros(
        shape: Vec<u32>,
        dtype: Option<NativeDType>,
        device: Option<String>,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Zeros {
            shape: shape.iter().map(|&d| d as usize).collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(device)?,
        }))
    }

    #[napi(factory)]
    pub fn ones(
        shape: Vec<u32>,
        dtype: Option<NativeDType>,
        device: Option<String>,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Ones {
            shape: shape.iter().map(|&d| d as usize).collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(device)?,
        }))
    }

    #[napi(factory)]
    pub fn full(
        shape: Vec<u32>,
        value: f64,
        dtype: Option<NativeDType>,
        device: Option<String>,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Full {
            shape: shape.iter().map(|&d| d as usize).collect(),
            value,
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(device)?,
        }))
    }

    #[napi(factory)]
    pub fn randn(
        shape: Vec<u32>,
        dtype: Option<NativeDType>,
        device: Option<String>,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Randn {
            shape: shape.iter().map(|&d| d as usize).collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(device)?,
        }))
    }

    #[napi(factory)]
    pub fn uniform(
        shape: Vec<u32>,
        lo: f64,
        hi: f64,
        dtype: Option<NativeDType>,
        device: Option<String>,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Uniform {
            lo,
            hi,
            shape: shape.iter().map(|&d| d as usize).collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(device)?,
        }))
    }

    #[napi(factory)]
    pub fn arange(
        start: f64,
        end: f64,
        step: f64,
        dtype: Option<NativeDType>,
        device: Option<String>,
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
            device: get_device(device)?,
        }))
    }

    #[napi(factory)]
    pub fn eye(n: u32, dtype: Option<NativeDType>, device: Option<String>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Eye {
            n: n as usize,
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(device)?,
        }))
    }

    // A shared 0-d constant: the same (value, dtype, device) triple maps to
    // one graph node forever instead of allocating a fresh node per use.
    // Nodes hold no buffers, so the cache is cheap; it is size-bounded so
    // cold values rotate through. Devices are process singletons, so the
    // device kind is the whole key.
    #[napi(factory)]
    pub fn constant(value: f64, dtype: Option<NativeDType>, device: Option<String>) -> Result<Self> {
        let device = get_device(device)?;
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
        device: Option<String>,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::FromBytes {
            data: data.to_vec(),
            shape: shape.iter().map(|&d| d as usize).collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(device)?,
        }))
    }

    #[napi(factory)]
    pub fn from_materialized(tensor: &NativeTensor) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Leaf(tensor.inner.clone())))
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
        device: Option<String>,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::Input {
            slot,
            shape: shape.iter().map(|&d| d as usize).collect(),
            dtype: dtype.unwrap_or(NativeDType::F32).into(),
            device: get_device(device)?,
        }))
    }

    #[napi(factory)]
    pub fn scalar_input(slot: u32, dtype: Option<NativeDType>, device: Option<String>) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::ScalarInput {
            slot,
            dtype: dtype.unwrap_or(NativeDType::F64).into(),
            device: get_device(device)?,
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
        lazy_ctor!(Node::new(NodeKind::CrossEntropy {
            logits: self.node.clone(),
            target: target.node.clone(),
            ignore_index,
        }))
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
        lazy_ctor!(autodiff::vmap(&self.node, &x.node, &batched_x.node, dim as usize))
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
    cache: std::collections::HashMap<u64, val::Val>,
    // AdamW step id -> (next m, next v); the step node's own value is the
    // updated parameter, stored in the regular cache
    adamw: std::collections::HashMap<u64, [val::Val; 2]>,
    sgd: std::collections::HashMap<u64, val::Val>,
    // FusedElementwiseMulti id -> all outputs; the node's own cache entry
    // holds output 0 so the evaluator's single-value invariant holds
    multi: std::collections::HashMap<u64, Vec<val::Val>>,
    // LayerNormBackward id -> (dw, db); the node's own cache entry is dx.
    ln: std::collections::HashMap<u64, [val::Val; 2]>,
    // Optimizer step scalars (lr, bias corrections) cast to a given
    // dtype/device, memoized per walk: identical for every parameter,
    // and the naive path copies each scalar per parameter per step.
    step_scalars: std::collections::HashMap<(u64, DType, u8), val::Val>,
    // Device-packed fusion scalar buffers (cat of the 0-d step scalars),
    // memoized per walk: every AdamW group in a step packs the same triple.
    scalar_packs: std::collections::HashMap<(u64, u64, u64), val::Val>,
    consumers: std::collections::HashMap<u64, usize>,
    roots: HashSet<u64>,
    // RFC 0008: Input/ScalarInput node id -> argument buffer, populated by
    // CompiledProgram::run. Empty for ordinary eval_lazy walks.
    slots: std::collections::HashMap<u64, val::Val>,
    // RFC 0010: the pool + sequence a kv program runs against. None for
    // ordinary and non-kv compiled walks; KvAttention nodes error without
    // it.
    kv: Option<Arc<KvContext>>,
    // Deferred cross-entropy status checks (fused CE): (buffer,
    // forward?, classes). Reading the status requires a device sync,
    // which would split the walk's encode/execute pipeline mid-graph —
    // so the fused kernels record their status here and the walk
    // validates them after its final synchronize, preserving the exact
    // error semantics.
    ce_checks: Vec<(val::Val, bool, usize)>,
}

impl Evaluator {
    fn new(roots: &[Arc<Node>]) -> Self {
        Self::with_slots(roots, std::collections::HashMap::new())
    }

    fn with_slots(roots: &[Arc<Node>], slots: std::collections::HashMap<u64, val::Val>) -> Self {
        Self::with_kv(roots, slots, None)
    }

    fn with_kv(
        roots: &[Arc<Node>],
        slots: std::collections::HashMap<u64, val::Val>,
        kv: Option<Arc<KvContext>>,
    ) -> Self {
        let mut consumers = std::collections::HashMap::new();
        for root in roots {
            count_consumers(root, &mut consumers);
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

    // Runs the deferred fused-CE status checks after the walk's final
    // synchronize: forward statuses are [loss, active, invalid],
    // backward counts are [active]. Errors are exactly the composed
    // path's, raised from the same eval call.
    fn run_ce_checks(&self) -> crate::err::Res<()> {
        for (buffer, forward, classes) in &self.ce_checks {
            let values = buffer.to_f32_vec()?;
            if *forward {
                let (active, invalid) = (values[1] as usize, values[2] as usize);
                if active == 0 {
                    return Err(
                        "cross_entropy: no active targets (all positions are ignored)".to_string(),
                    );
                }
                if invalid > 0 {
                    return Err(format!(
                        "cross_entropy: target out of range [0, {classes}) at an active position"
                    ));
                }
            } else if values[0] == 0.0 {
                return Err(
                    "cross_entropy: no active targets (all positions are ignored)".to_string(),
                );
            }
        }
        Ok(())
    }

    fn value(&self, id: u64) -> crate::err::Res<val::Val> {
        self.cache.get(&id).cloned().ok_or_else(|| format!("internal error: unevaluated node {id}"))
    }

    // A step scalar cast to the target dtype/device, memoized per walk
    // (one copy per distinct dtype instead of one per parameter).
    fn step_scalar(&mut self, id: u64, dtype: DType, device: &Device) -> crate::err::Res<val::Val> {
        let device_key = match device {
            Device::Cpu => 0u8,
            Device::Metal => 2,
        };
        let key = (id, dtype, device_key);
        if let Some(cached) = self.step_scalars.get(&key) {
            return Ok(cached.clone());
        }
        let cast = match self.value(id)? {
            val::Val::Cpu(t) => val::Val::Cpu(t.cast(dtype)),
            val::Val::Metal(t) => val::Val::Metal(metal_ops::cast(&t, dtype)?),
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

fn count_consumers(root: &Arc<Node>, consumers: &mut std::collections::HashMap<u64, usize>) {
    let mut visited = HashSet::new();
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

fn node_children(kind: &NodeKind) -> Vec<Arc<Node>> {
    match kind {
        NodeKind::Leaf(_)
        | NodeKind::Input { .. }
        | NodeKind::ScalarInput { .. }
        | NodeKind::FromBytes { .. }
        | NodeKind::Zeros { .. }
        | NodeKind::Ones { .. }
        | NodeKind::Full { .. }
        | NodeKind::Randn { .. }
        | NodeKind::Uniform { .. }
        | NodeKind::Arange { .. }
        | NodeKind::Eye { .. } => vec![],
        NodeKind::Add { a, b }
        | NodeKind::Sub { a, b }
        | NodeKind::Mul { a, b }
        | NodeKind::Div { a, b }
        | NodeKind::Eq { a, b }
        | NodeKind::Gt { a, b }
        | NodeKind::Lt { a, b }
        | NodeKind::Ge { a, b }
        | NodeKind::Le { a, b }
        | NodeKind::Maximum { a, b }
        | NodeKind::Minimum { a, b }
        | NodeKind::Concat { a, b, .. }
        | NodeKind::Matmul { a, b } => vec![a.clone(), b.clone()],
        NodeKind::Solve { a, b } => vec![a.clone(), b.clone()],
        NodeKind::Neg { a }
        | NodeKind::Abs { a }
        | NodeKind::Sqrt { a }
        | NodeKind::Exp { a }
        | NodeKind::Log { a }
        | NodeKind::Sin { a }
        | NodeKind::Cos { a }
        | NodeKind::Tanh { a }
        | NodeKind::Relu { a }
        | NodeKind::Erf { a }
        | NodeKind::Floor { a }
        | NodeKind::Ceil { a }
        | NodeKind::Round { a }
        | NodeKind::Sign { a }
        | NodeKind::Argmax { a, .. }
        | NodeKind::Argmin { a, .. }
        | NodeKind::Inverse { a }
        | NodeKind::Det { a }
        | NodeKind::Cumsum { a, .. }
        | NodeKind::Pow { a, .. }
        | NodeKind::Cast { a, .. }
        | NodeKind::Sum { a, .. }
        | NodeKind::Mean { a, .. }
        | NodeKind::Max { a, .. }
        | NodeKind::Min { a, .. }
        | NodeKind::Prod { a, .. }
        | NodeKind::Reshape { a, .. }
        | NodeKind::Permute { a, .. }
        | NodeKind::Slice { a, .. }
        | NodeKind::BroadcastTo { a, .. }
        | NodeKind::Checkpoint { a }
        | NodeKind::StopGradient { a } => vec![a.clone()],
        NodeKind::Where { cond, a, b } => vec![cond.clone(), a.clone(), b.clone()],
        NodeKind::IndexSelect { a, indexes, .. } => vec![a.clone(), indexes.clone()],
        NodeKind::Gather { a, indexes, .. } => vec![a.clone(), indexes.clone()],
        NodeKind::CrossEntropy {
            logits, target, ..
        }
        | NodeKind::CrossEntropyBackward {
            logits, target, ..
        } => vec![logits.clone(), target.clone()],
        NodeKind::Sdpa { q, k, v, .. } => vec![q.clone(), k.clone(), v.clone()],
        NodeKind::KvAttention { q, k, v, .. } => vec![q.clone(), k.clone(), v.clone()],
        NodeKind::PositionEmbedding { weight, .. } => vec![weight.clone()],
        NodeKind::RotaryEmbedding { x, .. } => vec![x.clone()],
        NodeKind::RotaryEmbeddingBackward { g, .. } => vec![g.clone()],
        NodeKind::LayerNorm { x, weight, bias, .. } => vec![x.clone(), weight.clone(), bias.clone()],
        NodeKind::LayerNormBackward { x, weight, g, .. } => {
            vec![x.clone(), weight.clone(), g.clone()]
        }
        NodeKind::LayerNormBackwardOut { of, .. } => vec![of.clone()],
        NodeKind::Linear { x, weight, bias } => vec![x.clone(), weight.clone(), bias.clone()],
        NodeKind::SdpaBackward { q, k, v, g, fwd, .. } => {
            vec![q.clone(), k.clone(), v.clone(), g.clone(), fwd.clone()]
        }
        NodeKind::SdpaBackwardOut { of, .. } => vec![of.clone()],
        NodeKind::Conv1d { x, w, .. }
        | NodeKind::Conv2d { x, w, .. }
        | NodeKind::ConvTranspose1d { x, w, .. }
        | NodeKind::ConvTranspose2d { x, w, .. } => vec![x.clone(), w.clone()],
        NodeKind::Conv1dBackwardW { x, g, .. } | NodeKind::Conv2dBackwardW { x, g, .. } => {
            vec![x.clone(), g.clone()]
        }
        NodeKind::ScatterAdd {
            a,
            indexes,
            src,
            ..
        } => vec![a.clone(), indexes.clone(), src.clone()],
        NodeKind::AdamWStep {
            param,
            grad,
            m,
            v,
            lr,
            c1,
            c2,
            ..
        } => vec![
            param.clone(),
            grad.clone(),
            m.clone(),
            v.clone(),
            lr.clone(),
            c1.clone(),
            c2.clone(),
        ],
        NodeKind::AdamWOut { step, .. } => vec![step.clone()],
        NodeKind::AdamWStepGroup {
            params,
            grads,
            ms,
            vs,
            lr,
            c1,
            c2,
            ..
        } => params
            .iter()
            .chain(grads.iter())
            .chain(ms.iter())
            .chain(vs.iter())
            .cloned()
            .chain([lr.clone(), c1.clone(), c2.clone()])
            .collect(),
        NodeKind::AdamWGroupOut { of, .. } => vec![of.clone()],
        NodeKind::SgdStep {
            param,
            grad,
            velocity,
            first,
            lr,
            ..
        } => vec![
            param.clone(),
            grad.clone(),
            velocity.clone(),
            first.clone(),
            lr.clone(),
        ],
        NodeKind::SgdOut { step, .. } => vec![step.clone()],
        NodeKind::FusedElementwise { inputs, .. } => inputs.clone(),
        NodeKind::FusedElementwiseMulti { inputs, .. } => inputs.clone(),
        NodeKind::FusedReduce { inputs, .. } => inputs.clone(),
        NodeKind::FusedPick { of, .. } => vec![of.clone()],
    }
}

// Rebuilds a node kind with its children mapped through `f`. Used to
// deep-copy subgraphs with fresh node ids (checkpoint recompute).
fn remap_children(kind: &NodeKind, f: &dyn Fn(&Arc<Node>) -> Arc<Node>) -> NodeKind {
    match kind {
        NodeKind::Leaf(t) => NodeKind::Leaf(t.clone()),
        NodeKind::Input {
            slot,
            shape,
            dtype,
            device,
        } => NodeKind::Input {
            slot: *slot,
            shape: shape.clone(),
            dtype: *dtype,
            device: device.clone(),
        },
        NodeKind::ScalarInput {
            slot,
            dtype,
            device,
        } => NodeKind::ScalarInput {
            slot: *slot,
            dtype: *dtype,
            device: device.clone(),
        },
        NodeKind::FromBytes {
            data,
            shape,
            dtype,
            device,
        } => NodeKind::FromBytes {
            data: data.clone(),
            shape: shape.clone(),
            dtype: *dtype,
            device: device.clone(),
        },
        NodeKind::Zeros {
            shape,
            dtype,
            device,
        } => NodeKind::Zeros {
            shape: shape.clone(),
            dtype: *dtype,
            device: device.clone(),
        },
        NodeKind::Ones {
            shape,
            dtype,
            device,
        } => NodeKind::Ones {
            shape: shape.clone(),
            dtype: *dtype,
            device: device.clone(),
        },
        NodeKind::Full {
            shape,
            value,
            dtype,
            device,
        } => NodeKind::Full {
            shape: shape.clone(),
            value: *value,
            dtype: *dtype,
            device: device.clone(),
        },
        NodeKind::Randn {
            shape,
            dtype,
            device,
        } => NodeKind::Randn {
            shape: shape.clone(),
            dtype: *dtype,
            device: device.clone(),
        },
        NodeKind::Uniform {
            lo,
            hi,
            shape,
            dtype,
            device,
        } => NodeKind::Uniform {
            lo: *lo,
            hi: *hi,
            shape: shape.clone(),
            dtype: *dtype,
            device: device.clone(),
        },
        NodeKind::Arange {
            start,
            end,
            step,
            dtype,
            device,
        } => NodeKind::Arange {
            start: *start,
            end: *end,
            step: *step,
            dtype: *dtype,
            device: device.clone(),
        },
        NodeKind::Eye { n, dtype, device } => NodeKind::Eye {
            n: *n,
            dtype: *dtype,
            device: device.clone(),
        },
        NodeKind::Add { a, b } => NodeKind::Add { a: f(a), b: f(b) },
        NodeKind::Sub { a, b } => NodeKind::Sub { a: f(a), b: f(b) },
        NodeKind::Mul { a, b } => NodeKind::Mul { a: f(a), b: f(b) },
        NodeKind::Div { a, b } => NodeKind::Div { a: f(a), b: f(b) },
        NodeKind::Eq { a, b } => NodeKind::Eq { a: f(a), b: f(b) },
        NodeKind::Gt { a, b } => NodeKind::Gt { a: f(a), b: f(b) },
        NodeKind::Lt { a, b } => NodeKind::Lt { a: f(a), b: f(b) },
        NodeKind::Ge { a, b } => NodeKind::Ge { a: f(a), b: f(b) },
        NodeKind::Le { a, b } => NodeKind::Le { a: f(a), b: f(b) },
        NodeKind::Maximum { a, b } => NodeKind::Maximum { a: f(a), b: f(b) },
        NodeKind::Minimum { a, b } => NodeKind::Minimum { a: f(a), b: f(b) },
        NodeKind::Concat { a, b, dim } => NodeKind::Concat {
            a: f(a),
            b: f(b),
            dim: *dim,
        },
        NodeKind::Matmul { a, b } => NodeKind::Matmul { a: f(a), b: f(b) },
        NodeKind::Solve { a, b } => NodeKind::Solve { a: f(a), b: f(b) },
        NodeKind::Where { cond, a, b } => NodeKind::Where {
            cond: f(cond),
            a: f(a),
            b: f(b),
        },
        NodeKind::IndexSelect { a, dim, indexes } => NodeKind::IndexSelect {
            a: f(a),
            dim: *dim,
            indexes: f(indexes),
        },
        NodeKind::Gather { a, dim, indexes } => NodeKind::Gather {
            a: f(a),
            dim: *dim,
            indexes: f(indexes),
        },
        NodeKind::CrossEntropy {
            logits,
            target,
            ignore_index,
        } => NodeKind::CrossEntropy {
            logits: f(logits),
            target: f(target),
            ignore_index: *ignore_index,
        },
        NodeKind::CrossEntropyBackward {
            logits,
            target,
            ignore_index,
        } => NodeKind::CrossEntropyBackward {
            logits: f(logits),
            target: f(target),
            ignore_index: *ignore_index,
        },
        NodeKind::Sdpa {
            q,
            k,
            v,
            scale,
            causal,
        } => NodeKind::Sdpa {
            q: f(q),
            k: f(k),
            v: f(v),
            scale: *scale,
            causal: *causal,
        },
        NodeKind::SdpaBackward {
            q,
            k,
            v,
            g,
            fwd,
            scale,
            causal,
        } => NodeKind::SdpaBackward {
            q: f(q),
            k: f(k),
            v: f(v),
            g: f(g),
            fwd: f(fwd),
            scale: *scale,
            causal: *causal,
        },
        NodeKind::SdpaBackwardOut { of, index } => NodeKind::SdpaBackwardOut {
            of: f(of),
            index: *index,
        },
        NodeKind::PositionEmbedding { weight, seq_len } => NodeKind::PositionEmbedding {
            weight: f(weight),
            seq_len: *seq_len,
        },
        NodeKind::KvAttention {
            q,
            k,
            v,
            scale,
            layer,
            window,
        } => NodeKind::KvAttention {
            q: f(q),
            k: f(k),
            v: f(v),
            scale: *scale,
            layer: *layer,
            window: *window,
        },
        NodeKind::RotaryEmbedding {
            x,
            seq_len,
            theta,
            offset,
        } => NodeKind::RotaryEmbedding {
            x: f(x),
            seq_len: *seq_len,
            theta: *theta,
            offset: *offset,
        },
        NodeKind::RotaryEmbeddingBackward {
            g,
            shape,
            seq_len,
            theta,
        } => NodeKind::RotaryEmbeddingBackward {
            g: f(g),
            shape: shape.clone(),
            seq_len: *seq_len,
            theta: *theta,
        },
        NodeKind::LayerNorm { x, weight, bias, eps } => NodeKind::LayerNorm {
            x: f(x),
            weight: f(weight),
            bias: f(bias),
            eps: *eps,
        },
        NodeKind::LayerNormBackward { x, weight, g, eps } => NodeKind::LayerNormBackward {
            x: f(x),
            weight: f(weight),
            g: f(g),
            eps: *eps,
        },
        NodeKind::LayerNormBackwardOut { of, index } => NodeKind::LayerNormBackwardOut {
            of: f(of),
            index: *index,
        },
        NodeKind::Linear { x, weight, bias } => NodeKind::Linear {
            x: f(x),
            weight: f(weight),
            bias: f(bias),
        },
        NodeKind::Conv1d {
            x,
            w,
            stride,
            padding,
            dilation,
            groups,
        } => NodeKind::Conv1d {
            x: f(x),
            w: f(w),
            stride: *stride,
            padding: *padding,
            dilation: *dilation,
            groups: *groups,
        },
        NodeKind::Conv2d {
            x,
            w,
            stride,
            padding,
            dilation,
            groups,
        } => NodeKind::Conv2d {
            x: f(x),
            w: f(w),
            stride: *stride,
            padding: *padding,
            dilation: *dilation,
            groups: *groups,
        },
        NodeKind::ConvTranspose1d {
            x,
            w,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        } => NodeKind::ConvTranspose1d {
            x: f(x),
            w: f(w),
            stride: *stride,
            padding: *padding,
            output_padding: *output_padding,
            dilation: *dilation,
            groups: *groups,
        },
        NodeKind::ConvTranspose2d {
            x,
            w,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        } => NodeKind::ConvTranspose2d {
            x: f(x),
            w: f(w),
            stride: *stride,
            padding: *padding,
            output_padding: *output_padding,
            dilation: *dilation,
            groups: *groups,
        },
        NodeKind::Conv1dBackwardW {
            x,
            g,
            kernel,
            out_channels,
            stride,
            padding,
            dilation,
            groups,
        } => NodeKind::Conv1dBackwardW {
            x: f(x),
            g: f(g),
            kernel: *kernel,
            out_channels: *out_channels,
            stride: *stride,
            padding: *padding,
            dilation: *dilation,
            groups: *groups,
        },
        NodeKind::Conv2dBackwardW {
            x,
            g,
            kernel,
            out_channels,
            stride,
            padding,
            dilation,
            groups,
        } => NodeKind::Conv2dBackwardW {
            x: f(x),
            g: f(g),
            kernel: *kernel,
            out_channels: *out_channels,
            stride: *stride,
            padding: *padding,
            dilation: *dilation,
            groups: *groups,
        },
        NodeKind::ScatterAdd {
            a,
            dim,
            indexes,
            src,
        } => NodeKind::ScatterAdd {
            a: f(a),
            dim: *dim,
            indexes: f(indexes),
            src: f(src),
        },
        NodeKind::Neg { a } => NodeKind::Neg { a: f(a) },
        NodeKind::Abs { a } => NodeKind::Abs { a: f(a) },
        NodeKind::Sqrt { a } => NodeKind::Sqrt { a: f(a) },
        NodeKind::Exp { a } => NodeKind::Exp { a: f(a) },
        NodeKind::Log { a } => NodeKind::Log { a: f(a) },
        NodeKind::Sin { a } => NodeKind::Sin { a: f(a) },
        NodeKind::Cos { a } => NodeKind::Cos { a: f(a) },
        NodeKind::Tanh { a } => NodeKind::Tanh { a: f(a) },
        NodeKind::Relu { a } => NodeKind::Relu { a: f(a) },
        NodeKind::Erf { a } => NodeKind::Erf { a: f(a) },
        NodeKind::Floor { a } => NodeKind::Floor { a: f(a) },
        NodeKind::Ceil { a } => NodeKind::Ceil { a: f(a) },
        NodeKind::Round { a } => NodeKind::Round { a: f(a) },
        NodeKind::Sign { a } => NodeKind::Sign { a: f(a) },
        NodeKind::Argmax { a, dim } => NodeKind::Argmax { a: f(a), dim: *dim },
        NodeKind::Argmin { a, dim } => NodeKind::Argmin { a: f(a), dim: *dim },
        NodeKind::Inverse { a } => NodeKind::Inverse { a: f(a) },
        NodeKind::Det { a } => NodeKind::Det { a: f(a) },
        NodeKind::Cumsum { a, dim } => NodeKind::Cumsum { a: f(a), dim: *dim },
        NodeKind::Pow { a, exp } => NodeKind::Pow { a: f(a), exp: *exp },
        NodeKind::Cast { a, dtype } => NodeKind::Cast {
            a: f(a),
            dtype: *dtype,
        },
        NodeKind::Sum { a, dims, keepdims } => NodeKind::Sum {
            a: f(a),
            dims: dims.clone(),
            keepdims: *keepdims,
        },
        NodeKind::Mean { a, dims, keepdims } => NodeKind::Mean {
            a: f(a),
            dims: dims.clone(),
            keepdims: *keepdims,
        },
        NodeKind::Max { a, dims, keepdims } => NodeKind::Max {
            a: f(a),
            dims: dims.clone(),
            keepdims: *keepdims,
        },
        NodeKind::Min { a, dims, keepdims } => NodeKind::Min {
            a: f(a),
            dims: dims.clone(),
            keepdims: *keepdims,
        },
        NodeKind::Prod { a, dims, keepdims } => NodeKind::Prod {
            a: f(a),
            dims: dims.clone(),
            keepdims: *keepdims,
        },
        NodeKind::Reshape { a, shape } => NodeKind::Reshape {
            a: f(a),
            shape: shape.clone(),
        },
        NodeKind::Permute { a, dims } => NodeKind::Permute {
            a: f(a),
            dims: dims.clone(),
        },
        NodeKind::Slice { a, ranges } => NodeKind::Slice {
            a: f(a),
            ranges: ranges.clone(),
        },
        NodeKind::BroadcastTo { a, shape } => NodeKind::BroadcastTo {
            a: f(a),
            shape: shape.clone(),
        },
        NodeKind::Checkpoint { a } => NodeKind::Checkpoint { a: f(a) },
        NodeKind::StopGradient { a } => NodeKind::StopGradient { a: f(a) },
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
        } => NodeKind::AdamWStep {
            param: f(param),
            grad: f(grad),
            m: f(m),
            v: f(v),
            lr: f(lr),
            c1: f(c1),
            c2: f(c2),
            beta1: *beta1,
            beta2: *beta2,
            eps: *eps,
            weight_decay: *weight_decay,
        },
        NodeKind::AdamWOut { step, index } => NodeKind::AdamWOut {
            step: f(step),
            index: *index,
        },
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
        } => NodeKind::AdamWStepGroup {
            params: params.iter().map(|p| f(p)).collect(),
            grads: grads.iter().map(|g| f(g)).collect(),
            ms: ms.iter().map(|m| f(m)).collect(),
            vs: vs.iter().map(|v| f(v)).collect(),
            lr: f(lr),
            c1: f(c1),
            c2: f(c2),
            beta1: *beta1,
            beta2: *beta2,
            eps: *eps,
            weight_decay: *weight_decay,
        },
        NodeKind::AdamWGroupOut { of, param, index } => NodeKind::AdamWGroupOut {
            of: f(of),
            param: *param,
            index: *index,
        },
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
        } => NodeKind::SgdStep {
            param: f(param),
            grad: f(grad),
            velocity: f(velocity),
            first: f(first),
            lr: f(lr),
            momentum: *momentum,
            dampening: *dampening,
            nesterov: *nesterov,
            weight_decay: *weight_decay,
        },
        NodeKind::SgdOut { step, index } => NodeKind::SgdOut {
            step: f(step),
            index: *index,
        },
        NodeKind::FusedElementwise {
            inputs,
            strides,
            shape,
            expr,
        } => NodeKind::FusedElementwise {
            inputs: inputs.iter().map(|i| f(i)).collect(),
            strides: strides.clone(),
            shape: shape.clone(),
            expr: expr.clone(),
        },
        NodeKind::FusedElementwiseMulti {
            inputs,
            strides,
            shape,
            exprs,
        } => NodeKind::FusedElementwiseMulti {
            inputs: inputs.iter().map(|i| f(i)).collect(),
            strides: strides.clone(),
            shape: shape.clone(),
            exprs: exprs.clone(),
        },
        NodeKind::FusedPick { of, index } => NodeKind::FusedPick {
            of: f(of),
            index: *index,
        },
        NodeKind::FusedReduce {
            inputs,
            strides,
            in_shape,
            expr,
            op,
            dims,
            keepdims,
            shape,
        } => NodeKind::FusedReduce {
            inputs: inputs.iter().map(|i| f(i)).collect(),
            strides: strides.clone(),
            in_shape: in_shape.clone(),
            expr: expr.clone(),
            op: *op,
            dims: dims.clone(),
            keepdims: *keepdims,
            shape: shape.clone(),
        },
    }
}


// depth, so chains of arbitrary length evaluate on a fixed stack. Children
// are always computed before their parents, and `eval_uncached` reads their
// values straight from the cache.
fn metal_to_cpu(t: &runtime::metal::run::MetalTensor) -> err::Res<runtime::cpu::Tensor> {
    let shape = t.layout.shape().to_vec();
    let values = val::Val::Metal(t.clone()).to_f32_vec()?;
    Ok(runtime::cpu::Tensor::from_vec(values, shape))
}

fn cpu_to_metal(t: &runtime::cpu::Tensor) -> err::Res<runtime::metal::run::MetalTensor> {
    let t = t.contiguous();
    let bytes: Vec<u8> = match &t.buffer {
        runtime::cpu::CpuBuffer::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        runtime::cpu::CpuBuffer::F64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        runtime::cpu::CpuBuffer::F16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        runtime::cpu::CpuBuffer::BF16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        runtime::cpu::CpuBuffer::U8(v) => v.as_slice().to_vec(),
        runtime::cpu::CpuBuffer::U32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        runtime::cpu::CpuBuffer::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    };
    Ok(runtime::metal::run::MetalTensor {
        buffer: runtime::metal::device::MetalDevice::get().upload_bytes(&bytes),
        layout: runtime::layout::Layout::contiguous(t.shape().to_vec()),
        dtype: t.dtype(),
    })
}

fn index_ids_u32(indexes: &val::Val) -> crate::err::Res<Vec<u32>> {
    indexes.to_u32_vec()
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
) -> crate::err::Res<(runtime::metal::run::MetalTensor, runtime::metal::run::MetalTensor, runtime::metal::run::MetalTensor)> {
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
        let decay = metal_ops::binary(p, &metal_ops::binary(lr, &wd, metal_ops::BinOp::Mul)?, metal_ops::BinOp::Mul)?;
        metal_ops::binary(
            &metal_ops::binary(p, &decay, metal_ops::BinOp::Sub)?,
            &adjusted,
            metal_ops::BinOp::Sub,
        )?
    };
    Ok((next_p, next_m, next_v))
}

fn eval_node(
    root: &Arc<Node>,
    cancelled: &AtomicBool,
    ev: &mut Evaluator,
) -> crate::err::Res<val::Val> {
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
            let output = eval_uncached(&node, ev)?;
            if let Some(t0) = t0 {
                kind_timing_nanos(node_kind_name(&node.kind), t0.elapsed().as_nanos() as u64);
            }
            ev.cache.insert(node.id, output);
            ev.release_children(&node);
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

fn eval_uncached(node: &Arc<Node>, ev: &mut Evaluator) -> crate::err::Res<val::Val> {
    let output = match &node.kind {
        NodeKind::Leaf(v) => v.clone(),
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
            device,
        } => {
            let nd = *dtype;
            if device.is_cpu() {
                let t = match nd {
                    runtime::dtype::DType::F32 => runtime::cpu::Tensor::from_vec(
                        data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                        shape.clone(),
                    ),
                    runtime::dtype::DType::F64 => runtime::cpu::Tensor::from_vec(
                        data.chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect(),
                        shape.clone(),
                    ),
                    runtime::dtype::DType::I64 => runtime::cpu::Tensor::from_vec(
                        data.chunks_exact(8).map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect(),
                        shape.clone(),
                    ),
                    runtime::dtype::DType::U8 => runtime::cpu::Tensor::from_vec(data.clone(), shape.clone()),
                    runtime::dtype::DType::U32 => runtime::cpu::Tensor::from_vec(
                        data.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                        shape.clone(),
                    ),
                    runtime::dtype::DType::F16 => runtime::cpu::Tensor::from_vec(
                        data.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]])).collect(),
                        shape.clone(),
                    ),
                    runtime::dtype::DType::BF16 => runtime::cpu::Tensor::from_vec(
                        data.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]])).collect(),
                        shape.clone(),
                    ),
                };
                val::Val::Cpu(t)
            } else {
                let buffer = runtime::metal::device::MetalDevice::get().upload_bytes(data);
                val::Val::Metal(runtime::metal::run::MetalTensor {
                    buffer,
                    layout: runtime::layout::Layout::contiguous(shape.clone()),
                    dtype: nd,
                })
            }
        }
        NodeKind::Zeros {
            shape,
            dtype,
            device,
        } => {
            if device.is_cpu() {
                val::Val::Cpu(runtime::cpu::Tensor::zeros(shape, *dtype))
            } else {
                val::Val::Metal(metal_ops::fill(shape, 0.0, *dtype)?)
            }
        }
        NodeKind::Ones {
            shape,
            dtype,
            device,
        } => {
            if device.is_cpu() {
                val::Val::Cpu(runtime::cpu::Tensor::ones(shape, *dtype))
            } else {
                val::Val::Metal(metal_ops::fill(shape, 1.0, *dtype)?)
            }
        }
        NodeKind::Full {
            shape,
            value,
            dtype,
            device,
        } => {
            if device.is_cpu() {
                val::Val::Cpu(runtime::cpu::Tensor::full(shape, *value, *dtype))
            } else {
                val::Val::Metal(metal_ops::fill(shape, *value, *dtype)?)
            }
        }
        NodeKind::Randn {
            shape,
            dtype,
            device,
        } => {
            if device.is_cpu() {
                val::Val::Cpu(runtime::cpu::Tensor::randn(shape, *dtype))
            } else {
                static SEED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(299792458);
                let seed = SEED.fetch_add(1, Ordering::Relaxed);
                let t = metal_ops::randn(shape, seed)?;
                val::Val::Metal(metal_ops::cast(&t, *dtype)?)
            }
        }
        NodeKind::Uniform {
            lo,
            hi,
            shape,
            dtype,
            device,
        } => {
            if device.is_cpu() {
                val::Val::Cpu(runtime::cpu::Tensor::uniform(*lo, *hi, shape, *dtype))
            } else {
                static SEED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(78778899);
                let seed = SEED.fetch_add(1, Ordering::Relaxed);
                let t = metal_ops::uniform(*lo, *hi, shape, seed)?;
                val::Val::Metal(metal_ops::cast(&t, *dtype)?)
            }
        }
        NodeKind::Arange {
            start,
            end,
            step,
            dtype,
            device,
        } => {
            if device.is_cpu() {
                val::Val::Cpu(runtime::cpu::Tensor::arange(*start, *end, *step, *dtype))
            } else {
                val::Val::Metal(metal_ops::arange(*start, *end, *step, *dtype)?)
            }
        }
        NodeKind::Eye { n, dtype, device } => {
            if device.is_cpu() {
                val::Val::Cpu(runtime::cpu::Tensor::eye(*n, *dtype))
            } else {
                val::Val::Metal(metal_ops::eye(*n, *dtype)?)
            }
        }
        NodeKind::Add { a, b } => {
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            match (&a, &b) {
                (val::Val::Cpu(x), val::Val::Cpu(y)) => val::Val::Cpu(x.add(y)),
                (val::Val::Metal(x), val::Val::Metal(y)) => {
                    val::Val::Metal(metal_ops::binary_promote(x, y, metal_ops::BinOp::Add)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Sub { a, b } => {
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            match (&a, &b) {
                (val::Val::Cpu(x), val::Val::Cpu(y)) => val::Val::Cpu(x.sub(y)),
                (val::Val::Metal(x), val::Val::Metal(y)) => {
                    val::Val::Metal(metal_ops::binary_promote(x, y, metal_ops::BinOp::Sub)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Mul { a, b } => {
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            match (&a, &b) {
                (val::Val::Cpu(x), val::Val::Cpu(y)) => val::Val::Cpu(x.mul(y)),
                (val::Val::Metal(x), val::Val::Metal(y)) => {
                    val::Val::Metal(metal_ops::binary_promote(x, y, metal_ops::BinOp::Mul)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Div { a, b } => {
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            match (&a, &b) {
                (val::Val::Cpu(x), val::Val::Cpu(y)) => val::Val::Cpu(x.div(y)),
                (val::Val::Metal(x), val::Val::Metal(y)) => {
                    val::Val::Metal(metal_ops::binary_promote(x, y, metal_ops::BinOp::Div)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Eq { a, b } => {
            let (x, y) = (ev.value(a.id)?, ev.value(b.id)?);
            match (&x, &y) {
                (val::Val::Cpu(a), val::Val::Cpu(b)) => val::Val::Cpu(a.eq(b)),
                (val::Val::Metal(a), val::Val::Metal(b)) => {
                    val::Val::Metal(metal_ops::compare(a, b, metal_ops::BinOp::Eq)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Gt { a, b } => {
            let (x, y) = (ev.value(a.id)?, ev.value(b.id)?);
            match (&x, &y) {
                (val::Val::Cpu(a), val::Val::Cpu(b)) => val::Val::Cpu(a.gt(b)),
                (val::Val::Metal(a), val::Val::Metal(b)) => {
                    val::Val::Metal(metal_ops::compare(a, b, metal_ops::BinOp::Gt)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Lt { a, b } => {
            let (x, y) = (ev.value(a.id)?, ev.value(b.id)?);
            match (&x, &y) {
                (val::Val::Cpu(a), val::Val::Cpu(b)) => val::Val::Cpu(a.lt(b)),
                (val::Val::Metal(a), val::Val::Metal(b)) => {
                    val::Val::Metal(metal_ops::compare(a, b, metal_ops::BinOp::Lt)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Ge { a, b } => {
            let (x, y) = (ev.value(a.id)?, ev.value(b.id)?);
            match (&x, &y) {
                (val::Val::Cpu(a), val::Val::Cpu(b)) => val::Val::Cpu(a.ge(b)),
                (val::Val::Metal(a), val::Val::Metal(b)) => {
                    val::Val::Metal(metal_ops::compare(a, b, metal_ops::BinOp::Ge)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Le { a, b } => {
            let (x, y) = (ev.value(a.id)?, ev.value(b.id)?);
            match (&x, &y) {
                (val::Val::Cpu(a), val::Val::Cpu(b)) => val::Val::Cpu(a.le(b)),
                (val::Val::Metal(a), val::Val::Metal(b)) => {
                    val::Val::Metal(metal_ops::compare(a, b, metal_ops::BinOp::Le)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Maximum { a, b } => {
            let (x, y) = (ev.value(a.id)?, ev.value(b.id)?);
            match (&x, &y) {
                (val::Val::Cpu(a), val::Val::Cpu(b)) => val::Val::Cpu(a.maximum(b)),
                (val::Val::Metal(a), val::Val::Metal(b)) => {
                    val::Val::Metal(metal_ops::binary_promote(a, b, metal_ops::BinOp::Max)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Minimum { a, b } => {
            let (x, y) = (ev.value(a.id)?, ev.value(b.id)?);
            match (&x, &y) {
                (val::Val::Cpu(a), val::Val::Cpu(b)) => val::Val::Cpu(a.minimum(b)),
                (val::Val::Metal(a), val::Val::Metal(b)) => {
                    val::Val::Metal(metal_ops::binary_promote(a, b, metal_ops::BinOp::Min)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Neg { a } => {
            let x = ev.value(a.id)?;
            match &x {
                val::Val::Cpu(t) => val::Val::Cpu(t.neg()),
                val::Val::Metal(t) => val::Val::Metal(metal_ops::unary_promote(t, metal_ops::UnOp::Neg)?),
            }
        }
        NodeKind::Abs { a } => {
            let x = ev.value(a.id)?;
            match &x {
                val::Val::Cpu(t) => val::Val::Cpu(t.abs()),
                val::Val::Metal(t) => val::Val::Metal(metal_ops::unary_promote(t, metal_ops::UnOp::Abs)?),
            }
        }
        NodeKind::Sqrt { a } => {
            let x = ev.value(a.id)?;
            match &x {
                val::Val::Cpu(t) => val::Val::Cpu(t.sqrt()),
                val::Val::Metal(t) => val::Val::Metal(metal_ops::unary_promote(t, metal_ops::UnOp::Sqrt)?),
            }
        }
        NodeKind::Exp { a } => {
            let x = ev.value(a.id)?;
            match &x {
                val::Val::Cpu(t) => val::Val::Cpu(t.exp()),
                val::Val::Metal(t) => val::Val::Metal(metal_ops::unary_promote(t, metal_ops::UnOp::Exp)?),
            }
        }
        NodeKind::Log { a } => {
            let x = ev.value(a.id)?;
            match &x {
                val::Val::Cpu(t) => val::Val::Cpu(t.log()),
                val::Val::Metal(t) => val::Val::Metal(metal_ops::unary_promote(t, metal_ops::UnOp::Log)?),
            }
        }
        NodeKind::Sin { a } => {
            let x = ev.value(a.id)?;
            match &x {
                val::Val::Cpu(t) => val::Val::Cpu(t.sin()),
                val::Val::Metal(t) => val::Val::Metal(metal_ops::unary_promote(t, metal_ops::UnOp::Sin)?),
            }
        }
        NodeKind::Cos { a } => {
            let x = ev.value(a.id)?;
            match &x {
                val::Val::Cpu(t) => val::Val::Cpu(t.cos()),
                val::Val::Metal(t) => val::Val::Metal(metal_ops::unary_promote(t, metal_ops::UnOp::Cos)?),
            }
        }
        NodeKind::Tanh { a } => {
            let x = ev.value(a.id)?;
            match &x {
                val::Val::Cpu(t) => val::Val::Cpu(t.tanh()),
                val::Val::Metal(t) => val::Val::Metal(metal_ops::unary_promote(t, metal_ops::UnOp::Tanh)?),
            }
        }
        NodeKind::Relu { a } => {
            let a = ev.value(a.id)?;
            match &a {
                val::Val::Cpu(t) => val::Val::Cpu(t.relu()),
                val::Val::Metal(t) => val::Val::Metal(metal_ops::relu(t)?),
            }
        }
        NodeKind::Erf { a } => {
            let x = ev.value(a.id)?;
            match &x {
                val::Val::Cpu(t) => val::Val::Cpu(t.erf()),
                val::Val::Metal(t) => val::Val::Metal(metal_ops::unary_promote(t, metal_ops::UnOp::Erf)?),
            }
        }
        NodeKind::Floor { a } => {
            let x = ev.value(a.id)?;
            match &x {
                val::Val::Cpu(t) => val::Val::Cpu(t.floor()),
                val::Val::Metal(t) => val::Val::Metal(metal_ops::unary_promote(t, metal_ops::UnOp::Floor)?),
            }
        }
        NodeKind::Ceil { a } => {
            let x = ev.value(a.id)?;
            match &x {
                val::Val::Cpu(t) => val::Val::Cpu(t.ceil()),
                val::Val::Metal(t) => val::Val::Metal(metal_ops::unary_promote(t, metal_ops::UnOp::Ceil)?),
            }
        }
        NodeKind::Round { a } => {
            let x = ev.value(a.id)?;
            match &x {
                val::Val::Cpu(t) => val::Val::Cpu(t.round()),
                val::Val::Metal(t) => val::Val::Metal(metal_ops::unary_promote(t, metal_ops::UnOp::Round)?),
            }
        }
        NodeKind::Sign { a } => {
            let x = ev.value(a.id)?;
            match &x {
                val::Val::Cpu(t) => val::Val::Cpu(t.sign()),
                val::Val::Metal(t) => val::Val::Metal(metal_ops::unary_promote(t, metal_ops::UnOp::Sign)?),
            }
        }
        NodeKind::Where { cond, a, b } => {
            let cond = ev.value(cond.id)?;
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            match (&a, &b, &cond) {
                (val::Val::Cpu(x), val::Val::Cpu(y), val::Val::Cpu(c)) => {
                    val::Val::Cpu(x.where_(c, y))
                }
                (val::Val::Metal(x), val::Val::Metal(y), val::Val::Metal(c)) => {
                    val::Val::Metal(metal_ops::where_(c, x, y)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Argmax { a, dim } => {
            let x = ev.value(a.id)?;
            match &x {
                val::Val::Cpu(t) => val::Val::Cpu(t.argmax(*dim).cast(runtime::dtype::DType::I64)),
                val::Val::Metal(t) => {
                    let r = metal_ops::argreduce(t, *dim, true)?;
                    val::Val::Metal(metal_ops::cast(&r, crate::runtime::dtype::DType::I64)?)
                }
            }
        }
        NodeKind::Argmin { a, dim } => {
            let x = ev.value(a.id)?;
            match &x {
                val::Val::Cpu(t) => val::Val::Cpu(t.argmin(*dim).cast(runtime::dtype::DType::I64)),
                val::Val::Metal(t) => {
                    let r = metal_ops::argreduce(t, *dim, false)?;
                    val::Val::Metal(metal_ops::cast(&r, crate::runtime::dtype::DType::I64)?)
                }
            }
        }
        NodeKind::Cumsum { a, dim } => {
            let x = ev.value(a.id)?;
            match &x {
                val::Val::Cpu(t) => val::Val::Cpu(t.cumsum(*dim)),
                val::Val::Metal(t) => val::Val::Metal(metal_ops::cumsum(t, *dim)?),
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
                (val::Val::Cpu(x), val::Val::Cpu(s)) => {
                    val::Val::Cpu(x.scatter_add(*dim, indexes.as_cpu()?, s))
                }
                (val::Val::Metal(x), val::Val::Metal(s)) => {
                    let ids = index_ids_u32(&indexes)?;
                    val::Val::Metal(metal_ops::scatter_add(x, *dim, &ids, s)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Gather { a, dim, indexes } => {
            let a = ev.value(a.id)?;
            let indexes = ev.value(indexes.id)?;
            match &a {
                val::Val::Cpu(x) => {
                    val::Val::Cpu(x.gather(*dim, indexes.as_cpu()?))
                }
                val::Val::Metal(x) => {
                    let ids = index_ids_u32(&indexes)?;
                    val::Val::Metal(metal_ops::gather(x, *dim, &ids, &indexes.shape())?)
                }
            }
        }
        NodeKind::IndexSelect { a, dim, indexes } => {
            let a = ev.value(a.id)?;
            let indexes = ev.value(indexes.id)?;
            match &a {
                val::Val::Cpu(x) => {
                    val::Val::Cpu(x.index_select(*dim, indexes.as_cpu()?))
                }
                val::Val::Metal(x) => {
                    let ids = index_ids_u32(&indexes)?;
                    val::Val::Metal(metal_ops::index_select(x, *dim, &ids)?)
                }
            }
        }
        NodeKind::CrossEntropy {
            logits,
            target,
            ignore_index,
        } => {
            let logits_t = ev.value(logits.id)?;
            let target_t = ev.value(target.id)?;
            match (&logits_t, &target_t) {
                (val::Val::Cpu(l), val::Val::Cpu(t)) => {
                    let r = runtime::cpu::composed::cross_entropy_forward(l, t, *ignore_index)?;
                    val::Val::Cpu(r)
                }
                (val::Val::Metal(l), val::Val::Metal(t)) => {
                    if loss::is_supported(l, t) {
                        let (loss_t, status) = loss::ce_forward(l, t, *ignore_index)?;
                        let classes = l.layout.shape()[l.layout.shape().len() - 1];
                        ev.ce_checks.push((val::Val::Metal(status), true, l.numel() / classes));
                        val::Val::Metal(loss_t)
                    } else {
                        let l32 = metal_ops::to_f32(l)?;
                        let r = composed::cross_entropy_forward(&l32, t, *ignore_index)?;
                        val::Val::Metal(r)
                    }
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::CrossEntropyBackward {
            logits,
            target,
            ignore_index,
        } => {
            let logits_t = ev.value(logits.id)?;
            let target_t = ev.value(target.id)?;
            match (&logits_t, &target_t) {
                (val::Val::Cpu(l), val::Val::Cpu(t)) => {
                    let r = runtime::cpu::composed::cross_entropy_backward(l, t, *ignore_index)?;
                    val::Val::Cpu(r)
                }
                (val::Val::Metal(l), val::Val::Metal(t)) => {
                    if loss::is_supported(l, t) {
                        let (grad, count) = loss::ce_backward(l, t, *ignore_index)?;
                        ev.ce_checks.push((val::Val::Metal(count), false, 0));
                        val::Val::Metal(grad)
                    } else {
                        let l32 = metal_ops::to_f32(l)?;
                        let r = composed::cross_entropy_backward(&l32, t, *ignore_index)?;
                        val::Val::Metal(r)
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
                (val::Val::Cpu(q), val::Val::Cpu(k), val::Val::Cpu(v)) => {
                    val::Val::Cpu(runtime::cpu::composed::sdpa_forward(q, k, v, *scale, *causal))
                }
                (val::Val::Metal(q), val::Val::Metal(k), val::Val::Metal(v)) => {
                    if q.dtype == runtime::dtype::DType::F32 {
                        let (o, l) = flash::forward(q, k, v, *scale, *causal)?;
                        // L rides the evaluator for the chunked backward; the
                        // node's own cache entry holds O.
                        ev.multi.insert(node.id, vec![val::Val::Metal(o.clone()), val::Val::Metal(l)]);
                        val::Val::Metal(o)
                    } else {
                        let q32 = metal_ops::to_f32(q)?;
                        let k32 = metal_ops::to_f32(k)?;
                        let v32 = metal_ops::to_f32(v)?;
                        let r = composed::sdpa_forward(&q32, &k32, &v32, *scale, *causal)?;
                        val::Val::Metal(metal_ops::from_f32(&r, q.dtype)?)
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
            fwd,
            scale,
            causal,
        } => {
            let q = ev.value(q.id)?;
            let k = ev.value(k.id)?;
            let v = ev.value(v.id)?;
            let g = ev.value(g.id)?;
            let o = ev.value(fwd.id)?;
            let l = ev.multi.get(&fwd.id).and_then(|outs| outs.get(1)).cloned();
            let (dq, dk, dv) = match (&q, &k, &v, &g) {
                (val::Val::Cpu(q), val::Val::Cpu(k), val::Val::Cpu(v), val::Val::Cpu(g)) => {
                    let (dq, dk, dv) = runtime::cpu::composed::sdpa_backward(q, k, v, g, *scale, *causal);
                    (
                        val::Val::Cpu(dq),
                        val::Val::Cpu(dk),
                        val::Val::Cpu(dv),
                    )
                }
                (val::Val::Metal(q), val::Val::Metal(k), val::Val::Metal(v), val::Val::Metal(g)) => {
                    if let (Some(val::Val::Metal(l)), true) = (&l, q.dtype == runtime::dtype::DType::F32) {
                        let o = o.as_metal()?;
                        let (dq, dk, dv) = flash::backward_fused(q, k, v, o, l, g, *scale, *causal)?;
                        (
                            val::Val::Metal(dq),
                            val::Val::Metal(dk),
                            val::Val::Metal(dv),
                        )
                    } else {
                        let q32 = metal_ops::to_f32(q)?;
                        let k32 = metal_ops::to_f32(k)?;
                        let v32 = metal_ops::to_f32(v)?;
                        let g32 = metal_ops::to_f32(g)?;
                        let (dq, dk, dv) =
                            composed::sdpa_backward(&q32, &k32, &v32, &g32, *scale, *causal)?;
                        (
                            val::Val::Metal(metal_ops::from_f32(&dq, q.dtype)?),
                            val::Val::Metal(metal_ops::from_f32(&dk, q.dtype)?),
                            val::Val::Metal(metal_ops::from_f32(&dv, q.dtype)?),
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
        NodeKind::PositionEmbedding { weight, seq_len } => {
            let w = ev.value(weight.id)?;
            match &w {
                val::Val::Cpu(w) => {
                    val::Val::Cpu(w.view(w.layout.narrow(0, 0, *seq_len)).contiguous())
                }
                val::Val::Metal(w) => {
                    let n = runtime::metal::run::MetalTensor {
                        buffer: w.buffer.clone(),
                        layout: w.layout.narrow(0, 0, *seq_len),
                        dtype: w.dtype,
                    };
                    val::Val::Metal(metal_ops::contiguous(&n)?)
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
                val::Val::Cpu(x) => {
                    let r = runtime::cpu::composed::rotary_forward(x, &offsets, *theta, 1.0)
                        .map_err(|e| e)?;
                    val::Val::Cpu(r)
                }
                val::Val::Metal(x) => {
                    if x.dtype == runtime::dtype::DType::F32 {
                        val::Val::Metal(rotary::rotary(x, &offsets, *theta, 1.0)?)
                    } else {
                        let x32 = metal_ops::to_f32(x)?;
                        let r = composed::rotary_forward(&x32, &offsets, *theta, 1.0)?;
                        val::Val::Metal(metal_ops::from_f32(&r, x.dtype)?)
                    }
                }
            }
        }
        NodeKind::RotaryEmbeddingBackward { g, theta, .. } => {
            let g = ev.value(g.id)?;
            match &g {
                val::Val::Cpu(g) => {
                    let r = runtime::cpu::composed::rotary_forward(g, &[0usize], *theta, -1.0)
                        .map_err(|e| e)?;
                    val::Val::Cpu(r)
                }
                val::Val::Metal(g) => {
                    if g.dtype == runtime::dtype::DType::F32 {
                        // Transpose rotation == forward with negated angles.
                        val::Val::Metal(rotary::rotary(g, &[0usize], *theta, -1.0)?)
                    } else {
                        let g32 = metal_ops::to_f32(g)?;
                        let r = composed::rotary_forward(&g32, &[0usize], *theta, -1.0)?;
                        val::Val::Metal(metal_ops::from_f32(&r, g.dtype)?)
                    }
                }
            }
        }
        NodeKind::Linear { x, weight, bias } => {
            let x = ev.value(x.id)?;
            let weight = ev.value(weight.id)?;
            let bias = ev.value(bias.id)?;
            match (&x, &weight, &bias) {
                (val::Val::Cpu(x), val::Val::Cpu(w), val::Val::Cpu(b)) => {
                    val::Val::Cpu(x.matmul(w).add(b))
                }
                (val::Val::Metal(x), val::Val::Metal(w), val::Val::Metal(b)) => {
                    if x.dtype == runtime::dtype::DType::F32 {
                        val::Val::Metal(linear::linear_forward(x, w, b)?)
                    } else {
                        let x32 = metal_ops::to_f32(x)?;
                        let w32 = metal_ops::to_f32(w)?;
                        let b32 = metal_ops::to_f32(b)?;
                        let r = metal_ops::binary(
                            &metal_ops::matmul(&x32, &w32)?,
                            &b32,
                            metal_ops::BinOp::Add,
                        )?;
                        val::Val::Metal(metal_ops::from_f32(&r, x.dtype)?)
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
                (val::Val::Cpu(x), val::Val::Cpu(w), val::Val::Cpu(b)) => {
                    val::Val::Cpu(runtime::cpu::composed::layer_norm_forward(x, w, b, *eps))
                }
                (val::Val::Metal(x), val::Val::Metal(w), val::Val::Metal(b)) => {
                    if layer_norm::is_supported(x, w) {
                        val::Val::Metal(layer_norm::ln_forward(x, w, b, *eps)?)
                    } else {
                        let x32 = metal_ops::to_f32(x)?;
                        let w32 = metal_ops::to_f32(w)?;
                        let b32 = metal_ops::to_f32(b)?;
                        let r = composed::layer_norm_forward(&x32, &w32, &b32, *eps)?;
                        val::Val::Metal(metal_ops::from_f32(&r, x.dtype)?)
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
                (val::Val::Cpu(x), val::Val::Cpu(w), val::Val::Cpu(g)) => {
                    let (dx, dw, db) = runtime::cpu::composed::layer_norm_backward(x, w, g, *eps);
                    ev.ln.insert(node.id, [val::Val::Cpu(dw), val::Val::Cpu(db)]);
                    val::Val::Cpu(dx)
                }
                (val::Val::Metal(x), val::Val::Metal(w), val::Val::Metal(g)) => {
                    if layer_norm::is_supported(x, w) {
                        let (dx, xh) = layer_norm::ln_backward(x, w, g, *eps)?;
                        let dw = metal_ops::reduce(
                            &metal_ops::binary(g, &xh, metal_ops::BinOp::Mul)?,
                            &(0..x.layout.shape().len() - w.layout.shape().len()).collect::<Vec<_>>(),
                            false,
                            crate::fusion::ReduceOp::Sum,
                        )?;
                        let db = metal_ops::reduce(
                            g,
                            &(0..x.layout.shape().len() - w.layout.shape().len()).collect::<Vec<_>>(),
                            false,
                            crate::fusion::ReduceOp::Sum,
                        )?;
                        ev.ln.insert(node.id, [val::Val::Metal(dw), val::Val::Metal(db)]);
                        val::Val::Metal(dx)
                    } else {
                        let x32 = metal_ops::to_f32(x)?;
                        let w32 = metal_ops::to_f32(w)?;
                        let g32 = metal_ops::to_f32(g)?;
                        let (dx, dw, db) =
                            composed::layer_norm_backward(&x32, &w32, &g32, *eps)?;
                        ev.ln.insert(node.id, [
                            val::Val::Metal(metal_ops::from_f32(&dw, w.dtype)?),
                            val::Val::Metal(metal_ops::from_f32(&db, w.dtype)?),
                        ]);
                        val::Val::Metal(metal_ops::from_f32(&dx, x.dtype)?)
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
                (val::Val::Cpu(x), val::Val::Cpu(w)) => {
                    val::Val::Cpu(runtime::cpu::conv::conv1d(x, w, *stride, *padding, *dilation, *groups))
                }
                (val::Val::Metal(x), val::Val::Metal(w)) => {
                    let xn = metal_ops::contiguous(x)?;
                    let wn = metal_ops::contiguous(w)?;
                    val::Val::Metal(runtime::metal::conv::conv1d(
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
                (val::Val::Cpu(x), val::Val::Cpu(w)) => {
                    val::Val::Cpu(runtime::cpu::conv::conv2d(x, w, *stride, *padding, *dilation, *groups))
                }
                (val::Val::Metal(x), val::Val::Metal(w)) => {
                    let xn = metal_ops::contiguous(x)?;
                    let wn = metal_ops::contiguous(w)?;
                    val::Val::Metal(runtime::metal::conv::conv2d(
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
                (val::Val::Cpu(x), val::Val::Cpu(w)) => {
                    val::Val::Cpu(runtime::cpu::conv::conv_transpose1d(
                        x,
                        w,
                        *stride,
                        *padding,
                        *output_padding,
                        *dilation,
                        *groups,
                    ))
                }
                (val::Val::Metal(x), val::Val::Metal(w)) => {
                    let xn = metal_ops::contiguous(x)?;
                    let wn = metal_ops::contiguous(w)?;
                    val::Val::Metal(runtime::metal::conv::conv_transpose1d(
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
                (val::Val::Cpu(x), val::Val::Cpu(w)) => {
                    val::Val::Cpu(runtime::cpu::conv::conv_transpose2d(
                        x,
                        w,
                        *stride,
                        *padding,
                        *output_padding,
                        *dilation,
                        *groups,
                    ))
                }
                (val::Val::Metal(x), val::Val::Metal(w)) => {
                    let xn = metal_ops::contiguous(x)?;
                    let wn = metal_ops::contiguous(w)?;
                    val::Val::Metal(runtime::metal::conv::conv_transpose2d(
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
                (val::Val::Cpu(x), val::Val::Cpu(g)) => {
                    let sq = |t: &runtime::cpu::Tensor| {
                        let s = t.shape();
                        t.contiguous().view(runtime::layout::Layout::contiguous(vec![s[0], s[1], s[2], 1]))
                    };
                    let dw = runtime::cpu::conv::conv2d_backward_w(
                        &sq(x),
                        &sq(g),
                        [*kernel, 1],
                        *out_channels,
                        *stride,
                        *padding,
                        *dilation,
                        *groups,
                    );
                    let s = dw.shape();
                    val::Val::Cpu(dw.contiguous().view(runtime::layout::Layout::contiguous(vec![s[0], s[1], s[2]])))
                }
                (val::Val::Metal(x), val::Val::Metal(g)) => {
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
                    val::Val::Metal(squeeze3(&dw))
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
                (val::Val::Cpu(x), val::Val::Cpu(g)) => {
                    val::Val::Cpu(runtime::cpu::conv::conv2d_backward_w(
                        x,
                        g,
                        *kernel,
                        *out_channels,
                        *stride,
                        *padding,
                        *dilation,
                        *groups,
                    ))
                }
                (val::Val::Metal(x), val::Val::Metal(g)) => {
                    let xn = metal_ops::contiguous(x)?;
                    let gn = metal_ops::contiguous(g)?;
                    val::Val::Metal(runtime::metal::conv::conv2d_backward_w(
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
                val::Val::Cpu(t) => val::Val::Cpu(t.powf(*exp)),
                val::Val::Metal(t) => val::Val::Metal(metal_ops::powf(&metal_ops::to_f32(t)?, *exp)?),
            }
        }
        NodeKind::Cast { a, dtype } => {
            let x = ev.value(a.id)?;
            match &x {
                val::Val::Cpu(t) => val::Val::Cpu(t.cast(*dtype)),
                val::Val::Metal(t) => {
                    val::Val::Metal(metal_ops::cast(t, *dtype)?)
                }
            }
        }
        NodeKind::Sum { a, dims, keepdims } => {
            let t = ev.value(a.id)?;
            match &t {
                val::Val::Cpu(x) => {
                    let r = x.sum(dims);
                    let r = if *keepdims { r } else { r.squeeze_dims(dims) };
                    val::Val::Cpu(r)
                }
                val::Val::Metal(x) => {
                    val::Val::Metal(metal_ops::reduce(&metal_ops::to_f32(x)?, dims, *keepdims, crate::fusion::ReduceOp::Sum)?)
                }
            }
        }
        NodeKind::Mean { a, dims, keepdims } => {
            let t = ev.value(a.id)?;
            match &t {
                val::Val::Cpu(x) => {
                    let r = x.mean(dims);
                    let r = if *keepdims { r } else { r.squeeze_dims(dims) };
                    val::Val::Cpu(r)
                }
                val::Val::Metal(x) => {
                    val::Val::Metal(metal_ops::reduce(&metal_ops::to_f32(x)?, dims, *keepdims, crate::fusion::ReduceOp::Mean)?)
                }
            }
        }
        NodeKind::Max { a, dims, keepdims } => {
            let t = ev.value(a.id)?;
            match &t {
                val::Val::Cpu(x) => {
                    let r = x.max(dims);
                    let r = if *keepdims { r } else { r.squeeze_dims(dims) };
                    val::Val::Cpu(r)
                }
                val::Val::Metal(x) => {
                    val::Val::Metal(metal_ops::reduce(&metal_ops::to_f32(x)?, dims, *keepdims, crate::fusion::ReduceOp::Max)?)
                }
            }
        }
        NodeKind::Min { a, dims, keepdims } => {
            let t = ev.value(a.id)?;
            match &t {
                val::Val::Cpu(x) => {
                    let r = x.min(dims);
                    let r = if *keepdims { r } else { r.squeeze_dims(dims) };
                    val::Val::Cpu(r)
                }
                val::Val::Metal(x) => {
                    val::Val::Metal(metal_ops::reduce(&metal_ops::to_f32(x)?, dims, *keepdims, crate::fusion::ReduceOp::Min)?)
                }
            }
        }
        NodeKind::Prod { a, dims, keepdims } => {
            let t = ev.value(a.id)?;
            match &t {
                val::Val::Cpu(x) => {
                    let r = x.prod(dims);
                    let r = if *keepdims { r } else { r.squeeze_dims(dims) };
                    val::Val::Cpu(r)
                }
                val::Val::Metal(x) => {
                    val::Val::Metal(metal_ops::reduce(&metal_ops::to_f32(x)?, dims, *keepdims, crate::fusion::ReduceOp::Prod)?)
                }
            }
        }
        NodeKind::Reshape { a, shape } => {
            let x = ev.value(a.id)?;
            match &x {
                val::Val::Cpu(t) => {
                    val::Val::Cpu(t.contiguous().view(runtime::layout::Layout::contiguous(shape.clone())))
                }
                val::Val::Metal(t) => {
                    let r = metal_ops::contiguous(t)?;
                    val::Val::Metal(runtime::metal::run::MetalTensor {
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
                val::Val::Cpu(t) => {
                    val::Val::Cpu(t.view(t.layout.permute(dims)).contiguous())
                }
                val::Val::Metal(t) => {
                    val::Val::Metal(metal_ops::permute(t, dims)?)
                }
            }
        }
        NodeKind::Slice { a, ranges } => {
            let t = ev.value(a.id)?;
            match &t {
                val::Val::Cpu(x) => {
                    let mut r = x.clone();
                    for (dim, &(start, stop, stride)) in ranges.iter().enumerate() {
                        let len = stop.saturating_sub(start).div_ceil(stride);
                        if len == 0 {
                            let mut shape = r.shape().to_vec();
                            shape[dim] = 0;
                            r = runtime::cpu::Tensor::zeros(&shape, r.dtype());
                            continue;
                        }
                        r = r.view(r.layout.narrow(dim, start, (len - 1) * stride + 1)).contiguous();
                        if stride > 1 {
                            let idx: Vec<u32> = (0..len as u32).map(|i| i * stride as u32).collect();
                            let idx = runtime::cpu::Tensor::from_vec(idx, vec![len]);
                            r = r.index_select(dim, &idx);
                        }
                    }
                    val::Val::Cpu(r)
                }
                val::Val::Metal(x) => {
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
                            r = metal_ops::index_select(&r, dim, &idx)?;
                        }
                    }
                    val::Val::Metal(r)
                }
            }
        }
        NodeKind::Concat { a, b, dim } => {
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            match (&a, &b) {
                (val::Val::Cpu(x), val::Val::Cpu(y)) => {
                    val::Val::Cpu(runtime::cpu::Tensor::cat(&[x, y], *dim))
                }
                (val::Val::Metal(x), val::Val::Metal(y)) => {
                    val::Val::Metal(metal_ops::cat(x, y, *dim)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::BroadcastTo { a, shape } => {
            let x = ev.value(a.id)?;
            match &x {
                val::Val::Cpu(t) => {
                    val::Val::Cpu(t.view(t.layout.broadcast_to(shape)).contiguous())
                }
                val::Val::Metal(t) => {
                    val::Val::Metal(metal_ops::broadcast_to(t, shape)?)
                }
            }
        }
        NodeKind::Matmul { a, b } => {
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            match (&a, &b) {
                (val::Val::Cpu(x), val::Val::Cpu(y)) => val::Val::Cpu(x.matmul(y)),
                (val::Val::Metal(x), val::Val::Metal(y)) => {
                    val::Val::Metal(metal_ops::matmul(x, y)?)
                }
                _ => return Err("device mismatch".to_string()),
            }
        }
        NodeKind::Inverse { a } => {
            let t = ev.value(a.id)?;
            match &t {
                val::Val::Cpu(x) => val::Val::Cpu(x.inverse()),
                val::Val::Metal(x) => {
                    let cpu = metal_to_cpu(x)?;
                    val::Val::Metal(cpu_to_metal(&cpu.inverse())?)
                }
            }
        }
        NodeKind::Det { a } => {
            let t = ev.value(a.id)?;
            match &t {
                val::Val::Cpu(x) => val::Val::Cpu(x.det()),
                val::Val::Metal(x) => {
                    let cpu = metal_to_cpu(x)?;
                    val::Val::Metal(cpu_to_metal(&cpu.det())?)
                }
            }
        }
        NodeKind::Solve { a, b } => {
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            match (&a, &b) {
                (val::Val::Cpu(x), val::Val::Cpu(y)) => val::Val::Cpu(x.solve(y)),
                (val::Val::Metal(x), val::Val::Metal(y)) => {
                    let xc = metal_to_cpu(x)?;
                    let yc = metal_to_cpu(y)?;
                    val::Val::Metal(cpu_to_metal(&xc.solve(&yc))?)
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
            // parameter dtype and copy to its device (they may be CPU f64 —
            // the bias corrections are computed in f64 to avoid
            // cancellation — and 0-d copies are free).
            let lr_t = ev.step_scalar(lr.id, p.dtype(), &p.device())?;
            let c1_t = ev.step_scalar(c1.id, p.dtype(), &p.device())?;
            let c2_t = ev.step_scalar(c2.id, p.dtype(), &p.device())?;
            let fused = if fusion::is_supported(&p.device(), p.dtype()) {
                if p.device().is_cpu() {
                    let exprs = fusion::adamw_exprs(*beta1, *beta2, *eps, *weight_decay);
                    fusion::run(
                        &exprs,
                        &[p.clone(), g.clone(), m_t.clone(), v_t.clone()],
                        None,
                        &[lr_t.clone(), c1_t.clone(), c2_t.clone()],
                        p.numel(),
                        &p.shape(),
                        p.dtype(),
                        &p.device(),
                    )
                    .ok()
                } else {
                    let plan = fusion::adamw_group_plan(1, &p.shape(), *beta1, *beta2, *eps, *weight_decay);
                    let pack = match ev.scalar_packs.entry((lr.id, c1.id, c2.id)) {
                        std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                        std::collections::hash_map::Entry::Vacant(e) => {
                            e.insert(fusion::pack_scalars_metal(&[lr_t.clone(), c1_t.clone(), c2_t.clone()])?)
                        }
                    };
                    fusion::run_group_metal(&plan, &[p.clone(), g.clone(), m_t.clone(), v_t.clone()], pack, &p.shape()).ok()
                }
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
                    val::Val::Cpu(p) => {
                        let (next_p, next_m, next_v) = runtime::cpu::composed::adamw_step(
                            p,
                            g.as_cpu()?,
                            m_t.as_cpu()?,
                            v_t.as_cpu()?,
                            lr_t.as_cpu()?,
                            c1_t.as_cpu()?,
                            c2_t.as_cpu()?,
                            *beta1,
                            *beta2,
                            *eps,
                            *weight_decay,
                        );
                        ev.adamw.insert(node.id, [val::Val::Cpu(next_m), val::Val::Cpu(next_v)]);
                        val::Val::Cpu(next_p)
                    }
                    val::Val::Metal(p) => {
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
                        ev.adamw.insert(node.id, [val::Val::Metal(nm), val::Val::Metal(nv)]);
                        val::Val::Metal(np)
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
            let outs = if !first_p.device().is_cpu() && fusion::is_supported(&first_p.device(), first_p.dtype()) {
                let plan = fusion::adamw_group_plan(params.len(), &first_p.shape(), *beta1, *beta2, *eps, *weight_decay);
                let pack = match ev.scalar_packs.entry((lr.id, c1.id, c2.id)) {
                    std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(fusion::pack_scalars_metal(&[lr_t, c1_t, c2_t])?)
                    }
                };
                fusion::run_group_metal(&plan, &inputs, pack, &first_p.shape())?
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
                    val::Val::Cpu(p) => {
                        let (next_p, next_v) = runtime::cpu::composed::sgd_step(
                            p,
                            g.as_cpu()?,
                            v_t.as_cpu()?,
                            lr_t.as_cpu()?,
                            first_t.as_cpu()?,
                            *momentum,
                            *dampening,
                            *nesterov,
                            *weight_decay,
                        );
                        ev.sgd.insert(node.id, val::Val::Cpu(next_v));
                        val::Val::Cpu(next_p)
                    }
                    val::Val::Metal(p) => {
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
                        ev.sgd.insert(node.id, val::Val::Metal(next_v));
                        val::Val::Metal(next_p)
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
            let ts: Vec<val::Val> = inputs
                .iter()
                .map(|i| ev.value(i.id))
                .collect::<crate::err::Res<Vec<_>>>()?;
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
            let ts: Vec<val::Val> = inputs
                .iter()
                .map(|i| ev.value(i.id))
                .collect::<crate::err::Res<Vec<_>>>()?;
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
            let ts: Vec<val::Val> = inputs
                .iter()
                .map(|i| ev.value(i.id))
                .collect::<crate::err::Res<Vec<_>>>()?;
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

// Layer normalization, composed of candle ops (CPU path and the
// reference for the fused Metal kernels in layer_norm.rs).

// Reverse-mode automatic differentiation: adjoints are built from the same
// node vocabulary as the forward graph, so gradients can be differentiated
// again and executor optimizations apply uniformly.
mod autodiff {
    use super::*;
    use std::collections::HashMap;

    fn mk(kind: NodeKind) -> std::result::Result<Arc<Node>, String> {
        Node::new(kind)
    }

    fn full(value: f64, dtype: DType, device: &Device) -> std::result::Result<Arc<Node>, String> {
        mk(NodeKind::Full {
            shape: vec![],
            value,
            dtype,
            device: device.clone(),
        })
    }

    fn zeros_like(target: &Node) -> std::result::Result<Arc<Node>, String> {
        mk(NodeKind::Zeros {
            shape: target.shape.clone(),
            dtype: target.dtype,
            device: target.device.clone(),
        })
    }

    fn ones(dtype: DType, device: &Device) -> std::result::Result<Arc<Node>, String> {
        mk(NodeKind::Ones {
            shape: vec![],
            dtype,
            device: device.clone(),
        })
    }

    fn add(a: Arc<Node>, b: Arc<Node>) -> std::result::Result<Arc<Node>, String> {
        mk(NodeKind::Add { a, b })
    }

    fn sub(a: Arc<Node>, b: Arc<Node>) -> std::result::Result<Arc<Node>, String> {
        mk(NodeKind::Sub { a, b })
    }

    fn mul(a: Arc<Node>, b: Arc<Node>) -> std::result::Result<Arc<Node>, String> {
        mk(NodeKind::Mul { a, b })
    }

    fn div(a: Arc<Node>, b: Arc<Node>) -> std::result::Result<Arc<Node>, String> {
        mk(NodeKind::Div { a, b })
    }

    fn neg(a: Arc<Node>) -> std::result::Result<Arc<Node>, String> {
        mk(NodeKind::Neg { a })
    }

    fn cast(a: Arc<Node>, dtype: DType) -> std::result::Result<Arc<Node>, String> {
        if a.dtype == dtype {
            return Ok(a);
        }
        mk(NodeKind::Cast { a, dtype })
    }

    fn reshape(a: Arc<Node>, shape: Vec<usize>) -> std::result::Result<Arc<Node>, String> {
        if a.shape == shape {
            return Ok(a);
        }
        mk(NodeKind::Reshape { a, shape })
    }

    fn broadcast_to(a: Arc<Node>, shape: &[usize]) -> std::result::Result<Arc<Node>, String> {
        if a.shape == shape {
            return Ok(a);
        }
        mk(NodeKind::BroadcastTo {
            a,
            shape: shape.to_vec(),
        })
    }

    fn transpose2(a: &Arc<Node>) -> std::result::Result<Arc<Node>, String> {
        let rank = a.shape.len();
        let mut dims: Vec<usize> = (0..rank).collect();
        dims.swap(rank - 2, rank - 1);
        mk(NodeKind::Permute {
            a: a.clone(),
            dims,
        })
    }

    // Sum g over the dims that broadcasting expanded, then reshape to target.
    fn sum_to_shape(g: &Arc<Node>, target: &[usize]) -> std::result::Result<Arc<Node>, String> {
        if g.shape == target {
            return Ok(g.clone());
        }
        if g.shape.len() < target.len() {
            return Err(format!(
                "grad: cannot reduce {:?} to higher-rank shape {target:?}",
                g.shape
            ));
        }
        let extra = g.shape.len() - target.len();
        let mut dims: Vec<usize> = (0..extra).collect();
        for i in extra..g.shape.len() {
            if target[i - extra] == 1 && g.shape[i] != 1 {
                dims.push(i);
            }
        }
        let out = if dims.is_empty() {
            g.clone()
        } else {
            mk(NodeKind::Sum {
                a: g.clone(),
                dims,
                keepdims: true,
            })?
        };
        reshape(out, target.to_vec())
    }

    // Broadcast a reduced cotangent (and output) back to the input shape,
    // re-inserting size-1 dims when keepdims was false.
    fn expand_reduced(
        g: &Arc<Node>,
        dims: &[usize],
        keepdims: bool,
        target: &[usize],
    ) -> std::result::Result<Arc<Node>, String> {
        let g = if keepdims {
            g.clone()
        } else {
            let kept: Vec<usize> = target
                .iter()
                .enumerate()
                .map(|(i, &d)| if dims.contains(&i) { 1 } else { d })
                .collect();
            reshape(g.clone(), kept)?
        };
        broadcast_to(g, target)
    }

    fn topo(loss: &Arc<Node>) -> Vec<Arc<Node>> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        let mut stack = vec![(loss.clone(), false)];
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
        order
    }

    // Nodes of `root`'s graph whose subtree contains `x_id` — the subgraph
    // that must be rebuilt under vmap.
    fn descendants_of(root: &Arc<Node>, x_id: u64) -> HashSet<u64> {
        let mut set = HashSet::new();
        for node in topo(root) {
            if node.id == x_id || node_children(&node.kind).iter().any(|c| set.contains(&c.id)) {
                set.insert(node.id);
            }
        }
        set
    }

    fn shift_dim(d: usize, batch_dim: usize) -> usize {
        if d >= batch_dim { d + 1 } else { d }
    }

    fn insert_batch(shape: &[usize], dim: usize, batch: usize) -> Vec<usize> {
        let mut out = shape.to_vec();
        out.insert(dim.min(out.len()), batch);
        out
    }

    // Unsqueezes shared indexes at the batch dim and broadcasts them across
    // it, so rank-matched indexing kernels apply per batch element.
    fn broadcast_batch_indexes(
        indexes: &Arc<Node>,
        dim: usize,
        batch: usize,
    ) -> std::result::Result<Arc<Node>, String> {
        let unsqueezed = mk(NodeKind::Reshape {
            a: indexes.clone(),
            shape: insert_batch(&indexes.shape, dim, 1),
        })?;
        mk(NodeKind::BroadcastTo {
            a: unsqueezed,
            shape: insert_batch(&indexes.shape, dim, batch),
        })
    }

    // Per-op batching rules: rebuild a node for a graph whose input gained
    // a leading-dim-style batch axis at `dim`. Elementwise ops, matmul and
    // wrappers are unchanged (broadcasting carries the batch); shape and
    // reduction metadata shifts around the inserted axis; random sources
    // draw per batch element; indexing with data-dependent indexes and
    // gather/scatter are rejected for now.
    fn vmap_rebuild(
        node: &Node,
        dim: usize,
        batch: usize,
        f: &dyn Fn(&Arc<Node>) -> Arc<Node>,
        is_batched: &dyn Fn(u64) -> bool,
    ) -> std::result::Result<NodeKind, String> {
        let shift_dims = |dims: &[usize]| dims.iter().map(|&d| shift_dim(d, dim)).collect();
        match &node.kind {
            NodeKind::Randn {
                shape,
                dtype,
                device,
            } => Ok(NodeKind::Randn {
                shape: insert_batch(shape, dim, batch),
                dtype: *dtype,
                device: device.clone(),
            }),
            NodeKind::Uniform {
                lo,
                hi,
                shape,
                dtype,
                device,
            } => Ok(NodeKind::Uniform {
                lo: *lo,
                hi: *hi,
                shape: insert_batch(shape, dim, batch),
                dtype: *dtype,
                device: device.clone(),
            }),
            NodeKind::Sum { a, dims, keepdims } => Ok(NodeKind::Sum {
                a: f(a),
                dims: shift_dims(dims),
                keepdims: *keepdims,
            }),
            NodeKind::Mean { a, dims, keepdims } => Ok(NodeKind::Mean {
                a: f(a),
                dims: shift_dims(dims),
                keepdims: *keepdims,
            }),
            NodeKind::Max { a, dims, keepdims } => Ok(NodeKind::Max {
                a: f(a),
                dims: shift_dims(dims),
                keepdims: *keepdims,
            }),
            NodeKind::Min { a, dims, keepdims } => Ok(NodeKind::Min {
                a: f(a),
                dims: shift_dims(dims),
                keepdims: *keepdims,
            }),
            NodeKind::Prod { a, dims, keepdims } => Ok(NodeKind::Prod {
                a: f(a),
                dims: shift_dims(dims),
                keepdims: *keepdims,
            }),
            NodeKind::Argmax { a, dim: d } => Ok(NodeKind::Argmax {
                a: f(a),
                dim: shift_dim(*d, dim),
            }),
            NodeKind::Argmin { a, dim: d } => Ok(NodeKind::Argmin {
                a: f(a),
                dim: shift_dim(*d, dim),
            }),
            NodeKind::Cumsum { a, dim: d } => Ok(NodeKind::Cumsum {
                a: f(a),
                dim: shift_dim(*d, dim),
            }),
            NodeKind::Reshape { a, shape } => Ok(NodeKind::Reshape {
                a: f(a),
                shape: insert_batch(shape, dim, batch),
            }),
            NodeKind::Permute { a, dims } => {
                let mut out: Vec<usize> = dims.iter().map(|&d| shift_dim(d, dim)).collect();
                out.insert(dim, dim);
                Ok(NodeKind::Permute { a: f(a), dims: out })
            }
            NodeKind::Slice { a, ranges } => {
                let mut out = ranges.clone();
                out.insert(dim, (0, batch, 1));
                Ok(NodeKind::Slice { a: f(a), ranges: out })
            }
            NodeKind::Concat { a, b, dim: d } => Ok(NodeKind::Concat {
                a: f(a),
                b: f(b),
                dim: shift_dim(*d, dim),
            }),
            NodeKind::BroadcastTo { a, shape } => Ok(NodeKind::BroadcastTo {
                a: f(a),
                shape: insert_batch(shape, dim, batch),
            }),
            NodeKind::IndexSelect { a, dim: d, indexes } => {
                if is_batched(indexes.id) {
                    return Err(
                        "vmap: index_select with data-dependent indexes is not supported"
                            .to_string(),
                    );
                }
                Ok(NodeKind::IndexSelect {
                    a: f(a),
                    dim: shift_dim(*d, dim),
                    indexes: indexes.clone(),
                })
            }
            NodeKind::Gather { indexes, .. } | NodeKind::ScatterAdd { indexes, .. } => {
                if is_batched(indexes.id) {
                    Err(
                        "vmap: gather and scatterAdd with data-dependent indexes are not supported"
                            .to_string(),
                    )
                } else {
                    Err(
                        "vmap: gather and scatterAdd with shared indexes are not supported under vmap (requires a batched-gather kernel)"
                            .to_string(),
                    )
                }
            }
            NodeKind::CrossEntropy { .. } | NodeKind::CrossEntropyBackward { .. } => Err(
                "vmap: crossEntropy uses data-dependent indexing and is not supported under vmap"
                    .to_string(),
            ),
            NodeKind::SdpaBackward { .. } | NodeKind::SdpaBackwardOut { .. } => Err(
                "vmap: sdpa backward nodes are internal to autodiff".to_string(),
            ),
            NodeKind::PositionEmbedding { .. } | NodeKind::KvAttention { .. } => Err(
                "vmap: position embedding and kv attention nodes are not supported under vmap"
                    .to_string(),
            ),
            NodeKind::RotaryEmbedding { .. } => {
                Err("vmap: rotary embedding nodes are not supported under vmap".to_string())
            }
            NodeKind::Conv1d { .. }
            | NodeKind::Conv2d { .. }
            | NodeKind::ConvTranspose1d { .. }
            | NodeKind::ConvTranspose2d { .. }
            | NodeKind::Conv1dBackwardW { .. }
            | NodeKind::Conv2dBackwardW { .. } => {
                Err("vmap: convolution nodes are not supported under vmap".to_string())
            }
            NodeKind::FusedElementwise { .. }
            | NodeKind::FusedElementwiseMulti { .. }
            | NodeKind::FusedPick { .. }
            | NodeKind::FusedReduce { .. } => {
                Err("vmap: fused elementwise nodes are internal to evaluation".to_string())
            }
            _ => Ok(remap_children(&node.kind, f)),
        }
    }

    pub fn vmap(
        y: &Arc<Node>,
        x: &Arc<Node>,
        batched: &Arc<Node>,
        dim: usize,
    ) -> std::result::Result<Arc<Node>, String> {
        if batched.shape.len() != x.shape.len() + 1 || dim >= batched.shape.len() {
            return Err(format!(
                "vmap: batched input shape {:?} must be the input shape {:?} with one dimension inserted",
                batched.shape, x.shape
            ));
        }
        for (i, &d) in x.shape.iter().enumerate() {
            let at = if i < dim { i } else { i + 1 };
            if batched.shape[at] != d {
                return Err(format!(
                    "vmap: batched input shape {:?} does not match input shape {:?} outside dim {dim}",
                    batched.shape, x.shape
                ));
            }
        }
        if batched.dtype != x.dtype {
            return Err(format!(
                "vmap: dtype mismatch, got {:?} and {:?}",
                batched.dtype, x.dtype
            ));
        }
        let batch = batched.shape[dim];
        let descendants = descendants_of(y, x.id);
        if !descendants.contains(&y.id) {
            return Err("vmap: the output does not depend on the input".to_string());
        }
        let mut map: HashMap<u64, Arc<Node>> = HashMap::new();
        map.insert(x.id, batched.clone());
        for node in topo(y) {
            // random sources inside the mapped graph draw per batch element
            // even when they do not depend on the input; everything else is
            // rebuilt only when it descends from the input
            let is_random = matches!(node.kind, NodeKind::Randn { .. } | NodeKind::Uniform { .. });
            if node.id == x.id || (!is_random && !descendants.contains(&node.id)) {
                continue;
            }
            let child_of = |child: &Arc<Node>| map.get(&child.id).cloned().unwrap_or_else(|| child.clone());
            let rebuilt = match &node.kind {
                // shared indexes are reshaped and broadcast across the batch
                // dim so the rank-matched gather/scatter kernels apply per
                // batch element
                NodeKind::Gather { a, dim: d, indexes }
                    if !descendants.contains(&indexes.id) && !is_random =>
                {
                    let idx = broadcast_batch_indexes(indexes, dim, batch)?;
                    mk(NodeKind::Gather {
                        a: child_of(a),
                        dim: shift_dim(*d, dim),
                        indexes: idx,
                    })?
                }
                NodeKind::ScatterAdd {
                    a,
                    dim: d,
                    indexes,
                    src,
                } if !descendants.contains(&indexes.id)
                    && descendants.contains(&a.id)
                    && !is_random =>
                {
                    let idx = broadcast_batch_indexes(indexes, dim, batch)?;
                    mk(NodeKind::ScatterAdd {
                        a: child_of(a),
                        dim: shift_dim(*d, dim),
                        indexes: idx,
                        src: child_of(src),
                    })?
                }
                _ => mk(vmap_rebuild(
                    &node,
                    dim,
                    batch,
                    &child_of,
                    &|id: u64| map.contains_key(&id),
                )?)?,
            };
            map.insert(node.id, rebuilt);
        }
        Ok(map.get(&y.id).expect("vmap root").clone())
    }

    pub fn grad(
        loss: &Arc<Node>,
        wrt: &[Arc<Node>],
    ) -> std::result::Result<Vec<Arc<Node>>, String> {
        if !loss.shape.is_empty() {
            return Err(format!(
                "grad: expected a scalar (0-d) loss, got shape {:?}",
                loss.shape
            ));
        }
        if !loss.dtype.is_float() {
            return Err(format!(
                "grad: loss dtype must be floating point, got {:?}",
                loss.dtype
            ));
        }
        for target in wrt {
            if !target.dtype.is_float() {
                return Err(format!(
                    "grad: cannot differentiate with respect to non-float dtype {:?}",
                    target.dtype
                ));
            }
        }
        let order = topo(loss);
        let mut cotangents: HashMap<u64, Arc<Node>> = HashMap::new();
        cotangents.insert(loss.id, ones(loss.dtype, &loss.device)?);
        backward(&order, &mut cotangents)?;
        Ok(wrt
            .iter()
            .map(|target| match cotangents.get(&target.id) {
                Some(g) => g.clone(),
                None => zeros_like(target).expect("zeros_like"),
            })
            .collect())
    }

    // Nodes reachable from the walk's root without passing through the
    // checkpoint — these are the region's inputs and stay shared.
    fn outside_set(order: &[Arc<Node>], checkpoint_id: u64) -> HashSet<u64> {
        let mut visited = HashSet::new();
        if let Some(root) = order.last() {
            let mut stack = vec![root.clone()];
            while let Some(node) = stack.pop() {
                if node.id == checkpoint_id {
                    continue;
                }
                if !visited.insert(node.id) {
                    continue;
                }
                for child in node_children(&node.kind) {
                    stack.push(child);
                }
            }
        }
        visited
    }

    fn backward(
        order: &[Arc<Node>],
        cotangents: &mut HashMap<u64, Arc<Node>>,
    ) -> std::result::Result<(), String> {
        for node in order.iter().rev() {
            let Some(g) = cotangents.get(&node.id).cloned() else {
                continue;
            };
            // Gradients do not flow through non-float nodes (comparisons,
            // integer arithmetic): their mathematical gradient is zero
            // almost everywhere, so the cotangent is dropped here.
            if !node.dtype.is_float() {
                continue;
            }
            let mut accumulate = |input: &Arc<Node>,
                                  contribution: std::result::Result<Arc<Node>, String>| {
                let contribution = contribution?;
                cotangents
                    .entry(input.id)
                    .and_modify(|existing| {
                        *existing = add(existing.clone(), contribution.clone())
                            .expect("grad accumulation broadcast")
                    })
                    .or_insert(contribution);
                Ok::<(), String>(())
            };
            match &node.kind {
                NodeKind::Add { a, b } => {
                    accumulate(a, sum_to_shape(&g, &a.shape))?;
                    accumulate(b, sum_to_shape(&g, &b.shape))?;
                }
                NodeKind::Sub { a, b } => {
                    accumulate(a, sum_to_shape(&g, &a.shape))?;
                    accumulate(b, sum_to_shape(&neg(g)?, &b.shape))?;
                }
                NodeKind::Mul { a, b } => {
                    accumulate(a, sum_to_shape(&mul(g.clone(), b.clone())?, &a.shape))?;
                    accumulate(b, sum_to_shape(&mul(g.clone(), a.clone())?, &b.shape))?;
                }
                NodeKind::Div { a, b } => {
                    accumulate(a, sum_to_shape(&div(g.clone(), b.clone())?, &a.shape))?;
                    let gb = neg(div(mul(g.clone(), a.clone())?, mul(b.clone(), b.clone())?)?)?;
                    accumulate(b, sum_to_shape(&gb, &b.shape))?;
                }
                NodeKind::Neg { a } => {
                    accumulate(a, neg(g))?;
                }
                NodeKind::Maximum { a, b } | NodeKind::Minimum { a, b } => {
                    let is_max = matches!(&node.kind, NodeKind::Maximum { .. });
                    // ties route the gradient to the left operand
                    let (mask_a, mask_b) = if is_max {
                        (
                            mk(NodeKind::Ge { a: a.clone(), b: b.clone() })?,
                            mk(NodeKind::Lt { a: a.clone(), b: b.clone() })?,
                        )
                    } else {
                        (
                            mk(NodeKind::Le { a: a.clone(), b: b.clone() })?,
                            mk(NodeKind::Gt { a: a.clone(), b: b.clone() })?,
                        )
                    };
                    let dtype = node.dtype;
                    let ga = mul(g.clone(), cast(mask_a, dtype)?)?;
                    let gb = mul(g, cast(mask_b, dtype)?)?;
                    accumulate(a, sum_to_shape(&ga, &a.shape))?;
                    accumulate(b, sum_to_shape(&gb, &b.shape))?;
                }
                NodeKind::Abs { a } => {
                    let zero = full(0.0, a.dtype, &a.device)?;
                    let sign = mk(NodeKind::Sub {
                        a: cast(mk(NodeKind::Gt { a: a.clone(), b: zero.clone() })?, a.dtype)?,
                        b: cast(mk(NodeKind::Lt { a: a.clone(), b: zero })?, a.dtype)?,
                    })?;
                    accumulate(a, mul(g, sign))?;
                }
                NodeKind::Sqrt { a } => {
                    let half = full(0.5, node.dtype, &node.device)?;
                    accumulate(a, div(mul(g, half)?, node.clone()))?;
                }
                NodeKind::Exp { a } => {
                    accumulate(a, mul(g, node.clone()))?;
                }
                NodeKind::Log { a } => {
                    accumulate(a, div(g, a.clone()))?;
                }
                NodeKind::Sin { a } => {
                    accumulate(a, mul(g, mk(NodeKind::Cos { a: a.clone() })?))?;
                }
                NodeKind::Cos { a } => {
                    accumulate(a, neg(mul(g, mk(NodeKind::Sin { a: a.clone() })?)?))?;
                }
                NodeKind::Tanh { a } => {
                    let one = full(1.0, node.dtype, &node.device)?;
                    accumulate(a, mul(g, add(one, neg(mul(node.clone(), node.clone())?)?)?))?;
                }
                NodeKind::Relu { a } => {
                    let zero = full(0.0, a.dtype, &a.device)?;
                    let mask = cast(mk(NodeKind::Gt { a: a.clone(), b: zero })?, a.dtype)?;
                    accumulate(a, mul(g, mask))?;
                }
                NodeKind::Erf { a } => {
                    let c = full(2.0 / std::f64::consts::PI.sqrt(), a.dtype, &a.device)?;
                    let e = mk(NodeKind::Exp {
                        a: neg(mul(a.clone(), a.clone())?)?,
                    })?;
                    accumulate(a, mul(mul(g, c)?, e))?;
                }
                // zero almost everywhere; the cotangent is an explicit zero
                // rather than a drop so higher-order walks stay total
                NodeKind::Floor { a }
                | NodeKind::Ceil { a }
                | NodeKind::Round { a }
                | NodeKind::Sign { a } => {
                    accumulate(a, zeros_like(a))?;
                }
                NodeKind::Where { cond, a, b } => {
                    let zero = full(0.0, node.dtype, &node.device)?;
                    let ga = mk(NodeKind::Where {
                        cond: cond.clone(),
                        a: g.clone(),
                        b: zero.clone(),
                    })?;
                    let gb = mk(NodeKind::Where {
                        cond: cond.clone(),
                        a: zero,
                        b: g.clone(),
                    })?;
                    accumulate(a, sum_to_shape(&ga, &a.shape))?;
                    accumulate(b, sum_to_shape(&gb, &b.shape))?;
                }
                NodeKind::Cumsum { a, dim } => {
                    // d out[i] / d x[j] = 1 when i >= j, so the adjoint is the
                    // reverse cumulative sum: total - cumsum(g) + g
                    let total = mk(NodeKind::Sum {
                        a: g.clone(),
                        dims: vec![*dim],
                        keepdims: true,
                    })?;
                    let total = broadcast_to(total, &a.shape)?;
                    let cs = mk(NodeKind::Cumsum {
                        a: g.clone(),
                        dim: *dim,
                    })?;
                    accumulate(a, add(g.clone(), sub(total, cs)?))?;
                }
                NodeKind::IndexSelect { a, dim, indexes } => {
                    // scatter the cotangent back into a zero tensor of the
                    // input shape at the selected positions
                    let mut ishape = vec![1usize; a.shape.len()];
                    ishape[*dim] = indexes.shape[0];
                    let idx = reshape(indexes.clone(), ishape)?;
                    let idx = broadcast_to(idx, &g.shape)?;
                    accumulate(
                        a,
                        mk(NodeKind::ScatterAdd {
                            a: zeros_like(a.as_ref())?,
                            dim: *dim,
                            indexes: idx,
                            src: g,
                        }),
                    )?;
                }
                NodeKind::ScatterAdd {
                    a,
                    dim,
                    indexes,
                    src,
                } => {
                    accumulate(a, Ok(g.clone()))?;
                    accumulate(
                        src,
                        mk(NodeKind::Gather {
                            a: g,
                            dim: *dim,
                            indexes: indexes.clone(),
                        }),
                    )?;
                }
                NodeKind::Gather { a, dim, indexes } => {
                    // the scatter kernel requires src to match the target
                    // outside dim, so pad the cotangent and the indexes with
                    // harmless zeros (index 0, value 0) where they are smaller
                    let mut g_padded = g;
                    let mut idx_padded = indexes.clone();
                    for i in 0..a.shape.len() {
                        if i == *dim {
                            continue;
                        }
                        let missing = a.shape[i].saturating_sub(g_padded.shape[i]);
                        if missing > 0 {
                            let mut zshape = g_padded.shape.clone();
                            zshape[i] = missing;
                            let mut ishape = idx_padded.shape.clone();
                            ishape[i] = missing;
                            g_padded = mk(NodeKind::Concat {
                                a: g_padded.clone(),
                                b: mk(NodeKind::Zeros {
                                    shape: zshape,
                                    dtype: g_padded.dtype,
                                    device: g_padded.device.clone(),
                                })?,
                                dim: i,
                            })?;
                            idx_padded = mk(NodeKind::Concat {
                                a: idx_padded.clone(),
                                b: mk(NodeKind::Zeros {
                                    shape: ishape,
                                    dtype: DType::I64,
                                    device: idx_padded.device.clone(),
                                })?,
                                dim: i,
                            })?;
                        }
                    }
                    accumulate(
                        a,
                        mk(NodeKind::ScatterAdd {
                            a: zeros_like(a.as_ref())?,
                            dim: *dim,
                            indexes: idx_padded,
                            src: g_padded,
                        }),
                    )?;
                }
                NodeKind::Prod { a, dims, keepdims } => {
                    // d prod / d x_i = prod / x_i; undefined when any factor
                    // is zero (the true adjoint needs the zero-free subproducts)
                    let out_b = expand_reduced(&node.clone(), dims, *keepdims, &a.shape)?;
                    let g_b = expand_reduced(&g, dims, *keepdims, &a.shape)?;
                    accumulate(a, div(mul(g_b, out_b)?, a.clone()))?;
                }
                NodeKind::Pow { a, exp } => {
                    let c = full(*exp, a.dtype, &a.device)?;                    let base = mk(NodeKind::Pow {
                        a: a.clone(),
                        exp: exp - 1.0,
                    })?;
                    accumulate(a, mul(mul(g, c)?, base))?;
                }
                NodeKind::Cast { a, .. } => {
                    if a.dtype.is_float() {
                        accumulate(a, cast(g, a.dtype))?;
                    }
                }
                NodeKind::Sum { a, dims, keepdims } => {
                    accumulate(a, expand_reduced(&g, dims, *keepdims, &a.shape))?;
                }
                NodeKind::Mean { a, dims, keepdims } => {
                    let count: usize = dims.iter().map(|&d| a.shape[d]).product();
                    let scaled = div(g, full(count as f64, a.dtype, &a.device)?)?;
                    accumulate(a, expand_reduced(&scaled, dims, *keepdims, &a.shape))?;
                }
                NodeKind::Max { a, dims, keepdims } | NodeKind::Min { a, dims, keepdims } => {
                    let kept: Vec<usize> = a
                        .shape
                        .iter()
                        .enumerate()
                        .map(|(i, &d)| if dims.contains(&i) { 1 } else { d })
                        .collect();
                    let out_r = if *keepdims {
                        node.clone()
                    } else {
                        reshape(node.clone(), kept.clone())?
                    };
                    let g_r = if *keepdims { g.clone() } else { reshape(g, kept)? };
                    let out_b = broadcast_to(out_r, &a.shape)?;
                    let mask = cast(
                        mk(NodeKind::Eq {
                            a: a.clone(),
                            b: out_b,
                        })?,
                        a.dtype,
                    )?;
                    let denom = broadcast_to(
                        mk(NodeKind::Sum {
                            a: mask.clone(),
                            dims: dims.clone(),
                            keepdims: true,
                        })?,
                        &a.shape,
                    )?;
                    accumulate(a, div(mul(broadcast_to(g_r, &a.shape)?, mask)?, denom))?;
                }
                NodeKind::Reshape { a, .. } => {
                    accumulate(a, reshape(g, a.shape.clone()))?;
                }
                NodeKind::Permute { a, dims } => {
                    let mut inverse = vec![0usize; dims.len()];
                    for (i, &d) in dims.iter().enumerate() {
                        inverse[d] = i;
                    }
                    accumulate(a, mk(NodeKind::Permute { a: g, dims: inverse }))?;
                }
                NodeKind::Slice { a, ranges } => {
                    let mut cur = g;
                    for (dim, &(start, stop, stride)) in ranges.iter().enumerate() {
                        let n = a.shape[dim];
                        if stride != 1 && cur.shape[dim] > 0 {
                            // dilate the cotangent along the sliced dim by
                            // interleaving stride-1 zeros, so it lines up with
                            // the positions the forward pass actually read
                            let len = cur.shape[dim];
                            let mut g_shape = cur.shape.clone();
                            g_shape.insert(dim + 1, 1);
                            let mut z_shape = cur.shape.clone();
                            z_shape.insert(dim + 1, stride - 1);
                            let mut expanded = cur.shape.clone();
                            expanded[dim] = len * stride;
                            let g_r = reshape(cur, g_shape)?;
                            let z = mk(NodeKind::Zeros {
                                shape: z_shape,
                                dtype: g_r.dtype,
                                device: g_r.device.clone(),
                            })?;
                            let cat = mk(NodeKind::Concat {
                                a: g_r,
                                b: z,
                                dim: dim + 1,
                            })?;
                            let mut wide = reshape(cat, expanded)?;
                            let keep = (len - 1) * stride + 1;
                            if keep < len * stride {
                                let trim: Vec<(usize, usize, usize)> = (0..wide.shape.len())
                                    .map(|i| {
                                        if i == dim {
                                            (0, keep, 1)
                                        } else {
                                            (0, wide.shape[i], 1)
                                        }
                                    })
                                    .collect();
                                wide = mk(NodeKind::Slice {
                                    a: wide,
                                    ranges: trim,
                                })?;
                            }
                            cur = wide;
                        }
                        if start > 0 {
                            let mut zshape = cur.shape.clone();
                            zshape[dim] = start;
                            cur = mk(NodeKind::Concat {
                                a: mk(NodeKind::Zeros {
                                    shape: zshape,
                                    dtype: cur.dtype,
                                    device: cur.device.clone(),
                                })?,
                                b: cur.clone(),
                                dim,
                            })?;
                        }
                        if stop < n {
                            let mut zshape = cur.shape.clone();
                            zshape[dim] = n - stop;
                            cur = mk(NodeKind::Concat {
                                a: cur.clone(),
                                b: mk(NodeKind::Zeros {
                                    shape: zshape,
                                    dtype: cur.dtype,
                                    device: cur.device.clone(),
                                })?,
                                dim,
                            })?;
                        }
                    }
                    accumulate(a, Ok(cur))?;
                }
                NodeKind::Concat { a, b, dim } => {
                    let mut offset = 0usize;
                    for input in [a, b] {
                        let len = input.shape[*dim];
                        let ranges: Vec<(usize, usize, usize)> = (0..g.shape.len())
                            .map(|i| {
                                if i == *dim {
                                    (offset, offset + len, 1)
                                } else {
                                    (0, g.shape[i], 1)
                                }
                            })
                            .collect();
                        accumulate(input, mk(NodeKind::Slice { a: g.clone(), ranges }))?;
                        offset += len;
                    }
                }
                NodeKind::BroadcastTo { a, .. } => {
                    accumulate(a, sum_to_shape(&g, &a.shape))?;
                }
                NodeKind::Matmul { a, b } => {
                    let ga = mk(NodeKind::Matmul {
                        a: g.clone(),
                        b: transpose2(b)?,
                    })?;
                    accumulate(a, sum_to_shape(&ga, &a.shape))?;
                    let gb = mk(NodeKind::Matmul {
                        a: transpose2(a)?,
                        b: g.clone(),
                    })?;
                    accumulate(b, sum_to_shape(&gb, &b.shape))?;
                }
                NodeKind::Inverse { a } => {
                    // d inv = -inv^T @ g @ inv^T
                    let t = transpose2(&node)?;
                    accumulate(
                        a,
                        neg(mk(NodeKind::Matmul {
                            a: mk(NodeKind::Matmul {
                                a: t.clone(),
                                b: g,
                            })?,
                            b: t,
                        })?),
                    )?;
                }
                NodeKind::Det { a } => {
                    // d det = det * inv^T; the batch-shaped det and cotangent
                    // are expanded across the matrix dimensions
                    let inv_t = transpose2(&mk(NodeKind::Inverse { a: a.clone() })?)?;
                    let rank = a.shape.len();
                    let dims = vec![rank - 2, rank - 1];
                    let det_b = expand_reduced(&node.clone(), &dims, false, &a.shape)?;
                    let g_b = expand_reduced(&g, &dims, false, &a.shape)?;
                    accumulate(a, mul(g_b, mul(det_b, inv_t)?))?;
                }
                NodeKind::Solve { a, b } => {
                    // out = a^-1 b; g_b = a^-T g; g_a = -g_b @ out^T
                    let inv_t = transpose2(&mk(NodeKind::Inverse { a: a.clone() })?)?;
                    let gb = mk(NodeKind::Matmul {
                        a: inv_t,
                        b: g.clone(),
                    })?;
                    accumulate(b, Ok(gb.clone()))?;
                    accumulate(
                        a,
                        neg(mk(NodeKind::Matmul {
                            a: gb,
                            b: transpose2(&node)?,
                        })?),
                    )?;
                }
                NodeKind::StopGradient { .. } => {}
                NodeKind::CrossEntropy {
                    logits,
                    target,
                    ignore_index,
                } => {
                    let gb = mk(NodeKind::CrossEntropyBackward {
                        logits: logits.clone(),
                        target: target.clone(),
                        ignore_index: *ignore_index,
                    })?;
                    accumulate(logits, mul(g, gb))?;
                }
                NodeKind::CrossEntropyBackward { .. } => {
                    return Err(
                        "grad: cross-entropy backward nodes are not differentiable (no second-order)"
                            .to_string(),
                    );
                }
                NodeKind::RotaryEmbeddingBackward { .. } => {
                    return Err(
                        "grad: rotary backward nodes are not differentiable (no second-order)"
                            .to_string(),
                    );
                }
                NodeKind::LayerNorm { x, weight, bias, eps } => {
                    let bwd = mk(NodeKind::LayerNormBackward {
                        x: x.clone(),
                        weight: weight.clone(),
                        g: g.clone(),
                        eps: *eps,
                    })?;
                    accumulate(x, Ok(bwd.clone()))?;
                    accumulate(
                        weight,
                        mk(NodeKind::LayerNormBackwardOut {
                            of: bwd.clone(),
                            index: 1,
                        }),
                    )?;
                    accumulate(
                        bias,
                        mk(NodeKind::LayerNormBackwardOut {
                            of: bwd,
                            index: 2,
                        }),
                    )?;
                }
                NodeKind::LayerNormBackward { .. } | NodeKind::LayerNormBackwardOut { .. } => {
                    return Err(
                        "grad: layer norm backward nodes are not differentiable (no second-order)"
                            .to_string(),
                    );
                }
                NodeKind::Linear { x, weight, bias } => {
                    // y = x·W + b over the last dim: dx = g·Wᵀ,
                    // dw = xᵀ·g (reduced over leading dims), db = Σ g.
                    let wt = mk(NodeKind::Permute {
                        a: weight.clone(),
                        dims: vec![1, 0],
                    })?;
                    accumulate(x, mk(NodeKind::Matmul { a: g.clone(), b: wt }))?;
                    let rank = x.shape.len();
                    let (k, n) = (weight.shape[0], weight.shape[1]);
                    let rows: usize = x.shape[..rank - 1].iter().product();
                    let x2d = mk(NodeKind::Reshape {
                        a: x.clone(),
                        shape: vec![rows, k],
                    })?;
                    let x2d_t = mk(NodeKind::Permute {
                        a: x2d,
                        dims: vec![1, 0],
                    })?;
                    let g2d = mk(NodeKind::Reshape {
                        a: g.clone(),
                        shape: vec![rows, n],
                    })?;
                    accumulate(
                        weight,
                        mk(NodeKind::Matmul {
                            a: x2d_t,
                            b: g2d.clone(),
                        }),
                    )?;
                    let reduce_dims: Vec<usize> = (0..rank - 1).collect();
                    accumulate(
                        bias,
                        mk(NodeKind::Sum {
                            a: g.clone(),
                            dims: reduce_dims,
                            keepdims: false,
                        }),
                    )?;
                }
                NodeKind::Sdpa {
                    q,
                    k,
                    v,
                    scale,
                    causal,
                } => {
                    let bw = mk(NodeKind::SdpaBackward {
                        q: q.clone(),
                        k: k.clone(),
                        v: v.clone(),
                        g: g.clone(),
                        fwd: node.clone(),
                        scale: *scale,
                        causal: *causal,
                    })?;
                    for (input, index) in [(q, 0u8), (k, 1u8), (v, 2u8)] {
                        let out = mk(NodeKind::SdpaBackwardOut {
                            of: bw.clone(),
                            index,
                        })?;
                        accumulate(input, Ok(out))?;
                    }
                }
                NodeKind::SdpaBackward { .. } | NodeKind::SdpaBackwardOut { .. } => {
                    return Err(
                        "grad: sdpa backward nodes are not differentiable (no second-order)"
                            .to_string(),
                    );
                }
                NodeKind::PositionEmbedding { weight, seq_len } => {
                    // dW: rows 0..seq_len-1 accumulate the cotangent, the
                    // rest stay zero — scatter-add of g into zeros_like(W)
                    // at rows arange(seq_len) (indexes padded to g's shape
                    // per the scatter contract).
                    let t = *seq_len;
                    let e = weight.shape[1];
                    let rows = mk(NodeKind::Arange {
                        start: 0.0,
                        end: t as f64,
                        step: 1.0,
                        dtype: DType::I64,
                        device: weight.device.clone(),
                    })?;
                    let rows = mk(NodeKind::Reshape {
                        a: rows,
                        shape: vec![t, 1],
                    })?;
                    let indexes = mk(NodeKind::BroadcastTo {
                        a: rows,
                        shape: vec![t, e],
                    })?;
                    accumulate(
                        weight,
                        mk(NodeKind::ScatterAdd {
                            a: zeros_like(weight.as_ref())?,
                            dim: 0,
                            indexes,
                            src: g.clone(),
                        }),
                    )?;
                }
                NodeKind::KvAttention { .. } => {
                    return Err(
                        "grad: kv attention is an inference-only node and is not differentiable"
                            .to_string(),
                    );
                }
                NodeKind::RotaryEmbedding { x, seq_len, theta, offset } => {
                    if *offset != PositionOffset::Absolute {
                        return Err(
                            "grad: cursor-offset rotary embedding is not differentiable".to_string(),
                        );
                    }
                    // y = R x with R orthogonal per position: dx = Rᵀ g,
                    // the same rotation with negated angles — a single
                    // semantic node (the fused kernel's sign flip).
                    accumulate(
                        x,
                        mk(NodeKind::RotaryEmbeddingBackward {
                            g: g.clone(),
                            shape: x.shape.clone(),
                            seq_len: *seq_len,
                            theta: *theta,
                        }),
                    )?;
                }
            NodeKind::Conv1d {
                    x,
                    w,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    // dX is the full convolution of the cotangent with the
                    // weight: a transposed convolution whose output_padding
                    // fills the stride remainder; since our forward is a
                    // correlation, the same unflipped weight is the adjoint
                    // kernel.
                    let out_pad =
                        x.shape[2] - ((g.shape[2] - 1) * stride + dilation * (w.shape[2] - 1) + 1
                            - 2 * padding);
                    accumulate(
                        x,
                        mk(NodeKind::ConvTranspose1d {
                            x: g.clone(),
                            w: w.clone(),
                            stride: *stride,
                            padding: *padding,
                            output_padding: out_pad,
                            dilation: *dilation,
                            groups: *groups,
                        }),
                    )?;
                    accumulate(
                        w,
                        mk(NodeKind::Conv1dBackwardW {
                            x: x.clone(),
                            g,
                            kernel: w.shape[2],
                            out_channels: w.shape[0],
                            stride: *stride,
                            padding: *padding,
                            dilation: *dilation,
                            groups: *groups,
                        }),
                    )?;
                }
                NodeKind::Conv2d {
                    x,
                    w,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    let out_pad_h =
                        x.shape[2] - ((g.shape[2] - 1) * stride + dilation * (w.shape[2] - 1) + 1
                            - 2 * padding);
                    let out_pad_w =
                        x.shape[3] - ((g.shape[3] - 1) * stride + dilation * (w.shape[3] - 1) + 1
                            - 2 * padding);
                    // candle's conv_transpose2d takes a single output_padding;
                    // when the per-dim stride remainders differ, compute with
                    // the smaller one and append the missing strip explicitly
                    // — remainder strips beyond the full convolution are
                    // always zeros, so this is exact
                    let min_pad = out_pad_h.min(out_pad_w);
                    let mut dx = mk(NodeKind::ConvTranspose2d {
                        x: g.clone(),
                        w: w.clone(),
                        stride: *stride,
                        padding: *padding,
                        output_padding: min_pad,
                        dilation: *dilation,
                        groups: *groups,
                    })?;
                    for (dim, out_pad) in [(2usize, out_pad_h), (3usize, out_pad_w)] {
                        if out_pad > min_pad {
                            let mut zshape = dx.shape.clone();
                            zshape[dim] = out_pad - min_pad;
                            dx = mk(NodeKind::Concat {
                                a: dx,
                                b: mk(NodeKind::Zeros {
                                    shape: zshape,
                                    dtype: g.dtype,
                                    device: g.device.clone(),
                                })?,
                                dim,
                            })?;
                        }
                    }
                    accumulate(x, Ok(dx))?;
                    accumulate(
                        w,
                        mk(NodeKind::Conv2dBackwardW {
                            x: x.clone(),
                            g,
                            kernel: [w.shape[2], w.shape[3]],
                            out_channels: w.shape[0],
                            stride: *stride,
                            padding: *padding,
                            dilation: *dilation,
                            groups: *groups,
                        }),
                    )?;
                }
                NodeKind::ConvTranspose1d { .. }
                | NodeKind::ConvTranspose2d { .. }
                | NodeKind::Conv1dBackwardW { .. }
                | NodeKind::Conv2dBackwardW { .. } => {
                    return Err(
                        "grad: convolution backward nodes are not differentiable (no second-order)"
                            .to_string(),
                    );
                }
                NodeKind::AdamWStep { .. }
                | NodeKind::AdamWOut { .. }
                | NodeKind::AdamWStepGroup { .. }
                | NodeKind::AdamWGroupOut { .. }
                | NodeKind::SgdStep { .. }
                | NodeKind::SgdOut { .. } => {
                    return Err(
                        "grad: optimizer update nodes are not differentiable".to_string(),
                    );
                }
                NodeKind::FusedElementwise { .. }
                | NodeKind::FusedElementwiseMulti { .. }
                | NodeKind::FusedPick { .. }
                | NodeKind::FusedReduce { .. } => {
                    return Err(
                        "grad: fused elementwise nodes are internal to evaluation".to_string(),
                    );
                }
                NodeKind::Checkpoint { a } => {
                    // Deep-copy the region's interior with fresh node ids and
                    // build the adjoint over the copy: forward intermediates
                    // are recomputed in the backward phase instead of being
                    // retained. Region inputs (nodes also reachable from
                    // outside the checkpoint) and constructor leaves are
                    // shared, so randn draws and constants are not re-run.
                    let outside = outside_set(order, node.id);
                    let region_topo = topo(a);
                    let mut map: HashMap<u64, Arc<Node>> = HashMap::new();
                    let mut shared: HashMap<u64, Arc<Node>> = HashMap::new();
                    for rn in &region_topo {
                        if outside.contains(&rn.id) || node_children(&rn.kind).is_empty() {
                            shared.insert(rn.id, rn.clone());
                            continue;
                        }
                        let kind = remap_children(&rn.kind, &|child: &Arc<Node>| {
                            map.get(&child.id).cloned().unwrap_or_else(|| child.clone())
                        });
                        let copied = Node::new(kind)?;
                        map.insert(rn.id, copied);
                    }
                    if let Some(copied_root) = map.get(&a.id).cloned() {
                        let copied_order: Vec<Arc<Node>> = region_topo
                            .iter()
                            .filter_map(|rn| map.get(&rn.id).cloned())
                            .collect();
                        let copy_ids: HashSet<u64> = map.values().map(|n| n.id).collect();
                        let mut sub: HashMap<u64, Arc<Node>> = HashMap::new();
                        sub.insert(copied_root.id, g.clone());
                        backward(&copied_order, &mut sub)?;
                        for (id, contribution) in sub {
                            if copy_ids.contains(&id) {
                                continue;
                            }
                            if let Some(input) = shared.get(&id) {
                                accumulate(input, Ok(contribution))?;
                            }
                        }
                    } else {
                        accumulate(a, Ok(g.clone()))?;
                    }
                }
                NodeKind::Leaf(_)
                | NodeKind::Input { .. }
                | NodeKind::ScalarInput { .. }
                | NodeKind::FromBytes { .. }
                | NodeKind::Zeros { .. }
                | NodeKind::Ones { .. }
                | NodeKind::Full { .. }
                | NodeKind::Randn { .. }
                | NodeKind::Uniform { .. }
                | NodeKind::Arange { .. }
                | NodeKind::Eye { .. } => {}
                NodeKind::Eq { .. }
                | NodeKind::Gt { .. }
                | NodeKind::Lt { .. }
                | NodeKind::Ge { .. }
                | NodeKind::Le { .. }
                | NodeKind::Argmax { .. }
                | NodeKind::Argmin { .. } => {
                    unreachable!("non-float nodes are filtered above")
                }
            }
        }
        Ok(())
    }
}

#[napi]
pub fn grad(loss: &LazyTensor, wrt: Vec<&LazyTensor>) -> Result<Vec<LazyTensor>> {
    let targets: Vec<Arc<Node>> = wrt.iter().map(|t| t.node.clone()).collect();
    let grads = autodiff::grad(&loss.node, &targets)
        .map_err(|message| Error::new(Status::GenericFailure, message))?;
    Ok(grads.into_iter().map(|node| LazyTensor { node }).collect())
}

#[napi]
pub fn is_device_available(device: String) -> bool {
    get_device(Some(device)).is_ok()
}

async fn run_compute<T: Send + 'static>(
    token: Option<&CancellationToken>,
    compute: impl FnOnce(&AtomicBool) -> Result<T> + Send + 'static,
) -> Result<T> {
    let flag = token.map(|t| t.cancelled.clone());
    let handle = tokio::task::spawn_blocking(move || {
        let cancelled = flag.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        // Metal command buffers, encoders, and their temporaries are
        // autoreleased Objective-C objects. tokio blocking threads have no
        // run loop, so without an explicit pool those objects accumulate for
        // the thread's lifetime and every subsequent Metal driver call gets
        // slower (candle issue #2271). Draining per compute keeps both
        // memory and per-call cost flat across long training loops.
        #[cfg(target_os = "macos")]
        {
            objc2::rc::autoreleasepool(|_| compute(&cancelled))
        }
        #[cfg(not(target_os = "macos"))]
        {
            compute(&cancelled)
        }
    });
    match token {
        Some(token) => {
            if token.cancelled.load(Ordering::Relaxed) {
                return Err(Error::new(
                    Status::Cancelled,
                    "operation aborted".to_string(),
                ));
            }
            let notify = token.notify.clone();
            tokio::select! {
                result = handle => result.map_err(to_join_err)?,
                _ = notify.notified() => Err(Error::new(
                    Status::Cancelled,
                    "operation aborted".to_string(),
                )),
            }
        }
        None => handle.await.map_err(to_join_err)?,
    }
}

// RFC 0007 phase 2: folds maximal single-consumer chains of elementwise
// ops into FusedElementwise nodes, each evaluated as one kernel. Runs on a
// throwaway rewrite at evaluation time, so autodiff, vmap and checkpoint
// always see the unfused graph. EFFECT_TORCH_NO_FUSION disables it.
// Freeze-time optimizer grouping: same-shape AdamW steps collapse
// into AdamWStepGroup nodes of ≤4 params (one fused launch per group,
// 4 lanes + 3 outputs per param + 3 scalars against Metal's
// 31-buffer limit). DISABLED by default: measured at GPT-scale
// (≤64K-element params), the 28-lane mega-kernel's per-element cost
// is ~10× a single-param fused step and outweighs the saved launches
// (12.9ms vs 11.5ms per step). Set EFFECT_TORCH_OPT_GROUPS=1 to
// enable — it should win once parameters are large enough that kernel
// efficiency dominates launch count.
fn group_optimizer_steps(roots: &[Arc<Node>]) -> std::result::Result<Vec<Arc<Node>>, String> {
    if std::env::var_os("EFFECT_TORCH_OPT_GROUPS").is_none() {
        return Ok(roots.to_vec());
    }
    let mut order: Vec<Arc<Node>> = Vec::new();
    let mut visited = HashSet::new();
    fn visit(n: &Arc<Node>, visited: &mut HashSet<u64>, order: &mut Vec<Arc<Node>>) {
        if !visited.insert(n.id) {
            return;
        }
        for c in node_children(&n.kind) {
            visit(&c, visited, order);
        }
        order.push(n.clone());
    }
    for r in roots {
        visit(r, &mut visited, &mut order);
    }
    // Bucket steps by (shape, dtype, hyperparameters).
    type Key = (Vec<usize>, DType, u64, u64, u64, u64);
    let mut buckets: HashMap<Key, Vec<Arc<Node>>> = HashMap::new();
    let mut bucket_order: Vec<Key> = Vec::new();
    for node in &order {
        if let NodeKind::AdamWStep {
            param,
            beta1,
            beta2,
            eps,
            weight_decay,
            ..
        } = &node.kind
        {
            let key: Key = (
                param.shape.clone(),
                param.dtype,
                beta1.to_bits(),
                beta2.to_bits(),
                eps.to_bits(),
                weight_decay.to_bits(),
            );
            buckets.entry(key.clone()).or_insert_with(|| {
                bucket_order.push(key);
                Vec::new()
            }).push(node.clone());
        }
    }
    if bucket_order.iter().all(|k| buckets[k].len() < 2) {
        return Ok(roots.to_vec());
    }
    // step id -> (group node, param index)
    let mut grouped: HashMap<u64, (Arc<Node>, u32)> = HashMap::new();
    for key in &bucket_order {
        let steps = &buckets[key];
        for chunk in steps.chunks(4) {
            if chunk.len() < 2 {
                continue;
            }
            let first = &chunk[0];
            let NodeKind::AdamWStep {
                lr,
                c1,
                c2,
                beta1,
                beta2,
                eps,
                weight_decay,
                ..
            } = &first.kind
            else {
                unreachable!()
            };
            let group = Node::new(NodeKind::AdamWStepGroup {
                params: chunk
                    .iter()
                    .map(|s| match &s.kind {
                        NodeKind::AdamWStep { param, .. } => param.clone(),
                        _ => unreachable!(),
                    })
                    .collect(),
                grads: chunk
                    .iter()
                    .map(|s| match &s.kind {
                        NodeKind::AdamWStep { grad, .. } => grad.clone(),
                        _ => unreachable!(),
                    })
                    .collect(),
                ms: chunk
                    .iter()
                    .map(|s| match &s.kind {
                        NodeKind::AdamWStep { m, .. } => m.clone(),
                        _ => unreachable!(),
                    })
                    .collect(),
                vs: chunk
                    .iter()
                    .map(|s| match &s.kind {
                        NodeKind::AdamWStep { v, .. } => v.clone(),
                        _ => unreachable!(),
                    })
                    .collect(),
                lr: lr.clone(),
                c1: c1.clone(),
                c2: c2.clone(),
                beta1: *beta1,
                beta2: *beta2,
                eps: *eps,
                weight_decay: *weight_decay,
            })?;
            for (i, step) in chunk.iter().enumerate() {
                grouped.insert(step.id, (group.clone(), i as u32));
            }
        }
    }
    // Rebuild bottom-up: outs and direct references remap to the group.
    let mut map: HashMap<u64, Arc<Node>> = HashMap::new();
    for node in &order {
        let rebuilt = match &node.kind {
            NodeKind::AdamWStep { .. } if grouped.contains_key(&node.id) => {
                let (group, param) = &grouped[&node.id];
                NodeKind::AdamWGroupOut {
                    of: group.clone(),
                    param: *param,
                    index: 0,
                }
            }
            NodeKind::AdamWOut { step, index } if grouped.contains_key(&step.id) => {
                let (group, param) = &grouped[&step.id];
                NodeKind::AdamWGroupOut {
                    of: group.clone(),
                    param: *param,
                    index: *index,
                }
            }
            kind => {
                let remap = |child: &Arc<Node>| map.get(&child.id).cloned().unwrap_or_else(|| child.clone());
                remap_children(kind, &remap)
            }
        };
        map.insert(node.id, Node::new(rebuilt)?);
    }
    Ok(roots
        .iter()
        .map(|r| map.get(&r.id).cloned().unwrap_or_else(|| r.clone()))
        .collect())
}

fn fuse_roots(roots: &[Arc<Node>]) -> std::result::Result<Vec<Arc<Node>>, String> {
    let roots = &group_optimizer_steps(roots)?;
    if std::env::var_os("EFFECT_TORCH_NO_FUSION").is_some() {
        return Ok(roots.to_vec());
    }
    use fusion::Expr as E;

    let mut order: Vec<Arc<Node>> = Vec::new();
    let mut visited = HashSet::new();
    fn visit(n: &Arc<Node>, visited: &mut HashSet<u64>, order: &mut Vec<Arc<Node>>) {
        if !visited.insert(n.id) {
            return;
        }
        for c in node_children(&n.kind) {
            visit(&c, visited, order);
        }
        order.push(n.clone());
    }
    for r in roots {
        visit(r, &mut visited, &mut order);
    }
    let mut consumers: HashMap<u64, usize> = HashMap::new();
    for n in &order {
        for c in node_children(&n.kind) {
            *consumers.entry(c.id).or_insert(0) += 1;
        }
    }

    enum OpT {
        Unary(Box<dyn Fn(E) -> E>),
        Binary(Box<dyn Fn(E, E) -> E>),
        // where(cmp(x, y), a, b): the comparison constructor, applied to
        // the comparison's two inputs. Logical children are [x, y, a, b].
        Select(Box<dyn Fn(E, E) -> E>),
        // A reduce terminates the region feeding it (RFC 0007 phase 3a):
        // the chain compiles into the reduce loop instead of
        // materializing. Carries (op, dims, keepdims).
        Reduce(fusion::ReduceOp, Vec<usize>, bool),
    }
    // An input qualifies when it broadcasts into the output shape
    // (right-aligned dims equal or 1; scalars included). Broadcast lanes
    // are read through stride-0 dims inside the region instead of being
    // materialized at the output shape.
    fn input_ok(c: &Node, out: &[usize]) -> bool {
        fusion::broadcast_compatible(&c.shape, out)
    }
    let fusable = |node: &Node| -> Option<OpT> {
        if !fusion::is_supported(&node.device, node.dtype) {
            return None;
        }
        match &node.kind {
            NodeKind::Add { a, b } if input_ok(a, &node.shape) && input_ok(b, &node.shape) => {
                Some(OpT::Binary(Box::new(|a, b| E::Add(Box::new(a), Box::new(b)))))
            }
            NodeKind::Sub { a, b } if input_ok(a, &node.shape) && input_ok(b, &node.shape) => {
                Some(OpT::Binary(Box::new(|a, b| E::Sub(Box::new(a), Box::new(b)))))
            }
            NodeKind::Mul { a, b } if input_ok(a, &node.shape) && input_ok(b, &node.shape) => {
                Some(OpT::Binary(Box::new(|a, b| E::Mul(Box::new(a), Box::new(b)))))
            }
            NodeKind::Div { a, b } if input_ok(a, &node.shape) && input_ok(b, &node.shape) => {
                Some(OpT::Binary(Box::new(|a, b| E::Div(Box::new(a), Box::new(b)))))
            }
            NodeKind::Maximum { a, b } if input_ok(a, &node.shape) && input_ok(b, &node.shape) => {
                Some(OpT::Binary(Box::new(|a, b| E::Max(Box::new(a), Box::new(b)))))
            }
            NodeKind::Minimum { a, b } if input_ok(a, &node.shape) && input_ok(b, &node.shape) => {
                Some(OpT::Binary(Box::new(|a, b| E::Min(Box::new(a), Box::new(b)))))
            }
            NodeKind::Neg { .. } => Some(OpT::Unary(Box::new(|a| E::Neg(Box::new(a))))),
            NodeKind::Sqrt { .. } => Some(OpT::Unary(Box::new(|a| E::Sqrt(Box::new(a))))),
            NodeKind::Exp { .. } => Some(OpT::Unary(Box::new(|a| E::Exp(Box::new(a))))),
            NodeKind::Log { .. } => Some(OpT::Unary(Box::new(|a| E::Log(Box::new(a))))),
            NodeKind::Sin { .. } => Some(OpT::Unary(Box::new(|a| E::Sin(Box::new(a))))),
            NodeKind::Cos { .. } => Some(OpT::Unary(Box::new(|a| E::Cos(Box::new(a))))),
            NodeKind::Relu { .. } => {
                Some(OpT::Unary(Box::new(|a| E::Max(Box::new(a), Box::new(E::cst(0.0))))))
            }
            NodeKind::Tanh { .. } => Some(OpT::Unary(Box::new(|a| E::Tanh(Box::new(a))))),
            NodeKind::Abs { .. } => Some(OpT::Unary(Box::new(|a| E::Abs(Box::new(a))))),
            NodeKind::Erf { .. } => Some(OpT::Unary(Box::new(|a| E::Erf(Box::new(a))))),
            NodeKind::Floor { .. } => Some(OpT::Unary(Box::new(|a| E::Floor(Box::new(a))))),
            NodeKind::Ceil { .. } => Some(OpT::Unary(Box::new(|a| E::Ceil(Box::new(a))))),
            NodeKind::Round { .. } => Some(OpT::Unary(Box::new(|a| E::Round(Box::new(a))))),
            NodeKind::Pow { exp, .. } => {
                let exp = *exp;
                Some(OpT::Unary(Box::new(move |a| fusion::pow_expr(a, exp))))
            }
            // sign(x) = (x > 0) ? 1 : ((x < 0) ? -1 : 0); NaN yields 0,
            // matching candle's CPU and Metal kernels.
            NodeKind::Sign { .. } => Some(OpT::Unary(Box::new(|a| {
                E::Select(
                    Box::new(E::Gt(Box::new(a.clone()), Box::new(E::cst(0.0)))),
                    Box::new(E::cst(1.0)),
                    Box::new(E::Select(
                        Box::new(E::Lt(Box::new(a), Box::new(E::cst(0.0)))),
                        Box::new(E::cst(-1.0)),
                        Box::new(E::cst(0.0)),
                    )),
                )
            }))),
            // A dtype-preserving cast is the identity inside a region.
            // Cross-dtype casts stay region boundaries: lanes are loaded in
            // the region's single dtype.
            NodeKind::Cast { a, dtype } if a.dtype == *dtype => {
                Some(OpT::Unary(Box::new(|a| a)))
            }
            // where(cond, a, b) fuses only when cond is a comparison with
            // no other consumer: the comparison lowers to a float mask
            // feeding a true select, so the u8 mask never materializes.
            // A cond shared with another consumer must stay a real u8
            // tensor, which a float region cannot produce.
            NodeKind::Where { cond, a, b }
                if consumers.get(&cond.id).copied().unwrap_or(0) == 1
                    && input_ok(a, &node.shape)
                    && input_ok(b, &node.shape) =>
            {
                let cmp: Option<Box<dyn Fn(E, E) -> E>> = match &cond.kind {
                    NodeKind::Eq { a: x, b: y }
                        if input_ok(x, &node.shape)
                            && input_ok(y, &node.shape)
                            && fusion::is_supported(&x.device, x.dtype) =>
                    {
                        Some(Box::new(|a, b| E::Eq(Box::new(a), Box::new(b))))
                    }
                    NodeKind::Gt { a: x, b: y }
                        if input_ok(x, &node.shape)
                            && input_ok(y, &node.shape)
                            && fusion::is_supported(&x.device, x.dtype) =>
                    {
                        Some(Box::new(|a, b| E::Gt(Box::new(a), Box::new(b))))
                    }
                    NodeKind::Lt { a: x, b: y }
                        if input_ok(x, &node.shape)
                            && input_ok(y, &node.shape)
                            && fusion::is_supported(&x.device, x.dtype) =>
                    {
                        Some(Box::new(|a, b| E::Lt(Box::new(a), Box::new(b))))
                    }
                    NodeKind::Ge { a: x, b: y }
                        if input_ok(x, &node.shape)
                            && input_ok(y, &node.shape)
                            && fusion::is_supported(&x.device, x.dtype) =>
                    {
                        Some(Box::new(|a, b| E::Ge(Box::new(a), Box::new(b))))
                    }
                    NodeKind::Le { a: x, b: y }
                        if input_ok(x, &node.shape)
                            && input_ok(y, &node.shape)
                            && fusion::is_supported(&x.device, x.dtype) =>
                    {
                        Some(Box::new(|a, b| E::Le(Box::new(a), Box::new(b))))
                    }
                    _ => None,
                };
                cmp.map(OpT::Select)
            }
            NodeKind::Sum { dims, keepdims, .. } if !dims.is_empty() => {
                Some(OpT::Reduce(fusion::ReduceOp::Sum, dims.clone(), *keepdims))
            }
            NodeKind::Mean { dims, keepdims, .. } if !dims.is_empty() => {
                Some(OpT::Reduce(fusion::ReduceOp::Mean, dims.clone(), *keepdims))
            }
            NodeKind::Max { dims, keepdims, .. } if !dims.is_empty() => {
                Some(OpT::Reduce(fusion::ReduceOp::Max, dims.clone(), *keepdims))
            }
            NodeKind::Min { dims, keepdims, .. } if !dims.is_empty() => {
                Some(OpT::Reduce(fusion::ReduceOp::Min, dims.clone(), *keepdims))
            }
            _ => None,
        }
    };
    // A uniform-value child folds into an IR constant when it broadcasts
    // into the output shape (scalar, broadcast-smaller, or output-shaped).
    let const_value = |child: &Node, out_shape: &[usize]| -> Option<f64> {
        match &child.kind {
            NodeKind::Full { shape, value, .. }
                if fusion::broadcast_compatible(shape, out_shape) =>
            {
                Some(*value)
            }
            NodeKind::Zeros { shape, .. } if fusion::broadcast_compatible(shape, out_shape) => {
                Some(0.0)
            }
            _ => None,
        }
    };

    struct Region {
        expr: E,
        inputs: Vec<Arc<Node>>,
        lane_of: HashMap<u64, u32>,
        ops: usize,
    }
    impl Region {
        fn empty() -> Self {
            Region {
                expr: E::cst(0.0),
                inputs: Vec::new(),
                lane_of: HashMap::new(),
                ops: 0,
            }
        }
        fn lane(&mut self, n: &Arc<Node>) -> E {
            if let Some(&k) = self.lane_of.get(&n.id) {
                return E::Input(k);
            }
            let k = self.inputs.len() as u32;
            self.inputs.push(n.clone());
            self.lane_of.insert(n.id, k);
            E::Input(k)
        }
        // Takes ownership of another region's lanes; returns its expr with
        // lane indices remapped into this region's namespace. Lanes the
        // two regions share are reused, not duplicated.
        fn absorb(&mut self, other: Region) -> E {
            let mut remap: HashMap<u32, u32> = HashMap::new();
            for (k, input) in other.inputs.iter().enumerate() {
                let idx = match self.lane_of.get(&input.id) {
                    Some(&existing) => existing,
                    None => {
                        let idx = self.inputs.len() as u32;
                        self.lane_of.insert(input.id, idx);
                        self.inputs.push(input.clone());
                        idx
                    }
                };
                remap.insert(k as u32, idx);
            }
            self.ops += other.ops;
            other.expr.remap_lanes(&remap)
        }
    }

    // Metal kernels accept at most 31 buffer arguments; one slot is the
    // output, so regions are capped at 30 input lanes. Overflow closes the
    // region (it materializes as a fused node) and the op continues with
    // that fused node as a plain input lane.
    const MAX_LANES: usize = 30;
    // Emits a closed region as a FusedElementwise node, or rebuilds the
    // node plainly (children already emitted) when fusion does not apply:
    // single-op regions, lane-less constant regions, regions too large for
    // the Metal kernel's i32 indexing, or a lane that fails to broadcast
    // (unreachable by construction, handled defensively).
    fn emit_region(
        node: &Node,
        region: Region,
        map: &mut HashMap<u64, Arc<Node>>,
    ) -> std::result::Result<(), String> {
        let n: usize = node.shape.iter().product();
        let strides: Option<Vec<Vec<usize>>> = region
            .inputs
            .iter()
            .map(|lane| fusion::lane_strides(&lane.shape, &node.shape))
            .collect();
        let fused = match strides {
            Some(strides)
                if region.ops >= 2
                    && !region.inputs.is_empty()
                    && !(matches!(node.device, dev::Device::Metal)
                        && n > i32::MAX as usize) =>
            {
                Node::new(NodeKind::FusedElementwise {
                    inputs: region.inputs,
                    strides,
                    shape: node.shape.clone(),
                    expr: region.expr,
                })?
            }
            _ => Node::new(remap_children(&node.kind, &|ch| {
                map.get(&ch.id).cloned().unwrap_or_else(|| ch.clone())
            }))?,
        };
        map.insert(node.id, fused);
        Ok(())
    }
    let mut open: HashMap<u64, Region> = HashMap::new();
    let mut map: HashMap<u64, Arc<Node>> = HashMap::new();
    for node in &order {
        let children = node_children(&node.kind);
        let opt = fusable(node);
        // close regions this node will not extend
        for c in &children {
            if open.contains_key(&c.id)
                && (opt.is_none() || consumers.get(&c.id).copied().unwrap_or(0) != 1)
            {
                let region = open.remove(&c.id).unwrap();
                emit_region(c, region, &mut map)?;
            }
        }
        match opt {
            None => {
                let rebuilt = remap_children(&node.kind, &|ch| {
                    map.get(&ch.id).cloned().unwrap_or_else(|| ch.clone())
                });
                map.insert(node.id, Node::new(rebuilt)?);
            }
            Some(OpT::Unary(f)) => {
                let c = children[0].clone();
                let (mut region, expr) = match open.remove(&c.id) {
                    Some(r) => {
                        let e = f(r.expr.clone());
                        (r, e)
                    }
                    None => {
                        let mut r = Region::empty();
                        let l = if let Some(v) = const_value(&c, &node.shape) {
                            E::cst(v)
                        } else {
                            r.lane(&map.get(&c.id).cloned().unwrap_or_else(|| c.clone()))
                        };
                        (r, f(l))
                    }
                };
                region.expr = expr;
                region.ops += 1;
                open.insert(node.id, region);
            }
            Some(OpT::Select(cmpf)) => {
                let (ca, cb, a, b) = match &node.kind {
                    NodeKind::Where { cond, a, b } => match &cond.kind {
                        NodeKind::Eq { a: x, b: y }
                        | NodeKind::Gt { a: x, b: y }
                        | NodeKind::Lt { a: x, b: y }
                        | NodeKind::Ge { a: x, b: y }
                        | NodeKind::Le { a: x, b: y } => (x, y, a, b),
                        _ => unreachable!("fusion: select guard"),
                    },
                    _ => unreachable!("fusion: select guard"),
                };
                // Fold the four logical children into one region. The
                // comparison's inputs never have open regions (the
                // non-fusable comparison closed them when it was visited);
                // the branches may. On lane-cap overflow with nothing left
                // to give, abandon: dropped regions are still covered by
                // the original subgraphs, and the node rebuilds plain.
                let logical: [&Arc<Node>; 4] = [ca, cb, a, b];
                let mut region = Region::empty();
                let mut exprs: Vec<E> = Vec::with_capacity(4);
                let mut abandon = false;
                for child in logical {
                    if abandon {
                        break;
                    }
                    if let Some(r) = open.remove(&child.id) {
                        if region.inputs.len() + r.inputs.len() > MAX_LANES {
                            emit_region(child, r, &mut map)?;
                            let resolved = map.get(&child.id).cloned().unwrap();
                            if region.inputs.len() >= MAX_LANES
                                && !region.lane_of.contains_key(&resolved.id)
                            {
                                abandon = true;
                            } else {
                                exprs.push(region.lane(&resolved));
                            }
                        } else {
                            exprs.push(region.absorb(r));
                        }
                    } else if let Some(v) = const_value(child, &node.shape) {
                        exprs.push(E::cst(v));
                    } else {
                        let resolved =
                            map.get(&child.id).cloned().unwrap_or_else(|| child.clone());
                        if region.inputs.len() >= MAX_LANES
                            && !region.lane_of.contains_key(&resolved.id)
                        {
                            abandon = true;
                        } else {
                            exprs.push(region.lane(&resolved));
                        }
                    }
                }
                if abandon {
                    let rebuilt = remap_children(&node.kind, &|ch| {
                        map.get(&ch.id).cloned().unwrap_or_else(|| ch.clone())
                    });
                    map.insert(node.id, Node::new(rebuilt)?);
                } else {
                    let mut it = exprs.into_iter();
                    let (e0, e1, e2, e3) = (
                        it.next().unwrap(),
                        it.next().unwrap(),
                        it.next().unwrap(),
                        it.next().unwrap(),
                    );
                    region.expr = E::Select(
                        Box::new(cmpf(e0, e1)),
                        Box::new(e2),
                        Box::new(e3),
                    );
                    region.ops += 1;
                    open.insert(node.id, region);
                }
            }
            Some(OpT::Binary(f)) => {
                let a = children[0].clone();
                let b = children[1].clone();
                let mut ra = open.remove(&a.id);
                let mut rb = open.remove(&b.id);
                // A merged region must stay within the lane cap; otherwise
                // the smaller side materializes first.
                if let (Some(r1), Some(r2)) = (&ra, &rb) {
                    if r1.inputs.len() + r2.inputs.len() > MAX_LANES {
                        emit_region(&b, rb.take().unwrap(), &mut map)?;
                    }
                }
                // Extending a region with a brand-new lane must stay within
                // the cap; otherwise that region materializes first and the
                // op reads it back as a plain lane.
                if let Some(r) = &ra {
                    if rb.is_none() {
                        let resolved = map.get(&b.id).map(|n| n.id).unwrap_or(b.id);
                        if const_value(&b, &node.shape).is_none()
                            && !r.lane_of.contains_key(&resolved)
                            && r.inputs.len() >= MAX_LANES
                        {
                            let region = ra.take().unwrap();
                            emit_region(&a, region, &mut map)?;
                        }
                    }
                }
                if let Some(r) = &rb {
                    if ra.is_none() {
                        let resolved = map.get(&a.id).map(|n| n.id).unwrap_or(a.id);
                        if const_value(&a, &node.shape).is_none()
                            && !r.lane_of.contains_key(&resolved)
                            && r.inputs.len() >= MAX_LANES
                        {
                            let region = rb.take().unwrap();
                            emit_region(&b, region, &mut map)?;
                        }
                    }
                }
                let (mut region, expr) = match (ra, rb) {
                    (Some(mut r1), Some(r2)) => {
                        let b_expr = r1.absorb(r2);
                        let e = f(r1.expr.clone(), b_expr);
                        (r1, e)
                    }
                    (Some(mut r), None) => {
                        let l = if let Some(v) = const_value(&b, &node.shape) {
                            E::cst(v)
                        } else {
                            r.lane(&map.get(&b.id).cloned().unwrap_or_else(|| b.clone()))
                        };
                        let e = f(r.expr.clone(), l);
                        (r, e)
                    }
                    (None, Some(mut r)) => {
                        let l = if let Some(v) = const_value(&a, &node.shape) {
                            E::cst(v)
                        } else {
                            r.lane(&map.get(&a.id).cloned().unwrap_or_else(|| a.clone()))
                        };
                        let e = f(l, r.expr.clone());
                        (r, e)
                    }
                    (None, None) => {
                        let mut r = Region::empty();
                        let la = if let Some(v) = const_value(&a, &node.shape) {
                            E::cst(v)
                        } else {
                            r.lane(&map.get(&a.id).cloned().unwrap_or_else(|| a.clone()))
                        };
                        let lb = if let Some(v) = const_value(&b, &node.shape) {
                            E::cst(v)
                        } else {
                            r.lane(&map.get(&b.id).cloned().unwrap_or_else(|| b.clone()))
                        };
                        (r, f(la, lb))
                    }
                };
                region.expr = expr;
                region.ops += 1;
                open.insert(node.id, region);
            }
            Some(OpT::Reduce(op, mut dims, keepdims)) => {
                let a = children[0].clone();
                let in_shape = a.shape.clone();
                dims.sort_unstable();
                dims.dedup();
                let rank = in_shape.len();
                let guards_ok = !dims.is_empty()
                    && dims.iter().all(|&d| d < rank)
                    && dims.iter().map(|&d| in_shape[d]).product::<usize>() > 0
                    && !(matches!(node.device, dev::Device::Metal) && {
                        let in_n: usize = in_shape.iter().product();
                        let out_n: usize =
                            reduced_shape(&in_shape, &dims, keepdims).iter().product();
                        in_n > i32::MAX as usize || out_n > i32::MAX as usize
                    });
                match open.remove(&a.id) {
                    Some(region) if guards_ok && !region.inputs.is_empty() => {
                        let strides: Option<Vec<Vec<usize>>> = region
                            .inputs
                            .iter()
                            .map(|lane| fusion::lane_strides(&lane.shape, &in_shape))
                            .collect();
                        match strides {
                            Some(strides) => {
                                let fused = Node::new(NodeKind::FusedReduce {
                                    inputs: region.inputs,
                                    strides,
                                    in_shape: in_shape.clone(),
                                    expr: region.expr,
                                    op,
                                    dims: dims.clone(),
                                    keepdims,
                                    shape: reduced_shape(&in_shape, &dims, keepdims),
                                })?;
                                map.insert(node.id, fused);
                            }
                            None => {
                                emit_region(&a, region, &mut map)?;
                                let rebuilt = remap_children(&node.kind, &|ch| {
                                    map.get(&ch.id).cloned().unwrap_or_else(|| ch.clone())
                                });
                                map.insert(node.id, Node::new(rebuilt)?);
                            }
                        }
                    }
                    region => {
                        // No single-consumer region to compile into the
                        // reduce loop (or a degenerate reduce): emit the
                        // region plainly if one stayed open and rebuild.
                        if let Some(r) = region {
                            emit_region(&a, r, &mut map)?;
                        }
                        let rebuilt = remap_children(&node.kind, &|ch| {
                            map.get(&ch.id).cloned().unwrap_or_else(|| ch.clone())
                        });
                        map.insert(node.id, Node::new(rebuilt)?);
                    }
                }
            }
        }
    }
    // close regions whose end has no consumer in the graph (graph roots)
    for (id, region) in open.drain() {
        let node = order.iter().find(|n| n.id == id).unwrap();
        emit_region(node, region, &mut map)?;
    }
    Ok(roots
        .iter()
        .map(|r| map.get(&r.id).cloned().unwrap_or_else(|| r.clone()))
        .collect::<Vec<_>>())
        .and_then(|roots| merge_shared_regions(&roots))
}

// RFC 0007 multi-output merge: when a fused region materializes because
// its value has several consumers, and some of those consumers are
// themselves fused regions of one common shape, compile the prefix and
// the continuations as a single multi-output kernel (the prefix's
// expression is inlined into each continuation, so the shared
// intermediate stays in registers instead of round-tripping through a
// buffer). Runs as a post-pass on the rewritten graph: kernel signatures
// are fixed at compile time, so the merge needs the full consumer set,
// which only exists once the region sweep has finished. Repeats to a
// fixpoint; each merge removes at least one FusedElementwise node, so it
// terminates.
fn merge_shared_regions(roots: &[Arc<Node>]) -> std::result::Result<Vec<Arc<Node>>, String> {
    if std::env::var_os("EFFECT_TORCH_NO_MULTI_FUSION").is_some() {
        return Ok(roots.to_vec());
    }
    // Total expression size bound for one merged kernel: keeps register
    // pressure and shader compile time sane on pathological shares.
    const MAX_MERGED_OPS: usize = 512;

    struct Plan {
        // the shared prefix node (a FusedElementwise)
        prefix: u64,
        // fused continuations (FusedElementwise nodes of one common shape)
        group: Vec<u64>,
        // whether the prefix must stay materialized for unfused consumers
        keep_prefix: bool,
        multi: NodeKind,
    }

    fn analyze(roots: &[Arc<Node>]) -> (Vec<Arc<Node>>, HashMap<u64, Vec<u64>>) {
        let mut order: Vec<Arc<Node>> = Vec::new();
        let mut visited = HashSet::new();
        fn visit(n: &Arc<Node>, visited: &mut HashSet<u64>, order: &mut Vec<Arc<Node>>) {
            if !visited.insert(n.id) {
                return;
            }
            for c in node_children(&n.kind) {
                visit(&c, visited, order);
            }
            order.push(n.clone());
        }
        for r in roots {
            visit(r, &mut visited, &mut order);
        }
        let mut consumers: HashMap<u64, Vec<u64>> = HashMap::new();
        for n in &order {
            for c in node_children(&n.kind) {
                consumers.entry(c.id).or_default().push(n.id);
            }
        }
        (order, consumers)
    }

    fn find_merge(order: &[Arc<Node>], consumers: &HashMap<u64, Vec<u64>>) -> Option<Plan> {
        let by_id: HashMap<u64, &Arc<Node>> = order.iter().map(|n| (n.id, n)).collect();
        for node in order {
            let NodeKind::FusedElementwise {
                inputs,
                strides,
                shape,
                expr,
            } = &node.kind
            else {
                continue;
            };
            let Some(cons) = consumers.get(&node.id) else {
                continue;
            };
            if cons.len() < 2 {
                continue;
            }
            // group fused consumers by common output shape, keeping the
            // largest group (encounter order breaks ties deterministically)
            let mut groups: Vec<(Vec<usize>, Vec<&Arc<Node>>)> = Vec::new();
            for cid in cons {
                let c = by_id[cid];
                if let NodeKind::FusedElementwise { .. } = &c.kind {
                    match groups.iter_mut().find(|(s, _)| s == &c.shape) {
                        Some((_, g)) => g.push(c),
                        None => groups.push((c.shape.clone(), vec![c])),
                    }
                }
            }
            groups.sort_by_key(|(_, g)| std::cmp::Reverse(g.len()));
            for (out_shape, group) in groups {
                let group_ids: HashSet<u64> = group.iter().map(|g| g.id).collect();
                // a continuation that reads another group member would need
                // nested inlining; skip the whole group in that case
                if group.iter().any(|g| {
                    node_children(&g.kind).iter().any(|ch| group_ids.contains(&ch.id))
                }) {
                    continue;
                }
                let keep_prefix = cons.iter().any(|cid| !group_ids.contains(cid));
                if group.len() + usize::from(keep_prefix) < 2 {
                    continue;
                }
                // A materialized prefix output is evaluated at the group's
                // coordinates, so it only equals the prefix's own value
                // when the shapes match; a broadcast-smaller prefix is
                // only safe to inline (never to emit).
                if keep_prefix && shape != &out_shape {
                    continue;
                }
                let Some(f_as_lane) = fusion::lane_strides(shape, &out_shape) else {
                    continue;
                };
                let offset = out_shape.len() - shape.len();
                // merged lanes: the prefix's lanes first (strides composed
                // through the prefix's own broadcast into out_shape), then
                // each continuation's extra lanes
                let mut lanes: Vec<Arc<Node>> = Vec::new();
                let mut lane_strides_out: Vec<Vec<usize>> = Vec::new();
                let mut lane_index: HashMap<u64, u32> = HashMap::new();
                for (input, s) in inputs.iter().zip(strides.iter()) {
                    lane_index.insert(input.id, lanes.len() as u32);
                    lanes.push(input.clone());
                    lane_strides_out.push(
                        f_as_lane
                            .iter()
                            .enumerate()
                            .map(|(d, &fs)| if fs == 0 { 0 } else { s[d - offset] })
                            .collect(),
                    );
                }
                let mut exprs: Vec<fusion::Expr> = Vec::new();
                if keep_prefix {
                    exprs.push(expr.clone());
                }
                let mut total_ops: usize = expr.ops();
                let mut ok = true;
                for g in &group {
                    let NodeKind::FusedElementwise {
                        inputs: g_inputs,
                        strides: g_strides,
                        expr: g_expr,
                        ..
                    } = &g.kind
                    else {
                        unreachable!()
                    };
                    let f_lane = g_inputs
                        .iter()
                        .position(|i| i.id == node.id)
                        .expect("fusion: group member must read the prefix")
                        as u32;
                    let mut remap: HashMap<u32, u32> = HashMap::new();
                    for (j, (input, s)) in g_inputs.iter().zip(g_strides.iter()).enumerate() {
                        if input.id == node.id {
                            continue;
                        }
                        let idx = match lane_index.get(&input.id) {
                            Some(&k) => k,
                            None => {
                                let k = lanes.len() as u32;
                                lane_index.insert(input.id, k);
                                lanes.push(input.clone());
                                lane_strides_out.push(s.clone());
                                k
                            }
                        };
                        remap.insert(j as u32, idx);
                    }
                    let merged = g_expr.merge_lane(f_lane, expr, &remap);
                    total_ops += merged.ops();
                    exprs.push(merged);
                }
                // Metal allows 31 buffer arguments per kernel; every lane
                // and every output takes one
                ok &= lanes.len() + exprs.len() <= 31;
                ok &= total_ops <= MAX_MERGED_OPS;
                ok &= !group.iter().any(|g| {
                    g.dtype != node.dtype || !g.device.same_device(&node.device)
                });
                if matches!(node.device, dev::Device::Metal) {
                    ok &= out_shape.iter().product::<usize>() <= i32::MAX as usize;
                }
                if !ok {
                    continue;
                }
                return Some(Plan {
                    prefix: node.id,
                    group: group.iter().map(|g| g.id).collect(),
                    keep_prefix,
                    multi: NodeKind::FusedElementwiseMulti {
                        inputs: lanes,
                        strides: lane_strides_out,
                        shape: out_shape.clone(),
                        exprs,
                    },
                });
            }
        }
        None
    }

    let mut current = roots.to_vec();
    loop {
        let (order, consumers) = analyze(&current);
        if std::env::var_os("EFFECT_TORCH_FUSION_DEBUG").is_some() {
            let mut fe = 0;
            let mut multi = 0;
            let mut pick = 0;
            let mut red = 0;
            for n in &order {
                match &n.kind {
                    NodeKind::FusedElementwise { .. } => fe += 1,
                    NodeKind::FusedElementwiseMulti { .. } => multi += 1,
                    NodeKind::FusedPick { .. } => pick += 1,
                    NodeKind::FusedReduce { .. } => red += 1,
                    _ => {}
                }
            }
            eprintln!(
                "[fusion] analyze: {} nodes (fe {fe}, multi {multi}, pick {pick}, reduce {red})",
                order.len()
            );
        }
        let Some(plan) = find_merge(&order, &consumers) else {
            return Ok(current);
        };
        if std::env::var_os("EFFECT_TORCH_FUSION_DEBUG").is_some() {
            eprintln!(
                "[fusion] multi-merge: prefix {} -> group {:?} (keep {})",
                plan.prefix, plan.group, plan.keep_prefix
            );
        }
        // Remaps a node depth-first through rebuilt subtrees, memoized
        // into `map`. The multi's lanes are remapped this way BEFORE the
        // main rewrite so the multi never references the original lane
        // nodes: keeping the originals would retain their whole
        // ancestry (fused regions included), which the next fixpoint
        // round would see and merge again — duplicating a generation of
        // the subgraph per round. A lane's ancestry can include the
        // prefix itself (a continuation's extra lane descending from
        // it); that path rebuilds the prefix plainly — single-consumer,
        // so it cannot be re-merged and the duplication is bounded.
        fn remap_deep(
            n: &Arc<Node>,
            map: &mut HashMap<u64, Arc<Node>>,
        ) -> std::result::Result<Arc<Node>, String> {
            if let Some(r) = map.get(&n.id) {
                return Ok(r.clone());
            }
            let children = node_children(&n.kind);
            let mut resolved: HashMap<u64, Arc<Node>> = HashMap::with_capacity(children.len());
            for ch in &children {
                resolved.insert(ch.id, remap_deep(ch, map)?);
            }
            let kind = remap_children(&n.kind, &|ch| {
                resolved.get(&ch.id).cloned().unwrap_or_else(|| ch.clone())
            });
            let rebuilt = Node::new(kind)?;
            map.insert(n.id, rebuilt.clone());
            Ok(rebuilt)
        }
        let mut map: HashMap<u64, Arc<Node>> = HashMap::new();
        let multi = {
            let NodeKind::FusedElementwiseMulti {
                inputs,
                strides,
                shape,
                exprs,
            } = &plan.multi
            else {
                unreachable!("fusion: merge plan must build a multi node")
            };
            let mut remapped_inputs = Vec::with_capacity(inputs.len());
            for lane in inputs {
                remapped_inputs.push(remap_deep(lane, &mut map)?);
            }
            Node::new(NodeKind::FusedElementwiseMulti {
                inputs: remapped_inputs,
                strides: strides.clone(),
                shape: shape.clone(),
                exprs: exprs.clone(),
            })?
        };
        let mut pick_index = u8::from(plan.keep_prefix);
        let mut picks: HashMap<u64, Arc<Node>> = HashMap::new();
        if plan.keep_prefix {
            picks.insert(
                plan.prefix,
                Node::new(NodeKind::FusedPick {
                    of: multi.clone(),
                    index: 0,
                })?,
            );
        }
        for gid in &plan.group {
            picks.insert(
                *gid,
                Node::new(NodeKind::FusedPick {
                    of: multi.clone(),
                    index: pick_index,
                })?,
            );
            pick_index += 1;
        }
        for node in &order {
            if map.contains_key(&node.id) {
                continue;
            }
            if let Some(pick) = picks.get(&node.id) {
                map.insert(node.id, pick.clone());
                continue;
            }
            let rebuilt = remap_children(&node.kind, &|ch| {
                map.get(&ch.id).cloned().unwrap_or_else(|| ch.clone())
            });
            map.insert(node.id, Node::new(rebuilt)?);
        }
        current = current
            .iter()
            .map(|r| map.get(&r.id).cloned().unwrap_or_else(|| r.clone()))
            .collect();
    }
}

// The vendored Metal backend keeps one shared "current" command buffer per
// device (candle-metal-kernels `Commands`): concurrent evaluation walks on
// the same device interleave their kernels into each other's buffers, and a
// walk can read back outputs whose producing kernels were committed — and
// only awaited — by another walk. Serialize Metal walks process-wide; GPU
// work serializes on the command queue anyway, so this costs nothing.
static METAL_EVAL_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

fn metal_eval_guard(nodes: &[Arc<Node>]) -> Option<std::sync::MutexGuard<'static, ()>> {
    if nodes.iter().any(|node| matches!(node.device, Device::Metal)) {
        Some(METAL_EVAL_LOCK.lock().unwrap_or_else(|e| e.into_inner()))
    } else {
        None
    }
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
        eprintln!("[walk] fuse_roots {:.1}us ({} nodes)", t0.elapsed().as_micros(), nodes.len());
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
            entries.iter().map(|(k, n)| format!("{k}×{n}")).collect::<Vec<_>>().join(" ")
        );
    }
    run_compute(token, move |cancelled| {
        let t1 = std::time::Instant::now();
        let _guard = metal_eval_guard(&nodes);
        let mut ev = Evaluator::new(&nodes);
        let mut outputs = Vec::with_capacity(nodes.len());
        for node in &nodes {
            let output = eval_node(node, cancelled, &mut ev).map_err(to_napi_err)?;
            outputs.push(NativeTensor::wrap(output));
        }
        // Synchronize once: per-root syncs would fully serialize CPU
        // encoding and GPU execution (readback synchronizes itself).
        // The sync is device-global; one call covers every output.
        if let Some(first) = outputs.first() {
            first.inner.synchronize();
        }
        // Deferred fused-CE status checks (would have split the
        // pipeline mid-walk).
        ev.run_ce_checks().map_err(to_napi_err)?;
        if walk_timing {
            eprintln!("[walk] eval {:.1}us ({} nodes)", t1.elapsed().as_micros(), nodes.len());
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
    static CACHE: LazyLock<Mutex<(u64, HashMap<Vec<u64>, (u64, Vec<Arc<Node>>)>)>> =
        LazyLock::new(|| Mutex::new((0, HashMap::new())));
    let key: Vec<u64> = roots.iter().map(|r| r.id).collect();
    {
        let mut cache = CACHE.lock().map_err(|e| {
            Error::new(Status::GenericFailure, format!("fusion cache lock poisoned: {e}"))
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
        Error::new(Status::GenericFailure, format!("fusion cache lock poisoned: {e}"))
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
    if calls % 4096 == 0 {
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
        NodeKind::Sum { .. } | NodeKind::Mean { .. } | NodeKind::Max { .. } | NodeKind::Min { .. } | NodeKind::Prod { .. } => "Reduce",
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
        NodeKind::FromBytes { .. } | NodeKind::Zeros { .. } | NodeKind::Ones { .. } | NodeKind::Full { .. } | NodeKind::Randn { .. } | NodeKind::Uniform { .. } | NodeKind::Arange { .. } | NodeKind::Eye { .. } => "Const",
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
            .map(|ids| ids.iter().filter(|&&id| self.refcounts[id as usize] == 0).count())
            .sum()
    }
}

enum PoolSlab {
    Native(runtime::cpu::pool::Slab),
    NativeMetal(runtime::metal::run::MetalTensor),
}

impl PoolSlab {
    fn dtype(&self) -> DType {
        match self {
            PoolSlab::Native(s) => s.dtype,
            PoolSlab::NativeMetal(t) => t.dtype,
        }
    }

    fn metal(&self) -> crate::err::Res<&runtime::metal::run::MetalTensor> {
        match self {
            PoolSlab::NativeMetal(t) => Ok(t),
            PoolSlab::Native(_) => Err(err::err_str(
                "kv pool: paged Metal path requires native Metal slabs".to_string(),
            )),
        }
    }

    fn native(&self) -> crate::err::Res<&runtime::cpu::pool::Slab> {
        match self {
            PoolSlab::Native(s) => Ok(s),
            PoolSlab::NativeMetal(_) => Err(err::err_str(
                "kv pool: native CPU path requires native-backed slabs".to_string(),
            )),
        }
    }
}

struct PoolInner {
    // Per layer, flat [max_tokens, kv_heads, head_dim] slabs; block b
    // occupies rows b*block_size..(b+1)*block_size. Slab dtype u8 means
    // int8-quantized storage (RFC 0012 storage tier): rows are
    // symmetric-quantized with a per-(token, head) absmax scale held in
    // `scales` — two slabs per layer (k then v) when the data slabs are
    // u8, empty otherwise. CPU pools hold first-party mutable slabs;
    // Metal pools hold candle buffers until the phase-2 device lands.
    k: Vec<PoolSlab>,
    v: Vec<PoolSlab>,
    scales: Vec<PoolSlab>,
    kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_tokens: usize,
    blocks: Mutex<BlockStore>,
    device: dev::Device,
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
    paged_tables: Mutex<Option<(runtime::metal::run::MetalTensor, runtime::metal::run::MetalTensor)>>,
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
    q: &val::Val,
    k: &val::Val,
    v: &val::Val,
    scale: f64,
    window: Option<usize>,
) -> crate::err::Res<val::Val> {
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
    let rank = dims.len();
    let (t, h, d) = (dims[rank - 2], dims[rank - 3], dims[rank - 1]);
    if q.is_metal() && paged::is_supported(q.as_metal()?, kv.pool.k[layer as usize].dtype(), d) {
        return kv_attention_paged(kv, layer, q.as_metal()?, k.as_metal()?, v.as_metal()?, scale, window, batch, t, h, d);
    }
    if batch == 1 {
        let mut state = kv.slots[0].lock().map_err(|e| {
            err::err_str(format!("kv attention: sequence lock poisoned: {e}"))
        })?;
        return kv_attention_slot(&kv.pool, &mut state, layer, q, k, v, scale, window);
    }
    let mut outs = Vec::with_capacity(batch);
    for (b, slot) in kv.slots.iter().enumerate() {
        let mut state = slot.lock().map_err(|e| {
            err::err_str(format!("kv attention: sequence lock poisoned: {e}"))
        })?;
        let narrow = |x: &val::Val| -> crate::err::Res<val::Val> {
            match x {
                val::Val::Cpu(x) => Ok(val::Val::Cpu(x.view(x.layout.narrow(0, b, 1)).contiguous())),
                val::Val::Metal(x) => {
                    let n = runtime::metal::run::MetalTensor {
                        buffer: x.buffer.clone(),
                        layout: x.layout.narrow(0, b, 1),
                        dtype: x.dtype,
                    };
                    Ok(val::Val::Metal(metal_ops::contiguous(&n)?))
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
        val::Val::Cpu(_) => {
            let refs: Vec<&runtime::cpu::Tensor> = outs.iter().map(|o| o.as_cpu().unwrap()).collect();
            Ok(val::Val::Cpu(runtime::cpu::Tensor::cat(&refs, 0)))
        }
        val::Val::Metal(_) => {
            let refs: Vec<&runtime::metal::run::MetalTensor> = outs.iter().map(|o| o.as_metal().unwrap()).collect();
            metal_ops::cat(refs[0], refs[1], 0).map(val::Val::Metal)
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
) -> crate::err::Res<val::Val> {
    let pool = &kv.pool;
    let layer = layer as usize;
    let mut ctxlens = Vec::with_capacity(batch);
    let mut starts = Vec::with_capacity(batch);
    let mut advance = 0usize;
    for slot in kv.slots.iter() {
        let mut state = slot.lock().map_err(|e| {
            err::err_str(format!("kv attention: sequence lock poisoned: {e}"))
        })?;
        advance = state.advance;
        let (_cursor, needed, start) = kv_prepare(pool, &mut state, layer, window, h, d, t)?;
        ctxlens.push(needed as u32);
        starts.push(start);
    }
    // Tables/ctxlens settle at layer 0; later layers reuse them.
    let mut cache = kv.paged_tables.lock().map_err(|e| {
        err::err_str(format!("kv attention: table cache lock poisoned: {e}"))
    })?;
    if cache.is_none() {
        let mut tables: Vec<u32> = Vec::new();
        let mut max_blocks = 0usize;
        for slot in &kv.slots {
            let state = slot.lock().map_err(|e| {
                err::err_str(format!("kv attention: sequence lock poisoned: {e}"))
            })?;
            max_blocks = max_blocks.max(state.blocks.len());
        }
        for slot in &kv.slots {
            let state = slot.lock().map_err(|e| {
                err::err_str(format!("kv attention: sequence lock poisoned: {e}"))
            })?;
            tables.extend_from_slice(&state.blocks);
            tables.resize(tables.len() + (max_blocks - state.blocks.len()), 0);
        }
        *cache = Some((
            runtime::metal::run::MetalTensor {
                buffer: runtime::metal::device::MetalDevice::get().alloc_with_data_u32(&tables),
                layout: runtime::layout::Layout::contiguous(vec![batch, max_blocks]),
                dtype: crate::runtime::dtype::DType::U32,
            },
            runtime::metal::run::MetalTensor {
                buffer: runtime::metal::device::MetalDevice::get().alloc_with_data_u32(&ctxlens),
                layout: runtime::layout::Layout::contiguous(vec![batch]),
                dtype: crate::runtime::dtype::DType::U32,
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
        let mut state = slot.lock().map_err(|e| {
            err::err_str(format!("kv attention: sequence lock poisoned: {e}"))
        })?;
        kv_evict(pool, &mut state, starts[b]);
    }
    Ok(val::Val::Metal(out))
}

#[allow(clippy::too_many_arguments)]
fn kv_attention_slot(
    pool: &Arc<PoolInner>,
    state: &mut SeqState,
    layer: u32,
    q: &val::Val,
    k: &val::Val,
    v: &val::Val,
    scale: f64,
    window: Option<usize>,
) -> crate::err::Res<val::Val> {
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
        state.blocks[p / pool.block_size] * pool.block_size as u32
            + (p % pool.block_size) as u32
    };
    // Gather the attended context through the block table: positions
    // [start, needed) are real; rows past the real frontier are
    // zero-padded (only pad q rows, whose outputs are discarded, can
    // attend to them). The gather spans [start, cursor+t) so the
    // causal mask aligns for every q row.
    let ctx_rows: Vec<u32> = (start..needed).map(&physical).collect();
    let ctx = full - start;
    if pool.device.is_cpu() {
        let gather_rows = |slab: &PoolSlab, scale: Option<&PoolSlab>| -> crate::err::Res<runtime::cpu::Tensor> {
            let raw = slab.native()?.read_rows_f32(&ctx_rows);
            let real = match scale {
                Some(scale) => {
                    let scales = scale.native()?.read_rows_f32(&ctx_rows);
                    runtime::cpu::pool::dequantize_int8(&raw, &scales, ctx_rows.len(), h, d)
                }
                None => raw,
            };
            let pad = ctx - ctx_rows.len();
            let mut full = real;
            full.resize(ctx * h * d, 0.0);
            let _ = pad;
            // [ctx, H, D] -> [1, H, ctx, D]
            let rows = ctx;
            let mut perm = vec![0f32; rows * h * d];
            for r in 0..rows {
                for hh in 0..h {
                    for dd in 0..d {
                        perm[(hh * rows + r) * d + dd] = full[(r * h + hh) * d + dd];
                    }
                }
            }
            Ok(runtime::cpu::Tensor::from_vec(perm, vec![1, h, ctx, d]))
        };
        let (k_scale, v_scale) = match slab_dtype {
            DType::U8 => (Some(&pool.scales[2 * layer]), Some(&pool.scales[2 * layer + 1])),
            _ => (None, None),
        };
        let kn = gather_rows(&pool.k[layer], k_scale)?;
        let vn = gather_rows(&pool.v[layer], v_scale)?;
        let qn = q.as_cpu()?;
        let out = runtime::cpu::composed::sdpa_forward(qn, &kn, &vn, scale, true);
        kv_evict(pool, state, start);
        return Ok(val::Val::Cpu(out));
    }
    if pool.device.is_metal() {
        // Native Metal fallback: gather the context rows through the
        // block table, dequantize when int8, attend via the composed
        // native sdpa.
        let gather_rows = |slab: &PoolSlab, scale: Option<&PoolSlab>| -> crate::err::Res<runtime::metal::run::MetalTensor> {
            let gathered = runtime::metal::indexing::index_select(
                runtime::metal::device::MetalDevice::get(),
                slab.metal()?,
                0,
                &ctx_rows,
            )?;
            let real = match scale {
                Some(scale) => {
                    let g32 = metal_ops::cast(&gathered, crate::runtime::dtype::DType::F32)?;
                    let off = metal_ops::fill(g32.layout.shape(), 128.0, g32.dtype)?;
                    let centered = metal_ops::binary(&g32, &off, metal_ops::BinOp::Sub)?;
                    let scales = runtime::metal::indexing::index_select(
                        runtime::metal::device::MetalDevice::get(),
                        scale.metal()?,
                        0,
                        &ctx_rows,
                    )?;
                    let scales = runtime::metal::run::MetalTensor {
                        buffer: scales.buffer.clone(),
                        layout: runtime::layout::Layout::contiguous(vec![ctx_rows.len(), h, 1]),
                        dtype: scales.dtype,
                    };
                    metal_ops::binary(&centered, &scales, metal_ops::BinOp::Mul)?
                }
                None => metal_ops::cast(&gathered, crate::runtime::dtype::DType::F32)?,
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
            DType::U8 => (Some(&pool.scales[2 * layer]), Some(&pool.scales[2 * layer + 1])),
            _ => (None, None),
        };
        let kn = gather_rows(&pool.k[layer], k_scale)?;
        let vn = gather_rows(&pool.v[layer], v_scale)?;
        let qn = q.as_metal()?;
        let q32 = metal_ops::to_f32(qn)?;
        let out = composed::sdpa_forward(&q32, &kn, &vn, scale, true)?;
        kv_evict(pool, state, start);
        return Ok(val::Val::Metal(out));
    }
    unreachable!("kv attention slot: no composed fallback remains");
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
) -> crate::err::Res<(usize, usize, usize)> {
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
    k: &val::Val,
    v: &val::Val,
    h: usize,
    d: usize,
) -> crate::err::Res<()> {
    let slab_dtype = pool.k[layer].dtype();
    let cursor = state.cursor;
    let advance = state.advance;
    let needed = cursor + advance;
    // Logical position -> physical row, through the sequence's block
    // table (append-only; entries below `head` are dead and never
    // written).
    let physical = |p: usize| -> u32 {
        state.blocks[p / pool.block_size] * pool.block_size as u32
            + (p % pool.block_size) as u32
    };
    let write_rows: Vec<u32> = (cursor..needed).map(&physical).collect();
    if pool.device.is_cpu() {
        // [1, H, T, D] -> [T, H, D], real rows only, as flat f32.
        let new_rows = |x: &val::Val| -> crate::err::Res<Vec<f32>> {
            let x = x.as_cpu()?;
            let p = x.view(x.layout.permute(&[0, 2, 1, 3])).contiguous();
            let n = p.view(p.layout.narrow(1, 0, advance)).contiguous();
            let n = n.view(runtime::layout::Layout::contiguous(vec![advance, h, d]));
            let n = n.cast(runtime::dtype::DType::F32).contiguous();
            let runtime::cpu::CpuBuffer::F32(v) = &n.buffer else { unreachable!() };
            Ok(v.as_slice().to_vec())
        };
        let k_rows = new_rows(k)?;
        let v_rows = new_rows(v)?;
        if slab_dtype == DType::U8 {
            let (qk, sk) = runtime::cpu::pool::quantize_int8(&k_rows, advance, h, d);
            let (qv, sv) = runtime::cpu::pool::quantize_int8(&v_rows, advance, h, d);
            pool.k[layer].native()?.write_rows_u8(&write_rows, &qk);
            pool.v[layer].native()?.write_rows_u8(&write_rows, &qv);
            pool.scales[2 * layer].native()?.write_rows_f32(&write_rows, &sk);
            pool.scales[2 * layer + 1].native()?.write_rows_f32(&write_rows, &sv);
        } else {
            pool.k[layer].native()?.write_rows_f32(&write_rows, &k_rows);
            pool.v[layer].native()?.write_rows_f32(&write_rows, &v_rows);
        }
        return Ok(());
    }
    if pool.device.is_metal() {
        // Native Metal fallback (paged unsupported): compute rows on
        // device, scatter into slabs with computed physical indices.
        let new_rows = |x: &val::Val| -> crate::err::Res<runtime::metal::run::MetalTensor> {
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
            let quantize = |x: &runtime::metal::run::MetalTensor| -> crate::err::Res<(runtime::metal::run::MetalTensor, runtime::metal::run::MetalTensor)> {
                let abs = metal_ops::unary(x, metal_ops::UnOp::Abs)?;
                let amax = metal_ops::reduce(&abs, &[2], true, crate::fusion::ReduceOp::Max)?;
                let scale = metal_ops::binary(&amax, &metal_ops::fill(amax.layout.shape(), 127.0, amax.dtype)?, metal_ops::BinOp::Div)?;
                let scale = metal_ops::binary(&scale, &metal_ops::fill(scale.layout.shape(), 1e-12, scale.dtype)?, metal_ops::BinOp::Add)?;
                let q = metal_ops::binary(x, &scale, metal_ops::BinOp::Div)?;
                let q = metal_ops::binary(&q, &metal_ops::fill(q.layout.shape(), 128.0, q.dtype)?, metal_ops::BinOp::Add)?;
                let q = metal_ops::unary(&q, metal_ops::UnOp::Round)?;
                let q = metal_ops::cast(&q, crate::runtime::dtype::DType::U8)?;
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
        return Ok(());
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
    let mut cursor_tensor = false;
    let mut geometry: Option<(usize, usize)> = None;
    for node in &order {
        let remap = |child: &Arc<Node>| map.get(&child.id).cloned().unwrap_or_else(|| child.clone());
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
    if layers == 0 {
        return Err(
            "decode: model has no cacheable attention (no causal sdpa node in the forward graph)"
                .to_string(),
        );
    }
    let (kv_heads, head_dim) = geometry.expect("layers > 0 implies geometry");
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
        device: Option<String>,
        dtype: Option<NativeDType>,
    ) -> Result<Self> {
        let device = get_device(device)?;
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
        if layers == 0 || kv_heads == 0 || head_dim == 0 {
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
            if device.is_cpu() {
                k.push(PoolSlab::Native(runtime::cpu::pool::Slab::new(max_tokens, kv_heads * head_dim, dtype)));
                v.push(PoolSlab::Native(runtime::cpu::pool::Slab::new(max_tokens, kv_heads * head_dim, dtype)));
                if dtype == DType::U8 {
                    for _ in 0..2 {
                        scales.push(PoolSlab::Native(runtime::cpu::pool::Slab::new(max_tokens, kv_heads, runtime::dtype::DType::F32)));
                    }
                }
                continue;
            }
            if device.is_metal() {
                let nd = dtype;
                let mslab = |row_width: usize, dt: crate::runtime::dtype::DType| {
                    PoolSlab::NativeMetal(runtime::metal::run::MetalTensor {
                        buffer: runtime::metal::device::MetalDevice::get()
                            .alloc((max_tokens * row_width).max(1), dt),
                        layout: runtime::layout::Layout::contiguous(vec![max_tokens, row_width]),
                        dtype: dt,
                    })
                };
                k.push(mslab(kv_heads * head_dim, nd));
                v.push(mslab(kv_heads * head_dim, nd));
                if dtype == DType::U8 {
                    for _ in 0..2 {
                        scales.push(mslab(kv_heads, crate::runtime::dtype::DType::F32));
                    }
                }
                continue;
            }
            return Err(Error::new(
                Status::InvalidArg,
                "kv pool: device must be Cpu or Metal",
            ));
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
                device,
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
        self.state.lock().map(|state| state.cursor as u32).unwrap_or(0)
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
                format!("kv run: this program is batched (batch {}), use run_batched", self.batch),
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
        if seqs.len() != self.batch as usize {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "kv run: program expects exactly {} sequences, got {}",
                    self.batch,
                    seqs.len()
                ),
            ));
        }
        self.run_inner(inputs, seqs, tokens, token).await
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
            if seqs[..i].iter().any(|other| Arc::ptr_eq(&other.state, &seq.state)) {
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
                format!("program expected {caller_inputs} tensor inputs, got {}", inputs.len()),
            ));
        }
        for (slot, declared) in inner.slots.iter().enumerate() {
            if declared.scalar || (self.cursor_tensor && slot as u32 == self.cursor_slot) {
                continue;
            }
            let input_index = slot
                - inner.slots.iter().take(slot).filter(|s| s.scalar).count()
                - usize::from(self.cursor_tensor && (slot as u32) > self.cursor_slot);
            let got = &inputs[input_index].inner;
            if got.shape() != declared.shape.as_slice()
                || got.dtype() != declared.dtype
                || device_key(&got.device()) != device_key(&declared.device)
            {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!("input slot {slot}: expected {}, got {:?}", declared.signature(), got.shape()),
                ));
            }
        }
        let slots = inner.slots.clone();
        let roots = inner.roots.clone();
        let leaves = inner.leaves.clone();
        let inputs: Vec<val::Val> = inputs.iter().map(|input| input.inner.clone()).collect();
        let kv = Arc::new(KvContext {
            pool: seqs[0].pool.clone(),
            slots: seqs.iter().map(|seq| seq.state.clone()).collect(),
            paged_tables: Mutex::new(None),
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
        run_compute(token, move |cancelled| {
            let _run_guards: Vec<_> = run_locks
                .iter()
                .map(|lock| {
                    lock.lock().map_err(|e| {
                        Error::new(Status::GenericFailure, format!("kv sequence lock poisoned: {e}"))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let _guard = metal_eval_guard(&roots);
            for (i, state) in slot_states.iter().enumerate() {
                state.lock().map_err(|e| {
                    Error::new(Status::GenericFailure, format!("kv sequence lock poisoned: {e}"))
                })?.advance = tokens[i].len();
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
                    let cursor = slot_states[0].lock().map_err(|e| {
                        Error::new(Status::GenericFailure, format!("kv sequence lock poisoned: {e}"))
                    })?.cursor;
                    scalar_binding(cursor as f64, declared.dtype, &declared.device).map_err(to_napi_err)?
                } else if cursor_tensor && slot as u32 == cursor_slot {
                    let mut cursors = Vec::with_capacity(batch);
                    for state in &slot_states {
                        cursors.push(state.lock().map_err(|e| {
                            Error::new(Status::GenericFailure, format!("kv sequence lock poisoned: {e}"))
                        })?.cursor as i64);
                    }
                    if declared.device.is_cpu() {
                        val::Val::Cpu(runtime::cpu::Tensor::from_vec(cursors, vec![batch]))
                    } else {
                        let data: Vec<u8> = cursors.iter().flat_map(|c| c.to_le_bytes()).collect();
                        val::Val::Metal(runtime::metal::run::MetalTensor {
                            buffer: runtime::metal::device::MetalDevice::get().upload_bytes(&data),
                            layout: runtime::layout::Layout::contiguous(vec![batch]),
                            dtype: runtime::dtype::DType::I64,
                        })
                    }
                } else {
                    tensors.next().expect("tensor count checked").clone()
                };
                bindings.insert(slot as u64, binding);
            }
            let by_id: std::collections::HashMap<u64, val::Val> = leaves
                .iter()
                .map(|(id, slot)| {
                    Ok((*id, bindings[&(*slot as u64)].clone()))
                })
                .collect::<crate::err::Res<_>>().map_err(to_napi_err)?;
            // Blocks allocated by a failed run roll back: the cursor did
            // not advance, so every block beyond the pre-run frontier is
            // unreferenced and returns to the pool (a poisoned sequence
            // must not take the pool down with it).
            let frontiers: Vec<usize> = slot_states
                .iter()
                .map(|state| state.lock().map(|s| s.blocks.len()).unwrap_or(0))
                .collect();
            let mut ev = Evaluator::with_kv(&roots, by_id, Some(kv.clone()));
            let mut outputs = Vec::with_capacity(roots.len());
            for node in &roots {
                let output = match eval_node(node, cancelled, &mut ev) {
                    Ok(output) => output,
                    Err(error) => {
                        for (i, state) in slot_states.iter().enumerate() {
                            if let Ok(mut state) = state.lock() {
                                for block in state.blocks.split_off(frontiers[i]) {
                                    kv.pool.unref_block(block);
                                }
                                state.advance = 0;
                            }
                        }
                        return Err(to_napi_err(error));
                    }
                };
                outputs.push(NativeTensor::wrap(output));
            }
            // Synchronize once: per-root syncs would fully serialize
            // CPU encoding and GPU execution. Device-global: one call.
            if let Some(first) = outputs.first() {
                first.inner.synchronize();
            }
            ev.run_ce_checks().map_err(to_napi_err)?;
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
pub fn compile_decode(roots: Vec<&LazyTensor>, window: Option<u32>, batch: Option<u32>) -> Result<DecodeProgram> {
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
    let (nodes, geometry) =
        decode_rewrite(&nodes, window.map(|w| w as usize), batch as usize).map_err(|e| Error::new(Status::GenericFailure, e))?;
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
        },
        cursor_slot: geometry.cursor_slot,
        layers: geometry.layers as u32,
        kv_heads: geometry.kv_heads as u32,
        head_dim: geometry.head_dim as u32,
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

#[derive(Clone)]
struct ProgramSlot {
    scalar: bool,
    shape: Vec<usize>,
    dtype: runtime::dtype::DType,
    device: dev::Device,
}

impl ProgramSlot {
    fn signature(&self) -> String {
        let shape = if self.scalar {
            "scalar".to_string()
        } else {
            self.shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x")
        };
        format!("{}:{}@{}", shape, dtype_name(self.dtype), device_key(&self.device))
    }
}

struct ProgramInner {
    roots: Vec<Arc<Node>>,
    slots: Vec<ProgramSlot>,
    // Placeholder node id -> slot index, collected once at freeze time.
    leaves: Vec<(u64, u32)>,
    signature: String,
}

#[napi]
pub struct CompiledProgram {
    inner: ProgramInner,
}

fn collect_program_slots(
    roots: &[Arc<Node>],
) -> std::result::Result<(Vec<ProgramSlot>, Vec<(u64, u32)>), String> {
    let mut slots: Vec<Option<ProgramSlot>> = Vec::new();
    let mut leaves: Vec<(u64, u32)> = Vec::new();
    let mut visited = HashSet::new();
    let mut stack: Vec<Arc<Node>> = roots.to_vec();
    while let Some(node) = stack.pop() {
        if !visited.insert(node.id) {
            continue;
        }
        let declared = match &node.kind {
            NodeKind::Input {
                slot,
                shape,
                dtype,
                device,
            } => Some((
                *slot,
                ProgramSlot {
                    scalar: false,
                    shape: shape.clone(),
                    dtype: *dtype,
                    device: device.clone(),
                },
            )),
            NodeKind::ScalarInput {
                slot,
                dtype,
                device,
            } => Some((
                *slot,
                ProgramSlot {
                    scalar: true,
                    shape: vec![],
                    dtype: *dtype,
                    device: device.clone(),
                },
            )),
            _ => None,
        };
        if let Some((slot, declared)) = declared {
            leaves.push((node.id, slot));
            let slot = slot as usize;
            if slot >= slots.len() {
                slots.resize_with(slot + 1, || None);
            }
            match &slots[slot] {
                Some(existing) => {
                    if existing.scalar != declared.scalar
                        || existing.shape != declared.shape
                        || existing.dtype != declared.dtype
                        || device_key(&existing.device) != device_key(&declared.device)
                    {
                        return Err(format!(
                            "compile: slot {slot} is used with conflicting signatures ({} vs {})",
                            existing.signature(),
                            declared.signature()
                        ));
                    }
                }
                None => slots[slot] = Some(declared),
            }
        }
        stack.extend(node_children(&node.kind));
    }
    let mut out = Vec::with_capacity(slots.len());
    for (slot, declared) in slots.into_iter().enumerate() {
        out.push(declared.ok_or_else(|| format!("compile: slot {slot} is declared but never used"))?);
    }
    Ok((out, leaves))
}

fn scalar_binding(value: f64, dtype: DType, device: &Device) -> err::Res<val::Val> {
    let nd = dtype;
    if device.is_cpu() {
        Ok(val::Val::Cpu(runtime::cpu::Tensor::full(&[], value, nd)))
    } else {
        Ok(val::Val::Metal(metal_ops::fill(&[], value, nd)?))
    }
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
            let got = &input.inner;
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
        let inputs: Vec<val::Val> = inputs.iter().map(|input| input.inner.clone()).collect();
    run_compute(token, move |cancelled| {
        let _guard = metal_eval_guard(&roots);
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
        let by_id: std::collections::HashMap<u64, val::Val> = leaves
            .iter()
            .map(|(id, slot)| {
                Ok((*id, bindings[&(*slot as u64)].clone()))
            })
            .collect::<crate::err::Res<_>>().map_err(to_napi_err)?;
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
            // Synchronize once: per-root syncs would fully serialize CPU
            // encoding and GPU execution. Consumers that need values on
            // the host synchronize at readback; device-side reuse needs
            // no host round-trip. Device-global: one call.
            if let Some(first) = outputs.first() {
                first.inner.synchronize();
            }
            let t_sync = t1.elapsed() - t_encode;
            ev.run_ce_checks().map_err(to_napi_err)?;
            if walk_timing {
                let (d, s, n) = crate::runtime::metal::device::dispatch_stats_reset();
                eprintln!("[walk] program eval {:.1}us ({} roots) encode {:.1}us sync {:.1}us dispatches {} syncs {} sync_wait {:.1}us", t1.elapsed().as_micros(), roots.len(), t_encode.as_micros(), t_sync.as_micros(), d, s, n as f64 / 1000.0);
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
    let nodes: Vec<Arc<Node>> = tensors.iter().map(|t| t.node.clone()).collect();
    run_compute(token, move |cancelled| {
        let _guard = metal_eval_guard(&nodes);
        let mut ev = Evaluator::new(&nodes);
        let mut map = std::collections::HashMap::with_capacity(names.len());
        for (name, node) in names.iter().zip(nodes.iter()) {
            let output = eval_node(node, cancelled, &mut ev).map_err(to_napi_err)?;
            output.synchronize();
            map.insert(name.clone(), output);
        }
        safetensors::save(&map, &path).map_err(to_napi_err)
    })
    .await
}

// Loads a safetensors file straight into native tensors on the given device;
// JS only receives opaque handles and names. Entries are sorted by name so
// the result is deterministic.
#[napi]
pub async fn load_tensors(
    path: String,
    device: Option<String>,
    token: Option<&CancellationToken>,
) -> Result<(Vec<String>, Vec<NativeTensor>)> {
    let dev = get_device(device)?;
    run_compute(token, move |_cancelled| {
        let mut entries = safetensors::load(&path, &dev).map_err(to_napi_err)?;
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Ok((
            entries.iter().map(|(name, _)| name.clone()).collect(),
            entries
                .into_iter()
                .map(|(_, tensor)| NativeTensor::wrap(tensor))
                .collect(),
        ))
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
mod kv_tests {
    use super::*;

    #[test]
    fn scatter_gather_roundtrip() {
        let slab = runtime::cpu::pool::Slab::new(8, 2 * 3, runtime::dtype::DType::F32);
        let src: Vec<f32> = (0..12).map(|v| v as f32).collect();
        slab.write_rows_f32(&[4, 5], &src);
        let got = slab.read_rows_f32(&[4, 5]);
        assert_eq!(got, src);
    }

    #[test]
    fn kv_attention_matches_sdpa() {
        let device = Device::Cpu;
        let pool = Arc::new(PoolInner {
            k: vec![PoolSlab::Native(runtime::cpu::pool::Slab::new(8, 2 * 4, runtime::dtype::DType::F32))],
            v: vec![PoolSlab::Native(runtime::cpu::pool::Slab::new(8, 2 * 4, runtime::dtype::DType::F32))],
            scales: vec![],
            kv_heads: 2,
            head_dim: 4,
            block_size: 4,
            max_tokens: 8,
            blocks: Mutex::new(BlockStore::new(2)),
            device: device.clone(),
        });
        let state = Arc::new(Mutex::new(SeqState {
            blocks: Vec::new(),
            head: 0,
            cursor: 0,
            advance: 0,
            last_hash: HASH_SEED,
            pending: Vec::new(),
        }));
        let kv = KvContext { pool: pool.clone(), slots: vec![state.clone()], paged_tables: Mutex::new(None) };
        let q = val::Val::Cpu(runtime::cpu::Tensor::from_vec(
            (0..24).map(|v| v as f32).collect(),
            vec![1, 2, 3, 4],
        ));
        let k = val::Val::Cpu(runtime::cpu::Tensor::from_vec(
            (24..48).map(|v| v as f32 * 0.01).collect(),
            vec![1, 2, 3, 4],
        ));
        let v = val::Val::Cpu(runtime::cpu::Tensor::from_vec(
            (48..72).map(|v| v as f32 * 0.01).collect(),
            vec![1, 2, 3, 4],
        ));
        state.lock().unwrap().advance = 3;
        let got = kv_attention(&kv, 0, &q, &k, &v, 0.5, None).unwrap();
        let want = val::Val::Cpu(runtime::cpu::composed::sdpa_forward(
            q.as_cpu().unwrap(),
            k.as_cpu().unwrap(),
            v.as_cpu().unwrap(),
            0.5,
            true,
        ));
        let got = got.to_f32_vec().unwrap();
        let want = want.to_f32_vec().unwrap();
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-6, "{g} vs {w}");
        }
        assert_eq!(state.lock().unwrap().advance, 3);
    }

    #[test]
    fn sdpa_single_token() {
        let q = runtime::cpu::Tensor::from_vec((0..8).map(|v| v as f32).collect(), vec![1, 2, 1, 4]);
        let k = runtime::cpu::Tensor::ones(&[1, 2, 1, 4], runtime::dtype::DType::F32);
        let v = runtime::cpu::Tensor::from_vec((8..16).map(|v| v as f32).collect(), vec![1, 2, 1, 4]);
        let out = runtime::cpu::composed::sdpa_forward(&q, &k, &v, 0.5, true);
        let got = val::Val::Cpu(out).to_f32_vec().unwrap();
        let want = val::Val::Cpu(v).to_f32_vec().unwrap();
        assert_eq!(got, want);
    }

    fn block_store_pool(blocks: usize) -> PoolInner {
        let device = Device::Cpu;
        PoolInner {
            k: vec![],
            v: vec![],
            scales: vec![],
            kv_heads: 1,
            head_dim: 1,
            block_size: 2,
            max_tokens: blocks * 2,
            blocks: Mutex::new(BlockStore::new(blocks)),
            device,
        }
    }

    #[test]
    fn prefix_cache_take_and_reclaim() {
        let pool = block_store_pool(2);
        let a = pool.alloc_block().unwrap();
        let b = pool.alloc_block().unwrap();
        assert!(pool.alloc_block().is_none(), "pool of two is exhausted");
        pool.set_hash(a, 42);
        pool.unref_block(a);
        assert_eq!(pool.cached_count(), 1);
        assert_eq!(pool.available(), 1, "the cached block is reclaimable");
        // A matching prompt takes the resident block.
        assert_eq!(pool.take_block(42), Some(a));
        assert_eq!(pool.cached_count(), 0);
        pool.unref_block(a);
        // A different allocation reclaims the cached block via LRU.
        let c = pool.alloc_block().unwrap();
        assert_eq!(c, a);
        assert_eq!(pool.cached_count(), 0);
        // Unhashed blocks go straight back to the free list.
        pool.unref_block(c);
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
        };
        state.note_tokens(&pool, &[7, 8, 9]);
        let first = chain_hash(HASH_SEED, &[7, 8]);
        let second = chain_hash(first, &[9, 5]);
        let store = pool.blocks.lock().unwrap();
        assert_eq!(store.hashes[state.blocks[0] as usize], Some(first));
        assert_eq!(store.hashes[state.blocks[1] as usize], None, "partial tail");
        drop(store);
        state.cursor = 3; // the first run advanced past its tokens
        state.note_tokens(&pool, &[5]);
        let store = pool.blocks.lock().unwrap();
        assert_eq!(store.hashes[state.blocks[1] as usize], Some(second));
    }
}

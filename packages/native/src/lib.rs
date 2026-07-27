use candle_core::{CpuStorage, DType, Device, Storage, Tensor};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

fn to_napi_err(err: candle_core::Error) -> Error {
    Error::new(Status::GenericFailure, err.to_string())
}

fn to_join_err(err: tokio::task::JoinError) -> Error {
    Error::new(Status::GenericFailure, err.to_string())
}

fn exported_buffers() -> &'static Mutex<HashSet<usize>> {
    static EXPORTED: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();
    EXPORTED.get_or_init(|| Mutex::new(HashSet::new()))
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
        tensor: Tensor,
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
}

impl From<NativeDType> for DType {
    fn from(dtype: NativeDType) -> Self {
        match dtype {
            NativeDType::F32 => DType::F32,
            NativeDType::F64 => DType::F64,
            NativeDType::I64 => DType::I64,
            NativeDType::U8 => DType::U8,
            NativeDType::U32 => DType::U32,
        }
    }
}

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::U8 => "u8",
        DType::U32 => "u32",
        DType::I64 => "i64",
        DType::BF16 => "bf16",
        DType::F16 => "f16",
        DType::F32 => "f32",
        DType::F64 => "f64",
        _ => "unknown",
    }
}

fn get_device(device: Option<String>) -> Result<Device> {
    match device.as_deref().unwrap_or("cpu") {
        "cpu" => Ok(Device::Cpu),
        "metal" => {
            #[cfg(target_os = "macos")]
            {
                static METAL: OnceLock<Device> = OnceLock::new();
                match METAL.get() {
                    Some(device) => Ok(device.clone()),
                    None => {
                        let device = Device::new_metal(0).map_err(to_napi_err)?;
                        Ok(METAL.get_or_init(|| device).clone())
                    }
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                Err(Error::new(
                    Status::InvalidArg,
                    "metal device is only available on macOS builds".to_string(),
                ))
            }
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                static CUDA: OnceLock<Device> = OnceLock::new();
                match CUDA.get() {
                    Some(device) => Ok(device.clone()),
                    None => {
                        let device = Device::new_cuda(0).map_err(to_napi_err)?;
                        Ok(CUDA.get_or_init(|| device).clone())
                    }
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(Error::new(
                    Status::InvalidArg,
                    "cuda support is not compiled in, rebuild with --features cuda".to_string(),
                ))
            }
        }
        other => Err(Error::new(
            Status::InvalidArg,
            format!("unsupported device: {other}"),
        )),
    }
}

#[napi(custom_finalize)]
pub struct NativeTensor {
    pub(crate) inner: Tensor,
    bytes: i64,
}

impl NativeTensor {
    fn wrap(inner: Tensor) -> Self {
        // Buffers cost at least a memory page regardless of the tensor's
        // logical size (Metal allocates 4KB-granular, malloc similar). Without
        // reporting that floor, a stream of tiny tensors looks free to V8 and
        // collection is deferred indefinitely — the backend allocator then
        // can't reuse the pooled buffers (candle's Metal pool requires
        // strong_count == 1) and both memory and per-allocation cost grow
        // without bound.
        let bytes = (inner.elem_count() * inner.dtype().size_in_bytes()).max(4096) as i64;
        Self { inner, bytes }
    }
}

static EXTERNAL_MEMORY_BYTES: AtomicI64 = AtomicI64::new(0);

// V8's GC only sees the small JS handle; report the native buffer size so
// collection is scheduled with knowledge of native memory pressure.
impl ObjectFinalize for NativeTensor {
    fn finalize(self, env: Env) -> Result<()> {
        EXTERNAL_MEMORY_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
        env.adjust_external_memory(-self.bytes)?;
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
    pub fn new() -> Self {
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
        self.inner.dims().iter().map(|&d| d as u32).collect()
    }

    #[napi(getter)]
    pub fn dtype(&self) -> String {
        dtype_name(self.inner.dtype()).to_string()
    }

    #[napi(getter)]
    pub fn device(&self) -> String {
        match self.inner.device() {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(_) => "cuda".to_string(),
            Device::Metal(_) => "metal".to_string(),
        }
    }

    #[napi(getter)]
    pub fn bytes(&self) -> i64 {
        self.bytes
    }

    // Explicitly releases the underlying buffer: the tensor is replaced by an
    // empty CPU scalar, dropping the last strong reference to the real buffer
    // (unless graph leaves still share it) so the backend allocator can reuse
    // it immediately instead of waiting for GC. Using the tensor afterwards
    // fails at evaluation time.
    #[napi]
    pub fn dispose(&mut self, env: Env) -> Result<()> {
        let bytes = std::mem::take(&mut self.bytes);
        if bytes != 0 {
            EXTERNAL_MEMORY_BYTES.fetch_sub(bytes, Ordering::Relaxed);
            env.adjust_external_memory(-bytes)?;
        }
        self.inner = Tensor::zeros(&[], DType::F32, &Device::Cpu).map_err(to_napi_err)?;
        Ok(())
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

fn readback_blocking(inner: &Tensor) -> Result<Readback> {
    let flat = inner.flatten_all().map_err(to_napi_err)?;
    let elem_size = flat.dtype().size_in_bytes();
    let elem_count = flat.elem_count();
    let byte_len = elem_count * elem_size;
    let base: *const u8 = {
        let (storage, _) = flat.storage_and_layout();
        match &*storage {
            Storage::Cpu(CpuStorage::U8(data)) => data.as_ptr() as *const u8,
            Storage::Cpu(CpuStorage::U32(data)) => data.as_ptr() as *const u8,
            Storage::Cpu(CpuStorage::I64(data)) => data.as_ptr() as *const u8,
            Storage::Cpu(CpuStorage::BF16(data)) => data.as_ptr() as *const u8,
            Storage::Cpu(CpuStorage::F16(data)) => data.as_ptr() as *const u8,
            Storage::Cpu(CpuStorage::F32(data)) => data.as_ptr() as *const u8,
            Storage::Cpu(CpuStorage::F64(data)) => data.as_ptr() as *const u8,
            #[cfg(target_os = "macos")]
            Storage::Metal(storage) => storage.buffer().contents() as *const u8,
            _ => std::ptr::null(),
        }
    };
    let offset = flat.storage_and_layout().1.start_offset() * elem_size;
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
        DType::F32 => vec_to_bytes(flat.to_vec1::<f32>().map_err(to_napi_err)?),
        DType::F64 => vec_to_bytes(flat.to_vec1::<f64>().map_err(to_napi_err)?),
        DType::I64 => vec_to_bytes(flat.to_vec1::<i64>().map_err(to_napi_err)?),
        DType::U8 => vec_to_bytes(flat.to_vec1::<u8>().map_err(to_napi_err)?),
        DType::U32 => vec_to_bytes(flat.to_vec1::<u32>().map_err(to_napi_err)?),
        dtype => {
            return Err(Error::new(
                Status::InvalidArg,
                format!("readback not implemented for dtype: {dtype:?}"),
            ))
        }
    };
    Ok(Readback {
        data: ptr,
        byte_len: len,
        hint: FinalizeHint::Owned { ptr, len, cap },
    })
}

enum NodeKind {
    Leaf(Tensor),
    FromBytes {
        data: Vec<u8>,
        shape: Vec<usize>,
        dtype: DType,
        device: Device,
    },
    Zeros {
        shape: Vec<usize>,
        dtype: DType,
        device: Device,
    },
    Ones {
        shape: Vec<usize>,
        dtype: DType,
        device: Device,
    },
    Full {
        shape: Vec<usize>,
        value: f64,
        dtype: DType,
        device: Device,
    },
    Randn {
        shape: Vec<usize>,
        dtype: DType,
        device: Device,
    },
    Uniform {
        lo: f64,
        hi: f64,
        shape: Vec<usize>,
        dtype: DType,
        device: Device,
    },
    Arange {
        start: f64,
        end: f64,
        step: f64,
        dtype: DType,
        device: Device,
    },
    Eye {
        n: usize,
        dtype: DType,
        device: Device,
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
        dtype: DType,
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
    AdamWStep {
        param: Arc<Node>,
        grad: Arc<Node>,
        m: Arc<Node>,
        v: Arc<Node>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        t: f64,
    },
    AdamWOut {
        step: Arc<Node>,
        index: u8,
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
    dtype: DType,
    device: Device,
    kind: NodeKind,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

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
                tensor.dims().to_vec(),
                tensor.dtype(),
                tensor.device().clone(),
            ),
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
                if indexes.dtype != DType::I64 {
                    return Err(format!(
                        "index_select: indexes must be i64, got {:?}",
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
                if indexes.dtype != DType::I64 {
                    return Err(format!(
                        "scatter_add: indexes must be i64, got {:?}",
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
                if indexes.dtype != DType::I64 {
                    return Err(format!("gather: indexes must be i64, got {:?}", indexes.dtype));
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
                if a.shape.len() != 2 || a.shape[0] != a.shape[1] {
                    return Err(format!(
                        "linalg: expected a square rank-2 tensor, got shape {:?}",
                        a.shape
                    ));
                }
                if !a.dtype.is_float() {
                    return Err(format!("linalg: dtype must be floating point, got {:?}", a.dtype));
                }
                if matches!(&kind, NodeKind::Det { .. }) {
                    (vec![], a.dtype, a.device.clone())
                } else {
                    (a.shape.clone(), a.dtype, a.device.clone())
                }
            }
            NodeKind::Solve { a, b } => {
                if a.shape.len() != 2 || a.shape[0] != a.shape[1] {
                    return Err(format!(
                        "solve: expected a square rank-2 coefficient matrix, got shape {:?}",
                        a.shape
                    ));
                }
                if b.shape.len() != 2 || b.shape[0] != a.shape[0] {
                    return Err(format!(
                        "solve: expected a rank-2 right-hand side with {} rows, got shape {:?}",
                        a.shape[0], b.shape
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
                (param.shape.clone(), param.dtype, param.device.clone())
            }
            NodeKind::AdamWOut { step, index } => {
                if *index > 2 {
                    return Err(format!("adamw_out: index must be 0, 1 or 2, got {index}"));
                }
                (step.shape.clone(), step.dtype, step.device.clone())
            }
        };
        Ok(Arc::new(Node {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            shape,
            dtype,
            device,
            kind,
        }))
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
    // Replaces the node with an empty constant, dropping the reference to the
    // subgraph (including any materialized leaf buffer) so it can be freed
    // immediately instead of waiting for GC. Using the tensor afterwards
    // fails at evaluation time.
    #[napi]
    pub fn dispose(&mut self) -> Result<()> {
        self.node = Node::new(NodeKind::Full {
            shape: vec![],
            value: 0.0,
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .map_err(|message| Error::new(Status::GenericFailure, message))?;
        Ok(())
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
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        t: f64,
    ) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::AdamWStep {
            param: self.node.clone(),
            grad: grad.node.clone(),
            m: m.node.clone(),
            v: v.node.clone(),
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t,
        }))
    }

    #[napi]
    pub fn adamw_out(&self, index: u8) -> Result<Self> {
        lazy_ctor!(Node::new(NodeKind::AdamWOut {
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
    cache: std::collections::HashMap<u64, Tensor>,
    // AdamW step id -> (next m, next v); the step node's own value is the
    // updated parameter, stored in the regular cache
    adamw: std::collections::HashMap<u64, [Tensor; 2]>,
    consumers: std::collections::HashMap<u64, usize>,
    roots: HashSet<u64>,
}

impl Evaluator {
    fn new(roots: &[Arc<Node>]) -> Self {
        let mut consumers = std::collections::HashMap::new();
        for root in roots {
            count_consumers(root, &mut consumers);
        }
        Self {
            cache: std::collections::HashMap::new(),
            adamw: std::collections::HashMap::new(),
            consumers,
            roots: roots.iter().map(|root| root.id).collect(),
        }
    }

    fn value(&self, id: u64) -> candle_core::Result<Tensor> {
        self.cache.get(&id).cloned().ok_or_else(|| {
            candle_core::Error::Msg("internal error: child evaluated out of order".to_string())
        })
    }

    fn release_children(&mut self, node: &Arc<Node>) {
        for child in node_children(&node.kind) {
            if let Some(count) = self.consumers.get_mut(&child.id) {
                *count -= 1;
                if *count == 0 && !self.roots.contains(&child.id) {
                    self.cache.remove(&child.id);
                    self.adamw.remove(&child.id);
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
            ..
        } => vec![param.clone(), grad.clone(), m.clone(), v.clone()],
        NodeKind::AdamWOut { step, .. } => vec![step.clone()],
    }
}

// Rebuilds a node kind with its children mapped through `f`. Used to
// deep-copy subgraphs with fresh node ids (checkpoint recompute).
fn remap_children(kind: &NodeKind, f: &dyn Fn(&Arc<Node>) -> Arc<Node>) -> NodeKind {
    match kind {
        NodeKind::Leaf(t) => NodeKind::Leaf(t.clone()),
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
            beta1,
            beta2,
            eps,
            weight_decay,
            t,
        } => NodeKind::AdamWStep {
            param: f(param),
            grad: f(grad),
            m: f(m),
            v: f(v),
            lr: *lr,
            beta1: *beta1,
            beta2: *beta2,
            eps: *eps,
            weight_decay: *weight_decay,
            t: *t,
        },
        NodeKind::AdamWOut { step, index } => NodeKind::AdamWOut {
            step: f(step),
            index: *index,
        },
    }
}

fn eval_broadcast_binary(
    a: &Arc<Node>,
    b: &Arc<Node>,
    ev: &Evaluator,
    f: impl Fn(&Tensor, &Tensor) -> candle_core::Result<Tensor>,
) -> candle_core::Result<Tensor> {
    let a = ev.value(a.id)?;
    let b = ev.value(b.id)?;
    let shape = a.shape().broadcast_shape_binary_op(b.shape(), "cmp")?;
    let a = a.broadcast_as(shape.clone())?;
    let b = b.broadcast_as(shape)?;
    f(&a, &b)
}

macro_rules! linalg_impls {
    ($ty:ty, $inverse:ident, $det:ident) => {
        fn $inverse(t: &Tensor) -> candle_core::Result<Tensor> {
            let n = t.dims()[0];
            let mut a = t.to_vec2::<$ty>()?;
            let mut inv: Vec<Vec<$ty>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| if i == j { 1.0 as $ty } else { 0.0 as $ty })
                        .collect()
                })
                .collect();
            for col in 0..n {
                let mut pivot = col;
                let mut best = a[col][col].abs();
                for (r, row) in a.iter().enumerate().skip(col + 1) {
                    let v = row[col].abs();
                    if v > best {
                        best = v;
                        pivot = r;
                    }
                }
                if best == 0.0 as $ty {
                    return Err(candle_core::Error::Msg(
                        "inverse: matrix is singular".to_string(),
                    ));
                }
                if pivot != col {
                    a.swap(col, pivot);
                    inv.swap(col, pivot);
                }
                let d = a[col][col];
                for j in 0..n {
                    a[col][j] /= d;
                    inv[col][j] /= d;
                }
                for r in 0..n {
                    if r == col {
                        continue;
                    }
                    let f = a[r][col];
                    if f == 0.0 as $ty {
                        continue;
                    }
                    for j in 0..n {
                        a[r][j] -= f * a[col][j];
                        inv[r][j] -= f * inv[col][j];
                    }
                }
            }
            let flat: Vec<$ty> = inv.into_iter().flatten().collect();
            Tensor::from_vec(flat, (n, n), &Device::Cpu)
        }

        fn $det(t: &Tensor) -> candle_core::Result<Tensor> {
            let n = t.dims()[0];
            let mut a = t.to_vec2::<$ty>()?;
            let mut sign = 1.0 as $ty;
            let mut det = 1.0 as $ty;
            for col in 0..n {
                let mut pivot = col;
                let mut best = a[col][col].abs();
                for (r, row) in a.iter().enumerate().skip(col + 1) {
                    let v = row[col].abs();
                    if v > best {
                        best = v;
                        pivot = r;
                    }
                }
                if best == 0.0 as $ty {
                    return Tensor::from_vec(vec![0.0 as $ty], (), &Device::Cpu);
                }
                if pivot != col {
                    a.swap(col, pivot);
                    sign = -sign;
                }
                det *= a[col][col];
                for r in col + 1..n {
                    let f = a[r][col] / a[col][col];
                    for j in col..n {
                        a[r][j] -= f * a[col][j];
                    }
                }
            }
            Tensor::from_vec(vec![sign * det], (), &Device::Cpu)
        }
    };
}

linalg_impls!(f32, inverse_f32, det_f32);
linalg_impls!(f64, inverse_f64, det_f64);

fn cpu_inverse(t: &Tensor) -> candle_core::Result<Tensor> {
    match t.dtype() {
        DType::F32 => inverse_f32(t),
        DType::F64 => inverse_f64(t),
        other => Err(candle_core::Error::Msg(format!(
            "linalg: unsupported dtype {other:?}"
        ))),
    }
}

fn cpu_det(t: &Tensor) -> candle_core::Result<Tensor> {
    match t.dtype() {
        DType::F32 => det_f32(t),
        DType::F64 => det_f64(t),
        other => Err(candle_core::Error::Msg(format!(
            "linalg: unsupported dtype {other:?}"
        ))),
    }
}

fn reduce_dims(
    t: &Tensor,
    dims: &[usize],
    keepdims: bool,    f: impl Fn(&Tensor, usize) -> candle_core::Result<Tensor>,
) -> candle_core::Result<Tensor> {
    let mut out = t.clone();
    for &d in dims.iter().rev() {
        out = f(&out, d)?;
    }
    if keepdims {
        for &d in dims {
            out = out.unsqueeze(d)?;
        }
    }
    Ok(out)
}

// Iterative post-order evaluation: recursion depth is independent of graph
// depth, so chains of arbitrary length evaluate on a fixed stack. Children
// are always computed before their parents, and `eval_uncached` reads their
// values straight from the cache.
fn eval_node(
    root: &Arc<Node>,
    cancelled: &AtomicBool,
    ev: &mut Evaluator,
) -> candle_core::Result<Tensor> {
    let mut stack: Vec<(Arc<Node>, bool)> = vec![(root.clone(), false)];
    while let Some((node, processed)) = stack.pop() {
        if cancelled.load(Ordering::Relaxed) {
            return Err(candle_core::Error::Msg("operation aborted".to_string()));
        }
        if ev.cache.contains_key(&node.id) {
            continue;
        }
        if processed {
            let output = eval_uncached(&node, ev)?;
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

fn eval_uncached(node: &Arc<Node>, ev: &mut Evaluator) -> candle_core::Result<Tensor> {
    let output = match &node.kind {
        NodeKind::Leaf(tensor) => tensor.clone(),
        NodeKind::FromBytes {
            data,
            shape,
            dtype,
            device,
        } => match dtype {
            DType::F32 => {
                let v: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Tensor::from_vec(v, shape.clone(), device)?
            }
            DType::F64 => {
                let v: Vec<f64> = data
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect();
                Tensor::from_vec(v, shape.clone(), device)?
            }
            DType::I64 => {
                let v: Vec<i64> = data
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect();
                Tensor::from_vec(v, shape.clone(), device)?
            }
            DType::U8 => Tensor::from_vec(data.clone(), shape.clone(), device)?,
            DType::U32 => {
                let v: Vec<u32> = data
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Tensor::from_vec(v, shape.clone(), device)?
            }
            dtype => {
                return Err(candle_core::Error::Msg(format!(
                    "fromBytes not supported for dtype {dtype:?}"
                )))
            }
        },
        NodeKind::Zeros {
            shape,
            dtype,
            device,
        } => Tensor::zeros(shape.clone(), *dtype, device)?,
        NodeKind::Ones {
            shape,
            dtype,
            device,
        } => Tensor::ones(shape.clone(), *dtype, device)?,
        NodeKind::Full {
            shape,
            value,
            dtype,
            device,
        } => match dtype {
            DType::F32 => Tensor::full(*value as f32, shape.clone(), device)?,
            DType::F64 => Tensor::full(*value, shape.clone(), device)?,
            DType::I64 => Tensor::full(*value as i64, shape.clone(), device)?,
            DType::U8 => Tensor::full(*value as u8, shape.clone(), device)?,
            DType::U32 => Tensor::full(*value as u32, shape.clone(), device)?,
            dtype => {
                return Err(candle_core::Error::Msg(format!(
                    "full not supported for dtype {dtype:?}"
                )))
            }
        },
        NodeKind::Randn {
            shape,
            dtype,
            device,
        } => Tensor::randn(0f32, 1f32, shape.clone(), device)?.to_dtype(*dtype)?,
        NodeKind::Uniform {
            lo,
            hi,
            shape,
            dtype,
            device,
        } => Tensor::rand(*lo, *hi, shape.clone(), device)?.to_dtype(*dtype)?,
        NodeKind::Arange {
            start,
            end,
            step,
            dtype,
            device,
        } => {
            let n = ((end - start) / step).ceil().max(0.0) as usize;
            let base = Tensor::arange(0u32, n as u32, device)?;
            let scaled = (base * *step)?;
            (scaled + *start)?.to_dtype(*dtype)?
        }
        NodeKind::Eye { n, dtype, device } => {
            let i = Tensor::arange(0u32, *n as u32, device)?.reshape((*n, 1))?;
            let j = Tensor::arange(0u32, *n as u32, device)?.reshape((1, *n))?;
            i.broadcast_eq(&j)?.to_dtype(*dtype)?
        }
        NodeKind::Add { a, b } => {
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            a.broadcast_add(&b)?
        }
        NodeKind::Sub { a, b } => {
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            a.broadcast_sub(&b)?
        }
        NodeKind::Mul { a, b } => {
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            a.broadcast_mul(&b)?
        }
        NodeKind::Div { a, b } => {
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            a.broadcast_div(&b)?
        }
        NodeKind::Eq { a, b } => eval_broadcast_binary(a, b, ev, |a, b| a.eq(b))?,
        NodeKind::Gt { a, b } => eval_broadcast_binary(a, b, ev, |a, b| a.gt(b))?,
        NodeKind::Lt { a, b } => eval_broadcast_binary(a, b, ev, |a, b| a.lt(b))?,
        NodeKind::Ge { a, b } => eval_broadcast_binary(a, b, ev, |a, b| a.ge(b))?,
        NodeKind::Le { a, b } => eval_broadcast_binary(a, b, ev, |a, b| a.le(b))?,
        NodeKind::Maximum { a, b } => eval_broadcast_binary(a, b, ev, |a, b| a.maximum(b))?,
        NodeKind::Minimum { a, b } => eval_broadcast_binary(a, b, ev, |a, b| a.minimum(b))?,
        NodeKind::Neg { a } => ev.value(a.id)?.neg()?,
        NodeKind::Abs { a } => ev.value(a.id)?.abs()?,
        NodeKind::Sqrt { a } => ev.value(a.id)?.sqrt()?,
        NodeKind::Exp { a } => ev.value(a.id)?.exp()?,
        NodeKind::Log { a } => ev.value(a.id)?.log()?,
        NodeKind::Sin { a } => ev.value(a.id)?.sin()?,
        NodeKind::Cos { a } => ev.value(a.id)?.cos()?,
        NodeKind::Tanh { a } => ev.value(a.id)?.tanh()?,
        NodeKind::Relu { a } => {
            let a = ev.value(a.id)?;
            a.maximum(&a.zeros_like()?)?
        }
        NodeKind::Erf { a } => ev.value(a.id)?.erf()?,
        NodeKind::Floor { a } => ev.value(a.id)?.floor()?,
        NodeKind::Ceil { a } => ev.value(a.id)?.ceil()?,
        NodeKind::Round { a } => ev.value(a.id)?.round()?,
        NodeKind::Sign { a } => ev.value(a.id)?.sign()?,
        NodeKind::Where { cond, a, b } => {
            let cond = ev.value(cond.id)?;
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            let shape = cond
                .shape()
                .broadcast_shape_binary_op(a.shape(), "where")?
                .broadcast_shape_binary_op(b.shape(), "where")?;
            let cond = cond.broadcast_as(shape.clone())?;
            let a = a.broadcast_as(shape.clone())?;
            let b = b.broadcast_as(shape)?;
            cond.where_cond(&a, &b)?
        }
        NodeKind::Argmax { a, dim } => ev.value(a.id)?.argmax(*dim)?.to_dtype(DType::I64)?,
        NodeKind::Argmin { a, dim } => ev.value(a.id)?.argmin(*dim)?.to_dtype(DType::I64)?,
        NodeKind::Cumsum { a, dim } => {
            // cumsum is implemented as a matmul internally and chokes on
            // stride-0 broadcast inputs
            ev.value(a.id)?.contiguous()?.cumsum(*dim)?
        }
        NodeKind::IndexSelect { a, dim, indexes } => {
            let a = ev.value(a.id)?;
            let indexes = ev.value(indexes.id)?;
            a.contiguous()?.index_select(&indexes, *dim)?
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
            a.contiguous()?
                .scatter_add(&indexes.contiguous()?, &src.contiguous()?, *dim)?
        }
        NodeKind::Gather { a, dim, indexes } => {
            let a = ev.value(a.id)?;
            let indexes = ev.value(indexes.id)?;
            a.contiguous()?.gather(&indexes.contiguous()?, *dim)?
        }
        NodeKind::Pow { a, exp } => ev.value(a.id)?.powf(*exp)?,
        NodeKind::Cast { a, dtype } => ev.value(a.id)?.to_dtype(*dtype)?,
        NodeKind::Sum { a, dims, keepdims } => {
            let t = ev.value(a.id)?;
            reduce_dims(&t, dims, *keepdims, |t, d| t.sum(d))?
        }
        NodeKind::Mean { a, dims, keepdims } => {
            let t = ev.value(a.id)?;
            reduce_dims(&t, dims, *keepdims, |t, d| t.mean(d))?
        }
        NodeKind::Max { a, dims, keepdims } => {
            let t = ev.value(a.id)?;
            reduce_dims(&t, dims, *keepdims, |t, d| t.max(d))?
        }
        NodeKind::Min { a, dims, keepdims } => {
            let t = ev.value(a.id)?;
            reduce_dims(&t, dims, *keepdims, |t, d| t.min(d))?
        }
        NodeKind::Prod { a, dims, keepdims } => {
            // no product kernel in candle: fold narrow+mul per reduced dim,
            // keeping reduced dims as size 1 so later indices stay valid
            let mut t = ev.value(a.id)?;
            for &d in dims {
                let n = t.dims()[d];
                if n == 0 {
                    let mut shape = t.dims().to_vec();
                    shape[d] = 1;
                    t = Tensor::ones(shape, t.dtype(), t.device())?;
                    continue;
                }
                let mut acc = t.narrow(d, 0, 1)?;
                for i in 1..n {
                    acc = acc.mul(&t.narrow(d, i, 1)?)?;
                }
                t = acc;
            }
            if !keepdims {
                let shape: Vec<usize> = t
                    .dims()
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !dims.contains(i))
                    .map(|(_, &d)| d)
                    .collect();
                t = t.reshape(shape)?;
            }
            t
        }
        NodeKind::Reshape { a, shape } => ev.value(a.id)?.reshape(shape.clone())?,
        NodeKind::Permute { a, dims } => ev.value(a.id)?.permute(dims.clone())?,
        NodeKind::Slice { a, ranges } => {
            let mut t = ev.value(a.id)?;
            for (dim, &(start, stop, stride)) in ranges.iter().enumerate() {
                let len = stop.saturating_sub(start).div_ceil(stride);
                if len == 0 {
                    t = t.narrow(dim, 0, 0)?;
                    continue;
                }
                t = t.narrow(dim, start, (len - 1) * stride + 1)?;
                if stride > 1 {
                    let idx: Vec<u32> = (0..len as u32).map(|i| i * stride as u32).collect();
                    let idx = Tensor::from_vec(idx, len, t.device())?;
                    t = t.contiguous()?.index_select(&idx, dim)?;
                }
            }
            t
        }
        NodeKind::Concat { a, b, dim } => {
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            Tensor::cat(&[&a, &b], *dim)?
        }
        NodeKind::BroadcastTo { a, shape } => {
            ev.value(a.id)?.broadcast_as(shape.clone())?
        }
        NodeKind::Matmul { a, b } => {
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            // candle's matmul requires contiguous operands; permuted or
            // broadcast layouts (common in backward graphs) must be
            // materialized first.
            let a = if a.is_contiguous() { a } else { a.contiguous()? };
            let b = if b.is_contiguous() { b } else { b.contiguous()? };
            a.broadcast_matmul(&b)?
        }
        NodeKind::Inverse { a } => {
            // linalg is CPU-only: round-trip to the host
            let t = ev.value(a.id)?;
            let cpu = t.to_device(&Device::Cpu)?;
            cpu_inverse(&cpu)?.to_device(t.device())?
        }
        NodeKind::Det { a } => {
            let t = ev.value(a.id)?;
            let cpu = t.to_device(&Device::Cpu)?;
            cpu_det(&cpu)?.to_device(t.device())?
        }
        NodeKind::Solve { a, b } => {
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            let a_cpu = a.to_device(&Device::Cpu)?;
            let b_cpu = b.to_device(&Device::Cpu)?;
            cpu_inverse(&a_cpu)?.matmul(&b_cpu)?.to_device(a.device())?
        }
        NodeKind::StopGradient { a } => ev.value(a.id)?,
        NodeKind::Checkpoint { a } => ev.value(a.id)?,
        NodeKind::AdamWStep {
            param,
            grad,
            m,
            v,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t,
        } => {
            let p = ev.value(param.id)?;
            let g = ev.value(grad.id)?;
            let m_t = ev.value(m.id)?;
            let v_t = ev.value(v.id)?;
            let one_minus_beta1 = 1.0 - *beta1;
            let one_minus_beta2 = 1.0 - *beta2;
            let next_m = ((m_t * *beta1)? + (&g * one_minus_beta1)?)?;
            let next_v = ((v_t * *beta2)? + ((&g * &g)? * one_minus_beta2)?)?;
            let m_hat = (&next_m / (1.0 - beta1.powf(*t)))?;
            let v_hat = (&next_v / (1.0 - beta2.powf(*t)))?;
            let adjusted = ((m_hat / (v_hat.sqrt()? + *eps)?)? * *lr)?;
            let next_p = if *weight_decay == 0.0 {
                (p - adjusted)?
            } else {
                ((p * (1.0 - lr * weight_decay))? - adjusted)?
            };
            ev.adamw.insert(node.id, [next_m, next_v]);
            next_p
        }
        NodeKind::AdamWOut { step, index } => {
            // the step is evaluated before its projections; make sure of it
            let _ = ev.value(step.id)?;
            let outputs = ev.adamw.get(&step.id).ok_or_else(|| {
                candle_core::Error::Msg("adamw_out: step has no stored moments".to_string())
            })?;
            match index {
                0 => ev.value(step.id)?,
                1 => outputs[0].clone(),
                _ => outputs[1].clone(),
            }
        }
    };
    Ok(output)
}

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
            NodeKind::Gather { .. } | NodeKind::ScatterAdd { .. } => Err(
                "vmap: gather and scatterAdd are not supported under vmap".to_string(),
            ),
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
            let kind = vmap_rebuild(
                &node,
                dim,
                batch,
                &|child: &Arc<Node>| map.get(&child.id).cloned().unwrap_or_else(|| child.clone()),
                &|id: u64| map.contains_key(&id),
            )?;
            let rebuilt = mk(kind)?;
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
                    // d det = det * inv^T
                    let inv_t = transpose2(&mk(NodeKind::Inverse { a: a.clone() })?)?;
                    accumulate(a, mul(g, mul(node.clone(), inv_t)?))?;
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
                NodeKind::AdamWStep { .. } | NodeKind::AdamWOut { .. } => {
                    return Err(
                        "grad: optimizer update nodes are not differentiable".to_string(),
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

#[napi]
pub async fn eval_lazy(
    tensors: Vec<&LazyTensor>,
    token: Option<&CancellationToken>,
) -> Result<Vec<NativeTensor>> {
    let nodes: Vec<Arc<Node>> = tensors.iter().map(|t| t.node.clone()).collect();
    run_compute(token, move |cancelled| {
        let mut ev = Evaluator::new(&nodes);
        let mut outputs = Vec::with_capacity(nodes.len());
        for node in &nodes {
            let output = eval_node(node, cancelled, &mut ev).map_err(to_napi_err)?;
            output.device().synchronize().map_err(to_napi_err)?;
            outputs.push(NativeTensor::wrap(output));
        }
        Ok(outputs)
    })
    .await
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
        let mut ev = Evaluator::new(&nodes);
        let mut map = std::collections::HashMap::with_capacity(names.len());
        for (name, node) in names.iter().zip(nodes.iter()) {
            let output = eval_node(node, cancelled, &mut ev).map_err(to_napi_err)?;
            output.device().synchronize().map_err(to_napi_err)?;
            map.insert(name.clone(), output);
        }
        candle_core::safetensors::save(&map, &path).map_err(to_napi_err)
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
        let mut entries: Vec<(String, Tensor)> = candle_core::safetensors::load(&path, &dev)
            .map_err(to_napi_err)?
            .into_iter()
            .collect();
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

// Positive half of the external-memory accounting (the negative half runs in
// NativeTensor's finalizer). A sync call so it executes on the main thread:
// async napi functions run their body on the tokio runtime, where touching
// the env is not allowed.
#[napi]
pub fn report_external_memory(env: Env, bytes: i64) -> Result<()> {
    EXTERNAL_MEMORY_BYTES.fetch_add(bytes, Ordering::Relaxed);
    env.adjust_external_memory(bytes)?;
    Ok(())
}

// Native bytes currently retained by JS-reachable tensors. Exposed so tests
// can verify the accounting returns to baseline: Node's
// `process.memoryUsage().external` does not reflect
// `napi_adjust_external_memory`.
#[napi]
pub fn external_memory_bytes() -> i64 {
    EXTERNAL_MEMORY_BYTES.load(Ordering::Relaxed)
}


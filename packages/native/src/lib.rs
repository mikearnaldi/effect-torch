use candle_core::{CpuStorage, DType, Device, Storage, Tensor};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
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

#[napi]
pub struct NativeTensor {
    pub(crate) inner: Tensor,
}

impl NativeTensor {
    fn wrap(inner: Tensor) -> Self {
        Self { inner }
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

enum LazyNode {
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
        a: Arc<LazyNode>,
        b: Arc<LazyNode>,
    },
    Sub {
        a: Arc<LazyNode>,
        b: Arc<LazyNode>,
    },
    Mul {
        a: Arc<LazyNode>,
        b: Arc<LazyNode>,
    },
    Div {
        a: Arc<LazyNode>,
        b: Arc<LazyNode>,
    },
    Eq {
        a: Arc<LazyNode>,
        b: Arc<LazyNode>,
    },
    Gt {
        a: Arc<LazyNode>,
        b: Arc<LazyNode>,
    },
    Lt {
        a: Arc<LazyNode>,
        b: Arc<LazyNode>,
    },
    Ge {
        a: Arc<LazyNode>,
        b: Arc<LazyNode>,
    },
    Le {
        a: Arc<LazyNode>,
        b: Arc<LazyNode>,
    },
    Neg {
        a: Arc<LazyNode>,
    },
    Abs {
        a: Arc<LazyNode>,
    },
    Sqrt {
        a: Arc<LazyNode>,
    },
    Exp {
        a: Arc<LazyNode>,
    },
    Log {
        a: Arc<LazyNode>,
    },
    Sin {
        a: Arc<LazyNode>,
    },
    Cos {
        a: Arc<LazyNode>,
    },
    Pow {
        a: Arc<LazyNode>,
        exp: f64,
    },
    Cast {
        a: Arc<LazyNode>,
        dtype: DType,
    },
    Sum {
        a: Arc<LazyNode>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    Mean {
        a: Arc<LazyNode>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    Max {
        a: Arc<LazyNode>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    Min {
        a: Arc<LazyNode>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    Reshape {
        a: Arc<LazyNode>,
        shape: Vec<usize>,
    },
    Permute {
        a: Arc<LazyNode>,
        dims: Vec<usize>,
    },
    Slice {
        a: Arc<LazyNode>,
        ranges: Vec<(usize, usize, usize)>,
    },
    Concat {
        a: Arc<LazyNode>,
        b: Arc<LazyNode>,
        dim: usize,
    },
    BroadcastTo {
        a: Arc<LazyNode>,
        shape: Vec<usize>,
    },
    Matmul {
        a: Arc<LazyNode>,
        b: Arc<LazyNode>,
    },
}

#[napi]
pub struct LazyTensor {
    node: Arc<LazyNode>,
}

#[napi]
impl LazyTensor {
    #[napi(factory)]
    pub fn zeros(
        shape: Vec<u32>,
        dtype: Option<NativeDType>,
        device: Option<String>,
    ) -> Result<Self> {
        Ok(Self {
            node: Arc::new(LazyNode::Zeros {
                shape: shape.iter().map(|&d| d as usize).collect(),
                dtype: dtype.unwrap_or(NativeDType::F32).into(),
                device: get_device(device)?,
            }),
        })
    }

    #[napi(factory)]
    pub fn ones(
        shape: Vec<u32>,
        dtype: Option<NativeDType>,
        device: Option<String>,
    ) -> Result<Self> {
        Ok(Self {
            node: Arc::new(LazyNode::Ones {
                shape: shape.iter().map(|&d| d as usize).collect(),
                dtype: dtype.unwrap_or(NativeDType::F32).into(),
                device: get_device(device)?,
            }),
        })
    }

    #[napi(factory)]
    pub fn full(
        shape: Vec<u32>,
        value: f64,
        dtype: Option<NativeDType>,
        device: Option<String>,
    ) -> Result<Self> {
        Ok(Self {
            node: Arc::new(LazyNode::Full {
                shape: shape.iter().map(|&d| d as usize).collect(),
                value,
                dtype: dtype.unwrap_or(NativeDType::F32).into(),
                device: get_device(device)?,
            }),
        })
    }

    #[napi(factory)]
    pub fn randn(
        shape: Vec<u32>,
        dtype: Option<NativeDType>,
        device: Option<String>,
    ) -> Result<Self> {
        Ok(Self {
            node: Arc::new(LazyNode::Randn {
                shape: shape.iter().map(|&d| d as usize).collect(),
                dtype: dtype.unwrap_or(NativeDType::F32).into(),
                device: get_device(device)?,
            }),
        })
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
        Ok(Self {
            node: Arc::new(LazyNode::Arange {
                start,
                end,
                step,
                dtype: dtype.unwrap_or(NativeDType::F32).into(),
                device: get_device(device)?,
            }),
        })
    }

    #[napi(factory)]
    pub fn eye(n: u32, dtype: Option<NativeDType>, device: Option<String>) -> Result<Self> {
        Ok(Self {
            node: Arc::new(LazyNode::Eye {
                n: n as usize,
                dtype: dtype.unwrap_or(NativeDType::F32).into(),
                device: get_device(device)?,
            }),
        })
    }

    #[napi(factory)]
    pub fn from_bytes(
        data: Uint8Array,
        shape: Vec<u32>,
        dtype: Option<NativeDType>,
        device: Option<String>,
    ) -> Result<Self> {
        Ok(Self {
            node: Arc::new(LazyNode::FromBytes {
                data: data.to_vec(),
                shape: shape.iter().map(|&d| d as usize).collect(),
                dtype: dtype.unwrap_or(NativeDType::F32).into(),
                device: get_device(device)?,
            }),
        })
    }

    #[napi(factory)]
    pub fn from_materialized(tensor: &NativeTensor) -> Self {
        Self {
            node: Arc::new(LazyNode::Leaf(tensor.inner.clone())),
        }
    }

    #[napi]
    pub fn add(&self, other: &LazyTensor) -> Self {
        Self {
            node: Arc::new(LazyNode::Add {
                a: self.node.clone(),
                b: other.node.clone(),
            }),
        }
    }

    #[napi]
    pub fn sub(&self, other: &LazyTensor) -> Self {
        Self {
            node: Arc::new(LazyNode::Sub {
                a: self.node.clone(),
                b: other.node.clone(),
            }),
        }
    }

    #[napi]
    pub fn mul(&self, other: &LazyTensor) -> Self {
        Self {
            node: Arc::new(LazyNode::Mul {
                a: self.node.clone(),
                b: other.node.clone(),
            }),
        }
    }

    #[napi]
    pub fn div(&self, other: &LazyTensor) -> Self {
        Self {
            node: Arc::new(LazyNode::Div {
                a: self.node.clone(),
                b: other.node.clone(),
            }),
        }
    }

    #[napi]
    pub fn eq(&self, other: &LazyTensor) -> Self {
        Self {
            node: Arc::new(LazyNode::Eq {
                a: self.node.clone(),
                b: other.node.clone(),
            }),
        }
    }

    #[napi]
    pub fn gt(&self, other: &LazyTensor) -> Self {
        Self {
            node: Arc::new(LazyNode::Gt {
                a: self.node.clone(),
                b: other.node.clone(),
            }),
        }
    }

    #[napi]
    pub fn lt(&self, other: &LazyTensor) -> Self {
        Self {
            node: Arc::new(LazyNode::Lt {
                a: self.node.clone(),
                b: other.node.clone(),
            }),
        }
    }

    #[napi]
    pub fn ge(&self, other: &LazyTensor) -> Self {
        Self {
            node: Arc::new(LazyNode::Ge {
                a: self.node.clone(),
                b: other.node.clone(),
            }),
        }
    }

    #[napi]
    pub fn le(&self, other: &LazyTensor) -> Self {
        Self {
            node: Arc::new(LazyNode::Le {
                a: self.node.clone(),
                b: other.node.clone(),
            }),
        }
    }

    #[napi]
    pub fn matmul(&self, other: &LazyTensor) -> Self {
        Self {
            node: Arc::new(LazyNode::Matmul {
                a: self.node.clone(),
                b: other.node.clone(),
            }),
        }
    }

    #[napi]
    pub fn neg(&self) -> Self {
        Self {
            node: Arc::new(LazyNode::Neg {
                a: self.node.clone(),
            }),
        }
    }

    #[napi]
    pub fn abs(&self) -> Self {
        Self {
            node: Arc::new(LazyNode::Abs {
                a: self.node.clone(),
            }),
        }
    }

    #[napi]
    pub fn sqrt(&self) -> Self {
        Self {
            node: Arc::new(LazyNode::Sqrt {
                a: self.node.clone(),
            }),
        }
    }

    #[napi]
    pub fn exp(&self) -> Self {
        Self {
            node: Arc::new(LazyNode::Exp {
                a: self.node.clone(),
            }),
        }
    }

    #[napi]
    pub fn log(&self) -> Self {
        Self {
            node: Arc::new(LazyNode::Log {
                a: self.node.clone(),
            }),
        }
    }

    #[napi]
    pub fn sin(&self) -> Self {
        Self {
            node: Arc::new(LazyNode::Sin {
                a: self.node.clone(),
            }),
        }
    }

    #[napi]
    pub fn cos(&self) -> Self {
        Self {
            node: Arc::new(LazyNode::Cos {
                a: self.node.clone(),
            }),
        }
    }

    #[napi]
    pub fn pow(&self, exp: f64) -> Self {
        Self {
            node: Arc::new(LazyNode::Pow {
                a: self.node.clone(),
                exp,
            }),
        }
    }

    #[napi]
    pub fn cast(&self, dtype: NativeDType) -> Self {
        Self {
            node: Arc::new(LazyNode::Cast {
                a: self.node.clone(),
                dtype: dtype.into(),
            }),
        }
    }

    #[napi]
    pub fn sum(&self, dims: Vec<u32>, keepdims: bool) -> Self {
        Self {
            node: Arc::new(LazyNode::Sum {
                a: self.node.clone(),
                dims: dims.iter().map(|&d| d as usize).collect(),
                keepdims,
            }),
        }
    }

    #[napi]
    pub fn mean(&self, dims: Vec<u32>, keepdims: bool) -> Self {
        Self {
            node: Arc::new(LazyNode::Mean {
                a: self.node.clone(),
                dims: dims.iter().map(|&d| d as usize).collect(),
                keepdims,
            }),
        }
    }

    #[napi]
    pub fn max(&self, dims: Vec<u32>, keepdims: bool) -> Self {
        Self {
            node: Arc::new(LazyNode::Max {
                a: self.node.clone(),
                dims: dims.iter().map(|&d| d as usize).collect(),
                keepdims,
            }),
        }
    }

    #[napi]
    pub fn min(&self, dims: Vec<u32>, keepdims: bool) -> Self {
        Self {
            node: Arc::new(LazyNode::Min {
                a: self.node.clone(),
                dims: dims.iter().map(|&d| d as usize).collect(),
                keepdims,
            }),
        }
    }

    #[napi]
    pub fn reshape(&self, shape: Vec<u32>) -> Self {
        Self {
            node: Arc::new(LazyNode::Reshape {
                a: self.node.clone(),
                shape: shape.iter().map(|&d| d as usize).collect(),
            }),
        }
    }

    #[napi]
    pub fn permute(&self, dims: Vec<u32>) -> Self {
        Self {
            node: Arc::new(LazyNode::Permute {
                a: self.node.clone(),
                dims: dims.iter().map(|&d| d as usize).collect(),
            }),
        }
    }

    #[napi]
    pub fn slice(&self, ranges: Vec<Vec<u32>>) -> Self {
        Self {
            node: Arc::new(LazyNode::Slice {
                a: self.node.clone(),
                ranges: ranges
                    .iter()
                    .map(|r| (r[0] as usize, r[1] as usize, r[2] as usize))
                    .collect(),
            }),
        }
    }

    #[napi]
    pub fn concat(&self, other: &LazyTensor, dim: u32) -> Self {
        Self {
            node: Arc::new(LazyNode::Concat {
                a: self.node.clone(),
                b: other.node.clone(),
                dim: dim as usize,
            }),
        }
    }

    #[napi]
    pub fn broadcast_to(&self, shape: Vec<u32>) -> Self {
        Self {
            node: Arc::new(LazyNode::BroadcastTo {
                a: self.node.clone(),
                shape: shape.iter().map(|&d| d as usize).collect(),
            }),
        }
    }
}

fn eval_cmp(
    a: &Arc<LazyNode>,
    b: &Arc<LazyNode>,
    cancelled: &AtomicBool,
    cache: &mut std::collections::HashMap<*const LazyNode, Tensor>,
    f: impl Fn(&Tensor, &Tensor) -> candle_core::Result<Tensor>,
) -> candle_core::Result<Tensor> {
    let a = eval_node(a, cancelled, cache)?;
    let b = eval_node(b, cancelled, cache)?;
    let shape = a.shape().broadcast_shape_binary_op(b.shape(), "cmp")?;
    let a = a.broadcast_as(shape.clone())?;
    let b = b.broadcast_as(shape)?;
    f(&a, &b)
}

fn reduce_dims(
    t: &Tensor,
    dims: &[usize],
    keepdims: bool,
    f: impl Fn(&Tensor, usize) -> candle_core::Result<Tensor>,
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

fn eval_node(
    node: &Arc<LazyNode>,
    cancelled: &AtomicBool,
    cache: &mut std::collections::HashMap<*const LazyNode, Tensor>,
) -> candle_core::Result<Tensor> {
    if cancelled.load(Ordering::Relaxed) {
        return Err(candle_core::Error::Msg("operation aborted".to_string()));
    }
    if let Some(cached) = cache.get(&Arc::as_ptr(node)) {
        return Ok(cached.clone());
    }
    let output = match &**node {
        LazyNode::Leaf(tensor) => tensor.clone(),
        LazyNode::FromBytes {
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
        LazyNode::Zeros {
            shape,
            dtype,
            device,
        } => Tensor::zeros(shape.clone(), *dtype, device)?,
        LazyNode::Ones {
            shape,
            dtype,
            device,
        } => Tensor::ones(shape.clone(), *dtype, device)?,
        LazyNode::Full {
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
        LazyNode::Randn {
            shape,
            dtype,
            device,
        } => Tensor::randn(0f32, 1f32, shape.clone(), device)?.to_dtype(*dtype)?,
        LazyNode::Arange {
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
        LazyNode::Eye { n, dtype, device } => {
            let i = Tensor::arange(0u32, *n as u32, device)?.reshape((*n, 1))?;
            let j = Tensor::arange(0u32, *n as u32, device)?.reshape((1, *n))?;
            i.broadcast_eq(&j)?.to_dtype(*dtype)?
        }
        LazyNode::Add { a, b } => {
            let a = eval_node(a, cancelled, cache)?;
            let b = eval_node(b, cancelled, cache)?;
            a.broadcast_add(&b)?
        }
        LazyNode::Sub { a, b } => {
            let a = eval_node(a, cancelled, cache)?;
            let b = eval_node(b, cancelled, cache)?;
            a.broadcast_sub(&b)?
        }
        LazyNode::Mul { a, b } => {
            let a = eval_node(a, cancelled, cache)?;
            let b = eval_node(b, cancelled, cache)?;
            a.broadcast_mul(&b)?
        }
        LazyNode::Div { a, b } => {
            let a = eval_node(a, cancelled, cache)?;
            let b = eval_node(b, cancelled, cache)?;
            a.broadcast_div(&b)?
        }
        LazyNode::Eq { a, b } => eval_cmp(a, b, cancelled, cache, |a, b| a.eq(b))?,
        LazyNode::Gt { a, b } => eval_cmp(a, b, cancelled, cache, |a, b| a.gt(b))?,
        LazyNode::Lt { a, b } => eval_cmp(a, b, cancelled, cache, |a, b| a.lt(b))?,
        LazyNode::Ge { a, b } => eval_cmp(a, b, cancelled, cache, |a, b| a.ge(b))?,
        LazyNode::Le { a, b } => eval_cmp(a, b, cancelled, cache, |a, b| a.le(b))?,
        LazyNode::Neg { a } => eval_node(a, cancelled, cache)?.neg()?,
        LazyNode::Abs { a } => eval_node(a, cancelled, cache)?.abs()?,
        LazyNode::Sqrt { a } => eval_node(a, cancelled, cache)?.sqrt()?,
        LazyNode::Exp { a } => eval_node(a, cancelled, cache)?.exp()?,
        LazyNode::Log { a } => eval_node(a, cancelled, cache)?.log()?,
        LazyNode::Sin { a } => eval_node(a, cancelled, cache)?.sin()?,
        LazyNode::Cos { a } => eval_node(a, cancelled, cache)?.cos()?,
        LazyNode::Pow { a, exp } => eval_node(a, cancelled, cache)?.powf(*exp)?,
        LazyNode::Cast { a, dtype } => eval_node(a, cancelled, cache)?.to_dtype(*dtype)?,
        LazyNode::Sum { a, dims, keepdims } => {
            let t = eval_node(a, cancelled, cache)?;
            reduce_dims(&t, dims, *keepdims, |t, d| t.sum(d))?
        }
        LazyNode::Mean { a, dims, keepdims } => {
            let t = eval_node(a, cancelled, cache)?;
            reduce_dims(&t, dims, *keepdims, |t, d| t.mean(d))?
        }
        LazyNode::Max { a, dims, keepdims } => {
            let t = eval_node(a, cancelled, cache)?;
            reduce_dims(&t, dims, *keepdims, |t, d| t.max(d))?
        }
        LazyNode::Min { a, dims, keepdims } => {
            let t = eval_node(a, cancelled, cache)?;
            reduce_dims(&t, dims, *keepdims, |t, d| t.min(d))?
        }
        LazyNode::Reshape { a, shape } => eval_node(a, cancelled, cache)?.reshape(shape.clone())?,
        LazyNode::Permute { a, dims } => eval_node(a, cancelled, cache)?.permute(dims.clone())?,
        LazyNode::Slice { a, ranges } => {
            let mut t = eval_node(a, cancelled, cache)?;
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
                    t = t.index_select(&idx, dim)?;
                }
            }
            t
        }
        LazyNode::Concat { a, b, dim } => {
            let a = eval_node(a, cancelled, cache)?;
            let b = eval_node(b, cancelled, cache)?;
            Tensor::cat(&[&a, &b], *dim)?
        }
        LazyNode::BroadcastTo { a, shape } => {
            eval_node(a, cancelled, cache)?.broadcast_as(shape.clone())?
        }
        LazyNode::Matmul { a, b } => {
            let a = eval_node(a, cancelled, cache)?;
            let b = eval_node(b, cancelled, cache)?;
            a.matmul(&b)?
        }
    };
    cache.insert(Arc::as_ptr(node), output.clone());
    Ok(output)
}

#[napi]
pub fn is_device_available(device: String) -> bool {
    get_device(Some(device)).is_ok()
}

#[napi]
pub async fn eval_lazy(
    tensor: &LazyTensor,
    token: Option<&CancellationToken>,
) -> Result<NativeTensor> {
    let node = tensor.node.clone();
    let flag = token.map(|t| t.cancelled.clone());
    let compute = tokio::task::spawn_blocking(move || {
        let cancelled = flag.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        let mut cache = std::collections::HashMap::new();
        let output = eval_node(&node, &cancelled, &mut cache).map_err(to_napi_err)?;
        output.device().synchronize().map_err(to_napi_err)?;
        Ok(NativeTensor::wrap(output))
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

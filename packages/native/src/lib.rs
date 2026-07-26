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
    ZeroCopy { tensor: Tensor, addr: usize },
    Owned { ptr: *mut u8, len: usize, cap: usize },
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
}

impl From<NativeDType> for DType {
    fn from(dtype: NativeDType) -> Self {
        match dtype {
            NativeDType::F32 => DType::F32,
            NativeDType::F64 => DType::F64,
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
    #[napi(factory)]
    pub async fn zeros(
        shape: Vec<u32>,
        dtype: Option<NativeDType>,
        device: Option<String>,
    ) -> Result<Self> {
        let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let dtype = dtype.unwrap_or(NativeDType::F32).into();
        Tensor::zeros(shape, dtype, &get_device(device)?)
            .map(Self::wrap)
            .map_err(to_napi_err)
    }

    #[napi(factory)]
    pub async fn randn(
        shape: Vec<u32>,
        dtype: Option<NativeDType>,
        device: Option<String>,
    ) -> Result<Self> {
        let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let dtype: DType = dtype.unwrap_or(NativeDType::F32).into();
        Tensor::randn(0f32, 1f32, shape, &get_device(device)?)
            .and_then(|t| t.to_dtype(dtype))
            .map(Self::wrap)
            .map_err(to_napi_err)
    }

    #[napi]
    pub async fn add(&self, other: &NativeTensor) -> Result<Self> {
        let a = self.inner.clone();
        let b = other.inner.clone();
        (&a + &b).map(Self::wrap).map_err(to_napi_err)
    }

    #[napi]
    pub async fn matmul(
        &self,
        other: &NativeTensor,
        token: Option<&CancellationToken>,
    ) -> Result<Self> {
        let a = self.inner.clone();
        let b = other.inner.clone();
        let compute =
            tokio::task::spawn_blocking(move || a.matmul(&b).map(Self::wrap).map_err(to_napi_err));
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
    pub async fn readback(&self) -> Result<Readback> {
        let flat = self.inner.flatten_all().map_err(to_napi_err)?;
        let elem_size = flat.dtype().size_in_bytes();
        let elem_count = flat.elem_count();
        let byte_len = elem_count * elem_size;
        let base: *const u8 = {
            let (storage, _) = flat.storage_and_layout();
            match &*storage {
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
}

enum LazyNode {
    Leaf(Tensor),
    Zeros {
        shape: Vec<usize>,
        dtype: DType,
        device: Device,
    },
    Randn {
        shape: Vec<usize>,
        dtype: DType,
        device: Device,
    },
    Add {
        a: Arc<LazyNode>,
        b: Arc<LazyNode>,
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
    pub fn matmul(&self, other: &LazyTensor) -> Self {
        Self {
            node: Arc::new(LazyNode::Matmul {
                a: self.node.clone(),
                b: other.node.clone(),
            }),
        }
    }
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
        LazyNode::Zeros {
            shape,
            dtype,
            device,
        } => Tensor::zeros(shape.clone(), *dtype, device)?,
        LazyNode::Randn {
            shape,
            dtype,
            device,
        } => Tensor::randn(0f32, 1f32, shape.clone(), device)?.to_dtype(*dtype)?,
        LazyNode::Add { a, b } => {
            let a = eval_node(a, cancelled, cache)?;
            let b = eval_node(b, cancelled, cache)?;
            (&a + &b)?
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

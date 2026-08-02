use candle_core::{CpuStorage, DType, Device, Storage, Tensor};
use candle_core::D;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

mod fusion;
mod flash;
mod layer_norm;
mod loss;
mod paged;
mod rotary;
mod tokenizer;

fn to_napi_err(err: candle_core::Error) -> Error {
    Error::new(Status::GenericFailure, err.to_string())
}

fn to_join_err(err: tokio::task::JoinError) -> Error {
    Error::new(Status::GenericFailure, err.to_string())
}

// Applies a rank-2 linalg kernel to each matrix of a batched [..., n, m]
// tensor and stacks the results back into the leading shape.
fn batch_linalg(
    t: &Tensor,
    f: &dyn Fn(&Tensor) -> candle_core::Result<Tensor>,
) -> candle_core::Result<Tensor> {
    let dims = t.dims();
    let rank = dims.len();
    if rank == 2 {
        return f(t);
    }
    let (n, m) = (dims[rank - 2], dims[rank - 1]);
    let batch: usize = dims[..rank - 2].iter().product();
    if batch == 0 {
        return Err(candle_core::Error::Msg("linalg: empty batch".to_string()));
    }
    let flat = t.reshape((batch, n, m))?;
    let mut outs = Vec::with_capacity(batch);
    for i in 0..batch {
        outs.push(f(&flat.get(i)?)?);
    }
    let mut out_shape: Vec<usize> = dims[..rank - 2].to_vec();
    out_shape.extend_from_slice(outs[0].dims());
    Tensor::stack(&outs, 0)?.reshape(out_shape)
}

fn batch_solve(a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    let dims = a.dims();
    let rank = dims.len();
    if rank == 2 {
        return cpu_inverse(a)?.matmul(b);
    }
    let n = dims[rank - 2];
    let k = b.dims()[rank - 1];
    let batch: usize = dims[..rank - 2].iter().product();
    if batch == 0 {
        return Err(candle_core::Error::Msg("linalg: empty batch".to_string()));
    }
    let a_flat = a.reshape((batch, n, n))?;
    let b_flat = b.reshape((batch, n, k))?;
    let mut outs = Vec::with_capacity(batch);
    for i in 0..batch {
        outs.push(cpu_inverse(&a_flat.get(i)?)?.matmul(&b_flat.get(i)?)?);
    }
    Tensor::stack(&outs, 0)?.reshape(b.shape())
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

// dW for a 2-D convolution: im2col of the (padded) input contracted with
// the flattened cotangent, computed per group with on-device candle ops.
fn conv2d_backward_w(
    x: &Tensor,
    g: &Tensor,
    kernel: [usize; 2],
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> candle_core::Result<Tensor> {
    let (n, c_in, _, _) = x.dims4()?;
    let (_, _, oh, ow) = g.dims4()?;
    let (kh, kw) = (kernel[0], kernel[1]);
    let c_per = c_in / groups;
    let cout_per = out_channels / groups;
    let mut group_outs = Vec::with_capacity(groups);
    for gi in 0..groups {
        let xg = x.narrow(1, gi * c_per, c_per)?;
        let gg = g.narrow(1, gi * cout_per, cout_per)?;
        let xp = if padding > 0 {
            xg.pad_with_zeros(2, padding, padding)?
                .pad_with_zeros(3, padding, padding)?
        } else {
            xg
        };
        let mut windows = Vec::with_capacity(kh * kw);
        for ky in 0..kh {
            for kx in 0..kw {
                let mut win = xp
                    .narrow(2, ky * dilation, (oh - 1) * stride + 1)?
                    .narrow(3, kx * dilation, (ow - 1) * stride + 1)?;
                if stride > 1 {
                    let idx_h = Tensor::arange_step(
                        0u32,
                        ((oh - 1) * stride + 1) as u32,
                        stride as u32,
                        win.device(),
                    )?;
                    let idx_w = Tensor::arange_step(
                        0u32,
                        ((ow - 1) * stride + 1) as u32,
                        stride as u32,
                        win.device(),
                    )?;
                    win = win
                        .contiguous()?
                        .index_select(&idx_h, 2)?
                        .index_select(&idx_w, 3)?;
                }
                windows.push(win);
            }
        }
        let stacked = Tensor::stack(&windows, 0)?;
        let cols = stacked
            .permute([1usize, 3, 4, 2, 0])?
            .contiguous()?
            .reshape((n * oh * ow, c_per * kh * kw))?;
        let g2 = gg
            .permute([1usize, 0, 2, 3])?
            .contiguous()?
            .reshape((cout_per, n * oh * ow))?;
        group_outs.push(g2.matmul(&cols)?.reshape((cout_per, c_per, kh, kw))?);
    }
    Tensor::cat(&group_outs, 0)
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

// The additive causal mask [T, S]: 0 where j <= i + (S - T) (the offset
// right-aligns the window, so single-row queries against a longer cache
// attend to everything up to the current position), -inf elsewhere.
fn sdpa_causal_additive_mask(t: usize, s: usize, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
    let i = Tensor::arange(0u32, t as u32, device)?.reshape((t, 1))?;
    let j = Tensor::arange(0u32, s as u32, device)?.reshape((1, s))?;
    let allowed = j.broadcast_le(&(i + s.saturating_sub(t) as f64)?)?;
    let zeros = Tensor::zeros((t, s), dtype, device)?;
    let neg = match dtype {
        DType::F32 => Tensor::full(f32::NEG_INFINITY, (t, s), device)?,
        DType::F64 => Tensor::full(f64::NEG_INFINITY, (t, s), device)?,
        dtype => return Err(candle_core::Error::UnsupportedDTypeForOp(dtype, "sdpa").bt()),
    };
    allowed.where_cond(&zeros, &neg)
}

// The multiplicative gate [T, S]: 1 where attention is allowed, 0
// elsewhere — masks the score gradients in the backward pass.
fn sdpa_causal_gate(t: usize, s: usize, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
    let i = Tensor::arange(0u32, t as u32, device)?.reshape((t, 1))?;
    let j = Tensor::arange(0u32, s as u32, device)?.reshape((1, s))?;
    let allowed = j.broadcast_le(&(i + s.saturating_sub(t) as f64)?)?;
    let ones = Tensor::ones((t, s), dtype, device)?;
    allowed.where_cond(&ones, &ones.zeros_like()?)
}

// Softmax over the last dim with max-subtraction (matching the composed
// Tensor.softmax path op for op).
fn sdpa_softmax(t: &Tensor) -> candle_core::Result<Tensor> {
    let m = t.max(D::Minus1)?.unsqueeze(D::Minus1)?;
    let e = t.broadcast_sub(&m)?.exp()?;
    let s = e.sum(D::Minus1)?.unsqueeze(D::Minus1)?;
    e.broadcast_div(&s)
}

fn sdpa_scores(q: &Tensor, k: &Tensor, scale: f64, causal: bool) -> candle_core::Result<Tensor> {
    let rank = q.rank();
    let kt = k.transpose(rank - 2, rank - 1)?.contiguous()?;
    let s = q.contiguous()?.broadcast_matmul(&kt)?;
    let s = (s * scale)?;
    if causal {
        let dims = s.dims();
        let (t, sq) = (dims[rank - 2], dims[rank - 1]);
        s.broadcast_add(&sdpa_causal_additive_mask(t, sq, s.dtype(), s.device())?)
    } else {
        Ok(s)
    }
}

// The reference implementation: composed candle ops. A fused flash
// kernel replaces this arm (and only this arm) when it lands.
fn sdpa_forward(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, causal: bool) -> candle_core::Result<Tensor> {
    let s = sdpa_scores(q, k, scale, causal)?;
    let p = sdpa_softmax(&s)?;
    p.broadcast_matmul(&v.contiguous()?)
}

// Closed-form backward, recomputing P from q/k (retains nothing beyond
// the caller's tensors): dV = Pᵀ·g, dP = g·Vᵀ, dS = P ∘ (dP − Σ(P∘dP)),
// dQ = dS·K·scale, dK = dSᵀ·Q·scale.
fn sdpa_backward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    scale: f64,
    causal: bool,
) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
    let rank = q.rank();
    let s = sdpa_scores(q, k, scale, causal)?;
    let p = sdpa_softmax(&s)?;
    let g = g.contiguous()?;
    let dv = p
        .transpose(rank - 2, rank - 1)?
        .contiguous()?
        .broadcast_matmul(&g)?;
    let dp = g.broadcast_matmul(&v.transpose(rank - 2, rank - 1)?.contiguous()?)?;
    let dp_sum = (&p * &dp)?.sum(D::Minus1)?.unsqueeze(D::Minus1)?;
    let mut ds = (&p * &dp.broadcast_sub(&dp_sum)?)?;
    if causal {
        let dims = ds.dims();
        let (t, sq) = (dims[rank - 2], dims[rank - 1]);
        ds = ds.broadcast_mul(&sdpa_causal_gate(t, sq, ds.dtype(), ds.device())?)?;
    }
    let dq = (ds.broadcast_matmul(&k.contiguous()?)? * scale)?;
    let dk = (ds
        .transpose(rank - 2, rank - 1)?
        .contiguous()?
        .broadcast_matmul(&q.contiguous()?)?
        * scale)?;
    Ok((dq, dk, dv))
}

// u8 mask that is 1 where target == ignore_index. A negative ignore_index
// can never match a u32 target, so u32 targets are all active in that case.
fn cross_entropy_ignored_mask(target: &Tensor, ignore_index: i64) -> candle_core::Result<Tensor> {
    match target.dtype() {
        DType::I64 => target.eq(ignore_index),
        DType::U32 => {
            if ignore_index < 0 || ignore_index > u32::MAX as i64 {
                // u8 zeros, not zeros_like: Metal has no u32-cond where_cond
                Tensor::zeros(target.shape(), DType::U8, target.device())
            } else {
                target.eq(ignore_index as u32)
            }
        }
        dtype => Err(candle_core::Error::UnsupportedDTypeForOp(dtype, "cross_entropy").bt()),
    }
}

fn cross_entropy_active_count(target: &Tensor, ignored: &Tensor) -> candle_core::Result<f64> {
    let total = target.elem_count() as f64;
    // f32: counts are small integers, exact, and Metal has no u8 -> f64 cast
    let ignored_count = ignored.to_dtype(DType::F32)?.sum_all()?.to_vec0::<f32>()? as f64;
    Ok(total - ignored_count)
}

fn cross_entropy_check_labels(target: &Tensor, ignored: &Tensor, classes: usize) -> candle_core::Result<()> {
    let invalid = match target.dtype() {
        DType::I64 => (target.lt(0i64)? + target.ge(classes as i64)?)?,
        DType::U32 => target.ge(classes as u32)?,
        dtype => {
            return Err(candle_core::Error::UnsupportedDTypeForOp(dtype, "cross_entropy").bt())
        }
    };
    let active = ignored.eq(&ignored.zeros_like()?)?;
    let invalid_active = (invalid * active)?.to_dtype(DType::F32)?.sum_all()?.to_vec0::<f32>()?;
    if invalid_active > 0.0 {
        return Err(candle_core::Error::Msg(format!(
            "cross_entropy: target out of range [0, {classes}) at an active position"
        )));
    }
    Ok(())
}

fn cross_entropy_forward(logits: &Tensor, target: &Tensor, ignore_index: i64) -> candle_core::Result<Tensor> {
    let rank = logits.rank();
    let classes = logits.dim(rank - 1)?;
    let ignored = cross_entropy_ignored_mask(target, ignore_index)?;
    let count = cross_entropy_active_count(target, &ignored)?;
    if count == 0.0 {
        return Err(candle_core::Error::Msg(
            "cross_entropy: no active targets (all positions are ignored)".to_string(),
        ));
    }
    cross_entropy_check_labels(target, &ignored, classes)?;
    let lse = logits.log_sum_exp(rank - 1)?;
    // ignored positions may hold out-of-range values (e.g. -100); gather at 0
    // there instead and mask the result below
    let safe_target = ignored.where_cond(&target.zeros_like()?, target)?;
    let picked = logits
        .gather(&safe_target.unsqueeze(rank - 1)?.contiguous()?, rank - 1)?
        .reshape(target.shape())?;
    let per_position = (lse - picked)?;
    let masked = ignored.where_cond(&per_position.zeros_like()?, &per_position)?;
    masked.sum_all()? * (1.0 / count)
}

fn cross_entropy_backward(logits: &Tensor, target: &Tensor, ignore_index: i64) -> candle_core::Result<Tensor> {
    let rank = logits.rank();
    let classes = logits.dim(rank - 1)?;
    let ignored = cross_entropy_ignored_mask(target, ignore_index)?;
    let count = cross_entropy_active_count(target, &ignored)?;
    if count == 0.0 {
        return Err(candle_core::Error::Msg(
            "cross_entropy: no active targets (all positions are ignored)".to_string(),
        ));
    }
    // p = softmax(logits) computed as exp(z - logsumexp(z))
    let lse = logits.log_sum_exp(rank - 1)?.unsqueeze(rank - 1)?;
    let probs = (logits - &lse.broadcast_as(logits.shape())?)?.exp()?;
    // one-hot at the target positions, zeroed where ignored
    let classes_ix = Tensor::arange(0u32, classes as u32, logits.device())?
        .to_dtype(target.dtype())?
        .broadcast_as(logits.shape())?;
    let one_hot = target.unsqueeze(rank - 1)?.broadcast_as(logits.shape())?.eq(&classes_ix)?;
    let ignored_b = ignored.unsqueeze(rank - 1)?.broadcast_as(one_hot.shape())?;
    let one_hot = ignored_b.where_cond(&one_hot.zeros_like()?, &one_hot)?;
    let grad = (probs - one_hot.to_dtype(logits.dtype())?)? * (1.0 / count);
    let grad = grad?;
    ignored_b.where_cond(&grad.zeros_like()?, &grad)
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
    // f16/bf16 read back as f32: JS has no half typed arrays we can
    // rely on, and the conversion keeps the destructor surface small.
    let flat = if matches!(inner.dtype(), DType::F16 | DType::BF16) {
        inner.to_dtype(DType::F32).map_err(to_napi_err)?
    } else {
        inner.clone()
    };
    let flat = flat.flatten_all().map_err(to_napi_err)?;
    // flatten_all (and the f16 conversion) materialize with an async device
    // copy; wait for it before reading the buffer.
    flat.device().synchronize().map_err(to_napi_err)?;
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

// Where a position-indexed semantic node reads its base position:
// zero in user graphs, the sequence cursor in decode-rewritten ones.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PositionOffset {
    Absolute,
    Cursor,
}

enum NodeKind {
    Leaf(Tensor),
    // RFC 0008: placeholder leaves for compiled programs. An Input carries
    // the declared signature of one call argument; it evaluates only inside
    // CompiledProgram::run, which binds the slot to an argument buffer.
    Input {
        slot: u32,
        shape: Vec<usize>,
        dtype: DType,
        device: Device,
    },
    ScalarInput {
        slot: u32,
        dtype: DType,
        device: Device,
    },
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
    dtype: DType,
    device: Device,
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

fn device_key(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal(_) => "metal",
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
                tensor.dims().to_vec(),
                tensor.dtype(),
                tensor.device().clone(),
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
    if matches!(dtype, DType::F64) && matches!(device, Device::Metal(_)) {
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
    cache: std::collections::HashMap<u64, Tensor>,
    // AdamW step id -> (next m, next v); the step node's own value is the
    // updated parameter, stored in the regular cache
    adamw: std::collections::HashMap<u64, [Tensor; 2]>,
    sgd: std::collections::HashMap<u64, Tensor>,
    // FusedElementwiseMulti id -> all outputs; the node's own cache entry
    // holds output 0 so the evaluator's single-value invariant holds
    multi: std::collections::HashMap<u64, Vec<Tensor>>,
    // LayerNormBackward id -> (dw, db); the node's own cache entry is dx.
    ln: std::collections::HashMap<u64, [Tensor; 2]>,
    consumers: std::collections::HashMap<u64, usize>,
    roots: HashSet<u64>,
    // RFC 0008: Input/ScalarInput node id -> argument buffer, populated by
    // CompiledProgram::run. Empty for ordinary eval_lazy walks.
    slots: std::collections::HashMap<u64, Tensor>,
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
    ce_checks: Vec<(Tensor, bool, usize)>,
}

impl Evaluator {
    fn new(roots: &[Arc<Node>]) -> Self {
        Self::with_slots(roots, std::collections::HashMap::new())
    }

    fn with_slots(roots: &[Arc<Node>], slots: std::collections::HashMap<u64, Tensor>) -> Self {
        Self::with_kv(roots, slots, None)
    }

    fn with_kv(
        roots: &[Arc<Node>],
        slots: std::collections::HashMap<u64, Tensor>,
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
    fn run_ce_checks(&self) -> candle_core::Result<()> {
        for (buffer, forward, classes) in &self.ce_checks {
            let values = buffer.to_vec1::<f32>()?;
            if *forward {
                let (active, invalid) = (values[1] as usize, values[2] as usize);
                if active == 0 {
                    return Err(candle_core::Error::Msg(
                        "cross_entropy: no active targets (all positions are ignored)".to_string(),
                    ));
                }
                if invalid > 0 {
                    return Err(candle_core::Error::Msg(format!(
                        "cross_entropy: target out of range [0, {classes}) at an active position"
                    )));
                }
            } else if values[0] == 0.0 {
                return Err(candle_core::Error::Msg(
                    "cross_entropy: no active targets (all positions are ignored)".to_string(),
                ));
            }
        }
        Ok(())
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
        if matches!(out.device(), Device::Metal(_)) && out.rank() > 4 {
            // candle's Metal reduce kernel miscomputes non-trailing dims at
            // rank > 4; collapse the untouched dims so each single-dim
            // reduce runs at rank 3
            let shape = out.dims().to_vec();
            let before: usize = shape[..d].iter().product();
            let after: usize = shape[d + 1..].iter().product();
            let collapsed = out.contiguous()?.reshape((before, shape[d], after))?;
            let mut reduced_shape: Vec<usize> = shape[..d].to_vec();
            reduced_shape.extend_from_slice(&shape[d + 1..]);
            out = f(&collapsed, 1)?.reshape(reduced_shape)?;
        } else {
            out = f(&out, d)?;
        }
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

fn eval_uncached(node: &Arc<Node>, ev: &mut Evaluator) -> candle_core::Result<Tensor> {
    let output = match &node.kind {
        NodeKind::Leaf(tensor) => tensor.clone(),
        NodeKind::Input { slot, .. } | NodeKind::ScalarInput { slot, .. } => {
            ev.slots.get(&node.id).cloned().ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "input slot {slot} is unbound: placeholder leaves evaluate only inside a compiled program run"
                ))
            })?
        }
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
            DType::F16 => {
                let v: Vec<half::f16> = data
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
                    .collect();
                Tensor::from_vec(v, shape.clone(), device)?
            }
            DType::BF16 => {
                let v: Vec<half::bf16> = data
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_le_bytes([c[0], c[1]]))
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
            DType::F16 => Tensor::full(half::f16::from_f64(*value), shape.clone(), device)?,
            DType::BF16 => Tensor::full(half::bf16::from_f64(*value), shape.clone(), device)?,
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
        } => Tensor::rand(*lo as f32, *hi as f32, shape.clone(), device)?.to_dtype(*dtype)?,
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
        NodeKind::CrossEntropy {
            logits,
            target,
            ignore_index,
        } => {
            let logits_t = ev.value(logits.id)?;
            let target_t = ev.value(target.id)?;
            if loss::is_supported(&logits_t, &target_t) {
                let (loss_t, status) = loss::ce_forward(&logits_t, &target_t, *ignore_index)?;
                ev.ce_checks.push((status, true, logits_t.elem_count() / logits_t.dim(logits_t.rank() - 1)?));
                loss_t
            } else {
                cross_entropy_forward(&logits_t, &target_t, *ignore_index)?
            }
        }
        NodeKind::CrossEntropyBackward {
            logits,
            target,
            ignore_index,
        } => {
            let logits_t = ev.value(logits.id)?;
            let target_t = ev.value(target.id)?;
            if loss::is_supported(&logits_t, &target_t) {
                let (grad, count) = loss::ce_backward(&logits_t, &target_t, *ignore_index)?;
                ev.ce_checks.push((count, false, 0));
                grad
            } else {
                cross_entropy_backward(&logits_t, &target_t, *ignore_index)?
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
            if flash::is_supported(&q) {
                let (o, l) = flash::forward(&q, &ev.value(k.id)?, &ev.value(v.id)?, *scale, *causal)?;
                // L rides the evaluator for the chunked backward; the
                // node's own cache entry holds O.
                ev.multi.insert(node.id, vec![o.clone(), l]);
                o
            } else {
                sdpa_forward(&q, &ev.value(k.id)?, &ev.value(v.id)?, *scale, *causal)?
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
            let o = ev.value(fwd.id)?;
            let l = ev.multi.get(&fwd.id).and_then(|outs| outs.get(1)).cloned();
            let (dq, dk, dv) = match l {
                Some(l) if flash::is_supported(&q) => flash::backward(
                    &q,
                    &ev.value(k.id)?,
                    &ev.value(v.id)?,
                    &o,
                    &l,
                    &ev.value(g.id)?,
                    *scale,
                    *causal,
                )?,
                _ => sdpa_backward(
                    &q,
                    &ev.value(k.id)?,
                    &ev.value(v.id)?,
                    &ev.value(g.id)?,
                    *scale,
                    *causal,
                )?,
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
                candle_core::Error::Msg("sdpa backward out: outputs missing".to_string())
            })?,
        NodeKind::PositionEmbedding { weight, seq_len } => ev
            .value(weight.id)?
            .narrow(0, 0, *seq_len)?
            .contiguous()?,
        NodeKind::KvAttention {
            q,
            k,
            v,
            scale,
            layer,
            window,
        } => {
            let kv = ev.kv.clone().ok_or_else(|| {
                candle_core::Error::Msg(
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
                        candle_core::Error::Msg(
                            "rotary embedding: cursor offset outside a kv program run".to_string(),
                        )
                    })?;
                    // One cursor per batch slot (RFC 0013).
                    let mut offsets = Vec::with_capacity(kv.slots.len());
                    for slot in &kv.slots {
                        offsets.push(slot.lock().map_err(|e| {
                            candle_core::Error::Msg(format!(
                                "rotary embedding: sequence lock poisoned: {e}"
                            ))
                        })?.cursor);
                    }
                    offsets
                }
            };
            if rotary::is_supported(&x) {
                rotary::rotary(&x, &offsets, *theta, 1.0)?
            } else {
                rotary_forward(&x, &offsets, *theta, 1.0)?
            }
        }
        NodeKind::RotaryEmbeddingBackward { g, theta, .. } => {
            let g = ev.value(g.id)?;
            if rotary::is_supported(&g) {
                rotary::rotary(&g, &[0usize], *theta, -1.0)?
            } else {
                // Transpose rotation == forward with negated angles.
                rotary_forward(&g, &[0usize], *theta, -1.0)?
            }
        }
        NodeKind::LayerNorm { x, weight, bias, eps } => {
            let x = ev.value(x.id)?;
            let weight = ev.value(weight.id)?;
            let bias = ev.value(bias.id)?;
            if layer_norm::is_supported(&x, &weight) {
                layer_norm::ln_forward(&x, &weight, &bias, *eps)?
            } else {
                layer_norm_composed(&x, &weight, &bias, *eps)?
            }
        }
        NodeKind::LayerNormBackward { x, weight, g, eps } => {
            let x = ev.value(x.id)?;
            let weight = ev.value(weight.id)?;
            let g = ev.value(g.id)?;
            let (dx, dw, db) = layer_norm_backward(&x, &weight, &g, *eps)?;
            ev.ln.insert(node.id, [dw, db]);
            dx
        }
        NodeKind::LayerNormBackwardOut { of, index } => {
            let _ = ev.value(of.id)?;
            ev.ln
                .get(&of.id)
                .and_then(|outs| outs.get(*index as usize - 1))
                .cloned()
                .ok_or_else(|| {
                    candle_core::Error::Msg(
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
            x.contiguous()?
                .conv1d(&w.contiguous()?, *padding, *stride, *dilation, *groups)?
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
            x.contiguous()?
                .conv2d(&w.contiguous()?, *padding, *stride, *dilation, *groups)?
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
            x.contiguous()?.conv_transpose1d(
                &w.contiguous()?,
                *padding,
                *output_padding,
                *stride,
                *dilation,
                *groups,
            )?
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
            if *groups == 1 {
                x.contiguous()?.conv_transpose2d(
                    &w.contiguous()?,
                    *padding,
                    *output_padding,
                    *stride,
                    *dilation,
                )?
            } else {
                // candle's conv_transpose2d has no groups parameter
                let xs = x.chunk(*groups, 1)?;
                let ws = w.chunk(*groups, 0)?;
                let mut outs = Vec::with_capacity(*groups);
                for (xb, wb) in xs.iter().zip(&ws) {
                    outs.push(xb.contiguous()?.conv_transpose2d(
                        &wb.contiguous()?,
                        *padding,
                        *output_padding,
                        *stride,
                        *dilation,
                    )?);
                }
                Tensor::cat(&outs, 1)?
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
            let x = ev.value(x.id)?.unsqueeze(3)?;
            let g = ev.value(g.id)?.unsqueeze(3)?;
            conv2d_backward_w(
                &x,
                &g,
                [*kernel, 1],
                *out_channels,
                *stride,
                *padding,
                *dilation,
                *groups,
            )?
            .squeeze(3)?
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
            conv2d_backward_w(
                &x,
                &g,
                *kernel,
                *out_channels,
                *stride,
                *padding,
                *dilation,
                *groups,
            )?
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
            batch_linalg(&cpu, &cpu_inverse)?.to_device(t.device())?
        }
        NodeKind::Det { a } => {
            let t = ev.value(a.id)?;
            let cpu = t.to_device(&Device::Cpu)?;
            batch_linalg(&cpu, &cpu_det)?.to_device(t.device())?
        }
        NodeKind::Solve { a, b } => {
            let a = ev.value(a.id)?;
            let b = ev.value(b.id)?;
            let a_cpu = a.to_device(&Device::Cpu)?;
            let b_cpu = b.to_device(&Device::Cpu)?;
            batch_solve(&a_cpu, &b_cpu)?.to_device(a.device())?
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
            let lr_t = ev.value(lr.id)?.to_dtype(p.dtype())?.to_device(p.device())?;
            let c1_t = ev.value(c1.id)?.to_dtype(p.dtype())?.to_device(p.device())?;
            let c2_t = ev.value(c2.id)?.to_dtype(p.dtype())?.to_device(p.device())?;
            let fused = if fusion::is_supported(&p.device(), p.dtype()) {
                let exprs = fusion::adamw_exprs(*beta1, *beta2, *eps, *weight_decay);
                fusion::run(
                    &exprs,
                    &[p.clone(), g.clone(), m_t.clone(), v_t.clone()],
                    None,
                    &[lr_t.clone(), c1_t.clone(), c2_t.clone()],
                    p.elem_count(),
                    p.dims(),
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
                    ev.adamw
                        .insert(node.id, [it.next().unwrap(), it.next().unwrap()]);
                    next_p
                }
                None => {
                    let one_minus_beta1 = 1.0 - *beta1;
                    let one_minus_beta2 = 1.0 - *beta2;
                    let next_m = ((m_t * *beta1)? + (&g * one_minus_beta1)?)?;
                    let next_v = ((v_t * *beta2)? + ((&g * &g)? * one_minus_beta2)?)?;
                    let m_hat = next_m.broadcast_div(&c1_t)?;
                    let v_hat = next_v.broadcast_div(&c2_t)?;
                    let adjusted = (m_hat.broadcast_div(&(v_hat.sqrt()? + *eps)?)?)
                        .broadcast_mul(&lr_t)?;
                    let next_p = if *weight_decay == 0.0 {
                        (p - adjusted)?
                    } else {
                        // p * (1 - lr * weight_decay) - adjusted, factored as
                        // p - p * (lr * weight_decay) - adjusted to keep the
                        // decay a tensor op.
                        let decay = p.broadcast_mul(&(&lr_t * *weight_decay)?)?;
                        ((p - decay)? - adjusted)?
                    };
                    ev.adamw.insert(node.id, [next_m, next_v]);
                    next_p
                }
            }
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
            let first_t = ev.value(first.id)?.to_dtype(p.dtype())?.to_device(p.device())?;
            let lr_t = ev.value(lr.id)?.to_dtype(p.dtype())?.to_device(p.device())?;
            let fused = if fusion::is_supported(&p.device(), p.dtype()) {
                let exprs = fusion::sgd_exprs(*momentum, *dampening, *nesterov, *weight_decay);
                fusion::run(
                    &exprs,
                    &[p.clone(), g.clone(), v_t.clone()],
                    None,
                    &[lr_t.clone(), first_t.clone()],
                    p.elem_count(),
                    p.dims(),
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
                None => {
                    let g = if *weight_decay == 0.0 {
                        g
                    } else {
                        (&g + (&p * *weight_decay)?)?
                    };
                    // next_v = first ? g : momentum * v + (1 - dampening) * g,
                    // as tensor arithmetic: velocity is zeros on the first
                    // step, so the (1 - first) branch contributes nothing.
                    let continued = ((v_t * *momentum)? + (&g * (1.0 - dampening))?)?;
                    let not_first = ((&first_t * -1.0)? + 1.0)?;
                    let next_v = (first_t.broadcast_mul(&g)? + not_first.broadcast_mul(&continued)?)?;
                    let used = if *nesterov {
                        (&g + (&next_v * *momentum)?)?
                    } else {
                        next_v.clone()
                    };
                    let next_p = p.broadcast_sub(&used.broadcast_mul(&lr_t)?)?;
                    ev.sgd.insert(node.id, next_v);
                    next_p
                }
            }
        }
        NodeKind::SgdOut { step, index } => {
            let _ = ev.value(step.id)?;
            match index {
                0 => ev.value(step.id)?,
                _ => ev.sgd.get(&step.id).cloned().ok_or_else(|| {
                    candle_core::Error::Msg("sgd_out: step has no stored velocity".to_string())
                })?,
            }
        }
        NodeKind::FusedElementwise {
            inputs,
            strides,
            shape,
            expr,
        } => {
            let ts: Vec<Tensor> = inputs
                .iter()
                .map(|i| ev.value(i.id))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let first = &ts[0];
            let outs = fusion::run(
                std::slice::from_ref(expr),
                &ts,
                Some(strides),
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
            let ts: Vec<Tensor> = inputs
                .iter()
                .map(|i| ev.value(i.id))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let first = &ts[0];
            let outs = fusion::run(
                exprs,
                &ts,
                Some(strides),
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
                candle_core::Error::Msg("fused pick: multi output missing".to_string())
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
            let ts: Vec<Tensor> = inputs
                .iter()
                .map(|i| ev.value(i.id))
                .collect::<candle_core::Result<Vec<_>>>()?;
            if fine {
                eprintln!("[fine] reduce collect {:.1}us ({} inputs)", t0.elapsed().as_micros(), ts.len());
            }
            let first = &ts[0];
            fusion::run_reduce(
                *op,
                expr,
                &ts,
                strides,
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
fn layer_norm_composed(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
) -> candle_core::Result<Tensor> {
    let rank = x.rank();
    let dims: Vec<usize> = (rank - weight.dims().len()..rank).collect();
    let mean = x.mean_keepdim(dims.clone())?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(dims.clone())?;
    let inv = (var + eps)?.sqrt()?.recip()?;
    centered
        .broadcast_mul(&inv)?
        .broadcast_mul(weight)?
        .broadcast_add(bias)
}

// Layer-norm backward: dx plus (dw, db). The fused Metal kernel emits
// dx and x̂ in one launch; dw/db are plain row reduces either way.
fn layer_norm_backward(
    x: &Tensor,
    weight: &Tensor,
    g: &Tensor,
    eps: f64,
) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
    let rank = x.rank();
    let k = weight.dims().len();
    let reduce_dims: Vec<usize> = (0..rank - k).collect();
    if layer_norm::is_supported(x, weight) {
        let (dx, xh) = layer_norm::ln_backward(x, weight, g, eps)?;
        let dw = g.mul(&xh)?.sum(reduce_dims.clone())?;
        let db = g.sum(reduce_dims)?;
        return Ok((dx, dw, db));
    }
    let dims: Vec<usize> = (rank - k..rank).collect();
    let mean = x.mean_keepdim(dims.clone())?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(dims.clone())?;
    let rstd = (var + eps)?.sqrt()?.recip()?;
    let xh = centered.broadcast_mul(&rstd)?;
    // dx = (dyw − mean(dyw) − x̂·mean(dyw·x̂)) · rstd
    let dyw = g.broadcast_mul(weight)?;
    let m1 = dyw.mean_keepdim(dims.clone())?;
    let m2 = dyw.broadcast_mul(&xh)?.mean_keepdim(dims)?;
    let dx = ((dyw.broadcast_sub(&m1)? - xh.broadcast_mul(&m2)?)?).broadcast_mul(&rstd)?;
    let dw = g.mul(&xh)?.sum(reduce_dims.clone())?;
    let db = g.sum(reduce_dims)?;
    Ok((dx, dw, db))
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
fn fuse_roots(roots: &[Arc<Node>]) -> std::result::Result<Vec<Arc<Node>>, String> {
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
                    && !(matches!(node.device, candle_core::Device::Metal(_))
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
                    && !(matches!(node.device, candle_core::Device::Metal(_)) && {
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
                if matches!(node.device, candle_core::Device::Metal(_)) {
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
    if nodes.iter().any(|node| matches!(node.device, Device::Metal(_))) {
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
        for output in &outputs {
            output.inner.device().synchronize().map_err(to_napi_err)?;
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

struct PoolInner {
    // Per layer, flat [max_tokens, kv_heads, head_dim] slabs; block b
    // occupies rows b*block_size..(b+1)*block_size. Slab dtype u8 means
    // int8-quantized storage (RFC 0012 storage tier): rows are
    // symmetric-quantized with a per-(token, head) absmax scale held in
    // `scales` — two slabs per layer (k then v) when the data slabs are
    // u8, empty otherwise.
    k: Vec<Tensor>,
    v: Vec<Tensor>,
    scales: Vec<Tensor>,
    kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_tokens: usize,
    blocks: Mutex<BlockStore>,
    device: Device,
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
    paged_tables: Mutex<Option<(Tensor, Tensor)>>,
}

// RoPE forward: x [.., T, D] (D even), one position offset per leading
// batch element (a single offset for unbatched graphs, one per slot in
// batched kv runs, RFC 0013). GPT-NeoX half-split rotation with
// theta^(-2j/D).
pub(crate) fn rotary_forward(x: &Tensor, offsets: &[usize], theta: f64, sign: f64) -> candle_core::Result<Tensor> {
    let dims = x.dims();
    let rank = dims.len();
    let (t, d) = (dims[rank - 2], dims[rank - 1]);
    let batch = dims[0];
    if offsets.len() != 1 && offsets.len() != batch {
        return Err(candle_core::Error::Msg(format!(
            "rotary embedding: {} offsets for batch {batch}",
            offsets.len()
        )));
    }
    let half = d / 2;
    let device = x.device();
    let inv_freq: Vec<f32> = (0..half)
        .map(|j| theta.powf(-2.0 * j as f64 / d as f64) as f32)
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?;
    let positions: Vec<f32> = if offsets.len() == 1 {
        (0..t).map(|p| (offsets[0] + p) as f32).collect()
    } else {
        offsets
            .iter()
            .flat_map(|base| (0..t).map(move |p| (*base + p) as f32))
            .collect()
    };
    let rows = if offsets.len() == 1 { 1 } else { batch };
    let positions = Tensor::from_vec(positions, (rows * t, 1), device)?;
    let angles = (positions.matmul(&inv_freq)? * sign)?; // [R*T, half]
    let mut table_shape = vec![1usize; rank - 2];
    if offsets.len() != 1 {
        table_shape[0] = batch;
    }
    table_shape.extend([t, half]);
    let cos = angles.cos()?.reshape(table_shape.as_slice())?;
    let sin = angles.sin()?.reshape(table_shape.as_slice())?;
    let first = x.narrow(rank - 1, 0, half)?;
    let second = x.narrow(rank - 1, half, half)?;
    let out_first = (first.broadcast_mul(&cos)? - second.broadcast_mul(&sin)?)?;
    let out_second = (second.broadcast_mul(&cos)? + first.broadcast_mul(&sin)?)?;
    Tensor::cat(&[&out_first, &out_second], rank - 1)?.contiguous()
}

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
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    window: Option<usize>,
) -> candle_core::Result<Tensor> {
    let dims = q.dims();
    let batch: usize = dims[..dims.len() - 3].iter().product();
    if batch != kv.slots.len() {
        return Err(candle_core::Error::Msg(format!(
            "kv attention: batch {batch} does not match {} kv slots",
            kv.slots.len()
        )));
    }
    // Metal: one fused scatter + one kernel launch attend all slots
    // over the pool slabs in place — no gather copy, decode and
    // chunked prefill alike (RFC 0013, stage 2). Everything else
    // falls back to the composed per-slot path.
    let rank = dims.len();
    let (t, h, d) = (dims[rank - 2], dims[rank - 3], dims[rank - 1]);
    if paged::is_supported(q, kv.pool.k[layer as usize].dtype(), d) {
        return kv_attention_paged(kv, layer, q, k, v, scale, window, batch, t, h, d);
    }
    if batch == 1 {
        let mut state = kv.slots[0].lock().map_err(|e| {
            candle_core::Error::Msg(format!("kv attention: sequence lock poisoned: {e}"))
        })?;
        return kv_attention_slot(&kv.pool, &mut state, layer, q, k, v, scale, window);
    }
    let mut outs = Vec::with_capacity(batch);
    for (b, slot) in kv.slots.iter().enumerate() {
        let mut state = slot.lock().map_err(|e| {
            candle_core::Error::Msg(format!("kv attention: sequence lock poisoned: {e}"))
        })?;
        outs.push(kv_attention_slot(
            &kv.pool,
            &mut state,
            layer,
            &q.narrow(0, b, 1)?,
            &k.narrow(0, b, 1)?,
            &v.narrow(0, b, 1)?,
            scale,
            window,
        )?);
    }
    Tensor::cat(&outs.iter().collect::<Vec<_>>(), 0)
}

// Metal paged attention (RFC 0013, stage 2): per-slot prepare
// (validate, allocate), one fused scatter of every slot's new rows,
// one kernel launch attending over the slabs in place (per-row causal
// lengths cover decode and chunked prefill), then per-slot eviction.
#[allow(clippy::too_many_arguments)]
fn kv_attention_paged(
    kv: &KvContext,
    layer: u32,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    window: Option<usize>,
    batch: usize,
    t: usize,
    h: usize,
    d: usize,
) -> candle_core::Result<Tensor> {
    let pool = &kv.pool;
    let layer = layer as usize;
    let mut ctxlens = Vec::with_capacity(batch);
    let mut starts = Vec::with_capacity(batch);
    let mut advance = 0usize;
    for slot in kv.slots.iter() {
        let mut state = slot.lock().map_err(|e| {
            candle_core::Error::Msg(format!("kv attention: sequence lock poisoned: {e}"))
        })?;
        advance = state.advance;
        let (_cursor, needed, start) = kv_prepare(pool, &mut state, layer, window, h, d, t)?;
        ctxlens.push(needed as u32);
        starts.push(start);
    }
    // Tables/ctxlens settle at layer 0; later layers reuse them.
    let mut cache = kv.paged_tables.lock().map_err(|e| {
        candle_core::Error::Msg(format!("kv attention: table cache lock poisoned: {e}"))
    })?;
    if cache.is_none() {
        let mut tables: Vec<u32> = Vec::new();
        let mut max_blocks = 0usize;
        for slot in &kv.slots {
            let state = slot.lock().map_err(|e| {
                candle_core::Error::Msg(format!("kv attention: sequence lock poisoned: {e}"))
            })?;
            max_blocks = max_blocks.max(state.blocks.len());
        }
        for slot in &kv.slots {
            let state = slot.lock().map_err(|e| {
                candle_core::Error::Msg(format!("kv attention: sequence lock poisoned: {e}"))
            })?;
            tables.extend_from_slice(&state.blocks);
            tables.resize(tables.len() + (max_blocks - state.blocks.len()), 0);
        }
        *cache = Some((
            Tensor::from_vec(tables, (batch, max_blocks), &pool.device)?,
            Tensor::from_vec(ctxlens.clone(), batch, &pool.device)?,
        ));
    }
    let (tables, ctxlens) = cache.as_ref().expect("populated above");
    let slab_dtype = pool.k[layer].dtype();
    let (k_scales, v_scales) = match slab_dtype {
        DType::U8 => (
            Some(&pool.scales[2 * layer]),
            Some(&pool.scales[2 * layer + 1]),
        ),
        _ => (None, None),
    };
    // One fused scatter for all slots, one attention kernel launch.
    paged::scatter(
        k,
        v,
        &pool.k[layer],
        &pool.v[layer],
        k_scales,
        v_scales,
        tables,
        ctxlens,
        pool.block_size,
        advance,
    )?;
    let out = paged::decode(
        q,
        &pool.k[layer],
        &pool.v[layer],
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
            candle_core::Error::Msg(format!("kv attention: sequence lock poisoned: {e}"))
        })?;
        kv_evict(pool, &mut state, starts[b]);
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn kv_attention_slot(
    pool: &Arc<PoolInner>,
    state: &mut SeqState,
    layer: u32,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    window: Option<usize>,
) -> candle_core::Result<Tensor> {
    if q.dtype() != DType::F32 {
        return Err(candle_core::Error::Msg(format!(
            "kv attention: dtype must be f32, got {:?}",
            q.dtype()
        )));
    }
    let layer = layer as usize;
    let dims = q.dims();
    let rank = dims.len();
    let (t, h, d) = (dims[rank - 2], dims[rank - 3], dims[rank - 1]);
    let (cursor, needed, start) = kv_prepare(pool, state, layer, window, h, d, t)?;
    kv_scatter_rows(pool, state, layer, k, v, h, d)?;
    let slab_dtype = pool.k[layer].dtype();
    let full = cursor + t;
    // row indexes [T] broadcast to the scatter/gather contract [T, H, D]
    let row_index = |rows: Vec<u32>| -> candle_core::Result<Tensor> {
        let n = rows.len();
        Tensor::from_vec(rows, (n, 1, 1), &pool.device)?.broadcast_as((n, h, d))?.contiguous()
    };
    // Row indexes [T] broadcast to the scale contract [T, H].
    let scale_index = |rows: Vec<u32>| -> candle_core::Result<Tensor> {
        let n = rows.len();
        Tensor::from_vec(rows, (n, 1), &pool.device)?.broadcast_as((n, h))?.contiguous()
    };
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
    let ctx_index = row_index(ctx_rows.clone())?;
    let ctx_scale_index = scale_index(ctx_rows)?;
    let zeros = (ctx > needed - start)
        .then(|| Tensor::zeros((ctx - (needed - start), h, d), DType::F32, &pool.device))
        .transpose()?;
    let gather_rows = |slab: &Tensor, scale: Option<&Tensor>| -> candle_core::Result<Tensor> {
        let gathered = slab.gather(&ctx_index, 0)?;
        let real = match scale {
            // Dequantize: (q - 128) * scale, per (token, head).
            Some(scale) => ((gathered.to_dtype(DType::F32)? - 128.0)?
                .broadcast_mul(&scale.gather(&ctx_scale_index, 0)?.unsqueeze(candle_core::D::Minus1)?))?,
            None => gathered.to_dtype(DType::F32)?,
        };
        let full = match &zeros {
            Some(pad) => Tensor::cat(&[&real, pad], 0)?,
            None => real,
        };
        full.permute((1, 0, 2))?.unsqueeze(0)?.contiguous()
    };
    let (k_scale, v_scale) = match slab_dtype {
        DType::U8 => (Some(&pool.scales[2 * layer]), Some(&pool.scales[2 * layer + 1])),
        _ => (None, None),
    };
    let out = sdpa_forward(
        q,
        &gather_rows(&pool.k[layer], k_scale)?,
        &gather_rows(&pool.v[layer], v_scale)?,
        scale,
        true,
    )?;
    kv_evict(pool, state, start);
    Ok(out)
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
) -> candle_core::Result<(usize, usize, usize)> {
    if layer >= pool.k.len() {
        return Err(candle_core::Error::Msg(format!(
            "kv attention: layer {layer} out of range for {} pool layers",
            pool.k.len()
        )));
    }
    if h != pool.kv_heads || d != pool.head_dim {
        return Err(candle_core::Error::Msg(format!(
            "kv attention: layer {layer} shape [{h}, {d}] does not match pool geometry [{}, {}]",
            pool.kv_heads, pool.head_dim
        )));
    }
    let cursor = state.cursor;
    // Chunked prefill: q carries the chunk length t, only `advance`
    // rows are real; the rest are pads whose outputs the caller
    // discards (causality keeps real rows from ever attending to them).
    let advance = state.advance;
    if advance == 0 || advance > t {
        return Err(candle_core::Error::Msg(format!(
            "kv attention: advance {advance} out of range for chunk length {t}"
        )));
    }
    let needed = cursor + advance;
    // Live rows after this step: everything from the attention window
    // frontier on. Blocks fully below the frontier are dead and their
    // capacity is reclaimed, so a windowed sequence's footprint is
    // O(window) however long it generates.
    let full = cursor + t;
    let start = window.map_or(0, |w| full.saturating_sub(w));
    if needed - start > pool.max_tokens {
        return Err(candle_core::Error::Msg(format!(
            "kv attention: live context {} exceeds pool capacity {}",
            needed - start,
            pool.max_tokens
        )));
    }
    while state.blocks.len() * pool.block_size < needed {
        let block = pool.alloc_block().ok_or_else(|| {
            candle_core::Error::Msg(format!(
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
    k: &Tensor,
    v: &Tensor,
    h: usize,
    d: usize,
) -> candle_core::Result<()> {
    let slab_dtype = pool.k[layer].dtype();
    let cursor = state.cursor;
    let advance = state.advance;
    let needed = cursor + advance;
    // [1, H, T, D] -> [T, H, D], real rows only
    let new_rows = |x: &Tensor| -> candle_core::Result<Tensor> {
        x.permute((0, 2, 1, 3))?
            .contiguous()?
            .narrow(1, 0, advance)?
            .reshape((advance, h, d))
    };
    // row indexes [T] broadcast to the scatter contract [T, H, D]
    let row_index = |rows: Vec<u32>| -> candle_core::Result<Tensor> {
        let n = rows.len();
        Tensor::from_vec(rows, (n, 1, 1), &pool.device)?.broadcast_as((n, h, d))?.contiguous()
    };
    // Logical position -> physical row, through the sequence's block
    // table (append-only; entries below `head` are dead and never
    // written).
    let physical = |p: usize| -> u32 {
        state.blocks[p / pool.block_size] * pool.block_size as u32
            + (p % pool.block_size) as u32
    };
    let write_rows: Vec<u32> = (cursor..needed).map(&physical).collect();
    let write_index = row_index(write_rows.clone())?;
    // Row indexes [T] broadcast to the scale contract [T, H].
    let scale_index = |rows: Vec<u32>| -> candle_core::Result<Tensor> {
        let n = rows.len();
        Tensor::from_vec(rows, (n, 1), &pool.device)?.broadcast_as((n, h))?.contiguous()
    };
    let write_scales = |layer: usize, slot: usize, scale: &Tensor| -> candle_core::Result<()> {
        pool.scales[2 * layer + slot].scatter_set(&scale_index(write_rows.clone())?, scale, 0)
    };
    if slab_dtype == DType::U8 {
        // int8 storage tier: symmetric quantization on a ±127 grid
        // (offset 128 for the u8 layout) with a per-(token, head)
        // absmax scale. The grid is deliberately not arithmetic — only
        // this kernel quantizes on write and dequantizes on read.
        let quantize = |x: &Tensor| -> candle_core::Result<(Tensor, Tensor)> {
            let scale = (x.abs()?.max(candle_core::D::Minus1)? / 127.0)?;
            let scale = (scale + 1e-12)?; // zero rows stay finite
            let q = ((x.broadcast_div(&scale.unsqueeze(candle_core::D::Minus1)?)? + 128.0)?
                .round()?
                .to_dtype(DType::U8))?;
            Ok((q, scale))
        };
        let (qk, sk) = quantize(&new_rows(k)?)?;
        let (qv, sv) = quantize(&new_rows(v)?)?;
        pool.k[layer].scatter_set(&write_index, &qk, 0)?;
        pool.v[layer].scatter_set(&write_index, &qv, 0)?;
        write_scales(layer, 0, &sk)?;
        write_scales(layer, 1, &sv)?;
    } else {
        pool.k[layer].scatter_set(&write_index, &new_rows(k)?.to_dtype(slab_dtype)?, 0)?;
        pool.v[layer].scatter_set(&write_index, &new_rows(v)?.to_dtype(slab_dtype)?, 0)?;
    }
    Ok(())
}

// Evicts dead blocks: fully below the window frontier, never attended
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
            k.push(
                Tensor::zeros((max_tokens, kv_heads, head_dim), dtype, &device)
                    .map_err(to_napi_err)?,
            );
            v.push(
                Tensor::zeros((max_tokens, kv_heads, head_dim), dtype, &device)
                    .map_err(to_napi_err)?,
            );
            if dtype == DType::U8 {
                for _ in 0..2 {
                    scales.push(
                        Tensor::zeros((max_tokens, kv_heads), DType::F32, &device)
                            .map_err(to_napi_err)?,
                    );
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

#[napi]
pub struct NativeKvSequence {
    pool: Arc<PoolInner>,
    state: Arc<Mutex<SeqState>>,
    // Serializes runs of this sequence; other sequences run concurrently
    // (their blocks are disjoint by allocation).
    run_lock: Arc<Mutex<()>>,
    released: AtomicBool,
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
    inner: Option<ProgramInner>,
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
        self.inner
            .as_ref()
            .map(|inner| inner.signature.clone())
            .ok_or_else(|| Error::new(Status::GenericFailure, "program is disposed".to_string()))
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

    #[napi]
    pub fn dispose(&mut self) {
        self.inner = None;
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
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| Error::new(Status::GenericFailure, "program is disposed".to_string()))?;
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
            if got.dims() != declared.shape.as_slice()
                || got.dtype() != declared.dtype
                || device_key(got.device()) != device_key(&declared.device)
            {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!("input slot {slot}: expected {}, got {:?}", declared.signature(), got.dims()),
                ));
            }
        }
        let slots = inner.slots.clone();
        let roots = inner.roots.clone();
        let leaves = inner.leaves.clone();
        let inputs: Vec<Tensor> = inputs.iter().map(|input| input.inner.clone()).collect();
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
                    Tensor::from_vec(cursors, batch, &declared.device).map_err(to_napi_err)?
                } else {
                    tensors.next().expect("tensor count checked").clone()
                };
                bindings.insert(slot as u64, binding);
            }
            let by_id: std::collections::HashMap<u64, Tensor> = leaves
                .iter()
                .map(|(id, slot)| (*id, bindings[&(*slot as u64)].clone()))
                .collect();
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
            // CPU encoding and GPU execution.
            for output in &outputs {
                output.inner.device().synchronize().map_err(to_napi_err)?;
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
        inner: Some(ProgramInner {
            roots: nodes,
            slots,
            leaves,
            signature,
        }),
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
    dtype: DType,
    device: Device,
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
    inner: Option<ProgramInner>,
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

fn scalar_binding(value: f64, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
    match dtype {
        DType::F32 => Tensor::full(value as f32, vec![], device),
        DType::F64 => Tensor::full(value, vec![], device),
        DType::I64 => Tensor::full(value as i64, vec![], device),
        DType::U8 => Tensor::full(value as u8, vec![], device),
        DType::U32 => Tensor::full(value as u32, vec![], device),
        DType::F16 => Tensor::full(half::f16::from_f64(value), vec![], device),
        DType::BF16 => Tensor::full(half::bf16::from_f64(value), vec![], device),
        dtype => Err(candle_core::Error::Msg(format!(
            "scalar input not supported for dtype {dtype:?}"
        ))),
    }
}

#[napi]
impl CompiledProgram {
    #[napi(getter)]
    pub fn signature(&self) -> Result<String> {
        self.inner
            .as_ref()
            .map(|inner| inner.signature.clone())
            .ok_or_else(|| Error::new(Status::GenericFailure, "program is disposed".to_string()))
    }

    // Drops the frozen graphs; constant/parameter leaf buffers stay alive
    // through any NativeTensor handles that share them. Running a disposed
    // program is an error.
    #[napi]
    pub fn dispose(&mut self) {
        self.inner = None;
    }

    #[napi]
    pub async fn run(
        &self,
        inputs: Vec<&NativeTensor>,
        scalars: Vec<f64>,
        token: Option<&CancellationToken>,
    ) -> Result<Vec<NativeTensor>> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| Error::new(Status::GenericFailure, "program is disposed".to_string()))?;
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
            if got.dims() != declared.shape.as_slice()
                || got.dtype() != declared.dtype
                || device_key(got.device()) != device_key(&declared.device)
            {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!(
                        "input slot {slot}: expected {}, got {}:{}@{}",
                        declared.signature(),
                        got.dims()
                            .iter()
                            .map(|d| d.to_string())
                            .collect::<Vec<_>>()
                            .join("x"),
                        dtype_name(got.dtype()),
                        device_key(got.device())
                    ),
                ));
            }
        }
        let slots = inner.slots.clone();
        let roots = inner.roots.clone();
        let leaves = inner.leaves.clone();
        let inputs: Vec<Tensor> = inputs.iter().map(|input| input.inner.clone()).collect();
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
        let by_id: std::collections::HashMap<u64, Tensor> = leaves
            .iter()
            .map(|(id, slot)| (*id, bindings[&(*slot as u64)].clone()))
            .collect();
        let mut ev = Evaluator::with_slots(&roots, by_id);
            let walk_timing = std::env::var_os("EFFECT_TORCH_WALK_TIMING").is_some();
            let t1 = std::time::Instant::now();
            let mut outputs = Vec::with_capacity(roots.len());
            for node in &roots {
                let output = eval_node(node, cancelled, &mut ev).map_err(to_napi_err)?;
                outputs.push(NativeTensor::wrap(output));
            }
            // Synchronize once: per-root syncs would fully serialize CPU
            // encoding and GPU execution. Consumers that need values on
            // the host synchronize at readback; device-side reuse needs
            // no host round-trip.
            for output in &outputs {
                output.inner.device().synchronize().map_err(to_napi_err)?;
            }
            ev.run_ce_checks().map_err(to_napi_err)?;
            if walk_timing {
                eprintln!("[walk] program eval {:.1}us ({} roots)", t1.elapsed().as_micros(), roots.len());
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
        inner: Some(ProgramInner {
            roots: nodes,
            slots,
            leaves,
            signature,
        }),
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


#[cfg(test)]
mod kv_tests {
    use super::*;

    #[test]
    fn scatter_gather_roundtrip() {
        let device = Device::Cpu;
        let slab = Tensor::zeros((8, 2, 3), DType::F32, &device).unwrap();
        let src = Tensor::arange(0f32, 12f32, &device).unwrap().reshape((2, 2, 3)).unwrap();
        let idx = Tensor::from_vec(vec![4u32, 5u32], (2, 1, 1), &device)
            .unwrap()
            .broadcast_as((2, 2, 3))
            .unwrap()
            .contiguous()
            .unwrap();
        slab.scatter_set(&idx, &src, 0).unwrap();
        let got = slab.gather(&idx, 0).unwrap();
        assert_eq!(got.to_vec3::<f32>().unwrap(), src.to_vec3::<f32>().unwrap());
    }

    #[test]
    fn kv_attention_matches_sdpa() {
        let device = Device::Cpu;
        let pool = Arc::new(PoolInner {
            k: vec![Tensor::zeros((8, 2, 4), DType::F32, &device).unwrap()],
            v: vec![Tensor::zeros((8, 2, 4), DType::F32, &device).unwrap()],
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
        let q = Tensor::arange(0f32, 24f32, &device).unwrap().reshape((1, 2, 3, 4)).unwrap();
        let k = (Tensor::arange(24f32, 48f32, &device).unwrap() * 0.01).unwrap().reshape((1, 2, 3, 4)).unwrap();
        let v = (Tensor::arange(48f32, 72f32, &device).unwrap() * 0.01).unwrap().reshape((1, 2, 3, 4)).unwrap();
        state.lock().unwrap().advance = 3;
        let got = kv_attention(&kv, 0, &q, &k, &v, 0.5, None).unwrap();
        let want = sdpa_forward(&q, &k, &v, 0.5, true).unwrap();
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let want = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got, want);
        assert_eq!(state.lock().unwrap().advance, 3);
    }

    #[test]
    fn sdpa_single_token() {
        let device = Device::Cpu;
        let q = Tensor::arange(0f32, 8f32, &device).unwrap().reshape((1, 2, 1, 4)).unwrap();
        let k = Tensor::ones((1, 2, 1, 4), DType::F32, &device).unwrap();
        let v = Tensor::arange(8f32, 16f32, &device).unwrap().reshape((1, 2, 1, 4)).unwrap();
        let out = sdpa_forward(&q, &k, &v, 0.5, true).unwrap();
        let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let want = v.flatten_all().unwrap().to_vec1::<f32>().unwrap();
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

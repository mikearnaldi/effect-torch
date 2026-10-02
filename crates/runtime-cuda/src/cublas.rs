//! Row-major BF16 GEMM through cuBLAS with F32 accumulation.
//!
//! cuBLAS is column-major. A row-major GEMM out[m,n] = sum_k x[m,k] w[k,n] is
//! expressed as a column-major product of the transposed operands. The weight
//! operand is either a row-major [k,n] matrix or a row-major [n,k] matrix
//! viewed through a transpose, which is the row-oriented linearRows weight.
//! Both feed the same call through a different transpose operation and leading
//! dimension, so no weight copy is required.
//!
//! Products accumulate in F32 (CUBLAS_COMPUTE_32F), never TF32. BF16 results
//! use cuBLAS default math, which permits BF16 rounding of split-K partials
//! before the final reduction, matching the standard PyTorch BF16 GEMM path.
//! F32 accumulator outputs prohibit that partial rounding so the following
//! bias operation retains its single BF16 result boundary. The hardware suite
//! checks BF16 subnormals explicitly; there is no data-dependent widening fallback.

use cudarc::cublas::sys;
use cudarc::driver::CudaStream;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

#[path = "ordinary_k16.rs"]
pub(crate) mod ordinary_k16;

#[path = "expert_graphs.rs"]
pub(crate) mod expert_graphs;

/// Minimum compute capability with BF16 tensor-core GEMM.
pub(crate) const BF16_GEMM_MIN_MAJOR: i32 = 8;

/// Declared by each GEMM instruction, accounted in static diagnostics, and
/// reused by the invocation memory planner. A 1 MiB workspace excludes the
/// default split-K algorithms at larger projection shapes and changes BF16
/// results. 32 MiB matches the deterministic :4096:8 cuBLAS workspace.
pub(crate) const CUBLAS_WORKSPACE_BYTES: usize = 32 << 20;
pub(crate) const EXPERT_BLAS_STREAMS: usize = 32;
/// Three arrays of device pointers, independently planned from cuBLAS workspace.
pub(crate) const EXPERT_GROUPED_POINTER_BYTES: usize = 3 * 128 * 8;
pub(crate) const EXPERT_SPLITK_DESCRIPTOR_BYTES: usize = 128 * 64;
pub(crate) const EXPERT_SPLITK_MAX_SPLITS: usize = 8;
/// Three pointer arrays for up to eight slices of each of 128 experts.
pub(crate) const EXPERT_PARTIAL_POINTER_BYTES: usize = 3 * 128 * EXPERT_SPLITK_MAX_SPLITS * 8;

/// Which semantic operation the row-major GEMM realizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowGemmKind {
    Linear,
    Matmul,
}

/// Planned row-major GEMM geometry. All element counts fit the cuBLAS c_int
/// dimensions and are non-zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Bf16GemmPlan {
    /// Rows of the row-major activation matrix.
    pub(crate) m: usize,
    /// Columns of the row-major output and leading dimension of the weight.
    pub(crate) n: usize,
    /// Shared reduction width.
    pub(crate) k: usize,
    /// Strided batch count.
    pub(crate) batch: usize,
    /// Activation matrix stride in elements. Zero when broadcast.
    pub(crate) stride_x: usize,
    /// Weight matrix stride in elements. Zero when shared or broadcast.
    pub(crate) stride_weight: usize,
    /// Output matrix stride in elements.
    pub(crate) stride_out: usize,
}

/// Exact-size expert shapes for which grouped cuBLAS uses the same complete
/// F32 reduction as the ordinary call. One-row GEMV and larger first projections
/// use different reduction orders or BF16 split-K partials and stay ordinary.
pub(crate) fn supports_exact_grouped_expert(plan: Bf16GemmPlan) -> bool {
    plan.batch == 1
        && match (plan.n, plan.k) {
            (1408, 2816) => (2..=16).contains(&plan.m),
            (2816, 704) => (2..=128).contains(&plan.m),
            _ => false,
        }
}

fn product(values: &[usize]) -> Option<usize> {
    values
        .iter()
        .try_fold(1usize, |total, value| total.checked_mul(*value))
}

fn as_i32(value: usize) -> Option<i32> {
    i32::try_from(value).ok()
}

fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(rank);
    for position in 0..rank {
        let left = position
            .checked_sub(rank - a.len())
            .and_then(|index| a.get(index))
            .copied()
            .unwrap_or(1);
        let right = position
            .checked_sub(rank - b.len())
            .and_then(|index| b.get(index))
            .copied()
            .unwrap_or(1);
        if left != right && left != 1 && right != 1 {
            return None;
        }
        out.push(if left == 1 { right } else { left });
    }
    Some(out)
}

/// Plans a row-major BF16 GEMM for a supported Linear or Matmul.
///
/// x_shape is the activation operand, weight_shape is the logical [k, n]
/// weight, and out_shape is the logical result. Returns None when the shapes
/// require per-axis broadcasting that strided batched GEMM cannot express,
/// when a dimension is zero, or when a dimension exceeds c_int. Callers route
/// those cases through the explicit F32 legalization instead.
pub(crate) fn plan_row_bf16_gemm(
    kind: RowGemmKind,
    x_shape: &[usize],
    weight_shape: &[usize],
    out_shape: &[usize],
) -> Option<Bf16GemmPlan> {
    if x_shape.len() < 2 || weight_shape.len() < 2 || out_shape.len() < 2 {
        return None;
    }
    let m = x_shape[x_shape.len() - 2];
    let k = x_shape[x_shape.len() - 1];
    let n = weight_shape[weight_shape.len() - 1];
    if weight_shape[weight_shape.len() - 2] != k
        || out_shape[out_shape.len() - 2] != m
        || out_shape[out_shape.len() - 1] != n
    {
        return None;
    }
    if m == 0 || n == 0 || k == 0 {
        return None;
    }
    as_i32(m)?;
    as_i32(n)?;
    as_i32(k)?;
    let matrix_x = m.checked_mul(k)?;
    let matrix_weight = k.checked_mul(n)?;
    let matrix_out = m.checked_mul(n)?;
    let x_leading = &x_shape[..x_shape.len() - 2];
    let weight_leading = &weight_shape[..weight_shape.len() - 2];
    let out_leading = &out_shape[..out_shape.len() - 2];
    let (batch, stride_x, stride_weight) = match kind {
        RowGemmKind::Linear => {
            if weight_shape.len() != 2 || x_leading != out_leading {
                return None;
            }
            (product(x_leading)?, matrix_x, 0)
        }
        RowGemmKind::Matmul => {
            let expected = broadcast_shapes(x_leading, weight_leading)?;
            if expected != out_leading {
                return None;
            }
            let batch = product(expected.as_slice())?;
            let stride_x = if product(x_leading)? == batch {
                matrix_x
            } else if x_leading.iter().all(|dim| *dim == 1) {
                0
            } else {
                return None;
            };
            let stride_weight = if product(weight_leading)? == batch {
                matrix_weight
            } else if weight_leading.iter().all(|dim| *dim == 1) {
                0
            } else {
                return None;
            };
            (batch, stride_x, stride_weight)
        }
    };
    if batch == 0 {
        return None;
    }
    as_i32(batch)?;
    if stride_x > i64::MAX as usize
        || stride_weight > i64::MAX as usize
        || matrix_out > i64::MAX as usize
    {
        return None;
    }
    // All pointer arithmetic, including the last batch, must fit in a byte
    // address. Four bytes also covers the optional F32 accumulator output.
    product(x_shape)?.checked_mul(2)?;
    product(weight_shape)?.checked_mul(2)?;
    product(out_shape)?.checked_mul(4)?;
    Some(Bf16GemmPlan {
        m,
        n,
        k,
        batch,
        stride_x,
        stride_weight,
        stride_out: matrix_out,
    })
}

struct BlasHandle(sys::cublasHandle_t);

// SAFETY: the handle is accessed only under CudaBlas::handle, with its CUDA
// context bound on the calling thread. All calls use the same stream.
unsafe impl Send for BlasHandle {}

impl Drop for BlasHandle {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cublasDestroy_v2(self.0);
        }
    }
}

/// Per-device cuBLAS handle. Calls and workspace changes are serialized.
pub(crate) struct CudaBlas {
    handle: Mutex<BlasHandle>,
    stream: Arc<CudaStream>,
    pub(crate) version: i32,
    sequence_graphs: Mutex<expert_graphs::Cache>,
    ordinary_k16: Option<Arc<ordinary_k16::OrdinaryK16>>,
    attention75: Option<Arc<crate::attention75::Attention75>>,
}

impl CudaBlas {
    pub(crate) fn new(stream: Arc<CudaStream>) -> Result<Self, String> {
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        let mut handle = std::mem::MaybeUninit::uninit();
        unsafe { sys::cublasCreate_v2(handle.as_mut_ptr()) }
            .result()
            .map_err(|error| format!("CUDA cuBLAS handle creation failed: {error}"))?;
        // Own the handle before any fallible setup so every error destroys it.
        let handle = BlasHandle(unsafe { handle.assume_init() });
        unsafe { sys::cublasSetStream_v2(handle.0, stream.cu_stream() as _) }
            .result()
            .map_err(|error| format!("CUDA cuBLAS stream binding failed: {error}"))?;
        unsafe {
            sys::cublasSetPointerMode_v2(
                handle.0,
                sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
            )
        }
        .result()
        .map_err(|error| format!("CUDA cuBLAS pointer-mode setup failed: {error}"))?;
        let mut version = 0;
        unsafe { sys::cublasGetVersion_v2(handle.0, &mut version) }
            .result()
            .map_err(|error| format!("CUDA cuBLAS version query failed: {error}"))?;
        let attention75 = crate::attention75::Attention75::from_env(stream.context())?;
        Ok(Self {
            handle: Mutex::new(handle),
            stream,
            version,
            sequence_graphs: Mutex::new(expert_graphs::Cache::default()),
            ordinary_k16: None,
            attention75,
        })
    }

    pub(crate) fn set_ordinary_k16(&mut self, kernel: Option<Arc<ordinary_k16::OrdinaryK16>>) {
        self.ordinary_k16 = kernel;
    }

    pub(crate) fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub(crate) fn attention75(&self) -> Option<&crate::attention75::Attention75> {
        self.attention75.as_deref()
    }

    /// Creates an independent handle for a graph-owned execution frame on the
    /// same context and stream, preserving the immutable GEMM dispatch policy.
    /// Mutable workspace bindings and sequence-graph caches are never shared.
    /// The frame must retain this handle through its final checked completion.
    pub(crate) fn fork_for_graph(&self) -> Result<Self, String> {
        let mut graph_blas = Self::new(self.stream.clone())?;
        if graph_blas.version != self.version {
            return Err("CUDA graph cuBLAS version differs from the source handle".into());
        }
        graph_blas.ordinary_k16 = self.ordinary_k16.clone();
        graph_blas.attention75 = self.attention75.clone();
        // gemm_bf16 selects the exact output-specific math mode under this
        // handle's own lock on every call; transient source-handle state is
        // intentionally not copied into the new frame.
        Ok(graph_blas)
    }

    /// Executes one row-major BF16 GEMM with F32 accumulation.
    ///
    /// weight_transposed selects a row-oriented [n, k] weight that is consumed
    /// directly, without a transpose or copy. out_is_f32 selects an F32
    /// accumulator so a following bias step can round once to BF16.
    ///
    /// # Safety
    /// Pointers must refer to correctly sized, device-local allocations retained
    /// until the stream finishes. Workspace must have CUBLAS_WORKSPACE_BYTES
    /// bytes, 256-byte alignment, and no overlapping use on another stream.
    pub(crate) unsafe fn gemm_bf16(
        &self,
        plan: Bf16GemmPlan,
        weight_transposed: bool,
        x: u64,
        weight: u64,
        out: u64,
        out_is_f32: bool,
        workspace: u64,
    ) -> Result<(), String> {
        self.stream
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        if let Some(kernel) = &self.ordinary_k16 {
            if ordinary_k16::supports(plan, weight_transposed, out_is_f32, [x, weight, out]) {
                // SAFETY: the same caller-owned buffers/fence cover this exact
                // fixed-shape dispatch, on the same stream as ordinary cuBLAS.
                return unsafe { kernel.launch(&self.stream, plan, x, weight, out) };
            }
        }
        let handle = self
            .handle
            .lock()
            .map_err(|_| "CUDA cuBLAS handle lock poisoned")?;
        // Select the reduction contract for this output under the same lock
        // as workspace binding and submission. A preceding F32 bias GEMM must
        // not leave its stricter mode active for a subsequent BF16 result.
        let math_mode = if out_is_f32 {
            sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION
        } else {
            sys::cublasMath_t::CUBLAS_DEFAULT_MATH
        };
        unsafe { sys::cublasSetMathMode(handle.0, math_mode) }
            .result()
            .map_err(|error| format!("CUDA cuBLAS math-mode setup failed: {error}"))?;
        unsafe {
            sys::cublasSetWorkspace_v2(handle.0, workspace as *mut c_void, CUBLAS_WORKSPACE_BYTES)
        }
        .result()
        .map_err(|error| format!("CUDA cuBLAS workspace setup failed: {error}"))?;
        // cuBLAS computes C_cb[N,M] = op(A) op(B) with C_cb = out^T.
        let m = as_i32(plan.n).ok_or("CUDA cuBLAS dimension N exceeds i32")?;
        let n = as_i32(plan.m).ok_or("CUDA cuBLAS dimension M exceeds i32")?;
        let k = as_i32(plan.k).ok_or("CUDA cuBLAS dimension K exceeds i32")?;
        let (transa, lda) = if weight_transposed {
            (sys::cublasOperation_t::CUBLAS_OP_T, k)
        } else {
            (sys::cublasOperation_t::CUBLAS_OP_N, m)
        };
        let ldb = k;
        let ldc = m;
        let c_type = if out_is_f32 {
            sys::cudaDataType_t::CUDA_R_32F
        } else {
            sys::cudaDataType_t::CUDA_R_16BF
        };
        let batch = as_i32(plan.batch).ok_or("CUDA cuBLAS batch count exceeds i32")?;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = unsafe {
            sys::cublasGemmStridedBatchedEx(
                handle.0,
                transa,
                sys::cublasOperation_t::CUBLAS_OP_N,
                m,
                n,
                k,
                (&alpha) as *const f32 as *const c_void,
                weight as *const c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                lda,
                plan.stride_weight as i64,
                x as *const c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                ldb,
                plan.stride_x as i64,
                (&beta) as *const f32 as *const c_void,
                out as *mut c_void,
                c_type,
                ldc,
                plan.stride_out as i64,
                batch,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        };
        status
            .result()
            .map_err(|error| format!("CUDA cuBLAS BF16 GEMM failed: {error}"))
    }

    /// Executes independent row-major BF16 GEMMs while retaining one handle
    /// lock, math-mode setup, and workspace binding for the sequence.
    ///
    /// # Safety
    /// Every pointer must refer to the geometry in its plan and remain valid
    /// until the stream completes. Workspace follows [`Self::gemm_bf16`].
    pub(crate) unsafe fn gemm_bf16_sequence(
        &self,
        groups: &[(Bf16GemmPlan, u64, u64, u64)],
        workspace: u64,
    ) -> Result<(), String> {
        unsafe { self.gemm_bf16_sequence_impl(groups, workspace, true) }
    }
    pub(crate) unsafe fn gemm_bf16_sequence_eager(
        &self,
        groups: &[(Bf16GemmPlan, u64, u64, u64)],
        workspace: u64,
    ) -> Result<(), String> {
        unsafe { self.gemm_bf16_sequence_impl(groups, workspace, false) }
    }
    unsafe fn gemm_bf16_sequence_impl(
        &self,
        groups: &[(Bf16GemmPlan, u64, u64, u64)],
        workspace: u64,
        allow_graphs: bool,
    ) -> Result<(), String> {
        if groups.is_empty() {
            return Ok(());
        }
        self.stream
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        let handle = self
            .handle
            .lock()
            .map_err(|_| "CUDA cuBLAS handle lock poisoned")?;
        let submit = || {
            unsafe { sys::cublasSetMathMode(handle.0, sys::cublasMath_t::CUBLAS_DEFAULT_MATH) }
                .result()
                .map_err(|error| format!("CUDA cuBLAS math-mode setup failed: {error}"))?;
            unsafe {
                sys::cublasSetWorkspace_v2(
                    handle.0,
                    workspace as *mut c_void,
                    CUBLAS_WORKSPACE_BYTES,
                )
            }
            .result()
            .map_err(|error| format!("CUDA cuBLAS workspace setup failed: {error}"))?;
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            for (plan, x, weight, out) in groups {
                let m = as_i32(plan.n).ok_or("CUDA cuBLAS dimension N exceeds i32")?;
                let n = as_i32(plan.m).ok_or("CUDA cuBLAS dimension M exceeds i32")?;
                let k = as_i32(plan.k).ok_or("CUDA cuBLAS dimension K exceeds i32")?;
                let status = unsafe {
                    sys::cublasGemmStridedBatchedEx(
                        handle.0,
                        sys::cublasOperation_t::CUBLAS_OP_T,
                        sys::cublasOperation_t::CUBLAS_OP_N,
                        m,
                        n,
                        k,
                        (&alpha) as *const f32 as *const c_void,
                        *weight as *const c_void,
                        sys::cudaDataType_t::CUDA_R_16BF,
                        k,
                        0,
                        *x as *const c_void,
                        sys::cudaDataType_t::CUDA_R_16BF,
                        k,
                        plan.stride_x as i64,
                        (&beta) as *const f32 as *const c_void,
                        *out as *mut c_void,
                        sys::cudaDataType_t::CUDA_R_16BF,
                        m,
                        plan.stride_out as i64,
                        1,
                        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                };
                status
                    .result()
                    .map_err(|error| format!("CUDA cuBLAS BF16 GEMM failed: {error}"))?;
            }
            Ok(())
        };
        if allow_graphs
            && std::env::var("EFFECT_TORCH_CUDA_EXPERT_SEQUENCE_GRAPHS")
                .is_ok_and(|value| value == "1")
        {
            self.sequence_graphs
                .lock()
                .map_err(|_| "ordinary expert graph cache lock poisoned")?
                .execute(
                    &self.stream,
                    expert_graphs::Key::new(groups, workspace),
                    submit,
                )
        } else {
            submit()
        }
    }

    /// Submit exact-size expert groups in one persistent grouped cuBLAS kernel.
    /// Device pointer arrays have weights, inputs, and outputs in plan order;
    /// each plan describes one matrix, without padding or arithmetic changes.
    ///
    /// # Safety
    /// `device_pointers` must hold three device arrays of `plans.len()` pointers,
    /// ordered weight/input/output. All matrices and pointer arrays remain alive
    /// until this stream completes. Pointer storage must not alias workspace.
    /// Workspace follows [`Self::gemm_bf16`].
    pub(crate) unsafe fn gemm_bf16_grouped(
        &self,
        plans: &[Bf16GemmPlan],
        device_pointers: [u64; 3],
        workspace: u64,
    ) -> Result<(), String> {
        if plans.is_empty() {
            return Ok(());
        }
        if plans.len() > 128
            || plans
                .iter()
                .any(|plan| !supports_exact_grouped_expert(*plan))
        {
            return Err("CUDA grouped cuBLAS geometry exceeds exact expert subset".into());
        }
        self.stream
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        let handle = self
            .handle
            .lock()
            .map_err(|_| "CUDA cuBLAS handle lock poisoned")?;
        unsafe { sys::cublasSetMathMode(handle.0, sys::cublasMath_t::CUBLAS_DEFAULT_MATH) }
            .result()
            .map_err(|error| format!("CUDA cuBLAS math-mode setup failed: {error}"))?;
        unsafe {
            sys::cublasSetWorkspace_v2(handle.0, workspace as *mut c_void, CUBLAS_WORKSPACE_BYTES)
        }
        .result()
        .map_err(|error| format!("CUDA cuBLAS workspace setup failed: {error}"))?;
        // Host metadata is consumed by cuBLAS during submission. Fixed capacity
        // keeps this operation allocation-free; matrix pointer banks are planned
        // invocation scratch, uploaded by the caller on this stream.
        let transa = [sys::cublasOperation_t::CUBLAS_OP_T; 128];
        let transb = [sys::cublasOperation_t::CUBLAS_OP_N; 128];
        let mut m = [0i32; 128];
        let mut n = [0i32; 128];
        let mut k = [0i32; 128];
        let alpha = [1f32; 128];
        let beta = [0f32; 128];
        let group_size = [1i32; 128];
        for (index, plan) in plans.iter().enumerate() {
            m[index] = as_i32(plan.n).ok_or("CUDA grouped cuBLAS dimension N exceeds i32")?;
            n[index] = as_i32(plan.m).ok_or("CUDA grouped cuBLAS dimension M exceeds i32")?;
            k[index] = as_i32(plan.k).ok_or("CUDA grouped cuBLAS dimension K exceeds i32")?;
        }
        unsafe {
            sys::cublasGemmGroupedBatchedEx(
                handle.0,
                transa.as_ptr(),
                transb.as_ptr(),
                m.as_ptr(),
                n.as_ptr(),
                k.as_ptr(),
                alpha.as_ptr() as *const c_void,
                device_pointers[0] as *const *const c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                k.as_ptr(),
                device_pointers[1] as *const *const c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                k.as_ptr(),
                beta.as_ptr() as *const c_void,
                device_pointers[2] as *const *mut c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                m.as_ptr(),
                plans.len() as i32,
                group_size.as_ptr(),
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            )
        }
        .result()
        .map_err(|error| format!("CUDA grouped cuBLAS BF16 GEMM failed: {error}"))
    }
    /// Device pointer arrays have weights, inputs, and outputs in plan order;
    /// each plan describes one matrix, without padding or arithmetic changes.
    ///
    /// # Safety
    /// `device_pointers` must hold three device arrays of `plans.len()` pointers,
    /// ordered weight/input/output. The second plan field is the original
    /// row stride; plan.k is the width of this split-K slice. Each partial is
    /// rounded to BF16 before the caller performs the ordered reduction.
    /// All matrices and pointer arrays remain alive
    /// until this stream completes. Pointer storage must not alias workspace.
    /// Workspace follows [`Self::gemm_bf16`].
    pub(crate) unsafe fn gemm_bf16_grouped_partials(
        &self,
        plans: &[(Bf16GemmPlan, usize)],
        device_pointers: [u64; 3],
        workspace: u64,
    ) -> Result<(), String> {
        if plans.is_empty() {
            return Ok(());
        }
        if plans.len() > 1024
            || plans.iter().any(|(plan, leading)| {
                plan.batch != 1
                    || !(2..=512).contains(&plan.m)
                    || plan.k == 0
                    || plan.k > *leading
                    || !matches!((plan.n, *leading), (1408, 2816) | (2816, 704))
            })
        {
            return Err("CUDA grouped partial geometry exceeds mapped expert shapes".into());
        }
        self.stream
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        let handle = self
            .handle
            .lock()
            .map_err(|_| "CUDA cuBLAS handle lock poisoned")?;
        unsafe { sys::cublasSetMathMode(handle.0, sys::cublasMath_t::CUBLAS_DEFAULT_MATH) }
            .result()
            .map_err(|error| format!("CUDA cuBLAS math-mode setup failed: {error}"))?;
        unsafe {
            sys::cublasSetWorkspace_v2(handle.0, workspace as *mut c_void, CUBLAS_WORKSPACE_BYTES)
        }
        .result()
        .map_err(|error| format!("CUDA cuBLAS workspace setup failed: {error}"))?;
        // Host metadata is consumed by cuBLAS during submission. Fixed capacity
        // keeps this operation allocation-free; matrix pointer banks are planned
        // invocation scratch, uploaded by the caller on this stream.
        let transa = [sys::cublasOperation_t::CUBLAS_OP_T; 1024];
        let transb = [sys::cublasOperation_t::CUBLAS_OP_N; 1024];
        let mut m = [0i32; 1024];
        let mut n = [0i32; 1024];
        let mut k = [0i32; 1024];
        let mut leading = [0i32; 1024];
        let alpha = [1f32; 1024];
        let beta = [0f32; 1024];
        let mut group_size = [1i32; 1024];
        let merge = std::env::var("EFFECT_TORCH_CUDA_EXPERT_PARTIAL_GROUPS")
            .is_ok_and(|value| value == "1");
        let mut group_count = 0;
        for (position, (plan, row_stride)) in plans.iter().enumerate() {
            if merge && position > 0 && plans[position - 1] == plans[position] {
                group_size[group_count - 1] += 1;
                continue;
            }
            let index = group_count;
            group_count += 1;
            m[index] = as_i32(plan.n).ok_or("CUDA grouped cuBLAS dimension N exceeds i32")?;
            n[index] = as_i32(plan.m).ok_or("CUDA grouped cuBLAS dimension M exceeds i32")?;
            k[index] = as_i32(plan.k).ok_or("CUDA grouped cuBLAS dimension K exceeds i32")?;
            leading[index] =
                as_i32(*row_stride).ok_or("CUDA grouped partial leading dimension exceeds i32")?;
        }
        unsafe {
            sys::cublasGemmGroupedBatchedEx(
                handle.0,
                transa.as_ptr(),
                transb.as_ptr(),
                m.as_ptr(),
                n.as_ptr(),
                k.as_ptr(),
                alpha.as_ptr() as *const c_void,
                device_pointers[0] as *const *const c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                leading.as_ptr(),
                device_pointers[1] as *const *const c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                leading.as_ptr(),
                beta.as_ptr() as *const c_void,
                device_pointers[2] as *const *mut c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                m.as_ptr(),
                group_count as i32,
                group_size.as_ptr(),
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            )
        }
        .result()
        .map_err(|error| format!("CUDA grouped cuBLAS BF16 GEMM failed: {error}"))
    }
}

impl Drop for CudaBlas {
    fn drop(&mut self) {
        // Fields drop in declaration order, so the context outlives the handle.
        let _ = self.stream.context().bind_to_thread();
    }
}

#[cfg(test)]
mod sequence_graph_tests {
    use super::*;
    use crate::{CudaDevice, CudaValue};
    use cudarc::driver::DevicePtr;
    use effect_torch_runtime::DType;

    #[test]
    #[ignore = "requires pinned CUDA device and EFFECT_TORCH_CUDA_ORDINARY_K16_PTX"]
    fn ordinary_k16_runtime_dispatch_matches_cublas_exactly() {
        let device = CudaDevice::get(0).unwrap();
        assert!(
            device.cublas.ordinary_k16.is_some(),
            "ordinary kernel must be loaded"
        );
        let reference = CudaBlas::new(device.stream.clone()).unwrap();
        assert!(reference.ordinary_k16.is_none());
        let workspace = unsafe { device.stream.alloc::<u8>(CUBLAS_WORKSPACE_BYTES) }.unwrap();
        let (workspace_address, guard) = workspace.device_ptr(&device.stream);
        for n in [2048, 2112] {
            let plan = Bf16GemmPlan {
                m: 256,
                n,
                k: 2816,
                batch: 1,
                stride_x: 256 * 2816,
                stride_weight: 0,
                stride_out: 256 * n,
            };
            for seed in [17u32, 313] {
                for special in [false, true] {
                    let values = |count: usize| {
                        (0..count)
                            .map(|i| {
                                let h = (i as u32).wrapping_mul(1664525).wrapping_add(seed);
                                let bits = if special {
                                    [
                                        0u16, 0x8000, 1, 0x8001, 0x3f80, 0xbf80, 0x7f80, 0xff80,
                                        0x7fc1, 0x7f7f,
                                    ][(h as usize) % 10]
                                } else {
                                    (((h >> 16) & 0x8000)
                                        | ((((h >> 24) % 9) + 120) << 7)
                                        | ((h >> 8) & 127))
                                        as u16
                                };
                                half::bf16::from_bits(bits).to_f64()
                            })
                            .collect::<Vec<_>>()
                    };
                    let x = CudaValue::from_host(
                        device.clone(),
                        vec![256, 2816],
                        DType::BF16,
                        &values(256 * 2816),
                    )
                    .unwrap();
                    let w = CudaValue::from_host(
                        device.clone(),
                        vec![n, 2816],
                        DType::BF16,
                        &values(n * 2816),
                    )
                    .unwrap();
                    let output = || {
                        CudaValue::from_host(
                            device.clone(),
                            vec![256, n],
                            DType::BF16,
                            &vec![0.; 256 * n],
                        )
                        .unwrap()
                    };
                    let actual = output();
                    let expected = output();
                    assert!(ordinary_k16::supports(
                        plan,
                        true,
                        false,
                        [
                            x.storage_address(),
                            w.storage_address(),
                            actual.storage_address()
                        ]
                    ));
                    unsafe {
                        reference
                            .gemm_bf16(
                                plan,
                                true,
                                x.storage_address(),
                                w.storage_address(),
                                expected.storage_address(),
                                false,
                                workspace_address,
                            )
                            .unwrap();
                        device
                            .cublas
                            .gemm_bf16(
                                plan,
                                true,
                                x.storage_address(),
                                w.storage_address(),
                                actual.storage_address(),
                                false,
                                workspace_address,
                            )
                            .unwrap();
                    }
                    device.stream.synchronize().unwrap();
                    let a = device.stream.clone_dtoh(actual.buffer.as_ref()).unwrap();
                    let b = device.stream.clone_dtoh(expected.buffer.as_ref()).unwrap();
                    assert_eq!(a, b, "n={n} seed={seed} special={special}");
                }
            }
        }
        drop(guard);
    }

    #[test]
    #[ignore = "requires CUDA and EFFECT_TORCH_CUDA_EXPERT_SEQUENCE_GRAPHS=1"]
    fn ordinary_sequence_graphs_recompute_mutated_inputs_exactly() {
        assert!(std::env::var("EFFECT_TORCH_CUDA_EXPERT_SEQUENCE_GRAPHS").is_ok_and(|v| v == "1"));
        let device = CudaDevice::get(0).unwrap();
        let blas = CudaBlas::new(device.stream.clone()).unwrap();
        let workspace = unsafe { device.stream.alloc::<u8>(CUBLAS_WORKSPACE_BYTES) }.unwrap();
        let (workspace_address, guard) = workspace.device_ptr(&device.stream);
        for (m, n, k) in [(17, 1408, 2816), (33, 1408, 2816), (1, 2816, 704)] {
            let values = |count: usize, seed: usize| {
                (0..count)
                    .map(|i| {
                        let bits = i.wrapping_mul(1664525).wrapping_add(seed);
                        (((bits >> 8) % 255) as f64 - 127.0)
                            * 2.0f64.powi(((bits >> 19) % 15) as i32 - 12)
                    })
                    .collect::<Vec<_>>()
            };
            let x =
                CudaValue::from_host(device.clone(), vec![m, k], DType::BF16, &values(m * k, 17))
                    .unwrap();
            let w =
                CudaValue::from_host(device.clone(), vec![n, k], DType::BF16, &values(n * k, 313))
                    .unwrap();
            let output = || {
                CudaValue::from_host(device.clone(), vec![m, n], DType::BF16, &vec![0.0; m * n])
                    .unwrap()
            };
            let actual = output();
            let expected = output();
            let plan = Bf16GemmPlan {
                m,
                n,
                k,
                batch: 1,
                stride_x: m * k,
                stride_weight: n * k,
                stride_out: m * n,
            };
            for iteration in 0..4 {
                let bytes = crate::value::dense_bytes_from_host(
                    &values(m * k, 17 + iteration * 99991),
                    DType::BF16,
                );
                let mut buffer = x.buffer.as_ref().clone();
                device.stream.memcpy_htod(&bytes, &mut buffer).unwrap();
                unsafe {
                    blas.gemm_bf16(
                        plan,
                        true,
                        x.storage_address(),
                        w.storage_address(),
                        expected.storage_address(),
                        false,
                        workspace_address,
                    )
                    .unwrap();
                    blas.gemm_bf16_sequence(
                        &[(
                            plan,
                            x.storage_address(),
                            w.storage_address(),
                            actual.storage_address(),
                        )],
                        workspace_address,
                    )
                    .unwrap();
                }
                assert_eq!(
                    actual.readback().unwrap(),
                    expected.readback().unwrap(),
                    "shape {m},{n},{k}, iteration {iteration}"
                );
            }
        }
        device.stream.synchronize().unwrap();
        let cache = blas.sequence_graphs.lock().unwrap();
        assert_eq!(cache.captures, 3);
        assert_eq!(cache.hits, 6);
        drop(guard);
    }
}

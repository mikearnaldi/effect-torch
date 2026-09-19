//! Row-major BF16 GEMM through cuBLAS with F32 accumulation.
//!
//! cuBLAS is column-major. A row-major GEMM out[m,n] = sum_k x[m,k] w[k,n] is
//! expressed as a column-major product of the transposed operands. The weight
//! operand is either a row-major [k,n] matrix or a row-major [n,k] matrix
//! viewed through a transpose, which is the row-oriented linearRows weight.
//! Both feed the same call through a different transpose operation and leading
//! dimension, so no weight copy is required.
//!
//! Inputs and outputs are BF16. Accumulation is F32 (CUBLAS_COMPUTE_32F),
//! never TF32 or a reduced-precision compute type. The hardware suite checks
//! BF16 subnormals explicitly; there is no data-dependent widening fallback.

use cudarc::cublas::sys;
use cudarc::driver::CudaStream;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

/// Minimum compute capability with BF16 tensor-core GEMM.
pub(crate) const BF16_GEMM_MIN_MAJOR: i32 = 8;

/// Declared by each GEMM instruction and reused by the memory planner.
pub(crate) const CUBLAS_WORKSPACE_BYTES: usize = 1 << 20;

/// Which semantic operation the row-major GEMM realizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowGemmKind {
    Linear,
    Matmul,
}

/// Planned row-major GEMM geometry. All element counts fit the cuBLAS c_int
/// dimensions and are non-zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
        // This flag adds to DEFAULT_MATH (zero). BF16 output must not let
        // split-K algorithms truncate partial reductions back to BF16.
        unsafe {
            sys::cublasSetMathMode(
                handle.0,
                sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
            )
        }
        .result()
        .map_err(|error| format!("CUDA cuBLAS math-mode setup failed: {error}"))?;
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
        Ok(Self {
            handle: Mutex::new(handle),
            stream,
            version,
        })
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
        let handle = self
            .handle
            .lock()
            .map_err(|_| "CUDA cuBLAS handle lock poisoned")?;
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
}

impl Drop for CudaBlas {
    fn drop(&mut self) {
        // Fields drop in declaration order, so the context outlives the handle.
        let _ = self.stream.context().bind_to_thread();
    }
}

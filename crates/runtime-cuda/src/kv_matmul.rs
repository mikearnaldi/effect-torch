//! Stepwise BF16 KV attention with F32-only cuBLAS dot accumulation.
//! Storage is worst-case planned; GEMM dimensions use actual retained rows.
use crate::cublas::{Bf16GemmPlan, CudaBlas, CUBLAS_WORKSPACE_BYTES};
use crate::executable::CudaKernelArgs;
use crate::CudaDevice;

#[derive(Clone, Copy, Debug)]
pub(crate) struct KvMatmulWorkspace {
    pub(crate) vnorm_store: Option<crate::vnorm_store::VNormStorePlan>,
    q: usize,
    k: usize,
    v: usize,
    probabilities: usize,
    output: usize,
    blas: usize,
    pub(crate) bytes: usize,
}
impl KvMatmulWorkspace {
    pub(crate) fn new(
        base: usize,
        heads: usize,
        tokens: usize,
        positions: usize,
        dim: usize,
    ) -> Option<Self> {
        for n in [heads, tokens, positions, dim] {
            if n == 0 || i32::try_from(n).is_err() {
                return None;
            }
        }
        let matrix = heads
            .checked_mul(positions)?
            .checked_mul(dim)?
            .checked_mul(2)?;
        let query = heads
            .checked_mul(tokens)?
            .checked_mul(dim)?
            .checked_mul(2)?;
        let probs = heads
            .checked_mul(tokens)?
            .checked_mul(positions)?
            .checked_mul(2)?;
        let mut end = base;
        let mut reserve = |bytes: usize| -> Option<usize> {
            let start = end.checked_add(255)? / 256 * 256;
            end = start.checked_add(bytes)?;
            Some(start)
        };
        let q = reserve(query)?;
        let k = reserve(matrix)?;
        let v = reserve(matrix)?;
        let probabilities = reserve(probs)?;
        let output = reserve(query.checked_mul(2)?)?;
        let blas = reserve(CUBLAS_WORKSPACE_BYTES)?;
        Some(Self {
            vnorm_store: None,
            q,
            k,
            v,
            probabilities,
            output,
            blas,
            bytes: end,
        })
    }
}

/// The caller retains the planned buffers and state transaction until completion.
/// All lengths are derived from the same validated host metadata as the row table.
pub(crate) fn execute(
    device: &CudaDevice,
    args: &CudaKernelArgs,
    workspace: KvMatmulWorkspace,
    ranges: &[(usize, usize, usize)],
    launch: impl FnMut(&str, &CudaKernelArgs) -> Result<(), String>,
) -> Result<(), String> {
    execute_with_blas(&device.cublas, args, workspace, ranges, launch)
}

/// Executes the unchanged exact QK/PV sequence with an explicitly owned cuBLAS
/// handle. The launch closure must use that handle's stream, and the caller
/// retains the handle, planned buffers and state transaction through completion.
pub(crate) fn execute_with_blas(
    blas: &CudaBlas,
    args: &CudaKernelArgs,
    workspace: KvMatmulWorkspace,
    ranges: &[(usize, usize, usize)],
    mut launch: impl FnMut(&str, &CudaKernelArgs) -> Result<(), String>,
) -> Result<(), String> {
    let output_width = output_bytes(args.output_dtype)?;
    let mut zero = *args;
    zero.scalars[0] = 0.;
    launch("et_fill", &zero)?;
    if args.elements == 0 {
        return Ok(());
    }
    if let Some(plan) = workspace.vnorm_store {
        let mut store = *args;
        store.integers[12] = plan.tokens as u64;
        store.integers[13] = plan.heads as u64;
        store.integers[14] = plan.dim as u64;
        store.scalars[1] = plan.eps;
        launch("et_vnorm_store56", &store)?;
    } else {
        launch("et_kv_store", args)?;
    }
    let (heads, declared_tokens, dim) = (
        args.integers[11] as usize,
        args.integers[7] as usize,
        args.integers[10] as usize,
    );
    let base = args.scratch[2];
    for &(lane, positions, tokens) in ranges {
        if positions == 0
            || positions > args.integers[9] as usize
            || tokens == 0
            || tokens > declared_tokens
        {
            return Err("execute: invalid KV GEMM range".into());
        }
        if let Some(fused) = blas.attention75().filter(|fused| fused.supports(args)) {
            // The original fill/store above and invocation fence remain in
            // force. Only the readonly attention contraction is replaced.
            unsafe { fused.launch(blas.stream(), args, lane, positions, tokens)? };
            continue;
        }
        let mut a = *args;
        a.inputs[5] = base + workspace.q as u64;
        a.inputs[6] = base + workspace.k as u64;
        a.scratch[1] = base + workspace.v as u64;
        a.inputs[1] = base + workspace.probabilities as u64;
        a.inputs[2] = base + workspace.output as u64;
        a.integers[13] = positions as u64;
        a.integers[14] = lane as u64;
        a.integers[15] = tokens as u64;
        a.output += (lane * heads * declared_tokens * dim * output_width) as u64;
        a.elements = (heads * dim * positions.max(tokens)) as u64;
        launch("et_kv_gemm_gather", &a)?;
        let qk = Bf16GemmPlan {
            m: tokens,
            n: positions,
            k: dim,
            batch: heads,
            stride_x: tokens * dim,
            stride_weight: positions * dim,
            stride_out: tokens * positions,
        };
        // F32 outputs select DISALLOW_REDUCED_PRECISION_REDUCTION under the
        // existing handle lock. Only the explicit score/probability/PV boundaries narrow.
        unsafe {
            blas.gemm_bf16(
                qk,
                true,
                a.inputs[5],
                a.inputs[6],
                a.inputs[4],
                true,
                base + workspace.blas as u64,
            )?;
        }
        a.elements = (heads * tokens * 32) as u64;
        launch("et_kv_gemm_softmax", &a)?;
        let pv = Bf16GemmPlan {
            m: tokens,
            n: dim,
            k: positions,
            batch: heads,
            stride_x: tokens * positions,
            stride_weight: positions * dim,
            stride_out: tokens * dim,
        };
        unsafe {
            blas.gemm_bf16(
                pv,
                false,
                a.inputs[1],
                a.scratch[1],
                a.inputs[2],
                true,
                base + workspace.blas as u64,
            )?;
        }
        a.elements = (heads * tokens * dim) as u64;
        launch("et_kv_gemm_round", &a)?;
    }
    Ok(())
}

fn output_bytes(dtype: u32) -> Result<usize, String> {
    match dtype {
        1 => Ok(4),
        3 => Ok(2),
        _ => Err("KV GEMM output must be F32 or BF16".into()),
    }
}

#[cfg(test)]
mod output82_tests {
    #[test]
    fn fallback_uses_actual_output_storage_width() {
        assert_eq!(super::output_bytes(1).unwrap(), 4);
        assert_eq!(super::output_bytes(3).unwrap(), 2);
        for dtype in [0, 2, 4, 5, 6] {
            assert!(super::output_bytes(dtype).is_err());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checked_aligned_workspace_includes_all_matrices_and_blas() {
        let p = KvMatmulWorkspace::new(19, 16, 256, 790, 512).unwrap();
        let ranges = [
            (p.q, 16 * 256 * 512 * 2),
            (p.k, 16 * 790 * 512 * 2),
            (p.v, 16 * 790 * 512 * 2),
            (p.probabilities, 16 * 256 * 790 * 2),
            (p.output, 16 * 256 * 512 * 4),
            (p.blas, CUBLAS_WORKSPACE_BYTES),
        ];
        let mut end = 19;
        for (offset, bytes) in ranges {
            assert_eq!(offset % 256, 0);
            assert!(offset >= end);
            end = offset + bytes;
        }
        assert_eq!(end, p.bytes);
        assert!(KvMatmulWorkspace::new(usize::MAX, 16, 256, 790, 512).is_none());
        assert!(KvMatmulWorkspace::new(0, 16, 256, usize::MAX, 512).is_none());
        assert!(KvMatmulWorkspace::new(0, 16, 0, 790, 512).is_none());
    }
}

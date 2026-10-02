//! Exact opt-in ordinary GEMM on the narrowly proved no-split projection shapes.
use super::Bf16GemmPlan;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

pub(crate) const PATH_ENV: &str = "EFFECT_TORCH_CUDA_ORDINARY_K16_PTX";
#[repr(C)]
struct Arguments {
    x: u64,
    weight: u64,
    out: u64,
    columns: u32,
    reserved: u32,
}
// SAFETY: initialized repr(C) scalar fields match the standalone CUDA ABI.
unsafe impl DeviceRepr for Arguments {}

pub(crate) fn supports(
    plan: Bf16GemmPlan,
    transposed: bool,
    out_f32: bool,
    pointers: [u64; 3],
) -> bool {
    plan.m == 256
        && matches!(plan.n, 2048 | 2112)
        && plan.k == 2816
        && plan.batch == 1
        && plan.stride_x == 256 * 2816
        && (plan.stride_weight == 0 || plan.stride_weight == plan.n * plan.k)
        && plan.stride_out == 256 * plan.n
        && transposed
        && !out_f32
        && pointers
            .into_iter()
            .zip([
                plan.m * plan.k * 2,
                plan.n * plan.k * 2,
                plan.m * plan.n * 2,
            ])
            .all(|(pointer, bytes)| {
                pointer != 0 && pointer % 16 == 0 && pointer.checked_add(bytes as u64).is_some()
            })
}
pub(crate) fn fingerprint_matches(capability: (i32, i32), name: &str, version: i32) -> bool {
    capability == (12, 0)
        && name == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
        && version == 120901
}

pub(crate) struct OrdinaryK16 {
    function: CudaFunction,
}
impl OrdinaryK16 {
    pub(crate) fn load(context: &Arc<CudaContext>, path: &str) -> Result<Self, String> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| format!("CUDA ordinary K16 PTX {path}: {e}"))?;
        let module = context
            .load_module(Ptx::from_src(source))
            .map_err(|e| e.to_string())?;
        let function = module
            .load_function("et_ordinary_k16_v1")
            .map_err(|e| e.to_string())?;
        Ok(Self { function })
    }
    /// Caller retains operands and output under the existing GEMM invocation
    /// fence. Parameters are copied by value, with no mutable persistent scratch.
    pub(crate) unsafe fn launch(
        &self,
        stream: &Arc<CudaStream>,
        plan: Bf16GemmPlan,
        x: u64,
        weight: u64,
        out: u64,
    ) -> Result<(), String> {
        let args = Arguments {
            x,
            weight,
            out,
            columns: plan.n as u32,
            reserved: 0,
        };
        let mut launch = stream.launch_builder(&self.function);
        launch.arg(&args);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (plan.n.div_ceil(64) as u32, 4, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 49152,
            })
        }
        .map_err(|e| e.to_string())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordinary_k16_guards_shape_dtype_orientation_strides_addresses_and_fingerprint() {
        let plan = Bf16GemmPlan {
            m: 256,
            n: 2112,
            k: 2816,
            batch: 1,
            stride_x: 256 * 2816,
            stride_weight: 0,
            stride_out: 256 * 2112,
        };
        assert!(supports(plan, true, false, [16, 32, 48]));
        // Bias-free linearRows lowers to Matmul. For batch one its canonical
        // weight stride is N*K rather than zero; neither advances the pointer.
        for n in [2048, 2112] {
            for rank_three in [false, true] {
                let x = if rank_three {
                    vec![1, 256, 2816]
                } else {
                    vec![256, 2816]
                };
                let out = if rank_three {
                    vec![1, 256, n]
                } else {
                    vec![256, n]
                };
                let actual = crate::cublas::plan_row_bf16_gemm(
                    crate::cublas::RowGemmKind::Matmul,
                    &x,
                    &[2816, n],
                    &out,
                )
                .unwrap();
                assert_eq!(actual.stride_weight, n * 2816);
                assert!(supports(actual, true, false, [16, 32, 48]));
                assert!(!supports(
                    Bf16GemmPlan { batch: 2, ..actual },
                    true,
                    false,
                    [16, 32, 48]
                ));
                assert!(!supports(
                    Bf16GemmPlan {
                        stride_weight: 1,
                        ..actual
                    },
                    true,
                    false,
                    [16, 32, 48]
                ));
            }
        }
        assert!(!supports(plan, false, false, [16, 32, 48]));
        assert!(!supports(plan, true, true, [16, 32, 48]));
        for pointers in [[0, 32, 48], [16, 33, 48], [16, 32, u64::MAX - 15]] {
            assert!(!supports(plan, true, false, pointers));
        }
        for invalid in [
            Bf16GemmPlan { m: 255, ..plan },
            Bf16GemmPlan { n: 2113, ..plan },
            Bf16GemmPlan { k: 2817, ..plan },
            Bf16GemmPlan { batch: 2, ..plan },
            Bf16GemmPlan {
                stride_x: 0,
                ..plan
            },
            Bf16GemmPlan {
                stride_weight: 2112 * 2816 + 1,
                ..plan
            },
            Bf16GemmPlan {
                stride_out: 0,
                ..plan
            },
        ] {
            assert!(!supports(invalid, true, false, [16, 32, 48]));
        }
        assert!(fingerprint_matches(
            (12, 0),
            "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            120901
        ));
        assert!(!fingerprint_matches((12, 0), "other", 120901));
        assert!(!fingerprint_matches(
            (12, 1),
            "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            120901
        ));
        assert!(!fingerprint_matches(
            (12, 0),
            "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            120900
        ));
        assert_eq!(std::mem::size_of::<Arguments>(), 32);
    }
}

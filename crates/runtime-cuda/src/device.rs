#[path = "expert_branch_graphs.rs"]
mod expert_branch_graphs;
#[path = "expert_gemv_overlap.rs"]
pub(crate) mod expert_gemv_overlap;
#[cfg(test)]
#[path = "expert_gemv_overlap_tests.rs"]
mod expert_gemv_overlap_tests;
#[path = "expert_merged.rs"]
mod expert_merged;
use crate::cublas::{
    supports_exact_grouped_expert, Bf16GemmPlan, CudaBlas, CUBLAS_WORKSPACE_BYTES,
    EXPERT_BLAS_STREAMS, EXPERT_GROUPED_POINTER_BYTES, EXPERT_PARTIAL_POINTER_BYTES,
    EXPERT_SPLITK_DESCRIPTOR_BYTES,
};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, Weak};

const TYPED_HEADER: &str = include_str!("kernels/typed.cuh");
const TYPED_BINARY_SOURCE: &str = include_str!("kernels/typed_binary.cu");
const TYPED_SOURCE: &str = include_str!("kernels/typed.cu");
const GROUPED_ROWS_SOURCE: &str = include_str!("kernels/grouped_rows.cu");
const EXPERT_ROUTE_RANK_SOURCE: &str = include_str!("kernels/expert_route_rank.cu");
const ORDERED_SORTED_SOURCE: &str = include_str!("kernels/ordered_sorted.cu");
const GROUPED_INVERSE_SOURCE: &str = include_str!("kernels/grouped_inverse.cu");
const DUAL_ARGMAX_SOURCE: &str = include_str!("kernels/dual_argmax.cu");
const ROUTER_TAIL_SOURCE: &str = include_str!("kernels/router_tail.cu");
const ATTN_FFN_ENTRANCE_SOURCE: &str = include_str!("kernels/attention_ffn_entrance.cu");
const RMS_RESIDUAL_SOURCE: &str = include_str!("kernels/rms_residual.cu");
const FFN_NEXT_NORM63_SOURCE: &str = include_str!("kernels/ffn_next_norm63.cu");
const FFN_TAIL_SOURCE: &str = include_str!("kernels/ffn_tail.cu");
const RELAXED_NORM76_SOURCE: &str = include_str!("kernels/relaxed_norm76.cuh");
const GROUPED_COPY_SOURCE: &str = include_str!("kernels/grouped_copy.cu");
const COMMON_SOURCE: &str = include_str!("kernels/common.cuh");
const NORM_ROPE_SOURCE: &str = include_str!("kernels/norm_rope.cu");
const SMALL_SOFTMAX_SOURCE: &str = include_str!("kernels/small_softmax.cu");
const ENTROPY_SOURCE: &str = include_str!("kernels/entropy.cu");
const ENTROPY81_SOURCE: &str = include_str!("kernels/entropy81.cu");
const COMPUTE_WRAPPERS: &str = include_str!("kernels/compute.cu");
const SAMPLER83_SOURCE: &str = include_str!("kernels/sampler83.cu");
const RNG_ARG80_SOURCE: &str = include_str!("kernels/rng_arg80.cu");
const POINTWISE_SOURCE: &str = include_str!("kernels/pointwise.cu");
const TENSOR_SOURCE: &str = include_str!("kernels/tensor.cu");
const LINALG_SOURCE: &str = include_str!("kernels/linalg.cu");
const NEURAL_SOURCE: &str = include_str!("kernels/neural.cu");
const STATEFUL_SOURCE: &str = include_str!("kernels/stateful.cu");
const QUANTIZED_SOURCE: &str = include_str!("kernels/quantized.cu");
const CACHE_SOURCE: &str = include_str!("kernels/cache.cu");
const VNORM_STORE_SOURCE: &str = include_str!("kernels/vnorm_store.cu");
const EXPERT_SPLITK_SOURCE: &str = include_str!("kernels/expert_splitk.cu");

pub(crate) const CUDA_TOP_K_BLOCKS: usize = 128;
pub(crate) const CUDA_TOP_K_LIMIT: usize = 40;

// Applied after typed.cuh, so descriptor fields, casts, indexes, masks and
// persistent state do not inherit the compute type substitution.
const F32_PRELUDE: &str = r#"
#define ET_COMPUTE_F32 1
#define double float
#define fabs fabsf
#define sqrt sqrtf
#define exp expf
#define log logf
#define sin sinf
#define cos cosf
#define tanh tanhf
#define erf erff
#define floor floorf
#define ceil ceilf
#define round roundf
#define pow powf
#define fmax fmaxf
#define fmin fminf
#define nearbyint nearbyintf
"#;

pub(crate) struct CudaF32Kernels {
    pub(crate) greedy_argmax: CudaFunction,
    pub(crate) topk: CudaFunction,
}

const TYPED_KERNELS: &[&str] = &[
    "et_fill",
    "et_convert",
    "et_binary",
    "et_binary_fixed_0_3_3_3_1",
    "et_binary_fixed_1_3_3_3_1",
    "et_binary_fixed_2_3_3_3_1",
    "et_binary_fixed_3_3_3_3_1",
    "et_binary_fixed_4_3_3_3_1",
    "et_binary_fixed_5_3_3_3_1",
    "et_binary_fixed_0_3_3_3_2",
    "et_binary_fixed_1_3_3_3_2",
    "et_binary_fixed_2_3_3_3_2",
    "et_binary_fixed_3_3_3_3_2",
    "et_binary_fixed_4_3_3_3_2",
    "et_binary_fixed_5_3_3_3_2",
    "et_binary_fixed_0_1_1_1_1",
    "et_binary_fixed_1_1_1_1_1",
    "et_binary_fixed_2_1_1_1_1",
    "et_binary_fixed_3_1_1_1_1",
    "et_binary_fixed_4_1_1_1_1",
    "et_binary_fixed_5_1_1_1_1",
    "et_binary_fixed_0_1_1_1_2",
    "et_binary_fixed_1_1_1_1_2",
    "et_binary_fixed_2_1_1_1_2",
    "et_binary_fixed_3_1_1_1_2",
    "et_binary_fixed_4_1_1_1_2",
    "et_binary_fixed_5_1_1_1_2",
    "et_binary_fixed_0_3_1_1_1",
    "et_binary_fixed_1_3_1_1_1",
    "et_binary_fixed_2_3_1_1_1",
    "et_binary_fixed_3_3_1_1_1",
    "et_binary_fixed_4_3_1_1_1",
    "et_binary_fixed_5_3_1_1_1",
    "et_binary_fixed_0_3_1_1_2",
    "et_binary_fixed_1_3_1_1_2",
    "et_binary_fixed_2_3_1_1_2",
    "et_binary_fixed_3_3_1_1_2",
    "et_binary_fixed_4_3_1_1_2",
    "et_binary_fixed_5_3_1_1_2",
    "et_div_feedback",
    "et_packed_projection77_split",
    "et_unary",
    "et_reindex",
    "et_rotary_reindex",
    "et_where",
    "et_concat",
    "et_index",
    "et_scatter_add_inner",
    "et_ordered_scatter_reduce",
    "et_arg_index_last_wide",
    "et_top_k_indices",
    "et_expert_linear_rows",
    "et_grouped_counts",
    "et_grouped_offsets",
    "et_grouped_rows",
    "et_grouped_rows_block",
    "et_expert_route_rank",
    "et_ordered_sorted_reduce",
    "et_grouped_rows_inverse_block",
    "et_grouped_rows_inverse_warp",
    "et_rms_residual_bf16",
    "et_attention_ffn_entrance_bf16",
    "et_router_tail",
    "et_dual_argmax",
    "et_ffn_tail_bf16",
    "et_ffn_next_norm63_bf16",
    "et_grouped_gather",
    "et_grouped_scatter",
    "et_grouped_gather_vector",
    "et_grouped_scatter_vector",
    "et_grouped_pointer_banks",
    "et_expert_partial_pointer_banks",
    "et_grouped_matmul_f32",
    "et_sequence",
    "et_last_token",
    "et_optimizer",
    "et_reduce_integer",
];

struct ComputeModule {
    name: &'static str,
    define: &'static str,
    source: &'static str,
    kernels: &'static [&'static str],
}

const COMPUTE_MODULES: &[ComputeModule] = &[
    ComputeModule {
        name: "pointwise",
        define: "#define ET_POINTWISE",
        source: POINTWISE_SOURCE,
        kernels: &["et_random", "et_random_dual_arg80", "et_random_sampler83"],
    },
    ComputeModule {
        name: "tensor",
        define: "#define ET_TENSOR",
        source: TENSOR_SOURCE,
        kernels: &[
            "et_reduce",
            "et_matmul",
            "et_rms_norm",
            "et_cross_entropy",
            "et_chunked_head_ce",
        ],
    },
    ComputeModule {
        name: "linalg",
        define: "#define ET_LINALG",
        source: LINALG_SOURCE,
        kernels: &["et_conv", "et_linalg", "et_linear", "et_linear_bias"],
    },
    ComputeModule {
        name: "neural",
        define: "#define ET_NEURAL",
        source: NEURAL_SOURCE,
        kernels: &["et_layer_norm", "et_sdpa", "et_rotary"],
    },
    ComputeModule {
        name: "stateful",
        define: "#define ET_STATEFUL",
        source: STATEFUL_SOURCE,
        kernels: &["et_short_conv", "et_kda"],
    },
];

fn f32_bf16_bits_prelude() -> &'static str {
    if std::env::var("EFFECT_TORCH_CUDA_F32_BF16_BITS").as_deref() == Ok("1") {
        "#define ET_CUDA_F32_BF16_BITS 1\n"
    } else {
        ""
    }
}

fn compile_module(
    context: &Arc<CudaContext>,
    name: &str,
    sources: &[&str],
) -> Result<Arc<CudaModule>, String> {
    compile_module_artifact(context, name, sources, false).map(|v| v.0)
}
fn compile_module_artifact(
    context: &Arc<CudaContext>,
    name: &str,
    sources: &[&str],
    graph61: bool,
) -> Result<
    (
        Arc<CudaModule>,
        Option<Arc<crate::explicit_graph61::RawModule61>>,
        Option<Arc<[u8]>>,
    ),
    String,
> {
    let source = format!("{}{}", f32_bf16_bits_prelude(), sources.join("\n"));
    let (major, minor) = context
        .compute_capability()
        .map_err(|error| error.to_string())?;
    let arch: &'static str = Box::leak(format!("compute_{major}{minor}").into_boxed_str());
    let include_paths = if matches!(name, "expert-splitk.cu" | "expert-device.cu") {
        let mut paths = ["CUDA_HOME", "CUDA_PATH"]
            .into_iter()
            .filter_map(|key| std::env::var(key).ok())
            .map(|path| format!("{path}/include"))
            .collect::<Vec<_>>();
        paths.push("/usr/local/cuda/include".into());
        paths.retain(|path| std::path::Path::new(path).is_dir());
        paths.sort();
        paths.dedup();
        paths
    } else {
        Vec::new()
    };
    let ptx = compile_ptx_with_opts(
        &source,
        CompileOptions {
            arch: Some(arch),
            name: Some(name.to_string()),
            // Preserve F32 operations in canonical packed decode.
            fmad: Some(false),
            include_paths,
            ..Default::default()
        },
    )
    .map_err(|error| format!("CUDA compile {name}: {error}"))?;
    let retained_image =
        if crate::executable::expert_pair61::enabled() && name == "fused-elementwise.cu" {
            Some(Arc::<[u8]>::from(
                ptx.as_bytes().ok_or("CUDA graph61 expected fused image")?,
            ))
        } else {
            None
        };
    // Both loaders consume this one NVRTC result, with identical image bytes.
    let graph = if graph61 {
        Some(crate::explicit_graph61::RawModule61::load(
            context.clone(),
            ptx.as_bytes()
                .ok_or("CUDA graph61 expected compiled PTX image")?,
        )?)
    } else {
        None
    };
    Ok((
        context
            .load_module(ptx)
            .map_err(|error| error.to_string())?,
        graph,
        retained_image,
    ))
}

fn load(module: &Arc<CudaModule>, name: &str) -> Result<CudaFunction, String> {
    module
        .load_function(name)
        .map_err(|error| format!("CUDA kernel {name}: {error}"))
}

static DEVICES: LazyLock<Mutex<HashMap<u32, Weak<CudaDevice>>>> = LazyLock::new(Default::default);

/// By-value launch parameters are copied by the driver before launch returns;
/// no host DMA buffer or per-invocation device allocation is needed.
#[repr(C)]
struct GroupedPointerBanks {
    values: [[u64; 128]; 3],
    output: u64,
    count: u32,
}

// SAFETY: repr(C), initialized scalar arrays, matching EtGroupedPointerBanks.
unsafe impl DeviceRepr for GroupedPointerBanks {}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ExpertSplitKDescriptor {
    x: u64,
    weight: u64,
    out: u64,
    partial_offset: u64,
    rows: u32,
    splits: u32,
    slice_k: u32,
    compute_prefix: u32,
    reduce_prefix: u32,
    reserved: [u32; 3],
}

#[repr(C)]
struct ExpertSplitKUpload {
    descriptors: [ExpertSplitKDescriptor; 32],
}

// SAFETY: initialized repr(C) scalar fields match the CUDA descriptor ABI.
unsafe impl DeviceRepr for ExpertSplitKUpload {}

/// A failed submission may occur before completion events were joined onto the
/// primary stream. Drain every participating worker before invocation leases
/// can be released by the caller's primary-stream fence.
struct ExpertSubmissionFence<'a> {
    workers: &'a [CudaBlas],
    joined: bool,
}

impl Drop for ExpertSubmissionFence<'_> {
    fn drop(&mut self) {
        if !self.joined {
            for worker in self.workers {
                let _ = worker.stream().synchronize();
            }
        }
    }
}

#[cfg(test)]
thread_local! {
    static FAIL_EXPERT_SUBMISSION_BEFORE_JOIN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FAIL_EXPERT_SUBMISSION_AFTER_MERGED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn device_expert_failure_pending() -> bool {
    FAIL_EXPERT_SUBMISSION_BEFORE_JOIN.with(|f| f.get())
        || FAIL_EXPERT_SUBMISSION_AFTER_MERGED.with(|f| f.get())
}

#[cfg(test)]
pub(crate) fn with_device_expert_failure<T>(after_merged: bool, run: impl FnOnce() -> T) -> T {
    let flag = if after_merged {
        &FAIL_EXPERT_SUBMISSION_AFTER_MERGED
    } else {
        &FAIL_EXPERT_SUBMISSION_BEFORE_JOIN
    };
    flag.with(|f| f.set(true));
    let result = run();
    flag.with(|f| f.set(false));
    result
}

fn supports_exact_expert_gemv(plan: Bf16GemmPlan) -> bool {
    plan.batch == 1 && (plan.m, plan.n, plan.k) == (1, 2816, 704)
}

/// Split-K reuses the GEMV descriptor bank, so those launches must stay on
/// worker zero. Grouped cuBLAS uses a separate pointer bank and can overlap.
fn expert_worker_layout(
    grouped: bool,
    custom: bool,
    gemv: bool,
    dedicated_requested: bool,
    independent_merged_bank: bool,
) -> (usize, usize) {
    let gemv_worker = usize::from(
        gemv && ((independent_merged_bank && custom)
            || (dedicated_requested && grouped && !custom)),
    );
    let ordinary_start = usize::from(grouped || custom || gemv) + gemv_worker;
    (gemv_worker, ordinary_start)
}

fn expert_splitk_geometry(plan: Bf16GemmPlan) -> Option<(u32, u32)> {
    if plan.batch != 1 {
        return None;
    }
    match (plan.n, plan.k, plan.m) {
        (1408, 2816, 2..=16 | 65..=128 | 257..=448) => Some((1, 2816)),
        (1408, 2816, 17..=30) => Some((8, 384)),
        (1408, 2816, 31..=32 | 43..=57 | 193..=256 | 449..=512) => Some((4, 704)),
        (1408, 2816, 33..=42 | 58..=64) => Some((2, 1408)),
        (1408, 2816, 129..=192) => Some((5, 576)),
        (2816, 704, 2..=256 | 417..=512) => Some((1, 704)),
        (2816, 704, 257..=416) => Some((2, 384)),
        _ => None,
    }
}

#[cfg(test)]
mod splitk_tests {
    use super::*;

    #[test]
    fn dedicated_gemv_worker_keeps_shared_descriptors_serial() {
        for grouped in [false, true] {
            for custom in [false, true] {
                for gemv in [false, true] {
                    for requested in [false, true] {
                        let (worker, start) =
                            expert_worker_layout(grouped, custom, gemv, requested, false);
                        let dedicated = requested && grouped && !custom && gemv;
                        assert_eq!(worker, usize::from(dedicated));
                        assert_eq!(start, usize::from(grouped || custom || gemv) + worker);
                        if gemv {
                            assert!(worker < start);
                        }
                        if custom {
                            assert_eq!(worker, 0);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn overlapping_gemv_workers_preserve_empty_m1_and_all_m1_layouts() {
        assert_eq!(expert_worker_layout(false, true, true, false, true), (1, 2));
        assert_eq!(
            expert_worker_layout(false, true, false, false, true),
            (0, 1)
        );
        assert_eq!(
            expert_worker_layout(false, false, true, false, false),
            (0, 1)
        );
        assert_eq!(
            expert_worker_layout(false, false, false, false, false),
            (0, 0)
        );
        assert_eq!(
            expert_worker_layout(false, true, true, false, false),
            (0, 1)
        );
    }

    #[test]
    #[ignore = "requires SM120, grouped/GEMV/dedicated-stream flags=1 and split-K flags=0"]
    fn dedicated_gemv_submission_joins_and_drains_all_workers() {
        use crate::CudaValue;
        use cudarc::driver::DevicePtr;
        use effect_torch_runtime::DType;
        for key in [
            "EFFECT_TORCH_CUDA_GROUPED_EXACT",
            "EFFECT_TORCH_CUDA_EXPERT_GEMV",
            "EFFECT_TORCH_CUDA_EXPERT_GEMV_DEDICATED_STREAM",
        ] {
            assert!(std::env::var(key).is_ok_and(|v| v == "1"));
        }
        for key in [
            "EFFECT_TORCH_CUDA_EXPERT_SPLITK",
            "EFFECT_TORCH_CUDA_EXPERT_GROUPED_PARTIALS",
        ] {
            assert!(!std::env::var(key).is_ok_and(|v| v == "1"));
        }
        let device = CudaDevice::get(0).unwrap();
        assert!(device.exact_splitk_fingerprint);
        // Four grouped products, one GEMV and one ordinary product exercise
        // workers zero, one and two, with disjoint output slices.
        let rows = [2, 3, 4, 5, 1, 129];
        let total: usize = rows.iter().sum();
        let (n, k) = (2816, 704);
        let x = CudaValue::from_host(
            device.clone(),
            vec![total, k],
            DType::BF16,
            &vec![1.0; total * k],
        )
        .unwrap();
        let w = CudaValue::from_host(device.clone(), vec![n, k], DType::BF16, &vec![1.0; n * k])
            .unwrap();
        let bytes = CUBLAS_WORKSPACE_BYTES * EXPERT_BLAS_STREAMS
            + EXPERT_GROUPED_POINTER_BYTES
            + EXPERT_SPLITK_DESCRIPTOR_BYTES;
        // Failure first, then fresh invocation storage: error cleanup must not
        // leave either custom worker accessing released descriptors or outputs.
        for fail in [true, false] {
            let output = CudaValue::from_host(
                device.clone(),
                vec![total, n],
                DType::BF16,
                &vec![-1.0; total * n],
            )
            .unwrap();
            let mut offset = 0;
            let groups = rows.map(|m| {
                let plan = Bf16GemmPlan {
                    m,
                    n,
                    k,
                    batch: 1,
                    stride_x: m * k,
                    stride_weight: n * k,
                    stride_out: m * n,
                };
                let group = (
                    plan,
                    x.storage_address() + (offset * k * 2) as u64,
                    w.storage_address(),
                    output.storage_address() + (offset * n * 2) as u64,
                );
                offset += m;
                group
            });
            let workspace = unsafe { device.stream.alloc::<u8>(bytes) }.unwrap();
            let (address, guard) = workspace.device_ptr(&device.stream);
            FAIL_EXPERT_SUBMISSION_BEFORE_JOIN.with(|flag| flag.set(fail));
            let result = unsafe { device.grouped_gemm_bf16(&groups, address, true, bytes) };
            let events = if fail {
                assert_eq!(
                    result.unwrap_err(),
                    "injected expert submission failure before join"
                );
                Vec::new()
            } else {
                let events = result.unwrap();
                assert_eq!(events.len(), 3);
                events
            };
            assert!(output
                .readback()
                .unwrap()
                .iter()
                .all(|value| *value == 704.0));
            drop((events, guard));
        }
    }

    #[test]
    #[ignore = "requires SM120/cuBLAS 12.9.1 and EFFECT_TORCH_CUDA_EXPERT_SPLITK=1"]
    fn failed_mixed_expert_submission_drains_workers_before_scratch_release() {
        use crate::CudaValue;
        use cudarc::driver::DevicePtr;
        use effect_torch_runtime::DType;
        let device = CudaDevice::get(0).unwrap();
        assert!(
            device.exact_splitk_fingerprint,
            "split-K fingerprint and startup opt-in are required"
        );
        let (n, k) = (2816, 704);
        let plan = |m| Bf16GemmPlan {
            m,
            n,
            k,
            batch: 1,
            stride_x: m * k,
            stride_weight: n * k,
            stride_out: m * n,
        };
        let x = CudaValue::from_host(
            device.clone(),
            vec![513, k],
            DType::BF16,
            &vec![1.0; 513 * k],
        )
        .unwrap();
        let weight =
            CudaValue::from_host(device.clone(), vec![n, k], DType::BF16, &vec![1.0; n * k])
                .unwrap();
        let output = CudaValue::from_host(
            device.clone(),
            vec![513, n],
            DType::BF16,
            &vec![-1.0; 513 * n],
        )
        .unwrap();
        let workspace_bytes = CUBLAS_WORKSPACE_BYTES * EXPERT_BLAS_STREAMS
            + EXPERT_GROUPED_POINTER_BYTES
            + EXPERT_SPLITK_DESCRIPTOR_BYTES
            + EXPERT_PARTIAL_POINTER_BYTES
            + 8 * 513 * n * 2;
        let workspace = unsafe { device.stream.alloc::<u8>(workspace_bytes) }.unwrap();
        let (workspace_address, workspace_guard) = workspace.device_ptr(&device.stream);
        let groups = [
            (
                plan(512),
                x.storage_address(),
                weight.storage_address(),
                output.storage_address(),
            ),
            (
                plan(1),
                x.storage_address() + (512 * k * 2) as u64,
                weight.storage_address(),
                output.storage_address() + (512 * n * 2) as u64,
            ),
        ];
        FAIL_EXPERT_SUBMISSION_BEFORE_JOIN.with(|flag| flag.set(true));
        let error =
            unsafe { device.grouped_gemm_bf16(&groups, workspace_address, true, workspace_bytes) }
                .unwrap_err();
        assert_eq!(error, "injected expert submission failure before join");
        // No successful event joins were installed. The failed submission must
        // nevertheless have completed both the custom and ordinary outputs.
        assert!(output
            .readback()
            .unwrap()
            .iter()
            .all(|value| *value == 704.0));
        drop(workspace_guard);
        drop(workspace);
        // A following submission verifies the failed scratch/worker state did
        // not poison later invocation ownership or stream scheduling.
        let workspace = unsafe { device.stream.alloc::<u8>(workspace_bytes) }.unwrap();
        let (workspace_address, workspace_guard) = workspace.device_ptr(&device.stream);
        let completed =
            unsafe { device.grouped_gemm_bf16(&groups, workspace_address, true, workspace_bytes) }
                .unwrap();
        device.stream.synchronize().unwrap();
        assert!(output
            .readback()
            .unwrap()
            .iter()
            .all(|value| *value == 704.0));
        drop(completed);
        drop(workspace_guard);
    }

    #[test]
    #[ignore = "requires SM120/cuBLAS 12.9.1 and EFFECT_TORCH_CUDA_EXPERT_GEMV=1"]
    fn gemv_descriptor_chunks_match_ordinary_cublas() {
        use crate::CudaValue;
        use cudarc::driver::DevicePtr;
        use effect_torch_runtime::DType;
        assert!(std::env::var("EFFECT_TORCH_CUDA_EXPERT_GEMV").is_ok_and(|v| v == "1"));
        let device = CudaDevice::get(0).unwrap();
        assert!(device.exact_splitk_fingerprint);
        let (n, k, experts) = (2816, 704, 33);
        let plan = Bf16GemmPlan {
            m: 1,
            n,
            k,
            batch: 1,
            stride_x: k,
            stride_weight: n * k,
            stride_out: n,
        };
        // Signed powers and mantissas exercise cancellation and BF16 rounding;
        // 33 descriptors cross the by-value upload boundary.
        let values = |count: usize, seed: usize| {
            (0..count)
                .map(|i| {
                    let bits = i.wrapping_mul(1664525).wrapping_add(seed);
                    let mantissa = ((bits >> 8) % 255) as f64 - 127.0;
                    mantissa * 2.0f64.powi(((bits >> 19) % 15) as i32 - 12)
                })
                .collect::<Vec<_>>()
        };
        let x = CudaValue::from_host(
            device.clone(),
            vec![experts, k],
            DType::BF16,
            &values(experts * k, 17),
        )
        .unwrap();
        let w = CudaValue::from_host(device.clone(), vec![n, k], DType::BF16, &values(n * k, 313))
            .unwrap();
        let make_output = || {
            CudaValue::from_host(
                device.clone(),
                vec![experts, n],
                DType::BF16,
                &vec![0.0; experts * n],
            )
            .unwrap()
        };
        let actual = make_output();
        let expected = make_output();
        let groups = |output: &CudaValue| {
            (0..experts)
                .map(|i| {
                    (
                        plan,
                        x.storage_address() + (i * k * 2) as u64,
                        w.storage_address(),
                        output.storage_address() + (i * n * 2) as u64,
                    )
                })
                .collect::<Vec<_>>()
        };
        let workspace_bytes = CUBLAS_WORKSPACE_BYTES * EXPERT_BLAS_STREAMS
            + EXPERT_GROUPED_POINTER_BYTES
            + EXPERT_SPLITK_DESCRIPTOR_BYTES;
        let workspace = unsafe { device.stream.alloc::<u8>(workspace_bytes) }.unwrap();
        let (address, guard) = workspace.device_ptr(&device.stream);
        let reference = unsafe {
            device.grouped_gemm_bf16(&groups(&expected), address, false, workspace_bytes)
        }
        .unwrap();
        let optimized =
            unsafe { device.grouped_gemm_bf16(&groups(&actual), address, true, workspace_bytes) }
                .unwrap();
        let actual = actual.readback().unwrap();
        let expected = expected.readback().unwrap();
        assert!(actual
            .iter()
            .zip(&expected)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        drop((reference, optimized, guard));
    }

    #[test]
    fn descriptor_abi_matches_cuda() {
        assert_eq!(std::mem::size_of::<ExpertSplitKDescriptor>(), 64);
        assert_eq!(std::mem::size_of::<ExpertSplitKUpload>(), 32 * 64);
        assert_eq!(std::mem::offset_of!(ExpertSplitKDescriptor, rows), 32);
        assert_eq!(
            std::mem::offset_of!(ExpertSplitKDescriptor, compute_prefix),
            44
        );
        assert_eq!(
            std::mem::offset_of!(ExpertSplitKDescriptor, reduce_prefix),
            48
        );
    }

    #[test]
    fn exact_gemv_only_accepts_second_projection_single_row() {
        let mut plan = Bf16GemmPlan {
            m: 1,
            n: 2816,
            k: 704,
            batch: 1,
            stride_x: 704,
            stride_weight: 2816 * 704,
            stride_out: 2816,
        };
        assert!(supports_exact_expert_gemv(plan));
        plan.m = 2;
        assert!(!supports_exact_expert_gemv(plan));
        plan.m = 1;
        plan.batch = 2;
        assert!(!supports_exact_expert_gemv(plan));
        plan.batch = 1;
        plan.n = 1408;
        plan.k = 2816;
        assert!(!supports_exact_expert_gemv(plan));
    }

    #[test]
    fn exact_schedule_boundaries_and_fallbacks() {
        let plan = |m, n, k| Bf16GemmPlan {
            m,
            n,
            k,
            batch: 1,
            stride_x: m * k,
            stride_weight: n * k,
            stride_out: m * n,
        };
        for (start, end, schedule) in [
            (2, 16, (1, 2816)),
            (17, 30, (8, 384)),
            (31, 32, (4, 704)),
            (33, 42, (2, 1408)),
            (43, 57, (4, 704)),
            (58, 64, (2, 1408)),
            (65, 128, (1, 2816)),
            (129, 192, (5, 576)),
            (193, 256, (4, 704)),
            (257, 448, (1, 2816)),
            (449, 512, (4, 704)),
        ] {
            for m in start..=end {
                assert_eq!(expert_splitk_geometry(plan(m, 1408, 2816)), Some(schedule));
            }
        }
        for m in 2..=512 {
            assert_eq!(
                expert_splitk_geometry(plan(m, 2816, 704)),
                Some(if (257..=416).contains(&m) {
                    (2, 384)
                } else {
                    (1, 704)
                })
            );
        }
        for (m, n, k) in [
            (0, 1408, 2816),
            (1, 1408, 2816),
            (513, 1408, 2816),
            (1, 2816, 704),
            (513, 2816, 704),
            (32, 1407, 2816),
            (32, 2816, 705),
        ] {
            assert_eq!(expert_splitk_geometry(plan(m, n, k)), None);
        }
        let mut batched = plan(32, 1408, 2816);
        batched.batch = 2;
        assert_eq!(expert_splitk_geometry(batched), None);
    }
}

/// One CUDA context, stream, and eagerly compiled kernel registry per device.
pub struct CudaDevice {
    pub(crate) ordinal: u32,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) f32: CudaF32Kernels,
    pub(crate) cublas: CudaBlas,
    pub(crate) dense_cublas: CudaBlas,
    pub(crate) kv_pair: Option<crate::kv_pair::Kernel>,
    expert_cublas: Vec<CudaBlas>,
    expert_branch_graphs: Mutex<crate::cublas::expert_graphs::Cache>,
    exact_splitk_fingerprint: bool,
    merged_expert: Option<expert_merged::MergedExpert>,
    device59_blocks: Option<u32>,
    /// Serializes graph capture/replay on the shared stream. CUDA graphs are
    /// not internally synchronized and capture must not overlap submissions
    /// from another executable using this device.
    pub(crate) graph_execution: Mutex<()>,
    pub(crate) greedy_argmax_output: Mutex<CudaSlice<u32>>,
    pub(crate) topk_output: Mutex<CudaSlice<f32>>,
    kernels: HashMap<String, CudaFunction>,
    fused_kernels: Mutex<HashMap<String, CudaFunction>>,
    graph61_kernels: HashMap<String, Arc<crate::explicit_graph61::RawKernel61>>,
    graph61_fused: Mutex<HashMap<String, Arc<crate::explicit_graph61::RawKernel61>>>,
    graph61_fused_images: Mutex<HashMap<String, Arc<[u8]>>>,
}

impl CudaDevice {
    /// Returns the process-wide runtime device for this ordinal.
    pub fn get(ordinal: u32) -> Result<Arc<Self>, String> {
        let mut devices = DEVICES.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(device) = devices.get(&ordinal).and_then(Weak::upgrade) {
            return Ok(device);
        }
        let device = Arc::new(Self::new(ordinal)?);
        devices.insert(ordinal, Arc::downgrade(&device));
        Ok(device)
    }

    /// Looks up an eagerly compiled production entrypoint. Execution never
    /// compiles kernels or creates alternate representations of weights.
    pub(crate) fn kernel(&self, name: &str) -> Result<&CudaFunction, String> {
        self.kernels
            .get(name)
            .ok_or_else(|| format!("CUDA kernel {name} is not registered"))
    }

    pub(crate) fn fused_elementwise(&self, source: &str) -> Result<CudaFunction, String> {
        self.fused_elementwise_with_graph61(source, false)
    }
    pub(crate) fn fused_elementwise_with_graph61(
        &self,
        source: &str,
        graph61: bool,
    ) -> Result<CudaFunction, String> {
        let cache_key = format!("{}{source}", f32_bf16_bits_prelude());
        let mut kernels = self
            .fused_kernels
            .lock()
            .map_err(|_| "CUDA fused-kernel cache lock poisoned")?;
        if let Some(function) = kernels.get(&cache_key) {
            if graph61 {
                let mut raw = self
                    .graph61_fused
                    .lock()
                    .map_err(|_| "CUDA graph61 fused registry lock poisoned")?;
                if !raw.contains_key(&cache_key) {
                    if let Some(image) = self
                        .graph61_fused_images
                        .lock()
                        .map_err(|_| "CUDA graph61 fused image lock poisoned")?
                        .get(&cache_key)
                    {
                        let module = crate::explicit_graph61::RawModule61::load(
                            self.stream.context().clone(),
                            image,
                        )?;
                        raw.insert(cache_key.clone(), module.function("et_fused_elementwise")?);
                    }
                }
            }
            return Ok(function.clone());
        }
        let (module, graph_module, image) = compile_module_artifact(
            self.stream.context(),
            "fused-elementwise.cu",
            &[TYPED_HEADER, source],
            graph61,
        )?;
        let function = load(&module, "et_fused_elementwise")?;
        if let Some(module) = graph_module {
            self.graph61_fused
                .lock()
                .map_err(|_| "CUDA graph61 fused registry lock poisoned")?
                .insert(cache_key.clone(), module.function("et_fused_elementwise")?);
        }
        if let Some(image) = image {
            self.graph61_fused_images
                .lock()
                .map_err(|_| "CUDA graph61 fused image lock poisoned")?
                .insert(cache_key.clone(), image);
        }
        kernels.insert(cache_key, function.clone());
        Ok(function)
    }

    #[cfg(test)]
    pub(crate) fn graph61_retained_fused_image(&self, source: &str) -> Option<Arc<[u8]>> {
        self.graph61_fused_images
            .lock()
            .unwrap()
            .get(&format!("{}{source}", f32_bf16_bits_prelude()))
            .cloned()
    }
    pub(crate) fn graph61_kernel(
        &self,
        name: &str,
    ) -> Option<Arc<crate::explicit_graph61::RawKernel61>> {
        self.graph61_kernels.get(name).cloned()
    }
    pub(crate) fn graph61_fused(
        &self,
        source: &str,
    ) -> Option<Arc<crate::explicit_graph61::RawKernel61>> {
        self.graph61_fused
            .lock()
            .ok()?
            .get(&format!("{}{source}", f32_bf16_bits_prelude()))
            .cloned()
    }
    pub(crate) fn graph61_workers(
        &self,
    ) -> Option<(Arc<crate::explicit_graph61::RawKernel61>, u32, u32)> {
        let merged = self.merged_expert.as_ref()?;
        Some((
            merged.graph61_compute.clone()?,
            merged.graph61_blocks(),
            self.device59_blocks?,
        ))
    }
    pub(crate) fn device_expert_ready(&self) -> bool {
        self.device59_blocks.is_some() && crate::expert_device::enabled()
    }

    /// Inputs, metadata and disjoint worker outputs remain invocation-owned.
    pub(crate) unsafe fn device_expert_gemm(
        &self,
        workspace: u64,
        columns: usize,
        inner: usize,
    ) -> Result<Vec<CudaEvent>, String> {
        let blocks = self
            .device59_blocks
            .ok_or("CUDA device expert unavailable")?;
        let descriptors = workspace;
        let shapes = workspace + 8192;
        let m1 = shapes + 1536;
        let ready = self.stream.record_event(None).map_err(|e| e.to_string())?;
        let workers = &self.expert_cublas[..2];
        let mut fence = ExpertSubmissionFence {
            workers,
            joined: false,
        };
        workers[0]
            .stream()
            .wait(&ready)
            .map_err(|e| e.to_string())?;
        unsafe {
            self.merged_expert
                .as_ref()
                .ok_or("CUDA merged expert missing")?
                .launch_device59(
                    descriptors,
                    shapes,
                    columns as u32,
                    inner as u32,
                    workers[0].stream(),
                )?;
        }
        #[cfg(test)]
        if FAIL_EXPERT_SUBMISSION_AFTER_MERGED.with(|flag| flag.replace(false)) {
            return Err("injected expert submission failure after merged launch".into());
        }
        let stream = workers[1].stream();
        stream.wait(&ready).map_err(|e| e.to_string())?;
        let first = (columns, inner) == (1408, 2816);
        let function = self.kernel(if first {
            "et_expert_device_first59"
        } else {
            "et_expert_device_second59"
        })?;
        let mut launch = stream.launch_builder(function);
        launch.arg(&descriptors).arg(&m1);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (if first { 16 } else { 32 }, 4, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| e.to_string())?;
        #[cfg(test)]
        if FAIL_EXPERT_SUBMISSION_BEFORE_JOIN.with(|flag| flag.replace(false)) {
            return Err("injected expert submission failure before join".into());
        }
        let mut completed = Vec::new();
        for worker in workers {
            let event = worker
                .stream()
                .record_event(None)
                .map_err(|e| e.to_string())?;
            self.stream.wait(&event).map_err(|e| e.to_string())?;
            completed.push(event);
        }
        fence.joined = true;
        Ok(completed)
    }

    /// Runs independent expert products on separate streams and joins them
    /// back into the primary stream before the caller scatters their rows.
    ///
    /// # Safety
    /// The pointers must remain valid through the primary-stream completion
    /// fence and each worker must receive a disjoint workspace and output.
    pub(crate) unsafe fn grouped_gemm_bf16(
        &self,
        groups: &[(Bf16GemmPlan, u64, u64, u64)],
        workspace: u64,
        splitk_workspace: bool,
        workspace_bytes: usize,
    ) -> Result<Vec<CudaEvent>, String> {
        if groups.is_empty() {
            return Ok(Vec::new());
        }
        let custom_workspace = splitk_workspace
            && self.exact_splitk_fingerprint
            && groups.len() <= 128
            && groups
                .iter()
                .all(|(plan, ..)| (plan.n, plan.k) == (groups[0].0.n, groups[0].0.k));
        let merged_count = if custom_workspace && self.merged_expert.is_some() {
            groups
                .iter()
                .filter(|group| expert_merged::supports(group))
                .count()
        } else {
            0
        };
        let merged = merged_count != 0;
        let splitk = !merged
            && custom_workspace
            && [
                "EFFECT_TORCH_CUDA_EXPERT_SPLITK",
                "EFFECT_TORCH_CUDA_EXPERT_GROUPED_PARTIALS",
            ]
            .into_iter()
            .any(|key| std::env::var(key).is_ok_and(|value| value == "1"));
        let separate_gemv = custom_workspace
            && self.merged_expert.is_some()
            && expert_gemv_overlap::enabled()
            && (groups[0].0.n, groups[0].0.k) == (2816, 704);
        let gemv = separate_gemv
            || (!merged
                && custom_workspace
                && std::env::var("EFFECT_TORCH_CUDA_EXPERT_GEMV").is_ok_and(|value| value == "1"));
        let gemv_count = groups
            .iter()
            .filter(|(plan, ..)| gemv && supports_exact_expert_gemv(*plan))
            .count();
        let custom_count = if splitk {
            groups
                .iter()
                .filter(|(plan, ..)| expert_splitk_geometry(*plan).is_some())
                .count()
        } else {
            0
        };
        let custom = custom_count != 0;
        let grouped_requested =
            std::env::var("EFFECT_TORCH_CUDA_GROUPED_EXACT").is_ok_and(|value| value == "1");
        let eligible = groups
            .iter()
            .filter(|(plan, ..)| supports_exact_grouped_expert(*plan))
            .count();
        let grouped = !merged && !custom && grouped_requested && (4..=128).contains(&eligible);
        let (gemv_worker, worker_start) = expert_worker_layout(
            grouped,
            custom || merged,
            gemv_count != 0,
            std::env::var("EFFECT_TORCH_CUDA_EXPERT_GEMV_DEDICATED_STREAM")
                .is_ok_and(|value| value == "1"),
            separate_gemv && merged,
        );
        let gemv_descriptor_offset = if separate_gemv && gemv_count != 0 {
            Some(expert_gemv_overlap::descriptor_offset(
                workspace,
                workspace_bytes,
            )?)
        } else {
            None
        };
        let ordinary_count = groups.len()
            - gemv_count
            - if merged {
                merged_count
            } else if custom {
                custom_count
            } else if grouped {
                eligible
            } else {
                0
            };
        if std::env::var("EFFECT_TORCH_CUDA_EXPERT_DISPATCH_TRACE").is_ok_and(|value| value == "1")
        {
            let grouped_partials_requested =
                std::env::var("EFFECT_TORCH_CUDA_EXPERT_GROUPED_PARTIALS")
                    .is_ok_and(|value| value == "1");
            eprintln!(
                "CUDA expert dispatch grouped_partials_requested={grouped_partials_requested} grouped_partials_enabled={} custom_enabled={custom} grouped_enabled={grouped} fingerprint={} splitk_workspace={splitk_workspace} merged_count={merged_count} custom_count={custom_count} gemv_count={gemv_count} separate_gemv={separate_gemv} gemv_worker={gemv_worker} ordinary_count={ordinary_count} groups={} shape=({},{}) workspace_bytes={workspace_bytes}",
                custom && grouped_partials_requested,
                self.exact_splitk_fingerprint,
                groups.len(),
                groups[0].0.n,
                groups[0].0.k,
            );
        }
        let workers = ordinary_count.min(self.expert_cublas.len() - worker_start);
        let mut partitions = vec![Vec::new(); workers];
        let mut loads = vec![0usize; workers];
        let mut ordered = groups
            .iter()
            .copied()
            .filter(|group @ (plan, ..)| {
                if gemv && supports_exact_expert_gemv(*plan) {
                    return false;
                }
                if merged {
                    !expert_merged::supports(group)
                } else if custom {
                    expert_splitk_geometry(*plan).is_none()
                } else {
                    !grouped || !supports_exact_grouped_expert(*plan)
                }
            })
            .collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|(plan, ..)| std::cmp::Reverse(plan.m));
        for group in ordered {
            let worker = loads
                .iter()
                .enumerate()
                .min_by_key(|(_, load)| **load)
                .map(|(worker, _)| worker)
                .ok_or("CUDA expert worker is missing")?;
            loads[worker] = loads[worker]
                .checked_add(group.0.m)
                .ok_or("CUDA expert worker load overflow")?;
            partitions[worker].push(group);
        }
        let ready = self
            .stream
            .record_event(None)
            .map_err(|error| error.to_string())?;
        let mut submission_fence = ExpertSubmissionFence {
            workers: &self.expert_cublas[..workers + worker_start],
            joined: false,
        };
        if merged {
            let worker = self.expert_cublas[0].stream();
            worker.wait(&ready).map_err(|error| error.to_string())?;
            // SAFETY: the common submission fence retains invocation scratch
            // and drains this worker on every success/error path.
            unsafe {
                self.merged_expert
                    .as_ref()
                    .ok_or("CUDA merged expert module missing")?
                    .launch(groups, workspace, workspace_bytes, worker)?;
            }
        }
        #[cfg(test)]
        if FAIL_EXPERT_SUBMISSION_AFTER_MERGED.with(|flag| flag.replace(false)) {
            return Err("injected expert submission failure after merged launch".into());
        }
        if custom {
            let worker = self.expert_cublas[0].stream();
            worker.wait(&ready).map_err(|error| error.to_string())?;
            // SAFETY: all descriptors address invocation-owned gathered inputs,
            // disjoint projected outputs and the bounded workspace below.
            unsafe {
                self.launch_expert_splitk(groups, workspace, workspace_bytes, worker)?;
            }
        }
        if grouped {
            let mut pointers = GroupedPointerBanks {
                values: [[0; 128]; 3],
                output: workspace + (EXPERT_BLAS_STREAMS * CUBLAS_WORKSPACE_BYTES) as u64,
                count: eligible as u32,
            };
            let mut plans = [groups[0].0; 128];
            for (index, (plan, x, weight, out)) in groups
                .iter()
                .filter(|(plan, ..)| supports_exact_grouped_expert(*plan))
                .enumerate()
            {
                plans[index] = *plan;
                pointers.values[0][index] = *weight;
                pointers.values[1][index] = *x;
                pointers.values[2][index] = *out;
            }
            let worker = &self.expert_cublas[0];
            worker
                .stream()
                .wait(&ready)
                .map_err(|error| error.to_string())?;
            let function = self.kernel("et_grouped_pointer_banks")?;
            let mut launch = worker.stream().launch_builder(function);
            launch.arg(&pointers);
            // SAFETY: the pointer banks are disjoint planned invocation scratch.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: ((3 * eligible).div_ceil(128) as u32, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|error| error.to_string())?;
            let banks = [
                pointers.output,
                pointers.output + 128 * 8,
                pointers.output + 256 * 8,
            ];
            // SAFETY: the bank producer and grouped GEMM share the worker stream.
            unsafe {
                worker.gemm_bf16_grouped(&plans[..eligible], banks, workspace)?;
            }
        }
        if gemv_count != 0 {
            let worker = self.expert_cublas[gemv_worker].stream();
            worker.wait(&ready).map_err(|error| error.to_string())?;
            // SAFETY: the overlapping path has an independently planned bank.
            // Older shared-bank paths remain serialized on their existing worker.
            // Every participating worker is covered by the submission fence.
            unsafe {
                self.launch_expert_gemv(
                    groups,
                    workspace,
                    workspace_bytes,
                    worker,
                    gemv_descriptor_offset,
                )?;
            }
        }
        let branch_graphs = workers != 0
            && std::env::var("EFFECT_TORCH_CUDA_EXPERT_BRANCH_GRAPHS").is_ok_and(|v| v == "1");
        let mut branch_events = Vec::new();
        if branch_graphs {
            branch_events =
                self.submit_ordinary_branch(&partitions, worker_start, workspace, &ready)?;
        } else {
            for (index, (worker, groups)) in self.expert_cublas
                [worker_start..worker_start + workers]
                .iter()
                .zip(&partitions)
                .enumerate()
            {
                worker
                    .stream()
                    .wait(&ready)
                    .map_err(|error| error.to_string())?;
                unsafe {
                    worker.gemm_bf16_sequence(
                        groups,
                        workspace + ((index + worker_start) * CUBLAS_WORKSPACE_BYTES) as u64,
                    )?;
                }
            }
        }
        #[cfg(test)]
        if FAIL_EXPERT_SUBMISSION_BEFORE_JOIN.with(|flag| flag.replace(false)) {
            return Err("injected expert submission failure before join".into());
        }
        // A joined graph's origin completion covers all CUDA-internal child
        // streams. Eager admissions also explicitly join children to origin.
        // Preserve all eager fork/join events in the invocation completion set.
        let joined_workers = worker_start + if branch_graphs { 1 } else { workers };
        let mut completed = branch_events;
        completed.reserve(joined_workers);
        for worker in &self.expert_cublas[..joined_workers] {
            let event = worker
                .stream()
                .record_event(None)
                .map_err(|error| error.to_string())?;
            self.stream
                .wait(&event)
                .map_err(|error| error.to_string())?;
            completed.push(event);
        }
        submission_fence.joined = true;
        Ok(completed)
    }

    unsafe fn launch_expert_gemv(
        &self,
        groups: &[(Bf16GemmPlan, u64, u64, u64)],
        workspace: u64,
        workspace_bytes: usize,
        worker: &Arc<CudaStream>,
        independent_offset: Option<usize>,
    ) -> Result<(), String> {
        let base = independent_offset
            .unwrap_or(CUBLAS_WORKSPACE_BYTES * EXPERT_BLAS_STREAMS + EXPERT_GROUPED_POINTER_BYTES);
        if groups.len() > 128
            || base
                .checked_add(EXPERT_SPLITK_DESCRIPTOR_BYTES)
                .is_none_or(|end| end > workspace_bytes)
        {
            return Err("CUDA expert GEMV workspace bounds exceeded".into());
        }
        let address = workspace
            .checked_add(base as u64)
            .ok_or("CUDA expert GEMV address overflow")?;
        let mut descriptors = Vec::new();
        for &(plan, x, weight, out) in groups {
            if supports_exact_expert_gemv(plan) {
                descriptors.push(ExpertSplitKDescriptor {
                    x,
                    weight,
                    out,
                    rows: 1,
                    compute_prefix: descriptors.len() as u32 * (2816 / 4),
                    ..ExpertSplitKDescriptor::default()
                });
            }
        }
        let count = descriptors.len() as u32;
        if count == 0 {
            return Ok(());
        }
        for (index, chunk) in descriptors.chunks(32).enumerate() {
            let mut batch = ExpertSplitKUpload {
                descriptors: [ExpertSplitKDescriptor::default(); 32],
            };
            batch.descriptors[..chunk.len()].copy_from_slice(chunk);
            let first = (index * 32) as u32;
            let batch_count = chunk.len() as u32;
            let mut launch = worker.launch_builder(self.kernel("et_expert_splitk_upload")?);
            launch
                .arg(&address)
                .arg(&first)
                .arg(&batch_count)
                .arg(&batch);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|error| error.to_string())?;
        }
        let mut launch = worker.launch_builder(self.kernel("et_expert_gemv_second")?);
        launch.arg(&address).arg(&count);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (count * (2816 / 4), 1, 1),
                block_dim: (32, 4, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| error.to_string())?;
        Ok(())
    }

    unsafe fn launch_expert_splitk(
        &self,
        groups: &[(Bf16GemmPlan, u64, u64, u64)],
        workspace: u64,
        workspace_bytes: usize,
        worker: &Arc<CudaStream>,
    ) -> Result<(), String> {
        let base = CUBLAS_WORKSPACE_BYTES
            .checked_mul(EXPERT_BLAS_STREAMS)
            .and_then(|bytes| bytes.checked_add(EXPERT_GROUPED_POINTER_BYTES))
            .ok_or("CUDA split-K workspace overflow")?;
        let partial_base = base
            .checked_add(EXPERT_SPLITK_DESCRIPTOR_BYTES)
            .and_then(|bytes| bytes.checked_add(EXPERT_PARTIAL_POINTER_BYTES))
            .ok_or("CUDA split-K workspace overflow")?;
        let mut descriptors = Vec::new();
        let mut elements = 0usize;
        let mut compute_blocks = 0usize;
        let mut reduce_blocks = 0usize;
        for &(plan, x, weight, out) in groups {
            let Some((splits, slice_k)) = expert_splitk_geometry(plan) else {
                continue;
            };
            descriptors.push(ExpertSplitKDescriptor {
                x,
                weight,
                out,
                partial_offset: u64::try_from(elements)
                    .map_err(|_| "CUDA split-K offset overflow")?,
                rows: u32::try_from(plan.m).map_err(|_| "CUDA split-K rows overflow")?,
                splits,
                slice_k,
                compute_prefix: u32::try_from(compute_blocks)
                    .map_err(|_| "CUDA split-K grid overflow")?,
                reduce_prefix: u32::try_from(reduce_blocks)
                    .map_err(|_| "CUDA split-K grid overflow")?,
                reserved: [0; 3],
            });
            let output_elements = plan
                .m
                .checked_mul(plan.n)
                .ok_or("CUDA split-K output overflow")?;
            elements = elements
                .checked_add(
                    output_elements
                        .checked_mul(splits as usize)
                        .ok_or("CUDA split-K partial overflow")?,
                )
                .ok_or("CUDA split-K partial overflow")?;
            compute_blocks = compute_blocks
                .checked_add(
                    plan.m
                        .div_ceil(32)
                        .checked_mul(plan.n.div_ceil(32))
                        .and_then(|blocks| blocks.checked_mul(splits as usize))
                        .ok_or("CUDA split-K grid overflow")?,
                )
                .ok_or("CUDA split-K grid overflow")?;
            reduce_blocks = reduce_blocks
                .checked_add(output_elements.div_ceil(256))
                .ok_or("CUDA split-K grid overflow")?;
        }
        let required = elements
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(partial_base))
            .ok_or("CUDA split-K workspace overflow")?;
        if required > workspace_bytes || descriptors.len() > 128 {
            return Err("CUDA split-K workspace bounds exceeded".into());
        }
        let descriptor_address = workspace
            .checked_add(base as u64)
            .ok_or("CUDA split-K address overflow")?;
        let partial_address = workspace
            .checked_add(partial_base as u64)
            .ok_or("CUDA split-K address overflow")?;
        let count = descriptors.len() as u32;
        for (batch_index, chunk) in descriptors.chunks(32).enumerate() {
            let mut batch = ExpertSplitKUpload {
                descriptors: [ExpertSplitKDescriptor::default(); 32],
            };
            batch.descriptors[..chunk.len()].copy_from_slice(chunk);
            let first = (batch_index * 32) as u32;
            let batch_count = chunk.len() as u32;
            let mut launch = worker.launch_builder(self.kernel("et_expert_splitk_upload")?);
            launch
                .arg(&descriptor_address)
                .arg(&first)
                .arg(&batch_count)
                .arg(&batch);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|error| error.to_string())?;
        }
        let columns = groups[0].0.n as u32;
        let inner = groups[0].0.k as u32;
        let grouped_partials = std::env::var("EFFECT_TORCH_CUDA_EXPERT_GROUPED_PARTIALS")
            .is_ok_and(|value| value == "1");
        if grouped_partials {
            let bank_address = descriptor_address
                .checked_add(EXPERT_SPLITK_DESCRIPTOR_BYTES as u64)
                .ok_or("CUDA grouped partial pointer address overflow")?;
            let mut plans = Vec::new();
            let mut pointers = Vec::new();
            for descriptor in &descriptors {
                for split in 0..descriptor.splits {
                    let offset = split
                        .checked_mul(descriptor.slice_k)
                        .ok_or("CUDA grouped partial K overflow")?;
                    let k = inner
                        .checked_sub(offset)
                        .ok_or("CUDA grouped partial K overflow")?
                        .min(descriptor.slice_k);
                    plans.push((
                        Bf16GemmPlan {
                            m: descriptor.rows as usize,
                            n: columns as usize,
                            k: k as usize,
                            batch: 1,
                            stride_x: descriptor.rows as usize * inner as usize,
                            stride_weight: 0,
                            stride_out: descriptor.rows as usize * columns as usize,
                        },
                        inner as usize,
                    ));
                    let partial_offset = descriptor
                        .partial_offset
                        .checked_add(
                            u64::from(split) * u64::from(descriptor.rows) * u64::from(columns),
                        )
                        .and_then(|elements| elements.checked_mul(2))
                        .ok_or("CUDA grouped partial offset overflow")?;
                    pointers.push([
                        descriptor
                            .weight
                            .checked_add(u64::from(offset) * 2)
                            .ok_or("CUDA grouped partial weight overflow")?,
                        descriptor
                            .x
                            .checked_add(u64::from(offset) * 2)
                            .ok_or("CUDA grouped partial input overflow")?,
                        partial_address
                            .checked_add(partial_offset)
                            .ok_or("CUDA grouped partial output overflow")?,
                    ]);
                }
            }
            if plans.len() > 1024 {
                return Err("CUDA grouped partial pointer bounds exceeded".into());
            }
            if std::env::var("EFFECT_TORCH_CUDA_EXPERT_PARTIAL_GROUPS")
                .is_ok_and(|value| value == "1")
            {
                // Keep the pointer bank order identical to metadata order while
                // collecting equal geometries into one cuBLAS batch group.
                let mut calls = plans.into_iter().zip(pointers).collect::<Vec<_>>();
                calls.sort_unstable_by_key(|((plan, leading), _)| {
                    (plan.n, plan.m, plan.k, *leading)
                });
                (plans, pointers) = calls.into_iter().unzip();
            }
            for (chunk_index, chunk) in pointers.chunks(128).enumerate() {
                let mut bank = GroupedPointerBanks {
                    values: [[0; 128]; 3],
                    output: bank_address,
                    count: chunk.len() as u32,
                };
                for (lane, pointers) in chunk.iter().enumerate() {
                    for (role, pointer) in pointers.iter().enumerate() {
                        bank.values[role][lane] = *pointer;
                    }
                }
                let first = (chunk_index * 128) as u32;
                let mut upload =
                    worker.launch_builder(self.kernel("et_expert_partial_pointer_banks")?);
                upload.arg(&bank).arg(&first);
                unsafe {
                    upload.launch(LaunchConfig {
                        grid_dim: ((3 * chunk.len()).div_ceil(128) as u32, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|error| error.to_string())?;
            }
            // SAFETY: all three banks were uploaded on this worker; partial
            // GEMMs retain original input leading dimensions and disjoint output.
            unsafe {
                self.expert_cublas[0].gemm_bf16_grouped_partials(
                    &plans,
                    [
                        bank_address,
                        bank_address + 1024 * 8,
                        bank_address + 2048 * 8,
                    ],
                    workspace,
                )?;
            }
        } else {
            let mut compute = worker.launch_builder(self.kernel("et_expert_splitk_compute")?);
            compute
                .arg(&descriptor_address)
                .arg(&count)
                .arg(&partial_address)
                .arg(&columns)
                .arg(&inner);
            unsafe {
                compute.launch(LaunchConfig {
                    grid_dim: (
                        u32::try_from(compute_blocks).map_err(|_| "CUDA split-K grid overflow")?,
                        1,
                        1,
                    ),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|error| error.to_string())?;
        }
        let mut reduce = worker.launch_builder(self.kernel("et_expert_splitk_reduce")?);
        reduce
            .arg(&descriptor_address)
            .arg(&count)
            .arg(&partial_address)
            .arg(&columns);
        unsafe {
            reduce.launch(LaunchConfig {
                grid_dim: (
                    u32::try_from(reduce_blocks).map_err(|_| "CUDA split-K grid overflow")?,
                    1,
                    1,
                ),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| error.to_string())?;
        Ok(())
    }

    fn new(ordinal: u32) -> Result<Self, String> {
        let count = Self::count()?;
        if ordinal >= count {
            return Err(format!(
                "CUDA device ordinal {ordinal} is out of range for {count} devices"
            ));
        }
        let context = CudaContext::new(ordinal as usize).map_err(|error| error.to_string())?;
        let stream = context.new_stream().map_err(|error| error.to_string())?;
        let mut kernels = HashMap::new();
        let mut graph61_kernels = HashMap::new();
        let graph61 = crate::executable::expert_pair61::enabled();
        let (typed, typed_graph61, _) = compile_module_artifact(
            &context,
            "typed.cu",
            &[
                TYPED_HEADER,
                TYPED_SOURCE,
                TYPED_BINARY_SOURCE,
                GROUPED_COPY_SOURCE,
                GROUPED_ROWS_SOURCE,
                if std::env::var("EFFECT_TORCH_CUDA_RELAXED_NORM76").as_deref() == Ok("1") {
                    "#define ET_RELAXED_NORM76 1\n"
                } else {
                    ""
                },
                RELAXED_NORM76_SOURCE,
                FFN_TAIL_SOURCE,
                FFN_NEXT_NORM63_SOURCE,
                RMS_RESIDUAL_SOURCE,
                ATTN_FFN_ENTRANCE_SOURCE,
                ROUTER_TAIL_SOURCE,
                DUAL_ARGMAX_SOURCE,
                ORDERED_SORTED_SOURCE,
                GROUPED_INVERSE_SOURCE,
                EXPERT_ROUTE_RANK_SOURCE,
            ],
            graph61,
        )?;
        for &name in TYPED_KERNELS {
            kernels.insert(name.to_string(), load(&typed, name)?);
            if let Some(module) = &typed_graph61 {
                if matches!(
                    name,
                    "et_fill"
                        | "et_grouped_counts"
                        | "et_grouped_offsets"
                        | "et_grouped_rows"
                        | "et_grouped_rows_block"
                        | "et_grouped_gather"
                        | "et_grouped_gather_vector"
                        | "et_grouped_scatter"
                        | "et_grouped_scatter_vector"
                ) {
                    graph61_kernels.insert(name.to_string(), module.function(name)?);
                }
            }
        }
        let quantized =
            compile_module(&context, "quantized.cu", &[TYPED_HEADER, QUANTIZED_SOURCE])?;
        for name in ["et_quantized_linear", "et_quantized_embedding"] {
            kernels.insert(name.to_string(), load(&quantized, name)?);
        }
        let cache = compile_module(
            &context,
            "cache.cu",
            &[TYPED_HEADER, CACHE_SOURCE, VNORM_STORE_SOURCE],
        )?;
        for name in [
            "et_kv_store",
            "et_vnorm_store56",
            "et_kv_attention",
            "et_kv_gemm_gather",
            "et_kv_gemm_softmax",
            "et_kv_gemm_round",
        ] {
            kernels.insert(name.to_string(), load(&cache, name)?);
        }
        let mut sampling = None;
        for definition in COMPUTE_MODULES {
            for (suffix, prelude) in [("f32", F32_PRELUDE), ("f64", "")] {
                let module = compile_module(
                    &context,
                    &format!("{}-{suffix}.cu", definition.name),
                    &[
                        TYPED_HEADER,
                        prelude,
                        COMMON_SOURCE,
                        definition.define,
                        definition.source,
                        RNG_ARG80_SOURCE,
                        SAMPLER83_SOURCE,
                        COMPUTE_WRAPPERS,
                        ENTROPY_SOURCE,
                        ENTROPY81_SOURCE,
                        SMALL_SOFTMAX_SOURCE,
                        NORM_ROPE_SOURCE,
                    ],
                )?;
                for &name in definition.kernels {
                    kernels.insert(format!("{name}_{suffix}"), load(&module, name)?);
                }
                if definition.name == "tensor" && suffix == "f32" {
                    kernels.insert(
                        "et_norm_rope_bf16".into(),
                        load(&module, "et_norm_rope_bf16")?,
                    );
                    for name in [
                        "et_mean256_f32",
                        "et_small_softmax_f32",
                        "et_bf16_softmax_prepare",
                        "et_bf16_softmax_store",
                        "et_entropy_max",
                        "et_entropy_sum",
                        "et_entropy_normalized_max",
                        "et_entropy_normalized_sum",
                        "et_entropy_finish",
                        crate::entropy81::KERNEL,
                    ] {
                        kernels.insert(name.into(), load(&module, name)?);
                    }
                    kernels.insert(
                        "et_shared_rms_norm_f32".into(),
                        load(&module, "et_shared_rms_norm_f32")?,
                    );
                    kernels.insert(
                        "et_rms_norm_wide_f32".into(),
                        load(&module, "et_rms_norm_wide")?,
                    );
                    kernels.insert(
                        "et_rms_norm_wide_vector_f32".into(),
                        load(&module, "et_rms_norm_wide_vector")?,
                    );
                    for name in [
                        "et_rms_norm_static2816_1_1",
                        "et_rms_norm_static2816_1_3",
                        "et_rms_norm_static2816_3_1",
                        "et_rms_norm_static2816_3_3",
                    ] {
                        kernels.insert(name.into(), load(&module, name)?);
                    }
                    kernels.insert("et_sum_wide_f32".into(), load(&module, "et_sum_wide")?);
                    kernels.insert(
                        "et_reduce_last_wide_f32".into(),
                        load(&module, "et_reduce_last_wide")?,
                    );
                }
                if definition.name == "pointwise" && suffix == "f32" {
                    sampling = Some(CudaF32Kernels {
                        greedy_argmax: load(&module, "greedy_argmax_f64")?,
                        topk: load(&module, "topk_f64")?,
                    });
                }
            }
        }
        let mut cublas = CudaBlas::new(stream.clone())?;
        let ordinary_path = std::env::var(crate::cublas::ordinary_k16::PATH_ENV)
            .ok()
            .filter(|p| !p.is_empty());
        let ordinary_k16 = if ordinary_path.is_some()
            && crate::cublas::ordinary_k16::fingerprint_matches(
                context.compute_capability().map_err(|e| e.to_string())?,
                &context.name().map_err(|e| e.to_string())?,
                cublas.version,
            ) {
            ordinary_path
                .as_deref()
                .map(|path| {
                    crate::cublas::ordinary_k16::OrdinaryK16::load(&context, path).map(Arc::new)
                })
                .transpose()?
        } else {
            None
        };
        let kv_pair = if crate::kv_pair::enabled() && ordinary_k16.is_some() {
            let path = std::env::var(crate::kv_pair::PATH_ENV).map_err(|e| e.to_string())?;
            Some(crate::kv_pair::Kernel::load(&context, &path)?)
        } else {
            None
        };
        cublas.set_ordinary_k16(ordinary_k16.clone());
        // These cuBLAS schedules were exhaustively compared on the existing
        // SM120 server with cuBLAS 12.9.1. Other libraries/devices stay ordinary.
        let splitk_requested = [
            expert_gemv_overlap::ENV,
            "EFFECT_TORCH_CUDA_EXPERT_GEMV",
            "EFFECT_TORCH_CUDA_EXPERT_SPLITK",
            "EFFECT_TORCH_CUDA_EXPERT_GROUPED_PARTIALS",
        ]
        .into_iter()
        .any(|key| std::env::var(key).is_ok_and(|value| value == "1"));
        let merged_path = std::env::var(expert_merged::PATH_ENV)
            .ok()
            .filter(|path| !path.is_empty());
        let exact_splitk_fingerprint = (splitk_requested || merged_path.is_some())
            && context
                .compute_capability()
                .map_err(|error| error.to_string())?
                == (12, 0)
            && context.name().map_err(|error| error.to_string())?
                == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
            && cublas.version == 120901;
        if exact_splitk_fingerprint && splitk_requested {
            let module = compile_module(&context, "expert-splitk.cu", &[EXPERT_SPLITK_SOURCE])?;
            for name in [
                "et_expert_gemv_second",
                "et_expert_splitk_upload",
                "et_expert_splitk_compute",
                "et_expert_splitk_reduce",
            ] {
                kernels.insert(name.to_string(), load(&module, name)?);
            }
        }
        let merged_expert = if exact_splitk_fingerprint {
            merged_path
                .as_deref()
                .map(|path| expert_merged::MergedExpert::load(&context, path))
                .transpose()?
        } else {
            None
        };
        let device59_blocks = if crate::expert_device::enabled()
            && merged_expert.as_ref().is_some_and(|m| m.device59_artifact)
        {
            let (module, graph_module, _) = compile_module_artifact(
                &context,
                "expert-device.cu",
                &[TYPED_HEADER, include_str!("kernels/expert_device.cu")],
                graph61,
            )?;
            for name in [
                "et_expert_device_metadata59",
                "et_expert_device_sanitize59",
                "et_expert_device_first59",
                "et_expert_device_second59",
            ] {
                kernels.insert(name.to_string(), load(&module, name)?);
                if let Some(module) = &graph_module {
                    graph61_kernels.insert(name.to_string(), module.function(name)?);
                }
            }
            let sms = context.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
                .map_err(|e| e.to_string())?;
            Some(
                u32::try_from(sms)
                    .ok()
                    .and_then(|n| n.checked_mul(16))
                    .filter(|n| *n != 0)
                    .ok_or("CUDA device expert invalid SM count")?,
            )
        } else {
            None
        };
        let mut dense_cublas =
            CudaBlas::new(context.new_stream().map_err(|error| error.to_string())?)?;
        dense_cublas.set_ordinary_k16(ordinary_k16);
        let expert_cublas = (0..EXPERT_BLAS_STREAMS)
            .map(|_| {
                let worker = context.new_stream().map_err(|error| error.to_string())?;
                CudaBlas::new(worker)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let greedy_argmax_output =
            unsafe { stream.alloc::<u32>(2) }.map_err(|error| error.to_string())?;
        let topk_output =
            unsafe { stream.alloc::<f32>(CUDA_TOP_K_BLOCKS * (2 * CUDA_TOP_K_LIMIT + 1)) }
                .map_err(|error| error.to_string())?;
        Ok(Self {
            ordinal,
            stream,
            f32: sampling.ok_or("CUDA sampling module was not registered")?,
            cublas,
            dense_cublas,
            kv_pair,
            expert_cublas,
            expert_branch_graphs: Mutex::new(crate::cublas::expert_graphs::Cache::branch()),
            exact_splitk_fingerprint,
            merged_expert,
            device59_blocks,
            graph_execution: Mutex::new(()),
            greedy_argmax_output: Mutex::new(greedy_argmax_output),
            topk_output: Mutex::new(topk_output),
            kernels,
            fused_kernels: Mutex::new(HashMap::new()),
            graph61_kernels,
            graph61_fused: Mutex::new(HashMap::new()),
            graph61_fused_images: Mutex::new(HashMap::new()),
        })
    }

    /// Number of CUDA devices visible to the process.
    pub fn count() -> Result<u32, String> {
        let count = CudaContext::device_count().map_err(|error| error.to_string())?;
        u32::try_from(count).map_err(|_| format!("CUDA returned an invalid device count {count}"))
    }
}

#[cfg(test)]
#[path = "kernels/host_tests.rs"]
mod tests;

use crate::buffer::CudaBuffer;
use cudarc::cublas::{result as cublas_result, sys as cublas_sys, CudaBlas};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use effect_torch_runtime::GgmlKQuant;
use half::bf16;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, Weak};

const COMMON_SOURCE: &str = include_str!("kernels/common.cuh");
const POINTWISE_SOURCE: &str = include_str!("kernels/pointwise.cu");
const TENSOR_SOURCE: &str = include_str!("kernels/tensor.cu");
const LINALG_SOURCE: &str = include_str!("kernels/linalg.cu");
const QUANTIZED_SOURCE: &str = include_str!("kernels/quantized.cu");
const MMVQ_SOURCE: &str = include_str!("kernels/mmvq.cu");
const MMQ_SOURCE: &str = include_str!("kernels/mmq.cu");
const MUSE_QUANTIZE_SOURCE: &str = include_str!("kernels/muse-quantize.cu");
const SGLANG_GGUF_COMMON_SOURCE: &str = include_str!("kernels/sglang-gguf-common.cuh");
const SGLANG_GGUF_VECDOT_SOURCE: &str = include_str!("kernels/sglang-gguf-vecdotq.cuh");
const SGLANG_GGUF_MMQ_SOURCE: &str = include_str!("kernels/sglang-gguf-mmq.cuh");
const NEURAL_SOURCE: &str = include_str!("kernels/neural.cu");
const STATEFUL_SOURCE: &str = include_str!("kernels/stateful.cu");
pub(crate) const CUDA_TOP_K_BLOCKS: usize = 128;
pub(crate) const CUDA_TOP_K_LIMIT: usize = 40;
const SGLANG_GGUF_PRELUDE: &str = r#"
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned short half;
struct half2 { half x; half y; };
namespace c10 { typedef unsigned short BFloat16; }
__device__ __forceinline__ float __half2float(half value) {
    float output;
    asm("cvt.f32.f16 %0, %1;" : "=f"(output) : "h"(value));
    return output;
}
__device__ __forceinline__ half __float2half(float value) {
    half output;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(output) : "f"(value));
    return output;
}
__device__ __forceinline__ float2 __half22float2(half2 value) {
    return {__half2float(value.x), __half2float(value.y)};
}
__device__ __forceinline__ float __low2float(half2 value) {
    return __half2float(value.x);
}
__device__ __forceinline__ half2 __hmul2(half2 left, half2 right) {
    return {
        __float2half(__half2float(left.x) * __half2float(right.x)),
        __float2half(__half2float(left.y) * __half2float(right.y))
    };
}
__device__ __forceinline__ c10::BFloat16 __float2bfloat16(float value) {
    unsigned int bits = __float_as_uint(value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return (c10::BFloat16)(bits >> 16);
}
#define WARP_SIZE 32
#define SGLANG_SHFL_XOR_SYNC(mask, var, lane_mask) __shfl_xor_sync((mask), (var), (lane_mask))
#define SGLANG_SHFL_XOR_SYNC_WIDTH(mask, var, lane_mask, width) __shfl_xor_sync((mask), (var), (lane_mask), (width))
#define SGLANG_LDG(arg) __ldg(arg)
"#;
const F32_PRELUDE: &str = r#"
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
    pub(crate) fill: CudaFunction,
    pub(crate) binary: CudaFunction,
    pub(crate) unary: CudaFunction,
    pub(crate) muse_gate: CudaFunction,
    pub(crate) greedy_argmax: CudaFunction,
    pub(crate) topk: CudaFunction,
    pub(crate) random: CudaFunction,
    pub(crate) sequence: CudaFunction,
    pub(crate) reindex: CudaFunction,
    pub(crate) where_: CudaFunction,
    pub(crate) concat: CudaFunction,
    pub(crate) cast_from_i64: CudaFunction,
    pub(crate) reduce: CudaFunction,
    pub(crate) matmul: CudaFunction,
    pub(crate) index: CudaFunction,
    pub(crate) rms_norm: CudaFunction,
    pub(crate) rms_norm_block: CudaFunction,
    pub(crate) muse_residual_rms_norm: CudaFunction,
    pub(crate) cross_entropy: CudaFunction,
    pub(crate) cross_entropy_backward: CudaFunction,
    pub(crate) chunked_head_ce: CudaFunction,
    pub(crate) chunked_head_ce_backward: CudaFunction,
    pub(crate) conv: CudaFunction,
    pub(crate) linalg: CudaFunction,
    pub(crate) linear: CudaFunction,
    pub(crate) layer_norm: CudaFunction,
    pub(crate) sdpa: CudaFunction,
    pub(crate) rotary: CudaFunction,
    pub(crate) optimizer: CudaFunction,
    pub(crate) short_conv: CudaFunction,
    pub(crate) kda_forward: CudaFunction,
    pub(crate) kda_backward: CudaFunction,
    pub(crate) last_token: CudaFunction,
    pub(crate) kv_append: CudaFunction,
    pub(crate) kv_append_packed: CudaFunction,
    pub(crate) kv_attention: CudaFunction,
    pub(crate) kv_attention_packed: CudaFunction,
    pub(crate) kv_attention_warp: CudaFunction,
    pub(crate) kv_attention_packed_warp: CudaFunction,
    pub(crate) muse_gqa_append_attention_split: CudaFunction,
    pub(crate) muse_gqa_append_attention_grouped: CudaFunction,
    pub(crate) muse_gqa_merge_grouped: CudaFunction,
    pub(crate) quantize_q8: CudaFunction,
    pub(crate) quantized_linear_k_mmv_q8: CudaFunction,
    pub(crate) quantized_linear_k_mmq_q8: CudaFunction,
}

fn compile_module(
    context: &Arc<CudaContext>,
    name: &str,
    source: &str,
) -> Result<Arc<CudaModule>, String> {
    compile_module_sources(context, name, &[COMMON_SOURCE, source])
}

fn compile_module_sources(
    context: &Arc<CudaContext>,
    name: &str,
    sources: &[&str],
) -> Result<Arc<CudaModule>, String> {
    let source = sources.concat();
    let (major, minor) = context
        .compute_capability()
        .map_err(|error| error.to_string())?;
    let arch: &'static str = Box::leak(format!("compute_{major}{minor}").into_boxed_str());
    let ptx = compile_ptx_with_opts(
        &source,
        CompileOptions {
            arch: Some(arch),
            name: Some(name.to_string()),
            use_fast_math: (name == "mmvq.cu").then_some(true),
            ..Default::default()
        },
    )
    .map_err(|error| error.to_string())?;
    context.load_module(ptx).map_err(|error| error.to_string())
}

static DEVICES: LazyLock<Mutex<HashMap<u32, Weak<CudaDevice>>>> = LazyLock::new(Default::default);

/// One CUDA context, stream, and kernel module for a device ordinal.
pub struct CudaDevice {
    pub(crate) ordinal: u32,
    pub(crate) stream: Arc<CudaStream>,
    blas: CudaBlas,
    pub(crate) f32: CudaF32Kernels,
    pub(crate) greedy_argmax_output: Mutex<CudaSlice<u32>>,
    pub(crate) topk_output: Mutex<CudaSlice<f32>>,
    pub(crate) fill_f64: CudaFunction,
    pub(crate) binary_f64: CudaFunction,
    pub(crate) binary_i64: CudaFunction,
    pub(crate) compare_i64: CudaFunction,
    pub(crate) unary_f64: CudaFunction,
    pub(crate) unary_i64: CudaFunction,
    pub(crate) cast_from_i64: CudaFunction,
    pub(crate) cast_to_i64: CudaFunction,
    pub(crate) random_f64: CudaFunction,
    pub(crate) sequence_f64: CudaFunction,
    pub(crate) reindex_f64: CudaFunction,
    pub(crate) where_f64: CudaFunction,
    pub(crate) concat_f64: CudaFunction,
    pub(crate) reduce_f64: CudaFunction,
    pub(crate) matmul_f64: CudaFunction,
    pub(crate) index_f64: CudaFunction,
    pub(crate) rms_norm_f64: CudaFunction,
    pub(crate) rms_norm_block_f64: CudaFunction,
    pub(crate) cross_entropy_f64: CudaFunction,
    pub(crate) cross_entropy_backward_f64: CudaFunction,
    pub(crate) chunked_head_ce_f64: CudaFunction,
    pub(crate) chunked_head_ce_backward_f64: CudaFunction,
    pub(crate) conv_f64: CudaFunction,
    pub(crate) linalg_f64: CudaFunction,
    pub(crate) linear_f64: CudaFunction,
    pub(crate) quantize_q8_f64: CudaFunction,
    pub(crate) quantized_linear_k_mmv_q8: CudaFunction,
    pub(crate) quantized_linear_k_mmq_q8: CudaFunction,
    pub(crate) quantized_embedding_k_f64: CudaFunction,
    pub(crate) quantized_embedding_k_f32: CudaFunction,
    pub(crate) quantize_q8_1_f32: CudaFunction,
    pub(crate) muse_gate_quantize_q8_1_f32: CudaFunction,
    pub(crate) cast_f32_bf16: CudaFunction,
    pub(crate) dequantize_q2_k_bf16: CudaFunction,
    pub(crate) dequantize_q3_k_bf16: CudaFunction,
    pub(crate) dequantize_q4_k_bf16: CudaFunction,
    pub(crate) dequantize_q5_k_bf16: CudaFunction,
    pub(crate) dequantize_q6_k_bf16: CudaFunction,
    pub(crate) rms_norm_quantize_q8_1_f32: CudaFunction,
    pub(crate) residual_rms_norm_quantize_q8_1_f32: CudaFunction,
    pub(crate) mmvq_q2_k_f32: CudaFunction,
    pub(crate) mmvq_q2_k_f32_pair: CudaFunction,
    pub(crate) mmvq_q3_k_f32: CudaFunction,
    pub(crate) mmvq_q3_k_f32_pair: CudaFunction,
    pub(crate) mmvq_q2_q3_k_f32_pair: CudaFunction,
    pub(crate) mmvq_q3_q2_k_f32_pair: CudaFunction,
    pub(crate) mmvq_q4_k_f32: CudaFunction,
    pub(crate) mmvq_q5_k_f32: CudaFunction,
    pub(crate) mmvq_q6_k_f32: CudaFunction,
    pub(crate) mmq_q2_k_f32: CudaFunction,
    pub(crate) mmq_q3_k_f32: CudaFunction,
    pub(crate) mmq_q4_k_f32: CudaFunction,
    pub(crate) mmq_q5_k_f32: CudaFunction,
    pub(crate) mmq_q6_k_f32: CudaFunction,
    pub(crate) mmvq_q2_k_f32_cooperative: CudaFunction,
    pub(crate) mmvq_q3_k_f32_cooperative: CudaFunction,
    pub(crate) mmvq_q4_k_f32_cooperative: CudaFunction,
    pub(crate) mmvq_q5_k_f32_cooperative: CudaFunction,
    pub(crate) mmvq_q6_k_f32_cooperative: CudaFunction,
    pub(crate) layer_norm_f64: CudaFunction,
    pub(crate) sdpa_f64: CudaFunction,
    pub(crate) rotary_f64: CudaFunction,
    pub(crate) optimizer_f64: CudaFunction,
    pub(crate) short_conv_f64: CudaFunction,
    pub(crate) kda_forward_f64: CudaFunction,
    pub(crate) kda_backward_f64: CudaFunction,
    pub(crate) last_token_f64: CudaFunction,
    pub(crate) kv_append_f64: CudaFunction,
    pub(crate) kv_attention_f64: CudaFunction,
    pub(crate) kv_attention_warp_f64: CudaFunction,
    pub(crate) kv_attention_warp_f32: CudaFunction,
    pub(crate) kv_append_packed_f64: CudaFunction,
    pub(crate) kv_attention_packed_f64: CudaFunction,
    pub(crate) kv_attention_packed_warp_f64: CudaFunction,
    pub(crate) kv_attention_packed_warp_f32: CudaFunction,
}

impl CudaDevice {
    /// Returns the process-wide runtime device for `ordinal`.
    pub fn get(ordinal: u32) -> Result<Arc<Self>, String> {
        let mut devices = DEVICES.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(device) = devices.get(&ordinal).and_then(Weak::upgrade) {
            return Ok(device);
        }
        let device = Arc::new(Self::new(ordinal)?);
        devices.insert(ordinal, Arc::downgrade(&device));
        Ok(device)
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
        let blas = CudaBlas::new(stream.clone()).map_err(|error| error.to_string())?;
        let pointwise = compile_module(&context, "pointwise.cu", POINTWISE_SOURCE)?;
        let tensor = compile_module(&context, "tensor.cu", TENSOR_SOURCE)?;
        let linalg = compile_module(&context, "linalg.cu", LINALG_SOURCE)?;
        let pointwise_f32 = compile_module_sources(
            &context,
            "pointwise-f32.cu",
            &[F32_PRELUDE, COMMON_SOURCE, POINTWISE_SOURCE],
        )?;
        let tensor_f32 = compile_module_sources(
            &context,
            "tensor-f32.cu",
            &[F32_PRELUDE, COMMON_SOURCE, TENSOR_SOURCE],
        )?;
        let linalg_f32 = compile_module_sources(
            &context,
            "linalg-f32.cu",
            &[F32_PRELUDE, COMMON_SOURCE, LINALG_SOURCE],
        )?;
        let quantized = compile_module_sources(
            &context,
            "quantized.cu",
            &[
                COMMON_SOURCE,
                SGLANG_GGUF_PRELUDE,
                SGLANG_GGUF_COMMON_SOURCE,
                SGLANG_GGUF_VECDOT_SOURCE,
                QUANTIZED_SOURCE,
            ],
        )?;
        let muse_quantize = compile_module_sources(
            &context,
            "muse-quantize.cu",
            &[
                SGLANG_GGUF_PRELUDE,
                SGLANG_GGUF_COMMON_SOURCE,
                MUSE_QUANTIZE_SOURCE,
            ],
        )?;
        let mmvq = compile_module_sources(
            &context,
            "mmvq.cu",
            &[
                SGLANG_GGUF_PRELUDE,
                SGLANG_GGUF_COMMON_SOURCE,
                SGLANG_GGUF_VECDOT_SOURCE,
                SGLANG_GGUF_MMQ_SOURCE,
                MMVQ_SOURCE,
                MMQ_SOURCE,
            ],
        )?;
        let neural = compile_module(&context, "neural.cu", NEURAL_SOURCE)?;
        let stateful = compile_module(&context, "stateful.cu", STATEFUL_SOURCE)?;
        let neural_f32 = compile_module_sources(
            &context,
            "neural-f32.cu",
            &[F32_PRELUDE, COMMON_SOURCE, NEURAL_SOURCE],
        )?;
        let stateful_f32 = compile_module_sources(
            &context,
            "stateful-f32.cu",
            &[F32_PRELUDE, COMMON_SOURCE, STATEFUL_SOURCE],
        )?;
        let quantized_f32 = compile_module_sources(
            &context,
            "quantized-f32.cu",
            &[F32_PRELUDE, COMMON_SOURCE, QUANTIZED_SOURCE],
        )?;
        let f32 = CudaF32Kernels {
            fill: pointwise_f32
                .load_function("fill_f64")
                .map_err(|error| error.to_string())?,
            binary: pointwise_f32
                .load_function("binary_f64")
                .map_err(|error| error.to_string())?,
            unary: pointwise_f32
                .load_function("unary_f64")
                .map_err(|error| error.to_string())?,
            muse_gate: pointwise_f32
                .load_function("muse_gate_f32")
                .map_err(|error| error.to_string())?,
            greedy_argmax: pointwise_f32
                .load_function("greedy_argmax_f64")
                .map_err(|error| error.to_string())?,
            topk: pointwise_f32
                .load_function("topk_f64")
                .map_err(|error| error.to_string())?,
            random: pointwise_f32
                .load_function("random_f64")
                .map_err(|error| error.to_string())?,
            sequence: pointwise_f32
                .load_function("sequence_f64")
                .map_err(|error| error.to_string())?,
            reindex: pointwise_f32
                .load_function("reindex_f64")
                .map_err(|error| error.to_string())?,
            where_: pointwise_f32
                .load_function("where_f64")
                .map_err(|error| error.to_string())?,
            concat: pointwise_f32
                .load_function("concat_f64")
                .map_err(|error| error.to_string())?,
            cast_from_i64: pointwise_f32
                .load_function("cast_from_i64")
                .map_err(|error| error.to_string())?,
            reduce: tensor_f32
                .load_function("reduce_f64")
                .map_err(|error| error.to_string())?,
            matmul: tensor_f32
                .load_function("matmul_f64")
                .map_err(|error| error.to_string())?,
            index: tensor_f32
                .load_function("index_f64")
                .map_err(|error| error.to_string())?,
            rms_norm: tensor_f32
                .load_function("rms_norm_f64")
                .map_err(|error| error.to_string())?,
            rms_norm_block: tensor_f32
                .load_function("rms_norm_block_f64")
                .map_err(|error| error.to_string())?,
            muse_residual_rms_norm: tensor_f32
                .load_function("muse_residual_rms_norm_f32")
                .map_err(|error| error.to_string())?,
            cross_entropy: tensor_f32
                .load_function("cross_entropy_f64")
                .map_err(|error| error.to_string())?,
            cross_entropy_backward: tensor_f32
                .load_function("cross_entropy_backward_f64")
                .map_err(|error| error.to_string())?,
            chunked_head_ce: tensor_f32
                .load_function("chunked_head_ce_f64")
                .map_err(|error| error.to_string())?,
            chunked_head_ce_backward: tensor_f32
                .load_function("chunked_head_ce_backward_f64")
                .map_err(|error| error.to_string())?,
            conv: linalg_f32
                .load_function("conv_f64")
                .map_err(|error| error.to_string())?,
            linalg: linalg_f32
                .load_function("linalg_f64")
                .map_err(|error| error.to_string())?,
            linear: linalg_f32
                .load_function("linear_f64")
                .map_err(|error| error.to_string())?,
            layer_norm: neural_f32
                .load_function("layer_norm_f64")
                .map_err(|error| error.to_string())?,
            sdpa: neural_f32
                .load_function("sdpa_f64")
                .map_err(|error| error.to_string())?,
            rotary: neural_f32
                .load_function("rotary_f64")
                .map_err(|error| error.to_string())?,
            optimizer: neural_f32
                .load_function("optimizer_f64")
                .map_err(|error| error.to_string())?,
            short_conv: stateful_f32
                .load_function("short_conv_f64")
                .map_err(|error| error.to_string())?,
            kda_forward: stateful_f32
                .load_function("kda_forward_f64")
                .map_err(|error| error.to_string())?,
            kda_backward: stateful_f32
                .load_function("kda_backward_f64")
                .map_err(|error| error.to_string())?,
            last_token: stateful_f32
                .load_function("last_token_f64")
                .map_err(|error| error.to_string())?,
            kv_append: stateful_f32
                .load_function("kv_append_f64")
                .map_err(|error| error.to_string())?,
            kv_append_packed: stateful_f32
                .load_function("kv_append_packed_f64")
                .map_err(|error| error.to_string())?,
            kv_attention: stateful_f32
                .load_function("kv_attention_f64")
                .map_err(|error| error.to_string())?,
            kv_attention_packed: stateful_f32
                .load_function("kv_attention_packed_f64")
                .map_err(|error| error.to_string())?,
            kv_attention_warp: stateful_f32
                .load_function("kv_attention_warp_f32")
                .map_err(|error| error.to_string())?,
            kv_attention_packed_warp: stateful_f32
                .load_function("kv_attention_packed_warp_f32")
                .map_err(|error| error.to_string())?,
            muse_gqa_append_attention_split: stateful_f32
                .load_function("muse_gqa_append_attention_split_f64")
                .map_err(|error| error.to_string())?,
            muse_gqa_append_attention_grouped: stateful_f32
                .load_function("muse_gqa_append_attention_grouped_f64")
                .map_err(|error| error.to_string())?,
            muse_gqa_merge_grouped: stateful_f32
                .load_function("muse_gqa_merge_grouped_f64")
                .map_err(|error| error.to_string())?,
            quantize_q8: quantized_f32
                .load_function("quantize_q8_f64")
                .map_err(|error| error.to_string())?,
            quantized_linear_k_mmv_q8: quantized_f32
                .load_function("quantized_linear_k_mmv_q8")
                .map_err(|error| error.to_string())?,
            quantized_linear_k_mmq_q8: quantized_f32
                .load_function("quantized_linear_k_mmq_q8")
                .map_err(|error| error.to_string())?,
        };
        let fill_f64 = pointwise
            .load_function("fill_f64")
            .map_err(|error| error.to_string())?;
        let binary_f64 = pointwise
            .load_function("binary_f64")
            .map_err(|error| error.to_string())?;
        let binary_i64 = pointwise
            .load_function("binary_i64")
            .map_err(|error| error.to_string())?;
        let compare_i64 = pointwise
            .load_function("compare_i64")
            .map_err(|error| error.to_string())?;
        let unary_f64 = pointwise
            .load_function("unary_f64")
            .map_err(|error| error.to_string())?;
        let unary_i64 = pointwise
            .load_function("unary_i64")
            .map_err(|error| error.to_string())?;
        let cast_from_i64 = pointwise
            .load_function("cast_from_i64")
            .map_err(|error| error.to_string())?;
        let cast_to_i64 = pointwise
            .load_function("cast_to_i64")
            .map_err(|error| error.to_string())?;
        let random_f64 = pointwise
            .load_function("random_f64")
            .map_err(|error| error.to_string())?;
        let sequence_f64 = pointwise
            .load_function("sequence_f64")
            .map_err(|error| error.to_string())?;
        let reindex_f64 = pointwise
            .load_function("reindex_f64")
            .map_err(|error| error.to_string())?;
        let where_f64 = pointwise
            .load_function("where_f64")
            .map_err(|error| error.to_string())?;
        let concat_f64 = pointwise
            .load_function("concat_f64")
            .map_err(|error| error.to_string())?;
        let reduce_f64 = tensor
            .load_function("reduce_f64")
            .map_err(|error| error.to_string())?;
        let matmul_f64 = tensor
            .load_function("matmul_f64")
            .map_err(|error| error.to_string())?;
        let index_f64 = tensor
            .load_function("index_f64")
            .map_err(|error| error.to_string())?;
        let rms_norm_f64 = tensor
            .load_function("rms_norm_f64")
            .map_err(|error| error.to_string())?;
        let rms_norm_block_f64 = tensor
            .load_function("rms_norm_block_f64")
            .map_err(|error| error.to_string())?;
        let cross_entropy_f64 = tensor
            .load_function("cross_entropy_f64")
            .map_err(|error| error.to_string())?;
        let cross_entropy_backward_f64 = tensor
            .load_function("cross_entropy_backward_f64")
            .map_err(|error| error.to_string())?;
        let chunked_head_ce_f64 = tensor
            .load_function("chunked_head_ce_f64")
            .map_err(|error| error.to_string())?;
        let chunked_head_ce_backward_f64 = tensor
            .load_function("chunked_head_ce_backward_f64")
            .map_err(|error| error.to_string())?;
        let conv_f64 = linalg
            .load_function("conv_f64")
            .map_err(|error| error.to_string())?;
        let linalg_f64 = linalg
            .load_function("linalg_f64")
            .map_err(|error| error.to_string())?;
        let linear_f64 = linalg
            .load_function("linear_f64")
            .map_err(|error| error.to_string())?;
        let quantize_q8_f64 = quantized
            .load_function("quantize_q8_f64")
            .map_err(|error| error.to_string())?;
        let quantized_linear_k_mmv_q8 = quantized
            .load_function("quantized_linear_k_mmv_q8")
            .map_err(|error| error.to_string())?;
        let quantized_linear_k_mmq_q8 = quantized
            .load_function("quantized_linear_k_mmq_q8")
            .map_err(|error| error.to_string())?;
        let quantized_embedding_k_f64 = quantized
            .load_function("quantized_embedding_k_f64")
            .map_err(|error| error.to_string())?;
        let quantized_embedding_k_f32 = quantized
            .load_function("quantized_embedding_k_f32")
            .map_err(|error| error.to_string())?;
        let quantize_q8_1_f32 = mmvq
            .load_function("quantize_q8_1_f32")
            .map_err(|error| error.to_string())?;
        let muse_gate_quantize_q8_1_f32 = muse_quantize
            .load_function("muse_gate_quantize_q8_1_f32")
            .map_err(|error| error.to_string())?;
        let cast_f32_bf16 = muse_quantize
            .load_function("cast_f32_bf16")
            .map_err(|error| error.to_string())?;
        let dequantize_q2_k_bf16 = muse_quantize
            .load_function("dequantize_q2_k_bf16")
            .map_err(|error| error.to_string())?;
        let dequantize_q3_k_bf16 = muse_quantize
            .load_function("dequantize_q3_k_bf16")
            .map_err(|error| error.to_string())?;
        let dequantize_q4_k_bf16 = muse_quantize
            .load_function("dequantize_q4_k_bf16")
            .map_err(|error| error.to_string())?;
        let dequantize_q5_k_bf16 = muse_quantize
            .load_function("dequantize_q5_k_bf16")
            .map_err(|error| error.to_string())?;
        let dequantize_q6_k_bf16 = muse_quantize
            .load_function("dequantize_q6_k_bf16")
            .map_err(|error| error.to_string())?;
        let rms_norm_quantize_q8_1_f32 = mmvq
            .load_function("rms_norm_quantize_q8_1_f32")
            .map_err(|error| error.to_string())?;
        let residual_rms_norm_quantize_q8_1_f32 = mmvq
            .load_function("residual_rms_norm_quantize_q8_1_f32")
            .map_err(|error| error.to_string())?;
        let mmvq_q2_k_f32 = mmvq
            .load_function("mmvq_q2_k_f32")
            .map_err(|error| error.to_string())?;
        let mmvq_q2_k_f32_pair = mmvq
            .load_function("mmvq_q2_k_f32_pair")
            .map_err(|error| error.to_string())?;
        let mmvq_q3_k_f32 = mmvq
            .load_function("mmvq_q3_k_f32")
            .map_err(|error| error.to_string())?;
        let mmvq_q3_k_f32_pair = mmvq
            .load_function("mmvq_q3_k_f32_pair")
            .map_err(|error| error.to_string())?;
        let mmvq_q2_q3_k_f32_pair = mmvq
            .load_function("mmvq_q2_q3_k_f32_pair")
            .map_err(|error| error.to_string())?;
        let mmvq_q3_q2_k_f32_pair = mmvq
            .load_function("mmvq_q3_q2_k_f32_pair")
            .map_err(|error| error.to_string())?;
        let mmvq_q4_k_f32 = mmvq
            .load_function("mmvq_q4_k_f32")
            .map_err(|error| error.to_string())?;
        let mmvq_q5_k_f32 = mmvq
            .load_function("mmvq_q5_k_f32")
            .map_err(|error| error.to_string())?;
        let mmvq_q6_k_f32 = mmvq
            .load_function("mmvq_q6_k_f32")
            .map_err(|error| error.to_string())?;
        let mmq_q2_k_f32 = mmvq
            .load_function("mmq_q2_k_f32")
            .map_err(|error| error.to_string())?;
        let mmq_q3_k_f32 = mmvq
            .load_function("mmq_q3_k_f32")
            .map_err(|error| error.to_string())?;
        let mmq_q4_k_f32 = mmvq
            .load_function("mmq_q4_k_f32")
            .map_err(|error| error.to_string())?;
        let mmq_q5_k_f32 = mmvq
            .load_function("mmq_q5_k_f32")
            .map_err(|error| error.to_string())?;
        let mmq_q6_k_f32 = mmvq
            .load_function("mmq_q6_k_f32")
            .map_err(|error| error.to_string())?;
        let mmvq_q2_k_f32_cooperative = mmvq
            .load_function("mmvq_q2_k_f32_cooperative")
            .map_err(|error| error.to_string())?;
        let mmvq_q3_k_f32_cooperative = mmvq
            .load_function("mmvq_q3_k_f32_cooperative")
            .map_err(|error| error.to_string())?;
        let mmvq_q4_k_f32_cooperative = mmvq
            .load_function("mmvq_q4_k_f32_cooperative")
            .map_err(|error| error.to_string())?;
        let mmvq_q5_k_f32_cooperative = mmvq
            .load_function("mmvq_q5_k_f32_cooperative")
            .map_err(|error| error.to_string())?;
        let mmvq_q6_k_f32_cooperative = mmvq
            .load_function("mmvq_q6_k_f32_cooperative")
            .map_err(|error| error.to_string())?;
        let layer_norm_f64 = neural
            .load_function("layer_norm_f64")
            .map_err(|error| error.to_string())?;
        let sdpa_f64 = neural
            .load_function("sdpa_f64")
            .map_err(|error| error.to_string())?;
        let rotary_f64 = neural
            .load_function("rotary_f64")
            .map_err(|error| error.to_string())?;
        let optimizer_f64 = neural
            .load_function("optimizer_f64")
            .map_err(|error| error.to_string())?;
        let short_conv_f64 = stateful
            .load_function("short_conv_f64")
            .map_err(|error| error.to_string())?;
        let kda_forward_f64 = stateful
            .load_function("kda_forward_f64")
            .map_err(|error| error.to_string())?;
        let kda_backward_f64 = stateful
            .load_function("kda_backward_f64")
            .map_err(|error| error.to_string())?;
        let last_token_f64 = stateful
            .load_function("last_token_f64")
            .map_err(|error| error.to_string())?;
        let kv_append_f64 = stateful
            .load_function("kv_append_f64")
            .map_err(|error| error.to_string())?;
        let kv_attention_f64 = stateful
            .load_function("kv_attention_f64")
            .map_err(|error| error.to_string())?;
        let kv_attention_warp_f64 = stateful
            .load_function("kv_attention_warp_f64")
            .map_err(|error| error.to_string())?;
        let kv_attention_warp_f32 = stateful
            .load_function("kv_attention_warp_f32")
            .map_err(|error| error.to_string())?;
        let kv_append_packed_f64 = stateful
            .load_function("kv_append_packed_f64")
            .map_err(|error| error.to_string())?;
        let kv_attention_packed_f64 = stateful
            .load_function("kv_attention_packed_f64")
            .map_err(|error| error.to_string())?;
        let kv_attention_packed_warp_f64 = stateful
            .load_function("kv_attention_packed_warp_f64")
            .map_err(|error| error.to_string())?;
        let kv_attention_packed_warp_f32 = stateful
            .load_function("kv_attention_packed_warp_f32")
            .map_err(|error| error.to_string())?;
        let greedy_argmax_output =
            unsafe { stream.alloc::<u32>(2) }.map_err(|error| error.to_string())?;
        let topk_output =
            unsafe { stream.alloc::<f32>(CUDA_TOP_K_BLOCKS * (2 * CUDA_TOP_K_LIMIT + 1)) }
                .map_err(|error| error.to_string())?;
        Ok(Self {
            ordinal,
            stream,
            blas,
            f32,
            greedy_argmax_output: Mutex::new(greedy_argmax_output),
            topk_output: Mutex::new(topk_output),
            fill_f64,
            binary_f64,
            binary_i64,
            compare_i64,
            unary_f64,
            unary_i64,
            cast_from_i64,
            cast_to_i64,
            random_f64,
            sequence_f64,
            reindex_f64,
            where_f64,
            concat_f64,
            reduce_f64,
            matmul_f64,
            index_f64,
            rms_norm_f64,
            rms_norm_block_f64,
            cross_entropy_f64,
            cross_entropy_backward_f64,
            chunked_head_ce_f64,
            chunked_head_ce_backward_f64,
            conv_f64,
            linalg_f64,
            linear_f64,
            quantize_q8_f64,
            quantized_linear_k_mmv_q8,
            quantized_linear_k_mmq_q8,
            quantized_embedding_k_f64,
            quantized_embedding_k_f32,
            quantize_q8_1_f32,
            muse_gate_quantize_q8_1_f32,
            cast_f32_bf16,
            dequantize_q2_k_bf16,
            dequantize_q3_k_bf16,
            dequantize_q4_k_bf16,
            dequantize_q5_k_bf16,
            dequantize_q6_k_bf16,
            rms_norm_quantize_q8_1_f32,
            residual_rms_norm_quantize_q8_1_f32,
            mmvq_q2_k_f32,
            mmvq_q2_k_f32_pair,
            mmvq_q3_k_f32,
            mmvq_q3_k_f32_pair,
            mmvq_q2_q3_k_f32_pair,
            mmvq_q3_q2_k_f32_pair,
            mmvq_q4_k_f32,
            mmvq_q5_k_f32,
            mmvq_q6_k_f32,
            mmq_q2_k_f32,
            mmq_q3_k_f32,
            mmq_q4_k_f32,
            mmq_q5_k_f32,
            mmq_q6_k_f32,
            mmvq_q2_k_f32_cooperative,
            mmvq_q3_k_f32_cooperative,
            mmvq_q4_k_f32_cooperative,
            mmvq_q5_k_f32_cooperative,
            mmvq_q6_k_f32_cooperative,
            layer_norm_f64,
            sdpa_f64,
            rotary_f64,
            optimizer_f64,
            short_conv_f64,
            kda_forward_f64,
            kda_backward_f64,
            last_token_f64,
            kv_append_f64,
            kv_attention_f64,
            kv_attention_warp_f64,
            kv_attention_warp_f32,
            kv_append_packed_f64,
            kv_attention_packed_f64,
            kv_attention_packed_warp_f64,
            kv_attention_packed_warp_f32,
        })
    }

    pub(crate) fn dequantize_weight_bf16(
        &self,
        packed: &CudaBuffer<u8>,
        codec: GgmlKQuant,
        rows: u32,
        columns: u32,
    ) -> Result<CudaSlice<bf16>, String> {
        let elements = usize::try_from(rows)
            .ok()
            .and_then(|rows| rows.checked_mul(columns as usize))
            .ok_or_else(|| "CUDA BF16 weight size overflowed usize".to_string())?;
        if !elements.is_multiple_of(256) {
            return Err("CUDA BF16 weight size is not K-quant block aligned".to_string());
        }
        let blocks = u32::try_from(elements / 256)
            .map_err(|_| "CUDA BF16 weight block count exceeds u32".to_string())?;
        let mut output =
            unsafe { self.stream.alloc::<bf16>(elements) }.map_err(|error| error.to_string())?;
        let (function, threads) = match codec {
            GgmlKQuant::Q2K => (&self.dequantize_q2_k_bf16, 64),
            GgmlKQuant::Q3K => (&self.dequantize_q3_k_bf16, 64),
            GgmlKQuant::Q4K => (&self.dequantize_q4_k_bf16, 32),
            GgmlKQuant::Q5K => (&self.dequantize_q5_k_bf16, 64),
            GgmlKQuant::Q6K => (&self.dequantize_q6_k_bf16, 64),
        };
        let mut launch = self.stream.launch_builder(function);
        launch.arg(packed);
        launch.arg(&mut output);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| error.to_string())?;
        Ok(output)
    }

    pub(crate) fn gemm_bf16_f32(
        &self,
        weight: &CudaSlice<bf16>,
        input: &CudaBuffer<bf16>,
        output: &mut CudaBuffer<f32>,
        rows: u32,
        vectors: u32,
        columns: u32,
    ) -> Result<(), String> {
        let (weight, _weight_sync) = weight.device_ptr(&self.stream);
        let (input, _input_sync) = input.device_ptr(&self.stream);
        let (output, _output_sync) = output.device_ptr_mut(&self.stream);
        let alpha = 1.0f32;
        let beta = 0.0f32;
        unsafe {
            cublas_result::gemm_ex(
                *self.blas.handle(),
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                rows as i32,
                vectors as i32,
                columns as i32,
                &alpha as *const f32 as *const _,
                weight as *const _,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                columns as i32,
                input as *const _,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                columns as i32,
                &beta as *const f32 as *const _,
                output as *mut _,
                cublas_sys::cudaDataType_t::CUDA_R_32F,
                rows as i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        }
        .map_err(|error| error.to_string())
    }

    /// Number of CUDA devices visible to the process.
    pub fn count() -> Result<u32, String> {
        let count = CudaContext::device_count().map_err(|error| error.to_string())?;
        u32::try_from(count).map_err(|_| format!("CUDA returned an invalid device count {count}"))
    }
}

use crate::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, Weak};

const TYPED_HEADER: &str = include_str!("kernels/typed.cuh");
const TYPED_SOURCE: &str = include_str!("kernels/typed.cu");
const COMMON_SOURCE: &str = include_str!("kernels/common.cuh");
const COMPUTE_WRAPPERS: &str = include_str!("kernels/compute.cu");
const POINTWISE_SOURCE: &str = include_str!("kernels/pointwise.cu");
const TENSOR_SOURCE: &str = include_str!("kernels/tensor.cu");
const LINALG_SOURCE: &str = include_str!("kernels/linalg.cu");
const NEURAL_SOURCE: &str = include_str!("kernels/neural.cu");
const STATEFUL_SOURCE: &str = include_str!("kernels/stateful.cu");
const QUANTIZED_SOURCE: &str = include_str!("kernels/quantized.cu");
const CACHE_SOURCE: &str = include_str!("kernels/cache.cu");

pub(crate) const CUDA_TOP_K_BLOCKS: usize = 128;
pub(crate) const CUDA_TOP_K_LIMIT: usize = 40;

// Applied after typed.cuh, so descriptor fields, casts, indexes, masks and
// persistent state do not inherit the compute type substitution.
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
    pub(crate) greedy_argmax: CudaFunction,
    pub(crate) topk: CudaFunction,
}

const TYPED_KERNELS: &[&str] = &[
    "et_fill",
    "et_convert",
    "et_binary",
    "et_unary",
    "et_reindex",
    "et_where",
    "et_concat",
    "et_index",
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
        kernels: &["et_random"],
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

fn compile_module(
    context: &Arc<CudaContext>,
    name: &str,
    sources: &[&str],
) -> Result<Arc<CudaModule>, String> {
    let source = sources.join("\n");
    let (major, minor) = context
        .compute_capability()
        .map_err(|error| error.to_string())?;
    let arch: &'static str = Box::leak(format!("compute_{major}{minor}").into_boxed_str());
    let ptx = compile_ptx_with_opts(
        &source,
        CompileOptions {
            arch: Some(arch),
            name: Some(name.to_string()),
            // Preserve F32 operations in canonical packed decode.
            fmad: Some(false),
            ..Default::default()
        },
    )
    .map_err(|error| format!("CUDA compile {name}: {error}"))?;
    context.load_module(ptx).map_err(|error| error.to_string())
}

fn load(module: &Arc<CudaModule>, name: &str) -> Result<CudaFunction, String> {
    module
        .load_function(name)
        .map_err(|error| format!("CUDA kernel {name}: {error}"))
}

static DEVICES: LazyLock<Mutex<HashMap<u32, Weak<CudaDevice>>>> = LazyLock::new(Default::default);

/// One CUDA context, stream, and eagerly compiled kernel registry per device.
pub struct CudaDevice {
    pub(crate) ordinal: u32,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) f32: CudaF32Kernels,
    pub(crate) cublas: CudaBlas,
    pub(crate) greedy_argmax_output: Mutex<CudaSlice<u32>>,
    pub(crate) topk_output: Mutex<CudaSlice<f32>>,
    kernels: HashMap<String, CudaFunction>,
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
        let typed = compile_module(&context, "typed.cu", &[TYPED_HEADER, TYPED_SOURCE])?;
        for &name in TYPED_KERNELS {
            kernels.insert(name.to_string(), load(&typed, name)?);
        }
        let quantized =
            compile_module(&context, "quantized.cu", &[TYPED_HEADER, QUANTIZED_SOURCE])?;
        for name in ["et_quantized_linear", "et_quantized_embedding"] {
            kernels.insert(name.to_string(), load(&quantized, name)?);
        }
        let cache = compile_module(&context, "cache.cu", &[TYPED_HEADER, CACHE_SOURCE])?;
        kernels.insert(
            "et_kv_attention".to_string(),
            load(&cache, "et_kv_attention")?,
        );
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
                        COMPUTE_WRAPPERS,
                    ],
                )?;
                for &name in definition.kernels {
                    kernels.insert(format!("{name}_{suffix}"), load(&module, name)?);
                }
                if definition.name == "pointwise" && suffix == "f32" {
                    sampling = Some(CudaF32Kernels {
                        greedy_argmax: load(&module, "greedy_argmax_f64")?,
                        topk: load(&module, "topk_f64")?,
                    });
                }
            }
        }
        let cublas = CudaBlas::new(stream.clone())?;
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
            greedy_argmax_output: Mutex::new(greedy_argmax_output),
            topk_output: Mutex::new(topk_output),
            kernels,
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

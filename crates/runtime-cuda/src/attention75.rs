//! Opt-in offline-compiled BF16 online attention over invocation-owned KV tables.
//! Arithmetic is intentionally different from the default stepwise contract.
use crate::executable::CudaKernelArgs;
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Arc;

pub(crate) const DIRECTORY_ENV: &str = "EFFECT_TORCH_CUDA_ATTENTION75_DIRECTORY";

#[cfg(test)]
thread_local! { static TEST_LAUNCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
#[cfg(test)]
pub(crate) fn test_launches() -> usize {
    TEST_LAUNCHES.with(|count| count.get())
}

struct Variant {
    dim: u64,
    q_dtype: u32,
    output_dtype: u32,
    function: CudaFunction,
    shared: u32,
    extra_zero_parameters: usize,
}

pub(crate) struct Attention75 {
    variants: Vec<Variant>,
}

fn number(value: &serde_json::Value, key: &str) -> Result<u64, String> {
    value[key]
        .as_u64()
        .ok_or_else(|| format!("attention75 manifest missing integer {key}"))
}

fn string<'a>(value: &'a serde_json::Value, key: &str) -> Result<&'a str, String> {
    value[key]
        .as_str()
        .ok_or_else(|| format!("attention75 manifest missing string {key}"))
}

impl Attention75 {
    pub(crate) fn from_env(context: &Arc<CudaContext>) -> Result<Option<Arc<Self>>, String> {
        let Some(directory) = std::env::var_os(DIRECTORY_ENV) else {
            return Ok(None);
        };
        let directory = Path::new(&directory);
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.join("manifest.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let abi = number(&manifest, "abi")?;
        if !matches!(abi, 1 | 2)
            || context.compute_capability().map_err(|e| e.to_string())? != (12, 0)
            || manifest["compute_capability"] != serde_json::json!([12, 0])
        {
            return Err("attention75 requires ABI1/2 and compute capability12.0".into());
        }
        let rows = manifest["variants"]
            .as_array()
            .ok_or("attention75 variants missing")?;
        if rows.len() != if abi == 1 { 2 } else { 4 } {
            return Err("attention75 requires a complete dtype/dimension variant matrix".into());
        }
        let mut variants = Vec::new();
        for row in rows {
            let dim = number(row, "dim")?;
            let q_dtype = if abi == 1 { 1 } else { number(row, "q_dtype")? };
            let output_dtype = if abi == 1 {
                1
            } else {
                number(row, "output_dtype")?
            };
            let shared = number(row, "shared")?;
            let extra = number(row, "extra_zero_u64_parameters")?;
            if !variant_key_valid(abi, dim, q_dtype, output_dtype)
                || variants.iter().any(|v: &Variant| {
                    v.dim == dim
                        && u64::from(v.q_dtype) == q_dtype
                        && u64::from(v.output_dtype) == output_dtype
                })
                || number(row, "num_warps")? != 4
                || shared > 99_000
                || extra > 2
            {
                return Err("attention75 invalid launch manifest".into());
            }
            let file = string(row, "ptx")?;
            if Path::new(file).file_name().and_then(|s| s.to_str()) != Some(file) {
                return Err("attention75 PTX must be a manifest-local filename".into());
            }
            let source = std::fs::read(directory.join(file)).map_err(|e| e.to_string())?;
            if format!("{:x}", Sha256::digest(&source)) != string(row, "sha256")? {
                return Err("attention75 PTX digest mismatch".into());
            }
            let source = String::from_utf8(source).map_err(|e| e.to_string())?;
            let module = context
                .load_module(Ptx::from_src(source))
                .map_err(|e| e.to_string())?;
            let function = module
                .load_function(string(row, "entry")?)
                .map_err(|e| e.to_string())?;
            function
                .set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared as i32,
                )
                .map_err(|e| e.to_string())?;
            variants.push(Variant {
                dim,
                q_dtype: q_dtype as u32,
                output_dtype: output_dtype as u32,
                function,
                shared: shared as u32,
                extra_zero_parameters: extra as usize,
            });
        }
        Ok(Some(Arc::new(Self { variants })))
    }

    fn geometry_supported(args: &CudaKernelArgs) -> bool {
        matches!((args.input_dtypes[0], args.output_dtype), (1, 1) | (3, 3))
            && args.compute_dtype == 1
            && args.integers[1] == 3
            && args.integers[2] > 0
            && args.integers[5] > 0
            && args.integers[6] == 1
            && args.integers[8] == 3
            && args.integers[11] == 16
            && matches!(args.integers[10], 256 | 512)
            && (1..=256).contains(&args.integers[7])
            && args.integers[3] <= u32::MAX as u64
            && args.integers[9] <= u32::MAX as u64
            && args.scalars[0].is_finite()
            && (args.scalars[0] as f32).is_finite()
            && args.operation <= 1
            && [
                args.inputs[0],
                args.inputs[3],
                args.inputs[7],
                args.output,
                args.scratch[0],
                args.scratch[3],
                args.metadata,
            ]
            .into_iter()
            .all(|p| p != 0)
    }

    pub(crate) fn supports(&self, args: &CudaKernelArgs) -> bool {
        Self::geometry_supported(args)
            && self.variants.iter().any(|v| {
                v.dim == args.integers[10]
                    && v.q_dtype == args.input_dtypes[0]
                    && v.output_dtype == args.output_dtype
            })
    }

    /// The existing invocation fence retains all pointers through completion.
    /// No persistent mutable scratch or host copies are introduced here.
    pub(crate) unsafe fn launch(
        &self,
        stream: &Arc<CudaStream>,
        args: &CudaKernelArgs,
        lane: usize,
        positions: usize,
        rows: usize,
    ) -> Result<(), String> {
        if !self.supports(args)
            || positions == 0
            || positions as u64 > args.integers[9]
            || rows == 0
            || rows as u64 > args.integers[7]
            || lane > u32::MAX as usize
            || args.integers[5]
                .checked_mul(args.integers[2])
                .is_none_or(|lanes| lane as u64 >= lanes)
        {
            return Err("attention75 invalid native admission".into());
        }
        let variant = self
            .variants
            .iter()
            .find(|v| {
                v.dim == args.integers[10]
                    && v.q_dtype == args.input_dtypes[0]
                    && v.output_dtype == args.output_dtype
            })
            .ok_or("attention75 dimension variant missing")?;
        let lane = lane as u32;
        let sequence = (u64::from(lane) / args.integers[2]) as u32;
        let tokens = args.integers[7] as u32;
        let rows = rows as u32;
        let positions = positions as u32;
        let window = args.integers[3] as u32;
        let scale = args.scalars[0] as f32;
        let noncausal = u32::from(args.integers[4] != 0);
        let token_major = args.operation;
        let zeros = [0_u64; 2];
        let mut launch = stream.launch_builder(&variant.function);
        launch.arg(&args.inputs[0]);
        launch.arg(&args.inputs[3]);
        launch.arg(&args.output);
        launch.arg(&args.scratch[3]);
        launch.arg(&args.scratch[0]);
        launch.arg(&args.inputs[7]);
        launch.arg(&args.metadata);
        launch.arg(&lane);
        launch.arg(&sequence);
        launch.arg(&tokens);
        launch.arg(&rows);
        launch.arg(&positions);
        launch.arg(&window);
        launch.arg(&scale);
        launch.arg(&args.error_context);
        launch.arg(&noncausal);
        launch.arg(&token_major);
        for zero in &zeros[..variant.extra_zero_parameters] {
            launch.arg(zero);
        }
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (rows.div_ceil(16), 16, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: variant.shared,
            })
        }
        .map_err(|e| e.to_string())?;
        #[cfg(test)]
        TEST_LAUNCHES.with(|count| count.set(count.get() + 1));
        Ok(())
    }
}

fn variant_key_valid(abi: u64, dim: u64, q: u64, out: u64) -> bool {
    matches!(dim, 256 | 512) && ((q == 1 && out == 1) || (abi == 2 && q == 3 && out == 3))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires CUDA SM120 and an ABI1 ATTENTION75_DIRECTORY"]
    fn attention82_legacy_manifest_loads_without_bf16_relabeling() {
        let device = crate::CudaDevice::get(0).unwrap();
        let native = device.cublas.attention75().expect("ABI1 variants required");
        assert_eq!(native.variants.len(), 2);
        assert!(native
            .variants
            .iter()
            .all(|v| v.q_dtype == 1 && v.output_dtype == 1));
    }
    #[test]
    fn admission_rejects_other_storage_and_geometry() {
        let mut a = CudaKernelArgs::default();
        a.input_dtypes[0] = 1;
        a.output_dtype = 1;
        a.compute_dtype = 1;
        a.inputs = [16; 8];
        a.scratch = [16; 4];
        a.output = 16;
        a.metadata = 16;
        a.integers[1] = 3;
        a.integers[2] = 1;
        a.integers[5] = 1;
        a.integers[6] = 1;
        a.integers[7] = 256;
        a.integers[8] = 3;
        a.integers[9] = 4096;
        a.integers[10] = 256;
        a.integers[11] = 16;
        assert!(Attention75::geometry_supported(&a));
        for (slot, value) in [(1, 1), (2, 0), (6, 0), (7, 257), (8, 1), (10, 128), (11, 8)] {
            let mut bad = a;
            bad.integers[slot] = value;
            assert!(!Attention75::geometry_supported(&bad));
        }
        a.input_dtypes[0] = 3;
        assert!(!Attention75::geometry_supported(&a));
        a.output_dtype = 3;
        assert!(Attention75::geometry_supported(&a));
        a.input_dtypes[0] = 1;
        assert!(!Attention75::geometry_supported(&a));
    }
    #[test]
    fn abi82_never_relabels_a_legacy_f32_pointer_variant() {
        for dim in [256, 512] {
            assert!(variant_key_valid(1, dim, 1, 1));
            assert!(!variant_key_valid(1, dim, 3, 3));
            assert!(variant_key_valid(2, dim, 3, 3));
            assert!(!variant_key_valid(2, dim, 1, 3));
            assert!(!variant_key_valid(2, dim, 3, 1));
        }
        assert!(!variant_key_valid(2, 128, 3, 3));
    }
}

//! Optional complete, fixed-image BF16 self-conditioning softmax pipeline.
//! All scratch and output storage is invocation-owned; no execution fences change.
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use sha2::{Digest, Sha256};
use std::{path::Path, sync::Arc};

pub(crate) const DIRECTORY_ENV: &str = "EFFECT_TORCH_CUDA_TRITON_SOFTMAX100_DIRECTORY";
pub(crate) const ROWS: usize = 256;
pub(crate) const WIDTH: usize = 262144;
pub(crate) const SCRATCH_ELEMENTS: usize = 2048;
pub(crate) fn admitted(
    rows: usize,
    width: usize,
    dtype: effect_torch_runtime::DType,
    dense: bool,
) -> bool {
    rows == ROWS && width == WIDTH && dtype == effect_torch_runtime::DType::BF16 && dense
}
const MATRIX_BYTES: u64 = (ROWS * WIDTH * 2) as u64;
const IMAGES: [(&str, &str, u32, u32, u32, usize, usize); 5] = [
    (
        "triton_red_fused__to_copy_prepare_softmax_online_20",
        "64d01104167bf740fb918941c33660fccddfcf89bcade05767bb6f39f876f33f",
        768,
        512,
        64,
        2,
        2,
    ),
    (
        "triton_per_fused_logsumexp_2",
        "1cb7b99b44f365d48c690babc34c33eb27ce52e6274df46d28bad9b90cb3ae4e",
        2,
        64,
        512,
        2,
        2,
    ),
    (
        "triton_red_fused__to_copy_prepare_softmax_online_21",
        "48a18b46910598fadf376118378a9f76de189892e8bec693b5cf67dba09a7a45",
        768,
        512,
        4096,
        3,
        2,
    ),
    (
        "triton_per_fused_logsumexp_4",
        "b2f3e5de4021ec83c6ac60c2b5cfc1282c82b6ab808dc158037c9c2baa791569",
        8,
        64,
        128,
        2,
        2,
    ),
    (
        "triton_poi_fused__softmax__to_copy_exp_sub_22",
        "31f2fd1aebff66fab15042ef8b7f4708bc15567c3442ed25764c92d1c05ecbbd",
        65536,
        128,
        0,
        4,
        1,
    ),
];

pub(crate) struct Softmax100 {
    context: Arc<CudaContext>,
    functions: Vec<CudaFunction>,
}

#[derive(Clone, Copy)]
pub(crate) struct Invocation100 {
    pub(crate) input: u64,
    pub(crate) output: u64,
    /// F32[2048]: partial maximum[768], maximum[256], partial sum[768], sum[256].
    pub(crate) scratch: u64,
}

impl Invocation100 {
    pub(crate) fn validate(&self) -> Result<(), String> {
        let ranges = [
            (self.input, MATRIX_BYTES),
            (self.output, MATRIX_BYTES),
            (self.scratch, (SCRATCH_ELEMENTS * 4) as u64),
        ]
        .map(|(start, bytes)| {
            if start == 0 || start % 16 != 0 {
                return Err("softmax100 requires nonzero aligned pointers".to_string());
            }
            let end = start
                .checked_add(bytes)
                .ok_or("softmax100 pointer overflow")?;
            Ok((start, end))
        });
        let ranges = ranges.into_iter().collect::<Result<Vec<_>, String>>()?;
        for i in 0..ranges.len() {
            for j in i + 1..ranges.len() {
                if ranges[i].0 < ranges[j].1 && ranges[j].0 < ranges[i].1 {
                    return Err("softmax100 input, scratch and output must be separate".into());
                }
            }
        }
        Ok(())
    }
}

fn validate_manifest(manifest: &serde_json::Value) -> Result<(), String> {
    if manifest["abi"] != 1 || manifest["shape"] != serde_json::json!([ROWS, WIDTH]) {
        return Err("softmax100 manifest shape/ABI mismatch".into());
    }
    let variants = manifest["variants"]
        .as_array()
        .ok_or("softmax100 missing variants")?;
    if variants.len() != IMAGES.len() {
        return Err("softmax100 requires all five stages".into());
    }
    for (i, (row, &(symbol, hash, grid, block, shared, pointers, integers))) in
        variants.iter().zip(IMAGES.iter()).enumerate()
    {
        let types = [vec!["u64"; pointers], vec!["u32"; integers], vec!["u64"; 2]].concat();
        if row["file"] != format!("stage{i}.ptx")
            || row["symbol"] != symbol
            || row["sha256"] != hash
            || row["grid"] != grid
            || row["block"] != block
            || row["shared"] != shared
            || row["parameters"] != serde_json::json!(types)
            || row["metadata"]["num_warps"] != block / 32
            || row["metadata"]["shared"] != shared
            || row["metadata"]["global_scratch_size"] != 0
            || row["metadata"]["profile_scratch_size"] != 0
        {
            return Err(format!("softmax100 stage{i} image/launch/ABI mismatch"));
        }
    }
    Ok(())
}

impl Softmax100 {
    pub(crate) fn from_env(context: &Arc<CudaContext>) -> Result<Option<Arc<Self>>, String> {
        let Some(directory) = std::env::var_os(DIRECTORY_ENV) else {
            return Ok(None);
        };
        Self::from_directory(context, Path::new(&directory)).map(Some)
    }

    pub(crate) fn from_directory(
        context: &Arc<CudaContext>,
        directory: &Path,
    ) -> Result<Arc<Self>, String> {
        if context.compute_capability().map_err(|e| e.to_string())? != (12, 0) {
            return Err("softmax100 requires SM120".into());
        }
        let manifest = serde_json::from_slice(
            &std::fs::read(directory.join("launch-manifest.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        validate_manifest(&manifest)?;
        let mut functions = Vec::with_capacity(5);
        for (i, &(symbol, expected, _, _, shared, _, _)) in IMAGES.iter().enumerate() {
            let source = std::fs::read(directory.join(format!("stage{i}.ptx")))
                .map_err(|e| e.to_string())?;
            if format!("{:x}", Sha256::digest(&source)) != expected {
                return Err(format!("softmax100 stage{i} PTX digest mismatch"));
            }
            let source = String::from_utf8(source).map_err(|e| e.to_string())?;
            let module = context
                .load_module(Ptx::from_src(source))
                .map_err(|e| e.to_string())?;
            let function = module.load_function(symbol).map_err(|e| e.to_string())?;
            if shared > 0 {
                function.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared as i32).map_err(|e| e.to_string())?;
            }
            functions.push(function);
        }
        Ok(Arc::new(Self {
            context: context.clone(),
            functions,
        }))
    }

    /// Caller proves the fixed extents and retains this plan and all three owners
    /// through the existing completion fence. Every write range is disjoint.
    pub(crate) unsafe fn launch(
        &self,
        stream: &Arc<CudaStream>,
        args: Invocation100,
    ) -> Result<(), String> {
        if !Arc::ptr_eq(&self.context, stream.context()) {
            return Err("softmax100 stream context mismatch".into());
        }
        args.validate()?;
        let pm = args.scratch;
        let gm = pm + 768 * 4;
        let ps = gm + 256 * 4;
        let gs = ps + 768 * 4;
        let pointers = [
            vec![args.input, pm],
            vec![pm, gm],
            vec![args.input, gm, ps],
            vec![ps, gs],
            vec![args.input, gm, gs, args.output],
        ];
        let integers = [
            vec![768_u32, 87382],
            vec![256, 3],
            vec![768, 87382],
            vec![256, 3],
            vec![(ROWS * WIDTH) as u32],
        ];
        let zero = 0_u64;
        for i in 0..5 {
            let mut launch = stream.launch_builder(&self.functions[i]);
            for pointer in &pointers[i] {
                launch.arg(pointer);
            }
            for integer in &integers[i] {
                launch.arg(integer);
            }
            launch.arg(&zero).arg(&zero);
            let (_, _, grid, block, shared, _, _) = IMAGES[i];
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: shared,
                })
            }
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "triton_softmax100_tests.rs"]
mod tests;

//! Fixed-image local QKV normalization and RoPE, retaining ordinary output layout.
//!
//! Admission is separate from dispatch. The caller proves the packed projection
//! and absorbed normalization paths private, preserves the original selected66
//! table bytes, and retains every owner through the existing completion fence.
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Arc;

pub(crate) const DIRECTORY_ENV: &str = "EFFECT_TORCH_CUDA_TRITON_NORMROPE101_DIRECTORY";
pub(crate) const TOKENS: usize = 256;
pub(crate) const WIDTH: usize = 256;
pub(crate) const Q_HEADS: usize = 16;
pub(crate) const KV_HEADS: usize = 8;
pub(crate) const PACKED_BYTES: usize = TOKENS * 8192 * 2;
pub(crate) const Q_BYTES: usize = TOKENS * Q_HEADS * WIDTH * 2;
pub(crate) const KV_BYTES: usize = TOKENS * KV_HEADS * WIDTH * 2;
pub(crate) const HALF_TABLE_BYTES: usize = TOKENS * WIDTH / 2 * 2;
pub(crate) const TABLE_BYTES: usize = TOKENS * WIDTH * 2;
pub(crate) const POSITION_BYTES: usize = TOKENS * 8;
pub(crate) const Q_SUM_BYTES: usize = TOKENS * Q_HEADS * 4;
pub(crate) const K_SUM_BYTES: usize = TOKENS * KV_HEADS * 4;
const REDUCTION: &str = "triton_red_fused_3";
const POINTWISE: &str = "triton_poi_fused_4";
const REDUCTION_SHA: &str = "71925415c29170bbdc09bffba6eb75f3b8670a4b32e3227b074c6b3056099889";
const POINTWISE_SHA: &str = "b5c757f91fc75f56e4fb06b131ec21a53d1f739eea4ce9e631b8329fb58b947a";
const HELPERS_SHA: &str = "c3984b2208ee1cdeb4094fb975be3136b2cbe604dafbb744f6fad0bd0a9f5e33";

pub(crate) struct NormRope101 {
    context: Arc<CudaContext>,
    prepare: CudaFunction,
    reduction: CudaFunction,
    pointwise: CudaFunction,
    transpose: CudaFunction,
}

/// Every mutable range is invocation-owned and disjoint from all other ranges.
/// Final Q/K/V values preserve [1,H,T,D] physical layout and ordinary leases.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Invocation101 {
    /// Immutable BF16 [T256,8192] packed77 projection result, Q/K/V offsets0/4096/6144.
    pub(crate) packed: u64,
    pub(crate) query_weight: u64,
    pub(crate) key_weight: u64,
    /// Original BF16 half-table owners or proven repeated full-width table owners.
    /// Both retain real request positions and remain borrowed throughout dispatch.
    pub(crate) cosine: u64,
    pub(crate) sine: u64,
    pub(crate) cosine_stride: u32,
    pub(crate) sine_stride: u32,
    /// Private BF16 [T256,cos128+sin128] and real GPU-produced I64 identity row indices.
    pub(crate) table: u64,
    pub(crate) positions: u64,
    pub(crate) sum_q: u64,
    pub(crate) sum_k: u64,
    /// Private token-major Q/K/V scratch; the unchanged PTX writes these ranges.
    pub(crate) raw_query: u64,
    pub(crate) raw_key: u64,
    pub(crate) raw_value: u64,
    /// Original head-major Q/K/V planned outputs, written by the screened transpose.
    pub(crate) query: u64,
    pub(crate) key: u64,
    pub(crate) value: u64,
}

#[derive(Clone, Copy)]
struct Range {
    begin: u64,
    end: u64,
}
fn range(pointer: u64, bytes: usize) -> Result<Range, String> {
    if pointer == 0 || pointer % 16 != 0 || bytes == 0 {
        return Err("normrope101 requires nonzero 16-byte-aligned pointers".into());
    }
    Ok(Range {
        begin: pointer,
        end: pointer
            .checked_add(bytes as u64)
            .ok_or("normrope101 pointer range overflow")?,
    })
}
impl Invocation101 {
    pub(crate) fn validate(&self) -> Result<(), String> {
        let source_bytes = |stride: u32| match stride {
            128 | 256 => Ok(TOKENS * stride as usize * 2),
            _ => Err("normrope101 table source strides must be 128 or 256".to_string()),
        };
        let read = [
            range(self.packed, PACKED_BYTES)?,
            range(self.query_weight, WIDTH * 2)?,
            range(self.key_weight, WIDTH * 2)?,
            range(self.cosine, source_bytes(self.cosine_stride)?)?,
            range(self.sine, source_bytes(self.sine_stride)?)?,
        ];
        let write = [
            range(self.table, TABLE_BYTES)?,
            range(self.positions, POSITION_BYTES)?,
            range(self.sum_q, Q_SUM_BYTES)?,
            range(self.sum_k, K_SUM_BYTES)?,
            range(self.raw_query, Q_BYTES)?,
            range(self.raw_key, KV_BYTES)?,
            range(self.raw_value, KV_BYTES)?,
            range(self.query, Q_BYTES)?,
            range(self.key, KV_BYTES)?,
            range(self.value, KV_BYTES)?,
        ];
        let overlaps = |a: Range, b: Range| a.begin < b.end && b.begin < a.end;
        for (index, destination) in write.iter().enumerate() {
            if read.iter().any(|source| overlaps(*source, *destination))
                || write[..index]
                    .iter()
                    .any(|other| overlaps(*other, *destination))
            {
                return Err(
                    "normrope101 mutable storage overlaps a borrowed input or another output"
                        .into(),
                );
            }
        }
        Ok(())
    }
}

fn manifest_rows(manifest: &serde_json::Value) -> Result<Vec<&serde_json::Value>, String> {
    if manifest["abi"] != 1
        || manifest["computeCapability"] != serde_json::json!([12, 0])
        || manifest["tokens"] != TOKENS
        || manifest["qHeads"] != Q_HEADS
        || manifest["kvHeads"] != KV_HEADS
        || manifest["headWidth"] != WIDTH
        || manifest["helperAbi"] != 2
        || manifest["helperSourceTableStrides"] != serde_json::json!([128, 256])
    {
        return Err(
            "normrope101 requires ABI1 SM120 T256/Q16/KV8/D256 and stride-aware helper ABI2".into(),
        );
    }
    let rows = manifest["variants"]
        .as_array()
        .ok_or("normrope101 variants missing")?;
    if rows.len() != 2 {
        return Err("normrope101 requires exactly reduction and pointwise images".into());
    }
    let mut selected = Vec::with_capacity(2);
    for (role, symbol, digest, pointers, integers, grid, block, shared, xblock, rblock) in [
        (
            "reduction",
            REDUCTION,
            REDUCTION_SHA,
            4,
            3,
            128,
            512,
            256,
            64,
            Some(64),
        ),
        (
            "pointwise",
            POINTWISE,
            POINTWISE_SHA,
            11,
            2,
            1536,
            256,
            0,
            512,
            None,
        ),
    ] {
        let matches = rows
            .iter()
            .filter(|row| row["role"] == role)
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err("normrope101 duplicate or missing image role".into());
        }
        let row = matches[0];
        let file = row["file"]
            .as_str()
            .ok_or("normrope101 image filename missing")?;
        if Path::new(file).file_name().and_then(|x| x.to_str()) != Some(file)
            || row["symbol"] != symbol
            || row["sha256"] != digest
            || row["grid"] != serde_json::json!([grid, 1, 1])
            || row["block"] != serde_json::json!([block, 1, 1])
            || row["numWarps"] != block / 32
            || row["dynamicSharedBytes"] != shared
            || row["xBlock"] != xblock
            || row["rBlock"] != serde_json::json!(rblock)
            || row["globalScratchSize"] != 0
            || row["profileScratchSize"] != 0
        {
            return Err("normrope101 requires the screened fixed images and launch ABI".into());
        }
        let parameters = row["parameters"]
            .as_array()
            .ok_or("normrope101 parameters missing")?;
        let types = parameters
            .iter()
            .map(|parameter| parameter["type"].as_str())
            .collect::<Vec<_>>();
        let expected = [
            vec![Some("u64"); pointers],
            vec![Some("u32"); integers],
            vec![Some("u64"); 2],
        ]
        .concat();
        if types != expected {
            return Err("normrope101 parameter ABI mismatch".into());
        }
        selected.push(row);
    }
    Ok(selected)
}
fn module(
    context: &Arc<CudaContext>,
    path: &Path,
    expected: &str,
) -> Result<Arc<cudarc::driver::CudaModule>, String> {
    let source = std::fs::read(path).map_err(|e| e.to_string())?;
    if format!("{:x}", Sha256::digest(&source)) != expected {
        return Err("normrope101 PTX digest mismatch".into());
    }
    context
        .load_module(Ptx::from_src(
            String::from_utf8(source).map_err(|e| e.to_string())?,
        ))
        .map_err(|e| e.to_string())
}
impl NormRope101 {
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
            return Err("normrope101 requires CUDA SM120".into());
        }
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.join("launch-manifest.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let rows = manifest_rows(&manifest)?;
        let reduction_module = module(
            context,
            &directory.join(rows[0]["file"].as_str().unwrap()),
            REDUCTION_SHA,
        )?;
        let pointwise_module = module(
            context,
            &directory.join(rows[1]["file"].as_str().unwrap()),
            POINTWISE_SHA,
        )?;
        let helper_module = module(context, &directory.join("helpers.ptx"), HELPERS_SHA)?;
        let reduction = reduction_module
            .load_function(REDUCTION)
            .map_err(|e| e.to_string())?;
        reduction.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 256).map_err(|e| e.to_string())?;
        Ok(Arc::new(Self {
            context: context.clone(),
            prepare: helper_module
                .load_function("prepare101")
                .map_err(|e| e.to_string())?,
            reduction,
            pointwise: pointwise_module
                .load_function(POINTWISE)
                .map_err(|e| e.to_string())?,
            transpose: helper_module
                .load_function("transpose101")
                .map_err(|e| e.to_string())?,
        }))
    }
    /// The caller retains this plan and every owner through completion. All
    /// pointers, including all scratch and final outputs, are validated BEFORE
    /// enqueue, so an inadmissible later stage cannot partially mutate storage.
    pub(crate) unsafe fn launch(
        &self,
        stream: &Arc<CudaStream>,
        args: &Invocation101,
    ) -> Result<(), String> {
        if !Arc::ptr_eq(&self.context, stream.context()) {
            return Err("normrope101 launch stream belongs to another CUDA context".into());
        }
        args.validate()?;
        let mut prepare = crate::executable::CudaKernelArgs::default();
        prepare.inputs[0] = args.cosine;
        prepare.inputs[1] = args.sine;
        prepare.output = args.table;
        prepare.scratch[0] = args.positions;
        prepare.integers[0] = u64::from(args.cosine_stride);
        prepare.integers[1] = u64::from(args.sine_stride);
        unsafe {
            self.helper(stream, &self.prepare, &prepare, 128)?;
        }
        let pointers = [args.packed, args.sum_q, args.sum_k, args.raw_value];
        let extents = [4096_u32, 2048, 2048];
        unsafe {
            self.triton(stream, &self.reduction, &pointers, &extents, 128, 512, 256)?;
        }
        let pointers = [
            args.packed,
            args.sum_k,
            args.key_weight,
            args.positions,
            args.table,
            args.sum_q,
            args.query_weight,
            args.raw_key,
            args.raw_key + 256,
            args.raw_query,
            args.raw_query + 256,
        ];
        let extents = [262144_u32, 524288];
        unsafe {
            self.triton(stream, &self.pointwise, &pointers, &extents, 1536, 256, 0)?;
        }
        let mut transpose = crate::executable::CudaKernelArgs::default();
        transpose.inputs[..6].copy_from_slice(&[
            args.raw_query,
            args.raw_key,
            args.raw_value,
            args.query,
            args.key,
            args.value,
        ]);
        unsafe {
            self.helper(stream, &self.transpose, &transpose, 8192)?;
        }
        Ok(())
    }
    unsafe fn helper(
        &self,
        stream: &Arc<CudaStream>,
        function: &CudaFunction,
        args: &crate::executable::CudaKernelArgs,
        grid: u32,
    ) -> Result<(), String> {
        let mut launch = stream.launch_builder(function);
        launch.arg(args);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| e.to_string())?;
        Ok(())
    }
    unsafe fn triton(
        &self,
        stream: &Arc<CudaStream>,
        function: &CudaFunction,
        pointers: &[u64],
        extents: &[u32],
        grid: u32,
        block: u32,
        shared: u32,
    ) -> Result<(), String> {
        let zero = 0_u64;
        let mut launch = stream.launch_builder(function);
        for pointer in pointers {
            launch.arg(pointer);
        }
        for extent in extents {
            launch.arg(extent);
        }
        launch.arg(&zero).arg(&zero);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared,
            })
        }
        .map_err(|e| e.to_string())?;
        Ok(())
    }
}
#[cfg(test)]
#[path = "triton_normrope101_tests.rs"]
mod tests;

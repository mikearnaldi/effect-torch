//! Opt-in, fixed-image Triton normalization. All mutable storage is invocation-owned.
//!
//! The matched entrance/tail admission must prove the residual private, retain
//! its original attention/hidden/weight inputs across the FFN, and retain both
//! output owners through publication. This module neither allocates launch
//! scratch nor changes completion, cancellation or publication fences.
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Arc;

pub(crate) const DIRECTORY_ENV: &str = "EFFECT_TORCH_CUDA_TRITON_NORM98_DIRECTORY";
const WIDTH: u64 = 2816;
const SHARED: u32 = 32;
const ENTRANCE: &str = "triton_red_fused__to_copy_add_moe_forward_mul_rms_norm_0";
const TAIL: &str = "triton_red_fused_add_mul_rms_norm_2";
const ENTRANCE_SHA: &str = "df8877a070740b4caf0a7e06012183ead8d78404bd7211f6d13ac2331a3da23b";
const TAIL_SHA: &str = "7c93f9dda6a2f4a64b01b62f62adcd12781df5ee052c88e5e78acfa47b0e0c97";

pub(crate) struct Norm98 {
    context: Arc<CudaContext>,
    entrance: CudaFunction,
    tail: CudaFunction,
}

/// An immutable, completed BF16 conversion of a proven F32 scalar constant.
/// Compiled plans retain this owner; it is never converted per layer/invocation.
pub(crate) struct Rho98 {
    owner: Arc<CudaSlice<u16>>,
    address: u64,
}

fn narrow(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return ((bits >> 16 & 0x8000) | 0x7fc0) as u16;
    }
    (bits.wrapping_add(0x7fff + (bits >> 16 & 1)) >> 16) as u16
}

impl Rho98 {
    pub(crate) fn new(stream: &Arc<CudaStream>, value: f32) -> Result<Arc<Self>, String> {
        let bits = narrow(value);
        if !value.is_finite() || bits & 0x7f80 == 0x7f80 {
            return Err("norm98 rho must be a finite BF16-representable scalar constant".into());
        }
        let owner = Arc::new(stream.clone_htod(&[bits]).map_err(|e| e.to_string())?);
        // Setup is outside kernel dispatch. The owner is ready on every stream
        // in this context before the immutable plan is made callable.
        if let Err(error) = stream.synchronize() {
            std::mem::forget(owner);
            return Err(error.to_string());
        }
        let (address, ready) = owner.device_ptr(stream);
        drop(ready);
        Ok(Arc::new(Self { owner, address }))
    }

    pub(crate) fn address(&self, context: &Arc<CudaContext>) -> Result<u64, String> {
        if !Arc::ptr_eq(self.owner.stream().context(), context) {
            return Err("norm98 rho belongs to another CUDA context".into());
        }
        Ok(self.address)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Entrance98 {
    pub(crate) rows: u32,
    /// Invocation-private attention copy, also retained for the matched tail.
    pub(crate) attention: u64,
    pub(crate) attention_weight: u64,
    pub(crate) hidden: u64,
    pub(crate) rho_bf16: u64,
    pub(crate) router_weight: u64,
    pub(crate) expert_weight: u64,
    pub(crate) dense_weight: u64,
    /// F32[rows], retained unchanged from entrance until tail completion.
    pub(crate) sum_a: u64,
    pub(crate) router: u64,
    pub(crate) expert: u64,
    pub(crate) dense: u64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Tail98 {
    pub(crate) rows: u32,
    /// Exclusive BF16[rows,2816] attention copy, overwritten with nextHidden.
    pub(crate) private_attention: u64,
    pub(crate) dense: u64,
    pub(crate) expert: u64,
    pub(crate) dense_weight: u64,
    pub(crate) expert_weight: u64,
    pub(crate) combined_weight: u64,
    pub(crate) sum_a: u64,
    pub(crate) attention_weight: u64,
    pub(crate) hidden: u64,
    pub(crate) scale: u64,
    pub(crate) next_weight: u64,
    /// Exclusive F32[rows,2816] scratch, retained through the completion fence.
    pub(crate) combined_f32: u64,
    /// Separately owned BF16[rows,2816] output; cannot alias nextHidden.
    pub(crate) next_norm: u64,
}

#[derive(Clone, Copy)]
struct Range {
    begin: u64,
    end: u64,
}

fn range(pointer: u64, bytes: u64) -> Result<Range, String> {
    if pointer == 0 || pointer % 16 != 0 || bytes == 0 {
        return Err("norm98 requires nonzero 16-byte-aligned pointers".into());
    }
    Ok(Range {
        begin: pointer,
        end: pointer
            .checked_add(bytes)
            .ok_or("norm98 pointer range overflow")?,
    })
}

fn matrix_bytes(rows: u32) -> Result<u64, String> {
    if !matches!(rows, 64 | 256) {
        return Err("norm98 requires rows64/256 and width2816".into());
    }
    Ok(u64::from(rows) * WIDTH * 2)
}

fn separate(read: &[Range], write: &[Range]) -> Result<(), String> {
    let overlaps = |a: Range, b: Range| a.begin < b.end && b.begin < a.end;
    for (index, destination) in write.iter().enumerate() {
        if read.iter().any(|source| overlaps(*source, *destination))
            || write[..index]
                .iter()
                .any(|other| overlaps(*other, *destination))
        {
            return Err(
                "norm98 mutable storage overlaps a borrowed input or another output".into(),
            );
        }
    }
    Ok(())
}

impl Entrance98 {
    fn pointers(&self) -> [u64; 11] {
        [
            self.attention,
            self.attention_weight,
            self.hidden,
            self.rho_bf16,
            self.router_weight,
            self.expert_weight,
            self.dense_weight,
            self.sum_a,
            self.router,
            self.expert,
            self.dense,
        ]
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        let bytes = matrix_bytes(self.rows)?;
        let read = [
            range(self.attention, bytes)?,
            range(self.attention_weight, WIDTH * 2)?,
            range(self.hidden, bytes)?,
            range(self.rho_bf16, 2)?,
            range(self.router_weight, WIDTH * 2)?,
            range(self.expert_weight, WIDTH * 2)?,
            range(self.dense_weight, WIDTH * 2)?,
        ];
        let write = [
            range(self.sum_a, u64::from(self.rows) * 4)?,
            range(self.router, bytes)?,
            range(self.expert, bytes)?,
            range(self.dense, bytes)?,
        ];
        separate(&read, &write)
    }
}

impl Tail98 {
    fn pointers(&self) -> [u64; 13] {
        [
            self.private_attention,
            self.dense,
            self.expert,
            self.dense_weight,
            self.expert_weight,
            self.combined_weight,
            self.sum_a,
            self.attention_weight,
            self.hidden,
            self.scale,
            self.next_weight,
            self.combined_f32,
            self.next_norm,
        ]
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        let bytes = matrix_bytes(self.rows)?;
        let read = [
            range(self.dense, bytes)?,
            range(self.expert, bytes)?,
            range(self.dense_weight, WIDTH * 2)?,
            range(self.expert_weight, WIDTH * 2)?,
            range(self.combined_weight, WIDTH * 2)?,
            range(self.sum_a, u64::from(self.rows) * 4)?,
            range(self.attention_weight, WIDTH * 2)?,
            range(self.hidden, bytes)?,
            range(self.scale, 2)?,
            range(self.next_weight, WIDTH * 2)?,
        ];
        let write = [
            range(self.private_attention, bytes)?,
            range(self.combined_f32, bytes * 2)?,
            range(self.next_norm, bytes)?,
        ];
        separate(&read, &write)
    }
}

fn manifest_rows(manifest: &serde_json::Value) -> Result<Vec<&serde_json::Value>, String> {
    if manifest["abi"] != 1 || manifest["computeCapability"] != serde_json::json!([12, 0]) {
        return Err("norm98 requires ABI1 and compute capability12.0".into());
    }
    let rows = manifest["variants"]
        .as_array()
        .ok_or("norm98 variants missing")?;
    if rows.len() != 2 {
        return Err("norm98 requires exactly entrance and BF16 tail images".into());
    }
    let mut selected = Vec::with_capacity(2);
    for (role, entry, digest, pointers) in [
        ("entrance", ENTRANCE, ENTRANCE_SHA, 11),
        ("tail", TAIL, TAIL_SHA, 13),
    ] {
        let matches = rows
            .iter()
            .filter(|row| row["role"] == role)
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err("norm98 duplicate or missing image role".into());
        }
        let row = matches[0];
        let file = row["file"]
            .as_str()
            .ok_or("norm98 image filename missing")?;
        if Path::new(file).file_name().and_then(|x| x.to_str()) != Some(file)
            || row["entry"] != entry
            || row["sha256"] != digest
            || row["numWarps"] != 8
            || row["dynamicSharedBytes"] != SHARED
            || row["globalScratchSize"] != 0
            || row["profileScratchSize"] != 0
        {
            return Err("norm98 requires the screened fixed image and launch ABI".into());
        }
        let parameters = row["parameters"]
            .as_array()
            .ok_or("norm98 parameters missing")?;
        let types = parameters
            .iter()
            .map(|p| p["type"].as_str())
            .collect::<Vec<_>>();
        let expected = [
            vec![Some("u64"); pointers],
            vec![Some("u32"); 2],
            vec![Some("u64"); 2],
        ]
        .concat();
        if types != expected {
            return Err("norm98 parameter ABI mismatch".into());
        }
        selected.push(row);
    }
    Ok(selected)
}

impl Norm98 {
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
            return Err("norm98 requires CUDA SM120".into());
        }
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.join("launch-manifest.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let rows = manifest_rows(&manifest)?;
        let mut functions = Vec::with_capacity(2);
        for row in rows {
            let source = std::fs::read(directory.join(row["file"].as_str().unwrap()))
                .map_err(|e| e.to_string())?;
            if format!("{:x}", Sha256::digest(&source)) != row["sha256"].as_str().unwrap() {
                return Err("norm98 PTX digest mismatch".into());
            }
            let source = String::from_utf8(source).map_err(|e| e.to_string())?;
            let module = context
                .load_module(Ptx::from_src(source))
                .map_err(|e| e.to_string())?;
            let function = module
                .load_function(row["entry"].as_str().unwrap())
                .map_err(|e| e.to_string())?;
            function.set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                SHARED as i32,
            ).map_err(|e| e.to_string())?;
            functions.push(function);
        }
        let tail = functions.pop().ok_or("norm98 tail missing")?;
        let entrance = functions.pop().ok_or("norm98 entrance missing")?;
        Ok(Arc::new(Self {
            context: context.clone(),
            entrance,
            tail,
        }))
    }

    fn stream_valid(&self, stream: &Arc<CudaStream>) -> Result<(), String> {
        if !Arc::ptr_eq(&self.context, stream.context()) {
            return Err("norm98 launch stream belongs to another CUDA context".into());
        }
        Ok(())
    }

    /// Caller retains every allocation and this module through the existing
    /// completion fence. Attention is a private copy; hidden/weights are borrowed.
    pub(crate) unsafe fn launch_entrance(
        &self,
        stream: &Arc<CudaStream>,
        args: &Entrance98,
    ) -> Result<(), String> {
        self.stream_valid(stream)?;
        args.validate()?;
        unsafe { self.launch(stream, &self.entrance, args.rows, &args.pointers()) }
    }

    /// Caller proves the paired residual private and retains original inputs,
    /// row stats, F32 scratch, and both independent output owners to completion.
    pub(crate) unsafe fn launch_tail(
        &self,
        stream: &Arc<CudaStream>,
        args: &Tail98,
    ) -> Result<(), String> {
        self.stream_valid(stream)?;
        args.validate()?;
        unsafe { self.launch(stream, &self.tail, args.rows, &args.pointers()) }
    }

    unsafe fn launch(
        &self,
        stream: &Arc<CudaStream>,
        function: &CudaFunction,
        rows: u32,
        pointers: &[u64],
    ) -> Result<(), String> {
        let width = WIDTH as u32;
        let zero = 0_u64;
        let mut launch = stream.launch_builder(function);
        for pointer in pointers {
            launch.arg(pointer);
        }
        launch.arg(&rows).arg(&width).arg(&zero).arg(&zero);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (rows, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: SHARED,
            })
        }
        .map_err(|e| e.to_string())?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "triton_norm98_tests.rs"]
mod tests;

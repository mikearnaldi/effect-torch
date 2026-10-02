//! Opt-in, ahead-compiled exact expert tensor path. Mutable descriptors and
//! shapes use existing invocation scratch; only immutable functions are cached.
use super::{expert_splitk_geometry, Bf16GemmPlan, ExpertSplitKDescriptor, ExpertSplitKUpload};
use crate::cublas::{
    CUBLAS_WORKSPACE_BYTES, EXPERT_BLAS_STREAMS, EXPERT_GROUPED_POINTER_BYTES,
    EXPERT_SPLITK_DESCRIPTOR_BYTES,
};
use cudarc::driver::{
    sys::CUdevice_attribute, CudaContext, CudaFunction, CudaStream, DeviceRepr, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

pub(super) const PATH_ENV: &str = "EFFECT_TORCH_CUDA_EXPERT_MERGED_PTX";
const SHARED_BYTES: u32 = 36880;
#[repr(C)]
struct WideUpload {
    descriptors: [ExpertSplitKDescriptor; 128],
}
// SAFETY: initialized repr(C) scalar fields match the CUDA v2 upload ABI.
unsafe impl DeviceRepr for WideUpload {}

pub(super) struct MergedExpert {
    upload: CudaFunction,
    wide_upload: Option<CudaFunction>,
    compute: CudaFunction,
    pub(super) graph61_compute: Option<Arc<crate::explicit_graph61::RawKernel61>>,
    blocks: u32,
    pub(super) device59_artifact: bool,
}

pub(super) fn supports(group: &(Bf16GemmPlan, u64, u64, u64)) -> bool {
    expert_splitk_geometry(group.0).is_some()
        && [group.1, group.2, group.3]
            .into_iter()
            .all(|pointer| pointer != 0 && pointer % 16 == 0)
}

fn addresses(workspace: u64, bytes: usize) -> Result<(u64, u64), String> {
    let shapes = EXPERT_BLAS_STREAMS * CUBLAS_WORKSPACE_BYTES;
    let descriptors = shapes + EXPERT_GROUPED_POINTER_BYTES;
    let required = descriptors + EXPERT_SPLITK_DESCRIPTOR_BYTES;
    if bytes < required || workspace == 0 || workspace % 16 != 0 {
        return Err("CUDA merged expert workspace bounds/alignment exceeded".into());
    }
    workspace
        .checked_add(required as u64)
        .ok_or("CUDA merged expert address overflow")?;
    Ok((workspace + shapes as u64, workspace + descriptors as u64))
}

impl MergedExpert {
    pub(super) fn load(context: &Arc<CudaContext>, path: &str) -> Result<Self, String> {
        // Read eagerly to report missing artifacts as ordinary errors, and keep
        // artifact contents stable for this device's entire lifetime.
        let source = std::fs::read_to_string(path)
            .map_err(|error| format!("CUDA merged expert PTX {path}: {error}"))?;
        use sha2::{Digest, Sha256};
        let device59_artifact = format!("{:x}", Sha256::digest(source.as_bytes()))
            == "56e1f7b1934ac08afc48625600d3e25f3ffe1f5d64e4fcee04a87359b63f6b4e";
        let graph61_compute = if crate::executable::expert_pair61::enabled() {
            Some(
                crate::explicit_graph61::RawModule61::load(context.clone(), source.as_bytes())?
                    .function("et_expert_merged_compute_v1")?,
            )
        } else {
            None
        };
        let module = context
            .load_module(Ptx::from_src(source))
            .map_err(|error| error.to_string())?;
        let upload = super::load(&module, "et_expert_merged_upload_v1")?;
        // A v1 artifact remains valid even when the optional v2 flag is set.
        let wide_upload =
            if std::env::var("EFFECT_TORCH_CUDA_EXPERT_UPLOAD128").as_deref() == Ok("1") {
                super::load(&module, "et_expert_merged_upload_v2").ok()
            } else {
                None
            };
        let compute = super::load(&module, "et_expert_merged_compute_v1")?;
        let sms = context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|error| error.to_string())?;
        let blocks = u32::try_from(sms)
            .ok()
            .filter(|&n| n > 0)
            .and_then(|n| n.checked_mul(2))
            .ok_or("CUDA merged expert invalid multiprocessor count")?;
        Ok(Self {
            upload,
            wide_upload,
            compute,
            graph61_compute,
            blocks,
            device59_artifact,
        })
    }

    pub(super) fn graph61_blocks(&self) -> u32 {
        self.blocks
    }

    pub(super) unsafe fn launch_device59(
        &self,
        descriptors: u64,
        shapes: u64,
        columns: u32,
        inner: u32,
        stream: &Arc<CudaStream>,
    ) -> Result<(), String> {
        let count = 128u32;
        let mut launch = stream.launch_builder(&self.compute);
        launch
            .arg(&descriptors)
            .arg(&shapes)
            .arg(&count)
            .arg(&columns)
            .arg(&inner);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (self.blocks, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: SHARED_BYTES,
            })
        }
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Caller retains all inputs/output/workspace through the common expert
    /// submission fence; all metadata writes and reads use this worker stream.
    pub(super) unsafe fn launch(
        &self,
        groups: &[(Bf16GemmPlan, u64, u64, u64)],
        workspace: u64,
        bytes: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<(), String> {
        let (shape_address, descriptor_address) = addresses(workspace, bytes)?;
        let mut descriptors = Vec::new();
        for &(plan, x, weight, out) in groups.iter().filter(|group| supports(group)) {
            let (splits, slice_k) =
                expert_splitk_geometry(plan).ok_or("CUDA merged expert unsupported plan")?;
            descriptors.push(ExpertSplitKDescriptor {
                x,
                weight,
                out,
                rows: plan.m as u32,
                splits,
                slice_k,
                ..ExpertSplitKDescriptor::default()
            });
        }
        if descriptors.is_empty() || descriptors.len() > 128 {
            return Err("CUDA merged expert descriptor count exceeded".into());
        }
        let columns = groups[0].0.n as u32;
        let inner = groups[0].0.k as u32;
        if groups
            .iter()
            .any(|(plan, ..)| (plan.n, plan.k) != (columns as usize, inner as usize))
        {
            return Err("CUDA merged expert heterogeneous matrix dimensions".into());
        }
        let count = descriptors.len() as u32;
        if let Some(upload) = &self.wide_upload {
            let mut batch = WideUpload {
                descriptors: [ExpertSplitKDescriptor::default(); 128],
            };
            batch.descriptors[..descriptors.len()].copy_from_slice(&descriptors);
            let first = 0u32;
            let mut launch = stream.launch_builder(upload);
            launch
                .arg(&descriptor_address)
                .arg(&shape_address)
                .arg(&first)
                .arg(&count)
                .arg(&columns)
                .arg(&inner)
                .arg(&batch);
            // SAFETY: CUDA 12.1+/SM70+ support this 8224-byte parameter block.
            // The v2 module targets SM120; all descriptors are copied by value
            // and output addresses refer to bounded invocation scratch.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|error| error.to_string())?;
        } else {
            for (index, chunk) in descriptors.chunks(32).enumerate() {
                let mut batch = ExpertSplitKUpload {
                    descriptors: [ExpertSplitKDescriptor::default(); 32],
                };
                batch.descriptors[..chunk.len()].copy_from_slice(chunk);
                let first = (index * 32) as u32;
                let batch_count = chunk.len() as u32;
                let mut launch = stream.launch_builder(&self.upload);
                launch
                    .arg(&descriptor_address)
                    .arg(&shape_address)
                    .arg(&first)
                    .arg(&batch_count)
                    .arg(&columns)
                    .arg(&inner)
                    .arg(&batch);
                // SAFETY: bounded invocation scratch, copied by-value descriptor bank.
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|error| error.to_string())?;
            }
        }
        let mut launch = stream.launch_builder(&self.compute);
        launch
            .arg(&descriptor_address)
            .arg(&shape_address)
            .arg(&count)
            .arg(&columns)
            .arg(&inner);
        // SAFETY: disjoint outputs and planner-owned metadata remain alive until
        // both success and failure paths drain the participating worker stream.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (self.blocks, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: SHARED_BYTES,
            })
        }
        .map_err(|error| error.to_string())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires pinned SM120/cuBLAS and EFFECT_TORCH_CUDA_EXPERT_MERGED_PTX"]
    fn merged_descriptor_chunks_dynamic_routing_and_error_ownership() {
        use crate::{
            device::{CudaDevice, FAIL_EXPERT_SUBMISSION_BEFORE_JOIN},
            CudaValue,
        };
        use cudarc::driver::DevicePtr;
        use effect_torch_runtime::DType;
        let device = CudaDevice::get(0).unwrap();
        let merged = device.merged_expert.as_ref().unwrap();
        let artifact = std::fs::read_to_string(std::env::var(PATH_ENV).unwrap()).unwrap();
        assert_eq!(
            merged.wide_upload.is_some(),
            std::env::var("EFFECT_TORCH_CUDA_EXPERT_UPLOAD128").as_deref() == Ok("1")
                && artifact.contains("et_expert_merged_upload_v2")
        );
        let values = |count: usize, seed: usize| {
            (0..count)
                .map(|i| {
                    let bits = i.wrapping_mul(1664525).wrapping_add(seed);
                    (((bits >> 8) % 255) as f64 - 127.0)
                        * 2.0f64.powi(((bits >> 19) % 15) as i32 - 12)
                })
                .collect::<Vec<_>>()
        };
        for (n, k) in [(1408, 2816), (2816, 704)] {
            let weight =
                CudaValue::from_host(device.clone(), vec![n, k], DType::BF16, &values(n * k, 313))
                    .unwrap();
            let mut retained = Vec::new();
            for phase in 0..4 {
                // Cross the 32-descriptor upload boundary, then vary routing,
                // descriptor count and both supported/ordinary worker paths.
                let mut rows = vec![2; 33];
                rows.extend([1, 17, 31, 33, 43, 58, 65, 129, 193, 257, 417, 449, 513]);
                rows.rotate_left(phase * 3);
                if phase == 1 {
                    rows.truncate(19);
                } else if phase == 3 {
                    // Fill every descriptor in the wide upload bank.
                    rows = vec![2; 128];
                }
                let total: usize = rows.iter().sum();
                let x = CudaValue::from_host(
                    device.clone(),
                    vec![total, k],
                    DType::BF16,
                    &values(total * k, 17 + phase),
                )
                .unwrap();
                let output = || {
                    CudaValue::from_host(
                        device.clone(),
                        vec![total, n],
                        DType::BF16,
                        &vec![0.0; total * n],
                    )
                    .unwrap()
                };
                let actual = output();
                let expected = output();
                let groups = |output: &CudaValue| {
                    let mut offset = 0;
                    rows.iter()
                        .map(|&m| {
                            let plan = Bf16GemmPlan {
                                m,
                                n,
                                k,
                                batch: 1,
                                stride_x: m * k,
                                stride_weight: 0,
                                stride_out: m * n,
                            };
                            let group = (
                                plan,
                                x.storage_address() + (offset * k * 2) as u64,
                                weight.storage_address(),
                                output.storage_address() + (offset * n * 2) as u64,
                            );
                            offset += m;
                            group
                        })
                        .collect::<Vec<_>>()
                };
                let bytes = EXPERT_BLAS_STREAMS * CUBLAS_WORKSPACE_BYTES
                    + EXPERT_GROUPED_POINTER_BYTES
                    + EXPERT_SPLITK_DESCRIPTOR_BYTES;
                let workspace = unsafe { device.stream.alloc::<u8>(bytes) }.unwrap();
                let (address, guard) = workspace.device_ptr(&device.stream);
                let reference =
                    unsafe { device.grouped_gemm_bf16(&groups(&expected), address, false, bytes) }
                        .unwrap();
                let reference_values = expected.readback().unwrap();
                if phase == 2 {
                    FAIL_EXPERT_SUBMISSION_BEFORE_JOIN.with(|flag| flag.set(true));
                }
                let result =
                    unsafe { device.grouped_gemm_bf16(&groups(&actual), address, true, bytes) };
                if phase == 2 {
                    assert_eq!(
                        result.unwrap_err(),
                        "injected expert submission failure before join"
                    );
                } else {
                    drop(result.unwrap());
                }
                let actual_values = actual.readback().unwrap();
                assert!(
                    actual_values
                        .iter()
                        .zip(&reference_values)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "projection {n} phase {phase}"
                );
                drop((reference, guard));
                drop(workspace);
                retained.push((actual, reference_values));
                for (value, reference) in &retained {
                    assert!(value
                        .readback()
                        .unwrap()
                        .iter()
                        .zip(reference)
                        .all(|(a, b)| a.to_bits() == b.to_bits()));
                }
            }
        }
    }

    #[test]
    fn workspace_and_descriptor_abi_are_bounded() {
        let bytes = EXPERT_BLAS_STREAMS * CUBLAS_WORKSPACE_BYTES
            + EXPERT_GROUPED_POINTER_BYTES
            + EXPERT_SPLITK_DESCRIPTOR_BYTES;
        assert!(addresses(256, bytes).is_ok());
        assert!(addresses(256, bytes - 1).is_err());
        assert!(addresses(257, bytes).is_err());
        assert!(addresses(u64::MAX - 15, bytes).is_err());
        assert_eq!(std::mem::size_of::<ExpertSplitKDescriptor>(), 64);
        assert_eq!(std::mem::size_of::<ExpertSplitKUpload>(), 2048);
        assert_eq!(std::mem::size_of::<WideUpload>(), 8192);
        assert!(128 * 12 <= EXPERT_GROUPED_POINTER_BYTES);
    }
    #[test]
    fn unsupported_shapes_and_misaligned_pointers_fall_back() {
        let mut plan = Bf16GemmPlan {
            m: 17,
            n: 1408,
            k: 2816,
            batch: 1,
            stride_x: 17 * 2816,
            stride_weight: 0,
            stride_out: 17 * 1408,
        };
        assert!(supports(&(plan, 16, 32, 48)));
        assert!(!supports(&(plan, 17, 32, 48)));
        plan.m = 1;
        assert!(!supports(&(plan, 16, 32, 48)));
        plan.m = 513;
        assert!(!supports(&(plan, 16, 32, 48)));
        plan.m = 17;
        plan.batch = 2;
        assert!(!supports(&(plan, 16, 32, 48)));
    }
}

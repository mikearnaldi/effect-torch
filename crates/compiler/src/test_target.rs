//! Deterministic capability fixtures for compiler structural tests only.
use crate::*;
use effect_torch_graph::Device;
use effect_torch_runtime::DType;
use std::cell::Cell;

pub(crate) struct TestTarget {
    pub device: Device,
    pub fingerprint: TargetFingerprint,
    pub revision: u64,
    pub reject_regions: bool,
    pub reject_multi_output: bool,
    pub reject_nodes: bool,
    pub promote_half: bool,
    pub f64_algorithms: bool,
    pub region_queries: Cell<usize>,
    pub node_queries: Cell<usize>,
}

impl TestTarget {
    pub fn for_index(index: &GraphIndex) -> Self {
        Self::new(
            index
                .order
                .first()
                .map_or(Device::Cpu(0), |node| node.device.clone()),
        )
    }
    pub fn new(device: Device) -> Self {
        let backend = match device {
            Device::Cpu(_) => TargetBackend::Cpu,
            Device::Metal(_) => TargetBackend::Metal,
            Device::Cuda(_) => TargetBackend::Cuda,
        };
        Self {
            device,
            fingerprint: TargetFingerprint::new(backend, "compiler-test", 1),
            revision: 1,
            reject_regions: false,
            reject_multi_output: false,
            reject_nodes: false,
            promote_half: false,
            f64_algorithms: false,
            region_queries: Cell::new(0),
            node_queries: Cell::new(0),
        }
    }
}

impl TargetDTypeCapabilities for TestTarget {
    fn device(&self) -> &Device {
        &self.device
    }
    fn fingerprint(&self) -> &TargetFingerprint {
        &self.fingerprint
    }
    fn policy_revision(&self) -> u64 {
        self.revision
    }
    fn storage_support(&self, value: &ValueSpec<'_>) -> StorageSupport {
        match value.validate() {
            Ok(()) => StorageSupport::Supported,
            Err(reason) => StorageSupport::Unsupported(UnsupportedDType::new(
                DTypeRequirement::Storage,
                reason,
            )),
        }
    }
    fn classify_node(&self, spec: &OperationDTypeSpec<'_>) -> DTypeDisposition {
        self.node_queries.set(self.node_queries.get() + 1);
        if self.reject_nodes {
            return DTypeDisposition::Unsupported(UnsupportedDType::new(
                DTypeRequirement::Compute,
                "test independent operation rejected",
            ));
        }
        if self.f64_algorithms && spec.required_numerics.f64_compute.is_some() {
            let execution = spec.f64_execution().unwrap();
            return if execution.realization == ExecutionRealization::DirectKernel {
                DTypeDisposition::Native(execution)
            } else {
                DTypeDisposition::Legalize(execution)
            };
        }
        let native = spec.native_execution();
        if self.promote_half && spec.required_numerics.permits_f32_compute {
            DTypeDisposition::Legalize(
                native.promote_half(ExecutionRealization::MaterializedTransforms),
            )
        } else {
            DTypeDisposition::Native(native)
        }
    }
    fn classify_region(&self, spec: &RegionDTypeSpec<'_>) -> DTypeDisposition {
        self.region_queries.set(self.region_queries.get() + 1);
        let dtype = spec.boundary_results[0].value.semantic_dtype;
        let accepted_dtype = match self.device {
            Device::Cpu(_) => matches!(dtype, DType::F32 | DType::F64) || self.promote_half,
            Device::Metal(_) => matches!(dtype, DType::F32 | DType::BF16) || self.promote_half,
            Device::Cuda(_) => false,
        };
        let supported_size = !self.device.is_metal()
            || spec.boundary_results.iter().all(|result| {
                result.value.logical_shape.iter().product::<usize>() <= i32::MAX as usize
            });
        if self.reject_regions
            || !accepted_dtype
            || !supported_size
            || (self.reject_multi_output && matches!(spec.region, NativeRegion::MultiOutput(_)))
        {
            return DTypeDisposition::Unsupported(UnsupportedDType::new(
                DTypeRequirement::Region,
                "test region rejected",
            ));
        }
        let native = spec.native_execution();
        if self.promote_half
            && spec
                .operations
                .iter()
                .any(|operation| operation.required_numerics.permits_f32_compute)
        {
            DTypeDisposition::Legalize(native.promote_half(ExecutionRealization::KernelLocal))
        } else {
            DTypeDisposition::Native(native)
        }
    }
}

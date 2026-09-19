//! Pure CUDA dtype policy for the typed reference-kernel ABI and native
//! row-major BF16 GEMM.
use crate::cublas::{plan_row_bf16_gemm, RowGemmKind, BF16_GEMM_MIN_MAJOR};
use effect_torch_compiler::{
    DTypeDisposition, DTypeRequirement, ExecutionRealization, LayoutConstraintSpec,
    OperationDTypeSpec, RegionDTypeSpec, StorageSupport, TargetBackend, TargetDTypeCapabilities,
    TargetFingerprint, UnsupportedDType, ValueRole, ValueSpec,
};
use effect_torch_graph::{Device, NodeKind};
use effect_torch_runtime::{DType, PackedFormat, StorageRepresentation};

#[derive(Clone, Debug)]
pub(crate) struct CudaCapabilities {
    device: Device,
    fingerprint: TargetFingerprint,
    bf16_gemm: bool,
}
impl CudaCapabilities {
    pub(crate) fn new(ordinal: u32, major: i32, minor: i32) -> Self {
        let mut fingerprint =
            TargetFingerprint::new(TargetBackend::Cuda, format!("sm_{major}{minor}"), 4);
        let bf16_gemm = major >= BF16_GEMM_MIN_MAJOR;
        let mut features = vec!["typed-reference-kernels-v1".into()];
        if bf16_gemm {
            features.push("cublas-bf16-row-major-f32-accum-v2".into());
            features.push("cublas-no-reduced-precision-reduction".into());
            features.push("cublas-status-free-bias-epilogue-v1".into());
            features.push(format!(
                "cublas-workspace-bytes-{}",
                crate::cublas::CUBLAS_WORKSPACE_BYTES
            ));
        }
        features.sort();
        features.dedup();
        fingerprint.features = features.into_boxed_slice();
        Self {
            device: Device::Cuda(ordinal),
            fingerprint,
            bf16_gemm,
        }
    }

    pub(crate) fn for_device(device: &crate::CudaDevice) -> Result<Self, String> {
        let (major, minor) = device
            .stream
            .context()
            .compute_capability()
            .map_err(|e| e.to_string())?;
        let mut driver = 0;
        unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut driver) }
            .result()
            .map_err(|e| e.to_string())?;
        let (mut nvrtc_major, mut nvrtc_minor) = (0, 0);
        unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut nvrtc_major, &mut nvrtc_minor) }
            .result()
            .map_err(|e| e.to_string())?;
        let mut capabilities = Self::new(device.ordinal, major, minor);
        capabilities.fingerprint.libraries = vec![
            format!("cublas-{}", device.cublas.version),
            format!("cuda-driver-{driver}"),
            format!("nvrtc-{nvrtc_major}.{nvrtc_minor}"),
        ]
        .into_boxed_slice();
        Ok(capabilities)
    }

    /// Native row-major BF16 GEMM recipe when hardware and shapes support it.
    fn native_bf16_gemm(
        &self,
        spec: &OperationDTypeSpec<'_>,
    ) -> Option<effect_torch_compiler::DTypeExecution> {
        if !self.bf16_gemm {
            return None;
        }
        let kind = match spec.operation {
            NodeKind::Linear { .. } => RowGemmKind::Linear,
            NodeKind::Matmul { .. } => RowGemmKind::Matmul,
            _ => return None,
        };
        if spec.required_numerics.compute_dtype != Some(DType::BF16) {
            return None;
        }
        if spec
            .operands
            .iter()
            .chain(spec.results.iter())
            .any(|operand| {
                operand.value.semantic_dtype != DType::BF16
                    || operand.value.storage.representation != StorageRepresentation::Dense
            })
        {
            return None;
        }
        let activation = spec.operands.first()?;
        let weight = spec.operands.get(1)?;
        let result = spec.results.first()?;
        plan_row_bf16_gemm(
            kind,
            activation.value.logical_shape,
            weight.value.logical_shape,
            result.value.logical_shape,
        )?;
        Some(spec.native_execution())
    }
}
fn unsupported(requirement: DTypeRequirement, reason: impl Into<String>) -> DTypeDisposition {
    DTypeDisposition::Unsupported(UnsupportedDType::new(requirement, reason))
}
impl TargetDTypeCapabilities for CudaCapabilities {
    fn device(&self) -> &Device {
        &self.device
    }
    fn fingerprint(&self) -> &TargetFingerprint {
        &self.fingerprint
    }
    fn policy_revision(&self) -> u64 {
        // 5: status-free BF16 linear epilogues, retaining one final rounding.
        5
    }
    fn storage_support(&self, value: &ValueSpec<'_>) -> StorageSupport {
        let invalid = |reason| {
            StorageSupport::Unsupported(UnsupportedDType::new(DTypeRequirement::Storage, reason))
        };
        if matches!(
            value.storage.layout_constraint,
            LayoutConstraintSpec::DenseStrided { .. }
        ) {
            return invalid("CUDA binding storage must be canonical contiguous");
        }
        let elements = match crate::value::element_count(value.logical_shape) {
            Ok(n) => n,
            Err(_) => return invalid("CUDA shape overflows usize"),
        };
        match value.storage.representation {
            StorageRepresentation::Dense => {
                if elements
                    .checked_mul(value.semantic_dtype.size_in_bytes())
                    .is_none()
                {
                    invalid("CUDA dense byte size overflows usize")
                } else {
                    StorageSupport::Supported
                }
            }
            StorageRepresentation::Packed(PackedFormat::GgmlKQuant(codec)) => {
                if value.semantic_dtype != DType::F32 || value.logical_shape.len() != 2 {
                    return invalid("CUDA packed weights require logical F32 matrices");
                }
                match codec
                    .encoded_row_bytes(value.logical_shape[1])
                    .and_then(|bytes| bytes.checked_mul(value.logical_shape[0]))
                {
                    Some(_) => StorageSupport::Supported,
                    None => invalid("invalid CUDA packed block geometry"),
                }
            }
        }
    }
    fn classify_node(&self, spec: &OperationDTypeSpec<'_>) -> DTypeDisposition {
        if spec.placement != &self.device {
            return unsupported(
                DTypeRequirement::Layout,
                "CUDA device placement differs from the target",
            );
        }
        if let NodeKind::Argmax { a, dim } | NodeKind::Argmin { a, dim } = spec.operation {
            if a.shape.get(*dim).is_none_or(|extent| *extent == 0) {
                return unsupported(
                    DTypeRequirement::Layout,
                    "CUDA argmax/argmin requires a nonempty reduction axis",
                );
            }
        }
        for value in spec.operands.iter().chain(spec.results.iter()) {
            if let StorageSupport::Unsupported(error) = self.storage_support(&value.value) {
                return DTypeDisposition::Unsupported(error);
            }
        }
        for operand in &spec.operands {
            if operand.value.storage.representation != StorageRepresentation::Dense
                && !(operand.role == ValueRole::Weight
                    && matches!(
                        spec.operation,
                        NodeKind::QuantizedLinear { .. } | NodeKind::QuantizedEmbedding { .. }
                    ))
            {
                return unsupported(
                    DTypeRequirement::Operand(operand.role),
                    "packed CUDA values require canonical packed linear or embedding access",
                );
            }
        }
        if !matches!(spec.operation, NodeKind::Leaf(_) | NodeKind::Input { .. })
            && spec
                .results
                .iter()
                .any(|r| r.value.storage.representation != StorageRepresentation::Dense)
        {
            return unsupported(
                DTypeRequirement::Representation,
                "CUDA operation results must be dense",
            );
        }
        for result in spec.operands.iter().chain(spec.results.iter()) {
            let count = match crate::value::element_count(result.value.logical_shape) {
                Ok(count) => count,
                Err(error) => return unsupported(DTypeRequirement::Storage, error),
            };
            if count.div_ceil(256) > i32::MAX as usize {
                return unsupported(
                    DTypeRequirement::Realization,
                    "CUDA typed kernel grid exceeds the device grid limit",
                );
            }
            if result.value.logical_shape.len() > 64
                || count > u32::MAX as usize
                || result
                    .value
                    .logical_shape
                    .iter()
                    .any(|d| *d > u32::MAX as usize)
            {
                return unsupported(DTypeRequirement::Layout,"CUDA reference kernels require rank <=64 and u32 element counts and dimensions");
            }
        }
        if spec
            .required_numerics
            .compute_dtype
            .is_some_and(|dtype| !dtype.is_float())
            && !matches!(
                spec.operation,
                NodeKind::Add { .. }
                    | NodeKind::Sub { .. }
                    | NodeKind::Mul { .. }
                    | NodeKind::Div { .. }
                    | NodeKind::Maximum { .. }
                    | NodeKind::Minimum { .. }
                    | NodeKind::Eq { .. }
                    | NodeKind::Gt { .. }
                    | NodeKind::Lt { .. }
                    | NodeKind::Ge { .. }
                    | NodeKind::Le { .. }
                    | NodeKind::Neg { .. }
                    | NodeKind::Abs { .. }
                    | NodeKind::Relu { .. }
                    | NodeKind::Sign { .. }
                    | NodeKind::Floor { .. }
                    | NodeKind::Ceil { .. }
                    | NodeKind::Round { .. }
                    | NodeKind::Sum { .. }
                    | NodeKind::Prod { .. }
                    | NodeKind::Max { .. }
                    | NodeKind::Min { .. }
                    | NodeKind::Mean { .. }
                    | NodeKind::Argmax { .. }
                    | NodeKind::Argmin { .. }
                    | NodeKind::Cumsum { .. }
                    | NodeKind::IndexSelect { .. }
                    | NodeKind::Gather { .. }
                    | NodeKind::ScatterAdd { .. }
                    | NodeKind::Where { .. }
                    | NodeKind::Arange { .. }
                    | NodeKind::Eye { .. }
                    | NodeKind::LastTokenRow { .. }
                    | NodeKind::PositionEmbedding { .. }
            )
        {
            return unsupported(
                DTypeRequirement::Compute,
                "CUDA operation has no exact integer kernel",
            );
        }
        if let Some(execution) = self.native_bf16_gemm(spec) {
            return DTypeDisposition::Native(execution);
        }
        let execution = spec.native_execution();
        if spec.required_numerics.permits_f32_compute {
            return DTypeDisposition::Legalize(
                execution.promote_half(ExecutionRealization::MaterializedTransforms),
            );
        }
        if matches!(
            spec.required_numerics.compute_dtype,
            Some(DType::F16 | DType::BF16)
        ) && !matches!(
            spec.operation,
            NodeKind::Argmax { .. }
                | NodeKind::Argmin { .. }
                | NodeKind::IndexSelect { .. }
                | NodeKind::Gather { .. }
                | NodeKind::LastTokenRow { .. }
                | NodeKind::PositionEmbedding { .. }
        ) {
            return unsupported(
                DTypeRequirement::Compute,
                "this half operation has no approved F32 execution contract",
            );
        }
        DTypeDisposition::Native(execution)
    }
    fn classify_region(&self, _spec: &RegionDTypeSpec<'_>) -> DTypeDisposition {
        unsupported(
            DTypeRequirement::Region,
            "CUDA region lowering is not hardware-validated",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use effect_torch_compiler::{GraphIndex, OperationDTypeSpec};
    use effect_torch_graph::Node;
    use std::sync::Arc;

    #[test]
    fn empty_arg_reductions_fail_before_lowering() {
        use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
        for shape in [vec![0], vec![0, 2]] {
            for minimum in [false, true] {
                let a = Node::new(NodeKind::Zeros {
                    shape: shape.clone(),
                    dtype: DType::F32,
                    device: Device::Cuda(0),
                })
                .unwrap();
                let root = Node::new(if minimum {
                    NodeKind::Argmin { a, dim: 0 }
                } else {
                    NodeKind::Argmax { a, dim: 0 }
                })
                .unwrap();
                let prepared = ProgramRequest::from_roots(vec![root], CompileOptions::default())
                    .prepare()
                    .unwrap();
                let capabilities = CudaCapabilities::new(0, 12, 0);
                let error = match CompilerDriver::new(&prepared, &capabilities) {
                    Ok(_) => panic!("empty arg reduction must fail before lowering"),
                    Err(error) => error,
                };
                assert!(error.contains("nonempty reduction axis"), "{error}");
            }
        }
    }

    fn bf16_linear(dtype: DType, transposed: bool) -> Arc<Node> {
        let x = Node::new(NodeKind::Zeros {
            shape: vec![2, 3],
            dtype,
            device: Device::Cuda(0),
        })
        .unwrap();
        let weight = Node::new(NodeKind::Zeros {
            shape: if transposed { vec![2, 3] } else { vec![3, 2] },
            dtype,
            device: Device::Cuda(0),
        })
        .unwrap();
        let weight = if transposed {
            Node::new(NodeKind::Permute {
                a: weight,
                dims: vec![1, 0],
            })
            .unwrap()
        } else {
            weight
        };
        let bias = Node::new(NodeKind::Zeros {
            shape: vec![2],
            dtype,
            device: Device::Cuda(0),
        })
        .unwrap();
        Node::new(NodeKind::Linear { x, weight, bias }).unwrap()
    }

    fn classify(node: &Arc<Node>, major: i32) -> DTypeDisposition {
        let index = GraphIndex::new(std::slice::from_ref(node)).unwrap();
        let spec = OperationDTypeSpec::new(&index, index.roots[0]).unwrap();
        CudaCapabilities::new(0, major, 0).classify_node(&spec)
    }

    #[test]
    fn bf16_linear_is_native_on_ampere_and_newer() {
        for transposed in [false, true] {
            let node = bf16_linear(DType::BF16, transposed);
            match classify(&node, 12) {
                DTypeDisposition::Native(execution) => {
                    assert_eq!(execution.operations[0].compute_dtype, Some(DType::BF16));
                    assert_eq!(
                        execution.operations[0].accumulation.unwrap().dtype,
                        DType::F32
                    );
                    assert_eq!(execution.realization, ExecutionRealization::DirectKernel);
                }
                other => panic!("expected native BF16 linear, got {other:?}"),
            }
            assert!(matches!(classify(&node, 7), DTypeDisposition::Legalize(_)));
        }
    }

    #[test]
    fn f16_linear_keeps_materialized_f32_legalization() {
        let node = bf16_linear(DType::F16, false);
        assert!(matches!(classify(&node, 12), DTypeDisposition::Legalize(_)));
    }

    #[test]
    fn bf16_matmul_partial_broadcast_is_not_native() {
        let a = Node::new(NodeKind::Zeros {
            shape: vec![2, 1, 3, 4],
            dtype: DType::BF16,
            device: Device::Cuda(0),
        })
        .unwrap();
        let b = Node::new(NodeKind::Zeros {
            shape: vec![1, 5, 4, 6],
            dtype: DType::BF16,
            device: Device::Cuda(0),
        })
        .unwrap();
        let node = Node::new(NodeKind::Matmul { a, b }).unwrap();
        assert!(matches!(classify(&node, 12), DTypeDisposition::Legalize(_)));
    }

    #[test]
    fn target_fingerprint_records_native_bf16_realization() {
        let capabilities = CudaCapabilities::new(0, 12, 0);
        assert_eq!(capabilities.policy_revision(), 5);
        assert!(capabilities
            .fingerprint()
            .features
            .iter()
            .any(|feature| feature == "cublas-bf16-row-major-f32-accum-v2"));
        assert_eq!(capabilities.fingerprint().architecture, "sm_120");
        let off = CudaCapabilities::new(0, 7, 0);
        assert!(!off
            .fingerprint()
            .features
            .iter()
            .any(|feature| feature == "cublas-bf16-row-major-f32-accum-v2"));
    }

    #[test]
    fn half_arithmetic_records_materialized_boundaries() {
        let x = Node::new(NodeKind::Zeros {
            shape: vec![3],
            dtype: DType::BF16,
            device: Device::Cuda(0),
        })
        .unwrap();
        let root = Node::new(NodeKind::Add { a: x.clone(), b: x }).unwrap();
        let index = GraphIndex::new(&[root]).unwrap();
        let spec = OperationDTypeSpec::new(&index, index.roots[0]).unwrap();
        let DTypeDisposition::Legalize(execution) =
            CudaCapabilities::new(0, 8, 0).classify_node(&spec)
        else {
            panic!("half add must legalize")
        };
        assert_eq!(
            execution.realization,
            ExecutionRealization::MaterializedTransforms
        );
        assert_eq!(execution.operations[0].compute_dtype, Some(DType::F32));
        assert!(matches!(
            execution.operations[0].results[0].completion,
            effect_torch_compiler::ResultCompletion::ConvertToBoundary(_)
        ));
    }
    #[test]
    fn integer_arithmetic_does_not_convert_to_float() {
        let x = Node::new(NodeKind::Zeros {
            shape: vec![3],
            dtype: DType::I64,
            device: Device::Cuda(0),
        })
        .unwrap();
        let root = Node::new(NodeKind::Mul { a: x.clone(), b: x }).unwrap();
        let index = GraphIndex::new(&[root]).unwrap();
        let spec = OperationDTypeSpec::new(&index, index.roots[0]).unwrap();
        let DTypeDisposition::Native(execution) =
            CudaCapabilities::new(0, 8, 0).classify_node(&spec)
        else {
            panic!("integer multiply must remain typed")
        };
        assert_eq!(execution.operations[0].compute_dtype, Some(DType::I64));
    }
}

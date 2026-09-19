//! Immutable Metal dtype capabilities. Queries inspect metadata only.

use crate::device::MetalDevice;
use effect_torch_compiler::{
    DTypeDisposition, DTypeRequirement, ExecutionRealization, LayoutConstraintSpec, NativeRegion,
    OperationDTypeSpec, RegionDTypeSpec, StorageSupport, TargetBackend, TargetDTypeCapabilities,
    TargetFingerprint, UnsupportedDType, ValueSpec,
};
use effect_torch_graph::{Device, NodeKind};
use effect_torch_runtime::{DType, StorageRepresentation};
use objc2_metal::MTLDevice as _;

/// Bump when classification or realization changes.
pub(crate) const POLICY_REVISION: u64 = 4;

#[derive(Debug, Clone)]
pub(crate) struct MetalDTypeCapabilities {
    device: Device,
    fingerprint: TargetFingerprint,
    max_threadgroup_bytes: usize,
    policy_revision: u64,
}

impl MetalDTypeCapabilities {
    /// Capture immutable hardware facts before optimization. Classification does
    /// not acquire a device, inspect environment, allocate buffers, or compile MSL.
    pub(crate) fn snapshot(device: &MetalDevice) -> Self {
        let raw = device.raw();
        let max_threadgroup_bytes = raw.maxThreadgroupMemoryLength();
        let mut fingerprint =
            TargetFingerprint::new(TargetBackend::Metal, raw.name().to_string(), 1);
        fingerprint.features =
            vec![format!("threadgroup_bytes={max_threadgroup_bytes}")].into_boxed_slice();
        fingerprint.libraries = vec![
            "MSL-3.1".to_string(),
            "effect-torch-metal-kernels-1".to_string(),
        ]
        .into_boxed_slice();
        Self {
            device: Device::Metal(device.ordinal()),
            fingerprint,
            max_threadgroup_bytes,
            policy_revision: POLICY_REVISION,
        }
    }
}

fn unsupported(requirement: DTypeRequirement, reason: impl Into<String>) -> DTypeDisposition {
    DTypeDisposition::Unsupported(UnsupportedDType::new(requirement, reason))
}

fn float(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
}

impl TargetDTypeCapabilities for MetalDTypeCapabilities {
    fn device(&self) -> &Device {
        &self.device
    }
    fn fingerprint(&self) -> &TargetFingerprint {
        &self.fingerprint
    }
    fn policy_revision(&self) -> u64 {
        self.policy_revision
    }

    fn storage_support(&self, value: &ValueSpec<'_>) -> StorageSupport {
        if let Err(error) = value.validate() {
            return StorageSupport::Unsupported(UnsupportedDType::new(
                DTypeRequirement::Storage,
                error,
            ));
        }
        let failure = if value.semantic_dtype == DType::F64 {
            Some((
                DTypeRequirement::Storage,
                "Metal does not support F64 storage or arithmetic",
            ))
        } else if matches!(
            value.storage.representation,
            StorageRepresentation::Packed(_)
        ) && value.semantic_dtype != DType::F32
        {
            Some((
                DTypeRequirement::Representation,
                "GGML packed values must represent F32",
            ))
        } else if let LayoutConstraintSpec::DenseStrided(layout) = &value.storage.layout_constraint
        {
            if layout.offset() != 0
                || !layout.is_contiguous()
                || layout.shape() != value.logical_shape
            {
                Some((
                    DTypeRequirement::Layout,
                    "Metal bindings require a zero-offset dense layout",
                ))
            } else {
                None
            }
        } else {
            None
        };
        match failure {
            Some((requirement, reason)) => {
                StorageSupport::Unsupported(UnsupportedDType::new(requirement, reason))
            }
            None => StorageSupport::Supported,
        }
    }

    fn classify_node(&self, spec: &OperationDTypeSpec<'_>) -> DTypeDisposition {
        if !spec.placement.same_device(&self.device) {
            return unsupported(
                DTypeRequirement::Realization,
                "operation placement differs from the Metal target",
            );
        }
        for value in spec.operands.iter().chain(spec.results.iter()) {
            if let StorageSupport::Unsupported(error) = self.storage_support(&value.value) {
                return DTypeDisposition::Unsupported(error);
            }
        }
        if matches!(spec.operation, NodeKind::Input { .. })
            && matches!(
                spec.results[0].value.storage.layout_constraint,
                LayoutConstraintSpec::Unconstrained
            )
        {
            return unsupported(DTypeRequirement::Layout,
                "Metal input bindings require zero-offset contiguous layout; arbitrary strides, offset-contiguous policies, and canonicalization are unsupported");
        }
        let dtype = spec.results[0].value.semantic_dtype;
        let input_dtype = spec
            .operands
            .first()
            .map(|value| value.value.semantic_dtype)
            .unwrap_or(dtype);
        let native = || DTypeDisposition::Native(spec.native_execution());
        let promote = || {
            DTypeDisposition::Legalize(
                spec.native_execution()
                    .promote_half(ExecutionRealization::MaterializedTransforms),
            )
        };
        let float_operands = || {
            spec.operands
                .iter()
                .all(|value| float(value.value.semantic_dtype))
        };
        let arithmetic = || {
            // The graph explicitly gives mixed floating 0-d operands the tensor
            // dtype before arithmetic. Lowering materializes that semantic
            // conversion before the independent kernel consumes the operand.
            let scalar_boundary = float(dtype)
                && spec.operands.len() == 2
                && spec.operands[0].value.semantic_dtype != spec.operands[1].value.semantic_dtype
                && spec
                    .operands
                    .iter()
                    .any(|operand| operand.value.logical_shape.is_empty());
            if scalar_boundary && float_operands() {
                return native();
            }
            if !float_operands() || !float(input_dtype) {
                unsupported(DTypeRequirement::Compute, "Metal has no exact integer implementation for this arithmetic operation; integer-to-F32 legalization is forbidden")
            } else if spec.required_numerics.permits_f32_compute {
                promote()
            } else {
                native()
            }
        };
        match spec.operation {
            NodeKind::Inverse { .. } => unsupported(
                DTypeRequirement::Compute,
                "inverse is not supported on Metal",
            ),
            NodeKind::Det { .. } => {
                unsupported(DTypeRequirement::Compute, "det is not supported on Metal")
            }
            NodeKind::Solve { .. } => {
                unsupported(DTypeRequirement::Compute, "solve is not supported on Metal")
            }
            NodeKind::Arange { step, .. } if *step == 0.0 => {
                unsupported(DTypeRequirement::Compute, "arange step must not be zero")
            }
            NodeKind::Leaf(_)
            | NodeKind::Input { .. }
            | NodeKind::ScalarInput { .. }
            | NodeKind::FromBytes { .. }
            | NodeKind::Zeros { .. }
            | NodeKind::Ones { .. }
            | NodeKind::Full { .. }
            | NodeKind::Arange { .. }
            | NodeKind::Eye { .. }
            | NodeKind::Cast { .. }
            | NodeKind::Reshape { .. }
            | NodeKind::Permute { .. }
            | NodeKind::Slice { .. }
            | NodeKind::BroadcastTo { .. }
            | NodeKind::Concat { .. }
            | NodeKind::StopGradient { .. }
            | NodeKind::Checkpoint { .. }
            | NodeKind::Expose { .. }
            | NodeKind::SdpaBackwardOut { .. }
            | NodeKind::ChunkedHeadCeBackwardOut { .. }
            | NodeKind::KdaBackwardOut { .. }
            | NodeKind::LayerNormBackwardOut { .. }
            | NodeKind::AdamWOut { .. }
            | NodeKind::SgdOut { .. }
            | NodeKind::Gather { .. }
            | NodeKind::IndexSelect { .. }
            | NodeKind::LastTokenRow { .. }
            | NodeKind::PositionEmbedding { .. }
            | NodeKind::Argmax { .. }
            | NodeKind::Argmin { .. }
            | NodeKind::Cumsum { .. } => native(),
            NodeKind::Relu { .. } if !input_dtype.is_float() => native(),
            NodeKind::Add { a, b }
            | NodeKind::Sub { a, b }
            | NodeKind::Mul { a, b }
            | NodeKind::Eq { a, b }
            | NodeKind::Gt { a, b }
            | NodeKind::Lt { a, b }
            | NodeKind::Ge { a, b }
            | NodeKind::Le { a, b }
            | NodeKind::Maximum { a, b }
            | NodeKind::Minimum { a, b }
                if !input_dtype.is_float() && a.dtype == b.dtype =>
            {
                native()
            }
            NodeKind::Min { .. } | NodeKind::Max { .. } if !input_dtype.is_float() => native(),
            NodeKind::Add { .. }
            | NodeKind::Sub { .. }
            | NodeKind::Mul { .. }
            | NodeKind::Div { .. }
            | NodeKind::Eq { .. }
            | NodeKind::Gt { .. }
            | NodeKind::Lt { .. }
            | NodeKind::Ge { .. }
            | NodeKind::Le { .. }
            | NodeKind::Maximum { .. }
            | NodeKind::Minimum { .. }
            | NodeKind::Neg { .. }
            | NodeKind::Abs { .. }
            | NodeKind::Sqrt { .. }
            | NodeKind::Exp { .. }
            | NodeKind::Log { .. }
            | NodeKind::Sin { .. }
            | NodeKind::Cos { .. }
            | NodeKind::Tanh { .. }
            | NodeKind::Relu { .. }
            | NodeKind::Erf { .. }
            | NodeKind::Gelu { .. }
            | NodeKind::Floor { .. }
            | NodeKind::Ceil { .. }
            | NodeKind::Round { .. }
            | NodeKind::Sign { .. }
            | NodeKind::Pow { .. }
            | NodeKind::Sum { .. }
            | NodeKind::Mean { .. }
            | NodeKind::Max { .. }
            | NodeKind::Min { .. }
            | NodeKind::Prod { .. } => arithmetic(),
            NodeKind::Where { .. } => native(),
            NodeKind::QuantizedLinear { .. } | NodeKind::QuantizedEmbedding { .. } => {
                if dtype != DType::F32 {
                    unsupported(
                        DTypeRequirement::Result(effect_torch_compiler::ValueRole::Result(0)),
                        "canonical GGML kernels produce F32",
                    )
                } else if self.max_threadgroup_bytes < 8192 {
                    unsupported(
                        DTypeRequirement::Realization,
                        "canonical packed MMA requires 8 KiB threadgroup memory",
                    )
                } else {
                    let mut execution = spec.native_execution();
                    if let NodeKind::QuantizedLinear { x, .. } = spec.operation {
                        let columns = x.shape.last().copied().unwrap_or(0);
                        let vectors = if columns == 0 {
                            0
                        } else {
                            x.shape.iter().product::<usize>() / columns
                        };
                        if vectors >= 8 && vectors.is_multiple_of(8) {
                            execution.operations[0].operands[1].preparation =
                                effect_torch_compiler::OperandPreparation::CanonicalPacked(
                                    effect_torch_compiler::PackedOperandAccess::TileBounded {
                                        max_logical_elements: 2048,
                                        max_scratch_bytes: 8192,
                                    },
                                );
                        }
                    }
                    DTypeDisposition::Native(execution)
                }
            }
            NodeKind::KdaChunk { .. }
            | NodeKind::KdaRecurrence { .. }
            | NodeKind::KdaBackward { .. }
            | NodeKind::ShortConv1d { .. }
            | NodeKind::ShortConv1dBackwardX { .. }
            | NodeKind::ShortConv1dBackwardW { .. }
            | NodeKind::ConvState { .. } => {
                if matches!(input_dtype, DType::F32 | DType::BF16) {
                    native()
                } else {
                    unsupported(
                        DTypeRequirement::Compute,
                        "KDA and short convolution require F32 or BF16",
                    )
                }
            }
            NodeKind::Matmul { .. }
            | NodeKind::Linear { .. }
            | NodeKind::CrossEntropy { .. }
            | NodeKind::CrossEntropyBackward { .. }
            | NodeKind::ChunkedHeadCe { .. }
            | NodeKind::ChunkedHeadCeBackward { .. }
            | NodeKind::Sdpa { .. }
            | NodeKind::SdpaBackward { .. }
            | NodeKind::KvAttention { .. }
            | NodeKind::RotaryEmbedding { .. }
            | NodeKind::RotaryEmbeddingBackward { .. }
            | NodeKind::LayerNorm { .. }
            | NodeKind::RmsNorm { .. }
            | NodeKind::LayerNormBackward { .. }
            | NodeKind::Conv1d { .. }
            | NodeKind::Conv2d { .. }
            | NodeKind::ConvTranspose1d { .. }
            | NodeKind::ConvTranspose2d { .. }
            | NodeKind::Conv1dBackwardW { .. }
            | NodeKind::Conv2dBackwardW { .. }
            | NodeKind::ScatterAdd { .. }
            | NodeKind::AdamWStep { .. }
            | NodeKind::SgdStep { .. }
            | NodeKind::Randn { .. }
            | NodeKind::Uniform { .. } => {
                if float(input_dtype) && float(dtype) {
                    native()
                } else {
                    unsupported(
                        DTypeRequirement::Compute,
                        "Metal kernel requires F32, F16, or BF16 arithmetic",
                    )
                }
            }
        }
    }

    fn classify_region(&self, spec: &RegionDTypeSpec<'_>) -> DTypeDisposition {
        let Some(first) = spec.boundary_results.first() else {
            return unsupported(DTypeRequirement::Region, "region has no results");
        };
        let dtype = first.value.semantic_dtype;
        if !float(dtype) {
            return unsupported(
                DTypeRequirement::Region,
                "Metal fusion requires floating storage",
            );
        }
        // All lanes in the existing fused emitter have the same storage type.
        if spec.boundary_inputs.iter().any(|value| {
            value.value.semantic_dtype != dtype
                || value.value.storage.representation != StorageRepresentation::Dense
        }) {
            return unsupported(
                DTypeRequirement::Region,
                "Metal fusion requires uniform dense lane types",
            );
        }
        if matches!(
            spec.region,
            NativeRegion::AdamW(_) | NativeRegion::AdamWGroup(_) | NativeRegion::Sgd(_)
        ) && dtype != DType::F32
        {
            return unsupported(
                DTypeRequirement::Region,
                "low-precision optimizer regions require separate semantic boundary support",
            );
        }
        let mut execution = spec.native_execution();
        let mut transformed = false;
        for (operation, recipe) in spec.operations.iter().zip(execution.operations.iter_mut()) {
            if operation.operands.len() == 2
                && operation.operands[0].value.semantic_dtype
                    != operation.operands[1].value.semantic_dtype
                && operation
                    .operands
                    .iter()
                    .any(|operand| operand.value.logical_shape.is_empty())
            {
                return unsupported(
                    DTypeRequirement::Rounding,
                    "mixed scalar conversion uses the independent typed scalar ABI",
                );
            }
            match self.classify_node(operation) {
                DTypeDisposition::Unsupported(error) => {
                    return DTypeDisposition::Unsupported(error)
                }
                DTypeDisposition::Native(plan) => *recipe = plan.operations[0].clone(),
                DTypeDisposition::Legalize(plan) => {
                    transformed = true;
                    *recipe = plan.operations[0].clone();
                }
            }
        }
        if transformed {
            execution.realization = ExecutionRealization::KernelLocal;
            DTypeDisposition::Legalize(execution)
        } else {
            DTypeDisposition::Native(execution)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use effect_torch_compiler::{
        CompileOptions, OperandPreparation, ProgramRequest, ResultCompletion,
    };
    use effect_torch_graph::Node;
    use effect_torch_runtime::{Layout, StorageMetadata};

    fn target() -> MetalDTypeCapabilities {
        MetalDTypeCapabilities {
            device: Device::Metal(0),
            fingerprint: TargetFingerprint::new(TargetBackend::Metal, "test-apple", 1),
            max_threadgroup_bytes: 32768,
            policy_revision: POLICY_REVISION,
        }
    }

    #[test]
    fn metadata_queries_preserve_half_boundaries_and_reject_unsafe_integer_math() {
        let target = target();
        for dtype in [DType::F16, DType::BF16, DType::F32, DType::I64] {
            let input = Node::new(NodeKind::Input {
                slot: 0,
                shape: vec![4],
                dtype,
                device: Device::Metal(0),
                storage: StorageMetadata::dense(),
            })
            .unwrap();
            let root = Node::new(NodeKind::Div {
                a: input.clone(),
                b: input,
            })
            .unwrap();
            let prepared = ProgramRequest::from_roots(vec![root], CompileOptions::default())
                .prepare()
                .unwrap();
            let spec = OperationDTypeSpec::new(&prepared.index, prepared.index.roots[0]).unwrap();
            let first = target.classify_node(&spec);
            assert_eq!(first, target.classify_node(&spec));
            match dtype {
                DType::F16 | DType::BF16 => {
                    let DTypeDisposition::Legalize(execution) = first else {
                        panic!("expected half legalization")
                    };
                    let operation = &execution.operations[0];
                    assert_eq!(
                        execution.realization,
                        ExecutionRealization::MaterializedTransforms
                    );
                    assert_eq!(operation.compute_dtype, Some(DType::F32));
                    assert!(operation.operands.iter().all(|operand| matches!(
                        operand.preparation,
                        OperandPreparation::Convert(_)
                    )));
                    assert!(matches!(
                        operation.results[0].completion,
                        ResultCompletion::ConvertToBoundary(_)
                    ));
                    assert_eq!(operation.rounding_boundaries[0].dtype, dtype);
                }
                DType::F32 => assert!(matches!(first, DTypeDisposition::Native(_))),
                _ => assert!(matches!(first, DTypeDisposition::Unsupported(_))),
            }
        }
    }

    #[test]
    fn storage_policy_checks_geometry_layout_and_f64() {
        let target = target();
        let mut spec = ValueSpec::dense(DType::F16, &[2, 3]);
        assert_eq!(target.storage_support(&spec), StorageSupport::Supported);
        let transposed = Layout::contiguous(vec![3, 2]).permute(&[1, 0]);
        spec.storage.layout_constraint = LayoutConstraintSpec::DenseStrided(&transposed);
        assert!(matches!(
            target.storage_support(&spec),
            StorageSupport::Unsupported(_)
        ));
        assert!(matches!(
            target.storage_support(&ValueSpec::dense(DType::F64, &[1])),
            StorageSupport::Unsupported(_)
        ));
        let storage = StorageMetadata::packed(effect_torch_runtime::GgmlKQuant::Q4K);
        let packed = ValueSpec {
            semantic_dtype: DType::F32,
            logical_shape: &[2, 256],
            storage: storage.as_spec(),
        };
        assert_eq!(target.storage_support(&packed), StorageSupport::Supported);
        let invalid = ValueSpec {
            logical_shape: &[2, 255],
            ..packed
        };
        assert!(matches!(
            target.storage_support(&invalid),
            StorageSupport::Unsupported(_)
        ));
    }
}

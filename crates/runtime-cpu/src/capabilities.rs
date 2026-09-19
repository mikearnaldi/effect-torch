//! CPU operation legality and immutable dtype execution recipes.
//! Queries use semantic metadata only; kernel preparation and allocation happen later.

use effect_torch_compiler::{
    DTypeDisposition, DTypeRequirement, ExecutionRealization, LayoutConstraintSpec, NativeRegion,
    OperationDTypeSpec, RegionDTypeSpec, StorageSupport, TargetBackend, TargetDTypeCapabilities,
    TargetFingerprint, UnsupportedDType, ValueRole, ValueSpec,
};
use effect_torch_graph::{node_children, Device, NodeKind};
use effect_torch_runtime::{DType, StorageRepresentation};

/// CPU reference-kernel policy. Revisions participate in executable cache identity.
#[derive(Debug, Clone)]
pub struct CpuDTypeCapabilities {
    device: Device,
    fingerprint: TargetFingerprint,
}

impl Default for CpuDTypeCapabilities {
    fn default() -> Self {
        let mut fingerprint = TargetFingerprint::new(TargetBackend::Cpu, std::env::consts::ARCH, 1);
        fingerprint.libraries = vec!["effect-torch-reference-v1".to_string()].into_boxed_slice();
        Self {
            device: Device::Cpu(0),
            fingerprint,
        }
    }
}

fn unsupported(requirement: DTypeRequirement, reason: impl Into<String>) -> DTypeDisposition {
    DTypeDisposition::Unsupported(UnsupportedDType::new(requirement, reason))
}

fn half(dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::BF16)
}

impl TargetDTypeCapabilities for CpuDTypeCapabilities {
    fn device(&self) -> &Device {
        &self.device
    }
    fn fingerprint(&self) -> &TargetFingerprint {
        &self.fingerprint
    }
    fn policy_revision(&self) -> u64 {
        2
    }

    fn storage_support(&self, value: &ValueSpec<'_>) -> StorageSupport {
        let error = |reason| {
            StorageSupport::Unsupported(UnsupportedDType::new(DTypeRequirement::Storage, reason))
        };
        if value
            .logical_shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .and_then(|n| n.checked_mul(value.semantic_dtype.size_in_bytes()))
            .is_none()
        {
            return error("CPU value byte size overflows");
        }
        if let Err(reason) = value.validate() {
            return StorageSupport::Unsupported(UnsupportedDType::new(
                DTypeRequirement::Storage,
                reason,
            ));
        }
        match value.storage.representation {
            StorageRepresentation::Dense => StorageSupport::Supported,
            StorageRepresentation::Packed(format) => {
                if value.semantic_dtype != DType::F32 || value.logical_shape.is_empty() {
                    return error("packed CPU values require logical f32 rows");
                }
                let effect_torch_runtime::PackedFormat::GgmlKQuant(codec) = format;
                if codec
                    .encoded_row_bytes(*value.logical_shape.last().unwrap())
                    .is_none()
                {
                    return error("invalid packed CPU row geometry");
                }
                if matches!(
                    value.storage.layout_constraint,
                    LayoutConstraintSpec::DenseStrided(_)
                ) {
                    return error("packed CPU storage must use canonical layout");
                }
                StorageSupport::Supported
            }
        }
    }

    fn classify_node(&self, spec: &OperationDTypeSpec<'_>) -> DTypeDisposition {
        if !spec.placement.same_device(&self.device) {
            return unsupported(
                DTypeRequirement::Layout,
                "CPU runtime only supports device cpu:0",
            );
        }
        let children = node_children(spec.operation);
        if children.len() != spec.operands.len() || spec.results.is_empty() {
            return unsupported(
                DTypeRequirement::Realization,
                "incomplete CPU operation operand/result specification",
            );
        }
        if matches!(spec.operation, NodeKind::Input { .. })
            && matches!(
                spec.results[0].value.storage.layout_constraint,
                LayoutConstraintSpec::Unconstrained
            )
        {
            return unsupported(
                DTypeRequirement::Layout,
                "CPU inputs require canonical or exact strided layout metadata",
            );
        }
        let selector = matches!(
            spec.operation,
            NodeKind::SdpaBackwardOut { .. }
                | NodeKind::LayerNormBackwardOut { .. }
                | NodeKind::ChunkedHeadCeBackwardOut { .. }
                | NodeKind::KdaBackwardOut { .. }
                | NodeKind::AdamWOut { .. }
                | NodeKind::SgdOut { .. }
        );
        for (index, (child, operand)) in children.iter().zip(spec.operands.iter()).enumerate() {
            let expected = if selector {
                spec.results[0].value
            } else {
                child.value_spec()
            };
            if operand.index as usize != index
                || operand.role != operand_role(spec.operation, index)
            {
                return unsupported(
                    DTypeRequirement::Operand(operand.role),
                    "CPU operand role does not match the operation",
                );
            }
            if expected.semantic_dtype != operand.value.semantic_dtype
                || expected.logical_shape != operand.value.logical_shape
                || expected.storage.representation != operand.value.storage.representation
            {
                return unsupported(
                    DTypeRequirement::Operand(operand.role),
                    "operand metadata differs from the semantic operation",
                );
            }
            if let StorageSupport::Unsupported(error) = self.storage_support(&operand.value) {
                return DTypeDisposition::Unsupported(error);
            }
            if operand.value.storage.representation != StorageRepresentation::Dense
                && !(operand.role == ValueRole::Weight
                    && matches!(
                        spec.operation,
                        NodeKind::QuantizedLinear { .. } | NodeKind::QuantizedEmbedding { .. }
                    ))
            {
                return unsupported(
                    DTypeRequirement::Operand(operand.role),
                    "packed storage is only consumed by canonical quantized operations",
                );
            }
        }
        for result in &spec.results {
            if let StorageSupport::Unsupported(error) = self.storage_support(&result.value) {
                return DTypeDisposition::Unsupported(error);
            }
        }
        if let Err(error) = validate_operation(spec.operation) {
            return unsupported(DTypeRequirement::Compute, error);
        }
        // Only these kernels implement the shared, operation-specific F64 algorithms.
        if matches!(
            spec.operation,
            NodeKind::Inverse { .. }
                | NodeKind::Det { .. }
                | NodeKind::Solve { .. }
                | NodeKind::Randn { .. }
                | NodeKind::Uniform { .. }
                | NodeKind::Arange { .. }
        ) {
            return match spec.f64_execution() {
                Ok(execution) if execution.realization == ExecutionRealization::DirectKernel => {
                    DTypeDisposition::Native(execution)
                }
                Ok(execution) => DTypeDisposition::Legalize(execution),
                Err(reason) => unsupported(DTypeRequirement::Compute, reason),
            };
        }
        let execution = spec.native_execution();
        let dtype = spec.results[0].value.semantic_dtype;
        if half(dtype)
            && matches!(
                spec.operation,
                NodeKind::Matmul { .. }
                    | NodeKind::Linear { .. }
                    | NodeKind::Sdpa { .. }
                    | NodeKind::SdpaBackward { .. }
            )
        {
            return DTypeDisposition::Legalize(
                execution.promote_half(ExecutionRealization::MaterializedTransforms),
            );
        }
        if spec.required_numerics.permits_f32_compute {
            return DTypeDisposition::Legalize(
                execution.promote_half(ExecutionRealization::KernelLocal),
            );
        }
        DTypeDisposition::Native(execution)
    }

    fn classify_region(&self, spec: &RegionDTypeSpec<'_>) -> DTypeDisposition {
        match spec.region {
            NativeRegion::Elementwise(_)
            | NativeRegion::ElementwiseReduce(_)
            | NativeRegion::MultiOutput(_) => {}
            NativeRegion::AdamW(_) | NativeRegion::AdamWGroup(_) | NativeRegion::Sgd(_) => {
                if spec
                    .boundary_results
                    .iter()
                    .any(|value| half(value.value.semantic_dtype))
                {
                    return unsupported(
                        DTypeRequirement::Region,
                        "half optimizer fusion has no approved internal rounding recipe",
                    );
                }
            }
            NativeRegion::LinearResidual(_) | NativeRegion::LinearGelu(_) => {
                return unsupported(
                    DTypeRequirement::Region,
                    "CPU linear epilogue regions are not implemented by this policy",
                );
            }
        }
        let Some(first) = spec.boundary_inputs.first() else {
            return unsupported(DTypeRequirement::Region, "CPU fusion requires an input");
        };
        let dtype = first.value.semantic_dtype;
        if !dtype.is_float()
            || spec
                .boundary_inputs
                .iter()
                .chain(spec.boundary_results.iter())
                .any(|value| {
                    value.value.semantic_dtype != dtype
                        || value.value.storage.representation != StorageRepresentation::Dense
                })
        {
            return unsupported(
                DTypeRequirement::Region,
                "CPU fusion requires uniform dense floating boundary lanes",
            );
        }
        for operation in &spec.operations {
            if operation
                .operands
                .iter()
                .chain(operation.results.iter())
                .any(|value| value.value.semantic_dtype != dtype)
            {
                return unsupported(DTypeRequirement::Region, "CPU expression interpreter requires one semantic storage dtype throughout a region");
            }
            match self.classify_node(operation) {
                DTypeDisposition::Unsupported(error) => {
                    return DTypeDisposition::Unsupported(error)
                }
                DTypeDisposition::Legalize(execution)
                    if execution.realization != ExecutionRealization::KernelLocal =>
                {
                    return unsupported(
                        DTypeRequirement::Region,
                        "CPU fusion cannot materialize conversions inside an expression",
                    );
                }
                _ => {}
            }
        }
        let execution = spec.native_execution();
        if half(dtype) {
            DTypeDisposition::Legalize(execution.promote_half(ExecutionRealization::KernelLocal))
        } else {
            DTypeDisposition::Native(execution)
        }
    }
}

fn operand_role(operation: &NodeKind, index: usize) -> ValueRole {
    match operation {
        NodeKind::Where { .. } => [
            ValueRole::Condition,
            ValueRole::TrueValue,
            ValueRole::FalseValue,
        ][index],
        NodeKind::Matmul { .. }
        | NodeKind::Add { .. }
        | NodeKind::Sub { .. }
        | NodeKind::Mul { .. }
        | NodeKind::Div { .. }
        | NodeKind::Eq { .. }
        | NodeKind::Gt { .. }
        | NodeKind::Lt { .. }
        | NodeKind::Ge { .. }
        | NodeKind::Le { .. }
        | NodeKind::Maximum { .. }
        | NodeKind::Minimum { .. } => {
            if index == 0 {
                ValueRole::Lhs
            } else {
                ValueRole::Rhs
            }
        }
        NodeKind::Linear { .. } | NodeKind::QuantizedLinear { .. } => {
            [ValueRole::Activation, ValueRole::Weight, ValueRole::Bias][index]
        }
        NodeKind::QuantizedEmbedding { .. } => [ValueRole::Indices, ValueRole::Weight][index],
        NodeKind::AdamWStep { .. } => match index {
            0 => ValueRole::Parameter,
            1 => ValueRole::Gradient,
            2 => ValueRole::FirstMoment,
            3 => ValueRole::SecondMoment,
            _ => ValueRole::Scalar(index as u32 - 4),
        },
        NodeKind::SgdStep { .. } => match index {
            0 => ValueRole::Parameter,
            1 => ValueRole::Gradient,
            2 => ValueRole::Velocity,
            _ => ValueRole::Scalar(index as u32 - 3),
        },
        _ => ValueRole::Input(index as u32),
    }
}

fn validate_operation(kind: &NodeKind) -> Result<(), String> {
    let floating = |dtype: DType| {
        if dtype.is_float() {
            Ok(())
        } else {
            Err(format!(
                "CPU operation requires floating dtype, got {dtype}"
            ))
        }
    };
    match kind {
        NodeKind::Neg { a } if a.dtype == DType::U8 => {
            Err("neg does not support CPU dtype u8".into())
        }
        NodeKind::Argmax { a, dim } | NodeKind::Argmin { a, dim }
            if a.shape.get(*dim) == Some(&0) =>
        {
            Err("argmax/argmin cannot reduce an empty axis".into())
        }
        NodeKind::Abs { a }
        | NodeKind::Sqrt { a }
        | NodeKind::Exp { a }
        | NodeKind::Log { a }
        | NodeKind::Sin { a }
        | NodeKind::Cos { a }
        | NodeKind::Tanh { a }
        | NodeKind::Erf { a }
        | NodeKind::Gelu { a, .. }
        | NodeKind::Floor { a }
        | NodeKind::Ceil { a }
        | NodeKind::Round { a }
        | NodeKind::Sign { a }
        | NodeKind::Pow { a, .. }
        | NodeKind::Mean { a, .. } => floating(a.dtype),
        NodeKind::Matmul { a, b } => {
            if a.dtype != b.dtype || a.shape.len() < 2 || b.shape.len() < 2 {
                return Err("matmul requires same dtype and rank >= 2".into());
            }
            Ok(())
        }
        NodeKind::Linear { x, weight, bias } => {
            if x.dtype != weight.dtype || x.dtype != bias.dtype {
                return Err("linear requires matching activation/weight/bias dtypes".into());
            }
            Ok(())
        }
        NodeKind::Sdpa { q, k, v, .. } | NodeKind::SdpaBackward { q, k, v, .. } => {
            floating(q.dtype)?;
            if q.dtype != k.dtype || q.dtype != v.dtype {
                return Err("sdpa q/k/v dtypes must match".into());
            }
            if q.shape.len() < 2
                || q.shape.len() != k.shape.len()
                || q.shape.len() != v.shape.len()
                || q.shape.contains(&0)
                || k.shape.contains(&0)
                || v.shape.contains(&0)
            {
                return Err("sdpa requires matching positive ranks and dimensions".into());
            }
            let rank = q.shape.len();
            if matches!(kind, NodeKind::SdpaBackward { .. })
                && rank >= 3
                && q.shape[rank - 3] != k.shape[rank - 3]
            {
                return Err("sdpa backward: grouped-query attention is not differentiable".into());
            }
            Ok(())
        }
        NodeKind::ChunkedHeadCe { x, .. } | NodeKind::ChunkedHeadCeBackward { x, .. } => {
            if matches!(x.dtype, DType::F32 | DType::F64) {
                Ok(())
            } else {
                Err("chunked head CE requires CPU f32 or f64".into())
            }
        }
        NodeKind::KvAttention { q, .. } if q.dtype != DType::F32 => {
            Err("kv_attention requires CPU f32".into())
        }
        NodeKind::LayerNorm {
            x, weight, bias, ..
        } => {
            floating(x.dtype)?;
            if x.dtype != weight.dtype
                || x.dtype != bias.dtype
                || weight.shape.contains(&0)
                || x.shape.contains(&0)
            {
                return Err(
                    "layer_norm requires matching float dtypes and nonzero dimensions".into(),
                );
            }
            Ok(())
        }
        NodeKind::RmsNorm { x, weight, .. } => {
            floating(x.dtype)?;
            if weight
                .as_ref()
                .is_some_and(|weight| weight.dtype != x.dtype)
                || x.shape.contains(&0)
            {
                return Err(
                    "rms_norm requires matching float dtypes and nonzero dimensions".into(),
                );
            }
            Ok(())
        }
        NodeKind::LayerNormBackward { x, weight, g, .. } => {
            floating(x.dtype)?;
            if x.dtype != weight.dtype || x.dtype != g.dtype {
                return Err("layer_norm backward requires matching float dtypes".into());
            }
            Ok(())
        }
        NodeKind::Arange { step, .. } if *step == 0.0 => Err("arange step must not be zero".into()),
        NodeKind::Slice { ranges, .. }
            if ranges
                .iter()
                .any(|&(start, stop, stride)| stop <= start || stride == 0) =>
        {
            Err("zero-length CPU slices are unsupported".into())
        }
        NodeKind::Leaf(_)
        | NodeKind::Input { .. }
        | NodeKind::ScalarInput { .. }
        | NodeKind::FromBytes { .. }
        | NodeKind::Zeros { .. }
        | NodeKind::Ones { .. }
        | NodeKind::Full { .. }
        | NodeKind::Randn { .. }
        | NodeKind::Uniform { .. }
        | NodeKind::Arange { .. }
        | NodeKind::Eye { .. }
        | NodeKind::Add { .. }
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
        | NodeKind::Relu { .. }
        | NodeKind::Where { .. }
        | NodeKind::Cast { .. }
        | NodeKind::Sum { .. }
        | NodeKind::Max { .. }
        | NodeKind::Min { .. }
        | NodeKind::Prod { .. }
        | NodeKind::Argmax { .. }
        | NodeKind::Argmin { .. }
        | NodeKind::Cumsum { .. }
        | NodeKind::IndexSelect { .. }
        | NodeKind::ScatterAdd { .. }
        | NodeKind::Gather { .. }
        | NodeKind::CrossEntropy { .. }
        | NodeKind::CrossEntropyBackward { .. }
        | NodeKind::SdpaBackwardOut { .. }
        | NodeKind::PositionEmbedding { .. }
        | NodeKind::KvAttention { .. }
        | NodeKind::KdaChunk { .. }
        | NodeKind::KdaRecurrence { .. }
        | NodeKind::KdaBackward { .. }
        | NodeKind::KdaBackwardOut { .. }
        | NodeKind::ShortConv1d { .. }
        | NodeKind::ConvState { .. }
        | NodeKind::LastTokenRow { .. }
        | NodeKind::ChunkedHeadCeBackwardOut { .. }
        | NodeKind::ShortConv1dBackwardX { .. }
        | NodeKind::ShortConv1dBackwardW { .. }
        | NodeKind::RotaryEmbedding { .. }
        | NodeKind::RotaryEmbeddingBackward { .. }
        | NodeKind::LayerNormBackwardOut { .. }
        | NodeKind::QuantizedLinear { .. }
        | NodeKind::QuantizedEmbedding { .. }
        | NodeKind::Conv1d { .. }
        | NodeKind::Conv2d { .. }
        | NodeKind::ConvTranspose1d { .. }
        | NodeKind::ConvTranspose2d { .. }
        | NodeKind::Conv1dBackwardW { .. }
        | NodeKind::Conv2dBackwardW { .. }
        | NodeKind::Reshape { .. }
        | NodeKind::Permute { .. }
        | NodeKind::Slice { .. }
        | NodeKind::Concat { .. }
        | NodeKind::BroadcastTo { .. }
        | NodeKind::Inverse { .. }
        | NodeKind::Det { .. }
        | NodeKind::Solve { .. }
        | NodeKind::AdamWStep { .. }
        | NodeKind::AdamWOut { .. }
        | NodeKind::SgdStep { .. }
        | NodeKind::SgdOut { .. }
        | NodeKind::StopGradient { .. }
        | NodeKind::Checkpoint { .. }
        | NodeKind::Expose { .. } => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use effect_torch_compiler::{
        CompileOptions, CompilerDriver, DenseConversionContract, OperandPreparation,
        ProgramRequest, ResultCompletion,
    };
    use effect_torch_graph::Node;
    use effect_torch_runtime::StorageMetadata;

    #[test]
    fn f64_algorithm_plans_record_conversions_and_preserve_random_sources() {
        let capabilities = CpuDTypeCapabilities::default();
        for dtype in [
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
            DType::U8,
            DType::U32,
            DType::I64,
        ] {
            let input = Node::new(NodeKind::Input {
                slot: 0,
                shape: vec![2, 2],
                dtype,
                device: Device::Cpu(0),
                storage: StorageMetadata::dense(),
            })
            .unwrap();
            let mut roots = vec![
                Node::new(NodeKind::Randn {
                    shape: vec![2, 3],
                    dtype,
                    device: Device::Cpu(0),
                })
                .unwrap(),
                Node::new(NodeKind::Arange {
                    start: 0.,
                    end: 6.,
                    step: 1.,
                    dtype,
                    device: Device::Cpu(0),
                })
                .unwrap(),
            ];
            if dtype.is_float() {
                roots.extend([
                    Node::new(NodeKind::Uniform {
                        lo: -1.,
                        hi: 1.,
                        shape: vec![2, 3],
                        dtype,
                        device: Device::Cpu(0),
                    })
                    .unwrap(),
                    Node::new(NodeKind::Inverse { a: input.clone() }).unwrap(),
                    Node::new(NodeKind::Det { a: input.clone() }).unwrap(),
                    Node::new(NodeKind::Solve {
                        a: input.clone(),
                        b: input.clone(),
                    })
                    .unwrap(),
                ]);
            }
            let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
                .prepare()
                .unwrap();
            let driver = CompilerDriver::new(&prepared, &capabilities).unwrap();
            for &root in &prepared.index.roots {
                let unit = driver
                    .legalization()
                    .units()
                    .iter()
                    .find(|unit| unit.execution().operation(root).is_some())
                    .unwrap();
                let execution = unit.execution();
                let operation = execution.operation(root).unwrap();
                let spec = OperationDTypeSpec::new(&prepared.index, root).unwrap();
                assert_eq!(operation.compute_dtype, Some(DType::F64));
                assert_eq!(
                    operation
                        .accumulation
                        .map(|accumulation| accumulation.dtype),
                    if matches!(
                        spec.operation,
                        NodeKind::Inverse { .. } | NodeKind::Det { .. } | NodeKind::Solve { .. }
                    ) {
                        Some(DType::F64)
                    } else {
                        None
                    }
                );
                assert_eq!(
                    operation.rounding_boundaries,
                    spec.required_numerics.rounding_boundaries
                );
                assert_eq!(operation.random, spec.required_numerics.random);
                if matches!(
                    spec.operation,
                    NodeKind::Randn { .. } | NodeKind::Uniform { .. }
                ) {
                    let random = operation.random.unwrap();
                    assert_eq!(random.samples, 6);
                    assert_eq!(random.source.node, root);
                    assert_eq!(
                        random.source.provenance,
                        prepared.index.order[root.index()].id
                    );
                }
                assert_eq!(
                    execution.realization,
                    if dtype == DType::F64 {
                        ExecutionRealization::DirectKernel
                    } else {
                        ExecutionRealization::KernelLocal
                    }
                );
                for operand in &operation.operands {
                    assert_eq!(operand.execution_dtype, DType::F64);
                    assert_eq!(
                        operand.preparation,
                        if dtype == DType::F64 {
                            OperandPreparation::Direct
                        } else {
                            OperandPreparation::Convert(DenseConversionContract::new(
                                dtype,
                                DType::F64,
                            ))
                        }
                    );
                }
                for result in &operation.results {
                    assert_eq!(result.execution_dtype, DType::F64);
                    assert_eq!(
                        result.completion,
                        if dtype == DType::F64 {
                            ResultCompletion::Direct
                        } else {
                            ResultCompletion::ConvertToBoundary(DenseConversionContract::new(
                                DType::F64,
                                dtype,
                            ))
                        }
                    );
                }
            }
            let add = Node::new(NodeKind::Add {
                a: input.clone(),
                b: input,
            })
            .unwrap();
            let prepared = ProgramRequest::from_roots(vec![add], CompileOptions::default())
                .prepare()
                .unwrap();
            let spec = OperationDTypeSpec::new(&prepared.index, prepared.index.roots[0]).unwrap();
            assert!(spec.f64_execution().is_err());
            let disposition = capabilities.classify_node(&spec);
            let (DTypeDisposition::Native(execution) | DTypeDisposition::Legalize(execution)) =
                disposition
            else {
                panic!("expected arithmetic recipe")
            };
            assert_eq!(
                execution.operations[0].compute_dtype,
                Some(if half(dtype) { DType::F32 } else { dtype })
            );
        }
    }

    #[test]
    fn metadata_queries_select_half_materialization_and_preserve_f64_and_i64() {
        let capabilities = CpuDTypeCapabilities::default();
        for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64, DType::I64] {
            let input = |slot| {
                Node::new(NodeKind::Input {
                    slot,
                    shape: vec![2, 2],
                    dtype,
                    device: Device::Cpu(0),
                    storage: StorageMetadata::dense(),
                })
                .unwrap()
            };
            let root = Node::new(NodeKind::Matmul {
                a: input(0),
                b: input(1),
            })
            .unwrap();
            let prepared = ProgramRequest::from_roots(vec![root], CompileOptions::default())
                .prepare()
                .unwrap();
            let mut spec =
                OperationDTypeSpec::new(&prepared.index, prepared.index.roots[0]).unwrap();
            let disposition = {
                let _guard = crate::ExecutableAllocationGuard::enter();
                capabilities.classify_node(&spec)
            };
            match disposition {
                DTypeDisposition::Legalize(execution) if half(dtype) => {
                    assert_eq!(
                        execution.realization,
                        ExecutionRealization::MaterializedTransforms
                    );
                    assert_eq!(execution.operations[0].compute_dtype, Some(DType::F32));
                    assert_eq!(
                        execution.operations[0].results[0].execution_dtype,
                        DType::F32
                    );
                }
                DTypeDisposition::Native(execution) if !half(dtype) => {
                    assert_eq!(execution.operations[0].compute_dtype, Some(dtype))
                }
                other => panic!("unexpected CPU disposition {other:?}"),
            }
            spec.operands[1].role = ValueRole::Indices;
            assert!(matches!(
                capabilities.classify_node(&spec),
                DTypeDisposition::Unsupported(_)
            ));
        }
    }

    #[test]
    fn storage_queries_reject_invalid_packed_and_overflowing_values() {
        let capabilities = CpuDTypeCapabilities::default();
        let storage = StorageMetadata::packed(effect_torch_runtime::GgmlKQuant::Q4K);
        for (dtype, shape) in [
            (DType::U8, vec![2, 256]),
            (DType::F32, vec![2, 255]),
            (DType::F32, vec![usize::MAX, 256]),
        ] {
            let spec = ValueSpec {
                semantic_dtype: dtype,
                logical_shape: &shape,
                storage: storage.as_spec(),
            };
            assert!(matches!(
                capabilities.storage_support(&spec),
                StorageSupport::Unsupported(_)
            ));
        }
        let spec = ValueSpec {
            semantic_dtype: DType::F32,
            logical_shape: &[2, 256],
            storage: storage.as_spec(),
        };
        assert!(matches!(
            capabilities.storage_support(&spec),
            StorageSupport::Supported
        ));
    }
}

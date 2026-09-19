//! Checked, same-device execution contracts for semantic lowering units.
//!
//! Targets classify complete operations and regions. The compiler validates the
//! returned recipes, retains accepted region decisions, and freezes one recipe
//! per lowering unit before any backend lowering begins.

use crate::{DenseNodeId, GraphIndex, KernelExpr, LoweringUnit, NativeRegion, OptimizationPlan};
use effect_torch_graph::{Device, Node, NodeKind};
use effect_torch_runtime::{DType, StorageRepresentation};
pub use effect_torch_runtime::{LayoutConstraintSpec, StorageSpec, ValueSpec};
use std::collections::HashMap;
use std::sync::Arc;

/// Stable target facts used by executable caches. Feature and library entries
/// must be canonical and sorted; process identities and timings do not belong here.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TargetFingerprint {
    pub backend: TargetBackend,
    pub architecture: String,
    pub features: Box<[String]>,
    pub libraries: Box<[String]>,
    pub lowering_abi_revision: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetBackend {
    Cpu,
    Metal,
    Cuda,
}

impl TargetFingerprint {
    pub fn new(
        backend: TargetBackend,
        architecture: impl Into<String>,
        lowering_abi_revision: u64,
    ) -> Self {
        Self {
            backend,
            architecture: architecture.into(),
            features: Box::new([]),
            libraries: Box::new([]),
            lowering_abi_revision,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), String> {
        let backend_matches = matches!(
            (self.backend, device),
            (TargetBackend::Cpu, Device::Cpu(_))
                | (TargetBackend::Metal, Device::Metal(_))
                | (TargetBackend::Cuda, Device::Cuda(_))
        );
        if !backend_matches || self.architecture.is_empty() {
            return Err(
                "legalization: target fingerprint does not identify the selected device".into(),
            );
        }
        if self.features.windows(2).any(|pair| pair[0] >= pair[1])
            || self.libraries.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(
                "legalization: target features and libraries must be sorted and unique".into(),
            );
        }
        Ok(())
    }
}

/// An operand's semantic role, independent of its representation or compute type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueRole {
    Input(u32),
    Lhs,
    Rhs,
    Condition,
    TrueValue,
    FalseValue,
    Activation,
    Weight,
    Bias,
    Indices,
    Parameter,
    Gradient,
    FirstMoment,
    SecondMoment,
    Velocity,
    Scalar(u32),
    Result(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationValueSpec<'a> {
    pub index: u32,
    pub role: ValueRole,
    pub source: DenseNodeId,
    pub source_result: u32,
    pub value: ValueSpec<'a>,
}

/// Canonical dense conversion semantics, determined by the source/destination
/// dtype pair. These are operation classes, not caller-selectable rounding modes.
/// All conversions are deterministic and consume no random state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DenseConversionMode {
    /// Preserve every bit, including signed zero and NaN payload/signaling bits.
    Identity,
    /// Preserve the exact value when representable. Narrowing to U8/U32 keeps
    /// the low 8/32 bits, including for negative I64, modulo 2^destination_bits.
    /// Widening unsigned integers to I64 zero-extends. No saturation or float
    /// intermediate is allowed.
    IntegerToIntegerWrapping,
    /// Round the original integer once to the destination float, nearest with
    /// ties to even. Finite overflow becomes signed infinity. An intermediate
    /// F32/F64 conversion may not discard integer bits before final rounding.
    IntegerToFloatNearestEven,
    /// Truncate toward zero, then clamp to the destination integer range. NaN
    /// becomes zero; -infinity/+infinity become the minimum/maximum integer.
    /// For unsigned destinations, every negative input becomes zero. Compare
    /// against the exact integer bounds, not a bound rounded into the source
    /// float. This is saturation, not integer-to-integer wrapping.
    FloatToIntegerTruncateSaturate,
    /// Round once from the exact source value, nearest with ties to even.
    /// Widening is exact. Finite overflow becomes signed infinity; underflow
    /// is gradual, preserving representable subnormals and the sign of a zero
    /// result. No flush-to-zero or denormals-are-zero behavior is permitted.
    /// Signed zero and infinity retain their signs. NaN remains NaN; cross-dtype
    /// casts do not specify its payload, sign, or signaling state. Intermediate
    /// rounding may not change the result, e.g. F64 -> F32 -> BF16 is not a
    /// valid implementation of an F64 -> BF16 cast.
    FloatToFloatNearestEven,
}

impl DenseConversionMode {
    pub const fn canonical(source: DType, destination: DType) -> Self {
        use DType::*;
        match (source, destination) {
            (F32, F32)
            | (F64, F64)
            | (F16, F16)
            | (BF16, BF16)
            | (U8, U8)
            | (U32, U32)
            | (I64, I64) => Self::Identity,
            (U8 | U32 | I64, U8 | U32 | I64) => Self::IntegerToIntegerWrapping,
            (U8 | U32 | I64, F32 | F64 | F16 | BF16) => Self::IntegerToFloatNearestEven,
            (F32 | F64 | F16 | BF16, U8 | U32 | I64) => Self::FloatToIntegerTruncateSaturate,
            (F32 | F64 | F16 | BF16, F32 | F64 | F16 | BF16) => Self::FloatToFloatNearestEven,
        }
    }
}

/// One canonical dense conversion. The mode is derived by [`Self::new`] and
/// checked when a recipe enters the legalization plan. It governs explicit
/// graph casts, scalar coercion, and inserted execution conversions alike.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DenseConversionContract {
    pub source: DType,
    pub destination: DType,
    pub mode: DenseConversionMode,
}

impl DenseConversionContract {
    pub const fn new(source: DType, destination: DType) -> Self {
        Self {
            source,
            destination,
            mode: DenseConversionMode::canonical(source, destination),
        }
    }

    pub fn validate(self) -> Result<(), String> {
        let expected = DenseConversionMode::canonical(self.source, self.destination);
        if self.mode != expected {
            return Err(format!(
                "legalization: noncanonical dense conversion {} -> {}: expected {:?}; actual {:?}",
                self.source, self.destination, expected, self.mode,
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionOrder {
    BackendDefined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandInterpretation {
    Direct,
    CanonicalPacked,
    /// The language coerces one 0-D float operand to the tensor operand dtype
    /// before arithmetic. This is a semantic boundary, including in native kernels.
    ScalarCoercion(DenseConversionContract),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultFormation {
    Direct,
    Cast(DenseConversionContract),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AccumulationExecution {
    pub dtype: DType,
    pub order: ReductionOrder,
}

/// A semantic result formation. Its identity survives expression inlining.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RoundingBoundary {
    pub node: DenseNodeId,
    pub result: u32,
    pub dtype: DType,
}

/// Legalization preserves the target generator and its mapping from the
/// invocation seed/nonce, semantic source, and logical element index to draws.
/// Conversion instructions consume no draws and introduce no random sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RandomStreamMapping {
    PreserveSemanticSource,
}

/// A random operation keeps its graph-assigned stream and logical sample count,
/// even when the target generates samples in F32 and then rounds to half.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RandomExecution {
    pub source: crate::RandomSource,
    pub stream: RandomStreamMapping,
    pub samples: usize,
}

/// Named algorithms allowed to compute in F64 before canonical result
/// conversion. This permission does not extend to elementwise arithmetic or
/// other composites, and never allows an F64 semantic operation to use F32.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum F64ComputeClass {
    /// Inverse, determinant, and solve may factor and solve in F64, with F64
    /// accumulation. Operands are converted on entry to the algorithm's scratch.
    LinearAlgebra,
    /// Randn and Uniform may sample in F64 using the target generator. The
    /// semantic source, invocation seed/nonce, and element mapping stay fixed.
    RandomSampling,
    /// Arange may evaluate its F64 start/step attributes and logical index in
    /// F64, then perform one canonical conversion to the declared result dtype.
    Arange,
}

/// Derived once from the semantic operation; a target may not weaken it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumericsContract {
    pub operand_interpretations: Box<[OperandInterpretation]>,
    pub result_formations: Box<[ResultFormation]>,
    pub compute_dtype: Option<DType>,
    pub permits_f32_compute: bool,
    pub f64_compute: Option<F64ComputeClass>,
    pub accumulation: Option<AccumulationExecution>,
    pub rounding_boundaries: Box<[RoundingBoundary]>,
    pub random: Option<RandomExecution>,
}

pub struct OperationDTypeSpec<'a> {
    pub node: DenseNodeId,
    pub operation: &'a NodeKind,
    pub operands: Box<[OperationValueSpec<'a>]>,
    pub results: Box<[OperationValueSpec<'a>]>,
    pub placement: &'a Device,
    pub required_numerics: NumericsContract,
}

pub struct RegionDTypeSpec<'a> {
    pub region: &'a NativeRegion,
    /// All evaluated operations, including inlined copies of shared prefixes.
    pub operations: Box<[OperationDTypeSpec<'a>]>,
    pub boundary_inputs: Box<[OperationValueSpec<'a>]>,
    pub boundary_results: Box<[OperationValueSpec<'a>]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageSupport {
    Supported,
    Unsupported(UnsupportedDType),
}

pub trait TargetDTypeCapabilities {
    fn device(&self) -> &Device;
    fn fingerprint(&self) -> &TargetFingerprint;
    fn policy_revision(&self) -> u64;
    fn storage_support(&self, value: &ValueSpec<'_>) -> StorageSupport;
    fn classify_node(&self, spec: &OperationDTypeSpec<'_>) -> DTypeDisposition;
    fn classify_region(&self, spec: &RegionDTypeSpec<'_>) -> DTypeDisposition;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DTypeDisposition {
    Native(DTypeExecution),
    Legalize(DTypeExecution),
    Unsupported(UnsupportedDType),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedDType {
    pub requirement: DTypeRequirement,
    pub reason: String,
}

impl UnsupportedDType {
    pub fn new(requirement: DTypeRequirement, reason: impl Into<String>) -> Self {
        Self {
            requirement,
            reason: reason.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DTypeRequirement {
    Storage,
    Operand(ValueRole),
    Result(ValueRole),
    Compute,
    Accumulation,
    Rounding,
    Representation,
    Layout,
    Region,
    Realization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionLayout {
    PreserveBoundary,
    DenseContiguous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackedOperandAccess {
    KernelLocal,
    TileBounded {
        max_logical_elements: usize,
        max_scratch_bytes: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperandPreparation {
    Direct,
    Convert(DenseConversionContract),
    CanonicalPacked(PackedOperandAccess),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResultCompletion {
    Direct,
    ConvertToBoundary(DenseConversionContract),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecompositionKind {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionRealization {
    DirectKernel,
    /// Conversions execute inside the kernel, in registers or declared scratch.
    /// Operand/result execution dtypes describe arithmetic values. External
    /// buffers keep the conversion's source/destination boundary dtype; this
    /// recipe does not require full materialized conversion tensors.
    KernelLocal,
    MaterializedTransforms,
    Decomposition(DecompositionKind),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperandExecution {
    pub index: u32,
    pub role: ValueRole,
    pub interpretation: OperandInterpretation,
    pub execution_dtype: DType,
    pub layout: ExecutionLayout,
    pub preparation: OperandPreparation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResultExecution {
    pub index: u32,
    pub role: ValueRole,
    pub execution_dtype: DType,
    pub layout: ExecutionLayout,
    pub completion: ResultCompletion,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationExecution {
    pub node: DenseNodeId,
    pub operands: Box<[OperandExecution]>,
    pub results: Box<[ResultExecution]>,
    pub compute_dtype: Option<DType>,
    pub accumulation: Option<AccumulationExecution>,
    pub rounding_boundaries: Box<[RoundingBoundary]>,
    pub random: Option<RandomExecution>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DTypeExecution {
    pub operations: Box<[OperationExecution]>,
    pub realization: ExecutionRealization,
}

impl DTypeExecution {
    pub fn operation(&self, node: DenseNodeId) -> Option<&OperationExecution> {
        self.operations
            .iter()
            .find(|operation| operation.node == node)
    }

    /// Changes only approved half arithmetic to F32. Validation still checks the
    /// completed recipe against every semantic operation before accepting it.
    pub fn promote_half(mut self, realization: ExecutionRealization) -> Self {
        self.realization = realization;
        for operation in &mut self.operations {
            if matches!(operation.compute_dtype, Some(DType::F16 | DType::BF16)) {
                operation.compute_dtype = Some(DType::F32);
                for operand in &mut operation.operands {
                    if matches!(operand.execution_dtype, DType::F16 | DType::BF16) {
                        operand.preparation = OperandPreparation::Convert(
                            DenseConversionContract::new(operand.execution_dtype, DType::F32),
                        );
                        operand.execution_dtype = DType::F32;
                    }
                }
                for result in &mut operation.results {
                    if matches!(result.execution_dtype, DType::F16 | DType::BF16) {
                        result.completion = ResultCompletion::ConvertToBoundary(
                            DenseConversionContract::new(DType::F32, result.execution_dtype),
                        );
                        result.execution_dtype = DType::F32;
                    }
                }
            }
        }
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutableDTypePlan {
    Native(Arc<DTypeExecution>),
    Legalize(Arc<DTypeExecution>),
}

impl ExecutableDTypePlan {
    pub fn execution(&self) -> &DTypeExecution {
        match self {
            Self::Native(execution) | Self::Legalize(execution) => execution,
        }
    }
    pub fn is_legalized(&self) -> bool {
        matches!(self, Self::Legalize(_))
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LegalizationWork {
    pub capability_queries: usize,
    pub native_lowering_units: usize,
    pub legalized_lowering_units: usize,
    pub kernel_local_legalizations: usize,
    pub materialized_conversions: usize,
    pub materialized_conversion_bytes: usize,
    pub decompositions: usize,
    pub rejected_region_candidates: usize,
    pub unsupported_independent_units: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegalizedUnit {
    unit: LoweringUnit,
    disposition: ExecutableDTypePlan,
}

impl LegalizedUnit {
    pub fn unit(&self) -> LoweringUnit {
        self.unit
    }
    pub fn disposition(&self) -> &ExecutableDTypePlan {
        &self.disposition
    }
    pub fn execution(&self) -> &DTypeExecution {
        self.disposition.execution()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegalizationPlan {
    semantic_ids: Box<[u64]>,
    target: TargetFingerprint,
    policy_revision: u64,
    units: Box<[LegalizedUnit]>,
    work: LegalizationWork,
}

impl LegalizationPlan {
    pub fn units(&self) -> &[LegalizedUnit] {
        &self.units
    }
    pub fn target(&self) -> &TargetFingerprint {
        &self.target
    }
    pub fn policy_revision(&self) -> u64 {
        self.policy_revision
    }
    pub fn work(&self) -> &LegalizationWork {
        &self.work
    }

    pub(crate) fn build<C: TargetDTypeCapabilities>(
        index: &GraphIndex,
        optimization: &OptimizationPlan,
        target: &C,
    ) -> Result<Self, String> {
        if index.roots.iter().any(|root| {
            index.order[root.index()].storage.representation != StorageRepresentation::Dense
        }) {
            return Err("legalization: packed program outputs are not supported".into());
        }
        let mut work = LegalizationWork {
            capability_queries: optimization.work.capability_queries,
            rejected_region_candidates: optimization.work.rejected_region_candidates,
            ..LegalizationWork::default()
        };
        let mut units = Vec::with_capacity(optimization.lowering_order.len());
        for &unit in &optimization.lowering_order {
            let disposition = match unit {
                LoweringUnit::Region(region) => {
                    let spec = RegionDTypeSpec::new(index, &optimization.regions[region.index()])?;
                    for value in spec
                        .boundary_inputs
                        .iter()
                        .chain(spec.boundary_results.iter())
                    {
                        work.capability_queries += 1;
                        if let StorageSupport::Unsupported(error) =
                            target.storage_support(&value.value)
                        {
                            return Err(format!(
                                "compile: region {region} boundary {:?}={} on {} requires {:?}: {}",
                                value.role,
                                value.value.semantic_dtype,
                                target.device(),
                                error.requirement,
                                error.reason
                            ));
                        }
                    }
                    optimization
                        .region_dtype_plans
                        .get(region.index())
                        .cloned()
                        .ok_or_else(|| {
                            format!(
                                "legalization: region {region} has no classified execution plan"
                            )
                        })?
                }
                LoweringUnit::Node(node) => {
                    let spec = OperationDTypeSpec::new(index, node)?;
                    if !spec.placement.same_device(target.device()) {
                        return Err(format!(
                            "legalization: node {node} uses {}, target uses {}",
                            spec.placement,
                            target.device()
                        ));
                    }
                    for value in spec.operands.iter().chain(spec.results.iter()) {
                        work.capability_queries += 1;
                        if let StorageSupport::Unsupported(error) =
                            target.storage_support(&value.value)
                        {
                            return Err(unsupported_message(node, &spec, target, &error));
                        }
                    }
                    work.capability_queries += 1;
                    match target.classify_node(&spec) {
                        DTypeDisposition::Unsupported(error) => {
                            return Err(unsupported_message(node, &spec, target, &error))
                        }
                        disposition => {
                            validate_disposition(std::slice::from_ref(&spec), disposition)?
                        }
                    }
                }
            };
            match &disposition {
                ExecutableDTypePlan::Native(_) => work.native_lowering_units += 1,
                ExecutableDTypePlan::Legalize(execution) => {
                    work.legalized_lowering_units += 1;
                    match execution.realization {
                        ExecutionRealization::KernelLocal => work.kernel_local_legalizations += 1,
                        ExecutionRealization::Decomposition(_) => work.decompositions += 1,
                        _ => {}
                    }
                }
            }
            units.push(LegalizedUnit { unit, disposition });
        }
        let plan = Self {
            semantic_ids: index.order.iter().map(|node| node.id).collect(),
            target: target.fingerprint().clone(),
            policy_revision: target.policy_revision(),
            units: units.into_boxed_slice(),
            work,
        };
        plan.validate(index, optimization, target)?;
        Ok(plan)
    }

    pub fn validate<C: TargetDTypeCapabilities>(
        &self,
        index: &GraphIndex,
        optimization: &OptimizationPlan,
        target: &C,
    ) -> Result<(), String> {
        self.target.validate(target.device())?;
        if self.semantic_ids.len() != index.order.len()
            || self
                .semantic_ids
                .iter()
                .zip(index.order.iter())
                .any(|(id, node)| *id != node.id)
        {
            return Err(
                "legalization: plan belongs to a different semantic graph generation".into(),
            );
        }
        if index
            .order
            .iter()
            .any(|node| !node.device.same_device(target.device()))
        {
            return Err("legalization: graph and target placement mismatch".into());
        }
        if &self.target != target.fingerprint() || self.policy_revision != target.policy_revision()
        {
            return Err("legalization: target fingerprint or policy revision mismatch".into());
        }
        if self.units.len() != optimization.lowering_order.len() {
            return Err("legalization: lowering unit count mismatch".into());
        }
        for (entry, expected) in self.units.iter().zip(optimization.lowering_order.iter()) {
            if entry.unit != *expected {
                return Err("legalization: lowering unit order mismatch".into());
            }
            match entry.unit {
                LoweringUnit::Node(node) => validate_execution(
                    std::slice::from_ref(&OperationDTypeSpec::new(index, node)?),
                    &entry.disposition,
                )?,
                LoweringUnit::Region(region) => {
                    let spec = RegionDTypeSpec::new(index, &optimization.regions[region.index()])?;
                    validate_execution(&spec.operations, &entry.disposition)?;
                }
            }
        }
        Ok(())
    }
}

fn unsupported_message<C: TargetDTypeCapabilities>(
    node: DenseNodeId,
    spec: &OperationDTypeSpec<'_>,
    target: &C,
    error: &UnsupportedDType,
) -> String {
    let operands = spec
        .operands
        .iter()
        .map(|value| {
            format!(
                "{:?}={}:{:?}",
                value.role, value.value.semantic_dtype, value.value.storage.representation
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("compile: node {node} {} is unsupported on {} ({:?} {}, policy {}): {operands}; results={:?}, compute={:?}, accumulation={:?}; requirement {:?}: {}", operation_name(spec.operation), target.device(), target.fingerprint().backend, target.fingerprint().architecture, target.policy_revision(), spec.results.iter().map(|value| (value.role, value.value.semantic_dtype)).collect::<Vec<_>>(), spec.required_numerics.compute_dtype, spec.required_numerics.accumulation, error.requirement, error.reason)
}

fn value_spec(node: &Node) -> ValueSpec<'_> {
    node.value_spec()
}

impl<'a> OperationDTypeSpec<'a> {
    pub fn new(index: &'a GraphIndex, node: DenseNodeId) -> Result<Self, String> {
        let semantic = index
            .node(node)
            .ok_or_else(|| format!("legalization: missing semantic node {node}"))?;
        index
            .value_spec(node)
            .ok_or_else(|| "legalization: value storage table is incomplete".to_string())?
            .validate()?;
        let operands = index.children[node.index()]
            .iter()
            .enumerate()
            .map(|(operand, &source)| OperationValueSpec {
                index: operand as u32,
                role: operand_role(&semantic.kind, operand),
                source,
                source_result: selected_result(&semantic.kind).unwrap_or(0),
                value: if selected_result(&semantic.kind).is_some() {
                    value_spec(semantic)
                } else {
                    index
                        .value_spec(source)
                        .expect("GraphIndex contains operand storage")
                },
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let result_nodes: Vec<&Node> = match &semantic.kind {
            NodeKind::AdamWStep { param, m, v, .. } => vec![param, m, v],
            NodeKind::SgdStep {
                param, velocity, ..
            } => vec![param, velocity],
            NodeKind::SdpaBackward { q, k, v, .. } => vec![q, k, v],
            NodeKind::KdaBackward {
                q,
                k,
                v,
                log_decay,
                beta,
                ..
            } => vec![q, k, v, log_decay, beta],
            NodeKind::LayerNormBackward { x, weight, .. } => vec![x, weight, weight],
            NodeKind::ChunkedHeadCeBackward {
                x, weight, bias, ..
            } => vec![x, weight, bias],
            _ => vec![semantic],
        };
        let results = result_nodes
            .into_iter()
            .enumerate()
            .map(|(result, value)| {
                let mut value = value_spec(value);
                value.storage = index.value_storage[node.index()].as_spec();
                OperationValueSpec {
                    index: result as u32,
                    role: ValueRole::Result(result as u32),
                    source: node,
                    source_result: result as u32,
                    value,
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let mut required_numerics = numerics(node, semantic, &operands);
        if matches!(
            semantic.kind,
            NodeKind::Randn { .. } | NodeKind::Uniform { .. }
        ) {
            let source = index
                .random_sources
                .get(node.index())
                .copied()
                .flatten()
                .and_then(|source| index.random_source_order.get(source.index()))
                .copied()
                .filter(|source| source.node == node && source.provenance == semantic.id)
                .ok_or_else(|| {
                    format!("legalization: random node {node} has no matching source metadata")
                })?;
            let samples = semantic
                .shape
                .iter()
                .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
                .ok_or_else(|| {
                    format!("legalization: random node {node} sample count overflows")
                })?;
            required_numerics.random = Some(RandomExecution {
                source,
                stream: RandomStreamMapping::PreserveSemanticSource,
                samples,
            });
        }
        required_numerics.result_formations = results
            .iter()
            .map(|_| {
                if let NodeKind::Cast { a, dtype } = &semantic.kind {
                    ResultFormation::Cast(DenseConversionContract::new(a.dtype, *dtype))
                } else {
                    ResultFormation::Direct
                }
            })
            .collect();
        if required_numerics.compute_dtype.is_some() {
            required_numerics.rounding_boundaries = results
                .iter()
                .filter(|result| result.value.semantic_dtype.is_float())
                .map(|result| RoundingBoundary {
                    node,
                    result: result.index,
                    dtype: result.value.semantic_dtype,
                })
                .collect();
        }
        Ok(Self {
            node,
            operation: &semantic.kind,
            operands,
            results,
            placement: &semantic.device,
            required_numerics,
        })
    }

    pub fn native_execution(&self) -> DTypeExecution {
        DTypeExecution {
            operations: vec![self.direct_operation()].into_boxed_slice(),
            realization: ExecutionRealization::DirectKernel,
        }
    }

    /// Describes the approved F64 algorithm for linalg, random sampling, or
    /// arange. Narrower boundaries use kernel-local canonical conversions and
    /// retain semantic layouts, rounding boundaries, and random identities.
    /// Classify the returned DirectKernel recipe as Native, otherwise Legalize.
    /// Selecting this recipe asserts that the target implements F64 arithmetic.
    pub fn f64_execution(&self) -> Result<DTypeExecution, String> {
        if self.required_numerics.f64_compute.is_none() {
            return Err(format!(
                "legalization: node {} {} on {} has no approved f64 compute algorithm",
                self.node,
                operation_name(self.operation),
                self.placement,
            ));
        }
        let mut operation = self.direct_operation();
        operation.compute_dtype = Some(DType::F64);
        if let Some(accumulation) = &mut operation.accumulation {
            accumulation.dtype = DType::F64;
        }
        let mut realization = ExecutionRealization::DirectKernel;
        for operand in &mut operation.operands {
            if operand.execution_dtype != DType::F64 {
                operand.preparation = OperandPreparation::Convert(DenseConversionContract::new(
                    operand.execution_dtype,
                    DType::F64,
                ));
                operand.execution_dtype = DType::F64;
                realization = ExecutionRealization::KernelLocal;
            }
        }
        for result in &mut operation.results {
            if result.execution_dtype != DType::F64 {
                result.completion = ResultCompletion::ConvertToBoundary(
                    DenseConversionContract::new(DType::F64, result.execution_dtype),
                );
                result.execution_dtype = DType::F64;
                realization = ExecutionRealization::KernelLocal;
            }
        }
        Ok(DTypeExecution {
            operations: vec![operation].into_boxed_slice(),
            realization,
        })
    }

    fn direct_operation(&self) -> OperationExecution {
        OperationExecution {
            node: self.node,
            operands: self
                .operands
                .iter()
                .map(|operand| OperandExecution {
                    index: operand.index,
                    role: operand.role,
                    interpretation: self.required_numerics.operand_interpretations
                        [operand.index as usize],
                    execution_dtype: match self.required_numerics.operand_interpretations
                        [operand.index as usize]
                    {
                        OperandInterpretation::ScalarCoercion(conversion) => conversion.destination,
                        _ => operand.value.semantic_dtype,
                    },
                    layout: ExecutionLayout::PreserveBoundary,
                    preparation: match operand.value.storage.representation {
                        StorageRepresentation::Dense => OperandPreparation::Direct,
                        StorageRepresentation::Packed(_) => {
                            OperandPreparation::CanonicalPacked(PackedOperandAccess::KernelLocal)
                        }
                    },
                })
                .collect(),
            results: self
                .results
                .iter()
                .map(|result| ResultExecution {
                    index: result.index,
                    role: result.role,
                    execution_dtype: result.value.semantic_dtype,
                    layout: ExecutionLayout::PreserveBoundary,
                    completion: ResultCompletion::Direct,
                })
                .collect(),
            compute_dtype: self.required_numerics.compute_dtype,
            accumulation: self.required_numerics.accumulation,
            rounding_boundaries: self.required_numerics.rounding_boundaries.clone(),
            random: self.required_numerics.random,
        }
    }
}

impl<'a> RegionDTypeSpec<'a> {
    pub fn new(index: &'a GraphIndex, region: &'a NativeRegion) -> Result<Self, String> {
        let mut nodes = region.nodes().to_vec();
        // A root Linear can remain independently materialized while an epilogue
        // recomputes it. Ownership alone does not describe evaluated operations.
        if let NativeRegion::LinearResidual(region) = region {
            nodes.extend(
                index.children[region.output.index()]
                    .iter()
                    .copied()
                    .filter(|node| {
                        matches!(index.order[node.index()].kind, NodeKind::Linear { .. })
                    }),
            );
        }
        for expression in region_expressions(region) {
            nodes.extend(expression.semantic_nodes());
        }
        nodes.sort_unstable();
        nodes.dedup();
        let operations = nodes
            .into_iter()
            .map(|node| OperationDTypeSpec::new(index, node))
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice();
        let boundary_inputs = region
            .inputs()
            .iter()
            .enumerate()
            .map(|(position, &node)| OperationValueSpec {
                index: position as u32,
                role: ValueRole::Input(position as u32),
                source: node,
                source_result: 0,
                value: index
                    .value_spec(node)
                    .expect("GraphIndex contains boundary storage"),
            })
            .collect();
        let boundary_results = region
            .semantic_outputs()
            .into_iter()
            .map(|output| OperationValueSpec {
                index: output.index,
                role: ValueRole::Result(output.index),
                source: output.semantic_node,
                source_result: 0,
                value: index
                    .value_spec(output.semantic_node)
                    .expect("GraphIndex contains result storage"),
            })
            .collect();
        Ok(Self {
            region,
            operations,
            boundary_inputs,
            boundary_results,
        })
    }
    pub fn native_execution(&self) -> DTypeExecution {
        DTypeExecution {
            operations: self
                .operations
                .iter()
                .map(OperationDTypeSpec::direct_operation)
                .collect(),
            realization: ExecutionRealization::DirectKernel,
        }
    }
}

pub(crate) fn validate_disposition(
    specs: &[OperationDTypeSpec<'_>],
    disposition: DTypeDisposition,
) -> Result<ExecutableDTypePlan, String> {
    let plan = match disposition {
        DTypeDisposition::Native(execution) => ExecutableDTypePlan::Native(Arc::new(execution)),
        DTypeDisposition::Legalize(execution) => ExecutableDTypePlan::Legalize(Arc::new(execution)),
        DTypeDisposition::Unsupported(error) => {
            return Err(format!(
                "legalization: {:?}: {}",
                error.requirement, error.reason
            ))
        }
    };
    validate_execution(specs, &plan)?;
    Ok(plan)
}

pub(crate) fn validate_execution(
    specs: &[OperationDTypeSpec<'_>],
    plan: &ExecutableDTypePlan,
) -> Result<(), String> {
    let execution = plan.execution();
    let native = !plan.is_legalized();
    if native != matches!(execution.realization, ExecutionRealization::DirectKernel) {
        return Err("legalization: disposition does not match execution realization".into());
    }
    if execution.operations.len() != specs.len() {
        return Err("legalization: operation coverage mismatch".into());
    }
    let mut transforms = 0;
    for (spec, operation) in specs.iter().zip(execution.operations.iter()) {
        if spec.node != operation.node
            || spec.operands.len() != operation.operands.len()
            || spec.results.len() != operation.results.len()
        {
            return Err(format!(
                "legalization: node {} operand/result coverage mismatch",
                spec.node
            ));
        }
        let required = &spec.required_numerics;
        for formation in &required.result_formations {
            if let ResultFormation::Cast(conversion) = formation {
                conversion.validate()?;
            }
        }
        if operation.random != required.random {
            return Err(format!(
                "legalization: node {} {} on {} random stream/counter contract mismatch: expected {:?}; actual {:?}",
                spec.node, operation_name(spec.operation), spec.placement, required.random, operation.random,
            ));
        }
        let f64_algorithm =
            required.f64_compute.is_some() && operation.compute_dtype == Some(DType::F64);
        if operation.compute_dtype != required.compute_dtype
            && !(required.permits_f32_compute && operation.compute_dtype == Some(DType::F32))
            && !f64_algorithm
        {
            return Err(format!(
                "legalization: node {} {} on {} compute dtype violates semantic contract: expected {}{}{}; actual {}",
                spec.node,
                operation_name(spec.operation),
                spec.placement,
                required.compute_dtype.map_or_else(|| "none".into(), |dtype| dtype.to_string()),
                if required.permits_f32_compute { " or f32" } else { "" },
                if required.f64_compute.is_some() && required.compute_dtype != Some(DType::F64) { " or f64" } else { "" },
                operation.compute_dtype.map_or_else(|| "none".into(), |dtype| dtype.to_string()),
            ));
        }
        if f64_algorithm
            && operation.compute_dtype != required.compute_dtype
            && execution.realization != ExecutionRealization::KernelLocal
        {
            return Err(format!(
                "legalization: node {} {} on {} widened f64 algorithm requires kernel-local conversions",
                spec.node, operation_name(spec.operation), spec.placement,
            ));
        }
        let expected_accumulation = required.accumulation.map(|mut accumulation| {
            if f64_algorithm {
                accumulation.dtype = DType::F64;
            }
            accumulation
        });
        if operation.accumulation != expected_accumulation {
            return Err(format!(
                "legalization: node {} {} on {} accumulation contract mismatch: expected {:?}; actual {:?}",
                spec.node,
                operation_name(spec.operation),
                spec.placement,
                expected_accumulation,
                operation.accumulation,
            ));
        }
        if operation.rounding_boundaries != required.rounding_boundaries {
            return Err(format!(
                "legalization: node {} semantic rounding boundaries mismatch",
                spec.node
            ));
        }
        for (value, operand) in spec.operands.iter().zip(operation.operands.iter()) {
            if let OperandInterpretation::ScalarCoercion(conversion) = operand.interpretation {
                conversion.validate()?;
            }
            if let OperandPreparation::Convert(conversion) = operand.preparation {
                conversion.validate()?;
            }
            value.value.validate()?;
            if value.index != operand.index || value.role != operand.role {
                return Err("legalization: operand role mismatch".into());
            }
            if native && operand.layout != ExecutionLayout::PreserveBoundary {
                return Err("legalization: native operand changes boundary layout".into());
            }
            if operand.layout != ExecutionLayout::PreserveBoundary
                && !matches!(operand.preparation, OperandPreparation::Convert(_))
            {
                return Err("legalization: operand layout change has no declared transform".into());
            }
            let interpretation = required.operand_interpretations[value.index as usize];
            if operand.interpretation != interpretation {
                return Err(
                    "legalization: operand interpretation or scalar coercion mismatch".into(),
                );
            }
            let interpreted_dtype = match interpretation {
                OperandInterpretation::ScalarCoercion(conversion) => conversion.destination,
                _ => value.value.semantic_dtype,
            };
            if f64_algorithm && operand.execution_dtype != DType::F64 {
                return Err(format!(
                    "legalization: node {} {} on {} f64 algorithm operand {:?}: expected f64; actual {}",
                    spec.node, operation_name(spec.operation), spec.placement, operand.role, operand.execution_dtype,
                ));
            }
            match (value.value.storage.representation, operand.preparation) {
                (StorageRepresentation::Dense, OperandPreparation::Direct)
                    if operand.execution_dtype == interpreted_dtype => {}
                (StorageRepresentation::Dense, OperandPreparation::Convert(conversion))
                    if conversion.source == interpreted_dtype
                        && conversion.destination == operand.execution_dtype
                        && conversion.source != conversion.destination
                        && operation.compute_dtype == Some(conversion.destination) =>
                {
                    let half_promotion = required.permits_f32_compute
                        && matches!(conversion.source, DType::F16 | DType::BF16)
                        && conversion.destination == DType::F32;
                    let f64_promotion = f64_algorithm
                        && conversion.destination == DType::F64
                        && execution.realization == ExecutionRealization::KernelLocal;
                    if !half_promotion && !f64_promotion {
                        return Err("legalization: unapproved operand conversion".into());
                    }
                    transforms += 1;
                }
                (StorageRepresentation::Packed(_), OperandPreparation::CanonicalPacked(access))
                    if operand.role == ValueRole::Weight
                        && operand.execution_dtype == DType::F32
                        && value.value.semantic_dtype == DType::F32
                        && matches!(
                            spec.operation,
                            NodeKind::QuantizedLinear { .. } | NodeKind::QuantizedEmbedding { .. }
                        ) =>
                {
                    if let PackedOperandAccess::TileBounded {
                        max_logical_elements,
                        max_scratch_bytes,
                    } = access
                    {
                        if max_logical_elements == 0 || max_scratch_bytes == 0 {
                            return Err("legalization: packed tile has no static bound".into());
                        }
                    }
                }
                _ => {
                    return Err(
                        "legalization: operand representation or conversion contract mismatch"
                            .into(),
                    )
                }
            }
        }
        for (value, result) in spec.results.iter().zip(operation.results.iter()) {
            if let ResultCompletion::ConvertToBoundary(conversion) = result.completion {
                conversion.validate()?;
            }
            value.value.validate()?;
            if value.index != result.index || value.role != result.role {
                return Err("legalization: result role mismatch".into());
            }
            // Packed values may only enter through validated bindings.
            let binding = matches!(spec.operation, NodeKind::Input { .. } | NodeKind::Leaf(_));
            if value.value.storage.representation != StorageRepresentation::Dense && !binding {
                return Err("legalization: packed operation results are not supported".into());
            }
            if native && result.layout != ExecutionLayout::PreserveBoundary {
                return Err("legalization: native result changes boundary layout".into());
            }
            if f64_algorithm && result.execution_dtype != DType::F64 {
                return Err(format!(
                    "legalization: node {} {} on {} f64 algorithm result {:?}: expected f64; actual {}",
                    spec.node, operation_name(spec.operation), spec.placement, result.role, result.execution_dtype,
                ));
            }
            match result.completion {
                ResultCompletion::Direct
                    if result.execution_dtype == value.value.semantic_dtype => {}
                ResultCompletion::ConvertToBoundary(conversion)
                    if conversion.source == result.execution_dtype
                        && conversion.destination == value.value.semantic_dtype
                        && conversion.source != conversion.destination
                        && (conversion.source == DType::F32
                            && matches!(conversion.destination, DType::F16 | DType::BF16)
                            && required.permits_f32_compute
                            || conversion.source == DType::F64
                                && f64_algorithm
                                && execution.realization == ExecutionRealization::KernelLocal) =>
                {
                    transforms += 1
                }
                _ => {
                    return Err(
                        "legalization: result does not restore semantic boundary dtype".into(),
                    )
                }
            }
        }
    }
    if native && transforms != 0 {
        return Err("legalization: native plan contains inserted conversions".into());
    }
    if !native && transforms == 0 {
        return Err("legalization: legalized plan contains no transformation".into());
    }
    Ok(())
}

fn selected_result(operation: &NodeKind) -> Option<u32> {
    match operation {
        NodeKind::AdamWOut { index, .. }
        | NodeKind::SgdOut { index, .. }
        | NodeKind::SdpaBackwardOut { index, .. }
        | NodeKind::KdaBackwardOut { index, .. }
        | NodeKind::LayerNormBackwardOut { index, .. }
        | NodeKind::ChunkedHeadCeBackwardOut { index, .. } => Some(u32::from(*index)),
        _ => None,
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
        | NodeKind::Lt { .. }
        | NodeKind::Gt { .. }
        | NodeKind::Le { .. }
        | NodeKind::Ge { .. }
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

/// Required coercion for the intentional mixed scalar/tensor float rule.
/// Tensor-tensor arithmetic has no implicit promotion.
pub fn scalar_coercion(operation: &NodeKind, operand: usize) -> Option<DenseConversionContract> {
    let (a, b) = match operation {
        NodeKind::Add { a, b }
        | NodeKind::Sub { a, b }
        | NodeKind::Mul { a, b }
        | NodeKind::Div { a, b }
        | NodeKind::Maximum { a, b }
        | NodeKind::Minimum { a, b }
        | NodeKind::Eq { a, b }
        | NodeKind::Lt { a, b }
        | NodeKind::Gt { a, b }
        | NodeKind::Le { a, b }
        | NodeKind::Ge { a, b } => (a, b),
        _ => return None,
    };
    if a.dtype == b.dtype
        || !a.dtype.is_float()
        || !b.dtype.is_float()
        || a.shape.is_empty() == b.shape.is_empty()
    {
        return None;
    }
    match operand {
        0 if a.shape.is_empty() => Some(DenseConversionContract::new(a.dtype, b.dtype)),
        1 if b.shape.is_empty() => Some(DenseConversionContract::new(b.dtype, a.dtype)),
        _ => None,
    }
}

fn numerics(
    node: DenseNodeId,
    semantic: &Node,
    operands: &[OperationValueSpec<'_>],
) -> NumericsContract {
    let no_compute = selected_result(&semantic.kind).is_some()
        || matches!(
            semantic.kind,
            NodeKind::Leaf(_)
                | NodeKind::Input { .. }
                | NodeKind::ScalarInput { .. }
                | NodeKind::FromBytes { .. }
                | NodeKind::Zeros { .. }
                | NodeKind::Ones { .. }
                | NodeKind::Full { .. }
                | NodeKind::Cast { .. }
                | NodeKind::Reshape { .. }
                | NodeKind::Permute { .. }
                | NodeKind::Slice { .. }
                | NodeKind::BroadcastTo { .. }
                | NodeKind::Concat { .. }
                | NodeKind::StopGradient { .. }
                | NodeKind::Checkpoint { .. }
                | NodeKind::Expose { .. }
        );
    let compute_dtype = if no_compute {
        None
    } else if matches!(
        semantic.kind,
        NodeKind::Eq { .. }
            | NodeKind::Gt { .. }
            | NodeKind::Lt { .. }
            | NodeKind::Ge { .. }
            | NodeKind::Le { .. }
            | NodeKind::Argmax { .. }
            | NodeKind::Argmin { .. }
    ) {
        operands.first().map(|operand| {
            scalar_coercion(&semantic.kind, 0).map_or(operand.value.semantic_dtype, |conversion| {
                conversion.destination
            })
        })
    } else {
        Some(semantic.dtype)
    };
    let permits_f32_compute = matches!(compute_dtype, Some(DType::F16 | DType::BF16))
        && matches!(
            semantic.kind,
            NodeKind::Add { .. }
                | NodeKind::Sub { .. }
                | NodeKind::Mul { .. }
                | NodeKind::Div { .. }
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
                | NodeKind::Where { .. }
                | NodeKind::Eq { .. }
                | NodeKind::Gt { .. }
                | NodeKind::Lt { .. }
                | NodeKind::Ge { .. }
                | NodeKind::Le { .. }
                | NodeKind::Sum { .. }
                | NodeKind::Mean { .. }
                | NodeKind::Max { .. }
                | NodeKind::Min { .. }
                | NodeKind::Prod { .. }
                | NodeKind::Matmul { .. }
                | NodeKind::Linear { .. }
                | NodeKind::Sdpa { .. }
                | NodeKind::SdpaBackward { .. }
                // These composites form one semantic result per output. Their
                // statistics, nonlinear arithmetic, and rotation run in F32
                // before restoring half outputs. They do not permit widening
                // a chain of separately rounded semantic operations.
                | NodeKind::LayerNorm { .. }
                | NodeKind::LayerNormBackward { .. }
                | NodeKind::RmsNorm { .. }
                | NodeKind::CrossEntropy { .. }
                | NodeKind::CrossEntropyBackward { .. }
                | NodeKind::RotaryEmbedding { .. }
                | NodeKind::RotaryEmbeddingBackward { .. }
                | NodeKind::Conv1d { .. }
                | NodeKind::Conv2d { .. }
                | NodeKind::ConvTranspose1d { .. }
                | NodeKind::ConvTranspose2d { .. }
                | NodeKind::Conv1dBackwardW { .. }
                | NodeKind::Conv2dBackwardW { .. }
                | NodeKind::ShortConv1d { .. }
                | NodeKind::ShortConv1dBackwardX { .. }
                | NodeKind::ShortConv1dBackwardW { .. }
                | NodeKind::ConvState { .. }
                | NodeKind::KdaChunk { .. }
                | NodeKind::KdaRecurrence { .. }
                | NodeKind::KdaBackward { .. }
                | NodeKind::AdamWStep { .. }
                | NodeKind::SgdStep { .. }
                | NodeKind::Cumsum { .. }
                // Random sampling retains its target generator, source, and
                // element-counter mapping. Only sample arithmetic and the
                // final half conversion may use this F32 realization.
                | NodeKind::Randn { .. }
                | NodeKind::Uniform { .. }
        );
    let f64_compute = match semantic.kind {
        NodeKind::Inverse { .. } | NodeKind::Det { .. } | NodeKind::Solve { .. } => {
            Some(F64ComputeClass::LinearAlgebra)
        }
        NodeKind::Randn { .. } | NodeKind::Uniform { .. } => Some(F64ComputeClass::RandomSampling),
        NodeKind::Arange { .. } => Some(F64ComputeClass::Arange),
        _ => None,
    };
    let accumulation = if matches!(
        semantic.kind,
        NodeKind::Sum { .. }
            | NodeKind::Mean { .. }
            | NodeKind::Prod { .. }
            | NodeKind::Cumsum { .. }
            | NodeKind::Matmul { .. }
            | NodeKind::Linear { .. }
            | NodeKind::QuantizedLinear { .. }
            | NodeKind::Sdpa { .. }
            | NodeKind::SdpaBackward { .. }
            | NodeKind::KvAttention { .. }
            | NodeKind::LayerNorm { .. }
            | NodeKind::LayerNormBackward { .. }
            | NodeKind::RmsNorm { .. }
            | NodeKind::CrossEntropy { .. }
            | NodeKind::CrossEntropyBackward { .. }
            | NodeKind::ChunkedHeadCe { .. }
            | NodeKind::ChunkedHeadCeBackward { .. }
            | NodeKind::Conv1d { .. }
            | NodeKind::Conv2d { .. }
            | NodeKind::ConvTranspose1d { .. }
            | NodeKind::ConvTranspose2d { .. }
            | NodeKind::Conv1dBackwardW { .. }
            | NodeKind::Conv2dBackwardW { .. }
            | NodeKind::ShortConv1d { .. }
            | NodeKind::ShortConv1dBackwardX { .. }
            | NodeKind::ShortConv1dBackwardW { .. }
            | NodeKind::ConvState { .. }
            | NodeKind::KdaChunk { .. }
            | NodeKind::KdaRecurrence { .. }
            | NodeKind::KdaBackward { .. }
            | NodeKind::Inverse { .. }
            | NodeKind::Det { .. }
            | NodeKind::Solve { .. }
    ) {
        Some(AccumulationExecution {
            dtype: if matches!(semantic.dtype, DType::F16 | DType::BF16) {
                DType::F32
            } else {
                semantic.dtype
            },
            order: ReductionOrder::BackendDefined,
        })
    } else {
        None
    };
    let rounding_boundaries = if compute_dtype.is_some() && semantic.dtype.is_float() {
        vec![RoundingBoundary {
            node,
            result: 0,
            dtype: semantic.dtype,
        }]
        .into_boxed_slice()
    } else {
        Box::new([])
    };
    NumericsContract {
        operand_interpretations: operands
            .iter()
            .map(|operand| match operand.value.storage.representation {
                StorageRepresentation::Dense => {
                    scalar_coercion(&semantic.kind, operand.index as usize).map_or(
                        OperandInterpretation::Direct,
                        OperandInterpretation::ScalarCoercion,
                    )
                }
                StorageRepresentation::Packed(_) => OperandInterpretation::CanonicalPacked,
            })
            .collect(),
        result_formations: Box::new([]),
        compute_dtype,
        permits_f32_compute,
        f64_compute,
        accumulation,
        rounding_boundaries,
        random: None,
    }
}

/// Stable diagnostic operation name without recursively formatting NodeKind.
pub fn operation_name(operation: &NodeKind) -> &'static str {
    match operation {
        NodeKind::Leaf(_) => "leaf",
        NodeKind::Input { .. } => "input",
        NodeKind::ScalarInput { .. } => "scalar_input",
        NodeKind::FromBytes { .. } => "from_bytes",
        NodeKind::Zeros { .. } => "zeros",
        NodeKind::Ones { .. } => "ones",
        NodeKind::Full { .. } => "full",
        NodeKind::Randn { .. } => "randn",
        NodeKind::Uniform { .. } => "uniform",
        NodeKind::Arange { .. } => "arange",
        NodeKind::Eye { .. } => "eye",
        NodeKind::Add { .. } => "add",
        NodeKind::Sub { .. } => "sub",
        NodeKind::Mul { .. } => "mul",
        NodeKind::Div { .. } => "div",
        NodeKind::Eq { .. } => "eq",
        NodeKind::Gt { .. } => "gt",
        NodeKind::Lt { .. } => "lt",
        NodeKind::Ge { .. } => "ge",
        NodeKind::Le { .. } => "le",
        NodeKind::Maximum { .. } => "maximum",
        NodeKind::Minimum { .. } => "minimum",
        NodeKind::Neg { .. } => "neg",
        NodeKind::Abs { .. } => "abs",
        NodeKind::Sqrt { .. } => "sqrt",
        NodeKind::Exp { .. } => "exp",
        NodeKind::Log { .. } => "log",
        NodeKind::Sin { .. } => "sin",
        NodeKind::Cos { .. } => "cos",
        NodeKind::Tanh { .. } => "tanh",
        NodeKind::Relu { .. } => "relu",
        NodeKind::Erf { .. } => "erf",
        NodeKind::Gelu { .. } => "gelu",
        NodeKind::Floor { .. } => "floor",
        NodeKind::Ceil { .. } => "ceil",
        NodeKind::Round { .. } => "round",
        NodeKind::Sign { .. } => "sign",
        NodeKind::Where { .. } => "where",
        NodeKind::Pow { .. } => "pow",
        NodeKind::Cast { .. } => "cast",
        NodeKind::Sum { .. } => "sum",
        NodeKind::Mean { .. } => "mean",
        NodeKind::Max { .. } => "max",
        NodeKind::Min { .. } => "min",
        NodeKind::Prod { .. } => "prod",
        NodeKind::Argmax { .. } => "argmax",
        NodeKind::Argmin { .. } => "argmin",
        NodeKind::Cumsum { .. } => "cumsum",
        NodeKind::IndexSelect { .. } => "index_select",
        NodeKind::ScatterAdd { .. } => "scatter_add",
        NodeKind::Gather { .. } => "gather",
        NodeKind::CrossEntropy { .. } => "cross_entropy",
        NodeKind::CrossEntropyBackward { .. } => "cross_entropy_backward",
        NodeKind::Sdpa { .. } => "sdpa",
        NodeKind::SdpaBackward { .. } => "sdpa_backward",
        NodeKind::SdpaBackwardOut { .. } => "sdpa_backward_out",
        NodeKind::PositionEmbedding { .. } => "position_embedding",
        NodeKind::KvAttention { .. } => "kv_attention",
        NodeKind::KdaChunk { .. } => "kda_chunk",
        NodeKind::KdaRecurrence { .. } => "kda_recurrence",
        NodeKind::KdaBackward { .. } => "kda_backward",
        NodeKind::KdaBackwardOut { .. } => "kda_backward_out",
        NodeKind::ShortConv1d { .. } => "short_conv1d",
        NodeKind::ConvState { .. } => "conv_state",
        NodeKind::LastTokenRow { .. } => "last_token_row",
        NodeKind::ChunkedHeadCe { .. } => "chunked_head_ce",
        NodeKind::ChunkedHeadCeBackward { .. } => "chunked_head_ce_backward",
        NodeKind::ChunkedHeadCeBackwardOut { .. } => "chunked_head_ce_backward_out",
        NodeKind::ShortConv1dBackwardX { .. } => "short_conv1d_backward_x",
        NodeKind::ShortConv1dBackwardW { .. } => "short_conv1d_backward_w",
        NodeKind::RotaryEmbedding { .. } => "rotary_embedding",
        NodeKind::RotaryEmbeddingBackward { .. } => "rotary_embedding_backward",
        NodeKind::LayerNorm { .. } => "layer_norm",
        NodeKind::RmsNorm { .. } => "rms_norm",
        NodeKind::LayerNormBackward { .. } => "layer_norm_backward",
        NodeKind::LayerNormBackwardOut { .. } => "layer_norm_backward_out",
        NodeKind::Linear { .. } => "linear",
        NodeKind::QuantizedLinear { .. } => "quantized_linear",
        NodeKind::QuantizedEmbedding { .. } => "quantized_embedding",
        NodeKind::Conv1d { .. } => "conv1d",
        NodeKind::Conv2d { .. } => "conv2d",
        NodeKind::ConvTranspose1d { .. } => "conv_transpose1d",
        NodeKind::ConvTranspose2d { .. } => "conv_transpose2d",
        NodeKind::Conv1dBackwardW { .. } => "conv1d_backward_w",
        NodeKind::Conv2dBackwardW { .. } => "conv2d_backward_w",
        NodeKind::Reshape { .. } => "reshape",
        NodeKind::Permute { .. } => "permute",
        NodeKind::Slice { .. } => "slice",
        NodeKind::Concat { .. } => "concat",
        NodeKind::BroadcastTo { .. } => "broadcast_to",
        NodeKind::Matmul { .. } => "matmul",
        NodeKind::Inverse { .. } => "inverse",
        NodeKind::Det { .. } => "det",
        NodeKind::Solve { .. } => "solve",
        NodeKind::AdamWStep { .. } => "adam_w_step",
        NodeKind::AdamWOut { .. } => "adam_w_out",
        NodeKind::SgdStep { .. } => "sgd_step",
        NodeKind::SgdOut { .. } => "sgd_out",
        NodeKind::StopGradient { .. } => "stop_gradient",
        NodeKind::Checkpoint { .. } => "checkpoint",
        NodeKind::Expose { .. } => "expose",
    }
}

pub(crate) fn region_expressions(region: &NativeRegion) -> Vec<&KernelExpr> {
    match region {
        NativeRegion::Elementwise(region) => vec![&region.output.expression],
        NativeRegion::ElementwiseReduce(region) => vec![&region.expression],
        NativeRegion::MultiOutput(region) => region
            .outputs
            .iter()
            .map(|output| &output.expression)
            .collect(),
        NativeRegion::AdamW(region) => region.expressions.iter().collect(),
        NativeRegion::AdamWGroup(region) => region.expressions.iter().collect(),
        NativeRegion::Sgd(region) => region.expressions.iter().collect(),
        NativeRegion::LinearResidual(_) | NativeRegion::LinearGelu(_) => Vec::new(),
    }
}

/// Applies checked semantic boundaries to every expression. Register conversions
/// round to the semantic dtype and widen back to the expression's carrier type.
/// This function never classifies operations or changes the selected policy.
pub fn legalize_region_expressions(
    region: &NativeRegion,
    plan: &ExecutableDTypePlan,
) -> Result<Box<[KernelExpr]>, String> {
    let execution = plan.execution();
    let operations = execution
        .operations
        .iter()
        .map(|operation| (operation.node, operation))
        .collect::<HashMap<_, _>>();
    region_expressions(region)
        .into_iter()
        .map(|expression| expression.apply_execution(&operations))
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_boxed_slice)
}

#[cfg(test)]
mod plan_tests {
    use super::*;
    use crate::test_target::TestTarget;
    use crate::{CompileOptions, CompilerDriver, ProgramRequest};

    #[test]
    fn missing_duplicate_and_reordered_units_are_rejected() {
        let x = Node::new(NodeKind::Input {
            slot: 0,
            shape: vec![1],
            dtype: DType::F32,
            device: Device::Cpu(0),
            storage: effect_torch_runtime::StorageMetadata::dense(),
        })
        .unwrap();
        let root = Node::new(NodeKind::Neg { a: x }).unwrap();
        let prepared = ProgramRequest::from_roots(vec![root], CompileOptions::default())
            .prepare()
            .unwrap();
        let target = TestTarget::for_index(&prepared.index);
        let driver = CompilerDriver::new(&prepared, &target).unwrap();
        let valid = driver.legalization();
        let mut missing = valid.clone();
        missing.units = missing.units.into_vec().into_iter().skip(1).collect();
        assert!(missing
            .validate(&prepared.index, driver.optimization(), &target)
            .unwrap_err()
            .contains("count"));
        let mut duplicate = valid.clone();
        duplicate.units[1] = duplicate.units[0].clone();
        assert!(duplicate
            .validate(&prepared.index, driver.optimization(), &target)
            .unwrap_err()
            .contains("order"));
        let mut reordered = valid.clone();
        reordered.units.swap(0, 1);
        assert!(reordered
            .validate(&prepared.index, driver.optimization(), &target)
            .unwrap_err()
            .contains("order"));
    }
}

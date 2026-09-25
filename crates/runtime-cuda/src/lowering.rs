//! CUDA lowering with independent semantic, instruction, and materialized value IDs.
use crate::cublas::{
    plan_row_bf16_gemm, Bf16GemmPlan, RowGemmKind, CUBLAS_WORKSPACE_BYTES, EXPERT_BLAS_STREAMS,
};
use crate::executable::{
    dtype_code, CudaKernelArgs, CudaStateLayout, Instruction, KernelSpec, StateAccess,
};
use crate::workspace::{CudaMemorySpace, CUDA_STORAGE_ALIGNMENT};
use crate::{CudaDevice, CudaValue};
use cudarc::driver::CudaFunction;
use effect_torch_compiler::{
    legalize_region_expressions, DenseNodeId, ExecutableDTypePlan, ExecutionRealization,
    GraphIndex, InstructionEffects, LegalizationPlan, LoweredInstruction, LoweredProgram,
    LoweredValue, LoweringUnit, NativeRegion, OperandInterpretation, OperandPreparation,
    OperationDTypeSpec, OptimizationPlan, OutputDecl, RegionId, ResultCompletion, ValueDecl,
    ValueStorage, ValueUse,
};
use effect_torch_graph::{node_children, Node, NodeKind};
use effect_torch_runtime::{
    DType, InstructionId, Location, SegmentOwnership, StorageClass, StorageMetadata,
    StorageRepresentation, ValueId, ValueSpec,
};

pub(super) type CudaLoweredProgram = LoweredProgram<&'static str, CudaMemorySpace, CudaValueMeta>;
pub(super) const BF16_LINEAR_BIAS_KERNEL: &str = "et_linear_bias_f32";

#[cfg(test)]
#[path = "executable_tests.rs"]
mod tests;

fn selected_result(kind: &NodeKind) -> Option<(&std::sync::Arc<Node>, u8)> {
    match kind {
        NodeKind::AdamWOut { step, index } | NodeKind::SgdOut { step, index } => {
            Some((step, *index))
        }
        NodeKind::SdpaBackwardOut { of, index }
        | NodeKind::KdaBackwardOut { of, index }
        | NodeKind::LayerNormBackwardOut { of, index }
        | NodeKind::ChunkedHeadCeBackwardOut { of, index } => Some((of, *index)),
        _ => None,
    }
}

fn kernel_name(name: &'static str, compute: u32) -> Result<&'static str, String> {
    if name == "et_reduce" && compute >= 4 {
        return Ok("et_reduce_integer");
    }
    let pair = match name {
        "et_reduce" => ("et_reduce_f32", "et_reduce_f64"),
        "et_matmul" => ("et_matmul_f32", "et_matmul_f64"),
        "et_rms_norm" => ("et_rms_norm_f32", "et_rms_norm_f64"),
        "et_cross_entropy" => ("et_cross_entropy_f32", "et_cross_entropy_f64"),
        "et_chunked_head_ce" => ("et_chunked_head_ce_f32", "et_chunked_head_ce_f64"),
        "et_conv" => ("et_conv_f32", "et_conv_f64"),
        "et_linalg" => ("et_linalg_f32", "et_linalg_f64"),
        "et_linear" => ("et_linear_f32", "et_linear_f64"),
        "et_linear_bias" => ("et_linear_bias_f32", "et_linear_bias_f64"),
        "et_layer_norm" => ("et_layer_norm_f32", "et_layer_norm_f64"),
        "et_sdpa" => ("et_sdpa_f32", "et_sdpa_f64"),
        "et_rotary" => ("et_rotary_f32", "et_rotary_f64"),
        "et_short_conv" => ("et_short_conv_f32", "et_short_conv_f64"),
        "et_kda" => ("et_kda_f32", "et_kda_f64"),
        "et_random" => ("et_random_f32", "et_random_f64"),
        _ => return Ok(name),
    };
    match compute {
        0 => Ok(pair.1),
        1 => Ok(pair.0),
        _ => Err(format!(
            "compile: {name} requires planned F32 or F64 operands"
        )),
    }
}

#[derive(Clone, Debug)]
pub(super) struct CudaValueMeta {
    pub(super) decl: ValueDecl<CudaMemorySpace>,
    pub(super) shape: Vec<usize>,
    pub(super) dtype: DType,
    pub(super) storage: StorageMetadata,
}
impl CudaValueMeta {
    pub(super) fn spec(&self) -> ValueSpec<'_> {
        ValueSpec {
            semantic_dtype: self.dtype,
            logical_shape: &self.shape,
            storage: self.storage.as_spec(),
        }
    }
}
impl LoweredValue<CudaMemorySpace> for CudaValueMeta {
    fn value_decl(&self) -> &ValueDecl<CudaMemorySpace> {
        &self.decl
    }
}

pub(super) enum CommandKind {
    Prepare,
    Value(CudaValue),
    Input {
        binding: usize,
    },
    Scalar {
        binding: usize,
    },
    Cursor {
        tensor: bool,
    },
    Alias {
        source: ValueId,
    },
    Gemm {
        x: ValueId,
        weight: ValueId,
        weight_transposed: bool,
        plan: Bf16GemmPlan,
        out_f32: bool,
        workspace: ValueId,
    },
    /// Exact active group sizes require bounded host control readback. This
    /// command is non-capturable; all payloads and stable grouping stay on GPU.
    GroupedExpert {
        x: ValueId,
        weight: ValueId,
        indexes: ValueId,
        rows: usize,
        columns: usize,
        inner: usize,
        experts: usize,
        control: ValueId,
        row_map: ValueId,
        gathered: ValueId,
        projected: ValueId,
        workspace: Option<ValueId>,
        reuse_routing: bool,
        source_rows: usize,
    },
    /// Infallible numerical epilogue with by-value geometry and no status I/O.
    LinearBias {
        accumulator: ValueId,
        bias: ValueId,
        args: CudaKernelArgs,
    },
    FusedElementwise {
        function: CudaFunction,
        args: CudaKernelArgs,
        inputs: [Option<ValueId>; 8],
    },
    Kernel {
        name: &'static str,
        args: CudaKernelArgs,
        inputs: [Option<ValueId>; 8],
        scratch: [Option<ValueId>; 3],
        status: Option<ValueId>,
        checked: bool,
        metadata: Vec<u64>,
        state: StateAccess,
        state_buffers: [Option<ValueId>; 4],
        kv_matmul: Option<crate::kv_matmul::KvMatmulWorkspace>,
    },
}
pub(super) struct Command {
    pub(super) output: Option<ValueId>,
    pub(super) kind: CommandKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum StateComponent {
    Keys(usize),
    Values(usize),
    KeyScales(usize),
    ValueScales(usize),
    Kda(usize),
    Conv(usize),
}

fn product_checked(shape: &[usize]) -> Result<usize, String> {
    crate::value::element_count(shape)
}

fn supports_inline_conversion(name: &str) -> bool {
    matches!(
        name,
        "et_binary"
            | "et_unary"
            | "et_reindex"
            | "et_where"
            | "et_concat"
            | "et_index"
            | "et_rms_norm"
    )
}

fn kernel_can_report_error(name: &str, args: &CudaKernelArgs) -> bool {
    match name {
        "et_convert" | "et_unary" | "et_where" | "et_reindex" | "et_concat" | "et_rms_norm_f32"
        | "et_rms_norm_f64" | "et_random_f32" | "et_random_f64" | "et_sequence"
        | "et_reduce_f32" | "et_reduce_f64" => false,
        "et_binary" => args.compute_dtype >= 4 && args.operation == 3,
        // Empty arg reductions fail during compilation. Cumsum performs no
        // indexed read, so only selection and scatter variants can report an
        // out-of-range index.
        "et_index" => args.operation >= 3,
        _ => true,
    }
}

#[derive(Clone, Copy)]
struct NativeGemm {
    geometry: Bf16GemmPlan,
    weight: usize,
    weight_transposed: bool,
}

/// Immutable physical choices derived from the already accepted dtype plan.
/// Transpose views never escape into graph outputs or canonical kernels.
struct CudaGemmLoweringPlan {
    operations: Vec<Option<NativeGemm>>,
    weight_views: Vec<bool>,
}

impl CudaGemmLoweringPlan {
    fn new(index: &GraphIndex, legalization: &LegalizationPlan) -> Result<Self, String> {
        let mut operations = vec![None; index.order.len()];
        for entry in legalization.units() {
            let LoweringUnit::Node(dense) = entry.unit() else {
                continue;
            };
            let execution = entry.disposition().execution();
            let operation = execution
                .operation(dense)
                .ok_or("compile: missing CUDA dtype plan")?;
            if execution.realization != ExecutionRealization::DirectKernel
                || operation.compute_dtype != Some(DType::BF16)
            {
                continue;
            }
            let node = index.node(dense).ok_or("compile: missing GEMM node")?;
            let (kind, x, weight) = match &node.kind {
                NodeKind::Linear { x, weight, .. } => (RowGemmKind::Linear, x, weight),
                NodeKind::Matmul { a, b } => (RowGemmKind::Matmul, a, b),
                _ => continue,
            };
            if operation
                .operands
                .iter()
                .any(|operand| operand.preparation != OperandPreparation::Direct)
                || operation
                    .results
                    .iter()
                    .any(|result| result.completion != ResultCompletion::Direct)
            {
                return Err(
                    "compile: native BF16 GEMM requires direct operands and results".into(),
                );
            }
            let geometry = plan_row_bf16_gemm(kind, &x.shape, &weight.shape, &node.shape)
                .ok_or("compile: accepted native BF16 GEMM has unsupported geometry")?;
            let (source, weight_transposed) = match &weight.kind {
                NodeKind::Permute { a, dims } if a.shape.len() == 2 && dims == &[1, 0] => (a, true),
                _ => (weight, false),
            };
            operations[dense.index()] = Some(NativeGemm {
                geometry,
                weight: index
                    .dense_id(source.id)
                    .ok_or("compile: missing GEMM weight source")?
                    .index(),
                weight_transposed,
            });
        }
        let mut weight_views = vec![false; index.order.len()];
        for (position, node) in index.order.iter().enumerate() {
            let NodeKind::Permute { a, dims } = &node.kind else {
                continue;
            };
            if a.shape.len() != 2
                || dims != &[1, 0]
                || index.roots.iter().any(|root| root.index() == position)
                || index.consumers[position].is_empty()
            {
                continue;
            }
            weight_views[position] = index.consumers[position].iter().all(|consumer| {
                let Some(gemm) = operations[consumer.index()] else {
                    return false;
                };
                if !gemm.weight_transposed {
                    return false;
                }
                let children = node_children(&index.order[consumer.index()].kind);
                children.get(1).is_some_and(|child| child.id == node.id)
                    && children
                        .iter()
                        .enumerate()
                        .all(|(role, child)| role == 1 || child.id != node.id)
            });
        }
        Ok(Self {
            operations,
            weight_views,
        })
    }
}

pub(super) struct CudaProgramBuilder {
    pub(super) values: Vec<CudaValueMeta>,
    pub(super) commands: Vec<Command>,
    lowered: Vec<LoweredInstruction<&'static str>>,
    semantic_values: Vec<Option<ValueId>>,
    semantic_results: Vec<Vec<ValueId>>,
    elementwise_views: Vec<Option<ElementwiseView>>,
    rms_row_views: Vec<Option<ElementwiseView>>,
    routed_row_views: Vec<Option<RoutedRowsView>>,
    grouped_routing: std::collections::HashMap<(usize, usize, usize), (ValueId, ValueId)>,
    state_layout: Option<CudaStateLayout>,
    state_values: std::collections::HashMap<StateComponent, (ValueId, Option<ValueId>)>,
    status: Option<ValueId>,
    pub(super) conversion_count: usize,
    pub(super) conversion_bytes: usize,
    gemms: CudaGemmLoweringPlan,
    bf16_kv_matmul: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ElementwiseView {
    source: ValueId,
    shape: Vec<usize>,
    strides: Vec<usize>,
    moduli: Vec<usize>,
    offset: usize,
}

#[derive(Clone, Debug)]
struct RoutedRowsView {
    source: ValueId,
    routes: usize,
    rows: usize,
    width: usize,
}

fn contiguous_strides(shape: &[usize]) -> Result<Vec<usize>, String> {
    let mut strides = vec![1usize; shape.len()];
    for dimension in (0..shape.len().saturating_sub(1)).rev() {
        strides[dimension] = strides[dimension + 1]
            .checked_mul(shape[dimension + 1])
            .ok_or("compile: CUDA view stride overflow")?;
    }
    Ok(strides)
}

fn reshape_view(view: &ElementwiseView, shape: &[usize]) -> Option<ElementwiseView> {
    if view.moduli.iter().any(|&modulus| modulus != 0) {
        return None;
    }
    let old = view
        .shape
        .iter()
        .copied()
        .zip(view.strides.iter().copied())
        .filter(|(size, _)| *size != 1)
        .collect::<Vec<_>>();
    let new = shape
        .iter()
        .copied()
        .filter(|size| *size != 1)
        .collect::<Vec<_>>();
    if old.iter().map(|(size, _)| *size).ne(new.iter().copied()) {
        return None;
    }
    let mut physical = old.iter().map(|(_, stride)| *stride);
    let strides = shape
        .iter()
        .map(|&size| if size == 1 { Some(0) } else { physical.next() })
        .collect::<Option<Vec<_>>>()?;
    Some(ElementwiseView {
        source: view.source,
        shape: shape.to_vec(),
        strides,
        moduli: vec![0; shape.len()],
        offset: view.offset,
    })
}

fn rotary_reindex_source(node: &Node, index: &GraphIndex) -> Option<(DenseNodeId, usize)> {
    if !node.dtype.is_float() {
        return None;
    }
    let NodeKind::Concat { a, b, dim } = &node.kind else {
        return None;
    };
    let NodeKind::Neg { a: negative } = &a.kind else {
        return None;
    };
    let NodeKind::Slice {
        a: negative_source,
        ranges: negative_ranges,
    } = &negative.kind
    else {
        return None;
    };
    let NodeKind::Slice {
        a: positive_source,
        ranges: positive_ranges,
    } = &b.kind
    else {
        return None;
    };
    if negative_source.id != positive_source.id
        || *dim + 1 != negative_source.shape.len()
        || node.shape != negative_source.shape
        || negative_ranges.len() != negative_source.shape.len()
        || positive_ranges.len() != negative_source.shape.len()
    {
        return None;
    }
    let width = negative_source.shape[*dim];
    if width == 0 || width % 2 != 0 {
        return None;
    }
    let half = width / 2;
    for (dimension, &size) in negative_source.shape.iter().enumerate() {
        let negative_expected = if dimension == *dim {
            (half, width, 1)
        } else {
            (0, size, 1)
        };
        let positive_expected = if dimension == *dim {
            (0, half, 1)
        } else {
            (0, size, 1)
        };
        if negative_ranges[dimension] != negative_expected
            || positive_ranges[dimension] != positive_expected
        {
            return None;
        }
    }
    index
        .dense_id(negative_source.id)
        .map(|source| (source, width))
}

fn repeated_trailing_concat_source(
    node: &Node,
    index: &GraphIndex,
) -> Option<(DenseNodeId, usize)> {
    let NodeKind::Concat { a, b, dim } = &node.kind else {
        return None;
    };
    if a.id != b.id
        || *dim + 1 != node.shape.len()
        || a.shape.len() != node.shape.len()
        || a.dtype != node.dtype
        || a.shape[*dim] == 0
        || a.shape[*dim].checked_mul(2) != Some(node.shape[*dim])
        || a.shape
            .iter()
            .zip(&node.shape)
            .enumerate()
            .any(|(dimension, (source, output))| dimension != *dim && source != output)
    {
        return None;
    }
    index.dense_id(a.id).map(|source| (source, a.shape[*dim]))
}

fn compact_inner_scatter_indexes(
    node: &Node,
    index: &GraphIndex,
) -> Result<Option<(DenseNodeId, usize, usize)>, String> {
    let NodeKind::ScatterAdd {
        a,
        dim,
        indexes,
        src,
    } = &node.kind
    else {
        return Ok(None);
    };
    if *dim + 1 >= a.shape.len() || indexes.shape != src.shape || src.shape != a.shape {
        return Ok(None);
    }
    let NodeKind::BroadcastTo { a: broadcast, .. } = &indexes.kind else {
        return Ok(None);
    };
    let outer = product_checked(&a.shape[..*dim])?;
    let routes = a.shape[*dim];
    let inner = product_checked(&a.shape[*dim + 1..])?;
    if routes == 0 || routes > 32 {
        return Ok(None);
    }
    let compact_elements = outer
        .checked_mul(routes)
        .ok_or("compile: CUDA compact scatter geometry overflow")?;
    if product_checked(&broadcast.shape)? != compact_elements {
        return Ok(None);
    }
    let mut compact = broadcast;
    while let NodeKind::Reshape { a, .. } = &compact.kind {
        compact = a;
    }
    if product_checked(&compact.shape)? != compact_elements
        || !matches!(compact.dtype, DType::I64 | DType::U32)
    {
        return Ok(None);
    }
    Ok(index
        .dense_id(compact.id)
        .map(|dense| (dense, routes, inner)))
}

impl CudaProgramBuilder {
    pub(super) fn new(
        index: &GraphIndex,
        state_layout: Option<CudaStateLayout>,
        legalization: &LegalizationPlan,
    ) -> Result<Self, String> {
        Ok(Self {
            values: Vec::new(),
            commands: Vec::new(),
            lowered: Vec::new(),
            semantic_values: vec![None; index.order.len()],
            semantic_results: vec![Vec::new(); index.order.len()],
            elementwise_views: vec![None; index.order.len()],
            rms_row_views: vec![None; index.order.len()],
            routed_row_views: vec![None; index.order.len()],
            grouped_routing: std::collections::HashMap::new(),
            state_layout,
            state_values: std::collections::HashMap::new(),
            status: None,
            conversion_count: 0,
            conversion_bytes: 0,
            gemms: CudaGemmLoweringPlan::new(index, legalization)?,
            bf16_kv_matmul: legalization
                .target()
                .features
                .iter()
                .any(|f| f == "stepwise-bf16-kv-f32-gemm-active-rows-v1"),
        })
    }

    fn value(
        &mut self,
        shape: Vec<usize>,
        dtype: DType,
        storage: StorageMetadata,
        name: &str,
        allocation: ValueStorage<CudaMemorySpace>,
    ) -> Result<ValueId, String> {
        let id = ValueId::from_index(self.values.len()).ok_or("compile: too many CUDA values")?;
        let bytes = ValueSpec {
            semantic_dtype: dtype,
            logical_shape: &shape,
            storage: storage.as_spec(),
        }
        .canonical_geometry()?
        .byte_len;
        self.values.push(CudaValueMeta {
            decl: ValueDecl {
                id,
                name: format!("{name}_{}", id.get()),
                bytes,
                storage: allocation,
            },
            shape,
            dtype,
            storage,
        });
        Ok(id)
    }
    fn planned(&mut self, shape: Vec<usize>, dtype: DType, name: &str) -> Result<ValueId, String> {
        self.value(
            shape,
            dtype,
            StorageMetadata::dense(),
            name,
            ValueStorage::Planned {
                class: StorageClass::Workspace,
                alignment: CUDA_STORAGE_ALIGNMENT,
                memory_space: CudaMemorySpace::Device,
                ownership: SegmentOwnership::Workspace,
            },
        )
    }
    fn resolve(&self, semantic: usize) -> Result<ValueId, String> {
        self.semantic_values
            .get(semantic)
            .copied()
            .flatten()
            .ok_or_else(|| format!("compile: missing CUDA semantic value {semantic}"))
    }

    fn feeds_only_fused_consumer(
        index: &GraphIndex,
        optimization: &OptimizationPlan,
        dense: DenseNodeId,
    ) -> bool {
        if index.roots.iter().any(|root| root.index() == dense.index()) {
            return false;
        }
        let Some(consumers) = index.consumers_of(dense) else {
            return false;
        };
        let Some(producer) = index.node(dense) else {
            return false;
        };
        !consumers.is_empty()
            && consumers.iter().all(|&consumer| {
                if let Some(region) = optimization.node_region[consumer.index()] {
                    return matches!(
                        optimization.regions.get(region.index()),
                        Some(NativeRegion::Elementwise(_))
                    );
                }
                if let Some(node) = index.node(consumer) {
                    if matches!(
                        &node.kind,
                        NodeKind::ScatterAdd { indexes, .. } if indexes.id == producer.id
                    ) && compact_inner_scatter_indexes(node, index)
                        .is_ok_and(|specialized| specialized.is_some())
                    {
                        return true;
                    }
                }
                matches!(
                    index.node(consumer).map(|node| &node.kind),
                    Some(NodeKind::Reshape { .. })
                ) && Self::feeds_only_fused_consumer(index, optimization, consumer)
            })
    }

    fn feeds_direct_fused_consumer(
        index: &GraphIndex,
        optimization: &OptimizationPlan,
        dense: DenseNodeId,
    ) -> bool {
        index.consumers_of(dense).is_some_and(|consumers| {
            !consumers.is_empty()
                && consumers.iter().all(|consumer| {
                    optimization.node_region[consumer.index()].is_some_and(|region| {
                        matches!(
                            optimization.regions.get(region.index()),
                            Some(NativeRegion::Elementwise(_))
                        )
                    })
                })
        })
    }

    fn feeds_only_grouped_input(index: &GraphIndex, dense: DenseNodeId) -> bool {
        if index.roots.iter().any(|root| root.index() == dense.index()) {
            return false;
        }
        let Some(producer) = index.node(dense) else {
            return false;
        };
        let Some(consumers) = index.consumers_of(dense) else {
            return false;
        };
        !consumers.is_empty()
            && consumers.iter().all(|&consumer| {
                let Some(node) = index.node(consumer) else {
                    return false;
                };
                match &node.kind {
                    NodeKind::GroupedExpertLinearRows { x, .. } => x.id == producer.id,
                    NodeKind::Reshape { .. } => Self::feeds_only_grouped_input(index, consumer),
                    _ => false,
                }
            })
    }

    fn feeds_only_rotary_reindex(index: &GraphIndex, dense: DenseNodeId) -> bool {
        if index.roots.iter().any(|root| root.index() == dense.index()) {
            return false;
        }
        let Some(producer) = index.node(dense) else {
            return false;
        };
        let Some(consumers) = index.consumers_of(dense) else {
            return false;
        };
        !consumers.is_empty()
            && consumers.iter().all(|&consumer| {
                let Some(node) = index.node(consumer) else {
                    return false;
                };
                if rotary_reindex_source(node, index).is_some() {
                    return matches!(
                        &node.kind,
                        NodeKind::Concat { a, b, .. }
                            if a.id == producer.id || b.id == producer.id
                    );
                }
                let NodeKind::Neg { a } = &node.kind else {
                    return false;
                };
                a.id == producer.id && Self::feeds_only_rotary_reindex(index, consumer)
            })
    }

    fn feeds_only_rms_input(index: &GraphIndex, dense: DenseNodeId) -> bool {
        if index.roots.iter().any(|root| root.index() == dense.index()) {
            return false;
        }
        let Some(producer) = index.node(dense) else {
            return false;
        };
        let Some(consumers) = index.consumers_of(dense) else {
            return false;
        };
        !consumers.is_empty()
            && consumers.iter().all(|&consumer| {
                matches!(
                    index.node(consumer).map(|node| &node.kind),
                    Some(NodeKind::RmsNorm { x, .. }) if x.id == producer.id
                )
            })
    }

    fn emit(
        &mut self,
        name: &'static str,
        kind: CommandKind,
        output: Option<ValueId>,
        inputs: Vec<ValueUse>,
        defines: bool,
    ) -> Result<(), String> {
        let id = InstructionId::from_index(self.lowered.len())
            .ok_or("compile: too many CUDA instructions")?;
        let outputs = if defines {
            output.into_iter().map(OutputDecl::new).collect()
        } else {
            Vec::new()
        };
        self.lowered
            .push(LoweredInstruction::new(id, name, inputs, outputs));
        self.commands.push(Command { output, kind });
        Ok(())
    }

    fn conversion(&mut self, source: ValueId, destination: DType) -> Result<ValueId, String> {
        let metadata = &self.values[source.index()];
        if metadata.storage.representation != StorageRepresentation::Dense {
            return Err("compile: dense conversion cannot decode packed storage".into());
        }
        let output = self.planned(metadata.shape.clone(), destination, "convert")?;
        let mut spec = KernelSpec {
            name: "et_convert",
            args: CudaKernelArgs::default(),
            inputs: [None; 8],
            tail: Vec::new(),
            scratch: [None; 3],
            state: StateAccess::None,
        };
        spec.args.compute_dtype = dtype_code(destination);
        let mut inputs = [None; 8];
        inputs[0] = Some(source);
        self.kernel(spec, inputs, output)?;
        self.conversion_count += 1;
        self.conversion_bytes = self
            .conversion_bytes
            .checked_add(self.values[output.index()].decl.bytes)
            .ok_or("compile: conversion bytes overflow")?;
        Ok(output)
    }

    fn state_value(
        &mut self,
        component: StateComponent,
        elements: usize,
        dtype: DType,
        transaction: Option<usize>,
    ) -> Result<(ValueId, Option<ValueId>), String> {
        if let Some(ids) = self.state_values.get(&component) {
            return Ok(*ids);
        }
        let fixed = self.value(
            vec![elements],
            dtype,
            StorageMetadata::dense(),
            "state",
            ValueStorage::Fixed {
                class: StorageClass::PersistentState,
                location: Location::Persistent {
                    slot: u32::try_from(self.values.len())
                        .map_err(|_| "compile: too many state slots")?,
                },
            },
        )?;
        let staging = if let Some(elements) = transaction {
            let id = self.value(
                vec![elements],
                dtype,
                StorageMetadata::dense(),
                "state_transaction",
                ValueStorage::Planned {
                    class: StorageClass::Workspace,
                    alignment: CUDA_STORAGE_ALIGNMENT,
                    memory_space: CudaMemorySpace::Device,
                    ownership: SegmentOwnership::StateTransaction,
                },
            )?;
            let instruction = InstructionId::from_index(self.lowered.len())
                .ok_or("compile: too many state copies")?;
            self.lowered.push(
                LoweredInstruction::new(
                    instruction,
                    "state_prepare",
                    Vec::new(),
                    vec![OutputDecl::new(id)],
                )
                .with_resources(
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    vec![ValueUse::read(fixed)],
                ),
            );
            Some(id)
        } else {
            None
        };
        self.state_values.insert(component, (fixed, staging));
        Ok((fixed, staging))
    }

    fn kernel(
        &mut self,
        mut spec: KernelSpec,
        inputs: [Option<ValueId>; 8],
        output: ValueId,
    ) -> Result<(), String> {
        spec.name = kernel_name(spec.name, spec.args.compute_dtype)?;
        if matches!(spec.name, "et_reduce_f32" | "et_reduce_f64") {
            let input = inputs[0].ok_or("compile: CUDA reduction input is missing")?;
            if self.values[input.index()].shape.len() > 64 {
                return Err("compile: CUDA reduction rank exceeds 64".into());
            }
        }
        let mut kv_matmul = None;
        let mut state_buffers = [None; 4];
        let mut state_uses = Vec::new();
        match spec.state {
            StateAccess::Kv {
                layer,
                heads,
                dim,
                batch,
            } => {
                let layout = self
                    .state_layout
                    .clone()
                    .ok_or("compile: CUDA KV attention requires an explicit state layout")?;
                let descriptor = layout
                    .kv_layers
                    .iter()
                    .find(|d| d.layer_id as usize == layer)
                    .ok_or("compile: KV layer descriptor missing")?;
                if descriptor.kv_heads != heads || descriptor.head_dim != dim {
                    return Err("compile: KV layer geometry mismatch".into());
                }
                let tokens = spec.args.integers[7] as usize;
                let temporary_rows = product_checked(&[batch, tokens, heads])?;
                let temporary_elements = product_checked(&[temporary_rows, dim])?;
                let table_rows = (layout.capacity as usize)
                    .checked_add(
                        tokens
                            .checked_mul(layout.packed_rows_per_sequence.unwrap_or(1) as usize)
                            .ok_or("compile: KV rows overflow")?,
                    )
                    .and_then(|n| n.checked_add(1))
                    .ok_or("compile: KV table overflow")?;
                // Keep the pointer table and per-query score rows in one planned
                // invocation allocation. Scores become probabilities in place;
                // no persistent KV storage is copied or used as scratch.
                let table_bytes = product_checked(&[layout.slots as usize, table_rows, 4, 8])?;
                let score_width = if spec.args.compute_dtype == 0 && spec.args.integers[6] == 0 {
                    8
                } else {
                    4
                };
                let score_bytes = product_checked(&[
                    batch,
                    tokens,
                    spec.args.integers[11] as usize,
                    table_rows - 1,
                    score_width,
                ])?;
                spec.args.integers[9] = (table_rows - 1) as u64;
                spec.args.integers[12] = table_bytes as u64;
                let mut workspace_bytes = table_bytes
                    .checked_add(score_bytes)
                    .ok_or("compile: KV score workspace overflow")?;
                if self.bf16_kv_matmul
                    && descriptor.dtype == DType::BF16
                    && spec.args.integers[6] == 1
                    && spec.args.integers[8] == 3
                    && spec.args.compute_dtype == 1
                    && self.values[output.index()].dtype == DType::F32
                {
                    kv_matmul = crate::kv_matmul::KvMatmulWorkspace::new(
                        workspace_bytes,
                        spec.args.integers[11] as usize,
                        tokens,
                        table_rows - 1,
                        dim,
                    );
                    if let Some(plan) = kv_matmul {
                        workspace_bytes = plan.bytes;
                    }
                }
                spec.scratch[2] = Some((workspace_bytes, DType::U8));
                layout.validate()?;
                if product_checked(&[
                    layout.slots as usize,
                    layout.packed_rows_per_sequence.unwrap_or(1) as usize,
                ])? != batch
                {
                    return Err("compile: KV state layout batch mismatch".into());
                }
                let rows =
                    product_checked(&[layout.slots as usize, layout.capacity as usize, heads])?;
                let elements = product_checked(&[rows, dim])?;
                for (slot, component) in [
                    StateComponent::Keys(layer),
                    StateComponent::Values(layer),
                    StateComponent::KeyScales(layer),
                    StateComponent::ValueScales(layer),
                ]
                .into_iter()
                .enumerate()
                {
                    if slot >= 2 && descriptor.dtype != DType::U8 {
                        continue;
                    }
                    let (_, transaction) = self.state_value(
                        component,
                        if slot < 2 { elements } else { rows },
                        if slot < 2 {
                            descriptor.dtype
                        } else {
                            DType::F32
                        },
                        Some(if slot < 2 {
                            temporary_elements
                        } else {
                            temporary_rows
                        }),
                    )?;
                    state_buffers[slot] = transaction;
                    state_uses.push(ValueUse::read_write(
                        transaction.ok_or("compile: missing state transaction")?,
                    ));
                }
            }
            StateAccess::Kda {
                layer: Some(layer),
                batch,
                elements_per_sequence,
                ..
            }
            | StateAccess::Conv {
                layer: Some(layer),
                batch,
                elements_per_sequence,
            } => {
                if self.state_layout.as_ref().is_some_and(|layout| {
                    layout.access == effect_torch_runtime::StateAccessMode::ReadOnly
                }) {
                    return Err(
                        "compile: read-only state does not support recurrent mutation".into(),
                    );
                }
                let component = if matches!(spec.state, StateAccess::Kda { .. }) {
                    StateComponent::Kda(layer)
                } else {
                    StateComponent::Conv(layer)
                };
                let (fixed, _) = self.state_value(
                    component,
                    product_checked(&[batch, elements_per_sequence])?,
                    DType::F32,
                    None,
                )?;
                state_uses.push(ValueUse::read_write(fixed));
            }
            _ => {}
        }
        let mut scratch = [None; 3];
        for (slot, requirement) in spec.scratch.iter().enumerate() {
            if let Some((elements, dtype)) = requirement {
                scratch[slot] = Some(self.planned(vec![*elements], *dtype, "scratch")?);
            }
        }
        let checked = kernel_can_report_error(spec.name, &spec.args);
        let (status, define_status) = if checked {
            if let Some(status) = self.status {
                (Some(status), false)
            } else {
                let status = self.planned(vec![1], DType::I64, "status")?;
                self.status = Some(status);
                (Some(status), true)
            }
        } else {
            (None, false)
        };
        let mut definitions = scratch
            .iter()
            .flatten()
            .copied()
            .map(OutputDecl::new)
            .collect::<Vec<_>>();
        if define_status {
            definitions.push(OutputDecl::new(
                status.ok_or("compile: CUDA status is missing")?,
            ));
        }
        let prepare_id = InstructionId::from_index(self.lowered.len())
            .ok_or("compile: too many CUDA instructions")?;
        self.lowered.push(LoweredInstruction::new(
            prepare_id,
            "prepare",
            Vec::new(),
            definitions,
        ));
        self.commands.push(Command {
            output: None,
            kind: CommandKind::Prepare,
        });
        let out = &self.values[output.index()];
        spec.args.elements = crate::value::element_count(&out.shape)? as u64;
        if spec.name == "et_rms_norm_f32" {
            spec.args.integers[0] = *out.shape.last().ok_or("compile: RMS input rank")? as u64;
        }
        spec.args.output_dtype = dtype_code(out.dtype);
        let mut metadata = vec![out.shape.len() as u64];
        metadata.extend(
            inputs
                .iter()
                .map(|id| id.map_or(0, |id| self.values[id.index()].shape.len() as u64)),
        );
        metadata.extend(out.shape.iter().map(|d| *d as u64));
        for (slot, id) in inputs.iter().enumerate() {
            if let Some(id) = id {
                let value = &self.values[id.index()];
                metadata.extend(value.shape.iter().map(|d| *d as u64));
                spec.args.input_dtypes[slot] = dtype_code(
                    if value.storage.representation == StorageRepresentation::Dense {
                        value.dtype
                    } else {
                        DType::U8
                    },
                );
            }
        }
        metadata.extend(spec.tail);
        let metadata_bytes = metadata
            .len()
            .checked_mul(8)
            .ok_or("compile: CUDA metadata size overflow")?;
        let metadata_id = self.value(
            vec![metadata_bytes],
            DType::U8,
            StorageMetadata::dense(),
            "metadata",
            ValueStorage::Fixed {
                class: StorageClass::PersistentConstant,
                location: Location::Persistent {
                    slot: u32::try_from(self.values.len())
                        .map_err(|_| "compile: metadata slots overflow")?,
                },
            },
        )?;
        let mut uses = inputs.iter().flatten().copied().collect::<Vec<_>>();
        uses.sort_unstable();
        uses.dedup();
        uses.push(metadata_id);
        let stateful = matches!(
            spec.state,
            StateAccess::Kv { .. }
                | StateAccess::Kda { layer: Some(_), .. }
                | StateAccess::Conv { layer: Some(_), .. }
        );
        let id = InstructionId::from_index(self.lowered.len())
            .ok_or("compile: CUDA instruction count overflow")?;
        self.lowered.push(
            LoweredInstruction::new(
                id,
                if kv_matmul.is_some() {
                    "kv_stepwise_bf16_gemm_active_rows"
                } else {
                    spec.name
                },
                uses.into_iter().map(ValueUse::read).collect::<Vec<_>>(),
                vec![OutputDecl::new(output)],
            )
            .with_resources(
                scratch
                    .iter()
                    .flatten()
                    .copied()
                    .map(ValueUse::read_write)
                    .collect::<Vec<_>>(),
                Vec::new(),
                status
                    .into_iter()
                    .map(ValueUse::read_write)
                    .collect::<Vec<_>>(),
                state_uses,
            )
            .with_effects(InstructionEffects {
                may_fail: checked,
                has_side_effects: stateful,
            }),
        );
        self.commands.push(Command {
            output: Some(output),
            kind: CommandKind::Kernel {
                name: spec.name,
                args: spec.args,
                inputs,
                scratch,
                status,
                checked,
                metadata,
                state: spec.state,
                state_buffers,
                kv_matmul,
            },
        });
        Ok(())
    }

    pub(super) fn add(
        &mut self,
        dense: DenseNodeId,
        node: &Node,
        index: &GraphIndex,
        optimization: &OptimizationPlan,
        instruction: Instruction,
        plan: &ExecutableDTypePlan,
    ) -> Result<(), String> {
        let execution = plan.execution();
        if !matches!(
            execution.realization,
            ExecutionRealization::DirectKernel | ExecutionRealization::MaterializedTransforms
        ) {
            return Err(
                "compile: CUDA lowerer requires a direct or materialized execution plan".into(),
            );
        }
        let operation = execution
            .operation(dense)
            .ok_or("compile: CUDA legalization entry is missing")?;
        let boundary_storage = node.storage.clone();
        let direct_boundary = operation
            .operands
            .iter()
            .all(|operand| operand.preparation == OperandPreparation::Direct)
            && operation
                .results
                .iter()
                .all(|result| result.completion == ResultCompletion::Direct);
        let fused_consumer = Self::feeds_only_fused_consumer(index, optimization, dense);
        let grouped_input = Self::feeds_only_grouped_input(index, dense);
        let rotary_reindex = Self::feeds_only_rotary_reindex(index, dense);
        let rms_input = Self::feeds_only_rms_input(index, dense);
        if let NodeKind::Neg { a } = &node.kind {
            if rotary_reindex {
                let parent = index
                    .dense_id(a.id)
                    .ok_or("compile: CUDA rotary negation source is missing")?;
                let source = self.resolve(parent.index())?;
                self.semantic_values[dense.index()] = Some(source);
                self.semantic_results[dense.index()] = vec![source];
                return Ok(());
            }
        }
        if direct_boundary
            && boundary_storage.representation == StorageRepresentation::Dense
            && (fused_consumer || grouped_input || rotary_reindex || rms_input)
        {
            if Self::feeds_direct_fused_consumer(index, optimization, dense) {
                if let Some((parent, modulus)) = repeated_trailing_concat_source(node, index) {
                    let source = self.resolve(parent.index())?;
                    let source_shape = &index
                        .node(parent)
                        .ok_or("compile: CUDA repeated concat source is missing")?
                        .shape;
                    let base =
                        self.elementwise_views[parent.index()]
                            .clone()
                            .unwrap_or(ElementwiseView {
                                source,
                                shape: source_shape.clone(),
                                strides: contiguous_strides(source_shape)?,
                                moduli: vec![0; node.shape.len()],
                                offset: 0,
                            });
                    if base.moduli.iter().all(|&value| value == 0) {
                        let mut moduli = base.moduli.clone();
                        *moduli
                            .last_mut()
                            .ok_or("compile: CUDA repeated concat rank is zero")? = modulus;
                        let view = ElementwiseView {
                            source: base.source,
                            shape: node.shape.clone(),
                            strides: base.strides,
                            moduli,
                            offset: base.offset,
                        };
                        self.semantic_values[dense.index()] = Some(base.source);
                        self.semantic_results[dense.index()] = vec![base.source];
                        self.elementwise_views[dense.index()] = Some(view);
                        return Ok(());
                    }
                }
            }
            if let NodeKind::BroadcastTo { a, .. } = &node.kind {
                if grouped_input
                    && a.shape.len() == 3
                    && node.shape.len() == 3
                    && a.shape[0] == 1
                    && a.shape[1] == node.shape[1]
                    && a.shape[2] == node.shape[2]
                    && node.shape[0] != 0
                {
                    let parent = index
                        .dense_id(a.id)
                        .ok_or("compile: CUDA routed-row source is missing")?;
                    let source = self.resolve(parent.index())?;
                    let view = RoutedRowsView {
                        source,
                        routes: node.shape[0],
                        rows: node.shape[1],
                        width: node.shape[2],
                    };
                    self.semantic_values[dense.index()] = Some(source);
                    self.semantic_results[dense.index()] = vec![source];
                    self.routed_row_views[dense.index()] = Some(view);
                    return Ok(());
                }
                if fused_consumer && node.dtype == a.dtype && a.shape.len() <= node.shape.len() {
                    let parent = index
                        .dense_id(a.id)
                        .ok_or("compile: CUDA broadcast source is missing")?;
                    let source = self.resolve(parent.index())?;
                    let base =
                        self.elementwise_views[parent.index()]
                            .clone()
                            .unwrap_or(ElementwiseView {
                                source,
                                shape: a.shape.clone(),
                                strides: contiguous_strides(&a.shape)?,
                                moduli: vec![0; a.shape.len()],
                                offset: 0,
                            });
                    let rank_offset = node.shape.len() - a.shape.len();
                    let mut strides = Vec::with_capacity(node.shape.len());
                    let mut moduli = Vec::with_capacity(node.shape.len());
                    for (dimension, &size) in node.shape.iter().enumerate() {
                        if dimension < rank_offset {
                            strides.push(0);
                            moduli.push(0);
                            continue;
                        }
                        let source_dimension = dimension - rank_offset;
                        let source_size = a.shape[source_dimension];
                        if source_size != 1 && source_size != size {
                            return Err(
                                "compile: CUDA elementwise broadcast view is invalid".into()
                            );
                        }
                        strides.push(if source_size == 1 {
                            0
                        } else {
                            base.strides[source_dimension]
                        });
                        moduli.push(if source_size == 1 {
                            0
                        } else {
                            base.moduli[source_dimension]
                        });
                    }
                    let view = ElementwiseView {
                        source: base.source,
                        shape: node.shape.clone(),
                        strides,
                        moduli,
                        offset: base.offset,
                    };
                    self.semantic_values[dense.index()] = Some(base.source);
                    self.semantic_results[dense.index()] = vec![base.source];
                    self.elementwise_views[dense.index()] = Some(view);
                    return Ok(());
                }
            }
            if let NodeKind::Slice { a, ranges } = &node.kind {
                if (fused_consumer || rotary_reindex)
                    && node.dtype == a.dtype
                    && ranges.len() == a.shape.len()
                {
                    let parent = index
                        .dense_id(a.id)
                        .ok_or("compile: CUDA slice source is missing")?;
                    let source = self.resolve(parent.index())?;
                    let base =
                        self.elementwise_views[parent.index()]
                            .clone()
                            .unwrap_or(ElementwiseView {
                                source,
                                shape: a.shape.clone(),
                                strides: contiguous_strides(&a.shape)?,
                                moduli: vec![0; a.shape.len()],
                                offset: 0,
                            });
                    let mut offset = base.offset;
                    let mut strides = Vec::with_capacity(ranges.len());
                    for ((&(start, _, step), &stride), (&source_size, &result_size)) in ranges
                        .iter()
                        .zip(&base.strides)
                        .zip(a.shape.iter().zip(&node.shape))
                    {
                        if step == 0 || start >= source_size {
                            return Err("compile: CUDA elementwise slice view is invalid".into());
                        }
                        offset = offset
                            .checked_add(
                                start
                                    .checked_mul(stride)
                                    .ok_or("compile: CUDA slice offset overflow")?,
                            )
                            .ok_or("compile: CUDA slice offset overflow")?;
                        strides.push(
                            stride
                                .checked_mul(step)
                                .ok_or("compile: CUDA slice stride overflow")?,
                        );
                        if result_size == 0 {
                            return Err("compile: empty CUDA elementwise slice view".into());
                        }
                    }
                    let view = ElementwiseView {
                        source: base.source,
                        shape: node.shape.clone(),
                        strides,
                        moduli: vec![0; node.shape.len()],
                        offset,
                    };
                    self.semantic_values[dense.index()] = Some(base.source);
                    self.semantic_results[dense.index()] = vec![base.source];
                    self.elementwise_views[dense.index()] = Some(view);
                    return Ok(());
                }
            }
            if let NodeKind::Permute { a, dims } = &node.kind {
                if (fused_consumer
                    || (rms_input
                        && node.dtype != DType::F64
                        && a.shape.len() <= 7
                        && dims.last() == Some(&(a.shape.len() - 1))))
                    && node.dtype == a.dtype
                    && dims.len() == a.shape.len()
                    && dims.iter().all(|&dimension| dimension < a.shape.len())
                {
                    let parent = index
                        .dense_id(a.id)
                        .ok_or("compile: CUDA permute source is missing")?;
                    let source = self.resolve(parent.index())?;
                    let base =
                        self.elementwise_views[parent.index()]
                            .clone()
                            .unwrap_or(ElementwiseView {
                                source,
                                shape: a.shape.clone(),
                                strides: contiguous_strides(&a.shape)?,
                                moduli: vec![0; a.shape.len()],
                                offset: 0,
                            });
                    let strides = dims
                        .iter()
                        .map(|&dimension| base.strides[dimension])
                        .collect::<Vec<_>>();
                    let moduli = dims
                        .iter()
                        .map(|&dimension| base.moduli[dimension])
                        .collect::<Vec<_>>();
                    let view = ElementwiseView {
                        source: base.source,
                        shape: node.shape.clone(),
                        strides,
                        moduli,
                        offset: base.offset,
                    };
                    self.semantic_values[dense.index()] = Some(base.source);
                    self.semantic_results[dense.index()] = vec![base.source];
                    if rms_input {
                        for &consumer in index.consumers_of(dense).into_iter().flatten() {
                            self.rms_row_views[consumer.index()] = Some(view.clone());
                        }
                    }
                    self.elementwise_views[dense.index()] = Some(view);
                    return Ok(());
                }
            }
            if let NodeKind::Reshape { a, .. } = &node.kind {
                let parent = index
                    .dense_id(a.id)
                    .ok_or("compile: CUDA reshape source is missing")?;
                if let Some(view) = self.routed_row_views[parent.index()].clone() {
                    let routed_rows = view
                        .routes
                        .checked_mul(view.rows)
                        .ok_or("compile: CUDA routed-row geometry overflow")?;
                    if node.shape.as_slice() == [routed_rows, view.width] {
                        self.semantic_values[dense.index()] = Some(view.source);
                        self.semantic_results[dense.index()] = vec![view.source];
                        self.routed_row_views[dense.index()] = Some(view);
                        return Ok(());
                    }
                }
                if let Some(view) = fused_consumer
                    .then(|| self.elementwise_views[parent.index()].as_ref())
                    .flatten()
                    .and_then(|view| reshape_view(view, &node.shape))
                {
                    self.semantic_values[dense.index()] = Some(view.source);
                    self.semantic_results[dense.index()] = vec![view.source];
                    self.elementwise_views[dense.index()] = Some(view);
                    return Ok(());
                }
            }
        }
        // A 2D transpose that only feeds a row-oriented BF16 weight position is
        // a physical view. Lowering keeps the original bytes and lets the GEMM
        // consume them through its transpose operand, so no weight transpose
        // copy and no F32 weight conversion is ever materialized.
        if let Instruction::Reindex {
            op: 1,
            a,
            parameters,
        } = &instruction
        {
            if self.gemms.weight_views[dense.index()]
                && parameters.as_slice() == [1, 0]
                && node.shape.len() == 2
            {
                let source = self.resolve(*a)?;
                let id = self.value(
                    node.shape.clone(),
                    node.dtype,
                    boundary_storage,
                    "cublas_transpose_view",
                    ValueStorage::Alias {
                        source,
                        byte_offset: 0,
                    },
                )?;
                self.emit(
                    "cublas_transpose_view",
                    CommandKind::Alias { source },
                    Some(id),
                    vec![ValueUse::read(source)],
                    false,
                )?;
                self.semantic_values[dense.index()] = Some(id);
                self.semantic_results[dense.index()] = vec![id];
                return Ok(());
            }
        }
        if let Some((parent, selected)) = selected_result(&node.kind) {
            let parent = index
                .dense_id(parent.id)
                .ok_or("compile: missing selected producer")?;
            let source = *self.semantic_results[parent.index()]
                .get(selected as usize)
                .ok_or("compile: CUDA producer result missing")?;
            self.semantic_values[dense.index()] = Some(source);
            self.semantic_results[dense.index()] = vec![source];
            return Ok(());
        }
        let native_gemm = self.gemms.operations[dense.index()];
        let instruction = match instruction {
            Instruction::Concat { a, b, dim } => {
                if let Some((source, width)) = rotary_reindex_source(node, index) {
                    Instruction::RotaryReindex {
                        x: source.index(),
                        width,
                    }
                } else {
                    Instruction::Concat { a, b, dim }
                }
            }
            instruction => instruction,
        };
        let output = match instruction {
            Instruction::GroupedExpertLinearRows { x, weight, indexes } => {
                self.grouped_expert(x, weight, indexes)?
            }
            Instruction::Linear { x, bias, .. } if native_gemm.is_some() => {
                self.row_major_gemm(native_gemm.unwrap(), x, node.shape.clone(), Some(bias))?
            }
            Instruction::Matmul { a, .. } if native_gemm.is_some() => {
                self.row_major_gemm(native_gemm.unwrap(), a, node.shape.clone(), None)?
            }
            Instruction::Value(value) => {
                let id = self.value(
                    node.shape.clone(),
                    node.dtype,
                    boundary_storage,
                    "constant",
                    ValueStorage::Fixed {
                        class: StorageClass::PersistentConstant,
                        location: Location::Persistent {
                            slot: self.values.len() as u32,
                        },
                    },
                )?;
                self.emit(
                    "value",
                    CommandKind::Value(value),
                    Some(id),
                    Vec::new(),
                    false,
                )?;
                id
            }
            Instruction::Input {
                binding,
                scalar: false,
                ..
            } => {
                let id = self.value(
                    node.shape.clone(),
                    node.dtype,
                    boundary_storage,
                    "input",
                    ValueStorage::Fixed {
                        class: StorageClass::ExternalInput,
                        location: Location::External {
                            slot: u32::try_from(binding)
                                .map_err(|_| "compile: too many bindings")?,
                        },
                    },
                )?;
                self.emit(
                    "input",
                    CommandKind::Input { binding },
                    Some(id),
                    Vec::new(),
                    false,
                )?;
                id
            }
            Instruction::Input {
                binding,
                scalar: true,
                ..
            } => {
                let id = self.planned(node.shape.clone(), node.dtype, "scalar")?;
                self.emit(
                    "scalar",
                    CommandKind::Scalar { binding },
                    Some(id),
                    Vec::new(),
                    true,
                )?;
                id
            }
            Instruction::StateCursor { tensor, .. } => {
                let id = self.planned(node.shape.clone(), node.dtype, "cursor")?;
                self.emit(
                    "cursor",
                    CommandKind::Cursor { tensor },
                    Some(id),
                    Vec::new(),
                    true,
                )?;
                id
            }
            Instruction::Alias { a, .. } => {
                let source = self.resolve(a)?;
                if boundary_storage.representation != StorageRepresentation::Dense {
                    return Err("compile: packed CUDA aliases are unsupported".into());
                }
                let id = self.value(
                    node.shape.clone(),
                    node.dtype,
                    boundary_storage,
                    "alias",
                    ValueStorage::Alias {
                        source,
                        byte_offset: 0,
                    },
                )?;
                self.emit(
                    "alias",
                    CommandKind::Alias { source },
                    Some(id),
                    vec![ValueUse::read(source)],
                    false,
                )?;
                id
            }
            instruction => {
                let mut spec = instruction.kernel()?;
                let mut inputs = [None; 8];
                for (slot, semantic) in spec.inputs.iter().enumerate() {
                    if let Some(semantic) = semantic {
                        inputs[slot] = Some(self.resolve(*semantic)?);
                    }
                }
                let children = node_children(&node.kind);
                for operand in &operation.operands {
                    let child = children
                        .get(operand.index as usize)
                        .ok_or("compile: operand missing")?;
                    let semantic = index
                        .dense_id(child.id)
                        .ok_or("compile: operand source missing")?
                        .index();
                    let mut source = self.resolve(semantic)?;
                    if let OperandInterpretation::ScalarCoercion(conversion) =
                        operand.interpretation
                    {
                        if self.values[source.index()].dtype != conversion.source {
                            return Err("compile: scalar coercion source mismatch".into());
                        }
                        source = self.conversion(source, conversion.destination)?;
                    }
                    if let OperandPreparation::Convert(conversion) = operand.preparation {
                        if self.values[source.index()].dtype != conversion.source {
                            return Err("compile: conversion source mismatch".into());
                        }
                        if !supports_inline_conversion(spec.name) {
                            source = self.conversion(source, conversion.destination)?;
                        }
                    }
                    for (slot, input) in spec.inputs.iter().enumerate() {
                        if *input == Some(semantic) {
                            inputs[slot] = Some(source);
                        }
                    }
                }
                if spec.name == "et_rms_norm" {
                    if let Some(view) = &self.rms_row_views[dense.index()] {
                        let width = *view
                            .shape
                            .last()
                            .ok_or("compile: CUDA RMS view rank is zero")?;
                        let outer_rank = view.shape.len() - 1;
                        if width == 0
                            || view.strides.last() != Some(&1)
                            || view.offset % width != 0
                            || view.strides[..outer_rank]
                                .iter()
                                .any(|stride| stride % width != 0)
                        {
                            return Err("compile: CUDA RMS row view is invalid".into());
                        }
                        spec.args.integers[1] = outer_rank as u64;
                        spec.args.integers[2] = (view.offset / width) as u64;
                        for dimension in 0..outer_rank {
                            spec.args.integers[3 + dimension] = view.shape[dimension] as u64;
                            spec.args.integers[3 + outer_rank + dimension] =
                                (view.strides[dimension] / width) as u64;
                        }
                    }
                }
                if let Some((compact, routes, inner)) = compact_inner_scatter_indexes(node, index)?
                {
                    spec.name = "et_scatter_add_inner";
                    spec.args.integers[0] = routes as u64;
                    spec.args.integers[1] = inner as u64;
                    inputs[1] = Some(self.resolve(compact.index())?);
                }
                let semantic_spec = OperationDTypeSpec::new(index, dense)?;
                let mut results = Vec::with_capacity(operation.results.len());
                for (position, result) in operation.results.iter().enumerate() {
                    let boundary = &semantic_spec.results[position].value;
                    let inline_destination = match result.completion {
                        ResultCompletion::ConvertToBoundary(conversion)
                            if supports_inline_conversion(spec.name) =>
                        {
                            if result.execution_dtype != conversion.source
                                || boundary.semantic_dtype != conversion.destination
                            {
                                return Err("compile: CUDA boundary dtype mismatch".into());
                            }
                            Some(conversion.destination)
                        }
                        _ => None,
                    };
                    let id = self.planned(
                        boundary.logical_shape.to_vec(),
                        inline_destination.unwrap_or(result.execution_dtype),
                        "result",
                    )?;
                    let mut selected = spec.clone();
                    if operation.results.len() > 1 {
                        match selected.name {
                            "et_optimizer" => selected.args.integers[0] = position as u64,
                            "et_kda" | "et_sdpa" => selected.args.operation = position as u32,
                            "et_layer_norm" | "et_chunked_head_ce" => {
                                selected.args.operation = position as u32 + 1
                            }
                            _ => {
                                return Err(
                                    "compile: CUDA multi-result family has no selector".into()
                                )
                            }
                        }
                    }
                    if selected.name == "et_kv_attention" {
                        selected.args.integers[8] = u64::from(dtype_code(node.dtype));
                    }
                    selected.args.compute_dtype =
                        dtype_code(operation.compute_dtype.unwrap_or(result.execution_dtype));
                    self.kernel(selected, inputs, id)?;
                    results.push(match result.completion {
                        ResultCompletion::Direct => id,
                        ResultCompletion::ConvertToBoundary(conversion) => {
                            if result.execution_dtype != conversion.source
                                || boundary.semantic_dtype != conversion.destination
                            {
                                return Err("compile: CUDA boundary dtype mismatch".into());
                            }
                            if inline_destination.is_some() {
                                id
                            } else {
                                self.conversion(id, conversion.destination)?
                            }
                        }
                    });
                }
                let first = results[0];
                self.semantic_results[dense.index()] = results;
                first
            }
        };
        self.semantic_values[dense.index()] = Some(output);
        if self.semantic_results[dense.index()].is_empty() {
            self.semantic_results[dense.index()] = vec![output];
        }
        Ok(())
    }

    pub(super) fn add_region(
        &mut self,
        index: &GraphIndex,
        optimization: &OptimizationPlan,
        region_id: RegionId,
        dtype_plan: &ExecutableDTypePlan,
        device: &CudaDevice,
    ) -> Result<(), String> {
        if !matches!(
            dtype_plan.execution().realization,
            ExecutionRealization::DirectKernel | ExecutionRealization::KernelLocal
        ) {
            return Err("compile: CUDA region requires direct or kernel-local execution".into());
        }
        let native = optimization
            .regions
            .get(region_id.index())
            .ok_or_else(|| format!("compile: CUDA region {region_id} is out of range"))?;
        let NativeRegion::Elementwise(region) = native else {
            return Err("compile: unsupported CUDA optimization region".into());
        };
        if !region.device.is_cuda() {
            return Err("compile: CUDA region targets another device".into());
        }
        if region.inputs.len() > 8 {
            return Err("compile: CUDA fused region has more than eight inputs".into());
        }
        let expressions = legalize_region_expressions(native, dtype_plan)?;
        let expression = expressions
            .first()
            .ok_or("compile: CUDA fused region has no expression")?;
        let mut inputs = [None; 8];
        let mut lane_strides = region.lane_strides.to_vec();
        let mut lane_moduli =
            vec![vec![0; region.shape.len()].into_boxed_slice(); region.inputs.len()];
        let mut lane_offsets = vec![0; region.inputs.len()];
        for (slot, semantic) in region.inputs.iter().enumerate() {
            if let Some(view) = &self.elementwise_views[semantic.index()] {
                if view.shape != region.shape.as_ref()
                    || lane_strides[slot].as_ref() != contiguous_strides(&view.shape)?.as_slice()
                {
                    return Err(
                        "compile: CUDA elementwise view requires an unbroadcast region input"
                            .into(),
                    );
                }
                inputs[slot] = Some(view.source);
                lane_strides[slot] = view.strides.clone().into_boxed_slice();
                lane_moduli[slot] = view.moduli.clone().into_boxed_slice();
                lane_offsets[slot] = view.offset;
            } else {
                inputs[slot] = Some(self.resolve(semantic.index())?);
            }
        }
        let output = self.planned(region.shape.to_vec(), region.dtype, "fused_elementwise")?;
        let source = crate::emit::elementwise(
            expression,
            &lane_strides,
            &lane_moduli,
            &lane_offsets,
            &region.shape,
        )?;
        let function = device.fused_elementwise(&source)?;
        let mut args = CudaKernelArgs {
            elements: product_checked(&region.shape)? as u64,
            output_dtype: dtype_code(region.dtype),
            compute_dtype: dtype_code(DType::F32),
            ..Default::default()
        };
        for (slot, value) in inputs.iter().enumerate() {
            if let Some(value) = value {
                args.input_dtypes[slot] = dtype_code(self.values[value.index()].dtype);
            }
        }
        self.emit(
            "fused_elementwise",
            CommandKind::FusedElementwise {
                function,
                args,
                inputs,
            },
            Some(output),
            inputs
                .iter()
                .flatten()
                .copied()
                .map(ValueUse::read)
                .collect(),
            true,
        )?;
        let semantic = region.output.semantic_node;
        self.semantic_values[semantic.index()] = Some(output);
        self.semantic_results[semantic.index()] = vec![output];
        let node = index
            .node(semantic)
            .ok_or("compile: CUDA fused output semantic node is missing")?;
        if node.shape.as_slice() != region.shape.as_ref() || node.dtype != region.dtype {
            return Err("compile: CUDA fused output metadata mismatch".into());
        }
        Ok(())
    }

    fn emit_gemm(
        &mut self,
        plan: &Bf16GemmPlan,
        weight_transposed: bool,
        x: ValueId,
        weight: ValueId,
        output: ValueId,
        out_f32: bool,
    ) -> Result<(), String> {
        let workspace =
            self.planned(vec![CUBLAS_WORKSPACE_BYTES], DType::U8, "cublas_workspace")?;
        self.emit(
            "prepare",
            CommandKind::Prepare,
            Some(workspace),
            Vec::new(),
            true,
        )?;
        let id = InstructionId::from_index(self.lowered.len())
            .ok_or("compile: too many CUDA instructions")?;
        self.lowered.push(
            LoweredInstruction::new(
                id,
                "cublas_bf16_gemm_f32_accum",
                vec![ValueUse::read(x), ValueUse::read(weight)],
                vec![OutputDecl::new(output)],
            )
            .with_resources(
                vec![ValueUse::read_write(workspace)],
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
            .with_effects(InstructionEffects {
                may_fail: true,
                has_side_effects: false,
            }),
        );
        self.commands.push(Command {
            output: Some(output),
            kind: CommandKind::Gemm {
                x,
                weight,
                weight_transposed,
                plan: *plan,
                out_f32,
                workspace,
            },
        });
        Ok(())
    }

    fn grouped_expert(
        &mut self,
        x: usize,
        weight: usize,
        indexes: usize,
    ) -> Result<ValueId, String> {
        let routed = self.routed_row_views[x].clone();
        let indexes_semantic = indexes;
        let (x, weight, indexes) = (
            self.resolve(x)?,
            self.resolve(weight)?,
            self.resolve(indexes)?,
        );
        let dtype = self.values[x.index()].dtype;
        let (rows, inner, source_rows) = if let Some(view) = routed {
            (
                view.routes
                    .checked_mul(view.rows)
                    .ok_or("compile: CUDA routed-row count overflow")?,
                view.width,
                view.rows,
            )
        } else {
            (
                self.values[x.index()].shape[0],
                self.values[x.index()].shape[1],
                self.values[x.index()].shape[0],
            )
        };
        let experts = self.values[weight.index()].shape[0];
        let columns = self.values[weight.index()].shape[1];
        let routing_key = (indexes_semantic, rows, experts);
        let previous_routing = self.grouped_routing.get(&routing_key).copied();
        let (control, row_map, reuse_routing) = if let Some((control, row_map)) = previous_routing {
            (control, row_map, true)
        } else {
            (
                self.planned(
                    vec![experts.checked_add(2).ok_or("group control overflow")?],
                    DType::U32,
                    "group_control",
                )?,
                self.planned(vec![rows], DType::U32, "group_rows")?,
                false,
            )
        };
        let gathered = self.planned(vec![rows, inner], dtype, "group_input")?;
        let projected = self.planned(vec![rows, columns], dtype, "group_output")?;
        let workspace = if dtype == DType::BF16 && rows != 0 && inner != 0 && columns != 0 {
            Some(self.planned(
                vec![CUBLAS_WORKSPACE_BYTES * EXPERT_BLAS_STREAMS],
                DType::U8,
                "cublas_workspace",
            )?)
        } else {
            None
        };
        let owned_scratch = [
            (!reuse_routing).then_some(control),
            (!reuse_routing).then_some(row_map),
            Some(gathered),
            Some(projected),
            workspace,
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        for &id in &owned_scratch {
            self.emit("prepare", CommandKind::Prepare, Some(id), Vec::new(), true)?;
        }
        let scratch = [
            Some(control),
            Some(row_map),
            Some(gathered),
            Some(projected),
            workspace,
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        let output = self.planned(vec![rows, columns], dtype, "result")?;
        let id = InstructionId::from_index(self.lowered.len())
            .ok_or("compile: too many CUDA instructions")?;
        self.lowered.push(
            LoweredInstruction::new(
                id,
                "grouped_expert_linear_rows_host_control_non_capturable",
                vec![
                    ValueUse::read(x),
                    ValueUse::read(weight),
                    ValueUse::read(indexes),
                ],
                vec![OutputDecl::new(output)],
            )
            .with_resources(
                scratch
                    .into_iter()
                    .map(ValueUse::read_write)
                    .collect::<Vec<_>>(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
            .with_effects(InstructionEffects {
                may_fail: true,
                has_side_effects: false,
            }),
        );
        self.commands.push(Command {
            output: Some(output),
            kind: CommandKind::GroupedExpert {
                x,
                weight,
                indexes,
                rows,
                columns,
                inner,
                experts,
                control,
                row_map,
                gathered,
                projected,
                workspace,
                reuse_routing,
                source_rows,
            },
        });
        if !reuse_routing && rows != 0 && inner != 0 && columns != 0 {
            self.grouped_routing.insert(routing_key, (control, row_map));
        }
        Ok(output)
    }

    /// Native BF16 row-major GEMM for a Linear or Matmul.
    fn row_major_gemm(
        &mut self,
        native: NativeGemm,
        activation: usize,
        output_shape: Vec<usize>,
        bias: Option<usize>,
    ) -> Result<ValueId, String> {
        let x = self.resolve(activation)?;
        let w = self.resolve(native.weight)?;
        let plan = native.geometry;
        let weight_transposed = native.weight_transposed;
        if let Some(bias) = bias {
            // Add bias before the single BF16 rounding boundary. The GEMM
            // accumulates into F32 and the bias kernel performs the only
            // rounding to BF16.
            let accumulator =
                self.planned(output_shape.clone(), DType::F32, "linear_accumulator")?;
            self.emit_gemm(&plan, weight_transposed, x, w, accumulator, true)?;
            let bias = self.resolve(bias)?;
            let mut args = CudaKernelArgs {
                elements: product_checked(&output_shape)? as u64,
                compute_dtype: dtype_code(DType::F32),
                output_dtype: dtype_code(DType::BF16),
                ..Default::default()
            };
            args.integers[0] = *output_shape
                .last()
                .ok_or("compile: CUDA linear result has no columns")?
                as u64;
            args.input_dtypes[0] = dtype_code(DType::F32);
            args.input_dtypes[1] = dtype_code(DType::BF16);
            let result = self.planned(output_shape, DType::BF16, "result")?;
            self.emit(
                BF16_LINEAR_BIAS_KERNEL,
                CommandKind::LinearBias {
                    accumulator,
                    bias,
                    args,
                },
                Some(result),
                vec![ValueUse::read(accumulator), ValueUse::read(bias)],
                true,
            )?;
            return Ok(result);
        }
        let result = self.planned(output_shape, DType::BF16, "result")?;
        self.emit_gemm(&plan, weight_transposed, x, w, result, false)?;
        Ok(result)
    }

    pub(super) fn finish(
        mut self,
        index: &GraphIndex,
    ) -> Result<(CudaLoweredProgram, Vec<Command>), String> {
        let mut transactions = self
            .state_values
            .iter()
            .filter_map(|(component, (persistent, transaction))| {
                transaction.map(|transaction| (*component, *persistent, transaction))
            })
            .collect::<Vec<_>>();
        transactions.sort_by_key(|(_, persistent, _)| *persistent);
        for (_component, persistent, transaction) in transactions {
            let id = InstructionId::from_index(self.lowered.len())
                .ok_or("compile: too many state commits")?;
            let append = self.state_layout.as_ref().is_some_and(|layout| {
                layout.access == effect_torch_runtime::StateAccessMode::Append
            });
            self.lowered.push(
                LoweredInstruction::new(
                    id,
                    if append {
                        "state_commit"
                    } else {
                        "state_discard"
                    },
                    Vec::new(),
                    Vec::new(),
                )
                .with_resources(
                    Vec::new(),
                    vec![ValueUse::read(transaction)],
                    Vec::new(),
                    if append {
                        vec![ValueUse::write(persistent)]
                    } else {
                        Vec::new()
                    },
                )
                .with_effects(InstructionEffects {
                    may_fail: false,
                    has_side_effects: append,
                }),
            );
        }
        if let Some(status) = self.status {
            let id = InstructionId::from_index(self.lowered.len())
                .ok_or("compile: too many CUDA instructions")?;
            self.lowered.push(
                LoweredInstruction::new(
                    id,
                    "status_check",
                    vec![ValueUse::read(status)],
                    Vec::new(),
                )
                .with_effects(InstructionEffects {
                    may_fail: true,
                    has_side_effects: false,
                }),
            );
        }
        let outputs = index
            .roots
            .iter()
            .map(|root| self.resolve(root.index()))
            .collect::<Result<Vec<_>, _>>()?;
        for root in &outputs {
            let mut source = *root;
            while let ValueStorage::Alias { source: parent, .. } =
                self.values[source.index()].decl.storage
            {
                source = parent;
            }
            if let ValueStorage::Planned {
                class, ownership, ..
            } = &mut self.values[source.index()].decl.storage
            {
                *class = StorageClass::EscapingOutput;
                *ownership = SegmentOwnership::ProvisionalOutput;
            }
        }
        Ok((
            LoweredProgram::new(self.values, self.lowered, outputs),
            self.commands,
        ))
    }
}

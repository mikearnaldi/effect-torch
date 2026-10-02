//! CUDA lowering with independent semantic, instruction, and materialized value IDs.
use crate::cublas::{
    plan_row_bf16_gemm, Bf16GemmPlan, RowGemmKind, CUBLAS_WORKSPACE_BYTES, EXPERT_BLAS_STREAMS,
    EXPERT_GROUPED_POINTER_BYTES, EXPERT_PARTIAL_POINTER_BYTES, EXPERT_SPLITK_DESCRIPTOR_BYTES,
    EXPERT_SPLITK_MAX_SPLITS,
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
use std::sync::Arc;

pub(super) type CudaLoweredProgram = LoweredProgram<&'static str, CudaMemorySpace, CudaValueMeta>;
pub(super) const BF16_LINEAR_BIAS_KERNEL: &str = "et_linear_bias_f32";

#[cfg(test)]
#[path = "executable_tests.rs"]
mod tests;

#[path = "rotary_reuse66.rs"]
pub(crate) mod rotary_reuse66;

#[path = "packed_projection77.rs"]
mod packed_projection77;

#[path = "normrope101_lowering.rs"]
mod normrope101_lowering;

#[path = "literal_leaf88.rs"]
pub(super) mod literal_leaf88;

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
    Softmax100 {
        source: ValueId,
        scratch: ValueId,
        plan: Arc<crate::triton_softmax100::Softmax100>,
    },
    NormRope101 {
        packed: ValueId,
        borrowed: [ValueId; 4],
        table_strides: [u32; 2],
        scratch: [ValueId; 7],
        outputs: [ValueId; 3],
        plan: Arc<crate::triton_normrope101::NormRope101>,
    },
    Norm98Entrance {
        inputs: [ValueId; 6],
        sum_a: ValueId,
        rows: u32,
        rho: Arc<crate::triton_norm98::Rho98>,
        plan: Arc<crate::triton_norm98::Norm98>,
    },
    Norm98Tail {
        inputs: [ValueId; 8],
        entrance: ValueId,
        sum_a: ValueId,
        attention_weight: ValueId,
        hidden: ValueId,
        combined: ValueId,
        rows: u32,
        plan: Arc<crate::triton_norm98::Norm98>,
    },
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
    /// Slice of an allocation owned by this invocation's memory plan.
    PlannedAlias,
    Gemm {
        x: ValueId,
        weight: ValueId,
        weight_transposed: bool,
        plan: Bf16GemmPlan,
        out_f32: bool,
        workspace: ValueId,
    },
    GemmPair {
        x: ValueId,
        weights: [ValueId; 2],
        second_output: ValueId,
        plan: Bf16GemmPlan,
        workspaces: [ValueId; 2],
    },
    PackedProjection77 {
        /// Only admitted private101 groups consume the packed temporary directly.
        skip_split: bool,
        x: ValueId,
        weight: ValueId,
        outputs: Vec<ValueId>,
        widths: Vec<usize>,
        temporary: ValueId,
        workspace: ValueId,
        plan: Bf16GemmPlan,
    },
    /// Model-only relaxed BF16 MoE; all scratch is invocation-planned.
    FusedMoe75 {
        x: ValueId,
        indexes: ValueId,
        scales: ValueId,
        down: ValueId,
        workspace: ValueId,
        map: ValueId,
        routes: ValueId,
        status: ValueId,
        plan: Arc<crate::fused_moe75::Plan>,
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
        splitk_workspace: bool,
        source_rows: usize,
        /// Private region values only; the same stable routing must already exist.
        input_sorted: bool,
        /// Publish the projected buffer as a private physical value, omitting scatter.
        output_sorted: bool,
        inverse_routing: bool,
        device_control: Option<(ValueId, ValueId)>,
    },
    /// Infallible numerical epilogue with by-value geometry and no status I/O.
    LinearBias {
        accumulator: ValueId,
        bias: ValueId,
        args: CudaKernelArgs,
    },
    FusedElementwise {
        wide_sum: bool,
        wide_arg: bool,
        function: CudaFunction,
        graph61_function: Option<Arc<crate::explicit_graph61::RawKernel61>>,
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum CommandOverlap {
    #[default]
    Primary,
    Worker {
        branch: usize,
        start: bool,
        finish: bool,
    },
    Join {
        branch: usize,
    },
}

pub(super) struct Command {
    pub(super) output: Option<ValueId>,
    pub(super) kind: CommandKind,
    pub(super) overlap: CommandOverlap,
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

fn grouped_workspace_bytes(rows: usize, columns: usize, splitk: bool) -> Result<usize, String> {
    let ordinary = CUBLAS_WORKSPACE_BYTES
        .checked_mul(EXPERT_BLAS_STREAMS)
        .and_then(|bytes| bytes.checked_add(EXPERT_GROUPED_POINTER_BYTES))
        .ok_or("compile: CUDA grouped workspace overflow")?;
    if !splitk {
        return Ok(ordinary);
    }
    rows.checked_mul(columns)
        .and_then(|elements| elements.checked_mul(EXPERT_SPLITK_MAX_SPLITS))
        .and_then(|elements| elements.checked_mul(2))
        .and_then(|bytes| bytes.checked_add(EXPERT_SPLITK_DESCRIPTOR_BYTES))
        .and_then(|bytes| bytes.checked_add(EXPERT_PARTIAL_POINTER_BYTES))
        .and_then(|bytes| bytes.checked_add(ordinary))
        .ok_or_else(|| "compile: CUDA split-K workspace overflow".to_owned())
}

// Zero selects the generic rank/stride descriptor. These special cases only
// change indexing; typed loads and scalar coercions keep their existing path.
fn binary_source_geometry(output: &[usize], input: &[usize]) -> (u64, u64) {
    if output == input {
        return (1, 0);
    }
    if input.iter().all(|dimension| *dimension == 1) {
        return (2, 0);
    }
    let Some(&width) = output.last() else {
        return (0, 0);
    };
    if width == 0 || input.len() > output.len() || input.last() != Some(&1) {
        return (0, 0);
    }
    let leading = output.len() - input.len();
    if output[..output.len() - 1]
        .iter()
        .enumerate()
        .all(|(axis, dimension)| {
            *dimension == axis.checked_sub(leading).map_or(1, |axis| input[axis])
        })
    {
        if width.is_power_of_two() {
            return (4, u64::from(width.trailing_zeros()));
        }
        return (3, width as u64);
    }
    (0, 0)
}

#[cfg(test)]
mod metadata_tests {
    use super::{binary_source_geometry, grouped_workspace_bytes};
    use crate::cublas::{
        CUBLAS_WORKSPACE_BYTES, EXPERT_BLAS_STREAMS, EXPERT_GROUPED_POINTER_BYTES,
        EXPERT_PARTIAL_POINTER_BYTES, EXPERT_SPLITK_DESCRIPTOR_BYTES,
    };

    #[test]
    fn binary_metadata_preserves_broadcast_geometry() {
        assert_eq!(binary_source_geometry(&[2, 8], &[2, 8]), (1, 0));
        assert_eq!(binary_source_geometry(&[2, 8], &[1, 1]), (2, 0));
        assert_eq!(binary_source_geometry(&[2, 8], &[2, 1]), (4, 3));
        assert_eq!(binary_source_geometry(&[2, 7], &[2, 1]), (3, 7));
        assert_eq!(binary_source_geometry(&[2, 8], &[8]), (0, 0));
        assert_eq!(binary_source_geometry(&[2, 3, 8], &[3, 1]), (0, 0));
        assert_eq!(binary_source_geometry(&[1, 3, 8], &[3, 1]), (4, 3));
        assert_eq!(binary_source_geometry(&[2, 0], &[2, 1]), (0, 0));
    }

    #[test]
    fn splitk_workspace_is_opt_in_and_checked() {
        let ordinary = CUBLAS_WORKSPACE_BYTES * EXPERT_BLAS_STREAMS + EXPERT_GROUPED_POINTER_BYTES;
        assert_eq!(grouped_workspace_bytes(65, 1408, false).unwrap(), ordinary);
        assert_eq!(
            grouped_workspace_bytes(65, 1408, true).unwrap(),
            ordinary
                + EXPERT_SPLITK_DESCRIPTOR_BYTES
                + EXPERT_PARTIAL_POINTER_BYTES
                + 8 * 65 * 1408 * 2
        );
        assert!(grouped_workspace_bytes(usize::MAX, 1408, true).is_err());
        assert_eq!(
            grouped_workspace_bytes(usize::MAX, 1408, false).unwrap(),
            ordinary
        );
    }
}

#[cfg(test)]
thread_local! {
    static KV_BF16_INPUT_TEST_POLICY: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    static ATTENTION82_TEST_POLICY: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_attention82_test_policy<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            ATTENTION82_TEST_POLICY.with(|policy| policy.set(self.0));
        }
    }
    let _restore = Restore(ATTENTION82_TEST_POLICY.with(|policy| policy.replace(Some(enabled))));
    run()
}

#[cfg(test)]
pub(crate) fn with_kv_bf16_input_test_policy<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            KV_BF16_INPUT_TEST_POLICY.with(|policy| policy.set(self.0));
        }
    }
    let _restore = Restore(KV_BF16_INPUT_TEST_POLICY.with(|policy| policy.replace(Some(enabled))));
    run()
}

fn mean256_eligible(node: &Node, enabled: bool) -> bool {
    if !enabled || node.dtype != DType::F32 || !node.shape.is_empty() {
        return false;
    }
    matches!(&node.kind, NodeKind::Mean { a, dims, keepdims }
        if !keepdims && a.dtype == DType::F32 && a.shape.as_slice() == [1, 256]
            && a.storage.representation == StorageRepresentation::Dense
            && dims.as_slice() == [0, 1])
}

fn kv_bf16_inputs_eligible(
    node: &Node,
    layout: Option<&CudaStateLayout>,
    enabled: bool,
    bf16_kv_matmul: bool,
) -> bool {
    if !enabled || !bf16_kv_matmul || node.dtype != DType::BF16 {
        return false;
    }
    let NodeKind::KvAttention {
        q,
        k,
        v,
        layer,
        rounding,
        ..
    } = &node.kind
    else {
        return false;
    };
    *rounding == effect_torch_graph::AttentionRounding::Stepwise
        && [q, k, v].iter().all(|value| {
            value.dtype == DType::BF16
                && value.storage.representation == StorageRepresentation::Dense
                && value.shape.len() == 4
                && value
                    .shape
                    .iter()
                    .all(|&n| n > 0 && i32::try_from(n).is_ok())
        })
        && layout.is_some_and(|layout| {
            layout.dtype == DType::BF16
                && layout.kv_layers.iter().any(|descriptor| {
                    descriptor.layer_id == *layer
                        && descriptor.dtype == DType::BF16
                        && descriptor.kv_heads == k.shape[1]
                        && descriptor.head_dim == q.shape[3]
                })
        })
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
        "et_dual_argmax"
        | "et_convert"
        | "et_expert_route_rank"
        | "et_rms_residual_bf16"
        | "et_ffn_tail_bf16"
        | "et_ffn_next_norm63_bf16"
        | "et_unary"
        | "et_where"
        | "et_reindex"
        | "et_concat"
        | "et_norm_rope_bf16"
        | "et_rms_norm_f32"
        | "et_rms_norm_f64"
        | "et_entropy_max"
        | "et_entropy_sum"
        | "et_entropy_normalized_max"
        | "et_entropy_normalized_sum"
        | "et_entropy_finish"
        | "et_entropy_relaxed81_moment_finish"
        | "et_small_softmax_f32"
        | "et_bf16_softmax_prepare"
        | "et_bf16_softmax_store"
        | "et_shared_rms_norm_f32"
        | "et_random_f32"
        | "et_random_f64"
        | "et_sequence"
        | "et_reduce_f32"
        | "et_reduce_f64" => false,
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

struct Norm98Candidate {
    command: usize,
    instruction: usize,
    prepare_instruction: usize,
    packed: ValueId,
    inputs: [ValueId; 6],
    rho: f32,
}

pub(super) struct CudaProgramBuilder {
    pub(super) softmax100: Option<Arc<crate::triton_softmax100::Softmax100>>,
    pub(super) norm98: Option<(Arc<crate::triton_norm98::Norm98>, Arc<CudaDevice>)>,
    norm98_candidates: std::collections::HashMap<DenseNodeId, Norm98Candidate>,
    norm98_rhos: std::collections::HashMap<u32, Arc<crate::triton_norm98::Rho98>>,
    pub(super) allow_fused_moe75: bool,
    pub(super) values: Vec<CudaValueMeta>,
    pub(super) commands: Vec<Command>,
    lowered: Vec<LoweredInstruction<&'static str>>,
    semantic_values: Vec<Option<ValueId>>,
    semantic_results: Vec<Vec<ValueId>>,
    elementwise_views: Vec<Option<ElementwiseView>>,
    rms_row_views: Vec<Option<ElementwiseView>>,
    sequence_major_attention: Vec<bool>,
    routed_row_views: Vec<Option<RoutedRowsView>>,
    grouped_routing:
        std::collections::HashMap<(usize, usize, usize, bool, bool), (ValueId, ValueId)>,
    state_layout: Option<CudaStateLayout>,
    state_values: std::collections::HashMap<StateComponent, (ValueId, Option<ValueId>)>,
    status: Option<ValueId>,
    pub(super) conversion_count: usize,
    pub(super) conversion_bytes: usize,
    gemms: CudaGemmLoweringPlan,
    bf16_kv_matmul: bool,
    typed_binary: bool,
    kv_bf16_inputs: bool,
    attention82: bool,
    mean256: bool,
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
        let kv_bf16_inputs = legalization
            .target()
            .features
            .iter()
            .any(|feature| feature == "kv-bf16-inputs-f32-output-v1-true");
        #[cfg(test)]
        let kv_bf16_inputs =
            KV_BF16_INPUT_TEST_POLICY.with(|policy| policy.get().unwrap_or(kv_bf16_inputs));
        let attention82 = legalization
            .target()
            .features
            .iter()
            .any(|feature| feature == "attention82-bf16-io-v1-true");
        #[cfg(test)]
        let attention82 =
            ATTENTION82_TEST_POLICY.with(|policy| policy.get().unwrap_or(attention82));
        Ok(Self {
            softmax100: None,
            allow_fused_moe75: false,
            norm98: None,
            norm98_candidates: std::collections::HashMap::new(),
            norm98_rhos: std::collections::HashMap::new(),
            kv_bf16_inputs,
            attention82,
            mean256: legalization
                .target()
                .features
                .iter()
                .any(|feature| feature == "mean256-contiguous-f32-sequential-v1-true"),
            values: Vec::new(),
            commands: Vec::new(),
            lowered: Vec::new(),
            semantic_values: vec![None; index.order.len()],
            semantic_results: vec![Vec::new(); index.order.len()],
            elementwise_views: vec![None; index.order.len()],
            rms_row_views: vec![None; index.order.len()],
            sequence_major_attention: vec![false; index.order.len()],
            routed_row_views: vec![None; index.order.len()],
            grouped_routing: std::collections::HashMap::new(),
            state_layout,
            state_values: std::collections::HashMap::new(),
            status: None,
            conversion_count: 0,
            conversion_bytes: 0,
            gemms: CudaGemmLoweringPlan::new(index, legalization)?,
            typed_binary: legalization
                .target()
                .features
                .iter()
                .any(|f| f == "typed-binary-static-op-dtype-dense-scalar-v1-true"),
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
                    return match optimization.regions.get(region.index()) {
                        Some(NativeRegion::Elementwise(_)) => true,
                        Some(NativeRegion::NormRope(region)) => {
                            dense == region.cosine || dense == region.sine
                        }
                        _ => false,
                    };
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
                        match optimization.regions.get(region.index()) {
                            Some(NativeRegion::Elementwise(_)) => true,
                            Some(NativeRegion::NormRope(region)) => {
                                dense == region.cosine || dense == region.sine
                            }
                            _ => false,
                        }
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

    fn feeds_only_sequence_major_permute(index: &GraphIndex, dense: DenseNodeId) -> bool {
        let Some(producer) = index.node(dense) else {
            return false;
        };
        if producer.shape.len() != 4 || index.roots.iter().any(|root| root.index() == dense.index())
        {
            return false;
        }
        index.consumers_of(dense).is_some_and(|consumers| {
            !consumers.is_empty()
                && consumers.iter().all(|consumer| {
                    matches!(
                        index.node(*consumer).map(|node| &node.kind),
                        Some(NodeKind::Permute { a, dims })
                            if a.id == producer.id && dims.as_slice() == [0, 2, 1, 3]
                    )
                })
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
        self.commands.push(Command {
            output,
            kind,
            overlap: CommandOverlap::Primary,
        });
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
                    && (self.values[output.index()].dtype == DType::F32
                        || (self.attention82 && self.values[output.index()].dtype == DType::BF16))
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
            overlap: CommandOverlap::Primary,
        });
        let out = &self.values[output.index()];
        spec.args.elements = crate::value::element_count(&out.shape)? as u64;
        if spec.name == "et_binary" {
            for role in 0..2 {
                let input = inputs[role].ok_or("compile: binary input is missing")?;
                let (mode, parameter) =
                    binary_source_geometry(&out.shape, &self.values[input.index()].shape);
                spec.args.integers[2 + role] = mode;
                spec.args.integers[4 + role] = parameter;
            }
        }
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
        // Status and inline-coercion planning above deliberately used the
        // original name. Select only after physical dtype/geometry is complete.
        if let Some(name) =
            crate::typed_binary::typed_binary_kernel(spec.name, &spec.args, self.typed_binary)
        {
            spec.name = name;
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
            overlap: CommandOverlap::Primary,
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
        mut instruction: Instruction,
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
        if let Instruction::KvAttention { sequence_major, .. } = &mut instruction {
            *sequence_major = Self::feeds_only_sequence_major_permute(index, dense);
            self.sequence_major_attention[dense.index()] = *sequence_major;
        }
        let sequence_major_alias = match &node.kind {
            NodeKind::Permute { a, dims } if dims.as_slice() == [0, 2, 1, 3] => index
                .dense_id(a.id)
                .is_some_and(|parent| self.sequence_major_attention[parent.index()]),
            _ => false,
        };
        let fused_consumer = Self::feeds_only_fused_consumer(index, optimization, dense);
        let grouped_input = Self::feeds_only_grouped_input(index, dense);
        let rotary_reindex = Self::feeds_only_rotary_reindex(index, dense);
        let rms_input = Self::feeds_only_rms_input(index, dense)
            && index
                .consumers_of(dense)
                .into_iter()
                .flatten()
                .all(|consumer| {
                    optimization.node_region[consumer.index()].is_none_or(|region| {
                        !matches!(
                            optimization.regions[region.index()],
                            NativeRegion::SharedRmsNorm(_)
                                | NativeRegion::FfnTail(_)
                                | NativeRegion::FfnNextNorm(_)
                                | NativeRegion::RmsResidual(_)
                                | NativeRegion::AttentionFfnEntrance(_)
                        )
                    })
                });
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
            && !sequence_major_alias
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
        let instruction = if sequence_major_alias {
            let NodeKind::Permute { a, .. } = &node.kind else {
                unreachable!()
            };
            Instruction::Alias {
                a: index
                    .dense_id(a.id)
                    .ok_or("compile: CUDA sequence-major attention source is missing")?
                    .index(),
            }
        } else {
            instruction
        };
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
                if mean256_eligible(node, self.mean256) {
                    spec.name = "et_mean256_f32";
                }
                // Optional pointer-native attention retains F32 computation but
                // completes the final BF16 boundary directly into BF16 storage.
                let inline_kv_inputs = kv_bf16_inputs_eligible(
                    node,
                    self.state_layout.as_ref(),
                    self.kv_bf16_inputs || self.attention82,
                    self.bf16_kv_matmul,
                ) && operation.compute_dtype == Some(DType::F32)
                    && operation.results.len() == 1
                    && operation.results[0].execution_dtype == DType::F32
                    && spec.inputs[..3].iter().all(|input| {
                        input.is_some_and(|input| {
                            self.elementwise_views[input].is_none()
                                && self.routed_row_views[input].is_none()
                        })
                    });
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
                        let inline_kv_operand = inline_kv_inputs
                            && operand.index < 3
                            && conversion.source == DType::BF16
                            && conversion.destination == DType::F32;
                        if !supports_inline_conversion(spec.name) && !inline_kv_operand {
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
                            if supports_inline_conversion(spec.name)
                                || (self.attention82
                                    && inline_kv_inputs
                                    && spec.name == "et_kv_attention"
                                    && conversion.source == DType::F32
                                    && conversion.destination == DType::BF16) =>
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
                                );
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

    pub(super) fn add_attention_ffn_entrance_region(
        &mut self,
        index: &GraphIndex,
        optimization: &OptimizationPlan,
        region: &effect_torch_compiler::AttentionFfnEntranceRegion,
        dtype_plan: &ExecutableDTypePlan,
    ) -> Result<(), String> {
        // Pure physical preflight before allocating or emitting the fused owner.
        // Ordinary CUDA values are contiguous. Explicit lowerer views are the
        // exceptions; decline them and replay the original semantic recipes.
        let dense = region.inputs.iter().all(|id| {
            self.elementwise_views[id.index()].is_none()
                && !self.sequence_major_attention[id.index()]
                && !self.gemms.weight_views[id.index()]
                && self.semantic_values[id.index()].is_some_and(|value| {
                    self.values[value.index()].shape == index.order[id.index()].shape
                        && self.values[value.index()].storage.representation
                            == StorageRepresentation::Dense
                })
        });
        if !dense {
            let mut execution = dtype_plan.execution().clone();
            execution.realization = ExecutionRealization::MaterializedTransforms;
            let decomposed = ExecutableDTypePlan::Legalize(std::sync::Arc::new(execution));
            for &id in region.nodes.iter() {
                let node = &index.order[id.index()];
                let child = |n: &std::sync::Arc<Node>| index.dense_id(n.id).unwrap().index();
                let instruction = match &node.kind {
                    NodeKind::RmsNorm { x, weight, eps } => Instruction::RmsNorm {
                        x: child(x),
                        weight: weight.as_ref().map(child),
                        eps: *eps,
                    },
                    NodeKind::Add { a, b } | NodeKind::Mul { a, b } => Instruction::Binary {
                        op: if matches!(node.kind, NodeKind::Add { .. }) {
                            0
                        } else {
                            2
                        },
                        a: child(a),
                        b: child(b),
                    },
                    NodeKind::Cast { a, .. } => Instruction::Unary {
                        op: 17,
                        a: child(a),
                        parameter: 0.0,
                    },
                    NodeKind::Reshape { a, .. } => Instruction::Alias { a: child(a) },
                    _ => return Err("compile: entrance decomposition node mismatch".into()),
                };
                self.add(id, node, index, optimization, instruction, &decomposed)?;
            }
            return Ok(());
        }
        let elements = region
            .rows
            .checked_mul(2816)
            .ok_or("compile: entrance extent overflow")?;
        // One declared owner, four checked slices; retained semantic outputs
        // and residual reshapes each keep the same ordinary output lease.
        let packed = self.planned(vec![4, elements], DType::BF16, "attention_ffn_entrance")?;
        let mut inputs = [None; 8];
        for (slot, input) in region.inputs.iter().enumerate() {
            inputs[slot] = Some(self.resolve(input.index())?);
        }
        let mut spec = KernelSpec::new("et_attention_ffn_entrance_bf16", &[]);
        spec.args.compute_dtype = dtype_code(DType::F32);
        spec.args.integers[0] = region.rows as u64;
        self.kernel(spec, inputs, packed)?;
        if self.norm98.is_some() {
            if let Some((tail, rho)) = crate::norm98_pair::find_pair(index, optimization, region) {
                self.norm98_candidates.insert(
                    tail,
                    Norm98Candidate {
                        command: self.commands.len() - 1,
                        instruction: self.lowered.len() - 1,
                        prepare_instruction: self.lowered.len() - 2,
                        packed,
                        inputs: std::array::from_fn(|slot| inputs[slot].unwrap()),
                        rho,
                    },
                );
            }
        }
        let bytes = elements
            .checked_mul(2)
            .ok_or("compile: entrance bytes overflow")?;
        if self.values[packed.index()].decl.bytes
            != bytes
                .checked_mul(4)
                .ok_or("compile: entrance packed overflow")?
        {
            return Err("compile: entrance packed extent mismatch".into());
        }
        for (node, slot) in region
            .outputs
            .iter()
            .copied()
            .enumerate()
            .map(|(slot, node)| (node, slot))
            .chain(region.residual_views.iter().copied().map(|node| (node, 0)))
        {
            let output = self.value(
                index.order[node.index()].shape.clone(),
                DType::BF16,
                StorageMetadata::dense(),
                "attention_ffn_entrance_output",
                ValueStorage::Alias {
                    source: packed,
                    byte_offset: slot
                        .checked_mul(bytes)
                        .ok_or("compile: entrance offset overflow")?,
                },
            )?;
            self.emit(
                "attention_ffn_entrance_output",
                CommandKind::PlannedAlias,
                Some(output),
                vec![ValueUse::read(packed)],
                false,
            )?;
            self.semantic_values[node.index()] = Some(output);
            self.semantic_results[node.index()] = vec![output];
        }
        Ok(())
    }

    pub(super) fn add_ffn_next_norm_region(
        &mut self,
        index: &GraphIndex,
        optimization: &OptimizationPlan,
        region: &effect_torch_compiler::FfnNextNormRegion,
        dtype_plan: &ExecutableDTypePlan,
    ) -> Result<(), String> {
        // Pure physical preflight before allocating or emitting the fused owner.
        // Ordinary CUDA values are contiguous. Explicit lowerer views are the
        // exceptions; decline them and replay the original semantic recipes.
        let elements = region
            .rows
            .checked_mul(2816)
            .ok_or("compile: FFN next norm extent overflow")?;
        let bytes = elements
            .checked_mul(2)
            .ok_or("compile: FFN next norm bytes overflow")?;
        let packed_bytes = bytes
            .checked_mul(2)
            .ok_or("compile: FFN next norm packed overflow")?;
        let mut inputs = [None; 8];
        let dense = matches!(region.rows, 64 | 256)
            && region.inputs.len() == 8
            && region
                .outputs
                .iter()
                .chain(region.residual_views.iter())
                .all(|id| {
                    let node = &index.order[id.index()];
                    node.dtype == DType::BF16
                        && node.shape.last() == Some(&2816)
                        && node.shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
                            == Some(elements)
                })
            && region.inputs.iter().enumerate().all(|(slot, id)| {
                let expected_bytes = if slot < 3 {
                    bytes
                } else if slot == 6 {
                    2
                } else {
                    2816 * 2
                };
                self.elementwise_views[id.index()].is_none()
                    && !self.sequence_major_attention[id.index()]
                    && !self.gemms.weight_views[id.index()]
                    && self.semantic_values[id.index()].is_some_and(|value| {
                        inputs[slot] = Some(value);
                        let actual = &self.values[value.index()];
                        actual.shape == index.order[id.index()].shape
                            && actual.dtype == DType::BF16
                            && actual.decl.bytes == expected_bytes
                            && actual.storage.representation == StorageRepresentation::Dense
                    })
            });
        if !dense {
            let mut execution = dtype_plan.execution().clone();
            execution.realization = ExecutionRealization::MaterializedTransforms;
            let decomposed = ExecutableDTypePlan::Legalize(std::sync::Arc::new(execution));
            for &id in region.nodes.iter() {
                let node = &index.order[id.index()];
                let child = |n: &std::sync::Arc<Node>| index.dense_id(n.id).unwrap().index();
                let instruction = match &node.kind {
                    NodeKind::RmsNorm { x, weight, eps } => Instruction::RmsNorm {
                        x: child(x),
                        weight: weight.as_ref().map(child),
                        eps: *eps,
                    },
                    NodeKind::Add { a, b } | NodeKind::Mul { a, b } => Instruction::Binary {
                        op: if matches!(node.kind, NodeKind::Add { .. }) {
                            0
                        } else {
                            2
                        },
                        a: child(a),
                        b: child(b),
                    },
                    NodeKind::Cast { a, .. } => Instruction::Unary {
                        op: 17,
                        a: child(a),
                        parameter: 0.0,
                    },
                    NodeKind::Reshape { a, .. } | NodeKind::Expose { a, .. } => {
                        Instruction::Alias { a: child(a) }
                    }
                    _ => return Err("compile: FFN next norm decomposition node mismatch".into()),
                };
                self.add(id, node, index, optimization, instruction, &decomposed)?;
            }
            return Ok(());
        }
        if let Some(candidate) = self.norm98_candidates.remove(&region.outputs[0]) {
            return self.add_norm98_tail(index, region, inputs.map(Option::unwrap), candidate);
        }
        // One declared owner, two checked slices. Every semantic result retains
        // its ordinary invocation/output lease; no output is hidden scratch.
        let packed = self.planned(vec![2, elements], DType::BF16, "ffn_next_norm63")?;
        let mut spec = KernelSpec::new("et_ffn_next_norm63_bf16", &[]);
        spec.args.compute_dtype = dtype_code(DType::F32);
        spec.args.integers[0] = region.rows as u64;
        self.kernel(spec, inputs, packed)?;
        if self.values[packed.index()].decl.bytes != packed_bytes {
            return Err("compile: FFN next norm packed extent mismatch".into());
        }
        for (node, slot) in region
            .outputs
            .iter()
            .copied()
            .enumerate()
            .map(|(slot, node)| (node, slot))
            .chain(region.residual_views.iter().copied().map(|node| (node, 0)))
        {
            let output = self.value(
                index.order[node.index()].shape.clone(),
                DType::BF16,
                StorageMetadata::dense(),
                "ffn_next_norm63_output",
                ValueStorage::Alias {
                    source: packed,
                    byte_offset: slot
                        .checked_mul(bytes)
                        .ok_or("compile: FFN next norm offset overflow")?,
                },
            )?;
            self.emit(
                "ffn_next_norm63_output",
                CommandKind::PlannedAlias,
                Some(output),
                vec![ValueUse::read(packed)],
                false,
            )?;
            self.semantic_values[node.index()] = Some(output);
            self.semantic_results[node.index()] = vec![output];
        }
        Ok(())
    }

    fn add_norm98_tail(
        &mut self,
        index: &GraphIndex,
        region: &effect_torch_compiler::FfnNextNormRegion,
        inputs: [ValueId; 8],
        candidate: Norm98Candidate,
    ) -> Result<(), String> {
        let (plan, device) = self
            .norm98
            .clone()
            .ok_or("compile: norm98 module missing")?;
        let rho = if let Some(rho) = self.norm98_rhos.get(&candidate.rho.to_bits()) {
            rho.clone()
        } else {
            let _gate = device
                .graph_execution
                .lock()
                .map_err(|_| "norm98 setup gate poisoned")?;
            let rho = crate::triton_norm98::Rho98::new(&device.stream, candidate.rho)?;
            self.norm98_rhos
                .insert(candidate.rho.to_bits(), rho.clone());
            rho
        };
        let elements = region
            .rows
            .checked_mul(2816)
            .ok_or("compile: norm98 extent overflow")?;
        let sum_a = self.planned(vec![region.rows], DType::F32, "norm98_sum_a")?;
        let combined = self.planned(vec![elements], DType::F32, "norm98_combined")?;
        let next_norm = self.planned(
            index.order[region.outputs[1].index()].shape.clone(),
            DType::BF16,
            "norm98_next_norm",
        )?;
        // Replace entrance only after BOTH semantic liveness and physical tail
        // admission passed. A declined pair keeps the original 76/63 commands.
        let prepare = candidate
            .command
            .checked_sub(1)
            .ok_or("compile: norm98 prepare missing")?;
        if !matches!(self.commands[prepare].kind, CommandKind::Prepare) {
            return Err("compile: norm98 entrance prepare mismatch".into());
        }
        // Stateful preparation instructions do not have executable commands.
        // Preserve the independent indices captured when kernel() emitted each.
        if self.lowered[candidate.prepare_instruction].kind != "prepare"
            || self.lowered[candidate.instruction].kind != "et_attention_ffn_entrance_bf16"
            || self.lowered[candidate.instruction].outputs.as_ref()
                != [OutputDecl::new(candidate.packed)]
        {
            return Err("compile: norm98 lowered entrance identity mismatch".into());
        }
        let mut definitions = self.lowered[candidate.prepare_instruction].outputs.to_vec();
        definitions.push(OutputDecl::new(sum_a));
        self.lowered[candidate.prepare_instruction].outputs = definitions.into();
        let mut entrance_uses = candidate
            .inputs
            .iter()
            .copied()
            .map(ValueUse::read)
            .collect::<Vec<_>>();
        entrance_uses.sort_by_key(|value| value.value);
        entrance_uses.dedup_by_key(|value| value.value);
        let instruction = &mut self.lowered[candidate.instruction];
        *instruction = LoweredInstruction::new(
            instruction.id,
            "triton_norm98_entrance",
            entrance_uses,
            vec![OutputDecl::new(candidate.packed)],
        )
        .with_resources(
            vec![ValueUse::read_write(sum_a)],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        self.commands[candidate.command].kind = CommandKind::Norm98Entrance {
            inputs: candidate.inputs,
            sum_a,
            rows: region.rows as u32,
            rho,
            plan: plan.clone(),
        };
        self.emit(
            "norm98_scratch",
            CommandKind::Prepare,
            Some(combined),
            Vec::new(),
            true,
        )?;
        let mut uses = inputs
            .iter()
            .enumerate()
            .filter(|(slot, _)| *slot != 2)
            .map(|(_, value)| ValueUse::read(*value))
            .collect::<Vec<_>>();
        uses.extend([
            ValueUse::read_write(candidate.packed),
            ValueUse::read(sum_a),
            // Keep the original attention owner as well as its private copy
            // live through the complete paired operation.
            ValueUse::read(candidate.inputs[0]),
            ValueUse::read(candidate.inputs[2]),
            ValueUse::read(candidate.inputs[1]),
            ValueUse::read_write(combined),
        ]);
        uses.sort_by_key(|value| value.value);
        uses.dedup_by_key(|value| value.value);
        self.emit(
            "triton_norm98_tail",
            CommandKind::Norm98Tail {
                inputs,
                entrance: candidate.packed,
                sum_a,
                attention_weight: candidate.inputs[2],
                hidden: candidate.inputs[1],
                combined,
                rows: region.rows as u32,
                plan,
            },
            Some(next_norm),
            uses,
            true,
        )?;
        self.semantic_values[region.outputs[1].index()] = Some(next_norm);
        self.semantic_results[region.outputs[1].index()] = vec![next_norm];
        // The private entrance A slot now owns nextHidden. Every public alias
        // retains its ordinary allocation lease; nextNorm has a distinct owner.
        for node in std::iter::once(region.outputs[0]).chain(region.residual_views.iter().copied())
        {
            let output = self.value(
                index.order[node.index()].shape.clone(),
                DType::BF16,
                StorageMetadata::dense(),
                "norm98_next_hidden",
                ValueStorage::Alias {
                    source: candidate.packed,
                    byte_offset: 0,
                },
            )?;
            self.emit(
                "norm98_next_hidden",
                CommandKind::PlannedAlias,
                Some(output),
                vec![ValueUse::read(candidate.packed)],
                false,
            )?;
            self.semantic_values[node.index()] = Some(output);
            self.semantic_results[node.index()] = vec![output];
        }
        Ok(())
    }

    fn add_vnorm_region(
        &mut self,
        index: &GraphIndex,
        optimization: &OptimizationPlan,
        region: &effect_torch_compiler::VNormKvAttentionRegion,
        dtype_plan: &ExecutableDTypePlan,
    ) -> Result<(), String> {
        let attention = &index.order[region.output.index()];
        let NodeKind::KvAttention {
            q,
            k,
            scale,
            layer,
            window,
            mode,
            rounding,
            ..
        } = &attention.kind
        else {
            return Err("compile: V norm/store attention region mismatch".into());
        };
        let sequence_major = Self::feeds_only_sequence_major_permute(index, region.output);
        let instruction = Instruction::KvAttention {
            q: region.q.index(),
            k: region.k.index(),
            v: region.normalized.index(),
            q_shape: q.shape.clone(),
            k_shape: k.shape.clone(),
            scale: *scale,
            layer: *layer as usize,
            window: *window,
            bidirectional: *mode == effect_torch_graph::KvAttentionMode::BidirectionalBlock,
            rounding: *rounding,
            sequence_major,
        };
        let operation = dtype_plan
            .execution()
            .operation(region.output)
            .ok_or("compile: V norm/store KV recipe missing")?;
        let normalized_recipe = dtype_plan
            .execution()
            .operation(region.normalized)
            .ok_or("compile: V norm/store RMS recipe missing")?;
        let convert = |from, to| effect_torch_compiler::DenseConversionContract::new(from, to);
        let recipes_supported = operation.compute_dtype == Some(DType::F32)
            && operation.attention_rounding
                == Some(effect_torch_graph::AttentionRounding::Stepwise)
            && operation.operands.len() == 3
            && operation.operands.iter().all(|operand| {
                operand.execution_dtype == DType::F32
                    && operand.preparation
                        == OperandPreparation::Convert(convert(DType::BF16, DType::F32))
            })
            && operation.results.len() == 1
            && operation.results[0].execution_dtype == DType::F32
            && operation.results[0].completion
                == ResultCompletion::ConvertToBoundary(convert(DType::F32, DType::BF16))
            && normalized_recipe.compute_dtype == Some(DType::F32);
        let view = self.rms_row_views[region.normalized.index()].as_ref();
        let [_, heads, tokens, dim] = region.shape.as_ref() else {
            return Err("compile: V norm/store shape".into());
        };
        let view_supported = view.is_some_and(|view| {
            view.shape == region.shape.as_ref()
                && view.offset == 0
                && view.moduli.iter().all(|&m| m == 0)
                && view.strides.as_slice() == [tokens * heads * dim, *dim, heads * dim, 1]
                && self.values[view.source.index()].dtype == DType::BF16
                && self.values[view.source.index()].storage.representation
                    == StorageRepresentation::Dense
        });
        let store_plan = crate::vnorm_store::VNormStorePlan::for_layout(
            self.state_layout.as_ref(),
            *layer,
            &region.shape,
            region.eps,
        );
        if !self.bf16_kv_matmul || !recipes_supported || !view_supported || store_plan.is_none() {
            // A valid unsupported physical layout decomposes into the two original
            // approved recipes. Never relabel raw V as its normalized semantic value.
            let mut execution = dtype_plan.execution().clone();
            execution.realization = ExecutionRealization::MaterializedTransforms;
            let decomposed = ExecutableDTypePlan::Legalize(std::sync::Arc::new(execution));
            self.add(
                region.normalized,
                &index.order[region.normalized.index()],
                index,
                optimization,
                Instruction::RmsNorm {
                    x: region.source.index(),
                    weight: None,
                    eps: region.eps,
                },
                &decomposed,
            )?;
            self.add(
                region.output,
                attention,
                index,
                optimization,
                instruction,
                &decomposed,
            )?;
            return Ok(());
        }
        let raw = view.unwrap().source;
        let mut inputs = [None; 8];
        for (slot, semantic) in [region.q, region.k].into_iter().enumerate() {
            let source = self.resolve(semantic.index())?;
            if self.values[source.index()].dtype != DType::BF16 {
                return Err("compile: V norm/store QK dtype".into());
            }
            inputs[slot] = Some(self.conversion(source, DType::F32)?);
        }
        inputs[6] = Some(raw);
        let output = self.planned(attention.shape.clone(), DType::F32, "kv_vnorm_f32")?;
        let mut spec = instruction.kernel()?;
        spec.args.integers[8] = u64::from(dtype_code(DType::BF16));
        spec.args.compute_dtype = dtype_code(DType::F32);
        self.kernel(spec, inputs, output)?;
        let Some(Command {
            kind:
                CommandKind::Kernel {
                    kv_matmul: Some(workspace),
                    ..
                },
            ..
        }) = self.commands.last_mut()
        else {
            return Err("compile: V norm/store preflight/workspace disagreement".into());
        };
        workspace.vnorm_store = store_plan;
        self.lowered.last_mut().unwrap().kind = "kv_stepwise_bf16_gemm_vnorm_store";
        self.sequence_major_attention[region.output.index()] = sequence_major;
        let boundary = self.conversion(output, DType::BF16)?;
        self.semantic_values[region.output.index()] = Some(boundary);
        self.semantic_results[region.output.index()] = vec![boundary];
        return Ok(());
    }

    fn device_expert_aligned_value(&self, mut id: ValueId) -> bool {
        loop {
            match &self.values[id.index()].decl.storage {
                ValueStorage::Alias {
                    source,
                    byte_offset,
                } if byte_offset % 16 == 0 => id = *source,
                ValueStorage::Planned { alignment, .. } => return *alignment >= 16,
                ValueStorage::Fixed {
                    class: StorageClass::PersistentConstant,
                    ..
                } => {
                    return self.commands.iter().find(|c| c.output == Some(id))
                        .is_some_and(|c| matches!(&c.kind, CommandKind::Value(v) if v.storage_address() % 16 == 0));
                }
                _ => return false,
            }
        }
    }

    fn fused_moe75(
        &mut self,
        index: &GraphIndex,
        optimization: &OptimizationPlan,
        region: &effect_torch_compiler::GroupedExpertGatedRegion,
    ) -> Result<bool, String> {
        if !self.allow_fused_moe75 || !crate::fused_moe75::enabled() {
            return Ok(false);
        }
        let Some(finalizer) = &region.finalizer else {
            return Ok(false);
        };
        if finalizer.routes != 8
            || !matches!(finalizer.rows, 64 | 256)
            || region.rows != finalizer.rows * 8
            || region.intermediate != 704
        {
            return Ok(false);
        }
        let Some(view) = self.routed_row_views[region.inputs[0].index()].clone() else {
            return Ok(false);
        };
        if (view.rows, view.routes, view.width) != (finalizer.rows, 8, 2816) {
            return Ok(false);
        }
        let route = index
            .node(region.inputs[3])
            .ok_or("MoE75 missing route node")?;
        if crate::expert_device::topk_rows(route) != Some(finalizer.rows) {
            return Ok(false);
        }
        // The proved TopK node produces valid unique expert IDs. Its original
        // token-major U32 bytes are also valid I32 values in 0..128.
        let NodeKind::Reshape { a: transposed, .. } = &route.kind else {
            unreachable!()
        };
        let NodeKind::Permute { a: topk, .. } = &transposed.kind else {
            unreachable!()
        };
        let topk = index.dense_id(topk.id).ok_or("MoE75 topk missing")?;
        let ranks = region.inputs[4];
        let rank_region = optimization
            .node_region
            .get(ranks.index())
            .copied()
            .flatten()
            .and_then(|id| optimization.regions.get(id.index()));
        if !crate::fused_moe75::proves_route_ranks(rank_region, ranks, topk, finalizer.rows) {
            return Ok(false);
        }
        let indexes = self.resolve(topk.index())?;
        let gate = self.resolve(region.inputs[1].index())?;
        let down = self.resolve(region.inputs[2].index())?;
        let scales = self.resolve(region.inputs[5].index())?;
        // Packed-weight caching is restricted to captured constant owners.
        let Some(weight) = self.commands.iter().find_map(|command| {
            if command.output != Some(gate) {
                return None;
            }
            match &command.kind {
                CommandKind::Value(value) => Some(value.clone()),
                _ => None,
            }
        }) else {
            return Ok(false);
        };
        if self.values[down.index()].shape != [128, 2816, 704]
            || self.values[down.index()].dtype != DType::BF16
            || self.values[scales.index()].dtype != DType::F32
            || product_checked(&self.values[scales.index()].shape)? != finalizer.rows * 8
        {
            return Ok(false);
        }
        let plan = crate::fused_moe75::prepare(&weight.device, &weight, finalizer.rows)?;
        // The derived layout is private physical storage, never a semantic
        // replacement for the canonical gate/up weight tensor.
        let packed_buffer = plan.packed_buffer();
        let packed_shape = vec![packed_buffer.len()];
        let packed_value = CudaValue::from_planned_buffer(
            weight.device.clone(),
            ValueSpec::dense(DType::U8, &packed_shape),
            packed_buffer,
        )?;
        let packed = self.value(
            packed_shape,
            DType::U8,
            StorageMetadata::dense(),
            "moe75_packed_weight",
            ValueStorage::Fixed {
                class: StorageClass::PersistentConstant,
                location: Location::Persistent {
                    slot: self.values.len() as u32,
                },
            },
        )?;
        self.emit(
            "value",
            CommandKind::Value(packed_value),
            Some(packed),
            Vec::new(),
            false,
        )?;
        let status = self
            .status
            .ok_or("MoE75 requires deferred status storage")?;
        let workspace = self.planned(vec![plan.workspace_bytes], DType::U8, "moe75_workspace")?;
        let map = self.planned(vec![plan.map_bytes], DType::U8, "moe75_map")?;
        let routes = self.planned(vec![plan.route_scratch_bytes], DType::U8, "moe75_routes")?;
        for value in [workspace, map, routes] {
            self.emit(
                "prepare",
                CommandKind::Prepare,
                Some(value),
                Vec::new(),
                true,
            )?;
        }
        let output = self.planned(vec![finalizer.rows, 2816], DType::BF16, "moe75_output")?;
        let id =
            InstructionId::from_index(self.lowered.len()).ok_or("MoE75 instruction overflow")?;
        self.lowered.push(
            LoweredInstruction::new(
                id,
                "fused_moe75",
                [view.source, indexes, scales, gate, packed, down]
                    .into_iter()
                    .map(ValueUse::read)
                    .collect::<Vec<_>>(),
                vec![OutputDecl::new(output)],
            )
            .with_resources(
                vec![
                    ValueUse::read_write(workspace),
                    ValueUse::read_write(map),
                    ValueUse::read_write(routes),
                    ValueUse::read_write(status),
                ],
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
            overlap: CommandOverlap::Primary,
            kind: CommandKind::FusedMoe75 {
                x: view.source,
                indexes,
                scales,
                down,
                workspace,
                map,
                routes,
                status,
                plan,
            },
        });
        self.semantic_values[region.output.index()] = Some(output);
        self.semantic_results[region.output.index()] = vec![output];
        Ok(true)
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
        if let NativeRegion::VNormKvAttention(region) = native {
            return self.add_vnorm_region(index, optimization, region, dtype_plan);
        }
        if let NativeRegion::ExpertRouteRank(region) = native {
            let output = self.planned(
                vec![region.rows, region.routes],
                DType::U32,
                "expert_route_rank",
            )?;
            let mut inputs = [None; 8];
            inputs[0] = Some(self.resolve(region.inputs[0].index())?);
            inputs[1] = Some(self.resolve(region.inputs[1].index())?);
            let mut spec = KernelSpec::new("et_expert_route_rank", &[]);
            spec.args.integers[0] = region.rows as u64;
            spec.args.integers[1] = region.routes as u64;
            self.kernel(spec, inputs, output)?;
            self.semantic_values[region.output.index()] = Some(output);
            self.semantic_results[region.output.index()] = vec![output];
            return Ok(());
        }
        if let NativeRegion::FfnNextNorm(region) = native {
            return self.add_ffn_next_norm_region(index, optimization, region, dtype_plan);
        }
        if let NativeRegion::AttentionFfnEntrance(region) = native {
            return self.add_attention_ffn_entrance_region(index, optimization, region, dtype_plan);
        }
        if let NativeRegion::RmsResidual(region) = native {
            let mut inputs = [None; 8];
            for (slot, input) in region.inputs.iter().enumerate() {
                inputs[slot] = Some(self.resolve(input.index())?);
            }
            let output = self.planned(region.shape.to_vec(), DType::BF16, "rms_residual")?;
            let mut spec = KernelSpec::new("et_rms_residual_bf16", &[]);
            spec.args.compute_dtype = dtype_code(DType::F32);
            spec.args.scalars[0] = region.eps;
            self.kernel(spec, inputs, output)?;
            self.semantic_values[region.output.index()] = Some(output);
            self.semantic_results[region.output.index()] = vec![output];
            return Ok(());
        }
        if let NativeRegion::FfnTail(region) = native {
            let mut inputs = [None; 8];
            for (slot, input) in region.inputs.iter().enumerate() {
                inputs[slot] = Some(self.resolve(input.index())?);
            }
            let output = self.planned(region.shape.clone(), DType::BF16, "ffn_tail")?;
            let mut spec = KernelSpec::new("et_ffn_tail_bf16", &[]);
            spec.args.compute_dtype = dtype_code(DType::F32);
            self.kernel(spec, inputs, output)?;
            self.semantic_values[region.output.index()] = Some(output);
            self.semantic_results[region.output.index()] = vec![output];
            return Ok(());
        }
        if let NativeRegion::GroupedExpertGated(region) = native {
            if self.fused_moe75(index, optimization, region)? {
                return Ok(());
            }
            let device_control = device.device_expert_ready()
                && self.status.is_some()
                && region.inputs[1..3].iter().all(|id| {
                    self.semantic_values[id.index()]
                        .is_some_and(|value| self.device_expert_aligned_value(value))
                })
                && index
                    .node(region.inputs[3])
                    .and_then(|n| crate::expert_device::topk_rows(n))
                    .is_some_and(|tokens| tokens * 8 == region.rows)
                && index
                    .node(region.inputs[1])
                    .is_some_and(|n| n.dtype == DType::BF16 && n.shape == [128, 1408, 2816])
                && index
                    .node(region.inputs[2])
                    .is_some_and(|n| n.dtype == DType::BF16 && n.shape == [128, 2816, 704]);
            let sorted = self.grouped_expert_layout(
                region.inputs[0].index(),
                region.inputs[1].index(),
                region.inputs[3].index(),
                None,
                true,
                region.finalizer.is_some(),
                device_control,
            )?;
            let expressions = legalize_region_expressions(native, dtype_plan)?;
            let expression = expressions
                .first()
                .ok_or("compile: grouped activation missing")?;
            let shape = [region.rows, region.intermediate];
            let lanes = vec![vec![2 * region.intermediate, 1].into_boxed_slice(); 2];
            let moduli = vec![vec![0, 0].into_boxed_slice(); 2];
            let source = crate::emit::elementwise_with_sum(
                false,
                false,
                expression,
                &lanes,
                &moduli,
                &[0, region.intermediate],
                &shape,
            )?;
            let source =
                if std::env::var("EFFECT_TORCH_CUDA_FUSED_STATIC_DTYPES").as_deref() == Ok("1") {
                    crate::emit::specialize_storage_dtypes(
                        source,
                        &[dtype_code(DType::BF16); 2],
                        dtype_code(DType::BF16),
                    )?
                } else {
                    source
                };
            let function = device.fused_elementwise_with_graph61(
                &source,
                crate::executable::expert_pair61::enabled(),
            )?;
            let activation =
                self.planned(shape.to_vec(), DType::BF16, "group_sorted_activation")?;
            let mut inputs = [None; 8];
            inputs[0] = Some(sorted);
            inputs[1] = Some(sorted);
            let mut args = CudaKernelArgs {
                elements: product_checked(&shape)? as u64,
                output_dtype: dtype_code(DType::BF16),
                compute_dtype: dtype_code(DType::F32),
                ..Default::default()
            };
            args.input_dtypes[..2].fill(dtype_code(DType::BF16));
            self.emit(
                "group_sorted_activation",
                CommandKind::FusedElementwise {
                    wide_sum: false,
                    wide_arg: false,
                    function,
                    graph61_function: device.graph61_fused(&source),
                    args,
                    inputs,
                },
                Some(activation),
                vec![ValueUse::read(sorted)],
                true,
            )?;
            // x semantic argument is not resolved when sorted_input is supplied.
            let output = self.grouped_expert_layout(
                region.inputs[0].index(),
                region.inputs[2].index(),
                region.inputs[3].index(),
                Some(activation),
                region.finalizer.is_some(),
                region.finalizer.is_some(),
                device_control,
            )?;
            let output = if let Some(finalizer) = &region.finalizer {
                let (_, row_map) = self
                    .grouped_routing
                    .get(&(
                        region.inputs[3].index(),
                        region.rows,
                        128,
                        true,
                        device_control,
                    ))
                    .copied()
                    .ok_or("compile: private inverse routing missing")?;
                let inverse = self.value(
                    vec![region.rows],
                    DType::U32,
                    StorageMetadata::dense(),
                    "group_inverse_rows",
                    ValueStorage::Alias {
                        source: row_map,
                        byte_offset: region
                            .rows
                            .checked_mul(4)
                            .ok_or("compile: inverse map offset overflow")?,
                    },
                )?;
                self.emit(
                    "group_inverse_rows",
                    CommandKind::PlannedAlias,
                    Some(inverse),
                    vec![ValueUse::read(row_map)],
                    false,
                )?;
                let reduced =
                    self.planned(vec![finalizer.rows, 2816], DType::BF16, "expert_finalize")?;
                let mut inputs = [None; 8];
                inputs[0] = Some(output);
                inputs[1] = Some(self.resolve(region.inputs[4].index())?);
                inputs[2] = Some(self.resolve(region.inputs[5].index())?);
                inputs[3] = Some(inverse);
                let mut spec = KernelSpec::new("et_ordered_sorted_reduce", &[]);
                spec.args.compute_dtype = dtype_code(DType::F32);
                spec.args.integers[0] = finalizer.routes as u64;
                spec.args.integers[1] = 2816;
                spec.args.integers[8] = 1;
                spec.args.integers[9] = finalizer.rows as u64;
                self.kernel(spec, inputs, reduced)?;
                reduced
            } else {
                output
            };
            self.semantic_values[region.output.index()] = Some(output);
            self.semantic_results[region.output.index()] = vec![output];
            return Ok(());
        }
        if let NativeRegion::Entropy(region) = native {
            let rows = product_checked(&region.shape)? / region.width;
            let source = self.resolve(region.inputs[0].index())?;
            let mut inputs = [None; 8];
            inputs[0] = Some(source);
            let stages = crate::entropy81::stages(crate::entropy81::enabled());
            for (stage, &name) in stages.iter().enumerate() {
                let final_stage = stage + 1 == stages.len();
                let shape = if final_stage {
                    index.order[region.output.index()].shape.clone()
                } else {
                    vec![rows]
                };
                let output = self.planned(shape, DType::F32, "entropy_row")?;
                let mut spec = KernelSpec::new(name, &[]);
                spec.args.compute_dtype = dtype_code(DType::F32);
                spec.args.integers[0] = region.width as u64;
                spec.args.integers[1] = rows as u64;
                self.kernel(spec, inputs, output)?;
                if final_stage {
                    self.semantic_values[region.output.index()] = Some(output);
                    self.semantic_results[region.output.index()] = vec![output];
                } else {
                    inputs[stage + 1] = Some(output);
                }
            }
            return Ok(());
        }
        if let NativeRegion::NormRope(region) = native {
            let width = region.shape[3];
            let mut inputs = [None; 8];
            inputs[0] = Some(self.resolve(region.source.index())?);
            inputs[1] = region.weight.map(|w| self.resolve(w.index())).transpose()?;
            let mut spec = KernelSpec::new("et_norm_rope_bf16", &[]);
            // Bits 0/1 describe physical table views; bit 2 selects exact mask indexing.
            if std::env::var("EFFECT_TORCH_CUDA_NORM_ROPE_MASK").as_deref() == Ok("1") {
                spec.args.operation |= 4;
            }
            spec.args.compute_dtype = dtype_code(DType::F32);
            spec.args.integers[0] = width as u64;
            spec.args.integers[10] = region.shape[1] as u64;
            spec.args.integers[11] = region.shape[2] as u64;
            spec.args.scalars[0] = region.eps;
            if let Some(view) = &self.rms_row_views[region.normalized.index()] {
                let outer_rank = view.shape.len() - 1;
                if view.shape.as_slice() != region.shape.as_ref()
                    || view.strides.last() != Some(&1)
                    || view.offset % width != 0
                    || view.strides[..outer_rank]
                        .iter()
                        .any(|stride| stride % width != 0)
                    || view.moduli.iter().any(|&modulus| modulus != 0)
                {
                    return Err("compile: norm/RoPE RMS row view is invalid".into());
                }
                inputs[0] = Some(view.source);
                spec.args.integers[1] = outer_rank as u64;
                spec.args.integers[2] = (view.offset / width) as u64;
                for axis in 0..outer_rank {
                    spec.args.integers[3 + axis] = view.shape[axis] as u64;
                    spec.args.integers[3 + outer_rank + axis] = (view.strides[axis] / width) as u64;
                }
            }
            // Table nodes can be aliases over repeated half-width storage.
            // Keep their source allocations and exact physical indexing rather
            // than treating a semantic full-width shape as materialized bytes.
            for (slot, semantic) in [region.cosine, region.sine].into_iter().enumerate() {
                // Private provenance consumed only by late101 admission; the
                // ordinary NormRope kernel does not read integer slot15.
                if normrope101_lowering::repeated_half_table(&index.order[semantic.index()], width)
                {
                    spec.args.integers[15] |= 1 << slot;
                }
                let layout = if let Some(view) = &self.elementwise_views[semantic.index()] {
                    if view.shape != [1, 1, region.shape[2], width]
                        || view.strides.len() != 4
                        || view.moduli.len() != 4
                    {
                        return Err("compile: norm/RoPE table view is invalid".into());
                    }
                    inputs[slot + 2] = Some(view.source);
                    [
                        view.offset,
                        view.strides[2],
                        view.strides[3],
                        view.moduli[2],
                        view.moduli[3],
                    ]
                } else {
                    inputs[slot + 2] = Some(self.resolve(semantic.index())?);
                    [0, width, 1, 0, 0]
                };
                if layout != [0, width, 1, 0, 0] {
                    spec.args.operation |= 1 << slot;
                }
                spec.tail.extend(layout.map(|value| value as u64));
            }
            let output = self.planned(region.shape.to_vec(), DType::BF16, "norm_rope")?;
            self.kernel(spec, inputs, output)?;
            self.semantic_values[region.output.index()] = Some(output);
            self.semantic_results[region.output.index()] = vec![output];
            return Ok(());
        }
        if let NativeRegion::DualArgmax(region) = native {
            let packed = self.planned(vec![2, region.rows], DType::I64, "dual_argmax")?;
            let bytes = region
                .rows
                .checked_mul(8)
                .ok_or("compile: dual argmax offset overflow")?;
            let mut inputs = [None; 8];
            for (slot, node) in region.inputs.iter().enumerate() {
                inputs[slot] = Some(self.resolve(node.index())?);
            }
            let mut spec = KernelSpec::new("et_dual_argmax", &[]);
            spec.args.compute_dtype = dtype_code(DType::F32);
            spec.args.integers[0] = region.rows as u64;
            spec.args.integers[1] = region.width as u64;
            self.kernel(spec, inputs, packed)?;
            for (node, byte_offset) in [(region.plain, 0), (region.noisy, bytes)] {
                let output = self.value(
                    region.shape.to_vec(),
                    DType::I64,
                    StorageMetadata::dense(),
                    "dual_argmax_output",
                    ValueStorage::Alias {
                        source: packed,
                        byte_offset,
                    },
                )?;
                self.emit(
                    "dual_argmax_output",
                    CommandKind::PlannedAlias,
                    Some(output),
                    vec![ValueUse::read(packed)],
                    false,
                )?;
                self.semantic_values[node.index()] = Some(output);
                self.semantic_results[node.index()] = vec![output];
            }
            return Ok(());
        }
        if let NativeRegion::RouterTail(region) = native {
            let packed = self.planned(vec![2, region.rows, 8], DType::F32, "router_tail")?;
            let bytes = region
                .rows
                .checked_mul(8)
                .and_then(|n| n.checked_mul(4))
                .ok_or("compile: router tail output offset overflow")?;
            let mut inputs = [None; 8];
            for (slot, node) in region.inputs.iter().enumerate() {
                inputs[slot] = Some(self.resolve(node.index())?);
            }
            let mut spec = KernelSpec::new("et_router_tail", &[]);
            spec.args.compute_dtype = dtype_code(DType::F32);
            spec.args.integers[0] = region.rows as u64;
            self.kernel(spec, inputs, packed)?;
            // Both semantic results are explicit typed aliases over disjoint
            // ranges of the planned allocation, retaining their common owner.
            for (node, dtype, byte_offset) in [
                (region.weights, DType::F32, 0),
                (region.indices, DType::U32, bytes),
            ] {
                let output = self.value(
                    vec![region.rows, 8],
                    dtype,
                    StorageMetadata::dense(),
                    "router_tail_output",
                    ValueStorage::Alias {
                        source: packed,
                        byte_offset,
                    },
                )?;
                self.emit(
                    "router_tail_output",
                    CommandKind::PlannedAlias,
                    Some(output),
                    vec![ValueUse::read(packed)],
                    false,
                )?;
                self.semantic_values[node.index()] = Some(output);
                self.semantic_results[node.index()] = vec![output];
            }
            return Ok(());
        }
        if let NativeRegion::SmallSoftmax(region) = native {
            let source = self.resolve(region.inputs[0].index())?;
            let mut inputs = [None; 8];
            inputs[0] = Some(source);
            let output = self.planned(region.shape.to_vec(), DType::F32, "small_softmax")?;
            let mut spec = KernelSpec::new("et_small_softmax_f32", &[]);
            spec.args.compute_dtype = dtype_code(DType::F32);
            self.kernel(spec, inputs, output)?;
            self.semantic_values[region.output.index()] = Some(output);
            self.semantic_results[region.output.index()] = vec![output];
            return Ok(());
        }
        if let NativeRegion::Bf16Softmax(region) = native {
            let elements = product_checked(&region.shape)?;
            let rows = elements / region.width;
            let source = self.resolve(region.inputs[0].index())?;
            if let Some(plan) = self.softmax100.clone().filter(|_| {
                crate::triton_softmax100::admitted(
                    rows,
                    region.width,
                    self.values[source.index()].dtype,
                    self.values[source.index()].storage.representation
                        == StorageRepresentation::Dense,
                )
            }) {
                let scratch = self.planned(
                    vec![crate::triton_softmax100::SCRATCH_ELEMENTS],
                    DType::F32,
                    "softmax100_scratch",
                )?;
                let output = self.planned(region.shape.to_vec(), DType::BF16, "softmax100")?;
                self.emit("prepare", CommandKind::Prepare, Some(scratch), vec![], true)?;
                self.emit(
                    "triton_softmax100",
                    CommandKind::Softmax100 {
                        source,
                        scratch,
                        plan,
                    },
                    Some(output),
                    vec![ValueUse::read(source), ValueUse::read_write(scratch)],
                    true,
                )?;
                self.semantic_values[region.output.index()] = Some(output);
                self.semantic_results[region.output.index()] = vec![output];
                return Ok(());
            }
            let scratch_elements = elements
                .checked_add(rows)
                .ok_or("compile: BF16 softmax scratch overflow")?;
            let scratch =
                self.planned(vec![scratch_elements], DType::F32, "bf16_softmax_scratch")?;
            let mut inputs = [None; 8];
            inputs[0] = Some(source);
            let mut prepare = KernelSpec::new("et_bf16_softmax_prepare", &[]);
            prepare.args.compute_dtype = dtype_code(DType::F32);
            prepare.args.integers[0] = region.width as u64;
            prepare.args.integers[1] = rows as u64;
            self.kernel(prepare, inputs, scratch)?;
            let output = self.planned(region.shape.to_vec(), DType::BF16, "bf16_softmax")?;
            inputs[0] = Some(scratch);
            let mut finish = KernelSpec::new("et_bf16_softmax_store", &[]);
            finish.args.compute_dtype = dtype_code(DType::F32);
            finish.args.integers[0] = region.width as u64;
            self.kernel(finish, inputs, output)?;
            self.semantic_values[region.output.index()] = Some(output);
            self.semantic_results[region.output.index()] = vec![output];
            return Ok(());
        }
        if let NativeRegion::SharedRmsNorm(region) = native {
            let packed = self.planned(
                vec![region.nodes.len(), region.elements],
                region.dtype,
                "shared_rms",
            )?;
            let mut inputs = [None; 8];
            inputs[0] = Some(self.resolve(region.source.index())?);
            for (i, weight) in region.weights.iter().enumerate() {
                inputs[i + 1] = weight.map(|w| self.resolve(w.index())).transpose()?;
            }
            let mut spec = KernelSpec::new("et_shared_rms_norm_f32", &[]);
            spec.args.compute_dtype = dtype_code(DType::F32);
            spec.args.integers[0] = region.width as u64;
            spec.args.integers[1] = region.elements as u64;
            spec.args.integers[2] = region.nodes.len() as u64;
            spec.args.scalars[0] = region.eps;
            self.kernel(spec, inputs, packed)?;
            let bytes = self.values[packed.index()].decl.bytes / region.nodes.len();
            for (i, node) in region.nodes.iter().enumerate() {
                let output = self.value(
                    index.order[node.index()].shape.clone(),
                    region.dtype,
                    StorageMetadata::dense(),
                    "shared_rms_output",
                    ValueStorage::Alias {
                        source: packed,
                        byte_offset: i * bytes,
                    },
                )?;
                self.emit(
                    "shared_rms_output",
                    CommandKind::PlannedAlias,
                    Some(output),
                    vec![ValueUse::read(packed)],
                    false,
                )?;
                self.semantic_values[node.index()] = Some(output);
                self.semantic_results[node.index()] = vec![output];
            }
            return Ok(());
        }
        if let NativeRegion::OrderedScatterReduce(region) = native {
            let output = self.planned(
                region.shape.to_vec(),
                region.dtype,
                "ordered_scatter_reduce",
            )?;
            let source = self.resolve(region.inputs[0].index())?;
            let indexes = self.resolve(region.inputs[1].index())?;
            let mut inputs = [None; 8];
            inputs[0] = Some(source);
            inputs[1] = Some(indexes);
            if region.weighted_source {
                inputs[2] = Some(self.resolve(region.inputs[2].index())?);
            }
            let mut spec = KernelSpec::new("et_ordered_scatter_reduce", &[]);
            spec.args.compute_dtype = dtype_code(DType::F32);
            spec.args.integers[0] = region.routes as u64;
            spec.args.integers[1] = region.shape[1] as u64;
            spec.args.integers[8] = u64::from(region.weighted_source);
            spec.args.integers[9] = region.shape[0] as u64;
            self.kernel(spec, inputs, output)?;
            self.semantic_values[region.output.index()] = Some(output);
            self.semantic_results[region.output.index()] = vec![output];
            return Ok(());
        }
        let reduced;
        let (region, output_shape, wide_sum) = match native {
            NativeRegion::Elementwise(region) => (region, region.shape.as_ref(), false),
            NativeRegion::ElementwiseReduce(region) => {
                reduced = effect_torch_compiler::ElementwiseRegion {
                    nodes: region.nodes.clone(),
                    inputs: region.inputs.clone(),
                    lane_strides: region.lane_strides.clone(),
                    output: effect_torch_compiler::ElementwiseOutput {
                        semantic_node: region.output,
                        expression: region.expression.clone(),
                    },
                    shape: region.input_shape.clone(),
                    dtype: region.dtype,
                    device: region.device.clone(),
                };
                (&reduced, region.shape.as_ref(), true)
            }
            NativeRegion::ElementwiseArgReduce(region) => {
                reduced = effect_torch_compiler::ElementwiseRegion {
                    nodes: region.nodes.clone(),
                    inputs: region.inputs.clone(),
                    lane_strides: region.lane_strides.clone(),
                    output: effect_torch_compiler::ElementwiseOutput {
                        semantic_node: region.output,
                        expression: region.expression.clone(),
                    },
                    shape: region.input_shape.clone(),
                    dtype: region.dtype,
                    device: region.device.clone(),
                };
                (&reduced, region.shape.as_ref(), false)
            }
            _ => return Err("compile: unsupported CUDA optimization region".into()),
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
        let output = self.planned(output_shape.to_vec(), region.dtype, "fused_elementwise")?;
        let wide_arg = matches!(native, NativeRegion::ElementwiseArgReduce(_));
        let source = crate::emit::elementwise_with_sum(
            wide_sum,
            wide_arg,
            expression,
            &lane_strides,
            &lane_moduli,
            &lane_offsets,
            &region.shape,
        )?;
        let source = if std::env::var("EFFECT_TORCH_CUDA_FUSED_STATIC_DTYPES").as_deref() == Ok("1")
        {
            let input_dtypes = inputs[..region.inputs.len()]
                .iter()
                .map(|value| {
                    value
                        .map(|value| dtype_code(self.values[value.index()].dtype))
                        .ok_or("compile: CUDA fused physical input is missing")
                })
                .collect::<Result<Vec<_>, _>>()?;
            crate::emit::specialize_storage_dtypes(
                source,
                &input_dtypes,
                dtype_code(self.values[output.index()].dtype),
            )?
        } else {
            source
        };
        let function = device.fused_elementwise(&source)?;
        let mut args = CudaKernelArgs {
            elements: product_checked(output_shape)? as u64,
            output_dtype: dtype_code(region.dtype),
            compute_dtype: dtype_code(DType::F32),
            ..Default::default()
        };
        if let NativeRegion::ElementwiseArgReduce(region) = native {
            args.operation = u32::from(!region.maximum);
        }
        if wide_sum || wide_arg {
            args.integers[1] = *region
                .shape
                .last()
                .ok_or("compile: empty reduction shape")? as u64;
        }
        for (slot, value) in inputs.iter().enumerate() {
            if let Some(value) = value {
                args.input_dtypes[slot] = dtype_code(self.values[value.index()].dtype);
            }
        }
        self.emit(
            "fused_elementwise",
            CommandKind::FusedElementwise {
                wide_sum,
                wide_arg,
                function,
                graph61_function: None,
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
        if node.shape.as_slice() != output_shape || node.dtype != region.dtype {
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
            overlap: CommandOverlap::Primary,
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
        self.grouped_expert_layout(x, weight, indexes, None, false, false, false)
    }

    fn grouped_expert_layout(
        &mut self,
        x: usize,
        weight: usize,
        indexes: usize,
        sorted_input: Option<ValueId>,
        output_sorted: bool,
        inverse_routing: bool,
        device_control: bool,
    ) -> Result<ValueId, String> {
        let routed = if sorted_input.is_some() {
            None
        } else {
            self.routed_row_views[x].clone()
        };
        let input_sorted = sorted_input.is_some();
        let indexes_semantic = indexes;
        let (x, weight, indexes) = (
            match sorted_input {
                Some(value) => value,
                None => self.resolve(x)?,
            },
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
        let routing_key = (
            indexes_semantic,
            rows,
            experts,
            inverse_routing,
            device_control,
        );
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
                self.planned(
                    vec![rows
                        .checked_mul(if inverse_routing { 2 } else { 1 })
                        .ok_or("compile: inverse map capacity overflow")?],
                    DType::U32,
                    "group_rows",
                )?,
                false,
            )
        };
        if input_sorted && previous_routing.is_none() {
            return Err("compile: sorted group input requires preceding routing".into());
        }
        let gathered = if input_sorted {
            x
        } else {
            self.planned(vec![rows, inner], dtype, "group_input")?
        };
        let projected = self.planned(vec![rows, columns], dtype, "group_output")?;
        let custom_workspace_shape = dtype == DType::BF16
            && rows != 0
            && experts <= 128
            && matches!((columns, inner), (1408, 2816) | (2816, 704));
        let splitk_partials = custom_workspace_shape
            && [
                "EFFECT_TORCH_CUDA_EXPERT_SPLITK",
                "EFFECT_TORCH_CUDA_EXPERT_GROUPED_PARTIALS",
            ]
            .into_iter()
            .any(|key| std::env::var(key).is_ok_and(|value| value == "1"));
        let merged_descriptors = custom_workspace_shape
            && std::env::var("EFFECT_TORCH_CUDA_EXPERT_MERGED_PTX")
                .is_ok_and(|path| !path.is_empty());
        let gemv_descriptors = custom_workspace_shape
            && (columns, inner) == (2816, 704)
            && std::env::var("EFFECT_TORCH_CUDA_EXPERT_GEMV").is_ok_and(|value| value == "1");
        // The runtime's custom-path guard requires planner-owned descriptors
        // for GEMV too. GEMV writes its final output directly, so its opt-in
        // needs neither split-K partials nor extra first-projection scratch.
        // The merged tensor path uses the same bounded descriptor bank and
        // stores its shapes in the disjoint grouped-pointer bank.
        let splitk_workspace = splitk_partials || gemv_descriptors || merged_descriptors;
        let workspace_bytes = if splitk_partials {
            grouped_workspace_bytes(rows, columns, true)?
        } else if gemv_descriptors || merged_descriptors {
            grouped_workspace_bytes(rows, columns, false)?
                .checked_add(EXPERT_SPLITK_DESCRIPTOR_BYTES)
                .ok_or("compile: CUDA GEMV descriptor workspace overflow")?
        } else {
            grouped_workspace_bytes(rows, columns, false)?
        };
        let workspace_bytes = if merged_descriptors
            && (columns, inner) == (2816, 704)
            && crate::device::expert_gemv_overlap::enabled()
        {
            workspace_bytes
                .checked_add(crate::device::expert_gemv_overlap::DESCRIPTOR_BYTES)
                .ok_or("compile: CUDA overlapping GEMV workspace overflow")?
        } else {
            workspace_bytes
        };
        let workspace = if dtype == DType::BF16 && rows != 0 && inner != 0 && columns != 0 {
            Some(self.planned(vec![workspace_bytes], DType::U8, "cublas_workspace")?)
        } else {
            None
        };
        let device_control = if device_control {
            Some((
                self.planned(
                    vec![crate::expert_device::BYTES],
                    DType::U8,
                    "expert_device_metadata",
                )?,
                self.status
                    .ok_or("compile: device expert requires checked TopK status")?,
            ))
        } else {
            None
        };
        let owned_scratch = [
            device_control.map(|(metadata, _)| metadata),
            (!reuse_routing).then_some(control),
            (!reuse_routing).then_some(row_map),
            (!input_sorted).then_some(gathered),
            (!output_sorted).then_some(projected),
            workspace,
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        for &id in &owned_scratch {
            self.emit("prepare", CommandKind::Prepare, Some(id), Vec::new(), true)?;
        }
        let scratch = [
            device_control.map(|(metadata, _)| metadata),
            device_control.map(|(_, status)| status),
            Some(control),
            Some(row_map),
            (!input_sorted).then_some(gathered),
            (!output_sorted).then_some(projected),
            workspace,
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        let output = if output_sorted {
            projected
        } else {
            self.planned(vec![rows, columns], dtype, "result")?
        };
        let id = InstructionId::from_index(self.lowered.len())
            .ok_or("compile: too many CUDA instructions")?;
        self.lowered.push(
            LoweredInstruction::new(
                id,
                if device_control.is_some() {
                    "grouped_expert_linear_rows_device59_non_capturable"
                } else {
                    "grouped_expert_linear_rows_host_control_non_capturable"
                },
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
            overlap: CommandOverlap::Primary,
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
                splitk_workspace,
                source_rows,
                input_sorted,
                output_sorted,
                inverse_routing,
                device_control,
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
        if packed_projection77::admitted_policy(
            self.allow_fused_moe75,
            std::env::var("EFFECT_TORCH_CUDA_PACKED_PROJECTION77").as_deref() == Ok("1"),
        ) {
            self.pack_projections77()?;
            let roots = index
                .roots
                .iter()
                .map(|root| self.resolve(root.index()))
                .collect::<Result<Vec<_>, _>>()?;
            self.fuse_normrope101(&roots)?;
        }
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
        let mut program = LoweredProgram::new(self.values, self.lowered, outputs);
        if std::env::var("EFFECT_TORCH_CUDA_DIV_FEEDBACK").as_deref() == Ok("1") {
            crate::div_feedback::fuse(&mut program, &mut self.commands)?;
        }
        if crate::rng_arg80::enabled() {
            crate::rng_arg80::fuse(&mut program, &mut self.commands)?;
        }
        if crate::sampler83::enabled() {
            crate::sampler83::fuse(&mut program, &mut self.commands)?;
        }
        if crate::kv_pair::enabled() {
            crate::kv_pair::fuse(&mut program, &mut self.commands)?;
        }
        if std::env::var("EFFECT_TORCH_CUDA_DENSE_OVERLAP").is_ok_and(|value| value == "1") {
            crate::planned_overlap::plan_dense_overlap(&mut program, &mut self.commands)?;
        }
        Ok((program, self.commands))
    }
}

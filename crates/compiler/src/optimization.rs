//! Partitions a semantic graph into codegen regions and orders them for
//! deterministic lowering.
//!
//! Selection runs these passes over the [`GraphIndex`] in order. Each pass
//! reserves nodes before the next pass starts:
//!
//! 1. With Metal fusion and epilogues enabled, `Linear` absorbs a following
//!    residual `Add` or `Gelu` into [`LinearResidualRegion`] or
//!    [`LinearGeluRegion`].
//! 2. `AdamWStep` and `SgdStep`, including their `*Out` pickers, become
//!    [`AdamWRegion`] or [`SgdRegion`]. Enabling optimizer groups lets up to
//!    four compatible AdamW steps share one [`AdamWGroupRegion`].
//! 3. Targets may select exact BF16 softmax, shared RMS normalization, and
//!    ordered scatter regions before generic elementwise fusion.
//! 4. Broadcast-compatible elementwise chains form one [`ElementwiseRegion`]
//!    per endpoint, subject to `MAX_LANES`. A chain that feeds a reduction
//!    becomes an [`ElementwiseReduceRegion`].
//! 5. An elementwise prefix with several same-shaped continuations can merge
//!    into a [`MultiOutputRegion`]. `MAX_BUFFERS`, `MAX_MERGED_OPS`, and
//!    dependency analysis keep the merge from consuming values still needed
//!    elsewhere.
//!
//! [`OptimizationPlan::validate`] checks these rules before selection returns:
//!
//! - Selection never rebuilds semantic nodes. Every covered node belongs to
//!   exactly one region, and `semantic_nodes_rebuilt` remains zero.
//! - Each semantic value comes from one region output or one independent
//!   node. Internal region values do not escape.
//! - The lowering order is a topological sort over regions and independent
//!   nodes, so backends do not need to recover topology.
//! - Identical graphs and options select identical regions. Dense IDs break
//!   ties.

use crate::schedule::{DenseNodeId, GraphIndex};
use crate::{
    adamw_exprs, broadcast_compatible, lane_strides, pow_expr, sgd_exprs, CompileOptions,
    KernelExpr, ReduceOp,
};
use crate::{DTypeDisposition, ExecutableDTypePlan, RegionDTypeSpec, TargetDTypeCapabilities};
use effect_torch_graph::{Device, Node, NodeKind};
use effect_torch_runtime::{DType, DenseId};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::fmt;

/// Maximum per-element input lanes in one fused expression, based on backend
/// kernel buffer limits.
const MAX_LANES: usize = 30;
/// Maximum input and output buffers in a merged multi-output region.
const MAX_BUFFERS: usize = 31;
/// Maximum expression nodes in a merged multi-output region. This limits the
/// emitted kernel size.
const MAX_MERGED_OPS: usize = 512;

/// Dense identity of a selected code-generation region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionId(u32);

impl RegionId {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub fn from_index(index: usize) -> Option<Self> {
        u32::try_from(index).ok().map(Self)
    }
}

impl fmt::Display for RegionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl DenseId for RegionId {
    fn from_u32(value: u32) -> Self {
        Self(value)
    }

    fn as_u32(self) -> u32 {
        self.0
    }
}

/// The physical output of a region that implements a semantic value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionOutput {
    pub region: RegionId,
    pub index: u32,
}

/// One semantic value routed from a region output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SemanticOutput {
    pub semantic_node: DenseNodeId,
    pub index: u32,
}

/// A backend lowering unit in topological order. Backends do not need to
/// recover topology from semantic `Arc<Node>` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoweringUnit {
    Node(DenseNodeId),
    Region(RegionId),
}

/// Resolution of a semantic value at a region or independent-node boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueSource {
    Independent(DenseNodeId),
    Region(RegionOutput),
}

/// Structural work counters. Region selection creates no semantic nodes, so
/// `semantic_nodes_rebuilt` must remain zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OptimizationWork {
    pub graph_index_builds: usize,
    pub semantic_nodes_scanned: usize,
    pub semantic_nodes_rebuilt: usize,
    pub fusion_candidates: usize,
    pub capability_queries: usize,
    pub rejected_region_candidates: usize,
    pub multi_output_work_items: usize,
    pub multi_output_dependency_edges: usize,
    pub multi_output_dependency_passes: usize,
    pub multi_output_dependency_edge_visits: usize,
    pub multi_output_dependency_queries: usize,
    pub region_table_merges: usize,
    pub selected_regions: usize,
}

/// A fused expression and its output semantic node.
#[derive(Debug, Clone, PartialEq)]
pub struct ElementwiseOutput {
    pub semantic_node: DenseNodeId,
    pub expression: KernelExpr,
}

/// A chain of elementwise operations fused into one kernel over named lanes.
/// `inputs` lists boundary nodes read per element. `lane_strides` gives
/// their broadcast strides against `shape`. `output` computes the chain
/// endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct ElementwiseRegion {
    pub nodes: Box<[DenseNodeId]>,
    pub inputs: Box<[DenseNodeId]>,
    pub lane_strides: Box<[Box<[usize]>]>,
    pub output: ElementwiseOutput,
    pub shape: Box<[usize]>,
    pub dtype: DType,
    pub device: Device,
}

/// An elementwise chain folded into its terminating reduction. The reduce loop
/// evaluates the expression per input element without materializing the
/// chain's intermediate.
#[derive(Debug, Clone, PartialEq)]
pub struct ElementwiseReduceRegion {
    pub nodes: Box<[DenseNodeId]>,
    pub inputs: Box<[DenseNodeId]>,
    pub lane_strides: Box<[Box<[usize]>]>,
    pub output: DenseNodeId,
    pub expression: KernelExpr,
    pub op: ReduceOp,
    pub dims: Box<[usize]>,
    pub keepdims: bool,
    pub input_shape: Box<[usize]>,
    pub shape: Box<[usize]>,
    pub dtype: DType,
    pub device: Device,
}

/// An elementwise chain folded into an index reduction. The expression
/// retains its floating storage boundaries; the result has index dtype.
/// Backends must opt in and preserve their standalone tie and NaN policy.
#[derive(Debug, Clone, PartialEq)]
pub struct ElementwiseArgReduceRegion {
    pub nodes: Box<[DenseNodeId]>,
    pub inputs: Box<[DenseNodeId]>,
    pub lane_strides: Box<[Box<[usize]>]>,
    pub output: DenseNodeId,
    pub expression: KernelExpr,
    pub maximum: bool,
    pub dim: usize,
    pub input_shape: Box<[usize]>,
    pub shape: Box<[usize]>,
    pub dtype: DType,
    pub device: Device,
}

/// Two exact I64 argmax outputs sharing F32 logits; external uniforms retain RNG ownership.
#[derive(Debug, Clone, PartialEq)]
pub struct DualArgmaxRegion {
    pub nodes: Box<[DenseNodeId]>,
    pub inputs: Box<[DenseNodeId]>,
    pub plain: DenseNodeId,
    pub noisy: DenseNodeId,
    pub shape: Box<[usize]>,
    pub rows: usize,
    pub width: usize,
}

/// Exact stable route ranks from U32 expert IDs and explicit U32 positions.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertRouteRankRegion {
    pub nodes: Box<[DenseNodeId]>,
    pub inputs: Box<[DenseNodeId]>,
    pub output: DenseNodeId,
    pub rows: usize,
    pub routes: usize,
}

/// Scatter-add into positive zero, then a left-associated sum of every
/// destination slice in ascending order. Both nested addition orders and
/// every semantic storage rounding boundary are retained by the target.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderedScatterReduceRegion {
    pub nodes: Box<[DenseNodeId]>,
    /// Weighted source [rows,routes,width], compact indexes [rows,routes].
    /// With weighted_source, input zero is route-major BF16 [routes*rows,width]
    /// and input two is F32 weights [rows,routes,1].
    pub inputs: Box<[DenseNodeId]>,
    pub weighted_source: bool,
    pub output: DenseNodeId,
    pub shape: Box<[usize]>,
    pub routes: usize,
    pub dtype: DType,
    pub device: Device,
}

/// Several elementwise continuations of a shared prefix, merged into one
/// kernel that writes every continuation's output in a single pass over
/// `shape`.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiOutputRegion {
    pub nodes: Box<[DenseNodeId]>,
    pub inputs: Box<[DenseNodeId]>,
    pub lane_strides: Box<[Box<[usize]>]>,
    pub outputs: Box<[ElementwiseOutput]>,
    pub shape: Box<[usize]>,
    pub dtype: DType,
    pub device: Device,
}

/// A `Linear` GEMM with its residual `Add` absorbed into the epilogue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearResidualRegion {
    pub nodes: Box<[DenseNodeId]>,
    /// `[x, weight, bias, residual]` in native GEMM input order.
    pub inputs: Box<[DenseNodeId]>,
    pub output: DenseNodeId,
    pub shape: Box<[usize]>,
    pub dtype: DType,
    pub device: Device,
}

/// A `Linear` GEMM with a following `Gelu` absorbed into the epilogue.
/// A `dual` region also materializes the pre-activation value when another
/// part of the graph consumes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearGeluRegion {
    pub nodes: Box<[DenseNodeId]>,
    /// `[x, weight, bias]` in native GEMM input order.
    pub inputs: Box<[DenseNodeId]>,
    pub pre_activation: DenseNodeId,
    pub output: DenseNodeId,
    pub approximate: bool,
    pub dual: bool,
    pub shape: Box<[usize]>,
    pub dtype: DType,
    pub device: Device,
}

/// Hyperparameters of one fused AdamW step, mirrored into the region's
/// constant expressions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamWOptions {
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

/// Hyperparameters of one fused momentum-SGD step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SgdOptions {
    pub momentum: f64,
    pub dampening: f64,
    pub nesterov: bool,
    pub weight_decay: f64,
}

/// Identifies the physical output for an [`OptimizerOutput`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizerOutputKind {
    Parameter,
    FirstMoment,
    SecondMoment,
    Velocity,
}

/// Routes one physical optimizer output to the step and any
/// `AdamWOut` or `SgdOut` pickers. All routed nodes alias the same buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizerOutput {
    pub index: u32,
    pub parameter: u32,
    pub kind: OptimizerOutputKind,
    /// Semantic producers and selectors that alias this physical output.
    pub semantic_nodes: Box<[DenseNodeId]>,
}

/// One AdamW step fused into a single elementwise kernel over
/// `[param, grad, m, v]`, writing updated parameter and both moments.
#[derive(Debug, Clone, PartialEq)]
pub struct AdamWRegion {
    pub nodes: Box<[DenseNodeId]>,
    /// Tensor lanes followed by scalar lanes, in backend binding order.
    pub inputs: Box<[DenseNodeId]>,
    pub tensor_inputs: [DenseNodeId; 4],
    /// `[lr, 1 - beta1^t, 1 - beta2^t]`.
    pub scalar_inputs: [DenseNodeId; 3],
    pub outputs: Box<[OptimizerOutput]>,
    pub expressions: Box<[KernelExpr]>,
    pub options: AdamWOptions,
    pub shape: Box<[usize]>,
    pub dtype: DType,
    pub device: Device,
}

/// Up to four AdamW steps with identical hyperparameters and shapes in one
/// kernel. Interleaved lanes share launch and scalar-binding costs.
#[derive(Debug, Clone, PartialEq)]
pub struct AdamWGroupRegion {
    pub nodes: Box<[DenseNodeId]>,
    /// Interleaved `[param, grad, m, v]` lanes for each parameter, then scalars.
    pub inputs: Box<[DenseNodeId]>,
    pub parameter_inputs: Box<[[DenseNodeId; 4]]>,
    pub scalar_inputs: [DenseNodeId; 3],
    pub outputs: Box<[OptimizerOutput]>,
    pub expressions: Box<[KernelExpr]>,
    pub options: AdamWOptions,
    pub shape: Box<[usize]>,
    pub dtype: DType,
    pub device: Device,
}

/// One momentum-SGD step fused over `[param, grad, velocity]`, writing the
/// updated parameter and velocity.
#[derive(Debug, Clone, PartialEq)]
pub struct SgdRegion {
    pub nodes: Box<[DenseNodeId]>,
    /// Tensor lanes followed by scalar lanes, in backend binding order.
    pub inputs: Box<[DenseNodeId]>,
    pub tensor_inputs: [DenseNodeId; 3],
    /// `[lr, first]`, matching `KernelExpr::Scalar` indices.
    pub scalar_inputs: [DenseNodeId; 2],
    pub outputs: Box<[OptimizerOutput]>,
    pub expressions: Box<[KernelExpr]>,
    pub options: SgdOptions,
    pub shape: Box<[usize]>,
    pub dtype: DType,
    pub device: Device,
}

/// Three private BF16 normalizations followed by a residual and learned scale.
#[derive(Debug, Clone, PartialEq)]
pub struct FfnTailRegion {
    pub shape: Vec<usize>,
    pub nodes: Box<[DenseNodeId]>,
    /// Dense/expert/residual matrices, three weight vectors, and one scalar.
    pub inputs: Box<[DenseNodeId]>,
    pub output: DenseNodeId,
    pub rows: usize,
}

/// FFN tail and next learned RMS with two independently observable BF16 results.
#[derive(Debug, Clone, PartialEq)]
pub struct FfnNextNormRegion {
    pub nodes: Box<[DenseNodeId]>,
    pub inputs: Box<[DenseNodeId]>,
    pub outputs: [DenseNodeId; 2],
    pub residual_views: Box<[DenseNodeId]>,
    pub rows: usize,
}

/// A private pair of grouped projections with row-local gated activation.
/// Physical expert-sorted values never enter the semantic value map.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupedExpertGatedRegion {
    pub finalizer: Option<GroupedExpertFinalize>,
    pub nodes: Box<[DenseNodeId]>,
    /// Raw activation, gate/up bank, down bank, and identical route indexes.
    pub inputs: Box<[DenseNodeId]>,
    pub output: DenseNodeId,
    pub rows: usize,
    pub intermediate: usize,
    /// Two physical column lanes of the first expert-sorted projection.
    pub expression: KernelExpr,
}

/// Private inverse-mapped ordered weighted finalization of a grouped projection.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupedExpertFinalize {
    pub rows: usize,
    pub routes: usize,
}

/// Exact categorical entropy, recomputing private normalized probabilities.
/// Keeps each F32 boundary and the target's existing maximum/sum trees.
#[derive(Debug, Clone, PartialEq)]
pub struct EntropyRegion {
    pub nodes: Box<[DenseNodeId]>,
    pub inputs: Box<[DenseNodeId]>,
    pub output: DenseNodeId,
    pub shape: Box<[usize]>,
    pub width: usize,
}

/// BF16 -> F32 softmax -> BF16 with no escaping intermediate values.
/// The target must preserve the existing maximum/sum reduction trees and each
/// F32 subtraction, exponential and division rounding boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct Bf16SoftmaxRegion {
    pub nodes: Box<[DenseNodeId]>,
    pub inputs: Box<[DenseNodeId]>,
    pub output: DenseNodeId,
    pub shape: Box<[usize]>,
    pub width: usize,
}

/// F32 last-axis softmax at width 128 with private reduction intermediates.
/// Keeps the backend's short-reduction lane order; no dtype boundary is removed.
#[derive(Debug, Clone, PartialEq)]
pub struct SmallSoftmaxRegion {
    pub nodes: Box<[DenseNodeId]>,
    pub inputs: Box<[DenseNodeId]>,
    pub output: DenseNodeId,
    pub shape: Box<[usize]>,
    pub width: usize,
}

/// Exact width-128 softmax, probability-key top-eight, and selected-weight tail.
/// Only weights and indices escape; all F32 arithmetic boundaries are retained.
#[derive(Debug, Clone, PartialEq)]
pub struct RouterTailRegion {
    pub nodes: Box<[DenseNodeId]>,
    pub inputs: Box<[DenseNodeId]>,
    pub weights: DenseNodeId,
    pub indices: DenseNodeId,
    pub rows: usize,
}

/// Private unweighted V normalization consumed by one stateful KV attention.
#[derive(Debug, Clone, PartialEq)]
pub struct VNormKvAttentionRegion {
    pub nodes: Box<[DenseNodeId]>,
    pub inputs: Box<[DenseNodeId]>,
    pub output: DenseNodeId,
    pub normalized: DenseNodeId,
    pub source: DenseNodeId,
    pub q: DenseNodeId,
    pub k: DenseNodeId,
    pub shape: Box<[usize]>,
    pub eps: f64,
}

/// Private BF16 RMS followed by HalfSplit RoPE, retaining every BF16 boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct NormRopeRegion {
    pub nodes: Box<[DenseNodeId]>,
    pub inputs: Box<[DenseNodeId]>,
    pub output: DenseNodeId,
    pub normalized: DenseNodeId,
    pub source: DenseNodeId,
    pub weight: Option<DenseNodeId>,
    pub cosine: DenseNodeId,
    pub sine: DenseNodeId,
    pub shape: Box<[usize]>,
    pub eps: f64,
}

/// Private BF16 RMS followed by an ordered residual add, retaining the RMS store.
#[derive(Debug, Clone, PartialEq)]
pub struct RmsResidualRegion {
    pub nodes: Box<[DenseNodeId]>,
    /// Source, weight, residual, in physical kernel role order.
    pub inputs: Box<[DenseNodeId]>,
    pub output: DenseNodeId,
    pub shape: Box<[usize]>,
    pub rows: usize,
    pub eps: f64,
}

/// Exact attention residual and three FFN entrance normalizations. Four
/// physical outputs retain their semantic boundaries; residual reshapes alias R.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionFfnEntranceRegion {
    pub nodes: Box<[DenseNodeId]>,
    /// A, H, attention weight, dense weight, expert weight, router weight, F32 rho.
    pub inputs: Box<[DenseNodeId]>,
    pub outputs: [DenseNodeId; 4],
    pub residual_views: Box<[DenseNodeId]>,
    pub rows: usize,
}

/// RMS normalizations sharing one exact reduction and inverse. Each output
/// still evaluates the F32 input/inverse/weight products before its own store.
#[derive(Debug, Clone, PartialEq)]
pub struct SharedRmsNormRegion {
    pub nodes: Box<[DenseNodeId]>,
    pub inputs: Box<[DenseNodeId]>,
    pub source: DenseNodeId,
    pub weights: Box<[Option<DenseNodeId>]>,
    pub width: usize,
    pub elements: usize,
    pub eps: f64,
    pub dtype: DType,
}

/// A selected codegen region. Each variant covers a disjoint set of semantic
/// nodes and routes at least one semantic output. The module documentation
/// defines selection order and coverage rules.
#[derive(Debug, Clone, PartialEq)]
pub enum NativeRegion {
    DualArgmax(DualArgmaxRegion),
    RouterTail(RouterTailRegion),
    RmsResidual(RmsResidualRegion),
    AttentionFfnEntrance(AttentionFfnEntranceRegion),
    VNormKvAttention(VNormKvAttentionRegion),
    FfnTail(FfnTailRegion),
    FfnNextNorm(FfnNextNormRegion),
    GroupedExpertGated(GroupedExpertGatedRegion),
    Entropy(EntropyRegion),
    Bf16Softmax(Bf16SoftmaxRegion),
    SmallSoftmax(SmallSoftmaxRegion),
    SharedRmsNorm(SharedRmsNormRegion),
    NormRope(NormRopeRegion),
    Elementwise(ElementwiseRegion),
    ElementwiseReduce(ElementwiseReduceRegion),
    ElementwiseArgReduce(ElementwiseArgReduceRegion),
    ExpertRouteRank(ExpertRouteRankRegion),
    OrderedScatterReduce(OrderedScatterReduceRegion),
    MultiOutput(MultiOutputRegion),
    LinearResidual(LinearResidualRegion),
    LinearGelu(LinearGeluRegion),
    AdamW(AdamWRegion),
    AdamWGroup(AdamWGroupRegion),
    Sgd(SgdRegion),
}

impl NativeRegion {
    /// Semantic nodes covered by this region, sorted and deduplicated.
    pub fn nodes(&self) -> &[DenseNodeId] {
        match self {
            Self::DualArgmax(region) => &region.nodes,
            Self::RouterTail(region) => &region.nodes,
            Self::ExpertRouteRank(region) => &region.nodes,
            Self::RmsResidual(region) => &region.nodes,
            Self::AttentionFfnEntrance(region) => &region.nodes,
            Self::VNormKvAttention(region) => &region.nodes,
            Self::FfnTail(region) => &region.nodes,
            Self::FfnNextNorm(region) => &region.nodes,
            Self::GroupedExpertGated(region) => &region.nodes,
            Self::Entropy(region) => &region.nodes,
            Self::Bf16Softmax(region) => &region.nodes,
            Self::SmallSoftmax(region) => &region.nodes,
            Self::SharedRmsNorm(region) => &region.nodes,
            Self::NormRope(region) => &region.nodes,
            Self::Elementwise(region) => &region.nodes,
            Self::ElementwiseReduce(region) => &region.nodes,
            Self::ElementwiseArgReduce(region) => &region.nodes,
            Self::OrderedScatterReduce(region) => &region.nodes,
            Self::MultiOutput(region) => &region.nodes,
            Self::LinearResidual(region) => &region.nodes,
            Self::LinearGelu(region) => &region.nodes,
            Self::AdamW(region) => &region.nodes,
            Self::AdamWGroup(region) => &region.nodes,
            Self::Sgd(region) => &region.nodes,
        }
    }

    /// Boundary nodes read but not covered by this region.
    pub fn inputs(&self) -> &[DenseNodeId] {
        match self {
            Self::DualArgmax(region) => &region.inputs,
            Self::RouterTail(region) => &region.inputs,
            Self::ExpertRouteRank(region) => &region.inputs,
            Self::RmsResidual(region) => &region.inputs,
            Self::AttentionFfnEntrance(region) => &region.inputs,
            Self::VNormKvAttention(region) => &region.inputs,
            Self::FfnTail(region) => &region.inputs,
            Self::FfnNextNorm(region) => &region.inputs,
            Self::GroupedExpertGated(region) => &region.inputs,
            Self::Entropy(region) => &region.inputs,
            Self::Bf16Softmax(region) => &region.inputs,
            Self::SmallSoftmax(region) => &region.inputs,
            Self::SharedRmsNorm(region) => &region.inputs,
            Self::NormRope(region) => &region.inputs,
            Self::Elementwise(region) => &region.inputs,
            Self::ElementwiseReduce(region) => &region.inputs,
            Self::ElementwiseArgReduce(region) => &region.inputs,
            Self::OrderedScatterReduce(region) => &region.inputs,
            Self::MultiOutput(region) => &region.inputs,
            Self::LinearResidual(region) => &region.inputs,
            Self::LinearGelu(region) => &region.inputs,
            Self::AdamW(region) => &region.inputs,
            Self::AdamWGroup(region) => &region.inputs,
            Self::Sgd(region) => &region.inputs,
        }
    }

    /// Number of physical outputs the region writes.
    pub fn output_count(&self) -> usize {
        match self {
            Self::Elementwise(_)
            | Self::ElementwiseReduce(_)
            | Self::OrderedScatterReduce(_)
            | Self::ElementwiseArgReduce(_)
            | Self::LinearResidual(_) => 1,
            Self::ExpertRouteRank(_)
            | Self::RmsResidual(_)
            | Self::VNormKvAttention(_)
            | Self::FfnTail(_)
            | Self::GroupedExpertGated(_)
            | Self::Entropy(_)
            | Self::Bf16Softmax(_)
            | Self::SmallSoftmax(_)
            | Self::NormRope(_) => 1,
            Self::DualArgmax(_) | Self::RouterTail(_) | Self::FfnNextNorm(_) => 2,
            Self::AttentionFfnEntrance(_) => 4,
            Self::SharedRmsNorm(region) => region.nodes.len(),
            Self::MultiOutput(region) => region.outputs.len(),
            Self::LinearGelu(region) => 1 + usize::from(region.dual),
            Self::AdamW(region) => region.expressions.len(),
            Self::AdamWGroup(region) => region.expressions.len(),
            Self::Sgd(region) => region.expressions.len(),
        }
    }

    /// Semantic values routed from this region's physical outputs. Every
    /// entry must name a covered node and a valid output index.
    pub fn semantic_outputs(&self) -> Vec<SemanticOutput> {
        match self {
            Self::DualArgmax(region) => vec![
                SemanticOutput {
                    semantic_node: region.plain,
                    index: 0,
                },
                SemanticOutput {
                    semantic_node: region.noisy,
                    index: 1,
                },
            ],
            Self::RouterTail(region) => vec![
                SemanticOutput {
                    semantic_node: region.weights,
                    index: 0,
                },
                SemanticOutput {
                    semantic_node: region.indices,
                    index: 1,
                },
            ],
            Self::ExpertRouteRank(region) => vec![SemanticOutput {
                semantic_node: region.output,
                index: 0,
            }],
            Self::VNormKvAttention(region) => vec![SemanticOutput {
                semantic_node: region.output,
                index: 0,
            }],
            Self::AttentionFfnEntrance(region) => region
                .outputs
                .iter()
                .enumerate()
                .map(|(index, &semantic_node)| SemanticOutput {
                    semantic_node,
                    index: index as u32,
                })
                .chain(
                    region
                        .residual_views
                        .iter()
                        .map(|&semantic_node| SemanticOutput {
                            semantic_node,
                            index: 0,
                        }),
                )
                .collect(),
            Self::FfnNextNorm(region) => region
                .outputs
                .iter()
                .enumerate()
                .map(|(index, &semantic_node)| SemanticOutput {
                    semantic_node,
                    index: index as u32,
                })
                .chain(
                    region
                        .residual_views
                        .iter()
                        .map(|&semantic_node| SemanticOutput {
                            semantic_node,
                            index: 0,
                        }),
                )
                .collect(),
            Self::RmsResidual(region) => vec![SemanticOutput {
                semantic_node: region.output,
                index: 0,
            }],
            Self::FfnTail(region) => vec![SemanticOutput {
                semantic_node: region.output,
                index: 0,
            }],
            Self::GroupedExpertGated(region) => vec![SemanticOutput {
                semantic_node: region.output,
                index: 0,
            }],
            Self::Entropy(region) => vec![SemanticOutput {
                semantic_node: region.output,
                index: 0,
            }],
            Self::SmallSoftmax(region) => vec![SemanticOutput {
                semantic_node: region.output,
                index: 0,
            }],
            Self::Bf16Softmax(region) => vec![SemanticOutput {
                semantic_node: region.output,
                index: 0,
            }],
            Self::NormRope(region) => vec![SemanticOutput {
                semantic_node: region.output,
                index: 0,
            }],
            Self::SharedRmsNorm(region) => region
                .nodes
                .iter()
                .enumerate()
                .map(|(index, &semantic_node)| SemanticOutput {
                    semantic_node,
                    index: index as u32,
                })
                .collect(),
            Self::Elementwise(region) => vec![SemanticOutput {
                semantic_node: region.output.semantic_node,
                index: 0,
            }],
            Self::ElementwiseReduce(region) => vec![SemanticOutput {
                semantic_node: region.output,
                index: 0,
            }],
            Self::OrderedScatterReduce(region) => vec![SemanticOutput {
                semantic_node: region.output,
                index: 0,
            }],
            Self::ElementwiseArgReduce(region) => vec![SemanticOutput {
                semantic_node: region.output,
                index: 0,
            }],
            Self::MultiOutput(region) => region
                .outputs
                .iter()
                .enumerate()
                .map(|(index, output)| SemanticOutput {
                    semantic_node: output.semantic_node,
                    index: index as u32,
                })
                .collect(),
            Self::LinearResidual(region) => vec![SemanticOutput {
                semantic_node: region.output,
                index: 0,
            }],
            Self::LinearGelu(region) if region.dual => vec![
                SemanticOutput {
                    semantic_node: region.pre_activation,
                    index: 0,
                },
                SemanticOutput {
                    semantic_node: region.output,
                    index: 1,
                },
            ],
            Self::LinearGelu(region) => vec![SemanticOutput {
                semantic_node: region.output,
                index: 0,
            }],
            Self::AdamW(region) => optimizer_semantic_outputs(&region.outputs),
            Self::AdamWGroup(region) => optimizer_semantic_outputs(&region.outputs),
            Self::Sgd(region) => optimizer_semantic_outputs(&region.outputs),
        }
    }

    /// Returns the smallest dense index among routed outputs, or among covered
    /// nodes if no output exists. This key orders plan regions by semantic
    /// postorder.
    fn ordering_key(&self) -> usize {
        self.semantic_outputs()
            .iter()
            .map(|output| output.semantic_node.index())
            .min()
            .or_else(|| self.nodes().iter().map(|node| node.index()).min())
            .unwrap_or(usize::MAX)
    }
}

fn optimizer_semantic_outputs(outputs: &[OptimizerOutput]) -> Vec<SemanticOutput> {
    outputs
        .iter()
        .flat_map(|output| {
            output
                .semantic_nodes
                .iter()
                .copied()
                .map(move |semantic_node| SemanticOutput {
                    semantic_node,
                    index: output.index,
                })
        })
        .collect()
}

/// Selected implementation regions and semantic-output routing.
///
/// [`OptimizationPlan::validate`] checks that the plan tables are
/// index-parallel to the graph index and follow these rules:
///
/// - `node_region[n]` is the region covering node `n`, if any.
/// - `outputs[n]` is the region output for node `n`, if a region routes it.
///   An independent node has neither entry. An internal region node has a
///   `node_region` entry but no `outputs` entry.
/// - `lowering_order` is a topological order over regions and independent
///   nodes. Backends consume it directly.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationPlan<R = NativeRegion> {
    pub regions: Box<[R]>,
    pub(crate) region_dtype_plans: Box<[ExecutableDTypePlan]>,
    pub node_region: Box<[Option<RegionId>]>,
    pub outputs: Box<[Option<RegionOutput>]>,
    pub lowering_order: Box<[LoweringUnit]>,
    pub work: OptimizationWork,
}

impl OptimizationPlan<NativeRegion> {
    /// Selects native regions for one indexed graph under the given options.
    pub fn select<C: TargetDTypeCapabilities>(
        index: &GraphIndex,
        options: &CompileOptions,
        target: &C,
    ) -> Result<Self, String> {
        build_optimization_plan(index, options, target)
    }

    /// Selects regions from a prepared program's shared index and options.
    pub fn from_prepared<C: TargetDTypeCapabilities>(
        program: &crate::PreparedProgram,
        target: &C,
    ) -> Result<Self, String> {
        build_optimization_plan(&program.index, &program.options, target)
    }

    /// Resolves a semantic value to an independent lowering or region output.
    /// Internal region values never materialize and cannot resolve. Out-of-range
    /// nodes also fail.
    pub fn resolve(&self, node: DenseNodeId) -> Result<ValueSource, String> {
        let Some(output) = self.outputs.get(node.index()) else {
            return Err(format!("optimization: dense node {node} is out of range"));
        };
        if let Some(output) = output {
            return Ok(ValueSource::Region(*output));
        }
        match self.node_region.get(node.index()).copied().flatten() {
            None => Ok(ValueSource::Independent(node)),
            Some(region) => Err(format!(
                "optimization: semantic node {node} is internal to region {region}"
            )),
        }
    }

    /// Derives ownership tables, output routing, and lowering order from the
    /// region list, then compares them with the stored tables. It also requires
    /// every graph root to materialize a value. Selection validates before it
    /// returns. This method also catches invalid hand-built or mutated plans.
    pub fn validate(&self, index: &GraphIndex) -> Result<(), String> {
        let node_count = index.order.len();
        if self.region_dtype_plans.len() != self.regions.len() {
            return Err(
                "optimization: every selected region must retain one execution plan".into(),
            );
        }
        if self.node_region.len() != node_count || self.outputs.len() != node_count {
            return Err("optimization: plan tables do not match the graph index".to_string());
        }

        let mut expected_owner = vec![None; node_count];
        let mut expected_outputs = vec![None; node_count];
        for (region_index, region) in self.regions.iter().enumerate() {
            let region_id = region_id(region_index)?;
            if region.output_count() == 0 {
                return Err(format!("optimization: region {region_id} has no outputs"));
            }
            for &node in region.nodes() {
                if node.index() >= node_count {
                    return Err(format!(
                        "optimization: region {region_id} covers out-of-range node {node}"
                    ));
                }
                if let Some(other) = expected_owner[node.index()] {
                    return Err(format!(
                        "optimization: node {node} is covered by regions {other} and {region_id}"
                    ));
                }
                expected_owner[node.index()] = Some(region_id);
            }
            for input in region.inputs() {
                if input.index() >= node_count {
                    return Err(format!(
                        "optimization: region {region_id} has out-of-range input {input}"
                    ));
                }
                if region.nodes().contains(input) {
                    return Err(format!(
                        "optimization: region {region_id} lists covered node {input} as an input"
                    ));
                }
            }
            for output in region.semantic_outputs() {
                if output.semantic_node.index() >= node_count {
                    return Err(format!(
                        "optimization: region {region_id} routes an out-of-range semantic output"
                    ));
                }
                if output.index as usize >= region.output_count() {
                    return Err(format!(
                        "optimization: region {region_id} output {} is out of range",
                        output.index
                    ));
                }
                let route = RegionOutput {
                    region: region_id,
                    index: output.index,
                };
                if expected_owner[output.semantic_node.index()] != Some(region_id) {
                    return Err(format!(
                        "optimization: region {region_id} routes semantic node {} without covering it",
                        output.semantic_node
                    ));
                }
                match expected_outputs[output.semantic_node.index()] {
                    Some(existing) if existing != route => {
                        return Err(format!(
                            "optimization: semantic node {} has conflicting output routes",
                            output.semantic_node
                        ));
                    }
                    _ => expected_outputs[output.semantic_node.index()] = Some(route),
                }
            }
        }
        if self.node_region.as_ref() != expected_owner.as_slice()
            || self.outputs.as_ref() != expected_outputs.as_slice()
        {
            return Err("optimization: ownership or output tables are inconsistent".to_string());
        }

        for &root in index.roots.iter() {
            self.resolve(root).map_err(|_| {
                format!("optimization: semantic root {root} has no materialized value")
            })?;
        }
        let expected_order = build_lowering_order(index, self)?;
        if self.lowering_order.as_ref() != expected_order.as_slice() {
            return Err("optimization: lowering-unit order is inconsistent".to_string());
        }
        Ok(())
    }
}

/// Selects optimization regions for an indexed graph.
pub fn select_optimization_regions<C: TargetDTypeCapabilities>(
    index: &GraphIndex,
    options: &CompileOptions,
    target: &C,
) -> Result<OptimizationPlan, String> {
    build_optimization_plan(index, options, target)
}

/// Selects optimization regions for a prepared program.
pub fn optimize_prepared_program<C: TargetDTypeCapabilities>(
    program: &crate::PreparedProgram,
    target: &C,
) -> Result<OptimizationPlan, String> {
    OptimizationPlan::from_prepared(program, target)
}

/// Runs the selection pipeline described in the module documentation. With
/// `optimize` disabled, returns an empty region set and a lowering order of
/// independent nodes only. Every path ends in plan validation.
pub fn build_optimization_plan<C: TargetDTypeCapabilities>(
    index: &GraphIndex,
    options: &CompileOptions,
    target: &C,
) -> Result<OptimizationPlan, String> {
    if !options.optimize {
        let mut plan = OptimizationPlan {
            regions: Vec::new().into_boxed_slice(),
            region_dtype_plans: Box::new([]),
            node_region: vec![None; index.order.len()].into_boxed_slice(),
            outputs: vec![None; index.order.len()].into_boxed_slice(),
            lowering_order: index
                .order
                .iter()
                .enumerate()
                .map(|(node, _)| {
                    LoweringUnit::Node(
                        DenseNodeId::from_index(node)
                            .expect("graph index already validated its dense node count"),
                    )
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            work: OptimizationWork {
                graph_index_builds: index.work.graph_index_builds,
                ..OptimizationWork::default()
            },
        };
        plan.work.selected_regions = 0;
        plan.validate(index)?;
        return Ok(plan);
    }

    let mut selector = RegionSelector::new(index, options, target);
    if options.environment.fusion && options.environment.gemm_epilogues {
        selector.select_gemm_epilogues();
    }
    selector.select_optimizers();
    if options.environment.fusion {
        selector.select_vnorm_kv_attention();
        selector.select_attention_ffn_entrance();
        selector.select_ffn_next_norm();
        selector.select_ffn_tail();
        selector.select_rms_residual();
        selector.select_grouped_expert_finalize();
        selector.select_grouped_expert_gated();
        selector.select_entropy();
        selector.select_bf16_softmax();
        selector.select_router_tail();
        selector.select_dual_argmax();
        selector.select_small_softmax();
        selector.select_norm_rope();
        selector.select_shared_rms_norm();
        selector.select_expert_route_rank();
        selector.select_ordered_scatter_reduce();
        selector.select_elementwise()?;
        if options.environment.multi_output_fusion {
            selector.select_multi_output();
        }
    }
    selector.finish()
}

/// A candidate region during selection. A later merge can deactivate a draft
/// without removing it, so each subsequent pass checks `active`.
struct DraftRegion {
    region: NativeRegion,
    disposition: ExecutableDTypePlan,
    active: bool,
}

/// Mutable selection state for one graph. `reserved` marks nodes claimed by a
/// committed region. Later passes must skip them. `roots` marks values that
/// must materialize. `drafts` stores candidates in creation order.
struct RegionSelector<'a, C> {
    index: &'a GraphIndex,
    options: &'a CompileOptions,
    target: &'a C,
    error: Option<String>,
    roots: Vec<bool>,
    reserved: Vec<bool>,
    drafts: Vec<DraftRegion>,
    work: OptimizationWork,
}

impl<'a, C: TargetDTypeCapabilities> RegionSelector<'a, C> {
    fn new(index: &'a GraphIndex, options: &'a CompileOptions, target: &'a C) -> Self {
        let mut roots = vec![false; index.order.len()];
        for root in index.roots.iter() {
            roots[root.index()] = true;
        }
        Self {
            index,
            options,
            target,
            error: None,
            roots,
            reserved: vec![false; index.order.len()],
            drafts: Vec::new(),
            work: OptimizationWork {
                graph_index_builds: index.work.graph_index_builds,
                semantic_nodes_rebuilt: 0,
                ..OptimizationWork::default()
            },
        }
    }

    fn dense(&self, node: &std::sync::Arc<Node>) -> DenseNodeId {
        self.index
            .dense_id(node.id)
            .expect("every semantic child is present in GraphIndex")
    }

    fn classify_region(&mut self, region: &NativeRegion) -> Option<ExecutableDTypePlan> {
        self.work.fusion_candidates += 1;
        self.work.capability_queries += 1;
        let result = RegionDTypeSpec::new(self.index, region).and_then(|spec| {
            match self.target.classify_region(&spec) {
                DTypeDisposition::Unsupported(_) => Ok(None),
                disposition => {
                    crate::legalization::validate_disposition(&spec.operations, disposition)
                        .map(Some)
                }
            }
        });
        match result {
            Ok(Some(plan)) => Some(plan),
            Ok(None) => {
                self.work.rejected_region_candidates += 1;
                None
            }
            Err(error) => {
                self.error.get_or_insert(error);
                None
            }
        }
    }

    fn add_region(&mut self, region: NativeRegion) -> Option<usize> {
        let disposition = self.classify_region(&region)?;
        for node in region.nodes() {
            self.reserved[node.index()] = true;
        }
        let draft = self.drafts.len();
        self.drafts.push(DraftRegion {
            region,
            disposition,
            active: true,
        });
        Some(draft)
    }

    fn select_vnorm_kv_attention(&mut self) {
        self.work.semantic_nodes_scanned += self.index.order.len();
        for endpoint in (0..self.index.order.len()).rev() {
            let output = DenseNodeId::from_index(endpoint).unwrap();
            if let Some(region) = self.match_vnorm_kv_attention(output) {
                self.add_region(NativeRegion::VNormKvAttention(region));
            }
        }
    }

    fn match_vnorm_kv_attention(&self, output: DenseNodeId) -> Option<VNormKvAttentionRegion> {
        let endpoint = &self.index.order[output.index()];
        let NodeKind::KvAttention {
            q, k, v, rounding, ..
        } = &endpoint.kind
        else {
            return None;
        };
        let NodeKind::RmsNorm {
            x,
            weight: None,
            eps,
        } = &v.kind
        else {
            return None;
        };
        let [1, heads, tokens, dim] = v.shape.as_slice() else {
            return None;
        };
        if !matches!((*heads, *dim), (8, 256) | (2, 512))
            || !matches!(*tokens, 64 | 256)
            || eps.to_bits() != 1e-6_f64.to_bits()
            || *rounding != effect_torch_graph::AttentionRounding::Stepwise
            || q.shape.as_slice() != [1, 16, *tokens, *dim]
            || k.shape != v.shape
            || x.shape != v.shape
            || [endpoint, q, k, v, x]
                .iter()
                .any(|n| n.dtype != DType::BF16 || n.device != endpoint.device)
            || q.id == v.id
            || k.id == v.id
        {
            return None;
        }
        let normalized = self.dense(v);
        if self.reserved[output.index()]
            || self.reserved[normalized.index()]
            || self.roots[normalized.index()]
            || self
                .index
                .consumers_of(normalized)
                .is_none_or(|uses| uses != [output])
        {
            return None;
        }
        let q = self.dense(q);
        let k = self.dense(k);
        let source = self.dense(x);
        let mut inputs = vec![q, k, source];
        inputs.sort();
        inputs.dedup();
        Some(VNormKvAttentionRegion {
            nodes: vec![normalized, output].into_boxed_slice(),
            inputs: inputs.into_boxed_slice(),
            output,
            normalized,
            source,
            q,
            k,
            shape: v.shape.clone().into_boxed_slice(),
            eps: *eps,
        })
    }

    fn select_attention_ffn_entrance(&mut self) {
        self.work.semantic_nodes_scanned += self.index.order.len();
        for i in (0..self.index.order.len()).rev() {
            if let Some(region) =
                self.match_attention_ffn_entrance(DenseNodeId::from_index(i).unwrap())
            {
                self.add_region(NativeRegion::AttentionFfnEntrance(region));
            }
        }
    }

    fn match_attention_ffn_entrance(
        &self,
        output: DenseNodeId,
    ) -> Option<AttentionFfnEntranceRegion> {
        let endpoint = &self.index.order[output.index()];
        let NodeKind::Cast { a: scaled, .. } = &endpoint.kind else {
            return None;
        };
        let NodeKind::Mul { a: widened, b: rho } = &scaled.kind else {
            return None;
        };
        let NodeKind::Cast { a: learned, .. } = &widened.kind else {
            return None;
        };
        let NodeKind::Mul {
            a: normalized,
            b: router_weight,
        } = &learned.kind
        else {
            return None;
        };
        let NodeKind::RmsNorm {
            x: router_source,
            weight: None,
            eps,
        } = &normalized.kind
        else {
            return None;
        };
        let rows = match endpoint.shape.as_slice() {
            [rows, 2816] => *rows,
            _ => return None,
        };
        if !matches!(rows, 64 | 256)
            || eps.to_bits() != 1e-6f64.to_bits()
            || endpoint.dtype != DType::BF16
            || scaled.dtype != DType::F32
            || widened.dtype != DType::F32
            || learned.dtype != DType::BF16
            || normalized.dtype != DType::BF16
            || rho.dtype != DType::F32
            || rho.shape.iter().product::<usize>() != 1
            || [scaled, widened, learned, normalized]
                .iter()
                .any(|n| n.shape != endpoint.shape)
        {
            return None;
        }
        // Only reshape aliases of a newly dense residual may be absorbed. Route
        // every absorbed view as a semantic alias of output0, including roots.
        let canonical = |mut node: DenseNodeId| {
            let mut views = Vec::new();
            while let NodeKind::Reshape { a, .. } = &self.index.order[node.index()].kind {
                let current = &self.index.order[node.index()];
                if current.shape.last() != Some(&2816)
                    || a.shape.last() != Some(&2816)
                    || current.dtype != DType::BF16
                    || a.dtype != DType::BF16
                    || current.shape.iter().product::<usize>() != rows * 2816
                    || a.shape.iter().product::<usize>() != rows * 2816
                {
                    break;
                }
                views.push(node);
                node = self.dense(a);
            }
            (node, views)
        };
        let (residual_id, mut views) = canonical(self.dense(router_source));
        let residual = &self.index.order[residual_id.index()];
        let NodeKind::Add {
            a: hidden,
            b: post_attention,
        } = &residual.kind
        else {
            return None;
        };
        let NodeKind::RmsNorm {
            x: attention,
            weight: Some(attention_weight),
            eps: attention_eps,
        } = &post_attention.kind
        else {
            return None;
        };
        if attention_eps.to_bits() != eps.to_bits()
            || !matches!(residual.shape.as_slice(), [r, 2816] | [1, r, 2816] if *r == rows)
            || [attention, hidden, post_attention]
                .iter()
                .any(|n| n.shape != residual.shape)
        {
            return None;
        }
        let mut weighted = Vec::new();
        for (i, node) in self.index.order.iter().enumerate() {
            let NodeKind::RmsNorm {
                x,
                weight: Some(weight),
                eps: norm_eps,
            } = &node.kind
            else {
                continue;
            };
            let (source, source_views) = canonical(self.dense(x));
            if source == residual_id {
                if node.dtype != DType::BF16
                    || node.shape.iter().product::<usize>() != rows * 2816
                    || node.shape.last() != Some(&2816)
                    || norm_eps.to_bits() != eps.to_bits()
                {
                    return None;
                }
                weighted.push((DenseNodeId::from_index(i).unwrap(), self.dense(weight)));
                views.extend(source_views);
            }
        }
        if weighted.len() != 2 {
            return None;
        }
        let outputs = [residual_id, weighted[0].0, weighted[1].0, output];
        views.sort();
        views.dedup();
        let inputs = vec![
            self.dense(attention),
            self.dense(hidden),
            self.dense(attention_weight),
            weighted[0].1,
            weighted[1].1,
            self.dense(router_weight),
            self.dense(rho),
        ];
        let earliest = self.dense(post_attention).index();
        for (slot, &input) in inputs.iter().enumerate() {
            let node = &self.index.order[input.index()];
            if node.device != endpoint.device
                || node.dtype != if slot == 6 { DType::F32 } else { DType::BF16 }
                || (input.index() >= earliest
                    && !matches!(
                        node.kind,
                        NodeKind::Leaf(_)
                            | NodeKind::FromBytes { .. }
                            | NodeKind::Zeros { .. }
                            | NodeKind::Ones { .. }
                            | NodeKind::Full { .. }
                    ))
                || ((2..=5).contains(&slot) && node.shape.as_slice() != [2816])
            {
                return None;
            }
        }
        let private = [
            self.dense(post_attention),
            self.dense(normalized),
            self.dense(learned),
            self.dense(widened),
            self.dense(scaled),
        ];
        let mut nodes = private.to_vec();
        nodes.extend(outputs);
        nodes.extend(views.iter().copied());
        nodes.sort();
        nodes.dedup();
        if nodes.iter().any(|id| {
            self.reserved[id.index()] || self.index.order[id.index()].device != endpoint.device
        }) || private.iter().any(|id| {
            self.roots[id.index()]
                || self
                    .index
                    .consumers_of(*id)
                    .is_none_or(|uses| uses.iter().any(|use_| !nodes.contains(use_)))
        }) {
            return None;
        }
        // Contract only across known pure operations. In particular lazy input
        // binding, indexed operations, random/stateful work and checks must not
        // move before an earlier semantic failure. Unsupported graphs decompose.
        if self.index.order[earliest..=output.index()]
            .iter()
            .enumerate()
            .any(|(offset, node)| {
                let id = DenseNodeId::from_index(earliest + offset).unwrap();
                !nodes.contains(&id)
                    && !matches!(
                        node.kind,
                        NodeKind::Leaf(_)
                            | NodeKind::FromBytes { .. }
                            | NodeKind::Zeros { .. }
                            | NodeKind::Ones { .. }
                            | NodeKind::Full { .. }
                            | NodeKind::RmsNorm { .. }
                            | NodeKind::Add { .. }
                            | NodeKind::Sub { .. }
                            | NodeKind::Mul { .. }
                            | NodeKind::Div { .. }
                            | NodeKind::Neg { .. }
                            | NodeKind::Cast { .. }
                            | NodeKind::Reshape { .. }
                            | NodeKind::Permute { .. }
                            | NodeKind::Matmul { .. }
                            | NodeKind::Linear { .. }
                            | NodeKind::Gelu { .. }
                            | NodeKind::Tanh { .. }
                            | NodeKind::Exp { .. }
                    )
            })
        {
            return None;
        }
        // All non-private values, including R aliases, remain explicit outputs.
        // Late external bindings are rejected; late immutable constants carry
        // no invocation-time binding failure or dependency on the region.
        Some(AttentionFfnEntranceRegion {
            nodes: nodes.into_boxed_slice(),
            inputs: inputs.into_boxed_slice(),
            outputs,
            residual_views: views.into_boxed_slice(),
            rows,
        })
    }

    fn select_rms_residual(&mut self) {
        self.work.semantic_nodes_scanned += self.index.order.len();
        for endpoint in (0..self.index.order.len()).rev() {
            let output = DenseNodeId::from_index(endpoint).unwrap();
            if let Some(region) = self.match_rms_residual(output) {
                self.add_region(NativeRegion::RmsResidual(region));
            }
        }
    }

    fn match_rms_residual(&self, output: DenseNodeId) -> Option<RmsResidualRegion> {
        let endpoint = &self.index.order[output.index()];
        let NodeKind::Add {
            a: residual,
            b: normalized,
        } = &endpoint.kind
        else {
            return None;
        };
        let NodeKind::RmsNorm {
            x: source,
            weight: Some(weight),
            eps,
        } = &normalized.kind
        else {
            return None;
        };
        let rows = match endpoint.shape.as_slice() {
            [rows, 2816] | [1, rows, 2816] => *rows,
            _ => return None,
        };
        if !matches!(rows, 64 | 256)
            || !eps.is_finite()
            || *eps < 0.
            || [source, residual, normalized]
                .iter()
                .any(|n| n.shape != endpoint.shape)
            || [endpoint, source, residual, normalized]
                .iter()
                .any(|n| n.dtype != DType::BF16 || n.device != endpoint.device)
            || weight.shape.as_slice() != [2816]
            || weight.dtype != DType::BF16
            || weight.device != endpoint.device
        {
            return None;
        }
        let normalized = self.dense(normalized);
        if self.reserved[output.index()]
            || self.reserved[normalized.index()]
            || self.roots[normalized.index()]
            || self
                .index
                .consumers_of(normalized)
                .is_none_or(|uses| uses.iter().any(|&node| node != output))
        {
            return None;
        }
        let mut nodes = vec![normalized, output];
        nodes.sort();
        Some(RmsResidualRegion {
            nodes: nodes.into_boxed_slice(),
            inputs: vec![self.dense(source), self.dense(weight), self.dense(residual)]
                .into_boxed_slice(),
            output,
            shape: endpoint.shape.clone().into_boxed_slice(),
            rows,
            eps: *eps,
        })
    }

    fn select_ffn_next_norm(&mut self) {
        self.work.semantic_nodes_scanned += self.index.order.len();
        for i in (0..self.index.order.len()).rev() {
            if let Some(region) = self.match_ffn_next_norm(DenseNodeId::from_index(i).unwrap()) {
                self.add_region(NativeRegion::FfnNextNorm(region));
            }
        }
    }

    fn match_ffn_next_norm(&self, output: DenseNodeId) -> Option<FfnNextNormRegion> {
        let endpoint = &self.index.order[output.index()];
        let NodeKind::RmsNorm {
            x,
            weight: Some(weight),
            eps,
        } = &endpoint.kind
        else {
            return None;
        };
        if endpoint.dtype != DType::BF16
            || eps.to_bits() != 1e-6f64.to_bits()
            || weight.dtype != DType::BF16
            || weight.device != endpoint.device
            || weight.shape.as_slice() != [2816]
        {
            return None;
        }
        let mut source = self.dense(x);
        // Expose is a named forward identity. Keep every absorbed identity as
        // an explicit slot0 semantic output, even when rooted or later-read.
        // The semantic graph and exposure names remain unchanged.
        let mut views = Vec::new();
        while let NodeKind::Reshape { a, .. } | NodeKind::Expose { a, .. } =
            &self.index.order[source.index()].kind
        {
            let view = &self.index.order[source.index()];
            if view.dtype != DType::BF16
                || view.device != endpoint.device
                || view.shape.last() != Some(&2816)
                || a.shape.last() != Some(&2816)
                || view.shape.iter().product::<usize>() != endpoint.shape.iter().product::<usize>()
            {
                return None;
            }
            views.push(source);
            source = self.dense(a);
        }
        let tail = self.match_ffn_tail(source)?;
        if !matches!(tail.rows, 64 | 256)
            || endpoint.device != self.index.order[source.index()].device
            || endpoint.shape.last() != Some(&2816)
            || endpoint.shape.iter().product::<usize>() != tail.rows * 2816
        {
            return None;
        }
        let earliest = tail.nodes[0].index();
        let mut inputs = tail.inputs.into_vec();
        inputs.push(self.dense(weight));
        // The existing seven FFN-tail inputs already have the original tail's
        // dependency proof: notably the expert producer follows the dense RMS
        // in semantic postorder. Only the newly absorbed next-norm weight needs
        // this additional no-late-external-binding check.
        let weight_id = self.dense(weight);
        if weight_id.index() >= earliest
            && !matches!(
                weight.kind,
                NodeKind::Leaf(_)
                    | NodeKind::FromBytes { .. }
                    | NodeKind::Zeros { .. }
                    | NodeKind::Ones { .. }
                    | NodeKind::Full { .. }
            )
        {
            return None;
        }
        let mut nodes = tail.nodes.into_vec();
        nodes.push(output);
        nodes.extend(views.iter().copied());
        nodes.sort();
        nodes.dedup();
        if nodes.iter().any(|id| self.reserved[id.index()]) {
            return None;
        }
        // Do not contract across a failure, state update, random operation, or
        // unrelated numerical work. The observed model gap is aliases/constants.
        if self.index.order[source.index() + 1..output.index()]
            .iter()
            .enumerate()
            .any(|(offset, node)| {
                let id = DenseNodeId::from_index(source.index() + 1 + offset).unwrap();
                !nodes.contains(&id)
                    && !matches!(
                        node.kind,
                        NodeKind::Leaf(_)
                            | NodeKind::FromBytes { .. }
                            | NodeKind::Zeros { .. }
                            | NodeKind::Ones { .. }
                            | NodeKind::Full { .. }
                            | NodeKind::Reshape { .. }
                    )
            })
        {
            return None;
        }
        Some(FfnNextNormRegion {
            nodes: nodes.into_boxed_slice(),
            inputs: inputs.into_boxed_slice(),
            outputs: [source, output],
            residual_views: views.into_boxed_slice(),
            rows: tail.rows,
        })
    }

    fn select_ffn_tail(&mut self) {
        self.work.semantic_nodes_scanned += self.index.order.len();
        for i in (0..self.index.order.len()).rev() {
            let output = DenseNodeId::from_index(i).unwrap();
            if let Some(region) = self.match_ffn_tail(output) {
                self.add_region(NativeRegion::FfnTail(region));
            }
        }
    }

    fn match_ffn_tail(&self, output: DenseNodeId) -> Option<FfnTailRegion> {
        let endpoint = &self.index.order[output.index()];
        let NodeKind::Mul { a: added, b: scale } = &endpoint.kind else {
            return None;
        };
        let NodeKind::Add {
            a: residual,
            b: combined,
        } = &added.kind
        else {
            return None;
        };
        let NodeKind::RmsNorm {
            x: summed,
            weight: Some(wc),
            eps: ec,
        } = &combined.kind
        else {
            return None;
        };
        let NodeKind::Add {
            a: dense_norm,
            b: expert_norm,
        } = &summed.kind
        else {
            return None;
        };
        let NodeKind::RmsNorm {
            x: dense,
            weight: Some(wd),
            eps: ed,
        } = &dense_norm.kind
        else {
            return None;
        };
        let NodeKind::RmsNorm {
            x: expert,
            weight: Some(we),
            eps: ee,
        } = &expert_norm.kind
        else {
            return None;
        };
        let rows = match endpoint.shape.as_slice() {
            [rows, 2816] | [1, rows, 2816] => *rows,
            _ => return None,
        };
        if rows == 0
            || rows > 65535
            || [ec, ed, ee]
                .into_iter()
                .any(|eps| eps.to_bits() != 1e-6f64.to_bits())
            || [
                dense,
                expert,
                residual,
                dense_norm,
                expert_norm,
                summed,
                combined,
                added,
            ]
            .iter()
            .any(|node| node.shape != endpoint.shape)
            || [wd, we, wc]
                .iter()
                .any(|node| node.shape.as_slice() != [2816])
            || scale.shape.as_slice() != [1]
            || [
                endpoint,
                added,
                combined,
                summed,
                dense_norm,
                expert_norm,
                dense,
                expert,
                residual,
                wd,
                we,
                wc,
                scale,
            ]
            .iter()
            .any(|node| node.dtype != DType::BF16 || node.device != endpoint.device)
        {
            return None;
        }
        let mut nodes = [endpoint, added, combined, summed, dense_norm, expert_norm]
            .into_iter()
            .map(|node| self.dense(node))
            .collect::<Vec<_>>();
        nodes.sort();
        nodes.dedup();
        for &node in &nodes {
            if self.reserved[node.index()]
                || (node != output
                    && (self.roots[node.index()]
                        || self
                            .index
                            .consumers_of(node)
                            .into_iter()
                            .flatten()
                            .any(|consumer| !nodes.contains(consumer))))
            {
                return None;
            }
        }
        Some(FfnTailRegion {
            shape: endpoint.shape.clone(),
            nodes: nodes.into_boxed_slice(),
            inputs: [dense, expert, residual, wd, we, wc, scale]
                .into_iter()
                .map(|node| self.dense(node))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            output,
            rows,
        })
    }

    fn select_grouped_expert_finalize(&mut self) {
        self.work.semantic_nodes_scanned += self.index.order.len();
        for i in (0..self.index.order.len()).rev() {
            let output = DenseNodeId::from_index(i).unwrap();
            let Some(base) = self.match_ordered_scatter_reduce(output) else {
                continue;
            };
            let Some(finalizer) = self.match_ordered_scatter_weighted_source(&base) else {
                continue;
            };
            let Some(mut gated) = self.match_grouped_expert_gated(finalizer.inputs[0]) else {
                continue;
            };
            if finalizer.shape[1] != 2816
                || finalizer.shape[0] == 0
                || !(1..=32).contains(&finalizer.routes)
                || finalizer.shape[0].checked_mul(finalizer.routes) != Some(gated.rows)
            {
                continue;
            }
            let mut nodes = gated.nodes.to_vec();
            nodes.extend_from_slice(&finalizer.nodes);
            normalize_nodes(&mut nodes);
            // The down projection can no longer be a semantic output: only
            // this union's final reduced endpoint may escape the private region.
            if nodes.iter().any(|&node| {
                self.reserved[node.index()]
                    || (node != output
                        && (self.roots[node.index()]
                            || self.index.consumers[node.index()]
                                .iter()
                                .any(|c| !nodes.contains(c))))
            }) {
                continue;
            }
            let mut inputs = gated.inputs.to_vec();
            inputs.extend_from_slice(&finalizer.inputs[1..]);
            gated.nodes = nodes.into_boxed_slice();
            gated.inputs = inputs.into_boxed_slice();
            gated.output = output;
            gated.finalizer = Some(GroupedExpertFinalize {
                rows: finalizer.shape[0],
                routes: finalizer.routes,
            });
            self.add_region(NativeRegion::GroupedExpertGated(gated));
        }
    }

    fn select_grouped_expert_gated(&mut self) {
        self.work.semantic_nodes_scanned += self.index.order.len();
        for i in (0..self.index.order.len()).rev() {
            let output = DenseNodeId::from_index(i).unwrap();
            if let Some(region) = self.match_grouped_expert_gated(output) {
                self.add_region(NativeRegion::GroupedExpertGated(region));
            }
        }
    }

    fn match_grouped_expert_gated(&self, output: DenseNodeId) -> Option<GroupedExpertGatedRegion> {
        let endpoint = &self.index.order[output.index()];
        let NodeKind::GroupedExpertLinearRows {
            x: product,
            weight: down,
            indexes,
        } = &endpoint.kind
        else {
            return None;
        };
        // Start deliberately narrow: exact model graph, no shape-changing views.
        // Even identity reshapes fall back until separately tested.
        let NodeKind::Mul {
            a: activated,
            b: up,
        } = &product.kind
        else {
            return None;
        };
        let NodeKind::Gelu {
            a: gate,
            approximate: true,
        } = &activated.kind
        else {
            return None;
        };
        let NodeKind::Slice {
            a: first,
            ranges: gate_ranges,
        } = &gate.kind
        else {
            return None;
        };
        let NodeKind::Slice {
            a: up_first,
            ranges: up_ranges,
        } = &up.kind
        else {
            return None;
        };
        let NodeKind::GroupedExpertLinearRows {
            x: raw,
            weight: gate_up,
            indexes: first_indexes,
        } = &first.kind
        else {
            return None;
        };
        if endpoint.shape.len() != 2 {
            return None;
        }
        let rows = endpoint.shape[0];
        let intermediate = 704;
        if rows == 0
            || rows > u32::MAX as usize
            || endpoint.shape.as_slice() != [rows, 2816]
            || raw.shape.as_slice() != [rows, 2816]
            || first.shape.as_slice() != [rows, 1408]
            || gate_up.shape.as_slice() != [128, 1408, 2816]
            || down.shape.as_slice() != [128, 2816, 704]
            || indexes.shape.as_slice() != [rows]
            || !matches!(indexes.dtype, DType::U32 | DType::I64)
            || first_indexes.id != indexes.id
            || up_first.id != first.id
            || gate_ranges.as_slice() != [(0, rows, 1), (0, 704, 1)]
            || up_ranges.as_slice() != [(0, rows, 1), (704, 1408, 1)]
            || [gate, up, activated, product]
                .iter()
                .any(|node| node.shape.as_slice() != [rows, 704])
            || [
                endpoint, raw, first, gate_up, down, gate, up, activated, product,
            ]
            .iter()
            .any(|node| node.dtype != DType::BF16 || node.device != endpoint.device)
            || indexes.device != endpoint.device
        {
            return None;
        }
        let mut nodes = [endpoint, first, gate, up, activated, product]
            .into_iter()
            .map(|node| self.dense(node))
            .collect::<Vec<_>>();
        nodes.sort();
        nodes.dedup();
        for &node in &nodes {
            if self.reserved[node.index()]
                || (node != output
                    && (self.roots[node.index()]
                        || self
                            .index
                            .consumers_of(node)
                            .into_iter()
                            .flatten()
                            .any(|consumer| !nodes.contains(consumer))))
            {
                return None;
            }
        }
        let expression = KernelExpr::Mul(
            Box::new(
                KernelExpr::GeluTanh(Box::new(KernelExpr::Input(0)))
                    .semantic(self.dense(activated), DType::BF16),
            ),
            Box::new(KernelExpr::Input(1)),
        )
        .semantic(self.dense(product), DType::BF16);
        Some(GroupedExpertGatedRegion {
            finalizer: None,
            nodes: nodes.into_boxed_slice(),
            inputs: vec![
                self.dense(raw),
                self.dense(gate_up),
                self.dense(down),
                self.dense(indexes),
            ]
            .into_boxed_slice(),
            output,
            rows,
            intermediate,
            expression,
        })
    }

    fn select_entropy(&mut self) {
        self.work.semantic_nodes_scanned += self.index.order.len();
        for i in (0..self.index.order.len()).rev() {
            let output = DenseNodeId::from_index(i).unwrap();
            if let Some(region) = self.match_entropy(output) {
                self.add_region(NativeRegion::Entropy(region));
            }
        }
    }

    fn match_entropy(&self, output: DenseNodeId) -> Option<EntropyRegion> {
        let endpoint = &self.index.order[output.index()];
        let NodeKind::Neg { a: entropy_sum } = &endpoint.kind else {
            return None;
        };
        let NodeKind::Sum {
            a: product,
            dims,
            keepdims: false,
        } = &entropy_sum.kind
        else {
            return None;
        };
        let NodeKind::Mul {
            a: clamped,
            b: probability,
        } = &product.kind
        else {
            return None;
        };
        let NodeKind::Maximum {
            a: normalized,
            b: minimum,
        } = &clamped.kind
        else {
            return None;
        };
        let NodeKind::Full { value, .. } = &minimum.kind else {
            return None;
        };
        if *value != -(f32::MAX as f64) {
            return None;
        }
        let NodeKind::Div {
            a: exponential,
            b: denominator,
        } = &probability.kind
        else {
            return None;
        };
        let NodeKind::Sum {
            a: summed,
            dims: sum_dims,
            keepdims: true,
        } = &denominator.kind
        else {
            return None;
        };
        if summed.id != exponential.id || sum_dims != dims {
            return None;
        }
        let NodeKind::Exp { a: shifted } = &exponential.kind else {
            return None;
        };
        let NodeKind::Sub {
            a: shifted_source,
            b: maximum,
        } = &shifted.kind
        else {
            return None;
        };
        let NodeKind::Max {
            a: maximized,
            dims: max_dims,
            keepdims: true,
        } = &maximum.kind
        else {
            return None;
        };
        if shifted_source.id != normalized.id || maximized.id != normalized.id || max_dims != dims {
            return None;
        }
        let NodeKind::Sub {
            a: source,
            b: logsumexp,
        } = &normalized.kind
        else {
            return None;
        };
        let rank = source.shape.len();
        if rank == 0 || dims.as_slice() != [rank - 1] || source.shape.contains(&0) {
            return None;
        }
        let width = source.shape[rank - 1];
        let rows = source.shape[..rank - 1]
            .iter()
            .try_fold(1usize, |rows, &extent| rows.checked_mul(extent))?;
        // The specialized kernel executes one CTA per row; larger programs
        // retain the ordinary reduction path instead of a capped launch.
        if width < 4096 || rows > 65535 {
            return None;
        }
        let NodeKind::Add {
            a: source_maximum,
            b: logarithm,
        } = &logsumexp.kind
        else {
            return None;
        };
        let NodeKind::Log { a: source_sum } = &logarithm.kind else {
            return None;
        };
        let NodeKind::Sum {
            a: source_exponential,
            dims: source_sum_dims,
            keepdims: true,
        } = &source_sum.kind
        else {
            return None;
        };
        let NodeKind::Exp { a: source_shifted } = &source_exponential.kind else {
            return None;
        };
        let NodeKind::Sub {
            a: shifted_input,
            b: shifted_maximum,
        } = &source_shifted.kind
        else {
            return None;
        };
        let NodeKind::Max {
            a: maximized_input,
            dims: source_max_dims,
            keepdims: true,
        } = &source_maximum.kind
        else {
            return None;
        };
        if shifted_input.id != source.id
            || maximized_input.id != source.id
            || shifted_maximum.id != source_maximum.id
            || source_sum_dims != dims
            || source_max_dims != dims
        {
            return None;
        }
        let mut nodes = [
            endpoint,
            entropy_sum,
            product,
            clamped,
            minimum,
            probability,
            denominator,
            exponential,
            shifted,
            maximum,
            normalized,
            logsumexp,
            logarithm,
            source_sum,
            source_exponential,
            source_shifted,
            source_maximum,
        ]
        .into_iter()
        .map(|n| self.dense(n))
        .collect::<Vec<_>>();
        nodes.sort();
        nodes.dedup();
        if source.dtype != DType::F32
            || nodes
                .iter()
                .any(|n| self.index.order[n.index()].dtype != DType::F32)
        {
            return None;
        }
        for &node in &nodes {
            if self.reserved[node.index()]
                || (node != output
                    && (self.roots[node.index()]
                        || self
                            .index
                            .consumers_of(node)
                            .into_iter()
                            .flatten()
                            .any(|consumer| !nodes.contains(consumer))))
            {
                return None;
            }
        }
        Some(EntropyRegion {
            nodes: nodes.into_boxed_slice(),
            inputs: vec![self.dense(source)].into_boxed_slice(),
            output,
            shape: source.shape.clone().into_boxed_slice(),
            width,
        })
    }

    fn select_dual_argmax(&mut self) {
        self.work.semantic_nodes_scanned += self.index.order.len();
        for i in (0..self.index.order.len()).rev() {
            if let Some(region) = self.match_dual_argmax(DenseNodeId::from_index(i).unwrap()) {
                self.add_region(NativeRegion::DualArgmax(region));
            }
        }
    }

    fn match_dual_argmax(&self, noisy: DenseNodeId) -> Option<DualArgmaxRegion> {
        let endpoint = &self.index.order[noisy.index()];
        let NodeKind::Argmax { a: added, dim } = &endpoint.kind else {
            return None;
        };
        let NodeKind::Add {
            a: processed,
            b: noise,
        } = &added.kind
        else {
            return None;
        };
        let NodeKind::Neg { a: outer_log } = &noise.kind else {
            return None;
        };
        let NodeKind::Log { a: negative } = &outer_log.kind else {
            return None;
        };
        let NodeKind::Neg { a: inner_log } = &negative.kind else {
            return None;
        };
        let NodeKind::Log { a: uniform } = &inner_log.kind else {
            return None;
        };
        let rank = processed.shape.len();
        let width = *processed.shape.last()?;
        let rows = endpoint
            .shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))?;
        if *dim + 1 != rank
            || !(4096..=u32::MAX as usize).contains(&width)
            || !(1..=65535).contains(&rows)
            || endpoint.dtype != DType::I64
            || [
                processed, uniform, added, noise, outer_log, negative, inner_log,
            ]
            .iter()
            .any(|n| n.dtype != DType::F32 || n.shape != processed.shape)
        {
            return None;
        }
        let source = self.dense(processed);
        let plain = self.index.consumers_of(source)?.iter().copied().find(|&id| {
            let node = &self.index.order[id.index()];
            !self.reserved[id.index()] && node.dtype == DType::I64 && node.shape == endpoint.shape
                && matches!(&node.kind, NodeKind::Argmax { a, dim: axis } if a.id == processed.id && axis == dim)
        })?;
        let mut nodes = [
            noisy,
            plain,
            self.dense(added),
            self.dense(noise),
            self.dense(outer_log),
            self.dense(negative),
            self.dense(inner_log),
        ]
        .to_vec();
        nodes.sort();
        nodes.dedup();
        for &node in &nodes {
            if self.reserved[node.index()]
                || (node != plain
                    && node != noisy
                    && (self.roots[node.index()]
                        || self
                            .index
                            .consumers_of(node)
                            .into_iter()
                            .flatten()
                            .any(|consumer| !nodes.contains(consumer))))
            {
                return None;
            }
        }
        Some(DualArgmaxRegion {
            nodes: nodes.into_boxed_slice(),
            inputs: vec![source, self.dense(uniform)].into_boxed_slice(),
            plain,
            noisy,
            shape: endpoint.shape.clone().into_boxed_slice(),
            rows,
            width,
        })
    }

    fn select_router_tail(&mut self) {
        self.work.semantic_nodes_scanned += self.index.order.len();
        for i in (0..self.index.order.len()).rev() {
            let output = DenseNodeId::from_index(i).unwrap();
            if let Some(region) = self.match_router_tail(output) {
                self.add_region(NativeRegion::RouterTail(region));
            }
        }
    }

    fn match_router_tail(&self, output: DenseNodeId) -> Option<RouterTailRegion> {
        let endpoint = &self.index.order[output.index()];
        let NodeKind::Mul {
            a: normalized,
            b: scale_float,
        } = &endpoint.kind
        else {
            return None;
        };
        if endpoint.dtype != DType::F32
            || endpoint.shape.len() != 2
            || endpoint.shape[1] != 8
            || !(1..=65535).contains(&endpoint.shape[0])
        {
            return None;
        }
        let rows = endpoint.shape[0];
        let NodeKind::Div {
            a: selected,
            b: total,
        } = &normalized.kind
        else {
            return None;
        };
        let NodeKind::Sum {
            a: summed,
            dims,
            keepdims: true,
        } = &total.kind
        else {
            return None;
        };
        if dims.as_slice() != [1] || summed.id != selected.id {
            return None;
        }
        let NodeKind::Gather {
            a: probabilities,
            dim: 1,
            indexes: indices,
        } = &selected.kind
        else {
            return None;
        };
        let NodeKind::TopKIndices { a: sorted, k: 8 } = &indices.kind else {
            return None;
        };
        if sorted.id != probabilities.id
            || indices.dtype != DType::U32
            || indices.shape != endpoint.shape
            || probabilities.shape != [rows, 128]
        {
            return None;
        }
        let softmax = self.match_small_softmax(self.dense(probabilities))?;
        let mut nodes = softmax.nodes.to_vec();
        let scale_view = match &scale_float.kind {
            NodeKind::Cast {
                a,
                dtype: DType::F32,
            } => {
                nodes.push(self.dense(scale_float));
                a
            }
            _ if scale_float.dtype == DType::F32 => scale_float,
            _ => return None,
        };
        let NodeKind::Reshape { a: taken, .. } = &scale_view.kind else {
            return None;
        };
        let NodeKind::IndexSelect {
            a: scale,
            dim: 0,
            indexes: flat_indices,
        } = &taken.kind
        else {
            return None;
        };
        let NodeKind::Reshape { a: unflattened, .. } = &flat_indices.kind else {
            return None;
        };
        if unflattened.id != indices.id
            || flat_indices.shape != [rows.checked_mul(8)?]
            || scale.shape != [128]
            || !matches!(scale.dtype, DType::BF16 | DType::F32)
            || scale_view.shape != endpoint.shape
            || scale_view.dtype != scale.dtype
            || scale_float.dtype != DType::F32
            || scale_float.shape != endpoint.shape
            || [endpoint, normalized, selected, total]
                .iter()
                .any(|n| n.dtype != DType::F32)
            || normalized.shape != endpoint.shape
            || selected.shape != endpoint.shape
            || total.shape != [rows, 1]
        {
            return None;
        }
        nodes.extend(
            [
                endpoint,
                normalized,
                selected,
                total,
                indices,
                scale_view,
                taken,
                flat_indices,
            ]
            .into_iter()
            .map(|n| self.dense(n)),
        );
        nodes.sort();
        nodes.dedup();
        let indices = self.dense(indices);
        for &node in &nodes {
            if self.reserved[node.index()]
                || (node != output
                    && node != indices
                    && (self.roots[node.index()]
                        || self
                            .index
                            .consumers_of(node)
                            .into_iter()
                            .flatten()
                            .any(|consumer| !nodes.contains(consumer))))
            {
                return None;
            }
        }
        Some(RouterTailRegion {
            nodes: nodes.into_boxed_slice(),
            inputs: vec![softmax.inputs[0], self.dense(scale)].into_boxed_slice(),
            weights: output,
            indices,
            rows,
        })
    }

    fn select_small_softmax(&mut self) {
        self.work.semantic_nodes_scanned += self.index.order.len();
        for i in (0..self.index.order.len()).rev() {
            let output = DenseNodeId::from_index(i).unwrap();
            if let Some(region) = self.match_small_softmax(output) {
                self.add_region(NativeRegion::SmallSoftmax(region));
            }
        }
    }

    fn match_small_softmax(&self, output: DenseNodeId) -> Option<SmallSoftmaxRegion> {
        let endpoint = &self.index.order[output.index()];
        let NodeKind::Div {
            a: exponential,
            b: denominator,
        } = &endpoint.kind
        else {
            return None;
        };
        let NodeKind::Sum {
            a: summed,
            dims,
            keepdims: true,
        } = &denominator.kind
        else {
            return None;
        };
        let rank = endpoint.shape.len();
        if rank == 0
            || endpoint.shape[rank - 1] != 128
            || endpoint.shape.contains(&0)
            || dims.as_slice() != [rank - 1]
            || summed.id != exponential.id
        {
            return None;
        }
        let NodeKind::Exp { a: shifted } = &exponential.kind else {
            return None;
        };
        let NodeKind::Sub {
            a: source,
            b: maximum,
        } = &shifted.kind
        else {
            return None;
        };
        let NodeKind::Max {
            a: maximized,
            dims: max_dims,
            keepdims: true,
        } = &maximum.kind
        else {
            return None;
        };
        if max_dims != dims
            || maximized.id != source.id
            || source.shape != endpoint.shape
            || source.dtype != DType::F32
        {
            return None;
        }
        let covered = [endpoint, denominator, exponential, shifted, maximum];
        if covered.iter().any(|node| node.dtype != DType::F32) {
            return None;
        }
        let mut nodes = covered
            .into_iter()
            .map(|node| self.dense(node))
            .collect::<Vec<_>>();
        nodes.sort();
        nodes.dedup();
        for &node in &nodes {
            if self.reserved[node.index()]
                || (node != output
                    && (self.roots[node.index()]
                        || self
                            .index
                            .consumers_of(node)
                            .into_iter()
                            .flatten()
                            .any(|consumer| !nodes.contains(consumer))))
            {
                return None;
            }
        }
        Some(SmallSoftmaxRegion {
            nodes: nodes.into_boxed_slice(),
            inputs: vec![self.dense(source)].into_boxed_slice(),
            output,
            shape: endpoint.shape.clone().into_boxed_slice(),
            width: 128,
        })
    }

    fn select_bf16_softmax(&mut self) {
        self.work.semantic_nodes_scanned += self.index.order.len();
        for i in (0..self.index.order.len()).rev() {
            let output = DenseNodeId::from_index(i).unwrap();
            if let Some(region) = self.match_bf16_softmax(output) {
                self.add_region(NativeRegion::Bf16Softmax(region));
            }
        }
    }

    fn match_bf16_softmax(&self, output: DenseNodeId) -> Option<Bf16SoftmaxRegion> {
        let endpoint = &self.index.order[output.index()];
        let NodeKind::Cast {
            a: quotient,
            dtype: DType::BF16,
        } = &endpoint.kind
        else {
            return None;
        };
        let NodeKind::Div {
            a: exponential,
            b: denominator,
        } = &quotient.kind
        else {
            return None;
        };
        let NodeKind::Sum {
            a: summed,
            dims,
            keepdims: true,
        } = &denominator.kind
        else {
            return None;
        };
        let rank = endpoint.shape.len();
        if rank == 0 || dims.as_slice() != [rank - 1] || summed.id != exponential.id {
            return None;
        }
        let width = endpoint.shape[rank - 1];
        if width < 4096 || endpoint.shape.contains(&0) {
            return None;
        }
        let NodeKind::Exp { a: shifted } = &exponential.kind else {
            return None;
        };
        let NodeKind::Sub {
            a: floating,
            b: maximum,
        } = &shifted.kind
        else {
            return None;
        };
        let NodeKind::Max {
            a: maximized,
            dims: max_dims,
            keepdims: true,
        } = &maximum.kind
        else {
            return None;
        };
        if max_dims != dims || maximized.id != floating.id {
            return None;
        }
        let NodeKind::Cast {
            a: source,
            dtype: DType::F32,
        } = &floating.kind
        else {
            return None;
        };
        if source.dtype != DType::BF16 || source.shape != endpoint.shape {
            return None;
        }
        let mut nodes = [
            endpoint,
            quotient,
            denominator,
            exponential,
            shifted,
            maximum,
            floating,
        ]
        .into_iter()
        .map(|node| self.dense(node))
        .collect::<Vec<_>>();
        nodes.sort();
        nodes.dedup();
        for &node in &nodes {
            if self.reserved[node.index()] {
                return None;
            }
            if node != output
                && (self.roots[node.index()]
                    || self
                        .index
                        .consumers_of(node)
                        .into_iter()
                        .flatten()
                        .any(|consumer| !nodes.contains(consumer)))
            {
                return None;
            }
        }
        Some(Bf16SoftmaxRegion {
            nodes: nodes.into_boxed_slice(),
            inputs: vec![self.dense(source)].into_boxed_slice(),
            output,
            shape: endpoint.shape.clone().into_boxed_slice(),
            width,
        })
    }

    fn select_norm_rope(&mut self) {
        self.work.semantic_nodes_scanned += self.index.order.len();
        for i in (0..self.index.order.len()).rev() {
            let output = DenseNodeId::from_index(i).unwrap();
            if let Some(region) = self.match_norm_rope(output) {
                self.add_region(NativeRegion::NormRope(region));
            }
        }
    }

    fn match_norm_rope(&self, output: DenseNodeId) -> Option<NormRopeRegion> {
        let endpoint = &self.index.order[output.index()];
        let shape = endpoint.shape.as_slice();
        if endpoint.dtype != DType::BF16
            || shape.len() != 4
            || shape[0] != 1
            || !matches!(shape[1], 8 | 16)
            || shape[2] != 256
            || !matches!(shape[3], 256 | 512)
        {
            return None;
        }
        let width = shape[3];
        let NodeKind::Add {
            a: direct,
            b: cross,
        } = &endpoint.kind
        else {
            return None;
        };
        let NodeKind::Mul {
            a: normalized,
            b: cosine,
        } = &direct.kind
        else {
            return None;
        };
        let NodeKind::Mul {
            a: rotated,
            b: sine,
        } = &cross.kind
        else {
            return None;
        };
        let NodeKind::RmsNorm {
            x: source,
            weight,
            eps,
        } = &normalized.kind
        else {
            return None;
        };
        if normalized.shape != endpoint.shape
            || source.shape != endpoint.shape
            || source.dtype != DType::BF16
            || normalized.dtype != DType::BF16
            || weight
                .as_ref()
                .is_some_and(|w| w.dtype != DType::BF16 || w.shape != [width])
            || [cosine, sine]
                .iter()
                .any(|t| t.dtype != DType::BF16 || t.shape != [1, 1, 256, width])
        {
            return None;
        }
        let mut nodes = vec![
            self.dense(endpoint),
            self.dense(direct),
            self.dense(cross),
            self.dense(normalized),
        ];
        let mut rotation = rotated;
        while let NodeKind::Reshape { a, .. } = &rotation.kind {
            if rotation.shape != a.shape || rotation.dtype != a.dtype {
                return None;
            }
            nodes.push(self.dense(rotation));
            rotation = a;
        }
        let NodeKind::Concat {
            a: negative,
            b: first,
            dim,
        } = &rotation.kind
        else {
            return None;
        };
        let NodeKind::Neg { a: second } = &negative.kind else {
            return None;
        };
        let NodeKind::Slice {
            a: first_source,
            ranges: first_ranges,
        } = &first.kind
        else {
            return None;
        };
        let NodeKind::Slice {
            a: second_source,
            ranges: second_ranges,
        } = &second.kind
        else {
            return None;
        };
        if *dim != 3
            || rotation.shape != endpoint.shape
            || first_source.id != normalized.id
            || second_source.id != normalized.id
            || first_ranges.len() != 4
            || second_ranges.len() != 4
        {
            return None;
        }
        for (axis, &extent) in shape.iter().enumerate() {
            let expected_first = if axis == 3 {
                (0, width / 2, 1)
            } else {
                (0, extent, 1)
            };
            let expected_second = if axis == 3 {
                (width / 2, width, 1)
            } else {
                (0, extent, 1)
            };
            if first_ranges[axis] != expected_first || second_ranges[axis] != expected_second {
                return None;
            }
        }
        nodes.extend([rotation, negative, first, second].map(|node| self.dense(node)));
        nodes.sort();
        nodes.dedup();
        for &node in &nodes {
            if self.reserved[node.index()] || self.index.order[node.index()].dtype != DType::BF16 {
                return None;
            }
            if node != output
                && (self.roots[node.index()]
                    || self
                        .index
                        .consumers_of(node)
                        .into_iter()
                        .flatten()
                        .any(|consumer| !nodes.contains(consumer)))
            {
                return None;
            }
        }
        let mut inputs = vec![self.dense(source), self.dense(cosine), self.dense(sine)];
        inputs.extend(weight.as_ref().map(|w| self.dense(w)));
        inputs.sort();
        inputs.dedup();
        Some(NormRopeRegion {
            nodes: nodes.into_boxed_slice(),
            inputs: inputs.into_boxed_slice(),
            output,
            normalized: self.dense(normalized),
            source: self.dense(source),
            cosine: self.dense(cosine),
            sine: self.dense(sine),
            weight: weight.as_ref().map(|w| self.dense(w)),
            shape: endpoint.shape.clone().into_boxed_slice(),
            eps: *eps,
        })
    }

    fn select_shared_rms_norm(&mut self) {
        self.work.semantic_nodes_scanned += self.index.order.len();
        // Retain reshapes as independent boundary values, including escaped
        // views. Canonicalization changes only which identical dense bytes
        // the region reads; it never changes ownership of a view.
        let mut groups: Vec<SharedRmsNormRegion> = Vec::new();
        for (i, node) in self.index.order.iter().enumerate() {
            if self.reserved[i] {
                continue;
            }
            let NodeKind::RmsNorm { x, weight, eps } = &node.kind else {
                continue;
            };
            let Some(&width) = node.shape.last() else {
                continue;
            };
            if width < 1024 || !matches!(node.dtype, DType::BF16 | DType::F32) {
                continue;
            }
            let Some(elements) = node.shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
            else {
                continue;
            };
            if elements == 0 {
                continue;
            }
            let mut canonical = x;
            while let NodeKind::Reshape { a, .. } = &canonical.kind {
                if a.shape.last() != Some(&width) {
                    break;
                }
                canonical = a;
            }
            let source = self.dense(canonical);
            let weight = weight.as_ref().map(|w| self.dense(w));
            let dense = DenseNodeId::from_index(i).unwrap();
            let existing = groups.iter_mut().find(|group| {
                // Later leaf weights are independent. A non-leaf weight must
                // precede every grouped output to rule out scheduling cycles.
                let independent_weight = weight.is_none_or(|weight| {
                    weight.index() < group.nodes[0].index()
                        || self.index.children[weight.index()].is_empty()
                });
                group.source == source
                    && group.width == width
                    && group.elements == elements
                    && group.dtype == node.dtype
                    && group.eps.to_bits() == eps.to_bits()
                    && group.nodes.len() < 3
                    && independent_weight
            });
            if let Some(group) = existing {
                group.nodes = group.nodes.iter().copied().chain([dense]).collect();
                group.weights = group.weights.iter().copied().chain([weight]).collect();
                let mut inputs = group.inputs.to_vec();
                inputs.push(self.dense(x));
                inputs.extend(weight);
                inputs.sort();
                inputs.dedup();
                group.inputs = inputs.into_boxed_slice();
            } else {
                let mut inputs = vec![source, self.dense(x)];
                inputs.extend(weight);
                inputs.sort();
                inputs.dedup();
                groups.push(SharedRmsNormRegion {
                    nodes: vec![dense].into_boxed_slice(),
                    inputs: inputs.into_boxed_slice(),
                    source,
                    weights: vec![weight].into_boxed_slice(),
                    width,
                    elements,
                    eps: *eps,
                    dtype: node.dtype,
                });
            }
        }
        for group in groups {
            if group.nodes.len() >= 2 {
                self.add_region(NativeRegion::SharedRmsNorm(group));
            }
        }
    }

    fn select_expert_route_rank(&mut self) {
        self.work.semantic_nodes_scanned += self.index.order.len();
        for i in (0..self.index.order.len()).rev() {
            let output = DenseNodeId::from_index(i).unwrap();
            if let Some(region) = self.match_expert_route_rank(output) {
                self.add_region(NativeRegion::ExpertRouteRank(region));
            }
        }
    }

    fn match_expert_route_rank(&self, output: DenseNodeId) -> Option<ExpertRouteRankRegion> {
        let endpoint = &self.index.order[output.index()];
        let NodeKind::Cast {
            a: counts,
            dtype: DType::U32,
        } = &endpoint.kind
        else {
            return None;
        };
        let NodeKind::Sum {
            a: floating,
            dims,
            keepdims: false,
        } = &counts.kind
        else {
            return None;
        };
        let NodeKind::Cast {
            a: precedes,
            dtype: DType::F32,
        } = &floating.kind
        else {
            return None;
        };
        let NodeKind::Add { a: smaller, b: tie } = &precedes.kind else {
            return None;
        };
        let NodeKind::Mul {
            a: same,
            b: earlier,
        } = &tie.kind
        else {
            return None;
        };
        let NodeKind::Lt { a: right, b: left } = &smaller.kind else {
            return None;
        };
        let NodeKind::Eq {
            a: eq_right,
            b: eq_left,
        } = &same.kind
        else {
            return None;
        };
        let NodeKind::Lt { a: column, b: row } = &earlier.kind else {
            return None;
        };
        let NodeKind::Reshape { a: experts, .. } = &right.kind else {
            return None;
        };
        let NodeKind::Reshape {
            a: left_experts, ..
        } = &left.kind
        else {
            return None;
        };
        let NodeKind::Reshape { a: positions, .. } = &column.kind else {
            return None;
        };
        let NodeKind::Reshape {
            a: row_positions, ..
        } = &row.kind
        else {
            return None;
        };
        let [rows, routes] = endpoint.shape.as_slice() else {
            return None;
        };
        let (rows, routes) = (*rows, *routes);
        if rows == 0
            || !(1..=32).contains(&routes)
            || rows.checked_mul(routes)? > u32::MAX as usize
            || dims.as_slice() != [2]
            || counts.shape != [rows, routes]
            || experts.shape != [rows, routes]
            || positions.shape != [routes]
            || right.shape != [rows, 1, routes]
            || left.shape != [rows, routes, 1]
            || column.shape != [1, routes]
            || row.shape != [routes, 1]
            || experts.id != left_experts.id
            || positions.id != row_positions.id
            || eq_right.id != right.id
            || eq_left.id != left.id
            || [floating, precedes, smaller, tie, same]
                .iter()
                .any(|n| n.shape != [rows, routes, routes])
            || earlier.shape != [routes, routes]
            || [experts, positions, right, left, column, row, endpoint]
                .iter()
                .any(|n| n.dtype != DType::U32)
            || [precedes, smaller, tie, same, earlier]
                .iter()
                .any(|n| n.dtype != DType::U8)
            || counts.dtype != DType::F32
            || floating.dtype != DType::F32
            || [
                endpoint, counts, floating, precedes, smaller, tie, same, earlier, right, left,
                column, row, experts, positions,
            ]
            .iter()
            .any(|n| n.device != endpoint.device)
        {
            return None;
        }
        // Boundary positions remain explicit: a host leaf shaped [K] does not
        // prove values0..K-1. Arbitrary positions retain the original tie order.
        let mut nodes = [
            endpoint, counts, floating, precedes, smaller, tie, same, earlier, right, left, column,
            row,
        ]
        .into_iter()
        .map(|n| self.dense(n))
        .collect::<Vec<_>>();
        normalize_nodes(&mut nodes);
        for &node in &nodes {
            if self.reserved[node.index()]
                || (node != output
                    && (self.roots[node.index()]
                        || self.index.consumers[node.index()]
                            .iter()
                            .any(|c| !nodes.contains(c))))
            {
                return None;
            }
        }
        Some(ExpertRouteRankRegion {
            nodes: nodes.into_boxed_slice(),
            inputs: vec![self.dense(experts), self.dense(positions)].into_boxed_slice(),
            output,
            rows,
            routes,
        })
    }

    fn select_ordered_scatter_reduce(&mut self) {
        self.work.semantic_nodes_scanned += self.index.order.len();
        for endpoint in (0..self.index.order.len()).rev() {
            let output = DenseNodeId::from_index(endpoint).unwrap();
            if self.reserved[endpoint] {
                continue;
            }
            if let Some(region) = self.match_ordered_scatter_reduce(output) {
                if let Some(weighted) = self.match_ordered_scatter_weighted_source(&region) {
                    if self
                        .add_region(NativeRegion::OrderedScatterReduce(weighted))
                        .is_some()
                    {
                        continue;
                    }
                }
                self.add_region(NativeRegion::OrderedScatterReduce(region));
            }
        }
    }

    fn match_ordered_scatter_reduce(
        &self,
        output: DenseNodeId,
    ) -> Option<OrderedScatterReduceRegion> {
        let endpoint = &self.index.order[output.index()];
        if endpoint.shape.len() != 2 {
            return None;
        }
        let rows = endpoint.shape[0];
        let width = endpoint.shape[1];
        let mut current = endpoint.clone();
        let mut nodes = Vec::new();
        let mut selections = Vec::new();
        let mut scatter = None;
        while let NodeKind::Add { a, b } = &current.kind {
            let NodeKind::Reshape { a: sliced, .. } = &b.kind else {
                return None;
            };
            let NodeKind::Slice {
                a: candidate,
                ranges,
            } = &sliced.kind
            else {
                return None;
            };
            if ranges.len() != 3
                || ranges[0] != (0, rows, 1)
                || ranges[2] != (0, width, 1)
                || ranges[1].2 != 1
                || ranges[1].1 != ranges[1].0.checked_add(1)?
                || current.shape != endpoint.shape
                || b.shape != endpoint.shape
                || current.dtype != endpoint.dtype
                || b.dtype != endpoint.dtype
            {
                return None;
            }
            if let Some(previous) = scatter {
                if previous != candidate.id {
                    return None;
                }
            } else {
                scatter = Some(candidate.id);
            }
            selections.push(ranges[1].0);
            nodes.extend([self.dense(&current), self.dense(b), self.dense(sliced)]);
            current = a.clone();
        }
        let NodeKind::Zeros { .. } = &current.kind else {
            return None;
        };
        if current.shape != endpoint.shape || current.dtype != endpoint.dtype {
            return None;
        }
        nodes.push(self.dense(&current));
        let scatter = self.index.dense_id(scatter?)?;
        let scatter_node = &self.index.order[scatter.index()];
        let NodeKind::ScatterAdd {
            a: empty,
            dim: 1,
            indexes,
            src,
        } = &scatter_node.kind
        else {
            return None;
        };
        if src.shape.len() != 3
            || src.shape[0] != rows
            || src.shape[2] != width
            || src.shape != empty.shape
            || src.shape != indexes.shape
            || src.dtype != endpoint.dtype
            || !matches!(&empty.kind, NodeKind::Zeros { .. })
        {
            return None;
        }
        let routes = src.shape[1];
        if routes == 0 || selections.len() != routes || selections.into_iter().rev().ne(0..routes) {
            return None;
        }
        let NodeKind::BroadcastTo { a: expanded, .. } = &indexes.kind else {
            return None;
        };
        if expanded.shape != [rows, routes, 1] {
            return None;
        }
        let mut compact = expanded.clone();
        nodes.extend([scatter, self.dense(empty), self.dense(indexes)]);
        while let NodeKind::Reshape { a, .. } = &compact.kind {
            nodes.push(self.dense(&compact));
            compact = a.clone();
        }
        if compact.shape != [rows, routes] || !matches!(compact.dtype, DType::U32 | DType::I64) {
            return None;
        }
        normalize_nodes(&mut nodes);
        // No covered intermediate may escape. In particular the scatter result
        // can have many slice users, but all must belong to this exact chain.
        for &node in &nodes {
            if self.reserved[node.index()]
                || (node != output
                    && (self.roots[node.index()]
                        || self.index.consumers[node.index()]
                            .iter()
                            .any(|consumer| !nodes.contains(consumer))))
            {
                return None;
            }
        }
        Some(OrderedScatterReduceRegion {
            nodes: nodes.into_boxed_slice(),
            inputs: vec![self.dense(src), self.dense(&compact)].into_boxed_slice(),
            weighted_source: false,
            output,
            shape: endpoint.shape.clone().into_boxed_slice(),
            routes,
            dtype: endpoint.dtype,
            device: endpoint.device.clone(),
        })
    }

    fn match_ordered_scatter_weighted_source(
        &self,
        base: &OrderedScatterReduceRegion,
    ) -> Option<OrderedScatterReduceRegion> {
        if base.dtype != DType::BF16 {
            return None;
        }
        let source = &self.index.order[base.inputs[0].index()];
        let NodeKind::Cast {
            a: product,
            dtype: DType::BF16,
        } = &source.kind
        else {
            return None;
        };
        let NodeKind::Mul {
            a: floating,
            b: weights,
        } = &product.kind
        else {
            return None;
        };
        let NodeKind::Cast {
            a: token_major,
            dtype: DType::F32,
        } = &floating.kind
        else {
            return None;
        };
        let NodeKind::Permute {
            a: route_major,
            dims,
        } = &token_major.kind
        else {
            return None;
        };
        let NodeKind::Reshape { a: projected, .. } = &route_major.kind else {
            return None;
        };
        let rows = base.shape[0];
        let width = base.shape[1];
        if dims.as_slice() != [1, 0, 2]
            || route_major.shape != [base.routes, rows, width]
            || projected.shape != [base.routes.checked_mul(rows)?, width]
            || projected.dtype != DType::BF16
            || weights.shape != [rows, base.routes, 1]
            || weights.dtype != DType::F32
            || product.dtype != DType::F32
        {
            return None;
        }
        let mut region = base.clone();
        let mut nodes = region.nodes.to_vec();
        nodes.extend([source, product, floating, token_major, route_major].map(|n| self.dense(n)));
        normalize_nodes(&mut nodes);
        // Absorbing a weighted materialization must not hide an externally
        // observable intermediate or a source shared with another branch.
        for &node in &nodes {
            if self.reserved[node.index()]
                || (node != region.output
                    && (self.roots[node.index()]
                        || self.index.consumers[node.index()]
                            .iter()
                            .any(|consumer| !nodes.contains(consumer))))
            {
                return None;
            }
        }
        region.nodes = nodes.into_boxed_slice();
        region.inputs =
            vec![self.dense(projected), base.inputs[1], self.dense(weights)].into_boxed_slice();
        region.weighted_source = true;
        Some(region)
    }

    fn select_gemm_epilogues(&mut self) {
        self.work.semantic_nodes_scanned += self.index.order.len();
        for dense_index in 0..self.index.order.len() {
            let dense = DenseNodeId::from_index(dense_index)
                .expect("GraphIndex validated the semantic node count");
            if self.reserved[dense_index] {
                continue;
            }
            let node = &self.index.order[dense_index];
            match &node.kind {
                NodeKind::Add { .. } => {
                    let children = &self.index.children[dense_index];
                    let candidates = [(children[0], children[1]), (children[1], children[0])];
                    let mut selected = None;
                    for (linear, residual) in candidates {
                        if self.reserved[linear.index()]
                            || self.index.consumers[linear.index()].len() != 1
                        {
                            continue;
                        }
                        let Some(linear_inputs) = self.absorbable_linear(linear) else {
                            continue;
                        };
                        let linear_node = &self.index.order[linear.index()];
                        let residual_node = &self.index.order[residual.index()];
                        if residual_node.shape != linear_node.shape
                            || residual_node.dtype != linear_node.dtype
                            || !residual_node.device.is_metal()
                        {
                            continue;
                        }
                        selected = Some((linear, residual, linear_inputs));
                        break;
                    }
                    if let Some((linear, residual, linear_inputs)) = selected {
                        let mut nodes = vec![dense];
                        if !self.roots[linear.index()] {
                            nodes.push(linear);
                        }
                        nodes.sort_unstable();
                        let mut inputs = linear_inputs.to_vec();
                        inputs.push(residual);
                        self.add_region(NativeRegion::LinearResidual(LinearResidualRegion {
                            nodes: nodes.into_boxed_slice(),
                            inputs: inputs.into_boxed_slice(),
                            output: dense,
                            shape: node.shape.clone().into_boxed_slice(),
                            dtype: node.dtype,
                            device: node.device.clone(),
                        }));
                    }
                }
                NodeKind::Gelu { approximate, .. } => {
                    let linear = self.index.children[dense_index][0];
                    if self.reserved[linear.index()] {
                        continue;
                    }
                    let Some(inputs) = self.absorbable_linear(linear) else {
                        continue;
                    };
                    let dual = self.index.consumers[linear.index()].len() != 1
                        || self.roots[linear.index()];
                    self.add_region(NativeRegion::LinearGelu(LinearGeluRegion {
                        nodes: vec![linear, dense].into_boxed_slice(),
                        inputs: inputs.to_vec().into_boxed_slice(),
                        pre_activation: linear,
                        output: dense,
                        approximate: *approximate,
                        dual,
                        shape: node.shape.clone().into_boxed_slice(),
                        dtype: node.dtype,
                        device: node.device.clone(),
                    }));
                }
                _ => {}
            }
        }
    }

    fn absorbable_linear(&self, dense: DenseNodeId) -> Option<[DenseNodeId; 3]> {
        let node = &self.index.order[dense.index()];
        if !node.device.is_metal() {
            return None;
        }
        match &node.kind {
            NodeKind::Linear { .. } => self.index.children[dense.index()].as_ref().try_into().ok(),
            _ => None,
        }
    }

    fn select_optimizers(&mut self) {
        if !self.options.environment.fusion && !self.options.environment.optimizer_groups {
            return;
        }
        self.work.semantic_nodes_scanned += self.index.order.len();
        let mut adam_steps = Vec::new();
        for (dense_index, node) in self.index.order.iter().enumerate() {
            let dense = DenseNodeId::from_index(dense_index)
                .expect("GraphIndex validated the semantic node count");
            if self.reserved[dense_index] {
                continue;
            }
            match &node.kind {
                NodeKind::AdamWStep { .. } => adam_steps.push(dense),
                NodeKind::SgdStep { .. } if self.options.environment.fusion => {
                    self.select_sgd(dense)
                }
                _ => {}
            }
        }

        let mut grouped = vec![false; self.index.order.len()];
        if self.options.environment.optimizer_groups {
            #[derive(Clone, PartialEq, Eq, Hash)]
            struct Key(Vec<usize>, DType, u64, u64, u64, u64, [DenseNodeId; 3]);
            let mut buckets: HashMap<Key, Vec<DenseNodeId>> = HashMap::new();
            let mut order = Vec::new();
            for &step in &adam_steps {
                let node = &self.index.order[step.index()];
                let NodeKind::AdamWStep {
                    param,
                    beta1,
                    beta2,
                    eps,
                    weight_decay,
                    ..
                } = &node.kind
                else {
                    unreachable!()
                };
                let scalar_inputs = self.index.children[step.index()][4..7]
                    .try_into()
                    .expect("AdamW graph nodes have three runtime scalar inputs");
                let key = Key(
                    param.shape.clone(),
                    param.dtype,
                    beta1.to_bits(),
                    beta2.to_bits(),
                    eps.to_bits(),
                    weight_decay.to_bits(),
                    scalar_inputs,
                );
                buckets
                    .entry(key.clone())
                    .or_insert_with(|| {
                        order.push(key);
                        Vec::new()
                    })
                    .push(step);
            }
            for key in order {
                for chunk in buckets[&key].chunks(4) {
                    if chunk.len() >= 2 {
                        self.select_adamw_group(chunk);
                        for step in chunk {
                            grouped[step.index()] = self.reserved[step.index()];
                        }
                    }
                }
            }
        }
        for step in adam_steps {
            if !grouped[step.index()] && self.options.environment.fusion {
                self.select_adamw(step);
            }
        }
    }

    fn adamw_options(&self, step: DenseNodeId) -> AdamWOptions {
        let NodeKind::AdamWStep {
            beta1,
            beta2,
            eps,
            weight_decay,
            ..
        } = &self.index.order[step.index()].kind
        else {
            unreachable!()
        };
        AdamWOptions {
            beta1: *beta1,
            beta2: *beta2,
            eps: *eps,
            weight_decay: *weight_decay,
        }
    }

    fn optimizer_routes(
        &self,
        producer: DenseNodeId,
        parameter: u32,
    ) -> (Vec<DenseNodeId>, Vec<OptimizerOutput>) {
        let mut nodes = vec![producer];
        let kinds = [
            OptimizerOutputKind::Parameter,
            OptimizerOutputKind::FirstMoment,
            OptimizerOutputKind::SecondMoment,
        ];
        let mut routes = [vec![producer], Vec::new(), Vec::new()];
        for &consumer in self.index.consumers[producer.index()].iter() {
            let output = match &self.index.order[consumer.index()].kind {
                NodeKind::AdamWOut { index, .. } => Some(*index as usize),
                _ => None,
            };
            if let Some(output) = output {
                if output < routes.len() {
                    routes[output].push(consumer);
                    nodes.push(consumer);
                }
            }
        }
        nodes.sort_unstable();
        nodes.dedup();
        let outputs = routes
            .into_iter()
            .enumerate()
            .map(|(index, mut semantic_nodes)| {
                semantic_nodes.sort_unstable();
                semantic_nodes.dedup();
                OptimizerOutput {
                    index: parameter * 3 + index as u32,
                    parameter,
                    kind: kinds[index],
                    semantic_nodes: semantic_nodes.into_boxed_slice(),
                }
            })
            .collect();
        (nodes, outputs)
    }

    fn select_adamw(&mut self, step: DenseNodeId) {
        let node = &self.index.order[step.index()];
        let children = &self.index.children[step.index()];
        let tensor_inputs: [DenseNodeId; 4] = children[..4].try_into().unwrap();
        let scalar_inputs: [DenseNodeId; 3] = children[4..7].try_into().unwrap();
        let options = self.adamw_options(step);
        let (nodes, outputs) = self.optimizer_routes(step, 0);
        self.add_region(NativeRegion::AdamW(AdamWRegion {
            nodes: nodes.into_boxed_slice(),
            inputs: children.to_vec().into_boxed_slice(),
            tensor_inputs,
            scalar_inputs,
            outputs: outputs.into_boxed_slice(),
            expressions: adamw_exprs(
                options.beta1,
                options.beta2,
                options.eps,
                options.weight_decay,
            )
            .into_iter()
            .collect::<Vec<_>>()
            .into_boxed_slice(),
            options,
            shape: node.shape.clone().into_boxed_slice(),
            dtype: node.dtype,
            device: node.device.clone(),
        }));
    }

    fn select_adamw_group(&mut self, steps: &[DenseNodeId]) {
        let first = steps[0];
        let first_node = &self.index.order[first.index()];
        let options = self.adamw_options(first);
        let first_children = &self.index.children[first.index()];
        let scalar_inputs: [DenseNodeId; 3] = first_children[4..7].try_into().unwrap();
        let mut inputs = Vec::with_capacity(steps.len() * 4 + 3);
        let mut parameter_inputs = Vec::with_capacity(steps.len());
        let mut nodes = Vec::new();
        let mut outputs = Vec::with_capacity(steps.len() * 3);
        let base = adamw_exprs(
            options.beta1,
            options.beta2,
            options.eps,
            options.weight_decay,
        );
        let mut expressions = Vec::with_capacity(steps.len() * 3);
        for (parameter, &step) in steps.iter().enumerate() {
            let lanes: [DenseNodeId; 4] =
                self.index.children[step.index()][..4].try_into().unwrap();
            parameter_inputs.push(lanes);
            inputs.extend(lanes);
            let (route_nodes, route_outputs) = self.optimizer_routes(step, parameter as u32);
            nodes.extend(route_nodes);
            outputs.extend(route_outputs);
            let remap = (0..4)
                .map(|lane| (lane, parameter as u32 * 4 + lane))
                .collect::<HashMap<_, _>>();
            expressions.extend(base.iter().map(|expr| expr.remap_lanes(&remap)));
        }
        inputs.extend(scalar_inputs);
        nodes.sort_unstable();
        nodes.dedup();
        self.add_region(NativeRegion::AdamWGroup(AdamWGroupRegion {
            nodes: nodes.into_boxed_slice(),
            inputs: inputs.into_boxed_slice(),
            parameter_inputs: parameter_inputs.into_boxed_slice(),
            scalar_inputs,
            outputs: outputs.into_boxed_slice(),
            expressions: expressions.into_boxed_slice(),
            options,
            shape: first_node.shape.clone().into_boxed_slice(),
            dtype: first_node.dtype,
            device: first_node.device.clone(),
        }));
    }

    fn select_sgd(&mut self, step: DenseNodeId) {
        let node = &self.index.order[step.index()];
        let NodeKind::SgdStep {
            momentum,
            dampening,
            nesterov,
            weight_decay,
            ..
        } = &node.kind
        else {
            unreachable!()
        };
        let children = &self.index.children[step.index()];
        let tensor_inputs: [DenseNodeId; 3] = children[..3].try_into().unwrap();
        let scalar_inputs = [children[4], children[3]];
        let mut routes = [vec![step], Vec::new()];
        let mut nodes = vec![step];
        for &consumer in self.index.consumers[step.index()].iter() {
            if let NodeKind::SgdOut { index, .. } = &self.index.order[consumer.index()].kind {
                routes[*index as usize].push(consumer);
                nodes.push(consumer);
            }
        }
        nodes.sort_unstable();
        nodes.dedup();
        let outputs = [
            OptimizerOutput {
                index: 0,
                parameter: 0,
                kind: OptimizerOutputKind::Parameter,
                semantic_nodes: std::mem::take(&mut routes[0]).into_boxed_slice(),
            },
            OptimizerOutput {
                index: 1,
                parameter: 0,
                kind: OptimizerOutputKind::Velocity,
                semantic_nodes: std::mem::take(&mut routes[1]).into_boxed_slice(),
            },
        ];
        let options = SgdOptions {
            momentum: *momentum,
            dampening: *dampening,
            nesterov: *nesterov,
            weight_decay: *weight_decay,
        };
        self.add_region(NativeRegion::Sgd(SgdRegion {
            nodes: nodes.into_boxed_slice(),
            inputs: vec![
                tensor_inputs[0],
                tensor_inputs[1],
                tensor_inputs[2],
                scalar_inputs[0],
                scalar_inputs[1],
            ]
            .into_boxed_slice(),
            tensor_inputs,
            scalar_inputs,
            outputs: outputs.into_iter().collect::<Vec<_>>().into_boxed_slice(),
            expressions: sgd_exprs(
                options.momentum,
                options.dampening,
                options.nesterov,
                options.weight_decay,
            )
            .into_iter()
            .collect::<Vec<_>>()
            .into_boxed_slice(),
            options,
            shape: node.shape.clone().into_boxed_slice(),
            dtype: node.dtype,
            device: node.device.clone(),
        }));
    }

    /// Makes one forward pass over the dense postorder and grows open
    /// elementwise chains. A non-fusible consumer, multiple consumers, or a
    /// graph root closes a chain. Profitable closed chains become regions. A
    /// reduction either absorbs its open input chain into a fused-reduce region
    /// or closes the chain first.
    fn select_elementwise(&mut self) -> Result<(), String> {
        self.work.semantic_nodes_scanned += self.index.order.len();
        let mut open: Vec<Option<OpenRegion>> = (0..self.index.order.len()).map(|_| None).collect();
        for dense_index in 0..self.index.order.len() {
            let dense = DenseNodeId::from_index(dense_index)
                .expect("GraphIndex validated the semantic node count");
            let operation = self.element_operation(dense);
            let children = self.index.children[dense_index].to_vec();
            for child in &children {
                if open[child.index()].is_some()
                    && (operation.is_none()
                        || self.index.consumers[child.index()].len() != 1
                        || self.roots[child.index()])
                {
                    let region = open[child.index()].take().unwrap();
                    self.emit_elementwise(*child, region)?;
                }
            }
            match operation {
                None => {}
                Some(ElementOperation::Unary(operation)) => {
                    let child = children[0];
                    let (mut region, expression) = match open[child.index()].take() {
                        Some(mut region) => {
                            let expression = operation.apply(std::mem::replace(
                                &mut region.expression,
                                KernelExpr::cst(0.0),
                            ));
                            (region, expression)
                        }
                        None => {
                            let mut region = OpenRegion::empty();
                            let lane = self.element_operand(
                                &mut region,
                                child,
                                &self.index.order[dense_index].shape,
                            );
                            (region, operation.apply(lane))
                        }
                    };
                    region.expression =
                        expression.semantic(dense, self.index.order[dense_index].dtype);
                    region.ops += 1;
                    region.nodes.push(dense);
                    open[dense_index] = Some(region);
                }
                Some(ElementOperation::Binary(operation)) => {
                    let a = children[0];
                    let b = children[1];
                    let mut left = open[a.index()].take();
                    let mut right = open[b.index()].take();
                    if let (Some(left_region), Some(right_region)) = (&left, &right) {
                        if left_region.inputs.len() + right_region.inputs.len() > MAX_LANES {
                            self.emit_elementwise(b, right.take().unwrap())?;
                        }
                    }
                    if let Some(region) = &left {
                        if right.is_none()
                            && self
                                .const_value(b, &self.index.order[dense_index].shape)
                                .is_none()
                            && !region.lane_of.contains_key(&b)
                            && region.inputs.len() >= MAX_LANES
                        {
                            self.emit_elementwise(a, left.take().unwrap())?;
                        }
                    }
                    if let Some(region) = &right {
                        if left.is_none()
                            && self
                                .const_value(a, &self.index.order[dense_index].shape)
                                .is_none()
                            && !region.lane_of.contains_key(&a)
                            && region.inputs.len() >= MAX_LANES
                        {
                            self.emit_elementwise(b, right.take().unwrap())?;
                        }
                    }
                    let (mut region, expression) = match (left, right) {
                        (Some(mut left), Some(right)) => {
                            let right_expression = left.absorb(right);
                            let left_expression =
                                std::mem::replace(&mut left.expression, KernelExpr::cst(0.0));
                            (
                                left,
                                operation.apply(
                                    self.semantic_operand(left_expression, dense, 0),
                                    self.semantic_operand(right_expression, dense, 1),
                                ),
                            )
                        }
                        (Some(mut region), None) => {
                            let right = self.element_operand(
                                &mut region,
                                b,
                                &self.index.order[dense_index].shape,
                            );
                            let left =
                                std::mem::replace(&mut region.expression, KernelExpr::cst(0.0));
                            (
                                region,
                                operation.apply(
                                    self.semantic_operand(left, dense, 0),
                                    self.semantic_operand(right, dense, 1),
                                ),
                            )
                        }
                        (None, Some(mut region)) => {
                            let left = self.element_operand(
                                &mut region,
                                a,
                                &self.index.order[dense_index].shape,
                            );
                            let right =
                                std::mem::replace(&mut region.expression, KernelExpr::cst(0.0));
                            (
                                region,
                                operation.apply(
                                    self.semantic_operand(left, dense, 0),
                                    self.semantic_operand(right, dense, 1),
                                ),
                            )
                        }
                        (None, None) => {
                            let mut region = OpenRegion::empty();
                            let left = self.element_operand(
                                &mut region,
                                a,
                                &self.index.order[dense_index].shape,
                            );
                            let right = self.element_operand(
                                &mut region,
                                b,
                                &self.index.order[dense_index].shape,
                            );
                            (
                                region,
                                operation.apply(
                                    self.semantic_operand(left, dense, 0),
                                    self.semantic_operand(right, dense, 1),
                                ),
                            )
                        }
                    };
                    region.expression =
                        expression.semantic(dense, self.index.order[dense_index].dtype);
                    region.ops += 1;
                    region.nodes.push(dense);
                    open[dense_index] = Some(region);
                }
                Some(ElementOperation::Select(comparison)) => {
                    let cond = children[0];
                    let cond_children = &self.index.children[cond.index()];
                    let logical = [cond_children[0], cond_children[1], children[1], children[2]];
                    let mut region = OpenRegion::empty();
                    let mut expressions = Vec::with_capacity(4);
                    let mut abandon = false;
                    for child in logical {
                        if let Some(child_region) = open[child.index()].take() {
                            if region.inputs.len() + child_region.inputs.len() > MAX_LANES {
                                self.emit_elementwise(child, child_region)?;
                                if region.inputs.len() >= MAX_LANES
                                    && !region.lane_of.contains_key(&child)
                                {
                                    abandon = true;
                                    break;
                                }
                                expressions.push(region.lane(child));
                            } else {
                                expressions.push(region.absorb(child_region));
                            }
                        } else if let Some(value) =
                            self.const_value(child, &self.index.order[dense_index].shape)
                        {
                            expressions.push(KernelExpr::typed_constant(
                                value,
                                self.index.order[child.index()].dtype,
                            ));
                        } else if region.inputs.len() >= MAX_LANES
                            && !region.lane_of.contains_key(&child)
                        {
                            abandon = true;
                            break;
                        } else {
                            expressions.push(region.lane(child));
                        }
                    }
                    if !abandon {
                        let mut expressions = expressions.into_iter();
                        let condition = comparison
                            .apply(
                                self.semantic_operand(expressions.next().unwrap(), cond, 0),
                                self.semantic_operand(expressions.next().unwrap(), cond, 1),
                            )
                            .semantic(cond, self.index.order[cond.index()].dtype);
                        region.expression = KernelExpr::Select(
                            Box::new(condition),
                            Box::new(expressions.next().unwrap()),
                            Box::new(expressions.next().unwrap()),
                        )
                        .semantic(dense, self.index.order[dense_index].dtype);
                        region.ops += 1;
                        if self.index.consumers[cond.index()].len() == 1
                            && !self.roots[cond.index()]
                        {
                            region.nodes.push(cond);
                        }
                        region.nodes.push(dense);
                        open[dense_index] = Some(region);
                    }
                }
                Some(ElementOperation::ArgReduce(maximum, dim)) => {
                    let input = children[0];
                    let input_shape = &self.index.order[input.index()].shape;
                    if let Some(region) = open[input.index()].take() {
                        let strides = region
                            .inputs
                            .iter()
                            .map(|input| {
                                lane_strides(&self.index.order[input.index()].shape, input_shape)
                                    .map(Vec::into_boxed_slice)
                            })
                            .collect::<Option<Vec<_>>>();
                        if dim < input_shape.len()
                            && input_shape[dim] > 0
                            && !region.inputs.is_empty()
                            && strides.is_some()
                        {
                            let mut nodes = region.nodes.clone();
                            nodes.push(dense);
                            normalize_nodes(&mut nodes);
                            if self
                                .add_region(NativeRegion::ElementwiseArgReduce(
                                    ElementwiseArgReduceRegion {
                                        nodes: nodes.into_boxed_slice(),
                                        inputs: region.inputs.clone().into_boxed_slice(),
                                        lane_strides: strides.unwrap().into_boxed_slice(),
                                        output: dense,
                                        expression: region.expression.clone(),
                                        maximum,
                                        dim,
                                        input_shape: input_shape.clone().into_boxed_slice(),
                                        shape: self.index.order[dense_index]
                                            .shape
                                            .clone()
                                            .into_boxed_slice(),
                                        dtype: self.index.order[dense_index].dtype,
                                        device: self.index.order[dense_index].device.clone(),
                                    },
                                ))
                                .is_none()
                            {
                                self.emit_elementwise(input, region)?;
                            }
                        } else {
                            self.emit_elementwise(input, region)?;
                        }
                    }
                }
                Some(ElementOperation::Reduce(op, mut dims, keepdims)) => {
                    let input = children[0];
                    let input_shape = self.index.order[input.index()].shape.clone();
                    dims.sort_unstable();
                    dims.dedup();
                    let rank = input_shape.len();
                    let output_shape = reduced_shape(&input_shape, &dims, keepdims);
                    let guards_ok = !dims.is_empty()
                        && dims.iter().all(|&dim| dim < rank)
                        && dims.iter().map(|&dim| input_shape[dim]).product::<usize>() > 0;
                    if let Some(mut region) = open[input.index()].take() {
                        if guards_ok && !region.inputs.is_empty() {
                            let strides = region
                                .inputs
                                .iter()
                                .map(|input| {
                                    lane_strides(
                                        &self.index.order[input.index()].shape,
                                        &input_shape,
                                    )
                                    .map(Vec::into_boxed_slice)
                                })
                                .collect::<Option<Vec<_>>>();
                            if let Some(strides) = strides {
                                region.nodes.push(dense);
                                normalize_nodes(&mut region.nodes);
                                self.add_region(NativeRegion::ElementwiseReduce(
                                    ElementwiseReduceRegion {
                                        nodes: region.nodes.into_boxed_slice(),
                                        inputs: region.inputs.into_boxed_slice(),
                                        lane_strides: strides.into_boxed_slice(),
                                        output: dense,
                                        expression: region.expression,
                                        op,
                                        dims: dims.into_boxed_slice(),
                                        keepdims,
                                        input_shape: input_shape.into_boxed_slice(),
                                        shape: output_shape.into_boxed_slice(),
                                        dtype: self.index.order[dense_index].dtype,
                                        device: self.index.order[dense_index].device.clone(),
                                    },
                                ));
                            } else {
                                self.emit_elementwise(input, region)?;
                            }
                        } else {
                            self.emit_elementwise(input, region)?;
                        }
                    }
                }
            }
        }
        for dense_index in 0..open.len() {
            if let Some(region) = open[dense_index].take() {
                self.emit_elementwise(
                    DenseNodeId::from_index(dense_index)
                        .expect("GraphIndex validated the semantic node count"),
                    region,
                )?;
            }
        }
        Ok(())
    }

    fn semantic_operand(
        &self,
        expression: KernelExpr,
        node: DenseNodeId,
        operand: usize,
    ) -> KernelExpr {
        match crate::scalar_coercion(&self.index.order[node.index()].kind, operand) {
            Some(conversion) => KernelExpr::Cast(Box::new(expression), conversion.destination),
            None => expression,
        }
    }

    fn element_operand(
        &self,
        region: &mut OpenRegion,
        child: DenseNodeId,
        output_shape: &[usize],
    ) -> KernelExpr {
        self.const_value(child, output_shape)
            .map(|value| KernelExpr::typed_constant(value, self.index.order[child.index()].dtype))
            .unwrap_or_else(|| region.lane(child))
    }

    fn const_value(&self, child: DenseNodeId, output_shape: &[usize]) -> Option<f64> {
        match &self.index.order[child.index()].kind {
            NodeKind::Full { shape, value, .. } if broadcast_compatible(shape, output_shape) => {
                Some(*value)
            }
            NodeKind::Zeros { shape, .. } if broadcast_compatible(shape, output_shape) => Some(0.0),
            _ => None,
        }
    }

    fn element_operation(&self, dense: DenseNodeId) -> Option<ElementOperation> {
        if self.reserved[dense.index()] {
            return None;
        }
        let node = &self.index.order[dense.index()];
        let input_ok =
            |child: &std::sync::Arc<Node>| broadcast_compatible(&child.shape, &node.shape);
        match &node.kind {
            NodeKind::Add { a, b } if input_ok(a) && input_ok(b) => {
                Some(ElementOperation::Binary(BinaryOperation::Add))
            }
            NodeKind::Sub { a, b } if input_ok(a) && input_ok(b) => {
                Some(ElementOperation::Binary(BinaryOperation::Sub))
            }
            NodeKind::Mul { a, b } if input_ok(a) && input_ok(b) => {
                Some(ElementOperation::Binary(BinaryOperation::Mul))
            }
            NodeKind::Div { a, b } if input_ok(a) && input_ok(b) => {
                Some(ElementOperation::Binary(BinaryOperation::Div))
            }
            NodeKind::Maximum { a, b } if input_ok(a) && input_ok(b) => {
                Some(ElementOperation::Binary(BinaryOperation::Maximum))
            }
            NodeKind::Minimum { a, b } if input_ok(a) && input_ok(b) => {
                Some(ElementOperation::Binary(BinaryOperation::Minimum))
            }
            NodeKind::Neg { .. } => Some(ElementOperation::Unary(UnaryOperation::Neg)),
            NodeKind::Sqrt { .. } => Some(ElementOperation::Unary(UnaryOperation::Sqrt)),
            NodeKind::Exp { .. } => Some(ElementOperation::Unary(UnaryOperation::Exp)),
            NodeKind::Log { .. } => Some(ElementOperation::Unary(UnaryOperation::Log)),
            NodeKind::Sin { .. } => Some(ElementOperation::Unary(UnaryOperation::Sin)),
            NodeKind::Cos { .. } => Some(ElementOperation::Unary(UnaryOperation::Cos)),
            NodeKind::Relu { .. } => Some(ElementOperation::Unary(UnaryOperation::Relu)),
            NodeKind::Tanh { .. } => Some(ElementOperation::Unary(UnaryOperation::Tanh)),
            NodeKind::Gelu { approximate, .. } => {
                Some(ElementOperation::Unary(UnaryOperation::Gelu(*approximate)))
            }
            NodeKind::Abs { .. } => Some(ElementOperation::Unary(UnaryOperation::Abs)),
            NodeKind::Erf { .. } => Some(ElementOperation::Unary(UnaryOperation::Erf)),
            NodeKind::Floor { .. } => Some(ElementOperation::Unary(UnaryOperation::Floor)),
            NodeKind::Ceil { .. } => Some(ElementOperation::Unary(UnaryOperation::Ceil)),
            NodeKind::Round { .. } => Some(ElementOperation::Unary(UnaryOperation::Round)),
            NodeKind::Pow { exp, .. } => Some(ElementOperation::Unary(UnaryOperation::Pow(*exp))),
            NodeKind::Sign { .. } => Some(ElementOperation::Unary(UnaryOperation::Sign)),
            NodeKind::Cast { a, dtype } if a.dtype.is_float() && dtype.is_float() => {
                Some(ElementOperation::Unary(UnaryOperation::Cast(*dtype)))
            }
            NodeKind::Where { cond, a, b }
                if self.index.consumers[self.dense(cond).index()].len() == 1
                    && input_ok(a)
                    && input_ok(b) =>
            {
                let comparison = match &cond.kind {
                    NodeKind::Eq { a, b } if input_ok(a) && input_ok(b) => {
                        Some(ComparisonOperation::Eq)
                    }
                    NodeKind::Gt { a, b } if input_ok(a) && input_ok(b) => {
                        Some(ComparisonOperation::Gt)
                    }
                    NodeKind::Lt { a, b } if input_ok(a) && input_ok(b) => {
                        Some(ComparisonOperation::Lt)
                    }
                    NodeKind::Ge { a, b } if input_ok(a) && input_ok(b) => {
                        Some(ComparisonOperation::Ge)
                    }
                    NodeKind::Le { a, b } if input_ok(a) && input_ok(b) => {
                        Some(ComparisonOperation::Le)
                    }
                    _ => None,
                };
                comparison.map(ElementOperation::Select)
            }
            NodeKind::Argmax { dim, .. } => Some(ElementOperation::ArgReduce(true, *dim)),
            NodeKind::Argmin { dim, .. } => Some(ElementOperation::ArgReduce(false, *dim)),
            NodeKind::Sum { dims, keepdims, .. } if !dims.is_empty() => Some(
                ElementOperation::Reduce(ReduceOp::Sum, dims.clone(), *keepdims),
            ),
            NodeKind::Mean { dims, keepdims, .. } if !dims.is_empty() => Some(
                ElementOperation::Reduce(ReduceOp::Mean, dims.clone(), *keepdims),
            ),
            NodeKind::Max { dims, keepdims, .. } if !dims.is_empty() => Some(
                ElementOperation::Reduce(ReduceOp::Max, dims.clone(), *keepdims),
            ),
            NodeKind::Min { dims, keepdims, .. } if !dims.is_empty() => Some(
                ElementOperation::Reduce(ReduceOp::Min, dims.clone(), *keepdims),
            ),
            _ => None,
        }
    }

    /// Emits a closed chain as an [`ElementwiseRegion`] when it has at least
    /// two fused operations and one real input lane. Its strides must be
    /// broadcast-compatible, and its element count must fit the backend index
    /// range. Other chains lower as independent nodes.
    fn emit_elementwise(
        &mut self,
        endpoint: DenseNodeId,
        mut region: OpenRegion,
    ) -> Result<(), String> {
        let node = &self.index.order[endpoint.index()];
        let strides = region
            .inputs
            .iter()
            .map(|input| {
                lane_strides(&self.index.order[input.index()].shape, &node.shape)
                    .map(Vec::into_boxed_slice)
            })
            .collect::<Option<Vec<_>>>();
        if region.ops >= 2 && !region.inputs.is_empty() {
            if let Some(strides) = strides {
                normalize_nodes(&mut region.nodes);
                self.add_region(NativeRegion::Elementwise(ElementwiseRegion {
                    nodes: region.nodes.into_boxed_slice(),
                    inputs: region.inputs.into_boxed_slice(),
                    lane_strides: strides.into_boxed_slice(),
                    output: ElementwiseOutput {
                        semantic_node: endpoint,
                        expression: region.expression,
                    },
                    shape: node.shape.clone().into_boxed_slice(),
                    dtype: node.dtype,
                    device: node.device.clone(),
                }));
            }
        }
        Ok(())
    }

    /// Merges elementwise regions that share a prefix into multi-output
    /// regions. A prefix qualifies when its output fans out to at least two
    /// same-shaped elementwise continuations. `RegionDependencyIndex` rejects
    /// a merge if a continuation's extra inputs depend on other prefix
    /// descendants. A merge also fails if it exceeds buffer or operation
    /// limits, or if an externally materialized prefix has a different shape.
    /// A split merge keeps the prefix as a separate draft for external users.
    fn select_multi_output(&mut self) {
        if self.options.environment.fusion_debug {
            let elementwise = self
                .drafts
                .iter()
                .filter(|draft| {
                    draft.active && matches!(draft.region, NativeRegion::Elementwise(_))
                })
                .count();
            eprintln!(
                "[fusion] analyze: {} nodes ({elementwise} elementwise regions)",
                self.index.order.len()
            );
        }
        let mut owner = vec![None; self.index.order.len()];
        let mut worklist = VecDeque::new();
        for (draft, region) in self.drafts.iter().enumerate() {
            if !region.active {
                continue;
            }
            for node in region.region.nodes() {
                owner[node.index()] = Some(draft);
            }
            if matches!(region.region, NativeRegion::Elementwise(_)) {
                worklist.push_back(draft);
            }
        }
        let mut dependencies = RegionDependencyIndex::new(self.index, &self.drafts, &owner);
        while let Some(prefix) = worklist.pop_back() {
            self.work.multi_output_work_items += 1;
            if !self.drafts[prefix].active
                || !matches!(self.drafts[prefix].region, NativeRegion::Elementwise(_))
            {
                continue;
            }
            let prefix_output = match &self.drafts[prefix].region {
                NativeRegion::Elementwise(region) => region.output.semantic_node,
                _ => unreachable!(),
            };
            if self.index.consumers[prefix_output.index()].len()
                + usize::from(self.roots[prefix_output.index()])
                < 2
            {
                continue;
            }
            let mut external = self.roots[prefix_output.index()];
            let mut consumers = Vec::new();
            for &consumer in self.index.consumers[prefix_output.index()].iter() {
                let Some(region) = owner[consumer.index()] else {
                    external = true;
                    continue;
                };
                if region == prefix || !self.drafts[region].active {
                    external = true;
                    continue;
                }
                let reads_prefix = self.drafts[region].region.inputs().contains(&prefix_output);
                if matches!(self.drafts[region].region, NativeRegion::Elementwise(_))
                    && reads_prefix
                {
                    if !consumers.contains(&region) {
                        consumers.push(region);
                    }
                } else {
                    external = true;
                }
            }
            if consumers.is_empty() {
                continue;
            }
            dependencies.mark_descendants(prefix);
            let mut groups: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
            for consumer in consumers {
                let shape = match &self.drafts[consumer].region {
                    NativeRegion::Elementwise(region) => region.shape.to_vec(),
                    _ => unreachable!(),
                };
                match groups.iter_mut().find(|(candidate, _)| candidate == &shape) {
                    Some((_, group)) => group.push(consumer),
                    None => groups.push((shape, vec![consumer])),
                }
            }
            groups.sort_by_key(|(_, group)| Reverse(group.len()));
            for (shape, group) in groups {
                let group_set = group.iter().copied().collect::<HashSet<_>>();
                let extra_inputs = group
                    .iter()
                    .flat_map(|&continuation| {
                        self.drafts[continuation]
                            .region
                            .inputs()
                            .iter()
                            .copied()
                            .filter(|input| *input != prefix_output)
                    })
                    .collect::<Vec<_>>();
                let split = extra_inputs
                    .iter()
                    .copied()
                    .any(|input| dependencies.depends_on_marked(input));
                let keep_prefix = external
                    || self.index.consumers[prefix_output.index()]
                        .iter()
                        .any(|consumer| {
                            owner[consumer.index()]
                                .map(|owner| !group_set.contains(&owner))
                                .unwrap_or(true)
                        });
                if !split && group.len() + usize::from(keep_prefix) < 2 {
                    continue;
                }
                // A split may consume an extra lane that depends on the prefix.
                // It must not consume a continuation that the merge would cover.
                if split && dependencies.any_draft_ancestor(&group_set, &extra_inputs) {
                    continue;
                }
                if let Some(region) =
                    self.merge_multi_region(prefix, &group, &shape, keep_prefix, split)
                {
                    let candidate = NativeRegion::MultiOutput(region);
                    let Some(disposition) = self.classify_region(&candidate) else {
                        continue;
                    };
                    if self.options.environment.fusion_debug {
                        eprintln!(
                            "[fusion] multi-merge: prefix {prefix_output} -> {} continuations (keep {keep_prefix}, split {split})",
                            group.len(),
                        );
                    }
                    let selected = if split {
                        let selected = self.drafts.len();
                        self.drafts.push(DraftRegion {
                            region: candidate,
                            disposition,
                            active: true,
                        });
                        selected
                    } else {
                        self.drafts[prefix].region = candidate;
                        self.drafts[prefix].disposition = disposition;
                        prefix
                    };
                    for &merged in &group {
                        self.drafts[merged].active = false;
                        for node in self.drafts[merged].region.nodes() {
                            owner[node.index()] = Some(selected);
                        }
                    }
                    self.work.region_table_merges += 1;
                    break;
                }
            }
        }
        debug_assert!(
            dependencies.edge_visits <= dependencies.edges.saturating_mul(dependencies.passes)
        );
        debug_assert!(
            dependencies.passes
                <= self
                    .work
                    .multi_output_work_items
                    .saturating_add(self.work.fusion_candidates)
        );
        self.work.multi_output_dependency_edges = dependencies.edges;
        self.work.multi_output_dependency_passes = dependencies.passes;
        self.work.multi_output_dependency_edge_visits = dependencies.edge_visits;
        self.work.multi_output_dependency_queries = dependencies.queries;
    }

    /// Builds a multi-output region from one prefix and its continuations.
    /// The merge rebases prefix lanes to the output shape. Continuation
    /// expressions inline the prefix expression for their shared lane. The
    /// merge fails on dtype or device mismatches, buffer or operation limit
    /// violations, or outputs beyond the backend index range.
    fn merge_multi_region(
        &self,
        prefix: usize,
        group: &[usize],
        output_shape: &[usize],
        keep_prefix: bool,
        split: bool,
    ) -> Option<MultiOutputRegion> {
        let NativeRegion::Elementwise(prefix_region) = &self.drafts[prefix].region else {
            return None;
        };
        if !split && keep_prefix && prefix_region.shape.as_ref() != output_shape {
            return None;
        }
        let prefix_as_lane = lane_strides(&prefix_region.shape, output_shape)?;
        let offset = output_shape.len() - prefix_region.shape.len();
        let mut inputs = Vec::new();
        let mut strides = Vec::new();
        let mut lane_index = HashMap::new();
        for (input, input_strides) in prefix_region
            .inputs
            .iter()
            .zip(prefix_region.lane_strides.iter())
        {
            lane_index.insert(*input, inputs.len() as u32);
            inputs.push(*input);
            strides.push(
                prefix_as_lane
                    .iter()
                    .enumerate()
                    .map(|(dim, &stride)| {
                        if stride == 0 {
                            0
                        } else {
                            input_strides[dim - offset]
                        }
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            );
        }
        let mut outputs = Vec::new();
        if !split && keep_prefix {
            outputs.push(prefix_region.output.clone());
        }
        let mut total_ops = if !split && keep_prefix {
            prefix_region.output.expression.ops()
        } else {
            0
        };
        let mut nodes = if split {
            Vec::new()
        } else {
            prefix_region.nodes.to_vec()
        };
        for &draft in group {
            let NativeRegion::Elementwise(continuation) = &self.drafts[draft].region else {
                return None;
            };
            if continuation.dtype != prefix_region.dtype
                || !continuation.device.same_device(&prefix_region.device)
            {
                return None;
            }
            let prefix_lane = continuation
                .inputs
                .iter()
                .position(|input| *input == prefix_region.output.semantic_node)?
                as u32;
            let mut remap = HashMap::new();
            for (lane, (input, input_strides)) in continuation
                .inputs
                .iter()
                .zip(continuation.lane_strides.iter())
                .enumerate()
            {
                if *input == prefix_region.output.semantic_node {
                    continue;
                }
                let merged_lane = match lane_index.get(input) {
                    Some(&lane) => lane,
                    None => {
                        let lane = inputs.len() as u32;
                        lane_index.insert(*input, lane);
                        inputs.push(*input);
                        strides.push(input_strides.clone());
                        lane
                    }
                };
                remap.insert(lane as u32, merged_lane);
            }
            let expression = continuation.output.expression.merge_lane(
                prefix_lane,
                &prefix_region.output.expression,
                &remap,
            );
            total_ops += expression.ops();
            outputs.push(ElementwiseOutput {
                semantic_node: continuation.output.semantic_node,
                expression,
            });
            nodes.extend_from_slice(&continuation.nodes);
        }
        if inputs.len() + outputs.len() > MAX_BUFFERS || total_ops > MAX_MERGED_OPS {
            return None;
        }
        normalize_nodes(&mut nodes);
        Some(MultiOutputRegion {
            nodes: nodes.into_boxed_slice(),
            inputs: inputs.into_boxed_slice(),
            lane_strides: strides.into_boxed_slice(),
            outputs: outputs.into_boxed_slice(),
            shape: output_shape.to_vec().into_boxed_slice(),
            dtype: prefix_region.dtype,
            device: prefix_region.device.clone(),
        })
    }

    /// Finalizes selection by sorting active drafts on their semantic key,
    /// building ownership and routing tables, and computing the lowering order.
    /// It rejects overlap and duplicate routing, then validates the plan.
    fn finish(mut self) -> Result<OptimizationPlan, String> {
        if let Some(error) = self.error {
            return Err(error);
        }
        let mut active = self
            .drafts
            .into_iter()
            .enumerate()
            .filter_map(|(draft, region)| {
                region.active.then_some((
                    region.region.ordering_key(),
                    draft,
                    region.region,
                    region.disposition,
                ))
            })
            .collect::<Vec<_>>();
        active.sort_by_key(|(key, draft, _, _)| (*key, *draft));
        let (regions, region_dtype_plans): (Vec<_>, Vec<_>) = active
            .into_iter()
            .map(|(_, _, region, disposition)| (region, disposition))
            .unzip();
        let mut node_region = vec![None; self.index.order.len()];
        let mut outputs = vec![None; self.index.order.len()];
        for (region_index, region) in regions.iter().enumerate() {
            let region_id = region_id(region_index)?;
            for node in region.nodes() {
                if let Some(previous) = node_region[node.index()] {
                    return Err(format!(
                        "optimization: node {node} selected by regions {previous} and {region_id}"
                    ));
                }
                node_region[node.index()] = Some(region_id);
            }
            for output in region.semantic_outputs() {
                let route = RegionOutput {
                    region: region_id,
                    index: output.index,
                };
                match outputs[output.semantic_node.index()] {
                    Some(previous) if previous != route => {
                        return Err(format!(
                            "optimization: semantic node {} has two selected outputs",
                            output.semantic_node
                        ));
                    }
                    _ => outputs[output.semantic_node.index()] = Some(route),
                }
            }
        }
        self.work.selected_regions = regions.len();
        let mut plan = OptimizationPlan {
            regions: regions.into_boxed_slice(),
            region_dtype_plans: region_dtype_plans.into_boxed_slice(),
            node_region: node_region.into_boxed_slice(),
            outputs: outputs.into_boxed_slice(),
            lowering_order: Vec::new().into_boxed_slice(),
            work: self.work,
        };
        plan.lowering_order = build_lowering_order(self.index, &plan)?.into_boxed_slice();
        plan.validate(self.index)?;
        Ok(plan)
    }
}

/// Dependency index over draft regions and independent nodes. Multi-output
/// merging uses it for two reachability queries without rebuilding adjacency:
/// whether an input descends from a marked region, following consumers, and
/// whether any input has a merge-group ancestor, following dependencies.
/// Generation stamps in `seen` and `descendants` replace per-pass visited
/// sets. Query counters feed `OptimizationWork`.
struct RegionDependencyIndex {
    unit_of_node: Vec<usize>,
    dependencies: Vec<Vec<usize>>,
    consumers: Vec<Vec<usize>>,
    descendants: Vec<u32>,
    descendant_generation: u32,
    seen: Vec<u32>,
    generation: u32,
    stack: Vec<usize>,
    edges: usize,
    passes: usize,
    edge_visits: usize,
    queries: usize,
}

impl RegionDependencyIndex {
    fn new(index: &GraphIndex, drafts: &[DraftRegion], owner: &[Option<usize>]) -> Self {
        let mut unit_of_node = vec![usize::MAX; index.order.len()];
        for (node, &draft) in owner.iter().enumerate() {
            if let Some(draft) = draft {
                unit_of_node[node] = draft;
            }
        }
        let mut unit_count = drafts.len();
        for unit in &mut unit_of_node {
            if *unit == usize::MAX {
                *unit = unit_count;
                unit_count += 1;
            }
        }
        let mut dependencies = vec![Vec::new(); unit_count];
        let mut add_dependency = |unit: usize, dependency: usize| {
            if unit != dependency && !dependencies[unit].contains(&dependency) {
                dependencies[unit].push(dependency);
            }
        };
        for (draft, region) in drafts.iter().enumerate() {
            for input in region.region.inputs() {
                add_dependency(draft, unit_of_node[input.index()]);
            }
        }
        for dense_index in 0..index.order.len() {
            if owner[dense_index].is_some() {
                continue;
            }
            let unit = unit_of_node[dense_index];
            for child in index.children[dense_index].iter() {
                add_dependency(unit, unit_of_node[child.index()]);
            }
        }
        let edges = dependencies.iter().map(Vec::len).sum();
        let mut consumers = vec![Vec::new(); unit_count];
        for (unit, inputs) in dependencies.iter().enumerate() {
            for &dependency in inputs {
                consumers[dependency].push(unit);
            }
        }
        Self {
            unit_of_node,
            dependencies,
            consumers,
            descendants: vec![0; unit_count],
            descendant_generation: 0,
            seen: vec![0; unit_count],
            generation: 0,
            stack: Vec::new(),
            edges,
            passes: 0,
            edge_visits: 0,
            queries: 0,
        }
    }

    fn next_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.seen.fill(0);
            self.generation = 1;
        }
    }

    fn mark_descendants(&mut self, draft: usize) {
        self.descendant_generation = self.descendant_generation.wrapping_add(1);
        if self.descendant_generation == 0 {
            self.descendants.fill(0);
            self.descendant_generation = 1;
        }
        self.passes += 1;
        self.stack.clear();
        self.stack.push(draft);
        while let Some(unit) = self.stack.pop() {
            if self.descendants[unit] == self.descendant_generation {
                continue;
            }
            self.descendants[unit] = self.descendant_generation;
            self.edge_visits += self.consumers[unit].len();
            self.stack.extend(self.consumers[unit].iter().copied());
        }
    }

    fn depends_on_marked(&mut self, node: DenseNodeId) -> bool {
        self.queries += 1;
        self.descendants[self.unit_of_node[node.index()]] == self.descendant_generation
    }

    fn any_draft_ancestor(&mut self, drafts: &HashSet<usize>, inputs: &[DenseNodeId]) -> bool {
        self.next_generation();
        self.passes += 1;
        self.queries += inputs.len();
        self.stack.clear();
        self.stack
            .extend(inputs.iter().map(|input| self.unit_of_node[input.index()]));
        while let Some(unit) = self.stack.pop() {
            if self.seen[unit] == self.generation {
                continue;
            }
            self.seen[unit] = self.generation;
            if drafts.contains(&unit) {
                return true;
            }
            self.edge_visits += self.dependencies[unit].len();
            self.stack.extend(self.dependencies[unit].iter().copied());
        }
        false
    }
}

/// An elementwise chain under construction. It stores the current expression,
/// boundary `inputs`, stable lane indexes in `lane_of`, covered semantic
/// nodes, and the operation count used by the `ops >= 2` emission threshold.
struct OpenRegion {
    expression: KernelExpr,
    inputs: Vec<DenseNodeId>,
    lane_of: HashMap<DenseNodeId, u32>,
    nodes: Vec<DenseNodeId>,
    ops: usize,
}

impl OpenRegion {
    fn empty() -> Self {
        Self {
            expression: KernelExpr::cst(0.0),
            inputs: Vec::new(),
            lane_of: HashMap::new(),
            nodes: Vec::new(),
            ops: 0,
        }
    }

    /// Returns the lane expression for `node` and allocates an input lane on
    /// first use.
    fn lane(&mut self, node: DenseNodeId) -> KernelExpr {
        if let Some(&lane) = self.lane_of.get(&node) {
            return KernelExpr::Input(lane);
        }
        let lane = self.inputs.len() as u32;
        self.inputs.push(node);
        self.lane_of.insert(node, lane);
        KernelExpr::Input(lane)
    }

    /// Merges another open region into this one. It appends inputs and uses
    /// `lane_of` to remove duplicates. It transfers covered nodes and the
    /// operation count, then returns the expression in the merged lane namespace.
    fn absorb(&mut self, other: OpenRegion) -> KernelExpr {
        let mut remap = HashMap::new();
        for (lane, input) in other.inputs.iter().enumerate() {
            let merged_lane = match self.lane_of.get(input) {
                Some(&lane) => lane,
                None => {
                    let lane = self.inputs.len() as u32;
                    self.inputs.push(*input);
                    self.lane_of.insert(*input, lane);
                    lane
                }
            };
            remap.insert(lane as u32, merged_lane);
        }
        self.nodes.extend(other.nodes);
        self.ops += other.ops;
        other.expression.remap_lanes(&remap)
    }
}

#[derive(Clone, Copy)]
enum UnaryOperation {
    Neg,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Relu,
    Tanh,
    Gelu(bool),
    Abs,
    Erf,
    Floor,
    Ceil,
    Round,
    Pow(f64),
    Sign,
    Cast(DType),
}

impl UnaryOperation {
    fn apply(self, input: KernelExpr) -> KernelExpr {
        match self {
            Self::Neg => KernelExpr::Neg(Box::new(input)),
            Self::Sqrt => KernelExpr::Sqrt(Box::new(input)),
            Self::Exp => KernelExpr::Exp(Box::new(input)),
            Self::Log => KernelExpr::Log(Box::new(input)),
            Self::Sin => KernelExpr::Sin(Box::new(input)),
            Self::Cos => KernelExpr::Cos(Box::new(input)),
            Self::Relu => KernelExpr::Max(Box::new(input), Box::new(KernelExpr::cst(0.0))),
            Self::Tanh => KernelExpr::Tanh(Box::new(input)),
            Self::Gelu(true) => KernelExpr::GeluTanh(Box::new(input)),
            Self::Gelu(false) => KernelExpr::Gelu(Box::new(input)),
            Self::Abs => KernelExpr::Abs(Box::new(input)),
            Self::Erf => KernelExpr::Erf(Box::new(input)),
            Self::Floor => KernelExpr::Floor(Box::new(input)),
            Self::Ceil => KernelExpr::Ceil(Box::new(input)),
            Self::Round => KernelExpr::Round(Box::new(input)),
            Self::Pow(exponent) => pow_expr(input, exponent),
            Self::Sign => KernelExpr::Select(
                Box::new(KernelExpr::Gt(
                    Box::new(input.clone()),
                    Box::new(KernelExpr::cst(0.0)),
                )),
                Box::new(KernelExpr::cst(1.0)),
                Box::new(KernelExpr::Select(
                    Box::new(KernelExpr::Lt(
                        Box::new(input),
                        Box::new(KernelExpr::cst(0.0)),
                    )),
                    Box::new(KernelExpr::cst(-1.0)),
                    Box::new(KernelExpr::cst(0.0)),
                )),
            ),
            Self::Cast(dtype) => KernelExpr::Cast(Box::new(input), dtype),
        }
    }
}

#[derive(Clone, Copy)]
enum BinaryOperation {
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    Minimum,
}

impl BinaryOperation {
    fn apply(self, left: KernelExpr, right: KernelExpr) -> KernelExpr {
        match self {
            Self::Add => KernelExpr::Add(Box::new(left), Box::new(right)),
            Self::Sub => KernelExpr::Sub(Box::new(left), Box::new(right)),
            Self::Mul => KernelExpr::Mul(Box::new(left), Box::new(right)),
            Self::Div => KernelExpr::Div(Box::new(left), Box::new(right)),
            Self::Maximum => KernelExpr::Max(Box::new(left), Box::new(right)),
            Self::Minimum => KernelExpr::Min(Box::new(left), Box::new(right)),
        }
    }
}

#[derive(Clone, Copy)]
enum ComparisonOperation {
    Eq,
    Gt,
    Lt,
    Ge,
    Le,
}

impl ComparisonOperation {
    fn apply(self, left: KernelExpr, right: KernelExpr) -> KernelExpr {
        match self {
            Self::Eq => KernelExpr::Eq(Box::new(left), Box::new(right)),
            Self::Gt => KernelExpr::Gt(Box::new(left), Box::new(right)),
            Self::Lt => KernelExpr::Lt(Box::new(left), Box::new(right)),
            Self::Ge => KernelExpr::Ge(Box::new(left), Box::new(right)),
            Self::Le => KernelExpr::Le(Box::new(left), Box::new(right)),
        }
    }
}

enum ElementOperation {
    Unary(UnaryOperation),
    Binary(BinaryOperation),
    Select(ComparisonOperation),
    Reduce(ReduceOp, Vec<usize>, bool),
    ArgReduce(bool, usize),
}

fn reduced_shape(shape: &[usize], dims: &[usize], keepdims: bool) -> Vec<usize> {
    if keepdims {
        shape
            .iter()
            .enumerate()
            .map(|(index, &dim)| if dims.contains(&index) { 1 } else { dim })
            .collect()
    } else {
        shape
            .iter()
            .enumerate()
            .filter_map(|(index, &dim)| (!dims.contains(&index)).then_some(dim))
            .collect()
    }
}

fn normalize_nodes(nodes: &mut Vec<DenseNodeId>) {
    nodes.sort_unstable();
    nodes.dedup();
}

fn region_id(index: usize) -> Result<RegionId, String> {
    RegionId::from_index(index).ok_or_else(|| "optimization: too many selected regions".to_string())
}

/// Builds a topological lowering order with one unit per region and independent
/// node. Ready regions use their ordering key. Ready nodes use their index.
/// This priority keeps identical plans stable. A remaining indegree indicates
/// a dependency cycle, reported with the first blocked units.
fn build_lowering_order(
    index: &GraphIndex,
    plan: &OptimizationPlan,
) -> Result<Vec<LoweringUnit>, String> {
    let region_count = plan.regions.len();
    let mut node_units = vec![None; index.order.len()];
    let mut units = (0..region_count)
        .map(|region| LoweringUnit::Region(region_id(region).unwrap()))
        .collect::<Vec<_>>();
    for dense_index in 0..index.order.len() {
        if plan.node_region[dense_index].is_none() {
            let unit = units.len();
            node_units[dense_index] = Some(unit);
            units.push(LoweringUnit::Node(
                DenseNodeId::from_index(dense_index)
                    .expect("GraphIndex validated the semantic node count"),
            ));
        }
    }
    let mut dependencies = vec![Vec::new(); units.len()];
    let source_unit = |node: DenseNodeId| -> Result<usize, String> {
        if let Some(output) = plan.outputs[node.index()] {
            Ok(output.region.index())
        } else if let Some(unit) = node_units[node.index()] {
            Ok(unit)
        } else {
            Err(format!(
                "optimization: internal semantic node {node} escapes its region"
            ))
        }
    };
    for (region, native) in plan.regions.iter().enumerate() {
        for &input in native.inputs() {
            let dependency = source_unit(input)?;
            if dependency != region && !dependencies[region].contains(&dependency) {
                dependencies[region].push(dependency);
            }
        }
    }
    for dense_index in 0..index.order.len() {
        let Some(unit) = node_units[dense_index] else {
            continue;
        };
        for &child in index.children[dense_index].iter() {
            let dependency = source_unit(child)?;
            if dependency != unit && !dependencies[unit].contains(&dependency) {
                dependencies[unit].push(dependency);
            }
        }
    }
    let mut consumers = vec![Vec::new(); units.len()];
    let mut indegree = vec![0usize; units.len()];
    for (unit, deps) in dependencies.iter().enumerate() {
        indegree[unit] = deps.len();
        for &dependency in deps {
            consumers[dependency].push(unit);
        }
    }
    let priority = |unit: usize| match units[unit] {
        LoweringUnit::Node(node) => (node.index(), 1u8, node.index()),
        LoweringUnit::Region(region) => (
            plan.regions[region.index()].ordering_key(),
            0u8,
            region.index(),
        ),
    };
    let mut ready = BinaryHeap::new();
    for unit in 0..units.len() {
        if indegree[unit] == 0 {
            ready.push(Reverse(priority(unit)));
        }
    }
    let mut order = Vec::with_capacity(units.len());
    while let Some(Reverse((_, kind, identity))) = ready.pop() {
        let unit = if kind == 0 {
            identity
        } else {
            node_units[identity].expect("ready independent node has a lowering unit")
        };
        order.push(units[unit]);
        for &consumer in &consumers[unit] {
            indegree[consumer] -= 1;
            if indegree[consumer] == 0 {
                ready.push(Reverse(priority(consumer)));
            }
        }
    }
    if order.len() != units.len() {
        let mut blocked = Vec::new();
        for (unit, &degree) in indegree.iter().enumerate() {
            if degree == 0 {
                continue;
            }
            let dependency = dependencies[unit]
                .iter()
                .copied()
                .find(|&dependency| indegree[dependency] != 0);
            let unit = lowering_unit_name(units[unit]);
            match dependency {
                Some(dependency) => blocked.push(format!(
                    "{unit} depends on {}",
                    lowering_unit_name(units[dependency])
                )),
                None => blocked.push(unit),
            }
            if blocked.len() == 8 {
                break;
            }
        }
        return Err(format!(
            "optimization: lowering-unit dependency cycle ({})",
            blocked.join(", ")
        ));
    }
    Ok(order)
}

fn lowering_unit_name(unit: LoweringUnit) -> String {
    match unit {
        LoweringUnit::Node(node) => format!("node {node}"),
        LoweringUnit::Region(region) => format!("region {region}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn input(slot: u32, shape: &[usize], dtype: DType, device: Device) -> Arc<Node> {
        Node::new(NodeKind::Input {
            storage: effect_torch_runtime::StorageMetadata::dense(),
            slot,
            shape: shape.to_vec(),
            dtype,
            device,
        })
        .unwrap()
    }

    fn zeros(shape: &[usize], dtype: DType, device: Device) -> Arc<Node> {
        Node::new(NodeKind::Zeros {
            shape: shape.to_vec(),
            dtype,
            device,
        })
        .unwrap()
    }

    fn dense(index: &GraphIndex, node: &Arc<Node>) -> DenseNodeId {
        index.dense_id(node.id).unwrap()
    }

    fn elementwise_shared_graph(
        materialize_prefix: bool,
    ) -> (Vec<Arc<Node>>, Arc<Node>, Arc<Node>, Arc<Node>) {
        let x = input(0, &[2, 3], DType::F32, Device::Cpu(0));
        let y = input(1, &[2, 3], DType::F32, Device::Cpu(0));
        let z = input(2, &[3], DType::F32, Device::Cpu(0));
        let sum = Node::new(NodeKind::Add { a: x, b: y }).unwrap();
        let prefix = Node::new(NodeKind::Tanh { a: sum.clone() }).unwrap();
        let left = Node::new(NodeKind::Neg { a: prefix.clone() }).unwrap();
        let left = Node::new(NodeKind::Exp { a: left }).unwrap();
        let right = Node::new(NodeKind::Mul {
            a: prefix.clone(),
            b: z,
        })
        .unwrap();
        let right = Node::new(NodeKind::Sin { a: right }).unwrap();
        let roots = if materialize_prefix {
            vec![prefix.clone(), left.clone(), right.clone()]
        } else {
            vec![left.clone(), right.clone()]
        };
        (roots, prefix, left, right)
    }

    #[test]
    fn indexed_selection_preserves_all_semantic_ids_and_arc_identities() {
        let x = input(0, &[4], DType::F32, Device::Cpu(0));
        let y = input(1, &[4], DType::F32, Device::Cpu(0));
        let sum = Node::new(NodeKind::Add { a: x, b: y }).unwrap();
        let root = Node::new(NodeKind::Tanh { a: sum }).unwrap();
        let index = GraphIndex::new(std::slice::from_ref(&root)).unwrap();
        let identities = index
            .order
            .iter()
            .map(|node| (node.id, Arc::as_ptr(node) as usize))
            .collect::<Vec<_>>();

        let plan = build_optimization_plan(
            &index,
            &CompileOptions::default(),
            &crate::test_target::TestTarget::for_index(&index),
        )
        .unwrap();

        assert_eq!(plan.regions.len(), 1);
        let NativeRegion::Elementwise(region) = &plan.regions[0] else {
            panic!("expected one elementwise region")
        };
        assert_eq!(region.output.semantic_node, dense(&index, &root));
        assert_eq!(region.inputs.len(), 2);
        assert_eq!(
            identities,
            index
                .order
                .iter()
                .map(|node| (node.id, Arc::as_ptr(node) as usize))
                .collect::<Vec<_>>()
        );
        assert_eq!(plan.work.graph_index_builds, 1);
        assert_eq!(plan.work.semantic_nodes_rebuilt, 0);
        plan.validate(&index).unwrap();
    }

    #[test]
    fn optimize_false_is_an_empty_plan_and_duplicate_roots_stay_routed() {
        let x = input(0, &[4], DType::F32, Device::Cpu(0));
        let neg = Node::new(NodeKind::Neg { a: x }).unwrap();
        let root = Node::new(NodeKind::Tanh { a: neg }).unwrap();
        let index = GraphIndex::new(&[root.clone(), root.clone()]).unwrap();
        let options = CompileOptions {
            optimize: false,
            ..CompileOptions::default()
        };
        let plan = build_optimization_plan(
            &index,
            &options,
            &crate::test_target::TestTarget::for_index(&index),
        )
        .unwrap();
        assert!(plan.regions.is_empty());
        assert!(plan.node_region.iter().all(Option::is_none));
        assert!(plan.outputs.iter().all(Option::is_none));
        assert_eq!(
            index.roots.as_ref(),
            [dense(&index, &root), dense(&index, &root)]
        );
        assert_eq!(
            plan.resolve(dense(&index, &root)),
            Ok(ValueSource::Independent(dense(&index, &root)))
        );
        plan.validate(&index).unwrap();
    }

    #[test]
    fn duplicate_optimized_roots_share_one_region_output() {
        let x = input(0, &[4], DType::F32, Device::Cpu(0));
        let neg = Node::new(NodeKind::Neg { a: x }).unwrap();
        let root = Node::new(NodeKind::Tanh { a: neg }).unwrap();
        let index = GraphIndex::new(&[root.clone(), root.clone()]).unwrap();
        let plan = build_optimization_plan(
            &index,
            &CompileOptions::default(),
            &crate::test_target::TestTarget::for_index(&index),
        )
        .unwrap();
        let route = plan.outputs[dense(&index, &root).index()].unwrap();
        assert_eq!(route.index, 0);
        assert_eq!(index.roots[0], index.roots[1]);
        assert_eq!(plan.resolve(index.roots[0]), plan.resolve(index.roots[1]));
    }

    #[test]
    fn reduction_region_records_normalized_geometry_and_expression_inputs() {
        let x = input(0, &[2, 3], DType::F64, Device::Cpu(0));
        let y = input(1, &[3], DType::F64, Device::Cpu(0));
        let add = Node::new(NodeKind::Add { a: x, b: y }).unwrap();
        let root = Node::new(NodeKind::Mean {
            a: add.clone(),
            dims: vec![1, 1],
            keepdims: true,
        })
        .unwrap();
        let index = GraphIndex::new(std::slice::from_ref(&root)).unwrap();
        let plan = build_optimization_plan(
            &index,
            &CompileOptions::default(),
            &crate::test_target::TestTarget::for_index(&index),
        )
        .unwrap();
        let NativeRegion::ElementwiseReduce(region) = &plan.regions[0] else {
            panic!("expected a fused reduction")
        };
        assert_eq!(
            region.nodes.as_ref(),
            [dense(&index, &add), dense(&index, &root)]
        );
        assert_eq!(region.dims.as_ref(), [1]);
        assert_eq!(region.input_shape.as_ref(), [2, 3]);
        assert_eq!(region.shape.as_ref(), [2, 1]);
        assert_eq!(region.op, ReduceOp::Mean);
        assert_eq!(region.lane_strides[1].as_ref(), [0, 1]);
        assert_eq!(plan.outputs[dense(&index, &root).index()].unwrap().index, 0);
    }

    #[test]
    fn where_fuses_its_single_use_comparison_as_a_true_select() {
        let x = input(0, &[4], DType::F32, Device::Cpu(0));
        let y = input(1, &[4], DType::F32, Device::Cpu(0));
        let condition = Node::new(NodeKind::Gt {
            a: x.clone(),
            b: y.clone(),
        })
        .unwrap();
        let selected = Node::new(NodeKind::Where {
            cond: condition.clone(),
            a: Node::new(NodeKind::Full {
                shape: vec![1],
                value: 2.0,
                dtype: DType::F32,
                device: Device::Cpu(0),
            })
            .unwrap(),
            b: zeros(&[4], DType::F32, Device::Cpu(0)),
        })
        .unwrap();
        let root = Node::new(NodeKind::Tanh { a: selected }).unwrap();
        let index = GraphIndex::new(std::slice::from_ref(&root)).unwrap();
        let plan = build_optimization_plan(
            &index,
            &CompileOptions::default(),
            &crate::test_target::TestTarget::for_index(&index),
        )
        .unwrap();
        let NativeRegion::Elementwise(region) = &plan.regions[0] else {
            panic!("expected an elementwise select region")
        };
        let KernelExpr::Semantic(continuation, _, _) = &region.output.expression else {
            panic!("expected the continuation semantic boundary")
        };
        let KernelExpr::Tanh(expression) = continuation.as_ref() else {
            panic!("expected the continuation after select")
        };
        let KernelExpr::Semantic(selected, _, _) = expression.as_ref() else {
            panic!("expected the select semantic boundary")
        };
        assert!(matches!(selected.as_ref(), KernelExpr::Select(..)));
        assert!(region.nodes.contains(&dense(&index, &condition)));
        assert_eq!(
            region.inputs.as_ref(),
            [dense(&index, &x), dense(&index, &y)]
        );
    }

    #[test]
    fn metal_linear_residual_has_exact_coverage_inputs_and_output() {
        let x = input(0, &[2, 3], DType::BF16, Device::Metal(0));
        let weight = input(1, &[3, 4], DType::BF16, Device::Metal(0));
        let bias = input(2, &[4], DType::BF16, Device::Metal(0));
        let residual = input(3, &[2, 4], DType::BF16, Device::Metal(0));
        let linear = Node::new(NodeKind::Linear { x, weight, bias }).unwrap();
        let root = Node::new(NodeKind::Add {
            a: linear.clone(),
            b: residual.clone(),
        })
        .unwrap();
        let index = GraphIndex::new(std::slice::from_ref(&root)).unwrap();
        let plan = build_optimization_plan(
            &index,
            &CompileOptions::default(),
            &crate::test_target::TestTarget::for_index(&index),
        )
        .unwrap();
        let NativeRegion::LinearResidual(region) = &plan.regions[0] else {
            panic!("expected a linear residual region")
        };
        assert_eq!(
            region.nodes.as_ref(),
            [dense(&index, &linear), dense(&index, &root)]
        );
        assert_eq!(region.inputs[3], dense(&index, &residual));
        assert_eq!(region.output, dense(&index, &root));
        assert!(plan.outputs[dense(&index, &linear).index()].is_none());
        assert_eq!(plan.outputs[dense(&index, &root).index()].unwrap().index, 0);
    }

    #[test]
    fn metal_linear_gelu_dual_routes_pre_activation_and_gelu() {
        let x = input(0, &[2, 3], DType::F32, Device::Metal(0));
        let weight = input(1, &[3, 4], DType::F32, Device::Metal(0));
        let bias = input(2, &[4], DType::F32, Device::Metal(0));
        let linear = Node::new(NodeKind::Linear { x, weight, bias }).unwrap();
        let gelu = Node::new(NodeKind::Gelu {
            a: linear.clone(),
            approximate: true,
        })
        .unwrap();
        let other = Node::new(NodeKind::Neg { a: linear.clone() }).unwrap();
        let index = GraphIndex::new(&[gelu.clone(), other]).unwrap();
        let plan = build_optimization_plan(
            &index,
            &CompileOptions::default(),
            &crate::test_target::TestTarget::for_index(&index),
        )
        .unwrap();
        let NativeRegion::LinearGelu(region) = &plan.regions[0] else {
            panic!("expected a linear gelu region")
        };
        assert!(region.dual);
        assert!(region.approximate);
        assert_eq!(
            plan.outputs[dense(&index, &linear).index()].unwrap().index,
            0
        );
        assert_eq!(plan.outputs[dense(&index, &gelu).index()].unwrap().index, 1);
        assert_eq!(region.inputs.len(), 3);
        plan.validate(&index).unwrap();
    }

    #[test]
    fn multi_output_inlines_a_nonmaterialized_shared_prefix() {
        let (roots, prefix, left, right) = elementwise_shared_graph(false);
        let index = GraphIndex::new(&roots).unwrap();
        let plan = build_optimization_plan(
            &index,
            &CompileOptions::default(),
            &crate::test_target::TestTarget::for_index(&index),
        )
        .unwrap();
        assert_eq!(plan.regions.len(), 1);
        let NativeRegion::MultiOutput(region) = &plan.regions[0] else {
            panic!("expected a multi-output region")
        };
        assert_eq!(region.outputs.len(), 2);
        assert_eq!(region.outputs[0].semantic_node, dense(&index, &left));
        assert_eq!(region.outputs[1].semantic_node, dense(&index, &right));
        assert!(plan.outputs[dense(&index, &prefix).index()].is_none());
        assert_eq!(plan.outputs[dense(&index, &left).index()].unwrap().index, 0);
        assert_eq!(
            plan.outputs[dense(&index, &right).index()].unwrap().index,
            1
        );
        assert_eq!(plan.work.region_table_merges, 1);
    }

    #[test]
    fn multi_output_materializes_a_root_prefix_at_output_zero() {
        let (roots, prefix, left, right) = elementwise_shared_graph(true);
        let index = GraphIndex::new(&roots).unwrap();
        let plan = build_optimization_plan(
            &index,
            &CompileOptions::default(),
            &crate::test_target::TestTarget::for_index(&index),
        )
        .unwrap();
        let NativeRegion::MultiOutput(region) = &plan.regions[0] else {
            panic!("expected a multi-output region")
        };
        assert_eq!(region.outputs.len(), 3);
        assert_eq!(region.outputs[0].semantic_node, dense(&index, &prefix));
        assert_eq!(region.outputs[1].semantic_node, dense(&index, &left));
        assert_eq!(region.outputs[2].semantic_node, dense(&index, &right));
        assert_eq!(
            plan.outputs[dense(&index, &prefix).index()].unwrap().index,
            0
        );
        assert_eq!(plan.outputs[dense(&index, &left).index()].unwrap().index, 1);
        assert_eq!(
            plan.outputs[dense(&index, &right).index()].unwrap().index,
            2
        );
    }

    #[test]
    fn multi_output_incorporates_a_direct_nested_lane_before_its_upstream_merge() {
        let x = input(0, &[2, 3], DType::F32, Device::Cpu(0));
        let y = input(1, &[2, 3], DType::F32, Device::Cpu(0));
        let sum = Node::new(NodeKind::Add { a: x, b: y }).unwrap();
        let prefix = Node::new(NodeKind::Tanh { a: sum }).unwrap();
        let safe = Node::new(NodeKind::Neg { a: prefix.clone() }).unwrap();
        let safe = Node::new(NodeKind::Exp { a: safe }).unwrap();
        let nested = Node::new(NodeKind::Add {
            a: prefix.clone(),
            b: safe.clone(),
        })
        .unwrap();
        let nested = Node::new(NodeKind::Sin { a: nested }).unwrap();
        let sibling = Node::new(NodeKind::Abs { a: prefix.clone() }).unwrap();
        let sibling = Node::new(NodeKind::Sqrt { a: sibling }).unwrap();
        let index = GraphIndex::new(&[safe.clone(), nested.clone(), sibling.clone()]).unwrap();

        let plan = build_optimization_plan(
            &index,
            &CompileOptions::default(),
            &crate::test_target::TestTarget::for_index(&index),
        )
        .unwrap();

        assert_eq!(plan.work.region_table_merges, 2);
        let multis = plan
            .regions
            .iter()
            .filter_map(|region| match region {
                NativeRegion::MultiOutput(region) => Some(region.outputs.as_ref()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(multis.len(), 2);
        assert!(multis.iter().any(|outputs| {
            outputs[0].semantic_node == dense(&index, &prefix)
                && outputs[1].semantic_node == dense(&index, &sibling)
        }));
        assert!(multis.iter().any(|outputs| {
            outputs[0].semantic_node == dense(&index, &safe)
                && outputs[1].semantic_node == dense(&index, &nested)
        }));
        assert_eq!(
            plan.outputs[dense(&index, &prefix).index()].unwrap().index,
            0
        );
        assert_eq!(plan.outputs[dense(&index, &safe).index()].unwrap().index, 0);
        assert_eq!(
            plan.outputs[dense(&index, &nested).index()].unwrap().index,
            1
        );
        assert_eq!(
            plan.outputs[dense(&index, &sibling).index()].unwrap().index,
            1
        );
        plan.validate(&index).unwrap();
    }

    #[test]
    fn multi_output_splits_a_lane_with_transitive_prefix_ancestry() {
        let x = input(0, &[2, 3], DType::F32, Device::Cpu(0));
        let y = input(1, &[2, 3], DType::F32, Device::Cpu(0));
        let sum = Node::new(NodeKind::Add {
            a: x.clone(),
            b: y.clone(),
        })
        .unwrap();
        let prefix = Node::new(NodeKind::Tanh { a: sum.clone() }).unwrap();
        let safe = Node::new(NodeKind::Neg { a: prefix.clone() }).unwrap();
        let safe = Node::new(NodeKind::Exp { a: safe }).unwrap();
        let reduced_input = Node::new(NodeKind::Abs { a: prefix.clone() }).unwrap();
        let reduced = Node::new(NodeKind::Sum {
            a: reduced_input,
            dims: vec![1],
            keepdims: true,
        })
        .unwrap();
        let nested = Node::new(NodeKind::Div {
            a: prefix.clone(),
            b: reduced.clone(),
        })
        .unwrap();
        let nested = Node::new(NodeKind::Sin { a: nested }).unwrap();
        let index = GraphIndex::new(&[safe.clone(), nested.clone()]).unwrap();

        let plan = build_optimization_plan(
            &index,
            &CompileOptions::default(),
            &crate::test_target::TestTarget::for_index(&index),
        )
        .unwrap();

        assert_eq!(plan.work.region_table_merges, 1);
        let (multi_region_id, multi) = plan
            .regions
            .iter()
            .enumerate()
            .find_map(|(region_id, region)| match region {
                NativeRegion::MultiOutput(region) => Some((region_id, region)),
                _ => None,
            })
            .expect("expected a split continuation region");
        assert_eq!(multi.outputs[0].semantic_node, dense(&index, &safe));
        assert_eq!(multi.outputs[1].semantic_node, dense(&index, &nested));
        assert!(!multi.nodes.contains(&dense(&index, &sum)));
        assert!(!multi.nodes.contains(&dense(&index, &prefix)));
        assert!(multi.nodes.contains(&dense(&index, &safe)));
        assert!(multi.nodes.contains(&dense(&index, &nested)));
        assert!(multi.inputs.contains(&dense(&index, &x)));
        assert!(multi.inputs.contains(&dense(&index, &y)));
        assert!(multi.inputs.contains(&dense(&index, &reduced)));
        assert!(!multi.inputs.contains(&dense(&index, &prefix)));
        let prefix_route = plan.outputs[dense(&index, &prefix).index()].unwrap();
        assert_ne!(prefix_route.region.index(), multi_region_id);
        assert!(matches!(
            &plan.regions[prefix_route.region.index()],
            NativeRegion::Elementwise(region)
                if region.output.semantic_node == dense(&index, &prefix)
        ));
        assert_eq!(
            plan.outputs[dense(&index, &safe).index()]
                .unwrap()
                .region
                .index(),
            multi_region_id
        );
        assert_eq!(
            plan.outputs[dense(&index, &nested).index()]
                .unwrap()
                .region
                .index(),
            multi_region_id
        );
        assert!(plan.regions.iter().any(
            |region| matches!(region, NativeRegion::ElementwiseReduce(region) if region.output == dense(&index, &reduced))
        ));
        let reduced_route = plan.outputs[dense(&index, &reduced).index()].unwrap();
        let lowering_position = |region| {
            plan.lowering_order
                .iter()
                .position(|unit| *unit == LoweringUnit::Region(region))
                .unwrap()
        };
        assert!(lowering_position(prefix_route.region) < lowering_position(reduced_route.region));
        assert!(
            lowering_position(reduced_route.region)
                < lowering_position(RegionId::from_index(multi_region_id).unwrap())
        );
        assert_eq!(plan.work.semantic_nodes_rebuilt, 0);
        plan.validate(&index).unwrap();
    }

    #[test]
    fn multi_output_dependency_work_is_bounded_on_a_wide_graph() {
        let width = 256;
        let x = input(0, &[8], DType::F32, Device::Cpu(0));
        let y = input(1, &[8], DType::F32, Device::Cpu(0));
        let prefix = Node::new(NodeKind::Tanh {
            a: Node::new(NodeKind::Add { a: x.clone(), b: y }).unwrap(),
        })
        .unwrap();
        let mut extra = x;
        for _ in 0..width {
            extra = Node::new(NodeKind::Neg { a: extra }).unwrap();
        }
        let mut roots = Vec::with_capacity(width + 1);
        roots.push(prefix.clone());
        for _ in 0..width {
            let branch = Node::new(NodeKind::Add {
                a: prefix.clone(),
                b: extra.clone(),
            })
            .unwrap();
            roots.push(Node::new(NodeKind::Sin { a: branch }).unwrap());
        }
        let index = GraphIndex::new(&roots).unwrap();

        let plan = build_optimization_plan(
            &index,
            &CompileOptions::default(),
            &crate::test_target::TestTarget::for_index(&index),
        )
        .unwrap();

        assert!(plan.work.multi_output_dependency_edges > 0);
        assert!(plan.work.multi_output_dependency_passes <= plan.work.fusion_candidates + 1);
        assert!(
            plan.work.multi_output_dependency_edge_visits
                <= plan.work.multi_output_dependency_edges
                    * plan.work.multi_output_dependency_passes
        );
        assert!(
            plan.work.multi_output_dependency_queries <= plan.work.fusion_candidates * MAX_BUFFERS
        );
        plan.validate(&index).unwrap();
    }

    #[test]
    fn multi_output_worklist_preserves_nested_shared_prefix_opportunities() {
        let x = input(0, &[2, 3], DType::F32, Device::Cpu(0));
        let y = input(1, &[2, 3], DType::F32, Device::Cpu(0));
        let sum = Node::new(NodeKind::Add { a: x, b: y }).unwrap();
        let prefix = Node::new(NodeKind::Tanh { a: sum }).unwrap();
        let sibling = Node::new(NodeKind::Neg { a: prefix.clone() }).unwrap();
        let sibling = Node::new(NodeKind::Exp { a: sibling }).unwrap();
        let nested_prefix = Node::new(NodeKind::Sin { a: prefix.clone() }).unwrap();
        let nested_prefix = Node::new(NodeKind::Cos { a: nested_prefix }).unwrap();
        let left = Node::new(NodeKind::Neg {
            a: nested_prefix.clone(),
        })
        .unwrap();
        let left = Node::new(NodeKind::Exp { a: left }).unwrap();
        let right = Node::new(NodeKind::Abs {
            a: nested_prefix.clone(),
        })
        .unwrap();
        let right = Node::new(NodeKind::Sqrt { a: right }).unwrap();
        let index = GraphIndex::new(&[sibling, left, right]).unwrap();

        let plan = build_optimization_plan(
            &index,
            &CompileOptions::default(),
            &crate::test_target::TestTarget::for_index(&index),
        )
        .unwrap();

        assert_eq!(plan.work.region_table_merges, 2);
        assert_eq!(
            plan.regions
                .iter()
                .filter(|region| matches!(region, NativeRegion::MultiOutput(_)))
                .count(),
            2
        );
        assert!(plan.outputs[dense(&index, &prefix).index()].is_some());
        assert!(plan.outputs[dense(&index, &nested_prefix).index()].is_none());
        plan.validate(&index).unwrap();
    }

    #[test]
    fn shared_softmax_gradient_reduction_topology_has_an_acyclic_plan() {
        let x = input(0, &[4, 4], DType::F32, Device::Cpu(0));
        let weight = input(1, &[4, 4], DType::F32, Device::Cpu(0));
        let row_max = Node::new(NodeKind::Max {
            a: x.clone(),
            dims: vec![1],
            keepdims: true,
        })
        .unwrap();
        let centered = Node::new(NodeKind::Sub {
            a: x.clone(),
            b: row_max,
        })
        .unwrap();
        let numerator = Node::new(NodeKind::Exp { a: centered }).unwrap();
        let denominator = Node::new(NodeKind::Sum {
            a: numerator.clone(),
            dims: vec![1],
            keepdims: true,
        })
        .unwrap();
        let probabilities = Node::new(NodeKind::Div {
            a: numerator,
            b: denominator,
        })
        .unwrap();
        let weighted = Node::new(NodeKind::Mul {
            a: probabilities.clone(),
            b: weight,
        })
        .unwrap();
        let loss = Node::new(NodeKind::Sum {
            a: weighted,
            dims: vec![0, 1],
            keepdims: false,
        })
        .unwrap();
        let gradient = effect_torch_autodiff::grad(&loss, std::slice::from_ref(&x))
            .unwrap()
            .remove(0);
        let index = GraphIndex::new(&[probabilities, gradient]).unwrap();

        let plan = build_optimization_plan(
            &index,
            &CompileOptions::default(),
            &crate::test_target::TestTarget::for_index(&index),
        )
        .unwrap();

        assert!(plan
            .regions
            .iter()
            .any(|region| matches!(region, NativeRegion::ElementwiseReduce(_))));
        assert_eq!(plan.work.semantic_nodes_rebuilt, 0);
        plan.validate(&index).unwrap();
    }

    #[test]
    fn plan_selection_is_deterministic() {
        let (roots, _, _, _) = elementwise_shared_graph(true);
        let index = GraphIndex::new(&roots).unwrap();
        let first = build_optimization_plan(
            &index,
            &CompileOptions::default(),
            &crate::test_target::TestTarget::for_index(&index),
        )
        .unwrap();
        let second = build_optimization_plan(
            &index,
            &CompileOptions::default(),
            &crate::test_target::TestTarget::for_index(&index),
        )
        .unwrap();
        assert_eq!(first, second);
    }

    fn adamw_step(dtype: DType, device: Device, scalar_inputs: &[Arc<Node>; 3]) -> Arc<Node> {
        Node::new(NodeKind::AdamWStep {
            param: zeros(&[8], dtype, device.clone()),
            grad: zeros(&[8], dtype, device.clone()),
            m: zeros(&[8], dtype, device.clone()),
            v: zeros(&[8], dtype, device),
            lr: scalar_inputs[0].clone(),
            c1: scalar_inputs[1].clone(),
            c2: scalar_inputs[2].clone(),
            beta1: 0.9,
            beta2: 0.99,
            eps: 1e-8,
            weight_decay: 0.01,
        })
        .unwrap()
    }

    #[test]
    fn grouped_adamw_preserves_bucket_order_and_maps_each_semantic_output() {
        let scalar_inputs = [
            zeros(&[], DType::F32, Device::Cpu(0)),
            zeros(&[], DType::F32, Device::Cpu(0)),
            zeros(&[], DType::F32, Device::Cpu(0)),
        ];
        let first = adamw_step(DType::F32, Device::Cpu(0), &scalar_inputs);
        let first_m = Node::new(NodeKind::AdamWOut {
            step: first.clone(),
            index: 1,
        })
        .unwrap();
        let second = adamw_step(DType::F32, Device::Cpu(0), &scalar_inputs);
        let second_v = Node::new(NodeKind::AdamWOut {
            step: second.clone(),
            index: 2,
        })
        .unwrap();
        let roots = [
            first.clone(),
            first_m.clone(),
            second.clone(),
            second_v.clone(),
        ];
        let index = GraphIndex::new(&roots).unwrap();
        let mut options = CompileOptions::default();
        options.environment.optimizer_groups = true;
        let plan = build_optimization_plan(
            &index,
            &options,
            &crate::test_target::TestTarget::for_index(&index),
        )
        .unwrap();
        assert_eq!(plan.regions.len(), 1);
        let NativeRegion::AdamWGroup(region) = &plan.regions[0] else {
            panic!("expected a grouped AdamW region")
        };
        assert_eq!(region.parameter_inputs.len(), 2);
        assert_eq!(region.inputs.len(), 11);
        assert_eq!(region.expressions.len(), 6);
        assert_eq!(
            plan.outputs[dense(&index, &first).index()].unwrap().index,
            0
        );
        assert_eq!(
            plan.outputs[dense(&index, &first_m).index()].unwrap().index,
            1
        );
        assert_eq!(
            plan.outputs[dense(&index, &second).index()].unwrap().index,
            3
        );
        assert_eq!(
            plan.outputs[dense(&index, &second_v).index()]
                .unwrap()
                .index,
            5
        );
        assert_eq!(region.scalar_inputs, scalar_inputs_for_step(&index, &first));
        plan.validate(&index).unwrap();
    }

    #[test]
    fn grouped_adamw_requires_exact_runtime_scalar_ids_on_cpu_and_metal() {
        for device in [Device::Cpu(0), Device::Metal(0)] {
            let shared = [
                zeros(&[], DType::F32, device.clone()),
                zeros(&[], DType::F32, device.clone()),
                zeros(&[], DType::F32, device.clone()),
            ];
            let variants = [
                shared.clone(),
                [
                    zeros(&[], DType::F32, device.clone()),
                    shared[1].clone(),
                    shared[2].clone(),
                ],
                [
                    shared[0].clone(),
                    zeros(&[], DType::F32, device.clone()),
                    shared[2].clone(),
                ],
                [
                    shared[0].clone(),
                    shared[1].clone(),
                    zeros(&[], DType::F32, device.clone()),
                ],
            ];
            let steps = variants
                .iter()
                .map(|scalars| adamw_step(DType::F32, device.clone(), scalars))
                .collect::<Vec<_>>();
            let index = GraphIndex::new(&steps).unwrap();
            let mut options = CompileOptions::default();
            options.environment.optimizer_groups = true;

            let plan = build_optimization_plan(
                &index,
                &options,
                &crate::test_target::TestTarget::for_index(&index),
            )
            .unwrap();

            assert_eq!(plan.regions.len(), variants.len());
            assert!(plan
                .regions
                .iter()
                .all(|region| matches!(region, NativeRegion::AdamW(_))));
            for (step, expected) in steps.iter().zip(&variants) {
                let route = plan.outputs[dense(&index, step).index()].unwrap();
                let NativeRegion::AdamW(region) = &plan.regions[route.region.index()] else {
                    unreachable!()
                };
                assert_eq!(
                    region.scalar_inputs,
                    expected
                        .iter()
                        .map(|scalar| dense(&index, scalar))
                        .collect::<Vec<_>>()
                        .as_slice()
                );
            }
            plan.validate(&index).unwrap();
        }
    }

    fn scalar_inputs_for_step(index: &GraphIndex, step: &Arc<Node>) -> [DenseNodeId; 3] {
        index.children[dense(index, step).index()][4..7]
            .try_into()
            .unwrap()
    }

    #[test]
    fn sgd_region_records_expression_scalar_order_and_selector_routes() {
        let param = zeros(&[8], DType::F64, Device::Cpu(0));
        let grad = zeros(&[8], DType::F64, Device::Cpu(0));
        let velocity = zeros(&[8], DType::F64, Device::Cpu(0));
        let first = zeros(&[], DType::F64, Device::Cpu(0));
        let lr = zeros(&[], DType::F64, Device::Cpu(0));
        let step = Node::new(NodeKind::SgdStep {
            param,
            grad,
            velocity,
            first: first.clone(),
            lr: lr.clone(),
            momentum: 0.9,
            dampening: 0.1,
            nesterov: true,
            weight_decay: 0.01,
        })
        .unwrap();
        let velocity_out = Node::new(NodeKind::SgdOut {
            step: step.clone(),
            index: 1,
        })
        .unwrap();
        let index = GraphIndex::new(&[step.clone(), velocity_out.clone()]).unwrap();
        let plan = build_optimization_plan(
            &index,
            &CompileOptions::default(),
            &crate::test_target::TestTarget::for_index(&index),
        )
        .unwrap();
        let NativeRegion::Sgd(region) = &plan.regions[0] else {
            panic!("expected an SGD region")
        };
        assert_eq!(
            region.scalar_inputs,
            [dense(&index, &lr), dense(&index, &first)]
        );
        assert_eq!(region.expressions.len(), 2);
        assert_eq!(plan.outputs[dense(&index, &step).index()].unwrap().index, 0);
        assert_eq!(
            plan.outputs[dense(&index, &velocity_out).index()]
                .unwrap()
                .index,
            1
        );
        assert_eq!(region.options.momentum, 0.9);
    }

    #[test]
    fn validation_rejects_a_corrupt_output_route() {
        let x = input(0, &[4], DType::F32, Device::Cpu(0));
        let neg = Node::new(NodeKind::Neg { a: x }).unwrap();
        let root = Node::new(NodeKind::Tanh { a: neg }).unwrap();
        let index = GraphIndex::new(std::slice::from_ref(&root)).unwrap();
        let mut plan = build_optimization_plan(
            &index,
            &CompileOptions::default(),
            &crate::test_target::TestTarget::for_index(&index),
        )
        .unwrap();
        plan.outputs[dense(&index, &root).index()]
            .as_mut()
            .unwrap()
            .index = 99;
        assert_eq!(
            plan.validate(&index).unwrap_err(),
            "optimization: ownership or output tables are inconsistent"
        );
    }
}

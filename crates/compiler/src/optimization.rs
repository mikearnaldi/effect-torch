//! Region selection: partitioning a semantic graph into codegen regions and
//! the deterministic lowering order that ties them back together.
//!
//! Selection runs a fixed sequence of passes over the [`GraphIndex`], each
//! reserving nodes for one region kind before the next pass sees them:
//!
//! 1. **GEMM epilogues** (Metal, fusion + epilogues enabled): `Linear` nodes
//!    absorb a following residual `Add` or `Gelu` into a single
//!    [`LinearResidualRegion`]/[`LinearGeluRegion`].
//! 2. **Optimizer steps**: `AdamWStep`/`SgdStep` nodes (with their `*Out`
//!    pickers) become [`AdamWRegion`]/[`SgdRegion`], or — with optimizer
//!    groups enabled — batches of up to four compatible AdamW steps share
//!    one [`AdamWGroupRegion`].
//! 3. **Elementwise fusion**: chains of broadcast-compatible elementwise ops
//!    grow into one [`ElementwiseRegion`] per chain endpoint (bounded by
//!    `MAX_LANES`), and an elementwise chain feeding a reduction becomes an
//!    [`ElementwiseReduceRegion`].
//! 4. **Multi-output merge**: an elementwise region consumed by several
//!    elementwise continuations of one shape merges with them into a
//!    [`MultiOutputRegion`] (bounded by `MAX_BUFFERS` and `MAX_MERGED_OPS`),
//!    with dependency analysis ensuring the merge never swallows a value the
//!    rest of the graph still needs.
//!
//! Cross-cutting invariants, all verified by [`OptimizationPlan::validate`]
//! before a plan is returned:
//!
//! - **Semantic identity is preserved.** Selection never rebuilds a semantic
//!   node (`semantic_nodes_rebuilt == 0`); regions only *cover* nodes, and
//!   every covered node belongs to exactly one region.
//! - **Every value has exactly one source.** A semantic node's value is
//!   either routed from one region output or materialized independently;
//!   internal nodes of a region never escape it.
//! - **The lowering order is a deterministic topological sort** over regions
//!   and independent nodes, so backends never recover topology themselves.
//! - **Determinism.** Identical graphs and options select identical regions;
//!   all tie breaks are by dense ID.

use crate::schedule::{DenseNodeId, GraphIndex};
use crate::{
    adamw_exprs, broadcast_compatible, is_supported, lane_strides, pow_expr, sgd_exprs,
    CompileOptions, KernelExpr, ReduceOp,
};
use effect_torch_graph::{Device, Node, NodeKind};
use effect_torch_runtime::{DType, DenseId};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::fmt;

/// Maximum per-element input lanes one fused expression may reference;
/// bounded by backend kernel buffer limits.
const MAX_LANES: usize = 30;
/// Maximum total buffers (inputs + outputs) of a merged multi-output region.
const MAX_BUFFERS: usize = 31;
/// Maximum expression-node count of a merged multi-output region, bounding
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

/// A deterministic backend lowering unit. The order is already topological;
/// backends do not need to recover topology from semantic `Arc<Node>` values.
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

/// Structural work counters. Region selection never creates semantic nodes,
/// so `semantic_nodes_rebuilt` is always zero and asserted as such.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OptimizationWork {
    pub graph_index_builds: usize,
    pub semantic_nodes_scanned: usize,
    pub semantic_nodes_rebuilt: usize,
    pub fusion_candidates: usize,
    pub multi_output_work_items: usize,
    pub multi_output_dependency_edges: usize,
    pub multi_output_dependency_passes: usize,
    pub multi_output_dependency_edge_visits: usize,
    pub multi_output_dependency_queries: usize,
    pub region_table_merges: usize,
    pub selected_regions: usize,
}

/// One fused expression paired with the semantic node whose value it
/// computes.
#[derive(Debug, Clone, PartialEq)]
pub struct ElementwiseOutput {
    pub semantic_node: DenseNodeId,
    pub expression: KernelExpr,
}

/// A chain of elementwise ops fused into a single kernel over named input
/// lanes: `inputs` are the boundary nodes read per element, `lane_strides`
/// their broadcast strides against `shape`, and `output` the fused
/// expression computing the chain endpoint's value.
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

/// An elementwise chain folded directly into a terminating reduction: the
/// expression is evaluated per input element inside the reduce loop, so the
/// chain's intermediate never materializes.
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
/// `dual` regions additionally materialize the pre-activation value because
/// the graph consumes it elsewhere.
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

/// Which physical output of a fused optimizer step an [`OptimizerOutput`]
/// describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizerOutputKind {
    Parameter,
    FirstMoment,
    SecondMoment,
    Velocity,
}

/// Routing from one physical optimizer output to the semantic nodes that
/// read it: the step itself plus any `AdamWOut`/`SgdOut` pickers, which all
/// alias the same physical buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizerOutput {
    pub index: u32,
    pub parameter: u32,
    pub kind: OptimizerOutputKind,
    /// Semantic producer/selectors which alias this physical output.
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

/// Up to four AdamW steps with identical hyperparameters and shape fused
/// into one kernel, interleaving their lanes to amortize launch and scalar
/// binding costs.
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

/// A selected codegen region. Every variant covers a disjoint set of
/// semantic nodes and routes at least one semantic output; see the module
/// documentation for the selection order and covering invariants.
#[derive(Debug, Clone, PartialEq)]
pub enum NativeRegion {
    Elementwise(ElementwiseRegion),
    ElementwiseReduce(ElementwiseReduceRegion),
    MultiOutput(MultiOutputRegion),
    LinearResidual(LinearResidualRegion),
    LinearGelu(LinearGeluRegion),
    AdamW(AdamWRegion),
    AdamWGroup(AdamWGroupRegion),
    Sgd(SgdRegion),
}

impl NativeRegion {
    /// Semantic nodes covered by this region (sorted, deduplicated).
    pub fn nodes(&self) -> &[DenseNodeId] {
        match self {
            Self::Elementwise(region) => &region.nodes,
            Self::ElementwiseReduce(region) => &region.nodes,
            Self::MultiOutput(region) => &region.nodes,
            Self::LinearResidual(region) => &region.nodes,
            Self::LinearGelu(region) => &region.nodes,
            Self::AdamW(region) => &region.nodes,
            Self::AdamWGroup(region) => &region.nodes,
            Self::Sgd(region) => &region.nodes,
        }
    }

    /// Boundary nodes read by this region; never covered by the region.
    pub fn inputs(&self) -> &[DenseNodeId] {
        match self {
            Self::Elementwise(region) => &region.inputs,
            Self::ElementwiseReduce(region) => &region.inputs,
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
            Self::Elementwise(_) | Self::ElementwiseReduce(_) | Self::LinearResidual(_) => 1,
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
            Self::Elementwise(region) => vec![SemanticOutput {
                semantic_node: region.output.semantic_node,
                index: 0,
            }],
            Self::ElementwiseReduce(region) => vec![SemanticOutput {
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

    /// Deterministic ordering key: the smallest dense index among routed
    /// outputs (falling back to covered nodes), so region order in the plan
    /// reflects semantic postorder.
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

/// Selected implementation regions and complete semantic-output routing.
///
/// The plan's tables are index-parallel to the graph index and mutually
/// consistent (checked by [`OptimizationPlan::validate`]):
///
/// - `node_region[n]` is the region covering node `n`, if any.
/// - `outputs[n]` is the region output implementing node `n`'s value, if it
///   is region-routed; an independent node has neither entry, and a node
///   internal to a region has `node_region` set but no `outputs` entry.
/// - `lowering_order` is a topological order over regions and independent
///   nodes; backends consume it directly.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationPlan<R = NativeRegion> {
    pub regions: Box<[R]>,
    pub node_region: Box<[Option<RegionId>]>,
    pub outputs: Box<[Option<RegionOutput>]>,
    pub lowering_order: Box<[LoweringUnit]>,
    pub work: OptimizationWork,
}

impl OptimizationPlan<NativeRegion> {
    /// Selects native regions for one indexed graph under the given options.
    pub fn select(index: &GraphIndex, options: &CompileOptions) -> Result<Self, String> {
        build_optimization_plan(index, options)
    }

    /// Selects regions from a prepared program's shared index and options.
    pub fn from_prepared(program: &crate::PreparedProgram) -> Result<Self, String> {
        build_optimization_plan(&program.index, &program.options)
    }

    /// Where a semantic node's value comes from: an independent lowering or
    /// a region output. Fails for nodes internal to a region (their values
    /// never materialize) and for out-of-range nodes.
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

    /// Re-derives the ownership tables, output routing, and lowering order
    /// from the region list and checks them against the stored tables; also
    /// requires every graph root to have a materialized value. Selection
    /// runs this before returning, so a stored plan is consistent by
    /// construction — this exists to catch hand-built or mutated plans.
    pub fn validate(&self, index: &GraphIndex) -> Result<(), String> {
        let node_count = index.order.len();
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

/// Convenience wrapper around [`OptimizationPlan::select`].
pub fn select_optimization_regions(
    index: &GraphIndex,
    options: &CompileOptions,
) -> Result<OptimizationPlan, String> {
    build_optimization_plan(index, options)
}

/// Convenience wrapper around [`OptimizationPlan::from_prepared`].
pub fn optimize_prepared_program(
    program: &crate::PreparedProgram,
) -> Result<OptimizationPlan, String> {
    OptimizationPlan::from_prepared(program)
}

/// Runs the selection pipeline described in the module documentation. With
/// `optimize` disabled, returns an empty region set and a lowering order of
/// independent nodes only. Every path ends in plan validation.
pub fn build_optimization_plan(
    index: &GraphIndex,
    options: &CompileOptions,
) -> Result<OptimizationPlan, String> {
    if !options.optimize {
        let mut plan = OptimizationPlan {
            regions: Vec::new().into_boxed_slice(),
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

    let mut selector = RegionSelector::new(index, options);
    if options.environment.fusion && options.environment.gemm_epilogues {
        selector.select_gemm_epilogues();
    }
    selector.select_optimizers();
    if options.environment.fusion {
        selector.select_elementwise()?;
        if options.environment.multi_output_fusion {
            selector.select_multi_output();
        }
    }
    selector.finish()
}

/// A candidate region during selection: drafts may be deactivated by a
/// later merge without being removed, so `active` gates every subsequent
/// pass.
struct DraftRegion {
    region: NativeRegion,
    active: bool,
}

/// Mutable selection state for one graph. `reserved` marks nodes already
/// claimed by a committed region (later passes must leave them alone),
/// `roots` marks graph roots (whose values must always materialize), and
/// `drafts` accumulates candidates in creation order.
struct RegionSelector<'a> {
    index: &'a GraphIndex,
    options: &'a CompileOptions,
    roots: Vec<bool>,
    reserved: Vec<bool>,
    drafts: Vec<DraftRegion>,
    work: OptimizationWork,
}

impl<'a> RegionSelector<'a> {
    fn new(index: &'a GraphIndex, options: &'a CompileOptions) -> Self {
        let mut roots = vec![false; index.order.len()];
        for root in index.roots.iter() {
            roots[root.index()] = true;
        }
        Self {
            index,
            options,
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

    fn add_region(&mut self, region: NativeRegion) -> usize {
        self.work.fusion_candidates += 1;
        let draft = self.drafts.len();
        self.drafts.push(DraftRegion {
            region,
            active: true,
        });
        draft
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
                            || !matches!(residual_node.device, Device::Metal)
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
                            self.reserved[linear.index()] = true;
                        }
                        nodes.sort_unstable();
                        let mut inputs = linear_inputs.to_vec();
                        inputs.push(residual);
                        self.reserved[dense_index] = true;
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
                    self.reserved[linear.index()] = true;
                    self.reserved[dense_index] = true;
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
        if !matches!(node.device, Device::Metal) || !matches!(node.dtype, DType::F32 | DType::BF16)
        {
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
            if self.reserved[dense_index] || !is_supported(&node.device, node.dtype) {
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
                            grouped[step.index()] = true;
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
        for node in &nodes {
            self.reserved[node.index()] = true;
        }
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
        for node in &nodes {
            self.reserved[node.index()] = true;
        }
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
        for node in &nodes {
            self.reserved[node.index()] = true;
        }
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

    /// Single forward pass over the dense postorder growing open elementwise
    /// chains. A chain is closed (and emitted when profitable) at any node
    /// whose consumer is not a fusible elementwise op, whose value is read
    /// by multiple consumers, or which is a graph root; reductions either
    /// absorb their open input chain into a fused-reduce region or close it
    /// first.
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
                    region.expression = expression;
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
                            (left, operation.apply(left_expression, right_expression))
                        }
                        (Some(mut region), None) => {
                            let right = self.element_operand(
                                &mut region,
                                b,
                                &self.index.order[dense_index].shape,
                            );
                            let left =
                                std::mem::replace(&mut region.expression, KernelExpr::cst(0.0));
                            (region, operation.apply(left, right))
                        }
                        (None, Some(mut region)) => {
                            let left = self.element_operand(
                                &mut region,
                                a,
                                &self.index.order[dense_index].shape,
                            );
                            let right =
                                std::mem::replace(&mut region.expression, KernelExpr::cst(0.0));
                            (region, operation.apply(left, right))
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
                            (region, operation.apply(left, right))
                        }
                    };
                    region.expression = expression;
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
                            expressions.push(KernelExpr::cst(value));
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
                            .apply(expressions.next().unwrap(), expressions.next().unwrap());
                        region.expression = KernelExpr::Select(
                            Box::new(condition),
                            Box::new(expressions.next().unwrap()),
                            Box::new(expressions.next().unwrap()),
                        );
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
                Some(ElementOperation::Reduce(op, mut dims, keepdims)) => {
                    let input = children[0];
                    let input_shape = self.index.order[input.index()].shape.clone();
                    dims.sort_unstable();
                    dims.dedup();
                    let rank = input_shape.len();
                    let output_shape = reduced_shape(&input_shape, &dims, keepdims);
                    let guards_ok = !dims.is_empty()
                        && dims.iter().all(|&dim| dim < rank)
                        && dims.iter().map(|&dim| input_shape[dim]).product::<usize>() > 0
                        && !(matches!(self.index.order[dense_index].device, Device::Metal)
                            && (input_shape.iter().product::<usize>() > i32::MAX as usize
                                || output_shape.iter().product::<usize>() > i32::MAX as usize));
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

    fn element_operand(
        &self,
        region: &mut OpenRegion,
        child: DenseNodeId,
        output_shape: &[usize],
    ) -> KernelExpr {
        self.const_value(child, output_shape)
            .map(KernelExpr::cst)
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
        if !is_supported(&node.device, node.dtype) {
            return None;
        }
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
            NodeKind::Cast { a, dtype } if a.dtype == *dtype => {
                Some(ElementOperation::Unary(UnaryOperation::Identity))
            }
            NodeKind::Where { cond, a, b }
                if self.index.consumers[self.dense(cond).index()].len() == 1
                    && input_ok(a)
                    && input_ok(b) =>
            {
                let comparison = match &cond.kind {
                    NodeKind::Eq { a, b }
                        if input_ok(a) && input_ok(b) && is_supported(&a.device, a.dtype) =>
                    {
                        Some(ComparisonOperation::Eq)
                    }
                    NodeKind::Gt { a, b }
                        if input_ok(a) && input_ok(b) && is_supported(&a.device, a.dtype) =>
                    {
                        Some(ComparisonOperation::Gt)
                    }
                    NodeKind::Lt { a, b }
                        if input_ok(a) && input_ok(b) && is_supported(&a.device, a.dtype) =>
                    {
                        Some(ComparisonOperation::Lt)
                    }
                    NodeKind::Ge { a, b }
                        if input_ok(a) && input_ok(b) && is_supported(&a.device, a.dtype) =>
                    {
                        Some(ComparisonOperation::Ge)
                    }
                    NodeKind::Le { a, b }
                        if input_ok(a) && input_ok(b) && is_supported(&a.device, a.dtype) =>
                    {
                        Some(ComparisonOperation::Le)
                    }
                    _ => None,
                };
                comparison.map(ElementOperation::Select)
            }
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

    /// Emits a closed chain as an [`ElementwiseRegion`] when it is
    /// profitable (at least two fused ops over at least one real input lane)
    /// and representable (broadcast-compatible strides, element count within
    /// backend index range). Unprofitable chains are dropped silently: their
    /// nodes simply lower independently.
    fn emit_elementwise(
        &mut self,
        endpoint: DenseNodeId,
        mut region: OpenRegion,
    ) -> Result<(), String> {
        let node = &self.index.order[endpoint.index()];
        let element_count = node.shape.iter().product::<usize>();
        let strides = region
            .inputs
            .iter()
            .map(|input| {
                lane_strides(&self.index.order[input.index()].shape, &node.shape)
                    .map(Vec::into_boxed_slice)
            })
            .collect::<Option<Vec<_>>>();
        if region.ops >= 2
            && !region.inputs.is_empty()
            && !(matches!(node.device, Device::Metal) && element_count > i32::MAX as usize)
        {
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

    /// Merges elementwise regions that share a common prefix region into
    /// multi-output regions. A prefix qualifies when its output fans out to
    /// at least two elementwise continuations of one shape; the merge is
    /// skipped when a continuation's extra inputs depend on the prefix's
    /// other descendants (dependency analysis via `RegionDependencyIndex`),
    /// when the prefix must stay materialized for external consumers but has
    /// a different shape, or when buffer/op limits would be exceeded. On a
    /// "split" merge the prefix region is preserved as a separate draft so
    /// external consumers keep their value.
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
                // The split may consume a prefix-dependent extra lane, but it
                // must not consume any continuation it would also cover.
                if split && dependencies.any_draft_ancestor(&group_set, &extra_inputs) {
                    continue;
                }
                if let Some(region) =
                    self.merge_multi_region(prefix, &group, &shape, keep_prefix, split)
                {
                    if self.options.environment.fusion_debug {
                        eprintln!(
                            "[fusion] multi-merge: prefix {prefix_output} -> {} continuations (keep {keep_prefix}, split {split})",
                            group.len(),
                        );
                    }
                    let selected = if split {
                        let selected = self.drafts.len();
                        self.drafts.push(DraftRegion {
                            region: NativeRegion::MultiOutput(region),
                            active: true,
                        });
                        selected
                    } else {
                        self.drafts[prefix].region = NativeRegion::MultiOutput(region);
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

    /// Constructs the merged multi-output region for one prefix and its
    /// continuation group: prefix lanes are re-based into the output shape,
    /// continuation expressions inline the prefix expression for their
    /// shared lane, and the merge is refused when dtype/device disagree,
    /// buffer or op limits would be exceeded, or the output is beyond
    /// backend index range.
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
        if matches!(prefix_region.device, Device::Metal)
            && output_shape.iter().product::<usize>() > i32::MAX as usize
        {
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

    /// Finalizes selection: active drafts are sorted by their semantic
    /// ordering key, ownership and routing tables are built with overlap and
    /// double-routing rejected, and the lowering order is computed and
    /// validated before the plan is returned.
    fn finish(mut self) -> Result<OptimizationPlan, String> {
        let mut active = self
            .drafts
            .into_iter()
            .enumerate()
            .filter_map(|(draft, region)| {
                region
                    .active
                    .then_some((region.region.ordering_key(), draft, region.region))
            })
            .collect::<Vec<_>>();
        active.sort_by_key(|(key, draft, _)| (*key, *draft));
        let regions = active
            .into_iter()
            .map(|(_, _, region)| region)
            .collect::<Vec<_>>();
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

/// Dependency index over lowering units (draft regions plus independent
/// nodes) used by the multi-output merge to answer two reachability queries
/// without rebuilding adjacency per query: "does this input descend from a
/// marked region" (forward, over consumers) and "does any of these inputs
/// have a member of the merge group as an ancestor" (backward, over
/// dependencies). Stamp generations (`seen`/`descendants`) replace per-pass
/// visited sets; counters feed `OptimizationWork`.
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

/// An elementwise chain under construction: the fused expression so far,
/// the boundary nodes it reads (`inputs` with `lane_of` assigning each a
/// stable lane index), the semantic nodes it covers, and the fused op count
/// used by the `ops >= 2` emission threshold.
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

    /// The lane expression for `node`, allocating a fresh input lane on
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

    /// Merges another open region into this one: its inputs are appended
    /// (deduplicated through `lane_of`), its covered nodes and op count are
    /// transferred, and its expression is returned with lanes remapped into
    /// the merged namespace.
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
    Identity,
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
            Self::Identity => input,
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

/// Builds the plan's lowering order: one unit per region plus one per
/// independent node, topologically sorted by dependency. Ready units pop in
/// deterministic priority order (region ordering key before node index), so
/// the result is stable for identical plans. A leftover indegree means a
/// dependency cycle, which is reported with the first blocked units.
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
        let x = input(0, &[2, 3], DType::F32, Device::Cpu);
        let y = input(1, &[2, 3], DType::F32, Device::Cpu);
        let z = input(2, &[3], DType::F32, Device::Cpu);
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
        let x = input(0, &[4], DType::F32, Device::Cpu);
        let y = input(1, &[4], DType::F32, Device::Cpu);
        let sum = Node::new(NodeKind::Add { a: x, b: y }).unwrap();
        let root = Node::new(NodeKind::Tanh { a: sum }).unwrap();
        let index = GraphIndex::new(std::slice::from_ref(&root)).unwrap();
        let identities = index
            .order
            .iter()
            .map(|node| (node.id, Arc::as_ptr(node) as usize))
            .collect::<Vec<_>>();

        let plan = build_optimization_plan(&index, &CompileOptions::default()).unwrap();

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
        let x = input(0, &[4], DType::F32, Device::Cpu);
        let neg = Node::new(NodeKind::Neg { a: x }).unwrap();
        let root = Node::new(NodeKind::Tanh { a: neg }).unwrap();
        let index = GraphIndex::new(&[root.clone(), root.clone()]).unwrap();
        let options = CompileOptions {
            optimize: false,
            ..CompileOptions::default()
        };
        let plan = build_optimization_plan(&index, &options).unwrap();
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
        let x = input(0, &[4], DType::F32, Device::Cpu);
        let neg = Node::new(NodeKind::Neg { a: x }).unwrap();
        let root = Node::new(NodeKind::Tanh { a: neg }).unwrap();
        let index = GraphIndex::new(&[root.clone(), root.clone()]).unwrap();
        let plan = build_optimization_plan(&index, &CompileOptions::default()).unwrap();
        let route = plan.outputs[dense(&index, &root).index()].unwrap();
        assert_eq!(route.index, 0);
        assert_eq!(index.roots[0], index.roots[1]);
        assert_eq!(plan.resolve(index.roots[0]), plan.resolve(index.roots[1]));
    }

    #[test]
    fn reduction_region_records_normalized_geometry_and_expression_inputs() {
        let x = input(0, &[2, 3], DType::F64, Device::Cpu);
        let y = input(1, &[3], DType::F64, Device::Cpu);
        let add = Node::new(NodeKind::Add { a: x, b: y }).unwrap();
        let root = Node::new(NodeKind::Mean {
            a: add.clone(),
            dims: vec![1, 1],
            keepdims: true,
        })
        .unwrap();
        let index = GraphIndex::new(std::slice::from_ref(&root)).unwrap();
        let plan = build_optimization_plan(&index, &CompileOptions::default()).unwrap();
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
        let x = input(0, &[4], DType::F32, Device::Cpu);
        let y = input(1, &[4], DType::F32, Device::Cpu);
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
                device: Device::Cpu,
            })
            .unwrap(),
            b: zeros(&[4], DType::F32, Device::Cpu),
        })
        .unwrap();
        let root = Node::new(NodeKind::Tanh { a: selected }).unwrap();
        let index = GraphIndex::new(std::slice::from_ref(&root)).unwrap();
        let plan = build_optimization_plan(&index, &CompileOptions::default()).unwrap();
        let NativeRegion::Elementwise(region) = &plan.regions[0] else {
            panic!("expected an elementwise select region")
        };
        let KernelExpr::Tanh(expression) = &region.output.expression else {
            panic!("expected the continuation after select")
        };
        assert!(matches!(expression.as_ref(), KernelExpr::Select(..)));
        assert!(region.nodes.contains(&dense(&index, &condition)));
        assert_eq!(
            region.inputs.as_ref(),
            [dense(&index, &x), dense(&index, &y)]
        );
    }

    #[test]
    fn metal_linear_residual_has_exact_coverage_inputs_and_output() {
        let x = input(0, &[2, 3], DType::BF16, Device::Metal);
        let weight = input(1, &[3, 4], DType::BF16, Device::Metal);
        let bias = input(2, &[4], DType::BF16, Device::Metal);
        let residual = input(3, &[2, 4], DType::BF16, Device::Metal);
        let linear = Node::new(NodeKind::Linear { x, weight, bias }).unwrap();
        let root = Node::new(NodeKind::Add {
            a: linear.clone(),
            b: residual.clone(),
        })
        .unwrap();
        let index = GraphIndex::new(std::slice::from_ref(&root)).unwrap();
        let plan = build_optimization_plan(&index, &CompileOptions::default()).unwrap();
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
        let x = input(0, &[2, 3], DType::F32, Device::Metal);
        let weight = input(1, &[3, 4], DType::F32, Device::Metal);
        let bias = input(2, &[4], DType::F32, Device::Metal);
        let linear = Node::new(NodeKind::Linear { x, weight, bias }).unwrap();
        let gelu = Node::new(NodeKind::Gelu {
            a: linear.clone(),
            approximate: true,
        })
        .unwrap();
        let other = Node::new(NodeKind::Neg { a: linear.clone() }).unwrap();
        let index = GraphIndex::new(&[gelu.clone(), other]).unwrap();
        let plan = build_optimization_plan(&index, &CompileOptions::default()).unwrap();
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
        let plan = build_optimization_plan(&index, &CompileOptions::default()).unwrap();
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
        let plan = build_optimization_plan(&index, &CompileOptions::default()).unwrap();
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
        let x = input(0, &[2, 3], DType::F32, Device::Cpu);
        let y = input(1, &[2, 3], DType::F32, Device::Cpu);
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

        let plan = build_optimization_plan(&index, &CompileOptions::default()).unwrap();

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
        let x = input(0, &[2, 3], DType::F32, Device::Cpu);
        let y = input(1, &[2, 3], DType::F32, Device::Cpu);
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

        let plan = build_optimization_plan(&index, &CompileOptions::default()).unwrap();

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
        let x = input(0, &[8], DType::F32, Device::Cpu);
        let y = input(1, &[8], DType::F32, Device::Cpu);
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

        let plan = build_optimization_plan(&index, &CompileOptions::default()).unwrap();

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
        let x = input(0, &[2, 3], DType::F32, Device::Cpu);
        let y = input(1, &[2, 3], DType::F32, Device::Cpu);
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

        let plan = build_optimization_plan(&index, &CompileOptions::default()).unwrap();

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
        let x = input(0, &[4, 4], DType::F32, Device::Cpu);
        let weight = input(1, &[4, 4], DType::F32, Device::Cpu);
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

        let plan = build_optimization_plan(&index, &CompileOptions::default()).unwrap();

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
        let first = build_optimization_plan(&index, &CompileOptions::default()).unwrap();
        let second = build_optimization_plan(&index, &CompileOptions::default()).unwrap();
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
            zeros(&[], DType::F32, Device::Cpu),
            zeros(&[], DType::F32, Device::Cpu),
            zeros(&[], DType::F32, Device::Cpu),
        ];
        let first = adamw_step(DType::F32, Device::Cpu, &scalar_inputs);
        let first_m = Node::new(NodeKind::AdamWOut {
            step: first.clone(),
            index: 1,
        })
        .unwrap();
        let second = adamw_step(DType::F32, Device::Cpu, &scalar_inputs);
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
        let plan = build_optimization_plan(&index, &options).unwrap();
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
        for device in [Device::Cpu, Device::Metal] {
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

            let plan = build_optimization_plan(&index, &options).unwrap();

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
        let param = zeros(&[8], DType::F64, Device::Cpu);
        let grad = zeros(&[8], DType::F64, Device::Cpu);
        let velocity = zeros(&[8], DType::F64, Device::Cpu);
        let first = zeros(&[], DType::F64, Device::Cpu);
        let lr = zeros(&[], DType::F64, Device::Cpu);
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
        let plan = build_optimization_plan(&index, &CompileOptions::default()).unwrap();
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
        let x = input(0, &[4], DType::F32, Device::Cpu);
        let neg = Node::new(NodeKind::Neg { a: x }).unwrap();
        let root = Node::new(NodeKind::Tanh { a: neg }).unwrap();
        let index = GraphIndex::new(std::slice::from_ref(&root)).unwrap();
        let mut plan = build_optimization_plan(&index, &CompileOptions::default()).unwrap();
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

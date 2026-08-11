# RFC 0021: Compiler Pipeline Refactor - Shared Analysis, Region Fusion, and Authoritative Lowered IR

- **Status**: Implemented
- **Created**: 2026-08-10
- **Implemented**: 2026-08-11
- **Depends on**: RFC 0007 (kernel fusion), RFC 0017 (multi-backend
  runtime), RFC 0019 (executable compilation), RFC 0020 (invocation
  ownership)
- **Updates**: RFC 0007 implementation architecture and RFC 0019 compiler
  pipeline architecture

## Summary

Preserve every benchmark-proven optimization and the existing executable
runtime while replacing the compiler's repeated immutable graph rewrites with
one indexed analysis, an ephemeral optimization plan, and one authoritative
backend-lowered program.

The pre-refactor implementation had the right execution model and performance
features, but its compile-time data flow was wider than necessary:

```text
semantic NodeKind graph
    -> rewritten Fused* NodeKind graph
    -> CpuOp / MetalOp
    -> CpuCommand / MetalCommand
    -> string-valued LoweredInstruction projection
    -> MemoryPlan
```

Several optimization stages independently recover topological order and
consumer information. Most graph rewrites allocate a replacement `Node` for
every reachable node, including unchanged operations. Multi-output fusion
applies one merge, rebuilds the graph, and starts another whole-graph fixpoint
iteration. Backend compilation then constructs its real commands before
constructing a second, string-valued schedule used only for liveness and memory
planning.

This RFC changes that flow to:

```text
ProgramRequest
    -> PreparedProgram: semantic roots + one GraphIndex
    -> OptimizationPlan: selected regions and output mappings
    -> LoweredProgram<BackendInstruction, MemorySpace>
    -> MemoryPlan and physical synchronization plan
    -> ExecutablePlan
```

`GraphIndex` and `OptimizationPlan` are ephemeral compiler analyses, not
additional executable IRs. The semantic graph remains the source program.
`LoweredProgram` becomes the sole authoritative tensor instruction schedule.
The memory planner, diagnostics, pipeline preparation, and executor all consume
that same schedule. A final physical command stream may add barriers, commits,
events, or status gates, but it references lowered instruction IDs and does not
duplicate tensor-operation semantics.

Fusion remains a first-class compiler optimization. The scalar expression IR
is retained and renamed `KernelExpr` to make its scope explicit: it describes
the per-element body of one fused instruction, not a whole tensor program.
Elementwise, reduction, multi-output, GEMM-epilogue, and optimizer fusion are
selected as regions over the indexed semantic graph. They no longer require
internal `FusedElementwise`, `FusedElementwiseMulti`, `FusedPick`, or
`FusedReduce` nodes to be inserted into a replacement semantic graph.

The refactor was accepted only after preserving numerical behavior, kernel and
command quality, memory plans, randomness, state transactions, concurrency,
and the optimization topology that motivated the existing implementation.

Motivation and migration passages below intentionally retain descriptions of
the pre-implementation architecture as historical context. The implementation
record before the acceptance criteria is authoritative for the as-built state.

## Decision

Effect Torch adopts these compiler boundaries:

| Concern | Authoritative representation | Lifetime |
|---|---|---|
| User and autodiff semantics | Immutable semantic graph | Tensor/program lifetime |
| Topology, consumers, slots, leaves, provenance | `GraphIndex` side tables | One prepared compile request |
| Fusion and epilogue choices | `OptimizationPlan` regions | Compilation only |
| Values, instructions, resources, and effects | `LoweredProgram<I, M>` | Executable-plan lifetime |
| Segment assignment and aliases | `MemoryPlan<M>` | Executable-plan lifetime |
| Backend barriers, commits, and events | Physical plan referencing instruction IDs | Executable-plan lifetime |
| Pipelines and immutable kernel artifacts | Backend executable plan | Executable-plan lifetime |
| Workspace and output backing | Invocation/tensor ownership from RFC 0020 | Invocation/handle lifetime |

The following rules are normative:

1. Code-generation optimizations do not rebuild the semantic graph.
2. The compiler driver constructs one `GraphIndex` for each semantic graph
   generation it receives. Passes do not call `graph_post_order` independently.
3. Inference specialization may produce one new semantic graph when it changes
   the program's state contract. The specialized graph is indexed once before
   code-generation optimization.
4. Fusion passes record regions and semantic-output mappings in side tables.
   They do not emit `NodeKind::Fused*` nodes.
5. Multi-output fusion selects compatible non-conflicting regions in batches or
   updates an indexed worklist locally. It does not perform one whole-graph
   rebuild per merge.
6. Backend lowering directly produces typed lowered instructions with complete
   value access, output, scratch, staging, status, state, and algorithm plans.
7. Memory planning consumes those typed instructions. Production lowering does
   not create a string-valued shadow schedule.
8. Execution consumes the same typed instruction identities, optionally through
   a physical command plan that only adds backend synchronization mechanics.
9. Existing benchmark-proven optimization decisions remain enabled under the
   same conditions unless a replacement is measured to be equal or better.
10. Unused generic compiler/runtime shells are either connected to this path or
    removed. Parallel abstractions without a production consumer are not kept.

## Motivation

### The optimizations are not the problem

Kernel fusion and semantic kernels were introduced from end-to-end measurement,
not from an assumption that compiler sophistication is inherently useful. RFC
0007 records, among other results:

- fusion formerly cost about 5 microseconds per graph node when repeated per
  walk, enough to turn the optimization into a regression before memoization;
- moving synchronization to the walk boundary reduced a measured 209-root walk
  from 32 ms to 11.6 ms;
- native cross-entropy reduced one measured path from 1.1 ms to 12 microseconds;
- the reference compiled GPT training step moved from 53 ms to 12.5 ms.

The implementation also keeps optimizer grouping disabled by default because
measurement showed the larger grouped kernel losing to individual fused steps
for the tested GPT-scale parameter shapes. This is the desired decision model:
optimization policy follows evidence.

This RFC does not remove, weaken, or re-litigate those execution-time wins. It
addresses how the compiler represents and selects them.

### A normal optimized cache miss performs repeated graph walks

With default optimization and a usable structural executable-cache key, a
native cache miss currently performs approximately these whole-graph walks:

| Stage | Purpose | Rebuilds semantic nodes |
|---|---|---|
| Native cache preparation | Collect generated leaves before lookup | No |
| Backend compile entry | Count semantic nodes and prepare random-source metadata | No |
| GEMM epilogue pass | Consumer analysis and epilogue rewrite | Yes |
| Elementwise/reduction pass | Consumer analysis and region rewrite | Yes |
| Multi-output analysis | Consumer graph and merge search | On every successful merge |
| Program-slot collection | Validate tensor/scalar placeholders | No |
| Backend lowering order | Traverse optimized roots | No |
| Native cache insertion | Collect and reorder generated leaves again | No |

Optimizer grouping adds another traversal when enabled. Decode specialization
performs additional graph work before this sequence.

Several linear scans are not inherently wrong. Compiler passes often require
ordered, separate analyses. The avoidable costs are:

- rediscovering the same topological and consumer information through pointer
  traversal;
- reconstructing every unchanged semantic node after a local rewrite;
- assigning new node identities to unchanged semantics;
- rebuilding a full graph generation to communicate a code-generation choice;
- rescanning roots for slots, leaves, and lowering after that information was
  already available;
- constructing backend commands and then projecting them into a second schedule
  for planning.

### Immutable graph rewriting amplifies local changes

The graph stores child relationships as `Arc<Node>` values inside `NodeKind`.
When one child changes, its consumer must be reconstructed to refer to the new
child. That reconstruction propagates to roots. Current passes therefore use a
bottom-up map and call `Node::new` for unchanged operations as well as matched
ones.

This is appropriate for semantic graph transforms such as autodiff, vmap, or a
stateful inference specialization: those transforms define a new requested
computation. It is unnecessary for code-generation decisions that leave tensor
semantics unchanged. Choosing to implement `mul -> add -> tanh` with one kernel
does not require a new semantic graph.

Graph reconstruction also changes node IDs. CPU compilation consequently maps
optimized random nodes back to pre-optimization random nodes to preserve
semantic random provenance. Side-table optimization over stable semantic node
IDs removes that source of identity churn.

### Multi-output fusion has a whole-graph fixpoint

The current multi-output pass:

```text
analyze the complete rewritten graph
find one merge plan
rebuild the complete graph with that merge
repeat until no plan remains
```

For `N` nodes and `M` independently applicable merge steps, this can approach
`O(N * M)` graph work and graph allocation. The final unsuccessful search adds
another full analysis. The runtime value of multi-output fusion does not require
this compile-time algorithm. Candidate regions and their consumers can be
represented explicitly, conflict-checked, and selected deterministically in one
analysis plus local worklist updates.

### The planning schedule is not the executable schedule

`LoweredSchedule<K, M>` is documented as the dense compiler IR consumed by
liveness and backend command planning. In production CPU and Metal compilation,
however, backend commands are built first. The compiler then creates
`LoweredInstruction<String>` entries containing names such as
`"fused_elementwise"` or `"matmul"` and enough value uses to run the memory
planner. That schedule is discarded after planning. Execution retains the
separate `CpuCommand` or `MetalCommand` array.

This has several consequences:

- resource access facts are copied from commands into another representation;
- diagnostics describe a projection rather than the executable instructions;
- planner and executor can drift in how scratch or state effects are modeled;
- the generic type parameter does not carry the backend operation in production;
- adding a resource requires updating both command construction and projection;
- `MemoryPlan.reuse_edges` are computed separately from the physical schedule
  that must enforce safe reuse.

The lowered schedule should contain the real backend instruction kinds and
their exact resource contracts.

### The compiler driver is split across backend crates

The shared compiler crate owns graph ordering, fusion rewrites, expression IR,
liveness, memory packing, and diagnostics. CPU and Metal each own a separate
driver that invokes those pieces, maps nodes, prepares algorithm requirements,
constructs a planning schedule, and assembles an executable.

The runtime crate also defines `ProgramSignature`, `Invocation`,
`NativeExecutable`, and `ExecutableBackend`, while the production native
backends use bespoke executable and invocation types. The compiler crate defines
`ProgramRequest`, but native compilation does not use it as its actual boundary.

The resulting crate names imply a cleaner separation than the production call
graph currently provides. This RFC makes the production path explicit and
removes shells that remain unused after migration.

## Goals

1. Preserve every benchmark-proven fusion, semantic-kernel, scheduling, and
   memory optimization.
2. Preserve optimized and unoptimized numerical behavior, including documented
   tolerance and reduction-order rules.
3. Build topological, consumer, slot, leaf, and provenance analysis once per
   semantic graph generation.
4. Represent code-generation optimization as regions and side tables rather
   than replacement semantic nodes.
5. Eliminate whole-graph one-merge fixpoint reconstruction.
6. Make one typed lowered program authoritative for liveness, diagnostics,
   pipeline preparation, physical planning, and execution.
7. Make the common compiler driver own pass order and analysis reuse while
   preserving backend-specific instruction and algorithm selection.
8. Keep compile output deterministic for identical requests and environment
   snapshots.
9. Reduce first-compile wall time, host allocation, and peak compiler memory for
   large forward/backward/update graphs.
10. Preserve RFC 0019 static memory planning and RFC 0020 invocation ownership.
11. Make the roles of semantic graph, `KernelExpr`, lowered program, memory
    plan, and physical synchronization plan unambiguous.
12. Keep the public TypeScript `Runtime.compile` and `Runtime.execute` contract
    unchanged.

## Non-goals

- Remove or disable a benchmark-proven optimization.
- Replace first-party CPU or Metal kernels with a third-party compiler.
- Introduce symbolic dimensions or shape-polymorphic executables.
- Move graph construction, autodiff, fusion, or lowering into TypeScript.
- Change optimizer mathematics, random semantics, dtype policy, or reduction
  numerics.
- Change output lifetime, workspace leasing, Metal submission ownership, or
  sequence-state transaction semantics.
- Create a portable serialized executable format.
- Require CPU and Metal to use the same instruction enum or kernel plans.
- Introduce a dynamic Rust plugin ABI.
- Guarantee that compilation performs only one linear scan. Distinct passes may
  scan indexed arrays when their ordering and responsibilities are clear.
- Optimize compile time by accepting worse executable performance.

## Terminology

| Term | Meaning |
|---|---|
| Semantic graph | Immutable typed tensor-operation DAG constructed by the frontend and semantic transforms |
| Semantic transform | A transform that changes the requested mathematical/stateful program, such as autodiff, vmap, or decode specialization |
| Code-generation optimization | A semantics-preserving choice of implementation regions, kernels, epilogues, or grouped instructions |
| `GraphIndex` | Ephemeral dense analysis of one immutable semantic graph generation |
| Region | A selected set of semantic nodes implemented by one lowered instruction or one fixed instruction group |
| `OptimizationPlan` | Ephemeral deterministic mapping from semantic nodes and outputs to selected regions |
| `KernelExpr` | Nested scalar expression IR inside an elementwise or fused-reduction instruction |
| Lowered value | Dense backend value with shape, dtype, layout, storage class, and ownership requirements |
| Lowered instruction | Typed backend operation with exact resource accesses, outputs, algorithm plan, and effects |
| `LoweredProgram<I, M>` | Authoritative logical value and instruction schedule for one backend |
| Physical plan | Backend barriers, command-buffer boundaries, events, and status gates referencing lowered instructions |
| Executable plan | Immutable lowered program, memory plan, physical plan, pipelines, signature, and diagnostics |

An ephemeral analysis is not an IR merely because it is structured data. It
does not define execution semantics, survive compilation, or become an input to
the runtime independently of the semantic or lowered program.

## Semantic and Performance Invariants

### Optimized and unoptimized paths share semantics

`optimize: false` continues to lower every supported semantic operation through
the same backend instruction and execution framework. It does not select an
interpreter or a separate allocation path.

For every supported graph and invocation:

```text
execute(compile(graph, optimize = true), invocation)
```

must match:

```text
execute(compile(graph, optimize = false), invocation)
```

under each operation's existing exact or tolerance-based numerical contract.

### Proven fusion boundaries are preserved

The migration records old and new compiler diagnostics for the benchmark and
parity corpus. Unless an intentional replacement is separately measured and
approved, the new compiler preserves:

- the number and shape of fused elementwise regions;
- fused-reduction coverage;
- multi-output region coverage and materialized-prefix behavior;
- GEMM residual and GELU epilogues;
- AdamW and SGD fused implementation selection;
- the default-disabled policy for optimizer grouping on measured losing shapes;
- semantic kernel selection for attention, KDA, convolution, layer norm,
  cross-entropy, rotary operations, and chunked head loss;
- pipeline cache keys or equivalent kernel-reuse behavior;
- command count and synchronization behavior relevant to measured throughput;
- planned workspace and escaping-output classes.

Changes to region boundaries are permitted only when correctness passes and the
relevant benchmark is equal or better under the project's established
measurement procedure.

### Random behavior is preserved

This refactor does not change either backend's seed-derivation ABI. Preparation
records random-source order and the provenance metadata needed by lowering.
CPU's semantic-source provenance and Metal's current lowered-command-derived
seed token remain stable through migration unless a separate randomness change
is measured, specified, and approved.

Code-generation optimization cannot merge distinct random sources. A shared
random node still produces one shared draw across roots, and distinct random
nodes remain distinct even when they have identical shape and attributes.
Concurrent invocation nonce behavior remains defined by RFCs 0019 and 0020.

### Stateful specialization remains transactional

Decode specialization changes the semantic program because causal attention,
KDA, and short convolution acquire explicit state interactions. It remains a
semantic transform before `GraphIndex` construction. The resulting lowered
program declares state reads, staging, transaction outputs, status gates, and
commit points exactly as today.

The refactor does not move state mutation into optimization side tables or make
state effects implicit in kernel expressions.

### Compile-time behavior is measurable

Compiler diagnostics add internal test-only or debug-visible metrics:

```rust
pub struct CompilerWorkReport {
    pub semantic_nodes: usize,
    pub graph_index_builds: usize,
    pub graph_edge_visits: usize,
    pub semantic_nodes_rebuilt: usize,
    pub pass_scans: Box<[PassScanCount]>,
    pub fusion_candidates: usize,
    pub selected_regions: usize,
    pub lowered_values: usize,
    pub lowered_instructions: usize,
    pub compile_phases: Box<[CompilePhaseTiming]>,
}
```

These counters are not a stable public API. They exist to prevent a later
refactor from silently restoring per-pass pointer-graph traversals or
whole-graph reconstruction.

## Architecture

### End-to-end pipeline

```text
TypeScript graph construction
        |
        | native lazy roots
        v
ProgramRequest
        |
        | optional semantic specialization
        | (decode state, other explicitly authorized semantics)
        v
PreparedProgram
  - semantic roots
  - GraphIndex
  - binding and invocation contract
  - compile options and environment snapshot
        |
        | code-generation optimization
        v
OptimizationPlan
  - regions
  - covered semantic nodes
  - semantic output -> region output mapping
        |
        | backend lowering and algorithm selection
        v
LoweredProgram<BackendInstruction, MemorySpace>
        |
        | liveness and segment assignment
        v
MemoryPlan<MemorySpace>
        |
        | backend physical hazard planning and pipeline preparation
        v
ExecutablePlan
        |
        | RFC 0020 invocation frame and storage leases
        v
Execute
```

Autodiff, vmap, and checkpoint semantics remain graph construction or semantic
transforms before `ProgramRequest`. Training continues to submit forward, loss,
backward, and optimizer-update roots to the same compiler.

### Program request and preparation

`ProgramRequest` becomes the actual internal native compile boundary rather
than an unused model:

```rust
pub struct ProgramRequest {
    pub roots: Vec<Arc<ProgramNode>>,
    pub bindings: Vec<BindingDecl>,
    pub invocation: InvocationSignature,
    pub options: CompileOptions,
    pub state: Option<StateRequest>,
}

pub struct PreparedProgram {
    pub roots: Box<[Arc<ProgramNode>]>,
    pub index: Arc<GraphIndex>,
    pub bindings: Box<[BindingDecl]>,
    pub invocation: InvocationSignature,
    pub options: CompileOptions,
    pub state: Option<StateSchema>,
}
```

Exact type placement may differ to avoid crate dependency cycles, but there is
one production request and one preparation step.

Preparation performs:

- root and device validation;
- optional semantic inference specialization;
- one deterministic stack-safe postorder;
- dense node indexing;
- child and consumer adjacency construction;
- placeholder slot validation;
- generated-leaf collection and deterministic binding order;
- random-source collection and backend provenance/seed metadata;
- source-node diagnostics;
- immutable environment snapshot validation.

Native executable cache lookup consumes `PreparedProgram`. A cache miss passes
the same prepared object into compilation. A cache insertion uses its generated
binding order. The NAPI layer does not walk roots again after compilation.

### Graph index

An illustrative index is:

```rust
pub struct GraphIndex {
    pub order: Box<[Arc<ProgramNode>]>,
    pub dense_by_node: HashMap<NodeId, DenseNodeId>,
    pub children: Box<[Box<[DenseNodeId]>]>,
    pub consumers: Box<[Box<[DenseNodeId]>]>,
    pub roots: Box<[DenseNodeId]>,
    pub slots: Box<[ProgramSlot]>,
    pub leaves: Box<[GeneratedBinding]>,
    pub random_sources: Box<[Option<RandomSourceId>]>,
}
```

Passes iterate dense node IDs and indexed adjacency. They do not recursively
recover children from `NodeKind`, allocate per-pass ID hash maps, or call
`graph_post_order` themselves.

The index preserves caller root order and child order. Shared subgraphs appear
once. Construction remains stack-safe for deep graphs.

### Analysis ownership and invalidation

The compiler driver owns analyses. Passes declare which indexed facts they
consume and which plan tables they produce. Code-generation passes cannot
invalidate semantic topology because they do not mutate or rebuild it.

Examples:

| Pass | Reads | Produces |
|---|---|---|
| GEMM epilogue selection | node kinds, consumers, shape/dtype/device | epilogue regions |
| Elementwise region formation | node kinds, consumers, broadcast metadata | elementwise regions and `KernelExpr` |
| Reduction termination | elementwise regions, reduction metadata | fused-reduction regions |
| Multi-output selection | region consumer DAG, buffer/operation limits | multi-output regions |
| Optimizer specialization | optimizer nodes, shape/dtype/hyperparameters | fused/grouped optimizer regions |
| Backend lowering | all selected regions and uncovered nodes | typed lowered program |

Passes may make separate linear scans over `GraphIndex.order`. This RFC rejects
repeated pointer-DAG discovery and full graph allocation, not clear pass
boundaries.

### Optimization plan

The optimization result is a region plan keyed by stable semantic node IDs:

```rust
pub struct OptimizationPlan<R> {
    pub regions: Box<[R]>,
    pub node_region: Box<[Option<RegionId>]>,
    pub outputs: Box<[Option<RegionOutput>]>,
}

pub struct RegionOutput {
    pub region: RegionId,
    pub index: u32,
}

pub enum NativeRegion {
    Elementwise(ElementwiseRegion),
    ElementwiseReduce(ElementwiseReduceRegion),
    MultiOutput(MultiOutputRegion),
    LinearResidual(LinearResidualRegion),
    LinearGelu(LinearGeluRegion),
    AdamW(OptimizerRegion),
    AdamWGroup(OptimizerGroupRegion),
    Sgd(OptimizerRegion),
}
```

The exact enum can be split into backend-neutral and backend-specific plans.
The invariant is that a region describes implementation coverage and output
routing without claiming to be a new semantic tensor operation.

Every semantic node is either:

- covered as an internal node of exactly one selected region;
- the producer of one or more selected region outputs;
- or lowered independently.

Every semantic root and every use resolves through the output mapping to one
lowered value. Multi-output selectors are compile-time mappings, not graph
nodes and not runtime side maps.

### Kernel expression IR

The existing `Expr` becomes `KernelExpr`. Its scope remains intentionally
narrow:

```text
KernelExpr := InputLane(index)
            | ScalarLane(index)
            | Constant(bits)
            | Unary(op, KernelExpr)
            | Binary(op, KernelExpr, KernelExpr)
            | Select(condition, true_value, false_value)
```

It remains structurally comparable and hashable. CPU lowers it once to a
flattened `CpuFusionProgram`; Metal emits SSA-form MSL and keys the pipeline
cache from the expression, lane strides, shape, scalar count, and dtype.

`KernelExpr` does not contain tensor shapes, memory locations, backend buffers,
state effects, instruction scheduling, or ownership. Those belong to the
region, lowered instruction, and memory plan.

### Region formation

Elementwise region formation retains the existing legality rules:

- supported device and dtype;
- right-aligned broadcast-compatible input lanes;
- constant folding behavior;
- single-consumer extension rules;
- input-lane and Metal buffer-argument limits;
- expression operation-count limits;
- exact `select` behavior for masked NaN-producing branches;
- reduction geometry and index-width guards;
- stable root and output ordering.

Instead of storing an open `Region` and eventually replacing semantic nodes,
the pass records:

```rust
pub struct ElementwiseRegion {
    pub nodes: Box<[DenseNodeId]>,
    pub inputs: Box<[DenseNodeId]>,
    pub lane_strides: Box<[Box<[usize]>]>,
    pub outputs: Box<[ElementwiseOutput]>,
}

pub struct ElementwiseOutput {
    pub semantic_node: DenseNodeId,
    pub expression: KernelExpr,
}
```

The backend receives one region output for a single-output kernel or several
for a multi-output kernel.

### Multi-output selection without whole-graph fixpoint rebuilding

After base elementwise regions are formed, the compiler builds a region
consumer DAG. It discovers all legal shared-prefix candidates and orders them
deterministically by semantic postorder and current tie-break rules.

Selection uses one of these equivalent bounded strategies:

1. Choose a deterministic maximal set of non-conflicting candidates in one
   pass, apply them to the region table, then update only affected neighbors.
2. Maintain a priority/work queue keyed by candidate order; when a merge is
   selected, invalidate and recompute candidates touching its regions.

Neither strategy may rebuild or re-index the semantic graph. A full indexed
array scan is permitted between bounded optimization phases, but not once per
individual merge.

The migration must prove that the selected regions preserve current behavior
for:

- shared prefixes that still require a materialized output;
- prefixes used only by fused continuations;
- output-shape grouping;
- broadcast-smaller prefixes that may be inlined but not materialized at the
  continuation shape;
- nested or conflicting continuation groups;
- the 31-buffer Metal argument limit;
- the maximum merged expression operation count;
- deterministic output index assignment.

### Authoritative lowered program

Backend lowering emits the program consumed by planning and execution:

```rust
pub struct LoweredProgram<I, M> {
    pub values: Box<[LoweredValue<M>]>,
    pub instructions: Box<[LoweredInstruction<I>]>,
    pub outputs: Box<[ValueId]>,
}

pub struct LoweredInstruction<I> {
    pub id: InstructionId,
    pub kind: I,
    pub inputs: Box<[ValueUse]>,
    pub outputs: Box<[OutputDecl]>,
    pub scratch: Box<[ValueUse]>,
    pub staging: Box<[ValueUse]>,
    pub status: Box<[ValueUse]>,
    pub state: Box<[ValueUse]>,
    pub effects: InstructionEffects,
}
```

The resource arrays may be represented by typed table ranges rather than
separate boxes. They are shown explicitly to make the contract clear.

`I` is the real backend instruction kind:

```rust
pub enum CpuInstruction {
    Unary { op: CpuUnaryOp },
    Binary { op: CpuBinaryOp },
    Matmul { plan: MatmulRequirements },
    Elementwise { program: CpuFusionProgram, geometry: FusionGeometry },
    // ...
}

pub enum MetalInstruction {
    Unary { op: MetalUnaryOp, pipelines: PipelineRange },
    Gemm { plan: GemmRequirements, pipelines: PipelineRange },
    Elementwise { exprs: Box<[KernelExpr]>, geometry: FusionGeometry },
    // ...
}
```

There is no `LoweredInstruction<String>` in production. Instruction names for
diagnostics are derived from typed kinds.

Algorithm requirements are selected during lowering and retained in the typed
instruction. Scratch and staging values derived from those requirements are
added before liveness analysis. Execution does not recalculate requirements or
discover allocations.

### Memory planning

The existing planner remains responsible for:

- alias normalization;
- read-before-definition and dense-table validation;
- value live intervals;
- storage classes and ownership separation;
- memory-space and alignment constraints;
- best-fit free-range reuse;
- bounded segment growth;
- output locations and reuse edges;
- deterministic memory reports.

Its input changes from a string-valued projection to the authoritative typed
program. A common accessor flattens each instruction's declared resource uses
for liveness. The planner never needs to understand backend instruction
semantics beyond those declared accesses and storage requirements.

CPU and Metal continue to choose different alignment, segment-capacity, and
memory-space policies.

### Physical synchronization planning

Logical instruction order and memory reuse establish dependencies. Metal also
needs barriers, command-buffer commits, events, completion boundaries, and
status gates. These remain backend-specific.

The physical plan does not duplicate `MetalOp` and its value/resource lists. It
references lowered instruction IDs:

```rust
pub enum MetalPhysicalCommand {
    Encode(InstructionId),
    Barrier,
    Commit,
    Wait(EventId),
    Signal(EventId),
    StatusGate(StatusId),
}
```

If one lowered instruction expands to a fixed internal dispatch sequence, that
sequence is declared by its retained algorithm plan and pipeline ranges. The
memory planner sees the conservative exact resource lifetime required by the
instruction. A later refinement may expose nested dispatches as instructions
when doing so improves liveness, but the planner and executor still share one
authoritative declaration.

CPU may use a physical program containing only `Encode(InstructionId)` entries
when no separate synchronization commands are required.

### Executable plan

An executable retains immutable compilation artifacts and no semantic roots for
execution:

```rust
pub struct NativeExecutablePlan<I, P, M, C> {
    pub signature: ProgramSignature,
    pub program: Arc<LoweredProgram<I, M>>,
    pub memory: MemoryPlan<M>,
    pub physical: Box<[C]>,
    pub pipelines: Box<[P]>,
    pub diagnostics: ExecutableDiagnostics,
}
```

CPU and Metal may use type aliases or backend wrappers around this shape. They
do not maintain a second instruction list with independently copied resource
metadata.

RFC 0020 remains authoritative for invocation frames, workspace acquisition,
provisional output retention, producer completion, failure publication, and
state commit.

### Compiler driver

The shared compiler crate owns the sequence:

```rust
prepare request
select optimization regions
ask backend to lower indexed nodes and regions
validate lowered program
plan logical memory
ask backend to plan physical hazards and prepare pipelines
build diagnostics
publish executable plan
```

Backend implementations provide statically linked Rust types and callbacks for:

- capability and dtype validation;
- backend instruction kinds;
- algorithm and kernel selection;
- resource requirement derivation;
- memory-space configuration;
- pipeline preparation;
- physical synchronization planning;
- invocation encoding.

This is a compile-time Rust boundary, not a dynamic plugin ABI. The compiler
driver does not match on `Device::{Cpu, Metal}` to implement backend kernels.

### Compile options

Every public compile option must satisfy one of these conditions:

1. It changes preparation, region selection, lowering, physical planning, or
   pipeline preparation and joins the executable cache key.
2. It is removed from the public/internal request until implemented.

An option may not be parsed and stored while having no production lowering
consumer. The implemented request retains `optimize` and inference-only
`constant_weights`. The executable precision option was removed because it had
no lowering consumer; it remains absent until a precision policy is specified.
Trainer `mixedBf16` is a separate training graph policy, not an executable
compile option. Environment switches are captured once in
`CompileOptions.environment` and join structural cache identity.

## Crate Responsibilities

### `effect-torch-graph`

Owns:

- semantic `Node` and user/transform-visible `NodeKind` operations;
- shape, dtype, and device metadata validation;
- semantic child enumeration and remapping utilities needed by real semantic
  transforms;
- leaf storage contracts.

Does not own code-generation-only fused node kinds after migration.

### `effect-torch-autodiff`

Owns semantic graph transforms for gradients and vmap. It runs before compiler
optimization and never sees `OptimizationPlan`, `KernelExpr`, or backend
instructions.

### `effect-torch-compiler`

Owns:

- `ProgramRequest` preparation;
- `GraphIndex`;
- optimization pass order and region selection;
- `KernelExpr` and fusion utilities;
- authoritative generic lowered-program tables;
- lowered-program validation;
- liveness and memory planning;
- compiler diagnostics and work metrics;
- the generic statically linked compiler driver.

### `effect-torch-runtime`

Owns backend-neutral runtime contracts:

- dense IDs;
- dtypes, layouts, placement, and erased buffers;
- program and invocation signatures;
- memory-plan and ownership types;
- executable diagnostics;
- workspace pools and leases;
- cancellation and backend errors.

Unused executable traits or generic shells are removed unless the production
CPU and Metal path adopts them during this RFC.

### `effect-torch-runtime-cpu`

Owns CPU instructions, algorithm requirements, kernels, pipeline/program
preparation, physical execution, workspace allocation, and CPU-specific state
adapters.

### `effect-torch-runtime-metal`

Owns Metal instructions, algorithm requirements, MSL emission, pipeline
preparation, physical synchronization planning, submission contexts, Metal
workspace allocation, and Metal-specific state adapters.

### Backend TypeScript adapters and NAPI

Own opaque handle validation, cancellation bridging, public request mapping,
and backend-specific state handles. They do not independently recover compiler
analyses after `PreparedProgram` exists.

## Cache Architecture

The existing cache layers remain distinct:

| Cache | Key | Value |
|---|---|---|
| TypeScript compiled-function cache | Runtime identity plus input placement/shape/dtype signature | User-level compiled program handle |
| Native structural executable cache | Structural graph, options, state, capabilities, and generated-binding contract | Shared immutable executable plan |
| Metal pipeline cache | Exact emitted-kernel structural key | Compiled Metal pipeline |

`PreparedProgram` is used for native structural cache lookup and cache-miss
compilation. Generated leaves are collected once in deterministic semantic
order and rebound to a cached immutable plan when their signatures match.

The refactor does not let a structural cache entry retain arbitrary concrete
user buffers. RFC 0019 generated-binding ownership remains unchanged.

## Diagnostics

Public executable diagnostics continue to report semantic node counts,
instruction counts, pipeline count, command count, synchronization count, and
memory totals.

Their source becomes authoritative:

- semantic counts come from `GraphIndex`;
- instruction counts come from typed lowered instructions;
- pipeline counts come from prepared pipeline ranges;
- command and synchronization counts come from the physical plan;
- memory diagnostics come from `MemoryPlan`;
- compile phase timings are populated by the compiler driver rather than left
  structurally present but empty.

Debug diagnostics may additionally expose region coverage:

```text
semantic nodes
nodes covered by each optimization kind
selected region count and output count
uncovered materializing nodes
lowered instruction count
graph index builds and edge visits
semantic nodes rebuilt by code-generation optimization (required: zero)
```

## Correctness Strategy

### Differential compiler mode

During migration, tests can run both old and new compilers from the same roots
and compare:

- output metadata;
- input and scalar slot declarations;
- generated binding order;
- random-source order and backend seed derivation;
- optimization region coverage;
- lowered instruction names and counts;
- planned storage classes, segment sizes, locations, aliases, and reuse edges;
- physical command and synchronization counts;
- optimized execution results;
- failure and cancellation behavior.

The old path is test-only during migration and is deleted when the new path
passes all gates. It is not retained as a compatibility or runtime fallback.

### Optimized versus unoptimized execution

Every fusion family keeps focused parity tests against `optimize: false`:

- scalar and broadcast elementwise chains;
- comparisons and true select with NaN-producing unselected branches;
- deep expression chains;
- fused reductions over every supported dimension/keepdims geometry;
- shared-prefix multi-output regions with and without a materialized prefix;
- GEMM residual and GELU epilogues, including dual outputs;
- AdamW, grouped AdamW where enabled, and SGD trajectories;
- F32/F64 CPU and F32/BF16 Metal behavior;
- zero-sized and large-index boundary behavior where supported;
- multi-root deduplication and random sharing.

### Planner validation

Existing planner tests remain. Integration tests additionally validate that
every value used by execution was declared to liveness with the same access
mode. Debug builds may poison retired ranges or execute with reuse disabled
while retaining the same lowered instructions.

### Deep graph safety

Graph indexing, region formation, kernel-expression construction, lowering,
validation, hashing, cloning, and destruction remain stack-safe for the current
deep-graph test limits.

## Performance Strategy

### Execution benchmarks are release gates

Before replacing a pass, record the current compiler output and runtime result
for the benchmark corpus that motivated it. At minimum this includes:

- elementwise chain fusion;
- broadcast and scalar fusion;
- fused reductions;
- multi-output shared-prefix fusion;
- GEMM residual and GELU epilogues;
- AdamW and SGD implementations, including the measured grouped-kernel losing
  case;
- semantic cross-entropy, layer norm, rotary, attention, KDA, convolution, and
  chunked-head paths;
- reference compiled training steps;
- prefill, single-token decode, and batched decode.

On the same machine, build, model, shape, dtype, and measurement protocol, a
replacement may not regress median steady-state execution outside the
repository's established noise threshold. If output topology intentionally
changes, the RFC implementation record includes the before/after command,
pipeline, memory, and timing measurements.

### Compile-time benchmarks are added

Compile benchmarks cover representative graph sizes and structures:

- long linear elementwise chains;
- wide shared-prefix graphs with many multi-output opportunities;
- full forward/backward/update training graphs;
- decode-specialized graphs;
- many-root optimizer programs;
- 50,000- and 100,000-node stack-safety stress graphs.

They record:

- wall-clock compile time;
- peak host resident memory where practical;
- total graph-index edge visits;
- semantic `Node` allocations during code-generation optimization;
- region-candidate and selected-region counts;
- lowered values and instructions;
- shader/pipeline compile time separately from graph compiler time.

The structural acceptance targets are deterministic:

- one `GraphIndex` construction per semantic graph generation;
- zero semantic `Node` allocations by code-generation optimization passes;
- no whole-graph traversal per individual multi-output merge;
- no production string-valued lowered schedule;
- no repeated generated-binding traversal on one cache miss;
- final lowering consumes the existing indexed order.

Wall-clock improvement is expected but does not replace these structural gates.

## Migration

### Phase 0: Freeze baselines and instrument compiler work

- Add compile phase timings and `CompilerWorkReport` counters.
- Record current region, instruction, pipeline, command, synchronization, and
  memory diagnostics for the correctness and benchmark corpus.
- Add focused compile-time benchmarks for wide multi-output graphs and complete
  training graphs.
- Add a test hook that compares old optimized output against `optimize: false`.

No production architecture changes in this phase.

### Phase 1: Prepared requests and shared graph index

- Make `ProgramRequest` the native internal compile input.
- Introduce stack-safe `GraphIndex` construction.
- Move node count, topological order, consumers, slots, generated leaves, and
  random-source metadata into the index.
- Pass one `PreparedProgram` through native cache lookup and compilation.
- Remove repeated `semantic_generated_bindings`, `collect_program_slots`, and
  backend-local `graph_post_order` calls from the compile path.
- Preserve current graph rewrites temporarily while validating the new index.

### Phase 2: Side-table elementwise and reduction regions

- Rename `Expr` to `KernelExpr` and update CPU/Metal nested-kernel consumers.
- Implement elementwise region formation over `GraphIndex`.
- Implement fused-reduction termination over region tables.
- Lower selected regions directly to typed CPU and Metal instructions.
- Differentially compare region boundaries and results with the old rewrite.
- Remove production construction of `FusedElementwise` and `FusedReduce` after
  parity and benchmark gates pass.

### Phase 3: Side-table epilogues and multi-output regions

- Represent linear residual and GELU epilogues as selected regions.
- Build the region consumer DAG.
- Replace one-merge graph fixpoint behavior with deterministic batched or local
  worklist selection.
- Preserve dual-output and materialized-prefix behavior.
- Remove `FusedElementwiseMulti`, `FusedPick`, `LinearResidual`, and
  `LinearGelu` code-generation-only semantic variants when no semantic
  transform requires them.

If a semantic API directly creates `Linear`, it remains a semantic node.
Only compiler-created epilogue variants are removed from semantic `NodeKind`.

### Phase 4: Optimizer region specialization

- Move AdamW/SGD kernel-expression selection and optional grouping into the
  optimization plan.
- Preserve current shape, dtype, hyperparameter, lane, and buffer-limit rules.
- Preserve the benchmark-derived default policy for grouping.
- Remove compiler-created grouped/selective optimizer graph variants once all
  outputs map through region output tables.

### Phase 5: Authoritative typed lowered program

- Extend `LoweredInstruction<I>` with exact resource and effect declarations.
- Make CPU lowering produce `LoweredProgram<CpuInstruction, CpuMemorySpace>`.
- Make Metal lowering produce
  `LoweredProgram<MetalInstruction, MetalMemorySpace>`.
- Run liveness and memory planning directly over those programs.
- Derive diagnostics from typed instructions.
- Delete the string-valued schedule projection.
- Update executors to resolve typed instructions or physical commands that
  reference their IDs.

CPU may migrate first because it has fewer physical synchronization concerns.
Metal migration follows with explicit status and submission-boundary parity.

### Phase 6: Common driver and specialization boundary

- Move compile sequencing into the shared compiler driver.
- Make CPU and Metal provide backend lowering, memory policy, pipeline, and
  physical-planning implementations.
- Consolidate duplicated decode specialization where its semantics are backend
  neutral; retain only storage/kernel-specific state adapters in backends.
- Populate compile phase diagnostics.
- Audit every compile option for a production consumer.

### Phase 7: Remove superseded shells and old compiler

- Remove `NodeKind::Fused*` variants no longer used by semantic transforms.
- Remove old graph rewrite implementations and test-only dual compilation.
- Connect `ProgramSignature` and common invocation validation to the production
  executable path.
- Remove or replace `NativeExecutable` and `ExecutableBackend` if they remain
  parallel shells rather than the production plan and backend boundary.
- Remove backend-local duplicate compile drivers.
- Update RFC 0007 and RFC 0019 implementation notes and architecture diagrams.
- Update the README architecture description to match the production pipeline.

Effect Torch remains pre-release. No backward-compatibility adapter is added for
internal Rust types or native executable handles unless a concrete persisted or
external consumer is identified.

## Risks and Mitigations

### Fusion boundaries accidentally change

Region selection over side tables may differ subtly from graph rewrite order,
especially for shared prefixes, broadcasting, root consumers, and fixpoint
merges.

Mitigation: differential region diagnostics, old/new compiler tests during
migration, exact kernel-expression comparison where applicable, and benchmark
gates for every fusion family.

### Stable node IDs expose accidental ordering dependencies

The old compiler minted new node IDs during rewrites. Some behavior may
accidentally depend on rewritten IDs rather than semantic order.

Mitigation: define deterministic ordering from the original postorder, caller
root order, child order, and explicit pass tie-breaks. Random provenance is
separate from physical instruction position.

### One authoritative instruction becomes too broad

CPU and Metal need different algorithm plans, resources, status handling, and
synchronization. Over-generalizing `LoweredInstruction` could recreate a large
lowest-common-denominator abstraction.

Mitigation: parameterize the instruction kind and memory space. Share dense
tables, access declarations, validation, and planning contracts, not backend
operation enums.

### Physical command planning duplicates semantics again

A Metal physical stream could grow into another complete command IR with copied
operands and resources.

Mitigation: physical commands reference `InstructionId`. Backend tensor
semantics and resource declarations remain in the lowered instruction only.

### Compiler memory remains high through retained semantic roots

`GraphIndex.order` retains `Arc<Node>` values while lowering. Large training
graphs therefore remain live through compilation even after side-table fusion.

Mitigation: this is one source graph generation rather than several rewritten
generations. Measure peak memory first. A later arena or compact semantic table
is justified only if the retained source graph remains a measured bottleneck.

### Decode specialization remains duplicated

Moving fusion alone would leave CPU and Metal decode rewrites separate.

Mitigation: phase 6 moves backend-neutral stateful semantic transformation into
the shared preparation pipeline after stateless fusion parity is established.
The larger refactor is not blocked on decode consolidation.

### Compile caches hide regressions

Steady-state benchmarks may hit executable or pipeline caches and miss first
compile regressions.

Mitigation: benchmark cold structural compile, warm executable-cache lookup,
cold pipeline preparation, and steady-state execution separately.

## Alternatives Considered

### Keep the current graph rewrites

This preserves known behavior but retains repeated graph traversal, full graph
allocation for local choices, identity churn, the multi-output fixpoint, and the
shadow planning schedule. It does not address the motivation.

### Keep graph rewrites but add a pass manager

Shared topological and consumer analyses would remove some repeated work, but a
local rewrite would still reconstruct ancestors and optimization choices would
still be encoded as semantic nodes. This is useful as a migration step, not the
endpoint.

### Move the semantic graph to a mutable arena first

Stable dense node IDs and mutable edges would make graph rewriting cheaper, but
would change graph ownership, lazy tensor handles, autodiff, and thread-safety
before establishing that semantic graph storage is itself the bottleneck.
Side-table optimization obtains the required stability without that migration.

### Fuse only during backend execution

Runtime fusion would rediscover regions per invocation, make memory planning
incomplete, and conflict with RFC 0019's fixed executable schedule. Rejected.

### Remove advanced fusion

The optimization families were introduced from measured execution bottlenecks.
Removing them would simplify compiler code by giving up established throughput
and memory wins. Rejected.

### Adopt an external general-purpose compiler IR

MLIR, StableHLO, XLA, or another system could provide mature pass and analysis
infrastructure, but integrating one would be substantially larger than fixing
the current first-party compiler boundaries. It would also need representations
for Effect Torch's optimizer, KDA, decode state, paged KV, transactional state,
and native kernel contracts. This RFC does not prevent a future delegated
backend, but does not require one for native CPU and Metal.

## As-Built Implementation Record (2026-08-11)

### Compiler and optimization pipeline

- Each production compile submission has one `ProgramRequest`, one
  `PreparedProgram`, and one `GraphIndex` for its semantic graph generation.
  Shared decode specialization is implemented once in the compiler crate and
  runs before indexing when a stateful inference request authorizes a new
  semantic graph.
- The semantic graph remains the nongeneric `Node`/`NodeKind` tensor DAG.
  `GraphIndex` provides deterministic dense order, children, consumers, roots,
  slots, slot leaves, generated bindings, and random-source provenance as side
  tables. Construction, hashing, optimization, decode specialization, and deep
  graph teardown remain iterative and stack-safe.
- Generated leaves are collected once into the index. Native structural cache
  lookup, insertion, and generated-binding order consume that collection rather
  than walking roots again.
- `OptimizationPlan` records elementwise, fused-reduction, multi-output,
  linear-residual, linear-GELU, AdamW/grouped-AdamW, and SGD regions plus
  semantic-output routing. The plan does not create semantic nodes.
- Multi-output selection builds indexed region dependencies and uses a bounded
  worklist. It retains materialized-prefix and output-shape behavior and creates
  split duplicated-expression regions when an extra lane has transitive prefix
  ancestry, preserving an acyclic lowering order without a graph fixpoint.
- The scalar fusion IR is `KernelExpr`. It is contained inside regions and
  backend fusion programs; it is not a tensor-program or ownership IR.
- The old fusion/GEMM graph-rewrite path, compiler-created `Fused*`,
  `FusedPick`, `LinearResidual`, and `LinearGelu` semantic `NodeKind`s, the
  string-valued lowered projection, duplicate backend decode transforms, and
  unused generic executable shells were removed. Semantic transforms such as
  autodiff, vmap, checkpointing, and shared decode specialization remain graph
  transforms because they change the requested program.

### Lowering, planning, and execution

- CPU and Metal directly produce authoritative
  `LoweredProgram<CpuInstruction, ...>` and
  `LoweredProgram<MetalInstruction, ...>` values. Their backend-specific value
  records retain shape/layout/storage metadata, while each typed instruction
  declares its inputs, outputs, scratch, staging, status, state, effects, and
  selected algorithm requirements.
- Logical liveness, aliases, reuse edges, segment assignment, output ownership,
  and memory diagnostics are computed from those exact value and resource
  declarations. Reports distinguish external, persistent, state, escaping
  output, workspace, transaction, peak-live, and packing-overhead bytes.
- Physical plans reference the authoritative instruction table by
  `InstructionId`. CPU uses `Encode(InstructionId)` entries. Metal adds
  `StatusGate(InstructionId)`, commit, and completion boundaries without
  copying the tensor operation or its resources into another command IR.
- Pipeline preparation occurs during executable compilation. Execution only
  validates the invocation, resolves planned resources, follows the physical
  ID plan, observes status, commits state, and publishes owned outputs; it
  performs no graph traversal, kernel compilation, optimization, allocation
  discovery, or fallback.
- Executables retain typed programs, physical and memory plans, prepared
  pipelines/constants, signatures, and diagnostics. RFC 0020 invocation frames
  own workspace, staging, provisional output, completion, and transaction
  resources, so immutable CPU and Metal executables are concurrently callable.
- `ProgramSignature` is authoritative for binding metadata, scalar/runtime
  values, RNG counters, and outputs. Decode preparation turns the semantic
  state-cursor slot into a validated scalar or bounded tensor runtime-value
  contract, and execution validates that contract against the state schema
  before staging cursors.
- Inference-only `constant_weights` is implemented: authorized concrete leaves
  become retained constants and the request bypasses structural executable
  caching rather than storing arbitrary user buffers in a shared cache entry.
  Ordinary generated leaves remain invocation bindings.
- Executable compile precision was removed because no production lowering
  policy consumed it. Trainer mixed-BF16 remains a separate policy that builds
  the appropriate cast graph while retaining F32 master parameters and state.
  `optimize: false` is the same typed compiler, planner, executor, and ownership
  baseline with optional code-generation regions disabled.
- Compiler diagnostics populate phase timings for graph indexing, optimization,
  lowering, lowered-program validation, memory planning, physical planning,
  pipeline preparation, artifact assembly, submission, and publication where
  applicable. `CompilerWorkReport` records graph-index builds and edge visits,
  pass scans, rebuilt semantic nodes, fusion candidates, multi-output dependency
  work, selected regions, and lowered value/instruction counts.

### Verification and measurements

Completion verification passed the complete Rust workspace and pnpm parity
suites:

```bash
cargo test --workspace --features napi-addon
pnpm test
```

Those suites include optimized/unoptimized CPU and Metal topology and numerical
parity; 50,000/100,000-node deep-graph coverage; same-executable concurrency;
state cursor, commit, and rollback behavior; random-source sharing and fresh
invocation draws; and deferred-status/error tests. The benchmark entry points
used for compile and execution smoke measurements are:

```bash
cargo bench -p effect-torch-compiler --bench pipeline
cargo bench -p effect-torch-compiler --bench pipeline -- --workload wide --size 256
cargo bench -p effect-torch-compiler --bench pipeline -- --workload stress
pnpm bench:compile
pnpm bench:compile -- --runtime cpu --workload wide,training,decode
pnpm bench:compile -- --runtime metal --workload wide,training,decode
pnpm bench
```

Latest Apple M4 Max samples from the implemented compiler were:

| Measurement | Workload | Median |
|---|---|---:|
| Rust graph analysis | wide, size 256 | 0.910 ms |
| TypeScript CPU cold / warm | wide | 1.394 / 0.504 ms |
| TypeScript CPU cold / warm | training | 2.086 / 2.093 ms |
| TypeScript CPU cold / warm | decode | 1.478 / 0.622 ms |
| TypeScript Metal cold / warm | wide | 2.607 / 0.538 ms |
| TypeScript Metal cold / warm | training | 3.461 / 3.393 ms |
| TypeScript Metal cold / warm | decode | 9.787 / 0.673 ms |

The Rust wide sample had 773 semantic nodes, 1,027 graph edges, one graph-index
build, zero rebuilt semantic nodes, and one multi-output dependency pass with
256 dependency-edge visits. The TypeScript samples use the benchmark's default
wide/training/decode sizes and report full cold native compile versus warm
structural-cache lookup. Metal decode reported 6.106 ms in the native
`pipeline_preparation` phase; that phase is part of the cold compile, not a
third wall-clock sample.

An earlier graph-analysis output-schema sample measured 50,000 and 100,000-node
stress graphs at 15.535 ms and 37.403 ms respectively. It is retained only as
an earlier graph-analysis sample because the benchmark output schema changed;
it is not presented as directly comparable to the latest rows. Current matmul
smoke measured 42.265 ms/op on CPU and 0.270 ms/op on Metal.

No historical old-pipeline/new-pipeline wall-clock pair was captured under one
measurement protocol, so this record makes no before/after speedup claim. The
acceptance evidence is instead deterministic: one index, zero code-generation
semantic-node rebuilds, bounded DAG/worklist dependency work, no shadow
schedule, and one generated-leaf collection. Runtime topology/parity tests and
the current matmul smoke also passed.

## Acceptance Criteria

The RFC is complete when:

1. CPU and Metal production compilation use one prepared graph index per
   semantic graph generation.
2. Code-generation optimization creates no replacement semantic `Node` values.
3. Elementwise, fused-reduction, multi-output, GEMM-epilogue, and optimizer
   regions retain their correctness and benchmark behavior.
4. Multi-output selection performs no whole-graph operation per individual
   merge.
5. Native cache miss handling does not collect generated bindings twice.
6. CPU and Metal lower directly to typed authoritative programs.
7. Memory planning consumes those programs without a string-valued shadow
   schedule.
8. Execution and physical synchronization reference the same lowered
   instruction IDs and resource declarations.
9. Public executable diagnostics derive from authoritative compiler artifacts.
10. Compile phase timing and structural work metrics are populated and tested.
11. `optimize: false` uses the same lowering, planning, ownership, and execution
    path.
12. Random, cancellation, error precedence, state rollback/commit, output
    lifetime, and concurrent invocation tests remain green.
13. Cold-compile benchmarks record current wall-clock baselines and deterministic
    structural metrics prove zero graph reconstruction for wide and deep
    optimized graphs.
14. Runtime topology/parity tests and the current execution smoke corpus have no
    unexplained regression; no unmeasured historical speedup is claimed.
15. Superseded fused semantic nodes, duplicate compiler drivers, and unused
    generic shells are removed or connected to production with one documented
    responsibility each.

## Relationship to Existing RFCs

### RFC 0007: Kernel fusion

RFC 0007 remains authoritative for fusion legality, expression semantics,
numerics, backend kernel generation, and benchmark motivation. This RFC changes
the implementation from graph replacement to indexed region selection and
renames the nested scalar expression IR to clarify its scope.

### RFC 0017: Multi-backend runtime

Backends remain independently packaged and statically link shared Rust compiler
machinery. The common driver is an internal Rust boundary, not a runtime plugin
ABI. CPU and Metal retain independent typed instruction kinds and kernel plans.

### RFC 0019: Executable compilation

RFC 0019's semantic graph -> lowering -> memory planning -> executable model
remains authoritative. This RFC makes its intended lowered IR real in
production, removes the temporary string-valued planning projection, and
clarifies that code-generation optimization need not create a replacement
semantic graph.

### RFC 0020: Invocation ownership

This RFC does not change invocation ownership. Executable plans remain immutable
and shared. Workspace, staging, status, transaction, and output resources remain
invocation- or tensor-owned and are instantiated from the compiler's logical
memory plan.

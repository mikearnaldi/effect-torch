# RFC 0019: Executable Compilation - One Execution Path and Static Memory Planning

- **Status**: Draft
- **Author**: Michael Arnaldi
- **Date**: 2026-08-09
- **Depends on**: RFC 0002 (autodiff), RFC 0003 (memory management), RFC 0007
  (kernel fusion), RFC 0008 (compilation), RFC 0010 (inference), RFC 0012
  (dtypes), RFC 0015 (native backend), RFC 0017 (multi-backend runtime)
- **Updates**: RFC 0003 (runtime early-free becomes compile-time liveness for
  executable intermediates), RFC 0008 (compilation produces an executable,
  not a retained graph), RFC 0010 (decode uses the common executable), RFC
  0016 (capture/replay arenas and permanent compiled/uncompiled allocation
  modes are superseded), RFC 0017 (`evaluate` becomes compile-and-execute)

## Summary

Effect Torch has one semantic programming model: TypeScript builds immutable,
shape-known tensor graphs. The runtime should therefore have one production
execution model:

```text
Graph -> Compile -> Executable -> Execute
```

Compilation performs every structural decision: validation, shape and dtype
specialization, graph optimization, fusion, backend lowering, kernel
selection, instruction scheduling, liveness analysis, segmented memory
assignment, and synchronization planning. Execution validates bindings,
leases the planned storage, encodes the fixed schedule, submits it, and returns
the declared outputs. It does not walk a semantic graph, discover allocations,
run optimization passes, or silently choose another execution engine.

`Tensor.compute`, explicitly reusable programs, compiled training steps, and
inference/decode programs all use this pipeline. They differ in ownership,
bindings, state, and compile options, not in evaluator implementation.
"Frozen" ceases to be an execution mode: an explicit program pins an
executable, while `compute` obtains an equivalent executable from a cache or
compiles a transient one. "Fused" likewise names an optimization or lowered
instruction, never a run method.

Every lowered instruction declares its outputs, aliases, scratch storage, and
bounded invocation staging. The compiler computes a complete logical memory
layout for the Effect Torch-controlled resources in its selected schedule and
packs non-escaping intermediates into multiple right-sized backend segments.
On Metal, no segment may exceed `MTLDevice.maxBufferLength`; there is no
hard-coded 4 GiB slab and no first-run allocation capture. The layout is exact
for its declared resources in the chosen schedule, although the packing
heuristic need not prove a globally minimal layout.

There is no production graph interpreter and no silent fallback to one. An
unoptimized correctness baseline is produced by the same compiler and uses the
same executable IR, memory planner, and executor.

## Motivation

### The current `compile` is a retained graph

RFC 0008 removed repeated TypeScript graph construction, autodiff, and fusion
from a compiled call, but native `compile` currently stops after retaining a
post-fusion `Arc<Node>` DAG plus input declarations. `CompiledProgram::run`
still creates an `Evaluator`, counts consumers, walks nodes, invokes
`eval_uncached`, discovers temporary allocations through `MetalDevice::alloc`,
and synchronizes exactly like `eval_lazy`.

This is graph caching, not full compilation. It leaves execution order,
temporary storage, internal composed-operation workspaces, and command-buffer
structure implicit until every call.

### Multiple orchestration paths are already drifting

Ordinary evaluation and frozen-program execution duplicate evaluator setup,
root iteration, synchronization, deferred status checks, diagnostics, and
error cleanup. Decode adds specialized bindings, state locks, and commit
semantics around another graph walk. Fusion runs from both ordinary evaluation
and freeze. Arena capture/replay is enabled only for frozen programs.

The resulting behavior matrix includes:

```text
ordinary evaluation, fusion enabled
ordinary evaluation, fusion disabled
frozen first run, allocation capture
frozen later run, arena replay
frozen replay divergence, pool fallback
decode/prefill execution
semantic fused nodes with allocations hidden inside native loops
```

The modes are not independent in practice. A semantic node can contain a
chunk loop whose local values are invisible to outer evaluator liveness.
Manually reporting those values to the arena planner is both duplicated memory
semantics and a silent-corruption risk. The allocator cannot be made robust
while execution and lowering remain implicit.

### Shape-static graphs permit complete planning

Effect Torch graph construction knows every graph tensor's shape, dtype, and
device. Training traces produce the complete forward, backward, and
optimizer-update roots before backend compilation. Inference and decode
specialize fixed graph-shape buckets. Decode additionally consumes bounded
state-dependent values such as context lengths, block tables, real token
counts, and active batch width. Those values may change launch dimensions and
useful bytes, but their capacities, staging, instruction topology, and state
effects must be declared by the executable.

Once kernel and algorithm selection are fixed, the compiler can know every
Effect Torch-controlled device-visible allocation:

- graph-visible outputs and intermediates;
- materializations required by layout or dtype normalization;
- auxiliary outputs of multi-output instructions;
- split-K GEMM partials;
- attention, KDA, convolution, layer-normalization, and CE workspaces;
- chunk-loop accumulators and per-chunk scratch;
- deferred device-status buffers;
- command-buffer and synchronization boundaries that constrain reuse.

The first execution should not need to allocate one temporary working set to
discover those facts and a second working set to replay them.

### Memory planning belongs after lowering

Graph-level liveness alone is insufficient. Fusion can remove intermediates;
kernel selection can introduce scratch; semantic nodes can lower to loops or
multiple instructions; views can alias inputs; outputs and mutable runtime
state escape. Memory assignment must run over the explicit lowered schedule,
not over the pre-lowering semantic DAG and not over allocation calls observed
at runtime.

## Goals

1. Every production tensor execution goes through compile and execute.
2. `compute`, reusable programs, training, validation, serialization, prefill,
   and decode use one executable representation and one executor.
3. Fusion and all other optimizations happen only during compilation.
4. Optimizations can be disabled without selecting another runtime.
5. Every Effect Torch-controlled output, scratch, invocation-staging, and state
   transaction allocation is explicit in lowered IR.
6. The compiler computes liveness and a deterministic segmented memory plan
   before the first execution.
7. Execution performs no graph traversal and no intermediate allocation
   discovery.
8. Unsupported lowering fails compilation clearly; there is no silent CPU or
   interpreter fallback.
9. The architecture applies to CPU, Metal, and future runtime backends while
   permitting backend-specific instructions and memory spaces.
10. Current random, cancellation, multi-root deduplication, strict dtype, and
    transactional decode-state semantics remain observable behavior.

## Non-goals

- Symbolic dynamic dimensions. Shape changes select or compile another
  executable.
- Portable executable serialization.
- A public stream, command-buffer, or allocator API.
- Globally optimal memory packing. The selected plan must be deterministic and
  measured, not an optimality proof.
- Automatic graph partitioning across unrelated runtimes.
- Silent operation fallback between backends.
- Making Metal execution concurrent as part of this RFC. The design permits
  multiple workspace instances later, but the current runtime remains
  serialized.
- Moving backend lowering, scheduling, or memory planning into TypeScript.
- Requiring Metal heaps. They are an optional backing strategy for planned
  segments and must be benchmarked.

## Terminology

| Term | Meaning |
|---|---|
| Semantic graph | Immutable typed operation DAG built through the TypeScript API and native graph constructors |
| Program request | Roots, declared or generated bindings, compile options, and optional state schema submitted to the compiler |
| Lowered instruction | Backend-executable operation with explicit inputs, outputs, aliases, and scratch requirements |
| Schedule | Deterministic order and dependency structure of lowered instructions |
| Memory plan | Assignment of logical values and scratch intervals to external storage or backend segment slices |
| Executable | Immutable signature, instructions, pipelines, schedule, memory plan, outputs, and state schema |
| Executable instance | An executable plus a lease on concrete backing segments for one in-flight invocation |
| Transient compile | Compilation initiated by `Tensor.compute`; ownership is automatic and cacheable |
| Reusable compile | Explicitly owned, parameterized executable returned by the program API |
| Inference specialization | Optional compile-time rewrites and state lowering, such as KV/KDA/convolution caches, over the same compiler pipeline |

## Architecture

### One pipeline

```text
TypeScript graph construction
        |
        | arbitrary roots
        | training supplies forward/loss/autodiff/update roots
        v
ProgramRequest
        v
Validate and specialize
        v
Optional inference specialization
        v
PreparedProgram + one GraphIndex
        v
Side-table OptimizationPlan
        v
Typed backend LoweredProgram
        v
Liveness and memory assignment
        v
Physical InstructionId plan and prepared pipelines
        v
Executable
        v
Acquire workspace lease
        v
Execute
```

Program construction remains responsible for semantic graph transforms that
define the requested computation. Under the current API, training constructs
autodiff and optimizer roots before backend compilation. There is no training
compiler or training execution path: the common compiler sees additional
roots and dependencies. Optional inference settings explicitly authorize
inference-only rewrites and state interfaces; they do not create an alternative
compiler or executor.

### Program requests and inference specialization

Illustrative native types:

```rust
pub struct ProgramRequest {
    pub roots: Vec<Arc<Node>>,
    pub bindings: Vec<BindingDecl>,
    pub invocation: InvocationSignature,
    pub options: CompileOptions,
    pub state_cursor: Option<StateCursorSlot>,
}

pub struct CompileOptions {
    pub optimize: bool,
    pub inference: Option<InferenceOptions>,
    pub environment: EnvironmentOptions,
}

pub struct InferenceOptions {
    pub constant_weights: bool,
}

pub struct StateCursorSlot {
    pub slot: u32,
    pub tensor: bool,
}

pub struct BindingDecl {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub placement: Placement,
    pub layout: BindingLayoutPolicy,
    pub aliasing: BindingAliasing,
}

pub enum BindingLayoutPolicy {
    Require(LayoutConstraint),
    Canonicalize { target: Layout },
}

pub enum LayoutConstraint {
    Exact(Layout),
    Contiguous,
    AnyStrided,
}

pub struct InvocationSignature {
    pub scalars: Vec<ScalarDecl>,
    pub runtime_values: Vec<RuntimeValueDecl>,
    pub rng: Option<RngDecl>,
}
```

`optimize: false` disables optional code-generation regions and fusion. It does
not bypass lowering, scheduling, memory planning, or executable construction.

The illustrative executable `PrecisionPolicy` was removed from the implemented
compile request because it had no lowering consumer. It remains absent until an
executable precision policy is specified. Trainer mixed-BF16 is separate: the
Trainer builds a cast graph with F32 master parameters/state rather than setting
an executable compile option.

With `inference: None`, the compiler optimizes the requested roots without
inference-only assumptions. This is the path for arbitrary `Tensor.compute` and
for training graphs. Backward saved values, rematerialization boundaries,
optimizer updates, and their lifetimes are ordinary dependencies visible in
those roots.

`inference: Some(...)` authorizes constant-weight retention. A stateful public
compile request separately authorizes the shared semantic specialization that
lowers causal attention, KDA recurrence, and convolution history to persistent
decode state before `ProgramRequest` preparation. The resulting
`StateCursorSlot` becomes a bounded scalar or batched cursor runtime value in
`ProgramSignature`. Prefill, single-token decode, and batched decode remain
shape-specialized executables, not compiler modes of their own.

`Require` rejects a binding that does not satisfy its constraint.
`Canonicalize` always emits the declared copy into the schedule and memory
plan, even when a particular input could have skipped it; runtime layout does
not change instruction topology. Layout is never an unvalidated assumption.
Donation additionally requires the binding's alias/disjointness contract; an
ordinary borrowed JavaScript tensor handle is not assumed unique.

### Transient and reusable programs

`Tensor.compute(roots)` remains the ergonomic materialization API, but its
runtime meaning changes:

```text
parameterize concrete leaves where legal
compile or obtain cached executable
bind the original concrete values
execute
```

An explicitly reusable program traces against declared input slots and pins
the resulting executable. Both paths call the same compiler and executor.
There is no `eval_lazy` execution semantics distinct from program execution.

The distinction is ownership:

| Property | `compute` | Explicit program |
|---|---|---|
| Input slots | Generated or explicit | Explicit |
| Executable ownership | Runtime cache or invocation | User-visible handle |
| Expected calls | One or incidental reuse | Repeated reuse |
| Execution function | Common | Common |
| Memory planner | Common | Common |

Concrete tensor leaves are not baked into a globally cached executable by
default. The compiler classifies them as generated bindings so a structural
cache entry does not retain arbitrary user buffers. Every leaf retained by an
executable joins the key by identity and storage version. A folded constant
also joins by value hash. Any leaf that cannot provide stable identity/version
semantics is parameterized as a binding.

### Compile cache

The executable cache key includes every fact that can affect lowering,
scheduling, or layout:

```text
structural graph hash
complete binding declarations, layout policies, aliasing, and ownership
complete invocation signature and bounded runtime-value declarations
compile and inference-specialization options
runtime/client identity and backend/device capabilities
kernel-selection policy
complete state schema and geometry
retained-leaf identities/versions and folded-constant value hashes
compiler version/generation
```

Environment variables used for A/B testing must be read into explicit compile
options and therefore join the key. Execution must not re-read an option that
could change allocation count, instruction order, chunk size, algorithm, or
workspace requirements.

Transient cache entries are weakly held or bounded and must not retain
generated concrete bindings. Explicit program handles strongly own their
executable. Kernel pipelines may remain in backend-global structurally keyed
caches.

## Executable IR

### Values and instructions

Lowering converts prepared semantic nodes and RFC 0021 regions into an explicit
typed instruction schedule. One graph node may lower to zero instructions (a
view or eliminated value), one generic instruction, one fused instruction, or a
typed algorithm plan with fixed dispatch topology. Bounded runtime values may
select launch dimensions or active ranges, but may not introduce instructions,
choose another algorithm, or increase an allocation beyond its declared
capacity.

```rust
pub type ValueId = u32;
pub type InstructionId = u32;

pub struct Instruction {
    pub id: InstructionId,
    pub kind: InstructionKind,
    pub inputs: Vec<ValueUse>,
    pub outputs: Vec<OutputDecl>,
    pub scratch: Vec<ScratchDecl>,
    pub effects: InstructionEffects,
}

pub struct OutputDecl {
    pub value: ValueId,
    pub layout: Layout,
    pub dtype: DType,
    pub storage: StorageRequirement,
}

pub struct ScratchDecl {
    pub value: ValueId,
    pub bytes: usize,
    pub alignment: usize,
    pub memory_space: MemorySpace,
    pub lifetime: ScratchLifetime,
}
```

Instruction implementations receive assigned destinations and workspaces:

```rust
fn encode(
    instruction: &Instruction,
    inputs: &[BufferView],
    outputs: &mut [BufferView],
    scratch: &mut [BufferView],
    encoder: &mut CommandEncoder,
) -> Result<()>;
```

Production instruction encoding may not call an ambient tensor allocator for
an unplanned intermediate. Backend or driver allocations opaque to Effect
Torch, such as pipeline compiler state, are tracked as external headroom and
are not falsely represented as planned tensor memory.

Instruction kinds form a finite backend executable IR. Encoding may not compile
a pipeline, inspect tensor data to choose a schedule, emit an undeclared
variable-length instruction sequence, or allocate undeclared device storage.
An opaque composite instruction is permitted only when it declares its complete
dispatch topology, a conservative lifetime covering all of its scratch, and a
bounded opaque-workspace recipe. Native compilation rejects caller-provided or
Effect Torch-controlled workspace without a declared bound. Only allocator,
driver, or delegated-runtime internals outside Effect Torch's control may remain
opaque measured headroom; delegated opaque runtimes expose only the diagnostics
their executable protocol supports.

### Generic lowering is the correctness baseline

Every supported semantic operation has a generic lowering. With optional
optimizations disabled, the compiler emits approximately one instruction per
materializing graph operation while still using the common scheduler, memory
planner, and executor.

Optimized lowering selects side-table regions that combine those operations but
targets the same IR. Correctness tests compare optimized and unoptimized
executables. There is no graph interpreter whose allocation, synchronization,
or error semantics can drift from production execution.

### Fused and semantic operations

Fusion is complete before memory planning. A fused expression instruction
declares its materialized inputs and outputs; eliminated internal values never
enter liveness analysis.

Semantic operations with internal schedules must expose their complete storage
contract. For example, chunked head cross-entropy lowers conceptually to:

```text
validate targets / initialize active count
initialize f32 loss accumulator
for each fixed row chunk:
    head GEMM into one chunk-logits workspace
    CE reduction into chunk scalar/status workspace
    accumulate loss
backward, when requested:
    initialize dW/db accumulators
    for each fixed row chunk:
        recompute chunk logits
        form one chunk grad-logits workspace
        accumulate dW/db
        write the corresponding dX slice
finalize outputs
```

Loop-carried values and chunk scratch are explicit. Their lifetimes are derived
from the lowered schedule rather than reported by manually assembled pointer
sets. KDA, flash attention, split-K GEMM, layer normalization, convolution,
indexing materializations, and optimizer instructions follow the same rule.

### Views and aliases

Reshape, transpose, narrow, and other non-materializing operations produce
alias metadata instead of allocations. Each value is assigned an alias root
and byte/stride layout. The planner extends the root's lifetime through every
alias use.

In-place donation is explicit and legal only when:

- the donated input has no later use;
- its capacity, memory space, alignment, and layout satisfy the output;
- ownership permits mutation or transfer;
- external callers cannot observe the old value after execution;
- backend synchronization guarantees are preserved.

Inference input donation and functional training-state updates are compile
policies, never allocator guesses.

## Scheduling and Physical Hazards

The compiler first creates a deterministic logical instruction schedule over
all roots. Shared subgraphs become one value and one instruction sequence.
Multi-output instructions define every result together, and result selectors
are aliases to those declared outputs rather than evaluator side maps.

Logical scheduling fixes:

- instruction order;
- fixed-topology nested schedules and bounded launch geometry;
- kernel/algorithm variants;
- synchronization and deferred-status points;
- state commit points;
- cancellation-safe boundaries.

Memory planning runs after scheduling because last use is a property of the
selected order. A future memory-budgeted scheduler may trade recomputation for
shorter lifetimes, but it must emit one concrete schedule before assignment.

Memory assignment introduces physical anti-dependencies: a later value that
reuses a byte range must be ordered after the prior tenant's final access. The
planner emits reuse edges. A backend command-planning pass then realizes all
semantic and reuse edges as command-buffer boundaries, barriers, fences, or
events.

The implemented Metal physical plan uses one queue and explicit commit,
completion, and status-gate boundaries. Every submitted command buffer
contributes completion and error status to the invocation token.

### Authoritative lowered and physical programs

Graph and region structures are compiler inputs. Backend lowering emits one
dense typed `LoweredProgram`; memory planning and diagnostics consume that same
program. A physical plan adds synchronization mechanics by referencing lowered
`InstructionId`s and does not copy operation operands or resource declarations.
The executable does not retain `Arc<Node>` roots for execution and does not
perform reachability, topological, consumer-count, or last-use traversal at
invocation time.

Illustrative structure:

```rust
pub struct NativeExecutable<I, M, V, C, P> {
    pub signature: ProgramSignature,
    pub program: Arc<LoweredProgram<I, M, V>>,
    pub memory: MemoryPlan<M>,
    pub physical: Box<[C]>,
    pub pipelines: Box<[P]>,
}

pub enum CpuPhysicalCommand {
    Encode(InstructionId),
}

pub enum MetalPhysicalCommand {
    Encode(InstructionId),
    StatusGate(InstructionId),
    Commit,
    Complete,
}
```

CPU and Metal define different typed instruction, value, and algorithm-plan
records. Their lowered instructions declare inputs, outputs, scratch, staging,
status, and state resources. Physical commands refer to those instructions by
dense IDs rather than graph pointers or a second tensor-command enum.

Fixed internal dispatch sequences and loops are retained in typed algorithm
plans with conservative declared resources. Status gates may terminate a
suffix, but they do not discover new instructions. Launch geometry can be
computed from the invocation ABI; instruction topology and allocation remain
fixed.

After invocation resources are resolved, execution is conceptually:

```rust
let resolved = resolve_locations(executable, invocation, resources)?;
for command in executable.physical.iter() {
    match command {
        Encode(id) => dispatch(executable.program.instruction(*id), &resolved)?,
        // status/commit/completion mechanics do not duplicate tensor semantics
        _ => apply_physical_boundary(command, &resolved)?,
    }
}
```

Source maps and semantic-node diagnostics may be retained separately for error
reporting, but no production dispatch decision depends on traversing them.

## Static Memory Planning

### Storage classes

Logical storage is classified before packing:

```rust
pub enum StorageClass {
    ExternalInput,
    PersistentConstant,
    PersistentState,
    EscapingOutput,
    Workspace,
    DeviceStatus,
}

pub enum NativeMemorySpace {
    Cpu,
    MetalShared,
    MetalPrivate,
}
```

`MemorySpace` is backend-defined in the executable protocol.
`NativeMemorySpace` is the native CPU/Metal implementation, not a closed list
imposed on PJRT or remote runtimes.

External inputs, persistent constants/state, and escaping outputs are not
overwritten as workspace. Device statuses may be workspace only when their
readback/check lifetime ends within execution; otherwise they receive an
explicit escaping or staging allocation.

Shared and private Metal values are never packed into the same segment. The
first implementation may keep all Metal tensor storage shared. Private
intermediates require explicit shared upload/readback staging and a measured
benefit before becoming default.

### Live intervals

For every materialized value or scratch declaration, the compiler computes:

```text
birth = instruction that first writes the allocation
death = last instruction that reads or writes it, plus required GPU ordering
```

An interval can be reused only after the prior tenant's final GPU use is
ordered before the next tenant's first write. Aliases extend their root
interval. Loop-local scratch can reuse one location across iterations when the
schedule and barriers establish that order.

Persistent and escaping storage is excluded from ordinary workspace packing.
The memory report nevertheless includes it separately so executable admission
can account for the full runtime-owned footprint.

### Segment assignment

The planner assigns non-overlapping intervals to aligned slices in one or more
segments. A deterministic best-fit or heap-simulation heuristic is acceptable.
The plan records exact chosen capacities and offsets:

```rust
pub struct MemoryPlan {
    pub segments: Vec<SegmentDecl>,
    pub locations: Vec<Location>,
    pub outputs: Vec<OutputSlot>,
    pub state_transaction: Option<StateTransactionPlan>,
    pub workspace_bytes: usize,
    pub persistent_bytes: usize,
    pub escaping_bytes: usize,
}

pub struct SegmentDecl {
    pub bytes: usize,
    pub alignment: usize,
    pub memory_space: MemorySpace,
    pub ownership: SegmentOwnership,
}

pub enum Location {
    External { slot: u32 },
    Persistent { slot: u32 },
    InlineScalar { slot: u32 },
    Segment {
        segment: u32,
        offset: usize,
        bytes: usize,
    },
    Output {
        slot: u32,
        byte_offset: usize,
        bytes: usize,
    },
    State {
        slot: u32,
        byte_offset: usize,
        bytes: usize,
    },
    Alias {
        root: ValueId,
        byte_offset: usize,
    },
}
```

Segments are right-sized to the maximum assigned end offset, rounded only to
the backend's required granularity. A tiny executable gets tiny workspace. A
large executable gets multiple bounded segments. No final segment is inflated
to an arbitrary global cap.

`SegmentOwnership` distinguishes reusable workspace, provisional output,
invocation staging, and state-transaction shadow storage. Output backing is
acquired before dispatch and owned by the invocation until success publishes it
or failure releases it. A returned output alias retains the owner of its
storage. Workspace is never published accidentally.

### Metal constraints

The Metal planner queries and records device capabilities:

- `maxBufferLength` is a hard upper bound for one segment and for one logical
  value unless that value has an explicitly sharded kernel representation;
- `heapBufferSizeAndAlign` supplies size/alignment if `MTLHeap` placement is
  used;
- `recommendedMaxWorkingSetSize` is an admission and pressure signal, not a
  per-buffer limit or allocation guarantee;
- `currentAllocatedSize` is runtime pressure telemetry, not compiler-visible
  free memory.

One planned segment may use a standalone `MTLBuffer`, an automatic heap, or a
placement heap. That backing choice does not alter logical locations and is a
backend implementation detail. The first implementation uses standalone
buffers because existing binding offsets and barriers already support them.
Heaps are adopted only if allocation overhead or fragmentation measurements
justify their aliasing and accounting complexity.

A single logical value larger than `maxBufferLength` is a compile error unless
the lowering itself shards the value and every consumer understands that
representation. The memory planner never silently lets one tensor straddle
unrelated Metal buffers.

### Invocation values and bounded extents

The executable ABI classifies every per-invocation value:

- tensor bindings with layout and alias constraints;
- inline scalar values encoded directly where the backend supports them;
- planned scalar or argument staging buffers;
- RNG seed, nonce, and counters;
- decode cursors, context lengths, active batch width, token advances, block
  tables, and padding metadata;
- status and validation outputs.

A runtime value may alter launch dimensions or masks only within a compile-time
bound. Fixed-capacity storage and predication cover the maximum declared extent;
diagnostics report both capacity and useful runtime bytes. Runtime values may
not change instruction topology, kernel selection, or allocation capacity.

Decode `StateSchema` includes maximum tokens, block size, KV dtype, window,
batch width, recurrent geometry, and the invocation values needed by prefill or
decode. Variable-length block-table and context staging is either planned to
the schema maximum or implemented by allocation-free indirect access.

### State transactions

Stateful executables contain an explicit transaction plan:

```rust
pub struct StateTransactionPlan {
    pub slots: Vec<StateSlotPlan>,
    pub shadows: Vec<SegmentDecl>,
    pub reservations: Vec<ReservationPlan>,
    pub commit: Vec<CommitAction>,
    pub rollback: Vec<RollbackAction>,
}

pub enum StateAccess {
    ReadOnly,
    AppendBeforeMetadataCommit,
    OutOfPlaceNextState,
    ShadowedMutation,
}
```

Every state write is classified. Persistent-state leases are exclusive or
version-checked. KDA/convolution snapshots, next-state buffers, KV block
reservations, block tables, journals, and rollback metadata are represented in
the transaction plan and memory report. A delayed metadata commit is not
claimed to undo an unshadowed GPU write.

### Exactness

For its selected schedule, the executable knows the complete logical allocation
recipe for declared Effect Torch-controlled device-visible resources:

- every declared tensor, scratch, staging, output, and state-transaction
  allocation;
- every segment capacity, alignment, and memory space;
- every value's segment and offset;
- every external, persistent, state, provisional-output, and donated location;
- how many workspace instances one invocation requires.

This does not claim exact total process or driver memory. Pipeline compilation,
Metal driver metadata, backend libraries, command buffers, and allocations by
other processes remain outside the tensor plan. Runtime admission preserves
measured headroom for those costs. Reports distinguish logical planned
capacity, actual leased capacity, executable-owned persistent bytes, referenced
external bytes, provisional output/transaction bytes, and opaque measured
headroom. A compatible cached segment may be larger than the logical request;
that difference is reported rather than hidden.

## Workspace Ownership and Execution

### Plans do not imply permanently retained backing

Compilation produces a `MemoryPlan`; it need not allocate or permanently own
all backing segments. Retaining one complete training workspace per cached
shape would multiply device memory by cache capacity.

The runtime owns a bounded workspace pool keyed by runtime/device identity,
memory space, storage mode, hazard mode, cache mode, alignment, and capacity.
Compatible capacity may exceed the logical request, but actual leased bytes are
accounted. Segment-set acquisition is atomic: partial acquisition is rolled
back before reclamation/retry. Idle sets are best-fit reused, byte-bounded, and
LRU-evicted under pressure. Execution acquires all invocation resources:

```rust
pub struct ExecutableInstance<'a> {
    pub executable: &'a Executable,
    pub resources: InvocationResources,
}

pub struct InvocationResources {
    pub workspace: WorkspaceLease,
    pub provisional_outputs: OwnedOutputs,
    pub staging: StagingLease,
    pub state_transaction: Option<StateTransactionLease>,
}
```

Concurrent Metal execution acquires an independent segment lease and submission
context for every in-flight invocation. Contexts own their event-ordered command
buffer chain and completion result while sharing the device command queue. An
immutable executable remains concurrently callable; workspace and submission
state belong to the invocation, not the executable.

After submission, resources move into an `InFlightInvocation` retained until
every backend completion token finishes. Cancellation or a host-side error
cannot return storage to the pool while a command still references it.

### One executor

```rust
pub fn execute(
    executable: &Executable,
    invocation: Invocation,
    cancellation: &CancellationToken,
) -> Result<Vec<ConcreteValue>>;
```

Execution performs:

1. Signature and state validation.
2. Atomic acquisition of workspace, provisional outputs, staging, and state
   transaction resources.
3. Resolution of every `Location` into a concrete `BufferView`.
4. Dispatch of the fixed instruction and synchronization schedule.
5. Completion and error inspection for every submitted backend command.
6. Execution of required status gates with deterministic error precedence.
7. Transactional state commit after success.
8. Publication of already-owned escaping outputs.
9. Resource release after backend completion makes reuse safe.

Execution does not perform:

- graph reachability or consumer counting;
- fusion or other rewrites;
- kernel/algorithm selection;
- intermediate tensor allocation;
- allocation capture or replay validation;
- option-dependent schedule changes;
- interpreter fallback.

### Outputs

Values returned to TypeScript escape the invocation and cannot refer to
workspace that will be reused after the lease ends. The plan therefore assigns
them separately owned output storage, aliases them to a legally donated input,
or transfers ownership of a dedicated segment explicitly.

An immutable output view may alias a borrowed input or persistent constant when
the returned handle retains that storage owner for its full lifetime. This is
not donation and does not mutate or consume the source. If the backend cannot
export such an alias safely, lowering inserts an output-boundary
materialization. Mutable output reuse remains subject to donation rules.

Provisional output buffers exist before instruction dispatch and remain owned
by the invocation until completion. Success publishes handles; failure drops or
caches them only after backend completion. Donation of an external input is
deferred until an API can consume/invalidate that handle or the runtime can
otherwise prove unique ownership and non-aliasing.

RFC 0020 supersedes the compile-time output-generation bound. Ordinary
provisional output segments are leased independently from a runtime-wide output
pool and transfer their leases to returned tensors. A later invocation may grow
the pool within backend memory limits, but cannot overwrite a live result.
Native structural-cache hits share immutable lowering and have no per-handle
output quota.

State-transaction storage is not published as output storage. Recurrent decode
state is allocated when a sequence is created; successful CPU and Metal commits
copy transaction results into those persistent per-sequence buffers. The fixed
transaction shadow can therefore be reused after commit without pinning one
shadow generation per live sequence.

Training loss, next parameters, and next optimizer state are all declared
outputs. Donation can reduce their allocation cost only where the functional
API and caller ownership make the transfer unobservable.

### Randomness

Semantic random operations receive stable provenance identities before
optimization. Lowering, fusion, scheduling, and rematerialization preserve
those identities. Counter-based draws derive from runtime seed, invocation
nonce, semantic random identity, and element index. CSE cannot merge distinct
random sources; rematerializing one source reproduces its draw. Each invocation
receives a fresh nonce while all roots share one draw for a shared semantic
source. The runtime specifies whether failed/cancelled invocations consume a
nonce and applies that rule consistently under concurrency.

### Cancellation and failure

Cancellation stops encoding at declared safe points. Already submitted GPU work
is drained or retired according to the backend schedule before the workspace
lease can be reused. Stateful decode updates commit only after successful
completion and validation.

Status-producing instructions declare a status gate. A status may be deferred
only when the remaining schedule is memory-safe for every status value;
otherwise the plan inserts a completion/readback gate before dependent work.
The plan reserves the maximum-path memory regardless of status. Error
precedence is deterministic: binding/state validation, backend submission or
command failure, device status, cancellation, then state commit/publication.

Allocation admission failure is an execution error containing planned bytes,
current backend allocation, working-set guidance, and requested segment sizes.
The runtime may release idle workspace/cache entries and retry acquisition. It
does not switch execution models or discard the memory plan silently.

## TypeScript and Runtime API

The public tensor API remains graph-oriented. The backend contract becomes
explicitly compile/execute-oriented:

```ts
export interface RuntimeService {
  readonly compile: (
    request: CompileRequest
  ) => Effect.Effect<ExecutableHandle, BackendError>

  readonly execute: (
    executable: ExecutableHandle,
    invocation: ExecutionInvocation
  ) => Effect.Effect<ReadonlyArray<ConcreteTensorHandle>, BackendError>
}

export interface ExecutionInvocation {
  readonly bindings: ReadonlyArray<ConcreteTensorHandle>
  readonly scalars: ReadonlyArray<number>
  readonly runtimeValues: Readonly<Record<string, number | Uint32Array>>
  readonly rng?: RngInvocation
  readonly state?: ExecutionStateHandle
}
```

`Runtime.evaluate` is removed as a distinct backend execution primitive.
`Tensor.compute` parameterizes/compiles and then calls `execute`. Existing
high-level `runProgram` APIs remain useful but delegate to the same operation.

Every API that accepts lazy roots, including `Tensor.save`, must transform the
graph or delegate materialization to compile/execute. Serialization after
materialization and tensor readback are transfer services, not graph execution
engines. Transfer services declare conversion/staging storage and use the same
backend submission coordinator and completion/error tracking as executable
dispatch. Batched-decode padding and block-table/cursor staging belong to its
invocation plan rather than allocations performed before `execute`.

Backends own executable and concrete tensor handles per RFC 0017. "One path"
means one compile/execute protocol and one orchestration path per runtime, not
one universal lowered IR implementation. The shared Rust graph/compiler crates
may define neutral executable concepts, while each backend lowers to
instructions and memory spaces it can implement. Native CPU/Metal can share
planner infrastructure. A PJRT backend may delegate schedule and memory
assignment to PJRT, wrap an opaque executable, and expose only supported
diagnostics; exact native memory-plan acceptance criteria do not apply to that
opaque implementation.

## Compiler Diagnostics

Every executable exposes a stable diagnostic report:

```text
semantic nodes before/after optimization
lowered instruction count by kind
kernel/pipeline count
external, persistent, state, output, and workspace bytes
segment list: memory space, capacity, alignment
peak live bytes and packing overhead
largest values and scratch allocations
donated/aliased values
command-buffer and synchronization counts
compile phase timings
```

Debug builds can validate memory plans by:

- checking all slices against segment bounds and backend alignment;
- proving overlapping byte ranges have non-overlapping live intervals;
- poisoning dead regions where backend cost permits;
- comparing optimized and unoptimized executable results;
- running with reuse disabled while preserving the same schedule.

These checks operate on one executable IR rather than comparing different
execution engines.

## Concurrency

Executables are immutable and may be shared. Bindings, state transactions,
random streams, cancellation, and workspace leases are per invocation.

The current native Metal runtime serializes all walks around one command
manager. This RFC preserves that behavior initially and allows compatible
workspace sharing. The API must not claim actual overlapping execution merely
because the executable handle is `Send`/`Sync`.

If Metal execution later becomes concurrent, the runtime allocates multiple
workspace instances or schedules leases with GPU events. A single planned
workspace cannot be written by concurrent invocations.

## Relationship to Existing RFCs

### RFC 0003: Memory management

External-memory reporting and concrete handle lifetime remain. Runtime
consumer-counting early-free is replaced for executable intermediates by
compile-time last-use analysis. Dynamic backend caches may still recycle
escaping outputs and allocations outside executable workspace.

### RFC 0007: Kernel fusion

Fusion remains an optimization over semantic graphs and scalar expression IR,
but runs only during compilation. RFC 0021 side-table regions lower to explicit
typed instructions with storage contracts; compiler-created fused semantic
nodes are no longer used. The CPU correctness path is generic executable
lowering, not a separate per-element graph interpreter.

### RFC 0008: Compilation

Placeholder tracing, shape specialization, owned reusable programs, and
functional bindings remain. "Freeze" no longer means retaining a graph for
the evaluator; it means retaining the resulting executable. Claims that a
program holds no device storage are replaced by the workspace-lease model.

### RFC 0010: Inference and decode

Decode graph rewriting becomes an optional inference-specialization pass in the
single compiler. KV/KDA/convolution state remains persistent explicit state.
Prefill, single decode, and batched decode are separate shape-specialized
executables using the common executor and transactional state hooks.

### RFC 0016: Frozen-program memory

The measured need for liveness-based reuse, chunked head CE, and segmented
backing remains. Allocation-call capture, replay cursors, manually reported
checkpoints, one shared arena, and the claim that pool plus arena are permanent
execution modes are superseded. Chunked CE becomes an explicitly lowered
schedule with declared scratch and loop-carried lifetimes.

### RFC 0017: Multi-backend runtime

The runtime continues to own allocators, compiler caches, tensor handles, and
executables. Its ordinary materialization boundary becomes compile plus execute
rather than a backend-specific `evaluate` path. Backends remain free to
delegate compilation to systems such as PJRT while preserving opaque handles.

## Migration

Effect Torch is pre-release, so migration optimizes for one coherent endpoint
rather than preserving duplicate execution APIs.

### Phase 0: Stabilize current correctness

- Add compiled replay parity tests for semantic operations with internal
  allocations, beginning with chunked head CE forward and all gradients.
- Fix current arena liveness, upload accounting, replay-overrun, weak-slot, and
  right-sizing defects so the existing path remains a trustworthy baseline
  during migration.
- Propagate errors from every submitted Metal command buffer before output
  publication, state commit, or workspace release.
- Record allocation and execution traces used to validate the new plans.

### Phase 1: Executable shell and common executor

- Introduce `ProgramRequest`, `Executable`, `InferenceOptions`, and one
  `execute` entry point.
- Route explicit programs, ordinary compute, lazy-root serialization, and
  decode orchestration through this entry point. The temporary shell may carry
  one graph-walk instruction until generic lowering replaces it.
- Centralize binding validation, root/output handling, synchronization,
  deferred checks, cancellation cleanup, diagnostics, and state commit.
- Route readback/conversion through the same backend submission coordinator.
- Keep the existing node walk only inside the temporary instruction; do not
  expose it as a permanent alternate mode.

### Phase 2: Generic lowering

- Define backend executable IR and value identities.
- Lower every supported `NodeKind` to generic instructions.
- Topologically schedule lowered instructions into dense integer-indexed arrays;
  graph/SSA structures remain compiler-only.
- Replace evaluator side maps for multi-output operations with explicit
  instruction outputs.
- Emit views as aliases.
- Add a lowering/support matrix covering every `NodeKind`, backend, dtype, and
  layout constraint.
- Make `optimize: false` a complete executable path.

At the end of this phase, remove production `eval_node`/`eval_uncached` graph
execution. Unsupported nodes fail compilation.

### Phase 3: Destination and scratch contracts

- Convert kernels from allocate-and-return to encode-into-destination APIs.
- Declare scratch for GEMM, attention, KDA, convolution, normalization, loss,
  indexing, and optimizer instructions.
- Define the invocation ABI for scalars, RNG, decode cursors/context lengths,
  block tables, token advances, padding, statuses, and staging.
- Add owned provisional output destinations and explicit state transaction
  plans.
- Lower fixed-topology loops such as chunked head CE into explicit nested
  schedules.
- Compile every required kernel/pipeline during compilation, not first execute.
- Reject ambient intermediate allocation while encoding an executable.

### Phase 4: Static schedule and memory plan

- Compute instruction last uses, aliases, and escapes.
- Assign aligned slices across right-sized bounded segments.
- Add physical reuse edges, then derive command-buffer boundaries and
  synchronization from semantic plus reuse dependencies.
- Finalize backend-specific linear command streams after physical hazard
  planning.
- Add workspace admission, leases, diagnostics, and plan validation.
- Allocate provisional outputs and state shadows before dispatch; keep donation
  disabled until ownership is provable.
- Switch compiled training and inference to planned storage.

At the end of this phase, delete allocation capture/replay and arena TLS.

### Phase 5: Make all materialization compiled

- Add structural transient caching and concrete-leaf parameterization to the
  already-common `Tensor.compute` path.
- Remove `Runtime.evaluate` from the backend contract.
- Remove the temporary `Runtime.evaluate` adapter, graph-walk instruction, and
  dead allocator/evaluator code left after generic lowering.
- Retain dynamic allocation only for concrete external storage, executable
  workspace acquisition, output ownership, and backend services outside tensor
  execution.

### Phase 6: Memory-aware optimizations

- Add legal input/output donation.
- Allow memory-budgeted rematerialization and algorithm selection.
- Benchmark standalone Metal segments against automatic and placement heaps.
- Add private intermediate storage only with explicit staging and measured
  benefit.

## Implementation Status (2026-08-10)

The unified CPU and Metal executor, static device-visible memory plans,
destination-oriented kernels, bounded structural caches, runtime-owned
workspace/output leases, and transactional KV/KDA/convolution decode are
implemented. Inference construction eagerly compiles prefill, single-decode,
batched-decode, and logits row-extraction programs; token steps do not trace or
compile extraction graphs. Consumed logits and transient full decode outputs are
released deterministically.

Current verification:

- CPU native tests: 114 passed;
- Metal native tests: 125 passed;
- core CPU/Metal tests: 664 passed;
- release nano-GPT: 400 training steps in 2.1 seconds and all prompt generations
  completed;
- release FineWeb: approximately 0.544-0.546 seconds per steady-state step,
  faster than the historical approximately 0.569 second intermediate baseline
  (this predates the RFC 0020 Metal submission-context change);
- release FineWeb-KDA: approximately 0.731 seconds per steady-state step.

A fresh 20-step release FineWeb run reported 3,253,010,432 bytes maximum RSS
and a 74,225,184 byte macOS peak-memory footprint. Unified CPU/GPU mappings
inflate RSS on Apple silicon, so both operating-system figures are retained.

The backend-storage criterion is met after pool warm-up: compatible workspace and
released output segments are reused without CPU segment or Metal buffer
allocation. A call retaining additional live outputs may grow the runtime output
pool instead of failing at an executable-local capacity. Literal
host-heap-allocation freedom is not yet met. The current audit identifies these
remaining Effect Torch-controlled allocations:

- per-invocation owner, resolved-value, command-input, scratch, staging, status,
  and result vectors in both executors;
- deep `Layout` clones because shape and stride metadata still use owned vectors;
- operation metadata recomputed by several Metal destination paths;
- stateful NAPI transaction, padding, lock, and checkpoint vectors;
- async NAPI cancellation/task frames and fresh native/JavaScript output handles.

Metal command buffers, encoders, and driver bookkeeping are opaque platform
allocations and cannot be claimed as statically planned. Fresh output handles are
publication metadata rather than tensor backing, but they remain allocations and
must be explicitly classified in the final acceptance boundary. Until reusable
host invocation frames are preallocated and the publication/driver exceptions
are finalized, this RFC remains `Draft` and must not claim zero total dynamic
allocation.

### RFC 0021 update (2026-08-11)

RFC 0021 completed the compiler representation used by this executable model.
One `ProgramRequest` prepares one `ProgramSignature` and `GraphIndex`; compiler
optimizations are side-table regions over the nongeneric semantic graph. CPU
and Metal now retain authoritative typed `LoweredProgram` values and
instructions whose exact input, output, scratch, staging, status, and state
resources feed liveness, memory planning, diagnostics, and execution.

The backend physical plans reference lowered `InstructionId`s. CPU physical
entries are encodes; Metal adds status gates, commits, and completion boundaries.
There is no independently copied dense tensor-command array or string-valued
planning schedule. `ProgramSignature` validates binding/scalar/runtime/RNG and
output contracts, while decode's scalar or batched state cursor is represented
as a bounded runtime-value contract derived during shared specialization.
Inference-only constant weights remain implemented. The unused executable
precision option was removed; Trainer mixed-BF16 remains separate.

## Acceptance Criteria

### Architecture

- `Tensor.compute`, explicit reusable programs, compiled trainer steps, held-out
  evaluation, lazy-root serialization, prefill, single decode, and batched
  decode all call one executor.
- Readback and serialization transfer work use the same submission coordinator
  and completion/error propagation as executable dispatch.
- No production path executes a semantic graph directly.
- A native executable dispatches typed lowered instructions through a physical
  plan of dense `InstructionId` references; invocation performs no graph
  reachability, topological sort, consumer counting, or last-use traversal.
- Production native executable handles do not retain semantic graph roots needed
  for execution; optional source maps are diagnostics-only.
- Fusion and kernel selection do not occur during execute.
- No kernel source or pipeline is compiled during execute.
- Optimizations-disabled and optimized programs use the same executable IR and
  memory planner.
- Unsupported operations fail compilation without silent fallback.
- The lowering matrix has an explicit result for every supported
  `NodeKind x backend x dtype x layout` combination.

### Memory

- Every declared Effect Torch-controlled intermediate, scratch, staging,
  provisional output, and state-transaction allocation appears in the
  executable memory report.
- A debug allocation guard observes no unplanned intermediate allocation while
  encoding or executing a planned program.
- Executable diagnostics expose logical output bytes per invocation, and
  retained-output tests prove later invocations neither overwrite nor invalidate
  eight simultaneously live results.
- After warm-up, reusable host invocation frames cover executor-owned owner,
  value-resolution, command-argument, status, and transaction metadata; any
  unavoidable output-publication or platform-driver allocation is classified
  explicitly rather than included in a zero-allocation claim.
- Segment capacities are right-sized, backend-aligned, and individually within
  device limits.
- Actual leased capacity, logical planned capacity, referenced external bytes,
  provisional output/transaction bytes, and opaque headroom are reported
  separately.
- A program with at most 1 MiB of logical workspace leases no segment larger
  than that plan plus documented backend alignment/rounding.
- The first execution does not hold a capture working set plus a newly built
  replay arena.
- Fixed memory-planner fixtures prove no overlapping address assignment for
  overlapping live intervals and deterministic reuse for disjoint intervals.
- Workspace-pool tests compile many incompatible plans and prove idle capacity
  returns below the configured global byte limit after pressure/LRU eviction.
- FineWeb training at `FINEWEB_BLOCK=4096`, `FINEWEB_BATCH=64` runs without a
  single oversized Metal allocation and reports planned versus external memory
  separately.

### Correctness

- Optimized versus unoptimized executable parity follows the lowering matrix;
  exact operations are bitwise-equal and floating operations use per-operation
  tolerances recorded by the existing backend contract tests.
- Training parity covers loss, every parameter gradient, next parameters, and
  optimizer state across repeated steps.
- Chunked head CE parity covers forward and all backward outputs under forced
  chunking and memory reuse.
- Random operations draw freshly per invocation and remain shared consistently
  across multiple roots.
- Cancellation and backend errors cannot release workspace before submitted
  GPU use completes.
- Injected failure in a non-final command buffer is reported before outputs are
  published or state commits.
- Failure injection after every stateful decode instruction preserves cursor,
  hashes, KV block ownership, KDA state, and convolution state according to the
  transaction plan.
- Compile-time validation errors and execution-time binding/status errors occur
  in their specified phase with stable error categories.

### Performance

- Compile phase timings and cache hit rates are visible.
- Cached `compute` and explicit-program execution have equivalent steady-state
  schedules and allocation behavior.
- On the same release build and machine, the median of at least 20 FineWeb
  steady-state steps does not regress by more than 5% from the pre-migration
  baseline; planned packing overhead is reported against schedule peak live
  bytes and tracked as a benchmark metric.
- Executable cache eviction releases metadata and does not pin one complete
  workspace per cached shape.

## Alternatives Considered

### Keep an interpreter as a correctness fallback

Rejected. It duplicates execution, allocation, synchronization, status, random,
and error semantics. An unoptimized executable is the correctness baseline.

### Keep ordinary evaluation and frozen execution separate

Rejected. The TypeScript API always builds a graph, so ordinary materialization
can compile a transient or cached executable. Separate paths add no semantic
capability.

### Keep capture/replay arena planning

Rejected as the endpoint. Capture sees hidden allocations but learns their
lifetimes through evaluator implementation details and manual checkpoints. It
requires a full first execution, can double warm-up memory, and detects only
some forms of replay divergence. Explicit lowering provides the same knowledge
before execution.

### Use only a dynamic caching allocator

Rejected for executable intermediates. Split/coalescing caches remain useful
for external and output storage, but a static schedule can pack lifetimes more
deterministically and with less fragmentation. Dynamic allocation is not a
substitute for compile-time workspace contracts.

### One whole `MTLBuffer` per reusable value slot

Rejected as the default plan. It avoids suballocation aliasing but cannot split
a dead large slot among multiple simultaneous smaller values, increasing
reserved memory. It remains a useful diagnostic mode.

### One giant arena buffer

Rejected. Aggregate free unified memory does not imply that Metal can create
one resource of that size. It also makes residency, purgeability, and failure
recovery unnecessarily coarse. Planned storage is segmented.

### Require `MTLHeap`

Rejected. Heaps are a backing option, not a lifetime analysis. Placement and
aliasing introduce alignment, hazard, ownership, and accounting requirements.
Start with standalone segments and adopt heaps only after measurement.

### Delegate native Metal compilation to XLA

Rejected. There is no supported XLA Metal backend. Future PJRT runtimes remain
valid implementations of the RFC 0017 backend contract for supported devices.

## Risks and Open Questions

1. **Lowering surface**: every currently evaluator-implemented operation needs
   generic lowering before the interpreter can be removed.
2. **Workspace contracts**: native kernels currently allocate internal values;
   destination-oriented APIs are a broad but mechanical refactor.
3. **Compile latency**: scheduling and planning add work to one-off `compute`.
   Structural caching and a simple generic lowering path must keep this bounded.
4. **Concrete-leaf parameterization**: cache reuse must not retain arbitrary
   user tensors or accidentally treat mutable identity as a constant.
5. **Output donation**: the pure API makes legal ownership transfer explicit
   but may limit donation without linear/unique ownership evidence.
6. **Memory-budget feedback**: algorithm selection affects scratch and
   scheduling. The initial compiler chooses algorithms first and reports the
   resulting plan; iterative budget-aware selection follows later.
7. **Backend-neutral versus backend-specific IR**: scheduling concepts should
   be shared, but forced uniformity must not hide Metal, CPU, or PJRT semantics.
8. **Command-buffer completion implementation**: every submitted command must
   contribute to one invocation completion/error token; implementing this
   without unnecessary host waits is required before planned workspace ships.
9. **Concurrent execution**: one immutable executable can be called from many
   fibers, but actual overlap requires independent workspace instances and
   backend queue semantics.
10. **Plan verification**: liveness errors can silently corrupt results. The
    compiler needs structural overlap checks and aggressive debug validation,
    not only end-to-end loss comparisons.
11. **Bounded decode capacity**: worst-case block-table, padding, and state
    transaction storage can exceed useful per-step bytes; schemas and reports
    must make that reserved capacity visible.
12. **Host allocation boundary**: backing-buffer guards do not detect Rust
    `Vec`/`Box`/`Arc` control-block allocation, NAPI task frames, JavaScript
    output objects, or opaque Metal driver objects. Acceptance requires reusable
    executor-owned host frames plus an explicit exception boundary for output
    publication and platform-owned submission metadata.

## References

- RFC 0003: `docs/rfcs/0003-memory-management.md`
- RFC 0007: `docs/rfcs/0007-kernel-fusion.md`
- RFC 0008: `docs/rfcs/0008-compilation.md`
- RFC 0016: `docs/rfcs/0016-frozen-program-memory.md`
- RFC 0017: `docs/rfcs/0017-multi-backend-runtime.md`
- PyTorch MPS allocator:
  https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/mps/MPSAllocator.mm
- PyTorch CUDA caching allocator and graph-private pools:
  https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp
- MLX Metal allocator:
  https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/allocator.cpp
- XLA buffer assignment:
  https://github.com/openxla/xla/blob/main/xla/service/buffer_assignment.cc
- Metal `maxBufferLength`:
  https://developer.apple.com/documentation/metal/mtldevice/maxbufferlength
- Metal heap sizing and alignment:
  https://developer.apple.com/documentation/metal/mtldevice/heapbuffersizeandalign(length:options:)
- Metal heap aliasing and fences:
  https://developer.apple.com/documentation/metal/implementing-a-multistage-image-filter-using-heaps-and-fences

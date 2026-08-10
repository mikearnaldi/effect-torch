# RFC 0020: Invocation Ownership - Execution Frames, Runtime Pools, and Output Lifetimes

- **Status**: Partially implemented - ordinary output ownership, runtime pools,
  shared plans, and concurrent CPU frames are complete; host-frame pooling,
  overlapping Metal execution, donation, and explicit inference contexts are
  deferred.
- **Created**: 2026-08-10
- **Depends on**: RFC 0003 (memory management), RFC 0008 (compilation),
  RFC 0010 (inference), RFC 0017 (multi-backend runtime), RFC 0019
  (executable compilation)
- **Updates**: RFC 0019 output ownership, resource ownership, and concurrency

## Summary

Separate four concepts that are currently represented by one executable object:

1. The **compiler** decides logical value lifetimes, aliases, and offsets.
2. The **runtime allocator** supplies and recycles physical backing storage.
3. An **invocation frame** owns resources needed by one in-flight execution.
4. A returned **tensor handle** owns or retains the storage of an escaping
   output.

An ordinary compiled function is an immutable, concurrently callable plan. Each
call leases an independent invocation frame. Workspace returns to the runtime as
soon as backend work completes; output storage transfers to returned tensors and
returns only after those tensors die or are explicitly cleared. A later call
must never overwrite a still-live ordinary output.

This removes `outputCapacity` from ordinary compilation semantics. Retaining
many outputs may consume memory and eventually produce a normal allocation or
budget error, but it does not exhaust a ring embedded in one compiled function.
Limits on contexts or frames constrain work that is simultaneously in flight,
not the number of completed outputs that may remain live.

Specialized ownership remains explicit:

- training may donate parameters and optimizer state;
- inference may reserve contexts and bind caller-owned outputs;
- generation owns a separate paged KV/recurrent-state pool;
- a future fixed-address capture/replay API may expose overwrite restrictions,
  but those restrictions do not apply to ordinary `Tensor.compile` or
  `Tensor.compute`.

## Decision

Effect Torch adopts the following ownership model:

| Concern | Owner | Reuse boundary |
| --- | --- | --- |
| Lowered commands and memory layout | `ExecutablePlan` | Shared indefinitely |
| Intermediate and scratch backing | Invocation frame | Backend completion |
| Host command metadata | Invocation frame | Backend completion/publication |
| Ordinary output backing | Returned tensor handle | Handle release/clear |
| Donated input backing | Receiving invocation/output | Explicit ownership transfer |
| Caller-bound output backing | Caller | Caller-defined |
| KV/KDA/convolution state | Generation session/pool | Sequence/session release |
| Fixed replay addresses | Explicit replay context | Replay API contract |

The reusable plan contains no mutable per-call storage. Runtime pools may cache
idle memory, but cache residency is not ownership by an executable and does not
change tensor lifetime semantics.

RFC 0019 remains authoritative for semantic optimization, lowering, executable
IR, liveness, destination-oriented kernels, synchronization, and transactional
execution. This RFC supersedes its fixed `output_capacity` generations and
clarifies that all ordinary workspace, staging, output, status, and host-frame
resources are invocation- or tensor-owned rather than executable-owned.

## Motivation

### The current design conflates independent concerns

RFC 0019 correctly assigns intermediate lifetimes and offsets statically. The
current implementation then allocates workspace, state-transaction segments,
and a fixed ring of provisional output segments inside each native executable.
`outputCapacity` determines the number of output generations allocated during
compilation. A call fails when every generation is still retained by a previous
result.

That arrangement provides bounded post-compile backing allocation, but it also
makes an implementation strategy observable as function semantics:

```text
f = compile(x => -x, { outputCapacity: 2 })

a = f(x0)
b = f(x1)
c = f(x2) // fails while a and b remain live
```

Nothing in the pure function, its shapes, or the available runtime memory
implies this two-result limit. Compiling the same graph twice creates two
unrelated limits. Changing a cache key changes how many valid results callers
may retain. These are properties of an executable-local ring, not properties of
the computation.

The same embedding also obstructed real concurrency. Before this RFC, one
executable contained mutable or reusable invocation resources, while CPU and
Metal protected execution with process-wide locks. Calls were thread-safe only
because they were serialized. Making the command scheduler concurrent without
changing ownership would have allowed invocations to write the same workspace
and transaction storage.

### Static planning does not require static physical ownership

A compiler can prove that values `a` and `b` have disjoint live intervals and
assign both offset zero without deciding which physical allocation will back
offset zero on every call. The plan is logical:

```text
segment 0: 32 MiB, alignment 256, workspace
  value a: offset 0, live [command 1, command 4]
  value b: offset 0, live [command 5, command 8]
```

At execution time, any compatible exclusive 32 MiB segment can instantiate
that plan. Concurrent calls need distinct segments; sequential calls can lease
the same segment. The plan remains identical in both cases.

Escaping outputs are different. Their lifetime is not bounded by the command
schedule because it extends into user code. Static liveness can determine an
output's size and legal aliases, but only handle lifetime determines when its
storage becomes reusable.

### The established split is consistent across runtimes

Production systems vary in API and allocator details, but broadly converge on:

| Concern | Common solution |
| --- | --- |
| Intermediate layout | Static liveness and offset planning |
| Physical memory | Runtime-wide caching allocator or arena |
| Concurrent execution | One frame/context per in-flight call |
| Escaping outputs | Independently owned or caller-bound buffers |
| Training updates | Mutation, donation, or explicit ownership transfer |
| Inference state | Dedicated KV/state pools managed separately |

The unusual part of the current Effect Torch design is not static planning or
preallocation. It is combining immutable compilation, invocation workspace, and
escaping output storage in one executable instance.

## Goals

- Preserve pure tensor semantics: every successful ordinary result remains
  valid until its handle is released.
- Preserve RFC 0019's static liveness, aliases, offsets, and allocation-free
  intermediate execution after warm-up.
- Allow one immutable executable plan to serve sequential and concurrent calls.
- Make memory limits runtime properties with coherent accounting and pressure
  behavior.
- Make storage-changing optimizations explicit through donation, output
  binding, state pools, or specialized replay APIs.
- Keep generation capacity separate from ordinary output lifetime.
- Make CPU, Metal, and future remote runtimes implement the same ownership
  contract even when their scheduling capabilities differ.
- Retain deterministic cleanup and transactional state behavior across success,
  failure, and cancellation.

## Non-goals

- Guarantee unlimited memory or that every invocation succeeds under pressure.
- Guarantee overlapping execution on backends that currently serialize work.
- Require one physical allocator implementation for every storage class.
- Expose allocator internals or native pointers through the public tensor API.
- Make donation mandatory for ordinary tensor operations.
- Replace the paged KV architecture with general output allocation.
- Claim that backend drivers, NAPI, or JavaScript publication perform no host
  allocation.
- Specify a public CUDA Graph-like API before a backend needs one.

## Semantic Invariants

The following are observable requirements, independent of backend strategy.

### Ordinary outputs do not expire

After a successful ordinary invocation, each returned tensor remains readable
and unchanged until one of these events:

- the caller performs an explicit mutable operation authorized by the API;
- the caller donates or consumes the tensor;
- the caller clears/releases the tensor and stops using the handle;
- the owning runtime or process is closed under its documented shutdown rules.

Executing the same or another program is not such an event. A runtime must not
reuse output backing while a reachable tensor can still observe it.

### Invocation state is exclusive

Mutable workspace, staging, command metadata, status words, random invocation
state, and transaction shadows belong to one invocation from admission through
backend completion. They cannot be shared by overlapping invocations unless
the backend proves read-only access or protects disjoint ranges with completion
events.

### Plans are immutable and shareable

An `ExecutablePlan` can be cached structurally and shared by any number of
public executable handles. Sharing a plan does not share mutable invocation
resources or impose independent output quotas on those handles.

### Reuse follows ownership

A block becomes reusable only after its current owner releases it and all
backend work that references it has completed. Refcounts, explicit leases,
completion tokens, or equivalent backend mechanisms enforce this rule. A
planner's last-use command is sufficient for intermediates inside one frame; it
is not sufficient for escaping outputs.

### Bounds fail before corruption

Frame limits, runtime memory budgets, generation pool capacity, and replay-slot
capacity may reject or queue work. They must never be resolved by overwriting a
live ordinary tensor, reusing an in-flight frame, or partially committing
persistent state.

## Prior Art

### PyTorch: planned intermediates, cached backing, mutable training state

PyTorch's caching allocators retain released blocks for later allocations while
live tensors retain their own storage. Inductor contains a static memory
planning pass for generated programs, but ordinary compiled outputs continue to
follow tensor ownership rather than a fixed per-function output ring.

Typical PyTorch optimizers update parameters and optimizer state in place. The
large, persistent training state therefore does not require a fresh independent
copy on every step. Explicit in-place and `out=` operations make storage reuse
visible when user code requests it.

Relevant sources:

- [Inductor memory planning][pytorch-inductor-memory]
- [PyTorch CUDA memory management][pytorch-cuda-memory]

### JAX/XLA: static assignment plus explicit donation

XLA performs static buffer assignment and may alias compatible values according
to liveness and ownership constraints. JAX keeps a functional surface while
letting callers explicitly donate buffers:

```python
update = jax.jit(update_fn, donate_argnums=(0, 1))
params, optimizer_state = update(params, optimizer_state)
```

Donation promises that the old arguments will not be used after dispatch. The
compiler may reuse their backing for compatible outputs, and JAX invalidates the
donated inputs. Donation is permission rather than a guarantee that every byte
will alias.

JAX's GPU allocator can preallocate a large process-wide region and reuse it.
That allocator policy controls physical backing and fragmentation; it does not
alter the lifetime of live array values.

Relevant sources:

- [JAX buffer donation][jax-donation]
- [JAX GPU memory allocation][jax-gpu-memory]
- [XLA buffer assignment][xla-buffer-assignment]

### TensorFlow: resource variables and pooled allocation

TensorFlow represents parameters and optimizer state with mutable resource
variables while ordinary tensors remain reference-counted values. Its BFC
allocator caches, splits, and coalesces released blocks. XLA compilation can add
static assignment for internal values. Persistent mutable state, static
intermediate planning, and runtime backing allocation remain separate concerns.

Relevant source:

- [TensorFlow/XLA BFC allocator][tensorflow-bfc]

### TensorRT: engine versus execution context

TensorRT provides the closest architecture for compiled inference. An
`ICudaEngine` is the compiled, shareable plan. An `IExecutionContext` contains
per-inference activation state. Multiple contexts from one engine can execute
concurrently; concurrent use of one context is not the model. Context device
memory may be runtime-managed or supplied by the application, and concurrently
running contexts require non-overlapping activation storage.

Inputs and outputs are bound by address. Output lifetime is therefore controlled
by the application rather than by an output ring hidden in the engine.

Relevant source:

- [TensorRT runtime memory and threading][tensorrt-architecture]

### ONNX Runtime: session, execution frame, arena, and I/O binding

ONNX Runtime shares an optimized `InferenceSession`, creates per-run execution
state, and obtains backing from runtime allocators/arenas. Shape-specific memory
patterns can consolidate activation allocation and assign internal offsets
without making completed outputs aliases of later runs.

I/O binding allows callers to provide device-resident input/output storage.
When output shape is not known, ONNX Runtime can allocate the output on the
target device. This separates compilation, per-run state, allocation policy,
and output ownership.

Relevant sources:

- [ONNX Runtime I/O binding][ort-io-binding]
- [ONNX Runtime execution frame][ort-execution-frame]
- [ONNX Runtime inference session][ort-inference-session]

### General array APIs: reuse is explicit

PyTorch, JAX, TensorFlow, NumPy, and MLX ordinarily return values whose storage
remains valid while the value is live. Released storage may enter an allocator
cache. Holding many values consumes real memory and may eventually produce an
out-of-memory error, but ordinary APIs do not impose a per-function count of
live results.

When callers want a specific storage identity, the contract is explicit:

| Library/runtime | Explicit mechanism |
| --- | --- |
| JAX | `donate_argnums` |
| NumPy | `out=` |
| PyTorch | in-place operations or `out=` |
| TensorRT | caller-bound tensor addresses |
| ONNX Runtime | I/O binding |
| TensorFlow | mutable resource variables |

MLX separately reports active and cached memory and exposes cache and memory
limits. This is the accounting model Effect Torch needs: cached free storage is
not live tensor storage, even if both contribute to process residency.

Relevant source:

- [MLX memory management][mlx-memory]

### vLLM: generation state is a scheduled pool

Autoregressive generation has different ownership from a pure function call.
Model weights are shared, while KV blocks persist across many decode steps for a
request. vLLM preallocates paged KV storage and schedules requests into model
steps, including continuous batching. Requests return tokens or text rather
than tensor views into reusable activation workspace.

Effect Torch already follows the main state-ownership direction through a paged
KV pool and sequence-owned KDA/convolution state. A future serving scheduler can
batch ready sequences without moving this state into ordinary executable
outputs.

Relevant source:

- [vLLM architecture][vllm-architecture]
- [PagedAttention paper][paged-attention]

### CUDA Graphs: the explicit fixed-address exception

CUDA Graph replay requires stable virtual addresses. PyTorch documents that
some CUDA Graph Tree outputs can be overwritten by a later replay and that
callers must clone outputs that need to survive. This is a specialized mode
with restrictions caused by fixed-address capture.

The current Effect Torch output-generation ring resembles this restriction
applied to every compiled function, despite ordinary CPU/Metal execution not
requiring captured output addresses. The lesson is not that overwrite semantics
are forbidden; it is that they require a distinct API and cannot silently
replace normal tensor ownership.

Relevant source:

- [PyTorch CUDA Graph Trees limitations][pytorch-cudagraph-limitations]

## Architecture

### Immutable executable plan

Compilation produces a shareable plan containing only compile-time artifacts:

```rust
pub struct ExecutablePlan {
    pub signature: ProgramSignature,
    pub commands: Box<[Command]>,
    pub values: Box<[ValueDecl]>,
    pub memory: MemoryPlan,
    pub pipelines: PipelineSet,
    pub randomness: RandomPlan,
    pub diagnostics: PlanDiagnostics,
}
```

The runtime owns the atomic invocation sequencer and combines its nonce with the
plan's immutable random identities. Pipeline cache objects and immutable
constant/weight references may be shared. Persistent mutable sequence state,
workspace buffers, output buffers, status words, and host vectors are not fields
of the plan.

The structural compiler cache stores `Arc<ExecutablePlan>`. A cache hit returns
another reference to the same immutable plan; it does not clone commands merely
to manufacture a second resource ring.

### Runtime resource pools

The runtime owns allocation and idle-cache policy. The logical interfaces are:

```rust
pub struct RuntimePools {
    pub workspace: WorkspacePool,
    pub outputs: OutputPool,
    pub host_frames: HostFramePool,
}
```

These may share one physical backend allocator. They remain distinct logical
classes because their lease boundaries and diagnostics differ.

#### Workspace pool

The workspace pool supplies segments for intermediates, scratch, invocation
staging, status, and transaction shadows. Requests are derived entirely from
the plan. Compatible idle segments can back different plans at different times.

The pool:

- keys allocations by runtime/device, memory space, storage/hazard/cache mode,
  alignment, and capacity class;
- acquires all required segments atomically or rolls back a partial set;
- uses best-fit or equivalent reuse to control fragmentation;
- tracks active and idle bytes separately;
- evicts idle allocations under cache pressure;
- never evicts an active lease;
- retains a lease through the last backend completion event that references it.

The existing generic `WorkspacePool` is the foundation. CPU must route
production execution through it, and Metal's currently test-only workspace pool
must replace executable-owned fixed backing.

#### Output pool

The output pool supplies storage for ordinary escaping values. It may be the
same physical caching allocator as workspace, but each output allocation moves
into a lease retained by the concrete tensor owner.

```rust
pub struct OutputLease {
    allocation: Allocation,
    pool: Weak<OutputPoolInner>,
}
```

Dropping the last owner returns the allocation to the idle cache after backend
completion. An output allocation can be reused for any compatible later output,
not just for another call to the same plan.

An idle-cache byte limit controls retention of free blocks. An optional hard
runtime budget controls active plus cached allocation. If satisfying a new
output requires exceeding the hard budget, the runtime evicts idle blocks and
then returns a typed memory-pressure error if capacity remains unavailable. It
does not search for a still-live result to overwrite.

#### Host frame pool

The host frame pool stores reusable invocation metadata whose maximum lengths
are known from the plan:

- resolved owner/value tables;
- command input/output references;
- scratch and staging descriptors;
- status and deferred-check records;
- state-transaction metadata;
- result publication descriptors.

Pool entries retain vector capacities and reset lengths/content before reuse.
Opaque driver command objects, async NAPI task frames, and JavaScript/native
handle publication may still allocate. Diagnostics and zero-allocation claims
must classify those boundaries rather than treating tensor-backing allocation
as proof of zero total host allocation.

### Invocation frame

One call acquires one frame:

```rust
pub struct InvocationFrame {
    pub workspace: WorkspaceLease,
    pub provisional_outputs: Vec<OutputLease>,
    pub host: HostFrameLease,
    pub state_transaction: Option<StateTransactionLease>,
    pub completion: CompletionSet,
}
```

`provisional_outputs` are output-owned from allocation onward but are not
published until execution succeeds. On success, they move into returned tensor
handles. On failure, they return to the allocator only after submitted work is
complete. They never become general workspace.

Frame acquisition is an admission decision. A runtime may queue, reject, or
allocate another frame according to concurrency and memory policy. A frame
count is not an output count: completion returns the frame even while its
published outputs remain live.

### Execution lifecycle

Ordinary execution follows:

```text
validate invocation
  -> reserve nonce and admission
  -> acquire host frame and workspace set
  -> acquire/bind/donate provisional outputs
  -> resolve planned locations
  -> encode/execute fixed commands
  -> await or retain all backend completion tokens
  -> inspect errors and deferred checks
  -> commit persistent state transaction
  -> publish output owners
  -> release workspace and host frame
```

Cancellation or failure follows the same ownership order. Submitted work is
drained or retired before any referenced lease returns. Persistent state commits
only after success. Provisional ordinary outputs are never published on error.

### Output ownership modes

Every output declaration selects one of the following modes during lowering or
invocation validation.

#### Owned output

The runtime obtains independent storage and transfers its lease to the returned
tensor. This is the default for ordinary `Tensor.compute`, `Tensor.compile`, and
model calls.

#### Retained immutable alias

An output may be a view of an immutable input or constant when the returned
handle retains the source owner and the layout is exportable. This does not
consume or mutate the source. If the backend cannot retain the alias safely,
lowering materializes an owned output.

#### Caller-bound output

An explicit API may bind an output to caller-owned storage. Validation checks
runtime/device identity, dtype, shape, layout, alignment, writable range, and
illegal overlap before dispatch. The caller controls lifetime and must keep the
storage alive through backend completion.

Caller binding is useful for low-latency inference, interoperability, and
application-managed double buffering. It is never inferred merely because an
input has a compatible size.

#### Donated output

An invocation may consume an input and authorize compatible output aliasing.
Donation is explicit and invalidates the donated input at ownership transfer.
The planner may reuse the backing only when:

- runtime/device and memory-space ownership match;
- size, alignment, dtype, and layout constraints are compatible;
- no non-donated alias can observe mutation;
- all reads of the old value precede the first output write;
- backend completion and failure cleanup remain safe.

Donation is permission, not a guarantee of zero allocation. Ownership transfers
after validation when the invocation is admitted; a later execution failure does
not make the caller's old handle valid again. If reuse is not legal, the runtime
may allocate a separate output while still honoring the documented consume
semantics, or reject an API that promises mandatory aliasing.

#### Fixed-address replay output

A future capture/replay mode may require stable output addresses. It must use an
explicit replay context, scoped result, or caller-bound storage. If replay can
overwrite an earlier result, that result must not masquerade as an ordinary
immutable tensor. The API documents invalidation and requires cloning or binding
for persistence.

### Public API direction

`outputCapacity` is removed from `ExecutableCompileOptions`. It is neither a
semantic property of the graph nor a compiler optimization control.

Resource policy belongs to the runtime:

```ts
interface RuntimeMemoryPolicy {
  readonly maxAllocatedBytes?: number
  readonly maxIdleBytes?: number
  readonly maxInFlight?: number
}
```

The exact public shape may differ by runtime, but the meanings are fixed:

- `maxAllocatedBytes` is a hard admission/allocation budget, not an overwrite
  policy;
- `maxIdleBytes` bounds cached free storage and can be reduced without
  invalidating tensors;
- `maxInFlight` limits admitted frames or contexts, not retained outputs.

An optional prewarm operation may reserve frame/output geometries to remove
first-call allocation latency. Prewarming is an optimization hint. Calls beyond
the warmed count may allocate or wait according to runtime policy; completed
outputs do not expire when the warm set is busy.

Caller-bound outputs and donation require explicit APIs because both change
ownership assumptions. They can begin as runtime-internal facilities for
`Trainer` and `InferenceProgram` before becoming general public operations.

### Concurrency

An executable plan is safe to call from multiple tasks. Each admitted call gets
an exclusive frame. This is the same conceptual split as a TensorRT engine with
one execution context per in-flight request.

Backend capabilities determine scheduling:

- CPU may execute frames on separate worker threads once shared global state is
  removed or synchronized at finer granularity.
- Metal may encode frames independently and order reuse with command-buffer
  completion events.
- A serial backend may queue calls behind one coordinator and acquire large
  resources only when the call reaches the execution frontier.
- A remote backend may map one frame to a server request or stream.

Serialization is a valid implementation limit, but it is not executable
ownership. Diagnostics should report whether a backend is serial, concurrent,
or scheduler-batched. `Send`/`Sync` means calls are safe; it does not by itself
promise overlapping kernels.

Process-wide CPU and Metal locks must remain until frame ownership is complete.
Removing them before workspace, command metadata, allocator access, randomness,
and state transactions are independently safe is forbidden. The current
implementation reached that boundary for CPU and Metal in phase 4.

### Training

Training has large outputs that immediately become the next step's inputs:
parameters, gradients where retained, and optimizer state. Allocating independent
copies each step is unnecessary when `Trainer` owns the old state exclusively.

The trainer uses internal donation:

```text
old parameters/state
  -- consumed by compiled step -->
new parameters/state, aliasing compatible old storage where legal
```

The functional user model remains: step N returns the state for step N+1. The
runtime implementation may mutate backing only when no externally retained
handle can observe the old value. If users request a checkpoint, snapshot, or
parameter handle that survives the next step, the trainer must copy, use
copy-on-write, or disable donation for the aliased storage.

Loss and other user-retained metrics are ordinary owned outputs. Final trained
parameters escape the trainer with normal independent ownership. Donation
therefore controls steady-state update memory without imposing an output count
on general compute.

The first donation implementation may remain internal to `Trainer`. A later
general API must specify:

- which argument handles are consumed;
- when invalidation becomes observable;
- alias and view restrictions;
- behavior on compilation, admission, execution, and backend failure;
- whether reuse is permitted or required.

### Inference

Inference separates three layers:

```text
InferenceProgram
  immutable plans and shared weights

InferenceContext
  one reserved/leased invocation frame

GenerationSession
  sequence handles and persistent KV/recurrent state
```

Ordinary concurrent inference leases one context per in-flight call. An
explicit `InferenceContext` may reserve resources for predictable latency, but
one context cannot be used concurrently. Outputs remain independently owned or
caller-bound and may outlive the context invocation.

Context capacity is therefore a concurrency/admission limit. If four contexts
exist, at most four calls run concurrently; any number of completed outputs may
remain live subject to runtime memory.

### Generation

Generation state is not an ordinary function output:

- weights and executable plans are shared;
- KV blocks come from a fixed-size paged state pool;
- KDA and short-convolution state belongs to a sequence slot;
- transactions protect state updates until a model step succeeds;
- a scheduler assigns ready sequences to prefill/decode batches;
- generated tokens or independently owned logits leave the step.

The KV/state pool may have a hard sequence or token capacity because users
explicitly acquire a generation session with bounded persistent state. Capacity
errors concern new sequences/blocks, not live tensor outputs. Recurrent state
remains allocated at sequence creation and transaction shadows return after
commit.

Single-sequence and batched decode executables use ordinary invocation frames
for transient activations. They do not need `decodeBatch * 2` output generations
once consumed logits have independent leases or caller-bound scheduler buffers.

Longer term, a serving scheduler should batch ready sequences in the vLLM
style rather than treating every decode token as an unrelated concurrent model
call. This improves utilization while preserving the same state ownership.

### Randomness

Concurrent calls preserve RFC 0019's provenance-based randomness. Each admitted
invocation receives a unique nonce. The runtime specifies whether admission or
successful dispatch consumes the nonce and applies the rule consistently under
queueing, cancellation, and failure.

The random stream is not mutable scratch embedded in one executable frame.
Counter derivation must remain independent for overlapping calls even if backend
completion order differs from admission order.

### Stateful transactions

Persistent sequence state is separate from frame storage. A frame leases a
transaction shadow, executes against the shadow, and copies/swaps it into the
sequence only after all commands succeed. Failure returns the shadow after
completion without changing persistent state.

Concurrent mutation of the same sequence remains rejected or serialized by the
sequence run lock. Distinct sequences may execute concurrently or in one batch
when backend scheduling and frame ownership permit it.

## Memory Accounting

Diagnostics distinguish planned geometry, active ownership, and idle cache:

### Plan diagnostics

- logical workspace bytes;
- logical output bytes per invocation;
- transaction and staging bytes;
- peak live bytes and packing overhead;
- alignment and memory-space requirements;
- immutable constants and referenced external weights.

### Runtime pool diagnostics

- active/leased bytes and allocation count by storage class;
- idle/cached bytes and allocation count by storage class;
- high-water marks;
- allocator growth, reuse, and eviction counts;
- hard-budget and admission failures;
- opaque backend-reported or estimated memory separately.

### Invocation diagnostics

- logical requested versus actual leased capacity;
- newly allocated versus cache-reused bytes;
- queue/admission time;
- frame lease duration and backend completion duration;
- output bytes transferred, aliased, bound, or donated;
- state transaction bytes and commit/rollback outcome.

An executable diagnostic no longer reports `outputCapacity`. It reports output
requirements per invocation. Runtime diagnostics report current and peak live
output allocation globally.

Apple unified-memory reporting must continue to distinguish allocator bytes,
process RSS, and macOS physical footprint. Shared CPU/GPU mappings can inflate
RSS and should not be presented as independent resident copies.

## Failure and Pressure Policy

Memory pressure is handled in this order:

1. Reuse compatible idle allocations.
2. Evict incompatible or cold idle cache entries.
3. Allocate within the runtime hard budget and backend limits.
4. Queue when policy permits and an in-flight frame is expected to release
   sufficient workspace.
5. Return a typed allocation, budget, or admission error.

Waiting for workspace can be useful because workspace has an invocation-bounded
lifetime. Waiting for arbitrary user-held outputs is not the default because
the runtime cannot know when or whether user code will release them. It may
report live output owners and bytes in diagnostics, but must not invalidate
them.

Allocation failure occurs before dispatch whenever requirements are known. If a
backend allocation or command fails after partial acquisition/submission, the
runtime retains all referenced resources until safe completion and rolls back
persistent state.

## Implementation Status (2026-08-10)

Phases 0-2 and the CPU and Metal portions of phase 4 are implemented:

- `MemoryPlan` remains the immutable logical layout for workspace, provisional
  output, staging, and state-transaction segments.
- CPU and Metal production execution acquire workspace and transaction segments
  from runtime-wide `WorkspacePool` instances.
- Each provisional output segment receives a separate pool lease retained by
  CPU storage or the Metal buffer-owner chain. Workspace returns after execution
  while output leases remain active until the last tensor/view/readback owner
  releases them.
- `outputCapacity` and the executable-local output rings are removed from Rust,
  NAPI, TypeScript, diagnostics, cache keys, tests, and inference construction.
- Structural cache hits share one native executable plan while keeping generated
  bindings invocation-handle-local.
- CPU compiled randomness uses local nonce-derived RNG state, so calls to one
  plan can execute concurrently with independent frames; the process-wide CPU
  execution lock is removed.
- Metal executable invocations and compile-time constant construction use
  explicit submission contexts. Eager kernels use per-thread implicit contexts.
  Contexts share one command queue but own their encoder/event chain, submitted
  command buffers, failures, retirement, and completion boundary, so unrelated
  kernel streams can overlap without failure or synchronization theft.
- Every physical Metal allocation has a shared usage token retained by each
  referencing command buffer. Dynamic allocator reuse and eviction require both
  host-idle and GPU-idle storage, including aliases and unsubmitted encoders.
- Destination writes publish producer completion and persistent failure state to
  their backing storage. Executable bindings and readback fence that producer
  without consuming another invocation's completion result.
- Metal compiled randomness reserves one invocation nonce and derives command
  seeds locally instead of assigning streams from per-operation atomics.
- KV sequence release, prefix matching, and run admission linearize on the same
  per-sequence lock and recheck release after admission on CPU and Metal.
- Retained-output tests keep eight ordinary outputs live across later sequential
  and concurrent calls on CPU and Metal. Structural-cache tests retain eight
  outputs from each of two handles sharing one plan.
- Native suites pass with 114 CPU and 125 Metal tests; core CPU/Metal tests pass
  664/664. Release nano-GPT completes 400 steps and all generation paths in 2.1
  seconds. The FineWeb data fixture is absent from this worktree, so the prior
  approximately 0.545-second steady-state baseline was not rerun after enabling
  Metal submission concurrency.

Remaining work is intentionally narrower than the original mismatch:

- host owner/value/argument/status vectors are still allocated per invocation;
- runtime policies expose idle limits through backend configuration rather than
  one public `RuntimeMemoryPolicy` API, and CPU has no explicit hard byte budget;
- trainer donation, caller-bound outputs, explicit reserved inference contexts,
  and fixed-address replay remain future specialized APIs;
- NAPI/JavaScript publication and opaque backend-driver allocations remain
  outside the no-backing-allocation guarantee.

## Migration

Effect Torch is pre-release, so migration targets one coherent endpoint rather
than permanent compatibility with `outputCapacity`.

### Phase 0: Lock semantic tests

- Add a test that retains more outputs than the former default capacity and
  proves every result remains unchanged and readable.
- Add the same test through structural-cache-equivalent executable handles.
- Add concurrent-call tests that retain outputs across completion, initially
  accepting serialized backend execution.
- Preserve failure, cancellation, random-source, and state-transaction parity.

### Phase 1: Extract immutable plans

- Split backend executables into immutable plans and invocation acquisition.
- Move fixed workspace, output generations, status, and transaction resources
  out of `CpuExecutable` and Metal `ExecutableResources`.
- Make structural caches return shared immutable plans.
- Keep process-wide execution locks during this phase.

### Phase 2: Route production execution through runtime pools

- Use the generic workspace pool for CPU production execution.
- Promote Metal's workspace allocator/pool from tests to production.
- Add output leases whose owners move into concrete tensor handles.
- Make idle-cache and hard-budget configuration runtime-wide.
- Remove `outputCapacity` from native and TypeScript compile options,
  diagnostics, cache keys, and inference call sites.

### Phase 3: Pool host invocation frames

- Pre-size owner, value, argument, staging, status, transaction, and result
  metadata from plan maxima.
- Clear all references before returning a frame to the pool.
- Classify NAPI publication and platform-driver allocations explicitly.
- Re-run allocation guards with workspace, output, and host metadata reported
  independently.

### Phase 4: Enable safe concurrency

- Replace process-wide execution locks only after independent frame tests pass.
- Add backend completion-aware lease retirement.
- Prove overlapping calls never share writable ranges.
- Expose backend concurrency capability and admission metrics without promising
  overlap on serial devices.

### Phase 5: Add specialized ownership

- Implement internal trainer donation with unique-ownership checks.
- Add reserved inference contexts and caller-bound outputs where benchmarks
  justify them.
- Keep generation KV/recurrent state under session/pool ownership and add a
  scheduler independently of ordinary execution.
- Design fixed-address capture/replay only as a separate explicit API.

## Alternatives Considered

### Keep `outputCapacity` as ordinary semantics

Rejected. It bounds completed value lifetime rather than in-flight work, makes
equivalent compiled functions behave differently, multiplies memory by program
instances, and is inconsistent with ordinary immutable tensor expectations.

### Increase the default output ring

Rejected. A larger arbitrary limit delays the same semantic failure and
preallocates worst-case memory even when outputs are immediately consumed.

### Treat `outputCapacity` as a prewarm hint without renaming it

Rejected. Existing behavior is a hard live-output limit, so retaining the name
would preserve ambiguity. If prewarming is exposed, it must not affect output
validity and should be named for frames or allocation geometries.

### Put outputs in invocation workspace and retain the whole frame

Rejected as the default. It preserves correctness but pins all intermediate,
scratch, status, and host-frame capacity for the lifetime of a small result.
Dedicated output leases let workspace return immediately.

Transferring a dedicated segment is acceptable when the plan intentionally
separates it from reusable workspace.

### Allocate one permanent workspace per executable

Rejected. Cached shape-specialized plans would multiply resident memory, and
one workspace cannot support overlapping calls. Runtime-wide leasing reuses
physical memory across plans and admits concurrency explicitly.

### Use one global lock as the permanent concurrency model

Rejected as architecture. Serialization may remain a backend implementation
choice, but process-wide locking hides ownership defects, prevents overlap, and
couples unrelated runtimes/programs.

### Make all training outputs independently allocated

Correct but unnecessarily expensive. Explicit internal donation can preserve
functional API semantics while reusing exclusively owned state. Ordinary
outputs remain independent when ownership is not transferable.

### Use the general output allocator for KV state

Rejected. KV/recurrent state has sequence identity, transactional updates,
paging, prefix sharing, and scheduler lifetime. It belongs to generation
sessions, not to one returned tensor.

### Apply capture/replay overwrite rules to every compiled program

Rejected. Fixed addresses are a specialized optimization constraint. Ordinary
execution does not gain enough to justify implicit output invalidation.

## Risks

### Allocator fragmentation

Independent outputs can produce varied live lifetimes. Size classes, best-fit
reuse, alignment-aware keys, idle eviction, and active/cached diagnostics are
required. Hard budgets must fail predictably rather than cause unbounded cache
growth.

### Donation aliases observable state

Incorrect uniqueness or view analysis can mutate a tensor still visible to the
caller. Donation ships only after alias tests cover base tensors, views,
retained checkpoints, repeated arguments, failure, and asynchronous completion.

### Concurrency exposes hidden global state

Removing process locks may reveal unsafe pipeline caches, command managers,
random streams, allocator access, NAPI state, or sequence transactions. The
locks remain until race tests and backend synchronization prove each component.

### Pooling retains memory

Caching allocators improve steady state but can make process residency look like
a leak. Active and idle bytes, eviction controls, and explicit trim operations
must be observable.

### Host allocation claims remain misleading

Eliminating tensor-backing allocations does not eliminate Rust vector growth,
driver objects, async task state, or JS handle publication. Acceptance criteria
separate controlled backing allocation from opaque/platform host allocation.

## Acceptance Criteria

### Semantics

- An ordinary compiled function can produce and retain more outputs than the
  former `outputCapacity` without invalidation or a ring-capacity error.
- A later invocation never changes an earlier live ordinary output.
- Structural-cache sharing does not create or partition live-output quotas.
- Output aliases retain their owner; donated inputs are explicitly invalidated;
  caller-bound outputs validate ownership and layout.
- Memory pressure produces a typed error or admission wait, never overwrite.

### Ownership

- Backend executable plans contain no mutable invocation workspace, staging,
  transaction, status, host-frame, or output-generation storage.
- Every in-flight invocation has an exclusive frame or disjoint event-protected
  ranges.
- Workspace and host frames return after backend completion even when outputs
  remain live.
- Output backing returns only when its last tensor owner releases it and backend
  completion permits reuse.
- Persistent generation state remains owned by sequence/session pools.

### Allocation and accounting

- CPU and Metal production execution lease workspace from runtime-wide pools.
- Runtime diagnostics distinguish active, idle, requested, actual, output,
  workspace, host, persistent-state, and opaque bytes.
- Idle cache limits can trim free storage without invalidating live tensors.
- Hard budgets account for active plus cached allocations and evict idle storage
  before failing.
- After warm-up, intermediate backing and Effect Torch-controlled host metadata
  reuse pool entries; publication and driver exceptions are reported honestly.

### Concurrency

- Concurrent calls to one plan are race-free even on a serialized backend.
- Before process-wide locks are removed, stress tests prove independent writable
  storage, output stability, deterministic random identities, and transactional
  state rollback.
- One inference context rejects concurrent use while multiple contexts from one
  program can be admitted independently.
- Same-sequence generation updates remain serialized; distinct sequences can be
  scheduled together or independently.

### Training and inference

- Trainer donation preserves loss, parameter, and optimizer parity and cannot
  mutate an externally retained snapshot.
- Final parameters and retained metrics have ordinary output ownership.
- Caller-bound inference outputs match runtime-owned outputs and reject invalid
  device/layout/alias bindings before dispatch.
- Generation no longer configures executable output rings from decode batch
  size; KV/state capacity remains independently bounded and diagnosed.

### Regression gates

- CPU and Metal native and core test suites remain green.
- Release nano-GPT and FineWeb training retain correctness and do not regress
  steady-state throughput beyond an agreed measurement threshold.
- Batched generation, KDA recurrence, cancellation, and transaction rollback
  remain correct.
- Retained-output churn reaches a stable allocator pattern after release/GC and
  returns idle bytes below the configured cache limit after trim/pressure.

## References

[pytorch-inductor-memory]: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codegen/memory_planning.py
[pytorch-cuda-memory]: https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management
[jax-donation]: https://docs.jax.dev/en/latest/buffer_donation.html
[jax-gpu-memory]: https://docs.jax.dev/en/latest/gpu_memory_allocation.html
[xla-buffer-assignment]: https://github.com/openxla/xla/blob/main/xla/service/buffer_assignment.h
[tensorflow-bfc]: https://github.com/openxla/xla/blob/main/xla/tsl/framework/bfc_allocator.h
[tensorrt-architecture]: https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html
[ort-io-binding]: https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html
[ort-execution-frame]: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/execution_frame.cc
[ort-inference-session]: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/session/inference_session.cc
[mlx-memory]: https://ml-explore.github.io/mlx/build/html/python/memory_management.html
[vllm-architecture]: https://docs.vllm.ai/en/stable/design/arch_overview/
[paged-attention]: https://arxiv.org/abs/2309.06180
[pytorch-cudagraph-limitations]: https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html#limitations

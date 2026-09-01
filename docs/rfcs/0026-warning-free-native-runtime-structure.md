# RFC 0026: Warning-free native runtime structure

- **Status**: Draft
- **Author**: Michael Arnaldi
- **Created**: 2026-09-01
- **Depends on**: RFC 0015 (native backend), RFC 0017 (multi-backend
  runtime), RFC 0019 (executable compilation), RFC 0020 (invocation
  ownership), RFC 0021 (compiler pipeline refactor), RFC 0025 (target dtype
  legalization)
- **Updates**: RFC 0015 (backend implementation layout), RFC 0019 (runtime
  executable boundaries), RFC 0021 (backend ownership of lowered programs)

## Summary

Adopt one conceptual source layout for the CPU, Metal, and CUDA native runtime
crates, while preserving backend-specific instructions, kernels, memory
spaces, submission models, and optimization policy. Each runtime is organized
around the same boundaries: runtime and invocation ownership, target
capabilities, devices, values, workspace, executable compilation and execution,
stateful inference, operation families, kernel implementation, and N-API
adaptation.

Make warning-free compilation a repository invariant for workspace-owned native
runtime code. A loaded kernel, retained field, helper, feature branch, or
instruction attribute must have a production consumer, be correctly gated to
the configuration that consumes it, or be removed. Blanket warning suppression
is not an acceptable substitute for ownership.

The restructuring is initially mechanical. It does not create one shared
instruction enum, one accelerator base class, or a common device implementation.
CPU and Metal continue to lower the shared semantic graph into their own typed
programs. CUDA first replaces its parallel string-valued lowered schedule and
executable instruction table with one authoritative typed program, as RFC 0021
already requires. Exact code moves into a shared crate only after the new module
boundaries expose identical semantics, ownership, failure behavior, and
lifecycle requirements.

## Decision

1. `runtime-cpu`, `runtime-metal`, and `runtime-cuda` use the common top-level
   module structure defined by this RFC. Backend-only modules are permitted,
   but every common concern has the same owner and dependency direction.
2. `lib.rs` is a crate boundary, not an implementation module. It declares
   modules, re-exports the deliberate Rust API, and installs trait
   implementations that cannot live elsewhere.
3. `runtime.rs` owns runtime identity, admission and shutdown, backend device
   access, reusable runtime pools, and invocation nonces.
4. `capabilities.rs` owns the backend's immutable operation and dtype support
   policy. It combines device properties with policy revisions before compiler
   optimization and contributes to executable cache identity.
5. Backend instruction types, lowering, artifact preparation, physical
   scheduling, invocation frames, and execution live under `executable/`.
6. Persistent decode state, pools, sequences, prefix-cache bookkeeping, and
   state transactions live under `state/`. They do not live in N-API modules.
7. Backend operation planning and dispatch live under `ops/`, grouped by
   semantic family rather than by one file per public tensor method.
8. CUDA, Metal, and other device source or source-generation code lives under
   `kernels/` when it is not an ordinary Rust operation implementation. Kernel
   registries are grouped by operation family and do not retain unreachable
   handles.
9. A backend-specific `replay.rs` may own fixed-address replay resources needed
   by both executable and state code. Capture types do not create a dependency
   from `state` back into `executable`.
10. N-API modules validate and translate the Node boundary, own JavaScript
    handles, and sequence asynchronous work. Native runtime code owns numerical
    policy, invocation lifetime, prefix caching, and late-result cleanup.
11. Every supported feature and target configuration compiles without warnings
    from workspace-owned code. CI promotes those warnings to errors.
12. Dead code is removed rather than annotated. Narrow warning allowances are
    permitted only when required by generated code or a platform ABI and must
    carry a local explanation. Crate-wide or module-wide `dead_code` allowances
    are prohibited in production runtime code.
13. File movement and behavior changes are separate review units. A mechanical
    restructuring must preserve numerical results, diagnostics, memory plans,
    kernel selection, initialization behavior, cancellation, release, and
    measured performance.
14. Similar names do not imply a shared implementation. Shared Rust types are
    introduced only where all backends have the same contract.
15. Unsupported behavior remains an early compile or boundary error. Removing
    a dormant handle must not create a CPU fallback, dtype substitution, or
    delayed runtime failure.

## Motivation

### The package boundary is coherent

RFC 0017 deliberately gives each backend an independently installable package
and native addon. Core owns the semantic API. Shared Rust crates own graph,
autodiff, compiler, runtime, and N-API helpers. Each backend owns concrete
devices, storage, kernels, executables, and state. That package boundary is not
changed by this RFC.

The internal source boundaries have not converged to the same degree. This is
not required for backend independence, but it now obscures equivalent
responsibilities and makes parity work unnecessarily difficult.

### The current layouts reflect implementation history

CPU and Metal were implemented first during the native backend migration. They
grew operation-oriented modules with requirement planning and allocating and
`*_into` execution forms. Metal additionally owns MSL emission, command-buffer
submission, pipeline warming, buffer hazard management, and several specialized
inference kernels.

CUDA arrived later through a different implementation path. Instead of the
IR-to-PTX emitter anticipated by RFC 0015, it uses handwritten CUDA, NVRTC,
cuBLAS, CUDA graphs, and a device-local function registry. This is a valid
backend choice. It does not require lowering, state planning, dispatch, and
Node adaptation to remain concentrated in a few files.

As of this RFC, the approximate Rust and device-source sizes are:

| Crate | Source lines | Largest concentration |
|---|---:|---|
| `runtime-cpu` | 36,432 | `napi/mod.rs`, `executable.rs`, `composed.rs` |
| `runtime-metal` | 59,033 | `executable.rs`, `napi/mod.rs`, specialized ops |
| `runtime-cuda` | 18,818 | `executable.rs`, `napi.rs`, broad CUDA sources |

Metal has more operation modules, but it is not a clean template to copy. Its
executable and N-API modules are each larger than twelve thousand lines. CPU has
the same N-API concentration. CUDA is smaller but combines more responsibilities
inside `executable.rs` and `napi.rs`. The target must be a new common
organization, not one backend's current tree imposed on the others.

### Warnings expose registry and dispatch drift

The CUDA runtime currently compiles with dead-code warnings. Representative
cases include:

- K-quant MMQ kernels that are compiled, loaded, and stored but never selected;
- F32-specialized function handles whose operations dispatch through another
  kernel;
- generic quantized and stateful handles superseded by specialized paths;
- a split GQA kernel superseded by grouped GQA;
- instruction metadata retained after dispatch stopped reading it;
- constructors with no production caller.

These warnings are not harmless editor noise. They identify one of three
conditions:

1. a rejected or superseded implementation was not deleted;
2. an intended implementation was never connected to dispatch;
3. a symbol belongs only to another feature or target but is not gated there.

Only the second condition can represent missing functionality, and a warning
alone cannot decide which condition applies. The owner must trace the complete
path from compilation or construction through dispatch, validate the advertised
capability, and then connect, gate, or remove the symbol.

CPU and Metal currently compile cleanly in the primary N-API configuration,
but that does not establish a durable policy. Broad `dead_code` allowances and
untested feature combinations can hide the same drift. Warning-free compilation
must cover every supported configuration rather than one release command.

### CUDA also has duplicate executable schedules

CUDA currently stores a string-valued `CudaLoweredProgram` and a separate
`Instruction` table that execution actually reads. That violates RFC 0021's
requirement for one authoritative typed lowered program. It also makes a purely
mechanical executable split impossible because there is no single schedule to
assign to the new modules.

Before restructuring CUDA, its typed `Instruction` becomes the instruction
payload in `LoweredProgram`. Dispatch uses lowered instruction IDs, and the
parallel executable instruction and resource schedules are removed. This is a
contract repair required by RFC 0021, not a new cross-backend abstraction.

### Monolithic modules weaken ownership

An executable module legitimately coordinates many concerns, but it should not
define every concern. Today the largest runtime files combine several of the
following:

- backend instruction definitions;
- semantic lowering;
- memory and scratch declarations;
- pipeline or module preparation;
- operation dispatch;
- state schema and transaction logic;
- capture and replay;
- prefix-cache planning;
- N-API handle ownership and validation;
- serialization and sampling entry points.

This makes changes harder to review and lets resource contracts drift. A kernel
can remain in a device registry after its dispatch branch is removed. State
policy can become coupled to JavaScript handle layout. Planning metadata can
survive after execution stops using it. Similar backend bugs are fixed in
different places because equivalent responsibilities have different owners.

### Consistency should expose differences, not erase them

CUDA streams are not Metal command buffers. Metal pipelines are not CUDA
modules. CPU destinations are not device slices. CUDA graph capture has no
required CPU analogue. Metal buffer hazard tracking is not a generic
accelerator feature.

The common architecture therefore standardizes module responsibility and
dependency direction. It does not standardize algorithms or force platform
mechanics behind a lowest-common-denominator trait. A contributor should know
where to find lowering, state ownership, or sampling on every backend while
still seeing the real device model in that code.

## Goals

1. Make all workspace-owned code in the three native runtime crates compile
   without warnings in every supported target and feature configuration.
2. Give CPU, Metal, and CUDA the same top-level conceptual module layout.
3. Make one module the clear owner of every backend instruction, kernel handle,
   capability policy, invocation frame, state object, workspace resource, and
   N-API handle.
4. Keep N-API adaptation thin and move runtime algorithms out of Node-facing
   modules.
5. Make kernel registry entries traceable to tested dispatch branches.
6. Preserve RFC 0019's static memory plan and establish its authoritative
   lowered program in CUDA.
7. Preserve RFC 0020's runtime pools, invocation frames, completion, and output
   ownership.
8. Preserve backend-specific instruction enums, memory spaces, kernel plans,
   synchronization, and submission models.
9. Reduce the cost of adding an operation or fixing parity across all three
   backends.
10. Make exact cross-backend duplication visible enough to extract safely.
11. Preserve benchmark-proven CUDA and Metal inference paths.
12. Leave the public TypeScript runtime and tensor APIs unchanged.

## Non-goals

- Create a common CPU/GPU instruction enum.
- Create a general `AcceleratorDevice` trait over CUDA and Metal.
- Require identical kernel families or dtype support on every backend.
- Replace handwritten CUDA or Metal kernels with a portable compiler.
- Implement missing tensor operations as part of file movement.
- Fix all existing CUDA numerical test failures in the restructuring patches.
- Introduce automatic CPU fallback or graph partitioning.
- Change native package names, targets, loaders, or generated N-API declarations
  except where source module movement requires regeneration.
- Optimize code by deleting a slower path before its lack of production use is
  proved.
- Set an arbitrary maximum line count for source files. Responsibility, not
  file length alone, determines a valid boundary.
- Extract speculative shared helpers before backend-local ownership is clear.

## Target source layout

Each runtime adopts the following conceptual tree:

```text
src/
  lib.rs
  runtime.rs
  capabilities.rs
  device.rs
  value.rs
  workspace.rs
  replay.rs           # optional fixed-address replay resources

  executable/
    mod.rs
    ir.rs
    lower.rs
    compile.rs
    physical.rs       # required when the backend has a physical schedule
    invocation.rs
    run.rs

  state/
    mod.rs
    schema.rs
    pool.rs
    sequence.rs
    prefix.rs         # optional prefix-cache policy
    transaction.rs    # optional staged state commit

  ops/
    mod.rs
    pointwise.rs
    indexing.rs
    linalg.rs
    neural.rs
    quantized.rs
    attention.rs
    recurrent.rs
    sampling.rs
    fusion.rs          # optional backend fusion realization

  kernels/             # optional device source and source generation
    mod.rs

  napi/
    mod.rs
    error.rs
    runtime.rs
    value.rs
    executable.rs
    state.rs
    io/
      mod.rs
      gguf.rs
      safetensors.rs
```

The listed leaf files define ownership categories, not empty-file
requirements. A backend omits `physical.rs`, `replay.rs`, `prefix.rs`,
`fusion.rs`, or the entire `kernels/` directory when it has no implementation.
If normal execution requires a physical synchronization schedule, as Metal
does, `physical.rs` is not optional for that backend. A large operation family
may use a subdirectory. A small backend may combine adjacent leaf files under
the same parent when the combined module still has one lifecycle and dependency
boundary.

The common requirements are the top-level concerns and their responsibilities.
For example, CPU may retain aligned storage implementation in
`device/storage.rs`, Metal may retain MSL generation in `kernels/emit.rs`, and
CUDA may retain `.cu` and `.cuh` sources under `kernels/`.

## Module responsibilities

### `lib.rs`

`lib.rs` contains:

- crate documentation;
- module declarations;
- deliberate public re-exports;
- runtime `Buffer` or identity implementations when they cannot be placed on
  the owning type without creating a cycle.

It does not contain allocation policy, operation dispatch, state management,
or N-API implementation. Public visibility is minimized. Types used only by
the package-local addon remain `pub(crate)`.

### `runtime`

The runtime module owns the long-lived backend service described by RFCs 0017
and 0020:

- runtime identity and placement registration;
- device selection and access;
- admission, shutdown, and invocation nonces;
- reusable workspace, output, host-frame, and completion pools;
- cleanup of interrupted work that completes after its caller returns.

It does not contain operation dispatch or JavaScript classes. An executable
receives the runtime context needed to acquire a per-call invocation frame. The
frame retains all resources until backend completion, even when the JavaScript
request has already been cancelled.

### `capabilities`

The capabilities module owns the backend policy required by RFC 0025. It
combines immutable device properties with operation and dtype support tables,
legalization policy, policy revision, and target fingerprint. The compiler
driver receives this object before region selection. Its fingerprint and policy
revision participate in executable cache identity.

The device reports physical facts such as architecture and supported storage
types. It does not inspect semantic operations. Final lowering consumes an
already selected legalization plan rather than making a second capability
decision.

### `device`

The device module owns process or runtime device identity and platform
submission resources. Examples include:

- CPU placement and execution policy;
- Metal devices, queues, command buffers, encoders, pipeline caches, events,
  and buffer hazard tracking;
- CUDA contexts, streams, cuBLAS handles, CUDA modules, and architecture
  capability.

The device owns allocator and kernel or pipeline cache roots, but registries
are typed by operation family. A flat struct containing every historical kernel
handle is not the target design. Device creation initializes the platform
context and baseline services only. Executable compilation requests
family-specific artifacts from a device-scoped cache keyed by target
fingerprint and compile options.

### `value`

The value module owns concrete tensor storage views, shape, layout, dtype,
placement validation, host transfer primitives, and deterministic release. It
does not own semantic graph nodes or executable scheduling. Backend-specific
buffer or storage modules may sit below `value` or `device`.

### `workspace`

The workspace module maps RFC 0019 memory segments to concrete reusable
storage. It owns acquisition, lease lifetime, alignment, and release. It does
not rediscover operation scratch at execution time.

### `executable`

The executable directory owns the complete path from compiler preparation to
fixed execution:

- `ir.rs` defines the backend instruction enum and instruction-owned algorithm
  plans;
- `lower.rs` consumes `PreparedProgram`, `GraphIndex`, `OptimizationPlan`, and
  legalization decisions and emits the authoritative `LoweredProgram`;
- `compile.rs` performs memory planning, requests pipeline or module artifacts,
  publishes constants, and assembles the executable;
- `physical.rs` owns the backend's normal physical schedule, including required
  Metal synchronization planning;
- `invocation.rs` owns the RFC 0020 frame that retains inputs, workspace leases,
  provisional outputs, state transactions, host staging, and completion;
- `run.rs` validates invocation bindings, acquires the frame through the runtime,
  dispatches the fixed program, synchronizes where required, and publishes
  outputs.

`executable/mod.rs` defines the executable artifact and coordinates these
stages. It does not contain operation kernel bodies, N-API classes, prefix-cache
policy, replay-cache ownership, or serialization.

The three backends do not share an instruction enum. RFC 0021's typed backend
lowering remains authoritative.

### `state`

The state directory owns persistent inference state independently of Node:

- state geometry and compatibility validation;
- KV, KDA, and convolution pools;
- sequence leases and exclusive-use rules;
- block allocation and eviction;
- prefix identity and snapshot policy;
- staged state writes, commit, rollback, and late-result cleanup;
- device-cache retention needed by the backend.

N-API wrappers contain an opaque reference to these types. They do not
implement their algorithms. Executable dispatch consumes state through a
backend-local invocation or transaction interface that prevents a dependency
from `state` back into N-API.

State policy is shared only where semantics are exactly equal. Metal paged
slabs, CUDA retained device caches, and CPU reference storage may continue to
use different physical representations.

State resolves pools and sequences into an operation-specific set of device
values before operation dispatch. Operation modules may accept those values as
ordinary inputs, but they do not depend on pool, sequence, prefix-cache, or
transaction types.

### `ops`

Operation modules own backend algorithm requirements, validation, kernel
selection, and dispatch into caller-assigned destinations and scratch. They are
grouped by semantic family so a contributor can compare implementations without
creating a file for every enum variant.

An operation API separates preparation from execution where the backend has
compile-time artifacts:

```rust
fn requirements(spec: &OperationSpec<'_>) -> Result<OperationPlan, String>;
fn prepare(device: &Device, plan: &OperationPlan) -> Result<PreparedOp, String>;
fn encode_into(
    device: &Device,
    prepared: &PreparedOp,
    inputs: &[Value],
    output: &mut Value,
    scratch: &mut [Value],
) -> Result<(), String>;
```

The names and exact types remain backend-specific. The invariant is that
planning declares resources before RFC 0019 memory assignment and execution
does not allocate unplanned intermediates. Executable and state code resolve
persistent state into these input bindings before calling an operation.

### `kernels`

The kernels directory owns CUDA source, Metal source generation or fixed MSL,
entry-point names, low-level launch ABI, and imported kernel fragments. It does
not own graph lowering or JavaScript policy.

Kernel registries are grouped, for example:

```rust
struct PointwiseKernels { /* ... */ }
struct QuantizedKernels { /* ... */ }
struct AttentionKernels { /* ... */ }
struct RecurrentKernels { /* ... */ }
```

Each static stored handle has at least one reachable dispatch branch under the
same feature and target configuration. Generated Metal kernels follow the same
rule at the template level: the generator domain, launch ABI, cache identity,
and representative numerical cases are tested. The RFC does not require one
registry field or test for every generated specialization.

An experimental alternative is either behind an explicit compile option with
tests and benchmarks or absent from the production registry. Merely compiling
and storing a handle does not preserve a feature.

### `replay`

A backend may define `replay.rs` for a fixed-address replay context shared by
state and executable code. The context owns the capture object and every input,
output, workspace lease, staging allocation, and completion resource whose
address or lifetime the capture assumes. CUDA decode graph caches use this
owner.

Executable code may build and launch a replay context. State code may retain and
evict one. Neither module reaches through the other to manage replay resources.
Normal Metal physical scheduling remains in `executable/physical.rs`; it is not
modeled as optional replay.

### `napi`

The N-API directory owns:

- JavaScript-visible classes and generated declaration inputs;
- conversion and validation of JavaScript values;
- cancellation token adaptation and forwarding;
- asynchronous worker submission while the native invocation frame owns
  interruption and late-result cleanup;
- opaque JavaScript handle ownership and deterministic `clear` or `release`;
- GGUF and safetensors file boundaries.

It calls runtime modules for graph construction, compilation, execution,
sampling, state, and serialization. It does not contain backend operation
selection, KV allocation algorithms, prefix-key construction, or physical
command planning.

`napi/mod.rs` is an export and wiring module. Runtime, value, executable, and
state wrappers live in separate files even when they delegate to closely
related backend types.

## Dependency direction

The target dependency direction is:

```text
napi
  -> runtime, executable, state, device, value

runtime
  -> capabilities, device, value, workspace

executable
  -> runtime, capabilities, ops, state, replay, device, value, workspace
  -> compiler, graph, effect-torch-runtime

capabilities
  -> device, compiler, graph, effect-torch-runtime

ops
  -> kernels, device, value

state
  -> replay, device, value, workspace

replay
  -> device, value, workspace

workspace
  -> device, value, effect-torch-runtime

value
  -> device, effect-torch-runtime

device
  -> kernels, effect-torch-runtime
```

Back edges are prohibited:

- native runtime modules do not depend on N-API;
- operation modules do not inspect JavaScript handles or state pool types;
- device modules do not inspect semantic graph nodes;
- kernel modules do not choose compiler regions;
- state modules do not call package adapters or executable modules;
- workspace modules do not infer scratch from operation type at runtime.

Cycles are resolved with narrow backend-local traits or data plans at the
owner boundary, not by moving unrelated code back into `lib.rs` or
`executable/mod.rs`.

## Warning policy

### Scope

The zero-warning rule covers diagnostics emitted for workspace-owned code by:

- `rustc` during check, build, test compilation, and documentation;
- TypeScript and Oxlint for backend packages and generated adapters;
- NVRTC for first-party CUDA source;
- Metal shader compilation for first-party MSL;
- native declaration generation and package verification.

Warnings originating entirely inside an external dependency or the user's
toolchain installation are not converted into source changes in this RFC. They
must remain distinguishable from workspace diagnostics. Vendored or copied
kernel source maintained in this repository is workspace-owned unless its
warning policy is explicitly documented at the import boundary.

### Required disposition

Every warning receives one explicit disposition:

1. **Connect.** The implementation is required, so dispatch and focused tests
   are added. Capability declarations and compile policy select it.
2. **Gate.** The implementation is used only under a supported feature or
   target, so its definition, construction, and tests share the same `cfg`.
3. **Remove.** The implementation is rejected, superseded, redundant, or
   unreachable, so its fields, construction, source inclusion, and tests are
   deleted together.
4. **Allow narrowly.** An ABI, generated declaration, or platform callback
   requires a symbol Rust cannot observe. The smallest item gets an `allow`
   with a comment naming that requirement.

Renaming a field with `_`, adding a broad `allow`, or reading a value solely to
silence the compiler is not a disposition.

### Kernel registry invariant

For each static runtime-loaded kernel entry point, tests or compile-time
inventory establish:

- the source and exported entry-point name;
- the typed registry field;
- the compile or dispatch branch that selects it;
- the dtype, shape, feature, and target constraints;
- at least one focused correctness test;
- a benchmark record when it competes with another algorithm.

For generated kernels, the inventory records the generator, allowed
specialization domain, compile and dispatch consumer, cache key, launch ABI,
and representative correctness tests.

A kernel retained only as a possible future optimization is source history,
not production code. It is removed and can be recovered from version control.

### Feature configurations

Each native runtime must compile warning-free in both of its supported library
roles:

- the Rust library without `napi-addon`;
- the package-local addon with `napi-addon`.

Tests and examples may add feature-specific consumers, but they must not be the
only reason a production field appears used. Test-only helpers are explicitly
gated with `#[cfg(test)]`.

Metal checks run on macOS. CUDA Rust compilation checks run on supported hosts;
device initialization and NVRTC checks run on the CUDA devbox.

## Backend mapping

### CPU

The CPU migration maps existing modules approximately as follows:

| Current owner | Target owner |
|---|---|
| `executable.rs` | `executable/{ir,lower,compile,invocation,run}.rs` |
| identity and runtime pools | `runtime.rs` |
| operation and dtype support policy | `capabilities.rs` |
| `pool.rs` and state internals in `napi/mod.rs` | `state/` |
| `ops.rs`, `reduce.rs`, `indexing.rs` | `ops/{pointwise,indexing}.rs` |
| `matmul.rs`, `linalg.rs`, `conv.rs` | `ops/linalg/` |
| `quantized.rs`, relevant `composed.rs` sections | matching `ops/` families |
| `storage.rs`, `tensor.rs`, `value.rs` | `device/` support and `value.rs` |
| `napi/mod.rs` | `napi/{runtime,value,executable,state}.rs` |

The CPU reference implementations remain straightforward and test-oriented.
The restructuring does not force GPU-style kernel registries onto CPU.

### Metal

The Metal migration maps existing modules approximately as follows:

| Current owner | Target owner |
|---|---|
| `executable.rs` | `executable/{ir,lower,compile,physical,invocation,run}.rs` |
| identity and runtime pools | `runtime.rs` |
| operation and dtype support policy | `capabilities.rs` |
| `run.rs`, `value.rs` | `value.rs` and operation execution helpers |
| `ops.rs`, `gemm.rs`, `conv.rs`, `indexing.rs` | matching `ops/` families |
| `flash.rs`, `paged.rs` | `ops/attention/` plus persistent pieces in `state/` |
| `kda.rs`, `shortconv.rs` | `ops/recurrent/` plus persistent pieces in `state/` |
| `emit.rs`, fixed MSL strings | `kernels/` and `ops/fusion.rs` |
| submission and allocation in `device.rs` | remains device-owned |
| `napi/mod.rs` | `napi/{runtime,value,executable,state}.rs` |

Metal pipeline warming remains a compile responsibility. Command-buffer and
buffer-hazard ownership remain device responsibilities. The split must not
move pipeline creation or allocation into executable dispatch.

### CUDA

The CUDA migration maps existing modules approximately as follows:

| Current owner | Target owner |
|---|---|
| `executable.rs` instruction enum | `executable/ir.rs` |
| `lowering.rs` and compile assembly | `executable/{lower,compile}.rs` |
| executable dispatch and invocation resources | `executable/{invocation,run}.rs` |
| graph capture object and fixed-address resources | `replay.rs` |
| identity and runtime pools | `runtime.rs` |
| operation and dtype support policy | `capabilities.rs` |
| KV pool, sequence, prefix, retained replay cache in `napi.rs` | `state/` |
| sampling launch and candidate readback | `ops/sampling.rs` |
| quantized selection and launch | `ops/quantized.rs` |
| attention and recurrence dispatch | corresponding `ops/` modules |
| flat `CudaDevice` function fields | typed registries under `kernels/` |
| `.cu` and `.cuh` source | remains under `kernels/` by family |
| N-API classes and I/O | split `napi/` modules |

CUDA graph capture remains CUDA-specific. Its replay context keeps every
fixed-address resource alive and may be retained by the state cache. cuBLAS and
NVRTC remain device or kernel preparation details. The current ordinary Muse
Glimmer path remains a first-class production consumer and is not generalized
away.

## Migration plan

### Phase 0: Inventory and baselines

Before moving code:

1. Record every warning for each crate, target, feature set, test compile, and
   host native build.
2. Classify every warning as connect, gate, remove, or narrow allow.
3. Record module dependency graphs and the public and `pub(crate)` Rust API.
4. Record native declaration hashes and package verification output.
5. Record focused numerical, ownership, cancellation, interruption, release,
   and state tests.
6. Record CPU and Metal benchmark baselines.
7. Record the CUDA command, prompt, exact token counts, warmup, repetition count,
   environment, build profile, GPU, driver, toolkit, and model artifact hashes
   used for the Muse Glimmer prefill, decode, sampled, and long-generation
   baselines.
8. Inventory kernel entry points, generated kernel domains, and dispatch
   consumers.
9. Inventory broad warning suppressions and justify or remove each one.
10. Record current target capabilities, policy revisions, and executable cache
    fingerprints.

The inventory is checked into the implementation pull request or an RFC
implementation record. It is not kept as an undocumented local note.

### Phase 1: Warning cleanup

Clean warnings before large moves so dead and live code are not reorganized
together. For each disposition:

- remove all parts of a dead path, including source inclusion and registry
  construction;
- connect intended paths through explicit compile policy and tests;
- gate feature-only paths consistently;
- remove redundant instruction fields at construction and dispatch together;
- regenerate declarations when N-API exports change;
- run focused correctness and performance checks after every performance-path
  deletion or dispatch change.

CUDA MMQ, duplicate generic quantized handles, inactive attention variants,
unused instruction metadata, and unused value constructors are initial audit
candidates. This RFC does not prejudge each result. Current dispatch and
capability requirements decide whether each candidate is connected or removed.

Phase 1 ends only when the warning matrix is zero without broad suppression.

### Phase 2: Restore authoritative CUDA lowering

Replace CUDA's string-valued lowered instructions and parallel executable
instruction table with `LoweredProgram<Instruction, ...>`. Dispatch iterates
the lowered instruction IDs and reads all instruction-owned plans from that one
program. Delete the duplicate schedule and resource metadata only after focused
compilation diagnostics, numerical parity, graph replay, and inference tests
pass.

This phase repairs the RFC 0021 contract before files move. It is reviewed as a
behavior-preserving compiler change, not hidden in a mechanical patch.

### Phase 3: Establish top-level boundaries

Create `runtime.rs`, `capabilities.rs`, `executable/`, `state/`, `ops/`, and
split `napi/` boundaries in all three crates. Use source moves and visibility
adjustments only. Preserve names, types, match arms, and call order where
possible.

Each backend remains buildable after its own move. CPU, Metal, and CUDA changes
do not need to land in one indivisible patch. The target tree and dependency
rules are fixed before the first move so temporary divergence is directional.

Recommended order:

1. CPU establishes the smallest complete structure and exercises general
   tensor execution without accelerator lifecycle concerns.
2. Metal applies the structure while preserving pipeline, submission, and
   transaction boundaries.
3. CUDA applies the structure after dead registry entries are gone, preserving
   its measured inference paths and graph capture.

No backend is treated as the implementation template. Review compares each
move against this RFC's responsibilities.

### Phase 4: Separate state and N-API

Move pool, sequence, prefix, transaction, and retained-device-cache policy into
`state/`. Move fixed-address CUDA graph resources into `replay.rs`, then let the
state cache retain that complete replay context. Split Node wrappers by handle
domain. Replace direct access from N-API to state internals with narrow
backend-local methods.

This phase must preserve:

- deterministic release;
- exclusive sequence leases;
- fork and prefix-cache reference counts;
- cancellation and late-result cleanup;
- commit-after-success behavior;
- eviction timing;
- device-cache and graph lifetime;
- no hidden CPU fallback.

State changes remain mechanical unless a separately reviewed bug fix is needed.
A discovered bug gets a focused failing test and an explicit behavior change,
not an incidental fix inside a move.

### Phase 5: Split operation and kernel registries

Move algorithm selection and dispatch from executable matches into operation
families. Executable instructions retain typed plans and call allocation-free
operation entry points with assigned outputs and scratch.

Group accelerator kernel handles by family. Add registry-to-dispatch inventory
tests where practical. Device initialization must not compile or load rejected
alternatives. Compile options that select alternatives join executable and
kernel cache keys as required by RFC 0019.

This phase may improve initialization time by removing unused source and handle
loads, but it may not change execution algorithms without benchmark evidence.

### Phase 6: Extract exact shared code

After all runtimes use the common boundaries, review exact duplication. Good
shared candidates may include:

- backend-neutral state geometry validation;
- block and cursor arithmetic with identical semantics;
- N-API scalar and shape validation;
- sampling policy validation already rooted in `runtime`;
- diagnostics assembly already rooted in `compiler`;
- file-format validation that does not own device storage.

Do not extract code merely because names match. Prefix-cache eviction, device
state retention, memory ownership, and submission cleanup remain backend-local
when their failure or lifecycle behavior differs.

Any new shared crate or expansion of `runtime` or `napi` states its ownership
contract and proves that separately linked addons do not share process-local
handles accidentally.

### Phase 7: Documentation and enforcement

Update crate-level documentation to describe the final modules and invariants.
Add CI warning gates and retain the kernel inventory checks. Remove migration
aliases and compatibility modules when no external or persisted consumer
requires them.

Mark this RFC implemented only after all three crates use the target boundaries
and pass the complete acceptance matrix.

## Correctness and performance invariants

### Semantic behavior

For each backend, optimized and unoptimized execution continue to implement the
same semantic graph under existing exact or tolerance-based contracts. Module
movement does not change dtype legalization, random provenance, reduction
order, output ordering, or error timing.

### Memory and ownership

The restructuring preserves:

- static workspace segment assignments;
- escaping and provisional output ownership;
- no ambient intermediate allocation during executable dispatch;
- deterministic concrete value release;
- state transaction commit only after successful execution;
- interruption retention and late-result cleanup;
- command-buffer, stream, replay context, completion, and workspace lifetimes.

Moving a type does not move its ownership responsibility unless the new owner
implements and tests the same lifecycle.

### Kernel and physical plans

Mechanical phases preserve backend diagnostics for:

- lowered instruction names and count;
- selected optimization regions;
- scratch and workspace sizes;
- kernel or algorithm selection;
- pipeline or module preparation;
- physical synchronization or capture topology.

Intentional changes require separate benchmark and correctness evidence.

### Performance

No restructuring phase may regress an established backend benchmark beyond the
tolerance recorded in Phase 0 without approval. Compare repeated medians on the
same machine, power state, build profile, artifacts, command, prompt, and
environment.

The current ordinary CUDA Muse Glimmer gate uses
`MUSE_GLIMMER_DFLASH=false` on the NVIDIA RTX PRO 6000 Blackwell Max-Q. Phase 0
sets that exact `CUDA_DEVBOX_GPU_ID`; the repository's default Server Edition
GPU is not interchangeable and requires a separately approved rebaseline. The
target model SHA-256 is
`3d63a1daff23fdc2a6927316151e855cacffe89b5cb9b9397a5aec0c412ec08d`, and the
tokenizer SHA-256 is
`c9dbee66967b58f31a7c27f723c3760da3526ccd0427578e8905b0abb0031c4d`. Phase 0
records the driver, toolkit, prompt, command, and remaining environment. The
final migration must retain at least:

- a 1,000 token-per-second median prefill over at least three warmed runs with
  an exact 3,000-token context;
- a 60 token-per-second median decode over the same runs with production
  sampling;
- a 60 token-per-second average over one real generation of exactly 3,500
  tokens;
- the context-64, 64-token greedy hash `5fed5630` when numerical policy is
  unchanged.

CPU and Metal retain their existing focused training, inference, memory, and
backend benchmark gates. A backend-specific algorithm change discovered during
restructuring is measured and reviewed independently.

## Verification matrix

The implementation runs warning-as-error checks for each supported runtime. The
exact CI wrapper may avoid rebuilding dependencies repeatedly, but it must be
equivalent to:

```bash
RUSTFLAGS="-D warnings" cargo check -p effect-torch-runtime-cpu --all-targets
RUSTFLAGS="-D warnings" cargo check -p effect-torch-runtime-cpu --all-targets --features napi-addon

RUSTFLAGS="-D warnings" cargo check -p effect-torch-runtime-metal --all-targets
RUSTFLAGS="-D warnings" cargo check -p effect-torch-runtime-metal \
  --all-targets --features napi-addon

RUSTFLAGS="-D warnings" cargo check -p effect-torch-runtime-cuda --all-targets
RUSTFLAGS="-D warnings" cargo check -p effect-torch-runtime-cuda --all-targets --features napi-addon
```

Metal commands run on macOS. CUDA addon build and runtime kernel compilation run
in the CUDA devbox. CI also runs the checks for every Rust target listed for the
three packages by `scripts/native-packages.mjs`; the host-only commands above do
not replace that target matrix. Rust documentation is warning-free under:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p effect-torch-runtime-cpu --features napi-addon
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p effect-torch-runtime-metal --features napi-addon
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p effect-torch-runtime-cuda --features napi-addon
```

A CUDA test harness reads the complete NVRTC program log for successful
first-party source compilation and fails on warnings. A macOS test harness
materializes static and representative generated MSL and compiles it with
`xcrun metal -Werror`. The runtime Metal API exposes diagnostics only on failure,
so successful-warning enforcement belongs in that CLI harness rather than the
runtime adapter.

The repository also runs:

```bash
pnpm typecheck
pnpm lint
pnpm test

cargo check --workspace --features napi-addon
cargo test --workspace --features napi-addon
cargo fmt --all -- --check

pnpm check:native-types
pnpm verify:native-packages
```

Focused suites run before the broad matrix:

- executable compilation, memory-plan, and cache tests;
- value clear, output release, interruption, and late-result tests;
- state pool, sequence fork/release, prefix matching, eviction, and rollback;
- dtype and layout parity for moved operation families;
- quantized GGUF dispatch and sampling;
- Metal command-buffer failure and allocation guards;
- CUDA kernel initialization, graph capture and replay lifetime, sampled
  generation, and long-context inference;
- target-specific package builds and first-party NVRTC or Metal warning-log
  checks.

Known pre-existing test failures are inventoried in Phase 0. The restructuring
may not add failures. Acceptance does not relabel a pre-existing failure as
unsupported or hide it behind a fallback. Fixing those failures is encouraged
but remains a separately visible behavior change.

## Review rules

1. Prefer `git mv` and minimal import changes for mechanical phases.
2. Do not combine source movement, algorithm changes, and warning suppression in
   one patch.
3. Preserve public and generated N-API names unless a separate API change is
   approved.
4. Regenerate native declarations from Rust exports; never hand-edit generated
   declarations.
5. Record before-and-after diagnostics and benchmarks for performance-sensitive
   files.
6. Delete temporary forwarding modules after all internal users migrate.
7. Do not add backward-compatibility modules for unexported Rust paths.
8. Keep platform-specific comments with the platform mechanism they explain.
9. Require a focused test for every warning disposition classified as connect.
10. Require proof of no production consumer before deleting a kernel path.

## Alternatives considered

### Keep independent layouts

Independent internals are allowed by RFC 0017, but the current divergence does
not consistently correspond to platform differences. Equivalent lowering,
state, and N-API responsibilities are hidden in unrelated files. Keeping the
layouts preserves avoidable review and parity cost.

### Make CUDA mirror Metal

Rejected. Metal has useful operation boundaries, but its executable and N-API
modules are also oversized, and its pipeline and command-buffer model should
not determine CUDA structure. The new layout is defined independently and
applies to CPU and Metal as well.

### Introduce one shared accelerator runtime

Rejected. CUDA and Metal differ in memory visibility, submission, failure,
capture, pipeline, and synchronization semantics. A shared base would either
leak platform branches through every method or erase ownership facts needed for
correctness. Share exact policy after restructuring, not device identity.

### Suppress warnings

Rejected. A broad `dead_code` allowance cannot distinguish a rejected kernel
from an unconnected required capability. It also permits device initialization
and compile time to grow with unreachable implementations.

### Restructure before warning cleanup

Rejected. Moving dead alternatives gives them a false appearance of deliberate
architecture and makes later deletion harder to review. The warning inventory
is the first ownership pass.

### Perform one cross-backend rewrite

Rejected. A flag-day move creates large merge conflicts and makes performance
or lifecycle regressions difficult to attribute. The target boundaries are
common, but each backend migrates in independently verifiable phases.

## Risks

### Mechanical changes can alter behavior

Rust movement often changes visibility, initialization order, static lifetime,
or feature gating. Accelerator registries and caches are especially sensitive.
Mitigation: preserve types and call order first, run diagnostics and benchmarks
after each move, and separate behavior changes.

### Invocation resources can be released too early

Splitting runtime, executable, state, replay, and N-API code can scatter one
invocation's resources across owners. A cancelled caller then risks releasing a
workspace, output, or fixed-address capture resource before backend completion.
Mitigation: one invocation frame retains the complete resource set, and focused
interruption and late-result tests run after every ownership move.

### CUDA schedule consolidation can change dispatch

Replacing CUDA's parallel schedules can change instruction order, resource
lookup, diagnostics, or graph capture even when every instruction body remains
the same. Mitigation: compare lowered diagnostics and instruction IDs before and
after the change, then run numerical, replay, and performance gates before the
module split.

### Warning cleanup can delete latent functionality

An unused handle may represent an intended capability whose dispatch was never
completed. Mitigation: compare the public capability and dtype contract, search
construction and dispatch, add a focused test, and explicitly choose connect or
remove. Source history is not treated as a runtime feature.

### Common names can encourage false sharing

Matching `state` or `ops` modules may tempt premature generic traits.
Mitigation: the first migration is backend-local, instruction enums remain
distinct, and shared extraction requires identical ownership and failures.

### Active CUDA work can conflict with movement

CUDA kernels and dispatch are under active performance development. Large moves
can obscure benchmark diffs and produce difficult merges. Mitigation: complete
the warning cleanup first, move one ownership boundary at a time, and keep
performance patches separate.

### Platform verification is asymmetric

Metal requires macOS and CUDA runtime checks require a provisioned GPU.
Mitigation: CI and the CUDA devbox own explicit target jobs; no backend is
silently validated through CPU.

### More modules can become ceremony

A mechanically identical tree can still hide poor boundaries. Mitigation: leaf
files may be combined when they have one owner, and no arbitrary line-count
limit is imposed. Dependency direction and lifecycle are the review criteria.

## Acceptance criteria

1. CPU, Metal, and CUDA use the target top-level `runtime`, `capabilities`,
   `executable`, `state`, `ops`, and split `napi` boundaries, with backend-only
   omissions documented.
2. Every supported target, default, test, documentation, and `napi-addon` build
   of the three crates emits zero warnings for workspace-owned code under
   warning-as-error settings.
3. First-party NVRTC and Metal source compilation emits no warnings.
4. No production runtime crate or module has a blanket `dead_code` allowance.
5. Every static accelerator kernel handle has a reachable, constrained, and
   tested dispatch consumer. Generated kernels have a constrained and tested
   generator domain. Rejected alternatives are not loaded.
6. `lib.rs` files contain only crate wiring, documentation, deliberate
   re-exports, and unavoidable trait implementations.
7. N-API modules contain no prefix-cache algorithm, backend kernel selection,
   memory planning, state transaction implementation, invocation lifetime, or
   late-result cleanup.
8. Each backend has one authoritative `LoweredProgram` with its own typed
   instructions and physical execution plan in accordance with RFC 0021.
9. Every invocation frame retains inputs, workspace, provisional outputs, state
   transactions, host staging, and completion until backend work finishes.
10. Executable dispatch performs no unplanned intermediate allocation and
    preserves RFC 0019 memory diagnostics.
11. Target capability policy runs before region selection and contributes its
    fingerprint and revision to executable cache identity.
12. State ownership, cancellation, release, interruption, fork, rollback,
    replay eviction, and late-result tests pass on every available backend
    without fallback.
13. Native declarations are regenerated and package verification passes.
14. CPU and Metal focused and broad tests add no failures relative to the Phase
    0 inventory. CUDA adds no failures and does not hide its inventoried gaps.
15. CUDA retains the exact ordinary Muse Glimmer correctness and performance
    gates defined by this RFC and the Phase 0 implementation record.
16. Repository typecheck, lint, Rust formatting, documentation, workspace
    checks, target builds, shader warning checks, and relevant Rust tests pass.
17. Crate-level documentation describes the final module ownership and
    platform-specific invariants.
18. Any shared code introduced after restructuring has an explicit owner and an
    identical tested contract across every consumer.
19. Temporary migration modules, aliases, and unjustified warning suppressions
    are removed.

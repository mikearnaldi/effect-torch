# RFC 0017: Multi-Backend Runtime Architecture

- **Status**: Draft
- **Created**: 2026-08-07
- **Depends on**: RFC 0002 (autodiff), RFC 0007 (fusion), RFC 0008
  (compilation), RFC 0010 (inference), RFC 0012 (dtypes), RFC 0015
  (native backend)
- **Updates**: RFC 0006 (roadmap), RFC 0015 (CUDA phase)

## Summary

Decouple `@effect-torch/core` from the former monolithic native addon so
CPU, Metal, and future implementations can be installed, versioned, and
evolved independently.

Core will depend on one Effect service, `Runtime.Runtime`, whose value is
a live `Runtime.RuntimeService` bound to a default placement. The runtime owns its
  devices, contexts, queues, allocators, compiler caches, tensor handles,
  executables, and optional extensions. Backend packages acquire
that runtime and provide it as a Layer at the application or test-program
boundary. Tensors retain only opaque handles and static metadata; all
backend dispatch uses the ambient service, which validates those handles.

The TypeScript boundary uses opaque, runtime-owned handles. Native
backend addons statically link shared Rust crates for the semantic
graph, validation, autodiff, neutral transforms, and compilation
machinery. Rust traits are internal compile-time boundaries, not an ABI
between separately loaded addons. No project-specific C plugin ABI is
introduced; PJRT backends use PJRT's existing versioned C API.

This RFC deliberately does not move the graph implementation into
TypeScript, add a global per-op dispatcher, partition one graph across
unrelated runtimes, or silently fall back to CPU. Those choices would
either rewrite the project's most mature subsystem or hide transfers
and numerical/performance changes.

The project remains experimental and pre-release. This architecture is
not constrained by backward compatibility with the current API,
packages, checkpoints, native handles, or compiled artifacts. Migration
shims and deprecation periods are added only when a concrete persisted
or external compatibility requirement is identified; otherwise the old
surface is removed when its replacement lands.

## Motivation

The existing architecture was the correct proof-of-concept: one N-API
addon established that an Effect-native API over a lazy Rust graph can
train real models, compile reusable programs, and support first-party
CPU and Metal kernels. It is now the scaling boundary.

The coupling is deeper than `CurrentDevice`:

- Core had a mandatory dependency on the monolithic addon.
- `Tensor.Any` stores that addon's `LazyTensor`; a concrete tensor also
  stores its `NativeTensor`.
- Tensor construction, ordinary ops, evaluation, autodiff, compilation,
  safetensors, tokenizer tensor creation, decode programs, KV pools, and
  KV sequences all call concrete exports from the same addon.
- TypeScript device identity is the closed union `"cpu" | "metal"`.
- Rust device identity is the closed `Device::{Cpu, Metal}` enum and
  materialized storage is the closed `Val::{Cpu, Metal}` union.
- `eval_uncached` is one exhaustive CPU/Metal cross-product dispatcher;
  fusion, synchronization, readback, safetensors, and compiled-program
  memory planning contain further device matches.

Adding a `Cuda` enum variant would make the monolith larger without
creating a backend boundary. It would also force users without CUDA to
install CUDA-linked artifacts and tie all backend releases to one native
package. ROCm has a different toolchain and release matrix; TPU is a
compiler/client integration rather than a family of eager kernels; a
remote runtime may have no native addon at all.

The existing lazy architecture is an advantage. Whole-graph evaluation,
native autodiff, and frozen programs already fit graph-oriented backends
such as PJRT better than a PyTorch-style eager operator dispatcher. The
new boundary should preserve that advantage.

## Goals

1. `@effect-torch/core` has no mandatory native dependency.
2. Backend implementations are independently installable packages.
3. Multiple backend packages can be loaded in one Node process without
   sharing native classes or global registration; each Effect program has
   one authoritative runtime service.
4. Every tensor and executable uses an opaque backend-owned handle
   whose compatibility is validated by the active runtime.
5. Existing lazy evaluation, one-walk semantics, autodiff, cancellation,
   compilation, strict dtype behavior, and early errors are preserved.
6. Native backends reuse one semantic graph and transform
   implementation instead of reimplementing framework semantics.
7. The boundary can represent local accelerators, PJRT clients, and
   remote sessions without pretending that they are identical devices.
8. Backend lifecycle, availability, capabilities, and errors are
   Effect-native and testable through Layers.

## Non-goals

- Automatic graph partitioning across unrelated backends.
- Implicit device movement or CPU fallback.
- A public stream/event API in the first version.
- A stable C ABI for arbitrary third-party effect-torch plugins.
- Moving the semantic graph, autodiff, or fusion compiler to TypeScript.
- Portable serialized executables.
- A StableHLO representation of every inference-specific operation in
  the first migration.
- Compiler-driven sharding or changes to RFC 0001's process-group model.

## Compatibility Policy

Breaking changes are explicitly acceptable while effect-torch remains
experimental. The migration should optimize for the smallest coherent
long-term architecture, not preserve interfaces created for the initial
CPU/Metal proof-of-concept.

In particular, this RFC permits breaking changes to:

- TypeScript module names, service requirements, tensor internals, and
  public constructor signatures;
- npm package names and dependency relationships;
- backend contract and capability shapes before their first stable
  release;
- checkpoint metadata and other persisted formats;
- native addon exports, handle classes, and compiled-program artifacts;
- inference, tokenizer, and safetensors APIs coupled to the current
  addon.

Old APIs are removed rather than deprecated by default. Core does not
carry compatibility adapters after the corresponding migration phase.

## Terminology

The architecture distinguishes concepts currently collapsed into the
string `"metal"`:

| Concept | Meaning |
|---|---|
| Backend | An implementation family/package, such as CUDA, ROCm, native, PJRT, or remote |
| Runtime | A live instance owning clients, contexts, queues, allocators, caches, and sessions |
| Device | A physical or logical endpoint owned by a runtime |
| Placement | A device plus a memory space or placement policy |
| Tensor handle | An opaque lazy or concrete value owned and interpreted by one runtime, carrying only static metadata |
| Executable handle | An opaque compiled artifact owned by one runtime |

A backend is code. A runtime is a resource. A device cannot be compared
correctly without its owning runtime: two values both displayed as
`cuda:0` may belong to different CUDA contexts or processes and are not
necessarily interoperable.

## Design

### The `Runtime.Runtime` service

Core defines a single service for tensor construction and backend
dispatch. The following is illustrative; exact method grouping may
change during implementation, but the ownership boundary is normative:

```ts
export interface RuntimeService {
  readonly identity: object
  readonly backend: BackendInfo
  readonly placement: Placement
  readonly capabilities: Capabilities

  readonly node: (
    request: NodeRequest
  ) => Effect.Effect<LazyTensorHandle, BackendError>

  readonly evaluate: (
    roots: ReadonlyArray<TensorHandle>
  ) => Effect.Effect<ReadonlyArray<ConcreteTensorHandle>, BackendError>

  readonly grad: (
    loss: TensorHandle,
    wrt: ReadonlyArray<TensorHandle>
  ) => Effect.Effect<ReadonlyArray<LazyTensorHandle>, BackendError>

  readonly compile: (
    roots: ReadonlyArray<TensorHandle>
  ) => Effect.Effect<ExecutableHandle, BackendError>

  readonly run: (
    executable: ExecutableHandle,
    inputs: ReadonlyArray<ConcreteTensorHandle>,
    scalars: ReadonlyArray<number>
  ) => Effect.Effect<ReadonlyArray<ConcreteTensorHandle>, BackendError>
}

export class Runtime extends Context.Service<Runtime, RuntimeService>()(
  "@effect-torch/core/Runtime"
) {}
```

Service methods have no Effect requirements of their own. Dependencies
such as configuration, credentials, filesystem access, native addon
loading, or logging are resolved while constructing the Layer, rather
than leaking through every tensor operation.

`identity` partitions backend-owned caches when a compiled function is used
under different runtime services. Core never stores it on tensors or uses it
as a package compatibility/version check.

The first service value is bound to one default placement. A runtime may
internally enumerate multiple devices, but core does not initially
provide an independent `CurrentDevice` string that could disagree with
the runtime. If lexical placement switching becomes necessary, it must
select a placement owned by the same runtime.

Backend packages expose Layers rather than registering globally:

```ts
import { Cuda } from "@effect-torch/backend-cuda"

program.pipe(
  Effect.provide(Cuda.layer({ device: 1 }))
)
```

Local CPU implementations may use an unscoped Layer when acquisition is
trivial. CUDA contexts, PJRT clients, remote channels, background
workers, and similar resources use `Layer.scoped` and
`Effect.acquireRelease`.

### Ambient runtime and tensor handles

One ambient `Runtime.Runtime` service is authoritative for the whole Effect
program. Tensors never retain, select, or provide a runtime. They are the
opaque backend handle and expose only static tensor metadata:

```ts
interface TensorHandle {
  readonly _tag: "LazyTensor" | "Tensor"
  readonly shape: ReadonlyArray<number>
  readonly dtype: DType
  readonly device: string
  readonly placement: Placement
}

interface LazyTensorHandle extends TensorHandle {
  readonly _tag: "LazyTensor"
}

interface ConcreteTensorHandle extends TensorHandle {
  readonly _tag: "Tensor"
}
```

The native class types currently exposed through `lazy`, `materialized`,
compiled programs, decode programs, and KV objects disappear from core's
types. Handles are internal opaque values. The runtime package that
creates a handle is the only component allowed to interpret or destroy
it.

All backend dispatch, including graph construction that needs a factory,
evaluation, autodiff, compilation, readback, disposal, serialization, and
extensions, uses the ambient service. Tensor-like metadata can derive
from inputs, but input tensors never provide runtime dependencies.

Core does not compare or recover runtime identity from tensors. A backend
owns its handles and must reject an incompatible or foreign handle with a
typed `BackendError` at its boundary. Applications provide one runtime
Layer at the program boundary and do not locally override it.

### No stripped internal API

The implementation must not declare or depend on `@internal` APIs. A
symbol is either a normal published contract or an unexported
implementation detail. `stripInternal` remains enabled as a publication
safeguard, but source code must not rely on declaration-stripped fields or
helpers.

### Semantic operation contract

Core and backend packages share a semantic operation
vocabulary. `NodeRequest` is a discriminated union covering public graph
semantics: constructors, elementwise operations, reductions, shape
operations, indexing, linalg, and semantic neural-network operations.
It does not expose backend-lowered nodes such as Metal arena plans,
fused MSL expressions, or backend-specific GEMM epilogues.

A single semantic request method is preferred to an interface with one
method per operation:

- It gives the backend contract one coherent surface.
- Older backends can reject an unknown operation cleanly.
- Native adapters can map requests to existing N-API methods during the
  migration and later accept the request directly.
- Remote implementations can serialize requests into a local graph
  builder without one network call per operation.

Node construction remains local and synchronous in implementation where
possible, but returns an Effect so validation and capability failures
stay typed. No evaluation or remote round trip is implied by `node`.

Core keeps enough static metadata to provide the current synchronous API
experience and validate compatibility before dispatch. Backends validate
the same metadata at their boundary; moving to one authoritative metadata
implementation is part of the Rust extraction, not a prerequisite for
the TypeScript decoupling.

### Package compatibility and capabilities

Every runtime reports:

- its backend package name;
- a stable display description of its default placement;
- dtype and placement constraints;
- available optional features and extensions.

The npm package graph is the compatibility mechanism. Backend packages
declare a semver range for `@effect-torch/core` as a peer dependency, and
their own package version identifies the backend implementation. Core does
not maintain a second runtime protocol, operation-set, or extension version that
can drift from those package versions. Incompatible contract changes use
normal semver and are rejected by package installation rather than by a
duplicate runtime version negotiation path.

The general tensor backend has one required baseline semantic contract. It is
not a bag of optional per-op callbacks with CPU fallback. A backend may
reject an operation/dtype/layout combination at graph construction or
compile time, but the error must identify the unsupported capability at
the earliest boundary.

Capabilities that imply different state or lifecycle models are typed
optional extensions. Initial candidates are:

- compiled inference and stateful KV sessions;
- paged attention and prefix caching;
- collectives and device meshes;
- quantized storage/import;
- remote/server-side model loading;
- external buffer import and export.

### Evaluation and cancellation

`evaluate` preserves the current multi-root contract: one call shares
one graph walk, deduplication cache, random draws, and liveness analysis.
The active backend validates every root handle before evaluation. A future
orchestrator may partition roots, but that behavior is not part of this RFC.

Runtime Effects are interruptible. Fiber interruption requests backend
cancellation; it does not promise that an already submitted GPU or TPU
kernel can be preempted. A runtime must retain input buffers, native
state, and late outputs until underlying work actually completes. Late
outputs from an interrupted call are released rather than published.

Runtime shutdown stops new submissions, requests cancellation where
supported, drains in-flight resource ownership, and then destroys
contexts. Finalizers remain a safety net, not the primary capacity
management mechanism.

### Compilation

Compiled programs contain an opaque backend-owned program handle, not a
runtime service reference. The active runtime validates that handle before
execution. Backend-owned program cache keys include at least:

- the backend runtime instance where isolation requires it;
- backend package and compiler versions;
- placement and memory space;
- input shapes and dtypes;
- compile options and relevant capabilities.

Shape-specialized compilation from RFC 0008 and RFC 0016 remains
unchanged. A cache hit can never reuse an executable across runtime
instances merely because both display the same device name.

Tracing and helper-tensor construction use the ambient runtime. Exemplar
inputs contribute metadata and opaque handles but never select or provide a
runtime. The backend rejects foreign exemplar or helper handles before
compilation.

Programs and tensors are separate opaque handle classes in the contract.
The runtime validates executable and concrete-input compatibility before dispatch;
core does not compare runtime identities.

### Transfers

The baseline contract does not pass runtime services to tensor operations
and does not provide an in-program cross-runtime transfer operation. Data
movement between independently provided runtime programs is explicit: read
back values from the source program and construct a new tensor in the target
program. A later application-level extension may
negotiate peer-to-peer, shared-memory, DLPack-like, or external GPU resource
import, but zero-copy is not promised by the base API.

There is no implicit transfer for binary operations, evaluation,
compilation, model execution, or fallback. This keeps performance,
precision, and synchronization visible.

### Errors

Backend implementations return structured `BackendError` values rather
than throwing across the service boundary. Core maps them to the public
error appropriate to the operation while preserving backend name,
operation, phase, and diagnostic details.

The backend error taxonomy must distinguish at least:

- unavailable backend or device;
- incompatible core/backend package contract;
- unsupported operation, dtype, layout, or placement;
- invalid or foreign handle;
- compilation failure;
- execution failure;
- transfer failure;
- cancelled operation;
- closed runtime.

Remote backends may add retryability, authentication, and transport
details without flattening them into message strings.

### Backend selection

Core does not probe optional packages or maintain a global mutable
registry. Applications compose the backends they installed. A future
selection helper accepts explicit candidate Layers and falls back only
on typed availability errors; authentication, configuration, or
compilation failures must not silently select CPU.

No combined `Best` Layer is provided. Applications import one backend package
and provide its Layer explicitly; application-owned fallback policy must handle
typed availability errors without hiding failures or transfers.

### Inference extensions

The existing `DecodeProgram`, `NativeKvPool`, and `NativeKvSequence`
encode one CPU/Metal inference strategy. They do not belong in the base
runtime contract.

Inference becomes an optional typed runtime extension with shared
high-level semantics but backend-owned physical state. For example:

- CUDA may use paged device memory and continuous batching.
- PJRT may use static or donated cache buffers compiled into the
  executable signature.
- A remote runtime may return a server-owned generation session and a
  token stream without exposing KV buffers at all.

Inference artifacts, pools, and sessions carry stronger backend-owned
relationships than ordinary tensors: compiled artifact, model or pool, and
session generation must match. The active runtime validates those opaque
handles. Stateful resources are scoped and explicitly releasable; GC
finalization is fallback only.

### Tokenizers and safetensors

Tokenization is a CPU utility, not a tensor backend. The tokenizer package
returns `Uint32Array`; `Tensor.fromTypedArray` snapshots that token data into
the selected runtime. The tokenizer cannot construct another addon's tensor
handle.

Safetensors is an optional direct-loading runtime extension. It consumes
runtime-owned tensor handles for writes and returns concrete tensor handles
plus structured archive metadata for reads. Core does not define a second
host-side tensor representation or a byte codec API.

A local path is not meaningful to every runtime. Remote APIs must
distinguish client bytes, URLs or object-store references, and
server-local paths. Requested placement is never silently changed while
loading.

Local runtimes may expose a structured path extension carrying named entries
and metadata. The native extension uses the official Rust `safetensors` crate,
writes through a same-directory temporary file and rename, rejects malformed
layout and unsupported placement (including f64 on Metal), and never falls
back to CPU. `Tensor.save` and `Tensor.load` retain the direct path API;
`Tensor.loadArchive` returns metadata explicitly.

## Native Architecture

The pre-CUDA Rust implementation is separated into reusable crates. Phase 3
uses these boundaries:

```text
effect-torch-graph
  production nodes, static metadata validation, traversal/remapping, leaf ownership

effect-torch-autodiff
  reverse-mode gradients, vmap, checkpoint transforms

effect-torch-compiler
  neutral fusion IR and CPU interpreter, graph rewrites, frozen-program planning

effect-torch-runtime
  dtype, layout, errors, cancellation and runtime execution contracts

effect-torch-runtime-cpu
effect-torch-runtime-metal
  storage, allocators, kernels, lowering, synchronization

effect-torch-napi
  shared cancellation, async-work, readback ownership and lifecycle support
```

Native npm packages statically link the graph version and runtime crates
they support. No Rust object crosses between `.node` addons. This permits
normal Rust traits, enums, `Arc`, and futures internally without claiming
an ABI stability Rust does not provide.

The graph representation is layered over time:

1. semantic graph: stable framework meaning (`Add`, `Linear`, `Sdpa`,
   `LayerNorm`, optimizer semantics);
2. normalized/differentiated graph;
3. backend-lowered graph (`FusedElementwise`, GEMM epilogues, arena
   plans, launch-specific nodes);
4. optional stateful inference graph.

Backend-specific lowered nodes do not leak into the TypeScript semantic
contract consumed by another backend. Phase 3 moves the authoritative graph
and validation into `effect-torch-graph`; its remaining lowered variants are
the allowed incremental seam. Neutral fusion expressions and rewrite logic
are owned by `effect-torch-compiler`, while Metal emission, kernels, and arena
planning are owned by the macOS-gated `effect-torch-runtime-metal` crate.

Rust traits are not the plugin boundary. A project-specific C function
table would require permanent decisions about allocation, ownership,
async callbacks, cancellation, panic containment, loading/unloading,
code signing, and linked driver ABIs. That work is deferred until a real
third-party binary-plugin requirement exists.

## Package Architecture

The intended package topology is:

```text
@effect-torch/core
@effect-torch/backend-cpu
@effect-torch/backend-apple-native
@effect-torch/tokenizers
```

Each backend package owns one runtime, one Layer, and one package-local N-API
addon. CPU never exposes or selects Metal. Apple Metal never exports CPU and
does not silently execute Metal graphs on CPU. Both addons statically link the
shared root Rust graph, autodiff, compiler, runtime, and N-API implementation
crates. Tokenizers owns a separate tokenizer-only addon and returns host-owned
token ids.

Node-API stabilizes the Node-facing ABI, not the ABI of CUDA, ROCm,
Metal, PJRT, or their linked libraries. Each backend package owns its
prebuild matrix, runtime driver probing, diagnostics, and release cadence.

## PJRT and TPU

PJRT validates the runtime-oriented model. A PJRT client owns devices
and memory spaces; buffers and loaded executables retain client
affinity; execution and buffer readiness may be asynchronous.

PJRT is not an operation API and does not execute effect-torch's current
Rust graph directly. A PJRT backend requires:

```text
effect-torch semantic graph
  -> neutral transforms and autodiff
  -> StableHLO lowering
  -> PJRT compile
  -> PJRT loaded executable
  -> PJRT buffers and readiness futures
```

StableHLO is the compiler interchange format, not necessarily the
framework's complete graph representation. Effect-torch-specific
inference state may lower to ordinary StableHLO, buffer aliasing or
donation, custom calls, or a separate extension depending on the target.

The PJRT package uses the existing versioned PJRT C plugin API. It does
not cause effect-torch to introduce a second dynamic plugin ABI.

## Backend Conformance

Core's existing semantic tests run against CPU and every available Apple Metal
placement through the test device helper. There is no conformance package,
runner abstraction, or second set of wrapper tests. The shared suite covers:

- constructors and strict dtype behavior;
- every baseline operation, shape, broadcasting, and error contract;
- multi-root evaluation and random-node consistency;
- autodiff and finite-difference checks;
- compilation, cache isolation, concurrent calls, and scalar slots;
- cancellation and late-resource cleanup;
- explicit clear/disposal and use-after-close behavior;
- direct serialization round trips;
- foreign-handle rejection;
- memory boundedness under deferred GC.

Optional extensions publish separate suites. Backend-specific numerical
tolerances remain explicit; unsupported combinations fail instead of
being omitted silently.

## Alternatives Considered

### Keep one native addon and add device variants

Rejected as the target. It preserves maximum Rust code sharing but ties
all toolchains, binaries, platform dependencies, and release cycles to
one package. TPU and remote execution remain unnatural. The monolithic addon
was deleted when the package-local backend addons landed.

### Move the graph and transforms to TypeScript

Deferred. A TypeScript-owned serializable IR would make arbitrary
backend languages and remote transport straightforward and eliminate
one N-API call per graph node. It would also relocate the mature graph,
roughly 85 node variants, validation, autodiff, vmap, fusion, decode
rewrites, evaluator semantics, and liveness logic to the JavaScript
heap. It is too large and risky a prerequisite for backend decoupling.

A portable graph serialization may be added later without making
TypeScript the graph execution engine.

### Rust dynamic plugin ABI

Rejected. Rust traits are not ABI-stable. A custom C ABI is possible but
premature while first-party native backends can statically link shared
crates and PJRT already supplies a suitable plugin ABI for its ecosystem.

### Global per-operation dispatcher

Rejected. It is optimized for eager frameworks, loses whole-graph
placement and compilation, introduces registration and version conflicts,
and is unsuitable for remote or TPU submission. Backend-local kernel
dispatch after lowering remains an implementation detail.

### ONNX Runtime-style graph partitioning and fallback

Deferred. Provider capability negotiation and subgraph compilation are
proven mechanisms, but automatic partitioning requires transfer costs,
scheduling, numerical policy, and failure semantics that the project
does not yet need. One graph is interpreted by one backend runtime in the
first version.

## Migration

### Phase 1: TypeScript boundary

1. Add the `Runtime.Runtime` service contract, opaque handle types, and
   structured backend errors.
2. Wrap the native graph implementation behind the runtime boundary.
3. Route all tensor construction, operations, evaluation, autodiff, and
   compilation through the runtime.
4. Remove direct native-addon imports and dependencies from core.
5. Replace `DeviceKind` and the dumb `CurrentDevice` service with
   runtime-owned placement metadata.
6. Require backends to reject foreign or incompatible handles at every
   multi-handle boundary.

Each backend package exposes one logical runtime instance. The Apple package
does not pretend that the current process-global Metal state is multiple
isolated clients.

### Phase 2: Peripheral boundaries

1. Move tokenizer-native integration behind a tokenizer package returning
   token arrays.
2. Move safetensors behind optional direct runtime I/O.
3. Move decode/KV APIs behind an inference extension.
4. Remove hard-coded best-device selection and require an explicit backend Layer.

### Phase 3: Shared native crates (implemented)

1. The production graph, metadata validation, traversal/remapping, and leaf
   ownership live in `effect-torch-graph`.
2. Reverse-mode gradients, vmap, and checkpoint transforms live in
   `effect-torch-autodiff`; neutral fusion, rewrites, CPU expression
   interpretation, and frozen slot planning live in `effect-torch-compiler`.
3. CPU and macOS-gated Metal storage and kernels live in their runtime crates
   and implement the shared runtime contracts; reusable cancellation and
   readback lifecycle mechanics live in `effect-torch-napi`.
4. The CPU and Apple packages each own a thin N-API wrapper and binary. Shared
   orchestration source remains in the root N-API crate; there is no generic
   npm native package or combined backend adapter.

### Phase 4: Independent accelerators (deferred)

1. Ship CUDA as a separate addon and run the full backend suite.
2. Validate binary packaging, availability errors, lifecycle, compiled
   caches, and memory ownership on real hardware.
3. Add ROCm as an independent package using the same semantic crates.
4. Keep every accelerator in its own package rather than adding it to either
   existing backend addon.

### Phase 5: Compiler and remote runtimes

1. Add semantic graph serialization or a direct StableHLO lowerer as
   required by the PJRT implementation.
2. Add the PJRT runtime and test against an available plugin before
   claiming TPU support.
3. Add a remote runtime with structured transport/authentication errors,
   server-side lifecycle, and explicit data-source semantics.

## Risks

- **False isolation around global native state.** The current Metal
  device and caches are process-global. The adapter must expose their
  real ownership until the native runtime is made instance-based.
- **Graph-semantic drift between packages.** Shared Rust crates,
  peer-dependency semver ranges, and one conformance suite are the
  mitigation.
- **Handle lifetime across Effect scopes.** Escaped tensors can outlive
  a scoped runtime value. Closed runtimes reject new use; releases drain
  in-flight ownership; tests cover use-after-close and worker teardown.
- **Compiled cache aliasing.** Device display strings are insufficient.
  Each backend owns cache isolation and validates every executable and input
  handle before dispatch.
- **Inference over-generalization.** Forcing CPU/Metal paged-KV objects
  onto PJRT or remote serving would produce a leaky base interface.
  Typed extensions keep physical state backend-owned.
- **Contract growth.** A giant optional API becomes impossible to
  implement honestly. The semantic baseline stays small; incompatible
  changes follow package semver;
  stateful domains use extensions.
- **Packaging burden.** Separate CUDA and ROCm prebuilds increase release
  work. This is intentional: their dependencies and compatibility
  matrices already differ and should not make core installation fragile.
- **Performance regression during indirection.** Graph construction adds
  one TypeScript service dispatch but no device work. Compilation and
  evaluation remain one backend call. Benchmarks gate the adapter.

## Acceptance Criteria

1. `@effect-torch/core` imports and depends on no native package.
2. The existing CPU and Metal suite passes through a `Runtime`
   adapter with unchanged semantics and no material performance
   regression.
3. Tensor, buffer, program, decode, and KV native classes no longer
   appear in core's public or internal backend-neutral types.
4. Different runtime Layers can execute independent programs in one Node
   process; a backend rejects foreign handles at the earliest boundary even
   when placement display names match.
5. Fiber interruption and runtime shutdown have tested, bounded resource
   behavior.
6. The core suite passes against every installed first-party backend.
7. No operation silently falls back or transfers to CPU.
8. Incompatible core/backend package versions are rejected through the
   backend's peer dependency before Layer construction.
9. PJRT/TPU support is not declared until an effect-torch graph lowers,
    compiles, executes, and transfers buffers through a real PJRT plugin.

## Open Questions

1. Whether semantic graph serialization should be an effect-torch schema
   or only StableHLO lowering. The first remote/PJRT implementation will
   provide the concrete requirement.
2. Which inference features form the first extension contract. The
   current API is the behavioral reference, not a required physical
   representation.

## Prior Art

- Effect services and Layers: dependency declaration, scoped acquisition,
  and construction-time dependency resolution.
  <https://effect.website/docs/requirements-management/services/>
  <https://effect.website/docs/requirements-management/layers/>
- PJRT: opaque clients, devices, memory spaces, buffers, loaded
  executables, and asynchronous readiness.
  <https://openxla.org/xla/pjrt>
  <https://openxla.org/xla/pjrt/cpp_api_overview>
- StableHLO: a versioned compiler interchange operation set.
  <https://openxla.org/stablehlo/spec>
- ONNX Runtime execution providers: independently packaged providers,
  capability discovery, allocators, transfers, and subgraph compilation.
  <https://onnxruntime.ai/docs/execution-providers/>
- PyTorch PrivateUse1: demonstrates the secondary integration surface of
  a per-op dispatcher and its limitation to one out-of-tree backend key.
  <https://docs.pytorch.org/tutorials/advanced/privateuseone.html>
- IREE HAL: explicit separation of drivers, devices, buffers, executable
  dispatch, and synchronization.
  <https://iree.dev/reference/bindings/c-api/#hardware-abstraction-layer-hal>

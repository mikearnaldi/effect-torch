# RFC 0022: Pretrained Quantized Inference - Model Registry, Artifact Binding, and Portable Packed Execution

- **Status**: Draft
- **Created**: 2026-08-11
- **Depends on**: RFC 0005 (models), RFC 0010 (inference), RFC 0012
  (dtypes), RFC 0017 (multi-backend runtime), RFC 0019 (executable
  compilation), RFC 0020 (invocation ownership), RFC 0021 (compiler
  pipeline)
- **Updates**: RFC 0015's quantized-format non-goal and RFC 0017's
  quantized-storage extension point

## Summary

Add native loading and inference for pretrained quantized models while
preserving the existing model contract:

```text
Model.Model + Model.Params -> Model.inference
```

A locally available artifact is inspected without a tensor runtime. Its
architecture identifiers are resolved through an Effect service populated by
explicit model-registration Layers. The selected architecture implementation
constructs an ordinary `Model.Model` and a complete parameter-binding plan.
The plan is validated against the artifact before any multi-gigabyte tensor is
materialized. Hydration then preflights the complete plan's import/storage and
generic-consumer requirements against the active runtime before importing any
selected tensor range and returns `Model.Params` in `model.names` order.

The complete path is:

```text
pinned local files
    -> format-specific artifact catalog
    -> architecture-key extraction
    -> ModelRegistry lookup
    -> architecture-specific config and binding plan
    -> complete validation
    -> runtime capability preflight and hydration
    -> Model.Model + Model.Params
    -> Model.inference
```

Hugging Face is an acquisition source, not an execution API or file format.
Downloading, revision resolution, authentication, caching, and offline policy
remain a separate phase. This RFC begins with local files and never combines
"download" with "run".

Quantized weights are not new arithmetic dtypes and do not pretend to be
ordinary floating tensors. A quantized weight is an immutable descriptor over
ordinary physical component tensors, such as packed `u8` data and optional
scale or zero-point tensors. Dedicated `QuantizedLinear` and
`QuantizedEmbedding` semantic nodes consume those components and produce
ordinary compute tensors. The components remain flattened into `Model.Params`,
so graph discovery, binding ownership, memory accounting, executable
invocation, and inference retention continue to use the existing tensor
machinery.

Every accepted storage format has a correctness path on each runtime that
advertises it. An optimized kernel is optional. CPU begins with a direct block
decoder and dot-product implementation. Metal uses shared GEMV/GEMM skeletons
parameterized by a format decoder, decoding blocks into registers or
threadgroup memory. Exact hand-tuned kernels and statically planned workload-
specific algorithms are lowering choices beside the semantic graph; value-
dependent execution repacking is deferred. An unsupported optimized Metal path
therefore falls back to generic packed execution rather than expanding the
complete model or silently moving the operation to CPU.

Architectures remain separate installed TypeScript Layers. Compiler and backend
code implements reusable operations and encodings only and contains no Muse or
other model-specific branches, so applications can register custom
architectures through the same public service.

The first end-to-end target is text-only inference for:

```text
unsloth/Muse-Glimmer-30B-GGUF
Muse-Glimmer-30B-UD-Q2_K_XL.gguf
```

This artifact is approximately 11.59 GiB and mixes F32, Q2_K, Q3_K, Q4_K,
Q5_K, and Q6_K tensors. Calling it "2-bit" does not mean implementing Q2_K
alone is sufficient.

## Decision

The following rules are normative.

1. There is one model abstraction. Directly initialized, densely loaded, and
   quantized pretrained architectures all produce `Model.Model` plus
   `Model.Params`.
2. Network acquisition is separate from local artifact inspection, model
   reconstruction, parameter hydration, compilation, and execution.
3. Model architecture implementations are resolved by a `ModelRegistry`
   Effect service. Architecture packages contribute explicit Layers; there is
   no process-global registration or runtime code downloaded from a model
   repository.
4. Backends and compiler passes are architecture-blind. They implement reusable
   semantic operations and storage encodings only; they never branch on Muse,
   Llama, Qwen, architecture keys, source tensor names, or layer numbers.
5. Artifact readers parse containers. They do not implement model semantics,
   choose execution kernels, or allocate backend tensors.
6. Architecture implementations interpret model configuration, construct the
   model graph, and own semantic weight binding: expected names, shapes,
   aliases, ties, synthesis, and format-specific transforms.
7. `Model` gains a public definition constructor and an explicit physical
   parameter schema. `model.names` is derived from unique physical storage
   slots; aliases and tied logical uses never duplicate ownership slots.
8. Quantized pretrained parameterizations are load-only until a format
   quantizer exists. Their `init` fails explicitly with `ModelError`; it never
   invents invalid packed bytes or silently initializes a dense variant.
9. Artifact preparation validates architecture and source bindings without a
   runtime. Hydration then obtains the active runtime, preflights every required
   import/storage and shape-independent generic consumer capability, and imports
   no ranges unless the complete hydration preflight succeeds. Graph- and shape-
   complete admission occurs during compilation.
10. Missing, ambiguous, duplicate, malformed, or incompatible tensors fail
    without a partially loaded model.
11. Quantized storage formats are not added to `Runtime.DType`. Packed payloads
    use their actual physical dtypes and shapes.
12. A quantized weight descriptor is not itself a tensor or opaque runtime
    resource. Its physical components are ordinary tensor graph dependencies
    consumed by dedicated semantic operations.
13. `QuantizedLinear` and `QuantizedEmbedding` define their result as if the
    logical weight were dequantized, but lowering is not required to materialize
    that logical weight.
14. A storage format is supported on a runtime only when a generic correctness
    decoder exists there. Optimized-kernel coverage is a separate capability.
15. Backend algorithm selection happens during compilation and is recorded in
    the authoritative lowered program and diagnostics. Execution does not
    discover unsupported kernels after work has begun.
16. Quantized fallback never silently partitions one graph across independent
    runtimes, transfers an operation to CPU, or persistently expands the whole
    model to a dense dtype.
17. `optimize: false` remains correct. It selects generic decoder-based
    execution, but it does not reject a format merely because an optimized
    kernel was disabled.
18. Packed source encoding and backend execution packing are distinct. This
    RFC hydrates canonical source encodings; value-dependent execution repacking
    is deferred until an explicit shared prepared-weight lifetime is designed.
19. Grouped-query attention, per-node attention windows, RoPE layout, and
    per-retention-group KV state are semantic contracts, not Muse-specific
    backend exceptions.
20. Model-program serialization is not part of this RFC. Existing model and
    trainer state persistence remain separate concerns.

## Motivation

### The library has model execution but not pretrained model loading

`Model.Model` already describes a reusable parameterized architecture:

- stable ordered parameter names;
- an initialization recipe;
- a `forward(params, input)` graph builder;
- a compiled execution cache;
- the inference specialization that traces the same `forward` graph.

Current `Model.save` and `Model.load` persist or restore values against a model
the caller has already constructed. FineWeb reconstructs its GPT architecture
from TypeScript code before loading safetensors. That is the correct state
checkpoint relationship, but it does not answer how a local external artifact
selects installed architecture code, maps foreign names and layouts, or retains
packed quantized storage.

### Known-model deployment requires installed semantics

Hugging Face repositories and GGUF files normally contain configuration,
tokenizer assets, and named tensors. They do not generally contain an
executable model graph. GGUF explicitly reserves computation graphs as future
work.

vLLM's apparent "model ID to serving" path still performs these stages:

```text
config.json
    -> installed architecture registry
    -> empty model construction
    -> checkpoint loader
    -> model-specific load_weights
    -> execution
```

Its GGUF plugin also constructs an installed vLLM/Transformers architecture,
maps GGUF names to that implementation, and substitutes quantized
linear/embedding execution. Unknown model semantics still require installed
code. Effect Torch adopts the same fundamental separation without depending on
vLLM, Transformers, llama.cpp, or a remote-code mechanism.

### Quantization is storage plus consuming operations

RFC 0012 distinguishes compute dtypes from storage formats. The current runtime
depends on that distinction:

- `DType::size_in_bytes` assigns one fixed size to each scalar;
- `Layout` describes logical element strides and offsets;
- program bindings validate one shape, dtype, placement, and layout;
- generic operations assume ordinary scalar tensors;
- readback and safetensors serialize logical scalar arrays.

GGML K-quants do not fit those contracts. One block stores packed codes plus
shared scales and minima; bytes per logical element are fractional and depend
on block geometry. Treating Q2_K as a dtype or pretending that Q2_K bytes are a
logical f32 tensor would make memory accounting, generic operations, cache keys,
readback, and invocation validation dishonest.

The storage remains quarantined behind operations that understand it, exactly
as RFC 0012 anticipated for a `QuantizedLinear`-style node.

### Optimized coverage cannot define correctness coverage

Muse-Glimmer's Dynamic 2.0 file mixes several encodings. Future artifacts will
add affine int4/int8, I-quants, floating block formats, and vendor-specific
packed layouts. Requiring a hand-written Metal GEMM for every format before the
file can execute makes architecture coverage move at the speed of kernel
optimization.

The projects with broad packed support separate decoding from tuned algorithms:

- llama.cpp attaches per-format decoders to shared CPU and Metal matmul
  skeletons;
- MLX decodes affine packed tiles inside shared quantized GEMV/GEMM kernels;
- MLC represents dequantization semantics and relies on compiler fusion and
  generic GPU schedules;
- vLLM uses direct MMVQ/MMQ where available and dequantization-based paths for
  unsupported workload/format combinations;
- torchao keeps ordinary linear module semantics while specialized weight
  representations dispatch supported operations.

Effect Torch requires a generic decoder path as the admission criterion and
treats hand-tuned kernels as later lowering improvements.

### Full-model dequantization defeats the target

The Muse text model has approximately 27.85 billion parameters. Expanding the
complete model to f16 would require roughly 52 GiB before KV state, activations,
workspace, allocator capacity, or the vision and speculative companions. A
fallback that loads the 11.59 GiB packed file and silently creates a dense copy
does not support the intended deployment.

Fallback must preserve packed residency and use bounded temporary storage.

## Goals

1. Load local known-model artifacts into the existing `Model.Model` and
   `Model.Params` contract.
2. Make architecture support explicitly composable through Effect Layers.
3. Keep acquisition, artifact parsing, architecture semantics, weight binding,
   storage import, and execution as separate testable boundaries.
4. Parse GGUF v3 metadata and tensor catalogs without reading the complete file
   through JavaScript or allocating device tensors.
5. Support split artifacts and large tensor ranges without a full-file host
   copy.
6. Validate model configuration and every required weight before materializing
   the first large parameter.
7. Support mixed quantization formats within one model and one layer stack.
8. Preserve correct inference when an optimized CPU or Metal kernel for the
   exact format is absent.
9. Keep packed memory resident through decode and ordinary prefill workloads.
10. Record algorithm and fallback choices in executable diagnostics and static
    memory plans.
11. Add the modern transformer semantics needed by Muse-Glimmer as reusable
    Model/Tensor/compiler features.
12. Establish an extension path for Llama, Qwen, Gemma, and other installed
    architecture Layers without changing artifact or runtime fundamentals.
13. Make load-only parameterization and physical component schemas explicit in
    `Model`, rather than leaving artifact loaders to construct structural model
    objects or infer arity from names alone.

## Non-goals

- Download models, resolve Hub revisions, manage authentication, or define a
  combined download-and-run API.
- Execute repository-provided TypeScript, JavaScript, Python, native code, or
  another form of `trust_remote_code`.
- Wrap llama.cpp, vLLM, MLX, Transformers, or another model runtime.
- Serialize arbitrary effect-torch `Model.forward` closures or portable model
  programs.
- Resume training from quantized inference parameters.
- Quantization-aware training, straight-through estimators, or gradients with
  respect to packed components.
- Automatically quantize a dense model into every supported artifact format.
- Preserve a packed artifact's encoding through existing `Model.save` or
  safetensors without an explicit quantization manifest.
- Silently partition a graph between CPU and Metal.
- Add a dynamic Rust or C plugin ABI for third-party quantization codecs.
- Implement Muse-Glimmer vision or DFlash in the first text-only milestone.
- Implement a general Jinja engine or OpenAI-compatible server in this RFC.
- Promise successful execution when physical memory or address-space capacity
  is insufficient.
- Implement config-plus-safetensors artifact loading in the first milestone.
  GGUF v3 is the only artifact format implemented by this RFC; the catalog and
  registry boundaries deliberately leave room for dense formats.

## Terminology

| Term | Meaning |
|---|---|
| Acquisition | Producing pinned local files from a repository, object store, removable medium, or application bundle |
| Artifact format | Physical container convention such as GGUF or a Hugging Face config plus safetensors shards |
| Artifact catalog | Backend-free metadata and named tensor descriptors over local byte sources |
| Architecture key | Namespaced installed-model identifier such as an HF architecture, HF model type, or GGUF architecture |
| Architecture implementation | Installed code that decodes config, constructs a `Model`, and binds source tensors |
| Binding plan | Complete validated mapping from artifact entries to ordered physical model parameter components |
| Hydration | Importing planned tensor ranges into one active runtime and producing concrete `Model.Params` |
| Logical weight | Mathematical dense matrix or embedding table represented by packed storage |
| Storage encoding | Canonical persisted quantization format and version, such as GGML Q2_K |
| Execution packing | Optional backend-prepared compressed representation used by a kernel |
| Generic decoder path | Correct packed execution using a shared operation skeleton plus a format decoder |
| Accelerated path | Shape/device-specific hand-tuned or workload-specific execution faster than the generic path |

## Architecture

### End-to-end flow

```text
application-owned acquisition
        |
        | pinned local file set
        v
artifact reader
  - format validation
  - metadata
  - tensor names, shapes, encodings
  - file/range ownership
  - tokenizer/config assets
        |
        | ArtifactCatalog
        v
architecture-key extraction
        |
        | exact namespaced keys
        v
ModelRegistry.resolve
        |
        | installed architecture implementation
        v
architecture preparation
  - canonical config
  - ordinary Model.Model
  - complete BindingPlan
        |
        | validate all entries and transforms
        v
runtime hydration
  - acquire active Runtime.Runtime
  - preflight every import/storage and generic consumer requirement
  - path/range import
  - concrete physical component tensors
        |
        | Model.Params in model.names order
        v
Model.inference(model, params, config)
        |
        | semantic graph, including QuantizedLinear/Embedding
        v
decode specialization -> optimization plan -> backend lowering
        |
        | exact or generic packed instructions
        v
execute
```

### Architecture registry service

The registry is an Effect service containing a private atomically updated map.
Mutation is available only to the implementation of registration Layers; the
public service is read-only and it is never a module global. Registration adds
installed code, not model values or runtime state.

An illustrative contract is:

```ts
interface ArchitectureKey {
  readonly namespace: string
  readonly value: string
}

declare const ArchitectureKey: {
  readonly make: (
    namespace: string,
    value: string
  ) => ArchitectureKey
}

interface ArchitecturePreparation {
  readonly model: Model.Model
  readonly bindings: BindingPlanDraft
}

interface ModelArchitecture {
  readonly id: string
  readonly keys: ReadonlyArray<ArchitectureKey>
  readonly prepare: (
    artifact: ArtifactCatalog
  ) => Effect.Effect<ArchitecturePreparation, ArchitecturePreparationError>
}

interface ModelRegistryService {
  readonly resolve: (
    keys: ReadonlyArray<ArchitectureKey>
  ) => Effect.Effect<ModelArchitecture, ModelRegistryError>
}
```

`ModelArchitecture`, `ModelRegistry`, and `registerModel` are public extension
points in `@effect-torch/core`. A custom architecture supplies the same
interface as a built-in architecture; it does not subclass a backend type or
modify a central enum. Its `prepare` implementation may use any application
code already installed in the process, but artifact files themselves cannot
install that code. `ArchitectureKey.make(namespace, value)` validates a
non-empty namespaced key; the type is not closed over HF or GGUF ecosystems.

`ModelRegistryLive` is one stable Layer value creating a fresh map and a private
registration capability per Layer graph. `registerModel` snapshots and freezes
the supplied ID, keys, and callback, then uses a scoped internal registration
operation and re-emits only the read-only service while providing
`ModelRegistryLive` locally:

```ts
const registerModel = (
  architecture: ModelArchitecture
): Layer.Layer<ModelRegistry, ModelRegistryError> =>
  Layer.effect(
    ModelRegistry,
    Effect.gen(function*() {
      const registry = yield* ModelRegistry
      const registration = yield* ModelRegistryRegistration
      yield* Effect.acquireRelease(
        registration.add(snapshot(architecture)),
        registration.remove
      )
      return registry
    })
  ).pipe(Layer.provide(ModelRegistryLive))
```

Architecture packages expose ordinary Layers:

```ts
export const Muse = registerModel(museGlimmer)
export const Llama = registerModel(llama)
export const Qwen = registerModel(qwen)
```

Applications explicitly choose support:

```ts
const Models = Layer.mergeAll(Muse, Llama, Qwen)
```

An application-defined architecture is identical:

```ts
export const MyArchitecture = registerModel({
  id: "acme.my-transformer.v1",
  keys: [ArchitectureKey.make("hf.model_type", "acme_transformer")],
  prepare: prepareAcmeTransformer
})

const Models = Layer.mergeAll(Muse, MyArchitecture)
```

All registration Layers reference the same `ModelRegistryLive` Layer identity,
so normal Layer memoization builds one registry in that graph. The helper must
not construct a new live Layer per registration or apply `Layer.fresh`.

One atomic `Ref.modify` validates a non-empty unique architecture ID, duplicate
keys within the contribution, and collisions against all existing IDs/keys
before inserting anything. It returns an unregistration token tied to exactly
that snapshot. If any sibling of `Layer.mergeAll` fails or construction is
interrupted, Layer scope finalization removes every contribution installed by
that failed build; retry cannot observe a partial registry. Consumers cannot
register after construction because the mutation capability is not exported.

Artifact readers provide ordered `architectureCandidates` on the catalog. The
registry performs exact lookups in that order and rejects no match or matches to
multiple distinct architecture IDs. Multiple candidates resolving to the same
installed architecture are not ambiguous. HF/GGUF metadata key interpretation
therefore remains in each format reader rather than in the registry.

Registration Effects do not require `Runtime.Runtime`. Architecture entries are
pure installed code and configuration decoders.

### Architecture-blind compiler and backends

The official Muse implementation is TypeScript architecture code in a separate
`@effect-torch/models` package and is exposed as a selectable `Muse` Layer. Core
contains only the registry contract, artifact abstractions, and reusable tensor
semantics. The CPU and Apple packages and the Rust compiler contain no official
or custom architecture implementation.

Package dependencies are normative:

```text
@effect-torch/artifact-gguf -> @effect-torch/core
@effect-torch/models       -> @effect-torch/core
backend packages           -> @effect-torch/core runtime contracts
application                -> selected reader + models + backend
```

`inspectGguf` belongs to `@effect-torch/artifact-gguf`; generic
`prepare`/`hydrate`, the registry, binding constructors, and semantic operations
belong to core. Core, compiler crates, and backend packages may not depend on
`@effect-torch/models` or the GGUF reader. The model package may inspect generic
catalog metadata during preparation but cannot call backend-specific APIs.

After architecture preparation, downstream systems can observe only ordinary
graph structure and generic descriptors such as:

- quantized linear or embedding plus an encoding descriptor;
- grouped-query scaled dot-product attention and its head ratio;
- causal or sliding-window attention policy;
- rotary layout, dimensions, positions, and theta;
- RMS normalization options;
- elementwise sigmoid, multiply, residual, and logit transforms;
- decode-state retention groups.

It is forbidden for backend code to inspect an architecture ID, GGUF metadata
key, source tensor name, or a Muse layer index to select behavior. If Muse
reveals a missing operation, the implementation adds that reusable operation to
Tensor/compiler/runtime contracts, and the Muse Layer composes it. The same
operation is then available to `MyArchitecture` without backend changes.

Storage codecs are also architecture-independent. A GGML Q4_K decoder accepts
the validated Q4_K descriptor and physical blocks regardless of which
architecture referenced them. Architecture code decides which source tensor
becomes which graph parameter; backend code does not.

### Model definition and physical parameters

The current `Model.Model` shape remains the public inference contract. The
current private constructor is insufficient for installed architecture modules,
however, and `model.names` alone cannot describe the arity or physical shape of
a packed parameterization. This RFC therefore adds a public definition
constructor and a physical parameter schema. It does not replace `Model.Model`
with an artifact-specific class.

```ts
export type PhysicalParameterStorage =
  | {
      readonly _tag: "Dense"
    }
  | {
      readonly _tag: "QuantizedComponent"
      readonly encoding: QuantizedEncoding
      readonly component: QuantizedComponent
    }

export type ParameterInitialization =
  | {
      readonly _tag: "Initializer"
      readonly make: Effect.Effect<
        Tensor.Any,
        Tensor.TensorError,
        Runtime.Runtime
      >
    }
  | {
      readonly _tag: "ArtifactOnly"
      readonly reason: string
    }

export interface PhysicalParameterSlot {
  readonly name: string
  readonly shape: ReadonlyArray<number>
  readonly dtype: Runtime.DType
  readonly storage: PhysicalParameterStorage
  readonly initialization: ParameterInitialization
}

export interface QuantizedParameterSchema extends QuantizedWeightDescriptor {
  readonly id: string
  readonly componentSlots: ReadonlyArray<string>
}

export interface PhysicalParameterSchema {
  readonly slots: ReadonlyArray<PhysicalParameterSlot>
  readonly quantized: ReadonlyArray<QuantizedParameterSchema>
}

export interface Definition {
  readonly parameters: PhysicalParameterSchema
  readonly forward: Model["forward"]
}

export declare const define: (
  definition: Definition
) => Effect.Effect<Model, ModelError>

export declare const validateParams: (
  model: Model,
  params: Params
) => Effect.Effect<void, ModelError>
```

`Model.define` is the only new public construction boundary. It validates that
each physical slot has a unique non-empty name, legal static shape, legal
physical dtype/storage pairing, and one initialization policy. It derives
`model.names` from `parameters.slots`; callers cannot supply a second name list.
Multiple logical uses may index the same slot, so aliases and tied embeddings do
not create duplicate storage ownership.

Every `QuantizedComponent` slot belongs to exactly one quantized group, in the
same order as that group's component schema; dense slots belong to none. Group
IDs are unique logical parameter identities. Multiple graph uses of a tied
weight reference the same group and component slot indices.

The schema describes physical graph inputs, not logical dequantized matrices.
A dense matrix has one `Dense` slot. A quantized parameterization has one or
more `QuantizedComponent` slots. Source-native GGML K-quants use one `u8` block
component shaped as `[rows, blocksPerRow, bytesPerBlock]`; scales, minima, and
codes remain fields inside each canonical GGML block. Formats whose persisted
representation stores data, scales, or metadata separately use separate slots.
In both cases the descriptor passed to `QuantizedLinear` explicitly identifies
the component tensors and their internal schema; it is never a hidden runtime
resource.

`Model.Model` exposes the validated schema as
`model.parameters: PhysicalParameterSchema`. Existing consumers of
`model.names` remain valid because `names` is the ordered projection of
`parameters.slots`. `Model.validateParams` checks arity, per-slot shape and
physical dtype. `Model.define` validates static cross-component group geometry,
and `Tensor.quantizedWeight` revalidates each actual component against that
group descriptor when the forward graph is built. `Model.execute`,
`Model.inference`, `Model.load`, and artifact hydration use
`Model.validateParams`; `Model.load` releases all newly loaded handles if final
validation fails. Authoritative runtime invocation still validates handle
ownership, placement, dtype, shape, and layout. `Model.forward` retains its
ordinary lazy graph role and cannot bypass per-operation graph validation.

Existing dense constructors populate one `Initializer` per slot and preserve
their current initialization equations. Imported pretrained parameterizations
mark every slot `ArtifactOnly` in this RFC, including ordinary F32 slots, because
the model must be loaded as one coherent checkpoint. `Model.init` preflights all
slot policies before running any `make` Effect and fails with `ModelError` when
any slot is `ArtifactOnly`. Consequently its public error type widens from
`TensorError` to `ModelError | TensorError`. It does not fill packed bytes
randomly, partially initialize the model, or silently substitute a dense graph.
A future format quantizer may supply valid `Initializer` Effects without
changing artifact hydration.

Without an explicit quantization manifest, existing `Model.save` and
`Model.load` reject schemas containing quantized groups. Dense `Model.load`
adopts final `Model.validateParams` checking as described above. This preserves
the non-goal of pretending ordinary safetensors encode the quantized descriptor.

### Artifact readers and catalogs

Hugging Face is not represented as an artifact format. Acquisition may produce:

- a repository snapshot with `config.json` and safetensors shards;
- one GGUF file;
- split GGUF files;
- tokenizer and processor assets;
- separate projector or speculative-model artifacts.

The caller selects local files before inspection. Format readers expose one
catalog shape without normalizing away source facts:

```ts
interface ArtifactByteSource {
  readonly id: string
  readonly size: bigint
}

type ArtifactTensorEncoding =
  | { readonly _tag: "Scalar"; readonly dtype: Runtime.DType }
  | {
      readonly _tag: "Quantized"
      readonly sourceType: number
      readonly encoding: QuantizedEncoding
    }
  | { readonly _tag: "Unsupported"; readonly sourceType: number }

interface ArtifactTensor {
  readonly name: string
  readonly source: ArtifactByteSource
  readonly byteOffset: bigint
  readonly byteLength: bigint
  readonly shape: ReadonlyArray<number>
  readonly encoding: ArtifactTensorEncoding
}

interface ArtifactAsset {
  readonly name: string
  readonly source: ArtifactByteSource
  readonly byteOffset: bigint
  readonly byteLength: bigint
  readonly mediaType?: string
}

interface ArtifactCatalog {
  readonly format: string
  readonly architectureCandidates: ReadonlyArray<ArchitectureKey>
  readonly metadata: ReadonlyMap<string, unknown>
  readonly tensors: ReadonlyMap<string, ArtifactTensor>
  readonly assets: ReadonlyMap<string, ArtifactAsset>
  readonly close: Effect.Effect<void>
}
```

`ArtifactByteSource` is an opaque reference-counted owner of one open local file
descriptor, not a path to reopen later. It records an immutable source ID,
overflow-checked size, and inspection-time file identity. Tensor offsets are
absolute offsets in that opened source; the GGUF reader converts each shard's
data-section-relative offset before publishing the catalog. The source is
restatted immediately before import and rejects size, device/inode, or
high-resolution modification-identity changes. Acquisition checksum verification
and an application policy preventing concurrent in-place mutation remain the
strong integrity boundary; path replacement cannot redirect an already opened
source.

The first public API is deliberately staged:

```ts
declare const PreparedModelTypeId: unique symbol
export interface PreparedModel {
  readonly [PreparedModelTypeId]: typeof PreparedModelTypeId
  readonly model: Model.Model
  readonly bindings: BindingPlan
  readonly close: Effect.Effect<void>
}

export interface HydratedModel {
  readonly model: Model.Model
  readonly params: ReadonlyArray<Tensor.Concrete>
  readonly close: Effect.Effect<void, Tensor.TensorError, Runtime.Runtime>
}

export declare const inspectGguf: (
  files: ReadonlyArray<string>
) => Effect.Effect<ArtifactCatalog, GgufError>

export declare const prepare: (
  artifact: ArtifactCatalog
) => Effect.Effect<PreparedModel, PrepareModelError, ModelRegistry>

export declare const hydrate: (
  prepared: PreparedModel
) => Effect.Effect<HydratedModel, HydrateModelError, Runtime.Runtime>

export declare const retain: (
  tensor: Tensor.Concrete
) => Effect.Effect<Tensor.Concrete, Tensor.TensorError, Runtime.Runtime>

export interface InferenceProgram {
  // Existing generation API remains.
  readonly close: Effect.Effect<void, Tensor.TensorError, Runtime.Runtime>
}
```

The ownership API requires corresponding runtime operations:

```ts
interface RuntimeService {
  // Existing operations remain.
  readonly retain: (
    tensor: ConcreteTensorHandle
  ) => Effect.Effect<ConcreteTensorHandle, BackendError>
  readonly releaseExecutable: (
    executable: ExecutableHandle
  ) => Effect.Effect<void, BackendError>
}

interface DecodeRuntime {
  // Existing operations remain.
  readonly releasePool: (
    pool: KvPoolHandle
  ) => Effect.Effect<void, BackendError>
}
```

`retain` creates an independent handle owner over the same storage and performs
no tensor-data copy. `releaseExecutable` and `releasePool` invalidate exactly
their passed owner and are explicit counterparts to native finalizers. Core
provides `Tensor.retain` and internal checked wrappers for program/pool release;
double release and cross-runtime handles fail through the typed backend channel.

`inspectGguf` parses and validates local files only. `prepare` resolves an
installed architecture, decodes canonical config, constructs the structural
model, and validates all source bindings; it does not require or inspect a
runtime. Core, not the architecture callback, validates the returned
`BindingPlanDraft`, attaches retained references to the original catalog's byte
sources, and constructs the opaque `PreparedModel`. `hydrate` is the sole phase
that obtains `Runtime.Runtime`. Before importing the first byte range, it checks
the complete plan for local-range import support, physical dtype/layout and
placement, source-encoding decoder admission for each declared linear/embedding
consumer class and compute dtype, and bounded synthesis allocation. Activation-
shape and complete graph-operation admission are impossible before tracing and
remain compiler-authoritative. Any hydration-preflight failure performs no
allocation, mapping, or range import.

The catalog interface is format-neutral so a later config-plus-safetensors
reader can produce it. Only `inspectGguf` and GGUF v3 are implemented by this
RFC. The generic interface is not a commitment to ship every possible reader in
the first milestone.

Catalog, prepared, hydrated, and inference values each retain exactly the owners
they need and expose idempotent `close` Effects; native finalization is a leak-
safety fallback. Preparing retains source references independently of the
catalog. Hydrated concrete tensors retain any no-copy mapping independently of
the prepared value. `Tensor.retain` creates another handle owner without copying
storage, and `Model.inference` uses it for concrete params so the hydrated value
may be closed after the inference program is built. `InferenceProgram.close`
atomically rejects new sessions/steps, waits for admitted invocations, invalidates
and releases live sessions, then releases programs, pools, and parameter
references. It is idempotent; finalization remains the fallback for abandonment.
Catalog inspection validates all offsets, lengths, alignments, integer
conversions, duplicate names, split indices, total tensor counts, and overlap
rules before returning.

Initial GGUF support includes:

- GGUF v3 header and typed metadata;
- little-endian scalar and array decoding;
- configurable alignment and overflow-safe tensor ranges;
- `split.no`, `split.count`, and `split.tensors.count` assembly;
- duplicate-name and inconsistent-shard rejection;
- `general.architecture` and architecture-specific metadata preservation;
- tokenizer metadata and chat-template preservation;
- F32 and GGML Q2_K, Q3_K, Q4_K, Q5_K, and Q6_K tensor encodings required by
  the pinned Muse artifact.

The reader may recognize additional GGML tensor type codes to produce precise
diagnostics, but recognition is not admission. An encoding is admitted only
when its block geometry is implemented by the artifact validator and the active
runtime reports a correct decoder path. New encodings extend an append-only
immutable codec table; they do not change architecture lookup or the catalog
ABI.

The reader does not convert Muse Q/K weights, centered norms, or synthesized
normalization tensors. The published GGUF conversion already performed those
architecture-specific transformations.

### Architecture preparation and parameter binding

The selected architecture implementation converts source metadata into one
canonical architecture config and constructs the same model code used for
direct construction. Dense and quantized parameterizations may select different
weight consumers, but they do not duplicate transformer-block semantics. It
returns an `ArchitecturePreparation` containing the model and a public draft;
only core can validate and brand a final `BindingPlan`.

The draft is a closed data algebra, not hydration callbacks:

```ts
export type BindingTransform =
  | { readonly _tag: "Direct" }
  | { readonly _tag: "BytePreservingReshape" }
  | { readonly _tag: "Transpose2D" }
  | { readonly _tag: "CanonicalGgmlBlocks"; readonly encoding: QuantizedEncoding }

export type BindingPlanDraftEntry =
  | {
      readonly _tag: "Source"
      readonly destination: string
      readonly sourceCandidates: ReadonlyArray<string>
      readonly transform: BindingTransform
    }
  | {
      readonly _tag: "Fill"
      readonly destination: string
      readonly value: number | ReadonlyArray<number>
    }

export interface BindingPlanDraft {
  readonly entries: ReadonlyArray<BindingPlanDraftEntry>
  readonly unexpected: "Reject" | "Ignore"
}

declare const BindingPlanTypeId: unique symbol
export interface BindingPlan {
  readonly [BindingPlanTypeId]: typeof BindingPlanTypeId
}
```

Core exports constructors for each draft form and freezes their inputs. The
first milestone permits only direct import, byte-preserving physical reshape,
ordinary two-dimensional transpose, canonical GGML-block interpretation, and
bounded numeric fill. A future source convention such as centered safetensors
may extend this closed algebra with a generic checked transform; arbitrary
architecture code never executes during hydration or backend lowering.

The validated plan contains, for every physical parameter component:

- destination parameter name and order;
- one resolved source tensor name or bounded fill rule;
- expected logical role and shape;
- expected physical encoding and geometry;
- orientation and source-layout interpretation;
- one closed transform descriptor;
- component relationship for one logical quantized weight;
- optionality policy;
- shared source ownership identity.

The architecture owns semantic mapping because only it knows whether a source
tensor is a Q projection, a centered normalization weight, an untied output
head, or an optional bias. The generic artifact reader cannot derive those
facts from names alone.

Core checks the draft against `model.parameters` and the complete catalog before
runtime preflight:

- every required source exists exactly once;
- every destination slot is bound exactly once and no unknown destination is
  present;
- every physical byte range is legal;
- source and expected dimensions agree after documented orientation rules;
- every encoding has known source block geometry;
- component blocks, scales, minima, and zero points agree geometrically;
- quantized component groups match the schema-declared descriptor;
- unexpected entries follow an explicit ignore/reject policy;
- no transform intended for original HF weights is repeated on converted GGUF
  tensors.

Validation returns an immutable prepared plan. Hydration does not reinterpret
model names, choose a different architecture, or alter logical bindings.

Tied logical weights are represented once in `PhysicalParameterSchema` and
referenced by multiple graph uses; they do not require two binding destinations
or duplicate `Model.Params` entries. Two genuinely distinct destinations may
share one source owner only when their complete physical schemas and closed
transforms are compatible.

For a direct GGML K-quant binding, preparation maps a logical GGUF tensor to one
physical `u8` block slot with shape `[rows, ne0 / blockElements, blockBytes]`.
It validates that `ne0` is divisible by the format block size and that the
source byte length equals the physical slot byte length. This is a view of the
canonical source encoding, not execution repacking. F32 tensors bind to ordinary
`f32` slots. Architecture-synthesized F32 vectors are declared as synthesized
bindings and allocated only after runtime preflight succeeds.

### Large tensor hydration

JavaScript must not copy a 12 GiB artifact through one `ArrayBuffer`.
`Runtime.RuntimeService` gains an optional typed local-range import extension
for local runtimes. It consumes a retained-source/range/physical-tensor request
and returns an ordinary concrete tensor handle owned by that runtime.

The extension is format-neutral. It knows:

- retained local byte-source token and immutable source identity;
- byte offset and length;
- physical tensor shape and dtype;
- destination placement and import policy;
- required byte order and alignment.

It does not know GGUF metadata keys, architecture names, tensor roles, or model
parameter names.

The runtime exposes two side-effect-free, architecture-neutral capability
queries. Hydration uses storage admission:

```ts
interface QuantizedStorageAdmissionRequest {
  readonly operation: "Linear" | "Embedding"
  readonly encoding: QuantizedEncoding
  readonly componentSchema: ReadonlyArray<QuantizedComponentSchema>
  readonly logicalShape: readonly [number, number]
  readonly computeDType: "f32" | "f16" | "bf16"
  readonly accumulationDType: "f32"
  readonly placement: Runtime.Placement
}
```

This shape-independent query succeeds only when the runtime has the mandatory
generic packed decoder path for that encoding/operation/dtype. The compiler
later extends the same facts with actual activation/output shapes, alignments,
compile options, and device features to obtain the admitted algorithm set.
Neither query compiles pipelines, allocates buffers, maps files, imports ranges,
or accepts architecture-defined strings.

Hydration performs these steps in order:

1. Validate that local-range import is available for every planned source.
2. Query every distinct quantized storage admission required by the complete
   plan's schema-declared linear/embedding groups.
3. Validate all ordinary physical dtypes, placements, synthesized bindings, and
   total byte counts.
4. If any check failed, aggregate the failures and return without imports.
5. Import direct ranges and construct synthesized tensors into temporary owned
   `Model.Params` storage.
6. Validate the completed params against `model.parameters` and publish the
   hydrated pair atomically.

Compiler lowering performs complete operation and shape admission for the actual
specialized graph. Hydration proves only that the parameter storage has a
generic consumer floor; compilation proves GQA, RoPE, normalization, activation
shape, workspace, and exact executable support. A disagreement caused by a
changed runtime or unsupported specialization is a typed compile error, never a
late execution fallback.

CPU may retain an mmap-backed immutable buffer or copy into owned storage.
Metal may retain safely mapped shared storage, create a no-copy shared buffer
where lifetime guarantees permit, or copy the exact range into runtime-owned
storage. In every case:

- the concrete tensor owns or retains its backing bytes;
- dropping the artifact catalog cannot invalidate a live parameter;
- imported bytes participate in external-memory accounting;
- cancellation and failure release partial imports;
- shared/tied source ranges are hydrated once and referenced consistently;
- the runtime never creates an unreported dense duplicate.

Remote runtimes may reject retained local-source imports or add a distinct
server-side artifact extension later. Local paths are not added to the baseline
tensor contract.

## Quantized Parameter Representation

### Typed bundle over physical tensors

A quantized logical weight is an immutable public bundle constructed by
model/layer code:

```ts
export type QuantizedComponent =
  | "Blocks"
  | "Codes"
  | "Scales"
  | "Minimums"
  | "ZeroPoints"
  | "Metadata"

export interface QuantizedComponentSchema {
  readonly role: QuantizedComponent
  readonly shape: ReadonlyArray<number>
  readonly dtype: Runtime.DType
}

declare const QuantizedEncodingTypeId: unique symbol
export interface QuantizedEncoding {
  readonly [QuantizedEncodingTypeId]: typeof QuantizedEncodingTypeId
  readonly id: QuantizedEncodingId
  readonly version: number
}

export type QuantizedEncodingId =
  | "ggml.q2_k"
  | "ggml.q3_k"
  | "ggml.q4_k"
  | "ggml.q5_k"
  | "ggml.q6_k"

export interface QuantizedEncodingSchema {
  readonly encoding: QuantizedEncoding
  readonly codec:
    | "GgmlQ2K"
    | "GgmlQ3K"
    | "GgmlQ4K"
    | "GgmlQ5K"
    | "GgmlQ6K"
  readonly blockElements: number
  readonly blockBytes: number
  readonly componentRoles: ReadonlyArray<QuantizedComponent>
  readonly legalComputeDTypes: ReadonlyArray<"f32" | "f16" | "bf16">
  readonly partialBlock: "Forbidden"
}

export declare const QuantizedEncodings: {
  readonly GgmlQ2K: QuantizedEncoding
  readonly GgmlQ3K: QuantizedEncoding
  readonly GgmlQ4K: QuantizedEncoding
  readonly GgmlQ5K: QuantizedEncoding
  readonly GgmlQ6K: QuantizedEncoding
}

export declare const quantizedEncodingSchema: (
  encoding: QuantizedEncoding
) => QuantizedEncodingSchema

export type QuantizedWeightOrientation = "LinearInOut" | "EmbeddingRows"

export interface QuantizedWeightDescriptor {
  readonly logicalShape: readonly [number, number]
  readonly computeDType: "f32" | "f16" | "bf16"
  readonly accumulationDType: "f32"
  readonly orientation: QuantizedWeightOrientation
  readonly encoding: QuantizedEncoding
  readonly componentSchema: ReadonlyArray<QuantizedComponentSchema>
}

export interface QuantizedWeight extends QuantizedWeightDescriptor {
  readonly components: ReadonlyArray<Tensor.Any>
}

export declare const quantizedWeight: (
  descriptor: QuantizedWeightDescriptor,
  components: ReadonlyArray<Tensor.Any>
) => Effect.Effect<QuantizedWeight, Tensor.TensorError>
```

`QuantizedWeight`, its append-only built-in encodings, and
`Tensor.quantizedWeight` are public because custom architecture Layers must be
able to construct the same semantic graph as official Layers. Normative
properties are:

- the descriptor is not `Tensor.Any`;
- components expose their actual physical shapes and dtypes;
- format metadata is immutable graph semantics;
- components are flattened into `Model.Params` with stable names;
- one validator proves the descriptor and components agree before node
  construction;
- generic tensor operations cannot consume the descriptor accidentally.

`QuantizedEncoding` is nominal and can be obtained only from exported canonical
constants. `(id, version)` resolves through one immutable codec table shared by
graph validation, host reference decoding, compiler capability checks, CPU, and
Metal. Model registration cannot add codecs or construct a new ID. Adding a
complex codec is a reusable core/compiler/backend feature; a future generic
affine encoding constructor may accept a fully declarative validated schema.

The canonical codec table records every decoding fact that affects semantics or
kernel selection:

- stable encoding ID and schema version;
- logical block element count and encoded byte count;
- bit and nibble order;
- signedness, codebook, or affine zero-point interpretation;
- block, group, and quantization axis;
- scale/minimum/zero-point representation;
- row padding and legal tail rules;
- block-internal field layout and legal physical component schemas;
- legal activation and accumulation dtypes.

Simple affine families should be parameterized by bit width and group geometry
rather than multiplied into unrelated enum cases. Complex GGML block formats
use named, append-only encoding IDs. Serialized IDs are distinct from backend
algorithm enums, and runtime validation rejects an unknown ID/version before
lowering.

The first built-in source encodings are exact canonical GGML K-quant blocks:

| Encoding ID | Elements per block | Bytes per block | Physical component |
|---|---:|---:|---|
| `ggml.q2_k` v1 | 256 | 84 | one `u8` `Blocks` tensor |
| `ggml.q3_k` v1 | 256 | 110 | one `u8` `Blocks` tensor |
| `ggml.q4_k` v1 | 256 | 144 | one `u8` `Blocks` tensor |
| `ggml.q5_k` v1 | 256 | 176 | one `u8` `Blocks` tensor |
| `ggml.q6_k` v1 | 256 | 210 | one `u8` `Blocks` tensor |

For these encodings the block-internal f16 scales, minima, high bits, and codes
remain inside the byte-exact `Blocks` component and are described by the built-
in encoding schema. A two-dimensional GGUF tensor with dimensions `[ne0, ne1]`
has physical component shape `[ne1, ne0 / 256, blockBytes]`; `ne0` must be a
multiple of 256 and partial K-quant blocks are rejected. Other public encodings
may declare separate component tensors, but cannot reuse these IDs with
different bytes or decoding semantics.

Orientation defines the mathematical tensor exactly:

- `LinearInOut` has logical shape `[inFeatures, outFeatures]`. Physical GGML row
  `o` decodes `inFeatures` values and defines `W[i, o] = row(o)[i]`.
- `EmbeddingRows` has logical shape `[vocabulary, embeddingDim]`. Physical GGML
  row `token` decodes the logical embedding row `E[token, :]`.

Thus GGML `ne[0]` is the input/embedding-column dimension and `ne[1]` is the
output/vocabulary-row dimension. Source GGUF dimension order is resolved by the
binding transform; backends receive only one of these mathematical orientations.

### Semantic operations

Core and both runtimes add two public semantic operations:

```ts
export declare const quantizedLinear: (
  input: Tensor.Any,
  weight: QuantizedWeight,
  bias?: Tensor.Any
) => Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime>

export declare const quantizedEmbedding: (
  weight: QuantizedWeight,
  indexes: Tensor.Any
) => Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime>
```

Their mathematical definitions are:

```text
QuantizedLinear(x, q, bias) = Linear(x, Dequantize(q), bias)
QuantizedEmbedding(q, ids)  = Embedding(Dequantize(q), ids)
```

These equations define parity and tests. They do not require a dense
`Dequantize(q)` tensor to exist in a production lowered program.

For the first milestone, activation, optional bias, and result dtype of
`quantizedLinear` all equal `weight.computeDType`; its input's final dimension
equals `inFeatures`, and bias follows existing linear broadcasting. Accumulation
is f32 before one cast to the result dtype. `quantizedEmbedding` accepts existing
`u32`/`i64` index dtypes, returns
`[...indexShape, embeddingDim]` in `computeDType`, and also decodes/accumulates in
f32. Unsupported dtype combinations fail graph construction rather than promote
implicitly.

The native semantic descriptors contain only `QuantizedWeightDescriptor`,
component count, and bias presence. Child order is
`[input, ...components, bias?]` for linear and
`[indexes, ...components]` for embedding. Child count, descriptor bytes, and
every component shape/dtype are validated in TypeScript, N-API decode, native
graph construction, remapping, and runtime invocation.

The TypeScript-to-native wire form carries canonical encoding ID and version,
not the nominal symbol. Native decode resolves that pair through its immutable
codec table and rejects unknown or mismatched schemas before constructing a
node.

The operations expose every physical component through ordinary graph child
edges. `GraphIndex`, generated binding discovery, placeholder slots, ownership,
memory accounting, tied component identity, and invocation validation therefore
continue to work without a new resource-edge ABI.

The complete descriptor is part of semantic node structure and executable cache
identity. Component values are ordinary rebindable parameter values and do not
join structural cache keys.

An explicit standalone full-weight dequantize node is not the canonical model
representation. If optimization is disabled or one fusion is unavailable, such
a graph could require multi-gigabyte logical intermediates for Muse's embedding
or output matrix. Bounded fallback is part of quantized-operation lowering.

### Model parameterization

An architecture implementation uses one shared block construction and a
weight-consumer abstraction:

```text
dense binding       -> ordinary Linear / Embedding
quantized binding   -> QuantizedLinear / QuantizedEmbedding
```

The exact artifact format does not appear in attention, normalization, residual,
or FFN architecture code. A Dynamic 2.0 model may choose a different format for
every projection while using one model implementation.

Quantized model parameters remain ordinary component tensors in the flat
parameter array. Optimizers and gradients reject inference-only packed
parameterizations explicitly. This RFC does not widen `Model.Params` to native
resources or arbitrary objects.

### Pipeline order

The existing `Model.inference(model, params, config)` remains the generation
entry point. Artifact APIs produce its two existing arguments; they do not add a
second compiler or executable type. The complete order is:

1. `inspectGguf` validates local container structure and creates a retained
   backend-free catalog.
2. `prepare` resolves installed architecture code, decodes config, calls
   `Model.define`, creates a complete binding plan, and validates source facts.
3. `hydrate` obtains the runtime, performs side-effect-free import/storage
   preflight over the complete binding plan, imports physical slots, and calls
   `Model.validateParams`.
4. `Model.inference` validates its complete config and parameter metadata before
   retaining/materializing params, then traces prefill/decode graph signatures
   with ordinary parameter placeholders.
5. Decode specialization resolves per-node attention policy and GQA geometry,
   remaps every quantized component child, and creates the stateful semantic
   graph.
6. RFC 0021 builds exactly one `GraphIndex` over that specialized graph and
   visits every quantized component edge.
7. Optimization planning may rewrite only proven semantic equivalents; with
   `optimize: false`, quantized operations remain intact for generic lowering.
8. Backend lowering repeats capability admission for specialized shapes,
   selects an explicit algorithm, and emits static memory requirements.
9. Pipeline preparation for all selected instructions succeeds before prefill,
   decode, batched decode, or shared KV state is published.
10. Runtime execution follows the frozen program and never resolves architecture
    metadata or chooses a fallback dynamically.

The prefill and decode traces must produce identical physical parameter schemas
and compatible retention-group schemas. They may select different quantized
algorithms because activation shape differs. Any disagreement fails
`Model.inference` before an `InferenceProgram` is returned.

The API change is explicit:

```ts
export interface InferenceConfig {
  // Existing fields remain.
  readonly optimize?: boolean
}

export declare const compileDecodeProgram: (
  roots: ReadonlyArray<Tensor.Any>,
  state: Runtime.DecodeStateRequest,
  options?: { readonly optimize?: boolean }
) => Effect.Effect<Tensor.DecodeProgram, Tensor.TensorError, Runtime.Runtime>
```

Omission preserves the current optimized default. Parameters are dynamic input
bindings even though the inference program retains their concrete values;
`constantWeights: true` applies only to eligible captured concrete leaves, not
parameter placeholders, and grants no replacement-buffer ownership. With
`optimize: false`, decode specialization still runs because it defines stateful
semantics, while optional optimization plans and exact packed kernels are
disabled. Quantized lowerers consult the same compile option and emit generic
packed instructions directly.

## Compiler and Runtime Execution

### Format admission versus acceleration

Each runtime distinguishes:

```text
decodable(format, operation, compute dtype)
accelerated(format, operation, compute dtype, shape, device capabilities)
```

`decodable` is the support floor. A format is not accepted merely because the
artifact parser recognizes its numeric GGUF type. Every admitted format has:

- overflow-safe block geometry validation;
- a host reference decoder;
- a correctness CPU implementation;
- a generic Metal decoder path when Metal advertises it;
- embedding row decoding where the format may hold embeddings;
- numerical parity fixtures;
- malformed-input and invalid partial-block rejection tests.

Capabilities are preflight information. The compiler remains authoritative for
shape-specific algorithm selection because support can depend on activation
rows, matrix dimensions, alignment, compute dtype, and hardware features.

### Backend lowering choices

The semantic graph remains unchanged. Backend lowering selects one typed
instruction plan per quantized node:

1. **Exact packed kernel** - hand-tuned for the format, shape class, activation
   dtype, and hardware, consuming the canonical hydrated source encoding.
2. **Generic packed decoder kernel** - shared GEMV/GEMM skeleton parameterized
   by a format decoder.
3. **Bounded panel dequantization plus dense GEMM** - explicit global workspace
   and partial accumulation selected by the cost model for a large prefill
   shape.
4. **Full-layer dequantization plus dense GEMM** - permitted only when selected
   deliberately for a measured large-M workload and proven to fit the static
   memory plan.

Every admitted format/operation pair has candidate 2, including all legal shape
regimes. Candidates 3 and 4 are optional performance algorithms, not the reason
a format is considered supported. Candidate 4 is a bounded per-instruction
workspace choice; it is not persistent whole-model dense expansion.

Before authoritative lowering, the backend capability table reports only
algorithms whose pipeline family and device features are available. With
`optimize: true`, the planner chooses among that admitted set according to its
cost model; if candidate 1 is absent, candidate 2 remains available. With
`optimize: false`, lowering selects candidate 2 directly. It never rejects an
admitted format because candidate 1 was disabled, and it never rewrites the
semantic operation into a standalone full-weight dequantize graph.

After algorithm selection, one authoritative lowered program proceeds through
validation, memory planning, physical planning, and pipeline preparation. An
unexpected pipeline compilation failure fails the complete compile transaction;
it does not mutate the planned instruction to generic execution. All selected
pipelines and workspace proofs succeed before any executable is published, and
runtime execution never catches a failed kernel to change algorithms.

Algorithm choice is recorded in:

- the backend instruction kind and requirements;
- pipeline cache identity;
- static scratch and persistent-memory declarations;
- executable diagnostics.

Suggested diagnostic names are:

```text
quantized_linear.optimized_packed
quantized_linear.generic_packed
quantized_linear.panel_dequantized
quantized_linear.full_dequantized
quantized_embedding.generic_row
quantized_embedding.optimized_row
```

The current aggregate instruction-kind diagnostics are extended with structured
architecture-neutral records:

```ts
export interface QuantizedInstructionDiagnostic {
  readonly operation: "Linear" | "Embedding"
  readonly encoding: QuantizedEncodingId
  readonly encodingVersion: number
  readonly algorithm: "ExactPacked" | "GenericPacked" | "Panel" | "FullLayer"
  readonly count: number
  readonly scratchBytes: number
}

export interface DecodeRetentionGroupDiagnostic {
  readonly groupId: number
  readonly policy: CausalAttentionPolicy
  readonly layerSlots: number
  readonly keyValueHeads: number
  readonly keyHeadDim: number
  readonly valueHeadDim: number
  readonly storage: KvStorage
  readonly retainedPositions: number
  readonly tokenBudget: number
  readonly storageBytes: number
  readonly transactionBytes: number
}
```

`ExecutableDiagnostics` exposes both arrays in deterministic order. Records may
be aggregated only when every listed field is equal. They contain no model ID,
source tensor name, or architecture layer number.

### CPU correctness floor

For every admitted format, CPU implements:

- one exact block decoder into thread-local f32 values;
- quantized GEMV/GEMM loops that decode blocks inside the dot product;
- quantized embedding that decodes only requested rows;
- f32 accumulation initially;
- ordinary output casting according to the operation contract.

Scratch is bounded by threads times block/tile size, not logical weight size.
SIMD `vec_dot`, Accelerate integration, and format-specific packing are measured
follow-ups. The scalar/block reference path is retained as a correctness oracle
even after acceleration lands.

### Metal generic packed path

Metal has shared GEMV and GEMM skeletons with per-format decoder functions.

For decode and other small-M shapes:

```text
load packed block -> decode into registers -> accumulate dot product
```

For GEMM/prefill:

```text
load packed K tile -> decode into threadgroup memory -> simdgroup MMA
```

The complete logical weight is never written to global memory. Adding a new
format normally adds a decoder and template instantiation before adding any
hand-tuned kernel.

The prefill and decode executables compile independently, so the same weight may
use generic GEMV for decode and panel/full dequantization for a measured large-M
prefill shape. This is a compiler choice, not a different model.

When bounded panel fallback is selected, workspace capacity, panel width,
dispatch count, accumulation behavior, and synchronization are explicit in the
lowered instruction and memory plan. A stock dense GEMM is not called as if it
could consume a partial weight without output accumulation.

### No implicit CPU fallback

RFC 0017's one-authoritative-runtime rule remains in force. A Metal graph does
not execute one unsupported projection through the CPU runtime. Apple unified
memory does not remove command ordering, synchronization, ownership, or layout
costs, and such a path would make placement and performance implicit.

If a format has only a CPU decoder, Metal reports it as unsupported during
hydration preflight. To satisfy this RFC's portability target, every format
claimed by Metal must provide the generic Metal decoder path. Load-time dense
expansion may be an explicit policy later, but it is not the definition of Metal
support.

### Memory and ownership

The physical component representation gives the existing planner accurate
bytes:

- packed bytes use actual `u8` storage;
- scales/minima/zero points use actual component storage;
- outputs use ordinary compute tensor storage;
- panel or full-layer fallback scratch is a typed backend requirement.

`Model.inference` retains concrete params and binds placeholders on each call.
Quantized components follow the same path, so this RFC hydrates and executes the
canonical persisted encoding only. Value-dependent execution repacking is
deferred. A future proposal must define a shared prepared-weight owner, tied
weight identity, source/replacement memory overlap, failure atomicity, and cache
identity before adding a repacked candidate; `constantWeights` on placeholders
is not such a lifecycle.

## Autodiff and Transform Semantics

The first quantized operations are inference-only.

- Gradient requests that traverse `QuantizedLinear` or
  `QuantizedEmbedding` fail with a typed, explicit unsupported-operation error.
- Packed `u8` components are never differentiation targets.
- Floating scale components do not become trainable accidentally merely because
  their physical dtype is f32/f16/bf16.
- Vmap, checkpoint remapping, graph copying, and decode specialization preserve
  descriptors and all component child edges.
- Activation-only gradients may be designed later with a deliberate transpose
  kernel and stop-gradient policy.
- Quantization-aware training and dense-master-weight updates require a separate
  RFC.

## Reusable Transformer Semantics

Muse is implemented by composing the following generic contracts. None of the
contracts contains an architecture identifier.

### Grouped-query attention

`Tensor.scaledDotProductAttention` is extended from equal-head attention to
grouped-query attention. For canonical `[batch, heads, sequence, headDim]`
inputs it accepts:

```text
Q: [B, Hq, Q, D]
K: [B, Hkv, K, D]
V: [B, Hkv, K, Dv]
```

Validation requires `Hq >= Hkv` and `Hq % Hkv == 0`. Query head `h` consumes KV
head `floor(h / (Hq / Hkv))`. The mathematical result is equivalent to
repeat-interleaving K and V across query heads, but graph transforms, memory
planning, runtime bindings, CPU kernels, and Metal kernels retain only `Hkv`
heads. The result shape is `[B, Hq, Q, Dv]`. Materializing repeated KV tensors
is not a legal lowering.

The public operation options gain an optional static attention policy:

```ts
export type CausalAttentionPolicy =
  | { readonly _tag: "Full" }
  | { readonly _tag: "SlidingWindow"; readonly tokens: number }

export interface ScaledDotProductAttentionOptions {
  readonly scale?: number
  readonly causal?: boolean
  readonly policy?: CausalAttentionPolicy
}
```

The existing semantic node keeps child order `[query, key, value]` and adds the
resolved policy to its attributes and cache identity. Head counts and K/V
dimensions remain derivable from child shapes; no architecture or source-name
field is added.

`tokens` includes the current token. At absolute query position `p`, a sliding
window may attend positions `[max(0, p - tokens + 1), p]`, further intersected
with any batch padding validity. Existing call sites that omit policy remain
valid. During inference specialization, the existing
`InferenceConfig.attentionWindow` applies only to omitted policies; otherwise an
omitted policy resolves to `Full`. An explicit per-node policy always wins, so a
global compatibility option cannot turn Muse's global layers into local layers.
`policy` is legal only when `causal: true`; arbitrary mask-tensor support is not
introduced by this RFC.

CPU and Metal receive only generic head counts, strides, scale, mask policy, and
tensor bindings. They cannot infer GQA or a window from an architecture name.
The backward rule accumulates `dK` and `dV` contributions from every query head
mapped to one KV head directly into that KV head; autodiff may not materialize
repeated K/V values merely to reuse equal-head backward code.

### Per-retention-group decode state

The caller supplies capacity and executable-shape options only. Decode
specialization derives attention geometry and publishes separate pool and
executable schemas:

```ts
export type KvStorage =
  | { readonly _tag: "Dense"; readonly dtype: "f32" | "f16" | "bf16" }
  | {
      readonly _tag: "SymmetricInt8"
      readonly dataDtype: "u8"
      readonly scaleDtype: "f32"
    }

export interface DecodeCompileOptions {
  readonly maxTokens: number
  readonly blockSize: number
  readonly prefillChunk: number
  readonly compiledBatch: number
  readonly maxLiveSequences: number
  readonly kvStorage: KvStorage
}

export interface DecodeRetentionGroupSchema {
  readonly id: number
  readonly layerSlots: number
  readonly keyValueHeads: number
  readonly keyHeadDim: number
  readonly valueHeadDim: number
  readonly blockSize: number
  readonly tokenBudget: number
  readonly retainedPositions: number
  readonly storage: KvStorage
  readonly policy: CausalAttentionPolicy
}

export interface DecodePoolSchema {
  readonly attentionGroups: ReadonlyArray<DecodeRetentionGroupSchema>
  readonly fingerprint: string
  // Existing KDA and short-convolution pool schemas remain here.
}

export interface DecodeAttentionBinding {
  readonly attentionOrdinal: number
  readonly groupId: number
  readonly layerSlot: number
}

export interface DecodeExecutableStateSchema {
  readonly compiledBatch: number
  readonly attention: ReadonlyArray<DecodeAttentionBinding>
  // Existing cursor, KDA, and short-convolution executable mappings remain here.
}
```

The public `Runtime.DecodeStateRequest` becomes these compile options; it does
not contain graph node IDs, head geometry, or architecture-provided layer data.
During specialization the compiler visits attention nodes in deterministic
semantic order, resolves omitted policies, and assigns zero-based attention
ordinals. Raw graph node IDs remain executable-local and never enter pool
compatibility or cache identity.

`Model.inference` maps `InferenceConfig.decodeBatch` to
`maxLiveSequences`, supplies `compiledBatch = 1` for prefill and single decode
and `compiledBatch = decodeBatch` for batched decode, and maps existing
`kvDtype: "int8"` to `SymmetricInt8`. Other `kvDtype` values map to `Dense`.
All attention groups use that one configured storage descriptor in this RFC;
per-node or per-group KV dtype selection is deferred. The descriptor remains on
each group so its data/scale page geometry and bytes are explicit.

`DecodeProgram` exposes one `DecodePoolSchema` and one
`DecodeExecutableStateSchema` instead of global `layers`, `kvHeads`, `headDim`,
and window scalars. Prefill, single-decode, and batched-decode have different
`compiledBatch` values and executable mappings but must have the same pool
fingerprint. `Model.inference` compiles all programs, compares that fingerprint,
and only then calls `makeKvPool(poolSchema)`. Existing KDA and short-convolution
state remains in the corresponding shared-pool and executable-local portions.

The fingerprint includes the ordered normalized state topology and every
attention-ordinal-to-group/layer-slot assignment, with batch/sequence extents
normalized out; it is not merely a tuple of group counts. A shape-dependent
forward that inserts, removes, or reorders stateful nodes therefore fails before
pool creation even if aggregate geometries happen to match.

The decoder assigns attention ordinals to groups by validated KV head count, K/V
dimensions, and retention policy, independent of
architecture or source layer name. Each group owns separate K and V page
storage. Dense pages have shapes:

```text
K: [layerSlots, blockSize, keyValueHeads, keyHeadDim]
V: [layerSlots, blockSize, keyValueHeads, valueHeadDim]
```

`SymmetricInt8` groups additionally own f32 K/V scale pages shaped
`[layerSlots, blockSize, keyValueHeads]`. All data and scale bytes appear in the
pool schema's static diagnostics.

Each live sequence has one block table per retention group. `maxTokens` is the
logical token-row budget shared by live sequences and prefix-cache entries in
each full group, not an aggregate byte budget split among groups. Every token
requires rows in every full group, just as one current pool row stores every
layer. Full groups therefore use `tokenBudget = maxTokens` and retain every live
absolute position.
Sliding groups retain at most `tokens` positions per sequence and receive a
separate statically computed `tokenBudget <= maxTokens`, based on the window,
page frontier, prefill chunk, and `maxLiveSequences`. Compilation fails if the
configured budget cannot support one legal prefill/decode transaction. Pages
reclaimed from a sliding group are reusable only within that group; they never
alias a full group's pages. A backend may implement the same schema as a dense
ring when the static plan proves equivalent, but ownership and accounting remain
per group.

The executable schema maps every attention ordinal to exactly one group and
layer slot. Runtime validation checks each state read/write against that mapping.
Prefill writes all keys and values needed by that prefill computation, then
retains only the policy-required suffix. Decode appends to full groups and
reclaims expired sliding-group blocks. Absolute token positions, not page or
ring indices, drive causal masking and RoPE. Prefix-cache identity includes the
retention-group schema, and a cached prefix shares corresponding blocks in every
group or none of them.

Muse produces two groups for the pinned configuration: 39 local layers with
`retainedPositions = 2048` and 13 global layers with full retention. Both store
two KV heads with key/value dimension 128. For the initial batch-one milestone
at an 8K configured token budget, local pages are bounded by the 2048-token
window plus the statically accounted page/prefill frontier instead of retaining
8K entries for 39 layers. The state never expands two KV heads to 32. Models
whose layers all resolve to one geometry and policy naturally produce one group.

The cached K value is the semantic key after Q/K normalization and any RoPE.
The cached V value is the projected value. Q is never cached.

### Rotary pair layout

The rotary operation gains an explicit layout:

```ts
export type RotaryLayout = "HalfSplit" | "InterleavedPairs"

export interface RotaryEmbeddingOptions {
  readonly layout?: RotaryLayout
  readonly rotaryDimension?: number
}

export declare const rotaryEmbedding: (
  input: Tensor.Any,
  sequenceLength: number,
  theta: number,
  options?: RotaryEmbeddingOptions
) => Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime>
```

For a pair `(a, b)` and angle `t`, both layouts compute:

```text
(a', b') = (a * cos(t) - b * sin(t), b * cos(t) + a * sin(t))
```

`HalfSplit` pairs dimensions `(i, i + rotaryDimension / 2)`, preserving the
existing operation. `InterleavedPairs` pairs `(2i, 2i + 1)`, matching GGML
RoPE-ready projection layout. Both require an even `rotaryDimension` not larger
than `headDim`; dimensions after it pass through unchanged. Layout is part of
node equality, graph hashing, backend instructions, and executable cache keys.

Existing callers that omit the new option retain `HalfSplit`; architecture code
must pass a layout explicitly. Muse's published GGUF Q/K rows were already
permuted at conversion time for GGML and therefore use `InterleavedPairs`
without a load-time weight transform. Local layers use theta 500000 and global
layers omit the rotary node entirely.

### RMS normalization

The reusable RMS normalization operation supports an optional effective scale
tensor and explicit epsilon:

```ts
export declare const rmsNorm: (
  input: Tensor.Any,
  scale: Tensor.Any | undefined,
  epsilon: number
) => Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime>
```

```text
rms(x, epsilon) = x * rsqrt(mean(lastDimension, x^2) + epsilon)
RmsNorm(x, none, epsilon) = rms(x, epsilon)
RmsNorm(x, scale, epsilon) = rms(x, epsilon) * scale
```

Reduction and scale multiplication use f32 semantic precision before casting to
the input compute dtype. Scale broadcasting over a final head dimension is
ordinary shape semantics, so the same operation handles hidden-wide and
per-head Q/K normalization.

The semantic node has child order `[input, scale?]` and attributes
`{ epsilon, hasScale, precision: "f32" }`. Epsilon must be finite and positive;
scale must match the final dimension and input dtype/placement. Its ordinary
autodiff and vmap rules are architecture-independent.

Some source models persist a centered weight `w` whose effective scale is
`1 + w`. Source binding owns that representation transform. The Muse GGUF
converter has already added one to centered layer-norm weights, so GGUF binding
imports them directly as effective scales and the backend executes ordinary
`RmsNorm`; it does not know the centered convention.

## Muse-Glimmer First Target

### Artifact

The first required file is:

```text
repository: unsloth/Muse-Glimmer-30B-GGUF
revision:   faa5b025c584459c13febfa5c59883516710ae39
file:       Muse-Glimmer-30B-UD-Q2_K_XL.gguf
bytes:      12,444,212,256
sha256:     3d63a1daff23fdc2a6927316151e855cacffe89b5cb9b9397a5aec0c412ec08d
```

It is one unsharded GGUF v3 text-model file containing 731 tensors. Vision and
speculative companions are separate artifacts and are deferred. Acquisition
must verify both byte size and SHA-256 before this pinned target is considered
present.

The file is not uniformly Q2_K. It contains F32 plus Q2_K, Q3_K, Q4_K,
Q5_K, and Q6_K. All must pass parser, decoder, binding, CPU, and Metal fallback
coverage before the complete model can execute.

### Text architecture requirements

The installed Muse architecture Layer resolves:

```text
hf.architecture:MuseGlimmerForConditionalGeneration
hf.model_type:muse_glimmer
gguf.architecture:muse-glimmer
```

Its text model requires:

- 52 dense transformer layers;
- hidden size 6656 and FFN size 19968;
- vocabulary size 202048;
- 32 query heads, 2 KV heads, and head dimension 128;
- 16:1 grouped-query attention without repeated KV storage;
- repeating local/local/local/global attention;
- local window 2048;
- RoPE theta 500000 on local layers and NoPE on global layers;
- scaleless embedding RMS normalization with epsilon `1e-5`;
- effective-scale pre-attention and pre-FFN RMS normalization with epsilon
  `1e-5`;
- effective-scale post-attention and post-FFN RMS normalization with epsilon
  `1e-8`;
- per-head Q/K RMS normalization over dimension 128 with epsilon `1e-5`, Q
  scale 3.87, and K scale one;
- sigmoid-gated attention output;
- dense SwiGLU FFNs;
- final effective-scale RMS normalization with epsilon `1e-5`;
- output multiplier `0.19611613513818404`;
- final logit transform `20 * tanh(logits * multiplier / 20)`;
- untied token embedding and output head.

The Muse Layer decodes those values into a private canonical config and rejects
the artifact before `Model.define` if any invariant fails: 52 layer policies,
`6656 -> {4096, 256, 256, 4096}` Q/K/V/gate projection widths, 32 divisible by
2, head dimension 128, all hidden/FFN matrix dimensions positive, vocabulary
202048, window 2048, context 131072, and a local/local/local/global policy for
indices `0..51`. The two norm epsilons, Q scale, output multiplier, and softcap
are architecture semantics decoded from metadata where represented and checked
against the installed Muse contract where GGUF omits them. This validation lives
in the Muse Layer and does not add a runtime capability or backend branch.

These become reusable semantic/model capabilities rather than Muse-only backend
code. In particular, decode state must support per-layer local/full retention.
Repeating two KV heads into 32 is not an acceptable fallback because it expands
Muse's KV storage by 16 times.

For hidden state `h`, one text layer is constructed in this exact order:

```text
a = RmsNorm(h, attn_norm_scale, 1e-5)
q = transpose(reshape(attn_q(a), [B, S, 32, 128]), sequence, heads)
k = transpose(reshape(attn_k(a), [B, S,  2, 128]), sequence, heads)
v = transpose(reshape(attn_v(a), [B, S,  2, 128]), sequence, heads)
g = reshape(attn_gate(a), [B, S, 4096])
q = RmsNorm(q, q_norm_scale_3_87, 1e-5)
k = RmsNorm(k, k_norm_scale_1, 1e-5)
(q, k) = local ? Rotary(q, k, theta=500000, layout=InterleavedPairs) : (q, k)
a = GQA(q, k, v, scale=1/sqrt(128), policy=local ? Window(2048) : Full)
a = reshape(a, [B, S, 4096]) * sigmoid(g)
a = attn_output(a)
h = h + RmsNorm(a, attn_post_norm_scale, 1e-8)
f = RmsNorm(h, ffn_norm_scale, 1e-5)
f = ffn_down(silu(ffn_gate(f)) * ffn_up(f))
h = h + RmsNorm(f, ffn_post_norm_scale, 1e-8)
```

The gate projection consumes `a`, the pre-attention normalized hidden state,
and gates the reshaped GQA output before the attention output projection. This
ordering is architecture code expressed with generic graph nodes; it is not a
fused Muse backend instruction.

Token lookup is followed by scaleless `RmsNorm(..., 1e-5)`. After all 52 layers,
the final effective-scale RMS norm feeds the untied output projection. The raw
output projection is multiplied by `0.19611613513818404`, divided by 20, passed
through tanh, and multiplied by 20.

### GGUF binding rules

The architecture binds three global tensors and exactly fourteen tensors per
layer:

```text
token_embd.weight
output_norm.weight
output.weight
blk.N.attn_norm.weight
blk.N.attn_post_norm.weight
blk.N.attn_q.weight
blk.N.attn_k.weight
blk.N.attn_v.weight
blk.N.attn_q_norm.weight
blk.N.attn_k_norm.weight
blk.N.attn_gate.weight
blk.N.attn_output.weight
blk.N.ffn_norm.weight
blk.N.ffn_post_norm.weight
blk.N.ffn_gate.weight
blk.N.ffn_up.weight
blk.N.ffn_down.weight
```

The binding plan honors GGML dimension order and quantization along `ne[0]`.
It does not repeat conversion-time operations. The artifact already contains:

- Q/K rows unpermuted into GGML's interleaved RoPE layout;
- one added to centered layer-normalization weights;
- synthesized Q-normalization vectors containing 3.87;
- synthesized K-normalization vectors containing one.

Those Q/K normalization vectors are source tensors in this GGUF, not hydration-
time synthesis. The binding plan imports all 731 entries and rejects extras for
the pinned text-only artifact. Future artifact adapters may synthesize equivalent
effective-scale vectors only when their source convention and architecture
config make that transform explicit.

Logit/token parity tests compare against checked-in golden fixtures produced
from the pinned artifact. External runtimes may generate development fixtures,
but they are not build, test, or runtime dependencies.

### Initial deployment scope

The first end-to-end milestone is deliberately:

- text-only;
- batch-one prefill and decode as the Phase 4 gate, with existing fixed-width
  decode batching exercised during Phase 5 parity;
- model maximum context 131072 preserved in config, with initial
  `InferenceConfig.maxTokens` set to 4096-8192 before full-capacity optimization;
- greedy output parity before probabilistic sampling integration;
- CPU correctness and Metal execution;
- no mmproj;
- no DFlash;
- no OpenAI server.

## Errors and Failure Atomicity

Failures are typed by phase:

- artifact I/O, malformed header, invalid metadata, split inconsistency, range
  overflow, duplicate tensors;
- no registered architecture, conflicting registration, ambiguous aliases;
- unsupported config value or missing model capability;
- missing/extra tensor policy, shape mismatch, orientation mismatch, malformed
  quant block geometry;
- format recognized by the reader but not decodable by the selected runtime;
- hydration I/O, mapping, allocation, placement, or cancellation failure;
- quantized compilation, pipeline preparation, or memory-plan failure;
- ordinary inference capacity and execution failures.

The public aliases preserve phase information rather than collapsing everything
to one message-only loader error:

```ts
export type PrepareModelError =
  | ModelRegistryError
  | ArchitecturePreparationError
  | BindingPlanError
  | Model.ModelError

export type HydrateModelError =
  | BindingPlanError
  | RuntimeCapabilityError
  | RuntimeImportError
  | Model.ModelError
  | Tensor.TensorError
```

`GgufError` separately covers malformed container/source failures. Every variant
contains its phase, source or destination name where applicable, expected and
actual geometry, and an underlying runtime diagnostic without requiring callers
to parse strings.

Hydration is failure-atomic. On failure, every concrete component created by
that attempt is released. The returned model/params pair is published only when
all bindings have succeeded. Source catalogs remain reusable after a failed
attempt unless the underlying file changed, in which case identity validation
fails.

Closing catalog, prepared, hydrated, or inference owners is idempotent. Use
after close fails with a typed lifecycle error. `Model.inference` acquires all
parameter retains before publishing its program and releases partial retains and
executables if any trace, schema comparison, pipeline, or pool allocation fails.

## Security and Reproducibility

- Artifacts are data, never executable repository code.
- Architecture support comes only from explicitly imported Layers.
- Acquisition records repository, resolved revision, filenames, sizes, and
  available checksums outside the runtime loading phase; the pinned Muse target
  requires its declared SHA-256.
- Relative split paths cannot escape the selected artifact directory.
- Every byte-range arithmetic operation is overflow checked.
- Tensor-count and metadata-array limits prevent unbounded parser allocation.
- Format IDs and block schemas are versioned and append-only for persisted
  compatibility.
- Executable diagnostics record exact storage format and selected algorithm.
- Environment switches that affect quantized algorithm selection join the
  existing compile-options snapshot discipline from RFC 0021.

## Alternatives Considered

### Put Hugging Face IDs directly on `Model.load`

Rejected. Repository resolution, authentication, caching, revisions, file
selection, and execution are separate failure and ownership domains. Local
loading must be deterministic and offline-capable.

### Treat GGUF as executable model serialization

Rejected. GGUF contains metadata and tensors, not general executable semantics.
`general.architecture` dispatches installed model code.

### Global mutable architecture registry

Rejected. It makes import order, test isolation, and supported-model sets
implicit. Effect Layers provide an explicit fresh registry per application
graph.

### One registry Layer plus externally ordered registration effects

Rejected as the primary ergonomics. Architecture packages expose Layers that
locally provide and re-emit the memoized registry, allowing the supported set to
be written directly as `Layer.mergeAll(Muse, Llama, Qwen)`.

### Add Q2_K/Q4_K as `Runtime.DType`

Rejected. Block formats are not scalar arithmetic types and have no fixed byte
size per logical element.

### Make a packed weight look like an ordinary f32 tensor layout

Rejected. Current layout, binding, readback, generic operation, and memory
contracts would become dishonest or require a framework-wide logical/physical
tensor redesign.

### Add opaque quantized-weight resource handles to graph invocation

Rejected as the canonical representation. It would add resource edges,
invocation slots, ownership rules, cache signatures, serialization, and planner
semantics parallel to ordinary tensor dependencies. Immutable packed components
already fit tensor ownership correctly.

### Store unrelated packed/scales tensors in Params without a typed descriptor

Rejected. Component arrays alone do not prove format identity, logical shape,
block geometry, or relationship. A validated typed bundle and dedicated
consumer are required.

### Make ordinary Linear inspect quantized Params automatically

Rejected for the initial design. `Tensor.Any` has no honest representation for
the multi-buffer block storage, and existing Linear assumes ordinary compatible
compute tensors. Architecture code can keep one block implementation while its
weight-consumer abstraction selects dense or quantized semantic operations.

### Serialize `Dequantize(weight) -> Linear`

Rejected as the canonical semantic form. It gives a simple reference equation
but can force a full logical weight intermediate when fusion is unavailable.
`QuantizedLinear` retains the same semantics while requiring lowering to plan a
bounded implementation.

### Dequantize the complete model at load time

Rejected as a default or correctness contract. It defeats the memory purpose of
the first target and may make a physically loadable artifact impossible to
execute.

### Fall back from Metal to CPU per operation

Rejected. It violates RFC 0017's explicit placement and no-implicit-transfer
rules. Generic Metal decoding is the Metal correctness floor.

### Require hand-tuned kernels for every format

Rejected. Decoder-based shared kernels separate broad format correctness from
incremental performance work.

### Wrap llama.cpp

Rejected. The project exists to implement the model, graph, compiler, storage,
and backend primitives directly. External runtimes may be offline parity
oracles only.

## Implementation Plan

### Phase 0: model, registry, and inspection contracts

- Add `PhysicalParameterSchema`, `Model.define`, schema-derived `model.names`,
  `Model.validateParams`, schema composition through existing combinators, and
  `ArtifactOnly` initialization with allocation-free failure tests.
- Add the Effect `ModelRegistry` service, stable live Layer, atomic
  registration, scoped rollback, exact resolution, and public `registerModel`
  helper with a read-only consumer service.
- Prove concurrent synthetic/custom architecture Layers compose into one
  populated registry and failed/interrupted merges leave no registrations.
- Add backend-free catalog, retained open-byte-source identity, ordered
  architecture candidates, closed binding-draft constructors, core plan
  validation/branding, and deterministic owner `close` contracts.
- Add nominal canonical K-quant encoding IDs and physical block schemas needed
  for overflow-safe artifact range validation, without execution admission.
- Create `@effect-torch/artifact-gguf` and `@effect-torch/models` with enforced
  dependency direction.
- Add the GGUF v3 reader and validate the pinned file's 731-entry catalog,
  absolute ranges, metadata, and tensor type distribution. Keep
  revision/size/SHA-256 in a separate acquisition manifest; catalog inspection
  does not hash-read 12 GiB.
- Add side-effect-free storage-admission and local retained-range import
  extension contracts plus tensor-retain/executable-release/pool-release
  ownership contracts, without importing a model.

Exit criterion: local inspection, architecture resolution, structural model
construction, and complete binding validation run without `Runtime.Runtime`.

### Phase 1: reusable transformer semantics

- Extend scaled dot-product attention to true GQA with unequal query/KV head
  counts, distinct K/V dimensions, direct grouped backward, and no repeated KV
  tensor.
- Replace global decode geometry with compiler-derived attention ordinals,
  normalized shared-pool schemas, executable-local mappings, and independently
  reclaimable full/sliding retention groups.
- Add dense and symmetric-int8 KV storage descriptors, static full/sliding
  attention policy, and absolute-position paged/ring semantics.
- Add `HalfSplit` and `InterleavedPairs` rotary layouts.
- Add optional-scale RMS normalization with explicit epsilon and f32 semantic
  precision.
- Compose biasless projections, sigmoid gating, SwiGLU, residual ordering,
  output scaling, and tanh softcap from generic nodes.
- Test all features on small dense synthetic models on CPU and Metal before
  adding Muse architecture code.

Exit criterion: a custom test architecture can express Muse's layer equation
and mixed retention pattern without an architecture-specific backend symbol.

### Phase 2: quantized semantic foundation

- Complete immutable codec decoding semantics for the Phase 0 source schemas.
- Add typed quantized-weight component validation and physical parameter slots.
- Add `QuantizedLinear` and `QuantizedEmbedding` semantic nodes.
- Update graph children/remapping, adapters, N-API, cache normalization,
  autodiff rejection, vmap/checkpoint handling, and diagnostics.
- Add host reference decoders and malformed-block fixtures for Q2_K, Q3_K,
  Q4_K, Q5_K, and Q6_K before admitting them in any runtime.

Exit criterion: descriptors and reference dequantization are complete for every
encoding present in the target, independent of model architecture.

### Phase 3: generic correctness execution

- Add CPU generic block-decoder linear, GEMM, and selected-row embedding
  implementations for every admitted target encoding.
- Add Metal generic decoder GEMV, GEMM/prefill, and selected-row embedding
  skeletons for the same encodings.
- Add hydration preflight that rejects a complete unsupported plan before the
  first allocation, file mapping, or range import, limited to import/storage and
  generic linear/embedding admission facts available before tracing.
- Implement no-copy tensor retain and explicit executable/pool release in CPU
  and Metal runtimes with finalizers as fallback.
- Add compiler capability checks, static panel/full scratch accounting,
  algorithm diagnostics, and executable publication atomicity.
- Prove correct execution with exact optimized kernels disabled and with
  `optimize: false`.

Exit criterion: mixed-format synthetic graphs execute on CPU and Metal through
generic packed paths without full logical weight allocations or cross-runtime
fallback.

### Phase 4: Muse architecture and hydration

- Implement Muse config decoding, the exact 52-layer text graph, and all 731
  source bindings in `@effect-torch/models`, not a backend package or Rust
  architecture module.
- Bind canonical GGML blocks, effective centered-norm scales, converted
  interleaved Q/K rows, and source Q/K norm vectors without repeating
  conversion transforms.
- Hydrate every text weight without a JavaScript full-file copy.
- Verify physical schema conformance, mixed formats, source ownership,
  capability rejection, cancellation, and cleanup behavior.
- Assert generic policy grouping derives 39 local slots with
  `retainedPositions = 2048` and 13 full slots, both with two KV heads.

Exit criterion: the pinned artifact produces ordinary `Model.Model` and
`Model.Params`, and both prefill and decode executables publish successfully on
CPU and Metal.

### Phase 5: parity and measured optimization

- Run fixed token-input/logit fixtures on CPU and Metal.
- Run greedy generation parity through `Model.inference`.
- Exercise existing fixed-width batched decode against batch-one outputs.
- Add deterministic `close` behavior for catalog, prepared, hydrated, and
  inference owners and verify no-copy parameter retains.
- Measure generic packed decode/prefill and add direct exact packed kernels only
  where profiles justify them.
- Record resident bytes, scratch, algorithm choices, tokens/s, prefill
  throughput, and local/global KV bytes.
- Extend tokenizer/chat integration after token-level parity is stable.

Execution repacking remains outside these phases and requires the separate
prepared-weight lifecycle described above.

## Testing

### Model definition

- Duplicate physical names, parameter arity, dtype, shape, and quantized-group
  mismatches.
- `model.names` is exactly the ordered unique physical-slot projection.
- Tied logical uses share one slot and one hydrated concrete tensor.
- `Model.validateParams` rejects missing, extra, or incompatible values, and
  `Model.load` releases completed loads when final validation fails.
- `Model.init` rejects an `ArtifactOnly` parameterization before allocating any
  dense or packed component and leaves dense initialization unchanged.

### Registry

- One, many, and zero architecture Layers.
- Shared registry identity under `Layer.mergeAll`.
- Fresh registry across independent provisions.
- Atomic duplicate-alias rejection.
- Ordered reader-provided architecture-candidate resolution.
- No global state leakage between tests.
- Application-defined architecture Layer composed with an official Layer.
- An out-of-tree-style fixture builds a model and binding draft using only
  published `@effect-torch/core` exports.
- Concurrent collision, interrupted construction, failed-merge rollback, retry,
  duplicate ID/key, and attempted post-construction mutation.
- No architecture ID or source tensor name in compiler/backend instructions or
  executable diagnostics.

### Artifact reader

- GGUF v3 scalar and array metadata fixtures.
- Alignment, offset, byte-length, and overflow failures.
- Split ordering, missing shards, duplicate names, and total-count mismatch.
- Unknown metadata preservation.
- File ownership after catalog creation and during hydration.
- Shard-relative to absolute offset conversion, retained-descriptor behavior
  after path replacement, and source identity change rejection.
- Idempotent close, use-after-close failure, and independent catalog/prepared/
  hydrated/no-copy tensor retains.
- Fuzz/property tests for malformed headers and tensor descriptors.
- Pinned revision, 12,444,212,256-byte size, SHA-256, and 731-tensor target
  manifest verified offline as acquisition-fixture setup, separately from
  metadata-only catalog inspection.
- Range-import instrumentation proves no full-file JavaScript `ArrayBuffer` and
  enforces a bounded peak JavaScript heap threshold during hydration.
- Unsupported runtime capability rejects before the range-import test double is
  called or any synthesized tensor is allocated.

### Quantized formats

- Golden decode vectors for every block format.
- Zero, extrema, scale/minimum, sign, and codebook cases; partial 256-element
  K-quant blocks are rejected.
- Logical orientation and GGML `ne[0]` quantization axis.
- Invalid component dtype/shape/block geometry.
- Stable format/cache identity across every semantic descriptor field.

### Operations

- Quantized linear versus explicitly dequantized f32 oracle.
- Quantized embedding versus selected dense rows.
- GEMV, GEMM, batch, and non-multiple activation/output-row shapes while the
  quantized input-column dimension remains block-aligned.
- Bias and output dtype behavior.
- Mixed formats in adjacent layers.
- Rebinding different values with identical descriptors.
- Explicit autodiff and unsupported-transform errors.

### Transformer semantics

- GQA head mapping for multiple ratios, including Muse's 32 query/2 KV heads,
  against an explicit repeat-interleave numerical oracle.
- Graph and memory plans contain two KV heads and no repeated-KV allocation.
- Full and sliding-window prefill/decode parity across window boundaries.
- Independent full/sliding retention, page reclamation or dense-ring equivalent,
  absolute positions, batch reset, and cancellation cleanup.
- Decode schemas with heterogeneous KV head counts, K/V dimensions, and window
  sizes across attention nodes under one configured KV storage policy.
- Distinct key/value dimensions and dense/symmetric-int8 data/scale page bytes.
- Prefill/single/batched executable-local mappings compare equal shared-pool
  fingerprints despite different graph node IDs and compiled batches.
- `HalfSplit` and `InterleavedPairs` RoPE against equation-level fixtures; node
  equality and cache keys differ by layout.
- Scaleless, effective-scale, hidden-wide, and per-head RMS normalization at
  `1e-5` and `1e-8`.
- Exact attention-gate placement, four-norm residual ordering, SwiGLU, output
  multiplier, and tanh softcap on a small dense reference model.

### Backends and fallback

- CPU scalar reference path always available for CPU-advertised formats.
- Metal generic path with optimized pipelines forcibly disabled.
- Pre-lowering capability omission of an exact pipeline selects generic packed
  lowering; an unexpected post-plan pipeline failure aborts compilation without
  mutating the plan.
- No CPU command or host readback in a Metal executable.
- No full logical weight allocation in generic packed diagnostics.
- Bounded panel workspace where selected.
- `optimize: true` and `false` numerical parity.
- Cancellation and partial-hydration cleanup.
- Hydration/compiler capability agreement and typed disagreement failure before
  executable publication.
- Exhaustive per-encoding/algorithm and per-retention-group diagnostic records.
- No compiler/backend package dependency on model or GGUF-reader packages, and
  no target-only architecture constants outside model/integration-test code.
- Two differently registered architectures with identical semantic descriptors
  produce identical normalized lowered programs and diagnostics.

### Muse integration

- Complete 731-tensor catalog and binding validation.
- Exact expected physical formats per bound projection.
- Fixed-prompt final logits within quantized reference tolerances.
- Greedy token-for-token generation parity.
- Correct two-KV-head pool geometry with 39 local slots at
  `retainedPositions = 2048`, a separately asserted local `tokenBudget`, and 13
  full slots.
- Correct local/global attention and interleaved local-layer RoPE/global-layer
  NoPE.
- Exact `1e-5` pre-norm, `1e-8` post-norm, Q scale 3.87, gate-before-output-
  projection, and final-logit transform ordering.
- Full model runs without persistent dense expansion.

## Performance Gates

Correctness lands before performance, but the implementation cannot declare the
deployment target complete without measuring:

- artifact inspection time and peak host memory;
- hydration time and duplicate resident bytes;
- generic and, where implemented, exact optimized prefill tokens/s;
- generic and, where implemented, exact optimized decode tokens/s;
- packed bytes and peak statically planned scratch;
- KV bytes under mixed local/global attention;
- compile time and quantized pipeline count;
- algorithm distribution across all model weights.

The generic packed path is allowed to be slow initially. It is not allowed to
materialize the complete dense model, hide cross-runtime execution, or produce
unbounded per-token temporary storage.

## Acceptance Criteria

1. `Model.define` validates a unique physical parameter schema and derives
   `model.names`; tied logical references never duplicate slots.
2. `Model.init` fails an imported quantized `ArtifactOnly` model with typed
   `ModelError` before allocation, while existing dense initialization remains
   green.
3. Merged synthetic, Muse, and application-defined architecture Layers provide
   one read-only registry containing all non-conflicting entries; duplicate
   aliases, sibling failure, or interruption roll back the complete Layer build.
4. A local artifact can be inspected, architecture-resolved, structurally
   constructed, and fully bound without `Runtime.Runtime` or device allocation.
5. Hydration preflights all import/storage and shape-independent generic
   linear/embedding admission facts available before tracing; failure performs
   zero imports and zero allocations, while complete operation/shape admission
   remains compiler-authoritative.
6. Any later hydration failure releases all components created by that attempt,
   and the model/params pair is published only after schema validation succeeds.
7. Offline acquisition-fixture setup verifies the pinned revision, filename,
   byte size, and SHA-256; independent GGUF inspection and core preparation
   produce the complete 731-tensor binding plan.
8. GGUF ranges are imported without reading the complete Q2_K_XL file into a
   JavaScript buffer.
9. Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, and F32 coexist in one hydrated model, and
   every quantized linear/embedding format matches reference dequantization on
   CPU and Metal.
10. Metal executes every admitted target format through a generic path when
    exact optimized kernels are disabled; `optimize: false` selects that path
    without semantic graph rewrites.
11. Generic packed execution allocates no full logical weight; any panel/full-
    layer algorithm is explicit, bounded, and present in the static memory plan.
12. Executable diagnostics identify storage encoding and selected generic or
    exact algorithm for every quantized instruction.
13. Core/compiler/backends have no dependency on model or GGUF-reader packages
    and contain no Muse/custom IDs, source names, metadata keys, target-only
    constants, layer-index semantic branches, or model-specific instructions.
    Backend decisions are functions only of generic descriptors, shapes, dtypes,
    placement, hardware, and compile options.
14. The pinned Muse artifact constructs an ordinary `Model.Model`, returns
    `Model.Params`, and generates through existing `Model.inference`.
15. Generic grouping derives two KV heads, 39 local slots with
    `retainedPositions = 2048`, and 13 full slots; Muse also preserves
    interleaved local RoPE, global NoPE, exact norm epsilons, gated attention,
    and logit softcap with fixed-logit and greedy-token parity.
16. No llama.cpp, vLLM, Transformers, MLX, Candle, or remote model code is a
    runtime, build, or test dependency.
17. Existing dense training and inference tests remain green with no model
    architecture Layer installed.
18. Catalog, prepared, hydrated, and inference owners close deterministically;
    no-copy retained parameters survive earlier-owner closure, and all failed
    construction paths release partial owners.
19. Differently named architecture Layers emitting identical semantic graphs
    produce identical normalized lowered programs and diagnostics.

## Risks

### Kernel breadth

Five K-quant decoders across CPU, Metal GEMV, Metal GEMM, and embedding are
substantial correctness work. Shared decoder contracts and golden block vectors
reduce duplication; format support lands one admitted codec at a time.

### Generic Metal performance

A decoder-pluggable kernel can be correct but far below a hand-tuned format
kernel. Diagnostics and benchmark gates keep generic fallback visible. The
first milestone values coverage; later optimization follows model-level
profiles.

### Model abstraction pressure

Quantized parameterizations use physical component names and inference-only
operations while `Model` currently assumes an initializer and generic tensor
params. This RFC deliberately changes `Model` through the public physical schema
and `ArtifactOnly` policy before artifact hydration lands. Implementation must
keep the flat parameter invariant without claiming packed components are
trainable or adding a second artifact-specific model abstraction.

### Mixed attention state

Muse's local/global pattern is not representable efficiently by the current
single-window KV pool. The per-retention-group schema is therefore a prerequisite
for Muse integration, not a post-parity optimization. It increases compiler,
runtime allocation, batch-reset, and cancellation state space and must first be
validated with small architecture-independent graphs.

### Artifact evolution

GGUF numeric type IDs and metadata conventions evolve. Persisted format IDs are
append-only; unknown encodings remain inspectable but not hydratable until their
decoder contract lands.

### Duplicate storage

Canonical packed bytes, file mappings, imported runtime buffers, and bounded
dequantization scratch can coexist accidentally. Hydration and lowering
diagnostics must account for every copy and preserve tied sharing. Execution
repacking is excluded specifically to avoid adding another persistent owner
without a defined lifecycle.

## Follow-up Work

- Device-side top-k/top-p/temperature sampling for the 202048-token vocabulary.
- Full embedded GGUF tokenizer construction and exact Muse chat-template
  rendering.
- Native optimized K-quant Metal and CPU kernels selected from profiles.
- An explicit shared prepared-weight lifecycle before any value-dependent
  compressed execution repacking.
- Explicit per-group KV storage policies if mixed KV dtypes prove useful.
- 131K mixed local/global KV capacity optimization.
- Muse vision encoder and `mmproj` artifact binding.
- DFlash speculative decoding.
- Llama, Qwen, and Gemma architecture Layers.
- Dense HF safetensors architecture loading through the same registry and
  binding pipeline.
- Explicit quantization/export of effect-torch-trained dense models.
- Model-program serialization in a separate RFC.

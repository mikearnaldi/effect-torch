# RFC 0022: Native Pretrained GGUF Inference

- **Status**: Draft
- **Created**: 2026-08-11
- **Depends on**: RFC 0005 (models), RFC 0010 (inference), RFC 0012
  (dtypes), RFC 0017 (multi-backend runtime), RFC 0019 (executable
  compilation), RFC 0020 (invocation ownership), RFC 0021 (compiler pipeline)

## Summary

Add zero-dependency, native inference for one pinned pretrained GGUF model while
preserving the existing execution contract:

```text
Model.Model + Model.Params -> Model.inference
```

The first target is text-only inference for:

```text
repository: unsloth/Muse-Glimmer-30B-GGUF
revision:   faa5b025c584459c13febfa5c59883516710ae39
file:       Muse-Glimmer-30B-UD-Q2_K_XL.gguf
bytes:      12,444,212,256
sha256:     3d63a1daff23fdc2a6927316151e855cacffe89b5cb9b9397a5aec0c412ec08d
```

The file is one unsharded GGUF v3 file with 731 tensors. It mixes F32, Q2_K,
Q3_K, Q4_K, Q5_K, and Q6_K.

`Gguf` is exported from `@effect-torch/core`, while `MuseGlimmer` is exported
from `@effect-torch/core/models`. The implementation is one direct vertical path:

```text
local GGUF path
  -> active CPU or Metal runtime inspects GGUF natively
  -> Gguf.load validates metadata and descriptors
  -> Registry.get("gguf:" + exact general.architecture)
  -> architecture.create(canonical config)
  -> name/shape bijection with model.parameterSpecs
  -> active runtime loads GGUF natively
  -> Gguf.load validates returned descriptors and opaque tensor handles
  -> ordinary Model.Model + Model.Params
  -> Model.inference
```

The CPU and Metal native extensions share the Rust GGUF parser. Native loading
uses positional exact reads directly into backend-owned storage. JavaScript sees
validated metadata, tensor descriptors, and opaque tensor handles, never tensor
payload bytes.

A `Registry` Effect service holds installed model implementations in one
ordinary `Map`, keyed by an exact source-qualified architecture identifier.
Applications register core or custom implementations explicitly.

## Decisions

1. There is one model abstraction. Pretrained inference produces an ordinary
   `Model.Model` and `Model.Params`.
2. `Model.Params` remains exactly `ReadonlyArray<Tensor.Any>`. Precision and
   physical encoding do not change parameter semantics.
3. Acquisition is outside the loader. The loader accepts one local path and
   performs no download, authentication, revision resolution, or cache policy.
4. A model architecture creates `Model.Model` from canonical configuration. It
   never receives weights, a GGUF file, a path, byte offsets, or runtime handles.
5. Core `Gguf.load(path)` owns native inspection, exact architecture lookup,
   canonical configuration, model construction, catalog bijection, native load,
   returned-descriptor validation, and failure cleanup.
6. `Registry` is one service containing one ordinary `Map<string,
   ModelArchitecture>`. It exposes `register`, `unregister`, and `get`; lookup is
   exact and has no aliases or candidate keys.
7. GGUF parsing and payload I/O are native. There is no TypeScript GGUF parser
   and no generic JavaScript file-range import API.
8. CPU and Metal runtime extensions expose native GGUF `inspect` and `load`.
   Both use the shared Rust parser to validate metadata, catalog geometry,
   offsets, and byte lengths.
9. Quantized formats are not `Runtime.DType`. An encoded weight is an ordinary
   tensor with logical f32 dtype and shape plus optional physical storage
   metadata on its handle.
10. `Tensor.embedding` and `Tensor.linearRows` transparently dispatch encoded
    storage. Public model code receives `Tensor.Any` and never switches on
    storage encoding or source format.
11. `QuantizedLinear` and `QuantizedEmbedding` are internal,
    architecture-neutral semantic nodes. They are not public model parameter
    types or public architecture APIs.
12. Every target format has a generic correctness implementation on CPU and
    Metal. Hand-tuned kernels and execution repacking are deferred.
13. A Metal graph never falls back to CPU for an unsupported operation.
14. The complete model is never persistently dequantized.
15. Compiler and backend execution is architecture-blind. It may switch on
    generic operations, tensor shapes, dtypes, and storage encodings, but never
    on Muse IDs, GGUF tensor names, or layer numbers.
16. Muse-required GQA, per-attention local/full masking, and RoPE layout are
    reusable tensor/compiler/runtime semantics.
17. The initial inference path is f32. F16/BF16 quantized compute is deferred.
18. Existing tensor, executable, and KV ownership remains unchanged. This RFC
    adds no new public ownership API.

## Motivation

### Existing model loading assumes the model already exists

`Model.load` restores safetensors into a model the caller already constructed.
It does not inspect foreign model metadata or select installed architecture
code.

Pretrained GGUF loading must therefore:

1. Inspect and validate the file natively.
2. Use installed architecture code to construct a model.
3. Validate an exact name/shape bijection.
4. Load tensor storage into the active runtime natively.

These jobs do not require another model abstraction or package.

### Architecture and storage are different

The Muse graph implements Muse's mathematics from canonical configuration. The
registered adapter is GGUF-specific because its canonical metadata translation,
parameter catalog, and synthesized Q/K norm tensors follow llama.cpp conversion.

The architecture creates the same logical model template regardless of storage
precision. `Model.Params` remains an ordered array of ordinary tensors.

### GGML K-quants are not scalar dtypes

`Runtime.DType` describes scalar elements with a fixed byte size. GGML K-quants
store 256 logical values in blocks containing encoded codes and shared metadata.
Their bytes per logical value are fractional.

Pretending Q2_K or Q4_K is a scalar dtype would break tensor size, stride,
readback, and generic operation assumptions. An encoded tensor therefore keeps
its logical f32 dtype and shape while its opaque handle records separate physical
storage metadata.

### Full-model dequantization is not viable

The encoded file is approximately 11.59 GiB. Expanding the approximately 27.85
billion text parameters to f16 would require roughly 52 GiB before KV state,
activations, or workspace.

Correct fallback must decode blocks inside linear or embedding work, not create
a dense model copy.

## Goals

- Load the pinned local GGUF into `Model.Model + Model.Params`.
- Keep Muse graph operations independent of GGUF while source-qualifying its
  converted-model adapter.
- Let applications register custom model implementations explicitly and load
  them by exact architecture identifier.
- Parse metadata and tensor catalogs natively without payload bytes entering
  JavaScript.
- Read exact tensor ranges directly into CPU or Metal backend-owned storage.
- Support all formats present in the target on CPU and Metal.
- Execute encoded linear and embedding operations without full-weight
  dequantization.
- Add true 32-query-head/2-KV-head GQA without repeated KV tensors.
- Implement Muse's local/local/local/global attention behavior and interleaved
  GGML RoPE layout.
- Match fixed logits and greedy tokens from checked-in fixtures.

## Non-goals

- Downloading from Hugging Face or another network source.
- Config-plus-safetensors loading.
- Split GGUF files.
- Architecture aliases or multi-key lookup.
- Automatic source-format detection.
- Generic binding transforms, tied-weight plans, or synthesized parameters.
- A TypeScript quantization schema or codec plugin API.
- Quantized training, gradients, or export.
- F16/BF16 quantized compute.
- Hand-tuned format-specific kernels.
- Panel or full-layer dequantization algorithms.
- Runtime execution repacking.
- Per-layer KV storage reclamation. Local layers retain old rows in the initial
  pool but do not attend to them.
- 131K deployment capacity in the first milestone.
- Tokenizer construction, chat-template rendering, sampling, serving, vision,
  or DFlash.
- Wrapping llama.cpp, vLLM, Transformers, MLX, Candle, or another model runtime.

## Public API

### Minimal model definition

Core architecture implementations and custom architectures use the same model
definition contract:

```ts
export interface ParameterSpec {
  readonly name: string
  readonly shape: ReadonlyArray<number>
  readonly initializer: ParameterInitializer
}

export type Params = ReadonlyArray<Tensor.Any>

export interface Definition {
  readonly parameterSpecs: ReadonlyArray<ParameterSpec>
  readonly forward: Model.Model["forward"]
}

export declare const define: (
  definition: Definition
) => Effect.Effect<Model.Model, Model.ModelError>
```

`Model.define` validates non-empty unique names, logical shapes, and initializer
descriptors, then attaches the existing compiled execution machinery. The
specification describes model semantics, not physical storage, codecs, files,
or runtime ownership.

The model parameter order remains exactly `model.parameterSpecs`. A tied logical use
reuses one parameter index. Existing dense constructors, training, save/load,
and inference retain their current behavior. GGUF loading does not add
format-specific model parameter behavior.

### Architecture implementation

```ts
export type ModelConfig = ReadonlyMap<string, unknown>

export interface ModelArchitecture {
  readonly id: string
  readonly create: (
    config: ModelConfig
  ) => Effect.Effect<Model.Model, Model.ModelError>
}
```

A `ModelArchitecture` is installed model code. `create` validates canonical
configuration and constructs the model template, including logical parameter
names and shapes. It never opens, reads, loads, or retains weights.

`ModelConfig` contains canonical architecture values, not container metadata or
weight storage. `Gguf.load` translates GGUF metadata before calling `create`.

### Architecture registry

```ts
export interface RegistryService {
  readonly register: (
    architecture: ModelArchitecture
  ) => Effect.Effect<boolean, RegistryError>
  readonly unregister: (id: string) => Effect.Effect<void>
  readonly get: (
    id: string
  ) => Effect.Effect<ModelArchitecture, RegistryError>
}
```

The live service owns one ordinary `Map<string, ModelArchitecture>`. `register`,
`unregister`, and `get` synchronously mutate or read that map. Registration
rejects an empty `architecture.id`, returns `true` when it inserts the
architecture, and returns `false` when that exact ID is already registered.

There is one lookup key: `"gguf:"` plus the exact `general.architecture` string
from GGUF. The core Muse implementation registers as `"gguf:muse-glimmer"`. There are no aliases,
namespaces, candidate lists, ambiguity rules, snapshots, freezing, or
registration tokens.

`Registry.registerAll(...architectures)` returns a Layer that uses the shared
registry. It records only architectures for which `register` returns `true` and
unregisters only those architectures in its scope finalizer.

`Registry.emptyLayer` provides an empty registry for custom installations.
`Registry.layer` is the default Layer and provides a registry containing every
built-in model architecture, including Muse-Glimmer:

```ts
import { Registry } from "@effect-torch/core"

const Models = Registry.layer
```

Applications may also register directly or compose custom registration Layers
with `Registry.emptyLayer`.

### GGUF loading

`Gguf` is exported by core:

```ts
export interface LoadedModel {
  readonly model: Model.Model
  readonly params: Model.Params
}

export declare const load: (
  path: string
) => Effect.Effect<
  LoadedModel,
  GgufError | RegistryError | Model.ModelError,
  Registry | Runtime.Runtime
>
```

`Gguf.load` performs these steps internally:

1. Obtain the active runtime's native GGUF extension.
2. Call native `inspect(path)` and validate returned metadata and descriptors.
3. Read the single non-empty `general.architecture` identifier.
4. Resolve `"gguf:" + identifier` with `Registry.get`.
5. Convert GGUF metadata into canonical `ModelConfig`.
6. Call `architecture.create(config)` to build the model template.
7. Prove a bijection between GGUF tensor names/logical shapes and
   `model.parameterSpecs`.
8. Call native backend `load(path)`.
9. Validate every loaded descriptor against inspection and every opaque handle
   against the runtime placement, logical f32 shape/dtype, and encoded storage
   metadata.
10. Order the returned tensors by `model.parameterSpecs` and return
    `{ model, params }`.
11. If loading, cancellation, or post-load validation fails, release every
    handle created by that load attempt.

The public loading flow is only:

```ts
const loaded = yield* Gguf.load(path)
const inference = yield* Model.inference(loaded.model, loaded.params, config)
```

Public loading has no intermediate lifecycle objects or alternate model
representation.

## Native GGUF Runtime

### Runtime extension

CPU and Metal runtimes expose the same core extension:

```ts
export interface GgufTensorDescriptor {
  readonly name: string
  readonly format: "F32" | Runtime.TensorStorageEncoding
  readonly logicalShape: ReadonlyArray<number>
  readonly logicalDtype: "f32"
  readonly physicalShape: ReadonlyArray<number>
  readonly physicalDtype: "f32" | "u8"
}

export interface GgufInspection {
  readonly metadata: ReadonlyArray<GgufMetadataEntry>
  readonly tensors: ReadonlyArray<GgufTensorDescriptor>
}

export interface GgufLoadEntry {
  readonly descriptor: GgufTensorDescriptor
  readonly tensor: Runtime.ConcreteTensorHandle
}

export interface GgufRuntime {
  readonly inspect: (
    path: string
  ) => Effect.Effect<GgufInspection, Runtime.BackendError>
  readonly load: (
    path: string
  ) => Effect.Effect<
    { readonly entries: ReadonlyArray<GgufLoadEntry> },
    Runtime.BackendError
  >
}
```

This is a GGUF-specific native runtime extension, not a generic file-range API.
The TypeScript boundary carries scalar metadata, descriptors, and opaque handles
only. It never carries tensor payload `ArrayBuffer`, `Uint8Array`, or block
components.

### Shared native validation

The shared Rust parser validates:

- GGUF magic and version 3;
- little-endian metadata scalars and arrays;
- safe counts, offsets, alignment, and byte-length arithmetic;
- duplicate tensor names;
- tensor dimensions and exact byte lengths;
- non-overlapping in-file tensor ranges;
- a non-empty `general.architecture`;
- F32 and the five required K-quant type codes.

Rust owns the GGML block geometry needed to calculate encoded row widths and
tensor byte lengths. Unsupported tensor types and partial blocks fail before a
handle is published.

Native `load` reparses and validates the file, then performs positional exact
reads for each validated tensor range directly into backend-owned CPU or Metal
storage. Short reads, overflow, cancellation, or concurrent truncation fail the
load. Mmap and no-copy storage remain optional future optimizations; they do not
change the public contract.

Acquisition verifies the pinned size and checksum before this RFC's target is
considered present. Concurrent mutation of the local file during inspection and
loading is unsupported.

## Encoded Tensor Semantics

### Tensor handle metadata

The minimal shared storage metadata is:

```ts
export interface EncodedTensorStorage {
  readonly encoding: "Q2_K" | "Q3_K" | "Q4_K" | "Q5_K" | "Q6_K"
  readonly physicalShape: ReadonlyArray<number>
  readonly physicalDtype: "u8"
}

export interface TensorHandle {
  readonly shape: ReadonlyArray<number>
  readonly dtype: Runtime.DType
  readonly storage?: Runtime.EncodedTensorStorage
  // Existing ownership and placement fields remain unchanged.
}
```

An encoded GGUF parameter is an ordinary `Tensor.Any` with its normal logical
f32 shape and `dtype: "f32"`. For a row-oriented logical matrix `[N, K]`, native
storage is contiguous `u8` with physical shape `[N, encodedRowBytes]`.

The encoding is storage metadata, not a scalar `DType`. Generic shape checks,
model arity, parameter order, and architecture code use the logical shape and
dtype. Code that consumes encoded storage also receives the physical metadata
needed to validate native geometry.

Compilation signatures include storage encoding, physical shape, and physical
dtype. Compiled input declarations and invocation binding preserve the same
metadata, and native input slots use the physical geometry for backend storage
validation. Rebinding a dense tensor or a differently encoded tensor therefore
cannot reuse an incompatible compiled signature.

### Public tensor operations

Architecture code uses ordinary tensor operations:

```ts
Tensor.embedding(indexes, { weight })
Tensor.linearRows(input, weight, bias?)
```

`Tensor.embedding` accepts a logical `[vocab, hidden]` weight and transparently
dispatches encoded storage to selected-row native decode. Dense storage keeps
the existing embedding path.

`Tensor.linearRows` is the generic row-oriented operation:

```text
input:  [..., K]
weight: [N, K]
bias:   optional [N] or [1, N]
output: [..., N]
```

Dense weights transpose into the existing linear/matmul path. Encoded weights
emit the internal quantized linear semantic node. Bias remains an ordinary
dense tensor. Public architecture code passes `Tensor.Any` in both cases and
does not inspect `weight.storage` or the source file.

The semantic correctness equations are:

```text
linearRows(x, w, b) = matmul(x, transpose(decode(w))) + b
embedding(ids, w)   = embedding(ids, decode(w))
```

Implementations decode only required blocks or selected rows. These equations
do not permit a persistent complete decoded weight.

### Internal semantic nodes

The graph and native runtime retain two architecture-neutral nodes:

```text
QuantizedLinear(input, weight, optionalBias, encoding, logicalShape)
QuantizedEmbedding(indexes, weight, encoding, logicalShape, optionalPaddingIndex)
```

The weight child is still the same ordinary tensor handle. Child enumeration,
remapping, generated bindings, invocation validation, cache identity, and
memory accounting include it as an ordinary graph input while preserving its
storage metadata.

Autodiff rejects both nodes explicitly. Quantized training is outside scope.

### Native codecs

Rust owns the canonical codec enum and block definitions for Q2_K, Q3_K, Q4_K,
Q5_K, and Q6_K. Native validation requires:

- exact block decoding fixtures for all five formats;
- overflow-safe row-width validation;
- logical input width equal to the encoded weight's logical K dimension;
- physical storage equal to `[N, encodedRowBytes]` u8;
- f32 accumulation;
- malformed and partial-block rejection.

No native execution code treats `Q2_K_XL` as a codec. It is only the pinned
file's mixed-quantization label.

## Compiler and Backend Execution

The compiler lowers each internal quantized semantic node directly to one
generic encoded instruction. Compilation and execution preserve both logical
tensor metadata and physical native input geometry. Compiler and backend
execution never inspect a model ID, source tensor name, metadata key, or layer
number.

### CPU

CPU implements:

- block decode inside GEMV/GEMM dot products;
- selected-row embedding decode;
- f32 accumulation;
- scratch bounded by a block or work tile.

The scalar implementation remains the correctness oracle. SIMD work is a later
optimization.

### Metal

Metal implements shared quantized linear and embedding kernels parameterized by
one of the five codec variants. The correctness path used for decode and
prefill is:

```text
load block -> decode values directly -> accumulate
```

The complete logical weight is never written to global memory. If Metal cannot
compile the generic path, compilation fails. It does not execute the operation
on CPU.

This RFC adds no exact-kernel selection, algorithm fallback hierarchy, panel
dequantization, or execution repacking.

## Reusable Transformer Changes

### Grouped-query attention

`Tensor.scaledDotProductAttention` accepts:

```text
Q: [B, Hq, Q, D]
K: [B, Hkv, K, D]
V: [B, Hkv, K, D]
```

Validation requires `Hq % Hkv === 0`. Query head `h` uses KV head:

```text
floor(h / (Hq / Hkv))
```

The result is `[B, Hq, Q, D]`. CPU, Metal, and paged attention index the KV head
directly. They do not materialize repeated K/V tensors.

Existing equal-head autodiff remains. Unequal-head autodiff fails explicitly in
this RFC.

### Per-attention window

Scaled dot-product attention options add:

```ts
readonly window?: number | null
```

- positive number: explicit causal window including the current token;
- `null`: explicit full causal attention;
- `undefined`: existing behavior, including the legacy global inference window.

Muse passes `2048` for local layers and `null` for global layers.

Decode specialization copies the resolved window onto each paged-attention
instruction. The existing pool still stores all layers and all rows up to
`InferenceConfig.maxTokens`. Local layers retain older rows but ignore them.
If every layer is local, legacy pool eviction is allowed only when the global
window is at least the largest resolved per-operation window; compilation rejects
an unsafe smaller global window. Any full layer disables pool eviction.

This is acceptable for the initial 4096-8192 token deployment target and avoids
a KV pool redesign. At 8192 tokens with f32 KV, all 52 layers require roughly
832 MiB.

### Rotary layout

The existing rotary operation adds one option:

```ts
readonly layout?: "HalfSplit" | "InterleavedPairs"
```

`HalfSplit` remains the default. `InterleavedPairs` rotates `(2i, 2i + 1)` and
matches the already converted GGML Q/K rows in the pinned file. Muse rotates the
complete 128-wide head, so no partial rotary-dimension API is added.

### RMS normalization

No new RMSNorm semantic node is required for the f32 milestone. Muse composes
existing tensor operations:

```text
rms(x, epsilon) = x * rsqrt(mean(x * x, lastDimension) + epsilon)
scaledRms(x, scale, epsilon) = rms(x, epsilon) * scale
```

The architecture uses these helpers for hidden-wide, scaleless, and per-head
Q/K normalization.

## Muse-Glimmer Architecture

`MuseGlimmer` is exported from the core package's `models` subpath. It exports
the exact registry ID `"gguf:muse-glimmer"` and its GGUF-specific `ModelArchitecture`. Default
registration is owned by `Registry.layer`, not the model module.

### Canonical configuration

The Muse architecture validates configuration and constructs:

- 52 transformer layers;
- hidden size 6656;
- FFN size 19968;
- vocabulary size 202048;
- context length 131072 in model metadata;
- 32 query heads;
- 2 KV heads;
- head dimension 128;
- local/full attention pattern from `attention.sliding_window_pattern` (4 for the pin);
- 39 local layers and 13 global layers;
- local window 2048;
- RoPE theta 500000 on local layers;
- NoPE on global layers;
- embedding scaleless RMS epsilon from `attention.layer_norm_rms_epsilon`;
- pre-attention and pre-FFN RMS epsilon from `attention.layer_norm_rms_epsilon`;
- post-attention and post-FFN RMS epsilon `1e-8`;
- per-head Q/K RMS epsilon from `attention.layer_norm_rms_epsilon`;
- Q scale 3.87 and K scale one;
- sigmoid attention gate;
- dense SwiGLU;
- final RMS epsilon from `attention.layer_norm_rms_epsilon`;
- output multiplier from canonical `logit_scale` configuration;
- final softcap from canonical `final_logit_softcapping` configuration;
- untied embedding and output head.

### Layer equation

For hidden state `h`:

```text
a = scaledRms(h, attn_norm, 1e-5)
q = heads(attn_q(a), 32, 128)
k = heads(attn_k(a), 2, 128)
v = heads(attn_v(a), 2, 128)
g = attn_gate(a)
q = scaledRms(q, q_norm, 1e-5)
k = scaledRms(k, k_norm, 1e-5)

if local:
  q, k = rotary(q, k, theta=500000, layout=InterleavedPairs)
  window = 2048
else:
  window = null

a = GQA(q, k, v, scale=1/sqrt(128), window=window)
a = reshape(a, [B, S, 4096]) * sigmoid(g)
a = attn_output(a)
h = h + scaledRms(a, attn_post_norm, 1e-8)

f = scaledRms(h, ffn_norm, 1e-5)
f = ffn_down(silu(ffn_gate(f)) * ffn_up(f))
h = h + scaledRms(f, ffn_post_norm, 1e-8)
```

Top level:

```text
h = embedding(token_embd, tokenIds)
h = rms(h, 1e-5)
h = 52 Muse layers
h = scaledRms(h, output_norm, 1e-5)
logits = linear(h, output)
logits = 20 * tanh(logits * 0.19611613513818404 / 20)
```

### Tensor names

The file contains three global tensors and fourteen tensors per layer:

```text
token_embd.weight
output_norm.weight
output.weight
blk.N.attn_norm.weight
blk.N.post_attention_norm.weight
blk.N.attn_q.weight
blk.N.attn_k.weight
blk.N.attn_v.weight
blk.N.attn_q_norm.weight
blk.N.attn_k_norm.weight
blk.N.attn_gate.weight
blk.N.attn_output.weight
blk.N.ffn_norm.weight
blk.N.post_ffw_norm.weight
blk.N.ffn_gate.weight
blk.N.ffn_up.weight
blk.N.ffn_down.weight
```

The architecture uses those names directly in `model.parameterSpecs`. The loader proves
that every source tensor is consumed exactly once with no missing or extra
entry.

The published GGUF already contains:

- Q/K rows converted to GGML interleaved RoPE layout;
- one added to centered normalization weights;
- Q-normalization vectors containing 3.87;
- K-normalization vectors containing one.

The loader does not repeat those transforms or synthesize replacement tensors.

## Package Boundaries

The design uses only existing modules, backend packages, and Rust crates:

```text
packages/core
  Model, Registry, Runtime tensor metadata, Tensor dispatch, Gguf,
  models/MuseGlimmer

packages/backend-cpu
packages/backend-apple-native
  native GGUF runtime extension and opaque-handle boundary

crates/runtime
  shared GGUF parser, catalog validation, and GGML codec definitions

crates/runtime-cpu
crates/runtime-metal
  native exact loading and generic encoded execution

crates/graph
crates/compiler
crates/autodiff
  architecture-neutral semantic nodes, lowering, and explicit gradient rejection
```

No new package is introduced. Applications depend on core and their selected
backend, then provide `Registry.layer`, or combine `Registry.emptyLayer` with
custom registration as needed.

Compiler and backend execution remain model-ID and tensor-name blind. Native
GGUF loading transports generic metadata and names for core bijection but never
branches on Muse, `"muse-glimmer"`, `token_embd.weight`, `blk.N`, or
`Q2_K_XL`. Generic codec names Q2_K-Q6_K are allowed in runtime and execution
code.

## Errors and Failure Atomicity

`GgufError` covers missing runtime support, native inspection/loading failures,
malformed boundary values, metadata failures, catalog mismatches, descriptor
mismatches, and invalid returned handles. Existing `ModelError`, `TensorError`,
and `BackendError` retain their current responsibilities.

Native inspection publishes no tensor handles. `Gguf.load` proves the complete
inspection-to-model name/shape bijection before native loading. After native
loading, it validates the complete returned descriptor/handle set before
publishing `LoadedModel`.

If native loading, cancellation, or later validation fails, all handles created
by that attempt are released. Cleanup attempts every handle even if one release
fails. Returned parameters are ordinary caller-owned concrete tensors and use
the existing `Tensor.clear` and inference ownership contracts.

The runtime extension owns handles until its `load` effect completes
successfully, including partial and late-success cleanup on interruption. Core
owns and brackets the returned archive immediately after that handoff.

The application must not modify or replace the local file while loading.

## Implementation Status and Plan

### Implemented vertical design

The corrected vertical surface is implemented across core, CPU, Metal, and the
shared Rust runtime:

- `Model.Params` remains tensor-only, and `Model.define` supports parameter
  specs plus allocation-free load-only initialization failure.
- `Registry` is an exact-key plain-map service.
- core exports `Gguf` from its root and `MuseGlimmer` from its `models` subpath;
  `Registry.layer` registers the built-in Muse-Glimmer GGUF architecture exactly as
  `"gguf:muse-glimmer"`.
- tensor handles carry optional `Runtime.EncodedTensorStorage` while retaining
  logical f32 shape/dtype.
- compiled signatures, input declarations, and bindings preserve encoded
  storage metadata and native physical geometry.
- CPU and Metal expose native GGUF inspect/load backed by the shared Rust parser.
- `Tensor.embedding` and `Tensor.linearRows` dispatch ordinary encoded tensors
  to internal quantized semantic nodes.
- CPU and Metal have generic execution paths for the five target encodings.
- GQA, explicit local/full windows, and interleaved RoPE are generic semantics.
- Muse constructs the exact load-only 731-parameter graph from canonical config.

Synthetic files and small fixtures cover the vertical contracts without the
pinned 12,444,212,256-byte file. This RFC does not claim that full-file loading,
fixed-logit parity, or greedy token parity has passed in environments where that
file is unavailable.

### Remaining validation

- Verify the pinned revision, filename, size, and SHA-256 during offline test
  setup.
- Inspect and load all 731 tensors on CPU and Metal without payload bytes
  entering JavaScript.
- Compile prefill/decode programs at the initial 4096-8192 token capacity.
- Compare fixed logits and greedy tokens against checked-in reference fixtures.
- Verify cancellation cleanup and no persistent complete decoded model at full
  scale.

### Later optimization

- Measure CPU and Metal decode/prefill before adding format-specific kernels.
- Add SIMD or specialized Metal paths only where profiles justify them.
- Preserve the same tensor, compiler, ownership, and no-fallback contracts.

## Testing

### Model

- `Model.define` rejects empty and duplicate names and invalid logical shapes.
- Load-only `init` fails before requesting a runtime.
- `Model.Params` accepts dense and encoded tensors without changing arity,
  ordering, or forward signatures.
- Existing constructors, combinators, training, save/load, and inference remain
  green.

### Registry

- Register, get, unregister, empty ID, duplicate ID, and unknown ID.
- Several architecture Layers populate the same service map.
- `Gguf.load` resolves one exact source-qualified `general.architecture` value.
- `Registry.layer` registers Muse-Glimmer exactly as `"gguf:muse-glimmer"`.
- A custom architecture loads without changing GGUF, compiler, or backend code.

### Native GGUF

- Version, every supported metadata scalar/array, alignment, offset, and
  overflow fixtures in the shared Rust parser.
- Duplicate names, unsupported types, malformed dimensions, overlapping ranges,
  short reads, and partial K-quant blocks.
- F32 and all five K-quant physical row-width and byte-length calculations.
- CPU and Metal inspection return metadata/descriptors without tensor handles.
- CPU and Metal loading uses exact positional reads into backend-owned storage.
- No tensor payload enters JavaScript.
- Inspection/load descriptor disagreement releases all returned handles.
- Pinned filename, size, checksum, metadata, and 731 tensor names when the
  external 12 GB test file is available; absence must skip or fail setup
  explicitly rather than report parity success.

### Encoded tensors and operations

- Handles retain logical f32 shape/dtype and exact encoding, physical shape, and
  physical u8 dtype.
- Compilation signatures distinguish dense and encoded storage and every
  physical geometry.
- Compiled identity/input binding preserves encoded metadata.
- Every block decoder matches checked-in f32 values.
- `Tensor.linearRows` matches explicit dequantization for decode and prefill
  shapes, with and without bias.
- `Tensor.embedding` matches explicitly dequantized selected rows.
- Dense and encoded weights execute through the same public APIs.
- Mixed formats execute in one graph.
- Invalid codec, dtype, rank, encoded row width, logical shape, or logical input
  width fails.
- Gradient requests fail explicitly for internal quantized nodes.
- Metal execution contains no CPU command or host readback.
- Generic execution never allocates a complete logical weight.

### Transformer semantics

- GQA matches an explicit repeated-KV numerical oracle for Muse's 16:1 ratio and
  one non-Muse ratio.
- Graph and KV pool retain two KV heads, not 32.
- Local/full attention matches explicit masks across the 2048 boundary.
- Compiled Muse graph has 39 local and 13 explicit full attention operations.
- Half-split RoPE regression and interleaved RoPE equation fixtures.
- Unequal-head autodiff fails explicitly; existing equal-head autodiff remains
  green.

### Muse

- Exact registry ID and registration Layer.
- Exact canonical config validation.
- Exact three-global-plus-fourteen-per-layer names and logical shapes.
- Load-only model has exactly 731 ordinary tensor parameters.
- Every source tensor is consumed exactly once with no extras.
- No load-time Q/K permutation, centered-weight adjustment, or Q/K norm
  synthesis.
- Exact gate, norm, residual, SwiGLU, and logit-transform ordering.
- Fixed-prompt final logits within quantized tolerance when the pinned file and
  fixture are available.
- Greedy token-for-token parity when the pinned file and fixture are available.
- No persistent dense model expansion.

### Boundaries

- Core Muse architecture code has no GGUF import and does not inspect tensor
  storage metadata.
- No separate GGUF or model package exists or is proposed.
- Compiler/backend execution contains no architecture identifiers or
  tensor-name branches.
- The TypeScript boundary receives no GGUF payload bytes.
- A custom registered architecture loads dynamically through the same API.
- No external model runtime is a build, test, or runtime dependency.

## Acceptance Criteria

1. `Gguf.load(path)` uses native inspection and native backend loading, resolves
   Muse exactly, and returns an ordinary `Model.Model` and tensor-only
   `Model.Params` for the pinned file.
2. The same model runs through existing `Model.inference` without an alternate
   executable path.
3. A custom architecture remains independent of GGUF and loads through exact
   registry lookup.
4. The shared Rust parser validates GGUF metadata, catalog geometry, and ranges;
   JavaScript receives metadata/descriptors and opaque handles but no payload
   bytes.
5. Native CPU and Metal loading performs positional exact reads directly into
   backend-owned storage and cleans up all handles on failure.
6. Encoded parameters remain ordinary logical f32 tensors with optional
   `Runtime.EncodedTensorStorage`; encoded formats are never `Runtime.DType`.
7. Compilation signatures and input binding preserve storage encoding, physical
   shape, physical dtype, and native input geometry.
8. Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, and F32 coexist in one model on CPU and Metal.
9. Public `Tensor.embedding` and `Tensor.linearRows` match explicit f32
   dequantization fixtures for dense and encoded weights.
10. Metal executes all five formats without CPU fallback or a complete
    logical-weight allocation.
11. Muse uses 32 query heads and stores two KV heads without repeated K/V
    tensors.
12. Muse applies 2048-token local attention to 39 layers and full attention to
    13 layers, with interleaved RoPE only on local layers.
13. Muse matches fixed logits and greedy tokens once the pinned file is available
    to the offline parity suite.
14. Existing dense tests remain green.
15. Compiler and backend execution contains no Muse-specific behavior or
    tensor-name branching.
16. No external model runtime or downloaded repository code is required.

## Follow-up Work

- Measured CPU SIMD and Metal format-specific kernels.
- Local-layer KV reclamation and larger context capacities.
- Split GGUF files.
- Other source formats producing params for the same registered architectures.
- Tokenizer/chat integration.
- Muse vision and DFlash.
- Additional architecture implementations.

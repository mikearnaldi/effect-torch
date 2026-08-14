# effect-torch

effect-torch is a native tensor runtime and machine-learning system for
TypeScript applications built with [Effect](https://effect.website).

The public API is backend-neutral TypeScript. The execution engine is Rust,
implemented in this repository from the graph IR through autodiff, compilation,
CPU kernels, Metal kernels, memory management, and Node-API bindings.

The current system includes:

- A lazy semantic tensor graph with strict shape, dtype, and placement checks.
- Reverse-mode autodiff, VJP, JVP, vmap, and gradient checkpointing.
- Explicit reusable executable compilation with bounded, runtime-aware caches.
- Independent native CPU and Apple Metal backends.
- Pure model, optimizer, trainer, checkpoint, and learning-rate APIs.
- Compiled training and paged KV-cache inference with batched decode.
- Native safetensors I/O and a standalone tokenizer package.
- Structured Effect errors, interruption, cancellation, and explicit resource
  release.
- A 14-artifact release build that compiles Darwin binaries natively on macOS
  and cross-compiles Linux binaries in the same run.

## Contents

- [Packages](#packages)
- [Quick Start](#quick-start)
- [Programming Model](#programming-model)
- [Architecture](#architecture)
- [Backend Capabilities](#backend-capabilities)
- [Public API](#public-api)
- [Compilation](#compilation)
- [Models and Training](#models-and-training)
- [Compiled Inference](#compiled-inference)
- [Safetensors](#safetensors)
- [Tokenizers](#tokenizers)
- [Errors and Cancellation](#errors-and-cancellation)
- [Native Distribution](#native-distribution)
- [Development](#development)
- [Repository Layout](#repository-layout)
- [Current Constraints](#current-constraints)
- [Design Documents](#design-documents)

## Packages

| Package                              | Responsibility                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------------- |
| `@effect-torch/core`                 | Backend-neutral tensors, autodiff, compilation, models, training, and inference |
| `@effect-torch/backend-cpu`          | CPU Runtime Layer and CPU-owned native addon                                    |
| `@effect-torch/backend-apple-native` | Apple Metal Runtime Layer and Metal-owned native addon                          |
| `@effect-torch/tokenizers`           | Native tokenizer loading, encoding, decoding, and training                      |
| `@effect-torch/examples`             | Private runnable examples                                                       |
| `@effect-torch/bench`                | Private CPU, Metal, and optional MLX benchmarks                                 |

`@effect-torch/core` has no dependency on a concrete backend. Applications
select one Runtime Layer at the edge of the Effect program.

The tokenizer package is also independent of core. It returns host-owned
`Uint32Array` token IDs that are imported explicitly into whichever tensor
runtime the application selected.

## Quick Start

The scoped packages are currently consumed from this workspace; they are not
published on the public npm registry yet. The package names in the examples are
the workspace and intended distribution names.

The repository is pinned to `effect@4.0.0-beta.101`. Code using these packages
must use the same Effect major/version family; npm's current unqualified
`effect` release is Effect 3 and is not API-compatible.

From a repository checkout, prepare the environment and a host CPU addon:

```bash
direnv allow
pnpm install
pnpm --filter @effect-torch/backend-cpu build:debug
```

The following is the minimal application shape:

```ts
import * as BackendCpu from "@effect-torch/backend-cpu"
import { Tensor } from "@effect-torch/core"
import { Effect } from "effect"

const program = Effect.gen(function*() {
  const a = yield* Tensor.randn([512, 512])
  const b = yield* Tensor.randn([512, 512])
  const product = yield* Tensor.matmul(a, b)
  const shifted = yield* Tensor.add(
    product,
    yield* Tensor.constantLike(product, 1)
  )
  const mean = yield* Tensor.mean(shifted)

  const [value] = yield* Tensor.compute([mean])
  const numbers = yield* Tensor.toNumberArray(value)
  return numbers[0]
})

const result = await Effect.runPromise(
  program.pipe(Effect.provide(BackendCpu.layer))
)
```

On macOS, build and provide the Apple backend instead:

```bash
pnpm --filter @effect-torch/backend-apple-native build:debug
```

```ts
import * as BackendApple from "@effect-torch/backend-apple-native"
import { Tensor } from "@effect-torch/core"
import { Effect } from "effect"

const program = Effect.gen(function*() {
  const tensor = yield* Tensor.ones([2, 2])
  return yield* Tensor.toNumberArray(tensor)
})

const result = await Effect.runPromise(
  program.pipe(Effect.provide(BackendApple.layer))
)

const reportedAvailable = await Effect.runPromise(
  BackendApple.isAvailable
)
```

The Apple package entrypoint is safe to import on every platform. `isAvailable`
defers loading the native addon and returns `false` on unsupported platforms,
unsupported architectures, missing artifacts, or when Metal device, command
queue, or shared-event creation fails. `makeRuntime` loads the addon only when
the Metal runtime is actually requested.

## Programming Model

### Runtime Is an Effect Service

`Runtime.Runtime` is the authoritative backend service for an Effect program.
Tensor values do not retain a service reference. They retain immutable metadata
and opaque backend-owned handles.

```ts
import { Runtime, Tensor } from "@effect-torch/core"
import { Effect } from "effect"

const inspect = Effect.gen(function*() {
  const runtime = yield* Runtime.Runtime
  const tensor = yield* Tensor.ones([2, 2])

  return {
    backend: runtime.backend.name,
    placement: tensor.placement,
    dtype: tensor.dtype,
    shape: tensor.shape
  }
})
```

Both native backend packages expose:

```ts
import { Runtime } from "@effect-torch/core"
import { Layer } from "effect"

declare const makeRuntime: () => Runtime.RuntimeService
declare const layer: Layer.Layer<Runtime.Runtime>
```

`makeRuntime()` is a lazy, memoized factory. Importing the public module does
not create the RuntimeService. The first direct call or first Layer build
creates it, and later calls return the same service object. `layer` uses
`Effect.sync(makeRuntime)` so service construction remains deferred.

The native addon itself is selected and loaded by the package loader when the
backend module is imported.

### Lazy and Concrete Tensors

The two principal tensor states are:

| Type              | Meaning                                            |
| ----------------- | -------------------------------------------------- |
| `Tensor.Lazy`     | A node in a backend-owned computation graph        |
| `Tensor.Concrete` | Materialized storage owned by the selected runtime |
| `Tensor.Any`      | Either state; accepted by graph operations         |

Constructors and operations return Effects that build graph nodes. Numeric
execution is deferred:

```ts
import { Tensor } from "@effect-torch/core"
import { Effect } from "effect"

const graph = Effect.gen(function*() {
  const x = yield* Tensor.ones([2, 3])
  const y = yield* Tensor.full([1, 3], 2)
  const z = yield* Tensor.mul(x, y)
  return yield* Tensor.sum(z)
})
```

Graph construction validates metadata and handle ownership. It does not run a
hidden CPU fallback, copy a foreign tensor, or execute a kernel.

### Compilation and Materialization

`Tensor.compute` submits related roots as one native compile request and executes
the resulting executable:

```ts
import { Tensor } from "@effect-torch/core"
import { Effect } from "effect"

const evaluate = (loss: Tensor.Any, gradient: Tensor.Any) =>
  Effect.gen(function*() {
    const [lossValue, gradientValue] = yield* Tensor.compute([
      loss,
      gradient
    ])
    return { lossValue, gradientValue }
  })
```

One request provides important semantics:

- Shared subgraphs lower and execute once.
- Multiple roots observe the same draw from shared random nodes.
- Intermediate lifetimes and reusable workspace are planned from final uses.
- Execution runs off the JavaScript event loop.
- Effect interruption is connected to native cancellation.

`Tensor.toTypedArray` and `Tensor.toNumberArray` materialize a lazy tensor when
needed, or read an existing concrete tensor.

Ordinary `compute` and explicitly reusable programs use the same compiler,
memory planner, and executor. `compute` obtains a transient or structurally
cached executable; `Tensor.compile`, a model's `execute` method, `Trainer.make`,
and the inference APIs retain reusable executable handles.

### Resource Ownership

Every lazy graph, concrete tensor, compiled program, decode program, KV pool,
and KV sequence is represented by an opaque immutable TypeScript handle. Backend
adapters maintain private ownership records in `WeakMap`s.

Consequences:

- CPU handles cannot be used by Metal and vice versa.
- Cleared handles fail with a typed `invalid-handle` error.
- Foreign handles fail with a typed `foreign-handle` error.
- Equivalent calls to one backend's memoized `makeRuntime()` share handle
  ownership and stable runtime identity.
- TypeScript signature caches belong to each compiled function, model, or
  trainer. Each runtime also owns a bounded structural executable cache whose
  entries share immutable plans without retaining generated concrete bindings.
  Inference artifacts own a fixed set of eagerly compiled prefill/decode
  programs rather than a shape-keyed program cache.
- An executable owns immutable typed instructions, memory and physical plans,
  pipelines, constants, signatures, and diagnostics, but not one permanent
  invocation workspace. Calls lease runtime-owned workspace and provisional
  output storage; successful outputs take independent ownership of their
  backing.
- There is no implicit cross-device transfer.

Concrete tensors can be released deterministically:

```ts
import { Tensor } from "@effect-torch/core"
import { Effect } from "effect"

const release = (graph: Tensor.Any) =>
  Effect.gen(function*() {
    const [value] = yield* Tensor.compute([graph])
    yield* Tensor.clear(value)
  })
```

Native finalizers remain a GC fallback. CPU external buffers are included in
Node external-memory accounting, and diagnostics expose the backend's current
external byte count when available.

### Dtypes

The tensor dtype vocabulary is:

```ts
type DType =
  | "f32"
  | "f64"
  | "f16"
  | "bf16"
  | "i64"
  | "u8"
  | "u32"
```

Operations are generally strict. Mixed dtypes fail and `Tensor.cast` performs
explicit conversion. The intentional exception is a 0-dimensional floating
scalar combined with a non-scalar floating tensor: the scalar is coerced to the
tensor's dtype. This allows runtime learning-rate scalars to participate in
BF16 graphs without changing tensor storage.

JavaScript has no BF16 typed array. F16 and BF16 readback is widened to
`Float32Array`. `Tensor.toNumberArray` rejects I64 to avoid silently converting
bigints to numbers.

## Architecture

```text
Application Effect program
            |
            v
@effect-torch/core
  backend-neutral TypeScript API
  Runtime service contract
  lazy/concrete opaque handles
            |
            +-------------------------------+
            |                               |
            v                               v
@effect-torch/backend-cpu          @effect-torch/backend-apple-native
  CPU adapter                         Metal adapter
  CPU handle registry                 Metal handle registry
  CPU N-API surface                   Metal N-API surface
            |                               |
            v                               v
effect-torch-runtime-cpu            effect-torch-runtime-metal
  CPU buffers and kernels              Metal buffers and kernels
  typed CPU executable                  typed Metal executable
            |                               |
            +---------------+---------------+
                            |
                  statically linked crates
      runtime, graph, compiler, autodiff, and N-API helpers

@effect-torch/tokenizers is a separate TypeScript + Rust N-API package.
It does not participate in Runtime.Runtime.
```

### TypeScript Boundary

`@effect-torch/core` defines the public contract:

- `RuntimeService` describes graph construction, compilation/execution,
  autodiff, readback, release, and its required extension facilities.
- Tensor handles expose only immutable shape, dtype, device, and placement
  metadata.
- Higher-level APIs build on the Runtime service without importing CPU or
  Metal code.
- Backend selection happens through an Effect Layer.

The CPU and Metal adapters translate between the public contract and their own
native addon. They validate every handle before native code receives it and map
native failures into structured `Runtime.BackendError` values.

### Rust Crates

| Crate                        | Responsibility                                                                                        |
| ---------------------------- | ----------------------------------------------------------------------------------------------------- |
| `effect-torch-runtime`       | Dtypes, layouts, dense IDs, signatures, memory/diagnostic contracts, ownership, cancellation          |
| `effect-torch-graph`         | Nongeneric semantic `Node`/`NodeKind` graph, metadata, leaves, and semantic traversal                 |
| `effect-torch-compiler`      | Requests, shared graph index, side-table regions, `KernelExpr`, typed lowered tables, memory planning |
| `effect-torch-autodiff`      | Reverse-mode graph transformation, vmap, JVP/VJP, and checkpoint semantics                            |
| `effect-torch-napi`          | Backend-neutral cancellation, async execution, and byte-buffer helpers                                |
| `effect-torch-runtime-cpu`   | CPU values/instructions, lowering, kernels, physical execution, storage, and CPU N-API addon          |
| `effect-torch-runtime-metal` | Metal values/instructions, lowering, pipelines, physical execution, storage, and Metal N-API addon    |
| `effect-torch-tokenizers`    | Tokenizer-only N-API addon backed by the Rust `tokenizers` crate                                      |

The direct shared-crate dependencies keep autodiff independent of the compiler:

```text
effect-torch-graph --------> effect-torch-runtime
effect-torch-autodiff -----> effect-torch-graph, effect-torch-runtime
effect-torch-compiler -----> effect-torch-graph, effect-torch-runtime

runtime-cpu and runtime-metal consume graph, compiler, runtime, and, for their
Node-API graph/autodiff surface, autodiff.
```

The shared compiler driver and typed tables are internal statically dispatched
Rust abstractions, not a stable plugin ABI. Each Node addon statically links the
shared Rust crates it needs.

### Independent Native Backends

CPU and Metal do not select behavior through one shared feature-gated addon.
Each runtime crate owns:

- Its concrete tensor value type.
- Its typed lowered values, instructions, algorithms, and physical executor.
- Its N-API classes and functions.
- Its safetensors integration.
- Its native `cdylib` output.

The CPU addon has no Metal branches or imports. The Metal addon has no CPU
branches or imports. Apple artifacts are Darwin-only and link Metal.framework;
CPU artifacts are verified not to link Metal.framework.

`effect-torch-napi` remains an `rlib` with backend-neutral utilities only. No
Rust object crosses between separately loaded `.node` files.

### Graph and Execution Engine

The native graph stores nongeneric semantic operations, child relationships,
shape, dtype, placement, and leaf ownership. Compilation accepts one
`ProgramRequest`, creates one `PreparedProgram` and stack-safe `GraphIndex`, and
uses dense side tables for topology, consumers, roots, slots, generated leaves,
and random provenance. Shared nodes appear once and caller root order is
preserved. Generated leaves are collected once for structural cache lookup,
insertion, and binding order.

Autodiff, vmap, and checkpointing construct semantic graphs before this
pipeline. Stateful inference uses one shared compiler specialization for CPU
and Metal to create its decode graph and state-cursor contract, then indexes
that specialized graph once.

CPU and Metal lower the prepared graph into backend-typed `LoweredProgram`
values and instructions. Each instruction declares inputs, outputs, scratch,
staging, status, state, and effects. The memory planner consumes those exact
declarations; backend physical plans add synchronization by `InstructionId`
without copying tensor semantics. Execution resolves the fixed plan against an
invocation frame and publishes separately owned output storage.

Invocation does not traverse a semantic graph, run fusion, discover
intermediate allocations, compile pipelines, or fall back to another execution
engine. `optimize: false` uses the same typed lowering, memory planning,
ownership, and execution path with optional regions disabled.

### Compiler and Fusion

The compiler records elementwise, fused-reduction, multi-output, GEMM-epilogue,
and optimizer choices as regions over `GraphIndex`. These are code-generation
side tables, not semantic `Fused*` node kinds. Multi-output selection uses a
region dependency DAG and bounded worklist, including split regions that
duplicate a prefix expression when required to preserve transitive ancestry.

`KernelExpr` is the narrow scalar expression body inside fused instructions.
Backend lowering turns regions and uncovered semantic nodes directly into typed
CPU or Metal instructions with retained algorithm/resource plans. Required
Metal pipelines are prepared during executable compilation, and compiler phase
timings plus structural instruction, memory, command, synchronization, and
region-work metrics come from the authoritative artifacts.

Current execution paths include:

- CPU elementwise and reduction fusion for F32 and F64.
- Metal elementwise and reduction fusion for F32 and BF16.
- Multi-output shared-prefix fusion and GEMM residual/GELU epilogues.
- Typed semantic-kernel instructions for layer normalization, loss, attention,
  KDA, convolution, rotary operations, and paged KV state where supported.
- Deterministic liveness-based segmented memory plans and runtime-owned
  workspace/output pools.

Executable compile options currently control optimization and inference-only
constant weights. The unused executable precision option was removed until a
lowering policy is specified; Trainer mixed-BF16 remains a separate graph and
training policy.

### CPU Runtime

The CPU runtime owns typed host buffers and implements tensor operations in
Rust. F32 and F64 GEMM use `matrixmultiply`; other operations use repository
kernels and composed primitives. It includes convolution, indexing, reduction,
pooling, random generation, linalg, safetensors, typed executable lowering and
execution, fusion kernels, KV-cache execution, and the complete CPU N-API
surface.

Unsupported capability paths return structured errors through the adapter. In
particular, F16 and BF16 storage are supported, but CPU half-precision matmul is
currently unsupported.

### Metal Runtime

The Metal runtime is implemented against Apple's Metal APIs through `objc2`.
It owns device buffers, command encoding, pipeline caches, generated Metal
shader source, GEMM, flash attention, convolutions, indexing, rotary kernels,
paged KV-cache operations, fusion kernels, typed lowering, physical instruction
plans, and runtime-owned segmented storage pools.

Each invocation owns its submission context and storage leases while immutable
executable plans and pipeline caches are shared. Metal never compiles or falls
back to CPU during execution of an unsupported program; unsupported lowering or
pipeline preparation fails executable compilation.

### Async Execution and Cancellation

Compiled materialization, reusable execution, decode, readback, and safetensors
I/O execute through asynchronous native promises backed by Tokio's blocking
task pool. The TypeScript adapter connects the Effect fiber's abort signal to a
native `CancellationToken`. Native cancellation and successful completion
atomically compete to commit one result. Graph construction, autodiff
transformation, and program compilation are synchronous native calls and are
not interruptible.

Interrupted work is drained before late native results are discarded. Late
tensor and archive results are explicitly cleaned up where the adapter owns
their buffers; discarded readback buffers are reclaimed by their external
ArrayBuffer finalizers. This applies to transient and reusable programs, decode,
readback, and safetensors I/O.

## Backend Capabilities

| Capability                   | CPU                                | Apple Metal                        |
| ---------------------------- | ---------------------------------- | ---------------------------------- |
| Platforms                    | macOS and Linux                    | macOS                              |
| Architectures                | arm64 and x64                      | arm64 and x64                      |
| Advertised tensor dtypes     | F32, F64, F16, BF16, I64, U8, U32  | F32, F16, BF16, I64, U8, U32       |
| F32 matmul                   | Yes                                | Yes                                |
| F64                          | Storage, math, matmul, and linalg  | Unsupported                        |
| F16/BF16 storage             | Yes                                | Yes                                |
| F16/BF16 matmul              | Not currently supported            | Yes                                |
| Graph compilation            | Yes                                | Yes                                |
| Autodiff                     | Yes                                | Yes                                |
| Elementwise/reduction fusion | F32, F64                           | F32, BF16                          |
| Scaled dot-product attention | Composed backend path              | Native flash path for F32 and BF16 |
| `inverse`, `det`, `solve`    | Yes                                | Explicitly rejected                |
| Mixed-BF16 training          | Not advertised                     | Yes                                |
| Paged KV cache               | F32, F16, BF16, INT8 storage tiers | F32, F16, BF16, INT8 storage tiers |
| Safetensors path I/O         | Yes                                | Yes, with Metal dtype validation   |

Backend capabilities are explicit. Requesting unsupported placement or dtype
behavior fails; no graph is silently moved to another runtime.

## Public API

`@effect-torch/core` exports namespaces rather than one flat symbol list:

| Namespace      | Responsibility                                                   |
| -------------- | ---------------------------------------------------------------- |
| `Runtime`      | Backend contract, handles, capabilities, errors, and service tag |
| `Tensor`       | Tensor graph construction, evaluation, compilation, and I/O      |
| `Gradient`     | Autodiff transforms                                              |
| `Loss`         | Regression and classification losses                             |
| `Model`        | Layers, composition, execution, and compiled inference           |
| `Optimizer`    | SGD, Adam, AdamW, clipping, and full-step execution              |
| `LearningRate` | Constant, exponential, stepwise, cosine, and warmup schedules    |
| `Trainer`      | Compiled and reference training loops                            |
| `Checkpoint`   | Trainer and sampler checkpoint persistence                       |
| `Sampler`      | Restorable shuffled token-window sampling                        |

### Tensor Constructors

```text
constant           constantLike
zeros              zerosLike
ones               onesLike
full               fullLike
randn              uniform
arange             linspace
eye                fromTypedArray
```

Constructors accept explicit dtype options where applicable.
`fromTypedArray` infers dtype from the JavaScript typed array.

### Elementwise and Activation Operations

```text
add                sub                 mul                 div
maximum            minimum             remainder           where
eq                 ne                  gt                  lt
ge                 le                  logicalAnd          logicalOr
logicalNot         clamp               cast

neg                abs                 sign                sqrt
rsqrt              square              reciprocal          pow
exp                expm1               log                 log1p
log2               log10               sin                 cos
tan                sinh                cosh                tanh
erf                floor               ceil                round

sigmoid            relu                silu                gelu
mish               elu                 leakyRelu           softplus
hardtanh
```

Binary operations broadcast like NumPy and support data-first and data-last
usage:

```ts
import { Tensor } from "@effect-torch/core"
import { Effect } from "effect"

const addBothWays = (a: Tensor.Any, b: Tensor.Any) =>
  Effect.gen(function*() {
    const first = yield* Tensor.add(a, b)
    const second = yield* a.pipe(Tensor.add(b))
    return { first, second }
  })
```

Numbers are not implicit tensor operands. Use `constantLike` when a scalar must
match an existing tensor's dtype and placement.

### Reductions

```text
sum                mean                max                 min
prod               variance            std                 norm
logsumexp          all                 any
argmax             argmin              cumsum
```

Most reductions accept `{ dims?, keepdims? }`. Negative dimensions count from
the end. Variance and standard deviation accept a correction; norm accepts an
order.

### Shape and Indexing

```text
reshape            flatten             squeeze             unsqueeze
transpose          slice               split               chunk
concat             stack               broadcastTo         tile
pad                take                gather              scatterAdd
flip               oneHot              embedding           triu
tril               trace
```

`take`, `gather`, and `embedding` accept I64 or U32 index tensors.
`scatterAdd` accumulates duplicate indexes and is used by indexing gradients.

### Neural-Network and Linear-Algebra Primitives

```text
matmul                         dot
linear                         layerNorm
positionEmbedding              rotaryEmbedding
softmax                        logSoftmax
scaledDotProductAttention      dropout
crossEntropy

conv1d                         conv2d
convTranspose1d                convTranspose2d
maxPool2d                      avgPool2d

inverse                        det
solve
```

Linalg placement constraints are listed in the backend capability table.

### Losses

`Loss` provides:

```text
mse
l1
huber
binaryCrossEntropy
crossEntropy
nll
klDiv
hinge
cosineEmbeddingLoss
```

Losses accept a reduction of `"mean"`, `"sum"`, or `"none"`; the default is
`"mean"`. A scalar mean loss can be passed directly to `Gradient.grad`.

### Autodiff

Autodiff transforms an existing lazy graph. It does not trace a JavaScript
function.

```ts
import { Gradient, Loss, Tensor } from "@effect-torch/core"
import { Effect } from "effect"

const step = (input: Tensor.Any, weight: Tensor.Any, target: Tensor.Any) =>
  Effect.gen(function*() {
    const prediction = yield* Tensor.matmul(input, weight)
    const loss = yield* Loss.mse(prediction, target)
    const [gradient] = yield* Gradient.grad(loss, [weight])

    const [lossValue, gradientValue] = yield* Tensor.compute([
      loss,
      gradient
    ])

    return { lossValue, gradientValue }
  })
```

The public transforms are:

| Export                           | Meaning                                                          |
| -------------------------------- | ---------------------------------------------------------------- |
| `grad(loss, wrt)`                | Reverse-mode gradients of scalar `loss`                          |
| `vjp(y, x, cotangent)`           | Vector-Jacobian product                                          |
| `jvp(y, x, tangent)`             | Jacobian-vector product using a double reverse-mode construction |
| `vmap(y, x, batchedX, options?)` | Native graph batching rewrite                                    |
| `stopGradient(tensor)`           | Blocks gradient flow                                             |
| `checkpoint(tensor)`             | Recomputes intermediates during backward                         |

Adjoints are ordinary semantic graph nodes, so higher derivatives work where
the complete operation path is differentiable. Optimized cross-entropy and
scaled-dot-product-attention backward paths are currently first-order only.

## Compilation

### Generic Compiled Functions

`Tensor.compile` creates a reusable function over tensor inputs:

```ts
import { Tensor } from "@effect-torch/core"
import { Effect } from "effect"

const runCompiled = (x: Tensor.Any, weight: Tensor.Any) =>
  Effect.gen(function*() {
    const compiled = yield* Tensor.compile(([input, parameter]) =>
      Effect.gen(function*() {
        const product = yield* Tensor.matmul(input, parameter)
        return [yield* Tensor.relu(product)]
      }), { cacheCapacity: 8 })

    const [output] = yield* compiled.call([x, weight])
    const stats = yield* compiled.stats

    yield* compiled.clear
    return { output, stats }
  })
```

Compilation behavior:

- `Tensor.compile` initially creates only the TypeScript compiled-function
  wrapper and its cache. It cannot trace yet because no input exemplars or
  runtime have been supplied.
- The first call for a signature traces against placeholder inputs.
- That call obtains `Runtime.Runtime`; this lets one compiled function specialize
  lazily for the backend, shape, placement, and dtype of its actual inputs.
- The semantic roots are prepared and lowered into a backend-owned typed
  executable.
- Later calls bind new inputs and execute that fixed plan.
- Signatures include runtime identity, placement, shape, and dtype.
- Each compiled function owns its cache; stable runtime identity partitions
  entries without creating duplicate entries for the same backend.
- Cache capacity defaults to 32 and uses bounded LRU eviction.
- Concurrent misses for one signature are single-flight.
- Failed traces are not cached.
- Random nodes draw fresh values on every program execution.
- Materializing a placeholder while tracing fails.

`compiled.stats` reports the number of currently cached programs and the total
number of trace attempts. `compiled.clear` drops the cache's references without
resetting that historical count; native program destruction follows normal
handle reachability and finalization.

### Lower-Level Program API

The Tensor namespace also exposes the primitives used by trainers and models:

```text
makeProgramCache       cachedProgram          signatureOf
makeInput              makeScalarInput        freezeProgram
runProgram             compileDecodeProgram  runDecodeProgram
```

Most applications should use `Tensor.compile`, a model's `execute` method,
`Trainer.make`, or `Model.inference` instead.

## Models and Training

### Models

A `Model.Model` describes a functional model containing:

- Ordered parameter names.
- An Effect that initializes a flat parameter array.
- A lazy `forward` graph builder.
- A compiled `execute` path.
- Compilation cache statistics and explicit cache clearing.

There is no mutable learned parameter or running tensor state and no
model-specific backward method. A model does memoize its compiled execution
function after the first `execute` call; `stats` and `clear` expose that cache.

Parameterized layers include:

```text
linear
conv1d
conv2d
embedding
positionEmbedding
layerNorm
multiHeadAttention
```

Parameterless layers include activations, softmax, flatten, dropout, and
pooling. Composition includes:

```text
chain
add
merge
residual
checkpoint
mapInput
```

Example:

```ts
import { Model, Tensor } from "@effect-torch/core"
import { Effect } from "effect"

const runModel = (input: Tensor.Any) =>
  Effect.gen(function*() {
    const model = yield* Model.chain(
      yield* Model.linear("fc1", 2, 8),
      yield* Model.tanh,
      yield* Model.linear("fc2", 8, 1),
      yield* Model.sigmoid
    )

    const params = yield* Tensor.compute(yield* model.init)
    const lazyOutput = yield* model.forward(params, input)
    const concreteOutput = yield* model.execute(params, input)
    return { lazyOutput, concreteOutput }
  })
```

Use `forward` for composition and differentiation. Use `execute` for repeated
materialized evaluation; it creates and reuses a compiled function lazily.

Multi-head attention uses fused QKV parameters named:

```text
<name>.qkv.weight
<name>.qkv.bias
<name>.wo.weight
<name>.wo.bias
```

### Optimizers

The optimizer API includes SGD, Adam, and AdamW. Optimizers are pure graph
transforms: parameters and state are tensors, and nothing is mutated in place.

```ts
import { Optimizer, Tensor } from "@effect-torch/core"
import { Effect } from "effect"

const update = (
  params: ReadonlyArray<Tensor.Any>,
  grads: ReadonlyArray<Tensor.Any>,
  lrTensor: Tensor.Any
) =>
  Effect.gen(function*() {
    const optimizer = yield* Optimizer.adamW({ weightDecay: 0.01 })
    const state = yield* optimizer.init(params)
    return yield* optimizer.step(params, grads, state, lrTensor)
  })
```

`Optimizer.step(optimizer, loss, params, state, lr)` is the full-step helper. It
builds gradients and updates, then compiles loss, new parameters, and optimizer
state as one multi-root executable request.

Gradient transforms include `clipByValue` and `clipByGlobalNorm` for custom
training loops.

### Learning-Rate Schedules

```text
constant
exponential
stepwise
cosine
withWarmup
```

Schedules are plain `(step: number) => number` functions. The trainer converts
the result to a runtime scalar for the compiled update graph.

### Compiled Trainer

`Trainer.make` compiles the entire forward, loss, backward, and optimizer update
for each input signature. `Trainer.makeUncompiled` provides the reference loop.

```ts
import { LearningRate, Loss, Model, Optimizer, Tensor, Trainer } from "@effect-torch/core"
import { Effect } from "effect"

const train = (model: Model.Model, input: Tensor.Any, target: Tensor.Any) =>
  Effect.gen(function*() {
    const trainer = yield* Trainer.make(model, {
      optimizer: yield* Optimizer.adam(),
      lr: LearningRate.constant(0.1),
      loss: Loss.mse,
      data: { input, target },
      stop: ({ step, loss }) => step >= 3000 || loss < 1e-4,
      onStep: ({ step, loss }) =>
        step % 100 === 0
          ? Effect.log(`step=${step} loss=${loss}`)
          : Effect.void
    })

    return yield* trainer.train()
  })
```

Training data can be fixed or produced by an effectful per-step function.
Trainer callbacks receive a 1-based step, loss, and elapsed duration. The
learning-rate schedule receives a 0-based step. At least one step executes, and
`onStep` runs before the stop policy is checked.

The compiled trainer releases parameter and state generations that it owns once
their replacements commit.

### Mixed BF16

Trainer precision is `"f32"` or `"mixedBf16"`.

Mixed BF16 keeps F32 master parameters and optimizer state, casts masters to
BF16 at the forward boundary, runs forward and backward in BF16, and propagates
gradients through the casts to the F32 update. It requires the runtime feature
`"mixed-bf16"`, currently advertised by Apple Metal. The trainer casts model
parameters, not the data source: floating inputs and regression targets must be
provided in a BF16-compatible dtype. Integer class targets remain appropriate
for classification losses such as cross-entropy. Every operation selected by
the model and loss must support BF16; for example, dropout and `Loss.nll` do not
currently support BF16.

### Checkpoints and Samplers

Trainer checkpoints use safetensors and include model parameters, optimizer
state roots, and the global step:

```ts
import { Checkpoint, Trainer } from "@effect-torch/core"
import { Effect } from "effect"

const saveAndResume = <S>(
  trainer: Trainer.Trainer<S>,
  trained: Trainer.Trained<S>
) =>
  Effect.gen(function*() {
    yield* Checkpoint.save("training.safetensors", trainer, trained)

    const restored = yield* Checkpoint.load(
      "training.safetensors",
      trainer
    )

    return yield* trainer.train(
      restored.params,
      restored.resume
    )
  })
```

`Checkpoint.saveWithSampler` and `loadWithSampler` also persist the complete
state of a `Sampler`, including shuffled order, cursor, epoch, and batch
configuration. This supports exact continuation within the saved permutation.
The sampler does not persist JavaScript RNG state, so reshuffling after that
permutation is exhausted is a new random event.

## Compiled Inference

`Model.inference` transforms a causal attention model into compiled prefill and
decode programs backed by a paged KV cache:

```ts
import { Model, Tensor } from "@effect-torch/core"
import { Effect } from "effect"

const generate = (
  model: Model.Model,
  params: Model.Params,
  promptTensor: Tensor.Any,
  generatedTokens: ReadonlyArray<number>
) =>
  Effect.gen(function*() {
    const inference = yield* Model.inference(model, params, {
      maxTokens: 8192,
      blockSize: 16,
      prefillChunk: 16,
      attentionWindow: 256,
      kvDtype: "bf16",
      decodeBatch: 8
    })

    const generation = yield* inference.generation()
    const entry = yield* generation.add(promptTensor)
    let logits = entry.logits

    for (const token of generatedTokens) {
      ;[logits] = yield* generation.step([
        { seq: entry.seq, token }
      ])
    }

    yield* entry.seq.finish()
    return logits
  })
```

The inference transform:

- Verifies parameter arity and materializes parameters once.
- Traces the model's existing `forward` graph.
- Rewrites causal attention into paged KV-cache operations.
- Rewrites supported position operations to cursor-aware forms.
- Allocates one shared block pool.
- Compiles fixed-shape prefill, single-sequence decode, and optional batched
  decode programs.
- Rejects models without cacheable causal attention.

Generation sessions support:

- Chunked prompt prefill.
- Single and batched token steps.
- Ragged batches padded internally by native code.
- Content-addressed whole-block prefix reuse.
- Explicit sequence finish and session close.
- Sliding-window attention.
- RoPE with bounded active context and unbounded sequence cursors.

`kvDtype: "int8"` is a KV storage tier, not a normal tensor dtype. Cached rows
are quantized with per-token, per-head scales and widened for attention math.

The low-level `Tensor` namespace also exposes KV pools, sequences, cursor
queries, prefix matching, decode compilation, and direct decode execution.

## Safetensors

Both tensor backends expose direct path-based safetensors I/O through a Runtime
extension:

```ts
import { Tensor } from "@effect-torch/core"
import { Effect } from "effect"

const roundTrip = (weight: Tensor.Any, bias: Tensor.Any) =>
  Effect.gen(function*() {
    yield* Tensor.save(
      "weights.safetensors",
      {
        "model.weight": weight,
        "model.bias": bias
      },
      {
        metadata: { framework: "effect-torch" }
      }
    )

    const archive = yield* Tensor.loadArchive("weights.safetensors")
    return archive.tensors["model.weight"]
  })
```

Properties:

- Lazy save entries are compiled and materialized together as one multi-root
  request.
- Loaded tensors are concrete runtime-owned handles.
- Metadata values are strings.
- `"__metadata__"` is reserved as a tensor name.
- I/O runs natively and is interruptible.
- Placement and dtype support are enforced by the selected backend.
- Metal rejects F64 archives rather than loading them on CPU.

Models provide named parameter persistence:

```ts
import { Model } from "@effect-torch/core"
import { Effect } from "effect"

const roundTrip = (model: Model.Model, params: Model.Params) =>
  Effect.gen(function*() {
    yield* Model.save(model, params, "model.safetensors")
    return yield* Model.load(model, "model.safetensors")
  })
```

Trainer checkpoints extend the same format with optimizer, step, and optional
sampler state.

## Tokenizers

`@effect-torch/tokenizers` wraps the Rust
[`tokenizers`](https://github.com/huggingface/tokenizers) crate through its own
N-API addon. It is not part of the tensor Runtime service.

```ts
import { Tensor } from "@effect-torch/core"
import * as Tokenizer from "@effect-torch/tokenizers"
import { Effect } from "effect"

const tokenize = Effect.gen(function*() {
  const tokenizer = yield* Tokenizer.fromFile(
    "tokenizer.json",
    Tokenizer.strictConfig
  )

  const encoded = yield* tokenizer.encode("Effect meets tensors")
  return yield* Tensor.fromTypedArray(
    encoded.data,
    [1, encoded.shape[0]]
  )
})
```

Loading supports HuggingFace-compatible `tokenizer.json` files or in-memory JSON.
Configuration makes padding, truncation, and special-token parsing explicit.

The package provides:

- Single, batched, and concatenated encoding.
- Single and batched decoding.
- Token-to-ID and ID-to-token lookup.
- Longest and fixed-length padding policies.
- Explicit truncation policies.
- BPE, WordPiece, Unigram, and WordLevel training.
- File-streamed or in-memory corpora.
- Effect-based training progress callbacks.
- Saving trained tokenizers as `tokenizer.json`.

The native tokenizer supports concurrent use. Token IDs are host-owned U32 data
until explicitly imported into a tensor runtime.

## Errors and Cancellation

The public error hierarchy includes:

| Error                        | Scope                                                   |
| ---------------------------- | ------------------------------------------------------- |
| `Runtime.BackendError`       | Structured backend operation and ownership failures     |
| `Tensor.TensorError`         | Graph, evaluation, readback, and serialization failures |
| `Gradient.GradError`         | Autodiff contract failures                              |
| `Model.ModelError`           | Model construction, arity, and checkpoint failures      |
| `Model.InferenceError`       | Inference transform and generation-session failures     |
| `Checkpoint.CheckpointError` | Invalid or incomplete trainer checkpoints               |
| `Sampler.SamplerError`       | Invalid sampler configuration or state                  |
| `Tokenizer.TokenizerError`   | Tokenizer load, train, encode, and decode failures      |

`Runtime.BackendError` records a reason, backend, operation, phase, message,
and optional details. The reason vocabulary can represent unsupported dtype or
placement, invalid and foreign handles, compilation failures, execution
failures, cancellation, I/O, and other backend states. Adapters classify
ownership and selected extension failures precisely; generic native graph and
kernel failures currently map to `execution-failed`.

Tensor-level errors preserve the originating backend error. Applications can
handle failures through normal Effect combinators without parsing panic output
or native exception strings.

Interrupting an Effect running a cancellable backend operation requests native
cancellation. Cancellation remains fiber interruption rather than being
converted to an ordinary typed failure.

## Native Distribution

### Native Artifact Platforms

| Package                              | macOS arm64 | macOS x64 | Linux arm64 GNU | Linux arm64 musl | Linux x64 GNU | Linux x64 musl |
| ------------------------------------ | ----------- | --------- | --------------- | ---------------- | ------------- | -------------- |
| `@effect-torch/backend-cpu`          | Yes         | Yes       | Yes             | Yes              | Yes           | Yes            |
| `@effect-torch/backend-apple-native` | Yes         | Yes       | No              | No               | No            | No             |
| `@effect-torch/tokenizers`           | Yes         | Yes       | Yes             | Yes              | Yes           | Yes            |

The complete build contains 14 `.node` artifacts:

- Six CPU binaries.
- Two Apple Metal binaries.
- Six tokenizer binaries.

There is currently no Windows package.

The Apple package's JavaScript entrypoint is installable on every platform so
applications can evaluate `isAvailable` safely. Its runtime and native
artifacts remain Darwin-only.

### Loader Selection

CPU and tokenizer loaders select one exact package-local binary from
`process.platform`, `process.arch`, and, on Linux, the presence of glibc in
`process.report`. The Apple loader performs the same platform and architecture
selection lazily when `isAvailable` or `makeRuntime` first requests native code.

There is no postinstall download, fallback search path, or dynamic CPU fallback.
GNU and musl binaries are shipped together in the Linux-capable packages.

### Static Linkage

Each backend addon is a self-contained `cdylib`. Shared Rust graph, compiler,
autodiff, runtime, and N-API helper crates are statically linked into the addon.

The package boundary is Node-API, not a Rust dynamic-plugin ABI. CPU and Metal
can evolve independently without passing Rust trait objects or allocations
across addon boundaries.

### Build Matrix

`pnpm build` must run on macOS because the release includes Apple Metal
artifacts that require Xcode, the macOS SDK, and Apple's linker tools. That one
command builds Darwin targets locally and cross-compiles the Linux targets with
Zig:

| Artifact suffix    | Build target                     |
| ------------------ | -------------------------------- |
| `darwin-arm64`     | `aarch64-apple-darwin`           |
| `darwin-x64`       | `x86_64-apple-darwin`            |
| `linux-arm64-gnu`  | `aarch64-unknown-linux-gnu.2.17` |
| `linux-arm64-musl` | `aarch64-unknown-linux-musl`     |
| `linux-x64-gnu`    | `x86_64-unknown-linux-gnu.2.17`  |
| `linux-x64-musl`   | `x86_64-unknown-linux-musl`      |

Darwin uses Cargo and Apple's system SDK. Linux uses `cargo-zigbuild`. Darwin
artifacts target macOS 11 or newer; GNU artifacts are checked against glibc
2.17. Musl addons are dynamically linked against musl libc.

### Package Verification

The build verifies:

- Exact package platform policy, `files` whitelist, and binary-name metadata.
- Exact native artifact sets with no missing or extra binaries.
- Artifact architecture.
- macOS deployment target and install ID.
- Absence of Nix, user-home, and Homebrew paths in Darwin linkage.
- Presence of Metal.framework in Apple artifacts.
- Absence of Metal.framework in CPU artifacts.
- Maximum glibc symbol version for GNU artifacts.
- Musl libc references and absence of glibc symbols in musl artifacts.
- Native files included by `npm pack --dry-run`.

`pnpm verify:native-packages` performs metadata and loader checks without
requiring assembled artifacts. Full artifact verification runs as part of the
release matrix build.

## Development

### Reproducible Environment

The repository provides a Nix flake and direnv configuration for macOS and
Linux development shells. The shell includes Node.js 22, Corepack, Rustup, Zig,
`cargo-zigbuild`, dprint, CMake, and pkg-config.

```bash
direnv allow
pnpm install
```

Without direnv:

```bash
nix develop
pnpm install
```

Rust is pinned in `rust-toolchain.toml`, including rustfmt, rust-analyzer, and
the complete standard-library target set.

Outside Nix, install Node, pnpm, the pinned Rust toolchain, Zig, and
`cargo-zigbuild`. Darwin builds also require Xcode Command Line Tools.

### Native Development Builds

Workspace TypeScript resolves directly to package source, but native packages
load addons from their own `dist/internal` directories. Build a host addon
before running code against a fresh checkout.

```bash
pnpm --filter @effect-torch/backend-cpu build:debug
pnpm --filter @effect-torch/backend-apple-native build:debug
pnpm --filter @effect-torch/tokenizers build:debug
```

The Apple command is macOS-only. On Linux, build CPU and tokenizers.

Host debug builds preserve any other already-assembled matrix artifacts. A
host release build is available through `scripts/build-native.mjs --host
--profile release` from a native package directory.

### Quality Commands

```bash
pnpm test
pnpm typecheck
pnpm lint

cargo check --workspace --features napi-addon
cargo test --workspace --features napi-addon
cargo fmt --all -- --check
```

`pnpm test` runs core, CPU backend, and Apple backend Vitest suites. Core tests
cover tensor operations, autodiff, compilation, fusion, models, optimizers,
training, memory ownership, safetensors, tokenizers, attention, inference, and
checkpointing. Backend-neutral suites run on CPU and, when available, Metal.

The `napi-addon` feature is important for Rust checks because each backend's
Node-facing module is feature-gated when the crate is used as a normal `rlib`.
VS Code configuration enables this feature for rust-analyzer.

### Release Build

```bash
pnpm build
```

The root build:

1. Builds the complete native release matrix.
2. Builds TypeScript for CPU, Apple, and tokenizers.
3. Verifies native artifacts and npm tarball contents.
4. Builds `@effect-torch/core`.

The build does not implicitly run tests, typechecking, lint, or Rust tests.
Run the quality commands separately before a release build.

The complete matrix is assembled on macOS because it includes native Apple
artifacts; Linux outputs are cross-compiled there with Zig. Linux remains a
supported CPU/tokenizer build and test host, while Metal tests require macOS.

### Examples

```bash
pnpm --filter @effect-torch/examples xor
pnpm --filter @effect-torch/examples nano-gpt # macOS
```

The examples include:

- XOR training on the CPU backend.
- Nano-GPT with tokenizer training, causal attention, RoPE, compiled training,
  paged KV-cache inference, and generation.
- FineWeb preparation from Parquet into flat token bins.
- FineWeb compiled AdamW training with restorable sampling and checkpoints.
- Mixed-BF16 full-epoch training.
- Checkpoint export and streaming generation.

### Benchmarks

```bash
pnpm bench
pnpm bench:compile
pnpm bench:mlx

cargo bench -p effect-torch-compiler --bench pipeline
cargo bench -p effect-torch-compiler --bench pipeline -- --workload stress
```

The benchmark package contains configurable matmul, shape, compiled-program,
native cold-compile/warm-structural-cache, attention, and optional MLX
comparisons. `N`, `ITERS`, and `METAL_ONLY` control the default matmul
benchmark. `pnpm bench:compile -- --help` lists backend, workload, size,
iteration, and optimization controls.

The Rust compiler benchmark measures `GraphIndex` plus side-table optimization
separately from graph construction and reports deterministic structural work.
Its `stress` workload runs 50,000- and 100,000-node graphs on a 256 KiB thread
stack; it does not include lowering, memory/physical planning, or pipeline
preparation.

Benchmark results are environment-specific and are intentionally not embedded
as fixed claims in this README. `pnpm bench` runs CPU measurements on Linux and
adds Metal when available on macOS. The MLX comparison is macOS-only.

## Repository Layout

```text
packages/
  core/                    Backend-neutral TypeScript API and tests
  backend-cpu/             CPU package, adapter, loader, and artifacts
  backend-apple-native/    Apple package, adapter, loader, and artifacts
  tokenizers/              TypeScript tokenizer API and Rust addon
  examples/                Runnable applications
  bench/                   Benchmarks

crates/
  runtime/                 IDs, signatures, memory, diagnostics, and ownership contracts
  graph/                   Nongeneric semantic graph and leaf contracts
  compiler/                Requests, graph index, regions, lowering tables, and memory planning
  autodiff/                Semantic graph differentiation and transforms
  napi/                    Backend-neutral Node-API helpers
  runtime-cpu/             Typed CPU executable runtime and CPU-owned addon
  runtime-metal/           Typed Metal executable runtime and Metal-owned addon

scripts/
  build-native.mjs         Host and release-matrix native builder
  native-packages.mjs      Package and target manifest
  verify-native-packages.mjs
                            Metadata, ABI, linkage, and tarball verifier
  clean-native-declarations.mjs
                            Publish-output cleanup

docs/rfcs/                 Architecture and feature design records
```

The pnpm workspace contains six packages. The Cargo workspace contains the
seven shared/backend crates plus the tokenizer Rust package.

## Current Constraints

The architecture is designed for independently packaged runtimes, but the
currently shipped implementations are CPU and Apple Metal.

- There is no CUDA, PJRT, remote, WebGPU, or Windows backend today.
- There is no implicit backend selection or cross-device tensor transfer.
- Apple Metal is macOS-only and never falls back to CPU.
- Metal does not support F64 or rank-2 linalg operations.
- CPU F16 and BF16 matmul is not currently implemented.
- Mixed-BF16 training is currently Metal-only.
- INT8 is currently a KV-cache storage tier, not a general tensor dtype.
- Some optimized attention and loss backward paths are first-order only.
- Full release-matrix assembly runs on macOS because Apple artifacts require the
  macOS SDK; Linux artifacts are cross-compiled with Zig.
- Native release publication, signing, notarization, and registry automation
  are not currently encoded in the repository.

The code is the source of truth for current behavior. RFCs describe design
intent and historical decisions; older RFC details may be superseded by the
implementation.

## Design Documents

The main architecture records are:

- [RFC 0021: Compiler Pipeline Refactor](docs/rfcs/0021-compiler-pipeline-refactor.md)
- [RFC 0020: Invocation Ownership](docs/rfcs/0020-invocation-ownership.md)
- [RFC 0019: Executable Compilation](docs/rfcs/0019-executable-compilation.md)
- [RFC 0017: Multi-Backend Runtime](docs/rfcs/0017-multi-backend-runtime.md)
- [RFC 0002: Autodiff](docs/rfcs/0002-autodiff.md)
- [RFC 0003: Memory Management](docs/rfcs/0003-memory-management.md)
- [RFC 0004: Optimizers](docs/rfcs/0004-optimizers.md)
- [RFC 0005: Models](docs/rfcs/0005-models.md)
- [RFC 0007: Kernel Fusion](docs/rfcs/0007-kernel-fusion.md)
- [RFC 0008: Compilation](docs/rfcs/0008-compilation.md)
- [RFC 0009: Tokenizers](docs/rfcs/0009-tokenizers.md)
- [RFC 0010: Inference](docs/rfcs/0010-inference.md)
- [RFC 0012: Dtype System](docs/rfcs/0012-dtype-system.md)
- [RFC 0013: Batched Decode](docs/rfcs/0013-batched-decode.md)
- [RFC 0016: Frozen Program Memory](docs/rfcs/0016-frozen-program-memory.md)

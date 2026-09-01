# effect-torch

effect-torch is a native tensor runtime and machine-learning system for
TypeScript applications built with [Effect](https://effect.website).

The public API is backend-neutral TypeScript. Rust implements the execution
engine in this repository, including the graph IR, autodiff, compilation, CPU,
Metal, and CUDA kernels, memory management, and Node-API bindings.

Implemented features include:

- A lazy semantic tensor graph with strict shape, dtype, and placement checks.
- Reverse-mode autodiff, VJP, JVP, vmap, and gradient checkpointing.
- Reusable executable compilation with bounded caches partitioned by runtime.
- Independent native CPU and Apple Metal backends, plus an experimental CUDA
  backend for x64 Linux.
- Pure model, optimizer, trainer, checkpoint, and learning-rate APIs.
- Compiled training and paged KV-cache inference with batched and speculative
  decode.
- Native safetensors and GGUF loading, including GGML K-quantized inference
  weights.
- Chat-template rendering, streaming chat generation, and a standalone
  tokenizer package.
- Structured Effect errors, interruption, cancellation, and explicit resource
  release.

Repository setup and contributor workflows are documented in
[DEVELOPMENT.md](DEVELOPMENT.md).

## Contents

- [Packages](#packages)
- [Quick start](#quick-start)
- [Programming model](#programming-model)
- [Backend capabilities](#backend-capabilities)
- [Public API](#public-api)
- [Compilation](#compilation)
- [Models and training](#models-and-training)
- [Compiled inference](#compiled-inference)
- [GGUF models](#gguf-models)
- [Chat](#chat)
- [Safetensors](#safetensors)
- [Tokenizers](#tokenizers)
- [Errors and cancellation](#errors-and-cancellation)
- [Native distribution](#native-distribution)
- [Current constraints](#current-constraints)
- [Development](DEVELOPMENT.md)

## Packages

| Package                              | Responsibility                                                                    |
| ------------------------------------ | --------------------------------------------------------------------------------- |
| `@effect-torch/core`                 | Backend-neutral tensors, compilation, models, training, GGUF, chat, and inference |
| `@effect-torch/backend-cpu`          | CPU Runtime Layer and CPU-owned native addon                                      |
| `@effect-torch/backend-apple-native` | Apple Metal Runtime Layer and Metal-owned native addon                            |
| `@effect-torch/backend-cuda`         | Experimental CUDA Runtime Layer and Linux x64 native addon                        |
| `@effect-torch/tokenizers`           | Native tokenizer loading, encoding, decoding, and training                        |

`@effect-torch/core` has no dependency on a concrete backend. Applications
select a backend by providing its runtime Layer to the Effect program.

The tokenizer package is independent of core. It returns host-owned
`Uint32Array` token IDs, which applications explicitly import into the selected
tensor runtime.

## Quick start

The packages listed above are available only from this workspace; npm does not
publish them yet. The examples use the workspace names, which are also the
intended distribution names.

The repository pins `effect@4.0.0-beta.101`. Projects using these packages must
use the compatible Effect 4 beta release line. The unqualified `effect` package
on npm is Effect 3 and is not API-compatible.

To run the examples from a checkout, follow the setup and native build steps in
[DEVELOPMENT.md](DEVELOPMENT.md).

A minimal CPU application looks like this:

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

To use Metal on macOS, provide the Apple backend instead:

```ts
import * as BackendApple from "@effect-torch/backend-apple-native"
import { Tensor } from "@effect-torch/core"
import { Effect } from "effect"

const program = Effect.gen(function*() {
  const tensor = yield* Tensor.ones([2, 2])
  return yield* Tensor.toNumberArray(tensor)
})

const result = await Effect.runPromise(
  program.pipe(Effect.provide(BackendApple.layer()))
)

// Select another enumerated Metal device. Omitting `device` selects metal:0.
const secondDevice = BackendApple.layer({ device: 1 })

const reportedAvailable = await Effect.runPromise(
  BackendApple.isAvailable
)
```

You can import the Apple package entrypoint on any platform without loading the
native addon. `isAvailable` loads it on demand and returns `false` if the
platform or architecture is unsupported, an artifact is missing, or Metal
cannot create a device, command queue, or shared event.
`BackendApple.layer()` returns a Layer without loading the addon. Effect loads
it when it builds that Layer to provide the Metal runtime.

The experimental CUDA package uses the same lazy interface:

```ts
import * as BackendCuda from "@effect-torch/backend-cuda"
import { Effect } from "effect"

const cudaAvailable = await Effect.runPromise(BackendCuda.isAvailable)
const cudaLayer = BackendCuda.layer({ device: 0 })
```

Its packaged addon supports x64 Linux with glibc. Importing the entrypoint on
another platform is safe; `isAvailable` returns `false` without selecting a
CPU fallback.

## Programming model

### Runtime is an Effect service

All backend operations in an Effect program go through `Runtime.Runtime`.
Tensor values retain immutable metadata and opaque backend-owned handles, not a
reference to the service.

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

The CPU backend package exposes a constant Layer:

```ts
import { Runtime } from "@effect-torch/core"
import { Layer } from "effect"

declare const layer: Layer.Layer<Runtime.Runtime>
```

The Apple and CUDA backend packages expose `layer(options?)`. They select
`metal:0` or `cuda:0` by default and accept a zero-based `device` ordinal.

Each Layer constructs its `RuntimeService` the first time Effect builds it.
Later builds reuse the same service object, which keeps runtime identity and
native caches stable without exposing a public constructor.

The CPU package selects and loads its native addon when imported. The Apple and
CUDA packages wait until `isAvailable` runs or Effect builds their Layer.

### Lazy and concrete tensors

Tensors have two states:

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

### Compilation and materialization

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

Batching roots into one request has these effects:

- The compiler lowers and executes shared subgraphs once.
- Multiple roots observe the same draw from shared random nodes.
- The memory planner uses each intermediate's final use to set its lifetime and
  reusable workspace.
- The native executor runs outside the JavaScript event loop.
- Interrupting the Effect cancels native execution.

`Tensor.toTypedArray` and `Tensor.toNumberArray` materialize a lazy tensor when
needed, or read an existing concrete tensor.

Ordinary `compute` calls and reusable programs use the same compiler, memory
planner, and executor. `compute` obtains a transient or structurally cached
executable. `Tensor.compile`, model `execute` methods, `Trainer.make`, and the
inference APIs retain reusable executable handles.

### Resource ownership

TypeScript uses opaque immutable handles for lazy graphs, concrete tensors,
compiled programs, decode programs, KV pools, and KV sequences. Backend
adapters maintain private ownership records in `WeakMap`s.

Consequences:

- Metal cannot use CPU handles, and CPU cannot use Metal handles.
- Cleared handles fail with a typed `invalid-handle` error.
- Foreign handles fail with a typed `foreign-handle` error.
- Builds of one backend's Layer share handle ownership and stable runtime
  identity.
- Each compiled function, model, and trainer owns a TypeScript signature cache.
- Each runtime owns a bounded structural executable cache. Its entries share
  immutable plans without retaining generated concrete bindings.
- Each inference artifact owns a fixed, eagerly compiled set of prefill and
  decode programs instead of a shape-keyed cache.
- An executable owns immutable typed instructions, memory and physical plans,
  pipelines, constants, signatures, and diagnostics. It does not own a
  permanent invocation workspace.
- Calls lease runtime-owned workspace and provisional output storage.
  Successful outputs take ownership of their backing storage.
- The runtime never transfers tensors between devices implicitly.

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

Native finalizers provide a GC fallback. CPU external buffers contribute to
Node's external-memory accounting. Backend diagnostics report the current
external byte count when available.

### Dtypes

Every tensor handle reports one logical dtype:

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

The host transfer types are fixed by that logical dtype:

| Dtype  | Meaning                 | `Tensor.fromTypedArray` | `Tensor.toTypedArray` |
| ------ | ----------------------- | ----------------------- | --------------------- |
| `f32`  | 32-bit float            | `Float32Array`          | `Float32Array`        |
| `f64`  | 64-bit float            | `Float64Array`          | `Float64Array`        |
| `f16`  | IEEE 16-bit float       | `Float16Array`          | `Float32Array`        |
| `bf16` | Brain floating point    | No direct host type     | `Float32Array`        |
| `i64`  | Signed 64-bit integer   | `BigInt64Array`         | `BigInt64Array`       |
| `u8`   | Unsigned 8-bit integer  | `Uint8Array`            | `Uint8Array`          |
| `u32`  | Unsigned 32-bit integer | `Uint32Array`           | `Uint32Array`         |

`fromTypedArray` snapshots the supplied view and infers the dtype shown
above. Construct BF16 values with a dtype option or `Tensor.cast`. Readback
widens F16 and BF16 values rather than exposing their storage bits.
`Tensor.toNumberArray` accepts every dtype except I64, whose full range cannot
be represented by JavaScript numbers.

Constructors default to F32 unless their contract says otherwise. Random
constructors accept floating dtypes, `linspace` accepts F32 or F64, and
`constantLike`, `zerosLike`, `onesLike`, and `fullLike` preserve the source
tensor's dtype and placement.

Most operations do not promote mixed dtypes. Use `Tensor.cast` to request an
explicit conversion, subject to backend support. One narrow exception applies
to the binary elementwise and comparison functions: when exactly one operand
is a 0-dimensional float and the other is a non-scalar float of another
dtype, the scalar is coerced to the non-scalar dtype. Arithmetic results keep
that dtype; comparisons return U8. Other operations still enforce their own
dtype contracts.

A runtime's `capabilities.dtypes` array means that it can represent those
logical dtypes. It does not promise that every operation accepts every listed
dtype. For example, CPU stores F16 and BF16 but rejects half-precision matmul,
and Metal rejects F64 when the graph node is constructed.

#### Encoded and state storage

GGML `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, and `Q6_K` are storage encodings, not
members of `DType`. A quantized GGUF weight remains a logical F32 tensor. Its
`storage` metadata records the encoding, a packed physical shape, and physical
dtype U8. Dedicated operations such as `Tensor.embedding` and
`Tensor.linearRows` can consume supported encoded weights. Ordinary dense
operations and `Tensor.toTypedArray` may reject the packed layout.

Likewise, `kvDtype: "int8"` selects quantized KV-cache storage for compiled
inference. `"int8"` is not a tensor dtype. The cache stores U8 payloads with
per-token, per-head F32 scales and widens values for attention math.

## Backend capabilities

| Capability                   | CPU                                | Apple Metal                        |
| ---------------------------- | ---------------------------------- | ---------------------------------- |
| Platforms                    | macOS and Linux                    | macOS                              |
| Architectures                | arm64 and x64                      | arm64 and x64                      |
| Advertised tensor dtypes     | F32, F64, F16, BF16, I64, U8, U32  | F32, F16, BF16, I64, U8, U32       |
| F32 matmul                   | Yes                                | Yes                                |
| F64                          | Storage, math, matmul, and linalg  | Unsupported                        |
| F16/BF16 storage             | Yes                                | Yes                                |
| F16/BF16 matmul              | No                                 | Yes                                |
| Graph compilation            | Yes                                | Yes                                |
| Autodiff                     | Yes                                | Yes                                |
| Elementwise/reduction fusion | F32, F64                           | F32, BF16                          |
| Scaled dot-product attention | Composed backend path              | Native flash path for F32 and BF16 |
| `inverse`, `det`, `solve`    | Yes                                | Explicitly rejected                |
| Mixed-BF16 training          | Not advertised                     | Yes                                |
| Paged KV cache               | F32, F16, BF16, INT8 storage tiers | F32, F16, BF16, INT8 storage tiers |
| Safetensors path I/O         | Yes                                | Yes, with Metal dtype validation   |

`@effect-torch/backend-cuda` is an experimental third backend. Its packaged
addon is Linux x64 GNU only and requires an NVIDIA driver usable through the
CUDA 12.9 NVRTC path. It exposes the same lazy `isAvailable` and
`layer({ device? })` interface as Metal and advertises all seven logical
dtypes. Its current operator coverage is narrower than the CPU and Metal
table above; the advertised dtype list is not a full operation matrix.

Unsupported placement or dtype requests fail; the runtime never moves the graph
to another backend.

## Public API

`@effect-torch/core` exports namespaces rather than one flat symbol list:

| Namespace      | Responsibility                                                    |
| -------------- | ----------------------------------------------------------------- |
| `Chat`         | Chat-template generation, segmented parsing, and streaming events |
| `Runtime`      | Backend contract, handles, capabilities, errors, and service tag  |
| `Tensor`       | Tensor graph construction, evaluation, compilation, and I/O       |
| `Gradient`     | Autodiff transforms                                               |
| `Gguf`         | Validated GGUF model and parameter-artifact loading               |
| `Loss`         | Regression and classification losses                              |
| `Model`        | Layers, composition, execution, and compiled inference            |
| `Optimizer`    | SGD, Adam, AdamW, clipping, and full-step execution               |
| `LearningRate` | Constant, exponential, stepwise, cosine, and warmup schedules     |
| `Trainer`      | Compiled and reference training loops                             |
| `Checkpoint`   | Trainer and sampler checkpoint persistence                        |
| `Sampler`      | Restorable shuffled token-window sampling                         |
| `Speculation`  | Autoregressive, history-lookup, and parallel-block proposers      |

Built-in architectures live at explicit subpath exports:

| Entry point                    | Export        | Purpose                               |
| ------------------------------ | ------------- | ------------------------------------- |
| `@effect-torch/core/models`    | `MuseGlimmer` | Muse-Glimmer model and GGUF loader    |
| `@effect-torch/core/proposers` | `DFlash`      | DFlash proposer graph and GGUF loader |

### Tensor constructors

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

### Elementwise and activation operations

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

### Shape and indexing

```text
reshape            flatten             squeeze             unsqueeze
transpose          slice               split               chunk
concat             stack               broadcastTo         tile
pad                take                gather              scatterAdd
flip               oneHot              embedding           triu
tril               trace
```

`take`, `gather`, and `embedding` accept I64 or U32 index tensors.
`scatterAdd` accumulates duplicate indexes. Indexing gradients use it.

### Neural-network and linear-algebra primitives

```text
matmul                         dot
linear                         linearRows
layerNorm                      rmsNorm
positionEmbedding              rotaryEmbedding
softmax                        logSoftmax
scaledDotProductAttention      dropout
crossEntropy

conv1d                         conv2d
convTranspose1d                convTranspose2d
shortConv1d                    kdaChunk
maxPool2d                      avgPool2d

inverse                        det
solve
```

Linalg placement constraints are listed in the backend capability table.

### Evaluation, resources, and I/O

```text
compute             sample
toTypedArray        toNumberArray
clear               clearAll
clearScoped         clearAllScoped
save                loadArchive          load
isLazyTensor        isTensor
shape               dtype                device
```

Readback materializes lazy inputs and borrows concrete inputs. `sample` selects
one token from a concrete rank-one floating logits tensor. The scoped clear
helpers register caller-owned handles for cleanup when an Effect scope closes;
library code should still release temporary handles at the operation boundary.

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

Adjoints are semantic graph nodes, so `grad` can compute higher derivatives when
every operation in the path is differentiable. Optimized cross-entropy and
scaled-dot-product-attention backward paths are first-order only.

## Compilation

### Generic compiled functions

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

- `Tensor.compile` creates the TypeScript wrapper and its cache without tracing.
- The first call traces the function against placeholder inputs after receiving
  actual input exemplars.
- That call obtains `Runtime.Runtime`. The compiled function then specializes
  for the inputs' backend, shape, placement, and dtype.
- The backend prepares and lowers the semantic roots into a typed executable.
- Later calls bind new inputs and execute that fixed plan.
- Signatures include runtime identity, placement, shape, and dtype.
- Each compiled function owns its cache. Runtime identity partitions entries
  without duplicating entries for the same backend.
- The cache holds 32 entries by default and evicts the least recently used entry
  when full.
- Concurrent misses for the same signature share one trace.
- Failed traces are not cached.
- Random nodes draw fresh values on every program execution.
- Materializing a placeholder while tracing fails.

`compiled.stats` reports the number of cached programs and total trace attempts.
`compiled.clear` drops cache references without resetting the trace count. Native
programs are destroyed when normal handle reachability and finalization permit.

### Lower-level program API

The Tensor namespace also exposes the primitives used by trainers and models:

```text
makeProgramCache       cachedProgram          signatureOf
makeInput              makeScalarInput        freezeProgram
runProgram

makeKvPool             makeKvSequence         kvPrefillMatch
kvSequenceCursor       releaseKvSequence
compileDecodeProgram   runDecodeProgram       runDecodeProgramSampled
runBatchedDecodeProgram
runBatchedDecodeProgramSampled
```

Most applications should use `Tensor.compile`, a model's `execute` method,
`Trainer.make`, or `Model.inference` instead.

## Models and training

### Models

A `Model.Model` defines a functional model with:

- Ordered parameter names.
- An Effect that initializes a flat parameter array.
- A lazy `forward` graph builder.
- A compiled `execute` path.
- Compilation cache statistics and explicit cache clearing.

Models do not mutate learned parameters or running tensor state, and they have
no model-specific backward method. After the first `execute` call, a model
memoizes its compiled execution function. `stats` and `clear` expose that cache.

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

    const params = yield* Tensor.compute(yield* Model.initialize(model))
    const lazyOutput = yield* model.forward(params, input)
    const concreteOutput = yield* model.execute(params, input)
    return { lazyOutput, concreteOutput }
  })
```

Use `forward` for composition and differentiation. Use `execute` for repeated
materialized evaluation. It creates the compiled function on first use and
reuses it.

Multi-head attention uses fused QKV parameters named:

```text
<name>.qkv.weight
<name>.qkv.bias
<name>.wo.weight
<name>.wo.bias
```

### Optimizers

The optimizer API includes SGD, Adam, and AdamW. Optimizers build update graphs
from parameter and state tensors without mutating them in place.

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

### Learning-rate schedules

```text
constant
exponential
stepwise
cosine
withWarmup
```

Schedules are plain `(step: number) => number` functions. The trainer converts
the result to a runtime scalar for the compiled update graph.

### Compiled trainer

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

    return yield* trainer.train(yield* Model.initialize(model))
  })
```

Training data can be fixed or produced by an effectful function on each step.
Trainer callbacks receive a 1-based step, loss, and elapsed duration. The
learning-rate schedule receives a 0-based step. The trainer always executes at
least one step and runs `onStep` before checking the stop policy.

The compiled trainer releases parameter and state generations that it owns once
their replacements commit.

### Mixed BF16

Trainer precision is `"f32"` or `"mixedBf16"`.

Mixed BF16 keeps master parameters and optimizer state in F32. Before the
forward pass, it casts master parameters to BF16, runs forward and backward in
BF16, and propagates gradients through the casts to the F32 update. The runtime
must report the `"mixed-bf16"` feature, which Apple Metal does. The trainer
casts model parameters, not data. Floating inputs and regression
targets must use a BF16-compatible dtype; integer class targets remain valid
for classification losses such as cross-entropy. Every operation used by the
model and loss must support BF16. Dropout and `Loss.nll`, for example, do not.

### Checkpoints and samplers

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
configuration. Restoring resumes at the exact position in the saved permutation.
The sampler does not persist JavaScript RNG state, so the next reshuffle after
exhausting that permutation uses a new random event.

## Compiled inference

`Model.inference` transforms a causal attention model into compiled prefill and
decode programs backed by a paged KV cache:

```ts
import { Model, Tensor } from "@effect-torch/core"
import { Effect } from "effect"

const generate = (
  model: Model.Model,
  params: Model.Params,
  promptTensor: Tensor.Any,
  maxNewTokens: number
) =>
  Effect.gen(function*() {
    const inference = yield* Model.inference(model, params, {
      maxTokens: 8192,
      blockSize: 16,
      prefillChunks: [16],
      attentionWindow: 256,
      kvDtype: "bf16",
      batchSize: 8,
      sampling: { temperature: 0, seed: 0 }
    })

    const generation = yield* inference.generation()
    const [first] = yield* generation.add([{
      prompt: promptTensor,
      maxTokens: maxNewTokens
    }])
    const tokens = [...first!.tokens]
    let page = first!

    while (page.stopReason === undefined) {
      ;[page] = yield* generation.step([{ seq: page.seq }])
      tokens.push(...page!.tokens)
    }

    yield* page.seq.finish()
    return tokens
  })
```

The inference transform:

- Verifies parameter arity and materializes parameters once.
- Traces the model's existing `forward` graph.
- Rewrites causal attention into paged KV-cache operations.
- Rewrites supported position operations to cursor-aware forms.
- Allocates one shared block pool.
- Compiles fixed-shape prefill and one fixed-width batched decode program.
- Rejects models without cacheable causal attention.

Generation sessions support:

- Chunked prompt prefill.
- A sampled batched token-page API. Batch size one uses the same API.
- Stable physical lanes with explicit inactive slots and ragged prefill lengths.
- Content-addressed whole-block prefix reuse.
- Explicit sequence finish and session close.
- Sliding-window attention.
- RoPE with bounded active context and unbounded sequence cursors.

Speculative decoding uses the same token-page API. `Speculation` constructs
autoregressive draft-model, deterministic history-lookup, and replayable
parallel-block proposers. `Model.inference` accepts one proposer and currently
uses a fixed proposal width:

```ts
import { Model, Speculation } from "@effect-torch/core"

const proposer = Speculation.autoregressive(draftModel, draftParams, {
  vocabulary: 32_000,
  maxDraftTokens: 4
})

const inference = yield* Model.inference(targetModel, targetParams, {
  maxTokens: 8192,
  prefillChunks: [32, 64, 128, 256],
  speculation: { proposer, maxDraftTokens: 4 },
  sampling: { temperature: 0.8, topK: 40, topP: 0.95, seed: 7 }
})
```

One native round proposes tokens, verifies them, performs exact rejection and
residual sampling, and commits target state plus any proposer state. A returned
page may contain multiple tokens. Consumers must append every token in each
page. Speculation does not currently combine with `attentionWindow` or adaptive
proposal scheduling.

`kvDtype: "int8"` is a KV storage tier, not a normal tensor dtype. Cached rows
are quantized with per-token, per-head scales and widened for attention math.

The low-level `Tensor` namespace also exposes KV pools, sequences, cursor
queries, prefix matching, decode compilation, and direct decode execution.

## GGUF models

`Gguf.loadModel` inspects a GGUF v3 file, validates its exact architecture and
tensor catalog against a `Gguf.ModelDefinition`, then loads the parameters on
the selected runtime. `Gguf.loadParameters` provides the same catalog and
ownership checks for target-coupled artifacts such as speculative proposers.

The built-in Muse-Glimmer loader wraps the generic model path:

```ts
import { Model, Tensor } from "@effect-torch/core"
import { MuseGlimmer } from "@effect-torch/core/models"
import { Effect } from "effect"

const prepare = Effect.gen(function*() {
  const loaded = yield* MuseGlimmer.loadGGUF("model.gguf")
  const program = yield* Model.inference(loaded.model, loaded.params, {
    maxTokens: 4096,
    blockSize: 16,
    prefillChunks: [32, 64, 128, 256],
    kvDtype: "f16",
    batchSize: 1
  })

  // Model.inference retains its own immutable parameter generation.
  yield* Tensor.clearAll(loaded.params)
  return { program, metadata: loaded.metadata }
})
```

The loader accepts dense F32 and `Q2_K` through `Q6_K` tensor payloads. Loaded
parameters are caller-owned concrete handles. `Model.inference` retains its
parameter generation independently, so clear the loader's handles after it
succeeds. The `DFlash.loadGGUF` export from `@effect-torch/core/proposers`
similarly returns caller-owned parameters plus a ready `artifact` for
`Model.inference`.

## Chat

`Chat.stream` joins a compiled inference program with a compatible tokenizer
and caller-supplied chat template. It renders and encodes the prompt once,
prefills the model, samples tokens, and emits ordered stream events:

```ts
import { Chat } from "@effect-torch/core"
import type { Model } from "@effect-torch/core"
import * as Tokenizers from "@effect-torch/tokenizers"
import { Effect, Stream } from "effect"

const chat = (program: Model.InferenceProgram, template: string, eosTokenId: number) =>
  Effect.gen(function*() {
    const tokenizer = yield* Tokenizers.fromFile("tokenizer.json", {
      ...Tokenizers.strictConfig,
      specialTokens: "Always"
    })

    yield* Chat.stream({
      program,
      tokenizer,
      template,
      messages: [{ role: "user", content: "Explain Effect in one sentence." }],
      maxTokens: 128,
      sampling: { temperature: 0.7, topP: 0.95, seed: 7 },
      controls: false,
      stopTokens: [eosTokenId]
    }).pipe(
      Stream.runForEach((event) =>
        event._tag === "delta" ? Effect.sync(() => process.stdout.write(event.text)) : Effect.void
      )
    )
  })
```

A successful stream starts with `prefill` and ends with `done`. Between them,
`start`, `delta`, and `end` events describe parsed response segments. Passing
`controls: false` produces one unsegmented assistant response; the default
controls parse models that use Chat's segmented control-token protocol.
Standard sampling stays in the native generation session. A custom sampler
function receives each logits row as a host typed array instead.

## Safetensors

The bundled runtime backends expose direct path-based safetensors I/O through a
Runtime extension:

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

- `Tensor.save` compiles and materializes lazy entries together in one multi-root
  request.
- Loaded tensors are concrete runtime-owned handles.
- Metadata values are strings.
- `"__metadata__"` is reserved as a tensor name.
- I/O runs natively and is interruptible.
- The selected backend validates placement and dtype support.
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

Loaders accept HuggingFace-compatible `tokenizer.json` files or in-memory JSON.
Configuration specifies padding, truncation, and special-token parsing.

The package provides:

- Single, batched, and concatenated encoding.
- Single and batched decoding.
- Stateful autoregressive decoding.
- Token-to-ID and ID-to-token lookup.
- Caller-supplied MiniJinja chat-template rendering.
- Longest and fixed-length padding policies.
- Explicit truncation policies.
- BPE, WordPiece, Unigram, and WordLevel training.
- File-streamed or in-memory corpora.
- Effect-based training progress callbacks.
- Saving trained tokenizers as `tokenizer.json`.

The native tokenizer supports concurrent use. Token IDs are host-owned U32 data
until explicitly imported into a tensor runtime.

## Errors and cancellation

The public error hierarchy includes:

| Error                        | Scope                                                   |
| ---------------------------- | ------------------------------------------------------- |
| `Chat.ChatError`             | Chat validation, sampling, and protocol failures        |
| `Runtime.BackendError`       | Structured backend operation and ownership failures     |
| `Tensor.TensorError`         | Graph, evaluation, readback, and serialization failures |
| `Gradient.GradError`         | Autodiff contract failures                              |
| `Gguf.GgufError`             | GGUF inspection, validation, loading, and ownership     |
| `Model.ModelError`           | Model construction, arity, and checkpoint failures      |
| `Model.InferenceError`       | Inference transform and generation-session failures     |
| `Checkpoint.CheckpointError` | Invalid or incomplete trainer checkpoints               |
| `Sampler.SamplerError`       | Invalid sampler configuration or state                  |
| `Tokenizer.TokenizerError`   | Tokenizer load, train, encode, and decode failures      |

`Runtime.BackendError` records a reason, backend, operation, phase, message,
and optional details. The reason distinguishes unsupported dtypes or placements,
invalid or foreign handles, compilation and execution failures, cancellation,
I/O, and other backend failures. Adapters assign specific reasons to ownership
and selected extension errors. Other native graph and kernel failures map to
`execution-failed`.

Tensor-level errors preserve the originating backend error. Applications can
handle failures through normal Effect combinators without parsing panic output
or native exception strings.

Interrupting an Effect running a cancellable backend operation requests native
cancellation. Effect reports cancellation as fiber interruption, not as a typed
failure.

## Native distribution

### Native artifact platforms

| Package                              | macOS arm64 | macOS x64 | Linux arm64 GNU | Linux arm64 musl | Linux x64 GNU | Linux x64 musl |
| ------------------------------------ | ----------- | --------- | --------------- | ---------------- | ------------- | -------------- |
| `@effect-torch/backend-cpu`          | Yes         | Yes       | Yes             | Yes              | Yes           | Yes            |
| `@effect-torch/backend-apple-native` | Yes         | Yes       | No              | No               | No            | No             |
| `@effect-torch/backend-cuda`         | No          | No        | No              | No               | Yes           | No             |
| `@effect-torch/tokenizers`           | Yes         | Yes       | Yes             | Yes              | Yes           | Yes            |

Windows binaries are not packaged.

Applications can install and import the Apple or CUDA package on any platform
to call `isAvailable`. Metal binaries remain Darwin-only. The CUDA binary
remains Linux x64 GNU-only.

### Loader selection

CPU and tokenizer loaders select one package-local binary from
`process.platform`, `process.arch`, and the presence of glibc in
`process.report` on Linux. The Apple and CUDA loaders perform selection only
when `isAvailable` runs or Effect builds their backend Layer. CUDA accepts only
Linux x64 with glibc.

Installation does not download binaries. Loaders do not search fallback paths
or switch to CPU. CPU and tokenizers include both GNU and musl Linux binaries;
CUDA includes one GNU binary.

## Current constraints

CPU and Apple Metal are the stable consumer runtimes. CUDA is experimental.

- CUDA currently ships only for x64 Linux with glibc and requires the CUDA 12.9
  NVRTC path.
- The repository implements no PJRT, remote, WebGPU, or Windows backend.
- The runtime does not select a backend or transfer tensors between devices
  implicitly.
- Apple Metal is macOS-only and never falls back to CPU.
- Metal does not support F64 or rank-2 linalg operations.
- CPU does not implement F16 or BF16 matmul.
- Mixed-BF16 training is Metal-only.
- INT8 is a KV-cache storage tier, not a general tensor dtype.
- GGML K-quants are encoded inference-weight layouts, not general tensor dtypes.
- Some optimized attention and loss backward paths are first-order only.

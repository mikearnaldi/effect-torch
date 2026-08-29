# effect-torch

effect-torch is a native tensor runtime and machine-learning system for
TypeScript applications built with [Effect](https://effect.website).

The public API is backend-neutral TypeScript. Rust implements the execution
engine in this repository, including the graph IR, autodiff, compilation, CPU
and Metal kernels, memory management, and Node-API bindings.

Implemented features include:

- A lazy semantic tensor graph with strict shape, dtype, and placement checks.
- Reverse-mode autodiff, VJP, JVP, vmap, and gradient checkpointing.
- Reusable executable compilation with bounded caches partitioned by runtime.
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
- [Quick start](#quick-start)
- [Programming model](#programming-model)
- [Architecture](#architecture)
- [Backend capabilities](#backend-capabilities)
- [Public API](#public-api)
- [Compilation](#compilation)
- [Models and training](#models-and-training)
- [Compiled inference](#compiled-inference)
- [Safetensors](#safetensors)
- [Tokenizers](#tokenizers)
- [Errors and cancellation](#errors-and-cancellation)
- [Native distribution](#native-distribution)
- [Development](#development)
- [Repository layout](#repository-layout)
- [Current constraints](#current-constraints)
- [Design documents](#design-documents)

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
select a backend by providing its runtime Layer to the Effect program.

The tokenizer package is independent of core. It returns host-owned
`Uint32Array` token IDs, which applications explicitly import into the selected
tensor runtime.

## Quick start

These scoped packages are available only from this workspace; npm does not
publish them yet. The examples use the workspace names, which are also the
intended distribution names.

The repository pins `effect@4.0.0-beta.101`. Projects using these packages must
use the compatible Effect 4 beta release line. The unqualified `effect` package
on npm is Effect 3 and is not API-compatible.

From a repository checkout, prepare the environment and a host CPU addon:

```bash
direnv allow
pnpm install
pnpm --filter @effect-torch/backend-cpu build:debug
```

Manage a disposable Blackwell CUDA devbox from the default Nix shell:

```bash
nix develop
cp .cuda-devbox.env.example .cuda-devbox.env
runpodctl doctor
./scripts/cuda-devbox.sh template
./scripts/cuda-devbox.sh create
./scripts/cuda-devbox.sh bootstrap
# Work on the pod, then stop billing when finished.
./scripts/cuda-devbox.sh destroy
```

The `CUDA devbox image` GitHub workflow publishes
`ghcr.io/mikearnaldi/effect-torch:cuda-devbox` when its dependency inputs change
on `main`. The GHCR package must be public before RunPod can pull it. Run
`template` after the first image build. It resolves the image tag to an immutable
digest, creates or updates a RunPod template, and saves the template ID in the
ignored `.cuda-devbox.env`. Later image builds require another `template` run to
move that RunPod template to the new digest.

The image extends RunPod's pinned Ubuntu base. It contains Determinate Nix, the
`.#cuda` closure, the Rust toolchain, and warm pnpm and Cargo caches. `bootstrap`
still reconciles changed lockfiles and runs the CUDA kernel, NVRTC, and cuBLASLt
checks. If no managed template exists, `create` falls back to the official RunPod
PyTorch template and `bootstrap` performs the full installation.

`create` requests one RTX PRO 6000 Blackwell and saves the pod ID, SSH address,
and port. Change the GPU, cloud, disk size, or other creation settings in
`.cuda-devbox.env`. Community Cloud requires `CUDA_DEVBOX_PUBLIC_IP=1` for
direct SSH. You can use an existing pod by setting `CUDA_DEVBOX_POD_ID`,
`CUDA_DEVBOX_ADDRESS`, and `CUDA_DEVBOX_PORT`.

`sync` uploads the non-ignored worktree. The flake pins CUDA 12.9 and compiles
the check for Blackwell `sm_120`. The default development shell remains
CUDA-free on macOS. Use `./scripts/cuda-devbox.sh ssh` for an interactive
connection or `./scripts/cuda-devbox.sh run <command>` to run a command in the
remote repository. Set `EFFECT_TORCH_CUDA_DEVBOX_CONFIG` to use a config file
outside the repository.

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

To use Metal on macOS, build and provide the Apple backend:

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

The Apple backend package exposes `layer(options?)`, which returns a Layer for
`metal:0` by default or for the requested `device`.

Each Layer constructs its `RuntimeService` the first time Effect builds it.
Later builds reuse the same service object, which keeps runtime identity and
native caches stable without exposing a public constructor.

The CPU package selects and loads its native addon when imported. The Apple
package waits until `isAvailable` runs or Effect builds its Layer.

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

Operations require matching dtypes. `Tensor.cast` performs explicit conversion.
One exception applies when a 0-dimensional floating scalar is combined with a
non-scalar floating tensor: the runtime coerces the scalar to the tensor's
dtype. This lets runtime learning-rate scalars participate in BF16 graphs
without changing tensor storage.

JavaScript has no BF16 typed array. Reading an F16 or BF16 tensor returns a
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

### TypeScript boundary

`@effect-torch/core` defines the public contract:

- `RuntimeService` defines graph construction, compilation, execution,
  autodiff, readback, release, and the backend extensions required by
  higher-level APIs.
- Tensor handles expose only immutable shape, dtype, device, and placement
  metadata.
- Higher-level APIs use the Runtime service without importing CPU or Metal
  code. Applications select a backend through an Effect Layer.

The CPU and Metal adapters translate between the public contract and their own
native addon. They validate every handle before native code receives it and map
native failures into structured `Runtime.BackendError` values.

### Rust crates

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

The compiler driver and typed tables use internal static dispatch; they do not
define a stable plugin ABI. Each Node addon statically links the shared Rust
crates it needs.

### Independent native backends

CPU and Metal use separate addons rather than selecting features in one shared
addon.
Each runtime crate owns:

- Its concrete tensor value type.
- Its typed lowered values, instructions, algorithms, and physical executor.
- Its N-API classes and functions.
- Its safetensors integration.
- Its native `cdylib` output.

The CPU addon has no Metal branches or imports. The Metal addon has no CPU
branches or imports. Apple artifacts are Darwin-only and link Metal.framework.
Release checks verify that CPU artifacts do not link Metal.framework.

`effect-torch-napi` remains an `rlib` with backend-neutral utilities only. No
Rust object crosses between separately loaded `.node` files.

### Graph and execution engine

The native graph stores nongeneric semantic operations, child relationships,
shape, dtype, placement, and leaf ownership. Compilation accepts one
`ProgramRequest` and creates one `PreparedProgram` and stack-safe `GraphIndex`.
Dense side tables store topology, consumers, roots, slots, generated leaves,
and random provenance. Shared nodes appear once, and the compiler preserves
caller root order. It collects generated leaves once for structural-cache
lookup, insertion, and binding.

Autodiff, vmap, and checkpointing construct semantic graphs before compilation.
Stateful inference uses the compiler specialization shared by CPU and Metal to
create its decode graph and state-cursor contract. The compiler indexes that
specialized graph once.

CPU and Metal lower the prepared graph into backend-typed `LoweredProgram`
values and instructions. Each instruction declares its inputs, outputs, scratch,
staging, status, state, and effects. The memory planner consumes those exact
declarations. Backend physical plans add synchronization by `InstructionId`
without duplicating tensor definitions. Execution binds the fixed plan to an
invocation frame and returns independently owned output storage.

Invocation does not traverse a semantic graph, run fusion, discover
intermediate allocations, compile pipelines, or fall back to another execution
engine. `optimize: false` uses the same typed lowering, memory planning,
ownership, and execution path with optional regions disabled.

### Compiler and fusion

The compiler records elementwise, fused-reduction, multi-output, GEMM-epilogue,
and optimizer choices as regions over `GraphIndex`. These are code-generation
side tables, not semantic `Fused*` node kinds. Multi-output selection uses a
region dependency DAG and bounded worklist. Split regions may duplicate a
prefix expression when required to preserve transitive ancestry.

`KernelExpr` contains the scalar expression evaluated by a fused instruction.
Backend lowering converts regions and uncovered semantic nodes directly into
typed CPU or Metal instructions while retaining algorithm and resource plans.
Executable compilation prepares required Metal pipelines. Compilation records
phase timings and derives instruction, memory, command, synchronization, and
region-work metrics from the resulting plans.

The compiler and runtimes implement:

- CPU elementwise and reduction fusion for F32 and F64.
- Metal elementwise and reduction fusion for F32 and BF16.
- Multi-output shared-prefix fusion and GEMM residual/GELU epilogues.
- Typed semantic-kernel instructions for layer normalization, loss, attention,
  KDA, convolution, rotary operations, and paged KV state where supported.
- Deterministic liveness-based segmented memory plans and runtime-owned
  workspace/output pools.

Executable compile options control optimization and inference-only constant
weights. `ExecutableCompileOptions` no longer contains the unused
precision option because no lowering policy uses it. Trainer mixed-BF16 uses a
separate graph and training policy.

### CPU runtime

The CPU runtime owns typed host buffers and implements tensor operations in
Rust. F32 and F64 GEMM use `matrixmultiply`; other operations use repository
kernels and composed primitives. It includes convolution, indexing, reduction,
pooling, random generation, linalg, safetensors, typed executable lowering and
execution, fusion kernels, KV-cache execution, and the CPU N-API bindings.

The adapter returns structured errors for unsupported operations. The CPU
runtime stores F16 and BF16 tensors but does not implement half-precision
matmul.

### Metal runtime

The Metal runtime calls Apple's Metal APIs through `objc2`.
It owns device buffers, command encoding, pipeline caches, generated Metal
shader source, GEMM, flash attention, convolutions, indexing, rotary kernels,
paged KV-cache operations, fusion kernels, typed lowering, physical instruction
plans, and runtime-owned segmented storage pools.

Each invocation owns its submission context and storage leases. Invocations
share immutable executable plans and pipeline caches. Metal never compiles or
falls back to CPU during execution of an unsupported program; unsupported
lowering or pipeline preparation fails executable compilation.

### Async execution and cancellation

Compiled materialization, reusable execution, decode, readback, and safetensors
I/O return native promises. Tokio's blocking task pool runs the work. The
TypeScript adapter connects the Effect fiber's abort signal to a native
`CancellationToken`. Cancellation and completion race through one atomic
commit, so only one result is returned. Graph construction, autodiff
transformation, and program compilation use synchronous native calls and cannot
be interrupted.

The adapter waits for interrupted native work to finish before discarding late
results. It cleans up late tensor and archive results when it owns their buffers.
External ArrayBuffer finalizers reclaim discarded readback buffers. The same
cleanup applies to transient and reusable programs, decode, readback, and
safetensors I/O.

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

Unsupported placement or dtype requests fail; the runtime never moves the graph
to another backend.

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
runProgram             compileDecodeProgram  runDecodeProgram
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

Exact autoregressive-chain speculation uses the same token-page API. It accepts
one KV-only proposer. The proposer must use an identity vocabulary map and
causal-normalized proposal distributions:

```ts
import { Model, Speculation } from "@effect-torch/core"

const proposer = yield* Speculation.artifact({
  components: [{ model: draftModel, params: draftParams }],
  plan: {
    target: { vocabulary: 32_000 },
    stages: [{ operation: { _tag: "Autoregressive", component: 0 } }],
    state: { _tag: "Kv", commit: { _tag: "AutoregressiveChain", stage: 0 } },
    output: { topology: "Chains", probabilities: "CausalNormalized" },
    tokenMap: { _tag: "Identity" },
    trainedMaxRows: 4
  }
})

const inference = yield* Model.inference(targetModel, targetParams, {
  maxTokens: 8192,
  speculation: { proposer, maxDraftTokens: 4 },
  sampling: { temperature: 0.8, topK: 40, topP: 0.95, seed: 7 }
})
```

One native round proposes tokens, verifies them, performs exact rejection and
residual sampling, and publishes paired target and proposer state. A returned
page may contain multiple tokens. Consumers must append every token in each
page.

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
- Token-to-ID and ID-to-token lookup.
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
| `Runtime.BackendError`       | Structured backend operation and ownership failures     |
| `Tensor.TensorError`         | Graph, evaluation, readback, and serialization failures |
| `Gradient.GradError`         | Autodiff contract failures                              |
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
| `@effect-torch/tokenizers`           | Yes         | Yes       | Yes             | Yes              | Yes           | Yes            |

`pnpm build` produces 14 `.node` artifacts:

- Six CPU binaries.
- Two Apple Metal binaries.
- Six tokenizer binaries.

Windows binaries are not packaged.

Applications can install and import the Apple package on any platform to call
`isAvailable`. The Metal runtime and native binaries remain Darwin-only.

### Loader selection

CPU and tokenizer loaders select one package-local binary from
`process.platform`, `process.arch`, and the presence of glibc in `process.report`
on Linux. The Apple loader performs platform and architecture selection only
when `isAvailable` runs or Effect builds the backend Layer.

Installation does not download binaries. Loaders do not search fallback paths
or switch to CPU. Linux-capable packages include both GNU and musl binaries.

### Static linkage

Each backend addon is a self-contained `cdylib`. Shared Rust graph, compiler,
autodiff, runtime, and N-API helper crates are statically linked into the addon.

Addons communicate through Node-API, not a Rust dynamic-plugin ABI. CPU and
Metal do not pass Rust trait objects or allocations between addons.

### Build matrix

`pnpm build` must run on macOS because Apple Metal artifacts require Xcode, the
macOS SDK, and Apple's linker tools. The command builds Darwin targets locally
and cross-compiles Linux targets with Zig:

| Artifact suffix    | Build target                     |
| ------------------ | -------------------------------- |
| `darwin-arm64`     | `aarch64-apple-darwin`           |
| `darwin-x64`       | `x86_64-apple-darwin`            |
| `linux-arm64-gnu`  | `aarch64-unknown-linux-gnu.2.17` |
| `linux-arm64-musl` | `aarch64-unknown-linux-musl`     |
| `linux-x64-gnu`    | `x86_64-unknown-linux-gnu.2.17`  |
| `linux-x64-musl`   | `x86_64-unknown-linux-musl`      |

Darwin uses Cargo and Apple's system SDK. Linux uses `cargo-zigbuild`. Darwin
artifacts target macOS 11 or newer. Release verification limits GNU artifacts
to glibc 2.17 symbols. Musl addons link dynamically against musl libc.

### Package verification

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

### Reproducible environment

A Nix flake and direnv configure the macOS and Linux development shells. The
shell includes Node.js 22, Corepack, Rustup, Zig, `cargo-zigbuild`, dprint,
CMake, and pkg-config.

```bash
direnv allow
pnpm install
```

Without direnv:

```bash
nix develop
pnpm install
```

`rust-toolchain.toml` pins Rust, rustfmt, rust-analyzer, and the complete
standard-library target set.

Outside Nix, install Node, pnpm, the pinned Rust toolchain, Zig, and
`cargo-zigbuild`. Darwin builds also require Xcode Command Line Tools.

### Native development builds

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

### Quality commands

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

Rust checks need the `napi-addon` feature because each backend gates its
Node-facing module when compiled as a normal `rlib`. VS Code configuration
enables this feature for rust-analyzer.

### Release build

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

`pnpm build` assembles the release matrix on macOS because Apple artifacts
require the macOS SDK. It cross-compiles Linux outputs with Zig. Linux can still
build and test CPU and tokenizer packages; Metal tests require macOS.

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
pnpm bench:muse-glimmer

cargo bench -p effect-torch-compiler --bench pipeline
cargo bench -p effect-torch-compiler --bench pipeline -- --workload stress
```

The benchmark package covers matmul shapes, compiled programs, cold native
compilation, warm structural caches, attention, and optional MLX comparisons.
`N`, `ITERS`, and `METAL_ONLY` configure the default matmul benchmark.
`pnpm bench:compile -- --help` lists backend, workload, size, iteration, and
optimization controls. `pnpm bench:muse-glimmer` benchmarks the local
Muse-Glimmer GGUF in standard and DFlash modes at several context depths and
writes JSONL records under `bench-results/`. Set `LLAMA_CPP_BIN` to a directory
containing `llama-bench`, `llama-cli`, and `llama-speculative-simple` to include
matched llama.cpp kernel and end-to-end rows.

The Rust compiler benchmark measures `GraphIndex` plus side-table optimization
separately from graph construction and reports deterministic structural-work
counts.
Its `stress` workload runs 50,000- and 100,000-node graphs on a 256 KiB thread
stack; it does not include lowering, memory/physical planning, or pipeline
preparation.

This README omits benchmark numbers because results depend on the hardware and
software environment. `pnpm bench` runs CPU measurements on Linux and adds
Metal when available on macOS. The MLX comparison is macOS-only.

## Repository layout

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

## Current constraints

The repository ships two independently packaged runtimes: CPU and Apple Metal.

- The repository implements no CUDA, PJRT, remote, WebGPU, or Windows backend.
- The runtime does not select a backend or transfer tensors between devices
  implicitly.
- Apple Metal is macOS-only and never falls back to CPU.
- Metal does not support F64 or rank-2 linalg operations.
- CPU does not implement F16 or BF16 matmul.
- Mixed-BF16 training is Metal-only.
- INT8 is a KV-cache storage tier, not a general tensor dtype.
- Some optimized attention and loss backward paths are first-order only.
- Release-matrix assembly runs on macOS because Apple artifacts require the
  macOS SDK; Zig cross-compiles the Linux artifacts.
- The repository does not automate native release publication, signing,
  notarization, or registry uploads.

Use the implementation to determine current behavior. RFCs record design intent
and historical decisions, and older details may no longer match the
implementation.

## Design documents

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

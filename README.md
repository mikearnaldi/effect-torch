# effect-torch

An experimental, learning-oriented tensor library for TypeScript, built on
[Effect](https://effect.website) and a Rust native backend powered by
[candle](https://github.com/huggingface/candle).

> **Disclaimer:** this is an experimental project built for learning purposes.
> It is not production-ready and the APIs may change at any time.

The objective is to expose the TypeScript community to the building blocks
behind artificial intelligence. Modern AI is, at its core, tensor algebra:
multidimensional arrays of numbers, a small vocabulary of operations over
them, and an execution engine that runs those operations efficiently on CPUs
and GPUs. This project rebuilds those blocks from first principles, in the
open, with an API surface small enough to read end to end:

- **Tensors and dtypes** — the raw data of AI models
- **A lazy computation graph** — operations describe _what_ to compute, a
  single `compute` decides _when_
- **Backend runtime abstraction** — the same program can target independently
  packaged CPU, Metal, CUDA, or remote implementations
- **Typed errors and cancellation** — because correctness and resource
  control matter as much as speed

## Architecture

```
┌──────────────────────────────────────────────────┐
│ @effect-torch/core (TypeScript)                  │
│   backend-neutral tensors and Runtime service    │
└──────────────────────┬───────────────────────────┘
                       │ opaque handles
┌──────────────────────┴───────────────────────────┐
│ @effect-torch/backend-native                     │
│   CPU / Metal Runtime Layers over the Rust addon │
└──────────────────────────────────────────────────┘
```

- **Lazy by design.** Every operation appends a node to a computation graph.
  Nothing runs until `compute`, which compiles the whole graph in one FFI
  call, deduplicates shared subexpressions, and frees intermediates when done.
- **Never blocks JavaScript.** Graph evaluation and data readback run on a
  Rust blocking thread pool; the JS event loop stays free.
- **Interruptible.** `compute` and `toTypedArray` wire Effect fiber
  interruption into a native cancellation token — interrupting the fiber
  aborts the native computation.
- **Strict dtypes.** No implicit promotion: mixing `f32` with `i64` fails with
  a typed `TensorError`, and `cast` is the only way to convert.
- **Runtime as a service.** One ambient `Runtime.Runtime` is authoritative for
  the Effect program. Tensors retain opaque handles and static metadata, not a
  runtime reference. Backend selection is explicit through a Layer at the
  program boundary.

## Installation

This is a pnpm monorepo; TypeScript runs directly through
[tsx](https://tsx.is), so no build step is needed for local development.
With [Nix](https://nixos.org/) and [direnv](https://direnv.net/) installed,
`direnv allow` provides Node, pnpm, and Rust; otherwise install them
manually.

```bash
pnpm install
pnpm build   # builds the native module and TypeScript packages
```

## Usage

```ts
import * as BackendNative from "@effect-torch/backend-native"
import { Tensor } from "@effect-torch/core"
import { Effect } from "effect"

const program = Effect.gen(function*() {
  // constructors build a lazy graph — nothing is computed yet
  const a = yield* Tensor.randn([512, 512])
  const b = yield* Tensor.randn([512, 512])

  // operations extend the graph, checking shapes/dtypes/devices eagerly
  const c = yield* Tensor.matmul(a, b)
  const d = yield* Tensor.add(c, yield* Tensor.constantLike(c, 1))
  const e = yield* Tensor.mean(d)

  // evaluate runs the whole graph on the device, off the JS thread
  const result = yield* Tensor.toTypedArray(e)
  console.log(result[0])
})

// backend selection is explicit at the program boundary
Effect.runPromise(Effect.provide(program, BackendNative.Metal))
```

## API

The primary backend-neutral APIs are exported through the `Tensor` and
`Runtime` namespaces. Backend packages provide concrete runtime Layers.

### `Runtime` and `@effect-torch/backend-native`

| Export                                 | Description                                        |
| -------------------------------------- | -------------------------------------------------- |
| `Runtime.Runtime`                      | Ambient Effect service tag                         |
| `Runtime.RuntimeService`               | Backend implementation contract                    |
| `Runtime.Placement`                    | Runtime-owned device and memory placement metadata |
| `BackendNative.Cpu` / `Metal` / `Best` | Layers providing the native CPU or Metal runtime   |
| `BackendNative.isAvailable(device)`    | Checks native device availability                  |

### `Tensor` — types

| Export                       | Description                                                   |
| ---------------------------- | ------------------------------------------------------------- |
| `Any`                        | Supertype accepted by every operation                         |
| `Lazy`                       | A tensor described by a lazy computation graph                |
| `Concrete`                   | A materialized tensor living on the device                    |
| `DType`                      | `"f32" \| "f64" \| "f16" \| "bf16" \| "i64" \| "u8" \| "u32"` |
| `TypedArray`                 | The JS typed arrays matching each `DType`                     |
| `TensorError`                | Tagged error raised by every operation                        |
| `isLazyTensor` / `isTensor`  | Refinements on `Any`                                          |
| `shape` / `dtype` / `device` | Getters                                                       |

### `Tensor` — constructors

All constructors return `Effect<Tensor.Lazy, TensorError, Runtime.Runtime>` and
accept an optional `{ dtype }`.

| Export                                          | Description                                 |
| ----------------------------------------------- | ------------------------------------------- |
| `zeros(shape, options?)`                        | Tensor filled with zeros                    |
| `ones(shape, options?)`                         | Tensor filled with ones                     |
| `full(shape, value, options?)`                  | Tensor filled with a constant               |
| `randn(shape, options?)`                        | Standard normal samples (float dtypes)      |
| `arange(start, end?, options?)`                 | Evenly spaced 1-d range, optional `step`    |
| `linspace(start, end, steps, options?)`         | Evenly spaced inclusive range, float dtypes |
| `eye(n, options?)`                              | Identity matrix                             |
| `fromTypedArray(data, shape?)`                  | From a JS typed array, dtype inferred       |
| `zerosLike` / `onesLike` / `fullLike(t, value)` | Filled tensors matching shape and dtype     |

### `Tensor` — elementwise operations

All dual (data-first and data-last), all lazy. Binary operations accept
`Tensor.Any` and broadcast like NumPy; use `constantLike` to lift a number
to a scalar with matching dtype and placement. Mixed dtypes fail, and the
backend rejects incompatible handles.

| Export                                                                   | Description                                                 |
| ------------------------------------------------------------------------ | ----------------------------------------------------------- |
| `add` / `sub` / `mul` / `div`                                            | Arithmetic with broadcasting                                |
| `maximum` / `minimum`                                                    | Elementwise max/min with broadcasting                       |
| `remainder(t, other)`                                                    | Floor-based remainder (PyTorch semantics)                   |
| `eq` / `ne` / `gt` / `lt` / `ge` / `le`                                  | Comparisons, return a `u8` tensor                           |
| `logicalAnd` / `logicalOr` / `logicalNot`                                | Boolean ops on `u8` tensors                                 |
| `where(cond, a, b)`                                                      | Elementwise select on a `u8` condition, 3-way broadcasting  |
| `clamp(t, { min?, max? })`                                               | Clamp into a range                                          |
| `neg` / `abs` / `sign`                                                   | Sign operations (`sign` has zero gradient)                  |
| `sqrt` / `rsqrt` / `square` / `reciprocal` / `pow(t, exponent)`          | Powers and roots                                            |
| `exp` / `expm1` / `log` / `log1p` / `log2` / `log10`                     | Exponentials and logarithms                                 |
| `sin` / `cos` / `tan` / `sinh` / `cosh` / `tanh`                         | Trigonometric and hyperbolic                                |
| `erf` / `floor` / `ceil` / `round`                                       | Error function and rounding (zero gradient a.e.)            |
| `sigmoid` / `relu`                                                       | Basic activations                                           |
| `silu` / `gelu` / `mish` / `elu` / `leakyRelu` / `softplus` / `hardtanh` | Neural network activations                                  |
| `matmul(a, b)` / `dot(a, b)`                                             | Batched matmul over the last two dims; `dot` reduces rank-1 |
| `cast(t, dtype)`                                                         | Explicit dtype conversion                                   |

### `Tensor` — reductions

`sum` / `mean` / `max` / `min` / `variance` / `std` / `norm` / `prod` /
`logsumexp` / `all` / `any`, each taking
`{ dims?: number[], keepdims?: boolean }` (`variance`/`std` add
`correction`, `norm` adds `ord`). Defaults to reducing all dimensions;
negative dims count from the end. `prod` is a naive slice fold (no backend
product kernel). Also `argmax` / `argmin` (indices along a dim, `i64`) and
`cumsum(dim)`.

### `Tensor` — neural network

| Export                                                                                           | Description                                                                                                                     |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `softmax(t, { dims? })` / `logSoftmax`                                                           | Stable, last dim by default                                                                                                     |
| `scaledDotProductAttention(q, k, v, { scale?, causal? })`                                        | Flash attention: single-kernel online-softmax forward on Metal, chunked-recompute backward; composed candle reference elsewhere |
| `dropout(t, { p? })`                                                                             | Functional inverted dropout; mask drawn at evaluation time                                                                      |
| `conv1d` / `conv2d(t, weight, { stride?, padding?, dilation?, groups? })`                        | Convolution via im2col + matmul — all backends, fully differentiable                                                            |
| `convTranspose1d` / `convTranspose2d(t, weight, { stride?, padding?, outputPadding?, groups? })` | Transposed convolution, composed from dilation + conv                                                                           |
| `maxPool2d` / `avgPool2d(t, { kernelSize, stride?, padding? })`                                  | Pooling via window slices                                                                                                       |

### `Tensor` — shape operations

| Export                                                                   | Description                                   |
| ------------------------------------------------------------------------ | --------------------------------------------- |
| `reshape(t, shape)`                                                      | Same element count, new shape                 |
| `flatten(t, { startDim?, endDim? })`                                     | Collapse a dim range                          |
| `squeeze(t, { dims? })` / `unsqueeze(t, dim)`                            | Remove/insert size-1 dims                     |
| `transpose(t, dims)`                                                     | Permute dimensions                            |
| `slice(t, { start?, end?, stride? })`                                    | Per-dim ranges, negative indices              |
| `split(t, sections, { dim? })` / `chunk(t, chunks, { dim? })`            | Split into parts along a dim                  |
| `concat([t1, t2, ...], { dim? })`                                        | Concatenate along an existing dim             |
| `stack([t1, t2, ...], { dim? })`                                         | Stack along a new dim                         |
| `broadcastTo(t, shape)`                                                  | Broadcast to a larger shape                   |
| `tile(t, reps)`                                                          | Repeat the tensor per dim                     |
| `pad(t, [[before, after], ...])`                                         | Zero-pad per dim                              |
| `take(t, indexes, { dim? })`                                             | Gather rows by `i64` indexes (differentiable) |
| `gather(t, indexes, { dim? })` / `scatterAdd(t, indexes, src, { dim? })` | Take-along-dim and its differentiable inverse |
| `oneHot(indexes, depth, { dtype? })`                                     | Class indexes to one-hot                      |
| `triu(t, { diagonal? })` / `tril(t, { diagonal? })`                      | Triangular masks                              |
| `flip(t, dims)`                                                          | Reverse element order along dims              |
| `trace(t)`                                                               | Sum of the diagonal of a square matrix        |

### `Tensor` — linear algebra

Rank-2, square, float. Runs on the CPU (other devices round-trip through
the host); all three are differentiable.

| Export        | Description          |
| ------------- | -------------------- |
| `inverse(t)`  | Matrix inverse       |
| `det(t)`      | Determinant (scalar) |
| `solve(a, b)` | Solve `a @ x = b`    |

### `Loss` — loss functions

All take the prediction first and accept `{ reduction?: "mean" | "sum" | "none" }`
(default `"mean"`, producing a scalar ready for `Gradient.grad`):

| Export                                                         | Description                            |
| -------------------------------------------------------------- | -------------------------------------- |
| `mse(pred, target)` / `l1` / `huber(pred, target, { delta? })` | Regression losses                      |
| `binaryCrossEntropy(pred, target, { fromLogits? })`            | BCE from probabilities or logits       |
| `crossEntropy(logits, targets)`                                | Log-softmax + NLL, `i64` class targets |
| `nll(logProbs, targets)`                                       | Negative log likelihood                |
| `klDiv(logPred, target)`                                       | KL divergence from log-probabilities   |
| `hinge(pred, target)`                                          | Hinge loss for ±1 targets              |
| `cosineEmbeddingLoss(a, b, targets, { margin? })`              | Cosine embedding loss for ±1 targets   |

### `Tensor` — evaluation

| Export                   | Description                                                                                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `compute([t1, t2, ...])` | `Effect<Tensor[], TensorError, Runtime.Runtime>` — runs one shared graph walk: shared subgraphs compute once and roots share each random draw; interruptible |
| `toTypedArray(t)`        | `Effect<TypedArray, TensorError, Runtime.Runtime>` — evaluate plus zero-copy readback where possible                                                         |
| `toNumberArray(t)`       | `Effect<number[], TensorError, Runtime.Runtime>` — fails for `i64` tensors instead of silently coercing bigints                                              |

### `Gradient` — autodiff

Reverse-mode automatic differentiation: `grad` operates directly
on the lazy graph — there is no tracing and no function transformation. The
backward transform runs natively in Rust; adjoints are ordinary lazy nodes,
so gradients can be differentiated again.

| Export                           | Description                                                                                                                       |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `grad(loss, wrt)`                | gradients of a scalar `loss` w.r.t. the given tensors; tensors not influencing the loss get zeros                                 |
| `vjp(y, x, v)`                   | pullback `J(x)ᵀ v` of an output graph `y` built from `x`                                                                          |
| `jvp(y, x, v)`                   | pushforward `J(x) v` (forward-over-reverse)                                                                                       |
| `vmap(y, x, batchedX, { dim? })` | batch the function implicit in `y` over `dim` of `batchedX` — a native graph rewrite with per-op batching rules, not a slice loop |
| `stopGradient(t)`                | blocks gradient flow through `t`                                                                                                  |
| `checkpoint(t)`                  | recomputes the subgraph's intermediates in backward instead of retaining them                                                     |
| `GradError`                      | typed error: non-scalar output, non-float dtype, or non-differentiable op                                                         |

```ts
const step = Effect.gen(function*() {
  const pred = yield* Tensor.add(yield* Tensor.matmul(x, w), b)
  const loss = yield* Loss.mse(pred, y)
  const [gw, gb] = yield* Gradient.grad(loss, [w, b])
  // loss and grads share the forward graph: evaluate them in one walk
  const [l, gW, gB] = yield* Tensor.compute([loss, gw, gb])
  // ...optimizer step...
})
```

### `Tensor` — serialization

Tensors are saved and loaded in the [safetensors](https://github.com/huggingface/safetensors)
format. All file I/O, graph evaluation, and serialization happen on the
native side — tensor data never crosses the JavaScript thread.

| Export                              | Description                                                                                                                  |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `save(path, { name: tensor, ... })` | `Effect<void, TensorError, Runtime.Runtime>` — evaluates all entries in one shared graph walk and writes through the runtime |
| `load(path)`                        | `Effect<Record<string, Tensor>, TensorError, Runtime.Runtime>` — imports materialized tensors through the active runtime     |

```ts
yield * Tensor.save("checkpoint.safetensors", { "model.w": w, "model.b": b })
const loaded = yield * Tensor.load("checkpoint.safetensors")
const w = loaded["model.w"] // ordinary materialized tensor
```

### `Optimizer`

Optimizers are pure graph transforms: `step` takes parameters, gradients,
and state and returns updated parameters and state as lazy values — nothing
is mutated or materialized inside. The convenience `Optimizer.step` runs a
full training step (gradients + update + evaluation) in a **single graph
walk**: one forward pass, one backward pass, one async boundary, and
gradient tensors are freed as soon as their update consumes them. Formulas
match PyTorch/candle exactly.

| Export                                                    | Description                                                                            |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `sgd({ momentum?, dampening?, nesterov?, weightDecay? })` | SGD with optional momentum and coupled L2 decay                                        |
| `adam({ beta1?, beta2?, eps? })`                          | Adam defaults (`0.9`, `0.999`, `1e-8`); f32 rejects >1% bias-correction rounding error |
| `adamW({ ..., weightDecay? })`                            | Adam with decoupled weight decay (default `0.01`)                                      |
| `step(optimizer, loss, params, state, lr)`                | Full gradient, update, and materialization step in one walk                            |
| `optimizer.init(params)`                                  | Zero-initialized state; validates float dtypes                                         |
| `optimizer.step(params, grads, state, lr)`                | Raw graph transform for custom loops; `lr` is a 0-d float tensor                       |

```ts
const trained = Effect.gen(function*() {
  const optimizer = yield* Optimizer.adam()
  let params = [w, b]
  let state = yield* optimizer.init(params)
  for (let i = 0; i < steps; i++) {
    const pred = yield* Tensor.add(yield* Tensor.matmul(x, params[0]), params[1])
    const loss = yield* Loss.mse(pred, y)
    const lr = yield* Tensor.constantLike(params[0], 0.1)
    const result = yield* Optimizer.step(optimizer, loss, params, state, lr)
    params = result.params // materialized leaves of the next step's graph
    state = result.state
  }
  return params
})
```

### `Model`

Models are pure values pairing parameter construction with a parameterised
forward graph — the Flax/Haiku design, flattened: parameters are a flat
array of tensors, so `Gradient.grad` and `Optimizer.step` work on any
model with zero adapter code. There is no mutable module state. Everything
that can fail returns an `Effect`: constructors validate into a
`ModelError`, and `Trainer` runs the full training loop.

| Export                                                            | Description                                                                                                                                 |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `Model.linear(name, in, out)`                                     | `Effect` of a fully-connected layer, `randn * 1/√in` weight, zero bias                                                                      |
| `Model.conv1d` / `Model.conv2d(name, in, out, k, opts?)`          | `Effect`s of convolution layers, fan-in-scaled weight, per-channel bias                                                                     |
| `Model.embedding(name, num, dim, opts?)`                          | `Effect` of an embedding lookup layer, unit-normal weight                                                                                   |
| `Model.positionEmbedding(name, max, dim)`                         | `Effect` of a learned absolute position embedding; reads only the input's sequence length                                                   |
| `Model.layerNorm(name, shape, { eps? })`                          | `Effect` of layer normalization over the trailing `shape` dims                                                                              |
| `Model.multiHeadAttention(name, embedDim, numHeads, { causal? })` | `Effect` of multi-head attention: wq/wk/wv/wo projections over `scaledDotProductAttention`                                                  |
| `Model.tanh` / `sigmoid` / `relu` / `silu` / `mish` / `softplus`  | `Effect`s of activations as parameterless models                                                                                            |
| `Model.gelu(opts?)` / `elu(opts?)` / `leakyRelu(opts?)`           | `Effect`s of option-taking activations                                                                                                      |
| `Model.softmax(dim?)` / `logSoftmax(dim?)` / `flatten(opts?)`     | `Effect`s of shape/reduction stages (`flatten` preserves the batch dim)                                                                     |
| `Model.dropout({ p? })`                                           | `Effect` of inverted dropout — always applies; build the eval chain without it                                                              |
| `Model.maxPool2d(opts)` / `avgPool2d(opts)`                       | `Effect`s of pooling stages                                                                                                                 |
| `Model.chain(...models)`                                          | `Effect` of sequential composition; parameter arrays concatenated in order, arity checked in `forward`                                      |
| `Model.checkpoint(block)`                                         | gradient-checkpoint a sub-model: recompute its forward during backward, trading FLOPs for peak memory                                       |
| `Model.residual(block)`                                           | add a skip connection: `forward = input + block(input)`                                                                                     |
| `Model.mapInput(model, f)`                                        | transform the input before it enters the sub-model (positions from a sequence length, patches from an image)                                |
| `Model.merge([a, b, ...], f)`                                     | fan one input into several models and combine their outputs with a variadic combiner — non-sequential tops like token + position embeddings |
| `Model.save(model, params, path)` / `Model.load(model, path)`     | named checkpoints via safetensors                                                                                                           |

```ts
const program = Effect.gen(function*() {
  const model = yield* Model.chain(
    yield* Model.linear("fc1", 2, 8),
    yield* Model.tanh,
    yield* Model.linear("fc2", 8, 1),
    yield* Model.sigmoid
  )

  const trainer = yield* Trainer.make(model, {
    optimizer: yield* Optimizer.adam(),
    lr: LearningRate.constant(0.1),
    loss: Loss.mse,
    data: { input: x, target: y },
    stop: ({ step, loss }) => step >= 3000 || loss < 1e-4,
    onStep: ({ step, loss }) => (step % 250 === 0 ? Effect.log(`step ${step} loss ${loss}`) : Effect.void)
  })
  const trained = yield* trainer.train()
  // trained.params: materialized leaves, ready for forward / save / more training
})
```

Errors are typed: every operation fails with `TensorError` (shape, dtype, or
device mismatch at graph-build time; backend errors at evaluation time).
Interruption is structured: interrupting the fiber running `compute` or
`toTypedArray` cancels the native work.

## Benchmarks

Measured on an Apple Silicon MacBook, f32, median over iterations; each
iteration builds a fresh graph of 10 chained matmuls and evaluates it once.
`pnpm bench` and `pnpm bench:mlx` reproduce these.

**matmul 512×512 @ 512×512 (50 iterations, 10 chained per iteration)**

| Backend            | ms/op | GFLOP/s |
| ------------------ | ----: | ------: |
| effect-torch CPU   | 0.152 |   1,766 |
| effect-torch Metal | 0.133 |   2,023 |

**matmul 2048×2048 @ 2048×2048 (50 iterations, 10 chained per iteration)**

| Backend            | ms/op | GFLOP/s |
| ------------------ | ----: | ------: |
| effect-torch CPU   | 7.831 |   2,194 |
| effect-torch Metal | 1.371 |  12,527 |

**Head-to-head with [MLX](https://github.com/frost-beta/node-mlx), 512×512,
10 chained matmuls (median of 5 runs)**

| Framework    | ms/op |
| ------------ | ----: |
| effect-torch | 0.061 |
| node-mlx     | 0.071 |

## Development

```bash
pnpm -r typecheck                        # typecheck all packages (against sources, no build needed)
pnpm --filter @effect-torch/core test    # @effect/vitest suite
pnpm bench                               # matmul benchmark (cpu + metal)
pnpm bench:mlx                           # head-to-head against MLX
pnpm --filter @effect-torch/examples xor      # train the XOR example
pnpm --filter @effect-torch/examples nano-gpt # char-level GPT (attention, flash kernel on Metal)
```

Layout:

- `packages/native` — Rust backend (candle, napi-rs), lazy graph + evaluator
- `packages/core` — public TypeScript API (`Tensor`, `Device`, `Optimizer`)
- `packages/examples` — runnable examples (`xor.ts`, `nano-gpt.ts`)
- `packages/bench` — benchmarks, including an MLX comparison

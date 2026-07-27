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
- **A lazy computation graph** — operations describe *what* to compute, a
  single `evaluate` decides *when*
- **Device abstraction** — the same program runs on CPU, Metal, or CUDA
- **Typed errors and cancellation** — because correctness and resource
  control matter as much as speed

## Architecture

```
┌─────────────────────────────────────────────┐
│ @effect-torch/core (TypeScript)             │
│   Tensor ops → Effect<LazyTensor, TensorError> │
│   shape/dtype/device checked at graph build  │
└──────────────────┬──────────────────────────┘
                   │ napi-rs (one FFI hop per evaluate)
┌──────────────────┴──────────────────────────┐
│ @effect-torch/native (Rust + candle)        │
│   lazy graph (Arc<LazyNode>) → candle ops    │
│   CPU / Metal / CUDA, tokio blocking pool    │
└─────────────────────────────────────────────┘
```

- **Lazy by design.** Every operation appends a node to a computation graph.
  Nothing runs until `evaluate`, which compiles the whole graph in one FFI
  call, deduplicates shared subexpressions, and frees intermediates when done.
- **Never blocks JavaScript.** Graph evaluation and data readback run on a
  Rust blocking thread pool; the JS event loop stays free.
- **Interruptible.** `evaluate` and `toTypedArray` wire Effect fiber
  interruption into a native cancellation token — interrupting the fiber
  aborts the native computation.
- **Strict dtypes.** No implicit promotion: mixing `f32` with `i64` fails with
  a typed `TensorError`, and `cast` is the only way to convert.
- **Device as a service.** Tensors are created within a `CurrentDevice`
  context, provided by a Layer — device selection is explicit and visible at
  the type level.

## Installation

This is a pnpm monorepo; TypeScript runs directly through
[tsx](https://tsx.is), so no build step is needed for local development.
With [Nix](https://nixos.org/) and [direnv](https://direnv.net/) installed,
`direnv allow` provides Node, pnpm, and Rust; otherwise install them
manually.

```bash
pnpm install
pnpm --filter @effect-torch/native build   # builds the Rust native module
```

## Usage

```ts
import { Device, Tensor } from "@effect-torch/core"
import { Effect } from "effect"

const program = Effect.gen(function* () {
  // constructors build a lazy graph — nothing is computed yet
  const a = yield* Tensor.randn([512, 512])
  const b = yield* Tensor.randn([512, 512])

  // operations extend the graph, checking shapes/dtypes/devices eagerly
  const c = yield* Tensor.matmul(a, b)
  const d = yield* Tensor.add(c, 1)
  const e = yield* Tensor.mean(d)

  // evaluate runs the whole graph on the device, off the JS thread
  const result = yield* Tensor.toTypedArray(e)
  console.log(result[0])
})

// device selection is explicit, via a Layer
Effect.runPromise(Effect.provide(program, Device.Metal))
```

## API

Everything is exported through two namespaces: `Tensor` and `Device`.

### `Device`

| Export | Description |
| --- | --- |
| `DeviceKind` | `"cpu" \| "metal" \| "cuda"` |
| `CurrentDevice` | `Context.Service` holding the device used by constructors |
| `Cpu` / `Metal` / `Cuda` / `layer(device)` | Layers providing `CurrentDevice` |
| `Best` | Layer providing the best available device — probes CUDA, then Metal, falls back to CPU |
| `isAvailable(device)` | `Effect<boolean>` — checks device availability at runtime |

### `Tensor` — types

| Export | Description |
| --- | --- |
| `GenericTensor` | Supertype accepted by every operation |
| `LazyTensor` | A tensor described by a lazy computation graph |
| `Tensor` | A materialized tensor living on the device |
| `DType` | `"f32" \| "f64" \| "i64" \| "u8" \| "u32"` |
| `TypedArray` | The JS typed arrays matching each `DType` |
| `TensorError` | Tagged error raised by every operation |
| `isLazyTensor` / `isTensor` | Refinements on `GenericTensor` |
| `shape` / `dtype` / `device` | Getters |

### `Tensor` — constructors

All constructors return `Effect<LazyTensor, TensorError, CurrentDevice>` and
accept an optional `{ dtype }`.

| Export | Description |
| --- | --- |
| `zeros(shape, options?)` | Tensor filled with zeros |
| `ones(shape, options?)` | Tensor filled with ones |
| `full(shape, value, options?)` | Tensor filled with a constant |
| `randn(shape, options?)` | Standard normal samples (float dtypes) |
| `arange(start, end?, options?)` | Evenly spaced 1-d range, optional `step` |
| `linspace(start, end, steps, options?)` | Evenly spaced inclusive range, float dtypes |
| `eye(n, options?)` | Identity matrix |
| `fromTypedArray(data, shape?)` | From a JS typed array, dtype inferred |
| `zerosLike` / `onesLike` / `fullLike(t, value)` | Filled tensors matching shape and dtype |

### `Tensor` — elementwise operations

All dual (data-first and data-last), all lazy. Binary operations accept
`GenericTensor | number` (a number is lifted to a scalar of the same
dtype/device) and broadcast like NumPy. Mixed dtypes or devices fail.

| Export | Description |
| --- | --- |
| `add` / `sub` / `mul` / `div` | Arithmetic with broadcasting |
| `maximum` / `minimum` | Elementwise max/min with broadcasting |
| `remainder(t, other)` | Floor-based remainder (PyTorch semantics) |
| `eq` / `ne` / `gt` / `lt` / `ge` / `le` | Comparisons, return a `u8` tensor |
| `logicalAnd` / `logicalOr` / `logicalNot` | Boolean ops on `u8` tensors |
| `where(cond, a, b)` | Elementwise select on a `u8` condition, 3-way broadcasting |
| `clamp(t, { min?, max? })` | Clamp into a range |
| `neg` / `abs` / `sign` | Sign operations (`sign` has zero gradient) |
| `sqrt` / `rsqrt` / `square` / `reciprocal` / `pow(t, exponent)` | Powers and roots |
| `exp` / `expm1` / `log` / `log1p` / `log2` / `log10` | Exponentials and logarithms |
| `sin` / `cos` / `tan` / `sinh` / `cosh` / `tanh` | Trigonometric and hyperbolic |
| `erf` / `floor` / `ceil` / `round` | Error function and rounding (zero gradient a.e.) |
| `sigmoid` / `relu` | Basic activations |
| `silu` / `gelu` / `mish` / `elu` / `leakyRelu` / `softplus` / `hardtanh` | Neural network activations |
| `matmul(a, b)` / `dot(a, b)` | Batched matmul over the last two dims; `dot` reduces rank-1 |
| `cast(t, dtype)` | Explicit dtype conversion |

### `Tensor` — reductions

`sum` / `mean` / `max` / `min` / `variance` / `std` / `norm` / `prod` /
`logsumexp` / `all` / `any`, each taking
`{ dims?: number[], keepdims?: boolean }` (`variance`/`std` add
`correction`, `norm` adds `ord`). Defaults to reducing all dimensions;
negative dims count from the end. `prod` is a naive slice fold (no backend
product kernel). Also `argmax` / `argmin` (indices along a dim, `i64`) and
`cumsum(dim)`.

### `Tensor` — neural network

| Export | Description |
| --- | --- |
| `softmax(t, { dims? })` / `logSoftmax` | Stable, last dim by default |
| `dropout(t, { p? })` | Functional inverted dropout; mask drawn at evaluation time |
| `conv1d` / `conv2d(t, weight, { stride?, padding?, dilation?, groups? })` | Convolution via im2col + matmul — all backends, fully differentiable |
| `convTranspose1d` / `convTranspose2d(t, weight, { stride?, padding?, outputPadding?, groups? })` | Transposed convolution, composed from dilation + conv |
| `maxPool2d` / `avgPool2d(t, { kernelSize, stride?, padding? })` | Pooling via window slices |

### `Tensor` — shape operations

| Export | Description |
| --- | --- |
| `reshape(t, shape)` | Same element count, new shape |
| `flatten(t, { startDim?, endDim? })` | Collapse a dim range |
| `squeeze(t, { dims? })` / `unsqueeze(t, dim)` | Remove/insert size-1 dims |
| `transpose(t, dims)` | Permute dimensions |
| `slice(t, { start?, end?, stride? })` | Per-dim ranges, negative indices |
| `split(t, sections, { dim? })` / `chunk(t, chunks, { dim? })` | Split into parts along a dim |
| `concat([t1, t2, ...], { dim? })` | Concatenate along an existing dim |
| `stack([t1, t2, ...], { dim? })` | Stack along a new dim |
| `broadcastTo(t, shape)` | Broadcast to a larger shape |
| `tile(t, reps)` | Repeat the tensor per dim |
| `pad(t, [[before, after], ...])` | Zero-pad per dim |
| `take(t, indexes, { dim? })` | Gather rows by `i64` indexes (differentiable) |
| `gather(t, indexes, { dim? })` / `scatterAdd(t, indexes, src, { dim? })` | Take-along-dim and its differentiable inverse |
| `oneHot(indexes, depth, { dtype? })` | Class indexes to one-hot |
| `triu(t, { diagonal? })` / `tril(t, { diagonal? })` | Triangular masks |
| `flip(t, dims)` | Reverse element order along dims |
| `trace(t)` | Sum of the diagonal of a square matrix |

### `Tensor` — linear algebra

Rank-2, square, float. Runs on the CPU (other devices round-trip through
the host); all three are differentiable.

| Export | Description |
| --- | --- |
| `inverse(t)` | Matrix inverse |
| `det(t)` | Determinant (scalar) |
| `solve(a, b)` | Solve `a @ x = b` |

### `Loss` — loss functions

All take the prediction first and accept `{ reduction?: "mean" | "sum" | "none" }`
(default `"mean"`, producing a scalar ready for `Gradient.grad`):

| Export | Description |
| --- | --- |
| `mse(pred, target)` / `l1` / `huber(pred, target, { delta? })` | Regression losses |
| `binaryCrossEntropy(pred, target, { fromLogits? })` | BCE from probabilities or logits |
| `crossEntropy(logits, targets)` | Log-softmax + NLL, `i64` class targets |
| `nll(logProbs, targets)` | Negative log likelihood |
| `klDiv(logPred, target)` | KL divergence from log-probabilities |
| `hinge(pred, target)` | Hinge loss for ±1 targets |
| `cosineEmbeddingLoss(a, b, targets, { margin? })` | Cosine embedding loss for ±1 targets |

### `Tensor` — evaluation

| Export | Description |
| --- | --- |
| `evaluate([t1, t2, ...])` | `Effect<Tensor[], TensorError>` — runs the graph in one shared walk: shared subgraphs computed once, single `randn` draw across roots; interruptible |
| `toTypedArray(t)` | `Effect<TypedArray, TensorError>` — evaluate + zero-copy readback where possible |
| `toNumberArray(t)` | `Effect<number[], TensorError>` — fails for `i64` tensors instead of silently coercing bigints |

### `Gradient` — autodiff

Reverse-mode automatic differentiation: `grad` operates directly
on the lazy graph — there is no tracing and no function transformation. The
backward transform runs natively in Rust; adjoints are ordinary lazy nodes,
so gradients can be differentiated again.

| Export | Description |
| --- | --- |
| `grad(loss, wrt)` | gradients of a scalar `loss` w.r.t. the given tensors; tensors not influencing the loss get zeros |
| `vjp(f, x, v)` | `f(x)` plus the pullback `J(x)ᵀ v` |
| `jvp(f, x, v)` | `f(x)` plus the pushforward `J(x) v` (forward-over-reverse) |
| `vmap(f, { dim? })(t)` | map `f` over a dimension, stacking results |
| `stopGradient(t)` | blocks gradient flow through `t` |
| `checkpoint(t)` | recomputes the subgraph's intermediates in backward instead of retaining them |
| `GradError` | typed error: non-scalar output, non-float dtype, or non-differentiable op |

```ts
const step = Effect.gen(function* () {
  const pred = yield* Tensor.add(yield* Tensor.matmul(x, w), b)
  const loss = yield* Tensor.mse(pred, y)
  const [gw, gb] = yield* Gradient.grad(loss, [w, b])
  // loss and grads share the forward graph: evaluate them in one walk
  const [l, gW, gB] = yield* Tensor.evaluate([loss, gw, gb])
  // ...optimizer step...
})
```

### `Tensor` — serialization

Tensors are saved and loaded in the [safetensors](https://github.com/huggingface/safetensors)
format. All file I/O, graph evaluation, and serialization happen on the
native side — tensor data never crosses the JavaScript thread.

| Export | Description |
| --- | --- |
| `save(path, { name: tensor, ... })` | `Effect<void, TensorError>` — evaluates all entries in one shared graph walk and writes the file natively |
| `load(path)` | `Effect<Record<string, Tensor>, TensorError, CurrentDevice>` — reads the file straight into materialized tensors on the current device |

```ts
yield* Tensor.save("checkpoint.safetensors", { "model.w": w, "model.b": b })
const loaded = yield* Tensor.load("checkpoint.safetensors")
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

| Export | Description |
| --- | --- |
| `sgd({ lr, momentum?, dampening?, nesterov?, weightDecay? })` | SGD with optional momentum and coupled L2 decay |
| `adam({ lr?, beta1?, beta2?, eps? })` | Adam, standard defaults (`1e-3`, `0.9`, `0.999`, `1e-8`) |
| `adamW({ ..., weightDecay? })` | Adam with decoupled weight decay (default `0.01`) |
| `step(optimizer, loss, params, state)` | `Effect<{ loss, params, state }>` — full step in one walk |
| `optimizer.init(params)` | zero-initialized state, validates float dtypes |
| `optimizer.step(params, grads, state)` | the raw graph transform, for custom loops |

```ts
const optimizer = Optimizer.adam({ lr: 0.1 })

const trained = Effect.gen(function* () {
  let params = [w, b]
  let state = yield* optimizer.init(params)
  for (let i = 0; i < steps; i++) {
    const pred = yield* Tensor.add(yield* Tensor.matmul(x, params[0]), params[1])
    const loss = yield* Tensor.mse(pred, y)
    const result = yield* Optimizer.step(optimizer, loss, params, state)
    params = result.params // materialized leaves of the next step's graph
    state = result.state
  }
  return params
})
```

Errors are typed: every operation fails with `TensorError` (shape, dtype, or
device mismatch at graph-build time; backend errors at evaluation time).
Interruption is structured: interrupting the fiber running `evaluate` or
`toTypedArray` cancels the native work.

## Benchmarks

Measured on an Apple Silicon MacBook, f32, median over iterations; each
iteration builds a fresh graph of 10 chained matmuls and evaluates it once.
`pnpm bench` and `pnpm bench:mlx` reproduce these.

**matmul 512×512 @ 512×512 (50 iterations, 10 chained per iteration)**

| Backend | ms/op | GFLOP/s |
| --- | ---: | ---: |
| effect-torch CPU | 0.152 | 1,766 |
| effect-torch Metal | 0.133 | 2,023 |

**matmul 2048×2048 @ 2048×2048 (50 iterations, 10 chained per iteration)**

| Backend | ms/op | GFLOP/s |
| --- | ---: | ---: |
| effect-torch CPU | 7.831 | 2,194 |
| effect-torch Metal | 1.371 | 12,527 |

**Head-to-head with [MLX](https://github.com/frost-beta/node-mlx), 512×512,
10 chained matmuls (median of 5 runs)**

| Framework | ms/op |
| --- | ---: |
| effect-torch | 0.061 |
| node-mlx | 0.071 |

## Development

```bash
pnpm -r typecheck                        # typecheck all packages (against sources, no build needed)
pnpm --filter @effect-torch/core test    # @effect/vitest suite
pnpm bench                               # matmul benchmark (cpu + metal)
pnpm bench:mlx                           # head-to-head against MLX
pnpm --filter @effect-torch/examples xor # train the XOR example
```

Layout:

- `packages/native` — Rust backend (candle, napi-rs), lazy graph + evaluator
- `packages/core` — public TypeScript API (`Tensor`, `Device`, `Optimizer`)
- `packages/examples` — runnable examples (`xor.ts`)
- `packages/bench` — benchmarks, including an MLX comparison

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

This is a pnpm monorepo. With [Nix](https://nixos.org/) and
[direnv](https://direnv.net/) installed, `direnv allow` provides Node, pnpm,
and Rust; otherwise install them manually.

```bash
pnpm install
pnpm --filter @effect-torch/native build   # builds the Rust native module
pnpm --filter @effect-torch/core build     # builds the TypeScript core
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
| `eye(n, options?)` | Identity matrix |
| `fromTypedArray(data, shape?)` | From a JS typed array, dtype inferred |

### `Tensor` — elementwise operations

All dual (data-first and data-last), all lazy. Binary operations accept
`GenericTensor | number` (a number is lifted to a scalar of the same
dtype/device) and broadcast like NumPy. Mixed dtypes or devices fail.

| Export | Description |
| --- | --- |
| `add` / `sub` / `mul` / `div` | Arithmetic with broadcasting |
| `eq` / `gt` / `lt` / `ge` / `le` | Comparisons, return a `u8` tensor |
| `neg` / `abs` / `sqrt` / `exp` / `log` / `sin` / `cos` | Unary math |
| `pow(t, exponent)` | Constant exponent |
| `matmul(a, b)` | Batched matmul over the last two dims |
| `cast(t, dtype)` | Explicit dtype conversion |

### `Tensor` — reductions

`sum` / `mean` / `max` / `min`, each taking
`{ dims?: number[], keepdims?: boolean }`. Defaults to reducing all
dimensions; negative dims count from the end.

### `Tensor` — shape operations

| Export | Description |
| --- | --- |
| `reshape(t, shape)` | Same element count, new shape |
| `transpose(t, dims)` | Permute dimensions |
| `slice(t, { start?, end?, stride? })` | Per-dim ranges, negative indices |
| `concat([t1, t2, ...], { dim? })` | Concatenate along an existing dim |
| `broadcastTo(t, shape)` | Broadcast to a larger shape |

### `Tensor` — evaluation

| Export | Description |
| --- | --- |
| `evaluate([t1, t2, ...])` | `Effect<Tensor[], TensorError>` — runs the graph in one shared walk: shared subgraphs computed once, single `randn` draw across roots; interruptible |
| `toTypedArray(t)` | `Effect<TypedArray, TensorError>` — evaluate + zero-copy readback where possible |

### `Tensor` — autodiff

Reverse-mode automatic differentiation: `grad` operates directly
on the lazy graph — there is no tracing and no function transformation. The
backward transform runs natively in Rust; adjoints are ordinary lazy nodes,
so gradients can be differentiated again.

| Export | Description |
| --- | --- |
| `grad(loss, wrt)` | gradients of a scalar `loss` w.r.t. the given tensors; tensors not influencing the loss get zeros |
| `stopGradient(t)` | blocks gradient flow through `t` |
| `GradError` | typed error: non-scalar output, non-float dtype, or non-differentiable op |

```ts
const step = Effect.gen(function* () {
  const pred = yield* Tensor.add(yield* Tensor.matmul(x, w), b)
  const err = yield* Tensor.sub(pred, y)
  const loss = yield* Tensor.mean(yield* Tensor.mul(err, err))
  const [gw, gb] = yield* Tensor.grad(loss, [w, b])
  // loss and grads share the forward graph: evaluate them in one walk
  const [l, gW, gB] = yield* Tensor.evaluate([loss, gw, gb])
  // ...optimizer step...
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
pnpm -r typecheck     # typecheck all packages (against sources, no build needed)
pnpm --filter @effect-torch/core test    # @effect/vitest suite
pnpm bench            # matmul benchmark (cpu + metal)
pnpm bench:mlx        # head-to-head against MLX
```

Layout:

- `packages/native` — Rust backend (candle, napi-rs), lazy graph + evaluator
- `packages/core` — public TypeScript API (`Tensor`, `Device`)
- `packages/bench` — benchmarks, including an MLX comparison

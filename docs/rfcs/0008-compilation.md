# RFC 0008: Compilation — Frozen, Reusable Graph Executables

- **Status**: Implemented
- **Author**: Michael Arnaldi
- **Date**: 2026-07-31
- **Depends on**: RFC 0002 (autodiff — `grad` runs at trace time), RFC 0003
  (memory management — per-call evaluator, handle finalization), RFC 0004
  (optimizers — the step graph this freezes), RFC 0005 (models — the
  forward graph this freezes), RFC 0007 (kernel fusion — the frozen DAG is
  the post-fusion root set; scalar lanes are the precedent for runtime
  scalars)

## Summary

Today every training step and every forward pass pays four rebuild costs:
JS/Effect graph construction (one `Effect`-wrapped closure per op),
one synchronous napi node allocation per op, a full native `grad`
topo-sort and adjoint build, and the `fuse_roots` + `merge_shared_regions`
rewrites inside every `eval_lazy` call. In a 3000-step training run all
four produce *bit-identical graphs* every step.

**Compilation** freezes all of it. A compiled function is a native
*program*: the post-fusion root DAG plus declared input/output slots,
produced once by tracing the ordinary graph builder against placeholder
leaves. Calling a compiled function is a single async napi call —
materialized input buffers and runtime scalars in, materialized outputs
out — with no JS graph construction, no node allocation, no adjoint
build, and no fusion rewrite per call.

A compiled function has **exactly the observable behaviour of the
original**: one call is one evaluator walk, so shared subgraphs dedup,
`randn`/`uniform`/`dropout` draw fresh per call and consistently across
roots within a call, and consumer-counting early-free applies unchanged.
Programs are immutable values; concurrent and parallel calls are safe by
construction (see Concurrency).

## Motivation

`packages/examples/xor.ts` and `packages/examples/nano-gpt.ts` rebuild
the full forward+backward+update AST at every step of training and the
full forward AST at every step of evaluation. The per-step overhead
scales with graph size and is identical every step; the benches
(`packages/bench/headtohead.ts`) time it directly, since both arms are
lazy-graph libraries and the effect-torch arm includes graph
construction per iteration. Every production-grade library offers the
same escape: define the model, compile it, define the trainer, compile
it — then pay graph construction once.

Compilation also forces a long-overdue module split: training
configuration (`Model.train`, `TrainConfig`, and friends) moves out of
`Model` into a dedicated `Trainer` module, so both the model and the
trainer are encapsulated values that can be compiled independently.
There is no backward-compatibility constraint; `Model.train` is removed,
not deprecated.

## Prior art

- **JAX `jit`**: trace-on-first-call per input signature, automatic
  recompilation keyed by shapes/dtypes, weakly-held program caches. We
  adopt the automatic shape-keyed cache and the functional calling
  convention (arguments in, results out — no mutable state).
- **PyTorch `torch.compile`**: guards + recompile on shape change over a
  mutable module. The mutable-module substrate is rejected here for the
  same reasons as RFC 0005; the guard machinery reduces, in our pure
  setting, to a cache-key check.
- **XLA**: symbolic dynamic dimensions. Non-goal (see Non-goals): our
  graph builders are ordinary TS that *reads* concrete shapes
  (`positionEmbedding` builds `arange(t)` from `input.shape`;
  `multiHeadAttention` derives reshape targets and permutations), so a
  shape change alters the graph's node set, not just edge metadata.
- **MLX**: re-runs the Python builder per call; no program cache. This
  is our status quo and the baseline the benches measure against.

## Design

### Native: placeholder leaves, freeze, run

**Placeholder leaves.** A new `NodeKind::Input { slot: u32 }` (tensor
inputs) and `NodeKind::ScalarInput { slot: u32 }` (0-d f64 scalar
inputs, resolved from the call's scalar vector — the graph-level
counterpart of RFC 0007's `Expr::Scalar` lanes). Both carry declared
shape/dtype/device like any other node and participate in shape
inference and validation at construction. Evaluating one outside a
program run is an error — which is what enforces, at trace time, that a
compiled builder never materializes its inputs.

**Trace.** The TS side creates placeholder `LazyTensor` handles
(`LazyTensor.input(slot, shape, dtype, device)`) matching the incoming
arguments and runs the ordinary builder once. Everything the builder
does — Effect composition, napi node allocation, `grad`, optimizer node
construction — is unchanged; the output is a set of lazy roots whose DAG
contains `Input` leaves wherever the arguments were read. Constants
captured by the builder (constructor leaves, checkpoint-loaded tensors)
stay ordinary leaves shared by the program.

**Freeze.** One synchronous napi call `compile(roots) -> CompiledProgram`:

1. Collects the slot declarations from the root DAG (slot indexes are one
   unified space across tensor and scalar inputs), validating that slots
   are contiguous from 0, that no slot is declared but unused, and that
   no slot is aliased to two different declared signatures.
2. Runs `fuse_roots` + `merge_shared_regions` **once** and stores the
   fused root DAG (`Arc<Node>`, immutable) with the per-root
   shape/dtype metadata. The slot declarations are collected from the
   **fused** DAG: the fusion rewrite rebuilds nodes with fresh ids, so
   declarations taken from the unfused graph would bind against node ids
   the program no longer contains.
3. Returns a napi handle (`CompiledProgram`) with `dispose()`; dropping
   the handle releases the Rust-side `Arc` graphs. A program holds no
   device buffers of its own: the constant/parameter leaves it references
   stay alive through their `NativeTensor` handles' accounting (the same
   sharing discipline as ordinary lazy graphs, RFC 0003).

A program holds **no device buffers**: inputs are rebound per call and
intermediates live only inside a call's walk. Its footprint is CPU graph
metadata; the GPU kernel binaries it may trigger live in the existing
structurally-keyed global pipeline caches (RFC 0007), which programs
share.

**Run.** One async napi call
`program.run(inputs: NativeTensor[], scalars: f64[], token?) -> Promise<NativeTensor[]>`:
validates each input against its declared slot signature (shape, dtype,
device — mismatch is a call-time `TensorError` naming expected vs. got),
builds a fresh `Evaluator`, resolves `Input` nodes from the argument
buffers and `ScalarInput` nodes from the scalar vector, and evaluates
the fused roots with the existing iterative walk: consumer-counting
early-free, dedup cache, cancellation token, roots pinned. The evaluator
is per-call local state exactly as today; nothing about RFC 0003's
early-free contract changes, because the frozen DAG is still a
single-use walk per call.

`grad` never appears inside a frozen trainer program: differentiation
happened at trace time, so adjoint graphs (including `Checkpoint`
region copies, which are materialized by the `grad` rewrite, not at
eval time) are already ordinary nodes in the frozen DAG.

### TypeScript: `Tensor.compile`

```ts
export interface CompileOptions {
  /** Number of runtime scalar slots (lr, step counts, ...). Default 0. */
  readonly scalars?: number
  /** Shape-cache capacity in programs. Default 32. */
  readonly cacheCapacity?: number
}

export interface CompiledFn<E, R> {
  /** One async napi call per invocation, after the first call per
      input signature pays the trace+freeze. */
  readonly call: (
    inputs: ReadonlyArray<Tensor.Any>,
    scalars?: ReadonlyArray<number>
  ) => Effect.Effect<Array<Tensor.Concrete>, Tensor.TensorError | E, R>
  /** Diagnostics: programs cached, traces performed. */
  readonly stats: () => CompileStats
  readonly dispose: () => Effect.Effect<void>
}

export const compile: <E, R>(
  build: (
    inputs: ReadonlyArray<Tensor.Lazy>,
    scalars: ReadonlyArray<Tensor.Lazy>
  ) => Effect.Effect<ReadonlyArray<Tensor.Any>, E, R>,
  options?: CompileOptions
) => Effect.Effect<CompiledFn<E, R>, never, CurrentDevice>
```

The builder receives tensor placeholders and one 0-d placeholder per
declared scalar slot; scalar values arrive as plain numbers at call time.

**Automatic shape-keyed recompilation (JAX-style).** `call` keys a
per-`CompiledFn` cache on the full input signature — shapes, dtypes,
device, and the declared scalar arity — and on a miss traces and freezes
a program for that signature. Recompiles-after-eviction are cheap: the
DAG is rebuilt but kernel binaries hit the global pipeline caches.
`stats().compiled` counts traces so accidental shape polymorphism
(feeding unbounded shape variants) is diagnosable; a warning fires past
a threshold of distinct signatures.

**Bounded cache, owned lifetime.** The cache is a bounded LRU (default
32 programs) stored in the `CompiledFn` closure — the same discipline as
`scalarLeafCache`, and deliberately *not* a global map: the cache is
reachable only through the `CompiledFn` value, so dropping the last
reference makes the whole cache collectable, and the native finalizers
release the Rust-side `Arc` graphs. Explicit `dispose()` gives
deterministic release. No WeakRef registry is needed — ownership, not
observation, defines the lifetime.

**Single-flight.** Concurrent misses on the same signature trace once:
the cache stores an in-flight `Deferred` alongside materialized
programs, so N fibers calling with the same new signature await one
trace and all receive the same program.

### `Model.compile`

```ts
export interface CompiledModel extends Model {
  /** Runs the frozen forward program: params and input in,
      materialized output out. */
  readonly execute: (
    params: Params,
    input: Tensor.Any
  ) => Effect.Effect<Tensor.Concrete, ModelError | Tensor.TensorError, CurrentDevice>
  readonly stats: () => Tensor.CompileStats
  readonly dispose: () => Effect.Effect<void>
}

export const compile: (
  model: Model,
  options?: { readonly cacheCapacity?: number }
) => Effect.Effect<CompiledModel, never, CurrentDevice>

export const isCompiled: (model: Model) => model is CompiledModel
```

A thin wrapper over `Tensor.compile`: the program's input vector is
`[...params, input]` (param shapes are fixed by the architecture, so in
practice the cache key varies only on the data shape), the output is the
single forward root. **A compiled model is still a `Model`** — crucially,
`forward` keeps the graph-builder contract (the original builder,
composable and differentiable), and the frozen program is exposed as a
separate `execute` method for evaluation loops. Substitution is total:
a compiled model composes, differentiates, trains, and checkpoints
exactly like the original.

### The `Trainer` module

Training configuration moves out of `Model` wholesale — `TrainData`,
`TrainDataSource`, `TrainStep`, `TrainConfig`, `Trained`, and `train` —
into a new `Trainer` module whose value encapsulates the configuration:

```ts
export interface Trainer<S, EL, RL, ED, RD, EO, RO> {
  readonly model: Model.Model
  readonly config: TrainConfig<S, EL, RL, ED, RD, EO, RO>
  /** The training loop, identical semantics for both forms. The initial
      parameters are the argument — omitted means `model.init`. */
  readonly train: (
    params?: Model.Params
  ) => Effect.Effect<Trained<S>, ...>
}

export interface CompiledTrainer<S, ...> extends Trainer<S, ...> {
  readonly stats: () => Tensor.CompileStats
  readonly dispose: () => Effect.Effect<void>
}

export const make: <S, ...>(
  model: Model.Model,
  config: TrainConfig<S, ...>
) => Effect.Effect<Trainer<S, ...>>

export const compile: <S, ...>(
  trainer: Trainer<S, ...>,
  options?: { readonly cacheCapacity?: number }
) => Effect.Effect<CompiledTrainer<S, ...>>

export const isCompiled: <S, ...>(
  trainer: Trainer<S, ...>
) => trainer is CompiledTrainer<S, ...>
```

The model is part of the trainer's configuration — the trainer traces
and compiles against its architecture — so `train` takes the one thing
that varies per run: the initial parameters. **A compiled trainer is
still a `Trainer`** (a `CompiledTrainer`, with `stats`/`dispose`
required; `isCompiled` narrows): its `train` method runs the same loop
with each step as one `program.run`: (params, stateRoots, input, target)
tensors in plus the step's scheduled learning rate as a runtime scalar,
(loss, newParams, newStateRoots) out — the `stateRoots`/`rebuildState`
contract (RFC 0004) is the program's input/output boundary. The compiled
step's trace is exactly the uncompiled step's graph transform:
placeholder params, state roots, input, and target in;
`[loss, ...nextParams, ...nextStateRoots]` out. Since there is one
semantic definition of a step, compiled and uncompiled loops agree
step-for-step on deterministic graphs and in distribution on stochastic
ones.

**Optimizer scalars.** Because the optimizer redesign (RFC 0004 follow-up)
made every step-varying value a tensor — the Adam step count, the SGD
`first` flag, and the per-step learning rate — a compiled trainer needs
exactly one runtime scalar slot (the learning rate); schedules from the
`LearningRate` module evaluate per step and flow through the frozen
program as data, so one program serves the whole schedule.

### Concurrency

The design brief requires a compiled function to be callable in
parallel/concurrently and behave like the original. This falls out of
three existing properties plus one new one:

1. The frozen DAG is an immutable `Arc<Node>` graph — shareable across
   calls and fibers with no synchronization.
2. Each call's `Evaluator` (cache, consumer counts, early-free) is
   per-call local state, as it already is for `eval_lazy`.
3. Native handles rely on auto `Send`/`Sync`; the only locks in the
   crate are the global kernel-pipeline caches, already contended safely
   today.
4. New: placeholder rebinding is *functional* — inputs arrive as call
   arguments, not writes into shared slots. There is no mutable slot to
   race on. (Mutable parameter slots stay rejected, per RFC 0006; here
   that rejection is what makes the concurrency story trivial.)

So a compiled forward can serve N concurrent fibers and a compiled
trainer step can run concurrently with itself (e.g. independent
replicas) with semantics identical to N sequential calls, modulo RNG
draws — exactly as N sequential walks differ today.

### Randomness

One call is one walk, and draws are keyed by node identity within a
walk: each invocation draws fresh `randn`/`uniform`/`dropout`
randomness, shared consistently across all of that call's outputs. This
is precisely the current per-step semantics, so dropout inside a
compiled training step behaves correctly with no special-casing, and
compiled-vs-uncompiled comparisons use the same "loss and gradients in
one walk" discipline as today.

## Failure modes and fallbacks

- **Materialization during trace** (`compute`/`toNumberArray` reachable
  from `build`, or an `Input` evaluated outside `run`): hard
  `TensorError` at the offending call — this is what keeps traced
  functions pure graph builders, and it fails loudly at compile time,
  never silently at run time.
- **Input signature mismatch on `run`**: call-time `TensorError` naming
  expected vs. actual shape/dtype/device per slot.
- **Fusion fallback**: unchanged from RFC 0007 — a fused node that fails
  kernel compilation at eval time falls back to composed candle ops.
  Compilation never introduces a new hard dependency: a program is an
  execution strategy over identical semantics.
- **Cache thrash**: past the LRU capacity and the distinct-signature
  warning threshold, behaviour degrades to recompile-per-call on
  alternating signatures — still correct, and visible via `stats()`.

## Non-goals

- **Symbolic dynamic shapes.** Shapes are baked at trace; varying data
  shapes are served by the shape-keyed cache. The realistic variances
  (partial last batch, different eval batch size) are handled by
  compiling per shape, dropping, or padding; variable-length workloads
  keep the established answers (fixed block size, padded buckets,
  fixed-size KV cache with masking).
- **Serialization of programs.** Node ids are process-global; a stable
  portable IR is future work.
- **Native-side step loops** (crossing the async boundary once per N
  steps, with `onStep` as a batched callback): a further overhead
  reduction, separable from program execution.
- **CUDA**: no change to its fusion status.

## Acceptance

- `xor.ts` and `nano-gpt.ts` rewritten to the
  define-model → compile-model → define-trainer → compile-trainer flow,
  training to the same tolerances as the uncompiled path.
- Parity tests: compiled vs. uncompiled forward outputs bitwise-equal on
  deterministic graphs; compiled vs. uncompiled training trajectories
  equal under seeded draws (same walk discipline); uncompiled and
  compiled `Trainer.train` agree step-for-step.
- Concurrency tests: N parallel `forward` calls equal N sequential
  calls; single-flight verified (one trace per signature under parallel
  first-calls).
- Memory tests: cache eviction frees native graphs (external-memory
  accounting unchanged); dropping a `CompiledFn` releases its programs
  via finalization.
- Benchmarks: `headtohead.ts` and a compiled-step variant showing the
  per-step graph-construction cost removed (the gap to node-mlx on small
  graphs should close substantially).

# RFC 0004: Optimizers — Pure Graph-Transform SGD, Adam, and AdamW

- **Status**: Implemented
- **Author**: Michael Arnaldi
- **Date**: 2026-07-27
- **Depends on**: RFC 0002 (autodiff), RFC 0003 (memory management) —
  training loops make bounded peak memory a hard requirement

## Summary

Add an `Optimizer` module to `@effect-torch/core` providing SGD (with
optional momentum), Adam, and AdamW. Optimizers are **pure graph
transforms**: given the current parameters, their gradients, and optimizer
state, `step` returns new parameters and new state as lazy graph values.
Nothing is mutated and nothing is materialized inside the optimizer; the
caller evaluates the updates — normally in the **same** `Tensor.compute`
walk as the loss and gradients, so one training step costs exactly one
forward and one backward pass.

## Motivation

Training is a stated core goal of the project. With autodiff (RFC 0002) we
can compute gradients; the missing piece is turning gradients into updated
parameters, repeatedly, across thousands of steps.

The two properties that matter most here are direct consequences of
decisions already taken:

1. **The update must stay in the graph.** Reading values back to JS
   (`toTypedArray`) inside a training loop would cross the async boundary
   per parameter per step, synchronizing the pipeline and forcing copies.
   Parameters must flow from one step to the next as opaque native handles.
2. **Peak memory must stay bounded across steps.** RFC 0003 guarantees
   bounded memory *within* one evaluation walk. If parameter updates were
   expressed as lazy nodes on top of the previous step's parameter *nodes*
   (rather than their materialized values), the graph would grow by one
   layer per step and memory would grow linearly with step count. State
   must therefore be re-materialized into leaves every step.

A pure-functional design satisfies both: it is also the natural fit for
Effect, mirrors MLX's `optimizer.update(...)` semantics, and keeps the
graph an ordinary value with no hidden mutable slots.

## Prior art: candle-nn optimizers

candle-nn (`candle-nn/src/optim.rs`) ships an `Optimizer` trait and three
implementations. Survey of the reference implementation we mirror
numerically:

- **`SGD`**: plain `param -= lr * grad`. Notably *no momentum* (unlike
  PyTorch). Skips vars with no grad entry and filters non-float vars at
  construction.
- **`AdamW`** (`ParamsAdamW { lr = 1e-3, beta1 = 0.9, beta2 = 0.999,
  eps = 1e-8, weight_decay = 0.01 }`):

  ```
  t       += 1
  m        = beta1 * m + (1 - beta1) * g
  v        = beta2 * v + (1 - beta2) * g²
  m_hat    = m / (1 - beta1^t)
  v_hat    = v / (1 - beta2^t)
  theta    = theta * (1 - lr * lambda)
           - lr * m_hat / (sqrt(v_hat) + eps)
  ```

  Decoupled weight decay (Loshchilov & Hutter), bias correction via
  `1/(1 - beta^t)` scalars, `eps` added to the *rooted* denominator —
  identical to PyTorch's `torch.optim.AdamW`. State `m`, `v` are per-param
  zero-initialized tensors of the same shape/dtype/device.
- Older revisions also had `Adam` (same without the decoupled decay) and
  `RMSprop`.

### Why we don't reuse it

candle-nn's optimizers are **eager and mutating**: `step(&GradStore)`
computes each intermediate tensor immediately and `var.set(...)`s new
values in place. That model is incompatible with ours in both directions:

- Adopting it would mean pulling candle-nn in as a dependency and routing
  every update through materialized tensors — one native round trip per
  intermediate op, or a bespoke eager update path that bypasses the lazy
  evaluator (and its early-free memory behavior) entirely.
- Every operation it performs (`mul`, `add`, `sqrt`, `div`, scalar
  broadcast) already exists in our node vocabulary. The update math is ~10
  ordinary graph ops per parameter. There is nothing to reuse but the
  formulas.

So: we take candle's *formulas* (they match PyTorch, which is the
compatibility target) and express them as lazy graph ops in TypeScript.

## Design

### The `Optimizer` interface

```ts
export interface Optimizer<S> {
  /** Zero-initialized (or null) state for a parameter set. */
  readonly init: (params: ReadonlyArray<Tensor.Any>) => S
  /** Pure graph transform: returns updated params and state as lazy values. */
  readonly step: (
    params: ReadonlyArray<Tensor.Any>,
    grads: ReadonlyArray<Tensor.Any>,
    state: S
  ) => { readonly params: Array<Tensor.Lazy>; readonly state: S }
}
```

`init` and `step` are synchronous and total — they only build graph nodes.
All failure modes (dtype/shape validation) are ordinary thrown errors at
graph-build time, consistent with the rest of the Tensor API, plus
`TensorError` at evaluation time.

Factories (no currying; config is captured in a plain object):

```ts
export const sgd: (config: SgdConfig) => Optimizer<SgdState>
export const adam: (config?: AdamConfig) => Optimizer<AdamState>
export const adamW: (config?: AdamWConfig) => Optimizer<AdamState>
```

### Configurations and state

```ts
export interface SgdConfig {
  readonly lr: number
  readonly momentum?: number      // default 0 (plain SGD, matches candle)
  readonly dampening?: number     // default 0
  readonly nesterov?: boolean     // default false
  readonly weightDecay?: number   // default 0; coupled L2: g += wd * param
}
export interface SgdState {
  readonly velocity: ReadonlyArray<Tensor.Any> | null // null until 2nd step
}

export interface AdamConfig {
  readonly lr?: number            // default 1e-3
  readonly beta1?: number         // default 0.9
  readonly beta2?: number         // default 0.999
  readonly eps?: number           // default 1e-8
}
export interface AdamWConfig extends AdamConfig {
  readonly weightDecay?: number   // default 0.01 (candle/PyTorch default)
}
export interface AdamState {
  readonly m: ReadonlyArray<Tensor.Any>
  readonly v: ReadonlyArray<Tensor.Any>
  readonly t: number              // step count; JS number, used for bias
                                  // correction scalars embedded in the graph
}
```

State tensors are always the params' shape/dtype/device; `t` is a JS
number. Bias-correction terms `1 / (1 - beta^t)` are computed in JS and
embedded as scalar constants — exactly what candle does with `scale_m` /
`scale_v`, and scalars are legitimate graph inputs.

### Update rules

SGD (PyTorch semantics, superset of candle):

```
if weightDecay: g += weightDecay * param
if momentum:
    v = momentum * v + (1 - dampening) * g     // v0 = g on first step
    g = nesterov ? g + momentum * v : v
param = param - lr * g
```

Adam: candle's rule minus the decay line. AdamW: candle's rule exactly,
with `weightDecay` defaulting to 0.01 and `AdamW = Adam` when
`weightDecay = 0` is *not* assumed — AdamW with wd=0 reduces to Adam
numerically, both are provided for API clarity and config compatibility.

### The training loop: one evaluation per step

The load-bearing property: because `grad` shares the forward graph and the
update ops extend the same graph further, **loss, updates, and new state
can all be roots of a single `compute` walk**. Per-walk dedup computes
the forward pass once, the backward pass once, and the update arithmetic
once.

```ts
const opt = Optimizer.sgd({ lr: 0.01, momentum: 0.9 })

const program = Effect.gen(function* () {
  let params = yield* initParams
  let state = opt.init(params)
  for (let step = 0; step < steps; step++) {
    const loss = mse(model(params, x), y)
    const grads = yield* Tensor.grad(loss, params)
    const next = opt.step(params, grads, state)
    // state tensors are flattened into the same walk:
    const roots = [loss, ...next.params, ...stateTensors(next.state)]
    const evaluated = yield* Tensor.compute(roots)
    // partition: [loss] | params | state; next step uses the materialized
    // tensors as leaves — graph depth stays O(one step)
    ...
  }
})
```

Because `compute` returns materialized `Tensor`s and every materialized
tensor wraps back into a lazy leaf (`fromMaterialized`), the next step's
graph references *values*, not the previous step's *nodes*. Graph depth
per walk is O(model depth), not O(step count) — this is what keeps RFC
0003's bounded-memory guarantee intact across a whole training run.

For convenience, a helper will be provided:

```ts
export const step: <S>(
  optimizer: Optimizer<S>,
  loss: Tensor.Any,
  params: ReadonlyArray<Tensor.Any>,
  state: S
) => Effect.Effect<
  { loss: Tensor; params: Array<Tensor>; state: S },
  GradError | TensorError
>
```

which composes `grad` + `Optimizer.step` + one `compute` + state
repacking. The low-level pieces remain public for custom loops
(gradient accumulation, clipping, multi-loss).

### Validation and dtypes

Following the strict-dtype rule (no promotion):

- All params must be float (`f32`/`f64`) — validated in `init`, mirroring
  candle's `is_float` filter but failing loudly instead of silently
  dropping.
- `params` and `grads` must have equal length and pairwise equal
  shape/dtype — validated in `step` at graph-build time.
- State created by `init` for params of dtype `D` is dtype `D`; mixing
  state from one param set into another fails on shape/dtype mismatch at
  graph-build time via existing binary-op checks.
- Unused params produce zero gradients (RFC 0002); the update for a zero
  gradient is the identity for SGD and pure moment decay for Adam — both
  harmless and consistent, so no special-casing.

### Module placement

- `packages/core/src/Optimizer.ts` — new module, exported as
  `export * as Optimizer` from `index.ts` (no re-exports across modules).
- `packages/core/test/Optimizer.test.ts` — new test file.
- No changes to the native crate: everything is composed from existing
  ops.

## Testing

1. **Hand-computed first steps** (f64 to avoid tolerance debates):
   - SGD: single param `p = [1, 2]`, grad `[0.5, -0.5]`, lr 0.1 →
     `[0.95, 2.05]`; with momentum 0.9 over 3 steps vs manual recurrence.
   - Adam/AdamW: verify the full candle formula for t=1 and t=2 against
     values computed independently (including bias correction and the
     decoupled `theta * (1 - lr * lambda)` term).
2. **Convergence**: linear regression `y = Wx` with SGD and AdamW, loss
   decreases monotonically-ish and parameters approach the ground truth;
   XOR-free tiny MSE fit for Adam.
3. **Graph-health properties**:
   - Training loop of N=50 steps: graph depth of the loss at step N equals
     depth at step 1 (state is re-materialized, no chaining).
   - RSS bounded across steps (relies on RFC 0003 early free).
   - Loss and updates evaluated in a single `compute` call produce
     identical loss values to the loss-only walk (dedup determinism).
4. **Errors**: non-float param → build-time error; mismatched params/grads
   lengths → build-time error; state from a different-shaped param set →
   build-time error.

## Drawbacks / considered alternatives

- **Mutable parameter slots (PyTorch-style in-place `param.copy_`)**: one
  fewer materialization boundary per step and slightly less garbage, but it
  punches a hole in graph immutability — a `Tensor.Lazy` would silently
  change value between evaluations, breaking the "same graph, same result"
  property and complicating RFC 0003's consumer-counting (a slot is a root
  that changes identity). Rejected; can be revisited as an optimization if
  profiling shows leaf churn matters.
- **Fused native `optim_step` op**: a single native node performing the
  whole AdamW update per param would cut op count (~10 → 1) and kernel
  launches. This is a pure optimization, invisible to the API; deferred
  until benchmarks show the update ops are a meaningful fraction of step
  time (for any non-trivial model they are dominated by matmuls).
- **More optimizers** (RMSprop, Adafactor, LAMB, Lion): the interface
  admits them trivially; deferred until needed.

## Future work

- **Learning-rate schedules**: since `lr` is a scalar embedded in the
  graph, schedules are JS functions `(step: number) => number` passed at
  step-construction time — no framework support needed beyond documenting
  the pattern, but a `Schedule` helper module may be added.
- **Gradient clipping**: global-norm clipping is a pure graph transform
  over the grads list; natural companion helper.
- **Gradient accumulation**: the low-level API already supports it
  (accumulate materialized grads across micro-batches, call `step` once).
- **State sharding (ZeRO-style)**: interacts with RFC 0001 (distributed);
  optimizer state is the largest memory component in Adam training
  (2× params), so this becomes pressing for multi-device work.

## Implementation notes (as built)

Deviations from the design above, forced by the existing API:

- **`init` and `step` return `Effect`s, they are not synchronous.** All
  tensor ops in the codebase return `Effect<Tensor.Lazy, TensorError>` (and
  `zeros` requires the `CurrentDevice` service), so
  `init: (params) => Effect<S, TensorError, CurrentDevice>` and
  `step: (...) => Effect<OptimizerUpdate<S>, TensorError>`. Validation
  failures are `TensorError`s in the error channel, not thrown — except
  invalid *configuration* (non-positive `lr`, out-of-range betas, nesterov
  without momentum), which throws from the factory, consistent with other
  construction-time misuse.
- **State materialization is part of the `OptimizerUpdate` contract, not
  the `Optimizer` interface.** `step` returns `{ params, state, stateRoots,
  rebuildState }`: `stateRoots` are the lazy tensors inside the new state
  that must be evaluated before the next step, and `rebuildState` repacks
  their materialized counterparts. This keeps `Optimizer` itself minimal
  (`init` + `step`) so user-land optimizers implement the same contract
  without fake `@internal` members — verified by a test that defines a
  custom optimizer entirely in user code. (A structural alternative —
  constraining state to a flat record and materializing it generically —
  was tried and rejected: identical runtime behavior, worse ergonomics.)
- Everything else is as designed: formulas (verified against hand-computed
  recurrences in the test suite), one `compute` walk per training step via
  the `step` helper, materialized state re-leafed each step, no native
  changes.

## Implementation plan

1. `packages/core/src/Optimizer.ts`: `Optimizer` interface, `SgdState` /
   `AdamState`, `sgd` / `adam` / `adamW` factories, `step` helper.
2. `index.ts`: `export * as Optimizer from "./Optimizer.js"`.
3. `packages/core/test/Optimizer.test.ts`: suites above.
4. README: short training-loop example using `Optimizer.step`.

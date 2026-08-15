# RFC 0005: Models — Composable, Pure-Function Sub-Graphs with Named Parameters

- **Status**: Draft
- **Author**: Michael Arnaldi
- **Date**: 2026-07-27
- **Depends on**: RFC 0002 (autodiff), RFC 0004 (optimizers) — a model is
  only useful if its parameters can be differentiated and updated

## Summary

Add a `Model` module to `@effect-torch/core`. A **model** is a pure value
pairing parameter *construction* with a parameterised *forward* graph
builder:

```ts
export interface Model<P extends ReadonlyArray<Tensor.Any>> {
  readonly names: ReadonlyArray<string>
  readonly init: Effect.Effect<P, Tensor.TensorError, Device.CurrentDevice>
  readonly forward: (
    params: P,
    input: Tensor.Any
  ) => Effect.Effect<Tensor.Lazy, Tensor.TensorError>
}
```

Primitives (`linear`, `tanh`, `sigmoid`) are models; `chain` composes
models into a model with the **same interface**, so sub-models compose
recursively. Parameters are always a **flat tuple of tensors**, which makes
the existing training path (`Tensor.grad`, `Optimizer.step`) work on any
model with zero adapter code. Parameter specs give every parameter a stable,
checkpoint-friendly identity that maps directly onto `Tensor.save` /
`Tensor.load`.

There is no backward mode and no mutable module state: the forward graph
is an ordinary lazy graph, so `Tensor.grad` differentiates it as-is.

## Motivation

Every training example so far (see `packages/examples/xor.ts`) hand-rolls
the same pattern: a `Params` tuple type, a `forward(params, x)` function, a
`createModel` effect that initializes each parameter, and manual name
management for checkpoints. The pattern is sound — it is exactly the
Flax/Haiku design — but without a library form:

1. **Composition is manual.** Stacking layers means writing a new
   hand-destructured tuple type and a new forward function per model. The
   type-level cost of "add one layer" should be zero.
2. **Parameter bookkeeping is manual.** Naming params for `Tensor.save`,
   counting them for `Tensor.grad`, keeping init order aligned with forward
   destructuring — all error-prone busywork the library can own.
3. **The design space needs pinning down.** PyTorch-style mutable modules
   are incompatible with our graph semantics (same reasoning as RFC 0004's
   rejection of mutable parameter slots). Writing down *why* the pure
   design is the only one that fits prevents re-litigating it.

## Prior art

- **PyTorch `nn.Module`**: mutable object, parameters registered as
  side effects, `forward` reads `self`. Rejected: parameter values would
  silently change graph meaning between evaluations, breaking "same graph,
  same result" and complicating RFC 0003's consumer counting. Also
  requires a mode switch (`train()`/`eval()`) we don't need.
- **Flax / Haiku (JAX)**: modules are pure; `init` returns a parameter
  pytree, `apply(params, x)` is a pure function. This is the design we
  adopt, flattened: our params are a tuple, not a tree (see alternatives).
- **MLX `nn.Module`**: eager-API facade over lazy arrays; parameters are
  mutable leaves with `update()` replacing values in place. Closest to
  usable in our world, but still relies on in-place mutation.
- **Keras `Sequential`**: the user-facing shape we want
  (`chain(a, b, c)`), without the mutable internals.

## Design

### The `Model` interface

```ts
export interface Model {
  readonly parameterSpecs: ReadonlyArray<ParameterSpec>
  /** Extends the graph: parameters and input in, lazy output out. */
  readonly forward: (
    params: P,
    input: Tensor.Any
  ) => Effect.Effect<Tensor.Lazy, Tensor.TensorError>
}

export interface ParameterSpec {
  readonly name: string
  readonly shape: ReadonlyArray<number>
  readonly initializer: ParameterInitializer
}
```

Notes:

- `P` is constrained to a **tuple of tensors** — "expect a tuple, ask for
  a tuple". Configuration (feature counts, activation choice) lives in the
  closure of the factory, never in `P`.
- `Model.initialize` returns **lazy** tensors: it interprets the declarative
  initializer in each parameter spec and is pure graph-building, consistent
  with every other constructor. Materialization happens in the first
  `Optimizer.step` walk (or an explicit `Tensor.compute`), so initial
  `randn` draws are consistent with the first loss within that walk.
- `forward` is single-input, single-output. DAGs (skip connections,
  multi-head) are explicitly out of scope (see Future work).
- Both `initialize` and `forward` return `Effect`s, like the rest of the Tensor
  API; failures are `TensorError`s in the error channel.

### Primitives

```ts
export const linear: (
  name: string,
  inFeatures: number,
  outFeatures: number
) => Model<readonly [weight: Tensor.Any, bias: Tensor.Any]>

export const tanh: Model<readonly []>
export const sigmoid: Model<readonly []>
export const relu: Model<readonly []>
```

- `linear("fc1", 2, 8)` has specs named `"fc1.weight"` and `"fc1.bias"`,
  `forward = add(matmul(input, w), b)`.
- Weight init: `randn([in, out]) * (1 / sqrt(in))`, bias zeros. (PyTorch
  uses kaiming-uniform with the same bound; we have no `uniform` op yet —
  scaled `randn` is adequate for v1, noted under Future work.)
- Activations are parameterless models (`parameterSpecs = []`, `forward` is the
  op). This uniformity is what makes `chain` total: every stage of a
  network is the same kind of value.
- `relu` is backed by a native `Relu` node (single kernel,
  `maximum(x, 0)`) with an adjoint of `g * cast(gt(x, 0))` — subgradient
  `0` at `x = 0`.

### Composition: `chain`

```ts
export const chain: <Ms extends ReadonlyArray<Any>>(
  ...models: Ms
) => Model<ParamsOf<Ms>>
```

with the parameter tuple computed at the type level:

```ts
export type ParamsOf<Ms> = Ms extends readonly [infer H, ...infer T]
  ? H extends Model<infer P>
    ? T extends ReadonlyArray<Any>
      ? readonly [...P, ...ParamsOf<T>]
      : readonly [...P]
    : readonly []
  : readonly []
```

Runtime behaviour:

- Parameter specs are concatenated in child order. **Duplicate names
  throw at construction time** — a name collision would silently overwrite
  entries in `Tensor.save`'s record, so it is construction-time misuse,
  consistent with invalid optimizer configs throwing from the factory.
- `forward` threads the input through each child in order, slicing each
  child's share of the parameter tuple by `parameterSpecs.length` (the arity).
- At least one model is required (an empty `chain` would be an identity
  with an ambiguous output type; not worth the special case).

Example — the xor network, library form:

```ts
const model = Model.chain(
  Model.linear("fc1", 2, 8),
  Model.tanh,
  Model.linear("fc2", 8, 1),
  Model.sigmoid
)
// Model<readonly [w1, b1, w2, b2]>
```

### Training: nothing new

Because `P` is already a flat tuple of tensors, the approved
RFC-0004 path applies verbatim:

```ts
const params = yield* Model.initialize(model)
const state = yield* optimizer.init(params)
// per step:
const loss = yield* Tensor.mse(yield* model.forward(params, x), y)
const next = yield* Optimizer.step(optimizer, loss, params, state)
// next.params: Materialized<P> — the same tuple, straight back into forward
```

`Tensor.grad(loss, params)` needs no model awareness; `Optimizer.step`'s
`Materialized<P>` tuple-in/tuple-out guarantee means the updated params
feed directly back into `model.forward`. There is deliberately **no**
`Model.trainStep` helper — it would be a one-line alias for
`Optimizer.step` and would imply a coupling that doesn't exist.

### Serialization

```ts
export const save: (
  model: Any,
  params: ReadonlyArray<Tensor.Any>,
  path: string
) => Effect.Effect<void, Tensor.TensorError>

export const load: (
  model: Any,
  path: string
) => Effect.Effect<Array<Tensor.Concrete>, Tensor.TensorError, Device.CurrentDevice>
```

`save` zips names from `model.parameterSpecs` with the param tuple into the record
`Tensor.save` already takes (arity mismatch throws — misuse). `load` reads
the file with `Tensor.load` and returns tensors in parameter-spec order;
missing keys fail with a `TensorError`. Shape/dtype validation against the
architecture is left to graph-build-time checks on first use.

## Validation and errors

- Duplicate names in `chain`, wrong-arity params in `save` → thrown
  defects at construction/call time (misuse, not runtime failure).
- Missing keys in `load` → `TensorError` in the error channel.
- Shape/dtype errors in `forward` → existing `TensorError` from the
  underlying ops.
- Non-float parameters are not special-cased: `Optimizer.init` already
  rejects them.

## Module placement

- `packages/core/src/Model.ts` — new module; `export * as Model` from
  `index.ts` (no re-exports across modules).
- `packages/core/test/Model.test.ts` — new test file.
- `packages/examples/xor.ts` — rewritten on `Model.chain` as the proof
  case; the manual `Params` tuple disappears.
- Native: one new `Relu` node (eval + adjoint), already landed alongside
  this RFC.

## Testing

1. **Type level**: `chain` infers the concatenated tuple
   (`linear, tanh, linear` → `readonly [w1, b1, w2, b2]`); `ParamsOf`
   handles parameterless models and nested chains.
2. **Composition correctness**: a chained 2→8→1 tanh/sigmoid MLP produces
   identical outputs and gradients to the hand-written xor `forward`
   given the same parameter values.
3. **Arity/slicing**: mixed parameterless/parameterised chains slice the
   param tuple correctly (forward output matches per-layer manual
   application).
4. **Names**: concatenation order; duplicate names throw; `names.length`
   always equals the runtime tuple length.
5. **Serialization**: `save`/`load` round-trip preserves values and
   order; loading a file with missing keys fails with `TensorError`;
   loading params saved from a *different* architecture fails at
   graph-build time.
6. **Training integration**: xor trained end-to-end via
   `Model.chain` + `Optimizer.step` converges (the rewritten example
   doubles as the smoke test).

## Drawbacks / considered alternatives

- **Nested parameter structures** (per-layer sub-tuples/records with a
  flatten/rebuild contract, mirroring `stateRoots`/`rebuildState`):
  strictly more expressive (layers could carry non-tensor data in `P`),
  but every layer author writes flatten/rebuild boilerplate, the type
  algebra is worse, and `Optimizer.step` would need an adapter. Rejected:
  trainable state *is* a list of tensors; anything else is config and
  belongs in the closure.
- **Structural walking** (reflect over arbitrary `P` trees to find
  tensors): zero boilerplate but constrains `P` to pure tensor trees
  implicitly and hides errors until runtime. The flat-tuple constraint
  says the same thing in the type system instead.
- **Mutable modules (PyTorch-style)**: rejected, per RFC 0004's
  argument against mutable parameter slots — identical reasoning.
- **Auto-indexed names** (`"0.weight"`, PyTorch-style): less to type,
  worse checkpoints (renumbering when inserting a layer invalidates
  saved files). Explicit names keep checkpoints stable under architecture
  edits; indices are available to users who want them.
- **`forward` taking a record of named inputs** (multi-input models):
  punts on the hard part (typing heterogeneous input records through
  `chain`) for a use case we don't have yet. Deferred.

## Future work

- **DAG models** (skip connections, multi-input/output): needs a
  graph-of-models abstraction; likely designed alongside RFC 0001
  (distributed execution), where placement also becomes per-subgraph.
- **Stateful models** (batchnorm running stats, dropout schedules): the
  `stateRoots`/`rebuildState` contract from RFC 0004 generalizes — a
  model gains a second tuple of non-trainable state tensors. Deferred
  until a concrete layer needs it.
- **Elementwise `where` native op** → `gelu`, residual clipping,
  piecewise activations.
- **`uniform` constructor** for kaiming/xavier-faithful init.
- **`Model.map` / parameter transforms** (freeze a sub-model by
  `stopGradient` on its params, fine-tuning workflows).

## Implementation plan

1. `packages/core/src/Model.ts`: `Model` interface, `Any`, `ParamsOf`,
   `linear`, `tanh`, `sigmoid`, `relu`, `chain`, `save`, `load`.
2. `index.ts`: `export * as Model from "./Model.ts"`.
3. `packages/core/test/Model.test.ts`: suites above.
4. Rewrite `packages/examples/xor.ts` on `Model.chain`.
5. README: model section with the chained xor example.

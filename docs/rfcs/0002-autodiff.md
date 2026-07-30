# RFC 0002: Reverse-Mode Automatic Differentiation

- **Status**: Draft
- **Author**: Michael Arnaldi
- **Date**: 2026-07-27
- **Depends on**: RFC 0001 (distributed execution) for collective adjoints
- **Blocks**: RFC 0001 Phase 2 (distributed training)

## Summary

Add reverse-mode automatic differentiation to effect-torch: a `grad` /
`grad` API whose graph transformation runs **entirely in Rust**. The
native backend walks the forward `LazyNode` graph, applies per-op adjoint
rules, and returns the backward graph as ordinary `LazyNode`s — evaluated by
the same executor with the same dedup, cancellation, and device semantics.

TypeScript remains a thin layer: it validates the API contract (scalar loss,
float dtype) and forwards to one native `grad` call. No tape, no per-op
recording overhead, no gradient math in JS.

## Motivation

Training is a stated goal of effect-torch, and training is gradient descent:
`loss.forward → backward → optimizer step`. Everything else in the training
stack — optimizers (SGD/Adam), LR schedules, gradient clipping, distributed
gradient synchronization (RFC 0001 Phase 2 DDP/FSDP) — builds on `grad`.

The lazy graph architecture makes autodiff unusually clean:

- **No mutation, no versioning problems.** Nodes are immutable `Arc` values;
  a backward rule can safely reference forward nodes (e.g. `mul`'s adjoint
  needs both operands) with zero lifetime or aliasing concerns.
- **Forward is computed once.** The backward graph references forward nodes;
  the executor's per-walk dedup guarantees shared forward computation happens
  a single time even though it appears in many adjoint expressions.
- **Backward is just more graph.** Fusions, device placement, and
  (eventually) collectives apply uniformly to forward and backward.
- **Native is fast.** Graph traversal, adjoint construction, and cotangent
  accumulation are pointer and `HashMap` operations in Rust; JS only ever
  crosses the napi boundary once per `grad` call.

## Prior Art

### PyTorch (tape-based autograd)

Eager tensors record operations on a dynamic tape (`grad_fn` graph) as they
execute; `backward()` walks it in reverse topological order. The tape exists
because eager execution has no whole-graph view. We have the whole graph
already — our equivalent of the tape is the `LazyNode` graph itself, and
"backward()" is a pure graph-to-graph transform.

Source: https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

### JAX (functional transformations)

`jax.grad(f)` is a pure function transformation over a traced jaxpr. We adopt
the scalar-loss
enforcement, `wrt` argument selection) but skip tracing: our "trace" is the
graph the function already built.

Source: https://docs.jax.dev/en/latest/jax.html#automatic-differentiation

### MLX (`mx.grad`)

The closest model: `mx.grad(f)` / `mx.value_and_grad(f)` over lazy arrays,
scalar-output enforcement, `stop_gradient`, gradient graphs that are
themselves lazy arrays (higher-order derivatives work by applying `grad`
again), and differentiable collectives (`all_sum`'s adjoint is `all_sum`) —
the contract RFC 0001 records.

Source: https://ml-explore.github.io/mlx/build/html/python/autodiff.html

### candle

candle ships its own reverse-mode (`Var`, `backprop`), but it operates on
eager candle tensors. We deliberately do **not** use it: our graph lives
above candle (per-op `LazyNode`s, custom collectives), and candle's autograd
would see only individual kernel launches, not our graph.

## Design

### Guiding principles

1. **Grad operates on graph values, not functions.** JAX/MLX wrap a function
   because they trace it; our lazy graph is already a first-class value, so
   `grad(loss, wrt)` transforms the graph directly — no tracing, no
   `requires_grad` flag, no gradient accumulation into tensors. Gradients are
   plain lazy tensors returned to the caller.
2. **The transform is native.** One napi call: `grad(loss, wrt) → grads`.
   Traversal, adjoint construction, and cotangent accumulation all happen in
   Rust. TS never sees graph internals.
3. **Adjoints reuse the existing node vocabulary.** Backward rules compose
   existing `LazyNode` variants (`Mul`, `Sum`, `BroadcastTo`, `Permute`, …).
   Backward graphs get every future executor optimization for free, and
   higher-order derivatives work by running the same transform again.
4. **Scalar loss enforcement (v1).** `grad` requires a 0-d
   tensor. Vector-Jacobian products come later via an explicit `vjp` API.

### Prerequisite: static metadata on `LazyNode`

The transform needs shapes and dtypes **at graph-build time** (broadcast
adjoints must know the target shape; dtype checks must reject integer
paths). Today shapes are tracked in TS (`Tensor.Any.shape`) and computed
in Rust only during evaluation. Change:

```rust
pub struct Node {
    pub id: u64,            // globally unique, from an atomic counter
    pub shape: Vec<usize>,  // static, computed at construction
    pub dtype: DType,       // static
    pub kind: NodeKind,     // the current LazyNode enum, renamed
}
```

- TS already computes output shapes for every op; it passes them into native
  constructors, which validate (defense in depth) and store them.
- `id` gives every node a stable identity for cotangent accumulation
  (`HashMap<u64, Arc<Node>>`) and lets TS name the `wrt` leaves.
- Constructors keep returning `Tensor.Lazy`; the wrapper gains cheap `id` /
  `shape` / `dtype` getters (TS already carries shape/dtype; `id` is new).

This refactor is independently valuable: shape errors surface at graph-build
time instead of at evaluation, and the executor's shape inference code is
computed once at construction rather than on every walk.

### The native transform

```rust
pub fn grad(loss: &Node, wrt: &[u64]) -> Result<Vec<Arc<Node>>>
```

1. **Topological sort**: iterative post-order walk from `loss`, dedup by
   `id`. The loss must be scalar and float; every `wrt` leaf must be float.
2. **Reverse accumulation**: iterate the sorted list in reverse, maintaining
   `cotangents: HashMap<u64, Arc<Node>>`. Seed: `cotangents[loss.id] =
   Ones([], loss.dtype)`. For each node, distribute its accumulated
   cotangent to its inputs per the adjoint table, summing contributions
   (`Add`) when a node feeds multiple consumers. **Cotangents are not
   distributed through non-float nodes** (comparisons, integer arithmetic):
   their gradient is zero almost everywhere (PyTorch semantics), so the
   walk simply drops them.
3. **Result**: for each id in `wrt`, return the accumulated cotangent, or
   `Zeros(shape, dtype)` if the leaf is not on the path (JAX behavior:
   unused inputs get zero gradients, not an error).

The transform is a pure function over immutable graphs; it neither mutates
the forward graph nor holds evaluation state.

### Adjoint rules

`sum_to_shape(g, shape)` below: sum `g` over the leading extra dims and over
dims where `shape` has 1, then reshape — composed from existing `Sum` +
`Reshape` nodes (native now knows all shapes statically).

| Forward | Backward (g = cotangent of output) |
| --- | --- |
| `Add(a, b)` | `sum_to_shape(g, a)`, `sum_to_shape(g, b)` |
| `Sub(a, b)` | `sum_to_shape(g, a)`, `sum_to_shape(neg(g), b)` |
| `Mul(a, b)` | `sum_to_shape(g * b, a)`, `sum_to_shape(g * a, b)` |
| `Div(a, b)` | `sum_to_shape(g / b, a)`, `sum_to_shape(-g * a / b², b)` |
| `Neg(a)` | `neg(g)` |
| `Abs(a)` | `g * sign(a)` where `sign(a) = cast(gt(a,0)) - cast(lt(a,0))` |
| `Sqrt(a)` | `g * 0.5 / out` |
| `Exp(a)` | `g * out` |
| `Log(a)` | `g / a` |
| `Sin(a)` | `g * cos(a)` |
| `Cos(a)` | `neg(g * sin(a))` |
| `Pow(a, c)` | `g * c * pow(a, c - 1)` (scalar exponent only, as today) |
| `Matmul(a, b)` | `matmul(g, transpose(b))`, `matmul(transpose(a), g)` |
| `Sum(dims, keepdims)` | `BroadcastTo(reshape(g))` to input shape |
| `Mean(dims, keepdims)` | as `Sum`, divided by element count |
| `Max/Min(dims, keepdims)` | `g * mask / sum(mask)` where `mask = eq(a, broadcast(out))` (ties split evenly, matching PyTorch) |
| `Reshape` | `Reshape(g)` to input shape |
| `Permute(perm)` | `Permute(g, inverse(perm))` |
| `Slice(d, start, end)` | `Concat(zeros(start), g, zeros(n - end), dim = d)` |
| `Concat(d, xs…)` | `Slice(g, d, …)` per input |
| `BroadcastTo` | `sum_to_shape(g, input shape)` |
| `Cast` | `Cast(g)` back to input dtype (float targets only) |
| `StopGradient` | none — cotangent is dropped, walk does not continue past it |
| `Randn`, `Arange`, `Eye`, `Zeros/Ones/Full`, `FromBytes`, parameter leaves | constant leaves: no adjoint |
| RFC 0001 collectives | per the backward-rules table in RFC 0001 |

Rules referencing forward outputs (`Sqrt`, `Exp`) reuse the forward `Arc`
directly — no recomputation, thanks to executor dedup.

Every adjoint is itself built from differentiable nodes, so differentiating a gradient graph
(higher-order derivatives) works structurally from day one.

### TS API

Unlike JAX/MLX there is **no function transformation**: the lazy graph is
already a first-class value, so `grad` operates on it directly — no tracing,
no currying.

```ts
// Gradients of a scalar (0-d) float `loss` w.r.t. the given tensors.
export const grad: (
  loss: Tensor.Any,
  wrt: ReadonlyArray<Tensor.Any>
) => Effect<Array<Tensor.Lazy>, GradError>

// Barrier: adjoint is dropped, reverse walk stops here
export const stopGradient: (
  self: Tensor.Any
) => Effect<Tensor.Lazy, TensorError>

export class GradError extends Data.TaggedError("GradError")<{
  readonly reason:
    | "non-scalar-output"
    | "non-float-dtype"
    | "not-differentiable"
  readonly detail: string
}> {}
```

`grad` validates the scalar/float contract in TS (cheap, shapes are already
there), then calls the native `grad` once. Op signatures are unchanged —
there is no tape and nothing threaded through context. There is no
`valueAndGrad`: the caller already holds the loss value, and evaluating it
together with the gradients is what `evaluateAll` is for.

### Single evaluation, and the `randn` hazard

`randn` re-randomizes per `compute` call by design. Therefore the loss and
all gradients returned by `grad` must be evaluated together with the loss in **one**
executor walk, or the reported loss and the gradients would come from
different random draws. `grad` returns lazy roots;
the required destructor is a multi-root evaluate:

```ts
export const evaluateAll: (
  roots: ReadonlyArray<Tensor.Any>
) => Effect<Array<Tensor>, TensorError>
```

Native side this is a single walk with multiple roots sharing one dedup
cache. `evaluateAll` is generally useful (validation metrics alongside loss)
and is part of this RFC's deliverables.

### What is deliberately absent

- **No `requires_grad`, no in-place accumulation.** Gradients are values,
  not side effects. An optimizer (future RFC) is a pure function
  `(params, grads, state) => (params', state')`.
- **No mutation of the forward graph.** The transform allocates a fresh
  backward graph referencing shared forward `Arc`s.

## Scope

### In scope (this RFC)

- `Node` refactor: `id`, static `shape`/`dtype` on every native node; TS
  passes shapes into constructors; native validates.
- Native `grad` transform + `StopGradient` node variant.
- All adjoint rules in the table above.
- `evaluateAll` (multi-root evaluation).
- TS wrappers: `grad`, `stopGradient`, `GradError`.
- Gradcheck test harness: finite-difference verification of every rule in
  f64 over random small shapes, plus a tiny end-to-end training loop
  (linear regression on CPU) asserting loss decreases.

### Out of scope (deferred)

- `vjp`/`jvp` APIs for non-scalar outputs and forward mode.
- Optimizers, LR schedules, gradient clipping (separate RFC; depends on
  this one).
- Gradient checkpointing / rematerialization (memory/compute tradeoff).
- Higher-order derivatives: supported structurally but not tested/guaranteed
  in v1.
- `batch`/`vmap` (needed for efficient per-example gradients).
- Differentiating through `toTypedArray`/readback (nonsensical; hard error).

## Testing strategy

- **Gradcheck**: for each differentiable op, compare `grad` output against
  central finite differences `(f(x+ε) − f(x−ε)) / 2ε` in f64, tolerance
  ~1e-6, across broadcast shape pairs, reduce dim/keepdims combinations, and
  slice/concat boundaries.
- **Broadcast-heavy cases**: `sum_to_shape` is the most error-prone rule;
  dedicated tests for `add/mul/div` with mismatched ranks.
- **Contract enforcement**: non-scalar outputs and integer `wrt` dtypes
  produce the expected `GradError`; comparisons on the path yield zero
  gradients; unused `wrt` inputs return zeros.
- **randn consistency**: a loss graph using `randn`, differentiated and evaluated
  loss and grads from the same draw (verified by reconstructing the forward
  result from the gradient graph).
- **End-to-end**: linear regression and a 2-layer MLP on CPU, fixed seed
  data, loss must decrease monotonically for N steps and match a
  hand-computed first step.

## Risks and open questions

1. **`Node` refactor churn.** Every native constructor signature changes to
   accept `shape` (and the enum moves under `NodeKind`). Mechanical, but
   touches all of `lib.rs` and every TS op wrapper. Worth doing before the
   op vocabulary grows further.
2. **Memory: backward graphs pin the entire forward graph.** Standard for
   reverse mode without checkpointing; noted, checkpointing deferred.
3. **Graph size.** Backward roughly doubles node count; the executor's
   per-walk dedup is a HashMap keyed by node pointer today — keying moves to
   `id`. Fine at Phase-1 scale; may need attention for deep models.
4. **max/min ties semantics.** Chosen: split evenly (PyTorch behavior).
   Alternative: route to first maximal index (needs argmax + scatter — ops
   we don't have). Even-split is expressible today; revisit if it surprises.
5. **`pow` with negative base and non-integer exponent** produces NaN
   gradients; acceptable (matches PyTorch), documented.
6. **Duplicated work across separate `grad` calls.** Each `grad` call
   re-derives backward from the forward graph; if a training loop rebuilds
   the graph per step (it will, like MLX), that's inherent. Graph caching /
   compilation is a separate concern.

## References

- MLX autodiff: https://ml-explore.github.io/mlx/build/html/python/autodiff.html
- JAX autodiff: https://docs.jax.dev/en/latest/jax.html#automatic-differentiation
- PyTorch autograd: https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
- candle autograd (deliberately unused): https://github.com/huggingface/candle/tree/main/candle-examples
- RFC 0001 (collective backward rules): docs/rfcs/0001-distributed-execution.md

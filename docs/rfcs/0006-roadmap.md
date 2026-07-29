# RFC 0006: Roadmap — Model Abstractions and Remaining Work

- **Status**: Draft
- **Author**: Michael Arnaldi
- **Date**: 2026-07-28
- **Depends on**: RFC 0004 (optimizers), RFC 0005 (models) — this document
  orders work that builds on both

## Summary

The tensor, autodiff, loss, optimizer, and scheduling layers are complete
and committed. This RFC records the ordered plan for what remains: the
`Model` abstraction (three stages), distributed execution, and a list of
smaller items to slot in as they become load-bearing. It exists so that
sequencing decisions (and their rationale) are written down once, not
re-derived per session.

## Completed foundation (for reference)

- Lazy computation graph with a single async boundary, iterative native
  evaluator with early-free, external-memory reporting, explicit `dispose`.
- Full tensor op surface: constructors, elementwise, reductions, shape
  ops, indexing (`take`/`gather`/`scatterAdd`, differentiable), NN
  functional (activations, softmax, dropout), convolution and pooling via
  im2col, transposed convolution, linalg (`inverse`/`det`/`solve`).
- `Gradient`: `grad`, `stopGradient`, `checkpoint` (native recompute),
  `vjp`/`jvp` (value-based), `vmap` (native graph rewrite with per-op
  batching rules).
- `Loss`: regression and classification losses with mean/sum/none
  reduction.
- `Optimizer`: pure graph-transform SGD/Adam/AdamW, one `compute` walk
  per training step, fused native AdamW update node (default),
  `clipByValue`/`clipByGlobalNorm`.
- `LearningRate`: constant/exponential/stepwise/cosine/warmup schedules.
- Metal allocator degradation fixed via a patched candle fork, pinned by
  revision.

## Ordered work

### 1. Model module (RFC 0005)

The last structural piece of the single-machine framework. A model is a
pure value pairing parameter construction with a parameterised forward
graph builder; primitives and chains share one interface.

Scope:

- `Model<P>` interface: `names`, `init`, `forward`; params always a flat
  tuple of tensors.
- `chain` with variadic tuple typing (`ParamsOf`), runtime slicing by
  arity; duplicate names throw at construction.
- `linear` (kaiming-scale init now that `uniform` exists), activation
  models (`tanh`, `sigmoid`, `relu`, `silu`, `gelu`), `flatten`, `dropout`
  as a model.
- `save`/`load` zipping `names` onto `Tensor.save`/`load`.
- xor example rewritten on `Model.chain` as the proof case.

Exit criteria: RFC 0005 marked Implemented with as-built notes; xor
trains 4/4 via the library path; type-level tests for `ParamsOf`.

### 2. Stateful models

Layers with non-trainable runtime state — batchnorm's running mean/var
being the canonical case. The state contract already exists: RFC 0004's
`stateRoots`/`rebuildState` generalizes — a stateful model carries a
second tuple of non-trainable tensors, re-materialized per step alongside
optimizer state.

Scope:

- `StatefulModel<P, S>` (or an extension of the `Model` interface with a
  state tuple; decide during implementation, record in RFC 0005's notes).
- `batchNorm` (training/inference statistics, momentum for running
  averages) as the first stateful layer.
- Serialization covering both param and state tuples.

### 3. DAG models

Multi-input/multi-output topologies: skip connections, shared
sub-modules, multi-head outputs. A `chain` cannot express these.

Scope:

- Named-node graph of models with explicit input/output wiring,
  runtime-validated at construction (cycle check, arity check). Full
  static typing of heterogeneous DAGs is not practical; type the common
  cases and validate the rest.
- Parameter naming already supports this (`names` are hierarchical).
- Design should anticipate RFC 0001: subgraphs of a DAG are the natural
  unit for device placement.

### 4. Distributed execution (RFC 0001)

Multi-device and multi-process training. Genuinely multi-session; RFC
0001 exists and needs an implementation plan refresh against the current
evaluator (iterative walk, early-free, fused nodes). Expected shape:
process-group initialization, device-mesh description, collective ops
(all-reduce over gradients as graph nodes), data-parallel training loop
first, sharding later. NCCL feature flag already exists in the native
crate (`build:nccl`).

## Slot-in items (pull forward when load-bearing)

- **Native conv kernels**: candle's `conv1d`/`conv2d` as a performance
  swap for im2col; acceptance = the existing reference-implementation
  tests. Also the adjoint question (transposed-conv backward, which we
  already have composed).
- **Batched linalg**: rank ≥ 3 `inverse`/`det`/`solve` (currently
  rank-2 only); blocked only on loop-and-stack in the eval arm.
- **vmap coverage**: data-dependent indexing, `gather`/`scatterAdd`
  (currently rejected under vmap).
- **Fused SGD node**: same pattern as fused AdamW, for momentum SGD.
- **True kernel fusion** (Metal/CUDA shaders): the real version of what
  the fused AdamW node approximates graph-wise. Large; only when
  profiling justifies.
- **`Model.load` shape validation** against the architecture without
  drawing random init values.

## Explicitly out of scope (recorded to avoid re-litigating)

- **Mutable parameter slots** (PyTorch-style in-place updates): rejected
  in RFC 0004; graph immutability is load-bearing for the one-walk
  training step and for `checkpoint`/`vmap` graph transforms.
- **Eager execution mode**: the lazy graph is the design; see the
  project history for the full argument (single async boundary, dedup,
  early-free, graph transforms).
- **Tape-based autodiff**: rejected with the same reasoning; backward is
  a graph transform, not a recorded trace.

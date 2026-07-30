# RFC 0001: Distributed Execution over Multiple GPUs/CPUs

- **Status**: Draft
- **Author**: Michael Arnaldi
- **Date**: 2026-07-27

## Summary

Add distributed execution to effect-torch: collective communication ops
(all-reduce, all-gather, broadcast, send/recv) as first-class nodes in the
lazy computation graph, a `ProcessGroup` service for rank/world management,
and sharding helpers for tensor-parallel and weight-sharded execution across
multiple GPUs and CPUs.

The design treats **training as a first-class goal**: effect-torch aims to be
a fully-fledged tensor framework, and distributed *training* is where
collectives matter most. Autodiff does not exist yet (tracked as a separate
RFC), so this RFC is phased: Phase 1 ships forward-pass collectives that are
immediately useful for inference and become the building blocks for training;
Phase 2 adds gradient synchronization once autodiff lands. Crucially, every
collective is specified **with its backward rule** from day one so Phase 1
choices never have to be revisited.

## Motivation

Single-device execution limits model size and throughput. The standard
remedies:

- **Tensor parallelism (TP)** — shard each matmul's weights across devices so
  a model that doesn't fit (or is too slow) on one GPU runs on several.
- **Data parallelism (DP)** — replicate the model, shard the batch across
  ranks; for training, gradients are all-reduced so replicas stay in sync
  (DDP).
- **Weight-sharded streaming (FSDP)** — shard weights, gradients, and
  optimizer state; all-gather weights before use, reduce-scatter gradients
  after backward.

All of these reduce to a small set of collectives: **all-reduce, all-gather,
reduce-scatter, broadcast, send/recv**. Our lazy graph is a natural fit:
collectives become graph nodes scheduled by the same executor, exactly as
MLX does in `mx.distributed`.

## Prior Art

### MLX (`mx.distributed`)

Collectives are **lazy arrays**, not side effects:

- `all_sum`, `all_max`, `all_min` (all-reduce variants)
- `all_gather`, `sum_scatter`
- `send` / `recv` / `recv_like` (point-to-point)
- `Group` object with `rank()`, `size()`, `split(color, key)`
- No-op when group size is 1 — programs are written rank-agnostic

Backends:

| Backend | Transport | Notes |
| --- | --- | --- |
| `ring` | TCP sockets | Always available, no deps; used for Thunderbolt rings |
| `mpi` | MPI (dlopen) | Full-featured, arbitrary send/recv, `mpirun` launch |
| `jaccl` | Thunderbolt RDMA | macOS 26.2+, ~10× lower latency than ring |
| `nccl` | NVIDIA NCCL | CUDA builds |

Sharding helpers: `mx.nn.layers.distributed.shard_linear` (column/row-parallel
Linear) and `PipelineMixin` in mlx-lm (stage split + p2p). Notably,
**node-mlx does not expose the distributed module** — a JS binding with
collectives would be ahead of the MLX JS ecosystem.

Sources:
- https://ml-explore.github.io/mlx/build/html/python/distributed.html
- https://ml-explore.github.io/mlx/build/html/usage/distributed.html

### PyTorch (`torch.distributed`)

| Strategy | Forward collectives | Backward collectives |
| --- | --- | --- |
| DDP | none (init broadcast) | all-reduce (bucketed) |
| FSDP | all-gather | all-gather + reduce-scatter |
| Tensor parallel (Megatron) | all-reduce | all-reduce |
| Pipeline parallel | send/recv | send/recv |

Megatron rules for forward: **column-parallel linear = no collective**,
**row-parallel linear = all-reduce**. An MLP block (column-parallel →
activation → row-parallel) needs exactly one all-reduce per forward pass.

Backend capability: NCCL (GPU-only, best perf), Gloo (CPU-first, full
collectives except all_to_all), MPI (optional).

Sources:
- https://arxiv.org/abs/1909.08053 (Megatron-LM)
- https://docs.pytorch.org/docs/stable/distributed.html
- https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html

### JAX (`jax.sharding`)

Compiler-oriented SPMD: `Mesh` (named device axes) + `PartitionSpec`
(sharding annotation per tensor dim). The XLA partitioner propagates layouts
and inserts collectives automatically. Sharding is a *type-level property of
arrays*; one mechanism uniformly expresses DP, TP, and pipeline-ish
parallelism. We borrow the vocabulary (`Mesh`, sharding specs) without
building a compiler.

Source: https://docs.jax.dev/en/latest/jax.sharding.html

### candle (our backend)

- `Device::new_cuda(ordinal)` supports multiple GPUs; cross-device ops are
  rejected, movement is explicit `to_device`.
- The `nccl` feature (merged Oct 2025) is a passthrough to **cudarc's NCCL
  module**, which exposes a safe `Comm` with all-reduce, all-gather,
  broadcast, send/recv over `CudaSlice`/`CudaView` — the same types
  `CudaStorage` wraps.
- There is **no CPU collective backend** in candle. All downstream tensor
  parallelism (mistral.rs, candle-vllm, atoma-infer) is implemented above
  candle-core using cudarc NCCL.

Sources:
- https://github.com/huggingface/candle/pull/3155
- https://docs.rs/cudarc/latest/cudarc/nccl/index.html
- https://ericlbuehler.github.io/mistral.rs/guides/perf/distributed-inference/

## Design

### Guiding principles

1. **Collectives are graph nodes.** No separate communication API with side
   effects; a collective is a `LazyNode` evaluated by the existing executor
   with the same cancellation semantics.
2. **Rank-agnostic programs.** World size 1 must be a no-op, so the same
   program runs on one device or many without branching.
3. **Effect-native lifecycle.** Process group init/teardown is a scoped
   `Effect.acquireRelease` layer; rank and world size come from services, not
   globals.
4. **Match MLX's API shape.** It is the closest model for a lazy runtime and
   keeps us comparable with the yardstick.
5. **Design for training from day one.** Every collective ships with a
   documented backward rule (see below) so the autodiff RFC can adopt them
   without changing the native layer.

### Native layer (Rust, `packages/native`)

#### New `LazyNode` variants

```rust
AllReduce { input: Arc<LazyNode>, op: ReduceOp }, // ReduceOp: Sum | Max | Min
AllGather { input: Arc<LazyNode> },               // concat on axis 0
ReduceScatter { input: Arc<LazyNode> },           // sum, then shard on axis 0
Broadcast { input: Arc<LazyNode>, src: u32 },
Send { input: Arc<LazyNode>, dst: u32 },
Recv { shape: Vec<usize>, dtype: DType, src: u32 },
```

Collectives that produce no value on some ranks (e.g. `Send`, `Broadcast` on
non-source ranks) still return the input tensor (or the broadcast result) so
the graph stays uniform across ranks — the rank-agnostic principle.

#### Backward rules (contract for the autodiff RFC)

Each collective has a well-known adjoint; these are recorded here so Phase 1
native work is final:

| Forward op | Backward op (on the cotangent) |
| --- | --- |
| `all_reduce(Sum)` | `all_reduce(Sum)` |
| `all_reduce(Max/Min)` | mask-and-route gradient to argmax/argmin rank(s) |
| `all_gather` | `reduce_scatter` |
| `reduce_scatter` | `all_gather` |
| `broadcast` | `all_reduce(Sum)` with gradient zeroed on non-src ranks, then kept on src |
| `send` | `recv` (reverse direction) |
| `recv` | `send` (reverse direction) |

This is why `ReduceScatter` is in the native node set even though Phase 1
never exposes it in the TS API: it is the adjoint of `all_gather`, and
adding a node variant later would force a native release for an autodiff-only
feature.

#### Communication backends

Two backends behind a `Comm` trait:

1. **NCCL** (CUDA): `cudarc::nccl::Comm::from_rank` / `from_devices`,
   enabled by the existing `nccl` cargo feature. All collectives map directly
   (including `reduce_scatter`, needed for FSDP training).
2. **Ring** (CPU, always available): ring all-reduce / all-gather over TCP
   sockets, modeled on MLX's ring backend. Serves multi-CPU data parallelism
   and CI/dev machines without GPUs. Implementation: each rank connects to
   its ring neighbor; scatter-reduce + all-gather phases over chunked
   buffers. `reduce_scatter` is the first phase of the ring all-reduce, so it
   comes nearly free.

MPI is explicitly out of scope for v1 (heavy operational dependency; MLX
itself treats it as optional).

#### Discovery / initialization

v1: environment-driven, mirroring `mlx.launch` conventions:

```
ET_WORLD_SIZE, ET_RANK, ET_MASTER_ADDR, ET_MASTER_PORT
```

`init_process_group` (blocking, on `spawn_blocking`) performs the rendezvous
(TCP store on master rank), builds the `Comm`, and returns an opaque napi
handle. Multi-GPU per-node: one process per GPU, each pinned to
`Device::new_cuda(local_rank)`.

### Core layer (TypeScript, `packages/core`)

#### `ProcessGroup` service

```ts
export class ProcessGroup extends Context.Service<
  ProcessGroup,
  {
    readonly rank: number
    readonly worldSize: number
    readonly backend: "nccl" | "ring"
    readonly handle: NativeProcessGroup
  }
>() {}

// scoped layer: acquireRelease around native init/teardown
export const layer: Layer<ProcessGroup, ProcessGroupError>

// singleton group for rank-agnostic testing: worldSize = 1
export const single: Layer<ProcessGroup>
```

#### Collective ops

Following the existing op combinator pattern, collectives take the ambient
`ProcessGroup` (and `CurrentDevice` for `recv`):

```ts
export const allSum: (
  self: Tensor.Any
) => Effect<Tensor.Lazy, TensorError, ProcessGroup>

export const allMax: (
  self: Tensor.Any
) => Effect<Tensor.Lazy, TensorError, ProcessGroup>

export const allGather: (
  self: Tensor.Any
) => Effect<Tensor.Lazy, TensorError, ProcessGroup>

export const broadcast: (
  self: Tensor.Any,
  options: { src?: number } // default 0
) => Effect<Tensor.Lazy, TensorError, ProcessGroup>

export const send: (
  self: Tensor.Any,
  options: { dst: number }
) => Effect<Tensor.Lazy, TensorError, ProcessGroup>

export const recv: (
  shape: ReadonlyArray<number>,
  options: { src: number; dtype?: DType }
) => Effect<Tensor.Lazy, TensorError, ProcessGroup | CurrentDevice>
```

World-size-1 fast path returns `self` without a graph node.

#### Sharding helpers (Phase 1)

Explicit, not compiler-driven. These cover both inference and (once autodiff
exists) training:

```ts
// Megatron column-parallel: shard weight on output dim, no forward collective
// (backward all-reduces the input gradient — handled by autodiff rules)
export const shardLinearColumn: ...

// Megatron row-parallel: shard weight on input dim, all-reduce the partial
export const shardLinearRow: ...

// FSDP: weight lives sharded; all-gather before the op
// (with autodiff, gradients are reduce-scattered back to shards)
export const withGatheredWeight: (
  shardedWeight: Tensor.Any,
  f: (full: Tensor.Lazy) => Effect<Tensor.Lazy, TensorError, ProcessGroup>
) => Effect<Tensor.Lazy, TensorError, ProcessGroup>
```

JAX's `Mesh`/`PartitionSpec` vocabulary is reserved for a future
compiler-oriented layer and deliberately **not** in Phase 1.

### Cancellation and failure

- Collectives block on the ring backend; they run on `spawn_blocking` and
  poll the existing `CancellationToken` between chunk transfers, so fiber
  interruption works as with `compute` today.
- NCCL collectives are device-synchronous; interruption is checked before
  launch and after completion (mid-collective abort requires `commAbort`,
  deferred).
- Rank failure semantics: v1 fails the whole `compute` with a
  `ProcessGroupError`; no elastic restart.

## Scope

### Phase 1 (this RFC, forward-pass distribution)

- `ProcessGroup` service + scoped layer, env-var discovery, single-process
  multi-rank testing via a ring backend in one process (threads).
- Collectives in TS: `allSum`/`allMax`, `allGather`, `broadcast`,
  `send`/`recv`. (`ReduceScatter` exists natively but stays internal until
  autodiff needs it.)
- Backends: ring-over-TCP (CPU, all platforms) and NCCL (CUDA only).
- Sharding helpers: column/row-parallel linear, weight-streaming all-gather.
- Bench: multi-rank matmul throughput vs single-device.

### Phase 2 (training, depends on the autodiff RFC)

- Reverse-mode autodiff over `LazyNode`, including the collective backward
  rules table above.
- **DDP**: gradient all-reduce, bucketed and overlapped with backward where
  the scheduler allows.
- **FSDP**: full shard semantics — all-gather weights before forward,
  reduce-scatter gradients after backward, sharded optimizer state.
- `reduceScatter` exposed in the TS API.
- Distributed data loading helpers (batch sharding by rank).

### Out of scope (deferred)

- `all_to_all` (MoE), barrier as a public op.
- Pipeline parallelism scheduling (micro-batching, GPipe/1F1B).
- Multi-node rendezvous beyond env-var TCP store; elastic groups.
- MPI, JACCL backends.
- Compiler-driven sharding (JAX-style automatic collective insertion).

## Testing strategy

- **Single process, world size 1**: every collective is an identity no-op —
  all existing tests run unchanged under `ProcessGroup.single`.
- **Single process, multi-rank**: spawn N threads each with its own ring
  group over loopback sockets; verify all-reduce sums, all-gather
  concatenation order, broadcast values, send/recv delivery.
- **Correctness cross-check**: distributed matmul result must equal
  single-device matmul result bit-for-bit (f64) or within tolerance (f32).
- CI: ring backend only (no GPU runners); NCCL paths behind a feature flag
  tested manually.

## Risks and open questions

1. **Ring backend performance.** TCP loopback ring is fine for correctness
   and CPU DP at small world sizes, but will not impress on benchmarks.
   Acceptable: NCCL is the perf path; JACCL is the interesting macOS follow-up.
2. **napi thread-safety of `Comm`.** The group handle must be `Send`-able
   into `spawn_blocking`; cudarc `Comm` is `Arc`-based and should be fine,
   but needs verification.
3. **One process per rank vs threads per rank.** Threads in one process
   simplify JS-side orchestration enormously (one Effect runtime, one test
   runner) but share one CUDA context story per device — fine for v1,
   possibly limiting later.
4. **Broadcast semantics on non-src ranks.** Options: (a) input ignored and
   replaced (PyTorch), (b) must be a placeholder node. Leaning (a) with the
   input evaluated only on src, keeping the graph uniform.
5. **Interaction with the autodiff RFC.** Order of work matters: landing
   collective nodes first means the autodiff RFC must define adjoints for
   them immediately (table above) or `grad()` will error on distributed
   graphs. Consider shipping Phase 1 collectives behind an
   `experimental.distributed` namespace until autodiff catches up.

## References

- MLX distributed API: https://ml-explore.github.io/mlx/build/html/python/distributed.html
- MLX distributed usage: https://ml-explore.github.io/mlx/build/html/usage/distributed.html
- Megatron-LM: https://arxiv.org/abs/1909.08053
- PyTorch distributed: https://docs.pytorch.org/docs/stable/distributed.html
- PyTorch FSDP: https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- JAX sharding: https://docs.jax.dev/en/latest/jax.sharding.html
- candle nccl feature: https://github.com/huggingface/candle/pull/3155
- cudarc nccl: https://docs.rs/cudarc/latest/cudarc/nccl/index.html
- mistral.rs distributed inference: https://ericlbuehler.github.io/mistral.rs/guides/perf/distributed-inference/

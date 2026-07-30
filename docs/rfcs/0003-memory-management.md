# RFC 0003: Memory Management — GC Visibility and Early Free During Evaluation

- **Status**: Implemented
- **Author**: Michael Arnaldi
- **Date**: 2026-07-27
- **Depends on**: RFC 0002 (autodiff) — backward graphs make peak-memory
  behavior matter

## Summary

Two independent improvements to how native tensor memory is managed:

1. **GC visibility**: report native tensor bytes to V8 via
   `napi_adjust_external_memory` so the JS garbage collector schedules
   collections with knowledge of native memory pressure.
2. **Early free during evaluation**: drop intermediate tensors from the
   executor's cache as soon as their last consumer has been evaluated,
   instead of holding every intermediate until the walk completes.

Both are transparent to the public API.

## Motivation

### GC blindness

Every JS-side handle (`Tensor.Lazy`, `NativeTensor`, zero-copy readback
`ArrayBuffer`) is a few dozen bytes of JS object wrapping potentially
megabytes of native/device memory. V8's GC decides when to collect based on
JS-heap pressure only; it cannot see the native side. Under a training loop
(step N builds a fresh graph, evaluates, discards), dead handles and their
native buffers can accumulate far longer than they should — the process can
balloon or OOM while V8 sees an almost-empty heap.

napi exposes `napi_adjust_external_memory` precisely for this: native code
reports bytes held/released and V8 factors them into GC scheduling.

### Peak memory during a walk

Today `eval_node` memoizes into `HashMap<u64, Tensor>` and the map lives for
the whole walk. Peak memory = **all** node values alive at once. For a
training step that's forward intermediates + backward intermediates,
simultaneously. But most intermediates have few consumers: once every
consumer of node X has been evaluated, X's tensor is unreachable and could
be freed immediately. With early free, peak memory approaches the
*live-range maximum* rather than the total — for a linear chain that halves
peak memory; for deep models the difference is the difference between
fitting and not fitting.

This is what production runtimes do: MLX's scheduler frees buffers when the
last consumer completes; XLA does static buffer assignment from liveness
analysis; PyTorch's CUDA caching allocator recycles freed blocks.

## Current state

- Graph nodes: `Arc<Node>` behind napi handles; freed by refcount when JS
  GC collects handles (no external-memory reporting).
- Evaluation: `cache: HashMap<u64, Tensor>` holds every intermediate until
  `eval_lazy` / `eval_lazy_all` returns; then all non-root values drop.
- Readback: zero-copy `ArrayBuffer`s hold the source candle `Tensor` via
  `FinalizeHint::ZeroCopy`; owned copies hold raw `Vec` allocations. Both
  free in napi finalizers, invisibly to V8.
- Candle itself: CPU tensors are `Vec`-backed; Metal buffers come from
  candle's Metal allocator (which already pools). Device buffers freed by
  candle when the `Tensor` drops.

## Design

### Part 1: external memory reporting

Report buffer bytes at the points where native memory becomes tied to a JS
object's lifetime:

- **`NativeTensor`**: on creation (end of `eval_lazy`/`eval_lazy_all`), call
  `napi_adjust_external_memory(env, +bytes)`; in a `Drop`/`finalize` hook,
  report `-bytes`. Bytes = `numel * dtype.size_in_bytes` of the materialized
  tensor.
- **Zero-copy readback `ArrayBuffer`**: the finalizer already runs; report
  `-byte_len` there, and `+byte_len` when the external ArrayBuffer is
  created. (Owned readback copies: V8 already accounts external ArrayBuffer
  backing stores in recent versions — verify before double-counting; the
  zero-copy tensor retention is the part V8 definitely can't see.)
- **Graph nodes**: proportional reporting is impractical (shared `Arc`
  subgraphs, unknown sharing). Skip; graph nodes are small relative to
  tensor data. Revisit if profiling says otherwise.

napi-rs specifics: `Env::adjust_external_memory(bytes: i64)` exists on
recent napi-rs versions — verify availability in our pinned version,
otherwise call the raw `napi_sys` function. The adjustment must happen on
the main (env) thread: evaluations complete on `spawn_blocking`, so the
accounting happens when the async fn resolves handles back on the JS side,
and in finalizers (which run on the main thread).

Failure mode to avoid: mismatched +/- reporting drifts V8's accounting.
Centralize in one helper per object type and test with a allocate/free
churn loop asserting `process.memoryUsage().external` returns to baseline.

### Part 2: early free via consumer counting

Before the walk, compute for each node the number of in-graph consumers
(parents), counting only edges from nodes reachable from the roots:

```
walk from roots; for each visited node, for each child: consumers[child] += 1
```

This is one extra O(nodes) pass over the same dedup walk the executor
already does (can share the topo/visit with evaluation setup).

During evaluation, after a node's output is computed, decrement
`consumers[child]` for each child; when a count hits zero **and** the child
is not a root, remove it from the cache — its candle `Tensor` drops,
freeing the buffer (or returning it to candle's Metal pool).

Roots are pinned by the caller (returned to JS), so they're exempt. `Leaf`
nodes own external memory (parameters, evaluated tensors) — freeing them
from the cache must NOT drop the underlying tensor if JS still holds the
`NativeTensor`; candle `Tensor` is refcounted, so dropping the cache entry
only releases the cache's reference. Correct by construction.

Notes:

- The recursion in `eval_node` becomes slightly awkward (post-order
  bookkeeping after children evaluate); an explicit iterative stack or a
  "pending count" wrapper may be cleaner than recursion.
- Cache key stays `node.id`.
- Early free interacts with `evaluateAll`: consumers counted across all
  roots — a shared forward node is freed only after the last root that
  needs it is done. Correct; this is exactly the training-step case
  (loss + all grads share the forward pass).
- No effect on results: purely a liveness optimization.

### What this explicitly does not do

- **No buffer pooling/reuse** (in-place re-use of dead buffers for new
  allocations, XLA-style). Candle's Metal allocator already pools on
  device; CPU goes through the system allocator. Cross-tensor buffer reuse
  would require alias analysis — out of scope.
- **No gradient checkpointing** (trading compute for memory by
  rematerializing forward intermediates in backward). Orthogonal, larger
  feature, remains deferred from RFC 0002.
- **No change to `randn`/lazy semantics.**

## Testing

- **Correctness**: full existing suite must pass unchanged (early free is
  invisible). Add a stress test: deep chain (10k nodes) evaluated before
  and after — results identical.
- **Early-free efficacy**: expose a debug counter (behind `cfg` or an env
  var) of live cache entries; assert peak during a long-chain walk is O(1)
  rather than O(n). Alternatively measure process RSS on a large chain.
- **External memory**: churn loop creating/discarding tensors; assert
  `process.memoryUsage().external` tracks allocation and returns to
  baseline after `global.gc()`.
- **Readback regression**: zero-copy ArrayBuffer contents remain valid
  after GC of unrelated handles; finalizer drops exactly once (existing
  debug-only double-export registry catches misuse).

## Risks and open questions

1. **napi-rs API surface**: whether `Env::adjust_external_memory` is
   exposed in our version, and whether calling it from finalizers is safe
   (it is — finalizers run on the main thread with a valid env, but verify
   against napi-rs docs).
2. **Double-counting ArrayBuffer memory**: V8 may already account
   `ArrayBuffer` backing stores created via
   `napi_create_external_arraybuffer`. Verify empirically
   (`memoryUsage().external` before/after) and only report the retained
   candle-tensor side if so.
3. **Consumer counting cost**: one extra O(nodes) pass per evaluate —
   negligible relative to kernel execution; must share the reachability
   walk rather than adding a second traversal.
4. **Early free vs. future graph caching**: if we ever cache evaluated
   subgraphs across walks (deliberately rejected for now), early free
   interacts with it; design the cache-invalidation hook when that RFC
   comes.

## References

- napi external memory: https://nodejs.org/api/n-api.html#napi_adjust_external_memory
- MLX lazy evaluation & scheduler: https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html
- XLA buffer assignment: https://openxla.org/xla/buffer_assignment
- PyTorch CUDA caching allocator: https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management

## Implementation notes (as built)

- **Positive reporting happens from JS.** `napi::Env` is not `Send`, so an
  async napi function (running on the tokio runtime) cannot call
  `adjust_external_memory` after evaluation. Instead the native side exposes
  `reportExternalMemory(bytes)` (sync, main thread) plus a `bytes` getter on
  `NativeTensor`; `Tensor.compute` reports the total right after the
  handles resolve. The negative half runs in `ObjectFinalize::finalize`
  (`#[napi(custom_finalize)]`).
- **Observability via our own counter.** Node's
  `process.memoryUsage().external` does not reflect
  `napi_adjust_external_memory` (it only tracks ArrayBuffer backing stores),
  so the native module keeps an `AtomicI64` exposed as
  `externalMemoryBytes()`; tests assert the accounting returns to baseline
  after GC. Native finalizers fire on a later event-loop turn — tests must
  pump the loop between `gc()` calls (and `it.effect` uses a TestClock, so
  waiting must go through real timers, not `Effect.sleep`).
- **Readback is not reported.** Node automatically accounts external
  `ArrayBuffer` backing stores, so adjusting for the zero-copy readback
  buffers would double-count the same memory.
- **Iterative evaluator.** The evaluator used to recurse per graph node and
  overflowed tokio's 2 MiB blocking-thread stack at ~1700-node chains (this
  predates the RFC). Evaluation is now an explicit post-order stack walk:
  children are computed before their parents and read straight from the
  cache, `count_consumers` is iterative as well, and compute stays on the
  tokio blocking pool. Chains of 200k+ nodes evaluate on a fixed stack.
- Consumer counting shares one reachability walk per `compute` call;
  `node_children` was hoisted out of the autodiff module for reuse.

# RFC 0013: Batched Decode — One Walk for N Sequences, and the Paged Attention Kernel

- **Status**: Implemented historically; generation API, fixed physical lanes,
  and inactive-lane execution are revised by RFC 0023.
- **Created**: 2026-08-02
- **Depends on**: RFC 0010 (inference), RFC 0012 (dtypes)
- **Updates**: —

## Summary

Decode becomes batched: one program run steps **N sequences one token
each** instead of N runs of one token. The artifact compiles a third
program at fixed batch width `B` (`[B, 1]` ids); the kv context
carries a slot per batch element; the cursor moves from a runtime
scalar to a `[B]` tensor slot; `KvAttention` evaluates per slot
(composed path, everywhere) and, on Metal, via a **paged decode
kernel** that reads pool slabs through block-table indirection with no
gather copy. Admission, scheduling, and padding policy stay with the
caller; the engine exposes `stepAll`.

## Motivation

Every decode step today is one eval walk: ~100µs of CPU-side encoding
for one token's worth of GPU work. N sequences generate N such walks,
and the GPU — which could process hundreds of tokens per walk — idles
between them. This is the vLLM lesson: decode throughput comes from
making each walk *wider*, not faster. A batched walk does one token
for every live sequence at nearly the cost of one.

The second motivation is the gather: `kv_attention` currently copies
the whole attended window out of the pool per layer per step (scatter
+ gather + composed sdpa). Correct and shape-stable, but pure
overhead — the kernel that attends over slab rows *in place* through
the block table removes the copy entirely.

## Design

### The batched decode program

`Model.inference` compiles three programs eagerly: prefill `[1,
prefillChunk]` (one sequence, many tokens — unchanged), decode `[1,
1]` (unchanged — single-sequence steps stay optimal at B=1), and
**batched decode `[B, 1]`** where `B = decodeBatch` (config, default
8). Fixed width, pads internally: a pad slot runs a real (allocated)
sequence whose outputs the caller discards. Fixed width means: exactly
three programs per artifact, no shape-keyed growth, no re-tracing —
the same argument as `prefillChunk`.

Why a fixed width rather than per-B programs: shape-keyed program
caches are JIT thinking; this artifact is AOT. Why not a dynamic
dimension: our frozen programs bake shapes; that is what makes them
cheap to validate and fast to bind.

### The cursor becomes a tensor

Single-sequence decode binds one cursor scalar. Batched sequences have
independent cursors, so the `[B, 1]` program takes a `cursors: [B]`
**tensor** input instead:

- `PositionEmbedding` (learned tables): the decode rewrite builds flat
  indices `cursors[b] + p` over `[maxPositions, E]`, gathers `[B·t]`
  rows, reshapes `[B, t, E]`. (Tables cap positions as today.)
- `RotaryEmbedding` offset `Cursor`: the rotary tables are built from
  a `[B, T]` position grid (`cursors[:, None] + arange(T)`) — a small
  generalization of `rotary_forward`, identical math per row.

### Batched `KvAttention`

The node keeps its semantics; the eval handles the leading batch dim:
slot *b* of the `KvContext` (a `Vec` of sequence states, one per
batch element) owns batch row *b* — its block table, cursor, window,
advance. Two implementations behind one semantic:

- **Composed (CPU, and fallback)**: per slot, the existing scatter +
  gather + sdpa on that slot's slice of q. Correct everywhere; already
  a throughput win because the *rest of the graph* (embedding, MLPs,
  head) runs batched in one walk.
- **Paged kernel (Metal)**: one launch, `B × H` threadgroups. Inputs:
  `q [B, H, 1, D]`, the K/V slabs, block tables `[B, maxBlocks]`
  (u32), context lengths `[B]` (u32), per-slot window, scale. Each
  threadgroup walks its slot's block table, streaming K/V rows from
  the slabs with online-softmax accumulation (flash-style, runtime
  context length — no baked shapes, unlike the training flash
  pipeline). No gather copy; scatter of the new rows stays as-is (B
  rows per layer — negligible).

The kernel lives in **this crate** (`paged.rs`), compiled at runtime
from source the way `flash.rs` already does — no candle fork changes.

Quantized pools (RFC 0012) compose naturally: the kernel dequantizes
rows (f16/bf16: native loads; int8: `(q − 128) · scale` with the scale
slab) in registers. f16/bf16/int8 slabs are supported from the start —
that is why the storage tier was designed at the scatter/gather
boundary.

### API: `stepAll`

```ts
interface InferenceProgram {
  sequence(): Effect<Sequence, InferenceError, Scope>
  // Steps each entry's sequence one token in a single batched run.
  // Entries are independent sequences (they may share pool blocks via
  // the prefix cache). At most `decodeBatch` entries per call; the
  // caller owns admission and scheduling. Returns one logits row per
  // entry, in order.
  stepAll(
    entries: ReadonlyArray<{ readonly sequence: Sequence; readonly token: Tensor.Any /* [1, 1] */ }>
  ): Effect<ReadonlyArray<Tensor.Concrete>, InferenceError | ..., CurrentDevice>
}
```

Deliberately *not* an engine-managed sequence set: admission,
backpressure, eviction policy, and step interleaving are caller
policy, per the standing rule. The engine guarantees: one walk per
call, disjoint writes (slot ownership), rollback on error, results
identical to per-sequence `step` calls.

`Sequence` needs a native-handle escape hatch for `stepAll` to bind
slots; prefill is unchanged (batched prefill is a non-goal — prefill
is compute-dense already, batching it saves little).

### Interaction with what exists

- **Prefix cache**: untouched — batched slots take/share/refcount
  blocks exactly as single sequences do; concurrent same-prefix
  batches hit the same `BlockStore` mutex.
- **Sliding window**: per-slot windows (the kernel takes a window per
  slot); eviction runs per slot after its slot's attention, unchanged.
- **Rollback**: a failed batched run rolls back every slot's
  allocation beyond its pre-run frontier — per-slot `frontier`
  snapshots, same rule as today.
- **Run locks**: `stepAll` acquires every entry's run lock in a fixed
  order (sorted by address) to avoid deadlocks when batches overlap.

## Non-goals

- **Batched/chunked prefill** (Sarathi-style mixed batches) — prefill
  is already compute-dense; the win is small next to decode.
- **Multiple bucket widths** (pad to 4/8/16/32) — one width is the
  v1; buckets are a config addition, not a redesign.
- **CUDA** — Metal first, composed path covers CPU.
- **A scheduler** — iteration-level scheduling, prefill/decode
  interleaving, and admission queues are server policy built *on*
  `stepAll`, not engine features.
- **Prefill through the paged kernel** — Tq > 1 causal masking in the
  kernel is straightforward but needless while prefill is per-sequence.

## Alternatives considered

- **Engine-managed batching** (register sequences, engine forms
  batches per step): rejected — bakes scheduling policy into the
  library; the caller knows its latency/throughput tradeoffs.
- **Per-batch-size program cache** (JIT, shape-keyed): rejected —
  reintroduces arbitrary program growth the AOT artifact eliminated.
- **Padding-free ragged batch** (varlen kernel, `[sum(1…)]`): the
  right end-state for the kernel itself (it is varlen internally
  anyway via context lengths), but the *program* still needs a fixed
  signature — fixed width + pads is the honest version of varlen at
  our abstraction level.
- **Keeping the cursor scalar and requiring equal cursors**: rejected
  — shared-prefix generations diverge immediately.

## Acceptance criteria

1. **Parity**: `stepAll` over N sequences matches per-sequence
   `step` logits bit-for-bit (f32 pool, composed path, CPU and Metal).
2. **Mixed slots**: sequences with different cursors, windows (some
   none), and shared prefix blocks in one batch behave exactly as
   sequential steps.
3. **Padding**: a partially filled batch (fewer entries than
   `decodeBatch`) is exact; pad slots neither fail nor perturb.
4. **Isolation**: pool exhaustion mid-batch rolls back all slots and
   leaves other sequences untouched.
5. **Kernel parity** (Metal): paged-kernel logits match the composed
   path within f32 tolerance across random block tables, cursors,
   windows, and pool dtypes (f32/f16/bf16/int8). *(Met: the entire
   Inference suite's Metal variants — token parity, windowing, prefix
   cache, stepAll, all four pool dtypes — run decode through the
   kernel against f32 references.)*
6. **Throughput sanity**: a 8-wide batch steps in < 2× the wall time
   of a single step on Metal (the walk dominates; composed attention
   is the interim). *(Measured after the kernel work below: 2.98× a
   single step — 2.7× faster than eight sequential steps — on a tiny
   proxy model where per-op launch cost dominates; the criterion still
   misses on the proxy. The residual is the rest of the graph's many
   tiny ops, i.e. RFC 0007 fusion territory, not the attention path.)*

## Addendum: kernel pass (implemented)

Second iteration over `paged.rs`, all verified by the full Inference
suite on Metal (decode AND prefill now run paged; the composed
scatter+gather path serves CPU only):

- **Unified attention kernel**: one kernel covers decode and chunked
  prefill — `grid (H, B, C)` with per-row causal lengths derived from
  `advance` (decode is `C=1`); pads clamp to the real frontier.
- **Fused scatter kernel**: one launch per layer writes every slot's
  new rows straight from the graph's `[B, H, C, D]` layout (no
  narrow/permute/contiguous/index-tensor ops at all), quantizing int8
  in-kernel with the same absmax grid as the composed path.
- **Memory optimizations**: q staged in threadgroup memory (was
  re-read from device per row), 128-bit vectorized slab loads
  (`packed_float4/half4/bfloat4`, packed-u32 for int8), simd-shuffle
  reduction (no NT×D scratch).
- **Run-level hoisting**: block tables + context lengths are built
  once per run in the `KvContext`, not per layer.

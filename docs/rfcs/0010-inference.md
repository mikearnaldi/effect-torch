# RFC 0010: Inference — Paged KV Caching, Decode Compilation, RoPE

- **Status**: Implemented
- **Author**: Michael Arnaldi
- **Date**: 2026-08-01
- **Depends on**: RFC 0007 (kernel fusion), RFC 0008 (compilation —
  `Tensor.compile`, `Model.compile`, shape-keyed `ProgramCache`, runtime
  scalar slots), the `scaledDotProductAttention` semantic node and its
  flash Metal kernel

## Summary

Generation used to recompute everything: each sampled token re-ran the
full (padded) window through `Model.forward`, re-deriving keys and
values that causality had already frozen. This RFC adds an inference
path with three pieces:

1. **A paged KV arena (native).** A fixed-capacity pool of key/value
   rows per attention layer, allocated once per inference artifact.
   Sequences are rows of metadata — a block table and a cursor — over
   the shared pool, in the style of vLLM's PagedAttention.
2. **Decode compilation as a native AST rewrite.** `Model.inference`
   traces the *same* `Model.forward` builder used by training, then
   rewrites the graph in Rust before freezing: causal `Sdpa` becomes
   `KvAttention` (scatter the new tokens into the pool, attend over the
   cached context), `PositionEmbedding` becomes a cursor-offset gather,
   `RotaryEmbedding` becomes cursor-offset rotary. Model authors write
   `forward` once; decode is a property of compilation, not of the
   model definition — and no ambient build mode exists anywhere.
3. **`Model.inference` → `InferenceProgram`.** A compiled artifact —
   sibling to `CompiledModel`, not a `Model` — holding the prefill and
   decode programs plus the pool, from which cheap, Scope-managed
   `Sequence` handles are acquired: `prefill(tokens)` then `step(token)`
   per generated token.

It also adds **RoPE** (`RotaryEmbedding` semantic node,
`multiHeadAttention(..., { rope })`), because relative positions are
what make cached generation *unbounded*: with an `attentionWindow`,
cached sliding-window attention matches window-relative training
exactly, while learned absolute embeddings hard-cap the context at the
table size.

## Motivation

**Autoregressive decode without a cache is quadratic and JIT-hostile.**
The naive cached alternative — handing a re-grown contiguous K/V tensor
to `scaledDotProductAttention` each step — changes the input signature
at every step, churning both shape-keyed caches (the `ProgramCache`,
LRU 32, and the flash kernel's pipeline cache, shapes baked as
`#define`s). A compile per token is strictly slower than the naive
full-window loop. Any caching scheme that does not keep shapes still is
worse than none.

**Decode is bandwidth-bound; capacity is the lever.** Serving
throughput is bounded by how many sequences fit in device memory.
Per-sequence contiguous KV buffers sized to max length waste 60–80% of
it; a paged pool wastes only the slack in each sequence's last block.

**Window-relative positions are cache-incompatible.** The pre-RFC
example loop generated unbounded text by re-running the last W tokens
with positions restarting at 0 — but sliding the window changes every
cached token's position, hence its K/V, hence nothing can be cached
(that is also why GPT-2-style learned absolute embeddings hard-cap
generation at the table size). Relative positions (RoPE) are the
mechanism that makes "generate infinite tokens" true: attention sees
only q−k offsets, K/V freeze at write time, and a sliding window over
the cache reproduces the training distribution exactly.

**The library had a training story and no inference story.** RFC 0008
froze the training step into compiled programs; generation was an
interpreted per-token walk in user code.

## Prior art

- **vLLM / PagedAttention** (SOSP'23): the OS virtual-memory model —
  tokens are bytes, blocks are pages, sequences are processes, block
  tables are page tables. One arena allocated at startup; blocks mapped
  on demand; the kernel chases the block table. Sharing (parallel
  sampling, beam search, prefix reuse) is block-table aliasing with
  refcounts and copy-on-write. We adopt the memory model; the sharing
  machinery is deferred (Non-goals).
- **Mistral sliding-window attention + RoPE** (LLaMA): the mechanism
  for unbounded cached generation; adopted as `attentionWindow` +
  `RotaryEmbedding`.
- **LMCache**: KV blocks as content-addressed, tiered artifacts keyed
  by chunk hash and model identity — confirms the direction (sealed
  blocks are immutable values; only the write frontier mutates) and is
  a Non-goal here.
- **HuggingFace `StaticCache`**: fixed-size buffers with a cursor —
  the same shape-stability insight without paging; paging additionally
  solves multi-sequence capacity.
- **Flash attention** (already ours): prefill and training keep the
  contiguous flash path; the kv eval deliberately feeds a
  shape-polymorphic composed path (see Design).

## Design

### Semantic nodes: `PositionEmbedding`, `RotaryEmbedding`

The decode rewrite must not reverse-engineer structure from composed
ops ("this gather is a position embedding" would be archaeology). So
the two position mechanisms are semantic nodes, like `Sdpa` before
them:

- `PositionEmbedding { weight, seq_len }` — rows `0..seq_len-1` of the
  `[maxPositions, E]` table (what `Model.positionEmbedding` lowers to;
  differentiable: scatter-add backward).
- `RotaryEmbedding { x, seq_len, theta, offset }` — GPT-NeoX half-split
  rotary over the last dim (even), positions `0..seq_len-1` rotated by
  `theta^(-2j/D)`; `offset` is `Absolute` in user graphs and `Cursor`
  after the decode rewrite. Differentiable (transpose-rotation
  backward, composed from the same tables). `Model.multiHeadAttention`
  applies it to q and k per head with `{ rope: theta }`.

### The decode rewrite (native)

`compile_decode(roots, window?)` walks the traced forward DAG and
rebuilds it:

- `Sdpa { q, k, v, scale, causal: true }` → `KvAttention { q, k, v,
  scale, layer, window }` — the layer ordinal is the node's order of
  first encounter in a deterministic post-order walk, so the prefill
  and decode traces of one model agree. Non-causal attention is
  rejected (`InferenceError`: only causal attention is cacheable).
- `PositionEmbedding { weight, t }` → `Gather(weight, broadcast(
  arange(t) + cursor))`, where `cursor` is a new `ScalarInput` slot
  appended after the last declared slot.
- `RotaryEmbedding { offset: Absolute }` → `offset: Cursor` (the eval
  reads the sequence's cursor from the kv context; no slot needed).
- Runtime scalar inputs are rejected (unsupported in inference graphs).
  Graphs with no cacheable or recurrent operation are valid and produce a
  zero-state decode geometry; their sequence cursor still tracks committed
  tokens and can drive specialized position operations.

The rewritten graph is fused and frozen with the RFC 0008 machinery.
One model yields exactly two programs: one decode (`[1, 1]` input) and
one prefill (`[1, prefillChunk]` input) — prompts are processed in
fixed-shape chunks, the last one zero-padded with only its real rows
scattered into the cache (the run's `advance` count), causality keeping
real positions from ever attending to pads. This is the chunked-prefill
regime real engines use (vLLM, Sarathi); a whole deployment compiles
two programs and never re-traces, whatever the prompt lengths.

### The pool and sequences (native)

`NativeKvPool(layers, kvHeads, headDim, maxTokens, blockSize = 16)`:
per-layer flat `[maxTokens, kvHeads, headDim]` f32 slabs; block `b`
occupies rows `[b·blockSize, (b+1)·blockSize)`; a free-list allocator
hands blocks to sequences. Geometry is *derived* from the rewritten
graph (layer count from the Sdpa ordinals, head geometry from their
shapes) — one source of truth; `maxTokens`/`blockSize` are the only
deployment knobs. `NativeKvSequence` is a block table + cursor over the
pool, Scope-managed: `release()` (or drop, or the sequence's scope
closing) returns its blocks.

`KvAttention` evaluation scatters the new tokens' k/v at the cursor
(`scatter_set` through the block table, allocating on block-boundary
crossings), then attends q causally within each operation's resolved `window`
(whole context when unset). Pool-level retention uses
the configured window only when every resolved attention operation is
windowed; any full-attention operation retains the whole context. Under safe retention, blocks
fully below the moving frontier are **evicted** back to the pool —
they are never attended again — so a windowed sequence's footprint is
O(window) however long it generates; the capacity error then bounds
*live* rows, not total context (without a window every cached row is
load-bearing and the check is the total context, as before). The gather
feeds the **composed** sdpa path — shape-polymorphic kernels, so no
pipeline recompile as the context grows; the flash kernel's shape-baked
pipeline cache is exactly what this avoids. A dedicated paged Metal
kernel (runtime context length, block-table indirection, no gather
copy) is the throughput follow-up, not v1.

### `Model.inference` → `InferenceProgram`

```ts
export interface InferenceConfig {
  readonly maxTokens: number
  readonly blockSize?: number        // default 16, must divide maxTokens
  readonly attentionWindow?: number  // sliding window; omit for full attention
  readonly prefillChunk?: number     // default blockSize; one [1, C] program for all prompts
  readonly tokenDtype?: "u32" | "i64" // id dtype of prefill/step inputs; default "u32"
}

export interface InferenceProgram {
  readonly sequence: () => Effect.Effect<Sequence, InferenceError, Scope.Scope>
}

export interface Sequence {
  readonly prefill: (tokens: Tensor.Any /* [1, T] */) => Effect.Effect<Tensor.Concrete /* [vocab] */, ...>
  readonly step: (token: Tensor.Any /* [1, 1] */) => Effect.Effect<Tensor.Concrete /* [vocab] */, ...>
  readonly cursor: () => Effect.Effect<number>
}

export const inference: (
  model: Model,
  params: Params,
  config: InferenceConfig
) => Effect.Effect<InferenceProgram, InferenceError | ModelError | Tensor.TensorError, CurrentDevice | Scope.Scope>
```

Deliberate shape decisions:

- **Not a `Model`.** `forward` accepts any input shape; the artifact
  accepts `[1, T]` prefill and `[1, 1]` decode against a bound pool,
  and composing it is meaningless. `InferenceProgram` is a sibling of
  `CompiledModel` — the method sets genuinely differ, and per the
  codebase rule that is a distinct interface, not optional members.
- **Params are frozen once.** They may be lazy graphs (init draws), and
  a compiled run materializes its inputs per call — without an up-front
  `compute`, every `prefill`/`step` would re-draw `randn` params. The
  facade computes them once; natively they remain program inputs
  (`[...params, token]`), preserving RFC 0008's functional rebinding.
- **Eager compilation.** Exactly two signatures exist
  (`[1, prefillChunk]`, `[1, 1]`) and dtype/device are config, so both
  programs and the pool are built at construction, and construction
  errors such as non-causal attention fail
  `inference` itself rather than the first call.
- **No explicit lifetime on the artifact.** The two programs are
  static (no shape-keyed growth, so RFC 0008's `dispose`-as-JIT-
  hygiene rationale does not apply) and the pool is device memory of
  the same kind tensors are — everything is released by native
  finalizers when the artifact is unreachable. **Sequences keep
  Scope**: they hold pool blocks, a capacity resource with no
  GC-visible pressure signal (a held block is a slice of an
  already-allocated slab — nothing new is allocated, so GC has no
  reason to collect a dead sequence promptly), so deterministic return
  is the contract, with `Drop` as the safety net.
- **State lives in `Sequence`, never in the program.** The artifact is
  immutable and parallel-safe; per-sequence metadata is native state
  behind the handle. Calls on one sequence serialize (a run lock);
  sequences run concurrently — disjoint blocks by allocation.
- **Errors roll back.** A failed run (pool exhausted, context overflow)
  returns the blocks it allocated to the free list and does not advance
  the cursor — a poisoned sequence must not take the pool down with it.

### Concurrency

RFC 0008's story (immutable graphs, per-call evaluators, functional
placeholder rebinding) is preserved, with the pool as the deliberate
exception: persistent device memory across calls, made safe by
**disjoint ownership** — a block belongs to exactly one live sequence,
and a run touches only its sequence's blocks. One addition to the RFC
0008 machinery fell out of testing: scalar bindings (`cursor`, and the
trainer's `lr`) allocate on the device, so binding construction moved
**inside** the Metal eval guard in both `CompiledProgram.run` and
`DecodeProgram.run` — allocating on the device concurrently with
another walk is unsafe on the single-shared-command-buffer Metal
backend.

## Non-goals

- **A dedicated paged Metal kernel.** The kv eval gathers through the
  block table and runs composed sdpa; correct and shape-stable, ~2× the
  memory traffic of a fused kernel. The kernel (runtime context length,
  block-table indirection baked, q_len = 1 fast path) is the follow-up.
- **Continuous batching** (one run stepping many sequences), **parallel
  sampling / beam search** (explicit block-table forking with
  copy-on-write — the prefix cache below covers content-addressed
  sharing; explicit forks serve search-shaped workloads, where the
  caller knows the lineage), **cross-process KV reuse** (LMCache-style
  tiered offload, P/D disaggregation).
- **Sampling**: argmax/temperature/top-k stay in user code over the
  returned logits.
- **Cache quantization** (f16/f8 KV), **GQA/MQA**, **CUDA**.
- **A standalone `KvPool.make`.** Pool geometry is derived at
  `inference` time (single source of truth). Explicit pools — capacity
  sharing across same-architecture programs, e.g. speculative
  draft/target pairs — are a documented extension point with bind-time
  validation, to be added when a use case exists.
- **Sliding-window *training***. `attentionWindow` serves models
  trained on W-token windows (the nano-gpt regime); masking full-corpus
  training to W is a trainer-side option if ever needed.

## Alternatives considered

- **Ambient decode mode during the TS build** (context tag / fiber ref
  reinterpreting `scaledDotProductAttention` while walking `forward`):
  rejected — `forward` builds a native AST; the Rust side has full
  access, so the reinterpretation belongs there as a graph rewrite.
  What made this total rather than fragile is that attention and both
  position mechanisms are *semantic nodes*: the rewrite never
  reverse-engineers composed ops.
- **A `decode` builder on `Model`** (authors write forward twice):
  rejected — the decode difference is localized to three node kinds;
  restating the architecture invites divergence.
- **Post-hoc graph rewrite without semantic position nodes**: rejected
  — recognizing "this gather is a position embedding" among composed
  ops is heuristics, not engineering.
- **Growing contiguous cache** (concat each step): rejected — input
  signature changes every step, churning both shape-keyed caches.
- **Hidden per-session state on the program**: rejected — mutable slots
  behind the handle break RFC 0008's functional rebinding and parallel
  calls. State lives in explicitly acquired `Sequence`s.
- **Bumping the position table to fake unbounded generation**: rejected
  for the example — window-relative learned positions are
  cache-incompatible, and absolute ones are out-of-distribution past
  the training range. RoPE + sliding window is the mechanism real
  systems use.

## Acceptance criteria

All met, in `packages/core/test/Inference.test.ts` (CPU and Metal):

1. **Parity**: greedy generation through `InferenceProgram` matches the
   naive full-window loop token-for-token across pool block boundaries
   (`blockSize` not dividing the context length), and matches the
   prefill logits numerically (f32 tolerance).
2. **Shape stability**: exactly two programs exist per artifact —
   compiled eagerly at construction — so multi-sequence, multi-length
   generation can never re-trace; chunked prefill of a prompt spanning
   several chunks matches the naive loop token-for-token.
3. **Concurrency**: N sequences stepped concurrently on one artifact
   produce results identical to N sequential runs.
4. **Isolation**: pool exhaustion fails the offending sequence's call
   with a typed error, rolls back its partial allocation, and leaves
   concurrent sequences unaffected; context overflow past `maxTokens`
   fails likewise.
5. **Lifetimes**: a released sequence's blocks return to the pool
   (verified by refilling the pool after scope close); the artifact's
   scope disposes its programs.
6. **Honest failures**: models without causal attention, non-causal
   attention, runtime scalar inputs, and malformed prefill/step shapes
   fail with typed `InferenceError`s.
7. **RoPE**: full-attention parity with the naive loop;
   `attentionWindow` generation matches the window-relative recompute
   token-for-token with contexts far past the window — in a pool
   smaller than the context, so dead-block eviction is exercised
   (unbounded generation at O(window) footprint); the rotary node
   differentiates (training loss decreases).
8. **Migration**: `nano-gpt.ts` generates through `Model.inference` —
   prefill once per prompt, one pooled step per token, sliding-window
   attention over the last `BLOCK` positions, EOS-terminated — with
   output quality matching the pre-RFC recompute loop.

## Addendum: automatic prefix caching (implemented)

Prompts arrive at a server independently — some share a prefix, most do
not, and none declares lineage. Sharing therefore falls out of the
pool, not the API: blocks are content-addressed, and a prefill whose
prefix is already resident reuses it (vLLM's automatic prefix
caching). Always on; there is no configuration.

- **Chained block hashes.** A block's hash is FNV-1a over the previous
  block's hash and its own token ids, so an equal hash implies equal
  tokens at equal absolute positions. With causal attention and RoPE,
  the cached K/V rows are then bit-identical to a recompute — no
  staleness, no invalidation, and model identity is implicit (the pool
  belongs to one artifact).
- **Full blocks only.** A block becomes hashable when its last row is
  written; a partial tail block's content is not final. Prefix matches
  truncate to whole blocks, and the block holding the prompt's last
  token is always computed — its logits are prefill's result.
- **Refcounted sharing across live sequences.** Every completed block
  is indexed by hash, owned or not. `prefill` walks the chain from the
  prompt's first block and takes references (refcount bumps) while
  resident; the first miss ends the match, and chunked prefill
  continues from the block-aligned cursor. Taken blocks are read-only:
  divergence always allocates fresh blocks.
- **The cache is the unreferenced subset.** Window eviction and
  `release` decrement refcounts; a block reaching zero with a known
  hash stays indexed and joins the LRU. Allocation takes from the free
  list first, then evicts the least-recently-used cached block — so
  cached prefixes cost nothing under pressure, and `freeBlocks`
  reports free + reclaimable.
- **Rollback stays exact.** A failed run unrefs the blocks it
  allocated (unhashed — hashes are recorded only on success), which
  return to the free list; shared blocks below the run's frontier are
  untouched.

What this deliberately is not: `Sequence.fork`. Forking assumes the
caller knows the lineage — true for tree search, n-best sampling, and
speculative draft/verify, false for HTTP traffic. Content addressing
subsumes the server case with zero API surface; an explicit fork with
copy-on-write tail blocks remains the mechanism for search-shaped
workloads, unimplemented until one exists.

Acceptance (in `Inference.test.ts`, CPU and Metal): a second live
sequence fits a pool smaller than two independent prompts only by
sharing; divergent suffixes after a shared prefix match an ordinary
forward exactly; a cached prefix is reclaimed under pressure and the
new tenant matches its naive reference; window-evicted blocks are
reused from the cache with exact parity; a second prefill on a used
sequence fails typed.

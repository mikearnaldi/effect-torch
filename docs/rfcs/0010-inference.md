# RFC 0010: Inference — Paged KV Caching and Decode Compilation

- **Status**: Proposed
- **Author**: Michael Arnaldi
- **Date**: 2026-08-01
- **Depends on**: RFC 0007 (kernel fusion), RFC 0008 (compilation —
  `Tensor.compile`, `Model.compile`, shape-keyed `ProgramCache`, runtime
  scalar slots), the `scaledDotProductAttention` semantic node and its
  flash Metal kernel

## Summary

Generation today recomputes everything: each sampled token re-runs the
full padded window through `Model.forward`, re-deriving keys and values
that causality has already frozen. This RFC adds an inference path to
`@effect-torch/core` with three pieces:

1. **`KvPool` — a paged KV arena (native).** A fixed-size, fixed-address
   pool of key/value blocks per attention layer, allocated once per
   inference artifact. Sequences are rows of metadata — a block table and
   a cursor — over the shared pool, in the style of vLLM's PagedAttention.
2. **Decode-mode compilation.** The same `Model.forward` builder is
   walked with `[1, 1]` placeholders and the cache-relevant primitives
   (`scaledDotProductAttention`, `positionEmbedding`) emit cache-aware
   nodes: scatter the new token's K/V into the pool, attend against it
   through the block table. Model authors write `forward` once; decode is
   a property of compilation, not of the model definition.
3. **`Model.inference` → `InferenceProgram`.** A compiled artifact —
   sibling to `CompiledModel`, not a `Model` — holding the prefill and
   decode programs plus the pool, from which cheap, Scope-managed
   `Sequence` handles are acquired: `prefill(tokens)` then `step(token)`
   per generated token.

## Motivation

**Autoregressive decode without a cache is quadratic and JIT-hostile.**
The naive cached alternative — handing a re-grown contiguous K/V tensor
to `scaledDotProductAttention` each step — changes the input signature at
every step. Two of our caches are keyed on exactly that signature: the
`ProgramCache` (RFC 0008, LRU 32) and the flash kernel's pipeline cache
(shapes baked as `#define`s, flash.rs). A growing cache means a compile
per token per generation and constant LRU eviction — strictly slower than
the naive full-window loop it replaced. Any caching scheme that does not
keep shapes still is worse than none.

**Decode is bandwidth-bound; batching is the lever.** Serving throughput
is bounded by how many sequences fit in device memory, because every step
re-reads all weights per token. Per-sequence contiguous KV buffers sized
to max length waste 60–80% of device memory (over-reservation +
fragmentation); a paged pool wastes only the slack in each sequence's
last block. More resident sequences, more tokens per weight-read.

**The library has a training story and no inference story.** RFC 0008
froze the training step into compiled programs; generation is still an
interpreted per-token walk in user code (`nano-gpt.ts`). A library for
building models needs the decode path to be as first-class as the
training step.

## Prior art

- **vLLM / PagedAttention** (SOSP'23): the OS virtual-memory model —
  tokens are bytes, blocks are pages, sequences are processes, block
  tables are page tables. One arena allocated at startup; sequences map
  logical blocks to non-contiguous physical blocks on demand; the
  attention kernel chases the block table instead of assuming contiguous
  K/V. Sharing (parallel sampling, beam search, prefix reuse) is
  block-table aliasing with refcounts and copy-on-write. We adopt the
  memory model wholesale; the sharing machinery is deferred (Non-goals).
- **LMCache**: KV blocks lifted out of the engine into a content-addressed,
  tiered store (GPU→CPU→disk→remote), keyed by chunk hash and model
  identity. Confirms the direction: sealed blocks are immutable values;
  only the write frontier mutates. Cross-request reuse is a Non-goal
  here, but the pool design does not preclude it.
- **HuggingFace `StaticCache`**: fixed-size per-layer buffers with a
  cursor — the same shape-stability insight without paging. We page
  because the pool also solves multi-sequence capacity, which a static
  per-sequence buffer does not.
- **Flash attention** (already ours): the paged decode kernel reuses the
  tiled online-softmax loop body; only the key-tile *source* changes
  (block-table indirection instead of contiguous strides).

## Design

### Native: `KvPool`, two graph ops, one kernel

```rust
NativeKvPool {
  // geometry fixed at allocation
  layers: usize,        // one K/V slab pair per SDPA node, in walk order
  kv_heads: usize,
  head_dim: usize,
  block_size: usize,    // tokens per block, default 16
  max_tokens: usize,    // total pool capacity across all sequences
  dtype: DType,         // f32 v1
  device: Device,       // from CurrentDevice
}
```

The pool is one device allocation per layer per K/V:
`[num_blocks, kv_heads, block_size, head_dim]`. It is a resource —
acquired in a `Scope`, released deterministically, freed with the pool's
last reference. A free-list allocator hands blocks to sequences; a block
is owned by exactly one live sequence at a time.

Two new semantic graph nodes, in the same sense as `Sdpa`:

- **`KvScatter(k_new, v_new, layer, cursor)`** — writes the new token's
  K/V into the sequence's current block at `cursor % block_size`,
  allocating a fresh block from the pool when `cursor` crosses a block
  boundary. The only mutating node in the graph language; its writes are
  confined to blocks the calling sequence owns.
- **`PagedAttention(q, layer, block_table, context_len)`** — `q` is
  `[1, kv_heads, 1, head_dim]` (decode is single-query); `block_table` is
  a `u32` tensor input `[max_blocks_per_seq]`; `context_len` arrives via
  the existing runtime-scalar mechanism (RFC 0008). The kernel iterates
  table entries, gathers key tiles from scattered blocks, and runs the
  flash online-softmax loop unchanged. `BLOCK_SIZE`, `HEAD_DIM`, and
  `MAX_BLOCKS` are baked into the pipeline key; **sequence length is a
  runtime value** — this is the change that keeps the pipeline cache
  still across a whole generation.

The Metal paged kernel is a decode-specialized sibling of the flash
kernel, not a replacement: prefill and training keep the contiguous flash
path. A CPU fallback composes gather-blocks → contiguous → existing
`sdpa_forward`, keeping tests portable; it is correct, not fast.

### Decode-mode compilation

`Model.inference` walks the **same `forward` builder** used by training,
with two reinterpretations applied at *node construction* (not as a
post-hoc graph rewrite — the structure is stated at the semantic node,
so nothing is reverse-engineered):

1. **`scaledDotProductAttention(q, k, v, causal)`** — with `[1, 1]`
   inputs, `k`/`v` are already just the new token's projections. The node
   becomes `KvScatter(k, v, layer=i, cursor)` followed by
   `PagedAttention(q, layer=i, block_table, cursor + 1)`, where `i` is
   the node's ordinal among SDPA nodes in walk order. Causality is
   structural: the kernel reads blocks up to `context_len`, so no mask is
   needed in decode.
2. **`positionEmbedding`** — positions come from the `cursor` scalar
   (`cursor + arange(1)`) instead of `arange(input.length)`.

Everything else — embeddings, layer norms, MLPs, residuals, heads — is
emitted byte-identically. Consequences:

- **One definition, three artifacts.** Train fwd+bwd, prefill, and decode
  are separate programs over shared weights, coexisting in the shape-keyed
  `ProgramCache`. A whole generation compiles exactly two programs
  (one prefill per distinct prompt-length signature, one decode).
- **Models without attention are rejected.** A walk that encounters zero
  SDPA nodes fails with a typed `InferenceError` ("no cacheable
  attention") — decode mode has no meaning there, and hand-rolled
  matmul+softmax attention is honestly unsupported rather than silently
  uncached.
- **Pool geometry is derived, not configured.** The walk counts SDPA
  nodes (layers) and reads head structure from their shapes; `KvPool`
  sizing takes only the genuinely deployment-specific knobs.

### Core: `Model.inference` → `InferenceProgram`

```ts
export interface InferenceConfig {
  /** Total KV capacity in tokens, shared by all live sequences. */
  readonly maxTokens: number
  /** Tokens per block. Default 16. */
  readonly blockSize?: number
}

export interface InferenceProgram {
  /** Acquire a sequence: a block table and cursor in the pool.
      Cheap, parallel, released with its scope. */
  readonly sequence: () => Effect.Effect<Sequence, InferenceError, Scope>
  readonly stats: () => Tensor.CompileStats
}

export interface Sequence {
  /** Forward the prompt, filling the sequence's blocks; returns the
      final-position logits [vocabSize]. */
  readonly prefill: (
    tokens: Tensor.Any
  ) => Effect.Effect<Tensor.Concrete, InferenceError | Tensor.TensorError, CurrentDevice>
  /** Append one token (id as a 0-d/[] u32 tensor); returns the next
      position's logits [vocabSize]. */
  readonly step: (
    token: Tensor.Any
  ) => Effect.Effect<Tensor.Concrete, InferenceError | Tensor.TensorError, CurrentDevice>
  /** Tokens consumed so far (prompt + generated). */
  readonly cursor: () => Effect.Effect<number>
}

export const inference: (
  model: Model,
  params: Params,
  config: InferenceConfig
) => Effect.Effect<InferenceProgram, InferenceError, CurrentDevice | Scope>
```

Deliberate shape decisions:

- **Not a `Model`.** `forward` on a `Model` accepts any input shape; the
  inference artifact accepts `[1, T]` prefill and `[1, 1]` decode against
  a specific bound pool, and composing it (`chain`, `add`) is
  meaningless. `InferenceProgram` is a sibling of `CompiledModel`, not a
  subtype — the method sets genuinely differ, and per the codebase rule
  that is a distinct interface, not optional members on a shared one.
- **Params close over the facade, placeholders natively.** Per RFC 0008,
  params remain program inputs (`[...params, token]`); `inference` closes
  over the params array and supplies it per run, so callers thread
  nothing. Functional placeholder rebinding is preserved — there is no
  mutable param slot to race on.
- **Scope-bearing constructor.** `inference` allocates the pool — device
  memory, a genuine resource — so unlike `Model.compile` it requires
  `Scope`. This is the first compile variant that acquires resources; the
  signature says so.
- **State lives in `Sequence`, never in the program.** The
  `InferenceProgram` is immutable and parallel-safe; per-sequence
  metadata (block table, cursor) is native state behind the `Sequence`
  handle, exactly the vLLM split of immutable program + per-sequence page
  table.

### Concurrency

RFC 0008's concurrency story rests on programs holding no device buffers
between calls. The pool deliberately breaks that invariant — persistent
device memory across calls is the point — so parallel safety is
re-established by **disjoint ownership** rather than immutability:

1. The allocator guarantees a block belongs to exactly one live sequence.
2. A run's writes (`KvScatter`) target only blocks in its sequence's
   table; reads (`PagedAttention`) likewise.
3. The block table travels as an input tensor; cursor as a runtime
   scalar. Programs remain pure functions of their inputs.

N concurrent sequences therefore behave as N sequential ones, each
writing disjoint memory — the same guarantee, differently grounded.
Prefill of one sequence may run concurrently with decode steps of others.

### Failure modes and fallbacks

- **Pool exhaustion**: `KvScatter` cannot allocate a block → typed
  `InferenceError` naming `maxTokens`; the sequence's existing blocks are
  unaffected, other sequences unaffected.
- **Context overflow**: `step` with `cursor == maxTokens` → typed error.
- **No cacheable attention**: zero SDPA nodes in the walk → typed error
  at `inference` time, before any allocation.
- **Non-Metal device**: CPU falls back to gather + `sdpa_forward`; other
  backends rejected at construction.
- **Dtype**: f32 v1 (matches the flash kernel); other dtypes rejected at
  construction.

## Non-goals

- **Continuous batching** (one program run stepping many sequences):
  sequences step independently in v1. The pool and block tables are
  designed for it; the decode program's batch dim is not.
- **Prefix sharing, parallel sampling, beam search** (block-table
  aliasing, refcounts, copy-on-write).
- **Cross-request / cross-process KV reuse** (LMCache-style content
  addressing, tiered offload, P/D disaggregation).
- **Sampling**: argmax/temperature/top-k stay in user code over the
  returned logits, as in `nano-gpt` today.
- **Cache quantization** (f16/f8 KV), **GQA/MQA** (`multiHeadAttention`
  is MHA-only today), **RoPE** (would make position offsets structural
  rather than data — compatible, not required).
- **A standalone `KvPool.make` constructor.** Pool geometry is derived
  from the model at `inference` time (single source of truth). Explicit
  pools — for capacity sharing across same-architecture programs, e.g.
  speculative draft/target pairs — are a documented extension point, to
  be added with bind-time geometry validation when a use case exists.

## Alternatives considered

- **Growing contiguous cache** (concat K/V each step): rejected — input
  signature changes every step, churning both the `ProgramCache` and the
  flash pipeline cache; a compile per token is worse than no cache.
- **Bucketed immutable cache** (pad to powers of two): keeps tensor
  immutability with log₂(max) programs, but copies the whole cache at
  every bucket crossing and still caps at one sequence per buffer. The
  pool subsumes it at fixed shapes with zero copies.
- **Hidden per-session state on the compiled program** (a
  `DecodeSession` owning its cache): rejected — it re-introduces mutable
  slots behind the program handle, breaking RFC 0008's functional
  rebinding and making parallel calls of one program unsafe by
  construction. State belongs in explicitly acquired `Sequence`s.
- **Post-hoc graph rewrite** (compile forward, then rewrite SDPA nodes
  into scatter+paged): the rewrite must reverse-engineer structure (head
  splitting, position wiring) that the builder states directly. Node-
  construction reinterpretation during a decode-mode walk is mechanical
  and total; graph archaeology is neither.
- **A `decode` builder on `Model`** (authors write forward twice):
  rejected — the decode difference is localized to two primitives; making
  every model author restate the architecture invites divergence between
  the two definitions.

## Acceptance criteria

1. **Parity**: greedy (argmax) generation through `InferenceProgram`
   matches the naive full-window loop token-for-token on the same params,
   for prompts of several lengths, across pool block boundaries
   (`blockSize` not dividing the context length).
2. **Shape stability**: `stats().compiled` counts exactly one decode
   program and one prefill program per distinct prompt length across an
   entire multi-sequence generation run; the flash pipeline cache is not
   re-entered per step.
3. **Concurrency**: N sequences stepped concurrently on one program
   produce results identical to N sequential runs (same RNG draws per
   sequence).
4. **Isolation**: exhausting the pool in one sequence errors that
   sequence's `step` and leaves concurrent sequences unaffected.
5. **Lifetimes**: closing the `inference` scope releases the pool
   (device memory returns); closing a sequence scope returns its blocks
   to the free list.
6. **Migration**: `nano-gpt.ts` generates through `Model.inference` —
   `prefill` once per prompt, `step` per token — with output quality
   unchanged and per-token latency flat in context length (no quadratic
   growth).

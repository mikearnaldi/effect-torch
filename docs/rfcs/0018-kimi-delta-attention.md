# RFC 0018: Kimi Delta Attention — Chunked Linear Attention, Recurrent Decode State, and No-RoPE Hybrid Stacks

- **Status**: Implemented — fused forward, closed-form backward, and
  recurrent decode kernels on Metal (composed reference paths on both
  backends), hybrid generation, and training, all gated.
- **Created**: 2026-08-08
- **Depends on**: RFC 0007 (fusion), RFC 0008 (compilation), RFC 0010
  (inference), RFC 0012 (dtypes), RFC 0013 (batched decode), RFC 0015
  (native backend), RFC 0016 (frozen-program memory)
- **Updates**: —

## As-built notes (2026-08-08)

Deviations from the phase sketch, all simplifications validated by the
gates:

- **Sequence-owned state, not a pool.** KDA and conv state are fixed-
  size per sequence, so they live directly on the sequence state
  (`SeqState::kda_states`/`conv_states`, lazily allocated from the
  decode geometry) instead of a slot arena. No blocks, hashing, or
  eviction; rollback is a pre-run snapshot (deep copy on Metal).
- **One stateful node pair covers prefill and decode.** `KdaChunk` →
  `KdaRecurrence` and `ShortConv1d` → `ConvState` rewrites run for all
  three programs; the stateful eval arms handle any T. Chunked prefill
  right-pads, so pad rows are masked to identity updates (β=0, log
  decay 0) and conv windows shift in only real rows — the KV path's
  pad-overwrite trick does not transfer to running state.
- **Phase 4 (backward) landed before the fused forward kernels.** The
  closed-form adjoint (`KdaBackward` + `KdaBackwardOut`, plus
  `ShortConv1dBackwardX/W`) runs the reverse-time delta-rule adjoint
  with per-chunk recompute — chunk-start states retained, per-token
  states recomputed within the chunk; no O(T) retention. Verified
  against central finite differences of the forward (< 1e-4 rel, f64,
  across chunk boundaries) on all five operands, plus TS gradchecks and
  an end-to-end training smoke.
- **Phase 3 fused kernels landed (forward, backward, decode).**
  Sequential register-resident scans, not the WY chunk pipeline: at
  Dk/Dv ≤ 128 the chunk algebra's materialization costs dominate on
  this hardware (the MLX evidence agrees — register recurrent kernels
  are competitive-to-better on Max-class parts), so the fused form is
  the plain delta-rule scan, one launch per (batch·head) strip. The
  backward keeps one threadgroup per batch·head (the dk-side gradients
  sum over the full value dim), recomputes chunk-start states then
  per-token states into fp32 workspace scratch, and walks the adjoint
  in reverse with threadgroup-memory cross-row reductions. Decode
  kernel: one launch per slot per KDA layer (49.4 vs 39.5 tok/s on the
  30M example). Training (30M, block 256, batch 64, M4 Max): **2.4 →
  0.94 s/step (2.5×)**, within 1.5× of the fused-attention baseline
  (~0.6 s/step). Parity: fused forward/backward match the composed
  (finite-difference-verified) reference to < 1e-3 including stateful
  carry-over across chunk boundaries; `EFFECT_TORCH_NO_KDA_FUSED`
  restores the composed paths everywhere for A/B.
- **Pure-KDA stacks** (zero KV layers) work, but have no block-anchored
  prefix cache.
- **Hybrid prefix caching** restores KV blocks plus KDA/conv recurrent-state
  snapshots at completed block boundaries.
- Gates: chunked forward matches a per-token recurrent oracle on CPU
  and Metal (< 5e-4 rel, cargo; 2e-3 vitest), hybrid and pure-KDA
  greedy generation match full-forward naive generation token-for-token
  (single and batched), backward finite-difference parity, 654/654
  core vitest, cargo workspace green; the fineweb-kda example trains
  (loss 11.33 → 8.75 over 12 steps) and generates through the fused
  decode path.

## Summary

Add Kimi Delta Attention (KDA) — the linear-attention mechanism from
Moonshot's Kimi Linear (arXiv:2510.26692), deployed at scale in Kimi K3
(arXiv:2607.24653) — as first-class native operations:

1. **`KdaRecurrence`** — a fused single-token decode kernel with a
   fixed-size per-(layer, sequence) recurrent state, integrated into the
   compiled generation machinery alongside `KvAttention`.
2. **`KdaChunk`** — the chunked parallel form (chunk 64, sub-chunk 16,
   WY representation + UT transform) for prefill and training, begun as
   graph composition over existing ops and fused into dedicated Metal
   kernels behind a measurement gate.
3. **`KdaBackward`** — a closed-form multi-output backward, recompute-
   first, following the `SdpaBackward` template.

KDA layers carry positional information in their learnable, data-
dependent state transition (`Diag(α) − βkkᵀDiag(α)` — diagonal decay
plus Householder reflection) and use **no positional encoding**; full-
attention layers in a hybrid stack can then run NoPE, which is exactly
the Kimi K3 configuration (69 KDA + 24 NoPE gated-MLA layers, no RoPE
anywhere in the language model). This RFC therefore also makes
`rotaryEmbedding` optional per attention layer rather than a stack-wide
assumption.

## Motivation

The library today implements exactly one attention mechanism: causal
softmax attention with RoPE (`Tensor.scaledDotProductAttention`,
`Tensor.rotaryEmbedding`, `Model.multiHeadAttention` with the native
`Sdpa → KvAttention` decode rewrite). There is no scan primitive beyond
`cumsum`, no recurrent-state cache (the decode architecture is
KV-cache-only), and RoPE is effectively mandatory.

KDA matters for three reasons:

- **Decode economics.** KDA layers hold a fixed `128×128` fp32 state per
  head regardless of context length. In Kimi Linear's 3:1 hybrid this
  yields ~75% KV-cache reduction and up to 6× decoding throughput at 1M
  context versus full MLA. For this library — where generation decode is
  the flagship inference path (RFC 0010/0013) — a fixed-size state is a
  strictly better fit for the frozen-program model than a growing paged
  cache: no block tables, no content hashing, no eviction, constant
  per-step work.
- **No RoPE.** The delta-rule transition is itself a multiplicative
  relative-position encoding (the Householder-reflection analogue of
  RoPE's orthogonal rotations; Kimi Linear §"Discussions"). Empirically,
  adding RoPE back to KDA layers *hurts* long-context performance, and
  K3 dropped RoPE entirely. Supporting KDA means supporting stacks with
  no positional encoding at all — which our current decode rewrite
  assumes away (RoPE → cursor-offset is one of its two rewrites).
- **Lineage.** DeltaNet → Gated DeltaNet (arXiv:2412.06464) → KDA is the
  most credible linear-attention line at frontier scale (Qwen3-Next,
  Kimi Linear, K3). The infrastructure built here (state pool, chunked
  scan, UT transform) transfers to GDN/MLA-hybrid variants with per-
  scalar gates as a strict simplification.

External evidence shapes the build order. On Apple Silicon, a **pure-ops
chunked** gated-delta forward+backward (mlx-lm PR #1389: blocked 16×16
forward substitution, log-domain decay, per-chunk checkpointing) is
~60× faster and ~21× lighter than the sequential loop at T=2048, and
*beats a dedicated Metal VJP kernel on M4 Max* (matmul throughput beats
latency-bound sequential sweeps on Max-class chips; the fused kernel
wins ~1.2–2× on Ultra). Conclusion: for prefill/training, composed
chunk math on our fused gemm + elementwise infrastructure is a
competitive baseline and the right first implementation; hand-written
kernels are justified only where profiling shows HBM round-trips
dominating (the intra+solve fusion and the inter-chunk state scan). For
**decode**, the fused recurrent kernel with register-resident state
(mlx-lm `gated_delta.py`, FLA `fused_recurrent_kda`) is unambiguously
the right shape — one walk per token cannot afford composed-op overhead.

## Background: the math and the numerics contract

Per head (d_k = d_v = 128 in all known deployments), per token:

```
q, k  = L2Norm(Swish(ShortConv₄(W_{q,k} x)))        # causal depthwise conv
v     = Swish(ShortConv₄(W_v x))
α_t   = decay(W_α↑ W_α↓ x) ∈ (0,1]^{d_k}            # channel-wise gate, log-space
β_t   = sigmoid(W_β x) ∈ [0,1]                      # scalar per head
S_t   = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ   ∈ R^{d_k×d_v}
o_t   = RMSNorm(S_tᵀ q_t) ⊙ sigmoid(W_g↑ W_g↓ x)    # then W_o
```

**Chunked parallel form** (FLA `chunk_kda`, the reference; chunk C=64,
sub-chunk BC=16), six stages:

1. **Gate + chunk-local log-cumsum** — decay in **log2 space**
   (`g *= 1/ln2` once; all kernels use `exp2`).
2. **Intra-chunk matrices** — `Aqk[i,j] = scale·⟨q_i⊙2^(g_i−g_j), k_j⟩`
   (lower-tri), `Akk[i,j] = β_i⟨k_i⊙2^(g_i−g_j), k_j⟩` (strictly lower).
3. **UT transform** — invert `(I + tril(Akk))` per chunk: forward
   substitution on the 16×16 diagonal blocks, then block merge to 64×64.
   Never divide by cumulative decay: reciprocals are rewritten as
   `exp2(pivot − g)` rescaling of the *other* matmul operand.
4. **WY representation** — `u = A(v⊙β)`, `w = A(k⊙β⊙2^g)`,
   `kg = k⊙2^(g_last−g)`.
5. **Inter-chunk state scan** — sequential over chunks, MMA per step:
   `v_new = u − w@h`; `h ⊙= 2^g_last`; `h += kgᵀ@v_new`; per-chunk
   boundary states stored for output/backward.
6. **Output** — `o = scale·(q⊙2^g)@h_chunk + tril(Aqk)@v_new`.

**Numerics contract** (from FLA + MLX field reports; treat as spec):

- State, gate cumsum, diagonal solve blocks, and all gradient
  accumulators in **fp32**; stored intermediates (Aqk, Akk, w, u, kg,
  chunk states) in the model dtype (bf16).
- Exponents always pivot-subtracted (sub-chunk start, chunk end, or
  mid-block) so `exp2` arguments stay bounded (<85).
- Optional `safe_gate` lower-bound clamp on the log gate (−5 ≤ lb < 0)
  bounds the decay range, enabling 16-row MMA diagonal kernels; without
  it the diagonal blocks use a token-parallel scalar loop.
- The 16×16 blocked substitution is **required for stability** — a
  global Neumann expansion overflows fp32 on repeated keys, and 32×32
  blocks diverge (rel ~2e11) on collinear β→1 stress while 16×16
  degrades gracefully (~4e-4).
- Decode drift: the recurrence accumulates fp32 error over ~20k+ steps;
  MLX needed Kahan-compensated summation in the decode kernel. We adopt
  Kahan in the `kv_mem = Sᵀk` reduction from the start (cost measured at
  +1.4% step time upstream).
- Decode kernel uses natural `exp` on raw log gates; chunked path uses
  `exp2` on the pre-scaled cumsum. Do not mix.

## Design

### New node kinds and public API

Two semantic nodes in `crates/graph` + `NodeOperationMap`
(Runtime.ts:202), with the standard ~10 touch points per node (graph
validation/`node_children`/`remap_children`, both TS adapters, both NAPI
crates, autodiff):

```ts
// Semantic (prefill/training) form. All inputs [B, H, T, D] contiguous,
// gates [B, H, T, D] (log-decay, pre-cumsum) and [B, H, T] (beta).
Tensor.kdaChunk(q, k, v, logDecay, beta, options?: {
  scale?: number          // default 1/sqrt(D)
  initialState?: Tensor   // [B, H, D, D] fp32, default zero
  safeGateLowerBound?: number
}): Tensor                 // [B, H, T, D]

// Decode form — emitted ONLY by decode_rewrite, never user-constructed.
// Holds { scale, layer } and reads/writes pool state via the eval context,
// exactly like KvAttention.
NodeKind::KdaRecurrence { q, k, v, logDecay, beta, scale, layer }
```

`Model.kimiDeltaAttention` (new constructor, mirroring
`multiHeadAttention` at Model.ts:558): fused qkv+gate projections, short
causal convs (`pad` left 3 + depthwise `conv1d`, Tensor.ts:2909),
L2-normed q/k (composed `norm`, Tensor.ts:1765), gate projections
(low-rank α, scalar β), `kdaChunk` core, sigmoid-gated composed RMSNorm
out. Parameters include `A_log`/`dt_bias` kept in fp32 under mixed
precision (the `cast_predicate` lesson from MLX). A hybrid stack is just
`chain`/`residual` composition of `kimiDeltaAttention` and
`multiHeadAttention` blocks in any ratio; `multiHeadAttention` gains
`rotary: false` for NoPE layers.

### The recurrent-state pool

New native object per backend, deliberately simpler than the KV pool
(RFC 0013): state is fixed-size, so there are no blocks, hashes, LRU, or
eviction.

- `NativeKdaPool::new(layers, heads, head_dim)` allocates per layer a
  flat slab `[maxSeqs? …]`— concretely: per (layer, slot) one fp32
  `[H, D, D]` matrix (128×128×4 B = 64 KB/head) plus three fp32 conv
  states `[3, D_proj]`. Allocated at pool construction via
  `MetalDevice::alloc` / the CPU slab equivalent — **outside any program
  walk**, per the arena-escape rule of RFC 0016 (KV-pool precedent,
  runtime-metal napi/mod.rs:4788-4804).
- A KDA sequence handle owns one slot index per pool. Hybrid generation
  binds **both** a KV pool (attention layers) and a KDA pool (linear
  layers) to the run; `KvContext` generalizes to a `DecodeContext` with
  optional KV and KDA halves.
- **Rollback**: KV's block-refcount rollback does not apply to in-place
  matrix state. On run failure, restore slot states from a pre-run
  snapshot taken into a scratch region of the same slab (one 64 KB/head
  memcpy per layer per active slot — negligible next to a failed walk).
- **Prefix caching** stores an immutable recurrent-state snapshot beside each
  completed content-addressed KV block boundary. A hybrid match takes the
  deepest boundary with both blocks and a snapshot, then resumes chunked
  prefill from it. Pure recurrent stacks have no block-anchored hash chain
  and therefore no prefix reuse in this version.

### decode_rewrite changes

`decode_rewrite` (runtime-metal napi/mod.rs:4524, runtime-cpu
napi/mod.rs:3198) gains a third arm beside `Sdpa → KvAttention` and
`RotaryEmbedding → cursor-offset`:

- `KdaChunk` with T matching the decode program's token count →
  `KdaRecurrence{scale, layer}` with `layer` assigned in the same
  post-order encounter counter as attention layers (prefill/decode
  traces agree by construction).
- Geometry: `DecodeGeometry` currently enforces one shared `(kvHeads,
  headDim)` across all layers. It becomes two independent uniform
  geometries — attention `(kvHeads, headDim, layers_kv)` and KDA
  `(heads, headDim, layers_kda)` — each enforced only across its own
  layer kind. A pure-KDA stack has zero KV layers and must not allocate
  a KV pool; a pure-attention stack compiles exactly as today (no
  regression surface).
- NoPE stacks simply contain no `RotaryEmbedding` nodes; the rewrite
  already tolerates their absence. `PositionEmbedding`-based stacks are
  unaffected.
- The short conv's 3-token window is **not** pool state at the node
  level: the decode program keeps the last 3 projected activations as a
  small cyclic buffer inside the `KdaRecurrence` eval arm, fed from the
  slot's conv-state slice — same layout as MLX's `ArraysCache`.

### Phase plan and gates

**Phase 1 — Decode: `KdaRecurrence` kernel + state pool (Metal + CPU).**
Fused single-token kernel, one threadgroup per (slot, head, V-slab of
32), state tile `[128, 32]` fp32 in registers, `simd_sum` reductions for
`Sᵀk` (Kahan-compensated) and `Sᵀq`; CPU backend: composed per-token
math over pool slabs (correctness reference, single-threaded per
RFC 0015's CPU stance). Gates: numerical parity vs a step-by-step
naive reference (FLA `naive_recurrent_kda` ported to a vitest fixture)
to <1e-4 rel on outputs and state after 8k steps; tok/s improvement on
a hybrid 3:1 toy stack vs full-attention decode at 32k+ context; no
regression on pure-attention stacks (628 vitest + cargo suite).

**Phase 2 — Chunked forward, composed.** `kdaChunk` evaluated as a
fixed graph expansion on both backends: log-cumsum via existing
`cumsum`, decay-rescaled matmuls, composed `tril` masks, the 16×16
substitution **unrolled in the graph** (static 16-step loop per
diagonal block — verbose but shape-specialized, and the elementwise
fusion pass (rewrite.rs:343) collapses the glue), state scan as a
Python-free unrolled loop over `T/64` chunk matmuls. This is the MLX
PR-#1389 strategy on our infrastructure, not a naive per-token loop.
Gate: prefill throughput within ~2× of the fused-kernel projection;
correctness vs FLA `naive_chunk_kda` fixture <1e-3 rel (bf16
reassociation budget). New native helpers only if profiling demands
(native `tril` is the likely first extraction — the TS-composed mask
materializes an i64 graph per call).

**Phase 3 — Fused `KdaChunk` Metal kernels.** Multi-dispatch eval arm
(flash-bwd/split-K precedent) implementing stages 1–6 as ≤4 kernel
launches: (a) gate+cumsum, (b) fused intra+Akk+UT-solve (one
threadgroup per chunk; the 16×16 substitution is a sequential loop in
one threadgroup, as in FLA), (c) fused state-scan+output (sequential
over chunks, MMA per step using the `gemm.rs` simdgroup-fragment idiom
— 8×8 fragments, bf16 staging, f32 accum; threadgroup budget check:
state slabs staged 64-wide, C=64/D=128 tiles must respect the same
~32 KB wall that gates flash at D=DV=64, else drop to C=32). Gate:
≥1.5× over Phase 2 on prefill at T≥8k on M4 Max, or the phase is
deferred (the composed form stays, matching the MLX outcome on
Max-class parts).

**Phase 4 — Closed-form backward.** `KdaBackward{q,k,v,logDecay,beta,
g, fwd}` multi-output node + `KdaBackwardOut` pickers (autodiff
template: `SdpaBackward`, autodiff/lib.rs:1188-1211). Recompute-first:
save (q, k, v, β, Akk, Akk⁻¹) from forward; recompute g/w/u/h per chunk
in backward (FLA's default `disable_recompute=False` policy), with the
chunk-boundary states optionally stashed via `Evaluator::multi` as flash
stashes (O, L). Gradients: dq, dk, dv, dlogDecay (reverse chunk-cumsum),
dβ, d(initialState). The UT-transform adjoint
`dA ← −tril(Aᵀ(dA·β_col)Aᵀ)` and the reverse cumsum are the only
non-matmul pieces. Second order rejected, matching Sdpa. Gate: gradient
parity vs composed-autodiff of the Phase-2 graph (finite-difference
spot checks on gate parameters) <1e-3; training-step time and peak
memory measured on a 30M hybrid preset.

**Phase 5 — Model/program integration.** `Model.inference` compiles
hybrid prefill+decode programs with both pools bound; `Generation.step`
works unchanged (pool selection is program-internal); `noRoPE` hybrid
presets documented; CPU parity throughout per the standing rule.

### Alternatives considered

- **Naive per-token graph loop for training**: rejected — O(T)-node
  graphs OOM autodiff at T≥2048 (mlx-lm issue #1206 reproduced the same
  wall) and are 60× slower than chunk math.
- **Fused kernels first**: rejected on the MLX evidence — composed chunk
  math wins on Max-class hardware and derisks the numerics before kernel
  effort is spent.
- **Generic `associative_scan` primitive instead of KDA-specific nodes**:
  rejected for now — the delta rule is not associative in its naive
  operator, the WY/UT chunk form is the actual fast path, and a generic
  scan would not subsume the UT transform. Revisit if a second SSM
  family lands.
- **Triton/external kernel deps**: rejected per RFC 0016 Phase 5's
  recorded stance (no CUDA-specific stack; Metal kernels are MSL in-
  crate).

## Risks

- **Numerical fidelity is the dominant risk.** The UT transform, log2
  decay, and pivot discipline exist because naive formulations silently
  diverge; every phase gates against an independent reference fixture
  (ported FLA naive kernels), not against our own composed form.
- **Decode-state rollback correctness**: in-place state plus batched
  runs plus failure paths is a corruption surface the KV pool never had
  (blocks are copy-on-alloc). Mitigation: snapshot-restore under the
  existing per-sequence run locks; poison-test fault injection in
  cargo tests.
- **Threadgroup memory at C=64, D=128** may force C=32 or slab-split
  state tiles in Phase 3 — accepted fallback, changes no semantics.
- **Two pool types in decode** is permanent complexity in
  `decode_rewrite`/`run_inner`. Contained by the `DecodeContext`
  generalization and by keeping pure-attention compilation byte-for-byte
  on today's path.
- **Prefix-cache scope**: hybrid stacks reuse prefixes through boundary
  snapshots; pure recurrent stacks remain full-prefill because they have no
  KV blocks anchoring the content-addressed chain.
- **Scope discipline**: KDA only; GDN/MLA-hybrid variants are
  follow-ups that reuse the pool and scan machinery.

## Verification

Standard gates every phase: `cargo test`, `pnpm vitest run` in
`packages/core`, warning-free `cargo build`, `pnpm typecheck`,
`git diff --check`; native perf builds via `napi build --release`;
heavy runs under `EFFECT_TORCH_MEMORY_CAP_MB`. Phase-specific:

- Reference fixtures: `naive_recurrent_kda` and `naive_chunk_kda`
  ported to vitest with fixed seeds; forward parity <1e-3 rel (bf16),
  decode-state parity <1e-4 after 8k steps, gradient parity <1e-3
  against composed autodiff.
- Decode: tok/s vs context length curves on a 3:1 hybrid toy stack
  (expect flat KDA-layer cost; headline metric is hybrid vs
  pure-attention at 32k/128k).
- Training: 30M hybrid preset step time + peak memory vs the
  pure-attention baseline at equal params; loss-curve equivalence on
  identical init/data order within bf16 reassociation, per the RFC 0016
  gate standard.
- Hybrid regression: pure-attention stacks must compile to the identical
  program structure as before this RFC (no new nodes, no pool
  allocations) — asserted in cargo tests on `decode_rewrite` output.

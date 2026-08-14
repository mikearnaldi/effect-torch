# RFC 0014: Speculative Decoding — Verification as a Primitive

- **Status**: Draft
- **Created**: 2026-08-02
- **Depends on**: RFC 0010 (inference), RFC 0013 (batched decode)
- **Updates**: —

## Summary

Speculative decoding: a cheap proposer guesses the next k tokens, the
target model verifies them in a single pass, the longest valid prefix
is accepted, and the kv cache rolls back to the cut. Every variant of
the idea — draft models, Medusa heads, EAGLE feature heads, n-gram
lookahead — reduces to the same verify/rollback primitive; only the
proposer and the mask shape differ. This RFC specifies that primitive
(native seeded sampling, sequence truncation, a native speculative
round engine) and the proposers in order: linear draft+verify first,
n-gram lookahead second, tree verification (Medusa/EAGLE) as recorded
future work.

## Motivation

Decode is serial: k tokens cost k full walks. The inference stack we
have (paged kv, prefix cache, chunked prefill, batched decode, paged
kernels) already makes single-token steps fast; the next order-of-
magnitude win is doing fewer of them. Speculative decoding is the only
technique in the field that cuts serial steps *exactly* — the output
distribution (speculative sampling) or tokens (greedy) is provably
identical to plain generation, so it composes with everything else we
ship without changing semantics.

## Prior art (the research summary)

The four families, per the survey behind this RFC:

- **Draft+verify** (Leviathan et al. 2211.17192; Chen et al.
  2211.15737): a small draft model proposes k tokens; the target
  verifies in one pass. Speculative sampling keeps the exact
  distribution; greedy verification keeps exact tokens. 2–3× typical.
  Draft sweet spot ≈ 5–10% of target parameters.
- **Medusa** (Cai et al. 2401.10774): k extra heads on the target
  predict t+1…t+k; candidates form a **token tree** verified in one
  pass with tree-attention masks. No draft model; needs head
  fine-tuning (Medusa-1 keeps the backbone frozen). 2.2–3.6×.
- **EAGLE** (Li et al. 2401.15077): best acceptance rates published
  (70–80%+): the draft head autoregresses second-to-top-layer
  *features* (easier than tokens) fed one-step-advanced token
  embeddings. Same tree verification as Medusa. 2.7–3.5×.
- **Model-free lookahead** (LLMA, Yang et al. 2304.04487; n-gram
  variants in llama.cpp/vLLM): draft tokens are copied from the
  prompt/reference by n-gram matching. Zero training, zero extra
  memory. Big wins on retrieval/editing/repetitive workloads, modest
  on open-ended generation. SpecInfer (Miao et al. 2305.09781) is the
  systems form of tree verification.

**The reduction**: propose (linear chain or tree) → verify in one
pass → accept the longest valid prefix → roll the kv cache back to
the cut. Acceptance rate is the only quality that matters: speedup ≈
(accepted + 1) / (cost of one target round), so proposer quality
decides the ceiling; the engine is proposer-agnostic.

## Design

### Native seeded sampling

`Tensor.sample(logits, { temperature, topK?, topP?, seed, counter? })`:

- `temperature: 0` — argmax (the greedy path; exact-token speculative
  verification compares these).
- `temperature > 0` — numerically stable categorical sampling;
  top-k is applied before the smallest top-p prefix containing the
  crossing token.
- Seeded SplitMix64 in Rust keys each stateless draw by
  `(seed, counter)`, so retries and concurrent generations share no
  hidden mutable RNG and draft/target streams can remain independent
  but reproducible.
- The direct runtime extension borrows native logits storage. CPU scans
  its tensor buffer directly; Metal samples with GPU kernels and returns
  only the winning id across N-API. Positive-temperature Metal top-p
  filtering requires top-k in `1..=64`; greedy and unfiltered categorical
  draws scan the complete vocabulary.
- Generation additionally exposes required `addSampled`/`stepSampled`
  operations on every backend.
  CPU samples before publishing its native state transaction. Metal keeps the
  stronger fused device path:
  decode, bounded top-k/top-p selection, the final fence, and transactional
  state publication share one explicit submission, so no logits tensor is
  published and no second host synchronization is introduced. Existing
  logits-returning generation operations remain available for custom host
  samplers and other consumers.

### Sequence truncation (`Sequence.truncate(position)`)

The rollback primitive. Rules:

- Truncation targets an arbitrary position `j` in the **live window**
  (`j ≥ frontier`). Rows beyond `j` in a partially-filled block are
  never attended again (gathers stop at the cursor), so only blocks
  *fully* beyond `j` are unreferenced to the pool; the partial block
  is kept and its tail rows harmlessly overwritten later.
- Hash-chain rebuild: `SeqState` keeps a rolling ring of the last
  `2 × blockSize` appended tokens, so `pending` (the partial block's
  tokens) is exactly reconstructible; `last_hash` reloads from the
  pool's per-block hash store (or the seed at position 0).
- Rejected rows' completed blocks land in the prefix cache with their
  (rejected-content) hashes — harmless, since chained hashes can only
  match the identical wrong token stream; the shared accepted prefix
  stays reusable.
- Anything deeper than the live window is a typed error (re-prefill).

### The speculative round engine (native)

One napi engine, rounds driven internally — no per-token host
round-trip:

```
generate(prompt, k, maxNew, sampler) -> Vec<u32>
  prefill prompt on draft and target sequences
  loop until maxNew or EOS:
    draft: k decode steps (native sampler per step)
    target: one prefill-chunk run over the k draft tokens (advance = k)
    j = longest prefix where target argmax == draft token
    emit j accepted tokens + one target-sampled token at the cut
    truncate both sequences to prefix + j (+1 corrected)
```

- **Greedy exactness**: target argmax at the cut is the corrected
  token; plain greedy produces the identical stream. This is the
  primary test invariant.
- **Sampling exactness** (speculative sampling, Leviathan §3.3):
  accept token i with probability `min(1, p_target(i) / p_draft(i))`;
  on rejection, resample from `normalize(max(0, p_target − p_draft))`.
  Requires per-token probabilities from both models (softmax over
  logits, native) and the seeded streams above.
- Engine inputs: two `InferenceProgram`s (draft, target) with their
  frozen programs, pools, and parameters exposed through internal
  handles; config = `{ k, maxTokens, sampler }`.
- Interaction with existing machinery: chunked prefill already runs
  the verification pass (a prefill chunk with `advance = k` is exactly
  a verify); the paged attention kernel handles it unchanged; prefix
  cache deduplicates draft/target-shared prefixes when architectures
  match (they don't share pools — separate programs).

### The n-gram proposer (phase 2)

Same verify/rollback, no draft program: the proposer scans the
sequence's own token history for the longest suffix matching an
earlier n-gram and continues from its successor. Config: n-gram size
and lookahead length. Wins on retrieval/editing/repetitive text at
zero cost; acceptance on open-ended text is low but never negative
(rejection costs one target pass ≈ one plain step plus ε).

### Tree verification (recorded future work)

Medusa/EAGLE-style proposers emit *trees*, not chains: verification
needs tree-attention masks (each candidate attends its ancestors
only). In our terms: the paged kernel takes a per-row mask buffer;
the prefill path takes candidate trees as input. Head-training
combinators (`Model.medusaHeads(model, k)` / an EAGLE feature head)
are a Model-level design of their own. Recorded here so the phase-1
engine keeps verify mask-shape-agnostic: today's mask is "causal over
a chain"; the tree mask is a generalization, not a redesign.

## Non-goals

- Medusa/EAGLE heads and training recipes (phase 3; needs tree masks
  and a Model-level head API).
- Cross-model kv sharing (draft and target keep separate pools;
  architectures and geometries may differ entirely).
- Beam/tree search over the target distribution (fork semantics —
  covered by explicit `Sequence.fork` when a search-shaped caller
  exists).
- CUDA (Metal + composed CPU fallback as everywhere else).

## Alternatives considered

- **TS orchestration loop**: rejected — per-token host round-trips
  destroy the speedup the technique exists to buy; rounds must be one
  native call.
- **Medusa-first**: rejected — head training + tree masks is the
  bigger commitment with the same engine primitives underneath; the
  linear loop ships the primitives with exactness tests today.
- **Rollback by re-prefill** (discard and recompute the accepted
  prefix on each round): rejected — O(prefix) recompute per round
  erases the win; truncation is O(rejected blocks).
- **Draft via distillation only**: n-gram proposer included precisely
  so the feature is useful with zero training.

## Acceptance criteria

1. **Greedy exactness**: speculative generation (draft ≠ target)
   matches plain greedy generation token-for-token across pool block
   boundaries, windows, and prompt lengths.
2. **Degenerate draft**: draft == target accepts k every round
   (verifies the accept path end-to-end).
3. **Sampling determinism**: seeded sampling reproduces token streams
   across runs; speculative sampling matches plain seeded sampling
   statistically (χ² over a fixed corpus at fixed seed, loose
   tolerance) — and exactly when p_draft ≡ p_target.
4. **Rollback**: truncation restores cursor/pending/hash state
   exactly (continued generation after a truncate matches a fresh
   sequence's), returns full blocks to the pool, and rejects
   out-of-window positions typed.
5. **N-gram proposer** (phase 2): on repetitive input (repeated
   paragraphs), acceptance > 0 and output matches plain greedy
   token-for-token; on random input, output still matches (rejection
   is safe).
6. **Throughput sanity**: with a good draft (acceptance ≥ 60%),
   speculative decode is measurably faster per generated token than
   plain decode on Metal.

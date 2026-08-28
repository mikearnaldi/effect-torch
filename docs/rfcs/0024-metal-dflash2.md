# RFC 0024: Metal DFlash 2 and Adaptive Speculation

- **Status**: Draft
- **Created**: 2026-08-28
- **Depends on**: RFC 0019 (executable compilation), RFC 0022 (pretrained
  quantized inference), RFC 0023 (batched speculative generation)
- **Updates**: RFC 0023 phases 3 and 4

## Summary

This RFC makes Muse-Glimmer DFlash consistently useful on Apple Metal. It
defines the remaining work as one cohesive path:

1. optimize small-row quantized target verification, beginning with direct
   multi-row Q3_K and Q2_K kernels;
2. load and execute the Muse-Glimmer DFlash 2 checkpoint without silently
   treating it as original DFlash;
3. keep proposal tokens, verification samples, acceptance, replay selection,
   and provisional state on the device until one completed token page is
   returned;
4. select a runtime draft width from measured Metal costs and accepted-length
   survival rather than aggregate token acceptance; and
5. permit an adaptive scheduler to select `K_policy = 0`, meaning an ordinary
   target round with proposer maintenance, when measured speculation cost per
   output token is worse.

The `K_policy = 0` path is a performance policy selected before a round. It is
not a CPU fallback, capability fallback, error recovery path, or retry after
partial execution. Speculative failures remain transactional errors. Fixed
scheduling never disables speculation for performance.

The primary target is Muse-Glimmer-30B with its 16-row DFlash and DFlash 2
checkpoints, batch size one, greedy and positive-temperature generation, and
Q2_K/Q3_K-heavy quantized target weights. The kernels and orchestration remain
general runtime facilities rather than model-name branches.

## Motivation

The current implementation has the correct high-level structure:

- a target-coupled DFlash artifact with five target hidden-state taps;
- one parallel anchor-plus-mask proposal program;
- causal all-row target verification;
- exact greedy matching and positive-temperature target-sample matching;
- replay of accepted target features into proposer state;
- transactional target/proposer publication;
- M=8 and M=16 quantized verification executables; and
- proposal, verification, replay, acceptance, and pool diagnostics.

It is not yet faster than ordinary decoding on the representative exact prompt:

```text
prompt: in TypeScript how do you define a union type?
output: 256 tokens
DFlash:  21.72 tokens/s
ordinary: 24.29 tokens/s
```

That run accepted 160 of 650 proposals across 95 rounds and committed 2.69
tokens per round. Average proposal and verification costs were approximately
17.8 ms and 102.6 ms. Q3_K and Q2_K linear operations accounted for roughly 77%
of verification time, while attention accounted for approximately 2%.

Verification also has a non-monotonic small-row kernel curve:

| Verification rows M | Observed cost |
|---:|---:|
| 1 | approximately 41 ms |
| 2 | approximately 67.7 ms |
| 4 | approximately 124.3 ms |
| 5 | approximately 215.9 ms |
| 8 | approximately 102.6 ms |

Acceptance-only scheduling cannot reason about this curve. A width with more
accepted tokens can still lose when its target pass enters a poor quantized
kernel shape. Improving proposal quality alone also cannot repair verification
that costs more per committed token than ordinary decode.

The existing DFlash path also differs from DFlash 2. It independently selects
one argmax token at every draft position. DFlash 2 adds block-local dynamic
convolutions and a sparse path selector that chooses a coherent chain through
each position's top candidates. Those additions improve accepted length, but
they do not remove the need for efficient target verification and cohesive
device orchestration.

## Goals

- Make target verification efficient for M=2 through M=8 quantized rows.
- Preserve exact greedy target output.
- Preserve the target distribution at positive temperature.
- Load the published Muse-Glimmer DFlash 2 safetensors/config checkpoint
  exactly, followed by a quantized GGUF path.
- Preserve the trained 16-row proposer geometry while allowing shorter logical
  prefixes for target verification.
- Keep full-vocabulary proposal rows private to the proposal executable.
- Move all intermediate speculative control off the host hot path.
- Select width using measured cost per expected committed token.
- Include an explicit, observable ordinary-round fallback in adaptive mode.
- Retain fixed mode for benchmarks, diagnosis, and users that require every
  eligible round to speculate.
- Match ordinary and speculative benchmark prompts exactly.

## Non-Goals

- Training or fine-tuning DFlash or DFlash 2 checkpoints.
- Candidate trees or ancestor target attention.
- EAGLE, MTP, Medusa, or DSpark implementation work.
- A CPU implementation of the optimized Metal kernels.
- Silent Metal-to-CPU operation fallback.
- Retrying an unsuccessful speculative round as ordinary decoding.
- Treating DFlash parallel marginals as causal proposal distributions.
- Exceeding the checkpoint's trained block geometry.
- Publishing a benchmark collected with Metal profiling enabled.

## Terminology

Let:

```text
B_train = total block rows used by the trained DFlash checkpoint
K_max   = B_train - 1 maximum proposal tokens after the anchor
K_policy = proposal tokens selected by fixed or adaptive policy
K_round = proposal tokens admitted after budget and state-capacity limits
M       = K_round + 1 target verification rows
A       = accepted proposal tokens in a round, 0 <= A <= K_round
P       = returned page length, normally A + 1
```

Later formulas use `K` as shorthand for an unconstrained positive
`K_policy = K_round`; sections that distinguish policy from admission use the
full names.

For Muse-Glimmer DFlash 2:

```text
B_train = 16
K_max   = 15
```

The proposal input is conceptually:

```text
[pending anchor, mask_1, mask_2, ..., mask_15]
```

`B_train` is a trained maximum geometry, not a claim that `K = 15` is the
fastest runtime choice. The current effect-torch DFlash executable always runs
the physical 16-row proposal graph and admits a logical prefix for target
verification. Consequently, reducing `K` saves verification work but does not
currently reduce all proposal work.

`K_policy = 0` means the adaptive scheduler selects RFC 0023's normal sampled
target round. The speculative proposer, verifier, and acceptance path do not
execute. Because DFlash is target-coupled, one-row proposer maintenance still
replays the consumed pending token's target features and installs the newly
sampled target token as the proposer pending token. This keeps RFC 0023's
target/proposer alignment invariant and allows a later positive-K probe.

`K_round` can also become zero after a positive `K_policy` when only one output
token or one state row remains. That is mandatory budget/capacity admission,
not cost fallback, and is reported separately.

## Decisions

1. DFlash 2 is a distinct checkpoint architecture and loader contract.
2. A DFlash 2 checkpoint must never be accepted by the original DFlash loader.
3. Muse-Glimmer's physical proposal geometry remains 16 rows in the first
   implementation.
4. Runtime width is a logical prefix `K` within the trained geometry.
5. Verification width is always stated as `M = K + 1`; APIs continue to state
   proposal limits as `K`.
6. Direct weight-reuse Q3_K and Q2_K kernels are the first performance work.
7. Kernel dispatch is shape- and codec-specialized at executable compilation;
   it is not selected by per-round timing branches.
8. DFlash 2 candidate selection remains sparse through verification and
   acceptance. Dense proposal vocabulary rows never cross N-API or become
   public tensors.
9. One native round owns proposal, target verification, target sampling,
   acceptance, replay, page assembly, and publication.
10. The host cannot inspect proposal tokens or accepted length between those
    phases.
11. Adaptive width uses accepted-length survival and measured phase costs.
12. Adaptive scheduling may select `K_policy = 0` when ordinary target decode
    plus required proposer maintenance is cheaper.
13. Fixed scheduling never selects `K_policy = 0` because of measured
    performance; budget or state capacity may still force `K_round = 0`.
14. A speculative execution error is returned; it is never retried through
    `K_policy = 0`.
15. No backend may use `K_policy = 0` to hide an unsupported operation, invalid
    checkpoint, missing kernel, or incompatible artifact.
16. Greedy DFlash output must be token-identical to ordinary greedy output.
17. Positive-temperature exactness uses target-sample matching unless the
    artifact exposes a validated causal normalized proposal distribution.
18. Performance claims use paired prompt manifests and release builds without
    `EFFECT_TORCH_METAL_PROFILE`.

## Existing Implementation

The RFC builds on code already present in the repository.

### Original DFlash graph

`packages/core/src/proposers/DFlash.ts` currently provides:

- canonical DFlash GGUF metadata and tensor validation;
- target-feature concatenation, projection, and normalization;
- reconstructed proposer K/V rows for every draft layer;
- one anchor and up to 15 mask rows;
- bidirectional attention within the current proposal block;
- independent per-position argmax candidate IDs; and
- replay over accepted target feature rows.

The artifact intentionally declares proposal probabilities unavailable. The
bidirectional per-position softmax rows are not a causal factorization and must
not be used for ordinary rejection sampling.

### Native speculative transaction

`crates/runtime-metal/src/napi/mod.rs` already owns speculative state and
publication inside one N-API call. It uses shadow target and proposer states,
samples target verification rows, computes accepted pages, replays accepted
features, and publishes only after the round succeeds.

The implementation still has intermediate host boundaries inside that native
call. It reads proposal IDs, builds verify metadata, reads sampled verification
tokens, compares acceptance, selects replay rows, and advances publication on
the Rust host. Avoiding JavaScript orchestration is necessary but insufficient;
these host decisions still create command-buffer synchronization points.

### Quantized target verification

`crates/runtime-metal/src/quantized.rs` contains packed-dot decode kernels and
half-operand MMA paths. M=8 and M=16 use dedicated MMA variants. Smaller
non-aligned Q2_K and Q3_K shapes use packed-dot paths that do not directly
reuse each loaded weight block across all active input rows.

### Current width heuristic

`select_verify_rows` estimates future survival by exponentiating the aggregate
ratio `accepted / proposed`. That ratio mixes all positions and does not
represent conditional survival at each accepted length. It also ignores the
measured verification cost curve. This RFC replaces it rather than tuning its
constant.

## DFlash 2 Checkpoint Contract

### Configuration

The loader reads the checkpoint configuration instead of inferring DFlash 2
from tensor presence. The published Muse-Glimmer `config.json` declares at
least:

```text
architectures[0]:                    DFlash2DraftModel
is_causal:                           false
dflash_config.block_size:            16
dflash_config.conv_kernel_size:      2
dflash_config.conv_group_size:       16
dflash_config.selector_rank:         256
dflash_config.selector_top_k:        16
dflash_config.target_layer_ids:      [1, 13, 25, 37, 49]
num_hidden_layers:                   5
hidden_size:                         6656
intermediate_size:                   19968
sliding_window:                      2048
dtype:                               bfloat16
```

The loader also validates the nested mask token, output multiplier, final logit
soft cap, top-level layer types, attention/head geometry, RoPE parameters,
normalization epsilon, vocabulary and token IDs, tensor names, tensor shapes,
and dtypes.

The published config does not contain a target checkpoint fingerprint. The
first-party Muse loader establishes compatibility through an explicit
Muse-Glimmer target profile: model architecture, vocabulary and token IDs,
hidden width, target layer count, tap IDs and shapes, RoPE contract, and shared
embedding/LM-head contracts must all match during `Model.inference`. A future
generic loader may additionally require a converter-supplied target fingerprint
but cannot pretend one came from the published config.

The first loader consumes the exact Hugging Face `config.json` and safetensors
layout. The later GGUF loader requires explicit DFlash 2 metadata for every
field that changes graph semantics. A converter may rename tensors but cannot
drop selector or convolution metadata.

Original DFlash and DFlash 2 use separate exported definitions. Hugging Face
checkpoints discriminate through `architectures`. Ecosystem GGUF conversions
use `general.architecture = dflash`, so GGUF discrimination uses explicit
DFlash metadata rather than inventing an incompatible general architecture.
New conversions write `dflash.variant = dflash2`. Existing converted DFlash 2
artifacts may instead be recognized by nonzero `dflash.selector_top_k` plus the
complete convolution and selector metadata. Tensor presence alone is never
sufficient. A DFlash 2 artifact passed to the original loader fails with a
specific variant error, and an original DFlash artifact passed to the DFlash 2
loader fails likewise.

### Dtype execution

The published safetensors are BF16. Loading bytes exactly is not enough: the
execution contract must bridge f32 target taps, BF16 DFlash 2 parameters and
activations, and any separately quantized target embedding or LM head.

The safetensors path retains BF16 parameters, converts routed target taps at an
explicit graph boundary, and uses BF16-capable dense linear, normalization,
attention, convolution, and selector kernels with f32 accumulation where the
reference does. It does not silently materialize the whole checkpoint as f32.
Every cast is represented in the compiled graph and covered by reference
fixtures. The GGUF path uses its declared encoded tensor types and the same
logical graph contract.

### Backbone

The five-layer target-conditioned Transformer backbone remains parallel. It
does not autoregressively rerun the backbone for each proposal token.

DFlash 2 adds two `GroupedDynamicCausalConv` modules per layer, one around
attention and one around the feed-forward sublayer. Each module performs a
prepare convolution before its sublayer and a finish convolution after it, for
four block-local two-tap convolutions per layer. For hidden row `t`:

```text
conv(x)_t = k(t, 0) * x_t + k(t, 1) * x_(t - 1)
```

Each module owns `base_kernel[2, kernel_size, hidden_size]`. One projection of
the normalized sublayer input produces two dynamic coefficient sets. The first
set and first base kernel transform the sublayer input; the second set is
retained and combined with the second base kernel to transform the sublayer
output. Groups of 16 channels share each dynamic correction. This ordering and
coefficient reuse are checkpoint semantics, not fusion choices.

The predecessor tap cannot cross a proposal block boundary. The first draft
position reads the pending anchor representation; it cannot read a prior packed
block's suffix. Persistent proposer K/V ends immediately before the pending
anchor.

The graph-level implementation may initially compose existing operations for
correctness. The production path adds one specialized block-local grouped
dynamic convolution kernel so the 20 small convolutions do not become a
launch-bound regression.

### Candidate selector

The final hidden rows pass through the LM head, then the configured output
multiplier and tanh soft cap, before top-k. For each proposal position, top-k
retains 16 token IDs and transformed unary scores. The selector scores each
predecessor/current candidate pair with the checkpoint's rank-256
factorization. Conceptually:

```text
score_t(a, b) = unary_t(b) + dot(A(a) * H(h_t), B(b))
```

All pair scores are computed in parallel. A short device-side walk starts from
the pending anchor and selects one successor per position. Greedy chooses the
maximum score. Positive-temperature proposal sampling uses an independent
proposal RNG domain.

The selector output contract is sparse:

```rust
struct SparseSelectorRows {
    token_ids: Buffer<u32>,       // [lanes, K_max, selector_top_k]
    probabilities: Buffer<f32>,   // same logical shape when normalized
    selected_tokens: Buffer<u32>, // [lanes, K_max]
}
```

This is an internal semantic shape, not a required physical layout. Fused
kernels may retain IDs, scores, and path state in a different arrangement.

The full-vocabulary LM-head result remains internal to top-k selection. It is
not retained as a proposal output, copied to the host, or published through the
core tensor API.

### Proposal probability contract

DFlash 2 may use one of two contracts under this RFC.

`Unavailable` is always valid. Target-sample matching compares the selected
path with independent target draws and preserves the target distribution.

`SparseCausalNormalized` is an extension permitted only when the implementation
can prove that each selector row is the normalized distribution actually used
to draw the next proposal token, conditioned on the selected predecessor and
prefix. The row contains the candidate IDs and probabilities for that selected
predecessor after all proposal sampling controls. An arbitrary softmax of unary
top-16 scores is not this distribution. Candidate sets and hidden values must
not depend on future proposal draws. Tokens outside the top-16 set have
proposal probability zero.

The immutable artifact contract and per-round value contract both gain the
sparse variant:

```rust
enum ProposalProbabilityContract {
    CausalNormalized,
    SparseCausalNormalized { top_k: u32 },
    Deterministic,
    Unavailable,
}

enum ProposalProbabilities {
    CausalNormalized(DistributionRows),
    SparseCausalNormalized(SparseDistributionRows),
    Deterministic,
    Unavailable,
}
```

For sparse rejection sampling, the target residual is:

```text
r(v) = max(0, p(v) - q(v))
```

where `q(v) = 0` outside the sparse selector support. The Metal residual sampler
subtracts sparse entries while reducing the target vocabulary; it does not
scatter a dense proposal row first.

Until normalization, sampling controls, and prefix conditioning are covered by
proof-oriented unit tests, the implementation uses `Unavailable` and
`TargetSampleMatch`. Greedy selection does not require proposal probabilities.

## Small-Row Quantized Linear Kernels

### Problem

The target model repeatedly computes quantized linears with only M=2 through
M=8 activation rows. Existing packed-dot kernels optimize decode-like vectors,
while the MMA path pays dequantization and staging costs that are not uniformly
profitable at these row counts.

Q3_K and Q2_K dominate the measured Muse verification path. The required
kernel is a quantized matrix-multiple-vector operation that reuses each packed
weight block across several input rows before moving to the next block.

### Direct multi-row design

For one output weight row and one 256-column K-quant block:

1. load the packed Q3_K or Q2_K block once;
2. unpack scales, minima, high masks, and quant fields once per SIMD group;
3. load or stream the corresponding activation segment for M rows;
4. accumulate M independent dot products in registers;
5. reduce each dot product within its SIMD group; and
6. write M output values with optional bias.

The kernel specializes M at pipeline compilation for M=2 through M=8. It does
not branch over M inside the inner packed-block loop. Q3_K is implemented first,
then Q2_K, because their decoding layouts and measured shares differ.

The production dispatch candidates are:

| Rows M | Candidate path |
|---:|---|
| 1 | existing packed decode/qmv path |
| 2-8 | direct multi-row weight-reuse path |
| 8 | compare direct path with existing M=8 MMA during development |
| 16 and aligned prefill | f32-faithful direct/MMA path; half MMA only after the target-distribution gate |

Development benchmarks determine the immutable dispatch table for each codec
and supported model shape. Production execution does not time two kernels and
choose between them per round.

### Target numerical contract

Speculative verification is an execution of the same target distribution as
ordinary decode, not a lower-precision approximation. The current half-operand
MMA kernels round decoded weights and activations differently from the M=1 f32
packed-dot path. Numerical closeness to a CPU decoder is insufficient for exact
sampling: a changed target probability makes target-sample matching exact for a
different distribution.

The direct multi-row kernels preserve the M=1 per-row dequantization,
accumulation order, f32 accumulator semantics, bias, and epilogues while sharing
weight loads across independent row accumulators. A verification kernel is
eligible for exact generation only when each causal verifier row produces the
same processed target distribution as sequential ordinary execution over the
same prefix.

Half-operand MMA may remain a prefill or profiling kernel. It cannot be selected
for exact speculative verification unless it passes target-distribution parity,
or ordinary decode intentionally adopts the same declared target arithmetic.
This RFC does not add an approximate speculative mode.

### Kernel correctness

Every new kernel is compared with the authoritative CPU K-quant decoder, the
existing Metal reference path, and sequential M=1 target execution across:

- every specialized M;
- multiple output and input tile boundaries;
- nonzero weights, mixed signs, extrema, and random packed blocks;
- bias and no-bias linears;
- odd output tails;
- buffer offsets rejected by the packed-weight contract; and
- repeated executions using pooled output buffers.

Kernel tests require bitwise per-linear parity where the shared arithmetic
contract permits it. Model tests compare every processed verification
distribution against cloned sequential ordinary state, then require pathwise
target-sample identity with the same keyed target coordinates. Greedy argmax
equality alone does not qualify a kernel for positive-temperature exact mode.

The disabled paired-decoder experiment is not enabled merely because it looks
similar. It must first pass nonzero-weight parity.

### Kernel gates

On the exact Muse prompt and production release build:

```text
M=8 verification < 89 ms   break-even gate
M=8 verification < 75 ms   completion gate
```

Operation-labelled profiling must also show that Q3_K/Q2_K verification time
falls without moving equivalent cost into conversion, copy, or synchronization
kernels.

## Device-Resident Round

### Required flow

One native `runRound` performs:

```text
proposal
  -> sparse selection
  -> target verification
  -> target sampling
  -> acceptance-prefix scan
  -> replay/catch-up
  -> device token-page assembly
  -> stage final KV cursors and status
  -> final completion boundary
  -> atomic host ownership commit
```

The only hot-path host-visible result is the completed token page and its final
length. Diagnostics are copied with that completion; they cannot introduce an
additional fence.

### Device buffers

The round owns fixed-capacity buffers for:

- selected proposal tokens;
- logical proposal lengths;
- target verification input rows and positions;
- target sampled tokens or rejection decisions;
- accepted prefix lengths;
- returned page tokens and page lengths;
- replay valid-row masks;
- target and proposer provisional cursors; and
- staged publication metadata;
- preallocated provisional block tables; and
- one round status/error record.

Logical lengths are device values inside physical fixed-capacity buffers.
Inactive rows have zero advance, cannot enter attention, and cannot publish KV.

### Compiler and runtime support

Metal events can order encoded work, but they cannot make host-constructed
invocation metadata dynamic. The compiler and runtime therefore add:

- direct executable-output to executable-input buffer bindings;
- device-authored valid lengths, positions, advances, and row offsets;
- paged kernels that consume those fixed-address metadata buffers without a
  host validation pass between phases;
- preallocated provisional target/proposer block tables for the maximum round;
- downstream predication on a device round-status buffer;
- a status code for sampler, bounds, and semantic failures discovered on
  device; and
- one final host commit operation for Rust block ownership, prefix-cache
  references, and sequence metadata.

Host-only invariants are validated before submission against maximum physical
geometry. Per-round logical values are validated or bounded on device. A
device error predicates later state writes off and is returned after the final
completion boundary; it cannot publish a partial round.

### Persistent proposer context

Prompt target features are projected into proposer K/V once. Newly accepted
target feature rows are appended once after each committed round. Rejected and
unreturned target rows never enter persistent proposer context.

The repeated proposal program receives only the physical pending-anchor-plus-
mask block and references persistent context K/V, which ends before that
pending token. It does not reproject the full committed context each round.

A policy-selected `K_policy = 0` target round still routes the consumed pending
token's target taps through one-row proposer replay before commit. The cost is
part of the fallback baseline. Deferring this maintenance would violate RFC
0023's synchronized pending-token invariant and is not in the first
implementation.

### Verification assembly

Proposal IDs remain in Metal buffers. A device kernel writes causal target
verification rows, positions, lane offsets, and active masks directly. The Rust
host does not construct a `Vec` of candidates or inspect token IDs.

Each compiled verification executable has a fixed physical M. The selected
logical prefix is represented through valid lengths and zero-advance rows. A
batch may use a common physical bucket with lane-specific logical lengths.

### Acceptance and replay

Greedy matching, target-sample matching, sparse or dense rejection, first
mismatch selection, EOS/budget cutting, and accepted-prefix length all execute
on Metal.

Replay uses a fixed physical executable and a device-written valid-row mask.
The host does not read accepted length to decide how many replay rows to submit.

Target and proposer KV writes go to unpublished tails or provisional blocks.
Metal writes final cursors and status, but cannot mutate Rust block-reference or
prefix-cache ownership. After the one final command-buffer completion, Rust
checks status and atomically commits those staged ownership changes. An
intermediate acceptance readback is not allowed.

### Synchronization target

The current implementation has approximately five synchronization entrypoints
per speculative round. The completed design has one page-completion boundary.
Command buffers may be split for driver or executable constraints, but device
events and dependencies connect them without host inspection.

## Width Selection

### Candidate widths

The checkpoint sets `K_max`; the backend sets the executable width set. For the
initial 16-row Muse artifact, development compiles and profiles:

```text
K: 1, 2, 3, 4, 5, 6, 7, 15
M: 2, 3, 4, 5, 6, 7, 8, 16
```

An implementation may remove a width only after adjacent choices dominate it
on measured cost and expected progress. M=2 through M=8 remain covered while
developing the direct multi-row kernels because the existing curve is
non-monotonic.

The scheduler never chooses `K > K_max`. Running a checkpoint trained at a
smaller block beyond its declared geometry is invalid, even if buffers fit.

### Survival estimate

For each policy bucket, maintain the accepted-length survival curve:

```text
S_K(j) = P(A >= j), 1 <= j <= K
```

One completed width-K round contributes an observation to every admitted
prefix position. The expected returned page length is:

```text
E[P_K] = 1 + sum(j = 1..K, S_K(j))
```

This replaces exponentiation of aggregate `accepted / proposed`. Aggregate
acceptance loses positional information and overstates or understates suffix
survival depending on where failures occur.

Survival observations use an exponentially decayed estimator and retain sample
count. Policy buckets include at least device/profile identity, target and
proposer fingerprints, quantization layout, batch-size tier, and a coarse
context-length bucket. Sampling controls may use separate greedy and stochastic
buckets when their acceptance differs materially.

### Cost estimate

For each K, measure without additional synchronization:

```text
C_draft(K)
C_verify(K)
C_accept_publish(K)
C_replay(K, A)
```

The primary observation is total submitted-round to committed-page wall time,
including queue delay, device work, final status/page transfer, and the atomic
Rust ownership commit. Phase values explain and extrapolate that total; they do
not replace it.

Phase timing uses Metal counter samples where supported, or GPU start/end times
from phase command buffers that are all submitted and device-ordered before the
single final wait. Timing data is read only after page completion. The adaptive
path does not enable `EFFECT_TORCH_METAL_PROFILE` or insert a host fence between
phases.

The expected cost per returned token is:

```text
T_K = E[C_draft + C_verify + C_accept_publish + C_replay] / E[P_K]
```

`C_draft(K)` must reflect physical execution. If the 16-row proposal graph runs
for every logical K, the cost model cannot pretend a shorter logical prefix
made drafting cheaper.

Ordinary decode is represented by:

```text
K_policy = 0
T_0 = target decode + one-row proposer maintenance + final commit/page cost
E[P_0] = 1
```

`T_0` is the cost of a recoverable ordinary round inside a target-coupled
artifact. It is distinct from the standalone ordinary artifact benchmark,
which owns no proposer and pays no proposer maintenance. Diagnostics and
reports show both values rather than labelling paired-state fallback as the
standalone baseline.

The controller chooses the K with the lowest eligible `T_K`, subject to warmup,
minimum sample count, and hysteresis.

### Calibration and exploration

Reachable K values are known at inference compilation so all required
executables and stable buffers exist before generation.

An adaptive session uses a cached profile keyed by the complete policy bucket
when available. Missing observations are collected through bounded probes:

- ordinary probes establish `T_0`;
- the configured initial width establishes a survival curve;
- narrower speculative widths reuse observed prefix survival but still require
  their own measured cost; and
- wider widths require explicit probes because shorter rounds reveal no suffix
  survival.

Probe decisions occur before a round and use only committed history and prior
measurements. They therefore satisfy RFC 0023's non-anticipating rule.

The benchmark harness can produce a persistent calibration report, but a
runtime cache is advisory. Missing, stale, or rejected cache data causes online
calibration, not use of unvalidated timings.

## Ordinary-Round Fallback

### Prior art

Major serving engines permit speculation to reduce to ordinary decoding:

- vLLM's released dynamic speculative decoding uses a configured batch-size to
  K schedule and allows K=0.
- SGLang's adaptive speculative parameters maintain acceptance estimates by
  batch tier, and larger-batch candidate sets include zero speculative steps.
- TensorRT-LLM current source contains a rolling-acceptance speculation gate
  that can disable speculation for the engine.
- Hugging Face assisted decoding shortens conventional assistant drafts using
  confidence and heuristic length control, though it is not the primary model
  for this DFlash policy.

These systems establish that ordinary-round selection is a legitimate serving
policy. They do not establish one portable controller: vLLM is primarily
load-configured, SGLang's current adaptive path is acceptance-oriented and
EAGLE-specific, and TensorRT-LLM's gate may disable permanently.

effect-torch uses measured latency per expected output token and does not adopt
permanent disablement.

### Public policy

RFC 0023's schedule field becomes:

```ts
export type SpeculationSchedule =
  | "fixed"
  | {
      readonly mode: "adaptive"
      readonly fallback?: "never" | "cost"
      readonly widths?: ReadonlyArray<number>
    }
```

`"fixed"` preserves the current behavior and selects the configured
`maxDraftTokens` for every eligible round. Budget or target-state capacity may
still admit `K_round = 0`; this is not a scheduling choice.
`{ mode: "adaptive" }` defaults to
`fallback: "cost"`. `widths` is an optional subset bounded by
`1..maxDraftTokens`; invalid, duplicate, unsorted, or unsupported widths fail
during inference construction.

`fallback: "never"` adapts only among positive policy K values. It is useful
for controlled speculative benchmarks and deployments that want speculation
on every eligible round.

### Entry and exit

The controller evaluates policy only at completed round boundaries.

It enters ordinary fallback when all of the following hold:

1. ordinary and at least one speculative choice meet their minimum sample
   counts;
2. the best speculative estimate exceeds ordinary cost by the entry margin;
3. the condition remains true for the configured consecutive evaluation
   windows; and
4. no explicit fixed-mode requirement applies.

The initial implementation uses backend constants rather than public tuning
knobs:

```text
evaluation window:       8 completed rounds
entry margin:            speculative T_K > 1.05 * T_0
exit margin:             speculative T_K < 0.95 * T_0
entry confirmation:      3 evaluation windows
ordinary cooldown:       32 rounds
speculative probe:       at least once per cooldown while generation continues
```

Constants may change only with benchmark evidence and deterministic controller
tests. Diagnostics report their effective values.

Fallback is not permanent. After cooldown, the scheduler probes the previously
best positive K. It exits fallback after the speculative estimate beats the
ordinary estimate by the exit margin for two evaluation windows. This allows
recovery when context, workload, sampling controls, thermal state, or system
load changes.

Short sequences may finish before calibration completes. They use the
configured initial positive K unless a valid cached profile already selects
another width or `K_policy = 0`. Cost fallback is therefore steady-state or
cached-profile protection, not a guarantee that an unseen cold short request
will avoid every slower speculative round. Calibration and probe costs are
reported separately.

### Load awareness

Batch-size tier and active-lane count are policy inputs because verification
cost changes with packed rows. System load may select a cached profile or K,
but it cannot change exactness or make an unsupported program valid.

Batch one is the first required implementation. Batch tiers are added only
after the batch-one controller and fixed-width kernels pass their gates.

### What fallback does not mean

The following are forbidden:

- catching a proposal, verification, sampler, replay, or publication error and
  retrying the same round with `K_policy = 0`;
- selecting ordinary decode because a required Metal operation is absent;
- running an unsupported DFlash 2 operation on CPU;
- loading a DFlash 2 checkpoint as original DFlash;
- advancing target state before a fallback decision is made; or
- hiding fallback rounds from diagnostics or benchmark output.

This preserves RFC 0023's required runtime contract and RFC 0022's no CPU
fallback rule. The scheduler chooses between two already compiled valid paths;
it does not probe optional backend methods.

## Exactness

### Greedy

For each proposal position, accept only while the selected proposal token
equals the target argmax for that causally verified row. At first mismatch,
emit the target argmax. On full acceptance, emit the target bonus argmax.

`K_policy = 0`, every positive K, every adaptive transition, and mandatory
`K_round = 0` admission must produce the same greedy token sequence as ordinary
decode. A target-kernel numerical discrepancy fails this requirement; it is not
an allowed exception.

### Positive temperature

Original DFlash and DFlash 2 with unavailable proposal probabilities use target
sample matching:

1. draw target tokens from every causally verified target row using the target
   RNG domain;
2. accept while target draw equals selected proposal token;
3. emit the first target mismatch; and
4. emit the target bonus draw after full acceptance.

DFlash 2 may use sparse rejection only after its sparse selector rows satisfy
the causal normalized contract above. Proposal draws use the `proposal` RNG
domain; target, acceptance, and residual draws use the RFC 0023 keyed domains.

An adaptive width or `K_policy = 0` decision depends only on prior completed
rounds, committed context, and current load. It cannot inspect candidates from
the round it is deciding whether to execute.

Greedy output equality is tested exactly. Positive-temperature correctness is
tested through deterministic replay, fixed small-distribution cases, and
statistical distribution tests. Target-sample matching uses the same keyed
target coordinates as ordinary sampling and must match it pathwise after the
target-distribution parity gate. Sparse rejection uses independent acceptance
and residual domains and is distribution-identical rather than pathwise
identical to ordinary sampling.

Load-aware adaptive sparse rejection may choose a different exact proposal
distribution `q` between runs. Deterministic replay of adaptive mode therefore
records and replays the selected-width trace and effective policy-bucket
observations, or uses a fixed scripted controller. A seed alone does not freeze
external load or thermal inputs.

## Diagnostics

Inference diagnostics add:

```ts
interface AdaptiveSpeculationDiagnostics {
  readonly selectedWidthRounds: ReadonlyArray<bigint>
  readonly ordinaryFallbackRounds: bigint
  readonly mandatoryOrdinaryRounds: bigint
  readonly fallbackEntries: bigint
  readonly fallbackExits: bigint
  readonly speculativeProbeRounds: bigint
  readonly proposerMaintenanceNanos: bigint
  readonly activePolicyBucket: string
  readonly estimatedPairedOrdinaryNanosPerToken?: number
  readonly standaloneOrdinaryNanosPerToken?: number
  readonly estimatedSpeculativeNanosPerToken: ReadonlyArray<number | undefined>
  readonly survivalByWidth: ReadonlyArray<ReadonlyArray<number>>
}
```

Exact field packing may differ at the native boundary, but the information is
normative. Existing aggregate `acceptedTokens / proposedTokens` remains a
summary metric and is never used as the width estimator.

Phase diagnostics distinguish:

```text
proposal backbone
dynamic convolution
vocabulary top-k
selector lattice/walk
target verification
target sampling/acceptance
replay
publication/page assembly
```

Profiling labels are available only in diagnostic builds and cannot alter the
production dispatch or final throughput measurements.

## Benchmark Contract

### Prompt pairing

`packages/bench/muse-glimmer.ts` first creates one immutable case manifest. A
case contains its ID, prompt text or token IDs, exact context length, sampling
configuration, and output limit.

Ordinary, fixed-width DFlash, adaptive DFlash, DFlash 2, and external-engine
runs consume the same case IDs. Mode iteration cannot increment the identity
used to construct the prompt. Greedy comparisons store and compare output
hashes by case ID.

Final matched measurements run each mode in a separate process over the same
manifest. Repetitions alternate mode order to reduce thermal and order bias.

### Workloads

The suite contains:

- the exact 256-token TypeScript union-type prompt;
- code prompts;
- arithmetic and reasoning prompts;
- ordinary chat prompts;
- deterministic exact-length synthetic prompts for kernel-cost isolation;
- greedy generation; and
- the model's representative stochastic sampling configuration.

At least five measured repetitions follow warmup. Final reports use medians and
retain per-run JSONL data.

### Metrics

Every speculative result reports:

- output tokens per second;
- time to first token and decode latency;
- proposal, verification, acceptance, replay, and publication time;
- accepted-length histogram and survival curve;
- committed page length;
- selected K and M distributions;
- ordinary fallback, entry, exit, and probe counts;
- pool high-water marks;
- output hash for greedy runs; and
- deterministic replay identity for stochastic runs.

`MODE=both` output is not accepted as paired evidence unless it consumes a
prebuilt shared manifest. `EFFECT_TORCH_METAL_PROFILE` must be unset for
throughput claims.

External llama.cpp results distinguish low-level decode timing from CLI
end-to-end timing. Different prompt text, tokenizer policy, output length, or
timing boundary cannot appear in the same speedup ratio.

### Performance gates

Completion requires:

```text
M=8 verification:                 below 75 ms
exact TypeScript prompt:          at least 30.4 output tokens/s
matched-suite median speedup:     at least 1.3x over ordinary decode
matched-suite individual cases:  none slower than ordinary decode
```

The earlier 89 ms M=8 target is an intermediate break-even gate, not completion.
Adaptive fallback may prevent a workload from remaining slower, but it does not
satisfy the fixed-width kernel or exact-prompt gates by itself.

## Implementation Plan

### Phase 0: retain and land current optimizations

1. Land batched verification-row sampling, state-only replay, bulk KV block
   references, M=8/M=16 output-partition MMA, Q2_K through Q6_K half decoders,
   verifier width compilation, and benchmark diagnostics.
2. Keep half-operand MMA available for prefill and measurement, but gate its
   exact verification selection on target-distribution parity.
3. Preserve unrelated workspace changes and stage only intended source files.
4. Update RFC 0023's Phase 3 status after the existing ParallelBlock path is
   committed and verified.

### Phase 1: benchmark correctness

1. Build a shared immutable prompt manifest.
2. Remove mode-dependent case ID advancement.
3. Add paired greedy output-hash assertions.
4. Add stochastic deterministic replay and distribution tests.
5. Record the ordinary M=1 baseline and all M=2 through M=8 and M=16 target
   verification costs in release mode.

### Phase 2: direct Q3_K and Q2_K row reuse

1. Add Q3_K direct M=2 through M=8 kernels and parity tests.
2. Measure every width and retain an existing verification path only where it
   wins and passes target-distribution parity.
3. Add Q2_K direct M=2 through M=8 kernels and parity tests.
4. Re-profile the exact Muse workload.
5. Continue only after the 89 ms intermediate gate is met; finish only after
   the 75 ms gate is met.

### Phase 3: DFlash 2 loader and reference graph

1. Add exact config/safetensors inspection and loading.
2. Validate DFlash 2 separately from original DFlash.
3. Add BF16 storage, mixed-dtype graph boundaries, and required Metal kernels.
4. Add convolution and selector parameter catalogs.
5. Implement exact prepare/finish grouped dynamic convolution ordering.
6. Apply output multiplier and soft cap before sparse top-16 extraction.
7. Implement rank-256 selector scoring and path selection.
8. Match reference greedy selector paths on saved fixtures.
9. Define ecosystem-compatible GGUF variant metadata and add loading.

### Phase 4: DFlash 2 Metal kernels

1. Fuse block-local grouped dynamic convolution.
2. Add device vocabulary top-k specialized for 16 retained candidates.
3. Add selector lattice scoring and device-side path walk.
4. Extend `Speculation.ts`, `Runtime.ts`, both backend adapters, native-addon
   schemas, and Rust plans with `SparseCausalNormalized`.
5. Keep selected tokens and selected-predecessor sparse probabilities
   device-resident.
6. Implement target-sample matching first.
7. Add sparse rejection only after normalization and exactness tests pass.

### Phase 5: cohesive device round

1. Remove proposal token readback.
2. Add direct output-to-input bindings and device-authored invocation metadata.
3. Preallocate provisional block tables and add a device status record.
4. Assemble target verification metadata on Metal.
5. Compute accepted prefix and output page on Metal.
6. Drive replay valid rows from device buffers.
7. Stage final device cursors, then atomically commit Rust ownership once.
8. Reduce the round to one host completion boundary.

### Phase 6: adaptive width and fallback

1. Replace `select_verify_rows` with survival and cost state.
2. Compile the reachable K set and validate user width subsets.
3. Add recoverable ordinary probes with one-row proposer maintenance.
4. Add per-width speculative probes and post-completion timing collection.
5. Implement hysteretic `K_policy = 0` entry, cooldown, probing, and exit.
6. Distinguish policy fallback from mandatory `K_round = 0` admission.
7. Expose complete diagnostics.
8. Test policy deterministically with synthetic timing and acceptance streams.
9. Enable adaptive scheduling for batch one.

### Phase 7: completion validation

1. Run all core and Metal tests.
2. Run dprint, TypeScript checks, Rust formatting, clippy where configured, and
   `git diff --check`.
3. Rebuild the release native addon.
4. Run the matched greedy and stochastic benchmark suite.
5. Retain raw JSONL results and exact commands.
6. Confirm all performance gates and no slower matched case.

## Testing

### Loader and graph

- Original DFlash rejects DFlash 2 metadata.
- DFlash 2 rejects original DFlash metadata.
- Missing convolution or selector tensors fail before allocation.
- Configured target layers and tensor shapes match Muse-Glimmer.
- Convolution predecessor taps cannot cross packed block boundaries.
- The first draft position reads the pending anchor and persistent K/V ends
  immediately before it.
- Top-16 IDs and unary scores match a reference implementation.
- Greedy selector paths match reference fixtures.

### Kernels

- Q3_K and Q2_K M=2 through M=8 match CPU decoding.
- Dynamic convolution matches the composed graph.
- Top-k handles ties deterministically under the documented ordering.
- Selector scoring and walk match f32 reference results within declared
  tolerance.
- BF16 storage and every mixed-dtype boundary match reference execution.
- Tail output rows and vocabulary tails do not write out of bounds.

### State and synchronization

- Every acceptance cut publishes exactly the returned page boundary.
- Full acceptance publishes the target bonus as pending and catches up proposer
  state correctly.
- Rejection and EOS cuts discard every later provisional row.
- Failure before final publication leaves both states unchanged.
- No candidate or acceptance buffer is read by the host before page completion.
- Metal capture verifies no additional synchronization entrypoint in the round.
- A `K_policy = 0` round replays one consumed target row and leaves target and
  proposer pending tokens aligned for a positive-K next round.

### Exactness

- Greedy fixed K values match ordinary output token for token.
- Greedy adaptive K transitions and `K_policy = 0` transitions match ordinary
  output.
- Every verifier row has the same processed target distribution as cloned
  sequential M=1 execution.
- Positive-temperature target-sample matching matches ordinary sampling
  pathwise under the same keyed target coordinates.
- Sparse rejection covers equal, disjoint, truncated, and target-only support.
- Statistical tests recover the target distribution for deliberately poor and
  strong proposal rows.

### Controller

- Aggregate acceptance and positional survival produce deliberately different
  choices in a fixture, and the controller follows survival.
- Non-monotonic verification costs choose the measured cheapest K.
- The controller enters fallback only after the required windows.
- Fallback probes and exits after workload improvement.
- Fallback never becomes permanent while generation continues.
- Fixed mode never selects `K_policy = 0`; mandatory admission may select
  `K_round = 0`.
- `fallback: "never"` selects only positive policy K.
- Failed rounds do not update timing, survival, or fallback state.
- Policy decisions do not depend on current-round candidate values.

## Acceptance Criteria

1. The published Muse-Glimmer DFlash 2 checkpoint loads through an explicit
   DFlash 2 contract.
2. Original DFlash and DFlash 2 cannot be silently interchanged.
3. Q3_K and Q2_K have correct direct multi-row paths for M=2 through M=8.
4. M=8 verification is below 75 ms on the development Apple Silicon host.
5. DFlash 2 convolution and selector outputs match reference fixtures.
6. Full-vocabulary proposal rows never cross N-API or become public tensors.
7. A speculative round has one host completion boundary and one final atomic
   Rust ownership commit.
8. Target and proposer state remain transactional across every acceptance cut
   and failure phase.
9. Greedy output is identical across ordinary, fixed, adaptive, and fallback
   execution.
10. Verification matches ordinary target distributions, and
    positive-temperature execution passes pathwise target-sample, deterministic
    replay, and distribution tests.
11. Width selection uses accepted-length survival and measured phase cost.
12. Adaptive cost fallback selects `K_policy = 0`, maintains proposer alignment,
    reports it, probes recovery, and can return to positive K.
13. Fixed mode and adaptive `fallback: "never"` never select `K_policy = 0`;
    mandatory zero-width admission is reported separately.
14. No speculative error, unsupported operation, or invalid artifact triggers
    ordinary fallback.
15. The exact 256-token prompt reaches at least 30.4 output tokens/s.
16. The matched-suite median is at least 1.3x ordinary decode with no slower
    individual case.
17. Final benchmark prompts, token counts, sampling controls, and output limits
    are paired exactly.
18. Final throughput results are collected without Metal profiling.

## Risks

### More accepted tokens can still be slower

DFlash 2 should improve accepted length, but target verification remains the
largest cost. The kernel phases precede claims based on selector quality.

### Physical 16-row drafting limits width savings

Logical K reduces verification but not the full proposal graph. The cost model
records this honestly. A future shorter physical proposer may be added only if
the checkpoint and convolution boundary semantics are validated at that shape.

### Sparse proposal exactness can be misstated

A top-k selector is not automatically a causal normalized proposal. The default
contract remains `Unavailable`; sparse rejection is gated by proof and tests.

### Controller oscillation

Thermal changes and noisy short runs can alternate between K choices. Minimum
sample counts, evaluation windows, hysteresis, and cooldown limit oscillation.

### Fallback can hide inadequate fixed-width performance

Adaptive results are reported separately from fixed-width results. The 75 ms
verification and 30.4 tokens/s exact-prompt gates cannot be passed by ordinary
fallback.

Paired-state fallback also includes proposer maintenance and is not identical
to a standalone ordinary artifact. It may remain slightly slower than the
standalone baseline until replay maintenance is optimized. Cold short requests
may finish during calibration. Reports expose both limitations instead of
crediting fallback with ordinary standalone performance.

### Exact target arithmetic can constrain kernel speed

Half-operand verification may be faster but change the target distribution.
Exact generation prefers f32-faithful row reuse even if this removes an
otherwise attractive benchmark result. An approximate mode would require a
separate RFC and cannot satisfy these gates.

### Too many compiled widths increase startup and memory

The development set intentionally covers the small-row cliff. Widths may be
pruned only from retained measurements showing domination, not from aggregate
acceptance assumptions.

### Device-resident control increases kernel coupling

Acceptance, replay, and publication share fixed buffers and cursor semantics.
Each phase retains an independent reference path and transaction tests so a
fused optimization does not become the only correctness oracle.

## Relationship to RFC 0023

RFC 0023 remains the normative public generation, exact sampling, RNG, pending
token, and transactional publication contract. This RFC provides the concrete
Metal implementation for its `ParallelBlock` and adaptive scheduling phases.

The `K_policy = 0` policy does not contradict RFC 0023's prohibition on
optional runtime-method fallback. Both ordinary and speculative rounds are
required, compiled paths of one inference artifact. The scheduler selects a
path before execution, and one-row replay keeps the proposer synchronized.
Capability probing and recovery retries remain forbidden.

This RFC also narrows RFC 0023's generic load-aware disablement: DFlash uses a
recoverable, measured cost gate rather than permanent disablement or an
acceptance-only threshold.

## References

- RFC 0023, *Batched Speculative Generation*:
  `docs/rfcs/0023-batched-speculative-generation.md`
- Chen et al., *DFlash: Block Diffusion for Flash Speculative Decoding*:
  <https://arxiv.org/abs/2602.06036>
- Inco AI, *DFlash 2: Keep Drafting Parallel*:
  <https://inco.ai/blog/dflash2/>
- DFlash reference implementation:
  <https://github.com/z-lab/dflash>
- Muse-Glimmer DFlash 2 checkpoint:
  <https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2>
- llama.cpp DFlash 2 pull request and Metal path:
  <https://github.com/ggml-org/llama.cpp/pull/27342>
- vLLM speculative decoding:
  <https://docs.vllm.ai/en/latest/features/speculative_decoding/>
- vLLM dynamic speculative decoding:
  <https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/>
- vLLM released batch-size to K scheduler source, inspected at `94a54f5`:
  <https://github.com/vllm-project/vllm/blob/94a54f581e1685a5706fe2809bd4ecb1a4c8e70b/vllm/v1/core/sched/scheduler.py#L255-L273>
- SGLang adaptive speculative decoding:
  <https://docs.sglang.ai/advanced_features/adaptive_speculative_decoding.html>
- SGLang adaptive K tiers, inspected at `b644771`:
  <https://github.com/sgl-project/sglang/blob/b644771e07e39c8a5997a230f4e87175364c020b/python/sglang/srt/speculative/adaptive_spec_params.py#L22-L47>
- TensorRT-LLM speculative decoding:
  <https://nvidia.github.io/TensorRT-LLM/latest/features/speculative-decoding.html>
- TensorRT-LLM rolling speculation gate, inspected at `7110513`:
  <https://github.com/NVIDIA/TensorRT-LLM/blob/7110513dd8a43d212a76767ca7a93a2a3d45ec5a/tensorrt_llm/_torch/speculative/speculation_gate.py#L7-L82>
- TensorRT-LLM DFlash context/current-state split:
  <https://github.com/NVIDIA/TensorRT-LLM/pull/13996>
- Hugging Face assisted decoding:
  <https://huggingface.co/docs/transformers/main/en/generation_strategies#speculative-decoding>
- MLX LM quantized generation and kernels:
  <https://github.com/ml-explore/mlx-lm>
- llama.cpp quantized matrix-vector kernels:
  <https://github.com/ggml-org/llama.cpp>

# RFC 0023: Batched Speculative Generation

- **Status**: Draft
- **Created**: 2026-08-14
- **Depends on**: RFC 0010 (inference), RFC 0013 (batched decode), RFC 0017
  (multi-backend runtime), RFC 0019 (executable compilation), RFC 0020
  (invocation ownership)
- **Supersedes**: RFC 0014 (speculative decoding)
- **Updates**: RFC 0010 and RFC 0013 generation API

## Summary

Generation has one public shape: a sampled, batch-first interface that returns
a nonempty token page for every active lane. Batch size one is the ordinary
single-sequence case. Normal decoding returns one token per page; speculative
decoding may return several accepted tokens followed by a target correction or
bonus token. Pages in one result may have different lengths.

Speculation is configured when calling `Model.inference`; it is not a separate
public session type. A proposer-neutral native round can execute conventional
draft models, EAGLE/MTP-style auxiliary drafters, DFlash, DSpark,
self-speculation, retrieval proposers, and linear or tree candidate layouts.
The stable consumer abstraction remains the same array of token pages.

Positive-temperature generation is a primary execution mode. A probabilistic
autoregressive proposal uses exact rejection and residual sampling. A proposal
without a usable normalized causal distribution uses direct target-sample
matching, or another acceptance rule only after that rule has a documented
proof of target-distribution preservation. Heuristic acceptance is never
labelled exact.

The target and proposer states participate in one internal native transaction.
Only state corresponding to the returned page is published. Rejected or
unreturned provisional rows are discarded. This RFC adds no public sequence
truncate operation.

## Motivation

The current API exposes four peer operations:

- logits-returning `Generation.add` and `Generation.step`;
- sampled `Generation.addSampled` and `Generation.stepSampled`;
- a dedicated batch-one decode executable;
- an additional fixed-width batched decode executable.

This makes batch size and sampler placement part of the consumer model even
though native sampled decode is already the normal efficient path. It also
assumes one invocation makes exactly one token of progress. Speculative
decoding violates that assumption by design.

The replacement has two ordered phases:

1. make all ordinary generation batched and sampled, with token pages as the
   result;
2. add exact speculative rounds without changing the consumer interface.

The ordering matters. Speculation must extend one coherent generation path,
not preserve the current single/batched and logits/sampled matrix.

## Decisions

1. `Generation.add` and `Generation.step` accept arrays and return arrays.
2. Every successful active lane returns a nonempty token page.
3. `Generation.addSampled` and `Generation.stepSampled` are removed.
4. Logits are not a peer generation result. Custom host sampling uses a lower
   level stateful execution interface.
5. One fixed-width batch program is compiled for each execution geometry.
   Batch one uses the same program with width one.
6. Sampling defaults are fixed at inference creation and may be overridden per
   lane. All tokens in one lane use the same effective controls for that round.
7. Sequence identity, not a transient array index, keys sampling counters.
8. Speculation is configured on `Model.inference` and lowered with the target
   model into one backend-owned inference artifact.
9. Core speculative execution methods are required runtime methods. There is
   no capability probing or optional-method fallback.
10. The first implementation supports target and proposer programs owned by
    the same runtime identity. Cross-runtime orchestration is deferred.
11. The first implementation publishes rollback only for KV-only target and
    proposer state. Models with arbitrary KDA, convolution, or other recurrent
    state reject speculative configuration at inference construction.
12. Per-token JavaScript orchestration is forbidden. TypeScript may schedule
    one native call per round; draft, verify, accept, and publication remain in
    that call.
13. `Tensor.clear` and `Tensor.clearAll` remain public, idempotent, and
    non-failing. This design adds no `Scope`, `Effect.uninterruptible`, or
    `Effect.uninterruptibleMask` requirement for tensor lifetimes.

## Public API

The names in this sketch are normative unless implementation uncovers a
TypeScript conflict; the array and token-page semantics are normative.

```ts
export interface SamplingOptions {
  readonly temperature: number
  readonly topK?: number
  readonly topP?: number
  readonly seed: number
}

export interface GenerationAdd {
  readonly prompt: Tensor.Any // [1, T], T >= 1
  readonly sampling?: Partial<SamplingOptions>
  /** Total output-token budget owned by the sequence. Omit for no budget. */
  readonly maxTokens?: number
  readonly eosTokens?: ReadonlyArray<number>
}

export interface GenerationStep {
  readonly seq: GenerationSeq
  readonly sampling?: Partial<SamplingOptions>
}

export interface TokenPage {
  readonly seq: GenerationSeq
  readonly tokens: ReadonlyArray<number> // always nonempty
  readonly stopReason?: "eos" | "maxTokens"
}

export interface Generation {
  readonly add: (
    entries: ReadonlyArray<GenerationAdd>
  ) => Effect.Effect<ReadonlyArray<TokenPage>, InferenceError | TensorError, Runtime>

  readonly step: (
    entries: ReadonlyArray<GenerationStep>
  ) => Effect.Effect<ReadonlyArray<TokenPage>, InferenceError | TensorError, Runtime>

  readonly live: () => Effect.Effect<number>
  readonly close: () => Effect.Effect<void, TensorError, Runtime>
}
```

`add([])` and `step([])` fail. Entries are returned in input order. A step may
contain fewer than the compiled width; unoccupied lanes are inactive and must
not execute, consume RNG draws, allocate provisional state, or advance a
cursor.

The session owns each sequence's pending emitted token. Callers no longer feed
that token back to `step`. This removes an easy mismatch between the visible
page and the next input and is necessary when a page contains several tokens.
The pending-token alignment is specified below.

`maxTokens` and `eosTokens` are sequence policy fixed by `add`. `maxTokens` is a
positive total output-token budget, not the inference pool's `maxTokens` field.
The sequence stores and decrements its remaining budget. A page that emits EOS
or exhausts the budget carries `stopReason` and makes the sequence terminal;
later `step` calls fail until the caller finishes it. Sampling overrides are
resolved before each native round and cannot change inside one page.

### Inference configuration

Proposer artifacts are attached at inference construction because compilation,
weight retention, hidden taps, vocabulary mapping, and state geometry must be
validated before a sequence can mutate.

```ts
export interface InferenceConfig {
  readonly batchSize?: number
  readonly prefillChunk?: number
  readonly sampling: SamplingOptions
  readonly speculation?: {
    readonly proposer: ProposerArtifact
    readonly maxDraftTokens: number
    readonly schedule?: "fixed" | "adaptive"
  }
  // Existing pool, block, retention, token dtype, and KV dtype fields remain.
}
```

`ProposerArtifact` is one opaque effect-torch artifact regardless of research
method. A conventional small LLM and target-coupled DFlash checkpoint differ in
their internal plan, not in `Model.inference`. First-party or model packages
construct artifacts through `Speculation.artifact`; convenience loaders such as
`DFlash.load` may validate named checkpoint formats but return the same type.

```ts
export interface ProposerArtifact {
  readonly [ProposerArtifactTypeId]: typeof ProposerArtifactTypeId
}

export interface ProposerArtifactInput {
  readonly components: ReadonlyArray<{
    readonly model: Model
    readonly params: Params
  }>
  readonly plan: ProposerPlan
}

export namespace Speculation {
  export const artifact: (
    input: ProposerArtifactInput
  ) => Effect.Effect<ProposerArtifact, InferenceError | ModelError>
}
```

`ProposerPlan` is a validated structural recipe described below. It declares
execution stages, target dependencies, candidate topology, probability
contract, state schema, trained limits, and token mapping. It does not contain
an `"eagle" | "dflash" | ...` switch. Retrieval and self-speculation artifacts
may have no separately retained parameters. Unsupported plans fail during
`Model.inference`, before opening a generation session or mutating a sequence.

### Raw logits escape hatch

Removing logits from `Generation` does not forbid custom samplers. The
inference artifact exposes a lower-level `openExecution()` interface with
explicit token inputs and caller-owned logits:

```ts
export interface StatefulExecution {
  readonly add: (
    prompts: ReadonlyArray<Tensor.Any>
  ) => Effect.Effect<
    ReadonlyArray<{ readonly seq: ExecutionSeq; readonly logits: Tensor.Concrete }>,
    InferenceError | TensorError,
    Runtime
  >

  readonly step: (
    entries: ReadonlyArray<{ readonly seq: ExecutionSeq; readonly token: number }>
  ) => Effect.Effect<ReadonlyArray<Tensor.Concrete>, InferenceError | TensorError, Runtime>

  readonly close: () => Effect.Effect<void, TensorError, Runtime>
}
```

It uses the same fixed batch geometry and transactional state advancement, but
performs no sampling, speculation, page construction, EOS handling, or pending
token ownership. `step` commits exactly the caller-supplied token and returns
one `[vocab]` row per entry. Each returned tensor is caller-owned and independent
of sequence lifetime. Execution sequences cannot be passed to `Generation`, or
vice versa. This is an execution API rather than a second generation policy.

## Batch Execution

### One compiled lane set

`batchSize` replaces `decodeBatch`. Inference compiles:

- decode input `[B, 1]` with active-lane and advance vectors;
- prefill input `[B, C]` with per-lane valid lengths and advances;
- verification layouts bucketed by total valid rows, up to the configured
  speculation bound.

There is no separately compiled `[1, 1]` fast path when `B > 1`. Backends may
specialize internally without changing the executable or public contract.
`lastTokenRow: true` remains suitable for ordinary decode. Verification uses
all-row output and does not overload the last-row selector.

Prefill chunks from several prompts share the fixed lane set. Padding rows have
zero advance and cannot enter attention, position accounting, prefix hashes, or
recurrent state. Prompt lengths may differ. A lane that finishes prefill can
produce its first page while another lane still needs chunks; the scheduler may
use separate prefill and decode submissions, but all submissions use the same
stable lane identities and batched public call.

### Lane ownership

Each generation session owns `B` physical lane slots. `add` assigns new
sequences to the lowest free slots in input order. A `GenerationSeq` retains
its slot until `finish`, after which the slot may be refilled. Array order in a
later `step` does not move sequence state between slots. The backend receives a
slot mask and maps outputs back to input order.

Before allocating or prefilling anything, `add` verifies that every entry fits
in the currently free slots. If not, the whole call fails. It never admits an
input prefix or returns fewer results than entries.

Stable sequence IDs are monotonically allocated within the artifact and are
never reused as RNG identities, even when physical lanes are reused. This makes
refill and array reordering independent of seeded outcomes.

## Token Alignment

The final emitted token is pending: it is visible to the caller but has not yet
been applied as an input row. This is the existing generation convention made
explicit and moved inside `GenerationSeq`.

`add` establishes this invariant for target and proposer together. It prefills
each stateful proposer from the same logical prompt using its token map and
commit plan, samples the first visible token from the target only, and installs
that token as pending in both states. A target-coupled plan may build proposer KV
from target prompt-feature taps instead of replaying token rows. The first
`runRound` cannot begin unless target and proposer committed cursors describe
the prompt and share the same pending target token.

Suppose the committed prefix ends in `A`, ordinary generation has emitted
pending token `B`, and a proposer drafts `C D E`:

```text
target input row       B       C       D       E
target distribution    p(C)    p(D)    p(E)    p(F)
candidate tested       C       D       E       -
```

The target verification pass needs `K + 1` output distributions for `K`
candidates. The first comes from executing the already-pending token. Each
candidate row produces the distribution for the next candidate, and the final
row produces the correction/bonus distribution after complete acceptance.

If `C` is accepted and `D` is rejected, the returned page is `[C, D*]`, where
`D*` is sampled by the exact rejection rule. Published target state contains
input rows `B C`; `D*` is the new pending token. If all candidates are accepted,
the page is `[C, D, E, F]`, published state contains `B C D E`, and `F` is
pending. Thus target cursor advance equals page length in both cases.

Every stateful proposer is synchronized to the same boundary: its committed
state represents all page tokens except the final token, and that final token is
its pending input. An autoregressive proposer that generated `C D E` has not
consumed `E`. On full acceptance it must run a state-only catch-up with `E`,
discard the unused proposal logits, and install target bonus `F` as pending. On
rejection at `D`, it retains state through `C`, discards provisional `D` state,
and installs `D*` as pending. A proposer plan must define this `commitThrough`
operation; publishing its raw post-proposal cursor is incorrect.

This alignment is validated as descriptor metadata. A backend must not infer it
from tensor length or silently discard the first or final distribution.

## Speculative Round

One native call performs the following transaction for all active lanes:

1. snapshot committed target and stateful proposer roots;
2. route committed target feature taps to the proposer if required;
3. produce candidates and provisional proposer state;
4. choose each lane's logical proposal length without violating the
   non-anticipating rule;
5. build and execute packed target verification with all required rows;
6. apply sampling transforms identically to ordinary generation;
7. run exact acceptance and correction/bonus sampling;
8. cut each page at its first EOS or output budget boundary;
9. synchronize every stateful proposer through all but the page's final token;
10. publish only target and proposer state represented by each returned page;
11. return rejected and unreturned provisional blocks to their pools;
12. publish nothing for any lane if any phase of the batched call fails or is
    interrupted before commit.

Publication is atomic across the active batch. Per-lane publication followed by
a later batch failure is forbidden. Interruption remains requested cancellation,
as in RFC 0017: already submitted device work may complete, but late provisional
outputs are discarded rather than published.

For KV-only state, provisional rows use newly allocated blocks or unpublished
tails. Full accepted blocks can be retained directly. A partially accepted
block records an accepted cursor; rows after that cursor are unreachable by
attention and are overwritten before reuse. Rejected complete blocks are
returned without entering the prefix cache under a committed sequence hash.
Prefix-cache ownership changes are staged with the round commit.

No public `GenerationSeq.truncate(position)` is added. Internal cut metadata is
valid only for the in-flight round and cannot move an already committed cursor.

## Proposer, Candidate, And Verification Descriptors

There are two descriptor levels. `ProposerPlan` is immutable compile-time
structure retained by `ProposerArtifact`. `CandidateBatch` and `VerifyLayout`
are per-round native values and may reference backend-owned buffers. Dense
vocabulary distributions and target hidden tensors never cross JavaScript.

```rust
struct ProposerPlan {
    target: TargetContract,
    stages: Vec<Stage>,
    state: ProposerState,
    output: ProposerOutput,
    token_map: TokenMap,
    trained_max_rows: u32,
}

struct TargetContract {
    graph_fingerprint: Option<Fingerprint>,
    checkpoint_fingerprint: Option<Fingerprint>,
    vocabulary: u32,
    token_map_fingerprint: Fingerprint,
    hidden_taps: Vec<HiddenTapContract>,
    shared_weights: Vec<SharedWeightContract>,
}

struct Stage {
    operation: StageOperation,
    inputs: Vec<InputBinding>,
    outputs: Vec<ValueSchema>,
}

struct InputBinding {
    slot: u32,
    value: ValueRef,
}

enum ValueRef {
    PendingTokens,
    CandidatePrefix,
    CommittedHistory,
    TargetHidden { layer: u32 },
    SharedTokenEmbedding,
    SharedLmHead,
    StageOutput { stage: u32, output: u32 },
}

enum StageOperation {
    // Repeat one compiled causal step without crossing the host boundary.
    Autoregressive { component: u32 },
    // Run one compiled program over anchor/mask/query rows in parallel.
    ParallelBlock { component: u32, layout: BlockInputLayout },
    // Add a cheap causal head over outputs from an earlier parallel stage.
    SequentialHead { component: u32 },
    // Expand candidate children from target or prior proposer features.
    TreeExpand { component: u32, search: TreeSearchLayout },
    // Run a reduced path through the target's immutable weights.
    TargetPath { path: TargetPathId },
    // Produce token rows from committed token history.
    HistoryLookup { layout: HistoryLookupLayout },
}

struct HistoryLookupLayout {
    // The only stable deterministic retrieval contract in Phase 3.
    id: "suffix-ngram-v1",
    min_match_tokens: u32,
    max_match_tokens: u32,
}

enum ProposerState {
    None,
    Kv { schema: DecodeStateSchema, commit: CommitPlan },
}

enum CommitPlan {
    // Keep the checkpoint after the accepted prefix and, on full acceptance,
    // consume the final accepted candidate without sampling another draft.
    AutoregressiveChain { stage: u32 },
    // Replay explicit stages over the accepted target rows and feature taps.
    Replay { stages: Vec<u32> },
}

struct ProposerOutput {
    topology: CandidateTopology,
    probabilities: ProposalProbabilityContract,
    token_ids: ValueRef,
    probability_rows: Option<ValueRef>,
    parents: Option<ValueRef>,
    confidence: Option<ValueRef>,
}

enum ProposalProbabilityContract {
    CausalNormalized,
    Deterministic,
    Unavailable,
}
```

Component indices address `ProposerArtifactInput.components`; stage indices and
output indices must refer backward and form an acyclic graph. Every operation's
input slots and output schemas are validated against its component model during
artifact construction. `ProposerOutput` explicitly identifies the values that
become candidates, probability rows, parents, and confidence. There are no
implicit adjacent-stage inputs or conventional output slot numbers.

`TargetContract` is checked during `Model.inference`. Token mapping, vocabulary,
declared tap shapes/dtypes, and shared-weight names and shapes must match before
proposer programs are retained. A standalone draft normally leaves graph and
checkpoint fingerprints empty. A target-coupled checkpoint supplies either or
both when its training compatibility requires them. The runtime never infers
compatibility from a method name.

These variants describe dataflow, not named algorithms. A conventional draft
uses `Autoregressive`; DFlash binds hidden taps into `ParallelBlock`; DSpark
feeds that stage's hidden output into `SequentialHead`; EAGLE combines hidden
taps, autoregressive expansion, and `TreeExpand`; MTP/Medusa bind target values
to tree or chain expansion. A block program owns its arbitrary compiled mask,
positions, mask/noise tokens, and diffusion timestep inputs. The backend
orchestrator follows bindings; it does not contain a DFlash or DSpark branch.

`CommitPlan` is part of the executable contract. `AutoregressiveChain` requires
the stage to expose a KV checkpoint after every proposal step and defines the
full-acceptance state-only catch-up. `Replay` reruns the listed DAG stages over
the accepted target rows using their ordinary bindings and discards outputs.
The backend rejects a plan whose stateful stages are not covered by its commit
recipe.

The first speculative milestone supports exactly one `Autoregressive` stage
with KV state, `AutoregressiveChain` commit, chain output, and
`CausalNormalized` probabilities. Other variants are present in the plan schema
so adding a trained proposer does not change `Model.inference` or `runRound`,
but a backend rejects unimplemented variants at compile time.

```rust
struct CandidateBatch {
    lane_ids: Buffer<u32>,
    lane_offsets: Buffer<u32>,       // flattened candidate ranges, R + 1
    token_ids: Buffer<u32>,          // candidates in verification row order
    parent_rows: Buffer<i32>,        // -1 = lane root, otherwise local parent
    positions: Buffer<u32>,          // absolute target positions
    proposal_lengths: Buffer<u32>,   // logical prefix/tree rows per lane
    topology: CandidateTopology,
    probabilities: ProposalProbabilities,
}

enum CandidateTopology {
    Chains,
    Trees,
}

enum ProposalProbabilities {
    CausalNormalized(DistributionRows),
    Deterministic,
    Unavailable,
}

struct VerifyLayout {
    lane_ids: Buffer<u32>,
    row_offsets: Buffer<u32>,
    input_tokens: Buffer<u32>,       // pending anchor plus verification rows
    positions: Buffer<u32>,
    parents: Buffer<i32>,
    attention: VerifyAttention,
    graph_rows: u32,                 // physical bucket, >= logical rows
}

enum VerifyAttention {
    CausalChains,
    AncestorTrees,
}
```

For chains, parent rows and positions are derivable but remain explicit
validation data. For trees, a row attends to the committed prefix and only its
ancestor rows. Multiple trees or chains from one lane can be packed by using
multiple root children. `lane_offsets`, parents, and logical lengths define
semantic rows; `graph_rows` describes physical padding only.

A block-parallel proposer does not imply block target attention. DFlash and
DSpark produce a linear candidate chain for ordinary causal target
verification. Their bidirectional or non-causal block mask is part of the
proposer executable. EAGLE/Medusa candidate trees require ancestor target
attention. Keeping proposer execution layout separate from `VerifyAttention`
prevents these concerns from being conflated.

`DistributionRows` is an ephemeral native probability/logit handle aligned to
candidate rows after the same temperature, top-k, top-p, vocabulary mapping,
and other supported sampling transforms as the target. It is consumed during
the round and never published as a tensor. It must define a normalized `q` in
target-vocabulary space. Its support may be smaller than the target's: a reduced
vocabulary is scattered into target-vocabulary space with zero probability for
unrepresented tokens, and positive target-only mass is handled by residual
sampling.

Acceptance has only two exact built-in modes:

```rust
enum AcceptanceMode {
    RejectionChain,
    TargetSampleMatch,
}
```

`RejectionChain` requires chain topology and causal normalized rows.
`TargetSampleMatch` supports chains and trees using the root-to-leaf rule below.
No open-ended strategy-specific `AcceptancePlan` or heuristic mode is part of
the core contract. A future probabilistic multi-branch algorithm must first
standardize another proven acceptance mode; it cannot smuggle probability
accounting into a proposer-specific callback.

## Exact Sampling

### Effective distributions

Let `p_i` be the target distribution and `q_i` the proposal distribution after
all sampling controls for position `i`. Temperature, top-k, and top-p are part
of the distribution, not post-acceptance decoration. Target and proposer must
apply identical token masks and transforms when rejection sampling compares
them. If a transform cannot be represented by both sides, normalized-proposal
acceptance is unavailable and the engine uses target-sample matching.

For a causal normalized proposal token `x_i`:

```text
accept x_i with probability min(1, p_i(x_i) / q_i(x_i))
on rejection sample y from normalize(max(0, p_i - q_i))
on full acceptance sample one bonus token from p_(K+1)
```

This is required at positive temperature from the first speculative
implementation. Computing only `p_i(x_i)` and `q_i(x_i)` is insufficient because
residual sampling needs the positive difference over the effective vocabulary.

### Target-sample matching

Deterministic retrieval candidates and parallel blocks that do not expose a
causal normalized `q_i` use target-sample matching:

1. draw `y_i ~ p_i` at each candidate position along the candidate path;
2. accept while `y_i == x_i`;
3. at the first mismatch emit `y_i` as the correction;
4. after complete acceptance draw and emit the target bonus.

Conditioned on the accepted prefix, the emitted mismatch or bonus is an
ordinary target draw, so this preserves the target distribution. Proposal
randomness must be independent of the target RNG domain. For a tree, the target
draw at each node selects a matching child when one exists; otherwise that draw
is emitted as the correction. This follows one target-sampled root-to-leaf path
rather than choosing a path retrospectively from future draws.

Treating a deterministic candidate as a point-mass `q` and residual-sampling
from target with the candidate removed is also exact, but target-sample matching
is simpler and allows pathwise comparison to ordinary sampling.

Confidence thresholds, posterior token scores, or draft-target similarity do
not replace either exact rule. A heuristic mode, if ever added, has a separate
`exact: false` configuration and is excluded from exactness claims.

### RNG domains

Random draws are keyed, not consumed from a mutable batch-global stream:

```text
(effective lane seed, sequence id, absolute output position, purpose, subcounter)
```

Purposes include `proposal`, `accept`, `residual`, and `target`. Physical lane,
batch array order, packed row, accepted page length in another lane, and retry
count are not keys. Target-sample matching uses the same `target` coordinate as
ordinary generation. Rejection sampling uses independent `accept` and
`residual` domains. Proposer randomness cannot consume target randomness.

The inference sampling seed is the lane default. A per-entry override replaces
it for that page before coordinates are derived; it is not mixed with the
default and does not mutate the default for later calls. The resolved seed is
passed once to the native round and applies to all purpose domains without
making those domains overlap.

The existing `(seed, counter)` sampler is extended to derive its counter from
these coordinates. Seeded speculative execution is reproducible, but
probabilistic rejection sampling is not required to produce the same token
sequence as ordinary sampling for one seed; the required invariant is the exact
target distribution plus deterministic replay of each mode.

### Non-anticipating scheduling

Per-lane proposal lengths may depend on committed history, system load, and
confidence available for the admitted prefix. They must not depend on a future
candidate in a way that conditions whether an earlier candidate is admitted.
DSpark's confidence scheduler requires prefix-respecting early stopping for this
reason. Any adaptive policy must document its filtration and exactness argument.

Logical proposal lengths are represented independently of physical buffers.
A proposer may fill `[R, K]` while `proposal_lengths[r] < K`. Only the logical
prefix participates in verification, accounting, state publication, and
metrics. Packed verification uses row offsets/indptr metadata and rounds the
total valid rows to a compiled bucket. This supports different lengths in one
batch without requiring equal token rows.

## Strategy Mappings

### Conventional draft model

A small autoregressive model has separate KV state and emits one chain per
lane. Each candidate row carries its normalized causal proposal distribution.
The native round performs `K` draft steps without host calls, one packed target
verification, exact rejection/residual sampling, and the state-only catch-up
needed when the final candidate is accepted. This is the preferred first
implementation because it proves positive-temperature exactness, provisional
state, proposer synchronization, and batch-ragged publication.

### EAGLE and MTP/Medusa

The target exports configured hidden taps into native ephemeral buffers.
Auxiliary heads or a feature drafter emit one or more chains or a candidate
tree. The descriptor carries parent rows and positions; target verification
uses ancestor attention for trees. MTP heads that emit one selected chain use
causal verification. The artifact manifest fixes tap layer IDs, feature shape,
embedding alignment, vocabulary mapping, and head weights.

Tree construction and exact tree acceptance are separate from tree attention.
The first implementation uses `TargetSampleMatch`: target draws select a
matching child along one root-to-leaf path and become the correction when no
child matches. Probabilistic multi-branch rejection is out of scope until a
separately proven common acceptance mode is standardized.

### DFlash

DFlash fuses hidden states from configured target layers and injects their
projected K/V values into every draft layer. Its draft input is a committed
anchor plus masked positions. Draft positions attend bidirectionally within the
draft block and to injected context, producing the candidate block in one
parallel pass. Target verification remains a causal chain with all-row logits.

The artifact declares target tap layers, projection, trained block size, mask
token, shared embedding/LM-head requirements, and draft KV geometry. A smaller
runtime block may be valid only when allowed by the artifact. DFlash papers and
current serving implementations demonstrate positive-temperature operation,
but an effect-torch artifact must either expose a normalized causal proposal
factorization suitable for rejection sampling or use target-sample matching;
parallel per-position marginals must not be misrepresented as causal `q_i`.

### DSpark

DSpark uses a DFlash-style parallel backbone, then samples a linear chain with a
small sequential Markov or recurrent head. The sequential head yields a causal
normalized factorization, so its processed distributions can drive ordinary
rejection/residual sampling. A reduced draft vocabulary is mapped into target
vocabulary space before acceptance.

An optional confidence head emits conditional survival estimates. A scheduler
selects per-lane logical prefixes using a profiled target-cost curve and packs
the selected rows. Fixed-length operation is the first stable mode. Adaptive,
load-aware lengths are a later phase and require the non-anticipating proof and
ragged layout tests; they are not represented as a different generation API.

### Self-speculation

The proposer executable is a reduced path through the target, such as skipped
layers or an early exit. It has its own provisional execution state but may
share immutable weights. Its chain distributions use rejection sampling when
normalized; otherwise it uses target-sample matching. Compile-time lowering
must make shared-buffer and write ownership explicit.

### Retrieval, n-gram, and suffix proposals

The proposer reads committed token history and emits deterministic chains with
no model state. It uses target-sample matching. It is useful for repetitive or
editing workloads and as the simplest deterministic-proposer conformance test,
but it is not the architecture or performance foundation.

Phase 3 standardizes one layout, `suffix-ngram-v1`. For each lane it searches
committed history for a previous occurrence of the longest suffix whose length
is within inclusive `minMatchTokens..=maxMatchTokens`, then proposes the tokens
that followed that occurrence, capped by `maxDraftTokens`. A miss produces no
candidate rows, so the ordinary target draw publishes a one-token page. Ties
between occurrences are resolved by the most recent occurrence. The artifact
has zero components, `None` state, chain topology, deterministic probabilities,
an input-free intrinsic stage, an identity token map, and no probability rows. Its generalized compile request
contains no proposer executable or proposer pool: only target prefill, decode,
and packed verification programs are compiled.

### Lookahead and Jacobi-style proposals

Parallel fixed-point iterations can emit several chains. They lower to packed
chain rows or a tree when prefixes overlap. Without normalized causal proposal
probabilities they use target-sample matching. Proposer-specific iteration
masks remain inside the proposer executable.

## Backend Contract

The inference runtime extension is a required cohesive contract. Its concrete
TypeScript grouping may differ, but every runtime used by `Model.inference`
implements these operations rather than advertising them optionally:

```ts
interface InferenceRuntime {
  compile(request: InferenceCompileRequest): Effect<InferenceArtifactHandle, BackendError>
  open(artifact: InferenceArtifactHandle): Effect<GenerationHandle, BackendError>
  add(handle: GenerationHandle, request: BatchedPrefillRequest): Effect<RoundResult, BackendError>
  runRound(
    handle: GenerationHandle,
    request: BatchedRoundRequest
  ): Effect<RoundResult, BackendError>
  finish(handle: GenerationHandle, sequenceIds: ReadonlyArray<bigint>): Effect<void, BackendError>
  close(handle: GenerationHandle): Effect<void, BackendError>
  diagnostics(artifact: InferenceArtifactHandle): Effect<InferenceDiagnostics, BackendError>
}
```

`compile` must support fixed-width active masks, per-lane advances, all-row
verification outputs, explicit positions, causal-chain and ancestor-tree
attention, native feature routing, and transactional provisional state. A
backend may reject a concrete model/layout/dtype combination during compile,
but it cannot omit the core methods or make callers probe for them.

`runRound` owns proposal execution, target verification, sampling, acceptance,
and atomic publication. No logits, hidden taps, draft probability matrices, or
provisional handles need cross N-API. CPU and Metal implement the same semantic
request independently. A paired target/proposer artifact requires one runtime
identity; same display device is insufficient.

The normal path is a degenerate round with zero proposal rows. Its implementation
may call the existing fused sampled decode directly. General descriptor support
must not allocate tree metadata, full probability matrices, or provisional
blocks on that path.

## EOS, Budgets, And Failures

The sequence owns its EOS set and remaining output budget from `add`. Proposal
length is capped so a correction/bonus can fit. After acceptance, the page is
cut at the first EOS and never exceeds the remaining budget. State publication
follows the cut, not the initially accepted length. The final visible token,
including EOS, remains pending; rows corresponding to tokens after it are not
committed.

A page ending in EOS or exhausting the budget atomically marks the sequence
terminal and reports `stopReason`. A terminal sequence still owns native state
until `finish` or `close`, but `step` rejects it before execution. No later page
can expose tokens computed after its terminal boundary.

Target/proposer vocabulary, tokenizer mapping, runtime identity, state kind,
feature taps, block bounds, and sampling-transform compatibility are validated
before sequence creation. A proposer, verifier, sampler, acceptance, pool, or
publication failure leaves all active target and proposer states unchanged.
The returned error identifies the phase. A failed round may be retried with the
same deterministic coordinates.

## Initial Implementation Plan

Each phase is independently reviewable and keeps the repository buildable. A
later phase does not preserve the API removed by an earlier one.

### Phase 1: one sampled batch API

**Status: Implemented.** The release-build ordinary-generation baseline on the
development Apple Silicon host was 718 us/token (CPU B=1), 493 us/token (CPU
B=8), 599 us/token (Metal B=1), and 104 us/token (Metal B=8), measured over 100
rounds with `pnpm bench:inference`.

1. In `Model.ts`, replace `decodeBatch` with `batchSize`, remove the separate
   single decode trace, and compile only `[B, 1]` decode.
2. Extend required decode runtime requests with active masks, per-lane advances,
   and batched prefill valid lengths. Implement the same contract on CPU and
   Metal before changing consumers.
3. Add `StatefulExecution`, migrate the existing logits-returning behavior to
   it, and preserve transactional input-token advancement and explicit tensor
   ownership.
4. Replace `Generation.addSampled`/`stepSampled` and logits-returning
   `Generation.add`/`step` with the array-based token-page API. Store pending
   tokens, EOS sets, remaining budgets, terminal state, and stable sequence IDs
   in the generation session.
5. Make batched `add` admission all-or-nothing and implement fixed-slot refill
   without changing sequence RNG identity.
6. Update `Chat.ts` to consume every token in a page, stop at the reported
   terminal reason, and never decode text or invoke callbacks for a discarded
   suffix.
7. Run existing CPU/Metal generation tests through batch sizes one and greater
   than one, then benchmark the ordinary one-token path before continuing.

### Phase 2: exact chain speculation

**Status: Implemented.** CPU and Metal use packed causal-chain verification and
cohesive native artifact/session/sequence handles with lossless keyed RNG,
exact rejection/residual sampling, durable request-bound receipts, atomic paired
prefix-cache publication, and real phase/acceptance/pool diagnostics. Native and
cross-backend tests cover every acceptance cut, failure/interruption rollback,
prefix reuse, high-bit seeds, and positive-temperature replay; benchmarks cover
ordinary, perfect-draft, and independent-draft modes across batch width, context
length, and concurrent sessions.

1. Add `ProposerArtifact` and implement only the structural
   `Autoregressive`/KV/chain/`CausalNormalized` plan. Validate target/proposer
   runtime, vocabulary mapping, sampling transforms, and state geometry during
   `Model.inference`.
2. Extend batched `add` to prefill target and proposer transactionally and
   install the first target-sampled token as pending in both states.
3. Compile target verification with all-row outputs and add packed row offsets,
   positions, active lanes, and logical proposal lengths to the required CPU
   and Metal contracts.
4. Add internal provisional KV roots, accepted cursors, staged prefix-cache
   ownership, and one atomic multi-sequence publication operation.
5. Implement the native conventional draft loop, target verification, exact
   positive-temperature acceptance/residual/bonus sampling, and keyed RNG
   domains in one `runRound` call.
6. Implement proposer `commitThrough`, including full-acceptance catch-up and
   rejection replacement alignment. Test the proposer and target cursor/pending
   token pair after every possible cut.
7. Add per-lane logical proposal lengths and packed ragged verification even
   though the first fixed scheduler normally selects one common maximum.
8. Treat this phase as the first useful speculative release. It must pass the
   positive-temperature and transactional acceptance criteria before adding
   another proposer stage.

### Phase 3: target-coupled proposer stages

**Status: In progress.** Structural plans, graph/checkpoint and target-value
validation, multi-input/multi-output stage tracing, immutable stage compilation,
direct CPU/Metal native value routing, complete `suffix-ngram-v1` history-lookup
rounds, and fixed-block `ParallelBlock` generation are implemented on CPU and
Metal. `ParallelBlock` covers target-sample matching, normalized-row rejection,
target-feature replay, and transactional publication. `SequentialHead` and tree
verification are not yet executable generation strategies.

1. Add ephemeral native target hidden-tap routing and shared embedding/LM-head
   bindings declared by `ValueRef` and `TargetContract`.
2. Implement `ParallelBlock` and fixed-length DFlash chain artifacts using
   `TargetSampleMatch`; keep draft block masks inside the compiled proposer.
3. Implement `SequentialHead` and fixed-length DSpark artifacts. Use
   `RejectionChain` only when the artifact exports normalized causal rows.
4. Implement `TreeExpand`, ancestor-tree verification, and EAGLE/MTP/Medusa
   artifacts using exact root-to-leaf `TargetSampleMatch` first.
5. Require each artifact loader to validate its declared target graph/checkpoint
   fingerprints, tap shapes, shared weights, token maps, trained row bounds,
   and state schema.
6. Implement deterministic `HistoryLookup` through the same artifact and
   `TargetSampleMatch` path as a low-cost conformance proposer, not as the
   performance foundation.

### Phase 4: adaptive scheduling

1. Profile verification cost buckets.
2. Add DSpark-style confidence scheduling with per-lane logical lengths and a
   documented non-anticipating admission proof.
3. Add load-aware speculation disablement and proposer selection.
4. Benchmark lane refill, mixed prefill/decode, and overlap scheduling before
   enabling adaptive policy by default.

## Acceptance Criteria

1. Batch size one and batch size greater than one use the same public and
   compiled path.
2. Ordinary generation returns one-token pages.
3. One speculative round can return different nonzero page lengths per lane.
4. Inactive lanes do not execute, allocate, draw randomness, or advance.
5. An oversized `add` fails without admitting, allocating, or prefilling any
   sequence; successful result cardinality always equals input cardinality.
6. EOS and `maxTokens` inside a computed page return only the allowed prefix and
   commit no later row.
7. A page ending at EOS or `maxTokens` marks its sequence terminal, and another
   `step` fails without execution.
8. Seeded ordinary and speculative modes replay independently across runs,
   array reorderings, lane refill, and different neighboring acceptance lengths.
9. Small synthetic distributions assert fixed-draw acceptance and exact residual
   outcomes, including `p = q`, disjoint support, poor drafts, and perfect
   drafts. Statistical tests supplement rather than replace these tests.
10. Positive-temperature speculative output matches the target distribution for
    deliberately poor and perfect draft distributions.
11. Exactness and publication hold across KV block boundaries and partially
    filled blocks.
12. Every acceptance cut leaves target and proposer state synchronized through
    all but the returned page's final pending token, including full-acceptance
    catch-up.
13. Failure or interruption in proposer, verifier, sampler, acceptance, or
    publication leaves every active target and proposer cursor unchanged.
14. Rejected provisional blocks return to the pool without becoming committed
    prefix-cache entries or corrupting shared prefix ownership.
15. Incompatible target/proposer runtime, vocabulary, feature, state, and
    artifact geometry fail before sequence mutation.
16. Stage bindings, output references, target contracts, commit recipes,
    candidate parents, positions, offsets, logical lengths, probabilities, and
    masks reject malformed inputs before execution.
17. `StatefulExecution` commits exactly caller-supplied tokens, returns
    caller-owned logits, and cannot exchange sequence handles with `Generation`.
18. Tree rows cannot attend siblings or descendants; chain rows remain causal.
19. DFlash/DSpark draft masks are tested independently from causal target
    verification masks.
20. The normal sampled path shows no material latency or allocation regression
    from descriptor generality.
21. Benchmarks report draft time, verify time, acceptance length distribution,
    output tokens/second, latency, pool high-water mark, and behavior across
    concurrency and context lengths. Greedy-only benchmarks do not satisfy the
    exact sampling milestone.

## Non-Goals

- Cross-runtime or cross-backend speculative orchestration in the first
  implementation.
- Arbitrary rollback of KDA, convolution, Mamba, or other recurrent state.
- A public mutable sequence cursor or truncate API.
- Per-token JavaScript draft loops.
- Training recipes for EAGLE, MTP, DFlash, or DSpark artifacts.
- Making n-gram lookup the primary general-workload optimization.
- Claiming an unproven tree/block heuristic preserves the target distribution.

## Relationship To RFC 0014

This RFC supersedes RFC 0014 outright. RFC 0014 remains in the repository as a
historical draft, but its proposed public `Sequence.truncate(position)`, rolling
`2 * blockSize` history assumption, single-sequence `generate` engine,
greedy-primary invariant, and n-gram-first phasing must not be implemented.

The native seeded sampler and fused sampled execution completed after RFC 0014
remain useful foundations. Their role here is expanded to batched token pages
and exact acceptance inside an internal round transaction.

## References

- Leviathan et al., *Fast Inference from Transformers via Speculative
  Decoding*: <https://arxiv.org/abs/2211.17192>
- Chen et al., *Accelerating Large Language Model Decoding with Speculative
  Sampling*: <https://arxiv.org/abs/2302.01318>
- Cai et al., *Medusa*: <https://arxiv.org/abs/2401.10774>
- Li et al., *EAGLE*: <https://arxiv.org/abs/2401.15077>
- Chen et al., *DFlash: Block Diffusion for Flash Speculative Decoding*, v2:
  <https://arxiv.org/abs/2602.06036>
- Cheng et al., *DSpark: Confidence-Scheduled Speculative Decoding with
  Semi-Autoregressive Generation*: <https://arxiv.org/abs/2607.05147>
- DFlash reference implementation: <https://github.com/z-lab/dflash>
- SGLang DFlash implementation and ragged verification (inspected 2026-08-14):
  <https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/speculative>
- vLLM rejection sampler and DFlash/DSpark implementations (inspected
  2026-08-14):
  <https://github.com/vllm-project/vllm/tree/main/vllm/v1/worker/gpu/spec_decode>

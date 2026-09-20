# RFC 0027: Model families and decision inference

- **Status**: Draft
- **Created**: 2026-09-20
- **Depends on**: [RFC 0005](0005-models.md), [RFC 0010](0010-inference.md),
  [RFC 0013](0013-batched-decode.md),
  [RFC 0019](0019-executable-compilation.md),
  [RFC 0020](0020-invocation-ownership.md),
  [RFC 0023](0023-batched-speculative-generation.md),
  [RFC 0025](0025-target-dtype-legalization.md)
- **Updates**: RFC 0005 public module responsibilities, RFC 0010 inference API
  ownership, RFC 0020 persistent read-only state, and RFC 0023 generation API
  scope.

## Summary

Organize the public API into four modules:

- `Model`: generic graph building blocks, parameters, composition, and ordinary
  forward execution.
- `AutoRegressiveModel`: causal inference, sequence progression, sampling, and
  speculative generation.
- `DiffusionModel`: encoder and denoiser contracts, self-conditioning, iterative
  refinement, stopping, and completed-block commits.
- `DecisionModel`: selected-answer scoring, independent evaluations, repeated-read
  aggregation, and decision distributions.

All four use the existing graph/compiler/runtime pipeline. Model families have
different execution contracts, but share parameter preparation, executable
construction, invocation memory, state storage, batching mechanisms, and cleanup.

Normal DiffusionGemma inference and Jev-like decision evaluation are both required.
The decision API is an additional consumer of model evaluation. It does not define
the limits of diffusion support. Completion requires both multi-step, multi-block
generation and independent decision reads to work through shared inference
infrastructure.

This RFC is an implementation plan. The new family modules and native state
extensions described below are not implemented by the current prefix executor.

## Current state and problem

`packages/core/src/Model.ts` combines generic model construction, ordinary forward
execution, autoregressive inference compilation, generation sessions, stateful
logits execution, and speculative-decoding orchestration.

The recent DiffusionGemma work introduced `Model.PrefixModel`, `Model.executor`,
and `Model.executeLayers`. They build and materialize one layer at a time through
`Tensor.compute`. This shares native tensor execution, but bypasses
`Model.inference` and its compiled artifacts, state pools, and prefix caching.
Moving that code from the model module to `Model.ts` did not integrate the two
inference paths.

The current DiffusionGemma executor implements independent reads with initial
self-conditioning and selected logits. Repeating a read resets that state. It
does not implement iterative denoising or normal block generation. CPU/Metal
tiny-model parity and component checks establish a reference for those reads;
they do not establish full-model parity or complete diffusion-generation support.

Several existing facilities should be reused:

- `InferenceProgram.execution()` already returns caller-owned logits for
  caller-selected tokens.
- `Tensor.compileDecodeProgram` and native `KvAttentionMode::BidirectionalBlock`
  already support bidirectional blocks for DFlash.
- Executables, invocation workspaces, cancellation, state transactions, output
  selection, and speculative state management already exist.
- `Tensor.expose` already identifies diagnostic and proposer-visible graph values.

There are also real gaps:

- Decode compilation currently requires uniform K/V head geometry across layers.
- Existing stateful execution appends K/V and advances sequences. Bidirectional
  visibility alone does not create read-only prefix execution.
- Composed stepwise attention and explicit RoPE preserve reference numerics but
  hide these operations from decode specialization.
- The target inference API assumes one forward builder traced for causal prefill
  and token decode. DiffusionGemma needs distinct encoder and denoiser graphs.

These gaps belong in shared compilation and state contracts. A second serving
runtime would reproduce the same work with different ownership rules.

## Decisions

1. Add `AutoRegressiveModel.ts`, `DiffusionModel.ts`, and `DecisionModel.ts` to
   `packages/core/src`, with exports from `packages/core/src/index.ts`.
2. Keep `Model.ts` responsible for architecture and ordinary graph execution.
   Family-specific inference configuration and sessions move to their family.
3. Keep DiffusionGemma architecture, configuration, and checkpoint interpretation
   in `packages/core/src/models/DiffusionGemma.ts`.
4. Implement full diffusion generation and raw diffusion evaluation over the same
   compiled model operations and retained parameter generation.
5. Let `DecisionModel` consume scoring operations supplied by either family. It
   does not require a `generate()` call or generated JSON.
6. Reuse common inference internals and native execution. There is no new public
   peer executor that builds a model layer by layer for production serving.
7. Describe state access, attention visibility, output selection, and per-layer
   geometry explicitly. Required operations belong to distinct interfaces or
   tagged variants, rather than a universal model with optional hooks.
8. Preserve explicit ownership and the numerical contracts of the graph. Model
   family selection cannot introduce a dtype substitution or backend fallback.

## Public module responsibilities

AutoRegressiveModel and DiffusionModel depend on Model and shared inference
internals. DecisionModel's adapters depend on the family APIs; the family modules
do not depend on DecisionModel. Shared core code does not import concrete models
or backends. Applications continue to select a runtime through an Effect Layer.

### Model

Retain `Model`, `Definition`, `Params`, parameter specifications, initialization,
composition, generic layers, model errors, and ordinary `Model.execute` caching.
The pure graph path remains usable for training and autodiff. Generic graph
exposure names such as `Model.hiddenExposure` also remain here.

Family definitions compose these building blocks and ordinary Tensor graphs.
The existing single-input/single-output `Model.forward` need not become an
optional multi-mode interface to describe encoder and denoiser entry points.

### AutoRegressiveModel

Move the existing `Model.inference` API to `AutoRegressiveModel.inference`, together
with its inference configuration, errors, program, generation, and stateful
execution types. Preserve behavior during this move:

- Fixed-shape prefill and decode programs with shared parameter retention.
- Batched sampled generation and caller-owned-logits execution.
- Prefix reuse, sequence cleanup, KDA and convolution state, and diagnostics.
- Speculative target/proposer compilation and the existing acceptance rules.

`Speculation` remains the proposer-description module. DFlash remains a
bidirectional proposer inside autoregressive generation; it is not reclassified
as a standalone diffusion model merely because it processes a parallel block.

The public nonempty-token-page contract from RFC 0023 remains valid for
autoregressive generation.

### DiffusionModel

Provide a family definition and compiled artifact for block-diffusion language
models, starting with DiffusionGemma. The initial contract covers causal context
encoding and denoising against an encoded prefix. Other diffusion architectures
require their own explicit contracts if their state semantics differ.

The definition supplies lazy graphs and typed metadata for:

- Context encoding, including model-specific encoder computations.
- Canvas evaluation with explicit positions, a borrowed prefix, and prediction
  feedback from the previous denoising step.
- Initial prediction state and self-conditioning computation.
- Readout from hidden rows, with full or selected outputs as required by the caller.

Configuration and parameter values remain explicit. Encoder and denoiser graphs
share tied parameters while retaining distinct parameters where the model defines
them, including DiffusionGemma's independent layer scalars. A mutable ambient
encoder/decoder mode must not change the meaning of a traced graph.

Use `DiffusionModel.inference` as the family compilation entry point. Its artifact
offers both a normal generation session and lower-level evaluation operations:

| Operation              | State contract                                                       | Result                                                    |
| ---------------------- | -------------------------------------------------------------------- | --------------------------------------------------------- |
| Encode context         | Causal append using the encoder graph                                | An owned, reusable prefix                                 |
| Evaluate canvas        | Borrow prefix; keep canvas K/V in invocation storage                 | Owned predictions and required feedback outputs           |
| Commit completed block | Encode accepted tokens causally into a new prefix version            | An extended prefix with the original snapshot still valid |
| Generate               | Own canvas, feedback, sampler state, and progress across evaluations | Committed output token pages                              |

These are behavioral contracts; exact session and method signatures are to be
settled with the shared state API in phase 2. A caller-supplied initial state and
a carried refinement state must be distinguishable without an optional collection
of unrelated tensors.

The normal driver performs the full model algorithm:

1. Encode the prompt.
2. Initialize a canvas and its prediction state.
3. Evaluate, sample/accept/renoise, update self-conditioning, and check stopping.
4. Emit the completed block and, when continuing, encode it causally as context
   for the next block.
5. Continue until sequence stopping or the requested output limit.

Canvas K/V from the denoiser cannot become committed encoder K/V. Completed tokens
must pass through the encoder computation. Output limits, EOS handling, partial
final blocks, padding, positions, and feedback ordering follow the pinned model
reference.

A refinement iteration may commit zero tokens. Represent that as internal
progress, separate from a committed token page or completion. A high-level stream
publishes committed pages; it does not mistake an empty refinement result for EOS.
This extends internal scheduling without weakening autoregressive page guarantees.

### DecisionModel

Provide a small scoring contract over prepared, tokenized inputs, with adapters
for each family's compiled artifact. The contract describes ordered
answer selections, evaluation inputs, result ownership, and supported read
policies. It carries typed errors and Effect requirements through the adapter.
It does not downcast models or probe for optional methods at execution time.

An autoregressive adapter can score the next position from causal inference
without generating an answer sequence. A diffusion adapter evaluates a supplied
canvas and selects answer-position logits. Both produce logits in the requested
answer order. The initial adapter contract uses verified single-token answer
codes; multi-token answers need an explicit sequence-scoring definition rather
than a sum of unrelated position logits.

Reusable decision operations include:

- Stable restricted softmax over the selected answer set.
- Aggregation of independently evaluated reads.
- Binary probability, categorical choice, and expected-value readouts.
- Batched independent scoring through the shared inference runtime.

The application chooses the read policy and prepares its inputs. For the current
independent-noise policy, average per-read probabilities, not logits. Four fresh
reads are not four successive denoising steps. An iterative decision policy must
explicitly carry prediction state within each refinement chain.

The sibling `../inference/decision-model` project retains Jev HTTP schemas,
scaffolds, tokenizer-specific answer-code verification, question routing, noise
seeding policy, confidence conventions, and response serialization. Core scoring
does not depend on those wire formats.

## Shared inference contracts

### One compilation and execution path

Extract common TypeScript preparation into private core internals. Reuse
`Tensor`, `Runtime`, and the existing Rust compiler and backend executables.
Shared responsibilities are parameter materialization/retention, graph tracing,
state-schema validation, executable construction, output metadata, and cleanup.
Autoregressive sampling and diffusion refinement remain family policies.

Compile reusable entry points for the configured shapes. Token IDs, positions,
prefix bindings, feedback, and eligible answer selections are invocation inputs.
Executable identity includes runtime, metadata, semantic attributes, and relevant
state schema. Model weights and prefixes must not become accidental value-specific
cache entries through repeated graph construction.

Prefill chunk widths, canvas widths, active batch lengths, and output selection
must be supported deliberately. The initial implementation may batch compatible
phases separately. Mixing causal and bidirectional requests within one attention
kernel is an optimization, not a prerequisite for sharing the executor.

Diagnostics use graph exposures and explicit diagnostic outputs. If native memory
planning needs layer partitions, those partitions belong to the compiled plan.
A JavaScript callback after each materialized layer is not the production model
execution contract.

### Persistent state versus invocation storage

Represent state access with explicit stateless, read-only, and append variants.
Attention visibility is a separate property. A bidirectional graph can still
write state under the existing implementation, so visibility must not imply
ownership or commit behavior.

For read-only execution:

- Query-row validity is independent of committed token count.
- The executable borrows immutable prefix pages and their logical positions.
- Current canvas K/V belongs to the invocation and is discarded after the read.
- Prefix values, cursors, block tables, cache identity, and recurrent state remain
  unchanged on success, failure, or interruption.
- Concurrent reads share the prefix while owning separate scratch and outputs.

For append execution, state changes publish transactionally after success.
An immutable prefix snapshot and an appendable sequence are distinct ownership
contracts. Appending while a snapshot is borrowed preserves existing pages and
uses a private writable tail where needed. A zero cursor increment alone does
not prove read-only behavior if a kernel has overwritten shared storage.

Logical prefix matching and physical page sharing must agree. In particular,
CUDA sequence forks that copy entire cache storage are not sufficient evidence
of shared-prefix efficiency. State accounting must expose actual retained and
copied bytes.

### Heterogeneous per-layer schemas

Replace the single K/V head-count and head-width pair in decode geometry with
ordered per-layer descriptors. They include logical layout, K/V shapes and
dtypes, retention requirements, and stable layer identity. Existing homogeneous
models become a repeated descriptor list with equivalent behavior.

Prefill, denoiser, and commit programs must agree on the persistent state schema
they exchange. Local storage length is independent of logical prefix length.
Retention policy and query attention masks are also distinct. DiffusionGemma's
local prefix trimming must not cause an inherited causal decoder mask or reset
absolute canvas positions. Use the pinned model's mask semantics as the oracle.

KDA and convolution state remain supported by the autoregressive path. The new
K/V schema must not erase their independent state contracts.

### Numerical operations and output selection

Carry explicit RoPE positions/frequencies and attention rounding semantics into
Rust semantic operations so state specialization can recognize them. Update
inference validation, lowering, autodiff where applicable, and backend capability
classification together. Preserve BF16 storage and rounding boundaries, F32
opmath, MoE combination order, and F32 logit softcap from the reference.

Treat full-canvas/full-vocabulary and selected-row/selected-label readout as
explicit output requirements. Normal self-conditioned denoising may require
predictions across the whole canvas. A selected decision head cannot silently
substitute for that feedback. The compiler can prune unused outputs and compute
feedback on-device where the model contract permits it.

The numerical reference stays independent of fused backend choices. Unsupported
operations fail on the chosen backend under the existing capability policy.

### Ownership and cancellation

Share one retained parameter generation across family entry points and decision
adapters. Compilation must not duplicate the checkpoint merely to expose both
generation and scoring. Prefix, session, invocation, and output ownership follow
RFC 0020, with explicit release for capacity-sensitive resources.

Record newly owned resources at their acquisition boundary. Effect exit handlers
protect registration and cleanup when an acquisition succeeds; the acquisition
itself remains responsible for partial resources on interruption. Native adapters
retain responsibility for late results. A success-only continuation is not a
substitute for interrupted-acquisition cleanup.

Returned logits own their handles independently of later invocations. Closing a
decision request releases its temporary state after all borrowers finish. Closing
a generation session releases its mutable state without invalidating separately
owned immutable prefix snapshots or completed output handles.

## Jev-like independence

The intended dependency remains:

```text
answer_i = f(model, state, question_i, read_policy)
```

Sibling questions, their order, their caller IDs, and their answers must not enter
another question's computation. Batched rows remain isolated through every layer.
Packing several question definitions into a shared bidirectional context would
violate this even if final answer slots were masked separately.

Reuse only identical encoded token prefixes. A shared state prefix may support
private question suffixes when the validated template permits that arrangement.
Changing the template to obtain more prefix sharing requires a separate quality
evaluation. Whole-prompt reuse remains useful while suffix branching is developed.

Caller IDs route outputs. Semantic question contents determine the model input
and, with a supplied seed, reproducible independent noise. Choice descriptions
and names remain model-visible. Decision outputs contain complete selected-answer
distributions; application code derives Noul, Choice, Score, and confidence values.

Ordinary autoregressive generation, normal diffusion generation, and Jev-style
evaluation each have acceptance gates. Passing one does not establish the others.

## Migration plan

### Phase 1: Extract autoregressive APIs without behavior changes

- Move inference configuration, program/session types, `inference`, generation,
  stateful execution, and autoregressive orchestration to `AutoRegressiveModel.ts`.
- Update `Chat.ts`, `Speculation.ts` references, model documentation, examples,
  benchmarks, tests, and root exports. Preserve generic graph utilities in `Model`.
- Publish a symbol migration map. Update in-repository consumers together;
  compatibility aliases, if required for a published release, must be direct
  re-exports with a removal point rather than duplicate implementations.
- Keep this a mechanical review unit. Gate it with existing generation,
  speculative, recurrent-state, chat, and ownership tests.

### Phase 2: Extract shared artifact preparation and define state contracts

- Separate common parameter/executable preparation from autoregressive scheduling
  in the existing inference implementation.
- Specify immutable prefix handles, append transactions, per-layer state schemas,
  read-only invocation bindings, and full/selected output metadata.
- Finalize the family artifact and decision-scoring signatures against those
  contracts. Record supported combinations explicitly rather than adding flags
  for each model name.
- Run the existing autoregressive path through the extracted implementation
  before adding a second consumer.

### Phase 3: Implement shared native support

- Extend `crates/graph`, `crates/compiler/src/decode.rs`, and shared runtime types
  for semantic attention/RoPE requirements and heterogeneous K/V schemas.
- Implement read-only prefix plus temporary-canvas execution in CPU, Metal, and
  CUDA, with per-layer storage and transactional append behavior.
- Audit CUDA physical prefix sharing, copy-on-write tails, and invocation storage;
  remove whole-cache copying from the intended shared-prefix read path.
- Update `Runtime.ts`, thin `Tensor.ts` bindings, and backend adapters. Regenerate
  Node-API declarations from Rust changes.
- Verify these operations using small graphs independent of DiffusionGemma before
  integrating the full model. Preserve existing causal and DFlash behavior.

### Phase 4: Implement complete DiffusionModel inference

- Add the diffusion definition/artifact and normal generation driver.
- Adapt DiffusionGemma graphs to compiled encoder, denoiser, feedback, and readout
  operations with shared parameters and the pinned checkpoint semantics.
- Implement initial noise, iterative feedback, acceptance/renoising, stopping,
  completed-block encoding, and continuation with the pinned generation policy.
- Add direct raw-evaluation access without going through text generation.
- Validate more than one denoising step and more than one generated block.
  Compare intermediate feedback and committed state, not only final sampled text.
- Replace the interim public `Model.PrefixModel` and `Model.executor` once the
  shared path passes its gates. Preserve a focused reference harness in tests or
  verification examples; reassess `executeLayers` for genuine generic callers.

### Phase 5: Add DecisionModel and migrate the Jev application

- Add family scoring adapters, ordered selections, restricted normalization, and
  independent-read aggregation. Reuse compiled artifacts and parameter owners.
- Test both autoregressive next-position scoring and diffusion canvas scoring.
- Migrate `../inference/decision-model/src/DiffusionRuntime.ts` to the new adapter
  and reuse core arithmetic where its documented semantics match.
- Preserve scaffold verification, deterministic semantic noise keys, question
  isolation, lease-safe caching, and the external response contract.
- Validate selected outputs against full readout, including label ordering, and
  verify that one/four independent reads retain their existing meaning.

### Phase 6: Integrate normal generation consumers and remove obsolete paths

- Adapt chat and generation examples to committed token pages from either family
  through the smallest shared consumer contract. Preserve autoregressive custom
  sampling behavior; family-specific controls remain explicitly typed.
- Add a runnable normal DiffusionGemma generation example and a decision example
  using the same model artifact and retained weights.
- Remove superseded execution helpers, stale exports, and obsolete documentation
  only after consumer migration and parity checks.
- Run end-to-end CUDA validation and workload measurements before declaring the
  new implementation the production path.

Phases 4 and 5 are both completion requirements. The ordering supports incremental
verification; it does not defer normal inference behind a decision-only release.

## Validation and acceptance

Use the existing reference baseline: `google/diffusiongemma-26B-A4B-it` checkpoint
revision `f7f5b7f5fa82ffc52addd066915886d497f5517b` and Transformers revision
`93ebf6b11127967f2725cf4d012aae55c3654f5a`. Existing tiny fixtures and oracle tooling
live in `../inference/decision-model`. Extend those fixtures for normal generation.
Record and replay random inputs when comparing sampler decisions across frameworks;
a shared seed alone does not establish identical random draws.

### Existing behavior

- Generic `Model` composition, training, autodiff, and ordinary execution remain
  usable without family imports.
- Autoregressive prefill/decode, sampling, batched lanes, speculative acceptance,
  recurrent state, prefix reuse, and chat output retain their established behavior.
- DFlash continues to use bidirectional blocks with its existing commit policy.

### Shared state and ownership

- Heterogeneous local/global K/V agrees across encode, read, and commit programs.
- Concurrent reads preserve prefix values and metadata exactly and never share
  writable canvas storage. Logical offsets survive local-prefix trimming.
- Append/commit preserves borrowed snapshots. Failure and interruption publish
  no partial state and release native late outputs.
- Releasing one result leaves other outputs, prefixes, and parameter owners valid.
- Measured physical storage demonstrates prefix sharing and one retained weight
  generation; logical cache-hit counters alone do not pass this gate.

### Normal diffusion inference

- Pinned tiny F32/BF16 tests cover feedback across multiple denoising iterations,
  sampler decisions, convergence and step limits, and initial-state reset only
  at the correct boundaries.
- Multi-block tests compare encoder-committed K/V, logical positions, the next
  block's inputs, EOS/length handling, and emitted token pages with the reference.
- Cancellation during encoding, refinement, and commit releases request-owned
  state while other sessions sharing the artifact remain usable.
- Selected full-checkpoint CUDA runs exercise the complete normal generation
  path. Existing official oracle success is not effect-torch full-model parity.

### Decision inference

- Full and selected readouts agree within the existing numerical criteria.
  Restricted distributions normalize correctly and independent reads average
  probabilities according to the declared policy.
- Rename, reorder, add, remove, and duplicate sibling questions without changing
  a target question's controlled-noise result. Test sibling-only fact leakage.
- Batched and isolated evaluations agree with the same inputs and noise.
- Both generation and decision evaluation can borrow the same artifact without
  state contamination or duplicate checkpoint materialization.
- Model quality and calibration are evaluated separately from API compatibility
  and numerical parity. One/four-read behavior is not a claim of Jev quality parity.

### Checks and measurements

Run targeted TypeScript and Rust tests first, then workspace typecheck/lint and
relevant native checks. Rust boundary changes require
`pnpm generate:native-types` and `pnpm check:native-types`; native package metadata
changes also require package verification. Tests use the selected backend and
must not hide unsupported behavior behind a CPU fallback.

Keep the established numerical tolerances for existing fixtures. Normal
multi-step fixtures add their own reference evidence rather than weakening those
criteria to accommodate a new execution path. Record CUDA validation separately
from CPU/Metal results.

Measure cold compilation, warm invocation, prefill, denoising, commit, and decision
latency separately. Track peak weight/state/workspace bytes, prefix hits and copied
bytes, concurrent request capacity, and request latency distributions. Compare
autoregressive regressions against its existing implementation. Decision latency
and decisions per second are separate from generated tokens per second; quality
targets and performance thresholds require explicit workload results.

## Prior art and scope

- [vLLM's DiffusionGemma integration](https://vllm.ai/blog/2026-06-10-diffusion-gemma)
  uses model-specific `ModelState` and sampling over its shared runner and cache.
  It reuses speculative scheduling to account for zero-token denoising progress
  and completed-block commits. Reuse of scheduling does not make denoising an
  exact speculative-rejection algorithm.
- [SGLang's DllmAlgorithm](https://github.com/sgl-project/sglang/blob/ee464fed/python/sglang/srt/dllm/algorithm/base.py)
  separates denoising algorithms from its model runner. Its specific
  [DiffusionGemma serving PR](https://github.com/sgl-project/sglang/pull/34061)
  was still open when checked on 2026-09-20; documentation alone is not evidence
  of shipped support.
- [Pinned Transformers generation](https://github.com/huggingface/transformers/blob/93ebf6b11127967f2725cf4d012aae55c3654f5a/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py)
  defines the reference refinement, self-conditioning, and block-generation
  behavior. Architecture ideas from other engines do not replace that numerical
  and masking oracle.

This plan does not require support for every diffusion family, new quantization,
or distributed deployment. Those can extend the same contracts after the normal
DiffusionGemma and Jev-style paths are implemented and measured.

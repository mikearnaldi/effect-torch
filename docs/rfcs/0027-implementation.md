# RFC 0027 implementation progress

Updated: 2026-09-20. Paused after passing pinned generation and decision acceptance.

## Current status

Full-model generation replay passes in run 65. Both width-256 blocks, all six
step argmax canvases and all 512 final tokens match the pinned reference exactly.
All eight recorded random canvases and six exponential inputs are consumed as
expected, and external memory returns to zero after cleanup. The original
checkpoint, initialized state 55, random draws and acceptance criteria are retained.

The resumed investigation identified two rounding differences:

- DiffusionGemma's scalar softcap division must use a rounded F32 reciprocal,
  as the pinned scalar operation does. Run 60 reconstructs all 786,432 saved
  first-step probe logits exactly with this form. The correction alone leaves
  the original generation failure in run 61.
- CUDA's single-warp Sum accumulates too many values sequentially on wide rows.
  On identical full-vocabulary feedback, run 62 finds 6,568 different BF16
  probabilities, 78 different conditioning-signal values and 2,350 different
  conditioning outputs. Reference capture 63 reproduces the original generation
  and confirms the same-input Torch conditioning output exactly. Block-sum run
  64 matches every captured conditioning stage, including all 67,108,864 BF16
  probabilities. Generation run 65 then closes the exact replay gate.

CUDA policy 13 uses a 1024-thread, vector-four block reduction for F32-opmath
Sum with at least 4,096 reduced elements, with scalar tails and arbitrary axes.
Short sums retain the existing warp schedule. This is a backend-defined Sum
order, not a promise of universal Torch.sum bit parity. Generic division and
transcendental operations remain unchanged. The model's feedback dtype and
single BF16 cast remain unchanged.

Because the readout and backend changed, strict four-read verification was
rechecked once in run 66. All 1,048,576 full logits are now bit-exact. Selected
readouts, independent replays, concurrent isolation and zero cleanup also pass.
Real HTTP smoke 68 passes all six labeled outcomes, repeat-answer equality,
prefix reuse and question independence on the final build.

Fresh generation 67 uses the default 48-step limit and converges after seven
refinements. It produces a useful two-sentence customer-service reply. This one
example and the two decision examples establish functional behavior; representative
quality, calibration, serving capacity and deployment work remain open.

Evidence 60 through 68 is under
../inference/decision-model/history/benchmarks/. Earlier failed runs remain
preserved. Trace 53 remains unused. The configuration-derived initialized-state
asset is still required to reproduce pinned RoPE initialization; portable-power
work remains stopped. All device jobs and the temporary HTTP server have ended.
At the user's request, pod llgdc53h3jwg9q was destroyed in shutdown run 69.
RunPod confirmed deletion; the follow-up lookup returned HTTP 404. Compact
evidence, source snapshots, initialized state and the final addon were verified
locally before deletion. The user approved discarding large raw tensor captures
and random draws. The handoff is ../inference/decision-model/PROGRESS.md.
Changes are uncommitted and unpushed.

The final CUDA addon SHA256 is
71a37c40212e8ada26e2285801746dc25de27f11ffdcf444c09151136e04e6fd,
with policy 13 and sum-f32-warp-or-block1024-vector4-v2.

## Implemented APIs

- `AutoRegressive` owns causal compilation, sampled generation, caller-driven
  logits execution, and speculative scheduling. All repository callers migrated.
- `Model.withParameters`, `Model.sameStateGeometry`, and `Model.makeStatePool`
  share parameter preparation and state geometry across the family artifacts.
- `Diffusion.compile` compiles encoder and initial/refinement denoisers
  separately from full and selected readout programs. The artifact owns one
  retained parameter generation and exposes `encode`, `commit`, `evaluate`,
  `score`, `generate`, `inspect`, and `release`.
- `Decision` provides family adapters, ordered selected logits, restricted
  normalization, independent probability averaging, and bounded scheduling.
- `DiffusionGemma.generate` supplies the pinned model policy and device-native
  sampling statistics to the family scheduler. Its example performs generation
  and selected decision scoring through the same compiled artifact.
- `Chat.streamWith` consumes nonempty committed token pages from either family.
  `Chat.stream` retains autoregressive options and custom-logits sampling.
- Removed the interim public `Model.PrefixModel`, `Model.executor`, and
  `DiffusionGemma.referenceDefinition`. The detailed numerical verifier uses
  model graphs and `Model.executeLayers` locally.

## Module placement

The user requires zero new internal modules for this design. Shared parameter
preparation belongs in Model; compilation and state operations belong in Tensor
and Runtime. Family orchestration belongs in its family module. All
DiffusionGemma-specific graphs, reference helpers, and generation policy stay in
`packages/models/src/DiffusionGemma.ts`.

## Agreed runtime contract

Stateful compilation distinguishes append and read-only access independently of
causal or bidirectional attention. Stateless compilation omits state. Immutable
prefix snapshots retain pages; a commit forks an appendable sequence and publishes
a new snapshot after successful encoder execution. Read-only invocations borrow
prefixes and keep canvas K/V in invocation storage. Query validity is independent
of committed token count.

Per-layer descriptors identify heterogeneous K/V storage and retention. Local
prefix retention does not impose a decoder canvas mask. Explicit positions remain
absolute, even after local-prefix eviction.

Refinement feedback uses the required model-declared `predictionDtype`; full
readout logits remain F32. DiffusionGemma temperature-scales F32 logits and casts
feedback once to its embedding dtype. Self-conditioning retains the reference
F32-softmax-then-probability-cast order.

## Validation status

The grouped-expert operation passes CPU, Metal, and CUDA component and
integration gates. Full generation replay passes in run 65 and prepared
four-read logits are bit-exact in run 66. The final HTTP consumer passes smoke
68. Work is paused; the devbox was destroyed as requested after saving compact
evidence. Large raw captures were discarded with the user's approval.

### CPU and Metal

- The combined `pnpm test` run passes 1,325 core tests and the backend package
  suites. Typecheck, lint, and native package verification pass. The sibling
  consumer passes all 142 tokenizer-enabled tests and typecheck. Logs and source
  hashes are in
  `../inference/decision-model/history/benchmarks/20260920-grouped-host-integration/`.
- `DiffusionGemmaInference.test.ts` passes eight compiled F32/BF16 cases with
  optimization enabled and disabled. It compares every decoder logit and encoder
  K/V across two blocks and three denoising steps per block, selected output
  order, immutable prefixes, and physical sharing after commits. It also runs
  the public `DiffusionGemma.generate` API with recorded random inputs and checks
  exact token/page replay for whole-block and exact-length output limits. Actual
  carried feedback also passes the strict numerical checks. Multi-tile Metal
  F32 GEMM now uses compensated accumulation to bound closed-loop projection
  error; fixtures and tolerances are unchanged.
- Eight nonzero self-conditioning component cases pass at the pinned tolerances.
- The family, migrated model ownership, semantic-prefix, and layer execution
  suites pass 68 cases, including interruption during final result cleanup.
- The generic driver and native model sampling suites pass 36 cases. Chat passes
  34 cases, including real multi-block scheduling, backpressure, and cancellation.
  Eager feedback and prefix replacement use protected `Scope.use` finalizers;
  three regressions cover cancellation while those releases are suspended.
  Chat emits all committed-page deltas before requesting the next page, so a
  subsequent generation failure preserves already committed text. Its error
  regression failed before this ordering fix and passes afterward.
- Snapshot tests cover heterogeneous geometry, retention of zero/one local rows,
  absolute positions beyond physical capacity, full-capacity read-only canvases,
  BF16 scalar opmath, independent outputs, and copy-on-write forks.
- The preserved per-layer reference verifier passes on each backend with 150
  stage checks, 12 results, zero mismatches, and zero final external bytes.
- Official CPU and Metal debug builds and generated declarations pass. CPU native
  tests pass 220 cases. Metal native tests pass 271 cases with five ignored.
- Workspace lint, typecheck, native package verification, native declaration
  checks, Rust formatting, and whitespace checks pass after the geometry fixes.
  Native reports and raw logs are preserved in
  `../inference/decision-model/history/benchmarks/20260920-rfc0027-native/`.

### Decision application

The sibling application passes typecheck, 142 tokenizer-enabled tests, and 14
Python tests. Its adapter uses compiled immutable prefixes and selected readouts.
Semantic noise keys exclude routing IDs and sibling questions; full-scaffold
answer-code verification preserves up to 255 choices in caller order. Every
independent read is normalized before probability averaging. Core decision
arithmetic, adapter, scheduling, and cleanup tests pass all 37 CPU/Metal cases.

The checkpoint-file verifier distinguishes header validation from full payload
SHA256 verification in its receipts. Application evidence is recorded under
`../inference/decision-model/.cache/rfc0027-consumer/`.

### CUDA and full checkpoint

CUDA validation on the actual GPU passes:

- 77 native tests, including physical sharing, cancellation rollback, borrowed
  prefix release, read-only concurrency, and exported-allocation accounting.
- 96 direct NVRTC paged-attention cases covering dtype, retention, padding, GQA,
  bidirectional visibility, and exact byte/replay checks.
- Five backend ownership tests and 131 core regression tests, including all
  72 autoregressive/DFlash inference cases.
- Five snapshot cases and four compiled tiny-model F32/BF16 reference replays
  with optimization on/off, including public generation.

The CUDA adapter retains independent native weight handles for inference
artifacts, so clearing prepared source handles preserves AR/DFlash execution.
Its external-memory diagnostic now deduplicates allocations owned by live
exported tensor slots, including retained aliases. That counter excludes
unexported compiled storage, K/V, and workspace; reports record static program
diagnostics and physical GPU samples for those categories.

On 2026-09-20 the user explicitly authorized `./scripts/cuda-devbox.sh create`.
Pod `llgdc53h3jwg9q` is running one RTX PRO 6000 96 GB and remains billed.
The repository bootstrap passed. All 11 checkpoint shards and six assets match
the frozen SHA256 manifest, totaling 51,680,000,442 bytes. The pinned Python
reference environment is ready.

The saved full-checkpoint oracle contains four independent reads with width-16
canvases. `packages/examples/scripts/diffusion-gemma/verify-full.ts` compiles those recorded widths exactly. Normal
generation uses the checkpoint default width of 256. Full-checkpoint compiled
parity, multi-block generation, and workload measurements are running. The
verifier writes partial evidence during loading and compilation, separate
isolated/replay/selected/concurrent timings, and physical prefix sharing.

The first compiled full-model attempt loaded 691 unique BF16 parameters totaling
50,501,973,624 bytes in 26.915 seconds, with peak physical GPU usage of 48,765 MiB.
Compilation then exposed an invalid retention-versus-capacity restriction. The
correction preserves the semantic window and rejects actual capacity exhaustion
at append. A new cross-backend test uses the full 16-query-head, local 8x256 and
global 2x512 geometry. It also exposed a Metal head-size guard before the
stepwise path. Both corrections pass the full-geometry test on CPU, Metal, and
CUDA. The failed run remains recorded
under `../inference/decision-model/history/benchmarks/20260920-effect-full-four-read-02/`.

The next compiled attempt loaded parameters in 27.076 seconds and compiled all
eight entry points in 6.475 seconds. Prefill remained incomplete after 438.744
seconds, so the owned verifier process was interrupted and released its GPU
allocations. Investigation found that CUDA attention used one warp per sequence
and recomputed QK for every four output dimensions.

Attention now runs across query/head/sequence warps and computes QK once per key
into planned invocation-owned score/probability storage. Current K/V storage is
a preceding kernel. Reduction order and rounding remain unchanged. All 96
kernel cases, 77 native tests, and 81 focused CUDA core tests pass after the
change. Four full-geometry cases are byte-identical to both the old kernel and
an independent reference. With 64 query rows, 16 heads, and 64 prefix rows,
BF16 attention took 1.824–2.283 ms instead of 9,421–43,864 ms. These are attention
measurements. Full-model attempt 04 still did not finish prefill after 581.109
seconds. It loaded parameters in 26.983 seconds and compiled in 7.289 seconds;
observed physical GPU peak was 48,989 MiB. The verifier exited with code 130
after graceful interruption, and GPU allocations returned to zero. Its evidence
is in `../inference/decision-model/history/benchmarks/20260920-effect-full-four-read-04/`.
Attempt 05 used opt-in command tracing and identified `et_index` scatter as the
remaining bottleneck. The final trace has 4,101 matched command pairs and no
unfinished command. Index commands took 176,490.876 ms across 58 calls, compared
with 108.099 ms in 38 expert linears and 29.922 ms in 19 K/V attention commands.
The slowest scatter took 9,298.259 ms for BF16 output shaped `[64, 8, 2816]`
along axis 1. The scatter correction visits only the matching axis slice,
preserving source order and per-addition BF16 rounding. All 49 direct dtype,
axis, duplicate, partial-shape, and error cases pass. Full-width BF16 cases
match both the old kernel and an independent reference byte-for-byte. Isolated
`[64, 8, 2816]` scatter dropped from 9,305.786 ms to 0.070679 ms; width 2817 with
duplicate indices dropped from 9,291.992 ms to 0.072709 ms. Full NVRTC checks
and 260 CUDA core tests pass after the fix, along with all 77 native and nine
backend tests. The rebuilt addon is ready for the untraced full-model retry.
Trace timings include per-command stream
synchronization and are diagnostic, not workload benchmark results. The
pre-interruption snapshot showed 175.931 seconds of incomplete prefill. The
process exited with code 130; observed physical peak was 48,957 MiB,
and cleanup returned GPU usage to zero.
Its evidence is in
`../inference/decision-model/history/benchmarks/20260920-effect-full-four-read-05-trace/`.

Untraced attempt 06 completed full-model prefill in 1,879.600 ms, after loading
in 27,101.813 ms and compiling in 7,370.774 ms. It then failed in the verifier
while converting unrelated I64 oracle routing tensors to JavaScript numbers.
The verifier now loads only `logits.answer`, the floating-point row used by
its comparison. All four actual oracle files pass that loader check. Numerical
thresholds are unchanged.

Attempt 07 completed all four reads, their exact replays, selected readouts, and
concurrent reads. Every selected/full comparison was exact across all 26 labels,
and concurrent results exactly matched isolated results. The oracle comparison
failed on every read: maximum absolute logit errors were 3.284, 3.485, 3.213, and
3.867, with relative L2 errors of 0.147, 0.169, 0.157, and 0.182. About 253,000
of 262,144 logits per read exceeded the existing one-BF16-step criterion.

The run measured 1,870.675 ms prefill and 243–247 ms isolated reads, but these
incorrect outputs do not pass the acceptance gate. Observed physical peak was
49,245 MiB; exported allocation accounting and physical GPU usage returned to
zero after cleanup. Evidence is in
`../inference/decision-model/history/benchmarks/20260920-effect-full-four-read-07/`.
Generation acceptance is paused while compiled-boundary probes locate the
first numerical divergence. Probe 08 matches the saved embedding and initial
conditioning exactly across all 2,816 elements. Decoder layer 0 is the first
observed difference: 1,203 elements match exactly, maximum absolute error is
0.25, relative L2 error is 0.00474823, and 901 elements exceed one BF16 step.
The final answer SHA exactly matches attempt 07, confirming that the diagnostic
roots preserve the failing computation. All 34 raw boundary rows are retained
in `../inference/decision-model/history/benchmarks/20260920-effect-full-boundary-probe-08/`.
Native and physical GPU allocations returned to zero.

Pinned Python capture 10 records layer-0 submodule inputs and outputs, encoder
prefix K/V, eager attention Q/K/V, rounded scores, probabilities, context, and
RoPE buffers. It preserves raw dtypes and records original shapes and strides.
Its answer matches the archived oracle exactly, its prefix is unchanged, and
its recomputed softmax matches the captured probabilities exactly. Both local
and global inverse-frequency buffers are F32. The 108,493,248-byte tensor file
and metadata are retained locally and remotely in
`../inference/decision-model/history/benchmarks/20260920-full-reference-layer0-probe-10/result/`.

The identical-input native attention replay matches 65,304 of 65,536 context
values exactly. At answer row 6, 4,095 of 4,096 values are exact; one value
differs by one BF16 step, with that row's QK scores and probabilities exact.
Feeding official rounded scores through native softmax reproduces all 75,264
probabilities exactly. Feeding official probabilities through native PV gives
nine one-step differences across 65,536 values. No production kernel changed.
Reports and raw replay outputs are archived under
`../inference/decision-model/history/benchmarks/20260920-rfc0027-native/cuda-layer0-attention-replay-10/`.

Component probe 11 resets operations to official capture-10 inputs. Decoder
Q/K/V/O projections and local RoPE match exactly. Decoder projected Q/K are
exact and V differs in one element; norm differences stay within one BF16 step.
Router probabilities differ across all 2,048 values, with relative L2 error
0.006486. Routed experts supplied with official indices and weights differ in
1,170 of 45,056 values, including 578 beyond the existing criterion. Decoder
dense MLP is exact, while encoder output projection and dense MLP show larger
errors. These results direct the next probe to router scaling/projection and
shape-dependent GEMM accumulation. Evidence is in
`../inference/decision-model/history/benchmarks/20260920-effect-layer0-components-11/`.
Cleanup returned native and physical GPU usage to zero.

Probe 12 isolates a native GEMM mismatch. Decoder router normalization and
scaling are exact across 45,056 values, but multiplying the official scaled
input by the router weight matches only 1,512 of 2,048 scores exactly, with
36 beyond the existing BF16 criterion. Native softmax on the official scores
differs by at most 1.86e-8. Encoder dense-MLP gate/up projections, GELU, and
product are exact; its down projection on the same official input has 33,504
values beyond the criterion. All decoder dense-MLP stages are exact.

Native replay 14 reproduced the encoder down/output-projection errors with
a 1 MiB cuBLAS workspace. Changing only workspace to 32 MiB made both
projections byte-exact across 782,848 values, and encoder router scores became
exact across 35,584 values. A subsequent enum audit found that the replay used
math mode 4/default rather than the production disallow-reduced-precision flag,
whose value is 16. These results establish workspace dependence for the tested
modes, not parity under the actual production configuration. The original raw
evidence remains intact.

The actual Effect call log in run 15 confirms math mode 16 and a 1 MiB
workspace. Corrected replay 16 reproduces all observed projection errors
byte-for-byte. Two settings matter independently: default math fixes the decoder
router, and default math plus 32 MiB workspace makes all eight captured
projections exact. Increasing workspace while retaining mode 16 leaves errors.
The pinned PyTorch call uses default math and a 32 MiB workspace. API, compute
type, and algorithm-only substitutions did not resolve the mismatch.

The native correction uses default math for BF16 outputs and plans 32 MiB
workspace, while retaining the stricter reduction mode for F32 bias accumulators.
BF16 split-K partials may round before the final output. Actual-addon probe 18
now matches all captured whole-row Q/K/V/O projections, all 12 split router/MLP
projections, and whole dense MLP outputs exactly. A deterministic dense
regression fails on the old settings and passes on the correction; all 78 native
tests, all 260 focused CUDA core tests, nine backend tests, and ten capability
fingerprint tests pass.

Two gaps remain before another strict full run. Routed experts still differ on
official inputs, including 578 decoder and 176,646 encoder values beyond the
criterion. Also, encoder projections run in 64-row chunks differ from the
oracle's 278-row projection because cuBLAS reduction choices depend on row
geometry. Whole-row K/V are nearly exact; chunked K/V have thousands of values
beyond the criterion under default math.

Official expert capture 19 exactly reproduces capture 10's accumulated encoder
and decoder outputs. Probe 20 reproduces all 282 expert group projections
exactly with grouped `Tensor.linearRows`. The custom `expertLinearRows` encoder
gate/up projection differs beyond the criterion in 111,627 values; its encoder
down and decoder gate/up/down projections stay within the criterion. GELU,
product, weighting, and ordered accumulation are exact across all 141 groups.
Replay 22 distinguishes intermediate BF16 rounding from F32 reduction order.
Default BF16 cuBLAS matches all ten captured expert projections exactly; the
unchanged custom kernel matches all ten probe-20 outputs exactly. For 28 of 96
examined encoder gate/up values, the official result lies outside a conservative
bound for any F32 reduction order followed by one BF16 rounding. Independent
exact-rational verification of the saved BF16 inputs and weights confirms all
28 counterexamples. This proves a contract difference without identifying the
internal split partition. The existing `expertLinearRows` F32-accumulation
contract stays unchanged.

The approved implementation adds `Tensor.groupedExpertLinearRows`. It preserves
input order within each exact-size expert group, applies ordinary backend
matmul semantics, and restores output row order. DiffusionGemma selects this
operation explicitly through the `gatedExperts` projection builder and uses
route-major input order. Shared graph/compiler/API work passes 128 Rust tests,
122 focused CPU/Metal core tests, and 24 backend tests. Workspace typecheck and
lint pass. All-package native declaration generation/checks and serial host
debug builds pass. Full native suites pass 223 CPU tests and 277 Metal tests,
with five existing Metal tests ignored. The final Metal rebuild includes a
producer-marking correction after the route fence; all 60 grouped/strict-expert
TypeScript tests pass against the current addons. Workspace Cargo check and
Rust formatting pass. Combined workspace and sibling-consumer checks pass,
including 1,325 core tests and 142 consumer tests.
The focused tests exposed a fast-completion cancellation race in all three
adapters: native output could finish before an interrupted Effect caller took
ownership. The adapters now retain cleanup ownership until delivery.

Metal preserves exact group dimensions with one route-control fence per
projection. It reads N U32 indices and uploads N U32 permutation entries in
planned invocation staging; floating-point work stays on Metal. Tests verify
ordinary F32/BF16 matrix equality at M=32/33 and M=128/129 algorithm transitions,
pipeline reuse across runtime totals, and at most 35 warmed pipelines for a
2,224-row plan. Exact group dimensions, bounded scratch, host control
synchronization, and capture restrictions are part of the operation contract.
Evidence and exact-rational verification are archived in
`../inference/decision-model/history/benchmarks/20260920-cuda-expert-rounding-proof-22/`.

CUDA grouped-projection run 24 matches all 282 captured gate/up and down
projections exactly across 9,934,848 BF16 elements. Exported allocations return
to zero. The report retains the known strict-dot mismatches and ordinary-matrix
controls. Four targeted CUDA tests, all 83 native tests including ignored tests,
NVRTC compilation, and 49 direct scatter cases pass. The initial F32 regression
caught explicit FMA differing from ordinary CUDA matmul compiled with
`fmad=false`; the grouped path now uses the same sequential multiply/add
semantics, and the failed log is retained. Actual source/addon hashes accompany
the results. Run 25 matches the whole routed-expert output exactly for encoder
782,848/782,848 values and decoder 45,056/45,056 values. Combined feed-forward
outputs still have 3,938 encoder and 562 decoder values beyond the criterion.
Its trace records eight grouped commands, each transferring 520 control bytes
for 128 experts. Control readback waits take 0.01195-0.02073 ms and total control
handling takes 0.02243-0.048539 ms. All run-25 timings are trace-perturbed.

The broader CUDA run initially passed 603 of 611 tests. Five memory assertions
needed physical allocation accounting: one omitted the fixed 32 MiB GEMM
workspace, and four bounded physical prefix bytes using logical payload size.
The corrected bank test independently measures ordinary GEMM workspace, retains
the original 589,824-byte variable-scratch limit, and checks less than 64 KiB
workspace growth when the bank grows eightfold. Prefix tests derive two
1,024-byte aligned append allocations and cross-check transaction diagnostics.
The ownership and immutable-sharing checks remain.

The final seven-file CUDA gate passes all 78 tests, including those five
assertions, strict and grouped experts, fast-completion cancellation, concurrent
output retention, and strict DiffusionGemma suites. The three training timeouts
resolve in a serialized 40-test run with a 60-second wall-time budget. Together
with the original 603 passing cases, the reruns resolve every failure from the
611-case selection. All 83 native tests, nine CUDA backend tests, and NVRTC/scatter
gates pass. Raw failures and corrected runs are retained. Numerical thresholds
are unchanged; full compiled parity remains open. The native coordinator
returned an idle GPU with zero allocations and no pending jobs to the full-model
owner for the next boundary and normalization probes.

Diagnostic 21 uses explicit prefill width `[278]` and a fresh uninstrumented
baseline. Instrumentation preserves that baseline exactly, but layer 0 still
has 1,037 values beyond the criterion and final logits have relative L2 error
0.167027. Matching prefill geometry alone does not resolve the mismatch. This
is separate from the earlier 64-row chunk evidence. The verifier now accepts
an optional final `prefill-chunks-csv` argument and records explicit/default
geometry; its default remains `[1,16,64]`. Family formulas, pinned PyTorch
settings, and numerical thresholds remain unchanged.

Diagnostic 23 captures actual compiled layer-0 current-canvas Q/K/V and attention
context while preserving diagnostic 21's uninstrumented answer SHA exactly.
Q, K, and V have 1,112, 476, and 544 values beyond the criterion, respectively;
context has 5,577, with relative L2 error 0.001898. It captures current K/V only,
not the encoder prefix. Rows with exact reset-input RMSNorm results also have
exact production Q. Sparse normalized-input differences coincide with Q drift
in the other rows. Answer-row Q is exact, but that row attends to the differing
current rows. This points to the normalization/projection chain as the next
reset-input diagnostic; it does not establish causality or exclude prefix drift.
Raw outputs, source hashes, addon policy strings, and row comparisons are in
`../inference/decision-model/history/benchmarks/20260920-effect-production-attention-23/`.
Native allocations returned to zero. After this run, the native coordinator
received exclusive GPU ownership for grouped-operation validation while the
full-model owner prepared the normalization probes locally.

The prepared `norms` component mode compares official pre-normalization input
through native RMSNorm and Q/K/V projections against projections of official
normalized input, at 14 encoder/decoder boundaries. The optional
`attention-norms` compiled probe exports four current attention tensors and
four encoder/decoder RMSNorm tensors. It requires one exact prompt chunk. Its
local CPU smoke preserves the independent baseline SHA and releases all
outputs. Post-encoder and post-decoder file-collision failures each leave atomic
failed evidence and zero native allocations; a wrong prompt chunk fails before
loading. The first smoke exposed instrumented prefill bypassing the observed
runtime; the failed report is retained and the corrected probe passes. Examples
typecheck and targeted lint/format checks pass.

Post-grouped diagnostic 26 uses prefill `[278]` and preserves its fresh
uninstrumented baseline exactly. Layer 0 remains the first differing boundary:
1,032/2,816 values are exact, 1,031 exceed the criterion, maximum absolute error
is 0.25, and relative L2 is 0.005403. Final logits have maximum absolute error
4.02348 and relative L2 0.155673, with 253,288/262,144 values beyond the criterion.
The untraced baseline prefill/read take 0.540/0.198 seconds. Cleanup returns
native and physical GPU allocation to zero. Raw outputs and source/addon
evidence are archived in
`../inference/decision-model/history/benchmarks/20260920-effect-grouped-boundary-26/`.
Grouped experts have not closed full-model parity.

Probe 27 establishes the decoder normalization/projection causal chain. On
official pre-RMSNorm input, native RMSNorm differs in 13/45,056 BF16 values,
each by one step. Passing that result through native Q/K/V reproduces all three
production-23 files byte-for-byte, including 1,112/476/544 values beyond the
criterion. Resetting only the RMSNorm output to the official tensor makes Q and
K exact and leaves one one-step V difference. Encoder RMSNorm differs in 283
values, each within one BF16 step; its chained Q/K/V have 4,644/2,602/2,928 values
beyond the criterion, versus 7/5/0 after the official normalization reset.
This bounded probe loads about 46 MB of projection weights and cleans up to zero.
The Q/K chain includes head normalization and RoPE after projection, so the
remaining 7/5 encoder differences after reset are not raw-linear regressions.

Actual normalization tap 28 confirms both compiled inputs are exact across all
782,848 encoder and 45,056 decoder values. Its native RMSNorm outputs are
byte-exact with probe 27, including the same 283 encoder and 13 decoder one-step
differences. Current Q/K/V and context match probe 23; the instrumented answer
SHA matches diagnostic 26's new independent baseline exactly. Cleanup returns
native and physical GPU usage to zero. The full-model owner then explicitly
hands the GPU to the native coordinator for a reduction/scale/rounding substage
replay. Raw evidence is in
`../inference/decision-model/history/benchmarks/20260920-effect-normalization-chain-27/`
and
`../inference/decision-model/history/benchmarks/20260920-effect-production-normalization-28/`.

RMSNorm replay 29 reproduces probes 27/28 byte-for-byte with the unchanged
production arithmetic, while the pinned PyTorch module and its staged expression
both reproduce capture 10. Source inspection confirms direct checkpoint weights
without a `+1` offset. The model computes F32 squared mean plus epsilon, multiplies
input by `pow(mean, -0.5)`, multiplies by F32 weight, then casts once. PyTorch 2.10
specializes that power to its CUDA reciprocal-square-root kernel. Native CUDA
uses sequential F32 square accumulation and division by `sqrtf(mean + eps)`,
with NVRTC `fmad=false`.

The first differing substage is the sum of squares, across all 294 captured
rows. A separate host-side check verifies all 827,904 preceding F32 squared
values against PyTorch byte-for-byte. Maximum sum error is 0.016845703125 in
decoder row 7 and 0.052734375 in encoder row 145. Switching only to
multiplication by reciprocal square root still leaves
13 decoder and 281 encoder BF16 differences. Supplying the PyTorch mean makes
all native F32 normalized and weighted values exact, as well as every final
BF16 output. Supplying the PyTorch sum and multiplying by the F32 reciprocal
width also makes both phases exact. The native coordinator is validating a
generic warp/vector reduction tree, reciprocal mean factor, and reciprocal-square-root
multiplication. This fits the existing F32 backend-defined reduction contract
and keeps one final BF16 boundary; no new operation or numeric policy is needed.
Independent verification of the copied raw bytes confirms both reference and
production guards, exact F32 normalized/weighted controls, and exact BF16 output
for both phases. The 66 MiB archive, checksums, and verification report are in
`../inference/decision-model/history/benchmarks/20260920-cuda-rms-substages-29/`.

Candidate 30 uses a generic four-partial F32 warp reduction, reciprocal width
factor, and reciprocal-square-root multiplication. It matches all captured
PyTorch sum/mean/epsilon/inverse-scale values across 294 rows, all 827,904 F32
normalized and weighted values, and all final BF16 values. Run 31 passes 50
independent synthetic cases at widths 1 through 8,193, including 256 and 512.
Three fixed BF16 witnesses fail against the old native implementation, with
raw words `[15825,47989,47891]` versus `[15826,47988,47890]`; the failed log is
retained. The scoped production correction changes F32-opmath RMSNorm only.
LayerNorm, F64 behavior, and workspace planning remain unchanged. Its capability
fingerprint records `rms-f32-warp-four-partials-mean-factor-rsqrt-v1` with policy
revision 10. The native gates pass 200 dtype/view cases, all 86 native tests,
nine backend tests, 136 selected CUDA core tests, and NVRTC/scatter checks.
Typecheck, lint, and formatting pass. CPU instances excluded by the CUDA test
filter are not CUDA skips.

Actual-addon run 32 makes all 14 normalization/QKV-chain boundaries exact. Run
33 makes all 20 RMSNorm outputs, whole Q/K/V/O projections, dense MLPs, and routed
experts exact. Decoder feed-forward is exact across 45,056 values. Encoder
feed-forward has five differences among 782,848 values, four beyond the criterion,
with maximum absolute error 0.001953125. The remaining indices are 370255,
371182, 371430, 600244, and 600461. Global-head RoPE cosine has four differing
values, two beyond the criterion. Chunk-64 projection differences retain their
separate geometry label.

Production replay 34 matches all 827,904 weighted F32 and BF16 values exactly
and reproduces the old normalization output with its control. Ten-launch GPU
event intervals include Python host-submission gaps and exclude planned casts
and materialization. Recorded old/new milliseconds are 0.04753/0.02285 at
16x2816, 0.34589/0.02470 at 278x2816, 0.30636/0.02487 at 256x2816,
0.04542/0.00644 at 4096x256, and 0.08434/0.00839 at 2048x512. The fix adds no
workspace. All jobs finish before the idle-device handback to the full-model
owner.

Compiled diagnostic 35 preserves its fresh baseline SHA exactly. Actual
encoder/decoder pre-RMSNorm inputs and outputs, plus all current Q/K/V values,
match capture 10. Attention context has 65,304/65,536 exact values, 232 differences,
and 132 beyond the criterion. Layer-0 answer hidden still has 831 values beyond
the criterion, and final-logit relative L2 is 0.1578236415. Cleanup returns to zero.
Two independent local comparisons show that the actual context is byte-exact
with archived native attention replay 10 after BF16 expansion and layout
conversion. That closes the context-reproduction guard without exporting prefix
state. Diagnostic 36 therefore traces actual versus official context through
O projection, post-attention RMSNorm, residual, and feed-forward output, keeping
all 16 rows and recording their routes and group sizes. Its first attempt stops
at I64 reference-index readback, cleans up to zero, and remains archived. The
corrected 36b casts captured expert indices to U32 and passes its updated CPU
fixture and failure-cleanup checks.

Run 36b reproduces all 2,816 actual layer-0 answer values from run 35 byte-for-byte.
Resetting only context to the official tensor makes O projection, post-attention
RMSNorm, residual, and all 45,056 decoder feed-forward values exact. Answer-row
context has one difference within the criterion. It leads to 32 differing O
projection values, five beyond the criterion; 29 post-attention norm differences,
five beyond; seven residual differences within the criterion; and 1,528 differing
feed-forward values, 831 beyond. Only two route positions swap experts 54/50
in answer row 6. Every expert group count stays identical, while routing weights
change. The separate 20 native-versus-PyTorch tie-order differences have equal
BF16 scores and identical expert sets, and the official reset still reproduces
feed-forward exactly. The bounded probe loads 1,582,051,586 weight bytes, takes
3.72 seconds, and records a 2,301 MiB sampled peak with zero final allocation.

The GPU then passes to the native coordinator for attention replay 37. An exact
rational audit finds 13 QK and nine PV BF16 differences in the earlier replay;
softmax on official scores is exact across 75,264 values. Answer row 6 has no QK
difference. Its PV witness at token 6, head 9, dimension 15 has exact product sum
0.15283200122283347: native returns 0.15234375 and PyTorch 0.1533203125. Native is
nearest to the exact sum, so the task is matching legal F32 reduction behavior,
not improving exact-sum accuracy or relaxing the contract.

Replays 37/38 reproduce the actual production context with the unchanged cache
kernel. Pinned PyTorch BF16 batched matmul matches all 75,264 QK and 65,536 PV
values from capture 10. Both wheel cuBLAS 120804 and native cuBLAS 120901 match
those results using math mode 16, F32 output, and one BF16 cast. Contiguous and
original strided Q layouts both pass. The 22 saved QK/PV witness intervals lie
within F32 reduction-order error bounds. This establishes a pure-F32 realization
without permission for intermediate BF16 partial rounding. Generic custom
partial orders do not match every value. Raw evidence and independent
verification are in
`../inference/decision-model/history/benchmarks/20260920-cuda-attention-stages-37/`
and the corresponding run-38 directory.

Pipeline 39 matches all QK, probabilities, and 65,536 context values at the
actual key count 294. Padding that count to 304, 320, 512, 528, 544, 560, or
1,024 leaves QK and probabilities exact but changes four PV results. Those
include the answer-row witness responsible for run 36b's layer drift. Independent
verification confirms the saved F32-to-BF16 rounding and every padded result.
Evidence is in
`../inference/decision-model/history/benchmarks/20260920-cuda-attention-stages-39/`.

The production correction therefore derives active matrix dimensions from
validated host prefix metadata while planning maximum storage capacity
separately. On capable GPUs, BF16 stepwise K/V attention uses invocation-owned
Q/K/V gathers and probability/matrix scratch with the existing locked cuBLAS
F32-output path and a declared 32 MiB workspace. It retains explicit score,
scale, probability, and final-output rounding. One lane reuses matrix scratch
across slots, and host metadata supplies dimensions without GPU control readback.
Actual query rows are packed before GEMM and scattered into the declared output
layout afterward. The key matrix includes all logical retained-prefix and valid
current rows, including masked columns. This dynamic-dimension path is explicitly
non-capturable. The capability fingerprint records policy 11 and
`stepwise-bf16-kv-f32-gemm-active-rows-v1`.

The compiled native regression matches every captured context byte at capacities
278, 544, and 1,024. Disabling only the new command branch reproduces all 232 old
differences, including the answer-row witness. Traces record the same active
Q16/P294 and seven numerical submissions per lane at all three capacities,
while combined scratch grows from 39,225,600 to 44,001,024 and 52,617,984 bytes.

Validation passes all 92 native tests, including the device and captured
fixtures; 162 generic attention cases, including zero retention and head widths
through 1,025; NVRTC checks and 49 ScatterAdd cases; nine rebuilt-addon backend
tests; and 136 selected CUDA core tests. Full local/global configurations cover
16Q/8KV/D256 and 16Q/2KV/D512 at Q278/P278 causal and Q256/P534/P790
bidirectional geometry. Windows of none, 128, and 1,024, retention 1,023,
eviction at cursor 1,100, changing Q/P, zero-query reuse, packed rows, concurrency,
retained outputs, cancellation, and failure rollback pass. The final rebuild
follow-up passes nine backend and 22 CUDA snapshot/runtime tests. Local
typecheck, lint, Rust formatting, and CUDA cargo checks pass. Source snapshots,
failed attempts, hashes, logs, and traces are archived under run 39's
`native-fix-evidence/`.

Bounded median host-wall timings include output readback, use three warmed
calls, and compare branches sharing the candidate memory plan. Local D256
Q278/P278, Q256/P534, and Q256/P790 improve from 3.272, 5.153, and 6.630 ms
to 2.072, 2.121, and 2.175 ms. Global D512/KV2 timings improve from 3.846,
7.475, and 10.484 ms to 1.425, 1.433, and 1.503 ms. These are not device-only
intervals or full-model throughput measurements.

Softmax replay 40 retains the legacy production-35 context guard. Independent
raw checks find exact F32 maxima, exponentials, denominators, and unrounded
probabilities across all 16 heads at answer token 6 for official versus legacy
QK inputs. Full-canvas differences are zero of 256 maxima, 12 of 75,264
exponentials, four of 256 denominators, and 1,184 of 75,264 probabilities. Host
BF16 rounding reproduces both uninstrumented controls. Official scores yield
all 75,264 captured probabilities exactly. The first full-canvas difference
is QK; the answer-row difference starts in PV.

After explicit idle-GPU handback, full compiled run 41 uses the final addon
`cf4f36a3f5fbc7ad3801f925b4209841406b1ed5a576074ed13964e2353ef4f0`
and one 278-token prefill chunk with tracing unset. Its fresh baseline and
instrumented answer hashes both equal
`1fd6ec2fee64153afa4bbd4a0fc58aa2da24cb276f4972b8865a7ea475233136`.
Layer-0 encoder/decoder input RMSNorm boundaries, current Q/K/V, all 65,536
context values, and all 2,816 decoder answer-hidden values are exact. The first
differing answer-hidden boundary moves to layer 1: 2,209 of 2,816 values are
exact, 304 exceed the criterion, maximum error is 0.0625, and relative L2 is
0.0016354584191180198. Final logits still have relative L2 0.13071254359391699,
maximum error 2.6055978536605835, and 251,901 values beyond the criterion.
External memory returns to zero, and the post-exit GPU check finds zero MiB
and no compute PIDs. Evidence is in
`../inference/decision-model/history/benchmarks/20260920-effect-attention-fixed-boundary-41/`.

The selected-layer probe preserves default layer-0 behavior byte-for-byte.
Source checks compare every default-layer raw buffer, selected-layer roots,
and baseline outputs. Injected failures after encoder and decoder compute
return allocations to zero and retain the fresh baseline. Fourteen tiny
comparison boundaries, historical run-41 comparisons, typecheck, and lint pass.

Pinned reference capture 42 targets layer 1, preserves the archived answer SHA
and capture-10 inputs/settings, and confirms that encoder/decoder layer-1 inputs
equal capture-10 layer-0 outputs. Reference encoder post-RoPE K/V, stored prefix,
and decoder inherited prefix are byte-identical. Actual probe 43 preserves
run 41's fresh baseline SHA exactly and cleans up to zero, with no GPU compute
PIDs afterward. All 45,056 decoder layer-0 hidden values, layer-1 normalization
inputs/outputs, and current Q/K/V are exact. Decoder layer-1 context has 96
differing values, 14 beyond the criterion, with maximum error 0.00390625.

Actual encoder layer-0 hidden has five differences, four beyond the criterion,
at indices 370255, 371182, 371430, 600244, and 600461. Its entire 782,848-value
F32 buffer is byte-identical to run 33's `encoder.feedForward`. Encoder layer-1
normalization has five differences; Q has 231, K 89, V 218, and context 2,067.
This connects the earlier captured-input feed-forward discrepancy to current
encoder-prefix computation. It does not yet identify routing as the cause.
The run paths are
`../inference/decision-model/history/benchmarks/20260920-full-reference-layer1-probe-42/`
and
`../inference/decision-model/history/benchmarks/20260920-effect-layer1-boundary-43/`.

Bounded replay 44 preserves all 278 encoder rows and reproduces the entire
782,848-value actual F32 encoder-hidden buffer from run 43 byte-for-byte. All
35,584 router logits match capture 10 exactly. Native router probabilities
have 27,389 F32 differences, with maximum error 1.1920929e-7; selected weights
have 1,291 differences, with maximum error 8.94e-8. The unreset feed-forward
output retains five differences, four beyond the criterion.

Resetting only router probabilities to the official values leaves 917 selected
weight differences, with maximum error 5.96e-8, and two feed-forward differences,
both beyond the criterion. Resetting only selected weights restores every
captured expert, post-normalization, combined, and full-hidden output exactly,
including all 782,848 encoder-hidden values. Native probabilities still have
the same 27,389 differences in this branch. All three branches have identical
selected indices, route order, and expert group counts. The separate 137
native-versus-PyTorch route-position permutations preserve expert membership;
raw reference U32 indices are retained for alignment checks.

The run saves 70 stage buffers, including F32 softmax maxima, exponentials,
denominators, selected-weight sums, and division outputs, under
`../inference/decision-model/history/benchmarks/20260920-effect-encoder-routing-chain-44/result/`.
It loads approximately 1.559 GB of weights, takes 8.52 seconds host wall time,
and samples a 2,333 MiB physical peak. Cleanup returns to zero and leaves no
GPU compute PIDs. Production arithmetic is unchanged.

A CPU-only audit replays all 278 native softmax denominators, 35,584
probability divisions, 278 top-8 sums, and 2,224 normalized divisions
bit-for-bit using sequential F32 summation and direct F32 division. Reusing
the identical saved native exponentials, changing only the 128-value reduction
to 32 lanes with four strided sequential values per lane and tree offsets
16/8/4/2/1 makes all 35,584 probabilities exact.

On official selected probabilities, a top-8 halving tree with offsets 4/2/1
and the same F32 division and scale multiplication matches all 2,224 aligned
weights exactly. The old sequential reduction leaves 917 differences. Its
denominator differs on 125 of 278 rows. Row 0 has exact rational sum
355686261/1073741824; the tree returns 0.3312586545944214 versus sequential
0.331258624792099, fixing all eight row-0 weight bit patterns. The inferred
BF16 expert scale is checked against every F32 product and for consistency
across each expert. Reference alignment uses expert IDs; all 137 route-order
permutations have equal probability bits and identical expert membership.
The saved audit script and JSON are `router-44-local-audit.py` and
`router-44-local-audit.json`, archived under run 44's `native-cpu-audit/`.
These captured-input controls require no change to exponentiation or division.

Control 45 replaces only encoder layer-0 hidden state with the reference
layer-1 input from capture 42, independently byte-equal to capture-10 layer-0
output. An extra all-rows output retains the unreset encoder graph and its K/V
writes, preserving the prefix schema and bytes. CPU checks and an injected
post-compute EEXIST failure pass cleanup and baseline-retention guards;
typecheck and lint pass.

The full run preserves the unreset baseline SHA from run 43 and independently
reproduces all 782,848 original encoder-hidden F32 values byte-for-byte. Exactly
one encoder-hidden boundary is replaced. All encoder layer-1 hidden inputs,
normalization, Q/K/V, and context then match capture 42. All decoder layer-1
normalization, current Q/K/V, 65,536 context values, and 45,056 hidden values
also match exactly. Encoder layer-1 hidden still has 22 differences, 12 beyond
the criterion, with maximum error 0.0078125. The reset branch's first differing
answer-hidden boundary is layer 2, with 438 values beyond the criterion. Final
logits retain relative L2 0.12253741864633702, maximum error 2.7728919982910156,
and 251,363 values beyond the criterion. Its separately labelled reset SHA is
`0cba924898475ea147a8d29729186d1cdf0ed51333ff47b4f25785fd91d1a6d4`.

Run 45 takes 44.10 seconds host wall time, with a 49,789 MiB sampled peak.
Cleanup returns to zero. After the process and evidence transfers finish,
the GPU reports zero MiB, zero utilization, and no compute PIDs. Evidence is
in `../inference/decision-model/history/benchmarks/20260920-effect-encoder-hidden-reset-45/`.
Run 44's saved `verify-causal.py` also independently passes the 137 tied-route
alignments, all 2,224 selected-weight alignments by expert ID, raw hashes,
and all branch index/order/group-count checks.

Local archives 41 through 45 contain raw buffers, source, READMEs, checksums,
and smoke/failure evidence. Run 42's `verify-reference.py` verifies source pins,
the unchanged reference answer, layer-input inheritance, and official prefix
inheritance. Run 43 preserves all 14 comparison boundaries and the whole-buffer
run-33 equivalence proof. Run 45's `verify-propagation.py` checks both phases
and all 30 encoder/decoder K/V descriptors, plus the unchanged baseline and
original hidden guard. Its actual encoder layer-1 K/V matches official stored
prefix values. This comparison uses computed K/V, not actual prefix-storage
readback. The 22 remaining encoder layer-1 hidden indices are retained in JSON.

The full-model owner explicitly hands the idle GPU to native for bounded
reduction replay 46. Its old-production control reproduces every run-44 maximum,
exponential, denominator, probability, selection, top-8 total, normalized weight,
and scale multiplication bit-for-bit. Pinned PyTorch softmax matches all 35,584
reference probabilities. Separate PyTorch exponentiation also matches all
35,584 saved native exponentials, supporting the unchanged exponentiation path.

Changing only the 128-value sum makes all probabilities exact and changes
225 of 278 native denominators. Changing only the top-8 sum leaves 978 weight
differences on native probabilities. Changing both reductions matches all
2,224 official weights. Top-8 totals on official selected probabilities match
PyTorch across all 278 rows. Raw checkpoint BF16 expert scales and their
selected provenance are saved and verified against every resulting weight.
The independent exact-rational row-0 witness fails the old sum and passes the
new schedule.

The bounded replay passes 212 generic dyadic dtype/width/axis/materialized-view
cases across widths zero through 65,537, plus special-value and guard-byte
checks. It preserves run-44/run-43 same-input, full-hidden, and official-reset
guards, but does not rerun feed-forward. Pre-edit evidence is saved remotely
at `/root/cuda-routing-reduction-46` and archived locally under
`../inference/decision-model/history/benchmarks/20260920-cuda-routing-reduction-46/`.
Its independent `verify-raw.py` reconstructs all old stages and all four
reduction combinations, with weight-difference counts 1,291/978/917/0. It
verifies checkpoint-scale lookup, expert-ID alignment, and the rational witness.
The saved unfused PyTorch sum over 128 values differs from the warp denominator
on 110 of 278 rows. The 128-value proof therefore establishes official softmax
realization parity, not parity with every PyTorch sum reduction order.

Run 47 integrates the generic reduction in the F32 compute module for Sum.
Each warp owns an output; lanes accumulate reduction-linear positions
`lane + 32*j` in F32 and combine at descending offsets 16/8/4/2/1. All widths
use that schedule, with existing input/output conversions and backend-defined
reduction order. The change adds no scratch, readback, or persistent state.
Pinned-source dispatch and contract evidence is in
`reduction46-evidence/source-audit.md`, alongside exact reproduction commands,
source hashes, and PyTorch source revision
`449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`.

The compiled control disables only the new Sum branch and reproduces every
run-44 maximum, exponential, denominator, probability, TopK/Gather, total,
normalization, and scale multiplication. Its independent witness returns
`0x3ea99abb`, failing the expected `0x3ea99abc`. Restoring the new branch passes
all four compiled tests, including all 35,584 probabilities and 2,224 weights,
native index order, seven dtypes, views, tails through 65,537, more than 65,535
output blocks, concurrency, cancellation, and retained outputs. Local typecheck,
lint, cargo check, and formatting pass.

The first full native gate exceeds the foreground 120-second limit. Its partial
log is preserved, and the owner waits for that process to finish before a
background rerun with permanent remote log/status files. GPU jobs do not overlap.
The completed gates pass 96 native tests, including all 44 normally ignored
device fixtures; all 13 NVRTC modules; 49 ScatterAdd cases; nine rebuilt-addon
backend tests; and 246 CUDA core tests across nine files. Local workspace cargo
check, typecheck, lint, Rust formatting, Python compilation, and 52 host native
tests pass.

The final capability fingerprint records policy 12 and
`sum-f32-warp-strided-descending-v1`. Final addon SHA256 is
`1f7d7be764450f40cd7eccb80b299f0a98623e7dfd9954ed5d829f31a31dc474`.
Final source hashes are verified equal locally and remotely. Run-47 logs,
failed attempts, commands, sources, hashes, and the disabled-old-branch control
are consolidated in the approved temporary `reduction47-evidence/` directory.
All GPU jobs, remote commands, and transfers finish before the final zero-MiB,
zero-utilization, no-compute-PID check and explicit handoff.

All-layer source checks pass F32 and BF16 tiny-model captures across 136
shared stages, including key, shape, and layout agreement, fresh-baseline
equality, and zero cleanup allocation. Encoder/decoder persistence failures,
interruption, and finalizer failure also return allocations to zero. Typecheck
and lint pass. The capture records 34 shared stages per phase/layer, covering
hidden inputs/outputs, normalization, Q/K/V, attention, routing, and feed-forward
boundaries.

Pinned reference run 48 completes both phases across all 30 layers in
17.35 seconds host wall time. It preserves the archived answer SHA and prefix
and writes 1,456,466,400 raw bytes remotely. Native run 49 asserts the final
policy-12 addon and completes a fresh `auto 278` baseline plus the all-layer
trace in 57.26 seconds. The baseline and instrumented results match. Final
answer logits pass the numerical criterion: zero values beyond tolerance,
maximum absolute error 3.814697265625e-6, and relative L2
6.60923615592504e-8. Layer-29 answer hidden state and final normalization are
exact across all 2,816 values.

The complete 2,040-stage comparison finds 1,919 byte-exact stages, 60
equal-score index-order differences, and 60 corresponding weight-order
differences that align exactly by expert ID. All 60 full layer-hidden outputs
are exact. Computed encoder K/V matches the official stored prefix. The only
non-routing difference is decoder layer-11 query at `[0,2,0,29]`: native
-0.01483154296875 versus reference -0.014892578125, one BF16 step. Its context
and hidden output are exact. Full raw data remains remote.

Strict four-read run 50 at prefill 278 passes read 0 with the same SHA as run
49, but fails the other reads:

| Read | Values beyond criterion | Relative L2 |
| --- | ---: | ---: |
| 0 | 0 | 6.60923615592504e-8 |
| 1 | 234,416 | 0.0475012377286 |
| 2 | 244,904 | 0.0806215961 |
| 3 | 247,466 | 0.0919269692 |

Read 1 has maximum absolute error 1.1295567. All four replays are exact, all
26 selected/full values match, both concurrent reads match, and cleanup returns
to zero. Source audit confirms that all four calls are Initial reads with the
same prefix and width-16 canvas geometry; there is no feedback-path difference.
The failed gate and its outputs remain archived. Generation 51 is not run.

The earlier run-33 global-head RoPE lead concerns inverse-frequency index 29,
where native/JS generation yields F32 0.20907999575138092 versus captured
0.20908001065254211. Its four decoder cosine differences occur at that pair
at positions 278 and 293. Diagnostic 52 uses a temporary copy of the unchanged
full verifier, with the same four Initial reads, prefill 278, canvas 16, addon,
and verification gates as run 50. It binds only capture-48's
`model.decoder.rotary_emb.full_attention_inv_freq` through the existing model
parameter mechanism. The buffer hash, dtype, shape, and encoder/decoder equality
are checked. The only changed value is index 29.

Run 52 passes every unchanged gate across all four reads. Read 0's SHA is
unchanged; reads 1 through 3 have zero values beyond tolerance, maximum error
3.814697265625e-6, and relative L2 approximately 6.6e-8. Host wall time is
45.98 seconds. The native addon is unchanged. This is a causal binding control,
not a production construction fix. Its evidence directory is
`../inference/decision-model/history/benchmarks/20260920-effect-full-frequency-reset-52/`.
Paired failing-read trace 53 stays unused and reserved.

The construction-only audit next compares the pinned Torch CPU exponent, power,
and reciprocal intermediates with the existing host expression. Local Node
22.14 reproduces the old value from both `DiffusionGemma.rotaryEmbedding` and
`Tensor.explicitRoPE`: base 1e6 is exactly F32 `0x49742400`, and exponent
58/512 is exactly 0.11328125, bits `0x3de80000`. JavaScript power is
4.782858141653791, narrowed to F32 4.782858371734619, bits `0x40990d2d`.
Its reciprocal is 0.20907999407001587 before narrowing to `0x3e561911`; the
reference buffer contains `0x3e561912`. This reproduces the old construction
and is preserved in `rope52-local-old-constructor.json` in the approved
temporary directory.

Initial construction replay 54 uses pinned Torch 2.10 CPU AVX512 at revision
`449b1768410104d3ed79d3bcfe4ba1d65c7f22c0` and matches all 384 captured
frequency values, including inactive zeros. All active exponents match.
The first difference is power: global index 29 is host `0x40990d2d` versus
Torch `0x40990d2c`. Sliding-attention index 111 has another difference, host
`0x4537eba3` versus Torch `0x4537eba2`, yielding inverse-frequency bits
`0x39b229fa` versus `0x39b229fb`. The global-only reset in run 52 does not
change this local value. Torch reciprocal equals scalar `1 / power` and
tensor-ones division. Source dispatch reaches `Vectorized<float>.pow` and
`Sleef_powf16_u10`; the completed direct-symbol replay below checks this dispatch.

An independent exact-rational certificate compares the relevant power
midpoints raised to denominator q against `base^p`, using p/q 29/256 and
111/128. It proves that both existing host F32 powers are mathematically
nearest. Independent 80- and 160-digit Decimal evaluations agree. The pinned
Torch power errors are -0.5174855314922096 and -0.5137532182993478 ULP; host
errors are +0.4825144685077904 and +0.4862467817006522 ULP. Exact reciprocal
intervals confirm correct rounding for each path's own F32 power input. This
is reference-realization parity, not an exact-power accuracy correction.
The proof files are `rope54-independent-pow-witness.{py,json}` in the approved
temporary directory. Numerical-contract review precedes any construction
proposal; the F32 power boundary and generic division remain intact.

The completed construction proof directly calls the exported pinned
`Sleef_powf16_u10` through a C++ ABI wrapper and reproduces all 192 active
Torch powers byte-for-byte. The generic and AVX512F exports share an address.
All 384 constructor outputs, including 192 inactive zeros, match the capture.
The archived proof includes pinned Torch sources and SLEEF revision
`5a1d179df9cf652951b59010a2d2075372d67f68`, independent rational witnesses,
and 120/180-digit Decimal checks. Evidence is in
`../inference/decision-model/history/benchmarks/20260920-frequency-construction-54/`.

Contract review finds no promise that generic Pow is correctly rounded to the
real-valued result, but also no generic SLEEF accuracy class or ULP allowance.
Cast rounding requirements apply to represented source values. The host
frequency initializer instead produces a `Float32Array` whose bytes become a
`FromBytes` constant through `Tensor.fromTypedArray`. The existing named-buffer
path already accepts that initialized state.

The adopted plan therefore closes arithmetic diagnostics and the proposed
portable-power port. Pinned vector and scalar-tail dispatch, FMA semantics,
and portability would make that port disproportionate to this model-preparation
task. A separate scratch prototype is cancelled without production changes.
Generate both frequency buffers from configuration with the existing pinned
CPU reference constructor, exporting safetensors plus configuration, geometry,
source-version, and content-hash provenance. The target asset has 384 F32
values, or 1,536 payload bytes. Configuration changes require regeneration.

Existing verification and generation preparation load the asset using
Safetensors and bind it through `DiffusionGemma.define`. They acquire and release
the two additional concrete tensors explicitly, including failure and
interruption. The runtime consumes ordinary model state and gains no Torch or
AVX512 dependency. Core/native arithmetic, generic Pow/division, checkpoint
loader validation, and numerical tolerances remain unchanged.

Asset 55 and fresh prepared strict run 56 subsequently pass. Generation 51
uses the same asset and completes all six steps but differs at eight final
tokens. Real consumer smoke 59 passes functional checks and two labeled examples.
The current-status section records the results; no further generation probe
has been started. Trace 53 remains unused. Runs 50, 52 and 54 remain archived
as the unprepared failure, causal binding control and construction proof.

The generation example records native prefill, denoiser, readout, commit, and
sampler phases separately, with raw samples and percentile summaries. Its
profiled tiny replay matches all six argmax canvases and final tokens, and
generation reads back only reduced statistics. A corrupted random-file SHA
fails after successful denoiser/readout execution and leaves an atomic failed
evidence record, no running samples, and a nonzero CLI exit.

The pinned Python oracle reproduced all four saved full-vocabulary answer-row
SHA256 values exactly on the new GPU. Its corrected full-generation recording
passes at width 256: two blocks, six refinement steps, and 512 output tokens.
It records 1,610,612,736 exponential-input bytes for cross-language replay.
Prefill took 1.458 seconds and generation took 10.643 seconds, including recorder
checks and serialization. Observed physical GPU peak was 55,115 MiB. Evidence
is in `../inference/decision-model/history/benchmarks/20260920-full-reference-generation-02/`.
The earlier width-16 recording remains a narrow diagnostic.

The initial sampler generates exponential draws on the host and uploads them.
At canvas 256 and vocabulary 262,144, each step uploads 256 MiB of random inputs.
A local measurement took 2.52 seconds for generation plus 0.36 seconds for
validation, before upload. Returned metrics expose this cost separately from
device processing and reduced-statistics readback. Full logits stay on device.

Numerical parity does not establish decision quality or calibration. Those
application evaluation gates remain distinct from this implementation.

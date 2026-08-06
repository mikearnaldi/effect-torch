# RFC 0016: Frozen-Program Memory — Arena Allocation and Structural Fusion

- **Status**: Draft
- **Created**: 2026-08-05
- **Depends on**: RFC 0007 (fusion), RFC 0008 (compilation), RFC 0012
  (dtypes), RFC 0015 (native backend)
- **Updates**: —

## Summary

Four optimizations to the compiled training step, ordered by
leverage-to-risk: (1) **buffer planning** — compile-time liveness on
the frozen program, intermediates suballocated from one arena buffer
instead of pooled per-op allocations that survive to `synchronize()`;
(2) **chunked head+CE** — a freeze-time graph rewrite that never
materializes the `[rows, vocab]` logits tensor, recomputing it per
chunk in backward; (3) **gemm epilogue/prologue fusion** — gelu into
the fc gemm, residual into the proj gemm; (4) **one-launch optimizer**
— AdamW over flat param/grad/moment arenas. The first two are the
difference between a 360M model fitting on this machine or not; all
four compound.

A standing design principle is recorded here because these phases rely
on it: **compilation is shape-specialized, permanently**. Frozen
programs bake shapes and strides into generated kernels as integer
literals; dynamic extents are handled by compiling one program per
shape signature (as generation already does: prefill/decode/batched).
This is the industry norm (XLA, torch.compile, TensorRT, MLX all
specialize per shape) and is what makes liveness, tile selection, and
constant-folded index arithmetic possible at all.

## Motivation

Measured on the 30M FineWeb model (batch 64, block 256, mixedBf16),
one training step allocates **5.7 GB cumulatively** while the true
concurrent live-set is far smaller: within a step every allocation is
fresh — dead intermediates are retired but only rejoin the pool at
`synchronize()`. On top of that, power-of-two pool bucketing rounds
the 3.29 GB logits buffer to 4 GB (−21% on one tensor). The batch-128
OOM that prompted this RFC (34 GB peak, command-buffer killed) turned
out to be silent f32 promotion (fixed separately), but the
allocation-trace instrumentation showed the structural waste clearly.

The concrete target: a 360M model at block 256 currently needs ~17 GB
at batch 16 and does not fit at batch 32. With arena planning and
chunked CE the same configuration projects to **~12 GB** (params+state
5.8 GB, activation arena ~5 GB, chunked logits ~0.4 GB), making batch
32 plausible — and batch 16 for 124M at long context comfortable.

Fusion philosophy, stated once: the win is not "compile the whole
graph" generically (XLA's bet) nor a single megakernel (MoK's bet,
CUDA-only hardware: SM partitioning, TMA, grid sync). It is applying
full-graph reasoning to the few tensors that dominate memory — for
us, the head-gemm→CE chain — with every step behind a measurement
gate. Cursor's MoK validates the scoped approach: hand-fusing the one
layer that mattered beat the generic systems 2.37×.

## Phase 1 — Buffer planning (arena allocation) — **DONE**

**Mechanism.** At `freezeProgram` time the schedule is fixed and every
intermediate's shape/dtype is known. Walk the schedule, compute live
intervals (first def → last use), assign 256B-aligned byte offsets
into a single arena `MTLBuffer`; the step then allocates one buffer
per program, once, and replays it every step. `set_buffer` already
takes byte offsets and the dtype-size offset discipline landed with
bf16, so kernels are address-agnostic.

**Rules.**

- *Escapes*: program inputs/outputs (params, optimizer state, loss,
  next params/state roots) are never arena-managed; `freezeProgram`'s
  root list already defines the boundary.
- *Hazards*: the memoryBarrier-per-dispatch model serializes all
  dispatches in the command buffer, so liveness-dead reuse is safe by
  construction. If barriers are ever relaxed for intra-buffer
  parallelism, the analysis must become anti-dependency-aware — out of
  scope here.
- *Scope*: frozen programs only. The pool allocator stays for the
  uncompiled step, generation, and one-off ops. Two allocation modes,
  chosen per program, not per tensor.
- *Alignment*: 256B suballocation granularity; arena size rounded up
  once.

**Gate.** Peak wired-memory delta at batch 128 drops measurably; step
time does not regress; loss curve **bit-identical** (same kernels,
same order, different addresses — this phase carries zero numerical
risk).

## Phase 2 — Chunked head + cross-entropy — **DONE**

**The observation.** `logsumexp` is per-row and chunks are rows: the
`[rows, 50257]` logits tensor has no cross-row dependency except the
final `sum(nll)/count`.

**As built** (differs from the original freeze-time sketch, simpler):
the rewrite happens at **graph construction** in `cross_entropy`
(`chunked_head_ce` in lib.rs), not at freeze. When the logits node is a
`Linear` head and `rows × vocab ≥ 2^28` (env-overridable:
`EFFECT_TORCH_CE_CHUNK_MIN`, `EFFECT_TORCH_CE_CHUNK_SIZE`):

- rows are split into `clamp(numel / 2^26, 2, 64)` chunks;
- per chunk: `Slice → Linear → CrossEntropy(Sum) → Checkpoint → Cast f32`;
  the Checkpoint makes backward recompute the chunk logits (one extra
  head gemm per chunk) instead of retaining them — peak logits memory
  is one chunk, forward and backward;
- the loss is `Σ chunk_sums / active_count` in f32, where the active
  count is an exact integer reduction over the targets — reproducing
  the Mean reduction without per-chunk normalization error.
- A new `CeReduction::{Mean, Sum}` on the CE nodes carries this: Sum
  never divides by the active count and never errors on an all-ignored
  chunk (a padding-heavy batch can produce one); Mean keeps the exact
  previous semantics, including the zero-active error. The fused Metal
  kernels take a runtime `mean` flag (no pipeline-key split); the CPU
  and Metal composed fallbacks mirror it.
- Validation runs through the plain node first, so construction-time
  error messages are identical whether or not the rewrite fires.
- If the same logits node is consumed elsewhere too (sampling + loss),
  the rewrite still fires for the CE path; the other consumer keeps the
  full logits alive as before — correct, just not smaller.

**Held-out evaluation** must build forward + CE as one lazy graph for
the rewrite to fire: `model.execute` materializes logits, so applying
`crossEntropy` to its result keeps the full `[rows, vocab]` tensor
alive (this was the eval-time memory spike — a plain forward was using
more than training). `heldOutLoss` now compiles `(params, input,
target) → loss` via `Tensor.compile`; its plan shares the training
arena's root.

**Two latent bugs fixed along the way.** (a) The fused-multi kernel
execution check reserved a phantom scalar-pack slot when `scalars` was
empty, erroring at exactly 31 real buffer arguments — the batch-256
loss combine hit it. (b) Pool bucketing rounded every allocation to a
power of two, so a 1.1GB activation pinned a 2GB buffer; large blocks
(≥64MB) now round to 64MB pages.

**Measured** (30M preset, block 1024, mixedBf16, M4 Max 48GB):
batch 128 step **2.86s chunked vs 3.06s unchunked** (faster — the
6.6GB f32 logits and their CE internals never leave the L2/fabric
budget), full-run peak ~13GB vs ~19.8GB; batch 192 trains at 4.26s/step;
**batch 256 trains end-to-end in a 25.6GB cap** (5.7s/step, val
included) — impossible before (logits alone were 13.2GB bf16). Loss
curves match within bf16 reassociation (val 9.4791 chunked vs 9.4804
unchunked from identical init).

**Original sketch for the record.** A freeze-time `ChunkedLinearCE`
node with a hand-written backward; the graph-construction rewrite plus
the existing Checkpoint machinery (deep-copy recompute) subsumes it
with no new node kind and automatic CPU parity.

## Phase 3 — Gemm epilogue/prologue fusion

**DONE (as built).** `gelu` became a real node (`NodeKind::Gelu`,
emitted by `Tensor.gelu`, with a composite grad rule and a fusion
`Expr::Gelu`/`GeluTanh` so it still folds into elementwise regions).
An eval-time pass in `fuse_roots` (`gemm_epilogue_pass`, before the
elementwise fold) rewrites two Metal-only patterns:

- **gelu into the fc gemm epilogue** — `Gelu(Linear)` becomes
  `LinearGelu`. With a backward graph present the pre-activation has
  further consumers, so the dual-store variant writes both the
  pre-activation (output 0, read through `FusedPick`) and the gelu
  output from one gemm launch; at inference the pre-activation buffer
  disappears entirely. The accumulator applies gelu in f32 before the
  bf16 store — strictly more accurate than the unfused path, which
  rounds the pre-activation to bf16 first.
- **residual add into the proj gemm epilogue** — `Add(Linear, r)`
  with the Linear single-consumer and an exactly output-shaped `r`
  becomes `LinearResidual`; the standalone proj output never
  materializes. Safe because Linear backward reads `x`/`w` and the
  routed grad, never its own output.

Both epilogues ride the existing bias-gemm kernel (`gemm_fused`,
`Epilogue` joins the pipeline-cache key). CPU keeps composed fallbacks.
`EFFECT_TORCH_NO_EPILOGUE` isolates the pass for A/B measurement.

**Not done:** the layernorm-backward audit — LN already has dedicated
fused forward/backward kernels; the reduce-epilogue gap stays open.

**Gate results (30M, block 1024, mixedBf16).** Step time is
noise-level either way: batch 32 A/B 0.83 vs 0.85 s/step (within
run-to-run variance), batch 128 fused 2.8 s/step ≈ the 2.86 baseline.
The win is structural, as predicted: 18 fewer kernel launches per
6-layer stack (`LinearResidual×12`, `LinearGelu×6`, `FusedPick×12` in
the compiled step graph), the transient proj-output buffers gone, and
the f32-accumulator gelu. Parity: 5 new cargo tests (fused vs unfused
forward + grads, dual-store structure, shared-linear guard, gelu-grad
finite differences), 81/81 cargo, 628/628 vitest.

**Original sketch for the record.** "Removes two [B, T, 4E]
materializations per layer (pre-gelu, post-gelu)" — not achievable as
stated: gelu backward needs the pre-activation and proj-gemm backward
needs the post-gelu, so both persist in training. The dual-store gemm
removes the separate gelu pass (one read of the pre-activation) and
its launch instead.

## Phase 4 — One-launch optimizer

**DEFERRED (measured, not worth it now).** Per-kind encode profiling
(`EFFECT_TORCH_KIND_TIMING`) at batch 32: AdamW self-time is
~1.5 ms/step (12.1 ms over 8 steps × 308 evals). Even a perfect
one-launch optimizer saves <0.5% of step time at our scale — the
flat-arena param/moment storage redesign does not pay for itself.
Revisit at 124M+ if optimizer share grows.

What the same profile actually exposed — **fixed instead**: every
`Gather`/`ScatterAdd` read its index tensor back to the host
(`to_u32_vec`), synchronizing the command queue mid-step: the wte
embedding-grad scatter alone cost ~530 ms/step of host-blocked time at
batch 32 (the GPU drained, then idled through the host wakeup, id
conversion, re-upload, and the remaining encode). Indices now stay on
the device end to end (`metal_ids_u32`: contiguous + cast to u32 on
Metal; host indices upload once via `ids_from_host`). Batch 32:
0.84 → 0.70 s/step (-16%). Batch 128: 2.85 ≈ 2.86 s/step — neutral,
the queue there is deep enough that the sync hid behind GPU work.

Removing that sync exposed **two latent executor bugs**, both fixed:

- *Retired-buffer accumulation.* Retired uploads and swept dead pool
  buckets were dropped only at `synchronize()`; with no mid-step sync
  the driver footprint grew unbounded inside a step (52 GB
  driver-allocated vs 19 GB rust-live) and batch 256 died with
  `kIOGPUCommandBufferCallbackErrorOutOfMemory`. Retired blocks now
  ride the command buffer being committed when they were retired and
  drop as buffers complete (`reap_completed`), and a memory
  backpressure (`memory_budget` = min(env cap, ½ × recommended working
  set), checked against the driver's `currentAllocatedSize`) waits on
  the oldest in-flight buffer before fresh allocations — the host may
  no longer outrun the GPU by more than the budget. Command buffers
  also commit early once they reference 4 GiB of distinct pool memory
  (`cb_track` in `set_buffer`), since an uncommitted buffer pins
  everything it references.
- *Cross-command-buffer execution overlap.* Metal may execute
  consecutive command buffers concurrently; the pool recycles buffers
  across them, and with dense byte-budgeted commits that overlap
  became a real corruption source (NaN losses at batch 128+). Command
  buffers are now serialized GPU-side with a shared event (wait at
  start, signal at end) — no host stall.

Verified: batch 256 bare (no cap) trains + validates end to end at
5.8 s/step with correct loss curves; batch 32 stays at 0.70 s/step.

**Gate.** Step time; checkpoint round-trip unchanged.

## Phase 5 — CUDA stance (recorded, not scheduled)

If XLA-grade compilation is ever wanted on CUDA, the path is an
existing stack (Triton or per-shape StableHLO via PJRT), never a
homegrown LLVM pipeline. Per-shape specialization does not block this
— PJRT executables are shape-specialized too. An XLA Metal backend
does not exist and is ruled out permanently; MLX is Apple's answer
there. The megakernel direction (MoK) is CUDA-only hardware and only
pays at MoE/distributed scale we do not have.

## Risks

- **Arena correctness is the phase that can silently corrupt**:
  liveness bugs write into live memory. Mitigation: the bit-exactness
  gate is a perfect detector (any reuse bug perturbs the loss curve),
  plus a debug mode that poisons dead intervals.
- **Chunked CE changes FP accumulation order** in `dW_head`: not
  bit-exact. Accepted, gated on loss-curve equivalence; chunk order is
  fixed so runs remain deterministic.
- **Two allocation modes** (pool + arena) is permanent complexity:
  the pool must remain for non-frozen paths. Contained by making the
  arena strictly a frozen-program property.
- **Epilogue creep**: every epilogue variant multiplies gemm pipeline
  cache keys. Mitigated by the existing dtype/shape-keyed cache
  discipline; epilogue kind joins the key.

## Verification

Standard gates after every phase: `cargo test` (74), `pnpm vitest run`
in `packages/core` (628, modulo the documented intermittent flakes),
warning-free `cargo build`, `pnpm typecheck`, nano-gpt end-to-end at
the 1.5s/400-step guardrail, fineweb step-time and peak-memory
measurements at batch 64/128/256. Phase-specific gates as listed
above. Native builds for perf comparisons are always
`napi build --release`. All heavy runs under
`EFFECT_TORCH_MEMORY_CAP_MB`.

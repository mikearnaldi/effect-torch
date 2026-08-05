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

## Phase 1 — Buffer planning (arena allocation)

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

## Phase 2 — Chunked head + cross-entropy

**The observation.** `logsumexp` is per-row and chunks are rows: the
`[rows, 50257]` logits tensor has no cross-row dependency except the
final `sum(nll)/count`. Split rows into chunks; per chunk, in forward:
`gemm(hidden_chunk, W_head) + b → CE → nll_chunk`, keeping nothing but
the nll and the hidden chunk (already live). In backward, recompute
the logits chunk (rematerialization — gradient checkpointing applied
to exactly one tensor), CE-backward it, and accumulate
`dW_head += dlogits_chunkᵀ @ hidden_chunk`,
`dhidden_chunk = dlogits_chunk @ W_headᵀ` in a fixed chunk order
(deterministic).

**Cost model.** The head gemm runs 1.5× (forward + recompute). At
batch 128 that is +0.42 TFLOP against a ~6 TFLOP step — **~7% step
time for −6.6 GB peak** (logits and dlogits both leave the live set).

**Architecture.** A freeze-time rewrite, not a new public op:
pattern-match `Linear(x, W, b) → CrossEntropy(target)` in the traced
graph and replace with a `ChunkedLinearCE` node carrying a
hand-written backward (`dhidden`, `dW`, `db`). The Model/Loss API is
untouched — the model still logically returns logits. The rewrite is
threshold-gated on `rows × vocab` so small models keep the simple
path, and verified against the unfused path on a fixed seed. A
forward-only variant serves held-out evaluation and inference without
recompute cost.

**Gate.** Loss curve equivalent to the unfused path within bf16
tolerance (dW accumulation order changes, so not bit-exact); peak
memory drops by the predicted amount; step-time regression ≤ the
cost-model 7%.

## Phase 3 — Gemm epilogue/prologue fusion

Extend the existing bias-epilogue machinery:

- **gelu into the fc gemm epilogue** — removes two `[B, T, 4E]`
  materializations per layer (pre-gelu, post-gelu).
- **residual add into the proj gemm epilogue** — the epilogue grows
  from bias-vector to full C-matrix add.
- **layernorm backward chains** — audit what the fusion engine already
  catches; `run_reduce` takes a prologue `Expr`, so
  prologue-into-reduce exists; close the reduce-epilogue gap.

Modest individual gains that shrink the live-set the arena then
compacts — multiplicative with Phase 1.

**Gate.** Step time at batch 64 and 128; allocation-trace intermediate
count.

## Phase 4 — One-launch optimizer

AdamW is pointwise given `(p, g, m, v, t)`. If params, grads, and both
moment buffers are each flat arenas, the entire optimizer step is one
elementwise kernel over the flat space — no per-tensor dispatch table,
no grid sync. The grad gather is free *because of Phase 1*: the arena
assignment can place grad slots directly into the grad arena. Side
benefit: checkpoints become three contiguous blobs.

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

Standard gates after every phase: `cargo test` (65), `pnpm vitest run`
in `packages/core` (620, modulo the three documented environmental
flakes), warning-free `cargo build`, `pnpm typecheck`, nano-gpt
end-to-end at the 2.3–2.6s/400-step guardrail, fineweb step-time and
peak-memory measurements at batch 64/128. Phase-specific gates as
listed above. Native builds for perf comparisons are always
`napi build --release`.

# RFC 0015 Phase 0: Backend Op Surface Inventory

- **Status**: Phases 1-3 landed. candle and ug are fully deleted —
  dependency tree, source, and fork pins. 65/65 rust tests, 601/601
  vitest on the first-party CPU+Metal backend. Phase 4 = CUDA.
- **Source**: `NodeKind` in `packages/native/src/lib.rs:808` (85 variants)
- **Date**: 2026-08-04 (updated after the Value flip + candle deletion)

## Landed

- `runtime/`: dtype, layout, cpu/ (tensor, ops, reduce, matmul,
  indexing, linalg, conv, composed, pool, random), metal/ (device,
  emit, run, gemm, kernels, indexing, conv).
- **CPU: every eval arm computes natively**, including kv attention
  (first-party mutable pool slabs).
- **Metal: every eval arm computes natively** — first-party IR→MSL
  emitter (SSA form, no bracket-depth limit), dtype-generic gemm
  (f32/f16/bf16, bias epilogue), flash fwd/bwd, CE fwd/bwd,
  LayerNorm fwd/bwd, rotary, paged scatter/decode + native Metal
  pool slabs, indexing, cast, creation, seeded random, conv family,
  fusion runner (elementwise + reduce) on the native emitter.
- **The Value flip (phase 3)**: `val.rs` (`Val = Cpu | Metal`) is
  the eval/napi value type end to end — Evaluator cache, slots,
  side tables, Leaf, NativeTensor, program bindings. `dev.rs` and
  `err.rs` replace candle Device/Error; `safetensors.rs` is a native
  implementation. `bridge.rs` and `metal_eval.rs` deleted.
- **candle-core, candle-metal-kernels, and the cuda/cudnn/mkl/nccl
  feature flags deleted from Cargo.toml**; `cargo tree` is
  candle-free.

## Hard-won correctness notes (do not regress)

- **Cache-key discipline**: every constant baked into a generated
  kernel must be in the pipeline-cache key (fill n, arange n, cast n,
  cat outdim, gather/scatter/index-select shapes AND strides,
  argreduce/cumsum kept shapes, strided_copy full shape+strides).
- **MetalTensor layout offsets**: `contig`/`to_f32_vec`/`to_u32_vec`/
  `to_u8_vec` must honor nonzero offsets (strided_copy short-circuit
  requires `offset == 0`, not just `is_contiguous`).
- **CPU `contiguous()`** short-circuit requires buffer length ==
  numel (a leading narrow of a larger buffer is contiguous-looking).
- **CPU index_select** writes row-major over the output shape, not
  kept-major.
- **erf** Horner coefficients (no extra `t` factor); MSL `round` is
  half-away-from-zero (JS semantics), `rint` is banker's.
- **conv padding bounds** compare against input extents, not padded.
- **conv_transpose** weight convention is `[c_in, c_out/groups, ...]`
  (matches the conv2d-backward call site).
- **GPU lifetime**: pooled buffers recycle freely (serial encoder
  orders writes after reads), but deallocation is deferred to the
  retire list and released at `synchronize()` — freeing an in-flight
  buffer corrupts nondeterministically.
- **Execution ordering**: untracked hazards mean Metal may overlap
  adjacent dispatches in one command buffer AND concurrent command
  buffers — a buffer-scope `memoryBarrier` follows every dispatch and
  a command buffer covers up to 4096 dispatches.
- **No host fences on the hot path**: `MetalTensor::zeros` is
  alloc + async fill; readbacks synchronize themselves; fusion
  batches scalar readbacks behind one fence.
- **Emitter**: SSA temporaries (see `emit_expr_ssa`) — nested-parens
  emission breaks at MSL's 256 bracket depth on long fused chains.

## Remaining

- Phase 4 — CUDA (later).

Every op the evaluator can dispatch, its current dispatch path, and
what the native backend must provide. Layout notes cover only what the
evaluator actually produces (view ops emit arbitrary strided layouts;
semantic kernels pre-flatten). The per-section "Today:" lines are the
phase-0 snapshot (pre-deletion) and are kept for reference.

## File map (post-deletion)

- `src/`: `lib.rs` (graph + evaluator + dispatch arms), `val.rs`
  (eval value type), `dev.rs`, `err.rs`, `fusion.rs` (fusion IR +
  CPU interpreter + Metal runner glue), `safetensors.rs`,
  `tokenizer.rs`.
- `src/runtime/cpu/`: the CPU backend (tensor, ops, reduce, matmul,
  indexing, linalg, conv, composed, pool, random).
- `src/runtime/metal/`: the Metal backend — `device` (allocator,
  encoder manager, pipeline cache), `emit` (IR→MSL emitter), `run`
  (MetalTensor + fused runners), `kernels`/`indexing`/`conv`/`gemm`
  (primitive kernels), `ops` (evaluator dispatch helpers), `composed`
  (composite fallbacks), `flash`/`loss`/`layer_norm`/`rotary`/`paged`/
  `linear` (semantic fused kernels).

## 1. No kernel needed (graph/metadata)

`Input`, `ScalarInput`, `StopGradient`, `Checkpoint`,
`Reshape`, `Permute`, `BroadcastTo` (views; strides only),
`Slice` (view when possible, else a strided copy kernel),
`FusedPick` (selects one output of a fused region).

Backend requirement: correct `Layout` arithmetic + `contiguous()`.

## 2. Creation

`Zeros`, `Ones`, `Full`, `Randn`, `Uniform`, `Arange`, `Eye`,
`FromBytes` (host upload), `Const` cache.

Backend: fill kernels (Metal), host writes (CPU), seeded RNG
(xoroshiro128+, both devices, deterministic per seed).

## 3. Elementwise binary/unary (fusion-eligible)

`Add`, `Sub`, `Mul`, `Div`, `Maximum`, `Minimum`, `Pow`, `Where`,
`Eq`, `Gt`, `Lt`, `Ge`, `Le` (comparisons → u8),
`Neg`, `Abs`, `Sqrt`, `Exp`, `Log`, `Sin`, `Cos`, `Tanh`, `Relu`,
`Erf`, `Floor`, `Ceil`, `Round`, `Sign`, `Cast`.


- Native: fusion IR → MSL emitter (Metal), IR interpreter (CPU —
  already first-party). Unfused singletons lower to the same emitter
  (a 1-op region), so there is exactly one elementwise code path.
- Dtypes: f32/f64/f16/bf16 + u8/u32/i64 where tier-legal; broadcast
  via stride-0; arbitrary strided inputs on CPU.

## 4. Reduce

`Sum`, `Mean`, `Max`, `Min`, `Prod`, `Argmax`, `Argmin`, `Cumsum`.

- Today: candle reduce kernels / FusedReduce (ug) for fused chains.
- Native: IR → MSL reduce emitter (threadgroup tree reduce, the
  FusedReduce pattern); CPU strided loops. Argmax/Argmin needed for
  sampling (RFC 0014) regardless.

## 5. Matmul / Linear

`Matmul`, `Linear` (semantic, bias epilogue), linalg `Inverse`,
`Det`, `Solve`.

- Metal: first-party tiled simdgroup gemm (MLX tile-selection as
  reference) + the existing gemm.rs bias epilogue. linalg: not needed
  on Metal (CPU-only, as today).
- CPU: Accelerate cblas for f32; loop nests other dtypes; linalg via
  small LU (only used by linalg combinators/tests).

## 6. Indexing / data movement

`Gather`, `IndexSelect`, `ScatterAdd`, `Concat`, `Slice`-copy,
`contiguous()` (strided copy), `copy2d` (block copies).

- Metal: gather/scatter_add/index_select kernels incl. the u8 variants
  (int8 kv — currently fork-only additions); concat as per-segment
  copy2d; strided copy kernel for contiguous().
- CPU: strided loops.

## 7. Optimizer steps

`AdamWStep/Out`, `AdamWStepGroup/GroupOut` (opt-in), `SgdStep/Out`.

- Today: fusion.rs expression tables → ug. Native: same tables → the
  first-party emitter. No new kernels.

## 8. Semantic kernels (already first-party, re-point only)

`CrossEntropy(+Backward)` (loss.rs), `Sdpa(+Backward/+Out)` (flash.rs),
`KvAttention` + kv scatter (paged.rs), `RotaryEmbedding(+Backward)`
(rotary.rs), `LayerNorm(+Backward/+Out)` (layer_norm.rs),
`PositionEmbedding` (wpe add — elementwise/fused).

These hold raw Metal objects today; the work is re-binding them to the
new device wrapper (encoder, allocator, pipeline cache). CPU paths are
composed and ride on §3–§6.

## 9. Conv (composed, no kernels)

`Conv1d/2d`, `ConvTranspose1d/2d`, `Conv1d/2dBackwardW` — composed via
im2col + matmul at the graph layer today. Needs: im2col/col2im as
strided copies (§6) + matmul (§5). No conv kernels, matching the
"no inherited generality" rule.

## 10. Fusion nodes

`FusedElementwise`, `FusedElementwiseMulti`, `FusedReduce`.


- Native: IR → first-party MSL emitter (unchanged IR, unchanged CPU
  interpreter). The emitter is the only genuinely new compiler piece:
  expression ops ≈ 30, vectorized 128-bit loads, f16/bf16 lanes,
  threadgroup reduce — est. 500-800 lines + tests against the
  interpreter (property: same IR, same result, every dtype).

## Kernel-family count

| Area | Families |
|---|---|
| Elementwise emitter | 1 (parametric over ~30 expr ops × dtypes) |
| Reduce emitter | 1 (parametric) + argmin/max variants |
| Gemm (+bias epilogue) | 1-2 |
| Indexing/data movement | ~6 |
| Creation/fill/random | ~4 |
| Semantic (already ours) | 7 modules, re-point only |
| **Total new Metal work** | **~15 kernel families + the emitter** |

## Device-layer checklist (Metal, phase 2)

- Allocator: pow2 buckets, rotating 8-probe reuse, 4096 cap,
  rate-limited sweep, shared intermediates default.
- Encoder: one compute encoder per command buffer, Serial dispatch
  type, no hazard tracking/fences/barriers; retire at N dispatches.
- One device-global `synchronize`; readbacks map shared buffers
  directly, blit for private.
- Runtime shader compile + pipeline cache (as flash.rs/paged.rs do
  today).
- Edge cases carried from the candle audit: MTLCopyAllDevices
  enumeration, simulator NULL arch guard, u64 seed buffer, CPU
  readback race discipline (blit + wait on the submitting buffer).

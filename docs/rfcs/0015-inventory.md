# RFC 0015 Phase 0: Backend Op Surface Inventory

- **Status**: Phase 1 (CPU) complete except KvAttention; Phase 2 (Metal) not started
- **Source**: `NodeKind` in `packages/native/src/lib.rs:808` (85 variants)
- **Date**: 2026-08-03 (updated after phase 1)

Phase 1 landed: `runtime/` (dtype, layout, cpu, ops, reduce, matmul,
indexing, random, linalg, conv, composed) + `bridge.rs`; every CPU
eval arm computes natively. **KvAttention CPU stays composed via
candle deliberately**: the kv pool slabs are shared cross-device
storage (the Metal paged kernels read them as candle Metal buffers),
so the pool flips with the Metal device work in phase 2/3, not here.
Optimizer steps: fused path (fusion interpreter, native) covers
f32/f64; f16/bf16 CPU uses native composed twins.

Every op the evaluator can dispatch, its current dispatch path, and
what the native backend must provide. Layout notes cover only what the
evaluator actually produces (view ops emit arbitrary strided layouts;
semantic kernels pre-flatten).

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

- Today: unfused → candle binary/unary kernels; fused → ug via fusion.rs.
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

- Today: IR → ug → MSL / CPU interpreter.
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

# RFC 0007: True Kernel Fusion

- **Status**: Implemented (phases 1–2; phase 3 recorded, not scheduled)

**Implementation notes (updated)**: the elementwise op set covers
`Add/Sub/Mul/Div/Min/Max/Neg/Sqrt/Exp/Log/Sin/Cos/Tanh/Abs/Erf/Floor/Ceil/Round`
plus constant-exponent `Pow` (small exponents lower to multiplies/sqrt),
`sign` (lowered to `(x > 0) ? 1 : (x < 0) ? -1 : 0`, matching candle on
NaN), identity casts, and `where(cmp, a, b)`: a comparison with a single
consumer lowers to a float mask feeding a true `select` ternary (an
arithmetic mask would propagate NaN from the unselected side — klDiv
relies on masking `log(0)`). `Log/Tanh/Abs/Floor/Ceil/Round/Pow`, the six
comparisons, and `Select` required extending `ug` itself — `ug`,
`ug-metal`, and `ug-cuda` are patched to the `mikearnaldi/ug` fork (`Erf`
stays an Abramowitz–Stegun expansion; Metal has no `erf`). Region inputs
need not match the output shape: any broadcast-compatible tensor becomes
a lane read through stride-0 dims (bias adds, softmax max-subtracts and
computed scalars fuse instead of materializing), with per-lane strides
baked into the Metal kernel (and keying its pipeline cache) and an
odometer walk in the CPU interpreter. Uniform constructors fold to
constants at any broadcast-compatible shape; a region that folds to zero
lanes evaluates plainly. Regions are capped at 30 input lanes (Metal
allows 31 buffer arguments per kernel; one slot is the output) — overflow
materializes the region and starts a new one. A multi-output post-pass
merges a materialized shared prefix with its fused continuations into one
kernel (the prefix's expression is inlined into each continuation;
unfused consumers keep it materialized as an extra output) — kernel
signatures are fixed at compile time, so the merge runs on the finished
rewrite where the full consumer set is known, and repeats to a fixpoint.
A broadcast-smaller prefix is only ever inlined, never emitted (its
materialized value would be computed at the wrong shape). CUDA fusion is
disabled until the `ug-cuda` path can be tested on real hardware.
`EFFECT_TORCH_NO_FUSION` disables the whole pass,
`EFFECT_TORCH_NO_MULTI_FUSION` only the multi-output merge.
- **Author**: Michael Arnaldi
- **Date**: 2026-07-28
- **Depends on**: RFC 0004 (optimizers — the fused update nodes this replaces
  under the hood), RFC 0006 (roadmap; this item moves from "slot-in" to
  committed work)

## Summary

Candle launches one GPU kernel per operation and materializes every
intermediate. Our fused AdamW/SGD nodes fuse at the *graph* level (one
node, ~10 candle kernel launches inside the eval arm), which removes
graph size and dedup overhead but not the real cost: intermediate device
buffers and per-kernel launch/memory traffic. True fusion compiles an
elementwise operation chain into a single GPU kernel.

The infrastructure already exists in our candle fork's dependency tree:
`candle-ug` is an SSA kernel DSL with code generators for Metal
(`ug-metal`, runtime-compiled via `new_library_with_source`) and CUDA
(`ug-cuda`, via NVRTC), wired into `MetalDevice::compile` and the CUDA
device behind the `ug` feature flag. There is no CPU codegen — the CPU
path is a small interpreter over our own expression IR in the eval arm.

## Design

### Expression IR

A small scalar expression tree in the native crate, defined over named
input lanes and f64 constants:

```
Expr = Input(index) | Const(f64)
     | Add(Expr, Expr) | Mul(Expr, Expr) | Sub | Div
     | Sqrt | Exp | Log | Tanh | Erf | Pow(Expr, Expr)
     | Neg | Abs | Max | Min | Cmp*(Expr, Expr) | Select
```

The IR is structurally hashable; the hash keys a compiled-kernel cache
(one compilation per distinct fused expression per device per dtype).

### FusedElementwise node

A `NodeKind::FusedElementwise { inputs: Vec<NodeRef>, expr: Expr, ... }`.
Phase-1 constraints: all inputs share one shape (scalars arrive as
constants baked into the IR); the output shape is that shape. Eval:

- **Metal / CUDA**: lower the IR to a `ug` SSA kernel, compile on first
  use (cached by IR hash + dtype), launch over the flattened contiguous
  input buffers into a fresh output buffer.
- **CPU**: interpret the IR per element over the input slices — one
  pass, no intermediates, bitwise-identical to the composed sequence.

### Phase 1: optimizer updates as real kernels

The bounded, immediately valuable target: replace the candle-op
sequences inside the `AdamWStep` and `SgdStep` eval arms with a single
fused kernel per backend (three outputs for AdamW — param, m, v — as
three launches of the same compiled pipeline or a multi-output kernel,
decided in implementation). Acceptance is the existing parity suite:
fused and composed trajectories identical, plus the optimizer tests
unchanged. This builds and proves the IR, the lowering, the cache, and
the CPU interpreter without needing graph surgery.

### Phase 2: elementwise region fusion (the general mechanism)

A rewrite pass (same architecture as the vmap rewrite) that folds
maximal chains of elementwise nodes into `FusedElementwise` regions,
delimited by reductions, shape-changing ops, indexing, and multi-consumer
nodes. Autodiff over a fused region: the adjoint of an elementwise
expression DAG is itself an elementwise expression DAG (chain rule over
the IR, sharing common subexpressions), so `backward` emits another
`FusedElementwise` node — no per-op adjoint graph blow-up, and no
forward intermediates retained for backward beyond the region inputs
(this is also the memory win for activation-function chains).

### Phase 3 (recorded, not scheduled): reduction fusion

Fused softmax/layernorm-style kernels (elementwise + reduction in one
kernel) need different machinery (tile/shared-memory codegen) and are
out of scope here. Matmul-adjacent fusion (XLA territory) is a
non-goal.

## Numerics

GPU kernels evaluate the same scalar op sequence as the composed graph
per element. With fast-math disabled and precise div/sqrt, expect
bitwise or last-ulp equality with the unfused path; CPU interpretation
is bitwise-identical by construction. Acceptance: optimizer parity
tests (exact), fused-op tests at 1e-9 f64 / 1e-6 f32 tolerance.

## Build changes

Enable the `ug` feature (plus its `metal`/`cuda` sub-features) on the
candle-core dependency in the fork; the native crate links
`candle-ug`/`ug-metal` directly for codegen. Runtime shader compilation
requires the Metal compiler at runtime (always present on macOS) and
NVRTC on CUDA hosts (already a candle CUDA requirement).

## Failure modes and fallbacks

If kernel compilation fails at runtime (driver limits, exotic dtypes),
the eval arm falls back to the existing composed candle-op sequence —
fusion is an optimization over identical semantics, never a hard
dependency. The fallback path is tested by forcing the cache to miss.

## Non-goals

- Automatic discovery of fusion regions in user graphs (phase 2 is a
  defined pass, not a heuristic search).
- Fusing across reductions or matmuls (phase 3 / never).
- Exposing the IR in the TypeScript API — fusion is an implementation
  detail of the native backend.

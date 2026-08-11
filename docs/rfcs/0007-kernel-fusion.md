# RFC 0007: True Kernel Fusion

- **Status**: Implemented (phases 1–3a and multi-output through RFC 0021;
  phase 3b recorded, 2026-08-11)
- **Author**: Michael Arnaldi
- **Date**: 2026-07-28
- **Depends on**: RFC 0004 (optimizers — the fused update nodes this replaces
  under the hood), RFC 0006 (roadmap; this item moves from "slot-in" to
  committed work)

**As-built implementation notes (updated 2026-08-11)**: phases 1, 2, and 3a,
including multi-output fusion, are implemented as RFC 0021 code-generation
regions over one `GraphIndex`. Elementwise, reduction, optimizer, GEMM-epilogue,
and multi-output choices live in `OptimizationPlan` side tables. They are not
semantic `FusedElementwise`, `FusedElementwiseMulti`, `FusedPick`, or
`FusedReduce` `NodeKind`s, and fusion does not rebuild the graph or run a
whole-graph rewrite fixpoint.

The current scalar IR is `KernelExpr`. Its operation set covers
`Add/Sub/Mul/Div/Min/Max/Neg/Sqrt/Exp/Log/Sin/Cos/Tanh/Abs/Erf/Floor/Ceil/Round`,
constant-exponent `Pow`, `sign`, identity casts, comparisons, and true `select`
for `where`. Region inputs may be any broadcast-compatible shape through
stride-0 lanes. Uniform constructors fold to constants; regions are capped at
30 input lanes and the Metal 31-buffer limit.

Multi-output selection uses an indexed region dependency DAG and bounded
worklist. It inlines a shared prefix into continuations, retains a materialized
prefix output for external consumers when legal, and creates a split
duplicated-expression region when a continuation lane has transitive prefix
ancestry. A broadcast-smaller prefix is inlined rather than emitted at the
wrong shape. `EFFECT_TORCH_NO_FUSION` disables region fusion and
`EFFECT_TORCH_NO_MULTI_FUSION` disables multi-output selection. CUDA fusion
remains disabled until it can be tested on hardware.

CPU and Metal lower selected regions to typed authoritative executable
instructions. Required Metal pipelines compile during executable compilation;
`execute` has no runtime kernel compilation or composed fallback. Compilation
fails if a selected instruction cannot be prepared. `optimize: false` disables
optional regions but uses the same typed compiler, memory planner, and executor
as the optimized path.

## Original Summary and Motivation (Historical)

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

## Original Design (Historical)

This section preserves the 2026-07-28 proposal and its original `Expr` and
semantic fused-node names. The as-built `KernelExpr` and side-table region
architecture is recorded above and in RFC 0021.

### Expression IR (original proposal)

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

### FusedElementwise node (original proposal)

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

### Phase 3: reduction fusion

Split into a general mechanism (3a) and optional specializations (3b).

**Phase 3a: fused-reduce regions (the general mechanism).** Reductions
(`Sum`/`Mean`/`Max`/`Min`) stop being hard barriers and become region
*terminators*: a region is an elementwise chain with an optional single
trailing reduce over fixed dims. `sum(exp(x - max(x)))` compiles to one
kernel — one thread per output element running a `Range` loop over the
reduce extent, evaluating the elementwise expression per step through the
existing broadcast-lane stride machinery, folding into an accumulator
(ug SSA `DefineAcc`/`Range`; CPU interpreter gets a nested loop).
Consumers of the reduced result read it as a broadcast lane (phase-2
machinery unchanged), so shared elementwise prefixes are *recomputed*
through lanes rather than materialized: softmax forward runs three
kernels with zero full-size intermediates, and the same holds for any
`Reduce(elementwise-chain)` pattern — `sum(x*y)`, variance, logsumexp,
the `sum(g*y)` adjoints in backward graphs. Autodiff needs no new
machinery: the adjoint of `reduce(f(lanes))` is an elementwise
expression per lane (`g * f'_i(lanes)` for sum/mean; a masked select for
max/min), i.e. an ordinary fused region over the broadcast gradient.
A cooperative `ReduceLocal` lowering (threadgroup-per-row tree reduce)
is a follow-up optimization for the few-rows/large-extent shape; the
scalar loop is the correct-by-construction baseline. Once the region
reduce covers all dims/keepdims combinations, routing *all* reductions
through it retires the candle Metal reduce path (and its rank>4
non-trailing-dim bug class). Numerics: tree/sequential fold order
differs from candle's reduce, so parity tests use tolerances rather than
bitwise equality.

**Phase 3b (recorded, not scheduled): single-kernel specializations.**
Softmax/layernorm in *one* kernel needs multi-stage tile codegen
(reduce, barrier, recompute from registers/shared, reduce again, write —
the flash-attention staging). The ug SSA vocabulary (`ReduceLocal`,
`DefineLocal`, `Barrier`, `Range`) already expresses it; what does not
exist is an optimizer that *chooses* the staging. ug's own
`LazyBuffer`/`Schedule` was evaluated and rejected as a shortcut: its
launch heuristic ignores reduce dims (explicit TODO upstream), the
cooperative path has no strided loop for extents beyond the block width
(large-vocab softmax degrades to O(V²) per row), and its multi-consumer
dedup rule materializes exactly the shared subtrees single-kernel
fusion must inline. If profiling after 3a shows launch overhead
dominating, handwritten `Softmax`/`LayerNorm` kernels (written directly
against the SSA, the `SgdStep` precedent: semantic op in TS, execution
strategy native) are the targeted fix — and the same staging is the core
loop of any future flash-attention kernel.

Matmul-adjacent fusion (XLA territory) remains a non-goal.

## Numerics (Historical Proposal)

GPU kernels evaluate the same scalar op sequence as the composed graph
per element. With fast-math disabled and precise div/sqrt, expect
bitwise or last-ulp equality with the unfused path; CPU interpretation
is bitwise-identical by construction. Acceptance: optimizer parity
tests (exact), fused-op tests at 1e-9 f64 / 1e-6 f32 tolerance.

## Build Changes (Historical Proposal)

Enable the `ug` feature (plus its `metal`/`cuda` sub-features) on the
candle-core dependency in the fork; the native crate links
`candle-ug`/`ug-metal` directly for codegen. Runtime shader compilation
requires the Metal compiler at runtime (always present on macOS) and
NVRTC on CUDA hosts (already a candle CUDA requirement).

## Failure Behavior (Current)

Fusion and pipeline preparation happen during executable compilation. A
pipeline or lowering failure is a compilation error. `execute` follows the
already prepared typed instruction and physical-ID plans and has no runtime
compile, graph evaluator, or composed-operation fallback. `optimize: false` is
the supported correctness baseline and uses the same compiler/executor path.

## Non-goals

- Automatic discovery of fusion regions in user graphs (phase 2 is a
  defined pass, not a heuristic search).
- Fusing *across* reductions (multi-stage tile codegen — phase 3b is the
  recorded exception) or matmuls (never).
- Exposing the IR in the TypeScript API — fusion is an implementation
  detail of the native backend.

## Historical Addendum: Semantic-Node Kernels and the Walk Pipeline

This addendum records the measured pre-RFC 0019/RFC 0021 evaluator
implementation. Its optimization motivation and kernel results remain useful;
its per-walk fusion cache and evaluator mechanics are superseded by executable
compilation and RFC 0021 side-table regions.

Two lessons fell out of measuring the fused path end to end
(EFFECT_TORCH_KIND_TIMING / FUSION_TIMING / WALK_TIMING, all kept as
env-gated instrumentation):

- **Fusion must be memoized.** fuse_roots is a pure function of the
  immutable graph but ran per walk at ~5µs/node — a 2.4ms tax that
  made fusion a net regression. It is now cached by root node ids
  (bounded LRU). Compile/freeze paths always fused once and were
  unaffected.
- **Synchronize once per walk.** Per-root device syncs serialized CPU
  encoding against GPU execution (a 209-root step walked in 32ms);
  the walk now syncs at the end (host readback syncs itself). The
  same walk: 11.6ms.

Beyond fused regions, the biggest eval-time costs were composed
implementations of semantic ops with synchronous host readbacks or
long op chains. These are now single-kernel execution strategies
behind semantic nodes, CPU fallbacks intact:

- **CrossEntropy** (loss.rs): one-pass forward (online logsumexp +
  nll + status flags) and backward (probs − one_hot, device-side
  count). The label/count error semantics require host reads, which
  would split the walk's pipeline — they are deferred through the
  evaluator's ce_checks to the walk's final sync. 1.1ms → 12µs.
- **RotaryEmbedding** (rotary.rs): angles in-register, one kernel for
  forward and the RotaryEmbeddingBackward node (transpose rotation =
  negated angles).
- **LayerNorm** (layer_norm.rs): semantic node (single/multi-dim
  trailing normalization), one launch forward, one backward (dx + x̂;
  dw/db are plain reduces). Grad is hand-derived like other semantic
  nodes.

Combined effect on the reference GPT training step (4 layers, E 128,
T 64, compiled trainer): 53ms → 12.5ms per step. Optimizer step
scalars (lr, bias corrections) are also memoized per walk instead of
copied per parameter.

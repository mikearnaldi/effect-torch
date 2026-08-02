# RFC 0012: Dtypes — A Strict, Honest Type System for Tensor Element Types

- **Status**: Implemented
- **Created**: 2026-08-02
- **Depends on**: RFC 0007 (kernel fusion), RFC 0008 (compilation), RFC 0010 (inference)
- **Updates**: —

## Summary

Dtype support becomes a designed system instead of an incidental one.
Three tiers of element types — **compute dtypes** (f32, f64, f16,
bf16), **integer dtypes** (i64, u32, u8 — ids, targets, masks), and
**storage dtypes** (fp8 variants, int4, mxfp4 — declared later,
alongside the kernels that consume them) — governed by two rules:

1. **Strictness**: no implicit promotion, ever. Mixed-dtype operations
   fail with a typed error naming both dtypes and the remedy (`cast`).
   This already holds for binary ops; this RFC codifies and extends it.
2. **Honesty**: support is a per-(op, dtype, device) matrix, enforced
   at the earliest possible boundary with typed errors. Nothing
   silently downcasts, emulates, or succeeds-until-compute.

The KV cache's `kvDtype` (f16/bf16 pool slabs) falls out as the first
application.

## Motivation

Today the library exposes six dtypes (`f32, f64, i64, u8, u32, f16`)
but supports them by accident: the compute path assumes f32 almost
everywhere (flash kernel, `kv_attention`, RoPE tables), bf16 is missing
from the enums entirely, and the failure modes are uneven — clean
typed guards in some ops (`scaledDotProductAttention`,
`crossEntropy`), cryptic backend errors in others (`Metal contiguous
to_dtype F32 F64 not implemented`), and silent acceptance at the device
boundary (an f64 tensor can be *created* on Metal and only fails at
compute time). The audit behind this RFC (every core op × every dtype
× both devices, run empirically) established the real matrix below.

Dtypes are also where the industry's sharpest footguns live: silent
f64→f32 truncation (JAX's x64 flag), promotion-to-widest ruining f16
performance (PyTorch eager), quantization formats presented as if they
were arithmetic types. The library's whole posture — explicit,
structural, fail at construction — points at the strict design.

## Definitions (the taxonomy)

Every floating dtype is (sign, exponent, mantissa); exponent buys
range, mantissa buys precision.

- **Compute dtypes** — ALUs do arithmetic on the bits directly:
  - `f32` (1/8/23): the reference; everything works everywhere.
  - `f64` (1/11/52): CPU only — Metal's shading language has no
    doubles (hard platform limit). Role: reference numerics,
    gradient checks, golden values.
  - `f16` (1/5/10): fine grid, small range (max 65504). Metal-native.
    Same thing as "fp16" — spelling differs by ecosystem (PyTorch
    `fp16`, Rust/candle `f16`).
  - `bf16` (1/8/7): f32's range, coarse grid; the training-friendly
    half. Currently missing from our enums — added by this RFC.
- **Integer dtypes** — `i64` (targets, positions), `u32` (token ids),
  `u8` (masks, bytes). No arithmetic beyond comparisons, gathers,
  casts; `add` on token ids stays an error.
- **Storage dtypes** — fp8 variants (`e4m3` precision-leaning for
  weights/activations, `e5m2` range-leaning for gradients), int4,
  microscaled block formats (mxfp4: 32 elements share one e8m0
  power-of-two scale; nvfp4: e4m3 scale per 16 + tensor f32 scale).
  These are **compression schemes, not arithmetic types**: no generic
  op accepts them. They exist only (a) as pool slab bytes consumed by
  a paged attention kernel that quantizes on scatter and dequantizes
  on read, or (b) as payload inside a `QuantizedLinear`-style node
  that presents a compute dtype externally. They are declared when
  those kernels exist — adding them earlier would be a lie. **Not in
  this RFC's implementation scope.**

### Storage vs compute, precisely

A storage dtype's lifecycle is `quantize once → store/move cheap →
dequantize inside the consuming kernel (never materialized)`. The
value is memory bandwidth: inference is bandwidth-bound, so shrinking
bytes-in-flight is the speedup. A compute dtype needs no such
boundary — Metal ALUs do f16/bf16 arithmetic natively. The distinction
decides where each format may appear in the AST: storage dtypes are
quarantined behind bespoke kernels; compute dtypes flow through the
general graph subject to the support matrix.

## Rules

### 1. Strictness: no implicit promotion

Binary/variadic ops require matching input dtypes; a mismatch fails
with `TensorError` naming op and both dtypes (`add: dtype mismatch,
got f32 and f16, use cast for explicit conversion` — current behavior,
codified). Widening or narrowing happens only via explicit `cast`,
visible in the AST.

What this is not: **op-internal accumulation** is an implementation
detail, not promotion. A softmax over f16 inputs may accumulate in f32
registers and return f16 — the graph's dtype semantics don't change,
exactly as PyTorch kernels accumulate f16 in f32 invisibly. Where an
op's numerics demand it (softmax, layer norms, cross-entropy, RoPE
tables), the op upcasts internally and downcasts at the boundary.

Mixed precision as a *policy* (run matmuls in bf16, keep reductions
f32) is a separate, deliberate act: a native graph rewrite in the
spirit of `compile_decode` — never an ambient mode, never implicit
promotion. Future work, out of scope here.

### 2. Honesty: the capability matrix

Support is per (op, dtype, device), enforced at the **earliest
boundary**:

- **Tensor creation** (`fromTypedArray`, `fromBuffer`, `randn`,
  `zeros`, `full`, …): creating an f64 tensor on Metal fails
  immediately — today it succeeds and explodes at compute time.
- **Device movement** (`to`): moving an unsupported dtype onto a
  device fails at the move.
- **Program freezing** (`compile`, `compile_decode`): walking a graph
  whose leaves or ops the target device can't execute fails at freeze,
  never mid-run.
- **Op guards**: ops with dtype requirements (sdpa, crossEntropy,
  flash, kv_attention) state them in typed errors, as they already do.

Error shape, everywhere:
`dtype f64 is not supported on device metal (supported: f32, f16, bf16, i64, u32, u8); cast explicitly or use device cpu`.

**Never silent downcast.** f64-on-Metal does not become f32 (JAX's
warn-and-truncate causes a steady drip of numerical confusion);
f16-on-CPU-matmul does not silently upcast-accumulate without saying
so in docs. **No emulation either**: double-single f64-on-GPU
arithmetic exists in HPC and is declined — ~10× slower, subtly
non-IEEE, and nobody trains in f64.

### 3. The audited matrix (this RFC's implementation target)

| op | f32 cpu/metal | f64 cpu | f64 metal | f16 cpu | f16 metal | bf16 cpu/metal |
|---|---|---|---|---|---|---|
| elementwise (`add`, `mul`, …) | ✓ | ✓ | ✗ boundary | ✓ | ✓ | ✓ |
| `matmul` | ✓ | ✓ | ✗ boundary | ✗ typed* | ✓ | ✓ metal; ✗ typed* cpu |
| `softmax` | ✓ | ✓ | ✗ boundary | ✓ | ✓ | ✓ |
| `randn` / init | ✓ | ✓ | ✗ boundary | ✓ | ✓ | ✓ |
| `cast` | ✓ | ✓ | ✗ boundary | ✓ | ✓ | ✓ |
| `scaledDotProductAttention` (composed) | ✓ | ✓ | ✗ boundary | ✗ guard | ✗ guard | ✗ guard |
| flash sdpa (Metal) | ✓ | — | — | ✗ guard | ✗ guard | ✗ guard |
| `crossEntropy` | ✓ | ✓ | ✗ boundary | ✗ guard | ✗ guard | ✗ guard |
| `kv_attention` pool | ✓ | — | — | ✗ guard | ✗ guard | follow-up: `kvDtype` |

`*` candle's accelerate CPU backend has no f16/bf16 GEMM; fails with a
clean typed error. A naive f16 CPU matmul is a possible follow-up, not
a correctness requirement (CPU f16 is a rarity; Metal is the f16
target).

Guards marked ✗ guard stay for now — extending sdpa/CE to f16 with
op-internal f32 accumulation is a numerics decision revisited with the
paged kernel (flash pipelines are dtype-specialized anyway). The
guards' errors are already clean and typed.

## Addendum: the first storage dtype — int8 kv pools (implemented)

`InferenceConfig.kvDtype` accepts `"f32" | "f16" | "bf16" | "int8"`.
The halves store rows as-is (widen on gather); `"int8"` is the first
*storage-tier* dtype: pool slabs are u8 bytes (no generic op may touch
them) plus per-(token, head) f32 absmax scales, symmetric-quantized on
a ±127 grid (offset 128) inside `kv_attention` — quantize on scatter,
dequantize on gather, attention always in f32. 4× footprint reduction.

Honest matrix note: candle's Metal backend originally lacked u8
scatter/gather kernel instantiations; the fork gained `s_u32_u8` and
`gather_u32_u8` (`mikearnaldi/candle` d6f2056c), so int8 pools run on
both devices. fp8-e4m3 (bit-level encoding, fused dequant) waits for
the paged attention kernel.

## API surface

- `DType` gains `"bf16"` (native `NativeDType` enum + core union).
  No other additions: storage dtypes arrive with their kernels.
- No promotion configuration, no autocast flag, no dtype context —
  strictness has no knobs.
- `InferenceConfig.kvDtype?: "f32" | "f16" | "bf16"` (default `"f32"`)
  — the first application, in the follow-up commit: pool slabs
  allocated in the cache dtype, scatter casts down, gather attends in
  the cache dtype where the composed path supports it (Metal f16/bf16
  matmul+softmax pass the audit) else casts up; doubles pool capacity
  and halves gather bandwidth.

## Non-goals

- Storage dtypes (fp8/int4/mxfp4) and their kernels — declared when
  the paged attention kernel / quantized linear exists.
- Mixed-precision training (autocast rewrite, loss scaling, f32 master
  weights) — a Trainer/graph-rewrite RFC of its own.
- f16/bf16 flash attention pipelines — with the paged kernel work.
- Naive f16/bf16 CPU matmul fallback.
- Complex dtypes, int8/int16/int32 arithmetic, bool as a distinct
  dtype (masks are u8, as today).

## Alternatives considered

- **Silent downcast at the device boundary** (JAX x64-style): rejected
  — implicit precision loss contradicts the library's strictness
  posture; a gradcheck silently run in f32 isn't a gradcheck.
- **PyTorch-style promotion lattice**: rejected — silent widening
  hides both bugs and bandwidth costs; mixed precision deserves a
  deliberate rewrite pass, not ambient rules.
- **f64 emulation on Metal** (double-single): rejected — slow,
  non-IEEE, no ML use case; the error message points at CPU.
- **Exposing storage dtypes in `DType` now**: rejected — a dtype no op
  accepts is a lie; they arrive with their kernels.
- **Per-op f16 enablement now** (sdpa/CE with internal f32
  accumulation): deferred — numerics policy deserves its own pass with
  parity tests, bundled with the flash/paged-kernel dtype work.

## Acceptance criteria

1. `"bf16"` works end-to-end: creation, elementwise, matmul (Metal),
   softmax, cast, RNG — CPU and Metal tests.
2. Every device boundary rejects unsupported dtype×device pairs with
   the typed matrix error: creation, `to`, `freeze`/`compile`,
   `compile_decode`/`makeKvPool`.
3. Strictness: mixed-dtype binary ops keep failing with the explicit
   remedy message; `cast` round-trips across all compute dtypes.
4. No silent f64 on Metal anywhere in the test matrix.
5. `kvDtype` (follow-up commit): f16 and bf16 pools pass the full
   Inference parity suite (token-for-token vs naive, prefix cache,
   windowing), at doubled capacity per byte.

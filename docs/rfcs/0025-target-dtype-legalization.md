# RFC 0025: Target dtype legalization

- **Status**: Draft
- **Author**: Michael Arnaldi
- **Created**: 2026-08-30
- **Depends on**: RFC 0007 (kernel fusion), RFC 0012 (dtype system),
  RFC 0017 (multi-backend runtime), RFC 0019 (executable compilation),
  RFC 0021 (compiler pipeline refactor)
- **Updates**: RFC 0012 (device support and emulation rules), RFC 0007
  (fusion eligibility), RFC 0021 (compiler stage order)

## Summary

Keep model operation and dtype semantics independent of backend implementation,
and preserve their declared dtypes. During compilation, the selected backend
classifies each operation or fusion region as native, legalizable on the same
device, or unsupported. Record the answer in an immutable
`LegalizationPlan` and lower that plan into the executable.

The classified unit is a complete typed numerical operation, not a dtype or
buffer in isolation. Classification includes every operand and result role,
semantic dtype, storage representation, physical layout constraint, compute
dtype, accumulation dtype, and required rounding boundary. For packed values,
the operation is defined over the numbers produced by the format's canonical
interpretation. Accepting, copying, or binding the packed bytes does not imply
that the target can perform an operation over those numbers.

For example, a BF16 graph remains a BF16 graph on every target. A CPU without
the required BF16 kernel may compile one operation as:

```text
BF16 input -> convert to F32 -> F32 compute -> convert result to BF16
```

The input contract, output contract, semantic node dtype, and model definition
remain BF16. The executable contains the target-specific conversions and F32
temporary storage. Compilation fails if the backend has no implemented and
tested legalization. It never changes an explicit model dtype, moves work to
another device, or retries on CPU.

This follows the useful part of XLA's model: portable typed semantics first,
target capability policy second, explicit normalization before final lowering.
It adapts that model to effect-torch's immutable `Arc<Node>` graph and RFC
0021 side tables rather than introducing a mutable HLO clone.

## Decision

1. `DType` remains a backend-neutral semantic vocabulary. Membership in the
   enum does not promise that every operation or backend can execute the type.
2. Graph construction keeps strict dtype checking. Mixed-dtype operations
   still require an explicit graph-level `cast`. Target legalization is not
   frontend promotion.
3. Backends classify complete operations and fusion regions, not only
   `(device, dtype)` pairs. Queries include operand and result roles, shapes,
   operation attributes, known layout constraints, and target features.
4. The compiler stores one legalization decision for every lowering unit.
   Backend lowering must consume that decision and may not rediscover a
   different dtype policy.
5. A legalization may use wider compute or accumulation types while preserving
   graph-visible storage and result types. Required conversion and rounding
   points are part of the plan.
6. Only statically defined, tested same-device legalizations are allowed. There
   is no general rule that a smaller float can always widen, and no cross-device
   fallback.
7. Unsupported independent operations are compile errors. An unsupported
   optimization candidate is skipped and its nodes are considered separately.
8. Target capability and legalization policy move out of the backend-neutral
   fusion IR. The current `is_fusion_supported(device, dtype)` helper is an
   interim guard, not the final API.
9. Operation support is support for the operation over the values represented
   by each operand. A backend may fuse packed decode into a kernel, but it may
   not change the quantization grid, scale interpretation, zero-point
   interpretation, or semantic equation.
10. Packed formats remain storage representations rather than semantic
    `DType` variants. Their operation specifications include the exact format,
    represented semantic dtype, component representations, block geometry,
    compute dtype, accumulation dtype, result dtype, and numerical contract.
11. Storage representations form a closed, source-defined vocabulary. Runtime
    and serialization boundaries reject unknown representation tags and bare
    vendor family names. Adding one requires an exact interpretation, geometry
    validation, operation contracts, backend capability cases, and tests.
12. Backend-private padding, tiling, transposition, and swizzling are physical
    layouts of a known representation. They may reorganize storage but may not
    change any represented value.
13. Representation-changing operations are explicit semantic operations.
    Quantization and requantization are lossy; dequantization materializes the
    canonical represented values. None may be introduced during legalization.
    Exact repacking changes only layout, so a target may insert it only as a
    declared and statically planned execution transform.

## Motivation

The current compiler has two conflicting properties.

The semantic graph is already a good portable model. Each `Node` has a fixed
shape, dtype, and device. RFC 0021 keeps that graph immutable while optimization
selects code-generation regions in side tables. Public program signatures are
derived from the semantic roots and inputs.

Backend capability is much less precise. Fusion selection currently calls a
compiler-owned `is_fusion_supported(device, dtype)` function. It admits only
F32 and F64 on CPU, F32 and BF16 on Metal, and no CUDA fusion. That function
describes the fused implementations which happen to exist today. It does not
describe general tensor storage, independent operation support, compute
precision, reduction accumulation, hardware generation, or whether a legal
same-device rewrite exists.

Broadening that helper would select regions that the lowerer cannot execute.
Keeping it as the long-term architecture duplicates backend knowledge in the
compiler and prevents target adaptation. Both outcomes are wrong.

The concrete use cases are already present:

- CPU can store F16 and BF16 values even when a particular vector or GEMM path
  only computes in F32.
- Metal supports F16 arithmetic, but the current fused expression lowerer does
  not accept every F16 region.
- Reductions often require a wider accumulator than their input and output
  storage type.
- Quantized weights and KV caches have storage formats that are not generic
  arithmetic dtypes.
- CUDA capability depends on GPU generation and on the selected kernel path.
- Metal has no F64 arithmetic. Converting an F64 computation to F32 would not
  preserve the requested program and must remain an error.

The compiler needs a place to state these facts without putting platform
branches into model construction.

### Numeric formats are not operation contracts

A scalar format defines a set of scalar values and their interpretation. A
packed format additionally defines how codes, scales, zero points, and block
geometry represent a logical array. Neither definition, by itself, specifies
the arithmetic performed by a complete tensor operation.

This is visible in current low-precision standards. Raw FP4 E2M1 is a scalar
format. OCP MXFP4 combines E2M1 elements with E8M0 block scales, but its dot
product still leaves internal precision implementation-defined. NVIDIA NVFP4
uses E2M1 values and E4M3 block scales, while concrete storage variants may
differ in global-scale representation, block geometry, operand orientation, and
physical scale layout. Scale selection and rounding belong to the operation
that produces an NVFP4 value, not to an imported value's storage identity. GGML
K-quants define still different block interpretations and are currently
consumed by operations with f32 compute and accumulation.

The shared compiler therefore does not infer arithmetic from component names
such as `E2M1`, `E4M3`, or `blockSize = 16`. The semantic operation defines the
equation and required numerical behavior. Its complete operation specification
then identifies how each operand represents the numbers in that equation.

## Current implementation boundaries

The implementation already has most of the stage boundaries this design needs:

- `crates/compiler/src/schedule.rs` builds one `GraphIndex` with dense
  semantic IDs, child and consumer tables, roots, slots, leaves, and random
  provenance.
- `crates/compiler/src/optimization.rs` builds `OptimizationPlan`, selected
  `NativeRegion` values, output routes, and a deterministic
  `lowering_order`. It currently owns the coarse fusion dtype checks.
- `crates/compiler/src/driver.rs` sequences optimization, backend lowering,
  lowered-program validation, memory planning, and backend completion phases.
  `CompilerDriver::new` currently selects regions before it has a backend
  capability value.
- `crates/compiler/src/lowered.rs` defines dense backend instruction and value
  tables. `LoweredProgram` already permits extra backend values, so explicit
  conversion temporaries do not need semantic `Node` identities.
- `crates/compiler/src/planner.rs` plans every lowered value from declared
  bytes, storage, aliases, and instruction uses. It can plan conversion buffers
  once lowerers declare them.
- `crates/compiler/src/request.rs` derives caller-visible input and output
  signatures from semantic dtypes. Legalization must not alter that step.
- `crates/runtime-cpu/src/executable.rs` and
  `crates/runtime-metal/src/executable.rs` each own a lowerer with backend
  value metadata, instruction kinds, resource planning, and final artifact
  assembly. Those lowerers are the realization boundary.

The missing artifact is a checked contract between `OptimizationPlan` and
backend lowering. Adding that contract is smaller and safer than introducing a
second graph IR.

## What XLA actually does

This RFC is based on XLA, StableHLO, and JAX at pinned upstream revisions from
2026-08-30. XLA does not have one global dtype support matrix. Its behavior is
split across several layers.

### Portable semantics

StableHLO defines more types than any one target implements. Its type grammar
includes integer, sub-byte floating, F16, BF16, F32, F64, TF32, complex, and
quantized tensor types. Operations then impose their own operand and result
constraints. A type's presence in StableHLO means that its semantics can be
expressed, not that all backends accept it.

StableHLO also distinguishes storage types from numerical algorithms.
`dot_general` can carry precision hints or an explicit `DotAlgorithm` with
operand computation types, accumulation type, component counts, and permitted
imprecision. A known algorithm can still be unavailable on the selected
device. In that case XLA reports an error rather than substituting different
numerics.

### Frontend policy is separate

JAX resolves weak scalar promotion, X64 policy, and explicit dtype requests
before target legalization. With X64 disabled, JAX may canonicalize a requested
64-bit dtype to 32 bits before constructing HLO. That is a frontend policy and
is not what `FloatNormalization` does.

effect-torch does not adopt JAX's implicit X64 truncation. The semantic graph
continues to hold the exact dtype requested by the caller.

### Target policy

XLA's `FloatSupport` describes support for one low/high precision pair. Its
queries inspect a complete HLO instruction and answer separately for:

- a specific operand index;
- the instruction output;
- mixed-precision execution;
- effective operand precision;
- computations that must not be rewritten.

Backend implementations refine those answers. CPU policy recognizes selected
compare, select, sort, dot, and convolution paths. GPU policy considers opcode,
CUDA or ROCm capability, FP8 encoding, fusion context, and available machine
instructions. The result cannot be represented by `supports_dtype(BF16)`.

`FloatSupport` also encodes an important negative rule. Widening is not valid
for every operation. Bit shifts and count-leading-zero operations are examples
where changing the representation changes the answer. XLA therefore requires
operation-specific approval.

### Normalization

`FloatNormalization` uses the target policy to insert conversions around an
unsupported low-precision operation. It can promote operands, change the
operation result to the replacement type, and convert the result back to the
program's declared type. It also repairs unsupported mixed-precision forms.

The conversions are part of compiler IR. They are not an undocumented runtime
retry. This matters for optimization, memory planning, diagnostics, and error
reporting.

Shared called computations need special handling. XLA may clone a computation
when one caller must preserve low precision and another caller may normalize
it. effect-torch has no HLO-style called computations today, but shared fusion
regions and future control flow have the same requirement: one target plan
cannot erase distinct caller contracts.

### Verification and pipeline order

XLA runs structural verifiers throughout its pass pipelines. Target dtype
normalization is not itself semantic verification. CPU generally normalizes
BF16 and narrow formats before layout assignment. GPU performs relevant
algorithm checks and GEMM rewrites after layout assignment, then runs float
normalization. GPU runs normalization again because later rewrites can create
new BF16 operations.

There is no universal position where every target performs normalization. The
general rule is stronger: every operation reaching target code generation must
have a valid target implementation, and any pass that creates unlegalized work
must be followed by another legality check.

### What this RFC copies

- Portable graph types do not depend on the target.
- Capability checks are operation-specific and operand-specific.
- Unsupported low-precision execution may become explicit wider computation.
- Graph-visible result types survive normalization.
- Exact numerical requests are implemented exactly or rejected.
- Legality is verified before target code generation.

### What this RFC does differently

- XLA rewrites mutable HLO. effect-torch records virtual conversions and
  execution precision in side tables, then realizes them in `LoweredProgram`.
- XLA's core `FloatSupport` queries return booleans. effect-torch decisions
  carry a reason and a complete execution contract so diagnostics do not lose
  the failed requirement.
- effect-torch initially supports a small allowlist of legalizations. It does
  not import XLA's broad type system, pass manager, control-flow machinery, or
  backend matrices.

## Terminology

**Semantic dtype** is the scalar dtype recorded on a graph value. It defines the
caller-visible scalar value domain, graph typing, and program signature. It
does not, by itself, identify a packed representation or promise an
implementation.

**Storage representation** defines how a materialized tensor represents its
logical values. Dense storage holds values directly in the semantic dtype. A
packed representation uses a closed, format-specific interpretation of codes,
scales, zero points, and block geometry. Packed representations are not
semantic `DType` variants.

**Represented value** is the logical scalar produced by applying a storage
representation's exact interpretation to one logical tensor position. A
semantic operation consumes represented values, not component buffers or their
physical scalar dtypes. Dense f32 and Q4_K may both expose semantic f32 values,
but they are distinct operand specifications for capability, legalization, and
cache purposes.

**Physical layout** defines byte arrangement, component shapes, padding,
strides, tiling, and target-specific swizzling. It may reorganize a known
storage representation but cannot change its represented values.

**Compute dtype** is the type used for an operation's arithmetic. A BF16
semantic operation may use F32 compute when the plan inserts the required
conversions.

**Accumulation dtype** is the type used for a reduction, dot-product sum, or
similar accumulator. It is independent of semantic dtype, storage
representation, compute dtype, and final result dtype.

**Numerical contract** defines the operation's interpretation of operands,
compute and accumulation behavior, result formation, and required conversion
or rounding boundaries. The semantic operation owns this contract; target
classification chooses an implementation that satisfies it.

**Legalization** is a compiler-approved, same-device transformation that makes
one operation or region executable while preserving its declared numerical
contract.

**Decomposition** replaces an operation with a statically known sequence of
simpler operations. It is legal only when that sequence implements the
operation's contract. Decomposition variants are closed source-level values,
not runtime registrations.

**Capability** is a target fact used to decide whether a complete operation or
region is native, legalizable, or unsupported.

## Semantic invariants

### Graph identity and signatures do not change

Legalization never mutates or replaces `Arc<Node>` values. `GraphIndex.order`,
semantic node IDs, root order, and input slots remain unchanged.
`ProgramSignature` continues to use the semantic dtype for every binding and
output. It also records the exact storage representation and caller-visible
layout ABI for each binding. Phase 1 outputs remain dense. Supporting packed
outputs later requires an explicit producing operation plus output allocation,
publication, and invocation ABI changes. A target plan may introduce execution
dtypes and private layouts, but it does not rewrite the caller-visible dtype or
binding representation.

A model definition can therefore be instantiated with different target
placements while retaining the same operation and dtype structure. Each
prepared graph still carries one explicit device placement and is compiled by
the matching backend. The backend produces its target plan without requiring
platform branches in the model definition.

### Operations own numerical meaning

The semantic operation defines its equation over represented operand values. A
storage representation defines how those values are obtained, not which
operation is performed on them. A quantized linear operation, for example, is
defined over the canonical interpretation of its packed weight and its declared
compute, accumulation, result, and rounding behavior.

Two representations with the same physical component dtypes and block size are
not interchangeable unless their complete interpretations are identical. Two
operations over the same representation are not interchangeable unless their
numerical contracts are identical. Capability classification receives both the
operation and representation and never derives one from the other.

### Strict graph typing remains

This RFC does not add a promotion lattice. For example, graph construction
still rejects `add(f16, f32)`. The caller must insert `cast` to choose the
semantic result type.

Compiler-internal F16-to-F32 conversion answers a different question: how to
execute an already valid F16 operation on this target. It must not leak an F32
value into the next semantic node unless that node's own plan calls for it.

Likewise, a packed operand is never treated as arithmetic on its component
buffers. Generic U8 arithmetic over Q4_K codes is not a legalization of an f32
operation over Q4_K represented values.

### Semantic rounding points remain observable

Widening a whole low-precision chain and converting only its final output can
change results. Consider two BF16 elementwise nodes. The semantic output of the
first node is BF16 even when it has one consumer and is fused. A widened fused
kernel must round at that boundary unless the operation contract explicitly
permits a wider intermediate.

The legalization plan therefore records conversion boundaries within regions.
For ordinary elementwise operations, a promoted implementation behaves as:

```text
load BF16 -> F32 operation -> round to BF16 -> widen for next operation
```

Backends may fold those conversions into register operations. They may not
delete the rounding step merely because the intermediate has no buffer.

Reductions are different when their documented contract specifies wider
accumulation. An F16 sum may load F16, accumulate in F32, and round only the
result. The operation contract, not a generic widening rule, grants that
behavior.

### F64 does not become F32

F64-to-F32 substitution changes range and precision. It is not a legalization
of an F64 operation. Metal F64 arithmetic remains unsupported unless a future
RFC defines and approves an F64 implementation with its own numerical and
performance contract.

### No cross-device fallback

A Metal or CUDA compile never inserts a CPU operation. An unsupported unit is
a compile error. This preserves RFC 0017's placement, ownership, cancellation,
and resource accounting rules.

### Optimized and unoptimized execution agree

Disabling optimization changes region selection, not semantic dtype behavior.
An optimized region and the corresponding independent operations must satisfy
the same conversion and accumulation contract. Differential tests compare both
paths.

## Capability model

The compiler crate defines backend-neutral query and result types. Backend
crates implement the policy. The exact Rust names may change during
implementation, but the boundary has this shape:

```rust
pub struct ValueSpec<'a> {
    pub semantic_dtype: DType,
    pub logical_shape: &'a [usize],
    pub storage: StorageSpec<'a>,
}

pub struct StorageSpec<'a> {
    pub representation: StorageRepresentationSpec,
    pub layout_constraint: LayoutConstraintSpec<'a>,
}

pub enum StorageRepresentationSpec {
    Dense,
    Packed(PackedFormat),
}

pub enum PackedFormat {
    GgmlKQuant(GgmlKQuant),
}

pub enum LayoutConstraintSpec<'a> {
    Unconstrained,
    Canonical,
    DenseStrided { byte_strides: &'a [isize] },
    BackendPrivate {
        abi: BackendLayoutAbi,
    },
}

pub struct OperationValueSpec<'a> {
    pub role: ValueRole,
    pub value: ValueSpec<'a>,
}

pub struct OperationDTypeSpec<'a> {
    pub node: SemanticNodeId,
    pub operation: &'a SemanticOperation,
    pub operands: &'a [OperationValueSpec<'a>],
    pub results: &'a [OperationValueSpec<'a>],
    pub attributes: &'a OperationAttributes,
    pub placement: &'a DevicePlacement,
    pub required_numerics: &'a NumericsContract,
}

pub trait TargetDTypeCapabilities {
    fn fingerprint(&self) -> &TargetFingerprint;
    fn policy_revision(&self) -> u64;

    fn storage_support(&self, value: &ValueSpec<'_>) -> StorageSupport;

    fn classify_node(
        &self,
        spec: &OperationDTypeSpec<'_>,
    ) -> DTypeDisposition;

    fn classify_region(
        &self,
        spec: &RegionDTypeSpec<'_>,
    ) -> DTypeDisposition;
}

pub enum DTypeDisposition {
    Native(DTypeExecution),
    Legalize(DTypeExecution),
    Unsupported(UnsupportedDType),
}

pub struct DTypeExecution {
    pub operands: Box<[OperandExecution]>,
    pub results: Box<[ResultExecution]>,
    pub compute_dtype: Option<DType>,
    pub accumulation: Option<AccumulationExecution>,
    pub realization: ExecutionRealization,
}

pub struct OperandExecution {
    pub role: ValueRole,
    pub execution_dtype: DType,
    pub layout: ExecutionLayout,
    pub preparation: OperandPreparation,
}

pub enum OperandPreparation {
    Direct,
    Convert(DenseConversionContract),
    CanonicalPacked(PackedOperandAccess),
    RepackExact(PackedOperandAccess),
}

pub struct ResultExecution {
    pub role: ValueRole,
    pub execution_dtype: DType,
    pub layout: ExecutionLayout,
    pub completion: ResultCompletion,
}

pub enum ResultCompletion {
    Direct,
    ConvertToBoundary(DenseConversionContract),
}

pub enum ExecutionLayout {
    PreserveBoundary,
    DenseContiguous,
    BackendPrivate(BackendLayoutAbi),
}

pub enum ExecutionRealization {
    DirectKernel,
    KernelLocal,
    MaterializedTransforms,
    Decomposition(DecompositionKind),
}

pub struct UnsupportedDType {
    pub requirement: DTypeRequirement,
    pub reason: String,
}

pub struct NumericsContract {
    pub operand_interpretations: Box<[OperandInterpretation]>,
    pub compute: Option<ComputeContract>,
    pub accumulation: Option<AccumulationContract>,
    pub result_formations: Box<[ResultFormation]>,
    pub rounding_boundaries: Box<[RoundingBoundary]>,
}

pub enum OperandInterpretation {
    Direct,
    CanonicalPacked,
}

pub enum ResultFormation {
    Direct,
    Convert(DenseConversionContract),
    Quantize(QuantizationContract),
    Requantize(QuantizationContract),
}

pub enum PackedOperandAccess {
    KernelLocal,
    TileBounded {
        max_logical_elements: usize,
        max_scratch_bytes: usize,
    },
}
```

These definitions show the required information, not a commitment to the exact
Rust ownership or collection types. `GgmlKQuant` contains the existing Q2_K,
Q3_K, Q4_K, Q5_K, and Q6_K cases. `PackedFormat`, `BackendLayoutAbi`,
conversion and quantization contracts, rounding modes, and decomposition kinds
are closed source-level values. They have no `Other`, opaque representation,
callback, or runtime registration case. The initial `PackedFormat` cases cover
only the formats already accepted by RFC 0022. A later format-specific RFC adds
a variant only with its exact interpretation, geometry, validation, operations,
and fixtures.

`ValueSpec` is the sole authority for semantic dtype and logical shape.
`StorageSpec` says how that value is represented if materialized; it does not
force an internal fusion value to have a buffer. The packed format definition
owns its component schema, block geometry, and byte-size calculation. Handle
creation validates observed buffers against that definition, then operation
classification carries only the format identity. Callers cannot redefine a
known format by supplying a second component list.
Packed results are invalid in phase 1. A later explicit `quantize` or
`requantize` operation may produce one only after the public output and handle
ABIs support it. Ordinary arithmetic never selects packed result storage as an
implementation detail.

`LayoutConstraintSpec` records graph and binding constraints.
`Unconstrained` leaves layout selection to lowering. `Canonical` and
`DenseStrided` describe caller-visible storage. `BackendPrivate` identifies an
already materialized private layout through a dedicated layout ABI whose
revision changes only when that byte layout changes. It is not tied to the full
target fingerprint or legalization policy revision. Layouts selected for new
lowered temporaries live in backend value metadata, not in the semantic graph.

`OperationDTypeSpec` identifies any semantic node and operation, dense or
packed. Every operand and result has a role and one complete `ValueSpec`. The
specification also contains operation attributes, device placement, layout
constraints, and the numerical contract
required by the semantic operation. A matmul query, for example, distinguishes
LHS, RHS, result, each operand representation, compute dtype, and accumulation
contract. A select query distinguishes its condition from its value operands.
The compiler derives `required_numerics` once from the validated operation,
attributes, and value specifications. It is a borrowed query view, not mutable
metadata or a second way to define operation semantics.

An explicit semantic cast changes `ValueSpec.semantic_dtype`. Quantization and
dequantization may preserve that dtype while changing represented values or
storage, as described below. The classifier always receives both semantic dtype
and storage representation; neither is inferred from the other.

`RegionDTypeSpec` contains the complete proposed region, its boundary values,
every semantic operation in expression order, output routes, broadcast strides,
reduction geometry, and required rounding boundaries. Region support cannot be
derived from the region's output dtype alone.

`TargetFingerprint` identifies target facts that can change a decision: backend
kind, device family or architecture, required feature bits, selected library
implementation and version, and backend lowering ABI revision. The lowering ABI
revision changes when compiler-selected instruction or private-layout ABIs
change. `policy_revision` changes whenever classification or realization rules
change. Neither value contains timing or process-local pointer data.

`NumericsContract` is graph-owned and is the single source of truth for operand
interpretation, compute behavior, accumulation, result formation, and rounding
boundaries. It is not a generic fast-math flag. `CanonicalPacked` means the
canonical interpretation of the corresponding operand's `PackedFormat`; it
does not repeat the format or decoded dtype.
`DenseConversionContract` and `QuantizationContract` own closed modes for
rounding, overflow, subnormals, non-finite values, and random provenance.
`AccumulationContract` owns accumulator dtype and permitted reduction order. A
`RoundingBoundary` identifies where a declared conversion or result formation
occurs; it references that contract and does not repeat its rounding mode.
When an existing operation permits more than one reduction order or accuracy
class, that permitted set is explicit. Phase 1 does not invent bitwise order for
operations that currently promise only f32 accumulation and tolerance-based
results.

`DTypeExecution` records how the target satisfies the required contract.
`OperandExecution` and `ResultExecution` contain execution dtypes, conversion
boundaries, packed access, and selected lowered layouts. Packed access may be
kernel-local or tile-bounded; a tile bound includes both logical elements and
scratch bytes. A graph-visible dense decode is not an execution choice. It can
exist only as an explicit dequantization node. `DTypeRequirement` identifies
the operand interpretation, compute,
accumulation, result formation, rounding, representation, layout, or internal
region requirement that failed.

### Three outcomes, not a boolean

`Native` means one backend implementation consumes the operation's boundary
representations and produces its boundary representations without
compiler-inserted conversions, repacks, or decomposition. Its
`ExecutionRealization` is `DirectKernel`. A native packed kernel may interpret
codes in registers or use statically bounded tile scratch; that does not make
the operation a legalization. Native support remains operation-specific.

`Legalize` is an execution plan that adds kernel-local or materialized
conversions, an exact repack, or a statically defined decomposition. The plan
does not contain a second semantic contract. The compiler verifies its
execution dtypes, accumulator, conversions, layouts, and rounding boundaries
against `OperationDTypeSpec.required_numerics` and the boundary `ValueSpec`
values. Classification cannot weaken the operation after graph construction.

`Unsupported` carries the failed requirement. A region-level unsupported result
only rejects that optimization candidate. A node-level unsupported result fails
the compile.

### Capability queries are pure

For a fixed target fingerprint and operation specification, classification must
be deterministic and side-effect free. A query does not compile a shader, tune
a kernel, allocate memory, or inspect mutable process environment.

Kernel compilation and autotuning happen later. If they discover a static
constraint that the capability query could have known, that is a backend bug.
The constraint must move into the query rather than adding a runtime fallback.

### Storage support is separate

`storage_support` answers whether the target can own, copy, bind, and expose a
materialized value of a given representation. It does not promise arithmetic.
Operation classification answers arithmetic and kernel legality.

Storage support also validates the exact known representation and layout ABI.
Handle creation validates the actual component count, roles, shapes, byte
lengths, and alignment against the format definition. Unknown format or layout
tags fail before a handle is published. A backend-private layout is accepted
only by the backend that owns its layout ABI and only at a compatible layout
revision.

This distinction replaces device-wide dtype claims. It also leaves packed
quantized storage behind bespoke value and kernel contracts rather than making
it look like a generic arithmetic type.

The first implementation preserves existing device-boundary behavior. In
particular, it does not start accepting F64 Metal tensors merely because a Metal
buffer can hold eight-byte elements. Storage support broadens only after copy,
cast, binding, output, and error paths are implemented and tested.

## Legalization plan

The compiler builds one immutable `LegalizationPlan` after selecting regions.
It is indexed in the same deterministic order as
`OptimizationPlan.lowering_order`.

```rust
pub struct LegalizationPlan {
    pub target: TargetFingerprint,
    pub policy_revision: u64,
    pub units: Box<[LegalizedUnit]>,
    pub work: LegalizationWork,
}

pub struct LegalizedUnit {
    pub unit: LoweringUnit,
    pub disposition: ExecutableDTypePlan,
}

pub enum ExecutableDTypePlan {
    Native(DTypeExecution),
    Legalize(DTypeExecution),
}
```

The plan has these invariants:

1. It contains exactly one entry for every lowering unit and no extra entries.
2. Unit order and IDs match the optimization plan.
3. Every semantic operand and output appears once with the correct role, shape,
   dtype, storage representation, and physical layout constraints.
4. During phase 1, every operation result, root, alias, and view has dense
   storage. Packed values may enter only through validated input or generated
   bindings consumed by the RFC 0022 operations.
5. Every graph-visible result ends in its required semantic dtype and storage
   representation.
6. Every materialized conversion has a source, destination dtype, shape,
   layout, and placement on the same device.
7. Every exact repack preserves `PackedFormat`, declares compatible source and
   destination layouts, belongs to the target's layout ABI, and places any
   temporary in `LoweredProgram` before memory planning.
8. Every promoted or representation-aware region preserves required operand
   interpretations, result formation, and semantic rounding points.
9. A decomposition kind is valid for the plan's policy revision.
10. A `Native` entry uses `DirectKernel`; its operands use only `Direct` or
    `CanonicalPacked`, its results use `Direct`, and all execution layouts
    preserve their boundary layouts. Bounded packed access is allowed.
11. A `Legalize` entry does not use `DirectKernel` and contains at least one
    conversion, exact repack, or decomposition.
12. No entry is `Unsupported`.

The compiler validates these invariants before backend lowering. Backend
lowering then validates that the requested realization is one it implements.
Both checks are needed. The compiler checks graph and table consistency; the
backend checks target-specific implementation consistency.

## Compiler pipeline

The current RFC 0021 pipeline is:

```text
GraphIndex
  -> OptimizationPlan
  -> backend LoweredProgram
  -> lowered-program validation
  -> memory planning
  -> physical planning and executable assembly
```

This RFC changes it to:

```text
GraphIndex and semantic ProgramSignature
  -> region selection with target candidate classification
  -> OptimizationPlan
  -> complete and validate LegalizationPlan
  -> backend lowering realizes native ops, conversions, and decompositions
  -> lowered-program validation
  -> memory planning, including conversion temporaries
  -> physical planning and executable assembly
```

### Region selection

The selector still discovers regions without rebuilding semantic nodes. Before
committing a candidate it calls `classify_region`.

- `Native` selects the region with its direct execution plan.
- `Legalize` selects it only if the region lowerer implements the returned
  recipe. The decision is stored and reused.
- `Unsupported` skips the candidate. It does not fail compilation because the
  independent nodes may still be legal.

This replaces `is_fusion_supported`. Device equality, buffer-count limits,
broadcast legality, DAG ownership, and other target-independent region checks
remain in the selector. Backend dtype and hardware facts leave the compiler's
fusion IR.

Some current regions are backend-specific, such as Metal GEMM epilogues. They
may keep backend-specific candidate construction during the first phase. Their
eligibility still moves behind target classification. A later cleanup can make
candidate construction generic without blocking dtype legalization.

### Completing the plan

After selection, the target planner walks the final lowering order once. It
reuses stored decisions for regions and calls `classify_node` for independent
nodes. The first unsupported independent node returns a structured compile
error. Successful results form `LegalizationPlan.units`.

The driver records a `target_legalization` phase after `optimization` and before
`lowering`. If a future optimization creates or changes lowering units after
this stage, it must either update the plan through a verified transform or run
target legalization again. No unplanned unit may reach lowering.

### Kernel-local realization

Kernel-local legalization converts values in registers and does not allocate a
full temporary tensor. This is the preferred implementation for fused
elementwise operations and reductions. The fused IR must gain typed conversion
expressions. Its current identity-only cast handling is insufficient.

For a promoted BF16 elementwise region, code generation loads BF16 lanes as
F32, evaluates each operation, inserts BF16 rounding at the planned semantic
boundaries, and stores BF16 outputs. Reduction regions use their declared
accumulation type.

A direct packed kernel interprets codes and metadata inside the consumer and
never exposes component values as semantic tensor results. It may decode a
statically bounded panel or row when the execution plan declares maximum
logical elements and scratch bytes. The result must equal the format's
canonical interpretation followed by the operation's numerical contract.

### Materialized realization

Some library kernels require actual F32 buffers. In that case lowering emits
conversion instructions and temporary values into `LoweredProgram`:

```text
semantic input value
  -> convert instruction
  -> planned F32 temporary
  -> target operation
  -> planned F32 result
  -> convert instruction
  -> semantic BF16 output value
```

The existing lowered program already permits backend lowerers to add values
that do not correspond one-to-one with semantic nodes. `ValueId` indexes the
lowered value table, and backend value metadata records shape and dtype. The
memory planner therefore needs no special allocation path. It sees conversion
temporaries and their real live ranges, sizes, memory spaces, and ownership.

No conversion buffer may be allocated lazily during invocation. That would
bypass RFC 0019 memory accounting and failure guarantees.

Materialized realization does not grant permission to decode an entire packed
parameter as a hidden fallback. RFC 0022 packed linear and embedding remain
kernel-local or tile-bounded. A complete dense result exists only for an
explicit graph-level dequantization operation whose output and allocation are
part of the semantic program and memory plan.

The first implementation may emit duplicate conversions. A later deterministic
common-subexpression pass may share `(source value, destination dtype, layout)`
conversions when liveness and aliasing permit it. Sharing is an optimization,
not part of correctness.

### Decomposition

A decomposition replaces one lowering unit with a statically defined sequence.
The plan identifies a closed decomposition variant tied to the target policy
revision. Lowering exhaustively matches that variant and emits all instructions
and temporary values before memory planning. There is no runtime decomposition
registration.

There is no generic permission to decompose arbitrary operations. Each
decomposition has focused semantic and numerical tests. Exact algorithm
requests, when introduced, may use a decomposition only if it implements the
requested algorithm rather than a nearby one.

## Precision contracts

The first implementation uses named internal contracts rather than a public
autocast mode. At minimum the compiler represents:

- semantic operand and result dtypes;
- storage representation and physical layout constraints for every value;
- direct, converted, or canonical packed interpretation for every operand;
- execution dtype for each operand and result;
- accumulation dtype when the operation has an accumulator;
- direct, converted, quantized, or requantized result formation;
- exact physical repacks selected by the execution plan;
- conversion, decode, encode, and rounding points before and after the
  operation;
- internal rounding points for a fused region;
- whether the plan is native, kernel-local, materialized, or decomposed.

No public precision hint or approximate-math flag is added by this RFC. If such
an option is added later, it belongs in `CompileOptions`, participates in cache
identity, and cannot weaken an explicit exact algorithm request.

The implementation starts with conversions whose behavior is already defined by
the dense dtype system. Legalization does not invent stochastic rounding,
flush-to-zero changes, relaxed transcendental accuracy, or reassociation. Those
require separate numerical contracts. A future explicit quantization operation
may request stochastic rounding only when its semantic contract includes random
provenance; a deterministic request may not legalize to stochastic rounding.

## Backend responsibilities

### CPU

The CPU backend owns policy for scalar, vector, BLAS or Accelerate, and fused
paths. CPU support is not one matrix because a dtype accepted by scalar
elementwise code may be rejected by a selected GEMM library.

The first useful CPU legalizations are:

- F16 and BF16 elementwise regions through F32 register compute with semantic
  rounding boundaries;
- F16 and BF16 reductions with operation-approved F32 accumulation;
- selected F16 and BF16 independent operations through materialized F32
  buffers when no suitable native kernel exists.

Each entry lands separately with tests. This list is not a blanket promise that
all CPU half operations legalize.

### Metal

Metal policy owns language support, GPU family requirements, library-kernel
constraints, buffer limits, and fused MSL capabilities. F32, F16, and BF16 may
have different native coverage even when the device can store all three.

The initial Metal work moves current F32/BF16 fusion checks into target policy
and adds F16 only when the fused MSL path passes numerical and optimized versus
unoptimized tests. Metal F64 remains unsupported.

### CUDA

CUDA policy includes compute capability and selected kernel path in its target
fingerprint. Fusion remains unsupported until it is implemented and tested on
hardware. This RFC does not turn the current `Device::Cuda(_) => false` fusion
guard into an optimistic claim.

When enabled, CUDA decisions must distinguish native BF16 and F16, scalar FP8
semantic dtypes, TF32-style compute policy, packed MX or NVIDIA formats,
operand orientation and scale layout, accumulation contracts, GPU generation,
and library or generated-kernel requirements. A bare `NVFP4` family tag is not
an operation specification. A future CUDA policy belongs in `runtime-cuda`, not
in `effect-torch-compiler`.

### Shared compiler

`effect-torch-compiler` owns specifications, plan structure, deterministic
ordering, validation, generic conversion boundaries, diagnostics, and work
counters. It does not import CPU, Metal, or CUDA implementations.

The driver remains statically dispatched. Backends pass a concrete capability
implementation into driver construction; no plugin ABI or runtime trait object
is required.

## Errors and diagnostics

An unsupported compile reports the complete failed requirement. For example:

```text
compile: node 42 matmul is unsupported on metal:0: lhs=f64, rhs=f64,
result=f64, accumulation=f64; Metal has no native f64 arithmetic and no
implemented semantics-preserving legalization
```

For an operand-specific failure:

```text
compile: node 17 select operand 0 requires u8 condition support on cuda:0;
the selected kernel accepts only predicate conditions
```

Errors include the semantic node or region ID, operation, operand or result
role, semantic dtype, requested execution or accumulation dtype when present,
target fingerprint summary, and backend reason. They do not suggest `cast` when
the compiler could have applied an approved semantics-preserving legalization.
They may suggest an explicit model cast when changing model semantics is the
actual remedy.

`CompilerWorkReport` gains structural counters for:

- capability queries;
- native lowering units;
- legalized lowering units;
- kernel-local legalizations;
- materialized conversions;
- decompositions;
- rejected region candidates;
- unsupported independent units.

Executable diagnostics expose a compact per-strategy summary and the
`target_legalization` phase timing. Detailed per-unit traces remain an opt-in
compiler debug output. Timings and debug text do not affect cache identity.

## Cache identity

The caller-supplied cache key identifies the semantic program under the current
cache API. It is not replaced by `ProgramSignature`, which describes the
program's calling convention rather than its internal operations. Legalization
is deterministic for this executable identity:

```text
caller-supplied semantic program key
+ CompileOptions
+ target fingerprint
+ backend legalization policy revision
```

Backend executable caches must compare this full contract on a user-supplied
cache-key hit. They also compare `ProgramSignature` and generated binding
metadata as ABI guards. The signature cannot replace the semantic program key,
and neither guard can replace the target fingerprint or policy revision.

The derived `LegalizationPlan` does not become a caller option and is not hashed
back into `ProgramSignature`. It is an artifact of compiling the semantic
program for a target. Persistent compiled artifacts, if added later, store the
fingerprint and policy revision and reject incompatible targets.

## Interaction with creation and movement

RFC 0012 required every unsupported dtype and device pair to fail at tensor
creation or movement. That rule mixed storage with arithmetic. This RFC narrows
it as follows:

- Creation and movement validate storage, copy, binding, and representation
  support.
- Graph construction validates semantic operation dtype rules.
- Compilation validates native or legalizable target execution for every
  lowering unit.
- Invocation performs no new static capability decisions.

This is still early failure. A tensor may be legal storage for one operation
and illegal for another, so the operation can only be judged when it is known.
No unsupported combination survives compilation.

The first implementation does not broaden storage support. It separates the
questions while preserving current accepted and rejected device values.

## Quantized and packed storage

Quantized weights and KV caches do not pass through generic float
normalization. Their operation specifications include the packed storage format,
scale and zero-point representation, group or block geometry, semantic dtype,
compute dtype, accumulation contract, result formation, and rounding
boundaries.

### Format interpretation

Every accepted packed format is a closed `PackedFormat` variant with one exact,
backend-neutral interpretation. The format definition specifies:

- the authoritative format and revision;
- the exact represented values and compatible semantic dtype constraints;
- physical component roles, byte encodings, and component shapes;
- logical-to-physical block geometry and byte-size calculation;
- the interpretation of codes, scales, zero points, codebooks, and special
  values;
- valid shapes, padding, alignment, and representation-preserving views;
- canonical fixtures for every component and boundary case.

That definition answers which numbers the bytes represent. It does not define
addition, multiplication, dot-product accumulation, result dtype, or
quantization of new input values. Those belong to semantic operations and their
numerical contracts. The same packed format may participate in several
operations, and an operation may accept different representations in different
operand roles.

Unknown serialized type codes, representation tags, format revisions, and
backend-private layout ABIs fail before handle or executable publication. An
unknown quantization policy fails when constructing or compiling an explicit
quantization operation; it is not metadata required to read an already encoded
value. A family label such as `NVFP4` is not sufficient when multiple concrete
interpretations or scale geometries exist. There is no opaque packed format,
generic component schema, or user codec registration.

### Operation contracts

A packed operation's semantic equation is written over represented values. RFC
0022, for example, defines quantized linear and embedding against the canonical
decode of the packed weight. The operation specification names the exact weight
format and separately fixes activation, bias, compute, accumulation, and result
dtypes. Physical component encodings never stand in for those roles.

A target classifies the complete operation as `Native` only when one direct
kernel implements that equation under the required numerical contract. That
kernel may perform canonical packed access in registers or bounded tile scratch.
`Legalize` applies when the compiler must add an exact representation-preserving
repack, dtype conversion, or statically defined decomposition. Merely accepting
the component buffers is not native operation support. Generic U8 arithmetic
over packed codes is never a legalization.

OCP MXFP4 illustrates the boundary. Its E2M1 values, E8M0 scales, and 32-value
blocks define represented inputs, while the OCP dot operation leaves internal
precision implementation-defined. An effect-torch dot or matmul contract must
therefore state the accepted operand representations, compute behavior,
accumulation dtype and order, result dtype, and rounding boundaries before a
backend can classify it.

Affine quantization has the same split. A concrete format fixes code signedness
and width, scale and zero-point representations, axis or group geometry, and an
exact decode equation such as `x = scale * (code - zeroPoint)`. The `quantize`
operation fixes how it selects and rounds scales, zero points, and codes. An
affine storage description cannot supply those operation semantics by itself.

NVIDIA NVFP4 requires the same separation. A concrete storage variant fixes its
E2M1 values, E4M3 block scales, global-scale representation where present,
block geometry, logical-to-physical mapping, and canonical layout. A
quantization operation separately defines global and block scale selection,
scale rounding, element rounding, saturation, non-finite handling, and any
random provenance. A matmul operation separately defines compute, accumulation,
and result behavior. A target-specific scale swizzle is an exact physical
repack only if it preserves every represented value at every logical position.

GGML K-quants initially have canonical imported representations and decode
semantics from RFC 0022. Quantized linear and quantized embedding consume them
today. A future explicit dequantization operation may also consume them.
Encoding dense values to a K-quant is unsupported until a format-specific
contract names and tests one exact quantization algorithm. A decoder does not
imply an encoder.

For a current Q4_K linear weight, the weight `ValueSpec` has semantic dtype f32
and `Packed(GgmlKQuant(Q4K))` storage. Its operand contract uses
`CanonicalPacked`; activations and results are dense f32; multiplication and
accumulation use f32 under RFC 0022's permitted reduction behavior. A direct
Q4_K kernel is `Native` even though it interprets blocks internally. An exact
layout repack followed by that kernel is `Legalize`. Generic U8 arithmetic or a
hidden complete f32 weight is `Unsupported`. Quantized embedding follows the
same represented-value rule but has no dot-product accumulator.

MXFP4 and NVIDIA NVFP4 are design cases in this RFC, not accepted
`PackedFormat` variants. Each needs a format-specific RFC before core, a loader,
or a backend accepts it. `GGML_TYPE_NVFP4` and NVIDIA Transformer Engine NVFP4
are distinct formats. GGML groups 64 values with four unsigned E4M3 scale
bytes and no per-tensor global scale. Transformer Engine defines each value as
E2M1 times a 16-value E4M3 block scale times a tensor-wide f32 scale, with
separate rowwise and columnwise layouts. Shared E2M1 payload values do not make
their storage or operation contracts interchangeable.

### Representation-changing operations

`cast` changes semantic dtype according to the dense dtype conversion contract.
It does not implicitly create, remove, or reinterpret a packed representation.

`quantize`, when introduced for a format, is an explicit lossy semantic
operation. Its destination representation and complete scale selection,
rounding, saturation, non-finite, and random contract are graph-visible and
participate in semantic program and executable cache identity.

`dequantize` materializes the canonical represented values in dense storage. It
preserves semantic dtype unless accompanied by an explicit cast. A backend may
fuse decode into a consumer only when the result still satisfies the consumer's
declared operation contract.

`requantize` means canonical decode followed by the destination quantization
contract. The plan preserves both boundaries. An exact repack is not a semantic
numeric operation. It changes physical layout only and must preserve codes,
scales, zero points, and every represented value at every logical position. The
compiler may insert one as a declared execution transform.

Legalization cannot introduce quantization, requantization, a different scale
selection rule, or stochastic rounding unless the semantic graph operation
already requests that exact behavior.

## Verification strategy

### Capability tests

Each backend has table-driven tests for every defined decision. Tests cover
positive and negative cases across operation, operand role, semantic dtype,
storage representation, shape, layout constraint, numerical contract, and
hardware feature. Unsupported results assert the reason, not only the enum
variant. A valid packed buffer may pass `storage_support` while generic
arithmetic and unrelated operations over it remain unsupported.

### Plan validation tests

Compiler tests construct corrupt plans and verify rejection of missing units,
duplicate units, dtype or representation mismatches, wrong operand
interpretation, wrong result formation, omitted region rounding points,
cross-device conversions, unknown decomposition variants, private-layout owner
mismatches, undeclared repack temporaries, invalid `Native` or `Legalize`
combinations, phase-1 packed results, and target fingerprint mismatch.

Deep graphs remain iterative. Plan construction performs one pass over the
final lowering order and does not recursively traverse semantic nodes.

### Lowered IR tests

Snapshot-style structural tests check instruction and value tables for:

- native operations with no conversion;
- kernel-local promoted regions;
- materialized operand and result conversions;
- multiple outputs;
- shared inputs;
- aliases and views;
- conversion temporary liveness and reuse;
- decomposition resource declarations;
- kernel-local and tile-bounded packed decode;
- exact repacks with no semantic conversion;
- explicit quantize, dequantize, and requantize boundaries.

Every materialized temporary appears in `LoweredProgram.values` before memory
planning. Every instruction use appears in one resource category.

### Numerical tests

For each legalization, compare against a reference that explicitly performs the
same operand interpretation, conversions, compute, accumulation, and result
formation. Packed operations compare against their canonical represented-value
oracle, not the original values before quantization. Tests include boundary
values, NaN, infinity, signed zero, subnormals where the backend preserves them,
overflow, scale boundaries, and reduction lengths that expose accumulator
behavior.

Optimized and unoptimized programs use the same inputs and compare at the
semantic dtype. Fused low-precision chains include cases where omitting an
intermediate rounding point changes the result.

Tests run on the actual target. Unsupported Metal or CUDA behavior is never
hidden behind CPU execution.

### Failure tests

Compile tests assert that:

- Metal F64 arithmetic fails before executable publication;
- an unsupported region candidate is rejected while legal independent
  operations still compile;
- an unsupported independent operation fails compilation;
- explicit exact numerical requests are not silently replaced;
- a cache entry with the wrong target fingerprint is rejected;
- a cache entry with the wrong backend lowering ABI revision is rejected;
- invocation never performs capability fallback or allocates undeclared
  conversion storage;
- unknown packed formats, revisions, and private layout ABIs fail before handle
  or executable publication;
- unknown quantization policies fail when an explicit quantization operation is
  constructed or compiled;
- packed execution never materializes a hidden complete decoded parameter;
- deterministic numerical contracts reject stochastic realizations.

## Expected code changes

The phased implementation is expected to touch these files and responsibilities:

- Add `crates/compiler/src/legalization.rs` for target specifications,
  dispositions, plans, validation, errors, and work counters. Re-export the
  public compiler types from `crates/compiler/src/lib.rs`.
- Add the closed storage-representation vocabulary and canonical packed-format
  interpretation beside shared runtime dtype definitions. Extend graph value
  metadata, `GraphIndex`, binding declarations, and program signatures so the
  compiler sees semantic dtype and storage representation together.
- Update `crates/compiler/src/optimization.rs` so region candidates call a
  supplied target classifier and accepted decisions stay attached to selected
  regions. Remove dtype policy imports from `ir.rs`.
- Update `crates/compiler/src/driver.rs` so construction receives concrete
  target capabilities, completes `LegalizationPlan`, records
  `TARGET_LEGALIZATION_PHASE`, and passes each unit's plan to the lowering
  callback.
- Extend `crates/compiler/src/diagnostics.rs` with legalization work counters.
  Keep timings out of cache identity.
- Extend `crates/compiler/src/ir.rs` only for typed conversion and rounding
  expressions, canonical packed interpretation, and exact repack boundaries
  needed by kernel-local realization. It must not regain backend policy.
- Add CPU policy beside CPU executable lowering. Update
  `crates/runtime-cpu/src/fusion.rs` and the CPU lowerer only as each approved
  realization lands.
- Add Metal policy beside Metal executable lowering. Keep device-family and MSL
  capability facts in `crates/runtime-metal`.
- Add CUDA policy in `crates/runtime-cuda` with all fusion results unsupported
  until hardware-tested lowering exists.
- Update backend cache entries to retain and compare the semantic program key,
  compile options, target fingerprint, and policy revision. Continue to compare
  `ProgramSignature` and generated bindings as ABI guards.

No generated declaration changes are needed for phase 1 unless diagnostics or
compile options become public through NAPI. If that happens, declarations are
regenerated from Rust rather than edited by hand.

## Implementation plan

### Phase 0: Freeze current behavior

1. Add backend tests that record the current independent-operation and fusion
   dtype matrix.
2. Add optimized versus unoptimized numerical fixtures for F32, F64 CPU, and
   BF16 Metal regions.
3. Keep CUDA fusion disabled.
4. Record current compile errors for unsupported independent nodes.
5. Record current Q2_K through Q6_K storage validation, quantized linear and
   embedding contracts, scratch bounds, and canonical represented-value
   results on every implemented backend.

### Phase 1: Add capability and plan plumbing

1. Add the backend-neutral capability specifications, dispositions, target
   fingerprint, legalization plan, validator, and work counters to
   `effect-torch-compiler`.
2. Make the shared `PackedFormat::GgmlKQuant` definition authoritative for
   Q2_K through Q6_K interpretation, geometry, and validation. Remove duplicate
   codec and logical-shape attributes from operation nodes where the weight
   `ValueSpec` already supplies them.
3. Carry dense and current GGML K-quant `StorageSpec` values through graph
   metadata, scheduling, signatures, and operation specifications. Preserve
   RFC 0022's f32 semantic dtype, canonical decode, f32 accumulation, current
   reduction-order allowance, and bounded scratch contract.
4. Pass concrete backend capability implementations into `CompilerDriver`.
5. Make region selection classify complete candidates and store accepted
   decisions.
6. Classify every final independent lowering unit.
7. Add `target_legalization` between optimization and lowering.
8. Implement CPU, Metal, and CUDA policies that reproduce current behavior
   exactly, including current packed operations.
9. Replace `is_fusion_supported` and other dtype-only region gates with target
   classification.

This phase must not broaden support or change numerical output.

### Phase 2: Kernel-local legalization

1. Extend `KernelExpr` with typed conversions and explicit low-precision
   rounding boundaries.
2. Teach CPU fused elementwise and reduction lowering to realize approved F16
   and BF16 plans through F32 registers and accumulators.
3. Add Metal F16 fusion only if the MSL implementation satisfies the same plan
   contract.
4. Add differential and edge-case numerical tests for each enabled operation.
5. Report strategy and conversion counts in diagnostics.

Support lands per operation family. A passing elementwise implementation does
not enable GEMM, transcendental, random, or indexing operations automatically.

### Phase 3: Materialized conversions

1. Add explicit backend conversion instruction kinds where existing cast
   instructions cannot express the plan.
2. Emit F32 temporary values for approved library-kernel paths.
3. Validate temporary shape, dtype, memory space, ownership, and liveness.
4. Add selected CPU half GEMM or other independent operation legalizations only
   after performance and numerical tests.
5. Add deterministic conversion sharing if measurements justify it.

### Phase 4: New packed formats and algorithm contracts

1. Add closed decomposition variants with exhaustive backend matches; do not
   add a runtime registry.
2. Separate precision hints from exact algorithms if the public graph gains
   either feature.
3. Require target validation before lowering an exact algorithm.
4. Add exact decompositions only with algorithm-specific tests.
5. Add a new packed format only through a format-specific RFC that defines its
   interpretation, supported semantic operations, numerical contracts, target
   cases, and fixtures.
6. Add explicit quantize, dequantize, or requantize operations only with their
   output ABI, structural, and numerical tests. Add exact repack strategies
   only with layout ABI and represented-value preservation tests.
7. Re-run legalization after any pass that can introduce new lowering units.

### Phase 5: Remove duplicate policy

1. Remove remaining compiler-owned backend dtype matrices.
2. Move statically knowable lowerer failures into capability classification.
3. Keep runtime checks only for dynamic invocation data and defensive
   validation.
4. Update RFC 0012's audited matrix to distinguish storage, native execution,
   legalizable execution, and unsupported execution.

## Alternatives considered

### Keep a device and dtype matrix

Rejected. It cannot represent operand roles, accumulation, shape restrictions,
hardware generation, selected kernel path, or fusion-region support. It also
puts backend facts in the shared compiler.

### Make every declared dtype supported

Rejected. A broad `DType` enum defines semantics, not implementations. Treating
it as a promise would defer failures to lowering or invocation and would select
invalid fusion regions.

### Make packed formats semantic dtypes

Rejected. Q4_K, affine INT4, MXFP4, and NVFP4 combine logical values with
block-level metadata and representation constraints. Treating them as scalar
`DType` values loses the distinction between represented semantic values,
component storage, compute dtype, accumulation dtype, and result dtype. It also
breaks generic element-size, stride, view, and cast assumptions. They remain
closed storage representations in complete operation specifications.

### Infer operations from a structural format descriptor

Rejected. Component formats and block geometry determine neither quantization
nor arithmetic. Identical-looking E2M1 payloads and E4M3 scales can participate
in different scale-selection, rounding, saturation, dot-product, accumulation,
and result contracts. The semantic operation states those rules explicitly; a
generic `BlockScaled { element, scale, block_shape }` value does not.

### Preserve opaque or unknown representations

Rejected. Core cannot validate represented values, geometry, ownership,
operations, or resource requirements for an unknown format. Runtime and file
boundaries reject unknown tags. New formats enter through closed source-level
variants and format-specific RFCs, not opaque IDs, callbacks, plugins, or
pass-through archive behavior.

### Rewrite the semantic graph

Rejected. Rebuilding `Arc<Node>` graphs would undo RFC 0021's identity,
complexity, and compile-time improvements. Legalization is target-specific and
does not belong in the model graph.

### Let each lowerer insert casts ad hoc

Rejected. Hidden casts cannot be checked before lowering, compared across
optimized and unoptimized paths, included in diagnostics, or treated as one
deterministic target plan. Policy would remain duplicated across backend
branches.

### Always materialize converted tensors

Rejected. It is simple but wastes bandwidth and memory for elementwise fusion.
Kernel-local conversion preserves static accounting without allocating full
temporaries. Materialization remains available for library kernels that require
it.

### Widen all narrow floats to F32

Rejected. Widening can change representation-sensitive operations, rounding
points, exact algorithms, and performance by an unacceptable amount. Every
legalization is operation-specific and tested.

### Fall back to CPU

Rejected. It hides transfers and target gaps, changes ownership and cancellation
behavior, and violates RFC 0017. Applications may partition work explicitly in
a future multi-device design. This compiler does not do it implicitly.

### Require callers to cast models manually

Rejected as the only mechanism. An explicit cast changes model semantics and
public value types. It remains the remedy when the caller intends that change.
It should not be required for an implementation detail such as an F32
accumulator that returns the original BF16 result.

## Risks

### Legalization changes low-precision rounding

The plan records semantic conversion boundaries, and tests include expressions
that differ when a boundary is removed. Region lowering cannot infer a single
compute dtype and round only at the final output.

### Capability policy drifts from lowering

Backends implement a second validation when consuming a plan. A mismatch is a
compile-time backend error. Static failures then move into capability tests and
policy rather than becoming fallback branches.

### Converted buffers increase peak memory

All materialized conversions enter the lowered value table before memory
planning. Diagnostics report their count and bytes. Kernel-local conversion is
preferred, and conversion sharing is measured rather than assumed.

### Region selection becomes target-dependent

It already is target-dependent through device checks and backend-specific
regions. This RFC makes that dependency explicit and testable. Semantic graph
identity and result contracts remain target-independent.

### Hardware fingerprints make cache behavior harder to inspect

Diagnostics print a compact fingerprint and policy revision. Cache tests cover
same-program compilation across distinct target contracts. No pointer identity
or transient driver state enters the key.

### The abstraction becomes a general kernel registry

The first implementation is limited to dtype execution contracts. Scheduling,
autotuning, pipeline compilation, and runtime dispatch remain backend-owned.
If a future capability system covers more kernel constraints, it should extend
the complete operation specification rather than introduce parallel registries.

## Acceptance criteria

1. The same model definition, instantiated with CPU or Metal placement, retains
   the same operation and dtype structure and compiles without target-specific
   model branches. Legalization changes neither semantic node IDs nor the
   prepared graph's `ProgramSignature`.
2. Every final lowering unit has one validated legalization entry before
   backend lowering begins.
3. No compiler-owned `(Device, DType)` fusion support matrix remains.
4. CPU and Metal phase 1 behavior matches the pre-RFC support matrix exactly.
5. Each newly enabled F16 or BF16 operation has capability, lowered-IR,
   optimized versus unoptimized, edge-case numerical, and failure tests.
6. Materialized conversions are visible to liveness and memory planning and do
   not allocate undeclared invocation storage.
7. Metal F64 arithmetic still fails compilation with an operation-specific
   reason.
8. CUDA fusion remains disabled until a hardware-tested policy and lowerer land.
9. No unsupported accelerator operation runs on CPU implicitly.
10. Cache hits require matching the caller-supplied semantic program key,
    compile options, target fingerprint, and legalization policy revision.
    `ProgramSignature` remains a separate ABI guard.
11. `cargo test -p effect-torch-compiler` and focused CPU and Metal executable
    suites cover plan validation and numerical parity. Relevant CUDA tests run
    on the CUDA devbox before CUDA decisions are enabled.
12. Compiler diagnostics report target legalization timing, strategies,
    conversion counts, and rejected region candidates without changing cache
    identity.

## Relationship to existing RFCs

### RFC 0007: Kernel fusion

This RFC replaces dtype-only fusion eligibility with complete target region
classification. It preserves region ownership, buffer limits, output routing,
and the rule that fusion never changes graph-visible semantics.

### RFC 0012: Dtype system

Strict graph typing remains. This RFC updates the blanket statements that a
dtype and device pair must be accepted or rejected globally and that emulation
is always disallowed. Storage and arithmetic are checked separately. A
statically defined same-device legalization is allowed only when it preserves
the operation's declared contract. Silent model downcasts, generic promotion,
and F64-to-F32 substitution remain forbidden.

### RFC 0017: Multi-backend runtime

Backend packages still own devices, kernels, compilers, and caches. Core stays
independent of concrete backends. Legalization never partitions a graph across
runtimes or falls back to CPU.

### RFC 0019: Executable compilation

All conversion and decomposition decisions occur during compilation. Their
temporaries and resources enter static memory planning. Invocation executes one
fixed program and makes no dtype capability decisions.

### RFC 0021: Compiler pipeline refactor

`LegalizationPlan` is another immutable side table over the existing
`GraphIndex` and final `LoweringUnit` order. It does not rebuild semantic nodes.
The compiler driver gains one verified phase between optimization and lowering.

### RFC 0022: Pretrained quantized inference

Packed formats keep bespoke operation contracts. Capability specifications may
describe their storage and semantic dtypes, but generic float widening does not
apply to packed payloads.

## Primary sources

Research snapshot: 2026-08-30.

- OCP MX formats and dot-product requirements: [OCP Microscaling Formats MX
  v1.0 specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- NVIDIA Transformer Engine NVFP4 representation, quantization, operand
  layouts, and scale swizzling: [`nvfp4.rst`](https://github.com/NVIDIA/TransformerEngine/blob/3ceb248e01a2c0dc1215fe0f46ebc235f289ba0d/docs/features/low_precision_training/nvfp4/nvfp4.rst#L27-L275)
- GGML MXFP4 and NVFP4 type identities: [`ggml.h`](https://github.com/ggml-org/ggml/blob/d4716378882593333721eb33f153144b6885caf2/include/ggml.h#L429-L430)
  and GGML NVFP4 block geometry: [`ggml-common.h`](https://github.com/ggml-org/ggml/blob/d4716378882593333721eb33f153144b6885caf2/src/ggml-common.h#L221-L227)
- StableHLO type and quantization semantics: [StableHLO specification, types](https://github.com/openxla/stablehlo/blob/b7c389d080653facae294ec49f6f0374e0672f0a/docs/spec.md#L89-L312)
- StableHLO dot precision and algorithm semantics: [StableHLO specification, dot_general](https://github.com/openxla/stablehlo/blob/b7c389d080653facae294ec49f6f0374e0672f0a/docs/spec.md#L2826-L2969)
- StableHLO algorithm verification: [`TypeInference.cpp`](https://github.com/openxla/stablehlo/blob/b7c389d080653facae294ec49f6f0374e0672f0a/stablehlo/dialect/TypeInference.cpp#L4293-L4321) and [`Base.cpp`](https://github.com/openxla/stablehlo/blob/b7c389d080653facae294ec49f6f0374e0672f0a/stablehlo/dialect/Base.cpp#L642-L711)
- XLA target float policy interface: [`float_support.h`](https://github.com/openxla/xla/blob/0644417f95ad247b511558a92bc48b765730dc5a/xla/service/float_support.h#L24-L83) and [`float_support.cc`](https://github.com/openxla/xla/blob/0644417f95ad247b511558a92bc48b765730dc5a/xla/service/float_support.cc#L27-L113)
- XLA normalization rewrite: [`float_normalization.cc`](https://github.com/openxla/xla/blob/0644417f95ad247b511558a92bc48b765730dc5a/xla/hlo/transforms/simplifiers/float_normalization.cc#L465-L580)
- XLA handling of shared computations: [`float_normalization.cc`](https://github.com/openxla/xla/blob/0644417f95ad247b511558a92bc48b765730dc5a/xla/hlo/transforms/simplifiers/float_normalization.cc#L627-L691) and [`float_normalization_test.cc`](https://github.com/openxla/xla/blob/0644417f95ad247b511558a92bc48b765730dc5a/xla/hlo/transforms/simplifiers/float_normalization_test.cc#L711-L750)
- XLA CPU policy examples: [`cpu_float_support.h`](https://github.com/openxla/xla/blob/0644417f95ad247b511558a92bc48b765730dc5a/xla/service/cpu/cpu_float_support.h#L32-L70) and [`onednn_float_support.cc`](https://github.com/openxla/xla/blob/0644417f95ad247b511558a92bc48b765730dc5a/xla/service/cpu/onednn_float_support.cc#L22-L67)
- XLA GPU policy: [`gpu_float_support.cc`](https://github.com/openxla/xla/blob/0644417f95ad247b511558a92bc48b765730dc5a/xla/service/gpu/gpu_float_support.cc#L39-L191)
- XLA HLO verifier and pass invariants: [`hlo_verifier.h`](https://github.com/openxla/xla/blob/0644417f95ad247b511558a92bc48b765730dc5a/xla/service/hlo_verifier.h#L43-L158), [`hlo_verifier.cc`](https://github.com/openxla/xla/blob/0644417f95ad247b511558a92bc48b765730dc5a/xla/service/hlo_verifier.cc#L2481-L2547), and [`hlo_pass_pipeline.h`](https://github.com/openxla/xla/blob/0644417f95ad247b511558a92bc48b765730dc5a/xla/hlo/pass/hlo_pass_pipeline.h#L69-L87)
- XLA GPU algorithm checks: [`algorithm_util.cc`](https://github.com/openxla/xla/blob/0644417f95ad247b511558a92bc48b765730dc5a/xla/service/algorithm_util.cc#L261-L348) and [`algorithm_checker.cc`](https://github.com/openxla/xla/blob/0644417f95ad247b511558a92bc48b765730dc5a/xla/backends/gpu/transforms/algorithm_checker.cc#L47-L106)
- XLA exact dot decomposition: [`dot_algorithm_rewriter.cc`](https://github.com/openxla/xla/blob/0644417f95ad247b511558a92bc48b765730dc5a/xla/backends/gpu/transforms/dot_algorithm_rewriter.cc#L330-L394)
- XLA CPU pipeline ordering: [`cpu_compiler.cc`](https://github.com/openxla/xla/blob/0644417f95ad247b511558a92bc48b765730dc5a/xla/service/cpu/cpu_compiler.cc#L795-L842) and [`cpu_compiler.cc`](https://github.com/openxla/xla/blob/0644417f95ad247b511558a92bc48b765730dc5a/xla/service/cpu/cpu_compiler.cc#L968-L1024)
- XLA GPU pipeline ordering: [`gpu_compiler.cc`](https://github.com/openxla/xla/blob/0644417f95ad247b511558a92bc48b765730dc5a/xla/service/gpu/gpu_compiler.cc#L2007-L2140) and [`gpu_compiler.cc`](https://github.com/openxla/xla/blob/0644417f95ad247b511558a92bc48b765730dc5a/xla/service/gpu/gpu_compiler.cc#L2200-L2267)
- JAX X64 canonicalization: [`dtypes.py`](https://github.com/jax-ml/jax/blob/5832e866449a41c3eea6333416528039119a0fde/jax/_src/dtypes.py#L349-L386) and [`config.py`](https://github.com/jax-ml/jax/blob/5832e866449a41c3eea6333416528039119a0fde/jax/_src/config.py#L1701-L1713)
- JAX dot precision lowering: [`lax.py`](https://github.com/jax-ml/jax/blob/5832e866449a41c3eea6333416528039119a0fde/jax/_src/lax/lax.py#L2528-L2628) and [`lax.py`](https://github.com/jax-ml/jax/blob/5832e866449a41c3eea6333416528039119a0fde/jax/_src/lax/lax.py#L6192-L6297)

/**
 * Backend service-provider interface for effect-torch.
 *
 * `RuntimeService` is the trust boundary between the backend-neutral graph API
 * and a concrete native implementation. The core submits immutable semantic
 * node requests, then delegates autodiff, compilation, execution, transfer,
 * and optional file or decode facilities to the runtime in the Effect
 * environment. Application code normally uses `Tensor`; backend adapters
 * implement this module.
 *
 * Handles are opaque capabilities, not structurally interchangeable records.
 * An implementation must maintain native ownership and liveness out of band,
 * reject forged, foreign, or released handles except for idempotent release,
 * and expose immutable logical metadata. `RuntimeService.identity` identifies
 * exactly one interchangeable handle/cache domain; placement ids are meaningful
 * only inside that domain.
 *
 * Effects that cross an asynchronous native boundary must cooperate with
 * interruption. Once interrupted, they must not publish a late result, and
 * must reclaim any native ownership produced after the caller stopped waiting.
 * Inputs are borrowed for the duration of an operation. Successful methods
 * that return concrete tensors transfer cleanup responsibility per distinct
 * returned handle unless their documentation says otherwise. Tensor release is
 * idempotent; ownership does not impose an exact-once call requirement.
 *
 * @since 0.1.0
 */
import { Context, Data, type Effect } from "effect"
import type { Pipeable } from "effect/Pipeable"

/**
 * Logical element data types understood by the common runtime protocol.
 * Individual runtimes advertise a subset through {@link Capabilities}; an
 * operation may impose a narrower subset. No implicit promotion rule is part
 * of this low-level type.
 *
 * @since 0.1.0
 * @category models
 */
export type DType = "f32" | "f64" | "f16" | "bf16" | "i64" | "u8" | "u32"

/**
 * Packed GGML K-quant storage encodings understood by native runtimes.
 *
 * @since 0.1.0
 * @category models
 */
export type TensorStorageEncoding = "Q2_K" | "Q3_K" | "Q4_K" | "Q5_K" | "Q6_K"

/**
 * Physical storage metadata for a logically `f32` encoded tensor. The logical
 * shape remains on {@link TensorHandle}; `physicalShape` describes the packed
 * `u8` buffer exposed by readback and binding validation. GGML K-quant rows
 * flatten all logical leading dimensions, encode the final dimension in
 * 256-element blocks, and use physical shape `[rows, rowBytes]`.
 *
 * Encoded storage is not a strided dense layout. Backends may accept it only in
 * dedicated packed operations and must report `unsupported-layout` elsewhere.
 *
 * @since 0.1.0
 * @category models
 */
export interface EncodedTensorStorage {
  /** Packed encoding used for each logical row. */
  readonly encoding: TensorStorageEncoding
  /** Shape of the packed byte buffer, independent of the logical shape. */
  readonly physicalShape: ReadonlyArray<number>
  /** Physical packed elements are always bytes. */
  readonly physicalDtype: "u8"
}

/**
 * Backend implementation metadata.
 *
 * @since 0.1.0
 * @category models
 */
export interface BackendInfo {
  /** Stable package or implementation name used in diagnostics. */
  readonly name: string
}

/**
 * A runtime-owned device and memory placement. Placement records are immutable
 * metadata; equality across runtime identities does not make handles
 * interchangeable.
 *
 * @since 0.1.0
 * @category models
 */
export interface Placement {
  /** Stable identity for this placement within its runtime identity domain. */
  readonly id: string
  /** Backend-neutral device family, such as `cpu` or `metal`. */
  readonly deviceType: string
  /** Human-readable placement description for logs and diagnostics. */
  readonly description: string
  /** Optional device index when a runtime exposes multiple devices of one type. */
  readonly ordinal?: number
  /** Optional backend-defined memory-space identifier. */
  readonly memorySpace?: string
}

/**
 * Capabilities advertised by a runtime. These are discovery metadata, not a
 * substitute for operation-specific validation; backend methods remain
 * authoritative and may reject unsupported dtype/layout combinations.
 *
 * @since 0.1.0
 * @category models
 */
export interface Capabilities {
  /** Element data types accepted by this runtime. */
  readonly dtypes: ReadonlyArray<DType>
  /** Optional backend features advertised as stable string identifiers. */
  readonly features: ReadonlyArray<string>
}

/**
 * Structured failures reported by backend runtimes. Effect interruption may
 * remain an interruption rather than becoming a `BackendError`; `cancelled` is
 * available when a backend reports cancellation as an ordinary typed failure.
 *
 * @since 0.1.0
 * @category errors
 */
export class BackendError extends Data.TaggedError("BackendError")<{
  /** Machine-readable failure classification. */
  readonly reason:
    | "backend-unavailable"
    | "device-unavailable"
    | "unsupported-operation"
    | "unsupported-dtype"
    | "unsupported-layout"
    | "unsupported-placement"
    | "invalid-handle"
    | "foreign-handle"
    | "compilation-failed"
    | "execution-failed"
    | "transfer-failed"
    | "cancelled"
    | "closed-runtime"
    | "io-failed"
  /** Name of the backend that reported the failure. */
  readonly backend: string
  /** Runtime operation being performed when the failure occurred. */
  readonly operation: string
  /** Lifecycle phase in which the failure occurred. */
  readonly phase: "graph" | "autodiff" | "compile" | "execute" | "readback" | "io" | "shutdown"
  /** Human-readable failure description. */
  readonly message: string
  /** Optional backend-specific diagnostic payload. */
  readonly details?: unknown
  /** Inference phase, when the operation belongs to that interface. */
  readonly inferencePhase?: InferenceFailurePhase | undefined
}> {}

/** Internal nominal brand for all tensor handles. */
declare const TensorHandleTypeId: unique symbol
/** Internal nominal brand for lazy tensor handles. */
declare const LazyTensorHandleTypeId: unique symbol
/** Internal nominal brand for concrete tensor handles. */
declare const ConcreteTensorHandleTypeId: unique symbol
/** Internal nominal brand for compiled executable handles. */
declare const ExecutableHandleTypeId: unique symbol
/** Internal nominal brand for paged KV pool handles. */
declare const KvPoolHandleTypeId: unique symbol
/** Internal nominal brand for paged KV sequence handles. */
declare const KvSequenceHandleTypeId: unique symbol
/** Internal nominal brand for a native inference artifact. */
declare const InferenceArtifactHandleTypeId: unique symbol
/** Internal nominal brand for a native generation session. */
declare const InferenceSessionHandleTypeId: unique symbol
/** Internal nominal brand for a native generation sequence. */
declare const InferenceSequenceHandleTypeId: unique symbol

/**
 * A backend-owned tensor capability with backend-neutral immutable metadata.
 * `shape`, `dtype`, and optional `storage` describe the logical value and are
 * available without execution. Dense handles expose no public strides: graph
 * operations use logical row-major element order and a backend owns any
 * internal view/materialization decisions.
 *
 * The TypeScript brands prevent accidental structural use at compile time but
 * provide no runtime security. Every runtime method must also validate the
 * handle's registered owner, kind, and liveness rather than trusting `_tag`,
 * placement, or other public fields.
 *
 * @since 0.1.0
 * @category models
 */
export interface TensorHandle extends Pipeable {
  /** Nominal tensor-handle brand. */
  readonly [TensorHandleTypeId]: typeof TensorHandleTypeId
  /** Whether this handle is lazy or materialized. */
  readonly _tag: "LazyTensor" | "Tensor"
  /** Logical dimensions; `[]` is a scalar and zero extents are permitted. */
  readonly shape: ReadonlyArray<number>
  /** Tensor element data type. */
  readonly dtype: DType
  /** Omitted for dense storage; present for a packed logical `f32` value. */
  readonly storage?: EncodedTensorStorage | undefined
  /** Device family that owns the tensor. */
  readonly device: string
  /** Exact runtime placement that owns the tensor. */
  readonly placement: Placement
}

/**
 * A backend-owned lazy tensor value. It denotes an immutable semantic graph
 * and carries no materialized result ownership of its own. A graph may retain
 * concrete dependencies; releasing such a dependency invalidates later uses
 * of that graph unless compilation explicitly retained it as executable state.
 *
 * @since 0.1.0
 * @category models
 */
export interface LazyTensorHandle extends TensorHandle {
  /** Nominal lazy-tensor-handle brand. */
  readonly [LazyTensorHandleTypeId]: typeof LazyTensorHandleTypeId
  /** Discriminates lazy tensor handles. */
  readonly _tag: "LazyTensor"
}

/**
 * A backend-owned materialized tensor value. Each distinct returned handle
 * transfers cleanup responsibility to the caller, but release is idempotent and
 * may be requested repeatedly. Aliases may share physical storage, but releasing
 * one handle invalidates only that ownership capability; physical reclamation
 * can be delayed by aliases, exports, in-flight work, or allocator caches.
 *
 * @since 0.1.0
 * @category models
 */
export interface ConcreteTensorHandle extends TensorHandle {
  /** Nominal concrete-tensor-handle brand. */
  readonly [ConcreteTensorHandleTypeId]: typeof ConcreteTensorHandleTypeId
  /** Discriminates materialized tensor handles. */
  readonly _tag: "Tensor"
}

/**
 * Compilation controls that affect lowering, scheduling, and memory planning.
 *
 * @since 0.1.0
 * @category models
 */
export interface ExecutableCompileOptions {
  /**
   * Enables semantics-preserving fusion, epilogue, and optimizer regions.
   * Defaults to `true`. Optimization records code-generation choices beside
   * the semantic graph; it does not rewrite that graph. `false` uses the same
   * typed lowering, memory planner, and executor with optional regions disabled.
   */
  readonly optimize?: boolean | undefined
  /**
   * Authorizes inference-only retention of eligible materialized graph leaves
   * as executable constants. The executable, rather than the source handle,
   * retains the storage it needs. Bundled runtimes bypass structural
   * executable-cache reuse in this mode because captured values are not part
   * of a value-independent structural key. Defaults to `false`; do not enable
   * for values expected to vary between invocations.
   */
  readonly constantWeights?: boolean | undefined
}

/**
 * Bounded persistent state requested for a generation executable.
 *
 * @since 0.1.0
 * @category models
 */
export interface DecodeStateRequest {
  /** Positive unsigned 32-bit token-row capacity of the compatible KV pool. */
  readonly maxTokens: number
  /** Positive unsigned 32-bit paging unit that must divide `maxTokens`. */
  readonly blockSize: number
  /** KV storage dtype: `f32`, `f16`, `bf16`, or quantized `u8`. */
  readonly kvDtype: DType
  /**
   * Optional unsigned 32-bit global attention window in `1..=maxTokens`.
   * A completed schema retains it as the KV eviction window only when every
   * resolved attention operation is windowed.
   */
  readonly window?: number | undefined
  /**
   * Visibility of rows staged by the current invocation. `Causal` preserves
   * autoregressive row-by-row visibility. `Bidirectional` exposes the complete
   * current block, in addition to committed cache rows. Defaults to `Causal`.
   */
  readonly currentBlockAttention?: "Causal" | "Bidirectional" | undefined
  /** Positive unsigned 32-bit fixed compiled batch width. */
  readonly batch: number
  /**
   * Packed causal-chain verification layout. The traced graph has
   * `batch * rowsPerSequence` independent one-token rows while `batch` remains
   * the physical sequence width. Backends stage one explicit position per graph
   * row and expose all graph rows as outputs.
   */
  readonly packedCausalChains?: PackedCausalChainsLayout | undefined
  /**
   * When true, every root must be `[batch, T, V]` and the decode rewrite
   * returns native state-driven last-token selectors: one `[V]` root for
   * batch 1, otherwise `batch` `[V]` roots in row order. Defaults to false.
   */
  readonly lastTokenRow?: boolean | undefined
  /**
   * Root-indexed output policy. The array must have exactly one entry per
   * compile root. `splitLastTokenRow` preserves the legacy lane-split logits
   * outputs, `batchedLastTokenRow` emits one `[batch, V]` output, and `allRows`
   * preserves the root. Mutually exclusive with `lastTokenRow`; packed
   * causal-chain compilation accepts only `allRows`.
   */
  readonly outputSelections?: ReadonlyArray<DecodeOutputSelection> | undefined
}

/**
 * Row retention policy for one decode compile root.
 *
 * @since 0.1.0
 * @category models
 */
export type DecodeOutputSelection = "allRows" | "splitLastTokenRow" | "batchedLastTokenRow"

/**
 * Static row layout for packed causal-chain target verification.
 *
 * @since 0.1.0
 * @category models
 */
export interface PackedCausalChainsLayout {
  /** Positive number of verifier rows reserved for each physical sequence. */
  readonly rowsPerSequence: number
}

/**
 * Complete bounded state schema attached to a compiled executable.
 *
 * @since 0.1.0
 * @category models
 */
export interface DecodeStateSchema extends DecodeStateRequest {
  /** Number of attention layers backed by the KV pool. */
  readonly layers: number
  /** Number of key/value heads per attention layer. */
  readonly kvHeads: number
  /** Width of each key/value head. */
  readonly headDim: number
  /** Number of KDA recurrent layers with per-sequence state. */
  readonly kdaLayers: number
  /** Number of heads per KDA layer. */
  readonly kdaHeads: number
  /** Key width of each KDA head. */
  readonly kdaHeadDim: number
  /** Value width of each KDA head. */
  readonly kdaValueDim: number
  /** Number of short-conv layers with per-sequence window state. */
  readonly convLayers: number
  /** Channel count of each short-conv layer. */
  readonly convChannels: number
  /** Kernel size of each short-conv layer. */
  readonly convKernel: number
}

/**
 * One lowered instruction category in executable diagnostics.
 *
 * @since 0.1.0
 * @category models
 */
export interface ExecutableInstructionDiagnostics {
  /** Backend-defined lowered instruction name. */
  readonly kind: string
  /** Number of lowered instructions in this category. */
  readonly count: number
}

/**
 * One observational wall-clock measurement from executable compilation.
 *
 * @since 0.1.0
 * @category models
 */
export interface ExecutableCompilePhaseDiagnostics {
  /** Stable compiler phase name, with optional backend-specific additions. */
  readonly phase: string
  /** Elapsed wall-clock time for the phase. */
  readonly nanoseconds: number
}

/**
 * Static logical byte totals derived from an executable's immutable memory
 * plan. These are not current allocation, allocator capacity, or process RSS.
 *
 * @since 0.1.0
 * @category models
 */
export interface ExecutableMemoryDiagnostics {
  /** Borrowed invocation or generated-input storage referenced by the plan. */
  readonly externalBytes: number
  /** Storage retained by executable constants. */
  readonly persistentBytes: number
  /** Compatible persistent state footprint used by a stateful executable. */
  readonly stateBytes: number
  /** Logical escaping-output storage required by one invocation. */
  readonly outputBytes: number
  /** Reusable per-invocation workspace and staging capacity. */
  readonly workspaceBytes: number
  /** Rollback and commit staging for mutable state. */
  readonly transactionBytes: number
  /** Peak simultaneous logical bytes among planned values. */
  readonly peakLiveBytes: number
  /** Planned segment capacity beyond peak logical liveness. */
  readonly packingOverheadBytes: number
}

/**
 * Static compilation and planning summary attached to an executable.
 * Structural counters and memory values describe the artifact;
 * `compilePhases` contains nondeterministic timing observations and does not
 * participate in cache identity. A structural-cache hit retains the original
 * artifact's timings rather than measuring the current `compile` call.
 *
 * @since 0.1.0
 * @category models
 */
export interface ExecutableDiagnostics {
  /** Number of semantic graph nodes presented to optimization. */
  readonly semanticNodesBeforeOptimization: number
  /**
   * Number of semantic nodes after identity-preserving planning. This currently
   * equals `semanticNodesBeforeOptimization`; lowered counts expose fusion.
   */
  readonly semanticNodesAfterOptimization: number
  /** Lowered instruction counts grouped by backend-defined kind. */
  readonly instructions: ReadonlyArray<ExecutableInstructionDiagnostics>
  /** Number of prepared backend pipelines, when applicable. */
  readonly pipelineCount: number
  /** Number of physical encode commands in the executable. */
  readonly commandCount: number
  /** Number of backend physical completion boundaries; Metal status gates and commits are excluded. */
  readonly synchronizationCount: number
  /** Static logical memory-plan totals. */
  readonly memory: ExecutableMemoryDiagnostics
  /** Ordered compiler timings; third-party runtimes may omit or extend them. */
  readonly compilePhases?: ReadonlyArray<ExecutableCompilePhaseDiagnostics>
}

/**
 * Opaque backend-owned executable plus its optional public state contract.
 * Executables are immutable and may be invoked concurrently. Stateful
 * invocations must nevertheless use disjoint sequence handles. There is no
 * common explicit release operation; implementations must finalize the native
 * wrapper after executable and cache references become unreachable.
 *
 * @since 0.1.0
 * @category models
 */
export interface ExecutableHandle {
  /** Nominal executable-handle brand. */
  readonly [ExecutableHandleTypeId]: typeof ExecutableHandleTypeId
  /** Complete state schema for generation executables. */
  readonly state?: DecodeStateSchema | undefined
  /** Immutable lowering and static-memory summary. */
  readonly diagnostics: ExecutableDiagnostics
}

/**
 * One semantic graph compilation request. Compilation borrows the roots but
 * may retain graph artifacts and any storage authorized by the options in the
 * resulting executable. Implementations may use a bounded structural cache;
 * root order, duplicate roots, options, state, and graph semantics all affect
 * observable output ordering or cache identity.
 *
 * @since 0.1.0
 * @category models
 */
export interface CompileRequest {
  /**
   * Nonempty semantic roots owned by this runtime and belonging to one exact
   * placement. Root order, including duplicates, defines executable output
   * order.
   */
  readonly roots: ReadonlyArray<TensorHandle>
  /** Explicit controls that join the executable cache key. */
  readonly options?: ExecutableCompileOptions | undefined
  /** Optional bounded persistent-state contract. */
  readonly state?: DecodeStateRequest | undefined
}

/**
 * Per-invocation state supplied to a stateful executable. The listed sequences
 * are mutably borrowed until the invocation finishes and must be distinct and
 * absent from every concurrent invocation or sequence operation.
 *
 * @since 0.1.0
 * @category models
 */
export interface ExecutionStateInvocation {
  /**
   * From `1` through the compiled batch width, distinct live sequences from
   * one schema-compatible pool, listed in API result order.
   */
  readonly sequences: ReadonlyArray<KvSequenceHandle>
  /**
   * One distinct physical executable slot per sequence. Slots are integers in
   * `[0, batch)`. Array order may differ from slot order and does not move
   * sequence state between lanes.
   */
  readonly slots: ReadonlyArray<number>
  /** Fixed-width activity mask; true entries must exactly equal `slots`. */
  readonly activeMask: ReadonlyArray<boolean>
  /** Fixed-width real-row count; inactive slots are zero. */
  readonly validLengths: ReadonlyArray<number>
  /** Fixed-width committed advances; Phase 1 requires `advances === validLengths`. */
  readonly advances: ReadonlyArray<number>
  /**
   * One nonempty row of unsigned 32-bit token ids per sequence. Rows may have
   * different lengths; each length matches that physical lane's valid length.
   * Unlisted slots have zero advance. Success commits state and advances
   * cursors atomically across all rows; failure or interruption before commit
   * rolls every row back.
   */
  readonly tokens: ReadonlyArray<ReadonlyArray<number>>
}

/**
 * Complete dynamic input to one immutable executable invocation. Collections
 * describe a single call and must not be mutated while its Effect is running.
 *
 * @since 0.1.0
 * @category models
 */
export interface ExecutionInvocation {
  /**
   * Materialized tensor bindings in ascending shared-slot order with scalar
   * slots omitted. Counts, metadata, layout, ownership, and placement must
   * match the compiled declarations. They are borrowed, not consumed, and
   * must not be released until the invocation completes.
   */
  readonly bindings: ReadonlyArray<ConcreteTensorHandle>
  /** Scalar bindings in ascending shared-slot order with tensor slots omitted. */
  readonly scalars: ReadonlyArray<number>
  /**
   * Named bounded values used by a fixed schedule. Bundled CPU and Metal
   * runtimes currently require this record to be empty for public invocations.
   */
  readonly runtimeValues: Readonly<Record<string, number | Uint32Array>>
  /** Stateful generation invocation, omitted for ordinary programs. */
  readonly state?: ExecutionStateInvocation | undefined
}

/**
 * Opaque backend-owned paged KV cache pool. A pool owns fixed-capacity storage
 * shared by its sequences. The common protocol has no explicit pool release;
 * it becomes finalizable only after the pool and all child sequence handles are
 * unreachable or released as applicable.
 *
 * @since 0.1.0
 * @category models
 */
export interface KvPoolHandle {
  /** Nominal KV-pool-handle brand. */
  readonly [KvPoolHandleTypeId]: typeof KvPoolHandleTypeId
}

/**
 * Opaque backend-owned mutable sequence allocated from a paged KV pool. It
 * retains its parent pool and must be released exactly once after its final
 * operation; finalization is only a fallback.
 *
 * @since 0.1.0
 * @category models
 */
export interface KvSequenceHandle {
  /** Nominal KV-sequence-handle brand. */
  readonly [KvSequenceHandleTypeId]: typeof KvSequenceHandleTypeId
}

/**
 * Opaque backend-owned inference artifact.
 *
 * @since 0.1.0
 * @category models
 */
export interface InferenceArtifactHandle {
  /** Nominal inference-artifact brand. */
  readonly [InferenceArtifactHandleTypeId]: typeof InferenceArtifactHandleTypeId
}

/**
 * Opaque backend-owned mutable generation session.
 *
 * @since 0.1.0
 * @category models
 */
export interface InferenceSessionHandle {
  /** Nominal inference-session brand. */
  readonly [InferenceSessionHandleTypeId]: typeof InferenceSessionHandleTypeId
}

/**
 * Opaque backend-owned sequence within one generation session.
 *
 * @since 0.1.0
 * @category models
 */
export interface InferenceSequenceHandle {
  /** Nominal inference-sequence brand. */
  readonly [InferenceSequenceHandleTypeId]: typeof InferenceSequenceHandleTypeId
}

/**
 * Inputs and attributes for every semantic graph operation. Requests are
 * backend-neutral descriptions, not instructions to execute a kernel. A
 * runtime validates every input capability, snapshots mutable attribute data
 * where documented, and returns a lazy node with authoritative logical
 * metadata. Compilation later performs typed lowering and layout selection.
 *
 * @since 0.1.0
 * @category models
 */
export interface NodeOperationMap {
  /** Creates a scalar constant, optionally matching an exemplar placement. */
  readonly constant: {
    readonly inputs: readonly [] | readonly [exemplar: TensorHandle]
    readonly attributes: { readonly value: number; readonly dtype: DType }
  }
  /** Creates a zero-filled tensor. */
  readonly zeros: {
    readonly inputs: readonly [] | readonly [exemplar: TensorHandle]
    readonly attributes: { readonly shape: ReadonlyArray<number>; readonly dtype: DType }
  }
  /** Creates a one-filled tensor. */
  readonly ones: {
    readonly inputs: readonly [] | readonly [exemplar: TensorHandle]
    readonly attributes: { readonly shape: ReadonlyArray<number>; readonly dtype: DType }
  }
  /** Creates a tensor filled with one scalar value. */
  readonly full: {
    readonly inputs: readonly [] | readonly [exemplar: TensorHandle]
    readonly attributes: { readonly shape: ReadonlyArray<number>; readonly value: number; readonly dtype: DType }
  }
  /** Creates a tensor sampled from a standard normal distribution. */
  readonly randn: {
    readonly inputs: readonly []
    readonly attributes: { readonly shape: ReadonlyArray<number>; readonly dtype: DType }
  }
  /** Creates a tensor sampled uniformly from `[lo, hi)`. */
  readonly uniform: {
    readonly inputs: readonly []
    readonly attributes: {
      readonly shape: ReadonlyArray<number>
      readonly lo: number
      readonly hi: number
      readonly dtype: DType
    }
  }
  /** Creates a one-dimensional arithmetic progression. */
  readonly arange: {
    readonly inputs: readonly []
    readonly attributes: { readonly start: number; readonly end: number; readonly step: number; readonly dtype: DType }
  }
  /** Creates a square identity matrix. */
  readonly eye: {
    readonly inputs: readonly []
    readonly attributes: { readonly n: number; readonly dtype: DType }
  }
  /** Imports a host byte snapshot as a tensor. */
  readonly fromBytes: {
    readonly inputs: readonly []
    /** The backend snapshots `data`; the caller retains ownership. */
    readonly attributes: { readonly data: Uint8Array; readonly shape: ReadonlyArray<number>; readonly dtype: DType }
  }
  /** Declares a tensor input slot in a compiled program. */
  readonly input: {
    readonly inputs: readonly [] | readonly [exemplar: TensorHandle]
    readonly attributes: {
      readonly slot: number
      readonly shape: ReadonlyArray<number>
      readonly dtype: DType
      readonly storage?: EncodedTensorStorage | undefined
    }
  }
  /** Declares a scalar input slot in a compiled program. */
  readonly scalarInput: {
    readonly inputs: readonly []
    readonly attributes: { readonly slot: number; readonly dtype: DType }
  }
  /** Adds two tensors elementwise with broadcasting. */
  readonly add: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  /** Subtracts the second tensor from the first elementwise. */
  readonly sub: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  /** Multiplies two tensors elementwise with broadcasting. */
  readonly mul: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  /** Divides the first tensor by the second elementwise. */
  readonly div: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  /** Selects the elementwise maximum of two tensors. */
  readonly maximum: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  /** Selects the elementwise minimum of two tensors. */
  readonly minimum: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  /** Compares two tensors for elementwise equality. */
  readonly eq: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  /** Compares whether the first tensor is elementwise greater than the second. */
  readonly gt: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  /** Compares whether the first tensor is elementwise less than the second. */
  readonly lt: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  /** Compares whether the first tensor is elementwise greater than or equal to the second. */
  readonly ge: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  /** Compares whether the first tensor is elementwise less than or equal to the second. */
  readonly le: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  /** Performs batched matrix multiplication. */
  readonly matmul: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  /** Solves a linear system with the first tensor as its coefficient matrix. */
  readonly solve: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  /** Concatenates two tensors along one dimension. */
  readonly concat: {
    readonly inputs: readonly [self: TensorHandle, other: TensorHandle]
    readonly attributes: { readonly dim: number }
  }
  /** Negates every tensor element. */
  readonly neg: { readonly inputs: readonly [self: TensorHandle] }
  /** Computes the elementwise absolute value. */
  readonly abs: { readonly inputs: readonly [self: TensorHandle] }
  /** Computes the elementwise square root. */
  readonly sqrt: { readonly inputs: readonly [self: TensorHandle] }
  /** Computes the elementwise exponential. */
  readonly exp: { readonly inputs: readonly [self: TensorHandle] }
  /** Computes the elementwise natural logarithm. */
  readonly log: { readonly inputs: readonly [self: TensorHandle] }
  /** Computes the elementwise sine. */
  readonly sin: { readonly inputs: readonly [self: TensorHandle] }
  /** Computes the elementwise cosine. */
  readonly cos: { readonly inputs: readonly [self: TensorHandle] }
  /** Computes the elementwise hyperbolic tangent. */
  readonly tanh: { readonly inputs: readonly [self: TensorHandle] }
  /** Applies the elementwise rectified linear unit. */
  readonly relu: { readonly inputs: readonly [self: TensorHandle] }
  /** Computes the elementwise error function. */
  readonly erf: { readonly inputs: readonly [self: TensorHandle] }
  /** Rounds every element down to the nearest integer. */
  readonly floor: { readonly inputs: readonly [self: TensorHandle] }
  /** Rounds every element up to the nearest integer. */
  readonly ceil: { readonly inputs: readonly [self: TensorHandle] }
  /** Rounds every element to the nearest integer. */
  readonly round: { readonly inputs: readonly [self: TensorHandle] }
  /** Returns the elementwise sign. */
  readonly sign: { readonly inputs: readonly [self: TensorHandle] }
  /** Computes the inverse of square matrices. */
  readonly inverse: { readonly inputs: readonly [self: TensorHandle] }
  /** Computes the determinant of square matrices. */
  readonly det: { readonly inputs: readonly [self: TensorHandle] }
  /** Preserves the value while stopping reverse-mode gradient propagation. */
  readonly stopGradient: { readonly inputs: readonly [self: TensorHandle] }
  /**
   * Identity node carrying a stable exposure name. The wrapped value stays
   * in the graph and is discoverable through {@link RuntimeService.exposures};
   * compilation lowers it to a zero-cost alias.
   */
  readonly expose: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly name: string }
  }
  /** Marks a value for recomputation during reverse-mode differentiation. */
  readonly checkpoint: { readonly inputs: readonly [self: TensorHandle] }
  /** Applies the Gaussian error linear unit. */
  readonly gelu: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly approximate?: boolean | null }
  }
  /** Raises every element to a scalar exponent. */
  readonly pow: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly exponent: number }
  }
  /** Converts tensor elements to another data type. */
  readonly cast: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly dtype: DType }
  }
  /** Selects elements from two tensors according to a condition tensor. */
  readonly whereCond: {
    readonly inputs: readonly [condition: TensorHandle, a: TensorHandle, b: TensorHandle]
  }
  /** Returns indices of maximum values along one dimension. */
  readonly argmax: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly dim: number }
  }
  /** Returns indices of minimum values along one dimension. */
  readonly argmin: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly dim: number }
  }
  /** Computes a cumulative sum along one dimension. */
  readonly cumsum: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly dim: number }
  }
  /** Selects slices using a one-dimensional index tensor. */
  readonly indexSelect: {
    readonly inputs: readonly [self: TensorHandle, indexes: TensorHandle]
    readonly attributes: { readonly dim: number }
  }
  /** Adds source values into indexed positions of the input tensor. */
  readonly scatterAdd: {
    readonly inputs: readonly [self: TensorHandle, indexes: TensorHandle, src: TensorHandle]
    readonly attributes: { readonly dim: number }
  }
  /** Gathers values according to an index tensor. */
  readonly gather: {
    readonly inputs: readonly [self: TensorHandle, indexes: TensorHandle]
    readonly attributes: { readonly dim: number }
  }
  /** Computes cross-entropy loss from logits and target indices. */
  readonly crossEntropy: {
    readonly inputs: readonly [self: TensorHandle, target: TensorHandle]
    readonly attributes: { readonly ignoreIndex: number }
  }
  /** Computes scaled dot-product attention. */
  readonly scaledDotProductAttention: {
    readonly inputs: readonly [q: TensorHandle, k: TensorHandle, v: TensorHandle]
    readonly attributes: {
      readonly scale: number
      readonly causal: boolean
      readonly window?: number | null | undefined
    }
  }
  /** Computes Kimi Delta Attention (gated delta-rule linear attention) in chunked form. */
  readonly kdaChunk: {
    readonly inputs: readonly [
      q: TensorHandle,
      k: TensorHandle,
      v: TensorHandle,
      logDecay: TensorHandle,
      beta: TensorHandle
    ]
    readonly attributes: { readonly scale: number }
  }
  /** Applies a causal depthwise short convolution over `[..., T, C]` with zero history. */
  readonly shortConv1d: {
    readonly inputs: readonly [self: TensorHandle, weight: TensorHandle]
    readonly attributes: Record<string, never>
  }
  /** Selects a prefix from a learned position-embedding table. */
  readonly positionEmbedding: {
    readonly inputs: readonly [weight: TensorHandle]
    readonly attributes: { readonly seqLen: number }
  }
  /** Applies rotary position embeddings to an attention tensor. */
  readonly rotaryEmbedding: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: {
      readonly seqLen: number
      readonly theta: number
      readonly layout: "HalfSplit" | "InterleavedPairs"
    }
  }
  /** Normalizes the trailing dimensions and applies affine parameters. */
  readonly layerNorm: {
    readonly inputs: readonly [self: TensorHandle, weight: TensorHandle, bias: TensorHandle]
    readonly attributes: { readonly eps: number }
  }
  /** Applies RMS normalization over the last dimension, optionally with a scale. */
  readonly rmsNorm: {
    readonly inputs: readonly [self: TensorHandle] | readonly [self: TensorHandle, weight: TensorHandle]
    readonly attributes: { readonly eps: number }
  }
  /** Applies a linear projection with a bias. */
  readonly linear: {
    readonly inputs: readonly [self: TensorHandle, weight: TensorHandle, bias: TensorHandle]
  }
  /** Applies a row-oriented packed linear projection, with an optional dense bias. */
  readonly quantizedLinear: {
    readonly inputs:
      | readonly [self: TensorHandle, weight: TensorHandle]
      | readonly [self: TensorHandle, weight: TensorHandle, bias: TensorHandle]
    readonly attributes: {
      readonly encoding: TensorStorageEncoding
      readonly logicalShape: readonly [rows: number, columns: number]
    }
  }
  /** Selects and decodes rows from a packed embedding table. */
  readonly quantizedEmbedding: {
    readonly inputs: readonly [indexes: TensorHandle, weight: TensorHandle]
    readonly attributes: {
      readonly encoding: TensorStorageEncoding
      readonly logicalShape: readonly [rows: number, columns: number]
      readonly paddingIndex?: number | undefined
    }
  }
  /** Applies a one-dimensional grouped convolution. */
  readonly conv1d: {
    readonly inputs: readonly [self: TensorHandle, weight: TensorHandle]
    readonly attributes: {
      readonly stride: number
      readonly padding: number
      readonly dilation: number
      readonly groups: number
    }
  }
  /** Applies a two-dimensional grouped convolution. */
  readonly conv2d: {
    readonly inputs: readonly [self: TensorHandle, weight: TensorHandle]
    readonly attributes: {
      readonly stride: number
      readonly padding: number
      readonly dilation: number
      readonly groups: number
    }
  }
  /** Sums elements over selected dimensions. */
  readonly sum: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly dims: ReadonlyArray<number>; readonly keepdims: boolean }
  }
  /** Multiplies elements over selected dimensions. */
  readonly prod: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly dims: ReadonlyArray<number>; readonly keepdims: boolean }
  }
  /** Averages elements over selected dimensions. */
  readonly mean: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly dims: ReadonlyArray<number>; readonly keepdims: boolean }
  }
  /** Selects maximum values over selected dimensions. */
  readonly max: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly dims: ReadonlyArray<number>; readonly keepdims: boolean }
  }
  /** Selects minimum values over selected dimensions. */
  readonly min: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly dims: ReadonlyArray<number>; readonly keepdims: boolean }
  }
  /** Changes tensor dimensions without changing element order. */
  readonly reshape: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly shape: ReadonlyArray<number> }
  }
  /** Reorders tensor dimensions. */
  readonly permute: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly dims: ReadonlyArray<number> }
  }
  /** Selects strided ranges from every tensor dimension. */
  readonly slice: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly ranges: ReadonlyArray<ReadonlyArray<number>> }
  }
  /** Broadcasts a tensor to a compatible target shape. */
  readonly broadcastTo: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly shape: ReadonlyArray<number> }
  }
  /** Maps the function implicit in `y` over an inserted batch dimension of `batchedX`. */
  readonly vmap: {
    readonly inputs: readonly [y: TensorHandle, x: TensorHandle, batchedX: TensorHandle]
    readonly attributes: { readonly dim: number }
  }
  /** Computes one fused AdamW parameter and optimizer-state update. */
  readonly adamwStep: {
    readonly inputs: readonly [
      param: TensorHandle,
      grad: TensorHandle,
      m: TensorHandle,
      v: TensorHandle,
      lr: TensorHandle,
      c1: TensorHandle,
      c2: TensorHandle
    ]
    readonly attributes: {
      readonly beta1: number
      readonly beta2: number
      readonly eps: number
      readonly weightDecay: number
    }
  }
  /** Selects one output from a fused AdamW update. */
  readonly adamwOut: {
    readonly inputs: readonly [step: TensorHandle]
    readonly attributes: { readonly index: number }
  }
  /** Computes one fused SGD parameter and velocity update. */
  readonly sgdStep: {
    readonly inputs: readonly [
      param: TensorHandle,
      grad: TensorHandle,
      velocity: TensorHandle,
      first: TensorHandle,
      lr: TensorHandle
    ]
    readonly attributes: {
      readonly momentum: number
      readonly dampening: number
      readonly nesterov: boolean
      readonly weightDecay: number
    }
  }
  /** Selects one output from a fused SGD update. */
  readonly sgdOut: {
    readonly inputs: readonly [step: TensorHandle]
    readonly attributes: { readonly index: number }
  }
}

/**
 * A type-checked semantic graph construction request. The mapped union checks
 * operation arity and attributes for implementors; runtime validation is still
 * required for untyped JavaScript and native boundaries.
 *
 * @since 0.1.0
 * @category models
 */
export type NodeRequest<Operation extends keyof NodeOperationMap = keyof NodeOperationMap> = {
  readonly [K in Operation]: { readonly op: K } & NodeOperationMap[K]
}[Operation]

/**
 * A named tensor entry for a direct safetensors write.
 *
 * @since 0.1.0
 * @category models
 */
export interface PathSafetensorsSaveEntry {
  /** Archive entry name. */
  readonly name: string
  /** Borrowed dense materialized tensor to serialize; saving does not release it. */
  readonly tensor: ConcreteTensorHandle
}

/**
 * A named materialized entry returned by a direct safetensors read.
 *
 * @since 0.1.0
 * @category models
 */
export interface PathSafetensorsLoadEntry {
  /** Archive entry name. */
  readonly name: string
  /** Caller-owned tensor loaded into the runtime placement. */
  readonly tensor: ConcreteTensorHandle
}

/**
 * Tensors and string metadata supplied to a direct safetensors write.
 *
 * @since 0.1.0
 * @category models
 */
export interface PathSafetensorsSaveArchive {
  /** Named materialized tensors to serialize. */
  readonly entries: ReadonlyArray<PathSafetensorsSaveEntry>
  /** Archive-level string metadata. */
  readonly metadata: Readonly<Record<string, string>>
}

/**
 * Tensors and string metadata returned by a direct safetensors read.
 *
 * @since 0.1.0
 * @category models
 */
export interface PathSafetensorsLoadArchive {
  /** Named tensors materialized by the runtime. */
  readonly entries: ReadonlyArray<PathSafetensorsLoadEntry>
  /** Archive-level string metadata. */
  readonly metadata: Readonly<Record<string, string>>
}

/**
 * Optional runtime extension for direct path-based safetensors I/O. Extension
 * methods are part of the same ownership domain as their parent runtime and
 * must apply the same native handle owner/liveness checks as common methods.
 *
 * @since 0.1.0
 * @category models
 */
export interface PathSafetensors {
  /**
   * Writes a borrowed dense archive without transferring tensor data through
   * JavaScript. Input handles remain caller-owned and usable after completion.
   */
  readonly save: (path: string, archive: PathSafetensorsSaveArchive) => Effect.Effect<void, BackendError>
  /**
   * Reads directly into materialized runtime tensors. On success, ownership of
   * every distinct returned handle transfers to the caller. On failure or
   * interruption, the implementation releases all partial and late results.
   */
  readonly load: (path: string) => Effect.Effect<PathSafetensorsLoadArchive, BackendError>
}

/**
 * A scalar GGUF metadata value after native validation.
 *
 * @since 0.1.0
 * @category models
 */
export type GgufMetadataScalar = number | string | boolean

/**
 * One ordered GGUF metadata entry.
 *
 * @since 0.1.0
 * @category models
 */
export interface GgufMetadataEntry {
  /** Original GGUF metadata key. */
  readonly key: string
  /** Validated scalar or homogeneous scalar-array payload. */
  readonly value: GgufMetadataScalar | ReadonlyArray<GgufMetadataScalar>
}

/**
 * Backend-neutral tensor catalog entry returned by native GGUF inspection.
 * Logical shape order is the tensor API order; physical shape describes the
 * bytes loaded into a concrete handle and can differ for packed formats.
 *
 * @since 0.1.0
 * @category models
 */
export interface GgufTensorDescriptor {
  /** Archive tensor name. */
  readonly name: string
  /** Dense or packed on-disk representation supported by this protocol. */
  readonly format: "F32" | TensorStorageEncoding
  /** Logical dimensions visible to tensor operations. */
  readonly logicalShape: ReadonlyArray<number>
  /** Packed values decode to logical `f32`. */
  readonly logicalDtype: "f32"
  /** Dense element shape or packed byte shape owned by the native tensor. */
  readonly physicalShape: ReadonlyArray<number>
  /** Dense F32 storage uses `f32`; packed storage uses `u8`. */
  readonly physicalDtype: "f32" | "u8"
}

/**
 * GGUF metadata and tensor descriptors without tensor payloads.
 *
 * @since 0.1.0
 * @category models
 */
export interface GgufInspection {
  /** Ordered validated metadata. */
  readonly metadata: ReadonlyArray<GgufMetadataEntry>
  /** Ordered tensor catalog without loaded storage. */
  readonly tensors: ReadonlyArray<GgufTensorDescriptor>
}

/**
 * One runtime-owned tensor returned by native GGUF loading.
 *
 * @since 0.1.0
 * @category models
 */
export interface GgufLoadEntry {
  /** Descriptor whose logical and physical metadata the tensor must match. */
  readonly descriptor: GgufTensorDescriptor
  /** Caller-owned concrete handle after successful archive completion. */
  readonly tensor: ConcreteTensorHandle
}

/**
 * Runtime-owned tensors returned by native GGUF loading.
 *
 * @since 0.1.0
 * @category models
 */
export interface GgufLoadArchive {
  /** Distinct owned handles in archive order. */
  readonly entries: ReadonlyArray<GgufLoadEntry>
}

/**
 * Optional native GGUF parser and loader extension. Inspection owns no tensor
 * storage. Loading follows the parent runtime's cancellation and ownership
 * rules.
 *
 * @since 0.1.0
 * @category models
 */
export interface GgufRuntime {
  /** Parses and validates metadata and tensor descriptors without loading payloads. */
  readonly inspect: (path: string) => Effect.Effect<GgufInspection, BackendError>
  /**
   * Loads runtime-owned tensors. The implementation owns every handle until
   * successful Effect completion and must release partial or late results when
   * interrupted; ownership transfers to the caller with the returned archive.
   */
  readonly load: (path: string) => Effect.Effect<GgufLoadArchive, BackendError>
}

/**
 * Normalized controls for one stateless native token draw from a logits row.
 * `topK = 0` disables top-k filtering. The same seed and counter replay the
 * same draw on a given backend.
 *
 * @since 0.1.0
 * @category models
 */
export interface SamplingOptions {
  /** Non-negative temperature; zero selects greedy sampling. */
  readonly temperature: number
  /** Top-k candidate count, or zero to disable top-k filtering. */
  readonly topK: number
  /** Cumulative probability threshold in `(0, 1]`. */
  readonly topP: number
  /** Non-negative deterministic sampler seed. */
  readonly seed: number
  /** Non-negative deterministic draw counter. */
  readonly counter: number
}

/**
 * Lossless normalized controls used by native inference.
 *
 * @since 0.1.0
 * @category models
 */
export interface InferenceSamplingOptions {
  /** Non-negative temperature; zero selects greedy sampling. */
  readonly temperature: number
  /** Top-k candidate count, or zero to disable top-k filtering. */
  readonly topK: number
  /** Cumulative probability threshold in `(0, 1]`. */
  readonly topP: number
  /** Unsigned 64-bit seed; it must never be folded through a JavaScript number. */
  readonly seed: bigint
}

/** Per-round or per-sequence overrides of native inference sampling controls. */
export interface InferenceSamplingOverrides {
  readonly temperature?: number | undefined
  readonly topK?: number | undefined
  readonly topP?: number | undefined
  readonly seed?: bigint | undefined
}

/**
 * A phase that can fail without partially publishing an inference round.
 *
 * @since 0.1.0
 * @category models
 */
export type InferenceFailurePhase =
  | "compile"
  | "open"
  | "admission"
  | "prefill"
  | "proposer"
  | "verify"
  | "sample"
  | "accept"
  | "publish"
  | "finish"
  | "close"
  | "inspect"

/**
 * Programs and state pools bundled into one native inference artifact.
 *
 * @since 0.1.0
 * @category models
 */
export interface InferenceCompileRequest {
  /** Target-model programs and their compatible state pool. */
  readonly target: {
    /**
     * Target prompt-prefill executables, one per compiled chunk shape in
     * `prefillChunks` order (ascending token width; the last entry is the
     * largest). Every entry shares the decode program's state geometry and
     * `pool`. The runtime serves each prompt chunk from the largest
     * executable covering its remaining length and skips the LM-head chain
     * for chunks that do not finish a prompt; backends without chunk
     * selection may serve every prompt from the last (largest) entry.
     */
    readonly prefill: ReadonlyArray<ExecutableHandle>
    /** Target one-token decode executable. */
    readonly decode: ExecutableHandle
    /**
     * Packed all-row verifiers, one per compiled rows-per-sequence width and
     * ascending (widest last); omitted for the zero-draft ordinary path. Exact
     * proposers use the widest; generalized plans adapt the width per round
     * from measured token rates.
     */
    readonly verify?: ReadonlyArray<ExecutableHandle> | undefined
    /** State pool compatible with every supplied target executable. */
    readonly pool: KvPoolHandle
  }
  /** Optional exact autoregressive proposer programs and state pool. */
  readonly proposer?: {
    /** Proposer prompt-prefill executable. */
    readonly prefill: ExecutableHandle
    /** Proposer one-token decode executable. */
    readonly decode: ExecutableHandle
    /** State pool compatible with both proposer executables. */
    readonly pool: KvPoolHandle
    /** Maximum draft length compiled into the proposer contract. */
    readonly maxDraftTokens: number
  } | undefined
  /** Optional backend-neutral generalized proposer schedule. */
  readonly generalizedProposer?: {
    /** Fully resolved routing and stage schedule. */
    readonly plan: InferenceProposerPlan
    /** Borrowed target weights shared with proposer stages. */
    readonly sharedTensors: ReadonlyArray<ConcreteTensorHandle>
    /** Stage executables in `plan.stages` order. */
    readonly stageExecutables: ReadonlyArray<ExecutableHandle>
    /** Optional autoregressive state replay programs. */
    readonly replay?: {
      /**
       * Replay prompt-prefill executables aligned with `target.prefill`:
       * entry N replays the target hidden taps of chunk shape N.
       */
      readonly prefill: ReadonlyArray<ExecutableHandle>
      /** Replay one-token decode executable. */
      readonly decode: ExecutableHandle
      /** Replay packed verification executables, one per verify width, ascending. */
      readonly verify: ReadonlyArray<ExecutableHandle>
      /** State pool compatible with all replay executables. */
      readonly pool: KvPoolHandle
    }
    /** Maximum candidate count produced by the generalized proposer. */
    readonly maxDraftTokens: number
  } | undefined
  /** Fixed physical sequence width of the artifact. */
  readonly batchSize: number
  /** Integer dtype used by token tensors. */
  readonly tokenDtype: "u32" | "i64"
  /** Default sampling policy copied into newly admitted sequences. */
  readonly sampling: InferenceSamplingOptions
}

/**
 * Fully resolved logical schema of one value routed by an inference artifact.
 *
 * @since 0.1.0
 * @category models
 */
export interface InferenceValueSchema {
  /** Logical element dtype. */
  readonly dtype: DType
  /** Fully resolved logical dimensions. */
  readonly shape: ReadonlyArray<number>
}

/**
 * Source and optional projection of one ordered inference-stage input.
 *
 * @since 0.1.0
 * @category models
 */
export interface InferenceValueRoute {
  /** Runtime source from which the value is routed. */
  readonly kind:
    | "PendingTokens"
    | "CandidatePrefix"
    | "CommittedHistory"
    | "TargetHidden"
    | "SharedTokenEmbedding"
    | "SharedLmHead"
    | "StageOutput"
  /** Target executable root for a `TargetHidden` route. */
  readonly targetOutput?: number
  /** Source stage index for a `StageOutput` route. */
  readonly stage?: number
  /** Output index within the source target or proposer stage. */
  readonly output?: number
  /** Expected logical schema after routing. */
  readonly value?: InferenceValueSchema
  /** Whether routing selects only the target row needed by this stage. */
  readonly selectTargetRow?: boolean
}

/**
 * One hidden activation exported by each target program for native routing.
 *
 * @since 0.1.0
 * @category models
 */
export interface InferenceTargetTapRoute {
  /** Name of the target-model exposure represented by this tap. */
  readonly name: string
  /** Semantic source-root index; backends resolve any lane-split outputs before this root. */
  readonly outputRoot: number
  /** Logical schema after any runtime row selection. */
  readonly value: InferenceValueSchema
}

/**
 * Complete generalized proposer schedule consumed by the native compiler.
 * Routes are backend-neutral and indexes refer to the ordered arrays in this
 * plan and its enclosing {@link InferenceCompileRequest}.
 *
 * @since 0.1.0
 * @category models
 */
export interface InferenceProposerPlan {
  /** Target vocabulary size. */
  readonly vocabulary: number
  /** Stable fingerprint of the resolved proposer-to-target token map. */
  readonly tokenMapFingerprint: string
  /** Target hidden taps used during ordinary decode. */
  readonly hiddenTaps: ReadonlyArray<InferenceTargetTapRoute>
  /** Phase-specific target hidden taps used during prefill. */
  readonly prefillHiddenTaps?: ReadonlyArray<InferenceTargetTapRoute>
  /** Phase-specific target hidden taps used during verification. */
  readonly verifyHiddenTaps?: ReadonlyArray<InferenceTargetTapRoute>
  /** Target tensors borrowed and routed into proposer stages. */
  readonly sharedTensors: ReadonlyArray<{
    /** Semantic role of the shared target tensor. */
    readonly kind: "TokenEmbedding" | "LmHead"
    /** Exact target parameter name. */
    readonly name: string
    /** Required logical tensor schema. */
    readonly value: InferenceValueSchema
  }>
  /** Ordered proposer operations and their value routes. */
  readonly stages: ReadonlyArray<{
    /** Stable backend-neutral operation identifier. */
    readonly operationId: string
    /** Optional operation-specific layout contract. */
    readonly layoutId?: string
    /** Complete backend-neutral configuration of a deterministic history intrinsic. */
    readonly historyLookup?: {
      /** Identifies the suffix n-gram intrinsic version. */
      readonly id: "suffix-ngram-v1"
      /** Minimum committed suffix length eligible for a match. */
      readonly minMatchTokens: number
      /** Maximum committed suffix length searched for a match. */
      readonly maxMatchTokens: number
    }
    /** Ordered stage inputs keyed by executable binding slot. */
    readonly inputs: ReadonlyArray<{
      /** Zero-based executable binding slot. */
      readonly slot: number
      /** Runtime value routed into the slot. */
      readonly value: InferenceValueRoute
    }>
    /** Logical schemas of stage outputs in executable order. */
    readonly outputs: ReadonlyArray<InferenceValueSchema>
  }>
  /** Persistent proposer-state and commit policy. */
  readonly state: {
    /** Whether the schedule maintains paged KV state. */
    readonly kind: "None" | "Kv"
    /** Stable identifier of the state schema when state is present. */
    readonly schemaId?: string
    /** How accepted candidates update proposer state. */
    readonly commitKind: "None" | "AutoregressiveChain" | "Replay"
    /** Stage indexes whose state participates in commit. */
    readonly commitStages: ReadonlyArray<number>
  }
  /** Candidate topology and routes published by the proposer. */
  readonly output: {
    /** Whether candidates form independent chains or a proposal tree. */
    readonly topology: "Chains" | "Trees"
    /** Probability semantics available to the acceptance algorithm. */
    readonly probabilities: "CausalNormalized" | "Deterministic" | "Unavailable"
    /** Route containing candidate token ids. */
    readonly tokenIds: InferenceValueRoute
    /** Optional route containing normalized candidate distributions. */
    readonly probabilityRows?: InferenceValueRoute | undefined
    /** Optional route containing parent indexes for tree proposals. */
    readonly parents?: InferenceValueRoute
    /** Optional route containing proposer confidence values. */
    readonly confidence?: InferenceValueRoute
  }
  /** Proposer-to-target vocabulary mapping and its fingerprint. */
  readonly tokenMap:
    & {
      /** Whether proposer token ids are already target ids or require a table. */
      readonly kind: "Identity" | "Table"
      /** Stable fingerprint of this complete mapping. */
      readonly fingerprint: string
    }
    & {
      /** Source vocabulary size when a table mapping is used. */
      readonly proposerVocabulary?: number
      /** Target token id for each proposer-vocabulary offset. */
      readonly targetIds?: ReadonlyArray<number>
    }
  /** Maximum proposal-row count supported by the trained artifact. */
  readonly trainedMaxRows: number
}

/**
 * Prompt and generation policy transferred atomically when admitting a sequence.
 *
 * @since 0.1.0
 * @category models
 */
export interface InferenceAddEntry {
  /** Borrowed dense rank-one token tensor. */
  readonly prompt: ConcreteTensorHandle
  /** Per-sequence overrides of the artifact's default sampling policy. */
  readonly sampling?: InferenceSamplingOverrides | undefined
  /** Optional maximum number of generated tokens for this sequence. */
  readonly maxTokens?: number | undefined
  /** Token ids that terminate generation after publication. */
  readonly eosTokens: ReadonlyArray<number>
}

/**
 * Ordered batch of sequences admitted to an inference session atomically.
 *
 * @since 0.1.0
 * @category models
 */
export interface InferenceAddRequest {
  /** Admission entries in result-page order. */
  readonly entries: ReadonlyArray<InferenceAddEntry>
}

/**
 * A selected native sequence and optional controls for one round only.
 *
 * @since 0.1.0
 * @category models
 */
export interface InferenceRoundEntry {
  /** Live sequence selected for this round. */
  readonly sequence: InferenceSequenceHandle
  /** Sampling overrides that do not alter the sequence's stored policy. */
  readonly sampling?: InferenceSamplingOverrides | undefined
}

/**
 * Ordered active sequence set for one native inference round.
 *
 * @since 0.1.0
 * @category models
 */
export interface InferenceRoundRequest {
  /** Distinct live sequences in result-page order. */
  readonly entries: ReadonlyArray<InferenceRoundEntry>
}

/**
 * One request-ordered, nonempty token page published by a native round.
 *
 * @since 0.1.0
 * @category models
 */
export interface InferenceTokenPage {
  /** Sequence that owns this page. */
  readonly sequence: InferenceSequenceHandle
  /** Stable native sequence identity suitable for durable correlation. */
  readonly sequenceId: bigint
  /** Newly committed target-vocabulary token ids. */
  readonly tokens: ReadonlyArray<number>
  /** Terminal policy satisfied by the final published token, when any. */
  readonly stopReason?: "eos" | "maxTokens" | undefined
}

/**
 * Durable completion receipt. `recovered` is true when the backend recovered a
 * previously committed result after completion won a cancellation/error race.
 *
 * @since 0.1.0
 * @category models
 */
export interface InferenceRoundResult {
  /** Monotonic artifact-local identifier acknowledged after consumption. */
  readonly roundId: bigint
  /** Whether this receipt was recovered from a completion race. */
  readonly recovered: boolean
  /** Nonempty pages in request order, omitting sequences with no publication. */
  readonly pages: ReadonlyArray<InferenceTokenPage>
}

/**
 * Durable logical state of one native inference sequence.
 *
 * @since 0.1.0
 * @category models
 */
export interface InferenceSequenceInspection {
  /** Stable native sequence identity. */
  readonly sequenceId: bigint
  /** Number of prompt and generated tokens committed to the sequence. */
  readonly cursor: bigint
  /** Terminal generation policy already satisfied by this sequence. */
  readonly terminal?: "eos" | "maxTokens" | undefined
}

/**
 * Cumulative native inference counters and timing observations.
 *
 * @since 0.1.0
 * @category models
 */
export interface InferenceDiagnostics {
  /** Number of inference rounds that entered execution. */
  readonly roundsStarted: bigint
  /** Number of rounds durably completed. */
  readonly roundsCompleted: bigint
  /** Number of completions recovered after a cancellation/error race. */
  readonly roundsRecovered: bigint
  /** Number of completed non-speculative rounds. */
  readonly ordinaryRounds: bigint
  /** Number of completed speculative rounds. */
  readonly speculativeRounds: bigint
  /** Total candidates produced by proposers. */
  readonly proposedTokens: bigint
  /** Total proposer candidates accepted by target verification. */
  readonly acceptedTokens: bigint
  /** Total target tokens published to callers. */
  readonly emittedTokens: bigint
  /** Number of provisional state blocks created. */
  readonly provisionalBlocks: bigint
  /** Number of provisional state blocks rolled back. */
  readonly rolledBackBlocks: bigint
  /** Cumulative proposer execution time. */
  readonly draftNanos: bigint
  /** Cumulative target verification time. */
  readonly verificationNanos: bigint
  /** Index is accepted candidate count; values are completed-lane counts. */
  readonly acceptedLengthHistogram: ReadonlyArray<bigint>
  /** Highest simultaneous target-pool block usage. */
  readonly targetPoolHighWaterBlocks: bigint
  /** Highest simultaneous proposer-pool block usage, when applicable. */
  readonly proposerPoolHighWaterBlocks?: bigint | undefined
  /** Most recently allocated round identifier. */
  readonly lastRoundId?: bigint | undefined
  /** Phase of the most recent inference failure. */
  readonly lastFailurePhase?: InferenceFailurePhase | undefined
}

/**
 * Legacy low-level exact-chain request retained for direct decode consumers.
 * Prefer {@link InferenceRuntime.runRound} for sampled generation.
 *
 * @since 0.1.0
 * @category models
 */
export interface SpeculativeRoundRequest {
  /** Packed target verifier executable. */
  readonly targetVerify: ExecutableHandle
  /** Autoregressive proposer decode executable. */
  readonly proposerDecode: ExecutableHandle
  /** Target sequences in lane order. */
  readonly targetSequences: ReadonlyArray<KvSequenceHandle>
  /** Proposer sequences corresponding to `targetSequences`. */
  readonly proposerSequences: ReadonlyArray<KvSequenceHandle>
  /** Physical executable slot for each sequence pair. */
  readonly slots: ReadonlyArray<number>
  /** One pending target token per active sequence. */
  readonly pendingTokens: ReadonlyArray<number>
  /** Sampling policy for each active sequence. */
  readonly sampling: ReadonlyArray<SamplingOptions>
  /** Maximum candidates drafted for each sequence. */
  readonly maxDraftTokens: number
  /** Remaining publication limit for each sequence. */
  readonly pageLimits: ReadonlyArray<number>
  /** End-of-sequence token ids for each sequence. */
  readonly eosTokens: ReadonlyArray<ReadonlyArray<number>>
}

/**
 * Required native artifact and session interface for sampled generation.
 * Session operations are transactional. A round either publishes a durable
 * receipt or leaves no partially visible result.
 *
 * @since 0.1.0
 * @category models
 */
export interface InferenceRuntime {
  /** Validates and bundles programs, pools, and routing into an immutable artifact. */
  readonly compile: (request: InferenceCompileRequest) => Effect.Effect<InferenceArtifactHandle, BackendError>
  /** Opens an independent mutable generation session over an artifact. */
  readonly open: (artifact: InferenceArtifactHandle) => Effect.Effect<InferenceSessionHandle, BackendError>
  /** Atomically admits and prefills prompts, returning their initial publication. */
  readonly add: (
    session: InferenceSessionHandle,
    request: InferenceAddRequest
  ) => Effect.Effect<InferenceRoundResult, BackendError>
  /** Runs one transactional sampled generation round for selected sequences. */
  readonly runRound: (
    session: InferenceSessionHandle,
    request: InferenceRoundRequest
  ) => Effect.Effect<InferenceRoundResult, BackendError>
  /** Releases a validated durable receipt after the caller accepted it. */
  readonly acknowledge: (
    session: InferenceSessionHandle,
    roundId: bigint
  ) => Effect.Effect<void, BackendError>
  /** Removes completed sequences and releases their mutable native state. */
  readonly finish: (
    session: InferenceSessionHandle,
    sequences: ReadonlyArray<InferenceSequenceHandle>
  ) => Effect.Effect<void, BackendError>
  /** Inspects durable state without mutating the sequence. */
  readonly inspect: (
    session: InferenceSessionHandle,
    sequence: InferenceSequenceHandle
  ) => Effect.Effect<InferenceSequenceInspection, BackendError>
  /** Closes a session and releases every sequence still owned by it. */
  readonly close: (session: InferenceSessionHandle) => Effect.Effect<void, BackendError>
  /** Returns cumulative diagnostics attached to an immutable artifact. */
  readonly diagnostics: (artifact: InferenceArtifactHandle) => Effect.Effect<InferenceDiagnostics, BackendError>
}

/**
 * Native next-token sampling extension. Direct sampling borrows one live,
 * dense, rank-one floating-point tensor. Fused decode execution samples one
 * rank-one output per active state sequence without publishing output tensors.
 * Both paths return only selected u32 offsets, so no tensor ownership transfers.
 *
 * @since 0.1.0
 * @category models
 */
export interface SamplingRuntime {
  /** Samples one already-materialized logits row without consuming it. */
  readonly sample: (
    logits: ConcreteTensorHandle,
    options: SamplingOptions
  ) => Effect.Effect<number, BackendError>
  /**
   * Executes one stateful decode invocation and samples its active outputs in
   * order. The invocation follows `RuntimeService.execute`'s input, state,
   * cancellation, and atomic-commit rules. `options` contains one normalized
   * entry per active output.
   */
  readonly executeDecode: (
    executable: ExecutableHandle,
    invocation: ExecutionInvocation,
    options: ReadonlyArray<SamplingOptions>
  ) => Effect.Effect<ReadonlyArray<number>, BackendError>
  /** @deprecated Generation uses {@link InferenceRuntime.runRound}. */
  readonly executeSpeculative: (
    request: SpeculativeRoundRequest
  ) => Effect.Effect<ReadonlyArray<ReadonlyArray<number>>, BackendError>
}

/**
 * Required runtime extension for compiled paged-KV and recurrent inference.
 * Pool geometry must exactly match the executable schema. Attention geometry
 * and each recurrent family are independently either all zero or all positive;
 * capacities and paging units are positive with exact divisibility. Every
 * opaque argument is runtime-owned and must be validated before native access.
 * Sequence operations mutate state and therefore must not overlap for the same
 * sequence, including with `RuntimeService.execute`.
 *
 * @since 0.1.0
 * @category models
 */
export interface DecodeRuntime {
  /**
   * Allocates a fixed-capacity decode-state pool. KV arenas and prefix-cache
   * content are shared by child sequences, while each sequence owns independent
   * mutable recurrent state. The returned pool belongs to this runtime and is
   * retained by its child sequences.
   */
  readonly makePool: (options: {
    /** Number of attention layers stored in the pool. */
    readonly layers: number
    /** Number of key/value heads per layer. */
    readonly kvHeads: number
    /** Width of each key/value head. */
    readonly headDim: number
    /** Total token-row capacity across live and cached sequences. */
    readonly maxTokens: number
    /** Number of token rows allocated and cached as one unit. */
    readonly blockSize: number
    /** Element data type used for KV storage. */
    readonly dtype: DType
    /** Number of KDA recurrent layers prepared for each sequence. */
    readonly kdaLayers: number
    /** Number of heads in each KDA recurrent layer. */
    readonly kdaHeads: number
    /** Key width of each KDA recurrent head. */
    readonly kdaHeadDim: number
    /** Value width of each KDA recurrent head. */
    readonly kdaValueDim: number
    /** Number of short-conv recurrent layers prepared for each sequence. */
    readonly convLayers: number
    /** Channel count of each short-conv recurrent layer. */
    readonly convChannels: number
    /** Kernel size of each short-conv recurrent layer. */
    readonly convKernel: number
  }) => Effect.Effect<KvPoolHandle, BackendError>
  /**
   * Creates an empty caller-owned sequence with independent cursor, block
   * table, and recurrent state, retaining the supplied pool.
   */
  readonly makeSequence: (pool: KvPoolHandle) => Effect.Effect<KvSequenceHandle, BackendError>
  /**
   * On an empty sequence, attaches the longest resident whole-block proper KV
   * prefix, leaving one token when input is nonempty. Hybrid pools with
   * recurrent geometry resume from snapshots published at completed block
   * boundaries; purely recurrent pools without KV blocks return zero. This
   * mutably borrows the sequence and must not overlap another operation on it.
   */
  readonly prefillMatch: (
    sequence: KvSequenceHandle,
    tokens: ReadonlyArray<number>
  ) => Effect.Effect<number, BackendError>
  /** Returns the sequence's absolute token cursor; the sequence must be live. */
  readonly sequenceCursor: (sequence: KvSequenceHandle) => Effect.Effect<number, BackendError>
  /** Releases a sequence and its block references; call exactly once. */
  readonly releaseSequence: (sequence: KvSequenceHandle) => Effect.Effect<void, BackendError>
}

/**
 * Runtime diagnostics that do not affect execution.
 *
 * @since 0.1.0
 * @category models
 */
export interface RuntimeDiagnostics {
  /**
   * Current bytes of native memory attributed to JavaScript-reachable tensors.
   * This is observational and may change concurrently; it is not total process
   * memory or a deterministic leak detector.
   */
  readonly externalMemoryBytes: Effect.Effect<number>
}

/**
 * One named exposure discovered in a lazy graph. It contains the name given to
 * `Tensor.expose` and the wrapped tensor.
 *
 * @since 0.1.0
 * @category models
 */
export interface NamedExposure {
  /** The exposure name. */
  readonly name: string
  /** The wrapped tensor (the exposure node's identity input). */
  readonly tensor: LazyTensorHandle
}

/**
 * A live tensor runtime bound to one default placement. Implementations are
 * responsible for native capability validation, immutable metadata, handle
 * ownership/liveness registries, interruption cleanup, and safe concurrent
 * use of immutable graphs and executables. Closing a backend, if supported by
 * its layer, invalidates all capabilities in this identity domain.
 *
 * @since 0.1.0
 * @category models
 */
export interface RuntimeService {
  /**
   * Stable object identity for the complete native ownership and cache domain.
   * Service wrappers may share it only when each can accept the other's live
   * handles. Core compilation caches use object identity, not serialization.
   */
  readonly identity: object
  /** Backend implementation metadata. */
  readonly backend: BackendInfo
  /** Default device and memory placement for newly created tensors. */
  readonly placement: Placement
  /** Data types and optional features supported by this runtime. */
  readonly capabilities: Capabilities
  /**
   * Constructs one lazy semantic graph node without executing tensor kernels.
   * The runtime validates all input handles and snapshots caller-owned mutable
   * attributes such as byte arrays before successful completion.
   */
  readonly node: (request: NodeRequest) => Effect.Effect<LazyTensorHandle, BackendError>
  /**
   * Walks the lazy graph reachable from `root` and returns every `expose`
   * exposure in deterministic first-visit order. Fails on duplicate names.
   * Discovery walks the traced graph only; nothing executes.
   */
  readonly exposures: (
    root: LazyTensorHandle
  ) => Effect.Effect<ReadonlyArray<NamedExposure>, BackendError>
  /**
   * Builds lazy reverse-mode gradient graphs without materializing the loss.
   * Inputs are borrowed, must be live and runtime-owned, and output order must
   * match `wrt`, including duplicates.
   */
  readonly grad: (
    loss: TensorHandle,
    wrt: ReadonlyArray<TensorHandle>
  ) => Effect.Effect<ReadonlyArray<LazyTensorHandle>, BackendError>
  /**
   * Compiles nonempty, live, runtime-owned, single-placement roots into one
   * immutable executable. Tensor and scalar input declarations share one
   * zero-based, gap-free slot namespace; repeated declarations must agree on
   * kind and complete logical metadata. Root order and duplicates define
   * output order. Implementations may reuse a structurally equivalent native
   * artifact when value capture and state options permit it.
   */
  readonly compile: (request: CompileRequest) => Effect.Effect<ExecutableHandle, BackendError>
  /**
   * Executes a live runtime-owned immutable program with one complete
   * invocation. Inputs and state sequences are borrowed until completion.
   * Returned handles are distinct caller-owned capabilities, survive later
   * invocations, and should be passed to idempotent `release` for deterministic
   * cleanup. Concurrent calls are supported for stateless invocations and for
   * stateful invocations using disjoint sequences.
   *
   * On failure or interruption, no output ownership transfers. The runtime
   * must retire submitted work safely, release partial or late output handles,
   * and roll back state not committed by a successful stateful invocation.
   */
  readonly execute: (
    executable: ExecutableHandle,
    invocation: ExecutionInvocation
  ) => Effect.Effect<ReadonlyArray<ConcreteTensorHandle>, BackendError>
  /**
   * Exposes the tensor's host-transfer representation through an `ArrayBuffer`.
   * The concrete handle is borrowed for the operation. Dense `f16` and `bf16`
   * values are widened to `f32`; other dense dtypes retain their logical dtype.
   * For encoded handles this is the packed `u8` representation, not logical
   * `f32` values. A backend may copy the data or directly export retained
   * runtime storage; callers must not rely on either mode. The returned buffer
   * remains readable after the handle is released; a direct export may defer
   * physical cleanup until the buffer becomes unreachable. Interruption
   * transfers no buffer.
   */
  readonly readback: (tensor: ConcreteTensorHandle) => Effect.Effect<ArrayBuffer, BackendError>
  /**
   * Releases this concrete handle's ownership and, on first success, invalidates
   * it and lazy graphs that directly captured it. Release is idempotent: repeated
   * calls for a handle successfully released by this runtime must also succeed,
   * although every other operation must continue to reject that cleared handle.
   * Forged, foreign, or wrong-kind handles still fail. Other owned handles and
   * executable-retained constants may keep storage live. A failure for a handle
   * not already released leaves backend-defined liveness and must not be treated
   * as a successful release.
   */
  readonly release: (tensor: ConcreteTensorHandle) => Effect.Effect<void, BackendError>
  /** Required facilities in this same identity and placement domain. */
  readonly extensions: {
    /** Direct path-based safetensors I/O. */
    readonly pathSafetensors: PathSafetensors
    /** Native GGUF inspection and loading. */
    readonly gguf: GgufRuntime
    /** Native next-token sampling and fused stateful decode execution. */
    readonly sampling: SamplingRuntime
    /** Compiled paged-KV inference. */
    readonly decode: DecodeRuntime
    /** Native sampled inference artifacts and sessions. */
    readonly inference: InferenceRuntime
    /** Runtime memory and execution diagnostics. */
    readonly diagnostics: RuntimeDiagnostics
  }
}

/**
 * The tensor runtime for the current Effect program.
 *
 * @since 0.1.0
 * @category services
 */
export class Runtime extends Context.Service<Runtime, RuntimeService>()(
  "@effect-torch/core/Runtime"
) {}

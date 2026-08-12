import { Context, Data, type Effect } from "effect"
import type { Pipeable } from "effect/Pipeable"

/**
 * Element data types supported by tensor runtimes.
 *
 * @since 0.1.0
 * @category models
 */
export type DType = "f32" | "f64" | "f16" | "bf16" | "i64" | "u8" | "u32"

/** Packed GGML K-quant encodings understood by native runtimes. */
export type TensorStorageEncoding = "Q2_K" | "Q3_K" | "Q4_K" | "Q5_K" | "Q6_K"

/** Physical storage metadata for a logically dense encoded tensor. */
export interface EncodedTensorStorage {
  readonly encoding: TensorStorageEncoding
  readonly physicalShape: ReadonlyArray<number>
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
 * A runtime-owned device and memory placement.
 *
 * @since 0.1.0
 * @category models
 */
export interface Placement {
  /** Stable identity for this placement within its runtime. */
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
 * Capabilities advertised by a runtime.
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
 * Structured failures reported by backend runtimes.
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

/**
 * A backend-owned tensor value with only backend-neutral static metadata.
 *
 * @since 0.1.0
 * @category models
 */
export interface TensorHandle extends Pipeable {
  /** Nominal tensor-handle brand. */
  readonly [TensorHandleTypeId]: typeof TensorHandleTypeId
  /** Whether this handle is lazy or materialized. */
  readonly _tag: "LazyTensor" | "Tensor"
  /** Logical tensor dimensions. */
  readonly shape: ReadonlyArray<number>
  /** Tensor element data type. */
  readonly dtype: DType
  /** Omitted for dense storage; present when native storage is encoded. */
  readonly storage?: EncodedTensorStorage
  /** Device family that owns the tensor. */
  readonly device: string
  /** Exact runtime placement that owns the tensor. */
  readonly placement: Placement
}

/**
 * A backend-owned lazy tensor value.
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
 * A backend-owned materialized tensor value.
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
  readonly optimize?: boolean
  /**
   * Authorizes inference-only retention of eligible materialized graph leaves
   * as executable constants. Their storage remains live with the executable,
   * and bundled runtimes bypass structural executable-cache reuse in this mode.
   * Defaults to `false`; do not enable for values expected to vary.
   */
  readonly constantWeights?: boolean
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
  readonly window?: number
  /** Positive unsigned 32-bit fixed compiled batch width. */
  readonly batch: number
  /**
   * When true, every root must be `[batch, T, V]` and the decode rewrite
   * returns native state-driven last-token selectors: one `[V]` root for
   * batch 1, otherwise `batch` `[V]` roots in row order. Defaults to false.
   */
  readonly lastTokenRow?: boolean
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
 *
 * @since 0.1.0
 * @category models
 */
export interface ExecutableHandle {
  /** Nominal executable-handle brand. */
  readonly [ExecutableHandleTypeId]: typeof ExecutableHandleTypeId
  /** Complete state schema for generation executables. */
  readonly state?: DecodeStateSchema
  /** Immutable lowering and static-memory summary. */
  readonly diagnostics: ExecutableDiagnostics
}

/**
 * One semantic graph compilation request.
 *
 * @since 0.1.0
 * @category models
 */
export interface CompileRequest {
  /**
   * Nonempty semantic roots owned by this runtime and belonging to one device.
   * Root order, including duplicates, defines executable output order.
   */
  readonly roots: ReadonlyArray<TensorHandle>
  /** Explicit controls that join the executable cache key. */
  readonly options?: ExecutableCompileOptions
  /** Optional bounded persistent-state contract. */
  readonly state?: DecodeStateRequest
}

/**
 * Per-invocation state supplied to a stateful executable.
 *
 * @since 0.1.0
 * @category models
 */
export interface ExecutionStateInvocation {
  /**
   * From `1` through the compiled batch width, distinct live sequences from
   * one schema-compatible pool.
   */
  readonly sequences: ReadonlyArray<KvSequenceHandle>
  /**
   * One equally sized, nonempty row of unsigned 32-bit token ids per sequence.
   * Success commits state and advances cursors; pre-commit failure rolls back.
   */
  readonly tokens: ReadonlyArray<ReadonlyArray<number>>
}

/**
 * Complete dynamic input to one immutable executable invocation.
 *
 * @since 0.1.0
 * @category models
 */
export interface ExecutionInvocation {
  /**
   * Materialized tensor bindings in ascending shared-slot order with scalar
   * slots omitted. Counts, metadata, layout, ownership, and placement must
   * match the compiled declarations.
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
  readonly state?: ExecutionStateInvocation
}

/**
 * Opaque backend-owned paged KV cache pool.
 *
 * @since 0.1.0
 * @category models
 */
export interface KvPoolHandle {
  /** Nominal KV-pool-handle brand. */
  readonly [KvPoolHandleTypeId]: typeof KvPoolHandleTypeId
}

/**
 * Opaque backend-owned sequence allocated from a paged KV pool.
 *
 * @since 0.1.0
 * @category models
 */
export interface KvSequenceHandle {
  /** Nominal KV-sequence-handle brand. */
  readonly [KvSequenceHandleTypeId]: typeof KvSequenceHandleTypeId
}

/**
 * Inputs and attributes for every semantic graph operation.
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
      readonly storage?: EncodedTensorStorage
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
      readonly window?: number | null
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
      readonly paddingIndex?: number
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
 * A type-checked semantic graph construction request.
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
 * Optional runtime extension for direct path-based safetensors I/O.
 *
 * @since 0.1.0
 * @category models
 */
export interface PathSafetensors {
  /** Writes a borrowed dense archive without transferring tensor data through JavaScript. */
  readonly save: (path: string, archive: PathSafetensorsSaveArchive) => Effect.Effect<void, BackendError>
  /** Reads an archive directly into materialized runtime tensors. */
  readonly load: (path: string) => Effect.Effect<PathSafetensorsLoadArchive, BackendError>
}

/** A scalar GGUF metadata value after native validation. */
export type GgufMetadataScalar = number | string | boolean

/** One ordered GGUF metadata entry. */
export interface GgufMetadataEntry {
  readonly key: string
  readonly value: GgufMetadataScalar | ReadonlyArray<GgufMetadataScalar>
}

/** Backend-neutral tensor catalog entry returned by native GGUF inspection. */
export interface GgufTensorDescriptor {
  readonly name: string
  readonly format: "F32" | TensorStorageEncoding
  readonly logicalShape: ReadonlyArray<number>
  readonly logicalDtype: "f32"
  readonly physicalShape: ReadonlyArray<number>
  readonly physicalDtype: "f32" | "u8"
}

/** GGUF metadata and tensor descriptors without tensor payloads. */
export interface GgufInspection {
  readonly metadata: ReadonlyArray<GgufMetadataEntry>
  readonly tensors: ReadonlyArray<GgufTensorDescriptor>
}

/** One runtime-owned tensor returned by native GGUF loading. */
export interface GgufLoadEntry {
  readonly descriptor: GgufTensorDescriptor
  readonly tensor: ConcreteTensorHandle
}

/** Runtime-owned tensors returned by native GGUF loading. */
export interface GgufLoadArchive {
  readonly entries: ReadonlyArray<GgufLoadEntry>
}

/** Optional native GGUF parser and loader extension. */
export interface GgufRuntime {
  readonly inspect: (path: string) => Effect.Effect<GgufInspection, BackendError>
  /**
   * Loads runtime-owned tensors. The implementation owns every handle until
   * successful Effect completion and must release partial or late results when
   * interrupted; ownership transfers to the caller with the returned archive.
   */
  readonly load: (path: string) => Effect.Effect<GgufLoadArchive, BackendError>
}

/**
 * Optional runtime extension for compiled paged-KV and recurrent inference.
 * Pool geometry must exactly match the executable schema. Attention geometry
 * and each recurrent family are independently either all zero or all positive;
 * capacities and paging units are positive with exact divisibility.
 *
 * @since 0.1.0
 * @category models
 */
export interface DecodeRuntime {
  /** Allocates the fixed-capacity paged KV storage shared by sequences. */
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
  /** Creates an empty sequence with independent cursor, block table, and recurrent state. */
  readonly makeSequence: (pool: KvPoolHandle) => Effect.Effect<KvSequenceHandle, BackendError>
  /**
   * On an empty sequence, attaches the longest resident whole-block proper KV
   * prefix, leaving one token when input is nonempty. Hybrid pools with
   * recurrent geometry resume from snapshots published at completed block
   * boundaries; purely recurrent pools without KV blocks return zero.
   */
  readonly prefillMatch: (
    sequence: KvSequenceHandle,
    tokens: ReadonlyArray<number>
  ) => Effect.Effect<number, BackendError>
  /** Returns the sequence's absolute token cursor. */
  readonly sequenceCursor: (sequence: KvSequenceHandle) => Effect.Effect<number, BackendError>
  /** Releases a sequence and its block references; call exactly once. */
  readonly releaseSequence: (sequence: KvSequenceHandle) => Effect.Effect<void, BackendError>
}

/**
 * Optional runtime diagnostics that do not affect execution.
 *
 * @since 0.1.0
 * @category models
 */
export interface RuntimeDiagnostics {
  /** Current bytes of native memory attributed to JavaScript-reachable tensors. */
  readonly externalMemoryBytes: Effect.Effect<number>
}

/**
 * A live tensor runtime bound to one default placement.
 *
 * @since 0.1.0
 * @category models
 */
export interface RuntimeService {
  /** Stable identity shared by equivalent service instances and used to isolate backend-owned caches. */
  readonly identity: object
  /** Backend implementation metadata. */
  readonly backend: BackendInfo
  /** Default device and memory placement for newly created tensors. */
  readonly placement: Placement
  /** Data types and optional features supported by this runtime. */
  readonly capabilities: Capabilities
  /** Constructs one lazy semantic graph node. */
  readonly node: (request: NodeRequest) => Effect.Effect<LazyTensorHandle, BackendError>
  /** Builds reverse-mode gradients of `loss` with respect to selected tensors. */
  readonly grad: (
    loss: TensorHandle,
    wrt: ReadonlyArray<TensorHandle>
  ) => Effect.Effect<ReadonlyArray<LazyTensorHandle>, BackendError>
  /**
   * Compiles nonempty, runtime-owned, single-device roots into one immutable
   * executable. Tensor and scalar input declarations share one zero-based,
   * gap-free slot namespace; repeated declarations must agree exactly.
   */
  readonly compile: (request: CompileRequest) => Effect.Effect<ExecutableHandle, BackendError>
  /**
   * Executes a runtime-owned immutable program with one complete invocation.
   * Inputs are borrowed. Returned handles are caller-owned, survive later
   * invocations, and require exactly one successful `release` for deterministic cleanup.
   */
  readonly execute: (
    executable: ExecutableHandle,
    invocation: ExecutionInvocation
  ) => Effect.Effect<ReadonlyArray<ConcreteTensorHandle>, BackendError>
  /**
   * Exposes physical tensor storage through an `ArrayBuffer`. For encoded
   * handles this is the packed byte representation, not logical f32 values. A
   * backend may copy the data or directly export retained runtime storage;
   * callers must not rely on either mode.
   */
  readonly readback: (tensor: ConcreteTensorHandle) => Effect.Effect<ArrayBuffer, BackendError>
  /**
   * Releases this concrete handle's ownership and invalidates it and lazy graphs
   * that captured it. Call exactly once; other aliases may still retain storage.
   */
  readonly release: (tensor: ConcreteTensorHandle) => Effect.Effect<void, BackendError>
  /** Optional backend facilities outside the common tensor runtime contract. */
  readonly extensions: {
    /** Direct path-based safetensors I/O, when supported. */
    readonly pathSafetensors?: PathSafetensors
    /** Native GGUF inspection and loading, when supported. */
    readonly gguf?: GgufRuntime
    /** Compiled paged-KV inference, when supported. */
    readonly decode?: DecodeRuntime
    /** Runtime memory and execution diagnostics, when supported. */
    readonly diagnostics?: RuntimeDiagnostics
  }
}

/**
 * The authoritative tensor runtime for the current Effect program.
 *
 * @since 0.1.0
 * @category services
 */
export class Runtime extends Context.Service<Runtime, RuntimeService>()(
  "@effect-torch/core/Runtime"
) {}

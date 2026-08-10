import { Context, Data, type Effect } from "effect"
import type { Pipeable } from "effect/Pipeable"

/**
 * Element data types supported by tensor runtimes.
 *
 * @since 0.1.0
 * @category models
 */
export type DType = "f32" | "f64" | "f16" | "bf16" | "i64" | "u8" | "u32"

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
  /** Enables optional rewrites and fusion. Defaults to `true`. */
  readonly optimize?: boolean
  /** Precision policy used by backend lowering. Defaults to `strict`. */
  readonly precision?: "strict" | "allow-reduced-precision"
  /** Authorizes inference-only constant-weight preparation. */
  readonly constantWeights?: boolean
  /** Number of live output generations preallocated during compilation. */
  readonly outputCapacity?: number
}

/**
 * Bounded persistent state requested for a generation executable.
 *
 * @since 0.1.0
 * @category models
 */
export interface DecodeStateRequest {
  /** Total token-row capacity of the compatible KV pool. */
  readonly maxTokens: number
  /** Number of token rows allocated and cached as one unit. */
  readonly blockSize: number
  /** Element data type used for KV storage. */
  readonly kvDtype: DType
  /** Optional bounded sliding-attention window. */
  readonly window?: number
  /** Fixed compiled batch width. */
  readonly batch: number
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

/** One lowered instruction category in executable diagnostics. */
export interface ExecutableInstructionDiagnostics {
  readonly kind: string
  readonly count: number
}

/** Static memory totals derived from an executable's immutable plan. */
export interface ExecutableMemoryDiagnostics {
  readonly externalBytes: number
  readonly persistentBytes: number
  readonly stateBytes: number
  readonly outputBytes: number
  readonly workspaceBytes: number
  readonly transactionBytes: number
  readonly peakLiveBytes: number
  readonly packingOverheadBytes: number
}

/** Deterministic compile and planning summary attached to an executable. */
export interface ExecutableDiagnostics {
  readonly semanticNodesBeforeOptimization: number
  readonly semanticNodesAfterOptimization: number
  readonly instructions: ReadonlyArray<ExecutableInstructionDiagnostics>
  readonly pipelineCount: number
  readonly commandCount: number
  readonly synchronizationCount: number
  /** Maximum live output generations admitted without allocating during execution. */
  readonly outputCapacity: number
  readonly memory: ExecutableMemoryDiagnostics
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
  /** Semantic graph roots to lower into one executable. */
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
  /** Active sequences, from one compatible pool. */
  readonly sequences: ReadonlyArray<KvSequenceHandle>
  /** Real unpadded token ids represented by each active sequence row. */
  readonly tokens: ReadonlyArray<ReadonlyArray<number>>
}

/**
 * Complete dynamic input to one immutable executable invocation.
 *
 * @since 0.1.0
 * @category models
 */
export interface ExecutionInvocation {
  /** Materialized tensor bindings in declaration order. */
  readonly bindings: ReadonlyArray<ConcreteTensorHandle>
  /** Scalar bindings in declaration order. */
  readonly scalars: ReadonlyArray<number>
  /** Named bounded runtime values used by the fixed command schedule. */
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
    readonly attributes: { readonly slot: number; readonly shape: ReadonlyArray<number>; readonly dtype: DType }
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
    readonly attributes: { readonly scale: number; readonly causal: boolean }
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
    readonly attributes: { readonly seqLen: number; readonly theta: number }
  }
  /** Normalizes the trailing dimensions and applies affine parameters. */
  readonly layerNorm: {
    readonly inputs: readonly [self: TensorHandle, weight: TensorHandle, bias: TensorHandle]
    readonly attributes: { readonly eps: number }
  }
  /** Applies a linear projection with a bias. */
  readonly linear: {
    readonly inputs: readonly [self: TensorHandle, weight: TensorHandle, bias: TensorHandle]
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
  /** Materialized tensor to serialize. */
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
  /** Materialized tensor loaded into the runtime placement. */
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
  /** Writes a materialized archive without transferring tensor data through JavaScript. */
  readonly save: (path: string, archive: PathSafetensorsSaveArchive) => Effect.Effect<void, BackendError>
  /** Reads an archive directly into materialized runtime tensors. */
  readonly load: (path: string) => Effect.Effect<PathSafetensorsLoadArchive, BackendError>
}

/**
 * Optional runtime extension for compiled paged-KV inference.
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
  /** Creates an empty sequence backed by a KV pool. */
  readonly makeSequence: (pool: KvPoolHandle) => Effect.Effect<KvSequenceHandle, BackendError>
  /** Attaches the longest resident whole-block proper prefix, leaving one token when input is non-empty. */
  readonly prefillMatch: (
    sequence: KvSequenceHandle,
    tokens: ReadonlyArray<number>
  ) => Effect.Effect<number, BackendError>
  /** Returns the sequence's absolute token cursor. */
  readonly sequenceCursor: (sequence: KvSequenceHandle) => Effect.Effect<number, BackendError>
  /** Releases a sequence and all block references it owns. */
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
  /** Compiles semantic graph roots into one immutable backend executable. */
  readonly compile: (request: CompileRequest) => Effect.Effect<ExecutableHandle, BackendError>
  /** Executes an immutable program with one complete invocation. */
  readonly execute: (
    executable: ExecutableHandle,
    invocation: ExecutionInvocation
  ) => Effect.Effect<ReadonlyArray<ConcreteTensorHandle>, BackendError>
  /** Copies a materialized tensor into a host-owned array buffer. */
  readonly readback: (tensor: ConcreteTensorHandle) => Effect.Effect<ArrayBuffer, BackendError>
  /** Deterministically releases the storage owned by a materialized tensor handle. */
  readonly release: (tensor: ConcreteTensorHandle) => Effect.Effect<void, BackendError>
  /** Optional backend facilities outside the common tensor runtime contract. */
  readonly extensions: {
    /** Direct path-based safetensors I/O, when supported. */
    readonly pathSafetensors?: PathSafetensors
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

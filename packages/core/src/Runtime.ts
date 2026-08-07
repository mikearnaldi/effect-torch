import { Context, Data, type Effect } from "effect"
import type { Pipeable } from "effect/Pipeable"

/** Element data types supported by tensor runtimes. */
export type DType = "f32" | "f64" | "f16" | "bf16" | "i64" | "u8" | "u32"

/**
 * Backend implementation metadata.
 *
 * @since 0.1.0
 * @category models
 */
export interface BackendInfo {
  readonly name: string
}

/**
 * A runtime-owned device and memory placement.
 *
 * @since 0.1.0
 * @category models
 */
export interface Placement {
  readonly id: string
  readonly deviceType: string
  readonly description: string
  readonly ordinal?: number
  readonly memorySpace?: string
}

/**
 * Capabilities advertised by a runtime.
 *
 * @since 0.1.0
 * @category models
 */
export interface Capabilities {
  readonly dtypes: ReadonlyArray<DType>
  readonly features: ReadonlyArray<string>
}

/**
 * Structured failures reported by backend runtimes.
 *
 * @since 0.1.0
 * @category errors
 */
export class BackendError extends Data.TaggedError("BackendError")<{
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
  readonly backend: string
  readonly operation: string
  readonly phase: "graph" | "autodiff" | "compile" | "execute" | "readback" | "io" | "shutdown"
  readonly message: string
  readonly details?: unknown
}> {}

declare const TensorHandleTypeId: unique symbol
declare const LazyTensorHandleTypeId: unique symbol
declare const ConcreteTensorHandleTypeId: unique symbol
declare const ProgramHandleTypeId: unique symbol
declare const DecodeProgramHandleTypeId: unique symbol
declare const KvPoolHandleTypeId: unique symbol
declare const KvSequenceHandleTypeId: unique symbol

/** A backend-owned tensor value with only backend-neutral static metadata. */
export interface TensorHandle extends Pipeable {
  readonly [TensorHandleTypeId]: typeof TensorHandleTypeId
  readonly _tag: "LazyTensor" | "Tensor"
  readonly shape: ReadonlyArray<number>
  readonly dtype: DType
  readonly device: string
  readonly placement: Placement
}

/** A backend-owned lazy tensor value. */
export interface LazyTensorHandle extends TensorHandle {
  readonly [LazyTensorHandleTypeId]: typeof LazyTensorHandleTypeId
  readonly _tag: "LazyTensor"
}

/** A backend-owned materialized tensor value. */
export interface ConcreteTensorHandle extends TensorHandle {
  readonly [ConcreteTensorHandleTypeId]: typeof ConcreteTensorHandleTypeId
  readonly _tag: "Tensor"
}

export interface ProgramHandle {
  readonly [ProgramHandleTypeId]: typeof ProgramHandleTypeId
}

export interface DecodeProgramHandle {
  readonly [DecodeProgramHandleTypeId]: typeof DecodeProgramHandleTypeId
}

export interface KvPoolHandle {
  readonly [KvPoolHandleTypeId]: typeof KvPoolHandleTypeId
}

export interface KvSequenceHandle {
  readonly [KvSequenceHandleTypeId]: typeof KvSequenceHandleTypeId
}

/**
 * Inputs and attributes for every semantic graph operation.
 *
 * @since 0.1.0
 * @category models
 */
export interface NodeOperationMap {
  readonly constant: {
    readonly inputs: readonly [] | readonly [exemplar: TensorHandle]
    readonly attributes: { readonly value: number; readonly dtype: DType }
  }
  readonly zeros: {
    readonly inputs: readonly [] | readonly [exemplar: TensorHandle]
    readonly attributes: { readonly shape: ReadonlyArray<number>; readonly dtype: DType }
  }
  readonly ones: {
    readonly inputs: readonly [] | readonly [exemplar: TensorHandle]
    readonly attributes: { readonly shape: ReadonlyArray<number>; readonly dtype: DType }
  }
  readonly full: {
    readonly inputs: readonly [] | readonly [exemplar: TensorHandle]
    readonly attributes: { readonly shape: ReadonlyArray<number>; readonly value: number; readonly dtype: DType }
  }
  readonly randn: {
    readonly inputs: readonly []
    readonly attributes: { readonly shape: ReadonlyArray<number>; readonly dtype: DType }
  }
  readonly uniform: {
    readonly inputs: readonly []
    readonly attributes: {
      readonly shape: ReadonlyArray<number>
      readonly lo: number
      readonly hi: number
      readonly dtype: DType
    }
  }
  readonly arange: {
    readonly inputs: readonly []
    readonly attributes: { readonly start: number; readonly end: number; readonly step: number; readonly dtype: DType }
  }
  readonly eye: {
    readonly inputs: readonly []
    readonly attributes: { readonly n: number; readonly dtype: DType }
  }
  readonly fromBytes: {
    readonly inputs: readonly []
    /** The backend snapshots `data`; the caller retains ownership. */
    readonly attributes: { readonly data: Uint8Array; readonly shape: ReadonlyArray<number>; readonly dtype: DType }
  }
  readonly input: {
    readonly inputs: readonly [] | readonly [exemplar: TensorHandle]
    readonly attributes: { readonly slot: number; readonly shape: ReadonlyArray<number>; readonly dtype: DType }
  }
  readonly scalarInput: {
    readonly inputs: readonly []
    readonly attributes: { readonly slot: number; readonly dtype: DType }
  }
  readonly add: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  readonly sub: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  readonly mul: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  readonly div: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  readonly maximum: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  readonly minimum: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  readonly eq: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  readonly gt: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  readonly lt: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  readonly ge: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  readonly le: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  readonly matmul: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  readonly solve: { readonly inputs: readonly [self: TensorHandle, other: TensorHandle] }
  readonly concat: {
    readonly inputs: readonly [self: TensorHandle, other: TensorHandle]
    readonly attributes: { readonly dim: number }
  }
  readonly neg: { readonly inputs: readonly [self: TensorHandle] }
  readonly abs: { readonly inputs: readonly [self: TensorHandle] }
  readonly sqrt: { readonly inputs: readonly [self: TensorHandle] }
  readonly exp: { readonly inputs: readonly [self: TensorHandle] }
  readonly log: { readonly inputs: readonly [self: TensorHandle] }
  readonly sin: { readonly inputs: readonly [self: TensorHandle] }
  readonly cos: { readonly inputs: readonly [self: TensorHandle] }
  readonly tanh: { readonly inputs: readonly [self: TensorHandle] }
  readonly relu: { readonly inputs: readonly [self: TensorHandle] }
  readonly erf: { readonly inputs: readonly [self: TensorHandle] }
  readonly floor: { readonly inputs: readonly [self: TensorHandle] }
  readonly ceil: { readonly inputs: readonly [self: TensorHandle] }
  readonly round: { readonly inputs: readonly [self: TensorHandle] }
  readonly sign: { readonly inputs: readonly [self: TensorHandle] }
  readonly inverse: { readonly inputs: readonly [self: TensorHandle] }
  readonly det: { readonly inputs: readonly [self: TensorHandle] }
  readonly stopGradient: { readonly inputs: readonly [self: TensorHandle] }
  readonly checkpoint: { readonly inputs: readonly [self: TensorHandle] }
  readonly gelu: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly approximate?: boolean | null }
  }
  readonly pow: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly exponent: number }
  }
  readonly cast: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly dtype: DType }
  }
  readonly whereCond: {
    readonly inputs: readonly [condition: TensorHandle, a: TensorHandle, b: TensorHandle]
  }
  readonly argmax: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly dim: number }
  }
  readonly argmin: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly dim: number }
  }
  readonly cumsum: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly dim: number }
  }
  readonly indexSelect: {
    readonly inputs: readonly [self: TensorHandle, indexes: TensorHandle]
    readonly attributes: { readonly dim: number }
  }
  readonly scatterAdd: {
    readonly inputs: readonly [self: TensorHandle, indexes: TensorHandle, src: TensorHandle]
    readonly attributes: { readonly dim: number }
  }
  readonly gather: {
    readonly inputs: readonly [self: TensorHandle, indexes: TensorHandle]
    readonly attributes: { readonly dim: number }
  }
  readonly crossEntropy: {
    readonly inputs: readonly [self: TensorHandle, target: TensorHandle]
    readonly attributes: { readonly ignoreIndex: number }
  }
  readonly scaledDotProductAttention: {
    readonly inputs: readonly [q: TensorHandle, k: TensorHandle, v: TensorHandle]
    readonly attributes: { readonly scale: number; readonly causal: boolean }
  }
  readonly positionEmbedding: {
    readonly inputs: readonly [weight: TensorHandle]
    readonly attributes: { readonly seqLen: number }
  }
  readonly rotaryEmbedding: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly seqLen: number; readonly theta: number }
  }
  readonly layerNorm: {
    readonly inputs: readonly [self: TensorHandle, weight: TensorHandle, bias: TensorHandle]
    readonly attributes: { readonly eps: number }
  }
  readonly linear: {
    readonly inputs: readonly [self: TensorHandle, weight: TensorHandle, bias: TensorHandle]
  }
  readonly conv1d: {
    readonly inputs: readonly [self: TensorHandle, weight: TensorHandle]
    readonly attributes: {
      readonly stride: number
      readonly padding: number
      readonly dilation: number
      readonly groups: number
    }
  }
  readonly conv2d: {
    readonly inputs: readonly [self: TensorHandle, weight: TensorHandle]
    readonly attributes: {
      readonly stride: number
      readonly padding: number
      readonly dilation: number
      readonly groups: number
    }
  }
  readonly sum: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly dims: ReadonlyArray<number>; readonly keepdims: boolean }
  }
  readonly prod: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly dims: ReadonlyArray<number>; readonly keepdims: boolean }
  }
  readonly mean: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly dims: ReadonlyArray<number>; readonly keepdims: boolean }
  }
  readonly max: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly dims: ReadonlyArray<number>; readonly keepdims: boolean }
  }
  readonly min: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly dims: ReadonlyArray<number>; readonly keepdims: boolean }
  }
  readonly reshape: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly shape: ReadonlyArray<number> }
  }
  readonly permute: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly dims: ReadonlyArray<number> }
  }
  readonly slice: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly ranges: ReadonlyArray<ReadonlyArray<number>> }
  }
  readonly broadcastTo: {
    readonly inputs: readonly [self: TensorHandle]
    readonly attributes: { readonly shape: ReadonlyArray<number> }
  }
  readonly vmap: {
    readonly inputs: readonly [y: TensorHandle, x: TensorHandle, batchedX: TensorHandle]
    readonly attributes: { readonly dim: number }
  }
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
  readonly adamwOut: {
    readonly inputs: readonly [step: TensorHandle]
    readonly attributes: { readonly index: number }
  }
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

/** A named tensor entry for a direct safetensors write. */
export interface PathSafetensorsSaveEntry {
  readonly name: string
  readonly tensor: TensorHandle
}

/** A named materialized entry returned by a direct safetensors read. */
export interface PathSafetensorsLoadEntry {
  readonly name: string
  readonly tensor: ConcreteTensorHandle
}

export interface PathSafetensorsSaveArchive {
  readonly entries: ReadonlyArray<PathSafetensorsSaveEntry>
  readonly metadata: Readonly<Record<string, string>>
}

export interface PathSafetensorsLoadArchive {
  readonly entries: ReadonlyArray<PathSafetensorsLoadEntry>
  readonly metadata: Readonly<Record<string, string>>
}

export interface DecodeProgramValue {
  readonly handle: DecodeProgramHandle
  readonly batch: number
  readonly layers: number
  readonly kvHeads: number
  readonly headDim: number
}

export interface PathSafetensors {
  readonly save: (path: string, archive: PathSafetensorsSaveArchive) => Effect.Effect<void, BackendError>
  readonly load: (path: string) => Effect.Effect<PathSafetensorsLoadArchive, BackendError>
}

export interface DecodeRuntime {
  readonly compile: (
    roots: ReadonlyArray<TensorHandle>,
    window?: number,
    batch?: number
  ) => Effect.Effect<DecodeProgramValue, BackendError>
  readonly makePool: (options: {
    readonly layers: number
    readonly kvHeads: number
    readonly headDim: number
    readonly maxTokens: number
    readonly blockSize: number
    readonly dtype: DType
  }) => Effect.Effect<KvPoolHandle, BackendError>
  readonly makeSequence: (pool: KvPoolHandle) => Effect.Effect<KvSequenceHandle, BackendError>
  readonly prefillMatch: (
    sequence: KvSequenceHandle,
    tokens: ReadonlyArray<number>
  ) => Effect.Effect<number, BackendError>
  readonly sequenceCursor: (sequence: KvSequenceHandle) => Effect.Effect<number, BackendError>
  readonly releaseSequence: (sequence: KvSequenceHandle) => Effect.Effect<void, BackendError>
  readonly run: (
    program: DecodeProgramHandle,
    inputs: ReadonlyArray<ConcreteTensorHandle>,
    sequence: KvSequenceHandle,
    tokens: ReadonlyArray<number>
  ) => Effect.Effect<ReadonlyArray<ConcreteTensorHandle>, BackendError>
  readonly runBatched: (
    program: DecodeProgramHandle,
    inputs: ReadonlyArray<ConcreteTensorHandle>,
    sequences: ReadonlyArray<KvSequenceHandle>,
    tokens: ReadonlyArray<ReadonlyArray<number>>
  ) => Effect.Effect<ReadonlyArray<ConcreteTensorHandle>, BackendError>
}

export interface RuntimeDiagnostics {
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
  readonly backend: BackendInfo
  readonly placement: Placement
  readonly capabilities: Capabilities
  readonly node: (request: NodeRequest) => Effect.Effect<LazyTensorHandle, BackendError>
  readonly evaluate: (
    roots: ReadonlyArray<TensorHandle>
  ) => Effect.Effect<ReadonlyArray<ConcreteTensorHandle>, BackendError>
  readonly grad: (
    loss: TensorHandle,
    wrt: ReadonlyArray<TensorHandle>
  ) => Effect.Effect<ReadonlyArray<LazyTensorHandle>, BackendError>
  readonly compile: (roots: ReadonlyArray<TensorHandle>) => Effect.Effect<ProgramHandle, BackendError>
  readonly run: (
    program: ProgramHandle,
    inputs: ReadonlyArray<ConcreteTensorHandle>,
    scalars: ReadonlyArray<number>
  ) => Effect.Effect<ReadonlyArray<ConcreteTensorHandle>, BackendError>
  readonly readback: (tensor: ConcreteTensorHandle) => Effect.Effect<ArrayBuffer, BackendError>
  readonly release: (tensor: ConcreteTensorHandle) => Effect.Effect<void, BackendError>
  readonly extensions: {
    readonly pathSafetensors?: PathSafetensors
    readonly decode?: DecodeRuntime
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

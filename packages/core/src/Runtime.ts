import { Context, Data, type Effect } from "effect"
import type { DType } from "./Tensor.ts"

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

declare const GraphHandleTypeId: unique symbol
declare const BufferHandleTypeId: unique symbol
declare const ProgramHandleTypeId: unique symbol
declare const DecodeProgramHandleTypeId: unique symbol
declare const KvPoolHandleTypeId: unique symbol
declare const KvSequenceHandleTypeId: unique symbol

export interface GraphHandle {
  readonly [GraphHandleTypeId]: typeof GraphHandleTypeId
  add(other: GraphHandle): GraphHandle
  sub(other: GraphHandle): GraphHandle
  mul(other: GraphHandle): GraphHandle
  div(other: GraphHandle): GraphHandle
  maximum(other: GraphHandle): GraphHandle
  minimum(other: GraphHandle): GraphHandle
  eq(other: GraphHandle): GraphHandle
  gt(other: GraphHandle): GraphHandle
  lt(other: GraphHandle): GraphHandle
  ge(other: GraphHandle): GraphHandle
  le(other: GraphHandle): GraphHandle
  matmul(other: GraphHandle): GraphHandle
  inverse(): GraphHandle
  det(): GraphHandle
  solve(other: GraphHandle): GraphHandle
  neg(): GraphHandle
  abs(): GraphHandle
  sqrt(): GraphHandle
  exp(): GraphHandle
  tanh(): GraphHandle
  gelu(approximate?: boolean | null): GraphHandle
  relu(): GraphHandle
  erf(): GraphHandle
  floor(): GraphHandle
  ceil(): GraphHandle
  round(): GraphHandle
  sign(): GraphHandle
  whereCond(a: GraphHandle, b: GraphHandle): GraphHandle
  argmax(dim: number): GraphHandle
  argmin(dim: number): GraphHandle
  cumsum(dim: number): GraphHandle
  indexSelect(dim: number, indexes: GraphHandle): GraphHandle
  scatterAdd(dim: number, indexes: GraphHandle, src: GraphHandle): GraphHandle
  gather(dim: number, indexes: GraphHandle): GraphHandle
  crossEntropy(target: GraphHandle, ignoreIndex: number): GraphHandle
  scaledDotProductAttention(k: GraphHandle, v: GraphHandle, scale: number, causal: boolean): GraphHandle
  positionEmbedding(seqLen: number): GraphHandle
  rotaryEmbedding(seqLen: number, theta: number): GraphHandle
  layerNorm(weight: GraphHandle, bias: GraphHandle, eps: number): GraphHandle
  linear(weight: GraphHandle, bias: GraphHandle): GraphHandle
  conv1d(weight: GraphHandle, stride: number, padding: number, dilation: number, groups: number): GraphHandle
  conv2d(weight: GraphHandle, stride: number, padding: number, dilation: number, groups: number): GraphHandle
  log(): GraphHandle
  sin(): GraphHandle
  cos(): GraphHandle
  pow(exp: number): GraphHandle
  cast(dtype: DType): GraphHandle
  sum(dims: Array<number>, keepdims: boolean): GraphHandle
  prod(dims: Array<number>, keepdims: boolean): GraphHandle
  mean(dims: Array<number>, keepdims: boolean): GraphHandle
  max(dims: Array<number>, keepdims: boolean): GraphHandle
  min(dims: Array<number>, keepdims: boolean): GraphHandle
  reshape(shape: Array<number>): GraphHandle
  permute(dims: Array<number>): GraphHandle
  slice(ranges: Array<Array<number>>): GraphHandle
  concat(other: GraphHandle, dim: number): GraphHandle
  broadcastTo(shape: Array<number>): GraphHandle
  stopGradient(): GraphHandle
  checkpoint(): GraphHandle
  vmap(x: GraphHandle, batchedX: GraphHandle, dim: number): GraphHandle
  adamwStep(
    grad: GraphHandle,
    m: GraphHandle,
    v: GraphHandle,
    lr: GraphHandle,
    c1: GraphHandle,
    c2: GraphHandle,
    beta1: number,
    beta2: number,
    eps: number,
    weightDecay: number
  ): GraphHandle
  adamwOut(index: number): GraphHandle
  sgdStep(
    grad: GraphHandle,
    velocity: GraphHandle,
    first: GraphHandle,
    lr: GraphHandle,
    momentum: number,
    dampening: number,
    nesterov: boolean,
    weightDecay: number
  ): GraphHandle
  sgdOut(index: number): GraphHandle
}

export interface BufferHandle {
  readonly [BufferHandleTypeId]: typeof BufferHandleTypeId
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

export interface GraphFactory {
  readonly constant: (value: number, dtype: DType) => GraphHandle
  readonly zeros: (shape: Array<number>, dtype: DType) => GraphHandle
  readonly ones: (shape: Array<number>, dtype: DType) => GraphHandle
  readonly full: (shape: Array<number>, value: number, dtype: DType) => GraphHandle
  readonly randn: (shape: Array<number>, dtype: DType) => GraphHandle
  readonly uniform: (shape: Array<number>, lo: number, hi: number, dtype: DType) => GraphHandle
  readonly arange: (start: number, end: number, step: number, dtype: DType) => GraphHandle
  readonly eye: (n: number, dtype: DType) => GraphHandle
  /** Snapshots `data` before returning; the caller retains ownership. */
  readonly fromBytes: (data: Uint8Array, shape: Array<number>, dtype: DType) => GraphHandle
  readonly fromBuffer: (buffer: BufferHandle) => GraphHandle
  readonly input: (slot: number, shape: Array<number>, dtype: DType) => GraphHandle
  readonly scalarInput: (slot: number, dtype: DType) => GraphHandle
}

export interface BufferValue {
  readonly handle: BufferHandle
  readonly shape: ReadonlyArray<number>
  readonly dtype: DType
  readonly placement: Placement
}

export interface DecodeProgramValue {
  readonly handle: DecodeProgramHandle
  readonly batch: number
  readonly layers: number
  readonly kvHeads: number
  readonly headDim: number
}

export interface PathSafetensors {
  readonly save: (
    path: string,
    names: ReadonlyArray<string>,
    tensors: ReadonlyArray<GraphHandle>
  ) => Effect.Effect<void, BackendError>
  readonly load: (path: string) => Effect.Effect<ReadonlyArray<readonly [string, BufferValue]>, BackendError>
}

export interface DecodeRuntime {
  readonly compile: (
    roots: ReadonlyArray<GraphHandle>,
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
    inputs: ReadonlyArray<BufferHandle>,
    sequence: KvSequenceHandle,
    tokens: ReadonlyArray<number>
  ) => Effect.Effect<ReadonlyArray<BufferValue>, BackendError>
  readonly runBatched: (
    program: DecodeProgramHandle,
    inputs: ReadonlyArray<BufferHandle>,
    sequences: ReadonlyArray<KvSequenceHandle>,
    tokens: ReadonlyArray<ReadonlyArray<number>>
  ) => Effect.Effect<ReadonlyArray<BufferValue>, BackendError>
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
  /** Stable identity used to isolate backend-owned caches. */
  readonly identity: object
  readonly backend: BackendInfo
  readonly placement: Placement
  readonly capabilities: Capabilities
  readonly graph: GraphFactory
  readonly validateGraph: (handles: ReadonlyArray<GraphHandle>) => Effect.Effect<void, BackendError>
  readonly evaluate: (roots: ReadonlyArray<GraphHandle>) => Effect.Effect<ReadonlyArray<BufferValue>, BackendError>
  readonly grad: (
    loss: GraphHandle,
    wrt: ReadonlyArray<GraphHandle>
  ) => Effect.Effect<ReadonlyArray<GraphHandle>, BackendError>
  readonly compile: (roots: ReadonlyArray<GraphHandle>) => Effect.Effect<ProgramHandle, BackendError>
  readonly run: (
    program: ProgramHandle,
    inputs: ReadonlyArray<BufferHandle>,
    scalars: ReadonlyArray<number>
  ) => Effect.Effect<ReadonlyArray<BufferValue>, BackendError>
  readonly readback: (buffer: BufferHandle) => Effect.Effect<ArrayBuffer, BackendError>
  readonly releaseBuffer: (buffer: BufferHandle) => Effect.Effect<void, BackendError>
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

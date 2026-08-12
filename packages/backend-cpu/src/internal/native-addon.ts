/* Structural types for the private napi-rs addon. */
/** @internal */
export declare class CancellationToken {
  constructor()
  cancel(): void
  get cancelled(): boolean
}

/** @internal */
export declare class Executable {
  get batch(): number
  get stateful(): boolean
  get allowsWindowEviction(): boolean
  get layers(): number
  get kvHeads(): number
  get headDim(): number
  get kdaLayers(): number
  get kdaHeads(): number
  get kdaHeadDim(): number
  get kdaValueDim(): number
  get convLayers(): number
  get convChannels(): number
  get convKernel(): number
  get diagnostics(): NativeExecutableDiagnostics
  execute(
    inputs: Array<NativeTensor>,
    scalars: Array<number>,
    sequences?: Array<NativeKvSequence> | undefined | null,
    tokens?: Array<Array<number>> | undefined | null,
    token?: CancellationToken | undefined | null
  ): Promise<Array<NativeTensor>>
}

/** @internal */
export declare class LazyTensor {
  get shape(): Array<number>
  get dtype(): string
  metadata(): [Array<number>, string]
  static zeros(shape: Array<number>, dtype?: NativeDType | undefined | null): LazyTensor
  static ones(shape: Array<number>, dtype?: NativeDType | undefined | null): LazyTensor
  static full(shape: Array<number>, value: number, dtype?: NativeDType | undefined | null): LazyTensor
  static randn(shape: Array<number>, dtype?: NativeDType | undefined | null): LazyTensor
  static uniform(shape: Array<number>, lo: number, hi: number, dtype?: NativeDType | undefined | null): LazyTensor
  static arange(start: number, end: number, step: number, dtype?: NativeDType | undefined | null): LazyTensor
  static eye(n: number, dtype?: NativeDType | undefined | null): LazyTensor
  static constant(value: number, dtype?: NativeDType | undefined | null): LazyTensor
  static fromBytes(data: Uint8Array, shape: Array<number>, dtype?: NativeDType | undefined | null): LazyTensor
  static fromMaterialized(tensor: NativeTensor): LazyTensor
  static input(slot: number, shape: Array<number>, dtype?: NativeDType | undefined | null): LazyTensor
  static scalarInput(slot: number, dtype?: NativeDType | undefined | null): LazyTensor
  add(other: LazyTensor): LazyTensor
  sub(other: LazyTensor): LazyTensor
  mul(other: LazyTensor): LazyTensor
  div(other: LazyTensor): LazyTensor
  maximum(other: LazyTensor): LazyTensor
  minimum(other: LazyTensor): LazyTensor
  eq(other: LazyTensor): LazyTensor
  gt(other: LazyTensor): LazyTensor
  lt(other: LazyTensor): LazyTensor
  ge(other: LazyTensor): LazyTensor
  le(other: LazyTensor): LazyTensor
  matmul(other: LazyTensor): LazyTensor
  inverse(): LazyTensor
  det(): LazyTensor
  solve(other: LazyTensor): LazyTensor
  neg(): LazyTensor
  abs(): LazyTensor
  sqrt(): LazyTensor
  exp(): LazyTensor
  tanh(): LazyTensor
  gelu(approximate?: boolean | undefined | null): LazyTensor
  relu(): LazyTensor
  erf(): LazyTensor
  floor(): LazyTensor
  ceil(): LazyTensor
  round(): LazyTensor
  sign(): LazyTensor
  whereCond(a: LazyTensor, b: LazyTensor): LazyTensor
  argmax(dim: number): LazyTensor
  argmin(dim: number): LazyTensor
  cumsum(dim: number): LazyTensor
  indexSelect(dim: number, indexes: LazyTensor): LazyTensor
  scatterAdd(dim: number, indexes: LazyTensor, src: LazyTensor): LazyTensor
  gather(dim: number, indexes: LazyTensor): LazyTensor
  crossEntropy(target: LazyTensor, ignoreIndex: number): LazyTensor
  scaledDotProductAttention(k: LazyTensor, v: LazyTensor, scale: number, causal: boolean, window: number): LazyTensor
  kdaChunk(k: LazyTensor, v: LazyTensor, logDecay: LazyTensor, beta: LazyTensor, scale: number): LazyTensor
  shortConv1d(weight: LazyTensor): LazyTensor
  positionEmbedding(seqLen: number): LazyTensor
  rotaryEmbedding(seqLen: number, theta: number, layout: NativeRotaryLayout): LazyTensor
  layerNorm(weight: LazyTensor, bias: LazyTensor, eps: number): LazyTensor
  rmsNorm(weight: LazyTensor | undefined, eps: number): LazyTensor
  linear(weight: LazyTensor, bias: LazyTensor): LazyTensor
  quantizedLinear(
    weight: LazyTensor,
    bias: LazyTensor | undefined | null,
    encoding: NativeGgmlKQuant,
    rows: number,
    columns: number
  ): LazyTensor
  quantizedEmbedding(
    weight: LazyTensor,
    encoding: NativeGgmlKQuant,
    rows: number,
    columns: number,
    paddingIndex?: number | undefined | null
  ): LazyTensor
  conv1d(w: LazyTensor, stride: number, padding: number, dilation: number, groups: number): LazyTensor
  conv2d(w: LazyTensor, stride: number, padding: number, dilation: number, groups: number): LazyTensor
  log(): LazyTensor
  sin(): LazyTensor
  cos(): LazyTensor
  pow(exp: number): LazyTensor
  cast(dtype: NativeDType): LazyTensor
  sum(dims: Array<number>, keepdims: boolean): LazyTensor
  prod(dims: Array<number>, keepdims: boolean): LazyTensor
  mean(dims: Array<number>, keepdims: boolean): LazyTensor
  max(dims: Array<number>, keepdims: boolean): LazyTensor
  min(dims: Array<number>, keepdims: boolean): LazyTensor
  reshape(shape: Array<number>): LazyTensor
  permute(dims: Array<number>): LazyTensor
  slice(ranges: Array<Array<number>>): LazyTensor
  concat(other: LazyTensor, dim: number): LazyTensor
  broadcastTo(shape: Array<number>): LazyTensor
  stopGradient(): LazyTensor
  checkpoint(): LazyTensor
  vmap(x: LazyTensor, batchedX: LazyTensor, dim: number): LazyTensor
  adamwStep(
    grad: LazyTensor,
    m: LazyTensor,
    v: LazyTensor,
    lr: LazyTensor,
    c1: LazyTensor,
    c2: LazyTensor,
    beta1: number,
    beta2: number,
    eps: number,
    weightDecay: number
  ): LazyTensor
  adamwOut(index: number): LazyTensor
  sgdStep(
    grad: LazyTensor,
    velocity: LazyTensor,
    first: LazyTensor,
    lr: LazyTensor,
    momentum: number,
    dampening: number,
    nesterov: boolean,
    weightDecay: number
  ): LazyTensor
  sgdOut(index: number): LazyTensor
}

/** @internal */
export declare class NativeKvPool {
  constructor(
    layers: number,
    kvHeads: number,
    headDim: number,
    maxTokens: number,
    blockSize?: number | undefined | null,
    dtype?: NativeDType | undefined | null,
    recurrent?: NativeRecurrentStateSchema | undefined | null
  )
  get capacity(): number
  get freeBlocks(): number
  get cachedBlocks(): number
  makeSequence(): NativeKvSequence
}

/** @internal */
export declare class NativeKvSequence {
  get cursor(): number
  release(): void
  prefillMatch(tokens: Array<number>): number
}

/** @internal */
export declare class NativeTensor {
  /**
   * Releases the tensor's buffer early instead of waiting for the
   * garbage collector. Using the handle — or any lazy graph built
   * from it — afterwards is a typed error.
   */
  clear(): void
  get shape(): Array<number>
  get dtype(): string
  get device(): string
  readback(token?: CancellationToken | undefined | null): Promise<ArrayBuffer>
}

/** @internal */
export declare function compile(
  roots: Array<LazyTensor>,
  options?: NativeCompileOptions | undefined | null,
  state?: NativeKvStateSchema | undefined | null,
  cacheKey?: string | undefined | null
): Executable

/** @internal */
export declare function externalMemoryBytes(): number

/** @internal */
export declare function grad(loss: LazyTensor, wrt: Array<LazyTensor>): Array<LazyTensor>

/** @internal */
export declare function isAvailable(): boolean

/** @internal */
export declare function loadTensors(
  path: string,
  token?: CancellationToken | undefined | null
): Promise<NativeSafetensorsArchive>

/** @internal */
export type NativeGgmlKQuant = "Q2_K" | "Q3_K" | "Q4_K" | "Q5_K" | "Q6_K"

/** @internal */
export interface NativeGgufMetadataEntry {
  key: string
  kind: string
  numberValue?: number
  stringValue?: string
  booleanValue?: boolean
  numberArray?: Array<number>
  stringArray?: Array<string>
  booleanArray?: Array<boolean>
}

/** @internal */
export interface NativeGgufTensorDescriptor {
  name: string
  format: "F32" | NativeGgmlKQuant
  logicalShape: Array<number>
  logicalDtype: "f32"
  physicalShape: Array<number>
  physicalDtype: "f32" | "u8"
}

/** @internal */
export interface NativeGgufInspection {
  metadata: Array<NativeGgufMetadataEntry>
  tensors: Array<NativeGgufTensorDescriptor>
}

/** @internal */
export interface NativeGgufArchive {
  entries: Array<{ descriptor: NativeGgufTensorDescriptor; tensor: NativeTensor }>
}

/** @internal */
export declare function inspectGguf(
  path: string,
  token?: CancellationToken | undefined | null
): Promise<NativeGgufInspection>

/** @internal */
export declare function loadGguf(
  path: string,
  token?: CancellationToken | undefined | null
): Promise<NativeGgufArchive>

/** @internal */
export declare const enum NativeDType {
  F32 = "f32",
  F64 = "f64",
  I64 = "i64",
  U8 = "u8",
  U32 = "u32",
  F16 = "f16",
  BF16 = "bf16"
}

export type NativeRotaryLayout = "HalfSplit" | "InterleavedPairs"

/** @internal */
export interface NativeCompileOptions {
  optimize?: boolean
  constantWeights?: boolean
}

/** @internal */
export interface NativeKvStateSchema {
  maxTokens: number
  blockSize: number
  kvDtype: NativeDType
  window?: number
  batch: number
  lastTokenRow?: boolean
}

/** @internal */
export interface NativeRecurrentStateSchema {
  kdaLayers: number
  kdaHeads: number
  kdaHeadDim: number
  kdaValueDim: number
  convLayers: number
  convChannels: number
  convKernel: number
}

/** @internal */
export interface NativeExecutableDiagnostics {
  semanticNodesBeforeOptimization: number
  semanticNodesAfterOptimization: number
  instructions: Array<{ kind: string; count: number }>
  pipelineCount: number
  commandCount: number
  synchronizationCount: number
  memory: {
    externalBytes: number
    persistentBytes: number
    stateBytes: number
    outputBytes: number
    workspaceBytes: number
    transactionBytes: number
    peakLiveBytes: number
    packingOverheadBytes: number
  }
  compilePhases: Array<{ phase: string; nanoseconds: number }>
}

/** @internal */
export interface NativeSafetensorsArchive {
  entries: Array<NativeSafetensorsEntry>
  metadata: Record<string, string>
}

/** @internal */
export interface NativeSafetensorsEntry {
  name: string
  tensor: NativeTensor
}

/** @internal */
export declare function saveTensors(
  path: string,
  names: Array<string>,
  tensors: Array<NativeTensor>,
  metadata: Record<string, string>,
  token?: CancellationToken | undefined | null
): Promise<void>

/** @internal */
export interface NativeAddon {
  readonly CancellationToken: typeof CancellationToken
  readonly Executable: typeof Executable
  readonly LazyTensor: typeof LazyTensor
  readonly NativeKvPool: typeof NativeKvPool
  readonly NativeKvSequence: typeof NativeKvSequence
  readonly NativeTensor: typeof NativeTensor
  readonly compile: typeof compile
  readonly externalMemoryBytes: typeof externalMemoryBytes
  readonly grad: typeof grad
  readonly isAvailable: typeof isAvailable
  readonly inspectGguf: typeof inspectGguf
  readonly loadGguf: typeof loadGguf
  readonly loadTensors: typeof loadTensors
  readonly saveTensors: typeof saveTensors
}

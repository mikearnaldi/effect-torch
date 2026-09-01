import { Runtime } from "@effect-torch/core"
import { Effect, Predicate } from "effect"
import { pipeArguments } from "effect/Pipeable"
import type {
  CancellationToken,
  CudaRuntime,
  Executable,
  LazyTensor,
  NativeAddon,
  NativeCurrentBlockAttention,
  NativeDecodeOutputSelection,
  NativeGgufMetadataEntry,
  NativeGgufTensorDescriptor,
  NativeKvPool,
  NativeKvSequence,
  NativeKvStateSchema,
  NativeTensor
} from "./native-addon.js"

const backendName = "@effect-torch/backend-cuda"

interface TensorRecord {
  readonly owner: object
  readonly kind: "lazy" | "concrete"
  readonly graph: LazyTensor
  readonly value?: NativeTensor | undefined
  disposed: boolean
}

interface ExecutableRecord {
  readonly owner: object
  readonly kind: "executable"
  readonly value: Executable
  readonly outputs: ReadonlyArray<{
    readonly shape: ReadonlyArray<number>
    readonly dtype: Runtime.DType
    readonly storage?: Runtime.EncodedTensorStorage | undefined
  }>
  readonly state?: Runtime.DecodeStateSchema | undefined
  readonly sourceShapes: ReadonlyArray<ReadonlyArray<number>>
}

type HandleRecord = TensorRecord | ExecutableRecord

const records = new WeakMap<object, HandleRecord>()
interface PoolRecord {
  readonly value: NativeKvPool
  readonly options: Parameters<Runtime.DecodeRuntime["makePool"]>[0]
}
interface SequenceRecord {
  readonly value: NativeKvSequence
  readonly pool: PoolRecord
  disposed: boolean
}
const poolRecords = new WeakMap<object, PoolRecord>()
const sequenceRecords = new WeakMap<object, SequenceRecord>()
interface InferenceArtifactRecord {
  readonly request: Runtime.InferenceCompileRequest
  readonly diagnostics: {
    roundsStarted: bigint
    roundsCompleted: bigint
    ordinaryRounds: bigint
    speculativeRounds: bigint
    proposedTokens: bigint
    acceptedTokens: bigint
    emittedTokens: bigint
    draftNanos: bigint
    verificationNanos: bigint
    acceptedLengthHistogram: Array<bigint>
    lastRoundId?: bigint | undefined
  }
}
interface InferenceSessionRecord {
  readonly artifact: InferenceArtifactRecord
  readonly sequences: Map<bigint, InferenceSequenceRecord>
  readonly receipts: Set<bigint>
  readonly tokenBindings: Array<TokenBindingRecord>
  nextSequenceId: bigint
  nextRoundId: bigint
  closed: boolean
}
interface TokenBindingRecord {
  readonly value: NativeTensor
  readonly shape: ReadonlyArray<number>
  readonly dtype: "u32" | "i64"
  inUse: boolean
}
interface InferenceSequenceRecord {
  readonly session: InferenceSessionRecord
  readonly id: bigint
  readonly target: NativeKvSequence
  readonly proposer?: NativeKvSequence | undefined
  readonly handle: Runtime.InferenceSequenceHandle
  readonly eos: Set<number>
  readonly maxTokens?: number | undefined
  readonly sampling: Runtime.InferenceSamplingOptions
  readonly history: Array<number>
  pending: number
  generated: number
  terminal?: "eos" | "maxTokens" | undefined
  finished: boolean
}
const inferenceArtifactRecords = new WeakMap<object, InferenceArtifactRecord>()
const inferenceSessionRecords = new WeakMap<object, InferenceSessionRecord>()
const inferenceSequenceRecords = new WeakMap<object, InferenceSequenceRecord>()
const backendHandlesKey = Symbol.for("@effect-torch/backend-handles")
interface BackendHandleRegistry {
  [backendHandlesKey]?: WeakSet<object>
}
// SAFETY: backend adapters reserve this global symbol for a WeakSet<object>.
const registry = globalThis as typeof globalThis & BackendHandleRegistry
const backendHandles = registry[backendHandlesKey] ??= new WeakSet<object>()

const opaqueHandle = <H extends object>(): H => {
  // SAFETY: runtime handles are opaque identity tokens whose state lives in a WeakMap.
  return Object.freeze({}) as H
}

const errorFor = (
  operation: string,
  phase: Runtime.BackendError["phase"],
  fallback: Runtime.BackendError["reason"]
) =>
(cause: unknown): Runtime.BackendError => {
  if (cause instanceof Runtime.BackendError) return cause
  const message = cause instanceof Error ? cause.message : String(cause)
  const reason: Runtime.BackendError["reason"] = message.includes("only f32") || message.includes("dtype")
    ? "unsupported-dtype"
    : message.includes("not supported")
    ? "unsupported-operation"
    : message.includes("cleared")
    ? "invalid-handle"
    : fallback
  return new Runtime.BackendError({
    reason,
    backend: backendName,
    operation,
    phase,
    message,
    details: { cause }
  })
}

const unsupported = (
  operation: string,
  phase: Runtime.BackendError["phase"] = "execute"
): Effect.Effect<never, Runtime.BackendError> =>
  Effect.fail(
    new Runtime.BackendError({
      reason: "unsupported-operation",
      backend: backendName,
      operation,
      phase,
      message: `${operation} is not supported by the CUDA backend yet`
    })
  )

const cancellable = <A>(
  native: NativeAddon,
  operation: string,
  phase: Runtime.BackendError["phase"],
  run: (token: CancellationToken) => Promise<A>,
  clearLate?: (value: A) => void
): Effect.Effect<A, Runtime.BackendError> =>
  Effect.callback<A, Runtime.BackendError>((resume, signal) => {
    const token = new native.CancellationToken()
    let lateValue: A | undefined
    let hasLateValue = false
    const clearLateValue = () => {
      if (!hasLateValue) return
      hasLateValue = false
      try {
        // SAFETY: hasLateValue is set only after lateValue receives a resolved A.
        clearLate?.(lateValue as A)
      } catch {
        // The interrupted caller cannot observe cleanup failure.
      }
      lateValue = undefined
    }
    const abort = () => {
      token.cancel()
      if (token.cancelled) clearLateValue()
    }
    if (signal.aborted) abort()
    else signal.addEventListener("abort", abort, { once: true })
    let pending: Promise<A>
    try {
      pending = run(token)
    } catch (cause) {
      signal.removeEventListener("abort", abort)
      resume(Effect.fail(errorFor(operation, phase, "execution-failed")(cause)))
      return
    }
    pending.then(
      (value) => {
        if (signal.aborted && token.cancelled) {
          lateValue = value
          hasLateValue = true
          clearLateValue()
          return
        }
        lateValue = value
        hasLateValue = true
        resume(Effect.suspend(() => {
          signal.removeEventListener("abort", abort)
          if (!hasLateValue) return Effect.interrupt
          hasLateValue = false
          lateValue = undefined
          return Effect.succeed(value)
        }))
      },
      (cause) => {
        signal.removeEventListener("abort", abort)
        resume(
          token.cancelled || (cause instanceof Error && cause.message.includes("aborted"))
            ? Effect.interrupt
            : Effect.fail(errorFor(operation, phase, "execution-failed")(cause))
        )
      }
    )
  })

const dtype = (value: string): Runtime.DType => {
  if (
    value === "f64" || value === "f32" || value === "f16" || value === "bf16" || value === "i64" ||
    value === "u32" || value === "u8"
  ) return value
  throw new Error(`native CUDA runtime returned unsupported dtype ${value}`)
}

const validShape = (value: ReadonlyArray<number>): boolean =>
  value.every((dimension) => Number.isSafeInteger(dimension) && dimension >= 0)

const sameShape = (left: ReadonlyArray<number>, right: ReadonlyArray<number>): boolean =>
  left.length === right.length && left.every((dimension, index) => dimension === right[index])

const isGgufFormat = (value: string): value is Runtime.GgufTensorDescriptor["format"] =>
  value === "F32" || value === "Q2_K" || value === "Q3_K" || value === "Q4_K" || value === "Q5_K" ||
  value === "Q6_K"

const encodedRowBytes = (encoding: Runtime.TensorStorageEncoding, columns: number): number | undefined => {
  if (columns % 256 !== 0) return undefined
  const blockBytes = encoding === "Q2_K"
    ? 84
    : encoding === "Q3_K"
    ? 110
    : encoding === "Q4_K"
    ? 144
    : encoding === "Q5_K"
    ? 176
    : 210
  return columns / 256 * blockBytes
}

const validEncodedGeometry = (
  logicalShape: ReadonlyArray<number>,
  storage: Runtime.EncodedTensorStorage
): boolean => {
  const columns = logicalShape.at(-1)
  const rows = logicalShape.slice(0, -1).reduce((total, dimension) => total * dimension, 1)
  const rowBytes = columns === undefined ? undefined : encodedRowBytes(storage.encoding, columns)
  return rowBytes !== undefined && sameShape(storage.physicalShape, [rows, rowBytes])
}

/** Builds the backend-neutral service around one native CUDA runtime. @internal */
export const createRuntimeAdapter = (native: NativeAddon, deviceOrdinal: number): Runtime.RuntimeService => {
  const owner = {}
  const placement: Runtime.Placement = Object.freeze({
    id: `cuda:${deviceOrdinal}`,
    deviceType: "cuda",
    description: `NVIDIA CUDA device ${deviceOrdinal}`,
    ordinal: deviceOrdinal
  })
  const runtime: CudaRuntime = new native.CudaRuntime(deviceOrdinal)
  const capabilities: Runtime.Capabilities = Object.freeze({
    dtypes: ["f64", "f32", "f16", "bf16", "i64", "u32", "u8"] as const,
    features: [] as const
  })

  const tensorObject = <H extends Runtime.TensorHandle>(
    tag: H["_tag"],
    shape: ReadonlyArray<number>,
    tensorDtype: string,
    nativeDevice: string,
    storage?: Runtime.EncodedTensorStorage
  ): H => {
    if (!validShape(shape)) throw new Error(`native CUDA runtime returned invalid shape [${shape}]`)
    if (nativeDevice !== placement.id) {
      throw new Error(`native CUDA runtime returned placement ${nativeDevice}, expected ${placement.id}`)
    }
    // SAFETY: the tag selects H and the object supplies every TensorHandle field.
    return Object.freeze({
      _tag: tag,
      shape: Object.freeze([...shape]),
      dtype: dtype(tensorDtype),
      storage: storage === undefined
        ? undefined
        : Object.freeze({
          encoding: storage.encoding,
          physicalShape: Object.freeze([...storage.physicalShape]),
          physicalDtype: "u8" as const
        }),
      device: "cuda",
      placement,
      pipe(this: Runtime.TensorHandle) {
        return pipeArguments(this, arguments)
      }
    }) as H
  }

  const lazyHandle = (
    graph: LazyTensor,
    logical?: {
      readonly shape: ReadonlyArray<number>
      readonly dtype: Runtime.DType
      readonly storage?: Runtime.EncodedTensorStorage | undefined
    }
  ): Runtime.LazyTensorHandle => {
    const handle = tensorObject<Runtime.LazyTensorHandle>(
      "LazyTensor",
      logical?.shape ?? graph.shape,
      logical?.dtype ?? graph.dtype,
      graph.device,
      logical?.storage
    )
    records.set(handle, { owner, kind: "lazy", graph, disposed: false })
    backendHandles.add(handle)
    return handle
  }

  const concreteHandle = (
    value: NativeTensor,
    logical?: {
      readonly shape: ReadonlyArray<number>
      readonly dtype: Runtime.DType
      readonly storage?: Runtime.EncodedTensorStorage | undefined
    }
  ): Runtime.ConcreteTensorHandle => {
    const expectedShape = logical?.storage?.physicalShape ?? logical?.shape
    const expectedDtype = logical?.storage?.physicalDtype ?? logical?.dtype
    if (expectedShape !== undefined && (!sameShape(value.shape, expectedShape) || value.dtype !== expectedDtype)) {
      throw new Error(
        `native CUDA runtime returned physical tensor ${value.dtype} [${value.shape}], expected ${expectedDtype} [${expectedShape}]`
      )
    }
    const graph = runtime.fromMaterialized(value)
    const handle = tensorObject<Runtime.ConcreteTensorHandle>(
      "Tensor",
      logical?.shape ?? value.shape,
      logical?.dtype ?? value.dtype,
      value.device,
      logical?.storage
    )
    records.set(handle, { owner, kind: "concrete", graph, value, disposed: false })
    backendHandles.add(handle)
    return handle
  }

  const handleRecord = (
    handle: Runtime.TensorHandle,
    operation: string,
    concrete = false
  ): TensorRecord => {
    const phase: Runtime.BackendError["phase"] = operation === "compile"
      ? "compile"
      : operation === "execute"
      ? "execute"
      : operation === "readback"
      ? "readback"
      : operation === "release"
      ? "shutdown"
      : "graph"
    const found = records.get(handle)
    if (found?.owner !== owner) {
      throw new Runtime.BackendError({
        reason: backendHandles.has(handle) ? "foreign-handle" : "invalid-handle",
        backend: backendName,
        operation,
        phase,
        message: `${operation}: tensor handle is not owned by this CUDA runtime`
      })
    }
    if (found.kind === "executable" || (concrete && found.kind !== "concrete")) {
      throw new Runtime.BackendError({
        reason: "invalid-handle",
        backend: backendName,
        operation,
        phase,
        message: `${operation}: tensor handle has the wrong kind`
      })
    }
    if (found.disposed) {
      throw new Runtime.BackendError({
        reason: "invalid-handle",
        backend: backendName,
        operation,
        phase,
        message: `${operation}: tensor handle was cleared`
      })
    }
    return found
  }

  const executableRecord = (handle: Runtime.ExecutableHandle, operation: string): ExecutableRecord => {
    const found = records.get(handle)
    if (found?.owner !== owner || found.kind !== "executable") {
      throw new Runtime.BackendError({
        reason: found === undefined && !backendHandles.has(handle) ? "invalid-handle" : "foreign-handle",
        backend: backendName,
        operation,
        phase: "execute",
        message: `${operation}: executable handle is not owned by this CUDA runtime`
      })
    }
    return found
  }

  const nativePool = (handle: Runtime.KvPoolHandle, operation: string): PoolRecord => {
    const found = poolRecords.get(handle)
    if (found === undefined) {
      throw new Runtime.BackendError({
        reason: backendHandles.has(handle) ? "foreign-handle" : "invalid-handle",
        backend: backendName,
        operation,
        phase: "execute",
        message: `${operation}: KV pool is not owned by this CUDA runtime`
      })
    }
    return found
  }

  const nativeSequence = (handle: Runtime.KvSequenceHandle, operation: string): SequenceRecord => {
    const found = sequenceRecords.get(handle)
    if (found === undefined || found.disposed) {
      throw new Runtime.BackendError({
        reason: found === undefined && backendHandles.has(handle) ? "foreign-handle" : "invalid-handle",
        backend: backendName,
        operation,
        phase: "execute",
        message: `${operation}: KV sequence is not live in this CUDA runtime`
      })
    }
    return found
  }

  // SAFETY: core and N-API expose the same string values for these decode enums.
  const nativeState = (state: Runtime.DecodeStateRequest): NativeKvStateSchema => ({
    maxTokens: state.maxTokens,
    blockSize: state.blockSize,
    kvDtype: state.kvDtype,
    window: state.window,
    currentBlockAttention: state.currentBlockAttention as NativeCurrentBlockAttention | undefined,
    batch: state.batch,
    packedCausalChains: state.packedCausalChains === undefined
      ? undefined
      : { rowsPerSequence: state.packedCausalChains.rowsPerSequence },
    lastTokenRow: state.lastTokenRow,
    outputSelections: state.outputSelections?.map((selection) =>
      selection === "allRows"
        ? "AllRows" as NativeDecodeOutputSelection
        : selection === "splitLastTokenRow"
        ? "SplitLastTokenRow" as NativeDecodeOutputSelection
        : "BatchedLastTokenRow" as NativeDecodeOutputSelection
    )
  })

  const inferenceArtifact = (
    handle: Runtime.InferenceArtifactHandle,
    operation: string
  ): InferenceArtifactRecord => {
    const found = inferenceArtifactRecords.get(handle)
    if (found === undefined) {
      throw new Error(`${operation}: inference artifact is not owned by this CUDA runtime`)
    }
    return found
  }

  const inferenceSession = (
    handle: Runtime.InferenceSessionHandle,
    operation: string
  ): InferenceSessionRecord => {
    const found = inferenceSessionRecords.get(handle)
    if (found === undefined || found.closed) {
      throw new Error(`${operation}: inference session is not live`)
    }
    return found
  }

  const inferenceSequence = (
    session: InferenceSessionRecord,
    handle: Runtime.InferenceSequenceHandle,
    operation: string
  ): InferenceSequenceRecord => {
    const found = inferenceSequenceRecords.get(handle)
    if (found === undefined || found.session !== session || found.finished) {
      throw new Error(`${operation}: inference sequence is not live in this session`)
    }
    return found
  }

  const foldedSeed = (seed: bigint): number => Number((seed ^ (seed >> 32n)) & 0x1f_ffff_ffff_ffffn)

  const resolvedSampling = (
    defaults: Runtime.InferenceSamplingOptions,
    overrides: Runtime.InferenceSamplingOverrides | undefined,
    sequenceId: bigint,
    position: number
  ): Runtime.SamplingOptions => ({
    temperature: overrides?.temperature ?? defaults.temperature,
    topK: overrides?.topK ?? defaults.topK,
    topP: overrides?.topP ?? defaults.topP,
    seed: foldedSeed(overrides?.seed ?? defaults.seed),
    counter: Number((sequenceId * 1_000_003n + BigInt(position)) % 9_007_199_254_740_991n)
  })

  const historyDraft = (
    history: ReadonlyArray<number>,
    minimum: number,
    maximum: number,
    limit: number
  ): ReadonlyArray<number> => {
    for (let width = Math.min(maximum, history.length - 1); width >= minimum; width--) {
      const suffix = history.slice(-width)
      for (let start = history.length - width - 1; start >= 0; start--) {
        if (suffix.every((token, index) => history[start + index] === token)) {
          return history.slice(start + width, start + width + limit)
        }
      }
    }
    return []
  }

  const readTokens = async (handle: Runtime.ConcreteTensorHandle, token: CancellationToken): Promise<Array<number>> => {
    const record = handleRecord(handle, "inferenceAdd", true)
    const promptLength = handle.shape.length === 1
      ? handle.shape[0]
      : handle.shape.length === 2 && handle.shape[0] === 1
      ? handle.shape[1]
      : undefined
    if (promptLength === undefined || (handle.dtype !== "u32" && handle.dtype !== "i64")) {
      throw new Error("inferenceAdd: prompt must be a [T] or [1, T] u32 or i64 tensor")
    }
    const bytes = await record.value!.readback(token)
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
    const values = new Array<number>(promptLength)
    for (let index = 0; index < values.length; index++) {
      const value = handle.dtype === "u32"
        ? view.getUint32(index * 4, true)
        : Number(view.getBigInt64(index * 8, true))
      if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
        throw new Error(`inferenceAdd: prompt token ${value} is outside u32 range`)
      }
      values[index] = value
    }
    return values
  }

  const uploadTokens = async (
    values: ReadonlyArray<number>,
    shape: ReadonlyArray<number>,
    tokenDtype: "u32" | "i64",
    session: InferenceSessionRecord
  ): Promise<TokenBindingRecord> => {
    const bytes = new Uint8Array(values.length * (tokenDtype === "u32" ? 4 : 8))
    const view = new DataView(bytes.buffer)
    for (let index = 0; index < values.length; index++) {
      if (tokenDtype === "u32") view.setUint32(index * 4, values[index]!, true)
      else view.setBigInt64(index * 8, BigInt(values[index]!), true)
    }
    const cached = session.tokenBindings.find((binding) =>
      !binding.inUse && binding.dtype === tokenDtype &&
      binding.shape.length === shape.length && binding.shape.every((dimension, index) => dimension === shape[index])
    )
    if (cached !== undefined) {
      cached.value.writeBytes(bytes)
      cached.inUse = true
      return cached
    }
    const binding: TokenBindingRecord = {
      value: runtime.uploadBytes(bytes, [...shape], tokenDtype),
      shape: [...shape],
      dtype: tokenDtype,
      inUse: true
    }
    session.tokenBindings.push(binding)
    return binding
  }

  const runStateBatch = async (options: {
    readonly executable: ExecutableRecord
    readonly sequences: ReadonlyArray<InferenceSequenceRecord>
    readonly slots: ReadonlyArray<number>
    readonly tokenRows: ReadonlyArray<ReadonlyArray<number>>
    readonly tokenDtype: "u32" | "i64"
    readonly sampling?: ReadonlyArray<Runtime.SamplingOptions> | undefined
    readonly token: CancellationToken
  }): Promise<ReadonlyArray<number> | undefined> => {
    const state = options.executable.state
    if (state === undefined) throw new Error("inference: executable is not stateful")
    const width = options.executable.sourceShapes[0]?.at(-2)
    if (width === undefined) throw new Error("inference: executable has no token width")
    const values = new Array<number>(state.batch * width).fill(0)
    const activeMask = new Array<boolean>(state.batch).fill(false)
    const validLengths = new Array<number>(state.batch).fill(0)
    for (let request = 0; request < options.sequences.length; request++) {
      const slot = options.slots[request]!
      const row = options.tokenRows[request]!
      if (slot < 0 || slot >= state.batch || row.length === 0 || row.length > width) {
        throw new Error("inference: invalid active slot or token chunk")
      }
      activeMask[slot] = true
      validLengths[slot] = row.length
      values.splice(slot * width, row.length, ...row)
    }
    const session = options.sequences[0]!.session
    const binding = await uploadTokens(values, [state.batch, width], options.tokenDtype, session)
    try {
      const nativeSequences = options.sequences.map((sequence) => sequence.target)
      if (options.sampling !== undefined) {
        return await options.executable.value.executeSampled(
          [binding.value],
          nativeSequences,
          [...options.slots],
          activeMask,
          validLengths,
          validLengths,
          options.tokenRows.map((row) => [...row]),
          options.sampling.map((sampling) => ({ ...sampling })),
          options.token
        )
      }
      const outputs = await options.executable.value.executeStateful(
        [binding.value],
        nativeSequences,
        [...options.slots],
        activeMask,
        validLengths,
        validLengths,
        options.tokenRows.map((row) => [...row]),
        options.token
      )
      for (const output of outputs) output.clear()
      return undefined
    } finally {
      binding.inUse = false
    }
  }

  const runStateBatchWithOutputs = async (options: {
    readonly executable: ExecutableRecord
    readonly sequences: ReadonlyArray<InferenceSequenceRecord>
    readonly slots: ReadonlyArray<number>
    readonly tokenRows: ReadonlyArray<ReadonlyArray<number>>
    readonly tokenDtype: "u32" | "i64"
    readonly sampling?: ReadonlyArray<Runtime.SamplingOptions> | undefined
    readonly token: CancellationToken
  }): Promise<{ readonly sampled?: ReadonlyArray<number>; readonly outputs: ReadonlyArray<NativeTensor> }> => {
    const state = options.executable.state
    if (state === undefined) throw new Error("inference: executable is not stateful")
    const width = options.executable.sourceShapes[0]?.at(-2)
    if (width === undefined) throw new Error("inference: executable has no token width")
    const values = new Array<number>(state.batch * width).fill(0)
    const activeMask = new Array<boolean>(state.batch).fill(false)
    const validLengths = new Array<number>(state.batch).fill(0)
    for (let request = 0; request < options.sequences.length; request++) {
      const slot = options.slots[request]!
      const row = options.tokenRows[request]!
      if (slot < 0 || slot >= state.batch || row.length === 0 || row.length > width) {
        throw new Error("inference: invalid active slot or token chunk")
      }
      activeMask[slot] = true
      validLengths[slot] = row.length
      values.splice(slot * width, row.length, ...row)
    }
    const session = options.sequences[0]!.session
    const binding = await uploadTokens(values, [state.batch, width], options.tokenDtype, session)
    try {
      const outputs = await options.executable.value.executeStateful(
        [binding.value],
        options.sequences.map((sequence) => sequence.target),
        [...options.slots],
        activeMask,
        validLengths,
        validLengths,
        options.tokenRows.map((row) => [...row]),
        options.token
      )
      try {
        const sampled = options.sampling === undefined
          ? undefined
          : await Promise.all(options.slots.map((slot, request) => {
            const output = outputs[slot]
            if (output === undefined) throw new Error("inference: target logits lane is missing")
            const sampling = options.sampling![request]!
            return output.sample(
              sampling.temperature,
              sampling.topK,
              sampling.topP,
              sampling.seed,
              sampling.counter,
              options.token
            )
          }))
        return sampled === undefined ? { outputs } : { sampled, outputs }
      } catch (cause) {
        for (const output of outputs) output.clear()
        throw cause
      }
    } finally {
      binding.inUse = false
    }
  }

  const runReplay = async (options: {
    readonly executable: ExecutableRecord
    readonly bindings: ReadonlyArray<NativeTensor>
    readonly sequences: ReadonlyArray<NativeKvSequence>
    readonly slots: ReadonlyArray<number>
    readonly lengths: ReadonlyArray<number>
    readonly tokens: ReadonlyArray<ReadonlyArray<number>>
    readonly token: CancellationToken
  }): Promise<void> => {
    const state = options.executable.state
    if (state === undefined) throw new Error("inference: replay executable is not stateful")
    const activeMask = new Array<boolean>(state.batch).fill(false)
    const validLengths = new Array<number>(state.batch).fill(0)
    for (let request = 0; request < options.sequences.length; request++) {
      const slot = options.slots[request]!
      activeMask[slot] = true
      validLengths[slot] = options.lengths[request]!
    }
    const outputs = await options.executable.value.executeStateful(
      [...options.bindings],
      [...options.sequences],
      [...options.slots],
      activeMask,
      validLengths,
      validLengths,
      options.tokens.map((row) => [...row]),
      options.token
    )
    for (const output of outputs) output.clear()
  }

  const hiddenTapOutputs = (
    outputs: ReadonlyArray<NativeTensor>,
    taps: ReadonlyArray<Runtime.InferenceTargetTapRoute>,
    batch: number,
    splitLogits: boolean
  ): ReadonlyArray<NativeTensor> =>
    taps.map((tap) => {
      const output = outputs[tap.outputRoot + (splitLogits ? batch - 1 : 0)]
      if (output === undefined) throw new Error("inference: hidden tap " + tap.name + " is missing")
      return output
    })

  const runSampledSteps = async (options: {
    readonly executable: ExecutableRecord
    readonly sequences: ReadonlyArray<InferenceSequenceRecord>
    readonly slots: ReadonlyArray<number>
    readonly tokens: ReadonlyArray<number>
    readonly tokenDtype: "u32" | "i64"
    readonly sampling: ReadonlyArray<ReadonlyArray<Runtime.SamplingOptions>>
    readonly token: CancellationToken
  }): Promise<ReadonlyArray<ReadonlyArray<number>>> => {
    const state = options.executable.state
    if (state === undefined) throw new Error("inference: executable is not stateful")
    const width = options.executable.sourceShapes[0]?.at(-2)
    if (width !== 1 || options.tokens.length !== options.sequences.length) {
      throw new Error("inference: multi-step decode requires one token per sequence")
    }
    const values = new Array<number>(state.batch).fill(0)
    const activeMask = new Array<boolean>(state.batch).fill(false)
    const validLengths = new Array<number>(state.batch).fill(0)
    for (let request = 0; request < options.sequences.length; request++) {
      const slot = options.slots[request]!
      if (slot < 0 || slot >= state.batch) throw new Error("inference: invalid active slot")
      activeMask[slot] = true
      validLengths[slot] = 1
      values[slot] = options.tokens[request]!
    }
    const session = options.sequences[0]!.session
    const binding = await uploadTokens(values, [state.batch, 1], options.tokenDtype, session)
    try {
      return await options.executable.value.executeSampledSteps(
        [binding.value],
        options.sequences.map((sequence) => sequence.target),
        [...options.slots],
        activeMask,
        validLengths,
        validLengths,
        options.tokens.map((token) => [token]),
        options.sampling.map((step) => step.map((sampling) => ({ ...sampling }))),
        options.token
      )
    } finally {
      binding.inUse = false
    }
  }

  const decodeOutputs = (
    roots: Runtime.CompileRequest["roots"],
    state: Runtime.DecodeStateRequest | undefined
  ): ExecutableRecord["outputs"] => {
    if (state === undefined) {
      return roots.map((root) => ({ shape: root.shape, dtype: root.dtype, storage: root.storage }))
    }
    const selections = state.outputSelections ?? roots.map(() => state.lastTokenRow ? "splitLastTokenRow" : "allRows")
    return roots.flatMap((root, index) => {
      const selection = selections[index]
      if (selection === "allRows") return [{ shape: root.shape, dtype: root.dtype }]
      const width = root.shape[root.shape.length - 1]!
      if (selection === "batchedLastTokenRow") {
        return [{ shape: [state.batch, width], dtype: root.dtype }]
      }
      return Array.from({ length: state.batch }, () => ({ shape: [width], dtype: root.dtype }))
    })
  }

  const node = (request: Runtime.NodeRequest): Effect.Effect<Runtime.LazyTensorHandle, Runtime.BackendError> =>
    Effect.try({
      try: () => {
        const inputs = request.inputs.map((input) => handleRecord(input, request.op).graph)
        switch (request.op) {
          case "constant":
            if (inputs.length > 1) throw new Error("constant: expected zero or one tensor input")
            return lazyHandle(runtime.constant(request.attributes.value, request.attributes.dtype))
          case "zeros":
            if (inputs.length > 1) throw new Error("zeros: expected zero or one tensor input")
            return lazyHandle(runtime.zeros([...request.attributes.shape], request.attributes.dtype))
          case "ones":
            if (inputs.length > 1) throw new Error("ones: expected zero or one tensor input")
            return lazyHandle(runtime.ones([...request.attributes.shape], request.attributes.dtype))
          case "full":
            if (inputs.length > 1) throw new Error("full: expected zero or one tensor input")
            return lazyHandle(
              runtime.full([...request.attributes.shape], request.attributes.value, request.attributes.dtype)
            )
          case "fromBytes":
            if (inputs.length !== 0) throw new Error("fromBytes: expected no tensor inputs")
            return lazyHandle(
              runtime.fromBytes(request.attributes.data, [...request.attributes.shape], request.attributes.dtype)
            )
          case "input": {
            const storage = request.attributes.storage
            if (storage !== undefined && !validEncodedGeometry(request.attributes.shape, storage)) {
              throw new Error("input: encoded storage does not match its logical GGML geometry")
            }
            return lazyHandle(
              runtime.graphNode(
                "input",
                inputs,
                JSON.stringify({
                  ...request.attributes,
                  shape: storage?.physicalShape ?? request.attributes.shape,
                  dtype: storage?.physicalDtype ?? request.attributes.dtype
                })
              ),
              {
                shape: request.attributes.shape,
                dtype: request.attributes.dtype,
                storage
              }
            )
          }
          default: {
            const attributes = "attributes" in request ? request.attributes : {}
            return lazyHandle(runtime.graphNode(request.op, inputs, JSON.stringify(attributes)))
          }
        }
      },
      catch: errorFor(request.op, "graph", "execution-failed")
    })

  const decode: Runtime.DecodeRuntime = {
    makePool: (options) =>
      Effect.try({
        try: () => {
          const value = new native.NativeKvPool(
            deviceOrdinal,
            options.layers,
            options.kvHeads,
            options.headDim,
            options.maxTokens,
            options.blockSize,
            options.dtype,
            {
              kdaLayers: options.kdaLayers,
              kdaHeads: options.kdaHeads,
              kdaHeadDim: options.kdaHeadDim,
              kdaValueDim: options.kdaValueDim,
              convLayers: options.convLayers,
              convChannels: options.convChannels,
              convKernel: options.convKernel
            }
          )
          const handle = opaqueHandle<Runtime.KvPoolHandle>()
          poolRecords.set(handle, { value, options: { ...options } })
          backendHandles.add(handle)
          return handle
        },
        catch: errorFor("makeKvPool", "execute", "execution-failed")
      }),
    makeSequence: (handle) =>
      Effect.try({
        try: () => {
          const pool = nativePool(handle, "makeKvSequence")
          const value = pool.value.makeSequence()
          const sequence = opaqueHandle<Runtime.KvSequenceHandle>()
          sequenceRecords.set(sequence, { value, pool, disposed: false })
          backendHandles.add(sequence)
          return sequence
        },
        catch: errorFor("makeKvSequence", "execute", "execution-failed")
      }),
    prefillMatch: (handle, tokens) =>
      Effect.try({
        try: () => nativeSequence(handle, "prefillMatch").value.prefillMatch([...tokens]),
        catch: errorFor("prefillMatch", "execute", "execution-failed")
      }),
    sequenceCursor: (handle) =>
      Effect.try({
        try: () => nativeSequence(handle, "sequenceCursor").value.cursor,
        catch: errorFor("sequenceCursor", "execute", "execution-failed")
      }),
    releaseSequence: (handle) =>
      Effect.try({
        try: () => {
          const sequence = nativeSequence(handle, "releaseSequence")
          sequence.value.release()
          sequence.disposed = true
        },
        catch: errorFor("releaseSequence", "execute", "execution-failed")
      })
  }
  const inference: Runtime.InferenceRuntime = {
    compile: (request) =>
      Effect.try({
        try: () => {
          if (request.batchSize < 1 || request.target.prefill.length === 0) {
            throw new Error("inferenceCompile: batch size and prefill programs must be nonzero")
          }
          for (
            const executable of [...request.target.prefill, request.target.decode, ...(request.target.verify ?? [])]
          ) {
            if (executableRecord(executable, "inferenceCompile").state === undefined) {
              throw new Error("inferenceCompile: target programs must be stateful")
            }
          }
          nativePool(request.target.pool, "inferenceCompile")
          if (request.proposer !== undefined) {
            executableRecord(request.proposer.prefill, "inferenceCompile")
            executableRecord(request.proposer.decode, "inferenceCompile")
            nativePool(request.proposer.pool, "inferenceCompile")
          }
          if (request.generalizedProposer !== undefined) {
            const generalized = request.generalizedProposer
            if (generalized.plan.stages[0]?.operationId === "HistoryLookup") {
              if (
                generalized.plan.stages.length !== 1 ||
                generalized.stageExecutables.length !== 0 ||
                generalized.sharedTensors.length !== 0 ||
                generalized.replay !== undefined
              ) {
                throw new Error("inferenceCompile: invalid HistoryLookup proposer")
              }
            } else {
              if (
                generalized.plan.stages.length !== 1 ||
                generalized.plan.stages[0]?.operationId !== "ParallelBlock" ||
                generalized.stageExecutables.length !== 1 ||
                generalized.sharedTensors.length !== 2 ||
                generalized.replay === undefined ||
                generalized.replay.prefill.length !== request.target.prefill.length ||
                generalized.replay.verify.length !== request.target.verify?.length ||
                request.target.verify === undefined ||
                request.target.verify.length === 0
              ) {
                throw new Error("inferenceCompile: CUDA requires one replayable ParallelBlock proposer")
              }
              for (const shared of generalized.sharedTensors) {
                handleRecord(shared, "inferenceCompile", true)
              }
              for (
                const executable of [
                  ...generalized.stageExecutables,
                  ...generalized.replay.prefill,
                  generalized.replay.decode,
                  ...generalized.replay.verify
                ]
              ) {
                if (executableRecord(executable, "inferenceCompile").state === undefined) {
                  throw new Error("inferenceCompile: proposer and replay programs must be stateful")
                }
              }
              nativePool(generalized.replay.pool, "inferenceCompile")
            }
          }
          const record: InferenceArtifactRecord = {
            request,
            diagnostics: {
              roundsStarted: 0n,
              roundsCompleted: 0n,
              ordinaryRounds: 0n,
              speculativeRounds: 0n,
              proposedTokens: 0n,
              acceptedTokens: 0n,
              emittedTokens: 0n,
              draftNanos: 0n,
              verificationNanos: 0n,
              acceptedLengthHistogram: []
            }
          }
          const handle = opaqueHandle<Runtime.InferenceArtifactHandle>()
          inferenceArtifactRecords.set(handle, record)
          backendHandles.add(handle)
          return handle
        },
        catch: errorFor("inferenceCompile", "compile", "compilation-failed")
      }),
    open: (handle) =>
      Effect.try({
        try: () => {
          const artifact = inferenceArtifact(handle, "inferenceOpen")
          const record: InferenceSessionRecord = {
            artifact,
            sequences: new Map(),
            receipts: new Set(),
            tokenBindings: [],
            nextSequenceId: 1n,
            nextRoundId: 1n,
            closed: false
          }
          const session = opaqueHandle<Runtime.InferenceSessionHandle>()
          inferenceSessionRecords.set(session, record)
          backendHandles.add(session)
          return session
        },
        catch: errorFor("inferenceOpen", "execute", "execution-failed")
      }),
    add: (handle, request) =>
      cancellable(
        native,
        "inferenceAdd",
        "execute",
        async (token) => {
          const session = inferenceSession(handle, "inferenceAdd")
          const artifact = session.artifact.request
          if (request.entries.length === 0 || session.sequences.size + request.entries.length > artifact.batchSize) {
            throw new Error(`inferenceAdd: session accepts at most ${artifact.batchSize} live sequences`)
          }
          const prompts = await Promise.all(request.entries.map((entry) => readTokens(entry.prompt, token)))
          if (prompts.some((prompt) => prompt.length === 0)) {
            throw new Error("inferenceAdd: prompts must be nonempty")
          }
          const pool = nativePool(artifact.target.pool, "inferenceAdd")
          const generalized = artifact.generalizedProposer
          const proposerPool = generalized?.replay === undefined
            ? undefined
            : nativePool(generalized.replay.pool, "inferenceAdd")
          const staged: Array<InferenceSequenceRecord> = []
          try {
            for (let index = 0; index < request.entries.length; index++) {
              const entry = request.entries[index]!
              const id = session.nextSequenceId + BigInt(index)
              const sequenceHandle = opaqueHandle<Runtime.InferenceSequenceHandle>()
              const defaults = artifact.sampling
              const sampling: Runtime.InferenceSamplingOptions = {
                temperature: entry.sampling?.temperature ?? defaults.temperature,
                topK: entry.sampling?.topK ?? defaults.topK,
                topP: entry.sampling?.topP ?? defaults.topP,
                seed: entry.sampling?.seed ?? defaults.seed
              }
              const sequence: InferenceSequenceRecord = {
                session,
                id,
                target: pool.value.makeSequence(),
                proposer: proposerPool?.value.makeSequence(),
                handle: sequenceHandle,
                eos: new Set(entry.eosTokens),
                maxTokens: entry.maxTokens,
                sampling,
                history: [...prompts[index]!],
                pending: 0,
                generated: 0,
                finished: false
              }
              staged.push(sequence)
              let offset = 0
              while (offset < prompts[index]!.length) {
                const remaining = prompts[index]!.length - offset
                const programs = artifact.target.prefill
                  .map((executable, program) => ({
                    executable: executableRecord(executable, "inferenceAdd"),
                    replay: generalized?.replay === undefined
                      ? undefined
                      : executableRecord(generalized.replay.prefill[program]!, "inferenceAdd")
                  }))
                  .sort((left, right) =>
                    left.executable.sourceShapes[0]!.at(-2)! - right.executable.sourceShapes[0]!.at(-2)!
                  )
                const selected = programs.find((program) => program.executable.sourceShapes[0]!.at(-2)! >= remaining)
                  ?? programs[programs.length - 1]!
                const executable = selected.executable
                const width = executable.sourceShapes[0]!.at(-2)!
                const length = Math.min(width, remaining)
                const row = prompts[index]!.slice(offset, offset + length)
                const final = offset + length === prompts[index]!.length
                const sampling = final
                  ? [resolvedSampling(sequence.sampling, undefined, sequence.id, 0)]
                  : undefined
                if (selected.replay === undefined) {
                  const sampled = await runStateBatch({
                    executable,
                    sequences: [sequence],
                    slots: [0],
                    tokenRows: [row],
                    tokenDtype: artifact.tokenDtype,
                    sampling,
                    token
                  })
                  if (sampled !== undefined) sequence.pending = sampled[0]!
                } else {
                  const result = await runStateBatchWithOutputs({
                    executable,
                    sequences: [sequence],
                    slots: [0],
                    tokenRows: [row],
                    tokenDtype: artifact.tokenDtype,
                    sampling,
                    token
                  })
                  try {
                    const taps = generalized!.plan.prefillHiddenTaps ?? generalized!.plan.hiddenTaps
                    await runReplay({
                      executable: selected.replay,
                      bindings: hiddenTapOutputs(result.outputs, taps, artifact.batchSize, true),
                      sequences: [sequence.proposer!],
                      slots: [0],
                      lengths: [length],
                      tokens: [row],
                      token
                    })
                    if (result.sampled !== undefined) sequence.pending = result.sampled[0]!
                  } finally {
                    for (const output of result.outputs) output.clear()
                  }
                }
                offset += length
              }
              sequence.generated = 1
              sequence.history.push(sequence.pending)
              if (sequence.eos.has(sequence.pending)) sequence.terminal = "eos"
              else if (sequence.maxTokens !== undefined && sequence.generated >= sequence.maxTokens) {
                sequence.terminal = "maxTokens"
              }
            }
          } catch (cause) {
            for (const sequence of staged) {
              sequence.target.release()
              sequence.proposer?.release()
            }
            throw cause
          }
          session.nextSequenceId += BigInt(staged.length)
          for (const sequence of staged) {
            session.sequences.set(sequence.id, sequence)
            inferenceSequenceRecords.set(sequence.handle, sequence)
            backendHandles.add(sequence.handle)
          }
          const roundId = session.nextRoundId++
          session.receipts.add(roundId)
          const pages = staged.map((sequence) => ({
            sequence: sequence.handle,
            sequenceId: sequence.id,
            tokens: [sequence.pending],
            terminal: sequence.terminal,
            stopReason: sequence.terminal
          }))
          const diagnostics = session.artifact.diagnostics
          diagnostics.roundsStarted++
          diagnostics.roundsCompleted++
          diagnostics.ordinaryRounds++
          diagnostics.emittedTokens += BigInt(staged.length)
          diagnostics.lastRoundId = roundId
          return { roundId, recovered: false, pages }
        }
      ),
    runRound: (handle, request) =>
      cancellable(
        native,
        "inferenceRound",
        "execute",
        async (token) => {
          const session = inferenceSession(handle, "inferenceRound")
          if (request.entries.length === 0 || request.entries.length > session.artifact.request.batchSize) {
            throw new Error("inferenceRound: invalid active sequence count")
          }
          const entries = request.entries.map((entry) => ({
            sequence: inferenceSequence(session, entry.sequence, "inferenceRound"),
            sampling: entry.sampling
          }))
          if (new Set(entries.map((entry) => entry.sequence)).size !== entries.length) {
            throw new Error("inferenceRound: sequences must be distinct")
          }
          const artifact = session.artifact.request
          const draftStarted = performance.now()
          const speculative = artifact.proposer !== undefined || artifact.generalizedProposer !== undefined
          const draftLimit = artifact.proposer?.maxDraftTokens
            ?? artifact.generalizedProposer?.maxDraftTokens
            ?? 0
          const decode = executableRecord(artifact.target.decode, "inferenceRound")
          const historyLookup = artifact.generalizedProposer?.plan.stages
            .find((stage) => stage.operationId === "HistoryLookup")
            ?.historyLookup
          const historyDrafts = new Map<InferenceSequenceRecord, ReadonlyArray<number>>()
          const limits = new Map(entries.map(({ sequence }) => {
            let limit = speculative ? draftLimit + 1 : 1
            if (historyLookup !== undefined) {
              const draft = historyDraft(
                sequence.history,
                historyLookup.minMatchTokens,
                historyLookup.maxMatchTokens,
                draftLimit
              )
              historyDrafts.set(sequence, draft)
              limit = draft.length + 1
            }
            if (sequence.maxTokens !== undefined) {
              limit = Math.min(limit, Math.max(0, sequence.maxTokens - sequence.generated))
            }
            if (decode.state?.window === undefined) {
              limit = Math.min(limit, Math.max(0, decode.state!.maxTokens - sequence.target.cursor))
            }
            return [sequence, limit] as const
          }))
          if (entries.some(({ sequence }) => sequence.terminal === undefined && limits.get(sequence) === 0)) {
            throw new Error("inferenceRound: sequence context exceeds pool capacity")
          }
          const pages = new Map<InferenceSequenceRecord, Array<number>>()
          const diagnostics = session.artifact.diagnostics
          diagnostics.roundsStarted++
          let draftNanos = 0n
          let verificationNanos = 0n
          let acceptedCandidates: number | undefined
          const roundSteps = Math.max(...limits.values())
          const append = (sequence: InferenceSequenceRecord, sampled: number) => {
            sequence.pending = sampled
            sequence.generated++
            sequence.history.push(sampled)
            const page = pages.get(sequence) ?? []
            page.push(sampled)
            pages.set(sequence, page)
            if (sequence.eos.has(sampled)) sequence.terminal = "eos"
            else if (sequence.maxTokens !== undefined && sequence.generated >= sequence.maxTokens) {
              sequence.terminal = "maxTokens"
            }
          }
          const verifierHandle = artifact.generalizedProposer === undefined
            ? undefined
            : artifact.target.verify?.at(-1)
          const verifier = verifierHandle === undefined
            ? undefined
            : executableRecord(verifierHandle, "inferenceRound")
          const verifyRows = verifier?.state === undefined
            ? 0
            : verifier.sourceShapes[0]![0]! / verifier.state.batch
          const targetMatching = roundSteps > 1 && verifier !== undefined && verifyRows >= roundSteps &&
            entries.every(({ sequence }) => sequence.terminal === undefined && limits.get(sequence)! > 0)
          const parallel = artifact.generalizedProposer?.plan.stages[0]?.operationId === "ParallelBlock"
            ? artifact.generalizedProposer
            : undefined
          const batchable = roundSteps > 1 && entries.every(({ sequence }) =>
            sequence.terminal === undefined && sequence.eos.size === 0 && limits.get(sequence) === roundSteps
          )
          if (speculative && parallel === undefined) {
            draftNanos = BigInt(Math.max(1, Math.round((performance.now() - draftStarted) * 1_000_000)))
          }
          if (targetMatching && parallel !== undefined) {
            const draftStarted = performance.now()
            const stage = executableRecord(parallel.stageExecutables[0]!, "inferenceRound")
            const replay = executableRecord(parallel.replay!.verify.at(-1)!, "inferenceRound")
            const slots = entries.map((_, index) => index)
            const draftSequences = entries.map(({ sequence }) => {
              if (sequence.proposer === undefined) {
                throw new Error("inferenceRound: proposer replay sequence is missing")
              }
              return sequence.proposer.fork()
            })
            const anchor = await uploadTokens(
              Array.from({ length: artifact.batchSize }, (_, slot) => entries[slot]?.sequence.pending ?? 0),
              [artifact.batchSize],
              artifact.tokenDtype,
              session
            )
            let stageOutputs: ReadonlyArray<NativeTensor> = []
            try {
              const activeMask = new Array<boolean>(artifact.batchSize).fill(false)
              const validLengths = new Array<number>(artifact.batchSize).fill(0)
              for (const slot of slots) {
                activeMask[slot] = true
                validLengths[slot] = parallel.plan.trainedMaxRows + 1
              }
              stageOutputs = await stage.value.executeStateful(
                [
                  anchor.value,
                  ...parallel.sharedTensors.map((shared) => handleRecord(shared, "inferenceRound", true).value!)
                ],
                draftSequences,
                slots,
                activeMask,
                validLengths,
                validLengths,
                entries.map(({ sequence }) =>
                  new Array<number>(parallel.plan.trainedMaxRows + 1).fill(sequence.pending)
                ),
                token
              )
            } finally {
              anchor.inUse = false
              for (const sequence of draftSequences) sequence.release()
            }
            draftNanos = BigInt(Math.max(1, Math.round((performance.now() - draftStarted) * 1_000_000)))
            try {
              const probabilities = stageOutputs[1]
              const proposalTokens = probabilities === undefined
                ? await (async () => {
                  const candidates = stageOutputs[0]
                  if (candidates === undefined || candidates.dtype !== "u32") {
                    throw new Error("inferenceRound: ParallelBlock token output is missing")
                  }
                  const bytes = await candidates.readback(token)
                  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
                  return entries.map(({ sequence }, request) => [
                    sequence.pending,
                    ...Array.from(
                      { length: limits.get(sequence)! - 1 },
                      (_, step) =>
                        view.getUint32(
                          (request * parallel.plan.trainedMaxRows + step) * 4,
                          true
                        )
                    )
                  ])
                })()
                : entries.map(({ sequence }) => [sequence.pending])
              const verificationStarted = performance.now()
              const matched = await verifier.value.executeTargetMatching(
                entries.map(({ sequence }) => sequence.target),
                slots,
                proposalTokens,
                Array.from({ length: verifyRows }, (_, step) =>
                  entries.map(({ sequence, sampling }) =>
                    resolvedSampling(sequence.sampling, sampling, sequence.id, sequence.generated + step)
                  )),
                entries.map(({ sequence }) =>
                  limits.get(sequence)!
                ),
                entries.map(({ sequence }) => [...sequence.eos]),
                probabilities,
                token
              )
              try {
                const taps = parallel.plan.verifyHiddenTaps ?? parallel.plan.hiddenTaps
                await runReplay({
                  executable: replay,
                  bindings: hiddenTapOutputs(matched.outputs, taps, artifact.batchSize, false),
                  sequences: entries.map(({ sequence }) => sequence.proposer!),
                  slots,
                  lengths: matched.pages.map((page) => page.length),
                  tokens: matched.pages.map((page, request) => [
                    entries[request]!.sequence.pending,
                    ...page.slice(0, -1)
                  ]),
                  token
                })
                acceptedCandidates = matched.accepted.reduce((total, count) => total + count, 0)
                for (let request = 0; request < entries.length; request++) {
                  const sequence = entries[request]!.sequence
                  for (const value of matched.pages[request]!) append(sequence, value)
                }
                verificationNanos = BigInt(
                  Math.max(1, Math.round((performance.now() - verificationStarted) * 1_000_000))
                )
              } finally {
                for (const output of matched.outputs) output.clear()
              }
            } finally {
              for (const output of stageOutputs) output.clear()
            }
          } else if (targetMatching) {
            const verificationStarted = performance.now()
            const matched = await verifier.value.executeTargetMatching(
              entries.map(({ sequence }) => sequence.target),
              entries.map((_, index) => index),
              entries.map(({ sequence }) => [
                sequence.pending,
                ...(historyDrafts.get(sequence) ?? new Array<number>(limits.get(sequence)! - 1).fill(0)).slice(
                  0,
                  limits.get(sequence)! - 1
                )
              ]),
              Array.from({ length: verifyRows }, (_, step) =>
                entries.map(({ sequence, sampling }) =>
                  resolvedSampling(sequence.sampling, sampling, sequence.id, sequence.generated + step)
                )),
              entries.map(({ sequence }) =>
                limits.get(sequence)!
              ),
              entries.map(({ sequence }) => [...sequence.eos]),
              undefined,
              token
            )
            try {
              acceptedCandidates = matched.accepted.reduce((total, count) => total + count, 0)
              for (let request = 0; request < entries.length; request++) {
                const sequence = entries[request]!.sequence
                for (const value of matched.pages[request]!) {
                  append(sequence, value)
                }
              }
            } finally {
              for (const output of matched.outputs) output.clear()
            }
            verificationNanos = BigInt(
              Math.max(1, Math.round((performance.now() - verificationStarted) * 1_000_000))
            )
          } else if (batchable) {
            const verificationStarted = performance.now()
            const sampled = await runSampledSteps({
              executable: decode,
              sequences: entries.map(({ sequence }) => sequence),
              slots: entries.map((_, index) => index),
              tokens: entries.map(({ sequence }) => sequence.pending),
              tokenDtype: artifact.tokenDtype,
              sampling: Array.from({ length: roundSteps }, (_, step) =>
                entries.map(({ sequence, sampling }) =>
                  resolvedSampling(sequence.sampling, sampling, sequence.id, sequence.generated + step)
                )),
              token
            })
            for (let request = 0; request < entries.length; request++) {
              const sequence = entries[request]!.sequence
              for (const token of sampled[request]!) {
                append(sequence, token)
              }
            }
            if (speculative) {
              verificationNanos = BigInt(
                Math.max(1, Math.round((performance.now() - verificationStarted) * 1_000_000))
              )
            }
          } else {
            const verificationStarted = performance.now()
            for (let step = 0; step < roundSteps; step++) {
              const active = entries.filter(({ sequence }) =>
                sequence.terminal === undefined && (pages.get(sequence)?.length ?? 0) < limits.get(sequence)!
              )
              if (active.length === 0) {
                break
              }
              const sampled = await runStateBatch({
                executable: decode,
                sequences: active.map(({ sequence }) =>
                  sequence
                ),
                slots: active.map((_, index) =>
                  index
                ),
                tokenRows: active.map(({ sequence }) => [sequence.pending]),
                tokenDtype: artifact.tokenDtype,
                sampling: active.map(({ sequence, sampling }) =>
                  resolvedSampling(sequence.sampling, sampling, sequence.id, sequence.generated)
                ),
                token
              })
              if (sampled === undefined) throw new Error("inferenceRound: decode did not sample outputs")
              for (let index = 0; index < active.length; index++) {
                append(active[index]!.sequence, sampled[index]!)
              }
            }
            if (speculative) {
              verificationNanos = BigInt(
                Math.max(1, Math.round((performance.now() - verificationStarted) * 1_000_000))
              )
            }
          }
          const roundId = session.nextRoundId++
          session.receipts.add(roundId)
          const resultPages = entries.flatMap(({ sequence }) => {
            const tokens = pages.get(sequence)
            return tokens === undefined
              ? []
              : [{
                sequence: sequence.handle,
                sequenceId: sequence.id,
                tokens,
                stopReason: sequence.terminal
              }]
          })
          diagnostics.roundsCompleted++
          diagnostics.emittedTokens += BigInt(resultPages.reduce((total, page) => total + page.tokens.length, 0))
          if (speculative) {
            diagnostics.speculativeRounds++
            const proposed = entries.reduce(
              (total, { sequence }) => total + Math.max(0, limits.get(sequence)! - 1),
              0
            )
            const accepted = acceptedCandidates ??
              resultPages.reduce((total, page) => total + Math.max(0, page.tokens.length - 1), 0)
            diagnostics.proposedTokens += BigInt(proposed)
            diagnostics.acceptedTokens += BigInt(accepted)
            diagnostics.draftNanos += draftNanos
            diagnostics.verificationNanos += verificationNanos
            for (const page of resultPages) {
              const accepted = Math.max(0, page.tokens.length - 1)
              diagnostics.acceptedLengthHistogram[accepted] = (diagnostics.acceptedLengthHistogram[accepted] ?? 0n) + 1n
            }
          } else diagnostics.ordinaryRounds++
          diagnostics.lastRoundId = roundId
          return { roundId, recovered: false, pages: resultPages }
        }
      ),
    acknowledge: (handle, roundId) =>
      Effect.try({
        try: () => {
          const session = inferenceSession(handle, "inferenceAcknowledge")
          if (!session.receipts.delete(roundId)) throw new Error("inferenceAcknowledge: unknown round")
        },
        catch: errorFor("inferenceAcknowledge", "execute", "execution-failed")
      }),
    finish: (handle, handles) =>
      Effect.try({
        try: () => {
          const session = inferenceSession(handle, "inferenceFinish")
          const sequences = handles.map((sequence) => inferenceSequence(session, sequence, "inferenceFinish"))
          if (new Set(sequences).size !== sequences.length) throw new Error("inferenceFinish: duplicate sequence")
          for (const sequence of sequences) {
            sequence.target.release()
            sequence.proposer?.release()
            sequence.finished = true
            session.sequences.delete(sequence.id)
          }
        },
        catch: errorFor("inferenceFinish", "execute", "execution-failed")
      }),
    inspect: (handle, sequenceHandle) =>
      Effect.try({
        try: () => {
          const session = inferenceSession(handle, "inferenceInspect")
          const sequence = inferenceSequence(session, sequenceHandle, "inferenceInspect")
          return {
            sequenceId: sequence.id,
            cursor: BigInt(sequence.target.cursor),
            terminal: sequence.terminal
          }
        },
        catch: errorFor("inferenceInspect", "execute", "execution-failed")
      }),
    close: (handle) =>
      Effect.try({
        try: () => {
          const session = inferenceSession(handle, "inferenceClose")
          for (const sequence of session.sequences.values()) {
            sequence.target.release()
            sequence.proposer?.release()
            sequence.finished = true
          }
          session.sequences.clear()
          for (const binding of session.tokenBindings) binding.value.clear()
          session.tokenBindings.length = 0
          session.closed = true
        },
        catch: errorFor("inferenceClose", "shutdown", "execution-failed")
      }),
    diagnostics: (handle) =>
      Effect.try({
        try: () => {
          const diagnostics = inferenceArtifact(handle, "inferenceDiagnostics").diagnostics
          return Object.freeze({
            roundsStarted: diagnostics.roundsStarted,
            roundsCompleted: diagnostics.roundsCompleted,
            roundsRecovered: 0n,
            ordinaryRounds: diagnostics.ordinaryRounds,
            speculativeRounds: diagnostics.speculativeRounds,
            proposedTokens: diagnostics.proposedTokens,
            acceptedTokens: diagnostics.acceptedTokens,
            emittedTokens: diagnostics.emittedTokens,
            provisionalBlocks: 0n,
            rolledBackBlocks: 0n,
            draftNanos: diagnostics.draftNanos,
            verificationNanos: diagnostics.verificationNanos,
            acceptedLengthHistogram: Object.freeze([...diagnostics.acceptedLengthHistogram]),
            targetPoolHighWaterBlocks: diagnostics.roundsStarted === 0n ? 0n : 1n,
            proposerPoolHighWaterBlocks: diagnostics.speculativeRounds === 0n ? 0n : 1n,
            lastRoundId: diagnostics.lastRoundId
          })
        },
        catch: errorFor("inferenceDiagnostics", "execute", "execution-failed")
      })
  }
  const sampling: Runtime.SamplingRuntime = {
    sample: (handle, options) =>
      cancellable(
        native,
        "sample",
        "execute",
        (token) =>
          handleRecord(handle, "sample", true).value!.sample(
            options.temperature,
            options.topK,
            options.topP,
            options.seed,
            options.counter,
            token
          )
      ),
    executeDecode: (handle, invocation, options) =>
      cancellable(
        native,
        "executeDecode",
        "execute",
        (token) => {
          const executable = executableRecord(handle, "executeDecode")
          const state = invocation.state
          if (executable.state === undefined || state === undefined) {
            throw new Error("executeDecode: requires a stateful executable and invocation")
          }
          if (invocation.scalars.length > 0 || Object.keys(invocation.runtimeValues).length > 0) {
            throw new Error("executeDecode: stateful CUDA programs do not accept scalar or runtime values")
          }
          return executable.value.executeSampled(
            invocation.bindings.map((binding) => handleRecord(binding, "executeDecode", true).value!),
            state.sequences.map((sequence) => nativeSequence(sequence, "executeDecode").value),
            [...state.slots],
            [...state.activeMask],
            [...state.validLengths],
            [...state.advances],
            state.tokens.map((row) => [...row]),
            options.map((option) => ({ ...option })),
            token
          )
        }
      ),
    executeSpeculative: () => unsupported("executeSpeculative")
  }
  const metadataValue = (entry: NativeGgufMetadataEntry): Runtime.GgufMetadataEntry => {
    if (!Predicate.isString(entry.key) || entry.key.length === 0 || !Predicate.isString(entry.kind)) {
      throw new Error("native GGUF metadata entry is invalid")
    }
    const numericKinds = ["u8", "i8", "u16", "i16", "u32", "i32", "f32", "u64", "i64", "f64"]
    const candidates: Array<Runtime.GgufMetadataScalar | ReadonlyArray<Runtime.GgufMetadataScalar>> = []
    if (entry.numberValue !== undefined) {
      if (!numericKinds.includes(entry.kind) || !Predicate.isNumber(entry.numberValue)) {
        throw new Error("invalid GGUF number metadata")
      }
      candidates.push(entry.numberValue)
    }
    if (entry.stringValue !== undefined) {
      if (entry.kind !== "string" || !Predicate.isString(entry.stringValue)) {
        throw new Error("invalid GGUF string metadata")
      }
      candidates.push(entry.stringValue)
    }
    if (entry.booleanValue !== undefined) {
      if (entry.kind !== "bool" || !Predicate.isBoolean(entry.booleanValue)) {
        throw new Error("invalid GGUF boolean metadata")
      }
      candidates.push(entry.booleanValue)
    }
    if (entry.numberArray !== undefined) {
      if (!numericKinds.includes(entry.kind) || !entry.numberArray.every(Predicate.isNumber)) {
        throw new Error("invalid GGUF number array metadata")
      }
      candidates.push(Object.freeze([...entry.numberArray]))
    }
    if (entry.stringArray !== undefined) {
      if (entry.kind !== "string" || !entry.stringArray.every(Predicate.isString)) {
        throw new Error("invalid GGUF string array metadata")
      }
      candidates.push(Object.freeze([...entry.stringArray]))
    }
    if (entry.booleanArray !== undefined) {
      if (entry.kind !== "bool" || !entry.booleanArray.every(Predicate.isBoolean)) {
        throw new Error("invalid GGUF boolean array metadata")
      }
      candidates.push(Object.freeze([...entry.booleanArray]))
    }
    if (candidates.length !== 1) {
      throw new Error(`native GGUF metadata ${entry.key} has invalid value fields`)
    }
    return Object.freeze({ key: entry.key, value: candidates[0]! })
  }
  const ggufDescriptor = (value: NativeGgufTensorDescriptor): Runtime.GgufTensorDescriptor => {
    const format = value.format
    const encoded = format !== "F32"
    if (
      !Predicate.isString(value.name) || value.name.length === 0 || !isGgufFormat(format) ||
      value.logicalDtype !== "f32" || value.physicalDtype !== (encoded ? "u8" : "f32") ||
      !validShape(value.logicalShape) || !validShape(value.physicalShape) ||
      (encoded && !validEncodedGeometry(value.logicalShape, {
        encoding: format,
        physicalShape: value.physicalShape,
        physicalDtype: "u8"
      }))
    ) {
      throw new Error("native GGUF tensor descriptor is invalid")
    }
    return Object.freeze({
      name: value.name,
      format,
      logicalShape: Object.freeze([...value.logicalShape]),
      logicalDtype: "f32",
      physicalShape: Object.freeze([...value.physicalShape]),
      physicalDtype: value.physicalDtype
    })
  }
  const gguf: Runtime.GgufRuntime = {
    inspect: (path) =>
      cancellable(native, "inspectGguf", "io", (token) => native.inspectGguf(path, token)).pipe(
        Effect.flatMap((inspection) =>
          Effect.try({
            try: () =>
              Object.freeze({
                metadata: Object.freeze(inspection.metadata.map(metadataValue)),
                tensors: Object.freeze(inspection.tensors.map(ggufDescriptor))
              }),
            catch: errorFor("inspectGguf", "io", "io-failed")
          })
        )
      ),
    load: (path) =>
      cancellable(
        native,
        "loadGguf",
        "io",
        (token) => native.loadGgufForDevice(path, deviceOrdinal, token),
        (archive) => {
          for (const entry of archive.entries) entry.tensor.clear()
        }
      ).pipe(
        Effect.flatMap((archive) =>
          Effect.try({
            try: () => {
              const values = archive.entries.map((entry) => entry.tensor)
              try {
                return Object.freeze({
                  entries: Object.freeze(archive.entries.map((entry) => {
                    const descriptor = ggufDescriptor(entry.descriptor)
                    const storage: Runtime.EncodedTensorStorage | undefined = descriptor.format === "F32"
                      ? undefined
                      : {
                        encoding: descriptor.format,
                        physicalShape: descriptor.physicalShape,
                        physicalDtype: "u8"
                      }
                    return Object.freeze({
                      descriptor,
                      tensor: concreteHandle(entry.tensor, {
                        shape: descriptor.logicalShape,
                        dtype: "f32",
                        storage
                      })
                    })
                  }))
                })
              } catch (error) {
                for (const value of values) value.clear()
                throw error
              }
            },
            catch: errorFor("loadGguf", "io", "io-failed")
          })
        )
      )
  }
  const pathSafetensors: Runtime.PathSafetensors = {
    save: (path, archive) =>
      archive.entries.some((entry) => entry.tensor.storage !== undefined)
        ? Effect.fail(
          new Runtime.BackendError({
            reason: "unsupported-operation",
            backend: backendName,
            operation: "save",
            phase: "io",
            message: "save: encoded tensors cannot be represented by safetensors"
          })
        )
        : cancellable(
          native,
          "save",
          "io",
          (token) =>
            native.saveTensors(
              path,
              archive.entries.map((entry) => entry.name),
              archive.entries.map((entry) => handleRecord(entry.tensor, "save", true).value!),
              { ...archive.metadata },
              token
            )
        ),
    load: (path) =>
      cancellable(
        native,
        "load",
        "io",
        (token) => native.loadTensors(path, deviceOrdinal, token),
        (archive) => {
          for (const entry of archive.entries) entry.tensor.clear()
        }
      ).pipe(
        Effect.flatMap((archive) =>
          Effect.try({
            try: () => ({
              entries: archive.entries.map((entry) => ({
                name: entry.name,
                tensor: concreteHandle(entry.tensor)
              })),
              metadata: Object.freeze({ ...archive.metadata })
            }),
            catch: errorFor("load", "io", "io-failed")
          })
        )
      )
  }
  const extensions: Runtime.RuntimeService["extensions"] = {
    decode,
    inference,
    sampling,
    gguf,
    pathSafetensors,
    diagnostics: { externalMemoryBytes: Effect.succeed(0) }
  }

  return {
    identity: owner,
    backend: Object.freeze({ name: backendName }),
    placement,
    capabilities,
    node,
    exposures: (root) =>
      Effect.try({
        try: () =>
          handleRecord(root, "exposures").graph.exposures().map((entry) => ({
            name: entry.name,
            tensor: lazyHandle(entry.tensor)
          })),
        catch: errorFor("exposures", "graph", "execution-failed")
      }),
    grad: (loss, wrt) =>
      Effect.try({
        try: () =>
          native.grad(
            handleRecord(loss, "grad").graph,
            wrt.map((target) => handleRecord(target, "grad").graph)
          ).map((graph) => lazyHandle(graph)),
        catch: errorFor("grad", "autodiff", "execution-failed")
      }),
    compile: (request) =>
      Effect.try({
        try: () => {
          if (request.roots.length === 0) throw new Error("compile: expected at least one root")
          const roots = request.roots.map((root) => handleRecord(root, "compile").graph)
          const value = runtime.compile(
            roots,
            request.options === undefined
              ? undefined
              : {
                optimize: request.options.optimize,
                constantWeights: request.options.constantWeights
              },
            request.state === undefined ? undefined : nativeState(request.state)
          )
          const state: Runtime.DecodeStateSchema | undefined = request.state === undefined
            ? undefined
            : Object.freeze({
              ...request.state,
              window: value.allowsWindowEviction ? request.state.window : undefined,
              layers: value.layers,
              kvHeads: value.kvHeads,
              headDim: value.headDim,
              kdaLayers: value.kdaLayers,
              kdaHeads: value.kdaHeads,
              kdaHeadDim: value.kdaHeadDim,
              kdaValueDim: value.kdaValueDim,
              convLayers: value.convLayers,
              convChannels: value.convChannels,
              convKernel: value.convKernel
            })
          const outputs = decodeOutputs(request.roots, state)
          const nativeDiagnostics = value.diagnostics
          const diagnostics: Runtime.ExecutableDiagnostics = Object.freeze({
            ...nativeDiagnostics,
            instructions: Object.freeze(
              nativeDiagnostics.instructions.map((instruction) => Object.freeze(instruction))
            ),
            memory: Object.freeze(nativeDiagnostics.memory),
            compilePhases: Object.freeze(
              nativeDiagnostics.compilePhases.map((phase) => Object.freeze(phase))
            )
          })
          // SAFETY: executable handles are opaque capabilities validated through records.
          const handle = Object.freeze(
            state === undefined ? { diagnostics } : { diagnostics, state }
          ) as Runtime.ExecutableHandle
          records.set(handle, {
            owner,
            kind: "executable",
            value,
            outputs,
            state,
            sourceShapes: request.roots.map((root) => root.shape)
          })
          backendHandles.add(handle)
          return handle
        },
        catch: errorFor("compile", "compile", "compilation-failed")
      }),
    execute: (handle, invocation) => {
      const lateClear = (values: ReadonlyArray<NativeTensor>) => {
        for (const value of new Set(values)) value.clear()
      }
      return cancellable(
        native,
        "execute",
        "execute",
        (token) => {
          const executable = executableRecord(handle, "execute")
          if (Object.keys(invocation.runtimeValues).length !== 0) {
            throw new Error("execute: CUDA runtime values are not supported yet")
          }
          const bindings = invocation.bindings.map((binding) => {
            const found = handleRecord(binding, "execute", true)
            return found.value!
          })
          if (invocation.state === undefined) {
            if (executable.state !== undefined) throw new Error("execute: stateful executable requires state")
            return executable.value.execute(bindings, [...invocation.scalars], token)
          }
          if (executable.state === undefined) throw new Error("execute: stateless executable does not accept state")
          if (invocation.scalars.length > 0) throw new Error("execute: stateful executable does not accept scalars")
          const state = invocation.state
          return executable.value.executeStateful(
            bindings,
            state.sequences.map((sequence) => nativeSequence(sequence, "execute").value),
            [...state.slots],
            [...state.activeMask],
            [...state.validLengths],
            [...state.advances],
            state.tokens.map((row) => [...row]),
            token
          )
        },
        lateClear
      ).pipe(
        Effect.flatMap((values) =>
          Effect.try({
            try: () => {
              const executable = executableRecord(handle, "execute")
              try {
                if (values.length !== executable.outputs.length) {
                  throw new Error("execute: native CUDA output count disagrees with the executable")
                }
                if (new Set(values).size !== values.length) {
                  throw new Error("execute: native CUDA runtime returned duplicate tensor ownership")
                }
                return values.map((value, index) => {
                  const output = executable.outputs[index]!
                  const physicalShape = output.storage?.physicalShape ?? output.shape
                  const physicalDtype = output.storage?.physicalDtype ?? output.dtype
                  if (
                    dtype(value.dtype) !== physicalDtype || !validShape(value.shape) ||
                    !sameShape(value.shape, physicalShape)
                  ) {
                    throw new Error(`execute: native CUDA output ${index} has invalid metadata`)
                  }
                  return concreteHandle(value, output)
                })
              } catch (cause) {
                lateClear(values)
                throw cause
              }
            },
            catch: errorFor("execute", "execute", "execution-failed")
          })
        )
      )
    },
    readback: (handle) =>
      cancellable(
        native,
        "readback",
        "readback",
        (token) => handleRecord(handle, "readback", true).value!.readback(token)
      ).pipe(
        Effect.map((buffer) => {
          const bytes = new Uint8Array(buffer.byteLength)
          bytes.set(buffer)
          return bytes.buffer
        }),
        Effect.mapError((error) =>
          error.reason === "execution-failed"
            ? new Runtime.BackendError({ ...error, reason: "transfer-failed" })
            : error
        )
      ),
    release: (handle) =>
      Effect.try({
        try: () => {
          const found = records.get(handle)
          if (found?.owner === owner && found.kind === "concrete" && found.disposed) return
          const concrete = handleRecord(handle, "release", true)
          concrete.value!.clear()
          concrete.disposed = true
        },
        catch: errorFor("release", "shutdown", "invalid-handle")
      }),
    extensions
  }
}

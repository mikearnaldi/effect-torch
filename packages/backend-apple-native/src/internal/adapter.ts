/*
 * Adapts the backend-neutral RuntimeService contract to the private Apple Metal
 * napi-rs addon. Public handles stay opaque. Before unwrapping a native object,
 * the adapter checks which addon owns its handle. It also records logical input
 * and output contracts that the addon does not expose, and converts native
 * exceptions to Runtime.BackendError values. Graph construction, autodiff, and
 * compilation are synchronous. Execution, file I/O, and readback use
 * cancellable native promises.
 */
import { Runtime } from "@effect-torch/core"
import { Effect, Predicate } from "effect"
import { pipeArguments } from "effect/Pipeable"
import type {
  Executable,
  LazyTensor,
  NativeAddon,
  NativeCompileOptions,
  NativeCurrentBlockAttention,
  NativeDecodeOutputSelection,
  NativeDType,
  NativeGgufMetadataEntry,
  NativeGgufTensorDescriptor,
  NativeInferenceArtifact,
  NativeInferenceRoundResult,
  NativeInferenceSamplingOptions,
  NativeInferenceSamplingOverride,
  NativeInferenceSequence,
  NativeInferenceSession,
  NativeKvPool,
  NativeKvSequence,
  NativeKvStateSchema,
  NativeProposerPlan,
  NativeTensor,
  NativeValueRef
} from "./native-addon.js"

type CancellationToken = InstanceType<NativeAddon["CancellationToken"]>
type HandleKind =
  | "lazy-tensor"
  | "concrete-tensor"
  | "executable"
  | "kv-pool"
  | "kv-sequence"
  | "inference-artifact"
  | "inference-session"
  | "inference-sequence"

interface StructuralNode {
  readonly op: string
  readonly inputs: ReadonlyArray<Runtime.TensorHandle>
  readonly attributes: unknown
}

interface TensorBindingDeclaration {
  readonly kind: "tensor"
  readonly slot: number
  readonly shape: ReadonlyArray<number>
  readonly dtype: Runtime.DType
  readonly storage?: Runtime.EncodedTensorStorage | undefined
}

interface ScalarBindingDeclaration {
  readonly kind: "scalar"
  readonly slot: number
  readonly dtype: Runtime.DType
}

type InputDeclaration = TensorBindingDeclaration | ScalarBindingDeclaration
type TensorBinding = Omit<TensorBindingDeclaration, "kind" | "slot">

const numberBits = new DataView(new ArrayBuffer(8))

/**
 * Returns the JSON-safe canonical form used for structural executable cache
 * keys. It sorts object keys and converts byte arrays to numeric arrays. It
 * stores non-finite numbers and negative zero as exact IEEE-754 bits so that
 * `JSON.stringify` does not collapse them.
 *
 * @internal
 */
// oxlint-disable anti-slop/no-known-value-widening, anti-slop/no-unknown-parameters, anti-slop/no-unknown-returns -- The cache normalizer intentionally accepts and preserves arbitrary recursive values.
export const normalizedStructure = (value: unknown): unknown => {
  // oxlint-disable-next-line anti-slop/no-runtime-typeof -- Recursive classification requires the intrinsic number category.
  if (typeof value === "number") {
    if (Number.isFinite(value) && !Object.is(value, -0)) return value
    numberBits.setFloat64(0, value, false)
    const high = numberBits.getUint32(0, false).toString(16).padStart(8, "0")
    const low = numberBits.getUint32(4, false).toString(16).padStart(8, "0")
    return { $number: `${high}${low}` }
  }
  if (value instanceof Uint8Array) return Array.from(value)
  if (Array.isArray(value)) return value.map(normalizedStructure)
  // oxlint-disable-next-line anti-slop/no-runtime-typeof -- Recursive classification requires the intrinsic object category.
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, entry]) => [key, normalizedStructure(entry)])
    )
  }
  return value
}
// oxlint-enable anti-slop/no-known-value-widening, anti-slop/no-unknown-parameters, anti-slop/no-unknown-returns

/** Canonicalizes a value with {@link normalizedStructure}, then serializes it. @internal */
// oxlint-disable-next-line anti-slop/no-unknown-parameters -- The cache-key boundary accepts any value handled by the normalizer.
export const structuralCacheKey = (value: unknown): string => JSON.stringify(normalizedStructure(value))

interface KvPoolInfo {
  readonly key: object
  readonly layers: number
  readonly kvHeads: number
  readonly headDim: number
  readonly maxTokens: number
  readonly blockSize: number
  readonly dtype: Runtime.DType
  readonly kdaLayers: number
  readonly kdaHeads: number
  readonly kdaHeadDim: number
  readonly kdaValueDim: number
  readonly convLayers: number
  readonly convChannels: number
  readonly convKernel: number
}

interface KvSequenceInfo {
  readonly pool: KvPoolInfo
}

interface ExecutableInfo {
  readonly bindings: ReadonlyArray<TensorBinding>
  readonly outputs: ReadonlyArray<{
    readonly shape: ReadonlyArray<number>
    readonly dtype: Runtime.DType
    readonly storage?: Runtime.EncodedTensorStorage | undefined
  }>
  readonly state?: Runtime.DecodeStateSchema | undefined
}

interface InferenceArtifactInfo {
  readonly sampling: Runtime.InferenceSamplingOptions
}

interface InferenceSessionInfo {
  readonly artifact: object
  readonly sampling: Runtime.InferenceSamplingOptions
  readonly sequences: Map<bigint, Runtime.InferenceSequenceHandle>
}

interface InferenceSequenceInfo {
  readonly session: object
  readonly sequenceId: bigint
  readonly sampling: Runtime.InferenceSamplingOptions
}

interface InferenceAddResolution {
  readonly nativeSession: NativeInferenceSession
  readonly prompts: Array<NativeTensor>
  readonly sampling: Array<Runtime.InferenceSamplingOptions>
}

interface InferenceRoundResolution {
  readonly nativeSession: NativeInferenceSession
  readonly sequences: Array<NativeInferenceSequence>
  readonly sampling: Array<Runtime.InferenceSamplingOptions>
  readonly ids: Array<bigint>
}

interface HandleData {
  readonly "lazy-tensor": {
    readonly graph: LazyTensor
    readonly structure?: StructuralNode | undefined
    readonly declarations?: ReadonlySet<InputDeclaration> | undefined
  }
  readonly "concrete-tensor": {
    readonly graph: LazyTensor
    readonly value: NativeTensor
    readonly structure?: StructuralNode | undefined
    readonly declarations?: ReadonlySet<InputDeclaration> | undefined
  }
  readonly executable: { readonly value: Executable; readonly info: ExecutableInfo }
  readonly "kv-pool": { readonly value: NativeKvPool; readonly info: KvPoolInfo }
  readonly "kv-sequence": { readonly value: NativeKvSequence; readonly info: KvSequenceInfo }
  readonly "inference-artifact": {
    readonly value: NativeInferenceArtifact
    readonly info: InferenceArtifactInfo
  }
  readonly "inference-session": { readonly value: NativeInferenceSession; readonly info: InferenceSessionInfo }
  readonly "inference-sequence": {
    readonly value: NativeInferenceSequence
    readonly info: InferenceSequenceInfo
  }
}

type HandleRecord<K extends HandleKind> = K extends HandleKind ? {
    readonly owner: object
    readonly kind: K
    disposed: boolean
  } & HandleData[K]
  : never
type AnyHandleRecord = { [K in HandleKind]: HandleRecord<K> }[HandleKind]
type TensorHandleRecord = HandleRecord<"lazy-tensor" | "concrete-tensor">
type OpaqueHandleKind = Exclude<HandleKind, "lazy-tensor" | "concrete-tensor" | "executable">
interface RuntimeHandle {
  readonly "lazy-tensor": Runtime.LazyTensorHandle
  readonly "concrete-tensor": Runtime.ConcreteTensorHandle
  readonly executable: Runtime.ExecutableHandle
  readonly "kv-pool": Runtime.KvPoolHandle
  readonly "kv-sequence": Runtime.KvSequenceHandle
  readonly "inference-artifact": Runtime.InferenceArtifactHandle
  readonly "inference-session": Runtime.InferenceSessionHandle
  readonly "inference-sequence": Runtime.InferenceSequenceHandle
}

interface BackendHandleRegistry {
  [key: symbol]: WeakSet<object> | undefined
}

const handleRecords = new WeakMap<object, AnyHandleRecord>()
// Each adapter module keeps its records private. The shared weak set lets
// separately loaded backend modules distinguish foreign opaque handles from
// arbitrary objects without retaining either.
const backendHandlesKey = Symbol.for("@effect-torch/backend-handles")
// SAFETY: Backend adapters reserve this shared symbol for a WeakSet<object>.
const backendHandleRegistry = globalThis as typeof globalThis & BackendHandleRegistry
const existingBackendHandles = backendHandleRegistry[backendHandlesKey]
const backendHandles = existingBackendHandles ?? new WeakSet<object>()
if (existingBackendHandles === undefined) backendHandleRegistry[backendHandlesKey] = backendHandles

const backendName = "@effect-torch/backend-apple-native"
const device = "metal"
const nativeDtype = (value: Runtime.DType): NativeDType => {
  switch (value) {
    case "f32":
    case "f64":
    case "f16":
    case "bf16":
    case "i64":
    case "u8":
    case "u32":
      // SAFETY: napi-rs generates these same strings as the NativeDType enum.
      return value as NativeDType
  }
}

const description = "Apple Metal"
const inferencePhases: ReadonlySet<string> = new Set([
  "compile",
  "open",
  "admission",
  "prefill",
  "proposer",
  "verify",
  "sample",
  "accept",
  "publish",
  "finish",
  "close",
  "inspect"
])
const isInferenceFailurePhase = (value: string): value is Runtime.InferenceFailurePhase => inferencePhases.has(value)
const isGgufFormat = (value: string): value is Runtime.GgufTensorDescriptor["format"] =>
  value === "F32" || value === "Q2_K" || value === "Q3_K" || value === "Q4_K" || value === "Q5_K" ||
  value === "Q6_K"

const backendError = (
  operation: string,
  phase: Runtime.BackendError["phase"],
  reason: Runtime.BackendError["reason"] = "execution-failed"
) =>
(cause: unknown): Runtime.BackendError =>
  cause instanceof Runtime.BackendError
    ? cause
    : (() => {
      const message = cause instanceof Error ? cause.message : String(cause)
      const nativeInferencePhase = /inference\[([^\]]+)\]/.exec(message)?.[1]
      const fallbackInferencePhase: Runtime.InferenceFailurePhase | undefined = operation === "inferenceCompile"
        ? "compile"
        : operation === "inferenceOpen"
        ? "open"
        : operation === "inferenceAdd"
        ? "prefill"
        : operation === "inferenceRound"
        ? "verify"
        : operation === "inferenceFinish"
        ? "finish"
        : operation === "inferenceInspect" || operation === "inferenceDiagnostics"
        ? "inspect"
        : operation === "inferenceClose"
        ? "close"
        : undefined
      const inferencePhase = nativeInferencePhase !== undefined && isInferenceFailurePhase(nativeInferencePhase)
        ? nativeInferencePhase
        : fallbackInferencePhase
      return new Runtime.BackendError({
        reason: message.includes("tensor was cleared")
          ? "invalid-handle"
          : message.includes("unsupported operation")
          ? "unsupported-operation"
          : reason,
        backend: backendName,
        operation,
        phase,
        message,
        details: { device, error: cause },
        inferencePhase
      })
    })()

/**
 * Connects Effect interruption to the addon's cooperative cancellation token.
 * Native work may finish after fiber interruption. The adapter passes such a
 * result to `onLateSuccess`, which can clear newly returned native tensors. A
 * cancellation rejection remains a fiber interruption. Other exceptions enter
 * the typed backend error channel.
 */
const cancellable = <A>(
  native: NativeAddon,
  operation: string,
  phase: Runtime.BackendError["phase"],
  register: (token: CancellationToken) => Promise<A>,
  onLateSuccess?: (value: A) => void,
  failureReason?: Runtime.BackendError["reason"]
): Effect.Effect<A, Runtime.BackendError> =>
  Effect.callback<A, Runtime.BackendError>((resume, signal) => {
    const token = new native.CancellationToken()
    let lateValue: A | undefined
    let hasLateValue = false
    const clearLateValue = () => {
      if (!hasLateValue) return
      hasLateValue = false
      try {
        // SAFETY: hasLateValue is set only when lateValue receives a resolved A.
        onLateSuccess?.(lateValue as A)
      } catch {
        // The interrupted fiber cannot observe cleanup failures.
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
      pending = register(token)
    } catch (error) {
      signal.removeEventListener("abort", abort)
      resume(Effect.fail(backendError(operation, phase, failureReason)(error)))
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
      (error) => {
        signal.removeEventListener("abort", abort)
        resume(
          token.cancelled || (error instanceof Error && error.message.includes("aborted"))
            ? Effect.interrupt
            : Effect.fail(
              backendError(
                operation,
                phase,
                failureReason ?? (phase === "io" ? "io-failed" : "execution-failed")
              )(error)
            )
        )
      }
    )
  })

/**
 * Constructs a Metal RuntimeService for one device in an already loaded addon.
 * The addon and device ordinal identify the handle ownership domain. Services
 * for different ordinals cannot exchange handles. Construction allocates only
 * adapter metadata; native operations initialize the selected device when first
 * needed.
 *
 * @internal
 */
export const createRuntimeAdapter = (
  native: NativeAddon,
  deviceOrdinal = 0
): Runtime.RuntimeService => {
  if (!Number.isSafeInteger(deviceOrdinal) || deviceOrdinal < 0 || deviceOrdinal > 0xffff_ffff) {
    throw new Error(`Metal device ordinal must be an integer in [0, 4294967295]; received ${deviceOrdinal}`)
  }
  const backendErrorFor = (
    operation: string,
    phase: Runtime.BackendError["phase"],
    reason?: Runtime.BackendError["reason"]
  ) => backendError(operation, phase, reason)
  const cancellableFor = <A>(
    operation: string,
    phase: Runtime.BackendError["phase"],
    register: (token: CancellationToken) => Promise<A>,
    onLateSuccess?: (value: A) => void,
    failureReason?: Runtime.BackendError["reason"]
  ) => cancellable(native, operation, phase, register, onLateSuccess, failureReason)
  const owner = Object.freeze({ native, deviceOrdinal })
  const placementId = `${device}:${deviceOrdinal}`
  const placement: Runtime.Placement = Object.freeze({
    id: placementId,
    deviceType: device,
    description: `${description} device ${deviceOrdinal}`,
    ordinal: deviceOrdinal
  })
  const invalidHandle = (
    operation: string,
    phase: Runtime.BackendError["phase"],
    reason: "invalid-handle" | "foreign-handle",
    kind: HandleKind | "tensor"
  ): Runtime.BackendError =>
    new Runtime.BackendError({
      reason,
      backend: backendName,
      operation,
      phase,
      message: `${operation}: ${reason === "foreign-handle" ? "foreign" : "invalid"} ${kind} handle`,
      details: { device, kind }
    })
  const record = <K extends HandleKind>(
    handle: RuntimeHandle[K],
    kind: K,
    operation: string,
    phase: Runtime.BackendError["phase"]
  ): HandleRecord<K> => {
    const found = handleRecords.get(handle)
    if (found === undefined || found.kind !== kind) {
      throw invalidHandle(operation, phase, backendHandles.has(handle) ? "foreign-handle" : "invalid-handle", kind)
    }
    if (found.disposed) {
      throw new Runtime.BackendError({
        reason: "invalid-handle",
        backend: backendName,
        operation,
        phase,
        message: `${operation}: handle was released`,
        details: { device, kind }
      })
    }
    if (found.owner !== owner) {
      throw invalidHandle(operation, phase, "foreign-handle", kind)
    }
    // SAFETY: The runtime kind check above establishes the generic record variant.
    return found as HandleRecord<K>
  }
  const tensorRecord = (
    handle: Runtime.TensorHandle,
    operation: string,
    phase: Runtime.BackendError["phase"]
  ): TensorHandleRecord => {
    const found = handleRecords.get(handle)
    if (
      found === undefined || (found.kind !== "lazy-tensor" && found.kind !== "concrete-tensor")
    ) {
      throw invalidHandle(
        operation,
        phase,
        backendHandles.has(handle)
          ? "foreign-handle"
          : "invalid-handle",
        "tensor"
      )
    }
    if (found.disposed) {
      throw new Runtime.BackendError({
        reason: "invalid-handle",
        backend: backendName,
        operation,
        phase,
        message: `${operation}: tensor was cleared`,
        details: { device, kind: found.kind }
      })
    }
    if (found.owner !== owner) throw invalidHandle(operation, phase, "foreign-handle", "tensor")
    return found
  }
  const wrapOpaque = <K extends OpaqueHandleKind>(
    kind: K,
    value: HandleData[K]["value"],
    info: HandleData[K]["info"]
  ): RuntimeHandle[K] => {
    // SAFETY: Public handles are opaque identities; their typed data lives only in handleRecords.
    const handle = Object.freeze({}) as RuntimeHandle[K]
    // SAFETY: K indexes the matching native value and metadata types in HandleData.
    const entry = { owner, kind, value, info, disposed: false } as HandleRecord<K>
    handleRecords.set(handle, entry)
    backendHandles.add(handle)
    return handle
  }
  const dtype = (value: string): Runtime.DType => {
    if (
      value === "f32" || value === "f64" || value === "f16" || value === "bf16" || value === "i64" || value === "u8" ||
      value === "u32"
    ) {
      return value
    }
    throw new Error(`native runtime returned unsupported dtype ${value}`)
  }
  const tensorObject = <H extends Runtime.TensorHandle>(
    tag: H["_tag"],
    shape: ReadonlyArray<number>,
    tensorDtype: string,
    tensorDevice: string,
    storage?: Runtime.EncodedTensorStorage
  ): H => {
    if (!shape.every((dimension) => Number.isSafeInteger(dimension) && dimension >= 0)) {
      throw new Error(`native runtime returned invalid shape [${shape}]`)
    }
    if (tensorDevice !== device) {
      throw new Error(`native runtime returned placement ${tensorDevice}, expected ${device}`)
    }
    if (
      storage !== undefined &&
      (tensorDtype !== "f32" || storage.physicalDtype !== "u8" ||
        !storage.physicalShape.every((dimension) => Number.isSafeInteger(dimension) && dimension >= 0))
    ) {
      throw new Error("native runtime returned invalid encoded tensor metadata")
    }
    // SAFETY: The tag argument selects H, and the object supplies every TensorHandle field.
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
      device: tensorDevice,
      placement,
      pipe(this: Runtime.TensorHandle) {
        return pipeArguments(this, arguments)
      }
    }) as H
  }
  // Because `node` calls are synchronous, these fields can carry one request's
  // JavaScript-only structure and declarations through the native constructor
  // into `lazyHandle`. The call consumes or resets them before returning.
  let pendingStructure: StructuralNode | undefined
  let pendingDeclarations: ReadonlySet<InputDeclaration> | undefined
  const lazyHandle = (
    value: LazyTensor,
    logical?: {
      readonly shape: ReadonlyArray<number>
      readonly dtype: Runtime.DType
      readonly storage?: Runtime.EncodedTensorStorage | undefined
    }
  ): Runtime.LazyTensorHandle => {
    const [nativeShape, nativeDtype] = value.metadata()
    const handle = tensorObject<Runtime.LazyTensorHandle>(
      "LazyTensor",
      logical?.shape ?? nativeShape,
      logical?.dtype ?? nativeDtype,
      device,
      logical?.storage
    )
    handleRecords.set(handle, {
      owner,
      kind: "lazy-tensor",
      graph: value,
      structure: pendingStructure,
      declarations: pendingDeclarations,
      disposed: false
    })
    pendingStructure = undefined
    pendingDeclarations = undefined
    backendHandles.add(handle)
    return handle
  }
  const graph = lazyHandle
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
    if (
      expectedShape !== undefined &&
      (value.shape.length !== expectedShape.length ||
        value.shape.some((dimension, index) => dimension !== expectedShape[index]) ||
        value.dtype !== expectedDtype)
    ) {
      throw new Error(
        `native runtime returned physical tensor ${value.dtype} [${value.shape}], expected ${expectedDtype} [${expectedShape}]`
      )
    }
    const graph = native.LazyTensor.fromMaterialized(value)
    const handle = tensorObject<Runtime.ConcreteTensorHandle>(
      "Tensor",
      logical?.shape ?? value.shape,
      logical?.dtype ?? value.dtype,
      value.device,
      logical?.storage
    )
    handleRecords.set(handle, {
      owner,
      kind: "concrete-tensor",
      graph,
      value,
      structure: {
        op: "materialized-parameter",
        inputs: [],
        attributes: { shape: handle.shape, dtype: handle.dtype, storage: handle.storage, device: handle.device }
      },
      disposed: false
    })
    backendHandles.add(handle)
    return handle
  }
  const nativeGraph = (
    handle: Runtime.TensorHandle,
    operation: string,
    phase: Runtime.BackendError["phase"] = "graph"
  ): LazyTensor => tensorRecord(handle, operation, phase).graph!
  const nativeTensor = (
    handle: Runtime.ConcreteTensorHandle,
    operation: string,
    phase: Runtime.BackendError["phase"] = "execute"
  ): NativeTensor => {
    const found = tensorRecord(handle, operation, phase)
    if (found.kind !== "concrete-tensor" || found.value === undefined) {
      throw invalidHandle(operation, phase, "invalid-handle", "concrete-tensor")
    }
    return found.value
  }
  const sameShape = (left: ReadonlyArray<number>, right: ReadonlyArray<number>): boolean =>
    left.length === right.length && left.every((dimension, index) => dimension === right[index])
  const sameStorage = (
    left: Runtime.EncodedTensorStorage | undefined,
    right: Runtime.EncodedTensorStorage | undefined
  ): boolean =>
    left === undefined
      ? right === undefined
      : right !== undefined && left.encoding === right.encoding && left.physicalDtype === right.physicalDtype &&
        sameShape(left.physicalShape, right.physicalShape)
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
  const sameBinding = (left: TensorBinding, right: TensorBinding): boolean =>
    left.dtype === right.dtype && sameShape(left.shape, right.shape) && sameStorage(left.storage, right.storage)
  // Native LazyTensor nodes contain the physical graph but do not expose the
  // shared public slot namespace. JavaScript carries the declarations so
  // compilation can reject gaps and tensor/scalar conflicts before splitting
  // invocation bindings into native tensor and scalar arrays.
  const declarationsFor = (request: Runtime.NodeRequest): ReadonlySet<InputDeclaration> => {
    const declarations = new Set<InputDeclaration>()
    const source = request.op === "constant" || request.op === "zeros" || request.op === "ones" ||
      request.op === "full" || request.op === "randn" || request.op === "uniform" || request.op === "arange" ||
      request.op === "eye" || request.op === "fromBytes" || request.op === "input" || request.op === "scalarInput"
    if (!source) {
      for (const input of request.inputs) {
        for (const declaration of tensorRecord(input, request.op, "graph").declarations ?? []) {
          declarations.add(declaration)
        }
      }
    }
    if (request.op === "input") {
      declarations.add({
        kind: "tensor",
        slot: request.attributes.slot,
        shape: Object.freeze([...request.attributes.shape]),
        dtype: request.attributes.dtype,
        storage: request.attributes.storage
      })
    } else if (request.op === "scalarInput") {
      declarations.add({ kind: "scalar", slot: request.attributes.slot, dtype: request.attributes.dtype })
    }
    return declarations
  }
  const executableBindings = (roots: ReadonlyArray<Runtime.TensorHandle>): ReadonlyArray<TensorBinding> => {
    const slots = new Map<number, InputDeclaration>()
    for (const root of roots) {
      for (const declaration of tensorRecord(root, "compile", "compile").declarations ?? []) {
        if (!Number.isSafeInteger(declaration.slot) || declaration.slot < 0 || declaration.slot > 0xffff_ffff) {
          throw new Error(`compile: input slot ${declaration.slot} is not an unsigned 32-bit integer`)
        }
        const existing = slots.get(declaration.slot)
        if (existing === undefined) {
          slots.set(declaration.slot, declaration)
        } else if (
          existing.kind !== declaration.kind || existing.dtype !== declaration.dtype ||
          (existing.kind === "tensor" && declaration.kind === "tensor" && !sameBinding(existing, declaration))
        ) {
          throw new Error(`compile: repeated input slot ${declaration.slot} has conflicting logical declarations`)
        }
      }
    }
    const ordered = [...slots].sort(([left], [right]) => left - right)
    for (let index = 0; index < ordered.length; index++) {
      if (ordered[index]![0] !== index) throw new Error("compile: input slots must be contiguous from zero")
    }
    return Object.freeze(ordered.flatMap(([, declaration]) =>
      declaration.kind === "scalar"
        ? []
        : [{
          shape: Object.freeze([...declaration.shape]),
          dtype: declaration.dtype,
          storage: declaration.storage === undefined
            ? undefined
            : Object.freeze({
              encoding: declaration.storage.encoding,
              physicalShape: Object.freeze([...declaration.storage.physicalShape]),
              physicalDtype: declaration.storage.physicalDtype
            })
        }]
    ))
  }
  const nativeBinding = (
    handle: Runtime.ConcreteTensorHandle,
    expected: TensorBinding,
    index: number,
    boundedBatch?: { readonly compiled: number; readonly active: number }
  ): NativeTensor => {
    const found = tensorRecord(handle, "execute", "execute")
    const shapeMatches = (actual: ReadonlyArray<number>, compiled: ReadonlyArray<number>): boolean =>
      sameShape(actual, compiled) ||
      (boundedBatch !== undefined && compiled.length > 0 && compiled[0] === boundedBatch.compiled &&
        actual.length === compiled.length && actual[0] === boundedBatch.active &&
        actual.slice(1).every((dimension, shapeIndex) => dimension === compiled[shapeIndex + 1]))
    const storageMatches = expected.storage === undefined
      ? handle.storage === undefined
      : handle.storage !== undefined && expected.storage.encoding === handle.storage.encoding &&
        expected.storage.physicalDtype === handle.storage.physicalDtype &&
        shapeMatches(handle.storage.physicalShape, expected.storage.physicalShape)
    if (
      found.kind !== "concrete-tensor" || found.value === undefined || handle.dtype !== expected.dtype ||
      !shapeMatches(handle.shape, expected.shape) || !storageMatches
    ) {
      throw new Error(`execute: tensor binding ${index} does not match its compiled logical declaration`)
    }
    return found.value
  }
  // Copy native diagnostics into recursively frozen public data. These values
  // are static artifact and planner measurements. Compile timings come from the
  // artifact, so cache hits retain the timings of the original structural-cache
  // entry.
  const executable = (
    value: Executable,
    bindings: ExecutableInfo["bindings"],
    outputs: ExecutableInfo["outputs"],
    state: Runtime.DecodeStateSchema | undefined
  ): Runtime.ExecutableHandle => {
    const nativeDiagnostics = value.diagnostics
    const diagnostics: Runtime.ExecutableDiagnostics = Object.freeze({
      ...nativeDiagnostics,
      instructions: Object.freeze(nativeDiagnostics.instructions.map((instruction) => Object.freeze(instruction))),
      memory: Object.freeze(nativeDiagnostics.memory),
      compilePhases: Object.freeze(nativeDiagnostics.compilePhases.map((phase) => Object.freeze(phase)))
    })
    // SAFETY: ExecutableHandle is an opaque identity whose public diagnostics are supplied here.
    const handle = Object.freeze(
      state === undefined ? { diagnostics } : { state, diagnostics }
    ) as Runtime.ExecutableHandle
    handleRecords.set(handle, {
      owner,
      kind: "executable",
      value,
      info: {
        bindings,
        outputs,
        state
      } satisfies ExecutableInfo,
      disposed: false
    })
    backendHandles.add(handle)
    return handle
  }
  const nativeExecutable = (
    handle: Runtime.ExecutableHandle,
    operation: string
  ): HandleRecord<"executable"> => record(handle, "executable", operation, "execute")
  const pool = (value: NativeKvPool, info: Omit<KvPoolInfo, "key">): Runtime.KvPoolHandle =>
    wrapOpaque("kv-pool", value, { ...info, key: value } satisfies KvPoolInfo)
  const nativePool = (handle: Runtime.KvPoolHandle, operation: string): HandleRecord<"kv-pool"> =>
    record(handle, "kv-pool", operation, "execute")
  const sequence = (value: NativeKvSequence, pool: KvPoolInfo): Runtime.KvSequenceHandle =>
    wrapOpaque("kv-sequence", value, { pool } satisfies KvSequenceInfo)
  const nativeSequence = (handle: Runtime.KvSequenceHandle, operation: string): HandleRecord<"kv-sequence"> =>
    record(handle, "kv-sequence", operation, "execute")
  // Each element in a native result array transfers one owning wrapper. Reject
  // duplicates so two public handles cannot own the same wrapper. Cleanup is
  // best-effort because this path is already discarding the result.
  const clearBuffers = (values: ReadonlyArray<NativeTensor>): void => {
    for (const value of new Set(values)) {
      try {
        value.clear()
      } catch {
        // Best-effort cleanup for interrupted or invalid backend results.
      }
    }
  }
  const mapTensors = (
    values: ReadonlyArray<NativeTensor>,
    logical?: ExecutableInfo["outputs"]
  ): ReadonlyArray<Runtime.ConcreteTensorHandle> => {
    if (new Set(values).size !== values.length) {
      throw new Error("native runtime returned duplicate tensor ownership")
    }
    if (logical !== undefined && logical.length !== values.length) {
      throw new Error(`native runtime returned ${values.length} outputs, expected ${logical.length}`)
    }
    return values.map((value, index) => concreteHandle(value, logical?.[index]))
  }
  // Build a value-independent graph description for the addon's bounded cache.
  // Materialized leaves contribute signatures, not payloads. The native cache
  // revalidates generated bindings. With constantWeights, values become
  // executable constants, so native compilation skips cache reuse. Gradients do
  // not include structure and therefore have no cache key.
  const executableCacheKey = (request: Runtime.CompileRequest): string | undefined => {
    const ids = new Map<object, number>()
    const nodes: Array<unknown> = []
    const visit = (handle: Runtime.TensorHandle): number | undefined => {
      const existing = ids.get(handle)
      if (existing !== undefined) return existing
      const found = tensorRecord(handle, "compile", "compile")
      if (found.structure === undefined) return undefined
      const inputs: Array<number> = []
      for (const input of found.structure.inputs) {
        const id = visit(input)
        if (id === undefined) return undefined
        inputs.push(id)
      }
      const id = nodes.length
      ids.set(handle, id)
      nodes.push({
        op: found.structure.op,
        inputs,
        attributes: normalizedStructure(found.structure.attributes),
        shape: handle.shape,
        dtype: handle.dtype,
        device: handle.device
      })
      return id
    }
    const roots: Array<number> = []
    for (const root of request.roots) {
      const id = visit(root)
      if (id === undefined) return undefined
      roots.push(id)
    }
    const options = request.options === undefined
      ? undefined
      : {
        optimize: request.options.optimize,
        constantWeights: request.options.constantWeights
      }
    return structuralCacheKey({ nodes, roots, options, state: request.state })
  }
  // Map each public request variant to its native LazyTensor constructor or
  // method. Copy public arrays before they cross N-API. Check every input's
  // owner and validate native metadata before creating the opaque result handle.
  const node = (request: Runtime.NodeRequest): Effect.Effect<Runtime.LazyTensorHandle, Runtime.BackendError> =>
    Effect.try({
      try: () => {
        pendingDeclarations = declarationsFor(request)
        pendingStructure = {
          op: request.op,
          inputs: [...request.inputs],
          attributes: "attributes" in request ? request.attributes : {}
        }
        const operation = request.op
        switch (request.op) {
          case "constant": {
            for (const exemplar of request.inputs) nativeGraph(exemplar, operation)
            return graph(
              native.LazyTensor.constant(
                request.attributes.value,
                nativeDtype(request.attributes.dtype),
                deviceOrdinal
              )
            )
          }
          case "zeros": {
            for (const exemplar of request.inputs) nativeGraph(exemplar, operation)
            return graph(
              native.LazyTensor.zeros(
                [...request.attributes.shape],
                nativeDtype(request.attributes.dtype),
                deviceOrdinal
              )
            )
          }
          case "ones": {
            for (const exemplar of request.inputs) nativeGraph(exemplar, operation)
            return graph(
              native.LazyTensor.ones(
                [...request.attributes.shape],
                nativeDtype(request.attributes.dtype),
                deviceOrdinal
              )
            )
          }
          case "full": {
            for (const exemplar of request.inputs) nativeGraph(exemplar, operation)
            return graph(
              native.LazyTensor.full(
                [...request.attributes.shape],
                request.attributes.value,
                nativeDtype(request.attributes.dtype),
                deviceOrdinal
              )
            )
          }
          case "randn":
            return graph(
              native.LazyTensor.randn(
                [...request.attributes.shape],
                nativeDtype(request.attributes.dtype),
                deviceOrdinal
              )
            )
          case "uniform":
            return graph(
              native.LazyTensor.uniform(
                [...request.attributes.shape],
                request.attributes.lo,
                request.attributes.hi,
                nativeDtype(request.attributes.dtype),
                deviceOrdinal
              )
            )
          case "arange":
            return graph(
              native.LazyTensor.arange(
                request.attributes.start,
                request.attributes.end,
                request.attributes.step,
                nativeDtype(request.attributes.dtype),
                deviceOrdinal
              )
            )
          case "eye":
            return graph(
              native.LazyTensor.eye(
                request.attributes.n,
                nativeDtype(request.attributes.dtype),
                deviceOrdinal
              )
            )
          case "fromBytes":
            return graph(
              native.LazyTensor.fromBytes(
                request.attributes.data,
                [...request.attributes.shape],
                nativeDtype(request.attributes.dtype),
                deviceOrdinal
              )
            )
          case "input": {
            for (const exemplar of request.inputs) nativeGraph(exemplar, operation)
            const storage = request.attributes.storage
            if (storage !== undefined && !validEncodedGeometry(request.attributes.shape, storage)) {
              throw new Error("input: encoded storage does not match its logical GGML geometry")
            }
            return lazyHandle(
              native.LazyTensor.input(
                request.attributes.slot,
                [...(storage?.physicalShape ?? request.attributes.shape)],
                nativeDtype(storage?.physicalDtype ?? request.attributes.dtype),
                deviceOrdinal
              ),
              {
                shape: request.attributes.shape,
                dtype: request.attributes.dtype,
                storage
              }
            )
          }
          case "scalarInput":
            return graph(
              native.LazyTensor.scalarInput(
                request.attributes.slot,
                nativeDtype(request.attributes.dtype),
                deviceOrdinal
              )
            )
          case "add":
            return graph(nativeGraph(request.inputs[0], operation).add(nativeGraph(request.inputs[1], operation)))
          case "sub":
            return graph(nativeGraph(request.inputs[0], operation).sub(nativeGraph(request.inputs[1], operation)))
          case "mul":
            return graph(nativeGraph(request.inputs[0], operation).mul(nativeGraph(request.inputs[1], operation)))
          case "div":
            return graph(nativeGraph(request.inputs[0], operation).div(nativeGraph(request.inputs[1], operation)))
          case "maximum":
            return graph(nativeGraph(request.inputs[0], operation).maximum(nativeGraph(request.inputs[1], operation)))
          case "minimum":
            return graph(nativeGraph(request.inputs[0], operation).minimum(nativeGraph(request.inputs[1], operation)))
          case "eq":
            return graph(nativeGraph(request.inputs[0], operation).eq(nativeGraph(request.inputs[1], operation)))
          case "gt":
            return graph(nativeGraph(request.inputs[0], operation).gt(nativeGraph(request.inputs[1], operation)))
          case "lt":
            return graph(nativeGraph(request.inputs[0], operation).lt(nativeGraph(request.inputs[1], operation)))
          case "ge":
            return graph(nativeGraph(request.inputs[0], operation).ge(nativeGraph(request.inputs[1], operation)))
          case "le":
            return graph(nativeGraph(request.inputs[0], operation).le(nativeGraph(request.inputs[1], operation)))
          case "matmul":
            return graph(nativeGraph(request.inputs[0], operation).matmul(nativeGraph(request.inputs[1], operation)))
          case "solve":
            return graph(nativeGraph(request.inputs[0], operation).solve(nativeGraph(request.inputs[1], operation)))
          case "concat":
            return graph(
              nativeGraph(request.inputs[0], operation).concat(
                nativeGraph(request.inputs[1], operation),
                request.attributes.dim
              )
            )
          case "neg":
            return graph(nativeGraph(request.inputs[0], operation).neg())
          case "abs":
            return graph(nativeGraph(request.inputs[0], operation).abs())
          case "sqrt":
            return graph(nativeGraph(request.inputs[0], operation).sqrt())
          case "exp":
            return graph(nativeGraph(request.inputs[0], operation).exp())
          case "log":
            return graph(nativeGraph(request.inputs[0], operation).log())
          case "sin":
            return graph(nativeGraph(request.inputs[0], operation).sin())
          case "cos":
            return graph(nativeGraph(request.inputs[0], operation).cos())
          case "tanh":
            return graph(nativeGraph(request.inputs[0], operation).tanh())
          case "relu":
            return graph(nativeGraph(request.inputs[0], operation).relu())
          case "erf":
            return graph(nativeGraph(request.inputs[0], operation).erf())
          case "floor":
            return graph(nativeGraph(request.inputs[0], operation).floor())
          case "ceil":
            return graph(nativeGraph(request.inputs[0], operation).ceil())
          case "round":
            return graph(nativeGraph(request.inputs[0], operation).round())
          case "sign":
            return graph(nativeGraph(request.inputs[0], operation).sign())
          case "inverse":
            return graph(nativeGraph(request.inputs[0], operation).inverse())
          case "det":
            return graph(nativeGraph(request.inputs[0], operation).det())
          case "stopGradient":
            return graph(nativeGraph(request.inputs[0], operation).stopGradient())
          case "expose":
            return graph(nativeGraph(request.inputs[0], operation).expose(request.attributes.name))
          case "checkpoint":
            return graph(nativeGraph(request.inputs[0], operation).checkpoint())
          case "gelu":
            return graph(nativeGraph(request.inputs[0], operation).gelu(request.attributes.approximate))
          case "pow":
            return graph(nativeGraph(request.inputs[0], operation).pow(request.attributes.exponent))
          case "cast":
            return graph(nativeGraph(request.inputs[0], operation).cast(nativeDtype(request.attributes.dtype)))
          case "whereCond":
            return graph(
              nativeGraph(request.inputs[0], operation).whereCond(
                nativeGraph(request.inputs[1], operation),
                nativeGraph(request.inputs[2], operation)
              )
            )
          case "argmax":
            return graph(nativeGraph(request.inputs[0], operation).argmax(request.attributes.dim))
          case "argmin":
            return graph(nativeGraph(request.inputs[0], operation).argmin(request.attributes.dim))
          case "cumsum":
            return graph(nativeGraph(request.inputs[0], operation).cumsum(request.attributes.dim))
          case "indexSelect":
            return graph(
              nativeGraph(request.inputs[0], operation).indexSelect(
                request.attributes.dim,
                nativeGraph(request.inputs[1], operation)
              )
            )
          case "scatterAdd":
            return graph(
              nativeGraph(request.inputs[0], operation).scatterAdd(
                request.attributes.dim,
                nativeGraph(request.inputs[1], operation),
                nativeGraph(request.inputs[2], operation)
              )
            )
          case "gather":
            return graph(
              nativeGraph(request.inputs[0], operation).gather(
                request.attributes.dim,
                nativeGraph(request.inputs[1], operation)
              )
            )
          case "crossEntropy":
            return graph(
              nativeGraph(request.inputs[0], operation).crossEntropy(
                nativeGraph(request.inputs[1], operation),
                request.attributes.ignoreIndex
              )
            )
          case "scaledDotProductAttention":
            return graph(
              nativeGraph(request.inputs[0], operation).scaledDotProductAttention(
                nativeGraph(request.inputs[1], operation),
                nativeGraph(request.inputs[2], operation),
                request.attributes.scale,
                request.attributes.causal,
                request.attributes.window === undefined
                  ? -1
                  : request.attributes.window === null
                  ? 0
                  : request.attributes.window
              )
            )
          case "kdaChunk":
            return graph(
              nativeGraph(request.inputs[0], operation).kdaChunk(
                nativeGraph(request.inputs[1], operation),
                nativeGraph(request.inputs[2], operation),
                nativeGraph(request.inputs[3], operation),
                nativeGraph(request.inputs[4], operation),
                request.attributes.scale
              )
            )
          case "shortConv1d":
            return graph(
              nativeGraph(request.inputs[0], operation).shortConv1d(nativeGraph(request.inputs[1], operation))
            )
          case "positionEmbedding":
            return graph(nativeGraph(request.inputs[0], operation).positionEmbedding(request.attributes.seqLen))
          case "rotaryEmbedding":
            return graph(
              nativeGraph(request.inputs[0], operation).rotaryEmbedding(
                request.attributes.seqLen,
                request.attributes.theta,
                request.attributes.layout
              )
            )
          case "layerNorm":
            return graph(
              nativeGraph(request.inputs[0], operation).layerNorm(
                nativeGraph(request.inputs[1], operation),
                nativeGraph(request.inputs[2], operation),
                request.attributes.eps
              )
            )
          case "rmsNorm":
            return graph(
              nativeGraph(request.inputs[0], operation).rmsNorm(
                request.inputs[1] === undefined ? undefined : nativeGraph(request.inputs[1], operation),
                request.attributes.eps
              )
            )
          case "linear":
            return graph(
              nativeGraph(request.inputs[0], operation).linear(
                nativeGraph(request.inputs[1], operation),
                nativeGraph(request.inputs[2], operation)
              )
            )
          case "quantizedLinear":
            return graph(
              nativeGraph(request.inputs[0], operation).quantizedLinear(
                nativeGraph(request.inputs[1], operation),
                request.inputs[2] === undefined ? undefined : nativeGraph(request.inputs[2], operation),
                request.attributes.encoding,
                request.attributes.logicalShape[0],
                request.attributes.logicalShape[1]
              )
            )
          case "quantizedEmbedding":
            return graph(
              nativeGraph(request.inputs[0], operation).quantizedEmbedding(
                nativeGraph(request.inputs[1], operation),
                request.attributes.encoding,
                request.attributes.logicalShape[0],
                request.attributes.logicalShape[1],
                request.attributes.paddingIndex
              )
            )
          case "conv1d":
            return graph(
              nativeGraph(request.inputs[0], operation).conv1d(
                nativeGraph(request.inputs[1], operation),
                request.attributes.stride,
                request.attributes.padding,
                request.attributes.dilation,
                request.attributes.groups
              )
            )
          case "conv2d":
            return graph(
              nativeGraph(request.inputs[0], operation).conv2d(
                nativeGraph(request.inputs[1], operation),
                request.attributes.stride,
                request.attributes.padding,
                request.attributes.dilation,
                request.attributes.groups
              )
            )
          case "sum":
            return graph(
              nativeGraph(request.inputs[0], operation).sum(
                [...request.attributes.dims],
                request.attributes.keepdims
              )
            )
          case "prod":
            return graph(
              nativeGraph(request.inputs[0], operation).prod(
                [...request.attributes.dims],
                request.attributes.keepdims
              )
            )
          case "mean":
            return graph(
              nativeGraph(request.inputs[0], operation).mean(
                [...request.attributes.dims],
                request.attributes.keepdims
              )
            )
          case "max":
            return graph(
              nativeGraph(request.inputs[0], operation).max(
                [...request.attributes.dims],
                request.attributes.keepdims
              )
            )
          case "min":
            return graph(
              nativeGraph(request.inputs[0], operation).min(
                [...request.attributes.dims],
                request.attributes.keepdims
              )
            )
          case "reshape":
            return graph(nativeGraph(request.inputs[0], operation).reshape([...request.attributes.shape]))
          case "permute":
            return graph(nativeGraph(request.inputs[0], operation).permute([...request.attributes.dims]))
          case "slice":
            return graph(
              nativeGraph(request.inputs[0], operation).slice(
                request.attributes.ranges.map((range) => [...range])
              )
            )
          case "broadcastTo":
            return graph(nativeGraph(request.inputs[0], operation).broadcastTo([...request.attributes.shape]))
          case "vmap":
            return graph(
              nativeGraph(request.inputs[0], operation).vmap(
                nativeGraph(request.inputs[1], operation),
                nativeGraph(request.inputs[2], operation),
                request.attributes.dim
              )
            )
          case "adamwStep":
            return graph(
              nativeGraph(request.inputs[0], operation).adamwStep(
                nativeGraph(request.inputs[1], operation),
                nativeGraph(request.inputs[2], operation),
                nativeGraph(request.inputs[3], operation),
                nativeGraph(request.inputs[4], operation),
                nativeGraph(request.inputs[5], operation),
                nativeGraph(request.inputs[6], operation),
                request.attributes.beta1,
                request.attributes.beta2,
                request.attributes.eps,
                request.attributes.weightDecay
              )
            )
          case "adamwOut":
            return graph(nativeGraph(request.inputs[0], operation).adamwOut(request.attributes.index))
          case "sgdStep":
            return graph(
              nativeGraph(request.inputs[0], operation).sgdStep(
                nativeGraph(request.inputs[1], operation),
                nativeGraph(request.inputs[2], operation),
                nativeGraph(request.inputs[3], operation),
                nativeGraph(request.inputs[4], operation),
                request.attributes.momentum,
                request.attributes.dampening,
                request.attributes.nesterov,
                request.attributes.weightDecay
              )
            )
          case "sgdOut":
            return graph(nativeGraph(request.inputs[0], operation).sgdOut(request.attributes.index))
        }
        const unhandled: never = request
        void unhandled
        throw new Runtime.BackendError({
          reason: "unsupported-operation",
          backend: backendName,
          operation,
          phase: "graph",
          message: `unsupported graph operation ${operation}`,
          details: { device }
        })
      },
      catch: (error) => {
        pendingStructure = undefined
        pendingDeclarations = undefined
        return backendErrorFor(request.op, "graph")(error)
      }
    })
  const mapCompileOptions = (
    options: Runtime.ExecutableCompileOptions | undefined
  ): NativeCompileOptions | undefined => {
    if (options === undefined) return undefined
    return {
      optimize: options.optimize,
      constantWeights: options.constantWeights
    }
  }
  const uint32 = (value: number, name: string, allowZero: boolean): number => {
    if (!Number.isSafeInteger(value) || value < (allowZero ? 0 : 1) || value > 0xffff_ffff) {
      throw new Error(`compile: ${name} must be ${allowZero ? "a non-negative" : "a positive"} uint32`)
    }
    return value
  }
  const mapStateRequest = (
    state: Runtime.DecodeStateRequest | undefined
  ):
    | {
      readonly request: Runtime.DecodeStateRequest
      readonly native: NativeKvStateSchema
    }
    | undefined =>
  {
    if (state === undefined) return undefined
    const request: Runtime.DecodeStateRequest = {
      maxTokens: uint32(state.maxTokens, "state.maxTokens", false),
      blockSize: uint32(state.blockSize, "state.blockSize", false),
      kvDtype: state.kvDtype,
      window: state.window === undefined ? undefined : uint32(state.window, "state.window", true),
      currentBlockAttention: state.currentBlockAttention,
      batch: uint32(state.batch, "state.batch", false),
      packedCausalChains: state.packedCausalChains === undefined
        ? undefined
        : {
          rowsPerSequence: uint32(
            state.packedCausalChains.rowsPerSequence,
            "state.packedCausalChains.rowsPerSequence",
            false
          )
        },
      lastTokenRow: state.lastTokenRow,
      outputSelections: state.outputSelections === undefined
        ? undefined
        : Object.freeze([...state.outputSelections])
    }
    // SAFETY: Decode-state unions match the generated napi-rs string enums.
    return {
      request,
      native: {
        maxTokens: request.maxTokens,
        blockSize: request.blockSize,
        kvDtype: nativeDtype(request.kvDtype),
        window: request.window,
        currentBlockAttention: request.currentBlockAttention as NativeCurrentBlockAttention | undefined,
        batch: request.batch,
        packedCausalChains: request.packedCausalChains === undefined
          ? undefined
          : { rowsPerSequence: request.packedCausalChains.rowsPerSequence },
        lastTokenRow: request.lastTokenRow,
        outputSelections: request.outputSelections?.map((selection) =>
          selection === "allRows"
            ? "AllRows" as NativeDecodeOutputSelection
            : selection === "splitLastTokenRow"
            ? "SplitLastTokenRow" as NativeDecodeOutputSelection
            : "BatchedLastTokenRow" as NativeDecodeOutputSelection
        )
      }
    }
  }
  const completeStateSchema = (
    value: Executable,
    requested: Runtime.DecodeStateRequest | undefined
  ): Runtime.DecodeStateSchema | undefined => {
    if (value.stateful !== (requested !== undefined)) {
      throw new Error("compile: native executable statefulness disagrees with the requested state")
    }
    if (!value.stateful) return undefined
    const state = requested!
    const geometry = {
      batch: value.batch,
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
    }
    for (const [name, inferred] of Object.entries(geometry)) {
      uint32(inferred, `native ${name}`, name !== "batch")
    }
    if (geometry.batch !== state.batch) {
      throw new Error(
        `compile: native batch ${geometry.batch} disagrees with requested batch ${state.batch}`
      )
    }
    const nativePackedRows = value.packedRowsPerSequence ?? undefined
    if (
      nativePackedRows !== state.packedCausalChains?.rowsPerSequence ||
      (nativePackedRows !== undefined &&
        (!Number.isSafeInteger(nativePackedRows) || nativePackedRows <= 0 ||
          geometry.batch * nativePackedRows > 0xffff_ffff))
    ) {
      throw new Error("compile: native packed causal-chain layout disagrees with the requested state")
    }
    if (!Predicate.isBoolean(value.allowsWindowEviction)) {
      throw new Error("compile: native executable returned an invalid window eviction policy")
    }
    return Object.freeze({
      maxTokens: state.maxTokens,
      blockSize: state.blockSize,
      kvDtype: state.kvDtype,
      window: state.window === undefined || !value.allowsWindowEviction ? undefined : state.window,
      currentBlockAttention: state.currentBlockAttention,
      lastTokenRow: state.lastTokenRow,
      outputSelections: state.outputSelections === undefined
        ? undefined
        : Object.freeze([...state.outputSelections]),
      packedCausalChains: state.packedCausalChains === undefined
        ? undefined
        : Object.freeze({
          rowsPerSequence: state.packedCausalChains.rowsPerSequence
        }),
      ...geometry
    })
  }
  const executionError = (
    message: string,
    reason: Runtime.BackendError["reason"] = "execution-failed"
  ): Runtime.BackendError =>
    new Runtime.BackendError({
      reason,
      backend: backendName,
      operation: "execute",
      phase: "execute",
      message,
      details: { device }
    })
  // A stateful invocation mutably borrows distinct sequences from one compatible
  // pool. Before native execution stages its transactional updates, the adapter
  // checks the completed compile schema, token-row shape, and non-windowed
  // capacity.
  const resolveExecutionState = (
    schema: Runtime.DecodeStateSchema | undefined,
    invocation: Runtime.ExecutionStateInvocation | undefined
  ):
    | readonly [
      sequences: Array<NativeKvSequence>,
      slots: Array<number>,
      activeMask: Array<boolean>,
      validLengths: Array<number>,
      advances: Array<number>,
      tokens: Array<Array<number>>
    ]
    | readonly [
      sequences: undefined,
      slots: undefined,
      activeMask: undefined,
      validLengths: undefined,
      advances: undefined,
      tokens: undefined
    ] =>
  {
    if (schema === undefined) {
      if (invocation !== undefined) {
        throw executionError("execute: stateless executable does not accept state")
      }
      return [undefined, undefined, undefined, undefined, undefined, undefined]
    }
    if (invocation === undefined) {
      throw executionError("execute: stateful executable requires state")
    }
    if (
      invocation.sequences.length === 0 ||
      invocation.sequences.length > schema.batch ||
      invocation.slots.length !== invocation.sequences.length ||
      invocation.tokens.length !== invocation.sequences.length ||
      invocation.activeMask.length !== schema.batch ||
      invocation.validLengths.length !== schema.batch ||
      invocation.advances.length !== schema.batch
    ) {
      throw executionError(
        `execute: expected 1..=${schema.batch} sequences with one token row each`
      )
    }
    const sequenceRecords = invocation.sequences.map((handle) => nativeSequence(handle, "execute"))
    const sequenceInfos = sequenceRecords.map((entry) => entry.info)
    const firstPool = sequenceInfos[0]!.pool
    if (
      sequenceInfos.some((entry) => entry.pool.key !== firstPool.key) ||
      new Set(invocation.sequences).size !== sequenceRecords.length ||
      invocation.slots.some((slot) => !Number.isSafeInteger(slot) || slot < 0 || slot >= schema.batch) ||
      new Set(invocation.slots).size !== sequenceRecords.length ||
      invocation.activeMask.some((active, slot) => active !== invocation.slots.includes(slot)) ||
      invocation.validLengths.some((length, slot) =>
        !Number.isSafeInteger(length) || length < 0 ||
        (invocation.activeMask[slot] ? length === 0 : length !== 0)
      ) ||
      invocation.advances.some((advance, slot) => advance !== invocation.validLengths[slot]) ||
      invocation.tokens.some((row, index) => row.length !== invocation.advances[invocation.slots[index]!]!) ||
      firstPool.maxTokens !== schema.maxTokens ||
      firstPool.blockSize !== schema.blockSize ||
      firstPool.dtype !== schema.kvDtype ||
      firstPool.layers !== schema.layers ||
      firstPool.kvHeads !== schema.kvHeads ||
      firstPool.headDim !== schema.headDim ||
      firstPool.kdaLayers !== schema.kdaLayers ||
      firstPool.kdaHeads !== schema.kdaHeads ||
      firstPool.kdaHeadDim !== schema.kdaHeadDim ||
      firstPool.kdaValueDim !== schema.kdaValueDim ||
      firstPool.convLayers !== schema.convLayers ||
      firstPool.convChannels !== schema.convChannels ||
      firstPool.convKernel !== schema.convKernel
    ) {
      throw invalidHandle("execute", "execute", "invalid-handle", "kv-sequence")
    }
    if (
      invocation.tokens.some((row) => row.length === 0) ||
      invocation.tokens.some((row) =>
        row.some((token) => !Number.isSafeInteger(token) || token < 0 || token > 0xffff_ffff)
      )
    ) {
      throw executionError("execute: invalid token rows for compiled state schema")
    }
    if (
      schema.window === undefined &&
      sequenceRecords.some((entry, index) => entry.value.cursor + invocation.tokens[index]!.length > schema.maxTokens)
    ) {
      throw executionError(`execute: sequence context exceeds pool capacity ${schema.maxTokens}`)
    }
    return [
      sequenceRecords.map((entry) => entry.value),
      [...invocation.slots],
      [...invocation.activeMask],
      [...invocation.validLengths],
      [...invocation.advances],
      invocation.tokens.map((row) => [...row])
    ]
  }
  const resolveExecutableInvocation = (
    handle: Runtime.ExecutableHandle,
    invocation: Runtime.ExecutionInvocation
  ) => {
    const executableRecord = nativeExecutable(handle, "execute")
    if (Object.keys(invocation.runtimeValues).length !== 0) {
      throw executionError(
        "execute: runtime values are not supported by the Apple Metal backend",
        "unsupported-operation"
      )
    }
    const info = executableRecord.info
    if (invocation.bindings.length !== info.bindings.length) {
      throw executionError(
        `execute: received ${invocation.bindings.length} tensor bindings, expected ${info.bindings.length}`
      )
    }
    const inputs = invocation.bindings.map((input, index) => nativeBinding(input, info.bindings[index]!, index))
    if (info.state !== undefined && invocation.scalars.length !== 0) {
      throw executionError("execute: stateful executable does not accept scalar inputs")
    }
    const [sequences, slots, activeMask, validLengths, advances, tokens] = resolveExecutionState(
      info.state,
      invocation.state
    )
    return {
      executable: executableRecord.value,
      info,
      inputs,
      scalars: [...invocation.scalars],
      sequences,
      slots,
      activeMask,
      validLengths,
      advances,
      tokens
    }
  }
  // Pools own fixed KV slabs, prefix-cache state, and recurrent geometry.
  // Sequence wrappers retain KV block references and per-sequence recurrent
  // tensors. releaseSequence invalidates the public sequence and returns its
  // block references. Native finalization handles all remaining sequence, pool,
  // and executable storage.
  const decode: Runtime.DecodeRuntime = {
    makePool: (options) =>
      Effect.try({
        try: () =>
          pool(
            native.NativeKvPool.forDevice(
              options.layers,
              options.kvHeads,
              options.headDim,
              options.maxTokens,
              options.blockSize,
              nativeDtype(options.dtype),
              {
                kdaLayers: options.kdaLayers,
                kdaHeads: options.kdaHeads,
                kdaHeadDim: options.kdaHeadDim,
                kdaValueDim: options.kdaValueDim,
                convLayers: options.convLayers,
                convChannels: options.convChannels,
                convKernel: options.convKernel
              },
              deviceOrdinal
            ),
            {
              layers: options.layers,
              kvHeads: options.kvHeads,
              headDim: options.headDim,
              maxTokens: options.maxTokens,
              blockSize: options.blockSize,
              dtype: options.dtype,
              kdaLayers: options.kdaLayers,
              kdaHeads: options.kdaHeads,
              kdaHeadDim: options.kdaHeadDim,
              kdaValueDim: options.kdaValueDim,
              convLayers: options.convLayers,
              convChannels: options.convChannels,
              convKernel: options.convKernel
            }
          ),
        catch: backendErrorFor("makeKvPool", "execute")
      }),
    makeSequence: (handle) =>
      Effect.try({
        try: () => {
          const poolRecord = nativePool(handle, "makeKvSequence")
          return sequence(poolRecord.value.makeSequence(), poolRecord.info)
        },
        catch: backendErrorFor("makeKvSequence", "execute")
      }),
    prefillMatch: (handle, tokens) =>
      Effect.try({
        try: () => nativeSequence(handle, "prefillMatch").value.prefillMatch([...tokens]),
        catch: backendErrorFor("prefillMatch", "execute")
      }),
    sequenceCursor: (handle) =>
      Effect.try({
        try: () => nativeSequence(handle, "sequenceCursor").value.cursor,
        catch: backendErrorFor("sequenceCursor", "execute")
      }),
    releaseSequence: (handle) =>
      Effect.try({
        try: () => {
          const sequenceRecord = nativeSequence(handle, "releaseSequence")
          const value = sequenceRecord.value
          value.release()
          sequenceRecord.disposed = true
        },
        catch: backendErrorFor("releaseSequence", "execute")
      })
  }
  const sampling: Runtime.SamplingRuntime = {
    sample: (handle, options) =>
      cancellableFor(
        "sample",
        "execute",
        (token) =>
          nativeTensor(handle, "sample").sample(
            options.temperature,
            options.topK,
            options.topP,
            options.seed,
            options.counter,
            token
          )
      ),
    executeDecode: (handle, invocation, options) =>
      cancellableFor(
        "executeDecode",
        "execute",
        (token) => {
          const resolved = resolveExecutableInvocation(handle, invocation)
          return resolved.executable.executeSampled(
            resolved.inputs,
            resolved.sequences ?? [],
            resolved.slots ?? [],
            resolved.activeMask ?? [],
            resolved.validLengths ?? [],
            resolved.advances ?? [],
            resolved.tokens ?? [],
            options.map((option) => ({ ...option })),
            token
          )
        }
      ),
    executeSpeculative: (request) =>
      cancellableFor(
        "executeSpeculative",
        "execute",
        (token) => {
          const target = nativeExecutable(request.targetVerify, "executeSpeculative").value
          const proposer = nativeExecutable(request.proposerDecode, "executeSpeculative").value
          const targetSequences = request.targetSequences.map((handle) =>
            nativeSequence(handle, "executeSpeculative").value
          )
          const proposerSequences = request.proposerSequences.map((handle) =>
            nativeSequence(handle, "executeSpeculative").value
          )
          return target.executeSpeculative(
            proposer,
            targetSequences,
            proposerSequences,
            [...request.slots],
            [...request.pendingTokens],
            request.sampling.map((options) => ({ ...options })),
            request.maxDraftTokens,
            [...request.pageLimits],
            request.eosTokens.map((tokens) => [...tokens]),
            token
          )
        }
      )
  }
  const normalizedInferenceSampling = (
    base: Runtime.InferenceSamplingOptions,
    override: Runtime.InferenceSamplingOverrides | undefined,
    operation: string
  ): Runtime.InferenceSamplingOptions => {
    const value = {
      temperature: override?.temperature ?? base.temperature,
      topK: override?.topK ?? base.topK,
      topP: override?.topP ?? base.topP,
      seed: override?.seed ?? base.seed
    }
    if (
      !Number.isFinite(value.temperature) || value.temperature < 0 ||
      !Number.isSafeInteger(value.topK) || value.topK < 0 ||
      !Number.isFinite(value.topP) || value.topP <= 0 || value.topP > 1 ||
      !Predicate.isBigInt(value.seed) || value.seed < 0n || value.seed > 0xffff_ffff_ffff_ffffn
    ) {
      throw new Error(`${operation}: invalid inference sampling controls`)
    }
    return Object.freeze(value)
  }
  const nativeInferenceSampling = (
    value: Runtime.InferenceSamplingOptions
  ): NativeInferenceSamplingOptions => ({ ...value })
  const nativeInferenceOverride = (
    value: Runtime.InferenceSamplingOverrides | undefined
  ): NativeInferenceSamplingOverride => value === undefined ? {} : { ...value }
  const inferenceArtifact = (
    value: NativeInferenceArtifact,
    sampling: Runtime.InferenceSamplingOptions
  ): Runtime.InferenceArtifactHandle =>
    wrapOpaque(
      "inference-artifact",
      value,
      {
        sampling
      } satisfies InferenceArtifactInfo
    )
  const inferenceSession = (
    value: NativeInferenceSession,
    artifact: Runtime.InferenceArtifactHandle,
    sampling: Runtime.InferenceSamplingOptions
  ): Runtime.InferenceSessionHandle =>
    wrapOpaque(
      "inference-session",
      value,
      {
        artifact,
        sampling,
        sequences: new Map()
      } satisfies InferenceSessionInfo
    )
  const mapInferenceResult = (
    sessionHandle: Runtime.InferenceSessionHandle,
    nativeSession: NativeInferenceSession,
    result: NativeInferenceRoundResult,
    expected: { readonly count: number; readonly ids?: ReadonlyArray<bigint> },
    newSampling?: ReadonlyArray<Runtime.InferenceSamplingOptions>
  ): Runtime.InferenceRoundResult => {
    const sessionRecord = record(sessionHandle, "inference-session", "inferenceResult", "execute")
    const sessionInfo = sessionRecord.info
    if (
      !Predicate.isBigInt(result.roundId) || result.roundId < 0n || result.roundId > 0xffff_ffff_ffff_ffffn ||
      !Predicate.isBoolean(result.recovered) || !Array.isArray(result.pages) || result.pages.length !== expected.count
    ) {
      throw new Error("inference[publish]: native runtime returned a malformed receipt")
    }
    const seen = new Set<bigint>()
    const additions: Array<readonly [bigint, Runtime.InferenceSequenceHandle]> = []
    const pages = result.pages.map((page, index): Runtime.InferenceTokenPage => {
      if (
        !Predicate.isBigInt(page.sequenceId) || page.sequenceId < 0n || page.sequenceId > 0xffff_ffff_ffff_ffffn ||
        seen.has(page.sequenceId) || (expected.ids !== undefined && page.sequenceId !== expected.ids[index]) ||
        !Array.isArray(page.tokens) || page.tokens.length === 0 ||
        page.tokens.some((token) => !Number.isInteger(token) || token < 0 || token > 0xffff_ffff) ||
        (page.stopReason !== undefined && page.stopReason !== "eos" && page.stopReason !== "maxTokens")
      ) {
        throw new Error("inference[publish]: native runtime returned a malformed token page")
      }
      seen.add(page.sequenceId)
      let handle = sessionInfo.sequences.get(page.sequenceId)
      const sampling = newSampling?.[index]
      if (handle !== undefined && newSampling !== undefined) {
        const existing = record(handle, "inference-sequence", "inferenceResult", "execute")
          .info
        if (
          !result.recovered || sampling === undefined || existing.sampling.temperature !== sampling.temperature ||
          existing.sampling.topK !== sampling.topK || existing.sampling.topP !== sampling.topP ||
          existing.sampling.seed !== sampling.seed
        ) {
          throw new Error("inference[publish]: native runtime returned an existing sequence")
        }
      } else if (handle === undefined) {
        if (sampling === undefined) {
          throw new Error("inference[publish]: native runtime returned an unknown sequence")
        }
        const nativeSequence = nativeSession.sequence(page.sequenceId)
        if (nativeSequence.sequenceId !== page.sequenceId) {
          throw new Error("inference[publish]: native sequence identity is not canonical")
        }
        handle = wrapOpaque(
          "inference-sequence",
          nativeSequence,
          {
            session: sessionHandle,
            sequenceId: page.sequenceId,
            sampling
          } satisfies InferenceSequenceInfo
        )
        additions.push([page.sequenceId, handle])
      }
      return Object.freeze({
        sequence: handle,
        sequenceId: page.sequenceId,
        tokens: Object.freeze([...page.tokens]),
        stopReason: page.stopReason
      })
    })
    const mapped = Object.freeze({
      roundId: result.roundId,
      recovered: result.recovered,
      pages: Object.freeze(pages)
    })
    for (const [sequenceId, handle] of additions) sessionInfo.sequences.set(sequenceId, handle)
    return mapped
  }
  const inference: Runtime.InferenceRuntime = {
    compile: (request) =>
      Effect.try({
        try: () => {
          const sampling = normalizedInferenceSampling(request.sampling, undefined, "inference[compile]")
          const targetPrefills = request.target.prefill.map((handle) =>
            record(handle, "executable", "inferenceCompile", "compile").value
          )
          const targetPrefill = targetPrefills[targetPrefills.length - 1]!
          const targetPrefillBuckets = targetPrefills.slice(0, -1)
          const targetDecode = record(request.target.decode, "executable", "inferenceCompile", "compile")
          const targetVerify = (request.target.verify ?? []).map((handle) =>
            record(handle, "executable", "inferenceCompile", "compile").value
          )
          const targetPool = record(request.target.pool, "kv-pool", "inferenceCompile", "compile")
          const proposerPrefill = request.proposer === undefined
            ? undefined
            : record(request.proposer.prefill, "executable", "inferenceCompile", "compile")
          const proposerDecode = request.proposer === undefined
            ? undefined
            : record(request.proposer.decode, "executable", "inferenceCompile", "compile")
          const proposerPool = request.proposer === undefined
            ? undefined
            : record(request.proposer.pool, "kv-pool", "inferenceCompile", "compile")
          const generalized = request.generalizedProposer
          const valueRef = (route: Runtime.InferenceValueRoute): NativeValueRef => {
            switch (route.kind) {
              case "TargetHidden": {
                const tap = generalized?.plan.hiddenTaps.find((tap) => tap.outputRoot === route.targetOutput)
                if (tap === undefined) throw new Error("inference[compile]: target hidden route has no tap contract")
                return {
                  kind: route.kind,
                  name: tap.name,
                  selectRow: route.selectTargetRow === true ? true : undefined
                }
              }
              case "SharedTokenEmbedding":
              case "SharedLmHead": {
                const kind = route.kind === "SharedTokenEmbedding" ? "TokenEmbedding" : "LmHead"
                const binding = generalized?.plan.sharedTensors.findIndex((candidate) => candidate.kind === kind) ?? -1
                if (binding < 0) throw new Error(`inference[compile]: ${route.kind} route has no shared binding`)
                return { kind: "SharedBinding", binding }
              }
              case "StageOutput":
                if (route.stage === undefined || route.output === undefined) {
                  throw new Error("inference[compile]: stage output route is incomplete")
                }
                return { kind: route.kind, stage: route.stage, output: route.output }
              default:
                return { kind: route.kind }
            }
          }
          const nativePlan: NativeProposerPlan | undefined = generalized === undefined
            ? undefined
            : {
              targetPrefillTaps: (generalized.plan.prefillHiddenTaps ?? []).map((tap) => ({
                name: tap.name,
                output: tap.outputRoot,
                shape: [...tap.value.shape],
                dtype: nativeDtype(tap.value.dtype)
              })),
              targetDecodeTaps: generalized.plan.hiddenTaps.map((tap) => ({
                name: tap.name,
                output: tap.outputRoot,
                shape: [...tap.value.shape],
                dtype: nativeDtype(tap.value.dtype)
              })),
              targetVerifyTaps: (generalized.plan.verifyHiddenTaps ?? []).map((tap) => ({
                name: tap.name,
                output: tap.outputRoot,
                shape: [...tap.value.shape],
                dtype: nativeDtype(tap.value.dtype)
              })),
              sharedTargetBindings: generalized.plan.sharedTensors.map((binding, tensor) => ({
                kind: binding.kind,
                name: binding.name,
                tensor,
                shape: [...binding.value.shape],
                dtype: nativeDtype(binding.value.dtype)
              })),
              stages: generalized.plan.stages.map((stage, executable) => {
                return {
                  executable: stage.operationId === "HistoryLookup" ? undefined : executable,
                  operationId: stage.operationId,
                  layoutId: stage.layoutId,
                  historyLookup: stage.historyLookup === undefined
                    ? undefined
                    : {
                      id: stage.historyLookup.id,
                      minMatchTokens: stage.historyLookup.minMatchTokens,
                      maxMatchTokens: stage.historyLookup.maxMatchTokens
                    },
                  inputs: stage.inputs.map((input) => ({ slot: input.slot, value: valueRef(input.value) })),
                  outputs: stage.outputs.map((output) => ({
                    shape: [...output.shape],
                    dtype: nativeDtype(output.dtype)
                  }))
                }
              }),
              state: generalized.plan.state.kind === "None"
                ? { kind: "None" }
                : {
                  kind: "Kv",
                  schemaId: generalized.plan.state.schemaId
                },
              commit: generalized.plan.state.commitKind === "None"
                ? undefined
                : generalized.plan.state.commitKind === "AutoregressiveChain"
                ? { kind: generalized.plan.state.commitKind, stage: generalized.plan.state.commitStages[0] }
                : { kind: generalized.plan.state.commitKind, stages: [...generalized.plan.state.commitStages] },
              output: {
                topology: generalized.plan.output.topology,
                probabilities: generalized.plan.output.probabilities,
                tokenIds: valueRef(generalized.plan.output.tokenIds),
                probabilityRows: generalized.plan.output.probabilityRows === undefined
                  ? undefined
                  : valueRef(generalized.plan.output.probabilityRows),
                parents: generalized.plan.output.parents === undefined
                  ? undefined
                  : valueRef(generalized.plan.output.parents),
                confidence: generalized.plan.output.confidence === undefined
                  ? undefined
                  : valueRef(generalized.plan.output.confidence)
              },
              tokenMap: generalized.plan.tokenMap.kind === "Identity"
                ? { kind: "Identity", fingerprint: generalized.plan.tokenMap.fingerprint }
                : {
                  kind: "Table",
                  fingerprint: generalized.plan.tokenMap.fingerprint,
                  proposerVocabulary: generalized.plan.tokenMap.proposerVocabulary,
                  targetIds: generalized.plan.tokenMap.targetIds === undefined
                    ? undefined
                    : [...generalized.plan.tokenMap.targetIds]
                },
              trainedMaxRows: generalized.plan.trainedMaxRows
            }
          const stageExecutables = generalized?.stageExecutables.map((executable) =>
            record(executable, "executable", "inferenceCompile", "compile").value
          )
          const sharedTargetTensors = generalized?.sharedTensors.map((tensor) =>
            nativeTensor(tensor, "inferenceCompile", "compile")
          )
          const replay = generalized?.replay
          const replayPrefills = replay?.prefill.map((handle) =>
            record(handle, "executable", "inferenceCompile", "compile").value
          )
          return inferenceArtifact(
            new native.NativeInferenceArtifact(
              targetPrefill,
              targetDecode.value,
              targetVerify,
              targetPool.value,
              proposerPrefill?.value,
              proposerDecode?.value,
              proposerPool?.value,
              request.proposer?.maxDraftTokens ?? generalized?.maxDraftTokens,
              request.batchSize,
              nativeDtype(request.tokenDtype),
              nativeInferenceSampling(sampling),
              nativePlan,
              stageExecutables,
              sharedTargetTensors,
              replayPrefills === undefined ? undefined : replayPrefills[replayPrefills.length - 1]!,
              replay === undefined
                ? undefined
                : record(replay.decode, "executable", "inferenceCompile", "compile").value,
              replay === undefined
                ? []
                : replay.verify.map((handle) => record(handle, "executable", "inferenceCompile", "compile").value),
              replay === undefined
                ? undefined
                : record(replay.pool, "kv-pool", "inferenceCompile", "compile").value,
              targetPrefillBuckets,
              replayPrefills?.slice(0, -1)
            ),
            sampling
          )
        },
        catch: backendErrorFor("inferenceCompile", "compile", "compilation-failed")
      }),
    open: (artifactHandle) =>
      Effect.try({
        try: () => {
          const artifactRecord = record(artifactHandle, "inference-artifact", "inferenceOpen", "execute")
          const info = artifactRecord.info
          return inferenceSession(
            artifactRecord.value.open(),
            artifactHandle,
            info.sampling
          )
        },
        catch: backendErrorFor("inferenceOpen", "execute")
      }),
    add: (sessionHandle, request) => {
      let resolved: InferenceAddResolution
      try {
        const sessionRecord = record(sessionHandle, "inference-session", "inferenceAdd", "execute")
        const info = sessionRecord.info
        resolved = {
          nativeSession: sessionRecord.value,
          prompts: request.entries.map((entry) => nativeTensor(entry.prompt, "inferenceAdd")),
          sampling: request.entries.map((entry) =>
            normalizedInferenceSampling(info.sampling, entry.sampling, "inference[admission]")
          )
        }
      } catch (error) {
        return Effect.fail(backendErrorFor("inferenceAdd", "execute")(error))
      }
      return cancellableFor(
        "inferenceAdd",
        "execute",
        (token) =>
          resolved.nativeSession.add(
            resolved.prompts,
            request.entries.map((entry) => nativeInferenceOverride(entry.sampling)),
            request.entries.map((entry) => entry.maxTokens),
            request.entries.map((entry) => [...entry.eosTokens]),
            token
          )
      ).pipe(
        Effect.flatMap((result) =>
          Effect.uninterruptible(
            Effect.try({
              try: () =>
                mapInferenceResult(
                  sessionHandle,
                  resolved.nativeSession,
                  result,
                  { count: request.entries.length },
                  resolved.sampling
                ),
              catch: backendErrorFor("inferenceAdd", "execute")
            })
          )
        )
      )
    },
    runRound: (sessionHandle, request) => {
      let resolved: InferenceRoundResolution
      try {
        const sessionRecord = record(sessionHandle, "inference-session", "inferenceRound", "execute")
        resolved = {
          nativeSession: sessionRecord.value,
          sequences: request.entries.map((entry) => {
            const sequenceRecord = record(entry.sequence, "inference-sequence", "inferenceRound", "execute")
            const info = sequenceRecord.info
            if (info.session !== sessionHandle) {
              throw invalidHandle("inferenceRound", "execute", "invalid-handle", "inference-sequence")
            }
            return sequenceRecord.value
          }),
          sampling: request.entries.map((entry) => {
            const info = record(entry.sequence, "inference-sequence", "inferenceRound", "execute")
              .info
            return normalizedInferenceSampling(info.sampling, entry.sampling, "inference[admission]")
          }),
          ids: request.entries.map((entry) =>
            record(entry.sequence, "inference-sequence", "inferenceRound", "execute").info
              .sequenceId
          )
        }
      } catch (error) {
        return Effect.fail(backendErrorFor("inferenceRound", "execute")(error))
      }
      return cancellableFor(
        "inferenceRound",
        "execute",
        (token) =>
          resolved.nativeSession.runRound(
            resolved.sequences,
            request.entries.map((entry) => nativeInferenceOverride(entry.sampling)),
            token
          )
      ).pipe(
        Effect.flatMap((result) =>
          Effect.uninterruptible(
            Effect.try({
              try: () =>
                mapInferenceResult(
                  sessionHandle,
                  resolved.nativeSession,
                  result,
                  { count: resolved.ids.length, ids: resolved.ids }
                ),
              catch: backendErrorFor("inferenceRound", "execute")
            })
          )
        )
      )
    },
    acknowledge: (sessionHandle, roundId) =>
      Effect.try({
        try: () => {
          const sessionRecord = record(sessionHandle, "inference-session", "inferenceAcknowledge", "execute")
          sessionRecord.value.acknowledge(roundId)
        },
        catch: backendErrorFor("inferenceAcknowledge", "execute")
      }),
    finish: (sessionHandle, sequences) =>
      Effect.try({
        try: () => {
          const sessionRecord = record(sessionHandle, "inference-session", "inferenceFinish", "execute")
          const records = sequences.map((sequence) => {
            const found = record(sequence, "inference-sequence", "inferenceFinish", "execute")
            if (found.info.session !== sessionHandle) {
              throw invalidHandle("inferenceFinish", "execute", "invalid-handle", "inference-sequence")
            }
            return found
          })
          sessionRecord.value.finish(
            records.map((found) => found.value)
          )
          const info = sessionRecord.info
          for (const found of records) {
            found.disposed = true
            info.sequences.delete(found.info.sequenceId)
          }
        },
        catch: backendErrorFor("inferenceFinish", "execute")
      }),
    inspect: (sessionHandle, sequenceHandle) =>
      Effect.try({
        try: () => {
          const sessionRecord = record(sessionHandle, "inference-session", "inferenceInspect", "execute")
          const sequenceRecord = record(sequenceHandle, "inference-sequence", "inferenceInspect", "execute")
          if (sequenceRecord.info.session !== sessionHandle) {
            throw invalidHandle("inferenceInspect", "execute", "invalid-handle", "inference-sequence")
          }
          const inspected = sessionRecord.value.inspect(
            sequenceRecord.value
          )
          if (
            inspected.terminal !== undefined && inspected.terminal !== "eos" && inspected.terminal !== "maxTokens"
          ) {
            throw new Error("inference[inspect]: native runtime returned a malformed inspection")
          }
          return Object.freeze({
            sequenceId: inspected.sequenceId,
            cursor: inspected.cursor,
            terminal: inspected.terminal
          })
        },
        catch: backendErrorFor("inferenceInspect", "execute")
      }),
    close: (sessionHandle) =>
      Effect.try({
        try: () => {
          const sessionRecord = handleRecords.get(sessionHandle)
          if (sessionRecord?.owner !== owner || sessionRecord.kind !== "inference-session") {
            throw invalidHandle(
              "inferenceClose",
              "execute",
              backendHandles.has(sessionHandle) ? "foreign-handle" : "invalid-handle",
              "inference-session"
            )
          }
          if (sessionRecord.disposed) return
          sessionRecord.value.close()
          const info = sessionRecord.info
          for (const sequence of info.sequences.values()) {
            const found = handleRecords.get(sequence)
            if (found !== undefined) found.disposed = true
          }
          info.sequences.clear()
          sessionRecord.disposed = true
        },
        catch: backendErrorFor("inferenceClose", "execute")
      }),
    diagnostics: (artifactHandle) =>
      Effect.try({
        try: () => {
          const artifactRecord = record(artifactHandle, "inference-artifact", "inferenceDiagnostics", "execute")
          const value = artifactRecord.value.inferenceDiagnostics
          const phase = value.lastFailurePhase
          if (phase !== undefined && !isInferenceFailurePhase(phase)) {
            throw new Error("inference[inspect]: native runtime returned an invalid failure phase")
          }
          return Object.freeze({
            roundsStarted: value.roundsStarted,
            roundsCompleted: value.roundsCompleted,
            roundsRecovered: value.roundsRecovered,
            ordinaryRounds: value.ordinaryRounds,
            speculativeRounds: value.speculativeRounds,
            proposedTokens: value.proposedTokens,
            acceptedTokens: value.acceptedTokens,
            emittedTokens: value.emittedTokens,
            provisionalBlocks: value.provisionalBlocks,
            rolledBackBlocks: value.rolledBackBlocks,
            draftNanos: value.draftNanos,
            verificationNanos: value.verificationNanos,
            acceptedLengthHistogram: Object.freeze([...value.acceptedLengthHistogram]),
            targetPoolHighWaterBlocks: value.targetPoolHighWaterBlocks,
            proposerPoolHighWaterBlocks: value.proposerPoolHighWaterBlocks,
            lastRoundId: value.lastRoundId,
            lastFailurePhase: phase
          })
        },
        catch: backendErrorFor("inferenceDiagnostics", "execute")
      })
  }
  // Saving to a path borrows tensors. Successful loads transfer new native
  // tensors to concrete handles owned by the caller. Safetensors cannot represent
  // the adapter's logical-f32 data backed by packed-u8 GGML storage, so the
  // adapter rejects encoded handles rather than writing misleading u8 data. Metal
  // rejects f64 archives instead of falling back to CPU.
  const pathSafetensors: Runtime.PathSafetensors = {
    save: (path, archive) =>
      archive.entries.some((entry) => entry.tensor.storage !== undefined)
        ? Effect.fail(
          new Runtime.BackendError({
            reason: "unsupported-layout",
            backend: backendName,
            operation: "save",
            phase: "io",
            message: "save: encoded tensors cannot be represented by safetensors"
          })
        )
        : cancellableFor(
          "save",
          "io",
          (token) =>
            native.saveTensors(
              path,
              archive.entries.map((entry) => entry.name),
              archive.entries.map((entry) => nativeTensor(entry.tensor, "save", "io")),
              { ...archive.metadata },
              token
            )
        ),
    load: (path) =>
      cancellableFor(
        "load",
        "io",
        (token) => native.loadTensorsForDevice(path, deviceOrdinal, token),
        (archive) => clearBuffers(archive.entries.map((entry) => entry.tensor))
      ).pipe(
        Effect.mapError((error) =>
          error.message.includes("f64 is not supported on Metal")
            ? new Runtime.BackendError({
              reason: "unsupported-placement",
              backend: error.backend,
              operation: error.operation,
              phase: error.phase,
              message: error.message,
              details: error.details
            })
            : error
        ),
        Effect.flatMap((archive) =>
          Effect.try({
            try: () => {
              const values = archive.entries.map((entry) => entry.tensor)
              try {
                const mapped = mapTensors(values)
                const metadata: Record<string, string> = Object.create(null)
                for (const [key, value] of Object.entries(archive.metadata)) {
                  if (!Predicate.isString(value)) throw new Error(`invalid safetensors metadata ${key}`)
                  metadata[key] = value
                }
                return {
                  entries: archive.entries.map((entry, index) => ({ name: entry.name, tensor: mapped[index]! })),
                  metadata: Object.freeze(metadata)
                }
              } catch (error) {
                clearBuffers(values)
                throw error
              }
            },
            catch: backendErrorFor("load", "io", "unsupported-placement")
          })
        )
      )
  }
  // N-API passes each GGUF metadata entry through one populated scalar or array
  // field because generated object declarations cannot express the native tagged
  // union.
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
      !Array.isArray(value.logicalShape) || !Array.isArray(value.physicalShape) ||
      !value.logicalShape.every((dimension) => Number.isSafeInteger(dimension) && dimension > 0) ||
      !value.physicalShape.every((dimension) => Number.isSafeInteger(dimension) && dimension > 0) ||
      (format !== "F32" && !validEncodedGeometry(value.logicalShape, {
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
  // Inspection returns validated metadata and logical and physical descriptors,
  // but no payloads. Loading streams supported F32 and K-quant payloads to Metal
  // shared storage. Quantized handles retain logical f32 metadata over packed u8
  // storage. Interrupted loads and mapping failures clear unpublished wrappers.
  // Successful mapping transfers them to caller-owned concrete handles.
  const gguf: Runtime.GgufRuntime = {
    inspect: (path) =>
      cancellableFor("inspectGguf", "io", (token) => native.inspectGguf(path, token), undefined, "io-failed").pipe(
        Effect.flatMap((inspection) =>
          Effect.try({
            try: () =>
              Object.freeze({
                metadata: Object.freeze(inspection.metadata.map(metadataValue)),
                tensors: Object.freeze(inspection.tensors.map(ggufDescriptor))
              }),
            catch: backendErrorFor("inspectGguf", "io", "io-failed")
          })
        )
      ),
    load: (path) =>
      cancellableFor(
        "loadGguf",
        "io",
        (token) => native.loadGgufForDevice(path, deviceOrdinal, token),
        (archive) => clearBuffers(archive.entries.map((entry) => entry.tensor)),
        "io-failed"
      ).pipe(
        Effect.flatMap((archive) =>
          Effect.try({
            try: () => {
              const values = archive.entries.map((entry) => entry.tensor)
              try {
                if (new Set(values).size !== values.length) {
                  throw new Error("native runtime returned duplicate tensor ownership")
                }
                const entries = archive.entries.map((entry) => {
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
                })
                return Object.freeze({ entries: Object.freeze(entries) })
              } catch (error) {
                clearBuffers(values)
                throw error
              }
            },
            catch: backendErrorFor("loadGguf", "io", "io-failed")
          })
        )
      )
  }
  const runtime: Runtime.RuntimeService = {
    identity: owner,
    backend: { name: backendName },
    placement,
    capabilities: {
      dtypes: ["f32", "f16", "bf16", "i64", "u8", "u32"],
      features: ["mixed-bf16"]
    },
    node,
    exposures: (root) =>
      Effect.try({
        try: () => {
          pendingStructure = undefined
          pendingDeclarations = undefined
          return nativeGraph(root, "exposures").exposures().map((entry) => ({
            name: entry.name,
            tensor: graph(entry.tensor)
          }))
        },
        catch: backendErrorFor("exposures", "graph")
      }),
    grad: (loss, wrt) =>
      Effect.try({
        try: () => {
          pendingStructure = undefined
          const declarations = tensorRecord(loss, "grad", "autodiff").declarations
          return native.grad(
            nativeGraph(loss, "grad", "autodiff"),
            wrt.map((target) => nativeGraph(target, "grad", "autodiff"))
          ).map((value) => {
            pendingDeclarations = declarations
            return graph(value)
          })
        },
        catch: (error) => {
          pendingDeclarations = undefined
          return backendErrorFor("grad", "autodiff")(error)
        }
      }),
    // Native compilation is synchronous and cannot be interrupted. It consumes
    // borrowed lazy roots, compile options, an optional decode specialization,
    // and the structural key. The returned immutable executable records logical
    // bindings and outputs, plus the completed state schema inferred by native
    // code.
    compile: (request) =>
      Effect.try({
        try: () => {
          if (!Array.isArray(request.roots) || request.roots.length === 0) {
            throw new Error("compile: expected at least one root")
          }
          const bindings = executableBindings(request.roots)
          const roots = request.roots.map((root) => nativeGraph(root, "compile", "compile"))
          const options = mapCompileOptions(request.options)
          const state = mapStateRequest(request.state)
          const value = native.compileForDevice(
            roots,
            options,
            state?.native,
            executableCacheKey(request),
            deviceOrdinal
          )
          const outputs = request.roots.flatMap((root, index) => {
            const base = {
              dtype: root.dtype,
              storage: root.storage
            }
            const selection = state?.request.outputSelections?.[index]
              ?? (state?.request.lastTokenRow === true ? "splitLastTokenRow" : "allRows")
            if (selection === "allRows") return [{ shape: root.shape, ...base }]
            if (selection === "batchedLastTokenRow") {
              return [{ shape: [state!.request.batch, root.shape[2]!], ...base }]
            }
            return Array.from({ length: state!.request.batch }, () => ({ shape: [root.shape[2]!], ...base }))
          })
          return executable(value, bindings, outputs, completeStateSchema(value, state?.request))
        },
        catch: backendErrorFor("compile", "compile", "compilation-failed")
      }),
    // Invocation borrows all inputs and state until the native promise settles.
    // On success, output wrappers transfer to independent handles owned by the
    // caller. Interruption or validation failure after native execution clears
    // every unpublished output. Runtime values are not part of this backend's
    // public contract.
    execute: (handle, invocation) =>
      cancellableFor(
        "execute",
        "execute",
        (token) => {
          const resolved = resolveExecutableInvocation(handle, invocation)
          return resolved.executable.execute(
            resolved.inputs,
            resolved.scalars,
            resolved.sequences,
            resolved.slots,
            resolved.activeMask,
            resolved.validLengths,
            resolved.advances,
            resolved.tokens,
            token
          )
        },
        clearBuffers
      ).pipe(
        Effect.flatMap((values) =>
          Effect.try({
            try: () => {
              try {
                return mapTensors(values, nativeExecutable(handle, "execute").info.outputs)
              } catch (error) {
                clearBuffers(values)
                throw error
              }
            },
            catch: backendErrorFor("execute", "execute")
          })
        )
      ),
    readback: (handle) =>
      cancellableFor(
        "readback",
        "readback",
        (token) => nativeTensor(handle, "readback", "readback").readback(token)
      ),
    // Deterministic release clears this concrete wrapper and invalidates its
    // public handle for other operations. Releasing it again is a no-op. Native
    // aliases, executable constants, in-flight work, and exported readback
    // buffers may still retain the underlying allocation.
    release: (handle) =>
      Effect.try({
        try: () => {
          const tensor = handleRecords.get(handle)
          if (tensor === undefined) {
            throw invalidHandle(
              "clear",
              "execute",
              backendHandles.has(handle)
                ? "foreign-handle"
                : "invalid-handle",
              "concrete-tensor"
            )
          }
          if (tensor.owner !== owner) {
            throw invalidHandle("clear", "execute", "foreign-handle", "concrete-tensor")
          }
          if (tensor.kind !== "concrete-tensor" || tensor.value === undefined) {
            throw invalidHandle("clear", "execute", "invalid-handle", "concrete-tensor")
          }
          if (tensor.disposed) return
          const value = tensor.value
          value.clear()
          tensor.disposed = true
        },
        catch: backendErrorFor("clear", "execute")
      }),
    extensions: {
      pathSafetensors,
      gguf,
      sampling,
      inference,
      decode,
      diagnostics: {
        // This is current native memory attributed to live NativeTensor wrappers.
        // Executable diagnostics instead contain static planned byte totals.
        externalMemoryBytes: Effect.sync(() => native.externalMemoryBytesForDevice(deviceOrdinal))
      }
    }
  }
  return runtime
}

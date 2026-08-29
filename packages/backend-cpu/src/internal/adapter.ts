/*
 * Adapts the backend-neutral RuntimeService contract to the private CPU napi-rs
 * addon. It keeps public handles opaque and checks their origin before accessing
 * native objects. It also records the logical input/output contract that the
 * addon exposes only implicitly and converts native exceptions to
 * Runtime.BackendError values. Graph construction, autodiff, and compilation are
 * synchronous. Execution and file/readback I/O use cancellable native promises.
 */
import { Runtime } from "@effect-torch/core"
import { Effect, Predicate } from "effect"
import { pipeArguments } from "effect/Pipeable"
import type {
  Executable as NativeExecutable,
  LazyTensor,
  NativeAddon,
  NativeCompileOptions,
  NativeCurrentBlockAttention,
  NativeDecodeOutputSelection,
  NativeDType,
  NativeGgufMetadataEntry,
  NativeGgufTensorDescriptor,
  NativeInferenceArtifact,
  NativeInferenceProposerPlan,
  NativeInferenceRoundResult,
  NativeInferenceSamplingOptions,
  NativeInferenceSamplingOverrides,
  NativeInferenceSequence,
  NativeInferenceSession,
  NativeKvPool,
  NativeKvSequence,
  NativeKvStateSchema,
  NativeTensor
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
 * Returns the JSON-safe canonical form used by structural executable cache keys.
 * It sorts object keys and converts byte arrays to numeric arrays. Non-finite
 * numbers and negative zero retain their exact IEEE-754 bits instead of
 * collapsing under `JSON.stringify`.
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

/** Serializes a value after {@link normalizedStructure} canonicalization. @internal */
// oxlint-disable-next-line anti-slop/no-unknown-parameters -- The cache-key boundary accepts any value handled by the normalizer.
export const structuralCacheKey = (value: unknown): string => JSON.stringify(normalizedStructure(value))

interface ExecutableInfo {
  readonly bindings: ReadonlyArray<TensorBinding>
  readonly outputs: ReadonlyArray<{
    readonly shape: ReadonlyArray<number>
    readonly dtype: Runtime.DType
    readonly storage?: Runtime.EncodedTensorStorage | undefined
  }>
  readonly state?: Runtime.DecodeStateSchema | undefined
}

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

interface InferenceArtifactInfo {
  readonly sampling: Runtime.InferenceSamplingOptions
}

interface InferenceSessionInfo {
  readonly artifact: Runtime.InferenceArtifactHandle
  readonly sequences: Map<bigint, Runtime.InferenceSequenceHandle>
}

interface InferenceSequenceInfo {
  readonly session: Runtime.InferenceSessionHandle
  readonly sequenceId: bigint
  readonly sampling: Runtime.InferenceSamplingOptions
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
  readonly executable: { readonly value: NativeExecutable; readonly info: ExecutableInfo }
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
// Each adapter module keeps its handle records private. The shared weak set lets
// independently loaded backend modules distinguish a foreign opaque handle from
// an arbitrary object without retaining either.
const backendHandlesKey = Symbol.for("@effect-torch/backend-handles")
// SAFETY: Backend adapters reserve this shared symbol for a WeakSet<object>.
const backendHandleRegistry = globalThis as typeof globalThis & BackendHandleRegistry
const existingBackendHandles = backendHandleRegistry[backendHandlesKey]
const backendHandles = existingBackendHandles ?? new WeakSet<object>()
if (existingBackendHandles === undefined) backendHandleRegistry[backendHandlesKey] = backendHandles

const backendName = "@effect-torch/backend-cpu"
const device = "cpu"
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

const description = "Native CPU"
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
      const nativeInferencePhase = message.match(
        /inference\[(compile|open|admission|prefill|proposer|verify|sample|accept|publish|finish|close|inspect)\]/
      )?.[1]
      const inferencePhase = nativeInferencePhase !== undefined && isInferenceFailurePhase(nativeInferencePhase)
        ? nativeInferencePhase
        : undefined
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
 * Native work may finish after the fiber is interrupted. A late result goes to
 * `onLateSuccess` so it can clear newly returned native tensors. Rejections
 * caused by cancellation remain fiber interruptions. Other exceptions enter the
 * typed backend error channel.
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
 * Creates a CPU RuntimeService for an already loaded addon namespace. The addon
 * object identifies the runtime and owns its handles. Services that use the same
 * namespace can exchange handles. Other backends and addon instances cannot.
 * Construction allocates only adapter metadata. Later operations create device
 * buffers and executable artifacts.
 *
 * @internal
 */
export const createRuntimeAdapter = (
  native: NativeAddon
): Runtime.RuntimeService => {
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
  const owner = native
  const placement: Runtime.Placement = Object.freeze({
    id: device,
    deviceType: device,
    description
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
  // `node` calls are synchronous, so these fields carry one request's JavaScript
  // structure and declarations through the native constructor into `lazyHandle`.
  // They are consumed or reset before `node` returns.
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
  // Native LazyTensor nodes contain the physical graph but not the shared public
  // slot namespace. The adapter propagates declarations in JavaScript so
  // compilation can reject gaps and conflicting tensor/scalar declarations
  // before splitting invocation bindings into native tensor and scalar arrays.
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
      if (ordered[index]![0] !== index) throw new Error(`compile: input slots must be contiguous from zero`)
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
  // Copy native diagnostics into recursively frozen public data. Artifact and
  // planner measurements are static. Compile phase timings come from the
  // artifact, so cache hits retain the timings from the original structural entry.
  const executableHandle = (
    value: NativeExecutable,
    bindings: ExecutableInfo["bindings"],
    outputs: ExecutableInfo["outputs"],
    state?: Runtime.DecodeStateSchema
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
  // Each element in a native result array owns its wrapper. Deduplication prevents
  // two public handles from claiming the same wrapper. Cleanup is best effort
  // because this path is already discarding the result.
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
  // Build a value-independent graph description for the addon's bounded
  // structural cache. Materialized leaves add signatures, not payloads. The
  // native cache revalidates generated bindings. With constantWeights, values
  // become executable constants, so native compilation skips cache reuse.
  // Gradients omit structure and return no cache key.
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
  // Map each discriminated public request to the corresponding native LazyTensor
  // constructor or method. Copy public arrays before they cross N-API, check the
  // origin of every input, and validate native metadata before assembling the
  // opaque handle.
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
              native.LazyTensor.constant(request.attributes.value, nativeDtype(request.attributes.dtype))
            )
          }
          case "zeros": {
            for (const exemplar of request.inputs) nativeGraph(exemplar, operation)
            return graph(
              native.LazyTensor.zeros([...request.attributes.shape], nativeDtype(request.attributes.dtype))
            )
          }
          case "ones": {
            for (const exemplar of request.inputs) nativeGraph(exemplar, operation)
            return graph(
              native.LazyTensor.ones([...request.attributes.shape], nativeDtype(request.attributes.dtype))
            )
          }
          case "full": {
            for (const exemplar of request.inputs) nativeGraph(exemplar, operation)
            return graph(
              native.LazyTensor.full(
                [...request.attributes.shape],
                request.attributes.value,
                nativeDtype(request.attributes.dtype)
              )
            )
          }
          case "randn":
            return graph(
              native.LazyTensor.randn([...request.attributes.shape], nativeDtype(request.attributes.dtype))
            )
          case "uniform":
            return graph(
              native.LazyTensor.uniform(
                [...request.attributes.shape],
                request.attributes.lo,
                request.attributes.hi,
                nativeDtype(request.attributes.dtype)
              )
            )
          case "arange":
            return graph(
              native.LazyTensor.arange(
                request.attributes.start,
                request.attributes.end,
                request.attributes.step,
                nativeDtype(request.attributes.dtype)
              )
            )
          case "eye":
            return graph(native.LazyTensor.eye(request.attributes.n, nativeDtype(request.attributes.dtype)))
          case "fromBytes":
            return graph(
              native.LazyTensor.fromBytes(
                request.attributes.data,
                [...request.attributes.shape],
                nativeDtype(request.attributes.dtype)
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
                nativeDtype(storage?.physicalDtype ?? request.attributes.dtype)
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
                nativeDtype(request.attributes.dtype)
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
  // Stateful calls mutably borrow distinct sequences from one compatible pool.
  // The adapter first checks the completed compile schema, token-row shape, and
  // non-windowed capacity. Native execution then stages transactional updates.
  const resolveExecutionState = (
    schema: Runtime.DecodeStateSchema,
    invocation: Runtime.ExecutionStateInvocation,
    operation: string
  ): ReadonlyArray<NativeKvSequence> => {
    const sequenceRecords = invocation.sequences.map((handle) => nativeSequence(handle, operation))
    const sequenceInfos = sequenceRecords.map((entry) => entry.info)
    const firstPool = sequenceInfos[0]?.pool
    if (
      firstPool === undefined ||
      sequenceInfos.some((entry) => entry.pool.key !== firstPool.key) ||
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
      firstPool.convKernel !== schema.convKernel ||
      sequenceRecords.length > schema.batch ||
      invocation.slots.length !== sequenceRecords.length ||
      invocation.tokens.length !== sequenceRecords.length ||
      invocation.activeMask.length !== schema.batch ||
      invocation.validLengths.length !== schema.batch ||
      invocation.advances.length !== schema.batch ||
      invocation.slots.some((slot) => !Number.isSafeInteger(slot) || slot < 0 || slot >= schema.batch) ||
      new Set(invocation.slots).size !== sequenceRecords.length ||
      invocation.activeMask.some((active, slot) => active !== invocation.slots.includes(slot)) ||
      invocation.validLengths.some((length, slot) =>
        !Number.isSafeInteger(length) || length < 0 ||
        (invocation.activeMask[slot] ? length === 0 : length !== 0)
      ) ||
      invocation.advances.some((advance, slot) => advance !== invocation.validLengths[slot]) ||
      invocation.tokens.some((row, index) => row.length !== invocation.advances[invocation.slots[index]!]!) ||
      new Set(invocation.sequences).size !== sequenceRecords.length
    ) {
      throw invalidHandle(operation, "execute", "invalid-handle", "kv-sequence")
    }
    if (
      invocation.tokens.some((row) => row.length === 0) ||
      invocation.tokens.some((row) =>
        row.some((token) => !Number.isSafeInteger(token) || token < 0 || token > 0xffff_ffff)
      )
    ) {
      throw new Error(`${operation}: invalid token rows for compiled state schema`)
    }
    if (
      schema.window === undefined &&
      sequenceRecords.some((entry, index) => entry.value.cursor + invocation.tokens[index]!.length > schema.maxTokens)
    ) {
      throw new Error(`${operation}: sequence context exceeds pool capacity ${schema.maxTokens}`)
    }
    return sequenceRecords.map((entry) => entry.value)
  }
  // Pools own fixed KV slabs, prefix-cache state, and recurrent geometry. Sequence
  // wrappers retain KV block references and per-sequence recurrent tensors.
  // releaseSequence invalidates the public sequence and returns its block
  // references. Native finalization releases the remaining sequence, pool, and
  // executable storage.
  const decode: Runtime.DecodeRuntime = {
    makePool: (options) =>
      Effect.try({
        try: () =>
          pool(
            new native.NativeKvPool(
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
              }
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
          const executableRecord = nativeExecutable(handle, "executeDecode")
          if (Object.keys(invocation.runtimeValues).length > 0) {
            throw new Runtime.BackendError({
              reason: "unsupported-operation",
              backend: backendName,
              operation: "executeDecode",
              phase: "execute",
              message: "executeDecode: CPU runtime values are not supported",
              details: { device }
            })
          }
          const value = executableRecord.value
          const info = executableRecord.info
          const schema = info.state
          if (schema === undefined) {
            throw new Error("executeDecode: requires a stateful executable")
          }
          const state = invocation.state
          if (state === undefined) {
            throw new Error("executeDecode: stateful executable requires state")
          }
          if (invocation.scalars.length > 0) {
            throw new Error("executeDecode: stateful executable does not accept scalar inputs")
          }
          if (invocation.bindings.length !== info.bindings.length) {
            throw new Error(
              `executeDecode: received ${invocation.bindings.length} tensor bindings, expected ${info.bindings.length}`
            )
          }
          const inputs = invocation.bindings.map((input, index) => nativeBinding(input, info.bindings[index]!, index))
          const sequences = resolveExecutionState(schema, state, "executeDecode")
          return value.executeSampled(
            inputs,
            [...sequences],
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
    executeSpeculative: () =>
      Effect.fail(
        new Runtime.BackendError({
          reason: "unsupported-operation",
          backend: backendName,
          operation: "executeSpeculative",
          phase: "execute",
          message: "executeSpeculative migrated to extensions.inference.runRound"
        })
      )
  }
  const inferenceSampling = (
    defaults: Runtime.InferenceSamplingOptions,
    override?: Runtime.InferenceSamplingOverrides
  ): NativeInferenceSamplingOptions => ({
    temperature: override?.temperature ?? defaults.temperature,
    topK: override?.topK ?? defaults.topK,
    topP: override?.topP ?? defaults.topP,
    seed: override?.seed ?? defaults.seed
  })
  const inferenceValueRef = (
    value: Runtime.InferenceValueRoute
  ): NativeInferenceProposerPlan["output"]["tokenIds"] => ({
    kind: value.kind,
    targetOutput: value.targetOutput,
    stage: value.stage,
    output: value.output,
    value: value.value === undefined
      ? undefined
      : { dtype: nativeDtype(value.value.dtype), shape: [...value.value.shape] },
    selectTargetRow: value.selectTargetRow
  })
  const inferenceProposerPlan = (plan: Runtime.InferenceProposerPlan): NativeInferenceProposerPlan => ({
    vocabulary: plan.vocabulary,
    tokenMapFingerprint: plan.tokenMapFingerprint,
    hiddenTaps: plan.hiddenTaps.map((tap) => ({
      name: tap.name,
      outputRoot: tap.outputRoot,
      value: { dtype: nativeDtype(tap.value.dtype), shape: [...tap.value.shape] }
    })),
    prefillHiddenTaps: plan.prefillHiddenTaps?.map((tap) => ({
      name: tap.name,
      outputRoot: tap.outputRoot,
      value: { dtype: nativeDtype(tap.value.dtype), shape: [...tap.value.shape] }
    })),
    verifyHiddenTaps: plan.verifyHiddenTaps?.map((tap) => ({
      name: tap.name,
      outputRoot: tap.outputRoot,
      value: { dtype: nativeDtype(tap.value.dtype), shape: [...tap.value.shape] }
    })),
    sharedTensors: plan.sharedTensors.map((tensor) => ({
      kind: tensor.kind,
      name: tensor.name,
      value: { dtype: nativeDtype(tensor.value.dtype), shape: [...tensor.value.shape] }
    })),
    stages: plan.stages.map((stage) => ({
      operationId: stage.operationId,
      layoutId: stage.layoutId,
      historyLookup: stage.historyLookup === undefined
        ? undefined
        : {
          id: stage.historyLookup.id,
          minMatchTokens: stage.historyLookup.minMatchTokens,
          maxMatchTokens: stage.historyLookup.maxMatchTokens
        },
      inputs: stage.inputs.map((input) => ({ slot: input.slot, value: inferenceValueRef(input.value) })),
      outputs: stage.outputs.map((output) => ({ dtype: nativeDtype(output.dtype), shape: [...output.shape] }))
    })),
    state: {
      kind: plan.state.kind,
      schemaId: plan.state.schemaId,
      commitKind: plan.state.commitKind,
      commitStages: [...plan.state.commitStages]
    },
    output: {
      topology: plan.output.topology,
      probabilities: plan.output.probabilities,
      tokenIds: inferenceValueRef(plan.output.tokenIds),
      probabilityRows: plan.output.probabilityRows === undefined
        ? undefined
        : inferenceValueRef(plan.output.probabilityRows),
      parents: plan.output.parents === undefined ? undefined : inferenceValueRef(plan.output.parents),
      confidence: plan.output.confidence === undefined ? undefined : inferenceValueRef(plan.output.confidence)
    },
    tokenMap: {
      kind: plan.tokenMap.kind,
      fingerprint: plan.tokenMap.fingerprint,
      proposerVocabulary: plan.tokenMap.proposerVocabulary,
      targetIds: plan.tokenMap.targetIds === undefined ? undefined : [...plan.tokenMap.targetIds]
    },
    trainedMaxRows: plan.trainedMaxRows
  })
  const nativeInferenceArtifact = (
    handle: Runtime.InferenceArtifactHandle,
    operation: string
  ): HandleRecord<"inference-artifact"> =>
    record(handle, "inference-artifact", operation, operation === "inferenceCompile" ? "compile" : "execute")
  const nativeInferenceSession = (
    handle: Runtime.InferenceSessionHandle,
    operation: string
  ): HandleRecord<"inference-session"> => record(handle, "inference-session", operation, "execute")
  const nativeInferenceSequence = (
    session: Runtime.InferenceSessionHandle,
    handle: Runtime.InferenceSequenceHandle,
    operation: string
  ): NativeInferenceSequence => {
    const found = record(handle, "inference-sequence", operation, "execute")
    if (found.info.session !== session) {
      throw invalidHandle(operation, "execute", "foreign-handle", "inference-sequence")
    }
    return found.value
  }
  const mapInferenceResult = (
    sessionHandle: Runtime.InferenceSessionHandle,
    result: NativeInferenceRoundResult,
    operation: string,
    expected: {
      readonly sequenceIds?: ReadonlyArray<bigint>
      readonly addSampling?: ReadonlyArray<Runtime.InferenceSamplingOptions>
    }
  ): Runtime.InferenceRoundResult => {
    const expectedCount = expected.sequenceIds?.length ?? expected.addSampling?.length
    if (
      !Predicate.isBigInt(result.roundId) || result.roundId < 0n || !Predicate.isBoolean(result.recovered) ||
      !Array.isArray(result.pages) || expectedCount === undefined || result.pages.length !== expectedCount
    ) {
      throw new Error(`${operation}: native inference returned a malformed receipt`)
    }
    const sessionRecord = nativeInferenceSession(sessionHandle, operation)
    const session = sessionRecord.value
    const info = sessionRecord.info
    const seen = new Set<bigint>()
    const stopReasons: Array<Runtime.InferenceTokenPage["stopReason"]> = []
    for (let index = 0; index < result.pages.length; index++) {
      const page = result.pages[index]!
      if (
        !Predicate.isBigInt(page.sequenceId) || page.sequenceId < 0n || seen.has(page.sequenceId) ||
        (expected.sequenceIds !== undefined && page.sequenceId !== expected.sequenceIds[index]) ||
        !Array.isArray(page.tokens) ||
        page.tokens.length === 0 ||
        page.tokens.some((token) => !Number.isInteger(token) || token < 0 || token > 0xffff_ffff) ||
        (page.stopReason !== undefined && page.stopReason !== "eos" && page.stopReason !== "maxTokens")
      ) throw new Error(`${operation}: native inference returned a malformed token page`)
      seen.add(page.sequenceId)
      stopReasons.push(page.stopReason)
    }
    if (
      expected.addSampling !== undefined &&
      result.pages.some((page, index) => index > 0 && page.sequenceId !== result.pages[0]!.sequenceId + BigInt(index))
    ) throw new Error(`${operation}: native inference returned token pages out of order`)
    const staged = result.pages.map((page, index) => {
      const existing = info.sequences.get(page.sequenceId)
      const sampling = expected.addSampling?.[index]
      if (expected.sequenceIds !== undefined) {
        if (existing === undefined) throw new Error(`${operation}: native inference returned an unknown sequence`)
        return { id: page.sequenceId, sequence: existing, fresh: false as const }
      }
      if (existing !== undefined) {
        const existingSampling = record(existing, "inference-sequence", operation, "execute").info
          .sampling
        if (
          !result.recovered || sampling === undefined || existingSampling.temperature !== sampling.temperature ||
          existingSampling.topK !== sampling.topK || existingSampling.topP !== sampling.topP ||
          existingSampling.seed !== sampling.seed
        ) throw new Error(`${operation}: native inference returned an existing sequence`)
        return { id: page.sequenceId, sequence: existing, fresh: false as const }
      }
      if (sampling === undefined) throw new Error(`${operation}: native inference returned an unknown sequence`)
      return {
        id: page.sequenceId,
        sequence: wrapOpaque(
          "inference-sequence",
          session.sequence(page.sequenceId),
          {
            session: sessionHandle,
            sequenceId: page.sequenceId,
            sampling
          } satisfies InferenceSequenceInfo
        ),
        fresh: true as const
      }
    })
    for (const entry of staged) {
      if (entry.fresh) info.sequences.set(entry.id, entry.sequence)
    }
    const pages = result.pages.map((page, index): Runtime.InferenceTokenPage => {
      const stopReason = stopReasons[index]
      return Object.freeze({
        sequence: staged[index]!.sequence,
        sequenceId: page.sequenceId,
        tokens: Object.freeze([...page.tokens]),
        stopReason
      })
    })
    return Object.freeze({ roundId: result.roundId, recovered: result.recovered, pages: Object.freeze(pages) })
  }
  const inferenceRound = (
    sessionHandle: Runtime.InferenceSessionHandle,
    operation: string,
    register: (session: NativeInferenceSession, token: CancellationToken) => Promise<NativeInferenceRoundResult>,
    expected: Parameters<typeof mapInferenceResult>[3]
  ): Effect.Effect<Runtime.InferenceRoundResult, Runtime.BackendError> =>
    cancellableFor(
      operation,
      "execute",
      (token) => register(nativeInferenceSession(sessionHandle, operation).value, token)
    ).pipe(
      Effect.flatMap((result) =>
        Effect.try({
          try: () => mapInferenceResult(sessionHandle, result, operation, expected),
          catch: backendErrorFor(operation, "execute")
        })
      )
    )
  const inference: Runtime.InferenceRuntime = {
    compile: (request) =>
      Effect.try({
        try: () => {
          const generalized = request.generalizedProposer
          // CPU serves every prompt from the largest compiled prefill chunk.
          const targetPrefill = nativeExecutable(
            request.target.prefill[request.target.prefill.length - 1]!,
            "inferenceCompile"
          ).value
          const targetDecode = nativeExecutable(request.target.decode, "inferenceCompile").value
          // CPU verifies at the widest compiled width as the correctness
          // reference. Only Metal uses adaptive widths to improve throughput.
          const verifyHandles = request.target.verify ?? []
          const targetVerify = verifyHandles.length === 0
            ? undefined
            : nativeExecutable(verifyHandles[verifyHandles.length - 1]!, "inferenceCompile")
              .value
          const targetPool = nativePool(request.target.pool, "inferenceCompile").value
          const proposerPrefill = request.proposer === undefined
            ? undefined
            : nativeExecutable(request.proposer.prefill, "inferenceCompile").value
          const proposerDecode = request.proposer === undefined
            ? undefined
            : nativeExecutable(request.proposer.decode, "inferenceCompile").value
          const proposerPool = request.proposer === undefined
            ? undefined
            : nativePool(request.proposer.pool, "inferenceCompile").value
          const replay = generalized?.replay
          const value = native.compileInference(
            targetPrefill,
            targetDecode,
            targetVerify,
            targetPool,
            proposerPrefill,
            proposerDecode,
            proposerPool,
            request.proposer?.maxDraftTokens ?? generalized?.maxDraftTokens ?? 0,
            request.batchSize,
            nativeDtype(request.tokenDtype),
            inferenceSampling(request.sampling),
            generalized === undefined ? undefined : inferenceProposerPlan(generalized.plan),
            generalized?.sharedTensors.map((tensor) => nativeTensor(tensor, "inferenceCompile")),
            generalized?.stageExecutables.map((stage) => nativeExecutable(stage, "inferenceCompile").value),
            replay === undefined
              ? undefined
              : nativeExecutable(replay.prefill[replay.prefill.length - 1]!, "inferenceCompile")
                .value,
            replay === undefined
              ? undefined
              : nativeExecutable(replay.decode, "inferenceCompile").value,
            replay === undefined
              ? undefined
              // CPU replays at the widest compiled verify width.
              : nativeExecutable(replay.verify[replay.verify.length - 1]!, "inferenceCompile")
                .value,
            replay === undefined ? undefined : nativePool(replay.pool, "inferenceCompile").value
          )
          return wrapOpaque(
            "inference-artifact",
            value,
            {
              sampling: Object.freeze({ ...request.sampling })
            } satisfies InferenceArtifactInfo
          )
        },
        catch: backendErrorFor("inferenceCompile", "compile", "compilation-failed")
      }),
    open: (artifact) =>
      Effect.try({
        try: () => {
          const artifactRecord = nativeInferenceArtifact(artifact, "inferenceOpen")
          return wrapOpaque(
            "inference-session",
            artifactRecord.value.open(),
            { artifact, sequences: new Map() } satisfies InferenceSessionInfo
          )
        },
        catch: backendErrorFor("inferenceOpen", "execute")
      }),
    add: (sessionHandle, request) => {
      const sessionInfo = nativeInferenceSession(sessionHandle, "inferenceAdd").info
      const artifactInfo = nativeInferenceArtifact(sessionInfo.artifact, "inferenceAdd").info
      const sampling = request.entries.map((entry) => inferenceSampling(artifactInfo.sampling, entry.sampling))
      return inferenceRound(sessionHandle, "inferenceAdd", (session, token) =>
        session.add(
          request.entries.map((entry) => nativeTensor(entry.prompt, "inferenceAdd")),
          sampling,
          request.entries.map((entry) => entry.maxTokens ?? 0),
          request.entries.map((entry) => [...entry.eosTokens]),
          token
        ), { addSampling: sampling.map((options) => Object.freeze({ ...options })) })
    },
    runRound: (sessionHandle, request) => {
      const sessionInfo = nativeInferenceSession(sessionHandle, "inferenceRound").info
      nativeInferenceArtifact(sessionInfo.artifact, "inferenceRound")
      const sequenceInfos = request.entries.map((entry) => {
        nativeInferenceSequence(sessionHandle, entry.sequence, "inferenceRound")
        return record(entry.sequence, "inference-sequence", "inferenceRound", "execute").info
      })
      return inferenceRound(sessionHandle, "inferenceRound", (session, token) =>
        session.runRound(
          request.entries.map((entry) => nativeInferenceSequence(sessionHandle, entry.sequence, "inferenceRound")),
          request.entries.map((entry) => ({ ...entry.sampling } satisfies NativeInferenceSamplingOverrides)),
          token
        ), { sequenceIds: sequenceInfos.map((info) => info.sequenceId) })
    },
    acknowledge: (sessionHandle, roundId) =>
      Effect.try({
        try: () => {
          if (!Predicate.isBigInt(roundId) || roundId < 0n) {
            throw new Error("inferenceAcknowledge: roundId must be an unsigned bigint")
          }
          const session = nativeInferenceSession(sessionHandle, "inferenceAcknowledge").value
          session.acknowledge(roundId)
        },
        catch: backendErrorFor("inferenceAcknowledge", "execute")
      }),
    finish: (sessionHandle, sequences) =>
      Effect.try({
        try: () => {
          const sessionRecord = nativeInferenceSession(sessionHandle, "inferenceFinish")
          sessionRecord.value.finish(
            sequences.map((sequence) => nativeInferenceSequence(sessionHandle, sequence, "inferenceFinish"))
          )
          const info = sessionRecord.info
          for (const sequence of sequences) {
            const found = record(sequence, "inference-sequence", "inferenceFinish", "execute")
            found.disposed = true
            info.sequences.delete(found.info.sequenceId)
          }
        },
        catch: backendErrorFor("inferenceFinish", "execute")
      }),
    inspect: (sessionHandle, sequence) =>
      Effect.try({
        try: () => {
          const session = nativeInferenceSession(sessionHandle, "inferenceInspect").value
          const inspection = session.inspect(nativeInferenceSequence(sessionHandle, sequence, "inferenceInspect"))
          if (
            !Predicate.isBigInt(inspection.sequenceId) || !Predicate.isBigInt(inspection.cursor) ||
            (inspection.terminal !== undefined && inspection.terminal !== "eos" &&
              inspection.terminal !== "maxTokens")
          ) {
            throw new Error("inferenceInspect: native inference returned malformed inspection")
          }
          return Object.freeze({
            sequenceId: inspection.sequenceId,
            cursor: inspection.cursor,
            terminal: inspection.terminal
          })
        },
        catch: backendErrorFor("inferenceInspect", "execute")
      }),
    close: (sessionHandle) =>
      Effect.try({
        try: () => {
          const existing = handleRecords.get(sessionHandle)
          if (
            existing?.kind === "inference-session" && existing.owner === owner && existing.disposed
          ) return
          const sessionRecord = nativeInferenceSession(sessionHandle, "inferenceClose")
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
    diagnostics: (artifact) =>
      Effect.try({
        try: () => {
          const value = nativeInferenceArtifact(artifact, "inferenceDiagnostics").value
            .diagnostics()
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
  // Saving borrows tensors. Successful loads transfer newly created native
  // tensors to caller-owned concrete handles. Safetensors cannot represent the
  // adapter's logical-f32/packed-u8 GGML storage, so the adapter rejects encoded
  // handles instead of writing misleading u8 data.
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
        (token) => native.loadTensors(path, token),
        (archive) => clearBuffers(archive.entries.map((entry) => entry.tensor))
      ).pipe(
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
  // Each GGUF metadata entry crosses N-API with one scalar or array field
  // populated. Generated object declarations cannot express the native tagged
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
  // Inspection returns validated metadata and logical/physical descriptors
  // without payloads. Loading preserves supported K-quant payloads as encoded u8
  // tensors with logical f32 metadata. If native loading is interrupted or
  // mapping fails, the adapter clears unpublished wrappers. Successful mapping
  // transfers them to caller-owned concrete handles.
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
        (token) => native.loadGguf(path, token),
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
      dtypes: ["f32", "f64", "f16", "bf16", "i64", "u8", "u32"],
      features: []
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
    // Native compilation is synchronous and cannot be interrupted. It takes
    // borrowed lazy roots, compile options, an optional decode specialization,
    // and the structural key. The returned immutable executable records logical
    // bindings and outputs plus the state schema completed by native compilation.
    compile: (request) =>
      Effect.try({
        try: () => {
          const bindings = executableBindings(request.roots)
          const roots = request.roots.map((root) => nativeGraph(root, "compile", "compile"))
          const options: NativeCompileOptions | undefined = request.options === undefined
            ? undefined
            : {
              optimize: request.options.optimize,
              constantWeights: request.options.constantWeights
            }
          // SAFETY: Decode-state unions match the generated napi-rs string enums.
          const state: NativeKvStateSchema | undefined = request.state === undefined
            ? undefined
            : {
              maxTokens: request.state.maxTokens,
              blockSize: request.state.blockSize,
              kvDtype: nativeDtype(request.state.kvDtype),
              window: request.state.window,
              currentBlockAttention: request.state.currentBlockAttention as NativeCurrentBlockAttention | undefined,
              batch: request.state.batch,
              packedCausalChains: request.state.packedCausalChains === undefined
                ? undefined
                : { rowsPerSequence: request.state.packedCausalChains.rowsPerSequence },
              lastTokenRow: request.state.lastTokenRow,
              outputSelections: request.state.outputSelections?.map((selection) =>
                selection === "allRows"
                  ? "AllRows" as NativeDecodeOutputSelection
                  : selection === "splitLastTokenRow"
                  ? "SplitLastTokenRow" as NativeDecodeOutputSelection
                  : "BatchedLastTokenRow" as NativeDecodeOutputSelection
              )
            }
          const value = native.compile(roots, options, state, executableCacheKey(request))
          const outputs = request.roots.flatMap((root, index) => {
            const base = {
              dtype: root.dtype,
              storage: root.storage
            }
            const selection = request.state?.outputSelections?.[index]
              ?? (request.state?.lastTokenRow === true ? "splitLastTokenRow" : "allRows")
            if (selection === "allRows") return [{ shape: root.shape, ...base }]
            if (selection === "batchedLastTokenRow") {
              return [{ shape: [request.state!.batch, root.shape[2]!], ...base }]
            }
            return Array.from({ length: request.state!.batch }, () => ({ shape: [root.shape[2]!], ...base }))
          })
          if (value.stateful !== (request.state !== undefined)) {
            throw new Error("compile: native executable state does not match the request")
          }
          if (request.state === undefined) return executableHandle(value, bindings, outputs)
          if (value.batch !== request.state.batch) {
            throw new Error(
              `compile: native batch ${value.batch} does not match requested batch ${request.state.batch}`
            )
          }
          if (!Predicate.isBoolean(value.allowsWindowEviction)) {
            throw new Error("compile: native executable returned an invalid window eviction policy")
          }
          const geometry = {
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
          for (const [name, dimension] of Object.entries(geometry)) {
            if (!Number.isSafeInteger(dimension) || dimension < 0) {
              throw new Error(`compile: native executable returned invalid ${name} ${dimension}`)
            }
          }
          const schema: Runtime.DecodeStateSchema = Object.freeze({
            maxTokens: request.state.maxTokens,
            blockSize: request.state.blockSize,
            kvDtype: request.state.kvDtype,
            window: request.state.window === undefined || !value.allowsWindowEviction
              ? undefined
              : request.state.window,
            currentBlockAttention: request.state.currentBlockAttention,
            lastTokenRow: request.state.lastTokenRow,
            outputSelections: request.state.outputSelections === undefined
              ? undefined
              : Object.freeze([...request.state.outputSelections]),
            packedCausalChains: request.state.packedCausalChains === undefined
              ? undefined
              : Object.freeze({
                rowsPerSequence: request.state.packedCausalChains.rowsPerSequence
              }),
            batch: request.state.batch,
            ...geometry
          })
          return executableHandle(value, bindings, outputs, schema)
        },
        catch: backendErrorFor("compile", "compile", "compilation-failed")
      }),
    // Invocation borrows all inputs and state until the native promise settles.
    // On success, output wrappers transfer to independent caller-owned handles.
    // Interruption or post-native validation failure clears every unpublished
    // output. Runtime values are not part of this backend's public contract.
    execute: (handle, invocation) =>
      cancellableFor(
        "execute",
        "execute",
        (token) => {
          const executableRecord = nativeExecutable(handle, "execute")
          if (Object.keys(invocation.runtimeValues).length > 0) {
            throw new Runtime.BackendError({
              reason: "unsupported-operation",
              backend: backendName,
              operation: "execute",
              phase: "execute",
              message: "execute: CPU runtime values are not supported",
              details: { device }
            })
          }
          const value = executableRecord.value
          const info = executableRecord.info
          if (invocation.bindings.length !== info.bindings.length) {
            throw new Error(
              `execute: received ${invocation.bindings.length} tensor bindings, expected ${info.bindings.length}`
            )
          }
          const inputs = invocation.bindings.map((input, index) => nativeBinding(input, info.bindings[index]!, index))
          const scalars = [...invocation.scalars]
          const schema = info.state
          if (schema === undefined) {
            if (invocation.state !== undefined) {
              throw new Error("execute: stateless executable does not accept state")
            }
            return value.execute(
              inputs,
              scalars,
              undefined,
              undefined,
              undefined,
              undefined,
              undefined,
              undefined,
              token
            )
          }
          if (invocation.state === undefined) {
            throw new Error("execute: stateful executable requires state")
          }
          if (scalars.length > 0) {
            throw new Error("execute: stateful executable does not accept scalar inputs")
          }
          const sequences = resolveExecutionState(schema, invocation.state, "execute")
          return value.execute(
            inputs,
            scalars,
            [...sequences],
            [...invocation.state.slots],
            [...invocation.state.activeMask],
            [...invocation.state.validLengths],
            [...invocation.state.advances],
            invocation.state.tokens.map((row) => [...row]),
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
    // Release clears this concrete wrapper and invalidates its public handle for
    // other operations. A second release is a no-op. Native aliases, executable
    // constants, in-flight work, or exported readback buffers may still retain
    // the underlying allocation.
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
      decode,
      inference,
      diagnostics: {
        // This is current native memory attributed to live NativeTensor wrappers.
        // Executable diagnostics instead contain static planned byte totals.
        externalMemoryBytes: Effect.sync(() => native.externalMemoryBytes())
      }
    }
  }
  return runtime
}

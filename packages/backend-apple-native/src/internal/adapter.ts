/*
 * Translation boundary between the backend-neutral RuntimeService contract and
 * the private Apple Metal napi-rs addon. The adapter keeps all public handles
 * opaque, validates their provenance before unwrapping native objects, records
 * the logical input/output contract that the addon exposes only implicitly, and
 * turns native exceptions into Runtime.BackendError values. Graph construction,
 * autodiff, and compilation are synchronous; execution and file/readback I/O
 * use cancellable native promises.
 */
import { Runtime } from "@effect-torch/core"
import { Effect } from "effect"
import { pipeArguments } from "effect/Pipeable"
import type {
  Executable,
  LazyTensor,
  NativeAddon,
  NativeCompileOptions,
  NativeDType,
  NativeGgmlKQuant,
  NativeGgufMetadataEntry,
  NativeGgufTensorDescriptor,
  NativeKvPool,
  NativeKvSequence,
  NativeKvStateSchema,
  NativeTensor
} from "./native-addon.js"

type CancellationToken = InstanceType<NativeAddon["CancellationToken"]>
type HandleKind = "lazy-tensor" | "concrete-tensor" | "executable" | "kv-pool" | "kv-sequence"

interface HandleRecord {
  readonly owner: object
  readonly kind: HandleKind
  readonly graph?: LazyTensor
  readonly value?: object
  readonly info?: unknown
  readonly structure?: StructuralNode
  readonly declarations?: ReadonlySet<InputDeclaration>
  disposed: boolean
}

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
  readonly storage?: Runtime.EncodedTensorStorage
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
 * Produces the JSON-safe canonical form used by the structural executable
 * cache key. Object keys are sorted, byte arrays become numeric arrays, and
 * non-finite numbers plus negative zero retain their exact IEEE-754 bits rather
 * than collapsing under `JSON.stringify`.
 *
 * @internal
 */
export const normalizedStructure = (value: unknown): unknown => {
  if (typeof value === "number") {
    if (Number.isFinite(value) && !Object.is(value, -0)) return value
    numberBits.setFloat64(0, value, false)
    const high = numberBits.getUint32(0, false).toString(16).padStart(8, "0")
    const low = numberBits.getUint32(4, false).toString(16).padStart(8, "0")
    return { $number: `${high}${low}` }
  }
  if (value instanceof Uint8Array) return Array.from(value)
  if (Array.isArray(value)) return value.map(normalizedStructure)
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, entry]) => [key, normalizedStructure(entry)])
    )
  }
  return value
}

/** Serializes a value after {@link normalizedStructure} canonicalization. @internal */
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
    readonly storage?: Runtime.EncodedTensorStorage
  }>
  readonly state?: Runtime.DecodeStateSchema
}

const handleRecords = new WeakMap<object, HandleRecord>()
// Every adapter module keeps its records private. The shared weak set only lets
// independently loaded backend modules distinguish a foreign opaque handle from
// an arbitrary object without retaining either one.
const backendHandlesKey = Symbol.for("@effect-torch/backend-handles")
const existingBackendHandles = Reflect.get(globalThis, backendHandlesKey) as WeakSet<object> | undefined
const backendHandles = existingBackendHandles ?? new WeakSet<object>()
if (existingBackendHandles === undefined) Reflect.set(globalThis, backendHandlesKey, backendHandles)

const backendName = "@effect-torch/backend-apple-native"
const device = "metal"
const description = "Apple Metal"

const backendError = (
  operation: string,
  phase: Runtime.BackendError["phase"],
  reason: Runtime.BackendError["reason"] = "execution-failed"
) =>
(error: unknown): Runtime.BackendError =>
  error instanceof Runtime.BackendError
    ? error
    : (() => {
      const message = error instanceof Error ? error.message : String(error)
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
        details: { device, error }
      })
    })()

/**
 * Bridges Effect interruption to the addon's cooperative cancellation token.
 * Native work is not assumed to stop immediately: a result that wins after the
 * fiber is interrupted is passed to `onLateSuccess` so newly returned native
 * tensors can be cleared. Rejections caused by cancellation remain fiber
 * interruption; other exceptions enter the typed backend error channel.
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
        onLateSuccess?.(lateValue as A)
      } catch {
        // The interrupted fiber cannot observe cleanup failures.
      }
      lateValue = undefined
    }
    const abort = () => {
      token.cancel()
      clearLateValue()
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
        if (signal.aborted) {
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
 * Constructs one Metal RuntimeService around an already loaded addon namespace.
 * The addon object is both runtime identity and the ownership key, so services
 * made around the same namespace can exchange handles while another backend or
 * another addon instance cannot. Construction allocates only adapter metadata;
 * the process-wide Metal device is initialized lazily by native operations.
 *
 * @internal
 */
export const makeRuntime = (
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
  const record = (
    handle: object,
    kind: HandleKind,
    operation: string,
    phase: Runtime.BackendError["phase"]
  ): HandleRecord => {
    const found = typeof handle === "object" && handle !== null ? handleRecords.get(handle) : undefined
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
    return found
  }
  const tensorRecord = (
    handle: Runtime.TensorHandle,
    operation: string,
    phase: Runtime.BackendError["phase"]
  ): HandleRecord => {
    const found = typeof handle === "object" && handle !== null ? handleRecords.get(handle) : undefined
    if (
      found === undefined || (found.kind !== "lazy-tensor" && found.kind !== "concrete-tensor") ||
      found.graph === undefined
    ) {
      throw invalidHandle(
        operation,
        phase,
        typeof handle === "object" && handle !== null && backendHandles.has(handle)
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
  const wrapOpaque = <H extends object>(kind: HandleKind, value: object, info?: unknown): H => {
    const handle = Object.freeze({}) as H
    handleRecords.set(handle, { owner, kind, value, info, disposed: false })
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
    return Object.freeze({
      _tag: tag,
      shape: Object.freeze([...shape]),
      dtype: dtype(tensorDtype),
      ...(storage === undefined
        ? {}
        : {
          storage: Object.freeze({
            encoding: storage.encoding,
            physicalShape: Object.freeze([...storage.physicalShape]),
            physicalDtype: "u8" as const
          })
        }),
      device: tensorDevice,
      placement,
      pipe(this: Runtime.TensorHandle) {
        return pipeArguments(this, arguments)
      }
    }) as unknown as H
  }
  // `node` calls are synchronous, so these fields safely carry one request's
  // JavaScript-only structure and declarations through the native constructor
  // into `lazyHandle`. They are always consumed or reset before control returns.
  let pendingStructure: StructuralNode | undefined
  let pendingDeclarations: ReadonlySet<InputDeclaration> | undefined
  const lazyHandle = (
    value: LazyTensor,
    logical?: {
      readonly shape: ReadonlyArray<number>
      readonly dtype: Runtime.DType
      readonly storage?: Runtime.EncodedTensorStorage
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
      ...(pendingStructure === undefined ? {} : { structure: pendingStructure }),
      ...(pendingDeclarations === undefined ? {} : { declarations: pendingDeclarations }),
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
      readonly storage?: Runtime.EncodedTensorStorage
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
    return found.value as NativeTensor
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
  // Native LazyTensor nodes know their physical graph but do not expose the
  // shared public slot namespace. Propagating declarations in JavaScript lets
  // compilation reject gaps and conflicting tensor/scalar declarations before
  // invocation bindings are separated into native tensor and scalar arrays.
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
        ...(request.attributes.storage === undefined ? {} : { storage: request.attributes.storage })
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
          ...(declaration.storage === undefined
            ? {}
            : {
              storage: Object.freeze({
                encoding: declaration.storage.encoding,
                physicalShape: Object.freeze([...declaration.storage.physicalShape]),
                physicalDtype: declaration.storage.physicalDtype
              })
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
    return found.value as NativeTensor
  }
  // Snapshot native diagnostics into recursively frozen public data. These are
  // static artifact/planner measurements; compile phase timings come from the
  // artifact and therefore remain those of the original structural-cache entry.
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
    const handle = Object.freeze(
      state === undefined ? { diagnostics } : { state, diagnostics }
    ) as unknown as Runtime.ExecutableHandle
    handleRecords.set(handle, {
      owner,
      kind: "executable",
      value,
      info: {
        bindings,
        outputs,
        ...(state === undefined ? {} : { state })
      } satisfies ExecutableInfo,
      disposed: false
    })
    backendHandles.add(handle)
    return handle
  }
  const nativeExecutable = (handle: Runtime.ExecutableHandle, operation: string): HandleRecord =>
    record(handle, "executable", operation, "execute")
  const pool = (value: NativeKvPool, info: Omit<KvPoolInfo, "key">): Runtime.KvPoolHandle =>
    wrapOpaque<Runtime.KvPoolHandle>("kv-pool", value, { ...info, key: value } satisfies KvPoolInfo)
  const nativePool = (handle: Runtime.KvPoolHandle, operation: string): HandleRecord =>
    record(handle, "kv-pool", operation, "execute")
  const sequence = (value: NativeKvSequence, pool: KvPoolInfo): Runtime.KvSequenceHandle =>
    wrapOpaque<Runtime.KvSequenceHandle>("kv-sequence", value, { pool } satisfies KvSequenceInfo)
  const nativeSequence = (handle: Runtime.KvSequenceHandle, operation: string): HandleRecord =>
    record(handle, "kv-sequence", operation, "execute")
  // Native result arrays transfer one owning wrapper per element. Deduplication
  // prevents two public handles from claiming the same wrapper, and cleanup is
  // deliberately best-effort because this path is already discarding a result.
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
  // Reconstruct a value-independent graph description for the addon's bounded
  // structural cache. Materialized leaves contribute signatures, not payloads;
  // the native cache revalidates generated bindings, and constantWeights makes
  // native compilation bypass cache reuse because values become executable
  // constants. Gradients omit structure and therefore intentionally return no
  // cache key.
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
  // Translate each discriminated public request to the corresponding native
  // LazyTensor constructor or method. Public arrays are copied before crossing
  // N-API, every input is provenance-checked, and native metadata is validated
  // while the resulting opaque handle is assembled.
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
              native.LazyTensor.constant(request.attributes.value, request.attributes.dtype as NativeDType)
            )
          }
          case "zeros": {
            for (const exemplar of request.inputs) nativeGraph(exemplar, operation)
            return graph(
              native.LazyTensor.zeros([...request.attributes.shape], request.attributes.dtype as NativeDType)
            )
          }
          case "ones": {
            for (const exemplar of request.inputs) nativeGraph(exemplar, operation)
            return graph(
              native.LazyTensor.ones([...request.attributes.shape], request.attributes.dtype as NativeDType)
            )
          }
          case "full": {
            for (const exemplar of request.inputs) nativeGraph(exemplar, operation)
            return graph(
              native.LazyTensor.full(
                [...request.attributes.shape],
                request.attributes.value,
                request.attributes.dtype as NativeDType
              )
            )
          }
          case "randn":
            return graph(
              native.LazyTensor.randn([...request.attributes.shape], request.attributes.dtype as NativeDType)
            )
          case "uniform":
            return graph(
              native.LazyTensor.uniform(
                [...request.attributes.shape],
                request.attributes.lo,
                request.attributes.hi,
                request.attributes.dtype as NativeDType
              )
            )
          case "arange":
            return graph(
              native.LazyTensor.arange(
                request.attributes.start,
                request.attributes.end,
                request.attributes.step,
                request.attributes.dtype as NativeDType
              )
            )
          case "eye":
            return graph(native.LazyTensor.eye(request.attributes.n, request.attributes.dtype as NativeDType))
          case "fromBytes":
            return graph(
              native.LazyTensor.fromBytes(
                request.attributes.data,
                [...request.attributes.shape],
                request.attributes.dtype as NativeDType
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
                (storage?.physicalDtype ?? request.attributes.dtype) as NativeDType
              ),
              {
                shape: request.attributes.shape,
                dtype: request.attributes.dtype,
                ...(storage === undefined ? {} : { storage })
              }
            )
          }
          case "scalarInput":
            return graph(
              native.LazyTensor.scalarInput(
                request.attributes.slot,
                request.attributes.dtype as NativeDType
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
          case "checkpoint":
            return graph(nativeGraph(request.inputs[0], operation).checkpoint())
          case "gelu":
            return graph(nativeGraph(request.inputs[0], operation).gelu(request.attributes.approximate))
          case "pow":
            return graph(nativeGraph(request.inputs[0], operation).pow(request.attributes.exponent))
          case "cast":
            return graph(nativeGraph(request.inputs[0], operation).cast(request.attributes.dtype as NativeDType))
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
                request.attributes.encoding as NativeGgmlKQuant,
                request.attributes.logicalShape[0],
                request.attributes.logicalShape[1]
              )
            )
          case "quantizedEmbedding":
            return graph(
              nativeGraph(request.inputs[0], operation).quantizedEmbedding(
                nativeGraph(request.inputs[1], operation),
                request.attributes.encoding as NativeGgmlKQuant,
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
        throw new Runtime.BackendError({
          reason: "unsupported-operation",
          backend: backendName,
          operation: String((unhandled as { readonly op?: unknown }).op),
          phase: "graph",
          message: `unsupported graph operation ${String((unhandled as { readonly op?: unknown }).op)}`,
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
      ...(options.optimize === undefined ? {} : { optimize: options.optimize }),
      ...(options.constantWeights === undefined ? {} : { constantWeights: options.constantWeights })
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
      ...(state.window === undefined ? {} : { window: uint32(state.window, "state.window", true) }),
      batch: uint32(state.batch, "state.batch", false),
      ...(state.lastTokenRow === undefined ? {} : { lastTokenRow: state.lastTokenRow })
    }
    return {
      request,
      native: {
        maxTokens: request.maxTokens,
        blockSize: request.blockSize,
        kvDtype: request.kvDtype as NativeDType,
        ...(request.window === undefined ? {} : { window: request.window }),
        batch: request.batch,
        ...(request.lastTokenRow === undefined ? {} : { lastTokenRow: request.lastTokenRow })
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
    if (typeof value.allowsWindowEviction !== "boolean") {
      throw new Error("compile: native executable returned an invalid window eviction policy")
    }
    return Object.freeze({
      maxTokens: state.maxTokens,
      blockSize: state.blockSize,
      kvDtype: state.kvDtype,
      ...(state.window === undefined || !value.allowsWindowEviction ? {} : { window: state.window }),
      ...(state.lastTokenRow === undefined ? {} : { lastTokenRow: state.lastTokenRow }),
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
  // Stateful invocation mutably borrows distinct sequences from one compatible
  // pool. The adapter checks the completed compile schema, token-row shape, and
  // non-windowed capacity before native execution stages transactional updates.
  const resolveExecutionState = (
    schema: Runtime.DecodeStateSchema | undefined,
    invocation: Runtime.ExecutionStateInvocation | undefined
  ):
    | readonly [sequences: Array<NativeKvSequence>, tokens: Array<Array<number>>]
    | readonly [sequences: undefined, tokens: undefined] =>
  {
    if (schema === undefined) {
      if (invocation !== undefined) {
        throw executionError("execute: stateless executable does not accept state")
      }
      return [undefined, undefined]
    }
    if (invocation === undefined) {
      throw executionError("execute: stateful executable requires state")
    }
    if (
      invocation.sequences.length === 0 ||
      invocation.sequences.length > schema.batch ||
      invocation.tokens.length !== invocation.sequences.length
    ) {
      throw executionError(
        `execute: expected 1..=${schema.batch} sequences with one token row each`
      )
    }
    const sequenceRecords = invocation.sequences.map((handle) => nativeSequence(handle, "execute"))
    const sequenceInfos = sequenceRecords.map((entry) => entry.info as KvSequenceInfo)
    const firstPool = sequenceInfos[0]!.pool
    if (
      sequenceInfos.some((entry) => entry.pool.key !== firstPool.key) ||
      new Set(invocation.sequences).size !== sequenceRecords.length ||
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
    const advance = invocation.tokens[0]!.length
    if (
      advance === 0 ||
      invocation.tokens.some((row) =>
        row.length !== advance || row.some((token) => !Number.isSafeInteger(token) || token < 0 || token > 0xffff_ffff)
      )
    ) {
      throw executionError("execute: invalid token rows for compiled state schema")
    }
    if (
      schema.window === undefined &&
      sequenceRecords.some((entry) => (entry.value as NativeKvSequence).cursor + advance > schema.maxTokens)
    ) {
      throw executionError(`execute: sequence context exceeds pool capacity ${schema.maxTokens}`)
    }
    return [
      sequenceRecords.map((entry) => entry.value as NativeKvSequence),
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
    const info = executableRecord.info as ExecutableInfo
    if (invocation.bindings.length !== info.bindings.length) {
      throw executionError(
        `execute: received ${invocation.bindings.length} tensor bindings, expected ${info.bindings.length}`
      )
    }
    const inputs = invocation.bindings.map((input, index) =>
      nativeBinding(
        input,
        info.bindings[index]!,
        index,
        info.state !== undefined && invocation.state !== undefined && index === info.bindings.length - 1
          ? { compiled: info.state.batch, active: invocation.state.sequences.length }
          : undefined
      )
    )
    if (info.state !== undefined && invocation.scalars.length !== 0) {
      throw executionError("execute: stateful executable does not accept scalar inputs")
    }
    const [sequences, tokens] = resolveExecutionState(info.state, invocation.state)
    return {
      executable: executableRecord.value as Executable,
      info,
      inputs,
      scalars: [...invocation.scalars],
      sequences,
      tokens
    }
  }
  // Pools own fixed KV slabs, prefix-cache state, and recurrent geometry;
  // sequence wrappers retain KV block references and per-sequence recurrent
  // tensors. releaseSequence invalidates the public sequence and returns its
  // block references; remaining sequence, pool, and executable storage relies
  // on native finalization.
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
              options.dtype as NativeDType,
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
          return sequence((poolRecord.value as NativeKvPool).makeSequence(), poolRecord.info as KvPoolInfo)
        },
        catch: backendErrorFor("makeKvSequence", "execute")
      }),
    prefillMatch: (handle, tokens) =>
      Effect.try({
        try: () => (nativeSequence(handle, "prefillMatch").value as NativeKvSequence).prefillMatch([...tokens]),
        catch: backendErrorFor("prefillMatch", "execute")
      }),
    sequenceCursor: (handle) =>
      Effect.try({
        try: () => (nativeSequence(handle, "sequenceCursor").value as NativeKvSequence).cursor,
        catch: backendErrorFor("sequenceCursor", "execute")
      }),
    releaseSequence: (handle) =>
      Effect.try({
        try: () => {
          const sequenceRecord = nativeSequence(handle, "releaseSequence")
          const value = sequenceRecord.value as NativeKvSequence
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
            resolved.tokens ?? [],
            options.map((option) => ({ ...option })),
            token
          )
        }
      )
  }
  // Direct path I/O borrows tensors on save and transfers newly loaded native
  // tensors to caller-owned concrete handles on success. Safetensors has no
  // representation for the adapter's logical-f32/packed-u8 GGML storage, so
  // encoded handles are rejected rather than serialized as misleading u8 data.
  // Metal rejects archives containing f64 instead of falling back to CPU.
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
        Effect.mapError((error) =>
          error.message.includes("f64 is not supported on Metal")
            ? new Runtime.BackendError({
              reason: "unsupported-placement",
              backend: error.backend,
              operation: error.operation,
              phase: error.phase,
              message: error.message,
              ...(error.details === undefined ? {} : { details: error.details })
            })
            : error
        ),
        Effect.flatMap((archive) =>
          Effect.try({
            try: () => {
              const values = archive.entries.map((entry) => entry.tensor)
              try {
                const mapped = mapTensors(values)
                const metadata = Object.create(null) as Record<string, string>
                for (const [key, value] of Object.entries(archive.metadata)) {
                  if (typeof value !== "string") throw new Error(`invalid safetensors metadata ${key}`)
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
  // GGUF metadata crosses N-API through one populated scalar/array field because
  // generated object declarations cannot express the native tagged union.
  const metadataValue = (entry: NativeGgufMetadataEntry): Runtime.GgufMetadataEntry => {
    if (typeof entry.key !== "string" || entry.key.length === 0 || typeof entry.kind !== "string") {
      throw new Error("native GGUF metadata entry is invalid")
    }
    const numericKinds = ["u8", "i8", "u16", "i16", "u32", "i32", "f32", "u64", "i64", "f64"]
    const candidates: Array<Runtime.GgufMetadataScalar | ReadonlyArray<Runtime.GgufMetadataScalar>> = []
    if (entry.numberValue !== undefined) {
      if (!numericKinds.includes(entry.kind) || typeof entry.numberValue !== "number") {
        throw new Error("invalid GGUF number metadata")
      }
      candidates.push(entry.numberValue)
    }
    if (entry.stringValue !== undefined) {
      if (entry.kind !== "string" || typeof entry.stringValue !== "string") {
        throw new Error("invalid GGUF string metadata")
      }
      candidates.push(entry.stringValue)
    }
    if (entry.booleanValue !== undefined) {
      if (entry.kind !== "bool" || typeof entry.booleanValue !== "boolean") {
        throw new Error("invalid GGUF boolean metadata")
      }
      candidates.push(entry.booleanValue)
    }
    if (entry.numberArray !== undefined) {
      if (!numericKinds.includes(entry.kind) || !entry.numberArray.every((value) => typeof value === "number")) {
        throw new Error("invalid GGUF number array metadata")
      }
      candidates.push(Object.freeze([...entry.numberArray]))
    }
    if (entry.stringArray !== undefined) {
      if (entry.kind !== "string" || !entry.stringArray.every((value) => typeof value === "string")) {
        throw new Error("invalid GGUF string array metadata")
      }
      candidates.push(Object.freeze([...entry.stringArray]))
    }
    if (entry.booleanArray !== undefined) {
      if (entry.kind !== "bool" || !entry.booleanArray.every((value) => typeof value === "boolean")) {
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
    const encoded = value.format !== "F32"
    if (
      typeof value.name !== "string" || value.name.length === 0 ||
      (value.format !== "F32" && value.format !== "Q2_K" && value.format !== "Q3_K" && value.format !== "Q4_K" &&
        value.format !== "Q5_K" && value.format !== "Q6_K") ||
      value.logicalDtype !== "f32" || value.physicalDtype !== (encoded ? "u8" : "f32") ||
      !Array.isArray(value.logicalShape) || !Array.isArray(value.physicalShape) ||
      !value.logicalShape.every((dimension) => Number.isSafeInteger(dimension) && dimension > 0) ||
      !value.physicalShape.every((dimension) => Number.isSafeInteger(dimension) && dimension > 0) ||
      (encoded && !validEncodedGeometry(value.logicalShape, {
        encoding: value.format as Runtime.TensorStorageEncoding,
        physicalShape: value.physicalShape,
        physicalDtype: "u8"
      }))
    ) {
      throw new Error("native GGUF tensor descriptor is invalid")
    }
    return Object.freeze({
      name: value.name,
      format: value.format,
      logicalShape: Object.freeze([...value.logicalShape]),
      logicalDtype: "f32",
      physicalShape: Object.freeze([...value.physicalShape]),
      physicalDtype: value.physicalDtype
    })
  }
  // Inspection returns validated metadata and logical/physical descriptors but
  // no payloads. Loading streams supported F32 and K-quant payloads directly to
  // Metal shared storage; quantized handles retain logical f32 metadata over
  // packed u8 storage. Interrupted native loading and mapping failures clear
  // unpublished wrappers; successful mapping transfers them to caller-owned
  // concrete handles.
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
                      ...(storage === undefined ? {} : { storage })
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
    // Native compilation is synchronous and non-interruptible. It consumes
    // borrowed lazy roots, compile options, an optional decode specialization,
    // and the structural key; the returned immutable executable records logical
    // bindings/outputs and a completed native-inferred state schema.
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
          const value = native.compile(roots, options, state?.native, executableCacheKey(request))
          const outputs = request.roots.flatMap((root) => {
            const base = {
              dtype: root.dtype,
              ...(root.storage === undefined ? {} : { storage: root.storage })
            }
            if (state?.request.lastTokenRow !== true) return [{ shape: root.shape, ...base }]
            return Array.from({ length: state.request.batch }, () => ({ shape: [root.shape[2]!], ...base }))
          })
          return executable(value, bindings, outputs, completeStateSchema(value, state?.request))
        },
        catch: backendErrorFor("compile", "compile", "compilation-failed")
      }),
    // Invocation borrows all inputs and state until the native promise settles.
    // Successful output wrappers transfer to independent caller-owned handles;
    // interruption or post-native validation failure clears every unpublished
    // output. Runtime values are not part of this backend's public contract.
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
                return mapTensors(values, (nativeExecutable(handle, "execute").info as ExecutableInfo).outputs)
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
    // Deterministic release clears this concrete wrapper and marks its public
    // handle invalid for other operations. Releasing it again is a no-op. Native
    // aliases, executable constants, in-flight work, or exported readback
    // buffers may still retain the underlying allocation.
    release: (handle) =>
      Effect.try({
        try: () => {
          const tensor = typeof handle === "object" && handle !== null ? handleRecords.get(handle) : undefined
          if (tensor === undefined) {
            throw invalidHandle(
              "clear",
              "execute",
              typeof handle === "object" && handle !== null && backendHandles.has(handle)
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
          const value = tensor.value as NativeTensor
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
      diagnostics: {
        // This is current native memory attributed to live NativeTensor wrappers,
        // unlike executable diagnostics, which contain static planned byte totals.
        externalMemoryBytes: Effect.sync(() => native.externalMemoryBytes())
      }
    }
  }
  return runtime
}

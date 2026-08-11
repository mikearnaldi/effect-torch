import { Runtime } from "@effect-torch/core"
import { Effect } from "effect"
import { pipeArguments } from "effect/Pipeable"
import type {
  Executable,
  LazyTensor,
  NativeAddon,
  NativeCompileOptions,
  NativeDType,
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
  disposed: boolean
}

interface StructuralNode {
  readonly op: string
  readonly inputs: ReadonlyArray<Runtime.TensorHandle>
  readonly attributes: unknown
}

const numberBits = new DataView(new ArrayBuffer(8))

/** @internal */
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

/** @internal */
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

const handleRecords = new WeakMap<object, HandleRecord>()
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
        reason: message.includes("tensor was cleared") ? "invalid-handle" : reason,
        backend: backendName,
        operation,
        phase,
        message,
        details: { device, error }
      })
    })()

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
    const abort = () => token.cancel()
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
        signal.removeEventListener("abort", abort)
        if (signal.aborted) {
          try {
            onLateSuccess?.(value)
          } catch {
            // The interrupted fiber cannot observe cleanup failures.
          }
          return
        }
        resume(Effect.succeed(value))
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

/** @internal */
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
    tensorDevice: string
  ): H => {
    if (!shape.every((dimension) => Number.isSafeInteger(dimension) && dimension >= 0)) {
      throw new Error(`native runtime returned invalid shape [${shape}]`)
    }
    if (tensorDevice !== device) {
      throw new Error(`native runtime returned placement ${tensorDevice}, expected ${device}`)
    }
    return Object.freeze({
      _tag: tag,
      shape: Object.freeze([...shape]),
      dtype: dtype(tensorDtype),
      device: tensorDevice,
      placement,
      pipe(this: Runtime.TensorHandle) {
        return pipeArguments(this, arguments)
      }
    }) as unknown as H
  }
  let pendingStructure: StructuralNode | undefined
  const lazyHandle = (value: LazyTensor): Runtime.LazyTensorHandle => {
    const [shape, tensorDtype] = value.metadata()
    const handle = tensorObject<Runtime.LazyTensorHandle>("LazyTensor", shape, tensorDtype, device)
    handleRecords.set(handle, {
      owner,
      kind: "lazy-tensor",
      graph: value,
      ...(pendingStructure === undefined ? {} : { structure: pendingStructure }),
      disposed: false
    })
    pendingStructure = undefined
    backendHandles.add(handle)
    return handle
  }
  const graph = lazyHandle
  const concreteHandle = (value: NativeTensor): Runtime.ConcreteTensorHandle => {
    const graph = native.LazyTensor.fromMaterialized(value)
    const handle = tensorObject<Runtime.ConcreteTensorHandle>("Tensor", value.shape, value.dtype, value.device)
    handleRecords.set(handle, {
      owner,
      kind: "concrete-tensor",
      graph,
      value,
      structure: {
        op: "materialized-parameter",
        inputs: [],
        attributes: { shape: handle.shape, dtype: handle.dtype, device: handle.device }
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
  const executable = (
    value: Executable,
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
    handleRecords.set(handle, { owner, kind: "executable", value, info: state, disposed: false })
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
  const clearBuffers = (values: ReadonlyArray<NativeTensor>): void => {
    for (const value of values) {
      try {
        value.clear()
      } catch {
        // Best-effort cleanup for interrupted or invalid backend results.
      }
    }
  }
  const mapTensors = (values: ReadonlyArray<NativeTensor>): ReadonlyArray<Runtime.ConcreteTensorHandle> => {
    try {
      return values.map(concreteHandle)
    } catch (error) {
      clearBuffers(values)
      throw error
    }
  }
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
  const node = (request: Runtime.NodeRequest): Effect.Effect<Runtime.LazyTensorHandle, Runtime.BackendError> =>
    Effect.try({
      try: () => {
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
            return graph(
              native.LazyTensor.input(
                request.attributes.slot,
                [...request.attributes.shape],
                request.attributes.dtype as NativeDType
              )
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
                request.attributes.causal
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
                request.attributes.theta
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
          case "linear":
            return graph(
              nativeGraph(request.inputs[0], operation).linear(
                nativeGraph(request.inputs[1], operation),
                nativeGraph(request.inputs[2], operation)
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
      catch: backendErrorFor(request.op, "graph")
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
      batch: uint32(state.batch, "state.batch", false)
    }
    return {
      request,
      native: {
        maxTokens: request.maxTokens,
        blockSize: request.blockSize,
        kvDtype: request.kvDtype as NativeDType,
        ...(request.window === undefined ? {} : { window: request.window }),
        batch: request.batch
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
    return Object.freeze({
      maxTokens: state.maxTokens,
      blockSize: state.blockSize,
      kvDtype: state.kvDtype,
      ...(state.window === undefined ? {} : { window: state.window }),
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
  const pathSafetensors: Runtime.PathSafetensors = {
    save: (path, archive) =>
      cancellableFor(
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
              const mapped = mapTensors(values)
              try {
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
          return native.grad(
            nativeGraph(loss, "grad", "autodiff"),
            wrt.map((target) => nativeGraph(target, "grad", "autodiff"))
          ).map(graph)
        },
        catch: backendErrorFor("grad", "autodiff")
      }),
    compile: (request) =>
      Effect.try({
        try: () => {
          if (!Array.isArray(request.roots) || request.roots.length === 0) {
            throw new Error("compile: expected at least one root")
          }
          const roots = request.roots.map((root) => nativeGraph(root, "compile", "compile"))
          const options = mapCompileOptions(request.options)
          const state = mapStateRequest(request.state)
          const value = native.compile(roots, options, state?.native, executableCacheKey(request))
          return executable(value, completeStateSchema(value, state?.request))
        },
        catch: backendErrorFor("compile", "compile", "compilation-failed")
      }),
    execute: (handle, invocation) =>
      cancellableFor(
        "execute",
        "execute",
        (token) => {
          const executableRecord = nativeExecutable(handle, "execute")
          if (Object.keys(invocation.runtimeValues).length !== 0) {
            throw executionError(
              "execute: runtime values are not supported by the Apple Metal backend",
              "unsupported-operation"
            )
          }
          const inputs = invocation.bindings.map((input) => nativeTensor(input, "execute"))
          if (executableRecord.info !== undefined && invocation.scalars.length !== 0) {
            throw executionError("execute: stateful executable does not accept scalar inputs")
          }
          const [sequences, tokens] = resolveExecutionState(
            executableRecord.info as Runtime.DecodeStateSchema | undefined,
            invocation.state
          )
          return (executableRecord.value as Executable).execute(
            inputs,
            [...invocation.scalars],
            sequences,
            tokens,
            token
          )
        },
        clearBuffers
      ).pipe(
        Effect.flatMap((values) =>
          Effect.try({
            try: () => mapTensors(values),
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
    release: (handle) =>
      Effect.try({
        try: () => {
          const tensor = tensorRecord(handle, "clear", "execute")
          if (tensor.kind !== "concrete-tensor" || tensor.value === undefined) {
            throw invalidHandle("clear", "execute", "invalid-handle", "concrete-tensor")
          }
          const value = tensor.value as NativeTensor
          value.clear()
          tensor.disposed = true
        },
        catch: backendErrorFor("clear", "execute")
      }),
    extensions: {
      pathSafetensors,
      decode,
      diagnostics: {
        externalMemoryBytes: Effect.sync(() => native.externalMemoryBytes())
      }
    }
  }
  return runtime
}

import { Runtime } from "@effect-torch/core"
import type { Tensor } from "@effect-torch/core"
import native, {
  type CompiledProgram,
  type DecodeProgram,
  type LazyTensor,
  type NativeDType,
  type NativeKvPool,
  type NativeKvSequence,
  type NativeTensor
} from "@effect-torch/native"
import { Effect, Layer } from "effect"

export type Device = "cpu" | "metal"
type CancellationToken = InstanceType<typeof native.CancellationToken>
type HandleKind = "graph" | "buffer" | "program" | "decode-program" | "kv-pool" | "kv-sequence"

interface HandleRecord {
  readonly owner: object
  readonly kind: HandleKind
  readonly value: object
  readonly info?: unknown
  disposed: boolean
}

interface DecodeProgramInfo {
  readonly batch: number
  readonly layers: number
  readonly kvHeads: number
  readonly headDim: number
}

interface KvPoolInfo {
  readonly key: object
  readonly layers: number
  readonly kvHeads: number
  readonly headDim: number
}

interface KvSequenceInfo {
  readonly pool: KvPoolInfo
}

const handleRecords = new WeakMap<object, HandleRecord>()

const backendError = (
  device: Device,
  operation: string,
  phase: Runtime.BackendError["phase"],
  reason: Runtime.BackendError["reason"] = "execution-failed"
) =>
(error: unknown): Runtime.BackendError =>
  error instanceof Runtime.BackendError
    ? error
    : new Runtime.BackendError({
      reason,
      backend: "@effect-torch/backend-native",
      operation,
      phase,
      message: error instanceof Error ? error.message : String(error),
      details: { device, error }
    })

const cancellable = <A>(
  device: Device,
  operation: string,
  phase: Runtime.BackendError["phase"],
  register: (token: CancellationToken) => Promise<A>,
  onLateSuccess?: (value: A) => void
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
      resume(Effect.fail(backendError(device, operation, phase)(error)))
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
              backendError(device, operation, phase, phase === "io" ? "io-failed" : "execution-failed")(error)
            )
        )
      }
    )
  })

const makeRuntime = (device: Device): Runtime.RuntimeService => {
  const owner = Object.freeze({})
  const invalidHandle = (
    operation: string,
    phase: Runtime.BackendError["phase"],
    reason: "invalid-handle" | "foreign-handle",
    kind: HandleKind
  ): Runtime.BackendError =>
    new Runtime.BackendError({
      reason,
      backend: "@effect-torch/backend-native",
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
      throw invalidHandle(operation, phase, "invalid-handle", kind)
    }
    if (found.disposed) {
      throw new Runtime.BackendError({
        reason: "invalid-handle",
        backend: "@effect-torch/backend-native",
        operation,
        phase,
        message: `${operation}: ${kind === "buffer" ? "buffer was cleared" : "handle was released"}`,
        details: { device, kind }
      })
    }
    if (found.owner !== owner) {
      throw invalidHandle(operation, phase, "foreign-handle", kind)
    }
    return found
  }
  const wrapOpaque = <H extends object>(kind: HandleKind, value: object, info?: unknown): H => {
    const handle = Object.freeze({}) as H
    handleRecords.set(handle, { owner, kind, value, info, disposed: false })
    return handle
  }
  const graph = (value: LazyTensor): Runtime.GraphHandle => {
    const handle = new Proxy(Object.create(null) as object, {
      get: (_target, property) => {
        const member = Reflect.get(value, property, value) as unknown
        if (typeof member !== "function") return member
        const operation = String(property)
        return (...args: ReadonlyArray<unknown>) => {
          const nativeArgs = args.map((argument) =>
            argument !== null && typeof argument === "object" && !Array.isArray(argument)
              ? nativeGraph(argument as Runtime.GraphHandle, operation)
              : argument
          )
          return graph(Reflect.apply(member, value, nativeArgs) as LazyTensor)
        }
      }
    }) as Runtime.GraphHandle
    handleRecords.set(handle, { owner, kind: "graph", value, disposed: false })
    return handle
  }
  const nativeGraph = (
    handle: Runtime.GraphHandle,
    operation: string,
    phase: Runtime.BackendError["phase"] = "graph"
  ): LazyTensor => record(handle, "graph", operation, phase).value as LazyTensor
  const buffer = (value: NativeTensor): Runtime.BufferHandle => wrapOpaque<Runtime.BufferHandle>("buffer", value)
  const nativeBuffer = (
    handle: Runtime.BufferHandle,
    operation: string,
    phase: Runtime.BackendError["phase"] = "execute"
  ): NativeTensor => record(handle, "buffer", operation, phase).value as NativeTensor
  const program = (value: CompiledProgram): Runtime.ProgramHandle => wrapOpaque<Runtime.ProgramHandle>("program", value)
  const nativeProgram = (handle: Runtime.ProgramHandle, operation: string): CompiledProgram =>
    record(handle, "program", operation, "execute").value as CompiledProgram
  const decodeProgram = (value: DecodeProgram): Runtime.DecodeProgramHandle =>
    wrapOpaque<Runtime.DecodeProgramHandle>(
      "decode-program",
      value,
      {
        batch: value.batch,
        layers: value.layers,
        kvHeads: value.kvHeads,
        headDim: value.headDim
      } satisfies DecodeProgramInfo
    )
  const nativeDecodeProgram = (handle: Runtime.DecodeProgramHandle, operation: string): HandleRecord =>
    record(handle, "decode-program", operation, "execute")
  const pool = (value: NativeKvPool, info: Omit<KvPoolInfo, "key">): Runtime.KvPoolHandle =>
    wrapOpaque<Runtime.KvPoolHandle>("kv-pool", value, { ...info, key: value } satisfies KvPoolInfo)
  const nativePool = (handle: Runtime.KvPoolHandle, operation: string): HandleRecord =>
    record(handle, "kv-pool", operation, "execute")
  const sequence = (value: NativeKvSequence, pool: KvPoolInfo): Runtime.KvSequenceHandle =>
    wrapOpaque<Runtime.KvSequenceHandle>("kv-sequence", value, { pool } satisfies KvSequenceInfo)
  const nativeSequence = (handle: Runtime.KvSequenceHandle, operation: string): HandleRecord =>
    record(handle, "kv-sequence", operation, "execute")
  const placement: Runtime.Placement = {
    id: device,
    deviceType: device,
    description: device === "cpu" ? "Native CPU" : "Apple Metal"
  }
  const toBufferValue = (value: NativeTensor): Runtime.BufferValue => {
    if (value.device !== device) {
      throw new Error(`native runtime returned placement ${value.device}, expected ${device}`)
    }
    return {
      handle: buffer(value),
      shape: value.shape,
      dtype: value.dtype as Tensor.DType,
      placement
    }
  }
  const clearBuffers = (values: ReadonlyArray<NativeTensor>): void => {
    for (const value of values) {
      try {
        value.clear()
      } catch {
        // Best-effort cleanup for interrupted or invalid backend results.
      }
    }
  }
  const mapBuffers = (values: ReadonlyArray<NativeTensor>): ReadonlyArray<Runtime.BufferValue> => {
    try {
      return values.map(toBufferValue)
    } catch (error) {
      clearBuffers(values)
      throw error
    }
  }
  const resolveDecode = (
    programHandle: Runtime.DecodeProgramHandle,
    sequenceHandles: ReadonlyArray<Runtime.KvSequenceHandle>,
    operation: string
  ): { readonly program: DecodeProgram; readonly sequences: ReadonlyArray<NativeKvSequence> } => {
    const programRecord = nativeDecodeProgram(programHandle, operation)
    const programInfo = programRecord.info as DecodeProgramInfo
    const sequenceRecords = sequenceHandles.map((handle) => nativeSequence(handle, operation))
    const sequenceInfos = sequenceRecords.map((entry) => entry.info as KvSequenceInfo)
    const firstPool = sequenceInfos[0]?.pool
    if (
      firstPool === undefined ||
      sequenceInfos.some((entry) => entry.pool.key !== firstPool.key) ||
      firstPool.layers !== programInfo.layers ||
      firstPool.kvHeads !== programInfo.kvHeads ||
      firstPool.headDim !== programInfo.headDim ||
      sequenceRecords.length > programInfo.batch
    ) {
      throw invalidHandle(operation, "execute", "invalid-handle", "kv-sequence")
    }
    return {
      program: programRecord.value as DecodeProgram,
      sequences: sequenceRecords.map((entry) => entry.value as NativeKvSequence)
    }
  }
  const decode: Runtime.DecodeRuntime = {
    compile: (roots, window, batch) =>
      Effect.try({
        try: () => {
          const value = native.compileDecode(
            roots.map((root) => nativeGraph(root, "compileDecode", "compile")),
            window,
            batch
          )
          return {
            handle: decodeProgram(value),
            batch: value.batch,
            layers: value.layers,
            kvHeads: value.kvHeads,
            headDim: value.headDim
          }
        },
        catch: backendError(device, "compileDecode", "compile", "compilation-failed")
      }),
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
              device,
              options.dtype as NativeDType
            ),
            { layers: options.layers, kvHeads: options.kvHeads, headDim: options.headDim }
          ),
        catch: backendError(device, "makeKvPool", "execute")
      }),
    makeSequence: (handle) =>
      Effect.try({
        try: () => {
          const poolRecord = nativePool(handle, "makeKvSequence")
          return sequence((poolRecord.value as NativeKvPool).makeSequence(), poolRecord.info as KvPoolInfo)
        },
        catch: backendError(device, "makeKvSequence", "execute")
      }),
    prefillMatch: (handle, tokens) =>
      Effect.try({
        try: () => (nativeSequence(handle, "prefillMatch").value as NativeKvSequence).prefillMatch([...tokens]),
        catch: backendError(device, "prefillMatch", "execute")
      }),
    sequenceCursor: (handle) =>
      Effect.try({
        try: () => (nativeSequence(handle, "sequenceCursor").value as NativeKvSequence).cursor,
        catch: backendError(device, "sequenceCursor", "execute")
      }),
    releaseSequence: (handle) =>
      Effect.try({
        try: () => {
          const sequenceRecord = nativeSequence(handle, "releaseSequence")
          const value = sequenceRecord.value as NativeKvSequence
          value.release()
          sequenceRecord.disposed = true
        },
        catch: backendError(device, "releaseSequence", "execute")
      }),
    run: (handle, inputs, seq, tokens) =>
      cancellable(
        device,
        "decode",
        "execute",
        (token) => {
          const resolved = resolveDecode(handle, [seq], "decode")
          return resolved.program.run(
            inputs.map((input) => nativeBuffer(input, "decode")),
            resolved.sequences[0]!,
            [...tokens],
            token
          )
        },
        clearBuffers
      ).pipe(
        Effect.flatMap((values) =>
          Effect.try({
            try: () => mapBuffers(values),
            catch: backendError(device, "decode", "execute")
          })
        )
      ),
    runBatched: (handle, inputs, sequences, tokens) =>
      cancellable(device, "decodeBatched", "execute", (token) => {
        const resolved = resolveDecode(handle, sequences, "decodeBatched")
        return resolved.program.runBatched(
          inputs.map((input) => nativeBuffer(input, "decodeBatched")),
          [...resolved.sequences],
          tokens.map((row) => [...row]),
          token
        )
      }, clearBuffers).pipe(
        Effect.flatMap((values) =>
          Effect.try({
            try: () => mapBuffers(values),
            catch: backendError(device, "decodeBatched", "execute")
          })
        )
      )
  }
  const pathSafetensors: Runtime.PathSafetensors = {
    save: (path, names, tensors) =>
      cancellable(
        device,
        "save",
        "io",
        (token) =>
          native.saveTensors(
            path,
            [...names],
            tensors.map((tensor) => nativeGraph(tensor, "save", "io")),
            token
          )
      ),
    load: (path) =>
      cancellable(
        device,
        "load",
        "io",
        (token) => native.loadTensors(path, device, token),
        ([, values]) => clearBuffers(values)
      ).pipe(
        Effect.flatMap(([names, values]) =>
          Effect.try({
            try: () => mapBuffers(values).map((value, index) => [names[index]!, value] as const),
            catch: backendError(device, "load", "io", "unsupported-placement")
          })
        )
      )
  }
  const runtime: Runtime.RuntimeService = {
    identity: owner,
    backend: { name: "@effect-torch/backend-native" },
    placement,
    capabilities: {
      dtypes: device === "cpu"
        ? ["f32", "f64", "f16", "bf16", "i64", "u8", "u32"]
        : ["f32", "f16", "bf16", "i64", "u8", "u32"],
      features: device === "metal" ? ["mixed-bf16"] : []
    },
    graph: {
      constant: (value, dtype) => graph(native.LazyTensor.constant(value, dtype as NativeDType, device)),
      zeros: (shape, dtype) => graph(native.LazyTensor.zeros(shape, dtype as NativeDType, device)),
      ones: (shape, dtype) => graph(native.LazyTensor.ones(shape, dtype as NativeDType, device)),
      full: (shape, value, dtype) => graph(native.LazyTensor.full(shape, value, dtype as NativeDType, device)),
      randn: (shape, dtype) => graph(native.LazyTensor.randn(shape, dtype as NativeDType, device)),
      uniform: (shape, lo, hi, dtype) => graph(native.LazyTensor.uniform(shape, lo, hi, dtype as NativeDType, device)),
      arange: (start, end, step, dtype) =>
        graph(native.LazyTensor.arange(start, end, step, dtype as NativeDType, device)),
      eye: (n, dtype) => graph(native.LazyTensor.eye(n, dtype as NativeDType, device)),
      fromBytes: (data, shape, dtype) => graph(native.LazyTensor.fromBytes(data, shape, dtype as NativeDType, device)),
      fromBuffer: (handle) => graph(native.LazyTensor.fromMaterialized(nativeBuffer(handle, "fromBuffer", "graph"))),
      input: (slot, shape, dtype) => graph(native.LazyTensor.input(slot, shape, dtype as NativeDType, device)),
      scalarInput: (slot, dtype) => graph(native.LazyTensor.scalarInput(slot, dtype as NativeDType, device))
    },
    validateGraph: (handles) =>
      Effect.try({
        try: () => {
          for (const handle of handles) nativeGraph(handle, "validate")
        },
        catch: backendError(device, "validate", "graph")
      }),
    evaluate: (roots) =>
      cancellable(
        device,
        "evaluate",
        "execute",
        (token) => native.evalLazy(roots.map((root) => nativeGraph(root, "evaluate")), token),
        clearBuffers
      ).pipe(
        Effect.flatMap((values) =>
          Effect.try({
            try: () => mapBuffers(values),
            catch: backendError(device, "evaluate", "execute")
          })
        )
      ),
    grad: (loss, wrt) =>
      Effect.try({
        try: () =>
          native.grad(
            nativeGraph(loss, "grad", "autodiff"),
            wrt.map((target) => nativeGraph(target, "grad", "autodiff"))
          ).map(graph),
        catch: backendError(device, "grad", "autodiff")
      }),
    compile: (roots) =>
      Effect.try({
        try: () => program(native.compile(roots.map((root) => nativeGraph(root, "compile", "compile")))),
        catch: backendError(device, "compile", "compile", "compilation-failed")
      }),
    run: (handle, inputs, scalars) =>
      cancellable(
        device,
        "run",
        "execute",
        (token) =>
          nativeProgram(handle, "run").run(inputs.map((input) => nativeBuffer(input, "run")), [...scalars], token),
        clearBuffers
      ).pipe(
        Effect.flatMap((values) =>
          Effect.try({
            try: () => mapBuffers(values),
            catch: backendError(device, "run", "execute")
          })
        )
      ),
    readback: (handle) =>
      cancellable(
        device,
        "readback",
        "readback",
        (token) => nativeBuffer(handle, "readback", "readback").readback(token)
      ),
    releaseBuffer: (handle) =>
      Effect.try({
        try: () => {
          const bufferRecord = record(handle, "buffer", "clear", "execute")
          const value = bufferRecord.value as NativeTensor
          value.clear()
          bufferRecord.disposed = true
        },
        catch: backendError(device, "clear", "execute")
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

export const cpu: Runtime.RuntimeService = makeRuntime("cpu")
export const metal: Runtime.RuntimeService = makeRuntime("metal")

export const Cpu: Layer.Layer<Runtime.Runtime> = Layer.succeed(Runtime.Runtime, cpu)
export const Metal: Layer.Layer<Runtime.Runtime> = Layer.succeed(Runtime.Runtime, metal)

export const isAvailable = (device: Device): boolean => {
  try {
    return native.isDeviceAvailable(device)
  } catch {
    return false
  }
}

export const Best: Layer.Layer<Runtime.Runtime> = Layer.effect(
  Runtime.Runtime,
  Effect.sync(() => isAvailable("metal") ? metal : cpu)
)

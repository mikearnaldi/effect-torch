/**
 * Safetensors archive inspection, tensor persistence, and named model parameters.
 *
 * Loading preserves archive dtypes and transfers ownership of concrete tensor
 * handles to the caller. Release them with {@link Tensor.clear} or
 * {@link Tensor.clearAll}. Header inspection allocates no tensor payloads, and
 * selected-name loading reads only the requested tensors. Saving borrows inputs
 * and materializes all entries together before writing.
 *
 * @since 0.1.0
 * @category modules
 */
import { Effect, Exit, Predicate } from "effect"
import * as Model from "./Model.ts"
import * as Runtime from "./Runtime.ts"
import * as Tensor from "./Tensor.ts"

const backendMessage = (error: Runtime.BackendError): string => error.message

const caughtTensorError = (op: string, cause: unknown): Tensor.TensorError =>
  cause instanceof Tensor.TensorError
    ? cause
    : cause instanceof Runtime.BackendError
    ? new Tensor.TensorError({ op, message: cause.message, backend: cause })
    : new Tensor.TensorError({ op, message: cause instanceof Error ? cause.message : String(cause) })

const fromBackend = <A>(
  op: string,
  effect: Effect.Effect<A, Runtime.BackendError>
): Effect.Effect<A, Tensor.TensorError> =>
  Effect.mapError(effect, (error) => new Tensor.TensorError({ op, message: backendMessage(error), backend: error }))

const validateMetadata = (op: string, metadata: Readonly<Record<string, string>>): Readonly<Record<string, string>> => {
  if (!Predicate.isObject(metadata)) {
    throw new Tensor.TensorError({ op, message: `${op}: metadata must be a record of strings` })
  }
  const output: Record<string, string> = Object.create(null)
  for (const [key, value] of Object.entries(metadata)) {
    if (!Predicate.isString(value)) {
      throw new Tensor.TensorError({ op, message: `${op}: metadata ${JSON.stringify(key)} must be a string` })
    }
    output[key] = value
  }
  return Object.freeze(output)
}

// Result-validation failures occur after the backend transferred output
// ownership. Attempt every release independently when validation fails.
const releaseTensors = (
  runtime: Runtime.RuntimeService,
  values: ReadonlyArray<Tensor.Concrete>
): Effect.Effect<void> => Effect.forEach(values, (value) => Effect.ignore(runtime.release(value)), { discard: true })

/**
 * Options for direct safetensors writes.
 *
 * @since 0.1.0
 * @category models
 */
export interface SaveOptions {
  /**
   * String metadata stored in the archive; `__metadata__` is reserved as a
   * tensor name.
   */
  readonly metadata?: Readonly<Record<string, string>>
}

/**
 * Materialized tensors and string metadata loaded from a safetensors archive.
 * Each tensor handle owns runtime storage that the caller may release with
 * {@link Tensor.clear} when no longer needed.
 *
 * @since 0.1.0
 * @category models
 */
export interface Archive {
  /** Null-prototype, frozen record of archive names to materialized tensors. */
  readonly tensors: Readonly<Record<string, Tensor.Concrete>>
  /** Frozen string metadata record. */
  readonly metadata: Readonly<Record<string, string>>
}

/**
 * Tensor-name selection for {@link loadArchive} and {@link load}.
 *
 * @since 0.1.0
 * @category models
 */
export type LoadOptions = Runtime.PathSafetensorsLoadOptions

/**
 * A tensor's name, dtype, shape, and exact payload size from an archive header.
 *
 * @since 0.1.0
 * @category models
 */
export type TensorInfo = Runtime.PathSafetensorsTensorInfo

/**
 * Frozen archive descriptors and string metadata, without tensor ownership.
 *
 * @since 0.1.0
 * @category models
 */
export type Inspection = Runtime.PathSafetensorsInspection

const safetensorsElementBytes = {
  f32: 4,
  f64: 8,
  f16: 2,
  bf16: 2,
  i64: 8,
  u8: 1,
  u32: 4
} satisfies Readonly<Record<Tensor.DType, number>>

/**
 * Reads tensor descriptors from a safetensors file or a Hugging Face
 * `.safetensors.index.json` and its shard headers. No tensor payloads are read
 * or allocated on the device. Returned metadata describes storage; it does
 * not guarantee that the active backend can execute every listed dtype.
 *
 * @since 0.1.0
 * @category constructors
 */
export const inspectArchive = (
  path: string
): Effect.Effect<Inspection, Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const inspection = yield* fromBackend(
      "inspectArchive",
      runtime.extensions.pathSafetensors.inspect(path)
    )
    return yield* Effect.try({
      try: () => {
        if (!Predicate.isObjectOrArray(inspection) || !Array.isArray(inspection.entries)) {
          throw new Error("inspectArchive: backend returned an invalid inspection")
        }
        const names = new Set<string>()
        const entries = Array.from(inspection.entries).map((entry: TensorInfo): TensorInfo => {
          if (
            !Predicate.isObjectOrArray(entry) || !Predicate.isString(entry.name) ||
            entry.name === "__metadata__" || names.has(entry.name) ||
            !Predicate.isString(entry.dtype) || !Object.hasOwn(safetensorsElementBytes, entry.dtype) ||
            !Array.isArray(entry.shape)
          ) {
            throw new Error("inspectArchive: backend returned an invalid tensor descriptor")
          }
          const shape = Array.from(entry.shape)
          if (!shape.every((dim) => Number.isSafeInteger(dim) && dim >= 0)) {
            throw new Error("inspectArchive: backend returned an invalid tensor shape")
          }
          const bytes = shape.includes(0)
            ? 0
            : shape.reduce((total, dim) => total * dim, safetensorsElementBytes[entry.dtype])
          if (!Number.isSafeInteger(bytes) || bytes !== entry.byteLength) {
            throw new Error(`inspectArchive: invalid payload size for ${JSON.stringify(entry.name)}`)
          }
          names.add(entry.name)
          return Object.freeze({
            name: entry.name,
            dtype: entry.dtype,
            shape: Object.freeze(shape),
            byteLength: entry.byteLength
          })
        })
        entries.sort((left, right) => left.name < right.name ? -1 : left.name > right.name ? 1 : 0)
        return Object.freeze({
          entries: Object.freeze(entries),
          metadata: validateMetadata("inspectArchive", inspection.metadata)
        })
      },
      catch: (error) => caughtTensorError("inspectArchive", error)
    })
  })

/**
 * Saves tensors through the runtime's direct path. All entries are
 * compiled and materialized together before the transfer service serializes
 * the resulting concrete tensors. After the save attempt, the wrapper attempts
 * to release every temporary in input order, independently ignoring release
 * failures. Original concrete inputs remain caller-owned. Encoded
 * tensors are rejected because safetensors cannot preserve their logical
 * storage metadata. Interruption requests cancellation of compilation,
 * execution, or I/O and transfers no ownership.
 *
 * @since 0.1.0
 * @category destructors
 */
export const save = (
  path: string,
  tensors: Readonly<Record<string, Tensor.Any>>,
  options: SaveOptions = {}
): Effect.Effect<void, Tensor.TensorError, Runtime.Runtime> => {
  const entries = Object.entries(tensors)
  if (entries.length === 0) return new Tensor.TensorError({ op: "save", message: "save: expected at least one tensor" })
  if (entries.some(([name]) => name === "__metadata__")) {
    return new Tensor.TensorError({ op: "save", message: "save: __metadata__ is reserved and cannot be a tensor name" })
  }
  const encoded = entries.find(([, tensor]) => tensor.storage !== undefined)
  if (encoded !== undefined) {
    return new Tensor.TensorError({
      op: "save",
      message: `save: encoded tensor ${JSON.stringify(encoded[0])} cannot be represented by safetensors`
    })
  }
  return Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const extension = runtime.extensions.pathSafetensors
    const metadata = yield* Effect.try({
      try: () => validateMetadata("save", options.metadata ?? {}),
      catch: (error) => caughtTensorError("save", error)
    })
    const materialized = yield* Tensor.compute(entries.map(([, tensor]) => tensor))
    yield* Effect.ensuring(
      fromBackend(
        "save",
        extension.save(path, {
          entries: entries.map(([name], index) => ({ name, tensor: materialized[index] })),
          metadata
        })
      ),
      releaseTensors(runtime, materialized)
    )
  })
}

/**
 * Loads tensors and archive metadata from a safetensors file or a Hugging Face
 * `.safetensors.index.json`. `options.names` selects unique tensor names;
 * missing names fail, an empty array loads none, and omission loads all.
 * Native readers index headers and read only selected payloads, with host
 * staging bounded by one tensor rather than the full archive.
 * Sharded loads merge string metadata from opened shards and reject conflicts;
 * index bookkeeping is excluded. An empty index selection has empty metadata.
 * The runtime validates native handles before returning them. Each concrete
 * handle independently owns storage; release it with {@link Tensor.clear} when
 * deterministic cleanup is required. The extension owns partial/late cleanup
 * for failed or interrupted loads. If a successful archive fails wrapper
 * validation, every discoverable handle receives an independent best-effort
 * release attempt in archive order.
 *
 * @since 0.1.0
 * @category constructors
 */
export const loadArchive = (
  path: string,
  options: LoadOptions = {}
): Effect.Effect<Archive, Tensor.TensorError, Runtime.Runtime> =>
  Effect.suspend(() => {
    let runtime: Runtime.RuntimeService | undefined
    let candidates: ReadonlyArray<Tensor.Concrete> = []
    return Effect.onExit(
      Effect.gen(function*() {
        const selected = yield* Effect.try({
          try: () => {
            if (options.names === undefined) return undefined
            if (!Array.isArray(options.names)) {
              throw new Error("loadArchive: names must be an array of unique strings")
            }
            const names = Array.from(options.names)
            if (
              names.some((name) => !Predicate.isString(name) || name === "__metadata__") ||
              new Set(names).size !== names.length
            ) {
              throw new Error("loadArchive: names must be unique strings and cannot include __metadata__")
            }
            return Object.freeze(names)
          },
          catch: (error) => caughtTensorError("loadArchive", error)
        })
        runtime = yield* Runtime.Runtime
        const extension = runtime.extensions.pathSafetensors
        const archive = yield* fromBackend(
          "loadArchive",
          extension.load(path, selected === undefined ? {} : { names: selected })
        )
        const loadedArchive = archive
        const discovered = new Set<Tensor.Concrete>()
        if (Predicate.isObjectOrArray(archive) && Array.isArray(archive.entries)) {
          for (const entry of loadedArchive.entries) {
            const loadedEntry = entry
            if (Predicate.isObjectOrArray(entry) && Predicate.isObjectOrArray(loadedEntry.tensor)) {
              discovered.add(loadedEntry.tensor)
            }
          }
        }
        candidates = Array.from(discovered)
        const checked = yield* Effect.try({
          try: () => {
            if (!Predicate.isObjectOrArray(archive) || !Array.isArray(archive.entries)) {
              throw new Tensor.TensorError({
                op: "loadArchive",
                message: "loadArchive: backend returned an invalid archive"
              })
            }
            const metadata = validateMetadata("loadArchive", loadedArchive.metadata)
            const names = new Set<string>()
            const handles = new Set<Tensor.Concrete>()
            const tensors: Record<string, Tensor.Concrete> = Object.create(null)
            for (const entry of loadedArchive.entries) {
              const loadedEntry = entry
              if (
                !Predicate.isObjectOrArray(entry) || !Predicate.isString(entry.name) ||
                entry.name === "__metadata__" || names.has(entry.name)
              ) {
                throw new Tensor.TensorError({
                  op: "loadArchive",
                  message: "loadArchive: backend returned invalid tensor names"
                })
              }
              names.add(loadedEntry.name)
              const tensor = loadedEntry.tensor
              if (handles.has(tensor)) {
                throw new Tensor.TensorError({
                  op: "loadArchive",
                  message: "loadArchive: backend returned duplicate tensor ownership"
                })
              }
              handles.add(tensor)
              tensors[loadedEntry.name] = tensor
            }
            if (
              selected !== undefined && (names.size !== selected.length || selected.some((name) => !names.has(name)))
            ) {
              throw new Tensor.TensorError({
                op: "loadArchive",
                message: "loadArchive: backend returned tensors that differ from the requested selection"
              })
            }
            return Object.freeze({ tensors: Object.freeze(tensors), metadata })
          },
          catch: (error) => caughtTensorError("loadArchive", error)
        })
        return checked
      }),
      (exit) =>
        Exit.isFailure(exit) && runtime !== undefined
          ? releaseTensors(runtime, candidates)
          : Effect.void
    )
  })

/**
 * Loads only the tensor record from {@link loadArchive}. Returned concrete
 * handles have the same ownership and cleanup requirements.
 *
 * @since 0.1.0
 * @category constructors
 */
export const load = (
  path: string,
  options: LoadOptions = {}
): Effect.Effect<Readonly<Record<string, Tensor.Concrete>>, Tensor.TensorError, Runtime.Runtime> =>
  Effect.map(loadArchive(path, options), (archive) => archive.tensors).pipe(
    Effect.mapError((error) =>
      new Tensor.TensorError({
        op: "load",
        message: error.message,
        backend: error.backend
      })
    )
  )

/**
 * Saves a model's parameters to a safetensors file, zipping parameter-spec names
 * with the parameter array into the record {@link save} takes.
 * Fails with a {@link Model.ModelError} if the parameter array's length does
 * not match the model's arity. It does not compare tensor shapes or dtypes with
 * {@link Model.Model.parameterSpecs}. Saving borrows parameters and does not clear them.
 *
 * @since 0.1.0
 * @category destructors
 */
export const saveModel = (
  model: Model.Model,
  params: Model.Params,
  path: string
): Effect.Effect<void, Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
  params.length !== model.parameterSpecs.length
    ? new Model.ModelError({
      op: "save",
      message: `model has ${model.parameterSpecs.length} parameters, got ${params.length}`
    })
    : save(
      path,
      Object.fromEntries(model.parameterSpecs.map((parameter, i) => [parameter.name, params[i]]))
    )

/**
 * Loads a safetensors file or sharded index. Returns tensors selected by
 * parameter-spec names in parameter-array order. A missing key fails with a
 * {@link Model.ModelError}; extra keys are ignored. This maps names and arity but
 * does not validate the architecture. It leaves shape, dtype, storage, and
 * placement compatibility unchecked until first use.
 *
 * Header inspection checks names before loading. Only declared parameters are
 * materialized; other payloads are not read or allocated. Failed or interrupted
 * loads release all partial results. On success, the selected handles are
 * caller-owned and should be released with
 * {@link Tensor.clear} when no longer needed.
 *
 * @since 0.1.0
 * @category destructors
 */
export const loadModel = (
  model: Model.Model,
  path: string
): Effect.Effect<ReadonlyArray<Tensor.Concrete>, Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.flatMap(
    Effect.gen(function*() {
      const inspection = yield* inspectArchive(path)
      const available = new Set(inspection.entries.map((entry) => entry.name))
      const names = model.parameterSpecs.map((parameter) => parameter.name)
      for (const name of names) {
        if (!available.has(name)) {
          return yield* new Model.ModelError({ op: "load", message: `missing parameter "${name}" in ${path}` })
        }
      }
      return yield* load(path, { names })
    }),
    (record) =>
      Effect.onExit(
        Effect.gen(function*() {
          const params: Array<Tensor.Concrete> = []
          for (const { name } of model.parameterSpecs) {
            const param = record[name]
            if (param === undefined) {
              return yield* new Model.ModelError({
                op: "load",
                message: `missing parameter "${name}" in ${path}`
              })
            }
            params.push(param)
          }
          const retained = new Set(params)
          for (const tensor of Object.values(record)) {
            if (retained.has(tensor)) continue
            yield* Tensor.clear(tensor)
          }
          return params
        }),
        (exit) => Exit.isFailure(exit) ? Tensor.clearAll(Object.values(record)) : Effect.void
      )
  )

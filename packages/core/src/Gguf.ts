/**
 * Native GGUF v3 inspection, registry-based model resolution, validation, and
 * loading. The selected runtime parses the file and creates concrete tensor
 * handles; this module canonicalizes metadata, resolves the exact
 * `gguf:<architecture>` registration, constructs the model, and verifies that
 * the inspected and loaded tensor catalogs form a bijection with its parameter
 * catalog.
 *
 * @since 0.1.0
 */
import { Data, Effect, Exit } from "effect"
import type * as Model from "./Model.ts"
import * as Registry from "./Registry.ts"
import * as Runtime from "./Runtime.ts"
import type * as Tensor from "./Tensor.ts"

/**
 * A native GGUF inspection/loading failure or a structural validation failure.
 * Missing architecture registrations remain {@link Registry.RegistryError}s,
 * and architecture construction failures remain `Model.ModelError`s.
 *
 * @since 0.1.0
 * @category errors
 */
export class GgufError extends Data.TaggedError("GgufError")<{
  /** Phase that inspected the file, validated metadata/catalogs, or loaded payloads. */
  readonly op: "inspect" | "validate" | "load"
  /** Human-readable diagnostic; branch on the error tag and `op`, not this text. */
  readonly message: string
  /** Original runtime failure for native inspection or payload loading. */
  readonly backend?: Runtime.BackendError
}> {}

/**
 * A constructed model, its concrete parameters in model order, and the
 * canonical configuration used to construct it.
 *
 * @since 0.1.0
 * @category models
 */
export interface LoadedModel {
  /** The model constructed from the artifact's architecture configuration. */
  readonly model: Model.Model
  /**
   * Caller-owned loaded tensors in `model.parameters` order. Release each
   * handle when no longer needed; repeated releases are no-ops.
   */
  readonly params: ReadonlyArray<Tensor.Concrete>
  /**
   * Canonical configuration passed to the architecture: the architecture
   * prefix and `general.` are stripped, other keys are retained, and
   * `vocab_size` may be derived from `tokenizer.ggml.tokens`. This is not the
   * raw ordered GGUF metadata table; callers should treat the map and any array
   * values as immutable. Original numeric kinds are erased to JavaScript
   * numbers by the runtime boundary, so 64-bit integer values may already have
   * lost precision.
   */
  readonly metadata: ReadonlyMap<string, unknown>
}

const fail = (op: GgufError["op"], message: string): GgufError => new GgufError({ op, message })

const fromBackend = <A>(
  op: "inspect" | "load",
  effect: Effect.Effect<A, Runtime.BackendError>
): Effect.Effect<A, GgufError> =>
  Effect.mapError(effect, (backend) => new GgufError({ op, message: backend.message, backend }))

const validateEffect = <A>(evaluate: () => A): Effect.Effect<A, GgufError> =>
  Effect.try({
    try: evaluate,
    catch: (error) => error instanceof GgufError ? error : fail("validate", String(error))
  })

const isScalar = (value: unknown): value is Runtime.GgufMetadataScalar =>
  typeof value === "number" || typeof value === "string" || typeof value === "boolean"

const validShape = (shape: unknown, rank?: number): shape is ReadonlyArray<number> =>
  Array.isArray(shape) && (rank === undefined || shape.length === rank) &&
  shape.every((dimension) => Number.isSafeInteger(dimension) && dimension > 0)

const sameShape = (left: ReadonlyArray<number>, right: ReadonlyArray<number>): boolean =>
  left.length === right.length && left.every((dimension, index) => dimension === right[index])

const storageEncoding = (value: unknown): value is Runtime.TensorStorageEncoding =>
  value === "Q2_K" || value === "Q3_K" || value === "Q4_K" || value === "Q5_K" || value === "Q6_K"

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

const validateDescriptor = (value: Runtime.GgufTensorDescriptor): Runtime.GgufTensorDescriptor => {
  if (typeof value !== "object" || value === null || typeof value.name !== "string" || value.name.length === 0) {
    throw fail("validate", "GGUF tensor descriptor has an invalid name")
  }
  if (!validShape(value.logicalShape) || value.logicalShape.length > 4 || !validShape(value.physicalShape)) {
    throw fail("validate", `GGUF tensor ${JSON.stringify(value.name)} has invalid dimensions`)
  }
  if (value.logicalDtype !== "f32") {
    throw fail("validate", `GGUF tensor ${JSON.stringify(value.name)} has invalid logical dtype`)
  }
  if (value.format === "F32") {
    if (value.physicalDtype !== "f32" || !sameShape(value.logicalShape, value.physicalShape)) {
      throw fail("validate", `GGUF tensor ${JSON.stringify(value.name)} has invalid F32 storage metadata`)
    }
  } else if (
    !storageEncoding(value.format) || value.physicalDtype !== "u8" || !validShape(value.physicalShape, 2)
  ) {
    throw fail("validate", `GGUF tensor ${JSON.stringify(value.name)} has invalid encoded storage metadata`)
  } else {
    const columns = value.logicalShape.at(-1)!
    const rows = value.logicalShape.slice(0, -1).reduce((total, dimension) => total * dimension, 1)
    const rowBytes = encodedRowBytes(value.format, columns)
    if (rowBytes === undefined || !sameShape(value.physicalShape, [rows, rowBytes])) {
      throw fail("validate", `GGUF tensor ${JSON.stringify(value.name)} has invalid encoded storage geometry`)
    }
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

const validateInspection = (value: Runtime.GgufInspection): Runtime.GgufInspection => {
  if (typeof value !== "object" || value === null || !Array.isArray(value.metadata) || !Array.isArray(value.tensors)) {
    throw fail("validate", "native GGUF inspection has an invalid structure")
  }
  const metadataKeys = new Set<string>()
  const metadata = value.metadata.map((entry) => {
    if (typeof entry !== "object" || entry === null || typeof entry.key !== "string" || entry.key.length === 0) {
      throw fail("validate", "GGUF metadata contains an invalid key")
    }
    if (metadataKeys.has(entry.key)) throw fail("validate", `duplicate GGUF metadata key ${JSON.stringify(entry.key)}`)
    metadataKeys.add(entry.key)
    const entryValue = entry.value
    if (!isScalar(entryValue) && !(Array.isArray(entryValue) && entryValue.every(isScalar))) {
      throw fail("validate", `GGUF metadata ${JSON.stringify(entry.key)} has an invalid value`)
    }
    return Object.freeze({
      key: entry.key,
      value: Array.isArray(entryValue) ? Object.freeze([...entryValue]) : entryValue
    })
  })
  const names = new Set<string>()
  const tensors = value.tensors.map((descriptor) => {
    const checked = validateDescriptor(descriptor)
    if (names.has(checked.name)) throw fail("validate", `duplicate GGUF tensor ${JSON.stringify(checked.name)}`)
    names.add(checked.name)
    return checked
  })
  return Object.freeze({ metadata: Object.freeze(metadata), tensors: Object.freeze(tensors) })
}

const descriptorEqual = (left: Runtime.GgufTensorDescriptor, right: Runtime.GgufTensorDescriptor): boolean =>
  left.name === right.name && left.format === right.format && left.logicalDtype === right.logicalDtype &&
  left.physicalDtype === right.physicalDtype && sameShape(left.logicalShape, right.logicalShape) &&
  sameShape(left.physicalShape, right.physicalShape)

const modelConfig = (
  inspection: Runtime.GgufInspection,
  architecture: string
): Registry.ModelConfig => {
  const prefix = `${architecture}.`
  const entries = inspection.metadata.map((entry) => ({
    source: entry.key,
    key: entry.key.startsWith(prefix)
      ? entry.key.slice(prefix.length)
      : entry.key.startsWith("general.")
      ? entry.key.slice("general.".length)
      : entry.key,
    value: entry.value
  }))
  const tokens = entries.find((entry) => entry.key === "tokenizer.ggml.tokens")?.value
  if (Array.isArray(tokens) && !entries.some((entry) => entry.key === "vocab_size")) {
    entries.push({ source: "tokenizer.ggml.tokens", key: "vocab_size", value: tokens.length })
  }
  entries.sort((left, right) => left.key.localeCompare(right.key) || left.source.localeCompare(right.source))
  const output = new Map<string, unknown>()
  for (const entry of entries) {
    if (entry.key.length === 0) {
      throw fail("validate", `GGUF metadata ${JSON.stringify(entry.source)} has an empty canonical key`)
    }
    if (output.has(entry.key)) {
      throw fail("validate", `GGUF metadata has duplicate canonical key ${JSON.stringify(entry.key)}`)
    }
    output.set(entry.key, entry.value)
  }
  if (output.get("architecture") !== architecture) {
    throw fail("validate", "GGUF canonical configuration does not preserve architecture")
  }
  return output
}

const validateCatalog = (model: Model.Model, tensors: ReadonlyArray<Runtime.GgufTensorDescriptor>): void => {
  if (model.parameters.length !== model.names.length || model.parameters.length !== tensors.length) {
    throw fail(
      "validate",
      `GGUF tensor catalog has ${tensors.length} entries but model requires ${model.parameters.length}`
    )
  }
  const catalog = new Map(tensors.map((tensor) => [tensor.name, tensor]))
  for (const parameter of model.parameters) {
    const tensor = catalog.get(parameter.name)
    if (tensor === undefined) {
      throw fail("validate", `GGUF is missing model parameter ${JSON.stringify(parameter.name)}`)
    }
    if (!sameShape(tensor.logicalShape, parameter.shape)) {
      throw fail(
        "validate",
        `GGUF parameter ${
          JSON.stringify(parameter.name)
        } has shape [${tensor.logicalShape}], expected [${parameter.shape}]`
      )
    }
    catalog.delete(parameter.name)
  }
  if (catalog.size !== 0) {
    throw fail("validate", `GGUF contains unexpected model parameter ${JSON.stringify(catalog.keys().next().value)}`)
  }
}

const expectedStorage = (
  descriptor: Runtime.GgufTensorDescriptor
): Runtime.EncodedTensorStorage | undefined =>
  descriptor.format === "F32"
    ? undefined
    : {
      encoding: descriptor.format,
      physicalShape: descriptor.physicalShape,
      physicalDtype: "u8"
    }

const validateTensor = (
  runtime: Runtime.RuntimeService,
  descriptor: Runtime.GgufTensorDescriptor,
  tensor: Runtime.ConcreteTensorHandle
): void => {
  if (typeof tensor !== "object" || tensor === null) {
    throw fail("validate", `native GGUF tensor ${JSON.stringify(descriptor.name)} is invalid`)
  }
  const storage = expectedStorage(descriptor)
  const actual = tensor.storage
  if (
    tensor._tag !== "Tensor" || tensor.dtype !== "f32" ||
    tensor.device !== runtime.placement.deviceType || tensor.placement.id !== runtime.placement.id ||
    !sameShape(tensor.shape, descriptor.logicalShape) ||
    (storage === undefined
      ? actual !== undefined
      : actual === undefined || actual.encoding !== storage.encoding || actual.physicalDtype !== "u8" ||
        !sameShape(actual.physicalShape, storage.physicalShape))
  ) {
    throw fail("validate", `native GGUF tensor ${JSON.stringify(descriptor.name)} has invalid logical metadata`)
  }
}

const clearLoaded = (
  runtime: Runtime.RuntimeService,
  entries: ReadonlyArray<unknown>
): Effect.Effect<void> => {
  const seen = new Set<object>()
  const tensors: Array<Runtime.ConcreteTensorHandle> = []
  for (const entry of entries) {
    if (
      typeof entry === "object" && entry !== null && "tensor" in entry &&
      typeof entry.tensor === "object" && entry.tensor !== null && !seen.has(entry.tensor)
    ) {
      seen.add(entry.tensor)
      tensors.push(entry.tensor as Runtime.ConcreteTensorHandle)
    }
  }
  return Effect.forEach(tensors, (tensor) => Effect.ignore(runtime.release(tensor)), { discard: true })
}

/**
 * Inspects, validates, and loads one native GGUF v3 file. Inspection happens
 * first without payload materialization. `general.architecture` must be a
 * non-empty string and resolves only the exact registry key
 * `gguf:<architecture>`. Canonical metadata strips that architecture prefix and
 * `general.`, derives `vocab_size` from tokenizer tokens when absent, and
 * rejects empty or colliding canonical keys before calling the architecture's
 * `create` effect.
 *
 * The inspected tensor catalog must exactly match the resulting model's names
 * and logical shapes. A second native operation then loads every payload. This
 * module supports dense F32 and GGML K-quant Q2_K through Q6_K descriptors;
 * all returned handles are logically f32, while quantized handles retain
 * encoded u8 storage metadata and are usable only by operations that support
 * that encoding. Inspection and loading are separate path reads: loaded
 * descriptors are compared with the inspected catalog, but metadata is not
 * returned by the load operation, so the caller must keep the file stable
 * between phases.
 *
 * Before the native `load` effect completes, the runtime owns partial results
 * and is responsible for interruption cleanup. Ownership transfers with a
 * successful archive. On validation failure or interruption after validation
 * begins, this function attempts to release every distinct returned handle,
 * ignores release failures so the original exit is preserved, and returns no
 * tensors. On success ownership of every parameter transfers to the caller;
 * release each handle when no longer needed. Inspection and load backend
 * failures are {@link GgufError}s, exact-key lookup failures are
 * {@link Registry.RegistryError}s, and architecture construction failures are
 * `Model.ModelError`s.
 *
 * @since 0.1.0
 * @category loading
 */
export const load = (
  path: string
): Effect.Effect<
  LoadedModel,
  GgufError | Registry.RegistryError | Model.ModelError,
  Runtime.Runtime | Registry.Registry
> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const registry = yield* Registry.Registry
    const gguf = runtime.extensions.gguf
    const inspected = yield* fromBackend("inspect", gguf.inspect(path))
    const inspection = yield* validateEffect(() => validateInspection(inspected))
    const architectureEntry = inspection.metadata.find((entry) => entry.key === "general.architecture")
    if (typeof architectureEntry?.value !== "string" || architectureEntry.value.length === 0) {
      return yield* fail("validate", "GGUF general.architecture must be a non-empty string")
    }
    const architecture = architectureEntry.value
    const implementation = yield* registry.get(`gguf:${architecture}`)
    const config = yield* validateEffect(() => modelConfig(inspection, architecture))
    const model = yield* implementation.create(config)
    yield* validateEffect(() => validateCatalog(model, inspection.tensors))
    return yield* Effect.flatMap(fromBackend("load", gguf.load(path)), (archive) => {
      const validArchive = typeof archive === "object" && archive !== null && Array.isArray(archive.entries)
      const entries = validArchive ? archive.entries : []
      const validated = validateEffect(() => {
        if (!validArchive) {
          throw fail("validate", "native GGUF load returned an invalid archive")
        }
        if (entries.length !== inspection.tensors.length) {
          throw fail("validate", "loaded GGUF tensor count differs from inspection")
        }
        const owned = new Set<Runtime.ConcreteTensorHandle>()
        for (const entry of entries) {
          if (typeof entry === "object" && entry !== null && owned.has(entry.tensor)) {
            throw fail("validate", "loaded GGUF archive contains duplicate tensor ownership")
          }
          if (
            typeof entry === "object" && entry !== null && typeof entry.tensor === "object" && entry.tensor !== null
          ) {
            owned.add(entry.tensor)
          }
        }
        const inspected = new Map(inspection.tensors.map((descriptor) => [descriptor.name, descriptor]))
        const loaded = new Map<string, Runtime.ConcreteTensorHandle>()
        for (const entry of entries) {
          const descriptor = validateDescriptor(entry.descriptor)
          const expected = inspected.get(descriptor.name)
          if (expected === undefined || !descriptorEqual(descriptor, expected) || loaded.has(descriptor.name)) {
            throw fail(
              "validate",
              `loaded GGUF descriptor for ${JSON.stringify(descriptor.name)} differs from inspection`
            )
          }
          validateTensor(runtime, descriptor, entry.tensor)
          loaded.set(descriptor.name, entry.tensor)
        }
        const params = model.parameters.map((parameter) => loaded.get(parameter.name)!)
        if (params.some((tensor) => tensor === undefined)) {
          throw fail("validate", "loaded GGUF parameter bijection failed")
        }
        return { model, params, metadata: config } satisfies LoadedModel
      })
      return Effect.onExit(
        validated,
        (exit) => Exit.isFailure(exit) ? clearLoaded(runtime, entries) : Effect.void
      )
    })
  })

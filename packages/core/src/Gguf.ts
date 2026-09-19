/**
 * Inspects and loads native GGUF v3 files. The selected runtime parses the file
 * and creates tensor handles. This module canonicalizes metadata and validates
 * the architecture expected by the caller. It builds the model or parameter
 * catalog, then checks the inspected and loaded tensor catalogs for exact
 * matches.
 *
 * @since 0.1.0
 */
import { Data, Effect, Exit, Predicate } from "effect"
import type * as Model from "./Model.ts"
import * as Runtime from "./Runtime.ts"
import type * as Tensor from "./Tensor.ts"

/**
 * A native GGUF inspection/loading failure or a structural validation failure.
 * Model construction failures remain `Model.ModelError`s.
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
 * Canonical architecture configuration produced from GGUF metadata.
 *
 * @since 0.1.0
 * @category models
 */
export type ModelConfig = ReadonlyMap<string, unknown>

/**
 * An explicitly selected GGUF model definition. Model modules provide one
 * definition to their dedicated loader; there is no ambient registry lookup.
 *
 * @since 0.1.0
 * @category models
 */
export interface ModelDefinition {
  /** Exact value required for `general.architecture`. */
  readonly architecture: string
  /** Constructs a model template from canonical metadata. */
  readonly create: (metadata: ModelConfig) => Effect.Effect<Model.Model, Model.ModelError>
}

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
   * Caller-owned loaded tensors in `model.parameterSpecs` order. Release each
   * handle when no longer needed; repeated releases are no-ops.
   */
  readonly params: ReadonlyArray<Tensor.Concrete>
  /**
   * Canonical configuration passed to the architecture. The architecture
   * prefix and `general.` are stripped, other keys remain, and
   * `vocab_size` may be derived from `tokenizer.ggml.tokens`. This is not the
   * raw ordered GGUF metadata table; callers should treat the map and any array
   * values as immutable. Original numeric kinds are erased to JavaScript
   * numbers by the runtime boundary, so 64-bit integer values may already have
   * lost precision.
   */
  readonly metadata: ReadonlyMap<string, unknown>
}

/**
 * One required GGUF tensor's stable archive name and logical shape.
 *
 * @since 0.1.0
 * @category models
 */
export interface TensorSpec {
  /** Exact tensor name expected in the archive. */
  readonly name: string
  /** Logical tensor dimensions expected by the artifact. */
  readonly shape: ReadonlyArray<number>
}

/**
 * Loaded GGUF parameters for an artifact that is not an ordinary one-input
 * model.
 *
 * @since 0.1.0
 * @category models
 */
export interface LoadedParameters {
  /** Validated parameter specifications in the same order as `params`. */
  readonly parameterSpecs: ReadonlyArray<TensorSpec>
  /** Caller-owned tensors in the requested parameter-catalog order. */
  readonly params: ReadonlyArray<Tensor.Concrete>
  /** Canonical metadata using the same normalization as {@link loadModel}. */
  readonly metadata: ReadonlyMap<string, unknown>
}

/**
 * Definition used by {@link loadParameters} after inspection and metadata
 * normalization.
 *
 * @since 0.1.0
 * @category models
 */
export interface ParameterArtifactDefinition {
  /** Exact value required for `general.architecture`. */
  readonly architecture: string
  /** Builds the exact tensor catalog from canonical metadata and inspected descriptors. */
  readonly parameterSpecs: (
    metadata: ReadonlyMap<string, unknown>,
    tensors: ReadonlyArray<Runtime.GgufTensorDescriptor>
  ) => Effect.Effect<ReadonlyArray<TensorSpec>, Model.ModelError>
}

/**
 * Selection for {@link loadParameters}. Names must belong to the artifact's
 * parameter catalog. Selected tensors are returned in selection order.
 *
 * @since 0.1.0
 * @category models
 */
export type LoadOptions = Runtime.GgufLoadOptions

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

const validateDescriptor = (value: Runtime.GgufTensorDescriptor): Runtime.GgufTensorDescriptor => {
  if (!Predicate.isObjectOrArray(value) || !Predicate.isString(value.name) || value.name.length === 0) {
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
    !Runtime.isTensorStorageEncoding(value.format) || value.physicalDtype !== "u8" ||
    !validShape(value.physicalShape, 2)
  ) {
    throw fail("validate", `GGUF tensor ${JSON.stringify(value.name)} has invalid encoded storage metadata`)
  } else {
    if (
      !Runtime.validEncodedStorage(value.logicalShape, {
        encoding: value.format,
        physicalShape: value.physicalShape,
        physicalDtype: value.physicalDtype
      })
    ) {
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
  const inspection = value
  if (
    !Predicate.isObjectOrArray(value) || !Array.isArray(value.metadata) || !Array.isArray(value.tensors)
  ) {
    throw fail("validate", "native GGUF inspection has an invalid structure")
  }
  const metadataKeys = new Set<string>()
  const metadata = inspection.metadata.map((entry) => {
    const metadataEntry = entry
    if (!Predicate.isObjectOrArray(entry) || !Predicate.isString(entry.key) || entry.key.length === 0) {
      throw fail("validate", "GGUF metadata contains an invalid key")
    }
    if (metadataKeys.has(metadataEntry.key)) {
      throw fail("validate", `duplicate GGUF metadata key ${JSON.stringify(metadataEntry.key)}`)
    }
    metadataKeys.add(metadataEntry.key)
    const entryValue = metadataEntry.value
    if (!isScalar(entryValue) && !(Array.isArray(entryValue) && entryValue.every(isScalar))) {
      throw fail("validate", `GGUF metadata ${JSON.stringify(metadataEntry.key)} has an invalid value`)
    }
    return Object.freeze({
      key: metadataEntry.key,
      value: Array.isArray(entryValue) ? Object.freeze([...entryValue]) : entryValue
    })
  })
  const names = new Set<string>()
  const tensors = inspection.tensors.map((descriptor) => {
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
): ModelConfig => {
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

const validateCatalog = (
  parameterSpecs: ReadonlyArray<TensorSpec>,
  tensors: ReadonlyArray<Runtime.GgufTensorDescriptor>
): void => {
  if (parameterSpecs.length !== tensors.length) {
    throw fail(
      "validate",
      `GGUF tensor catalog has ${tensors.length} entries but artifact requires ${parameterSpecs.length}`
    )
  }
  const catalog = new Map(tensors.map((tensor) => [tensor.name, tensor]))
  for (const parameter of parameterSpecs) {
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

const snapshotParameters = (parameters: ReadonlyArray<TensorSpec>): ReadonlyArray<TensorSpec> => {
  if (!Array.isArray(parameters)) throw fail("validate", "GGUF parameter catalog must be an array")
  const names = new Set<string>()
  return Object.freeze(Array.from(parameters, (parameter: TensorSpec) => {
    if (
      !Predicate.isObjectOrArray(parameter) || !Predicate.isString(parameter.name) || parameter.name.length === 0 ||
      names.has(parameter.name) || !Array.isArray(parameter.shape)
    ) {
      throw fail("validate", "GGUF parameter catalog must have unique nonempty names and valid shapes")
    }
    const shape = Array.from(parameter.shape)
    if (shape.length < 1 || shape.length > 4 || !shape.every((dim) => Number.isSafeInteger(dim) && dim > 0)) {
      throw fail("validate", `GGUF parameter ${JSON.stringify(parameter.name)} has invalid dimensions`)
    }
    names.add(parameter.name)
    return Object.freeze({ name: parameter.name, shape: Object.freeze(shape) })
  }))
}

const snapshotNames = (names: ReadonlyArray<string> | undefined): ReadonlyArray<string> | undefined => {
  if (names === undefined) return undefined
  if (!Array.isArray(names)) throw fail("validate", "GGUF names must be an array of unique nonempty strings")
  const selected = Array.from(names)
  if (
    selected.some((name) => !Predicate.isString(name) || name.length === 0) ||
    new Set(selected).size !== selected.length
  ) {
    throw fail("validate", "GGUF names must be unique nonempty strings")
  }
  return Object.freeze(selected)
}

const loadArchive = (
  path: string,
  runtime: Runtime.RuntimeService,
  descriptors: ReadonlyArray<Runtime.GgufTensorDescriptor>,
  parameterSpecs: ReadonlyArray<TensorSpec>,
  metadata: ReadonlyMap<string, unknown>,
  names?: ReadonlyArray<string>
): Effect.Effect<LoadedParameters, GgufError> =>
  Effect.flatMap(
    fromBackend("load", runtime.extensions.gguf.load(path, names === undefined ? {} : { names })),
    (archive) => {
      const loadedArchive = archive
      const validArchive = Predicate.isObjectOrArray(archive) && Array.isArray(archive.entries)
      const entries = validArchive ? loadedArchive.entries : []
      const validated = validateEffect(() => {
        if (!validArchive) throw fail("validate", "native GGUF load returned an invalid archive")
        if (entries.length !== descriptors.length) {
          throw fail("validate", "loaded GGUF tensor count differs from inspection")
        }
        const owned = new Set<Runtime.ConcreteTensorHandle>()
        for (const entry of entries) {
          const loadedEntry = entry
          if (Predicate.isObjectOrArray(entry) && owned.has(loadedEntry.tensor)) {
            throw fail("validate", "loaded GGUF archive contains duplicate tensor ownership")
          }
          if (
            Predicate.isObjectOrArray(entry) && Predicate.isObjectOrArray(loadedEntry.tensor)
          ) {
            owned.add(loadedEntry.tensor)
          }
        }
        const inspected = new Map(descriptors.map((descriptor) => [descriptor.name, descriptor]))
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
          loaded.set(descriptor.name, entry.tensor)
        }
        const params = parameterSpecs.map((parameter) => loaded.get(parameter.name))
        if (params.some((tensor) => tensor === undefined)) {
          throw fail("validate", "loaded GGUF parameter bijection failed")
        }
        return {
          parameterSpecs,
          params: params.filter((tensor) => tensor !== undefined),
          metadata
        } satisfies LoadedParameters
      })
      return Effect.onExit(validated, (exit) => Exit.isFailure(exit) ? clearLoaded(runtime, entries) : Effect.void)
    }
  )

/**
 * Loads GGUF parameters without constructing a {@link Model.Model}. Omitting
 * `options.names` requires an exact full-file catalog and loads every tensor.
 * Supplying names validates and loads only those catalog entries, in selection
 * order. Missing or duplicate names fail before payload loading. An empty
 * selection loads no tensors. Unselected payloads are never read or allocated.
 * This loader supports target-coupled checkpoints such as DFlash. It follows
 * the same two-read and ownership rules as {@link loadModel} for inspection,
 * architecture validation, catalog construction, and loading. On success, the
 * caller owns every tensor in `params`; on post-load validation failure every
 * discoverable returned handle receives a best-effort release attempt.
 *
 * @since 0.1.0
 * @category loading
 */
export const loadParameters = (
  path: string,
  definition: ParameterArtifactDefinition,
  options: LoadOptions = {}
): Effect.Effect<LoadedParameters, GgufError | Model.ModelError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const names = yield* validateEffect(() => snapshotNames(options.names))
    const runtime = yield* Runtime.Runtime
    const inspected = yield* fromBackend("inspect", runtime.extensions.gguf.inspect(path))
    const inspection = yield* validateEffect(() => validateInspection(inspected))
    const architectureEntry = inspection.metadata.find((entry) => entry.key === "general.architecture")
    if (architectureEntry?.value !== definition.architecture) {
      return yield* fail(
        "validate",
        `GGUF general.architecture must be exactly ${JSON.stringify(definition.architecture)}`
      )
    }
    const metadata = yield* validateEffect(() => modelConfig(inspection, definition.architecture))
    const catalog = yield* definition.parameterSpecs(metadata, inspection.tensors)
    const selection = yield* validateEffect(() => {
      const parameters = snapshotParameters(catalog)
      if (names === undefined) {
        validateCatalog(parameters, inspection.tensors)
        return { parameters, descriptors: inspection.tensors }
      }
      const byName = new Map(parameters.map((parameter) => [parameter.name, parameter]))
      const selected = Object.freeze(names.map((name) => {
        const parameter = byName.get(name)
        if (parameter === undefined) {
          throw fail("validate", `GGUF parameter catalog has no entry ${JSON.stringify(name)}`)
        }
        return parameter
      }))
      const inspectedByName = new Map(inspection.tensors.map((tensor) => [tensor.name, tensor]))
      const descriptors = names.map((name) => {
        const descriptor = inspectedByName.get(name)
        if (descriptor === undefined) throw fail("validate", `GGUF is missing model parameter ${JSON.stringify(name)}`)
        return descriptor
      })
      validateCatalog(selected, descriptors)
      return { parameters: selected, descriptors }
    })
    return yield* loadArchive(path, runtime, selection.descriptors, selection.parameters, metadata, names)
  })

const clearLoaded = (
  runtime: Runtime.RuntimeService,
  entries: ReadonlyArray<Runtime.GgufLoadEntry>
): Effect.Effect<void> => {
  const seen = new Set<object>()
  const tensors: Array<Runtime.ConcreteTensorHandle> = []
  for (const entry of entries) {
    if (
      Predicate.isObjectOrArray(entry) && "tensor" in entry &&
      Predicate.isObjectOrArray(entry.tensor) && !seen.has(entry.tensor)
    ) {
      seen.add(entry.tensor)
      tensors.push(entry.tensor)
    }
  }
  return Effect.forEach(tensors, (tensor) => Effect.ignore(runtime.release(tensor)), { discard: true })
}

/**
 * Inspects, validates, and loads one native GGUF v3 file for an explicitly
 * selected model definition. Inspection happens first without payload
 * materialization. `general.architecture` must equal the definition's exact
 * architecture value. Canonical metadata strips that architecture prefix and
 * `general.`, derives `vocab_size` from tokenizer tokens when absent, and
 * rejects empty or colliding canonical keys before calling `create`.
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
 * and handles interruption cleanup. When it returns an archive, ownership
 * transfers to this function. On validation failure or interruption after
 * validation begins, this function attempts to release every distinct returned
 * handle. It ignores release failures to preserve the original exit and returns
 * no tensors. On success, the caller owns every parameter. Release each handle
 * when no longer needed. Inspection, loading, and
 * architecture mismatches are {@link GgufError}s; model construction failures
 * are `Model.ModelError`s.
 *
 * @since 0.1.0
 * @category loading
 */
export const loadModel = (
  path: string,
  definition: ModelDefinition
): Effect.Effect<LoadedModel, GgufError | Model.ModelError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const gguf = runtime.extensions.gguf
    const inspected = yield* fromBackend("inspect", gguf.inspect(path))
    const inspection = yield* validateEffect(() => validateInspection(inspected))
    const architectureEntry = inspection.metadata.find((entry) => entry.key === "general.architecture")
    if (architectureEntry?.value !== definition.architecture) {
      return yield* fail(
        "validate",
        `GGUF general.architecture must be exactly ${JSON.stringify(definition.architecture)}`
      )
    }
    const config = yield* validateEffect(() => modelConfig(inspection, definition.architecture))
    const model = yield* definition.create(config)
    yield* validateEffect(() => validateCatalog(model.parameterSpecs, inspection.tensors))
    const parameters = yield* validateEffect(() => snapshotParameters(model.parameterSpecs))
    const loaded = yield* loadArchive(path, runtime, inspection.tensors, parameters, config)
    return { model, params: loaded.params, metadata: loaded.metadata }
  })

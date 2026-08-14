/**
 * Effect-environment registration and exact-key lookup for model architecture
 * factories. A registry is a mutable in-memory map supplied as an Effect
 * service; Layers determine which map is visible and can install registrations
 * for a scope. Nothing is process-global or persisted.
 *
 * @since 0.1.0
 */
import { Context, Data, Effect, Layer } from "effect"
import type * as Model from "./Model.ts"
import { MuseGlimmer } from "./models/index.ts"

/**
 * Canonical architecture configuration values. Producers define the key
 * normalization rules; the registry stores and forwards the map unchanged.
 *
 * @since 0.1.0
 * @category models
 */
export type ModelConfig = ReadonlyMap<string, unknown>

/**
 * Installed model code that constructs a model template from configuration.
 * Registrations are stored by object reference rather than cloned or frozen.
 *
 * @since 0.1.0
 * @category models
 */
export interface ModelArchitecture {
  /**
   * The exact source-qualified architecture identifier, such as
   * `gguf:muse-glimmer`.
   *
   * @since 0.1.0
   */
  readonly id: string
  /**
   * Constructs a model template from canonical architecture configuration.
   *
   * @since 0.1.0
   */
  readonly create: (config: ModelConfig) => Effect.Effect<Model.Model, Model.ModelError>
}

/**
 * A failed model-architecture registration or lookup.
 *
 * @since 0.1.0
 * @category errors
 */
export class RegistryError extends Data.TaggedError("RegistryError")<{
  /** The registry operation that failed. */
  readonly op: "register" | "get"
  /** A human-readable description of the failure. */
  readonly message: string
}> {}

/**
 * Exact-key model architecture registration and lookup operations.
 *
 * @since 0.1.0
 * @category models
 */
export interface RegistryService {
  /**
   * Registers an architecture under its exact, case-sensitive `id`, returning
   * whether it was newly inserted. The first registration wins; a duplicate
   * returns `false` and does not replace it. Only the empty string is rejected.
   *
   * @since 0.1.0
   */
  readonly register: (architecture: ModelArchitecture) => Effect.Effect<boolean, RegistryError>
  /**
   * Idempotently removes an architecture by exact identifier. When `expected`
   * is supplied, removal occurs only if that exact object is still registered;
   * scoped Layers use this guard so their finalizers cannot delete a later
   * replacement.
   *
   * @since 0.1.0
   */
  readonly unregister: (id: string, expected?: ModelArchitecture) => Effect.Effect<void>
  /**
   * Looks up an architecture by its exact, case-sensitive identifier.
   *
   * @since 0.1.0
   */
  readonly get: (id: string) => Effect.Effect<ModelArchitecture, RegistryError>
}

/**
 * The model architecture registry in the current Effect environment. Service
 * identity determines isolation: programs sharing one service share its
 * mutations.
 *
 * @since 0.1.0
 * @category services
 */
export class Registry extends Context.Service<Registry, RegistryService>()(
  "@effect-torch/core/Registry"
) {}

const RegistryInternalId = Symbol("@effect-torch/core/Registry/internal")

interface RegistryInternal {
  readonly [RegistryInternalId]: {
    readonly register: (
      architecture: ModelArchitecture
    ) => Effect.Effect<{ readonly id: string; readonly token: object } | undefined, RegistryError>
    readonly unregister: (id: string, token: object) => Effect.Effect<void>
  }
}

const makeService = Effect.sync(() => {
  const architectures = new Map<string, { readonly architecture: ModelArchitecture; readonly token: object }>()
  const register = (architecture: ModelArchitecture, token: object) =>
    Effect.suspend(() => {
      const id = architecture.id
      if (id === "") {
        return new RegistryError({ op: "register", message: "architecture id must not be empty" })
      }
      if (architectures.has(id)) return Effect.succeed(false)
      architectures.set(id, { architecture, token })
      return Effect.succeed(true)
    })
  return {
    register: (architecture) => register(architecture, {}),
    unregister: (id, expected) =>
      Effect.sync(() => {
        if (expected === undefined || architectures.get(id)?.architecture === expected) {
          architectures.delete(id)
        }
      }),
    get: (id) =>
      Effect.suspend(() => {
        const entry = architectures.get(id)
        return entry === undefined
          ? new RegistryError({ op: "get", message: `architecture ${id} is not registered` })
          : Effect.succeed(entry.architecture)
      }),
    [RegistryInternalId]: {
      register: (architecture) => {
        const id = architecture.id
        const token = {}
        return Effect.map(register(architecture, token), (inserted) => inserted ? { id, token } : undefined)
      },
      unregister: (id, token) =>
        Effect.sync(() => {
          if (architectures.get(id)?.token === token) architectures.delete(id)
        })
    }
  } satisfies RegistryService & RegistryInternal
})

/**
 * Provides one empty, mutable architecture map for each materialization of
 * this Layer. Layer memoization may share that service within one Layer graph;
 * explicitly built/provided instances are isolated.
 *
 * @since 0.1.0
 * @category layers
 */
export const emptyLayer: Layer.Layer<Registry> = Layer.effect(Registry, makeService)

/**
 * A registration Layer that requires an existing {@link Registry} service and
 * installs architectures into that shared map for its scope. It does not
 * create or provide a registry by itself.
 *
 * Existing registrations are preserved, and only identifiers newly inserted
 * during acquisition are recorded for removal when the Layer scope closes.
 * Registration is not reference counted: if overlapping scopes request the
 * same id, the scope that inserted it remains the sole owner and must outlive
 * scopes that observed the duplicate. Finalizers compare the current
 * registration by an internal insertion token, so they do not delete a
 * replacement installed after explicit unregister, even when the same object is
 * registered again. If a later registration fails, the finalizer removes
 * earlier inserts that it still owns.
 *
 * @since 0.1.0
 * @category layers
 */
export const registerAll = (
  ...architectures: ReadonlyArray<ModelArchitecture>
): Layer.Layer<never, RegistryError, Registry> =>
  Layer.effectDiscard(
    Effect.gen(function*() {
      const registry = yield* Registry
      const internal = (registry as RegistryService & Partial<RegistryInternal>)[RegistryInternalId]
      const releases: Array<Effect.Effect<void>> = []
      yield* Effect.addFinalizer(() =>
        Effect.gen(function*() {
          yield* Effect.forEach(releases, (release) => release, { discard: true })
        })
      )
      for (const architecture of architectures) {
        if (internal !== undefined) {
          const registration = yield* internal.register(architecture)
          if (registration !== undefined) releases.push(internal.unregister(registration.id, registration.token))
        } else {
          const id = architecture.id
          if (yield* registry.register(architecture)) releases.push(registry.unregister(id, architecture))
        }
      }
    })
  )

/**
 * Provides a fresh registry merged with scoped registration of every built-in
 * model architecture. Build/provide this Layer to use built-ins; providing
 * {@link emptyLayer} alone intentionally contains none.
 *
 * @since 0.1.0
 * @category layers
 */
export const layer = registerAll(MuseGlimmer.architecture).pipe(Layer.provideMerge(emptyLayer))

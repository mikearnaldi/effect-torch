/**
 * Registration, exact-key lookup, and default Layers for model architectures.
 *
 * @since 0.1.0
 */
import { Context, Data, Effect, Layer } from "effect"
import type * as Model from "./Model.ts"
import { MuseGlimmer } from "./models/index.ts"

/**
 * Canonical architecture configuration values.
 *
 * @since 0.1.0
 * @category models
 */
export type ModelConfig = ReadonlyMap<string, unknown>

/**
 * Installed model code that constructs a model template from configuration.
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
   * Registers an architecture, returning whether it was newly inserted.
   *
   * @since 0.1.0
   */
  readonly register: (architecture: ModelArchitecture) => Effect.Effect<boolean, RegistryError>
  /**
   * Removes an architecture if it is registered.
   *
   * @since 0.1.0
   */
  readonly unregister: (id: string) => Effect.Effect<void>
  /**
   * Looks up an architecture by its exact identifier.
   *
   * @since 0.1.0
   */
  readonly get: (id: string) => Effect.Effect<ModelArchitecture, RegistryError>
}

/**
 * The model architecture registry for the current Effect program.
 *
 * @since 0.1.0
 * @category services
 */
export class Registry extends Context.Service<Registry, RegistryService>()(
  "@effect-torch/core/Registry"
) {}

const makeService = Effect.sync(() => {
  const architectures = new Map<string, ModelArchitecture>()
  return {
    register: (architecture) =>
      Effect.suspend(() => {
        if (architecture.id === "") {
          return new RegistryError({ op: "register", message: "architecture id must not be empty" })
        }
        if (architectures.has(architecture.id)) {
          return Effect.succeed(false)
        }
        architectures.set(architecture.id, architecture)
        return Effect.succeed(true)
      }),
    unregister: (id) => Effect.sync(() => void architectures.delete(id)),
    get: (id) =>
      Effect.suspend(() => {
        const architecture = architectures.get(id)
        return architecture === undefined
          ? new RegistryError({ op: "get", message: `architecture ${id} is not registered` })
          : Effect.succeed(architecture)
      })
  } satisfies RegistryService
})

/**
 * Provides one empty, mutable architecture map.
 *
 * @since 0.1.0
 * @category layers
 */
export const emptyLayer: Layer.Layer<Registry> = Layer.effect(Registry, makeService)

/**
 * Registers architectures for the lifetime of the resulting Layer.
 *
 * Existing registrations are preserved and only newly inserted architectures
 * are removed when the Layer scope closes.
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
      const registered: Array<string> = []
      yield* Effect.addFinalizer(() =>
        Effect.gen(function*() {
          for (const id of registered) {
            yield* registry.unregister(id)
          }
        })
      )
      for (const architecture of architectures) {
        if (yield* registry.register(architecture)) {
          registered.push(architecture.id)
        }
      }
    })
  )

/**
 * Provides the default registry containing every built-in model architecture.
 *
 * @since 0.1.0
 * @category layers
 */
export const layer = registerAll(MuseGlimmer.architecture).pipe(Layer.provideMerge(emptyLayer))

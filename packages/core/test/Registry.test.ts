import { expect, it } from "@effect/vitest"
import { Context, Effect, Layer } from "effect"
import { Model, Registry, type Tensor } from "../src/index.ts"

const architecture = (id: string): Registry.ModelArchitecture => ({
  id,
  create: () =>
    Model.define({
      parameters: [],
      forward: (_, input) => Effect.succeed(input as Tensor.Lazy)
    })
})

it.effect("registers, gets, and idempotently unregisters exact IDs", () =>
  Effect.gen(function*() {
    const registry = yield* Registry.Registry
    const muse = architecture("gguf:muse-glimmer")
    expect(yield* registry.register(muse)).toBe(true)
    expect(yield* registry.get("gguf:muse-glimmer")).toBe(muse)
    expect((yield* Effect.flip(registry.get("Muse-Glimmer"))).op).toBe("get")
    yield* registry.unregister("gguf:muse-glimmer")
    yield* registry.unregister("gguf:muse-glimmer")
    expect((yield* Effect.flip(registry.get("gguf:muse-glimmer")))._tag).toBe("RegistryError")
  }).pipe(Effect.provide(Registry.emptyLayer)))

it.effect("rejects empty IDs and reports duplicate registrations", () =>
  Effect.gen(function*() {
    const registry = yield* Registry.Registry
    expect((yield* Effect.flip(registry.register(architecture("")))).op).toBe("register")
    const spaced = architecture(" ")
    expect(yield* registry.register(spaced)).toBe(true)
    expect(yield* registry.get(" ")).toBe(spaced)
    expect(yield* registry.register(architecture(" "))).toBe(false)
  }).pipe(Effect.provide(Registry.emptyLayer)))

// registerAll owns only registrations it actually inserted. Scope finalization
// must remove those entries without consuming pre-existing shared-registry state.
it.effect("registerAll scopes every architecture in the shared registry", () =>
  Effect.scoped(
    Effect.gen(function*() {
      const context = yield* Layer.build(Registry.emptyLayer)
      const registry = Context.get(context, Registry.Registry)
      const first = architecture("first")
      const second = architecture("second")
      const registration = Registry.registerAll(first, second).pipe(
        Layer.provide(Layer.succeed(Registry.Registry, registry))
      )

      yield* Effect.scoped(
        Effect.gen(function*() {
          yield* Layer.build(registration)
          expect(yield* registry.get("first")).toBe(first)
          expect(yield* registry.get("second")).toBe(second)
        })
      )

      expect((yield* Effect.flip(registry.get("first")))._tag).toBe("RegistryError")
      expect((yield* Effect.flip(registry.get("second")))._tag).toBe("RegistryError")
    })
  ))

it.effect("registerAll preserves existing registrations and cleans up only new ones", () =>
  Effect.scoped(
    Effect.gen(function*() {
      const context = yield* Layer.build(Registry.emptyLayer)
      const registry = Context.get(context, Registry.Registry)
      const first = architecture("first")
      const duplicate = architecture("duplicate")
      expect(yield* registry.register(duplicate)).toBe(true)
      const registration = Registry.registerAll(first, duplicate).pipe(
        Layer.provide(Layer.succeed(Registry.Registry, registry))
      )

      yield* Effect.scoped(
        Effect.gen(function*() {
          yield* Layer.build(registration)
          expect(yield* registry.get("first")).toBe(first)
          expect(yield* registry.get("duplicate")).toBe(duplicate)
        })
      )
      expect((yield* Effect.flip(registry.get("first")))._tag).toBe("RegistryError")
      expect(yield* registry.get("duplicate")).toBe(duplicate)
    })
  ))

it.effect("registerAll finalization preserves a later replacement", () =>
  Effect.scoped(
    Effect.gen(function*() {
      const context = yield* Layer.build(Registry.emptyLayer)
      const registry = Context.get(context, Registry.Registry)
      const first = architecture("shared")
      const replacement = architecture("shared")
      const registration = Registry.registerAll(first).pipe(
        Layer.provide(Layer.succeed(Registry.Registry, registry))
      )

      yield* Effect.scoped(
        Effect.gen(function*() {
          yield* Layer.build(registration)
          yield* registry.unregister("shared")
          expect(yield* registry.register(replacement)).toBe(true)
        })
      )

      expect(yield* registry.get("shared")).toBe(replacement)
    })
  ))

it.effect("registerAll finalization does not remove an ABA replacement", () =>
  Effect.scoped(
    Effect.gen(function*() {
      const context = yield* Layer.build(Registry.emptyLayer)
      const registry = Context.get(context, Registry.Registry)
      const shared = architecture("shared")
      const registration = Registry.registerAll(shared).pipe(
        Layer.provide(Layer.succeed(Registry.Registry, registry))
      )

      yield* Effect.scoped(
        Effect.gen(function*() {
          yield* Layer.build(registration)
          yield* registry.unregister("shared")
          expect(yield* registry.register(shared)).toBe(true)
        })
      )

      expect(yield* registry.get("shared")).toBe(shared)
    })
  ))

it.effect("registerAll finalization retains the insertion key when the object mutates", () =>
  Effect.scoped(
    Effect.gen(function*() {
      const context = yield* Layer.build(Registry.emptyLayer)
      const registry = Context.get(context, Registry.Registry)
      const mutable = architecture("first") as { id: string } & Registry.ModelArchitecture
      const registration = Registry.registerAll(mutable).pipe(
        Layer.provide(Layer.succeed(Registry.Registry, registry))
      )

      yield* Effect.scoped(
        Effect.gen(function*() {
          yield* Layer.build(registration)
          mutable.id = "second"
        })
      )

      expect((yield* Effect.flip(registry.get("first")))._tag).toBe("RegistryError")
    })
  ))

it.effect("registerAll cleans up earlier registrations when an empty ID fails", () =>
  Effect.scoped(
    Effect.gen(function*() {
      const context = yield* Layer.build(Registry.emptyLayer)
      const registry = Context.get(context, Registry.Registry)
      const registration = Registry.registerAll(architecture("first"), architecture("")).pipe(
        Layer.provide(Layer.succeed(Registry.Registry, registry))
      )

      expect((yield* Effect.flip(Effect.scoped(Layer.build(registration)))).op).toBe("register")
      expect((yield* Effect.flip(registry.get("first")))._tag).toBe("RegistryError")
    })
  ))

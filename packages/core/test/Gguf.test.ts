import { expect, it } from "@effect/vitest"
import { Deferred, Effect, Fiber, Layer } from "effect"
import * as fs from "node:fs"
import * as os from "node:os"
import * as path from "node:path"
import { Gguf, Model, Registry, Runtime, Tensor } from "../src/index.ts"
import { onDevices } from "./utils/devices.ts"

const placement: Runtime.Placement = Object.freeze({
  id: "gguf-test:0",
  deviceType: "test",
  description: "GGUF test runtime"
})

// The fake runtime keeps logical model metadata separate from encoded storage
// geometry. Its object handles are ownership tokens, so release assertions use
// identity rather than descriptor equality.
const denseDescriptor: Runtime.GgufTensorDescriptor = Object.freeze({
  name: "dense",
  format: "F32",
  logicalShape: Object.freeze([2]),
  logicalDtype: "f32",
  physicalShape: Object.freeze([2]),
  physicalDtype: "f32"
})

const encodedDescriptor: Runtime.GgufTensorDescriptor = Object.freeze({
  name: "packed",
  format: "Q4_K",
  logicalShape: Object.freeze([2, 256]),
  logicalDtype: "f32",
  physicalShape: Object.freeze([2, 144]),
  physicalDtype: "u8"
})

const tensor = (descriptor: Runtime.GgufTensorDescriptor): Tensor.Concrete =>
  Object.freeze({
    _tag: "Tensor",
    shape: descriptor.logicalShape,
    dtype: "f32",
    ...(descriptor.format === "F32"
      ? {}
      : {
        storage: Object.freeze({
          encoding: descriptor.format,
          physicalShape: descriptor.physicalShape,
          physicalDtype: "u8" as const
        })
      }),
    device: placement.deviceType,
    placement,
    pipe() {
      throw new Error("unused test handle pipe")
    }
  }) as unknown as Tensor.Concrete

const inspection: Runtime.GgufInspection = Object.freeze({
  metadata: Object.freeze([
    Object.freeze({ key: "general.architecture", value: "test-model" }),
    Object.freeze({ key: "test-model.context_length", value: 32 }),
    Object.freeze({ key: "general.name", value: "fixture" })
  ]),
  tensors: Object.freeze([encodedDescriptor, denseDescriptor])
})

const architecture = (
  capture: (config: Registry.ModelConfig) => void
): Registry.ModelArchitecture => ({
  id: "gguf:test-model",
  create: (config) => {
    capture(config)
    return Model.define({
      parameters: [
        { name: "dense", shape: [2] },
        { name: "packed", shape: [2, 256] }
      ],
      forward: (_, input) => Effect.succeed(input as Tensor.Lazy)
    })
  }
})

const provide = (runtime: Runtime.RuntimeService) =>
  Layer.merge(Registry.emptyLayer, Layer.succeed(Runtime.Runtime, runtime))

const u32 = (value: number): Buffer => {
  const bytes = Buffer.alloc(4)
  bytes.writeUInt32LE(value)
  return bytes
}

const u64 = (value: number): Buffer => {
  const bytes = Buffer.alloc(8)
  bytes.writeBigUInt64LE(BigInt(value))
  return bytes
}

const string = (value: string): Buffer => {
  const bytes = Buffer.from(value)
  return Buffer.concat([u64(bytes.length), bytes])
}

// Minimal GGUF v3: one Q4_K tensor, three metadata entries, 32-byte alignment,
// and exactly the encoded physical payload required for logical shape [2, 256].
const fixture = (): Buffer => {
  const header = Buffer.concat([
    Buffer.from("GGUF"),
    u32(3),
    u64(1),
    u64(3),
    string("general.architecture"),
    u32(8),
    string("compiled-identity"),
    string("general.alignment"),
    u32(4),
    u32(32),
    string("general.quantization_version"),
    u32(4),
    u32(2),
    string("packed"),
    u32(2),
    u64(256),
    u64(2),
    u32(12),
    u64(0)
  ])
  const padding = (32 - header.length % 32) % 32
  return Buffer.concat([header, Buffer.alloc(padding), Buffer.alloc(2 * 144)])
}

it.effect("loads by exact architecture and returns tensors in model parameter order", () => {
  const dense = tensor(denseDescriptor)
  const packed = tensor(encodedDescriptor)
  const paths: Array<string> = []
  const released: Array<Tensor.Concrete> = []
  const runtime = {
    placement,
    extensions: {
      gguf: {
        inspect: (path: string) =>
          Effect.sync(() => {
            paths.push(`inspect:${path}`)
            return inspection
          }),
        load: (path: string) =>
          Effect.sync(() => {
            paths.push(`load:${path}`)
            return {
              entries: [
                { descriptor: encodedDescriptor, tensor: packed },
                { descriptor: denseDescriptor, tensor: dense }
              ]
            }
          })
      }
    },
    release: (value: Tensor.Concrete) => Effect.sync(() => void released.push(value))
  } as unknown as Runtime.RuntimeService
  let config: Registry.ModelConfig | undefined

  return Effect.gen(function*() {
    const registry = yield* Registry.Registry
    yield* registry.register(architecture((value) => config = value))
    const loaded = yield* Gguf.load("model.gguf")

    expect(paths).toEqual(["inspect:model.gguf", "load:model.gguf"])
    expect([...config!]).toEqual([
      ["architecture", "test-model"],
      ["context_length", 32],
      ["name", "fixture"]
    ])
    expect(loaded.model.names).toEqual(["dense", "packed"])
    expect(loaded.params).toEqual([dense, packed])
    expect(loaded.params[1].storage).toEqual({
      encoding: "Q4_K",
      physicalShape: [2, 144],
      physicalDtype: "u8"
    })
    expect(released).toEqual([])
  }).pipe(Effect.provide(provide(runtime)))
})

it.effect("uses only the source-qualified architecture ID", () => {
  const runtime = {
    placement,
    extensions: {
      gguf: {
        inspect: () => Effect.succeed(inspection),
        load: () => Effect.succeed({ entries: [] })
      }
    }
  } as unknown as Runtime.RuntimeService
  return Effect.gen(function*() {
    const registry = yield* Registry.Registry
    yield* registry.register({
      ...architecture(() => {}),
      id: "test-model"
    })
    const error = yield* Effect.flip(Gguf.load("model.gguf"))
    expect(error._tag).toBe("RegistryError")
    if (error._tag !== "RegistryError") throw error
    expect(error.message).toContain("gguf:test-model")
  }).pipe(Effect.provide(provide(runtime)))
})

it.effect("releases every loaded tensor when load descriptors disagree with inspection", () => {
  const dense = tensor(denseDescriptor)
  const packed = tensor(encodedDescriptor)
  const released: Array<Tensor.Concrete> = []
  const mismatched: Runtime.GgufTensorDescriptor = {
    ...encodedDescriptor,
    physicalShape: [2, 145]
  }
  const runtime = {
    placement,
    extensions: {
      gguf: {
        inspect: () => Effect.succeed(inspection),
        load: () =>
          Effect.succeed({
            entries: [
              { descriptor: mismatched, tensor: packed },
              { descriptor: denseDescriptor, tensor: dense }
            ]
          })
      }
    },
    release: (value: Tensor.Concrete) =>
      Effect.suspend(() => {
        released.push(value)
        return value === packed
          ? Effect.fail(
            new Runtime.BackendError({
              reason: "execution-failed",
              backend: "gguf-test",
              operation: "release",
              phase: "execute",
              message: "first release failed"
            })
          )
          : Effect.void
      })
  } as unknown as Runtime.RuntimeService

  return Effect.gen(function*() {
    const registry = yield* Registry.Registry
    yield* registry.register(architecture(() => {}))
    const error = yield* Effect.flip(Gguf.load("model.gguf"))

    expect(error._tag).toBe("GgufError")
    if (error._tag !== "GgufError") throw error
    expect(error.op).toBe("validate")
    expect(released).toEqual([packed, dense])
  }).pipe(Effect.provide(provide(runtime)))
})

// A runtime archive may not transfer the same handle twice: cleanup must dedupe
// by handle identity or a failed validation would double-release native storage.
it.effect("rejects duplicate loaded handle ownership and releases it once", () => {
  const duplicate = tensor(encodedDescriptor)
  const released: Array<Tensor.Concrete> = []
  const runtime = {
    placement,
    extensions: {
      gguf: {
        inspect: () => Effect.succeed(inspection),
        load: () =>
          Effect.succeed({
            entries: [
              { descriptor: encodedDescriptor, tensor: duplicate },
              { descriptor: denseDescriptor, tensor: duplicate }
            ]
          })
      }
    },
    release: (value: Tensor.Concrete) => Effect.sync(() => void released.push(value))
  } as unknown as Runtime.RuntimeService

  return Effect.gen(function*() {
    const registry = yield* Registry.Registry
    yield* registry.register(architecture(() => {}))
    const error = yield* Effect.flip(Gguf.load("duplicate.gguf"))

    expect(error._tag).toBe("GgufError")
    if (error._tag !== "GgufError") throw error
    expect(error.message).toContain("duplicate tensor ownership")
    expect(released).toEqual([duplicate])
  }).pipe(Effect.provide(provide(runtime)))
})

it.effect("the runtime cleans up interruption before archive ownership transfers", () =>
  Effect.gen(function*() {
    const waiting = yield* Deferred.make<void>()
    const dense = tensor(denseDescriptor)
    const packed = tensor(encodedDescriptor)
    const released: Array<Tensor.Concrete> = []
    const archive = {
      entries: [
        { descriptor: encodedDescriptor, tensor: packed },
        { descriptor: denseDescriptor, tensor: dense }
      ]
    }
    const runtime = {
      placement,
      extensions: {
        gguf: {
          inspect: () => Effect.succeed(inspection),
          // The deferred marks the archive's use phase, ensuring interruption
          // occurs before load returns and transfers ownership to Gguf.load.
          load: () =>
            Effect.acquireUseRelease(
              Effect.succeed(archive),
              () => Deferred.succeed(waiting, undefined).pipe(Effect.andThen(Effect.never)),
              () =>
                Effect.sync(() => {
                  released.push(packed, dense)
                })
            )
        }
      },
      release: (value: Tensor.Concrete) => Effect.sync(() => void released.push(value))
    } as unknown as Runtime.RuntimeService
    const layer = provide(runtime)
    const program = Effect.gen(function*() {
      const registry = yield* Registry.Registry
      yield* registry.register(architecture(() => {}))
      return yield* Gguf.load("handoff.gguf")
    }).pipe(Effect.provide(layer))
    const target = yield* program.pipe(Effect.forkChild({ startImmediately: true }))
    yield* Deferred.await(waiting)
    yield* Fiber.interrupt(target)

    expect(released).toEqual([packed, dense])
  }))

onDevices("GGUF", () => (it) => {
  it.effect("preserves encoded metadata through a compiled identity input", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "effect-torch-gguf-"))
    const file = path.join(directory, "identity.gguf")
    fs.writeFileSync(file, fixture())

    return Effect.gen(function*() {
      const registry = yield* Registry.Registry
      yield* registry.register({
        id: "gguf:compiled-identity",
        create: () =>
          Model.define({
            parameters: [{ name: "packed", shape: [2, 256] }],
            forward: (_, input) => Effect.succeed(input as Tensor.Lazy)
          })
      })
      const loaded = yield* Gguf.load(file)
      const compiled = yield* Tensor.compile(([input]) => Effect.succeed([input]))
      const [identity] = yield* compiled.call([loaded.params[0]])

      expect(identity.shape).toEqual([2, 256])
      expect(identity.dtype).toBe("f32")
      expect(identity.storage).toEqual({
        encoding: "Q4_K",
        physicalShape: [2, 144],
        physicalDtype: "u8"
      })
      const readbackError = yield* Effect.flip(Tensor.toTypedArray(identity))
      expect(readbackError.op).toBe("toTypedArray")
      expect(readbackError.message).toContain("Q4_K")

      const savePath = path.join(directory, "encoded.safetensors")
      const saveError = yield* Effect.flip(Tensor.save(savePath, { packed: identity }))
      expect(saveError.op).toBe("save")
      expect(fs.existsSync(savePath)).toBe(false)
      const input = yield* Tensor.zeros([1, 256])
      const projected = yield* Tensor.linearRows(input, loaded.params[0])
      expect(projected.shape).toEqual([1, 2])
      const [result] = yield* Tensor.compute([projected])
      expect(yield* Tensor.toNumberArray(result)).toEqual([0, 0])
      yield* Tensor.clear(result)
      yield* Tensor.clear(identity)
      yield* Tensor.clearAll(loaded.params)
    }).pipe(
      Effect.provide(Registry.emptyLayer),
      Effect.ensuring(Effect.sync(() => fs.rmSync(directory, { recursive: true, force: true })))
    )
  })
})

import { expect, it } from "@effect/vitest"
import { Deferred, Effect, Fiber, Layer } from "effect"
import * as fs from "node:fs"
import * as os from "node:os"
import * as path from "node:path"
import { Gguf, Model, Runtime, Tensor } from "../src/index.ts"
import { onDevices } from "./utils/devices.ts"

const placement: Runtime.Placement = Object.freeze({
  id: "gguf-test:0",
  deviceType: "test",
  description: "GGUF test runtime"
})

type RuntimeDouble = Partial<Omit<Runtime.RuntimeService, "extensions">> & {
  readonly extensions?: Partial<Runtime.RuntimeService["extensions"]>
}
type TestHandle = Pick<Tensor.Any, "_tag" | "shape" | "dtype" | "storage" | "device" | "placement" | "pipe">

const runtimeDouble = (value: RuntimeDouble): Runtime.RuntimeService => {
  // SAFETY: Each test supplies every runtime member reached by the code under test.
  return value as Runtime.RuntimeService
}

const concreteHandle = (value: TestHandle): Tensor.Concrete => {
  // SAFETY: The tensor factory supplies all public metadata; only Runtime's private brands are absent.
  return value as Tensor.Concrete
}

const loaderOnlyIdentity = (value: Tensor.Any): Tensor.Lazy => {
  // SAFETY: Loader metadata tests never invoke these placeholder model forwards.
  return value as Tensor.Lazy
}

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

const tensor = (descriptor: Runtime.GgufTensorDescriptor): Tensor.Concrete => {
  const value = {
    _tag: "Tensor",
    shape: descriptor.logicalShape,
    dtype: "f32",
    device: placement.deviceType,
    placement,
    pipe() {
      throw new Error("unused test handle pipe")
    }
  } satisfies TestHandle
  if (descriptor.format !== "F32") {
    Object.assign(value, {
      storage: Object.freeze({
        encoding: descriptor.format,
        physicalShape: descriptor.physicalShape,
        physicalDtype: "u8"
      })
    })
  }
  return concreteHandle(Object.freeze(value))
}

const inspection: Runtime.GgufInspection = Object.freeze({
  metadata: Object.freeze([
    Object.freeze({ key: "general.architecture", value: "test-model" }),
    Object.freeze({ key: "test-model.context_length", value: 32 }),
    Object.freeze({ key: "general.name", value: "fixture" })
  ]),
  tensors: Object.freeze([encodedDescriptor, denseDescriptor])
})

const definition = (
  capture: (config: Gguf.ModelConfig) => void,
  architecture = "test-model"
): Gguf.ModelDefinition => ({
  architecture,
  create: (config) => {
    capture(config)
    return Model.define({
      parameterSpecs: [
        { name: "dense", shape: [2], initializer: { _tag: "Normal", scale: 1 } },
        { name: "packed", shape: [2, 256], initializer: { _tag: "Normal", scale: 1 } }
      ],
      forward: (_, input) => Effect.succeed(loaderOnlyIdentity(input))
    })
  }
})

const provide = (runtime: Runtime.RuntimeService) => Layer.succeed(Runtime.Runtime, runtime)

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
  const runtime = runtimeDouble({
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
  })
  let config: Gguf.ModelConfig | undefined

  return Effect.gen(function*() {
    const loaded = yield* Gguf.loadModel("model.gguf", definition((value) => config = value))

    expect(paths).toEqual(["inspect:model.gguf", "load:model.gguf"])
    expect([...config!]).toEqual([
      ["architecture", "test-model"],
      ["context_length", 32],
      ["name", "fixture"]
    ])
    expect(loaded.model.parameterSpecs.map(({ name }) => name)).toEqual(["dense", "packed"])
    expect(loaded.params).toEqual([dense, packed])
    expect(loaded.params[1].storage).toEqual({
      encoding: "Q4_K",
      physicalShape: [2, 144],
      physicalDtype: "u8"
    })
    expect(released).toEqual([])
  }).pipe(Effect.provide(provide(runtime)))
})

it.effect("rejects an architecture mismatch before model creation", () => {
  const runtime = runtimeDouble({
    placement,
    extensions: {
      gguf: {
        inspect: () => Effect.succeed(inspection),
        load: () => Effect.succeed({ entries: [] })
      }
    }
  })
  return Effect.gen(function*() {
    const error = yield* Effect.flip(Gguf.loadModel("model.gguf", definition(() => {}, "other-model")))
    expect(error._tag).toBe("GgufError")
    if (error._tag !== "GgufError") throw error
    expect(error.op).toBe("validate")
    expect(error.message).toContain("\"other-model\"")
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
  const runtime = runtimeDouble({
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
  })

  return Effect.gen(function*() {
    const error = yield* Effect.flip(Gguf.loadModel("model.gguf", definition(() => {})))

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
  const runtime = runtimeDouble({
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
  })

  return Effect.gen(function*() {
    const error = yield* Effect.flip(Gguf.loadModel("duplicate.gguf", definition(() => {})))

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
    const runtime = runtimeDouble({
      placement,
      extensions: {
        gguf: {
          inspect: () => Effect.succeed(inspection),
          // The deferred marks the archive's use phase. It pauses load so
          // interruption occurs before ownership transfers to Gguf.load.
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
    })
    const layer = provide(runtime)
    const program = Gguf.loadModel("handoff.gguf", definition(() => {})).pipe(Effect.provide(layer))
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
      const loaded = yield* Gguf.loadModel(file, {
        architecture: "compiled-identity",
        create: () =>
          Model.define({
            parameterSpecs: [{
              name: "packed",
              shape: [2, 256],
              initializer: { _tag: "Normal", scale: 1 }
            }],
            forward: (_, input) => Effect.succeed(loaderOnlyIdentity(input))
          })
      })
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
      Effect.ensuring(Effect.sync(() => fs.rmSync(directory, { recursive: true, force: true })))
    )
  })
})

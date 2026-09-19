import { expect, it } from "@effect/vitest"
import { Deferred, Effect, Fiber, Layer } from "effect"
import * as fs from "node:fs"
import * as os from "node:os"
import * as path from "node:path"
import { Gguf, Model, Runtime, Safetensors, Tensor } from "../src/index.ts"
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

const parameterDefinition: Gguf.ParameterArtifactDefinition = {
  architecture: "test-model",
  parameterSpecs: (_, tensors) =>
    Effect.succeed(tensors.map(({ name, logicalShape }) => ({ name, shape: logicalShape })))
}

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

const f16 = (buffer: Buffer, offset: number, bits: number): void => {
  buffer.writeUInt16LE(bits, offset)
}

const oneBlock = (format: Runtime.TensorStorageEncoding): Buffer => {
  switch (format) {
    case "Q2_K": {
      const block = Buffer.alloc(84)
      block.fill(0x01, 0, 16)
      block.fill(0x55, 16, 80)
      f16(block, 80, 0x3c00)
      return block
    }
    case "Q3_K": {
      const block = Buffer.alloc(110)
      block.fill(0xff, 0, 32)
      block.fill(0x55, 32, 96)
      block.fill(0x11, 96, 104)
      block.fill(0xaa, 104, 108)
      f16(block, 108, 0x3c00)
      return block
    }
    case "Q4_K":
    case "Q5_K": {
      const q5 = format === "Q5_K"
      const block = Buffer.alloc(q5 ? 176 : 144)
      f16(block, 0, 0x3c00)
      block.fill(0x01, 4, 8)
      block.fill(0x01, 12, 16)
      block.fill(0x11, q5 ? 48 : 16)
      return block
    }
    case "Q6_K": {
      const block = Buffer.alloc(210)
      block.fill(0x11, 0, 128)
      block.fill(0xaa, 128, 192)
      block.fill(0x01, 192, 208)
      f16(block, 208, 0x3c00)
      return block
    }
  }
}

const kquantFixture = (): Buffer => {
  const tensors = ([
    ["q2", "Q2_K", 10],
    ["q3", "Q3_K", 11],
    ["q4", "Q4_K", 12],
    ["q5", "Q5_K", 13],
    ["q6", "Q6_K", 14]
  ] as const).map(([name, format, type]) => ({
    name,
    format,
    type,
    block: Buffer.concat(Array.from({ length: 4 }, () => oneBlock(format)))
  }))
  let offset = 0
  const offsets = tensors.map(({ block }) => {
    const current = offset
    offset = Math.ceil((offset + block.length) / 32) * 32
    return current
  })
  const header = Buffer.concat([
    Buffer.from("GGUF"),
    u32(3),
    u64(tensors.length),
    u64(3),
    string("general.architecture"),
    u32(8),
    string("all-kquants"),
    string("general.alignment"),
    u32(4),
    u32(32),
    string("general.quantization_version"),
    u32(4),
    u32(2),
    ...tensors.flatMap(({ name, type }, index) => [
      string(name),
      u32(2),
      u64(1024),
      u64(1),
      u32(type),
      u64(offsets[index])
    ])
  ])
  const data = Buffer.alloc(offset)
  tensors.forEach(({ block }, index) => block.copy(data, offsets[index]))
  const padding = (32 - header.length % 32) % 32
  return Buffer.concat([header, Buffer.alloc(padding), data])
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

it.effect("loads selected GGUF parameters in requested order, including an empty selection", () => {
  const dense = tensor(denseDescriptor)
  const packed = tensor(encodedDescriptor)
  const calls: Array<ReadonlyArray<string> | undefined> = []
  const released: Array<Tensor.Concrete> = []
  const runtime = runtimeDouble({
    placement,
    extensions: {
      gguf: {
        inspect: () => Effect.succeed(inspection),
        load: (_, options) =>
          Effect.sync(() => {
            calls.push(options?.names)
            return {
              entries: [
                { descriptor: encodedDescriptor, tensor: packed },
                { descriptor: denseDescriptor, tensor: dense }
              ].filter((entry) => options?.names?.includes(entry.descriptor.name))
            }
          })
      }
    },
    release: (value) => Effect.sync(() => void released.push(value))
  })
  return Effect.gen(function*() {
    const selected = yield* Gguf.loadParameters("model.gguf", parameterDefinition, { names: ["dense"] })
    expect(selected.params).toEqual([dense])
    expect(selected.parameterSpecs).toEqual([{ name: "dense", shape: [2] }])
    expect(selected.metadata.get("architecture")).toBe("test-model")
    const ordered = yield* Gguf.loadParameters("model.gguf", parameterDefinition, { names: ["dense", "packed"] })
    expect(ordered.params).toEqual([dense, packed])
    const empty = yield* Gguf.loadParameters("model.gguf", parameterDefinition, { names: [] })
    expect(empty.params).toEqual([])
    expect(empty.parameterSpecs).toEqual([])
    expect(calls).toEqual([["dense"], ["dense", "packed"], []])
    expect(released).toEqual([])
  }).pipe(Effect.provide(provide(runtime)))
})

it.effect("rejects invalid GGUF selections before reading payloads and keeps full-catalog validation", () => {
  let loads = 0
  const runtime = runtimeDouble({
    placement,
    extensions: {
      gguf: {
        inspect: () => Effect.succeed(inspection),
        load: () =>
          Effect.sync(() => {
            loads++
            return { entries: [] }
          })
      }
    }
  })
  return Effect.gen(function*() {
    for (const names of [["absent"], ["dense", "dense"], [""], Array<string>(1)]) {
      const error = yield* Effect.flip(Gguf.loadParameters("model.gguf", parameterDefinition, { names }))
      expect(error._tag).toBe("GgufError")
    }
    for (
      const specs of [
        [{ name: "dense", shape: [3] }],
        [{ name: "dense", shape: [2] }, { name: "dense", shape: [2] }],
        [{ name: "dense", shape: Array<number>(1) }]
      ]
    ) {
      const badDefinition: Gguf.ParameterArtifactDefinition = {
        ...parameterDefinition,
        parameterSpecs: () => Effect.succeed(specs)
      }
      const error = yield* Effect.flip(Gguf.loadParameters("model.gguf", badDefinition, { names: ["dense"] }))
      expect(error._tag).toBe("GgufError")
    }
    const subsetDefinition: Gguf.ParameterArtifactDefinition = {
      ...parameterDefinition,
      parameterSpecs: () => Effect.succeed([{ name: "dense", shape: [2] }])
    }
    const fullError = yield* Effect.flip(Gguf.loadParameters("model.gguf", subsetDefinition))
    expect(fullError.message).toContain("catalog")
    expect(loads).toBe(0)
  }).pipe(Effect.provide(provide(runtime)))
})

it.effect("releases unrequested GGUF results when a partial load violates its selection", () => {
  const dense = tensor(denseDescriptor)
  const packed = tensor(encodedDescriptor)
  const released: Array<Tensor.Concrete> = []
  let entries = [{ descriptor: encodedDescriptor, tensor: packed }]
  const runtime = runtimeDouble({
    placement,
    extensions: {
      gguf: {
        inspect: () => Effect.succeed(inspection),
        load: () => Effect.sync(() => ({ entries }))
      }
    },
    release: (value) => Effect.sync(() => void released.push(value))
  })
  return Effect.gen(function*() {
    const wrong = yield* Effect.flip(Gguf.loadParameters("model.gguf", parameterDefinition, { names: ["dense"] }))
    expect(wrong.message).toContain("differs from inspection")
    expect(released).toEqual([packed])
    entries = [{ descriptor: denseDescriptor, tensor: dense }, { descriptor: encodedDescriptor, tensor: packed }]
    const extra = yield* Effect.flip(Gguf.loadParameters("model.gguf", parameterDefinition, { names: ["dense"] }))
    expect(extra.message).toContain("tensor count")
    expect(released).toEqual([packed, dense, packed])
  }).pipe(Effect.provide(provide(runtime)))
})

it.effect("snapshots GGUF selection and parameter shapes before a pending payload load", () =>
  Effect.gen(function*() {
    const dense = tensor(denseDescriptor)
    const shape = [2]
    const names = ["dense"]
    const started = yield* Deferred.make<void>()
    const complete = yield* Deferred.make<void>()
    let forwarded: ReadonlyArray<string> | undefined
    const runtime = runtimeDouble({
      placement,
      extensions: {
        gguf: {
          inspect: () => Effect.succeed(inspection),
          load: (_, options) =>
            Effect.gen(function*() {
              forwarded = options?.names
              yield* Deferred.succeed(started, undefined)
              yield* Deferred.await(complete)
              return { entries: [{ descriptor: denseDescriptor, tensor: dense }] }
            })
        }
      }
    })
    const definition: Gguf.ParameterArtifactDefinition = {
      ...parameterDefinition,
      parameterSpecs: () => Effect.succeed([{ name: "dense", shape }])
    }
    const fiber = yield* Gguf.loadParameters("model.gguf", definition, { names }).pipe(
      Effect.provide(provide(runtime)),
      Effect.forkChild({ startImmediately: true })
    )
    yield* Deferred.await(started)
    names[0] = "packed"
    shape[0] = 3
    yield* Deferred.succeed(complete, undefined)
    const loaded = yield* Fiber.join(fiber)
    expect(forwarded).toEqual(["dense"])
    expect(loaded.parameterSpecs).toEqual([{ name: "dense", shape: [2] }])
    expect(loaded.params).toEqual([dense])
  }))

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

onDevices("GGUF", (device) => (it) => {
  it.effect("rejects packed roots while allowing packed inputs to linear", () => {
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
      const compileError = yield* Effect.flip(compiled.call([loaded.params[0]]))
      expect(compileError.message).toMatch(/packed|encoded/i)
      const weight = loaded.params[0]

      expect(weight.shape).toEqual([2, 256])
      expect(weight.dtype).toBe("f32")
      expect(weight.storage).toEqual({
        encoding: "Q4_K",
        physicalShape: [2, 144],
        physicalDtype: "u8"
      })
      const readbackError = yield* Effect.flip(Tensor.toTypedArray(weight))
      expect(readbackError.op).toBe("toTypedArray")
      expect(readbackError.message).toContain("Q4_K")

      const savePath = path.join(directory, "encoded.safetensors")
      const saveError = yield* Effect.flip(Safetensors.save(savePath, { packed: weight }))
      expect(saveError.op).toBe("save")
      expect(fs.existsSync(savePath)).toBe(false)
      const input = yield* Tensor.zeros([1, 256])
      const projected = yield* Tensor.linearRows(input, loaded.params[0])
      expect(projected.shape).toEqual([1, 2])
      const [result] = yield* Tensor.compute([projected])
      expect(yield* Tensor.toNumberArray(result)).toEqual([0, 0])
      yield* Tensor.clear(result)
      yield* Tensor.clearAll(loaded.params)
    }).pipe(
      Effect.ensuring(Effect.sync(() => fs.rmSync(directory, { recursive: true, force: true })))
    )
  })

  it.effect("inference retains packed parameters and releases only materialized parameters on failure", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "effect-torch-packed-inference-"))
    const file = path.join(directory, "weights.gguf")
    fs.writeFileSync(file, fixture())

    return Effect.gen(function*() {
      const runtime = yield* Runtime.Runtime
      const loaded = yield* Gguf.loadModel(file, {
        architecture: "compiled-identity",
        create: () =>
          Model.define({
            parameterSpecs: [{
              name: "packed",
              shape: [2, 256],
              initializer: { _tag: "Normal", scale: 1 }
            }],
            forward: ([weight], input) => Tensor.embedding(input, { weight })
          })
      })
      const parameterSpecs = [...loaded.model.parameterSpecs, {
        name: "bias",
        shape: [256],
        initializer: { _tag: "Normal" as const, scale: 1 }
      }]
      const bias = yield* Tensor.full([256], 2)
      const params = [...loaded.params, bias]
      const config = { maxTokens: 16, blockSize: 4, prefillChunks: [1], batchSize: 1 }
      const before = yield* runtime.extensions.diagnostics.externalMemoryBytes
      const invalid = yield* Model.define({
        parameterSpecs,
        forward: (_, input) => Tensor.cast(input, "f32")
      })
      const error = yield* Effect.flip(Model.inference(invalid, params, config))
      expect(error.message).toContain("model output must be")
      expect(yield* runtime.extensions.diagnostics.externalMemoryBytes).toBe(before)

      const waiting = yield* Deferred.make<void>()
      const interrupted = yield* Model.define({
        parameterSpecs,
        forward: () => Deferred.succeed(waiting, undefined).pipe(Effect.andThen(Effect.never))
      })
      const fiber = yield* Model.inference(interrupted, params, config).pipe(
        Effect.forkChild({ startImmediately: true })
      )
      yield* Deferred.await(waiting)
      yield* Fiber.interrupt(fiber)
      expect(yield* runtime.extensions.diagnostics.externalMemoryBytes).toBe(before)

      const model = yield* Model.define({
        parameterSpecs,
        forward: ([weight, bias], input) =>
          Tensor.embedding(input, { weight }).pipe(Effect.flatMap((value) => Tensor.add(value, bias)))
      })
      const program = yield* Model.inference(model, params, config)
      yield* Tensor.clearAll(loaded.params)
      const execution = yield* program.execution()
      const prompt = yield* Tensor.fromTypedArray(new Uint32Array([0, 1]), [1, 2])
      const [entry] = yield* execution.add([prompt])
      expect(yield* Tensor.toNumberArray(entry.logits)).toEqual(new Array(256).fill(2))
      const [next] = yield* execution.step([{ seq: entry.seq, token: 0 }])
      expect(yield* Tensor.toNumberArray(next)).toEqual(new Array(256).fill(2))
      yield* Tensor.clearAll([entry.logits, next])
      yield* execution.close()
    }).pipe(
      Effect.ensuring(Effect.sync(() => fs.rmSync(directory, { recursive: true, force: true })))
    )
  })

  it.effect("loads selected dense and packed tensors from a sparse four-GiB GGUF", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "effect-torch-partial-gguf-"))
    const file = path.join(directory, "partial.gguf")
    const header = Buffer.concat([
      Buffer.from("GGUF"),
      u32(3),
      u64(3),
      u64(2),
      string("general.architecture"),
      u32(8),
      string("partial"),
      string("general.quantization_version"),
      u32(4),
      u32(2),
      string("dense"),
      u32(1),
      u64(2),
      u32(0),
      u64(0),
      string("packed"),
      u32(2),
      u64(256),
      u64(1),
      u32(12),
      u64(32),
      string("unused"),
      u32(1),
      u64(2 ** 30),
      u32(0),
      u64(192)
    ])
    const start = Math.ceil(header.length / 32) * 32
    const fd = fs.openSync(file, "w")
    try {
      fs.writeSync(fd, header)
      const dense = Buffer.alloc(8)
      dense.writeFloatLE(1.5, 0)
      dense.writeFloatLE(-2.25, 4)
      fs.writeSync(fd, dense, 0, dense.length, start)
      fs.writeSync(fd, oneBlock("Q4_K"), 0, 144, start + 32)
      fs.ftruncateSync(fd, start + 192 + 2 ** 32)
    } finally {
      fs.closeSync(fd)
    }
    const definition: Gguf.ParameterArtifactDefinition = { ...parameterDefinition, architecture: "partial" }
    return Effect.gen(function*() {
      const runtime = yield* Runtime.Runtime
      const before = yield* runtime.extensions.diagnostics.externalMemoryBytes
      const empty = yield* Gguf.loadParameters(file, definition, { names: [] })
      expect(empty.params).toEqual([])
      expect(yield* runtime.extensions.diagnostics.externalMemoryBytes).toBe(before)
      for (const names of [["dense", "absent"], ["packed", "packed"]]) {
        const error = yield* Effect.flip(runtime.extensions.gguf.load(file, { names }))
        expect(error.operation).toBe("loadGguf")
        expect(yield* runtime.extensions.diagnostics.externalMemoryBytes).toBe(before)
      }
      const loaded = yield* Gguf.loadParameters(file, definition, { names: ["packed", "dense"] })
      expect(loaded.parameterSpecs.map((entry) => entry.name)).toEqual(["packed", "dense"])
      expect(loaded.params[0].storage).toEqual({ encoding: "Q4_K", physicalDtype: "u8", physicalShape: [1, 144] })
      expect(loaded.params[0].dtype).toBe("f32")
      const allocated = (yield* runtime.extensions.diagnostics.externalMemoryBytes) - before
      // CPU and Metal charge at least 4096 bytes per handle. CUDA does not
      // expose this diagnostic; its native selection tests check device storage.
      if (device !== "cuda") expect(allocated).toBe(8192)
      expect(yield* Tensor.toNumberArray(loaded.params[1])).toEqual([1.5, -2.25])
      const projected = yield* Tensor.linearRows(yield* Tensor.ones([1, 256]), loaded.params[0])
      const [output] = yield* Tensor.compute([projected])
      expect(yield* Tensor.toNumberArray(output)).toEqual([256])
      yield* Tensor.clearAll([output, ...loaded.params])
    }).pipe(
      Effect.ensuring(Effect.sync(() => fs.rmSync(directory, { recursive: true, force: true })))
    )
  })

  it.effect("executes linear and embedding with every K-quant encoding", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "effect-torch-kquants-"))
    const file = path.join(directory, "all-kquants.gguf")
    fs.writeFileSync(file, kquantFixture())

    return Effect.gen(function*() {
      const loaded = yield* Gguf.loadModel(file, {
        architecture: "all-kquants",
        create: () =>
          Model.define({
            parameterSpecs: ["q2", "q3", "q4", "q5", "q6"].map((name) => ({
              name,
              shape: [1, 1024],
              initializer: { _tag: "Normal" as const, scale: 1 }
            })),
            forward: (_, input) => Effect.succeed(loaderOnlyIdentity(input))
          })
      })
      const indexes = yield* Tensor.fromTypedArray(new Uint32Array([0]), [1])
      const bias = yield* Tensor.full([1], 0.5)

      for (const weight of loaded.params) {
        const embedded = yield* Tensor.embedding(indexes, { weight })
        const [embeddedValue] = yield* Tensor.compute([embedded])
        expect(yield* Tensor.toNumberArray(embeddedValue)).toEqual(new Array(1024).fill(1))
        yield* Tensor.clear(embeddedValue)
        for (const rows of [1, 16]) {
          // This exact F32 value rounds to 1 in BF16. Packed execution must
          // consume it directly at both decode and prefill batch sizes.
          const input = yield* Tensor.full([rows, 1024], 1.00390625)
          const projected = yield* Tensor.linearRows(input, weight)
          const biased = yield* Tensor.linearRows(input, weight, bias)
          for (const optimize of [false, true]) {
            const program = yield* Tensor.freezeProgram([projected, biased], { optimize })
            const [projectedValue, biasedValue] = yield* Tensor.runProgram(program, [])
            expect(yield* Tensor.toNumberArray(projectedValue)).toEqual(new Array(rows).fill(1028))
            expect(yield* Tensor.toNumberArray(biasedValue)).toEqual(new Array(rows).fill(1028.5))
            yield* Tensor.clearAll([projectedValue, biasedValue])
          }
        }
      }

      yield* Tensor.clearAll(loaded.params)
    }).pipe(
      Effect.ensuring(Effect.sync(() => fs.rmSync(directory, { recursive: true, force: true })))
    )
  })
})

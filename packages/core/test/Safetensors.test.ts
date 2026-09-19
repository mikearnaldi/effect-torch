import { describe, expect } from "@effect/vitest"
import { Deferred, Effect, Fiber } from "effect"
import * as fs from "node:fs"
import * as os from "node:os"
import * as path from "node:path"
import { Model, Runtime, Safetensors, Tensor } from "../src/index.ts"
import { deep, floats, onDevices } from "./utils/devices.ts"

interface Entry {
  readonly name: string
  readonly dtype: string
  readonly shape: ReadonlyArray<number>
  readonly bytes: Uint8Array
}

interface ArchiveTensorHeader {
  readonly dtype: string
  readonly shape: ReadonlyArray<number>
  readonly data_offsets: readonly [number, number]
}

interface ArchiveHeader {
  [name: string]: ArchiveTensorHeader | Readonly<Record<string, string>>
}

const headerBytes = (header: ArchiveHeader): Buffer => {
  const json = Buffer.from(JSON.stringify(header))
  const data = Buffer.alloc(Math.ceil(json.length / 8) * 8, " ")
  json.copy(data)
  const length = Buffer.alloc(8)
  length.writeBigUInt64LE(BigInt(data.length))
  return Buffer.concat([length, data])
}

const writeArchive = (file: string, entries: ReadonlyArray<Entry>, metadata = { format: "pt" }) => {
  let offset = 0
  const header: ArchiveHeader = { __metadata__: metadata }
  for (const entry of entries) {
    header[entry.name] = {
      dtype: entry.dtype,
      shape: entry.shape,
      data_offsets: [offset, offset + entry.bytes.length]
    }
    offset += entry.bytes.length
  }
  fs.writeFileSync(file, Buffer.concat([headerBytes(header), ...entries.map((entry) => entry.bytes)]))
}

const withDirectory = <A, E, R>(use: (directory: string) => Effect.Effect<A, E, R>) =>
  Effect.acquireUseRelease(
    Effect.sync(() => fs.mkdtempSync(path.join(os.tmpdir(), "effect-torch-indexed-"))),
    use,
    (directory) => Effect.sync(() => fs.rmSync(directory, { recursive: true, force: true }))
  )

const weight: Entry = {
  name: "layer.weight",
  dtype: "BF16",
  shape: [2, 3],
  bytes: Buffer.from([0x80, 0x3f, 0x00, 0x40, 0x80, 0x40, 0x80, 0xbf, 0x00, 0x3f, 0x00, 0x00])
}
const bias: Entry = { name: "layer.bias", dtype: "BF16", shape: [2], bytes: Buffer.from([0x80, 0x3e, 0x80, 0xbe]) }

const mlp = Effect.gen(function*() {
  return yield* Model.chain(
    yield* Model.linear("fc1", 2, 8),
    yield* Model.tanh,
    yield* Model.linear("fc2", 8, 1),
    yield* Model.sigmoid
  )
})

onDevices("Safetensors", () => (it) => {
  it.effect("inspects exact BF16 geometry and loads only selected tensors", () =>
    withDirectory((directory) =>
      Effect.gen(function*() {
        const file = path.join(directory, "model.safetensors")
        writeArchive(file, [weight, bias])
        const runtime = yield* Runtime.Runtime
        const before = yield* runtime.extensions.diagnostics.externalMemoryBytes
        const inspection = yield* Safetensors.inspectArchive(file)
        expect(yield* runtime.extensions.diagnostics.externalMemoryBytes).toBe(before)
        expect(inspection.entries).toEqual([
          { name: bias.name, dtype: "bf16", shape: [2], byteLength: 4 },
          { name: weight.name, dtype: "bf16", shape: [2, 3], byteLength: 12 }
        ])
        expect(inspection.metadata).toEqual({ format: "pt" })
        expect(Object.isFrozen(inspection)).toBe(true)
        expect(Object.isFrozen(inspection.entries)).toBe(true)
        expect(Object.isFrozen(inspection.entries[0].shape)).toBe(true)
        const archive = yield* Safetensors.loadArchive(file, { names: [weight.name] })
        expect(Object.keys(archive.tensors)).toEqual([weight.name])
        expect(archive.tensors[weight.name].dtype).toBe("bf16")
        expect(yield* Tensor.toNumberArray(archive.tensors[weight.name])).toEqual([1, 2, 4, -1, 0.5, 0])
        yield* Tensor.clearAll(Object.values(archive.tensors))
        const empty = yield* Safetensors.loadArchive(file, { names: [] })
        expect(empty.tensors).toEqual({})
        expect(yield* runtime.extensions.diagnostics.externalMemoryBytes).toBe(before)
        const missing = yield* Effect.flip(Safetensors.load(file, { names: ["absent"] }))
        expect(missing.message).toContain("absent")
        const duplicate = yield* Effect.flip(Safetensors.load(file, { names: [weight.name, weight.name] }))
        expect(duplicate.message).toContain("unique")
        expect(yield* runtime.extensions.diagnostics.externalMemoryBytes).toBe(before)
      })
    ))

  it.effect("selects tensors across shards and uses loaded BF16 rows in a linear", () =>
    withDirectory((directory) =>
      Effect.scoped(Effect.gen(function*() {
        writeArchive(path.join(directory, "weights.safetensors"), [weight])
        writeArchive(path.join(directory, "bias.safetensors"), [bias])
        const index = path.join(directory, "model.safetensors.index.json")
        fs.writeFileSync(
          index,
          JSON.stringify({
            metadata: { total_size: 16 },
            weight_map: { [weight.name]: "weights.safetensors", [bias.name]: "bias.safetensors" }
          })
        )
        expect((yield* Safetensors.inspectArchive(index)).entries.map((entry) => entry.byteLength)).toEqual([4, 12])
        const tensors = yield* Safetensors.load(index, { names: [bias.name, weight.name] })
        yield* Tensor.clearAllScoped(Object.values(tensors))
        const input = yield* Tensor.cast(
          yield* Tensor.fromTypedArray(new Float32Array([1, 2, 3, -1, 0, 2]), [2, 3]),
          "bf16"
        )
        const output = yield* Tensor.linearRows(input, tensors[weight.name], tensors[bias.name])
        const [result] = yield* Tensor.compute([output]).pipe(Effect.flatMap(Tensor.clearAllScoped))
        expect(result.dtype).toBe("bf16")
        expect(yield* Tensor.toNumberArray(result)).toEqual([17.25, -0.25, 7.25, 0.75])
        fs.rmSync(path.join(directory, "bias.safetensors"))
        const selected = yield* Safetensors.load(index, { names: [weight.name] })
        expect(Object.keys(selected)).toEqual([weight.name])
        yield* Tensor.clearAll(Object.values(selected))
        expect((yield* Effect.flip(Safetensors.load(index))).message).toMatch(/bias|No such file/i)
      }))
    ))

  it.effect("preserves scalar and empty geometry, metadata, and one-argument loading", () =>
    withDirectory((directory) =>
      Effect.gen(function*() {
        const file = path.join(directory, "scalar.safetensors")
        writeArchive(file, [
          { name: "empty", dtype: "BF16", shape: [7, 0, 4], bytes: new Uint8Array() },
          { name: "scalar", dtype: "F32", shape: [], bytes: Buffer.from([0, 0, 0x80, 0x3f]) }
        ])
        const inspection = yield* Safetensors.inspectArchive(file)
        expect(inspection.entries).toEqual([
          { name: "empty", dtype: "bf16", shape: [7, 0, 4], byteLength: 0 },
          { name: "scalar", dtype: "f32", shape: [], byteLength: 4 }
        ])
        const tensors = yield* Safetensors.load(file)
        expect(Object.keys(tensors)).toEqual(["empty", "scalar"])
        expect(yield* Tensor.toNumberArray(tensors.empty)).toEqual([])
        expect(yield* Tensor.toNumberArray(tensors.scalar)).toEqual([1])
        yield* Tensor.clearAll(Object.values(tensors))
        for (const names of [["scalar"], []]) {
          const archive = yield* Safetensors.loadArchive(file, { names })
          expect(archive.metadata).toEqual(inspection.metadata)
          expect(Object.keys(archive.tensors)).toEqual(names)
          yield* Tensor.clearAll(Object.values(archive.tensors))
        }
      })
    ))

  it.effect("reads a small selection from a sparse multi-gigabyte archive", () =>
    withDirectory((directory) =>
      Effect.gen(function*() {
        const file = path.join(directory, "sparse.safetensors")
        const padding = 2 ** 32
        const header = headerBytes({
          selected: { dtype: "BF16", shape: [1], data_offsets: [0, 2] },
          unused: { dtype: "U8", shape: [2, 2 ** 31], data_offsets: [2, padding + 2] }
        })
        const fd = fs.openSync(file, "w")
        try {
          fs.writeSync(fd, header)
          fs.writeSync(fd, Buffer.from([0x80, 0x3f]))
          fs.ftruncateSync(fd, header.length + padding + 2)
        } finally {
          fs.closeSync(fd)
        }
        const inspected = yield* Safetensors.inspectArchive(file)
        expect(inspected.entries[1].byteLength).toBe(padding)
        const loaded = yield* Safetensors.load(file, { names: ["selected"] })
        expect(yield* Tensor.toNumberArray(loaded.selected)).toEqual([1])
        yield* Tensor.clear(loaded.selected)
        const model = yield* Model.define({
          parameterSpecs: [{ name: "selected", shape: [1], initializer: { _tag: "Constant", value: 0 } }],
          forward: ([value], input) => Tensor.mul(input, value)
        })
        const parameters = yield* Safetensors.loadModel(model, file)
        expect(parameters).toHaveLength(1)
        expect(yield* Tensor.toNumberArray(parameters[0])).toEqual([1])
        yield* Tensor.clearAll(parameters)
      })
    ))

  it.effect("cleans every handle when a backend returns tensors outside the selection", () =>
    Effect.gen(function*() {
      const runtime = yield* Runtime.Runtime
      const tensors = yield* Tensor.compute([yield* Tensor.ones([2]), yield* Tensor.zeros([3])])
      const released: Array<Tensor.Concrete> = []
      const incorrect: Runtime.RuntimeService = {
        ...runtime,
        extensions: {
          ...runtime.extensions,
          pathSafetensors: {
            ...runtime.extensions.pathSafetensors,
            load: () =>
              Effect.succeed({
                entries: [{ name: "wanted", tensor: tensors[0] }, { name: "extra", tensor: tensors[1] }],
                metadata: {}
              })
          }
        },
        release: (tensor) => Effect.sync(() => released.push(tensor)).pipe(Effect.andThen(runtime.release(tensor)))
      }
      const error = yield* Effect.flip(
        Safetensors.loadArchive("unused", { names: ["wanted"] }).pipe(Effect.provideService(Runtime.Runtime, incorrect))
      )
      expect(error.message).toContain("selection")
      expect(released).toEqual(tensors)
    }))

  it.effect("rejects invalid inspection geometry from a backend", () =>
    Effect.gen(function*() {
      const runtime = yield* Runtime.Runtime
      for (
        const geometry of [
          { shape: [5], byteLength: 40 },
          { shape: Array<number>(1), byteLength: 2 },
          { shape: [Number.MAX_SAFE_INTEGER], byteLength: Number.MAX_SAFE_INTEGER * 2 }
        ]
      ) {
        const incorrect: Runtime.RuntimeService = {
          ...runtime,
          extensions: {
            ...runtime.extensions,
            pathSafetensors: {
              ...runtime.extensions.pathSafetensors,
              inspect: () => Effect.succeed({ entries: [{ name: "bad", dtype: "bf16", ...geometry }], metadata: {} })
            }
          }
        }
        const error = yield* Effect.flip(
          Safetensors.inspectArchive("unused").pipe(Effect.provideService(Runtime.Runtime, incorrect))
        )
        expect(error.message).toMatch(/payload size|invalid tensor shape/)
      }
    }))

  it.effect("snapshots a selection while loading is pending", () =>
    Effect.gen(function*() {
      const runtime = yield* Runtime.Runtime
      const [tensor] = yield* Tensor.compute([yield* Tensor.ones([1])])
      const started = yield* Deferred.make<void>()
      const continueLoad = yield* Deferred.make<void>()
      let forwarded: ReadonlyArray<string> | undefined
      const pending: Runtime.RuntimeService = {
        ...runtime,
        extensions: {
          ...runtime.extensions,
          pathSafetensors: {
            ...runtime.extensions.pathSafetensors,
            load: (_, options) =>
              Effect.gen(function*() {
                forwarded = options?.names
                yield* Deferred.succeed(started, undefined)
                yield* Deferred.await(continueLoad)
                return { entries: [{ name: "original", tensor }], metadata: {} }
              })
          }
        }
      }
      const names = ["original"]
      const loading = yield* Safetensors.loadArchive("unused", { names }).pipe(
        Effect.provideService(Runtime.Runtime, pending),
        Effect.forkChild({ startImmediately: true })
      )
      yield* Deferred.await(started)
      names[0] = "changed"
      yield* Deferred.succeed(continueLoad, undefined)
      const archive = yield* Fiber.join(loading)
      expect(forwarded).toEqual(["original"])
      expect(Object.keys(archive.tensors)).toEqual(["original"])
      yield* Tensor.clear(tensor)
    }))

  describe("model parameters", () => {
    it.effect("saveModel/loadModel round-trips values and order", () =>
      withDirectory((dir) =>
        Effect.gen(function*() {
          const file = path.join(dir, "mlp.safetensors")
          const model = yield* mlp
          const params = yield* Tensor.compute(yield* Model.initialize(model))
          yield* Safetensors.saveModel(model, params, file)
          const loaded = yield* Safetensors.loadModel(model, file)
          expect(loaded.length).toBe(model.parameterSpecs.map(({ name }) => name).length)
          for (let i = 0; i < params.length; i++) {
            expect(loaded[i].shape).toEqual(params[i].shape)
            deep(yield* Tensor.toNumberArray(loaded[i]), yield* Tensor.toNumberArray(params[i]))
          }
          const x = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
          const [before] = yield* Tensor.compute([yield* model.forward(params, x)])
          const [after] = yield* Tensor.compute([yield* model.forward(loaded, x)])
          deep(yield* Tensor.toNumberArray(after), yield* Tensor.toNumberArray(before))
        })
      ))

    it.effect("saveModel fails with ModelError on an arity mismatch", () =>
      withDirectory((dir) =>
        Effect.gen(function*() {
          const model = yield* mlp
          const params = yield* Tensor.compute(yield* Model.initialize(model))
          const error = yield* Effect.flip(
            Safetensors.saveModel(model, params.slice(0, 3), path.join(dir, "x.safetensors"))
          )
          expect(error._tag).toBe("ModelError")
          expect(error.op).toBe("save")
          expect(error.message).toContain("4 parameters, got 3")
        })
      ))

    it.effect("loadModel fails with ModelError on missing keys", () =>
      withDirectory((dir) =>
        Effect.gen(function*() {
          const file = path.join(dir, "partial.safetensors")
          const small = yield* Model.linear("fc1", 2, 8)
          const params = yield* Tensor.compute(yield* Model.initialize(small))
          yield* Safetensors.saveModel(small, params, file)
          const error = yield* Effect.flip(Safetensors.loadModel(yield* mlp, file))
          expect(error._tag).toBe("ModelError")
          expect(error.op).toBe("load")
          expect(error.message).toContain("fc2.weight")
        })
      ))

    it.effect("params from a different architecture fail at graph-build time", () =>
      withDirectory((dir) =>
        Effect.gen(function*() {
          const file = path.join(dir, "wide.safetensors")
          const wide = yield* Model.chain(
            yield* Model.linear("fc1", 3, 8),
            yield* Model.tanh,
            yield* Model.linear("fc2", 8, 1)
          )
          yield* Safetensors.saveModel(wide, yield* Tensor.compute(yield* Model.initialize(wide)), file)
          const narrow = yield* Model.chain(
            yield* Model.linear("fc1", 2, 8),
            yield* Model.tanh,
            yield* Model.linear("fc2", 8, 1)
          )
          const params = yield* Safetensors.loadModel(narrow, file)
          const x = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
          const error = yield* Effect.flip(narrow.forward(params, x))
          expect(error._tag).toBe("TensorError")
          expect(error.op).toBe("linear")
        })
      ))
  })
})

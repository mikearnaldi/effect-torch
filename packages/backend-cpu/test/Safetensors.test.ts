import { Runtime } from "@effect-torch/core"
import { expect, layer as testLayer } from "@effect/vitest"
import { Effect } from "effect"
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import path from "node:path"
import { layer as backendLayer } from "../src/index.ts"

const withTempFile = <A, E, R>(prefix: string, use: (file: string) => Effect.Effect<A, E, R>) =>
  Effect.acquireUseRelease(
    Effect.tryPromise(() => mkdtemp(path.join(tmpdir(), prefix))),
    (directory) => use(path.join(directory, "archive.safetensors")),
    (directory) => Effect.promise(() => rm(directory, { recursive: true, force: true }))
  )

const f32Bytes = (values: ReadonlyArray<number>): Uint8Array => {
  const data = new Uint8Array(values.length * 4)
  const view = new DataView(data.buffer)
  values.forEach((value, index) => view.setFloat32(index * 4, value, true))
  return data
}

// This suite uses the host native addon and temporary files. Tests release every
// concrete handle returned by execution or loading.
testLayer(backendLayer)("CPU tensor handles and direct safetensors", (it) => {
  it.effect("creates frozen metadata-only handles and accepts concrete tensors as node inputs", () =>
    Effect.gen(function*() {
      const runtime = yield* Runtime.Runtime
      const lazy = yield* runtime.node({
        op: "fromBytes",
        inputs: [],
        attributes: { data: f32Bytes([-1, 2]), shape: [2], dtype: "f32" }
      })
      expect(Object.isFrozen(lazy)).toBe(true)
      expect(Object.isFrozen(lazy.shape)).toBe(true)
      expect(Object.isFrozen(lazy.placement)).toBe(true)
      expect(lazy).toMatchObject({ _tag: "LazyTensor", shape: [2], dtype: "f32", device: "cpu" })
      expect(lazy).not.toHaveProperty("lazy")
      expect(lazy).not.toHaveProperty("materialized")

      const initial = yield* runtime.compile({ roots: [lazy] })
      const [concrete] = yield* runtime.execute(initial, { bindings: [], scalars: [], runtimeValues: {} })
      const relu = yield* runtime.node({ op: "relu", inputs: [concrete] })
      const executable = yield* runtime.compile({ roots: [relu] })
      const [result] = yield* runtime.execute(executable, { bindings: [], scalars: [], runtimeValues: {} })
      expect([...new Float32Array(yield* runtime.readback(result))]).toEqual([0, 2])
      yield* runtime.release(result)
      yield* runtime.release(concrete)
    }))

  it.effect("exports logical values from noncontiguous tensors through readback", () =>
    Effect.gen(function*() {
      const runtime = yield* Runtime.Runtime
      const source = yield* runtime.node({
        op: "fromBytes",
        inputs: [],
        attributes: { data: f32Bytes([1, 2, 3, 4, 5, 6]), shape: [2, 3], dtype: "f32" }
      })
      const transposed = yield* runtime.node({
        op: "permute",
        inputs: [source],
        attributes: { dims: [1, 0] }
      })
      const executable = yield* runtime.compile({ roots: [transposed] })
      const [value] = yield* runtime.execute(executable, { bindings: [], scalars: [], runtimeValues: {} })
      expect(value.shape).toEqual([3, 2])
      expect([...new Float32Array(yield* runtime.readback(value))]).toEqual([1, 4, 2, 5, 3, 6])
      yield* runtime.release(value)
    }))

  it.effect("round trips direct metadata and escaped names", () =>
    withTempFile("effect-torch-cpu-safetensors-", (file) =>
      Effect.gen(function*() {
        const runtime = yield* Runtime.Runtime
        const name = "quoted\"\\\nname"
        const lazy = yield* runtime.node({
          op: "fromBytes",
          inputs: [],
          attributes: { data: new Uint8Array([7, 8]), shape: [2], dtype: "u8" }
        })
        const executable = yield* runtime.compile({ roots: [lazy] })
        const [tensor] = yield* runtime.execute(executable, { bindings: [], scalars: [], runtimeValues: {} })
        yield* runtime.extensions.pathSafetensors.save(file, {
          entries: [{ name, tensor }],
          metadata: { framework: "effect-torch", escaped: "\"\\\n" }
        })
        const archive = yield* runtime.extensions.pathSafetensors.load(file)
        expect(archive.entries.map((entry) => entry.name)).toEqual([name])
        expect(archive.metadata).toEqual({ framework: "effect-torch", escaped: "\"\\\n" })
        const loaded = archive.entries[0]!.tensor
        expect(loaded).toMatchObject({ _tag: "Tensor", shape: [2], dtype: "u8", device: "cpu" })
        expect([...new Uint8Array(yield* runtime.readback(loaded))]).toEqual([7, 8])
        yield* runtime.release(loaded)
        yield* runtime.release(tensor)
      })))

  it.effect("rejects malformed direct input", () =>
    withTempFile("effect-torch-cpu-safetensors-bad-", (file) =>
      Effect.gen(function*() {
        const runtime = yield* Runtime.Runtime
        yield* Effect.tryPromise(() => writeFile(file, new Uint8Array([1, 2, 3])))
        const error = yield* Effect.flip(runtime.extensions.pathSafetensors.load(file))
        expect(error.reason).toBe("io-failed")
      })))
})

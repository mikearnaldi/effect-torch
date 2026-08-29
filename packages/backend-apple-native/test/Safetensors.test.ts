import { Runtime } from "@effect-torch/core"
import { describe, expect, it } from "@effect/vitest"
import { Effect } from "effect"
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import path from "node:path"
import { isAvailable, layer as makeBackendLayer } from "../src/index.ts"

const backendLayer = makeBackendLayer()

const withTempFile = <A, E, R>(prefix: string, use: (file: string) => Effect.Effect<A, E, R>) =>
  Effect.acquireUseRelease(
    Effect.tryPromise(() => mkdtemp(path.join(tmpdir(), prefix))),
    (directory) => use(path.join(directory, "archive.safetensors")),
    (directory) => Effect.promise(() => rm(directory, { recursive: true, force: true }))
  )

const suite = Effect.runSync(isAvailable) ? describe : describe.skip

const writeF64Archive = (file: string): Promise<void> => {
  const encoded = new TextEncoder().encode(
    JSON.stringify({ x: { dtype: "F64", shape: [1], data_offsets: [0, 8] } })
  )
  const headerLength = Math.ceil(encoded.byteLength / 8) * 8
  const archive = new Uint8Array(8 + headerLength + 8)
  new DataView(archive.buffer).setBigUint64(0, BigInt(headerLength), true)
  archive.fill(0x20, 8, 8 + headerLength)
  archive.set(encoded, 8)
  return writeFile(file, archive)
}

// These file tests run only when Metal is available and never substitute CPU
// behavior. They release concrete execution results and loaded handles.
suite("Metal tensor handles and safetensors file I/O", () => {
  it("rejects an unavailable Metal ordinal", async () => {
    await expect(
      Effect.runPromise(Runtime.Runtime.pipe(Effect.provide(makeBackendLayer({ device: 0xffff_ffff }))))
    ).rejects.toThrow(/metal:4294967295 is unavailable/)
  })

  it.effect("writes and reloads Metal tensors and metadata", () =>
    withTempFile("effect-torch-metal-safetensors-", (file) =>
      Effect.gen(function*() {
        const runtime = yield* Runtime.Runtime
        expect(runtime.placement).toMatchObject({ id: "metal:0", deviceType: "metal", ordinal: 0 })
        const tensor = yield* runtime.node({
          op: "fromBytes",
          inputs: [],
          attributes: { data: new Uint8Array([7, 8]), shape: [2], dtype: "u8" }
        })
        expect(Object.isFrozen(tensor)).toBe(true)
        expect(tensor).toMatchObject({ _tag: "LazyTensor", shape: [2], dtype: "u8", device: "metal" })

        const executable = yield* runtime.compile({ roots: [tensor] })
        const [concrete] = yield* runtime.execute(executable, {
          bindings: [],
          scalars: [],
          runtimeValues: {}
        })
        expect(yield* runtime.extensions.diagnostics.externalMemoryBytes).toBeGreaterThanOrEqual(4096)

        yield* runtime.extensions.pathSafetensors.save(file, {
          entries: [{ name: "values", tensor: concrete }],
          metadata: { framework: "effect-torch" }
        })
        yield* runtime.release(concrete)
        const archive = yield* runtime.extensions.pathSafetensors.load(file)
        expect(archive.metadata).toEqual({ framework: "effect-torch" })
        const loaded = archive.entries[0]!.tensor
        expect(loaded).toMatchObject({ _tag: "Tensor", shape: [2], dtype: "u8", device: "metal" })
        expect([...new Uint8Array(yield* runtime.readback(loaded))]).toEqual([7, 8])
        yield* runtime.release(loaded)
      }).pipe(Effect.provide(backendLayer))))

  it.effect("rejects f64 archives instead of falling back to CPU", () =>
    withTempFile("effect-torch-metal-safetensors-f64-", (file) =>
      Effect.gen(function*() {
        const runtime = yield* Runtime.Runtime
        yield* Effect.tryPromise(() => writeF64Archive(file))
        const error = yield* Effect.flip(runtime.extensions.pathSafetensors.load(file))
        expect(error.reason).toBe("unsupported-placement")
      }).pipe(Effect.provide(backendLayer))))
})

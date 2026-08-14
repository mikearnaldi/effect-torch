import { describe, expect, it } from "@effect/vitest"
import { Effect } from "effect"
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import path from "node:path"
import { isAvailable, makeRuntime } from "../src/index.ts"

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

// Real artifact coverage runs only with Metal available and never substitutes
// CPU behavior; concrete execution and loaded handles are explicitly released.
suite("Apple Metal tensor handles and direct safetensors", () => {
  it.effect("round trips direct Metal tensors and metadata", () =>
    withTempFile("effect-torch-metal-safetensors-", (file) =>
      Effect.gen(function*() {
        const runtime = makeRuntime()
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
      })))

  it.effect("rejects f64 direct loading instead of executing on CPU", () =>
    withTempFile("effect-torch-metal-safetensors-f64-", (file) =>
      Effect.gen(function*() {
        const runtime = makeRuntime()
        yield* Effect.tryPromise(() => writeF64Archive(file))
        const error = yield* Effect.flip(runtime.extensions.pathSafetensors.load(file))
        expect(error.reason).toBe("unsupported-placement")
      })))
})

import { describe, expect, it } from "@effect/vitest"
import { Deferred, Effect, Fiber } from "effect"
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import path from "node:path"
import { vi } from "vitest"
import { isAvailable, makeRuntime } from "../src/index.ts"
import { makeRuntime as makeRuntimeAdapter } from "../src/internal/adapter.ts"
import type { NativeAddon, NativeGgufTensorDescriptor, NativeTensor } from "../src/internal/native-addon.ts"

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

const fixture = (): Buffer => {
  const header = Buffer.concat([
    Buffer.from("GGUF"),
    u32(3),
    u64(3),
    u64(2),
    string("general.architecture"),
    u32(8),
    string("direct"),
    string("general.quantization_version"),
    u32(4),
    u32(2),
    string("dense"),
    u32(1),
    u64(2),
    u32(0),
    u64(0),
    string("q2"),
    u32(2),
    u64(3072),
    u64(1),
    u32(10),
    u64(32),
    string("q4"),
    u32(2),
    u64(1792),
    u64(1),
    u32(12),
    u64(1056)
  ])
  const data = Buffer.alloc(2064)
  data.writeFloatLE(1.5, 0)
  data.writeFloatLE(-2.25, 4)
  for (let index = 0; index < 1008; index++) {
    data[32 + index] = index % 251
    data[1056 + index] = (250 - index) & 0xff
  }
  return Buffer.concat([header, Buffer.alloc((32 - header.length % 32) % 32), data])
}

const withFixture = <A, E, R>(use: (file: string) => Effect.Effect<A, E, R>) =>
  Effect.acquireUseRelease(
    Effect.tryPromise(async () => {
      const directory = await mkdtemp(path.join(tmpdir(), "effect-torch-metal-gguf-"))
      const file = path.join(directory, "archive.gguf")
      await writeFile(file, fixture())
      return { directory, file }
    }),
    ({ file }) => use(file),
    ({ directory }) => Effect.promise(() => rm(directory, { recursive: true, force: true }))
  )

const suite = Effect.runSync(isAvailable) ? describe : describe.skip

suite("Metal direct GGUF", () => {
  it.effect("loads exact payloads and rejects cross-codec physical collisions", () =>
    withFixture((file) =>
      Effect.gen(function*() {
        const runtime = makeRuntime()
        const archive = yield* runtime.extensions.gguf!.load(file)
        const dense = archive.entries.find((entry) => entry.descriptor.name === "dense")!
        const q2 = archive.entries.find((entry) => entry.descriptor.name === "q2")!
        const q4 = archive.entries.find((entry) => entry.descriptor.name === "q4")!

        expect([...new Float32Array(yield* runtime.readback(dense.tensor))]).toEqual([1.5, -2.25])
        expect([...new Uint8Array(yield* runtime.readback(q2.tensor))]).toEqual(
          Array.from({ length: 1008 }, (_, index) => index % 251)
        )
        expect(q2.descriptor.physicalShape).toEqual([1, 1008])
        expect(q4.descriptor.physicalShape).toEqual([1, 1008])

        const saveError = yield* Effect.flip(runtime.extensions.pathSafetensors!.save(
          path.join(path.dirname(file), "encoded.safetensors"),
          { entries: [{ name: "q2", tensor: q2.tensor }], metadata: {} }
        ))
        expect(saveError.reason).toBe("unsupported-layout")

        const q2Storage = {
          encoding: "Q2_K" as const,
          physicalShape: q2.descriptor.physicalShape,
          physicalDtype: "u8" as const
        }
        const input = yield* runtime.node({
          op: "input",
          inputs: [q2.tensor],
          attributes: {
            slot: 0,
            shape: q2.descriptor.logicalShape,
            dtype: "f32",
            storage: q2Storage
          }
        })
        const conflicting = yield* runtime.node({
          op: "input",
          inputs: [q4.tensor],
          attributes: {
            slot: 0,
            shape: q4.descriptor.logicalShape,
            dtype: "f32",
            storage: {
              encoding: "Q4_K",
              physicalShape: q4.descriptor.physicalShape,
              physicalDtype: "u8"
            }
          }
        })
        const repeated = yield* Effect.flip(runtime.compile({ roots: [input, conflicting] }))
        expect(repeated.message).toContain("conflicting logical declarations")

        const executable = yield* runtime.compile({ roots: [input] })
        const mismatch = yield* Effect.flip(runtime.execute(executable, {
          bindings: [q4.tensor],
          scalars: [],
          runtimeValues: {}
        }))
        expect(mismatch.message).toContain("does not match its compiled logical declaration")

        const [identity] = yield* runtime.execute(executable, {
          bindings: [q2.tensor],
          scalars: [],
          runtimeValues: {}
        })
        expect(identity.storage?.encoding).toBe("Q2_K")
        expect([...new Uint8Array(yield* runtime.readback(identity))]).toEqual(
          Array.from({ length: 1008 }, (_, index) => index % 251)
        )
        yield* runtime.release(identity)

        const scalar = yield* runtime.node({
          op: "scalarInput",
          inputs: [],
          attributes: { slot: 0, dtype: "f32" }
        })
        const shifted = yield* runtime.node({
          op: "input",
          inputs: [q2.tensor],
          attributes: { slot: 1, shape: q2.descriptor.logicalShape, dtype: "f32", storage: q2Storage }
        })
        const shiftedAgain = yield* runtime.node({
          op: "input",
          inputs: [q2.tensor],
          attributes: { slot: 1, shape: q2.descriptor.logicalShape, dtype: "f32", storage: q2Storage }
        })
        const interleaved = yield* runtime.compile({ roots: [scalar, shifted, shiftedAgain] })
        const values = yield* runtime.execute(interleaved, {
          bindings: [q2.tensor],
          scalars: [7],
          runtimeValues: {}
        })
        expect([...new Float32Array(yield* runtime.readback(values[0]!))]).toEqual([7])
        expect(values[1]!.storage?.encoding).toBe("Q2_K")
        expect(values[2]!.storage?.encoding).toBe("Q2_K")
        for (const value of values) yield* runtime.release(value)
        for (const entry of archive.entries) yield* runtime.release(entry.tensor)
      })
    ))
})

it.effect("rejects duplicate raw Metal GGUF ownership and clears it once", () => {
  const clear = vi.fn()
  const tensor = { shape: [1, 1008], dtype: "u8", device: "metal", clear } as unknown as NativeTensor
  const descriptor: NativeGgufTensorDescriptor = {
    name: "q2",
    format: "Q2_K",
    logicalShape: [1, 3072],
    logicalDtype: "f32",
    physicalShape: [1, 1008],
    physicalDtype: "u8"
  }
  class Token {
    cancelled = false
    cancel() {
      this.cancelled = true
    }
  }
  const native = {
    CancellationToken: Token,
    loadGguf: async () => ({
      entries: [{ descriptor, tensor }, { descriptor: { ...descriptor, name: "other" }, tensor }]
    })
  } as unknown as NativeAddon
  const runtime = makeRuntimeAdapter(native)

  return Effect.gen(function*() {
    const error = yield* Effect.flip(runtime.extensions.gguf!.load("duplicate.gguf"))
    expect(error.message).toContain("duplicate tensor ownership")
    expect(clear).toHaveBeenCalledTimes(1)
  })
})

it.effect("clears a late native Metal GGUF success after interruptible I/O is cancelled", () =>
  Effect.gen(function*() {
    const started = yield* Deferred.make<void>()
    const clear = vi.fn()
    const tensor = { shape: [2], dtype: "f32", device: "metal", clear } as unknown as NativeTensor
    const descriptor: NativeGgufTensorDescriptor = {
      name: "dense",
      format: "F32",
      logicalShape: [2],
      logicalDtype: "f32",
      physicalShape: [2],
      physicalDtype: "f32"
    }
    let resolve!: (
      archive: { entries: Array<{ descriptor: NativeGgufTensorDescriptor; tensor: NativeTensor }> }
    ) => void
    class Token {
      cancelled = false
      cancel() {
        this.cancelled = true
      }
    }
    const native = {
      CancellationToken: Token,
      loadGguf: () => {
        Effect.runSync(Deferred.succeed(started, undefined))
        return new Promise((resume) => resolve = resume)
      }
    } as unknown as NativeAddon
    const runtime = makeRuntimeAdapter(native)
    const fiber = yield* runtime.extensions.gguf!.load("late.gguf").pipe(
      Effect.forkChild({ startImmediately: true })
    )
    yield* Deferred.await(started)
    const interruption = yield* Fiber.interrupt(fiber).pipe(Effect.forkChild({ startImmediately: true }))
    yield* Effect.sync(() => resolve({ entries: [{ descriptor, tensor }] }))
    yield* Fiber.join(interruption)
    yield* Effect.promise(() => Promise.resolve())

    expect(clear).toHaveBeenCalledTimes(1)
  }))

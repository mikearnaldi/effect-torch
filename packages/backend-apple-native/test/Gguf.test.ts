import { Runtime } from "@effect-torch/core"
import { describe, expect, it } from "@effect/vitest"
import { Deferred, Effect, Fiber } from "effect"
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import path from "node:path"
import { vi } from "vitest"
import { isAvailable, layer as makeBackendLayer } from "../src/index.ts"
import { createRuntimeAdapter } from "../src/internal/adapter.ts"
import type { NativeAddon, NativeGgufTensorDescriptor } from "../src/internal/native-addon.js"

const backendLayer = makeBackendLayer()

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

// This GGUF v3 fixture has dense F32 data and Q2_K and Q4_K payloads with the
// same packed shape. It checks that codec identity does not come from shape.
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

type GgufLoadDouble = (
  ...args: Parameters<NativeAddon["loadGguf"]>
) => Promise<{
  entries: Array<{
    descriptor: NativeGgufTensorDescriptor
    tensor: {
      clear(): void
      device: string
      dtype: string
      shape: Array<number>
      storage?: { representation: string; format?: string }
    }
  }>
}>

class Token {
  cancelled = false
  cancel() {
    this.cancelled = true
  }
}

const makeNativeAddonDouble = (loadGguf: GgufLoadDouble): NativeAddon => {
  const loadGgufForDevice = (
    path: string,
    _deviceOrdinal: number,
    token?: Parameters<NativeAddon["loadGguf"]>[1]
  ) => loadGguf(path, token)
  const addon = { CancellationToken: Token, loadGgufForDevice }
  // SAFETY: GGUF ownership tests use only the typed loadGgufForDevice and CancellationToken fields.
  return addon as NativeAddon
}

const suite = Effect.runSync(isAvailable) ? describe : describe.skip

// File I/O cases require Metal. The ownership cases below inject a fake addon
// and run on any platform.
suite("Metal GGUF file I/O", () => {
  it.effect("loads logical packed tensors and rejects codec mismatches with identical physical shapes", () =>
    withFixture((file) =>
      Effect.gen(function*() {
        const runtime = yield* Runtime.Runtime
        const archive = yield* runtime.extensions.gguf.load(file)
        const dense = archive.entries.find((entry) => entry.descriptor.name === "dense")!
        const q2 = archive.entries.find((entry) => entry.descriptor.name === "q2")!
        const q4 = archive.entries.find((entry) => entry.descriptor.name === "q4")!

        expect([...new Float32Array(yield* runtime.readback(dense.tensor))]).toEqual([1.5, -2.25])
        expect(q2.tensor.shape).toEqual(q2.descriptor.logicalShape)
        expect(q2.tensor.dtype).toBe("f32")
        expect(q2.tensor.storage?.encoding).toBe("Q2_K")
        const packedReadback = yield* Effect.flip(runtime.readback(q2.tensor))
        expect(packedReadback.message).toContain("packed")
        expect(q2.descriptor.physicalShape).toEqual([1, 1008])
        expect(q4.descriptor.physicalShape).toEqual([1, 1008])

        const saveError = yield* Effect.flip(runtime.extensions.pathSafetensors.save(
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

        const rootError = yield* Effect.flip(runtime.compile({ roots: [input] }))
        expect(rootError.message).toContain("packed program outputs")
        const indexes = yield* runtime.node({ op: "zeros", inputs: [], attributes: { shape: [1], dtype: "u32" } })
        const embedded = yield* runtime.node({ op: "quantizedEmbedding", inputs: [indexes, input], attributes: {} })
        const executable = yield* runtime.compile({ roots: [embedded] })
        const mismatch = yield* Effect.flip(runtime.execute(executable, {
          bindings: [q4.tensor],
          scalars: [],
          runtimeValues: {}
        }))
        expect(mismatch.message).toContain("does not match its compiled logical declaration")

        const [result] = yield* runtime.execute(executable, {
          bindings: [q2.tensor],
          scalars: [],
          runtimeValues: {}
        })
        expect(result.storage).toBeUndefined()
        expect(result.shape).toEqual(q2.tensor.shape)
        expect(result.dtype).toBe("f32")
        expect(new Float32Array(yield* runtime.readback(result)).length).toBe(3072)
        yield* runtime.release(result)

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
        const first = yield* runtime.node({ op: "quantizedEmbedding", inputs: [indexes, shifted], attributes: {} })
        const second = yield* runtime.node({
          op: "quantizedEmbedding",
          inputs: [indexes, shiftedAgain],
          attributes: {}
        })
        const interleaved = yield* runtime.compile({ roots: [scalar, first, second] })
        const values = yield* runtime.execute(interleaved, {
          bindings: [q2.tensor],
          scalars: [7],
          runtimeValues: {}
        })
        expect([...new Float32Array(yield* runtime.readback(values[0]!))]).toEqual([7])
        expect(values[1]!.storage).toBeUndefined()
        expect(values[2]!.storage).toBeUndefined()
        expect(values[1]!.shape).toEqual([1, 3072])
        expect(values[2]!.shape).toEqual([1, 3072])
        for (const value of values) yield* runtime.release(value)
        for (const entry of archive.entries) yield* runtime.release(entry.tensor)
      })
    ).pipe(Effect.provide(backendLayer)))
})

// Each raw native wrapper must become one public handle or be cleared exactly
// once. This also applies when a successful result races with interruption.
it.effect("rejects duplicate GGUF wrapper ownership and clears the wrapper once", () => {
  const clear = vi.fn()
  const tensor = { shape: [1, 1008], dtype: "u8", device: "metal", clear }
  const descriptor: NativeGgufTensorDescriptor = {
    name: "q2",
    format: "Q2_K",
    logicalShape: [1, 3072],
    logicalDtype: "f32",
    physicalShape: [1, 1008],
    physicalDtype: "u8"
  }
  const native = makeNativeAddonDouble(
    async () => ({
      entries: [{ descriptor, tensor }, { descriptor: { ...descriptor, name: "other" }, tensor }]
    })
  )
  const runtime = createRuntimeAdapter(native)

  return Effect.gen(function*() {
    const error = yield* Effect.flip(runtime.extensions.gguf.load("duplicate.gguf"))
    expect(error.message).toContain("duplicate tensor ownership")
    expect(clear).toHaveBeenCalledTimes(1)
  })
})

it.effect("clears late GGUF results after I/O interruption", () =>
  Effect.gen(function*() {
    const started = yield* Deferred.make<void>()
    const clear = vi.fn()
    const tensor = { shape: [2], dtype: "f32", device: "metal", clear }
    const descriptor: NativeGgufTensorDescriptor = {
      name: "dense",
      format: "F32",
      logicalShape: [2],
      logicalDtype: "f32",
      physicalShape: [2],
      physicalDtype: "f32"
    }
    let resolve!: (
      archive: { entries: Array<{ descriptor: NativeGgufTensorDescriptor; tensor: typeof tensor }> }
    ) => void
    const native = makeNativeAddonDouble(
      () => {
        Effect.runSync(Deferred.succeed(started, undefined))
        return new Promise<{ entries: Array<{ descriptor: NativeGgufTensorDescriptor; tensor: typeof tensor }> }>(
          (resume) => resolve = resume
        )
      }
    )
    const runtime = createRuntimeAdapter(native)
    const fiber = yield* runtime.extensions.gguf.load("late.gguf").pipe(
      Effect.forkChild({ startImmediately: true })
    )
    yield* Deferred.await(started)
    const interruption = yield* Fiber.interrupt(fiber).pipe(Effect.forkChild({ startImmediately: true }))
    yield* Effect.sync(() => resolve({ entries: [{ descriptor, tensor }] }))
    yield* Fiber.join(interruption)
    yield* Effect.promise(() => Promise.resolve())

    expect(clear).toHaveBeenCalledTimes(1)
  }))

it.effect("rejects invalid native packed metadata and clears unpublished GGUF tensors", () =>
  Effect.gen(function*() {
    const descriptor: NativeGgufTensorDescriptor = {
      name: "q2",
      format: "Q2_K",
      logicalShape: [1, 3072],
      logicalDtype: "f32",
      physicalShape: [1, 1008],
      physicalDtype: "u8"
    }
    const invalidMetadata = [
      { shape: [1, 1008], dtype: "u8", storage: { representation: "dense" } },
      { shape: [1, 3072], dtype: "f32", storage: { representation: "dense" } },
      { shape: [1, 3072], dtype: "f32", storage: { representation: "packed", format: "Q4_K" } },
      { shape: [1, 3072], dtype: "f32", storage: { representation: "packed", format: "Q7_K" } },
      { shape: [1, 3072], dtype: "f32", storage: { representation: "vendor", format: "Q2_K" } },
      { shape: [1, 3072], dtype: "f32", storage: { representation: "dense", format: "Q2_K" } },
      { shape: [1, 3071], dtype: "f32", storage: { representation: "packed", format: "Q2_K" } }
    ]
    for (const metadata of invalidMetadata) {
      const clear = vi.fn()
      const tensor = { ...metadata, device: "metal", clear }
      const runtime = createRuntimeAdapter(makeNativeAddonDouble(async () => ({ entries: [{ descriptor, tensor }] })))
      const error = yield* Effect.flip(runtime.extensions.gguf.load("invalid-metadata.gguf"))
      expect(error.reason).toBe("io-failed")
      expect(error.message).toMatch(/logical declaration|storage representation|packed tensor geometry/)
      expect(clear).toHaveBeenCalledTimes(1)
    }
  }))

it.effect("passes logical F32 inputs and packed storage to the native graph", () =>
  Effect.gen(function*() {
    const shape = [2, 256]
    const storage = { encoding: "Q4_K", physicalShape: [2, 144], physicalDtype: "u8" } as const
    const nativeStorage = { representation: "packed", format: "Q4_K" }
    const input = vi.fn(() => ({ metadata: () => [shape, "f32"], storage: nativeStorage }))
    // SAFETY: This test exercises only LazyTensor.input and the returned metadata.
    // oxlint-disable-next-line anti-slop/no-chained-type-assertions -- The injected addon implements only the native methods exercised by this test.
    const native = { CancellationToken: Token, LazyTensor: { input } } as unknown as NativeAddon
    const runtime = createRuntimeAdapter(native)
    const handle = yield* runtime.node({
      op: "input",
      inputs: [],
      attributes: { slot: 0, shape, dtype: "f32", storage }
    })
    expect(input).toHaveBeenCalledWith(0, shape, "f32", 0, nativeStorage)
    expect(handle.shape).toEqual(shape)
    expect(handle.dtype).toBe("f32")
    expect(handle.storage).toEqual(storage)

    input.mockReturnValue({ metadata: () => [[2, 144], "u8"], storage: nativeStorage })
    const error = yield* Effect.flip(runtime.node({
      op: "input",
      inputs: [],
      attributes: { slot: 1, shape, dtype: "f32", storage }
    }))
    expect(error.message).toMatch(/logical declaration|packed tensor geometry/)
  }))

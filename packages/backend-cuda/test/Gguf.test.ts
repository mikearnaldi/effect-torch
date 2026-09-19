import { expect, it } from "@effect/vitest"
import { Deferred, Effect, Fiber } from "effect"
import { vi } from "vitest"
import { createRuntimeAdapter } from "../src/internal/adapter.ts"
import type { NativeAddon, NativeGgufTensorDescriptor } from "../src/internal/native-addon.js"

it.effect("forwards a partial selection and clears every late GGUF tensor independently", () =>
  Effect.gen(function*() {
    const started = yield* Deferred.make<void>()
    class CancellationTokenDouble {
      cancelled = false
      cancel() {
        this.cancelled = true
      }
    }
    class CudaRuntimeDouble {
      fromMaterialized() {
        throw new Error("unexpected tensor wrapping")
      }
    }
    const firstClear = vi.fn(() => {
      throw new Error("clear failed")
    })
    const secondClear = vi.fn()
    const descriptor: NativeGgufTensorDescriptor = {
      name: "first",
      format: "F32",
      logicalShape: [1],
      logicalDtype: "f32",
      physicalShape: [1],
      physicalDtype: "f32"
    }
    const archive = {
      entries: [
        { descriptor, tensor: { clear: firstClear } },
        { descriptor: { ...descriptor, name: "second" }, tensor: { clear: secondClear } }
      ]
    }
    let resume!: (value: typeof archive) => void
    let token: CancellationTokenDouble | undefined
    const addon = {
      CudaRuntime: CudaRuntimeDouble,
      CancellationToken: CancellationTokenDouble,
      loadGgufForDevice: (
        path: string,
        ordinal: number,
        cancellation: CancellationTokenDouble,
        names: Array<string>
      ) => {
        expect(path).toBe("partial.gguf")
        expect(ordinal).toBe(0)
        expect(names).toEqual(["first", "second"])
        token = cancellation
        Effect.runSync(Deferred.succeed(started, undefined))
        return new Promise<typeof archive>((resolve) => resume = resolve)
      }
    }
    // SAFETY: interruption discards these native wrappers before their tensor metadata is accessed.
    // oxlint-disable-next-line anti-slop/no-chained-type-assertions -- Only the cancellation/loading boundary is exercised.
    const runtime = createRuntimeAdapter(addon as unknown as NativeAddon, 0)
    const fiber = yield* runtime.extensions.gguf.load("partial.gguf", { names: ["first", "second"] }).pipe(
      Effect.forkChild({ startImmediately: true })
    )
    yield* Deferred.await(started)
    const stopped = yield* Fiber.interrupt(fiber).pipe(Effect.forkChild({ startImmediately: true }))
    yield* Effect.sync(() => resume(archive))
    yield* Fiber.join(stopped)
    yield* Effect.promise(() => Promise.resolve())
    expect(token?.cancelled).toBe(true)
    expect(firstClear).toHaveBeenCalledTimes(1)
    expect(secondClear).toHaveBeenCalledTimes(1)
  }))

it.effect("rejects duplicate GGUF ownership and clears the native wrapper once", () =>
  Effect.gen(function*() {
    class CancellationTokenDouble {
      cancel() {}
    }
    class CudaRuntimeDouble {
      fromMaterialized() {
        throw new Error("unexpected tensor wrapping")
      }
    }
    const clear = vi.fn()
    const tensor = { clear }
    const descriptor: NativeGgufTensorDescriptor = {
      name: "first",
      format: "F32",
      logicalShape: [1],
      logicalDtype: "f32",
      physicalShape: [1],
      physicalDtype: "f32"
    }
    const addon = {
      CudaRuntime: CudaRuntimeDouble,
      CancellationToken: CancellationTokenDouble,
      loadGgufForDevice: async () => ({
        entries: [
          { descriptor, tensor },
          { descriptor: { ...descriptor, name: "second" }, tensor }
        ]
      })
    }
    // SAFETY: duplicate identity is rejected before the native tensor's metadata is accessed.
    // oxlint-disable-next-line anti-slop/no-chained-type-assertions -- Only the duplicate ownership guard is exercised.
    const runtime = createRuntimeAdapter(addon as unknown as NativeAddon, 0)
    const error = yield* Effect.flip(runtime.extensions.gguf.load("duplicates.gguf", { names: ["first", "second"] }))
    expect(error.message).toContain("duplicate tensor ownership")
    expect(clear).toHaveBeenCalledTimes(1)
  }))

it.effect("passes logical F32 inputs and packed storage to the native CUDA graph", () =>
  Effect.gen(function*() {
    const shape = [2, 256]
    const storage = { encoding: "Q4_K", physicalShape: [2, 144], physicalDtype: "u8" } as const
    const nativeStorage = { representation: "packed", format: "Q4_K" }
    const graphNode = vi.fn(() => ({ shape, dtype: "f32", device: "cuda:0", storage: nativeStorage }))
    class CudaRuntimeDouble {
      graphNode = graphNode
    }
    // SAFETY: This test exercises only CudaRuntime.graphNode and the returned metadata.
    // oxlint-disable-next-line anti-slop/no-chained-type-assertions -- The injected addon implements only the native methods exercised by this test.
    const native = { CudaRuntime: CudaRuntimeDouble } as unknown as NativeAddon
    const runtime = createRuntimeAdapter(native, 0)
    const handle = yield* runtime.node({
      op: "input",
      inputs: [],
      attributes: { slot: 0, shape, dtype: "f32", storage }
    })
    expect(graphNode).toHaveBeenCalledWith(
      "input",
      [],
      JSON.stringify({ slot: 0, shape, dtype: "f32", storage: nativeStorage })
    )
    expect(handle.shape).toEqual(shape)
    expect(handle.dtype).toBe("f32")
    expect(handle.storage).toEqual(storage)

    graphNode.mockReturnValue({ shape: [2, 144], dtype: "u8", device: "cuda:0", storage: nativeStorage })
    const error = yield* Effect.flip(runtime.node({
      op: "input",
      inputs: [],
      attributes: { slot: 1, shape, dtype: "f32", storage }
    }))
    expect(error.message).toMatch(/logical declaration|packed tensor geometry/)
  }))

it.effect("rejects invalid native packed metadata and clears unpublished CUDA GGUF tensors", () =>
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
    class CancellationTokenDouble {
      cancelled = false
      cancel() {
        this.cancelled = true
      }
    }
    const fromMaterialized = vi.fn()
    class CudaRuntimeDouble {
      fromMaterialized = fromMaterialized
    }
    for (const metadata of invalidMetadata) {
      const clear = vi.fn()
      const tensor = { ...metadata, device: "cuda:0", clear }
      const addon = {
        CudaRuntime: CudaRuntimeDouble,
        CancellationToken: CancellationTokenDouble,
        loadGgufForDevice: async () => ({ entries: [{ descriptor, tensor }] })
      }
      // SAFETY: This test exercises only GGUF loading and rejects each tensor before graph creation.
      // oxlint-disable-next-line anti-slop/no-chained-type-assertions -- The injected addon implements only the native methods exercised by this test.
      const runtime = createRuntimeAdapter(addon as unknown as NativeAddon, 0)
      const error = yield* Effect.flip(runtime.extensions.gguf.load("invalid-metadata.gguf"))
      expect(error.reason).toBe("io-failed")
      expect(error.message).toMatch(/logical declaration|storage representation|packed tensor geometry/)
      expect(clear).toHaveBeenCalledTimes(1)
      expect(fromMaterialized).not.toHaveBeenCalled()
    }
  }))

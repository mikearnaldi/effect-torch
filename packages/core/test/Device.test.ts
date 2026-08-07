import * as BackendNative from "@effect-torch/backend-native"
import { expect, layer } from "@effect/vitest"
import { Effect } from "effect"
import { Runtime, Tensor } from "../src/index.ts"

layer(BackendNative.Best)("Runtime", (it) => {
  it.effect("Best provides an available runtime", () =>
    Effect.gen(function*() {
      const runtime = yield* Runtime.Runtime
      expect(runtime.backend.name).toBe("@effect-torch/backend-native")
      expect(BackendNative.isAvailable(runtime.placement.deviceType as "cpu" | "metal")).toBe(true)
    }))

  it.effect("rejects foreign graph and buffer handles", () =>
    Effect.gen(function*() {
      const graph = BackendNative.cpu.graph.ones([1], "f32")
      const graphError = yield* Effect.flip(BackendNative.metal.validateGraph([graph]))
      expect(graphError.reason).toBe("foreign-handle")

      const [value] = yield* BackendNative.cpu.evaluate([graph])
      const bufferError = yield* Effect.flip(BackendNative.metal.readback(value.handle))
      expect(bufferError.reason).toBe("foreign-handle")
      yield* BackendNative.cpu.releaseBuffer(value.handle)
      const releasedBufferError = yield* Effect.flip(BackendNative.cpu.readback(value.handle))
      expect(releasedBufferError.reason).toBe("invalid-handle")

      const decode = BackendNative.cpu.extensions.decode!
      const pool = yield* decode.makePool({
        layers: 1,
        kvHeads: 1,
        headDim: 2,
        maxTokens: 16,
        blockSize: 4,
        dtype: "f32"
      })
      const sequence = yield* decode.makeSequence(pool)
      yield* decode.releaseSequence(sequence)
      const releasedSequenceError = yield* Effect.flip(decode.sequenceCursor(sequence))
      expect(releasedSequenceError.reason).toBe("invalid-handle")
    }))

  it.effect("rejects foreign tensors during graph construction", () =>
    Effect.gen(function*() {
      const runtime = yield* Runtime.Runtime
      if (runtime.placement.deviceType === "cpu" && !BackendNative.isAvailable("metal")) return
      const foreign = runtime.placement.deviceType === "cpu" ? BackendNative.metal : BackendNative.cpu
      const tensor = Tensor.makeLazy(foreign.graph.ones([1], "f32"), [1], "f32", foreign.placement)
      const error = yield* Effect.flip(Tensor.relu(tensor))
      expect(error.backend?.reason).toBe("foreign-handle")
    }))

  it("partitions compiled signatures by runtime identity", () => {
    const tensor = Tensor.makeLazy(
      BackendNative.cpu.graph.ones([1], "f32"),
      [1],
      "f32",
      BackendNative.cpu.placement
    )
    const other: Runtime.RuntimeService = { ...BackendNative.cpu, identity: {} }
    expect(Tensor.signatureOf([tensor], BackendNative.cpu)).not.toBe(Tensor.signatureOf([tensor], other))
  })
})

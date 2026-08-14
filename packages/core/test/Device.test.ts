import * as BackendApple from "@effect-torch/backend-apple-native"
import * as BackendCpu from "@effect-torch/backend-cpu"
import { expect, it as test, layer } from "@effect/vitest"
import { Effect, Layer } from "effect"
import { Runtime, Tensor } from "../src/index.ts"

const metalAvailable = Effect.runSync(BackendApple.isAvailable)

// Handle ownership belongs to a runtime identity domain, not merely a device
// label. Cross-runtime and post-release checks must therefore reject before use.
layer(BackendCpu.layer)("Runtime", (it) => {
  it.effect("provides the CPU runtime", () =>
    Effect.gen(function*() {
      const runtime = yield* Runtime.Runtime
      expect(runtime.backend.name).toBe("@effect-torch/backend-cpu")
      expect(runtime.placement.deviceType).toBe("cpu")
    }))

  if (metalAvailable) {
    it.effect("rejects foreign lazy and concrete tensor handles", () =>
      Effect.gen(function*() {
        const cpu = yield* Runtime.Runtime
        const apple = BackendApple.makeRuntime()
        const graph = yield* cpu.node({
          op: "ones",
          inputs: [],
          attributes: { shape: [1], dtype: "f32" }
        })
        const lazyError = yield* Effect.flip(apple.node({ op: "relu", inputs: [graph] }))
        expect(lazyError.reason).toBe("foreign-handle")

        const executable = yield* cpu.compile({ roots: [graph] })
        const [value] = yield* cpu.execute(executable, { bindings: [], scalars: [], runtimeValues: {} })
        const concreteError = yield* Effect.flip(apple.readback(value))
        expect(concreteError.reason).toBe("foreign-handle")
        yield* cpu.release(value)
      }))

    it.effect("rejects foreign tensors during graph construction", () =>
      Effect.gen(function*() {
        const tensor = yield* Effect.provide(Tensor.ones([1]), BackendApple.layer)
        const error = yield* Effect.flip(Tensor.relu(tensor))
        expect(error.backend?.reason).toBe("foreign-handle")
      }))
  }

  it.effect("rejects released tensor and sequence handles", () =>
    Effect.gen(function*() {
      const cpu = yield* Runtime.Runtime
      const graph = yield* cpu.node({
        op: "ones",
        inputs: [],
        attributes: { shape: [1], dtype: "f32" }
      })
      const executable = yield* cpu.compile({ roots: [graph] })
      const [value] = yield* cpu.execute(executable, { bindings: [], scalars: [], runtimeValues: {} })
      yield* cpu.release(value)
      const releasedTensorError = yield* Effect.flip(cpu.readback(value))
      expect(releasedTensorError.reason).toBe("invalid-handle")

      const decode = cpu.extensions.decode
      const pool = yield* decode.makePool({
        layers: 1,
        kvHeads: 1,
        headDim: 2,
        maxTokens: 16,
        blockSize: 4,
        dtype: "f32",
        kdaLayers: 0,
        kdaHeads: 0,
        kdaHeadDim: 0,
        kdaValueDim: 0,
        convLayers: 0,
        convChannels: 0,
        convKernel: 0
      })
      const sequence = yield* decode.makeSequence(pool)
      yield* decode.releaseSequence(sequence)
      const releasedSequenceError = yield* Effect.flip(decode.sequenceCursor(sequence))
      expect(releasedSequenceError.reason).toBe("invalid-handle")
    }))

  it.effect("clearAll ignores invalid handles and continues", () =>
    Effect.gen(function*() {
      const invalid = (yield* Tensor.ones([1])) as unknown as Tensor.Concrete
      const second = (yield* Tensor.compute([yield* Tensor.zeros([1])]))[0]
      yield* Tensor.clearAll([invalid, second])
      const secondError = yield* Effect.flip(Tensor.toNumberArray(second))
      expect(secondError.message).toContain("cleared")
    }))

  it.effect("memoizes the runtime service", () =>
    Effect.gen(function*() {
      const runtime = yield* Runtime.Runtime
      expect(BackendCpu.makeRuntime()).toBe(runtime)
      expect(BackendCpu.makeRuntime()).toBe(runtime)
    }))
})

test.effect("the memoized runtime reuses compiled caches", () =>
  Effect.gen(function*() {
    // Tensor.compile keys native artifacts by runtime identity. makeRuntime must
    // return the same service object for separately provided layers to share it.
    const compiled = yield* Tensor.compile((inputs) => Effect.map(Tensor.relu(inputs[0]), (output) => [output]))
    const firstRuntime = BackendCpu.makeRuntime()
    const secondRuntime = BackendCpu.makeRuntime()
    expect(secondRuntime).toBe(firstRuntime)
    const FirstCpu = Layer.succeed(Runtime.Runtime, firstRuntime)
    const SecondCpu = Layer.succeed(Runtime.Runtime, secondRuntime)
    const input = yield* Effect.provide(
      Tensor.fromTypedArray(new Float32Array([-1, 2])),
      FirstCpu
    )

    const [first] = yield* Effect.provide(compiled.call([input]), FirstCpu)
    const [second] = yield* Effect.provide(compiled.call([input]), SecondCpu)

    expect(yield* Effect.provide(Tensor.toNumberArray(first), SecondCpu)).toEqual([0, 2])
    expect(yield* Effect.provide(Tensor.toNumberArray(second), FirstCpu)).toEqual([0, 2])
    expect(yield* compiled.stats).toEqual({ cached: 1, compiled: 1 })
  }))

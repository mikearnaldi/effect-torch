import { expect } from "@effect/vitest"
import { Deferred, Effect, Fiber } from "effect"
import { Model, Runtime, Tensor } from "../src/index.ts"
import { onDevices } from "./utils/devices.ts"

onDevices("Inference parameter preparation", () => (it) => {
  it.effect("captures tied parameters once and clears temporaries without invalidating constants", () =>
    Effect.gen(function*() {
      const runtime = yield* Runtime.Runtime
      const [source] = yield* Tensor.compute([yield* Tensor.full([2], 3)])
      const captured: Array<Tensor.Concrete> = []

      const program = yield* Model.withParameters([source, source], ([left, right]) =>
        Effect.gen(function*() {
          expect(left).toBe(right)
          expect(left).not.toBe(source)
          captured.push(left)

          return yield* Tensor.freezeProgram([yield* Tensor.add(left, right)], { constantWeights: true })
        }))

      expect((yield* Effect.flip(runtime.readback(captured[0]))).reason).toBe("invalid-handle")
      expect(yield* Tensor.toNumberArray(source)).toEqual([3, 3])
      yield* Tensor.clear(source)
      const [output] = yield* Tensor.runProgram(program, [])
      expect(yield* Tensor.toNumberArray(output)).toEqual([6, 6])
      yield* Tensor.clear(output)
    }))

  it.effect("releases temporary parameters when the callback is interrupted", () =>
    Effect.gen(function*() {
      const runtime = yield* Runtime.Runtime
      const ready = yield* Deferred.make<void>()
      const [source] = yield* Tensor.compute([yield* Tensor.ones([2])])
      const captured: Array<Tensor.Concrete> = []

      const fiber = yield* Model.withParameters([source], ([parameter]) =>
        Effect.gen(function*() {
          captured.push(parameter)
          yield* Deferred.succeed(ready, undefined)

          return yield* Effect.never
        })).pipe(Effect.forkChild({ startImmediately: true }))

      yield* Deferred.await(ready)
      yield* Fiber.interrupt(fiber)
      expect(captured).toHaveLength(1)
      expect((yield* Effect.flip(runtime.readback(captured[0]))).reason).toBe("invalid-handle")
      expect(yield* Tensor.toNumberArray(source)).toEqual([1, 1])
      yield* Tensor.clear(source)
    }))
})

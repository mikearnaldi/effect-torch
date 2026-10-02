import { expect } from "@effect/vitest"
import { Deferred, Effect, Exit, Fiber } from "effect"
import { Model, Runtime, Tensor } from "../src/index.ts"
import { onDevices } from "./utils/devices.ts"

onDevices("layer execution", () => (it) => {
  it.effect("retained outputs outlive intermediate states and independent calls", () =>
    Effect.scoped(Effect.gen(function*() {
      const [input] = yield* Tensor.compute([
        yield* Tensor.fromTypedArray(new Float32Array([-2, 3]))
      ]).pipe(Effect.flatMap(Tensor.clearAllScoped))
      const intermediates: Array<Tensor.Concrete> = []
      const build = (layer: number, hidden: Tensor.Concrete) =>
        Effect.gen(function*() {
          intermediates.push(hidden)
          const next = layer === 0 ? yield* Tensor.relu(hidden) : yield* Tensor.neg(hidden)
          const retained = yield* Tensor.sum(hidden)

          return [next, retained] as const
        })
      const [first, second] = yield* Effect.all([
        Model.executeLayers(input, 2, build),
        Model.executeLayers(input, 2, build)
      ], { concurrency: 2 })

      expect(yield* Tensor.toNumberArray(first.output)).toEqual([-0, -3])
      expect(yield* Tensor.toNumberArray(second.output)).toEqual([-0, -3])
      expect(yield* Tensor.toNumberArray(first.retained[0][0])).toEqual([1])
      expect(yield* Tensor.toNumberArray(first.retained[1][0])).toEqual([3])

      for (const hidden of intermediates) {
        expect((yield* Effect.flip(Tensor.toNumberArray(hidden)))._tag).toBe("TensorError")
      }

      yield* Tensor.clearAll([first.output, ...first.retained.flat()])
      expect(yield* Tensor.toNumberArray(second.output)).toEqual([-0, -3])
      yield* Tensor.clearAll([second.output, ...second.retained.flat()])
      expect(yield* Tensor.toNumberArray(input)).toEqual([-2, 3])

      const empty = yield* Model.executeLayers(input, 0, build)
      expect(empty.retained).toEqual([])
      yield* Tensor.clear(empty.output)
      expect(yield* Tensor.toNumberArray(input)).toEqual([-2, 3])
      expect((yield* Effect.flip(Model.executeLayers(input, -1, build)))._tag).toBe("ModelError")
    })))

  for (const failure of ["build", "observe", "interrupt"] as const) {
    it.effect(
      failure + " releases all partial results and leaves the borrowed input live",
      () =>
        Effect.scoped(Effect.gen(function*() {
          const [input] = yield* Tensor.compute([yield* Tensor.ones([2])]).pipe(Effect.flatMap(Tensor.clearAllScoped))
          const runtime = yield* Runtime.Runtime
          const produced: Array<Tensor.Concrete> = []
          const service: Runtime.RuntimeService = {
            ...runtime,
            execute: (program, invocation) =>
              runtime.execute(program, invocation).pipe(Effect.map((outputs) => {
                produced.push(...outputs)
                return outputs
              }))
          }
          const reached = yield* Deferred.make<void>()
          const error = new Model.ModelError({ op: "test", message: "injected layer failure" })
          const execution = Model.executeLayers(
            input,
            3,
            (layer, hidden) =>
              Effect.gen(function*() {
                if (failure === "build" && layer === 1) {
                  return yield* error
                }

                const next = yield* Tensor.neg(hidden)
                const retained = yield* Tensor.sum(hidden)

                return [next, retained] as const
              }),
            {
              observeLayer: (layer) =>
                Effect.gen(function*() {
                  if (layer !== 1) {
                    return
                  }

                  if (failure === "observe") {
                    return yield* error
                  }

                  if (failure === "interrupt") {
                    yield* Deferred.succeed(reached, undefined)
                    yield* Effect.never
                  }
                })
            }
          ).pipe(Effect.provideService(Runtime.Runtime, service))

          if (failure === "interrupt") {
            const fiber = yield* Effect.forkChild(execution)
            yield* Deferred.await(reached)
            yield* Fiber.interrupt(fiber)
            expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
          } else {
            expect(yield* Effect.flip(execution)).toBe(error)
          }

          expect(produced.length).toBeGreaterThan(1)

          for (const tensor of produced) {
            expect((yield* Effect.flip(runtime.readback(tensor))).reason).toBe("invalid-handle")
          }

          expect(yield* Tensor.toNumberArray(input)).toEqual([1, 1])
        }))
    )
  }
})

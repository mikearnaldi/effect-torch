import { describe, expect } from "@effect/vitest"
import { Effect } from "effect"
import { Gradient, LearningRate, Loss, Model, Optimizer, Tensor, Trainer } from "../src/index.ts"
import type { Runtime } from "../src/index.ts"
import { floats, onDevices } from "./utils/devices.ts"

const values = (t: Tensor.Any) => Tensor.toNumberArray(t)

onDevices("Compile", (device) => (it) => {
  describe("Tensor.compile", () => {
    it.effect("matches the uncompiled graph bitwise", () =>
      Effect.gen(function*() {
        const fn = yield* Tensor.compile(([a, b]) =>
          Effect.gen(function*() {
            const sum = yield* Tensor.add(a, b)
            const scaled = yield* Tensor.mul(sum, yield* Tensor.constantLike(sum, 2))
            return [yield* Tensor.tanh(scaled)]
          })
        )
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4]), [2, 2])
        const y = yield* Tensor.fromTypedArray(floats([5, 6, 7, 8]), [2, 2])
        const [expected] = yield* Tensor.compute(
          [yield* Tensor.tanh(yield* Tensor.mul(yield* Tensor.add(x, y), yield* Tensor.constantLike(x, 2)))]
        )
        const [actual] = yield* fn.call([x, y])
        expect(yield* values(actual)).toEqual(yield* values(expected))
        expect(yield* fn.stats).toEqual({ cached: 1, compiled: 1 })
      }))

    it.effect("reuses the program on a same-signature call and recompiles on a shape change", () =>
      Effect.gen(function*() {
        const fn = yield* Tensor.compile(([a]) => Effect.map(Tensor.relu(a), (out) => [out]))
        const x = yield* Tensor.fromTypedArray(floats([-1, 2, -3, 4]), [2, 2])
        const z = yield* Tensor.fromTypedArray(floats([-5, 6, -7, 8]), [2, 2])
        const w = yield* Tensor.fromTypedArray(floats([-1, 2, -3, 4, -5, 6]), [3, 2])
        const [first] = yield* fn.call([x])
        const [second] = yield* fn.call([z])
        const [third] = yield* fn.call([w])
        expect(yield* values(first)).toEqual([0, 2, 0, 4])
        expect(yield* values(second)).toEqual([0, 6, 0, 8])
        expect(yield* values(third)).toEqual([0, 2, 0, 4, 0, 6])
        expect(yield* fn.stats).toEqual({ cached: 2, compiled: 2 })
      }))

    it.effect("threads runtime scalars through the graph", () =>
      Effect.gen(function*() {
        // Runtime scalars are an internal mechanism (the Trainer's
        // learning rate): a scalar slot declared with makeScalarInput,
        // numbers bound at runProgram time.
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3]), [3])
        const a = yield* Tensor.makeInput(0, x)
        const scale = yield* Tensor.makeScalarInput(1, "f32")
        const program = yield* Tensor.freezeProgram([yield* Tensor.mul(a, scale)])
        expect(yield* values((yield* Tensor.runProgram(program, [x], [2]))[0])).toEqual([2, 4, 6])
        expect(yield* values((yield* Tensor.runProgram(program, [x], [-1]))[0])).toEqual([-1, -2, -3])
      }))

    it.effect("retains ordinary outputs across later invocations", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([1, 2]), [2])
        const input = yield* Tensor.makeInput(0, x)
        const program = yield* Tensor.freezeProgram([yield* Tensor.neg(input)])
        expect("outputCapacity" in program.handle.diagnostics).toBe(false)
        expect(program.handle.diagnostics.memory.outputBytes).toBeGreaterThan(0)
        // Each invocation transfers independent output ownership to the caller;
        // later executions may reuse workspace but cannot overwrite these handles.
        const retained: Array<Tensor.Concrete> = []
        for (let i = 0; i < 8; i++) {
          const value = yield* Tensor.fromTypedArray(floats([i, i + 1]), [2])
          retained.push((yield* Tensor.runProgram(program, [value]))[0])
        }
        for (let i = 0; i < retained.length; i++) {
          expect(yield* values(retained[i])).toEqual([-i, -(i + 1)])
        }
        yield* Effect.forEach(retained, (tensor) => Tensor.clear(tensor), { discard: true })
      }))

    it.effect("exposes frozen compile phase diagnostics", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([1, 2]), [2])
        const input = yield* Tensor.makeInput(0, x)
        const program = yield* Tensor.freezeProgram([yield* Tensor.neg(input)])
        const phases = program.handle.diagnostics.compilePhases
        expect(phases).toBeDefined()
        if (phases === undefined) throw new Error("native compile phases are missing")
        const typedPhases: ReadonlyArray<Runtime.ExecutableCompilePhaseDiagnostics> = phases

        expect(typedPhases.map(({ phase }) => phase)).toEqual([
          "graph_index",
          "optimization",
          "lowering",
          "lowered_program_validation",
          "memory_planning",
          "physical_planning",
          ...(device === "metal" ? ["pipeline_preparation"] : []),
          "artifact_assembly",
          ...(device === "metal" ? ["compile_submission"] : []),
          "publication"
        ])
        expect(Object.isFrozen(typedPhases)).toBe(true)
        for (const phase of typedPhases) {
          expect(Object.isFrozen(phase)).toBe(true)
          expect(Number.isFinite(phase.nanoseconds)).toBe(true)
          expect(phase.nanoseconds).toBeGreaterThanOrEqual(0)
        }
      }))

    it.effect("traces once under concurrent first calls (single-flight)", () =>
      Effect.gen(function*() {
        const fn = yield* Tensor.compile(([a]) => Effect.map(Tensor.neg(a), (out) => [out]))
        const inputs = yield* Effect.forEach(
          Array.from({ length: 8 }, (_, index) => index),
          (index) => Tensor.fromTypedArray(floats([index, index + 1]), [2])
        )
        const results = yield* Effect.forEach(
          inputs,
          (input) => fn.call([input]),
          { concurrency: "unbounded" }
        )
        for (let index = 0; index < results.length; index++) {
          expect(yield* values(results[index][0])).toEqual([-index, -(index + 1)])
        }
        expect(yield* fn.stats).toEqual({ cached: 1, compiled: 1 })
      }))

    it.effect("evicts least-recently-used programs past capacity", () =>
      Effect.gen(function*() {
        const fn = yield* Tensor.compile(([a]) => Effect.map(Tensor.relu(a), (out) => [out]), {
          cacheCapacity: 2
        })
        const of = (n: number) => Tensor.fromTypedArray(floats(Array.from({ length: n }, () => 1)), [n])
        yield* fn.call([yield* of(1)])
        yield* fn.call([yield* of(2)])
        yield* fn.call([yield* of(3)])
        expect(yield* fn.stats).toEqual({ cached: 2, compiled: 3 })
        yield* fn.call([yield* of(1)])
        expect(yield* fn.stats).toEqual({ cached: 2, compiled: 4 })
      }))

    it.effect("fails loudly when the builder materializes a placeholder", () =>
      Effect.gen(function*() {
        const fn = yield* Tensor.compile(([a]) =>
          Effect.gen(function*() {
            yield* Tensor.toNumberArray(a)
            return [a]
          })
        )
        const x = yield* Tensor.fromTypedArray(floats([1]), [1])
        const error = yield* Effect.flip(fn.call([x]))
        expect(error._tag).toBe("TensorError")
        expect((yield* fn.stats).cached).toBe(0)
      }))

    it.effect("recompiles on a dtype change and keeps both programs", () =>
      Effect.gen(function*() {
        const fn = yield* Tensor.compile(([a]) => Effect.map(Tensor.relu(a), (out) => [out]))
        const x = yield* Tensor.fromTypedArray(floats([1, -2]), [2])
        const y = yield* Tensor.fromTypedArray(new BigInt64Array([3n, -4n]), [2])
        expect(yield* values((yield* fn.call([x]))[0])).toEqual([1, 0])
        const ints = yield* Tensor.toTypedArray((yield* fn.call([y]))[0])
        if (!(ints instanceof BigInt64Array)) throw new Error("i64 readback must use BigInt64Array")
        expect(Array.from(ints)).toEqual([3n, 0n])
        expect(yield* fn.stats).toEqual({ cached: 2, compiled: 2 })
      }))

    it.effect("draws fresh randomness per call", () =>
      Effect.gen(function*() {
        const fn = yield* Tensor.compile(([a]) =>
          Effect.gen(function*() {
            const noise = yield* Tensor.randn(a.shape)
            return [yield* Tensor.add(a, noise)]
          })
        )
        const x = yield* Tensor.zeros([1024])
        const [first] = yield* fn.call([x])
        const [second] = yield* fn.call([x])
        expect(yield* values(first)).not.toEqual(yield* values(second))
      }))

    it.effect("the static executable plan matches direct computation across inputs", () =>
      Effect.gen(function*() {
        // Matmuls break fusion, so the graph has real intermediates for
        // the memory planner to reuse across repeated executions.
        const fn = yield* Tensor.compile(([a, b]) =>
          Effect.gen(function*() {
            const m1 = yield* Tensor.matmul(a, b)
            const scaled = yield* Tensor.mul(m1, yield* Tensor.constant(0.5))
            const m2 = yield* Tensor.matmul(scaled, b)
            return [yield* Tensor.tanh(yield* Tensor.add(m2, a))]
          })
        )
        for (let i = 0; i < 3; i++) {
          const x = yield* Tensor.fromTypedArray(floats([1 + i, 2, 3, 4 - i]), [2, 2])
          const y = yield* Tensor.fromTypedArray(floats([5, 6 - i, 7, 8]), [2, 2])
          const [expected] = yield* Tensor.compute([
            yield* Tensor.tanh(
              yield* Tensor.add(
                yield* Tensor.matmul(yield* Tensor.mul(yield* Tensor.matmul(x, y), yield* Tensor.constant(0.5)), y),
                x
              )
            )
          ])
          const [actual] = yield* fn.call([x, y])
          expect(yield* values(actual)).toEqual(yield* values(expected))
          yield* Tensor.clear(actual)
          yield* Tensor.clear(expected)
        }
      }))

    it.effect("clear releases cached programs", () =>
      Effect.gen(function*() {
        const fn = yield* Tensor.compile(([a]) => Effect.map(Tensor.relu(a), (out) => [out]))
        const x = yield* Tensor.fromTypedArray(floats([1]), [1])
        yield* fn.call([x])
        expect((yield* fn.stats).cached).toBe(1)
        yield* fn.clear
        expect(yield* fn.stats).toEqual({ cached: 0, compiled: 1 })
      }))
  })

  describe("Model execution (always compiled)", () => {
    const mlp = Effect.gen(function*() {
      return yield* Model.chain(
        yield* Model.linear("fc1", 2, 8),
        yield* Model.tanh,
        yield* Model.linear("fc2", 8, 1),
        yield* Model.sigmoid
      )
    })

    it.effect("execute matches forward bitwise", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const x = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
        const [expected] = yield* Tensor.compute([yield* model.forward(params, x)])
        const actual = yield* model.execute(params, x)
        expect(yield* values(actual)).toEqual(yield* values(expected))
        expect(yield* model.stats).toEqual({ cached: 1, compiled: 1 })
      }))

    it.effect("recompiles on a batch-shape change and serves concurrent calls", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const x2 = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
        const x4 = yield* Tensor.fromTypedArray(floats([0, 0, 0, 1, 1, 0, 1, 1]), [4, 2])
        const [a, b] = yield* Effect.all(
          [model.execute(params, x2), model.execute(params, x2)],
          { concurrency: "unbounded" }
        )
        expect(yield* values(a)).toEqual(yield* values(b))
        yield* model.execute(params, x4)
        expect(yield* model.stats).toEqual({ cached: 2, compiled: 2 })
      }))

    it.effect("forward stays a graph builder: it differentiates and composes", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const x = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
        const y = yield* Tensor.fromTypedArray(floats([1, 0]), [2, 1])
        const loss = yield* Loss.mse(yield* model.forward(params, x), y)
        const grads = yield* Gradient.grad(loss, params)
        const [value, ...evaluated] = yield* Tensor.compute([loss, ...grads])
        expect(Number.isFinite((yield* values(value))[0])).toBe(true)
        for (const g of evaluated) {
          expect((yield* values(g)).some((v) => v !== 0)).toBe(true)
        }
        const chained = yield* Model.chain(model, yield* Model.relu)
        const [out] = yield* Tensor.compute([yield* chained.forward(params, x)])
        expect(Number.isFinite((yield* values(out))[0])).toBe(true)
      }))

    it.effect("clear releases the forward programs", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const x = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
        yield* model.execute(params, x)
        expect((yield* model.stats).cached).toBe(1)
        yield* model.clear
        expect(yield* model.stats).toEqual({ cached: 0, compiled: 1 })
      }))

    it.effect("trains under a compiled trainer", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const input = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
        const target = yield* Tensor.fromTypedArray(floats([1, 0]), [2, 1])
        const initial = yield* Tensor.compute(yield* Model.initialize(model))
        const config: Trainer.TrainConfig<Optimizer.SgdState, Tensor.TensorError> = {
          optimizer: yield* Optimizer.sgd(),
          lr: LearningRate.constant(0.1),
          loss: Loss.mse,
          data: { input, target },
          stop: ({ step }) => step >= 10
        }
        const reference = yield* (yield* Trainer.make(model, config)).train(initial)
        const traced = yield* (yield* Trainer.make(model, config)).train(initial)
        expect(traced.loss).toBe(reference.loss)
      }))
  })
})

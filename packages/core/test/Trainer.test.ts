import { describe, expect } from "@effect/vitest"
import { Duration, Effect } from "effect"
import { TestClock } from "effect/testing"
import { LearningRate, Loss, Model, Optimizer, Tensor, Trainer } from "../src/index.ts"
import { floats, onDevices } from "./utils/devices.ts"

const values = (t: Tensor.Any) => Tensor.toNumberArray(t)

const mlp = Effect.gen(function*() {
  return yield* Model.chain(
    yield* Model.linear("fc1", 2, 8),
    yield* Model.tanh,
    yield* Model.linear("fc2", 8, 1),
    yield* Model.sigmoid
  )
})

const xor = Effect.gen(function*() {
  const input = yield* Tensor.fromTypedArray(floats([0, 0, 0, 1, 1, 0, 1, 1]), [4, 2])
  const target = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [4, 1])
  return { input, target } as const
})

onDevices("Trainer", (device) => (it) => {
  describe("stop policy", () => {
    it.effect("stops on a loss target", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const data = yield* xor
        let steps = 0
        const trainer = yield* Trainer.make(model, {
          optimizer: yield* Optimizer.adam(),
          lr: LearningRate.constant(0.1),
          loss: Loss.mse,
          data,
          stop: ({ step, loss }) => loss < 0.2 || step >= 2500,
          onStep: () => Effect.sync(() => steps++)
        })
        const { loss } = yield* trainer.train(yield* Model.initialize(trainer.model))
        expect(loss).toBeLessThan(0.2)
        expect(steps).toBeLessThan(2500)
      }))

    it.effect("stops on a step count, loss target, or external state", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const input = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
        const target = yield* Tensor.fromTypedArray(floats([1, 0]), [2, 1])
        let patience = 3
        const trainer = yield* Trainer.make(model, {
          optimizer: yield* Optimizer.sgd(),
          lr: LearningRate.constant(0.1),
          loss: Loss.mse,
          data: { input, target },
          stop: () => --patience === 0
        })
        const { loss } = yield* trainer.train(yield* Model.initialize(trainer.model))
        expect(patience).toBe(0)
        expect(Number.isFinite(loss)).toBe(true)
      }))
  })

  describe("train", () => {
    it.effect("trains a chained MLP on xor to convergence", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const data = yield* xor
        const trainer = yield* Trainer.make(model, {
          optimizer: yield* Optimizer.adam(),
          lr: LearningRate.constant(0.1),
          loss: Loss.mse,
          data,
          stop: ({ step }) => step >= 2500
        })
        const { params, loss } = yield* trainer.train(yield* Model.initialize(trainer.model))
        expect(loss).toBeLessThan(0.05)
        const [pred] = yield* Tensor.compute([yield* model.forward(params, data.input)])
        expect((yield* values(pred)).map((v) => (v > 0.5 ? 1 : 0))).toEqual([0, 1, 1, 0])
      }))

    it.effect("reports every step to onStep in order", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const input = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
        const target = yield* Tensor.fromTypedArray(floats([1, 0]), [2, 1])
        const seen: Array<Trainer.TrainStep> = []
        const trainer = yield* Trainer.make(model, {
          optimizer: yield* Optimizer.sgd(),
          lr: LearningRate.constant(0.1),
          loss: Loss.mse,
          data: { input, target },
          stop: ({ step }) => step >= 10,
          onStep: (info) => Effect.sync(() => seen.push(info))
        })
        yield* trainer.train(yield* Model.initialize(trainer.model))
        expect(seen.map(({ step }) => step)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        expect(seen.every(({ loss }) => Number.isFinite(loss))).toBe(true)
        expect(seen.every(({ elapsed }) => Duration.toMillis(elapsed) >= 0)).toBe(true)
        const millis = seen.map(({ elapsed }) => Duration.toMillis(elapsed))
        expect([...millis].sort((a, b) => a - b)).toEqual(millis)
      }))

    it.effect("each train run starts its own clock", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const input = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
        const target = yield* Tensor.fromTypedArray(floats([1, 0]), [2, 1])
        const runs: Array<Array<number>> = []
        const trainer = yield* Trainer.make(model, {
          optimizer: yield* Optimizer.sgd(),
          lr: LearningRate.constant(0.1),
          loss: Loss.mse,
          data: { input, target },
          stop: ({ step }) => step >= 3,
          onStep: ({ elapsed }) =>
            Effect.sync(() => runs.at(-1)!.push(Duration.toMillis(elapsed))).pipe(
              Effect.andThen(TestClock.adjust("100 millis"))
            )
        })
        runs.push([])
        const first = yield* trainer.train(yield* Model.initialize(trainer.model))
        // Final params and state roots transfer to the caller; release the first
        // run before reusing the trainer so this lifecycle test leaks no buffers.
        const firstOwned = [
          ...first.params,
          ...trainer.config.optimizer.stateRoots(first.state).filter(Tensor.isTensor)
        ]
        yield* Effect.forEach(Array.from(new Set(firstOwned)), (tensor) => Tensor.clear(tensor), { discard: true })
        runs.push([])
        yield* trainer.train(yield* Model.initialize(trainer.model))
        expect(runs).toEqual([[0, 100, 200], [0, 100, 200]])
      }))

    it.effect("a data sampler is re-drawn with the step number every step", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const batches = [
          [[0, 1, 1, 0], [1, 0]],
          [[1, 1, 0, 0], [0, 1]],
          [[0, 0, 1, 1], [0, 1]]
        ] as const
        const drawn: Array<number> = []
        const trainer = yield* Trainer.make(model, {
          optimizer: yield* Optimizer.sgd(),
          lr: LearningRate.constant(0.1),
          loss: Loss.mse,
          data: (step) =>
            Effect.gen(function*() {
              drawn.push(step)
              const [xs, ys] = batches[(step - 1) % batches.length]
              return {
                input: yield* Tensor.fromTypedArray(floats([...xs]), [2, 2]),
                target: yield* Tensor.fromTypedArray(floats([...ys]), [2, 1])
              }
            }),
          stop: ({ step }) => step >= 7
        })
        yield* trainer.train(yield* Model.initialize(trainer.model))
        expect(drawn).toEqual([1, 2, 3, 4, 5, 6, 7])
      }))

    it.effect("trains from explicit initial parameters", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const data = yield* xor
        const initial = yield* Tensor.compute(yield* Model.initialize(model))
        const lossOf = (params: Model.Params) =>
          Effect.gen(function*() {
            const [value] = yield* Tensor.compute([
              yield* Loss.mse(yield* model.forward(params, data.input), data.target)
            ])
            return (yield* values(value))[0]
          })
        const before = yield* lossOf(initial)
        const trainer = yield* Trainer.make(model, {
          optimizer: yield* Optimizer.adam(),
          lr: LearningRate.constant(0.1),
          loss: Loss.mse,
          data,
          stop: ({ step }) => step >= 200
        })
        const { params, loss } = yield* trainer.train(initial)
        expect(loss).toBeLessThan(before)
        expect(yield* lossOf(params)).toBeLessThan(before)
      }))
  })

  // Compiled tests compare the cached trace loop with its uncompiled reference.
  // Cache cleanup drops JS programs, not lower-level native pipeline artifacts.
  describe("compiled", () => {
    it.effect("agrees with the uncompiled loop step-for-step on a deterministic graph", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const input = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
        const target = yield* Tensor.fromTypedArray(floats([1, 0]), [2, 1])
        const initial = yield* Tensor.compute(yield* Model.initialize(model))
        const config: Trainer.TrainConfig<Optimizer.SgdState, Tensor.TensorError> = {
          optimizer: yield* Optimizer.sgd({ momentum: 0.9 }),
          lr: LearningRate.constant(0.05),
          loss: Loss.mse,
          data: { input, target },
          stop: ({ step }) => step >= 20
        }
        const reference = yield* (yield* Trainer.makeUncompiled(model, config)).train(initial)
        const tracedTrainer = yield* Trainer.make(model, config)
        const traced = yield* tracedTrainer.train(initial)
        expect(traced.loss).toBe(reference.loss)
        for (let i = 0; i < reference.params.length; i++) {
          expect(yield* values(traced.params[i])).toEqual(yield* values(reference.params[i]))
        }
        expect(yield* tracedTrainer.stats).toEqual({ cached: 1, compiled: 1 })
      }))

    it.effect("mixedBf16: bf16 compute with f32 masters converges; typed error on cpu", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const raw = yield* xor
        // Mixed precision casts parameters but does not cast input data. An LM
        // gets bf16 activations from its bf16 embedding, so feed bf16 features.
        const data = {
          input: yield* Tensor.cast(raw.input, "bf16"),
          target: yield* Tensor.cast(raw.target, "bf16")
        }
        const makeMixed = Effect.gen(function*() {
          return yield* Trainer.make(model, {
            optimizer: yield* Optimizer.adam(),
            lr: LearningRate.constant(0.1),
            loss: Loss.mse,
            data,
            stop: ({ step }) => step >= 1200,
            precision: "mixedBf16"
          })
        })
        if (device !== "metal") {
          const error = yield* Effect.flip(
            Effect.flatMap(makeMixed, (trainer) => Effect.flatMap(Model.initialize(model), trainer.train))
          )
          expect(error._tag).toBe("ModelError")
          return
        }
        const trainer = yield* makeMixed
        const { params, loss } = yield* trainer.train(yield* Model.initialize(trainer.model))
        expect(loss).toBeLessThan(0.1)
        // Masters stay f32 because the optimizer's update arithmetic is f32.
        expect(params.every((p) => p.dtype === "f32")).toBe(true)
        const forwardParams = yield* Effect.all(params.map((param) => Tensor.cast(param, "bf16")))
        const [pred] = yield* Tensor.compute([yield* model.forward(forwardParams, data.input)])
        expect((yield* values(pred)).map((v) => (v > 0.5 ? 1 : 0))).toEqual([0, 1, 1, 0])
      }))

    it.effect("resume continues with identical params, state, and step numbering", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const input = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
        const target = yield* Tensor.fromTypedArray(floats([1, 0]), [2, 1])
        const initial = yield* Tensor.compute(yield* Model.initialize(model))
        const makeTrainer = (stopStep: number) =>
          Effect.gen(function*() {
            return yield* Trainer.make(model, {
              optimizer: yield* Optimizer.adam(),
              lr: LearningRate.constant(0.05),
              loss: Loss.mse,
              data: { input, target },
              stop: ({ step }) => step >= stopStep
            })
          })
        const uninterrupted = yield* (yield* makeTrainer(20)).train(initial)
        const first = yield* (yield* makeTrainer(8)).train(initial)
        expect(first.step).toBe(8)
        const resumed = yield* (yield* makeTrainer(20)).train(first.params, { state: first.state, step: first.step })
        expect(resumed.step).toBe(20)
        expect(uninterrupted.step).toBe(20)
        expect(resumed.loss).toBe(uninterrupted.loss)
        for (let i = 0; i < uninterrupted.params.length; i++) {
          expect(yield* values(resumed.params[i])).toEqual(yield* values(uninterrupted.params[i]))
        }
      }))

    it.effect("agrees with the uncompiled loop under a learning-rate schedule", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const data = yield* xor
        const initial = yield* Tensor.compute(yield* Model.initialize(model))
        const config: Trainer.TrainConfig<Optimizer.AdamState, Tensor.TensorError> = {
          optimizer: yield* Optimizer.adam(),
          lr: LearningRate.cosine(0.1, { totalSteps: 30 }),
          loss: Loss.mse,
          data,
          stop: ({ step }) => step >= 30
        }
        const reference = yield* (yield* Trainer.makeUncompiled(model, config)).train(initial)
        const traced = yield* (yield* Trainer.make(model, config)).train(initial)
        expect(traced.loss).toBe(reference.loss)
        for (let i = 0; i < reference.params.length; i++) {
          expect(yield* values(traced.params[i])).toEqual(yield* values(reference.params[i]))
        }
      }))

    it.effect("trains xor to convergence through the compiled loop", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const data = yield* xor
        const trainer = yield* Trainer.make(model, {
          optimizer: yield* Optimizer.adam(),
          lr: LearningRate.constant(0.1),
          loss: Loss.mse,
          data,
          stop: ({ step }) => step >= 2500
        })
        const { params, loss } = yield* trainer.train(yield* Model.initialize(trainer.model))
        expect(loss).toBeLessThan(0.05)
        const [pred] = yield* Tensor.compute([yield* model.forward(params, data.input)])
        expect((yield* values(pred)).map((v) => (v > 0.5 ? 1 : 0))).toEqual([0, 1, 1, 0])
        expect(yield* trainer.stats).toEqual({ cached: 1, compiled: 1 })
      }))

    it.effect("recompiles when the batch signature changes", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const small = {
          input: yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2]),
          target: yield* Tensor.fromTypedArray(floats([1, 0]), [2, 1])
        }
        const large = yield* xor
        const trainer = yield* Trainer.make(model, {
          optimizer: yield* Optimizer.sgd(),
          lr: LearningRate.constant(0.1),
          loss: Loss.mse,
          data: (step) => Effect.succeed(step % 2 === 1 ? small : large),
          stop: ({ step }) => step >= 6
        })
        const { loss } = yield* trainer.train(yield* Model.initialize(trainer.model))
        expect(Number.isFinite(loss)).toBe(true)
        expect(yield* trainer.stats).toEqual({ cached: 2, compiled: 2 })
      }))

    it.effect("reports every step to onStep, drawn fresh per step", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const data = yield* xor
        const seen: Array<Trainer.TrainStep> = []
        const trainer = yield* Trainer.make(model, {
          optimizer: yield* Optimizer.sgd(),
          lr: LearningRate.constant(0.1),
          loss: Loss.mse,
          data,
          stop: ({ step }) => step >= 10,
          onStep: (info) => Effect.sync(() => seen.push(info))
        })
        yield* trainer.train(yield* Model.initialize(trainer.model))
        expect(seen.map(({ step }) => step)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        expect(seen.every(({ loss }) => Number.isFinite(loss))).toBe(true)
      }))

    it.effect("clear releases the cached programs", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const data = yield* xor
        const trainer = yield* Trainer.make(model, {
          optimizer: yield* Optimizer.sgd(),
          lr: LearningRate.constant(0.1),
          loss: Loss.mse,
          data,
          stop: ({ step }) => step >= 3
        })
        yield* trainer.train(yield* Model.initialize(trainer.model))
        expect((yield* trainer.stats).cached).toBe(1)
        yield* trainer.clear
        expect(yield* trainer.stats).toEqual({ cached: 0, compiled: 1 })
      }))
  })
})

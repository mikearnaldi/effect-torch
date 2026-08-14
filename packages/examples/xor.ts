import * as BackendCpu from "@effect-torch/backend-cpu"
import { LearningRate, Loss, Model, Optimizer, Runtime, Tensor, Trainer } from "@effect-torch/core"
import { NodeRuntime } from "@effect/platform-node"
import { Data, Effect } from "effect"

const HIDDEN = 8
const STEPS = 3000
const LR = 0.1

// Tutorial architecture: compose a 2 -> 8 -> 1 MLP from primitive models,
// train full-batch with Trainer's cached step, then use cached model inference.
const program = Effect.gen(function*() {
  const runtime = yield* Runtime.Runtime
  const x = yield* Tensor.fromTypedArray(new Float32Array([0, 0, 0, 1, 1, 0, 1, 1]), [4, 2])
  const y = yield* Tensor.fromTypedArray(new Float32Array([0, 1, 1, 0]), [4, 1])

  yield* Effect.log(`1) creating model: 2 -> ${HIDDEN} (tanh) -> 1 (sigmoid) on ${runtime.placement.description}`)
  const model = yield* createModel
  const params = yield* init(model)

  // Trainer traces the forward, loss, backward, and Adam update as one step.
  yield* Effect.log(`2) training: adam lr=${LR}, ${STEPS} steps (compiled)`)
  const trainer = yield* createTrainer(model, x, y)
  const trained = yield* trainer.train(params)

  yield* Effect.log("3) evaluating")
  yield* evaluate(model, trained.params, x, y)
})

const createModel = Effect.gen(function*() {
  const model = yield* Model.chain(
    yield* Model.linear("fc1", 2, HIDDEN),
    yield* Model.tanh,
    yield* Model.linear("fc2", HIDDEN, 1),
    yield* Model.sigmoid
  )
  return model
})

const init = (model: Model.Model) =>
  Effect.gen(function*() {
    const params = yield* model.init
    for (const [i, name] of model.names.entries()) {
      yield* Effect.log(`  ${name} [${params[i].shape}] ${params[i].dtype} initialized`)
    }
    return params
  })

const createTrainer = (
  model: Model.Model,
  x: Tensor.Any,
  y: Tensor.Any
) =>
  Effect.gen(function*() {
    const trainer = yield* Trainer.make(model, {
      optimizer: yield* Optimizer.adam(),
      lr: LearningRate.constant(LR),
      loss: Loss.mse,
      data: { input: x, target: y },
      stop: ({ step }) => step >= STEPS,
      onStep: ({ step, loss }) =>
        Effect.gen(function*() {
          if (step % 250 === 0) {
            const mem = process.memoryUsage()
            yield* Effect.log(
              `step ${String(step).padStart(4)}  loss ${loss.toFixed(6)}  rss ${(mem.rss / 1e6).toFixed(0)}MB  ext ${
                (mem.external / 1e6).toFixed(1)
              }MB  heap ${(mem.heapUsed / 1e6).toFixed(0)}MB`
            )
          }
        })
    })
    return trainer
  })

class MispredictionError extends Data.TaggedError("MispredictionError")<{
  readonly input: readonly [number, number]
  readonly expected: number
  readonly actual: number
}> {
  override get message() {
    return `misprediction on [${this.input[0]}, ${this.input[1]}]: expected ${this.expected}, got ${
      this.actual.toFixed(4)
    }`
  }
}

// Model.execute traces once for [1, 2], then reuses that inference program for
// each truth-table row; the first wrong classification fails the Effect.
const evaluate = (
  model: Model.Model,
  params: Model.Params,
  x: Tensor.Any,
  y: Tensor.Any
) =>
  Effect.gen(function*() {
    const inputs = yield* Tensor.toNumberArray(x)
    const targets = yield* Tensor.toNumberArray(y)
    for (let i = 0; i < targets.length; i++) {
      const single = yield* Tensor.fromTypedArray(new Float32Array([inputs[i * 2], inputs[i * 2 + 1]]), [1, 2])
      const pred = yield* model.execute(params, single)
      const [value] = yield* Tensor.toNumberArray(pred)
      const rounded = value > 0.5 ? 1 : 0
      const ok = rounded === targets[i]
      yield* Effect.log(
        `  ${inputs[i * 2]} ^ ${inputs[i * 2 + 1]} = ${targets[i]}  pred ${value.toFixed(4)} ${ok ? "ok" : "MISS"}`
      )
      if (!ok) {
        return yield* new MispredictionError({
          input: [inputs[i * 2], inputs[i * 2 + 1]],
          expected: targets[i],
          actual: value
        })
      }
    }
    yield* Effect.log(`${targets.length}/${targets.length} correct`)
  })

NodeRuntime.runMain(program.pipe(Effect.provide(BackendCpu.layer)))

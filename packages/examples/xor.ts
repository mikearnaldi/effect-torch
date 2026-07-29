import { Data, Effect } from "effect"
import { NodeRuntime } from "@effect/platform-node"
import { Device, Loss, Model, Optimizer, Tensor } from "@effect-torch/core"

class MispredictionError extends Data.TaggedError("MispredictionError")<{
  readonly input: readonly [number, number]
  readonly expected: number
  readonly actual: number
}> {
  override get message() {
    return `misprediction on [${this.input[0]}, ${this.input[1]}]: expected ${this.expected}, got ${this.actual.toFixed(4)}`
  }
}

const HIDDEN = 8
const STEPS = 3000
const LR = 0.1

const program = Effect.gen(function* () {
  const device = yield* Device.CurrentDevice
  const x = yield* Tensor.fromTypedArray(new Float32Array([0, 0, 0, 1, 1, 0, 1, 1]), [4, 2])
  const y = yield* Tensor.fromTypedArray(new Float32Array([0, 1, 1, 0]), [4, 1])

  // A 2 -> HIDDEN (tanh) -> 1 (sigmoid) MLP, composed from primitive models.
  // `Model.Params<typeof model>` computes the parameter tuple at the type
  // level: readonly [fc1.weight, fc1.bias, fc2.weight, fc2.bias].
  yield* Effect.log(`1) creating model: 2 -> ${HIDDEN} (tanh) -> 1 (sigmoid) on ${device}`)
  const model = yield* Model.chain(
    yield* Model.linear("fc1", 2, HIDDEN),
    yield* Model.tanh,
    yield* Model.linear("fc2", HIDDEN, 1),
    yield* Model.sigmoid
  )
  const params = yield* model.init
  for (const [i, name] of model.names.entries()) {
    yield* Effect.log(`  ${name} [${params[i].shape}] ${params[i].dtype} initialized`)
  }

  // Full-batch Adam on the MSE loss, one graph walk per step
  yield* Effect.log(`2) training: adam lr=${LR}, ${STEPS} steps`)
  const trained = yield* Model.train(model, {
    optimizer: Optimizer.adam({ lr: LR }),
    loss: Loss.mse,
    data: { input: x, target: y },
    stop: ({ loss }) => loss < 1e-6,
    params,
    onStep: ({ step, loss }) =>
      Effect.gen(function* () {
        if (step % 250 === 0) {
          const mem = process.memoryUsage()
          yield* Effect.log(
            `step ${String(step).padStart(4)}  loss ${loss.toFixed(6)}  rss ${(mem.rss / 1e6).toFixed(0)}MB  ext ${(mem.external / 1e6).toFixed(1)}MB  heap ${(mem.heapUsed / 1e6).toFixed(0)}MB`
          )
        }
      })
  })

  yield* Effect.log("3) evaluating")
  yield* evaluate(model, trained.params, x, y)
})

// One forward pass per input, failing on the first misprediction
const evaluate = <P extends ReadonlyArray<Tensor.GenericTensor>>(
  model: Model.Model<P>,
  params: P,
  x: Tensor.GenericTensor,
  y: Tensor.GenericTensor
) =>
  Effect.gen(function* () {
    const inputs = yield* Tensor.toNumberArray(x)
    const targets = yield* Tensor.toNumberArray(y)
    for (let i = 0; i < targets.length; i++) {
      const single = yield* Tensor.fromTypedArray(new Float32Array([inputs[i * 2], inputs[i * 2 + 1]]), [1, 2])
      const pred = yield* model.forward(params, single)
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

NodeRuntime.runMain(program.pipe(Effect.provide(Device.Best)))

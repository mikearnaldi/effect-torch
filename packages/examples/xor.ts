import { Data, Effect } from "effect"
import { NodeRuntime } from "@effect/platform-node"
import { Device, Loss, Optimizer, Tensor } from "@effect-torch/core"

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

type Params = readonly [
  w1: Tensor.GenericTensor,
  b1: Tensor.GenericTensor,
  w2: Tensor.GenericTensor,
  b2: Tensor.GenericTensor
]

// A 2 -> HIDDEN (tanh) -> 1 (sigmoid) MLP. `x` is a batch of rows, so any
// batch size works — including a single input of shape [1, 2].
const forward = ([w1, b1, w2, b2]: Params, x: Tensor.GenericTensor) =>
  Effect.gen(function* () {
    const h = yield* Tensor.tanh(yield* Tensor.add(yield* Tensor.matmul(x, w1), b1))
    return yield* Tensor.sigmoid(yield* Tensor.add(yield* Tensor.matmul(h, w2), b2))
  })

// 1) model creation: randomly initialized parameters, materialized once so
// they become plain leaves of every later graph
const createModel: Effect.Effect<Params, Tensor.TensorError, Device.CurrentDevice> = Effect.gen(
  function* () {
    const [w1, w2] = yield* Tensor.evaluate([
      yield* Tensor.randn([2, HIDDEN]),
      yield* Tensor.randn([HIDDEN, 1])
    ])
    const [b1, b2] = yield* Tensor.evaluate([
      yield* Tensor.zeros([1, HIDDEN]),
      yield* Tensor.zeros([1, 1])
    ])
    for (const [name, t] of [["w1", w1], ["b1", b1], ["w2", w2], ["b2", b2]] as const) {
      yield* Effect.log(`  ${name} [${t.shape}] ${t.dtype} initialized`)
    }
    return [w1, b1, w2, b2]
  }
)

// 2) model training: full-batch Adam on the MSE loss, one graph walk per step
const train = (params: Params, x: Tensor.GenericTensor, y: Tensor.GenericTensor) =>
  Effect.gen(function* () {
    const optimizer = Optimizer.adam({ lr: LR })
    let current = params
    let state = yield* optimizer.init(current)
    for (let i = 1; i <= STEPS; i++) {
      const loss = yield* Loss.mse(yield* forward(current, x), y)
      const result = yield* Optimizer.step(optimizer, loss, current, state)
      const [value] = yield* Tensor.toNumberArray(result.loss)
      if (i % 250 === 0) {
        const mem = process.memoryUsage()
        yield* Effect.log(
          `step ${String(i).padStart(4)}  loss ${value.toFixed(6)}  rss ${(mem.rss / 1e6).toFixed(0)}MB  ext ${(mem.external / 1e6).toFixed(1)}MB  heap ${(mem.heapUsed / 1e6).toFixed(0)}MB`
        )
      }
      current = result.params
      state = result.state
    }
    return current
  })

// 3) model evaluation: one forward pass per input, failing on the first
// misprediction
const evaluate = (params: Params, x: Tensor.GenericTensor, y: Tensor.GenericTensor) =>
  Effect.gen(function* () {
    const inputs = yield* Tensor.toNumberArray(x)
    const targets = yield* Tensor.toNumberArray(y)
    for (let i = 0; i < targets.length; i++) {
      const single = yield* Tensor.fromTypedArray(new Float32Array([inputs[i * 2], inputs[i * 2 + 1]]), [1, 2])
      const [pred] = yield* Tensor.evaluate([yield* forward(params, single)])
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

const program = Effect.gen(function* () {
  const device = yield* Device.CurrentDevice
  const x = yield* Tensor.fromTypedArray(new Float32Array([0, 0, 0, 1, 1, 0, 1, 1]), [4, 2])
  const y = yield* Tensor.fromTypedArray(new Float32Array([0, 1, 1, 0]), [4, 1])

  yield* Effect.log(`1) creating model: 2 -> ${HIDDEN} (tanh) -> 1 (sigmoid) on ${device}`)
  const params = yield* createModel

  yield* Effect.log(`2) training: adam lr=${LR}, ${STEPS} steps`)
  const trained = yield* train(params, x, y)

  yield* Effect.log("3) evaluating")
  yield* evaluate(trained, x, y)
})

NodeRuntime.runMain(program.pipe(Effect.provide(Device.Best)))

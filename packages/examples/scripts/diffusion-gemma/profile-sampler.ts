import * as BackendCuda from "@effect-torch/backend-cuda"
import { Tensor } from "@effect-torch/core"
import { NodeRuntime } from "@effect/platform-node"
import { Effect } from "effect"

const shape = [1, 256, 262144] as const

const statistics = (logits: Tensor.Any, temperature: Tensor.Any) =>
  Effect.gen(function*() {
    const processed = yield* Tensor.div(logits, temperature)
    const uniform = yield* Tensor.uniform(shape, { dtype: "f32" })
    const gumbel = yield* Tensor.neg(yield* Tensor.log(yield* Tensor.neg(yield* Tensor.log(uniform))))
    const sampled = yield* Tensor.cast(yield* Tensor.argmax(yield* Tensor.add(processed, gumbel), 2), "u32")
    const argmax = yield* Tensor.cast(yield* Tensor.argmax(processed, 2), "u32")
    const normalized = yield* Tensor.sub(processed, yield* Tensor.logsumexp(processed, { dims: [2], keepdims: true }))
    const clamped = yield* Tensor.clamp(normalized, { min: -3.4028234663852886e38 })
    const entropy = yield* Tensor.neg(
      yield* Tensor.sum(yield* Tensor.mul(clamped, yield* Tensor.softmax(normalized)), { dims: [2] })
    )
    const order = yield* Tensor.topKIndices(yield* Tensor.neg(entropy), shape[1])
    const mean = yield* Tensor.mean(entropy)

    return [yield* Tensor.cast(processed, "bf16"), sampled, argmax, entropy, order, mean]
  })

const main = Effect.scoped(Effect.gen(function*() {
  if (!(yield* BackendCuda.isAvailable)) throw new Error("CUDA is required")

  const sampler = yield* Tensor.compile(([logits, temperature]) => statistics(logits!, temperature!))
  const inputs = yield* Effect.acquireRelease(
    Effect.gen(function*() {
      const logits = yield* Tensor.zeros(shape)
      const temperature = yield* Tensor.full([], 0.5)

      return yield* Tensor.compute([logits, temperature])
    }),
    Tensor.clearAll
  )
  const started = performance.now()
  const outputs = yield* Effect.acquireRelease(sampler.call(inputs), Tensor.clearAll)
  const elapsedMilliseconds = performance.now() - started
  const reduced = yield* Effect.forEach(outputs.slice(1), Tensor.toNumberArray)

  console.log(JSON.stringify({ elapsedMilliseconds, reduced: reduced.map((values) => values.length) }))
}).pipe(Effect.provide(BackendCuda.layer())))

NodeRuntime.runMain(main)

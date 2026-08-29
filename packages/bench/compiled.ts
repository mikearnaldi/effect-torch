// Times a Metal compiled program at one training-scale bf16 QKV shape.
// `Tensor.compile` traces and compiles lazily. The untimed call triggers tracing
// and native compilation, then warms the pipelines. The supplied a/b values stay
// lazy. Every timed `call` materializes fresh randn+bf16 inputs, runs the cached
// matmul program, waits for execution, and clears the output. The GFLOP/s
// numerator counts GEMM arithmetic. Its timing denominator includes input
// generation, dispatch, execution, and release, so this is not isolated GEMM
// throughput. ITERS passes through `Number` without validation.

import * as BackendApple from "@effect-torch/backend-apple-native"
import { Tensor } from "@effect-torch/core"
import { Console, Effect } from "effect"
import { performance } from "node:perf_hooks"

const ITERS = Number(process.env.ITERS ?? 10)

const program = Effect.gen(function*() {
  const mk = (shape: ReadonlyArray<number>) => Effect.flatMap(Tensor.randn(shape), (t) => Tensor.cast(t, "bf16"))
  const [a, b] = yield* Effect.zip(mk([131072, 768]), mk([768, 2304]))
  const f = yield* Tensor.compile(([a, b]) => Effect.map(Tensor.matmul(a, b), (r) => [r]))
  const once = Effect.flatMap(f.call([a, b]), (outs) => Effect.forEach(outs, Tensor.clear, { discard: true }))
  yield* once
  const start = performance.now()
  for (let i = 0; i < ITERS; i++) {
    yield* once
  }
  const ms = (performance.now() - start) / ITERS
  yield* Console.log(
    `compiled qkv [131k,768]x[768,2304] bf16: ${ms.toFixed(3)} ms  ${
      (2 * 131072 * 768 * 2304 / ms / 1e6).toFixed(0)
    } GFLOP/s`
  )
})

Effect.runPromise(Effect.provide(program, BackendApple.layer()))

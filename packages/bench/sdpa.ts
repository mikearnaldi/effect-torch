// End-to-end Metal bf16 causal-SDPA latency at one fixed transformer shape.
// The untimed call warms compilation and pipelines. Each measured `toTypedArray`
// evaluates a newly built eager graph, including fresh randn+bf16 q/k/v inputs,
// and reads the full result to the host. Timing therefore includes graph
// construction, compile/cache lookup, input generation, completed execution,
// and readback rather than device-only attention time. ITERS is an unvalidated
// environment override.

import * as BackendApple from "@effect-torch/backend-apple-native"
import { Tensor } from "@effect-torch/core"
import { Console, Effect } from "effect"
import { performance } from "node:perf_hooks"

const ITERS = Number(process.env.ITERS ?? 20)

const program = Effect.gen(function*() {
  const shape = [8, 12, 1024, 64] as const
  const q = yield* Tensor.cast(yield* Tensor.randn(shape), "bf16")
  const k = yield* Tensor.cast(yield* Tensor.randn(shape), "bf16")
  const v = yield* Tensor.cast(yield* Tensor.randn(shape), "bf16")
  const once = Effect.flatMap(
    Tensor.scaledDotProductAttention(q, k, v, { causal: true }),
    Tensor.toTypedArray
  )
  yield* once
  const start = performance.now()
  for (let i = 0; i < ITERS; i++) {
    yield* once
  }
  yield* Console.log(`sdpa fwd [8,12,1024,64] bf16 causal: ${((performance.now() - start) / ITERS).toFixed(3)} ms/op`)
})

Effect.runPromise(Effect.provide(program, BackendApple.layer))

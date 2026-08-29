// Measures end-to-end eager matmul-chain throughput on CPU and, when available,
// Metal. One untimed iteration warms compilation and backend pipelines. Each
// measured iteration builds a BATCH-deep dependent graph. `toTypedArray` runs it
// to completion and copies the final result to the host. Reported ms/op is the
// chain wall time divided by BATCH. It includes graph construction, compile/cache
// lookup, execution, and readback. It does not isolate GEMM kernel time. N and
// ITERS pass through `Number` without validation. Any nonempty METAL_ONLY value
// skips CPU, including "0". An empty or unset value enables CPU.

import * as BackendApple from "@effect-torch/backend-apple-native"
import * as BackendCpu from "@effect-torch/backend-cpu"
import { Runtime, Tensor } from "@effect-torch/core"
import { Console, Effect } from "effect"
import { performance } from "node:perf_hooks"

const N = Number(process.env.N ?? 512)
const ITERS = Number(process.env.ITERS ?? 50)
const BATCH = 10
const flops = 2 * N ** 3

const bench = <E, R>(
  label: string,
  effect: Effect.Effect<unknown, E, R>,
  opsPerIter = 1
): Effect.Effect<void, E, R> =>
  Effect.gen(function*() {
    yield* effect
    const start = yield* Effect.sync(() => performance.now())
    yield* Effect.forEach(Array.from({ length: ITERS }), () => effect, { discard: true })
    const elapsed = yield* Effect.sync(() => performance.now() - start)
    const ms = elapsed / ITERS / opsPerIter
    yield* Console.log(`${label.padEnd(36)} ${ms.toFixed(3)} ms/op  ${(flops / ms / 1e6).toFixed(1)} GFLOP/s`)
  })

const chain = (
  a: Tensor.Any,
  b: Tensor.Any,
  n: number
): Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    let r = yield* Tensor.matmul(a, b)
    for (let i = 1; i < n; i++) {
      r = yield* Tensor.matmul(r, b)
    }
    return r
  })

const suite: Effect.Effect<void, Tensor.TensorError, Runtime.Runtime> = Effect.gen(function*() {
  const runtime = yield* Runtime.Runtime
  const [a, b] = yield* Effect.flatMap(
    Effect.zip(Tensor.randn([N, N]), Tensor.randn([N, N])),
    ([ra, rb]) => Tensor.compute([ra, rb])
  )
  yield* bench(
    `effect-torch ${runtime.placement.deviceType}`,
    Effect.flatMap(chain(a, b, BATCH), Tensor.toTypedArray),
    BATCH
  )
})

const main = async (): Promise<void> => {
  process.stdout.write(`matmul f32 ${N}x${N} @ ${N}x${N}, ${ITERS} iterations, ${BATCH} chained per iter\n`)
  if (!process.env.METAL_ONLY) {
    await Effect.runPromise(Effect.provide(suite, BackendCpu.layer))
  }
  if (await Effect.runPromise(BackendApple.isAvailable)) {
    await Effect.runPromise(Effect.provide(suite, BackendApple.layer()))
  }
}

main().catch((error) => {
  process.stderr.write(`${String(error)}\n`)
  process.exitCode = 1
})

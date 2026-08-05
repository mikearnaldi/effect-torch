import { Device, Tensor } from "@effect-torch/core"
import { Console, Effect } from "effect"
import { performance } from "node:perf_hooks"

const N = Number(process.env.N ?? 512)
const ITERS = Number(process.env.ITERS ?? 50)
const BATCH = 10
const flops = 2 * N ** 3

const bench = <E>(
  label: string,
  effect: Effect.Effect<unknown, E>,
  opsPerIter = 1
): Effect.Effect<void, E> =>
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
): Effect.Effect<Tensor.Lazy, Tensor.TensorError> =>
  Effect.gen(function*() {
    let r = yield* Tensor.matmul(a, b)
    for (let i = 1; i < n; i++) {
      r = yield* Tensor.matmul(r, b)
    }
    return r
  })

const suite: Effect.Effect<void, Tensor.TensorError, Device.CurrentDevice> = Effect.gen(function*() {
  const device = yield* Device.CurrentDevice
  const [a, b] = yield* Effect.flatMap(
    Effect.zip(Tensor.randn([N, N]), Tensor.randn([N, N])),
    ([ra, rb]) => Tensor.compute([ra, rb])
  )
  yield* bench(`effect-torch ${device}`, Effect.flatMap(chain(a, b, BATCH), Tensor.toTypedArray), BATCH)
})

const program = Effect.gen(function*() {
  yield* Console.log(`matmul f32 ${N}x${N} @ ${N}x${N}, ${ITERS} iterations, ${BATCH} chained per iter`)
  yield* Effect.provide(suite, Device.Cpu)
  if (yield* Device.isAvailable("metal")) {
    yield* Effect.provide(suite, Device.Metal)
  }
})

Effect.runPromise(program)

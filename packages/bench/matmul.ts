import { performance } from "node:perf_hooks"
import { Console, Effect } from "effect"
import { Tensor } from "@effect-torch/core"

const N = Number(process.env.N ?? 512)
const ITERS = Number(process.env.ITERS ?? 50)
const BATCH = 10
const flops = 2 * N ** 3

const bench = <E>(
  label: string,
  effect: Effect.Effect<unknown, E>,
  opsPerIter = 1
): Effect.Effect<void, E> =>
  Effect.gen(function* () {
    yield* effect
    const start = yield* Effect.sync(() => performance.now())
    yield* Effect.forEach(Array.from({ length: ITERS }), () => effect, { discard: true })
    const elapsed = yield* Effect.sync(() => performance.now() - start)
    const ms = elapsed / ITERS / opsPerIter
    yield* Console.log(`${label.padEnd(36)} ${ms.toFixed(3)} ms/op  ${(flops / ms / 1e6).toFixed(1)} GFLOP/s`)
  })

const chain = (
  a: Tensor.GenericTensor,
  b: Tensor.GenericTensor,
  n: number
): Effect.Effect<Tensor.LazyTensor, Tensor.TensorError> =>
  Effect.gen(function* () {
    let r = yield* Tensor.matmul(a, b)
    for (let i = 1; i < n; i++) {
      r = yield* Tensor.matmul(r, b)
    }
    return r
  })

const deviceAvailable = (device: Tensor.DeviceKind): Effect.Effect<boolean> =>
  Effect.gen(function* () {
    const probe = yield* Effect.exit(
      Effect.flatMap(Tensor.zeros([4, 4], { device }), Tensor.toTypedArray)
    )
    return probe._tag === "Success"
  })

const program = Effect.gen(function* () {
  yield* Console.log(`matmul f32 ${N}x${N} @ ${N}x${N}, ${ITERS} iterations, ${BATCH} chained per iter`)

  const a = yield* Effect.flatMap(Tensor.randn([N, N]), Tensor.evaluate)
  const b = yield* Effect.flatMap(Tensor.randn([N, N]), Tensor.evaluate)
  yield* bench("effect-torch cpu", Effect.flatMap(chain(a, b, BATCH), Tensor.toTypedArray), BATCH)

  if (yield* deviceAvailable("metal")) {
    const am = yield* Effect.flatMap(Tensor.randn([N, N], { device: "metal" }), Tensor.evaluate)
    const bm = yield* Effect.flatMap(Tensor.randn([N, N], { device: "metal" }), Tensor.evaluate)
    yield* bench(
      "effect-torch metal",
      Effect.flatMap(chain(am, bm, BATCH), Tensor.toTypedArray),
      BATCH
    )
  }
})

Effect.runPromise(program)

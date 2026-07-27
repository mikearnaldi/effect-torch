import { performance } from "node:perf_hooks"
import { Console, Effect } from "effect"
import { Device, Tensor } from "@effect-torch/core"
import mlx from "@frost-beta/mlx"

const mx = mlx.core
const N = Number(process.env.N ?? 512)
const TRIALS = 5
const INNER = 20
const BATCH = 10

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

const timed = <E>(effect: Effect.Effect<unknown, E>): Effect.Effect<number, E> =>
  Effect.gen(function* () {
    const start = yield* Effect.sync(() => performance.now())
    yield* Effect.forEach(Array.from({ length: INNER }), () => effect, { discard: true })
    const elapsed = yield* Effect.sync(() => performance.now() - start)
    return elapsed / INNER / BATCH
  })

const median = (values: ReadonlyArray<number>): number => [...values].sort((x, y) => x - y)[Math.floor(values.length / 2)]

const program = Effect.gen(function* () {
  const [am, bm] = yield* Effect.flatMap(
    Effect.zip(Tensor.randn([N, N]), Tensor.randn([N, N])),
    ([ra, rb]) => Tensor.evaluate([ra, rb])
  )

  const xa = yield* Effect.sync(() => mx.random.normal([N, N]))
  const xb = yield* Effect.sync(() => mx.random.normal([N, N]))
  yield* Effect.sync(() => mx.eval(xa, xb))

  const ours = yield* Effect.forEach(
    Array.from({ length: TRIALS }),
    () => timed(Effect.flatMap(chain(am, bm, BATCH), Tensor.toTypedArray))
  )

  const theirs = yield* Effect.forEach(Array.from({ length: TRIALS }), () =>
    timed(
      Effect.sync(() => {
        let r = xa
        for (let i = 0; i < BATCH; i++) r = mx.matmul(r, xb)
        mx.eval(r)
      })
    )
  )

  yield* Console.log(`N=${N}, ${BATCH} chained matmuls, ms/op (median of ${TRIALS})`)
  yield* Console.log(`effect-torch: ${median(ours).toFixed(4)}  (all: ${ours.map((x) => x.toFixed(3)).join(" ")})`)
  yield* Console.log(`node-mlx:     ${median(theirs).toFixed(4)}  (all: ${theirs.map((x) => x.toFixed(3)).join(" ")})`)
})

Effect.runPromise(Effect.provide(program, Device.Metal))

import * as BackendApple from "@effect-torch/backend-apple-native"
import { type Runtime, Tensor } from "@effect-torch/core"
import { Console, Effect } from "effect"
import { createRequire } from "node:module"
import { performance } from "node:perf_hooks"

// Compares median end-to-end time for dependent f32 matmul chains on Metal.
// Both frameworks finish each BATCH-deep chain before starting the next
// iteration. effect-torch finishes with `toTypedArray`. MLX uses `mx.eval`. The
// result is the chain wall time divided by BATCH. The effect-torch timer includes
// eager graph construction, compile/cache lookup, and host readback. The MLX
// timer includes graph construction and eval. This compares the two API paths,
// not isolated kernels. Trials run in a fixed order without an explicit untimed
// chain warmup.
//
// @frost-beta/mlx ships TypeScript sources that this repository's strict type
// program cannot compile. The benchmark loads the package dynamically and
// declares the API members it uses. Install the optional dependency before
// running this benchmark.
interface MlxArray {
  readonly shape: Array<number>
}
interface Mlx {
  readonly random: { normal(shape: Array<number>): MlxArray }
  matmul(a: MlxArray, b: MlxArray): MlxArray
  eval(...tensors: Array<MlxArray>): void
}
// SAFETY: this benchmark uses the documented core export of the optional @frost-beta/mlx package.
const mx = (createRequire(import.meta.url)("@frost-beta/mlx") as { core: Mlx }).core
const N = Number(process.env.N ?? 512)
const TRIALS = 5
const INNER = 20
const BATCH = 10

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

const timed = <E, R>(effect: Effect.Effect<unknown, E, R>): Effect.Effect<number, E, R> =>
  Effect.gen(function*() {
    const start = yield* Effect.sync(() => performance.now())
    yield* Effect.forEach(Array.from({ length: INNER }), () => effect, { discard: true })
    const elapsed = yield* Effect.sync(() => performance.now() - start)
    return elapsed / INNER / BATCH
  })

const median = (values: ReadonlyArray<number>): number =>
  [...values].sort((x, y) => x - y)[Math.floor(values.length / 2)]

const program = Effect.gen(function*() {
  const [am, bm] = yield* Effect.flatMap(
    Effect.zip(Tensor.randn([N, N]), Tensor.randn([N, N])),
    ([ra, rb]) => Tensor.compute([ra, rb])
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
    ))

  yield* Console.log(`N=${N}, ${BATCH} chained matmuls, ms/op (median of ${TRIALS})`)
  yield* Console.log(`effect-torch: ${median(ours).toFixed(4)}  (all: ${ours.map((x) => x.toFixed(3)).join(" ")})`)
  yield* Console.log(`node-mlx:     ${median(theirs).toFixed(4)}  (all: ${theirs.map((x) => x.toFixed(3)).join(" ")})`)
})

Effect.runPromise(Effect.provide(program, BackendApple.layer()))

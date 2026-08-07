import * as BackendApple from "@effect-torch/backend-apple-native"
import { type Runtime, Tensor } from "@effect-torch/core"
import { Console, Effect } from "effect"
import { performance } from "node:perf_hooks"

const ITERS = Number(process.env.ITERS ?? 20)

const CHAIN = Number(process.env.CHAIN ?? 20)

const bench = <A extends Tensor.Any>(
  label: string,
  a: A,
  b: A,
  flops: number,
  perStep: number
): Effect.Effect<void, Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    // CHAIN identical matmuls as roots of one compute: one sync per
    // iteration, so eager per-call overhead does not drown the kernel.
    const lazies = yield* Effect.forEach(
      Array.from({ length: CHAIN }),
      () => Tensor.matmul(a, b)
    )
    const once = Effect.flatMap(Tensor.compute(lazies), (outs) => Effect.forEach(outs, Tensor.clear, { discard: true }))
    yield* once
    const start = performance.now()
    for (let i = 0; i < ITERS; i++) {
      yield* once
    }
    const ms = (performance.now() - start) / ITERS / CHAIN
    yield* Console.log(
      `${label.padEnd(40)} ${ms.toFixed(3)} ms  ${(flops / ms / 1e6).toFixed(0)} GFLOP/s  x${perStep}/step = ${
        (ms * perStep).toFixed(0)
      } ms`
    )
  })

// real 30M fineweb model: EMBED=256, HEADS=4, LAYERS=6, BLOCK=256
const BT = 128 * 256 // tokens per step at batch 128

const program = Effect.gen(function*() {
  const mk = (shape: ReadonlyArray<number>) => Effect.flatMap(Tensor.randn(shape), (t) => Tensor.cast(t, "bf16"))
  const gemm = (m: number, k: number, n: number, perStep: number, label: string) =>
    Effect.flatMap(Effect.zip(mk([m, k]), mk([k, n])), ([a, b]) => bench(label, a, b, 2 * m * k * n, perStep))

  // trunk: 6 layers, each fwd + dX + dW
  yield* gemm(BT, 256, 768, 6, "qkv   [32k,256]x[256,768]    fwd")
  yield* gemm(BT, 256, 256, 6, "proj  [32k,256]x[256,256]    fwd")
  yield* gemm(BT, 256, 1024, 6, "fc    [32k,256]x[256,1024]   fwd")
  yield* gemm(BT, 1024, 256, 6, "proj2 [32k,1024]x[1024,256]  fwd")
  yield* gemm(BT, 768, 256, 6, "qkv   [32k,768]x[768,256]    dX")
  yield* gemm(256, BT, 768, 6, "qkv   [256,32k]x[32k,768]    dW")
  yield* gemm(BT, 1024, 256, 12, "fc/p2 [32k,1024]x[1024,256]  dX (fc+proj2)")
  yield* gemm(256, BT, 1024, 12, "fc/p2 [256,32k]x[32k,1024]  dW (fc+proj2)")
  // vocab head: 48 CE chunks of [BT/48,256]x[256,50257]
  const rows = Math.ceil(BT / 48)
  yield* gemm(rows, 256, 50257, 48, `head  [${rows},256]x[256,50257] fwd x48`)
  yield* gemm(rows, 50257, 256, 48, `head  [${rows},50k]x[50k,256]  dX x48`)
  yield* gemm(256, rows, 50257, 48, `head  [256,${rows}]x[${rows},50k]  dW x48`)
})

Effect.runPromise(Effect.provide(program, BackendApple.layer))

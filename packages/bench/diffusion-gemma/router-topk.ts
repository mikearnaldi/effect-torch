/** Focused exact router sort hardware gate and component timing. */
import * as BackendCuda from "@effect-torch/backend-cuda"
import { Tensor } from "@effect-torch/core"
import { Cause, Effect, Exit } from "effect"
import assert from "node:assert/strict"
import { performance } from "node:perf_hooks"

const flag = "EFFECT_TORCH_CUDA_ROUTER_BITONIC"
const original = process.env[flag]
const data = (rows: number, width: number): Float32Array => {
  const values = new Float32Array(rows * width)
  const bits = new Uint32Array(values.buffer)
  let random = 0x719a6c31
  for (let index = 0; index < values.length; index++) {
    random ^= random << 13
    random ^= random >>> 17
    random ^= random << 5
    bits[index] = (random & 0x7f800000) === 0x7f800000 ? random ^ 0x00800000 : random
  }
  for (let row = 0; row < rows; row++) {
    if (row % 3 === 0) {
      for (let index = 0; index < width; index++) values[row * width + index] = index % 7 - 3
    }
    values.set([-0, 0, Infinity, -Infinity, Infinity, 2 ** -149, -(2 ** -149)], row * width)
  }
  if (rows > 1) values.fill(0, width, 2 * width)
  return values
}

try {
  await Effect.runPromise(Effect.provide(
    Effect.scoped(Effect.gen(function*() {
      for (const enabled of [false, true]) {
        process.env[flag] = enabled ? "1" : "0"
        for (const width of [127, 128, 129, 256]) {
          for (const k of width === 128 ? [1, 8, 127, 128] : width === 256 ? [256] : [8]) {
            const rows = 256
            const values = data(rows, width)
            const expected: Array<number> = []
            for (let row = 0; row < rows; row++) {
              const indices = Array.from({ length: width }, (_, index) => index)
              indices.sort((a, b) => {
                const left = values[row * width + a]!
                const right = values[row * width + b]!
                return left > right ? -1 : left < right ? 1 : a - b
              })
              expected.push(...indices.slice(0, k))
            }
            yield* Effect.scoped(Effect.gen(function*() {
              const input = yield* Effect.acquireRelease(
                Tensor.compute([yield* Tensor.fromTypedArray(values, [rows, width])]),
                Tensor.clearAll,
                { interruptible: true }
              )
              const program = yield* Tensor.compile(([scores]) =>
                Effect.gen(function*() {
                  return [yield* Tensor.topKIndices(scores!, k)]
                })
              )
              yield* Effect.addFinalizer(() => program.clear)
              const durations: Array<number> = []
              for (let run = -2; run < (width === 128 && k === 8 ? 25 : 1); run++) {
                yield* Effect.scoped(Effect.gen(function*() {
                  const start = performance.now()
                  const output = yield* Effect.acquireRelease(program.call(input), Tensor.clearAll, {
                    interruptible: true
                  })
                  const actual = yield* Tensor.toTypedArray(output[0]!)
                  const elapsed = performance.now() - start
                  assert(actual instanceof Uint32Array)
                  assert.deepEqual(Array.from(actual), expected)
                  if (run >= 0) durations.push(elapsed)
                }))
              }
              durations.sort((a, b) => a - b)
              process.stdout.write(
                JSON.stringify({
                  enabled,
                  rows,
                  width,
                  k,
                  exact: true,
                  componentMedianMilliseconds: durations[Math.floor(durations.length / 2)]
                }) + "\n"
              )
            }))
          }
        }
        for (const index of [0, 127, 255]) {
          const values = data(2, 128)
          values[index] = NaN
          const result = yield* Effect.exit(Effect.scoped(Effect.gen(function*() {
            const input = yield* Tensor.fromTypedArray(values, [2, 128])
            const output = yield* Effect.acquireRelease(
              Tensor.compute([yield* Tensor.topKIndices(input, 8)]),
              Tensor.clearAll,
              { interruptible: true }
            )
            yield* Tensor.toTypedArray(output[0]!)
          })))
          assert(Exit.isFailure(result))
          assert.match(Cause.pretty(result.cause), /NaN/)
        }
      }
    })),
    BackendCuda.layer()
  ))
} finally {
  if (original === undefined) delete process.env[flag]
  else process.env[flag] = original
}

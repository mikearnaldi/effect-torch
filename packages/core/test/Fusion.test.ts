import { describe, layer } from "@effect/vitest"
import * as assert from "@effect/vitest/utils"
import { Effect } from "effect"
import { Device, Gradient, Tensor } from "../src/index.ts"

const values = (t: Tensor.GenericTensor) =>
  Effect.map(Tensor.toTypedArray(t), (arr) => Array.from<number | bigint>(arr).map(Number))

const withFusion = <A, E, R>(enabled: boolean, effect: Effect.Effect<A, E, R>): Effect.Effect<A, E, R> =>
  Effect.acquireUseRelease(
    Effect.sync(() => {
      const previous = process.env.EFFECT_TORCH_NO_FUSION
      if (enabled) {
        delete process.env.EFFECT_TORCH_NO_FUSION
      } else {
        process.env.EFFECT_TORCH_NO_FUSION = "1"
      }
      return previous
    }),
    () => effect,
    (previous) =>
      Effect.sync(() => {
        if (previous === undefined) {
          delete process.env.EFFECT_TORCH_NO_FUSION
        } else {
          process.env.EFFECT_TORCH_NO_FUSION = previous
        }
      })
  )

layer(Device.Cpu)("Fusion", (it) => {
  describe("region fusion", () => {
    it.effect("fused and unfused evaluation agree on values and gradients", () =>
      Effect.gen(function* () {
        const build = Effect.gen(function* () {
          const a = yield* Tensor.fromTypedArray(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3])
          const b = yield* Tensor.fromTypedArray(new Float64Array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]), [2, 3])
          // a fused chain with a scalar constant fold and a two-region merge:
          // mul and add each open a region, the outer add merges them
          const left = yield* Tensor.mul(a, b)
          const right = yield* Tensor.add(a, 2)
          const merged = yield* Tensor.add(left, right)
          const y = yield* Tensor.exp(yield* Tensor.sqrt(merged))
          const z = yield* Tensor.relu(yield* Tensor.sub(y, 2))
          const loss = yield* Tensor.sum(yield* Tensor.mul(y, yield* Tensor.sin(z)))
          const [ga, gb] = yield* Gradient.grad(loss, [a, b])
          return {
            y: yield* values(y),
            z: yield* values(z),
            ga: yield* values(ga),
            gb: yield* values(gb)
          }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["y", "z", "ga", "gb"] as const) {
          assert.deepStrictEqual(fused[key].length, unfused[key].length)
          fused[key].forEach((v, i) => {
            assert.assertTrue(Math.abs(v - unfused[key][i]) < 1e-12, `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      })
    )

    it.effect("tanh, abs and erf chains fuse and agree with unfused evaluation", () =>
      Effect.gen(function* () {
        const build = Effect.gen(function* () {
          const x = yield* Tensor.fromTypedArray(new Float64Array([0.5, -1, 2, -3, 0.25, 1.5]), [2, 3])
          // sigmoid and silu are composed from tanh; gelu from erf — all
          // should ride the same fused regions
          const y = yield* Tensor.silu(yield* Tensor.tanh(x))
          const z = yield* Tensor.gelu(yield* Tensor.abs(y))
          const loss = yield* Tensor.sum(yield* Tensor.mul(z, y))
          const [gx] = yield* Gradient.grad(loss, [x])
          return { y: yield* values(y), z: yield* values(z), gx: yield* values(gx) }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["y", "z", "gx"] as const) {
          fused[key].forEach((v, i) => {
            assert.assertTrue(Math.abs(v - unfused[key][i]) < 1e-9, `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      })
    )

    it.effect("broadcasting boundaries stay unfused but still correct", () =>
      Effect.gen(function* () {
        const build = Effect.gen(function* () {
          const x = yield* Tensor.fromTypedArray(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3])
          // [2, 3] - [1, 3] is a broadcasting boundary: not fusable
          const row = yield* Tensor.fromTypedArray(new Float64Array([0.5, 0.5, 0.5]), [1, 3])
          const y = yield* Tensor.sqrt(yield* Tensor.mul(yield* Tensor.sub(x, row), 2))
          return yield* values(y)
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        fused.forEach((v, i) => assert.assertTrue(Math.abs(v - unfused[i]) < 1e-12))
      })
    )
  })
})

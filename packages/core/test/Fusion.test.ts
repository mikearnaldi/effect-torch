import { describe, layer } from "@effect/vitest"
import * as assert from "@effect/vitest/utils"
import { Effect } from "effect"
import { Device, Gradient, Loss, Tensor } from "../src/index.ts"

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

    it.effect("log chains (softplus, mish, logSoftmax) fuse and agree with unfused evaluation", () =>
      Effect.gen(function* () {
        const build = Effect.gen(function* () {
          const x = yield* Tensor.fromTypedArray(new Float64Array([0.5, -1, 2, -3, 0.25, 1.5]), [2, 3])
          const a = yield* Tensor.softplus(x)
          const b = yield* Tensor.mish(x)
          const c = yield* Tensor.logSoftmax(x)
          const y = yield* Tensor.add(yield* Tensor.add(a, b), c)
          const loss = yield* Tensor.sum(y)
          const [gx] = yield* Gradient.grad(loss, [x])
          return { y: yield* values(y), gx: yield* values(gx) }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["y", "gx"] as const) {
          fused[key].forEach((v, i) => {
            assert.assertTrue(Math.abs(v - unfused[key][i]) < 1e-12, `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      })
    )

    it.effect("binary cross entropy chains fuse and agree with unfused evaluation", () =>
      Effect.gen(function* () {
        const build = Effect.gen(function* () {
          const logits = yield* Tensor.fromTypedArray(new Float64Array([1.5, -2, 0.25, 3, -0.5, 0.75]), [2, 3])
          const target = yield* Tensor.fromTypedArray(new Float64Array([1, 0, 1, 1, 0, 1]), [2, 3])
          const fromLogits = yield* Loss.binaryCrossEntropy(logits, target, { fromLogits: true })
          const probs = yield* Tensor.sigmoid(logits)
          const fromProbs = yield* Loss.binaryCrossEntropy(probs, target)
          const loss = yield* Tensor.add(fromLogits, fromProbs)
          const [g] = yield* Gradient.grad(loss, [logits])
          return {
            fromLogits: yield* values(fromLogits),
            fromProbs: yield* values(fromProbs),
            g: yield* values(g)
          }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["fromLogits", "fromProbs", "g"] as const) {
          fused[key].forEach((v, i) => {
            assert.assertTrue(Math.abs(v - unfused[key][i]) < 1e-12, `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      })
    )

    it.effect("pow exponents fuse (square, cube, sqrt forms, generic) and agree with unfused evaluation", () =>
      Effect.gen(function* () {
        const build = Effect.gen(function* () {
          const x = yield* Tensor.fromTypedArray(new Float64Array([0.5, 1, 2, 3, 0.25, 1.5]), [2, 3])
          const a = yield* Tensor.rsqrt(x)
          const b = yield* Tensor.reciprocal(x)
          const c = yield* Tensor.gelu(x, { approximate: "tanh" })
          const d = yield* Tensor.pow(x, 1.7)
          const y = yield* Tensor.add(yield* Tensor.add(a, b), yield* Tensor.add(c, d))
          const loss = yield* Tensor.sum(y)
          const [gx] = yield* Gradient.grad(loss, [x])
          return { y: yield* values(y), gx: yield* values(gx) }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["y", "gx"] as const) {
          fused[key].forEach((v, i) => {
            assert.assertTrue(Math.abs(v - unfused[key][i]) < 1e-9, `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      })
    )

    it.effect("floor, ceil and round fuse and agree with unfused evaluation", () =>
      Effect.gen(function* () {
        const build = Effect.gen(function* () {
          const x = yield* Tensor.fromTypedArray(new Float64Array([0.5, -1.5, 2.25, -3.75, 0.4, 1.6]), [2, 3])
          const y = yield* Tensor.add(
            yield* Tensor.add(yield* Tensor.floor(x), yield* Tensor.ceil(x)),
            yield* Tensor.mul(yield* Tensor.round(x), 2)
          )
          return yield* values(y)
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        fused.forEach((v, i) => assert.assertTrue(Math.abs(v - unfused[i]) < 1e-12, `[${i}]: ${v} != ${unfused[i]}`))
      })
    )

    it.effect("huber, hinge and klDiv elementwise chains fuse and agree with unfused evaluation", () =>
      Effect.gen(function* () {
        const build = Effect.gen(function* () {
          const pred = yield* Tensor.fromTypedArray(new Float64Array([1.5, -2, 0.25, 3, -0.5, 0.75]), [2, 3])
          const target = yield* Tensor.fromTypedArray(new Float64Array([1, 0.5, -1, 2.5, 0, 1.25]), [2, 3])
          const h1 = yield* Loss.huber(pred, target, { reduction: "none" })
          const h2 = yield* Loss.hinge(pred, target, { reduction: "none" })
          const probs = yield* Tensor.fromTypedArray(new Float64Array([0.2, 0.3, 0.5, 0.1, 0.4, 0.25]), [2, 3])
          const logPred = yield* Tensor.log(probs)
          const kl = yield* Loss.klDiv(logPred, probs, { reduction: "none" })
          return {
            h1: yield* values(h1),
            h2: yield* values(h2),
            kl: yield* values(kl)
          }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["h1", "h2", "kl"] as const) {
          fused[key].forEach((v, i) => {
            assert.assertTrue(Math.abs(v - unfused[key][i]) < 1e-12, `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      })
    )
    it.effect("where with a single-consumer comparison fuses as a true select (elu)", () =>
      Effect.gen(function* () {
        const build = Effect.gen(function* () {
          const x = yield* Tensor.fromTypedArray(new Float64Array([0.5, -1, 2, -3, 0.25, 1.5]), [2, 3])
          const y = yield* Tensor.elu(x, { alpha: 0.7 })
          const loss = yield* Tensor.sum(y)
          const [gx] = yield* Gradient.grad(loss, [x])
          return { y: yield* values(y), gx: yield* values(gx) }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["y", "gx"] as const) {
          fused[key].forEach((v, i) => {
            assert.assertTrue(Math.abs(v - unfused[key][i]) < 1e-12, `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      })
    )

    it.effect("select does not propagate NaN from the unselected side (klDiv with zero targets)", () =>
      Effect.gen(function* () {
        const build = Effect.gen(function* () {
          // log(0) = -inf and 0 * -inf = NaN in the masked branch: an
          // arithmetic mask would poison the result, a true select must not
          const probs = yield* Tensor.fromTypedArray(new Float64Array([0, 0.3, 0, 0.1, 0, 0.25]), [2, 3])
          const logPred = yield* Tensor.log(yield* Tensor.fromTypedArray(
            new Float64Array([0.2, 0.3, 0.5, 0.1, 0.4, 0.25]),
            [2, 3]
          ))
          const kl = yield* Loss.klDiv(logPred, probs, { reduction: "none" })
          return yield* values(kl)
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        fused.forEach((v, i) => {
          assert.assertTrue(Number.isFinite(v), `fused[${i}] is not finite: ${v}`)
          assert.assertTrue(Math.abs(v - unfused[i]) < 1e-12, `[${i}]: ${v} != ${unfused[i]}`)
        })
      })
    )

    it.effect("dropout's mask and scale fuse; survivors are exactly x / (1 - p)", () =>
      Effect.gen(function* () {
        const build = Effect.gen(function* () {
          const x = yield* Tensor.fromTypedArray(new Float64Array([0.5, -1, 2, -3, 0.25, 1.5, 4, -0.75]), [2, 4])
          const y = yield* Tensor.dropout(x, { p: 0.5 })
          return yield* values(y)
        })
        for (const fusion of [true, false]) {
          const out = yield* withFusion(fusion, build)
          const xs = [0.5, -1, 2, -3, 0.25, 1.5, 4, -0.75]
          out.forEach((v, i) => {
            assert.assertTrue(
              v === 0 || Math.abs(v - xs[i] / 0.5) < 1e-12,
              `[${i}]: ${v} is neither 0 nor ${xs[i] / 0.5} (fusion ${fusion})`
            )
          })
        }
      })
    )

    it.effect("where with a shared condition stays correct", () =>
      Effect.gen(function* () {
        const build = Effect.gen(function* () {
          const x = yield* Tensor.fromTypedArray(new Float64Array([0.5, -1, 2, -3, 0.25, 1.5]), [2, 3])
          // the condition has two consumers, so it must materialize as u8
          const cond = yield* Tensor.gt(x, 0)
          const a = yield* Tensor.where(cond, x, 0)
          const b = yield* Tensor.where(cond, 0, x)
          return { a: yield* values(a), b: yield* values(b) }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["a", "b"] as const) {
          fused[key].forEach((v, i) => {
            assert.assertTrue(Math.abs(v - unfused[key][i]) < 1e-12, `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      })
    )
  })
})

import { describe } from "@effect/vitest"
import * as assert from "@effect/vitest/utils"
import { Effect } from "effect"
import { Gradient, Loss, Tensor } from "../src/index.ts"
import { floats, onDevices, TOL } from "./utils/devices.ts"

const values = (t: Tensor.Any) =>
  Effect.map(Tensor.toTypedArray(t), (arr) => Array.from<number | bigint>(arr).map(Number))

// Fusion is selected from a process-global compiler input. Bracketing prevents
// one oracle arm, failure included, from changing later compilations.
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

onDevices("Fusion", () => (it) => {
  // fused and unfused paths agree up to float rounding, except erf (an
  // A&S approximation, ~1.5e-7: Metal has no erf) and pow with common
  // exponents (lowered to multiplies/sqrt instead of powf)
  const close = (a: number, b: number): boolean => Math.abs(a - b) <= TOL * Math.max(1, Math.abs(a), Math.abs(b))
  describe("region fusion", () => {
    it.effect("strided lanes: permuted and narrowed views fuse with correct storage strides", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const a = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4, 5, 6]), [2, 3])
          const b = yield* Tensor.fromTypedArray(floats([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]), [2, 3])
          // Permuted and narrowed views feeding a fused elementwise
          // chain: lanes must be read through their storage strides,
          // not as if dense.
          const ap = yield* Tensor.transpose(a, [1, 0])
          const bp = yield* Tensor.transpose(b, [1, 0])
          const sliced = yield* Tensor.slice(bp, { start: [0, 0], end: [3, 1] })
          const chain = yield* Tensor.tanh(
            yield* Tensor.mul(yield* Tensor.add(ap, sliced), yield* Tensor.add(ap, sliced))
          )
          return yield* values(chain)
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        assert.deepStrictEqual(fused, unfused)
        // and the values are the true tanh((a^T + b^T[.., :1])²):
        const aT = [1, 4, 2, 5, 3, 6]
        const bTs = [0.5, 3.5, 0.5, 3.5, 0.5, 3.5]
        const expected = aT.map((a, i) => {
          const s = a + bTs[i]!
          return Math.tanh(s * s)
        })
        for (let i = 0; i < expected.length; i++) {
          assert.assertTrue(Math.abs(fused[i]! - expected[i]!) < 1e-5, `${fused[i]} != ${expected[i]}`)
        }
      }))

    it.effect("fused and unfused evaluation agree on values and gradients", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const a = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4, 5, 6]), [2, 3])
          const b = yield* Tensor.fromTypedArray(floats([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]), [2, 3])
          // a fused chain with a scalar constant fold and a two-region merge:
          // mul and add each open a region, the outer add merges them
          const left = yield* Tensor.mul(a, b)
          const right = yield* Tensor.add(a, yield* Tensor.constantLike(a, 2))
          const merged = yield* Tensor.add(left, right)
          const y = yield* Tensor.exp(yield* Tensor.sqrt(merged))
          const z = yield* Tensor.relu(yield* Tensor.sub(y, yield* Tensor.constantLike(y, 2)))
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
            assert.assertTrue(close(v, unfused[key][i]), `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      }))

    it.effect("tanh, abs and erf chains fuse and agree with unfused evaluation", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const x = yield* Tensor.fromTypedArray(floats([0.5, -1, 2, -3, 0.25, 1.5]), [2, 3])
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
            assert.assertTrue(close(v, unfused[key][i]), `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      }))

    it.effect("broadcast lanes fuse (bias add, row subtract) and agree with unfused evaluation", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const x = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4, 5, 6]), [2, 3])
          // [2, 3] - [1, 3] rides the region as a broadcast lane
          const row = yield* Tensor.fromTypedArray(floats([0.5, 0.5, 0.5]), [1, 3])
          const y = yield* Tensor.sqrt(yield* Tensor.mul(yield* Tensor.sub(x, row), yield* Tensor.constantLike(x, 2)))
          const loss = yield* Tensor.sum(y)
          const [gx] = yield* Gradient.grad(loss, [x])
          return { y: yield* values(y), gx: yield* values(gx) }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["y", "gx"] as const) {
          fused[key].forEach((v, i) => assert.assertTrue(close(v, unfused[key][i])))
        }
      }))

    it.effect("softmax-style broadcast chains (row-max subtract, computed scalar) fuse", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const x = yield* Tensor.fromTypedArray(
            floats([1, -2, 3, -4, 0.5, -0.5, 2, -1, 0.1, 0.2, 0.3, 0.4, -3, 2.5, 1.5, -0.7]),
            [4, 4]
          )
          // [4,4] - [4,1] keepdims broadcast, the fused region crosses it
          const m = yield* Tensor.max(x, { dims: [1], keepdims: true })
          const y = yield* Tensor.tanh(yield* Tensor.exp(yield* Tensor.sub(x, m)))
          // a computed rank-0 scalar lane (not a constructor constant)
          const s = yield* Tensor.mean(y)
          const z = yield* Tensor.mul(y, yield* Tensor.exp(s))
          const loss = yield* Tensor.sum(z)
          const [gx] = yield* Gradient.grad(loss, [x])
          return { y: yield* values(y), z: yield* values(z), gx: yield* values(gx) }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["y", "z", "gx"] as const) {
          fused[key].forEach((v, i) => {
            assert.assertTrue(close(v, unfused[key][i]), `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      }))

    it.effect("rank-3 broadcast lanes fuse (middle-dim and trailing-dim strides)", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const a = yield* Tensor.fromTypedArray(
            floats(Array.from({ length: 24 }, (_, i) => (i % 7) - 3)),
            [2, 3, 4]
          )
          const b = yield* Tensor.fromTypedArray(floats([0.5, -1.5, 2]), [1, 3, 1])
          const c = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4]), [4])
          const y = yield* Tensor.relu(yield* Tensor.add(yield* Tensor.mul(a, b), c))
          const loss = yield* Tensor.sum(y)
          const [ga] = yield* Gradient.grad(loss, [a])
          return { y: yield* values(y), ga: yield* values(ga) }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["y", "ga"] as const) {
          fused[key].forEach((v, i) => assert.assertTrue(close(v, unfused[key][i])))
        }
      }))

    it.effect("sign fuses through comparisons and agrees with unfused evaluation", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const x = yield* Tensor.fromTypedArray(floats([-2, -0.5, 0, 0.5, 2, -1.5]), [2, 3])
          const y = yield* Tensor.mul(
            yield* Tensor.sign(yield* Tensor.sub(x, yield* Tensor.constantLike(x, 0.25))),
            yield* Tensor.constantLike(x, 3)
          )
          const loss = yield* Tensor.sum(y)
          const [gx] = yield* Gradient.grad(loss, [x])
          return { y: yield* values(y), gx: yield* values(gx) }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["y", "gx"] as const) {
          fused[key].forEach((v, i) => assert.assertTrue(close(v, unfused[key][i])))
        }
      }))

    it.effect("identity casts are transparent inside a region", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const x = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4]), [2, 2])
          const y = yield* Tensor.sqrt(
            yield* Tensor.cast(yield* Tensor.mul(x, yield* Tensor.constantLike(x, 2)), "f32")
          )
          return yield* values(y)
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        fused.forEach((v, i) => assert.assertTrue(close(v, unfused[i])))
      }))

    it.effect("a shared fused prefix compiles to one multi-output kernel", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          // y is a fused region with several consumers: the fused
          // continuations (z, w) merge with it into one multi-output
          // kernel, while the unfused consumer (sum) keeps y materialized.
          // Small magnitudes: sin of a large argument would amplify the
          // (valid) rounding difference between inlined and materialized y
          const a = yield* Tensor.fromTypedArray(floats([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), [2, 3])
          const b = yield* Tensor.fromTypedArray(floats([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]), [2, 3])
          const y = yield* Tensor.exp(
            yield* Tensor.sqrt(yield* Tensor.add(yield* Tensor.mul(a, b), yield* Tensor.constantLike(a, 2)))
          )
          const z = yield* Tensor.sin(y)
          const w = yield* Tensor.cos(y)
          const loss = yield* Tensor.sum(yield* Tensor.add(yield* Tensor.mul(y, z), w))
          const [ga, gb] = yield* Gradient.grad(loss, [a, b])
          return {
            y: yield* values(y),
            z: yield* values(z),
            w: yield* values(w),
            ga: yield* values(ga),
            gb: yield* values(gb)
          }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["y", "z", "w", "ga", "gb"] as const) {
          fused[key].forEach((v, i) => {
            assert.assertTrue(close(v, unfused[key][i]), `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      }))

    it.effect("regions folding to zero lanes evaluate plainly", () =>
      Effect.gen(function*() {
        // relu(full + full): both operands fold to constants, leaving a
        // region with no input lanes — it must not reach the fused node
        const y = yield* Tensor.relu(yield* Tensor.add(yield* Tensor.full([], 1), yield* Tensor.full([], 2)))
        const fused = yield* withFusion(true, values(y))
        const unfused = yield* withFusion(false, values(y))
        assert.deepStrictEqual(fused, unfused)
        assert.deepStrictEqual(fused, [3])
      }))

    it.effect("log chains (softplus, mish, logSoftmax) fuse and agree with unfused evaluation", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const x = yield* Tensor.fromTypedArray(floats([0.5, -1, 2, -3, 0.25, 1.5]), [2, 3])
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
            assert.assertTrue(close(v, unfused[key][i]), `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      }))

    it.effect("binary cross entropy chains fuse and agree with unfused evaluation", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const logits = yield* Tensor.fromTypedArray(floats([1.5, -2, 0.25, 3, -0.5, 0.75]), [2, 3])
          const target = yield* Tensor.fromTypedArray(floats([1, 0, 1, 1, 0, 1]), [2, 3])
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
            assert.assertTrue(close(v, unfused[key][i]), `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      }))

    it.effect("pow exponents fuse (square, cube, sqrt forms, generic) and agree with unfused evaluation", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const x = yield* Tensor.fromTypedArray(floats([0.5, 1, 2, 3, 0.25, 1.5]), [2, 3])
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
            assert.assertTrue(close(v, unfused[key][i]), `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      }))

    it.effect("floor, ceil and round fuse and agree with unfused evaluation", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const x = yield* Tensor.fromTypedArray(floats([0.5, -1.5, 2.25, -3.75, 0.4, 1.6]), [2, 3])
          const y = yield* Tensor.add(
            yield* Tensor.add(yield* Tensor.floor(x), yield* Tensor.ceil(x)),
            yield* Tensor.mul(yield* Tensor.round(x), yield* Tensor.constantLike(x, 2))
          )
          return yield* values(y)
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        fused.forEach((v, i) => assert.assertTrue(close(v, unfused[i]), `[${i}]: ${v} != ${unfused[i]}`))
      }))

    it.effect("huber, hinge and klDiv elementwise chains fuse and agree with unfused evaluation", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const pred = yield* Tensor.fromTypedArray(floats([1.5, -2, 0.25, 3, -0.5, 0.75]), [2, 3])
          const target = yield* Tensor.fromTypedArray(floats([1, 0.5, -1, 2.5, 0, 1.25]), [2, 3])
          const h1 = yield* Loss.huber(pred, target, { reduction: "none" })
          const h2 = yield* Loss.hinge(pred, target, { reduction: "none" })
          const probs = yield* Tensor.fromTypedArray(floats([0.2, 0.3, 0.5, 0.1, 0.4, 0.25]), [2, 3])
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
            assert.assertTrue(close(v, unfused[key][i]), `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      }))
    it.effect("where with a single-consumer comparison fuses as a true select (elu)", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const x = yield* Tensor.fromTypedArray(floats([0.5, -1, 2, -3, 0.25, 1.5]), [2, 3])
          const y = yield* Tensor.elu(x, { alpha: 0.7 })
          const loss = yield* Tensor.sum(y)
          const [gx] = yield* Gradient.grad(loss, [x])
          return { y: yield* values(y), gx: yield* values(gx) }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["y", "gx"] as const) {
          fused[key].forEach((v, i) => {
            assert.assertTrue(close(v, unfused[key][i]), `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      }))

    it.effect("select does not propagate NaN from the unselected side (klDiv with zero targets)", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          // log(0) = -inf and 0 * -inf = NaN in the masked branch: an
          // arithmetic mask would poison the result, a true select must not
          const probs = yield* Tensor.fromTypedArray(floats([0, 0.3, 0, 0.1, 0, 0.25]), [2, 3])
          const logPred = yield* Tensor.log(
            yield* Tensor.fromTypedArray(
              floats([0.2, 0.3, 0.5, 0.1, 0.4, 0.25]),
              [2, 3]
            )
          )
          const kl = yield* Loss.klDiv(logPred, probs, { reduction: "none" })
          return yield* values(kl)
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        fused.forEach((v, i) => {
          assert.assertTrue(Number.isFinite(v), `fused[${i}] is not finite: ${v}`)
          assert.assertTrue(close(v, unfused[i]), `[${i}]: ${v} != ${unfused[i]}`)
        })
      }))

    it.effect("dropout's mask and scale fuse; survivors are exactly x / (1 - p)", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const x = yield* Tensor.fromTypedArray(floats([0.5, -1, 2, -3, 0.25, 1.5, 4, -0.75]), [2, 4])
          const y = yield* Tensor.dropout(x, { p: 0.5 })
          return yield* values(y)
        })
        for (const fusion of [true, false]) {
          const out = yield* withFusion(fusion, build)
          const xs = [0.5, -1, 2, -3, 0.25, 1.5, 4, -0.75]
          out.forEach((v, i) => {
            assert.assertTrue(
              v === 0 || close(v, xs[i] / 0.5),
              `[${i}]: ${v} is neither 0 nor ${xs[i] / 0.5} (fusion ${fusion})`
            )
          })
        }
      }))

    it.effect("where with a shared condition stays correct", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const x = yield* Tensor.fromTypedArray(floats([0.5, -1, 2, -3, 0.25, 1.5]), [2, 3])
          // the condition has two consumers, so it must materialize as u8
          const cond = yield* Tensor.gt(x, yield* Tensor.constantLike(x, 0))
          const zero = yield* Tensor.constantLike(x, 0)
          const a = yield* Tensor.where(cond, x, zero)
          const b = yield* Tensor.where(cond, zero, x)
          return { a: yield* values(a), b: yield* values(b) }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["a", "b"] as const) {
          fused[key].forEach((v, i) => {
            assert.assertTrue(close(v, unfused[key][i]), `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      }))
  })

  describe("reduction fusion", () => {
    it.effect("a sum over an elementwise chain fuses into the reduce loop", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const x = yield* Tensor.fromTypedArray(floats([0.5, -1, 2, -3, 0.25, 1.5, 0.75, -0.5]), [2, 4])
          // the exp/mul chain must not materialize: sum folds it in-loop
          const two = yield* Tensor.constantLike(x, 2)
          const keep = yield* Tensor.sum(yield* Tensor.mul(yield* Tensor.exp(x), two), {
            dims: [1],
            keepdims: true
          })
          const drop = yield* Tensor.sum(yield* Tensor.mul(yield* Tensor.exp(x), two), { dims: [1] })
          const loss = yield* Tensor.sum(keep)
          const [gx] = yield* Gradient.grad(loss, [x])
          return { keep: yield* values(keep), drop: yield* values(drop), gx: yield* values(gx) }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["keep", "drop", "gx"] as const) {
          fused[key].forEach((v, i) => {
            assert.assertTrue(close(v, unfused[key][i]), `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      }))

    it.effect("softmax fuses its max/sum reduces with the exp chains (values and gradients)", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const x = yield* Tensor.fromTypedArray(
            floats([1, -2, 3, -4, 0.5, -0.5, 2, -1, 0.1, 0.2, 0.3, 0.4, -3, 2.5, 1.5, -0.7]),
            [4, 4]
          )
          const y = yield* Tensor.softmax(x)
          const loss = yield* Tensor.sum(
            yield* Tensor.mul(
              y,
              yield* Tensor.fromTypedArray(
                floats(Array.from({ length: 16 }, (_, i) => i * 0.25 - 2)),
                [4, 4]
              )
            )
          )
          const [gx] = yield* Gradient.grad(loss, [x])
          return { y: yield* values(y), gx: yield* values(gx) }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["y", "gx"] as const) {
          fused[key].forEach((v, i) => {
            assert.assertTrue(close(v, unfused[key][i]), `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      }))

    it.effect("mean over multiple dims (the variance pattern) fuses", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const x = yield* Tensor.fromTypedArray(
            floats(Array.from({ length: 24 }, (_, i) => (i % 7) - 3)),
            [2, 3, 4]
          )
          const mu = yield* Tensor.mean(x, { dims: [1, 2], keepdims: true })
          const centered = yield* Tensor.sub(x, mu)
          const variance = yield* Tensor.mean(yield* Tensor.mul(centered, centered), { dims: [1, 2] })
          const loss = yield* Tensor.sum(variance)
          const [gx] = yield* Gradient.grad(loss, [x])
          return {
            mu: yield* values(mu),
            variance: yield* values(variance),
            gx: yield* values(gx)
          }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["mu", "variance", "gx"] as const) {
          fused[key].forEach((v, i) => {
            assert.assertTrue(close(v, unfused[key][i]), `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      }))

    it.effect("max and min over a chain fuse", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const x = yield* Tensor.fromTypedArray(floats([0.5, -1, 2, -3, 0.25, 1.5, 0.75, -0.5, 1.25, -2.5, 3, 0.1]), [
            4,
            3
          ])
          const hi = yield* Tensor.max(yield* Tensor.tanh(x), { dims: [0] })
          const lo = yield* Tensor.min(yield* Tensor.tanh(x), { dims: [0] })
          return { hi: yield* values(hi), lo: yield* values(lo) }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["hi", "lo"] as const) {
          fused[key].forEach((v, i) => {
            assert.assertTrue(close(v, unfused[key][i]), `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      }))

    it.effect("non-trailing dims at rank 5 fuse without touching candle's reduce", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const x = yield* Tensor.fromTypedArray(
            floats(Array.from({ length: 2 * 3 * 2 * 3 * 2 }, (_, i) => ((i * 7) % 11) - 5)),
            [2, 3, 2, 3, 2]
          )
          const y = yield* Tensor.sum(yield* Tensor.exp(yield* Tensor.mul(x, yield* Tensor.constantLike(x, 0.25))), {
            dims: [1, 3]
          })
          const loss = yield* Tensor.sum(y)
          const [gx] = yield* Gradient.grad(loss, [x])
          return { y: yield* values(y), gx: yield* values(gx) }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["y", "gx"] as const) {
          fused[key].forEach((v, i) => {
            assert.assertTrue(close(v, unfused[key][i]), `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      }))

    it.effect("broadcast lanes inside a fused reduce read through stride-0 dims", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const x = yield* Tensor.fromTypedArray(
            floats([1, -2, 3, -4, 0.5, -0.5, 2, -1, 0.1, 0.2, 0.3, 0.4, -3, 2.5, 1.5, -0.7]),
            [4, 4]
          )
          const row = yield* Tensor.fromTypedArray(floats([0.5, -1, 2, 0.25]), [4, 1])
          const col = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4]), [4])
          const y = yield* Tensor.sum(yield* Tensor.mul(yield* Tensor.sub(x, row), col), { dims: [1] })
          const loss = yield* Tensor.sum(y)
          const [gx] = yield* Gradient.grad(loss, [x])
          return { y: yield* values(y), gx: yield* values(gx) }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["y", "gx"] as const) {
          fused[key].forEach((v, i) => {
            assert.assertTrue(close(v, unfused[key][i]), `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      }))

    it.effect("nested shared prefixes with deep lane ancestry merge and terminate", () =>
      Effect.gen(function*() {
        // Regression: the multi-output merge used to keep the original
        // lane nodes alive in the merged kernel's inputs, so every round
        // retained and re-merged the old subgraph — the rewrite grew
        // without bound on graphs with nested sharing (deep lane
        // ancestry). Each level here shares a fused prefix between fused
        // continuations and a reduce consumer, which is exactly the
        // shape that blew up.
        const build = Effect.gen(function*() {
          const x = yield* Tensor.fromTypedArray(floats([0.5, -1, 2, -3, 0.25, 1.5]), [2, 3])
          let acc = x as Tensor.Any
          const terms: Array<Tensor.Any> = []
          for (let level = 0; level < 6; level++) {
            const shared = yield* Tensor.tanh(
              yield* Tensor.exp(yield* Tensor.add(acc, yield* Tensor.constantLike(acc, level)))
            )
            const left = yield* Tensor.sin(yield* Tensor.mul(shared, yield* Tensor.constantLike(shared, 2)))
            const right = yield* Tensor.cos(yield* Tensor.add(shared, yield* Tensor.constantLike(shared, 1)))
            const reduced = yield* Tensor.sum(yield* Tensor.sqrt(yield* Tensor.abs(shared)), { dims: [1] })
            terms.push(yield* Tensor.sum(left), yield* Tensor.sum(right), yield* Tensor.sum(reduced))
            acc = yield* Tensor.add(shared, yield* Tensor.constantLike(shared, level + 1))
          }
          let loss = terms[0]
          for (const t of terms.slice(1)) {
            loss = yield* Tensor.add(loss, t)
          }
          const [gx] = yield* Gradient.grad(loss, [x])
          return { loss: yield* values(loss), gx: yield* values(gx) }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["loss", "gx"] as const) {
          fused[key].forEach((v, i) => {
            assert.assertTrue(close(v, unfused[key][i]), `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      }))

    it.effect("a prefix-dependent continuation uses the split multi-output topology", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const x = yield* Tensor.fromTypedArray(floats([0.5, -1, 2, -3, 0.25, 1.5]), [2, 3])
          const prefix = yield* Tensor.tanh(
            yield* Tensor.add(x, yield* Tensor.constantLike(x, 0.25))
          )
          const safe = yield* Tensor.exp(yield* Tensor.neg(prefix))
          const reduced = yield* Tensor.sum(yield* Tensor.abs(prefix), { dims: [1], keepdims: true })
          const dependent = yield* Tensor.sin(yield* Tensor.div(prefix, reduced))
          return {
            prefix: yield* values(prefix),
            safe: yield* values(safe),
            reduced: yield* values(reduced),
            dependent: yield* values(dependent)
          }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["prefix", "safe", "reduced", "dependent"] as const) {
          fused[key].forEach((v, i) => {
            assert.assertTrue(close(v, unfused[key][i]), `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      }))

    it.effect("a reduce over a materialized prefix stays unfused but correct", () =>
      Effect.gen(function*() {
        const build = Effect.gen(function*() {
          const x = yield* Tensor.fromTypedArray(floats([0.5, -1, 2, -3, 0.25, 1.5, 0.75, -0.5]), [2, 4])
          // e has two consumers (the sum and the product): it materializes
          // and the reduce reads it as a plain input
          const e = yield* Tensor.exp(x)
          const s = yield* Tensor.sum(e, { dims: [1], keepdims: true })
          const y = yield* Tensor.div(e, s)
          const loss = yield* Tensor.sum(y)
          const [gx] = yield* Gradient.grad(loss, [x])
          return { y: yield* values(y), gx: yield* values(gx) }
        })
        const fused = yield* withFusion(true, build)
        const unfused = yield* withFusion(false, build)
        for (const key of ["y", "gx"] as const) {
          fused[key].forEach((v, i) => {
            assert.assertTrue(close(v, unfused[key][i]), `${key}[${i}]: ${v} != ${unfused[key][i]}`)
          })
        }
      }))
  })
})

import { describe, expect } from "@effect/vitest"
import * as assert from "@effect/vitest/utils"
import { Effect } from "effect"
import { Gradient, Tensor } from "../src/index.ts"
import { deep, floats, GRADCHECK_EPS, GRADCHECK_TOL, onDevices, TOL } from "./utils/devices.ts"

const values = (t: Tensor.Any) => Tensor.toNumberArray(t)

const close = (a: number, b: number): boolean => Math.abs(a - b) <= TOL * Math.max(1, Math.abs(a), Math.abs(b))

// The composed ops the native reference eval arm mirrors; gradients flow
// through the general autodiff machinery here, so parity with the
// closed-form native backward is a real check.
const reference = (
  q: Tensor.Any,
  k: Tensor.Any,
  v: Tensor.Any,
  options: { readonly scale: number; readonly causal: boolean }
) =>
  Effect.gen(function*() {
    const rank = q.shape.length
    const t = q.shape[rank - 2]
    const s = k.shape[rank - 2]
    const perm = Array.from({ length: rank }, (_, i) => i)
    perm[rank - 2] = rank - 1
    perm[rank - 1] = rank - 2
    let scores = yield* Tensor.mul(
      yield* Tensor.matmul(q, yield* Tensor.transpose(k, perm)),
      yield* Tensor.constantLike(q, options.scale)
    )
    if (options.causal) {
      const i = yield* Tensor.reshape(yield* Tensor.arange(t), [t, 1])
      const j = yield* Tensor.reshape(yield* Tensor.arange(s), [1, s])
      const allowed = yield* Tensor.le(j, yield* Tensor.add(i, yield* Tensor.constantLike(i, Math.max(0, s - t))))
      const negInf = yield* Tensor.full([t, s], -Infinity)
      scores = yield* Tensor.add(scores, yield* Tensor.where(allowed, yield* Tensor.constantLike(negInf, 0), negInf))
    }
    return yield* Tensor.matmul(yield* Tensor.softmax(scores), v)
  })

// deterministic, asymmetric values
const pattern = (n: number): Array<number> => Array.from({ length: n }, (_, i) => ((i * 7 + 3) % 13 - 6) / 4)

onDevices("Attention", () => (it) => {
  const f32 = (data: ReadonlyArray<number>, shape: ReadonlyArray<number>) => Tensor.fromTypedArray(floats(data), shape)

  describe("scaledDotProductAttention", () => {
    it.effect("matches the composed reference (values and gradients)", () =>
      Effect.gen(function*() {
        const shape = [2, 2, 3, 4]
        const n = shape.reduce((a, b) => a * b, 1)
        const q = yield* f32(pattern(n), shape)
        const k = yield* f32(pattern(n).map((x) => x * 0.7 + 0.1), shape)
        const v = yield* f32(pattern(n).map((x) => x * -0.5 + 0.2), shape)
        const scale = 1 / Math.sqrt(4)
        const w = yield* f32(pattern(n).map((x) => x * 0.3), shape)

        const out = yield* Tensor.scaledDotProductAttention(q, k, v)
        const expected = yield* reference(q, k, v, { scale, causal: false })
        const [a, b] = [yield* values(out), yield* values(expected)]
        a.forEach((x, i) => assert.assertTrue(close(x, b[i]), `out[${i}]: ${x} != ${b[i]}`))

        const loss = yield* Tensor.sum(yield* Tensor.mul(out, w))
        const [dq, dk, dv] = yield* Tensor.compute(yield* Gradient.grad(loss, [q, k, v]))
        const lossRef = yield* Tensor.sum(yield* Tensor.mul(yield* reference(q, k, v, { scale, causal: false }), w))
        const [eq, ek, ev] = yield* Tensor.compute(yield* Gradient.grad(lossRef, [q, k, v]))
        for (const [g, e, name] of [[dq, eq, "dq"], [dk, ek, "dk"], [dv, ev, "dv"]] as const) {
          const gv = yield* values(g)
          const evv = yield* values(e)
          gv.forEach((x, i) => assert.assertTrue(close(x, evv[i]), `${name}[${i}]: ${x} != ${evv[i]}`))
        }
      }))

    it.effect("causal masking matches the composed reference (values and gradients)", () =>
      Effect.gen(function*() {
        const shape = [2, 2, 4, 4]
        const n = shape.reduce((a, b) => a * b, 1)
        const q = yield* f32(pattern(n), shape)
        const k = yield* f32(pattern(n).map((x) => x * 0.7 + 0.1), shape)
        const v = yield* f32(pattern(n).map((x) => x * -0.5 + 0.2), shape)
        const scale = 1 / Math.sqrt(4)

        const out = yield* Tensor.scaledDotProductAttention(q, k, v, { causal: true })
        const expected = yield* reference(q, k, v, { scale, causal: true })
        const [a, b] = [yield* values(out), yield* values(expected)]
        a.forEach((x, i) => assert.assertTrue(close(x, b[i]), `out[${i}]: ${x} != ${b[i]}`))

        const loss = yield* Tensor.sum(yield* Tensor.mul(out, v))
        const grads = yield* Tensor.compute(yield* Gradient.grad(loss, [q, k, v]))
        const lossRef = yield* Tensor.sum(yield* Tensor.mul(yield* reference(q, k, v, { scale, causal: true }), v))
        const expected_ = yield* Tensor.compute(yield* Gradient.grad(lossRef, [q, k, v]))
        for (let j = 0; j < 3; j++) {
          const gv = yield* values(grads[j])
          const ev = yield* values(expected_[j])
          gv.forEach((x, i) => assert.assertTrue(close(x, ev[i]), `grad[${j}][${i}]: ${x} != ${ev[i]}`))
        }
      }))

    it.effect("a custom scale matches the composed reference", () =>
      Effect.gen(function*() {
        const shape = [1, 2, 3, 8]
        const n = shape.reduce((a, b) => a * b, 1)
        const q = yield* f32(pattern(n), shape)
        const k = yield* f32(pattern(n).map((x) => x * 0.7 + 0.1), shape)
        const v = yield* f32(pattern(n).map((x) => x * -0.5 + 0.2), shape)
        const out = yield* Tensor.scaledDotProductAttention(q, k, v, { scale: 0.5 })
        const expected = yield* reference(q, k, v, { scale: 0.5, causal: false })
        const [a, b] = [yield* values(out), yield* values(expected)]
        a.forEach((x, i) => assert.assertTrue(close(x, b[i]), `out[${i}]: ${x} != ${b[i]}`))
      }))

    it.effect("queries shorter than the key sequence use a right-aligned causal window", () =>
      Effect.gen(function*() {
        const q = yield* f32(pattern(2 * 4), [1, 1, 2, 4])
        const k = yield* f32(pattern(5 * 4).map((x) => x * 0.7), [1, 1, 5, 4])
        const v = yield* f32(pattern(5 * 4).map((x) => x * -0.5), [1, 1, 5, 4])
        const scale = 1 / Math.sqrt(4)
        for (const causal of [false, true]) {
          const out = yield* Tensor.scaledDotProductAttention(q, k, v, { causal })
          expect(out.shape).toEqual([1, 1, 2, 4])
          const expected = yield* reference(q, k, v, { scale, causal })
          const [a, b] = [yield* values(out), yield* values(expected)]
          a.forEach((x, i) => assert.assertTrue(close(x, b[i]), `out[${i}] (causal ${causal}): ${x} != ${b[i]}`))
        }
      }))

    it.effect("numeric gradcheck", () =>
      Effect.gen(function*() {
        // gradcheck each of q, k, v element-wise against central differences
        const inputs = [pattern(8), pattern(8).map((x) => x * 0.7), pattern(8).map((x) => x * -0.5)]
        const shapes: Array<ReadonlyArray<number>> = [
          [1, 1, 2, 4],
          [1, 1, 2, 4],
          [1, 1, 2, 4]
        ]
        for (let which = 0; which < 3; which++) {
          const build = (vals: ReadonlyArray<number>) =>
            Effect.gen(function*() {
              const parts: Array<Tensor.Any> = []
              for (let j = 0; j < 3; j++) {
                parts.push(yield* f32(j === which ? vals : inputs[j], shapes[j]))
              }
              const out = yield* Tensor.scaledDotProductAttention(parts[0], parts[1], parts[2], {
                causal: true
              })
              return yield* Tensor.sum(out)
            })
          const x = yield* f32(inputs[which], shapes[which])
          const others: Array<Tensor.Any> = []
          for (let j = 0; j < 3; j++) {
            others.push(j === which ? x : yield* f32(inputs[j], shapes[j]))
          }
          const loss = yield* Tensor.scaledDotProductAttention(others[0], others[1], others[2], {
            causal: true
          })
          const total = yield* Tensor.sum(loss)
          const [analytic] = yield* Gradient.grad(total, [x])
          const analyticValues = yield* values(analytic)
          for (let i = 0; i < inputs[which].length; i++) {
            const plus = inputs[which].map((v, j) => (j === i ? v + GRADCHECK_EPS : v))
            const minus = inputs[which].map((v, j) => (j === i ? v - GRADCHECK_EPS : v))
            const fp = yield* values(yield* build(plus))
            const fm = yield* values(yield* build(minus))
            const numeric = (fp[0] - fm[0]) / (2 * GRADCHECK_EPS)
            expect(Math.abs(analyticValues[i] - numeric)).toBeLessThan(GRADCHECK_TOL)
          }
        }
      }))

    it.effect("validation failures are TensorError", () =>
      Effect.gen(function*() {
        const q = yield* f32(pattern(8), [1, 1, 2, 4])
        const kBad = yield* f32(pattern(6), [1, 1, 2, 3])
        const e1 = yield* Effect.flip(Tensor.scaledDotProductAttention(q, kBad, kBad))
        expect(e1._tag).toBe("TensorError")
        const kLead = yield* f32(pattern(16), [2, 1, 2, 4])
        const vLead = yield* f32(pattern(16), [2, 1, 2, 4])
        const e2 = yield* Effect.flip(Tensor.scaledDotProductAttention(q, kLead, vLead))
        expect(e2._tag).toBe("TensorError")
        const vSeq = yield* f32(pattern(12), [1, 1, 3, 4])
        const kOk = yield* f32(pattern(8), [1, 1, 2, 4])
        const e3 = yield* Effect.flip(Tensor.scaledDotProductAttention(q, kOk, vSeq))
        expect(e3._tag).toBe("TensorError")
      }))

    it.effect("grouped-query attention matches explicitly repeated K/V heads", () =>
      Effect.gen(function*() {
        for (const [queryHeads, kvHeads] of [[4, 2], [16, 1]] as const) {
          const q = yield* f32(pattern(queryHeads * 2 * 2), [1, queryHeads, 2, 2])
          const k = yield* f32(pattern(kvHeads * 3 * 2).map((x) => x * 0.7), [1, kvHeads, 3, 2])
          const v = yield* f32(pattern(kvHeads * 3 * 3).map((x) => x * -0.5), [1, kvHeads, 3, 3])
          const repeatHeads = (value: Tensor.Any, width: number) =>
            Effect.gen(function*() {
              const heads: Array<Tensor.Any> = []
              for (let head = 0; head < kvHeads; head++) {
                const current = yield* Tensor.slice(value, {
                  start: [0, head, 0, 0],
                  end: [1, head + 1, 3, width]
                })
                for (let group = 0; group < queryHeads / kvHeads; group++) heads.push(current)
              }
              return yield* Tensor.concat(heads as [Tensor.Any, Tensor.Any, ...Array<Tensor.Any>], { dim: 1 })
            })
          const repeatedK = yield* repeatHeads(k, 2)
          const repeatedV = yield* repeatHeads(v, 3)

          const actual = yield* Tensor.scaledDotProductAttention(q, k, v, { causal: true })
          const expected = yield* Tensor.scaledDotProductAttention(q, repeatedK, repeatedV, { causal: true })
          expect(actual.shape).toEqual([1, queryHeads, 2, 3])
          const [a, b] = [yield* values(actual), yield* values(expected)]
          a.forEach((x, i) => assert.assertTrue(close(x, b[i]), `${queryHeads}:${kvHeads} gqa[${i}]: ${x} != ${b[i]}`))
        }
      }))

    it.effect("rejects an invalid GQA ratio and unequal-head autodiff", () =>
      Effect.gen(function*() {
        const q = yield* f32(pattern(1 * 3 * 2 * 2), [1, 3, 2, 2])
        const k = yield* f32(pattern(1 * 2 * 2 * 2), [1, 2, 2, 2])
        const invalid = yield* Effect.flip(Tensor.scaledDotProductAttention(q, k, k))
        expect(invalid._tag).toBe("TensorError")

        const qGqa = yield* f32(pattern(1 * 4 * 2 * 2), [1, 4, 2, 2])
        const attention = yield* Tensor.scaledDotProductAttention(qGqa, k, k, { causal: true })
        const loss = yield* Tensor.sum(attention)
        const gradientError = yield* Effect.flip(Gradient.grad(loss, [qGqa]))
        expect(gradientError.message).toContain("grouped-query attention with unequal heads")
      }))

    it.effect("local and explicit-full causal windows apply per attention", () =>
      Effect.gen(function*() {
        const q = yield* f32([0, 0, 0, 0], [1, 1, 4, 1])
        const k = yield* f32([0, 0, 0, 0], [1, 1, 4, 1])
        const v = yield* f32([1, 2, 4, 8], [1, 1, 4, 1])
        const local = yield* values(
          yield* Tensor.scaledDotProductAttention(q, k, v, { causal: true, window: 2 })
        )
        const full = yield* values(
          yield* Tensor.scaledDotProductAttention(q, k, v, { causal: true, window: null })
        )
        local.forEach((x, i) => assert.assertTrue(close(x, [1, 1.5, 3, 6][i])))
        full.forEach((x, i) => assert.assertTrue(close(x, [1, 1.5, 7 / 3, 3.75][i])))

        const rightQ = yield* f32([0, 0], [1, 1, 2, 1])
        const rightK = yield* f32([0, 0, 0, 0, 0], [1, 1, 5, 1])
        const rightV = yield* f32([1, 2, 4, 8, 16], [1, 1, 5, 1])
        const rightLocal = yield* Tensor.scaledDotProductAttention(rightQ, rightK, rightV, {
          causal: true,
          window: 2
        })
        const rightFull = yield* Tensor.scaledDotProductAttention(rightQ, rightK, rightV, {
          causal: true,
          window: null
        })
        deep(yield* values(rightLocal), [6, 12])
        deep(yield* values(rightFull), [3.75, 6.2])
        const [localDv] = yield* Tensor.compute(yield* Gradient.grad(yield* Tensor.sum(rightLocal), [rightV]))
        deep(yield* values(localDv), [0, 0, 0.5, 1, 0.5])

        expect((yield* Effect.flip(Tensor.scaledDotProductAttention(q, k, v, { window: 2 })))._tag)
          .toBe("TensorError")
        expect((yield* Effect.flip(Tensor.scaledDotProductAttention(q, k, v, { causal: true, window: 0 })))._tag)
          .toBe("TensorError")
      }))

    it.effect(
      "large multi-tile shapes with non-divisible dims match the reference (values and gradients)",
      () =>
        Effect.gen(function*() {
          const shape = [2, 4, 100, 64]
          const n = shape.reduce((a, b) => a * b, 1)
          const q = yield* f32(pattern(n), shape)
          const k = yield* f32(pattern(n).map((x) => x * 0.7 + 0.1), shape)
          const v = yield* f32(pattern(n).map((x) => x * -0.5 + 0.2), shape)
          const scale = 1 / Math.sqrt(64)
          for (const causal of [false, true]) {
            const out = yield* Tensor.scaledDotProductAttention(q, k, v, { causal })
            const expected = yield* reference(q, k, v, { scale, causal })
            const [a, b] = [yield* values(out), yield* values(expected)]
            a.forEach((x, i) => assert.assertTrue(close(x, b[i]), `out[${i}] (causal ${causal}): ${x} != ${b[i]}`))
            const loss = yield* Tensor.sum(yield* Tensor.mul(out, v))
            const grads = yield* Tensor.compute(yield* Gradient.grad(loss, [q, k, v]))
            const lossRef = yield* Tensor.sum(
              yield* Tensor.mul(yield* reference(q, k, v, { scale, causal }), v)
            )
            const expected_ = yield* Tensor.compute(yield* Gradient.grad(lossRef, [q, k, v]))
            for (let j = 0; j < 3; j++) {
              const gv = yield* values(grads[j])
              const ev = yield* values(expected_[j])
              gv.forEach((x, i) =>
                assert.assertTrue(close(x, ev[i]), `grad[${j}][${i}] (causal ${causal}): ${x} != ${ev[i]}`)
              )
            }
          }
        }),
      30000
    )

    it.effect("value head dim differing from the key head dim", () =>
      Effect.gen(function*() {
        const q = yield* f32(pattern(2 * 5 * 8), [1, 2, 5, 8])
        const k = yield* f32(pattern(2 * 5 * 8).map((x) => x * 0.7), [1, 2, 5, 8])
        const v = yield* f32(pattern(2 * 5 * 6).map((x) => x * -0.5), [1, 2, 5, 6])
        const scale = 1 / Math.sqrt(8)
        for (const causal of [false, true]) {
          const out = yield* Tensor.scaledDotProductAttention(q, k, v, { causal })
          expect(out.shape).toEqual([1, 2, 5, 6])
          const expected = yield* reference(q, k, v, { scale, causal })
          const [a, b] = [yield* values(out), yield* values(expected)]
          a.forEach((x, i) => assert.assertTrue(close(x, b[i]), `out[${i}] (causal ${causal}): ${x} != ${b[i]}`))
        }
      }))

    it.effect("large-magnitude scores exercise the online rescaling", () =>
      Effect.gen(function*() {
        // scores up to ~±60: exp overflow without max subtraction, and the
        // running max must rescale across key tiles
        const shape = [1, 2, 40, 16]
        const n = shape.reduce((a, b) => a * b, 1)
        const q = yield* f32(pattern(n).map((x) => x * 8), shape)
        const k = yield* f32(pattern(n).map((x) => x * 8), shape)
        const v = yield* f32(pattern(n).map((x) => x * -0.5 + 0.2), shape)
        const scale = 1 / Math.sqrt(16)
        for (const causal of [false, true]) {
          const out = yield* Tensor.scaledDotProductAttention(q, k, v, { causal })
          const expected = yield* reference(q, k, v, { scale, causal })
          const [a, b] = [yield* values(out), yield* values(expected)]
          a.forEach((x, i) => {
            assert.assertTrue(Number.isFinite(x), `out[${i}] (causal ${causal}) is not finite: ${x}`)
            assert.assertTrue(close(x, b[i]), `out[${i}] (causal ${causal}): ${x} != ${b[i]}`)
          })
        }
      }))
  })

  describe("rotaryEmbedding", () => {
    it.effect("supports half-split and interleaved-pair layouts", () =>
      Effect.gen(function*() {
        const x = yield* f32([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 4])
        const half = yield* values(yield* Tensor.rotaryEmbedding(x, 2, 10_000))
        const interleavedTensor = yield* Tensor.rotaryEmbedding(x, 2, 10_000, { layout: "InterleavedPairs" })
        const interleaved = yield* values(interleavedTensor)
        const [sin0, cos0] = [Math.sin(1), Math.cos(1)]
        const [sin1, cos1] = [Math.sin(0.01), Math.cos(0.01)]
        const expectedHalf = [
          1,
          2,
          3,
          4,
          5 * cos0 - 7 * sin0,
          6 * cos1 - 8 * sin1,
          7 * cos0 + 5 * sin0,
          8 * cos1 + 6 * sin1
        ]
        const expectedInterleaved = [
          1,
          2,
          3,
          4,
          5 * cos0 - 6 * sin0,
          6 * cos0 + 5 * sin0,
          7 * cos1 - 8 * sin1,
          8 * cos1 + 7 * sin1
        ]
        half.forEach((value, index) => assert.assertTrue(close(value, expectedHalf[index])))
        interleaved.forEach((value, index) => assert.assertTrue(close(value, expectedInterleaved[index])))
        const [gradient] = yield* Tensor.compute(
          yield* Gradient.grad(yield* Tensor.sum(interleavedTensor), [x])
        )
        const expectedGradient = [1, 1, 1, 1, cos0 + sin0, cos0 - sin0, cos1 + sin1, cos1 - sin1]
        const gradientValues = yield* values(gradient)
        gradientValues.forEach((value, index) =>
          assert.assertTrue(close(value, expectedGradient[index]), `rope grad[${index}]`)
        )
      }))
  })
})

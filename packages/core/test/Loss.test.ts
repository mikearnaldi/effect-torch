import { describe, expect } from "@effect/vitest"
import * as assert from "@effect/vitest/utils"
import { Effect } from "effect"
import { Gradient, Loss, type Runtime, Tensor } from "../src/index.ts"
import { floats, GRADCHECK_EPS, GRADCHECK_TOL, onDevices, TOL } from "./utils/devices.ts"

const i64 = (data: ReadonlyArray<bigint>, shape?: ReadonlyArray<number>) =>
  Tensor.fromTypedArray(new BigInt64Array(data), shape)

const values = (t: Tensor.Any) => Tensor.toNumberArray(t)

const scalar = (t: Tensor.Any) => Effect.map(values(t), (v) => v[0])

// Central differences run in f32 on each real backend. The shared epsilon
// clears f32 rounding while the tolerance covers rounding and O(eps^2) error.
const gradcheck = (
  f: (x: Tensor.Lazy) => Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime>,
  input: ReadonlyArray<number>,
  shape: ReadonlyArray<number>
) =>
  Effect.gen(function*() {
    const x = yield* Tensor.fromTypedArray(floats(input), shape)
    const [analytic] = yield* Gradient.grad(yield* f(x), [x])
    const analyticValues = yield* values(analytic)
    for (let i = 0; i < input.length; i++) {
      const plus = input.map((v, j) => (j === i ? v + GRADCHECK_EPS : v))
      const minus = input.map((v, j) => (j === i ? v - GRADCHECK_EPS : v))
      const fp = yield* scalar(yield* f(yield* Tensor.fromTypedArray(floats(plus), shape)))
      const fm = yield* scalar(yield* f(yield* Tensor.fromTypedArray(floats(minus), shape)))
      const numeric = (fp - fm) / (2 * GRADCHECK_EPS)
      expect(Math.abs(analyticValues[i] - numeric)).toBeLessThan(GRADCHECK_TOL)
    }
  })

onDevices("Loss", () => (it) => {
  const f32 = (data: ReadonlyArray<number>, shape?: ReadonlyArray<number>) => Tensor.fromTypedArray(floats(data), shape)
  describe("values", () => {
    it.effect("mse / l1 / huber", () =>
      Effect.gen(function*() {
        const pred = yield* f32([1, 2, 4])
        const target = yield* f32([1, 0, 1])
        const [mseValue] = yield* values(yield* Loss.mse(pred, target))
        expect(Math.abs(mseValue - (0 + 4 + 9) / 3)).toBeLessThan(TOL)
        const [l1Value] = yield* values(yield* Loss.l1(pred, target))
        expect(Math.abs(l1Value - (0 + 2 + 3) / 3)).toBeLessThan(TOL)
        const [huberValue] = yield* values(yield* Loss.huber(pred, target))
        expect(Math.abs(huberValue - (0 + 1.5 + 2.5) / 3)).toBeLessThan(TOL)
        const [huberSum] = yield* values(yield* Loss.huber(pred, target, { reduction: "sum" }))
        expect(Math.abs(huberSum - 4)).toBeLessThan(TOL)
      }))

    it.effect("binaryCrossEntropy from probabilities and logits", () =>
      Effect.gen(function*() {
        const p = yield* f32([0.8, 0.3])
        const y = yield* f32([1, 0])
        const [bce] = yield* values(yield* Loss.binaryCrossEntropy(p, y))
        const expected = -(Math.log(0.8) + Math.log(0.7)) / 2
        expect(Math.abs(bce - expected)).toBeLessThan(TOL)
        const logits = yield* f32([2, -1])
        const [fromLogits] = yield* values(yield* Loss.binaryCrossEntropy(logits, y, { fromLogits: true }))
        const stable = (x: number, t: number) => Math.max(x, 0) - x * t + Math.log1p(Math.exp(-Math.abs(x)))
        expect(Math.abs(fromLogits - (stable(2, 1) + stable(-1, 0)) / 2)).toBeLessThan(TOL)
      }))

    it.effect("crossEntropy and nll", () =>
      Effect.gen(function*() {
        const logits = yield* f32([2, 1, 0, 0, 3, 1], [2, 3])
        const targets = yield* i64([0n, 1n])
        const [ce] = yield* values(yield* Loss.crossEntropy(logits, targets))
        const lsm = (row: Array<number>, cls: number) => {
          const m = Math.max(...row)
          return -(row[cls] - m - Math.log(row.reduce((a, v) => a + Math.exp(v - m), 0)))
        }
        const expected = (lsm([2, 1, 0], 0) + lsm([0, 3, 1], 1)) / 2
        expect(Math.abs(ce - expected)).toBeLessThan(TOL)
        const logProbs = yield* Tensor.logSoftmax(logits)
        const [nllValue] = yield* values(yield* Loss.nll(logProbs, targets))
        expect(Math.abs(nllValue - expected)).toBeLessThan(TOL)
      }))

    it.effect("klDiv / hinge / cosineEmbeddingLoss", () =>
      Effect.gen(function*() {
        const logPred = yield* f32([Math.log(0.5), Math.log(0.5)])
        const target = yield* f32([0.25, 0.75])
        const [kl] = yield* values(yield* Loss.klDiv(logPred, target))
        const klExpected = (0.25 * (Math.log(0.25) - Math.log(0.5)) + 0.75 * (Math.log(0.75) - Math.log(0.5))) / 2
        expect(Math.abs(kl - klExpected)).toBeLessThan(TOL)

        const pred = yield* f32([0.9, -0.3])
        const signs = yield* f32([1, -1])
        const [hingeValue] = yield* values(yield* Loss.hinge(pred, signs))
        expect(Math.abs(hingeValue - (0.1 + 0.7) / 2)).toBeLessThan(TOL)

        const a = yield* f32([1, 0, 0, 1], [2, 2])
        const b = yield* f32([1, 0, 0, -1], [2, 2])
        const targets = yield* f32([1, -1])
        const [cos] = yield* values(yield* Loss.cosineEmbeddingLoss(a, b, targets))
        expect(Math.abs(cos - 0)).toBeLessThan(TOL)
      }))

    it.effect("reduction none returns the unreduced loss", () =>
      Effect.gen(function*() {
        const pred = yield* f32([1, 2, 4])
        const none = yield* Loss.mse(pred, yield* Tensor.constantLike(pred, 0), { reduction: "none" })
        assert.deepStrictEqual(none.shape, [3])
        assert.deepStrictEqual(yield* values(none), [1, 4, 16])
      }))

    it.effect("crossEntropy rejects mismatched targets", () =>
      Effect.gen(function*() {
        const logits = yield* f32([1, 2, 3], [1, 3])
        const bad = yield* f32([0])
        const error = yield* Effect.flip(Loss.crossEntropy(logits, bad))
        expect(error.message).toContain("i64")
      }))

    it.effect("chunked head crossEntropy matches the unchunked head", () =>
      Effect.gen(function*() {
        // Force the RFC 0016 phase-2 rewrite on tiny shapes with one row per
        // chunk. The fused linear + mean-CE node recomputes logits chunkwise in
        // backward; loss and gradients must match the unfused graph.
        // The switches are process-global compiler inputs, so bracket each run
        // and restore absent variables as absent rather than as string values.
        const withChunkEnv = <A, E, R>(min: string, size: string, effect: Effect.Effect<A, E, R>) =>
          Effect.acquireUseRelease(
            Effect.sync(() => {
              const prev = [process.env.EFFECT_TORCH_CE_CHUNK_MIN, process.env.EFFECT_TORCH_CE_CHUNK_SIZE]
              process.env.EFFECT_TORCH_CE_CHUNK_MIN = min
              process.env.EFFECT_TORCH_CE_CHUNK_SIZE = size
              return prev
            }),
            () => effect,
            (prev) =>
              Effect.sync(() => {
                const keys = ["EFFECT_TORCH_CE_CHUNK_MIN", "EFFECT_TORCH_CE_CHUNK_SIZE"]
                for (const [i, key] of keys.entries()) {
                  const value = prev[i]
                  if (value === undefined) delete process.env[key]
                  else process.env[key] = value
                }
              })
          )
        const run = (min: string, size: string, targets: Tensor.Any) =>
          withChunkEnv(
            min,
            size,
            Effect.gen(function*() {
              const x = yield* f32(Array.from({ length: 24 }, (_, i) => Math.sin(i * 0.37)), [2, 3, 4])
              const w = yield* f32(Array.from({ length: 32 }, (_, i) => Math.cos(i * 0.11) * 0.5), [4, 8])
              const b = yield* f32(Array.from({ length: 8 }, (_, i) => i * 0.05 - 0.2), [8])
              const logits = yield* Tensor.linear(x, w, b)
              const loss = yield* Tensor.crossEntropy(logits, { target: targets, ignoreIndex: -100 })
              const [value] = yield* values(loss)
              const grads = yield* Gradient.grad(loss, [x, w, b])
              return { value, grads: yield* Effect.all(grads.map((g) => values(g))) }
            })
          )
        for (
          const targets of [
            yield* i64([0n, 1n, 2n, 3n, 4n, 5n], [2, 3]),
            yield* i64([-100n, 1n, 2n, 3n, -100n, 5n], [2, 3])
          ]
        ) {
          const plain = yield* run("999999999999", "67108864", targets)
          const chunked = yield* run("0", "1", targets)
          expect(Math.abs(plain.value - chunked.value)).toBeLessThan(1e-5)
          for (let g = 0; g < 3; g++) {
            const a = plain.grads[g]!
            const c = chunked.grads[g]!
            expect(a.length).toBe(c.length)
            for (let i = 0; i < a.length; i++) {
              expect(Math.abs(a[i] - c[i])).toBeLessThan(1e-4)
            }
          }
        }
        const emptyError = yield* Effect.flip(
          run("0", "1", yield* i64([-100n, -100n, -100n, -100n, -100n, -100n], [2, 3]))
        )
        expect(emptyError.message).toContain("no active targets")
        const rangeError = yield* Effect.flip(
          run("0", "1", yield* i64([0n, 1n, 2n, 3n, 8n, 5n], [2, 3]))
        )
        expect(rangeError.message).toContain("out of range")
      }))
  })

  describe("gradcheck (finite differences)", () => {
    it.effect("regression losses", () =>
      Effect.gen(function*() {
        yield* gradcheck((x) => Effect.flatMap(Tensor.constantLike(x, 1), (t) => Loss.mse(x, t)), [0.5, -1, 2], [3])
        yield* gradcheck((x) => Effect.flatMap(Tensor.constantLike(x, 0.25), (t) => Loss.l1(x, t)), [0.5, -1, 2], [3])
        yield* gradcheck((x) => Effect.flatMap(Tensor.constantLike(x, 0), (t) => Loss.huber(x, t)), [0.5, -1, 2], [3])
        yield* gradcheck((x) => Effect.flatMap(Tensor.constantLike(x, 0), (t) => Loss.huber(x, t, { delta: 0.75 })), [
          0.5,
          -1,
          2
        ], [3])
      }))

    it.effect("classification losses", () =>
      Effect.gen(function*() {
        const y = yield* f32([1, 0, 1])
        yield* gradcheck((x) => Loss.binaryCrossEntropy(x, y), [0.8, 0.3, 0.6], [3])
        yield* gradcheck((x) => Loss.binaryCrossEntropy(x, y, { fromLogits: true }), [2, -1, 0.5], [3])
        const targets = yield* i64([0n, 1n])
        yield* gradcheck((x) => Loss.crossEntropy(x, targets), [2, 1, 0, 0, 3, 1], [2, 3])
        const logProbs = yield* Tensor.logSoftmax(yield* f32([2, 1, 0, 0, 3, 1], [2, 3]))
        yield* gradcheck((x) => Loss.nll(x, targets), yield* Tensor.toNumberArray(logProbs), [2, 3])
        const signs = yield* f32([1, -1, 1])
        yield* gradcheck((x) => Loss.hinge(x, signs), [0.9, -0.3, 0.2], [3])
        const probs = yield* f32([0.25, 0.75, 0.1])
        yield* gradcheck((x) => Loss.klDiv(x, probs), [Math.log(0.5), Math.log(0.5), Math.log(0.9)], [3])
      }))
  })
})

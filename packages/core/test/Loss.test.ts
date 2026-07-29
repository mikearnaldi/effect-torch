import { describe, expect } from "@effect/vitest"
import * as assert from "@effect/vitest/utils"
import { Effect } from "effect"
import { Device, Gradient, Loss, Tensor } from "../src/index.ts"
import { floats, onDevices, type TestDevice } from "./devices.ts"

type F64 = (data: ReadonlyArray<number>, shape?: ReadonlyArray<number>) => ReturnType<typeof Tensor.fromTypedArray>

const i64 = (data: ReadonlyArray<bigint>, shape?: ReadonlyArray<number>) =>
  Tensor.fromTypedArray(new BigInt64Array(data), shape)

const values = (t: Tensor.GenericTensor) => Tensor.toNumberArray(t)

const scalar = (t: Tensor.GenericTensor) => Effect.map(values(t), (v) => v[0])

const EPS = 1e-6
const TOL = 1e-4

const gradcheck = (
  f64: F64,
  numTol: number,
  f: (x: Tensor.LazyTensor) => Effect.Effect<Tensor.LazyTensor, Tensor.TensorError, Device.CurrentDevice>,
  input: ReadonlyArray<number>,
  shape: ReadonlyArray<number>
) =>
  Effect.gen(function* () {
    const x = yield* f64(input, shape)
    const [analytic] = yield* Gradient.grad(yield* f(x), [x])
    const analyticValues = yield* values(analytic)
    for (let i = 0; i < input.length; i++) {
      const plus = input.map((v, j) => (j === i ? v + EPS : v))
      const minus = input.map((v, j) => (j === i ? v - EPS : v))
      const fp = yield* scalar(yield* f(yield* f64(plus, shape)))
      const fm = yield* scalar(yield* f(yield* f64(minus, shape)))
      const numeric = (fp - fm) / (2 * EPS)
      expect(Math.abs(analyticValues[i] - numeric)).toBeLessThan(numTol)
    }
  })

onDevices("Loss", (device: TestDevice) => (it) => {
  const f64: F64 = (data, shape) => Tensor.fromTypedArray(floats(device, data), shape)
  // f32 on Metal needs looser numerical bounds than f64 on CPU
  const tol = device === "metal" ? 1e-4 : 1e-12
  const numTol = device === "metal" ? 5e-2 : TOL
  describe("values", () => {
    it.effect("mse / l1 / huber", () =>
      Effect.gen(function* () {
        const pred = yield* f64([1, 2, 4])
        const target = yield* f64([1, 0, 1])
        const [mseValue] = yield* values(yield* Loss.mse(pred, target))
        expect(Math.abs(mseValue - (0 + 4 + 9) / 3)).toBeLessThan(tol)
        const [l1Value] = yield* values(yield* Loss.l1(pred, target))
        expect(Math.abs(l1Value - (0 + 2 + 3) / 3)).toBeLessThan(tol)
        const [huberValue] = yield* values(yield* Loss.huber(pred, target))
        expect(Math.abs(huberValue - (0 + 1.5 + 2.5) / 3)).toBeLessThan(tol)
        const [huberSum] = yield* values(yield* Loss.huber(pred, target, { reduction: "sum" }))
        expect(Math.abs(huberSum - 4)).toBeLessThan(tol)
      })
    )

    it.effect("binaryCrossEntropy from probabilities and logits", () =>
      Effect.gen(function* () {
        const p = yield* f64([0.8, 0.3])
        const y = yield* f64([1, 0])
        const [bce] = yield* values(yield* Loss.binaryCrossEntropy(p, y))
        const expected = -(Math.log(0.8) + Math.log(0.7)) / 2
        expect(Math.abs(bce - expected)).toBeLessThan(tol)
        const logits = yield* f64([2, -1])
        const [fromLogits] = yield* values(yield* Loss.binaryCrossEntropy(logits, y, { fromLogits: true }))
        const stable = (x: number, t: number) => Math.max(x, 0) - x * t + Math.log1p(Math.exp(-Math.abs(x)))
        expect(Math.abs(fromLogits - (stable(2, 1) + stable(-1, 0)) / 2)).toBeLessThan(tol)
      })
    )

    it.effect("crossEntropy and nll", () =>
      Effect.gen(function* () {
        const logits = yield* f64([2, 1, 0, 0, 3, 1], [2, 3])
        const targets = yield* i64([0n, 1n])
        const [ce] = yield* values(yield* Loss.crossEntropy(logits, targets))
        const lsm = (row: Array<number>, cls: number) => {
          const m = Math.max(...row)
          return -(row[cls] - m - Math.log(row.reduce((a, v) => a + Math.exp(v - m), 0)))
        }
        const expected = (lsm([2, 1, 0], 0) + lsm([0, 3, 1], 1)) / 2
        expect(Math.abs(ce - expected)).toBeLessThan(tol)
        const logProbs = yield* Tensor.logSoftmax(logits)
        const [nllValue] = yield* values(yield* Loss.nll(logProbs, targets))
        expect(Math.abs(nllValue - expected)).toBeLessThan(tol)
      })
    )

    it.effect("klDiv / hinge / cosineEmbeddingLoss", () =>
      Effect.gen(function* () {
        const logPred = yield* f64([Math.log(0.5), Math.log(0.5)])
        const target = yield* f64([0.25, 0.75])
        const [kl] = yield* values(yield* Loss.klDiv(logPred, target))
        const klExpected = (0.25 * (Math.log(0.25) - Math.log(0.5)) + 0.75 * (Math.log(0.75) - Math.log(0.5))) / 2
        expect(Math.abs(kl - klExpected)).toBeLessThan(tol)

        const pred = yield* f64([0.9, -0.3])
        const signs = yield* f64([1, -1])
        const [hingeValue] = yield* values(yield* Loss.hinge(pred, signs))
        expect(Math.abs(hingeValue - (0.1 + 0.7) / 2)).toBeLessThan(tol)

        const a = yield* f64([1, 0, 0, 1], [2, 2])
        const b = yield* f64([1, 0, 0, -1], [2, 2])
        const targets = yield* f64([1, -1])
        const [cos] = yield* values(yield* Loss.cosineEmbeddingLoss(a, b, targets))
        expect(Math.abs(cos - 0)).toBeLessThan(tol)
      })
    )

    it.effect("reduction none returns the unreduced loss", () =>
      Effect.gen(function* () {
        const pred = yield* f64([1, 2, 4])
        const none = yield* Loss.mse(pred, 0, { reduction: "none" })
        assert.deepStrictEqual(none.shape, [3])
        assert.deepStrictEqual(yield* values(none), [1, 4, 16])
      })
    )

    it.effect("crossEntropy rejects mismatched targets", () =>
      Effect.gen(function* () {
        const logits = yield* f64([1, 2, 3], [1, 3])
        const bad = yield* f64([0])
        const error = yield* Effect.flip(Loss.crossEntropy(logits, bad))
        expect(error.message).toContain("i64")
      })
    )
  })

  // gradcheck needs f64: finite differences with EPS = 1e-6 drown in f32
  // rounding. It validates the autodiff math, not device kernels, so a
  // single CPU run suffices (device kernel parity is covered by the
  // Fusion A/B tests).
  if (device === "cpu") describe("gradcheck (finite differences)", () => {
    it.effect("regression losses", () =>
      Effect.gen(function* () {
        yield* gradcheck(f64, numTol, (x) => Loss.mse(x, 1), [0.5, -1, 2], [3])
        yield* gradcheck(f64, numTol, (x) => Loss.l1(x, 0.25), [0.5, -1, 2], [3])
        yield* gradcheck(f64, numTol, (x) => Loss.huber(x, 0), [0.5, -1, 2], [3])
        yield* gradcheck(f64, numTol, (x) => Loss.huber(x, 0, { delta: 0.75 }), [0.5, -1, 2], [3])
      })
    )

    it.effect("classification losses", () =>
      Effect.gen(function* () {
        const y = yield* f64([1, 0, 1])
        yield* gradcheck(f64, numTol, (x) => Loss.binaryCrossEntropy(x, y), [0.8, 0.3, 0.6], [3])
        yield* gradcheck(f64, numTol, (x) => Loss.binaryCrossEntropy(x, y, { fromLogits: true }), [2, -1, 0.5], [3])
        const targets = yield* i64([0n, 1n])
        yield* gradcheck(f64, numTol, (x) => Loss.crossEntropy(x, targets), [2, 1, 0, 0, 3, 1], [2, 3])
        const logProbs = yield* Tensor.logSoftmax(yield* f64([2, 1, 0, 0, 3, 1], [2, 3]))
        yield* gradcheck(f64, numTol, (x) => Loss.nll(x, targets), yield* Tensor.toNumberArray(logProbs), [2, 3])
        const signs = yield* f64([1, -1, 1])
        yield* gradcheck(f64, numTol, (x) => Loss.hinge(x, signs), [0.9, -0.3, 0.2], [3])
        const probs = yield* f64([0.25, 0.75, 0.1])
        yield* gradcheck(f64, numTol, (x) => Loss.klDiv(x, probs), [Math.log(0.5), Math.log(0.5), Math.log(0.9)], [3])
      })
    )
  })
})

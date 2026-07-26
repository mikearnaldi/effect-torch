import { describe, expect, layer } from "@effect/vitest"
import * as assert from "@effect/vitest/utils"
import { Effect, Exit } from "effect"
import { Device, Tensor } from "../src/index.js"

const f64 = (data: ReadonlyArray<number>, shape?: ReadonlyArray<number>) =>
  Tensor.fromTypedArray(new Float64Array(data), shape)

const values = (t: Tensor.GenericTensor) =>
  Effect.map(Tensor.toTypedArray(t), (arr) => Array.from<number | bigint>(arr).map(Number))

const scalar = (t: Tensor.GenericTensor) => Effect.map(values(t), (v) => v[0])

type ScalarFn = (x: Tensor.LazyTensor) => Effect.Effect<Tensor.LazyTensor, Tensor.TensorError>

const sumOf = (
  op: (x: Tensor.LazyTensor) => Effect.Effect<Tensor.LazyTensor, Tensor.TensorError>
): ScalarFn => (x) => Effect.flatMap(op(x), (t) => Tensor.sum(t))

const EPS = 1e-6
const TOL = 1e-4

const gradcheck = (f: ScalarFn, input: ReadonlyArray<number>, shape: ReadonlyArray<number>) =>
  Effect.gen(function* () {
    const x = yield* f64(input, shape)
    const [analytic] = yield* Tensor.grad(yield* f(x), [x])
    const analyticValues = yield* values(analytic)
    for (let i = 0; i < input.length; i++) {
      const plus = input.map((v, j) => (j === i ? v + EPS : v))
      const minus = input.map((v, j) => (j === i ? v - EPS : v))
      const fp = yield* scalar(yield* f(yield* f64(plus, shape)))
      const fm = yield* scalar(yield* f(yield* f64(minus, shape)))
      const numeric = (fp - fm) / (2 * EPS)
      expect(Math.abs(analyticValues[i] - numeric)).toBeLessThan(TOL)
    }
  })

layer(Device.Cpu)("Autodiff", (it) => {
  describe("gradcheck (finite differences)", () => {
    it.effect("elementwise add/mul/div with broadcasting", () =>
      Effect.gen(function* () {
        const b = yield* f64([1, 2, 3], [1, 3])
        const input = [1, 2, 3, 4, 5, 6]
        yield* gradcheck(sumOf((x) => Tensor.add(x, b)), input, [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.mul(x, b)), input, [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.div(x, b)), input, [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.div(b, x)), input, [2, 3])
      })
    )

    it.effect("unary ops", () =>
      Effect.gen(function* () {
        yield* gradcheck(sumOf((x) => Tensor.neg(x)), [1, -2, 3], [3])
        yield* gradcheck(sumOf((x) => Tensor.abs(x)), [1, -2, 3], [3])
        yield* gradcheck(sumOf((x) => Tensor.sqrt(x)), [1, 2, 3], [3])
        yield* gradcheck(sumOf((x) => Tensor.exp(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.log(x)), [1, 2, 3], [3])
        yield* gradcheck(sumOf((x) => Tensor.sin(x)), [0.5, 1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.cos(x)), [0.5, 1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.pow(x, 3)), [0.5, 1, 2], [3])
      })
    )

    it.effect("matmul", () =>
      Effect.gen(function* () {
        const b = yield* f64([1, 2, 3, 4, 5, 6], [3, 2])
        yield* gradcheck(sumOf((x) => Tensor.matmul(x, b)), [1, 2, 3, 4, 5, 6], [2, 3])
        const a = yield* f64([1, 2, 3, 4, 5, 6], [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.matmul(a, x)), [1, 2, 3, 4, 5, 6], [3, 2])
      })
    )

    it.effect("batched matmul with broadcast batch dims", () =>
      Effect.gen(function* () {
        const b = yield* f64([1, 2, 3, 4, 5, 6], [3, 2])
        const input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        yield* gradcheck(sumOf((x) => Tensor.matmul(x, b)), input, [2, 2, 3])
        const c = yield* f64(input, [2, 2, 3])
        yield* gradcheck(sumOf((x) => Tensor.matmul(c, x)), [1, 2, 3, 4, 5, 6], [3, 2])
      })
    )

    it.effect("reductions sum/mean/max/min", () =>
      Effect.gen(function* () {
        const input = [1, 5, 3, 4, 2, 6]
        yield* gradcheck((x) => Tensor.sum(x), input, [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.sum(x, { dims: [0] })), input, [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.mean(x, { dims: [1], keepdims: true })), input, [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.max(x, { dims: [1] })), input, [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.min(x, { dims: [0], keepdims: true })), input, [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.mean(x, { dims: [0] })), input, [2, 3])
      })
    )

    it.effect("max/min split gradients evenly across ties", () =>
      Effect.gen(function* () {
        const x = yield* f64([1, 3, 3, 2], [4])
        const [gmax] = yield* Tensor.grad(yield* Tensor.max(x, { dims: [0] }), [x])
        assert.deepStrictEqual(yield* values(gmax), [0, 0.5, 0.5, 0])
        const [gmin] = yield* Tensor.grad(yield* Tensor.min(x, { dims: [0] }), [x])
        assert.deepStrictEqual(yield* values(gmin), [1, 0, 0, 0])
        const y = yield* f64([2, 1, 1, 3], [4])
        const [gymin] = yield* Tensor.grad(yield* Tensor.min(y, { dims: [0] }), [y])
        assert.deepStrictEqual(yield* values(gymin), [0, 0.5, 0.5, 0])
      })
    )

    it.effect("shape ops", () =>
      Effect.gen(function* () {
        const input = [1, 2, 3, 4, 5, 6]
        yield* gradcheck(sumOf((x) => Tensor.reshape(x, [3, 2])), input, [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.transpose(x, [1, 0])), input, [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.slice(x, { start: [0, 1], end: [2, 3] })), input, [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.broadcastTo(x, [2, 3])), [1, 2, 3], [1, 3])
        yield* gradcheck(sumOf((x) => Tensor.broadcastTo(x, [2, 3])), [1, 2, 3], [3])
        yield* gradcheck(sumOf((x) => Tensor.transpose(x, [2, 0, 1])), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2])
        const c = yield* f64([7, 8, 9], [1, 3])
        yield* gradcheck(sumOf((x) => Tensor.concat([x, c], { dim: 0 })), [1, 2, 3], [1, 3])
      })
    )

    it.effect("cast roundtrip through f32 has identity gradient", () =>
      Effect.gen(function* () {
        const x = yield* f64([1, 2, 3])
        const loss = yield* Effect.flatMap(Tensor.cast(x, "f32"), (t) => Tensor.cast(t, "f64"))
        const [gx] = yield* Tensor.grad(yield* Tensor.sum(loss), [x])
        assert.deepStrictEqual(yield* values(gx), [1, 1, 1])
      })
    )

    it.effect("composite graph with shared subexpressions", () =>
      Effect.gen(function* () {
        yield* gradcheck(
          (x) =>
            Effect.gen(function* () {
              const y = yield* Tensor.mul(x, x)
              const z = yield* Tensor.add(y, x)
              const out = yield* Tensor.mul(y, z)
              return yield* Tensor.sum(out)
            }),
          [0.5, 1.5, 2.5],
          [3]
        )
      })
    )
  })

  describe("contract", () => {
    it.effect("rejects non-scalar output", () =>
      Effect.gen(function* () {
        const x = yield* f64([1, 2, 3])
        const exit = yield* Effect.exit(Tensor.grad(x, [x]))
        assert.assertTrue(Exit.isFailure(exit))
      })
    )

    it.effect("rejects non-float wrt", () =>
      Effect.gen(function* () {
        const x = yield* Tensor.fromTypedArray(new BigInt64Array([1n, 2n]))
        const loss = yield* Effect.flatMap(Tensor.cast(x, "f64"), (t) => Tensor.sum(t))
        const exit = yield* Effect.exit(Tensor.grad(loss, [x]))
        assert.assertTrue(Exit.isFailure(exit))
      })
    )

    it.effect("rejects slice with stride > 1", () =>
      Effect.gen(function* () {
        const x = yield* f64([1, 2, 3, 4], [4])
        const sliced = yield* Tensor.slice(x, { stride: [2] })
        const loss = yield* Tensor.sum(sliced)
        const error = yield* Effect.flip(Tensor.grad(loss, [x]))
        assert.strictEqual(error.reason, "not-differentiable")
      })
    )

    it.effect("unused wrt argument yields zeros", () =>
      Effect.gen(function* () {
        const x = yield* f64([1, 2, 3])
        const y = yield* f64([4, 5, 6])
        const loss = yield* Effect.flatMap(Tensor.mul(x, x), (t) => Tensor.sum(t))
        const [gx, gy] = yield* Tensor.grad(loss, [x, y])
        assert.deepStrictEqual(yield* values(gx), [2, 4, 6])
        assert.deepStrictEqual(yield* values(gy), [0, 0, 0])
      })
    )

    it.effect("stopGradient blocks gradient flow", () =>
      Effect.gen(function* () {
        const x = yield* f64([2, 3])
        const stopped = yield* Tensor.stopGradient(x)
        const loss = yield* Tensor.sum(yield* Tensor.mul(stopped, x))
        const [gx] = yield* Tensor.grad(loss, [x])
        assert.deepStrictEqual(yield* values(gx), [2, 3])
      })
    )
  })

  describe("grad + evaluateAll", () => {
    it.effect("loss and gradients come from the same randn draw", () =>
      Effect.gen(function* () {
        const x = yield* f64([1, 2, 3])
        const r = yield* Tensor.randn([3], { dtype: "f64" })
        const loss = yield* Tensor.sum(yield* Tensor.mul(x, r))
        const [gx] = yield* Tensor.grad(loss, [x])
        const [l, g] = yield* Tensor.evaluateAll([loss, gx])
        const lv = yield* scalar(l)
        const gv = yield* values(g)
        const reconstructed = gv[0] * 1 + gv[1] * 2 + gv[2] * 3
        expect(Math.abs(lv - reconstructed)).toBeLessThan(1e-9)
      })
    )
  })

  describe("end-to-end", () => {
    it.effect("linear regression converges", () =>
      Effect.gen(function* () {
        const trueW = [2, -3]
        const trueB = 1
        const xs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        const ys = xs.map((x) => trueW[0] * x + trueW[1] + trueB)
        const features = xs.map((x) => [x, 1]).flat()
        const x = yield* f64(features, [8, 2])
        const y = yield* f64(ys, [8, 1])
        let w = yield* f64([0, 0], [2, 1])
        let b = yield* f64([0], [1, 1])
        const lr = 0.05
        const losses: Array<number> = []
        for (let step = 0; step < 200; step++) {
          const pred = yield* Tensor.add(yield* Tensor.matmul(x, w), b)
          const err = yield* Tensor.sub(pred, y)
          const loss = yield* Tensor.mean(yield* Tensor.mul(err, err))
          const [gw, gb] = yield* Tensor.grad(loss, [w, b])
          const [lt, gwt, gbt] = yield* Tensor.evaluateAll([loss, gw, gb])
          const l = yield* scalar(lt)
          const gwv = yield* values(gwt)
          const gbv = yield* values(gbt)
          losses.push(l)
          w = yield* f64((yield* values(w)).map((v, i) => v - lr * gwv[i]), [2, 1])
          b = yield* f64([(yield* values(b))[0] - lr * gbv[0]], [1, 1])
        }
        expect(losses[losses.length - 1]).toBeLessThan(losses[0] * 1e-3)
        const finalW = yield* values(w)
        const finalB = yield* values(b)
        expect(Math.abs(finalW[0] - trueW[0])).toBeLessThan(0.1)
        // w[1] and b are collinear (both multiply the constant feature 1),
        // only their sum is identified by the data
        expect(Math.abs(finalW[1] + finalB[0] - (trueW[1] + trueB))).toBeLessThan(0.1)
      })
    )
  })
})

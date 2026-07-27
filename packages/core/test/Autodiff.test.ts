import { describe, expect, layer } from "@effect/vitest"
import * as assert from "@effect/vitest/utils"
import { Effect, Exit } from "effect"
import { Device, Loss, Tensor } from "../src/index.ts"

const f64 = (data: ReadonlyArray<number>, shape?: ReadonlyArray<number>) =>
  Tensor.fromTypedArray(new Float64Array(data), shape)

const values = (t: Tensor.GenericTensor) => Tensor.toNumberArray(t)

const scalar = (t: Tensor.GenericTensor) => Effect.map(values(t), (v) => v[0])

type ScalarFn = (
  x: Tensor.LazyTensor
) => Effect.Effect<Tensor.LazyTensor, Tensor.TensorError, Device.CurrentDevice>

const sumOf = (
  op: (x: Tensor.LazyTensor) => Effect.Effect<Tensor.LazyTensor, Tensor.TensorError, Device.CurrentDevice>
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
        yield* gradcheck(sumOf((x) => Tensor.tanh(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.relu(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.maximum(x, 0.25)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.minimum(x, 0.25)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.sigmoid(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Loss.mse(x, 1)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.pow(x, 3)), [0.5, 1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.erf(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.floor(x)), [0.5, -1.3, 2.7], [3])
        yield* gradcheck(sumOf((x) => Tensor.ceil(x)), [0.5, -1.3, 2.7], [3])
        yield* gradcheck(sumOf((x) => Tensor.round(x)), [0.6, -1.3, 2.7], [3])
        yield* gradcheck(sumOf((x) => Tensor.sign(x)), [0.5, -1.3, 2.7], [3])
        yield* gradcheck(sumOf((x) => Tensor.square(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.rsqrt(x)), [0.5, 1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.reciprocal(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.expm1(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.log1p(x)), [0.5, 1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.log2(x)), [0.5, 1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.log10(x)), [0.5, 1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.sinh(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.cosh(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.tan(x)), [0.5, -1, 1.2], [3])
        yield* gradcheck(sumOf((x) => Tensor.remainder(x, 3)), [0.5, -1.3, 2.7], [3])
      })
    )

    it.effect("neural network ops", () =>
      Effect.gen(function* () {
        yield* gradcheck(sumOf((x) => Tensor.silu(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.softplus(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.elu(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.leakyRelu(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.gelu(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.gelu(x, { approximate: "tanh" })), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.mish(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.clamp(x, { min: -0.5, max: 1.5 })), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.hardtanh(x)), [0.5, -2, 0.3], [3])
        const w = yield* f64([0.5, -1, 2])
        yield* gradcheck(sumOf((x) => Effect.flatMap(Tensor.softmax(x), (s) => Tensor.mul(s, w))), [0.5, -1, 2], [3])
        yield* gradcheck(
          sumOf((x) => Effect.flatMap(Tensor.logSoftmax(x), (s) => Tensor.mul(s, w))),
          [0.5, -1, 2],
          [3]
        )
      })
    )

    it.effect("extended reductions and where", () =>
      Effect.gen(function* () {
        yield* gradcheck(sumOf((x) => Tensor.variance(x)), [1, 2, 3, 4], [4])
        yield* gradcheck(sumOf((x) => Tensor.std(x)), [1, 2, 3, 4], [4])
        yield* gradcheck(sumOf((x) => Tensor.norm(x, { ord: 2 })), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.norm(x, { ord: 3 })), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.logsumexp(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.prod(x)), [0.5, 1.5, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.cumsum(x, 0)), [0.5, -1, 2], [3])
        const cond = yield* Tensor.fromTypedArray(new Uint8Array([1, 0, 1]))
        const other = yield* f64([10, 20, 30])
        yield* gradcheck(sumOf((x) => Tensor.where(cond, x, other)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.where(cond, other, x)), [0.5, -1, 2], [3])
      })
    )

    it.effect("shape ops", () =>
      Effect.gen(function* () {
        yield* gradcheck(sumOf((x) => Tensor.flatten(x, { startDim: 1 })), [1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])
        yield* gradcheck(sumOf((x) => Tensor.tile(x, [2, 2])), [1, 2, 3, 4], [2, 2])
        yield* gradcheck(sumOf((x) => Tensor.pad(x, [[1, 1], [0, 2]])), [1, 2, 3, 4], [2, 2])
        yield* gradcheck(sumOf((x) => Tensor.triu(x)), [1, 2, 3, 4], [2, 2])
        yield* gradcheck(sumOf((x) => Tensor.tril(x)), [1, 2, 3, 4], [2, 2])
        yield* gradcheck(sumOf((x) => Tensor.trace(x)), [1, 2, 3, 4], [2, 2])
        const b = yield* f64([4, 5, 6])
        yield* gradcheck(sumOf((x) => Tensor.dot(x, b)), [1, 2, 3], [3])
        const parts = yield* Tensor.split(yield* f64([1, 2, 3, 4]), 2)
        assert.strictEqual(parts.length, 2)
      })
    )

    it.effect("take and gather scatter gradients back", () =>
      Effect.gen(function* () {
        const x = yield* f64([1, 2, 3, 4, 5, 6], [3, 2])
        const idx = yield* Tensor.fromTypedArray(new BigInt64Array([2n, 0n, 2n]))
        const loss = yield* Tensor.sum(yield* Tensor.take(x, idx))
        const [g] = yield* Tensor.grad(loss, [x])
        assert.deepStrictEqual(yield* values(g), [1, 1, 0, 0, 2, 2])

        const idx2 = yield* Tensor.fromTypedArray(new BigInt64Array([1n, 0n, 0n, 1n]), [2, 2])
        const g2 = yield* Tensor.sum(yield* Tensor.gather(x, idx2, { dim: 1 }))
        const [dg] = yield* Tensor.grad(g2, [x])
        assert.deepStrictEqual(yield* values(dg), [1, 1, 1, 1, 0, 0])
      })
    )

    it.effect("scatterAdd", () =>
      Effect.gen(function* () {
        yield* gradcheck(
          sumOf((x) => Effect.gen(function* () {
            const idx = yield* Tensor.fromTypedArray(new BigInt64Array([2n, 0n, 2n]))
            return yield* Tensor.take(x, idx)
          })),
          [1, 2, 3, 4, 5, 6],
          [3, 2]
        )
        const base = yield* f64([1, 1, 1, 1, 1, 1], [3, 2])
        const idx = yield* Tensor.fromTypedArray(new BigInt64Array([0n, 2n, 2n, 0n]), [2, 2])
        yield* gradcheck(
          sumOf((src) => Tensor.scatterAdd(base, idx, src)),
          [1, 2, 3, 4],
          [2, 2]
        )
      })
    )

    it.effect("strided slice", () =>
      Effect.gen(function* () {
        yield* gradcheck(
          sumOf((x) => Tensor.slice(x, { start: [1], end: [7], stride: [2] })),
          [1, 2, 3, 4, 5, 6, 7, 8],
          [8]
        )
      })
    )

    it.effect("convolution and pooling", () =>
      Effect.gen(function* () {
        const w = yield* f64([1, 0, 0, 1], [1, 1, 2, 2])
        yield* gradcheck(sumOf((x) => Tensor.conv2d(x, w)), [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 1, 3, 3])
        const x = yield* f64([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 1, 3, 3])
        yield* gradcheck(sumOf((w2) => Tensor.conv2d(x, w2)), [1, 2, 3, 4], [1, 1, 2, 2])
        const w2s = yield* f64([1, 0, 0, 1], [1, 1, 2, 2])
        yield* gradcheck(
          sumOf((xs) => Tensor.conv2d(xs, w2s, { stride: 2, padding: 1 })),
          [1, 2, 3, 4, 5, 6, 7, 8, 9],
          [1, 1, 3, 3]
        )
        const wg = yield* f64([1, 0, 0, 1, 0, 1, 1, 0], [2, 1, 2, 2])
        yield* gradcheck(
          sumOf((xg) => Tensor.conv2d(xg, wg, { groups: 2 })),
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
          [1, 2, 3, 3]
        )
        const w1 = yield* f64([1, 1], [1, 1, 2])
        yield* gradcheck(sumOf((x1) => Tensor.conv1d(x1, w1)), [1, 2, 3, 4], [1, 1, 4])
        yield* gradcheck(
          sumOf((xp) => Tensor.maxPool2d(xp, { kernelSize: 2, stride: 1 })),
          [1, 3, 2, 4, 5, 7, 6, 8, 9],
          [1, 1, 3, 3]
        )
        yield* gradcheck(
          sumOf((xp) => Tensor.avgPool2d(xp, { kernelSize: 2, stride: 1 })),
          [1, 3, 2, 4, 5, 7, 6, 8, 9],
          [1, 1, 3, 3]
        )
        const wt = yield* f64([1, 0, 0, 1], [1, 1, 2, 2])
        yield* gradcheck(sumOf((xt) => Tensor.convTranspose2d(xt, wt)), [1, 2, 3, 4], [1, 1, 2, 2])
        yield* gradcheck(
          sumOf((xt) => Tensor.convTranspose2d(xt, wt, { stride: 2 })),
          [1, 2, 3, 4],
          [1, 1, 2, 2]
        )
        const xt = yield* f64([1, 2, 3, 4], [1, 1, 2, 2])
        yield* gradcheck(sumOf((w2) => Tensor.convTranspose2d(xt, w2)), [1, 2, 3, 4], [1, 1, 2, 2])
      })
    )

    it.effect("linalg", () =>
      Effect.gen(function* () {
        yield* gradcheck(sumOf((x) => Tensor.det(x)), [4, 1, 1, 3], [2, 2])
        yield* gradcheck(sumOf((x) => Tensor.inverse(x)), [4, 1, 1, 3], [2, 2])
        const b = yield* f64([9, 8], [2, 1])
        yield* gradcheck(sumOf((x) => Tensor.solve(x, b)), [4, 1, 1, 3], [2, 2])
        const a = yield* f64([4, 1, 1, 3], [2, 2])
        yield* gradcheck(sumOf((x) => Tensor.solve(a, x)), [9, 8], [2, 1])
      })
    )

    it.effect("checkpoint preserves values and gradients, sharing randn draws", () =>
      Effect.gen(function* () {
        const f = (x: Tensor.GenericTensor) =>
          Effect.gen(function* () {
            return yield* Tensor.mul(yield* Tensor.sin(x), yield* Tensor.add(x, 1))
          })
        const x = yield* f64([0.5, 1])
        const plain = yield* f(x)
        const wrapped = yield* Tensor.checkpoint(yield* f(x))
        const plainLoss = yield* Tensor.sum(plain)
        const wrappedLoss = yield* Tensor.sum(wrapped)
        const [plainGrad] = yield* Tensor.grad(plainLoss, [x])
        const [wrappedGrad] = yield* Tensor.grad(wrappedLoss, [x])
        assert.deepStrictEqual(yield* values(wrappedLoss), yield* values(plainLoss))
        const pg = yield* values(plainGrad)
        const wg = yield* values(wrappedGrad)
        for (let i = 0; i < 2; i++) {
          expect(Math.abs(pg[i] - wg[i])).toBeLessThan(1e-12)
        }

        // the backward recompute must see the same randn draw as the forward
        const stochastic = (x: Tensor.GenericTensor) =>
          Effect.gen(function* () {
            return yield* Tensor.mul(x, yield* Tensor.randn([2], { dtype: "f64" }))
          })
        const x2 = yield* f64([2, 4])
        const out2 = yield* Tensor.checkpoint(yield* stochastic(x2))
        const loss2 = yield* Tensor.sum(out2)
        const [g2] = yield* Tensor.grad(loss2, [x2])
        const [outM, gM] = yield* Tensor.evaluate([out2, g2])
        const outValues = yield* Tensor.toNumberArray(outM)
        const gradValues = yield* Tensor.toNumberArray(gM)
        for (let i = 0; i < 2; i++) {
          expect(Math.abs(outValues[i] / [2, 4][i] - gradValues[i])).toBeLessThan(1e-12)
        }
      })
    )

    it.effect("vjp / jvp / vmap", () =>
      Effect.gen(function* () {
        const a = yield* f64([1, 2, 3, 4, 5, 6], [2, 3])
        const f = (x: Tensor.GenericTensor) => Tensor.matmul(a, x)
        const x = yield* f64([1, 1, 1], [3, 1])
        const v = yield* f64([1, 2], [2, 1])
        const { output, pullback } = yield* Tensor.vjp(f, x, v)
        assert.deepStrictEqual(yield* values(output), [6, 15])
        // J^T v = A^T v
        assert.deepStrictEqual(yield* values(pullback), [1 * 1 + 4 * 2, 2 * 1 + 5 * 2, 3 * 1 + 6 * 2])

        const t = yield* f64([1, 0, 0], [3, 1])
        const { tangent } = yield* Tensor.jvp(f, x, t)
        assert.deepStrictEqual(yield* values(tangent), [1, 4])

        const nonlinear = (x: Tensor.GenericTensor) => Tensor.sin(x)
        const xn = yield* f64([0.5, 1])
        const vn = yield* f64([2, 3])
        const { tangent: tn } = yield* Tensor.jvp(nonlinear, xn, vn)
        const tnValues = yield* values(tn)
        expect(Math.abs(tnValues[0] - Math.cos(0.5) * 2)).toBeLessThan(1e-12)
        expect(Math.abs(tnValues[1] - Math.cos(1) * 3)).toBeLessThan(1e-12)

        const m = yield* f64([1, 2, 3, 4, 5, 6], [2, 3])
        const rowSums = yield* Tensor.vmap((row) => Tensor.sum(row))(m)
        assert.deepStrictEqual(yield* values(rowSums), [6, 15])
        const mapped = yield* Tensor.vmap((row) => Tensor.relu(row))(m)
        assert.deepStrictEqual(mapped.shape, [2, 3])
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

    it.effect("strided slice is differentiable", () =>
      Effect.gen(function* () {
        const x = yield* f64([1, 2, 3, 4, 5, 6, 7], [7])
        const sliced = yield* Tensor.slice(x, { stride: [2] })
        const loss = yield* Tensor.sum(sliced)
        const [g] = yield* Tensor.grad(loss, [x])
        assert.deepStrictEqual(yield* values(g), [1, 0, 1, 0, 1, 0, 1])
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

  describe("grad + evaluate", () => {
    it.effect("loss and gradients come from the same randn draw", () =>
      Effect.gen(function* () {
        const x = yield* f64([1, 2, 3])
        const r = yield* Tensor.randn([3], { dtype: "f64" })
        const loss = yield* Tensor.sum(yield* Tensor.mul(x, r))
        const [gx] = yield* Tensor.grad(loss, [x])
        const [l, g] = yield* Tensor.evaluate([loss, gx])
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
          const loss = yield* Loss.mse(pred, y)
          const [gw, gb] = yield* Tensor.grad(loss, [w, b])
          const [lt, gwt, gbt] = yield* Tensor.evaluate([loss, gw, gb])
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

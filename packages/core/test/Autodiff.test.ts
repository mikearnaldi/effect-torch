import { describe, expect } from "@effect/vitest"
import * as assert from "@effect/vitest/utils"
import { Effect, Exit } from "effect"
import { Gradient, Loss, type Runtime, Tensor } from "../src/index.ts"
import { deep, floatDtype, floats, GRADCHECK_EPS, GRADCHECK_TOL, onDevices, TOL } from "./utils/devices.ts"

const values = (t: Tensor.Any) => Tensor.toNumberArray(t)

const scalar = (t: Tensor.Any) => Effect.map(values(t), (v) => v[0])

type ScalarFn = (
  x: Tensor.Lazy
) => Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime>

const sumOf = (
  op: (x: Tensor.Lazy) => Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime>
): ScalarFn =>
(x) => Effect.flatMap(op(x), (t) => Tensor.sum(t))

// Central differences run in f32 on each real backend. The shared epsilon
// clears f32 rounding while the tolerance covers rounding and O(eps^2) error.
const gradcheck = (f: ScalarFn, input: ReadonlyArray<number>, shape: ReadonlyArray<number>) =>
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

onDevices("Autodiff", (device) => (it) => {
  const f32 = (data: ReadonlyArray<number>, shape?: ReadonlyArray<number>) => Tensor.fromTypedArray(floats(data), shape)
  describe("gradcheck (finite differences)", () => {
    it.effect("elementwise add/mul/div with broadcasting", () =>
      Effect.gen(function*() {
        const b = yield* f32([1, 2, 3], [1, 3])
        const input = [1, 2, 3, 4, 5, 6]
        yield* gradcheck(sumOf((x) => Tensor.add(x, b)), input, [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.mul(x, b)), input, [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.div(x, b)), input, [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.div(b, x)), input, [2, 3])
      }))

    it.effect("unary ops", () =>
      Effect.gen(function*() {
        yield* gradcheck(sumOf((x) => Tensor.neg(x)), [1, -2, 3], [3])
        yield* gradcheck(sumOf((x) => Tensor.abs(x)), [1, -2, 3], [3])
        yield* gradcheck(sumOf((x) => Tensor.sqrt(x)), [1, 2, 3], [3])
        yield* gradcheck(sumOf((x) => Tensor.exp(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.log(x)), [1, 2, 3], [3])
        yield* gradcheck(sumOf((x) => Tensor.sin(x)), [0.5, 1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.cos(x)), [0.5, 1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.tanh(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.relu(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Effect.flatMap(Tensor.constantLike(x, 0.25), (c) => Tensor.maximum(x, c))), [
          0.5,
          -1,
          2
        ], [3])
        yield* gradcheck(sumOf((x) => Effect.flatMap(Tensor.constantLike(x, 0.25), (c) => Tensor.minimum(x, c))), [
          0.5,
          -1,
          2
        ], [3])
        yield* gradcheck(sumOf((x) => Tensor.sigmoid(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Effect.flatMap(Tensor.constantLike(x, 1), (t) => Loss.mse(x, t))), [0.5, -1, 2], [
          3
        ])
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
        yield* gradcheck(sumOf((x) => Effect.flatMap(Tensor.constantLike(x, 3), (c) => Tensor.remainder(x, c))), [
          0.5,
          -1.3,
          2.7
        ], [3])
      }))

    it.effect("neural network ops", () =>
      Effect.gen(function*() {
        yield* gradcheck(sumOf((x) => Tensor.silu(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.softplus(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.elu(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.leakyRelu(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.gelu(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.gelu(x, { approximate: "tanh" })), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.mish(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.clamp(x, { min: -0.5, max: 1.5 })), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.hardtanh(x)), [0.5, -2, 0.3], [3])
        const w = yield* f32([0.5, -1, 2])
        yield* gradcheck(sumOf((x) => Effect.flatMap(Tensor.softmax(x), (s) => Tensor.mul(s, w))), [0.5, -1, 2], [3])
        yield* gradcheck(
          sumOf((x) => Effect.flatMap(Tensor.logSoftmax(x), (s) => Tensor.mul(s, w))),
          [0.5, -1, 2],
          [3]
        )
      }))

    it.effect("extended reductions and where", () =>
      Effect.gen(function*() {
        yield* gradcheck(sumOf((x) => Tensor.variance(x)), [1, 2, 3, 4], [4])
        yield* gradcheck(sumOf((x) => Tensor.std(x)), [1, 2, 3, 4], [4])
        yield* gradcheck(sumOf((x) => Tensor.norm(x, { ord: 2 })), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.norm(x, { ord: 3 })), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.logsumexp(x)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.prod(x)), [0.5, 1.5, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.cumsum(x, 0)), [0.5, -1, 2], [3])
        const cond = yield* Tensor.fromTypedArray(new Uint8Array([1, 0, 1]))
        const other = yield* f32([10, 20, 30])
        yield* gradcheck(sumOf((x) => Tensor.where(cond, x, other)), [0.5, -1, 2], [3])
        yield* gradcheck(sumOf((x) => Tensor.where(cond, other, x)), [0.5, -1, 2], [3])
      }))

    it.effect("shape ops", () =>
      Effect.gen(function*() {
        yield* gradcheck(sumOf((x) => Tensor.flatten(x, { startDim: 1 })), [1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])
        yield* gradcheck(sumOf((x) => Tensor.tile(x, [2, 2])), [1, 2, 3, 4], [2, 2])
        yield* gradcheck(sumOf((x) => Tensor.pad(x, [[1, 1], [0, 2]])), [1, 2, 3, 4], [2, 2])
        yield* gradcheck(sumOf((x) => Tensor.triu(x)), [1, 2, 3, 4], [2, 2])
        yield* gradcheck(sumOf((x) => Tensor.tril(x)), [1, 2, 3, 4], [2, 2])
        yield* gradcheck(sumOf((x) => Tensor.trace(x)), [1, 2, 3, 4], [2, 2])
        const b = yield* f32([4, 5, 6])
        yield* gradcheck(sumOf((x) => Tensor.dot(x, b)), [1, 2, 3], [3])
        const parts = yield* Tensor.split(yield* f32([1, 2, 3, 4]), 2)
        assert.strictEqual(parts.length, 2)
      }))

    it.effect("take and gather scatter gradients back", () =>
      Effect.gen(function*() {
        const x = yield* f32([1, 2, 3, 4, 5, 6], [3, 2])
        const idx = yield* Tensor.fromTypedArray(new BigInt64Array([2n, 0n, 2n]))
        const loss = yield* Tensor.sum(yield* Tensor.take(x, idx))
        const [g] = yield* Gradient.grad(loss, [x])
        deep(yield* values(g), [1, 1, 0, 0, 2, 2])

        const idx2 = yield* Tensor.fromTypedArray(new BigInt64Array([1n, 0n, 0n, 1n]), [2, 2])
        const g2 = yield* Tensor.sum(yield* Tensor.gather(x, idx2, { dim: 1 }))
        const [dg] = yield* Gradient.grad(g2, [x])
        deep(yield* values(dg), [1, 1, 1, 1, 0, 0])
      }))

    it.effect("scatterAdd", () =>
      Effect.gen(function*() {
        yield* gradcheck(
          sumOf((x) =>
            Effect.gen(function*() {
              const idx = yield* Tensor.fromTypedArray(new BigInt64Array([2n, 0n, 2n]))
              return yield* Tensor.take(x, idx)
            })
          ),
          [1, 2, 3, 4, 5, 6],
          [3, 2]
        )
        const base = yield* f32([1, 1, 1, 1, 1, 1], [3, 2])
        const idx = yield* Tensor.fromTypedArray(new BigInt64Array([0n, 2n, 2n, 0n]), [2, 2])
        yield* gradcheck(
          sumOf((src) => Tensor.scatterAdd(base, idx, src)),
          [1, 2, 3, 4],
          [2, 2]
        )
      }))

    it.effect("strided slice", () =>
      Effect.gen(function*() {
        yield* gradcheck(
          sumOf((x) => Tensor.slice(x, { start: [1], end: [7], stride: [2] })),
          [1, 2, 3, 4, 5, 6, 7, 8],
          [8]
        )
      }))

    it.effect("convolution and pooling", () =>
      Effect.gen(function*() {
        const w = yield* f32([1, 0, 0, 1], [1, 1, 2, 2])
        yield* gradcheck(sumOf((x) => Tensor.conv2d(x, w)), [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 1, 3, 3])
        const x = yield* f32([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 1, 3, 3])
        yield* gradcheck(sumOf((w2) => Tensor.conv2d(x, w2)), [1, 2, 3, 4], [1, 1, 2, 2])
        const w2s = yield* f32([1, 0, 0, 1], [1, 1, 2, 2])
        yield* gradcheck(
          sumOf((xs) => Tensor.conv2d(xs, w2s, { stride: 2, padding: 1 })),
          [1, 2, 3, 4, 5, 6, 7, 8, 9],
          [1, 1, 3, 3]
        )
        const wg = yield* f32([1, 0, 0, 1, 0, 1, 1, 0], [2, 1, 2, 2])
        yield* gradcheck(
          sumOf((xg) => Tensor.conv2d(xg, wg, { groups: 2 })),
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
          [1, 2, 3, 3]
        )
        // non-uniform stride remainders: 5 % 2 != 6 % 2 with kernel 2
        const wnr = yield* f32([1, 0, 0, 1], [1, 1, 2, 2])
        const xnr = [
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10,
          11,
          12,
          13,
          14,
          15,
          16,
          17,
          18,
          19,
          20,
          21,
          22,
          23,
          24,
          25,
          26,
          27,
          28,
          29,
          30
        ]
        yield* gradcheck(
          sumOf((xr) => Tensor.conv2d(xr, wnr, { stride: 2 })),
          xnr,
          [1, 1, 5, 6]
        )
        const xnrT = yield* f32(xnr, [1, 1, 5, 6])
        yield* gradcheck(
          sumOf((wr) => Tensor.conv2d(xnrT, wr, { stride: 2 })),
          [1, 0, 0, 1],
          [1, 1, 2, 2]
        )
        const w1 = yield* f32([1, 1], [1, 1, 2])
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
        const wt = yield* f32([1, 0, 0, 1], [1, 1, 2, 2])
        yield* gradcheck(sumOf((xt) => Tensor.convTranspose2d(xt, wt)), [1, 2, 3, 4], [1, 1, 2, 2])
        yield* gradcheck(
          sumOf((xt) => Tensor.convTranspose2d(xt, wt, { stride: 2 })),
          [1, 2, 3, 4],
          [1, 1, 2, 2]
        )
        const xt = yield* f32([1, 2, 3, 4], [1, 1, 2, 2])
        yield* gradcheck(sumOf((w2) => Tensor.convTranspose2d(xt, w2)), [1, 2, 3, 4], [1, 1, 2, 2])
      }))

    if (device === "cpu") {
      it.effect("linalg", () =>
        Effect.gen(function*() {
          yield* gradcheck(sumOf((x) => Tensor.det(x)), [4, 1, 1, 3], [2, 2])
          yield* gradcheck(sumOf((x) => Tensor.inverse(x)), [4, 1, 1, 3], [2, 2])
          const b = yield* f32([9, 8], [2, 1])
          yield* gradcheck(sumOf((x) => Tensor.solve(x, b)), [4, 1, 1, 3], [2, 2])
          const a = yield* f32([4, 1, 1, 3], [2, 2])
          yield* gradcheck(sumOf((x) => Tensor.solve(a, x)), [9, 8], [2, 1])
        }))

      it.effect("batched linalg", () =>
        Effect.gen(function*() {
          yield* gradcheck(sumOf((x) => Tensor.det(x)), [4, 1, 1, 3, 2, 1, 1, 2], [2, 2, 2])
          yield* gradcheck(sumOf((x) => Tensor.inverse(x)), [4, 1, 1, 3, 2, 1, 1, 2], [2, 2, 2])
          const b = yield* f32([9, 8, 5, 4], [2, 2, 1])
          yield* gradcheck(sumOf((x) => Tensor.solve(x, b)), [4, 1, 1, 3, 2, 1, 1, 2], [2, 2, 2])
          const a = yield* f32([4, 1, 1, 3, 2, 1, 1, 2], [2, 2, 2])
          yield* gradcheck(sumOf((x) => Tensor.solve(a, x)), [9, 8, 5, 4], [2, 2, 1])
        }))
    }

    it.effect("checkpoint preserves values and gradients, sharing randn draws", () =>
      Effect.gen(function*() {
        const f = (x: Tensor.Any) =>
          Effect.gen(function*() {
            return yield* Tensor.mul(yield* Tensor.sin(x), yield* Tensor.add(x, yield* Tensor.constantLike(x, 1)))
          })
        const x = yield* f32([0.5, 1])
        const plain = yield* f(x)
        const wrapped = yield* Gradient.checkpoint(yield* f(x))
        const plainLoss = yield* Tensor.sum(plain)
        const wrappedLoss = yield* Tensor.sum(wrapped)
        const [plainGrad] = yield* Gradient.grad(plainLoss, [x])
        const [wrappedGrad] = yield* Gradient.grad(wrappedLoss, [x])
        deep(yield* values(wrappedLoss), yield* values(plainLoss))
        const pg = yield* values(plainGrad)
        const wg = yield* values(wrappedGrad)
        for (let i = 0; i < 2; i++) {
          expect(Math.abs(pg[i] - wg[i])).toBeLessThan(TOL)
        }

        // the backward recompute must see the same randn draw as the forward
        const stochastic = (x: Tensor.Any) =>
          Effect.gen(function*() {
            return yield* Tensor.mul(x, yield* Tensor.randn([2], { dtype: floatDtype }))
          })
        const x2 = yield* f32([2, 4])
        const out2 = yield* Gradient.checkpoint(yield* stochastic(x2))
        const loss2 = yield* Tensor.sum(out2)
        const [g2] = yield* Gradient.grad(loss2, [x2])
        const [outM, gM] = yield* Tensor.compute([out2, g2])
        const outValues = yield* Tensor.toNumberArray(outM)
        const gradValues = yield* Tensor.toNumberArray(gM)
        for (let i = 0; i < 2; i++) {
          expect(Math.abs(outValues[i] / [2, 4][i] - gradValues[i])).toBeLessThan(TOL)
        }
      }))

    it.effect("vjp / jvp", () =>
      Effect.gen(function*() {
        const a = yield* f32([1, 2, 3, 4, 5, 6], [2, 3])
        const x = yield* f32([1, 1, 1], [3, 1])
        const y = yield* Tensor.matmul(a, x)
        const v = yield* f32([1, 2], [2, 1])
        const pullback = yield* Gradient.vjp(y, x, v)
        // J^T v = A^T v
        deep(yield* values(pullback), [1 * 1 + 4 * 2, 2 * 1 + 5 * 2, 3 * 1 + 6 * 2])

        const t = yield* f32([1, 0, 0], [3, 1])
        const tangent = yield* Gradient.jvp(y, x, t)
        deep(yield* values(tangent), [1, 4])

        const xn = yield* f32([0.5, 1])
        const yn = yield* Tensor.sin(xn)
        const vn = yield* f32([2, 3])
        const tn = yield* Gradient.jvp(yn, xn, vn)
        const tnValues = yield* values(tn)
        expect(Math.abs(tnValues[0] - Math.cos(0.5) * 2)).toBeLessThan(TOL)
        expect(Math.abs(tnValues[1] - Math.cos(1) * 3)).toBeLessThan(TOL)
      }))

    it.effect("vmap rewrites the graph with per-op batching rules", () =>
      Effect.gen(function*() {
        // elementwise + reduction: per-row sum of squares
        const x1 = yield* f32([1, 2, 3])
        const y1 = yield* Tensor.sum(yield* Tensor.square(x1))
        const bx1 = yield* f32([1, 2, 3, 4, 5, 6], [2, 3])
        const out1 = yield* Gradient.vmap(y1, x1, bx1)
        deep(out1.shape, [2])
        deep(yield* values(out1), [14, 77])

        // matmul: per-example A @ x
        const a = yield* f32([1, 2, 3, 4, 5, 6], [2, 3])
        const x2 = yield* f32([1, 1, 1], [3, 1])
        const y2 = yield* Tensor.matmul(a, x2)
        const bx2 = yield* f32([1, 0, 0, 0, 1, 0, 0, 0, 1], [3, 3, 1])
        const out2 = yield* Gradient.vmap(y2, x2, bx2)
        deep(out2.shape, [3, 2, 1])
        deep(yield* values(out2), [1, 4, 2, 5, 3, 6])

        // softmax + reshape + transpose + slice: matches slice-restack reference
        const x3 = yield* f32([0.5, -1, 2])
        const y3 = yield* Tensor.softmax(x3)
        const bx3 = yield* f32([0.5, -1, 2, 1, 0.5, -0.5], [2, 3])
        const out3 = yield* Gradient.vmap(y3, x3, bx3)
        const ref0 = yield* values(yield* Tensor.softmax(yield* f32([0.5, -1, 2])))
        const ref1 = yield* values(yield* Tensor.softmax(yield* f32([1, 0.5, -0.5])))
        const actual3 = yield* values(out3)
        for (let i = 0; i < 6; i++) {
          expect(Math.abs(actual3[i] - [...ref0, ...ref1][i])).toBeLessThan(TOL)
        }

        // randn draws per batch element
        const x4 = yield* f32([1, 1])
        const y4 = yield* Tensor.mul(x4, yield* Tensor.randn([2], { dtype: floatDtype }))
        const bx4 = yield* f32([1, 1, 1, 1, 1, 1], [3, 2])
        const out4 = yield* values(yield* Gradient.vmap(y4, x4, bx4))
        const rows = [out4.slice(0, 2), out4.slice(2, 4), out4.slice(4, 6)]
        const distinct = new Set(rows.map((r) => r.map((v) => v.toFixed(6)).join(",")))
        assert.assertTrue(distinct.size > 1)

        // gradients flow through the rewritten graph
        const loss = yield* Tensor.sum(out1)
        const [g] = yield* Gradient.grad(loss, [bx1])
        deep(yield* values(g), [2, 4, 6, 8, 10, 12])

        // output that does not depend on the input fails
        const constant = yield* Tensor.ones([2])
        const error = yield* Effect.flip(Gradient.vmap(constant, x1, bx1))
        expect(error.message).toContain("does not depend")
      }))

    it.effect("vmap supports shared-index take, gather and scatterAdd", () =>
      Effect.gen(function*() {
        const x = yield* f32([1, 2, 3, 4, 5, 6], [3, 2])
        const bx = yield* f32([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2])
        const perBatch = (
          f: (slice: Tensor.Any) => Effect.Effect<Tensor.Any, Tensor.TensorError, Runtime.Runtime>
        ) =>
          Effect.gen(function*() {
            const outs: Array<Array<number>> = []
            for (const b of [0, 1]) {
              const slice = yield* Tensor.reshape(
                yield* Tensor.slice(bx, { start: [b, 0, 0], end: [b + 1, 3, 2] }),
                [3, 2]
              )
              outs.push(yield* values(yield* f(slice)))
            }
            return outs.flat()
          })

        const rowIdx = yield* Tensor.fromTypedArray(new BigInt64Array([2n, 0n]))
        const taken = yield* Gradient.vmap(yield* Tensor.take(x, rowIdx), x, bx)
        deep(taken.shape, [2, 2, 2])
        deep(
          yield* values(taken),
          yield* perBatch((s) => Tensor.take(s, rowIdx))
        )

        const gatherIdx = yield* Tensor.fromTypedArray(new BigInt64Array([1n, 0n, 0n, 1n, 1n, 1n]), [3, 2])
        const gathered = yield* Gradient.vmap(yield* Tensor.gather(x, gatherIdx, { dim: 1 }), x, bx)
        deep(gathered.shape, [2, 3, 2])
        deep(
          yield* values(gathered),
          yield* perBatch((s) => Tensor.gather(s, gatherIdx, { dim: 1 }))
        )

        const src = yield* Tensor.mul(x, yield* Tensor.constantLike(x, 2))
        const scattered = yield* Gradient.vmap(
          yield* Tensor.scatterAdd(x, gatherIdx, src, { dim: 1 }),
          x,
          bx
        )
        deep(scattered.shape, [2, 3, 2])
        deep(
          yield* values(scattered),
          yield* perBatch((s) =>
            Effect.gen(function*() {
              return yield* Tensor.scatterAdd(s, gatherIdx, yield* Tensor.mul(s, yield* Tensor.constantLike(s, 2)), {
                dim: 1
              })
            })
          )
        )
      }))

    it.effect("matmul", () =>
      Effect.gen(function*() {
        const b = yield* f32([1, 2, 3, 4, 5, 6], [3, 2])
        yield* gradcheck(sumOf((x) => Tensor.matmul(x, b)), [1, 2, 3, 4, 5, 6], [2, 3])
        const a = yield* f32([1, 2, 3, 4, 5, 6], [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.matmul(a, x)), [1, 2, 3, 4, 5, 6], [3, 2])
      }))

    it.effect("batched matmul with broadcast batch dims", () =>
      Effect.gen(function*() {
        const b = yield* f32([1, 2, 3, 4, 5, 6], [3, 2])
        const input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        yield* gradcheck(sumOf((x) => Tensor.matmul(x, b)), input, [2, 2, 3])
        const c = yield* f32(input, [2, 2, 3])
        yield* gradcheck(sumOf((x) => Tensor.matmul(c, x)), [1, 2, 3, 4, 5, 6], [3, 2])
      }))

    it.effect("reductions sum/mean/max/min", () =>
      Effect.gen(function*() {
        const input = [1, 5, 3, 4, 2, 6]
        yield* gradcheck((x) => Tensor.sum(x), input, [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.sum(x, { dims: [0] })), input, [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.mean(x, { dims: [1], keepdims: true })), input, [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.max(x, { dims: [1] })), input, [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.min(x, { dims: [0], keepdims: true })), input, [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.mean(x, { dims: [0] })), input, [2, 3])
      }))

    it.effect("max/min split gradients evenly across ties", () =>
      Effect.gen(function*() {
        const x = yield* f32([1, 3, 3, 2], [4])
        const [gmax] = yield* Gradient.grad(yield* Tensor.max(x, { dims: [0] }), [x])
        deep(yield* values(gmax), [0, 0.5, 0.5, 0])
        const [gmin] = yield* Gradient.grad(yield* Tensor.min(x, { dims: [0] }), [x])
        deep(yield* values(gmin), [1, 0, 0, 0])
        const y = yield* f32([2, 1, 1, 3], [4])
        const [gymin] = yield* Gradient.grad(yield* Tensor.min(y, { dims: [0] }), [y])
        deep(yield* values(gymin), [0, 0.5, 0.5, 0])
      }))

    it.effect("shape ops", () =>
      Effect.gen(function*() {
        const input = [1, 2, 3, 4, 5, 6]
        yield* gradcheck(sumOf((x) => Tensor.reshape(x, [3, 2])), input, [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.transpose(x, [1, 0])), input, [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.slice(x, { start: [0, 1], end: [2, 3] })), input, [2, 3])
        yield* gradcheck(sumOf((x) => Tensor.broadcastTo(x, [2, 3])), [1, 2, 3], [1, 3])
        yield* gradcheck(sumOf((x) => Tensor.broadcastTo(x, [2, 3])), [1, 2, 3], [3])
        yield* gradcheck(sumOf((x) => Tensor.transpose(x, [2, 0, 1])), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [
          2,
          3,
          2
        ])
        const c = yield* f32([7, 8, 9], [1, 3])
        yield* gradcheck(sumOf((x) => Tensor.concat([x, c], { dim: 0 })), [1, 2, 3], [1, 3])
      }))

    it.effect("cast roundtrip through f16 has identity gradient", () =>
      Effect.gen(function*() {
        const x = yield* f32([1, 2, 3])
        const loss = yield* Effect.flatMap(Tensor.cast(x, "f16"), (t) => Tensor.cast(t, "f32"))
        const [gx] = yield* Gradient.grad(yield* Tensor.sum(loss), [x])
        deep(yield* values(gx), [1, 1, 1])
      }))

    it.effect("composite graph with shared subexpressions", () =>
      Effect.gen(function*() {
        yield* gradcheck(
          (x) =>
            Effect.gen(function*() {
              const y = yield* Tensor.mul(x, x)
              const z = yield* Tensor.add(y, x)
              const out = yield* Tensor.mul(y, z)
              return yield* Tensor.sum(out)
            }),
          [0.5, 1.5, 2.5],
          [3]
        )
      }))
  })

  describe("contract", () => {
    it.effect("rejects non-scalar output", () =>
      Effect.gen(function*() {
        const x = yield* f32([1, 2, 3])
        const exit = yield* Effect.exit(Gradient.grad(x, [x]))
        assert.assertTrue(Exit.isFailure(exit))
      }))

    it.effect("rejects non-float wrt", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(new BigInt64Array([1n, 2n]))
        const loss = yield* Effect.flatMap(Tensor.cast(x, floatDtype), (t) => Tensor.sum(t))
        const exit = yield* Effect.exit(Gradient.grad(loss, [x]))
        assert.assertTrue(Exit.isFailure(exit))
      }))

    it.effect("strided slice is differentiable", () =>
      Effect.gen(function*() {
        const x = yield* f32([1, 2, 3, 4, 5, 6, 7], [7])
        const sliced = yield* Tensor.slice(x, { stride: [2] })
        const loss = yield* Tensor.sum(sliced)
        const [g] = yield* Gradient.grad(loss, [x])
        deep(yield* values(g), [1, 0, 1, 0, 1, 0, 1])
      }))

    it.effect("unused wrt argument yields zeros", () =>
      Effect.gen(function*() {
        const x = yield* f32([1, 2, 3])
        const y = yield* f32([4, 5, 6])
        const loss = yield* Effect.flatMap(Tensor.mul(x, x), (t) => Tensor.sum(t))
        const [gx, gy] = yield* Gradient.grad(loss, [x, y])
        deep(yield* values(gx), [2, 4, 6])
        deep(yield* values(gy), [0, 0, 0])
      }))

    it.effect("stopGradient blocks gradient flow", () =>
      Effect.gen(function*() {
        const x = yield* f32([2, 3])
        const stopped = yield* Gradient.stopGradient(x)
        const loss = yield* Tensor.sum(yield* Tensor.mul(stopped, x))
        const [gx] = yield* Gradient.grad(loss, [x])
        deep(yield* values(gx), [2, 3])
      }))
  })

  describe("grad + compute", () => {
    it.effect("loss and gradients come from the same randn draw", () =>
      Effect.gen(function*() {
        const x = yield* f32([1, 2, 3])
        const r = yield* Tensor.randn([3], { dtype: floatDtype })
        const loss = yield* Tensor.sum(yield* Tensor.mul(x, r))
        const [gx] = yield* Gradient.grad(loss, [x])
        const [l, g] = yield* Tensor.compute([loss, gx])
        const lv = yield* scalar(l)
        const gv = yield* values(g)
        const reconstructed = gv[0] * 1 + gv[1] * 2 + gv[2] * 3
        expect(Math.abs(lv - reconstructed)).toBeLessThan(TOL)
      }))
  })

  describe("end-to-end", () => {
    it.effect("linear regression converges", () =>
      Effect.gen(function*() {
        const trueW = [2, -3]
        const trueB = 1
        const xs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        const ys = xs.map((x) => trueW[0] * x + trueW[1] + trueB)
        const features = xs.flatMap((x) => [x, 1])
        const x = yield* f32(features, [8, 2])
        const y = yield* f32(ys, [8, 1])
        let w = yield* f32([0, 0], [2, 1])
        let b = yield* f32([0], [1, 1])
        const lr = 0.05
        const losses: Array<number> = []
        for (let step = 0; step < 200; step++) {
          const pred = yield* Tensor.add(yield* Tensor.matmul(x, w), b)
          const loss = yield* Loss.mse(pred, y)
          const [gw, gb] = yield* Gradient.grad(loss, [w, b])
          const [lt, gwt, gbt] = yield* Tensor.compute([loss, gw, gb])
          const l = yield* scalar(lt)
          const gwv = yield* values(gwt)
          const gbv = yield* values(gbt)
          losses.push(l)
          w = yield* f32((yield* values(w)).map((v, i) => v - lr * gwv[i]), [2, 1])
          b = yield* f32([(yield* values(b))[0] - lr * gbv[0]], [1, 1])
        }
        expect(losses[losses.length - 1]).toBeLessThan(losses[0] * 1e-3)
        const finalW = yield* values(w)
        const finalB = yield* values(b)
        expect(Math.abs(finalW[0] - trueW[0])).toBeLessThan(0.1)
        // w[1] and b are collinear (both multiply the constant feature 1),
        // only their sum is identified by the data
        expect(Math.abs(finalW[1] + finalB[0] - (trueW[1] + trueB))).toBeLessThan(0.1)
      }))
  })
})

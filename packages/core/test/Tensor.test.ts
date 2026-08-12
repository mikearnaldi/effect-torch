import { describe, expect } from "@effect/vitest"
import * as assert from "@effect/vitest/utils"
import { Effect, Exit } from "effect"
import { Gradient, Loss, Tensor } from "../src/index.ts"
import { deep, floatDtype, floats, onDevices, TOL } from "./utils/devices.ts"

const values = (t: Tensor.Any) =>
  Effect.map(Tensor.toTypedArray(t), (arr) => Array.from<number | bigint>(arr).map(Number))

onDevices("Tensor", (device) => (it) => {
  describe("constructors", () => {
    it.effect("zeros/ones/full produce the right values and dtype", () =>
      Effect.gen(function*() {
        deep(yield* values(yield* Tensor.zeros([2, 3])), [0, 0, 0, 0, 0, 0])
        deep(yield* values(yield* Tensor.ones([2, 2], { dtype: floatDtype })), [1, 1, 1, 1])
        deep(yield* values(yield* Tensor.full([3], 7, { dtype: "i64" })), [7, 7, 7])
      }))

    it.effect("arange with default and explicit step", () =>
      Effect.gen(function*() {
        deep(yield* values(yield* Tensor.arange(5)), [0, 1, 2, 3, 4])
        deep(yield* values(yield* Tensor.arange(1, 10, { step: 2 })), [1, 3, 5, 7, 9])
      }))

    it.effect("constant is a shared 0-d leaf equivalent across calls", () =>
      Effect.gen(function*() {
        const a = yield* Tensor.constant(0.5, { dtype: floatDtype })
        const b = yield* Tensor.constant(0.5, { dtype: floatDtype })
        assert.deepStrictEqual(a.shape, [])
        deep(yield* values(a), yield* values(b))
        const x = yield* Tensor.fromTypedArray(floats([2, 4]), [2])
        deep(yield* values(yield* Tensor.mul(x, a)), [1, 2])
        deep(yield* values(yield* Tensor.mul(x, b)), [1, 2])
      }))

    it.effect("eye", () =>
      Effect.gen(function*() {
        deep(yield* values(yield* Tensor.eye(2)), [1, 0, 0, 1])
      }))

    it.effect("fromTypedArray infers dtype and validates shape", () =>
      Effect.gen(function*() {
        const t = yield* Tensor.fromTypedArray(new Float32Array([1, 2, 3, 4]), [2, 2])
        assert.strictEqual(t.dtype, "f32")
        deep(yield* values(t), [1, 2, 3, 4])
        const u = yield* Tensor.fromTypedArray(new Uint32Array([1, 2, 3]))
        assert.strictEqual(u.dtype, "u32")
        const exit = yield* Effect.exit(Tensor.fromTypedArray(new Float32Array([1, 2, 3]), [2, 2]))
        assert.assertTrue(Exit.isFailure(exit))
      }))

    it.effect("fromTypedArray snapshots its input", () =>
      Effect.gen(function*() {
        const source = new Uint32Array([1, 2, 3])
        const tensor = yield* Tensor.fromTypedArray(source)
        source.fill(9)
        deep(yield* values(tensor), [1, 2, 3])
      }))

    it.effect("reads back non-contiguous views", () =>
      Effect.gen(function*() {
        const source = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4, 5, 6]), [2, 3])
        const transposed = yield* Tensor.transpose(source, [1, 0])
        const [materialized] = yield* Tensor.compute([transposed])
        deep(yield* values(materialized), [1, 4, 2, 5, 3, 6])
      }))
  })

  describe("elementwise", () => {
    it.effect("scalar union and broadcasting", () =>
      Effect.gen(function*() {
        const t = yield* Tensor.arange(4)
        deep(yield* values(yield* Tensor.add(t, yield* Tensor.constantLike(t, 10))), [10, 11, 12, 13])
        deep(yield* values(yield* Tensor.mul(t, yield* Tensor.constantLike(t, 2))), [0, 2, 4, 6])
        deep(yield* values(yield* Tensor.sub(t, yield* Tensor.constantLike(t, 1))), [-1, 0, 1, 2])
        deep(yield* values(yield* Tensor.div(t, yield* Tensor.constantLike(t, 2))), [0, 0.5, 1, 1.5])
      }))

    it.effect("tensor-tensor broadcasting [2,1] + [1,3]", () =>
      Effect.gen(function*() {
        const a = yield* Tensor.full([2, 1], 1)
        const b = yield* Tensor.full([1, 3], 2)
        const t = yield* Tensor.add(a, b)
        deep(t.shape, [2, 3])
        deep(yield* values(t), [3, 3, 3, 3, 3, 3])
      }))

    it.effect("strict dtype: f32 + i64 fails, cast fixes it", () =>
      Effect.gen(function*() {
        const bad = Effect.gen(function*() {
          const a = yield* Tensor.ones([2])
          const b = yield* Tensor.ones([2], { dtype: "i64" })
          return yield* Tensor.add(a, b)
        })
        assert.assertTrue(Exit.isFailure(yield* Effect.exit(bad)))
        const a = yield* Tensor.ones([2])
        const b = yield* Tensor.ones([2], { dtype: "i64" })
        const c = yield* Tensor.cast(b, "f32")
        deep(yield* values(yield* Tensor.add(a, c)), [2, 2])
      }))

    it.effect("comparisons return u8", () =>
      Effect.gen(function*() {
        const r = yield* Tensor.arange(4)
        const t = yield* Tensor.gt(r, yield* Tensor.constantLike(r, 1))
        assert.strictEqual(t.dtype, "u8")
        deep(yield* values(t), [0, 0, 1, 1])
      }))

    it.effect("unary ops", () =>
      Effect.gen(function*() {
        const a = yield* Tensor.fromTypedArray(new Float32Array([-4, 1, 4, 9]))
        const b = yield* Tensor.abs(a)
        deep(yield* values(yield* Tensor.sqrt(b)), [2, 1, 2, 3])
      }))

    it.effect("pow", () =>
      Effect.gen(function*() {
        deep(
          yield* values(yield* Tensor.pow(yield* Tensor.arange(3), 2)),
          [0, 1, 4]
        )
      }))

    it.effect("tanh and sigmoid", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([-1, 0, 1, 10]))
        const tanhValues = yield* values(yield* Tensor.tanh(x))
        const sigmoidValues = yield* values(yield* Tensor.sigmoid(x))
        for (let i = 0; i < 4; i++) {
          const v = [-1, 0, 1, 10][i]
          expect(Math.abs(tanhValues[i] - Math.tanh(v))).toBeLessThan(TOL)
          expect(Math.abs(sigmoidValues[i] - 1 / (1 + Math.exp(-v)))).toBeLessThan(TOL)
        }
      }))

    it.effect("relu", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([-2, -0.5, 0, 0.5, 3]))
        deep(yield* values(yield* Tensor.relu(x)), [0, 0, 0, 0.5, 3])
      }))

    it.effect("maximum and minimum with broadcasting and scalars", () =>
      Effect.gen(function*() {
        const a = yield* Tensor.fromTypedArray(floats([1, 5, -2, 0]), [2, 2])
        const b = yield* Tensor.fromTypedArray(floats([3, 4]), [1, 2])
        deep(yield* values(yield* Tensor.maximum(a, b)), [3, 5, 3, 4])
        deep(yield* values(yield* Tensor.minimum(a, b)), [1, 4, -2, 0])
        deep(yield* values(yield* Tensor.maximum(a, yield* Tensor.constantLike(a, 2))), [2, 5, 2, 2])
        deep(yield* values(yield* Tensor.minimum(a, yield* Tensor.constantLike(a, 2))), [1, 2, -2, 0])
      }))
  })

  describe("reductions", () => {
    const matrix = Tensor.fromTypedArray(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3])

    it.effect("sum over all dims and specific dims", () =>
      Effect.gen(function*() {
        const m = yield* matrix
        deep(yield* values(yield* Tensor.sum(m)), [21])
        const byRow = yield* Tensor.sum(m, { dims: [1] })
        deep(byRow.shape, [2])
        deep(yield* values(byRow), [6, 15])
      }))

    it.effect("keepdims", () =>
      Effect.gen(function*() {
        const t = yield* Tensor.sum(yield* matrix, { dims: [1], keepdims: true })
        deep(t.shape, [2, 1])
      }))

    it.effect("mean/max/min", () =>
      Effect.gen(function*() {
        const m = yield* matrix
        deep(yield* values(yield* Tensor.mean(m)), [3.5])
        deep(yield* values(yield* Tensor.max(m, { dims: [0] })), [4, 5, 6])
        deep(yield* values(yield* Tensor.min(m, { dims: [-1] })), [1, 4])
      }))

    it.effect("mse", () =>
      Effect.gen(function*() {
        const pred = yield* Tensor.fromTypedArray(floats([1, 2, 4]), [3])
        const target = yield* Tensor.fromTypedArray(floats([1, 1, 1]), [3])
        for (
          const loss of [yield* Loss.mse(pred, target), yield* Loss.mse(pred, yield* Tensor.constantLike(pred, 1))]
        ) {
          const [value] = yield* values(loss)
          expect(Math.abs(value - 10 / 3)).toBeLessThan(TOL)
        }
      }))
  })

  describe("shape ops", () => {
    const matrix = Tensor.fromTypedArray(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3])

    it.effect("reshape validates numel", () =>
      Effect.gen(function*() {
        const t = yield* Tensor.reshape(yield* matrix, [3, 2])
        deep(t.shape, [3, 2])
        const exit = yield* Effect.exit(Effect.flatMap(matrix, (m) => Tensor.reshape(m, [4, 2])))
        assert.assertTrue(Exit.isFailure(exit))
      }))

    it.effect("transpose", () =>
      Effect.gen(function*() {
        const t = yield* Tensor.transpose(yield* matrix, [1, 0])
        deep(t.shape, [3, 2])
        deep(yield* values(t), [1, 4, 2, 5, 3, 6])
      }))

    it.effect("slice with negatives and stride", () =>
      Effect.gen(function*() {
        const t = yield* Tensor.slice(yield* matrix, { start: [0, 1], end: [-1, 3] })
        deep(t.shape, [1, 2])
        deep(yield* values(t), [2, 3])
        const strided = yield* Tensor.slice(yield* Tensor.arange(10), { stride: [3] })
        deep(yield* values(strided), [0, 3, 6, 9])
      }))

    it.effect("concat along dim", () =>
      Effect.gen(function*() {
        const a = yield* Tensor.ones([2, 1])
        const b = yield* Tensor.full([2, 2], 2)
        const t = yield* Tensor.concat([a, b, a], { dim: 1 })
        deep(t.shape, [2, 4])
        deep(yield* values(t), [1, 2, 2, 1, 1, 2, 2, 1])
      }))

    it.effect("broadcastTo", () =>
      Effect.gen(function*() {
        const t = yield* Tensor.broadcastTo(yield* Tensor.ones([1, 3]), [2, 3])
        deep(t.shape, [2, 3])
        deep(yield* values(t), [1, 1, 1, 1, 1, 1])
      }))

    it.effect("toNumberArray returns numbers and fails on i64", () =>
      Effect.gen(function*() {
        const t = yield* Tensor.fromTypedArray(floats([1, 2]), [2])
        deep(yield* Tensor.toNumberArray(t), [1, 2])
        const ints = yield* Tensor.fromTypedArray(new BigInt64Array([1n]), [1])
        const error = yield* Effect.flip(Tensor.toNumberArray(ints))
        expect(error.message).toContain("i64")
      }))
  })

  describe("composition", () => {
    it.effect("matmul(eye) roundtrip and deep chains compute once", () =>
      Effect.gen(function*() {
        const a = yield* Tensor.fromTypedArray(new Float32Array([1, 2, 3, 4]), [2, 2])
        const id = yield* Tensor.eye(2)
        const b = yield* Tensor.matmul(a, id)
        const c = yield* Tensor.add(b, yield* Tensor.constantLike(b, 1))
        const d = yield* Tensor.sum(c)
        expect(Array.from<number | bigint>(yield* Tensor.toTypedArray(d))).toEqual([14])
      }))

    it.effect("batched matmul broadcasts batch dims", () =>
      Effect.gen(function*() {
        const a = yield* Tensor.fromTypedArray(
          floats([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
          [2, 2, 3]
        )
        const b = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4, 5, 6]), [3, 2])
        const out = yield* Tensor.matmul(a, b)
        deep(out.shape, [2, 2, 2])
        deep(yield* values(out), [22, 28, 49, 64, 76, 100, 103, 136])
      }))
  })

  describe("extended constructors", () => {
    it.effect("uniform produces values in [min, max)", () =>
      Effect.gen(function*() {
        const t = yield* Tensor.uniform([1000], { min: -2, max: 3 })
        const vs = yield* values(t)
        assert.assertTrue(vs.every((v) => v >= -2 && v < 3))
        const lo = Math.min(...vs)
        const hi = Math.max(...vs)
        assert.assertTrue(lo < -1 && hi > 2)
      }))

    it.effect("linspace", () =>
      Effect.gen(function*() {
        deep(yield* values(yield* Tensor.linspace(0, 1, 5, { dtype: floatDtype })), [
          0,
          0.25,
          0.5,
          0.75,
          1
        ])
        deep(yield* values(yield* Tensor.linspace(3, 9, 1, { dtype: floatDtype })), [3])
      }))

    it.effect("zerosLike/onesLike/fullLike", () =>
      Effect.gen(function*() {
        const t = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4]), [2, 2])
        const z = yield* Tensor.zerosLike(t)
        deep(z.shape, [2, 2])
        assert.strictEqual(z.dtype, floatDtype)
        deep(yield* values(z), [0, 0, 0, 0])
        deep(yield* values(yield* Tensor.onesLike(t)), [1, 1, 1, 1])
        deep(yield* values(yield* Tensor.fullLike(t, 9)), [9, 9, 9, 9])
      }))
  })

  describe("extended elementwise", () => {
    it.effect("erf/floor/ceil/round/sign", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([-1.5, 0.2, 0, 0.5, 2.7]))
        const erfValues = yield* values(yield* Tensor.erf(x))
        const ref = [-0.9661051, 0.2227026, 0, 0.5204999, 0.9998657]
        for (let i = 0; i < 5; i++) {
          expect(Math.abs(erfValues[i] - ref[i])).toBeLessThan(TOL)
        }
        deep(yield* values(yield* Tensor.floor(x)), [-2, 0, 0, 0, 2])
        deep(yield* values(yield* Tensor.ceil(x)), [-1, 1, 0, 1, 3])
        deep(yield* values(yield* Tensor.round(x)), [-2, 0, 0, 1, 3])
        deep(yield* values(yield* Tensor.sign(x)), [-1, 1, 0, 1, 1])
      }))

    it.effect("square/rsqrt/reciprocal/expm1/log1p/log2/log10", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([0.5, 1, 4]))
        deep(yield* values(yield* Tensor.square(x)), [0.25, 1, 16])
        const rsqrt = yield* values(yield* Tensor.rsqrt(x))
        for (let i = 0; i < 3; i++) {
          expect(Math.abs(rsqrt[i] - 1 / Math.sqrt([0.5, 1, 4][i]))).toBeLessThan(TOL)
        }
        deep(yield* values(yield* Tensor.reciprocal(x)), [2, 1, 0.25])
        const expm1 = yield* values(yield* Tensor.expm1(x))
        const log1p = yield* values(yield* Tensor.log1p(x))
        const log2 = yield* values(yield* Tensor.log2(x))
        const log10 = yield* values(yield* Tensor.log10(x))
        for (let i = 0; i < 3; i++) {
          const v = [0.5, 1, 4][i]
          expect(Math.abs(expm1[i] - (Math.exp(v) - 1))).toBeLessThan(TOL)
          expect(Math.abs(log1p[i] - Math.log1p(v))).toBeLessThan(TOL)
          expect(Math.abs(log2[i] - Math.log2(v))).toBeLessThan(TOL)
          expect(Math.abs(log10[i] - Math.log10(v))).toBeLessThan(TOL)
        }
      }))

    it.effect("sinh/cosh/tan", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([0.5, -1, 2]))
        const sinh = yield* values(yield* Tensor.sinh(x))
        const cosh = yield* values(yield* Tensor.cosh(x))
        const tan = yield* values(yield* Tensor.tan(x))
        for (let i = 0; i < 3; i++) {
          const v = [0.5, -1, 2][i]
          expect(Math.abs(sinh[i] - Math.sinh(v))).toBeLessThan(TOL)
          expect(Math.abs(cosh[i] - Math.cosh(v))).toBeLessThan(TOL)
          expect(Math.abs(tan[i] - Math.tan(v))).toBeLessThan(TOL)
        }
      }))

    it.effect("ne/logicalAnd/logicalOr/logicalNot/remainder", () =>
      Effect.gen(function*() {
        const a = yield* Tensor.fromTypedArray(floats([1, 2, 3]))
        deep(yield* values(yield* Tensor.ne(a, yield* Tensor.constantLike(a, 2))), [1, 0, 1])
        const c1 = yield* Tensor.fromTypedArray(new Uint8Array([1, 1, 0, 0]))
        const c2 = yield* Tensor.fromTypedArray(new Uint8Array([1, 0, 1, 0]))
        deep(yield* values(yield* Tensor.logicalAnd(c1, c2)), [1, 0, 0, 0])
        deep(yield* values(yield* Tensor.logicalOr(c1, c2)), [1, 1, 1, 0])
        deep(yield* values(yield* Tensor.logicalNot(c1)), [0, 0, 1, 1])
        const b = yield* Tensor.fromTypedArray(floats([5.5, -5.5, 7]))
        const r = yield* values(yield* Tensor.remainder(b, yield* Tensor.constantLike(b, 3)))
        for (let i = 0; i < 3; i++) {
          const v = [5.5, -5.5, 7][i]
          expect(Math.abs(r[i] - ((v % 3) + 3) % 3)).toBeLessThan(TOL)
        }
      }))

    it.effect("where with broadcasting and scalar branches", () =>
      Effect.gen(function*() {
        const cond = yield* Tensor.fromTypedArray(new Uint8Array([1, 0, 1]))
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3]))
        const y = yield* Tensor.fromTypedArray(floats([10, 20, 30]))
        deep(yield* values(yield* Tensor.where(cond, x, y)), [1, 20, 3])
        deep(yield* values(yield* Tensor.where(cond, x, yield* Tensor.constantLike(x, 0))), [1, 0, 3])
        const bad = yield* Tensor.fromTypedArray(new BigInt64Array([1n]))
        const error = yield* Effect.flip(Tensor.where(cond, x, bad))
        expect(error.message).toContain("dtype")
      }))
  })

  describe("extended reductions", () => {
    it.effect("argmax/argmin", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([1, 5, 3, 4, 0, 2]), [2, 3])
        const amax = yield* Tensor.argmax(x, 1)
        deep(amax.shape, [2])
        deep(amax.dtype, "i64")
        deep(yield* values(amax), [1, 0])
        deep(yield* values(yield* Tensor.argmin(x, 0)), [0, 1, 1])
      }))

    it.effect("cumsum", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4, 5, 6]), [2, 3])
        deep(yield* values(yield* Tensor.cumsum(x, 1)), [1, 3, 6, 4, 9, 15])
        deep(yield* values(yield* Tensor.cumsum(x, 0)), [1, 2, 3, 5, 7, 9])
      }))

    it.effect("variance/std", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4]))
        const [v] = yield* values(yield* Tensor.variance(x))
        expect(Math.abs(v - 5 / 3)).toBeLessThan(TOL)
        const [vp] = yield* values(yield* Tensor.variance(x, { correction: 0 }))
        expect(Math.abs(vp - 1.25)).toBeLessThan(TOL)
        const [s] = yield* values(yield* Tensor.std(x))
        expect(Math.abs(s - Math.sqrt(5 / 3))).toBeLessThan(TOL)
      }))

    it.effect("norm", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([3, -4]))
        const [l2] = yield* values(yield* Tensor.norm(x))
        expect(Math.abs(l2 - 5)).toBeLessThan(TOL)
        const [l1] = yield* values(yield* Tensor.norm(x, { ord: 1 }))
        expect(Math.abs(l1 - 7)).toBeLessThan(TOL)
        const [linf] = yield* values(yield* Tensor.norm(x, { ord: Infinity }))
        expect(Math.abs(linf - 4)).toBeLessThan(TOL)
        const [l3] = yield* values(yield* Tensor.norm(x, { ord: 3 }))
        expect(Math.abs(l3 - Math.cbrt(91))).toBeLessThan(TOL)
      }))

    it.effect("all/any", () =>
      Effect.gen(function*() {
        const t = yield* Tensor.fromTypedArray(new Uint8Array([1, 1, 0, 1]), [2, 2])
        deep(yield* values(yield* Tensor.all(t)), [0])
        deep(yield* values(yield* Tensor.all(t, { dims: [1] })), [1, 0])
        deep(yield* values(yield* Tensor.any(t)), [1])
        deep(yield* values(yield* Tensor.any(t, { dims: [0] })), [1, 1])
      }))

    it.effect("logsumexp", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4, 5, 6]), [2, 3])
        const lse = yield* values(yield* Tensor.logsumexp(x, { dims: [1] }))
        for (let i = 0; i < 2; i++) {
          const row = [1, 2, 3, 4, 5, 6].slice(i * 3, i * 3 + 3)
          const m = Math.max(...row)
          const expected = m + Math.log(row.reduce((a, v) => a + Math.exp(v - m), 0))
          expect(Math.abs(lse[i] - expected)).toBeLessThan(TOL)
        }
      }))

    it.effect("prod", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4, 5, 6]), [2, 3])
        deep(yield* values(yield* Tensor.prod(x)), [720])
        deep(yield* values(yield* Tensor.prod(x, { dims: [1] })), [6, 120])
        const kept = yield* Tensor.prod(x, { dims: [1], keepdims: true })
        deep(kept.shape, [2, 1])
      }))
  })

  describe("neural network ops", () => {
    it.effect("rmsNorm normalizes each last-dimension row with optional weight", () =>
      Effect.gen(function*() {
        const inputValues = [-2, -1, 0, 1, 1, 2, 3, 4, 4, 3, 2, 1, -1, -2, -3, -4]
        const weightValues = [0.5, 1, 1.5, 2]
        const eps = 1e-5
        const input = yield* Tensor.fromTypedArray(floats(inputValues), [2, 2, 4])
        const weight = yield* Tensor.fromTypedArray(floats(weightValues), [4])
        const expected = (weighted: boolean) =>
          inputValues.map((value, index) => {
            const rowStart = Math.floor(index / 4) * 4
            const row = inputValues.slice(rowStart, rowStart + 4)
            const scale = 1 / Math.sqrt(row.reduce((sum, x) => sum + x * x, 0) / row.length + eps)
            return value * scale * (weighted ? weightValues[index % 4] : 1)
          })

        const normalized = yield* Tensor.rmsNorm(input, undefined, eps)
        deep(normalized.shape, [2, 2, 4])
        deep(yield* values(normalized), expected(false))
        deep(yield* values(yield* Tensor.rmsNorm(input, weight, eps)), expected(true))
      }))

    it.effect("rmsNorm validates input and weight shapes", () =>
      Effect.gen(function*() {
        const scalar = yield* Tensor.constant(1, { dtype: floatDtype })
        const scalarError = yield* Effect.flip(Tensor.rmsNorm(scalar))
        expect(scalarError.message).toContain("rank at least 1")

        const input = yield* Tensor.ones([2, 4], { dtype: floatDtype })
        const weight = yield* Tensor.ones([2, 2], { dtype: floatDtype })
        const weightError = yield* Effect.flip(Tensor.rmsNorm(input, weight))
        expect(weightError.message).toContain("weight must be [4]")
      }))

    it.effect("softmax rows sum to 1 and logSoftmax agrees", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4, 5, 6]), [2, 3])
        const sm = yield* Tensor.softmax(x)
        deep(sm.shape, [2, 3])
        const rows = yield* values(yield* Tensor.sum(sm, { dims: [1] }))
        for (const r of rows) {
          expect(Math.abs(r - 1)).toBeLessThan(TOL)
        }
        const lsm = yield* values(yield* Tensor.logSoftmax(x))
        const smValues = yield* values(sm)
        for (let i = 0; i < 6; i++) {
          expect(Math.abs(lsm[i] - Math.log(smValues[i]))).toBeLessThan(TOL)
        }
      }))

    it.effect("silu/softplus/elu/leakyRelu/gelu/mish/hardtanh/clamp", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([0.5, -1, 2]))
        const vs = [0.5, -1, 2]
        const sig = (v: number) => 1 / (1 + Math.exp(-v))
        const silu = yield* values(yield* Tensor.silu(x))
        const softplus = yield* values(yield* Tensor.softplus(x))
        const elu = yield* values(yield* Tensor.elu(x))
        const leaky = yield* values(yield* Tensor.leakyRelu(x))
        const gelu = yield* values(yield* Tensor.gelu(x))
        const geluT = yield* values(yield* Tensor.gelu(x, { approximate: "tanh" }))
        const mish = yield* values(yield* Tensor.mish(x))
        for (let i = 0; i < 3; i++) {
          const v = vs[i]
          const sp = Math.max(v, 0) + Math.log1p(Math.exp(-Math.abs(v)))
          const phi = [0.6914624612740131, 0.15865525393145707, 0.9772498680518208][i]
          expect(Math.abs(silu[i] - v * sig(v))).toBeLessThan(TOL)
          expect(Math.abs(softplus[i] - sp)).toBeLessThan(TOL)
          expect(Math.abs(elu[i] - (v > 0 ? v : Math.exp(v) - 1))).toBeLessThan(TOL)
          expect(Math.abs(leaky[i] - (v > 0 ? v : 0.01 * v))).toBeLessThan(TOL)
          expect(Math.abs(gelu[i] - v * phi)).toBeLessThan(TOL)
          const inner = Math.sqrt(2 / Math.PI) * (v + 0.044715 * v ** 3)
          expect(Math.abs(geluT[i] - v * 0.5 * (1 + Math.tanh(inner)))).toBeLessThan(TOL)
          expect(Math.abs(mish[i] - v * Math.tanh(sp))).toBeLessThan(TOL)
        }
        deep(yield* values(yield* Tensor.hardtanh(x)), [0.5, -1, 1])
        deep(yield* values(yield* Tensor.clamp(x, { min: 0, max: 1 })), [0.5, 0, 1])
      }))

    it.effect("dropout p=0 is identity, invalid p fails", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3]))
        deep(yield* values(yield* Tensor.dropout(x, { p: 0 })), [1, 2, 3])
        const error = yield* Effect.flip(Tensor.dropout(x, { p: 1.5 }))
        expect(error.message).toContain("p must be")
      }))
  })

  describe("extended shape operations", () => {
    it.effect("flatten/squeeze/unsqueeze", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4, 5, 6, 7, 8]), [2, 2, 2])
        const f = yield* Tensor.flatten(x, { startDim: 1 })
        deep(f.shape, [2, 4])
        const f0 = yield* Tensor.flatten(x)
        deep(f0.shape, [8])
        const u = yield* Tensor.unsqueeze(x, 1)
        deep(u.shape, [2, 1, 2, 2])
        const s = yield* Tensor.squeeze(u, { dims: [1] })
        deep(s.shape, [2, 2, 2])
        const sAll = yield* Tensor.squeeze(yield* Tensor.reshape(x, [1, 2, 1, 2, 2]))
        deep(sAll.shape, [2, 2, 2])
      }))

    it.effect("stack/split/chunk", () =>
      Effect.gen(function*() {
        const a = yield* Tensor.fromTypedArray(floats([1, 2]))
        const b = yield* Tensor.fromTypedArray(floats([3, 4]))
        const stacked = yield* Tensor.stack([a, b], { dim: 1 })
        deep(stacked.shape, [2, 2])
        deep(yield* values(stacked), [1, 3, 2, 4])
        const parts = yield* Tensor.split(yield* Tensor.arange(7, undefined, { dtype: floatDtype }), 3)
        deep(parts.map((p) => p.shape), [[3], [3], [1]])
        deep(yield* values(parts[1]), [3, 4, 5])
        const chunks = yield* Tensor.chunk(yield* Tensor.arange(7, undefined, { dtype: floatDtype }), 3)
        deep(chunks.map((p) => p.shape), [[3], [3], [1]])
      }))

    it.effect("tile", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4]), [2, 2])
        const t = yield* Tensor.tile(x, [2, 3])
        deep(t.shape, [4, 6])
        deep(yield* values(t), [
          1,
          2,
          1,
          2,
          1,
          2,
          3,
          4,
          3,
          4,
          3,
          4,
          1,
          2,
          1,
          2,
          1,
          2,
          3,
          4,
          3,
          4,
          3,
          4
        ])
        const v = yield* Tensor.tile(yield* Tensor.fromTypedArray(floats([1, 2])), [3])
        deep(yield* values(v), [1, 2, 1, 2, 1, 2])
      }))

    it.effect("pad", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4]), [2, 2])
        const p = yield* Tensor.pad(x, [[1, 0], [0, 2]])
        deep(p.shape, [3, 4])
        deep(yield* values(p), [0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0])
      }))

    it.effect("take", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4, 5, 6]), [3, 2])
        const idx = yield* Tensor.fromTypedArray(new BigInt64Array([2n, 0n]))
        const t = yield* Tensor.take(x, idx)
        deep(t.shape, [2, 2])
        deep(yield* values(t), [5, 6, 1, 2])
        const t1 = yield* Tensor.take(x, yield* Tensor.fromTypedArray(new BigInt64Array([1n, 0n])), { dim: 1 })
        deep(t1.shape, [3, 2])
        deep(yield* values(t1), [2, 1, 4, 3, 6, 5])
      }))

    it.effect("gather and scatterAdd", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4, 5, 6]), [3, 2])
        const idx = yield* Tensor.fromTypedArray(new BigInt64Array([1n, 0n, 0n, 1n]), [2, 2])
        const g = yield* Tensor.gather(x, idx, { dim: 1 })
        deep(g.shape, [2, 2])
        deep(yield* values(g), [2, 1, 3, 4])
        const base = yield* Tensor.zeros([3, 2], { dtype: floatDtype })
        const s = yield* Tensor.scatterAdd(
          base,
          yield* Tensor.fromTypedArray(new BigInt64Array([1n, 0n, 0n, 1n, 1n, 0n]), [3, 2]),
          yield* Tensor.fromTypedArray(floats([10, 20, 30, 40, 50, 60]), [3, 2]),
          { dim: 1 }
        )
        deep(yield* values(s), [20, 10, 30, 40, 60, 50])
      }))

    it.effect("take, gather and scatterAdd accept u32 indexes", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4, 5, 6]), [3, 2])
        const idx = yield* Tensor.fromTypedArray(new Uint32Array([2, 0]))
        const t = yield* Tensor.take(x, idx)
        deep(t.shape, [2, 2])
        deep(yield* values(t), [5, 6, 1, 2])
        const g = yield* Tensor.gather(
          x,
          yield* Tensor.fromTypedArray(new Uint32Array([1, 0, 0, 1]), [2, 2]),
          { dim: 1 }
        )
        deep(yield* values(g), [2, 1, 3, 4])
        const s = yield* Tensor.scatterAdd(
          yield* Tensor.zeros([3, 2], { dtype: floatDtype }),
          yield* Tensor.fromTypedArray(new Uint32Array([1, 0, 0, 1, 1, 0]), [3, 2]),
          yield* Tensor.fromTypedArray(floats([10, 20, 30, 40, 50, 60]), [3, 2]),
          { dim: 1 }
        )
        deep(yield* values(s), [20, 10, 30, 40, 60, 50])
        const loss = yield* Tensor.sum(yield* Tensor.take(x, idx))
        const [gradX] = yield* Gradient.grad(loss, [x])
        deep(yield* values(gradX), [1, 1, 0, 0, 1, 1])
      }))

    it.effect("embedding", () =>
      Effect.gen(function*() {
        const weight = yield* Tensor.fromTypedArray(
          floats([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
          [4, 3]
        )
        const out = yield* Tensor.embedding(
          yield* Tensor.fromTypedArray(new BigInt64Array([2n, 0n, 1n, 3n]), [2, 2]),
          { weight }
        )
        deep(out.shape, [2, 2, 3])
        deep(yield* values(out), [7, 8, 9, 1, 2, 3, 4, 5, 6, 10, 11, 12])
      }))

    it.effect("embedding with paddingIndex returns the stored row but masks its gradient", () =>
      Effect.gen(function*() {
        const weight = yield* Tensor.fromTypedArray(
          floats([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
          [4, 3]
        )
        const indexes = yield* Tensor.fromTypedArray(new BigInt64Array([2n, 0n, 2n]))
        const out = yield* Tensor.embedding(indexes, { paddingIndex: 0, weight })
        deep(yield* values(out), [7, 8, 9, 1, 2, 3, 7, 8, 9])
        const [gradW] = yield* Gradient.grad(yield* Tensor.sum(out), [weight])
        deep(yield* values(gradW), [0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0])
      }))

    it.effect("embedding validates its arguments", () =>
      Effect.gen(function*() {
        const weight = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4]), [2, 2])
        const indexes = yield* Tensor.fromTypedArray(new BigInt64Array([0n]))
        const badRank = yield* Effect.flip(Tensor.embedding(indexes, {
          weight: yield* Tensor.reshape(weight, [1, 2, 2])
        }))
        deep(badRank.op, "embedding")
        const badPadding = yield* Effect.flip(Tensor.embedding(indexes, { paddingIndex: 2, weight }))
        deep(badPadding.op, "embedding")
        const badDtype = yield* Effect.flip(Tensor.embedding(
          yield* Tensor.cast(indexes, "f32"),
          { weight }
        ))
        deep(badDtype.op, "embedding")
      }))

    it.effect("crossEntropy matches a stable JS reference and differentiates", () =>
      Effect.gen(function*() {
        const rows = [[1, 2, 3], [0.5, -1, 2]]
        const targets = [2, 1]
        const logits = yield* Tensor.fromTypedArray(floats(rows.flat()), [2, 3])
        const target = yield* Tensor.fromTypedArray(new BigInt64Array(targets.map(BigInt)), [2])
        const loss = yield* Tensor.crossEntropy(logits, { target })
        const lse = (r: Array<number>) => {
          const m = Math.max(...r)
          return m + Math.log(r.reduce((acc, v) => acc + Math.exp(v - m), 0))
        }
        const expected = (lse(rows[0]) - rows[0][2] + lse(rows[1]) - rows[1][1]) / 2
        const [lossValue] = yield* values(loss)
        expect(Math.abs(lossValue - expected)).toBeLessThan(TOL)
        const [grad] = yield* Gradient.grad(loss, [logits])
        const softmax = (r: Array<number>) => {
          const m = Math.max(...r)
          const e = r.map((v) => Math.exp(v - m))
          const s = e.reduce((a, b) => a + b, 0)
          return e.map((v) => v / s)
        }
        const expectedGrad = [
          ...softmax(rows[0]).map((p, c) => (p - (c === 2 ? 1 : 0)) / 2),
          ...softmax(rows[1]).map((p, c) => (p - (c === 1 ? 1 : 0)) / 2)
        ]
        const gradValues = yield* values(grad)
        gradValues.forEach((g, i) => expect(Math.abs(g - expectedGrad[i])).toBeLessThan(TOL))
      }))

    it.effect("crossEntropy ignores positions matching ignoreIndex", () =>
      Effect.gen(function*() {
        const logits = yield* Tensor.fromTypedArray(floats([1, 2, 3, 0.5, -1, 2]), [2, 3])
        const target = yield* Tensor.fromTypedArray(new BigInt64Array([2n, -100n]), [2])
        const loss = yield* Tensor.crossEntropy(logits, { target })
        const m = 3
        const expected = m + Math.log(Math.exp(1 - m) + Math.exp(2 - m) + 1) - 3
        const [lossValue] = yield* values(loss)
        expect(Math.abs(lossValue - expected)).toBeLessThan(TOL)
        const [grad] = yield* Gradient.grad(loss, [logits])
        const gradValues = yield* values(grad)
        gradValues.slice(3).forEach((g) => expect(g).toBe(0))
        expect(gradValues[0]).toBeGreaterThan(0)
      }))

    it.effect("crossEntropy supports higher-rank logits and u32 targets", () =>
      Effect.gen(function*() {
        const logits = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4, 5, 6, 7, 8]), [2, 2, 2])
        const target = yield* Tensor.fromTypedArray(new Uint32Array([1, 0, 0, 1]), [2, 2])
        const loss = yield* Tensor.crossEntropy(logits, { target })
        const lse = (a: number, b: number) => {
          const m = Math.max(a, b)
          return m + Math.log(Math.exp(a - m) + Math.exp(b - m))
        }
        const expected = (lse(1, 2) - 2 + lse(3, 4) - 3 + lse(5, 6) - 5 + lse(7, 8) - 8) / 4
        const [lossValue] = yield* values(loss)
        expect(Math.abs(lossValue - expected)).toBeLessThan(TOL)
      }))

    it.effect("crossEntropy stays finite for large-magnitude logits", () =>
      Effect.gen(function*() {
        const logits = yield* Tensor.fromTypedArray(floats([1e4, 1e4 + 1, 1e4 - 1]), [1, 3])
        const target = yield* Tensor.fromTypedArray(new BigInt64Array([0n]), [1])
        const loss = yield* Tensor.crossEntropy(logits, { target })
        const [lossValue] = yield* values(loss)
        expect(Number.isFinite(lossValue)).toBe(true)
        const expected = Math.log(1 + Math.exp(1) + Math.exp(-1))
        // f32 computes lse - picked as (1e4 + 1.4) - 1e4: the subtraction
        // cancels catastrophically, leaving ~1e-3 of rounding behind
        expect(Math.abs(lossValue - expected)).toBeLessThan(5e-3)
        const [grad] = yield* Gradient.grad(loss, [logits])
        ;(yield* values(grad)).forEach((g) => expect(Number.isFinite(g)).toBe(true))
      }))

    it.effect("crossEntropy fails at evaluation on an empty active set or out-of-range target", () =>
      Effect.gen(function*() {
        const logits = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4, 5, 6]), [2, 3])
        const allIgnored = yield* Tensor.crossEntropy(logits, {
          target: yield* Tensor.fromTypedArray(new BigInt64Array([-100n, -100n]), [2])
        })
        const emptyError = yield* Effect.flip(Tensor.toTypedArray(allIgnored))
        expect(emptyError.message).toContain("no active targets")
        const outOfRange = yield* Tensor.crossEntropy(logits, {
          target: yield* Tensor.fromTypedArray(new BigInt64Array([5n, 0n]), [2])
        })
        const rangeError = yield* Effect.flip(Tensor.toTypedArray(outOfRange))
        expect(rangeError.message).toContain("out of range")
      }))

    it.effect("crossEntropy validates its arguments at construction", () =>
      Effect.gen(function*() {
        const logits = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4, 5, 6]), [2, 3])
        const badShape = yield* Effect.flip(Tensor.crossEntropy(logits, {
          target: yield* Tensor.fromTypedArray(new BigInt64Array([0n, 1n, 2n]), [3])
        }))
        deep(badShape.op, "crossEntropy")
        const badDtype = yield* Effect.flip(Tensor.crossEntropy(logits, {
          target: yield* Tensor.fromTypedArray(floats([0, 1]), [2])
        }))
        deep(badDtype.op, "crossEntropy")
      }))

    it.effect("flip", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4, 5, 6]), [2, 3])
        deep(yield* values(yield* Tensor.flip(x, [0])), [4, 5, 6, 1, 2, 3])
        deep(yield* values(yield* Tensor.flip(x, [1])), [3, 2, 1, 6, 5, 4])
        deep(yield* values(yield* Tensor.flip(x, [0, 1])), [6, 5, 4, 3, 2, 1])
      }))

    it.effect("oneHot", () =>
      Effect.gen(function*() {
        const idx = yield* Tensor.fromTypedArray(new BigInt64Array([0n, 2n, 1n]))
        const oh = yield* Tensor.oneHot(idx, 3)
        deep(oh.shape, [3, 3])
        deep(yield* values(oh), [1, 0, 0, 0, 0, 1, 0, 1, 0])
      }))

    it.effect("triu/tril", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4, 5, 6, 7, 8, 9]), [3, 3])
        deep(yield* values(yield* Tensor.triu(x)), [1, 2, 3, 0, 5, 6, 0, 0, 9])
        deep(yield* values(yield* Tensor.tril(x)), [1, 0, 0, 4, 5, 0, 7, 8, 9])
        deep(yield* values(yield* Tensor.triu(x, { diagonal: 1 })), [0, 2, 3, 0, 0, 6, 0, 0, 0])
      }))

    it.effect("dot/trace", () =>
      Effect.gen(function*() {
        const a = yield* Tensor.fromTypedArray(floats([1, 2, 3]))
        const b = yield* Tensor.fromTypedArray(floats([4, 5, 6]))
        deep(yield* values(yield* Tensor.dot(a, b)), [32])
        const m = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4]), [2, 2])
        deep(yield* values(yield* Tensor.trace(m)), [5])
      }))
    if (device === "cpu") {
      it.effect("inverse/det/solve", () =>
        Effect.gen(function*() {
          const a = yield* Tensor.fromTypedArray(floats([4, 1, 1, 3]), [2, 2])
          const inv = yield* Tensor.inverse(a)
          const identity = yield* values(yield* Tensor.matmul(a, inv))
          for (let i = 0; i < 4; i++) {
            expect(Math.abs(identity[i] - [1, 0, 0, 1][i])).toBeLessThan(TOL)
          }
          const [d] = yield* values(yield* Tensor.det(a))
          expect(Math.abs(d - 11)).toBeLessThan(TOL)
          const b = yield* Tensor.fromTypedArray(floats([9, 8]), [2, 1])
          const x = yield* Tensor.solve(a, b)
          const xValues = yield* values(x)
          expect(Math.abs(xValues[0] - 19 / 11)).toBeLessThan(TOL)
          expect(Math.abs(xValues[1] - 23 / 11)).toBeLessThan(TOL)
          const singular = yield* Tensor.fromTypedArray(floats([1, 2, 2, 4]), [2, 2])
          const error = yield* Effect.flip(Effect.flatMap(Tensor.inverse(singular), (t) => values(t)))
          expect(error.message).toContain("singular")
        }))

      it.effect("batched inverse/det/solve", () =>
        Effect.gen(function*() {
          const a = yield* Tensor.fromTypedArray(floats([4, 1, 1, 3, 1, 2, 3, 4]), [2, 2, 2])
          const inv = yield* Tensor.inverse(a)
          deep(inv.shape, [2, 2, 2])
          const identity = yield* values(yield* Tensor.matmul(a, inv))
          const expectedIdentity = [1, 0, 0, 1, 1, 0, 0, 1]
          for (let i = 0; i < 8; i++) {
            expect(Math.abs(identity[i] - expectedIdentity[i])).toBeLessThan(TOL)
          }
          const dets = yield* values(yield* Tensor.det(a))
          deep(dets.length, 2)
          expect(Math.abs(dets[0] - 11)).toBeLessThan(TOL)
          expect(Math.abs(dets[1] - -2)).toBeLessThan(TOL)
          const b = yield* Tensor.fromTypedArray(floats([9, 8, 1, 1]), [2, 2, 1])
          const x = yield* Tensor.solve(a, b)
          deep(x.shape, [2, 2, 1])
          const xValues = yield* values(x)
          expect(Math.abs(xValues[0] - 19 / 11)).toBeLessThan(TOL)
          expect(Math.abs(xValues[1] - 23 / 11)).toBeLessThan(TOL)
          expect(Math.abs(xValues[2] - -1)).toBeLessThan(TOL)
          expect(Math.abs(xValues[3] - 1)).toBeLessThan(TOL)
        }))
    } else {
      it.effect("rejects unsupported linalg instead of falling back to CPU", () =>
        Effect.gen(function*() {
          const a = yield* Tensor.fromTypedArray(floats([4, 1, 1, 3]), [2, 2])
          const b = yield* Tensor.fromTypedArray(floats([9, 8]), [2, 1])
          for (const operation of [Tensor.inverse(a), Tensor.det(a), Tensor.solve(a, b)]) {
            const error = yield* Effect.flip(Effect.flatMap(operation, values))
            expect(error.message).toContain("not supported on Metal")
          }
        }))
    }
  })

  describe("convolution and pooling", () => {
    const refConv2d = (
      x: Array<number>,
      xShape: [number, number, number, number],
      w: Array<number>,
      wShape: [number, number, number, number],
      stride: number,
      padding: number,
      dilation: number,
      groups: number
    ) => {
      const [n, cIn, h, wid] = xShape
      const [cOut, cPer, kh, kw] = wShape
      const oh = Math.floor((h + 2 * padding - dilation * (kh - 1) - 1) / stride) + 1
      const ow = Math.floor((wid + 2 * padding - dilation * (kw - 1) - 1) / stride) + 1
      const out = new Array<number>(n * cOut * oh * ow).fill(0)
      const xAt = (b: number, c: number, y: number, z: number) =>
        y < 0 || y >= h || z < 0 || z >= wid ? 0 : x[((b * cIn + c) * h + y) * wid + z]
      for (let b = 0; b < n; b++) {
        for (let g = 0; g < groups; g++) {
          for (let co = 0; co < cOut / groups; co++) {
            const oc = g * (cOut / groups) + co
            for (let oy = 0; oy < oh; oy++) {
              for (let oz = 0; oz < ow; oz++) {
                let acc = 0
                for (let ci = 0; ci < cPer; ci++) {
                  const ic = g * cPer + ci
                  for (let ky = 0; ky < kh; ky++) {
                    for (let kz = 0; kz < kw; kz++) {
                      acc += xAt(b, ic, oy * stride + ky * dilation - padding, oz * stride + kz * dilation - padding) *
                        w[((oc * cPer + ci) * kh + ky) * kw + kz]
                    }
                  }
                }
                out[((b * cOut + oc) * oh + oy) * ow + oz] = acc
              }
            }
          }
        }
      }
      return { out, shape: [n, cOut, oh, ow] }
    }

    const configs = [
      { stride: 1, padding: 0, dilation: 1, groups: 1 },
      { stride: 2, padding: 1, dilation: 1, groups: 1 },
      { stride: 1, padding: 0, dilation: 2, groups: 1 },
      { stride: 1, padding: 1, dilation: 1, groups: 2 }
    ] as const

    it.effect("conv2d matches a reference implementation across configs", () =>
      Effect.gen(function*() {
        const xData = Array.from({ length: 2 * 4 * 5 * 5 }, (_, i) => ((i * 7) % 13) - 6)
        const wData = Array.from({ length: 6 * 4 * 3 * 3 }, (_, i) => ((i * 5) % 7) - 3)
        for (const cfg of configs) {
          const cPer = 4 / cfg.groups
          const wFlat = wData.slice(0, 6 * cPer * 3 * 3)
          const x = yield* Tensor.fromTypedArray(floats(xData), [2, 4, 5, 5])
          const w = yield* Tensor.fromTypedArray(floats(wFlat), [6, cPer, 3, 3])
          const out = yield* Tensor.conv2d(x, w, cfg)
          const ref = refConv2d(
            xData,
            [2, 4, 5, 5],
            wFlat,
            [6, cPer, 3, 3],
            cfg.stride,
            cfg.padding,
            cfg.dilation,
            cfg.groups
          )
          deep([...out.shape], ref.shape)
          const actual = yield* values(out)
          for (let i = 0; i < ref.out.length; i++) {
            expect(Math.abs(actual[i] - ref.out[i])).toBeLessThan(TOL)
          }
        }
      }))

    it.effect("conv1d", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4]), [1, 1, 4])
        const w = yield* Tensor.fromTypedArray(floats([1, 1]), [1, 1, 2])
        deep(yield* values(yield* Tensor.conv1d(x, w)), [3, 5, 7])
        const strided = yield* Tensor.conv1d(x, w, { stride: 2 })
        deep(yield* values(strided), [3, 7])
      }))

    it.effect("maxPool2d and avgPool2d", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([1, 3, 2, 4, 5, 7, 6, 8, 9]), [1, 1, 3, 3])
        deep(yield* values(yield* Tensor.maxPool2d(x, { kernelSize: 2, stride: 1 })), [5, 7, 8, 9])
        deep(yield* values(yield* Tensor.avgPool2d(x, { kernelSize: 2, stride: 1 })), [
          3.25,
          4.25,
          5.75,
          7.25
        ])
        const padded = yield* Tensor.maxPool2d(x, { kernelSize: 3, stride: 1, padding: 1 })
        deep(padded.shape, [1, 1, 3, 3])
        deep(yield* values(padded), [5, 7, 7, 8, 9, 9, 8, 9, 9])
      }))

    it.effect("convTranspose2d matches a scatter-form reference across configs", () =>
      Effect.gen(function*() {
        const ref = (
          x: Array<number>,
          xShape: [number, number, number, number],
          w: Array<number>,
          wShape: [number, number, number, number],
          stride: number,
          padding: number,
          outputPadding: number
        ) => {
          const [n, cIn, h, wid] = xShape
          const [, cOut, kh, kw] = wShape
          const oh = (h - 1) * stride - 2 * padding + kh + outputPadding
          const ow = (wid - 1) * stride - 2 * padding + kw + outputPadding
          const out = new Array<number>(n * cOut * oh * ow).fill(0)
          for (let b = 0; b < n; b++) {
            for (let ci = 0; ci < cIn; ci++) {
              for (let iy = 0; iy < h; iy++) {
                for (let iz = 0; iz < wid; iz++) {
                  for (let co = 0; co < cOut; co++) {
                    for (let ky = 0; ky < kh; ky++) {
                      for (let kz = 0; kz < kw; kz++) {
                        const oy = iy * stride - padding + ky
                        const oz = iz * stride - padding + kz
                        if (oy >= 0 && oy < oh && oz >= 0 && oz < ow) {
                          out[((b * cOut + co) * oh + oy) * ow + oz] += x[((b * cIn + ci) * h + iy) * wid + iz] *
                            w[((ci * cOut + co) * kh + ky) * kw + kz]
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          return { out, shape: [n, cOut, oh, ow] }
        }
        const xData = Array.from({ length: 1 * 2 * 3 * 4 }, (_, i) => ((i * 7) % 11) - 5)
        const wData = Array.from({ length: 2 * 3 * 2 * 3 }, (_, i) => ((i * 5) % 7) - 3)
        const configs = [
          { stride: 1, padding: 0, outputPadding: 0 },
          { stride: 2, padding: 0, outputPadding: 1 },
          { stride: 2, padding: 1, outputPadding: 0 }
        ] as const
        for (const cfg of configs) {
          const x = yield* Tensor.fromTypedArray(floats(xData), [1, 2, 3, 4])
          const w = yield* Tensor.fromTypedArray(floats(wData), [2, 3, 2, 3])
          const out = yield* Tensor.convTranspose2d(x, w, cfg)
          const expected = ref(xData, [1, 2, 3, 4], wData, [2, 3, 2, 3], cfg.stride, cfg.padding, cfg.outputPadding)
          deep([...out.shape], expected.shape)
          const actual = yield* values(out)
          for (let i = 0; i < expected.out.length; i++) {
            expect(Math.abs(actual[i] - expected.out[i])).toBeLessThan(TOL)
          }
        }
      }))

    it.effect("convTranspose1d", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([1, 2]), [1, 1, 2])
        const w = yield* Tensor.fromTypedArray(floats([1, 1, 1]), [1, 1, 3])
        deep(yield* values(yield* Tensor.convTranspose1d(x, w)), [1, 3, 3, 2])
        const strided = yield* Tensor.convTranspose1d(x, w, { stride: 2 })
        deep(yield* values(strided), [1, 1, 3, 2, 2])
      }))
  })
})

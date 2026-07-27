import { describe, expect, layer } from "@effect/vitest"
import * as assert from "@effect/vitest/utils"
import { Effect, Exit } from "effect"
import { Device, Tensor } from "../src/index.ts"

const values = (t: Tensor.GenericTensor) =>
  Effect.map(Tensor.toTypedArray(t), (arr) => Array.from<number | bigint>(arr).map(Number))

layer(Device.Cpu)("Tensor", (it) => {
  describe("constructors", () => {
    it.effect("zeros/ones/full produce the right values and dtype", () =>
      Effect.gen(function* () {
        assert.deepStrictEqual(yield* values(yield* Tensor.zeros([2, 3])), [0, 0, 0, 0, 0, 0])
        assert.deepStrictEqual(yield* values(yield* Tensor.ones([2, 2], { dtype: "f64" })), [1, 1, 1, 1])
        assert.deepStrictEqual(yield* values(yield* Tensor.full([3], 7, { dtype: "i64" })), [7, 7, 7])
      })
    )

    it.effect("arange with default and explicit step", () =>
      Effect.gen(function* () {
        assert.deepStrictEqual(yield* values(yield* Tensor.arange(5)), [0, 1, 2, 3, 4])
        assert.deepStrictEqual(yield* values(yield* Tensor.arange(1, 10, { step: 2 })), [1, 3, 5, 7, 9])
      })
    )

    it.effect("eye", () =>
      Effect.gen(function* () {
        assert.deepStrictEqual(yield* values(yield* Tensor.eye(2)), [1, 0, 0, 1])
      })
    )

    it.effect("fromTypedArray infers dtype and validates shape", () =>
      Effect.gen(function* () {
        const t = yield* Tensor.fromTypedArray(new Float32Array([1, 2, 3, 4]), [2, 2])
        assert.strictEqual(t.dtype, "f32")
        assert.deepStrictEqual(yield* values(t), [1, 2, 3, 4])
        const u = yield* Tensor.fromTypedArray(new Uint32Array([1, 2, 3]))
        assert.strictEqual(u.dtype, "u32")
        const exit = yield* Effect.exit(Tensor.fromTypedArray(new Float32Array([1, 2, 3]), [2, 2]))
        assert.assertTrue(Exit.isFailure(exit))
      })
    )
  })

  describe("elementwise", () => {
    it.effect("scalar union and broadcasting", () =>
      Effect.gen(function* () {
        const t = yield* Tensor.arange(4)
        assert.deepStrictEqual(yield* values(yield* Tensor.add(t, 10)), [10, 11, 12, 13])
        assert.deepStrictEqual(yield* values(yield* Tensor.mul(t, 2)), [0, 2, 4, 6])
        assert.deepStrictEqual(yield* values(yield* Tensor.sub(t, 1)), [-1, 0, 1, 2])
        assert.deepStrictEqual(yield* values(yield* Tensor.div(t, 2)), [0, 0.5, 1, 1.5])
      })
    )

    it.effect("tensor-tensor broadcasting [2,1] + [1,3]", () =>
      Effect.gen(function* () {
        const a = yield* Tensor.full([2, 1], 1)
        const b = yield* Tensor.full([1, 3], 2)
        const t = yield* Tensor.add(a, b)
        assert.deepStrictEqual(t.shape, [2, 3])
        assert.deepStrictEqual(yield* values(t), [3, 3, 3, 3, 3, 3])
      })
    )

    it.effect("strict dtype: f32 + i64 fails, cast fixes it", () =>
      Effect.gen(function* () {
        const bad = Effect.gen(function* () {
          const a = yield* Tensor.ones([2])
          const b = yield* Tensor.ones([2], { dtype: "i64" })
          return yield* Tensor.add(a, b)
        })
        assert.assertTrue(Exit.isFailure(yield* Effect.exit(bad)))
        const a = yield* Tensor.ones([2])
        const b = yield* Tensor.ones([2], { dtype: "i64" })
        const c = yield* Tensor.cast(b, "f32")
        assert.deepStrictEqual(yield* values(yield* Tensor.add(a, c)), [2, 2])
      })
    )

    it.effect("comparisons return u8", () =>
      Effect.gen(function* () {
        const t = yield* Tensor.gt(yield* Tensor.arange(4), 1)
        assert.strictEqual(t.dtype, "u8")
        assert.deepStrictEqual(yield* values(t), [0, 0, 1, 1])
      })
    )

    it.effect("unary ops", () =>
      Effect.gen(function* () {
        const a = yield* Tensor.fromTypedArray(new Float32Array([-4, 1, 4, 9]))
        const b = yield* Tensor.abs(a)
        assert.deepStrictEqual(yield* values(yield* Tensor.sqrt(b)), [2, 1, 2, 3])
      })
    )

    it.effect("pow", () =>
      Effect.gen(function* () {
        assert.deepStrictEqual(
          yield* values(yield* Tensor.pow(yield* Tensor.arange(3), 2)),
          [0, 1, 4]
        )
      })
    )

    it.effect("tanh and sigmoid", () =>
      Effect.gen(function* () {
        const x = yield* Tensor.fromTypedArray(new Float64Array([-1, 0, 1, 10]))
        const tanhValues = yield* values(yield* Tensor.tanh(x))
        const sigmoidValues = yield* values(yield* Tensor.sigmoid(x))
        for (let i = 0; i < 4; i++) {
          const v = [-1, 0, 1, 10][i]
          expect(Math.abs(tanhValues[i] - Math.tanh(v))).toBeLessThan(1e-12)
          expect(Math.abs(sigmoidValues[i] - 1 / (1 + Math.exp(-v)))).toBeLessThan(1e-12)
        }
      })
    )
  })

  describe("reductions", () => {
    const matrix = Tensor.fromTypedArray(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3])

    it.effect("sum over all dims and specific dims", () =>
      Effect.gen(function* () {
        const m = yield* matrix
        assert.deepStrictEqual(yield* values(yield* Tensor.sum(m)), [21])
        const byRow = yield* Tensor.sum(m, { dims: [1] })
        assert.deepStrictEqual(byRow.shape, [2])
        assert.deepStrictEqual(yield* values(byRow), [6, 15])
      })
    )

    it.effect("keepdims", () =>
      Effect.gen(function* () {
        const t = yield* Tensor.sum(yield* matrix, { dims: [1], keepdims: true })
        assert.deepStrictEqual(t.shape, [2, 1])
      })
    )

    it.effect("mean/max/min", () =>
      Effect.gen(function* () {
        const m = yield* matrix
        assert.deepStrictEqual(yield* values(yield* Tensor.mean(m)), [3.5])
        assert.deepStrictEqual(yield* values(yield* Tensor.max(m, { dims: [0] })), [4, 5, 6])
        assert.deepStrictEqual(yield* values(yield* Tensor.min(m, { dims: [-1] })), [1, 4])
      })
    )

    it.effect("mse", () =>
      Effect.gen(function* () {
        const pred = yield* Tensor.fromTypedArray(new Float64Array([1, 2, 4]), [3])
        const target = yield* Tensor.fromTypedArray(new Float64Array([1, 1, 1]), [3])
        for (const loss of [yield* Tensor.mse(pred, target), yield* Tensor.mse(pred, 1)]) {
          const [value] = yield* values(loss)
          expect(Math.abs(value - 10 / 3)).toBeLessThan(1e-12)
        }
      })
    )
  })

  describe("shape ops", () => {
    const matrix = Tensor.fromTypedArray(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3])

    it.effect("reshape validates numel", () =>
      Effect.gen(function* () {
        const t = yield* Tensor.reshape(yield* matrix, [3, 2])
        assert.deepStrictEqual(t.shape, [3, 2])
        const exit = yield* Effect.exit(Effect.flatMap(matrix, (m) => Tensor.reshape(m, [4, 2])))
        assert.assertTrue(Exit.isFailure(exit))
      })
    )

    it.effect("transpose", () =>
      Effect.gen(function* () {
        const t = yield* Tensor.transpose(yield* matrix, [1, 0])
        assert.deepStrictEqual(t.shape, [3, 2])
        assert.deepStrictEqual(yield* values(t), [1, 4, 2, 5, 3, 6])
      })
    )

    it.effect("slice with negatives and stride", () =>
      Effect.gen(function* () {
        const t = yield* Tensor.slice(yield* matrix, { start: [0, 1], end: [-1, 3] })
        assert.deepStrictEqual(t.shape, [1, 2])
        assert.deepStrictEqual(yield* values(t), [2, 3])
        const strided = yield* Tensor.slice(yield* Tensor.arange(10), { stride: [3] })
        assert.deepStrictEqual(yield* values(strided), [0, 3, 6, 9])
      })
    )

    it.effect("concat along dim", () =>
      Effect.gen(function* () {
        const a = yield* Tensor.ones([2, 1])
        const b = yield* Tensor.full([2, 2], 2)
        const t = yield* Tensor.concat([a, b, a], { dim: 1 })
        assert.deepStrictEqual(t.shape, [2, 4])
        assert.deepStrictEqual(yield* values(t), [1, 2, 2, 1, 1, 2, 2, 1])
      })
    )

    it.effect("broadcastTo", () =>
      Effect.gen(function* () {
        const t = yield* Tensor.broadcastTo(yield* Tensor.ones([1, 3]), [2, 3])
        assert.deepStrictEqual(t.shape, [2, 3])
        assert.deepStrictEqual(yield* values(t), [1, 1, 1, 1, 1, 1])
      })
    )

    it.effect("toNumberArray returns numbers and fails on i64", () =>
      Effect.gen(function* () {
        const t = yield* Tensor.fromTypedArray(new Float64Array([1, 2]), [2])
        assert.deepStrictEqual(yield* Tensor.toNumberArray(t), [1, 2])
        const ints = yield* Tensor.fromTypedArray(new BigInt64Array([1n]), [1])
        const error = yield* Effect.flip(Tensor.toNumberArray(ints))
        expect(error.message).toContain("i64")
      })
    )
  })

  describe("composition", () => {
    it.effect("matmul(eye) roundtrip and deep chains evaluate once", () =>
      Effect.gen(function* () {
        const a = yield* Tensor.fromTypedArray(new Float32Array([1, 2, 3, 4]), [2, 2])
        const id = yield* Tensor.eye(2)
        const b = yield* Tensor.matmul(a, id)
        const c = yield* Tensor.add(b, 1)
        const d = yield* Tensor.sum(c)
        expect(Array.from<number | bigint>(yield* Tensor.toTypedArray(d))).toEqual([14])
      })
    )

    it.effect("batched matmul broadcasts batch dims", () =>
      Effect.gen(function* () {
        const a = yield* Tensor.fromTypedArray(
          new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
          [2, 2, 3]
        )
        const b = yield* Tensor.fromTypedArray(new Float64Array([1, 2, 3, 4, 5, 6]), [3, 2])
        const out = yield* Tensor.matmul(a, b)
        assert.deepStrictEqual(out.shape, [2, 2, 2])
        assert.deepStrictEqual(yield* values(out), [22, 28, 49, 64, 76, 100, 103, 136])
      })
    )
  })
})

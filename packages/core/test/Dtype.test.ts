import { describe, expect } from "@effect/vitest"
import { Effect } from "effect"
import { Tensor } from "../src/index.ts"
import { deep, onDevices } from "./utils/devices.ts"

const floats = (values: ReadonlyArray<number>) => new Float32Array(values)

const as = (dtype: Tensor.DType, values: ReadonlyArray<number>, shape: ReadonlyArray<number>) =>
  Effect.flatMap(Tensor.fromTypedArray(floats(values), shape), (tensor) => Tensor.cast(tensor, dtype))

// bf16 has seven stored fraction bits, or eight significant bits including the
// implicit leading bit. This bound also covers the finer f16 round-trips below.
const BF16_TOL = 1e-2

onDevices("Dtype", (device) => (it) => {
  describe("compute dtypes", () => {
    it.effect("bf16 supports elementwise, reduction, and matmul operations", () =>
      Effect.gen(function*() {
        const a = yield* as("bf16", [1, 2, 3, 4, 5, 6], [2, 3])
        const b = yield* as("bf16", [0.5, 1, 1.5, 2, 2.5, 3], [2, 3])
        expect(a.dtype).toBe("bf16")
        const [sum] = yield* Tensor.compute([yield* Tensor.add(a, b)])
        expect(sum.dtype).toBe("bf16")
        deep(yield* Tensor.toNumberArray(sum), [1.5, 3, 4.5, 6, 7.5, 9])
        const [soft] = yield* Tensor.compute([yield* Tensor.softmax(a)])
        const probs = yield* Tensor.toNumberArray(soft)
        expect(probs[0] + probs[1] + probs[2]).toBeCloseTo(1, 2)
        const m = yield* as("bf16", [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0], [3, 4])
        if (device !== "cpu") {
          const [product] = yield* Tensor.compute([yield* Tensor.matmul(a, m)])
          expect(product.dtype).toBe("bf16")
        } else {
          // Candle's Accelerate CPU backend has no bf16 GEMM. It should return
          // a typed error, not crash or silently upcast.
          const error = yield* Effect.flip(Effect.gen(function*() {
            return yield* Tensor.compute([yield* Tensor.matmul(a, m)])
          }))
          expect(error._tag).toBe("TensorError")
          expect(error.message).toMatch(/bf16|bfloat/i)
        }
      }))

    it.effect("bf16 random init and f32 round-trip", () =>
      Effect.gen(function*() {
        const [draw] = yield* Tensor.compute([yield* Tensor.randn([8, 8], { dtype: "bf16" })])
        expect(draw.dtype).toBe("bf16")
        const [wide] = yield* Tensor.compute([yield* Tensor.cast(draw, "f32")])
        const values = yield* Tensor.toNumberArray(wide)
        expect(values.some((v) => v !== 0)).toBe(true)
        const [back] = yield* Tensor.compute([yield* Tensor.cast(wide, "bf16")])
        expect(back.dtype).toBe("bf16")
      }))

    it.effect("f16 matmul works on Metal and fails with TensorError on CPU", () =>
      Effect.gen(function*() {
        const a = yield* as("f16", [1, 2, 3, 4, 5, 6], [2, 3])
        const m = yield* as("f16", [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0], [3, 4])
        if (device !== "cpu") {
          const [product] = yield* Tensor.compute([yield* Tensor.matmul(a, m)])
          expect(product.dtype).toBe("f16")
          deep(yield* Tensor.toNumberArray(product), [1, 5, 5, 1, 4, 11, 11, 4])
        } else {
          const error = yield* Effect.flip(Effect.gen(function*() {
            return yield* Tensor.compute([yield* Tensor.matmul(a, m)])
          }))
          expect(error._tag).toBe("TensorError")
          expect(error.message).toMatch(/f16/)
        }
      }))

    it.effect("a 0-d float scalar never promotes a float tensor's dtype", () =>
      Effect.gen(function*() {
        const t = yield* as("bf16", [1, 2, 3, 4], [2, 2])
        const scalar = yield* Tensor.constant(0.5, { dtype: "f32" })
        const [left] = yield* Tensor.compute([yield* Tensor.mul(scalar, t)])
        expect(left.dtype).toBe("bf16")
        deep(yield* Tensor.toNumberArray(left), [0.5, 1, 1.5, 2])
        const [right] = yield* Tensor.compute([yield* Tensor.mul(t, scalar)])
        expect(right.dtype).toBe("bf16")
        // bf16 reductions keep bf16 storage (f32 accumulation in-kernel)
        const [summed] = yield* Tensor.compute([yield* Tensor.sum(t, { dims: [0] })])
        expect(summed.dtype).toBe("bf16")
        deep(yield* Tensor.toNumberArray(summed), [4, 6])
      }))

    it.effect("guarded ops state their dtype requirements", () =>
      Effect.gen(function*() {
        const q = yield* as("f16", [1, 0, 0, 1], [1, 1, 2, 2])
        const error = yield* Effect.flip(Effect.gen(function*() {
          return yield* Tensor.compute([yield* Tensor.scaledDotProductAttention(q, q, q, { causal: true })])
        }))
        expect(error.message).toMatch(/dtype must be f32, f64 or bf16, got f16/)
        // bf16 cross-entropy runs on both devices (fused kernel with f32
        // accumulation on Metal, composed on CPU)
        const logits = yield* as("bf16", [1, 2, 3, 4, 5, 6], [2, 3])
        const target = yield* Tensor.fromTypedArray(new Uint32Array([0, 2]), [2])
        const [loss] = yield* Tensor.compute([yield* Tensor.crossEntropy(logits, { target })])
        const [v] = yield* Tensor.toNumberArray(loss)
        expect(v).toBeGreaterThan(0)
      }))
  })

  describe("strictness", () => {
    it.effect("i64 preserves the full signed range through arithmetic and comparison", () =>
      Effect.gen(function*() {
        if (device === "metal") return
        const values = new BigInt64Array([
          9_007_199_254_740_993n,
          -9_007_199_254_740_993n,
          9_223_372_036_854_775_805n,
          -9_223_372_036_854_775_806n
        ])
        const deltas = new BigInt64Array([2n, -2n, 1n, -1n])
        const expected = new BigInt64Array([
          9_007_199_254_740_995n,
          -9_007_199_254_740_995n,
          9_223_372_036_854_775_806n,
          -9_223_372_036_854_775_807n
        ])
        const input = yield* Tensor.fromTypedArray(values, [4])
        const delta = yield* Tensor.fromTypedArray(deltas, [4])
        const expectedTensor = yield* Tensor.fromTypedArray(expected, [4])
        const sum = yield* Tensor.add(input, delta)
        const [sumValue, comparison] = yield* Tensor.compute([sum, yield* Tensor.eq(sum, expectedTensor)])

        const inputValues = yield* Tensor.toTypedArray(input)
        const sumValues = yield* Tensor.toTypedArray(sumValue)
        const comparisonValues = yield* Tensor.toTypedArray(comparison)
        if (!(inputValues instanceof BigInt64Array) || !(sumValues instanceof BigInt64Array)) {
          throw new Error("i64 readback must use BigInt64Array")
        }
        if (!(comparisonValues instanceof Uint8Array)) {
          throw new Error("comparison readback must use Uint8Array")
        }
        expect(Array.from(inputValues)).toEqual(Array.from(values))
        expect(Array.from(sumValues)).toEqual(Array.from(expected))
        expect(Array.from(comparisonValues)).toEqual([1, 1, 1, 1])
      }))

    it.effect("mixed-dtype binary ops fail with the explicit remedy", () =>
      Effect.gen(function*() {
        const a = yield* Tensor.fromTypedArray(floats([1]), [1])
        const b = yield* as(device === "metal" ? "f16" : "bf16", [1], [1])
        const error = yield* Effect.flip(Effect.gen(function*() {
          return yield* Tensor.compute([yield* Tensor.add(a, b)])
        }))
        expect(error._tag).toBe("TensorError")
        expect(error.message).toMatch(/dtype mismatch/)
        expect(error.message).toMatch(/cast/)
      }))

    it.effect("cast is total across the compute dtypes on device", () =>
      Effect.gen(function*() {
        const base = yield* Tensor.fromTypedArray(floats([1.5, -2.25, 3.75]), [3])
        const dtypes: ReadonlyArray<Tensor.DType> = device === "metal" ? ["f16", "bf16"] : ["f64", "f16", "bf16"]
        for (const dtype of dtypes) {
          const narrowed = yield* Tensor.cast(base, dtype)
          expect(narrowed.dtype).toBe(dtype)
          const [round] = yield* Tensor.compute([yield* Tensor.cast(narrowed, "f32")])
          const values = yield* Tensor.toNumberArray(round)
          for (let i = 0; i < 3; i++) {
            expect(Math.abs(values[i]! - [1.5, -2.25, 3.75][i]!)).toBeLessThan(BF16_TOL)
          }
        }
      }))
  })

  describe("device capability matrix", () => {
    it.effect("f64 computes fully on CPU and is rejected at construction on Metal", () =>
      Effect.gen(function*() {
        if (device !== "metal") {
          const a = yield* Tensor.fromTypedArray(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3])
          const m = yield* Tensor.fromTypedArray(new Float64Array([1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0]), [3, 4])
          const [product] = yield* Tensor.compute([yield* Tensor.matmul(a, m)])
          deep(yield* Tensor.toNumberArray(product), [1, 5, 5, 1, 4, 11, 11, 4])
          const [soft] = yield* Tensor.compute([yield* Tensor.softmax(a)])
          expect(soft.dtype).toBe("f64")
          return
        }
        // Every construction path fails immediately with the matrix error.
        // Nothing is deferred to compute time.
        const expected = /dtype f64 is not supported on device metal/
        const fromArray = yield* Effect.flip(
          Effect.gen(function*() {
            const t = yield* Tensor.fromTypedArray(new Float64Array([1, 2]), [2])
            return yield* Tensor.compute([t])
          })
        )
        expect(fromArray.message).toMatch(expected)
        const viaCast = yield* Effect.flip(
          Effect.gen(function*() {
            const t = yield* Tensor.fromTypedArray(floats([1, 2]), [2])
            const wide = yield* Tensor.cast(t, "f64")
            return yield* Tensor.compute([wide])
          })
        )
        expect(viaCast.message).toMatch(expected)
        const viaRandn = yield* Effect.flip(Effect.gen(function*() {
          return yield* Tensor.compute([yield* Tensor.randn([4], { dtype: "f64" })])
        }))
        expect(viaRandn.message).toMatch(expected)
        const viaZeros = yield* Effect.flip(Effect.gen(function*() {
          return yield* Tensor.compute([yield* Tensor.zeros([4], { dtype: "f64" })])
        }))
        expect(viaZeros.message).toMatch(expected)
      }))
  })

  describe("half-precision interop", () => {
    it.effect("Float16Array round-trips through fromTypedArray and toTypedArray", () =>
      Effect.gen(function*() {
        if (!Object.hasOwn(globalThis, "Float16Array")) {
          return
        }
        const input = new globalThis.Float16Array([1.5, -2.5, 3.5])
        const tensor = yield* Tensor.fromTypedArray(input, [3])
        expect(tensor.dtype).toBe("f16")
        const back = yield* Tensor.toTypedArray(tensor)
        if (!(back instanceof Float32Array)) throw new Error("f16 readback must use Float32Array")
        const values = Array.from(back)
        deep(values, [1.5, -2.5, 3.5])
      }))

    it.effect("bf16 reads back as f32 values", () =>
      Effect.gen(function*() {
        const tensor = yield* as("bf16", [1.5, -2.5, 3.5], [3])
        const values = yield* Tensor.toNumberArray(tensor)
        for (let i = 0; i < 3; i++) {
          expect(Math.abs(values[i]! - [1.5, -2.5, 3.5][i]!)).toBeLessThan(BF16_TOL)
        }
      }))
  })
})

import { expect } from "@effect/vitest"
import { Effect, Fiber } from "effect"
import { Gradient, Tensor } from "../src/index.ts"
import { onDevices } from "./utils/devices.ts"

const reference = (values: Float32Array, width: number, k: number): Array<number> => {
  const result: Array<number> = []
  for (let row = 0; row < values.length / width; row++) {
    const indices = Array.from({ length: width }, (_, i) => i)
    indices.sort((a, b) => values[row * width + b] - values[row * width + a] || a - b)
    result.push(...indices.slice(0, k))
  }
  return result
}

onDevices("topKIndices", (device) => (it) => {
  for (const optimize of [false, true]) {
    it.effect(`top-8 across rank-3 router rows, optimize=` + optimize, () =>
      Effect.gen(function*() {
        const scores = Float32Array.from({ length: 2 * 3 * 128 }, (_, i) => ((i * 73 + 19) % 137) - 68)
        const source = yield* Tensor.fromTypedArray(scores, [2, 3, 128])
        const input = yield* Tensor.makeInput(0, source)
        const indices = yield* Tensor.topKIndices(input, 8)
        expect(indices.dtype).toBe("u32")
        expect(indices.shape).toEqual([2, 3, 8])
        const selected = yield* Tensor.gather(input, indices, { dim: 2 })
        const program = yield* Tensor.freezeProgram([indices, selected], { optimize })
        const outputs = yield* Tensor.runProgram(program, [source])
        const ids = yield* Tensor.toTypedArray(outputs[0])
        const expected = reference(scores, 128, 8)
        expect(ids).toBeInstanceOf(Uint32Array)
        expect(Array.from<number | bigint>(ids)).toEqual(expected)
        expect(yield* Tensor.toNumberArray(outputs[1])).toEqual(
          expected.map((index, slot) => scores[Math.floor(slot / 8) * 128 + index])
        )
        yield* Tensor.clearAll(outputs)
      }))

    it.effect(
      `stable ties, signed zero, infinities and subnormals, optimize=` + optimize,
      () =>
        Effect.gen(function*() {
          const scores = new Float32Array([3, 3, -0, 0, -Infinity, Infinity, 2 ** -149, -(2 ** -149), 3])
          const source = yield* Tensor.fromTypedArray(scores)
          for (const k of [1, 4, scores.length]) {
            const root = yield* source.pipe(Tensor.topKIndices(k))
            const program = yield* Tensor.freezeProgram([root], { optimize })
            const [out] = yield* Tensor.runProgram(program, [])
            expect(out.shape).toEqual([k])
            expect(out.dtype).toBe("u32")
            expect(yield* Tensor.toNumberArray(out)).toEqual(reference(scores, scores.length, k))
            yield* Tensor.clear(out)
          }
        })
    )

    it.effect(`supports strided views and zero leading dimensions, optimize=` + optimize, () =>
      Effect.gen(function*() {
        const source = yield* Tensor.fromTypedArray(new Float32Array([1, 5, 9, 7, 3, 2]), [3, 2])
        const view = yield* Tensor.transpose(source, [1, 0])
        const root = yield* Tensor.topKIndices(view, 2)
        const empty = yield* Tensor.topKIndices(yield* Tensor.zeros([2, 0, 8], { dtype: "f32" }), 3)
        const program = yield* Tensor.freezeProgram([root, empty], { optimize })
        const outputs = yield* Tensor.runProgram(program, [])
        expect(yield* Tensor.toNumberArray(outputs[0])).toEqual([1, 2, 1, 0])
        expect(outputs[1].shape).toEqual([2, 0, 3])
        expect(yield* Tensor.toTypedArray(outputs[1])).toEqual(new Uint32Array())
        yield* Tensor.clearAll(outputs)
      }))

    it.effect(
      `NaN fails without poisoning reusable outputs or status, optimize=` + optimize,
      () =>
        Effect.gen(function*() {
          const source = yield* Tensor.fromTypedArray(new Float32Array([3, 1, 2, 4]))
          const input = yield* Tensor.makeInput(0, source)
          const root = yield* Tensor.topKIndices(input, 2)
          const program = yield* Tensor.freezeProgram([root, yield* Tensor.gather(input, root, { dim: 0 })], {
            optimize
          })
          const retained = yield* Tensor.runProgram(program, [source])
          for (const scores of [[NaN, 1, 2, 4], [3, 1, 2, NaN]]) {
            const bad = yield* Tensor.fromTypedArray(new Float32Array(scores))
            const error = yield* Effect.flip(Tensor.runProgram(program, [bad]))
            expect(error._tag).toBe("TensorError")
            expect(error.message).toContain("NaN")
          }
          const next = yield* Tensor.runProgram(program, [source])
          expect(yield* Tensor.toNumberArray(next[0])).toEqual([3, 0])
          yield* Tensor.clearAll(next)
          expect(yield* Tensor.toNumberArray(retained[0])).toEqual([3, 0])
          expect(yield* Tensor.toNumberArray(retained[1])).toEqual([4, 3])
          yield* Tensor.clearAll(retained)
        })
    )
  }

  it.effect("rejects invalid k, scalar and empty-width inputs, and unsupported dtypes", () =>
    Effect.gen(function*() {
      const source = yield* Tensor.zeros([2, 8], { dtype: "f32" })
      for (const k of [0, -1, 9, 1.5, NaN, Infinity]) {
        const error = yield* Effect.flip(Tensor.topKIndices(source, k))
        expect(error._tag).toBe("TensorError")
        expect(error.message).toContain("k must be")
      }
      for (const shape of [[], [0], [2, 0]]) {
        const value = yield* Tensor.zeros(shape, { dtype: "f32" })
        expect((yield* Effect.flip(Tensor.topKIndices(value, 1)))._tag).toBe("TensorError")
      }
      for (const dtype of ["u32", "i64", "f16", "bf16", "f64"] as const) {
        if (device === "metal" && dtype === "f64") continue
        const value = yield* Tensor.zeros([8], { dtype })
        const error = yield* Effect.flip(Tensor.topKIndices(value, 1))
        expect(error._tag).toBe("TensorError")
        expect(error.message).toContain("expected f32")
      }
    }))

  it.effect("allows explicit value-exact BF16 widening before selection", () =>
    Effect.gen(function*() {
      const source = yield* Tensor.fromTypedArray(new Float32Array([1, 4, 4, 2, -8]))
      const half = yield* Tensor.cast(source, "bf16")
      const widened = yield* Tensor.cast(half, "f32")
      const [out] = yield* Tensor.compute([yield* Tensor.topKIndices(widened, 3)])
      expect(yield* Tensor.toNumberArray(out)).toEqual([1, 2, 3])
      yield* Tensor.clear(out)
    }))

  it.effect("interrupted native invocations leave the program reusable", () =>
    Effect.gen(function*() {
      const [source] = yield* Tensor.compute([yield* Tensor.ones([64, 128], { dtype: "f32" })])
      const input = yield* Tensor.makeInput(0, source)
      const program = yield* Tensor.freezeProgram([yield* Tensor.topKIndices(input, 8)])
      const fiber = yield* Tensor.runProgram(program, [source]).pipe(
        Effect.tap(Tensor.clearAll),
        Effect.forkChild({ startImmediately: true })
      )
      yield* Fiber.interrupt(fiber)
      const [result] = yield* Tensor.runProgram(program, [source])
      expect(yield* Tensor.toNumberArray(result)).toEqual(Array.from({ length: 64 * 8 }, (_, i) => i % 8))
      yield* Tensor.clearAll([result, source])
    }))

  it.effect("compiled captures survive source release and outputs own their storage", () =>
    Effect.gen(function*() {
      const [source] = yield* Tensor.compute([yield* Tensor.fromTypedArray(new Float32Array([2, 7, 7, 1]))])
      const root = yield* Tensor.topKIndices(source, 3)
      const program = yield* Tensor.freezeProgram([root])
      yield* Tensor.clear(source)
      const [first] = yield* Tensor.runProgram(program, [])
      const [second] = yield* Tensor.runProgram(program, [])
      yield* Tensor.clear(second)
      expect(yield* Tensor.toNumberArray(first)).toEqual([1, 2, 0])
      yield* Tensor.clear(first)
      yield* Tensor.clear(first)
      const error = yield* Effect.flip(Tensor.topKIndices(source, 1))
      expect(error._tag).toBe("TensorError")
    }))

  it.effect("rejects differentiation through indices with a typed error", () =>
    Effect.gen(function*() {
      const source = yield* Tensor.fromTypedArray(new Float32Array([2, 1, 3]))
      const indices = yield* Tensor.topKIndices(source, 2)
      const loss = yield* Tensor.sum(yield* Tensor.cast(indices, "f32"))
      const error = yield* Effect.flip(Gradient.grad(loss, [source]))
      expect(error._tag).toBe("TensorError")
      expect(error.message).toContain("topKIndices is not differentiable")
    }))
})

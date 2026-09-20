import { expect } from "@effect/vitest"
import { Effect, Fiber } from "effect"
import { Gradient, Tensor } from "../src/index.ts"
import { onDevices } from "./utils/devices.ts"

const dense = (values: ReadonlyArray<number>, shape: ReadonlyArray<number>, dtype: "f32" | "bf16") =>
  Effect.gen(function*() {
    const source = yield* Tensor.fromTypedArray(new Float32Array(values), shape)
    const [result] = yield* Tensor.compute([dtype === "f32" ? source : yield* Tensor.cast(source, dtype)])
    return result
  })

const bf16 = (value: number): number => {
  const bytes = new ArrayBuffer(4)
  const f = new Float32Array(bytes)
  const u = new Uint32Array(bytes)
  f[0] = value
  u[0] = (u[0] + 0x7fff + ((u[0] >>> 16) & 1)) & 0xffff0000
  return f[0]
}

const reference = (
  x: ReadonlyArray<number>,
  w: ReadonlyArray<number>,
  routes: ReadonlyArray<number>,
  inner: number,
  columns: number,
  dtype: "f32" | "bf16"
): Array<number> =>
  routes.flatMap((expert, row) =>
    Array.from({ length: columns }, (_, column) => {
      let sum = 0
      for (let i = 0; i < inner; i++) sum += x[row * inner + i] * w[(expert * columns + column) * inner + i]
      return dtype === "bf16" ? bf16(sum) : sum
    })
  )

onDevices("expertLinearRows", (device) => (it) => {
  for (const dtype of ["f32", "bf16"] as const) {
    for (const optimize of [false, true]) {
      it.effect(`${dtype} selected expert dots and native storage, optimize=${optimize}`, () =>
        Effect.gen(function*() {
          for (const [rows, inner, columns] of [[16, 13, 17], [64, 65, 13], [1025, 13, 9]]) {
            const experts = 3
            const xs = Array.from({ length: rows * inner }, (_, i) => i % 7 - 3)
            const ws = Array.from({ length: experts * columns * inner }, (_, i) => i % 11 - 5)
            const x = yield* dense(xs, [rows, inner], dtype)
            const w = yield* dense(ws, [experts, columns, inner], dtype)
            const routes = Array.from({ length: rows }, (_, row) => row % 9 === 0 ? 0 : row % 7 === 0 ? 1 : 2)
            const ids = yield* Tensor.fromTypedArray(new Uint32Array(routes))
            const root = yield* Tensor.expertLinearRows(
              yield* Tensor.makeInput(0, x),
              yield* Tensor.makeInput(1, w),
              yield* Tensor.makeInput(2, ids)
            )
            const program = yield* Tensor.freezeProgram([root], { optimize })
            const diagnostics = program.handle.diagnostics
            expect(root.shape).toEqual([rows, columns])
            expect(root.dtype).toBe(dtype)
            expect(diagnostics.legalization!.materializedConversions).toBe(0)
            expect(diagnostics.legalization!.materializedConversionBytes).toBe(0)
            expect(diagnostics.legalization!.legalizedLoweringUnits).toBe(0)
            expect(diagnostics.memory.externalBytes).toBe(
              (xs.length + ws.length) * (dtype === "bf16" ? 2 : 4) + rows * 4
            )
            expect(diagnostics.memory.workspaceBytes).toBeLessThanOrEqual(256)
            const [out] = yield* Tensor.runProgram(program, [x, w, ids])
            expect(out.dtype).toBe(dtype)
            expect(yield* Tensor.toNumberArray(out)).toEqual(reference(xs, ws, routes, inner, columns, dtype))
            yield* Tensor.clearAll([out, x, w])
          }
        }))

      it.effect(`${dtype} invalid routes fail and the program recovers, optimize=${optimize}`, () =>
        Effect.gen(function*() {
          const x = yield* dense([1, 2, 3, 4], [2, 2], dtype)
          const w = yield* dense([1, 0, 0, 1, 2, 0, 0, 3], [2, 2, 2], dtype)
          const ids = yield* Tensor.fromTypedArray(new Uint32Array([0, 1]))
          const root = yield* Tensor.expertLinearRows(
            yield* Tensor.makeInput(0, x),
            yield* Tensor.makeInput(1, w),
            yield* Tensor.makeInput(2, ids)
          )
          const program = yield* Tensor.freezeProgram([root], { optimize })
          const [retained] = yield* Tensor.runProgram(program, [x, w, ids])
          for (const bad of [[2, 1], [0, 0xffffffff]]) {
            const invalid = yield* Tensor.fromTypedArray(new Uint32Array(bad))
            const error = yield* Effect.flip(Tensor.runProgram(program, [x, w, invalid]))
            expect(error._tag).toBe("TensorError")
            expect(error.message).toContain("index is out of range")
          }
          const flipped = yield* Tensor.fromTypedArray(new Uint32Array([1, 0]))
          const [next] = yield* Tensor.runProgram(program, [x, w, flipped])
          expect(yield* Tensor.toNumberArray(next)).toEqual([2, 6, 3, 4])
          yield* Tensor.clear(next)
          expect(yield* Tensor.toNumberArray(retained)).toEqual([1, 2, 6, 12])
          yield* Tensor.clearAll([x, w, retained])
        }))

      it.effect(`${dtype} zero N/I/O and invalid routes with O=0, optimize=${optimize}`, () =>
        Effect.gen(function*() {
          for (const [rows, inner, columns] of [[0, 13, 4], [3, 0, 4], [3, 13, 0], [0, 0, 0]]) {
            const x = yield* Tensor.zeros([rows, inner], { dtype })
            const w = yield* Tensor.zeros([2, columns, inner], { dtype })
            const ids = yield* Tensor.fromTypedArray(new Uint32Array(rows))
            const root = yield* Tensor.expertLinearRows(x, w, yield* Tensor.makeInput(0, ids))
            const program = yield* Tensor.freezeProgram([root], { optimize })
            const [out] = yield* Tensor.runProgram(program, [ids])
            expect(out.shape).toEqual([rows, columns])
            expect(yield* Tensor.toNumberArray(out)).toEqual(Array(rows * columns).fill(0))
            yield* Tensor.clear(out)
            if (rows > 0 && columns === 0) {
              const invalid = yield* Tensor.fromTypedArray(new Uint32Array(rows).fill(2))
              const error = yield* Effect.flip(Tensor.runProgram(program, [invalid]))
              expect(error._tag).toBe("TensorError")
              expect(error.message).toContain("index is out of range")
              const [recovered] = yield* Tensor.runProgram(program, [ids])
              expect(yield* Tensor.toNumberArray(recovered)).toEqual([])
              yield* Tensor.clear(recovered)
            }
          }
        }))
    }
  }

  it.effect("handles offset and strided inputs, weights and routes", () =>
    Effect.gen(function*() {
      for (const dtype of ["f32", "bf16"] as const) {
        const x = yield* dense([99, 1, 2, 99, 3, 4], [2, 3], dtype)
        const w = yield* dense([99, 1, 2, 99, 3, 4, 99, 5, 6, 99, 7, 8], [2, 2, 3], dtype)
        const ids = yield* Tensor.fromTypedArray(new Uint32Array([99, 1, 99, 0]))
        const root = yield* Tensor.expertLinearRows(
          yield* Tensor.slice(x, { start: [0, 1] }),
          yield* Tensor.transpose(yield* Tensor.slice(w, { start: [0, 0, 1] }), [0, 2, 1]),
          yield* Tensor.slice(ids, { start: [1], stride: [2] })
        )
        for (const optimize of [false, true]) {
          const program = yield* Tensor.freezeProgram([root], { optimize })
          const [out] = yield* Tensor.runProgram(program, [])
          expect(yield* Tensor.toNumberArray(out)).toEqual([19, 22, 15, 22])
          yield* Tensor.clear(out)
        }
        yield* Tensor.clearAll([x, w])
      }
    }))

  it.effect("BF16 accumulates in F32 and rounds only once, ties to even", () =>
    Effect.gen(function*() {
      const x = yield* dense([1, 1, 1], [1, 3], "bf16")
      const w = yield* dense([256, 1, 0, 256, 1, 0.75, 256, 3, 0, -256, -1, 0, 1, 2 ** -8, 0], [1, 5, 3], "bf16")
      const ids = yield* Tensor.fromTypedArray(new Uint32Array([0]))
      for (const optimize of [false, true]) {
        const program = yield* Tensor.freezeProgram([yield* Tensor.expertLinearRows(x, w, ids)], { optimize })
        const [out] = yield* Tensor.runProgram(program, [])
        expect(yield* Tensor.toNumberArray(out)).toEqual([256, 258, 260, -256, 1])
        yield* Tensor.clear(out)
      }
      yield* Tensor.clearAll([x, w])
    }))

  it.effect("uses device-generated U32 routes without gathering the bank", () =>
    Effect.gen(function*() {
      const scores = yield* Tensor.fromTypedArray(new Float32Array([0, 2, 3, 0, 0, 1]), [3, 2])
      const ids = yield* Tensor.reshape(yield* Tensor.topKIndices(scores, 1), [3])
      const x = yield* Tensor.fromTypedArray(new Float32Array([1, 2, 3, 4, 5, 6]), [3, 2])
      const w = yield* Tensor.fromTypedArray(new Float32Array([1, 0, 0, 1, 2, 0, 0, 3]), [2, 2, 2])
      const [out] = yield* Tensor.compute([yield* Tensor.expertLinearRows(x, w, ids)])
      expect(yield* Tensor.toNumberArray(out)).toEqual([2, 6, 3, 4, 10, 18])
      yield* Tensor.clear(out)
    }))

  it.effect("compiles a BF16 bank above 2^32 elements without materializing it", () =>
    Effect.gen(function*() {
      // Metadata-only inputs: no 16-GiB bank is allocated or uploaded.
      const x = yield* Tensor.makeInput(0, yield* Tensor.zeros([1, 65536], { dtype: "bf16" }))
      const w = yield* Tensor.makeInput(1, yield* Tensor.zeros([2, 65536, 65536], { dtype: "bf16" }))
      const ids = yield* Tensor.makeInput(2, yield* Tensor.zeros([1], { dtype: "u32" }))
      const program = yield* Tensor.freezeProgram([yield* Tensor.expertLinearRows(x, w, ids)])
      const diagnostics = program.handle.diagnostics
      expect(diagnostics.memory.externalBytes).toBe((65536 + 2 * 65536 * 65536) * 2 + 4)
      expect(diagnostics.memory.workspaceBytes).toBeLessThanOrEqual(256)
      expect(diagnostics.memory.persistentBytes).toBeLessThan(1024)
      expect(diagnostics.legalization!.materializedConversionBytes).toBe(0)
      expect(diagnostics.legalization!.materializedConversions).toBe(0)
    }))

  it.effect("rejects invalid ranks, dimensions, route dtype and floating dtype", () =>
    Effect.gen(function*() {
      for (
        const [xs, ws, ids] of [
          [[2], [2, 3, 2], [1]],
          [[1, 2], [3, 2], [1]],
          [[1, 2], [2, 3, 2], [1, 1]],
          [[2, 2], [2, 3, 2], [1]],
          [[1, 2], [2, 3, 4], [1]],
          [[1, 2], [0, 3, 2], [1]]
        ]
      ) {
        const x = yield* Tensor.zeros(xs, { dtype: "f32" })
        const w = yield* Tensor.zeros(ws, { dtype: "f32" })
        const i = yield* Tensor.zeros(ids, { dtype: "u32" })
        expect((yield* Effect.flip(Tensor.expertLinearRows(x, w, i)))._tag).toBe("TensorError")
      }
      const ids = yield* Tensor.zeros([1], { dtype: "u32" })
      for (const dtype of ["f16", "f64", "u32"] as const) {
        if (device === "metal" && dtype === "f64") continue
        const x = yield* Tensor.zeros([1, 2], { dtype })
        const w = yield* Tensor.zeros([2, 3, 2], { dtype })
        expect((yield* Effect.flip(Tensor.expertLinearRows(x, w, ids))).message).toContain("expected f32 or bf16")
      }
      const x = yield* Tensor.zeros([1, 2], { dtype: "f32" })
      const w = yield* Tensor.zeros([2, 3, 2], { dtype: "f32" })
      expect((yield* Effect.flip(Tensor.expertLinearRows(x, w, yield* Tensor.zeros([1], { dtype: "i64" })))).message)
        .toContain("indices must be u32")
      expect((yield* Effect.flip(Tensor.expertLinearRows(x, yield* Tensor.cast(w, "bf16"), ids))).message)
        .toContain("dtype mismatch")
      const loss = yield* Tensor.sum(yield* Tensor.expertLinearRows(x, w, ids))
      expect((yield* Effect.flip(Gradient.grad(loss, [x]))).message).toContain("not differentiable")
    }))

  it.effect("captures survive source release, borrowed bindings survive interruption", () =>
    Effect.gen(function*() {
      const x = yield* dense([1, 2, 3, 4], [2, 2], "bf16")
      const w = yield* dense([1, 0, 0, 1, 2, 0, 0, 3], [2, 2, 2], "bf16")
      const [ids] = yield* Tensor.compute([yield* Tensor.fromTypedArray(new Uint32Array([0, 1]))])
      const captured = yield* Tensor.freezeProgram([yield* Tensor.expertLinearRows(x, w, ids)])
      const bound = yield* Tensor.freezeProgram([
        yield* Tensor.expertLinearRows(
          yield* Tensor.makeInput(0, x),
          yield* Tensor.makeInput(1, w),
          yield* Tensor.makeInput(2, ids)
        )
      ])
      const fiber = yield* Tensor.runProgram(bound, [x, w, ids]).pipe(
        Effect.tap(Tensor.clearAll),
        Effect.forkChild({ startImmediately: true })
      )
      yield* Fiber.interrupt(fiber)
      const [borrowed] = yield* Tensor.runProgram(bound, [x, w, ids])
      expect(yield* Tensor.toNumberArray(w)).toEqual([1, 0, 0, 1, 2, 0, 0, 3])
      const [completed] = yield* Tensor.runProgram(bound, [x, w, ids])
      // Invocation bindings are borrowed until completion. Returned outputs
      // and compiled captures have their own ownership after that point.
      yield* Tensor.clearAll([x, w, ids])
      expect(yield* Tensor.toNumberArray(completed)).toEqual([1, 2, 6, 12])
      yield* Tensor.clear(completed)
      const [first] = yield* Tensor.runProgram(captured, [])
      const [second] = yield* Tensor.runProgram(captured, [])
      yield* Tensor.clear(second)
      expect(yield* Tensor.toNumberArray(first)).toEqual([1, 2, 6, 12])
      expect(yield* Tensor.toNumberArray(borrowed)).toEqual([1, 2, 6, 12])
      yield* Tensor.clearAll([first, borrowed])
      expect((yield* Effect.flip(Tensor.runProgram(bound, [x, w, ids])))._tag).toBe("TensorError")
    }))
})

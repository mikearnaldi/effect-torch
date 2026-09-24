import { expect } from "@effect/vitest"
import { Effect, Fiber } from "effect"
import { Gradient, Runtime, Tensor } from "../src/index.ts"
import { onDevices } from "./utils/devices.ts"

const values = (data: ReadonlyArray<number>, shape: ReadonlyArray<number>, dtype: "f32" | "bf16") =>
  Tensor.fromTypedArray(Float32Array.from(data), shape).pipe(Effect.flatMap(Tensor.cast(dtype)))

// Independently gather each exact-size group, run ordinary linearRows, then
// reconstruct original row order. The projection never sees padded rows.
const groupedReference = (x: Tensor.Any, bank: Tensor.Any, routes: ReadonlyArray<number>) =>
  Effect.gen(function*() {
    const rows: Array<Tensor.Any> = []

    for (let expert = 0; expert < bank.shape[0]; expert++) {
      const selected = routes.flatMap((id, row) => id === expert ? [row] : [])

      if (selected.length === 0) continue

      const ids = yield* Tensor.fromTypedArray(Uint32Array.from(selected))

      const weight = yield* Tensor.reshape(
        yield* Tensor.slice(bank, { start: [expert, 0, 0], end: [expert + 1, bank.shape[1], bank.shape[2]] }),
        [bank.shape[1], bank.shape[2]]
      )

      const output = yield* Tensor.linearRows(yield* Tensor.take(x, ids), weight)

      for (let index = 0; index < selected.length; index++) {
        rows[selected[index]] = yield* Tensor.slice(output, { start: [index, 0], end: [index + 1, bank.shape[1]] })
      }
    }

    const [first, second, ...rest] = rows

    return yield* Tensor.concat([first, second, ...rest])
  })

onDevices("groupedExpertLinearRows", (device) => (it) => {
  for (const dtype of ["f32", "bf16"] as const) {
    for (const optimize of [false, true]) {
      it.effect(
        dtype + " exact active groups match ordinary projections, optimize=" + optimize,
        () =>
          Effect.scoped(Effect.gen(function*() {
            const routes = [2, 0, 2, 2, 0, 3, 2, 0, 2, 3, 2, 2, 2, 0, 3, 2, 0]

            const x = yield* values(Array.from({ length: routes.length * 65 }, (_, i) => (i % 17 - 8) / 8), [
              routes.length,
              65
            ], dtype)

            const bank = yield* values(
              Array.from({ length: 4 * 13 * 65 }, (_, i) => (i % 23 - 11) / 16),
              [4, 13, 65],
              dtype
            )

            const ids = yield* Tensor.fromTypedArray(Uint32Array.from(routes))
            const bindings = yield* Tensor.compute([x, bank, ids]).pipe(Effect.flatMap(Tensor.clearAllScoped))
            const inputs = yield* Effect.forEach(bindings, (value, slot) => Tensor.makeInput(slot, value))
            const root = yield* Tensor.groupedExpertLinearRows(inputs[0], inputs[1], inputs[2])
            const reference = yield* groupedReference(inputs[0], inputs[1], routes)
            const program = yield* Tensor.freezeProgram([root, reference], { optimize })
            const outputs = yield* Tensor.runProgram(program, bindings).pipe(Effect.flatMap(Tensor.clearAllScoped))
            expect(outputs[0].shape).toEqual([routes.length, 13])
            expect(outputs[0].dtype).toBe(dtype)
            expect(yield* Tensor.toNumberArray(outputs[0])).toEqual(yield* Tensor.toNumberArray(outputs[1]))
            expect(yield* Tensor.toNumberArray(bindings[1])).toEqual(yield* Tensor.toNumberArray(bank))
          }))
      )

      it.effect(
        dtype + " invalid routes, zero dimensions and recovery, optimize=" + optimize,
        () =>
          Effect.scoped(Effect.gen(function*() {
            for (const [rows, inner, columns] of [[0, 3, 2], [3, 0, 2], [3, 3, 0], [0, 0, 0], [3, 3, 2]]) {
              const x = yield* Tensor.ones([rows, inner], { dtype })
              const bank = yield* Tensor.ones([2, columns, inner], { dtype })
              const ids = yield* Tensor.zeros([rows], { dtype: "u32" })
              const root = yield* Tensor.groupedExpertLinearRows(x, bank, yield* Tensor.makeInput(0, ids))
              const program = yield* Tensor.freezeProgram([root], { optimize })

              if (rows > 0) {
                for (const invalid of [2, 0xffffffff]) {
                  const bad = yield* Tensor.fromTypedArray(new Uint32Array(rows).fill(invalid))
                  const error = yield* Effect.flip(Tensor.runProgram(program, [bad]))
                  expect(error.message).toContain("index is out of range")
                }
              }

              const [output] = yield* Tensor.runProgram(program, [ids]).pipe(Effect.flatMap(Tensor.clearAllScoped))
              expect(output.shape).toEqual([rows, columns])
              expect(yield* Tensor.toNumberArray(output)).toEqual(Array(rows * columns).fill(inner))
            }
          }))
      )
    }
  }

  it.effect("offset, strided and broadcast operands retain row order", () =>
    Effect.scoped(Effect.gen(function*() {
      for (const dtype of ["f32", "bf16"] as const) {
        const x = yield* values([99, 1, 2, 99, 3, 4], [2, 3], dtype)
        const bank = yield* values([99, 1, 2, 99, 3, 4, 99, 5, 6, 99, 7, 8], [2, 2, 3], dtype)
        const ids = yield* Tensor.fromTypedArray(new Uint32Array([99, 1, 99, 0]))

        const root = yield* Tensor.groupedExpertLinearRows(
          yield* Tensor.slice(x, { start: [0, 1] }),
          yield* Tensor.transpose(yield* Tensor.slice(bank, { start: [0, 0, 1] }), [0, 2, 1]),
          yield* Tensor.slice(ids, { start: [1], stride: [2] })
        )

        const broadcast = yield* Tensor.groupedExpertLinearRows(
          yield* Tensor.broadcastTo(yield* values([2, 3], [1, 2], dtype), [4, 2]),
          yield* Tensor.broadcastTo(yield* values([1, 0, 0, 2], [1, 2, 2], dtype), [3, 2, 2]),
          yield* Tensor.broadcastTo(yield* Tensor.fromTypedArray(new Uint32Array([2])), [4])
        )

        for (const optimize of [false, true]) {
          const program = yield* Tensor.freezeProgram([root, broadcast], { optimize })
          const outputs = yield* Tensor.runProgram(program, []).pipe(Effect.flatMap(Tensor.clearAllScoped))
          expect(yield* Tensor.toNumberArray(outputs[0])).toEqual([19, 22, 15, 22])
          expect(yield* Tensor.toNumberArray(outputs[1])).toEqual([2, 6, 2, 6, 2, 6, 2, 6])
        }
      }
    })))

  it.effect("device-generated routes remain dynamic across invocations", () =>
    Effect.scoped(Effect.gen(function*() {
      const x = yield* values([1, 2, 3, 4, 5, 6], [3, 2], "bf16")
      const bank = yield* values([1, 0, 0, 1, 2, 0, 0, 3], [2, 2, 2], "bf16")
      const scores = yield* Tensor.makeInput(0, yield* Tensor.zeros([3, 2]))
      const ids = yield* Tensor.reshape(yield* Tensor.topKIndices(scores, 1), [3])
      const program = yield* Tensor.freezeProgram([yield* Tensor.groupedExpertLinearRows(x, bank, ids)])

      for (
        const [data, expected] of [
          [[0, 2, 3, 0, 0, 1], [2, 6, 3, 4, 10, 18]],
          [[2, 0, 0, 3, 1, 0], [1, 2, 6, 12, 5, 6]]
        ]
      ) {
        const score = yield* Tensor.fromTypedArray(Float32Array.from(data), [3, 2])
        const [output] = yield* Tensor.runProgram(program, [score]).pipe(Effect.flatMap(Tensor.clearAllScoped))
        expect(yield* Tensor.toNumberArray(output)).toEqual(expected)
      }
    })))

  it.effect("rejects invalid metadata and differentiation", () =>
    Effect.gen(function*() {
      for (
        const [xs, ws, ids] of [
          [[2], [2, 3, 2], [1]],
          [[1, 2], [3, 2], [1]],
          [[1, 2], [2, 3, 2], [1, 1]],
          [[2, 2], [2, 3, 2], [1]],
          [[1, 2], [2, 3, 4], [1]],
          [[1, 2], [0, 3, 2], [1]],
          [[1, 0], [0x100000000, 0, 0], [1]]
        ]
      ) {
        const error = yield* Effect.flip(Tensor.groupedExpertLinearRows(
          yield* Tensor.zeros(xs),
          yield* Tensor.zeros(ws),
          yield* Tensor.zeros(ids, { dtype: "u32" })
        ))

        expect(error._tag).toBe("TensorError")
      }

      const ids = yield* Tensor.zeros([1], { dtype: "u32" })

      for (const dtype of ["f16", "f64", "u32"] as const) {
        if (device === "metal" && dtype === "f64") continue

        const error = yield* Effect.flip(Tensor.groupedExpertLinearRows(
          yield* Tensor.zeros([1, 2], { dtype }),
          yield* Tensor.zeros([2, 3, 2], { dtype }),
          ids
        ))

        expect(error.message).toContain("expected f32 or bf16")
      }

      const x = yield* Tensor.ones([1, 2])
      const bank = yield* Tensor.ones([2, 3, 2])
      expect((yield* Effect.flip(Tensor.groupedExpertLinearRows(x, yield* Tensor.cast(bank, "bf16"), ids))).message)
        .toContain("dtype mismatch")
      expect(
        (yield* Effect.flip(Tensor.groupedExpertLinearRows(x, bank, yield* Tensor.zeros([1], { dtype: "i64" }))))
          .message
      )
        .toContain("indices must be u32")
      const loss = yield* Tensor.sum(yield* Tensor.groupedExpertLinearRows(x, bank, ids))
      expect((yield* Effect.flip(Gradient.grad(loss, [x]))).message).toContain(
        "groupedExpertLinearRows is inference-only"
      )
    }))

  it.effect("captures and duplicate outputs survive release; interruption releases temporaries", () =>
    Effect.scoped(Effect.gen(function*() {
      const bindings = yield* Tensor.compute([
        yield* values([1, 2, 3, 4], [2, 2], "bf16"),
        yield* values([1, 0, 0, 1, 2, 0, 0, 3], [2, 2, 2], "bf16"),
        yield* Tensor.fromTypedArray(new Uint32Array([0, 1]))
      ]).pipe(Effect.flatMap(Tensor.clearAllScoped))

      const captured = yield* Tensor.freezeProgram([
        yield* Tensor.groupedExpertLinearRows(bindings[0], bindings[1], bindings[2])
      ])

      const inputs = yield* Effect.forEach(bindings, (value, slot) => Tensor.makeInput(slot, value))
      const root = yield* Tensor.groupedExpertLinearRows(inputs[0], inputs[1], inputs[2])
      const bound = yield* Tensor.freezeProgram([root, root])
      const runtime = yield* Runtime.Runtime
      yield* Effect.scoped(Tensor.runProgram(bound, bindings).pipe(Effect.flatMap(Tensor.clearAllScoped)))
      const before = yield* runtime.extensions.diagnostics.externalMemoryBytes

      const fiber = yield* Tensor.runProgram(bound, bindings).pipe(
        Effect.tap(Tensor.clearAll),
        Effect.forkChild({ startImmediately: true })
      )

      yield* Fiber.interrupt(fiber)
      yield* Effect.scoped(Effect.gen(function*() {
        const repeated = yield* Tensor.runProgram(bound, bindings).pipe(Effect.flatMap(Tensor.clearAllScoped))
        yield* Tensor.clear(repeated[0])
        expect(yield* Tensor.toNumberArray(repeated[1])).toEqual([1, 2, 6, 12])
      }))

      // Native work and its late-result cleanup may retire after the fiber exits.
      for (let attempt = 0; attempt < 30; attempt++) {
        if ((yield* runtime.extensions.diagnostics.externalMemoryBytes) === before) break

        yield* Effect.promise(() => new Promise((resolve) => setTimeout(resolve, 10)))
      }

      expect(yield* runtime.extensions.diagnostics.externalMemoryBytes).toBe(before)
      const [first] = yield* Tensor.runProgram(bound, bindings).pipe(Effect.flatMap(Tensor.clearAllScoped))
      expect(yield* Tensor.toNumberArray(bindings[1])).toEqual([1, 0, 0, 1, 2, 0, 0, 3])
      yield* Tensor.clearAll(bindings)
      const [second] = yield* Tensor.runProgram(captured, []).pipe(Effect.flatMap(Tensor.clearAllScoped))
      expect(yield* Tensor.toNumberArray(first)).toEqual([1, 2, 6, 12])
      expect(yield* Tensor.toNumberArray(second)).toEqual([1, 2, 6, 12])
      expect((yield* Effect.flip(Tensor.runProgram(bound, bindings)))._tag).toBe("TensorError")
    })))
})

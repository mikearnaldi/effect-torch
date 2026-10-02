import { expect } from "@effect/vitest"
import { Effect } from "effect"
import { Runtime, Tensor } from "../src/index.ts"
import { deep, onDevices } from "./utils/devices.ts"

const compile = (access: "Append" | "ReadOnly") =>
  Effect.gen(function*() {
    const roots: Array<Tensor.Lazy> = []

    for (const [layerId, heads, width] of [[0, 1, 1], [1, 2, 2]] as const) {
      const zeros = yield* Tensor.zeros([1, heads, 2, width])
      const values = yield* Tensor.makeInput(layerId, zeros)

      const options: Tensor.ScaledDotProductAttentionOptions = {
        layerId,
        retentionWindow: layerId === 0 ? 2 : null,
        causal: access === "Append",
        rounding: "stepwise"
      }

      roots.push(
        yield* Tensor.scaledDotProductAttention(
          zeros,
          zeros,
          values,
          access === "Append" && layerId === 0 ? { ...options, window: 3 } : options
        )
      )
    }

    return yield* Tensor.compileDecodeProgram(roots, {
      access,
      currentBlockAttention: access === "Append" ? "Causal" : "Bidirectional",
      maxTokens: 32,
      blockSize: 2,
      kvDtype: "f32",
      batch: 1
    }, { optimize: false })
  })

const inputs = (a: number, b: number) =>
  Effect.all([
    Tensor.fromTypedArray(new Float32Array([a, b]), [1, 1, 2, 1]),
    Tensor.fromTypedArray(new Float32Array([a * 10, a * 10, b * 10, b * 10, a * 10, a * 10, b * 10, b * 10]), [
      1,
      2,
      2,
      2
    ])
  ])

const clear = (values: ReadonlyArray<Tensor.Concrete>) => Effect.forEach(values, Tensor.clear, { discard: true })

onDevices("Native KV snapshots", () => (it) => {
  it.effect(
    "runs full DiffusionGemma head geometry with retention larger than deployment capacity",
    () =>
      Effect.gen(function*() {
        const make = (access: "Append" | "ReadOnly") =>
          Effect.gen(function*() {
            const roots = yield* Effect.forEach([[8, 256], [2, 512]] as const, ([heads, width], layerId) =>
              Effect.gen(function*() {
                const q = yield* Tensor.zeros([1, 16, 2, width], { dtype: "bf16" })
                const k = yield* Tensor.zeros([1, heads, 2, width], { dtype: "bf16" })
                const v = yield* Tensor.full([1, heads, 2, width], 2, { dtype: "bf16" })

                const options: Tensor.ScaledDotProductAttentionOptions = {
                  layerId,
                  retentionWindow: layerId === 0 ? 1023 : null,
                  causal: access === "Append",
                  rounding: "stepwise"
                }

                return yield* Tensor.scaledDotProductAttention(
                  q,
                  k,
                  v,
                  access === "Append" && layerId === 0 ? { ...options, window: 1024 } : options
                )
              }))

            return yield* Tensor.compileDecodeProgram(roots, {
              access,
              currentBlockAttention: access === "Append" ? "Causal" : "Bidirectional",
              maxTokens: 544,
              blockSize: 16,
              batch: 1,
              kvDtype: "bf16"
            })
          })

        const encoder = yield* make("Append")
        const denoiser = yield* make("ReadOnly")
        expect(encoder.kvLayers.map((layer) => layer.retentionWindow)).toEqual([1023, null])
        expect(Runtime.sameKvSchema(encoder.kvLayers, denoiser.kvLayers)).toBe(true)
        const pool = yield* Tensor.makeKvPool({ kvLayers: encoder.kvLayers, maxTokens: 544, blockSize: 16 })
        const sequence = yield* Tensor.makeKvSequence(pool)
        yield* clear(yield* Tensor.runDecodeProgram(encoder, [], sequence, [1, 2]))
        const prefix = yield* Tensor.snapshotKvSequence(sequence)
        const before = yield* Tensor.inspectKvPrefix(prefix)
        expect(before.layers.map((layer) => layer.startPosition)).toEqual([0, 0])
        const outputs = yield* Tensor.runReadOnlyDecodeProgram(denoiser, [], prefix, 2)

        for (const output of outputs) {
          expect(output.dtype).toBe("bf16")
          expect((yield* Tensor.toNumberArray(output)).every((value) => value === 2)).toBe(true)
        }

        expect(yield* Tensor.inspectKvPrefix(prefix)).toEqual(before)
        yield* clear(outputs)
        yield* Tensor.releaseKvPrefix(prefix)
        yield* Tensor.releaseKvSequence(sequence)
      }),
    { timeout: 60_000 }
  )

  it.effect("retains zero and one local rows while the absolute cursor exceeds pool capacity", () =>
    Effect.gen(function*() {
      const make = (access: "Append" | "ReadOnly") =>
        Effect.gen(function*() {
          const zero = yield* Tensor.zeros([1, 1, access === "Append" ? 1 : 2, 1])
          const value = yield* Tensor.makeInput(0, zero)

          const roots = yield* Effect.forEach([0, 1], (retentionWindow, layerId) => {
            const options = { layerId, retentionWindow, causal: access === "Append" }

            return Tensor.scaledDotProductAttention(
              zero,
              zero,
              value,
              access === "Append" ? { ...options, window: retentionWindow + 1 } : options
            )
          })

          return yield* Tensor.compileDecodeProgram(roots, {
            access,
            currentBlockAttention: access === "Append" ? "Causal" : "Bidirectional",
            window: access === "Append" ? 2 : undefined,
            maxTokens: 4,
            blockSize: 2,
            batch: 1,
            kvDtype: "f32"
          })
        })

      const program = yield* make("Append")
      const denoiser = yield* make("ReadOnly")
      expect(Runtime.sameKvSchema(program.kvLayers, denoiser.kvLayers)).toBe(true)
      expect(program.window).toBe(2)
      expect(denoiser.window).toBeUndefined()
      expect(Runtime.sameDecodeStateSchema(program, denoiser)).toBe(true)
      const pool = yield* Tensor.makeKvPool({ kvLayers: program.kvLayers, maxTokens: 4, blockSize: 2 })
      const sequence = yield* Tensor.makeKvSequence(pool)

      for (let token = 1; token <= 9; token++) {
        const [input] = yield* Tensor.compute(
          [yield* Tensor.fromTypedArray(new Float32Array([token]), [1, 1, 1, 1])] as const
        )

        yield* clear(yield* Tensor.runDecodeProgram(program, [input], sequence, [token]))
        yield* Tensor.clear(input)
      }

      const prefix = yield* Tensor.snapshotKvSequence(sequence)
      const inspection = yield* Tensor.inspectKvPrefix(prefix)
      expect(inspection.cursor).toBe(9)
      expect(inspection.layers.map((layer) => [layer.startPosition, layer.values])).toEqual([[9, []], [8, [9]]])

      const [canvas] = yield* Tensor.compute(
        [yield* Tensor.fromTypedArray(new Float32Array([10, 20]), [1, 1, 2, 1])] as const
      )

      const outputs = yield* Tensor.runReadOnlyDecodeProgram(denoiser, [canvas], prefix, 2)
      deep(yield* Tensor.toNumberArray(outputs[0]!), [15, 15])
      deep(yield* Tensor.toNumberArray(outputs[1]!), [13, 13])
      expect(yield* Tensor.inspectKvPrefix(prefix)).toEqual(inspection)
      yield* clear([canvas, ...outputs])
      yield* Tensor.releaseKvSequence(sequence)
      yield* Tensor.releaseKvPrefix(prefix)
    }))

  it.effect("reads a full-capacity prefix with invocation-owned canvas storage", () =>
    Effect.gen(function*() {
      const zero = yield* Tensor.zeros([1, 1, 2, 1])
      const value = yield* Tensor.makeInput(0, zero)

      const make = (access: "Append" | "ReadOnly") =>
        Effect.gen(function*() {
          const root = yield* Tensor.scaledDotProductAttention(zero, zero, value, {
            layerId: 0,
            retentionWindow: null,
            causal: access === "Append"
          })

          return yield* Tensor.compileDecodeProgram([root], {
            access,
            currentBlockAttention: access === "ReadOnly" ? "Bidirectional" : "Causal",
            maxTokens: 2,
            blockSize: 2,
            batch: 1,
            kvDtype: "f32"
          })
        })

      const encoder = yield* make("Append")
      const denoiser = yield* make("ReadOnly")
      const pool = yield* Tensor.makeKvPool({ kvLayers: encoder.kvLayers, maxTokens: 2, blockSize: 2 })
      const sequence = yield* Tensor.makeKvSequence(pool)

      const [input] = yield* Tensor.compute(
        [yield* Tensor.fromTypedArray(new Float32Array([2, 4]), [1, 1, 2, 1])] as const
      )

      yield* clear(yield* Tensor.runDecodeProgram(encoder, [input], sequence, [2, 4]))
      const prefix = yield* Tensor.snapshotKvSequence(sequence)
      const before = yield* Tensor.inspectKvPrefix(prefix)
      const [output] = yield* Tensor.runReadOnlyDecodeProgram(denoiser, [input], prefix, 1)
      deep(yield* Tensor.toNumberArray(output!), [8 / 3, 0])
      expect(yield* Tensor.inspectKvPrefix(prefix)).toEqual(before)
      yield* clear([input, output!])
      yield* Tensor.releaseKvSequence(sequence)
      yield* Tensor.releaseKvPrefix(prefix)
    }))

  it.effect("preserves BF16 score scaling and probability rounding with an empty prefix", () =>
    Effect.gen(function*() {
      const q = yield* Tensor.ones([1, 1, 2, 1], { dtype: "bf16" })
      const k = yield* Tensor.cast(yield* Tensor.fromTypedArray(new Float32Array([3, -2]), [1, 1, 2, 1]), "bf16")
      const v = yield* Tensor.cast(yield* Tensor.fromTypedArray(new Float32Array([0, 1]), [1, 1, 2, 1]), "bf16")

      const root = yield* Tensor.scaledDotProductAttention(q, k, v, {
        layerId: 0,
        retentionWindow: null,
        scale: 0.3,
        rounding: "stepwise"
      })

      const program = yield* Tensor.compileDecodeProgram([root], {
        access: "ReadOnly",
        currentBlockAttention: "Bidirectional",
        maxTokens: 2,
        blockSize: 2,
        batch: 1,
        kvDtype: "bf16"
      }, { optimize: false })

      const pool = yield* Tensor.makeKvPool({ kvLayers: program.kvLayers, maxTokens: 2, blockSize: 2 })
      const sequence = yield* Tensor.makeKvSequence(pool)
      const prefix = yield* Tensor.snapshotKvSequence(sequence)
      const [output] = yield* Tensor.runReadOnlyDecodeProgram(program, [], prefix, 2)
      expect(output!.dtype).toBe("bf16")
      expect(yield* Tensor.toNumberArray(output!)).toEqual([0.1826171875, 0.1826171875])
      expect(yield* Tensor.kvSequenceCursor(sequence)).toBe(0)
      yield* Tensor.clear(output!)
      yield* Tensor.releaseKvPrefix(prefix)
      yield* Tensor.releaseKvSequence(sequence)
    }))

  it.effect("shares heterogeneous prefixes across concurrent canvas reads and private commits", () =>
    Effect.gen(function*() {
      const encoder = yield* compile("Append")
      const denoiser = yield* compile("ReadOnly")
      expect(Runtime.sameKvSchema(encoder.kvLayers, denoiser.kvLayers)).toBe(true)
      expect(encoder.kvLayers.map((layer) => [layer.kvHeads, layer.headDim])).toEqual([[1, 1], [2, 2]])
      const pool = yield* Tensor.makeKvPool({ kvLayers: encoder.kvLayers, maxTokens: 32, blockSize: 2 })
      const sequence = yield* Tensor.makeKvSequence(pool)
      yield* clear(yield* Tensor.runDecodeProgram(encoder, yield* inputs(1, 2), sequence, [1, 2]))
      yield* clear(yield* Tensor.runDecodeProgram(encoder, yield* inputs(4, 0), sequence, [4]))
      const prefix = yield* Tensor.snapshotKvSequence(sequence)
      expect(prefix.tokenCount).toBe(3)
      expect(prefix.retainedBytes).toBeGreaterThan(0)
      const before = yield* Tensor.inspectKvPrefix(prefix)
      expect(before.cursor).toBe(3)
      expect(before.layers[0]!.startPosition).toBe(1)
      expect(before.layers[0]!.values).toEqual([2, 4])
      expect(before.layers[1]!.startPosition).toBe(0)
      expect(before.layers[1]!.values).toEqual([10, 10, 10, 10, 20, 20, 20, 20, 40, 40, 40, 40])
      expect(before.sharedBytes).toBeGreaterThan(0)
      expect(before.copiedBytes).toBe(0)

      const a = yield* inputs(10, 20)
      const b = yield* inputs(40, 80)

      const [first, second] = yield* Effect.all([
        Tensor.runReadOnlyDecodeProgram(denoiser, a, prefix, 2),
        Tensor.runReadOnlyDecodeProgram(denoiser, b, prefix, 2)
      ], { concurrency: "unbounded" })

      deep(yield* Tensor.toNumberArray(first[0]!), [9, 9])
      deep(yield* Tensor.toNumberArray(second[0]!), [31.5, 31.5])
      deep(yield* Tensor.toNumberArray(first[1]!), Array(8).fill(74))
      deep(yield* Tensor.toNumberArray(second[1]!), Array(8).fill(254))
      expect(yield* Tensor.inspectKvPrefix(prefix)).toEqual(before)
      expect(yield* Tensor.kvSequenceCursor(sequence)).toBe(3)

      const fork = yield* Tensor.forkKvPrefix(prefix)
      yield* clear(yield* Tensor.runDecodeProgram(encoder, yield* inputs(8, 0), fork, [8]))
      const committed = yield* Tensor.snapshotKvSequence(fork)
      const after = yield* Tensor.inspectKvPrefix(committed)
      expect(after.cursor).toBe(4)
      expect(after.layers[0]!.values).toEqual([4, 8])
      expect(after.copiedBytes - before.copiedBytes).toBeLessThan(before.retainedBytes)
      expect((yield* Tensor.inspectKvPrefix(prefix)).layers).toEqual(before.layers)
      yield* Tensor.releaseKvSequence(fork)
      yield* Tensor.releaseKvSequence(sequence)
      yield* Tensor.releaseKvPrefix(prefix)
      yield* Tensor.releaseKvPrefix(committed)
      deep(yield* Tensor.toNumberArray(first[0]!), [9, 9])
      yield* clear([...first, ...second])
    }))
})

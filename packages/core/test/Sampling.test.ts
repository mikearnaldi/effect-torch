import { describe, expect } from "@effect/vitest"
import * as assert from "@effect/vitest/utils"
import { Effect } from "effect"
import { Tensor } from "../src/index.ts"
import { floats, onDevices } from "./utils/devices.ts"

onDevices("Sampling", () => (it) => {
  const logits = (values: ReadonlyArray<number>) =>
    Effect.gen(function*() {
      const [tensor] = yield* Tensor.compute([
        yield* Tensor.fromTypedArray(floats(values), [values.length])
      ])
      return tensor
    })

  describe("sample", () => {
    it.effect("uses native greedy argmax with lower-token tie breaking", () =>
      Effect.gen(function*() {
        const tensor = yield* logits([1, 4, 4, 2])
        const token = yield* Tensor.sample(tensor, { temperature: 0, seed: 1 })
        assert.strictEqual(token, 1)
        assert.deepStrictEqual(yield* Tensor.toNumberArray(tensor), [1, 4, 4, 2])
        yield* Tensor.clear(tensor)
      }))

    it.effect("applies top-k and top-p before a seeded draw", () =>
      Effect.gen(function*() {
        const tensor = yield* logits([3, 2, 1, 0])
        const topK = yield* Tensor.sample(tensor, {
          temperature: 1,
          topK: 1,
          seed: 7
        })
        const topP = yield* Tensor.sample(tensor, {
          temperature: 1,
          topK: 3,
          topP: 0.5,
          seed: 7
        })
        assert.strictEqual(topK, 0)
        assert.strictEqual(topP, 0)
        yield* Tensor.clear(tensor)
      }))

    it.effect("replays the same seed and counter", () =>
      Effect.gen(function*() {
        const tensor = yield* logits([0, 0, 0, 0])
        const options = { temperature: 1, seed: 1234, counter: 9 }
        assert.strictEqual(yield* Tensor.sample(tensor, options), yield* Tensor.sample(tensor, options))
        yield* Tensor.clear(tensor)
      }))

    it.effect("validates sampling controls before native execution", () =>
      Effect.gen(function*() {
        const tensor = yield* logits([1, 2, 3])
        expect((yield* Effect.flip(Tensor.sample(tensor, { temperature: -1, seed: 0 }))).message).toContain(
          "temperature"
        )
        expect((yield* Effect.flip(Tensor.sample(tensor, { topK: 4, seed: 0 }))).message).toContain("topK")
        expect((yield* Effect.flip(Tensor.sample(tensor, { topP: 0, seed: 0 }))).message).toContain("topP")
        yield* Tensor.clear(tensor)
      }))
  })

  describe("sampled decode", () => {
    const program = (batch: number, vocabulary: number) =>
      Effect.gen(function*() {
        const exemplar = yield* Tensor.zeros([batch, 1, vocabulary])
        const input = yield* Tensor.makeInput(0, exemplar)
        return yield* Tensor.compileDecodeProgram([input], {
          maxTokens: 16,
          blockSize: 4,
          kvDtype: "f32",
          batch,
          lastTokenRow: true
        })
      })

    const sequences = (decode: Tensor.DecodeProgram, count: number) =>
      Effect.gen(function*() {
        const pool = yield* Tensor.makeKvPool(
          decode.layers,
          decode.kvHeads,
          decode.headDim,
          decode.maxTokens,
          decode.blockSize,
          decode.kvDtype
        )
        return yield* Effect.all(Array.from({ length: count }, () => Tensor.makeKvSequence(pool)))
      })

    it.effect("validates and samples one decode output without publishing logits", () =>
      Effect.gen(function*() {
        const decode = yield* program(1, 4)
        const [sequence] = yield* sequences(decode, 1)
        const invalid = yield* Effect.flip(
          Tensor.runDecodeProgramSampled(
            decode,
            [yield* Tensor.fromTypedArray(floats([1, 4, 2, 0]), [1, 1, 4])],
            sequence,
            [1],
            { topK: 5, seed: 0 }
          )
        )
        expect(invalid.message).toContain("topK")
        expect(yield* Tensor.kvSequenceCursor(sequence)).toBe(0)

        const token = yield* Tensor.runDecodeProgramSampled(
          decode,
          [yield* Tensor.fromTypedArray(floats([1, 4, 4, 0]), [1, 1, 4])],
          sequence,
          [1],
          { temperature: 0, topK: 4, seed: 7 }
        )
        expect(token).toBe(1)
        expect(yield* Tensor.kvSequenceCursor(sequence)).toBe(1)
        yield* Tensor.releaseKvSequence(sequence)
      }))

    it.effect("returns request-ordered tokens from sparse physical slots", () =>
      Effect.gen(function*() {
        const decode = yield* program(3, 4)
        const active = yield* sequences(decode, 2)
        const sampled = yield* Tensor.runBatchedDecodeProgramSampled(
          decode,
          [
            yield* Tensor.fromTypedArray(
              floats([
                9,
                1,
                2,
                3,
                0,
                0,
                0,
                0,
                1,
                2,
                3,
                8
              ]),
              [3, 1, 4]
            )
          ],
          active,
          [2, 0],
          [[1], [2]],
          [{ temperature: 0, seed: 1 }, { temperature: 0, seed: 2 }]
        )
        expect(sampled).toEqual([3, 0])
        expect(yield* Effect.all(active.map(Tensor.kvSequenceCursor))).toEqual([1, 1])
        yield* Effect.forEach(active, Tensor.releaseKvSequence)
      }))

    it.effect("zeroes inactive last-token outputs", () =>
      Effect.gen(function*() {
        const decode = yield* program(3, 4)
        const [sequence] = yield* sequences(decode, 1)
        const outputs = yield* Tensor.runBatchedDecodeProgram(
          decode,
          [yield* Tensor.fromTypedArray(floats([9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4]), [3, 1, 4])],
          [sequence],
          [2],
          [[1]]
        )
        expect(yield* Tensor.toNumberArray(outputs[0]!)).toEqual([0, 0, 0, 0])
        expect(yield* Tensor.toNumberArray(outputs[1]!)).toEqual([0, 0, 0, 0])
        expect(yield* Tensor.toNumberArray(outputs[2]!)).toEqual([1, 2, 3, 4])
        yield* Tensor.clearAll(outputs)
        yield* Tensor.releaseKvSequence(sequence)
      }))
  })
})

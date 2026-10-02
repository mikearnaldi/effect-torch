import { describe, expect, it } from "@effect/vitest"
import { Effect } from "effect"
import { DecisionNoise as Noise } from "../src/index.ts"

const config = { seed: "fixture", semanticKey: "semantic-fixture", readIndex: 3, length: 12, vocabularySize: 262144 }

describe("deterministic uniform vocabulary noise", () => {
  // Independently calculated using Python hashlib.sha256 and struct.unpack('>8I').
  it.effect("matches a fixed hash-counter fixture", () =>
    Effect.gen(function*() {
      const tokens = yield* Noise.make(config)
      expect(Array.from(tokens)).toEqual([
        200558,
        138262,
        66932,
        184469,
        48610,
        196643,
        190022,
        35584,
        148464,
        8158,
        438,
        156299
      ])
      expect(yield* Noise.make(config)).toEqual(tokens)
    }))

  it.effect("discards rejected words instead of using biased modulo", () =>
    Effect.gen(function*() {
      // Size just over 2^31 rejects about half of all uint32 words. The fixture
      // includes several rejected words, including the first word 3524207178.
      const tokens = yield* Noise.make({ ...config, vocabularySize: 2147483649 })
      expect(Array.from(tokens)).toEqual([
        1082479460,
        225444042,
        2056572949,
        1567887965,
        1675256926,
        2033051097,
        1559274711,
        1260242396,
        361850222,
        1605037654,
        1505986446,
        1040488708
      ])
      expect(tokens[0]).not.toBe(3524207178 % 2147483649)
    }))

  it.effect("distinguishes explicit reads, seeds, and semantic questions", () =>
    Effect.gen(function*() {
      const initial = yield* Noise.make({ ...config, length: 128 })
      const read = yield* Noise.make({ ...config, length: 128, readIndex: 4 })
      const seed = yield* Noise.make({ ...config, length: 128, seed: "different" })
      const question = yield* Noise.make({ ...config, length: 128, semanticKey: "different" })
      expect(read).not.toEqual(initial)
      expect(seed).not.toEqual(initial)
      expect(question).not.toEqual(initial)

      const reordered = yield* Effect.forEach([4, 3, 4], (readIndex) =>
        Noise.make({ ...config, length: 128, readIndex }), { concurrency: "unbounded" })

      expect(reordered).toEqual([read, initial, read])
    }))

  it.effect("samples the full configured vocabulary and preserves canvas prefixes", () =>
    Effect.gen(function*() {
      const tokens = yield* Noise.make({ ...config, length: 4096, vocabularySize: 7 })
      expect(new Set(tokens)).toEqual(new Set([0, 1, 2, 3, 4, 5, 6]))
      expect(yield* Noise.make({ ...config, length: 24 })).toEqual(
        (yield* Noise.make({ ...config, length: 64 })).slice(0, 24)
      )
      expect(yield* Noise.make({ ...config, vocabularySize: 1 })).toEqual(new Uint32Array(12))
      expect(yield* Noise.make({ ...config, length: 0 })).toEqual(new Uint32Array(0))
      const allUint32 = yield* Noise.make({ ...config, vocabularySize: 2 ** 32 })
      expect(allUint32).toHaveLength(12)
      expect(Array.from(allUint32).some((token) => token > 2 ** 31)).toBe(true)
    }))

  it.effect("returns fresh output without mutating configuration", () =>
    Effect.gen(function*() {
      const input = { ...config }
      const before = { ...input }
      const first = yield* Noise.make(input)
      first.fill(0)
      expect(yield* Noise.make(input)).not.toEqual(first)
      expect(input).toEqual(before)
    }))

  const invalid = [
    { vocabularySize: 0 },
    { vocabularySize: -1 },
    { vocabularySize: 1.5 },
    { vocabularySize: 2 ** 32 + 1 },
    { vocabularySize: Infinity },
    { readIndex: -1 },
    { readIndex: 0.5 },
    { readIndex: NaN },
    { readIndex: 2 ** 53 },
    { length: -1 },
    { length: 1.5 },
    { semanticKey: "" },
    { seed: undefined },
    { readIndex: undefined },
    { schedulerSlot: 1 }
  ]

  for (const [index, invalidConfig] of invalid.entries()) {
    it.effect("rejects invalid configuration " + index, () =>
      Effect.gen(function*() {
        // SAFETY: deliberately violates the type to verify runtime validation.
        const result = yield* Effect.flip(Noise.make({ ...config, ...invalidConfig } as Noise.Options))
        expect(result).toBeInstanceOf(Noise.NoiseError)
      }))
  }
})

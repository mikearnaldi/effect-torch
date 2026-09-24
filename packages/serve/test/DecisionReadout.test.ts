import { describe, expect, it } from "@effect/vitest"
import { Effect, Schema } from "effect"
import { Decision as Contract, DecisionPlanner as Planner, DecisionReadout as Readout } from "../src/index.ts"

const decodeQuestion = <Input>(q: Input) =>
  Contract.decodeRequest({ model: "local-m0", state: "", questions: { q } }).pipe(
    Effect.map((request) => request.questions.q)
  )

describe("complete decision readout", () => {
  // Values from INTERFACE.md, verified against the pinned official adapter
  // fb52b1030b7fc1f4f1cf39910afa5da54f9835e3/confidence_metrics.py.
  it.effect("matches the official Choice arithmetic rather than max probability", () =>
    Effect.gen(function*() {
      const q = yield* decodeQuestion({
        type: "choice",
        criteria: { billing: null, engineering: { means: "bugs" }, sales: null }
      })

      const answer = yield* Readout.answerFromProbabilities(q, [0.1, 0.8, 0.1])
      expect(answer.type).toBe("choice")

      if (answer.type !== "choice") return

      expect(answer.choice).toBe("engineering")
      expect(answer.probabilities).toEqual({ billing: 0.1, engineering: 0.8, sales: 0.1 })
      expect(answer.confidence).toBeCloseTo(0.7, 14)
      expect(answer.confidence).not.toBe(0.8)
    }))

  it.effect("returns Score expectation 1.5, confidence 0.25, and the original rich legend", () =>
    Effect.gen(function*() {
      const levels = [null, { severity: "alternative", details: [true, 2] }, ["blocking", { color: "red" }]]
      const q = yield* decodeQuestion({ type: "score", criteria: levels })
      const answer = yield* Readout.answerFromProbabilities(q, [0.1, 0.3, 0.6])
      expect(answer.type).toBe("score")

      if (answer.type !== "score") return

      expect(answer.score).toBe(1.5)
      expect(answer.confidence).toBe(0.25)
      expect(answer.legend).toEqual({ "0": levels[0], "1": levels[1], "2": levels[2] })
      expect(answer.probabilities).toEqual({ "0": 0.1, "1": 0.3, "2": 0.6 })
      expect(answer.legend["1"]).not.toBe(levels[1])
    }))

  it.effect("uses the first mode for Score ties and first Choice option for ties", () =>
    Effect.gen(function*() {
      expect(yield* Readout.scoreConfidence([0.1, 0.45, 0.45])).toBeCloseTo(0.175, 14)
      const q = yield* decodeQuestion({ type: "choice", criteria: { z: null, a: null } })
      expect(yield* Readout.answerFromProbabilities(q, [0.5, 0.5])).toEqual({
        type: "choice",
        choice: "z",
        probabilities: { z: 0.5, a: 0.5 },
        confidence: 0
      })
    }))

  it.effect("keeps full precision and the one-option special case", () =>
    Effect.gen(function*() {
      const p = [0.123456789123, 0.876543210877]
      expect(yield* Readout.choiceConfidence(p)).toBe((p[1] - 0.5) / 0.5)
      expect(yield* Readout.choiceConfidence([1])).toBe(1)
      expect(yield* Readout.scoreConfidence([1])).toBe(1)
      const q = yield* decodeQuestion({ type: "choice", criteria: { only: null } })
      expect(yield* Readout.answerFromLogits(q, [[-1000]])).toEqual({
        type: "choice",
        choice: "only",
        probabilities: { only: 1 },
        confidence: 1
      })
    }))

  it.effect("uses false/true order for Noul with no confidence field", () =>
    Effect.gen(function*() {
      const q = yield* decodeQuestion({ type: "noul" })
      expect(Planner.optionsFor(q).map((option) => option.name)).toEqual(["false", "true"])
      expect(yield* Readout.answerFromProbabilities(q, [0.2, 0.8])).toEqual({ type: "noul", noul: 0.8 })
    }))

  it.effect("uses stable restricted softmax including extreme finite logits", () =>
    Effect.gen(function*() {
      const actual = yield* Readout.restrictedSoftmax([1000, 1001, 999], 3)
      expect(actual[0]).toBeCloseTo(0.24472847105479764, 14)
      expect(actual[1]).toBeCloseTo(0.6652409557748218, 14)
      expect(actual[2]).toBeCloseTo(0.09003057317038046, 14)
      expect(yield* Readout.restrictedSoftmax([Number.MAX_VALUE, -Number.MAX_VALUE], 2)).toEqual([1, 0])
      expect(yield* Readout.restrictedSoftmax([0, 0], 2)).toEqual([0.5, 0.5])
    }))

  it.effect("averages read probabilities rather than logits", () =>
    Effect.gen(function*() {
      const q = yield* decodeQuestion({ type: "choice", criteria: { a: null, b: null } })
      const reads = [[Math.log(9), 0], [Math.log(3 / 2), 0]]
      const before = structuredClone(reads)
      const answer = yield* Readout.answerFromLogits(q, reads)
      expect(answer.type).toBe("choice")

      if (answer.type !== "choice") return

      expect(answer.probabilities.a).toBeCloseTo(0.75, 14)
      expect(answer.probabilities.b).toBeCloseTo(0.25, 14)
      const wrong = yield* Readout.restrictedSoftmax([(reads[0][0] + reads[1][0]) / 2, 0], 2)
      expect(Math.abs(answer.probabilities.a - wrong[0])).toBeGreaterThan(0.03)
      expect(yield* Readout.answerFromLogits(q, [...reads].reverse())).toEqual(answer)
      expect(reads).toEqual(before)
    }))

  it.effect("retains all 255 options and can select the final option", () =>
    Effect.gen(function*() {
      const criteria = Object.fromEntries(Array.from({ length: 255 }, (_, i) => ["option-" + i, { index: i }]))
      const q = yield* decodeQuestion({ type: "choice", criteria })
      const p = Array.from({ length: 255 }, (_, i) => i === 254 ? 0.5 : 0.5 / 254)
      const answer = yield* Readout.answerFromLogits(q, [p.map(Math.log)])
      expect(answer.type).toBe("choice")

      if (answer.type !== "choice") return

      expect(Object.keys(answer.probabilities)).toEqual(Object.keys(criteria))
      expect(Object.values(answer.probabilities)).toHaveLength(255)
      expect(answer.choice).toBe("option-254")
      expect(answer.probabilities["option-254"]).toBeCloseTo(0.5, 14)
      expect(answer.confidence).toBeCloseTo((0.5 - 1 / 255) / (1 - 1 / 255), 14)
      expect(Object.values(answer.probabilities).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 13)
    }))

  it.effect("uniform logits produce valid confidence across all Choice cardinalities", () =>
    Effect.gen(function*() {
      for (let count = 1; count <= 255; count++) {
        const p = yield* Readout.restrictedSoftmax(Array(count).fill(0), count)
        const confidence = yield* Readout.choiceConfidence(p)
        expect(confidence).toBeCloseTo(count === 1 ? 1 : 0, 14)
        expect(confidence).toBeGreaterThanOrEqual(0)
      }
    }))

  const invalidVectors: ReadonlyArray<readonly [string, unknown]> = [
    ["missing vector", undefined],
    ["null vector", null],
    ["zero length", []],
    ["wrong count", [1]],
    ["too many", [1, 2, 3]],
    ["sparse", Array(2)],
    ["missing logit", [0, undefined]],
    ["NaN", [0, NaN]],
    ["infinity", [0, Infinity]],
    ["minus infinity", [-Infinity, 0]],
    ["string", ["0", 1]]
  ]

  for (const [name, input] of invalidVectors) {
    it.effect("rejects " + name + " instead of manufacturing logits", () =>
      Effect.gen(function*() {
        // SAFETY: malformed values intentionally exercise the runtime boundary.
        const error = yield* Effect.flip(Readout.restrictedSoftmax(input as ReadonlyArray<number>, 2))
        expect(error).toBeInstanceOf(Readout.ReadoutError)
      }))
  }

  it.effect("rejects zero/non-normalized/negative probabilities and absent reads", () =>
    Effect.gen(function*() {
      for (const p of [[], [0, 0], [0.2, 0.2], [-0.1, 1.1], [0.5, NaN], [1 + 1e-13, 0]]) {
        expect(yield* Effect.flip(Readout.choiceConfidence(p))).toBeInstanceOf(Readout.ReadoutError)
        expect(yield* Effect.flip(Readout.scoreConfidence(p))).toBeInstanceOf(Readout.ReadoutError)
      }

      const q = yield* decodeQuestion({ type: "noul" })

      for (const reads of [[], [[]], [[0, 0], [0]], Array(1)]) {
        expect(yield* Effect.flip(Readout.answerFromLogits(q, reads))).toBeInstanceOf(Readout.ReadoutError)
      }

      expect(yield* Effect.flip(Readout.meanProbabilities([], 2))).toBeInstanceOf(Readout.ReadoutError)
      expect(yield* Effect.flip(Readout.meanProbabilities([[0.5, 0.5], [0, 0]], 2))).toBeInstanceOf(
        Readout.ReadoutError
      )
      expect(yield* Effect.flip(Readout.restrictedSoftmax([], 0))).toBeInstanceOf(Readout.ReadoutError)
    }))

  it.effect("routes semantic results with original IDs and real supplied usage", () =>
    Effect.gen(function*() {
      const question = { type: "noul" }
      const questions = Object.fromEntries([["__proto__", question], ["constructor", question]])
      const plan = yield* Planner.plan({ model: "local-m0", state: "", questions }, { modelId: "local-m0" })
      const key = plan.routes[0].decision.semanticKey
      const reads = new Map([[key, [[Math.log(0.2), Math.log(0.8)]]]])
      const usage = { input_tokens: 123, output_tokens: 2 }
      const response = yield* Readout.responseFromLogits(plan, reads, usage)
      expect(Object.keys(response.answers)).toEqual(["__proto__", "constructor"])
      expect(response.answers.__proto__).toEqual({ type: "noul", noul: 0.8 })
      expect(response.answers.constructor).toEqual(response.answers.__proto__)
      expect(response.model).toBe("local-m0")
      expect(response.usage).toEqual(usage)
      yield* Schema.decodeUnknownEffect(Contract.Response, { onExcessProperty: "error" })(response)
      expect(yield* Effect.flip(Readout.responseFromLogits(plan, new Map(), usage))).toBeInstanceOf(
        Readout.ReadoutError
      )
      expect(yield* Effect.flip(Readout.responseFromLogits(plan, reads, { ...usage, output_tokens: 0.5 })))
        .toBeInstanceOf(Readout.ReadoutError)
    }))
})

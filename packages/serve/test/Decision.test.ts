import { describe, expect, it } from "@effect/vitest"
import { Effect, Schema } from "effect"
import { Decision as Contract, DecisionPlanner as Planner } from "../src/index.ts"

const base = { model: "diffusiongemma-local-m0", state: "ticket", questions: { q: { type: "noul" } } }

describe("decision contract", () => {
  it.effect("retains structured state, instructions, and nullable descriptions", () =>
    Effect.gen(function*() {
      const input = {
        ...base,
        state: { nested: [1, true, null, { text: "hello" }] },
        questions: {
          noul: { type: "noul", instructions: ["check", { scope: "ticket" }], criteria: { true: { means: "yes" } } },
          choice: { type: "choice", instructions: null, criteria: { bug: { examples: ["crash"] }, other: null } },
          score: { type: "score", instructions: { task: "severity" }, criteria: [null, ["major", { blocking: true }]] }
        }
      }

      const decoded = yield* Contract.decodeRequest(input)
      expect(decoded.state).toEqual(input.state)
      expect(decoded.questions.noul).toEqual({
        ...input.questions.noul,
        criteria: { true: { means: "yes" }, false: null }
      })
      expect(decoded.questions.choice).toEqual(input.questions.choice)
      expect(decoded.questions.score).toEqual(input.questions.score)
    }))

  it.effect("normalizes omitted and null instructions equally without changing input", () =>
    Effect.gen(function*() {
      const before = structuredClone(base)
      const omitted = yield* Contract.decodeRequest(base)

      const explicit = yield* Contract.decodeRequest({
        ...base,
        questions: { q: { type: "noul", instructions: null, criteria: { true: null, false: null } } }
      })

      expect(omitted).toEqual(explicit)
      expect(omitted.questions.q).toEqual({ type: "noul", instructions: null, criteria: { true: null, false: null } })
      expect(base).toEqual(before)
    }))

  for (const state of ["", [], {}, [null, true, 12], { unknownKey: null }]) {
    it.effect("accepts JSON state " + JSON.stringify(state), () =>
      Effect.gen(function*() {
        expect((yield* Contract.decodeRequest({ ...base, state })).state).toEqual(state)
      }))
  }

  const invalidRequests: ReadonlyArray<readonly [string, unknown]> = [
    ["missing model", { state: "x", questions: base.questions }],
    ["empty model", { ...base, model: "" }],
    ["missing state", { model: base.model, questions: base.questions }],
    ["empty questions", { ...base, questions: {} }],
    ["root null state", { ...base, state: null }],
    ["root boolean state", { ...base, state: true }],
    ["root numeric state", { ...base, state: 1 }],
    ["nested undefined", { ...base, state: { x: undefined } }],
    ["nested sparse array", { ...base, state: { x: Array(2) } }],
    ["non-finite state", { ...base, state: [NaN] }],
    ["Date state", { ...base, state: new Date(0) }],
    ["non-JSON bigint", { ...base, state: [1n] }],
    ["extra request field", { ...base, unused: true }]
  ]

  for (const [name, input] of invalidRequests) {
    it.effect("rejects " + name + " with a typed error", () =>
      Effect.gen(function*() {
        const error = yield* Effect.flip(Contract.decodeRequest(input))
        expect(error).toBeInstanceOf(Contract.ValidationError)
        expect(error.message.length).toBeGreaterThan(0)
      }))
  }

  const invalidQuestions = [
    { type: "unknown" },
    { type: "noul", instructions: false },
    { type: "noul", instructions: undefined },
    { type: "noul", instructions: 12 },
    { type: "noul", criteria: null },
    { type: "noul", criteria: { true: 1 } },
    { type: "noul", criteria: { maybe: "maybe" } },
    { type: "noul", extra: "field" },
    { type: "choice" },
    { type: "choice", criteria: {} },
    { type: "choice", criteria: { a: true } },
    { type: "choice", criteria: Object.fromEntries(Array.from({ length: 256 }, (_, i) => ["o" + i, null])) },
    { type: "score" },
    { type: "score", criteria: [] },
    { type: "score", criteria: ["only"] },
    { type: "score", criteria: Array(11).fill(null) }
  ]

  for (const [index, question] of invalidQuestions.entries()) {
    it.effect("rejects malformed question " + index, () =>
      Effect.gen(function*() {
        expect(yield* Effect.flip(Contract.decodeRequest({ ...base, questions: { q: question } }))).toBeInstanceOf(
          Contract.ValidationError
        )
      }))
  }

  it.effect("accepts Choice 1/255 and Score 2/10 boundaries", () =>
    Effect.gen(function*() {
      for (const count of [1, 255]) {
        const criteria = Object.fromEntries(Array.from({ length: count }, (_, i) => ["o" + i, null]))
        expect(
          (yield* Contract.decodeRequest({ ...base, questions: { q: { type: "choice", criteria } } })).questions.q.type
        ).toBe("choice")
      }

      for (const count of [2, 10]) {
        expect(
          (yield* Contract.decodeRequest({
            ...base,
            questions: { q: { type: "score", criteria: Array(count).fill(null) } }
          })).questions.q.type
        ).toBe("score")
      }
    }))

  it.effect("rejects cyclic host input as a typed validation failure", () =>
    Effect.gen(function*() {
      interface Cycle {
        self?: Cycle
      }
      const state: Cycle = {}
      state.self = state
      expect(yield* Effect.flip(Contract.decodeRequest({ ...base, state }))).toBeInstanceOf(Contract.ValidationError)
    }))

  it.effect("requires an explicit matching local model configuration", () =>
    Effect.gen(function*() {
      expect((yield* Planner.plan(base, { modelId: base.model })).model).toBe(base.model)
      expect(yield* Effect.flip(Planner.plan({ ...base, model: "jev-latest" }, { modelId: base.model })))
        .toBeInstanceOf(Contract.ValidationError)
      // SAFETY: a missing required model ID must be rejected at runtime.
      expect(yield* Effect.flip(Planner.plan(base, {} as Contract.ModelConfig))).toBeInstanceOf(
        Contract.ValidationError
      )
    }))

  it.effect("validates response shape and supplied token counts", () =>
    Effect.gen(function*() {
      const response = {
        model: base.model,
        answers: { q: { type: "noul", noul: 0.8 } },
        usage: { input_tokens: 7, output_tokens: 1 }
      }

      const decode = Schema.decodeUnknownEffect(Contract.Response, { onExcessProperty: "error" })
      expect(yield* decode(response)).toEqual(response)
      yield* Effect.flip(decode({ ...response, usage: { input_tokens: -1, output_tokens: 1 } }))
      yield* Effect.flip(decode({ ...response, answers: { q: { type: "noul", noul: 0.8, confidence: 0.6 } } }))
    }))
})

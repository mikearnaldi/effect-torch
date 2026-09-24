import { describe, expect, it } from "@effect/vitest"
import { Effect } from "effect"
import { DecisionNoise as Noise, DecisionPlanner as Planner, DecisionReadout as Readout } from "../src/index.ts"

const model = "diffusiongemma-local-m0"

const config = { modelId: model }

const target = {
  type: "choice",
  instructions: { task: "Select a department" },
  criteria: { engineering: { description: "Bugs", examples: ["crash"] }, billing: null }
}

const sibling = { type: "noul", instructions: "SIBLING_SECRET_617 is true" }

const state = { ticket: "The export button crashes in Safari", workaround: ["Chrome", null] }

describe("isolated planning", () => {
  it.effect("renders exactly one question, with rich content and visible option names", () =>
    Effect.gen(function*() {
      const result = yield* Planner.plan({
        model,
        state,
        questions: { ROUTING_ONLY_TARGET: target, ROUTING_ONLY_SIBLING: sibling }
      }, config)

      const decision = result.routes[0].decision
      expect(decision.prompt).toContain("Select a department")
      expect(decision.prompt).toContain("engineering")
      expect(decision.prompt).toContain("billing")
      expect(decision.prompt).toContain("\"examples\":[\"crash\"]")
      expect(decision.prompt).toContain("The export button crashes in Safari")
      expect(decision.prompt).not.toContain("ROUTING_ONLY")
      expect(decision.prompt).not.toContain("SIBLING_SECRET_617")
      expect(JSON.stringify(decision)).not.toContain("ROUTING_ONLY")
      expect(result.routes.map((route) => route.questionId)).toEqual(["ROUTING_ONLY_TARGET", "ROUTING_ONLY_SIBLING"])

      const moved = yield* Planner.plan({
        model,
        state: { ...state, fact: "SIBLING_SECRET_617" },
        questions: { q: target }
      }, config)

      expect(moved.routes[0].decision.prompt).toContain("SIBLING_SECRET_617")
      expect(moved.routes[0].decision.semanticKey).not.toBe(decision.semanticKey)
    }))

  it.effect("keeps prompts, keys, noise, and answers invariant under rename/reorder/add/remove/duplicate", () =>
    Effect.gen(function*() {
      const isolated = yield* Planner.plan({ model, state, questions: { original: target } }, config)
      const expected = isolated.routes[0].decision

      const noiseOptions = {
        semanticKey: expected.semanticKey,
        seed: "test",
        readIndex: 0,
        length: 64,
        vocabularySize: 262144
      }

      const expectedNoise = yield* Noise.make(noiseOptions)
      const logits = [[Math.log(0.8), Math.log(0.2)]]
      const expectedAnswer = yield* Readout.answerFromLogits(expected.question, logits)

      const variants = [
        { renamed: target },
        { before: sibling, renamed: target },
        { renamed: target, after: sibling },
        { duplicate: target, before: sibling, renamed: target }
      ]

      for (const questions of variants) {
        const planned = yield* Planner.plan({ model, state, questions }, config)
        // Reverse insertion models scheduler completion order independently of routes.
        const completed = new Map([...planned.routes].reverse().map((route) => [route.decision.semanticKey, logits]))

        const response = yield* Readout.responseFromLogits(planned, completed, {
          input_tokens: 1,
          output_tokens: planned.routes.length
        })

        for (const route of planned.routes.filter((route) => route.decision.question.type === "choice")) {
          expect(route.decision).toEqual(expected)
          expect(yield* Noise.make({ ...noiseOptions, semanticKey: route.decision.semanticKey })).toEqual(expectedNoise)
          expect(response.answers[route.questionId]).toEqual(expectedAnswer)
        }
      }
    }))

  it.effect("canonicalizes JSON object keys, including integer-like keys, but preserves arrays", () =>
    Effect.gen(function*() {
      expect(Planner.canonicalJson({ "2": "b", "10": "a", z: { b: null, a: true } })).toBe(
        "{\"10\":\"a\",\"2\":\"b\",\"z\":{\"a\":true,\"b\":null}}"
      )
      expect(Planner.canonicalJson(["b", "a"])).toBe("[\"b\",\"a\"]")

      const first = yield* Planner.plan({
        model,
        state: { b: [1, 2], a: "x" },
        questions: { q: { type: "choice", instructions: { b: 2, a: 1 }, criteria: { x: { b: 2, a: 1 }, y: null } } }
      }, config)

      const second = yield* Planner.plan({
        model,
        state: { a: "x", b: [1, 2] },
        questions: { q: { type: "choice", instructions: { a: 1, b: 2 }, criteria: { x: { a: 1, b: 2 }, y: null } } }
      }, config)

      expect(first.routes[0].decision.semanticKey).toBe(second.routes[0].decision.semanticKey)
      expect(first.routes[0].decision.prompt).toBe(second.routes[0].decision.prompt)

      const changedArray = yield* Planner.plan({
        model,
        state: { b: [2, 1], a: "x" },
        questions: { q: first.routes[0].decision.question }
      }, config)

      expect(changedArray.routes[0].decision.semanticKey).not.toBe(first.routes[0].decision.semanticKey)
    }))

  it.effect("preserves Choice order and names as semantics, and Score level order", () =>
    Effect.gen(function*() {
      const variants = [
        { type: "choice", criteria: { a: null, b: null } },
        { type: "choice", criteria: { b: null, a: null } },
        { type: "choice", criteria: { renamed: null, b: null } },
        { type: "score", criteria: [{ label: "low" }, ["high"]] },
        { type: "score", criteria: [["high"], { label: "low" }] }
      ]

      const plans = yield* Effect.forEach(variants, (q) => Planner.plan({ model, state, questions: { q } }, config))
      const decisions = plans.map((p) => p.routes[0].decision)
      expect(new Set(decisions.map((p) => p.semanticKey)).size).toBe(5)
      expect(decisions[0].options.map((o) => o.name)).toEqual(["a", "b"])
      expect(decisions[1].options.map((o) => o.name)).toEqual(["b", "a"])
      expect(decisions[3].options.map((o) => o.description)).toEqual([{ label: "low" }, ["high"]])
    }))

  it.effect("does not mutate or retain mutable caller-owned input", () =>
    Effect.gen(function*() {
      const input = structuredClone({ model, state, questions: { q: target } })
      const before = structuredClone(input)
      const planned = yield* Planner.plan(input, config)
      const decision = planned.routes[0].decision
      const snapshot = JSON.stringify(planned)
      expect(input).toEqual(before)
      expect(Object.isFrozen(input)).toBe(false)
      expect(Object.isFrozen(input.questions.q.criteria.engineering)).toBe(false)
      input.questions.q.criteria.engineering.examples.push("caller change")
      input.state.ticket = "caller change"
      expect(JSON.stringify(planned)).toBe(snapshot)
      expect(Object.isFrozen(decision.question)).toBe(true)
      expect(Object.isFrozen(decision.options[0].description)).toBe(true)
      expect(Object.isFrozen(planned.routes)).toBe(true)
    }))

  it.effect("normalizes absent instructions and Noul descriptions before hashing", () =>
    Effect.gen(function*() {
      const absent = yield* Planner.plan({ model, state, questions: { q: { type: "noul" } } }, config)

      const explicit = yield* Planner.plan({
        model,
        state,
        questions: { q: { type: "noul", instructions: null, criteria: { false: null, true: null } } }
      }, config)

      expect(absent.routes[0].decision).toEqual(explicit.routes[0].decision)
    }))

  it.effect("includes configured model identity in semantic keys", () =>
    Effect.gen(function*() {
      const a = yield* Planner.plan({ model, state, questions: { q: target } }, config)

      const b = yield* Planner.plan({ model: "local-other-revision", state, questions: { q: target } }, {
        modelId: "local-other-revision"
      })

      expect(a.routes[0].decision.semanticKey).not.toBe(b.routes[0].decision.semanticKey)
    }))
})

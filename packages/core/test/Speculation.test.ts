import { describe, expect } from "@effect/vitest"
import { Effect } from "effect"
import { Model, Speculation } from "../src/index.ts"
import { onDevices } from "./utils/devices.ts"

const plan = (overrides: Partial<Model.ProposerPlan> = {}): Model.ProposerPlan => ({
  target: { vocabulary: 16 },
  stages: [{ operation: { _tag: "Autoregressive", component: 0 } }],
  state: { _tag: "Kv", commit: { _tag: "AutoregressiveChain", stage: 0 } },
  output: { topology: "Chains", probabilities: "CausalNormalized" },
  tokenMap: { _tag: "Identity" },
  trainedMaxRows: 4,
  ...overrides
})

onDevices("Speculation", () => (it) => {
  describe("artifact", () => {
    it.effect("constructs an opaque autoregressive chain artifact", () =>
      Effect.gen(function*() {
        const model = yield* Model.embedding("draft", 16, 8)
        const params = yield* model.init
        const artifact = yield* Speculation.artifact({
          components: [{ model, params }],
          plan: plan()
        })
        expect(artifact[Speculation.ProposerArtifactTypeId]).toBe(Speculation.ProposerArtifactTypeId)
        expect(Object.isFrozen(artifact)).toBe(true)
      }))

    it.effect("rejects invalid stage references and trained limits", () =>
      Effect.gen(function*() {
        const model = yield* Model.embedding("draft", 16, 8)
        const params = yield* model.init
        const stageError = yield* Effect.flip(Speculation.artifact({
          components: [{ model, params }],
          plan: plan({ stages: [{ operation: { _tag: "Autoregressive", component: 1 } }] })
        }))
        expect(stageError.message).toMatch(/component 0/)
        const limitError = yield* Effect.flip(Speculation.artifact({
          components: [{ model, params }],
          plan: plan({ trainedMaxRows: 0 })
        }))
        expect(limitError.message).toMatch(/trainedMaxRows/)
      }))

    it.effect("validates proposer parameter arity", () =>
      Effect.gen(function*() {
        const model = yield* Model.embedding("draft", 16, 8)
        const error = yield* Effect.flip(Speculation.artifact({
          components: [{ model, params: [] }],
          plan: plan()
        }))
        expect(error._tag).toBe("ModelError")
      }))

    it.effect("returns typed errors for malformed untyped artifacts", () =>
      Effect.gen(function*() {
        // @ts-expect-error Exercise the JavaScript trust boundary.
        const missingInput = yield* Effect.flip(Speculation.artifact(null))
        expect(missingInput._tag).toBe("InferenceError")

        const model = yield* Model.embedding("draft", 16, 8)
        const params = yield* model.init
        const malformedPlan = yield* Effect.flip(Speculation.artifact({
          components: [{ model, params }],
          // @ts-expect-error Exercise a malformed nested descriptor without an unsafe assertion.
          plan: { target: null }
        }))
        expect(malformedPlan._tag).toBe("InferenceError")
      }))
  })
})

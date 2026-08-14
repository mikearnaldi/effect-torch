import { describe, expect } from "@effect/vitest"
import { Effect } from "effect"
import { Model, Speculation, Tensor } from "../src/index.ts"
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

const targetCoupledPlan = (): Model.ProposerPlan => ({
  target: {
    graphFingerprint: "graph-v1",
    checkpointFingerprint: "checkpoint-v1",
    vocabulary: 16,
    tokenMapFingerprint: "identity-v1",
    hiddenTaps: [{ layer: 7, dtype: "f32", shape: ["Rows", 8] }],
    sharedWeights: [{ kind: "TokenEmbedding", name: "wte.weight", dtype: "f32", shape: [16, 8] }]
  },
  stages: [
    {
      operation: { _tag: "ParallelBlock", component: 0, layout: { id: "target-hidden-block-v1" } },
      inputs: [
        { slot: 0, value: { _tag: "TargetHidden", layer: 7 } },
        { slot: 1, value: { _tag: "SharedTokenEmbedding" } }
      ],
      outputs: [{ dtype: "f32", shape: ["Rows", 8] }]
    },
    {
      operation: { _tag: "SequentialHead", component: 1 },
      inputs: [{ slot: 0, value: { _tag: "StageOutput", stage: 0, output: 0 } }],
      outputs: [
        { dtype: "u32", shape: ["Rows"] },
        { dtype: "f32", shape: ["Rows", "Vocabulary"] }
      ]
    }
  ],
  state: { _tag: "None" },
  output: {
    topology: "Chains",
    probabilities: "CausalNormalized",
    tokenIds: { _tag: "StageOutput", stage: 1, output: 0 },
    probabilityRows: { _tag: "StageOutput", stage: 1, output: 1 }
  },
  tokenMap: { _tag: "Identity", fingerprint: "identity-v1" },
  trainedMaxRows: 8
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
        expect(stageError.message).toMatch(/component 1/)
        const limitError = yield* Effect.flip(Speculation.artifact({
          components: [{ model, params }],
          plan: plan({ trainedMaxRows: 0 })
        }))
        expect(limitError.message).toMatch(/trainedMaxRows/)
      }))

    it.effect("validates and retains a multi-stage target-coupled structure", () =>
      Effect.gen(function*() {
        const block = yield* Model.embedding("block", 16, 8)
        const head = yield* Model.embedding("head", 16, 8)
        const artifact = yield* Speculation.artifact({
          components: [{ model: block, params: yield* block.init }, { model: head, params: yield* head.init }],
          plan: targetCoupledPlan()
        })
        expect(artifact[Speculation.ProposerArtifactTypeId]).toBe(Speculation.ProposerArtifactTypeId)
      }))

    it.effect("accepts only the canonical zero-component history lookup contract", () =>
      Effect.gen(function*() {
        const valid: Model.ProposerPlan = {
          target: { vocabulary: 16 },
          stages: [{
            operation: {
              _tag: "HistoryLookup",
              layout: { id: "suffix-ngram-v1", minMatchTokens: 1, maxMatchTokens: 4 }
            },
            inputs: [],
            outputs: [{ dtype: "u32", shape: ["Rows"] }]
          }],
          state: { _tag: "None" },
          output: {
            topology: "Chains",
            probabilities: "Deterministic",
            tokenIds: { _tag: "StageOutput", stage: 0, output: 0 }
          },
          tokenMap: { _tag: "Identity" },
          trainedMaxRows: 4
        }
        const artifact = yield* Speculation.artifact({ components: [], plan: valid })
        expect(artifact[Speculation.ProposerArtifactTypeId]).toBe(Speculation.ProposerArtifactTypeId)

        const bounds = yield* Effect.flip(Speculation.artifact({
          components: [],
          plan: {
            ...valid,
            stages: [{
              operation: {
                _tag: "HistoryLookup",
                layout: { id: "suffix-ngram-v1", minMatchTokens: 4, maxMatchTokens: 3 }
              },
              inputs: [],
              outputs: [{ dtype: "u32", shape: ["Rows"] }]
            }]
          }
        }))
        expect(bounds.message).toMatch(/positive integer match bounds/)

        const probabilityRows = yield* Effect.flip(Speculation.artifact({
          components: [],
          plan: {
            ...valid,
            output: {
              ...valid.output,
              probabilities: "CausalNormalized",
              probabilityRows: { _tag: "StageOutput", stage: 0, output: 0 }
            }
          }
        }))
        expect(probabilityRows.message).toMatch(/probabilityRows|HistoryLookup requires/)
      }))

    it.effect("rejects forward, missing-output, and duplicate-slot references", () =>
      Effect.gen(function*() {
        const block = yield* Model.embedding("block", 16, 8)
        const head = yield* Model.embedding("head", 16, 8)
        const components = [{ model: block, params: yield* block.init }, { model: head, params: yield* head.init }]
        const valid = targetCoupledPlan()
        const first = valid.stages[0]!
        const second = valid.stages[1]!
        const forward = yield* Effect.flip(Speculation.artifact({
          components,
          plan: {
            ...valid,
            stages: [{
              ...first,
              inputs: [{ slot: 0, value: { _tag: "StageOutput", stage: 1, output: 0 } }]
            }, second]
          }
        }))
        expect(forward.message).toMatch(/backward stage/)
        const missing = yield* Effect.flip(Speculation.artifact({
          components,
          plan: {
            ...valid,
            stages: [first, {
              ...second,
              inputs: [{ slot: 0, value: { _tag: "StageOutput", stage: 0, output: 4 } }]
            }]
          }
        }))
        expect(missing.message).toMatch(/missing stage 0 output 4/)
        const duplicate = yield* Effect.flip(Speculation.artifact({
          components,
          plan: {
            ...valid,
            stages: [{
              ...first,
              inputs: [
                { slot: 0, value: { _tag: "TargetHidden", layer: 7 } },
                { slot: 0, value: { _tag: "SharedTokenEmbedding" } }
              ]
            }, second]
          }
        }))
        expect(duplicate.message).toMatch(/duplicate input slot 0/)
      }))

    it.effect("rejects malformed target, schema, output, token-map, and commit contracts", () =>
      Effect.gen(function*() {
        const block = yield* Model.embedding("block", 16, 8)
        const head = yield* Model.embedding("head", 16, 8)
        const components = [{ model: block, params: yield* block.init }, { model: head, params: yield* head.init }]
        const valid = targetCoupledPlan()
        const duplicateTap = yield* Effect.flip(Speculation.artifact({
          components,
          plan: {
            ...valid,
            target: {
              ...valid.target,
              hiddenTaps: [
                { layer: 7, dtype: "f32", shape: ["Rows", 8] },
                { layer: 7, dtype: "f16", shape: ["Rows", 8] }
              ]
            }
          }
        }))
        expect(duplicateTap.message).toMatch(/duplicate target hidden tap/)
        const badSchema = yield* Effect.flip(Speculation.artifact({
          components,
          plan: {
            ...valid,
            stages: [valid.stages[0]!, {
              ...valid.stages[1]!,
              outputs: [{ dtype: "f32", shape: ["Rows"] }, { dtype: "f32", shape: ["Rows", "Vocabulary"] }]
            }]
          }
        }))
        expect(badSchema.message).toMatch(/tokenIds must reference a rank-1 integer/)
        const treeWithoutParents = yield* Effect.flip(Speculation.artifact({
          components,
          plan: {
            ...valid,
            output: {
              topology: "Trees",
              probabilities: "CausalNormalized",
              tokenIds: { _tag: "StageOutput", stage: 1, output: 0 },
              probabilityRows: { _tag: "StageOutput", stage: 1, output: 1 }
            }
          }
        }))
        expect(treeWithoutParents.message).toMatch(/Trees output requires parents/)
        const mapMismatch = yield* Effect.flip(Speculation.artifact({
          components,
          plan: { ...valid, tokenMap: { _tag: "Identity", fingerprint: "other" } }
        }))
        expect(mapMismatch.message).toMatch(/fingerprint does not match/)
        const stateful: Model.ProposerPlan = {
          ...plan(),
          stages: [
            { operation: { _tag: "Autoregressive", component: 0 } },
            {
              operation: { _tag: "Autoregressive", component: 1 },
              inputs: [{ slot: 0, value: { _tag: "StageOutput", stage: 0, output: 0 } }],
              outputs: [{ dtype: "u32", shape: ["Rows"] }]
            }
          ],
          state: { _tag: "Kv", schema: { id: "kv-v1" }, commit: { _tag: "Replay", stages: [0] } }
        }
        const uncovered = yield* Effect.flip(Speculation.artifact({ components, plan: stateful }))
        expect(uncovered.message).toMatch(/does not cover stateful stage 1/)
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

    it.effect("accepts graph-builder components without a component tag", () =>
      Effect.gen(function*() {
        const artifact = yield* Speculation.artifact({
          components: [{
            params: [],
            build: (_, inputs) => Effect.map(Tensor.relu(inputs[0]!), (output) => [output])
          }],
          plan: {
            target: { vocabulary: 16 },
            stages: [{
              operation: { _tag: "SequentialHead", component: 0 },
              inputs: [{ slot: 0, value: { _tag: "PendingTokens" } }],
              outputs: [{ dtype: "u32", shape: ["Rows"] }]
            }],
            state: { _tag: "None" },
            output: {
              topology: "Chains",
              probabilities: "Deterministic",
              tokenIds: { _tag: "StageOutput", stage: 0, output: 0 }
            },
            tokenMap: { _tag: "Identity" },
            trainedMaxRows: 4
          }
        })
        expect(artifact[Speculation.ProposerArtifactTypeId]).toBe(Speculation.ProposerArtifactTypeId)
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

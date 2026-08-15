import { expect } from "@effect/vitest"
import { Effect } from "effect"
import { Model, Speculation, Tensor } from "../src/index.ts"
import { onDevices } from "./utils/devices.ts"

onDevices("Speculation", () => (it) => {
  it.effect("constructs an exact autoregressive artifact", () =>
    Effect.gen(function*() {
      const model = yield* Model.embedding("draft", 16, 8)
      const params = yield* model.init
      expect(Speculation.autoregressive(model, params, { vocabulary: 16, maxDraftTokens: 4 })).toEqual({
        _tag: "Autoregressive",
        model,
        params,
        vocabulary: 16,
        maxDraftTokens: 4
      })
    }))

  it("constructs a HistoryLookup artifact", () => {
    expect(Speculation.historyLookup({
      vocabulary: 16,
      maxDraftTokens: 4,
      minMatchTokens: 1,
      maxMatchTokens: 8
    })).toEqual({
      _tag: "HistoryLookup",
      vocabulary: 16,
      maxDraftTokens: 4,
      minMatchTokens: 1,
      maxMatchTokens: 8
    })
  })

  it.effect("constructs a replayable parallel block", () =>
    Effect.gen(function*() {
      const build = (_: Model.Params, input: Tensor.Any) => Tensor.relu(input)
      const replay = (_: Model.Params, inputs: ReadonlyArray<Tensor.Any>) =>
        Effect.succeed(inputs.map((input) => ({ key: input as Tensor.Lazy, value: input as Tensor.Lazy })))
      const artifact = Speculation.parallelBlock({
        params: [],
        vocabulary: 16,
        maxDraftTokens: 4,
        hiddenTaps: [{ layer: 2, dtype: "f32", shape: ["Rows", 8] }],
        tokenEmbedding: { name: "wte.weight", dtype: "f32", shape: [16, 8] },
        lmHead: { name: "head.weight", dtype: "f32", shape: [16, 8] },
        build,
        replay
      })
      expect(artifact._tag).toBe("ParallelBlock")
      expect(artifact.build).toBe(build)
      expect(artifact.replay).toBe(replay)
    }))
})

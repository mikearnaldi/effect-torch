import { expect } from "@effect/vitest"
import { Effect } from "effect"
import { Model, Speculation, Tensor } from "../src/index.ts"
import { onDevices } from "./utils/devices.ts"

onDevices("Speculation", () => (it) => {
  it.effect("constructs an exact autoregressive artifact", () =>
    Effect.gen(function*() {
      const model = yield* Model.embedding("draft", 16, 8)
      const params = yield* Model.initialize(model)
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
        Effect.sync(() =>
          inputs.map((input) => {
            if (!Tensor.isLazyTensor(input)) throw new Error("replay inputs must be lazy tensors")
            return { key: input, value: input }
          })
        )
      const artifact = Speculation.parallelBlock({
        params: [],
        vocabulary: 16,
        maxDraftTokens: 4,
        hiddenTaps: [{ name: "layers.2.hidden", dtype: "f32", shape: ["Rows", 8] }],
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

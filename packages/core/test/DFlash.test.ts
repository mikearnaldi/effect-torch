import { DFlash } from "@effect-torch/core/models"
import { expect, it } from "@effect/vitest"
import { Effect } from "effect"
import { Model, type Runtime } from "../src/index.ts"

const metadata = (): ReadonlyMap<string, unknown> =>
  new Map<string, unknown>([
    ["architecture", "dflash"],
    ["block_count", 5],
    ["context_length", 131072],
    ["embedding_length", 6656],
    ["feed_forward_length", 19968],
    ["attention.head_count", 32],
    ["attention.head_count_kv", 8],
    ["attention.key_length", 128],
    ["attention.value_length", 128],
    ["attention.layer_norm_rms_epsilon", Math.fround(1e-5)],
    ["attention.sliding_window", 2048],
    ["rope.freq_base", 500000],
    ["attention.sliding_window_pattern", [true, true, true, true, true]],
    ["block_size", 16],
    ["target_layers", [2, 14, 26, 38, 50]],
    ["tokenizer.ggml.mask_token_id", 201818],
    ["vocab_size", 202048]
  ])

const descriptor = (name: string): Runtime.GgufTensorDescriptor => ({
  name,
  format: "F32",
  logicalShape: [1],
  logicalDtype: "f32",
  physicalShape: [1],
  physicalDtype: "f32"
})

const catalog = (outputNorm = true): ReadonlyArray<Runtime.GgufTensorDescriptor> => [
  descriptor("fc.weight"),
  descriptor("enc.output_norm.weight"),
  ...Array.from({ length: 5 }, (_, layer) =>
    [
      "attn_norm.weight",
      "ffn_down.weight",
      "ffn_gate.weight",
      "ffn_up.weight",
      "ffn_norm.weight",
      "attn_k_norm.weight",
      "attn_k.weight",
      "attn_output.weight",
      "attn_q_norm.weight",
      "attn_q.weight",
      "attn_v.weight"
    ].map((name) => descriptor(`blk.${layer}.${name}`))).flat(),
  ...(outputNorm ? [descriptor("output_norm.weight")] : [])
]

it.effect("defines the exact official DFlash parameter catalog", () =>
  Effect.gen(function*() {
    const parameters = yield* DFlash.definition.parameters(metadata(), catalog())
    expect(DFlash.architecture).toBe("dflash")
    expect(parameters).toHaveLength(58)
    expect(parameters.slice(0, 2)).toEqual([
      { name: "fc.weight", shape: [6656, 33280] },
      { name: "enc.output_norm.weight", shape: [6656] }
    ])
    expect(parameters.slice(2, 13)).toEqual([
      { name: "blk.0.attn_norm.weight", shape: [6656] },
      { name: "blk.0.ffn_down.weight", shape: [6656, 19968] },
      { name: "blk.0.ffn_gate.weight", shape: [19968, 6656] },
      { name: "blk.0.ffn_up.weight", shape: [19968, 6656] },
      { name: "blk.0.ffn_norm.weight", shape: [6656] },
      { name: "blk.0.attn_k_norm.weight", shape: [128] },
      { name: "blk.0.attn_k.weight", shape: [1024, 6656] },
      { name: "blk.0.attn_output.weight", shape: [6656, 4096] },
      { name: "blk.0.attn_q_norm.weight", shape: [128] },
      { name: "blk.0.attn_q.weight", shape: [4096, 6656] },
      { name: "blk.0.attn_v.weight", shape: [1024, 6656] }
    ])
    expect(parameters.at(-1)).toEqual({ name: "output_norm.weight", shape: [6656] })

    const withoutOutputNorm = yield* DFlash.definition.parameters(metadata(), catalog(false))
    expect(withoutOutputNorm).toHaveLength(57)
    expect(withoutOutputNorm.some(({ name }) => name === "output_norm.weight")).toBe(false)
  }))

it.effect("rejects metadata that changes official DFlash geometry", () =>
  Effect.gen(function*() {
    for (
      const [key, value] of [
        ["architecture", "DFlash"],
        ["block_count", 4],
        ["context_length", 131071],
        ["embedding_length", 6655],
        ["feed_forward_length", 19967],
        ["attention.head_count", 31],
        ["attention.head_count_kv", 7],
        ["attention.key_length", 64],
        ["attention.layer_norm_rms_epsilon", 1e-6],
        ["attention.sliding_window", 1024],
        ["rope.freq_base", 10000],
        ["block_size", 15],
        ["vocab_size", 202047],
        ["target_layers", [1, 13, 25, 37, 49]]
      ] as const
    ) {
      const invalid = new Map(metadata())
      invalid.set(key, value)
      const error = yield* Effect.flip(DFlash.definition.parameters(invalid, catalog()))
      expect(error.message).toContain(key)
    }
  }))

it("declares proven residual taps and the KV Replay lifecycle", () => {
  expect(DFlash.targetLayers).toEqual([2, 14, 26, 38, 50])
  expect(DFlash.targetResidualTaps).toEqual([1, 13, 25, 37, 49])
  expect(DFlash.programs).toEqual([
    {
      kind: "FeatureFusionKvInjection",
      targetResidualTaps: [1, 13, 25, 37, 49],
      fusedWidth: 33280,
      kvLayers: 5,
      hiddenSize: 6656,
      kvHeads: 8,
      headDim: 128
    },
    {
      kind: "MaskedNonCausalBlockDecode",
      blockSize: 16,
      maskToken: 201818,
      hiddenSize: 6656,
      feedForwardSize: 19968,
      queryHeads: 32,
      kvHeads: 8,
      headDim: 128,
      slidingWindow: 2048,
      ropeBase: 500000,
      rmsEpsilon: 1e-5,
      sharedTokenEmbedding: "token_embd.weight",
      sharedLmHead: "output.weight"
    }
  ])
  expect(DFlash.plan.state).toEqual({
    _tag: "Kv",
    schema: { id: "dflash-kv-5x8x128-swa2048-v1" },
    commit: { _tag: "Replay", stages: [0] }
  })
  expect(DFlash.plan.target.hiddenTaps?.map(({ layer }) => layer)).toEqual([1, 13, 25, 37, 49])
  expect(DFlash.plan.target.sharedWeights?.map(({ name }) => name)).toEqual(["token_embd.weight", "output.weight"])
  expect(DFlash.plan.output.probabilities).toBe("Unavailable")
  expect(DFlash.plan.trainedMaxRows).toBe(15)
})

it.effect("validates the artifact and rejects unsupported stateful graph compilation", () =>
  Effect.gen(function*() {
    const component = {
      params: [],
      build: () =>
        new Model.ModelError({
          op: "dflashCompile",
          message: "DFlash KV injection is not represented by the tensor graph"
        })
    }
    const artifact = yield* Model.Speculation.artifact({ components: [component], plan: DFlash.plan })
    expect(artifact[Model.ProposerArtifactTypeId]).toBe(Model.ProposerArtifactTypeId)
    const error = yield* Effect.flip(component.build())
    expect(error.op).toBe("dflashCompile")
    expect(error.message).toContain("KV injection")
  }))

import { DFlash } from "@effect-torch/core/models"
import { expect, it } from "@effect/vitest"
import { Effect, Layer } from "effect"
import { Runtime, type Tensor } from "../src/index.ts"

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

const placement: Runtime.Placement = {
  id: "dflash-test:0",
  deviceType: "test",
  description: "DFlash shape-only graph runtime"
}

const tensor = (
  shape: ReadonlyArray<number>,
  dtype: Tensor.DType = "f32",
  storage?: Runtime.EncodedTensorStorage
): Tensor.Any =>
  ({
    _tag: "LazyTensor",
    shape,
    dtype,
    ...(storage === undefined ? {} : { storage }),
    device: placement.deviceType,
    placement,
    pipe() {
      throw new Error("unused test tensor pipe")
    }
  }) as unknown as Tensor.Any

const graphRequests: Array<Runtime.NodeRequest> = []
const graphRuntime = {
  identity: {},
  backend: { name: "dflash-test" },
  placement,
  capabilities: { dtypes: ["f32", "i64", "u32"], features: [] },
  extensions: {},
  node: (request: Runtime.NodeRequest) =>
    Effect.sync(() => {
      graphRequests.push(request)
      const first = request.inputs[0]!
      switch (request.op) {
        case "constant":
          return tensor([], request.attributes.dtype) as Tensor.Lazy
        case "full":
          return tensor(request.attributes.shape, request.attributes.dtype) as Tensor.Lazy
        case "quantizedEmbedding":
          return tensor(
            [...request.inputs[0].shape, request.attributes.logicalShape[1]],
            "f32"
          ) as Tensor.Lazy
        case "quantizedLinear":
          return tensor(
            [...request.inputs[0].shape.slice(0, -1), request.attributes.logicalShape[0]],
            "f32"
          ) as Tensor.Lazy
        case "concat": {
          const shape = [...request.inputs[0].shape]
          shape[request.attributes.dim] += request.inputs[1].shape[request.attributes.dim]
          return tensor(shape, first.dtype) as Tensor.Lazy
        }
        case "reshape":
          return tensor(request.attributes.shape, first.dtype) as Tensor.Lazy
        case "permute":
          return tensor(request.attributes.dims.map((axis) => first.shape[axis]), first.dtype) as Tensor.Lazy
        case "slice":
          return tensor(
            request.attributes.ranges.map(([start, stop, stride]) => Math.ceil((stop - start) / stride)),
            first.dtype
          ) as Tensor.Lazy
        case "argmax":
          return tensor(first.shape.filter((_, axis) => axis !== request.attributes.dim), "i64") as Tensor.Lazy
        case "cast":
          return tensor(first.shape, request.attributes.dtype) as Tensor.Lazy
        case "scaledDotProductAttention":
          return tensor([...first.shape.slice(0, -1), request.inputs[2].shape.at(-1)!], first.dtype) as Tensor.Lazy
        case "max":
        case "sum": {
          const dims = new Set(request.attributes.dims)
          return tensor(
            first.shape.flatMap((dimension, axis) =>
              dims.has(axis) ? request.attributes.keepdims ? [1] : [] : [dimension]
            ),
            first.dtype
          ) as Tensor.Lazy
        }
        case "rmsNorm":
        case "rotaryEmbedding":
        case "tanh":
        case "add":
        case "div":
        case "exp":
        case "mul":
        case "sub":
          return tensor(first.shape, first.dtype) as Tensor.Lazy
        default:
          throw new Error(`unexpected DFlash graph operation ${request.op}`)
      }
    })
} as unknown as Runtime.RuntimeService

const graphLayer = Layer.succeed(Runtime.Runtime, graphRuntime)

const packed = (shape: ReadonlyArray<number>): Tensor.Any =>
  tensor(shape, "f32", { encoding: "Q2_K", physicalShape: [shape[0], 1], physicalDtype: "u8" })

const tinyParams = (withOutputNorm = true): Array<Tensor.Any> => {
  const params: Array<Tensor.Any> = [packed([8, 40]), tensor([8])]
  for (let layer = 0; layer < 5; layer++) {
    params.push(
      tensor([8]),
      packed([8, 16]),
      packed([16, 8]),
      packed([16, 8]),
      tensor([8]),
      tensor([4]),
      packed([4, 8]),
      packed([8, 8]),
      tensor([4]),
      packed([8, 8]),
      packed([4, 8])
    )
  }
  if (withOutputNorm) params.push(tensor([8]))
  return params
}

it.effect("derives the reference checkpoint parameter catalog", () =>
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

    const requiredCatalog = yield* DFlash.definition.parameters(metadata(), catalog(false))
    expect(requiredCatalog).toHaveLength(58)
    expect(requiredCatalog.at(-1)).toEqual({ name: "output_norm.weight", shape: [6656] })
  }))

it.effect("loadGGUF rejects artifacts for another architecture", () => {
  let loaded = false
  const ggufRuntime = {
    extensions: {
      gguf: {
        inspect: () =>
          Effect.succeed({
            metadata: [{ key: "general.architecture", value: "muse-glimmer" }],
            tensors: []
          }),
        load: () =>
          Effect.sync(() => {
            loaded = true
            return { entries: [] }
          })
      }
    }
  } as unknown as Runtime.RuntimeService
  return Effect.gen(function*() {
    const error = yield* Effect.flip(DFlash.loadGGUF("target.gguf"))
    expect(error._tag).toBe("GgufError")
    expect(error.op).toBe("validate")
    expect(error.message).toContain("\"dflash\"")
    expect(loaded).toBe(false)
  }).pipe(Effect.provide(Layer.succeed(Runtime.Runtime, ggufRuntime)))
})

it.effect("derives catalogs and artifacts from changed valid metadata", () =>
  Effect.gen(function*() {
    const changed = new Map(metadata())
    for (
      const [key, value] of [
        ["block_count", 3],
        ["context_length", 128],
        ["embedding_length", 12],
        ["feed_forward_length", 24],
        ["attention.head_count", 4],
        ["attention.head_count_kv", 2],
        ["attention.key_length", 4],
        ["attention.value_length", 4],
        ["attention.layer_norm_rms_epsilon", 1e-6],
        ["attention.sliding_window", 32],
        ["rope.freq_base", 10000],
        ["attention.sliding_window_pattern", [true, true, true]],
        ["block_size", 8],
        ["target_layers", [1, 4, 9]],
        ["tokenizer.ggml.mask_token_id", 63],
        ["vocab_size", 64]
      ] as const
    ) changed.set(key, value)

    const config = yield* DFlash.configuration(changed)
    const parameters = yield* DFlash.definition.parameters(changed, catalog())
    const artifact = DFlash.artifact(config, [])

    expect(config.targetResidualTaps).toEqual([0, 3, 8])
    expect(parameters).toHaveLength(36)
    expect(parameters.slice(0, 3)).toEqual([
      { name: "fc.weight", shape: [12, 36] },
      { name: "enc.output_norm.weight", shape: [12] },
      { name: "blk.0.attn_norm.weight", shape: [12] }
    ])
    expect(parameters.find(({ name }) => name === "blk.2.attn_q.weight")?.shape).toEqual([16, 12])
    expect(parameters.find(({ name }) => name === "blk.2.attn_v.weight")?.shape).toEqual([8, 12])
    expect(parameters.at(-1)).toEqual({ name: "output_norm.weight", shape: [12] })
    expect(artifact.hiddenTaps.map(({ layer, shape }) => ({ layer, shape }))).toEqual([
      { layer: 0, shape: ["Rows", 12] },
      { layer: 3, shape: ["Rows", 12] },
      { layer: 8, shape: ["Rows", 12] }
    ])
    expect([artifact.tokenEmbedding.shape, artifact.lmHead.shape]).toEqual([[64, 12], [64, 12]])
    expect(artifact.maxDraftTokens).toBe(7)
  }))

it.effect("rejects invalid relational metadata", () =>
  Effect.gen(function*() {
    for (
      const [key, value, message] of [
        ["architecture", "DFlash", "architecture"],
        ["attention.head_count", 31, "divisible"],
        ["attention.value_length", 64, "must equal"],
        ["block_size", 1, "2..=context_length"],
        ["target_layers", [], "at least two"],
        ["target_layers", [2, 0], "greater than 0"],
        ["attention.sliding_window_pattern", [true], "length must equal block_count"],
        ["attention.sliding_window_pattern", [true, true, false, true, true], "all draft layers"],
        ["attention.sliding_window", 131073, "must not exceed"],
        ["tokenizer.ggml.mask_token_id", 202048, "less than vocab_size"]
      ] as const
    ) {
      const invalid = new Map(metadata())
      invalid.set(key, value)
      const error = yield* Effect.flip(DFlash.configuration(invalid))
      expect(error.message).toContain(message)
    }

    const oddHeadWidth = new Map(metadata())
    oddHeadWidth.set("attention.key_length", 3)
    oddHeadWidth.set("attention.value_length", 3)
    const error = yield* Effect.flip(DFlash.configuration(oddHeadWidth))
    expect(error.message).toContain("must be even")
  }))

it.effect("derives the reference checkpoint replay artifact", () =>
  Effect.gen(function*() {
    const config = yield* DFlash.configuration(metadata())
    const artifact = DFlash.artifact(config, [])
    expect(config.targetLayers).toEqual([2, 14, 26, 38, 50])
    expect(config.targetResidualTaps).toEqual([1, 13, 25, 37, 49])
    expect(artifact.hiddenTaps.map(({ layer }) => layer)).toEqual([1, 13, 25, 37, 49])
    expect([artifact.tokenEmbedding.name, artifact.lmHead.name]).toEqual(["token_embd.weight", "output.weight"])
    expect(artifact.maxDraftTokens).toBe(15)
    expect(artifact.currentBlockAttention).toBe("Bidirectional")
  }))

it.effect("builds replay and noncausal block graphs at tiny geometry", () => {
  const config: DFlash.Configuration = {
    blockCount: 5,
    contextLength: 32,
    embeddingLength: 8,
    feedForwardLength: 16,
    queryHeads: 2,
    kvHeads: 1,
    keyLength: 4,
    valueLength: 4,
    rmsEpsilon: 1e-5,
    slidingWindow: 8,
    ropeBase: 500000,
    blockSize: 4,
    vocabularySize: 32,
    maskToken: 31,
    targetLayers: [1, 2, 3, 4, 5],
    targetResidualTaps: [0, 1, 2, 3, 4],
    slidingWindowPattern: [true, true, true, true, true]
  }
  return Effect.gen(function*() {
    graphRequests.length = 0
    const params = tinyParams()
    const artifact = DFlash.artifact(config, params as unknown as ReadonlyArray<Tensor.Concrete>)
    const replay = yield* artifact.replay(params, Array.from({ length: 5 }, () => tensor([2, 8])))
    expect(replay).toHaveLength(5)
    expect(replay.map(({ key, value }) => ({
      key: key.shape,
      value: value.shape
    }))).toEqual(Array.from({ length: 5 }, () => ({
      key: [1, 2, 4],
      value: [1, 2, 4]
    })))
    const batchedReplay = yield* artifact.replay(params, Array.from({ length: 5 }, () => tensor([2, 3, 8])))
    expect(batchedReplay.map(({ key, value }) => ({
      key: key.shape,
      value: value.shape
    }))).toEqual(Array.from({ length: 5 }, () => ({
      key: [2, 1, 3, 4],
      value: [2, 1, 3, 4]
    })))

    const output = yield* artifact.build(params, tensor([2], "u32"), packed([32, 8]), packed([32, 8]), 2)
    expect(output.shape).toEqual([2, 2])
    expect(output.dtype).toBe("u32")
    const attention = graphRequests.filter(({ op }) => op === "scaledDotProductAttention")
    expect(attention).toHaveLength(5)
    expect(
      attention.every((request) => request.op === "scaledDotProductAttention" && request.attributes.causal === false)
    )
      .toBe(true)
    const rotary = graphRequests.filter(({ op }) => op === "rotaryEmbedding")
    expect(rotary).toHaveLength(20)
    expect(
      rotary.every((request) => request.op === "rotaryEmbedding" && request.attributes.layout === "HalfSplit")
    ).toBe(true)
    expect(graphRequests.some(({ op }) => op === "argmax")).toBe(true)
  }).pipe(Effect.provide(graphLayer))
})

it.effect("constructs the replayable block artifact", () =>
  Effect.gen(function*() {
    const config = yield* DFlash.configuration(metadata())
    const artifact = DFlash.artifact(config, [])
    expect(artifact._tag).toBe("ParallelBlock")
    expect(artifact.currentBlockAttention).toBe("Bidirectional")
    expect(artifact.attentionWindow).toBe(2048)
  }))

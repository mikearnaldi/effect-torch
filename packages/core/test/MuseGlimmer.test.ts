import { MuseGlimmer } from "@effect-torch/core/models"
import { expect, it } from "@effect/vitest"
import { Effect, Layer } from "effect"
import { Gguf, Model, Registry, Runtime, type Tensor } from "../src/index.ts"

const configEntries = [
  ["block_count", 52],
  ["embedding_length", 6656],
  ["feed_forward_length", 19968],
  ["context_length", 131072],
  ["attention.head_count", 32],
  ["attention.head_count_kv", 2],
  ["attention.key_length", 128],
  ["attention.value_length", 128],
  ["attention.layer_norm_rms_epsilon", 1e-5],
  ["attention.sliding_window", 2048],
  ["attention.sliding_window_pattern", 4],
  ["rope.freq_base", 500000],
  ["logit_scale", 0.19611613513818404],
  ["final_logit_softcapping", 20],
  ["vocab_size", 202048]
] as const

const config = (): Registry.ModelConfig => new Map<string, unknown>(configEntries)

const placement: Runtime.Placement = Object.freeze({
  id: "muse-test:0",
  deviceType: "test",
  description: "Muse-Glimmer graph test runtime"
})

const broadcast = (left: ReadonlyArray<number>, right: ReadonlyArray<number>): Array<number> => {
  const rank = Math.max(left.length, right.length)
  const shape: Array<number> = []
  for (let index = 0; index < rank; index++) {
    shape.unshift(Math.max(left[left.length - 1 - index] ?? 1, right[right.length - 1 - index] ?? 1))
  }
  return shape
}

const handle = (
  tag: "LazyTensor" | "Tensor",
  shape: ReadonlyArray<number>,
  dtype: Tensor.DType,
  storage?: Runtime.EncodedTensorStorage
): Tensor.Any =>
  Object.freeze({
    _tag: tag,
    shape,
    dtype,
    ...(storage === undefined ? {} : { storage }),
    device: placement.deviceType,
    placement,
    pipe() {
      throw new Error("unused test handle pipe")
    }
  }) as unknown as Tensor.Any

const requests: Array<Runtime.NodeRequest> = []

const runtime = {
  identity: {},
  backend: { name: "muse-test" },
  placement,
  capabilities: { dtypes: ["f32", "i64"], features: [] },
  extensions: {},
  node: (request: Runtime.NodeRequest) =>
    Effect.sync(() => {
      requests.push(request)
      switch (request.op) {
        case "constant":
          return handle("LazyTensor", [], request.attributes.dtype) as Tensor.Lazy
        case "quantizedEmbedding":
          return handle(
            "LazyTensor",
            [...request.inputs[0].shape, request.attributes.logicalShape[1]],
            "f32"
          ) as Tensor.Lazy
        case "quantizedLinear":
          return handle(
            "LazyTensor",
            [...request.inputs[0].shape.slice(0, -1), request.attributes.logicalShape[0]],
            "f32"
          ) as Tensor.Lazy
        case "rmsNorm":
          return handle("LazyTensor", request.inputs[0].shape, request.inputs[0].dtype) as Tensor.Lazy
        case "mean": {
          const dims = new Set(request.attributes.dims)
          const shape = request.inputs[0].shape.flatMap((dimension, index) =>
            dims.has(index) ? request.attributes.keepdims ? [1] : [] : [dimension]
          )
          return handle("LazyTensor", shape, request.inputs[0].dtype) as Tensor.Lazy
        }
        case "reshape":
          return handle("LazyTensor", request.attributes.shape, request.inputs[0].dtype) as Tensor.Lazy
        case "permute":
          return handle(
            "LazyTensor",
            request.attributes.dims.map((dimension) => request.inputs[0].shape[dimension]),
            request.inputs[0].dtype
          ) as Tensor.Lazy
        case "scaledDotProductAttention":
          return handle(
            "LazyTensor",
            [...request.inputs[0].shape.slice(0, -1), request.inputs[2].shape.at(-1)!],
            request.inputs[0].dtype
          ) as Tensor.Lazy
        case "add":
        case "div":
        case "mul":
          return handle(
            "LazyTensor",
            broadcast(request.inputs[0].shape, request.inputs[1].shape),
            request.inputs[0].shape.length === 0 ? request.inputs[1].dtype : request.inputs[0].dtype
          ) as Tensor.Lazy
        case "pow":
        case "rotaryEmbedding":
        case "tanh":
          return handle("LazyTensor", request.inputs[0].shape, request.inputs[0].dtype) as Tensor.Lazy
        default:
          throw new Error(`unexpected Muse-Glimmer graph operation ${request.op}`)
      }
    })
} as unknown as Runtime.RuntimeService

const runtimeLayer = Layer.succeed(Runtime.Runtime, runtime)

const modelParams = (parameters: ReadonlyArray<{ readonly shape: ReadonlyArray<number> }>): Array<Tensor.Any> =>
  parameters.map(({ shape }) =>
    handle(
      "Tensor",
      shape,
      "f32",
      shape.length === 2
        ? { encoding: "Q2_K", physicalShape: [shape[0], 1], physicalDtype: "u8" }
        : undefined
    )
  )

const expectedLayer = (layer: number) => {
  const prefix = `blk.${layer}`
  return [
    { name: `${prefix}.attn_norm.weight`, shape: [6656] },
    { name: `${prefix}.post_attention_norm.weight`, shape: [6656] },
    { name: `${prefix}.attn_q.weight`, shape: [4096, 6656] },
    { name: `${prefix}.attn_k.weight`, shape: [256, 6656] },
    { name: `${prefix}.attn_v.weight`, shape: [256, 6656] },
    { name: `${prefix}.attn_q_norm.weight`, shape: [128] },
    { name: `${prefix}.attn_k_norm.weight`, shape: [128] },
    { name: `${prefix}.attn_gate.weight`, shape: [4096, 6656] },
    { name: `${prefix}.attn_output.weight`, shape: [6656, 4096] },
    { name: `${prefix}.ffn_norm.weight`, shape: [6656] },
    { name: `${prefix}.post_ffw_norm.weight`, shape: [6656] },
    { name: `${prefix}.ffn_gate.weight`, shape: [19968, 6656] },
    { name: `${prefix}.ffn_up.weight`, shape: [19968, 6656] },
    { name: `${prefix}.ffn_down.weight`, shape: [6656, 19968] }
  ]
}

it.effect("exports and registers only the exact Muse-Glimmer architecture", () => {
  return Effect.gen(function*() {
    expect(Object.keys(MuseGlimmer).sort()).toEqual(["architecture", "id"])
    expect(MuseGlimmer.id).toBe("gguf:muse-glimmer")
    expect(MuseGlimmer.architecture.id).toBe("gguf:muse-glimmer")
    const registry = yield* Registry.Registry
    expect(yield* registry.get("gguf:muse-glimmer")).toBe(MuseGlimmer.architecture)
    expect((yield* Effect.flip(registry.get("muse-glimmer")))._tag).toBe("RegistryError")
    expect((yield* Effect.flip(registry.get("Muse-Glimmer")))._tag).toBe("RegistryError")
  }).pipe(Effect.provide(Registry.layer))
})

it.effect("validates canonical configuration with Schema", () =>
  Effect.gen(function*() {
    yield* MuseGlimmer.architecture.create(config())
    for (const [key] of configEntries) {
      const missing = new Map(config())
      missing.delete(key)
      const missingError = yield* Effect.flip(MuseGlimmer.architecture.create(missing))
      expect(missingError._tag).toBe("ModelError")
      expect(missingError.message).toContain(key)

      const invalid = new Map(config())
      invalid.set(key, 0)
      const invalidError = yield* Effect.flip(MuseGlimmer.architecture.create(invalid))
      expect(invalidError._tag).toBe("ModelError")
      expect(invalidError.message).toContain(key)
    }

    const invalidGqa = new Map(config())
    invalidGqa.set("attention.head_count", 31)
    expect((yield* Effect.flip(MuseGlimmer.architecture.create(invalidGqa))).message).toContain(
      "attention.head_count"
    )

    const invalidRope = new Map(config())
    invalidRope.set("attention.key_length", 127)
    expect((yield* Effect.flip(MuseGlimmer.architecture.create(invalidRope))).message).toContain(
      "attention.key_length"
    )

    const invalidWindow = new Map(config())
    invalidWindow.set("attention.sliding_window", 131073)
    expect((yield* Effect.flip(MuseGlimmer.architecture.create(invalidWindow))).message).toContain(
      "attention.sliding_window"
    )
  }))

it.effect("derives the parameter catalog and graph from configuration", () =>
  Effect.gen(function*() {
    const custom = new Map<string, unknown>([
      ["block_count", 1],
      ["embedding_length", 8],
      ["feed_forward_length", 16],
      ["context_length", 16],
      ["attention.head_count", 2],
      ["attention.head_count_kv", 1],
      ["attention.key_length", 4],
      ["attention.value_length", 3],
      ["attention.layer_norm_rms_epsilon", 1e-4],
      ["attention.sliding_window", 8],
      ["attention.sliding_window_pattern", 2],
      ["rope.freq_base", 10000],
      ["logit_scale", 0.25],
      ["final_logit_softcapping", 10],
      ["vocab_size", 12]
    ])
    const model = yield* MuseGlimmer.architecture.create(custom)
    expect(model.parameters).toEqual([
      { name: "token_embd.weight", shape: [12, 8] },
      { name: "output_norm.weight", shape: [8] },
      { name: "output.weight", shape: [12, 8] },
      { name: "blk.0.attn_norm.weight", shape: [8] },
      { name: "blk.0.post_attention_norm.weight", shape: [8] },
      { name: "blk.0.attn_q.weight", shape: [8, 8] },
      { name: "blk.0.attn_k.weight", shape: [4, 8] },
      { name: "blk.0.attn_v.weight", shape: [3, 8] },
      { name: "blk.0.attn_q_norm.weight", shape: [4] },
      { name: "blk.0.attn_k_norm.weight", shape: [4] },
      { name: "blk.0.attn_gate.weight", shape: [6, 8] },
      { name: "blk.0.attn_output.weight", shape: [8, 6] },
      { name: "blk.0.ffn_norm.weight", shape: [8] },
      { name: "blk.0.post_ffw_norm.weight", shape: [8] },
      { name: "blk.0.ffn_gate.weight", shape: [16, 8] },
      { name: "blk.0.ffn_up.weight", shape: [16, 8] },
      { name: "blk.0.ffn_down.weight", shape: [8, 16] }
    ])

    requests.length = 0
    const output = yield* model.forward(modelParams(model.parameters), handle("Tensor", [1, 2], "i64"))
    expect(output.shape).toEqual([1, 2, 12])
    const attention = requests.find(
      (request): request is Runtime.NodeRequest<"scaledDotProductAttention"> =>
        request.op === "scaledDotProductAttention"
    )
    expect(attention?.attributes).toMatchObject({ scale: 0.5, window: 8 })
    const rotary = requests.find(
      (request): request is Runtime.NodeRequest<"rotaryEmbedding"> => request.op === "rotaryEmbedding"
    )
    expect(rotary?.attributes.theta).toBe(10000)
    const normalizations = requests.filter(
      (request): request is Runtime.NodeRequest<"rmsNorm"> => request.op === "rmsNorm"
    )
    expect(normalizations.some(({ attributes }) => attributes.eps === 1e-4)).toBe(true)
    expect(normalizations.some(({ attributes }) => attributes.eps === 1e-8)).toBe(true)

    const contextError = yield* Effect.flip(
      model.forward(modelParams(model.parameters), handle("Tensor", [1, 17], "i64"))
    )
    expect(contextError.message).toContain("exceeds context length 16")
  }).pipe(Effect.provide(runtimeLayer)))

it.effect("receives vocab_size from generic GGUF tokenizer token translation", () => {
  let vocabSize: unknown
  const ggufRuntime = {
    ...runtime,
    extensions: {
      gguf: {
        inspect: () =>
          Effect.succeed({
            metadata: [
              { key: "general.architecture", value: "generic-vocab-test" },
              { key: "tokenizer.ggml.tokens", value: ["a", "b", "c"] }
            ],
            tensors: []
          }),
        load: () => Effect.succeed({ entries: [] })
      }
    }
  } as unknown as Runtime.RuntimeService
  const services = Layer.merge(
    Registry.emptyLayer,
    Layer.succeed(Runtime.Runtime, ggufRuntime)
  )
  return Effect.gen(function*() {
    const registry = yield* Registry.Registry
    yield* registry.register({
      id: "gguf:generic-vocab-test",
      create: (canonical) => {
        vocabSize = canonical.get("vocab_size")
        return Model.define({
          parameters: [],
          forward: (_, input) => Effect.succeed(input as Tensor.Lazy)
        })
      }
    })
    yield* Gguf.load("generic.gguf")
    expect(vocabSize).toBe(3)
  }).pipe(Effect.provide(services))
})

it.effect("defines the exact load-only parameter catalog", () =>
  Effect.gen(function*() {
    const model = yield* MuseGlimmer.architecture.create(config())
    expect(model.parameters).toHaveLength(731)
    expect(model.parameters.slice(0, 3)).toEqual([
      { name: "token_embd.weight", shape: [202048, 6656] },
      { name: "output_norm.weight", shape: [6656] },
      { name: "output.weight", shape: [202048, 6656] }
    ])
    expect(model.parameters.slice(3, 17)).toEqual(expectedLayer(0))
    expect(model.parameters.slice(3 + 27 * 14, 3 + 28 * 14)).toEqual(expectedLayer(27))
    expect(model.parameters.slice(-14)).toEqual(expectedLayer(51))
    expect(model.names).toEqual(model.parameters.map(({ name }) => name))

    const initError = yield* Effect.flip(model.init)
    expect(initError._tag).toBe("ModelError")
    expect(initError.op).toBe("init")
  }).pipe(Effect.provide(runtimeLayer)))

it.effect("reports parameter arity and token rank through ModelError", () =>
  Effect.gen(function*() {
    const model = yield* MuseGlimmer.architecture.create(config())
    const arityError = yield* Effect.flip(model.forward([], handle("Tensor", [1, 1], "i64")))
    expect(arityError._tag).toBe("ModelError")
    expect(arityError.message).toContain("expected 731 parameters")

    const rankError = yield* Effect.flip(model.forward(modelParams(model.parameters), handle("Tensor", [1], "i64")))
    expect(rankError._tag).toBe("ModelError")
    expect(rankError.message).toContain("[B, S]")
  }).pipe(Effect.provide(runtimeLayer)))

it.effect("builds the canonical graph with 39 local and 13 global layers", () =>
  Effect.gen(function*() {
    requests.length = 0
    const model = yield* MuseGlimmer.architecture.create(config())
    const output = yield* model.forward(modelParams(model.parameters), handle("Tensor", [1, 2], "i64"))
    expect(output.shape).toEqual([1, 2, 202048])

    const attention = requests.filter(
      (request): request is Runtime.NodeRequest<"scaledDotProductAttention"> =>
        request.op === "scaledDotProductAttention"
    )
    expect(attention).toHaveLength(52)
    expect(attention.map(({ attributes }) => attributes.window)).toEqual(
      Array.from({ length: 52 }, (_, layer) => layer % 4 === 3 ? null : 2048)
    )
    expect(attention.every(({ attributes }) => attributes.causal && attributes.scale === 1 / Math.sqrt(128))).toBe(
      true
    )

    const rotary = requests.filter(
      (request): request is Runtime.NodeRequest<"rotaryEmbedding"> => request.op === "rotaryEmbedding"
    )
    expect(rotary).toHaveLength(39 * 2)
    expect(
      rotary.every(({ attributes }) =>
        attributes.seqLen === 2 && attributes.theta === 500000 && attributes.layout === "InterleavedPairs"
      )
    ).toBe(true)
    expect(requests.slice(-8).map(({ op }) => op)).toEqual([
      "quantizedLinear",
      "constant",
      "mul",
      "constant",
      "div",
      "tanh",
      "constant",
      "mul"
    ])
  }).pipe(Effect.provide(runtimeLayer)))

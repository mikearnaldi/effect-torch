import { MuseGlimmer } from "@effect-torch/core/models"
import { expect, it } from "@effect/vitest"
import { Effect, Layer } from "effect"
import { Gguf, Model, Runtime, type Tensor } from "../src/index.ts"

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

const config = (): Gguf.ModelConfig => new Map<string, unknown>(configEntries)

// This shape-only runtime records graph requests and fabricates coherent handles.
// It does not implement compilation or execution. Model validation still
// observes the declared placement, dtype, and shape, so topology checks exercise
// the real API.
const placement: Runtime.Placement = Object.freeze({
  id: "muse-test:0",
  deviceType: "test",
  description: "Muse-Glimmer graph test runtime"
})

type RuntimeDouble = Partial<Omit<Runtime.RuntimeService, "extensions">> & {
  readonly extensions?: Partial<Runtime.RuntimeService["extensions"]>
}
type TestHandle = Pick<Tensor.Any, "_tag" | "shape" | "dtype" | "storage" | "device" | "placement" | "pipe">

const runtimeDouble = (value: RuntimeDouble): Runtime.RuntimeService => {
  // SAFETY: Each test supplies every runtime member reached by the code under test.
  return value as Runtime.RuntimeService
}

const brandedHandle = (value: TestHandle): Tensor.Any => {
  // SAFETY: The handle factory supplies all public metadata; only Runtime's private brands are absent.
  return value as Tensor.Any
}

const loaderOnlyIdentity = (value: Tensor.Any): Tensor.Lazy => {
  // SAFETY: Loader metadata tests never invoke these placeholder model forwards.
  return value as Tensor.Lazy
}

const isLazyHandle = (value: Tensor.Any): value is Tensor.Lazy => value._tag === "LazyTensor"

const broadcast = (left: ReadonlyArray<number>, right: ReadonlyArray<number>): Array<number> => {
  const rank = Math.max(left.length, right.length)
  const shape: Array<number> = []
  for (let index = 0; index < rank; index++) {
    shape.unshift(Math.max(left[left.length - 1 - index] ?? 1, right[right.length - 1 - index] ?? 1))
  }
  return shape
}

function handle(
  tag: "LazyTensor",
  shape: ReadonlyArray<number>,
  dtype: Tensor.DType,
  storage?: Runtime.EncodedTensorStorage
): Tensor.Lazy
function handle(
  tag: "Tensor",
  shape: ReadonlyArray<number>,
  dtype: Tensor.DType,
  storage?: Runtime.EncodedTensorStorage
): Tensor.Concrete
function handle(
  tag: "LazyTensor" | "Tensor",
  shape: ReadonlyArray<number>,
  dtype: Tensor.DType,
  storage?: Runtime.EncodedTensorStorage
): Tensor.Any {
  const value = {
    _tag: tag,
    shape,
    dtype,
    device: placement.deviceType,
    placement,
    pipe() {
      throw new Error("unused test handle pipe")
    }
  } satisfies TestHandle
  if (storage !== undefined) Object.assign(value, { storage })
  return brandedHandle(Object.freeze(value))
}

// Topology tests clear this module-local log immediately before building a graph.
const requests: Array<Runtime.NodeRequest> = []

const runtime = runtimeDouble({
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
          return handle("LazyTensor", [], request.attributes.dtype)
        case "quantizedEmbedding":
          return handle(
            "LazyTensor",
            [...request.inputs[0].shape, request.attributes.logicalShape[1]],
            "f32"
          )
        case "quantizedLinear":
          return handle(
            "LazyTensor",
            [...request.inputs[0].shape.slice(0, -1), request.attributes.logicalShape[0]],
            "f32"
          )
        case "rmsNorm":
          return handle("LazyTensor", request.inputs[0].shape, request.inputs[0].dtype)
        case "mean": {
          const dims = new Set(request.attributes.dims)
          const shape = request.inputs[0].shape.flatMap((dimension, index) =>
            dims.has(index) ? request.attributes.keepdims ? [1] : [] : [dimension]
          )
          return handle("LazyTensor", shape, request.inputs[0].dtype)
        }
        case "reshape":
          return handle("LazyTensor", request.attributes.shape, request.inputs[0].dtype)
        case "permute":
          return handle(
            "LazyTensor",
            request.attributes.dims.map((dimension) => request.inputs[0].shape[dimension]),
            request.inputs[0].dtype
          )
        case "scaledDotProductAttention":
          return handle(
            "LazyTensor",
            [...request.inputs[0].shape.slice(0, -1), request.inputs[2].shape.at(-1)!],
            request.inputs[0].dtype
          )
        case "add":
        case "div":
        case "mul":
          return handle(
            "LazyTensor",
            broadcast(request.inputs[0].shape, request.inputs[1].shape),
            request.inputs[0].shape.length === 0 ? request.inputs[1].dtype : request.inputs[0].dtype
          )
        case "pow":
        case "rotaryEmbedding":
        case "tanh":
          return handle("LazyTensor", request.inputs[0].shape, request.inputs[0].dtype)
        case "expose": {
          const input = request.inputs[0]
          if (!isLazyHandle(input)) throw new Error("expose input must be lazy")
          return input
        }
        default:
          throw new Error(`unexpected Muse-Glimmer graph operation ${request.op}`)
      }
    })
})

const runtimeLayer = Layer.succeed(Runtime.Runtime, runtime)

// Rank-two parameters carry encoded storage so the graph takes quantized
// embedding/linear paths without allocating the canonical model's huge weights.
const modelParams = (parameters: ReadonlyArray<{ readonly shape: ReadonlyArray<number> }>): Array<Tensor.Concrete> =>
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

const descriptors = (parameterSpecs: ReadonlyArray<Model.ParameterSpec>) =>
  parameterSpecs.map(({ name, shape }) => ({ name, shape }))

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

it.effect("exports the exact Muse-Glimmer GGUF definition and loader", () =>
  Effect.sync(() => {
    expect(Object.keys(MuseGlimmer).sort()).toEqual(["architecture", "create", "definition", "loadGGUF"])
    expect(MuseGlimmer.architecture).toBe("muse-glimmer")
    expect(MuseGlimmer.definition).toEqual({
      architecture: "muse-glimmer",
      create: MuseGlimmer.create
    })
  }))

it.effect("loadGGUF rejects artifacts for another architecture", () => {
  const ggufRuntime = runtimeDouble({
    ...runtime,
    extensions: {
      gguf: {
        inspect: () =>
          Effect.succeed({
            metadata: [{ key: "general.architecture", value: "other-model" }],
            tensors: []
          }),
        load: () => Effect.die(new Error("load must not be called"))
      }
    }
  })
  return Effect.gen(function*() {
    const error = yield* Effect.flip(MuseGlimmer.loadGGUF("other.gguf"))
    expect(error._tag).toBe("GgufError")
    if (error._tag !== "GgufError") throw error
    expect(error.op).toBe("validate")
    expect(error.message).toContain("\"muse-glimmer\"")
  }).pipe(Effect.provide(Layer.succeed(Runtime.Runtime, ggufRuntime)))
})

it.effect("validates canonical configuration with Schema", () =>
  Effect.gen(function*() {
    yield* MuseGlimmer.create(config())
    for (const [key] of configEntries) {
      const missing = new Map(config())
      missing.delete(key)
      const missingError = yield* Effect.flip(MuseGlimmer.create(missing))
      expect(missingError._tag).toBe("ModelError")
      expect(missingError.message).toContain(key)

      const invalid = new Map(config())
      invalid.set(key, 0)
      const invalidError = yield* Effect.flip(MuseGlimmer.create(invalid))
      expect(invalidError._tag).toBe("ModelError")
      expect(invalidError.message).toContain(key)
    }

    const invalidGqa = new Map(config())
    invalidGqa.set("attention.head_count", 31)
    expect((yield* Effect.flip(MuseGlimmer.create(invalidGqa))).message).toContain(
      "attention.head_count"
    )

    const oversizedBlocks = new Map(config())
    oversizedBlocks.set("block_count", 1025)
    expect((yield* Effect.flip(MuseGlimmer.create(oversizedBlocks))).message).toContain("block_count")

    const invalidRope = new Map(config())
    invalidRope.set("attention.key_length", 127)
    expect((yield* Effect.flip(MuseGlimmer.create(invalidRope))).message).toContain(
      "attention.key_length"
    )

    const invalidValueWidth = new Map(config())
    invalidValueWidth.set("attention.value_length", 126)
    const invalidValueWidthError = yield* Effect.flip(MuseGlimmer.create(invalidValueWidth))
    expect(invalidValueWidthError.message).toContain("attention.value_length")
    expect(invalidValueWidthError.message).toContain("attention.key_length")

    const invalidWindow = new Map(config())
    invalidWindow.set("attention.sliding_window", 131073)
    expect((yield* Effect.flip(MuseGlimmer.create(invalidWindow))).message).toContain(
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
      ["attention.value_length", 4],
      ["attention.layer_norm_rms_epsilon", 1e-4],
      ["attention.sliding_window", 8],
      ["attention.sliding_window_pattern", 2],
      ["rope.freq_base", 10000],
      ["logit_scale", 0.25],
      ["final_logit_softcapping", 10],
      ["vocab_size", 12]
    ])
    const model = yield* MuseGlimmer.create(custom)
    expect(descriptors(model.parameterSpecs)).toEqual([
      { name: "token_embd.weight", shape: [12, 8] },
      { name: "output_norm.weight", shape: [8] },
      { name: "output.weight", shape: [12, 8] },
      { name: "blk.0.attn_norm.weight", shape: [8] },
      { name: "blk.0.post_attention_norm.weight", shape: [8] },
      { name: "blk.0.attn_q.weight", shape: [8, 8] },
      { name: "blk.0.attn_k.weight", shape: [4, 8] },
      { name: "blk.0.attn_v.weight", shape: [4, 8] },
      { name: "blk.0.attn_q_norm.weight", shape: [4] },
      { name: "blk.0.attn_k_norm.weight", shape: [4] },
      { name: "blk.0.attn_gate.weight", shape: [8, 8] },
      { name: "blk.0.attn_output.weight", shape: [8, 8] },
      { name: "blk.0.ffn_norm.weight", shape: [8] },
      { name: "blk.0.post_ffw_norm.weight", shape: [8] },
      { name: "blk.0.ffn_gate.weight", shape: [16, 8] },
      { name: "blk.0.ffn_up.weight", shape: [16, 8] },
      { name: "blk.0.ffn_down.weight", shape: [8, 16] }
    ])

    requests.length = 0
    const output = yield* model.forward(modelParams(model.parameterSpecs), handle("Tensor", [1, 2], "i64"))
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
      model.forward(modelParams(model.parameterSpecs), handle("Tensor", [1, 17], "i64"))
    )
    expect(contextError.message).toContain("exceeds context length 16")
  }).pipe(Effect.provide(runtimeLayer)))

it.effect("receives vocab_size from generic GGUF tokenizer token translation", () => {
  let vocabSize: unknown
  const ggufRuntime = runtimeDouble({
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
  })
  const services = Layer.succeed(Runtime.Runtime, ggufRuntime)
  return Effect.gen(function*() {
    yield* Gguf.loadModel("generic.gguf", {
      architecture: "generic-vocab-test",
      create: (canonical) => {
        vocabSize = canonical.get("vocab_size")
        return Model.define({
          parameterSpecs: [],
          forward: (_, input) => Effect.succeed(loaderOnlyIdentity(input))
        })
      }
    })
    expect(vocabSize).toBe(3)
  }).pipe(Effect.provide(services))
})

it.effect("exposes canonical tokenizer metadata from GGUF loading", () => {
  const ggufRuntime = runtimeDouble({
    ...runtime,
    extensions: {
      gguf: {
        inspect: () =>
          Effect.succeed({
            metadata: [
              { key: "general.architecture", value: "generic-metadata-test" },
              { key: "tokenizer.chat_template", value: "{{ messages }}" },
              { key: "tokenizer.ggml.bos_token_id", value: 1 },
              { key: "tokenizer.ggml.eos_token_id", value: 2 },
              { key: "tokenizer.ggml.eot_token_id", value: 3 }
            ],
            tensors: []
          }),
        load: () => Effect.succeed({ entries: [] })
      }
    }
  })
  const services = Layer.succeed(Runtime.Runtime, ggufRuntime)
  return Effect.gen(function*() {
    const loaded = yield* Gguf.loadModel("generic.gguf", {
      architecture: "generic-metadata-test",
      create: () =>
        Model.define({
          parameterSpecs: [],
          forward: (_, input) => Effect.succeed(loaderOnlyIdentity(input))
        })
    })
    expect(loaded.metadata.get("tokenizer.chat_template")).toBe("{{ messages }}")
    expect(loaded.metadata.get("tokenizer.ggml.bos_token_id")).toBe(1)
    expect(loaded.metadata.get("tokenizer.ggml.eos_token_id")).toBe(2)
    expect(loaded.metadata.get("tokenizer.ggml.eot_token_id")).toBe(3)
  }).pipe(Effect.provide(services))
})

it.effect("defines the exact parameter catalog", () =>
  Effect.gen(function*() {
    const model = yield* MuseGlimmer.create(config())
    expect(model.parameterSpecs).toHaveLength(731)
    expect(descriptors(model.parameterSpecs.slice(0, 3))).toEqual([
      { name: "token_embd.weight", shape: [202048, 6656] },
      { name: "output_norm.weight", shape: [6656] },
      { name: "output.weight", shape: [202048, 6656] }
    ])
    expect(descriptors(model.parameterSpecs.slice(3, 17))).toEqual(expectedLayer(0))
    expect(descriptors(model.parameterSpecs.slice(3 + 27 * 14, 3 + 28 * 14))).toEqual(expectedLayer(27))
    expect(descriptors(model.parameterSpecs.slice(-14))).toEqual(expectedLayer(51))
    expect(model.parameterSpecs.every(({ initializer }) => initializer !== undefined)).toBe(true)
  }).pipe(Effect.provide(runtimeLayer)))

it.effect("reports parameter arity and token rank through ModelError", () =>
  Effect.gen(function*() {
    const model = yield* MuseGlimmer.create(config())
    const arityError = yield* Effect.flip(model.forward([], handle("Tensor", [1, 1], "i64")))
    expect(arityError._tag).toBe("ModelError")
    expect(arityError.message).toContain("expected 731 parameters")

    const rankError = yield* Effect.flip(model.forward(modelParams(model.parameterSpecs), handle("Tensor", [1], "i64")))
    expect(rankError._tag).toBe("ModelError")
    expect(rankError.message).toContain("[B, S]")
  }).pipe(Effect.provide(runtimeLayer)))

it.effect("builds the canonical graph with 39 local and 13 global layers", () =>
  Effect.gen(function*() {
    requests.length = 0
    const model = yield* MuseGlimmer.create(config())
    const output = yield* model.forward(modelParams(model.parameterSpecs), handle("Tensor", [1, 2], "i64"))
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

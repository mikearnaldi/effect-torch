/** Load-only DFlash proposer artifacts for the official Muse-Glimmer checkpoint. */
import { Effect } from "effect"
import * as Gguf from "../Gguf.ts"
import * as Model from "../Model.ts"
import type * as Runtime from "../Runtime.ts"
import * as Tensor from "../Tensor.ts"

/** Exact GGUF architecture value accepted by this loader. */
export const architecture = "dflash"

/** GGUF layer-input IDs stored by the checkpoint. */
export const targetLayers = [2, 14, 26, 38, 50] as const

/** Zero-based Muse residual taps corresponding to {@link targetLayers}. */
export const targetResidualTaps = [1, 13, 25, 37, 49] as const

export const blockSize = 16
export const maxDraftTokens = blockSize - 1
export const hiddenSize = 6656
export const feedForwardSize = 19968
export const vocabularySize = 202048

/** One native stateful program required by the DFlash component. */
export type Program =
  | {
    readonly kind: "FeatureFusionKvInjection"
    readonly targetResidualTaps: typeof targetResidualTaps
    readonly fusedWidth: number
    readonly hiddenSize: number
    readonly kvLayers: number
    readonly kvHeads: number
    readonly headDim: number
  }
  | {
    readonly kind: "MaskedNonCausalBlockDecode"
    readonly blockSize: number
    readonly maskToken: number
    readonly hiddenSize: number
    readonly feedForwardSize: number
    readonly queryHeads: number
    readonly kvHeads: number
    readonly headDim: number
    readonly slidingWindow: number
    readonly ropeBase: number
    readonly rmsEpsilon: number
    readonly sharedTokenEmbedding: "token_embd.weight"
    readonly sharedLmHead: "output.weight"
  }

/** DFlash proposer component retained by the generic proposer artifact. */
export interface Component extends Model.ProposerGraphComponent {
  readonly programs: readonly [Program, Program]
}

/** Loaded parameters, canonical metadata, and target-coupled proposer artifact. */
export interface Loaded {
  readonly params: ReadonlyArray<Tensor.Concrete>
  readonly metadata: ReadonlyMap<string, unknown>
  readonly parameters: ReadonlyArray<Model.ParameterSpec>
  readonly component: Component
  readonly artifact: Model.ProposerArtifact
}

/** The two native programs required by the checkpoint. */
export const programs: readonly [Program, Program] = [
  {
    kind: "FeatureFusionKvInjection",
    targetResidualTaps,
    fusedWidth: targetLayers.length * hiddenSize,
    hiddenSize,
    kvLayers: 5,
    kvHeads: 8,
    headDim: 128
  },
  {
    kind: "MaskedNonCausalBlockDecode",
    blockSize,
    maskToken: 201818,
    hiddenSize,
    feedForwardSize,
    queryHeads: 32,
    kvHeads: 8,
    headDim: 128,
    slidingWindow: 2048,
    ropeBase: 500000,
    rmsEpsilon: 1e-5,
    sharedTokenEmbedding: "token_embd.weight",
    sharedLmHead: "output.weight"
  }
]

/** Complete generic proposer contract attached by {@link load}. */
export const plan: Model.ProposerPlan = {
  target: {
    vocabulary: vocabularySize,
    hiddenTaps: targetResidualTaps.map((layer) => ({ layer, dtype: "f32", shape: ["Rows", hiddenSize] })),
    sharedWeights: [
      { kind: "TokenEmbedding", name: "token_embd.weight", dtype: "f32", shape: [vocabularySize, hiddenSize] },
      { kind: "LmHead", name: "output.weight", dtype: "f32", shape: [vocabularySize, hiddenSize] }
    ]
  },
  stages: [{
    operation: { _tag: "ParallelBlock", component: 0, layout: { id: "dflash-block-16-v1" } },
    inputs: [
      ...targetResidualTaps.map((layer, slot) => ({ slot, value: { _tag: "TargetHidden" as const, layer } })),
      { slot: 5, value: { _tag: "PendingTokens" } },
      { slot: 6, value: { _tag: "SharedTokenEmbedding" } },
      { slot: 7, value: { _tag: "SharedLmHead" } }
    ],
    outputs: [{ dtype: "u32", shape: ["Rows"] }]
  }],
  state: {
    _tag: "Kv",
    schema: { id: "dflash-kv-5x8x128-swa2048-v1" },
    commit: { _tag: "Replay", stages: [0] }
  },
  output: {
    topology: "Chains",
    probabilities: "Unavailable",
    tokenIds: { _tag: "StageOutput", stage: 0, output: 0 }
  },
  tokenMap: { _tag: "Identity" },
  trainedMaxRows: maxDraftTokens
}

const field = (metadata: ReadonlyMap<string, unknown>, key: string): unknown => metadata.get(key)

const exact = (
  metadata: ReadonlyMap<string, unknown>,
  key: string,
  expected: number
): Effect.Effect<void, Model.ModelError> => {
  const actual = field(metadata, key)
  return actual === expected
    ? Effect.void
    : new Model.ModelError({
      op: "create",
      message: `DFlash ${key} must be ${expected}, got ${JSON.stringify(actual)}`
    })
}

const exactArray = (
  metadata: ReadonlyMap<string, unknown>,
  key: string,
  expected: ReadonlyArray<number | boolean>
): Effect.Effect<void, Model.ModelError> => {
  const actual = field(metadata, key)
  const matches = Array.isArray(actual) && actual.length === expected.length &&
    actual.every((value, index) => value === expected[index])
  return matches
    ? Effect.void
    : new Model.ModelError({
      op: "create",
      message: `DFlash ${key} must be [${expected}], got ${JSON.stringify(actual)}`
    })
}

const makeParameters = (
  metadata: ReadonlyMap<string, unknown>,
  tensors: ReadonlyArray<Runtime.GgufTensorDescriptor>
): Effect.Effect<ReadonlyArray<Model.ParameterSpec>, Model.ModelError> =>
  Effect.gen(function*() {
    if (field(metadata, "architecture") !== architecture) {
      return yield* new Model.ModelError({
        op: "create",
        message: `DFlash architecture must be exactly ${JSON.stringify(architecture)}`
      })
    }
    yield* exact(metadata, "block_count", 5)
    yield* exact(metadata, "context_length", 131072)
    yield* exact(metadata, "embedding_length", hiddenSize)
    yield* exact(metadata, "feed_forward_length", feedForwardSize)
    yield* exact(metadata, "attention.head_count", 32)
    yield* exact(metadata, "attention.head_count_kv", 8)
    yield* exact(metadata, "attention.key_length", 128)
    yield* exact(metadata, "attention.value_length", 128)
    yield* exact(metadata, "attention.layer_norm_rms_epsilon", Math.fround(1e-5))
    yield* exact(metadata, "attention.sliding_window", 2048)
    yield* exact(metadata, "rope.freq_base", 500000)
    yield* exact(metadata, "block_size", blockSize)
    yield* exact(metadata, "vocab_size", vocabularySize)
    yield* exact(metadata, "tokenizer.ggml.mask_token_id", 201818)
    yield* exactArray(metadata, "target_layers", targetLayers)
    yield* exactArray(metadata, "attention.sliding_window_pattern", [true, true, true, true, true])

    const parameters: Array<Model.ParameterSpec> = [
      { name: "fc.weight", shape: [hiddenSize, targetLayers.length * hiddenSize] },
      { name: "enc.output_norm.weight", shape: [hiddenSize] }
    ]
    for (let layer = 0; layer < 5; layer++) {
      const prefix = `blk.${layer}`
      parameters.push(
        { name: `${prefix}.attn_norm.weight`, shape: [hiddenSize] },
        { name: `${prefix}.ffn_down.weight`, shape: [hiddenSize, feedForwardSize] },
        { name: `${prefix}.ffn_gate.weight`, shape: [feedForwardSize, hiddenSize] },
        { name: `${prefix}.ffn_up.weight`, shape: [feedForwardSize, hiddenSize] },
        { name: `${prefix}.ffn_norm.weight`, shape: [hiddenSize] },
        { name: `${prefix}.attn_k_norm.weight`, shape: [128] },
        { name: `${prefix}.attn_k.weight`, shape: [8 * 128, hiddenSize] },
        { name: `${prefix}.attn_output.weight`, shape: [hiddenSize, 32 * 128] },
        { name: `${prefix}.attn_q_norm.weight`, shape: [128] },
        { name: `${prefix}.attn_q.weight`, shape: [32 * 128, hiddenSize] },
        { name: `${prefix}.attn_v.weight`, shape: [8 * 128, hiddenSize] }
      )
    }
    if (tensors.some((tensor) => tensor.name === "output_norm.weight")) {
      parameters.push({ name: "output_norm.weight", shape: [hiddenSize] })
    }
    return parameters
  })

/** Registry-free GGUF definition used by {@link load} and focused catalog tests. */
export const definition: Gguf.ParameterArtifactDefinition = {
  architecture,
  parameters: makeParameters
}

const makeComponent = (params: ReadonlyArray<Tensor.Concrete>): Component => ({
  params,
  programs,
  build: () =>
    new Model.ModelError({
      op: "dflashCompile",
      message:
        "DFlash requires native feature-fusion/KV-injection and masked non-causal block-decode programs; the generic ParallelBlock tensor graph cannot represent this stateful contract"
    })
})

const makeArtifact = (
  component: Component
): Effect.Effect<Model.ProposerArtifact, Model.InferenceError | Model.ModelError> =>
  Model.Speculation.artifact({
    components: [component],
    plan
  })

/** Loads the official Muse-Glimmer DFlash checkpoint as a target-coupled proposer. */
export const load = (
  path: string
): Effect.Effect<Loaded, Gguf.GgufError | Model.ModelError | Model.InferenceError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const loaded = yield* Gguf.loadParameters(path, definition)
    const component = makeComponent(loaded.params)
    const artifact = yield* makeArtifact(component).pipe(
      Effect.onError(() => Effect.ignore(Tensor.clearAll(loaded.params)))
    )
    return { ...loaded, component, artifact }
  })

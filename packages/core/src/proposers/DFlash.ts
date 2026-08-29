/**
 * DFlash fixed-width parallel proposer graph and GGUF loader.
 *
 * The loader validates a target-coupled checkpoint, while {@link artifact}
 * describes the target residual taps and shared weights needed by native
 * speculative inference.
 *
 * @since 0.1.0
 */
import { Effect } from "effect"
import * as Schema from "effect/Schema"
import * as Gguf from "../Gguf.ts"
import * as Model from "../Model.ts"
import type * as Runtime from "../Runtime.ts"
import * as Speculation from "../Speculation.ts"
import * as Tensor from "../Tensor.ts"

/**
 * Exact `general.architecture` value accepted by {@link loadGGUF}.
 *
 * @since 0.1.0
 * @category identifiers
 */
export const architecture = "dflash"

/**
 * Validated canonical DFlash GGUF configuration.
 *
 * @since 0.1.0
 * @category models
 */
export interface Configuration {
  /** Number of proposer transformer blocks. */
  readonly blockCount: number
  /** Maximum sequence length declared by the artifact. */
  readonly contextLength: number
  /** Hidden-state width. */
  readonly embeddingLength: number
  /** Feed-forward intermediate width. */
  readonly feedForwardLength: number
  /** Number of query attention heads. */
  readonly queryHeads: number
  /** Number of key/value attention heads. */
  readonly kvHeads: number
  /** Width of each key head. */
  readonly keyLength: number
  /** Width of each value head. */
  readonly valueLength: number
  /** Epsilon used by RMS normalization. */
  readonly rmsEpsilon: number
  /** Causal attention window used by every proposer block. */
  readonly slidingWindow: number
  /** Base frequency used by rotary position embeddings. */
  readonly ropeBase: number
  /** Proposal block width including its anchor token. */
  readonly blockSize: number
  /** Number of target token ids. */
  readonly vocabularySize: number
  /** Vocabulary id inserted in unresolved proposal rows. */
  readonly maskToken: number
  /** One-based target layers named by the artifact. */
  readonly targetLayers: ReadonlyArray<number>
  /** Zero-based target residual layers consumed by replay; each is routed from the target's {@link Model.hiddenExposure} exposure. */
  readonly targetResidualTaps: ReadonlyArray<number>
  /** Per-proposer-layer sliding-window flags; validation requires all true. */
  readonly slidingWindowPattern: ReadonlyArray<boolean>
}

/**
 * Loaded parameters, canonical metadata, and target-coupled proposer artifact.
 *
 * @since 0.1.0
 * @category models
 */
export interface Loaded {
  /** Caller-owned proposer tensors in `parameterSpecs` order. */
  readonly params: ReadonlyArray<Tensor.Concrete>
  /** Canonical metadata used to validate the configuration. */
  readonly metadata: ReadonlyMap<string, unknown>
  /** Exact validated GGUF tensor catalog. */
  readonly parameterSpecs: ReadonlyArray<Gguf.TensorSpec>
  /** Decoded DFlash architecture configuration. */
  readonly config: Configuration
  /** Maximum candidates emitted per proposal block, equal to `blockSize - 1`. */
  readonly maxDraftTokens: number
  /** Parallel-block artifact ready for model inference compilation. */
  readonly artifact: Speculation.ParallelBlock
}

const PositiveInt = Schema.Int.check(Schema.isGreaterThan(0))
const NonNegativeInt = Schema.Int.check(Schema.isGreaterThanOrEqualTo(0))
const PositiveFinite = Schema.Finite.check(Schema.isGreaterThan(0))

const Config = Schema.Struct({
  architecture: Schema.Literal(architecture),
  block_count: PositiveInt,
  context_length: PositiveInt,
  embedding_length: PositiveInt,
  feed_forward_length: PositiveInt,
  "attention.head_count": PositiveInt,
  "attention.head_count_kv": PositiveInt,
  "attention.key_length": PositiveInt,
  "attention.value_length": PositiveInt,
  "attention.layer_norm_rms_epsilon": PositiveFinite,
  "attention.sliding_window": PositiveInt,
  "attention.sliding_window_pattern": Schema.Array(Schema.Boolean),
  "rope.freq_base": PositiveFinite,
  block_size: PositiveInt,
  vocab_size: PositiveInt,
  "tokenizer.ggml.mask_token_id": NonNegativeInt,
  target_layers: Schema.Array(PositiveInt)
}).check(
  Schema.makeFilter((config) => {
    const issues: Array<Schema.FilterIssue> = []
    if (config["attention.head_count"] % config["attention.head_count_kv"] !== 0) {
      issues.push({
        path: ["attention.head_count"],
        issue: "attention.head_count must be divisible by attention.head_count_kv"
      })
    }
    if (config["attention.key_length"] !== config["attention.value_length"]) {
      issues.push({
        path: ["attention.value_length"],
        issue: "attention.value_length must equal attention.key_length"
      })
    }
    if (config["attention.key_length"] % 2 !== 0) {
      issues.push({ path: ["attention.key_length"], issue: "attention.key_length must be even" })
    }
    if (config["attention.sliding_window"] > config.context_length) {
      issues.push({
        path: ["attention.sliding_window"],
        issue: "attention.sliding_window must not exceed context_length"
      })
    }
    if (config.block_size < 2 || config.block_size > config.context_length) {
      issues.push({ path: ["block_size"], issue: "block_size must be in 2..=context_length" })
    }
    if (config["tokenizer.ggml.mask_token_id"] >= config.vocab_size) {
      issues.push({
        path: ["tokenizer.ggml.mask_token_id"],
        issue: "tokenizer.ggml.mask_token_id must be less than vocab_size"
      })
    }
    if (config.target_layers.length < 2) {
      issues.push({ path: ["target_layers"], issue: "target_layers must contain at least two layers" })
    } else if (new Set(config.target_layers).size !== config.target_layers.length) {
      issues.push({ path: ["target_layers"], issue: "target_layers must not contain duplicates" })
    }
    const pattern = config["attention.sliding_window_pattern"]
    if (pattern.length !== config.block_count) {
      issues.push({
        path: ["attention.sliding_window_pattern"],
        issue: "attention.sliding_window_pattern length must equal block_count"
      })
    } else if (pattern.some((sliding) => !sliding)) {
      issues.push({
        path: ["attention.sliding_window_pattern"],
        issue: "all draft layers must use sliding-window attention"
      })
    }
    return issues
  })
)

type Config = Schema.Schema.Type<typeof Config>

/**
 * Decodes canonical GGUF metadata and validates cross-field DFlash invariants.
 * The metadata must already use the normalization performed by
 * {@link Gguf.loadParameters}.
 *
 * @since 0.1.0
 * @category constructors
 */
export const configuration = (
  metadata: ReadonlyMap<string, unknown>
): Effect.Effect<Configuration, Model.ModelError> =>
  Schema.decodeUnknownEffect(Config)(Object.fromEntries(metadata)).pipe(
    Effect.map((config: Config) => ({
      blockCount: config.block_count,
      contextLength: config.context_length,
      embeddingLength: config.embedding_length,
      feedForwardLength: config.feed_forward_length,
      queryHeads: config["attention.head_count"],
      kvHeads: config["attention.head_count_kv"],
      keyLength: config["attention.key_length"],
      valueLength: config["attention.value_length"],
      rmsEpsilon: config["attention.layer_norm_rms_epsilon"],
      slidingWindow: config["attention.sliding_window"],
      ropeBase: config["rope.freq_base"],
      blockSize: config.block_size,
      vocabularySize: config.vocab_size,
      maskToken: config["tokenizer.ggml.mask_token_id"],
      targetLayers: config.target_layers,
      targetResidualTaps: config.target_layers.map((layer) => layer - 1),
      slidingWindowPattern: config["attention.sliding_window_pattern"]
    })),
    Effect.mapError((error) => new Model.ModelError({ op: "create", message: `DFlash config: ${error.message}` }))
  )

const makeParameters = (
  metadata: ReadonlyMap<string, unknown>
): Effect.Effect<ReadonlyArray<Gguf.TensorSpec>, Model.ModelError> =>
  Effect.gen(function*() {
    const config = yield* configuration(metadata)

    const parameterSpecs: Array<Gguf.TensorSpec> = [
      {
        name: "fc.weight",
        shape: [config.embeddingLength, config.targetLayers.length * config.embeddingLength]
      },
      { name: "enc.output_norm.weight", shape: [config.embeddingLength] }
    ]
    for (let layer = 0; layer < config.blockCount; layer++) {
      const prefix = `blk.${layer}`
      parameterSpecs.push(
        { name: `${prefix}.attn_norm.weight`, shape: [config.embeddingLength] },
        { name: `${prefix}.ffn_down.weight`, shape: [config.embeddingLength, config.feedForwardLength] },
        { name: `${prefix}.ffn_gate.weight`, shape: [config.feedForwardLength, config.embeddingLength] },
        { name: `${prefix}.ffn_up.weight`, shape: [config.feedForwardLength, config.embeddingLength] },
        { name: `${prefix}.ffn_norm.weight`, shape: [config.embeddingLength] },
        { name: `${prefix}.attn_k_norm.weight`, shape: [config.keyLength] },
        { name: `${prefix}.attn_k.weight`, shape: [config.kvHeads * config.keyLength, config.embeddingLength] },
        {
          name: `${prefix}.attn_output.weight`,
          shape: [config.embeddingLength, config.queryHeads * config.valueLength]
        },
        { name: `${prefix}.attn_q_norm.weight`, shape: [config.keyLength] },
        { name: `${prefix}.attn_q.weight`, shape: [config.queryHeads * config.keyLength, config.embeddingLength] },
        { name: `${prefix}.attn_v.weight`, shape: [config.kvHeads * config.valueLength, config.embeddingLength] }
      )
    }
    parameterSpecs.push({ name: "output_norm.weight", shape: [config.embeddingLength] })
    return parameterSpecs
  })

/**
 * Explicit GGUF parameter-artifact definition used by {@link loadGGUF}.
 *
 * @since 0.1.0
 * @category models
 */
export const definition: Gguf.ParameterArtifactDefinition = {
  architecture,
  parameterSpecs: makeParameters
}

const graphError = (op: string, message: string) => new Model.ModelError({ op, message })

const buildGraph = (config: Configuration) => {
  const parameterCount = 2 + config.blockCount * 11
  const checkParams = (op: string, params: Model.Params) =>
    params.length === parameterCount + 1
      ? Effect.void
      : graphError(op, `expected ${parameterCount + 1} parameters, got ${params.length}`)
  const layerOffset = (layer: number) => 2 + layer * 11
  const headsFirst = (
    value: Tensor.Any,
    heads: number,
    rows: number,
    batch?: number
  ) =>
    batch === undefined
      ? Effect.gen(function*() {
        return yield* Tensor.transpose(
          yield* Tensor.reshape(value, [rows, heads, config.keyLength]),
          [1, 0, 2]
        )
      })
      : Effect.gen(function*() {
        return yield* Tensor.transpose(
          yield* Tensor.reshape(value, [batch, rows, heads, config.keyLength]),
          [0, 2, 1, 3]
        )
      })
  const fuse = (params: Model.Params, targetRows: ReadonlyArray<Tensor.Any>) =>
    Effect.gen(function*() {
      yield* checkParams("dflashReplay", params)
      if (targetRows.length !== config.targetResidualTaps.length || targetRows.length < 2) {
        return yield* graphError(
          "dflashReplay",
          `expected ${config.targetResidualTaps.length} target taps, got ${targetRows.length}`
        )
      }
      const [first, second, ...rest] = targetRows
      const features = yield* Tensor.concat(
        [first!, second!, ...rest],
        { dim: -1 }
      )
      return yield* Tensor.rmsNorm(
        yield* Tensor.linearRows(features, params[0]!),
        params[1],
        config.rmsEpsilon
      )
    })

  const replay = (params: Model.Params, targetRows: ReadonlyArray<Tensor.Any>) =>
    Effect.gen(function*() {
      const fused = yield* fuse(params, targetRows)
      if (fused.shape.length !== 2 && fused.shape.length !== 3) {
        return yield* graphError("dflashReplay", `expected [Rows, E] or [Batch, Rows, E], got [${fused.shape}]`)
      }
      const batch = fused.shape.length === 3 ? fused.shape[0] : undefined
      const rows = fused.shape.length === 3 ? fused.shape[1]! : fused.shape[0]!
      const layers: Array<Speculation.KeyValue> = []
      for (let layer = 0; layer < config.blockCount; layer++) {
        const offset = layerOffset(layer)
        let key = yield* headsFirst(
          yield* Tensor.linearRows(fused, params[offset + 6]!),
          config.kvHeads,
          rows,
          batch
        )
        const value = yield* headsFirst(
          yield* Tensor.linearRows(fused, params[offset + 10]!),
          config.kvHeads,
          rows,
          batch
        )
        key = yield* Tensor.rmsNorm(key, params[offset + 5], config.rmsEpsilon)
        key = yield* Tensor.rotaryEmbedding(key, rows, config.ropeBase, { layout: "HalfSplit" })
        layers.push({ key, value })
      }
      return layers
    })

  const buildOutput = (
    params: Model.Params,
    anchorTokens: Tensor.Any,
    sharedTokenEmbedding: Tensor.Any,
    sharedLmHead: Tensor.Any,
    maxDraftTokens: number
  ) =>
    Effect.gen(function*() {
      yield* checkParams("dflashBlock", params)
      if (anchorTokens.shape.length !== 1) {
        return yield* graphError("dflashBlock", `expected anchor token rows [Batch], got [${anchorTokens.shape}]`)
      }
      const batch = anchorTokens.shape[0]!
      const blockRows = maxDraftTokens + 1
      if (maxDraftTokens < 1 || blockRows > config.blockSize) {
        return yield* graphError(
          "dflashBlock",
          `maxDraftTokens must be in [1, ${config.blockSize - 1}], got ${maxDraftTokens}`
        )
      }
      const anchor = yield* Tensor.reshape(anchorTokens, [batch, 1])
      const masks = yield* Tensor.full([batch, maxDraftTokens], config.maskToken, {
        dtype: anchorTokens.dtype
      })
      const tokens = yield* Tensor.concat([anchor, masks], { dim: 1 })
      let hidden: Tensor.Any = yield* Tensor.embedding(tokens, { weight: sharedTokenEmbedding })

      for (let layer = 0; layer < config.blockCount; layer++) {
        const offset = layerOffset(layer)
        const attentionInput = yield* Tensor.rmsNorm(hidden, params[offset], config.rmsEpsilon)
        let query = yield* headsFirst(
          yield* Tensor.linearRows(attentionInput, params[offset + 9]!),
          config.queryHeads,
          blockRows,
          batch
        )
        let key = yield* headsFirst(
          yield* Tensor.linearRows(attentionInput, params[offset + 6]!),
          config.kvHeads,
          blockRows,
          batch
        )
        const value = yield* headsFirst(
          yield* Tensor.linearRows(attentionInput, params[offset + 10]!),
          config.kvHeads,
          blockRows,
          batch
        )
        query = yield* Tensor.rmsNorm(query, params[offset + 8], config.rmsEpsilon)
        key = yield* Tensor.rmsNorm(key, params[offset + 5], config.rmsEpsilon)
        query = yield* Tensor.rotaryEmbedding(query, blockRows, config.ropeBase, {
          layout: "HalfSplit"
        })
        key = yield* Tensor.rotaryEmbedding(key, blockRows, config.ropeBase, {
          layout: "HalfSplit"
        })
        let attention = yield* Tensor.scaledDotProductAttention(query, key, value, {
          causal: false,
          scale: 1 / Math.sqrt(config.keyLength)
        })
        attention = yield* Tensor.linearRows(
          yield* Tensor.reshape(
            yield* Tensor.transpose(attention, [0, 2, 1, 3]),
            [batch, blockRows, config.queryHeads * config.valueLength]
          ),
          params[offset + 7]!
        )
        hidden = yield* Tensor.add(hidden, attention)

        const ffnInput = yield* Tensor.rmsNorm(hidden, params[offset + 4], config.rmsEpsilon)
        const ffn = yield* Tensor.linearRows(
          yield* Tensor.mul(
            yield* Tensor.silu(yield* Tensor.linearRows(ffnInput, params[offset + 2]!)),
            yield* Tensor.linearRows(ffnInput, params[offset + 3]!)
          ),
          params[offset + 1]!
        )
        hidden = yield* Tensor.add(hidden, ffn)
      }

      hidden = yield* Tensor.rmsNorm(hidden, params[parameterCount], config.rmsEpsilon)
      const logits = yield* Tensor.linearRows(hidden, sharedLmHead)
      const candidateLogits = yield* Tensor.slice(logits, {
        start: [0, 1, 0],
        end: [batch, blockRows, config.vocabularySize]
      })
      const tokenIds = yield* Tensor.cast(yield* Tensor.argmax(candidateLogits, -1), "u32")
      const probabilityRows = yield* Tensor.softmax(candidateLogits, { dims: [-1] })
      return { tokenIds, probabilityRows }
    })

  const build = (
    params: Model.Params,
    anchorTokens: Tensor.Any,
    sharedTokenEmbedding: Tensor.Any,
    sharedLmHead: Tensor.Any,
    maxDraftTokens: number
  ): Effect.Effect<Tensor.Lazy, Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
    Effect.map(
      buildOutput(params, anchorTokens, sharedTokenEmbedding, sharedLmHead, maxDraftTokens),
      ({ tokenIds }) => tokenIds
    )

  return { build, buildWithProbabilities: buildOutput, replay }
}

/**
 * Constructs the replayable DFlash parallel-block artifact from validated
 * configuration and caller-owned parameters. Construction performs no tracing,
 * compilation, or ownership transfer.
 *
 * @since 0.1.0
 * @category constructors
 */
export const artifact = (
  config: Configuration,
  params: ReadonlyArray<Tensor.Concrete>
): Speculation.ParallelBlock => {
  const graph = buildGraph(config)
  return Speculation.parallelBlock({
    params,
    vocabulary: config.vocabularySize,
    maxDraftTokens: config.blockSize - 1,
    hiddenTaps: config.targetResidualTaps.map((layer) => ({
      name: Model.hiddenExposure(layer),
      dtype: "f32",
      shape: ["Rows", config.embeddingLength]
    })),
    tokenEmbedding: {
      name: "token_embd.weight",
      dtype: "f32",
      shape: [config.vocabularySize, config.embeddingLength]
    },
    lmHead: {
      name: "output.weight",
      dtype: "f32",
      shape: [config.vocabularySize, config.embeddingLength]
    },
    currentBlockAttention: "Bidirectional",
    attentionWindow: config.slidingWindow,
    build: graph.build,
    replay: graph.replay
  })
}

/**
 * Inspects, validates, and loads a DFlash GGUF checkpoint as a target-coupled
 * proposer. The returned parameter handles are caller-owned. If configuration
 * or artifact construction fails after loading, all loaded handles receive a
 * best-effort release attempt.
 *
 * @since 0.1.0
 * @category loading
 */
export const loadGGUF = (
  path: string
): Effect.Effect<Loaded, Gguf.GgufError | Model.ModelError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const loaded = yield* Gguf.loadParameters(path, definition)
    return yield* Effect.gen(function*() {
      const config = yield* configuration(loaded.metadata)
      const proposer = artifact(config, loaded.params)
      return {
        ...loaded,
        config,
        maxDraftTokens: config.blockSize - 1,
        artifact: proposer
      }
    }).pipe(Effect.onError(() => Tensor.clearAll(loaded.params)))
  })

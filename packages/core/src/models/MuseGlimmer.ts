/**
 * Muse-Glimmer model architecture definition.
 *
 * @since 0.1.0
 */
import { Effect } from "effect"
import * as Schema from "effect/Schema"
import * as Model from "../Model.ts"
import type * as Registry from "../Registry.ts"
import * as Tensor from "../Tensor.ts"

/**
 * The exact registry identifier for Muse-Glimmer models.
 *
 * @since 0.1.0
 * @category identifiers
 */
export const id = "gguf:muse-glimmer"

const PositiveInt = Schema.Int.check(Schema.isGreaterThan(0))
const PositiveFinite = Schema.Finite.check(Schema.isGreaterThan(0))

const Config = Schema.Struct({
  block_count: PositiveInt,
  embedding_length: PositiveInt,
  feed_forward_length: PositiveInt,
  context_length: PositiveInt,
  "attention.head_count": PositiveInt,
  "attention.head_count_kv": PositiveInt,
  "attention.key_length": PositiveInt,
  "attention.value_length": PositiveInt,
  "attention.layer_norm_rms_epsilon": PositiveFinite,
  "attention.sliding_window": PositiveInt,
  "attention.sliding_window_pattern": PositiveInt,
  "rope.freq_base": PositiveFinite,
  logit_scale: PositiveFinite,
  final_logit_softcapping: PositiveFinite,
  vocab_size: PositiveInt
}).check(
  Schema.makeFilter((config) => {
    const issues: Array<Schema.FilterIssue> = []
    if (config["attention.head_count"] % config["attention.head_count_kv"] !== 0) {
      issues.push({
        path: ["attention.head_count"],
        issue: "attention.head_count must be divisible by attention.head_count_kv"
      })
    }
    if (config["attention.key_length"] % 2 !== 0) {
      issues.push({
        path: ["attention.key_length"],
        issue: "attention.key_length must be even"
      })
    }
    if (config["attention.sliding_window"] > config.context_length) {
      issues.push({
        path: ["attention.sliding_window"],
        issue: "attention.sliding_window must not exceed context_length"
      })
    }
    return issues
  })
)

type Config = Schema.Schema.Type<typeof Config>

const decodeConfig = (config: Registry.ModelConfig): Effect.Effect<Config, Model.ModelError> =>
  Schema.decodeUnknownEffect(Config)(Object.fromEntries(config)).pipe(
    Effect.mapError((error) =>
      new Model.ModelError({
        op: "create",
        message: `MuseGlimmer config: ${error.message}`
      })
    )
  )

const makeParameters = (config: Config): ReadonlyArray<Model.ParameterSpec> => {
  const hiddenSize = config.embedding_length
  const feedForwardSize = config.feed_forward_length
  const querySize = config["attention.head_count"] * config["attention.key_length"]
  const keySize = config["attention.head_count_kv"] * config["attention.key_length"]
  const valueSize = config["attention.head_count_kv"] * config["attention.value_length"]
  const attentionSize = config["attention.head_count"] * config["attention.value_length"]
  const specs: Array<Model.ParameterSpec> = [
    { name: "token_embd.weight", shape: [config.vocab_size, hiddenSize] },
    { name: "output_norm.weight", shape: [hiddenSize] },
    { name: "output.weight", shape: [config.vocab_size, hiddenSize] }
  ]
  for (let layer = 0; layer < config.block_count; layer++) {
    const prefix = `blk.${layer}`
    specs.push(
      { name: `${prefix}.attn_norm.weight`, shape: [hiddenSize] },
      { name: `${prefix}.post_attention_norm.weight`, shape: [hiddenSize] },
      { name: `${prefix}.attn_q.weight`, shape: [querySize, hiddenSize] },
      { name: `${prefix}.attn_k.weight`, shape: [keySize, hiddenSize] },
      { name: `${prefix}.attn_v.weight`, shape: [valueSize, hiddenSize] },
      { name: `${prefix}.attn_q_norm.weight`, shape: [config["attention.key_length"]] },
      { name: `${prefix}.attn_k_norm.weight`, shape: [config["attention.key_length"]] },
      { name: `${prefix}.attn_gate.weight`, shape: [attentionSize, hiddenSize] },
      { name: `${prefix}.attn_output.weight`, shape: [hiddenSize, attentionSize] },
      { name: `${prefix}.ffn_norm.weight`, shape: [hiddenSize] },
      { name: `${prefix}.post_ffw_norm.weight`, shape: [hiddenSize] },
      { name: `${prefix}.ffn_gate.weight`, shape: [feedForwardSize, hiddenSize] },
      { name: `${prefix}.ffn_up.weight`, shape: [feedForwardSize, hiddenSize] },
      { name: `${prefix}.ffn_down.weight`, shape: [hiddenSize, feedForwardSize] }
    )
  }
  return specs
}

const rms = (
  input: Tensor.Any,
  epsilon: number,
  scale?: Tensor.Any
) => Tensor.rmsNorm(input, scale, epsilon)

const makeModel = (config: Config): Effect.Effect<Model.Model, Model.ModelError> => {
  const parameters = makeParameters(config)
  const queryHeads = config["attention.head_count"]
  const keyValueHeads = config["attention.head_count_kv"]
  const keyLength = config["attention.key_length"]
  const valueLength = config["attention.value_length"]
  const attentionSize = queryHeads * valueLength
  const rmsEpsilon = config["attention.layer_norm_rms_epsilon"]
  const slidingWindowPattern = config["attention.sliding_window_pattern"]
  return Model.define({
    parameters,
    forward: (params, input) =>
      Effect.gen(function*() {
        if (params.length !== parameters.length) {
          return yield* new Model.ModelError({
            op: "forward",
            message: `MuseGlimmer.forward: expected ${parameters.length} parameters, got ${params.length}`
          })
        }
        if (input.shape.length !== 2) {
          return yield* new Model.ModelError({
            op: "forward",
            message: `MuseGlimmer.forward: expected token IDs with shape [B, S], got [${input.shape}]`
          })
        }

        const [batch, sequence] = input.shape
        if (sequence > config.context_length) {
          return yield* new Model.ModelError({
            op: "forward",
            message: `MuseGlimmer.forward: sequence length ${sequence} exceeds context length ${config.context_length}`
          })
        }
        let hidden = yield* Tensor.embedding(input, { weight: params[0] })
        hidden = yield* rms(hidden, rmsEpsilon)
        const headsFirst = (value: Tensor.Any, heads: number, width: number) =>
          Effect.gen(function*() {
            if (sequence === 1) {
              return yield* Tensor.reshape(value, [batch, heads, sequence, width])
            }
            return yield* Tensor.transpose(
              yield* Tensor.reshape(value, [batch, sequence, heads, width]),
              [0, 2, 1, 3]
            )
          })

        for (let layer = 0; layer < config.block_count; layer++) {
          const offset = 3 + layer * 14
          const attentionInput = yield* rms(hidden, rmsEpsilon, params[offset])
          let query = yield* headsFirst(
            yield* Tensor.linearRows(attentionInput, params[offset + 2]),
            queryHeads,
            keyLength
          )
          let key = yield* headsFirst(
            yield* Tensor.linearRows(attentionInput, params[offset + 3]),
            keyValueHeads,
            keyLength
          )
          const value = yield* headsFirst(
            yield* Tensor.linearRows(attentionInput, params[offset + 4]),
            keyValueHeads,
            valueLength
          )
          const gate = yield* Tensor.linearRows(attentionInput, params[offset + 7])

          query = yield* rms(query, rmsEpsilon, params[offset + 5])
          key = yield* rms(key, rmsEpsilon, params[offset + 6])
          const local = (layer + 1) % slidingWindowPattern !== 0
          if (local) {
            query = yield* Tensor.rotaryEmbedding(query, sequence, config["rope.freq_base"], {
              layout: "InterleavedPairs"
            })
            key = yield* Tensor.rotaryEmbedding(key, sequence, config["rope.freq_base"], {
              layout: "InterleavedPairs"
            })
          }

          let attention = yield* Tensor.scaledDotProductAttention(query, key, value, {
            scale: 1 / Math.sqrt(keyLength),
            causal: true,
            window: local ? config["attention.sliding_window"] : null
          })
          if (sequence === 1) {
            attention = yield* Tensor.reshape(attention, [batch, sequence, attentionSize])
          } else {
            attention = yield* Tensor.reshape(
              yield* Tensor.transpose(attention, [0, 2, 1, 3]),
              [batch, sequence, attentionSize]
            )
          }
          attention = yield* Tensor.mul(attention, yield* Tensor.sigmoid(gate))
          attention = yield* Tensor.linearRows(attention, params[offset + 8])
          hidden = yield* Tensor.add(hidden, yield* rms(attention, 1e-8, params[offset + 1]))

          const ffnInput = yield* rms(hidden, rmsEpsilon, params[offset + 9])
          let ffn = yield* Tensor.mul(
            yield* Tensor.silu(yield* Tensor.linearRows(ffnInput, params[offset + 11])),
            yield* Tensor.linearRows(ffnInput, params[offset + 12])
          )
          ffn = yield* Tensor.linearRows(ffn, params[offset + 13])
          hidden = yield* Tensor.add(hidden, yield* rms(ffn, 1e-8, params[offset + 10]))
        }

        hidden = yield* rms(hidden, rmsEpsilon, params[1])
        let logits = yield* Tensor.linearRows(hidden, params[2])
        logits = yield* Tensor.mul(logits, yield* Tensor.constantLike(logits, config.logit_scale))
        logits = yield* Tensor.div(logits, yield* Tensor.constantLike(logits, config.final_logit_softcapping))
        return yield* Tensor.mul(
          yield* Tensor.tanh(logits),
          yield* Tensor.constantLike(logits, config.final_logit_softcapping)
        )
      })
  })
}

/**
 * The Muse-Glimmer model architecture.
 *
 * @since 0.1.0
 * @category models
 */
export const architecture: Registry.ModelArchitecture = {
  id,
  create: (config) => Effect.flatMap(decodeConfig(config), makeModel)
}

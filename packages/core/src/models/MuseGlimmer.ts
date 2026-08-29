/**
 * The load-only Muse-Glimmer transformer architecture used by GGUF artifacts.
 *
 * {@link loadGGUF} consumes the canonical configuration prepared from a GGUF
 * artifact, validates its tensor catalog, and loads owned parameters. The
 * resulting {@link Model.Model} declares the GGUF tensor names and logical
 * shapes, then builds a stateless full-sequence graph. `Model.inference` can
 * then specialize that same graph into paged-KV prefill and decode
 * programs.
 *
 * The graph treats supported storage formats alike.
 * Dense F32 parameters and `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, or `Q6_K` matrices
 * have the same logical catalog. Packed parameters retain logical F32 shape
 * and dtype over row-packed U8 storage; their last logical dimension must be a
 * complete 256-value block. `Tensor.embedding` and `Tensor.linearRows`
 * dispatch that storage internally, while normalization vectors are expected
 * to remain dense. No complete dequantized copy is constructed by this module.
 *
 * The expected GGUF is already converted for this graph: Q/K projection rows
 * use GGML's interleaved-pair RoPE convention, and centered normalization
 * scales have already been shifted into multiplicative RMS weights (including
 * the artifact's Q- and K-normalization scales). Neither this architecture nor
 * `loadGGUF` repeats those conversions or synthesizes replacement tensors.
 *
 * @since 0.1.0
 */
import { Effect } from "effect"
import * as Schema from "effect/Schema"
import * as Gguf from "../Gguf.ts"
import * as Model from "../Model.ts"
import type * as Runtime from "../Runtime.ts"
import * as Tensor from "../Tensor.ts"

/**
 * The exact `general.architecture` value required in a Muse-Glimmer GGUF
 * artifact.
 *
 * @since 0.1.0
 * @category identifiers
 */
export const architecture = "muse-glimmer"

const PositiveInt = Schema.Int.check(Schema.isGreaterThan(0))
const PositiveFinite = Schema.Finite.check(Schema.isGreaterThan(0))
const MAX_BLOCKS = 1024

/**
 * Architecture fields after GGUF metadata has been canonicalized.
 *
 * For a Muse-Glimmer artifact, `loadGGUF` removes the `muse-glimmer.` prefix
 * from architecture metadata, removes `general.` from general metadata, and
 * derives `vocab_size` from `tokenizer.ggml.tokens.length` when the metadata
 * does not provide it. This schema then requires the fifteen fields below.
 * Counts are positive integers and epsilon, RoPE, and logit constants are
 * positive finite numbers. In addition, the query-head count must be divisible
 * by the KV-head count, key and value widths must match for paged decode, the
 * key width must be even for RoPE, and the local window must not exceed the
 * declared context length.
 *
 * `context_length` bounds the `S` dimension accepted by a direct `forward`
 * call. `attention.sliding_window_pattern = P` makes every Pth one-based layer
 * global; all others use the explicit local window. Key and value lengths are
 * represented separately in GGUF metadata, but this architecture requires the
 * canonical artifact's equal widths because paged decode has one shared head
 * dimension for K and V storage.
 */
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
    if (config.block_count > MAX_BLOCKS) {
      issues.push({ path: ["block_count"], issue: `block_count must not exceed ${MAX_BLOCKS}` })
    }
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
    if (config["attention.value_length"] !== config["attention.key_length"]) {
      issues.push({
        path: ["attention.value_length"],
        issue: "attention.value_length must equal attention.key_length"
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

/**
 * Converts the canonical GGUF metadata map to the validated architecture closure.
 * Translation from container-qualified GGUF metadata has already happened in
 * `Gguf.loadModel`; direct callers of `create` must therefore provide these
 * canonical keys themselves. Schema failures become `ModelError` with
 * `op = "create"` before a parameter catalog is exposed.
 */
const decodeConfig = (config: Gguf.ModelConfig): Effect.Effect<Config, Model.ModelError> =>
  Schema.decodeUnknownEffect(Config)(Object.fromEntries(config)).pipe(
    Effect.mapError((error) =>
      new Model.ModelError({
        op: "create",
        message: `MuseGlimmer config: ${error.message}`
      })
    )
  )

/**
 * Builds the exact flat GGUF parameter catalog. Let:
 *
 * - `L = block_count`, `E = embedding_length`, `F = feed_forward_length`;
 * - `V = vocab_size`, `Hq = attention.head_count`;
 * - `Hkv = attention.head_count_kv`, `Dk = attention.key_length`;
 * - `Dv = attention.value_length`, and `A = Hq * Dv`.
 *
 * Array order starts with the following three global parameters:
 *
 * 1. `token_embd.weight [V, E]`
 * 2. `output_norm.weight [E]`
 * 3. `output.weight [V, E]`
 *
 * Each zero-based layer `N = 0..L-1` then contributes these fourteen entries,
 * in exactly this order:
 *
 * 1. `blk.N.attn_norm.weight [E]`
 * 2. `blk.N.post_attention_norm.weight [E]`
 * 3. `blk.N.attn_q.weight [Hq * Dk, E]`
 * 4. `blk.N.attn_k.weight [Hkv * Dk, E]`
 * 5. `blk.N.attn_v.weight [Hkv * Dv, E]`
 * 6. `blk.N.attn_q_norm.weight [Dk]`
 * 7. `blk.N.attn_k_norm.weight [Dk]`
 * 8. `blk.N.attn_gate.weight [A, E]`
 * 9. `blk.N.attn_output.weight [E, A]`
 * 10. `blk.N.ffn_norm.weight [E]`
 * 11. `blk.N.post_ffw_norm.weight [E]`
 * 12. `blk.N.ffn_gate.weight [F, E]`
 * 13. `blk.N.ffn_up.weight [F, E]`
 * 14. `blk.N.ffn_down.weight [E, F]`
 *
 * The model arity is `3 + 14 * L`. The spelling `post_ffw` is
 * the source tensor name, not a normalized API alias. `loadGGUF` checks that
 * names and shapes match this catalog one-to-one, then reorders loaded handles
 * into this array order before returning them.
 */
const makeParameters = (config: Config): ReadonlyArray<Model.ParameterSpec> => {
  const hiddenSize = config.embedding_length
  const feedForwardSize = config.feed_forward_length
  const querySize = config["attention.head_count"] * config["attention.key_length"]
  const keySize = config["attention.head_count_kv"] * config["attention.key_length"]
  const valueSize = config["attention.head_count_kv"] * config["attention.value_length"]
  const attentionSize = config["attention.head_count"] * config["attention.value_length"]
  const normal = (fanIn: number): Model.ParameterInitializer => ({ _tag: "Normal", scale: 1 / Math.sqrt(fanIn) })
  const one: Model.ParameterInitializer = { _tag: "Constant", value: 1 }
  const specs: Array<Model.ParameterSpec> = [
    { name: "token_embd.weight", shape: [config.vocab_size, hiddenSize], initializer: normal(hiddenSize) },
    { name: "output_norm.weight", shape: [hiddenSize], initializer: one },
    { name: "output.weight", shape: [config.vocab_size, hiddenSize], initializer: normal(hiddenSize) }
  ]
  for (let layer = 0; layer < config.block_count; layer++) {
    const prefix = `blk.${layer}`
    specs.push(
      { name: `${prefix}.attn_norm.weight`, shape: [hiddenSize], initializer: one },
      { name: `${prefix}.post_attention_norm.weight`, shape: [hiddenSize], initializer: one },
      { name: `${prefix}.attn_q.weight`, shape: [querySize, hiddenSize], initializer: normal(hiddenSize) },
      { name: `${prefix}.attn_k.weight`, shape: [keySize, hiddenSize], initializer: normal(hiddenSize) },
      { name: `${prefix}.attn_v.weight`, shape: [valueSize, hiddenSize], initializer: normal(hiddenSize) },
      { name: `${prefix}.attn_q_norm.weight`, shape: [config["attention.key_length"]], initializer: one },
      { name: `${prefix}.attn_k_norm.weight`, shape: [config["attention.key_length"]], initializer: one },
      { name: `${prefix}.attn_gate.weight`, shape: [attentionSize, hiddenSize], initializer: normal(hiddenSize) },
      { name: `${prefix}.attn_output.weight`, shape: [hiddenSize, attentionSize], initializer: normal(attentionSize) },
      { name: `${prefix}.ffn_norm.weight`, shape: [hiddenSize], initializer: one },
      { name: `${prefix}.post_ffw_norm.weight`, shape: [hiddenSize], initializer: one },
      { name: `${prefix}.ffn_gate.weight`, shape: [feedForwardSize, hiddenSize], initializer: normal(hiddenSize) },
      { name: `${prefix}.ffn_up.weight`, shape: [feedForwardSize, hiddenSize], initializer: normal(hiddenSize) },
      { name: `${prefix}.ffn_down.weight`, shape: [hiddenSize, feedForwardSize], initializer: normal(feedForwardSize) }
    )
  }
  return specs
}

/**
 * Defines the load-only forward graph.
 *
 * Input is token IDs shaped `[B, S]`; `Tensor.embedding` requires `u32` or
 * `i64`, and `S` must not exceed `context_length`. The output is logits shaped
 * `[B, S, V]`. Parameter arity, input rank, and the upper bound on `S` are the
 * only model-level forward checks; this function does not require positive
 * `B` or `S`. Individual tensor operations validate parameter shape, dtype,
 * placement, and token dtype; out-of-vocabulary token IDs fail when the
 * embedding executes.
 *
 * For each layer, Q/K/V have shapes `[B, Hq, S, Dk]`,
 * `[B, Hkv, S, Dk]`, and `[B, Hkv, S, Dv]`. Causal GQA produces
 * `[B, Hq, S, Dv]`, which is flattened to `[B, S, Hq * Dv]`, sigmoid-gated,
 * projected to `E`, post-normalized with epsilon `1e-8`, and added residually.
 * Dense SwiGLU follows the same pre-norm/residual structure and also uses
 * epsilon `1e-8` for its post-FFN normalization. Embedding, pre-layer, Q/K,
 * and final normalization use `attention.layer_norm_rms_epsilon`.
 *
 * A zero-based layer `N` is local exactly when `(N + 1) % P !== 0`. Local
 * layers apply full-width interleaved-pair RoPE and pass the explicit window
 * `W`; every Pth layer applies no positional encoding and passes `null` for
 * explicit full causal attention. Thus `L = 52, P = 4` gives 39 local and 13
 * global layers. The final head computes
 * `softcap * tanh(logit_scale * output(rms(h)) / softcap)`.
 *
 * Direct `forward` calls have no mutable state and recompute attention over the
 * supplied sequence. During `Model.inference`, all causal attention nodes are
 * rewritten to per-sequence paged KV state and local RoPE positions use the
 * sequence cursor. This graph contains no KDA or short-convolution nodes, so it
 * requests no KDA matrix state or recurrent convolution window. When the
 * pattern selects any global layer, as the canonical `L = 52, P = 4` pattern
 * does, that layer prevents window-based KV eviction for the whole shared pool.
 * `InferenceConfig.attentionWindow` cannot override explicit full attention or
 * an explicit local-layer window.
 */
const makeModel = (config: Config): Effect.Effect<Model.Model, Model.ModelError> => {
  const parameterSpecs = makeParameters(config)
  const queryHeads = config["attention.head_count"]
  const keyValueHeads = config["attention.head_count_kv"]
  const keyLength = config["attention.key_length"]
  const valueLength = config["attention.value_length"]
  const attentionSize = queryHeads * valueLength
  const rmsEpsilon = config["attention.layer_norm_rms_epsilon"]
  const slidingWindowPattern = config["attention.sliding_window_pattern"]
  return Model.define({
    parameterSpecs,
    forward: (params, input) =>
      Effect.gen(function*() {
        if (params.length !== parameterSpecs.length) {
          return yield* new Model.ModelError({
            op: "forward",
            message: `MuseGlimmer.forward: expected ${parameterSpecs.length} parameters, got ${params.length}`
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
        hidden = yield* Tensor.rmsNorm(hidden, undefined, rmsEpsilon)
        // [B, S, H * W] <-> [B, H, S, W]; unit-axis permutes lower to
        // zero-cost aliases, so the S=1 decode path needs no special case.
        const headsFirst = (value: Tensor.Any, heads: number, width: number) =>
          Effect.gen(function*() {
            return yield* Tensor.transpose(
              yield* Tensor.reshape(value, [batch, sequence, heads, width]),
              [0, 2, 1, 3]
            )
          })
        const mergeHeads = (value: Tensor.Any) =>
          Effect.gen(function*() {
            return yield* Tensor.reshape(
              yield* Tensor.transpose(value, [0, 2, 1, 3]),
              [batch, sequence, attentionSize]
            )
          })

        for (let layer = 0; layer < config.block_count; layer++) {
          const offset = 3 + layer * 14
          const attentionInput = yield* Tensor.rmsNorm(hidden, params[offset], rmsEpsilon)
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

          query = yield* Tensor.rmsNorm(query, params[offset + 5], rmsEpsilon)
          key = yield* Tensor.rmsNorm(key, params[offset + 6], rmsEpsilon)
          // Every Pth one-based layer is explicitly global/NoPE; all other
          // layers are explicitly local and use interleaved-pair RoPE.
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
          attention = yield* mergeHeads(attention)
          attention = yield* Tensor.mul(attention, yield* Tensor.sigmoid(gate))
          attention = yield* Tensor.linearRows(attention, params[offset + 8])
          hidden = yield* Tensor.add(hidden, yield* Tensor.rmsNorm(attention, params[offset + 1], 1e-8))

          const ffnInput = yield* Tensor.rmsNorm(hidden, params[offset + 9], rmsEpsilon)
          let ffn = yield* Tensor.mul(
            yield* Tensor.silu(yield* Tensor.linearRows(ffnInput, params[offset + 11])),
            yield* Tensor.linearRows(ffnInput, params[offset + 12])
          )
          ffn = yield* Tensor.linearRows(ffn, params[offset + 13])
          hidden = yield* Tensor.add(hidden, yield* Tensor.rmsNorm(ffn, params[offset + 10], 1e-8))
          hidden = yield* Tensor.expose(hidden, Model.hiddenExposure(layer))
        }

        hidden = yield* Tensor.rmsNorm(hidden, params[1], rmsEpsilon)
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
 * Validates canonical Muse-Glimmer metadata and returns a load-only model
 * template. This performs no runtime access, file I/O, or weight loading.
 *
 * @since 0.1.0
 * @category models
 */
export const create = (config: Gguf.ModelConfig): Effect.Effect<Model.Model, Model.ModelError> =>
  Effect.flatMap(decodeConfig(config), makeModel)

/**
 * Explicit GGUF model definition used by {@link loadGGUF}.
 *
 * @since 0.1.0
 * @category models
 */
export const definition: Gguf.ModelDefinition = {
  architecture,
  create
}

/**
 * Inspects, validates, and loads a Muse-Glimmer GGUF artifact. The artifact's
 * `general.architecture` must equal {@link architecture} exactly.
 *
 * @since 0.1.0
 * @category loading
 */
export const loadGGUF = (
  path: string
): Effect.Effect<Gguf.LoadedModel, Gguf.GgufError | Model.ModelError, Runtime.Runtime> =>
  Gguf.loadModel(path, definition)

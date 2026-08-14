/**
 * The load-only Muse-Glimmer transformer architecture used by GGUF artifacts.
 *
 * {@link architecture} consumes the canonical configuration prepared by
 * `Gguf.load`; it does not inspect a file or load weights. The resulting
 * {@link Model.Model} declares the GGUF tensor names and logical shapes, then
 * builds a stateless full-sequence graph. `Model.inference` can subsequently
 * specialize that same graph into paged-KV prefill and decode programs.
 *
 * The implementation is storage-independent at the architecture boundary.
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
 * `Gguf.load` repeats those conversions or synthesizes replacement tensors.
 *
 * @since 0.1.0
 */
import { Effect } from "effect"
import * as Schema from "effect/Schema"
import * as Model from "../Model.ts"
import type * as Registry from "../Registry.ts"
import * as Tensor from "../Tensor.ts"

/**
 * The exact registry identifier for Muse-Glimmer GGUF artifacts.
 *
 * `Gguf.load` reads `general.architecture = "muse-glimmer"`, prefixes it with
 * `"gguf:"`, and performs an exact registry lookup. There are no unqualified,
 * case-insensitive, or alternate aliases for this identifier.
 *
 * @since 0.1.0
 * @category identifiers
 */
export const id = "gguf:muse-glimmer"

const PositiveInt = Schema.Int.check(Schema.isGreaterThan(0))
const PositiveFinite = Schema.Finite.check(Schema.isGreaterThan(0))
const MAX_BLOCKS = 1024

/**
 * Architecture fields after GGUF metadata has been canonicalized.
 *
 * For a Muse-Glimmer artifact, `Gguf.load` removes the `muse-glimmer.` prefix
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
 * Converts the canonical registry map to the validated architecture closure.
 * Translation from container-qualified GGUF metadata has already happened in
 * `Gguf.load`; direct callers of `architecture.create` must therefore provide
 * these canonical keys themselves. Schema failures become `ModelError` with
 * `op = "create"` before a parameter catalog is exposed.
 */
const decodeConfig = (config: Registry.ModelConfig): Effect.Effect<Config, Model.ModelError> =>
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
 * The model arity is consequently `3 + 14 * L`. The spelling `post_ffw` is
 * the source tensor name, not a normalized API alias. `Gguf.load` proves an
 * exact name/shape bijection with this catalog and reorders loaded handles into
 * this array order before returning them.
 */
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

/**
 * Defines the load-only forward graph.
 *
 * Input is token IDs shaped `[B, S]`; `Tensor.embedding` requires `u32` or
 * `i64`, and `S` must not exceed `context_length`. The output is logits shaped
 * `[B, S, V]`. Parameter arity, input rank, and the upper bound on `S` are the
 * only model-level forward checks; this function does not require positive
 * `B` or `S`. Individual tensor operations validate parameter shape, dtype,
 * placement, and token dtype; out-of-vocabulary token IDs fail when the
 * embedding executes. Because no initializer is supplied to `Model.define`,
 * `model.init` fails and parameters must be loaded.
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
        // S=1 is already heads-first after reshape; longer sequences need the
        // sequence and head axes exchanged.
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
 * The Muse-Glimmer GGUF model architecture.
 *
 * `create` validates canonical metadata and returns a load-only model template;
 * it performs no runtime access, file I/O, weight loading, or registration.
 * The architecture is re-exported from `@effect-torch/core/models`, while the
 * core `Registry.layer` installs it under the exact {@link id}. Custom
 * registries must register this value explicitly before `Gguf.load` can resolve
 * a Muse-Glimmer artifact.
 *
 * @since 0.1.0
 * @category models
 */
export const architecture: Registry.ModelArchitecture = {
  id,
  create: (config) => Effect.flatMap(decodeConfig(config), makeModel)
}

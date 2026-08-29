import { Model } from "@effect-torch/core"
import type { Tensor } from "@effect-torch/core"
import { Effect } from "effect"

// FineWeb KDA pilot based on the FineWeb GPT layout and token mixing from Kimi
// Linear and K3 (arXiv:2510.26692, arXiv:2607.24653). Layers 1-3 and 5 use Kimi
// Delta Attention (Model.kimiDeltaAttention, RFC 0018). Layers 4 and 6 use full
// causal attention. This gives a 2:1 recurrent-to-full-attention split. The
// final layer uses global attention rather than K3's gated NoPE MLA. KDA's
// decayed delta-rule transition carries position information in recurrent
// layers, so this small stack has neither a position table nor RoPE. The cited
// NoPE result applies to the global attention layers, not KDA itself.
//
// Model.inference rewrites KDA layers into stateful recurrent decoding with
// fixed-size state for each sequence. Training uses the closed-form KDA backward
// pass from RFC 0018 phase 4.
//
/** Re-exports tokenizer, parameter I/O, windowing, and evaluation helpers. */
export { EOT, heldOutLoss, loadBin, loadParams, loadTokenizer, saveParams, windows } from "../fineweb/model.js"

/** Default path for bare hybrid-KDA model parameters. */
export const CHECKPOINT = new URL("../data/fineweb-kda-model.safetensors", import.meta.url).pathname

/** Training sequence length and full-attention inference window. */
export const BLOCK = Number(process.env.FINEWEB_BLOCK ?? 256)
/** Transformer residual width. */
export const EMBED = 256
/** Number of token-mixing heads per transformer block. */
export const HEADS = 4
/** Number of transformer blocks. */
export const LAYERS = 6

/** Builds the FineWeb hybrid KDA/full-attention model for a vocabulary size. */
export const createKdaGpt = (
  vocabSize: number
): Effect.Effect<Model.Model, Model.ModelError | Tensor.TensorError> =>
  Effect.gen(function*() {
    const embeddings = yield* Model.embedding("wte", vocabSize, EMBED)
    const blocks: Array<Model.Model> = []
    for (let i = 0; i < LAYERS; i++) {
      const isKda = i !== LAYERS - 1 && (i + 1) % 4 !== 0
      const core = isKda
        ? yield* Model.kimiDeltaAttention(`b${i}.attn`, EMBED, HEADS)
        : yield* Model.multiHeadAttention(`b${i}.attn`, EMBED, HEADS, { causal: true })
      const attn = yield* Model.chain(yield* Model.layerNorm(`b${i}.ln1`, EMBED), core)
      const mlp = yield* Model.chain(
        yield* Model.layerNorm(`b${i}.ln2`, EMBED),
        yield* Model.linear(`b${i}.fc`, EMBED, 4 * EMBED),
        yield* Model.gelu(),
        yield* Model.linear(`b${i}.proj`, 4 * EMBED, EMBED)
      )
      blocks.push(yield* Model.chain(yield* Model.residual(attn), yield* Model.residual(mlp)))
    }
    const model = yield* Model.chain(
      embeddings,
      ...blocks,
      yield* Model.layerNorm("lnf", EMBED),
      yield* Model.linear("head", EMBED, vocabSize)
    )
    return model
  })

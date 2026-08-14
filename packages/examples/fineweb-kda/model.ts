import { Model } from "@effect-torch/core"
import type { Tensor } from "@effect-torch/core"
import { Effect } from "effect"

// FineWeb KDA pilot: the FineWeb GPT layout with a Kimi Linear/K3-inspired
// token-mixing pattern (arXiv:2510.26692, arXiv:2607.24653). Layers 1-3 and 5
// use Kimi Delta Attention (Model.kimiDeltaAttention, RFC 0018); layers 4 and 6
// use ordinary full causal attention. This is an overall 2:1 recurrent/full
// split with the final layer forced to global attention, not K3's gated NoPE
// MLA. KDA's decayed delta-rule transition supplies positional information for
// recurrent layers, and this small stack uses neither a position table nor RoPE.
// The cited NoPE result concerns the global attention layers rather than adding
// RoPE to KDA itself.
//
// Generation runs through Model.inference, which rewrites the KDA layers
// into stateful recurrent decode with fixed-size per-sequence state.
// Training runs the closed-form KDA backward (RFC 0018 phase 4).
//
// Shared fineweb helpers (tokenizer, bare-parameter I/O, data windows,
// held-out loss) are re-exported from the sibling example. FINEWEB_BLOCK is
// Number-parsed at module load and is not encoded in the bare model artifact.

export { EOT, heldOutLoss, loadBin, loadParams, loadTokenizer, saveParams, windows } from "../fineweb/model.js"

export const CHECKPOINT = new URL("../data/fineweb-kda-model.safetensors", import.meta.url).pathname

export const BLOCK = Number(process.env.FINEWEB_BLOCK ?? 256)
export const EMBED = 256
export const HEADS = 4
export const LAYERS = 6

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

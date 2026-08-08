import { Model } from "@effect-torch/core"
import type { Tensor } from "@effect-torch/core"
import { Effect } from "effect"

// FineWeb KDA pilot: the fineweb GPT layout on the Kimi Linear / Kimi
// K3 attention recipe (arXiv:2510.26692, arXiv:2607.24653) — a 3:1
// hybrid: every fourth layer and the final layer are full causal
// attention, the rest are Kimi Delta Attention
// (Model.kimiDeltaAttention, RFC 0018). The full layers stand in for
// K3's gated NoPE MLA; KDA's decayed delta-rule state transition is
// itself the positional encoding, so the stack uses no position table
// and no RoPE anywhere. (Pure linear attention measurably loses on
// exact retrieval, which is why the global layers exist; adding RoPE
// back to KDA layers hurts long-context performance.)
//
// Generation runs through Model.inference, which rewrites the KDA layers
// into stateful recurrent decode with fixed-size per-sequence state.
// Training runs the closed-form KDA backward (RFC 0018 phase 4).
//
// Shared fineweb helpers (tokenizer, checkpoint I/O, data windows,
// held-out loss) are re-exported from the sibling example.

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

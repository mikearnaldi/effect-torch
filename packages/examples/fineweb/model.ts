import { Effect } from "effect"
import { Device, Model, Tensor, Tokenizer } from "@effect-torch/core"

// Shared pieces of the FineWeb pre-training pilot (see train.ts
// and infer.ts): the GPT-2-architecture model, its size
// constants, the pretrained GPT-2 BPE tokenizer, and checkpoint
// save/load (safetensors, keyed by model.names).

export const TOKENIZER_JSON = new URL("../data/gpt2-tokenizer.json", import.meta.url).pathname
export const CHECKPOINT = new URL("../data/fineweb-model.safetensors", import.meta.url).pathname
export const EOT = "<|endoftext|>"

export const BLOCK = 256
export const EMBED = 256
export const HEADS = 4
export const LAYERS = 6

export const createGpt = (
  vocabSize: number
): Effect.Effect<Model.CompiledModel, Model.ModelError | Tensor.TensorError, Device.CurrentDevice> =>
  Effect.gen(function* () {
    // Token embeddings; positions are relative (RoPE inside attention),
    // so generation is unbounded — no position table to outgrow.
    const embeddings = yield* Model.embedding("wte", vocabSize, EMBED)
    const blocks: Array<Model.Model> = []
    for (let i = 0; i < LAYERS; i++) {
      const attn = yield* Model.chain(
        yield* Model.layerNorm(`b${i}.ln1`, EMBED),
        yield* Model.multiHeadAttention(`b${i}.attn`, EMBED, HEADS, { causal: true, rope: 10000 })
      )
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
    return yield* Model.compile(model)
  })

export const loadTokenizer = Tokenizer.fromFile(TOKENIZER_JSON, {
  padding: Tokenizer.paddingNone,
  truncation: Tokenizer.truncationNone,
  specialTokens: "Always"
})

/** Saves params as a safetensors checkpoint keyed by parameter name. */
export const saveParams = (model: Model.Model, params: Model.Params, path: string) =>
  Tensor.save(path, Object.fromEntries(model.names.map((name, i) => [name, params[i]])))

/** Loads a checkpoint back into the model's parameter order. */
export const loadParams = (
  model: Model.Model,
  path: string
): Effect.Effect<Model.Params, Tensor.TensorError, Device.CurrentDevice> =>
  Effect.gen(function* () {
    const tensors = yield* Tensor.load(path)
    return model.names.map((name) => {
      const tensor = tensors[name]
      if (tensor === undefined) throw new Error(`checkpoint ${path} is missing parameter ${name}`)
      return tensor
    })
  })

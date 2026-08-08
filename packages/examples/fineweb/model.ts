import { Loss, Model, Tensor } from "@effect-torch/core"
import type { Runtime } from "@effect-torch/core"
import * as Tokenizer from "@effect-torch/tokenizers"
import { Effect } from "effect"
import fs from "node:fs"

// Shared pieces of the FineWeb pre-training pilot (see train.ts
// and infer.ts): the GPT-2-architecture model, its size
// constants, the pretrained GPT-2 BPE tokenizer, and checkpoint
// save/load (safetensors, keyed by model.names).

export const TOKENIZER_JSON = new URL("../data/gpt2-tokenizer.json", import.meta.url).pathname
export const CHECKPOINT = new URL("../data/fineweb-model.safetensors", import.meta.url).pathname
export const EOT = "<|endoftext|>"

export const BLOCK = Number(process.env.FINEWEB_BLOCK ?? 256)
export const EMBED = 256
export const HEADS = 4
export const LAYERS = 6

export const createGpt = (
  vocabSize: number
): Effect.Effect<Model.Model, Model.ModelError | Tensor.TensorError> =>
  Effect.gen(function*() {
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
    return model
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
): Effect.Effect<Model.Params, Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const tensors = yield* Tensor.load(path)
    return model.names.map((name) => {
      const tensor = tensors[name]
      if (tensor === undefined) throw new Error(`checkpoint ${path} is missing parameter ${name}`)
      return tensor
    })
  })

/** Reads a u16 token bin produced by prepare.ts. */
export const loadBin = (path: string) => {
  const buffer = fs.readFileSync(path)
  if (buffer.byteOffset % 2 !== 0) throw new Error("misaligned token bin buffer")
  return new Uint16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2)
}

/** Materializes batch windows (start offsets) into input/target id arrays. */
export const windows = (data: Uint16Array, starts: ReadonlyArray<number>, batch: number, block: number) => {
  const inputs = new Uint32Array(batch * block)
  const targets = new Uint32Array(batch * block)
  for (const [b, start] of starts.entries()) {
    for (let t = 0; t < block; t++) {
      inputs[b * block + t] = data[start + t]
      targets[b * block + t] = data[start + t + 1]
    }
  }
  return { inputs, targets }
}

/** Mean cross-entropy over `batches` random windows of `data`. */
export const heldOutLoss = (
  model: Model.Model,
  params: Model.Params,
  data: Uint16Array,
  batch: number,
  block: number,
  batches: number
) =>
  Effect.gen(function*() {
    // Forward + loss as ONE frozen program: the chunked-head rewrite
    // (RFC 0016 phase 2) fires inside it, so the full logits tensor is
    // never materialized during evaluation, and the program's arena plan
    // shares the training arena's memory instead of stacking on top.
    return yield* Effect.acquireUseRelease(
      Tensor.compile((inputs) =>
        Effect.gen(function*() {
          const logits = yield* model.forward(inputs.slice(0, -2), inputs[inputs.length - 2])
          return [yield* Loss.crossEntropy(logits, inputs[inputs.length - 1])]
        })
      ),
      (lossProgram) =>
        Effect.gen(function*() {
          let total = 0
          for (let i = 0; i < batches; i++) {
            const { inputs, targets } = windows(
              data,
              Array.from({ length: batch }, () => Math.floor(Math.random() * (data.length - block - 1))),
              batch,
              block
            )
            const input = yield* Tensor.fromTypedArray(inputs, [batch, block])
            const target = yield* Tensor.fromTypedArray(targets, [batch, block])
            const [lossTensor] = yield* lossProgram.call([...params, input, target])
            const [loss] = yield* Tensor.toNumberArray(lossTensor)
            yield* Tensor.clear(lossTensor)
            total += loss
          }
          return total / batches
        }),
      (lossProgram) => lossProgram.clear
    )
  })

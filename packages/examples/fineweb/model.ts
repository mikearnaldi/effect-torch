import { Loss, Model, Tensor } from "@effect-torch/core"
import type { Runtime } from "@effect-torch/core"
import * as Tokenizer from "@effect-torch/tokenizers"
import { Effect } from "effect"
import fs from "node:fs"

// Shared FineWeb model and data contracts. The training scripts consume flat
// u16 token streams, while inference consumes bare safetensors artifacts keyed
// by `model.names`; resumable Checkpoint archives use a different key schema
// and must be exported before `loadParams` can read them. Bare artifacts contain
// no architecture metadata, and loading checks names only; incompatible tensor
// shape/dtype semantics are deferred to later model execution.
// FINEWEB_BLOCK is Number-parsed once at module load and is not stored in bare
// artifacts; it controls both training windows and inference attention policy.

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
    // Token embeddings; RoPE inside attention means the architecture has no
    // learned position table to outgrow during windowed inference.
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

/** Saves a bare model artifact keyed by parameter name, without trainer state. */
export const saveParams = (model: Model.Model, params: Model.Params, path: string) =>
  Tensor.save(path, Object.fromEntries(model.names.map((name, i) => [name, params[i]])))

/**
 * Loads named bare parameters into model order. `Tensor.load` imports the
 * entire archive; {@link Model.load} releases extras and returns the required
 * handles.
 */
export const loadParams = (
  model: Model.Model,
  path: string
): Effect.Effect<ReadonlyArray<Tensor.Concrete>, Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Model.load(model, path)

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

/** Mean cross-entropy over random windows sampled with replacement and no fixed seed. */
export const heldOutLoss = (
  model: Model.Model,
  params: Model.Params,
  data: Uint16Array,
  batch: number,
  block: number,
  batches: number
) =>
  Effect.gen(function*() {
    // Forward and loss must be one compiled graph: this lets the chunked-head
    // cross-entropy rewrite consume the linear head directly, so evaluation
    // never exposes a full [batch, block, vocab] logits root. The compiled
    // signature is reused across batches; each program call completes before
    // its scalar loss is read back.
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

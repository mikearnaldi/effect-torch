import { Effect, Option } from "effect"
import { NodeRuntime } from "@effect/platform-node"
import { Device, Model, Tensor } from "@effect-torch/core"
import { BLOCK, createGpt, EOT, loadParams, loadTokenizer } from "./model.js"

// FineWeb inference: loads the checkpoint saved by train.ts and
// generates from prompts through the compiled inference artifact (paged
// kv cache, chunked prefill, sliding-window attention over the last
// BLOCK positions), sampling with temperature and stopping at
// <|endoftext|>.

const TEMPERATURE = 0.8
const MAX_NEW_TOKENS = 240
const CHECKPOINT = new URL("../backup/fineweb-ckpt.safetensors", import.meta.url).pathname

// Multinomial sampling with temperature over a probability vector.
const sampleCategorical = (probs: ReadonlyArray<number>, temperature: number) => {
  let sum = 0
  const scaled = probs.map((p) => {
    const v = Math.pow(Math.max(p, 1e-12), 1 / temperature)
    sum += v
    return v
  })
  let r = Math.random() * sum
  for (let i = 0; i < scaled.length; i++) {
    r -= scaled[i]
    if (r <= 0) return i
  }
  return scaled.length - 1
}

const program = Effect.gen(function* () {
  const tokenizer = yield* loadTokenizer
  const eotId = Option.getOrThrow(tokenizer.tokenToId(EOT))
  const model = yield* createGpt(tokenizer.vocabSize)
  yield* Effect.log(`fineweb-infer: loading checkpoint ${CHECKPOINT}`)
  const params = yield* loadParams(model, CHECKPOINT)

  const inference = yield* Model.inference(model, params, {
    maxTokens: 8192,
    blockSize: 16,
    attentionWindow: BLOCK
  })
  const prompts = [
    "The history of the printing press begins",
    "In a small village by the sea,",
    "Scientists have discovered that"
  ]
  for (const prompt of prompts) {
    const generated: Array<number> = []
    const gen = yield* inference.generation()
    const encoded = yield* tokenizer.encode(prompt)
    const entry = yield* gen.add(yield* Tensor.reshape(encoded, [1, encoded.shape[0]]))
    let logits = entry.logits
    for (let i = 0; i < MAX_NEW_TOKENS; i++) {
      const [probs] = yield* Tensor.compute([yield* Tensor.softmax(logits)])
      const token = sampleCategorical(yield* Tensor.toNumberArray(probs), TEMPERATURE)
      if (token === eotId) break
      generated.push(token)
      const [stepped] = yield* gen.step([{ seq: entry.seq, token }])
      logits = stepped
    }
    yield* gen.close()
    const text = yield* tokenizer.decode(generated)
    yield* Effect.log(`\n--- prompt: ${prompt}\n${prompt}${text}\n`)
  }
})

NodeRuntime.runMain(program.pipe(Effect.provide(Device.Best)))

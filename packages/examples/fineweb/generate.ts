import { Effect, Option } from "effect"
import { NodeRuntime } from "@effect/platform-node"
import { Device, Model, Tensor } from "@effect-torch/core"
import { BLOCK, CHECKPOINT, createGpt, EOT, loadParams, loadTokenizer } from "./model.js"

// Streaming generation: the prompt comes from argv and tokens are printed
// as they are produced (the full sequence is re-decoded each step and only
// the new suffix is written — BPE tokens don't split cleanly on word
// boundaries). Generation is unbounded (sliding-window attention, RoPE)
// and stops at <|endoftext|>. Usage:
//   pnpm tsx fineweb/generate.ts "The history of the printing press"
// FINEWEB_TEMPERATURE tunes the sampling.

const TEMPERATURE = Number(process.env.FINEWEB_TEMPERATURE ?? 0.8)

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
  const prompt = process.argv.slice(2).join(" ")
  if (prompt.length === 0) {
    yield* Effect.log("usage: pnpm tsx fineweb/generate.ts <prompt>")
    return
  }
  const tokenizer = yield* loadTokenizer
  const eotId = Option.getOrThrow(tokenizer.tokenToId(EOT))
  const model = yield* createGpt(tokenizer.vocabSize)
  const params = yield* loadParams(model, CHECKPOINT)

  const inference = yield* Model.inference(model, params, {
    maxTokens: 8192,
    blockSize: 16,
    attentionWindow: BLOCK
  })

  const generated: Array<number> = []
  let printed = ""
  const gen = yield* inference.generation()
  const encoded = yield* tokenizer.encode(prompt)
  const entry = yield* gen.add(yield* Tensor.reshape(encoded, [1, encoded.shape[0]]))
  let logits = entry.logits
  process.stdout.write(prompt)
  while (true) {
    const [probs] = yield* Tensor.compute([yield* Tensor.softmax(logits)])
    const token = sampleCategorical(yield* Tensor.toNumberArray(probs), TEMPERATURE)
    if (token === eotId) break
    generated.push(token)
    const text = yield* tokenizer.decode(generated)
    process.stdout.write(text.slice(printed.length))
    printed = text
    const [stepped] = yield* gen.step([{ seq: entry.seq, token }])
    logits = stepped
  }
  yield* gen.close()
  process.stdout.write("\n")
})

NodeRuntime.runMain(program.pipe(Effect.provide(Device.Best)))

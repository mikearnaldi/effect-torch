import * as BackendApple from "@effect-torch/backend-apple-native"
import { Model, Tensor } from "@effect-torch/core"
import { NodeRuntime } from "@effect/platform-node"
import { Effect, Option } from "effect"
import { BLOCK, CHECKPOINT, createGpt, EOT, loadParams, loadTokenizer } from "./model.js"

// Streaming generation: the prompt comes from argv and each token is
// decoded and printed as it is produced. Generation is unbounded
// (sliding-window attention, RoPE) and stops at <|endoftext|>. Usage:
//   pnpm tsx fineweb/generate.ts "The history of the printing press"
// FINEWEB_TEMPERATURE tunes the sampling.

const TEMPERATURE = Number(process.env.FINEWEB_TEMPERATURE ?? 0.2)

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

const program = Effect.gen(function*() {
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

  const gen = yield* inference.generation()
  const encoded = yield* tokenizer.encode(prompt)
  const entry = yield* gen.add(yield* Tensor.fromTypedArray(encoded.data, [1, encoded.shape[0]]))
  let logits = entry.logits
  // Incremental decode: re-decode the sequence per token, but hold back a
  // trailing run of U+FFFD — a merge boundary can split a multi-byte
  // codepoint, and the next token may complete it (genuine invalid bytes
  // still print, one token later; everything flushes at the end).
  const ids: Array<number> = []
  let emitted = 0
  let text = ""
  while (true) {
    const [probs] = yield* Tensor.compute([yield* Tensor.softmax(logits)])
    const token = sampleCategorical(yield* Tensor.toNumberArray(probs), TEMPERATURE)
    if (token === eotId) break
    ids.push(token)
    text = yield* tokenizer.decode(ids)
    let safe = text.length
    while (safe > emitted && text.charCodeAt(safe - 1) === 0xfffd) safe--
    if (safe > emitted) {
      process.stdout.write(text.slice(emitted, safe))
      emitted = safe
    }
    const [stepped] = yield* gen.step([{ seq: entry.seq, token }])
    logits = stepped
  }
  process.stdout.write(text.slice(emitted))
  yield* gen.close()
  process.stdout.write("\n")
})

NodeRuntime.runMain(program.pipe(Effect.provide(BackendApple.layer)))

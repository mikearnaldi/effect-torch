import * as BackendApple from "@effect-torch/backend-apple-native"
import { Model, Tensor } from "@effect-torch/core"
import { NodeRuntime } from "@effect/platform-node"
import { Effect, Option } from "effect"
import { BLOCK, CHECKPOINT, createKdaGpt, EOT, loadParams, loadTokenizer } from "./model.js"

// Streams generated text from the hybrid KDA model. Model.inference converts
// KDA and short-convolution layers to fixed-size recurrent state for each
// sequence. Only the two NoPE full-attention layers use the paged KV pool. The
// BLOCK window permits eviction after the pool reaches its 8,192-row capacity.
// Sampling reads the full vocabulary back to the host after each decode. The
// script sets no token limit. Only <|endoftext|> stops the loop. Usage:
//   pnpm tsx fineweb-kda/generate.ts "The history of the printing press"
// The script parses FINEWEB_TEMPERATURE with Number but does not validate its range.

const TEMPERATURE = Number(process.env.FINEWEB_TEMPERATURE ?? 0.2)

const sampleCategorical = (logits: ReadonlyArray<number>, temperature: number) => {
  const max = Math.max(...logits)
  let sum = 0
  const scaled = logits.map((logit) => {
    const v = Math.exp((logit - max) / temperature)
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

const program = Effect.scoped(Effect.gen(function*() {
  const prompt = process.argv.slice(2).join(" ")
  if (prompt.length === 0) {
    yield* Effect.log("usage: pnpm tsx fineweb-kda/generate.ts <prompt>")
    return
  }
  const tokenizer = yield* loadTokenizer
  const eotId = Option.getOrThrow(tokenizer.tokenToId(EOT))
  const model = yield* createKdaGpt(tokenizer.vocabSize)
  const params = yield* loadParams(model, CHECKPOINT)
  yield* Tensor.clearAllScoped(params)

  const inference = yield* Model.inference(model, params, {
    maxTokens: 8192,
    blockSize: 16,
    prefillChunks: [16],
    attentionWindow: BLOCK
  })

  const gen = yield* Effect.acquireRelease(
    inference.execution(),
    (gen) => Effect.ignore(gen.close())
  )
  const encoded = yield* tokenizer.encode(prompt)
  const entry = (yield* gen.add([yield* Tensor.fromTypedArray(encoded.data, [1, encoded.shape[0]])]))[0]!
  let logits = entry.logits
  // Decode the full generated sequence after each token. Hold back trailing
  // U+FFFD characters because a token boundary may split a multibyte codepoint.
  // The next token may complete it. Genuine invalid bytes print one token later,
  // and the final write flushes everything.
  const ids: Array<number> = []
  let emitted = 0
  let text = ""
  while (true) {
    const values = yield* Tensor.toNumberArray(logits).pipe(
      Effect.ensuring(Effect.ignore(Tensor.clear(logits)))
    )
    const token = sampleCategorical(values, TEMPERATURE)
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
  process.stdout.write("\n")
}))

NodeRuntime.runMain(program.pipe(Effect.provide(BackendApple.layer())))

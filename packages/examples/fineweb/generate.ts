import * as BackendApple from "@effect-torch/backend-apple-native"
import { Model, Tensor } from "@effect-torch/core"
import { NodeRuntime } from "@effect/platform-node"
import { Effect, Option } from "effect"
import { BLOCK, CHECKPOINT, createGpt, EOT, loadParams, loadTokenizer } from "./model.js"

// Single-sequence streaming generation from the bare model artifact written by
// train.ts or export.ts. Prompt prefill is chunked; each logits readback then
// transfers a completed decode row for host-side sampling. The BLOCK attention
// window and RoPE allow the logical cursor to advance beyond the 8,192-row KV
// capacity, but there is no application token limit: only <|endoftext|>
// terminates the loop, and the JavaScript id/text buffers continue to grow. Usage:
//   pnpm tsx fineweb/generate.ts "The history of the printing press"
// FINEWEB_TEMPERATURE is Number-parsed without range validation.

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
    yield* Effect.log("usage: pnpm tsx fineweb/generate.ts <prompt>")
    return
  }
  const tokenizer = yield* loadTokenizer
  const eotId = Option.getOrThrow(tokenizer.tokenToId(EOT))
  const model = yield* createGpt(tokenizer.vocabSize)
  const params = yield* loadParams(model, CHECKPOINT)
  yield* Tensor.clearAllScoped(params)

  const inference = yield* Model.inference(model, params, {
    maxTokens: 8192,
    blockSize: 16,
    attentionWindow: BLOCK
  })

  const gen = yield* Effect.acquireRelease(
    inference.generation(),
    (gen) => Effect.ignore(gen.close())
  )
  const encoded = yield* tokenizer.encode(prompt)
  const entry = yield* gen.add(yield* Tensor.fromTypedArray(encoded.data, [1, encoded.shape[0]]))
  let logits = entry.logits
  // Incremental decode: re-decode the sequence per token, but hold back a
  // trailing run of U+FFFD - a merge boundary can split a multi-byte
  // codepoint, and the next token may complete it (genuine invalid bytes
  // still print, one token later; everything flushes at the end).
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

NodeRuntime.runMain(program.pipe(Effect.provide(BackendApple.layer)))

import * as BackendApple from "@effect-torch/backend-apple-native"
import { Model, Tensor } from "@effect-torch/core"
import { NodeRuntime } from "@effect/platform-node"
import { Effect, Option } from "effect"
import { BLOCK, CHECKPOINT, createGpt, EOT, loadParams, loadTokenizer } from "./model.js"

// Runs batch-style FineWeb inference from a bare parameter artifact. `loadParams`
// cannot read resumable Checkpoint keys. Convert training archives with
// export.ts first. Model.inference freezes parameters and compiles fixed-shape
// prefill and decode programs backed by an 8,192-row paged pool. The BLOCK
// attention window allows eviction of old KV blocks. After each decode, the
// script reads the full vocabulary back to the host for sampling.

const TEMPERATURE = 0.8
const MAX_NEW_TOKENS = 240

// Multinomial sampling with temperature over a logits vector.
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
  const tokenizer = yield* loadTokenizer
  const eotId = Option.getOrThrow(tokenizer.tokenToId(EOT))
  const model = yield* createGpt(tokenizer.vocabSize)
  yield* Effect.log(`fineweb-infer: loading checkpoint ${CHECKPOINT}`)
  const params = yield* loadParams(model, CHECKPOINT)
  yield* Tensor.clearAllScoped(params)

  const inference = yield* Model.inference(model, params, {
    maxTokens: 8192,
    blockSize: 16,
    prefillChunks: [16],
    attentionWindow: BLOCK
  })
  const prompts = [
    "The history of the printing press begins",
    "In a small village by the sea,",
    "Scientists have discovered that"
  ]
  for (const prompt of prompts) {
    const generated: Array<number> = []
    yield* Effect.scoped(Effect.gen(function*() {
      const gen = yield* Effect.acquireRelease(
        inference.execution(),
        (gen) => Effect.ignore(gen.close())
      )
      const encoded = yield* tokenizer.encode(prompt)
      const entry = (yield* gen.add([yield* Tensor.fromTypedArray(encoded.data, [1, encoded.shape[0]])]))[0]!
      let logits = entry.logits
      for (let i = 0; i < MAX_NEW_TOKENS; i++) {
        const values = yield* Tensor.toNumberArray(logits).pipe(
          Effect.ensuring(Effect.ignore(Tensor.clear(logits)))
        )
        const token = sampleCategorical(values, TEMPERATURE)
        if (token === eotId) break
        generated.push(token)
        if (i + 1 < MAX_NEW_TOKENS) {
          const [stepped] = yield* gen.step([{ seq: entry.seq, token }])
          logits = stepped
        }
      }
    }))
    const text = yield* tokenizer.decode(generated)
    yield* Effect.log(`\n--- prompt: ${prompt}\n${prompt}${text}\n`)
  }
}))

NodeRuntime.runMain(program.pipe(Effect.provide(BackendApple.layer())))

// Inference smoke test that does not need a checkpoint. Randomly initialized
// weights run chunked prefill and eight stateful recurrent/KV decode steps. Each
// logits row used to choose the next token must contain only finite values. This
// checks inference specialization and state updates, not model quality.
// generate.ts requires a trained bare-parameter artifact.
import * as BackendApple from "@effect-torch/backend-apple-native"
import { Model, Tensor } from "@effect-torch/core"
import { NodeRuntime } from "@effect/platform-node"
import { Effect } from "effect"
import { createKdaGpt, loadTokenizer } from "./model.js"

const program = Effect.scoped(Effect.gen(function*() {
  const tokenizer = yield* loadTokenizer
  const model = yield* createKdaGpt(tokenizer.vocabSize)
  const params = yield* Tensor.compute(yield* Model.initialize(model))
  yield* Tensor.clearAllScoped(params)
  const total = params.reduce((sum, p) => sum + p.shape.reduce((a, b) => a * b, 1), 0)
  yield* Effect.log(`params: ${total.toLocaleString()}`)
  const inference = yield* Model.inference(model, params, {
    maxTokens: 1024,
    blockSize: 16,
    prefillChunks: [16],
    attentionWindow: 256
  })
  const gen = yield* Effect.acquireRelease(
    inference.execution(),
    (gen) => Effect.ignore(gen.close())
  )
  const encoded = yield* tokenizer.encode("The history of the printing press")
  const entry = (yield* gen.add([yield* Tensor.fromTypedArray(encoded.data, [1, encoded.shape[0]])]))[0]!
  let logits = entry.logits
  for (let i = 0; i < 8; i++) {
    const vals = yield* Tensor.toNumberArray(logits).pipe(
      Effect.ensuring(Effect.ignore(Tensor.clear(logits)))
    )
    const next = vals.reduce((best, v, j) => (v > vals[best] ? j : best), 0)
    const finite = vals.every(Number.isFinite)
    yield* Effect.log(`step ${i}: argmax ${next}, finite ${finite}`)
    if (!finite) return yield* Effect.fail(new Error("non-finite logits"))
    const [stepped] = yield* gen.step([{ seq: entry.seq, token: next }])
    logits = stepped
  }
  yield* Tensor.clear(logits)
  yield* Effect.log("smoke ok")
}))
NodeRuntime.runMain(program.pipe(Effect.provide(BackendApple.layer())))

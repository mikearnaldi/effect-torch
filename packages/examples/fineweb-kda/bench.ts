// Stateful single-sequence decode microbenchmark with random initialized
// weights. Model/parameter initialization, inference compilation, prompt
// prefill, and eight greedy warmup steps are outside the timed region. Every
// measured iteration includes one recurrent/KV decode, full-vocabulary host
// readback, JavaScript argmax, and output release; native execute already waits
// for device completion. This is latency methodology, not batched device-only
// kernel throughput.

import * as BackendApple from "@effect-torch/backend-apple-native"
import { Model, Tensor } from "@effect-torch/core"
import { NodeRuntime } from "@effect/platform-node"
import { Effect } from "effect"
import { createKdaGpt, loadTokenizer } from "./model.js"

const program = Effect.scoped(Effect.gen(function*() {
  const tokenizer = yield* loadTokenizer
  const model = yield* createKdaGpt(tokenizer.vocabSize)
  const params = yield* Tensor.compute(yield* model.init)
  yield* Tensor.clearAllScoped(params)
  const inference = yield* Model.inference(model, params, { maxTokens: 4096, blockSize: 16, attentionWindow: 256 })
  const gen = yield* Effect.acquireRelease(
    inference.generation(),
    (gen) => Effect.ignore(gen.close())
  )
  const encoded = yield* tokenizer.encode("The history of the printing press is a long one")
  const entry = yield* gen.add(yield* Tensor.fromTypedArray(encoded.data, [1, encoded.shape[0]]))
  let logits = entry.logits
  const argmax = (vals: Array<number>) => vals.reduce((best, v, j) => (v > vals[best] ? j : best), 0)
  // Warm the fixed decode artifact and backend pipelines before timing.
  let next = argmax(yield* Tensor.toNumberArray(logits).pipe(Effect.ensuring(Effect.ignore(Tensor.clear(logits)))))
  for (let i = 0; i < 8; i++) {
    const [l] = yield* gen.step([{ seq: entry.seq, token: next }])
    next = argmax(yield* Tensor.toNumberArray(l).pipe(Effect.ensuring(Effect.ignore(Tensor.clear(l)))))
  }
  const N = 64
  const t0 = Date.now()
  for (let i = 0; i < N; i++) {
    const [l] = yield* gen.step([{ seq: entry.seq, token: next }])
    next = argmax(yield* Tensor.toNumberArray(l).pipe(Effect.ensuring(Effect.ignore(Tensor.clear(l)))))
  }
  const ms = (Date.now() - t0) / N
  yield* Effect.log(`${ms.toFixed(2)} ms/token (${(1000 / ms).toFixed(1)} tok/s)`)
}))
NodeRuntime.runMain(program.pipe(Effect.provide(BackendApple.layer)))

import * as BackendApple from "@effect-torch/backend-apple-native"
import { Checkpoint, LearningRate, Loss, Optimizer, Tensor, Trainer } from "@effect-torch/core"
import { NodeRuntime } from "@effect/platform-node"
import { Effect } from "effect"
import { CHECKPOINT, createGpt, loadTokenizer, saveParams } from "./model.js"

// Exports a training checkpoint as a model artifact: strips the optimizer
// state, sampler state, and metadata, leaving bare parameter names that
// infer.ts (or any safetensors consumer) can load. Usage:
//   pnpm tsx fineweb/export.ts <checkpoint.safetensors> [out.safetensors]

const [, , source = new URL("../data/fineweb-epoch-ckpt.safetensors", import.meta.url).pathname, out = CHECKPOINT] =
  process.argv

const program = Effect.gen(function*() {
  const tokenizer = yield* loadTokenizer
  const model = yield* createGpt(tokenizer.vocabSize)
  // Checkpoint.load rebuilds optimizer state through the trainer's
  // optimizer, so it needs one — but it is never stepped here.
  const zero = yield* Tensor.zeros([1, 1])
  const trainer = yield* Trainer.make(model, {
    optimizer: yield* Optimizer.adamW(),
    lr: LearningRate.constant(0),
    loss: Loss.crossEntropy,
    data: { input: zero, target: yield* Tensor.zeros([1, 1], { dtype: "u32" }) },
    stop: () => true
  })
  yield* Effect.log(`exporting ${source}`)
  const checkpoint = yield* Checkpoint.load(source, trainer)
  yield* saveParams(model, checkpoint.params, out)
  yield* Effect.log(`exported ${checkpoint.params.length} parameters (step ${checkpoint.resume.step}) to ${out}`)
})

NodeRuntime.runMain(program.pipe(Effect.provide(BackendApple.layer)))

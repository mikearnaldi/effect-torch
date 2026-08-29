import * as BackendApple from "@effect-torch/backend-apple-native"
import { Checkpoint, LearningRate, Loss, Optimizer, Tensor, Trainer } from "@effect-torch/core"
import { NodeRuntime } from "@effect/platform-node"
import { Effect } from "effect"
import { CHECKPOINT, createGpt, loadTokenizer, saveParams } from "./model.js"

// Converts a compatible AdamW training archive into a bare model artifact.
// Checkpoint.load reconstructs optimizer state to decode the archive. The
// subsequent save writes only model parameters. It omits optimizer, sampler,
// step, and metadata entries. Checkpoint.load releases unselected imports, but
// the returned parameters and state remain caller-owned. The archive does not
// identify its trainer configuration, so this temporary trainer must use the
// source run's model and optimizer root schema even though it never steps. The
// bundled Metal backend atomically replaces the output. Usage:
//   pnpm tsx fineweb/export.ts <checkpoint.safetensors> [out.safetensors]

const [, , source = new URL("../data/fineweb-epoch-ckpt.safetensors", import.meta.url).pathname, out = CHECKPOINT] =
  process.argv

const program = Effect.gen(function*() {
  const tokenizer = yield* loadTokenizer
  const model = yield* createGpt(tokenizer.vocabSize)
  // Checkpoint.load uses the trainer's optimizer to rebuild state. The trainer
  // needs an optimizer even though it never steps here.
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

NodeRuntime.runMain(program.pipe(Effect.provide(BackendApple.layer())))

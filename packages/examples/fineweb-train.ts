import { Duration, Effect } from "effect"
import { NodeRuntime } from "@effect/platform-node"
import { Device, LearningRate, Loss, Optimizer, Tensor, Trainer } from "@effect-torch/core"
import fs from "node:fs"
import { BLOCK, CHECKPOINT, createGpt, loadTokenizer, saveParams } from "./fineweb-model.js"

// FineWeb pre-training: trains the shared GPT on the token bins produced
// by fineweb-prepare.ts (~745M GPT-2 BPE tokens, u16), reports a held-out
// loss estimate, and saves the trained parameters to a safetensors
// checkpoint for fineweb-infer.ts.

const TRAIN_BIN = new URL("./data/fineweb-train.bin", import.meta.url).pathname
const VAL_BIN = new URL("./data/fineweb-val.bin", import.meta.url).pathname

const BATCH = 32
const STEPS = 5000
const LR = 6e-4
const VAL_BATCHES = 20

const loadBin = (path: string) => {
  const buffer = fs.readFileSync(path)
  if (buffer.byteOffset % 2 !== 0) throw new Error("misaligned token bin buffer")
  return new Uint16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2)
}

const sampleBatch = (data: Uint16Array) =>
  Effect.gen(function* () {
    const inputs = new Uint32Array(BATCH * BLOCK)
    const targets = new Uint32Array(BATCH * BLOCK)
    for (let b = 0; b < BATCH; b++) {
      const start = Math.floor(Math.random() * (data.length - BLOCK - 1))
      for (let t = 0; t < BLOCK; t++) {
        inputs[b * BLOCK + t] = data[start + t]
        targets[b * BLOCK + t] = data[start + t + 1]
      }
    }
    return {
      input: yield* Tensor.fromTypedArray(inputs, [BATCH, BLOCK]),
      target: yield* Tensor.fromTypedArray(targets, [BATCH, BLOCK])
    }
  })

const program = Effect.gen(function* () {
  const train = loadBin(TRAIN_BIN)
  const val = loadBin(VAL_BIN)
  const tokenizer = yield* loadTokenizer
  yield* Effect.log(
    `fineweb-train: vocab ${tokenizer.vocabSize}, ${(train.length / 1e6).toFixed(0)}M train tokens, block ${BLOCK}, batch ${BATCH} (${(BATCH * BLOCK * STEPS / 1e6).toFixed(0)}M tokens over ${STEPS} steps)`
  )

  yield* Effect.log("1) creating model")
  const model = yield* createGpt(tokenizer.vocabSize)
  const params0 = yield* model.init
  const total = params0.reduce((sum, param) => sum + param.shape.reduce((a, b) => a * b, 1), 0)
  yield* Effect.log(`  total: ${total.toLocaleString()} parameters`)

  yield* Effect.log(`2) training: adamW lr=${LR}, ${STEPS} steps`)
  const trainer = yield* Trainer.compile(yield* Trainer.make(model, {
    optimizer: yield* Optimizer.adamW(),
    lr: LearningRate.constant(LR),
    loss: Loss.crossEntropy,
    data: () => sampleBatch(train),
    stop: ({ step }) => step >= STEPS,
    onStep: ({ step, loss, elapsed }) =>
      step % 50 === 0 || step === 1
        ? Effect.log(
          `step ${String(step).padStart(4)}  loss ${loss.toFixed(4)}  ${(Duration.toMillis(elapsed) / 1000).toFixed(1)}s`
        )
        : Effect.void
  }))
  const trained = yield* trainer.train(params0)
  const params = trained.params

  yield* Effect.log(`3) held-out loss over ${VAL_BATCHES} val batches`)
  let valLoss = 0
  for (let i = 0; i < VAL_BATCHES; i++) {
    const batch = yield* sampleBatch(val)
    const logits = yield* model.forward(params, batch.input)
    const [lossTensor] = yield* Tensor.compute([yield* Loss.crossEntropy(logits, batch.target)])
    const [loss] = yield* Tensor.toNumberArray(lossTensor)
    valLoss += loss
  }
  yield* Effect.log(`val loss ${(valLoss / VAL_BATCHES).toFixed(4)}`)

  yield* Effect.log(`4) saving checkpoint to ${CHECKPOINT}`)
  yield* saveParams(model, params, CHECKPOINT)
})

NodeRuntime.runMain(program.pipe(Effect.provide(Device.Best)))

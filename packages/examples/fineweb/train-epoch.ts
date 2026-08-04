import { Duration, Effect } from "effect"
import { NodeRuntime } from "@effect/platform-node"
import type { Model } from "@effect-torch/core"
import { Checkpoint, Device, LearningRate, Loss, Optimizer, Sampler, Tensor, Trainer } from "@effect-torch/core"
import fs from "node:fs"
import {
  BLOCK,
  CHECKPOINT,
  createGpt,
  heldOutLoss,
  loadBin,
  loadParams,
  loadTokenizer,
  saveParams,
  windows
} from "./model.js"

// Full-epoch training, warm-started from the pilot checkpoint
// (fineweb-model.safetensors, written by train.ts): parameters load from
// disk, AdamW starts fresh, and the learning rate follows linear warmup
// into a cosine decay over exactly one epoch of the 745M-token bin —
// every window seen exactly once (see Sampler). Checkpoints land in a
// separate file so a pilot checkpoint is never clobbered, and an
// interrupted epoch resumes bit-exactly (params, optimizer, step, data
// layout). The final parameters replace fineweb-model.safetensors.

const TRAIN_BIN = new URL("../data/fineweb-train.bin", import.meta.url).pathname
const VAL_BIN = new URL("../data/fineweb-val.bin", import.meta.url).pathname
const CKPT = new URL("../data/fineweb-epoch-ckpt.safetensors", import.meta.url).pathname

const BATCH = 64
const PEAK_LR = 3e-4
const MIN_LR = 3e-5
const WARMUP_FRACTION = 0.005
const CHECKPOINT_EVERY = Number(process.env.FINEWEB_CHECKPOINT_EVERY ?? 2000)
const VAL_BATCHES = 40

const program = Effect.gen(function* () {
  const train = loadBin(TRAIN_BIN)
  const val = loadBin(VAL_BIN)
  const tokenizer = yield* loadTokenizer
  const epochSteps = Math.floor(Math.floor((train.length - 1) / BLOCK) / BATCH)
  const totalSteps = process.env.FINEWEB_STEPS === undefined ? epochSteps : Number(process.env.FINEWEB_STEPS)
  const warmupSteps = Math.max(1, Math.floor(totalSteps * WARMUP_FRACTION))

  yield* Effect.log("1) creating model")
  const model = yield* createGpt(tokenizer.vocabSize)

  const samplerConfig = { length: train.length, block: BLOCK, batch: BATCH }
  let sampler: Sampler.Sampler
  const trainer = yield* Trainer.compile(yield* Trainer.make(model, {
    optimizer: yield* Optimizer.adamW(),
    lr: LearningRate.withWarmup(
      LearningRate.cosine(PEAK_LR, { totalSteps, minLr: MIN_LR }),
      warmupSteps
    ),
    loss: Loss.crossEntropy,
    data: () =>
      Effect.gen(function* () {
        const { inputs, targets } = windows(train, sampler.next(), BATCH, BLOCK)
        return {
          input: yield* Tensor.fromTypedArray(inputs, [BATCH, BLOCK]),
          target: yield* Tensor.fromTypedArray(targets, [BATCH, BLOCK])
        }
      }),
    stop: ({ step }) => step >= chunkTarget,
    onStep: ({ step, loss, elapsed }) =>
      step % 100 === 0 || step === 1
        ? Effect.log(
          `step ${String(step).padStart(5)}/${totalSteps}  loss ${loss.toFixed(4)}  ${(Duration.toMillis(elapsed) / 1000).toFixed(1)}s`
        )
        : Effect.void
  }))

  // A saved epoch checkpoint resumes bit-exactly; otherwise warm-start
  // from the pilot's parameters with fresh optimizer state at step 0.
  let params: Model.Params
  let step = 0
  let resume: Trainer.Resume<Optimizer.AdamState> | undefined
  let epoch = 1
  if (fs.existsSync(CKPT)) {
    const checkpoint = yield* Checkpoint.loadWithSampler(CKPT, trainer)
    sampler = yield* Sampler.restore(samplerConfig, checkpoint.sampler)
    params = checkpoint.params
    resume = checkpoint.resume
    step = checkpoint.resume.step
    epoch = checkpoint.sampler.epoch
    yield* Effect.log(`resuming epoch from step ${step}`)
  } else {
    sampler = yield* Sampler.make(samplerConfig)
    params = yield* loadParams(model, CHECKPOINT)
    yield* Effect.log(`warm start from ${CHECKPOINT}`)
  }
  yield* Effect.log(
    `2) training one epoch: ${totalSteps} steps, batch ${BATCH}, warmup ${warmupSteps} then cosine ${PEAK_LR} → ${MIN_LR} (checkpoint every ${CHECKPOINT_EVERY})`
  )

  let chunkTarget = Math.min(step + CHECKPOINT_EVERY, totalSteps)
  while (step < totalSteps) {
    const trained = yield* trainer.train(params, resume)
    params = trained.params
    step = trained.step
    resume = { state: trained.state, step }
    yield* Checkpoint.saveWithSampler(CKPT, trainer, trained, sampler)
    yield* Effect.log(`checkpoint at step ${step}`)
    chunkTarget = Math.min(step + CHECKPOINT_EVERY, totalSteps)
  }

  yield* Effect.log(`3) held-out loss over ${VAL_BATCHES} val batches`)
  const valLoss = yield* heldOutLoss(model, params, val, BATCH, BLOCK, VAL_BATCHES)
  yield* Effect.log(`val loss ${valLoss.toFixed(4)}`)

  yield* Effect.log(`4) saving model to ${CHECKPOINT}`)
  yield* saveParams(model, params, CHECKPOINT)
})

NodeRuntime.runMain(program.pipe(Effect.provide(Device.Best)))

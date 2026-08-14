import * as BackendApple from "@effect-torch/backend-apple-native"
import { Checkpoint, LearningRate, Loss, Optimizer, Sampler, Tensor, Trainer } from "@effect-torch/core"
import { NodeRuntime } from "@effect/platform-node"
import { Duration, Effect } from "effect"
import fs from "node:fs"
import { BLOCK, CHECKPOINT, createGpt, heldOutLoss, loadBin, loadTokenizer, saveParams, windows } from "./model.js"

// FineWeb pilot training over the u16 token streams produced by prepare.ts.
// `CKPT` is a resumable training archive; after training, `CHECKPOINT` is a
// separate bare-parameter artifact consumed by generate.ts and other model
// loaders.
//
// Every checkpoint chunk atomically replaces one safetensors file containing
// parameters, AdamW tensor state, global step, and the sampler's current
// permutation/cursor. Resume preserves the remaining draws in that permutation,
// but JavaScript RNG state is not persisted, so later epoch reshuffles need not
// match an uninterrupted run. The archive also carries no model, optimizer,
// learning-rate, or code-version provenance; those must remain compatible.
// FINEWEB_STEPS and FINEWEB_CHECKPOINT_EVERY are parsed with Number and receive
// no explicit finite/positive-integer validation in this script.

const TRAIN_BIN = new URL("../data/fineweb-train.bin", import.meta.url).pathname
const VAL_BIN = new URL("../data/fineweb-val.bin", import.meta.url).pathname
const CKPT = new URL("../data/fineweb-ckpt.safetensors", import.meta.url).pathname

const BATCH = 64
const STEPS = Number(process.env.FINEWEB_STEPS ?? 5000)
const CHECKPOINT_EVERY = Number(process.env.FINEWEB_CHECKPOINT_EVERY ?? 100)
const LR = 6e-4
const VAL_BATCHES = 20

const program = Effect.gen(function*() {
  const train = loadBin(TRAIN_BIN)
  const val = loadBin(VAL_BIN)
  const tokenizer = yield* loadTokenizer
  yield* Effect.log(
    `fineweb-train: vocab ${tokenizer.vocabSize}, ${
      (train.length / 1e6).toFixed(0)
    }M train tokens, block ${BLOCK}, batch ${BATCH} (${Math.floor((train.length - 1) / BLOCK / BATCH)} steps per epoch)`
  )

  yield* Effect.log("1) creating model")
  const model = yield* createGpt(tokenizer.vocabSize)
  const params0 = yield* model.init
  const total = params0.reduce((sum, param) => sum + param.shape.reduce((a, b) => a * b, 1), 0)
  yield* Effect.log(`  total: ${total.toLocaleString()} parameters`)

  yield* Effect.log(`2) training: adamW lr=${LR}, ${STEPS} steps (checkpoint every ${CHECKPOINT_EVERY})`)
  let sampler: Sampler.Sampler
  const optimizer = yield* Optimizer.adamW()
  const trainer = yield* Trainer.make(model, {
    optimizer,
    lr: LearningRate.constant(LR),
    loss: Loss.crossEntropy,
    data: () =>
      Effect.gen(function*() {
        const { inputs, targets } = windows(train, sampler.next(), BATCH, BLOCK)
        return {
          input: yield* Tensor.fromTypedArray(inputs, [BATCH, BLOCK]),
          target: yield* Tensor.fromTypedArray(targets, [BATCH, BLOCK])
        }
      }),
    stop: ({ step }) => step >= chunkTarget,
    onStep: ({ step, loss, elapsed }) =>
      Effect.log(
        `step ${String(step).padStart(4)}  loss ${loss.toFixed(4)}  ${(Duration.toMillis(elapsed) / 1000).toFixed(1)}s`
      )
  })

  // A training archive takes precedence over a fresh initialization. Restore
  // validates the saved token geometry and resumes the current permutation;
  // it cannot reproduce a future reshuffle because Math.random is not saved.
  const samplerConfig = { length: train.length, block: BLOCK, batch: BATCH }
  let params = params0
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
    yield* Effect.log(`resuming from step ${step} (epoch ${epoch})`)
  } else {
    sampler = yield* Sampler.make(samplerConfig)
  }

  let chunkTarget = Math.min(step + CHECKPOINT_EVERY, STEPS)
  while (step < STEPS) {
    const previous = params
    const previousState = resume?.state
    const trained = yield* trainer.train(params, resume)
    params = trained.params
    step = trained.step
    resume = { state: trained.state, step }
    yield* Checkpoint.saveWithSampler(CKPT, trainer, trained, sampler)
    const currentEpoch = sampler.state().epoch
    if (currentEpoch !== epoch) {
      epoch = currentEpoch
      yield* Effect.log(`epoch ${epoch}`)
    }
    yield* Effect.log(`checkpoint at step ${step}`)
    chunkTarget = Math.min(step + CHECKPOINT_EVERY, STEPS)
    yield* Tensor.clearAll(
      new Set([
        ...previous.filter(Tensor.isTensor),
        ...(previousState !== undefined ? optimizer.stateRoots(previousState).filter(Tensor.isTensor) : [])
      ])
    )
  }

  yield* Effect.log(`3) held-out loss over ${VAL_BATCHES} val batches`)
  const valLoss = yield* heldOutLoss(model, params, val, BATCH, BLOCK, VAL_BATCHES)
  yield* Effect.log(`val loss ${valLoss.toFixed(4)}`)

  yield* Effect.log(`4) saving checkpoint to ${CHECKPOINT}`)
  yield* saveParams(model, params, CHECKPOINT)
})

NodeRuntime.runMain(program.pipe(Effect.provide(BackendApple.layer)))

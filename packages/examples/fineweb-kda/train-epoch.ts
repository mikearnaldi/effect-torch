import * as BackendApple from "@effect-torch/backend-apple-native"
import { Checkpoint, LearningRate, Loss, Model, Optimizer, Sampler, Tensor, Trainer } from "@effect-torch/core"
import { NodeRuntime } from "@effect/platform-node"
import { Duration, Effect } from "effect"
import fs from "node:fs"
import {
  BLOCK,
  CHECKPOINT,
  createKdaGpt,
  heldOutLoss,
  loadBin,
  loadParams,
  loadTokenizer,
  saveParams,
  windows
} from "./model.js"

// `CKPT` stores resumable training state. The script rewrites it after every chunk.
// Without it, bare `CHECKPOINT` parameters start with fresh AdamW and sampler
// state. `OUT` receives the final bare parameters. By default,
// `OUT === CHECKPOINT` and `CKPT` remains separate. Future reshuffles are not
// reproducible because checkpoints do not save Math.random state.
//
// By default, the run covers every complete batch in one permutation. The
// warmup and cosine schedule use the same step count. FINEWEB_STEPS overrides
// both counts, so training may stop before an epoch ends or continue into the
// next one. Precision defaults to mixedBf16. FINEWEB_PRECISION=f32 disables it.
// mixedBf16 casts forward parameters at the model boundary but keeps f32 master
// weights and optimizer state. It does not enable general autocast or loss
// scaling. Values other than mixedBf16 use f32 behavior. The script parses
// FINEWEB_BATCH and FINEWEB_STEPS with Number but does not validate them early.
// An existing `CKPT` takes precedence on the next run.

const TRAIN_BIN = new URL("../data/fineweb-train.bin", import.meta.url).pathname
const VAL_BIN = new URL("../data/fineweb-val.bin", import.meta.url).pathname
const CKPT = process.env.FINEWEB_CKPT ?? new URL("../data/fineweb-kda-epoch-ckpt.safetensors", import.meta.url).pathname
const OUT = process.env.FINEWEB_OUT ?? CHECKPOINT

const BATCH = Number(process.env.FINEWEB_BATCH ?? 64)
const PEAK_LR = 3e-4
const MIN_LR = 3e-5
const WARMUP_FRACTION = 0.005
const CHECKPOINT_EVERY = 100
const PRECISION: Trainer.Precision = process.env.FINEWEB_PRECISION === undefined ||
    process.env.FINEWEB_PRECISION === "mixedBf16"
  ? "mixedBf16"
  : "f32"
const VAL_BATCHES = 40

const formatEta = (ms: number) => {
  const min = Math.round(ms / 60000)
  return min >= 60 ? `${Math.floor(min / 60)}h${String(min % 60).padStart(2, "0")}m` : `${min}m`
}

let firstStep: { step: number; ms: number } | undefined

const program = Effect.gen(function*() {
  const train = loadBin(TRAIN_BIN)
  const val = loadBin(VAL_BIN)
  const tokenizer = yield* loadTokenizer
  const epochSteps = Math.floor(Math.floor((train.length - 1) / BLOCK) / BATCH)
  const totalSteps = process.env.FINEWEB_STEPS === undefined ? epochSteps : Number(process.env.FINEWEB_STEPS)
  const warmupSteps = Math.max(1, Math.floor(totalSteps * WARMUP_FRACTION))

  yield* Effect.log("1) creating model (hybrid KDA, NoPE attention layers)")
  const model = yield* createKdaGpt(tokenizer.vocabSize)

  const samplerConfig = { length: train.length, block: BLOCK, batch: BATCH }
  let sampler: Sampler.Sampler
  const optimizer = yield* Optimizer.adamW()
  const trainer = yield* Trainer.make(model, {
    optimizer,
    lr: LearningRate.withWarmup(
      LearningRate.cosine(PEAK_LR, { totalSteps, minLr: MIN_LR }),
      warmupSteps
    ),
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
    precision: PRECISION,
    onStep: ({ step, loss, elapsed }) => {
      // Estimate ETA from the mean elapsed time per step. Exclude the first
      // observed step because it includes compilation. Resume.startedAt keeps
      // elapsed time continuous across checkpoint chunks in this process.
      const ms = Duration.toMillis(elapsed)
      if (firstStep === undefined) firstStep = { step, ms }
      const done = step - firstStep.step
      const remaining = totalSteps - step
      const eta = done > 0 && remaining > 0
        ? `  eta ${formatEta(((ms - firstStep.ms) / done) * remaining)}`
        : ""
      return Effect.log(
        `step ${String(step).padStart(5)}/${totalSteps}  loss ${loss.toFixed(4)}  ${(ms / 1000).toFixed(1)}s${eta}`
      )
    }
  })

  // Use the resumable archive before the bare pilot artifact. Loading bare
  // parameters starts fresh optimizer and sampler state. If neither file
  // exists, initialize random model parameters.
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
    yield* Effect.log(`resuming epoch ${epoch} from step ${step}`)
  } else if (fs.existsSync(CHECKPOINT)) {
    sampler = yield* Sampler.make(samplerConfig)
    params = yield* loadParams(model, CHECKPOINT)
    yield* Effect.log(`warm start from ${CHECKPOINT}`)
  } else {
    sampler = yield* Sampler.make(samplerConfig)
    params = yield* Model.initialize(model)
    yield* Effect.log(`cold start from random init (no checkpoint at ${CHECKPOINT})`)
  }
  yield* Effect.log(
    `2) training one epoch: ${totalSteps} steps, batch ${BATCH}, warmup ${warmupSteps} then cosine ${PEAK_LR} → ${MIN_LR} (checkpoint every ${CHECKPOINT_EVERY})`
  )

  let chunkTarget = Math.min(step + CHECKPOINT_EVERY, totalSteps)
  // Reuse one elapsed-time origin across chunks in this process. Checkpoints
  // omit `startedAt`, so restarting the process resets the ETA clock.
  const epochStartedAt = resume?.startedAt ?? Date.now()
  if (resume !== undefined) resume = { ...resume, startedAt: epochStartedAt }
  while (step < totalSteps) {
    const previous = params
    const previousState = resume?.state
    const trained = yield* trainer.train(params, resume)
    params = trained.params
    step = trained.step
    resume = { state: trained.state, step, startedAt: epochStartedAt }
    yield* Checkpoint.saveWithSampler(CKPT, trainer, trained, sampler)
    yield* Effect.log(`checkpoint at step ${step}`)
    chunkTarget = Math.min(step + CHECKPOINT_EVERY, totalSteps)
    // The caller retains ownership of concrete parameters and state passed to a
    // chunk. Clear stale roots once after the trainer returns replacements
    // instead of waiting for native finalizers. Skip lazy roots from a cold
    // initialization.
    const stale = new Set([
      ...previous.filter(Tensor.isTensor),
      ...(previousState !== undefined ? optimizer.stateRoots(previousState).filter(Tensor.isTensor) : [])
    ])
    yield* Tensor.clearAll(stale)
  }

  yield* Effect.log(`3) held-out loss over ${VAL_BATCHES} val batches`)
  const valLoss = yield* heldOutLoss(model, params, val, BATCH, BLOCK, VAL_BATCHES)
  yield* Effect.log(`val loss ${valLoss.toFixed(4)}`)

  yield* Effect.log(`4) saving model to ${OUT}`)
  yield* saveParams(model, params, OUT)
})

NodeRuntime.runMain(program.pipe(Effect.provide(BackendApple.layer())))

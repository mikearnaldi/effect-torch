import * as BackendApple from "@effect-torch/backend-apple-native"
import type { Model } from "@effect-torch/core"
import { Checkpoint, LearningRate, Loss, Optimizer, Sampler, Tensor, Trainer } from "@effect-torch/core"
import { NodeRuntime } from "@effect/platform-node"
import { Duration, Effect } from "effect"
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

// Longer FineWeb training uses `CKPT` as a mutable resumable archive, rewritten
// after every chunk and preferred when present. Without it, bare `CHECKPOINT`
// parameters warm-start training with fresh AdamW and sampler state. `OUT`
// receives final bare parameters; by default `OUT === CHECKPOINT`, so the final
// save replaces the warm-start artifact while `CKPT` remains resumable state.
//
// By default `totalSteps` is every complete sampler batch in one permutation;
// a trailing partial batch is skipped. FINEWEB_STEPS changes both the run and
// cosine-schedule horizon, so an override is not necessarily one epoch. Resume
// restores parameters, optimizer roots, step, and the remainder of the saved
// permutation. Future reshuffles can diverge because Math.random state is not
// checkpointed, and checkpoint files carry no hyperparameter provenance.
// FINEWEB_BATCH and FINEWEB_STEPS are Number-parsed without early validation.
// A completed `CKPT` remains present and wins on the next invocation; select or
// remove it before intending a fresh warm start.

const TRAIN_BIN = new URL("../data/fineweb-train.bin", import.meta.url).pathname
const VAL_BIN = new URL("../data/fineweb-val.bin", import.meta.url).pathname
const CKPT = process.env.FINEWEB_CKPT ?? new URL("../data/fineweb-epoch-ckpt.safetensors", import.meta.url).pathname
const OUT = process.env.FINEWEB_OUT ?? CHECKPOINT

const BATCH = Number(process.env.FINEWEB_BATCH ?? 64)
const PEAK_LR = 3e-4
const MIN_LR = 3e-5
const WARMUP_FRACTION = 0.005
const CHECKPOINT_EVERY = 100
const PRECISION: Trainer.Precision = "mixedBf16"
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

  yield* Effect.log("1) creating model")
  const model = yield* createGpt(tokenizer.vocabSize)

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
      // ETA from the mean step time since the epoch began, excluding
      // the first step seen (its compile overhead would skew
      // the average); the clock is continuous across checkpoint chunks
      // via Resume.startedAt.
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

  // The resumable archive wins over the pilot artifact. Without it, loading
  // bare pilot parameters deliberately creates fresh optimizer and sampler
  // state; random model initialization is only the final fallback.
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
    params = yield* model.init
    yield* Effect.log(`cold start from random init (no checkpoint at ${CHECKPOINT})`)
  }
  yield* Effect.log(
    `2) training one epoch: ${totalSteps} steps, batch ${BATCH}, warmup ${warmupSteps} then cosine ${PEAK_LR} → ${MIN_LR} (checkpoint every ${CHECKPOINT_EVERY})`
  )

  let chunkTarget = Math.min(step + CHECKPOINT_EVERY, totalSteps)
  // One elapsed clock is carried across chunks in this process. Checkpoint
  // persistence omits `startedAt`, so restarting the process starts a new
  // elapsed/ETA clock even though the global training step resumes.
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
    // Each chunk passes concrete params/state as caller-owned inputs, so the
    // trainer does not release that generation after producing replacements.
    // Clear its concrete roots exactly once rather than leaving GB-scale
    // generations to native finalizers; lazy initial roots are skipped.
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

NodeRuntime.runMain(program.pipe(Effect.provide(BackendApple.layer)))

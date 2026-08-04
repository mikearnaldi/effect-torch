import { Duration, Effect } from "effect"
import { NodeRuntime } from "@effect/platform-node"
import { Device, LearningRate, Loss, Optimizer, Tensor, Trainer } from "@effect-torch/core"
import fs from "node:fs"
import { BLOCK, CHECKPOINT, createGpt, loadTokenizer, saveParams } from "./model.js"

// FineWeb pre-training: trains the shared GPT on the token bins produced
// by prepare.ts (~745M GPT-2 BPE tokens, u16), reports a held-out
// loss estimate, and saves the trained parameters to a safetensors
// checkpoint for infer.ts.

const TRAIN_BIN = new URL("../data/fineweb-train.bin", import.meta.url).pathname
const VAL_BIN = new URL("../data/fineweb-val.bin", import.meta.url).pathname
const CKPT_BIN = new URL("../data/fineweb-ckpt.safetensors", import.meta.url).pathname
const CKPT_META = new URL("../data/fineweb-ckpt.json", import.meta.url).pathname
const CKPT_ORDER = new URL("../data/fineweb-ckpt-order.bin", import.meta.url).pathname

const BATCH = 64
const STEPS = Number(process.env.FINEWEB_STEPS ?? 5000)
const CHECKPOINT_EVERY = Number(process.env.FINEWEB_CHECKPOINT_EVERY ?? 1000)
const LR = 6e-4
const VAL_BATCHES = 20

const loadBin = (path: string) => {
  const buffer = fs.readFileSync(path)
  if (buffer.byteOffset % 2 !== 0) throw new Error("misaligned token bin buffer")
  return new Uint16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2)
}

// Epoch-based batching: all non-overlapping BLOCK-windows in a shuffled
// permutation, so every window is seen exactly once per epoch (no
// replacement); the permutation is reshuffled at each epoch boundary.
// The full sampler state (permutation, cursor, epoch) is restorable, so
// a checkpoint resumes the data layout exactly where it stopped.
export interface SamplerState {
  readonly order: Uint32Array
  readonly cursor: number
  readonly epoch: number
}

const shuffle = (order: Uint32Array) => {
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    const t = order[i]
    order[i] = order[j]
    order[j] = t
  }
}

const makeTrainSampler = (
  data: Uint16Array,
  onEpoch: (epoch: number) => void,
  restored?: SamplerState
) => {
  const windowCount = Math.floor((data.length - 1) / BLOCK)
  let order: Uint32Array
  let cursor: number
  let epoch: number
  if (restored !== undefined && restored.order.length === windowCount) {
    order = restored.order
    cursor = restored.cursor
    epoch = restored.epoch
  } else {
    order = new Uint32Array(windowCount)
    for (let i = 0; i < windowCount; i++) order[i] = i
    shuffle(order)
    cursor = 0
    epoch = 1
  }
  const next = () => {
    if (cursor + BATCH > windowCount) {
      shuffle(order)
      cursor = 0
      epoch += 1
      onEpoch(epoch)
    }
    const inputs = new Uint32Array(BATCH * BLOCK)
    const targets = new Uint32Array(BATCH * BLOCK)
    for (let b = 0; b < BATCH; b++) {
      const start = order[cursor + b] * BLOCK
      for (let t = 0; t < BLOCK; t++) {
        inputs[b * BLOCK + t] = data[start + t]
        targets[b * BLOCK + t] = data[start + t + 1]
      }
    }
    cursor += BATCH
    return { inputs, targets }
  }
  const state = (): SamplerState => ({ order, cursor, epoch })
  return { next, state }
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
    `fineweb-train: vocab ${tokenizer.vocabSize}, ${(train.length / 1e6).toFixed(0)}M train tokens, block ${BLOCK}, batch ${BATCH} (${Math.floor((train.length - 1) / BLOCK / BATCH)} steps per epoch)`
  )

  yield* Effect.log("1) creating model")
  const model = yield* createGpt(tokenizer.vocabSize)
  const params0 = yield* model.init
  const total = params0.reduce((sum, param) => sum + param.shape.reduce((a, b) => a * b, 1), 0)
  yield* Effect.log(`  total: ${total.toLocaleString()} parameters`)

  yield* Effect.log(`2) training: adamW lr=${LR}, ${STEPS} steps (checkpoint every ${CHECKPOINT_EVERY})`)
  // A checkpoint holds the trainer (params + optimizer state + step) AND
  // the data layout (permutation + cursor + epoch), so resuming
  // continues the epoch exactly where it stopped.
  const meta = fs.existsSync(CKPT_BIN) && fs.existsSync(CKPT_META) && fs.existsSync(CKPT_ORDER)
    ? JSON.parse(fs.readFileSync(CKPT_META, "utf8"))
    : undefined
  const restored = meta === undefined ? undefined : {
    order: new Uint32Array(fs.readFileSync(CKPT_ORDER).buffer.slice(0)),
    cursor: meta.cursor as number,
    epoch: meta.epoch as number
  }
  const sampler = makeTrainSampler(train, (epoch) => console.log(`epoch ${epoch}`), restored)
  const optimizer = yield* Optimizer.adamW()
  const trainer = yield* Trainer.compile(yield* Trainer.make(model, {
    optimizer,
    lr: LearningRate.constant(LR),
    loss: Loss.crossEntropy,
    data: () =>
      Effect.gen(function* () {
        const { inputs, targets } = sampler.next()
        return {
          input: yield* Tensor.fromTypedArray(inputs, [BATCH, BLOCK]),
          target: yield* Tensor.fromTypedArray(targets, [BATCH, BLOCK])
        }
      }),
    stop: ({ step }) => step >= chunkTarget,
    onStep: ({ step, loss, elapsed }) =>
      step % 50 === 0 || step === 1
        ? Effect.log(
          `step ${String(step).padStart(4)}  loss ${loss.toFixed(4)}  ${(Duration.toMillis(elapsed) / 1000).toFixed(1)}s`
        )
        : Effect.void
  }))

  // Resume when a checkpoint exists: params by name, optimizer state by
  // state-roots order, global step from the meta file. 0-d state roots
  // (AdamW's step count) live in the meta as numbers — the optimizer's
  // own f64 CPU encoding can't ride the safetensors onto the device —
  // and are rebuilt as f32 on the ambient device (AdamW casts the count
  // per use, so its storage dtype is immaterial).
  let params = params0
  let step = 0
  let resume: Trainer.Resume<Optimizer.AdamState> | undefined
  if (meta !== undefined) {
    const tensors = yield* Tensor.load(CKPT_BIN)
    params = model.names.map((name) => tensors[`param:${name}`])
    const fresh = yield* optimizer.init(params)
    const roots: Array<Tensor.Any> = []
    for (const [i, root] of optimizer.stateRoots(fresh).entries()) {
      roots.push(
        root.shape.length === 0
          ? yield* Tensor.full([], meta.scalars[i], { dtype: "f32" })
          : tensors[`state:${i}`]
      )
    }
    step = meta.step
    resume = { state: optimizer.rebuildState(fresh, roots), step }
    yield* Effect.log(`resuming from step ${step}`)
  }

  let chunkTarget = Math.min(step + CHECKPOINT_EVERY, STEPS)
  while (step < STEPS) {
    const trained = yield* trainer.train(params, resume)
    params = trained.params
    step = trained.step
    resume = { state: trained.state, step }
    const tensors: Record<string, Tensor.Any> = Object.fromEntries(
      model.names.map((name, i) => [`param:${name}`, params[i]])
    )
    const scalars: Record<number, number> = {}
    for (const [i, root] of optimizer.stateRoots(trained.state).entries()) {
      if (root.shape.length === 0) {
        const [materialized] = yield* Tensor.compute([root])
        const [value] = yield* Tensor.toNumberArray(materialized)
        scalars[i] = value
      } else {
        tensors[`state:${i}`] = root
      }
    }
    yield* Tensor.save(CKPT_BIN, tensors)
    const samplerState = sampler.state()
    fs.writeFileSync(CKPT_ORDER, Buffer.from(samplerState.order.buffer, samplerState.order.byteOffset, samplerState.order.byteLength))
    fs.writeFileSync(
      CKPT_META,
      JSON.stringify({ step, scalars, cursor: samplerState.cursor, epoch: samplerState.epoch })
    )
    yield* Effect.log(`checkpoint at step ${step}`)
    chunkTarget = Math.min(step + CHECKPOINT_EVERY, STEPS)
  }

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

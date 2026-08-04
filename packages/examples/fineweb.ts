import { Duration, Effect, Option } from "effect"
import { NodeRuntime } from "@effect/platform-node"
import { Device, LearningRate, Loss, Model, Optimizer, Tensor, Tokenizer, Trainer } from "@effect-torch/core"
import fs from "node:fs"

// Real-world pre-training pilot: a GPT-2-architecture model (pre-norm
// blocks, RoPE causal attention, tied-depth head) trained on FineWeb-Edu
// tokens prepared by fineweb-prepare.ts — ~750M GPT-2 BPE tokens in
// data/fineweb-train.bin (u16). Same combinators as nano-gpt, real data
// and a real tokenizer. After training: a held-out loss estimate and
// sampled generations through the compiled inference artifact.

const TRAIN_BIN = new URL("./data/fineweb-train.bin", import.meta.url).pathname
const VAL_BIN = new URL("./data/fineweb-val.bin", import.meta.url).pathname
const TOKENIZER_JSON = new URL("./data/gpt2-tokenizer.json", import.meta.url).pathname
const EOT = "<|endoftext|>"

const BLOCK = 256
const EMBED = 256
const HEADS = 4
const LAYERS = 6
const BATCH = 32
const STEPS = 5000
const LR = 6e-4
const TEMPERATURE = 0.8
const VAL_BATCHES = 20

const createGpt = (vocabSize: number) =>
  Effect.gen(function* () {
    // Token embeddings; positions are relative (RoPE inside attention),
    // so generation is unbounded — no position table to outgrow.
    const embeddings = yield* Model.embedding("wte", vocabSize, EMBED)
    const blocks: Array<Model.Model> = []
    for (let i = 0; i < LAYERS; i++) {
      const attn = yield* Model.chain(
        yield* Model.layerNorm(`b${i}.ln1`, EMBED),
        yield* Model.multiHeadAttention(`b${i}.attn`, EMBED, HEADS, { causal: true, rope: 10000 })
      )
      const mlp = yield* Model.chain(
        yield* Model.layerNorm(`b${i}.ln2`, EMBED),
        yield* Model.linear(`b${i}.fc`, EMBED, 4 * EMBED),
        yield* Model.gelu(),
        yield* Model.linear(`b${i}.proj`, 4 * EMBED, EMBED)
      )
      blocks.push(yield* Model.chain(yield* Model.residual(attn), yield* Model.residual(mlp)))
    }
    const model = yield* Model.chain(
      embeddings,
      ...blocks,
      yield* Model.layerNorm("lnf", EMBED),
      yield* Model.linear("head", EMBED, vocabSize)
    )
    return yield* Model.compile(model)
  })

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

const init = (model: Model.Model) =>
  Effect.gen(function* () {
    const params = yield* model.init
    const total = params.reduce((sum, param) => sum + param.shape.reduce((a, b) => a * b, 1), 0)
    yield* Effect.log(`  total: ${total.toLocaleString()} parameters`)
    return params
  })

const loadBin = (path: string) => {
  const buffer = fs.readFileSync(path)
  if (buffer.byteOffset % 2 !== 0) throw new Error("misaligned token bin buffer")
  return new Uint16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2)
}

const program = Effect.gen(function* () {
  const train = loadBin(TRAIN_BIN)
  const val = loadBin(VAL_BIN)
  const tokenizer = yield* Tokenizer.fromFile(TOKENIZER_JSON, {
    padding: Tokenizer.paddingNone,
    truncation: Tokenizer.truncationNone,
    specialTokens: "Always"
  })
  const eotId = Option.getOrThrow(tokenizer.tokenToId(EOT))
  const vocabSize = tokenizer.vocabSize
  yield* Effect.log(
    `fineweb: vocab ${vocabSize}, ${(train.length / 1e6).toFixed(0)}M train tokens, block ${BLOCK}, embed ${EMBED}, ${HEADS} heads, ${LAYERS} layers, batch ${BATCH} (${(BATCH * BLOCK * STEPS / 1e6).toFixed(0)}M tokens over ${STEPS} steps)`
  )

  yield* Effect.log("1) creating model")
  const model = yield* createGpt(vocabSize)
  const params0 = yield* init(model)

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

  yield* Effect.log(`4) generating from prompts (temperature ${TEMPERATURE}), stopping at ${EOT}:`)
  const inference = yield* Model.inference(model, params, {
    maxTokens: 8192,
    blockSize: 16,
    attentionWindow: BLOCK
  })
  const prompts = [
    "The history of the printing press begins",
    "In a small village by the sea,",
    "Scientists have discovered that"
  ]
  for (const prompt of prompts) {
    const generated: Array<number> = []
    const gen = yield* inference.generation()
    const entry = yield* gen.add(yield* tokenizer.encode(prompt))
    let logits = entry.logits
    for (let i = 0; i < 240; i++) {
      const [probs] = yield* Tensor.compute([yield* Tensor.softmax(logits)])
      const weights = yield* Tensor.toNumberArray(probs)
      const token = sampleCategorical(weights, TEMPERATURE)
      if (token === eotId) break
      generated.push(token)
      const [stepped] = yield* gen.step([{ seq: entry.seq, token }])
      logits = stepped
    }
    yield* gen.close()
    const text = yield* tokenizer.decode(generated)
    yield* Effect.log(`\n--- prompt: ${prompt}\n${prompt}${text}\n`)
  }
})

// Multinomial sampling with temperature over a probability vector.
const sampleCategorical = (probs: ReadonlyArray<number>, temperature: number) => {
  let sum = 0
  const scaled = probs.map((p) => {
    const v = Math.pow(Math.max(p, 1e-12), 1 / temperature)
    sum += v
    return v
  })
  let r = Math.random() * sum
  for (let i = 0; i < scaled.length; i++) {
    r -= scaled[i]
    if (r <= 0) return i
  }
  return scaled.length - 1
}

NodeRuntime.runMain(program.pipe(Effect.provide(Device.Best)))

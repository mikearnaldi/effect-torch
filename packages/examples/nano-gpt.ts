import * as BackendApple from "@effect-torch/backend-apple-native"
import { LearningRate, Loss, Model, Optimizer, Runtime, Tensor, Trainer } from "@effect-torch/core"
import * as Tokenizer from "@effect-torch/tokenizers"
import { NodeRuntime } from "@effect/platform-node"
import { Duration, Effect, Option } from "effect"

// Trains a Unigram tokenizer and pre-norm RoPE transformer on a small in-memory
// corpus, then freezes the learned parameters for compiled prefill and decode.
// Inference uses a 4,096-row paged pool and caches at most BLOCK positions per
// attention layer, so the logical cursor can advance past pool capacity until
// EOS.

const CORPUS = `Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance or nature's changing course untrimm'd;
But thy eternal summer shall not fade
Nor lose possession of that fair thou owest;
Nor shall Death brag thou wander'st in his shade,
When in eternal lines to time thou growest:
So long as men can breathe or eyes can see,
So long lives this and this gives life to thee.

To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them? To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life.

'Twas brillig, and the slithy toves
Did gyre and gimble in the wabe:
All mimsy were the borogoves,
And the mome raths outgrabe.
Beware the Jabberwock, my son!
The jaws that bite, the claws that catch!
Beware the Jubjub bird, and shun
The frumious Bandersnatch!
He took his vorpal sword in hand;
Long time the manxome foe he sought
So rested he by the Tumtum tree
And stood awhile in thought.
And, as in uffish thought he stood,
The Jabberwock, with eyes of flame,
Came whiffling through the tulgey wood,
And burbled as it came!
One, two! One, two! And through and through
The vorpal blade went snicker-snack!
He left it dead, and with its head
He went galumphing back.
And hast thou slain the Jabberwock?
Come to my arms, my beamish boy!
O frabjous day! Callooh! Callay!
He chortled in his joy.
`

const BLOCK = 32
const EMBED = 64
const HEADS = 4
const LAYERS = 2
const BATCH = 16
const STEPS = 400
const LR = 3e-3
const TEMPERATURE = 0.8
const VOCAB = 300
const TOKENIZER_MODEL: Tokenizer.TrainModel = "Unigram"
const BOS = "<|bos|>"
const EOS = "<|eos|>"

// BOS/EOS mark the three documents before their ids are concatenated. Random
// training windows can cross an EOS/BOS join; generation treats EOS as the
// stopping token rather than imposing a separate token limit.
const DOCUMENTS = CORPUS.split("\n\n").map((poem) => poem.trim() + "\n")

const createGpt = (vocabSize: number) =>
  Effect.gen(function*() {
    // Token embeddings; RoPE inside attention means the architecture has no
    // learned position table to outgrow during windowed inference.
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
    return model
  })

const ids = (values: ReadonlyArray<number>, shape: ReadonlyArray<number>) =>
  Tensor.fromTypedArray(new Uint32Array(values), shape)

const sampleBatch = (data: ReadonlyArray<number>) =>
  Effect.gen(function*() {
    const inputs: Array<number> = []
    const targets: Array<number> = []
    for (let b = 0; b < BATCH; b++) {
      const start = Math.floor(Math.random() * (data.length - BLOCK - 1))
      for (let t = 0; t < BLOCK; t++) {
        inputs.push(data[start + t])
        targets.push(data[start + t + 1])
      }
    }
    return {
      input: yield* ids(inputs, [BATCH, BLOCK]),
      target: yield* ids(targets, [BATCH, BLOCK])
    }
  })

// Create the trainer for the model, compiled. The batch shape is fixed,
// so the whole run is served by one frozen step program; the first step
// pays the trace.
const createTrainer = (model: Model.Model, data: ReadonlyArray<number>) =>
  Effect.gen(function*() {
    const trainer = yield* Trainer.make(model, {
      optimizer: yield* Optimizer.adamW(),
      lr: LearningRate.constant(LR),
      loss: Loss.crossEntropy,
      data: () => sampleBatch(data),
      stop: ({ step }) => step >= STEPS,
      onStep: ({ step, loss, elapsed }) =>
        step % 25 === 0 || step === 1
          ? Effect.log(
            `step ${String(step).padStart(4)}  loss ${loss.toFixed(4)}  ${
              (Duration.toMillis(elapsed) / 1000).toFixed(1)
            }s`
          )
          : Effect.void
    })
    return trainer
  })

const init = (model: Model.Model) =>
  Effect.gen(function*() {
    const params = yield* Model.initialize(model)
    for (const [i, { name }] of model.parameterSpecs.entries()) {
      yield* Effect.log(`  ${name} [${params[i].shape}] ${params[i].dtype} initialized`)
    }
    const total = params.reduce((sum, param) => sum + param.shape.reduce((a, b) => a * b, 1), 0)
    yield* Effect.log(`  total: ${total.toLocaleString()} parameters`)
    return params
  })

// Generates through one session of the compiled artifact: prompt prefill fills
// sequence blocks, then host sampling reads back and releases one logits row per
// step. The sliding window caps attention at the model's maximum training span,
// and EOS is the only application-level termination condition.
const generate = (
  program: Model.InferenceProgram,
  tokenizer: Tokenizer.Tokenizer,
  prompt: string
) =>
  Effect.gen(function*() {
    const bosId = Option.getOrThrow(tokenizer.tokenToId(BOS))
    const eosId = Option.getOrThrow(tokenizer.tokenToId(EOS))
    const promptIds = Array.from((yield* tokenizer.encode(prompt)).data)
    const gen = yield* program.execution()
    const sample = (logits: Tensor.Any) =>
      Effect.gen(function*() {
        const row = yield* Tensor.toNumberArray(logits)
        if (Tensor.isTensor(logits)) yield* Tensor.clear(logits)
        const max = Math.max(...row)
        const exps = row.map((x) => Math.exp((x - max) / TEMPERATURE))
        const total = exps.reduce((a, b) => a + b, 0)
        let draw = Math.random() * total
        for (let i = 0; i < exps.length; i++) {
          draw -= exps[i]
          if (draw <= 0) return i
        }
        return exps.length - 1
      })
    const entry = (yield* gen.add([yield* ids([bosId, ...promptIds], [1, 1 + promptIds.length])]))[0]!
    let logits = entry.logits
    const generated: Array<number> = []
    for (;;) {
      const next = yield* sample(logits)
      if (next === eosId) break
      generated.push(next)
      const [nextLogits] = yield* gen.step([{ seq: entry.seq, token: next }])
      logits = nextLogits
    }
    yield* gen.close()
    return yield* tokenizer.decode(generated)
  })

const program = Effect.gen(function*() {
  const runtime = yield* Runtime.Runtime

  yield* Effect.log(`0) training ${TOKENIZER_MODEL} tokenizer (target vocab ${VOCAB})`)
  const tokenizer = yield* Tokenizer.train(
    {
      source: Tokenizer.trainTexts(DOCUMENTS.flatMap((poem) => poem.split(/(?<=\n)/))),
      model: TOKENIZER_MODEL,
      vocabSize: VOCAB,
      minFrequency: 2,
      specialTokens: [BOS, EOS],
      progress: Tokenizer.trainProgressReport(
        256,
        (processed, total) => Effect.log(`tokenizer feed ${processed}/${total}`)
      )
    },
    Tokenizer.strictConfig
  )
  const vocabSize = tokenizer.vocabSize
  const bosId = Option.getOrThrow(tokenizer.tokenToId(BOS))
  const eosId = Option.getOrThrow(tokenizer.tokenToId(EOS))
  const data: Array<number> = []
  for (const poem of DOCUMENTS) {
    data.push(bosId, ...(yield* tokenizer.encode(poem)).data, eosId)
  }
  yield* Effect.log(
    `nano-gpt: vocab ${vocabSize} (${data.length} tokens in ${DOCUMENTS.length} documents), block ${BLOCK}, embed ${EMBED}, ${HEADS} heads, ${LAYERS} layers on ${runtime.placement.description}`
  )

  yield* Effect.log("1) creating model")
  const model = yield* createGpt(vocabSize)
  yield* Effect.log(`${model.parameterSpecs.length} tensors of parameters`)
  const params0 = yield* init(model)

  yield* Effect.log(`2) training: adamW lr=${LR}, ${STEPS} steps (compiled)`)
  const trainer = yield* createTrainer(model, data)
  const trained = yield* trainer.train(params0)
  const params = trained.params

  yield* Effect.log(`3) generating from prompts (temperature ${TEMPERATURE}), stopping at EOS:`)
  const inference = yield* Model.inference(model, params, {
    maxTokens: 4096,
    blockSize: 16,
    prefillChunks: [16],
    attentionWindow: BLOCK
  })
  const prompts = [
    "Shall I compare thee",
    "To be, or not to be",
    "Beware the Jabberwock"
  ]
  for (const prompt of prompts) {
    const text = yield* generate(inference, tokenizer, prompt)
    yield* Effect.log(`prompt: ${prompt}`)
    yield* Effect.log(`answer:\n${text}`)
  }
})

NodeRuntime.runMain(program.pipe(Effect.provide(BackendApple.layer())))

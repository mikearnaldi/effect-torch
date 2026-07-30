import { Effect } from "effect"
import { NodeRuntime } from "@effect/platform-node"
import { Device, Loss, Model, Optimizer, Tensor } from "@effect-torch/core"

// A character-level GPT trained on a few KB of public-domain verse:
// token and position embeddings, pre-norm transformer blocks (causal
// multi-head attention + MLP, each under a residual connection), a final
// layer norm and a vocabulary head. Everything is the stock Model
// combinators; attention is the fused flash kernel on Metal.

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
const GENERATE = 240
const TEMPERATURE = 0.8

const chars = [...new Set(CORPUS)].sort()
const vocabSize = chars.length
const encode = (text: string): Array<number> => text.split("").map((c) => chars.indexOf(c))
const decode = (ids: ReadonlyArray<number>): string => ids.map((i) => chars[i]).join("")
const data = encode(CORPUS)

interface Gpt {
  readonly model: Model.Model
  readonly arities: ReadonlyArray<number>
}

// The GPT topology is not a chain (embeddings add, logits head off a
// sequence), so the model slices the parameter array by hand and
// delegates to the sub-models' forwards.
const gpt = (parts: {
  readonly wte: Model.Model
  readonly wpe: Model.Model
  readonly blocks: ReadonlyArray<Model.Model>
  readonly lnF: Model.Model
  readonly head: Model.Model
}): Gpt => {
  const subModels = [parts.wte, parts.wpe, ...parts.blocks, parts.lnF, parts.head]
  const names = subModels.flatMap((m) => m.names)
  const arities = subModels.map((m) => m.names.length)
  const model: Model.Model = {
    names,
    init: Effect.gen(function* () {
      const params: Array<Tensor.Any> = []
      for (const m of subModels) {
        params.push(...(yield* m.init))
      }
      return params
    }),
    forward: (params, idx) =>
      Effect.gen(function* () {
        if (params.length !== names.length) {
          return yield* new Model.ModelError({
            op: "forward",
            message: `gpt: expected ${names.length} parameters, got ${params.length}`
          })
        }
        const slices: Array<Model.Params> = []
        let offset = 0
        for (const arity of arities) {
          slices.push(params.slice(offset, offset + arity))
          offset += arity
        }
        const t = idx.shape[idx.shape.length - 1]
        const tok = yield* parts.wte.forward(slices[0], idx)
        const pos = yield* parts.wpe.forward(
          slices[1],
          yield* Tensor.arange(t, undefined, { dtype: "i64" })
        )
        let x = yield* Tensor.add(tok, pos)
        for (let i = 0; i < parts.blocks.length; i++) {
          x = yield* parts.blocks[i].forward(slices[2 + i], x)
        }
        x = yield* parts.lnF.forward(slices[2 + parts.blocks.length], x)
        return yield* parts.head.forward(slices[slices.length - 1], x)
      })
  }
  return { model, arities }
}

const createGpt = Effect.gen(function* () {
  const wte = yield* Model.embedding("wte", vocabSize, EMBED)
  const wpe = yield* Model.embedding("wpe", BLOCK, EMBED)
  const blocks: Array<Model.Model> = []
  for (let i = 0; i < LAYERS; i++) {
    const attn = yield* Model.chain(
      yield* Model.layerNorm(`b${i}.ln1`, EMBED),
      yield* Model.multiHeadAttention(`b${i}.attn`, EMBED, HEADS, { causal: true })
    )
    const mlp = yield* Model.chain(
      yield* Model.layerNorm(`b${i}.ln2`, EMBED),
      yield* Model.linear(`b${i}.fc`, EMBED, 4 * EMBED),
      yield* Model.gelu(),
      yield* Model.linear(`b${i}.proj`, 4 * EMBED, EMBED)
    )
    blocks.push(yield* Model.chain(yield* Model.residual(attn), yield* Model.residual(mlp)))
  }
  const lnF = yield* Model.layerNorm("lnf", EMBED)
  const head = yield* Model.linear("head", EMBED, vocabSize)
  return gpt({ wte, wpe, blocks, lnF, head })
})

const ids = (values: ReadonlyArray<number>, shape: ReadonlyArray<number>) =>
  Tensor.fromTypedArray(new BigInt64Array(values.map(BigInt)), shape)

const sampleBatch = Effect.gen(function* () {
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
    target: yield* ids(targets, [BATCH * BLOCK])
  }
})

const program = Effect.gen(function* () {
  const device = yield* Device.CurrentDevice
  yield* Effect.log(
    `nano-gpt: vocab ${vocabSize}, block ${BLOCK}, embed ${EMBED}, ${HEADS} heads, ${LAYERS} layers on ${device}`
  )
  const { model } = yield* createGpt
  yield* Effect.log(`${model.names.length} tensors of parameters`)

  const optimizer = Optimizer.adamW({ lr: LR })
  let params: Model.Params = yield* model.init
  let state = yield* optimizer.init(params)
  const started = Date.now()
  for (let step = 1; step <= STEPS; step++) {
    const batch = yield* sampleBatch
    const flat = yield* model.forward(params, batch.input)
    const logits = yield* Tensor.reshape(flat, [BATCH * BLOCK, vocabSize])
    const loss = yield* Loss.crossEntropy(logits, batch.target)
    const result = yield* Optimizer.step(optimizer, loss, params, state)
    params = result.params
    state = result.state
    if (step % 25 === 0 || step === 1) {
      const [value] = yield* Tensor.toNumberArray(result.loss)
      const elapsed = ((Date.now() - started) / 1000).toFixed(1)
      yield* Effect.log(`step ${String(step).padStart(4)}  loss ${value.toFixed(4)}  ${elapsed}s`)
    }
  }

  // Greedy-windowed sampling with temperature: re-run the model on the
  // last BLOCK tokens and draw the next character from the final
  // position's softmax.
  yield* Effect.log(`generating ${GENERATE} characters (temperature ${TEMPERATURE}):`)
  let context = encode("\n")
  let generated = ""
  for (let n = 0; n < GENERATE; n++) {
    const window = context.slice(-BLOCK)
    const idx = yield* ids(window, [1, window.length])
    const [logits] = yield* Tensor.compute([yield* model.forward(params, idx)])
    const row = (yield* Tensor.toNumberArray(logits)).slice(-vocabSize)
    const max = Math.max(...row)
    const exps = row.map((x) => Math.exp((x - max) / TEMPERATURE))
    const total = exps.reduce((a, b) => a + b, 0)
    let draw = Math.random() * total
    let next = exps.length - 1
    for (let i = 0; i < exps.length; i++) {
      draw -= exps[i]
      if (draw <= 0) {
        next = i
        break
      }
    }
    context.push(next)
    generated += chars[next]
  }
  yield* Effect.log(`\n${generated}`)
})

NodeRuntime.runMain(program.pipe(Effect.provide(Device.Best)))

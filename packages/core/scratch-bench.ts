import { Effect } from "effect"
import { Device, LearningRate, Loss, Model, Optimizer, Tensor, Trainer } from "./src/index.ts"

const VOCAB = 128, EMBED = 128, HEADS = 4, LAYERS = 4, T = 64

const gpt = Effect.gen(function* () {
  let x: Model.Model = yield* Model.embedding("wte", VOCAB, EMBED)
  for (let i = 0; i < LAYERS; i++) {
    const attn = yield* Model.chain(
      yield* Model.layerNorm(`ln1${i}`, EMBED),
      yield* Model.multiHeadAttention(`attn${i}`, EMBED, HEADS, { causal: true, rope: 10000 })
    )
    const mlp = yield* Model.chain(
      yield* Model.layerNorm(`ln2${i}`, EMBED),
      yield* Model.linear(`up${i}`, EMBED, 4 * EMBED),
      yield* Model.gelu(),
      yield* Model.linear(`down${i}`, 4 * EMBED, EMBED)
    )
    x = yield* Model.chain(x, yield* Model.residual(attn), yield* Model.residual(mlp))
  }
  return yield* Model.chain(x, yield* Model.layerNorm("lnf", EMBED), yield* Model.linear("head", EMBED, VOCAB))
})

const program = Effect.gen(function* () {
  const model = yield* gpt
  const input = yield* Tensor.fromTypedArray(new Uint32Array(Array.from({ length: T }, (_, i) => i % VOCAB)), [1, T])
  const target = yield* Tensor.fromTypedArray(new Uint32Array(Array.from({ length: T }, (_, i) => (i + 1) % VOCAB)), [1, T])
  const trainer = yield* Trainer.make(model, {
    optimizer: yield* Optimizer.adamW({ lr: 1e-3 }),
    lr: LearningRate.constant(1e-3),
    loss: Loss.crossEntropy,
    data: { input, target },
    stop: ({ step }) => step >= 300
  })
  const t0 = performance.now()
  yield* trainer.train()
  console.log(`Trainer.make: ${((performance.now() - t0) / 100).toFixed(2)} ms/step`)
  const compiled = yield* Trainer.compile(trainer)
  const t1 = performance.now()
  yield* compiled.train()
  console.log(`Trainer.compile: ${((performance.now() - t1) / 300).toFixed(2)} ms/step avg`)
  console.log("cache stats:", yield* compiled.stats())
})

Effect.runPromiseExit(program.pipe(Effect.provide(Device.Metal))).then((e) => {
  if (e._tag !== "Success") console.log(e.cause.toString())
  process.exit(0)
})

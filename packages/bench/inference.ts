// Ordinary sampled-generation latency through the fixed-lane inference path.
// Compilation and prompt prefill are outside the timer. One untimed step warms
// backend pipelines; measured rounds step every live lane and report both round
// latency and per-token latency.

import * as BackendApple from "@effect-torch/backend-apple-native"
import * as BackendCpu from "@effect-torch/backend-cpu"
import { Model, Runtime, Tensor } from "@effect-torch/core"
import { Effect } from "effect"
import { performance } from "node:perf_hooks"

const ITERS = Number(process.env.ITERS ?? 100)
const VOCAB = 256
const EMBED = 64
const HEADS = 4
const PROMPT = 16

const makeModel = Effect.gen(function*() {
  const embedding = yield* Model.embedding("wte", VOCAB, EMBED)
  const attention = yield* Model.multiHeadAttention("attn", EMBED, HEADS, { causal: true, rope: 10_000 })
  const head = yield* Model.linear("head", EMBED, VOCAB)
  return yield* Model.chain(embedding, attention, head)
})

const suite = Effect.gen(function*() {
  const runtime = yield* Runtime.Runtime
  const model = yield* makeModel
  const params = yield* Tensor.compute(yield* model.init)
  for (const batchSize of [1, 8]) {
    const tokensPerLane = Math.ceil((PROMPT + ITERS + 2) / 16) * 16
    const maxTokens = batchSize * tokensPerLane
    const program = yield* Model.inference(model, params, {
      maxTokens,
      blockSize: 16,
      prefillChunk: 16,
      batchSize,
      sampling: { temperature: 0, seed: 0 }
    })
    const generation = yield* program.generation()
    const prompts: Array<Model.GenerationAdd> = []
    for (let lane = 0; lane < batchSize; lane++) {
      const tokens = Uint32Array.from({ length: PROMPT }, (_, index) => (lane * PROMPT + index) % VOCAB)
      prompts.push({ prompt: yield* Tensor.fromTypedArray(tokens, [1, PROMPT]) })
    }
    let pages = yield* generation.add(prompts)
    pages = yield* generation.step(pages.map(({ seq }) => ({ seq })))
    const started = performance.now()
    for (let iteration = 0; iteration < ITERS; iteration++) {
      pages = yield* generation.step(pages.map(({ seq }) => ({ seq })))
    }
    const elapsed = performance.now() - started
    const roundMs = elapsed / ITERS
    const tokenUs = elapsed * 1_000 / ITERS / batchSize
    process.stdout.write(
      `${runtime.placement.deviceType.padEnd(6)} B=${String(batchSize).padEnd(2)} ` +
        `${roundMs.toFixed(3)} ms/round  ${tokenUs.toFixed(1)} us/token\n`
    )
    yield* generation.close()
  }
  yield* Tensor.clearAll(params)
})

const main = async (): Promise<void> => {
  process.stdout.write(`ordinary sampled generation, ${ITERS} measured rounds\n`)
  await Effect.runPromise(Effect.provide(suite, BackendCpu.layer))
  if (await Effect.runPromise(BackendApple.isAvailable)) {
    await Effect.runPromise(Effect.provide(suite, BackendApple.layer))
  }
}

main().catch((error) => {
  process.stderr.write(`${String(error)}\n`)
  process.exitCode = 1
})

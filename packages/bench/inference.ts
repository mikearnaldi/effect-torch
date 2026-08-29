// Measures ordinary and exact-chain sampled-generation latency through fixed
// lanes. The timer excludes compilation and prompt prefill. One untimed step
// warms backend pipelines. Measured rounds report latency percentiles,
// throughput, acceptance, native phase timings, and pool high-water marks.
// CONTEXT sets the prompt length, CONCURRENCY sets the number of simultaneous
// sessions, and ITERS sets the measured round count.

import * as BackendApple from "@effect-torch/backend-apple-native"
import * as BackendCpu from "@effect-torch/backend-cpu"
import { Model, Runtime, Speculation, Tensor } from "@effect-torch/core"
import { Effect } from "effect"
import { performance } from "node:perf_hooks"

const ITERS = Number(process.env.ITERS ?? 100)
const VOCAB = 256
const EMBED = 64
const HEADS = 4
const PROMPT = Number(process.env.CONTEXT ?? 16)
const CONCURRENCY = Number(process.env.CONCURRENCY ?? 1)

const makeModel = Effect.gen(function*() {
  const embedding = yield* Model.embedding("wte", VOCAB, EMBED)
  const attention = yield* Model.multiHeadAttention("attn", EMBED, HEADS, { causal: true, rope: 10_000 })
  const head = yield* Model.linear("head", EMBED, VOCAB)
  return yield* Model.chain(embedding, attention, head)
})

const suite = Effect.gen(function*() {
  const runtime = yield* Runtime.Runtime
  const model = yield* makeModel
  const params = yield* Tensor.compute(yield* Model.initialize(model))
  const draftParams = yield* Tensor.compute(yield* Model.initialize(model))
  const makeProposer = (componentParams: Model.Params) =>
    Speculation.autoregressive(model, componentParams, { vocabulary: VOCAB, maxDraftTokens: 4 })
  const perfectProposer = makeProposer(params)
  const independentProposer = makeProposer(draftParams)
  const modes = [
    { name: "ordinary", draftTokens: 0, proposer: undefined },
    { name: "perfect", draftTokens: 4, proposer: perfectProposer },
    { name: "independent", draftTokens: 4, proposer: independentProposer }
  ] as const
  for (const mode of modes) {
    for (const batchSize of [1, 8]) {
      const progressPerRound = mode.draftTokens + 1
      const tokensPerLane = Math.ceil((PROMPT + (ITERS + 2) * progressPerRound) / 16) * 16
      const maxTokens = CONCURRENCY * batchSize * tokensPerLane
      const program = yield* Model.inference(model, params, {
        maxTokens,
        blockSize: 16,
        prefillChunks: [16],
        batchSize,
        sampling: { temperature: 0.8, topK: 64, topP: 0.95, seed: 0 },
        speculation: mode.proposer === undefined
          ? undefined
          : { proposer: mode.proposer, maxDraftTokens: mode.draftTokens }
      })
      const generations: Array<Model.Generation> = []
      const pagesBySession: Array<ReadonlyArray<Model.TokenPage>> = []
      for (let session = 0; session < CONCURRENCY; session++) {
        const generation = yield* program.generation()
        const prompts: Array<Model.GenerationAdd> = []
        for (let lane = 0; lane < batchSize; lane++) {
          const tokens = Uint32Array.from(
            { length: PROMPT },
            (_, index) => (session * batchSize * PROMPT + lane * PROMPT + index) % VOCAB
          )
          prompts.push({ prompt: yield* Tensor.fromTypedArray(tokens, [1, PROMPT]) })
        }
        generations.push(generation)
        pagesBySession.push(yield* generation.add(prompts))
      }
      const warmed = yield* Effect.all(
        generations.map((generation, index) => generation.step(pagesBySession[index]!.map(({ seq }) => ({ seq })))),
        { concurrency: "unbounded" }
      )
      pagesBySession.splice(0, pagesBySession.length, ...warmed)
      let outputTokens = 0
      const roundTimes: Array<number> = []
      const acceptance = new Map<number, number>()
      const started = performance.now()
      for (let iteration = 0; iteration < ITERS; iteration++) {
        const roundStarted = performance.now()
        const pages = yield* Effect.all(
          generations.map((generation, index) => generation.step(pagesBySession[index]!.map(({ seq }) => ({ seq })))),
          { concurrency: "unbounded" }
        )
        pagesBySession.splice(0, pagesBySession.length, ...pages)
        roundTimes.push(performance.now() - roundStarted)
        outputTokens += pages.flat().reduce((total, page) => total + page.tokens.length, 0)
        if (mode.draftTokens > 0) {
          for (const page of pages.flat()) {
            const accepted = page.tokens.length - 1
            acceptance.set(accepted, (acceptance.get(accepted) ?? 0) + 1)
          }
        }
      }
      const elapsed = performance.now() - started
      const diagnostics = yield* program.diagnostics()
      const roundMs = elapsed / ITERS
      const tokenUs = elapsed * 1_000 / outputTokens
      roundTimes.sort((left, right) => left - right)
      const percentile = (quantile: number) =>
        roundTimes[Math.min(roundTimes.length - 1, Math.floor(quantile * roundTimes.length))]!
      const histogram = mode.draftTokens === 0
        ? ""
        : `  accept={${
          Array.from(acceptance).sort(([left], [right]) => left - right).map(([length, count]) => `${length}:${count}`)
            .join(",")
        }}`
      const speculativeRounds = Math.max(1, Number(diagnostics.speculativeRounds))
      const nativePhases = mode.draftTokens === 0
        ? ""
        : `  draft=${(Number(diagnostics.draftNanos) / 1e6 / speculativeRounds).toFixed(3)} ms/round ` +
          `verify=${(Number(diagnostics.verificationNanos) / 1e6 / speculativeRounds).toFixed(3)} ms/round`
      const pools = `  poolHighWater=${diagnostics.targetPoolHighWaterBlocks}` +
        (diagnostics.proposerPoolHighWaterBlocks === undefined
          ? ""
          : `/${diagnostics.proposerPoolHighWaterBlocks}`)
      process.stdout.write(
        `${runtime.placement.deviceType.padEnd(6)} ${mode.name.padEnd(11)} ` +
          `C=${CONCURRENCY} B=${String(batchSize).padEnd(2)} ` +
          `${roundMs.toFixed(3)} ms/round  ${(outputTokens / ITERS).toFixed(2)} tokens/round  ` +
          `${tokenUs.toFixed(1)} us/token  ${(outputTokens * 1_000 / elapsed).toFixed(1)} token/s  ` +
          `p50=${percentile(0.5).toFixed(3)} ms p95=${percentile(0.95).toFixed(3)} ms` +
          `${nativePhases}${histogram}${pools}\n`
      )
      yield* Effect.all(generations.map((generation) => generation.close()), { concurrency: "unbounded" })
    }
  }
  yield* Tensor.clearAll(params)
  yield* Tensor.clearAll(draftParams)
})

const main = async (): Promise<void> => {
  process.stdout.write(
    `positive-temperature ordinary and exact-chain generation, context=${PROMPT}, ` +
      `concurrency=${CONCURRENCY}, ${ITERS} measured rounds\n`
  )
  await Effect.runPromise(Effect.provide(suite, BackendCpu.layer))
  if (await Effect.runPromise(BackendApple.isAvailable)) {
    await Effect.runPromise(Effect.provide(suite, BackendApple.layer()))
  }
}

main().catch((error) => {
  process.stderr.write(`${String(error)}\n`)
  process.exitCode = 1
})

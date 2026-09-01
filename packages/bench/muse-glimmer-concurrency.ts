// Measures ordinary Muse-Glimmer decode throughput with concurrent user sessions
// batched into one Generation. Each concurrency compiles a matching fixed-width
// CUDA artifact, admits one sequence per session, and steps all sequences in one
// call. Model loading, compilation, prompt prefill, and warmup are outside the
// timer.
//
// CONCURRENCIES defaults to 4,8,16,32. CONTEXT defaults to 64, DECODE_TOKENS
// defaults to 128 per session, RUNS defaults to 3, and WARMUP_STEPS defaults to
// 4. COOLDOWN_MS defaults to 5000. MAX_TOKENS overrides the computed per-case
// shared pool size. PREFILL_CHUNK defaults to 64,128,256. OUTPUT defaults to
// <repo>/bench-results/muse-glimmer/concurrency-<timestamp>.jsonl. MODEL_PATH and
// TOKENIZER_PATH override the bundled files.

// Run on the CUDA devbox with:
// ./scripts/cuda-devbox.sh sync
// ./scripts/cuda-devbox.sh run pnpm bench:muse-glimmer-concurrency

import * as BackendCuda from "@effect-torch/backend-cuda"
import { Model, type Runtime, Tensor } from "@effect-torch/core"
import { MuseGlimmer } from "@effect-torch/core/models"
import * as Tokenizers from "@effect-torch/tokenizers"
import { Duration, Effect, Option, Schema } from "effect"
import * as fs from "node:fs"
import path from "node:path"
import { performance } from "node:perf_hooks"
import { fileURLToPath } from "node:url"
import { buildPrompt } from "./muse-glimmer.ts"

const directory = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(directory, "../..")
const defaultModelPath = path.join(directory, "../examples/data/Muse-Glimmer-30B-UD-Q2_K_XL.gguf")
const defaultTokenizerPath = path.join(directory, "../examples/data/muse-glimmer-tokenizer.json")
const blockSize = 16

interface Config {
  readonly concurrencies: ReadonlyArray<number>
  readonly context: number
  readonly decodeTokens: number
  readonly runs: number
  readonly warmupSteps: number
  readonly cooldownMs: number
  readonly maxTokens: number | undefined
  readonly prefillChunks: ReadonlyArray<number>
  readonly seed: number
  readonly modelPath: string
  readonly tokenizerPath: string
  readonly output: string
}

interface BenchRecord {
  readonly timestamp: string
  readonly engine: "effect-cuda"
  readonly model: "Muse-Glimmer-30B"
  readonly concurrency: number
  readonly batchSize: number
  readonly run: number
  readonly context: number
  readonly requestedTokensPerSession: number
  readonly generatedTokens: number
  readonly decodeMs: number
  readonly aggregateTokPerSec: number
  readonly perSessionTokPerSec: number
  readonly roundP50Ms: number
  readonly roundP95Ms: number
  readonly loadMs: number
  readonly compileMs: number
  readonly maxTokens: number
  readonly rssBytes: number
}

const fail = (message: string): never => {
  throw new Error(message)
}

const envInt = (name: string, fallback: number): number => {
  const raw = process.env[name]
  if (raw === undefined || raw === "") return fallback
  const value = Number(raw)
  if (!Number.isInteger(value)) return fail(`${name} must be an integer, got ${raw}`)
  return value
}

const envIntList = (name: string, fallback: ReadonlyArray<number>): ReadonlyArray<number> => {
  const raw = process.env[name]
  if (raw === undefined || raw === "") return fallback
  return raw.split(",").map((entry) => {
    const value = Number(entry.trim())
    if (!Number.isInteger(value) || value <= 0) return fail(`${name} entries must be positive integers, got ${entry}`)
    return value
  })
}

const timestamp = (): string => new Date().toISOString().replace(/[:.]/g, "-")

const loadConfig = (): Config => {
  const concurrencies = envIntList("CONCURRENCIES", [4, 8, 16, 32])
  const context = envInt("CONTEXT", 64)
  const decodeTokens = envInt("DECODE_TOKENS", 128)
  const runs = envInt("RUNS", 3)
  const warmupSteps = envInt("WARMUP_STEPS", 4)
  const cooldownMs = envInt("COOLDOWN_MS", 5_000)
  const prefillChunks = envIntList("PREFILL_CHUNK", [64, 128, 256])
  if (context < 9) return fail(`CONTEXT must be at least 9, got ${context}`)
  if (decodeTokens <= 0) return fail(`DECODE_TOKENS must be positive, got ${decodeTokens}`)
  if (runs <= 0) return fail(`RUNS must be positive, got ${runs}`)
  if (warmupSteps < 0) return fail(`WARMUP_STEPS must be non-negative, got ${warmupSteps}`)
  if (cooldownMs < 0) return fail(`COOLDOWN_MS must be non-negative, got ${cooldownMs}`)
  const tokensPerSession = Math.ceil((context + warmupSteps + decodeTokens + 1) / blockSize) * blockSize
  const requiredMaxTokens = Math.max(...concurrencies) * tokensPerSession
  const maxTokens = process.env.MAX_TOKENS === undefined ? undefined : envInt("MAX_TOKENS", requiredMaxTokens)
  if (maxTokens !== undefined && (maxTokens < requiredMaxTokens || maxTokens % blockSize !== 0)) {
    return fail(
      `MAX_TOKENS must be a multiple of ${blockSize} and at least ${requiredMaxTokens}, got ${maxTokens}`
    )
  }
  return {
    concurrencies,
    context,
    decodeTokens,
    runs,
    warmupSteps,
    cooldownMs,
    maxTokens,
    prefillChunks,
    seed: envInt("SEED", 0),
    modelPath: process.env.MODEL_PATH ?? defaultModelPath,
    tokenizerPath: process.env.TOKENIZER_PATH ?? defaultTokenizerPath,
    output: process.env.OUTPUT ??
      path.join(repoRoot, "bench-results", "muse-glimmer", `concurrency-${timestamp()}.jsonl`)
  }
}

const metadataInt = (metadata: ReadonlyMap<string, unknown>, key: string): number => {
  const value = Option.getOrUndefined(Schema.decodeUnknownOption(Schema.Int)(metadata.get(key)))
  return value === undefined ? fail(`GGUF metadata ${key} must be an integer`) : value
}

const percentile = (sorted: ReadonlyArray<number>, quantile: number): number =>
  sorted[Math.min(sorted.length - 1, Math.floor(quantile * sorted.length))]!

interface CaseInput {
  readonly config: Config
  readonly program: Model.InferenceProgram
  readonly tokenizer: Tokenizers.Tokenizer
  readonly bosTokenId: number
  readonly concurrency: number
  readonly concurrencyIndex: number
  readonly run: number
  readonly loadMs: number
  readonly compileMs: number
  readonly maxTokens: number
}

const runCase = (input: CaseInput): Effect.Effect<BenchRecord, unknown, Runtime.Runtime> => {
  let generation: Model.Generation | undefined
  return Effect.gen(function*() {
    const { config } = input
    generation = yield* input.program.generation()
    const promptTensors: Array<Tensor.Any> = []
    for (let session = 0; session < input.concurrency; session++) {
      const caseId = ((input.concurrencyIndex * config.runs + input.run) * 1000) + session
      const prompt = yield* buildPrompt(input.tokenizer, config.seed, caseId, config.context)
      const ids = new Uint32Array(config.context)
      ids[0] = input.bosTokenId
      ids.set(prompt.ids, 1)
      promptTensors.push(yield* Tensor.fromTypedArray(ids, [1, config.context]))
    }
    let pages = yield* generation.add(
      promptTensors.map((prompt) => ({
        prompt,
        maxTokens: config.warmupSteps + config.decodeTokens + 1
      }))
    )
    for (let step = 0; step < config.warmupSteps; step++) {
      pages = yield* generation.step(pages.map(({ seq }) => ({ seq })))
    }

    let generatedTokens = 0
    const roundTimes: Array<number> = []
    const started = performance.now()
    for (let step = 0; step < config.decodeTokens; step++) {
      const roundStarted = performance.now()
      pages = yield* generation.step(pages.map(({ seq }) => ({ seq })))
      roundTimes.push(performance.now() - roundStarted)
      generatedTokens += pages.reduce((total, page) => total + page.tokens.length, 0)
    }
    const decodeMs = performance.now() - started
    const expectedTokens = input.concurrency * config.decodeTokens
    if (generatedTokens !== expectedTokens) {
      return yield* Effect.die(
        new Error(`decode returned ${generatedTokens} tokens, expected ${expectedTokens}`)
      )
    }
    roundTimes.sort((left, right) => left - right)
    const aggregateTokPerSec = generatedTokens / (decodeMs / 1000)
    return {
      timestamp: new Date().toISOString(),
      engine: "effect-cuda",
      model: "Muse-Glimmer-30B",
      concurrency: input.concurrency,
      batchSize: input.concurrency,
      run: input.run,
      context: config.context,
      requestedTokensPerSession: config.decodeTokens,
      generatedTokens,
      decodeMs,
      aggregateTokPerSec,
      perSessionTokPerSec: aggregateTokPerSec / input.concurrency,
      roundP50Ms: percentile(roundTimes, 0.5),
      roundP95Ms: percentile(roundTimes, 0.95),
      loadMs: input.loadMs,
      compileMs: input.compileMs,
      maxTokens: input.maxTokens,
      rssBytes: process.memoryUsage().rss
    } satisfies BenchRecord
  }).pipe(
    Effect.ensuring(
      Effect.suspend(() => generation === undefined ? Effect.void : Effect.ignore(generation.close()))
    )
  )
}

const median = (values: ReadonlyArray<number>): number => {
  const sorted = [...values].sort((left, right) => left - right)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[middle - 1]! + sorted[middle]!) / 2 : sorted[middle]!
}

const printRecord = (record: BenchRecord): void => {
  process.stdout.write(
    `C=${String(record.concurrency).padStart(2)} run=${record.run} ` +
      `${record.decodeMs.toFixed(1).padStart(9)} ms ` +
      `${record.aggregateTokPerSec.toFixed(2).padStart(9)} aggregate tok/s ` +
      `${record.perSessionTokPerSec.toFixed(2).padStart(7)} tok/s/session ` +
      `p50=${record.roundP50Ms.toFixed(2)} ms p95=${record.roundP95Ms.toFixed(2)} ms\n`
  )
}

const printSummary = (config: Config, records: ReadonlyArray<BenchRecord>): void => {
  process.stdout.write("\nmedian decode throughput\n")
  for (const concurrency of config.concurrencies) {
    const group = records.filter((record) => record.concurrency === concurrency)
    process.stdout.write(
      `C=${String(concurrency).padStart(2)} ` +
        `${median(group.map((record) => record.aggregateTokPerSec)).toFixed(2).padStart(9)} aggregate tok/s ` +
        `${median(group.map((record) => record.perSessionTokPerSec)).toFixed(2).padStart(7)} tok/s/session ` +
        `p50=${median(group.map((record) => record.roundP50Ms)).toFixed(2)} ms ` +
        `p95=${median(group.map((record) => record.roundP95Ms)).toFixed(2)} ms\n`
    )
  }
}

const suite = (config: Config, records: Array<BenchRecord>): Effect.Effect<void, unknown, Runtime.Runtime> =>
  Effect.gen(function*() {
    const tokenizer = yield* Tokenizers.fromFile(config.tokenizerPath, {
      ...Tokenizers.strictConfig,
      specialTokens: "Always"
    })
    const loadStarted = performance.now()
    const loaded = yield* MuseGlimmer.loadGGUF(config.modelPath)
    const loadMs = performance.now() - loadStarted
    const bosTokenId = metadataInt(loaded.metadata, "tokenizer.ggml.bos_token_id")
    process.stderr.write(`loaded in ${(loadMs / 1000).toFixed(2)}s\n`)
    return yield* Effect.gen(function*() {
      let first = true
      for (const [concurrencyIndex, concurrency] of config.concurrencies.entries()) {
        const tokensPerSession = Math.ceil(
          (config.context + config.warmupSteps + config.decodeTokens + 1) / blockSize
        ) * blockSize
        const maxTokens = config.maxTokens ?? concurrency * tokensPerSession
        const compileStarted = performance.now()
        const program = yield* Model.inference(loaded.model, loaded.params, {
          maxTokens,
          blockSize,
          kvDtype: "f16",
          prefillChunks: config.prefillChunks,
          batchSize: concurrency,
          sampling: { temperature: 0, topK: 0, topP: 1, seed: config.seed }
        })
        const compileMs = performance.now() - compileStarted
        process.stderr.write(
          `compiled batchSize=${concurrency} in ${(compileMs / 1000).toFixed(2)}s\n`
        )
        for (let run = 0; run < config.runs; run++) {
          if (!first && config.cooldownMs > 0) yield* Effect.sleep(Duration.millis(config.cooldownMs))
          first = false
          const record = yield* runCase({
            config,
            program,
            tokenizer,
            bosTokenId,
            concurrency,
            concurrencyIndex,
            run,
            loadMs,
            compileMs,
            maxTokens
          })
          records.push(record)
          printRecord(record)
        }
      }
    }).pipe(Effect.ensuring(Tensor.clearAll(loaded.params)))
  })

const writeOutput = (config: Config, records: ReadonlyArray<BenchRecord>): void => {
  fs.mkdirSync(path.dirname(config.output), { recursive: true })
  fs.writeFileSync(config.output, records.map((record) => JSON.stringify(record)).join("\n") + "\n")
  process.stderr.write(`wrote ${records.length} records to ${config.output}\n`)
}

const main = async (): Promise<void> => {
  const config = loadConfig()
  process.stderr.write(
    `concurrencies=${config.concurrencies.join(",")} context=${config.context} ` +
      `decodeTokens=${config.decodeTokens} runs=${config.runs} warmupSteps=${config.warmupSteps} ` +
      `maxTokens=${config.maxTokens ?? "per-concurrency"}\n`
  )
  const records: Array<BenchRecord> = []
  await Effect.runPromise(Effect.provide(suite(config, records), BackendCuda.layer()))
  writeOutput(config, records)
  printSummary(config, records)
}

main().catch((error) => {
  process.stderr.write(`${String(error)}\n`)
  process.exitCode = 1
})

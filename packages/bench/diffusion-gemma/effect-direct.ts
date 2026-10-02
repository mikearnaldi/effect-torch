import * as BackendCuda from "@effect-torch/backend-cuda"
import { Diffusion, type Runtime, Safetensors, Tensor } from "@effect-torch/core"
import { DiffusionGemma } from "@effect-torch/models"
import { Duration, Effect } from "effect"
import { createHash } from "node:crypto"
import * as fs from "node:fs"
import path from "node:path"
import { performance } from "node:perf_hooks"
import { fileURLToPath } from "node:url"
import { loadManifest, percentile, type PromptCase, promptCases } from "./common.ts"

const directory = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(directory, "../../..")
const manifestPath = process.env.MANIFEST ?? path.join(directory, "manifest.json")
const manifest = loadManifest(manifestPath)
const modelPath = process.env.MODEL_PATH
const initializedStatePath = process.env.INITIALIZED_STATE_PATH
const traceGeneration = process.env.GENERATION_TRACE === "1"

const defaults = DiffusionGemma.generationDefaults
if (
  manifest.model.id !== DiffusionGemma.pins.model ||
  manifest.model.revision !== DiffusionGemma.pins.revision ||
  manifest.model.tokenizerSha256 !== DiffusionGemma.pins.tokenizerSha256 ||
  manifest.model.chatTemplateSha256 !== DiffusionGemma.pins.chatTemplateSha256 ||
  manifest.generation.maxNewTokens !== defaults.maxNewTokens ||
  manifest.generation.maxSteps !== defaults.maxSteps ||
  manifest.generation.entropyBound !== defaults.entropyBound ||
  manifest.generation.minTemperature !== defaults.minTemperature ||
  manifest.generation.maxTemperature !== defaults.maxTemperature ||
  manifest.generation.stabilityThreshold !== defaults.stabilityThreshold ||
  manifest.generation.confidenceThreshold !== defaults.confidenceThreshold ||
  manifest.generation.padTokenId !== defaults.padTokenId ||
  JSON.stringify(manifest.generation.eosTokenIds) !== JSON.stringify(defaults.eosTokenIds)
) {
  throw new Error("benchmark manifest differs from the pinned DiffusionGemma contract")
}

if (modelPath === undefined || modelPath.length === 0) {
  throw new Error("MODEL_PATH must point to the pinned DiffusionGemma checkpoint directory")
}

if (initializedStatePath === undefined || initializedStatePath.length === 0) {
  throw new Error("INITIALIZED_STATE_PATH must point to the pinned initialized-rope.safetensors file")
}

const initializedStateSha256 = createHash("sha256").update(fs.readFileSync(initializedStatePath)).digest("hex")
if (initializedStateSha256 !== manifest.model.initializedStateSha256) {
  throw new Error("initialized state SHA-256 mismatch")
}

const timestamp = (): string => new Date().toISOString().replace(/[:.]/g, "-")
const outputPath = process.env.OUTPUT ??
  path.join(repoRoot, "bench-results", "diffusion-gemma", `effect-direct-${timestamp()}.jsonl`)

if (fs.existsSync(outputPath)) {
  throw new Error(`refusing to overwrite ${outputPath}`)
}

interface PreparedPrompt extends PromptCase {
  readonly ids: Uint32Array
}

interface RequestMeasurement {
  readonly prompt: PreparedPrompt
  readonly seed: number
  readonly tokens: Uint32Array
  readonly refinements: number
  readonly blocks: number
  readonly trace: ReadonlyArray<Diffusion.Progress>
  readonly generatedTokens: number
  readonly elapsedMilliseconds: number
  readonly firstPageMilliseconds: number
  readonly randomMilliseconds: number
  readonly processingMilliseconds: number
  readonly randomInputBytes: number
  readonly statisticsReadbackBytes: number
  readonly stop: "policy" | "length"
}

const writeRecord = (line: string): void => {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true })
  fs.appendFileSync(outputPath, line + "\n")
}

const requestSeed = (target: number, outputTokens: number, concurrency: number, run: number, request: number): number =>
  (manifest.generation.seed + target * 101 + outputTokens * 17 + concurrency * 13 + run * 7 + request) >>> 0

const runRequest = (
  artifact: Diffusion.Artifact,
  prompt: PreparedPrompt,
  outputTokens: number,
  seed: number
): Effect.Effect<RequestMeasurement, unknown, Runtime.Runtime> =>
  Effect.gen(function*() {
    const started = performance.now()
    let firstPageMilliseconds: number | undefined
    const trace: Array<Diffusion.Progress> = []
    let options: DiffusionGemma.GenerateOptions = {
      maxNewTokens: outputTokens,
      maxSteps: manifest.generation.maxSteps,
      entropyBound: manifest.generation.entropyBound,
      minTemperature: manifest.generation.minTemperature,
      maxTemperature: manifest.generation.maxTemperature,
      stabilityThreshold: manifest.generation.stabilityThreshold,
      confidenceThreshold: manifest.generation.confidenceThreshold,
      eosTokenIds: manifest.generation.eosTokenIds,
      padTokenId: manifest.generation.padTokenId,
      outputLimit: "exact",
      seed,
      onPage: () =>
        Effect.sync(() => {
          firstPageMilliseconds ??= performance.now() - started
        })
    }
    if (traceGeneration) {
      options = {
        ...options,
        onProgress: (progress) =>
          Effect.sync(() => {
            trace.push(progress)
          })
      }
    }
    const result = yield* DiffusionGemma.generate(artifact, prompt.ids, options)
    const elapsedMilliseconds = performance.now() - started

    return {
      prompt,
      seed,
      tokens: result.tokens,
      refinements: result.refinements,
      blocks: result.blocks,
      trace,
      generatedTokens: result.tokens.length,
      elapsedMilliseconds,
      firstPageMilliseconds: firstPageMilliseconds ?? elapsedMilliseconds,
      randomMilliseconds: result.randomMilliseconds,
      processingMilliseconds: result.processingMilliseconds,
      randomInputBytes: result.randomInputBytes,
      statisticsReadbackBytes: result.statisticsReadbackBytes,
      stop: result.stop
    }
  })

const runBatch = (
  artifact: Diffusion.Artifact,
  prompts: ReadonlyArray<PreparedPrompt>,
  outputTokens: number,
  target: number,
  run: number
): Effect.Effect<
  { readonly elapsedMilliseconds: number; readonly requests: ReadonlyArray<RequestMeasurement> },
  unknown,
  Runtime.Runtime
> =>
  Effect.gen(function*() {
    const started = performance.now()
    const requests = yield* Effect.forEach(
      prompts,
      (prompt, request) =>
        runRequest(artifact, prompt, outputTokens, requestSeed(target, outputTokens, prompts.length, run, request)),
      { concurrency: "unbounded" }
    )

    return { elapsedMilliseconds: performance.now() - started, requests }
  })

const suite = Effect.scoped(Effect.gen(function*() {
  const loadStarted = performance.now()
  const loaded = yield* Effect.acquireRelease(
    DiffusionGemma.loadCheckpoint(modelPath),
    (checkpoint) => Tensor.clearAll(checkpoint.ownedParameters),
    { interruptible: true }
  )
  const template = yield* DiffusionGemma.loadPinned(modelPath)
  const initializedState = yield* Effect.acquireRelease(
    Safetensors.load(initializedStatePath),
    (tensors) => Tensor.clearAll(Object.values(tensors)),
    { interruptible: true }
  )
  const expectedInitializedState = {
    "model.decoder.rotary_emb.full_attention_inv_freq": [256],
    "model.decoder.rotary_emb.sliding_attention_inv_freq": [128]
  } as const
  if (
    Object.keys(initializedState).length !== Object.keys(expectedInitializedState).length ||
    Object.entries(expectedInitializedState).some(([name, shape]) => {
      const tensor = initializedState[name]
      return tensor === undefined || tensor.dtype !== "f32" || tensor.shape.join(",") !== shape.join(",")
    })
  ) return yield* Effect.die("initialized state tensor metadata differs")
  const loadMilliseconds = performance.now() - loadStarted

  if (loaded.config.canvas_length !== manifest.generation.canvasLength) {
    return yield* Effect.die(
      `checkpoint canvas length ${loaded.config.canvas_length} does not match manifest ${manifest.generation.canvasLength}`
    )
  }

  const compileStarted = performance.now()
  const model = DiffusionGemma.fromTensors(loaded, { ...loaded.tensors, ...initializedState })
  const artifact = yield* Diffusion.compile(model.definition, model.parameters, {
    fuseFullReadout: process.env.DIFFUSION_FUSED_READOUT === "1",
    cachePositions: process.env.DIFFUSION_CACHE_POSITIONS === "1",
    maxTokens: manifest.deployment.maxTokens,
    blockSize: 16,
    prefillChunks: [16, 64, 256, 512],
    canvasLengths: [manifest.generation.canvasLength]
  })
  const compileMilliseconds = performance.now() - compileStarted

  const prepared = yield* Effect.forEach(promptCases(manifest), (prompt) =>
    Effect.gen(function*() {
      const rendered = yield* template.tokenizer.applyChatTemplate(template.chatTemplate, prompt.messages, {
        addGenerationPrompt: true,
        variables: { bos_token: "<bos>", enable_thinking: manifest.generation.enableThinking }
      })
      const encoded = yield* template.tokenizer.encode(rendered, { addSpecialTokens: false })
      if (encoded.data.length !== prompt.targetTokens) {
        return yield* Effect.die(
          `prompt ${prompt.id} encoded to ${encoded.data.length} tokens, expected ${prompt.targetTokens}`
        )
      }

      return { ...prompt, ids: encoded.data }
    }))

  for (const target of manifest.matrix.promptTargets) {
    const candidates = prepared.filter((prompt) => prompt.targetTokens === target)

    for (const outputTokens of manifest.matrix.outputTokens) {
      for (const concurrency of manifest.matrix.concurrencies) {
        const prompts = candidates.slice(0, concurrency)
        if (prompts.length !== concurrency) {
          return yield* Effect.die(`prompt target ${target} has ${prompts.length} cases, expected ${concurrency}`)
        }

        const warmupStarted = performance.now()
        for (let warmup = 0; warmup < manifest.matrix.warmupRuns; warmup++) {
          yield* runBatch(artifact, prompts, outputTokens, target, -(warmup + 1))
        }
        const warmupMilliseconds = performance.now() - warmupStarted

        for (let run = 0; run < manifest.matrix.measuredRuns; run++) {
          const measured = yield* runBatch(artifact, prompts, outputTokens, target, run)
          const generatedTokens = measured.requests.reduce((total, request) => total + request.generatedTokens, 0)
          const requestMilliseconds = measured.requests.map((request) => request.elapsedMilliseconds)
          const firstPageMilliseconds = measured.requests.map((request) => request.firstPageMilliseconds)

          const record = {
            schemaVersion: manifest.schemaVersion,
            timestamp: new Date().toISOString(),
            boundary: "direct",
            engine: "effect-torch",
            measurementMode: traceGeneration ? "generation-diagnostic" : "timing",
            model: manifest.model,
            generation: manifest.generation,
            deployment: manifest.deployment,
            targetPromptTokens: target,
            actualPromptTokens: prompts.map((prompt) => prompt.ids.length),
            promptIds: prompts.map((prompt) => prompt.id),
            promptTokenIds: prompts.map((prompt) => Array.from(prompt.ids)),
            promptContentSha256: prompts.map((prompt) => prompt.contentSha256),
            requestedOutputTokens: outputTokens,
            concurrency,
            run,
            generatedTokens,
            generatedTokensPerRequest: measured.requests.map((request) => request.generatedTokens),
            requestSeeds: measured.requests.map((request) => request.seed),
            generatedTokenIds: measured.requests.map((request) => Array.from(request.tokens)),
            refinementsPerRequest: measured.requests.map((request) => request.refinements),
            blocksPerRequest: measured.requests.map((request) => request.blocks),
            elapsedMilliseconds: measured.elapsedMilliseconds,
            requestMilliseconds,
            firstPageMilliseconds,
            aggregateTokensPerSecond: generatedTokens * 1000 / measured.elapsedMilliseconds,
            requestP50Milliseconds: percentile(requestMilliseconds, 0.5),
            requestP95Milliseconds: percentile(requestMilliseconds, 0.95),
            requestP99Milliseconds: percentile(requestMilliseconds, 0.99),
            firstPageP50Milliseconds: percentile(firstPageMilliseconds, 0.5),
            firstPageP95Milliseconds: percentile(firstPageMilliseconds, 0.95),
            firstPageP99Milliseconds: percentile(firstPageMilliseconds, 0.99),
            randomMilliseconds: measured.requests.reduce((total, request) => total + request.randomMilliseconds, 0),
            processingMilliseconds: measured.requests.reduce(
              (total, request) => total + request.processingMilliseconds,
              0
            ),
            randomInputBytes: measured.requests.reduce((total, request) => total + request.randomInputBytes, 0),
            statisticsReadbackBytes: measured.requests.reduce(
              (total, request) => total + request.statisticsReadbackBytes,
              0
            ),
            finishReasons: measured.requests.map((request) => request.stop === "policy" ? "stop" : "length"),
            loadMilliseconds,
            compileMilliseconds,
            warmupMilliseconds,
            rssBytes: process.memoryUsage().rss
          }
          if (traceGeneration) {
            Object.assign(record, {
              generationTrace: measured.requests.map((request) =>
                request.trace.map((progress) => ({
                  block: progress.block.index,
                  step: progress.step.index,
                  remaining: progress.step.remaining,
                  done: progress.done,
                  argmaxTokens: Array.from(progress.draft)
                }))
              )
            })
          }
          writeRecord(JSON.stringify(record))
        }

        if (manifest.matrix.cooldownMilliseconds > 0) {
          yield* Effect.sleep(Duration.millis(manifest.matrix.cooldownMilliseconds))
        }
      }
    }
  }

  yield* Effect.log(`Wrote ${outputPath}`)
}))

await Effect.runPromise(Effect.provide(suite, BackendCuda.layer()))

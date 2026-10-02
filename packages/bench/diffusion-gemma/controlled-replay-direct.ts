/** Controlled trajectory capture/timing, never reported as natural generation. */
import * as BackendCuda from "@effect-torch/backend-cuda"
import { Diffusion, Runtime, Safetensors, Tensor } from "@effect-torch/core"
import { DiffusionGemma } from "@effect-torch/models"
import { Effect } from "effect"
import assert from "node:assert/strict"
import { createHash } from "node:crypto"
import { appendFileSync, existsSync, mkdirSync } from "node:fs"
import { mkdir, readFile, writeFile } from "node:fs/promises"
import path from "node:path"
import { performance } from "node:perf_hooks"
import { fileURLToPath } from "node:url"
import { loadManifest, promptCases } from "./common.ts"
import { validateReplayManifest } from "./controlled-replay-validation.ts"
import {
  captureArtifact,
  includeFinalCommit,
  preloadReplay,
  replayArtifact,
  type ReplayManifest
} from "./controlled-replay.ts"

const directory = path.dirname(fileURLToPath(import.meta.url))
const manifest = loadManifest(process.env.MANIFEST ?? path.join(directory, "manifest.json"))
const modelPath = process.env.MODEL_PATH
const initializedPath = process.env.INITIALIZED_STATE_PATH
const replayDirectory = process.env.REPLAY_DIRECTORY
const mode = process.env.REPLAY_MODE ?? "capture"
const profiling = process.env.REPLAY_PROFILE === "1"
const graphDiagnostics = process.env.EFFECT_TORCH_CUDA_GRAPH_DIAGNOSTICS === "1"
const compileDiagnosticsPath = process.env.EFFECT_TORCH_COMPILE_DIAGNOSTICS_PATH
const timelineWindow = process.env.EFFECT_TORCH_CUPTI_WINDOW === "1"
const distinctSeeds = process.env.REPLAY_DISTINCT_SEEDS === "1"
const runs = timelineWindow ? [-1, -2, 0] : profiling ? [-1, 0] : [-1, -2, 0, 1, 2, 3, 4]
const caseRuns = distinctSeeds ? timelineWindow ? [-1, -2, 0, 1, 2, 3, 4] : runs : [0]
const wallNanoseconds = (): string => BigInt(Math.round((performance.timeOrigin + performance.now()) * 1e6)).toString()
const requestSeed = (target: number, run: number): number =>
  (manifest.generation.seed + target * 101 + 64 * 17 + 13 + run * 7) >>> 0
assert(modelPath && initializedPath && replayDirectory, "MODEL_PATH, INITIALIZED_STATE_PATH, REPLAY_DIRECTORY required")
assert(mode === "capture" || mode === "timing")
const sha = (bytes: Uint8Array): string => createHash("sha256").update(bytes).digest("hex")
assert.equal(sha(await readFile(initializedPath)), manifest.model.initializedStateSha256)
const outputPath = process.env.OUTPUT
if (outputPath !== undefined) {
  assert(!existsSync(outputPath), "refusing to overwrite timing output")
  mkdirSync(path.dirname(outputPath), { recursive: true })
}
const writeTiming = (line: string): void => {
  if (outputPath !== undefined) appendFileSync(outputPath, line)
  process.stdout.write(line)
}
const requestedTargets = (process.env.REPLAY_TARGETS ?? "32,128").split(",").map(Number)

await Effect.runPromise(Effect.provide(
  Effect.scoped(Effect.gen(function*() {
    const loaded = yield* Effect.acquireRelease(
      DiffusionGemma.loadCheckpoint(modelPath),
      (value) => Tensor.clearAll(value.ownedParameters),
      { interruptible: true }
    )
    const initialized = yield* Effect.acquireRelease(
      Safetensors.load(initializedPath),
      (value) => Tensor.clearAll(Object.values(value)),
      { interruptible: true }
    )
    const pinned = yield* DiffusionGemma.loadPinned(modelPath)
    const model = DiffusionGemma.fromTensors(loaded, { ...loaded.tensors, ...initialized })
    const compiled = yield* Diffusion.compile(model.definition, model.parameters, {
      cachePositions: process.env.DIFFUSION_CACHE_POSITIONS === "1",
      fuseFullReadout: process.env.DIFFUSION_FUSED_READOUT === "1",
      maxTokens: manifest.deployment.maxTokens,
      blockSize: 16,
      prefillChunks: [16, 64, 256, 512],
      canvasLengths: [manifest.generation.canvasLength]
    })
    const counts = { prefill: 0, reads: 0, samplers: 0, commits: 0 }
    const artifact = includeFinalCommit(compiled, counts)
    for (const target of requestedTargets) {
      yield* Effect.scoped(Effect.gen(function*() {
        const prompt = promptCases(manifest).find((value) => value.targetTokens === target)
        assert(prompt !== undefined)
        const rendered = yield* pinned.tokenizer.applyChatTemplate(pinned.chatTemplate, prompt.messages, {
          addGenerationPrompt: true,
          variables: { bos_token: "<bos>", enable_thinking: manifest.generation.enableThinking }
        })
        const encoded = yield* pinned.tokenizer.encode(rendered, { addSpecialTokens: false })
        assert.equal(encoded.data.length, target)
        const options: DiffusionGemma.GenerateOptions = {
          maxNewTokens: 64,
          maxSteps: manifest.generation.maxSteps,
          entropyBound: manifest.generation.entropyBound,
          minTemperature: manifest.generation.minTemperature,
          maxTemperature: manifest.generation.maxTemperature,
          stabilityThreshold: manifest.generation.stabilityThreshold,
          confidenceThreshold: manifest.generation.confidenceThreshold,
          eosTokenIds: manifest.generation.eosTokenIds,
          padTokenId: manifest.generation.padTokenId,
          outputLimit: "exact"
        }
        const cases = new Map<number, {
          readonly replay: ReplayManifest
          readonly digest: string
          readonly controlled: Diffusion.Artifact
        }>()
        // Preload every selected trajectory before any timing begins. Distinct-seed
        // runs use the original benchmark seed schedule, including separate warmups.
        for (const caseRun of caseRuns) {
          const seed = requestSeed(target, caseRun)
          const requestDirectory = path.join(replayDirectory, `prompt-${target}-seed-${seed}`)
          if (mode === "capture") {
            yield* Effect.promise(() => mkdir(requestDirectory, { recursive: true }))
            const replay: ReplayManifest = {
              schema: "effect-torch-controlled-trajectory-v1",
              label: "controlled-trajectory-replay-not-natural-generation",
              request: { promptTokenIds: Array.from(encoded.data), seed, maxNewTokens: 64 },
              steps: [],
              committedCanvasTokenIds: [],
              outputTokenIds: []
            }
            const capture = captureArtifact(
              artifact,
              requestDirectory,
              replay,
              (remaining) =>
                DiffusionGemma.generationTemperature(
                  options.minTemperature!,
                  options.maxTemperature!,
                  options.maxSteps!,
                  remaining
                )
            )
            const result = yield* DiffusionGemma.generate(capture, encoded.data, { ...options, seed })
            replay.outputTokenIds = Array.from(result.tokens)
            assert.equal(replay.steps.length, result.refinements)
            const bytes = Buffer.from(JSON.stringify(replay, null, 2))
            yield* Effect.promise(() => writeFile(path.join(requestDirectory, "manifest.json"), bytes, { flag: "wx" }))
            process.stdout.write(
              JSON.stringify({
                workload: "controlled-trajectory-replay",
                measurementMode: "diagnostic-capture",
                promptTokens: target,
                seed,
                refinements: result.refinements,
                requestDirectory,
                replayManifestSha256: sha(bytes)
              }) + "\n"
            )
          } else {
            const bytes = yield* Effect.promise(() => readFile(path.join(requestDirectory, "manifest.json")))
            // SAFETY: This paired benchmark reads manifests written by its capture mode.
            // The schema tag and request identity are checked below; preloadReplay verifies
            // each referenced feedback file hash, dtype, and shape before timing begins.
            const replay = JSON.parse(bytes.toString()) as ReplayManifest
            validateReplayManifest(replay, { promptTokenIds: Array.from(encoded.data), seed, maxNewTokens: 64 })
            const bank = yield* preloadReplay(requestDirectory, replay)
            cases.set(caseRun, { replay, digest: sha(bytes), controlled: replayArtifact(artifact, replay, bank) })
          }
        }
        if (mode === "timing") {
          for (const run of runs) {
            const caseRun = distinctSeeds ? run : 0
            const selected = cases.get(caseRun)
            assert(selected !== undefined)
            const { replay, digest, controlled } = selected
            const seed = requestSeed(target, caseRun)
            counts.prefill =
              counts.reads =
              counts.samplers =
              counts.commits =
                0
            const windowStart = timelineWindow ? wallNanoseconds() : undefined
            const started = performance.now()
            const result = yield* DiffusionGemma.generate(controlled, encoded.data, { ...options, seed })
            const elapsedMilliseconds = performance.now() - started
            const windowEnd = timelineWindow ? wallNanoseconds() : undefined
            assert.deepEqual(Array.from(result.tokens), replay.outputTokenIds)
            assert.equal(result.refinements, replay.steps.length)
            assert.deepEqual(counts, {
              prefill: 1,
              reads: replay.steps.length,
              samplers: replay.steps.length,
              commits: 1
            })
            if (run >= 0) {
              const measurement = {
                engine: "effect-torch",
                boundary: "direct",
                model: manifest.model,
                generation: manifest.generation,
                deployment: manifest.deployment,
                prefixCache: "disabled",
                workload: "controlled-trajectory-replay",
                measurementMode: profiling || timelineWindow || graphDiagnostics || compileDiagnosticsPath !== undefined
                  ? "diagnostic-profile"
                  : "timing",
                timestamp: new Date().toISOString(),
                replayManifestSha256: digest,
                replaySeedMode: distinctSeeds ? "distinct" : "fixed",
                replayFinalCommit: "included",
                encoderCommitsPerRequest: [counts.commits],
                repetitionCooldownMilliseconds: 0,
                prefillInvocationsPerRequest: [counts.prefill],
                modelReadInvocationsPerRequest: [counts.reads],
                decodeCallsPerRequest: [counts.reads + counts.commits],
                samplerInvocationsPerRequest: [counts.samplers],
                targetPromptTokens: target,
                actualPromptTokens: [target],
                promptIds: [prompt.id],
                promptContentSha256: [prompt.contentSha256],
                requestedOutputTokens: 64,
                generatedTokens: result.tokens.length,
                generatedTokensPerRequest: [result.tokens.length],
                concurrency: 1,
                run,
                elapsedMilliseconds,
                requestSeeds: [seed],
                promptTokenIds: [Array.from(encoded.data)],
                generatedTokenIds: [Array.from(result.tokens)],
                refinementsPerRequest: [result.refinements],
                blocksPerRequest: [result.blocks],
                finishReasons: [result.stop],
                measurementWindowWallNs: timelineWindow ? [windowStart, windowEnd] : undefined
              }
              writeTiming(JSON.stringify(measurement) + "\n")
            }
          }
        }
      }))
    }
  })).pipe(
    Effect.tap(() =>
      Effect.gen(function*() {
        const runtime = yield* Runtime.Runtime
        const cleanupBytes = yield* runtime.extensions.diagnostics.externalMemoryBytes
        process.stdout.write(
          JSON.stringify({
            cleanupBytes,
            measurementMode: graphDiagnostics || compileDiagnosticsPath !== undefined ? "diagnostic-profile" : mode
          }) + "\n"
        )
        assert.equal(cleanupBytes, 0)
      })
    ),
    Effect.updateService(Runtime.Runtime, (runtime) =>
      compileDiagnosticsPath === undefined ? runtime : {
        ...runtime,
        compile: (request) =>
          runtime.compile(request).pipe(Effect.tap((executable) =>
            Effect.sync(() => {
              appendFileSync(
                compileDiagnosticsPath,
                JSON.stringify({
                  event: "compile-diagnostics",
                  measurementMode: "diagnostic-profile",
                  roots: request.roots.length,
                  diagnostics: executable.diagnostics
                }) + "\n"
              )
            })
          ))
      })
  ),
  BackendCuda.layer()
))

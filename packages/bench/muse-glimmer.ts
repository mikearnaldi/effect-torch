// Muse-Glimmer benchmark harness comparing effect-torch (ordinary and DFlash
// speculative modes) against llama.cpp in one orchestrated run. The effect
// side loads MuseGlimmer.loadGGUF and DFlash.loadGGUF once, compiles one
// Model.inference artifact per mode, and drives the low-level
// Generation.add/step API directly (no Chat parsing or detokenization in the
// timed path). The llama.cpp side runs llama-bench once over all
// contexts/max-new plus representative llama-cli and llama-speculative-simple
// end-to-end cases when LLAMA_CPP_BIN points at a binary directory.
//
// Prompts are deterministic plain text with exact token counts built through
// tokenizer encode/decode round-trips; each case gets a unique prefix so the
// effect prefix cache cannot share blocks across runs. The effect prompt
// includes the GGUF BOS id explicitly while llama receives BOS-less text,
// matching llama.cpp's default BOS insertion. Prompt construction dies loudly
// when a round-trip changes the token count.
//
// Env knobs: ENGINE=effect|llama|all (default all), MODE=ordinary|dflash|both
// (default both), CONTEXTS (csv, default 64,512,1024,2048,3072), MAX_NEW
// (default 128), RUNS (default 2), WARMUP (default 1), MAX_TOKENS (default
// 4096), COOLDOWN_MS (default 5000), SEED (default 0),
// TEMPERATURE (default 0, greedy), TOP_K (default 0), TOP_P (default 1),
// PREFILL_CHUNK (single chunk size or csv shape buckets, default 256),
// DRAFT_TOKENS
// (default 7), OUTPUT (default
// <repo>/bench-results/muse-glimmer/<timestamp>.jsonl), MODEL_PATH, DRAFT_PATH,
// TOKENIZER_PATH, LLAMA_CPP_BIN (llama.cpp
// binary directory; enables the llama side when ENGINE asks for it),
// LLAMA_E2E_CONTEXTS (csv, default 3072).

import * as BackendApple from "@effect-torch/backend-apple-native"
import { Model, type Runtime, Tensor } from "@effect-torch/core"
import { MuseGlimmer } from "@effect-torch/core/models"
import { DFlash } from "@effect-torch/core/proposers"
import * as Tokenizers from "@effect-torch/tokenizers"
import { Duration, Effect } from "effect"
import { execFile } from "node:child_process"
import * as fs from "node:fs"
import path from "node:path"
import { performance } from "node:perf_hooks"
import { fileURLToPath, pathToFileURL } from "node:url"
import { promisify } from "node:util"

const execFileAsync = promisify(execFile)

const directory = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(directory, "../..")
const defaultModelPath = path.join(directory, "../examples/data/Muse-Glimmer-30B-UD-Q2_K_XL.gguf")
const defaultDraftPath = path.join(directory, "../examples/data/dflash-Muse-Glimmer-30B-Q4_K_M.gguf")
const defaultTokenizerPath = path.join(directory, "../examples/data/muse-glimmer-tokenizer.json")

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

type Engine = "effect" | "llama" | "all"
type Mode = "ordinary" | "dflash"

interface Config {
  readonly engine: Engine
  readonly modes: ReadonlyArray<Mode>
  readonly contexts: ReadonlyArray<number>
  readonly maxNew: number
  readonly runs: number
  readonly warmup: number
  readonly cooldownMs: number
  readonly seed: number
  readonly temperature: number
  readonly topK: number
  readonly topP: number
  readonly output: string
  readonly llamaBin: string | undefined
  readonly llamaE2eContexts: ReadonlyArray<number>
  readonly modelPath: string
  readonly draftPath: string
  readonly tokenizerPath: string
  readonly maxTokens: number
  readonly blockSize: number
  readonly prefillChunks: ReadonlyArray<number>
  readonly draftTokens: number
  readonly batchSize: number
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

const envFloat = (name: string, fallback: number): number => {
  const raw = process.env[name]
  if (raw === undefined || raw === "") return fallback
  const value = Number(raw)
  if (!Number.isFinite(value) || value < 0) return fail(`${name} must be a non-negative number, got ${raw}`)
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
  const engineRaw = process.env.ENGINE ?? "all"
  if (engineRaw !== "effect" && engineRaw !== "llama" && engineRaw !== "all") {
    return fail(`ENGINE must be effect|llama|all, got ${engineRaw}`)
  }
  const modeRaw = process.env.MODE ?? "both"
  if (modeRaw !== "ordinary" && modeRaw !== "dflash" && modeRaw !== "both") {
    return fail(`MODE must be ordinary|dflash|both, got ${modeRaw}`)
  }
  const maxTokens = envInt("MAX_TOKENS", 4096)
  if (maxTokens <= 0 || maxTokens % 16 !== 0) {
    return fail(`MAX_TOKENS must be a positive multiple of 16, got ${maxTokens}`)
  }
  const contexts = envIntList("CONTEXTS", [64, 512, 1024, 2048, 3072])
  const prefillChunks = envIntList("PREFILL_CHUNK", [256])
  const maxNew = envInt("MAX_NEW", 128)
  for (const context of contexts) {
    if (context + maxNew > maxTokens) {
      return fail(`context ${context} + MAX_NEW ${maxNew} exceeds the ${maxTokens}-token pool`)
    }
  }
  return {
    engine: engineRaw,
    modes: modeRaw === "both" ? ["ordinary", "dflash"] : [modeRaw],
    contexts,
    maxNew,
    runs: envInt("RUNS", 2),
    warmup: envInt("WARMUP", 1),
    cooldownMs: envInt("COOLDOWN_MS", 5_000),
    seed: envInt("SEED", 0),
    temperature: envFloat("TEMPERATURE", 0),
    topK: envInt("TOP_K", 0),
    topP: envFloat("TOP_P", 1),
    output: process.env.OUTPUT ?? path.join(repoRoot, "bench-results", "muse-glimmer", `${timestamp()}.jsonl`),
    llamaBin: process.env.LLAMA_CPP_BIN,
    llamaE2eContexts: envIntList("LLAMA_E2E_CONTEXTS", [3072]),
    modelPath: process.env.MODEL_PATH ?? defaultModelPath,
    draftPath: process.env.DRAFT_PATH ?? defaultDraftPath,
    tokenizerPath: process.env.TOKENIZER_PATH ?? defaultTokenizerPath,
    maxTokens,
    blockSize: 16,
    prefillChunks,
    draftTokens: envInt("DRAFT_TOKENS", 7),
    batchSize: 1
  }
}

// ---------------------------------------------------------------------------
// Normalized JSONL record
// ---------------------------------------------------------------------------

interface BenchRecord {
  readonly timestamp: string
  readonly engine: "effect" | "llama-bench" | "llama-cli" | "llama-speculative-simple"
  readonly mode: Mode
  readonly context: number
  readonly run: number
  readonly requestedTokens: number
  readonly generatedTokens?: number | undefined
  readonly loadMs?: number | undefined
  readonly compileMs?: number | undefined
  readonly prefillMs?: number | undefined
  readonly decodeMs?: number | undefined
  readonly e2eMs?: number | undefined
  readonly prefillTokPerSec?: number | undefined
  readonly decodeTokPerSec?: number | undefined
  readonly roundP50Ms?: number | undefined
  readonly roundP95Ms?: number | undefined
  readonly draftMs?: number | undefined
  readonly verificationMs?: number | undefined
  readonly acceptedTokens?: number | undefined
  readonly proposedTokens?: number | undefined
  readonly acceptanceRate?: number | undefined
  readonly speculativeRounds?: number | undefined
  readonly ordinaryRounds?: number | undefined
  readonly targetPoolHighWaterBlocks?: number | undefined
  readonly proposerPoolHighWaterBlocks?: number | undefined
  readonly rssBytes?: number | undefined
  readonly outputHash?: string | undefined
  readonly seed: number
  readonly temperature: number
  readonly topK: number
  readonly topP: number
}

// ---------------------------------------------------------------------------
// Deterministic prompt construction
// ---------------------------------------------------------------------------

const WORDS = [
  "system",
  "harbor",
  "signal",
  "meadow",
  "copper",
  "lantern",
  "orbit",
  "timber",
  "vector",
  "cascade",
  "archive",
  "willow",
  "granite",
  "compass",
  "ember",
  "tunnel",
  "fabric",
  "horizon",
  "kernel",
  "prairie",
  "quartz",
  "riddle",
  "summit",
  "tangle",
  "umbrella",
  "voyage",
  "whisper",
  "yardstick",
  "zephyr",
  "anchor",
  "boulder",
  "cinder",
  "delta",
  "engine",
  "falcon",
  "glacier",
  "helix",
  "island",
  "jungle",
  "karma",
  "lattice",
  "magnet",
  "nebula",
  "onyx",
  "pillar",
  "quiver",
  "raptor",
  "saddle",
  "tempest",
  "utopia",
  "valley",
  "wander",
  "xylem",
  "yonder",
  "zenith",
  "atlas",
  "bridge",
  "castle",
  "dagger",
  "eclipse",
  "fountain",
  "garden",
  "hammer",
  "ivory",
  "jacket",
  "kettle",
  "ladder",
  "mirror",
  "needle",
  "orchard",
  "palace",
  "quarry",
  "river",
  "stone",
  "tower",
  "uncle",
  "violet",
  "window",
  "yellow",
  "zebra",
  "apple",
  "basket",
  "candle",
  "dragon",
  "eagle",
  "forest",
  "guitar",
  "hotel",
  "ink",
  "jewel",
  "knife",
  "lemon",
  "market",
  "north",
  "ocean",
  "piano",
  "queen",
  "rocket",
  "silver",
  "tiger",
  "uniform",
  "village",
  "wagon",
  "xenon",
  "youth",
  "zero",
  "amber",
  "brave",
  "cloud",
  "dance",
  "earth",
  "flame",
  "grace",
  "heart",
  "iron",
  "jade",
  "king",
  "light",
  "mouse",
  "noble",
  "olive",
  "pearl",
  "quiet",
  "rose",
  "storm",
  "truth",
  "unity",
  "voice"
] as const

// Single-token suffix candidates tried when a small exact gap remains.
const FILLERS = [".", ",", "!", "?", ";", ":", "\n", " the", " a", " and", " of", " to", " in", " is"] as const

const mulberry32 = (seed: number): () => number => {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let value = state
    value = Math.imul(value ^ (value >>> 15), value | 1)
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61)
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296
  }
}

interface Prompt {
  /** BOS-less canonical text handed to llama.cpp. */
  readonly text: string
  /** Exact `context - 1` token ids; the caller prepends BOS for effect-torch. */
  readonly ids: Uint32Array
}

const encodePlain = (
  tokenizer: Tokenizers.Tokenizer,
  text: string
): Effect.Effect<Uint32Array, Tokenizers.TokenizerError> =>
  Effect.map(tokenizer.encode(text, { addSpecialTokens: false }), (ids) => ids.data)

/** Exported for smoke tests; the harness entrypoint stays guarded below. */
export const buildPrompt = (
  tokenizer: Tokenizers.Tokenizer,
  seed: number,
  caseId: number,
  context: number
): Effect.Effect<Prompt, Tokenizers.TokenizerError> =>
  Effect.gen(function*() {
    // One id is reserved for the explicit BOS prepended by the caller.
    const target = context - 1
    if (target < 8) {
      return yield* Effect.die(new Error(`context ${context} leaves only ${target} prompt tokens`))
    }
    const random = mulberry32((seed ^ Math.imul(caseId + 1, 0x9e3779b9)) >>> 0)
    // The unique per-case prefix defeats prefix-cache reuse across runs.
    let text = `Muse Glimmer benchmark case ${caseId} context ${context}.`
    let ids = yield* encodePlain(tokenizer, text)
    for (let attempt = 0; attempt < 8192 && ids.length !== target; attempt++) {
      if (ids.length > target) {
        const cut = text.lastIndexOf(" ")
        if (cut <= 0) {
          return yield* Effect.die(new Error(`prompt case ${caseId}: cannot trim ${ids.length} tokens to ${target}`))
        }
        text = text.slice(0, cut)
        ids = yield* encodePlain(tokenizer, text)
        continue
      }
      const gap = target - ids.length
      let placed = false
      if (gap <= 8) {
        for (const filler of FILLERS) {
          const candidate = text + filler
          const candidateIds = yield* encodePlain(tokenizer, candidate)
          if (candidateIds.length === target) {
            text = candidate
            ids = candidateIds
            placed = true
            break
          }
        }
      }
      if (!placed && ids.length !== target) {
        text += ` ${WORDS[Math.floor(random() * WORDS.length)]}`
        ids = yield* encodePlain(tokenizer, text)
      }
    }
    if (ids.length !== target) {
      return yield* Effect.die(
        new Error(`prompt case ${caseId}: produced ${ids.length} tokens, wanted exactly ${target}`)
      )
    }
    // Canonicalize the text through decode and prove the round-trip is stable,
    // so llama.cpp tokenizes the received text into the same ids.
    const roundTripText = yield* tokenizer.decode(ids)
    const roundTripIds = yield* encodePlain(tokenizer, roundTripText)
    const stable = roundTripIds.length === ids.length && roundTripIds.every((id, index) => id === ids[index])
    if (!stable) {
      return yield* Effect.die(
        new Error(
          `prompt case ${caseId}: decode/encode round-trip changed ${ids.length} tokens to ${roundTripIds.length}`
        )
      )
    }
    return { text: roundTripText, ids }
  })

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

const percentile = (sorted: ReadonlyArray<number>, quantile: number): number | undefined =>
  sorted.length === 0 ? undefined : sorted[Math.min(sorted.length - 1, Math.floor(quantile * sorted.length))]

const fnv1a = (tokens: ReadonlyArray<number>): string => {
  let hash = 0x811c9dc5
  for (const token of tokens) {
    for (let shift = 0; shift < 32; shift += 8) {
      hash ^= (token >>> shift) & 0xff
      hash = Math.imul(hash, 0x01000193)
    }
  }
  return (hash >>> 0).toString(16).padStart(8, "0")
}

const metadataInt = (metadata: ReadonlyMap<string, unknown>, key: string): number => {
  const value = metadata.get(key)
  if (typeof value !== "number" || !Number.isSafeInteger(value)) {
    return fail(`GGUF metadata ${key} must be an integer`)
  }
  return value
}

// ---------------------------------------------------------------------------
// effect-torch suite
// ---------------------------------------------------------------------------

interface EffectCaseInput {
  readonly program: Model.InferenceProgram
  readonly tokenizer: Tokenizers.Tokenizer
  readonly bosTokenId: number
  readonly mode: Mode
  readonly context: number
  readonly run: number
  readonly caseId: number
  readonly loadMs: number
  readonly compileMs: number
  readonly config: Config
}

const runEffectCase = (
  input: EffectCaseInput
): Effect.Effect<BenchRecord, unknown, Runtime.Runtime> =>
  Effect.gen(function*() {
    const { config, program } = input
    const prompt = yield* buildPrompt(input.tokenizer, config.seed, input.caseId, input.context)
    const allIds = new Uint32Array(input.context)
    allIds[0] = input.bosTokenId
    allIds.set(prompt.ids, 1)
    const generation = yield* program.generation()
    return yield* Effect.gen(function*() {
      const promptTensor = yield* Tensor.fromTypedArray(allIds, [1, allIds.length])
      const before = yield* program.diagnostics()
      const prefillStarted = performance.now()
      let pages = yield* generation.add([{ prompt: promptTensor, maxTokens: config.maxNew }])
      const prefillMs = performance.now() - prefillStarted
      const addTokens = pages.reduce((total, page) => total + page.tokens.length, 0)
      const outputTokens: Array<number> = pages.flatMap((page) => [...page.tokens])
      const roundTimes: Array<number> = []
      let decodeMs = 0
      let stopReason = pages[0]?.stopReason
      while (stopReason === undefined && outputTokens.length < config.maxNew) {
        const stepStarted = performance.now()
        pages = yield* generation.step(pages.map(({ seq }) => ({ seq })))
        const stepMs = performance.now() - stepStarted
        roundTimes.push(stepMs)
        decodeMs += stepMs
        for (const page of pages) outputTokens.push(...page.tokens)
        stopReason = pages[0]?.stopReason
      }
      const after = yield* program.diagnostics()
      const decodeTokens = outputTokens.length - addTokens
      const sortedRounds = [...roundTimes].sort((left, right) => left - right)
      const accepted = Number(after.acceptedTokens - before.acceptedTokens)
      const proposed = Number(after.proposedTokens - before.proposedTokens)
      const speculativeRounds = Number(after.speculativeRounds - before.speculativeRounds)
      const ordinaryRounds = Number(after.ordinaryRounds - before.ordinaryRounds)
      const e2eMs = prefillMs + decodeMs
      return {
        timestamp: new Date().toISOString(),
        engine: "effect" as const,
        mode: input.mode,
        context: input.context,
        run: input.run,
        requestedTokens: config.maxNew,
        generatedTokens: outputTokens.length,
        loadMs: input.loadMs,
        compileMs: input.compileMs,
        prefillMs,
        decodeMs,
        e2eMs,
        prefillTokPerSec: input.context / (prefillMs / 1000),
        decodeTokPerSec: decodeTokens > 0 && decodeMs > 0 ? decodeTokens / (decodeMs / 1000) : undefined,
        roundP50Ms: percentile(sortedRounds, 0.5),
        roundP95Ms: percentile(sortedRounds, 0.95),
        ...(input.mode === "dflash"
          ? {
            acceptedTokens: accepted,
            proposedTokens: proposed,
            acceptanceRate: proposed > 0 ? accepted / proposed : undefined,
            speculativeRounds,
            ordinaryRounds,
            draftMs: Number(after.draftNanos - before.draftNanos) / 1e6,
            verificationMs: Number(after.verificationNanos - before.verificationNanos) / 1e6
          }
          : {}),
        targetPoolHighWaterBlocks: Number(after.targetPoolHighWaterBlocks),
        proposerPoolHighWaterBlocks: after.proposerPoolHighWaterBlocks === undefined
          ? undefined
          : Number(after.proposerPoolHighWaterBlocks),
        rssBytes: process.memoryUsage().rss,
        outputHash: fnv1a(outputTokens),
        seed: config.seed,
        temperature: config.temperature,
        topK: config.topK,
        topP: config.topP
      } satisfies BenchRecord
    }).pipe(Effect.ensuring(Effect.ignore(generation.close())))
  })

const effectSuite = (
  config: Config,
  records: Array<BenchRecord>
): Effect.Effect<void, unknown, Runtime.Runtime> =>
  Effect.gen(function*() {
    let caseId = 0
    const tokenizer = yield* Tokenizers.fromFile(config.tokenizerPath, {
      ...Tokenizers.strictConfig,
      specialTokens: "Always"
    })
    const loadStarted = performance.now()
    const loaded = yield* MuseGlimmer.loadGGUF(config.modelPath)
    const loadMs = performance.now() - loadStarted
    process.stderr.write(`loaded model: ${(loadMs / 1000).toFixed(2)}s\n`)
    const bosTokenId = metadataInt(loaded.metadata, "tokenizer.ggml.bos_token_id")
    const draft = config.modes.includes("dflash") ? yield* DFlash.loadGGUF(config.draftPath) : undefined
    if (draft !== undefined) process.stderr.write(`loaded DFlash draft (maxDraftTokens=${draft.maxDraftTokens})\n`)

    for (const mode of config.modes) {
      const compileStarted = performance.now()
      const program = yield* Model.inference(loaded.model, loaded.params, {
        maxTokens: config.maxTokens,
        blockSize: config.blockSize,
        kvDtype: "f16",
        prefillChunks: config.prefillChunks,
        batchSize: config.batchSize,
        sampling: {
          temperature: config.temperature,
          topK: config.topK,
          topP: config.topP,
          seed: config.seed
        },
        ...(mode === "dflash" && draft !== undefined
          ? {
            speculation: {
              proposer: draft.artifact,
              maxDraftTokens: Math.min(config.draftTokens, draft.maxDraftTokens)
            }
          }
          : {})
      })
      const compileMs = performance.now() - compileStarted
      process.stderr.write(`compiled ${mode} inference: ${(compileMs / 1000).toFixed(2)}s\n`)
      for (const context of config.contexts) {
        if (caseId > 0 && config.cooldownMs > 0) {
          yield* Effect.sleep(Duration.millis(config.cooldownMs))
        }
        for (let warmup = 0; warmup < config.warmup; warmup++) {
          yield* runEffectCase({
            program,
            tokenizer,
            bosTokenId,
            mode,
            context,
            run: -1 - warmup,
            caseId: caseId++,
            loadMs,
            compileMs,
            config
          })
          process.stderr.write(`warmup ${warmup + 1}/${config.warmup} done: ${mode} context=${context}\n`)
          if (config.cooldownMs > 0) yield* Effect.sleep(Duration.millis(config.cooldownMs))
        }
        for (let run = 0; run < config.runs; run++) {
          const record = yield* runEffectCase({
            program,
            tokenizer,
            bosTokenId,
            mode,
            context,
            run,
            caseId: caseId++,
            loadMs,
            compileMs,
            config
          })
          records.push(record)
        }
      }
    }

    // Model.inference retains its own materialized parameter generation, so
    // the loader handles can be released after both modes are compiled.
    yield* Tensor.clearAll(loaded.params)
    if (draft !== undefined) yield* Tensor.clearAll(draft.params)
  })

// ---------------------------------------------------------------------------
// llama.cpp orchestration
// ---------------------------------------------------------------------------

interface CommandResult {
  readonly stdout: string
  readonly stderr: string
  readonly elapsedMs: number
}

const runCommand = async (command: string, args: ReadonlyArray<string>): Promise<CommandResult> => {
  const started = performance.now()
  const { stdout, stderr } = await execFileAsync(command, [...args], { maxBuffer: 64 * 1024 * 1024 })
  return { stdout, stderr, elapsedMs: performance.now() - started }
}

const metalArgs = (draft: boolean): Array<string> =>
  draft ? ["-ngl", "99", "-ngld", "99", "-ctk", "f16", "-ctv", "f16", "-fa", "on"] : [
    "-ngl",
    "99",
    "-ctk",
    "f16",
    "-ctv",
    "f16",
    "-fa",
    "on"
  ]

const samplingArgs = (config: Config): Array<string> => [
  "--temp",
  String(config.temperature),
  "--top-k",
  String(config.temperature === 0 ? 1 : config.topK),
  "--top-p",
  String(config.topP),
  "--seed",
  String(config.seed)
]

const cooldown = async (config: Config): Promise<void> => {
  if (config.cooldownMs > 0) await new Promise((resolve) => setTimeout(resolve, config.cooldownMs))
}

const numberAt = (text: string, pattern: RegExp, group: number): number | undefined => {
  const match = pattern.exec(text)
  if (match === null) return undefined
  const value = Number(match[group])
  return Number.isFinite(value) ? value : undefined
}

const runLlamaBench = async (config: Config, binary: string, records: Array<BenchRecord>): Promise<void> => {
  const common = [
    "-m",
    config.modelPath,
    "-r",
    String(config.runs),
    ...metalArgs(false),
    "--delay",
    String(config.cooldownMs / 1000),
    "-o",
    "jsonl"
  ]
  const commands = [
    // Prompt-processing sweep at each requested context.
    [...common, "-p", config.contexts.join(","), "-n", "0"],
    // Decode sweep with the KV cache pre-populated to each requested depth.
    [...common, "-p", "0", "-n", String(config.maxNew), "-d", config.contexts.join(",")]
  ]
  for (const args of commands) {
    process.stderr.write(`running llama-bench: ${args.join(" ")}\n`)
    const { stdout, stderr } = await runCommand(binary, args)
    const lines = stdout.split("\n").filter((line) => line.trim().startsWith("{"))
    if (lines.length === 0) {
      process.stderr.write(
        `llama-bench produced no jsonl rows; stderr tail:\n${stderr.split("\n").slice(-5).join("\n")}\n`
      )
      continue
    }
    for (const line of lines) {
      const row = JSON.parse(line) as Record<string, unknown>
      const nPrompt = Number(row.n_prompt ?? 0)
      const nGen = Number(row.n_gen ?? 0)
      const nDepth = Number(row.n_depth ?? 0)
      const avgNs = Number(row.avg_ns ?? NaN)
      const avgMs = Number.isFinite(avgNs) ? avgNs / 1e6 : undefined
      const tokens = nGen > 0 ? nGen : nPrompt
      const avgTs = typeof row.avg_ts === "number"
        ? row.avg_ts
        : avgMs !== undefined && avgMs > 0
        ? tokens / (avgMs / 1000)
        : undefined
      const base = {
        timestamp: new Date().toISOString(),
        engine: "llama-bench" as const,
        mode: "ordinary" as const,
        context: nPrompt > 0 ? nPrompt : nDepth,
        run: 0,
        requestedTokens: nGen > 0 ? nGen : config.maxNew,
        seed: config.seed,
        temperature: config.temperature,
        topK: config.topK,
        topP: config.topP
      }
      if (nGen > 0) {
        records.push({
          ...base,
          generatedTokens: nGen,
          decodeMs: avgMs,
          decodeTokPerSec: avgTs
        })
      } else {
        records.push({ ...base, prefillMs: avgMs, prefillTokPerSec: avgTs })
      }
    }
    process.stderr.write(`llama-bench: parsed ${lines.length} rows\n`)
  }
}

const LLAMA_PROMPT_PERF =
  /prompt eval time\s*=\s*([\d.]+) ms\s*\/\s*(\d+) tokens\s*\(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second\s*\)/
const LLAMA_EVAL_PERF =
  /^\S*:\s*eval time\s*=\s*([\d.]+) ms\s*\/\s*(\d+) (?:runs|tokens)\s*\(\s*[\d.]+ ms per (?:token|run),\s*([\d.]+) tokens per second\s*\)/m
const LLAMA_SIMPLE_PERF = /\[\s*Prompt:\s*([\d.]+) t\/s\s*\|\s*Generation:\s*([\d.]+) t\/s\s*\]/
const LLAMA_LOAD = /load time\s*=\s*([\d.]+) ms/
const SPEC_ENCODED = /encoded\s+(\d+)\s+tokens?\s+in\s+([\d.]+)\s*seconds?,\s*speed:\s*([\d.]+)\s*t\/s/i
const SPEC_DECODED = /decoded\s+(\d+)\s+tokens?\s+in\s+([\d.]+)\s*seconds?,\s*speed:\s*([\d.]+)\s*t\/s/i
const SPEC_ACCEPTED = /n_accept\s*=\s*(\d+)/
const SPEC_DRAFTED = /n_drafted\s*=\s*(\d+)/

const runLlamaCliE2e = async (
  config: Config,
  binary: string,
  tokenizer: Tokenizers.Tokenizer,
  records: Array<BenchRecord>
): Promise<void> => {
  for (const context of config.llamaE2eContexts) {
    if (context !== config.llamaE2eContexts[0]) await cooldown(config)
    const prompt = await Effect.runPromise(buildPrompt(tokenizer, config.seed, 100_000 + context, context))
    const args = [
      "-m",
      config.modelPath,
      "--prompt",
      prompt.text,
      "-n",
      String(config.maxNew),
      "-c",
      String(config.maxTokens),
      "-b",
      String(config.maxTokens),
      ...metalArgs(false),
      ...samplingArgs(config),
      "--single-turn",
      "--simple-io",
      "--no-display-prompt",
      "--color",
      "off",
      "--no-warmup",
      "--perf"
    ]
    process.stderr.write(`running llama-cli e2e context=${context}\n`)
    const { stdout, stderr, elapsedMs } = await runCommand(binary, args)
    const combined = `${stdout}\n${stderr}`
    let prefillMs = numberAt(combined, LLAMA_PROMPT_PERF, 1)
    let promptTokens = numberAt(combined, LLAMA_PROMPT_PERF, 2)
    let prefillTokPerSec = numberAt(combined, LLAMA_PROMPT_PERF, 3)
    let decodeMs = numberAt(combined, LLAMA_EVAL_PERF, 1)
    let generatedTokens = numberAt(combined, LLAMA_EVAL_PERF, 2)
    let decodeTokPerSec = numberAt(combined, LLAMA_EVAL_PERF, 3)
    const simplePrefill = numberAt(combined, LLAMA_SIMPLE_PERF, 1)
    const simpleDecode = numberAt(combined, LLAMA_SIMPLE_PERF, 2)
    if (prefillTokPerSec === undefined && simplePrefill !== undefined) {
      prefillTokPerSec = simplePrefill
      prefillMs = context / simplePrefill * 1000
      promptTokens = context
    }
    if (decodeTokPerSec === undefined && simpleDecode !== undefined) {
      decodeTokPerSec = simpleDecode
      decodeMs = config.maxNew / simpleDecode * 1000
      generatedTokens = config.maxNew
    }
    const loadMs = numberAt(combined, LLAMA_LOAD, 1)
    if (decodeMs === undefined) {
      process.stderr.write(`llama-cli context=${context}: could not parse eval timing\n`)
    }
    if (promptTokens !== undefined && promptTokens !== context) {
      process.stderr.write(`llama-cli context=${context}: prompt evaluated ${promptTokens} tokens, want ${context}\n`)
    }
    records.push({
      timestamp: new Date().toISOString(),
      engine: "llama-cli",
      mode: "ordinary",
      context,
      run: 0,
      requestedTokens: config.maxNew,
      generatedTokens,
      loadMs,
      prefillMs,
      decodeMs,
      e2eMs: prefillMs !== undefined && decodeMs !== undefined ? prefillMs + decodeMs : elapsedMs,
      prefillTokPerSec,
      decodeTokPerSec,
      seed: config.seed,
      temperature: config.temperature,
      topK: config.topK,
      topP: config.topP
    })
  }
}

const runLlamaSpeculativeE2e = async (
  config: Config,
  binary: string,
  tokenizer: Tokenizers.Tokenizer,
  records: Array<BenchRecord>
): Promise<void> => {
  for (const context of config.llamaE2eContexts) {
    if (context !== config.llamaE2eContexts[0]) await cooldown(config)
    const prompt = await Effect.runPromise(buildPrompt(tokenizer, config.seed, 200_000 + context, context))
    const args = [
      "-m",
      config.modelPath,
      "-md",
      config.draftPath,
      "--prompt",
      prompt.text,
      "-n",
      String(config.maxNew),
      "-c",
      String(config.maxTokens),
      "-b",
      String(config.maxTokens),
      ...metalArgs(true),
      ...samplingArgs(config),
      "--spec-type",
      "draft-dflash",
      "--spec-draft-n-max",
      String(config.draftTokens)
    ]
    process.stderr.write(`running llama-speculative-simple e2e context=${context}\n`)
    const { stdout, stderr, elapsedMs } = await runCommand(binary, args)
    const combined = `${stdout}\n${stderr}`
    const promptTokens = numberAt(combined, SPEC_ENCODED, 1)
    const prefillSeconds = numberAt(combined, SPEC_ENCODED, 2)
    const prefillTokPerSec = numberAt(combined, SPEC_ENCODED, 3)
    const generatedTokens = numberAt(combined, SPEC_DECODED, 1)
    const decodeSeconds = numberAt(combined, SPEC_DECODED, 2)
    const decodeTokPerSec = numberAt(combined, SPEC_DECODED, 3)
    const acceptedTokens = numberAt(combined, SPEC_ACCEPTED, 1)
    const proposedTokens = numberAt(combined, SPEC_DRAFTED, 1)
    if (decodeSeconds === undefined) {
      process.stderr.write(`llama-speculative-simple context=${context}: could not parse decode timing\n`)
    }
    records.push({
      timestamp: new Date().toISOString(),
      engine: "llama-speculative-simple",
      mode: "dflash",
      context: promptTokens ?? context,
      run: 0,
      requestedTokens: config.maxNew,
      generatedTokens,
      prefillMs: prefillSeconds === undefined ? undefined : prefillSeconds * 1000,
      decodeMs: decodeSeconds === undefined ? undefined : decodeSeconds * 1000,
      e2eMs: prefillSeconds !== undefined && decodeSeconds !== undefined
        ? (prefillSeconds + decodeSeconds) * 1000
        : elapsedMs,
      prefillTokPerSec,
      decodeTokPerSec,
      acceptedTokens,
      proposedTokens,
      acceptanceRate: acceptedTokens !== undefined && proposedTokens !== undefined && proposedTokens > 0
        ? acceptedTokens / proposedTokens
        : undefined,
      seed: config.seed,
      temperature: config.temperature,
      topK: config.topK,
      topP: config.topP
    })
  }
}

const llamaSuite = async (config: Config, records: Array<BenchRecord>): Promise<void> => {
  if (config.llamaBin === undefined) {
    process.stderr.write("LLAMA_CPP_BIN is not set; skipping llama.cpp orchestration\n")
    return
  }
  const binary = (name: string): string | undefined => {
    const candidate = path.join(config.llamaBin!, name)
    if (!fs.existsSync(candidate)) {
      process.stderr.write(`missing ${candidate}; skipping\n`)
      return undefined
    }
    return candidate
  }
  const bench = binary("llama-bench")
  const cli = binary("llama-cli")
  const speculative = binary("llama-speculative-simple")
  let tokenizer: Tokenizers.Tokenizer | undefined
  const getTokenizer = async (): Promise<Tokenizers.Tokenizer> => {
    if (tokenizer === undefined) {
      tokenizer = await Effect.runPromise(
        Tokenizers.fromFile(config.tokenizerPath, { ...Tokenizers.strictConfig, specialTokens: "Always" })
      )
    }
    return tokenizer
  }
  if (bench !== undefined && config.modes.includes("ordinary")) {
    await runLlamaBench(config, bench, records)
    await cooldown(config)
  }
  if (cli !== undefined && config.modes.includes("ordinary")) {
    await runLlamaCliE2e(config, cli, await getTokenizer(), records)
    await cooldown(config)
  }
  if (speculative !== undefined && config.modes.includes("dflash")) {
    await runLlamaSpeculativeE2e(config, speculative, await getTokenizer(), records)
  }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

const fixed = (value: number | undefined, digits: number, width: number): string =>
  (value === undefined ? "-" : value.toFixed(digits)).padStart(width)

const printRow = (record: BenchRecord): void => {
  const acceptance = record.acceptanceRate === undefined ? "-" : `${(record.acceptanceRate * 100).toFixed(1)}%`
  const draftPerRound = record.draftMs === undefined || record.speculativeRounds === undefined
      || record.speculativeRounds === 0
    ? undefined
    : record.draftMs / record.speculativeRounds
  const verifyPerRound = record.verificationMs === undefined || record.speculativeRounds === undefined
      || record.speculativeRounds === 0
    ? undefined
    : record.verificationMs / record.speculativeRounds
  const ordinary = record.mode === "dflash" && record.ordinaryRounds !== undefined && record.ordinaryRounds > 0
    ? ` ordinaryRounds=${record.ordinaryRounds}`
    : ""
  process.stdout.write(
    `${record.engine.padEnd(24)} ${record.mode.padEnd(8)} ctx=${String(record.context).padStart(4)} ` +
      `run=${String(record.run).padStart(2)} gen=${String(record.generatedTokens ?? "-").padStart(4)} ` +
      `prefill=${fixed(record.prefillMs, 1, 9)} ms decode=${fixed(record.decodeMs, 1, 9)} ms ` +
      `e2e=${fixed(record.e2eMs, 1, 9)} ms tok/s=${fixed(record.decodeTokPerSec, 2, 7)} ` +
      `p50=${fixed(record.roundP50Ms, 2, 7)} ms p95=${fixed(record.roundP95Ms, 2, 7)} ms draft=${
        fixed(draftPerRound, 2, 7)
      } ms verify=${fixed(verifyPerRound, 2, 7)} ms accept=${acceptance}${ordinary}\n`
  )
}

const printTable = (records: ReadonlyArray<BenchRecord>): void => {
  if (records.length === 0) return
  process.stdout.write("\n")
  for (const record of records) printRow(record)
}

const median = (values: ReadonlyArray<number>): number | undefined => {
  if (values.length === 0) return undefined
  const sorted = [...values].sort((left, right) => left - right)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[middle - 1]! + sorted[middle]!) / 2 : sorted[middle]
}

const printSummary = (records: ReadonlyArray<BenchRecord>): void => {
  const groups = new Map<string, Array<BenchRecord>>()
  for (const record of records) {
    const key = `${record.engine}\0${record.mode}\0${record.context}`
    groups.set(key, [...(groups.get(key) ?? []), record])
  }
  if (groups.size === 0) return
  process.stdout.write("\nsummary (median)\n")
  for (const group of groups.values()) {
    const first = group[0]!
    const prefill = median(group.flatMap((record) => record.prefillTokPerSec ?? []))
    const decode = median(group.flatMap((record) => record.decodeTokPerSec ?? []))
    const acceptance = median(group.flatMap((record) => record.acceptanceRate ?? []))
    const draft = median(
      group.flatMap((record) =>
        record.draftMs === undefined || record.speculativeRounds === undefined || record.speculativeRounds === 0
          ? []
          : record.draftMs / record.speculativeRounds
      )
    )
    const verify = median(
      group.flatMap((record) =>
        record.verificationMs === undefined || record.speculativeRounds === undefined || record.speculativeRounds === 0
          ? []
          : record.verificationMs / record.speculativeRounds
      )
    )
    process.stdout.write(
      `${first.engine.padEnd(24)} ${first.mode.padEnd(8)} ctx=${String(first.context).padStart(4)} ` +
        `prefill=${fixed(prefill, 2, 8)} tok/s decode=${fixed(decode, 2, 8)} tok/s ` +
        `draft=${fixed(draft, 2, 7)} ms verify=${fixed(verify, 2, 7)} ms ` +
        `accept=${acceptance === undefined ? "-" : `${(acceptance * 100).toFixed(1)}%`}\n`
    )
  }
}

const writeOutput = (config: Config, records: ReadonlyArray<BenchRecord>): void => {
  fs.mkdirSync(path.dirname(config.output), { recursive: true })
  const body = records.map((record) => JSON.stringify(record)).join("\n") + (records.length > 0 ? "\n" : "")
  fs.writeFileSync(config.output, body)
  process.stderr.write(`wrote ${records.length} records to ${config.output}\n`)
}

// ---------------------------------------------------------------------------
// Entrypoint
// ---------------------------------------------------------------------------

const main = async (): Promise<void> => {
  const config = loadConfig()
  process.stderr.write(
    `engine=${config.engine} modes=${config.modes.join(",")} contexts=${config.contexts.join(",")} ` +
      `maxNew=${config.maxNew} runs=${config.runs} warmup=${config.warmup} seed=${config.seed} ` +
      `temperature=${config.temperature} topK=${config.topK} topP=${config.topP}\n`
  )
  const records: Array<BenchRecord> = []
  if (config.engine !== "llama") {
    await Effect.runPromise(Effect.provide(effectSuite(config, records), BackendApple.layer))
  }
  if (config.engine !== "effect") {
    await llamaSuite(config, records)
  }
  writeOutput(config, records)
  printTable(records)
  printSummary(records)
}

const isMain = process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href

if (isMain) {
  main().catch((error) => {
    process.stderr.write(`${String(error)}\n`)
    process.exitCode = 1
  })
}

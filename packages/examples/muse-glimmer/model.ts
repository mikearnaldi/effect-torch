// Backend-neutral quantized Muse-Glimmer chat inference with its official
// DFlash draft. MUSE_GLIMMER_* variables configure paths and generation.
import { Chat, Model, Tensor } from "@effect-torch/core"
import type { Runtime } from "@effect-torch/core"
import { MuseGlimmer } from "@effect-torch/core/models"
import { DFlash } from "@effect-torch/core/proposers"
import * as Tokenizers from "@effect-torch/tokenizers"
import { Config, Effect, Option, Schema, Stream } from "effect"
import path from "node:path"
import { fileURLToPath } from "node:url"

const directory = path.dirname(fileURLToPath(import.meta.url))
const defaultModelPath = path.join(directory, "../data/Muse-Glimmer-30B-UD-Q2_K_XL.gguf")
const defaultDraftPath = path.join(directory, "../data/dflash-Muse-Glimmer-30B-Q4_K_M.gguf")
const defaultTokenizerPath = path.join(directory, "../data/muse-glimmer-tokenizer.json")

const timed = <A, E, R>(label: string, effect: Effect.Effect<A, E, R>): Effect.Effect<A, E, R> =>
  Effect.gen(function*() {
    const start = performance.now()
    process.stderr.write(`${label}...\n`)
    const value = yield* effect
    process.stderr.write(`${label}: ${((performance.now() - start) / 1000).toFixed(2)}s\n`)
    return value
  })

const metadataNumber = (metadata: ReadonlyMap<string, unknown>, key: string): number | undefined => {
  return Option.getOrUndefined(Schema.decodeUnknownOption(Schema.Int)(metadata.get(key)))
}

export const inference = (prompt: string): Effect.Effect<void, unknown, Runtime.Runtime> =>
  Effect.gen(function*() {
    const maxNewTokens = yield* Config.option(
      Config.schema(Schema.Int.check(Schema.isGreaterThan(0)), "MUSE_GLIMMER_MAX_NEW_TOKENS")
    ).pipe(
      Config.map(Option.getOrUndefined)
    )

    const reasoningStrength = yield* Config.nonEmptyString("MUSE_GLIMMER_REASONING_STRENGTH").pipe(
      Config.withDefault("high")
    )

    const temperature = yield* Config.schema(
      Schema.Finite.check(Schema.isGreaterThanOrEqualTo(0)),
      "MUSE_GLIMMER_TEMPERATURE"
    ).pipe(
      Config.withDefault(0.7)
    )

    const topP = yield* Config.schema(
      Schema.Finite.check(Schema.isGreaterThan(0), Schema.isLessThanOrEqualTo(1)),
      "MUSE_GLIMMER_TOP_P"
    ).pipe(
      Config.withDefault(0.95)
    )

    const topK = yield* Config.schema(
      Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
      "MUSE_GLIMMER_TOP_K"
    ).pipe(
      Config.withDefault(40)
    )

    const seed = yield* Config.option(
      Config.schema(Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)), "MUSE_GLIMMER_SEED")
    ).pipe(
      Config.map(Option.getOrUndefined)
    )

    const diagnostics = yield* Config.boolean("MUSE_GLIMMER_DIAGNOSTICS").pipe(
      Config.withDefault(false)
    )

    const useDFlash = yield* Config.boolean("MUSE_GLIMMER_DFLASH").pipe(
      Config.withDefault(true)
    )

    const draftTokens = yield* Config.schema(
      Schema.Int.check(Schema.isGreaterThan(0), Schema.isLessThanOrEqualTo(15)),
      "MUSE_GLIMMER_DRAFT_TOKENS"
    ).pipe(
      Config.withDefault(5)
    )

    // The runtime splits prompts across these compiled prefill widths. It uses
    // the largest width that covers the remaining tokens. The smaller width
    // reduces zero padding for short prompts.
    const prefillChunks = [32, 64, 128, 256]

    const modelPath = yield* Config.nonEmptyString("MUSE_GLIMMER_MODEL_PATH").pipe(
      Config.withDefault(defaultModelPath)
    )
    const draftPath = yield* Config.nonEmptyString("MUSE_GLIMMER_DRAFT_PATH").pipe(
      Config.withDefault(defaultDraftPath)
    )
    const tokenizerPath = yield* Config.nonEmptyString("MUSE_GLIMMER_TOKENIZER_PATH").pipe(
      Config.withDefault(defaultTokenizerPath)
    )

    const tokenizer = yield* timed(
      "Loading tokenizer",
      Tokenizers.fromFile(tokenizerPath, {
        ...Tokenizers.strictConfig,
        specialTokens: "Always"
      })
    )

    const loaded = yield* timed("Loading model", MuseGlimmer.loadGGUF(modelPath))
    const draft = useDFlash ? yield* timed("Loading DFlash draft", DFlash.loadGGUF(draftPath)) : undefined

    const chatTemplate = Option.getOrUndefined(
      Schema.decodeUnknownOption(Schema.NonEmptyString)(loaded.metadata.get("tokenizer.chat_template"))
    )

    if (chatTemplate === undefined) {
      return yield* Effect.die(new Error("GGUF tokenizer.chat_template must be a non-empty string"))
    }

    const bosTokenId = metadataNumber(loaded.metadata, "tokenizer.ggml.bos_token_id")

    if (bosTokenId === undefined) {
      return yield* Effect.die(new Error("GGUF tokenizer.ggml.bos_token_id must be an integer"))
    }

    if (diagnostics) {
      const formats = new Map<string, { tensors: number; bytes: number }>()
      const tails = new Map<string, number>()
      for (const parameter of loaded.params) {
        const format = parameter.storage?.encoding ?? parameter.dtype.toUpperCase()
        const physicalShape = parameter.storage?.physicalShape ?? parameter.shape
        const bytesPerElement = parameter.storage === undefined && parameter.dtype === "f32" ? 4 : 1
        const bytes = physicalShape.reduce((total, dimension) => total * dimension, bytesPerElement)
        const current = formats.get(format) ?? { tensors: 0, bytes: 0 }
        current.tensors++
        current.bytes += bytes
        formats.set(format, current)
        if (parameter.storage !== undefined && parameter.shape.length === 2) {
          const rowsPerGroup = parameter.storage.encoding === "Q2_K"
            ? 8
            : parameter.storage.encoding === "Q5_K"
            ? 2
            : 4
          if (parameter.shape[0]! % rowsPerGroup !== 0) {
            tails.set(parameter.storage.encoding, (tails.get(parameter.storage.encoding) ?? 0) + 1)
          }
        }
      }
      for (const [format, summary] of [...formats].sort(([left], [right]) => left.localeCompare(right))) {
        process.stderr.write(`${format}: ${summary.tensors} tensors, ${(summary.bytes / 1e9).toFixed(3)} GB\n`)
      }
      process.stderr.write(`partial linear threadgroups: ${JSON.stringify(Object.fromEntries(tails))}\n`)
    }

    const currentDate = new Date().toISOString().slice(0, 10)

    const inference = yield* timed(
      "Compiling inference",
      Model.inference(loaded.model, loaded.params, {
        maxTokens: 4096,
        blockSize: 16,
        kvDtype: "f16",
        prefillChunks,
        batchSize: 1,
        speculation: draft === undefined
          ? undefined
          : { proposer: draft.artifact, maxDraftTokens: Math.min(draftTokens, draft.maxDraftTokens) }
      })
    )

    // Model.inference creates and retains an immutable parameter generation.
    // Release the GGUF loader's handles after compilation.
    yield* Tensor.clearAll(loaded.params)
    if (draft !== undefined) yield* Tensor.clearAll(draft.params)

    let sawSegment = false

    yield* Stream.runForEach(
      Chat.stream({
        program: inference,
        tokenizer,
        template: chatTemplate,
        messages: [{ role: "user", content: prompt }],
        addGenerationPrompt: true,
        variables: {
          current_date: currentDate,
          reasoning_strength: reasoningStrength
        },
        bosTokenId,
        maxTokens: maxNewTokens,
        sampling: { temperature, topK, topP, seed }
      }),
      (event) =>
        Effect.sync(() => {
          switch (event._tag) {
            case "prefill": {
              process.stderr.write(`Prefilling ${event.tokens} tokens...\n`)
              process.stderr.write(`Prefilling ${event.tokens} tokens: ${(event.durationMs / 1000).toFixed(2)}s\n`)
              break
            }
            case "start": {
              sawSegment = true
              const label = event.segment.kind === "content"
                ? "response"
                : event.segment.kind === "reasoning"
                ? "reasoning"
                : event.segment.recipient ?? event.segment.role
              process.stdout.write(`[${label}]\n`)
              break
            }
            case "delta": {
              process.stdout.write(event.text)
              break
            }
            case "end": {
              process.stdout.write("\n")
              break
            }
            case "done": {
              const stats = event.result.stats
              process.stderr.write(
                `Decoding ${stats.generatedTokens} tokens: ${(stats.decodeMs / 1000).toFixed(2)}s (${
                  stats.decodeTokensPerSecond.toFixed(2)
                } tok/s)\n`
              )
              break
            }
          }
        })
    )
    if (diagnostics) {
      const stats = yield* inference.diagnostics()
      const acceptance = stats.proposedTokens === 0n
        ? 0
        : Number(stats.acceptedTokens) / Number(stats.proposedTokens)
      process.stderr.write(
        `speculation: ${stats.acceptedTokens}/${stats.proposedTokens} accepted (${(acceptance * 100).toFixed(1)}%), ` +
          `rounds ordinary=${stats.ordinaryRounds} speculative=${stats.speculativeRounds}, ` +
          `draft ${(Number(stats.draftNanos) / 1e9).toFixed(2)}s, verify ${
            (Number(stats.verificationNanos) / 1e9).toFixed(2)
          }s\n`
      )
    }
    if (!sawSegment) process.stdout.write("\n")
  })

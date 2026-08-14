// Quantized Muse-Glimmer chat inference. Model and tokenizer paths are resolved
// relative to this module, while optional generation/config controls come from
// MUSE_GLIMMER_* environment variables and invalid configured values fail via
// Effect Config. Gguf.load uses the provided Registry to select the architecture
// implementation, validates its tensor catalog, and imports encoded weights on
// the selected backend. Chat.stream owns the generation session, renders the
// GGUF template, emits parsed reasoning/content segments incrementally, and
// closes state on completion, failure, or interruption. The artifact has a
// 4,096-token full-context pool, so prompt plus decode must fit even when the
// optional application-side max-new-token limit is omitted. Native sampling
// defaults to temperature 0.7, top-p 0.95, and top-k 40;
// MUSE_GLIMMER_TEMPERATURE=0 selects greedy decoding and MUSE_GLIMMER_SEED
// makes stochastic runs replayable.

import * as BackendApple from "@effect-torch/backend-apple-native"
import { Chat, Gguf, Model, Registry, Tensor } from "@effect-torch/core"
import * as Tokenizers from "@effect-torch/tokenizers"
import { NodeRuntime } from "@effect/platform-node"
import { Config, Effect, Option, Schema, Stream } from "effect"
import path from "node:path"
import { fileURLToPath } from "node:url"

const directory = path.dirname(fileURLToPath(import.meta.url))
const modelPath = path.join(directory, "../data/Muse-Glimmer-30B-UD-Q2_K_XL.gguf")
const tokenizerPath = path.join(directory, "../data/muse-glimmer-tokenizer.json")

const timed = <A, E, R>(label: string, effect: Effect.Effect<A, E, R>): Effect.Effect<A, E, R> =>
  Effect.gen(function*() {
    const start = performance.now()
    process.stderr.write(`${label}...\n`)
    const value = yield* effect
    process.stderr.write(`${label}: ${((performance.now() - start) / 1000).toFixed(2)}s\n`)
    return value
  })

const metadataNumber = (metadata: ReadonlyMap<string, unknown>, key: string): number | undefined => {
  const value = metadata.get(key)
  return typeof value === "number" && Number.isSafeInteger(value) ? value : undefined
}

const program = Effect.gen(function*() {
  const prompt = process.argv.slice(2).join(" ")

  if (prompt.length === 0) {
    yield* Effect.log("usage: pnpm tsx muse-glimmer/inference.ts <prompt>")
    return
  }

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

  const tokenizer = yield* timed(
    "Loading tokenizer",
    Tokenizers.fromFile(tokenizerPath, {
      ...Tokenizers.strictConfig,
      specialTokens: "Always"
    })
  )

  const loaded = yield* timed("Loading model", Gguf.load(modelPath))

  const chatTemplate = loaded.metadata.get("tokenizer.chat_template")

  if (typeof chatTemplate !== "string" || chatTemplate.length === 0) {
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

  const inference = yield* timed(
    "Compiling inference",
    Model.inference(loaded.model, loaded.params, {
      maxTokens: 4096,
      blockSize: 16,
      kvDtype: "f16",
      prefillChunk: 16,
      decodeBatch: 1
    })
  )

  // Model.inference materializes and retains its own immutable parameter
  // generation, so the GGUF loader's handles can be released after compilation.
  yield* Tensor.clearAll(loaded.params)

  let sawSegment = false

  yield* Stream.runForEach(
    Chat.stream({
      program: inference,
      tokenizer,
      template: chatTemplate,
      messages: [{ role: "user", content: prompt }],
      addGenerationPrompt: true,
      variables: {
        current_date: new Date().toISOString().slice(0, 10),
        reasoning_strength: reasoningStrength
      },
      bosTokenId,
      maxTokens: maxNewTokens,
      sampling: { temperature, topK, topP, ...(seed === undefined ? {} : { seed }) }
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
  if (!sawSegment) process.stdout.write("\n")
})

NodeRuntime.runMain(
  program.pipe(
    Effect.provide(Registry.layer),
    Effect.provide(BackendApple.layer)
  )
)

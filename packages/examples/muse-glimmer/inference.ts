import * as BackendApple from "@effect-torch/backend-apple-native"
import { Gguf, Model, Registry, Tensor } from "@effect-torch/core"
import * as Tokenizers from "@effect-torch/tokenizers"
import { NodeRuntime } from "@effect/platform-node"
import { Effect, Option } from "effect"
import path from "node:path"
import { fileURLToPath } from "node:url"

const directory = path.dirname(fileURLToPath(import.meta.url))
const modelPath = path.join(directory, "../data/Muse-Glimmer-30B-UD-Q2_K_XL.gguf")
const tokenizerPath = path.join(directory, "../data/muse-glimmer-tokenizer.json")
const maxNewTokens = Number(process.env.MUSE_GLIMMER_MAX_NEW_TOKENS ?? 64)

const timed = <A, E, R>(label: string, effect: Effect.Effect<A, E, R>): Effect.Effect<A, E, R> =>
  Effect.gen(function*() {
    const start = performance.now()
    process.stderr.write(`${label}...\n`)
    const value = yield* effect
    process.stderr.write(`${label}: ${((performance.now() - start) / 1000).toFixed(2)}s\n`)
    return value
  })

const chatPrompt = (prompt: string) =>
  `<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\n\nReasoning strength: high.\n\n# Valid recipients: "self", "user".<|eot|><|start|>user<|message|>${prompt}<|eot|><|start|>assistant`

const program = Effect.gen(function*() {
  const prompt = process.argv.slice(2).join(" ")
  if (prompt.length === 0) {
    yield* Effect.log("usage: pnpm tsx muse-glimmer/inference.ts <prompt>")
    return
  }
  if (!Number.isSafeInteger(maxNewTokens) || maxNewTokens <= 0) {
    return yield* Effect.die(new Error("MUSE_GLIMMER_MAX_NEW_TOKENS must be a positive integer"))
  }

  const tokenizer = yield* timed(
    "Loading tokenizer",
    Tokenizers.fromFile(tokenizerPath, {
      ...Tokenizers.strictConfig,
      specialTokens: "Always"
    })
  )
  const stopTokens = new Set([
    Option.getOrThrow(tokenizer.tokenToId("<|end_of_text|>")),
    Option.getOrThrow(tokenizer.tokenToId("<|eot|>"))
  ])
  const loaded = yield* timed("Loading model", Gguf.load(modelPath))
  if (process.env.MUSE_GLIMMER_DIAGNOSTICS !== undefined) {
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
  yield* Tensor.clearAll(loaded.params)
  const generation = yield* inference.generation()
  const encoded = yield* tokenizer.encode(chatPrompt(prompt))
  const entry = yield* timed(
    `Prefilling ${encoded.data.length} tokens`,
    generation.add(yield* Tensor.fromTypedArray(encoded.data, [1, encoded.data.length]))
  )

  let logits: Tensor.Concrete | undefined = entry.logits
  const generated: Array<number> = []
  let emitted = 0
  for (let step = 0; step < maxNewTokens; step++) {
    const current = logits!
    logits = undefined
    const values = yield* Tensor.toTypedArray(current)
    yield* Tensor.clear(current)
    let token = 0
    for (let index = 1; index < values.length; index++) {
      if (Number(values[index]) > Number(values[token])) token = index
    }
    if (stopTokens.has(token)) break
    generated.push(token)
    const text = yield* tokenizer.decode(generated)
    if (text.length > emitted) {
      process.stdout.write(text.slice(emitted))
      emitted = text.length
    }
    if (step + 1 < maxNewTokens) {
      const start = performance.now()
      const [next] = yield* generation.step([{ seq: entry.seq, token }])
      process.stderr.write(` [${(1000 / (performance.now() - start)).toFixed(2)} tok/s]`)
      logits = next
    }
  }
  if (logits !== undefined) yield* Tensor.clear(logits)
  yield* generation.close()
  process.stdout.write("\n")
})

NodeRuntime.runMain(program.pipe(Effect.provide(Registry.layer), Effect.provide(BackendApple.layer)))

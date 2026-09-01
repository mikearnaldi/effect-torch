import * as Tokenizer from "@effect-torch/tokenizers"
import { describe, expect, it } from "@effect/vitest"
import { Effect, Option } from "effect"
import * as fs from "node:fs"
import * as os from "node:os"
import * as path from "node:path"

const tmpdir = Effect.sync(() => fs.mkdtempSync(path.join(os.tmpdir(), "effect-torch-tok-")))

const corpusLines = [
  "the quick brown fox jumps over the lazy dog",
  "the lazy dog sleeps while the quick brown fox runs",
  "hello world hello tokenizer hello effect torch",
  "tokenizers turn text into tensors and tensors into models",
  "a library for building models needs a real data plane",
  "byte pair encoding merges the most frequent pairs first"
]

const corpusTexts = Array.from({ length: 20 }, () => corpusLines).flat()

const trainBpe = (
  config: Tokenizer.TokenizerConfig = Tokenizer.strictConfig,
  specialTokens: ReadonlyArray<string> = ["<|endoftext|>"]
) =>
  Tokenizer.train(
    {
      source: Tokenizer.trainTexts(corpusTexts),
      model: "BPE",
      vocabSize: 300,
      minFrequency: 2,
      specialTokens,
      progress: Tokenizer.trainProgressNone
    },
    config
  )

const numbers = (ids: Tokenizer.TokenIds): Effect.Effect<Array<number>> => Effect.succeed(Array.from(ids.data))

// These are native integration tests: model training and artifacts exercise the
// package-local addon, while batching policies belong to the TypeScript facade.
describe("Tokenizer", () => {
  describe("BPE", () => {
    it.effect("encodes to a [T] u32 tensor and decodes losslessly, including unicode", () =>
      Effect.gen(function*() {
        const tokenizer = yield* trainBpe()
        const text = "hello world — こんにちは 😁 café, naïve"
        const ids = yield* tokenizer.encode(text)
        expect(ids.dtype).toBe("u32")
        expect(ids.shape.length).toBe(1)
        expect(ids.shape[0]).toBeGreaterThan(0)
        const decoded = yield* tokenizer.decode(ids)
        expect(decoded).toBe(text)
      }))

    it.effect("encodeBatchConcat flattens ragged encodings in order, no padding", () =>
      Effect.gen(function*() {
        const tokenizer = yield* trainBpe()
        const texts = ["hello world", "the lazy dog"]
        const flat = yield* tokenizer.encodeBatchConcat(texts)
        expect(flat.dtype).toBe("u32")
        expect(flat.shape.length).toBe(1)
        const singles = yield* Effect.forEach(texts, (text) => Effect.flatMap(tokenizer.encode(text), numbers))
        expect(yield* numbers(flat)).toEqual(singles.flat())
      }))

    it.effect("decodeBatch round-trips raw id arrays; vocab lookups are Options", () =>
      Effect.gen(function*() {
        const tokenizer = yield* trainBpe()
        const texts = ["hello world", "the lazy dog"]
        const batch = yield* tokenizer.encodeBatch(texts)
        expect(batch.shape.length).toBe(2)
        expect(batch.shape[0]).toBe(2)
        const flat = yield* numbers(batch)
        const cols = batch.shape[1]
        const rows = [flat.slice(0, cols), flat.slice(cols)]
        const decoded = yield* tokenizer.decodeBatch(rows)
        expect(decoded).toEqual(texts)
        expect(Option.isSome(tokenizer.tokenToId("<|endoftext|>"))).toBe(true)
        const id = Option.getOrElse(tokenizer.tokenToId("<|endoftext|>"), () => -1)
        expect(tokenizer.idToToken(id)).toEqual(Option.some("<|endoftext|>"))
        expect(yield* tokenizer.decode([id], { skipSpecialTokens: true })).toBe("")
        expect(tokenizer.tokenToId("not a token")).toEqual(Option.none())
        expect(tokenizer.vocabSize).toBeGreaterThan(256)
      }))

    it.effect("incrementally decodes Unicode and skips special tokens", () =>
      Effect.gen(function*() {
        const tokenizer = yield* trainBpe()
        const encoded = yield* tokenizer.encode("hello cafe — こんにちは 😁")
        const special = Option.getOrThrow(tokenizer.tokenToId("<|endoftext|>"))
        const ids = Array.from(encoded.data)
        ids.splice(Math.floor(ids.length / 2), 0, special)
        const expected = yield* tokenizer.decode(ids, { skipSpecialTokens: true })
        const decoder = tokenizer.decodeStream({ skipSpecialTokens: true })
        let actual = ""
        for (const id of ids) {
          actual += (yield* decoder.step(id)) ?? ""
        }
        expect(actual).toBe(expected)
      }))

    it.effect("trains from files; save and fromFile/fromJson preserve encoding exactly", () =>
      Effect.gen(function*() {
        const dir = yield* tmpdir
        const corpusFile = path.join(dir, "corpus.txt")
        yield* Effect.sync(() => fs.writeFileSync(corpusFile, corpusTexts.join("\n")))
        const tokenizer = yield* Tokenizer.train(
          {
            source: Tokenizer.trainFiles([corpusFile]),
            model: "BPE",
            vocabSize: 300,
            minFrequency: 2,
            specialTokens: ["<|endoftext|>"],
            progress: Tokenizer.trainProgressNone
          },
          Tokenizer.strictConfig
        )
        const saved = path.join(dir, "tokenizer.json")
        yield* tokenizer.save(saved)
        const text = "tokenizers turn text into tensors 😁"
        const expected = yield* numbers(yield* tokenizer.encode(text))

        const fromFile = yield* Tokenizer.fromFile(saved, Tokenizer.strictConfig)
        expect(yield* numbers(yield* fromFile.encode(text))).toEqual(expected)

        const fromJson = yield* Tokenizer.fromJson(fs.readFileSync(saved, "utf8"), Tokenizer.strictConfig)
        expect(yield* numbers(yield* fromJson.encode(text))).toEqual(expected)
      }))
  })

  describe("padding and truncation", () => {
    const texts = ["hello world", "tokenizers turn text into tensors and tensors into models"]

    it.effect("Longest pads to the longest encoding with padId", () =>
      Effect.gen(function*() {
        const tokenizer = yield* trainBpe({
          padding: Tokenizer.paddingLongest(0),
          truncation: Tokenizer.truncationNone,
          specialTokens: "Never"
        })
        const batch = yield* tokenizer.encodeBatch(texts)
        expect(batch.shape.length).toBe(2)
        const flat = yield* numbers(batch)
        const [cols] = batch.shape.slice(1)
        const shortRow = flat.slice(0, cols)
        const longRow = flat.slice(cols)
        expect(shortRow.length).toBe(longRow.length)
        expect(shortRow.some((id, i) => id === 0 && longRow[i] !== 0)).toBe(true)
        expect(yield* tokenizer.decode(longRow)).toBe(texts[1])
      }))

    it.effect("MaxLength pads and truncation caps overlong encodings", () =>
      Effect.gen(function*() {
        const tokenizer = yield* trainBpe({
          padding: Tokenizer.paddingMaxLength(8, 0),
          truncation: Tokenizer.truncationMaxLength(8),
          specialTokens: "Never"
        })
        const batch = yield* tokenizer.encodeBatch(texts)
        expect(batch.shape).toEqual([2, 8])
      }))

    it.effect("MaxLength padding without truncation fails on overlong encodings", () =>
      Effect.gen(function*() {
        const tokenizer = yield* trainBpe({
          padding: Tokenizer.paddingMaxLength(4, 0),
          truncation: Tokenizer.truncationNone,
          specialTokens: "Never"
        })
        const error = yield* Effect.flip(tokenizer.encodeBatch(texts))
        expect(error.message).toContain("truncation")
      }))

    it.effect("padding None fails on ragged encodings and on empty batches", () =>
      Effect.gen(function*() {
        const tokenizer = yield* trainBpe()
        const ragged = yield* Effect.flip(tokenizer.encodeBatch(texts))
        expect(ragged.message).toContain("ragged")
        const empty = yield* Effect.flip(tokenizer.encodeBatch([]))
        expect(empty.message).toContain("at least one text")
      }))
  })

  describe("special token policy", () => {
    it.effect("Never tokenizes special strings as ordinary text; Always parses them", () =>
      Effect.gen(function*() {
        const dir = yield* tmpdir
        const saved = path.join(dir, "tokenizer.json")
        const tokenizer = yield* trainBpe()
        yield* tokenizer.save(saved)
        const specialId = Option.getOrNull(tokenizer.tokenToId("<|endoftext|>"))

        const text = "<|endoftext|> hello world"
        const never = yield* Tokenizer.fromFile(saved, {
          padding: Tokenizer.paddingNone,
          truncation: Tokenizer.truncationNone,
          specialTokens: "Never"
        })
        const neverIds = yield* numbers(yield* never.encode(text))
        expect(neverIds).not.toContain(specialId)
        expect(neverIds.length).toBeGreaterThan(2)

        const always = yield* Tokenizer.fromFile(saved, {
          padding: Tokenizer.paddingNone,
          truncation: Tokenizer.truncationNone,
          specialTokens: "Always"
        })
        const alwaysIds = yield* numbers(yield* always.encode(text))
        expect(alwaysIds).toContain(specialId)
      }))
  })

  describe("chat templates", () => {
    it.effect("renders Jinja messages, namespaces, filters, and generation prompts", () =>
      Effect.gen(function*() {
        const tokenizer = yield* trainBpe()
        const rendered = yield* tokenizer.applyChatTemplate(
          `{%- set ns = namespace(items=[]) -%}{%- for item in values -%}{%- set ns.items = ns.items + [item] -%}{%- endfor -%}{{ bos_token }}{% for message in messages %}<|start|>{{ message.role }}<|message|>{{ message.content }}{{ '<|eot|>' if loop.last else '<|eom|>' }}{% endfor %}{% if add_generation_prompt %}<|start|>assistant{% endif %} {{ ns.items | join(',') }} {{ values | tojson }}`,
          [
            { role: "user", content: "hello" },
            { role: "assistant", content: "hi" }
          ],
          {
            addGenerationPrompt: true,
            variables: {
              bos_token: "<BOS>",
              values: [1, "two"]
            }
          }
        )
        expect(rendered).toBe(
          `<BOS><|start|>user<|message|>hello<|eom|><|start|>assistant<|message|>hi<|eot|><|start|>assistant 1,two [1,"two"]`
        )
      }))
  })

  describe("training progress", () => {
    // No waiting is needed. The feed iterator posts the final (total, total)
    // event before the merge phase, so all progress
    // callbacks are queued on the event loop ahead of train's resolution.
    it.effect("reports throttled byte progress ending at (total, total)", () =>
      Effect.gen(function*() {
        const events: Array<[number, number]> = []
        const total = corpusTexts.reduce((sum, text) => sum + text.length, 0)
        yield* Tokenizer.train(
          {
            source: Tokenizer.trainTexts(corpusTexts),
            model: "BPE",
            vocabSize: 300,
            minFrequency: 2,
            specialTokens: [],
            progress: Tokenizer.trainProgressReport(128, (processed, total) =>
              Effect.sync(() => events.push([processed, total])))
          },
          Tokenizer.strictConfig
        )
        expect(events.length).toBeGreaterThan(0)
        for (const [processed, reported] of events) {
          expect(reported).toBe(total)
          expect(processed).toBeLessThanOrEqual(total)
        }
        expect(events.at(-1)).toEqual([total, total])
        const processed = events.map(([p]) =>
          p
        )
        expect([...processed].sort((a, b) => a - b)).toEqual(processed)
        expect(events.length).toBeLessThanOrEqual(Math.ceil(total / 128) + 1)
      }))

    it.effect("everyBytes 0 disables reporting entirely", () =>
      Effect.gen(function*() {
        const events: Array<[number, number]> = []
        yield* Tokenizer.train(
          {
            source: Tokenizer.trainTexts(corpusTexts),
            model: "BPE",
            vocabSize: 300,
            minFrequency: 2,
            specialTokens: [],
            progress: Tokenizer.trainProgressReport(0, (processed, total) =>
              Effect.sync(() => events.push([processed, total])))
          },
          Tokenizer.strictConfig
        )
        expect(events).toEqual([])
      }))

    it.effect("everyBytes bounds the report frequency", () =>
      Effect.gen(function*() {
        const events: Array<[number, number]> = []
        const total = corpusTexts.reduce((sum, text) => sum + text.length, 0)
        yield* Tokenizer.train(
          {
            source: Tokenizer.trainTexts(corpusTexts),
            model: "BPE",
            vocabSize: 300,
            minFrequency: 2,
            specialTokens: [],
            progress: Tokenizer.trainProgressReport(total, (processed, total) =>
              Effect.sync(() => events.push([processed, total])))
          },
          Tokenizer.strictConfig
        )
        expect(events).toEqual([[total, total]])
      }))
  })

  describe("model families", () => {
    const train = (model: Tokenizer.TrainModel, vocabSize: number) =>
      Tokenizer.train(
        {
          source: Tokenizer.trainTexts(corpusTexts),
          model,
          vocabSize,
          minFrequency: 2,
          specialTokens: [],
          progress: Tokenizer.trainProgressNone
        },
        Tokenizer.strictConfig
      )

    it.effect("WordPiece round-trips in-corpus text", () =>
      Effect.gen(function*() {
        const tokenizer = yield* train("WordPiece", 200)
        const text = "the quick brown fox jumps over the lazy dog"
        const decoded = yield* tokenizer.decode(yield* tokenizer.encode(text))
        expect(decoded).toBe(text)
      }))

    it.effect("Unigram round-trips in-corpus text", () =>
      Effect.gen(function*() {
        const tokenizer = yield* train("Unigram", 100)
        const text = "the quick brown fox"
        const decoded = yield* tokenizer.decode(yield* tokenizer.encode(text))
        expect(decoded).toBe(text)
      }))

    it.effect("Unigram byte fallback round-trips out-of-vocabulary text losslessly", () =>
      Effect.gen(function*() {
        const tokenizer = yield* train("Unigram", 100)
        expect(Option.getOrElse(tokenizer.tokenToId("<unk>"), () => -1)).toBe(0)
        const text = "the quick\nbrown fox 😁 こんにちは"
        const ids = yield* numbers(yield* tokenizer.encode(text))
        expect(ids).not.toContain(0)
        const decoded = yield* tokenizer.decode(ids)
        expect(decoded).toBe(text)
      }))

    it.effect("WordLevel round-trips whitespace-separated words", () =>
      Effect.gen(function*() {
        const tokenizer = yield* train("WordLevel", 200)
        const text = "hello world hello tokenizer"
        const decoded = yield* tokenizer.decode(yield* tokenizer.encode(text))
        expect(decoded).toBe(text)
      }))
  })
})

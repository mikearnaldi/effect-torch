import * as Scaffold from "@effect-torch/models/DiffusionGemma"
import * as Noise from "@effect-torch/serve/DecisionNoise"
import * as Planner from "@effect-torch/serve/DecisionPlanner"
import * as Readout from "@effect-torch/serve/DecisionReadout"
import { prepareRead } from "@effect-torch/serve/DecisionScaffold"
import type * as Tokenizers from "@effect-torch/tokenizers"
import { describe, expect, it } from "@effect/vitest"
import { Effect, Option, Schema } from "effect"
import { createHash } from "node:crypto"
import { readFileSync, realpathSync } from "node:fs"
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { createRequire } from "node:module"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { beforeAll } from "vitest"

const fixture = Schema.decodeUnknownSync(Schema.Struct({
  assets: Schema.Record(Schema.String, Schema.String),
  cases: Schema.Array(Schema.Struct({
    name: Schema.String,
    messages: Schema.Array(Schema.Struct({ role: Schema.String, content: Schema.String })),
    prompt: Schema.String,
    promptIds: Schema.Array(Schema.Number),
    canvasIds: Schema.Array(Schema.Number),
    scaffold: Schema.String,
    slot: Schema.Number,
    labels: Schema.Array(Schema.String),
    labelIds: Schema.Array(Schema.Number)
  }))
}))(JSON.parse(readFileSync(new URL("./fixtures/diffusion-gemma-tokenization.json", import.meta.url), "utf8")))

const officialTemplate = readFileSync(
  new URL("../../tokenizers/test/fixtures/diffusiongemma/chat_template.jinja", import.meta.url),
  "utf8"
)

const hash = (text: string) => createHash("sha256").update(text).digest("hex")

const localModel = "diffusiongemma-local-scaffold-m0"

const config = { modelId: localModel }

const planQuestion = (question: Schema.Json, state: Schema.Json = "state") =>
  Planner.plan({ model: localModel, state, questions: { caller_only_id: question } }, config).pipe(
    Effect.map((plan) => plan.routes[0].decision)
  )

const choice = (count: number) => ({
  type: "choice",
  criteria: Object.fromEntries(Array.from({ length: count }, (_, i) => ["option_" + i, null]))
})

// This inline WordLevel tokenizer exercises the real native boundary and the
// checked-in official Jinja template without the ignored 32 MB DG tokenizer.
// Its token IDs are synthetic and provide no DiffusionGemma parity evidence.
const specials = [
  ["<pad>", 0],
  ["<unk>", 1],
  ["<bos>", 2],
  ["<|channel>", 100],
  ["<channel|>", 101],
  ["<|turn>", 105],
  ["<turn|>", 106]
] as const

const vocabulary = Array.from({ length: 107 }, (_, index) => "unused_" + index)

for (const [token, id] of specials) vocabulary[id] = token

vocabulary.push("thought", "answer:", "yes", "no", ...Array.from("ABCDEFGHIJ123456789"))

const tinyJson = JSON.stringify({
  version: "1.0",
  truncation: null,
  padding: null,
  normalizer: null,
  post_processor: null,
  decoder: null,
  added_tokens: specials.map(([content, id]) => ({
    id,
    content,
    special: true,
    single_word: false,
    lstrip: false,
    rstrip: false,
    normalized: false
  })),
  pre_tokenizer: { type: "WhitespaceSplit" },
  model: {
    type: "WordLevel",
    unk_token: "<unk>",
    vocab: Object.fromEntries(vocabulary.map((token, id) => [token, id]))
  }
})

const native = await import("@effect-torch/tokenizers").catch(() => undefined)

const tiny = native === undefined ? undefined : await Effect.runPromise(
  native.fromJson(tinyJson, {
    padding: { _tag: "None" },
    truncation: { _tag: "None" },
    specialTokens: "Always"
  }).pipe(
    Effect.map((tokenizer): Scaffold.ModelTemplate => ({
      tokenizer,
      chatTemplate: officialTemplate,
      tokenizerSha256: hash(tinyJson),
      modelRevision: "synthetic-wordlevel-test"
    }))
  )
)

const tinyModel = (): Scaffold.ModelTemplate => {
  if (tiny === undefined) throw new Error("Build @effect-torch/tokenizers host addon to run native scaffold tests")

  return tiny
}

describe("scaffold integration configuration", () => {
  it("resolves one pinned Effect installation for app, tokenizer, and effect/vitest", () => {
    const require = createRequire(import.meta.url)
    const tokenizerRequire = createRequire(require.resolve("@effect-torch/tokenizers/package.json"))
    const vitestRequire = createRequire(require.resolve("@effect/vitest"))
    const appEffect = realpathSync(require.resolve("effect"))
    expect(realpathSync(tokenizerRequire.resolve("effect"))).toBe(appEffect)
    expect(realpathSync(vitestRequire.resolve("effect"))).toBe(appEffect)
    expect(JSON.parse(readFileSync(require.resolve("effect/package.json"), "utf8")).version).toBe("4.0.0-beta.101")
  })

  it("matches the audited asset identities and checked-in official Jinja", () => {
    expect(Scaffold.pins.tokenizerSha256).toBe(fixture.assets["tokenizer.json"])
    expect(Scaffold.pins.chatTemplateSha256).toBe(fixture.assets["chat_template.jinja"])
    expect(hash(officialTemplate)).toBe(Scaffold.pins.chatTemplateSha256)
  })

  it.effect("fails missing local assets with a typed error", () =>
    Effect.gen(function*() {
      const error = yield* Effect.flip(
        Scaffold.loadPinned(fileURLToPath(new URL("./fixtures/missing-tokenizer-assets", import.meta.url)))
      )

      expect(error).toBeInstanceOf(Scaffold.ScaffoldError)
    }))

  it("rejects asset hash mismatches before constructing a tokenizer", async () => {
    const directory = await mkdtemp(join(tmpdir(), "m0-scaffold-"))

    try {
      await writeFile(join(directory, "tokenizer.json"), tinyJson)
      await writeFile(join(directory, "chat_template.jinja"), officialTemplate)
      const error = await Effect.runPromise(Effect.flip(Scaffold.loadPinned(directory)))
      expect(error).toBeInstanceOf(Scaffold.ScaffoldError)
      expect(error.message).toContain("SHA-256 mismatch")
    } finally {
      await rm(directory, { recursive: true, force: true })
    }
  })
})

describe.skipIf(native === undefined)("small native scaffold integration", () => {
  it.effect("maps Noul false/true to no/yes and retains rich descriptions", () =>
    Effect.gen(function*() {
      const q = yield* planQuestion({
        type: "noul",
        instructions: ["Check", { task: "billing" }],
        criteria: { false: { means: "absent" }, true: ["present"] }
      }, { nested: [null, 4] })

      const built = yield* Scaffold.build(q, tinyModel())
      expect(built.labels.map((label) => [label.code, label.optionName])).toEqual([["no", "false"], ["yes", "true"]])
      expect(built.prompt).toContain("Question answer: [\"Check\",{\"task\":\"billing\"}]")
      expect(built.prompt).toContain("no: {\"means\":\"absent\"}")
      expect(built.prompt).toContain("yes: [\"present\"]")
      expect(built.prompt).toContain("{\"nested\":[null,4]}")
      expect(built.prompt).not.toContain("caller_only_id")
      expect(built.canvasIds).toHaveLength(16)
      expect(built.canvasIds[built.contentLength - 1]).toBe(106)
      expect(Array.from(built.canvasIds.slice(built.contentLength))).toEqual(Array(16 - built.contentLength).fill(0))
    }))

  it.effect("preserves visible Choice keys and the zero-based Score mapping", () =>
    Effect.gen(function*() {
      const choicePlan = yield* planQuestion({
        type: "choice",
        criteria: { caller_option_z: { examples: ["bug"] }, caller_option_a: null }
      })

      const builtChoice = yield* Scaffold.build(choicePlan, tinyModel())
      expect(builtChoice.prompt).toContain("A: caller_option_z ({\"examples\":[\"bug\"]})")
      expect(builtChoice.prompt).toContain("B: caller_option_a")
      const levels = [null, { impact: "high" }, ["blocked"]]
      const scorePlan = yield* planQuestion({ type: "score", criteria: levels })
      const builtScore = yield* Scaffold.build(scorePlan, tinyModel())
      expect(builtScore.labels.map((label) => [label.code, label.optionName])).toEqual([["1", "0"], ["2", "1"], [
        "3",
        "2"
      ]])
      expect(builtScore.prompt).toContain("2: {\"impact\":\"high\"}")
      expect(yield* Readout.answerFromProbabilities(scorePlan.question, [0, 0, 1])).toMatchObject({
        type: "score",
        score: 2,
        legend: { "0": null, "1": levels[1], "2": levels[2] }
      })

      const ten = yield* Scaffold.build(
        yield* planQuestion({ type: "score", criteria: Array(10).fill(null) }),
        tinyModel()
      )

      expect(ten.labels.map((label) => label.code)).toEqual(Array.from("ABCDEFGHIJ"))
    }))

  it.effect("uses a singleton probe without exposing its token as an allowed label", () =>
    Effect.gen(function*() {
      const built = yield* Scaffold.build(yield* planQuestion(choice(1)), tinyModel())
      expect(built.labels.map((label) => label.code)).toEqual(["A"])
      expect(built.allowedTokenIds).toHaveLength(1)
      expect(built.canvasIds[built.slot]).toBe(built.allowedTokenIds[0])
    }))

  it.effect("keeps private baseline buffers and returns independent writable arrays", () =>
    Effect.gen(function*() {
      const built = yield* Scaffold.build(yield* planQuestion(choice(2)), tinyModel())
      const original = built.canvasIds
      built.canvasIds.fill(99)
      built.prefixIds.fill(99)
      built.allowedTokenIds.fill(99)
      expect(built.canvasIds).toEqual(original)
      expect(built.prefixIds[0]).toBe(2)
      expect(built.allowedTokenIds).not.toEqual(new Uint32Array([99, 99]))
      expect(Object.isFrozen(built)).toBe(true)
      expect(Object.isFrozen(built.labels[0])).toBe(true)
      expect(Object.isFrozen(original)).toBe(false)
      const a = yield* prepareRead(built, { seed: "test", readIndex: 0 })
      const b = yield* prepareRead(built, { seed: "test", readIndex: 0 })
      expect(a).toEqual(b)
      expect(a.canvasIds.buffer).not.toBe(b.canvasIds.buffer)
      expect(a.prefixIds.buffer).not.toBe(a.canvasIds.buffer)
      a.canvasIds.fill(99)
      a.prefixIds.fill(99)
      a.allowedTokenIds.fill(99)
      expect(yield* prepareRead(built, { seed: "test", readIndex: 0 })).toEqual(b)
    }))

  it.effect("corrupts only the answer slot with explicit per-read vocabulary noise", () =>
    Effect.gen(function*() {
      const built = yield* Scaffold.build(yield* planQuestion(choice(2)), tinyModel())
      const baseline = built.canvasIds
      const values = new Set<number>()

      for (const readIndex of [0, 1, 2, 1, 0]) {
        const read = yield* prepareRead(built, { seed: "test", readIndex })

        const noise = yield* Noise.make({
          semanticKey: built.semanticKey,
          seed: "test",
          readIndex,
          vocabularySize: built.vocabularySize,
          length: 1
        })

        expect(read.canvasIds[built.slot]).toBe(noise[0])
        values.add(noise[0])
        expect(read.prefixIds).toEqual(built.prefixIds)
        expect(read.allowedTokenIds).toEqual(built.allowedTokenIds)

        for (let index = 0; index < baseline.length; index++) {
          if (index !== built.slot) expect(read.canvasIds[index]).toBe(baseline[index])
        }
      }

      expect(values.size).toBeGreaterThan(1)
      expect(built.canvasIds).toEqual(baseline)
      expect(yield* Effect.flip(prepareRead(built, { seed: "test", readIndex: -1 }))).toBeInstanceOf(
        Noise.NoiseError
      )
    }))

  it.effect("keys actual model inputs, identities and canvas configuration", () =>
    Effect.gen(function*() {
      const q = yield* planQuestion(choice(2), { a: [1, 2] })
      const model = tinyModel()
      const base = yield* Scaffold.build(q, model)
      expect(base.semanticKey).not.toBe(q.semanticKey)

      const changedPlaceholder = yield* Scaffold.build({
        ...q,
        prompt: "ignored provisional prompt",
        semanticKey: "ignored provisional key"
      }, model)

      expect(changedPlaceholder.semanticKey).toBe(base.semanticKey)

      const variants = [
        yield* Scaffold.build({ ...q, model: "other-model" }, model),
        yield* Scaffold.build({ ...q, state: { a: [2, 1] } }, model),
        yield* Scaffold.build(q, { ...model, modelRevision: "other-revision" }),
        yield* Scaffold.build(q, { ...model, tokenizerSha256: "a".repeat(64) }),
        yield* Scaffold.build(q, { ...model, chatTemplate: model.chatTemplate + "{# template revision #}" }),
        yield* Scaffold.build(q, model, { canvasLength: 20 })
      ]

      expect(new Set([base, ...variants].map((value) => value.semanticKey)).size).toBe(7)
    }))

  it.effect("rejects undersized/non-integer canvases and invalid asset identities", () =>
    Effect.gen(function*() {
      const q = yield* planQuestion(choice(2))

      for (const canvasLength of [0, -1, 1.5, NaN, 1]) {
        expect(yield* Effect.flip(Scaffold.build(q, tinyModel(), { canvasLength }))).toBeInstanceOf(
          Scaffold.ScaffoldError
        )
      }

      const enough = yield* Scaffold.build(q, tinyModel())
      expect((yield* Scaffold.build(q, tinyModel(), { canvasLength: enough.contentLength })).canvasIds).toHaveLength(
        enough.contentLength
      )
      expect(yield* Effect.flip(Scaffold.build(q, { ...tinyModel(), tokenizerSha256: "missing" }))).toBeInstanceOf(
        Scaffold.ScaffoldError
      )
    }))

  const modifyEncode = (
    change: (text: string, ids: Tokenizers.TokenIds) => Tokenizers.TokenIds
  ): Scaffold.ModelTemplate => {
    const model = tinyModel()

    return {
      ...model,
      tokenizer: {
        ...model.tokenizer,
        encode: (text, options) => model.tokenizer.encode(text, options).pipe(Effect.map((ids) => change(text, ids)))
      }
    }
  }

  it.effect("rejects prompt/canvas retokenization even when the slot still looks valid", () =>
    Effect.gen(function*() {
      const model = modifyEncode((text, ids) => {
        if (text.startsWith("<bos>") && text.endsWith("<turn|>")) ids.data[1] = 1

        return ids
      })

      const error = yield* Effect.flip(Scaffold.build(yield* planQuestion(choice(2)), model))
      expect(error.message).toContain("boundary retokenizes")
    }))

  it.effect("checks every label, including a later normalization collision", () =>
    Effect.gen(function*() {
      const model = tinyModel()

      const colliding = {
        ...model,
        tokenizer: {
          ...model.tokenizer,
          encode: (text: string, options?: Tokenizers.EncodeOptions) =>
            model.tokenizer.encode(text.replace(/answer: C<turn\|>$/, "answer: B<turn|>"), options)
        }
      }

      const error = yield* Effect.flip(Scaffold.build(yield* planQuestion(choice(3)), colliding))
      expect(error.message).toContain("normalization collides")
    }))

  it.effect("rejects later multi-token labels and invalid special-token mappings", () =>
    Effect.gen(function*() {
      const model = tinyModel()

      const split = {
        ...model,
        tokenizer: {
          ...model.tokenizer,
          encode: (text: string, options?: Tokenizers.EncodeOptions) =>
            model.tokenizer.encode(text.replace(/answer: C<turn\|>$/, "answer: C C<turn|>"), options)
        }
      }

      const q = yield* planQuestion(choice(3))
      expect((yield* Effect.flip(Scaffold.build(q, split))).message).toContain("multiple scaffold tokens")
      const missing = { ...model, tokenizer: { ...model.tokenizer, tokenToId: () => Option.none<number>() } }
      expect((yield* Effect.flip(Scaffold.build(q, missing))).message).toContain("Unexpected special token")
    }))

  it.effect("preserves typed native tokenizer failures", () =>
    Effect.gen(function*() {
      if (native === undefined) throw new Error("Native tokenizer unavailable")

      const error = new native.TokenizerError({ op: "encode", message: "test native failure" })
      const model = { ...tinyModel(), tokenizer: { ...tinyModel().tokenizer, encode: () => Effect.fail(error) } }
      expect(yield* Effect.flip(Scaffold.build(yield* planQuestion(choice(2)), model))).toBe(error)
    }))
})

// Default tests need only the small native fixture. An explicit gate or asset
// path enables the real DG suite. Once requested, missing assets/addons FAIL.
const realRequested = process.env.EFFECT_TORCH_SERVE_TOKENIZER_GATE === "1" ||
  process.env.EFFECT_TORCH_DIFFUSION_GEMMA_TOKENIZER_ASSETS !== undefined

describe.skipIf(!realRequested)("pinned DiffusionGemma mandatory asset gate", () => {
  let model: Scaffold.ModelTemplate
  beforeAll(async () => {
    const directory = process.env.EFFECT_TORCH_DIFFUSION_GEMMA_TOKENIZER_ASSETS ??
      fileURLToPath(new URL("../../.cache/decision-model/assets", import.meta.url))

    model = await Effect.runPromise(Scaffold.loadPinned(directory))
  }, 30_000)

  for (const entry of fixture.cases) {
    it.effect("matches full-scaffold evidence for " + entry.name, () =>
      Effect.gen(function*() {
        const tokenizer = model.tokenizer

        const rendered = yield* tokenizer.applyChatTemplate(model.chatTemplate, entry.messages, {
          addGenerationPrompt: true,
          variables: { bos_token: "<bos>", enable_thinking: false }
        })

        expect(rendered).toBe(entry.prompt)
        expect(Array.from((yield* tokenizer.encode(rendered, { addSpecialTokens: false })).data)).toEqual(
          entry.promptIds
        )

        const question = entry.name === "noul" ?
          { type: "noul", criteria: { false: "option_1", true: "option_0" } }
          : entry.name.startsWith("score") ?
          { type: "score", criteria: entry.labels.map((_, i) => "option_" + i) }
          : choice(entry.labels.length)

        const q = yield* planQuestion(question, entry.messages[1].content)
        const built = yield* Scaffold.build(q, model)
        const expectedLabels = entry.name === "noul" ? ["no", "yes"] : entry.labels
        const expectedIds = expectedLabels.map((label) => entry.labelIds[entry.labels.indexOf(label)])
        expect(built.labels.map((label) => label.code)).toEqual(expectedLabels)
        expect(Array.from(built.allowedTokenIds)).toEqual(expectedIds)
        expect(new Set(built.allowedTokenIds).size).toBe(entry.labels.length)
        expect(built.slot).toBe(entry.slot)
        expect(built.slot).toBe(entry.name === "score-9" ? 7 : 6)
        const baseline = [...entry.canvasIds]
        baseline[entry.slot] = expectedIds[0]
        expect(Array.from(built.canvasIds.slice(0, built.contentLength))).toEqual([...baseline, 106])
        expect(built.contentLength).toBe(entry.name === "score-9" ? 9 : 8)

        if (entry.name !== "noul") {
          expect(built.prompt).toBe(entry.prompt)
          expect(Array.from(built.prefixIds)).toEqual(entry.promptIds)
        }

        for (const label of built.labels) {
          const full =
            (yield* tokenizer.encode(built.prompt + "<|channel>thought\n<channel|>answer: " + label.code + "<turn|>", {
              addSpecialTokens: false
            })).data

          const expectedCanvas = built.canvasIds.slice(0, built.contentLength)
          expectedCanvas[built.slot] = label.tokenId
          expect(Array.from(full)).toEqual([...built.prefixIds, ...expectedCanvas])
        }
      }))
  }

  it.effect("preserves actual prompts, keys, and noise across renamed/reordered/duplicate siblings", () =>
    Effect.gen(function*() {
      const target = {
        type: "score",
        instructions: { task: "assess impact" },
        criteria: [null, { severity: "blocking" }]
      }

      const isolated = yield* Scaffold.build(yield* planQuestion(target, { a: 1, b: 2 }), model)
      const firstRead = yield* prepareRead(isolated, { seed: "actual-model", readIndex: 0 })
      const otherRead = yield* prepareRead(isolated, { seed: "actual-model", readIndex: 1 })
      expect(otherRead.canvasIds[isolated.slot]).not.toBe(firstRead.canvasIds[isolated.slot])
      const sibling = { type: "noul", instructions: "SIBLING_SECRET_619" }

      for (
        const questions of [{ renamed: target, after: sibling }, {
          before: sibling,
          renamed: target,
          duplicate: target
        }]
      ) {
        const plan = yield* Planner.plan({ model: localModel, state: { b: 2, a: 1 }, questions }, config)

        for (const route of [...plan.routes].reverse().filter((route) => route.decision.question.type === "score")) {
          const built = yield* Scaffold.build(route.decision, model)
          expect(built.semanticKey).toBe(isolated.semanticKey)
          expect(built.prompt).toBe(isolated.prompt)
          expect(built.prompt).not.toContain("SIBLING_SECRET_619")
          expect(yield* prepareRead(built, { seed: "actual-model", readIndex: 0 })).toEqual(firstRead)
          expect(yield* prepareRead(built, { seed: "actual-model", readIndex: 1 })).toEqual(otherRead)
        }
      }
    }))
})

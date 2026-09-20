import { describe, expect, it } from "@effect/vitest"
import { Effect } from "effect"
import { createHash } from "node:crypto"
import { readFileSync } from "node:fs"
import * as Tokenizer from "../src/index.ts"
import diffusionGemmaCases from "./fixtures/diffusiongemma/cases.json" with { type: "json" }
import pythonMethodCases from "./fixtures/python-methods.json" with { type: "json" }

const tokenizerJson = JSON.stringify({
  version: "1.0",
  model: { type: "WordLevel", vocab: { "[UNK]": 0 }, unk_token: "[UNK]" }
})

const officialTemplate = readFileSync(new URL("./fixtures/diffusiongemma/chat_template.jinja", import.meta.url), "utf8")

describe("chat template Python methods", () => {
  for (const testCase of pythonMethodCases) {
    it.effect(testCase.name, () =>
      Effect.gen(function*() {
        const tokenizer = yield* Tokenizer.fromJson(tokenizerJson, Tokenizer.strictConfig)
        const { messages = [], ...variables } = testCase.context
        const rendered = yield* tokenizer.applyChatTemplate(testCase.template, messages, { variables })
        expect(rendered).toBe(testCase.expected)
      }))
  }

  it.effect("reports invalid and unknown methods as TokenizerError and can render again", () =>
    Effect.gen(function*() {
      const tokenizer = yield* Tokenizer.fromJson(tokenizerJson, Tokenizer.strictConfig)
      for (
        const template of [
          "{{ {}.get() }}",
          "{{ {}.get('a', 'b', 'c') }}",
          "{{ 'a,b'.split(',', 1, 2) }}",
          "{{ {}.unknown_method() }}",
          "{{ none.get('a') }}"
        ]
      ) {
        const error = yield* Effect.flip(tokenizer.applyChatTemplate(template, []))
        expect(error._tag).toBe("TokenizerError")
        expect(error.op).toBe("applyChatTemplate")
        expect(error.message).toContain("(in <string>:1)")
      }
      expect(yield* tokenizer.applyChatTemplate("{{ {}.get('missing', 'still usable') }}", [])).toBe("still usable")
    }))
})

describe("official DiffusionGemma chat template", () => {
  it("preserves the pinned template bytes", () => {
    expect(createHash("sha256").update(officialTemplate).digest("hex")).toBe(
      "9aeb7eac68ad87bba7567e9d4597ff203e5609f1b427d9e823437d0142cc61bf"
    )
  })

  for (const testCase of diffusionGemmaCases) {
    it.effect(testCase.name, () =>
      Effect.gen(function*() {
        const tokenizer = yield* Tokenizer.fromJson(tokenizerJson, Tokenizer.strictConfig)
        const { add_generation_prompt, messages, ...variables } = testCase.context
        const rendered = yield* tokenizer.applyChatTemplate(officialTemplate, messages, {
          addGenerationPrompt: add_generation_prompt,
          variables
        })
        expect(rendered).toBe(testCase.expected)
      }))
  }

  it.effect("preserves template validation errors for serialized tool arguments", () =>
    Effect.gen(function*() {
      const tokenizer = yield* Tokenizer.fromJson(tokenizerJson, Tokenizer.strictConfig)
      const error = yield* Effect.flip(tokenizer.applyChatTemplate(officialTemplate, [{
        role: "assistant",
        tool_calls: [{ function: { name: "status", arguments: "{}" } }]
      }]))
      expect(error._tag).toBe("TokenizerError")
      expect(error.message).toContain("tool_calls[].function.arguments must be a JSON object")
    }))
})

/**
 * Export pinned DiffusionGemma chat/scaffold token fixtures.
 *
 * pnpm --filter @effect-torch/examples exec tsx diffusion-gemma/tokenizer-audit.ts <assets> <output.json>
 * Fetch assets with inference/decision-model/tools/inspect_checkpoint.py first.
 */
import * as Tokenizer from "@effect-torch/tokenizers"
import { NodeRuntime } from "@effect/platform-node"
import { Effect } from "effect"
import { createHash } from "node:crypto"
import { mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { dirname, join } from "node:path"

const pins = {
  "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
  "chat_template.jinja": "9aeb7eac68ad87bba7567e9d4597ff203e5609f1b427d9e823437d0142cc61bf"
}
const scaffold = "<|channel>thought\n<channel|>"
const answer = (label: string): string => `${scaffold}answer: ${label}`
const state = "The user asks for help with a billing invoice."

const main = Effect.gen(function*() {
  const [directory, output] = process.argv.slice(2)
  if (directory === undefined || output === undefined) {
    throw new Error("usage: tokenizer-audit.ts <asset-directory> <output.json>")
  }
  for (const [name, expected] of Object.entries(pins)) {
    const actual = createHash("sha256").update(readFileSync(join(directory, name))).digest("hex")
    if (actual !== expected) throw new Error(`${name}: SHA256 mismatch`)
  }
  const tokenizer = yield* Tokenizer.fromFile(join(directory, "tokenizer.json"), {
    padding: Tokenizer.paddingNone,
    truncation: Tokenizer.truncationNone,
    specialTokens: "Always"
  })
  if (tokenizer.vocabSize !== 262144) throw new Error("unexpected vocabulary size")
  const encode = (text: string) =>
    tokenizer.encode(text, { addSpecialTokens: false }).pipe(Effect.map((ids) => Array.from(ids.data)))
  const base = yield* encode(answer("A"))
  const alternative = yield* encode(answer("B"))
  const differences = base.flatMap((id, index) => id === alternative[index] ? [] : [index])
  if (base.length !== alternative.length || differences.length !== 1) {
    throw new Error("A and B must differ at exactly one scaffold token")
  }
  const slot = differences[0]
  const candidates = Array.from({ length: 0x530 - 0x41 }, (_, index) => String.fromCodePoint(index + 0x41))
    .filter((label) => /^\p{L}$/u.test(label))
  const labels: Array<string> = []
  const labelIds: Array<number> = []
  const rejected: Array<string> = []
  for (const label of candidates) {
    const ids = yield* encode(answer(label))
    if (
      ids.length !== base.length || ids.some((id, index) => index !== slot && id !== base[index]) ||
      labelIds.includes(ids[slot])
    ) {
      rejected.push(label)
      continue
    }
    labels.push(label)
    labelIds.push(ids[slot])
    if (labels.length === 255) break
  }
  if (labels.length !== 255) throw new Error(`found only ${labels.length} distinct single-slot codes`)
  const template = readFileSync(join(directory, "chat_template.jinja"), "utf8")
  const cases = []
  const sets = [
    { name: "noul", labels: ["yes", "no"] },
    { name: "score-9", labels: Array.from({ length: 9 }, (_, i) => String(i + 1)) },
    { name: "score-10", labels: labels.slice(0, 10) },
    { name: "choice-1", labels: labels.slice(0, 1) },
    { name: "choice-26", labels: labels.slice(0, 26) },
    { name: "choice-255", labels }
  ]
  for (const entry of sets) {
    // The upstream single-question layout with a fixed internal identifier.
    // Caller IDs and sibling questions have no place in the encoded text.
    const system = "Answer a fixed set of questions about the state the user provides. " +
      "Each question lists its allowed answers; reply with exactly one label per question.\n" +
      "\nQuestion answer: Choose the matching option.\n" +
      entry.labels.map((label, index) => `  ${label}: option_${index}\n`).join("") +
      "\nReply with one line per question, in this order, formatted as \"id: label\"."
    const messages = [{ role: "system", content: system }, { role: "user", content: state }]
    const prompt = yield* tokenizer.applyChatTemplate(template, messages, {
      addGenerationPrompt: true,
      variables: { bos_token: "<bos>", enable_thinking: false }
    })
    const promptIds = yield* encode(prompt)
    const encodings = yield* Effect.forEach(entry.labels, (label) => encode(prompt + answer(label)))
    const first = encodings[0]
    const canvas = yield* encode(answer(entry.labels[0]))
    const second = encodings[1] ?? (yield* encode(prompt + answer("B")))
    const changed = first.flatMap((id, index) => id === second[index] ? [] : [index])
    if (first.length !== second.length || changed.length !== 1) {
      throw new Error(`${entry.name}: labels do not occupy one scaffold slot`)
    }
    const fullSlot = changed[0]
    const canvasSlot = fullSlot - promptIds.length
    if (first.length !== promptIds.length + canvas.length || !promptIds.every((id, i) => first[i] === id)) {
      throw new Error(`${entry.name}: prompt/canvas boundary retokenizes`)
    }
    for (let index = 0; index < encodings.length; index++) {
      const ids = encodings[index]
      if (ids.length !== first.length || ids.some((id, i) => i !== fullSlot && id !== first[i])) {
        throw new Error(`${entry.name}: ${entry.labels[index]} changes multiple scaffold tokens`)
      }
    }
    const ids = encodings.map((encoding) => encoding[fullSlot])
    if (new Set(ids).size !== ids.length) throw new Error(`${entry.name}: duplicate label IDs`)
    cases.push({
      name: entry.name,
      messages,
      prompt,
      promptIds,
      scaffold: answer(entry.labels[0]),
      canvasIds: canvas,
      slot: canvasSlot,
      labels: entry.labels,
      labelIds: ids
    })
  }
  const fixture = {
    model: "google/diffusiongemma-26B-A4B-it",
    revision: "f7f5b7f5fa82ffc52addd066915886d497f5517b",
    assets: pins,
    templateSource:
      "structured_server.py single-question lines layout at vllm 58eacf242207c9b9b54abff57e684fc1a7d74548",
    labelPolicy: "First 255 Unicode letters in U+0041..U+052F that occupy one distinct full-scaffold token.",
    qualityStatus: "Tokenization verified only. Large-option model quality has not been evaluated.",
    rejectedCandidates: rejected,
    cases
  }
  mkdirSync(dirname(output), { recursive: true })
  writeFileSync(output, JSON.stringify(fixture, null, 2) + "\n")
  console.log(JSON.stringify(
    cases.map(({ name, slot, labelIds, promptIds }) => ({
      name,
      slot,
      labels: labelIds.length,
      prefixTokens: promptIds.length
    })),
    null,
    2
  ))
})

NodeRuntime.runMain(main)

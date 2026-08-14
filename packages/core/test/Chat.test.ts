import { describe, expect } from "@effect/vitest"
import { Effect, Option, Stream } from "effect"
import { Chat, type Model, Tensor } from "../src/index.ts"
import { onDevices } from "./utils/devices.ts"

const START = 90
const MESSAGE = 91
const EOM = 92
const EOT = 93
const EOS = 94

const controlIds = new Map([
  ["<|start|>", START],
  ["<|message|>", MESSAGE],
  ["<|eom|>", EOM],
  ["<|eot|>", EOT],
  ["<|end_of_text|>", EOS]
])
const tokenTexts = new Map([
  [1, "to=self"],
  [2, "think"],
  [3, "assistant to=user"],
  [4, "answer"],
  [5, "plain"],
  [6, " text"]
])

const makeTokenizer = (captured: {
  rendered?: string
  addSpecialTokens?: boolean
  variables?: Readonly<Record<string, unknown>>
}): Chat.ChatTokenizer => ({
  applyChatTemplate: (_template, messages, options) =>
    Effect.sync(() => {
      captured.variables = options.variables ?? {}
      captured.rendered = messages.map((message) => `${message.role}:${String(message.content)}`).join("\n")
      return captured.rendered
    }),
  encode: (text, options) =>
    Effect.sync(() => {
      captured.addSpecialTokens = options?.addSpecialTokens ?? true
      expect(text).toBe(captured.rendered)
      return { data: new Uint32Array([1]) }
    }),
  decode: (ids, options) =>
    Effect.succeed(
      ids
        .filter((id) => options?.skipSpecialTokens !== true || id < START)
        .map((id) => tokenTexts.get(id) ?? "")
        .join("")
    ),
  tokenToId: (token) => Option.fromNullishOr(controlIds.get(token)),
  idToToken: (id) => Option.fromNullishOr([...controlIds.entries()].find(([, tokenId]) => tokenId === id)?.[0])
})

// The script models generation's contract: add returns logits for script[0],
// each step advances once, and close records deterministic session cleanup.
// Logits are real device tensors; only the scheduler/state machine is faked.
const makeProgram = (
  script: ReadonlyArray<number>,
  state: { closed: boolean; logits?: Array<Tensor.Concrete> }
): Model.InferenceProgram => {
  let step = 0
  const logitsFor = (token: number) =>
    Effect.gen(function*() {
      const values = new Float32Array(EOS + 1)
      values[token] = 1
      const [logits] = yield* Tensor.compute([yield* Tensor.fromTypedArray(values, [values.length])])
      state.logits?.push(logits)
      return logits
    })
  const seq = {
    sequence: {} as Tensor.KvSequence,
    cursor: () => Effect.succeed(0),
    finish: () => Effect.void
  }
  return {
    generation: () =>
      Effect.succeed({
        add: (_prompt: Tensor.Any) =>
          Effect.gen(function*() {
            return { seq, logits: yield* logitsFor(script[0]!) }
          }),
        step: () =>
          Effect.gen(function*() {
            step++
            return [yield* logitsFor(script[step]!)]
          }),
        live: () => Effect.succeed(1),
        close: () =>
          Effect.sync(() => {
            state.closed = true
          })
      })
  } as unknown as Model.InferenceProgram
}

onDevices("Chat", () => (it) => {
  describe("Chat.stream", () => {
    it.effect("streams structured reasoning and content segments from control tokens", () =>
      Effect.gen(function*() {
        const captured: {
          rendered?: string
          addSpecialTokens?: boolean
          variables?: Readonly<Record<string, unknown>>
        } = {}
        const programState = { closed: false }
        const events = Array.from(
          yield* Stream.runCollect(Chat.stream({
            program: makeProgram([1, MESSAGE, 2, EOM, START, 3, MESSAGE, 4, EOT], programState),
            tokenizer: makeTokenizer(captured),
            template: "{{ messages }}",
            messages: [{ role: "user", content: "hello" }],
            bosTokenId: EOS,
            variables: { current_date: "2026-08-13" }
          }))
        )

        expect(events.map((event) => event._tag)).toEqual([
          "prefill",
          "start",
          "delta",
          "end",
          "start",
          "delta",
          "end",
          "done"
        ])
        expect(captured.rendered).toBe("user:hello")
        expect(captured.addSpecialTokens).toBe(false)
        expect(captured.variables).toMatchObject({ current_date: "2026-08-13" })
        const first = events[1]
        const second = events[4]
        expect(first._tag === "start" && first.segment.kind).toBe("reasoning")
        expect(second._tag === "start" && second.segment.kind).toBe("content")
        const done = events.at(-1)
        expect(done?._tag).toBe("done")
        if (done?._tag === "done") {
          expect(done.result.content).toBe("answer")
          expect(done.result.reasoning).toBe("think")
          expect(done.result.finishReason).toBe("stop")
          expect(done.result.stats.promptTokens).toBe(1)
          expect(done.result.stats.generatedTokens).toBe(8)
        }
        expect(programState.closed).toBe(true)
      }))

    it.effect("supports unsegmented responses and reports max-token limits", () =>
      Effect.gen(function*() {
        const programState = { closed: false }
        const events = Array.from(
          yield* Stream.runCollect(Chat.stream({
            program: makeProgram([5, 6], programState),
            tokenizer: makeTokenizer({}),
            template: "{{ messages }}",
            messages: [{ role: "user", content: "hello" }],
            controls: false,
            stopTokens: [EOS],
            maxTokens: 2
          }))
        )

        expect(events.map((event) => event._tag)).toEqual([
          "prefill",
          "start",
          "delta",
          "delta",
          "end",
          "done"
        ])
        const done = events.at(-1)
        expect(done?._tag).toBe("done")
        if (done?._tag === "done") {
          expect(done.result.content).toBe("plain text")
          expect(done.result.reasoning).toBe("")
          expect(done.result.finishReason).toBe("maxTokens")
          expect(done.result.segments[0]?.finish).toBe("limit")
          expect(done.result.stats.generatedTokens).toBe(2)
        }
        expect(programState.closed).toBe(true)
      }))

    it.effect("clears unread logits when downstream stops after prefill", () =>
      Effect.gen(function*() {
        const programState: { closed: boolean; logits: Array<Tensor.Concrete> } = { closed: false, logits: [] }
        const events = Array.from(
          yield* Stream.runCollect(
            Chat.stream({
              program: makeProgram([5], programState),
              tokenizer: makeTokenizer({}),
              template: "{{ messages }}",
              messages: [{ role: "user", content: "hello" }],
              controls: false,
              stopTokens: [EOS]
            }).pipe(Stream.take(1))
          )
        )

        expect(events.map((event) => event._tag)).toEqual(["prefill"])
        expect(programState.closed).toBe(true)
        const error = yield* Effect.flip(Tensor.toNumberArray(programState.logits[0]!))
        expect(error.message).toContain("cleared")
      }))
  })
})

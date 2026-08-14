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

interface ProgramState {
  closed: boolean
  logits?: Array<Tensor.Concrete>
  sampled?: {
    add: Array<Tensor.SamplingOptions>
    step: Array<{ readonly token: number; readonly sampling: Tensor.SamplingOptions }>
  }
}

// The script models both generation contracts: add/step publish real device
// logits, while addSampled/stepSampled return only the scripted token ids.
const makeProgram = (
  script: ReadonlyArray<number>,
  state: ProgramState
): Model.InferenceProgram => {
  let step = 0
  const sampled = state.sampled
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
        addSampled: (_prompt: Tensor.Any, sampling: Tensor.SamplingOptions) =>
          Effect.sync(() => {
            sampled?.add.push(sampling)
            return { seq, token: script[0]! }
          }),
        stepSampled: (
          entries: ReadonlyArray<{
            readonly token: number
            readonly sampling: Tensor.SamplingOptions
          }>
        ) =>
          Effect.sync(() => {
            step++
            sampled?.step.push(...entries.map(({ token, sampling }) => ({ token, sampling })))
            return [script[step]!]
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
              stopTokens: [EOS],
              sampling: Chat.greedy
            }).pipe(Stream.take(1))
          )
        )

        expect(events.map((event) => event._tag)).toEqual(["prefill"])
        expect(programState.closed).toBe(true)
        const error = yield* Effect.flip(Tensor.toNumberArray(programState.logits[0]!))
        expect(error.message).toContain("cleared")
      }))

    it.effect("clears each consumed logits row before the stream ends", () =>
      Effect.gen(function*() {
        const programState: { closed: boolean; logits: Array<Tensor.Concrete> } = { closed: false, logits: [] }
        let inspected = false
        yield* Chat.stream({
          program: makeProgram([5, 6], programState),
          tokenizer: makeTokenizer({}),
          template: "{{ messages }}",
          messages: [{ role: "user", content: "hello" }],
          controls: false,
          stopTokens: [EOS],
          maxTokens: 2,
          sampling: Chat.greedy
        }).pipe(
          Stream.tap((event) => {
            if (inspected || event._tag !== "start") return Effect.void
            inspected = true
            return Effect.gen(function*() {
              const firstError = yield* Effect.flip(Tensor.toNumberArray(programState.logits[0]!))
              expect(firstError.message).toContain("cleared")
              expect(yield* Tensor.toNumberArray(programState.logits[1]!)).toHaveLength(EOS + 1)
            })
          }),
          Stream.runDrain
        )
        expect(inspected).toBe(true)
      }))

    it.effect("uses fused add and step for standard sampling without publishing logits", () =>
      Effect.gen(function*() {
        const publishedLogits: Array<Tensor.Concrete> = []
        const programState: ProgramState = {
          closed: false,
          logits: publishedLogits,
          sampled: { add: [], step: [] }
        }
        const events = Array.from(
          yield* Stream.runCollect(Chat.stream({
            program: makeProgram([5, 6], programState),
            tokenizer: makeTokenizer({}),
            template: "{{ messages }}",
            messages: [{ role: "user", content: "hello" }],
            controls: false,
            stopTokens: [EOS],
            maxTokens: 2,
            sampling: { seed: 7 }
          }))
        )

        expect(events[0]?._tag).toBe("prefill")
        expect(events.at(-1)?._tag).toBe("done")
        expect(programState.sampled?.add).toEqual([{
          temperature: 0,
          topK: 0,
          topP: 1,
          seed: 7,
          counter: 0
        }])
        expect(programState.sampled?.step).toEqual([{
          token: 5,
          sampling: {
            temperature: 0,
            topK: 0,
            topP: 1,
            seed: 7,
            counter: 1
          }
        }])
        expect(publishedLogits).toEqual([])
        expect(programState.closed).toBe(true)
      }))

    it.effect("keeps custom samplers on the logits path when fused sampling is available", () =>
      Effect.gen(function*() {
        let calls = 0
        const publishedLogits: Array<Tensor.Concrete> = []
        const programState: ProgramState = {
          closed: false,
          logits: publishedLogits,
          sampled: { add: [], step: [] }
        }
        const events = Array.from(
          yield* Stream.runCollect(Chat.stream({
            program: makeProgram([5], programState),
            tokenizer: makeTokenizer({}),
            template: "{{ messages }}",
            messages: [{ role: "user", content: "hello" }],
            controls: false,
            stopTokens: [EOS],
            maxTokens: 1,
            sampling: (logits) => {
              calls++
              expect(logits.length).toBe(EOS + 1)
              return 5
            }
          }))
        )
        expect(calls).toBe(1)
        expect(events.at(-1)?._tag).toBe("done")
        expect(programState.sampled).toEqual({ add: [], step: [] })
        expect(publishedLogits).toHaveLength(1)
        const error = yield* Effect.flip(Tensor.toNumberArray(publishedLogits[0]!))
        expect(error.message).toContain("cleared")
      }))
  })
})

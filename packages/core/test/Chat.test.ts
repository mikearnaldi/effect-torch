import { describe, expect } from "@effect/vitest"
import { Deferred, Effect, Fiber, Option, Stream } from "effect"
import * as Diffusion from "../src/Diffusion.ts"
import { type AutoRegressive, Chat, type Generation, Runtime, Tensor } from "../src/index.ts"
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

interface TokenizerCapture {
  rendered?: string
  addSpecialTokens?: boolean
  variables?: Parameters<Chat.ChatTokenizer["applyChatTemplate"]>[2]["variables"]
  decodeCalls?: number
  streamSteps?: number
}

const makeTokenizer = (
  captured: TokenizerCapture,
  incremental = false
): Chat.ChatTokenizer => {
  const tokenizer: Chat.ChatTokenizer = {
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
      Effect.sync(() => {
        captured.decodeCalls = (captured.decodeCalls ?? 0) + 1

        return ids
          .filter((id) => options?.skipSpecialTokens !== true || id < START)
          .map((id) => tokenTexts.get(id) ?? "")
          .join("")
      }),
    tokenToId: (token) => Option.fromNullishOr(controlIds.get(token)),
    idToToken: (id) => Option.fromNullishOr([...controlIds.entries()].find(([, tokenId]) => tokenId === id)?.[0])
  }

  if (!incremental) return tokenizer

  return {
    ...tokenizer,
    decodeStream: (options) => ({
      step: (id) =>
        Effect.sync(() => {
          captured.streamSteps = (captured.streamSteps ?? 0) + 1

          return options?.skipSpecialTokens === true && id >= START
            ? undefined
            : tokenTexts.get(id) ?? ""
        })
    })
  }
}

interface ProgramState {
  closed: boolean
  logits?: Array<Tensor.Concrete>
  sampled?: {
    add: Array<{
      readonly sampling: Partial<AutoRegressive.GenerationSamplingOptions> | undefined
      readonly maxTokens: number | undefined
      readonly eosTokens: ReadonlyArray<number> | undefined
    }>
    step: Array<{ readonly sampling: Partial<AutoRegressive.GenerationSamplingOptions> | undefined }>
  }
}

interface LogitsProgramState extends ProgramState {
  logits: Array<Tensor.Concrete>
}

// The script models sampled generation and lower-level logits execution.
const makeProgram = (
  script: ReadonlyArray<number>,
  state: ProgramState
): AutoRegressive.Artifact => {
  let step = 0
  let maxTokens: number | undefined
  let eosTokens: ReadonlyArray<number> = []
  const sampled = state.sampled

  const logitsFor = (token: number) =>
    Effect.gen(function*() {
      const values = new Float32Array(EOS + 1)
      values[token] = 1
      const [logits] = yield* Tensor.compute([yield* Tensor.fromTypedArray(values, [values.length])])
      state.logits?.push(logits)

      return logits
    })

  const generationSeq: AutoRegressive.GenerationSeq = {
    _tag: "GenerationSeq",
    cursor: () => Effect.succeed(0),
    finish: () => Effect.void
  }
  // SAFETY: The scripted program never submits this opaque sequence placeholder to a runtime.
  const executionSeq: AutoRegressive.StatefulExecutionSeq = {
    _tag: "StatefulExecutionSeq",
    sequence: {} as Tensor.KvSequence,
    cursor: () => Effect.succeed(0),
    finish: () => Effect.void
  }

  return {
    generation: () =>
      Effect.succeed({
        add: (entries: ReadonlyArray<AutoRegressive.GenerationAdd>) =>
          Effect.sync(() => {
            const entry = entries[0]!
            maxTokens = entry.maxTokens
            eosTokens = entry.eosTokens ?? []
            sampled?.add.push({
              sampling: entry.sampling,
              maxTokens: entry.maxTokens,
              eosTokens: entry.eosTokens
            })
            const token = script[0]!

            return [{
              seq: generationSeq,
              tokens: [token],
              ...(entry.eosTokens?.includes(token)
                ? { stopReason: "eos" as const }
                : entry.maxTokens === 1
                ? { stopReason: "maxTokens" as const }
                : {})
            }]
          }),
        step: (entries: ReadonlyArray<AutoRegressive.GenerationStep>) =>
          Effect.sync(() => {
            step++
            sampled?.step.push(...entries.map(({ sampling }) => ({ sampling })))
            const token = script[step]!

            return [{
              seq: generationSeq,
              tokens: [token],
              ...(eosTokens.includes(token)
                ? { stopReason: "eos" as const }
                : maxTokens !== undefined && step + 1 >= maxTokens
                ? { stopReason: "maxTokens" as const }
                : {})
            }]
          }),
        live: () => Effect.succeed(1),
        close: () =>
          Effect.sync(() => {
            state.closed = true
          })
      }),
    execution: () =>
      Effect.succeed({
        add: (_prompts: ReadonlyArray<Tensor.Any>) =>
          Effect.gen(function*() {
            return [{ seq: executionSeq, logits: yield* logitsFor(script[0]!) }]
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
      }),
    diagnostics: () =>
      Effect.succeed({
        roundsStarted: 0n,
        roundsCompleted: 0n,
        roundsRecovered: 0n,
        ordinaryRounds: 0n,
        speculativeRounds: 0n,
        proposedTokens: 0n,
        acceptedTokens: 0n,
        emittedTokens: 0n,
        provisionalBlocks: 0n,
        rolledBackBlocks: 0n,
        draftNanos: 0n,
        verificationNanos: 0n,
        acceptedLengthHistogram: [],
        targetPoolHighWaterBlocks: 0n
      })
  }
}

onDevices("Chat", () => (it) => {
  describe("Chat.streamWith", () => {
    it.effect("consumes only completed diffusion blocks across multiple refinements and commits", () =>
      Effect.gen(function*() {
        let liveResources = 0
        let refinements = 0
        let commits = 0
        let pages = 0

        const own = <A>(value: A) =>
          Effect.sync(() => {
            liveResources++

            return {
              value,
              release: Effect.sync(() => {
                liveResources--
              })
            }
          })

        const events = Array.from(
          yield* Chat.streamWith(
            ({ prompt, maxTokens, onPage }) =>
              Diffusion.runGeneration<number, number, number, number, never, never>({
                prompt,
                maxNewTokens: maxTokens!,
                outputLimit: "exact",
                callbacks: {
                  encode: (tokens) => own(tokens.length),
                  initialize: () => own({ canvas: new Uint32Array([4, 4]), feedback: { _tag: "Initial" as const } }),
                  evaluate: ({ step }) => own({ prediction: step.index, feedback: step.index }),
                  commit: (prefix, tokens) => {
                    commits++

                    return own(prefix + tokens.length)
                  }
                },
                policy: {
                  canvasLength: 2,
                  maxSteps: 2,
                  start: () => 0,
                  refine: ({ state, prediction }) =>
                    Effect.sync(() => {
                      refinements++
                      const canvas = new Uint32Array(prediction === 0 ? [4, 4] : [5, 6])

                      return { state: state + 1, canvas, draft: canvas, done: prediction === 1 }
                    }),
                  finish: (tokens) => ({ tokens, stop: false })
                },
                onPage: (tokens) => {
                  pages++

                  return onPage({ tokens })
                }
              }).pipe(Effect.map((result) => result.stop === "length" ? "maxTokens" : "stop")),
            {
              tokenizer: makeTokenizer({}),
              template: "{{ messages }}",
              messages: [{ role: "user", content: "hello" }],
              controls: false,
              stopTokens: [EOS],
              maxTokens: 4
            }
          ).pipe(Stream.runCollect)
        )

        const done = events.at(-1)
        expect(done?._tag).toBe("done")

        if (done?._tag === "done") {
          expect(done.result.content).toBe("plain textplain text")
          expect(done.result.finishReason).toBe("maxTokens")
          expect(done.result.stats.generatedTokens).toBe(4)
        }

        expect(refinements).toBe(4)
        expect(commits).toBe(1)
        expect(pages).toBe(2)
        expect(liveResources).toBe(0)
      }))

    it.effect("parses structured segments across committed pages and forwards prepared inputs", () =>
      Effect.gen(function*() {
        const captured: TokenizerCapture = {}
        let released = false

        const generate: Generation.Generator = (request) =>
          Effect.gen(function*() {
            expect(Array.from(request.prompt)).toEqual([1])
            expect(request.maxTokens).toBe(20)
            expect(request.eosTokens).toEqual([EOT, EOS])
            yield* request.onPage({ tokens: new Uint32Array([1, MESSAGE, 2, EOM, START]) })
            yield* request.onPage({ tokens: [3, MESSAGE, 4, EOT] })

            return "stop" as const
          }).pipe(Effect.ensuring(Effect.sync(() => {
            released = true
          })))

        const events = Array.from(
          yield* Stream.runCollect(Chat.streamWith(generate, {
            tokenizer: makeTokenizer(captured),
            template: "{{ messages }}",
            messages: [{ role: "user", content: "hello" }],
            maxTokens: 20,
            bosTokenId: EOS,
            variables: { current_date: "2026-09-20" }
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
        const done = events.at(-1)
        expect(done?._tag).toBe("done")

        if (done?._tag === "done") {
          expect(done.result.content).toBe("answer")
          expect(done.result.reasoning).toBe("think")
          expect(done.result.finishReason).toBe("stop")
          expect(done.result.stats.generatedTokens).toBe(8)
        }

        expect(released).toBe(true)
        expect(captured.rendered).toBe("user:hello")
        expect(captured.addSpecialTokens).toBe(false)
        expect(captured.variables).toMatchObject({ current_date: "2026-09-20", bos_token: "<|end_of_text|>" })
      }))

    it.effect("uses the generator's completion reason after its final committed page", () =>
      Effect.gen(function*() {
        let completed = false

        const events = Array.from(
          yield* Stream.runCollect(Chat.streamWith(({ onPage }) =>
            Effect.gen(function*() {
              yield* onPage({ tokens: [5, 6] })
              completed = true

              return "maxTokens" as const
            }), {
            tokenizer: makeTokenizer({}),
            template: "{{ messages }}",
            messages: [{ role: "user", content: "hello" }],
            controls: false,
            stopTokens: [EOS]
          }))
        )

        const done = events.at(-1)
        expect(done?._tag).toBe("done")

        if (done?._tag === "done") {
          expect(done.result.content).toBe("plain text")
          expect(done.result.finishReason).toBe("maxTokens")
          expect(done.result.stats.generatedTokens).toBe(2)
        }

        expect(completed).toBe(true)
      }))

    for (const stop of ["eos", "limit"] as const) {
      it.effect(
        "stops inside a multi-token page at " + stop + " and releases the producer",
        () =>
          Effect.gen(function*() {
            let released = 0
            let continued = false
            const captured: TokenizerCapture = {}

            const events = Array.from(
              yield* Stream.runCollect(Chat.streamWith(({ onPage }) =>
                Effect.gen(function*() {
                  yield* onPage({ tokens: stop === "eos" ? [5, EOS, 6] : [5, 6, 4] })
                  continued = true

                  return "stop" as const
                }).pipe(Effect.ensuring(Effect.sync(() => {
                  released++
                }))), {
                tokenizer: makeTokenizer(captured, true),
                template: "{{ messages }}",
                messages: [{ role: "user", content: "hello" }],
                controls: false,
                stopTokens: [EOS],
                maxTokens: 2
              }))
            )

            const done = events.at(-1)
            expect(done?._tag).toBe("done")

            if (done?._tag === "done") {
              expect(done.result.content).toBe(stop === "eos" ? "plain" : "plain text")
              expect(done.result.finishReason).toBe(stop === "eos" ? "stop" : "maxTokens")
              expect(done.result.stats.generatedTokens).toBe(stop === "eos" ? 1 : 2)
            }

            expect(captured.streamSteps).toBe(2)
            expect(continued).toBe(false)
            expect(released).toBe(1)
          })
      )
    }

    it.effect("cancels a producer blocked on page consumption without parsing the rest of the page", () =>
      Effect.gen(function*() {
        let released = 0
        let continued = false
        const captured: TokenizerCapture = {}

        const events = Array.from(
          yield* Chat.streamWith(({ onPage }) =>
            Effect.gen(function*() {
              yield* onPage({ tokens: [5, 6, 4] })
              continued = true

              return "stop" as const
            }).pipe(Effect.ensuring(Effect.sync(() => {
              released++
            }))), {
            tokenizer: makeTokenizer(captured, true),
            template: "{{ messages }}",
            messages: [{ role: "user", content: "hello" }],
            controls: false,
            stopTokens: [EOS]
          }).pipe(Stream.take(3), Stream.runCollect)
        )

        expect(events.map((event) => event._tag)).toEqual(["prefill", "start", "delta"])
        expect(captured.streamSteps).toBe(1)
        expect(continued).toBe(false)
        expect(released).toBe(1)
      }))

    it.effect("interrupts refinement before the first committed page and releases generation state", () =>
      Effect.gen(function*() {
        const refining = yield* Deferred.make<void>()
        let released = false
        const events: Array<Chat.ChatEvent> = []

        const fiber = yield* Chat.streamWith(() =>
          Deferred.succeed(refining, undefined).pipe(
            Effect.andThen(Effect.never),
            Effect.ensuring(Effect.sync(() => {
              released = true
            }))
          ), {
          tokenizer: makeTokenizer({}),
          template: "{{ messages }}",
          messages: [{ role: "user", content: "hello" }],
          controls: false
        }).pipe(
          Stream.tap((event) =>
            Effect.sync(() => {
              events.push(event)
            })
          ),
          Stream.runDrain,
          Effect.forkChild({ startImmediately: true })
        )

        yield* Deferred.await(refining)
        yield* Fiber.interrupt(fiber)
        expect(released).toBe(true)
        expect(events).toEqual([])
      }))

    it.effect("rejects empty published pages and releases their generator", () =>
      Effect.gen(function*() {
        let released = false

        const error = yield* Effect.flip(Stream.runCollect(Chat.streamWith(({ onPage }) =>
          onPage({ tokens: [] }).pipe(
            Effect.as("stop" as const),
            Effect.ensuring(Effect.sync(() => {
              released = true
            }))
          ), {
          tokenizer: makeTokenizer({}),
          template: "{{ messages }}",
          messages: [{ role: "user", content: "hello" }],
          controls: false
        })))

        expect(error._tag).toBe("ChatError")
        expect(error.message).toContain("empty token page")
        expect(released).toBe(true)
      }))

    it.effect("propagates generator errors and Effect requirements without synthesizing completion", () =>
      Effect.gen(function*() {
        const runtime = yield* Runtime.Runtime
        const failure = new Error("generation failed")
        let released = false
        const events: Array<Chat.ChatEvent> = []

        const generate: Generation.Generator<Error, Runtime.Runtime> = ({ onPage }) =>
          Effect.gen(function*() {
            expect((yield* Runtime.Runtime).identity).toBe(runtime.identity)
            yield* onPage({ tokens: [5, 6] })

            return yield* Effect.fail(failure)
          }).pipe(Effect.ensuring(Effect.sync(() => {
            released = true
          })))

        const error = yield* Effect.flip(
          Chat.streamWith(generate, {
            tokenizer: makeTokenizer({}),
            template: "{{ messages }}",
            messages: [{ role: "user", content: "hello" }],
            controls: false
          }).pipe(
            Stream.tap((event) =>
              Effect.sync(() => {
                events.push(event)
              })
            ),
            Stream.runDrain
          )
        )

        expect(error).toBe(failure)
        expect(events.filter((event) => event._tag === "delta").map((event) => event.text)).toEqual(["plain", " text"])
        expect(events.some((event) => event._tag === "done" || event._tag === "end")).toBe(false)
        expect(released).toBe(true)
      }))
  })

  describe("Chat.stream", () => {
    it.effect("streams structured reasoning and content segments from control tokens", () =>
      Effect.gen(function*() {
        const captured: TokenizerCapture = {}
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
        const captured: TokenizerCapture = {}

        const events = Array.from(
          yield* Stream.runCollect(Chat.stream({
            program: makeProgram([5, 6], programState),
            tokenizer: makeTokenizer(captured),
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

        expect(captured.decodeCalls).toBe(2)
        expect(captured.streamSteps).toBeUndefined()
        expect(programState.closed).toBe(true)
      }))

    it.effect("uses incremental decoding when the tokenizer provides it", () =>
      Effect.gen(function*() {
        const programState = { closed: false }
        const captured: TokenizerCapture = {}

        const events = Array.from(
          yield* Stream.runCollect(Chat.stream({
            program: makeProgram([5, 6], programState),
            tokenizer: makeTokenizer(captured, true),
            template: "{{ messages }}",
            messages: [{ role: "user", content: "hello" }],
            controls: false,
            stopTokens: [EOS],
            maxTokens: 2
          }))
        )

        const done = events.at(-1)
        expect(done?._tag).toBe("done")

        if (done?._tag === "done") {
          expect(done.result.content).toBe("plain text")
        }

        expect(captured.decodeCalls).toBeUndefined()
        expect(captured.streamSteps).toBe(2)
      }))

    it.effect("consumes every token in a terminal page without another step", () =>
      Effect.gen(function*() {
        const programState = { closed: false }
        const base = makeProgram([5, 6], programState)
        let steps = 0

        const seq: AutoRegressive.GenerationSeq = {
          _tag: "GenerationSeq",
          cursor: () => Effect.succeed(0),
          finish: () => Effect.void
        }

        const program: AutoRegressive.Artifact = {
          ...base,
          generation: () =>
            Effect.succeed({
              add: () => Effect.succeed([{ seq, tokens: [5, 6], stopReason: "maxTokens" as const }]),
              step: () =>
                Effect.sync(() => {
                  steps++

                  return []
                }),
              live: () => Effect.succeed(1),
              close: () => Effect.void
            })
        }

        const events = Array.from(
          yield* Stream.runCollect(Chat.stream({
            program,
            tokenizer: makeTokenizer({}),
            template: "{{ messages }}",
            messages: [{ role: "user", content: "hello" }],
            controls: false,
            stopTokens: [EOS],
            maxTokens: 2
          }))
        )

        const done = events.at(-1)
        expect(done?._tag).toBe("done")

        if (done?._tag === "done") {
          expect(done.result.content).toBe("plain text")
          expect(done.result.finishReason).toBe("maxTokens")
          expect(done.result.stats.generatedTokens).toBe(2)
        }

        expect(steps).toBe(0)
      }))

    it.effect("clears unread logits when downstream stops after prefill", () =>
      Effect.gen(function*() {
        const programState: LogitsProgramState = { closed: false, logits: Array<Tensor.Concrete>() }
        let samples = 0

        const events = Array.from(
          yield* Stream.runCollect(
            Chat.stream({
              program: makeProgram([5], programState),
              tokenizer: makeTokenizer({}),
              template: "{{ messages }}",
              messages: [{ role: "user", content: "hello" }],
              controls: false,
              stopTokens: [EOS],
              sampling: (logits) => {
                samples++

                return Chat.greedy(logits)
              }
            }).pipe(Stream.take(1))
          )
        )

        expect(events.map((event) => event._tag)).toEqual(["prefill"])
        expect(programState.closed).toBe(true)
        expect(samples).toBe(0)
        const error = yield* Effect.flip(Tensor.toNumberArray(programState.logits[0]!))
        expect(error.message).toContain("cleared")
      }))

    it.effect("clears consumed logits before emitting events or requesting the next row", () =>
      Effect.gen(function*() {
        const programState: LogitsProgramState = { closed: false, logits: Array<Tensor.Concrete>() }
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
              expect(programState.logits).toHaveLength(1)
            })
          }),
          Stream.runDrain
        )
        expect(inspected).toBe(true)
        expect(programState.logits).toHaveLength(2)
        const secondError = yield* Effect.flip(Tensor.toNumberArray(programState.logits[1]!))
        expect(secondError.message).toContain("cleared")
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
          sampling: { temperature: 0, topK: 0, topP: 1, seed: 7 },
          maxTokens: 2,
          eosTokens: [EOS]
        }])
        expect(programState.sampled?.step).toEqual([{
          sampling: { temperature: 0, topK: 0, topP: 1, seed: 7 }
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

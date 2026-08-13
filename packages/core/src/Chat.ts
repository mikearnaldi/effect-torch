/**
 * Streaming conversational inference over structured chat messages.
 *
 * @since 0.1.0
 */
import { Data, Effect, Option, Stream } from "effect"
import type * as Model from "./Model.ts"
import type * as Runtime from "./Runtime.ts"
import * as Tensor from "./Tensor.ts"

/**
 * A conversational inference failure.
 *
 * @since 0.1.0
 * @category errors
 */
export class ChatError extends Data.TaggedError("ChatError")<{
  readonly op: "validate" | "template" | "sample"
  readonly message: string
}> {}

/**
 * One structured message supplied to a model's Jinja chat template.
 *
 * @since 0.1.0
 * @category models
 */
export interface ChatMessage {
  readonly role: string
  readonly content?: unknown | undefined
  readonly [field: string]: unknown
}

/**
 * The tokenizer operations required by {@link stream}.
 *
 * @since 0.1.0
 * @category models
 */
export interface ChatTokenizer<E = never> {
  readonly applyChatTemplate: (
    template: string,
    messages: ReadonlyArray<ChatMessage>,
    options: {
      readonly addGenerationPrompt?: boolean | undefined
      readonly variables?: Readonly<Record<string, unknown>> | undefined
    }
  ) => Effect.Effect<string, E>
  readonly encode: (
    text: string,
    options?: { readonly addSpecialTokens?: boolean | undefined }
  ) => Effect.Effect<{ readonly data: Uint32Array }, E>
  readonly decode: (
    ids: ReadonlyArray<number>,
    options?: { readonly skipSpecialTokens?: boolean | undefined }
  ) => Effect.Effect<string, E>
  readonly tokenToId: (token: string) => Option.Option<number>
  readonly idToToken: (id: number) => Option.Option<string>
}

/**
 * Selects the next token from one logits row. The default is {@link greedy}.
 *
 * @since 0.1.0
 * @category models
 */
export type ChatSampler = (logits: Tensor.TypedArray) => number

/**
 * Greedy argmax sampling.
 *
 * @since 0.1.0
 * @category constructors
 */
export const greedy: ChatSampler = (logits) => {
  let selected = 0
  for (let index = 1; index < logits.length; index++) {
    if (Number(logits[index]) > Number(logits[selected])) selected = index
  }
  return selected
}

/**
 * Control-token strings for start/message/end-of-message segmented chat
 * formats. Set `controls` to `false` in {@link ChatStreamOptions} for an
 * unsegmented assistant response.
 *
 * @since 0.1.0
 * @category models
 */
export interface ChatControlTokens {
  readonly start: string
  readonly message: string
  readonly endOfMessage: string
  readonly endOfTurn?: string | undefined
  readonly endOfText: string
}

const defaultControls: ChatControlTokens = {
  start: "<|start|>",
  message: "<|message|>",
  endOfMessage: "<|eom|>",
  endOfTurn: "<|eot|>",
  endOfText: "<|end_of_text|>"
}

/**
 * Structured response segment classification.
 *
 * @since 0.1.0
 * @category models
 */
export type ChatSegmentKind = "content" | "reasoning" | "tool" | "other"

/**
 * One assistant, tool, or other response segment.
 *
 * @since 0.1.0
 * @category models
 */
export interface ChatSegment {
  readonly index: number
  readonly role: string
  readonly recipient?: string | undefined
  readonly kind: ChatSegmentKind
}

/**
 * Why one segment ended.
 *
 * @since 0.1.0
 * @category models
 */
export type ChatSegmentFinish = "message" | "turn" | "limit"

/**
 * A completed response segment with its decoded content.
 *
 * @since 0.1.0
 * @category models
 */
export interface CompletedChatSegment extends ChatSegment {
  readonly content: string
  readonly finish: ChatSegmentFinish
}

/**
 * Prompt and decode timing/token statistics.
 *
 * @since 0.1.0
 * @category models
 */
export interface ChatStats {
  readonly promptTokens: number
  readonly generatedTokens: number
  readonly prefillMs: number
  readonly decodeMs: number
  readonly decodeTokensPerSecond: number
}

/**
 * The final accumulated response.
 *
 * @since 0.1.0
 * @category models
 */
export interface ChatResult {
  readonly content: string
  readonly reasoning: string
  readonly segments: ReadonlyArray<CompletedChatSegment>
  readonly finishReason: "stop" | "maxTokens"
  readonly stats: ChatStats
}

/**
 * One streaming conversational inference event.
 *
 * @since 0.1.0
 * @category models
 */
export type ChatEvent =
  | { readonly _tag: "prefill"; readonly tokens: number; readonly durationMs: number }
  | { readonly _tag: "start"; readonly segment: ChatSegment }
  | { readonly _tag: "delta"; readonly segment: ChatSegment; readonly text: string }
  | { readonly _tag: "end"; readonly segment: ChatSegment; readonly finish: ChatSegmentFinish }
  | { readonly _tag: "done"; readonly result: ChatResult }

/**
 * Configuration for {@link stream}.
 *
 * @since 0.1.0
 * @category models
 */
export interface ChatStreamOptions<E = never> {
  readonly program: Model.InferenceProgram
  readonly tokenizer: ChatTokenizer<E>
  readonly template: string
  readonly messages: ReadonlyArray<ChatMessage>
  readonly addGenerationPrompt?: boolean | undefined
  readonly variables?: Readonly<Record<string, unknown>> | undefined
  readonly bosTokenId?: number | undefined
  /** Optional application-side generation limit; omitted means run until a stop token. */
  readonly maxTokens?: number | undefined
  readonly sample?: ChatSampler | undefined
  readonly controls?: Partial<ChatControlTokens> | false | undefined
  readonly stopTokens?: ReadonlyArray<number> | undefined
}

const fail = (op: ChatError["op"], message: string): ChatError => new ChatError({ op, message })

const requireTokenId = <E>(
  tokenizer: ChatTokenizer<E>,
  token: string
): Effect.Effect<number, ChatError> =>
  Option.match(tokenizer.tokenToId(token), {
    onNone: () => fail("validate", `chat control token ${JSON.stringify(token)} is not in the tokenizer`),
    onSome: (id) => Effect.succeed(id)
  })

const segmentKind = (role: string, recipient: string | undefined): ChatSegmentKind =>
  role === "assistant"
    ? recipient === "self"
      ? "reasoning"
      : recipient === undefined || recipient === "user"
      ? "content"
      : "tool"
    : "other"

interface ResolvedControls {
  readonly start: number
  readonly message: number
  readonly endOfMessage: number
  readonly endOfTurn: number | undefined
}

interface Parser<E> {
  readonly accept: (token: number) => Effect.Effect<Array<ChatEvent>, E>
  readonly finish: (finish: ChatSegmentFinish) => Array<ChatEvent>
  readonly segments: () => ReadonlyArray<CompletedChatSegment>
}

const makeParser = <E>(
  tokenizer: ChatTokenizer<E>,
  controls: ResolvedControls | undefined,
  initialRole: string,
  startsInHeader: boolean
): Parser<E> => {
  let state: "header" | "content" | "seekStart" | "done" = startsInHeader ? "header" : "seekStart"
  let role = initialRole
  let recipient: string | undefined
  let headerIds: Array<number> = []
  let contentIds: Array<number> = []
  let content = ""
  let segmentIndex = 0
  let current: ChatSegment | undefined
  const completed: Array<CompletedChatSegment> = []

  const begin = (): ChatEvent => {
    current = {
      index: segmentIndex++,
      role,
      ...(recipient === undefined ? {} : { recipient }),
      kind: segmentKind(role, recipient)
    }
    return { _tag: "start", segment: current }
  }

  const end = (finish: ChatSegmentFinish): Array<ChatEvent> => {
    if (current === undefined) return []
    completed.push({ ...current, content, finish })
    const event: ChatEvent = { _tag: "end", segment: current, finish }
    current = undefined
    headerIds = []
    contentIds = []
    content = ""
    if (finish === "turn" || finish === "limit") state = "done"
    else state = "seekStart"
    return [event]
  }

  if (controls === undefined) {
    return {
      accept: (token) =>
        Effect.gen(function*() {
          const events: Array<ChatEvent> = []
          if (current === undefined) events.push(begin())
          contentIds.push(token)
          const text = yield* tokenizer.decode(contentIds, { skipSpecialTokens: true })
          const delta = text.slice(content.length)
          content = text
          if (delta.length > 0 && current !== undefined) {
            events.push({ _tag: "delta", segment: current, text: delta })
          }
          return events
        }),
      finish: end,
      segments: () => completed
    }
  }

  return {
    accept: (token) =>
      Effect.gen(function*() {
        if (state === "done") return []
        if (state === "seekStart") {
          if (token !== controls.start) return []
          state = "header"
          role = initialRole
          recipient = undefined
          headerIds = []
          contentIds = []
          content = ""
          return []
        }
        if (state === "header") {
          if (token !== controls.message) {
            headerIds.push(token)
            return []
          }
          const header = (yield* tokenizer.decode(headerIds, { skipSpecialTokens: true })).trim()
          const first = header.split(/\s+/, 1)[0] ?? ""
          role = first.length === 0 || first.startsWith("to=") ? initialRole : first
          recipient = /(?:^|\s)to=([^\s]+)/.exec(header)?.[1]
          state = "content"
          headerIds = []
          contentIds = []
          content = ""
          return [begin()]
        }
        if (token === controls.endOfMessage || token === controls.endOfTurn) {
          return end(token === controls.endOfMessage ? "message" : "turn")
        }
        contentIds.push(token)
        const text = yield* tokenizer.decode(contentIds, { skipSpecialTokens: true })
        const delta = text.slice(content.length)
        content = text
        return delta.length > 0 && current !== undefined
          ? [{ _tag: "delta", segment: current, text: delta }]
          : []
      }),
    finish: end,
    segments: () => completed
  }
}

/**
 * Renders structured chat messages, prefills them, and streams parsed response
 * events. The generation session is closed when the stream completes, fails,
 * or is interrupted.
 *
 * @since 0.1.0
 * @category constructors
 */
export const stream = <E = never>(
  options: ChatStreamOptions<E>
): Stream.Stream<
  ChatEvent,
  ChatError | E | Model.InferenceError | Model.ModelError | Tensor.TensorError,
  Runtime.Runtime
> =>
  Stream.scoped(Stream.unwrap(Effect.gen(function*() {
    if (options.template.length === 0) {
      return yield* fail("template", "chat template must be non-empty")
    }
    if (options.messages.length === 0) {
      return yield* fail("validate", "messages must not be empty")
    }
    if (
      options.maxTokens !== undefined &&
      (!Number.isSafeInteger(options.maxTokens) || options.maxTokens <= 0)
    ) {
      return yield* fail("validate", `maxTokens must be a positive integer, got ${options.maxTokens}`)
    }
    const tokenizer = options.tokenizer
    const controls = options.controls === false
      ? undefined
      : { ...defaultControls, ...options.controls }
    const resolvedControls = controls === undefined
      ? undefined
      : {
        start: yield* requireTokenId(tokenizer, controls.start),
        message: yield* requireTokenId(tokenizer, controls.message),
        endOfMessage: yield* requireTokenId(tokenizer, controls.endOfMessage),
        endOfTurn: controls.endOfTurn === undefined
          ? undefined
          : yield* requireTokenId(tokenizer, controls.endOfTurn)
      }
    const stopTokens = options.stopTokens === undefined
      ? new Set([
        ...(resolvedControls?.endOfTurn === undefined ? [] : [resolvedControls.endOfTurn]),
        yield* requireTokenId(tokenizer, controls?.endOfText ?? defaultControls.endOfText)
      ])
      : new Set(options.stopTokens)
    const bosToken = options.bosTokenId === undefined
      ? Option.none<string>()
      : tokenizer.idToToken(options.bosTokenId)
    if (options.bosTokenId !== undefined && Option.isNone(bosToken)) {
      return yield* fail("validate", `bosTokenId ${options.bosTokenId} is not in the tokenizer`)
    }
    const rendered = yield* tokenizer.applyChatTemplate(options.template, options.messages, {
      addGenerationPrompt: options.addGenerationPrompt ?? true,
      variables: {
        ...(Option.isSome(bosToken) ? { bos_token: bosToken.value } : {}),
        ...options.variables
      }
    })
    const encoded = yield* tokenizer.encode(rendered, { addSpecialTokens: false })
    const generation = yield* Effect.acquireRelease(
      options.program.generation(),
      (generation) => Effect.ignore(generation.close())
    )
    const prefillStarted = Date.now()
    const entry = yield* generation.add(
      yield* Tensor.fromTypedArray(encoded.data, [1, encoded.data.length])
    )
    const prefillMs = Date.now() - prefillStarted
    const parser = makeParser(
      tokenizer,
      resolvedControls,
      "assistant",
      options.addGenerationPrompt ?? true
    )
    const decodeStarted = Date.now()
    let generatedTokens = 0

    type State =
      | { readonly _tag: "prefill" }
      | { readonly _tag: "run"; readonly logits: Tensor.Concrete; readonly step: number }

    return Stream.paginate(
      { _tag: "prefill" } satisfies State as State,
      (state): Effect.Effect<
        readonly [ReadonlyArray<ChatEvent>, Option.Option<State>],
        ChatError | E | Model.InferenceError | Model.ModelError | Tensor.TensorError,
        Runtime.Runtime
      > => {
        if (state._tag === "prefill") {
          return Effect.succeed([
            [{ _tag: "prefill", tokens: encoded.data.length, durationMs: prefillMs }] satisfies Array<ChatEvent>,
            Option.some({ _tag: "run", logits: entry.logits, step: 0 } satisfies State)
          ])
        }
        return Effect.gen(function*() {
          const logits = state.logits
          const values = yield* Tensor.toTypedArray(logits).pipe(
            Effect.ensuring(Effect.ignore(Tensor.clear(logits)))
          )
          if (values.length === 0) {
            return yield* fail("sample", "model produced an empty logits row")
          }
          const token = yield* Effect.try({
            try: () => (options.sample ?? greedy)(values),
            catch: (error) => fail("sample", error instanceof Error ? error.message : String(error))
          })
          if (!Number.isSafeInteger(token) || token < 0 || token >= values.length) {
            return yield* fail("sample", `sampler returned invalid token ${token} for ${values.length} logits`)
          }
          const events = yield* parser.accept(token)
          const stopped = stopTokens.has(token)
          if (!stopped) generatedTokens++
          if (stopped || (options.maxTokens !== undefined && state.step + 1 >= options.maxTokens)) {
            const finishReason = stopped ? "stop" : "maxTokens"
            events.push(...parser.finish(stopped ? "turn" : "limit"))
            const decodeMs = Date.now() - decodeStarted
            const segments = parser.segments()
            const result: ChatResult = {
              content: segments.filter((segment) => segment.kind === "content").map((segment) => segment.content)
                .join(""),
              reasoning: segments.filter((segment) => segment.kind === "reasoning").map((segment) => segment.content)
                .join("\n\n"),
              segments,
              finishReason,
              stats: {
                promptTokens: encoded.data.length,
                generatedTokens,
                prefillMs,
                decodeMs,
                decodeTokensPerSecond: generatedTokens === 0 || decodeMs === 0
                  ? 0
                  : generatedTokens * 1000 / decodeMs
              }
            }
            events.push({ _tag: "done", result })
            return [events, Option.none<State>()]
          }
          const [next] = yield* generation.step([{ seq: entry.seq, token }])
          return [
            events,
            Option.some({ _tag: "run", logits: next, step: state.step + 1 } satisfies State)
          ]
        })
      }
    )
  })))

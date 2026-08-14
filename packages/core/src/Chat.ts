/**
 * Template-driven, streaming chat orchestration over compiled generation.
 *
 * This module is the boundary between four independently supplied contracts:
 * structured {@link ChatMessage}s, a Jinja-compatible chat template, a
 * {@link ChatTokenizer}, and a decode-specialized {@link Model.InferenceProgram}.
 * {@link stream} renders messages once, encodes the complete prompt with
 * tokenizer-added special tokens disabled, prefills one generation sequence,
 * then repeatedly reads one logits row, selects a token synchronously, parses
 * it into {@link ChatEvent}s, and steps the sequence. It does not own a template
 * language, tokenizer vocabulary, conversation history store, tool executor,
 * or general sampling policy.
 *
 * Structured parsing targets start/header/message/end control-token formats.
 * Parser delimiters and tokenizer-derived default stops must be atomic tokens
 * addressable through `tokenToId`; generated headers are decoded as a
 * whitespace-delimited role with an optional unquoted `to=<recipient>` field.
 * Setting `controls: false`
 * bypasses that protocol and treats the generated text as one assistant content
 * segment. In either mode, deltas are computed by repeatedly decoding all
 * accumulated content ids and slicing the newly appended suffix. Tokenizer
 * decode must therefore be prefix-stable for emitted ids; this event protocol
 * has no replacement/retraction event for a decoder that revises prior text.
 *
 * The returned stream acquires one ordinary {@link Model.Generation} session
 * and attempts to close all of its live sequence state on normal completion,
 * failure, or interruption. Generated logits are internal tensors rather than event
 * payloads. `done` is emitted only for normal stop-token or `maxTokens`
 * termination; failure, interruption, or downstream cancellation may end the
 * stream without `end` or `done` events.
 *
 * @since 0.1.0
 */
import { Data, Effect, Option, Stream } from "effect"
import type * as Model from "./Model.ts"
import type * as Runtime from "./Runtime.ts"
import * as Tensor from "./Tensor.ts"

/**
 * A failure in chat-owned validation, template selection, or sampling.
 * Tokenizer/template-engine failures keep the tokenizer's generic error type;
 * compiled inference and tensor failures likewise keep their original types.
 * `message` is diagnostic text, not a stable machine-readable protocol.
 *
 * @since 0.1.0
 * @category errors
 */
export class ChatError extends Data.TaggedError("ChatError")<{
  /** Chat phase that detected the failure. */
  readonly op: "validate" | "template" | "sample"
  /** Human-readable diagnostic. */
  readonly message: string
}> {}

/**
 * One structured message passed unchanged to the chat-template engine.
 * `role` is the only field required by this TypeScript contract; `content` and
 * arbitrary extra fields may contain whatever the selected template supports.
 * {@link stream} does not validate role names, content schemas, tool-call
 * structures, chronology, or whether values are serializable by the engine.
 *
 * @since 0.1.0
 * @category models
 */
export interface ChatMessage {
  /** Template-defined role label, conventionally `system`, `user`, `assistant`, or `tool`. */
  readonly role: string
  /** Optional template-defined payload; strings are not required. */
  readonly content?: unknown | undefined
  /** Additional template-specific message fields. */
  readonly [field: string]: unknown
}

/**
 * The template/tokenizer operations required by {@link stream}.
 *
 * Chat owns orchestration but not normalization or vocabulary semantics. The
 * implementation must use one vocabulary consistently across template special
 * token strings, `encode`, `decode`, `tokenToId`, `idToToken`, and the inference
 * program's logits indices. `decode` is called repeatedly on growing id arrays
 * with `skipSpecialTokens: true`; emitted text assumes each result starts with
 * the previous result. Control strings must map directly to one id rather than
 * requiring `encode` into multiple ids.
 *
 * @since 0.1.0
 * @category models
 */
export interface ChatTokenizer<E = never> {
  /** Renders messages with the caller's template and variables. */
  readonly applyChatTemplate: (
    template: string,
    messages: ReadonlyArray<ChatMessage>,
    options: {
      readonly addGenerationPrompt?: boolean | undefined
      readonly variables?: Readonly<Record<string, unknown>> | undefined
    }
  ) => Effect.Effect<string, E>
  /**
   * Encodes the already-rendered prompt as u32 ids; chat always disables added
   * specials. Consequently, {@link stream} requires an inference program
   * compiled with `tokenDtype: "u32"`.
   */
  readonly encode: (
    text: string,
    options?: { readonly addSpecialTokens?: boolean | undefined }
  ) => Effect.Effect<{ readonly data: Uint32Array }, E>
  /** Decodes generated header/content ids; chat requests special-token skipping. */
  readonly decode: (
    ids: ReadonlyArray<number>,
    options?: { readonly skipSpecialTokens?: boolean | undefined }
  ) => Effect.Effect<string, E>
  /** Resolves an atomic control-token string to its vocabulary id. */
  readonly tokenToId: (token: string) => Option.Option<number>
  /** Resolves `bosTokenId` to the template string injected as `bos_token`. */
  readonly idToToken: (id: number) => Option.Option<string>
}

/**
 * Synchronously selects one vocabulary index from a host logits row. The array
 * is raw model output, not probabilities; its concrete typed-array class follows
 * the logits tensor dtype/readback rules. Return a non-negative safe integer
 * less than `logits.length`. Thrown exceptions become `ChatError("sample")`.
 * The default is {@link greedy}; temperature, penalties, top-k/top-p filtering,
 * randomness, and sampler state are application concerns.
 *
 * @since 0.1.0
 * @category models
 */
export type ChatSampler = (logits: Tensor.TypedArray) => number

/**
 * Greedy argmax sampling. Ties select the lowest index. Values are compared as
 * JavaScript numbers; this function does not reject NaN or infinite logits.
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
 * Atomic control-token strings for a start/header/message segmented response.
 * The expected generated wire form is conceptually
 * `<start><role> [to=<recipient>]<message><content><endOfMessage|endOfTurn>`.
 * With a generation prompt, parsing starts inside the first header and assumes
 * the template has already established the start/assistant context. Without
 * one, generated ids are ignored until `start` appears.
 *
 * `start`, `message`, and `endOfMessage` delimit parsing. `endOfTurn`, when
 * present, ends the current segment and is a default stop token. `endOfText` is
 * a default stop token but is not otherwise parser syntax. Parser delimiters
 * always resolve through {@link ChatTokenizer.tokenToId}; `endOfText` resolves
 * only when `stopTokens` is omitted. Set `controls` to `false` for one
 * unsegmented assistant response.
 *
 * @since 0.1.0
 * @category models
 */
export interface ChatControlTokens {
  /** Begins a generated segment header. */
  readonly start: string
  /** Ends the header and begins decoded segment content. */
  readonly message: string
  /** Ends one segment while permitting a later `start`. */
  readonly endOfMessage: string
  /** Optionally ends the current turn; also a default stop token. */
  readonly endOfTurn?: string | undefined
  /** Ends generation by default; it has no structural parser role. */
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
 * Heuristic structured-response classification derived from the parsed role
 * and recipient. Assistant-to-self is `reasoning`; assistant with no recipient
 * or recipient `user` is `content`; other assistant recipients are `tool`; a
 * non-assistant role is `other`.
 *
 * @since 0.1.0
 * @category models
 */
export type ChatSegmentKind = "content" | "reasoning" | "tool" | "other"

/**
 * Identity and classification of one parsed response segment. The same value is
 * attached to that segment's `start`, `delta`, and `end` events. Indexes are
 * zero-based and increase only when a header reaches `message` (or when the
 * unsegmented parser accepts its first token).
 *
 * @since 0.1.0
 * @category models
 */
export interface ChatSegment {
  /** Zero-based segment order within this stream. */
  readonly index: number
  /** Parsed first header word, or the default `assistant` role. */
  readonly role: string
  /** Unquoted non-whitespace value parsed from a `to=<recipient>` header field. */
  readonly recipient?: string | undefined
  /** Classification derived by chat's fixed role/recipient heuristic. */
  readonly kind: ChatSegmentKind
}

/**
 * Why one started segment ended. `message` means `endOfMessage`; `turn` means
 * `endOfTurn` or an otherwise-open segment closed by a stop token; `limit`
 * means an otherwise-open segment closed at `maxTokens`. If the final sampled
 * id is itself a segment delimiter, that delimiter's reason wins before outer
 * stop/limit handling. A stop token outside a started segment emits no `end`.
 *
 * @since 0.1.0
 * @category models
 */
export type ChatSegmentFinish = "message" | "turn" | "limit"

/**
 * A completed response segment with the latest full decoded content. Only
 * segments that emitted `start` and subsequently ended appear in results;
 * ignored pre-header tokens and incomplete headers do not.
 *
 * @since 0.1.0
 * @category models
 */
export interface CompletedChatSegment extends ChatSegment {
  readonly content: string
  readonly finish: ChatSegmentFinish
}

/**
 * Prompt and decode statistics measured with wall-clock `Date.now()`.
 * Durations are coarse elapsed times and are not monotonic device-kernel
 * profiling. Prompt rendering/encoding and control validation happen before
 * `prefillMs`; `decodeMs` starts after prefill and includes event consumption
 * backpressure, readback, sampling, tokenizer decoding, and decode steps.
 *
 * @since 0.1.0
 * @category models
 */
export interface ChatStats {
  /** Number of ids returned by prompt encoding, including any template-rendered specials. */
  readonly promptTokens: number
  /** Sampled non-stop ids, including parser controls and ignored/header ids. */
  readonly generatedTokens: number
  /** Elapsed milliseconds for prompt tensor construction plus `Generation.add`. */
  readonly prefillMs: number
  /** Elapsed milliseconds from completed prefill until normal termination. */
  readonly decodeMs: number
  /** `generatedTokens * 1000 / decodeMs`, or zero when either operand is zero. */
  readonly decodeTokensPerSecond: number
}

/**
 * Final accumulation emitted by the sole `done` event on normal termination.
 * `content` concatenates all completed `content` segments without a separator;
 * `reasoning` joins completed `reasoning` segments with two newlines. Tool and
 * `other` segments remain available only through `segments`.
 *
 * @since 0.1.0
 * @category models
 */
export interface ChatResult {
  /** Concatenation of completed segments classified as `content`. */
  readonly content: string
  /** Completed `reasoning` segments joined with `"\n\n"`. */
  readonly reasoning: string
  /** Completed segments in event order. */
  readonly segments: ReadonlyArray<CompletedChatSegment>
  /** Whether termination came from a stop id or the application token limit. */
  readonly finishReason: "stop" | "maxTokens"
  /** Wall-clock and token counters for this stream. */
  readonly stats: ChatStats
}

/**
 * One ordered streaming conversational inference event.
 *
 * A successful stream emits exactly one `prefill` first and one `done` last.
 * Each completed parsed segment emits `start`, zero or more nonempty `delta`s,
 * then `end`; multiple events may be emitted for one sampled token. Control and
 * header tokens generally emit no event. `delta.text` is a decoded string
 * suffix, not necessarily one token or one Unicode code point. A sampled stop
 * id is offered to the parser before stopping, is excluded from
 * `generatedTokens`, and is never passed to `Generation.step`. At `maxTokens`,
 * the final sampled non-stop token is parsed and counted but likewise is not
 * stepped because no subsequent logits are needed.
 *
 * Stream failure, interruption, or downstream cancellation performs scoped
 * cleanup but emits no synthetic `end` or `done` event.
 *
 * @since 0.1.0
 * @category models
 */
export type ChatEvent =
  /** Prompt prefill completed; always the first event on success. */
  | { readonly _tag: "prefill"; readonly tokens: number; readonly durationMs: number }
  /** A parsed segment began. */
  | { readonly _tag: "start"; readonly segment: ChatSegment }
  /** A nonempty append-only decoded suffix for the current segment. */
  | { readonly _tag: "delta"; readonly segment: ChatSegment; readonly text: string }
  /** The current segment completed. */
  | { readonly _tag: "end"; readonly segment: ChatSegment; readonly finish: ChatSegmentFinish }
  /** Normal generation completed; always the final event when present. */
  | { readonly _tag: "done"; readonly result: ChatResult }

/**
 * Configuration for one {@link stream} invocation. The program and tokenizer
 * must describe the same token-id vocabulary: encoded prompt ids are fed to the
 * program, logits indexes are returned to the tokenizer/parser, and control and
 * stop ids are compared numerically. Chat cannot validate that cross-component
 * agreement.
 *
 * @since 0.1.0
 * @category models
 */
export interface ChatStreamOptions<E = never> {
  /**
   * Compiled inference artifact used to open one generation session. It must
   * use `tokenDtype: "u32"`, matching `ChatTokenizer.encode`.
   */
  readonly program: Model.InferenceProgram
  /** Template/token vocabulary implementation paired with `program`. */
  readonly tokenizer: ChatTokenizer<E>
  /** Nonempty Jinja-compatible template passed verbatim to `applyChatTemplate`. */
  readonly template: string
  /** Nonempty structured history passed verbatim to the template engine. */
  readonly messages: ReadonlyArray<ChatMessage>
  /**
   * Passed to the template engine; defaults to `true`. It also selects the
   * parser's initial state: `true` assumes generation starts inside an assistant
   * header, while `false` waits for a generated `start` control token. Chat does
   * not inspect the rendered prompt to verify that this assumption is true.
   */
  readonly addGenerationPrompt?: boolean | undefined
  /**
   * Additional template variables. If `bosTokenId` is present, chat first
   * injects its token string as `bos_token`, then these variables are spread on
   * top and may override that value.
   */
  readonly variables?: Readonly<Record<string, unknown>> | undefined
  /**
   * Optional tokenizer id resolved with `idToToken` and exposed to the template
   * as `bos_token`. This does not prepend an id and prompt encoding still uses
   * `addSpecialTokens: false`.
   */
  readonly bosTokenId?: number | undefined
  /**
   * Positive application-side limit on sampled tokens, including a sampled stop
   * token for limit comparison but excluding that stop token from reported
   * `generatedTokens`. Omit to run until a configured stop id or failure; there
   * is no implicit safety limit.
   */
  readonly maxTokens?: number | undefined
  /** Synchronous next-token selector; defaults to {@link greedy}. */
  readonly sample?: ChatSampler | undefined
  /**
   * Partial override of the default segmented control strings, or `false` for
   * one unsegmented assistant response. In segmented mode the structural
   * controls are resolved as one tokenizer token before template rendering;
   * effective `endOfText` is also resolved when default stop ids are used.
   */
  readonly controls?: Partial<ChatControlTokens> | false | undefined
  /**
   * Exact numeric stop-id set. When omitted, chat resolves effective
   * `endOfTurn` when present plus effective `endOfText`. Supplying this option,
   * including `[]`, replaces those defaults; values are not prevalidated or
   * inferred from control strings.
   */
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

// The parser is intentionally token-level: delimiters must be atomic ids. Text
// is decoded from each complete accumulated id list so byte/BPE fragments can
// settle before a delta is emitted. Append-only events require prefix-stable
// decode output; a tokenizer that revises old text cannot be represented here.
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
 * Renders the supplied history once, encodes and prefills it, then samples and
 * parses one token at a time. Template rendering receives
 * `addGenerationPrompt` (default `true`) and merged variables. Encoding always
 * uses `addSpecialTokens: false`; templates are therefore responsible for all
 * model-required BOS/EOS/control text.
 *
 * Sampling reads the complete chat-owned logits tensor to a host typed array
 * and clears that tensor even if readback fails or is interrupted. A valid
 * non-stop token is parsed before being committed with `Generation.step`; the
 * final stop/limit token is parsed but not stepped because its successor logits
 * are not needed. Stop ids are protocol delimiters, not output filtering: a
 * custom stop id that decodes as text can emit a final delta before termination.
 *
 * The stream is scoped. Its generation session is closed on normal completion,
 * tokenizer/parser/model failure, interruption, or downstream cancellation.
 * Cleanup errors are ignored so they do not replace the primary exit. A logits
 * row is cleared after readback; each row also has a scoped fallback finalizer,
 * so an unread pagination row is released on downstream cancellation. Normal
 * termination emits `done`; other exits do not synthesize terminal events.
 *
 * Validation is intentionally narrow: the template and messages must be
 * nonempty, `maxTokens` must be a positive safe integer, parser controls,
 * default token-derived stops, and `bosTokenId` must resolve when used, and
 * sampler output must index the logits row.
 * Chat does not validate message schemas, template syntax, prompt non-emptiness
 * after encoding, stop-id ranges, control-id distinctness, tokenizer/program
 * vocabulary agreement, model vocabulary semantics, or decode prefix stability.
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
      (generation) => Effect.ignore(generation.close()),
      { interruptible: true }
    )
    const ownedLogits = new Set<Tensor.Concrete>()
    const releaseLogits = (logits: Tensor.Concrete) =>
      !ownedLogits.has(logits)
        ? Effect.void
        : Tensor.clear(logits).pipe(
          Effect.tap(() => Effect.sync(() => ownedLogits.delete(logits))),
          Effect.ignore
        )
    yield* Effect.addFinalizer(() => Effect.ignore(Tensor.clearAll(ownedLogits)))
    const prefillStarted = Date.now()
    const prompt = yield* Tensor.fromTypedArray(encoded.data, [1, encoded.data.length])
    const entry = yield* Effect.uninterruptibleMask((restore) =>
      Effect.tap(
        restore(generation.add(prompt)),
        (entry) => Effect.sync(() => ownedLogits.add(entry.logits))
      )
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

    // The current logits ownership travels in pagination state. Each run page
    // reads and clears it before either terminating or installing the next one.
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
            Effect.ensuring(releaseLogits(logits))
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
          const [next] = yield* Effect.uninterruptibleMask((restore) =>
            Effect.tap(
              restore(generation.step([{ seq: entry.seq, token }])),
              (outputs) => Effect.sync(() => outputs.forEach((output) => ownedLogits.add(output)))
            )
          )
          return [
            events,
            Option.some({ _tag: "run", logits: next, step: state.step + 1 } satisfies State)
          ]
        })
      }
    )
  })))

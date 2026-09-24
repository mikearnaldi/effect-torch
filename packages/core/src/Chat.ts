/**
 * Streams templated chat messages from autoregressive or block generation.
 *
 * This module connects four caller-supplied pieces: structured
 * {@link ChatMessage}s, a Jinja-compatible chat template, a
 * {@link ChatTokenizer}, and a generation operation. {@link streamWith}
 * consumes nonempty committed token pages from any family. Zero-token
 * refinement iterations stay inside the generator.
 * {@link stream} renders messages once, encodes the complete prompt with
 * tokenizer-added special tokens disabled, prefills one model sequence, then
 * repeatedly parses one sampled token into {@link ChatEvent}s before stepping
 * the sequence. Standard sampling uses a {@link AutoRegressive.Generation} session and
 * exposes no logits. A custom host callback uses a
 * {@link AutoRegressive.StatefulExecution} session and reads each logits row back.
 * Chat does not own a template language, tokenizer vocabulary, conversation
 * history store, or tool executor.
 *
 * Structured parsing targets start/header/message/end control-token formats.
 * Parser delimiters and tokenizer-derived default stops must be atomic tokens
 * addressable through `tokenToId`. The parser decodes generated headers as a
 * whitespace-delimited role with an optional unquoted `to=<recipient>` field.
 * Setting `controls: false` bypasses that protocol and treats the generated
 * text as one assistant content segment. In either mode, chat computes deltas
 * by repeatedly decoding all accumulated content ids and slicing the newly
 * appended suffix. Tokenizer decode must be prefix-stable for emitted ids. The
 * event protocol has no replacement/retraction event for a decoder that revises
 * prior text.
 *
 * The autoregressive stream opens one {@link AutoRegressive.Generation} session for standard
 * sampling or one {@link AutoRegressive.StatefulExecution} session for a custom sampler.
 * It attempts to close the session and all live sequence state on normal
 * completion, failure, or interruption. Custom-sampler logits remain internal
 * tensors rather than event payloads. `done` is emitted only for normal
 * stop-token, generator-policy, or `maxTokens` termination. Failure, interruption, or downstream
 * cancellation may end the stream without `end` or `done` events.
 *
 * @since 0.1.0
 */
import { Data, Deferred, Effect, Exit, Option, Predicate, Queue, type Scope, Stream } from "effect"
import type * as AutoRegressive from "./AutoRegressive.ts"
import type * as Generation from "./Generation.ts"
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
export interface ChatMessage<Value = unknown> {
  /** Template-defined role label, conventionally `system`, `user`, `assistant`, or `tool`. */
  readonly role: string
  /** Optional template-defined payload; strings are not required. */
  readonly content?: Value | undefined
  /** Additional template-specific message fields. */
  readonly [field: string]: Value | string | undefined
}

/**
 * The template/tokenizer operations required by {@link stream}.
 *
 * Chat coordinates these operations but does not define normalization or
 * vocabulary rules. The implementation must use one vocabulary consistently
 * across template special token strings, `encode`, `decode`, `tokenToId`,
 * `idToToken`, and the inference program's logits indices. Chat uses
 * `decodeStream` when supplied. Otherwise it repeatedly calls `decode` on
 * growing id arrays with `skipSpecialTokens: true`; emitted text assumes each
 * result starts with the previous result. Control strings must map directly to
 * one id rather than requiring `encode` into multiple ids.
 *
 * @since 0.1.0
 * @category models
 */
export interface ChatTokenizer<E = never, Value = unknown> {
  /** Renders messages with the caller's template and variables. */
  readonly applyChatTemplate: (
    template: string,
    messages: ReadonlyArray<ChatMessage<Value>>,
    options: {
      readonly addGenerationPrompt?: boolean | undefined
      readonly variables?: Readonly<Record<string, Value | string | undefined>> | undefined
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
  /** Creates an independent incremental decoder for generated content. */
  readonly decodeStream?:
    | ((options?: { readonly skipSpecialTokens?: boolean | undefined }) => {
      readonly step: (id: number) => Effect.Effect<string | undefined, E>
    })
    | undefined
  /** Resolves an atomic control-token string to its vocabulary id. */
  readonly tokenToId: (token: string) => Option.Option<number>
  /** Resolves `bosTokenId` to the template string injected as `bos_token`. */
  readonly idToToken: (id: number) => Option.Option<string>
}

/**
 * Custom host-side vocabulary selector. Supplying one requires full logits
 * readback; prefer {@link ChatSamplingOptions} for native temperature, top-k,
 * and top-p sampling. Return a non-negative safe integer less than
 * `logits.length`. Thrown exceptions become `ChatError("sample")`.
 *
 * @since 0.1.0
 * @category models
 */
export type ChatSampler = (logits: Tensor.TypedArray) => number

/**
 * Standard next-token sampling controls. Temperature defaults to `0` (greedy),
 * `topK` to `0` (disabled), and `topP` to `1` (disabled). A missing seed is
 * generated once per stream. Each successful draw advances a stream-local
 * counter, so draws share no process-global sampler state. Chat forwards these
 * controls to a {@link AutoRegressive.Generation} session, which returns token ids
 * without exposing logits. Metal requires `topK` in `1..=64` for
 * positive-temperature `topP` filtering and rejects positive-temperature
 * `topK > 64`.
 *
 * @since 0.1.0
 * @category models
 */
export interface ChatSamplingOptions {
  /** Non-negative sampling temperature; `0` selects greedy sampling. Defaults to `0`. */
  readonly temperature?: number | undefined
  /** Non-negative top-k candidate count; `0` disables top-k filtering. Defaults to `0`. */
  readonly topK?: number | undefined
  /** Nucleus probability in `(0, 1]`; `1` disables top-p filtering. Defaults to `1`. */
  readonly topP?: number | undefined
  /** Non-negative safe-integer seed; omitted seeds are generated once per stream. */
  readonly seed?: number | undefined
}

/**
 * Standard sampling controls or a custom host-side logits callback.
 *
 * @since 0.1.0
 * @category models
 */
export type ChatSampling = ChatSamplingOptions | ChatSampler

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
 * Atomic control-token strings for a segmented start/header/message response.
 * The generated wire form is
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
 * Chat uses a fixed heuristic to classify the parsed role and recipient.
 * Assistant-to-self is `reasoning`. Assistant with no recipient or recipient
 * `user` is `content`. Other assistant recipients are `tool`, and a
 * non-assistant role is `other`.
 *
 * @since 0.1.0
 * @category models
 */
export type ChatSegmentKind = "content" | "reasoning" | "tool" | "other"

/**
 * Identity and classification of one parsed response segment. The same value is
 * attached to that segment's `start`, `delta`, and `end` events. Indexes start
 * at zero. They increase only when a header reaches `message` or the
 * unsegmented parser accepts its first token.
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
 * segments that emitted `start` and later ended appear in results;
 * ignored pre-header tokens and incomplete headers do not.
 *
 * @since 0.1.0
 * @category models
 */
export interface CompletedChatSegment extends ChatSegment {
  /** Final full decoded content for this segment. */
  readonly content: string
  /** Reason the segment closed. */
  readonly finish: ChatSegmentFinish
}

/**
 * Prompt and decode statistics measured with wall-clock `Date.now()`.
 * Durations are coarse elapsed times, not monotonic device-kernel profiling.
 * Prompt rendering, encoding, and control validation happen before `prefillMs`.
 * `decodeMs` starts after prefill. It includes event-consumption backpressure,
 * generation, custom-sampler readback, and tokenizer decoding. For streamWith,
 * prefill includes refinement until the first committed page is available.
 *
 * @since 0.1.0
 * @category models
 */
export interface ChatStats {
  /** Number of ids returned by prompt encoding, including any template-rendered specials. */
  readonly promptTokens: number
  /** Sampled non-stop ids, including parser controls and ignored/header ids. */
  readonly generatedTokens: number
  /** Elapsed milliseconds until the first token page or initial custom-sampler logits are available. */
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
  /** Whether termination came from a stop/policy or a generation token limit. */
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
 * `generatedTokens`, and is never passed to a generation step. At `maxTokens`,
 * the final sampled non-stop token is parsed and counted but likewise is not
 * stepped because no subsequent token or logits are needed.
 *
 * Stream failure, interruption, or downstream cancellation performs scoped
 * cleanup but emits no synthetic `end` or `done` event.
 *
 * @since 0.1.0
 * @category models
 */
export type ChatEvent =
  /** Prompt prefill completed; always the first event on success. */
  | {
    readonly _tag: "prefill"
    readonly tokens: number
    readonly durationMs: number
  }
  /** A parsed segment began. */
  | {
    readonly _tag: "start"
    readonly segment: ChatSegment
  }
  /** A nonempty append-only decoded suffix for the current segment. */
  | {
    readonly _tag: "delta"
    readonly segment: ChatSegment
    readonly text: string
  }
  /** The current segment completed. */
  | {
    readonly _tag: "end"
    readonly segment: ChatSegment
    readonly finish: ChatSegmentFinish
  }
  /** Normal generation completed; always the final event when present. */
  | {
    readonly _tag: "done"
    readonly result: ChatResult
  }

/**
 * Template, tokenizer, and parser settings shared by {@link stream} and
 * {@link streamWith}. The generator and tokenizer must describe the same
 * token-id vocabulary. Chat compares control and stop ids numerically but
 * cannot validate that the components agree.
 *
 * @since 0.1.0
 * @category models
 */
export interface ChatOptions<E = never, Value = unknown> {
  /** Template/token vocabulary implementation paired with the generator. */
  readonly tokenizer: ChatTokenizer<E, Value>
  /** Nonempty Jinja-compatible template passed verbatim to `applyChatTemplate`. */
  readonly template: string
  /** Nonempty structured history passed verbatim to the template engine. */
  readonly messages: ReadonlyArray<ChatMessage<Value>>
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
  readonly variables?: Readonly<Record<string, Value | string | undefined>> | undefined
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

/**
 * Autoregressive generation and sampling for {@link stream}.
 *
 * @since 0.1.0
 * @category models
 */
export interface ChatStreamOptions<E = never, Value = unknown> extends ChatOptions<E, Value> {
  /** Compiled artifact using u32 tokens in the tokenizer vocabulary. */
  readonly program: AutoRegressive.Artifact
  /** Native sampling controls, or a custom caller-owned-logits selector. */
  readonly sampling?: ChatSampling | undefined
}

const fail = (op: ChatError["op"], message: string): ChatError => new ChatError({ op, message })

const requireTokenId = <E, Value>(
  tokenizer: ChatTokenizer<E, Value>,
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

// The parser works at the token level, so delimiters must be atomic ids. It
// prefers a stateful decoder and otherwise decodes each accumulated id list,
// allowing byte/BPE fragments to settle before emitting a delta. Append-only
// events require prefix-stable output.
const makeParser = <E, Value>(
  tokenizer: ChatTokenizer<E, Value>,
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
  const newContentDecoder = () => tokenizer.decodeStream?.({ skipSpecialTokens: true })
  let contentDecoder = controls === undefined ? newContentDecoder() : undefined

  const decodeContent = (token: number): Effect.Effect<string, E> => {
    if (contentDecoder !== undefined) {
      return Effect.map(contentDecoder.step(token), (delta) => {
        if (delta === undefined) return ""

        content += delta

        return delta
      })
    }

    return Effect.map(
      tokenizer.decode(contentIds, { skipSpecialTokens: true }),
      (text) => {
        const delta = text.slice(content.length)
        content = text

        return delta
      }
    )
  }

  const begin = (): ChatEvent => {
    current = {
      index: segmentIndex++,
      role,
      recipient,
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
    contentDecoder = undefined

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
          const delta = yield* decodeContent(token)

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
          contentDecoder = newContentDecoder()

          return [begin()]
        }

        if (token === controls.endOfMessage || token === controls.endOfTurn) {
          return end(token === controls.endOfMessage ? "message" : "turn")
        }

        contentIds.push(token)
        const delta = yield* decodeContent(token)

        return delta.length > 0 && current !== undefined
          ? [{ _tag: "delta", segment: current, text: delta }]
          : []
      }),
    finish: end,
    segments: () => completed
  }
}

type GenerationRead<E, R> =
  | {
    readonly _tag: "Page"
    readonly page: Effect.Effect<Generation.Page, E, R>
    readonly next: (token: number) => Effect.Effect<GenerationRead<E, R>, E, R>
  }
  | {
    readonly _tag: "Done"
    readonly finishReason: ChatResult["finishReason"]
  }

type OpenGeneration<E, R> = (
  request: Omit<Generation.Request, "onPage">
) => Effect.Effect<GenerationRead<E, R>, E, R | Scope.Scope>

type AutoregressiveError = ChatError | AutoRegressive.InferenceError | Model.ModelError | Tensor.TensorError

const resolveSampling = (options: ChatSampling | undefined) =>
  Effect.gen(function*() {
    let customSampler: ChatSampler | undefined
    let samplingOptions: ChatSamplingOptions | undefined

    if (Predicate.isFunction(options)) {
      customSampler = options
    } else {
      samplingOptions = options
    }

    const sampling = {
      temperature: samplingOptions?.temperature ?? 0,
      topK: samplingOptions?.topK ?? 0,
      topP: samplingOptions?.topP ?? 1,
      seed: samplingOptions?.seed ?? Math.floor(Math.random() * 0x1_0000_0000)
    }

    if (!Number.isFinite(sampling.temperature) || sampling.temperature < 0) {
      return yield* fail("validate", `temperature must be finite and non-negative, got ${sampling.temperature}`)
    }

    if (!Number.isSafeInteger(sampling.topK) || sampling.topK < 0) {
      return yield* fail("validate", `topK must be a non-negative integer, got ${sampling.topK}`)
    }

    if (!Number.isFinite(sampling.topP) || sampling.topP <= 0 || sampling.topP > 1) {
      return yield* fail("validate", `topP must be finite and in (0, 1], got ${sampling.topP}`)
    }

    if (!Number.isSafeInteger(sampling.seed) || sampling.seed < 0) {
      return yield* fail("validate", `seed must be a non-negative safe integer, got ${sampling.seed}`)
    }

    return customSampler === undefined
      ? { _tag: "Native" as const, sampling }
      : { _tag: "Custom" as const, sample: customSampler }
  })

const openAutoregressive = (
  program: AutoRegressive.Artifact,
  sampling: Effect.Success<ReturnType<typeof resolveSampling>>
): OpenGeneration<AutoregressiveError, Runtime.Runtime> =>
(request) =>
  Effect.gen(function*() {
    if (sampling._tag === "Native") {
      const generation = yield* Effect.acquireRelease(
        program.generation(),
        (session) => Effect.ignore(session.close()),
        {
          interruptible: true
        }
      )

      const prompt = yield* Tensor.fromTypedArray(request.prompt, [1, request.prompt.length])

      const read = (page: AutoRegressive.TokenPage): GenerationRead<AutoregressiveError, Runtime.Runtime> => ({
        _tag: "Page",
        page: Effect.succeed(page),
        next: () =>
          Effect.gen(function*() {
            const [next] = yield* generation.step([{ seq: page.seq, sampling: sampling.sampling }])

            if (next === undefined) return yield* fail("sample", "generation returned no token page")

            return read(next)
          })
      })

      const [page] = yield* generation.add([{
        prompt,
        sampling: sampling.sampling,
        maxTokens: request.maxTokens,
        eosTokens: request.eosTokens
      }])

      if (page === undefined) return yield* fail("sample", "generation returned no token page")

      return read(page)
    }

    const execution = yield* Effect.acquireRelease(program.execution(), (session) => Effect.ignore(session.close()), {
      interruptible: true
    })

    let currentLogits: Tensor.Concrete | undefined

    const releaseLogits = (logits: Tensor.Concrete) =>
      Tensor.clear(logits).pipe(
        Effect.onExit(() =>
          Effect.sync(() => {
            if (currentLogits === logits) currentLogits = undefined
          })
        )
      )

    yield* Effect.addFinalizer(() => currentLogits === undefined ? Effect.void : releaseLogits(currentLogits))
    const prompt = yield* Tensor.fromTypedArray(request.prompt, [1, request.prompt.length])

    const [entry] = yield* execution.add([prompt]).pipe(Effect.onExit((exit) => {
      if (Exit.isSuccess(exit)) currentLogits = exit.value[0]?.logits

      return Effect.void
    }))

    if (entry === undefined) return yield* fail("sample", "execution returned no sequence")

    const read = (logits: Tensor.Concrete): GenerationRead<AutoregressiveError, Runtime.Runtime> => ({
      _tag: "Page",
      page: Effect.gen(function*() {
        const token = yield* Effect.ensuring(
          Effect.gen(function*() {
            const values = yield* Tensor.toTypedArray(logits)

            if (values.length === 0) return yield* fail("sample", "model produced an empty logits row")

            return yield* Effect.try({
              try: () => sampling.sample(values),
              catch: (error) => fail("sample", error instanceof Error ? error.message : String(error))
            })
          }),
          releaseLogits(logits)
        )

        if (!Number.isSafeInteger(token) || token < 0 || token >= logits.shape[0]) {
          return yield* fail(
            "sample",
            "sampler returned invalid token " + token + " for " + logits.shape[0] + " logits"
          )
        }

        return { tokens: [token] }
      }),
      next: (token) =>
        Effect.gen(function*() {
          const [next] = yield* execution.step([{ seq: entry.seq, token }]).pipe(Effect.onExit((exit) => {
            if (Exit.isSuccess(exit)) currentLogits = exit.value[0]

            return Effect.void
          }))

          if (next === undefined) return yield* fail("sample", "execution returned no logits")

          return read(next)
        })
    })

    return read(entry.logits)
  })

const openCallback = <E, R>(generate: Generation.Generator<E, R>): OpenGeneration<E, R> => (request) =>
  Effect.gen(function*() {
    type Message =
      | {
        readonly _tag: "Page"
        readonly page: Generation.Page
        readonly consumed: Deferred.Deferred<void>
      }
      | {
        readonly _tag: "Done"
        readonly finishReason: ChatResult["finishReason"]
      }
    const queue = yield* Queue.make<Message, E>({ capacity: 1 })
    yield* Effect.addFinalizer(() => Queue.shutdown(queue))

    const read: Effect.Effect<GenerationRead<E, R>, E> = Effect.suspend(() =>
      Effect.map(Queue.take(queue), (message): GenerationRead<E, R> =>
        message._tag === "Done" ? message : {
          _tag: "Page",
          page: Effect.succeed(message.page),
          next: () => Deferred.succeed(message.consumed, undefined).pipe(Effect.andThen(read))
        })
    )

    yield* Effect.suspend(() =>
      generate({
        ...request,
        onPage: (page) =>
          Effect.gen(function*() {
            const consumed = yield* Deferred.make<void>()
            yield* Queue.offer(queue, { _tag: "Page", page, consumed })
            yield* Deferred.await(consumed)
          })
      })
    ).pipe(
      Effect.matchCauseEffect({
        onFailure: (cause) => Queue.failCause(queue, cause),
        onSuccess: (finishReason) => Queue.offer(queue, { _tag: "Done", finishReason })
      }),
      Effect.forkScoped
    )

    return yield* read
  })

const streamGeneration = <GE, GR, E, Value>(
  options: ChatOptions<E, Value>,
  prepareOpen: Effect.Effect<OpenGeneration<GE, GR>, GE, GR>
): Stream.Stream<ChatEvent, ChatError | E | GE, GR> =>
  Stream.unwrap(Effect.gen(function*() {
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

    const open = yield* prepareOpen
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
        bos_token: Option.isSome(bosToken) ? bosToken.value : undefined,
        ...options.variables
      }
    })

    const encoded = yield* tokenizer.encode(rendered, { addSpecialTokens: false })
    const prefillStarted = Date.now()

    const initialRead = yield* open({
      prompt: encoded.data,
      maxTokens: options.maxTokens,
      eosTokens: Array.from(stopTokens)
    })

    const prefillMs = Date.now() - prefillStarted
    const parser = makeParser(tokenizer, resolvedControls, "assistant", options.addGenerationPrompt ?? true)
    const decodeStarted = Date.now()
    let generatedTokens = 0
    let sampledTokens = 0

    const finish = (events: Array<ChatEvent>, finishReason: ChatResult["finishReason"]) => {
      events.push(...parser.finish(finishReason === "stop" ? "turn" : "limit"))
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

      return events
    }

    type State =
      | { readonly _tag: "Prefill" }
      | {
        readonly _tag: "Read"
        readonly read: GenerationRead<GE, GR>
      }
      | {
        readonly _tag: "Next"
        readonly read: Extract<GenerationRead<GE, GR>, { readonly _tag: "Page" }>
        readonly token: number
      }
      | {
        readonly _tag: "Tokens"
        readonly read: Extract<GenerationRead<GE, GR>, { readonly _tag: "Page" }>
        readonly page: Generation.Page
        readonly offset: number
      }
    const initialState: State = { _tag: "Prefill" }

    return Stream.paginate(
      initialState,
      (state): Effect.Effect<readonly [ReadonlyArray<ChatEvent>, Option.Option<State>], ChatError | E | GE, GR> =>
        Effect.gen(function*() {
          if (state._tag === "Prefill") {
            return [
              [{ _tag: "prefill", tokens: encoded.data.length, durationMs: prefillMs }],
              Option.some({ _tag: "Read", read: initialRead })
            ]
          }

          const read = state._tag === "Next" ? yield* state.read.next(state.token) : state.read

          if (read._tag === "Done") return [finish([], read.finishReason), Option.none()]

          const page = state._tag === "Tokens" ? state.page : yield* read.page
          const offset = state._tag === "Tokens" ? state.offset : 0
          const token = page.tokens[offset]

          if (token === undefined) return yield* fail("sample", "generation returned an empty token page")

          const events = yield* parser.accept(token)
          sampledTokens++
          const stopped = stopTokens.has(token)

          if (!stopped) generatedTokens++

          const pageStopped = offset + 1 === page.tokens.length ? page.stopReason : undefined

          if (
            stopped || pageStopped !== undefined ||
            (options.maxTokens !== undefined && sampledTokens >= options.maxTokens)
          ) {
            return [finish(events, stopped || pageStopped === "eos" ? "stop" : "maxTokens"), Option.none()]
          }

          if (offset + 1 < page.tokens.length) {
            return [events, Option.some({ _tag: "Tokens", read, page, offset: offset + 1 })]
          }

          return [events, Option.some({ _tag: "Next", read, token })]
        })
    )
  }))

/**
 * Renders the supplied history once, encodes and prefills it, then samples and
 * parses one token at a time. Template rendering receives
 * `addGenerationPrompt` (default `true`) and merged variables. Encoding always
 * uses `addSpecialTokens: false`; templates are therefore responsible for all
 * model-required BOS/EOS/control text.
 *
 * Standard sampling opens a {@link AutoRegressive.Generation} session and returns token
 * ids without output logits. A custom `sampling` callback opens a
 * {@link AutoRegressive.StatefulExecution} session and reads each complete logits row to
 * a host typed array. Chat clears that row if sampling, readback, or the
 * callback fails or is interrupted. Chat parses a valid non-stop token before
 * committing it with a session step. It parses the final stop or limit token
 * but does not step it because no successor is needed. Stop ids delimit the
 * protocol. They do not filter output. A custom stop id that decodes as text
 * can emit a final delta before termination.
 *
 * The selected session is closed on normal completion, tokenizer/parser/model
 * failure, interruption, or downstream cancellation. Cleanup errors are
 * ignored so they do not replace the primary exit. With a custom sampler, each
 * logits row is cleared after its token is selected. The stream retains only
 * the current unread row and releases it on downstream cancellation. Normal
 * termination emits `done`; other exits do not synthesize terminal events.
 *
 * Validation covers only a few conditions. The template and messages must be
 * nonempty, and `maxTokens` must be a positive safe integer. Parser controls,
 * default token-derived stops, and `bosTokenId` must resolve when used. Sampler
 * output must index the logits row.
 * Chat does not validate message schemas, template syntax, prompt non-emptiness
 * after encoding, stop-id ranges, control-id distinctness, tokenizer/program
 * vocabulary agreement, model vocabulary semantics, or decode prefix stability.
 *
 * @since 0.1.0
 * @category constructors
 */
export const stream = <E = never, Value = unknown>(
  options: ChatStreamOptions<E, Value>
): Stream.Stream<ChatEvent, E | AutoregressiveError, Runtime.Runtime> =>
  streamGeneration(
    options,
    Effect.map(resolveSampling(options.sampling), (sampling) => openAutoregressive(options.program, sampling))
  )

/**
 * Renders and tokenizes chat history once, then consumes committed pages from
 * the supplied generation operation using the same parser and events as stream.
 * Sampling and refinement policy belong to generate. Empty refinement steps
 * stay inside that operation; publishing an empty page fails with ChatError.
 *
 * Page tokens are parsed in order and their events are emitted before Chat
 * requests the next page. Chat stops at the first stop ID or token limit, even
 * within a page, and interrupts the generator. Awaiting onPage provides
 * backpressure. Generation completion supplies the stopping reason
 * when no token or page policy has already ended the response. Failure and
 * downstream cancellation emit no synthetic terminal events.
 *
 * The prefill event measures time until the first committed page is available.
 * For block generators this includes refinement of the first block.
 *
 * @since 0.1.0
 * @category constructors
 */
export const streamWith = <GE, GR, E = never, Value = unknown>(
  generate: Generation.Generator<GE, GR>,
  options: ChatOptions<E, Value>
): Stream.Stream<ChatEvent, ChatError | E | GE, GR> => streamGeneration(options, Effect.succeed(openCallback(generate)))

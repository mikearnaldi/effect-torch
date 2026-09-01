/**
 * Native-backed text tokenization using the Hugging Face `tokenizers` crate.
 * The TypeScript facade handles its own padding and truncation, maps errors,
 * sequences progress reports, and allocates `TokenIds`. A package-local addon
 * owns the immutable native tokenizer pipeline and its CPU state. The caller
 * owns each returned, writable id buffer. The native tokenizer does not retain
 * those buffers.
 *
 * Loading accepts the `tokenizer.json` components that the bundled crate
 * supports. Unsupported or custom components cause loading to fail. The loaded
 * pipeline keeps its serialized padding, truncation, postprocessing, and
 * decoding. Facade padding and truncation apply on top and do not change the
 * native tokenizer.
 *
 * `specialTokens: "Always"` leaves registered special-token matching enabled.
 * `"Never"` splits exact registered multi-character contents into separately
 * encoded pieces. Normalized aliases and one-character specials can still map
 * to special ids. Postprocessors can also insert special ids, and token merges
 * cannot cross split boundaries. `"Never"` is not a security guarantee for
 * untrusted text.
 *
 * Importing this module immediately loads a package-local native addon on macOS
 * or Linux `arm64`/`x64`. Unsupported hosts and loading errors throw during
 * module evaluation, before the code creates an Effect.
 *
 * @since 0.1.0
 * @module
 */
import { Data, Effect, Option, Queue, Stream } from "effect"
import { type Pipeable, pipeArguments } from "effect/Pipeable"
import native from "./internal/native.js"

// The `.node` addon exports this constructor at runtime. The class in
// `internal/native-addon.d.ts` is generated from the Rust N-API exports.
const { NativeTokenizer } = native
type NativeTokenizerType = InstanceType<typeof NativeTokenizer>

/**
 * Runtime marker installed on tokenizer values.
 *
 * @since 0.1.0
 * @category symbols
 */
export const TokenizerTypeId: unique symbol = Symbol.for(
  "@effect-torch/tokenizers/Tokenizer"
)

/**
 * Type of {@link TokenizerTypeId}.
 *
 * @since 0.1.0
 * @category symbols
 */
export type TokenizerTypeId = typeof TokenizerTypeId

/**
 * Failure from a tokenizer constructor or member that returns an Effect. It
 * contains `_tag: "TokenizerError"`, an operation label, and a diagnostic
 * message. Causes include native errors, file I/O, invalid serialized
 * tokenizers, chat-context serialization, and invalid batch shapes or padding.
 * Vocabulary lookups return {@link Option.Option} instead. File training treats
 * line-read and UTF-8 errors as described by {@link TrainSource}.
 *
 * @since 0.1.0
 * @category errors
 */
export class TokenizerError extends Data.TaggedError("TokenizerError")<{
  /** The facade operation that failed, such as `encode`, `train`, or `save`. */
  readonly op: string
  /** Human-readable diagnostic from the wrapper or native implementation. */
  readonly message: string
}> {}

/**
 * Batch padding applied by {@link Tokenizer.encodeBatch}. `None` requires equal
 * row lengths after facade truncation. `Longest` pads to the longest row.
 * `MaxLength` produces exactly `maxLength` columns and fails if a row remains
 * longer. Padding never truncates and is separate from serialized native
 * padding.
 *
 * `maxLength` must be a finite non-negative integer. `padId` must be an
 * unsigned 32-bit integer, but need not belong to the vocabulary. Constructors
 * and encoding do not fully validate these constraints. Typed-array operations
 * may coerce invalid values, produce invalid shape metadata, or fail the
 * encoding Effect.
 *
 * @since 0.1.0
 * @category models
 */
export type Padding =
  | {
    /** Selects no facade padding. */
    readonly _tag: "None"
  }
  | {
    /** Selects padding to the longest row in the batch. */
    readonly _tag: "Longest"
    /** Token id written into padded positions. */
    readonly padId: number
  }
  | {
    /** Selects padding to an explicit row length. */
    readonly _tag: "MaxLength"
    /** Exact number of columns in the padded batch. */
    readonly maxLength: number
    /** Token id written into padded positions. */
    readonly padId: number
  }

/**
 * Disables batch padding.
 *
 * @since 0.1.0
 * @category constructors
 */
export const paddingNone: Padding = { _tag: "None" }

/**
 * Pads batch rows to the longest post-truncation row with `padId`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const paddingLongest = (padId: number): Padding => ({
  _tag: "Longest",
  padId
})

/**
 * Pads batch rows to exactly `maxLength` with `padId`. Rows longer than
 * `maxLength` are not truncated.
 *
 * @since 0.1.0
 * @category constructors
 */
export const paddingMaxLength = (
  maxLength: number,
  padId: number
): Padding => ({
  _tag: "MaxLength",
  maxLength,
  padId
})

/**
 * Truncation applied after native encoding and before facade batch padding.
 * `maxLength` must be a finite non-negative integer. Construction and encoding
 * do not fully validate it. Typed-array slicing may coerce unsupported values.
 * With `MaxLength` padding, each truncated row must fit the padding length or
 * {@link Tokenizer.encodeBatch} fails.
 *
 * @since 0.1.0
 * @category models
 */
export type Truncation =
  | {
    /** Selects no facade truncation. */
    readonly _tag: "None"
  }
  | {
    /** Selects truncation to an explicit sequence length. */
    readonly _tag: "MaxLength"
    /** Maximum number of token ids retained from a sequence. */
    readonly maxLength: number
  }

/**
 * Disables truncation.
 *
 * @since 0.1.0
 * @category constructors
 */
export const truncationNone: Truncation = { _tag: "None" }

/**
 * Keeps at most the first `maxLength` ids from each encoded sequence.
 *
 * @since 0.1.0
 * @category constructors
 */
export const truncationMaxLength = (maxLength: number): Truncation => ({
  _tag: "MaxLength",
  maxLength
})

/**
 * Policy for registered special tokens found in input text. `"Always"` uses
 * the loaded tokenizer's added-token matching, including each token's
 * normalization and boundary options.
 *
 * `"Never"` scans the raw input for exact registered special-token contents,
 * longest first. It encodes surrounding text and matches as separate pieces,
 * subdividing exact multi-character matches to avoid whole-token lookup. These
 * new merge and normalization boundaries can produce different ids than one
 * uninterrupted encode. `"Never"` cannot suppress normalized aliases that
 * differ from the raw content or one-character specials, which cannot be
 * subdivided. Postprocessors may also insert special ids. Output can still
 * contain special-token ids.
 *
 * @since 0.1.0
 * @category models
 */
export type SpecialTokenPolicy = "Never" | "Always"

/**
 * Required facade configuration. Native construction copies `specialTokens`.
 * The returned tokenizer retains this object and reads its padding and
 * truncation policies while processing each encoding result. Nothing clones or
 * freezes the object or its nested policies. The caller owns them and must not
 * mutate them while using the tokenizer.
 *
 * {@link fromFile} and {@link fromJson} preserve native padding and truncation
 * from `tokenizer.json`. This config does not clear those settings. Native code
 * encodes each facade batch row as an independent sequence, so native
 * `BatchLongest` does not span the facade batch. Under `"Never"`, native code
 * also encodes each nonempty segment independently. Serialized native padding
 * or truncation can occur within the merged row. Facade truncation and batch
 * padding run after native encoding.
 *
 * @since 0.1.0
 * @category models
 */
export interface TokenizerConfig {
  /** Facade batch padding. Used only by `encodeBatch`. */
  readonly padding: Padding
  /** Facade per-sequence truncation applied after native encoding. */
  readonly truncation: Truncation
  /** Input special-token policy. Native construction fixes its value. */
  readonly specialTokens: SpecialTokenPolicy
}

/**
 * Token ids in a new, writable host buffer owned by the caller. The native
 * tokenizer neither retains the buffer nor observes mutations to it. This
 * module uses shape `[T]` for one or concatenated sequences and row-major
 * `[B, T]` for facade batches. The object, buffer, and shape are not frozen.
 *
 * @since 0.1.0
 * @category models
 */
export interface TokenIds {
  /** Caller-owned row-major token ids. */
  readonly data: Uint32Array
  /** Logical dimensions whose product equals `data.length` for module output. */
  readonly shape: ReadonlyArray<number>
  /** Element type metadata. Always unsigned 32-bit integers. */
  readonly dtype: "u32"
}

/**
 * Input accepted by tokenizer decode operations. Array elements must be
 * unsigned 32-bit ids. The facade copies input into a number array before
 * calling native code. For {@link TokenIds}, decoding reads only `data` and
 * neither validates nor interprets `shape` or `dtype`. Node-API may coerce
 * JavaScript numbers outside the supported range.
 *
 * @since 0.1.0
 * @category models
 */
export type TokenIdInput = TokenIds | Uint32Array | ReadonlyArray<number>

/**
 * Options for {@link Tokenizer.encode}. `addSpecialTokens` defaults to `true`.
 * The facade passes it to the native postprocessor. It does not enable or
 * disable input added-token matching. {@link SpecialTokenPolicy} controls that
 * matching. Native truncation can use this flag to reserve room for ids added
 * by the postprocessor.
 *
 * Under `"Never"`, native code encodes each segment with this flag disabled.
 * If requested, it applies the postprocessor once to the merged encoding. No
 * native truncation or padding follows that final application. Batch methods
 * always request special-token addition and offer no equivalent option.
 *
 * @since 0.1.0
 * @category models
 */
export interface EncodeOptions {
  /** Whether the native postprocessor may add its configured special tokens. */
  readonly addSpecialTokens?: boolean | undefined
}

/**
 * Decode options. `skipSpecialTokens` defaults to `false`. When true, it omits
 * ids whose token string is registered as a special added token.
 * It omits a facade `padId` only if that token is registered as special.
 *
 * @since 0.1.0
 * @category models
 */
export interface DecodeOptions {
  /** Whether decoding omits registered special-token ids. */
  readonly skipSpecialTokens?: boolean | undefined
}

/**
 * Stateful decoder for autoregressive token ids. A step can return no text
 * while the tokenizer waits for a complete byte sequence.
 *
 * @since 0.1.0
 * @category models
 */
export interface DecodeStream {
  /** Adds one token and returns the next stable text chunk, if available. */
  readonly step: (id: number) => Effect.Effect<string | undefined, TokenizerError>
}

/**
 * Shared configuration with no facade padding or truncation and the `"Never"`
 * special-token policy. It does not disable native padding or truncation from
 * a loaded tokenizer. {@link SpecialTokenPolicy} documents the limits of
 * `"Never"`. This object and its nested singleton policies are mutable objects
 * that callers must treat as read-only.
 *
 * @since 0.1.0
 * @category constructors
 */
export const strictConfig: TokenizerConfig = {
  padding: paddingNone,
  truncation: truncationNone,
  specialTokens: "Never"
}

/**
 * Model family to train. BPE uses byte-level preprocessing. WordPiece uses a
 * BERT normalizer and `##` continuations. Unigram uses the SentencePiece
 * metaspace convention and byte fallback. WordLevel uses whitespace tokens.
 * WordPiece adds `[UNK]`, and Unigram adds `<unk>` at id `0`. BPE and WordLevel
 * do not add an unknown token.
 *
 * @since 0.1.0
 * @category models
 */
export type TrainModel = "BPE" | "WordPiece" | "Unigram" | "WordLevel"

/**
 * Corpus source for {@link train}. `Files` streams UTF-8 text in path order,
 * one line per sequence. It removes `\n` and an optional preceding `\r`, so
 * line endings are not trained as characters. Native code stats and opens every
 * path before reading the corpus. A stat or open error fails the Effect. After
 * opening, a line-read or UTF-8 error is silently treated as EOF for that path,
 * and training continues with an incomplete corpus. `Texts` supplies each
 * string as one sequence.
 *
 * Source constructors retain the input arrays. They do not copy them. When the
 * lazy training Effect starts, it copies their strings into native vectors. Do
 * not mutate the arrays before then.
 *
 * @since 0.1.0
 * @category models
 */
export type TrainSource =
  | {
    /** Selects a corpus streamed from files. */
    readonly _tag: "Files"
    /** UTF-8 corpus files consumed in this order. */
    readonly paths: ReadonlyArray<string>
  }
  | {
    /** Selects an in-memory corpus. */
    readonly _tag: "Texts"
    /** Strings supplied as independent training sequences. */
    readonly texts: ReadonlyArray<string>
  }

/**
 * Creates a file source without copying `paths`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const trainFiles = (paths: ReadonlyArray<string>): TrainSource => ({
  _tag: "Files",
  paths
})

/**
 * Creates an in-memory source without copying `texts`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const trainTexts = (texts: ReadonlyArray<string>): TrainSource => ({
  _tag: "Texts",
  texts
})

/**
 * Progress for the corpus-feed phase of training, measured in UTF-8 bytes.
 * `Texts` totals are the sum of the strings' UTF-8 lengths. `Files` totals use
 * raw file sizes, while processed counts exclude stripped line endings. An
 * intermediate processed count can trail the total. A nonempty feed ends with
 * `(total, total)`, even if a swallowed read or UTF-8 error truncated
 * a file. This final report neither proves that every byte was read nor marks
 * completion of model computation. Model computation has no progress reports.
 *
 * Native code checks progress once per sequence and reports after at least
 * `everyBytes` more processed bytes. A large sequence produces one report, not
 * catch-up reports. It sends the final report even when the remaining count is
 * below the interval. `everyBytes` must be finite. The facade floors it and
 * replaces negative values with zero. Zero disables every report, including
 * completion, and an empty corpus sends no completion report. Native code posts
 * callbacks to JavaScript without blocking. The facade runs received report
 * Effects in order. A report failure or interruption stops the facade from
 * waiting for more reports or the result. Native training continues, and the
 * facade cannot cancel it.
 *
 * @since 0.1.0
 * @category models
 */
export type TrainProgress<E, R> =
  | {
    /** Selects no progress reporting. */
    readonly _tag: "None"
  }
  | {
    /** Selects Effect-based progress reporting. */
    readonly _tag: "Report"
    /**
     * Approximate minimum byte interval. Must be finite. Zero disables
     * reporting.
     */
    readonly everyBytes: number
    /** Handles one `(processed, total)` feed-progress event. */
    readonly report: (processed: number, total: number) => Effect.Effect<void, E, R>
  }

/**
 * Disables training progress reports.
 *
 * @since 0.1.0
 * @category constructors
 */
export const trainProgressNone: TrainProgress<never, never> = { _tag: "None" }

/**
 * Reports feed progress at sequence boundaries. `everyBytes` is an approximate
 * minimum interval.
 *
 * @since 0.1.0
 * @category constructors
 */
export const trainProgressReport = <E, R>(
  everyBytes: number,
  report: (processed: number, total: number) => Effect.Effect<void, E, R>
): TrainProgress<E, R> => ({ _tag: "Report", everyBytes, report })

/**
 * Configuration for {@link train}. The lazy Effect reads the object and its
 * arrays when it starts, then copies their values into native memory. The
 * resulting tokenizer does not retain them. `vocabSize` is a target, not an
 * exact result. Corpus size and frequency filtering can produce fewer ids.
 * Required alphabets and special tokens set a minimum, and Unigram appends
 * missing byte-fallback pieces after training.
 *
 * Native code represents the numeric fields as unsigned 32-bit integers, but
 * the facade does not validate them. Use a positive integer `vocabSize` and a
 * non-negative integer `minFrequency` within that range. Node-API may coerce
 * other numbers before training.
 *
 * @since 0.1.0
 * @category models
 */
export interface TrainConfig<E, R> {
  /** Corpus source read when the lazy training Effect starts. */
  readonly source: TrainSource
  /** Model family and built-in preprocessing pipeline. */
  readonly model: TrainModel
  /** Positive unsigned 32-bit target size, including trainer special tokens. */
  readonly vocabSize: number
  /** Unsigned 32-bit frequency cutoff. The Unigram trainer ignores it. */
  readonly minFrequency: number
  /** Registered special tokens, ordered before learned tokens by the trainer. */
  readonly specialTokens: ReadonlyArray<string>
  /** Corpus-feed progress policy. */
  readonly progress: TrainProgress<E, R>
}

/**
 * Value accepted by the native chat-template JSON context. Objects and arrays
 * are recursive; `undefined` fields are omitted by `JSON.stringify`. A
 * function is useful only as an object's `toJSON` field.
 *
 * @since 0.1.0
 * @category models
 */
export type ChatTemplateValue =
  | string
  | number
  | boolean
  | null
  | undefined
  | ReadonlyArray<ChatTemplateValue>
  | ChatTemplateObject
  | ChatTemplateSerializable
  | (() => ChatTemplateValue)

/** Recursive object accepted by the native chat-template JSON context. */
export interface ChatTemplateObject {
  readonly [field: string]: ChatTemplateValue
}

/** Object that controls its chat-template JSON representation. */
export interface ChatTemplateSerializable {
  readonly toJSON: () => ChatTemplateValue
}

/**
 * One message for a caller-provided Jinja chat template. The template receives
 * `role`, `content`, and extra fields as JSON values. `JSON.stringify` must be
 * able to serialize them. It omits `undefined` object fields, and unsupported
 * values such as `bigint` cause the rendering Effect to fail.
 *
 * @since 0.1.0
 * @category models
 */
export interface ChatMessage {
  /** Conversation role exposed to the template. */
  readonly role: string
  /** Optional message payload exposed to the template. */
  readonly content?: ChatTemplateValue
  /** Additional caller-defined fields exposed to the template. */
  readonly [field: string]: ChatTemplateValue
}

/**
 * Context options for a caller-provided Jinja chat template. `variables` adds
 * top-level, JSON-serializable fields such as `bos_token`, `eos_token`, tools,
 * dates, or cutoff metadata. The tokenizer does not supply token strings or
 * other model metadata. After spreading `variables`, the wrapper overwrites
 * `messages` and `add_generation_prompt`, so variables with those names cannot
 * replace the wrapper values. A custom `toJSON` property can transform the
 * whole context. Supply one only if you want it to transform the context.
 *
 * @since 0.1.0
 * @category models
 */
export interface ChatTemplateOptions {
  /**
   * Value exposed as `add_generation_prompt`. The template decides how to use
   * it.
   */
  readonly addGenerationPrompt?: boolean | undefined
  /** Additional variables exposed to the Jinja rendering context. */
  readonly variables?: Readonly<Record<string, ChatTemplateValue>> | undefined
}

/**
 * A native-backed text tokenizer. The TypeScript facade is a plain, unfrozen
 * object. Its methods share one immutable native handle and retain the supplied
 * {@link TokenizerConfig}. Native state supports concurrent batch work. Callers
 * must not mutate the facade or its retained configuration.
 *
 * The native handle owns vocabulary and pipeline data in CPU memory. It keeps
 * no file handle, device buffer, or worker open. Background work holds its own
 * native reference. Node-API reclaims the handle after the facade and in-flight
 * work become unreachable. The facade has no explicit disposal operation.
 *
 * @since 0.1.0
 * @category models
 */
export interface Tokenizer extends Pipeable {
  /**
   * Runtime marker used by {@link isTokenizer}. It does not authenticate the
   * value.
   */
  readonly [TokenizerTypeId]: TokenizerTypeId
  /**
   * Vocabulary size, including added and special tokens. The facade copies this
   * value when it creates the tokenizer.
   */
  readonly vocabSize: number
  /**
   * Encodes text into caller-owned `[T]` `u32` ids. Serialized native policy runs
   * first, then facade truncation. Facade batch padding does not apply. Under
   * `"Never"`, serialized native padding and truncation can run on each segment
   * before native code merges their ids. {@link EncodeOptions} describes the
   * postprocessor path. Native encoding runs synchronously on the JavaScript
   * thread. Interrupting the Effect cannot stop the call after it starts.
   */
  readonly encode: (
    text: string,
    options?: EncodeOptions
  ) => Effect.Effect<TokenIds, TokenizerError>
  /**
   * Encodes a nonempty batch in parallel into caller-owned, row-major `[B, T]`
   * `u32` ids in input order. Each text is a separate native sequence and always
   * requests postprocessor special-token addition. Native `BatchLongest` does
   * not span this facade batch. Under `"Never"`, native policy can run on each
   * segment. Facade truncation precedes facade padding. With `None` padding,
   * unequal row lengths fail.
   *
   * A failed row fails the whole Effect, with no partial result. Native work runs
   * off the JavaScript thread. Interrupting the Effect does not cancel work that
   * has started.
   */
  readonly encodeBatch: (
    texts: ReadonlyArray<string>
  ) => Effect.Effect<TokenIds, TokenizerError>
  /**
   * Encodes a batch in parallel and always requests postprocessor special-token
   * addition. It concatenates the facade-truncated rows into caller-owned
   * `[sum(T)]` `u32` ids in input order. Facade padding does not apply. Serialized
   * native policy may already have run on each sequence or `"Never"` segment.
   * The result has no row boundaries. An empty batch returns an empty buffer
   * with shape `[0]`. Native work runs off the JavaScript thread. Interruption
   * does not cancel work that has started.
   */
  readonly encodeBatchConcat: (
    texts: ReadonlyArray<string>
  ) => Effect.Effect<TokenIds, TokenizerError>
  /**
   * Decodes all supplied ids as one flat sequence and ignores a {@link TokenIds}
   * shape. Registered special tokens remain unless `skipSpecialTokens` is true.
   * The tokenizer silently omits ids outside the vocabulary. The loaded decoder
   * controls spacing, byte fallback, and other reconstruction. Without one, the
   * tokenizer joins token strings with spaces. Decoding need not reverse
   * encoding, truncation, padding, or normalization. The native call runs
   * synchronously on the JavaScript thread. Interrupting the Effect cannot stop
   * the call after it starts.
   */
  readonly decode: (
    ids: TokenIdInput,
    options?: DecodeOptions
  ) => Effect.Effect<string, TokenizerError>
  /**
   * Creates an independent stateful decoder for autoregressive token ids.
   *
   * @since 0.1.0
   * @category models
   */
  readonly decodeStream: (options?: DecodeOptions) => DecodeStream
  /**
   * Decodes each outer input as an independent sequence in outer order. It
   * ignores inner {@link TokenIds} shapes and does not split a `[B, T]` value
   * into rows. An empty outer array returns an empty array. JavaScript blocks
   * until the native batch decode finishes, although native code may use
   * parallel workers. Interrupting the Effect cannot stop the call after it
   * starts.
   */
  readonly decodeBatch: (
    ids: ReadonlyArray<TokenIdInput>,
    options?: DecodeOptions
  ) => Effect.Effect<ReadonlyArray<string>, TokenizerError>
  /**
   * Serializes `messages`, `add_generation_prompt`, and caller variables, then
   * renders the supplied template to text. It does not read a template from
   * `tokenizer.json` or tokenize the result. The lazy Effect serializes the
   * context when it runs.
   *
   * The native renderer is MiniJinja. It provides its compiled-in tests,
   * filters, and `tojson`, plus `raise_exception(message)` and
   * `strftime_now(format)`. `strftime_now` uses UTC and supports only `%Y`, `%m`,
   * `%d`, `%H`, `%M`, `%S`, and `%%`. Missing values use MiniJinja's lenient
   * undefined behavior. The renderer strips one final template newline. It has
   * no template loader or Transformers method-compatibility callback, so it
   * cannot handle includes, imports, or calls such as `mapping.items()`. Do not
   * assume full compatibility with Python Jinja or Transformers extensions.
   * Rendering runs synchronously on the JavaScript thread and cannot be
   * interrupted after it starts.
   */
  readonly applyChatTemplate: (
    template: string,
    messages: ReadonlyArray<ChatMessage>,
    options?: ChatTemplateOptions
  ) => Effect.Effect<string, TokenizerError>
  /** Synchronously returns the id for an exact vocabulary or added token. */
  readonly tokenToId: (token: string) => Option.Option<number>
  /**
   * Synchronously returns the token for an unsigned 32-bit id, or `None` when
   * unknown. Node-API may coerce other, unsupported numbers.
   */
  readonly idToToken: (id: number) => Option.Option<string>
  /**
   * Saves the native tokenizer as `tokenizer.json`. The file includes native
   * settings but not the facade {@link TokenizerConfig}, which callers must
   * supply again when loading. The native code creates or truncates the target
   * directly instead of writing atomically. Failure can leave a partial file.
   * File I/O and serialization block the JavaScript thread and cannot be
   * interrupted after they start.
   */
  readonly save: (path: string) => Effect.Effect<void, TokenizerError>
}

const toTokenizerError = (op: string) => (cause: unknown) =>
  new TokenizerError({
    op,
    message: cause instanceof Error ? cause.message : String(cause)
  })

const idsOf = (
  ids: TokenIdInput
): Effect.Effect<Array<number>, TokenizerError> =>
  "data" in ids
    ? Effect.succeed(Array.from(ids.data))
    : Effect.succeed(Array.from(ids))

const tokenIds = (data: Uint32Array, shape: ReadonlyArray<number>): TokenIds => ({ data, shape, dtype: "u32" })

const truncate = (data: Uint32Array, config: TokenizerConfig): Uint32Array =>
  config.truncation._tag === "MaxLength" && data.length > config.truncation.maxLength
    ? data.slice(0, config.truncation.maxLength)
    : data

const makeBatch = (rows: ReadonlyArray<Uint32Array>, config: TokenizerConfig): TokenIds => {
  if (rows.length === 0) {
    throw new Error("encodeBatch: expected at least one text")
  }
  const truncated = rows.map((row) => truncate(row, config))
  let columns: number
  let padId = 0
  switch (config.padding._tag) {
    case "None": {
      columns = truncated[0]!.length
      if (truncated.some((row) => row.length !== columns)) {
        throw new Error("encodeBatch: ragged encodings require an explicit padding policy")
      }
      break
    }
    case "Longest": {
      columns = Math.max(...truncated.map((row) => row.length))
      padId = config.padding.padId
      break
    }
    case "MaxLength": {
      columns = config.padding.maxLength
      padId = config.padding.padId
      if (truncated.some((row) => row.length > columns)) {
        throw new Error("encodeBatch: an encoding exceeds maxLength; configure truncation explicitly")
      }
      break
    }
  }
  const data = new Uint32Array(rows.length * columns)
  if (padId !== 0) data.fill(padId)
  for (let row = 0; row < truncated.length; row++) {
    data.set(truncated[row]!, row * columns)
  }
  return tokenIds(data, [rows.length, columns])
}

const TokenizerProto = {
  pipe() {
    return pipeArguments(this, arguments)
  }
}

const make = (
  handle: NativeTokenizerType,
  config: TokenizerConfig
): Tokenizer => {
  const self = Object.create(TokenizerProto)
  self[TokenizerTypeId] = TokenizerTypeId
  self.vocabSize = handle.vocabSize
  self.encode = (text: string, options?: EncodeOptions) =>
    Effect.try({
      try: () => {
        const data = truncate(handle.encode(text, options?.addSpecialTokens ?? true), config)
        return tokenIds(data, [data.length])
      },
      catch: toTokenizerError("encode")
    })
  self.encodeBatch = (texts: ReadonlyArray<string>) =>
    Effect.tryPromise({
      try: async () => makeBatch(await handle.encodeBatch([...texts]), config),
      catch: toTokenizerError("encodeBatch")
    })
  self.encodeBatchConcat = (texts: ReadonlyArray<string>) =>
    Effect.tryPromise({
      try: async () => {
        const rows = (await handle.encodeBatch([...texts])).map((row) => truncate(row, config))
        const length = rows.reduce((total, row) => total + row.length, 0)
        const data = new Uint32Array(length)
        let offset = 0
        for (const row of rows) {
          data.set(row, offset)
          offset += row.length
        }
        return tokenIds(data, [length])
      },
      catch: toTokenizerError("encodeBatchConcat")
    })
  self.decode = (ids: TokenIdInput, options?: DecodeOptions) =>
    Effect.flatMap(idsOf(ids), (resolved) =>
      Effect.try({
        try: () => handle.decode(resolved, options?.skipSpecialTokens ?? false),
        catch: toTokenizerError("decode")
      }))
  self.decodeStream = (options?: DecodeOptions): DecodeStream => {
    const stream = handle.decodeStream(options?.skipSpecialTokens ?? false)
    return {
      step: (id) =>
        Effect.try({
          try: () => stream.step(id) ?? undefined,
          catch: toTokenizerError("decodeStream")
        })
    }
  }
  self.decodeBatch = (
    batch: ReadonlyArray<TokenIdInput>,
    options?: DecodeOptions
  ) =>
    Effect.flatMap(
      Effect.forEach(batch, idsOf, { concurrency: "unbounded" }),
      (resolved) =>
        Effect.try({
          try: () => handle.decodeBatch(resolved, options?.skipSpecialTokens ?? false),
          catch: toTokenizerError("decodeBatch")
        })
    )
  self.applyChatTemplate = (
    template: string,
    messages: ReadonlyArray<ChatMessage>,
    options?: ChatTemplateOptions
  ) =>
    Effect.try({
      try: () =>
        handle.applyChatTemplate(
          template,
          JSON.stringify({
            ...options?.variables,
            messages,
            add_generation_prompt: options?.addGenerationPrompt ?? false
          })
        ),
      catch: toTokenizerError("applyChatTemplate")
    })
  self.tokenToId = (token: string) => Option.fromNullishOr(handle.tokenToId(token))
  self.idToToken = (id: number) => Option.fromNullishOr(handle.idToToken(id))
  self.save = (path: string) =>
    Effect.try({
      try: () => handle.save(path),
      catch: toTokenizerError("save")
    })
  return self
}

/**
 * Loads a tokenizer from a supported `tokenizer.json` file. File I/O and
 * parsing run synchronously on the JavaScript thread when the Effect runs.
 * Interrupting the Effect cannot stop them after they start. The returned
 * tokenizer retains the supplied config by reference. Its policies apply on
 * top of native padding and truncation from the document.
 *
 * @since 0.1.0
 * @category constructors
 */
export const fromFile = (
  path: string,
  config: TokenizerConfig
): Effect.Effect<Tokenizer, TokenizerError> =>
  Effect.try({
    try: () =>
      make(
        NativeTokenizer.fromFile(path, config.specialTokens === "Always"),
        config
      ),
    catch: toTokenizerError("fromFile")
  })

/**
 * Loads a tokenizer from a supported in-memory `tokenizer.json` document.
 * Parsing runs synchronously on the JavaScript thread when the Effect runs.
 * Interrupting the Effect cannot stop parsing after it starts. The returned
 * tokenizer retains the supplied config by reference. Its policies apply on
 * top of native padding and truncation from the document.
 *
 * @since 0.1.0
 * @category constructors
 */
export const fromJson = (
  json: string,
  config: TokenizerConfig
): Effect.Effect<Tokenizer, TokenizerError> =>
  Effect.try({
    try: () =>
      make(
        NativeTokenizer.fromJson(json, config.specialTokens === "Always"),
        config
      ),
    catch: toTokenizerError("fromJson")
  })

/**
 * Trains a tokenizer from a {@link TrainSource}. The job starts only when the
 * returned Effect runs. Native file reading, corpus feeding, and model training
 * run on a blocking worker, not the JavaScript thread. Interrupting the Effect
 * or failing a report Effect stops the facade from waiting for the result. It
 * does not cancel native work that has started. {@link TrainProgress} documents
 * the reporting limits. The result retains the tokenizer config by reference.
 * Training copies the training config into native-owned values.
 *
 * @since 0.1.0
 * @category constructors
 */
export const train = <E = never, R = never>(
  trainConfig: TrainConfig<E, R>,
  config: TokenizerConfig
): Effect.Effect<Tokenizer, TokenizerError | E, R> => {
  return Stream.callback<
    Effect.Effect<undefined | NativeTokenizerType, E | TokenizerError, R>
  >((queue) =>
    Effect.gen(function*() {
      const progress = trainConfig.progress
      const onProgress = progress._tag === "Report"
        ? (event: [number, number]) => {
          Queue.offerUnsafe(
            queue,
            Effect.as(undefined)(progress.report(event[0], event[1]))
          )
        }
        : () => {}
      const progressEveryBytes = progress._tag === "Report" ? Math.max(0, Math.floor(progress.everyBytes)) : 0
      NativeTokenizer.train(
        {
          model: trainConfig.model,
          vocabSize: trainConfig.vocabSize,
          minFrequency: trainConfig.minFrequency,
          specialTokens: [...trainConfig.specialTokens],
          source: trainConfig.source._tag === "Files"
            ? {
              tag: "Files",
              paths: [...trainConfig.source.paths]
            }
            : {
              tag: "Texts",
              texts: [...trainConfig.source.texts]
            }
        },
        config.specialTokens === "Always",
        onProgress,
        progressEveryBytes
      )
        .then((tensor) => {
          Queue.offerUnsafe(queue, Effect.succeed(tensor))
          Queue.endUnsafe(queue)
        })
        .catch((e) => {
          Queue.offerUnsafe(queue, Effect.fail(toTokenizerError("train")(e)))
          Queue.endUnsafe(queue)
        })
    })
  ).pipe(
    Stream.mapEffect((_) => _),
    Stream.filter((_) => _ !== undefined),
    Stream.runLast,
    Effect.flatMap((_) =>
      Effect.try({
        try: () => make(Option.getOrThrow(_), config),
        catch: toTokenizerError("train")
      })
    )
  )
}

/**
 * Checks only for {@link TokenizerTypeId}, including through the prototype
 * chain. It does not check the marker value or tokenizer members. Marked or
 * spoofed objects also pass.
 *
 * @since 0.1.0
 * @category guards
 */
export const isTokenizer = (value: unknown): value is Tokenizer =>
  typeof value === "object" && value !== null && TokenizerTypeId in value

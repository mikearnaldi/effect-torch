/**
 * Native-backed text tokenization using the Hugging Face `tokenizers` crate.
 * This module is the Effect-based TypeScript facade: it owns facade padding,
 * truncation, error mapping, progress sequencing, and `TokenIds` allocation.
 * A package-local addon owns the immutable native tokenizer pipeline and its
 * CPU-heap state. Returned id buffers are writable by the caller and are not
 * retained by the native tokenizer.
 *
 * Loading accepts `tokenizer.json` components supported by the bundled crate;
 * documents using unsupported or custom components fail. Serialized native
 * padding, truncation, postprocessing, and decoding remain part of the loaded
 * pipeline. Facade padding and truncation are additional policies and are not
 * written back to the native tokenizer.
 *
 * `specialTokens: "Always"` leaves registered special-token matching enabled.
 * `"Never"` is a best-effort raw-content policy: it splits exact registered
 * multi-character contents into separately encoded pieces. Normalized aliases
 * and one-character specials can still resolve to special ids, postprocessors
 * can insert special ids, and ordinary merges cannot cross split boundaries.
 * Do not treat `"Never"` as a security guarantee for untrusted text.
 *
 * Importing this module eagerly selects and loads a package-local native addon
 * for supported macOS or Linux `arm64`/`x64` hosts. Unsupported hosts and addon
 * loading failures throw during module evaluation, before an Effect is created.
 *
 * @since 0.1.0
 * @module
 */
import { Data, Effect, Option, Queue, Stream } from "effect"
import { type Pipeable, pipeArguments } from "effect/Pipeable"
import native from "./internal/native.js"

// The constructor is a runtime export of the loaded `.node` addon. The class
// in `internal/native-addon.ts` is only a compile-time declaration of that ABI.
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
 * Failure from an Effect-returning tokenizer constructor or member. Its
 * members are `_tag: "TokenizerError"`, the operation label, and a diagnostic
 * message. Failures include native errors, ordinary file I/O, invalid
 * serialized tokenizers, chat-context serialization, and batch shape or
 * padding errors. Vocabulary lookup members return {@link Option.Option}
 * instead. During file training, a line-read or UTF-8 error after a successful
 * open has the separate truncation behavior documented by {@link TrainSource}.
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
 * Facade batch-padding policy. `None` requires equal row lengths after facade
 * truncation; `Longest` pads to the longest row; `MaxLength` produces exactly
 * `maxLength` columns and fails if a row is still longer. Padding does not
 * truncate and applies only to {@link Tokenizer.encodeBatch}. It is independent
 * of any native padding serialized in a loaded tokenizer.
 *
 * A `maxLength` must be a finite non-negative integer. A `padId` must be an
 * unsigned 32-bit integer; membership in the vocabulary is not checked.
 * Neither constructors nor encoding comprehensively validate these numeric
 * constraints. Invalid values may be coerced by typed-array operations, produce
 * invalid shape metadata, or fail the encoding Effect.
 *
 * @since 0.1.0
 * @category models
 */
export type Padding =
  | { readonly _tag: "None" }
  | { readonly _tag: "Longest"; readonly padId: number }
  | {
    readonly _tag: "MaxLength"
    readonly maxLength: number
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
 * Pads batch rows to exactly `maxLength` with `padId`; it does not truncate
 * rows that exceed that length.
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
 * Facade per-encode truncation policy, applied after native encoding and before
 * facade batch padding. `maxLength` must be a finite non-negative integer.
 * Neither construction nor encoding validates this comprehensively; invalid
 * values are subject to typed-array slice coercion and are outside the
 * supported contract. With `MaxLength` padding, the post-truncation row must
 * fit the padding length or {@link Tokenizer.encodeBatch} fails.
 *
 * @since 0.1.0
 * @category models
 */
export type Truncation =
  | { readonly _tag: "None" }
  | { readonly _tag: "MaxLength"; readonly maxLength: number }

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
 * the loaded tokenizer's ordinary added-token matching, including each token's
 * normalization and boundary options.
 *
 * `"Never"` scans the raw input for exact registered special-token contents,
 * longest first, and encodes the surrounding text and occurrences as separate
 * pieces. Exact multi-character occurrences are subdivided so they do not
 * resolve through a whole-token lookup. This changes merge and normalization
 * boundaries and can therefore differ from one uninterrupted ordinary encode.
 * It does not suppress normalized aliases that differ from the raw content or
 * one-character special tokens, which cannot be subdivided. Postprocessors may
 * also insert special ids. The policy is consequently not an absolute promise
 * that output contains no special-token id.
 *
 * @since 0.1.0
 * @category models
 */
export type SpecialTokenPolicy = "Never" | "Always"

/**
 * Facade behavior configuration. All fields are required. Native construction
 * snapshots `specialTokens`; the returned facade retains this object and reads
 * its padding and truncation policies when each encoding result is processed.
 * The object and nested policies are not cloned or frozen. The caller retains
 * ownership but must keep them unchanged while the tokenizer is in use.
 *
 * {@link fromFile} and {@link fromJson} preserve native padding and truncation
 * serialized in `tokenizer.json`; this config supplements rather than clears
 * those settings. Each facade batch row is sent through native encoding as an
 * independent sequence. Native `BatchLongest` is therefore not computed over
 * the facade batch. Under `"Never"`, native encoding runs independently for
 * each nonempty segment, so serialized native padding or truncation can occur
 * inside the eventual merged row. Facade truncation and batch padding run only
 * after native encoding completes.
 *
 * @since 0.1.0
 * @category models
 */
export interface TokenizerConfig {
  /** Facade batch padding; used only by `encodeBatch`. */
  readonly padding: Padding
  /** Facade per-sequence truncation applied after native encoding. */
  readonly truncation: Truncation
  /** Input special-token policy snapshotted by the native handle at construction. */
  readonly specialTokens: SpecialTokenPolicy
}

/**
 * Token ids in a fresh, writable host buffer owned by the caller. The native
 * tokenizer does not retain or observe mutations to returned buffers. Values
 * returned by this module use shape `[T]` for one or concatenated sequences
 * and `[B, T]` for facade batches, with row-major data. The object, buffer, and
 * shape array are not frozen.
 *
 * @since 0.1.0
 * @category models
 */
export interface TokenIds {
  /** Caller-owned row-major token ids. */
  readonly data: Uint32Array
  /** Logical dimensions whose product equals `data.length` for module output. */
  readonly shape: ReadonlyArray<number>
  /** Element type metadata; always unsigned 32-bit integers. */
  readonly dtype: "u32"
}

/**
 * Values accepted by tokenizer decode operations. Supported array elements are
 * unsigned 32-bit integer ids. Inputs are copied to an ordinary number array
 * before entering native code. For {@link TokenIds}, decoding reads only
 * `data`; `shape` and `dtype` are not validated or interpreted. JavaScript
 * numbers outside the supported range can be coerced at the Node-API boundary.
 *
 * @since 0.1.0
 * @category models
 */
export type TokenIdInput = TokenIds | Uint32Array | ReadonlyArray<number>

/**
 * Per-call behavior for {@link Tokenizer.encode}. `addSpecialTokens` is passed
 * to the native postprocessor and defaults to `true`. It does not enable or
 * disable input added-token matching, which is governed separately by
 * {@link SpecialTokenPolicy}. Native truncation can also use this flag when it
 * reserves room for postprocessor-added ids.
 *
 * In the `"Never"` segmented path, each segment is natively encoded with this
 * flag disabled. If requested, the postprocessor is then applied once to the
 * merged encoding. That final application is not followed by another native
 * truncation or padding pass. Batch methods expose no equivalent option and
 * always request special-token addition.
 *
 * @since 0.1.0
 * @category models
 */
export interface EncodeOptions {
  readonly addSpecialTokens?: boolean | undefined
}

/**
 * Per-call decoding behavior. `skipSpecialTokens` omits ids whose token string
 * is registered as a special added token and defaults to `false`. It does not
 * automatically omit a facade `padId` unless that token is registered special.
 *
 * @since 0.1.0
 * @category models
 */
export interface DecodeOptions {
  readonly skipSpecialTokens?: boolean | undefined
}

/**
 * Shared configuration with no facade padding or truncation and the `"Never"`
 * special-token policy. It does not disable native padding or truncation in a
 * loaded tokenizer, and `"Never"` has the limitations documented by
 * {@link SpecialTokenPolicy}. This object and its nested singleton policies are
 * not frozen and must be treated as read-only.
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
 * The model family to train. BPE uses byte-level preprocessing, WordPiece a
 * BERT normalizer and `##` continuations, Unigram the SentencePiece metaspace
 * convention plus byte fallback, and WordLevel whitespace tokens. WordPiece
 * automatically adds `[UNK]`; Unigram automatically adds `<unk>` at id `0`.
 * BPE and WordLevel add no unknown token automatically.
 *
 * @since 0.1.0
 * @category models
 */
export type TrainModel = "BPE" | "WordPiece" | "Unigram" | "WordLevel"

/**
 * Where {@link train} reads its corpus. `Files` are UTF-8 text streamed in path
 * order and fed one line at a time with `\n` and an optional preceding `\r`
 * removed; line boundaries are sequence boundaries and are not trained as
 * characters. Every path is statted and opened before corpus consumption;
 * those failures fail the Effect. Any line-read or UTF-8 error after opening is
 * silently treated as EOF for that path, and training continues with an
 * incomplete corpus. `Texts` feeds each string as one sequence.
 *
 * Source constructors retain their input arrays without copying them. The lazy
 * training Effect snapshots their strings into native vectors when it starts.
 * Keep the arrays unchanged until that point.
 *
 * @since 0.1.0
 * @category models
 */
export type TrainSource =
  | { readonly _tag: "Files"; readonly paths: ReadonlyArray<string> }
  | { readonly _tag: "Texts"; readonly texts: ReadonlyArray<string> }

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
 * `Texts` totals sum the strings' UTF-8 lengths. `Files` totals use raw file
 * sizes while processed counts exclude stripped line terminators, so an
 * intermediate processed value can trail the total. Non-empty feed completion
 * is pinned to `(total, total)`. This pin does not mean model computation is
 * complete or prove that every file byte was consumed; it is still emitted
 * after a file is truncated by a swallowed read or UTF-8 error. No progress is
 * available for the model computation that follows corpus exhaustion.
 *
 * Reporting is checked once per sequence and emitted after at least
 * `everyBytes` more processed bytes; a large sequence produces one report, not
 * catch-up reports. The final pin is emitted even below the interval. The
 * supported `everyBytes` values are finite numbers; they are floored and
 * lower-bounded at zero, and zero disables every report, including completion.
 * An empty corpus emits no completion report. Native callbacks are posted
 * non-blockingly to JavaScript; the facade runs received report Effects in
 * order. A report failure or interruption stops the facade from awaiting
 * further reports or the result, but native training already started continues
 * and cannot be cancelled.
 *
 * @since 0.1.0
 * @category models
 */
export type TrainProgress<E, R> =
  | { readonly _tag: "None" }
  | {
    /** Selects Effect-based progress reporting. */
    readonly _tag: "Report"
    /** Finite approximate minimum byte interval; zero disables reporting. */
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
 * Reports feed progress at sequence boundaries using `everyBytes` as the
 * approximate minimum interval.
 *
 * @since 0.1.0
 * @category constructors
 */
export const trainProgressReport = <E, R>(
  everyBytes: number,
  report: (processed: number, total: number) => Effect.Effect<void, E, R>
): TrainProgress<E, R> => ({ _tag: "Report", everyBytes, report })

/**
 * Configuration for {@link train}. The object and its arrays are read when the
 * lazy Effect starts and copied across the native boundary; they are not
 * retained by the resulting tokenizer. `vocabSize` is a trainer target, not an
 * exact result: corpus size and frequency filtering can produce fewer ids,
 * required alphabets and special tokens constrain the minimum, and Unigram
 * appends missing byte-fallback pieces after training.
 *
 * Numeric fields are represented by unsigned 32-bit integers natively but are
 * not validated by this facade. Supply positive integer `vocabSize` and
 * non-negative integer `minFrequency` values within that range; other numbers
 * may be coerced by Node-API before training.
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
  /** Unsigned 32-bit frequency cutoff; not used by the Unigram trainer. */
  readonly minFrequency: number
  /** Registered special tokens, ordered before learned tokens by the trainer. */
  readonly specialTokens: ReadonlyArray<string>
  /** Corpus-feed progress policy. */
  readonly progress: TrainProgress<E, R>
}

/**
 * One conversational message for a caller-provided Jinja chat template.
 * `role`, `content`, and additional fields are exposed as JSON values. Values
 * therefore need to be serializable by `JSON.stringify`; `undefined` object
 * fields are omitted and unsupported values such as `bigint` cause the render
 * Effect to fail.
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
 * Context options for a caller-provided Jinja chat template. `variables`
 * supplies top-level, JSON-serializable fields such as `bos_token`,
 * `eos_token`, tools, dates, or cutoff metadata. No token strings or other
 * model metadata are inferred from the tokenizer. The wrapper overwrites direct
 * `messages` and `add_generation_prompt` keys after spreading `variables`, so
 * ordinary keys with those names cannot replace the wrapper values. As with
 * any `JSON.stringify` input, a custom `toJSON` property can transform the
 * entire context and should not be supplied unless that behavior is intended.
 *
 * @since 0.1.0
 * @category models
 */
export interface ChatTemplateOptions {
  /** Value exposed as `add_generation_prompt`; the template decides how to use it. */
  readonly addGenerationPrompt?: boolean | undefined
  /** Additional variables exposed to the Jinja rendering context. */
  readonly variables?: Readonly<Record<string, unknown>> | undefined
}

/**
 * A native-backed text tokenizer. The TypeScript facade is a plain, unfrozen
 * object whose methods close over one immutable native handle and the retained
 * {@link TokenizerConfig}. Native state is safe to share across concurrent
 * batch work; callers must not mutate the facade or retained configuration.
 *
 * The native handle owns vocabulary and pipeline data on the CPU heap. It owns
 * no persistent file handle, device buffer, or worker, and background work
 * keeps its own native reference alive. Node-API finalization reclaims it after
 * the facade and in-flight work become unreachable, so there is no explicit
 * disposal operation.
 *
 * @since 0.1.0
 * @category models
 */
export interface Tokenizer extends Pipeable {
  /** Runtime marker used by {@link isTokenizer}; not an authenticity check. */
  readonly [TokenizerTypeId]: TokenizerTypeId
  /**
   * Vocabulary size including added and special tokens, snapshotted when the
   * facade is created.
   */
  readonly vocabSize: number
  /**
   * Encodes text into caller-owned `[T]` `u32` ids. Serialized native policy
   * runs first, followed by facade truncation; facade batch padding is ignored.
   * Under `"Never"`, serialized native padding and truncation can apply to each
   * segment before their ids are merged. See {@link EncodeOptions} for the
   * postprocessor path. Native encoding runs synchronously on the JavaScript
   * thread and cannot be interrupted once the call starts.
   */
  readonly encode: (
    text: string,
    options?: EncodeOptions
  ) => Effect.Effect<TokenIds, TokenizerError>
  /**
   * Encodes a non-empty batch in parallel into caller-owned row-major `[B, T]`
   * `u32` ids, preserving input order. Every text is a separate native sequence
   * and always requests postprocessor special-token addition. Serialized native
   * `BatchLongest` is not computed across this facade batch; under `"Never"`,
   * native policy can apply per segment. Facade truncation runs before facade
   * padding, and unequal rows fail when facade padding is `None`.
   *
   * If any row fails, the whole Effect fails without returning partial rows.
   * Native work runs off the JavaScript thread, but interrupting the Effect does
   * not cancel work already started.
   */
  readonly encodeBatch: (
    texts: ReadonlyArray<string>
  ) => Effect.Effect<TokenIds, TokenizerError>
  /**
   * Encodes a batch in parallel, always requesting postprocessor special-token
   * addition, and concatenates its facade-truncated rows into caller-owned
   * `[sum(T)]` `u32` ids in input order. Facade padding is ignored; serialized
   * native policy may already have applied to each sequence or `"Never"`
   * segment. Row boundaries are not returned. An empty batch returns an empty
   * buffer with shape `[0]`. Native work runs off the JavaScript thread, but
   * interruption does not cancel work already started.
   */
  readonly encodeBatchConcat: (
    texts: ReadonlyArray<string>
  ) => Effect.Effect<TokenIds, TokenizerError>
  /**
   * Decodes all supplied ids as one flat sequence. A {@link TokenIds} shape is
   * ignored, registered special tokens are included unless
   * `skipSpecialTokens` is true, and ids absent from the vocabulary are
   * silently omitted. The loaded decoder determines spacing, byte fallback,
   * and other reconstruction behavior; without a decoder, token strings are
   * joined with spaces. Decoding is not guaranteed to invert encoding,
   * truncation, padding, or normalization. The native call runs synchronously
   * on the JavaScript thread and cannot be interrupted once started.
   */
  readonly decode: (
    ids: TokenIdInput,
    options?: DecodeOptions
  ) => Effect.Effect<string, TokenizerError>
  /**
   * Decodes each outer input as one independent sequence and preserves outer
   * order. Inner {@link TokenIds} shapes are ignored; a `[B, T]` value is not
   * split into rows automatically. An empty outer array returns an empty array.
   * The operation is synchronous from JavaScript's perspective and blocks until
   * the native batch decode, which may use parallel workers internally,
   * finishes. It cannot be interrupted once started.
   */
  readonly decodeBatch: (
    ids: ReadonlyArray<TokenIdInput>,
    options?: DecodeOptions
  ) => Effect.Effect<ReadonlyArray<string>, TokenizerError>
  /**
   * Serializes a context containing `messages`, `add_generation_prompt`, and
   * caller variables, then renders the supplied template to text. It neither
   * reads a template from `tokenizer.json` nor tokenizes the rendered result.
   * Context serialization happens when the lazy Effect runs.
   *
   * The native renderer is MiniJinja with its compiled-in built-in tests,
   * filters, and `tojson`, plus `raise_exception(message)` and
   * `strftime_now(format)`. `strftime_now` uses UTC and supports only `%Y`,
   * `%m`, `%d`, `%H`, `%M`, `%S`, and `%%`. Missing values use MiniJinja's
   * default lenient undefined behavior, one final template newline is stripped,
   * and no template loader or Transformers method-compatibility callback is
   * installed for includes, imports, or calls such as `mapping.items()`. This
   * is not a promise of full Python Jinja or Transformers-extension
   * compatibility.
   * Rendering runs synchronously on the JavaScript thread and cannot be
   * interrupted once started.
   */
  readonly applyChatTemplate: (
    template: string,
    messages: ReadonlyArray<ChatMessage>,
    options?: ChatTemplateOptions
  ) => Effect.Effect<string, TokenizerError>
  /** Synchronously returns the id for an exact vocabulary or added token. */
  readonly tokenToId: (token: string) => Option.Option<number>
  /**
   * Synchronously returns the token for an unsigned 32-bit integer id, or
   * `None` when unknown. Other numbers are unsupported and may be coerced by
   * Node-API.
   */
  readonly idToToken: (id: number) => Option.Option<string>
  /**
   * Saves the native tokenizer as a `tokenizer.json`. Native settings are
   * serialized, but the facade {@link TokenizerConfig} is not and must be
   * supplied again when loading. The target is created or truncated directly,
   * not written atomically, so failure can leave a partial file. File I/O and
   * serialization block the JS thread and cannot be interrupted once started.
   */
  readonly save: (path: string) => Effect.Effect<void, TokenizerError>
}

const toTokenizerError = (op: string) => (error: unknown) =>
  new TokenizerError({
    op,
    message: error instanceof Error ? error.message : String(error)
  })

const idsOf = (
  ids: TokenIdInput
): Effect.Effect<ReadonlyArray<number>, TokenizerError> =>
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
        try: () => handle.decode(resolved as Array<number>, options?.skipSpecialTokens ?? false),
        catch: toTokenizerError("decode")
      }))
  self.decodeBatch = (
    batch: ReadonlyArray<TokenIdInput>,
    options?: DecodeOptions
  ) =>
    Effect.flatMap(
      Effect.forEach(batch, idsOf, { concurrency: "unbounded" }),
      (resolved) =>
        Effect.try({
          try: () =>
            handle.decodeBatch(
              resolved.map((row) => row as Array<number>),
              options?.skipSpecialTokens ?? false
            ),
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
 * parsing run synchronously on the JavaScript thread when the Effect executes
 * and cannot be interrupted once started. The supplied config is retained by
 * reference by the returned tokenizer and supplements rather than replaces
 * native padding or truncation serialized in the document.
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
 * Parsing runs synchronously on the JavaScript thread when the Effect executes
 * and cannot be interrupted once started. The supplied config is retained by
 * reference by the returned tokenizer and supplements rather than replaces
 * native padding or truncation serialized in the document.
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
 * returned Effect runs; native file reading, corpus feeding, and model training
 * then run on a blocking worker rather than the JavaScript thread. Interrupting
 * the Effect or failing a report Effect stops awaiting the result but does not
 * cancel native work already started. Training progress has the limits in
 * {@link TrainProgress}. The tokenizer config is retained by reference in the
 * result, while the training config is copied into native-owned input values.
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
          specialTokens: trainConfig.specialTokens as Array<string>,
          source: trainConfig.source._tag === "Files"
            ? {
              tag: "Files",
              paths: trainConfig.source.paths as Array<string>
            }
            : {
              tag: "Texts",
              texts: trainConfig.source.texts as Array<string>
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
 * Tests only for presence of {@link TokenizerTypeId}, including through the
 * prototype chain. It does not check the marker value or validate tokenizer
 * members, so marked or spoofed objects also pass.
 *
 * @since 0.1.0
 * @category guards
 */
export const isTokenizer = (value: unknown): value is Tokenizer =>
  typeof value === "object" && value !== null && TokenizerTypeId in value

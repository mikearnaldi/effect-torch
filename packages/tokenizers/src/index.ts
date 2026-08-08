/**
 * Native-backed text tokenization using the HuggingFace `tokenizers` crate.
 * Encoding returns caller-owned token-id buffers for explicit import by a
 * tensor runtime. Loading accepts `tokenizer.json` components supported by
 * the bundled crate; documents using unsupported or custom components fail.
 *
 * `specialTokens: "Always"` explicitly parses registered special-token
 * strings. `"Never"` routes them through ordinary tokenization, although a
 * one-character special can still resolve through the tokenizer vocabulary.
 *
 * @since 0.1.0
 * @module
 */
import { Data, Effect, Option, Queue, Stream } from "effect"
import { type Pipeable, pipeArguments } from "effect/Pipeable"
import native from "./internal/native.js"

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
 * message. Failures include native errors, file I/O, invalid serialized
 * tokenizers, and batch shape or padding errors. Vocabulary lookup members
 * return {@link Option.Option} instead.
 *
 * @since 0.1.0
 * @category errors
 */
export class TokenizerError extends Data.TaggedError("TokenizerError")<{
  /** One of `fromFile`, `fromJson`, `train`, `encode`, `encodeBatch`, `encodeBatchConcat`, `decode`, `decodeBatch`, or `save`. */
  readonly op: string
  /** Human-readable diagnostic from the wrapper or native implementation. */
  readonly message: string
}> {}

/**
 * Batch padding policy. `None` requires equal row lengths after truncation;
 * `Longest` pads to the longest row; `MaxLength` produces exactly
 * `maxLength` columns and fails if a row is still longer. Padding does not
 * truncate and applies only to {@link Tokenizer.encodeBatch}.
 *
 * A `maxLength` must be a finite non-negative integer. A `padId` must be an
 * unsigned 32-bit integer; membership in the vocabulary is not checked.
 * These numeric constraints are not validated by the constructors.
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
 * Per-encode truncation policy, applied before batch padding. `maxLength`
 * must be a finite non-negative integer, but the constructor does not
 * validate it. With `MaxLength` padding, the post-truncation row must fit the
 * padding length or {@link Tokenizer.encodeBatch} fails.
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
 * Whether registered special-token strings in input text are parsed into
 * their special ids (`"Always"`) or routed through ordinary tokenization
 * (`"Never"`). A one-character registered string cannot be split and may
 * still resolve to its special id through the tokenizer vocabulary.
 * Configured post-processors may also insert tokens; this policy controls
 * input parsing, not structural insertion.
 *
 * @since 0.1.0
 * @category models
 */
export type SpecialTokenPolicy = "Never" | "Always"

/**
 * Tokenizer behavior configuration. All fields are required. Construction
 * captures `specialTokens`, then retains this object and reads its padding
 * and truncation policies during encoding. The object and nested policies are
 * not cloned or frozen. The caller retains ownership but must not mutate them
 * while the tokenizer is in use.
 *
 * @since 0.1.0
 * @category models
 */
export interface TokenizerConfig {
  /** Batch padding policy; ignored by single and concatenated encoding. */
  readonly padding: Padding
  /** Per-sequence truncation applied by all encoding members. */
  readonly truncation: Truncation
  /** Controls special-token parsing; captured when the tokenizer is constructed. */
  readonly specialTokens: SpecialTokenPolicy
}

/**
 * Token ids in a fresh, writable host buffer owned by the caller. Values
 * returned by this module use shape `[T]` for one or concatenated sequences
 * and `[B, T]` for padded batches, with row-major data. The object, buffer,
 * and shape array are not frozen.
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
 * Values accepted by tokenizer decode operations. Array elements must be
 * unsigned 32-bit integer ids. For {@link TokenIds}, decoding reads only
 * `data`; `shape` and `dtype` are not validated or interpreted.
 *
 * @since 0.1.0
 * @category models
 */
export type TokenIdInput = TokenIds | Uint32Array | ReadonlyArray<number>

/**
 * Shared configuration with no padding or truncation and ordinary parsing of
 * special-token strings. This object and its nested singleton policies are
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
 * BERT normalizer and `##` continuations, Unigram the SentencePiece `▁`
 * metaspace convention plus byte fallback, and WordLevel whitespace tokens.
 * WordPiece automatically adds `[UNK]`; Unigram automatically adds `<unk>`
 * at id `0`. BPE and WordLevel add no unknown token automatically.
 *
 * @since 0.1.0
 * @category models
 */
export type TrainModel = "BPE" | "WordPiece" | "Unigram" | "WordLevel"

/**
 * Where {@link train} reads its corpus. `Files` are UTF-8 text streamed in
 * path order and fed one line at a time with `\n` and optional preceding `\r`
 * removed; line boundaries are sequence boundaries and are not trained as
 * characters. A line read or UTF-8 error stops that file's feed. `Texts`
 * feeds each string as one sequence.
 *
 * Source constructors retain their input arrays without copying them. Keep
 * those arrays unchanged while a source may be used by a lazy training
 * Effect.
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
 * `Texts` totals sum encoded string lengths. `Files` totals use raw file
 * sizes while processed counts exclude stripped line terminators, so an
 * intermediate processed value can trail the total. Non-empty feed
 * completion is pinned to `(total, total)` and does not mean model computation
 * is complete.
 *
 * Reporting is checked once per sequence and emitted after at least
 * `everyBytes` more processed bytes; a large sequence produces one report,
 * not catch-up reports. The final pin is emitted even below the interval.
 * Finite `everyBytes` values are floored and lower-bounded at zero; zero
 * disables all reports. Report Effects run in order on the JS side. Failure
 * or interruption stops awaiting reports, but does not cancel native training
 * already in progress.
 *
 * @since 0.1.0
 * @category models
 */
export type TrainProgress<E, R> =
  | { readonly _tag: "None" }
  | {
    /** Selects Effect-based progress reporting. */
    readonly _tag: "Report"
    /** Finite approximate minimum byte interval; normalized at runtime. */
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
 * Configuration for {@link train}. `vocabSize` is a trainer target, not an
 * exact result: corpus size and frequency filtering can produce fewer ids,
 * required alphabets and special tokens constrain the minimum, and Unigram
 * appends missing byte-fallback pieces after training.
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
 * A native-backed text tokenizer. The facade and retained configuration are
 * not runtime-frozen; concurrent use requires callers not to mutate either.
 * The native handle owns CPU heap and is reclaimed by GC finalization, so no
 * explicit disposal is required.
 *
 * @since 0.1.0
 * @category models
 */
export interface Tokenizer extends Pipeable {
  /** Runtime marker used by {@link isTokenizer}; not an authenticity check. */
  readonly [TokenizerTypeId]: TokenizerTypeId
  /**
   * Vocabulary size including added and special tokens.
   */
  readonly vocabSize: number
  /**
   * Encodes text into caller-owned `[T]` `u32` ids. Truncation applies but
   * padding does not. Native encoding runs synchronously on the JS thread and
   * cannot be interrupted once the call starts.
   */
  readonly encode: (
    text: string
  ) => Effect.Effect<TokenIds, TokenizerError>
  /**
   * Encodes a non-empty batch in parallel into caller-owned `[B, T]` `u32`
   * ids. Rows are truncated before applying the padding policy; without
   * padding, unequal row lengths fail. Native encoding runs off the JS thread,
   * but interrupting the Effect does not cancel work already started.
   */
  readonly encodeBatch: (
    texts: ReadonlyArray<string>
  ) => Effect.Effect<TokenIds, TokenizerError>
  /**
   * Encodes a batch in parallel and concatenates its post-truncation rows into
   * caller-owned `[sum(T)]` `u32` ids in input order. Padding is ignored and
   * an empty batch returns `[0]`. Native work runs off the JS thread, but
   * interruption does not cancel work already started.
   */
  readonly encodeBatchConcat: (
    texts: ReadonlyArray<string>
  ) => Effect.Effect<TokenIds, TokenizerError>
  /**
   * Decodes all supplied data as one flat sequence. A {@link TokenIds} shape
   * is ignored, special tokens are included, and ids absent from the
   * vocabulary are silently omitted. Decoding is pipeline-dependent and is
   * not guaranteed to invert encoding. The native call runs synchronously on
   * the JS thread and cannot be interrupted once started.
   */
  readonly decode: (
    ids: TokenIdInput
  ) => Effect.Effect<string, TokenizerError>
  /**
   * Decodes each outer input as one sequence. Inner {@link TokenIds} shapes
   * are ignored; a `[B, T]` value is not split into rows automatically. The
   * native batch call runs synchronously on the JS thread.
   */
  readonly decodeBatch: (
    ids: ReadonlyArray<TokenIdInput>
  ) => Effect.Effect<ReadonlyArray<string>, TokenizerError>
  /** Synchronously returns the id for an exact vocabulary or added token. */
  readonly tokenToId: (token: string) => Option.Option<number>
  /** Synchronously returns the token for an id, or `None` when it is unknown. */
  readonly idToToken: (id: number) => Option.Option<string>
  /**
   * Saves the native tokenizer as a `tokenizer.json`. File I/O and
   * serialization run synchronously on the JS thread and cannot be
   * interrupted once started.
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
  self.encode = (text: string) =>
    Effect.try({
      try: () => {
        const data = truncate(handle.encode(text), config)
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
  self.decode = (ids: TokenIdInput) =>
    Effect.flatMap(idsOf(ids), (resolved) =>
      Effect.try({
        try: () => handle.decode(resolved as Array<number>),
        catch: toTokenizerError("decode")
      }))
  self.decodeBatch = (
    batch: ReadonlyArray<TokenIdInput>
  ) =>
    Effect.flatMap(
      Effect.forEach(batch, idsOf, { concurrency: "unbounded" }),
      (resolved) =>
        Effect.try({
          try: () => handle.decodeBatch(resolved.map((row) => row as Array<number>)),
          catch: toTokenizerError("decodeBatch")
        })
    )
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
 * parsing run synchronously on the JS thread when the Effect executes and
 * cannot be interrupted once started. The supplied config is retained by
 * reference by the returned tokenizer.
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
 * Parsing runs synchronously on the JS thread when the Effect executes and
 * cannot be interrupted once started. The supplied config is retained by
 * reference by the returned tokenizer.
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
 * Trains a tokenizer from a {@link TrainSource}. Native file reading and
 * training run off the JS thread. Interrupting the Effect or failing a report
 * Effect stops awaiting the result but does not cancel native work already
 * started. The tokenizer config is retained by reference in the result.
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

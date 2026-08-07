/**
 * RFC 0009: tokenizers. A `Tokenizer` turns text into host-owned token ids.
 * Encoding, batch encoding and training run natively (the HuggingFace
 * `tokenizers` crate over napi), while tensor runtimes explicitly import the
 * returned data. Loading is `tokenizer.json`-compatible, so every
 * HuggingFace Hub tokenizer works out of the box, and `train` builds BPE,
 * WordPiece, Unigram or WordLevel tokenizers from raw text files.
 *
 * Special-token strings in input text are never parsed into special-token
 * ids unless the tokenizer is configured with `specialTokens: "Always"` —
 * the tiktoken `allowed_special` discipline.
 */
import native, { type NativeTokenizer as NativeTokenizerType } from "@effect-torch/native"
import { Data, Effect, Option, Queue, Stream } from "effect"
import { type Pipeable, pipeArguments } from "effect/Pipeable"

const { NativeTokenizer } = native

/**
 * @since 0.1.0
 * @category symbols
 */
export const TokenizerTypeId: unique symbol = Symbol.for(
  "@effect-torch/tokenizers/Tokenizer"
)

/**
 * @since 0.1.0
 * @category symbols
 */
export type TokenizerTypeId = typeof TokenizerTypeId

/**
 * Error type raised by tokenizer operations: native load/train/encode
 * failures, ragged batches without padding, and decode of invalid ids.
 *
 * @since 0.1.0
 * @category errors
 */
export class TokenizerError extends Data.TaggedError("TokenizerError")<{
  readonly op: string
  readonly message: string
}> {}

/**
 * Batch padding policy. `None` makes {@link Tokenizer.encodeBatch} fail on
 * ragged encodings instead of silently picking a pad id.
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
 * @since 0.1.0
 * @category constructors
 */
export const paddingNone: Padding = { _tag: "None" }

/**
 * @since 0.1.0
 * @category constructors
 */
export const paddingLongest = (padId: number): Padding => ({
  _tag: "Longest",
  padId
})

/**
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
 * Per-encode truncation policy, applied before padding.
 *
 * @since 0.1.0
 * @category models
 */
export type Truncation =
  | { readonly _tag: "None" }
  | { readonly _tag: "MaxLength"; readonly maxLength: number }

/**
 * @since 0.1.0
 * @category constructors
 */
export const truncationNone: Truncation = { _tag: "None" }

/**
 * @since 0.1.0
 * @category constructors
 */
export const truncationMaxLength = (maxLength: number): Truncation => ({
  _tag: "MaxLength",
  maxLength
})

/**
 * Whether special-token strings occurring in input text are parsed into
 * their special ids (`"Always"`) or tokenized as ordinary text (`"Never"`).
 * Post-processors configured in the tokenizer (BOS/EOS templates) apply
 * regardless — this controls parsing, not structural insertion.
 *
 * @since 0.1.0
 * @category models
 */
export type SpecialTokenPolicy = "Never" | "Always"

/**
 * Immutable tokenizer behaviour configuration. All fields are required:
 * padding, truncation and special-token handling are explicit decisions.
 *
 * @since 0.1.0
 * @category models
 */
export interface TokenizerConfig {
  readonly padding: Padding
  readonly truncation: Truncation
  readonly specialTokens: SpecialTokenPolicy
}

/**
 * Host-owned token ids ready for explicit import by a tensor runtime.
 *
 * @since 0.1.0
 * @category models
 */
export interface TokenIds {
  readonly data: Uint32Array
  readonly shape: ReadonlyArray<number>
  readonly dtype: "u32"
}

/**
 * Values accepted by tokenizer decode operations.
 *
 * @since 0.1.0
 * @category models
 */
export type TokenIdInput = TokenIds | Uint32Array | ReadonlyArray<number>

/**
 * The strictest configuration: no padding, no truncation, special-token
 * strings tokenize as ordinary text.
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
 * The subword model family to train. Pipeline defaults follow the canonical
 * setups: BPE trains byte-level (GPT-2 style), WordPiece with the BERT
 * normalizer and `##` continuations, Unigram with the SentencePiece `▁`
 * metaspace convention, WordLevel on whitespace-split words.
 *
 * @since 0.1.0
 * @category models
 */
export type TrainModel = "BPE" | "WordPiece" | "Unigram" | "WordLevel"

/**
 * Where {@link train} reads its corpus from: raw text `Files` streamed
 * from disk, or `Texts` already in memory.
 *
 * @since 0.1.0
 * @category models
 */
export type TrainSource =
  | { readonly _tag: "Files"; readonly paths: ReadonlyArray<string> }
  | { readonly _tag: "Texts"; readonly texts: ReadonlyArray<string> }

/**
 * @since 0.1.0
 * @category constructors
 */
export const trainFiles = (paths: ReadonlyArray<string>): TrainSource => ({
  _tag: "Files",
  paths
})

/**
 * @since 0.1.0
 * @category constructors
 */
export const trainTexts = (texts: ReadonlyArray<string>): TrainSource => ({
  _tag: "Texts",
  texts
})

/**
 * Training progress reporting. The corpus feed is the dominant cost on
 * large corpora and is reported as `(processed, total)` corpus bytes; one
 * final `(total, total)` event signals that the feed is complete and the
 * (indeterminate) merge computation has begun. Reports are throttled by
 * `everyBytes`: the callback fires at most once per `everyBytes` corpus
 * bytes consumed; `everyBytes: 0` disables reporting (including the
 * completion event) entirely. Granularity is per sequence: progress
 * advances as
 * corpus sequences are pulled, so a corpus of many short sequences (e.g.
 * lines) reports smoothly while a single huge sequence reports once. The
 * callback runs on the JS thread and returns an Effect — what to do with
 * the events (log, render, ignore) is the caller's decision.
 *
 * @since 0.1.0
 * @category models
 */
export type TrainProgress<E, R> =
  | { readonly _tag: "None" }
  | {
    readonly _tag: "Report"
    readonly everyBytes: number
    readonly report: (processed: number, total: number) => Effect.Effect<void, E, R>
  }

/**
 * @since 0.1.0
 * @category constructors
 */
export const trainProgressNone: TrainProgress<never, never> = { _tag: "None" }

/**
 * Reports at most once per `everyBytes` corpus bytes consumed.
 *
 * @since 0.1.0
 * @category constructors
 */
export const trainProgressReport = <E, R>(
  everyBytes: number,
  report: (processed: number, total: number) => Effect.Effect<void, E, R>
): TrainProgress<E, R> => ({ _tag: "Report", everyBytes, report })

/**
 * Configuration for {@link train}. Training is deterministic and streams
 * from the given source. Ids are allocated special tokens first, then the
 * alphabet (BPE seeds the full 256-byte alphabet), then merges or pieces
 * in rank order.
 *
 * @since 0.1.0
 * @category models
 */
export interface TrainConfig<E, R> {
  readonly source: TrainSource
  readonly model: TrainModel
  readonly vocabSize: number
  readonly minFrequency: number
  readonly specialTokens: ReadonlyArray<string>
  readonly progress: TrainProgress<E, R>
}

/**
 * A text tokenizer. Values are immutable and safe for concurrent use. The
 * implementation owns only CPU heap (vocab tables, merges, regexes), so it
 * is reclaimed by ordinary GC finalization; no explicit disposal is needed.
 *
 * @since 0.1.0
 * @category models
 */
export interface Tokenizer extends Pipeable {
  readonly [TokenizerTypeId]: TokenizerTypeId
  /**
   * Vocabulary size including special tokens.
   */
  readonly vocabSize: number
  /**
   * Encodes text into host-owned `[T]` `u32` token ids.
   */
  readonly encode: (
    text: string
  ) => Effect.Effect<TokenIds, TokenizerError>
  /**
   * Encodes a batch into host-owned `[B, T]` `u32` ids, padded per the
   * tokenizer's {@link Padding} config with `paddingNone`, ragged
   * encodings fail with {@link TokenizerError}.
   */
  readonly encodeBatch: (
    texts: ReadonlyArray<string>
  ) => Effect.Effect<TokenIds, TokenizerError>
  /**
   * Encodes a batch into one flat host-owned `[ΣT]` `u32` value — the ragged
   * encodings concatenated in order, no padding. The document-stream
   * counterpart of {@link encodeBatch} for corpus tokenization.
   */
  readonly encodeBatchConcat: (
    texts: ReadonlyArray<string>
  ) => Effect.Effect<TokenIds, TokenizerError>
  /**
   * Decodes ids back to text, losslessly (special tokens are not skipped).
   */
  readonly decode: (
    ids: TokenIdInput
  ) => Effect.Effect<string, TokenizerError>
  /**
   * Batch counterpart of {@link Tokenizer.decode}.
   */
  readonly decodeBatch: (
    ids: ReadonlyArray<TokenIdInput>
  ) => Effect.Effect<ReadonlyArray<string>, TokenizerError>
  readonly tokenToId: (token: string) => Option.Option<number>
  readonly idToToken: (id: number) => Option.Option<string>
  /**
   * Saves the tokenizer as a self-contained `tokenizer.json`.
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
 * Loads a tokenizer from a `tokenizer.json` file (the format every
 * HuggingFace Hub tokenizer ships).
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
 * Loads a tokenizer from an in-memory `tokenizer.json` document.
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
 * Trains a tokenizer from a corpus ({@link TrainSource}): raw text files
 * streamed from disk, or texts already in memory. Runs natively off the
 * JS thread the result is immediately usable and `save`-able as
 * `tokenizer.json`.
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
 * Returns `true` if the value is a {@link Tokenizer}.
 *
 * @since 0.1.0
 * @category guards
 */
export const isTokenizer = (value: unknown): value is Tokenizer =>
  typeof value === "object" && value !== null && TokenizerTypeId in value

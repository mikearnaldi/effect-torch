/**
 * RFC 0009: tokenizers. A `Tokenizer` turns text into id tensors — the entry
 * point of the text data plane. Encoding, batch encoding and training run
 * natively (the HuggingFace `tokenizers` crate over napi); `encode` and
 * `encodeBatch` return `u32` tensors built in Rust, so the id buffer never
 * round-trips through JS. Loading is `tokenizer.json`-compatible, so every
 * HuggingFace Hub tokenizer works out of the box, and `train` builds BPE,
 * WordPiece, Unigram or WordLevel tokenizers from raw text files.
 *
 * Special-token strings in input text are never parsed into special-token
 * ids unless the tokenizer is configured with `specialTokens: "Always"` —
 * the tiktoken `allowed_special` discipline.
 */
import { Data, Effect, Option } from "effect"
import { pipeArguments, type Pipeable } from "effect/Pipeable"
import native, {
  type NativePadding as NativePaddingType,
  type NativeTokenizer as NativeTokenizerType,
  type NativeTruncation as NativeTruncationType
} from "@effect-torch/native"
import { CurrentDevice } from "./Device.ts"
import * as Tensor from "./Tensor.ts"

const { NativeTokenizer } = native

/**
 * @since 0.1.0
 * @category symbols
 */
export const TokenizerTypeId: unique symbol = Symbol.for("@effect-torch/core/Tokenizer")

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
  | { readonly _tag: "MaxLength"; readonly maxLength: number; readonly padId: number }

/**
 * @since 0.1.0
 * @category constructors
 */
export const paddingNone: Padding = { _tag: "None" }

/**
 * @since 0.1.0
 * @category constructors
 */
export const paddingLongest = (padId: number): Padding => ({ _tag: "Longest", padId })

/**
 * @since 0.1.0
 * @category constructors
 */
export const paddingMaxLength = (maxLength: number, padId: number): Padding => ({
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
export const truncationMaxLength = (maxLength: number): Truncation => ({ _tag: "MaxLength", maxLength })

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
export const trainFiles = (paths: ReadonlyArray<string>): TrainSource => ({ _tag: "Files", paths })

/**
 * @since 0.1.0
 * @category constructors
 */
export const trainTexts = (texts: ReadonlyArray<string>): TrainSource => ({ _tag: "Texts", texts })

/**
 * Training progress reporting. The corpus feed is the dominant cost on
 * large corpora and is reported as `(processed, total)` corpus bytes,
 * throttled natively; one final `(total, total)` event signals that the
 * feed is complete and the (indeterminate) merge computation has begun.
 * The callback runs on the JS thread — what to do with the events (log,
 * render, ignore) is the caller's decision.
 *
 * @since 0.1.0
 * @category models
 */
export type TrainProgress<E, R> =
  | { readonly _tag: "None" }
  | { readonly _tag: "Report"; readonly report: (processed: number, total: number) => Effect.Effect<void, E, R> }

/**
 * @since 0.1.0
 * @category constructors
 */
export const trainProgressNone: TrainProgress<never, never> = { _tag: "None" }

/**
 * @since 0.1.0
 * @category constructors
 */
export const trainProgressReport = (
  report: (processed: number, total: number) => void
): TrainProgress => ({ _tag: "Report", report })

/**
 * Configuration for {@link train}. Training is deterministic and streams
 * from the given source. Ids are allocated special tokens first, then the
 * alphabet (BPE seeds the full 256-byte alphabet), then merges or pieces
 * in rank order.
 *
 * @since 0.1.0
 * @category models
 */
export interface TrainConfig {
  readonly source: TrainSource
  readonly model: TrainModel
  readonly vocabSize: number
  readonly minFrequency: number
  readonly specialTokens: ReadonlyArray<string>
  readonly progress: TrainProgress
}

/**
 * A text tokenizer. Values are immutable and safe for concurrent use; the
 * native handle owns only CPU heap (vocab tables, merges, regexes), so it
 * is reclaimed by ordinary GC finalization — no explicit disposal.
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
   * Encodes text into a `[T]` `u32` tensor of token ids.
   */
  readonly encode: (text: string) => Effect.Effect<Tensor.Lazy, TokenizerError, CurrentDevice>
  /**
   * Encodes a batch into a `[B, T]` `u32` tensor, padded per the
   * tokenizer's {@link Padding} config; with `paddingNone`, ragged
   * encodings fail with {@link TokenizerError}.
   */
  readonly encodeBatch: (
    texts: ReadonlyArray<string>
  ) => Effect.Effect<Tensor.Lazy, TokenizerError, CurrentDevice>
  /**
   * Decodes ids back to text, losslessly (special tokens are not skipped).
   * Tensor inputs are materialized natively.
   */
  readonly decode: (ids: Tensor.Any | ReadonlyArray<number>) => Effect.Effect<string, TokenizerError>
  /**
   * Batch counterpart of {@link Tokenizer.decode}.
   */
  readonly decodeBatch: (
    ids: ReadonlyArray<Tensor.Any | ReadonlyArray<number>>
  ) => Effect.Effect<ReadonlyArray<string>, TokenizerError>
  readonly tokenToId: (token: string) => Option.Option<number>
  readonly idToToken: (id: number) => Option.Option<string>
  /**
   * Saves the tokenizer as a self-contained `tokenizer.json`.
   */
  readonly save: (path: string) => Effect.Effect<void, TokenizerError>
}

const toNativePadding = (padding: Padding): NativePaddingType => {
  switch (padding._tag) {
    case "None":
      return { tag: "None" }
    case "Longest":
      return { tag: "Longest", padId: padding.padId }
    case "MaxLength":
      return { tag: "MaxLength", maxLength: padding.maxLength, padId: padding.padId }
  }
}

const toNativeTruncation = (truncation: Truncation): NativeTruncationType => {
  switch (truncation._tag) {
    case "None":
      return { tag: "None" }
    case "MaxLength":
      return { tag: "MaxLength", maxLength: truncation.maxLength }
  }
}

const toTokenizerError = (op: string) => (error: unknown) =>
  new TokenizerError({ op, message: error instanceof Error ? error.message : String(error) })

const idsOf = (
  ids: Tensor.Any | ReadonlyArray<number>
): Effect.Effect<ReadonlyArray<number>, TokenizerError> =>
  Array.isArray(ids)
    ? Effect.succeed(ids as ReadonlyArray<number>)
    : Effect.map(
      Effect.mapError(Tensor.toTypedArray(ids as Tensor.Any), toTokenizerError("decode")),
      (data) => Array.from(data, Number)
    )

const TokenizerProto = {
  pipe() {
    return pipeArguments(this, arguments)
  }
}

const make = (handle: NativeTokenizerType, config: TokenizerConfig): Tokenizer => {
  const self = Object.create(TokenizerProto)
  self[TokenizerTypeId] = TokenizerTypeId
  self.vocabSize = handle.vocabSize
  self.encode = (text: string) =>
    Effect.gen(function* () {
      const device = yield* CurrentDevice
      return yield* Effect.try({
        try: () => {
          const lazy = handle.encodeTensor(text, device)
          return Tensor.makeLazy(lazy, lazy.shape, "u32", device)
        },
        catch: toTokenizerError("encode")
      })
    })
  self.encodeBatch = (texts: ReadonlyArray<string>) =>
    Effect.gen(function* () {
      const device = yield* CurrentDevice
      return yield* Effect.tryPromise({
        try: async () => {
          const lazy = await handle.encodeBatchTensor(
            texts as Array<string>,
            toNativePadding(config.padding),
            toNativeTruncation(config.truncation),
            device
          )
          return Tensor.makeLazy(lazy, lazy.shape, "u32", device)
        },
        catch: toTokenizerError("encodeBatch")
      })
    })
  self.decode = (ids: Tensor.Any | ReadonlyArray<number>) =>
    Effect.flatMap(idsOf(ids), (resolved) =>
      Effect.try({
        try: () => handle.decode(resolved as Array<number>),
        catch: toTokenizerError("decode")
      }))
  self.decodeBatch = (batch: ReadonlyArray<Tensor.Any | ReadonlyArray<number>>) =>
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
    try: () => make(NativeTokenizer.fromFile(path, config.specialTokens === "Always"), config),
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
    try: () => make(NativeTokenizer.fromJson(json, config.specialTokens === "Always"), config),
    catch: toTokenizerError("fromJson")
  })

/**
 * Trains a tokenizer from a corpus ({@link TrainSource}): raw text files
 * streamed from disk, or texts already in memory. Runs natively off the
 * JS thread; the result is immediately usable and `save`-able as
 * `tokenizer.json`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const train = (
  trainConfig: TrainConfig,
  config: TokenizerConfig
): Effect.Effect<Tokenizer, TokenizerError> => {
  const progress = trainConfig.progress
  const onProgress = progress._tag === "Report"
    ? (event: [number, number]) => progress.report(event[0], event[1])
    : () => {}
  return Effect.tryPromise({
    try: async () =>
      make(
        await NativeTokenizer.train(
          {
            model: trainConfig.model,
            vocabSize: trainConfig.vocabSize,
            minFrequency: trainConfig.minFrequency,
            specialTokens: trainConfig.specialTokens as Array<string>,
            source: trainConfig.source._tag === "Files"
              ? { tag: "Files", paths: trainConfig.source.paths as Array<string> }
              : { tag: "Texts", texts: trainConfig.source.texts as Array<string> }
          },
          config.specialTokens === "Always",
          onProgress
        ),
        config
      ),
    catch: toTokenizerError("train")
  })
}

/**
 * Returns `true` if the value is a {@link Tokenizer}.
 *
 * @since 0.1.0
 * @category guards
 */
export const isTokenizer = (value: unknown): value is Tokenizer =>
  typeof value === "object" && value !== null && TokenizerTypeId in value

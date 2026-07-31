# RFC 0009: Tokenizers — Native Text-to-Tensor Data Plane

- **Status**: Implemented
- **Author**: Michael Arnaldi
- **Date**: 2026-07-31
- **Depends on**: nothing architecturally; consumes the `Tensor` module
  (constructor leaves, `CurrentDevice`)

## Summary

Tokenization is the entry point of the text data plane: corpora are
tokenized once at dataset scale (GBs–TBs) and encode/decode happens around
every training and generation loop. This RFC adds a `Tokenizer` module to
`@effect-torch/core` with two halves:

1. **Native hot path.** The HuggingFace `tokenizers` crate (the Rust core
   behind every "fast" tokenizer on the HF Hub) becomes a dependency of
   `packages/native`, exposed over napi as a `NativeTokenizer` handle. All
   four standard models — BPE, Unigram, WordPiece, WordLevel — plus their
   trainers, normalizers, pre-tokenizers, post-processors and decoders are
   available, `tokenizer.json`-compatible, and parallel in batch encoding.
2. **Effect-native façade.** A `Tokenizer` value in `packages/core` whose
   `encode`/`encodeBatch` return **tensors, not id arrays** — `[T]` and
   padded `[B, T]` `u32` tensors built natively in the same call that
   tokenizes, so the id buffer never round-trips through JS. Constructors
   (`fromFile`, `fromJson`, `train`) return Effects, keeping the door open
   to a future tokenizer service in the environment.

## Motivation

**Throughput is a first-class requirement.** Reference points: SentencePiece
(C++) benches 7–127 MB/s depending on model and thread count, tiktoken
(Rust) is 3–6× faster than HF's Rust core on GPT-2 BPE, and single-threaded
pure-JS BPE sits around ~1 MB/s. Over a 10GB corpus that is the difference
between minutes and hours. A library for building models cannot ship the
slow option as its core; tokenization throughput gates data loading for
every text model.

**Tensors are the only currency.** Every consumer of tokenized text in this
library — embedding lookups, `Trainer` data functions, generation loops —
consumes tensors. Returning `number[]` would push a `Tensor.fromTypedArray`
onto every call site and copy the id buffer across the napi boundary twice
(native → JS array → native constructor leaf). Returning tensors directly is
one napi call per encode with zero JS element processing.

**Pretrained compatibility matters.** `tokenizer.json` is the self-contained
format every HF Hub tokenizer ships (GPT-2, LLaMA, Gemma, Mistral, Qwen).
Adopting the crate gives loading parity for free, instead of re-deriving
years of unicode-offset, byte-fallback and special-token correctness in TS.

## Prior art

- **HuggingFace `tokenizers`** (Rust, MIT): the pipeline decomposition —
  Normalizer → PreTokenizer → Model → PostProcessor, with a Decoder for the
  inverse — and the four model families. We adopt the crate itself for the
  native half and its vocabulary in our docs; we do *not* re-expose its
  full component zoo as TS types in v1 (see Non-goals).
- **tiktoken** (Rust core): byte-level BPE with regex pre-tokenization; the
  reference for throughput. Its `allowed_special` discipline — special
  tokens in input text are *never* parsed unless the caller explicitly opts
  in — is adopted verbatim: silently tokenizing attacker-controlled text
  into special tokens is a known footgun.
- **SentencePiece** (C++): language-agnostic, lossless by construction via
  the `▁` metaspace convention; covered by the crate's Metaspace
  pre-tokenizer/decoder and Unigram/BPE models. Its `.model` protobuf
  format is not loadable by the crate (see Non-goals).
- **minbpe** (Karpathy): the reference for the BPE training loop (word
  frequencies → count pairs → merge argmax → repeat) and the id allocation
  convention: byte tokens first, then merges in rank order, then special
  tokens appended after the last merge id.

## Design

### Native: `NativeTokenizer`

`packages/native` gains a `tokenizers` crate dependency and one napi class:

```rust
NativeTokenizer {
  // construction
  from_file(path: String) -> NativeTokenizer
  from_json(json: String) -> NativeTokenizer
  train(config: TrainConfig, files: Vec<String>) -> NativeTokenizer

  // tensor-producing encode (single napi call per encode)
  encode_tensor(text: String, device: DeviceKind) -> LazyTensor          // [T] u32
  encode_batch_tensor(texts: Vec<String>, padding: Padding,
                      truncation: Truncation, device: DeviceKind)
      -> LazyTensor                                                       // [B, T] u32

  // raw-id variants for decode-side utilities and tests
  encode(text: String) -> Uint32Array
  encode_batch(texts: Vec<String>) -> Vec<Uint32Array>
  decode(ids: Vec<u32>) -> String
  decode_batch(ids: Vec<Vec<u32>>) -> Vec<String>

  // vocab access & persistence
  token_to_id(token: String) -> Option<u32>
  id_to_token(id: u32) -> Option<String>
  vocab_size: u32  (getter)
  save(path: String)
}
```

- `encode_tensor` / `encode_batch_tensor` build the `u32` constructor leaf
  **in Rust** from the encoded buffer and return the ordinary `LazyTensor`
  napi handle — identical to what `Tensor.fromTypedArray` would produce,
  minus the JS round-trip. Buffers are handed to the leaf by move, not copy.
- `encode_batch_tensor` uses the crate's rayon-parallel batch encoder;
  padding and truncation are applied natively per call from the explicit
  config, never by mutating tokenizer state (the crate's
  `with_padding`/`with_truncation` setters are not used), so a handle stays
  immutable and shareable.
- Single-string `encode_tensor` may be synchronous napi (encoding is
  microseconds); `encode_batch_tensor` and `train` are async napi like
  other long-running native work.

### Core: `Tokenizer` module

`packages/core/src/Tokenizer.ts`, following module conventions (data-first,
brand type id, `Pipeable`, `Data.TaggedError`, no classes):

```ts
export class TokenizerError extends Data.TaggedError("TokenizerError")<{
  readonly op: string
  readonly message: string
}> {}

export type Padding =
  | { readonly _tag: "None" }                 // encodeBatch fails on ragged input
  | { readonly _tag: "Longest"; readonly padId: number }
  | { readonly _tag: "MaxLength"; readonly maxLength: number; readonly padId: number }

export type Truncation =
  | { readonly _tag: "None" }
  | { readonly _tag: "MaxLength"; readonly maxLength: number }

export interface TokenizerConfig {
  readonly padding: Padding
  readonly truncation: Truncation
  readonly specialTokens: "Never" | "Always"  // parse special-token strings in input text
}

export interface Tokenizer extends Pipeable {
  readonly [TokenizerTypeId]: TokenizerTypeId
  readonly vocabSize: number
  readonly encode: (text: string) => Effect<Tensor.Lazy, TokenizerError, CurrentDevice>
  readonly encodeBatch: (
    texts: ReadonlyArray<string>
  ) => Effect<Tensor.Lazy, TokenizerError, CurrentDevice>
  readonly decode: (
    ids: Tensor.Any | ReadonlyArray<number>
  ) => Effect<string, TokenizerError>
  readonly decodeBatch: (
    ids: ReadonlyArray<Tensor.Any | ReadonlyArray<number>>
  ) => Effect<ReadonlyArray<string>, TokenizerError>
  readonly tokenToId: (token: string) => Option<number>
  readonly idToToken: (id: number) => Option<string>
  readonly save: (path: string) => Effect<void, TokenizerError>
}
```

- **Constructors return Effects** (`fromFile(path, config)`,
  `fromJson(json, config)`, `train(trainConfig, config)`), per the
  backend-service rule: a remote tokenizer service can later satisfy the
  same interface.
- **No explicit disposal.** The native handle owns only CPU heap — vocab
  tables, merge lists, regexes — with no device buffers, file handles or
  threads, so it is reclaimed by ordinary napi finalization when the JS
  wrapper is garbage-collected. This deliberately differs from
  `CompiledProgram` (RFC 0008), whose `dispose` exists because it
  references device memory the GC cannot account for; pushing `Scope`
  onto every tokenizer constructor would tax every call site for a
  resource that is not scarce.
- **Config is explicit and total** — `padding`, `truncation` and
  `specialTokens` are required fields with explicit `None`/`Never`
  variants. No optional-parameter defaults that silently pad with id 0 or
  silently swallow `<|endoftext|>` appearing in user text.
- **Ragged batches are an error**, not an implicit decision: with
  `padding: { _tag: "None" }`, `encodeBatch` on unequal-length encodings
  fails with `TokenizerError`.
- **`decode` accepts tensors or raw ids**: generation loops hold tensors;
  tests and tooling hold arrays. Tensor inputs are materialized natively.

### Training

```ts
export type TrainModel = "BPE" | "WordPiece" | "Unigram" | "WordLevel"

export type TrainSource =
  | { readonly _tag: "Files"; readonly paths: ReadonlyArray<string> }
  | { readonly _tag: "Texts"; readonly texts: ReadonlyArray<string> }

export type TrainProgress<E, R> =
  | { readonly _tag: "None" }
  | {
    readonly _tag: "Report"
    readonly everyBytes: number
    readonly report: (processed: number, total: number) => Effect<void, E, R>
  }

export interface TrainConfig {
  readonly source: TrainSource
  readonly model: TrainModel
  readonly vocabSize: number
  readonly minFrequency: number
  readonly specialTokens: ReadonlyArray<string>
  readonly progress: TrainProgress
}
```

The corpus comes from raw text `Files` streamed line-by-line from disk
(dataset scale — feeding GBs never loads them) or `Texts` already in
memory (small corpora, generated data) — an explicit union, never a file
written just to satisfy the API.

Pipeline defaults follow the canonical setups: BPE trains byte-level
(GPT-2 style: no prefix space, regex splitting, full 256-byte alphabet
seeded), WordPiece with the BERT normalizer and `##` continuations,
Unigram with the SentencePiece `▁` metaspace convention, WordLevel on
whitespace-split words. Training is deterministic. Id allocation follows
the `tokenizers` crate convention: special tokens first, then the
alphabet, then merges or pieces in rank order; trained special tokens
are registered as added tokens so they persist in `tokenizer.json` and
are visible to the special-token policy. The resulting tokenizer is
immediately usable and `save`-able as `tokenizer.json`.

**Progress reporting.** The crate's built-in indicatif progress bars are
disabled (`show_progress(false)`) — a library never prints unprompted.
Instead the corpus iterator is instrumented: the crate consumes sequences
through an iterator owned by the native layer, which counts corpus bytes
as they are pulled and forwards `(processed, total)` reports to JS over a
napi `ThreadsafeFunction`, throttled by the caller's `everyBytes`
interval (at most one report per `everyBytes` corpus bytes consumed;
`0` disables reporting, including the completion event).
Totals are byte-exact (`Texts`: summed lengths; `Files`: summed file
metadata). The feed phase is the dominant cost at dataset scale; the
merge phase afterwards is a crate black box, so the protocol is
determinate feed progress followed by one final `(total, total)` report
when the iterator exhausts, then indeterminate until `train` resolves.
Report handlers are Effect-returning: `train` runs them in order through
a `Stream.callback` queue and yields the tokenizer as the stream's last
element, so logging, rendering or ignoring events is an ordinary Effect
composition.

### Tensor integration

- Token id dtype is **`u32`** (vocabs are ≤ a few hundred thousand).
  Implementation note: embedding/gather-style ops that currently expect
  `i64` indices must either accept `u32` or the module casts at the
  boundary; extending the ops is preferred (index dtypes are not semantic).
- A `[B, T]` padded batch plus `padId` gives attention masks by tensor ops
  (`notEqual(ids, padId)`); v1 does not return mask metadata from the
  tokenizer itself.
- `Tokenizer` never imports `Model`/`Trainer`; the dependency is one-way
  (examples compose them).

## Non-goals

- **Pure-TS reimplementation** of any algorithm (rejected: ~1 MB/s is not a
  data plane).
- **SentencePiece `.model` protobuf loading** — the crate doesn't read it;
  conversion to `tokenizer.json` happens via existing HF tooling.
- **tiktoken `.tiktoken` rank-file loading** — can be added later as a
  converter; GPT-2/4 vocabs are already reachable via `tokenizer.json`.
- **Full `Encoding` metadata** (offset mappings, word ids, type ids,
  attention masks) — v1 returns id tensors only.
- **Subword regularization** (BPE-dropout, unigram sampling) — a training
  regularization feature, deferred.
- **Re-exposing the pipeline component zoo in TS** — composition happens in
  `tokenizer.json` / trainer config; TS types cover only what the
  `Tokenizer` interface needs.
- **Tokenizer-free architectures** (byte-latent models) — out of scope.

## Alternatives considered

- **Hand-rolled Rust BPE/Unigram/WordPiece**: full control, but re-derives
  years of unicode-offset, byte-fallback and special-token correctness that
  the crate already encodes and tests. The crate is MIT-licensed and *is*
  the status quo; there is no architectural gain in duplicating it.
- **Wrapping existing npm bindings** (`tokenizers` node package, tiktoken
  WASM): external binary dependency, WASM is slower than native, and the
  tensor hand-off would still cross JS — losing the single-call property.
- **Returning raw id arrays**: rejected in favour of tensors (double napi
  crossing, per-caller `fromTypedArray` boilerplate, breaks
  tensors-as-currency).

## Acceptance criteria

1. Load `gpt2` `tokenizer.json`; `encode` output matches published
   reference token ids on a fixture corpus covering ASCII, accents, CJK,
   emoji, and mixed scripts; `decode` round-trips losslessly.
2. Train BPE on tiny-shakespeare to an 8k vocab via `Tokenizer.train`;
   round-trip is lossless; `save` → `fromFile` preserves behaviour exactly.
3. All four models train and round-trip on small fixture corpora.
4. `encodeBatch` with `Longest`/`MaxLength` padding and `MaxLength`
   truncation produces correctly shaped `[B, T]` `u32` tensors; `None`
   fails on ragged input; `specialTokens: "Never"` treats special-token
   strings as ordinary text.
5. Throughput benchmark: batch encoding of a 100MB corpus ≥ 50× a pure-TS
   baseline.
6. `packages/examples/nano-gpt.ts` switches from its ad-hoc char-level
   maps to a trained BPE tokenizer end-to-end (train, sample, generate).
7. Full typecheck + test suite green; native bindings regenerated.

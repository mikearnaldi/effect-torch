//! Native tokenizer implementation exposed to JavaScript over NAPI.
//!
//! # Special-token segmentation
//!
//! [`NativeTokenizer::from_file`] and [`NativeTokenizer::from_json`] take a
//! `parse_specials` flag. When it is `false`, encoding follows the tiktoken
//! `allowed_special = "none"` discipline: raw text must never resolve to a
//! special-token id. [`split_around_specials`] splits the input at every
//! occurrence of a special-token string (longest match first), each segment
//! is encoded separately with added-token matching disabled, and the piece
//! encodings are merged. A segment that *is* exactly a special string is
//! recursively split in half until no piece is itself special, defeating
//! whole-word vocabulary lookups that would otherwise return the special
//! id (see [`TokenizerInner::encode_segment`]). When `parse_specials` is
//! `true` (or the tokenizer defines no special tokens), the input is
//! passed to the underlying tokenizer untouched and specials parse
//! normally.
//!
//! # Post-processor behavior
//!
//! In the segmented path the model pipeline runs per segment with
//! `add_special_tokens = false`, so template post-processors (e.g. BERT's
//! `[CLS]`/`[SEP]` insertion) do not fire per segment. Instead, after all
//! segments are merged, the tokenizer's post-processor — if any — is
//! applied exactly once to the merged encoding when the caller requested
//! `add_special_tokens`. This mirrors encoding the whole text at once.
//!
//! # Chat-template rendering
//!
//! [`NativeTokenizer::apply_chat_template`] renders a Hugging Face chat
//! template with `minijinja`. The environment registers two helpers
//! compatible with the Jinja environment used by `transformers`:
//!
//! - `raise_exception(message)` — aborts rendering with `message` as the
//!   error, letting templates fail loudly on unsupported inputs.
//! - `strftime_now(format)` — formats the current UTC time. Supported
//!   conversion codes are `%Y` (4-digit year), `%m` (month), `%d` (day),
//!   `%H` (hour), `%M` (minute), `%S` (second), and `%%` (a literal `%`);
//!   any other code is an error. The calendar conversion is the
//!   civil-from-days algorithm ([`civil_from_days`]); no external date
//!   library is involved.
//!
//! The template context is supplied as a JSON string and deserialized to
//! `serde_json::Value` before rendering.
//!
//! # Training: progress, cancellation, and threading
//!
//! [`NativeTokenizer::train`] runs the whole training job inside
//! `tokio::task::spawn_blocking`, so the CPU-bound feed and merge phases
//! execute on the blocking thread pool and never stall the Node.js event
//! loop. The returned promise resolves when training completes.
//!
//! The `tokenizers` crate exposes no progress hook, so progress is
//! measured at the corpus feed: [`corpus_iter`] streams sequences from
//! files (line by line, with byte totals from file metadata) or from
//! in-memory texts, and [`ProgressFeed`] counts corpus bytes as the
//! trainer pulls them, invoking the JS callback with
//! `(processedBytes, totalBytes)` every `progressEveryBytes` bytes. A
//! final `(total, total)` report pins completion when the feed is
//! exhausted; the subsequent merge phase reports nothing. Passing
//! `progressEveryBytes = 0` disables reporting entirely. The callback is a
//! [`ThreadsafeFunction`] invoked in `NonBlocking` mode: calls are
//! enqueued to the JS thread without awaiting execution, and failures to
//! enqueue (e.g. after the JS side has torn down) are ignored.
//!
//! There is **no cancellation**: once started, training runs to completion
//! (or failure) on the blocking thread; dropping the promise on the JS
//! side does not stop the underlying work.
//!
//! # NAPI ownership
//!
//! [`NativeTokenizer`] holds its state behind an [`Arc`]<[`TokenizerInner`]>
//! containing only CPU-heap data (vocabulary tables, merge rules, regexes).
//! The `Arc` is cloned into `spawn_blocking` closures for batch encoding,
//! keeping the tokenizer alive for the duration of background work while
//! allowing concurrent use from JS. The native object owns no device
//! buffers or file handles, so there is no explicit `dispose` — memory is
//! reclaimed by NAPI finalization when the JS wrapper is garbage
//! collected.
//!
//! This module contains no `unsafe` code.

use minijinja::{Environment, Error as MiniError, ErrorKind as MiniErrorKind, Value as MiniValue};
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use rayon::prelude::*;
use std::io::BufRead;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::decoders::{
    byte_fallback::ByteFallback as ByteFallbackDecoder, byte_level::ByteLevel as ByteLevelDecoder,
    metaspace::Metaspace as MetaspaceDecoder, sequence::Sequence as SequenceDecoder,
    wordpiece::WordPiece as WordPieceDecoder,
};
use tokenizers::models::TrainerWrapper;
use tokenizers::models::{
    bpe::BPE, unigram::Unigram, wordlevel::WordLevel, wordpiece::WordPiece, ModelWrapper,
};
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::{
    byte_level::ByteLevel, metaspace::Metaspace, whitespace::Whitespace,
};
use tokenizers::{AddedToken, Encoding, PostProcessor, Tokenizer};

/// Training corpus selector for [`NativeTokenizer::train`].
///
/// Exactly one of `paths`/`texts` is expected, chosen by `tag`:
/// `tag = "Files"` streams the files at `paths` line by line, while
/// `tag = "Texts"` trains from the in-memory strings in `texts`.
#[napi(object)]
pub struct NativeTrainSource {
    /// Corpus kind: `"Files"` or `"Texts"`.
    pub tag: String,
    /// File paths to stream when `tag` is `"Files"`.
    pub paths: Option<Vec<String>>,
    /// In-memory sequences to train on when `tag` is `"Texts"`.
    pub texts: Option<Vec<String>>,
}

/// Configuration for [`NativeTokenizer::train`].
#[napi(object)]
pub struct NativeTrainConfig {
    /// Model architecture: `"BPE"`, `"WordPiece"`, `"Unigram"`, or
    /// `"WordLevel"`. Each architecture pairs the model with a fixed
    /// normalizer/pre-tokenizer/decoder pipeline (see `train_tokenizer`).
    pub model: String,
    /// Target vocabulary size, including special tokens.
    pub vocab_size: u32,
    /// Minimum corpus frequency for a token to enter the vocabulary.
    pub min_frequency: u32,
    /// Special tokens to register (e.g. `"<|endoftext|>"`). They are added
    /// to the vocabulary as special added tokens after training.
    pub special_tokens: Vec<String>,
    /// Training corpus.
    pub source: NativeTrainSource,
}

fn to_napi_error<E: std::fmt::Display>(err: E) -> Error {
    Error::new(Status::GenericFailure, err.to_string())
}

fn to_join_error(error: tokio::task::JoinError) -> Error {
    Error::new(Status::GenericFailure, error.to_string())
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097) as u64;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era as i64 + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * month_prime + 2) / 5 + 1) as u32;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

type MiniResult<T> = std::result::Result<T, MiniError>;

fn strftime_now(format: &str) -> MiniResult<String> {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| MiniError::new(MiniErrorKind::InvalidOperation, error.to_string()))?
        .as_secs() as i64;
    let days = seconds.div_euclid(86_400);
    let time = seconds.rem_euclid(86_400) as u32;
    let (year, month, day) = civil_from_days(days);
    let replacements = [
        ("Y", format!("{year:04}")),
        ("m", format!("{month:02}")),
        ("d", format!("{day:02}")),
        ("H", format!("{:02}", time / 3_600)),
        ("M", format!("{:02}", time % 3_600 / 60)),
        ("S", format!("{:02}", time % 60)),
        ("%", "%".to_string()),
    ];
    let mut rendered = String::new();
    let mut chars = format.chars();
    while let Some(char) = chars.next() {
        if char != '%' {
            rendered.push(char);
            continue;
        }
        let code = chars.next().ok_or_else(|| {
            MiniError::new(
                MiniErrorKind::InvalidOperation,
                "strftime_now format ends after '%'",
            )
        })?;
        let replacement = replacements
            .iter()
            .find(|(candidate, _)| *candidate == code.to_string())
            .map(|(_, replacement)| replacement)
            .ok_or_else(|| {
                MiniError::new(
                    MiniErrorKind::InvalidOperation,
                    format!("unsupported strftime_now format %{code}"),
                )
            })?;
        rendered.push_str(replacement);
    }
    Ok(rendered)
}

/// Renders `template` against the JSON context `context_json` using the
/// `minijinja` environment described in the module docs (with the
/// `raise_exception` and `strftime_now` helpers registered).
fn render_chat_template(template: &str, context_json: &str) -> Result<String> {
    let context: serde_json::Value = serde_json::from_str(context_json).map_err(to_napi_error)?;
    let mut environment = Environment::new();
    environment.add_function(
        "raise_exception",
        |message: String| -> MiniResult<MiniValue> {
            Err(MiniError::new(MiniErrorKind::InvalidOperation, message))
        },
    );
    environment.add_function("strftime_now", |format: String| strftime_now(&format));
    environment
        .template_from_str(template)
        .and_then(|template| template.render(&MiniValue::from_serialize(&context)))
        .map_err(to_napi_error)
}

// Splits `text` at occurrences of special-token strings, keeping every piece
// (the special occurrences included) as its own segment. Encoding each
// segment separately through the model then guarantees that parsing raw text
// can never produce a special-token id — the tiktoken `allowed_special`
// discipline — while the special strings still tokenize as ordinary text.
fn split_around_specials<'a>(text: &'a str, specials: &[String]) -> Vec<&'a str> {
    if specials.is_empty() {
        return vec![text];
    }
    let mut segments = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < text.len() {
        if text.is_char_boundary(i) {
            if let Some(special) = specials
                .iter()
                .find(|sp| text[i..].starts_with(sp.as_str()))
            {
                if start < i {
                    segments.push(&text[start..i]);
                }
                segments.push(&text[i..i + special.len()]);
                i += special.len();
                start = i;
                continue;
            }
        }
        i += 1;
    }
    if start < text.len() {
        segments.push(&text[start..]);
    }
    segments
}

struct TokenizerInner {
    tokenizer: Tokenizer,
    parse_specials: bool,
    specials: Vec<String>,
}

impl TokenizerInner {
    fn new(tokenizer: Tokenizer, parse_specials: bool) -> Self {
        let mut specials: Vec<String> = tokenizer
            .get_added_tokens_decoder()
            .values()
            .filter(|token| token.special)
            .map(|token| token.content.clone())
            .collect();
        // Longest first so overlapping specials match greedily.
        specials.sort_by(|a, b| b.len().cmp(&a.len()));
        Self {
            tokenizer,
            parse_specials,
            specials,
        }
    }

    // Encodes one segment in the "never parse specials" path. A segment that
    // is exactly a special-token string must tokenize as ordinary text, but
    // the model's whole-word vocabulary lookup would resolve it to the
    // special id directly — so split it until no piece is itself special.
    fn encode_segment(&self, segment: &str) -> Result<Encoding> {
        if segment.chars().count() > 1 && self.specials.iter().any(|sp| sp == segment) {
            let mut mid = segment.len() / 2;
            while !segment.is_char_boundary(mid) {
                mid -= 1;
            }
            if mid == 0 {
                mid = segment
                    .char_indices()
                    .nth(1)
                    .map(|(i, _)| i)
                    .unwrap_or(segment.len());
            }
            if mid < segment.len() {
                let mut merged = self.encode_segment(&segment[..mid])?;
                merged.merge_with(self.encode_segment(&segment[mid..])?, false);
                return Ok(merged);
            }
        }
        self.tokenizer.encode(segment, false).map_err(to_napi_error)
    }

    fn encode_ids_with(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        if self.parse_specials || self.specials.is_empty() {
            let encoding = self
                .tokenizer
                .encode(text, add_special_tokens)
                .map_err(to_napi_error)?;
            return Ok(encoding.get_ids().to_vec());
        }
        let segments = split_around_specials(text, &self.specials);
        let mut merged = Encoding::default();
        for segment in segments {
            if segment.is_empty() {
                continue;
            }
            merged.merge_with(self.encode_segment(segment)?, false);
        }
        if add_special_tokens {
            if let Some(post_processor) = self.tokenizer.get_post_processor() {
                merged = post_processor
                    .process(merged, None, true)
                    .map_err(to_napi_error)?;
            }
        }
        Ok(merged.get_ids().to_vec())
    }

    fn encode_ids(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_ids_with(text, true)
    }
}

/// A tokenizer instance exposed to JavaScript.
///
/// Wraps a [`tokenizers::Tokenizer`] plus its special-token policy in an
/// [`Arc`] so background batch work can hold a clone while the JS object
/// remains usable. The instance is immutable after construction: encoding
/// and decoding are safe to call concurrently, and there is no `dispose` —
/// the memory (CPU heap only) is reclaimed by NAPI finalization when the
/// JS wrapper is garbage-collected.
#[napi]
pub struct NativeTokenizer {
    // CPU-heap only (vocab tables, merges, regexes): reclaimed by napi
    // finalization when the JS wrapper is garbage-collected, so there is no
    // explicit dispose — unlike device-buffer handles.
    inner: Arc<TokenizerInner>,
}

#[napi]
impl NativeTokenizer {
    fn inner(&self) -> &Arc<TokenizerInner> {
        &self.inner
    }

    /// Loads a tokenizer from a JSON file on disk (the Hugging Face
    /// `tokenizer.json` format).
    ///
    /// `parse_specials` selects the special-token policy described in the
    /// module docs: `true` parses special-token strings in the input to
    /// their ids; `false` guarantees raw text never produces a special id.
    #[napi(factory)]
    pub fn from_file(path: String, parse_specials: bool) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(to_napi_error)?;
        Ok(Self {
            inner: Arc::new(TokenizerInner::new(tokenizer, parse_specials)),
        })
    }

    /// Loads a tokenizer from a JSON string (the Hugging Face
    /// `tokenizer.json` format). Same `parse_specials` semantics as
    /// [`NativeTokenizer::from_file`].
    #[napi(factory)]
    pub fn from_json(json: String, parse_specials: bool) -> Result<Self> {
        let tokenizer = Tokenizer::from_bytes(json.as_bytes()).map_err(to_napi_error)?;
        Ok(Self {
            inner: Arc::new(TokenizerInner::new(tokenizer, parse_specials)),
        })
    }

    /// Trains a new tokenizer according to `config`.
    ///
    /// The job runs on the tokio blocking thread pool; see the module docs
    /// for the progress reporting contract (`progress` receives
    /// `[processedBytes, totalBytes]` every `progressEveryBytes` bytes,
    /// with a final `[total, total]` on feed completion; `0` disables
    /// reports), the absence of cancellation, and the per-architecture
    /// pipeline setup. `parse_specials` has the same meaning as in
    /// [`NativeTokenizer::from_file`].
    #[napi(
        factory,
        ts_args_type = "config: NativeTrainConfig, parseSpecials: boolean, progress: (event: [number, number]) => void, progressEveryBytes: number"
    )]
    pub async fn train(
        config: NativeTrainConfig,
        parse_specials: bool,
        progress: ProgressCallback,
        progress_every_bytes: f64,
    ) -> Result<Self> {
        let tokenizer = tokio::task::spawn_blocking(move || {
            train_tokenizer(config, Some(progress), progress_every_bytes as u64)
        })
        .await
        .map_err(to_join_error)??;
        Ok(Self {
            inner: Arc::new(TokenizerInner::new(tokenizer, parse_specials)),
        })
    }

    /// Total vocabulary size, including added/special tokens.
    #[napi(getter)]
    pub fn vocab_size(&self) -> Result<u32> {
        Ok(self.inner().tokenizer.get_vocab_size(true) as u32)
    }

    /// Resolves a token string to its id, or `null` if it is not in the
    /// vocabulary.
    #[napi]
    pub fn token_to_id(&self, token: String) -> Result<Option<u32>> {
        Ok(self.inner().tokenizer.token_to_id(&token))
    }

    /// Resolves a token id to its string, or `null` if the id is out of
    /// range.
    #[napi]
    pub fn id_to_token(&self, id: u32) -> Result<Option<String>> {
        Ok(self.inner().tokenizer.id_to_token(id))
    }

    /// Serializes the tokenizer to `path` in the Hugging Face
    /// `tokenizer.json` format (without pretty-printing).
    #[napi]
    pub fn save(&self, path: String) -> Result<()> {
        self.inner()
            .tokenizer
            .save(path, false)
            .map_err(to_napi_error)
    }

    /// Encodes `text` to token ids.
    ///
    /// `add_special_tokens` (default `true`) controls whether the
    /// tokenizer's post-processor template tokens (e.g. `[CLS]`/`[SEP]`)
    /// are added. The special-token segmentation behavior is governed by
    /// the `parse_specials` flag chosen at construction, not by this flag.
    #[napi]
    pub fn encode(&self, text: String, add_special_tokens: Option<bool>) -> Result<Uint32Array> {
        Ok(self
            .inner()
            .encode_ids_with(&text, add_special_tokens.unwrap_or(true))?
            .into())
    }

    /// Encodes a batch of texts, always adding special tokens.
    ///
    /// Runs on the tokio blocking thread pool and parallelizes across
    /// texts with rayon; the input order is preserved in the output.
    #[napi]
    pub async fn encode_batch(&self, texts: Vec<String>) -> Result<Vec<Uint32Array>> {
        let inner = self.inner().clone();
        tokio::task::spawn_blocking(move || {
            texts
                .par_iter()
                .map(|text| inner.encode_ids(text).map(Uint32Array::from))
                .collect::<Result<Vec<_>>>()
        })
        .await
        .map_err(to_join_error)?
    }

    /// Decodes token ids back to text. `skip_special_tokens` (default
    /// `false`) drops special tokens from the output instead of rendering
    /// their string form.
    #[napi]
    pub fn decode(&self, ids: Vec<u32>, skip_special_tokens: Option<bool>) -> Result<String> {
        self.inner()
            .tokenizer
            .decode(&ids, skip_special_tokens.unwrap_or(false))
            .map_err(to_napi_error)
    }

    /// Decodes a batch of id sequences; same `skip_special_tokens`
    /// semantics as [`NativeTokenizer::decode`].
    #[napi]
    pub fn decode_batch(
        &self,
        ids: Vec<Vec<u32>>,
        skip_special_tokens: Option<bool>,
    ) -> Result<Vec<String>> {
        let refs: Vec<&[u32]> = ids.iter().map(|v| v.as_slice()).collect();
        self.inner()
            .tokenizer
            .decode_batch(&refs, skip_special_tokens.unwrap_or(false))
            .map_err(to_napi_error)
    }

    /// Renders a Hugging Face chat template against a JSON context. See
    /// the module docs for the rendering environment and the supported
    /// `raise_exception`/`strftime_now` helpers.
    #[napi]
    pub fn apply_chat_template(&self, template: String, context_json: String) -> Result<String> {
        render_chat_template(&template, &context_json)
    }
}

fn added_specials(special_tokens: &[String]) -> Vec<AddedToken> {
    special_tokens
        .iter()
        .map(|content| AddedToken::from(content.clone(), true))
        .collect()
}

type ProgressCallback = ThreadsafeFunction<(f64, f64), Unknown<'static>, (f64, f64), Status, false>;

// The crate offers no progress hook (its indicatif bars are internal and
// disabled above), but it consumes the corpus through an iterator we own:
// count corpus bytes as they are pulled and forward (processed, total) to
// JS. The feed phase is the dominant cost on large corpora; the merge phase
// afterwards is a black box, so completion of the feed is signalled by one
// final (total, total) report when the iterator is exhausted.
struct ProgressFeed<I> {
    inner: I,
    processed: u64,
    total: u64,
    step: u64,
    report: Option<ProgressCallback>,
    last_reported: u64,
    finished: bool,
}

impl<I: Iterator<Item = String>> Iterator for ProgressFeed<I> {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        match self.inner.next() {
            Some(item) => {
                self.processed += item.len() as u64;
                if let Some(callback) = &self.report {
                    // step 0 disables reporting entirely.
                    if self.step > 0 && self.processed - self.last_reported >= self.step {
                        self.last_reported = self.processed;
                        let _ = callback.call(
                            (self.processed as f64, self.total as f64),
                            ThreadsafeFunctionCallMode::NonBlocking,
                        );
                    }
                }
                Some(item)
            }
            None => {
                if !self.finished {
                    self.finished = true;
                    // Completion pins (total, total): Texts reach it exactly
                    // (skip if already reported), Files undershoot by the
                    // stripped newlines and need the pin.
                    if self.step > 0 && self.last_reported != self.total {
                        if let Some(callback) = &self.report {
                            let _ = callback.call(
                                (self.total as f64, self.total as f64),
                                ThreadsafeFunctionCallMode::NonBlocking,
                            );
                        }
                    }
                }
                None
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

// Streams the corpus as an iterator of sequences with a byte-exact total:
// Texts from memory, Files line-by-line (so feeding GBs does not load them)
// with totals from file metadata. Both sources share one training path.
fn corpus_iter(
    source: NativeTrainSource,
    report: Option<ProgressCallback>,
    step: u64,
) -> Result<ProgressFeed<Box<dyn Iterator<Item = String> + Send>>> {
    let (inner, total): (Box<dyn Iterator<Item = String> + Send>, u64) = match source.tag.as_str() {
        "Files" => {
            let paths = source.paths.ok_or_else(|| {
                Error::new(
                    Status::InvalidArg,
                    "train: Files source requires paths".to_string(),
                )
            })?;
            let mut readers = Vec::with_capacity(paths.len());
            let mut total = 0u64;
            for path in &paths {
                total += std::fs::metadata(path)
                    .map_err(|e| Error::new(Status::GenericFailure, format!("train: {path}: {e}")))?
                    .len();
                readers.push(std::io::BufReader::new(std::fs::File::open(path).map_err(
                    |e| Error::new(Status::GenericFailure, format!("train: {path}: {e}")),
                )?));
            }
            // Line lengths exclude the stripped newline; +1 per line
            // approximates raw bytes and the completion event pins total.
            let lines = readers
                .into_iter()
                .flat_map(|reader| reader.lines().map_while(|line| line.ok()))
                .map(|line| line);
            (Box::new(lines), total)
        }
        "Texts" => {
            let texts = source.texts.ok_or_else(|| {
                Error::new(
                    Status::InvalidArg,
                    "train: Texts source requires texts".to_string(),
                )
            })?;
            let total = texts.iter().map(|text| text.len() as u64).sum();
            (Box::new(texts.into_iter()), total)
        }
        tag => {
            return Err(Error::new(
                Status::InvalidArg,
                format!("train: unknown source tag {tag}"),
            ))
        }
    };
    Ok(ProgressFeed {
        inner,
        processed: 0,
        total,
        step,
        report,
        last_reported: 0,
        finished: false,
    })
}

fn run_trainer(
    tokenizer: &mut Tokenizer,
    trainer: &mut TrainerWrapper,
    source: NativeTrainSource,
    report: Option<ProgressCallback>,
    step: u64,
) -> Result<()> {
    let feed = corpus_iter(source, report, step)?;
    tokenizer.train(trainer, feed).map_err(to_napi_error)?;
    Ok(())
}

// Rebuilds a trained Unigram model with byte_fallback enabled: the 256
// `<0xXX>` byte pieces are appended to the trained vocab (existing ids are
// stable) so text the trained pieces cannot cover — newlines, emoji,
// scripts absent from the corpus — encodes as byte pieces and decodes back
// losslessly, instead of collapsing to <unk>. The trainer's unk token
// stays at id 0 for anything even bytes cannot rescue.
fn enable_byte_fallback(tokenizer: &mut Tokenizer) -> Result<()> {
    let ModelWrapper::Unigram(unigram) = tokenizer.get_model() else {
        return Err(Error::new(
            Status::GenericFailure,
            "train: expected a Unigram model".to_string(),
        ));
    };
    let mut pieces: Vec<(String, f64)> = unigram
        .iter()
        .map(|(token, score)| (token.clone(), *score))
        .collect();
    // Byte pieces get the lowest score: last-resort tokens, never chosen
    // over a trained piece.
    let byte_score = pieces
        .iter()
        .map(|(_, score)| *score)
        .fold(f64::INFINITY, f64::min)
        - 1.0;
    for byte in 0u8..=255 {
        let piece = format!("<0x{byte:02X}>");
        if !pieces.iter().any(|(token, _)| token == &piece) {
            pieces.push((piece, byte_score));
        }
    }
    let model = Unigram::from(pieces, Some(0), true).map_err(to_napi_error)?;
    tokenizer.with_model(model);
    Ok(())
}

fn train_tokenizer(
    config: NativeTrainConfig,
    report: Option<ProgressCallback>,
    progress_step: u64,
) -> Result<Tokenizer> {
    let special_tokens = added_specials(&config.special_tokens);
    let vocab_size = config.vocab_size as usize;
    let min_frequency = config.min_frequency;
    let source = config.source;
    match config.model.as_str() {
        "BPE" => {
            let mut tokenizer = Tokenizer::new(BPE::default());
            // GPT-2 byte-level setup: no prefix space, regex splitting on;
            // the full 256-byte alphabet is seeded so bytes absent from a
            // small corpus still encode (and decode) losslessly.
            tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
            tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
            let mut trainer = TrainerWrapper::from(
                tokenizers::models::bpe::BpeTrainer::builder()
                    .show_progress(false)
                    .vocab_size(vocab_size)
                    .min_frequency(min_frequency as u64)
                    .special_tokens(special_tokens.clone())
                    .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
                    .build(),
            );
            run_trainer(&mut tokenizer, &mut trainer, source, report, progress_step)?;
            tokenizer.add_special_tokens(&special_tokens);
            Ok(tokenizer)
        }
        "WordPiece" => {
            let mut tokenizer = Tokenizer::new(
                WordPiece::builder()
                    .unk_token("[UNK]".to_string())
                    .build()
                    .map_err(to_napi_error)?,
            );
            tokenizer.with_normalizer(Some(BertNormalizer::default()));
            tokenizer.with_pre_tokenizer(Some(Whitespace));
            tokenizer.with_decoder(Some(WordPieceDecoder::default()));
            let mut specials = vec![AddedToken::from("[UNK]".to_string(), true)];
            specials.extend(special_tokens);
            let mut trainer = TrainerWrapper::from(
                tokenizers::models::wordpiece::WordPieceTrainer::builder()
                    .show_progress(false)
                    .vocab_size(vocab_size)
                    .min_frequency(min_frequency as u64)
                    .special_tokens(specials.clone())
                    .build(),
            );
            run_trainer(&mut tokenizer, &mut trainer, source, report, progress_step)?;
            tokenizer.add_special_tokens(&specials);
            Ok(tokenizer)
        }
        "Unigram" => {
            let mut tokenizer = Tokenizer::new(Unigram::default());
            tokenizer.with_pre_tokenizer(Some(Metaspace::default()));
            // LLaMA-style decoding: byte-fallback pieces back to bytes,
            // then ▁ → space.
            tokenizer.with_decoder(Some(SequenceDecoder::new(vec![
                ByteFallbackDecoder::default().into(),
                MetaspaceDecoder::default().into(),
            ])));
            // SentencePiece convention: <unk> at id 0 with model.unk_id set
            // (the builder defaults unk_token to None).
            let mut specials = vec![AddedToken::from("<unk>".to_string(), true)];
            specials.extend(special_tokens);
            let mut trainer = TrainerWrapper::from(
                tokenizers::models::unigram::UnigramTrainer::builder()
                    .show_progress(false)
                    .vocab_size(vocab_size as u32)
                    .special_tokens(specials.clone())
                    .unk_token(Some("<unk>".to_string()))
                    .build()
                    .map_err(to_napi_error)?,
            );
            run_trainer(&mut tokenizer, &mut trainer, source, report, progress_step)?;
            tokenizer.add_special_tokens(&specials);
            enable_byte_fallback(&mut tokenizer)?;
            Ok(tokenizer)
        }
        "WordLevel" => {
            let mut tokenizer = Tokenizer::new(WordLevel::default());
            tokenizer.with_pre_tokenizer(Some(Whitespace));
            let mut trainer = TrainerWrapper::from(
                tokenizers::models::wordlevel::WordLevelTrainer::builder()
                    .show_progress(false)
                    .vocab_size(vocab_size)
                    .min_frequency(min_frequency as u64)
                    .special_tokens(special_tokens.clone())
                    .build()
                    .map_err(to_napi_error)?,
            );
            run_trainer(&mut tokenizer, &mut trainer, source, report, progress_step)?;
            tokenizer.add_special_tokens(&special_tokens);
            Ok(tokenizer)
        }
        model => Err(Error::new(
            Status::InvalidArg,
            format!("train: unknown model {model}"),
        )),
    }
}

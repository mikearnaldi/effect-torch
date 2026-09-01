//! Native tokenizer implementation exposed to JavaScript over NAPI.
//!
//! # Special-token segmentation
//!
//! [`NativeTokenizer::from_file`] and [`NativeTokenizer::from_json`] accept a
//! `parse_specials` flag. When it is `false`, raw text must not resolve to a
//! special-token id, matching tiktoken's `allowed_special = "none"` policy.
//! [`split_around_specials`] matches special-token strings longest first. It
//! encodes each segment without added special tokens, then merges the results.
//! [`TokenizerInner::encode_segment`] splits an exact special-token match until
//! no piece is itself special. This prevents a whole-token vocabulary lookup
//! from returning the special id. When `parse_specials` is `true`, or the
//! tokenizer has no special tokens, the underlying tokenizer receives the
//! input unchanged and parses special tokens normally.
//!
//! # Postprocessor behavior
//!
//! The segmented path encodes each segment with `add_special_tokens = false`,
//! so a template postprocessor does not add tokens such as BERT's `[CLS]` and
//! `[SEP]` to every segment. If the caller requests special tokens, the code
//! applies the tokenizer's postprocessor once after merging all segments.
//!
//! # Chat template rendering
//!
//! [`NativeTokenizer::apply_chat_template`] renders a Hugging Face chat
//! template with `minijinja`. Its environment provides two helpers compatible
//! with the Jinja environment in `transformers`.
//!
//! - `raise_exception(message)` stops rendering and returns `message` as the
//!   error.
//! - `strftime_now(format)` formats the current UTC time. It accepts `%Y`,
//!   `%m`, `%d`, `%H`, `%M`, `%S`, and `%%`. Any other conversion code is an
//!   error. [`civil_from_days`] performs the calendar conversion without an
//!   external date library.
//!
//! The renderer parses the JSON context into `serde_json::Value`.
//!
//! # Training
//!
//! [`NativeTokenizer::train`] runs the feed and merge phases in
//! `tokio::task::spawn_blocking` so they do not block the Node.js event loop.
//! The returned promise resolves when training finishes.
//!
//! The `tokenizers` crate has no progress hook, so progress covers only the
//! corpus feed. [`corpus_iter`] streams files line by line or reads in-memory
//! texts. File totals come from metadata. [`ProgressFeed`] calls JavaScript
//! with `(processedBytes, totalBytes)` after each `progressEveryBytes` bytes.
//! It reports `(total, total)` when the feed ends. The merge phase sends no
//! reports, and `progressEveryBytes = 0` disables reporting. The callback is a
//! [`ThreadsafeFunction`] called in `NonBlocking` mode. Calls wait in the
//! JavaScript queue without blocking training. The code ignores enqueue
//! failures, including failures after JavaScript has shut down.
//!
//! Training cannot be cancelled. Dropping the JavaScript promise does not stop
//! the blocking task.
//!
//! # NAPI ownership
//!
//! [`NativeTokenizer`] stores CPU data such as vocabulary tables, merge rules,
//! and regular expressions in an [`Arc`]<[`TokenizerInner`]>. Batch encoding
//! clones the `Arc` into `spawn_blocking` tasks. This keeps the tokenizer alive
//! during background work and permits concurrent JavaScript calls. The native
//! object owns no device buffers or file handles. NAPI finalization frees its
//! memory after JavaScript collects the wrapper, so it needs no explicit
//! `dispose`.
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
use tokenizers::tokenizer::step_decode_stream;
use tokenizers::{AddedToken, Encoding, PostProcessor, Tokenizer};

/// Training corpus selector for [`NativeTokenizer::train`].
///
/// `tag = "Files"` requires `paths` and streams those files line by line.
/// `tag = "Texts"` requires `texts` and trains from those in-memory strings.
#[napi(object)]
pub struct NativeTrainSource {
    /// Corpus kind. Must be `"Files"` or `"Texts"`.
    pub tag: String,
    /// File paths to stream when `tag` is `"Files"`.
    pub paths: Option<Vec<String>>,
    /// In-memory sequences to train on when `tag` is `"Texts"`.
    pub texts: Option<Vec<String>>,
}

/// Configuration for [`NativeTokenizer::train`].
#[napi(object)]
pub struct NativeTrainConfig {
    /// Model architecture. Accepts `"BPE"`, `"WordPiece"`, `"Unigram"`, or
    /// `"WordLevel"`. `train_tokenizer` defines each architecture's fixed
    /// normalizer, pre-tokenizer, and decoder pipeline.
    pub model: String,
    /// Target vocabulary size, including special tokens.
    pub vocab_size: u32,
    /// Minimum corpus frequency for a token to enter the vocabulary.
    pub min_frequency: u32,
    /// Special tokens to register, such as `"<|endoftext|>"`. Training adds
    /// them to the vocabulary as special added tokens.
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

/// Renders `template` against `context_json` with the module's `minijinja`
/// environment and its `raise_exception` and `strftime_now` helpers.
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

// Split around special-token strings and keep each match as a segment. Encoding
// the segments separately prevents raw text from producing a special-token id
// while still tokenizing the special strings as ordinary text.
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

/// Stateful decoder for autoregressive token streams.
#[napi]
pub struct NativeDecodeStream {
    inner: Arc<TokenizerInner>,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    skip_special_tokens: bool,
}

#[napi]
impl NativeDecodeStream {
    /// Adds one token and returns the next stable text chunk, if available.
    #[napi]
    pub fn step(&mut self, id: u32) -> Result<Option<String>> {
        step_decode_stream(
            &self.inner.tokenizer,
            vec![id],
            self.skip_special_tokens,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )
        .map_err(to_napi_error)
    }
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

    // An exact special-token string must tokenize as ordinary text. Split it
    // until no piece can resolve to the special id through a whole-token lookup.
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
/// An [`Arc`] holds the [`tokenizers::Tokenizer`] and its special-token policy
/// while batch work runs in the background. The instance is immutable after
/// construction, so encoding and decoding may run concurrently. NAPI
/// finalization frees its CPU memory after JavaScript collects the wrapper.
#[napi]
pub struct NativeTokenizer {
    // This holds only CPU data. NAPI finalization frees it, so unlike a device
    // buffer handle, it needs no explicit dispose operation.
    inner: Arc<TokenizerInner>,
}

#[napi]
impl NativeTokenizer {
    fn inner(&self) -> &Arc<TokenizerInner> {
        &self.inner
    }

    /// Loads a Hugging Face `tokenizer.json` file.
    ///
    /// With `parse_specials = true`, special-token strings resolve to their
    /// ids. With `false`, raw text does not produce a special id.
    #[napi(factory)]
    pub fn from_file(path: String, parse_specials: bool) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(to_napi_error)?;
        Ok(Self {
            inner: Arc::new(TokenizerInner::new(tokenizer, parse_specials)),
        })
    }

    /// Loads a Hugging Face `tokenizer.json` string. `parse_specials` behaves
    /// as described by [`NativeTokenizer::from_file`].
    #[napi(factory)]
    pub fn from_json(json: String, parse_specials: bool) -> Result<Self> {
        let tokenizer = Tokenizer::from_bytes(json.as_bytes()).map_err(to_napi_error)?;
        Ok(Self {
            inner: Arc::new(TokenizerInner::new(tokenizer, parse_specials)),
        })
    }

    /// Trains a new tokenizer according to `config`.
    ///
    /// The job runs on Tokio's blocking thread pool and cannot be cancelled.
    /// During the corpus feed, `progress` receives
    /// `[processedBytes, totalBytes]` every `progressEveryBytes` bytes and a
    /// final `[total, total]`. A zero interval disables reports. The module
    /// docs describe each architecture's pipeline. `parse_specials` behaves as
    /// described by [`NativeTokenizer::from_file`].
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

    /// Total vocabulary size, including added and special tokens.
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

    /// Saves the tokenizer to `path` as a compact Hugging Face
    /// `tokenizer.json` file.
    #[napi]
    pub fn save(&self, path: String) -> Result<()> {
        self.inner()
            .tokenizer
            .save(path, false)
            .map_err(to_napi_error)
    }

    /// Encodes `text` to token ids.
    ///
    /// `add_special_tokens` defaults to `true` and controls whether the
    /// postprocessor adds template tokens such as `[CLS]` and `[SEP]`. The
    /// `parse_specials` flag chosen at construction controls input segmentation.
    #[napi]
    pub fn encode(&self, text: String, add_special_tokens: Option<bool>) -> Result<Uint32Array> {
        Ok(self
            .inner()
            .encode_ids_with(&text, add_special_tokens.unwrap_or(true))?
            .into())
    }

    /// Encodes a batch of texts, always adding special tokens.
    ///
    /// Tokio runs the job on its blocking thread pool. Rayon processes the
    /// texts in parallel while preserving input order.
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

    /// Decodes token ids to text. `skip_special_tokens` defaults to `false`.
    /// When enabled, it omits special tokens instead of rendering their strings.
    #[napi]
    pub fn decode(&self, ids: Vec<u32>, skip_special_tokens: Option<bool>) -> Result<String> {
        self.inner()
            .tokenizer
            .decode(&ids, skip_special_tokens.unwrap_or(false))
            .map_err(to_napi_error)
    }

    /// Creates a stateful decoder for an autoregressive token stream.
    #[napi]
    pub fn decode_stream(&self, skip_special_tokens: Option<bool>) -> NativeDecodeStream {
        NativeDecodeStream {
            inner: self.inner().clone(),
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
            skip_special_tokens: skip_special_tokens.unwrap_or(false),
        }
    }

    /// Decodes a batch of id sequences. `skip_special_tokens` behaves as
    /// described by [`NativeTokenizer::decode`].
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

    /// Renders a Hugging Face chat template against a JSON context. The module
    /// docs describe the environment and its `raise_exception` and
    /// `strftime_now` helpers.
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

// The crate exposes no progress hook, and its internal indicatif bars are
// disabled. Count bytes as the trainer pulls from this iterator. Report feed
// completion as (total, total); the later merge phase exposes no progress.
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
                    // A zero step disables all reports.
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
                    // Texts reach the total exactly. Files can fall short after
                    // stripping newlines, so report (total, total) at the end.
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

// Stream files line by line instead of loading the whole corpus. In-memory
// texts use the same training path. File metadata provides byte totals.
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
            // Line lengths omit stripped newlines, so the completion event
            // supplies the exact file-metadata total.
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

// Rebuild the trained Unigram model with byte fallback. Append all 256
// `<0xXX>` pieces without changing existing ids. Text outside the trained
// vocabulary, including newlines, emoji, and unseen scripts, can then round
// trip through byte pieces instead of collapsing to <unk>. The trainer's
// unknown token remains at id 0.
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
    // Give byte pieces the lowest score so trained pieces win when available.
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
            // Match GPT-2 byte-level behavior with no prefix space and regex
            // splitting. Seed all 256 bytes so unseen bytes still round trip.
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
            // Decode byte-fallback pieces to bytes, then replace the LLaMA
            // metaspace marker with a space.
            tokenizer.with_decoder(Some(SequenceDecoder::new(vec![
                ByteFallbackDecoder::default().into(),
                MetaspaceDecoder::default().into(),
            ])));
            // SentencePiece reserves id 0 for <unk>. The builder otherwise
            // defaults unk_token to None.
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

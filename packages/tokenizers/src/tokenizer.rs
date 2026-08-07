use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use rayon::prelude::*;
use std::io::BufRead;
use std::sync::Arc;
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

#[napi(object)]
pub struct NativeTrainSource {
    pub tag: String,
    pub paths: Option<Vec<String>>,
    pub texts: Option<Vec<String>>,
}

#[napi(object)]
pub struct NativeTrainConfig {
    pub model: String,
    pub vocab_size: u32,
    pub min_frequency: u32,
    pub special_tokens: Vec<String>,
    pub source: NativeTrainSource,
}

fn to_napi_error<E: std::fmt::Display>(err: E) -> Error {
    Error::new(Status::GenericFailure, err.to_string())
}

fn to_join_error(error: tokio::task::JoinError) -> Error {
    Error::new(Status::GenericFailure, error.to_string())
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

    fn encode_ids(&self, text: &str) -> Result<Vec<u32>> {
        if self.parse_specials || self.specials.is_empty() {
            let encoding = self.tokenizer.encode(text, true).map_err(to_napi_error)?;
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
        if let Some(post_processor) = self.tokenizer.get_post_processor() {
            merged = post_processor
                .process(merged, None, true)
                .map_err(to_napi_error)?;
        }
        Ok(merged.get_ids().to_vec())
    }
}

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

    #[napi(factory)]
    pub fn from_file(path: String, parse_specials: bool) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(to_napi_error)?;
        Ok(Self {
            inner: Arc::new(TokenizerInner::new(tokenizer, parse_specials)),
        })
    }

    #[napi(factory)]
    pub fn from_json(json: String, parse_specials: bool) -> Result<Self> {
        let tokenizer = Tokenizer::from_bytes(json.as_bytes()).map_err(to_napi_error)?;
        Ok(Self {
            inner: Arc::new(TokenizerInner::new(tokenizer, parse_specials)),
        })
    }

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

    #[napi(getter)]
    pub fn vocab_size(&self) -> Result<u32> {
        Ok(self.inner().tokenizer.get_vocab_size(true) as u32)
    }

    #[napi]
    pub fn token_to_id(&self, token: String) -> Result<Option<u32>> {
        Ok(self.inner().tokenizer.token_to_id(&token))
    }

    #[napi]
    pub fn id_to_token(&self, id: u32) -> Result<Option<String>> {
        Ok(self.inner().tokenizer.id_to_token(id))
    }

    #[napi]
    pub fn save(&self, path: String) -> Result<()> {
        self.inner()
            .tokenizer
            .save(path, false)
            .map_err(to_napi_error)
    }

    #[napi]
    pub fn encode(&self, text: String) -> Result<Uint32Array> {
        Ok(self.inner().encode_ids(&text)?.into())
    }

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

    #[napi]
    pub fn decode(&self, ids: Vec<u32>) -> Result<String> {
        self.inner()
            .tokenizer
            .decode(&ids, false)
            .map_err(to_napi_error)
    }

    #[napi]
    pub fn decode_batch(&self, ids: Vec<Vec<u32>>) -> Result<Vec<String>> {
        let refs: Vec<&[u32]> = ids.iter().map(|v| v.as_slice()).collect();
        self.inner()
            .tokenizer
            .decode_batch(&refs, false)
            .map_err(to_napi_error)
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

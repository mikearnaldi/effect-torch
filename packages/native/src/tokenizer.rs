use crate::{get_device, LazyTensor};
use crate::{DType, Node, NodeKind};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;
use std::sync::Arc;
use tokenizers::decoders::{
    byte_level::ByteLevel as ByteLevelDecoder, metaspace::Metaspace as MetaspaceDecoder,
    wordpiece::WordPiece as WordPieceDecoder,
};
use tokenizers::models::{bpe::BPE, unigram::Unigram, wordlevel::WordLevel, wordpiece::WordPiece};
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::{
    byte_level::ByteLevel, metaspace::Metaspace, whitespace::Whitespace,
};
use tokenizers::models::TrainerWrapper;
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

#[napi(object)]
pub struct NativePadding {
    pub tag: String,
    pub pad_id: Option<u32>,
    pub max_length: Option<u32>,
}

#[napi(object)]
pub struct NativeTruncation {
    pub tag: String,
    pub max_length: Option<u32>,
}

fn to_napi_error<E: std::fmt::Display>(err: E) -> Error {
    Error::new(Status::GenericFailure, err.to_string())
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
            if let Some(special) = specials.iter().find(|sp| text[i..].starts_with(sp.as_str())) {
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

fn ids_to_tensor(ids: Vec<u32>, shape: Vec<usize>, device: Option<String>) -> Result<LazyTensor> {
    let mut data = Vec::with_capacity(ids.len() * 4);
    for id in ids {
        data.extend_from_slice(&id.to_le_bytes());
    }
    let node = Node::new(NodeKind::FromBytes {
        data,
        shape,
        dtype: DType::U32,
        device: get_device(device)?,
    })
    .map_err(to_napi_error)?;
    Ok(LazyTensor::from_node(node))
}

fn apply_truncation(ids: &mut Vec<u32>, truncation: &NativeTruncation) -> Result<()> {
    match truncation.tag.as_str() {
        "None" => Ok(()),
        "MaxLength" => {
            let max = truncation
                .max_length
                .ok_or_else(|| Error::new(Status::InvalidArg, "truncation MaxLength: missing maxLength".to_string()))?
                as usize;
            ids.truncate(max);
            Ok(())
        }
        tag => Err(Error::new(
            Status::InvalidArg,
            format!("truncation: unknown tag {tag}"),
        )),
    }
}

fn pad_batch(batch: Vec<Vec<u32>>, padding: &NativePadding) -> Result<(Vec<u32>, usize)> {
    if batch.is_empty() {
        return Err(Error::new(
            Status::InvalidArg,
            "encodeBatch: expected at least one text".to_string(),
        ));
    }
    match padding.tag.as_str() {
        "None" => {
            let len = batch[0].len();
            if batch.iter().any(|ids| ids.len() != len) {
                return Err(Error::new(
                    Status::InvalidArg,
                    "encodeBatch: ragged encodings with padding None; set padding to Longest or MaxLength".to_string(),
                ));
            }
            Ok((batch.into_iter().flatten().collect(), len))
        }
        "Longest" => {
            let pad_id = padding
                .pad_id
                .ok_or_else(|| Error::new(Status::InvalidArg, "padding Longest: missing padId".to_string()))?;
            let len = batch.iter().map(|ids| ids.len()).max().unwrap_or(0);
            let mut flat = Vec::with_capacity(batch.len() * len);
            for ids in batch {
                flat.extend_from_slice(&ids);
                flat.extend(std::iter::repeat(pad_id).take(len - ids.len()));
            }
            Ok((flat, len))
        }
        "MaxLength" => {
            let pad_id = padding.pad_id.ok_or_else(|| {
                Error::new(Status::InvalidArg, "padding MaxLength: missing padId".to_string())
            })?;
            let len = padding.max_length.ok_or_else(|| {
                Error::new(
                    Status::InvalidArg,
                    "padding MaxLength: missing maxLength".to_string(),
                )
            })? as usize;
            if batch.iter().any(|ids| ids.len() > len) {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!(
                        "encodeBatch: encoding longer than padding maxLength {len}; set truncation to MaxLength"
                    ),
                ));
            }
            let mut flat = Vec::with_capacity(batch.len() * len);
            for ids in batch {
                flat.extend_from_slice(&ids);
                flat.extend(std::iter::repeat(pad_id).take(len - ids.len()));
            }
            Ok((flat, len))
        }
        tag => Err(Error::new(
            Status::InvalidArg,
            format!("padding: unknown tag {tag}"),
        )),
    }
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
        self.tokenizer
            .encode(segment, false)
            .map_err(to_napi_error)
    }

    fn encode_ids(&self, text: &str) -> Result<Vec<u32>> {
        if self.parse_specials || self.specials.is_empty() {
            let encoding = self
                .tokenizer
                .encode(text, true)
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

    #[napi(factory)]
    pub async fn train(config: NativeTrainConfig, parse_specials: bool) -> Result<Self> {
        let tokenizer = tokio::task::spawn_blocking(move || train_tokenizer(config))
            .await
            .map_err(crate::to_join_err)??;
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
        .map_err(crate::to_join_err)?
    }

    #[napi]
    pub fn encode_tensor(&self, text: String, device: Option<String>) -> Result<LazyTensor> {
        let ids = self.inner().encode_ids(&text)?;
        let len = ids.len();
        ids_to_tensor(ids, vec![len], device)
    }

    #[napi]
    pub async fn encode_batch_tensor(
        &self,
        texts: Vec<String>,
        padding: NativePadding,
        truncation: NativeTruncation,
        device: Option<String>,
    ) -> Result<LazyTensor> {
        let inner = self.inner().clone();
        let batch = tokio::task::spawn_blocking(move || {
            texts
                .par_iter()
                .map(|text| {
                    let mut ids = inner.encode_ids(text)?;
                    apply_truncation(&mut ids, &truncation)?;
                    Ok(ids)
                })
                .collect::<Result<Vec<Vec<u32>>>>()
        })
        .await
        .map_err(crate::to_join_err)??;
        let rows = batch.len();
        let (flat, cols) = pad_batch(batch, &padding)?;
        ids_to_tensor(flat, vec![rows, cols], device)
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

fn run_trainer(
    tokenizer: &mut Tokenizer,
    trainer: &mut TrainerWrapper,
    source: NativeTrainSource,
) -> Result<()> {
    match source.tag.as_str() {
        "Files" => {
            tokenizer
                .train_from_files(
                    trainer,
                    source.paths.ok_or_else(|| {
                        Error::new(
                            Status::InvalidArg,
                            "train: Files source requires paths".to_string(),
                        )
                    })?,
                )
                .map_err(to_napi_error)?;
        }
        "Texts" => {
            tokenizer
                .train(
                    trainer,
                    source
                        .texts
                        .ok_or_else(|| {
                            Error::new(
                                Status::InvalidArg,
                                "train: Texts source requires texts".to_string(),
                            )
                        })?
                        .into_iter(),
                )
                .map_err(to_napi_error)?;
        }
        tag => {
            return Err(Error::new(
                Status::InvalidArg,
                format!("train: unknown source tag {tag}"),
            ))
        }
    }
    Ok(())
}

fn train_tokenizer(config: NativeTrainConfig) -> Result<Tokenizer> {
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
            run_trainer(&mut tokenizer, &mut trainer, source)?;
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
            run_trainer(&mut tokenizer, &mut trainer, source)?;
            tokenizer.add_special_tokens(&specials);
            Ok(tokenizer)
        }
        "Unigram" => {
            let mut tokenizer = Tokenizer::new(Unigram::default());
            tokenizer.with_pre_tokenizer(Some(Metaspace::default()));
            tokenizer.with_decoder(Some(MetaspaceDecoder::default()));
            let mut trainer = TrainerWrapper::from(
                tokenizers::models::unigram::UnigramTrainer::builder()
                    .show_progress(false)
                    .vocab_size(vocab_size as u32)
                    .special_tokens(special_tokens.clone())
                    .build()
                    .map_err(to_napi_error)?,
            );
            run_trainer(&mut tokenizer, &mut trainer, source)?;
            tokenizer.add_special_tokens(&special_tokens);
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
            run_trainer(&mut tokenizer, &mut trainer, source)?;
            tokenizer.add_special_tokens(&special_tokens);
            Ok(tokenizer)
        }
        model => Err(Error::new(
            Status::InvalidArg,
            format!("train: unknown model {model}"),
        )),
    }
}

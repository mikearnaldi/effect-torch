//! NAPI-RS bindings to the [`tokenizers`] crate.
//!
//! [`NativeTokenizer`] loads tokenizers from JSON, encodes and decodes single
//! inputs or batches, trains BPE, WordPiece, Unigram, and WordLevel models, and
//! renders Hugging Face chat templates. Training accepts files or in-memory
//! text and reports corpus progress by byte count.
//!
//! The `tokenizer` module documents special-token handling, postprocessing,
//! training, and NAPI ownership. This crate contains no `unsafe` code.

mod tokenizer;

pub use tokenizer::*;

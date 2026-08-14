//! Native tokenizer bindings for effect-torch.
//!
//! This crate is the NAPI-RS bridge between the JavaScript side of
//! `effect-torch` and the [`tokenizers`] crate. It exposes a single class,
//! [`NativeTokenizer`], that supports:
//!
//! - loading a tokenizer from a JSON file or string
//!   ([`NativeTokenizer::from_file`], [`NativeTokenizer::from_json`]),
//! - encoding and decoding, single or batched, with control over special
//!   tokens ([`NativeTokenizer::encode`], [`NativeTokenizer::encode_batch`],
//!   [`NativeTokenizer::decode`], [`NativeTokenizer::decode_batch`]),
//! - training BPE, WordPiece, Unigram, and WordLevel models from files or
//!   in-memory texts with byte-level progress reporting
//!   ([`NativeTokenizer::train`]), and
//! - rendering Hugging Face chat templates through a `minijinja`
//!   environment with `raise_exception` and `strftime_now` helpers
//!   ([`NativeTokenizer::apply_chat_template`]).
//!
//! See the `tokenizer` module for the full documentation of the
//! special-token segmentation strategy, the post-processor behavior, the
//! training progress/cancellation/threading model, and NAPI ownership
//! semantics.
//!
//! The crate contains no `unsafe` code; all memory safety invariants are
//! upheld by the safe `napi`/`tokenizers` APIs.

mod tokenizer;

pub use tokenizer::*;

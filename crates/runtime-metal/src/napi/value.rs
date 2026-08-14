//! NAPI-local view of the crate's concrete Metal leaf value.
//!
//! Keeping this as a re-export ensures graph leaves, serialization, GGUF
//! loading, and JavaScript tensor handles all use the same `Value` type.

pub use crate::value::*;

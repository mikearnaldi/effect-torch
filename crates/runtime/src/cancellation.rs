//! Cooperative cancellation for long-running runtime operations.
//!
//! Cancellation is never preemptive: producers call [`CancellationFlag::cancel`]
//! and workers poll [`CancellationFlag::is_cancelled`] at well-defined points
//! (per header chunk while parsing GGUF, per read chunk while loading tensor
//! payloads) and abort with a dedicated `Cancelled` error. A flag is one-shot —
//! once set it stays set — and safe to share across threads behind a reference.

use std::ops::Deref;
use std::sync::atomic::{AtomicBool, Ordering};

/// A shareable, one-shot cooperative cancellation signal.
///
/// `cancel` uses `Release` ordering and `is_cancelled` uses `Acquire`, so any
/// state written before cancellation is visible to a thread that observes the
/// flag set.
#[derive(Debug, Default)]
pub struct CancellationFlag(AtomicBool);

impl CancellationFlag {
    /// Creates a flag in the not-cancelled state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Permanently sets the flag. Idempotent.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Returns `true` once [`cancel`](Self::cancel) has been observed.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Dereferences to the inner [`AtomicBool`] so callers that need custom
/// orderings or compare-exchange semantics can use the atomic directly.
impl Deref for CancellationFlag {
    type Target = AtomicBool;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

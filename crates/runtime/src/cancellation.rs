//! Cooperative cancellation for long-running runtime operations.
//!
//! Cancellation is cooperative, not preemptive. Producers call
//! [`CancellationFlag::cancel`]. Workers poll
//! [`CancellationFlag::is_cancelled`] once per GGUF header or tensor payload
//! chunk and abort with a `Cancelled` error. Once set, a flag stays set.
//! Callers can share a reference across threads.

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
    /// Creates an unset flag.
    pub fn new() -> Self {
        Self::default()
    }

    /// Permanently sets the flag. Idempotent.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Returns `true` after [`cancel`](Self::cancel) sets the flag.
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

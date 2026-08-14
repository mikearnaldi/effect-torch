//! N-API support utilities: cooperative cancellation of blocking compute
//! and ownership hand-off of Rust buffers to JavaScript.
//!
//! Cancellation model: JS-facing async operations share a
//! [`CancellationState`] between the caller (who may cancel) and the worker
//! (who completes). Exactly one side wins — a completed operation returns
//! its result even if a cancel raced in afterwards, and a cancelled
//! operation discards its result and reports `Status::Cancelled`. The state
//! is a one-shot: neither `cancel` nor `complete` has any effect once the
//! other has committed.
//!
//! Buffer ownership: [`vec_to_bytes`] leaks a `Vec`'s allocation so it can
//! be wrapped in a JS `Buffer`/`ArrayBuffer` whose finalizer later
//! reconstructs and drops the `Vec`. The debug-only export registry
//! ([`try_register_export`]/[`unregister_export`]) detects double-exports of
//! one allocation during development; in release builds the registration is
//! compiled out and correctness rests on each leaked allocation being
//! exported exactly once.

use effect_torch_runtime::CancellationFlag;
use napi::{Error, Result, Status};
#[cfg(debug_assertions)]
use std::collections::HashSet;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
#[cfg(debug_assertions)]
use std::sync::{Mutex, OnceLock};

/// One-shot cancellation/commit arbitration for a single async operation.
///
/// `phase` encodes the lifecycle: `0` = in flight, `1` = cancelled,
/// `2` = completed. Transitions happen only from `0`, so the first of
/// [`CancellationState::cancel`] and [`CancellationState::complete`] to
/// commit wins and the loser observes the winner's phase.
pub struct CancellationState {
    cancelled: CancellationFlag,
    phase: AtomicU8,
}

impl CancellationState {
    pub fn new() -> Self {
        Self {
            cancelled: CancellationFlag::new(),
            phase: AtomicU8::new(0),
        }
    }

    /// The cooperative flag compute loops poll between chunks of work.
    pub fn flag(&self) -> &CancellationFlag {
        &self.cancelled
    }

    /// Attempts to cancel the operation. Returns `true` for the caller that
    /// actually committed the cancellation; `false` if the operation already
    /// completed or was already cancelled.
    pub fn cancel(&self) -> bool {
        if self
            .phase
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.cancelled.cancel();
            true
        } else {
            false
        }
    }

    /// Attempts to commit the operation's result. Returns `true` when the
    /// result should be delivered (this call committed, or completion
    /// already won); `false` when cancellation won and the result must be
    /// discarded.
    pub fn complete(&self) -> bool {
        match self
            .phase
            .compare_exchange(0, 2, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) | Err(2) => true,
            Err(1) => false,
            Err(_) => unreachable!("unknown cancellation phase"),
        }
    }
}

impl Default for CancellationState {
    fn default() -> Self {
        Self::new()
    }
}

/// Runs blocking `compute` on a blocking thread and races it against an
/// optional external cancellation notification.
///
/// The compute closure receives the cooperative flag and the state; when it
/// returns, [`CancellationState::complete`] arbitrates the outcome — a
/// cancelled operation's result is dropped and `Status::Cancelled` is
/// reported instead. With `notify` present, whichever of the worker
/// finishing or the notification firing comes first decides the await's
/// result; the biased select prefers a finished worker so a same-instant
/// completion is not lost to a late notification. The worker is always
/// awaited before returning, so no detached compute outlives the call.
pub async fn run_compute<T: Send + 'static>(
    state: Arc<CancellationState>,
    notify: Option<Arc<tokio::sync::Notify>>,
    compute: impl FnOnce(&CancellationFlag, &CancellationState) -> Result<T> + Send + 'static,
) -> Result<T> {
    let worker_state = state.clone();
    let mut handle = tokio::task::spawn_blocking(move || {
        let result = compute(worker_state.flag(), &worker_state);

        if worker_state.complete() {
            result
        } else {
            drop(result);
            Err(Error::new(Status::Cancelled, "operation aborted"))
        }
    });

    let Some(notify) = notify else {
        return handle.await.map_err(to_join_error)?;
    };
    if state.flag().is_cancelled() {
        let _ = handle.await.map_err(to_join_error)?;
        return Err(Error::new(Status::Cancelled, "operation aborted"));
    }
    tokio::select! {
        biased;
        result = &mut handle => result.map_err(to_join_error)?,
        _ = notify.notified() => {
            let _ = handle.await.map_err(to_join_error)?;
            Err(Error::new(Status::Cancelled, "operation aborted"))
        },
    }
}

/// Converts a worker-task join failure (panic or abort) into a N-API error.
pub fn to_join_error(error: tokio::task::JoinError) -> Error {
    Error::new(Status::GenericFailure, error.to_string())
}

/// Debug-build registry of exported buffer base addresses, used to assert
/// the exactly-once export contract (see the module documentation).
#[cfg(debug_assertions)]
fn exported_buffers() -> &'static Mutex<HashSet<usize>> {
    static BUFFERS: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();
    BUFFERS.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Registers an allocation base address as exported to JS. Returns `false`
/// in debug builds when the address is already exported (a double-export
/// bug); release builds always succeed and perform no bookkeeping.
pub fn try_register_export(addr: usize) -> bool {
    #[cfg(debug_assertions)]
    {
        exported_buffers().lock().unwrap().insert(addr)
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = addr;
        true
    }
}

/// Marks an exported allocation as reclaimed by its JS finalizer.
pub fn unregister_export(addr: usize) {
    #[cfg(debug_assertions)]
    exported_buffers().lock().unwrap().remove(&addr);
    #[cfg(not(debug_assertions))]
    let _ = addr;
}

/// Leaks a `Vec`'s allocation for hand-off to JavaScript, returning the
/// base address (for the export registry), the raw byte pointer, the live
/// byte length, and the byte capacity.
///
/// SAFETY: this is an ownership transfer out of Rust's tracking. The caller
/// must guarantee that (1) the returned pointer/length/capacity are used to
/// reconstruct exactly one `Vec<T>` — typically via `Vec::from_raw_parts` in
/// the JS buffer's finalizer — and drop it exactly once; (2) the allocation
/// is not accessed through Rust while JS owns it; and (3) `T`'s element
/// layout is what the JS side expects, since the pointer is type-erased to
/// bytes. Violating exactly-once reconstruction is a leak (never reclaimed)
/// or a double-free (reclaimed twice); the debug export registry exists to
/// catch the double-export half of this contract.
pub fn vec_to_bytes<T>(mut vec: Vec<T>) -> (usize, *mut u8, usize, usize) {
    let ptr = vec.as_mut_ptr().cast::<u8>();
    let len = std::mem::size_of_val(vec.as_slice());
    let cap = vec.capacity() * std::mem::size_of::<T>();
    let addr = ptr as usize;
    std::mem::forget(vec);
    (addr, ptr, len, cap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    #[test]
    fn cancellation_and_commit_have_one_winner() {
        for _ in 0..1_000 {
            let state = Arc::new(CancellationState::new());
            let barrier = Arc::new(Barrier::new(3));
            let cancel_state = state.clone();
            let cancel_barrier = barrier.clone();
            let cancel = std::thread::spawn(move || {
                cancel_barrier.wait();
                cancel_state.cancel()
            });
            let complete_state = state.clone();
            let complete_barrier = barrier.clone();
            let complete = std::thread::spawn(move || {
                complete_barrier.wait();
                complete_state.complete()
            });
            barrier.wait();
            assert_ne!(cancel.join().unwrap(), complete.join().unwrap());
        }
    }
}

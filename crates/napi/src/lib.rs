//! Utilities for cooperative cancellation of blocking N-API work and for
//! transferring Rust buffer ownership to JavaScript.
//!
//! The caller and worker for a JS-facing async operation share a
//! [`CancellationState`]. The caller can cancel the operation. The worker can
//! complete it. The first transition wins. If completion wins, the operation
//! returns its result even when cancellation races with it or arrives later.
//! If cancellation wins, the worker drops the result and reports
//! `Status::Cancelled`. The state is one-shot. `cancel` and `complete` cannot
//! change it after either method commits.
//!
//! [`vec_to_bytes`] leaks a `Vec` allocation so the caller can wrap it in a
//! JavaScript `Buffer` or `ArrayBuffer`. The buffer's finalizer must
//! reconstruct and drop the `Vec`. In debug builds,
//! [`try_register_export`] and [`unregister_export`] detect attempts to
//! export one allocation twice. Release builds omit this registry, so callers
//! must export each leaked allocation exactly once.

use effect_torch_runtime::CancellationFlag;
use napi::{Error, Result, Status};
#[cfg(debug_assertions)]
use std::collections::HashSet;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
#[cfg(debug_assertions)]
use std::sync::{Mutex, OnceLock};

/// Coordinates cancellation and completion for one async operation.
///
/// `phase` stores the lifecycle state. `0` means in flight, `1` means
/// cancelled, and `2` means completed. Only a transition from `0` can
/// commit. The first call to [`CancellationState::cancel`] or
/// [`CancellationState::complete`] that commits wins. A losing call reads the
/// committed phase.
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

    /// Returns the cooperative flag that compute loops poll between chunks.
    pub fn flag(&self) -> &CancellationFlag {
        &self.cancelled
    }

    /// Attempts to cancel the operation. Returns `true` only when this call
    /// commits cancellation. Returns `false` if the operation completed or
    /// another call already cancelled it.
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

    /// Attempts to commit the operation's result. Returns `true` if this call
    /// commits completion or another completion call already won. Returns
    /// `false` if cancellation won, in which case the caller must discard the
    /// result.
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

/// Runs the blocking `compute` closure on a blocking thread and races its
/// result against an optional external cancellation notification.
///
/// `compute` receives the cooperative flag and shared state. After it
/// returns, [`CancellationState::complete`] chooses the outcome. If
/// cancellation won, the worker drops the result and reports
/// `Status::Cancelled`. When `notify` is present, its signal races worker
/// completion. Whichever becomes ready first determines the result. If both
/// are ready, the biased select checks the worker first and returns its
/// result. The function awaits the worker in every branch, so `compute`
/// cannot outlive the call.
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

/// Converts a worker task's join error from a panic or abort into an N-API
/// error.
pub fn to_join_error(error: tokio::task::JoinError) -> Error {
    Error::new(Status::GenericFailure, error.to_string())
}

/// Stores exported buffer base addresses in debug builds to detect attempts
/// to export one allocation twice.
#[cfg(debug_assertions)]
fn exported_buffers() -> &'static Mutex<HashSet<usize>> {
    static BUFFERS: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();
    BUFFERS.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Registers an allocation's base address as exported to JavaScript. In debug
/// builds, it returns `false` for an address already in the registry because
/// the caller tried to export one allocation twice. In release builds, it
/// skips the registry and returns `true`.
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

/// Records that a JavaScript finalizer reclaimed an exported allocation.
pub fn unregister_export(addr: usize) {
    #[cfg(debug_assertions)]
    exported_buffers().lock().unwrap().remove(&addr);
    #[cfg(not(debug_assertions))]
    let _ = addr;
}

/// Leaks a `Vec` allocation so JavaScript can take ownership. Returns the
/// base address for the export registry, the raw byte pointer, the live byte
/// length, and the byte capacity.
///
/// SAFETY: This transfers ownership outside Rust's tracking. The caller must:
///
/// 1. Reconstruct exactly one `Vec<T>` from the returned pointer, length, and
///    capacity, then drop it exactly once. A JavaScript buffer finalizer can
///    do this with `Vec::from_raw_parts`.
/// 2. Do not access the allocation through Rust while JavaScript owns it.
/// 3. Make sure `T`'s element layout matches what JavaScript expects. The
///    returned byte pointer erases the element type.
///
/// If the caller never reconstructs the `Vec`, the allocation leaks.
/// Reconstructing it more than once causes a double-free. The debug export
/// registry catches attempts to export the same allocation twice.
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

use effect_torch_runtime::CancellationFlag;
use napi::{Error, Result, Status};
#[cfg(debug_assertions)]
use std::collections::HashSet;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
#[cfg(debug_assertions)]
use std::sync::{Mutex, OnceLock};

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

    pub fn flag(&self) -> &CancellationFlag {
        &self.cancelled
    }

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

pub fn to_join_error(error: tokio::task::JoinError) -> Error {
    Error::new(Status::GenericFailure, error.to_string())
}

#[cfg(debug_assertions)]
fn exported_buffers() -> &'static Mutex<HashSet<usize>> {
    static BUFFERS: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();
    BUFFERS.get_or_init(|| Mutex::new(HashSet::new()))
}

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

pub fn unregister_export(addr: usize) {
    #[cfg(debug_assertions)]
    exported_buffers().lock().unwrap().remove(&addr);
    #[cfg(not(debug_assertions))]
    let _ = addr;
}

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

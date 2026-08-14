//! Process-wide workspace pool backing executable scratch memory.
//!
//! Compiled executables describe their transient memory as a set of segment
//! requests (bytes + alignment). This module keys those requests into
//! [`CpuWorkspaceKey`] capacity classes, allocates backing [`CpuSegment`]s on
//! demand, and returns them to a shared LRU [`WorkspacePool`] when a lease
//! ends, so steady-state invocations of a compiled program reuse memory
//! instead of hitting the global allocator.
//!
//! Pool sizing is controlled by the `EFFECT_TORCH_CPU_WORKSPACE_POOL_MB`
//! environment variable (default 256 MiB of idle segments). Idle segments
//! beyond the budget are evicted least-recently-used first.
//!
//! Ownership model: the pool hands out `Arc<CpuSegment>` owners plus a lease
//! token. Output and scratch views created during an invocation keep the
//! segment alive through the `Arc`, and optionally retain the lease itself
//! (see `CpuStorageRetention`) so a published output tensor pins its pool
//! segment until the last view is dropped.

use crate::storage::CpuSegment;
use effect_torch_runtime::{
    NativeMemorySpace, WorkspaceAllocation, WorkspaceAllocator, WorkspaceLease, WorkspacePool,
    WorkspaceRequest,
};
use std::sync::{Arc, OnceLock};

/// Pool key for one class of CPU workspace segments.
///
/// `capacity_class` is the requested byte count rounded up to a multiple of
/// `alignment`; rounding keeps the pool's best-fit buckets coarse enough to
/// reuse segments across slightly different requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CpuWorkspaceKey {
    pub memory_space: NativeMemorySpace,
    pub alignment: usize,
    pub capacity_class: usize,
}

impl CpuWorkspaceKey {
    /// Builds a key, validating that `alignment` is a non-zero power of two
    /// and that the rounded capacity class does not overflow.
    pub(crate) fn new(bytes: usize, alignment: usize) -> Result<Self, String> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(format!("invalid CPU workspace alignment {alignment}"));
        }
        let capacity_class = bytes
            .checked_next_multiple_of(alignment)
            .ok_or_else(|| "CPU workspace capacity class overflow".to_string())?;
        Ok(Self {
            memory_space: NativeMemorySpace::Cpu,
            alignment,
            capacity_class,
        })
    }
}

/// Allocates fresh, zeroed, correctly aligned segments for the pool.
///
/// The allocator is stateless; all reuse policy lives in the generic
/// [`WorkspacePool`]. Allocation is validated against the key's memory space,
/// alignment, and capacity class so a mismatched request can never produce a
/// segment smaller than expected.
#[derive(Debug, Default)]
pub(crate) struct CpuWorkspaceAllocator;

impl WorkspaceAllocator<CpuWorkspaceKey> for CpuWorkspaceAllocator {
    type Workspace = Arc<CpuSegment>;
    type Error = String;

    fn allocate(
        &mut self,
        key: &CpuWorkspaceKey,
        minimum_bytes: usize,
    ) -> Result<WorkspaceAllocation<Self::Workspace>, Self::Error> {
        if key.memory_space != NativeMemorySpace::Cpu {
            return Err(format!(
                "unsupported CPU workspace memory space {:?}",
                key.memory_space
            ));
        }
        if key.alignment == 0 || !key.alignment.is_power_of_two() {
            return Err(format!("invalid CPU workspace alignment {}", key.alignment));
        }
        if minimum_bytes > key.capacity_class {
            return Err(format!(
                "CPU workspace request of {minimum_bytes} bytes exceeds capacity class {}",
                key.capacity_class
            ));
        }
        let segment = CpuSegment::allocate(minimum_bytes, key.alignment)
            .map_err(|error| error.to_string())?;
        Ok(WorkspaceAllocation::new(segment, minimum_bytes))
    }
}

pub(crate) type CpuWorkspacePool = WorkspacePool<CpuWorkspaceKey, CpuWorkspaceAllocator>;
pub(crate) type CpuWorkspaceLease = WorkspaceLease<CpuWorkspaceKey, CpuWorkspaceAllocator>;

/// Returns the shared workspace pool, initializing it on first use from
/// `EFFECT_TORCH_CPU_WORKSPACE_POOL_MB` (default: 256 MiB idle budget).
pub(crate) fn workspace_pool() -> &'static CpuWorkspacePool {
    static POOL: OnceLock<CpuWorkspacePool> = OnceLock::new();
    POOL.get_or_init(|| {
        let max_idle_bytes = std::env::var("EFFECT_TORCH_CPU_WORKSPACE_POOL_MB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .and_then(|megabytes| megabytes.checked_mul(1024 * 1024))
            .unwrap_or(256 * 1024 * 1024);
        CpuWorkspacePool::new(max_idle_bytes, CpuWorkspaceAllocator)
    })
}

/// Builds a validated pool request for `bytes` at `alignment`.
///
/// Zero-byte requests are rounded up to one byte so every lease maps to a
/// real (minimal) physical allocation.
pub(crate) fn workspace_request(
    bytes: usize,
    alignment: usize,
) -> Result<WorkspaceRequest<CpuWorkspaceKey>, String> {
    let bytes = bytes.max(1);
    Ok(WorkspaceRequest::new(
        CpuWorkspaceKey::new(bytes, alignment)?,
        bytes,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{CpuStorage, CpuStorageRetention};

    #[test]
    fn whole_segment_pool_is_right_sized_best_fit_and_lru_bounded() {
        let pool = CpuWorkspacePool::new(80, CpuWorkspaceAllocator);
        let key = CpuWorkspaceKey::new(64, 16).unwrap();
        let small = pool.acquire(&[WorkspaceRequest::new(key, 32)]).unwrap();
        let small_address = small.segments()[0].workspace().as_ptr() as usize;
        let large = pool.acquire(&[WorkspaceRequest::new(key, 48)]).unwrap();
        assert_eq!(small.segments()[0].capacity(), 32);
        assert_eq!(large.segments()[0].capacity(), 48);
        drop(small);
        drop(large);

        let reused = pool.acquire(&[WorkspaceRequest::new(key, 24)]).unwrap();
        assert_eq!(
            reused.segments()[0].workspace().as_ptr() as usize,
            small_address
        );
        assert_eq!(reused.segments()[0].capacity(), 32);
        drop(reused);
        drop(pool.acquire(&[WorkspaceRequest::new(key, 64)]).unwrap());
        assert!(pool.stats().idle_bytes <= 80);
        assert_eq!(pool.stats().leased_segments, 0);
    }

    #[test]
    fn whole_segment_set_acquisition_rolls_back_atomically() {
        let pool = CpuWorkspacePool::new(1024, CpuWorkspaceAllocator);
        let cached_key = CpuWorkspaceKey::new(64, 16).unwrap();
        let cached = pool
            .acquire(&[WorkspaceRequest::new(cached_key, 64)])
            .unwrap();
        let cached_address = cached.segments()[0].workspace().as_ptr() as usize;
        drop(cached);
        let before = pool.stats();

        let invalid_key = CpuWorkspaceKey {
            memory_space: NativeMemorySpace::MetalShared,
            alignment: 16,
            capacity_class: 16,
        };
        assert!(pool
            .acquire_set(&[
                WorkspaceRequest::new(cached_key, 32),
                WorkspaceRequest::new(invalid_key, 16),
            ])
            .is_err());
        assert_eq!(pool.stats(), before);
        let reused = pool
            .acquire(&[WorkspaceRequest::new(cached_key, 64)])
            .unwrap();
        assert_eq!(
            reused.segments()[0].workspace().as_ptr() as usize,
            cached_address
        );
    }

    #[test]
    fn pressure_evicts_the_least_recently_used_whole_segment() {
        let pool = CpuWorkspacePool::new(80, CpuWorkspaceAllocator);
        let old_key = CpuWorkspaceKey {
            memory_space: NativeMemorySpace::Cpu,
            alignment: 16,
            capacity_class: 48,
        };
        let hot_key = CpuWorkspaceKey {
            capacity_class: 64,
            ..old_key
        };
        let new_key = CpuWorkspaceKey {
            capacity_class: 80,
            ..old_key
        };

        let old = pool.acquire(&[WorkspaceRequest::new(old_key, 40)]).unwrap();
        let old_owner = Arc::downgrade(old.segments()[0].workspace());
        drop(old);
        let hot = pool.acquire(&[WorkspaceRequest::new(hot_key, 40)]).unwrap();
        let hot_owner = Arc::downgrade(hot.segments()[0].workspace());
        drop(hot);

        drop(pool.acquire(&[WorkspaceRequest::new(hot_key, 40)]).unwrap());
        drop(pool.acquire(&[WorkspaceRequest::new(new_key, 40)]).unwrap());

        assert!(old_owner.upgrade().is_none());
        assert!(hot_owner.upgrade().is_some());
        assert_eq!(pool.stats().idle_bytes, 80);
    }

    #[test]
    fn output_views_retain_the_pool_lease() {
        let pool = CpuWorkspacePool::new(1024, CpuWorkspaceAllocator);
        let request = workspace_request(16, 16).unwrap();
        let lease = Arc::new(pool.acquire(&[request]).unwrap());
        let owner = Arc::clone(lease.segments()[0].workspace());
        let retention: CpuStorageRetention = lease;
        let output =
            CpuStorage::<u8>::from_segment_with_retention(owner, 0, 16, Some(retention)).unwrap();
        let view = output.clone();
        assert_eq!(pool.stats().leased_segments, 1);
        drop(output);
        assert_eq!(pool.stats().leased_segments, 1);
        drop(view);
        assert_eq!(pool.stats().leased_segments, 0);
        assert_eq!(pool.stats().idle_segments, 1);
    }

    #[test]
    fn zero_byte_requests_account_for_the_physical_allocation_floor() {
        let pool = CpuWorkspacePool::new(0, CpuWorkspaceAllocator);
        let lease = pool.acquire(&[workspace_request(0, 16).unwrap()]).unwrap();
        assert_eq!(lease.requested_bytes(), 1);
        assert_eq!(lease.actual_bytes(), 1);
        assert_eq!(pool.stats().leased_bytes, 1);
        drop(lease);
        assert_eq!(pool.stats().idle_bytes, 0);
        assert_eq!(pool.stats().idle_segments, 0);
    }
}

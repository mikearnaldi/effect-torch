use crate::storage::{CpuSegment, CPU_STORAGE_ALIGNMENT};
use effect_torch_runtime::{
    NativeMemorySpace, WorkspaceAllocation, WorkspaceAllocator, WorkspaceLease, WorkspacePool,
    WorkspaceRequest,
};
use std::sync::{Arc, OnceLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CpuWorkspaceKey {
    pub memory_space: NativeMemorySpace,
    pub alignment: usize,
    pub capacity_class: usize,
}

impl CpuWorkspaceKey {
    pub fn new(bytes: usize, alignment: usize) -> Result<Self, String> {
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

#[derive(Debug, Default)]
pub struct CpuWorkspaceAllocator;

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

pub type CpuWorkspacePool = WorkspacePool<CpuWorkspaceKey, CpuWorkspaceAllocator>;
pub type CpuWorkspaceLease = WorkspaceLease<CpuWorkspaceKey, CpuWorkspaceAllocator>;

pub fn workspace_pool() -> &'static CpuWorkspacePool {
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

pub fn workspace_request(
    bytes: usize,
    alignment: usize,
) -> Result<WorkspaceRequest<CpuWorkspaceKey>, String> {
    Ok(WorkspaceRequest::new(
        CpuWorkspaceKey::new(bytes, alignment)?,
        bytes,
    ))
}

pub fn default_workspace_request(bytes: usize) -> WorkspaceRequest<CpuWorkspaceKey> {
    workspace_request(bytes, CPU_STORAGE_ALIGNMENT)
        .expect("default CPU workspace alignment is valid")
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

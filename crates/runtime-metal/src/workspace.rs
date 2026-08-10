use crate::device::{Buffer, BufferRetention, MetalDevice};
use effect_torch_runtime::{
    NativeMemorySpace, SegmentDecl, SegmentOwnership, WorkspaceAllocation, WorkspaceAllocator,
    WorkspaceLease, WorkspacePool, WorkspaceRequest,
};
use objc2_metal::MTLDevice;
use std::sync::{Arc, OnceLock};

#[cfg(test)]
pub(crate) const DEFAULT_ALIGNMENT: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalWorkspaceKey {
    pub memory_space: NativeMemorySpace,
    pub alignment: usize,
    pub capacity_class: usize,
}

impl MetalWorkspaceKey {
    fn new(bytes: usize, alignment: usize) -> Result<Self, String> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(format!("invalid Metal workspace alignment {alignment}"));
        }
        let capacity_class = bytes
            .checked_next_multiple_of(alignment)
            .ok_or_else(|| "Metal workspace capacity class overflow".to_string())?;
        Ok(Self {
            memory_space: NativeMemorySpace::MetalShared,
            alignment,
            capacity_class,
        })
    }
}

#[derive(Debug)]
pub(crate) struct MetalWorkspaceAllocator;

impl WorkspaceAllocator<MetalWorkspaceKey> for MetalWorkspaceAllocator {
    type Workspace = Arc<Buffer>;
    type Error = String;

    fn allocate(
        &mut self,
        key: &MetalWorkspaceKey,
        minimum_bytes: usize,
    ) -> Result<WorkspaceAllocation<Self::Workspace>, Self::Error> {
        if key.memory_space != NativeMemorySpace::MetalShared {
            return Err(format!(
                "unsupported executable Metal memory space {:?}",
                key.memory_space
            ));
        }
        let buffer = MetalDevice::get().alloc_raw_checked(minimum_bytes)?;
        let capacity = buffer.size;
        Ok(WorkspaceAllocation::new(buffer, capacity))
    }
}

pub(crate) type MetalWorkspacePool = WorkspacePool<MetalWorkspaceKey, MetalWorkspaceAllocator>;
pub(crate) type MetalWorkspaceLease = WorkspaceLease<MetalWorkspaceKey, MetalWorkspaceAllocator>;

pub(crate) fn workspace_pool() -> &'static MetalWorkspacePool {
    static POOL: OnceLock<MetalWorkspacePool> = OnceLock::new();
    POOL.get_or_init(|| {
        let recommended = MetalDevice::get().raw().recommendedMaxWorkingSetSize() as usize;
        let default_limit = recommended / 4;
        let max_idle_bytes = std::env::var("EFFECT_TORCH_WORKSPACE_POOL_MB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .and_then(|mb| mb.checked_mul(1024 * 1024))
            .unwrap_or(default_limit);
        WorkspacePool::new(max_idle_bytes, MetalWorkspaceAllocator)
    })
}

fn request(bytes: usize, alignment: usize) -> Result<WorkspaceRequest<MetalWorkspaceKey>, String> {
    let bytes = bytes.max(1);
    Ok(WorkspaceRequest::new(
        MetalWorkspaceKey::new(bytes, alignment)?,
        bytes,
    ))
}

pub(crate) struct InvocationResources {
    pub segments: Vec<Arc<Buffer>>,
    pub retentions: Vec<Option<BufferRetention>>,
    _workspace: MetalWorkspaceLease,
    pub actual_workspace_bytes: usize,
}

pub(crate) fn acquire(
    segments: &[SegmentDecl<NativeMemorySpace>],
) -> Result<InvocationResources, String> {
    let mut workspace_indices = Vec::new();
    let mut workspace_requests = Vec::new();
    for (index, segment) in segments.iter().enumerate() {
        if segment.memory_space != NativeMemorySpace::MetalShared {
            return Err(format!(
                "unsupported executable Metal memory space {:?} for segment {index}",
                segment.memory_space
            ));
        }
        if !matches!(segment.ownership, SegmentOwnership::ProvisionalOutput) {
            workspace_indices.push(index);
            workspace_requests.push(request(segment.bytes, segment.alignment)?);
        }
    }
    let workspace = workspace_pool()
        .acquire_set(&workspace_requests)
        .map_err(|error| format!("Metal workspace acquisition failed: {error}"))?;
    let mut owners: Vec<Option<Arc<Buffer>>> = std::iter::repeat_with(|| None)
        .take(segments.len())
        .collect();
    let mut retentions: Vec<Option<BufferRetention>> = std::iter::repeat_with(|| None)
        .take(segments.len())
        .collect();
    let mut actual_workspace_bytes = 0usize;
    for (&index, leased) in workspace_indices.iter().zip(workspace.segments()) {
        owners[index] = Some(Arc::clone(leased.workspace()));
        if matches!(
            segments[index].ownership,
            SegmentOwnership::Workspace | SegmentOwnership::InvocationStaging
        ) {
            actual_workspace_bytes = actual_workspace_bytes
                .checked_add(leased.capacity())
                .ok_or_else(|| "Metal workspace byte size overflow".to_string())?;
        }
    }
    for (index, segment) in segments.iter().enumerate() {
        if !matches!(segment.ownership, SegmentOwnership::ProvisionalOutput) {
            continue;
        }
        let output_request = request(segment.bytes, segment.alignment)?;
        let lease = Arc::new(
            workspace_pool()
                .acquire(std::slice::from_ref(&output_request))
                .map_err(|error| format!("Metal output acquisition failed: {error}"))?,
        );
        let owner = Arc::clone(lease.segments()[0].workspace());
        let retention: BufferRetention = lease;
        owners[index] = Some(owner);
        retentions[index] = Some(retention);
    }
    let segments = owners
        .into_iter()
        .enumerate()
        .map(|(index, owner)| {
            owner.ok_or_else(|| format!("Metal memory segment {index} was not acquired"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(InvocationResources {
        segments,
        retentions,
        _workspace: workspace,
        actual_workspace_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_segment_pool_is_best_fit_lru_bounded_and_atomic() {
        let pool = MetalWorkspacePool::new(80, MetalWorkspaceAllocator);
        let key = MetalWorkspaceKey {
            memory_space: NativeMemorySpace::MetalShared,
            alignment: DEFAULT_ALIGNMENT,
            capacity_class: 256,
        };
        let small = pool.acquire(&[WorkspaceRequest::new(key, 32)]).unwrap();
        let small_address = small.segments()[0].workspace().contents_ptr() as usize;
        let large = pool.acquire(&[WorkspaceRequest::new(key, 48)]).unwrap();
        drop(small);
        drop(large);

        let reused = pool.acquire(&[WorkspaceRequest::new(key, 24)]).unwrap();
        assert_eq!(
            reused.segments()[0].workspace().contents_ptr() as usize,
            small_address
        );
        assert_eq!(reused.segments()[0].capacity(), 32);
        drop(reused);
        drop(pool.acquire(&[WorkspaceRequest::new(key, 64)]).unwrap());
        assert!(pool.stats().idle_bytes <= 80);
        assert_eq!(pool.stats().leased_segments, 0);
    }

    #[test]
    fn small_plan_never_reuses_an_oversized_capacity_class() {
        let pool = MetalWorkspacePool::new(4 << 20, MetalWorkspaceAllocator);
        let large_key = MetalWorkspaceKey {
            memory_space: NativeMemorySpace::MetalShared,
            alignment: DEFAULT_ALIGNMENT,
            capacity_class: 2 << 20,
        };
        let small_key = MetalWorkspaceKey {
            capacity_class: 1 << 20,
            ..large_key
        };
        let large = pool
            .acquire(&[WorkspaceRequest::new(large_key, 2 << 20)])
            .unwrap();
        let large_address = large.segments()[0].workspace().contents_ptr() as usize;
        drop(large);

        let small = pool
            .acquire(&[WorkspaceRequest::new(small_key, 1 << 20)])
            .unwrap();
        assert_ne!(
            small.segments()[0].workspace().contents_ptr() as usize,
            large_address
        );
        assert_eq!(small.segments()[0].capacity(), 1 << 20);
    }

    #[test]
    fn output_views_retain_the_pool_lease() {
        let pool = MetalWorkspacePool::new(1024, MetalWorkspaceAllocator);
        let request = WorkspaceRequest::new(
            MetalWorkspaceKey {
                memory_space: NativeMemorySpace::MetalShared,
                alignment: DEFAULT_ALIGNMENT,
                capacity_class: DEFAULT_ALIGNMENT,
            },
            16,
        );
        let lease = Arc::new(pool.acquire(&[request]).unwrap());
        let root = Arc::clone(lease.segments()[0].workspace());
        let retention: BufferRetention = lease;
        let output = Arc::new(Buffer::suballoc_with_retention(
            &root,
            0,
            16,
            Some(retention),
        ));
        drop(root);
        let view = Arc::new(Buffer::suballoc(&output, 0, 8));
        assert_eq!(pool.stats().leased_segments, 1);
        drop(output);
        assert_eq!(pool.stats().leased_segments, 1);
        drop(view);
        assert_eq!(pool.stats().leased_segments, 0);
        assert_eq!(pool.stats().idle_segments, 1);
    }
}

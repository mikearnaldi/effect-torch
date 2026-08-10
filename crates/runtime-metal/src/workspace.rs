use crate::device::{Buffer, MetalDevice};
use effect_torch_runtime::{NativeMemorySpace, SegmentOwnership};
#[cfg(test)]
use effect_torch_runtime::{
    WorkspaceAllocation, WorkspaceAllocator, WorkspacePool, WorkspaceRequest,
};
#[cfg(test)]
use objc2_metal::MTLDevice;
use std::sync::Arc;
#[cfg(test)]
use std::sync::OnceLock;

#[cfg(test)]
pub(crate) const DEFAULT_ALIGNMENT: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
pub(crate) struct MetalWorkspaceKey {
    pub memory_space: NativeMemorySpace,
    pub alignment: usize,
    pub capacity_class: usize,
}

#[derive(Debug)]
#[cfg(test)]
pub(crate) struct MetalWorkspaceAllocator;

#[cfg(test)]
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

#[cfg(test)]
pub(crate) type MetalWorkspacePool = WorkspacePool<MetalWorkspaceKey, MetalWorkspaceAllocator>;

#[cfg(test)]
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

pub(crate) struct InvocationResources {
    pub segments: Vec<Arc<Buffer>>,
    pub actual_workspace_bytes: usize,
}

pub(crate) struct ExecutableResources {
    fixed: Box<[Option<Arc<Buffer>>]>,
    outputs: Box<[Box<[Option<Arc<Buffer>>]>]>,
    actual_workspace_bytes: usize,
}

impl ExecutableResources {
    pub fn prepare(
        segments: &[effect_torch_runtime::SegmentDecl<NativeMemorySpace>],
        output_capacity: usize,
    ) -> Result<Self, String> {
        if output_capacity == 0 {
            return Err("Metal executable output capacity must be positive".to_string());
        }
        let mut fixed = Vec::with_capacity(segments.len());
        let mut actual_workspace_bytes = 0usize;
        for (index, segment) in segments.iter().enumerate() {
            if segment.memory_space != NativeMemorySpace::MetalShared {
                return Err(format!(
                    "unsupported executable Metal memory space {:?} for segment {index}",
                    segment.memory_space
                ));
            }
            if matches!(
                segment.ownership,
                SegmentOwnership::Workspace | SegmentOwnership::InvocationStaging
            ) {
                actual_workspace_bytes = actual_workspace_bytes
                    .checked_add(segment.bytes)
                    .ok_or_else(|| "Metal executable workspace byte size overflow".to_string())?;
            }
            fixed.push(match segment.ownership {
                SegmentOwnership::ProvisionalOutput => None,
                SegmentOwnership::Workspace
                | SegmentOwnership::InvocationStaging
                | SegmentOwnership::StateTransaction => {
                    Some(MetalDevice::get().alloc_raw_checked(segment.bytes.max(1))?)
                }
            });
        }

        let output_generations = if segments
            .iter()
            .any(|segment| matches!(segment.ownership, SegmentOwnership::ProvisionalOutput))
        {
            output_capacity
        } else {
            1
        };
        let outputs = (0..output_generations)
            .map(|_| {
                segments
                    .iter()
                    .map(|segment| {
                        matches!(segment.ownership, SegmentOwnership::ProvisionalOutput)
                            .then(|| MetalDevice::get().alloc_raw_checked(segment.bytes.max(1)))
                            .transpose()
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map(Vec::into_boxed_slice)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            fixed: fixed.into_boxed_slice(),
            outputs: outputs.into_boxed_slice(),
            actual_workspace_bytes,
        })
    }

    pub fn acquire(&self) -> Result<InvocationResources, String> {
        let outputs = self
            .outputs
            .iter()
            .find(|generation| {
                generation.iter().flatten().all(|buffer| Arc::strong_count(buffer) == 1)
            })
            .ok_or_else(|| {
                format!(
                    "Metal executable resource capacity exhausted: all {} preallocated output generations are still live",
                    self.outputs.len()
                )
            })?;
        let segments = self
            .fixed
            .iter()
            .zip(outputs)
            .enumerate()
            .map(|(index, (fixed, output))| {
                fixed
                    .as_ref()
                    .or(output.as_ref())
                    .cloned()
                    .ok_or_else(|| format!("Metal memory segment {index} was not prepared"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(InvocationResources {
            segments,
            actual_workspace_bytes: self.actual_workspace_bytes,
        })
    }
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
}

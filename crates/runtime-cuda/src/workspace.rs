//! Device-local pooling for compiler-planned CUDA memory segments.

use crate::buffer::CudaBuffer;
use crate::CudaDevice;
use cudarc::driver::CudaSlice;
use effect_torch_runtime::{
    Location, SegmentDecl, SegmentOwnership, WorkspaceAllocation, WorkspaceAllocator,
    WorkspaceLease, WorkspacePool, WorkspaceRequest,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub(crate) const CUDA_STORAGE_ALIGNMENT: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CudaMemorySpace {
    Device,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CudaWorkspaceKey {
    device_ordinal: u32,
    memory_space: CudaMemorySpace,
    alignment: usize,
    capacity_class: usize,
}

impl CudaWorkspaceKey {
    fn new(device_ordinal: u32, bytes: usize, alignment: usize) -> Result<Self, String> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(format!("invalid CUDA workspace alignment {alignment}"));
        }
        let capacity_class = bytes
            .checked_next_multiple_of(alignment)
            .ok_or_else(|| "CUDA workspace capacity class overflow".to_string())?;
        Ok(Self {
            device_ordinal,
            memory_space: CudaMemorySpace::Device,
            alignment,
            capacity_class,
        })
    }
}

#[derive(Debug, Default)]
pub(crate) struct CudaWorkspaceAllocator;

impl WorkspaceAllocator<CudaWorkspaceKey> for CudaWorkspaceAllocator {
    type Workspace = Arc<CudaSlice<u8>>;
    type Error = String;

    fn allocate(
        &mut self,
        key: &CudaWorkspaceKey,
        minimum_bytes: usize,
    ) -> Result<WorkspaceAllocation<Self::Workspace>, Self::Error> {
        if key.memory_space != CudaMemorySpace::Device {
            return Err("unsupported CUDA workspace memory space".to_string());
        }
        if minimum_bytes > key.capacity_class {
            return Err(format!(
                "CUDA workspace request of {minimum_bytes} bytes exceeds capacity class {}",
                key.capacity_class
            ));
        }
        let device = CudaDevice::get(key.device_ordinal)?;
        let buffer = unsafe { device.stream.alloc::<u8>(minimum_bytes.max(1)) }
            .map_err(|error| error.to_string())?;
        let capacity = buffer.len();
        Ok(WorkspaceAllocation::new(Arc::new(buffer), capacity))
    }
}

type CudaWorkspacePool = WorkspacePool<CudaWorkspaceKey, CudaWorkspaceAllocator>;
type CudaWorkspaceLease = WorkspaceLease<CudaWorkspaceKey, CudaWorkspaceAllocator>;

fn workspace_pool(device_ordinal: u32) -> Result<&'static CudaWorkspacePool, String> {
    static POOLS: OnceLock<Mutex<HashMap<u32, &'static CudaWorkspacePool>>> = OnceLock::new();
    let pools = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut pools = pools.lock().unwrap_or_else(|error| error.into_inner());
    if let Some(pool) = pools.get(&device_ordinal) {
        return Ok(*pool);
    }
    let device = CudaDevice::get(device_ordinal)?;
    let default_limit = device
        .stream
        .context()
        .total_mem()
        .map_err(|error| error.to_string())?
        / 4;
    let max_idle_bytes = std::env::var("EFFECT_TORCH_CUDA_WORKSPACE_POOL_MB")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .and_then(|megabytes| megabytes.checked_mul(1024 * 1024))
        .unwrap_or(default_limit);
    let pool = Box::leak(Box::new(CudaWorkspacePool::new(
        max_idle_bytes,
        CudaWorkspaceAllocator,
    )));
    pools.insert(device_ordinal, pool);
    Ok(pool)
}

fn request(
    device_ordinal: u32,
    segment: &SegmentDecl<CudaMemorySpace>,
) -> Result<WorkspaceRequest<CudaWorkspaceKey>, String> {
    let bytes = segment.bytes.max(1);
    Ok(WorkspaceRequest::new(
        CudaWorkspaceKey::new(device_ordinal, bytes, segment.alignment)?,
        bytes,
    ))
}

struct SegmentAllocation {
    owner: Arc<CudaSlice<u8>>,
    retention: Option<Arc<dyn Send + Sync>>,
    full_buffer: Option<CudaBuffer<u8>>,
}

pub(crate) struct InvocationResources {
    segments: Box<[SegmentAllocation]>,
    selected71: Option<Box<[bool]>>,
    _workspace: CudaWorkspaceLease,
    pub(crate) _actual_workspace_bytes: usize,
}

impl InvocationResources {
    /// Pure range validation against the current lease, without querying any
    /// device pointer or materializing a buffer/event. Used before graph admission.
    pub(crate) fn preflight_graph61(
        &self,
        location: &Location,
        extra: usize,
        bytes: usize,
    ) -> Result<(), String> {
        let Location::Segment {
            segment,
            offset,
            bytes: available,
        } = location
        else {
            return Err("graph61 requires planned segment storage".into());
        };
        if self
            .selected71
            .as_ref()
            .is_some_and(|selected| !selected.get(segment.index()).copied().unwrap_or(false))
        {
            return Err("whole-read71 segment is outside the admitted body".into());
        }
        let allocation = self
            .segments
            .get(segment.index())
            .ok_or("graph61 missing current lease")?;
        validate_graph61_range(*offset, extra, bytes, *available, allocation.owner.len())
    }
    pub(crate) fn buffer<T: Send + Sync + 'static>(
        &self,
        location: &Location,
        additional_byte_offset: usize,
        len: usize,
    ) -> Result<CudaBuffer<T>, String> {
        let Location::Segment {
            segment,
            offset,
            bytes,
        } = location
        else {
            return Err(format!(
                "CUDA planned output requires a segment location, found {location:?}"
            ));
        };
        if self
            .selected71
            .as_ref()
            .is_some_and(|selected| !selected.get(segment.index()).copied().unwrap_or(false))
        {
            return Err("whole-read71 segment is outside the admitted body".into());
        }
        let requested_bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| "CUDA planned buffer byte size overflowed usize".to_string())?;
        let requested_end = additional_byte_offset
            .checked_add(requested_bytes)
            .ok_or_else(|| "CUDA planned buffer range overflowed usize".to_string())?;
        if requested_end > *bytes {
            return Err(format!(
                "CUDA planned buffer range 0..{requested_end} exceeds value allocation {bytes}"
            ));
        }
        let byte_offset = offset
            .checked_add(additional_byte_offset)
            .ok_or_else(|| "CUDA planned segment offset overflowed usize".to_string())?;
        let allocation = self
            .segments
            .get(segment.index())
            .ok_or_else(|| format!("CUDA memory segment {segment} is out of range"))?;
        if let Some(buffer) = &allocation.full_buffer {
            return buffer.planned_view(byte_offset, len);
        }
        CudaBuffer::from_segment(
            Arc::clone(&allocation.owner),
            byte_offset,
            len,
            allocation.retention.clone(),
        )
    }
}

pub(crate) fn acquire(
    device_ordinal: u32,
    segments: &[SegmentDecl<CudaMemorySpace>],
) -> Result<InvocationResources, String> {
    let cache_segment_buffers =
        std::env::var("EFFECT_TORCH_CUDA_CACHE_SEGMENT_BUFFERS").as_deref() == Ok("1");
    acquire_with_policy(device_ordinal, segments, cache_segment_buffers)
}

/// A separately retained private replay allocation, charged to the same device
/// pool as invocation segments. It cannot escape as a caller output.
pub(crate) fn acquire_private71(
    device_ordinal: u32,
    bytes: usize,
) -> Result<CudaBuffer<u8>, String> {
    let segment = SegmentDecl {
        bytes,
        alignment: CUDA_STORAGE_ALIGNMENT,
        memory_space: CudaMemorySpace::Device,
        ownership: SegmentOwnership::InvocationStaging,
    };
    let lease = Arc::new(
        workspace_pool(device_ordinal)?
            .acquire(std::slice::from_ref(&request(device_ordinal, &segment)?))
            .map_err(|error| format!("CUDA private replay acquisition failed: {error}"))?,
    );
    let owner = Arc::clone(lease.segments()[0].workspace());
    CudaBuffer::from_segment(owner, 0, bytes, Some(lease))
}

pub(crate) fn acquire_private_segments71(
    device_ordinal: u32,
    segments: &[SegmentDecl<CudaMemorySpace>],
    selected: &[effect_torch_runtime::SegmentId],
) -> Result<InvocationResources, String> {
    let mut included = vec![false; segments.len()];
    for id in selected {
        *included
            .get_mut(id.index())
            .ok_or("whole-read71 selected segment out of bounds")? = true;
    }
    let declarations = segments
        .iter()
        .zip(&included)
        .map(|(segment, included)| {
            let mut segment = segment.clone();
            if !included {
                // Preserve original indices without retaining any excluded arena.
                // The pool's minimum allocation is one byte; buffer() rejects access.
                segment.bytes = 0;
                segment.ownership = SegmentOwnership::Workspace;
            }
            segment
        })
        .collect::<Vec<_>>();
    let mut resources = acquire_with_policy(device_ordinal, &declarations, true)?;
    resources.selected71 = Some(included.into_boxed_slice());
    Ok(resources)
}

fn acquire_with_policy(
    device_ordinal: u32,
    segments: &[SegmentDecl<CudaMemorySpace>],
    cache_segment_buffers: bool,
) -> Result<InvocationResources, String> {
    let pool = workspace_pool(device_ordinal)?;
    let mut workspace_indices = Vec::new();
    let mut workspace_requests = Vec::new();
    for (index, segment) in segments.iter().enumerate() {
        if segment.memory_space != CudaMemorySpace::Device {
            return Err(format!("unsupported CUDA memory space for segment {index}"));
        }
        if !matches!(
            segment.ownership,
            SegmentOwnership::ProvisionalOutput | SegmentOwnership::StateTransaction
        ) {
            workspace_indices.push(index);
            workspace_requests.push(request(device_ordinal, segment)?);
        }
    }
    let workspace = pool
        .acquire_set(&workspace_requests)
        .map_err(|error| format!("CUDA workspace acquisition failed: {error}"))?;
    let mut allocations: Vec<Option<SegmentAllocation>> = std::iter::repeat_with(|| None)
        .take(segments.len())
        .collect();
    let mut actual_workspace_bytes = 0usize;
    for (&index, leased) in workspace_indices.iter().zip(workspace.segments()) {
        allocations[index] = Some(SegmentAllocation {
            owner: Arc::clone(leased.workspace()),
            retention: None,
            full_buffer: None,
        });
        if matches!(
            segments[index].ownership,
            SegmentOwnership::Workspace | SegmentOwnership::InvocationStaging
        ) {
            actual_workspace_bytes = actual_workspace_bytes
                .checked_add(leased.capacity())
                .ok_or_else(|| "CUDA workspace byte size overflow".to_string())?;
        }
    }
    for (index, segment) in segments.iter().enumerate() {
        if !matches!(
            segment.ownership,
            SegmentOwnership::ProvisionalOutput | SegmentOwnership::StateTransaction
        ) {
            continue;
        }
        let lease = Arc::new(
            pool.acquire(std::slice::from_ref(&request(device_ordinal, segment)?))
                .map_err(|error| format!("CUDA output acquisition failed: {error}"))?,
        );
        let owner = Arc::clone(lease.segments()[0].workspace());
        let retention: Arc<dyn Send + Sync> = lease;
        allocations[index] = Some(SegmentAllocation {
            owner,
            retention: Some(retention),
            full_buffer: None,
        });
    }
    let segments = allocations
        .into_iter()
        .enumerate()
        .map(|(index, allocation)| {
            let mut allocation = allocation
                .ok_or_else(|| format!("CUDA memory segment {index} was not acquired"))?;
            if cache_segment_buffers {
                allocation.full_buffer = Some(CudaBuffer::from_segment(
                    Arc::clone(&allocation.owner),
                    0,
                    allocation.owner.len(),
                    allocation.retention.clone(),
                )?);
            }
            Ok::<_, String>(allocation)
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_boxed_slice();
    Ok(InvocationResources {
        segments,
        selected71: None,
        _workspace: workspace,
        _actual_workspace_bytes: actual_workspace_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_keys_include_device_and_capacity_class() {
        let first = CudaWorkspaceKey::new(0, 257, CUDA_STORAGE_ALIGNMENT).unwrap();
        let same_class = CudaWorkspaceKey::new(0, 300, CUDA_STORAGE_ALIGNMENT).unwrap();
        let other_device = CudaWorkspaceKey::new(1, 257, CUDA_STORAGE_ALIGNMENT).unwrap();
        assert_eq!(first, same_class);
        assert_ne!(first, other_device);
        assert_eq!(first.capacity_class, 512);
    }
}

#[cfg(test)]
#[path = "workspace_tests.rs"]
mod planned_view_tests;

fn validate_graph61_range(
    offset: usize,
    extra: usize,
    bytes: usize,
    available: usize,
    capacity: usize,
) -> Result<(), String> {
    let end = extra
        .checked_add(bytes)
        .ok_or("graph61 value range overflow")?;
    if end > available {
        return Err("graph61 value exceeds planned range".into());
    }
    let end = offset
        .checked_add(end)
        .ok_or("graph61 segment range overflow")?;
    if end > capacity {
        return Err("graph61 value exceeds current lease".into());
    }
    Ok(())
}
#[cfg(test)]
mod graph61_range_tests {
    use super::validate_graph61_range;
    #[test]
    fn graph61_pure_range_checks_alias_bounds_current_capacity_and_overflow() {
        assert!(validate_graph61_range(256, 16, 64, 80, 336).is_ok());
        for values in [
            (256, 17, 64, 80, 1000),
            (256, 16, 64, 80, 335),
            (0, usize::MAX, 1, usize::MAX, usize::MAX),
            (usize::MAX, 0, 1, 1, usize::MAX),
        ] {
            assert!(
                validate_graph61_range(values.0, values.1, values.2, values.3, values.4).is_err()
            );
        }
    }
}

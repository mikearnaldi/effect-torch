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
}

pub(crate) struct InvocationResources {
    segments: Box<[SegmentAllocation]>,
    _workspace: CudaWorkspaceLease,
    pub(crate) _actual_workspace_bytes: usize,
}

impl InvocationResources {
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
    let pool = workspace_pool(device_ordinal)?;
    let mut workspace_indices = Vec::new();
    let mut workspace_requests = Vec::new();
    for (index, segment) in segments.iter().enumerate() {
        if segment.memory_space != CudaMemorySpace::Device {
            return Err(format!("unsupported CUDA memory space for segment {index}"));
        }
        if segment.ownership != SegmentOwnership::ProvisionalOutput {
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
        if segment.ownership != SegmentOwnership::ProvisionalOutput {
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
        });
    }
    let segments = allocations
        .into_iter()
        .enumerate()
        .map(|(index, allocation)| {
            allocation.ok_or_else(|| format!("CUDA memory segment {index} was not acquired"))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_boxed_slice();
    Ok(InvocationResources {
        segments,
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

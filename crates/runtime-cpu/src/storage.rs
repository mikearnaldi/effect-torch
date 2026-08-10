use half::{bf16, f16};
use std::alloc::{alloc_zeroed, dealloc, Layout as AllocationLayout};
use std::cell::Cell;
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;

pub const CPU_STORAGE_ALIGNMENT: usize = 64;
pub(crate) type CpuStorageRetention = Arc<dyn Send + Sync + std::panic::RefUnwindSafe>;

thread_local! {
    static EXECUTABLE_ALLOCATION_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Rejects ambient CPU tensor storage allocation on the current executor thread.
/// Segment leases must be acquired before entering the guard.
#[derive(Debug)]
pub struct ExecutableAllocationGuard {
    _thread_owned: PhantomData<Rc<()>>,
}

impl ExecutableAllocationGuard {
    pub fn enter() -> Self {
        EXECUTABLE_ALLOCATION_DEPTH.with(|depth| {
            depth.set(
                depth
                    .get()
                    .checked_add(1)
                    .expect("CPU executable allocation guard depth overflow"),
            );
        });
        Self {
            _thread_owned: PhantomData,
        }
    }

    pub fn is_active() -> bool {
        EXECUTABLE_ALLOCATION_DEPTH.with(|depth| depth.get() != 0)
    }
}

impl Drop for ExecutableAllocationGuard {
    fn drop(&mut self) {
        EXECUTABLE_ALLOCATION_DEPTH.with(|depth| {
            depth.set(
                depth
                    .get()
                    .checked_sub(1)
                    .expect("CPU executable allocation guard underflow"),
            );
        });
    }
}

pub(crate) fn assert_allocation_allowed(operation: &str) {
    assert!(
        !ExecutableAllocationGuard::is_active(),
        "{operation}: CPU tensor allocation is forbidden during executable execution"
    );
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuStorageError {
    InvalidAlignment(usize),
    AllocationFailed { bytes: usize, alignment: usize },
    ByteRangeOverflow,
    MisalignedView { offset: usize, alignment: usize },
    ViewOutOfBounds { end: usize, capacity: usize },
}

impl fmt::Display for CpuStorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAlignment(alignment) => {
                write!(f, "invalid CPU storage alignment {alignment}")
            }
            Self::AllocationFailed { bytes, alignment } => write!(
                f,
                "failed to allocate {bytes} bytes of CPU storage aligned to {alignment} bytes"
            ),
            Self::ByteRangeOverflow => f.write_str("CPU storage byte range overflow"),
            Self::MisalignedView { offset, alignment } => write!(
                f,
                "CPU storage view at byte offset {offset} is not aligned to {alignment} bytes"
            ),
            Self::ViewOutOfBounds { end, capacity } => write!(
                f,
                "CPU storage view ends at byte {end}, beyond segment capacity {capacity}"
            ),
        }
    }
}

impl Error for CpuStorageError {}

pub struct CpuSegment {
    pointer: NonNull<u8>,
    capacity: usize,
    allocation_bytes: usize,
    alignment: usize,
}

impl CpuSegment {
    pub fn new(bytes: usize, alignment: usize) -> Result<Arc<Self>, CpuStorageError> {
        Self::allocate(bytes, alignment)
    }

    pub fn allocate(bytes: usize, alignment: usize) -> Result<Arc<Self>, CpuStorageError> {
        assert_allocation_allowed("CpuSegment::allocate");
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(CpuStorageError::InvalidAlignment(alignment));
        }
        let allocation_bytes = bytes.max(1);
        let layout = AllocationLayout::from_size_align(allocation_bytes, alignment)
            .map_err(|_| CpuStorageError::InvalidAlignment(alignment))?;
        // SAFETY: `layout` is valid and the returned allocation is owned by the segment.
        let pointer = NonNull::new(unsafe { alloc_zeroed(layout) })
            .ok_or(CpuStorageError::AllocationFailed { bytes, alignment })?;
        Ok(Arc::new(Self {
            pointer,
            capacity: bytes,
            allocation_bytes,
            alignment,
        }))
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.pointer.as_ptr()
    }

    #[cfg(test)]
    pub(crate) fn view<T: CpuElement>(
        self: &Arc<Self>,
        byte_offset: usize,
        len: usize,
    ) -> Result<CpuStorage<T>, CpuStorageError> {
        CpuStorage::from_segment(Arc::clone(self), byte_offset, len)
    }
}

impl fmt::Debug for CpuSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuSegment")
            .field("pointer", &self.pointer)
            .field("capacity", &self.capacity)
            .field("alignment", &self.alignment)
            .finish()
    }
}

impl Drop for CpuSegment {
    fn drop(&mut self) {
        let layout = AllocationLayout::from_size_align(self.allocation_bytes, self.alignment)
            .expect("CPU segment retained its validated allocation layout");
        // SAFETY: this pointer was allocated with `layout` and this is its only deallocation.
        unsafe { dealloc(self.pointer.as_ptr(), layout) };
    }
}

// SAFETY: public views are immutable. Interior writes are available only through
// CpuDestination; planned writes additionally require executor liveness to prove
// that no read or write overlaps this range on another thread.
unsafe impl Send for CpuSegment {}
unsafe impl Sync for CpuSegment {}

mod sealed {
    pub trait Sealed {}
}

pub trait CpuElement: sealed::Sealed + Copy + Default + Send + Sync + 'static {}

macro_rules! cpu_elements {
    ($($type:ty),+ $(,)?) => {
        $(
            impl sealed::Sealed for $type {}
            impl CpuElement for $type {}
        )+
    };
}

cpu_elements!(f32, f64, f16, bf16, u8, u32, i64);

#[derive(Clone)]
pub struct CpuStorage<T: CpuElement> {
    owner: Arc<CpuSegment>,
    retention: Option<CpuStorageRetention>,
    byte_offset: usize,
    len: usize,
    _element: PhantomData<T>,
}

impl<T: CpuElement> CpuStorage<T> {
    pub fn from_segment(
        owner: Arc<CpuSegment>,
        byte_offset: usize,
        len: usize,
    ) -> Result<Self, CpuStorageError> {
        Self::from_segment_with_retention(owner, byte_offset, len, None)
    }

    pub(crate) fn from_segment_with_retention(
        owner: Arc<CpuSegment>,
        byte_offset: usize,
        len: usize,
        retention: Option<CpuStorageRetention>,
    ) -> Result<Self, CpuStorageError> {
        let byte_len = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(CpuStorageError::ByteRangeOverflow)?;
        let end = byte_offset
            .checked_add(byte_len)
            .ok_or(CpuStorageError::ByteRangeOverflow)?;
        let pointer = (owner.as_ptr() as usize)
            .checked_add(byte_offset)
            .ok_or(CpuStorageError::ByteRangeOverflow)?;
        if pointer % std::mem::align_of::<T>() != 0 {
            return Err(CpuStorageError::MisalignedView {
                offset: byte_offset,
                alignment: std::mem::align_of::<T>(),
            });
        }
        if end > owner.capacity() {
            return Err(CpuStorageError::ViewOutOfBounds {
                end,
                capacity: owner.capacity(),
            });
        }
        Ok(Self {
            owner,
            retention,
            byte_offset,
            len,
            _element: PhantomData,
        })
    }

    pub fn from_slice(values: &[T]) -> Result<Self, CpuStorageError> {
        let bytes = values
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(CpuStorageError::ByteRangeOverflow)?;
        let owner = CpuSegment::allocate(bytes, CPU_STORAGE_ALIGNMENT)?;
        let storage = Self::from_segment(owner, 0, values.len())?;
        // SAFETY: the segment is newly allocated and unpublished, and both slices
        // contain exactly `values.len()` initialized values of the same type.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                storage.pointer().cast::<T>(),
                values.len(),
            );
        }
        Ok(storage)
    }

    pub fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    pub fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[T] {
        // SAFETY: construction validates type alignment and bounds. CpuElement is
        // sealed to types for which every stored bit pattern is valid.
        unsafe { std::slice::from_raw_parts(self.pointer().cast::<T>(), self.len) }
    }

    pub fn as_ptr(&self) -> *const T {
        self.pointer().cast::<T>()
    }

    pub(crate) fn is_uniquely_owned(&self) -> bool {
        Arc::strong_count(&self.owner) == 1
    }

    pub(crate) unsafe fn as_mut_slice_for_destination(&self) -> &mut [T] {
        // SAFETY: the caller holds a CpuDestination capability. Its constructor
        // either proves unique ownership or delegates non-overlap to executable
        // liveness for a planned segment.
        unsafe { std::slice::from_raw_parts_mut(self.pointer().cast::<T>(), self.len) }
    }

    fn pointer(&self) -> *mut u8 {
        // Raw allocation storage is the narrowly contained interior-mutation
        // mechanism; no mutable pointer is exposed by the public API.
        unsafe { self.owner.pointer.as_ptr().add(self.byte_offset) }
    }
}

impl<T: CpuElement> Deref for CpuStorage<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: CpuElement> AsRef<[T]> for CpuStorage<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: CpuElement + fmt::Debug> fmt::Debug for CpuStorage<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuStorage")
            .field("owner", &self.owner)
            .field("retained_lease", &self.retention.is_some())
            .field("byte_offset", &self.byte_offset)
            .field("values", &self.as_slice())
            .finish()
    }
}

impl<T: CpuElement + PartialEq> PartialEq for CpuStorage<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: CpuElement + Eq> Eq for CpuStorage<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_views_are_aligned_exact_subranges_and_retain_the_owner() {
        let segment = CpuSegment::allocate(128, 64).unwrap();
        let weak = Arc::downgrade(&segment);
        let first = segment.view::<f32>(0, 4).unwrap();
        let second = segment.view::<f32>(64, 4).unwrap();
        assert_eq!(first.byte_len(), 16);
        assert_eq!(second.byte_offset(), 64);
        assert_eq!(first.as_ptr() as usize % 64, 0);
        assert_eq!(second.as_ptr() as usize % 64, 0);
        assert!(matches!(
            segment.view::<f32>(127, 1),
            Err(CpuStorageError::MisalignedView { .. })
                | Err(CpuStorageError::ViewOutOfBounds { .. })
        ));

        drop(segment);
        assert!(weak.upgrade().is_some());
        drop(first);
        drop(second);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn executable_guard_is_nested_and_thread_owned() {
        assert!(!ExecutableAllocationGuard::is_active());
        let outer = ExecutableAllocationGuard::enter();
        assert!(ExecutableAllocationGuard::is_active());
        {
            let _inner = ExecutableAllocationGuard::enter();
            assert!(ExecutableAllocationGuard::is_active());
        }
        assert!(ExecutableAllocationGuard::is_active());
        drop(outer);
        assert!(!ExecutableAllocationGuard::is_active());
    }
}

//! Aligned raw memory and typed views for the CPU runtime.
//!
//! [`CpuSegment`] is the single owner of a heap allocation: a zeroed,
//! power-of-two-aligned byte range with an exact capacity. [`CpuStorage<T>`]
//! is a typed, bounds-checked view into a segment; many views may alias one
//! segment, and each view keeps the segment alive through an `Arc`. Views may
//! additionally carry a *retention* token (for example a workspace pool
//! lease) that is released only when the last view of the allocation drops.
//!
//! Mutation discipline: the public API of `CpuStorage` is read-only. Interior
//! writes are possible only through a `CpuDestination` capability (see
//! `crate::tensor`), whose constructors prove either unique ownership of the
//! segment or — inside a compiled executable — that the fixed liveness
//! schedule keeps the written range free of concurrent access.
//!
//! [`ExecutableAllocationGuard`] is a thread-local, nestable guard that makes
//! every ambient allocation entry point panic while active. Compiled
//! executables run under the guard to prove that all memory was acquired up
//! front from the workspace pool and that kernels never allocate on the
//! execution path.

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

/// Alignment of every tensor storage allocation: one cache line, which also
/// satisfies the widest SIMD loads used by downstream kernels.
pub const CPU_STORAGE_ALIGNMENT: usize = 64;
/// Opaque keep-alive token attached to a storage view, typically a workspace
/// pool lease. Dropped only when the last view of the segment drops.
pub(crate) type CpuStorageRetention = Arc<dyn Send + Sync + std::panic::RefUnwindSafe>;

thread_local! {
    static EXECUTABLE_ALLOCATION_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Rejects ambient CPU tensor storage allocation on the current executor thread.
/// Segment leases must be acquired before entering the guard.
///
/// The guard is nestable (a depth counter, not a flag) and bound to the
/// entering thread via `PhantomData<Rc<()>>`, so it is neither `Send` nor
/// `Sync` and cannot leak the restriction to — or from — another thread.
#[derive(Debug)]
pub struct ExecutableAllocationGuard {
    _thread_owned: PhantomData<Rc<()>>,
}

impl ExecutableAllocationGuard {
    /// Enters (or nests) the guard on this thread.
    ///
    /// # Panics
    /// On depth-counter overflow (practically unreachable).
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

    /// Whether any guard is currently held on this thread.
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

/// Panics if an [`ExecutableAllocationGuard`] is active on this thread.
/// Called by every tensor/segment allocation entry point.
pub(crate) fn assert_allocation_allowed(operation: &str) {
    assert!(
        !ExecutableAllocationGuard::is_active(),
        "{operation}: CPU tensor allocation is forbidden during executable execution"
    );
}

/// Errors produced when allocating a segment or carving a typed view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuStorageError {
    /// Alignment was zero or not a power of two (or not representable in a
    /// valid allocation layout).
    InvalidAlignment(usize),
    /// The global allocator returned null for the requested layout.
    AllocationFailed { bytes: usize, alignment: usize },
    /// A byte-range computation overflowed `usize`.
    ByteRangeOverflow,
    /// A view's byte offset does not satisfy the element type's alignment.
    MisalignedView { offset: usize, alignment: usize },
    /// A view's end would extend past the owning segment's capacity.
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

/// Owner of one aligned, zero-initialized heap allocation.
///
/// Segments are the unit the workspace pool recycles and the unit tensor
/// views reference-count. `capacity` is the logical byte size requested by
/// the caller; the physical allocation is at least one byte so zero-capacity
/// segments still own a valid, dereferenceable-to-zero-extent pointer.
pub struct CpuSegment {
    pointer: NonNull<u8>,
    capacity: usize,
    allocation_bytes: usize,
    alignment: usize,
}

impl CpuSegment {
    /// Allocates a segment; see [`CpuSegment::allocate`].
    pub fn new(bytes: usize, alignment: usize) -> Result<Arc<Self>, CpuStorageError> {
        Self::allocate(bytes, alignment)
    }

    /// Allocates `bytes` zeroed bytes at `alignment`.
    ///
    /// Panics under an active [`ExecutableAllocationGuard`]: executables must
    /// lease segments from the workspace pool before entering the guard.
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

    /// Logical capacity in bytes, as requested at allocation time.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Alignment the allocation was created with.
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Base address of the allocation.
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

/// Element types allowed in [`CpuStorage`].
///
/// Sealed to `f32`, `f64`, `f16`, `bf16`, `u8`, `u32`, and `i64`: plain-data
/// types for which every bit pattern is valid, so a raw byte range can always
/// be viewed as a slice of `T` without a validity invariant.
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

/// A typed, bounds-checked view of `len` elements of `T` at `byte_offset`
/// within an owned [`CpuSegment`].
///
/// Cloning a view aliases the same bytes; uniqueness is observable via
/// `is_uniquely_owned` and is what the safe destination constructor requires
/// before handing out mutable access.
#[derive(Clone)]
pub struct CpuStorage<T: CpuElement> {
    owner: Arc<CpuSegment>,
    retention: Option<CpuStorageRetention>,
    byte_offset: usize,
    len: usize,
    _element: PhantomData<T>,
}

impl<T: CpuElement> CpuStorage<T> {
    /// Creates a view into `owner`, validating element alignment and bounds.
    pub fn from_segment(
        owner: Arc<CpuSegment>,
        byte_offset: usize,
        len: usize,
    ) -> Result<Self, CpuStorageError> {
        Self::from_segment_with_retention(owner, byte_offset, len, None)
    }

    /// Like [`CpuStorage::from_segment`], additionally attaching a retention
    /// token that outlives individual views (used to pin workspace pool
    /// leases for as long as any published output aliases the segment).
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

    /// Copies `values` into a freshly allocated segment and returns the
    /// uniquely owned view covering them.
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

    /// Byte offset of this view within the owning segment.
    pub fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    /// Byte length of this view (`len * size_of::<T>()`).
    pub fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Number of elements in this view.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this view covers zero elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrows the view as an immutable slice.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: construction validates type alignment and bounds. CpuElement is
        // sealed to types for which every stored bit pattern is valid.
        unsafe { std::slice::from_raw_parts(self.pointer().cast::<T>(), self.len) }
    }

    /// Base pointer of the viewed elements.
    pub fn as_ptr(&self) -> *const T {
        self.pointer().cast::<T>()
    }

    /// Whether this view holds the only strong reference to the segment.
    /// Unique ownership is what allows safe mutable access: no other view can
    /// observe or alias the bytes.
    pub(crate) fn is_uniquely_owned(&self) -> bool {
        Arc::strong_count(&self.owner) == 1
    }

    /// Reinterprets the view as mutable without checking aliasing.
    ///
    /// # Safety
    /// The caller must hold a `CpuDestination` capability covering this exact
    /// range: either the segment is uniquely owned (no aliases exist), or the
    /// executable's fixed liveness schedule guarantees no concurrent
    /// read/write overlaps this range.
    pub(crate) unsafe fn as_mut_slice_for_destination(&self) -> &mut [T] {
        // SAFETY: the caller holds a CpuDestination capability. Its constructor
        // either proves unique ownership or delegates non-overlap to executable
        // liveness for a planned segment.
        unsafe { std::slice::from_raw_parts_mut(self.pointer().cast::<T>(), self.len) }
    }

    fn pointer(&self) -> *mut u8 {
        // Raw allocation storage is the narrowly contained interior-mutation
        // mechanism; no mutable pointer is exposed by the public API.
        // SAFETY: `byte_offset` was bounds-checked against the segment
        // capacity at construction, so the offset stays within the allocation.
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

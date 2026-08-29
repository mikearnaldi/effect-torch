//! CPU tensors, buffers, element traits, and writable destinations.
//!
//! A [`Tensor`] pairs a dtype-tagged [`CpuBuffer`], which is a typed view
//! into an aligned [`CpuSegment`], with a [`Layout`] that records the
//! logical shape, strides, and storage offset. Cloning a tensor shares the
//! buffer and changes only the layout.
//!
//! Every operation has a `*_requirements` planner. It returns
//! [`CpuTensorRequirement`] values for the output and scratch tensors,
//! including their shape, dtype, byte size, and alignment. The matching
//! `*_into` function writes to a [`CpuDestination`] without allocating.
//! Allocating wrappers such as [`Tensor::full`] compose `empty`,
//! `destination`, and `*_into`.
//!
//! [`CpuDestination`] is the only way to mutate tensor data. Safe construction
//! requires a uniquely owned buffer with no aliasing views. A destination is
//! not `Send` or cloneable and borrows the tensor exclusively for its
//! lifetime. The compiled executor uses the unsafe
//! `CpuDestination::from_planned` constructor. Its fixed memory plan and
//! linear command schedule provide exclusive access to each planned range
//! while writing. Every write checks the dtype, shape, contiguity, and
//! capacity before touching memory.

use crate::storage::{
    assert_allocation_allowed, CpuElement, CpuSegment, CpuStorage, CpuStorageRetention,
    CPU_STORAGE_ALIGNMENT,
};
use effect_torch_runtime::{DType, Layout};
use half::{bf16, f16};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

/// Dense element storage of a [`Tensor`], tagged by dtype.
///
/// Variants wrap [`CpuStorage`] views. Cloning a buffer aliases the same
/// segment bytes.
#[derive(Debug, Clone)]
pub enum CpuBuffer {
    F32(CpuStorage<f32>),
    F64(CpuStorage<f64>),
    F16(CpuStorage<f16>),
    BF16(CpuStorage<bf16>),
    U8(CpuStorage<u8>),
    U32(CpuStorage<u32>),
    I64(CpuStorage<i64>),
}

impl CpuBuffer {
    /// Dtype of the stored elements.
    pub fn dtype(&self) -> DType {
        match self {
            CpuBuffer::F32(_) => DType::F32,
            CpuBuffer::F64(_) => DType::F64,
            CpuBuffer::F16(_) => DType::F16,
            CpuBuffer::BF16(_) => DType::BF16,
            CpuBuffer::U8(_) => DType::U8,
            CpuBuffer::U32(_) => DType::U32,
            CpuBuffer::I64(_) => DType::I64,
        }
    }

    /// Number of elements covered by this view.
    pub fn len(&self) -> usize {
        match self {
            CpuBuffer::F32(v) => v.len(),
            CpuBuffer::F64(v) => v.len(),
            CpuBuffer::F16(v) => v.len(),
            CpuBuffer::BF16(v) => v.len(),
            CpuBuffer::U8(v) => v.len(),
            CpuBuffer::U32(v) => v.len(),
            CpuBuffer::I64(v) => v.len(),
        }
    }

    /// Byte size of this view.
    pub fn byte_len(&self) -> usize {
        self.len() * self.dtype().size_in_bytes()
    }

    /// Creates a dense buffer viewing `len` elements of `dtype` at
    /// `byte_offset` within `owner`, validating alignment and bounds.
    pub fn from_segment(
        owner: Arc<CpuSegment>,
        byte_offset: usize,
        len: usize,
        dtype: DType,
    ) -> Result<Self, String> {
        Self::from_segment_with_retention(owner, byte_offset, len, dtype, None)
    }

    pub(crate) fn from_segment_with_retention(
        owner: Arc<CpuSegment>,
        byte_offset: usize,
        len: usize,
        dtype: DType,
        retention: Option<CpuStorageRetention>,
    ) -> Result<Self, String> {
        macro_rules! view {
            ($variant:ident, $type:ty) => {
                CpuStorage::<$type>::from_segment_with_retention(owner, byte_offset, len, retention)
                    .map(CpuBuffer::$variant)
                    .map_err(|error| error.to_string())
            };
        }
        match dtype {
            DType::F32 => view!(F32, f32),
            DType::F64 => view!(F64, f64),
            DType::F16 => view!(F16, f16),
            DType::BF16 => view!(BF16, bf16),
            DType::U8 => view!(U8, u8),
            DType::U32 => view!(U32, u32),
            DType::I64 => view!(I64, i64),
        }
    }

    fn is_uniquely_owned(&self) -> bool {
        match self {
            CpuBuffer::F32(v) => v.is_uniquely_owned(),
            CpuBuffer::F64(v) => v.is_uniquely_owned(),
            CpuBuffer::F16(v) => v.is_uniquely_owned(),
            CpuBuffer::BF16(v) => v.is_uniquely_owned(),
            CpuBuffer::U8(v) => v.is_uniquely_owned(),
            CpuBuffer::U32(v) => v.is_uniquely_owned(),
            CpuBuffer::I64(v) => v.is_uniquely_owned(),
        }
    }
}

/// Element operations for dtype-generic kernels dispatched across the
/// [`CpuBuffer`] variants.
///
/// Mixed-dtype kernels such as casts and linalg use `to_f64` and `from_f64`
/// to convert values through `f64`.
pub trait Elem: CpuElement {
    /// Copies a vector into freshly allocated, uniquely owned storage.
    fn buffer_of(v: Vec<Self>) -> CpuBuffer;
    /// The runtime dtype corresponding to `Self`.
    fn dtype() -> DType;
    /// Widens the element to `f64`.
    fn to_f64(self) -> f64;
    /// Narrows an `f64` into the element type (rounding/truncating as the
    /// type dictates).
    fn from_f64(x: f64) -> Self;
    /// Returns the logical elements of `t` as a dense slice, or `None` if the
    /// dtype does not match or the layout is not contiguous.
    fn slice_of(t: &Tensor) -> Option<&[Self]>;
    /// Returns the storage of `buffer` if its variant matches `Self`.
    fn storage_of(buffer: &CpuBuffer) -> Option<&CpuStorage<Self>>;
}

macro_rules! impl_elem {
    ($t:ty, $variant:ident, $dtype:expr, $to:expr, $from:expr) => {
        impl Elem for $t {
            fn buffer_of(v: Vec<Self>) -> CpuBuffer {
                let storage = CpuStorage::from_slice(&v)
                    .unwrap_or_else(|error| panic!("CPU tensor allocation failed: {error}"));
                CpuBuffer::$variant(storage)
            }
            fn dtype() -> DType {
                $dtype
            }
            fn to_f64(self) -> f64 {
                #[allow(clippy::redundant_closure_call)]
                let f: fn($t) -> f64 = $to;
                f(self)
            }
            fn from_f64(x: f64) -> Self {
                #[allow(clippy::redundant_closure_call)]
                let f: fn(f64) -> $t = $from;
                f(x)
            }
            fn slice_of(t: &Tensor) -> Option<&[Self]> {
                let CpuBuffer::$variant(v) = &t.buffer else {
                    return None;
                };
                if !t.layout.is_contiguous() {
                    return None;
                }
                let start = t.layout.offset();
                Some(&v[start..start + t.numel()])
            }
            fn storage_of(buffer: &CpuBuffer) -> Option<&CpuStorage<Self>> {
                let CpuBuffer::$variant(storage) = buffer else {
                    return None;
                };
                Some(storage)
            }
        }
    };
}

impl_elem!(f32, F32, DType::F32, |x| x as f64, |x| x as f32);
impl_elem!(f64, F64, DType::F64, |x| x, |x| x);
impl_elem!(f16, F16, DType::F16, |x: f16| x.to_f64(), f16::from_f64);
impl_elem!(
    bf16,
    BF16,
    DType::BF16,
    |x: bf16| x.to_f64(),
    bf16::from_f64
);
impl_elem!(u8, U8, DType::U8, |x| x as f64, |x| x as u8);
impl_elem!(u32, U32, DType::U32, |x| x as f64, |x| x as u32);
impl_elem!(i64, I64, DType::I64, |x| x as f64, |x| x as i64);

/// Exact shape, dtype, byte size, and alignment required for one tensor.
/// The `*_requirements` planners produce these values so callers and the
/// compiled executor can allocate destinations and scratch before running a
/// `*_into` kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuTensorRequirement {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub bytes: usize,
    pub alignment: usize,
}

impl CpuTensorRequirement {
    /// Requirement of a dense tensor of `shape` and `dtype`, aligned to
    /// [`CPU_STORAGE_ALIGNMENT`]. Panics on element-count or byte-size
    /// overflow.
    pub fn new(shape: &[usize], dtype: DType) -> Self {
        let elements = shape
            .iter()
            .try_fold(1usize, |total, &dimension| total.checked_mul(dimension))
            .expect("CPU tensor requirement element count overflow");
        let bytes = elements
            .checked_mul(dtype.size_in_bytes())
            .expect("CPU tensor requirement byte size overflow");
        Self {
            shape: shape.to_vec(),
            dtype,
            bytes,
            alignment: CPU_STORAGE_ALIGNMENT,
        }
    }
}

/// Output plus scratch requirements of one operation invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuOperationRequirements {
    pub output: CpuTensorRequirement,
    pub scratch: Vec<CpuTensorRequirement>,
}

impl CpuOperationRequirements {
    /// Requirement for an operation that needs only its output tensor.
    pub fn without_scratch(shape: &[usize], dtype: DType) -> Self {
        Self {
            output: CpuTensorRequirement::new(shape, dtype),
            scratch: Vec::new(),
        }
    }

    /// Total bytes of all scratch requirements. Panics on overflow.
    pub fn scratch_bytes(&self) -> usize {
        self.scratch
            .iter()
            .try_fold(0usize, |total, requirement| {
                total.checked_add(requirement.bytes)
            })
            .expect("CPU operation scratch byte size overflow")
    }
}

/// A CPU tensor with shared element storage and a logical layout.
///
/// Construction checks `layout.max_index() <= buffer.len()`, so every element
/// addressed through the layout is in bounds.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub buffer: CpuBuffer,
    pub layout: Layout,
}

impl Tensor {
    /// Wraps a buffer with a layout. Panics if the layout can address
    /// elements beyond the buffer.
    pub fn new(buffer: CpuBuffer, layout: Layout) -> Self {
        assert!(layout.max_index() <= buffer.len(), "layout exceeds buffer");
        Tensor { buffer, layout }
    }

    /// Creates a contiguous tensor viewing `shape.product()` elements of
    /// `dtype` at `byte_offset` within `owner`.
    pub fn from_segment(
        owner: Arc<CpuSegment>,
        byte_offset: usize,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Result<Self, String> {
        Self::from_segment_with_retention(owner, byte_offset, shape, dtype, None)
    }

    pub(crate) fn from_segment_with_retention(
        owner: Arc<CpuSegment>,
        byte_offset: usize,
        shape: Vec<usize>,
        dtype: DType,
        retention: Option<CpuStorageRetention>,
    ) -> Result<Self, String> {
        let len = shape
            .iter()
            .try_fold(1usize, |total, &dimension| total.checked_mul(dimension))
            .ok_or_else(|| "CPU tensor element count overflow".to_string())?;
        let buffer =
            CpuBuffer::from_segment_with_retention(owner, byte_offset, len, dtype, retention)?;
        Ok(Self::new(buffer, Layout::contiguous(shape)))
    }

    /// Element dtype.
    pub fn dtype(&self) -> DType {
        self.buffer.dtype()
    }

    /// Logical shape.
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    /// Number of logical elements.
    pub fn numel(&self) -> usize {
        self.layout.numel()
    }

    /// Copies `data` into a fresh contiguous tensor of `shape`.
    ///
    /// Panics under an active allocation guard, on shape/data length
    /// mismatch, or on allocation failure.
    pub fn from_vec<T: Elem>(data: Vec<T>, shape: Vec<usize>) -> Self {
        assert_allocation_allowed("Tensor::from_vec");
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Tensor::new(T::buffer_of(data), Layout::contiguous(shape))
    }

    /// Writes `data` into `destination`, which must match `shape` exactly.
    pub fn from_slice_into<T: Elem>(
        data: &[T],
        shape: &[usize],
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        let expected = shape
            .iter()
            .try_fold(1usize, |total, &dimension| total.checked_mul(dimension));
        if expected != Some(data.len()) {
            return Err(format!(
                "from_slice destination source has {} elements, expected {:?}",
                data.len(),
                expected
            ));
        }
        destination.write::<T, _>("from_slice", shape, |output| output.copy_from_slice(data))
    }

    pub fn from_vec_requirements<T: Elem>(shape: &[usize]) -> CpuOperationRequirements {
        CpuOperationRequirements::without_scratch(shape, T::dtype())
    }

    pub fn from_vec_output_requirements<T: Elem>(shape: &[usize]) -> CpuTensorRequirement {
        Self::from_vec_requirements::<T>(shape).output
    }

    pub fn from_vec_scratch_requirements<T: Elem>(
        _shape: &[usize],
    ) -> &'static [CpuTensorRequirement] {
        &[]
    }

    pub fn from_vec_into<T: Elem>(
        data: &[T],
        shape: &[usize],
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        Self::from_slice_into(data, shape, destination)
    }

    /// Allocates a contiguous tensor of `shape` and `dtype`. The contract
    /// treats its storage as uninitialized, though the allocator zeroes it.
    ///
    /// Panics under an active allocation guard or on allocation failure.
    pub fn empty(shape: &[usize], dtype: DType) -> Self {
        assert_allocation_allowed("Tensor::empty");
        let requirement = CpuTensorRequirement::new(shape, dtype);
        let owner = CpuSegment::allocate(requirement.bytes, requirement.alignment)
            .unwrap_or_else(|error| panic!("CPU tensor allocation failed: {error}"));
        Self::from_segment(owner, 0, requirement.shape, dtype)
            .expect("new CPU segment must contain its exact tensor view")
    }

    /// Allocates a tensor filled with zero.
    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        assert_allocation_allowed("Tensor::zeros");
        Self::full(shape, 0.0, dtype)
    }

    /// Allocates a tensor filled with one.
    pub fn ones(shape: &[usize], dtype: DType) -> Self {
        assert_allocation_allowed("Tensor::ones");
        Self::full(shape, 1.0, dtype)
    }

    /// Allocates a tensor filled with `value` (converted to `dtype`).
    pub fn full(shape: &[usize], value: f64, dtype: DType) -> Self {
        assert_allocation_allowed("Tensor::full");
        let mut output = Self::empty(shape, dtype);
        {
            let mut destination = output
                .destination()
                .expect("new CPU tensor storage must be unique");
            Self::full_into(shape, value, dtype, &mut destination)
                .expect("new CPU tensor must satisfy its fill requirement");
        }
        output
    }

    /// Requirement (output only) of [`Tensor::full`] for `shape`/`dtype`.
    pub fn full_requirements(shape: &[usize], dtype: DType) -> CpuOperationRequirements {
        CpuOperationRequirements::without_scratch(shape, dtype)
    }

    pub fn full_output_requirements(shape: &[usize], dtype: DType) -> CpuTensorRequirement {
        Self::full_requirements(shape, dtype).output
    }

    pub fn full_scratch_requirements(
        _shape: &[usize],
        _dtype: DType,
    ) -> &'static [CpuTensorRequirement] {
        &[]
    }

    pub fn zeros_requirements(shape: &[usize], dtype: DType) -> CpuOperationRequirements {
        Self::full_requirements(shape, dtype)
    }

    pub fn zeros_output_requirements(shape: &[usize], dtype: DType) -> CpuTensorRequirement {
        Self::zeros_requirements(shape, dtype).output
    }

    pub fn zeros_scratch_requirements(
        _shape: &[usize],
        _dtype: DType,
    ) -> &'static [CpuTensorRequirement] {
        &[]
    }

    pub fn ones_requirements(shape: &[usize], dtype: DType) -> CpuOperationRequirements {
        Self::full_requirements(shape, dtype)
    }

    pub fn ones_output_requirements(shape: &[usize], dtype: DType) -> CpuTensorRequirement {
        Self::ones_requirements(shape, dtype).output
    }

    pub fn ones_scratch_requirements(
        _shape: &[usize],
        _dtype: DType,
    ) -> &'static [CpuTensorRequirement] {
        &[]
    }

    /// Fills `destination` (whose dtype must equal `dtype`) with `value`.
    pub fn full_into(
        shape: &[usize],
        value: f64,
        dtype: DType,
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        macro_rules! fill {
            ($type:ty) => {
                destination.write::<$type, _>("full", shape, |output| {
                    output.fill(<$type as Elem>::from_f64(value));
                })
            };
        }
        if destination.dtype() != dtype {
            return Err(format!(
                "full destination dtype mismatch: expected {dtype}, got {}",
                destination.dtype()
            ));
        }
        match dtype {
            DType::F32 => fill!(f32),
            DType::F64 => fill!(f64),
            DType::F16 => fill!(f16),
            DType::BF16 => fill!(bf16),
            DType::U8 => fill!(u8),
            DType::U32 => fill!(u32),
            DType::I64 => fill!(i64),
        }
    }

    pub fn zeros_into(
        shape: &[usize],
        dtype: DType,
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        Self::full_into(shape, 0.0, dtype, destination)
    }

    pub fn ones_into(
        shape: &[usize],
        dtype: DType,
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        Self::full_into(shape, 1.0, dtype, destination)
    }

    /// Returns a view sharing this tensor's storage with a new layout.
    /// Panics if the layout can address elements beyond the buffer.
    pub fn view(&self, layout: Layout) -> Self {
        assert!(layout.max_index() <= self.buffer.len());
        Tensor {
            buffer: self.buffer.clone(),
            layout,
        }
    }

    /// Acquires the mutable destination capability for this tensor.
    ///
    /// Fails when the buffer is shared because another tensor handle could
    /// observe writes through an aliased view. Only uniquely owned storage can
    /// become a safe destination.
    pub fn destination(&mut self) -> Result<CpuDestination<'_>, String> {
        CpuDestination::new(self)
    }

    /// Requirement (output only) of gathering this tensor into contiguous
    /// form with the same shape and dtype.
    pub fn contiguous_requirements(&self) -> CpuOperationRequirements {
        CpuOperationRequirements::without_scratch(self.shape(), self.dtype())
    }

    pub fn contiguous_output_requirements(&self) -> CpuTensorRequirement {
        self.contiguous_requirements().output
    }

    pub fn contiguous_scratch_requirements(&self) -> &'static [CpuTensorRequirement] {
        &[]
    }

    pub fn copy_requirements(&self) -> CpuOperationRequirements {
        self.contiguous_requirements()
    }

    pub fn copy_output_requirements(&self) -> CpuTensorRequirement {
        self.copy_requirements().output
    }

    pub fn copy_scratch_requirements(&self) -> &'static [CpuTensorRequirement] {
        &[]
    }

    pub fn materialize_requirements(&self) -> CpuOperationRequirements {
        self.contiguous_requirements()
    }

    pub fn materialize_output_requirements(&self) -> CpuTensorRequirement {
        self.materialize_requirements().output
    }

    pub fn materialize_scratch_requirements(&self) -> &'static [CpuTensorRequirement] {
        &[]
    }

    /// Returns a densely packed contiguous tensor. Clones tensors that already
    /// have a contiguous layout, zero offset, and an exact-size buffer. Gathers
    /// all other tensors into new storage.
    pub fn contiguous(&self) -> Self {
        if self.layout.is_contiguous()
            && self.layout.offset() == 0
            && self.buffer.len() == self.layout.numel()
        {
            return self.clone();
        }
        assert_allocation_allowed("Tensor::contiguous");
        let mut output = Self::empty(self.shape(), self.dtype());
        {
            let mut destination = output
                .destination()
                .expect("new CPU tensor storage must be unique");
            self.contiguous_into(&mut destination)
                .expect("new CPU tensor must satisfy contiguous output requirement");
        }
        output
    }

    /// Gathers this tensor's logical elements (following its strides) into
    /// `destination`, which must match this tensor's shape and dtype.
    pub fn copy_into(&self, destination: &mut CpuDestination<'_>) -> Result<(), String> {
        macro_rules! copy {
            ($storage:expr, $type:ty) => {
                destination.write::<$type, _>("copy", self.shape(), |output| {
                    copy_strided($storage, &self.layout, output)
                })
            };
        }
        match &self.buffer {
            CpuBuffer::F32(values) => copy!(values, f32),
            CpuBuffer::F64(values) => copy!(values, f64),
            CpuBuffer::F16(values) => copy!(values, f16),
            CpuBuffer::BF16(values) => copy!(values, bf16),
            CpuBuffer::U8(values) => copy!(values, u8),
            CpuBuffer::U32(values) => copy!(values, u32),
            CpuBuffer::I64(values) => copy!(values, i64),
        }
    }

    /// `*_into` form of [`Tensor::contiguous`].
    pub fn contiguous_into(&self, destination: &mut CpuDestination<'_>) -> Result<(), String> {
        self.copy_into(destination)
    }

    /// Alias of [`Tensor::contiguous_into`] used by the executor when
    /// materializing a value into its planned location.
    pub fn materialize_into(&self, destination: &mut CpuDestination<'_>) -> Result<(), String> {
        self.contiguous_into(destination)
    }

    /// Gathers this tensor into `destination` using the destination's own
    /// shape, which must have the same element count and dtype. Used for
    /// planned reshapes, where the logical shape changes but the element
    /// order does not.
    pub fn materialize_reshaped_into(
        &self,
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        if self.numel() != destination.tensor.numel() || self.dtype() != destination.dtype() {
            return Err("reshape destination does not match source elements and dtype".to_string());
        }
        macro_rules! copy {
            ($storage:expr, $type:ty) => {
                destination.write_current::<$type, _>("reshape", |output| {
                    copy_strided($storage, &self.layout, output)
                })
            };
        }
        match &self.buffer {
            CpuBuffer::F32(values) => copy!(values, f32),
            CpuBuffer::F64(values) => copy!(values, f64),
            CpuBuffer::F16(values) => copy!(values, f16),
            CpuBuffer::BF16(values) => copy!(values, bf16),
            CpuBuffer::U8(values) => copy!(values, u8),
            CpuBuffer::U32(values) => copy!(values, u32),
            CpuBuffer::I64(values) => copy!(values, i64),
        }
    }

    /// Output-only requirement for casting to `dtype` with the same shape.
    pub fn cast_requirements(&self, dtype: DType) -> CpuOperationRequirements {
        CpuOperationRequirements::without_scratch(self.shape(), dtype)
    }

    pub fn cast_output_requirements(&self, dtype: DType) -> CpuTensorRequirement {
        self.cast_requirements(dtype).output
    }

    pub fn cast_scratch_requirements(&self, _dtype: DType) -> &'static [CpuTensorRequirement] {
        &[]
    }

    /// Converts every element to `dtype`, routing through `f64`. Same-dtype
    /// casts are cheap clones.
    pub fn cast(&self, dtype: DType) -> Self {
        if self.dtype() == dtype {
            return self.clone();
        }
        assert_allocation_allowed("Tensor::cast");
        let mut output = Self::empty(self.shape(), dtype);
        {
            let mut destination = output
                .destination()
                .expect("new CPU tensor storage must be unique");
            self.cast_into(dtype, &mut destination)
                .expect("new CPU tensor must satisfy cast output requirement");
        }
        output
    }

    /// `*_into` form of [`Tensor::cast`]. `destination` must have dtype
    /// `dtype` and this tensor's shape.
    pub fn cast_into(
        &self,
        dtype: DType,
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        if destination.dtype() != dtype {
            return Err(format!(
                "cast destination dtype mismatch: expected {dtype}, got {}",
                destination.dtype()
            ));
        }
        macro_rules! source {
            ($values:expr, $source:ty) => {
                match dtype {
                    DType::F32 => cast_strided::<$source, f32>($values, &self.layout, destination),
                    DType::F64 => cast_strided::<$source, f64>($values, &self.layout, destination),
                    DType::F16 => cast_strided::<$source, f16>($values, &self.layout, destination),
                    DType::BF16 => {
                        cast_strided::<$source, bf16>($values, &self.layout, destination)
                    }
                    DType::U8 => cast_strided::<$source, u8>($values, &self.layout, destination),
                    DType::U32 => cast_strided::<$source, u32>($values, &self.layout, destination),
                    DType::I64 => cast_strided::<$source, i64>($values, &self.layout, destination),
                }
            };
        }
        match &self.buffer {
            CpuBuffer::F32(values) => source!(values, f32),
            CpuBuffer::F64(values) => source!(values, f64),
            CpuBuffer::F16(values) => source!(values, f16),
            CpuBuffer::BF16(values) => source!(values, bf16),
            CpuBuffer::U8(values) => source!(values, u8),
            CpuBuffer::U32(values) => source!(values, u32),
            CpuBuffer::I64(values) => source!(values, i64),
        }
    }
}

/// Exclusive write capability for a tensor's storage.
///
/// This is the only way kernels obtain `&mut [T]` into tensor memory.
/// `_exclusive` makes it behave like an `&mut Tensor` borrow, allowing one
/// live destination per tensor. `_thread_owned`
/// (`PhantomData<Rc<()>>`) makes it neither `Send` nor `Sync`, so a
/// destination cannot cross a thread boundary.
///
/// [`CpuDestination::new`] safely constructs a destination only from uniquely
/// owned storage. No other view may alias the bytes. The compiled executor
/// uses unsafe `CpuDestination::from_planned`. Its fixed liveness schedule
/// provides exclusivity instead of ownership.
///
/// Every write checks the destination's dtype, contiguity, shape, and capacity
/// before producing a mutable slice. A stale or mismatched destination returns
/// an error instead of corrupting memory.
pub struct CpuDestination<'a> {
    tensor: &'a Tensor,
    _exclusive: PhantomData<&'a mut Tensor>,
    _thread_owned: PhantomData<Rc<()>>,
}

impl<'a> CpuDestination<'a> {
    /// Creates a destination for uniquely owned storage.
    pub fn new(tensor: &'a mut Tensor) -> Result<Self, String> {
        if !tensor.buffer.is_uniquely_owned() {
            return Err("CPU destination storage is shared and cannot be published mutably".into());
        }
        Ok(Self {
            tensor,
            _exclusive: PhantomData,
            _thread_owned: PhantomData,
        })
    }

    /// Creates an executor-owned destination into planned shared storage.
    ///
    /// # Safety
    /// The fixed schedule must keep this byte range live exclusively for the
    /// destination write, with no overlapping read/write on any other thread.
    #[allow(dead_code)]
    pub(crate) unsafe fn from_planned(tensor: &'a Tensor) -> Self {
        Self {
            tensor,
            _exclusive: PhantomData,
            _thread_owned: PhantomData,
        }
    }

    /// Shape of the destination tensor.
    pub fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    /// Dtype of the destination tensor.
    pub fn dtype(&self) -> DType {
        self.tensor.dtype()
    }

    /// Writes raw bytes into the destination's logical region without regard
    /// to dtype. NAPI loaders use this to stream file bytes directly into
    /// tensor storage. The destination must be contiguous.
    #[cfg(feature = "napi-addon")]
    pub(crate) fn write_bytes<R>(
        &mut self,
        operation: &str,
        write: impl FnOnce(&mut [u8]) -> R,
    ) -> Result<R, String> {
        if !self.tensor.layout.is_contiguous() {
            return Err(format!(
                "{operation} destination must be contiguous, got {:?}",
                self.tensor.layout
            ));
        }
        let element_size = self.dtype().size_in_bytes();
        let start = self
            .tensor
            .layout
            .offset()
            .checked_mul(element_size)
            .ok_or_else(|| format!("{operation} destination byte offset overflows"))?;
        let byte_len = self
            .tensor
            .numel()
            .checked_mul(element_size)
            .ok_or_else(|| format!("{operation} destination byte length overflows"))?;
        let end = start
            .checked_add(byte_len)
            .ok_or_else(|| format!("{operation} destination byte range overflows"))?;
        macro_rules! write_storage {
            ($storage:expr) => {{
                // SAFETY: CpuDestination provides exclusive access to this storage.
                let values = unsafe { $storage.as_mut_slice_for_destination() };
                // SAFETY: reinterprets the exclusively borrowed `values` as
                // bytes. `u8` has no invalid bit patterns and the byte slice
                // covers exactly the same live range.
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(
                        values.as_mut_ptr().cast::<u8>(),
                        std::mem::size_of_val(values),
                    )
                };
                if end > bytes.len() {
                    return Err(format!(
                        "{operation} destination requires {end} bytes, storage has {}",
                        bytes.len()
                    ));
                }
                return Ok(write(&mut bytes[start..end]));
            }};
        }
        match &self.tensor.buffer {
            CpuBuffer::F32(storage) => write_storage!(storage),
            CpuBuffer::F64(storage) => write_storage!(storage),
            CpuBuffer::F16(storage) => write_storage!(storage),
            CpuBuffer::BF16(storage) => write_storage!(storage),
            CpuBuffer::U8(storage) => write_storage!(storage),
            CpuBuffer::U32(storage) => write_storage!(storage),
            CpuBuffer::I64(storage) => write_storage!(storage),
        }
    }

    /// Validates dtype, contiguity, and capacity of the destination for a
    /// write of `T` over its current shape.
    pub(crate) fn validate_current<T: Elem>(&self, operation: &str) -> Result<(), String> {
        if self.dtype() != T::dtype() {
            return Err(format!(
                "{operation} destination dtype mismatch: expected {}, got {}",
                T::dtype(),
                self.dtype()
            ));
        }
        if !self.tensor.layout.is_contiguous() {
            return Err(format!(
                "{operation} destination must be contiguous, got {:?}",
                self.tensor.layout
            ));
        }
        if self.tensor.layout.max_index() > self.tensor.buffer.len() {
            return Err(format!(
                "{operation} destination requires {} elements, storage has capacity {}",
                self.tensor.layout.max_index(),
                self.tensor.buffer.len()
            ));
        }
        Ok(())
    }

    /// Runs `write` with a mutable slice of the destination after checking
    /// the destination shape equals `shape`.
    pub(crate) fn write<T: Elem, R>(
        &mut self,
        operation: &str,
        shape: &[usize],
        write: impl FnOnce(&mut [T]) -> R,
    ) -> Result<R, String> {
        if self.shape() != shape {
            return Err(format!(
                "{operation} destination shape mismatch: expected {shape:?}, got {:?}",
                self.shape()
            ));
        }
        self.write_current(operation, write)
    }

    /// Runs `write` with a mutable slice of the destination using its
    /// current shape.
    pub(crate) fn write_current<T: Elem, R>(
        &mut self,
        operation: &str,
        write: impl FnOnce(&mut [T]) -> R,
    ) -> Result<R, String> {
        self.write_current_shaped(operation, |_, output| write(output))
    }

    /// Like [`CpuDestination::write_current`], but also passes the destination
    /// shape to kernels that index output by logical coordinates.
    pub(crate) fn write_current_shaped<T: Elem, R>(
        &mut self,
        operation: &str,
        write: impl FnOnce(&[usize], &mut [T]) -> R,
    ) -> Result<R, String> {
        self.validate_current::<T>(operation)?;
        let storage = T::storage_of(&self.tensor.buffer)
            .expect("destination dtype was validated before dispatch");
        let start = self.tensor.layout.offset();
        let end = start
            .checked_add(self.tensor.numel())
            .ok_or_else(|| format!("{operation} destination range overflow"))?;
        // SAFETY: CpuDestination is non-cloneable and thread-owned. Safe
        // construction requires unique storage. Planned construction relies on
        // the executor's fixed liveness schedule for disjointness.
        let values = unsafe { storage.as_mut_slice_for_destination() };
        Ok(write(self.tensor.shape(), &mut values[start..end]))
    }
}

fn cast_strided<S: Elem, D: Elem>(
    source: &[S],
    layout: &Layout,
    destination: &mut CpuDestination<'_>,
) -> Result<(), String> {
    destination.write::<D, _>("cast", layout.shape(), |output| {
        for (index, value) in output.iter_mut().enumerate() {
            *value = D::from_f64(source[source_index(layout, index)].to_f64());
        }
    })
}

/// Converts a linear output index into a storage index under `layout`. Uses
/// a contiguous fast path or a row-major walk over the strides and offset.
pub(crate) fn source_index(layout: &Layout, output_index: usize) -> usize {
    if layout.is_contiguous() {
        return layout.offset() + output_index;
    }
    let mut source = layout.offset();
    let mut remainder = output_index;
    for dimension in (0..layout.shape().len()).rev() {
        let width = layout.shape()[dimension].max(1);
        let coordinate = remainder % width;
        remainder /= width;
        source += coordinate * layout.strides()[dimension];
    }
    source
}

/// Copies the `layout`-addressed elements of `src` into dense `dst`.
pub fn copy_strided<T: Copy>(src: &[T], layout: &Layout, dst: &mut [T]) {
    if layout.is_contiguous() {
        let start = layout.offset();
        dst.copy_from_slice(&src[start..start + layout.numel()]);
        return;
    }
    for (output_index, output) in dst.iter_mut().enumerate() {
        *output = src[source_index(layout, output_index)];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::ExecutableAllocationGuard;

    #[test]
    fn contiguous_gathers_strided() {
        let t = Tensor::from_vec((0..12).map(|x| x as f32).collect(), vec![3, 4]);
        let v = t.view(t.layout.permute(&[1, 0]));
        let c = v.contiguous();
        assert_eq!(c.shape(), &[4, 3]);
        let CpuBuffer::F32(data) = &c.buffer else {
            panic!()
        };
        assert_eq!(
            data.as_slice(),
            &[0., 4., 8., 1., 5., 9., 2., 6., 10., 3., 7., 11.]
        );
    }

    #[test]
    fn cast_roundtrip_f16() {
        let t = Tensor::from_vec(vec![1.5f32, -2.25, 100.0], vec![3]);
        let h = t.cast(DType::F16);
        assert_eq!(h.dtype(), DType::F16);
        let back = h.cast(DType::F32);
        let CpuBuffer::F32(data) = &back.buffer else {
            panic!()
        };
        assert_eq!(data.as_slice(), &[1.5, -2.25, 100.0]);
    }

    #[test]
    fn cast_int_float() {
        let t = Tensor::from_vec(vec![1u32, 2, 255], vec![3]);
        let f = t.cast(DType::F32);
        let CpuBuffer::F32(data) = &f.buffer else {
            panic!()
        };
        assert_eq!(data.as_slice(), &[1.0, 2.0, 255.0]);
        let b = f.cast(DType::U8);
        let CpuBuffer::U8(data) = &b.buffer else {
            panic!()
        };
        assert_eq!(data.as_slice(), &[1, 2, 255]);
    }

    #[test]
    fn planned_non_overlapping_destinations_write_one_segment() {
        let segment = CpuSegment::allocate(128, 64).unwrap();
        let left = Tensor::from_segment(Arc::clone(&segment), 0, vec![4], DType::F32).unwrap();
        let right = Tensor::from_segment(Arc::clone(&segment), 64, vec![4], DType::F32).unwrap();
        // SAFETY: these exact typed views occupy disjoint ranges in this test.
        let mut left_destination = unsafe { CpuDestination::from_planned(&left) };
        // SAFETY: these exact typed views occupy disjoint ranges in this test.
        let mut right_destination = unsafe { CpuDestination::from_planned(&right) };
        {
            // Segment acquisition and view resolution happen before execution.
            let _guard = ExecutableAllocationGuard::enter();
            Tensor::full_into(&[4], 2.0, DType::F32, &mut left_destination).unwrap();
            Tensor::full_into(&[4], 3.0, DType::F32, &mut right_destination).unwrap();
        }
        let CpuBuffer::F32(left) = &left.buffer else {
            panic!()
        };
        let CpuBuffer::F32(right) = &right.buffer else {
            panic!()
        };
        assert_eq!(left.as_slice(), &[2.0; 4]);
        assert_eq!(right.as_slice(), &[3.0; 4]);
    }

    #[test]
    fn shared_storage_cannot_become_a_safe_destination() {
        let mut tensor = Tensor::zeros(&[2], DType::F32);
        let alias = tensor.clone();
        assert!(tensor.destination().is_err());
        drop(alias);
        assert!(tensor.destination().is_ok());
    }

    #[test]
    fn allocation_guard_rejects_all_tensor_allocation_entry_points() {
        let source = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]);
        let strided = source.view(source.layout.permute(&[0]));
        let _guard = ExecutableAllocationGuard::enter();
        assert!(std::panic::catch_unwind(|| Tensor::zeros(&[1], DType::F32)).is_err());
        assert!(std::panic::catch_unwind(|| Tensor::ones(&[1], DType::F32)).is_err());
        assert!(std::panic::catch_unwind(|| Tensor::full(&[1], 2.0, DType::F32)).is_err());
        assert!(std::panic::catch_unwind(|| { Tensor::from_vec(vec![1.0f32], vec![1]) }).is_err());
        assert!(std::panic::catch_unwind(|| source.cast(DType::F64)).is_err());
        let narrowed = strided.view(Layout::new(vec![1], vec![1], 1));
        assert!(std::panic::catch_unwind(|| narrowed.contiguous()).is_err());
        assert!(std::panic::catch_unwind(|| CpuSegment::allocate(16, 8)).is_err());
    }

    #[test]
    fn materialization_requirements_are_exact_and_into_matches_wrappers() {
        let source = Tensor::from_vec((0..6).map(|value| value as f32).collect(), vec![2, 3]);
        let source = source.view(source.layout.permute(&[1, 0]));
        let requirements = source.cast_requirements(DType::F64);
        assert_eq!(requirements.output.bytes, 6 * std::mem::size_of::<f64>());
        assert_eq!(requirements.scratch_bytes(), 0);
        let mut output = Tensor::empty(&requirements.output.shape, requirements.output.dtype);
        source
            .cast_into(DType::F64, &mut output.destination().unwrap())
            .unwrap();
        let wrapped = source.cast(DType::F64);
        let CpuBuffer::F64(output) = &output.buffer else {
            panic!()
        };
        let CpuBuffer::F64(wrapped) = &wrapped.buffer else {
            panic!()
        };
        assert_eq!(output, wrapped);
    }

    #[test]
    fn into_rejects_insufficient_destination_capacity() {
        let source = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]);
        let short = Tensor::from_vec(vec![0.0f32], vec![1]);
        let mut invalid = Tensor {
            buffer: short.buffer,
            layout: Layout::contiguous(vec![2]),
        };
        assert!(source
            .copy_into(&mut invalid.destination().unwrap())
            .is_err());
    }
}

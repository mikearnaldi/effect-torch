use cudarc::driver::{
    sys, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceSlice, LaunchArgs, PushKernelArg,
    SyncOnDrop,
};
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;

/// A typed view into an owned CUDA allocation.
///
/// The allocation can have a different element type, which is required for
/// compiler-planned byte segments. All CUDA work in this backend uses one
/// device-local stream, so pointer views rely on stream ordering rather than
/// independent cudarc read/write events.
#[derive(Clone)]
pub(crate) struct CudaBuffer<T> {
    ptr: sys::CUdeviceptr,
    len: usize,
    stream: Arc<CudaStream>,
    _owner: Arc<dyn Send + Sync>,
    _retention: Option<Arc<dyn Send + Sync>>,
    _marker: PhantomData<T>,
}

impl<T: Send + Sync + 'static> CudaBuffer<T> {
    pub(crate) fn from_slice(slice: CudaSlice<T>) -> Self {
        Self::from_arc_slice(Arc::new(slice))
    }

    pub(crate) fn from_arc_slice(slice: Arc<CudaSlice<T>>) -> Self {
        let stream = Arc::clone(slice.stream());
        let (ptr, sync) = slice.device_ptr(&stream);
        drop(sync);
        let len = slice.len();
        let owner: Arc<dyn Send + Sync> = slice;
        Self {
            ptr,
            len,
            stream,
            _owner: owner,
            _retention: None,
            _marker: PhantomData,
        }
    }

    pub(crate) fn from_segment(
        owner: Arc<CudaSlice<u8>>,
        byte_offset: usize,
        len: usize,
        retention: Option<Arc<dyn Send + Sync>>,
    ) -> Result<Self, String> {
        if byte_offset % std::mem::align_of::<T>() != 0 {
            return Err(format!(
                "CUDA segment offset {byte_offset} is not aligned for {}",
                std::any::type_name::<T>()
            ));
        }
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| "CUDA buffer byte size overflowed usize".to_string())?;
        let end = byte_offset
            .checked_add(bytes)
            .ok_or_else(|| "CUDA buffer range overflowed usize".to_string())?;
        if end > owner.len() {
            return Err(format!(
                "CUDA buffer range {byte_offset}..{end} exceeds segment size {}",
                owner.len()
            ));
        }
        let stream = Arc::clone(owner.stream());
        let (base, sync) = owner.device_ptr(&stream);
        drop(sync);
        let offset =
            u64::try_from(byte_offset).map_err(|_| "CUDA buffer offset exceeds u64".to_string())?;
        let ptr = base
            .checked_add(offset)
            .ok_or_else(|| "CUDA device pointer overflow".to_string())?;
        let owner: Arc<dyn Send + Sync> = owner;
        Ok(Self {
            ptr,
            len,
            stream,
            _owner: owner,
            _retention: retention,
            _marker: PhantomData,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn address(&self) -> u64 {
        self.ptr
    }

    pub(crate) fn slice(&self, range: Range<usize>) -> Result<Self, String> {
        if range.start > range.end || range.end > self.len {
            return Err(format!(
                "CUDA buffer range {}..{} exceeds length {}",
                range.start, range.end, self.len
            ));
        }
        let byte_offset = range
            .start
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| "CUDA buffer slice offset overflowed usize".to_string())?;
        let byte_offset = u64::try_from(byte_offset)
            .map_err(|_| "CUDA buffer slice offset exceeds u64".to_string())?;
        Ok(Self {
            ptr: self
                .ptr
                .checked_add(byte_offset)
                .ok_or_else(|| "CUDA buffer slice pointer overflow".to_string())?,
            len: range.end - range.start,
            stream: Arc::clone(&self.stream),
            _owner: Arc::clone(&self._owner),
            _retention: self._retention.clone(),
            _marker: PhantomData,
        })
    }

    pub(crate) fn cast<U: Send + Sync + 'static>(
        &self,
        len: usize,
    ) -> Result<CudaBuffer<U>, String> {
        if self.ptr % std::mem::align_of::<U>() as u64 != 0 {
            return Err(format!(
                "CUDA device pointer is not aligned for {}",
                std::any::type_name::<U>()
            ));
        }
        let source_bytes = self
            .len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| "CUDA source buffer byte size overflowed usize".to_string())?;
        let requested_bytes = len
            .checked_mul(std::mem::size_of::<U>())
            .ok_or_else(|| "CUDA buffer byte size overflowed usize".to_string())?;
        if requested_bytes > source_bytes {
            return Err(format!(
                "CUDA typed view requires {requested_bytes} bytes, but the source has {source_bytes}"
            ));
        }
        Ok(CudaBuffer {
            ptr: self.ptr,
            len,
            stream: Arc::clone(&self.stream),
            _owner: Arc::clone(&self._owner),
            _retention: self._retention.clone(),
            _marker: PhantomData,
        })
    }
}

impl<T> DeviceSlice<T> for CudaBuffer<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl<T> DevicePtr<T> for CudaBuffer<T> {
    fn device_ptr<'a>(&'a self, _stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Sync(None))
    }
}

impl<T> DevicePtrMut<T> for CudaBuffer<T> {
    fn device_ptr_mut<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Sync(None))
    }
}

unsafe impl<'a, 'b: 'a, T> PushKernelArg<&'b CudaBuffer<T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b CudaBuffer<T>) -> &mut Self {
        PushKernelArg::<&sys::CUdeviceptr>::arg(self, &arg.ptr)
    }
}

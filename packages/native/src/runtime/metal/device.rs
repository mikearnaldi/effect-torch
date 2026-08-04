use crate::runtime::dtype::DType;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock};

const PROBES: usize = 8;
const MAX_BUCKET: usize = 4096;
const DISPATCHES_PER_BUFFER: usize = 4096;

pub static DISPATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static SYNCS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static SYNC_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn dispatch_stats_reset() -> (u64, u64, u64) {
    let d = DISPATCHES.swap(0, std::sync::atomic::Ordering::Relaxed);
    let s = SYNCS.swap(0, std::sync::atomic::Ordering::Relaxed);
    let n = SYNC_NANOS.swap(0, std::sync::atomic::Ordering::Relaxed);
    (d, s, n)
}
const SWEEP_MS: u64 = 100;

pub struct Buffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub size: usize,
}

impl Buffer {
    pub fn from_raw(raw: Retained<ProtocolObject<dyn MTLBuffer>>, size: usize) -> Self {
        Buffer { raw, size }
    }

    pub fn contents_ptr(&self) -> *mut std::ffi::c_void {
        self.raw.contents().as_ptr()
    }

    pub fn as_raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }

    pub fn read_f32(&self, offset_elems: usize, n: usize) -> Vec<f32> {
        assert!(offset_elems * 4 + n * 4 <= self.size);
        let ptr = unsafe { self.raw.contents().cast::<f32>().add(offset_elems) };
        unsafe { std::slice::from_raw_parts(ptr.as_ptr(), n) }.to_vec()
    }

    pub fn write_f32(&mut self, offset_elems: usize, data: &[f32]) {
        assert!(offset_elems * 4 + data.len() * 4 <= self.size);
        let ptr = unsafe { self.raw.contents().cast::<f32>().add(offset_elems) };
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.as_ptr(), data.len()) };
    }
}

#[derive(Clone)]
pub struct Pipeline {
    raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl Pipeline {
    pub fn as_raw(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.raw
    }
}

struct Allocator {
    buckets: HashMap<usize, Vec<Arc<Buffer>>>,
    cursor: usize,
    last_sweep: std::time::Instant,
}

impl Allocator {
    fn new() -> Self {
        Allocator {
            buckets: HashMap::new(),
            cursor: 0,
            last_sweep: std::time::Instant::now(),
        }
    }

    // Buffers whose only reference is the bucket's may still be read by
    // in-flight GPU dispatches (the serial encoder orders execution, not
    // completion). The caller moves them to the device's retire list instead
    // of deallocating; they are released at the next synchronize, when the
    // GPU is drained.
    fn sweep(&mut self, retired: &mut Vec<Arc<Buffer>>) {
        if self.last_sweep.elapsed() < std::time::Duration::from_millis(SWEEP_MS) {
            return;
        }
        self.last_sweep = std::time::Instant::now();
        for bucket in self.buckets.values_mut() {
            bucket.retain(|b| {
                if Arc::strong_count(b) > 1 {
                    true
                } else {
                    retired.push(b.clone());
                    false
                }
            });
        }
    }
}

struct EncoderManager {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    current: Option<(
        Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    )>,
    count: usize,
    in_flight: Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}

impl EncoderManager {
    fn new(queue: Retained<ProtocolObject<dyn MTLCommandQueue>>) -> Self {
        EncoderManager { queue, current: None, count: 0, in_flight: Vec::new() }
    }

    fn ensure_encoder(&mut self) {
        if self.current.is_none() {
            let cb = self.queue.commandBuffer().expect("metal command buffer");
            let encoder = cb.computeCommandEncoder().expect("metal compute encoder");
            self.current = Some((cb, encoder));
        }
    }

    fn finish_dispatch(&mut self) {
        // Untracked hazards: without a barrier, Metal may overlap adjacent
        // compute dispatches in the same command buffer. Our allocator
        // recycles buffers across dispatches, so every dispatch must be
        // ordered after the previous one.
        if let Some((_, encoder)) = &self.current {
            encoder.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
        }
        self.count += 1;
        DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if self.count >= DISPATCHES_PER_BUFFER {
            self.commit();
        }
    }

    fn commit(&mut self) {
        if let Some((cb, encoder)) = self.current.take() {
            encoder.endEncoding();
            cb.commit();
            self.in_flight.push(cb);
            self.count = 0;
        }
    }

    fn synchronize(&mut self) -> crate::err::Res<()> {
        self.commit();
        if let Some(last) = self.in_flight.last() {
            last.waitUntilCompleted();
            let status = last.status();
            if status != objc2_metal::MTLCommandBufferStatus::Completed {
                let description = last
                    .error()
                    .map(|e| e.localizedDescription().to_string())
                    .unwrap_or_else(|| "unknown error".to_string());
                self.in_flight.clear();
                return Err(format!(
                    "metal: GPU command buffer failed ({description}) — GPU work was lost; this is usually device memory exhaustion"
                ));
            }
        }
        self.in_flight.clear();
        Ok(())
    }
}

pub struct MetalDevice {
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
    allocator: Mutex<Allocator>,
    encoder: Mutex<EncoderManager>,
    pipelines: Mutex<HashMap<u64, Pipeline>>,
    retired: Mutex<Vec<Arc<Buffer>>>,
}

// Metal command queues serialize command buffer execution; our encoder
// manager additionally holds a mutex for the entire encode session.
// Buffers/pipelines are immutable after creation.
unsafe impl Send for MetalDevice {}
unsafe impl Sync for MetalDevice {}
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

static SHARED_OPTIONS: MTLResourceOptions =
    MTLResourceOptions(MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0);

impl MetalDevice {
    pub fn get() -> &'static MetalDevice {
        static DEVICE: OnceLock<MetalDevice> = OnceLock::new();
        DEVICE.get_or_init(|| MetalDevice::new(0).expect("metal device"))
    }

    pub fn new(ordinal: usize) -> Result<Self, String> {
        let devices = objc2_metal::MTLCopyAllDevices();
        if devices.is_empty() {
            return Err("no Metal devices available".to_string());
        }
        let raw = devices
            .to_vec()
            .swap_remove(ordinal.min(devices.len() - 1));
        let queue = raw.newCommandQueue().ok_or("failed to create command queue")?;
        Ok(MetalDevice {
            raw,
            allocator: Mutex::new(Allocator::new()),
            encoder: Mutex::new(EncoderManager::new(queue)),
            pipelines: Mutex::new(HashMap::new()),
            retired: Mutex::new(Vec::new()),
        })
    }

    pub fn raw(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.raw
    }

    pub fn alloc(&self, elements: usize, dtype: DType) -> Arc<Buffer> {
        let size = elements * dtype.size_in_bytes();
        let bucket_size = size.next_power_of_two().max(16);
        let mut alloc = self.allocator.lock().unwrap();
        {
            let mut retired = self.retired.lock().unwrap();
            alloc.sweep(&mut retired);
        }
        let cursor = alloc.cursor;
        alloc.cursor = alloc.cursor.wrapping_add(1);
        let bucket = alloc.buckets.entry(bucket_size).or_default();
        if !bucket.is_empty() {
            for k in 0..PROBES {
                let idx = cursor.wrapping_add(k) % bucket.len();
                if Arc::strong_count(&bucket[idx]) == 1 {
                    return bucket.swap_remove(idx);
                }
            }
        }
        let options = if std::env::var("EFFECT_TORCH_PRIVATE_INTERMEDIATES").is_ok() {
            MTLResourceOptions(MTLResourceOptions::StorageModePrivate.0 | MTLResourceOptions::HazardTrackingModeUntracked.0)
        } else {
            SHARED_OPTIONS
        };
        let raw = self
            .raw
            .newBufferWithLength_options(bucket_size, options)
            .expect("metal buffer allocation failed");
        let buffer = Arc::new(Buffer { raw, size: bucket_size });
        if bucket.len() < MAX_BUCKET {
            bucket.push(buffer.clone());
        }
        buffer
    }

    pub fn alloc_with_data(&self, data: &[f32]) -> Arc<Buffer> {
        self.upload_bytes(unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) })
    }

    pub fn alloc_with_data_u32(&self, data: &[u32]) -> Arc<Buffer> {
        self.upload_bytes(unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) })
    }

    pub fn upload_bytes(&self, data: &[u8]) -> Arc<Buffer> {
        let bucket_size = data.len().next_power_of_two().max(16);
        let raw = unsafe {
            self.raw.newBufferWithBytes_length_options(
                NonNull::new(data.as_ptr() as *const std::ffi::c_void as *mut std::ffi::c_void).unwrap(),
                bucket_size,
                SHARED_OPTIONS,
            )
        }
        .expect("metal buffer allocation failed");
        let buffer = Arc::new(Buffer { raw, size: bucket_size });
        // Host uploads retire only at the next synchronize. Uploads are
        // NEVER pooled: the only strong refs are the caller's and this
        // list's, so nothing can recycle the bytes before the GPU has
        // consumed them (concurrent walks/tests share the device).
        self.retired.lock().unwrap().push(buffer.clone());
        buffer
    }

    pub fn compile(&self, key: u64, source: &str, name: &str) -> Result<Pipeline, String> {
        if let Some(p) = self.pipeline_cached(key) {
            return Ok(p);
        }
        self.compile_slow(key, source, name)
    }

    // Cache-hit fast path that never builds the kernel source: MSL source
    // generation (SSA emission, format! graphs) is the dominant encode cost
    // on hot paths, and it is pure waste when the pipeline is cached.
    pub fn compile_lazy(&self, key: u64, name: &str, make_source: impl FnOnce() -> String) -> Result<Pipeline, String> {
        match self.pipeline_cached(key) {
            Some(p) => Ok(p),
            None => self.compile_slow(key, &make_source(), name),
        }
    }

    pub fn pipeline_cached(&self, key: u64) -> Option<Pipeline> {
        self.pipelines.lock().unwrap().get(&key).cloned()
    }

    pub fn compile_slow(&self, key: u64, source: &str, name: &str) -> Result<Pipeline, String> {
        let mut cache = self.pipelines.lock().unwrap();
        if let Some(p) = cache.get(&key) {
            return Ok(p.clone());
        }
        let opts = objc2_metal::MTLCompileOptions::new();
        #[allow(deprecated)]
        opts.setFastMathEnabled(false);
        let src_ns = objc2_foundation::NSString::from_str(source);
        let lib = self
            .raw
            .newLibraryWithSource_options_error(&src_ns, Some(&opts))
            .map_err(|e| format!("metal compile {name}: {e:?}"))?;
        let func_name = objc2_foundation::NSString::from_str(name);
        let func = lib
            .newFunctionWithName(&func_name)
            .ok_or_else(|| format!("metal function {name} not found"))?;
        let raw = self
            .raw
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| format!("metal pipeline {name}: {e:?}"))?;
        let pipeline = Pipeline { raw };
        cache.insert(key, pipeline.clone());
        Ok(pipeline)
    }

    pub fn with_encoder<R>(&self, f: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>) -> R) -> R {
        let mut manager = self.encoder.lock().unwrap();
        manager.ensure_encoder();
        let encoder = &manager.current.as_ref().unwrap().1;
        let out = f(encoder);
        manager.finish_dispatch();
        out
    }

    #[track_caller]
    pub fn synchronize(&self) -> crate::err::Res<()> {
        let t = std::time::Instant::now();
        if std::env::var_os("EFFECT_TORCH_SYNC_TRACE").is_some() {
            eprintln!("[sync] {}", std::panic::Location::caller());
        }
        self.encoder.lock().unwrap().synchronize()?;
        // The GPU has consumed everything submitted so far; retired
        // uploads may return to the pool.
        self.retired.lock().unwrap().clear();
        SYNCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        SYNC_NANOS.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    pub fn grid(width: usize, height: usize, depth: usize) -> MTLSize {
        MTLSize { width, height, depth }
    }
}

pub fn set_buffer(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, index: usize, buffer: &Buffer, offset: usize) {
    unsafe { encoder.setBuffer_offset_atIndex(Some(buffer.as_raw()), offset, index) };
}

pub fn set_bytes<T>(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, index: usize, data: &T) {
    let size = std::mem::size_of::<T>();
    let ptr = NonNull::new(data as *const T as *mut std::ffi::c_void).unwrap();
    unsafe { encoder.setBytes_length_atIndex(ptr, size, index) };
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILL_SRC: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void fill(device float* out [[buffer(0)]], constant float& v [[buffer(1)]], uint i [[thread_position_in_grid]]) {
            out[i] = v;
        }
    "#;

    #[test]
    fn fill_roundtrip() {
        let dev = MetalDevice::get();
        let out = dev.alloc(16, DType::F32);
        let value = 2.5f32;
        let pipeline = dev.compile(0xF111, FILL_SRC, "fill").unwrap();
        dev.with_encoder(|e| {
            e.setComputePipelineState(pipeline.as_raw());
            set_buffer(e, 0, &out, 0);
            set_bytes(e, 1, &value);
            e.dispatchThreads_threadsPerThreadgroup(
                MetalDevice::grid(16, 1, 1),
                MetalDevice::grid(16, 1, 1),
            );
        });
        dev.synchronize();
        assert_eq!(out.read_f32(0, 16), vec![2.5f32; 16]);
    }

    #[test]
    fn pool_reuse() {
        let dev = MetalDevice::get();
        let a = dev.alloc(64, DType::F32);
        let ptr1 = a.contents_ptr() as usize;
        drop(a);
        let b = dev.alloc(64, DType::F32);
        assert_eq!(ptr1, b.contents_ptr() as usize);
    }

    const ZSRC: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void zprobe(device float* out [[buffer(0)]], uint3 tgid [[threadgroup_position_in_grid]]) {
            out[tgid.z] = (float)tgid.z + 1.0f;
        }
    "#;

    #[test]
    fn z_dispatch() {
        let dev = MetalDevice::get();
        let out = dev.alloc(4, DType::F32);
        let pipeline = dev.compile(0x2222, ZSRC, "zprobe").unwrap();
        dev.with_encoder(|e| {
            e.setComputePipelineState(pipeline.as_raw());
            set_buffer(e, 0, &out, 0);
            e.dispatchThreadgroups_threadsPerThreadgroup(
                MetalDevice::grid(1, 1, 4),
                MetalDevice::grid(32, 1, 1),
            );
        });
        dev.synchronize();
        assert_eq!(out.read_f32(0, 4), vec![1.0, 2.0, 3.0, 4.0]);
    }
}

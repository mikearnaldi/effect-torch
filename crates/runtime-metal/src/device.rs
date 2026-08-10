use crate::runtime::dtype::DType;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

const PROBES: usize = 8;
const MAX_BUCKET: usize = 4096;
const DISPATCHES_PER_BUFFER: usize = 4096;

pub static DISPATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static SYNCS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static SYNC_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PIPELINE_COMPILE_MISSES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static EXECUTABLE_PIPELINE_MISS_ATTEMPTS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static EXECUTABLE_ALLOCATION_ATTEMPTS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
thread_local! {
    static TEST_DEVICE_BUFFER_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_test_device_buffer_allocations() {
    TEST_DEVICE_BUFFER_ALLOCATIONS.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn test_device_buffer_allocations() -> usize {
    TEST_DEVICE_BUFFER_ALLOCATIONS.with(Cell::get)
}

#[cfg(test)]
fn record_test_device_buffer_allocation() {
    TEST_DEVICE_BUFFER_ALLOCATIONS.with(|count| count.set(count.get() + 1));
}

// One logical Metal command stream at a time. The encoder manager itself is
// mutex-protected per dispatch, but interleaving whole operation sequences can
// let one thread synchronize and inspect another thread's partial sequence.
static EXECUTION_LOCK: Mutex<()> = Mutex::new(());

thread_local! {
    static EXECUTION_GUARD: RefCell<Option<MutexGuard<'static, ()>>> = const { RefCell::new(None) };
    static EXPLICIT_EXECUTION_DEPTH: Cell<usize> = const { Cell::new(0) };
}

fn claim_execution() {
    EXECUTION_GUARD.with(|guard| {
        if guard.borrow().is_none() {
            *guard.borrow_mut() = Some(
                EXECUTION_LOCK
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()),
            );
        }
    });
}

fn release_automatic_execution() {
    if EXPLICIT_EXECUTION_DEPTH.with(Cell::get) == 0 {
        EXECUTION_GUARD.with(|guard| {
            guard.borrow_mut().take();
        });
    }
}

pub(crate) struct MetalExecutionGuard;

pub(crate) fn begin_execution() -> MetalExecutionGuard {
    claim_execution();
    EXPLICIT_EXECUTION_DEPTH.with(|depth| depth.set(depth.get() + 1));
    MetalExecutionGuard
}

#[cfg(test)]
pub(crate) fn execution_claimed_for_test() -> bool {
    EXECUTION_GUARD.with(|guard| guard.borrow().is_some())
}

impl Drop for MetalExecutionGuard {
    fn drop(&mut self) {
        EXPLICIT_EXECUTION_DEPTH.with(|depth| {
            let next = depth.get().saturating_sub(1);
            depth.set(next);
            if next == 0 {
                EXECUTION_GUARD.with(|guard| {
                    guard.borrow_mut().take();
                });
            }
        });
    }
}

struct AutomaticExecutionRelease;

impl Drop for AutomaticExecutionRelease {
    fn drop(&mut self) {
        release_automatic_execution();
    }
}

#[cfg(test)]
thread_local! {
    static INJECTED_PRIOR_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub fn inject_prior_command_buffer_failure_for_test() {
    INJECTED_PRIOR_FAILURE.with(|failure| failure.set(true));
}

// Bytes of driver-allocated root buffers currently alive (pool, workspace,
// uploads). Suballocations share their segment's root and are not
// counted). A hard ceiling, set with EFFECT_TORCH_MEMORY_CAP_MB, turns
// memory runaways into a loud failure instead of a system freeze.
pub static LIVE_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn memory_cap() -> Option<usize> {
    static CAP: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("EFFECT_TORCH_MEMORY_CAP_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|mb| mb * 1024 * 1024)
    })
}

fn live_bytes_track(size: usize) {
    let live = LIVE_BYTES.fetch_add(size, std::sync::atomic::Ordering::Relaxed) + size;
    if let Some(cap) = memory_cap() {
        if live > cap {
            MetalDevice::get().dump_live_bytes();
            panic!(
                "metal: memory cap exceeded — {} MB live, cap {} MB (EFFECT_TORCH_MEMORY_CAP_MB)",
                live >> 20,
                cap >> 20
            );
        }
    }
}

fn live_bytes_untrack(size: usize) {
    LIVE_BYTES.fetch_sub(size, std::sync::atomic::Ordering::Relaxed);
}

// Host-GPU divergence guard. The walk encodes far faster than the GPU
// executes; without a bound, buffers pile up in dead pool buckets and
// in-flight command buffers faster than the driver can reclaim them,
// and a command buffer fails with kIOGPUCommandBufferCallbackError-
// OutOfMemory (this is what the pre-RFC-0016 mid-step index readback
// accidentally prevented by syncing every step). When live bytes pass
// the budget — the env cap if set, else 3/4 of the device's
// recommended working set — dead buckets are moved to the retired list
// and the host waits on the oldest in-flight command buffer until
// pressure subsides. Steps that fit never wait.
fn memory_budget() -> usize {
    static BUDGET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        let recommended = MetalDevice::get().raw.recommendedMaxWorkingSetSize() as usize;
        let budget = match std::env::var("EFFECT_TORCH_MEMORY_BUDGET_MB") {
            Ok(v) => {
                v.parse::<usize>()
                    .expect("EFFECT_TORCH_MEMORY_BUDGET_MB: not a number")
                    * 1024
                    * 1024
            }
            Err(_) => match memory_cap() {
                Some(cap) => cap.min(recommended / 2),
                None => recommended / 2,
            },
        };
        if std::env::var_os("EFFECT_TORCH_SYNC_TRACE").is_some() {
            eprintln!(
                "[sync] memory budget {} MB (recommended working set {} MB)",
                budget >> 20,
                recommended >> 20
            );
        }
        budget
    })
}

/// Forces all process-global memory policy environment reads at compilation.
/// Subsequent execution only observes these immutable snapshots.
pub fn snapshot_global_environment() {
    let _ = memory_cap();
    let _ = memory_budget();
}

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
    // Byte offset of this buffer's start within `raw`; planned slices share
    // one underlying MTLBuffer.
    pub base: usize,
    // Whether this handle owns driver memory or retains a segment root.
    counted: bool,
    _owner: Option<Arc<Buffer>>,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if self.counted {
            live_bytes_untrack(self.size);
        }
    }
}

impl Buffer {
    pub fn from_raw(raw: Retained<ProtocolObject<dyn MTLBuffer>>, size: usize) -> Self {
        Buffer {
            raw,
            size,
            base: 0,
            counted: false,
            _owner: None,
        }
    }

    pub fn suballoc(segment: &Arc<Buffer>, base: usize, size: usize) -> Self {
        assert!(base + size <= segment.size);
        Buffer {
            raw: segment.raw.clone(),
            size,
            base: segment.base + base,
            counted: false,
            _owner: Some(segment.clone()),
        }
    }

    pub fn contents_ptr(&self) -> *mut std::ffi::c_void {
        unsafe {
            self.raw
                .contents()
                .cast::<u8>()
                .add(self.base)
                .as_ptr()
                .cast()
        }
    }

    pub fn as_raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }

    pub fn read_f32(&self, offset_elems: usize, n: usize) -> Vec<f32> {
        assert!(offset_elems * 4 + n * 4 <= self.size);
        let ptr = unsafe { self.contents_ptr().cast::<f32>().add(offset_elems) };
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    }

    pub fn write_f32(&mut self, offset_elems: usize, data: &[f32]) {
        assert!(offset_elems * 4 + data.len() * 4 <= self.size);
        let ptr = unsafe { self.contents_ptr().cast::<f32>().add(offset_elems) };
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
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
    // Command-buffer serialization. The allocator recycles buffers
    // across command buffers and Metal may overlap their execution —
    // commit order is not execution order — so each buffer waits on
    // the event its predecessor signals. GPU-side ordering only; the
    // host never blocks on this. (Dense byte-budgeted commits made the
    // overlap a real corruption source at batch 128+: NaN losses.)
    order_event: Retained<ProtocolObject<dyn objc2_metal::MTLEvent>>,
    order_value: u64,
    // Submitted command buffers, each holding the buffers retired
    // before its commit. The queue is serial, so once a command buffer
    // completes every command buffer that could still reference those
    // retired blocks has finished: reaping completed entries drops them
    // back to the driver mid-step instead of accumulating until
    // synchronize (a mid-step readback used to force that drain;
    // without it, large command streams exhausted the driver,
    // kIOGPUCommandBufferCallbackErrorOutOfMemory at batch 256).
    in_flight: Vec<(
        u64,
        Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        Vec<Arc<Buffer>>,
    )>,
    // Failures survive mid-step reaping and backpressure waits. They are
    // consumed only after synchronize has drained every submitted buffer.
    failures: Vec<(u64, String)>,
}

impl EncoderManager {
    fn new(
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Self {
        let event = device.newSharedEvent().expect("metal shared event");
        // The wait/signal API takes the MTLEvent super-protocol; same
        // object, rewrapped at the type level.
        let order_event: Retained<ProtocolObject<dyn objc2_metal::MTLEvent>> =
            unsafe { Retained::cast_unchecked(event) };
        EncoderManager {
            queue,
            current: None,
            count: 0,
            order_event,
            order_value: 0,
            in_flight: Vec::new(),
            failures: Vec::new(),
        }
    }

    fn record_failure(&mut self, sequence: u64, cb: &ProtocolObject<dyn MTLCommandBuffer>) {
        if cb.status() != objc2_metal::MTLCommandBufferStatus::Error {
            return;
        }
        let description = cb
            .error()
            .map(|error| error.localizedDescription().to_string())
            .unwrap_or_else(|| "unknown error".to_string());
        self.failures.push((sequence, description));
    }

    fn reap_completed(&mut self) {
        while let Some((_, cb, _)) = self.in_flight.first() {
            let done = matches!(
                cb.status(),
                objc2_metal::MTLCommandBufferStatus::Completed
                    | objc2_metal::MTLCommandBufferStatus::Error
            );
            if done {
                let (sequence, cb, _) = self.in_flight.remove(0);
                self.record_failure(sequence, &cb);
            } else {
                break;
            }
        }
    }

    // Waits for the oldest submitted command buffer and reaps it,
    // dropping the retired blocks it carried. Returns false when no
    // command buffer is in flight (nothing more to reclaim).
    fn wait_oldest(&mut self) -> bool {
        if self.in_flight.is_empty() {
            return false;
        }
        if std::env::var_os("EFFECT_TORCH_SYNC_TRACE").is_some() {
            eprintln!(
                "[sync] backpressure: waiting on oldest command buffer ({} in flight)",
                self.in_flight.len()
            );
        }
        self.in_flight[0].1.waitUntilCompleted();
        self.reap_completed();
        true
    }

    fn ensure_encoder(&mut self) {
        self.reap_completed();
        if self.current.is_none() {
            let cb = self.queue.commandBuffer().expect("metal command buffer");
            if self.order_value > 0 {
                // Wait for the predecessor's signal before executing any
                // dispatch in this buffer.
                cb.encodeWaitForEvent_value(&self.order_event, self.order_value);
            }
            let encoder = cb.computeCommandEncoder().expect("metal compute encoder");
            self.current = Some((cb, encoder));
        }
    }

    fn finish_dispatch(&mut self, retired: &mut Vec<Arc<Buffer>>, allow_automatic_commit: bool) {
        // Untracked hazards: without a barrier, Metal may overlap adjacent
        // compute dispatches in the same command buffer. Our allocator
        // recycles buffers across dispatches, so every dispatch must be
        // ordered after the previous one.
        if let Some((_, encoder)) = &self.current {
            encoder.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
        }
        self.count += 1;
        DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if allow_automatic_commit
            && (self.count >= DISPATCHES_PER_BUFFER || cb_referenced_bytes() >= CB_REF_BYTES)
        {
            self.commit(retired);
        }
    }

    fn commit(&mut self, retired: &mut Vec<Arc<Buffer>>) {
        if let Some((cb, encoder)) = self.current.take() {
            encoder.endEncoding();
            self.order_value += 1;
            cb.encodeSignalEvent_value(&self.order_event, self.order_value);
            cb.commit();
            self.in_flight
                .push((self.order_value, cb, std::mem::take(retired)));
            self.count = 0;
            cb_refs_reset();
        }
    }

    fn synchronize(&mut self, retired: &mut Vec<Arc<Buffer>>) -> crate::err::Res<()> {
        self.commit(retired);
        while !self.in_flight.is_empty() {
            self.in_flight[0].1.waitUntilCompleted();
            self.reap_completed();
        }
        command_buffer_result(std::mem::take(&mut self.failures))
    }
}

fn command_buffer_result(failures: Vec<(u64, String)>) -> crate::err::Res<()> {
    if failures.is_empty() {
        return Ok(());
    }
    let count = failures.len();
    let descriptions = failures
        .into_iter()
        .map(|(sequence, description)| format!("#{sequence}: {description}"))
        .collect::<Vec<_>>()
        .join("; ");
    Err(format!(
        "metal: {count} GPU command buffer failure(s) ({descriptions}); GPU work was lost; this is usually device memory exhaustion"
    ))
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

static SHARED_OPTIONS: MTLResourceOptions = MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

thread_local! {
    static PRIVATE_INTERMEDIATES: std::cell::Cell<Option<bool>> = const {
        std::cell::Cell::new(None)
    };
    static MMA_ENABLED: std::cell::Cell<Option<bool>> = const {
        std::cell::Cell::new(None)
    };
    static EXECUTABLE_DISPATCH_GUARD: std::cell::Cell<bool> = const {
        std::cell::Cell::new(false)
    };
}

#[derive(Debug)]
pub(crate) struct ForbiddenExecutableAllocation {
    operation: &'static str,
}

impl std::fmt::Display for ForbiddenExecutableAllocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MetalDevice::{} is forbidden during executable dispatch",
            self.operation
        )
    }
}

pub(crate) struct ExecutableDispatchGuard;

impl Drop for ExecutableDispatchGuard {
    fn drop(&mut self) {
        EXECUTABLE_DISPATCH_GUARD.with(|active| active.set(false));
    }
}

/// Applies a compile-time allocation policy to all nested backend calls.
pub fn with_execution_environment<R>(private: bool, mma: bool, f: impl FnOnce() -> R) -> R {
    PRIVATE_INTERMEDIATES.with(|policy| {
        MMA_ENABLED.with(|mma_policy| {
            let previous_private = policy.replace(Some(private));
            let previous_mma = mma_policy.replace(Some(mma));
            struct Restore<'a> {
                private: &'a std::cell::Cell<Option<bool>>,
                previous_private: Option<bool>,
                mma: &'a std::cell::Cell<Option<bool>>,
                previous_mma: Option<bool>,
            }
            impl Drop for Restore<'_> {
                fn drop(&mut self) {
                    self.private.set(self.previous_private);
                    self.mma.set(self.previous_mma);
                }
            }
            let _restore = Restore {
                private: policy,
                previous_private,
                mma: mma_policy,
                previous_mma,
            };
            f()
        })
    })
}

fn private_intermediates() -> bool {
    PRIVATE_INTERMEDIATES.with(|policy| {
        policy.get().unwrap_or_else(|| {
            static DEFAULT: OnceLock<bool> = OnceLock::new();
            *DEFAULT
                .get_or_init(|| std::env::var_os("EFFECT_TORCH_PRIVATE_INTERMEDIATES").is_some())
        })
    })
}

pub fn mma_enabled() -> bool {
    MMA_ENABLED.with(|policy| {
        policy.get().unwrap_or_else(|| {
            static DEFAULT: OnceLock<bool> = OnceLock::new();
            *DEFAULT.get_or_init(|| std::env::var_os("EFFECT_TORCH_NO_MMA").is_none())
        })
    })
}

pub fn is_available() -> bool {
    objc2_metal::MTLCopyAllDevices()
        .iter()
        .any(|device| device.newCommandQueue().is_some() && device.newSharedEvent().is_some())
}

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
        let raw = devices.to_vec().swap_remove(ordinal.min(devices.len() - 1));
        let queue = raw
            .newCommandQueue()
            .ok_or("failed to create command queue")?;
        let encoder = EncoderManager::new(queue, &raw);
        Ok(MetalDevice {
            raw,
            allocator: Mutex::new(Allocator::new()),
            encoder: Mutex::new(encoder),
            pipelines: Mutex::new(HashMap::new()),
            retired: Mutex::new(Vec::new()),
        })
    }

    pub fn raw(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.raw
    }

    // See memory_budget: stalls the encoder only when the driver's own
    // allocation counter passes the budget, by waiting on the oldest
    // in-flight command buffer so the driver reclaims what it has
    // finished with. currentAllocatedSize sees everything the driver
    // holds (including what LIVE_BYTES does not). Lock order is
    // encoder -> allocator -> retired, matching synchronize.
    fn backpressure(&self) {
        if self.raw.currentAllocatedSize() <= memory_budget() {
            return;
        }
        let mut manager = self.encoder.lock().unwrap();
        {
            let mut alloc = self.allocator.lock().unwrap();
            let mut retired = self.retired.lock().unwrap();
            for (bucket_size, bucket) in alloc.buckets.iter_mut() {
                if *bucket_size < (1 << 20) {
                    continue;
                }
                bucket.retain(|b| {
                    if Arc::strong_count(b) == 1 {
                        retired.push(b.clone());
                        false
                    } else {
                        true
                    }
                });
            }
        }
        manager.reap_completed();
        if std::env::var_os("EFFECT_TORCH_SYNC_TRACE").is_some() {
            eprintln!(
                "[sync] backpressure at {} MB driver-allocated (budget {} MB)",
                self.raw.currentAllocatedSize() >> 20,
                memory_budget() >> 20
            );
        }
        while self.raw.currentAllocatedSize() > memory_budget() {
            if !manager.wait_oldest() {
                break;
            }
        }
    }

    pub fn alloc(&self, elements: usize, dtype: DType) -> Arc<Buffer> {
        self.reject_executable_allocation("alloc");
        let size = elements * dtype.size_in_bytes();
        self.backpressure();
        let bucket_size = if size >= (64 << 20) {
            // Large blocks: power-of-two bucketing pins up to 2x the
            // request per live block (a 1.1 GB activation would hold a
            // 2 GB buffer), which is what actually exhausts the driver
            // on large dynamic runs. Round to 64 MB pages instead; reuse
            // still works between equal-size requests.
            size.next_multiple_of(64 << 20)
        } else {
            size.next_power_of_two().max(16)
        };
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
                    let buffer = bucket.swap_remove(idx);
                    return buffer;
                }
            }
        }
        let options = if private_intermediates() {
            MTLResourceOptions(
                MTLResourceOptions::StorageModePrivate.0
                    | MTLResourceOptions::HazardTrackingModeUntracked.0,
            )
        } else {
            SHARED_OPTIONS
        };
        live_bytes_track(bucket_size);
        let raw = self
            .raw
            .newBufferWithLength_options(bucket_size, options)
            .expect("metal buffer allocation failed");
        #[cfg(test)]
        record_test_device_buffer_allocation();
        let buffer = Arc::new(Buffer {
            raw,
            size: bucket_size,
            base: 0,
            counted: true,
            _owner: None,
        });
        if bucket.len() < MAX_BUCKET {
            bucket.push(buffer.clone());
        }
        buffer
    }

    /// A right-sized root buffer outside the dynamic tensor pool.
    pub fn alloc_raw(&self, size: usize) -> Arc<Buffer> {
        self.alloc_raw_checked(size)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn alloc_raw_checked(&self, size: usize) -> Result<Arc<Buffer>, String> {
        self.reject_executable_allocation("alloc_raw_checked");
        self.backpressure();
        live_bytes_track(size.max(1));
        let Some(raw) = self
            .raw
            .newBufferWithLength_options(size.max(1), SHARED_OPTIONS)
        else {
            live_bytes_untrack(size.max(1));
            return Err(format!(
                "metal buffer allocation failed: requested {size} bytes, current allocated {} bytes, recommended working set {} bytes",
                self.raw.currentAllocatedSize(),
                self.raw.recommendedMaxWorkingSetSize()
            ));
        };
        #[cfg(test)]
        record_test_device_buffer_allocation();
        Ok(Arc::new(Buffer {
            raw,
            size: size.max(1),
            base: 0,
            counted: true,
            _owner: None,
        }))
    }

    pub fn alloc_with_data(&self, data: &[f32]) -> Arc<Buffer> {
        self.upload_bytes(unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        })
    }

    pub fn alloc_with_data_u32(&self, data: &[u32]) -> Arc<Buffer> {
        self.upload_bytes(unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        })
    }

    pub fn upload_bytes(&self, data: &[u8]) -> Arc<Buffer> {
        self.reject_executable_allocation("upload_bytes");
        // Length is exactly data.len(): newBufferWithBytes copies that
        // many bytes from the source — a rounded-up (bucketed) length
        // would read past the end of the caller's allocation. Uploads
        // are never pooled, so bucketing buys nothing.
        let size = data.len().max(1);
        live_bytes_track(size);
        let raw = unsafe {
            self.raw.newBufferWithBytes_length_options(
                NonNull::new(data.as_ptr() as *const std::ffi::c_void as *mut std::ffi::c_void)
                    .unwrap(),
                size,
                SHARED_OPTIONS,
            )
        }
        .expect("metal buffer allocation failed");
        #[cfg(test)]
        record_test_device_buffer_allocation();
        let buffer = Arc::new(Buffer {
            raw,
            size,
            base: 0,
            counted: true,
            _owner: None,
        });
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
    pub fn compile_lazy(
        &self,
        key: u64,
        name: &str,
        make_source: impl FnOnce() -> String,
    ) -> Result<Pipeline, String> {
        match self.pipeline_cached(key) {
            Some(p) => Ok(p),
            None if self.executable_dispatch_active() => {
                EXECUTABLE_PIPELINE_MISS_ATTEMPTS
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Err(format!(
                    "Metal executable dispatch attempted to compile missing pipeline {name} ({key:#x})"
                ))
            }
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
        if self.executable_dispatch_active() {
            EXECUTABLE_PIPELINE_MISS_ATTEMPTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Err(format!(
                "Metal executable dispatch attempted to compile missing pipeline {name} ({key:#x})"
            ));
        }
        PIPELINE_COMPILE_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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

    pub fn with_encoder<R>(
        &self,
        f: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>) -> R,
    ) -> R {
        claim_execution();
        let allow_automatic_commit = !self.executable_dispatch_active();
        let mut manager = self.encoder.lock().unwrap();
        manager.ensure_encoder();
        let encoder = &manager.current.as_ref().unwrap().1;
        let out = f(encoder);
        manager.finish_dispatch(&mut self.retired.lock().unwrap(), allow_automatic_commit);
        out
    }

    pub(crate) fn commit_executable_command(&self) {
        let mut manager = self.encoder.lock().unwrap();
        manager.commit(&mut self.retired.lock().unwrap());
    }

    pub(crate) fn begin_executable_dispatch(&self) -> Result<ExecutableDispatchGuard, String> {
        EXECUTABLE_DISPATCH_GUARD.with(|active| {
            if active.replace(true) {
                active.set(true);
                Err("nested Metal executable dispatch is not supported".to_string())
            } else {
                Ok(ExecutableDispatchGuard)
            }
        })
    }

    pub(crate) fn executable_dispatch_active(&self) -> bool {
        EXECUTABLE_DISPATCH_GUARD.with(std::cell::Cell::get)
    }

    fn reject_executable_allocation(&self, operation: &'static str) {
        if self.executable_dispatch_active() {
            EXECUTABLE_ALLOCATION_ATTEMPTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            std::panic::panic_any(ForbiddenExecutableAllocation { operation });
        }
    }

    #[track_caller]
    pub fn synchronize(&self) -> crate::err::Res<()> {
        claim_execution();
        let _release = AutomaticExecutionRelease;
        let t = std::time::Instant::now();
        if std::env::var_os("EFFECT_TORCH_SYNC_TRACE").is_some() {
            eprintln!("[sync] {}", std::panic::Location::caller());
        }
        {
            let mut manager = self.encoder.lock().unwrap();
            let mut retired = self.retired.lock().unwrap();
            manager.synchronize(&mut retired)?;
            // Anything retired after the last commit rides no command
            // buffer; the GPU is drained, so it drops here.
            retired.clear();
        }
        // The GPU has consumed everything submitted so far; retired
        // uploads may return to the pool. Dead buckets at or above 1 MB
        // are released too; planned workspace owns executable working sets,
        // so the dynamic pool must not accumulate dead giants across
        // phases; small buckets stay for cheap reuse.
        {
            let mut alloc = self.allocator.lock().unwrap();
            let mut retired = self.retired.lock().unwrap();
            for (bucket_size, bucket) in alloc.buckets.iter_mut() {
                if *bucket_size < (1 << 20) {
                    continue;
                }
                bucket.retain(|b| {
                    if Arc::strong_count(b) == 1 {
                        retired.push(b.clone());
                        false
                    } else {
                        true
                    }
                });
            }
            retired.clear();
        }
        SYNCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        SYNC_NANOS.fetch_add(
            t.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        #[cfg(test)]
        if INJECTED_PRIOR_FAILURE.with(|failure| failure.replace(false)) {
            return Err(
                "metal: 1 GPU command buffer failure(s) (#test: injected prior failure); GPU work was lost"
                    .to_string(),
            );
        }
        Ok(())
    }

    /// Breakdown of live bytes by pool bucket (dead = held only by the
    /// pool) plus retired uploads — printed when the memory cap trips.
    /// Lock-free best effort: the dump runs from the allocation path,
    /// which may already hold the allocator lock.
    pub fn dump_live_bytes(&self) {
        let mut rows: Vec<(usize, usize, usize)> = Vec::new();
        if let Ok(alloc) = self.allocator.try_lock() {
            for (bucket_size, bucket) in alloc.buckets.iter() {
                let live = bucket.iter().filter(|b| Arc::strong_count(b) > 1).count();
                let dead = bucket.len() - live;
                if live + dead > 0 {
                    rows.push((*bucket_size, live, dead));
                }
            }
            rows.sort_by_key(|(size, _, _)| std::cmp::Reverse(*size));
        }
        let retired_bytes: usize = self
            .retired
            .try_lock()
            .map(|r| r.iter().map(|b| b.size).sum())
            .unwrap_or(0);
        eprintln!(
            "[mem] live {} MB; pool buckets (size: live/dead): {}; retired {} MB",
            LIVE_BYTES.load(std::sync::atomic::Ordering::Relaxed) >> 20,
            rows.iter()
                .take(12)
                .map(|(s, l, d)| format!("{}MB:{l}/{d}", s >> 20))
                .collect::<Vec<_>>()
                .join(" "),
            retired_bytes >> 20
        );
    }

    pub fn grid(width: usize, height: usize, depth: usize) -> MTLSize {
        MTLSize {
            width,
            height,
            depth,
        }
    }

    /// Grid width of 64-bit flat kernels: index = gid.y * WIDE + gid.x.
    pub const WIDE: usize = 1 << 30;

    /// Grid and threadgroup for a flat elementwise kernel over an
    /// already-padded thread count: 1-D with uint indexing when small,
    /// a 2-D grid with a widened ulong index past u32::MAX threads.
    pub fn grid_flat(padded: usize) -> (MTLSize, MTLSize) {
        if padded > u32::MAX as usize {
            (
                Self::grid(Self::WIDE, padded.div_ceil(Self::WIDE), 1),
                Self::grid(256, 1, 1),
            )
        } else {
            (Self::grid(padded, 1, 1), Self::grid(256, 1, 1))
        }
    }
}

pub fn set_buffer(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    index: usize,
    buffer: &Buffer,
    offset: usize,
) {
    unsafe { encoder.setBuffer_offset_atIndex(Some(buffer.as_raw()), buffer.base + offset, index) };
    cb_track(buffer);
}

// Per-command-buffer referenced-bytes accounting (RFC 0016): a command
// buffer retains every buffer its dispatches reference until it
// completes, so one 4096-dispatch buffer can pin tens of GB that no
// amount of waiting on OLDER buffers reclaims. finish_dispatch commits
// early once the current buffer references CB_REF_BYTES of distinct
// pool memory. Arena views and externally wrapped buffers (counted ==
// false) are skipped because their root is accounted elsewhere.
thread_local! {
    static CB_REFS: std::cell::RefCell<(std::collections::HashSet<u64>, usize)> =
        std::cell::RefCell::new((std::collections::HashSet::new(), 0));
}

const CB_REF_BYTES: usize = 4 << 30;

fn cb_track(buffer: &Buffer) {
    if !buffer.counted {
        return;
    }
    let addr = buffer.as_raw().gpuAddress();
    CB_REFS.with(|c| {
        let mut c = c.borrow_mut();
        if c.0.insert(addr) {
            c.1 += buffer.size;
        }
    });
}

fn cb_referenced_bytes() -> usize {
    CB_REFS.with(|c| c.borrow().1)
}

fn cb_refs_reset() {
    CB_REFS.with(|c| {
        let mut c = c.borrow_mut();
        c.0.clear();
        c.1 = 0;
    });
}

pub fn set_bytes<T>(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    index: usize,
    data: &T,
) {
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
        dev.synchronize().unwrap();
        assert_eq!(out.read_f32(0, 16), vec![2.5f32; 16]);
    }

    #[test]
    fn pool_reuse() {
        let dev = MetalDevice::new(0).unwrap();
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
        dev.synchronize().unwrap();
        assert_eq!(out.read_f32(0, 4), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn command_buffer_failures_report_every_submission() {
        let error = command_buffer_result(vec![
            (2, "first failure".to_string()),
            (5, "later failure".to_string()),
        ])
        .unwrap_err();
        assert!(error.contains("#2: first failure"), "{error}");
        assert!(error.contains("#5: later failure"), "{error}");
    }

    #[test]
    fn uploads_are_live_byte_counted() {
        let upload = MetalDevice::get().upload_bytes(&[1, 2, 3, 4]);
        assert!(upload.counted);
        MetalDevice::get().synchronize().unwrap();
    }

    #[test]
    fn executable_guard_rejects_every_tensor_allocation_entry_point() {
        let device = MetalDevice::new(0).unwrap();
        let guard = device.begin_executable_dispatch().unwrap();
        for allocation in [
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = device.alloc(1, DType::F32);
            })),
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = device.alloc_raw_checked(4);
            })),
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = device.upload_bytes(&[0, 0, 0, 0]);
            })),
        ] {
            assert!(allocation.is_err());
        }
        drop(guard);
        assert_eq!(device.alloc_raw_checked(4).unwrap().size, 4);
    }
}

//! Metal device singleton, buffer pool allocator, encoder/submission
//! management, and pipeline cache.
//!
//! # Ownership model
//!
//! - [`MetalDevice`] is created once (`MetalDevice::get`) and owns the
//!   `MTLDevice`, one shared `MTLCommandQueue`, the bucketed buffer pool,
//!   the pipeline cache, and a registry of live submission contexts.
//! - [`Buffer`] wraps an `MTLBuffer` plus a byte `base` offset: planned
//!   slices share one physical allocation and one `BufferUsage` token.
//!   Every physical allocation is reference-counted through `BufferUsage`,
//!   which also tracks how many pending/in-flight command buffers can
//!   still touch the storage.
//! - Encoding happens inside a `SubmissionContext` (explicit via
//!   `begin_submission`, or an implicit per-thread one). Each context owns
//!   its `EncoderManager`: one open command buffer + compute encoder at a
//!   time, committed when full, on demand, or at synchronize.
//!
//! # Asynchronous command buffer hazards
//!
//! Metal executes committed command buffers asynchronously and may overlap
//! them, and this allocator recycles buffers aggressively. Three
//! mechanisms make that safe:
//!
//! 1. A shared `MTLEvent` serializes consecutive command buffers from one
//!    context (commit order is not execution order on the GPU).
//! 2. A `memoryBarrierWithScope(Buffers)` after every dispatch orders
//!    dispatches within one command buffer (hazard tracking is untracked).
//! 3. `BufferUse` tokens, attached to each committed command buffer, keep
//!    physical storage alive and un-recyclable until the command buffer
//!    completes; retired roots are reaped only at completion boundaries.
//!
//! Host reads must go through `synchronize`/`synchronize_buffer`, which
//! drain in-flight work and surface GPU command buffer failures (device
//! memory exhaustion surfaces as `kIOGPUCommandBufferCallbackErrorOutOfMemory`
//! and is reported as a lost-work error, not silent corruption).
//!
//! # Memory policy
//!
//! Live driver-allocated root bytes are accounted in [`LIVE_BYTES`] with
//! an optional hard cap (`EFFECT_TORCH_MEMORY_CAP_MB`). A soft budget
//! (`EFFECT_TORCH_MEMORY_BUDGET_MB`, default half the recommended working
//! set) triggers host-GPU backpressure: dead pool buckets are retired and
//! the host waits on the oldest in-flight command buffer until pressure
//! subsides. Environment policy is snapshotted once by
//! [`snapshot_global_environment`]; later execution observes only the
//! immutable snapshot.

use crate::runtime::dtype::DType;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// Linear probes per pool bucket before allocating a fresh root buffer.
const PROBES: usize = 8;
/// Maximum cached buffers per size bucket.
const MAX_BUCKET: usize = 4096;
/// Dispatches after which an open command buffer is committed.
const DISPATCHES_PER_BUFFER: usize = 4096;

/// Total compute dispatches encoded since the last [`dispatch_stats_reset`].
pub static DISPATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Total device-wide synchronizations since the last reset.
pub static SYNCS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Total nanoseconds spent inside `synchronize` since the last reset.
pub static SYNC_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Pipeline cache misses that required a full MSL compile.
pub static PIPELINE_COMPILE_MISSES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Compile attempts rejected because executable dispatch forbids them.
pub static EXECUTABLE_PIPELINE_MISS_ATTEMPTS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Allocation attempts rejected because executable dispatch forbids them.
pub static EXECUTABLE_ALLOCATION_ATTEMPTS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Monotonic sequence number assigned to each committed command buffer;
/// used to find the globally oldest in-flight submission for backpressure.
static SUBMISSION_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Monotonic id for each created `MetalDevice` (multi-device tests).
static DEVICE_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
thread_local! {
    static INJECTED_PRIOR_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Arms a one-shot injected command buffer failure for the calling thread,
/// consumed by the next `begin_submission`/`synchronize` (tests only).
#[cfg(test)]
pub fn inject_prior_command_buffer_failure_for_test() {
    INJECTED_PRIOR_FAILURE.with(|failure| failure.set(true));
}

#[cfg(test)]
fn take_injected_failure() -> crate::err::Res<()> {
    if INJECTED_PRIOR_FAILURE.with(|failure| failure.replace(false)) {
        Err(
            "metal: 1 GPU command buffer failure(s) (#test: injected prior failure); GPU work was lost"
                .to_string(),
        )
    } else {
        Ok(())
    }
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

fn try_live_bytes_track(size: usize) -> Result<(), String> {
    let cap = memory_cap();
    LIVE_BYTES
        .fetch_update(
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Relaxed,
            |live| {
                let next = live.checked_add(size)?;
                (!cap.is_some_and(|cap| next > cap)).then_some(next)
            },
        )
        .map(|_| ())
        .map_err(|live| {
            let requested = live.saturating_add(size);
            match cap {
                Some(cap) => format!(
                    "metal: memory cap exceeded - {} MB requested live, cap {} MB (EFFECT_TORCH_MEMORY_CAP_MB)",
                    requested >> 20,
                    cap >> 20
                ),
                None => "metal: live-byte accounting overflow".to_string(),
            }
        })
}

fn live_bytes_track(size: usize) {
    if let Err(error) = try_live_bytes_track(size) {
        MetalDevice::get().dump_live_bytes();
        panic!("{error}");
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
// the budget — the env cap if set, else 1/2 of the device's
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

/// Resets and returns the (dispatches, syncs, sync-nanos) counters.
pub fn dispatch_stats_reset() -> (u64, u64, u64) {
    let d = DISPATCHES.swap(0, std::sync::atomic::Ordering::Relaxed);
    let s = SYNCS.swap(0, std::sync::atomic::Ordering::Relaxed);
    let n = SYNC_NANOS.swap(0, std::sync::atomic::Ordering::Relaxed);
    (d, s, n)
}
const SWEEP_MS: u64 = 100;
/// Opaque owner kept alive by a buffer view: a workspace pool lease (or any
/// other resource) whose drop releases the underlying allocation. Views
/// clone it so dropping the root early cannot recycle pooled storage.
pub(crate) type BufferRetention = Arc<dyn Send + Sync>;

/// Per-physical-allocation bookkeeping shared by every [`Buffer`] view of
/// the same storage.
struct BufferUsage {
    // Keeps the physical allocation alive after its last Buffer wrapper drops
    // but while an encoded command buffer can still access it.
    _raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    tracked_size: Option<usize>,
    pending_or_in_flight: std::sync::atomic::AtomicUsize,
    producer: Mutex<Option<Weak<SubmissionContext>>>,
    producer_failure: Mutex<Option<String>>,
}

impl BufferUsage {
    fn new(raw: &Retained<ProtocolObject<dyn MTLBuffer>>, tracked_size: Option<usize>) -> Self {
        Self {
            _raw: raw.clone(),
            tracked_size,
            pending_or_in_flight: std::sync::atomic::AtomicUsize::new(0),
            producer: Mutex::new(None),
            producer_failure: Mutex::new(None),
        }
    }

    /// True while at least one pending or in-flight command buffer can
    /// still read or write this storage; the allocator must not recycle it.
    fn in_use(&self) -> bool {
        self.pending_or_in_flight
            .load(std::sync::atomic::Ordering::Acquire)
            > 0
    }

    /// Records `producer` as the submission context that will write this
    /// storage next; clears any previously published producer failure.
    fn set_producer(&self, producer: &Arc<SubmissionContext>) {
        self.producer_failure
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        *self
            .producer
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(Arc::downgrade(producer));
    }

    /// Blocks until the recorded producer of this storage has drained its
    /// pending work, unless `consumer` is that same context (a context never
    /// waits on itself). Propagates a recorded producer failure to every
    /// dependent, once per dependent.
    fn synchronize_producer(
        &self,
        consumer: Option<&Arc<SubmissionContext>>,
    ) -> crate::err::Res<()> {
        if let Some(error) = self
            .producer_failure
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
        {
            return Err(error);
        }
        let producer = self
            .producer
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .as_ref()
            .and_then(Weak::upgrade);
        let Some(producer) = producer else {
            return Ok(());
        };
        if consumer.is_some_and(|consumer| Arc::ptr_eq(consumer, &producer)) {
            return Ok(());
        }
        let result = if producer.has_pending_work() {
            producer.synchronize()
        } else {
            Ok(())
        };
        self.complete_producer(producer.as_ref(), &result);
        result
    }

    /// Called when a producer context finishes synchronizing: clears the
    /// producer registration and, on failure, publishes the error so later
    /// consumers fail fast instead of reading lost GPU work.
    fn complete_producer(&self, producer: &SubmissionContext, result: &crate::err::Res<()>) {
        let mut current = self
            .producer
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if current
            .as_ref()
            .and_then(Weak::upgrade)
            .is_some_and(|current| std::ptr::eq(current.as_ref(), producer))
        {
            if let Err(error) = result {
                *self
                    .producer_failure
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner()) = Some(error.clone());
            }
            current.take();
        }
    }
}

impl Drop for BufferUsage {
    fn drop(&mut self) {
        if let Some(size) = self.tracked_size {
            live_bytes_untrack(size);
        }
    }
}

/// A use-count token held by a command buffer for the lifetime of its
/// pending/in-flight state. Constructing it increments
/// `BufferUsage::pending_or_in_flight`; dropping decrements it. Also
/// carries the buffer's retention so pool leases ride the command buffer.
struct BufferUse {
    usage: Arc<BufferUsage>,
    _retention: Option<BufferRetention>,
}

impl BufferUse {
    fn new(buffer: &Buffer) -> Self {
        buffer
            .usage
            .pending_or_in_flight
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Self {
            usage: buffer.usage.clone(),
            _retention: buffer._retention.clone(),
        }
    }
}

impl Drop for BufferUse {
    fn drop(&mut self) {
        self.usage
            .pending_or_in_flight
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

/// A view onto Metal storage: an `MTLBuffer` handle plus a byte offset.
///
/// Root buffers own a driver allocation; suballocations (`suballoc`) clone
/// the same handle with a nonzero `base` and keep the root alive through
/// `_owner`. All views of one physical allocation share a single
/// `BufferUsage`, so in-flight tracking and producer/consumer
/// synchronization apply to the storage as a whole, not to any one view.
pub struct Buffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub size: usize,
    // Byte offset of this buffer's start within `raw`; planned slices share
    // one underlying MTLBuffer.
    pub base: usize,
    // Shared by every view of one physical allocation.
    usage: Arc<BufferUsage>,
    _owner: Option<Arc<Buffer>>,
    _retention: Option<BufferRetention>,
}

impl Buffer {
    /// Wraps an already-created `MTLBuffer` as a root buffer of `size`
    /// bytes, outside the pool and without live-byte accounting.
    pub fn from_raw(raw: Retained<ProtocolObject<dyn MTLBuffer>>, size: usize) -> Self {
        let usage = Arc::new(BufferUsage::new(&raw, None));
        Buffer {
            raw,
            size,
            base: 0,
            usage,
            _owner: None,
            _retention: None,
        }
    }

    /// Creates a `size`-byte view at byte offset `base` within `segment`,
    /// inheriting the segment's retention.
    ///
    /// # Panics
    ///
    /// Panics if `base + size` exceeds the segment size.
    pub fn suballoc(segment: &Arc<Buffer>, base: usize, size: usize) -> Self {
        Self::suballoc_with_retention(segment, base, size, segment._retention.clone())
    }

    /// Creates a suballocation view that additionally retains `retention`
    /// (e.g. the workspace pool lease backing the segment).
    pub(crate) fn suballoc_with_retention(
        segment: &Arc<Buffer>,
        base: usize,
        size: usize,
        retention: Option<BufferRetention>,
    ) -> Self {
        assert!(base + size <= segment.size);
        Buffer {
            raw: segment.raw.clone(),
            size,
            base: segment.base + base,
            usage: segment.usage.clone(),
            _owner: Some(segment.clone()),
            _retention: retention,
        }
    }

    /// Host pointer to the start of this view's contents (shared storage
    /// mode only; the GPU may not have produced the bytes yet — synchronize
    /// first).
    pub fn contents_ptr(&self) -> *mut std::ffi::c_void {
        // SAFETY: `raw` is a live shared-mode MTLBuffer, so `contents()` is
        // a valid host mapping for the allocation's full length; `base` is
        // bounded by construction (`suballoc` asserts `base + size` fits), so
        // the offset stays in bounds.
        unsafe {
            self.raw
                .contents()
                .cast::<u8>()
                .add(self.base)
                .as_ptr()
                .cast()
        }
    }

    /// The underlying `MTLBuffer` handle (without the view's byte offset).
    pub fn as_raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }

    fn in_use(&self) -> bool {
        self.usage.in_use()
    }

    /// Forwards to the shared usage token; see [`BufferUsage::set_producer`].
    fn set_producer(&self, producer: &Arc<SubmissionContext>) {
        self.usage.set_producer(producer);
    }

    /// Forwards to the shared usage token; see
    /// [`BufferUsage::synchronize_producer`].
    fn synchronize_producer(
        &self,
        consumer: Option<&Arc<SubmissionContext>>,
    ) -> crate::err::Res<()> {
        self.usage.synchronize_producer(consumer)
    }

    /// Copies `n` f32 elements starting at `offset_elems` (relative to this
    /// view) to the host. The caller must have synchronized the producer.
    ///
    /// # Panics
    ///
    /// Panics if the requested element range exceeds the view size.
    pub fn read_f32(&self, offset_elems: usize, n: usize) -> Vec<f32> {
        assert!(offset_elems * 4 + n * 4 <= self.size);
        // SAFETY: the assert bounds the f32 element range within this view;
        // contents_ptr is a valid host mapping of shared storage. The caller
        // contract (post-synchronize) ensures the GPU is not concurrently
        // writing these bytes.
        let ptr = unsafe { self.contents_ptr().cast::<f32>().add(offset_elems) };
        // SAFETY: `ptr` is valid for `n` f32 reads per the bounds above.
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    }

    /// Writes `data` into this view at `offset_elems`. Callers must ensure
    /// no GPU work concurrently reads or writes the range.
    ///
    /// # Panics
    ///
    /// Panics if the write range exceeds the view size.
    pub fn write_f32(&mut self, offset_elems: usize, data: &[f32]) {
        assert!(offset_elems * 4 + data.len() * 4 <= self.size);
        // SAFETY: the assert bounds the write range within this view;
        // `&mut self` guarantees no other host alias of this view, and the
        // caller contract excludes concurrent GPU access.
        let ptr = unsafe { self.contents_ptr().cast::<f32>().add(offset_elems) };
        // SAFETY: source and destination are valid for `data.len()` f32s and
        // cannot overlap (one is host memory, one is device-shared storage).
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
    }
}

/// Per-command-buffer set of physical allocations it can touch, keyed by
/// the shared usage token. Besides keeping storage alive until completion,
/// `referenced_bytes` drives the byte-budgeted early commit.
#[derive(Default)]
struct CommandBufferReferences {
    uses: HashMap<usize, BufferUse>,
    referenced_bytes: usize,
}

impl CommandBufferReferences {
    /// Records that the open command buffer references `buffer`'s storage
    /// (idempotent per physical allocation).
    fn track(&mut self, buffer: &Buffer) {
        let key = Arc::as_ptr(&buffer.usage) as usize;
        if self.uses.contains_key(&key) {
            return;
        }
        self.referenced_bytes = self
            .referenced_bytes
            .saturating_add(buffer.usage.tracked_size.unwrap_or(0));
        self.uses.insert(key, BufferUse::new(buffer));
    }

    /// Hands the accumulated use tokens to a committing command buffer.
    fn take(&mut self) -> Vec<BufferUse> {
        self.referenced_bytes = 0;
        std::mem::take(&mut self.uses).into_values().collect()
    }
}

/// A compiled compute pipeline. Cheap to clone (retained handle); the
/// device caches these by content key, so identical kernels compile once.
#[derive(Clone)]
pub struct Pipeline {
    raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl Pipeline {
    /// The underlying `MTLComputePipelineState` handle.
    pub fn as_raw(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.raw
    }
}

/// Bucketed pool of root buffers, keyed by bucket size. Buffers whose only
/// strong reference is the bucket's are candidates for reuse; busy or
/// in-flight roots are never recycled.
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

    // Buffers whose only host reference is the bucket's may still be read by
    // pending GPU dispatches. Busy roots remain cached; idle roots move to the
    // device retire list and can be released by the next completion boundary.
    fn sweep(&mut self, retired: &mut Vec<Arc<Buffer>>) {
        if self.last_sweep.elapsed() < std::time::Duration::from_millis(SWEEP_MS) {
            return;
        }
        self.last_sweep = std::time::Instant::now();
        for bucket in self.buckets.values_mut() {
            bucket.retain(|b| {
                if Arc::strong_count(b) > 1 || b.in_use() {
                    true
                } else {
                    retired.push(b.clone());
                    false
                }
            });
        }
    }
}

/// Owns one submission stream's encoding state: the open command
/// buffer/encoder pair, the in-flight list, ordering event, and deferred
/// failure records. One per `SubmissionContext`, always mutex-confined.
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
        CommandBufferResources,
        Option<&'static str>,
    )>,
    // Failures survive mid-step reaping and backpressure waits. They are
    // consumed only after synchronize has drained every submitted buffer.
    failures: Vec<(u64, String)>,
    profile_label: Option<&'static str>,
    profile_samples: Vec<(&'static str, f64)>,
}

/// Resources carried by a committed command buffer until it completes:
/// retired roots dropped into it and the use tokens of every referenced
/// physical allocation.
struct CommandBufferResources {
    _retired: Vec<Arc<Buffer>>,
    _uses: Vec<BufferUse>,
}

impl EncoderManager {
    fn new(
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Self {
        let event = device.newSharedEvent().expect("metal shared event");
        // SAFETY: `MTLSharedEvent` refines `MTLEvent` — same object,
        // rewrapped at the type level for the wait/signal API, which takes
        // the super-protocol.
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
            profile_label: None,
            profile_samples: Vec::new(),
        }
    }

    fn set_profile_label(&mut self, label: Option<&'static str>) {
        self.profile_label = label;
    }

    /// Records a failed command buffer's description in `failures`; the
    /// error is surfaced by the next full `synchronize`, never mid-stream.
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

    /// Drops completed (or errored) command buffers from the head of the
    /// in-flight list, releasing their carried resources back to the
    /// driver mid-step. Stops at the first still-running buffer — the
    /// queue is serial, so later buffers cannot have completed either.
    fn reap_completed(&mut self) {
        while let Some((_, cb, _, _)) = self.in_flight.first() {
            let done = matches!(
                cb.status(),
                objc2_metal::MTLCommandBufferStatus::Completed
                    | objc2_metal::MTLCommandBufferStatus::Error
            );
            if done {
                let (sequence, cb, _, label) = self.in_flight.remove(0);
                self.record_failure(sequence, &cb);
                if let Some(label) = label {
                    let elapsed = cb.GPUEndTime() - cb.GPUStartTime();
                    if elapsed.is_finite() && elapsed >= 0.0 {
                        self.profile_samples.push((label, elapsed));
                    }
                }
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

    /// Creates the open command buffer + compute encoder pair if none is
    /// open, wiring the cross-buffer ordering wait first.
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

    /// Closes out one dispatch: intra-buffer memory barrier, counters,
    /// and (unless the caller is a planned executable that commits
    /// explicitly) a size/byte-budgeted automatic commit.
    fn finish_dispatch(
        &mut self,
        references: &mut CommandBufferReferences,
        retired: &mut Vec<Arc<Buffer>>,
        allow_automatic_commit: bool,
    ) {
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
            && (self.count >= DISPATCHES_PER_BUFFER || references.referenced_bytes >= CB_REF_BYTES)
        {
            self.commit(references, retired);
        }
    }

    /// Ends encoding on the open command buffer, signals the ordering
    /// event, commits, and moves it (with its retired roots and use
    /// tokens) onto the in-flight list.
    fn commit(&mut self, references: &mut CommandBufferReferences, retired: &mut Vec<Arc<Buffer>>) {
        if let Some((cb, encoder)) = self.current.take() {
            encoder.endEncoding();
            self.order_value += 1;
            cb.encodeSignalEvent_value(&self.order_event, self.order_value);
            cb.commit();
            let sequence = SUBMISSION_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            self.in_flight.push((
                sequence,
                cb,
                CommandBufferResources {
                    _retired: std::mem::take(retired),
                    _uses: references.take(),
                },
                self.profile_label.take(),
            ));
            self.count = 0;
        }
    }

    /// Commits any open buffer and blocks until every in-flight command
    /// buffer has completed; then reports accumulated failures and drains
    /// profiling samples. After this returns `Ok`, all GPU-visible writes
    /// from this stream are host-visible.
    fn synchronize(
        &mut self,
        references: &mut CommandBufferReferences,
        retired: &mut Vec<Arc<Buffer>>,
    ) -> crate::err::Res<()> {
        self.commit(references, retired);
        while !self.in_flight.is_empty() {
            self.in_flight[0].1.waitUntilCompleted();
            self.reap_completed();
        }
        let result = command_buffer_result(std::mem::take(&mut self.failures));
        if metal_profile_enabled() && !self.profile_samples.is_empty() {
            let mut grouped = HashMap::<&'static str, (usize, f64)>::new();
            for (label, elapsed) in self.profile_samples.drain(..) {
                let entry = grouped.entry(label).or_default();
                entry.0 += 1;
                entry.1 += elapsed;
            }
            let mut grouped = grouped.into_iter().collect::<Vec<_>>();
            grouped.sort_by(|left, right| right.1 .1.total_cmp(&left.1 .1));
            let total = grouped.iter().map(|(_, (_, elapsed))| elapsed).sum::<f64>();
            eprintln!("[metal-profile] total {:.3} ms", total * 1_000.0);
            for (label, (count, elapsed)) in grouped {
                eprintln!(
                    "[metal-profile] {label}: {:.3} ms ({count} command buffers, {:.1}%)",
                    elapsed * 1_000.0,
                    elapsed / total * 100.0
                );
            }
        }
        result
    }

    /// True when this stream has an open buffer, in-flight work, or an
    /// unconsumed failure.
    fn has_pending_work(&self) -> bool {
        self.current.is_some() || !self.in_flight.is_empty() || !self.failures.is_empty()
    }

    /// Submission sequence of the oldest in-flight command buffer.
    fn oldest_submission(&self) -> Option<u64> {
        self.in_flight.first().map(|(sequence, _, _, _)| *sequence)
    }
}

/// Folds recorded command buffer failures into one lost-work error.
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

/// One isolated command stream: its encoder manager, the reference/retired
/// lists riding its commits, a sticky failure slot, and the set of buffers
/// it has been marked as writing (for producer/consumer synchronization).
/// Contexts share the device queue but never share encoders, so concurrent
/// host threads encode independently and failures stay isolated per stream.
struct SubmissionContext {
    device_id: u64,
    manager: Mutex<EncoderManager>,
    references: Mutex<CommandBufferReferences>,
    retired: Mutex<Vec<Arc<Buffer>>>,
    failure: Mutex<Option<String>>,
    writes: Mutex<HashMap<usize, Weak<BufferUsage>>>,
}

impl SubmissionContext {
    fn new(device_id: u64, manager: EncoderManager) -> Self {
        Self {
            device_id,
            manager: Mutex::new(manager),
            references: Mutex::new(CommandBufferReferences::default()),
            retired: Mutex::new(Vec::new()),
            failure: Mutex::new(None),
            writes: Mutex::new(HashMap::new()),
        }
    }

    /// Runs `f` against this stream's open compute encoder, then closes
    /// out the dispatch. Panics if the stream already failed — continuing
    /// to encode after lost GPU work would mask the corruption.
    fn with_encoder<R>(
        self: &Arc<Self>,
        allow_automatic_commit: bool,
        f: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>) -> R,
    ) -> R {
        if let Some(error) = self
            .failure
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
        {
            panic!("Metal submission already failed: {error}");
        }
        let mut manager = self
            .manager
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        manager.ensure_encoder();
        let encoder = &manager.current.as_ref().expect("encoder was created").1;
        let _encoding_guard = EncodingContextGuard::enter(self.clone());
        let out = f(encoder);
        let mut references = self
            .references
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut retired = self
            .retired
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        manager.finish_dispatch(&mut references, &mut retired, allow_automatic_commit);
        out
    }

    /// Commits the open command buffer without waiting.
    fn commit(&self) {
        let mut manager = self
            .manager
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut references = self
            .references
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut retired = self
            .retired
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        manager.commit(&mut references, &mut retired);
    }

    /// Drains the stream (see [`EncoderManager::synchronize`]), sticks any
    /// failure to the context, and publishes the outcome to every buffer
    /// this stream was marked as writing.
    fn synchronize(&self) -> crate::err::Res<()> {
        if let Some(error) = self
            .failure
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
        {
            let result = Err(error);
            self.publish_writes(&result);
            return result;
        }
        let mut manager = self
            .manager
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut references = self
            .references
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut retired = self
            .retired
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let result = manager.synchronize(&mut references, &mut retired);
        // Anything retired after the last commit rides no command buffer; the
        // context is drained, so it is safe to drop now.
        retired.clear();
        if let Err(error) = &result {
            *self
                .failure
                .lock()
                .unwrap_or_else(|poison| poison.into_inner()) = Some(error.clone());
        }
        self.publish_writes(&result);
        result
    }

    fn has_pending_work(&self) -> bool {
        let manager = self
            .manager
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        manager.has_pending_work()
            || self
                .failure
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .is_some()
            || !self
                .references
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .uses
                .is_empty()
            || !self
                .retired
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .is_empty()
    }

    fn reap_completed(&self) {
        self.manager
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .reap_completed();
    }

    fn wait_oldest(&self) -> bool {
        self.manager
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .wait_oldest()
    }

    fn oldest_submission(&self) -> Option<u64> {
        self.manager
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .oldest_submission()
    }

    /// Records `usage` as written by this stream so `publish_writes` can
    /// complete its producer registration at the next drain.
    fn track_write(&self, usage: &Arc<BufferUsage>) {
        self.writes
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(Arc::as_ptr(usage) as usize, Arc::downgrade(usage));
    }

    /// Completes the producer registration of every buffer written by this
    /// stream, propagating `result` (failures become visible to dependents).
    fn publish_writes(&self, result: &crate::err::Res<()>) {
        let writes = std::mem::take(
            &mut *self
                .writes
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
        );
        for usage in writes.into_values().filter_map(|usage| usage.upgrade()) {
            usage.complete_producer(self, result);
        }
    }
}

impl Drop for SubmissionContext {
    /// A dropped context drains its own stream first: no committed work
    /// may outlive the context that owns its failure bookkeeping.
    fn drop(&mut self) {
        let result = {
            let manager = self
                .manager
                .get_mut()
                .unwrap_or_else(|error| error.into_inner());
            let references = self
                .references
                .get_mut()
                .unwrap_or_else(|error| error.into_inner());
            let retired = self
                .retired
                .get_mut()
                .unwrap_or_else(|error| error.into_inner());
            let result = manager.synchronize(references, retired);
            retired.clear();
            result
        };
        self.publish_writes(&result);
    }
}

/// Whether `[metal-profile]` reporting is on (`EFFECT_TORCH_METAL_PROFILE`).
fn metal_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("EFFECT_TORCH_METAL_PROFILE").is_some())
}

/// RAII region attributing committed command buffers to one profile label;
/// dropping commits the stream and clears the label.
pub(crate) struct MetalProfileRegion {
    context: Arc<SubmissionContext>,
}

impl Drop for MetalProfileRegion {
    fn drop(&mut self) {
        self.context.commit();
        self.context
            .manager
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .set_profile_label(None);
    }
}

// SAFETY: Metal command queues, command buffers, and encoders are
// thread-safe at the API boundary. Each EncoderManager is mutex-confined,
// each encoder is used by one host thread at a time, and the reference /
// retired / failure / writes fields are all independently mutex-guarded.
unsafe impl Send for SubmissionContext {}
// SAFETY: see the `Send` impl above; all shared state is synchronized.
unsafe impl Sync for SubmissionContext {}

thread_local! {
    static ACTIVE_SUBMISSIONS: RefCell<Vec<Arc<SubmissionContext>>> = const { RefCell::new(Vec::new()) };
    static IMPLICIT_SUBMISSIONS: RefCell<HashMap<u64, Arc<SubmissionContext>>> = RefCell::new(HashMap::new());
    static ENCODING_SUBMISSIONS: RefCell<Vec<Arc<SubmissionContext>>> = const { RefCell::new(Vec::new()) };
}

/// RAII guard marking `context` as the encoding submission while
/// `with_encoder` runs its closure; lets free functions like [`set_buffer`]
/// find the owning stream for reference tracking.
struct EncodingContextGuard;

impl EncodingContextGuard {
    fn enter(context: Arc<SubmissionContext>) -> Self {
        ENCODING_SUBMISSIONS.with(|contexts| contexts.borrow_mut().push(context));
        Self
    }
}

impl Drop for EncodingContextGuard {
    fn drop(&mut self) {
        ENCODING_SUBMISSIONS.with(|contexts| {
            contexts.borrow_mut().pop();
        });
    }
}

/// RAII handle for an explicit submission (`begin_submission`): dropping
/// it synchronizes any remaining work and pops the thread-local active
/// stack.
pub(crate) struct MetalSubmissionGuard<'a> {
    device: &'a MetalDevice,
    context: Arc<SubmissionContext>,
}

impl Drop for MetalSubmissionGuard<'_> {
    fn drop(&mut self) {
        if self.context.has_pending_work() {
            let _ = self.context.synchronize();
        }
        ACTIVE_SUBMISSIONS.with(|contexts| {
            let popped = contexts.borrow_mut().pop();
            debug_assert!(popped.is_some_and(|context| Arc::ptr_eq(&context, &self.context)));
        });
        debug_assert_eq!(self.context.device_id, self.device.device_id());
    }
}

/// The Metal device facade: raw device + shared command queue, the buffer
/// pool, the pipeline cache, the device-level retired list, and a weak
/// registry of live submission contexts (used for global backpressure).
pub struct MetalDevice {
    id: u64,
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    allocator: Mutex<Allocator>,
    pipelines: Mutex<HashMap<u64, Pipeline>>,
    retired: Mutex<Vec<Arc<Buffer>>>,
    submissions: Mutex<Vec<Weak<SubmissionContext>>>,
}

// SAFETY: Metal device/queue/buffer/pipeline objects are thread-safe at the
// API boundary. All mutable host-side state (allocator, caches, lists) is
// mutex-confined; Buffer fields are immutable after construction except the
// atomics/mutexes inside BufferUsage. GPU-side hazards are handled by the
// ordering event, per-dispatch barriers, and use tokens documented at the
// module level.
unsafe impl Send for MetalDevice {}
// SAFETY: see above; shared access only touches synchronized or immutable
// state.
unsafe impl Sync for MetalDevice {}
// SAFETY: `raw` is an immutable retained handle; `size`/`base` are immutable
// after construction; hazards on the shared storage go through the atomics
// and mutexes in the shared `BufferUsage`.
unsafe impl Send for Buffer {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for Buffer {}
// SAFETY: fields are a retained handle, an atomic, and mutex-guarded
// producer state — all safe to share across threads.
unsafe impl Send for BufferUsage {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for BufferUsage {}

/// Storage options for host-visible buffers: shared storage (unified
/// memory) with untracked hazards — ordering is the runtime's job (event +
/// barriers), not the driver's.
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

/// Panic payload raised when an allocation entry point is hit during
/// executable dispatch (where every byte must come from the plan).
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

/// RAII guard for executable dispatch: while held on this thread,
/// allocation and pipeline compilation are forbidden, and command buffers
/// are only committed explicitly (no size-triggered automatic commit).
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

/// Whether pool allocations default to private storage (only legal for
/// intermediates the host never reads); thread-local policy with an
/// `EFFECT_TORCH_PRIVATE_INTERMEDIATES` env default.
fn private_intermediates() -> bool {
    PRIVATE_INTERMEDIATES.with(|policy| {
        policy.get().unwrap_or_else(|| {
            static DEFAULT: OnceLock<bool> = OnceLock::new();
            *DEFAULT
                .get_or_init(|| std::env::var_os("EFFECT_TORCH_PRIVATE_INTERMEDIATES").is_some())
        })
    })
}

/// Whether simdgroup-MMA kernels are enabled (thread-local policy;
/// `EFFECT_TORCH_NO_MMA` forces the naive tiled path).
pub fn mma_enabled() -> bool {
    MMA_ENABLED.with(|policy| {
        policy.get().unwrap_or_else(|| {
            static DEFAULT: OnceLock<bool> = OnceLock::new();
            *DEFAULT.get_or_init(|| std::env::var_os("EFFECT_TORCH_NO_MMA").is_none())
        })
    })
}

/// True when some Metal device can create a command queue and shared
/// event — the two capabilities this runtime requires.
pub fn is_available() -> bool {
    objc2_metal::MTLCopyAllDevices()
        .iter()
        .any(|device| device.newCommandQueue().is_some() && device.newSharedEvent().is_some())
}

impl MetalDevice {
    /// The process-wide device singleton (ordinal 0).
    ///
    /// # Panics
    ///
    /// Panics if no Metal device is available or the queue cannot be
    /// created.
    pub fn get() -> &'static MetalDevice {
        static DEVICE: OnceLock<MetalDevice> = OnceLock::new();
        DEVICE.get_or_init(|| MetalDevice::new(0).expect("metal device"))
    }

    /// Creates a device handle for `ordinal` (clamped to the last device)
    /// with its own pool, caches, and submission registry.
    pub fn new(ordinal: usize) -> Result<Self, String> {
        let devices = objc2_metal::MTLCopyAllDevices();
        if devices.is_empty() {
            return Err("no Metal devices available".to_string());
        }
        let raw = devices.to_vec().swap_remove(ordinal.min(devices.len() - 1));
        let queue = raw
            .newCommandQueue()
            .ok_or("failed to create command queue")?;
        Ok(MetalDevice {
            id: DEVICE_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            raw,
            queue,
            allocator: Mutex::new(Allocator::new()),
            pipelines: Mutex::new(HashMap::new()),
            retired: Mutex::new(Vec::new()),
            submissions: Mutex::new(Vec::new()),
        })
    }

    /// The underlying `MTLDevice` handle.
    pub fn raw(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.raw
    }

    fn device_id(&self) -> u64 {
        self.id
    }

    fn new_submission_context(&self) -> Arc<SubmissionContext> {
        let context = Arc::new(SubmissionContext::new(
            self.device_id(),
            EncoderManager::new(self.queue.clone(), &self.raw),
        ));
        let mut submissions = self
            .submissions
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        submissions.retain(|submission| submission.strong_count() > 0);
        submissions.push(Arc::downgrade(&context));
        context
    }

    fn active_submission(&self) -> Option<Arc<SubmissionContext>> {
        ACTIVE_SUBMISSIONS.with(|contexts| {
            contexts
                .borrow()
                .iter()
                .rev()
                .find(|context| context.device_id == self.device_id())
                .cloned()
        })
    }

    fn implicit_submission(&self) -> Arc<SubmissionContext> {
        IMPLICIT_SUBMISSIONS.with(|contexts| {
            let mut contexts = contexts.borrow_mut();
            contexts
                .entry(self.device_id())
                .or_insert_with(|| self.new_submission_context())
                .clone()
        })
    }

    fn existing_implicit_submission(&self) -> Option<Arc<SubmissionContext>> {
        IMPLICIT_SUBMISSIONS.with(|contexts| contexts.borrow().get(&self.device_id()).cloned())
    }

    fn remove_implicit_submission(&self) {
        IMPLICIT_SUBMISSIONS.with(|contexts| {
            contexts.borrow_mut().remove(&self.device_id());
        });
    }

    /// The stream new dispatches encode into: the innermost active
    /// explicit submission for this device on this thread, else the
    /// thread's implicit (lazily created) stream.
    fn dispatch_submission(&self) -> Arc<SubmissionContext> {
        self.active_submission()
            .unwrap_or_else(|| self.implicit_submission())
    }

    /// Opens an explicit submission for this device on the current thread.
    /// Errors on nesting; drains a pending implicit stream first so the
    /// explicit stream starts clean. Dropping the guard synchronizes any
    /// remaining work.
    pub(crate) fn begin_submission(&self) -> Result<MetalSubmissionGuard<'_>, String> {
        if self.active_submission().is_some() {
            return Err("nested Metal submission is not supported".to_string());
        }
        if self
            .existing_implicit_submission()
            .is_some_and(|context| context.has_pending_work())
        {
            self.synchronize()?;
        }
        #[cfg(test)]
        take_injected_failure()?;
        let context = self.new_submission_context();
        ACTIVE_SUBMISSIONS.with(|contexts| contexts.borrow_mut().push(context.clone()));
        Ok(MetalSubmissionGuard {
            device: self,
            context,
        })
    }

    fn submission_snapshot(&self) -> Vec<Arc<SubmissionContext>> {
        let mut submissions = self
            .submissions
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut live = Vec::with_capacity(submissions.len());
        submissions.retain(|submission| {
            if let Some(submission) = submission.upgrade() {
                live.push(submission);
                true
            } else {
                false
            }
        });
        live
    }

    // See memory_budget: commit the caller's pending stream, then wait on the
    // globally oldest submitted context until the driver falls below budget.
    // The registry is snapshotted before waiting; no device allocator lock is
    // held while a command buffer completes.
    fn backpressure(&self) {
        if self.raw.currentAllocatedSize() <= memory_budget() {
            return;
        }
        let contexts = self.submission_snapshot();
        for context in &contexts {
            context.commit();
            context.reap_completed();
        }
        {
            let mut alloc = self.allocator.lock().unwrap();
            let mut retired = self.retired.lock().unwrap();
            for (bucket_size, bucket) in alloc.buckets.iter_mut() {
                if *bucket_size < (1 << 20) {
                    continue;
                }
                bucket.retain(|b| {
                    if Arc::strong_count(b) == 1 && !b.in_use() {
                        retired.push(b.clone());
                        false
                    } else {
                        true
                    }
                });
            }
        }
        if std::env::var_os("EFFECT_TORCH_SYNC_TRACE").is_some() {
            eprintln!(
                "[sync] backpressure at {} MB driver-allocated (budget {} MB)",
                self.raw.currentAllocatedSize() >> 20,
                memory_budget() >> 20
            );
        }
        while self.raw.currentAllocatedSize() > memory_budget() {
            let Some(context) = contexts
                .iter()
                .filter_map(|context| {
                    context
                        .oldest_submission()
                        .map(|sequence| (sequence, context))
                })
                .min_by_key(|(sequence, _)| *sequence)
                .map(|(_, context)| context)
            else {
                break;
            };
            if !context.wait_oldest() {
                break;
            }
        }
    }

    /// Allocates a pooled root buffer of at least `elements * dtype` bytes.
    ///
    /// Small requests round up to a power-of-two bucket (min 16 bytes);
    /// requests of 64 MiB or more round to 64 MiB pages so large
    /// activations never pin 2x their size. Idle, non-in-flight buffers of
    /// the same bucket are reused (up to `PROBES` linear probes).
    ///
    /// # Panics
    ///
    /// Panics during executable dispatch, on driver allocation failure,
    /// or when the `EFFECT_TORCH_MEMORY_CAP_MB` live-byte cap trips.
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
                if Arc::strong_count(&bucket[idx]) == 1 && !bucket[idx].in_use() {
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
        let Some(raw) = self.raw.newBufferWithLength_options(bucket_size, options) else {
            live_bytes_untrack(bucket_size);
            panic!("metal buffer allocation failed");
        };
        let usage = Arc::new(BufferUsage::new(&raw, Some(bucket_size)));
        let buffer = Arc::new(Buffer {
            raw,
            size: bucket_size,
            base: 0,
            usage,
            _owner: None,
            _retention: None,
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

    /// Fallible form of [`MetalDevice::alloc_raw`]: returns an error
    /// instead of panicking when the cap trips or the driver refuses.
    pub fn alloc_raw_checked(&self, size: usize) -> Result<Arc<Buffer>, String> {
        self.reject_executable_allocation("alloc_raw_checked");
        self.backpressure();
        try_live_bytes_track(size.max(1))?;
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
        let usage = Arc::new(BufferUsage::new(&raw, Some(size.max(1))));
        Ok(Arc::new(Buffer {
            raw,
            size: size.max(1),
            base: 0,
            usage,
            _owner: None,
            _retention: None,
        }))
    }

    /// Uploads f32 host data into a fresh unpooled buffer.
    pub fn alloc_with_data(&self, data: &[f32]) -> Arc<Buffer> {
        // SAFETY: `data` is a live slice of `data.len()` f32s, so the byte
        // view covers `data.len() * 4` valid bytes; upload_bytes copies
        // synchronously before returning.
        self.upload_bytes(unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        })
    }

    /// Uploads u32 host data into a fresh unpooled buffer.
    pub fn alloc_with_data_u32(&self, data: &[u32]) -> Arc<Buffer> {
        // SAFETY: same argument as alloc_with_data, for u32 elements.
        self.upload_bytes(unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        })
    }

    /// Copies `data` into a fresh exactly-sized shared buffer.
    ///
    /// Uploads are never pooled and are retired until the next
    /// `synchronize`, so concurrent walkers can never recycle the bytes
    /// before the GPU has consumed them.
    ///
    /// # Panics
    ///
    /// Panics during executable dispatch, on driver allocation failure, or
    /// when the live-byte cap trips.
    pub fn upload_bytes(&self, data: &[u8]) -> Arc<Buffer> {
        self.reject_executable_allocation("upload_bytes");
        // Length is exactly data.len(): newBufferWithBytes copies that
        // many bytes from the source — a rounded-up (bucketed) length
        // would read past the end of the caller's allocation. Uploads
        // are never pooled, so bucketing buys nothing.
        let size = data.len().max(1);
        live_bytes_track(size);
        let zero = 0u8;
        let source = if data.is_empty() {
            std::slice::from_ref(&zero)
        } else {
            data
        };
        // SAFETY: `source` is valid for exactly `size` bytes: non-empty input
        // uses the caller's slice (`size == data.len()`), while empty input
        // uses the live one-byte `zero` slice. Metal copies the bytes during
        // this call and does not retain the host pointer.
        let Some(raw) = (unsafe {
            self.raw.newBufferWithBytes_length_options(
                NonNull::new(source.as_ptr() as *const std::ffi::c_void as *mut std::ffi::c_void)
                    .unwrap(),
                size,
                SHARED_OPTIONS,
            )
        }) else {
            live_bytes_untrack(size);
            panic!("metal buffer allocation failed");
        };
        let usage = Arc::new(BufferUsage::new(&raw, Some(size)));
        let buffer = Arc::new(Buffer {
            raw,
            size,
            base: 0,
            usage,
            _owner: None,
            _retention: None,
        });
        // Host uploads retire only at the next synchronize. Uploads are
        // NEVER pooled: the only strong refs are the caller's and this
        // list's, so nothing can recycle the bytes before the GPU has
        // consumed them (concurrent walks/tests share the device).
        self.retired.lock().unwrap().push(buffer.clone());
        buffer
    }

    /// Returns the cached pipeline for `key`, compiling `source` on a
    /// miss. `name` is the MSL kernel function name.
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

    /// Looks up the cached pipeline for `key` without compiling.
    pub fn pipeline_cached(&self, key: u64) -> Option<Pipeline> {
        self.pipelines.lock().unwrap().get(&key).cloned()
    }

    /// Unconditionally compiles and caches `source` under `key` (fast math
    /// disabled — bitwise-stable numerics across recompiles). Errors
    /// instead of compiling during executable dispatch.
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

    /// Encodes one dispatch: runs `f` with the current stream's compute
    /// encoder. The closure must bind buffers via [`set_buffer`] (so the
    /// stream tracks the reference) and end with a `dispatch*` call.
    pub fn with_encoder<R>(
        &self,
        f: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>) -> R,
    ) -> R {
        let allow_automatic_commit = !self.executable_dispatch_active();
        self.dispatch_submission()
            .with_encoder(allow_automatic_commit, f)
    }

    /// Explicitly commits the current stream's open command buffer; used by
    /// executable dispatch, which disables automatic commit.
    pub(crate) fn commit_executable_command(&self) {
        self.dispatch_submission().commit();
    }

    /// Opens a labeled profiling region when `EFFECT_TORCH_METAL_PROFILE`
    /// is set; command buffers committed inside are timed and reported at
    /// the next synchronize.
    pub(crate) fn profile_region(&self, label: &'static str) -> Option<MetalProfileRegion> {
        if !metal_profile_enabled() {
            return None;
        }
        let context = self.dispatch_submission();
        context.commit();
        context
            .manager
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .set_profile_label(Some(label));
        Some(MetalProfileRegion { context })
    }

    /// Marks this thread as executing a planned executable: allocation and
    /// compilation become hard errors, and command buffers commit only
    /// explicitly. Not nestable.
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

    /// Blocks until the calling thread's current stream (explicit or
    /// implicit) is fully drained, then retires oversized idle pool roots.
    /// Returns any GPU command buffer failure recorded since the last
    /// drain. This is the only host-side completion fence.
    #[track_caller]
    pub fn synchronize(&self) -> crate::err::Res<()> {
        let t = std::time::Instant::now();
        if std::env::var_os("EFFECT_TORCH_SYNC_TRACE").is_some() {
            eprintln!("[sync] {}", std::panic::Location::caller());
        }
        let active = self.active_submission();
        let context = active
            .clone()
            .or_else(|| self.existing_implicit_submission());
        let submission_result = context.map_or(Ok(()), |context| context.synchronize());
        if active.is_none() {
            self.remove_implicit_submission();
        }
        submission_result?;
        // This context has consumed its resources. Busy roots owned by another
        // context remain protected by their BufferUsage tokens.
        {
            let mut alloc = self.allocator.lock().unwrap();
            let mut retired = self.retired.lock().unwrap();
            for (bucket_size, bucket) in alloc.buckets.iter_mut() {
                if *bucket_size < (1 << 20) {
                    continue;
                }
                bucket.retain(|b| {
                    if Arc::strong_count(b) == 1 && !b.in_use() {
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
        take_injected_failure()?;
        Ok(())
    }

    /// Waits for whatever stream produced `buffer` (possibly on another
    /// thread), then performs a full [`MetalDevice::synchronize`]. The
    /// host-side read fence for tensor contents.
    #[track_caller]
    pub(crate) fn synchronize_buffer(&self, buffer: &Buffer) -> crate::err::Res<()> {
        let consumer = self
            .active_submission()
            .or_else(|| self.existing_implicit_submission());
        buffer.synchronize_producer(consumer.as_ref())?;
        self.synchronize()
    }

    /// Waits only for `buffer`'s producer stream, without draining the
    /// caller's own stream.
    pub(crate) fn synchronize_buffer_producer(&self, buffer: &Buffer) -> crate::err::Res<()> {
        let consumer = self
            .active_submission()
            .or_else(|| self.existing_implicit_submission());
        buffer.synchronize_producer(consumer.as_ref())
    }

    /// Registers the current stream as `buffer`'s next writer: first waits
    /// on any prior producer (cross-stream hazard), then records the new
    /// producer so later readers synchronize against this stream.
    pub(crate) fn mark_buffer_write(&self, buffer: &Buffer) -> crate::err::Res<()> {
        let producer = self.dispatch_submission();
        buffer.synchronize_producer(Some(&producer))?;
        buffer.set_producer(&producer);
        producer.track_write(&buffer.usage);
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
                let live = bucket
                    .iter()
                    .filter(|b| Arc::strong_count(b) > 1 || b.in_use())
                    .count();
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

    /// Convenience constructor for an `MTLSize`.
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
    let context = ENCODING_SUBMISSIONS.with(|contexts| {
        contexts
            .borrow()
            .last()
            .cloned()
            .expect("set_buffer must be called from MetalDevice::with_encoder")
    });
    unsafe { encoder.setBuffer_offset_atIndex(Some(buffer.as_raw()), buffer.base + offset, index) };
    context
        .references
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .track(buffer);
}

// A command buffer retains unique physical-storage usage tokens until it
// completes. Besides driving byte-budgeted commits, these tokens prevent the
// dynamic allocator from recycling an eager intermediate across contexts.
const CB_REF_BYTES: usize = 4 << 30;

/// Copies the POD value `data` into the encoder's argument space at
/// `index` (Metal copies the bytes immediately, so no lifetime hazard).
pub fn set_bytes<T>(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    index: usize,
    data: &T,
) {
    let size = std::mem::size_of::<T>();
    let ptr = NonNull::new(data as *const T as *mut std::ffi::c_void).unwrap();
    // SAFETY: `ptr` points to `size` valid bytes of the live `data`
    // reference; setBytes copies them synchronously into the command
    // buffer's argument storage, so no borrow outlives this call. Callers
    // pass only POD scalars whose layout matches the kernel signature.
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
    fn implicit_kernel_streams_overlap_and_isolate_failures() {
        let dev = Arc::new(MetalDevice::new(0).unwrap());
        dev.compile(0xF111, FILL_SRC, "fill").unwrap();
        dev.synchronize().unwrap();
        let encoding = Arc::new(std::sync::Barrier::new(2));
        let submitted = Arc::new(std::sync::Barrier::new(2));
        let threads = (0..2)
            .map(|index| {
                let dev = dev.clone();
                let encoding = encoding.clone();
                let submitted = submitted.clone();
                std::thread::spawn(move || {
                    let out = dev.alloc(16, DType::F32);
                    let pipeline = dev.compile(0xF111, FILL_SRC, "fill").unwrap();
                    let value = index as f32 + 1.0;
                    dev.with_encoder(|encoder| {
                        // Both host threads must own an encoder concurrently;
                        // a process-wide execution lock deadlocks here.
                        encoding.wait();
                        encoder.setComputePipelineState(pipeline.as_raw());
                        set_buffer(encoder, 0, &out, 0);
                        set_bytes(encoder, 1, &value);
                        encoder.dispatchThreads_threadsPerThreadgroup(
                            MetalDevice::grid(16, 1, 1),
                            MetalDevice::grid(16, 1, 1),
                        );
                    });
                    submitted.wait();
                    if index == 0 {
                        inject_prior_command_buffer_failure_for_test();
                    }
                    let result = dev.synchronize();
                    (result, out.read_f32(0, 16))
                })
            })
            .collect::<Vec<_>>();
        let mut results = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();
        assert!(results[0].0.is_err());
        assert!(results[1].0.is_ok());
        assert_eq!(results.remove(0).1, vec![1.0; 16]);
        assert_eq!(results.remove(0).1, vec![2.0; 16]);
    }

    #[test]
    fn cross_thread_readback_waits_for_the_eager_producer() {
        let dev = MetalDevice::get();
        crate::kernels::compile_fill(dev, &[16], 3.0, DType::F32).unwrap();
        dev.synchronize().unwrap();
        let (output_tx, output_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let out = crate::run::MetalTensor {
            buffer: dev.alloc(16, DType::F32),
            layout: crate::runtime::layout::Layout::contiguous(vec![16]),
            dtype: DType::F32,
        };
        let worker = std::thread::spawn(move || {
            let dev = MetalDevice::get();
            crate::kernels::fill_into(dev, &out, 3.0).unwrap();
            output_tx.send(out).unwrap();
            release_rx.recv().unwrap();
        });

        let out = output_rx.recv().unwrap();
        dev.synchronize_buffer(&out.buffer).unwrap();
        assert_eq!(out.buffer.read_f32(0, 16), vec![3.0; 16]);
        release_tx.send(()).unwrap();
        worker.join().unwrap();
    }

    #[test]
    fn allocator_does_not_recycle_a_pending_buffer_across_contexts() {
        let dev = Arc::new(MetalDevice::new(0).unwrap());
        dev.compile(0xF111, FILL_SRC, "fill").unwrap();
        dev.synchronize().unwrap();
        let pending = Arc::new(std::sync::Barrier::new(2));
        let release = Arc::new(std::sync::Barrier::new(2));
        let worker = {
            let dev = dev.clone();
            let pending = pending.clone();
            let release = release.clone();
            std::thread::spawn(move || {
                let out = dev.alloc(16, DType::F32);
                let address = out.contents_ptr() as usize;
                let pipeline = dev.compile(0xF111, FILL_SRC, "fill").unwrap();
                dev.with_encoder(|encoder| {
                    encoder.setComputePipelineState(pipeline.as_raw());
                    set_buffer(encoder, 0, &out, 0);
                    set_bytes(encoder, 1, &1.0f32);
                    encoder.dispatchThreads_threadsPerThreadgroup(
                        MetalDevice::grid(16, 1, 1),
                        MetalDevice::grid(16, 1, 1),
                    );
                });
                drop(out);
                pending.wait();
                release.wait();
                dev.synchronize().unwrap();
                address
            })
        };
        pending.wait();
        let concurrent = dev.alloc(16, DType::F32);
        let concurrent_address = concurrent.contents_ptr() as usize;
        release.wait();
        let pending_address = worker.join().unwrap();
        assert_ne!(pending_address, concurrent_address);
    }

    #[test]
    fn pending_buffer_use_retains_its_workspace_owner() {
        let dev = MetalDevice::new(0).unwrap();
        let root = dev.alloc_raw(16);
        let retention: BufferRetention = Arc::new(());
        let weak_retention = Arc::downgrade(&retention);
        let buffer = Buffer::suballoc_with_retention(&root, 0, 16, Some(retention.clone()));
        let mut references = CommandBufferReferences::default();
        references.track(&buffer);
        drop(buffer);
        drop(retention);

        assert!(weak_retention.upgrade().is_some());
        drop(references);
        assert!(weak_retention.upgrade().is_none());
    }

    #[test]
    fn producer_failure_is_visible_to_every_dependent() {
        let dev = MetalDevice::new(0).unwrap();
        let producer = dev.new_submission_context();
        *producer.failure.lock().unwrap() = Some("producer failed".to_string());
        let buffer = dev.alloc_raw(16);
        buffer.set_producer(&producer);

        for _ in 0..2 {
            assert_eq!(
                buffer.synchronize_producer(None).unwrap_err(),
                "producer failed"
            );
        }
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
        assert_eq!(upload.usage.tracked_size, Some(4));
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

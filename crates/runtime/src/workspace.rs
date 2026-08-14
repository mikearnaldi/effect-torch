//! Pooled recycling of transient workspace allocations.
//!
//! Executing a compiled program needs scratch buffers whose sizes are known
//! at compile time but whose contents are dead by the end of the call.
//! [`WorkspacePool`] recycles those buffers: callers
//! [`acquire`](WorkspacePool::acquire) a [`WorkspaceLease`] covering a set
//! of keyed size requests, use the segments, and simply drop the lease —
//! the segments return to the pool and become available for reuse.
//!
//! # Semantics
//!
//! - **Keys.** Each request carries a key `K` (e.g. a device or memory
//!   space); idle segments are only reused for requests with an equal key,
//!   so incompatible storage never migrates between keys.
//! - **Best-fit reuse.** Among idle segments matching the key with
//!   sufficient capacity, the pool picks the smallest capacity, breaking
//!   ties by lowest id — deterministic and fragmentation-averse.
//! - **All-or-error multi-segment acquisition.**
//!   [`WorkspacePool::acquire`] publishes a lease only after every request
//!   succeeds. On failure, reused pending segments return to the idle list;
//!   fresh pending allocations are dropped. If the allocator's first attempt
//!   failed, existing idle segments may also have been evicted before the
//!   retry, so failure preserves accounting invariants but not the exact idle
//!   cache contents.
//! - **Eviction.** Idle memory is capped by `max_idle_bytes`; when the cap
//!   is exceeded the least-recently-used idle segments are dropped until
//!   the pool fits. On allocation failure the pool first evicts *all* idle
//!   segments and retries once before reporting the error.
//! - **Accounting invariants.** `idle_bytes` always equals the summed
//!   capacity of the idle list and the leased counters track live leases;
//!   every mutation uses checked arithmetic, with internal inconsistencies
//!   panicking (`expect`) and user-triggerable overflow reported as
//!   [`WorkspacePoolError::ByteSizeOverflow`] without mutating the pool.
//! - **LRU clock.** `last_used` ticks are monotonic per pool; when the
//!   `u64` tick counter would wrap, the idle list's ticks are compacted to
//!   a dense 0..n range preserving order.
//!
//! Thread safety comes from a single `Mutex` around the pool state. Poison is
//! ignored (`lock` recovers the guard) so a panic does not permanently wedge
//! the pool; subsequent checked accounting and validation still fail loudly
//! if state is inconsistent. Leases return their segments from `Drop`, so
//! scope exit — including unwinding — recycles memory.

use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

/// Backend hook that produces fresh workspace storage on demand.
///
/// Called only when no idle segment fits; the returned capacity must be at
/// least `minimum_bytes` (enforced by the pool).
pub trait WorkspaceAllocator<K> {
    type Workspace;
    type Error;

    fn allocate(
        &mut self,
        key: &K,
        minimum_bytes: usize,
    ) -> Result<WorkspaceAllocation<Self::Workspace>, Self::Error>;
}

/// A freshly allocated workspace plus its capacity in bytes.
#[derive(Debug)]
pub struct WorkspaceAllocation<T> {
    workspace: T,
    capacity: usize,
}

impl<T> WorkspaceAllocation<T> {
    pub fn new(workspace: T, capacity: usize) -> Self {
        Self {
            workspace,
            capacity,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn into_inner(self) -> T {
        self.workspace
    }
}

/// One keyed size request within an acquisition set.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkspaceRequest<K> {
    pub key: K,
    pub bytes: usize,
}

impl<K> WorkspaceRequest<K> {
    pub fn new(key: K, bytes: usize) -> Self {
        Self { key, bytes }
    }
}

/// Point-in-time pool counters (see the module documentation for the
/// invariants they satisfy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspacePoolStats {
    pub idle_segments: usize,
    pub idle_bytes: usize,
    pub leased_segments: usize,
    pub leased_bytes: usize,
}

/// A cloneable handle to a shared workspace pool.
///
/// All clones share one idle list and one set of counters. The pool outlives
/// its leases: a [`WorkspaceLease`] holds an `Arc` to the same state.
pub struct WorkspacePool<K, A>
where
    A: WorkspaceAllocator<K>,
{
    inner: Arc<Mutex<WorkspacePoolInner<K, A>>>,
}

impl<K, A> Clone for WorkspacePool<K, A>
where
    A: WorkspaceAllocator<K>,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<K, A> WorkspacePool<K, A>
where
    K: Clone + Eq,
    A: WorkspaceAllocator<K>,
{
    /// Creates a pool that retains at most `max_idle_bytes` of idle
    /// workspace, allocating misses through `allocator`.
    pub fn new(max_idle_bytes: usize, allocator: A) -> Self {
        Self {
            inner: Arc::new(Mutex::new(WorkspacePoolInner {
                allocator,
                max_idle_bytes,
                idle: Vec::new(),
                idle_bytes: 0,
                leased_segments: 0,
                leased_bytes: 0,
                clock: 0,
                next_id: 0,
            })),
        }
    }

    pub fn max_idle_bytes(&self) -> usize {
        lock(&self.inner).max_idle_bytes
    }

    /// Raises or lowers the idle-memory cap, evicting LRU segments
    /// immediately if the current idle set exceeds the new cap.
    pub fn set_max_idle_bytes(&self, max_idle_bytes: usize) {
        let mut inner = lock(&self.inner);
        inner.max_idle_bytes = max_idle_bytes;
        inner.evict_to_limit();
    }

    pub fn stats(&self) -> WorkspacePoolStats {
        let inner = lock(&self.inner);
        WorkspacePoolStats {
            idle_segments: inner.idle.len(),
            idle_bytes: inner.idle_bytes,
            leased_segments: inner.leased_segments,
            leased_bytes: inner.leased_bytes,
        }
    }

    /// Atomically acquires one segment per request.
    ///
    /// Reuses the best-fitting idle segment with a matching key when
    /// possible, otherwise allocates through the allocator (evicting all
    /// idle segments and retrying once on failure). On error, reused pending
    /// segments are rolled back and fresh pending allocations are dropped.
    /// Idle entries evicted for the retry are not restored.
    pub fn acquire(
        &self,
        requests: &[WorkspaceRequest<K>],
    ) -> Result<WorkspaceLease<K, A>, WorkspacePoolError<A::Error>> {
        let requested_bytes = requests.iter().try_fold(0usize, |total, request| {
            total
                .checked_add(request.bytes)
                .ok_or(WorkspacePoolError::ByteSizeOverflow)
        })?;
        let mut inner = lock(&self.inner);
        let mut pending = Vec::with_capacity(requests.len());

        for request in requests {
            if let Some(index) = inner.best_fit(&request.key, request.bytes) {
                let entry = inner.idle.remove(index);
                inner.idle_bytes = inner
                    .idle_bytes
                    .checked_sub(entry.capacity)
                    .expect("workspace pool idle byte accounting underflow");
                pending.push(PendingEntry {
                    entry,
                    requested_bytes: request.bytes,
                    reused: true,
                });
                continue;
            }

            let allocation = match inner.allocator.allocate(&request.key, request.bytes) {
                Ok(allocation) => allocation,
                Err(error) if inner.idle.is_empty() => {
                    inner.rollback(&mut pending);
                    return Err(WorkspacePoolError::Allocation(error));
                }
                Err(_) => {
                    inner.evict_all_idle();
                    match inner.allocator.allocate(&request.key, request.bytes) {
                        Ok(allocation) => allocation,
                        Err(error) => {
                            inner.rollback(&mut pending);
                            return Err(WorkspacePoolError::Allocation(error));
                        }
                    }
                }
            };
            if allocation.capacity < request.bytes {
                let capacity = allocation.capacity;
                drop(allocation);
                inner.rollback(&mut pending);
                return Err(WorkspacePoolError::InsufficientCapacity {
                    requested: request.bytes,
                    actual: capacity,
                });
            }
            pending.push(PendingEntry {
                entry: IdleEntry {
                    id: 0,
                    key: request.key.clone(),
                    capacity: allocation.capacity,
                    workspace: allocation.workspace,
                    last_used: 0,
                },
                requested_bytes: request.bytes,
                reused: false,
            });
        }

        let actual_bytes = match pending.iter().try_fold(0usize, |total, entry| {
            total.checked_add(entry.entry.capacity)
        }) {
            Some(actual_bytes) => actual_bytes,
            None => {
                inner.rollback(&mut pending);
                return Err(WorkspacePoolError::ByteSizeOverflow);
            }
        };
        let leased_bytes = match inner.leased_bytes.checked_add(actual_bytes) {
            Some(leased_bytes) => leased_bytes,
            None => {
                inner.rollback(&mut pending);
                return Err(WorkspacePoolError::ByteSizeOverflow);
            }
        };
        let leased_segments = match inner.leased_segments.checked_add(pending.len()) {
            Some(leased_segments) => leased_segments,
            None => {
                inner.rollback(&mut pending);
                return Err(WorkspacePoolError::ByteSizeOverflow);
            }
        };
        let new_count = pending.iter().filter(|entry| !entry.reused).count();
        let new_count = match u64::try_from(new_count) {
            Ok(new_count) => new_count,
            Err(_) => {
                inner.rollback(&mut pending);
                return Err(WorkspacePoolError::IdentifierOverflow);
            }
        };
        let next_id = match inner.next_id.checked_add(new_count) {
            Some(next_id) => next_id,
            None => {
                inner.rollback(&mut pending);
                return Err(WorkspacePoolError::IdentifierOverflow);
            }
        };

        let mut id = inner.next_id;
        for entry in &mut pending {
            if !entry.reused {
                entry.entry.id = id;
                id = id
                    .checked_add(1)
                    .expect("workspace identifier was checked before assignment");
            }
        }
        inner.next_id = next_id;
        inner.leased_bytes = leased_bytes;
        inner.leased_segments = leased_segments;

        Ok(WorkspaceLease {
            pool: Arc::clone(&self.inner),
            segments: Some(
                pending
                    .into_iter()
                    .map(|entry| LeasedWorkspace {
                        entry: entry.entry,
                        requested_bytes: entry.requested_bytes,
                    })
                    .collect(),
            ),
            requested_bytes,
            actual_bytes,
        })
    }

    /// Alias for [`acquire`](Self::acquire); exists to emphasize the
    /// all-or-nothing semantics at call sites acquiring multiple segments.
    pub fn acquire_set(
        &self,
        requests: &[WorkspaceRequest<K>],
    ) -> Result<WorkspaceLease<K, A>, WorkspacePoolError<A::Error>> {
        self.acquire(requests)
    }
}

impl<K, A> fmt::Debug for WorkspacePool<K, A>
where
    K: Clone + Eq,
    A: WorkspaceAllocator<K>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkspacePool")
            .field("max_idle_bytes", &self.max_idle_bytes())
            .field("stats", &self.stats())
            .finish()
    }
}

/// RAII guard owning the segments of one acquisition.
///
/// Dropping the lease returns every segment to the pool (stamping it with
/// a fresh LRU tick) and triggers eviction down to the idle cap. The
/// segments are inaccessible after the lease is dropped.
pub struct WorkspaceLease<K, A>
where
    A: WorkspaceAllocator<K>,
{
    pool: Arc<Mutex<WorkspacePoolInner<K, A>>>,
    segments: Option<Vec<LeasedWorkspace<K, A::Workspace>>>,
    requested_bytes: usize,
    actual_bytes: usize,
}

impl<K, A> WorkspaceLease<K, A>
where
    A: WorkspaceAllocator<K>,
{
    /// The acquired segments, in request order.
    pub fn segments(&self) -> &[LeasedWorkspace<K, A::Workspace>] {
        self.segments.as_deref().unwrap_or_default()
    }

    /// Mutable access to the acquired segments, in request order.
    pub fn segments_mut(&mut self) -> &mut [LeasedWorkspace<K, A::Workspace>] {
        self.segments.as_deref_mut().unwrap_or_default()
    }

    /// Total bytes requested.
    pub fn requested_bytes(&self) -> usize {
        self.requested_bytes
    }

    /// Total capacity actually acquired (`>= requested_bytes`).
    pub fn actual_bytes(&self) -> usize {
        self.actual_bytes
    }
}

impl<K, A> fmt::Debug for WorkspaceLease<K, A>
where
    K: fmt::Debug,
    A: WorkspaceAllocator<K>,
    A::Workspace: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkspaceLease")
            .field("segments", &self.segments)
            .field("requested_bytes", &self.requested_bytes)
            .field("actual_bytes", &self.actual_bytes)
            .finish()
    }
}

impl<K, A> Drop for WorkspaceLease<K, A>
where
    A: WorkspaceAllocator<K>,
{
    fn drop(&mut self) {
        let Some(segments) = self.segments.take() else {
            return;
        };
        let mut inner = lock(&self.pool);
        inner.leased_bytes = inner
            .leased_bytes
            .checked_sub(self.actual_bytes)
            .expect("workspace pool leased byte accounting underflow");
        inner.leased_segments = inner
            .leased_segments
            .checked_sub(segments.len())
            .expect("workspace pool leased segment accounting underflow");
        for segment in segments {
            let last_used = inner.next_lru_tick();
            let mut entry = segment.entry;
            entry.last_used = last_used;
            inner.idle_bytes = inner
                .idle_bytes
                .checked_add(entry.capacity)
                .expect("workspace pool idle byte accounting overflow");
            inner.idle.push(entry);
        }
        inner.evict_to_limit();
    }
}

/// One segment of a [`WorkspaceLease`]: the workspace plus its key,
/// requested size and actual capacity.
pub struct LeasedWorkspace<K, T> {
    entry: IdleEntry<K, T>,
    requested_bytes: usize,
}

impl<K, T> LeasedWorkspace<K, T> {
    /// The key this segment was requested (and is reused) under.
    pub fn key(&self) -> &K {
        &self.entry.key
    }

    /// Bytes requested for this segment.
    pub fn requested_bytes(&self) -> usize {
        self.requested_bytes
    }

    /// Actual capacity of the segment (`>= requested_bytes`).
    pub fn capacity(&self) -> usize {
        self.entry.capacity
    }

    pub fn workspace(&self) -> &T {
        &self.entry.workspace
    }

    pub fn workspace_mut(&mut self) -> &mut T {
        &mut self.entry.workspace
    }
}

impl<K: fmt::Debug, T: fmt::Debug> fmt::Debug for LeasedWorkspace<K, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LeasedWorkspace")
            .field("key", &self.entry.key)
            .field("requested_bytes", &self.requested_bytes)
            .field("capacity", &self.entry.capacity)
            .field("workspace", &self.entry.workspace)
            .finish()
    }
}

/// Why an acquisition failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkspacePoolError<E> {
    /// The allocator failed (after one eviction-and-retry attempt, if any
    /// idle segments existed).
    Allocation(E),
    /// The allocator returned a segment smaller than the request.
    InsufficientCapacity { requested: usize, actual: usize },
    /// Byte accounting overflowed `usize`.
    ByteSizeOverflow,
    /// The pool's segment id counter (`u64`) was exhausted.
    IdentifierOverflow,
}

impl<E: fmt::Display> fmt::Display for WorkspacePoolError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkspacePoolError::Allocation(error) => {
                write!(f, "workspace allocation failed: {error}")
            }
            WorkspacePoolError::InsufficientCapacity { requested, actual } => write!(
                f,
                "workspace allocator returned {actual} bytes for a {requested}-byte request"
            ),
            WorkspacePoolError::ByteSizeOverflow => f.write_str("workspace byte size overflow"),
            WorkspacePoolError::IdentifierOverflow => f.write_str("workspace identifier overflow"),
        }
    }
}

impl<E: Error + 'static> Error for WorkspacePoolError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            WorkspacePoolError::Allocation(error) => Some(error),
            _ => None,
        }
    }
}

struct WorkspacePoolInner<K, A>
where
    A: WorkspaceAllocator<K>,
{
    allocator: A,
    max_idle_bytes: usize,
    idle: Vec<IdleEntry<K, A::Workspace>>,
    idle_bytes: usize,
    leased_segments: usize,
    leased_bytes: usize,
    clock: u64,
    next_id: u64,
}

impl<K, A> WorkspacePoolInner<K, A>
where
    A: WorkspaceAllocator<K>,
{
    // Smallest sufficient capacity, ties broken by lowest id so repeated
    // acquire/drop cycles are deterministic.
    fn best_fit(&self, key: &K, bytes: usize) -> Option<usize>
    where
        K: Eq,
    {
        self.idle
            .iter()
            .enumerate()
            .filter(|(_, entry)| &entry.key == key && entry.capacity >= bytes)
            .min_by_key(|(_, entry)| (entry.capacity, entry.id))
            .map(|(index, _)| index)
    }

    // Returns reused segments to the idle list after a failed acquisition.
    // Freshly allocated segments are dropped instead (they were never idle,
    // and their `id`/`last_used` placeholders were never published).
    fn rollback(&mut self, pending: &mut Vec<PendingEntry<K, A::Workspace>>) {
        for entry in pending.drain(..) {
            if entry.reused {
                self.idle_bytes = self
                    .idle_bytes
                    .checked_add(entry.entry.capacity)
                    .expect("workspace pool rollback byte accounting overflow");
                self.idle.push(entry.entry);
            }
        }
    }

    fn evict_to_limit(&mut self) {
        while self.idle_bytes > self.max_idle_bytes {
            let Some((index, _)) = self
                .idle
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| (entry.last_used, entry.id))
            else {
                break;
            };
            let entry = self.idle.remove(index);
            self.idle_bytes = self
                .idle_bytes
                .checked_sub(entry.capacity)
                .expect("workspace pool eviction byte accounting underflow");
        }
    }

    fn evict_all_idle(&mut self) {
        self.idle.clear();
        self.idle_bytes = 0;
    }

    // Monotonic LRU tick; on u64 wrap, idle ticks are compacted to a dense
    // 0..n range preserving the eviction order, then ticking resumes.
    fn next_lru_tick(&mut self) -> u64 {
        if let Some(next) = self.clock.checked_add(1) {
            self.clock = next;
            return next;
        }

        let mut order: Vec<_> = (0..self.idle.len()).collect();
        order.sort_by_key(|&index| (self.idle[index].last_used, self.idle[index].id));
        for (rank, index) in order.into_iter().enumerate() {
            self.idle[index].last_used =
                u64::try_from(rank).expect("workspace count exceeded the identifier range");
        }
        self.clock = u64::try_from(self.idle.len().saturating_sub(1))
            .expect("workspace count exceeded the identifier range");
        self.clock = self
            .clock
            .checked_add(1)
            .expect("workspace LRU counter exhausted after compaction");
        self.clock
    }
}

struct IdleEntry<K, T> {
    id: u64,
    key: K,
    capacity: usize,
    workspace: T,
    last_used: u64,
}

struct PendingEntry<K, T> {
    entry: IdleEntry<K, T>,
    requested_bytes: usize,
    reused: bool,
}

// Recovers the guard from a poisoned mutex so one panic does not wedge every
// future acquisition. Checked accounting continues to detect inconsistent
// state rather than silently accepting it.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug)]
    struct TestWorkspace {
        id: usize,
    }

    struct TestAllocator {
        calls: Arc<AtomicUsize>,
        fail_at: Option<usize>,
    }

    impl WorkspaceAllocator<&'static str> for TestAllocator {
        type Workspace = TestWorkspace;
        type Error = &'static str;

        fn allocate(
            &mut self,
            _key: &&'static str,
            minimum_bytes: usize,
        ) -> Result<WorkspaceAllocation<Self::Workspace>, Self::Error> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail_at == Some(call) {
                return Err("injected failure");
            }
            Ok(WorkspaceAllocation::new(
                TestWorkspace { id: call },
                minimum_bytes,
            ))
        }
    }

    fn pool(
        max_idle_bytes: usize,
        fail_at: Option<usize>,
    ) -> (WorkspacePool<&'static str, TestAllocator>, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (
            WorkspacePool::new(
                max_idle_bytes,
                TestAllocator {
                    calls: Arc::clone(&calls),
                    fail_at,
                },
            ),
            calls,
        )
    }

    #[test]
    fn acquisition_uses_deterministic_best_fit() {
        let (pool, calls) = pool(1024, None);
        let small = pool
            .acquire(&[WorkspaceRequest::new("shared", 64)])
            .unwrap();
        let large = pool
            .acquire(&[WorkspaceRequest::new("shared", 128)])
            .unwrap();
        let small_id = small.segments()[0].workspace().id;
        let large_id = large.segments()[0].workspace().id;
        drop(small);
        drop(large);

        let fit = pool
            .acquire(&[WorkspaceRequest::new("shared", 60)])
            .unwrap();
        assert_eq!(fit.segments()[0].capacity(), 64);
        assert_eq!(fit.segments()[0].workspace().id, small_id);
        assert_ne!(fit.segments()[0].workspace().id, large_id);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn set_acquisition_rolls_back_reused_segments_atomically() {
        let (pool, calls) = pool(1024, Some(1));
        let cached = pool
            .acquire(&[WorkspaceRequest::new("cached", 64)])
            .unwrap();
        let cached_id = cached.segments()[0].workspace().id;
        drop(cached);
        let before = pool.stats();

        let error = pool.acquire(&[
            WorkspaceRequest::new("cached", 32),
            WorkspaceRequest::new("fails", 16),
        ]);
        assert!(matches!(
            error,
            Err(WorkspacePoolError::Allocation("injected failure"))
        ));
        assert_eq!(pool.stats(), before);

        let reused = pool
            .acquire(&[WorkspaceRequest::new("cached", 64)])
            .unwrap();
        assert_eq!(reused.segments()[0].workspace().id, cached_id);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn lru_pressure_evicts_the_oldest_idle_segment() {
        let (pool, calls) = pool(80, None);
        drop(pool.acquire(&[WorkspaceRequest::new("old", 40)]).unwrap());
        drop(pool.acquire(&[WorkspaceRequest::new("hot", 40)]).unwrap());

        let hot = pool.acquire(&[WorkspaceRequest::new("hot", 40)]).unwrap();
        drop(hot);
        drop(pool.acquire(&[WorkspaceRequest::new("new", 40)]).unwrap());
        assert_eq!(pool.stats().idle_bytes, 80);

        let calls_before = calls.load(Ordering::SeqCst);
        drop(pool.acquire(&[WorkspaceRequest::new("hot", 40)]).unwrap());
        assert_eq!(calls.load(Ordering::SeqCst), calls_before);
        drop(pool.acquire(&[WorkspaceRequest::new("old", 40)]).unwrap());
        assert_eq!(calls.load(Ordering::SeqCst), calls_before + 1);
        assert!(pool.stats().idle_bytes <= 80);
    }

    #[derive(Debug)]
    struct BudgetWorkspace {
        bytes: usize,
        live: Arc<AtomicUsize>,
    }

    impl Drop for BudgetWorkspace {
        fn drop(&mut self) {
            self.live.fetch_sub(self.bytes, Ordering::SeqCst);
        }
    }

    struct BudgetAllocator {
        limit: usize,
        live: Arc<AtomicUsize>,
    }

    impl WorkspaceAllocator<&'static str> for BudgetAllocator {
        type Workspace = BudgetWorkspace;
        type Error = &'static str;

        fn allocate(
            &mut self,
            _key: &&'static str,
            minimum_bytes: usize,
        ) -> Result<WorkspaceAllocation<Self::Workspace>, Self::Error> {
            let live = self.live.load(Ordering::SeqCst);
            if live.saturating_add(minimum_bytes) > self.limit {
                return Err("budget exhausted");
            }
            self.live.fetch_add(minimum_bytes, Ordering::SeqCst);
            Ok(WorkspaceAllocation::new(
                BudgetWorkspace {
                    bytes: minimum_bytes,
                    live: Arc::clone(&self.live),
                },
                minimum_bytes,
            ))
        }
    }

    #[test]
    fn allocation_pressure_evicts_incompatible_idle_storage_and_retries() {
        let live = Arc::new(AtomicUsize::new(0));
        let pool = WorkspacePool::new(
            1024,
            BudgetAllocator {
                limit: 64,
                live: Arc::clone(&live),
            },
        );
        drop(pool.acquire(&[WorkspaceRequest::new("cold", 64)]).unwrap());
        assert_eq!(pool.stats().idle_bytes, 64);
        assert_eq!(live.load(Ordering::SeqCst), 64);

        let replacement = pool
            .acquire(&[WorkspaceRequest::new("replacement", 64)])
            .unwrap();
        assert_eq!(pool.stats().idle_bytes, 0);
        assert_eq!(pool.stats().leased_bytes, 64);
        assert_eq!(live.load(Ordering::SeqCst), 64);
        drop(replacement);
    }

    struct InfallibleAllocator;

    impl WorkspaceAllocator<()> for InfallibleAllocator {
        type Workspace = ();
        type Error = Infallible;

        fn allocate(
            &mut self,
            _key: &(),
            minimum_bytes: usize,
        ) -> Result<WorkspaceAllocation<Self::Workspace>, Self::Error> {
            Ok(WorkspaceAllocation::new((), minimum_bytes))
        }
    }

    #[test]
    fn request_byte_overflow_does_not_mutate_the_pool() {
        let pool = WorkspacePool::new(usize::MAX, InfallibleAllocator);
        let result = pool.acquire(&[
            WorkspaceRequest::new((), usize::MAX),
            WorkspaceRequest::new((), 1),
        ]);
        assert!(matches!(result, Err(WorkspacePoolError::ByteSizeOverflow)));
        assert_eq!(
            pool.stats(),
            WorkspacePoolStats {
                idle_segments: 0,
                idle_bytes: 0,
                leased_segments: 0,
                leased_bytes: 0,
            }
        );
    }
}

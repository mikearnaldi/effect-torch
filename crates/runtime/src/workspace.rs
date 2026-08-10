use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

pub trait WorkspaceAllocator<K> {
    type Workspace;
    type Error;

    fn allocate(
        &mut self,
        key: &K,
        minimum_bytes: usize,
    ) -> Result<WorkspaceAllocation<Self::Workspace>, Self::Error>;
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspacePoolStats {
    pub idle_segments: usize,
    pub idle_bytes: usize,
    pub leased_segments: usize,
    pub leased_bytes: usize,
}

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
    pub fn segments(&self) -> &[LeasedWorkspace<K, A::Workspace>] {
        self.segments.as_deref().unwrap_or_default()
    }

    pub fn segments_mut(&mut self) -> &mut [LeasedWorkspace<K, A::Workspace>] {
        self.segments.as_deref_mut().unwrap_or_default()
    }

    pub fn requested_bytes(&self) -> usize {
        self.requested_bytes
    }

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

pub struct LeasedWorkspace<K, T> {
    entry: IdleEntry<K, T>,
    requested_bytes: usize,
}

impl<K, T> LeasedWorkspace<K, T> {
    pub fn key(&self) -> &K {
        &self.entry.key
    }

    pub fn requested_bytes(&self) -> usize {
        self.requested_bytes
    }

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkspacePoolError<E> {
    Allocation(E),
    InsufficientCapacity { requested: usize, actual: usize },
    ByteSizeOverflow,
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

//! Arena allocation for frozen programs (RFC 0016 phase 1).
//!
//! A compiled program replays the exact same allocation sequence on
//! every run: the graph is frozen, the walk order is deterministic, and
//! every intermediate's size is fixed. The first run captures that
//! sequence — every `MetalDevice::alloc` with its size, plus which
//! buffers the evaluator still references at each node boundary — and
//! an offline planner packs the recorded live intervals into one arena
//! buffer. Subsequent runs replay: `alloc` call i returns a
//! suballocation at the planned offset, skipping the pool entirely.
//! Buffers that escape the walk (program outputs, bindings) are planned
//! as pool slots, so callers own them exactly as before.
//!
//! Reuse safety: every dispatch in a command buffer is followed by a
//! memory barrier and command buffers on the serial queue execute in
//! submission order, so a write into a dead buffer's region is always
//! GPU-ordered after the last read of its previous tenant.
//!
//! Replay is fail-safe: if the allocation sequence diverges from the
//! plan (size mismatch or overrun), the run falls back to the pool for
//! the remaining allocations and the program's arena is disabled.

use super::device::Buffer;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

const ALIGN: usize = 256;

fn align(n: usize) -> usize {
    (n + ALIGN - 1) & !(ALIGN - 1)
}

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

struct Event {
    clock: usize,
    size: usize,
    kind: Option<&'static str>,
}

#[derive(Default)]
struct Capture {
    clock: usize,
    checkpoints: usize,
    max_open: usize,
    /// One per allocation, in call order.
    events: Vec<Event>,
    /// Buffer ptr identity -> birth clock, for allocations still open.
    open: HashMap<usize, usize>,
    /// Birth clock -> death clock for closed intervals.
    deaths: HashMap<usize, usize>,
}

/// The captured allocation sequence of one program run.
pub struct CaptureResult {
    /// Sizes in allocation-call order.
    pub sizes: Vec<usize>,
    /// Per allocation: Some((birth, death)) for arena candidates, None
    /// for escapes (still referenced when the walk ended).
    pub intervals: Vec<Option<(usize, usize)>>,
    /// Node kind that performed each allocation (debug).
    pub kinds: Vec<Option<&'static str>>,
    /// Buffer ptrs still referenced at walk end (debug).
    pub live_at_end: usize,
    /// Checkpoints observed (debug).
    pub checkpoints: usize,
    pub max_open: usize,
}

thread_local! {
    static CAPTURE: RefCell<Option<Capture>> = const { RefCell::new(None) };
    static CURRENT_KIND: RefCell<Option<&'static str>> = const { RefCell::new(None) };
}

pub fn set_current_kind(kind: &'static str) {
    CURRENT_KIND.with(|k| *k.borrow_mut() = Some(kind));
}

pub fn capture_begin() {
    CAPTURE.with(|c| *c.borrow_mut() = Some(Capture::default()));
}

pub fn capture_active() -> bool {
    CAPTURE.with(|c| c.borrow().is_some())
}

/// Records one allocation. A reused buffer identity (the pool handed
/// back a dead buffer) closes the previous interval at this point.
pub fn capture_alloc(ptr: usize, size: usize) {
    CAPTURE.with(|c| {
        let mut guard = c.borrow_mut();
        let Some(cap) = guard.as_mut() else { return };
        cap.clock += 1;
        if let Some(birth) = cap.open.remove(&ptr) {
            cap.deaths.insert(birth, cap.clock);
        }
        cap.open.insert(ptr, cap.clock);
        cap.max_open = cap.max_open.max(cap.open.len());
        let kind = CURRENT_KIND.with(|k| *k.borrow());
        cap.events.push(Event { clock: cap.clock, size, kind });
    });
}

/// Node-boundary checkpoint: `live` is the set of buffer ptrs the
/// evaluator still references. Any open allocation absent from `live`
/// died at this point in the schedule.
pub fn capture_checkpoint(live: &HashSet<usize>) {
    CAPTURE.with(|c| {
        let mut guard = c.borrow_mut();
        let Some(cap) = guard.as_mut() else { return };
        cap.clock += 1;
        cap.checkpoints += 1;
        let open = std::mem::take(&mut cap.open);
        for (ptr, birth) in open {
            if live.contains(&ptr) {
                cap.open.insert(ptr, birth);
            } else {
                cap.deaths.insert(birth, cap.clock);
            }
        }
    });
}

/// Ends capture. `live` is the set of buffer ptrs still referenced at
/// the end of the walk; allocations still open are escapes.
pub fn capture_end(live: &HashSet<usize>) -> CaptureResult {
    capture_checkpoint(live);
    CAPTURE.with(|c| {
        let cap = c.borrow_mut().take().expect("capture_end without capture_begin");
        let escapes: HashSet<usize> = cap.open.values().copied().collect();
        let sizes = cap.events.iter().map(|e| e.size).collect();
        let intervals = cap
            .events
            .iter()
            .map(|e| {
                if escapes.contains(&e.clock) {
                    None
                } else {
                    Some((e.clock, cap.deaths.get(&e.clock).copied().unwrap_or(cap.clock)))
                }
            })
            .collect();
        CaptureResult { sizes, intervals, kinds: cap.events.iter().map(|e| e.kind).collect(), live_at_end: live.len(), checkpoints: cap.checkpoints, max_open: cap.max_open }
    })
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

pub enum PlanEntry {
    /// Suballocate at this byte offset in the arena.
    Arena { offset: usize, size: usize },
    /// Allocate from the pool (the buffer escapes the walk).
    Pool { size: usize },
}

pub struct Plan {
    pub entries: Vec<PlanEntry>,
    /// Bytes of arena this plan needs.
    pub total: usize,
    /// The backing buffer. Program runs are serialized by the eval
    /// guard, so plans share one buffer (grown to the largest plan);
    /// the static holds only a Weak reference, so the arena is released
    /// when the last program using it is dropped.
    pub arena: Arc<Buffer>,
}

/// The shared arena's Weak handle, grown (never shrunk) as plans demand.
static SHARED_ARENA: std::sync::Mutex<Option<std::sync::Weak<Buffer>>> = std::sync::Mutex::new(None);

pub fn shared_arena_size() -> usize {
    SHARED_ARENA
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|w| w.upgrade())
        .map(|a| a.size)
        .unwrap_or(0)
}

/// Returns an arena of at least `total` bytes, sharing the live one
/// when it is big enough and replacing the Weak handle otherwise.
fn shared_arena(total: usize, make_arena: &dyn Fn(usize) -> Arc<Buffer>) -> Arc<Buffer> {
    let mut guard = SHARED_ARENA.lock().unwrap();
    let existing = guard.as_ref().and_then(|w| w.upgrade());
    match existing {
        Some(arena) if arena.size >= total => arena,
        _ => {
            let arena = make_arena(total);
            *guard = Some(Arc::downgrade(&arena));
            arena
        }
    }
}

/// Packs captured intervals with a best-fit free-list allocator over a
/// flat address space. Deterministic: intervals are placed in event
/// order, best fit breaks ties by lowest offset. Returns the placement
/// per allocation and the total arena size.
pub fn plan_layout(captured: &CaptureResult) -> (Vec<PlanEntry>, usize) {
    // Releases sorted by death time, processed as births advance.
    let mut releases: Vec<(usize, (usize, usize))> = Vec::new();
    let mut free: Vec<(usize, usize)> = Vec::new();
    let mut top = 0usize;

    let mut entries: Vec<PlanEntry> = Vec::with_capacity(captured.sizes.len());
    for (i, size) in captured.sizes.iter().enumerate() {
        let Some((birth, death)) = captured.intervals[i] else {
            entries.push(PlanEntry::Pool { size: *size });
            continue;
        };
        let aligned = align(*size).max(ALIGN);
        releases.sort_by_key(|(d, _)| *d);
        let mut r = 0;
        while r < releases.len() && releases[r].0 <= birth {
            free.push(releases[r].1);
            r += 1;
        }
        releases.drain(..r);
        free.sort_by_key(|(offset, _)| *offset);
        coalesce(&mut free);
        // Best fit: smallest free region that holds the allocation.
        let mut best: Option<usize> = None;
        for (k, (_, rsize)) in free.iter().enumerate() {
            if *rsize >= aligned && best.is_none_or(|b| free[b].1 > *rsize) {
                best = Some(k);
            }
        }
        let offset = match best {
            Some(k) => {
                let (roffset, rsize) = free.remove(k);
                if rsize > aligned {
                    free.push((roffset + aligned, rsize - aligned));
                }
                roffset
            }
            None => {
                let offset = top;
                top += aligned;
                offset
            }
        };
        releases.push((death.max(birth + 1), (offset, aligned)));
        entries.push(PlanEntry::Arena { offset, size: *size });
    }
    (entries, top.max(ALIGN))
}

pub fn plan(captured: &CaptureResult, make_arena: &dyn Fn(usize) -> Arc<Buffer>) -> Plan {
    let (entries, total) = plan_layout(captured);
    let arena = shared_arena(total, make_arena);
    Plan { entries, total, arena }
}

fn coalesce(free: &mut Vec<(usize, usize)>) {
    if free.len() < 2 {
        return;
    }
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(free.len());
    let (mut cur_offset, mut cur_size) = free[0];
    for &(offset, size) in &free[1..] {
        if cur_offset + cur_size == offset {
            cur_size += size;
        } else {
            out.push((cur_offset, cur_size));
            cur_offset = offset;
            cur_size = size;
        }
    }
    out.push((cur_offset, cur_size));
    *free = out;
}

/// Oracle summary: what the arena costs vs what the walk requested.
pub fn report(captured: &CaptureResult, arena_size: usize) -> String {
    let total: usize = captured.sizes.iter().sum();
    let escapes: usize = captured
        .sizes
        .iter()
        .zip(&captured.intervals)
        .filter(|(_, i)| i.is_none())
        .map(|(s, _)| s)
        .sum();
    let escape_count = captured.intervals.iter().filter(|i| i.is_none()).count();
    let mut by_kind: HashMap<&'static str, (usize, usize)> = HashMap::new();
    for (i, interval) in captured.intervals.iter().enumerate() {
        if interval.is_none() {
            let entry = by_kind.entry(captured.kinds[i].unwrap_or("?")).or_default();
            entry.0 += 1;
            entry.1 += captured.sizes[i];
        }
    }
    let mut kinds: Vec<(&str, (usize, usize))> = by_kind.into_iter().collect();
    kinds.sort_by_key(|(_, (_, bytes))| std::cmp::Reverse(*bytes));
    let breakdown = kinds
        .iter()
        .take(6)
        .map(|(kind, (n, bytes))| format!("{kind}:{}MB×{n}", bytes >> 20))
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "{} allocs, {} MB requested, {} escapes ({} MB: {}), arena {} MB, live_at_end {} max_open {} over {} checkpoints",
        captured.sizes.len(),
        total >> 20,
        escape_count,
        escapes >> 20,
        breakdown,
        arena_size >> 20,
        captured.live_at_end,
        captured.max_open,
        captured.checkpoints
    )
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

struct Replay {
    plan: Arc<Plan>,
    cursor: usize,
    poisoned: bool,
}

thread_local! {
    static REPLAY: RefCell<Option<Replay>> = const { RefCell::new(None) };
}

pub fn replay_begin(plan: Arc<Plan>) {
    REPLAY.with(|r| *r.borrow_mut() = Some(Replay { plan, cursor: 0, poisoned: false }));
}

/// Ends replay. Returns true if the replay diverged from the plan (the
/// caller should disable the program's arena permanently).
pub fn replay_end() -> bool {
    REPLAY.with(|r| {
        let mut guard = r.borrow_mut();
        match guard.take() {
            Some(replay) => replay.poisoned || replay.cursor != replay.plan.entries.len(),
            None => true,
        }
    })
}

pub fn replay_active() -> bool {
    REPLAY.with(|r| r.borrow().is_some())
}

/// RAII cleanup for program runs: if an error path skips the explicit
/// replay_end/capture_end, dropping the guard clears the thread-local
/// session so a stale cursor never leaks into later allocations on this
/// thread.
pub struct SessionGuard {
    _private: (),
}

impl SessionGuard {
    pub fn new() -> Self {
        SessionGuard { _private: () }
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        if replay_active() {
            replay_end();
        }
        CAPTURE.with(|c| {
            if c.borrow().is_some() {
                *c.borrow_mut() = None;
            }
        });
    }
}

/// Serves one allocation from the arena. Returns None to fall through
/// to the pool (pool-planned entries and divergence fallback). A size
/// mismatch poisons the replay: the rest of the run uses the pool.
pub fn replay_alloc(size: usize) -> Option<Arc<Buffer>> {
    REPLAY.with(|r| {
        let mut guard = r.borrow_mut();
        let replay = guard.as_mut()?;
        if replay.poisoned || replay.cursor >= replay.plan.entries.len() {
            return None;
        }
        let entry = &replay.plan.entries[replay.cursor];
        replay.cursor += 1;
        let planned = match entry {
            PlanEntry::Arena { size, .. } => *size,
            PlanEntry::Pool { size } => *size,
        };
        if planned != size {
            replay.poisoned = true;
            return None;
        }
        match entry {
            PlanEntry::Pool { .. } => None,
            PlanEntry::Arena { offset, .. } => {
                Some(Arc::new(Buffer::suballoc(&replay.plan.arena, *offset, size)))
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cr(sizes: Vec<usize>, intervals: Vec<Option<(usize, usize)>>) -> CaptureResult {
        CaptureResult { sizes, intervals, kinds: vec![], live_at_end: 0, checkpoints: 0, max_open: 0 }
    }

    fn offsets(entries: &[PlanEntry]) -> Vec<Option<usize>> {
        entries
            .iter()
            .map(|e| match e {
                PlanEntry::Arena { offset, .. } => Some(*offset),
                PlanEntry::Pool { .. } => None,
            })
            .collect()
    }

    #[test]
    fn dead_intervals_are_reused() {
        // A dies at 2, B is born at 3: B reuses A's region.
        let captured = cr(vec![100, 100], vec![Some((1, 2)), Some((3, 4))]);
        let (entries, total) = plan_layout(&captured);
        assert_eq!(offsets(&entries), vec![Some(0), Some(0)]);
        assert_eq!(total, 256);
    }

    #[test]
    fn overlapping_intervals_do_not_alias() {
        let captured = cr(vec![100, 100], vec![Some((1, 4)), Some((2, 3))]);
        let (entries, total) = plan_layout(&captured);
        let offsets = offsets(&entries);
        assert_eq!(offsets[0], Some(0));
        assert_eq!(offsets[1], Some(256));
        assert_eq!(total, 512);
    }

    #[test]
    fn escapes_go_to_the_pool() {
        let captured = cr(vec![100, 42], vec![Some((1, 2)), None]);
        let (entries, _) = plan_layout(&captured);
        assert!(matches!(entries[0], PlanEntry::Arena { .. }));
        assert!(matches!(entries[1], PlanEntry::Pool { size: 42 }));
    }

    #[test]
    fn best_fit_picks_the_smallest_region() {
        // A(1024) dies at 2; B(256) takes A's tail; C(200) born at 4
        // reuses the coalesced A+B region; D(1000) born at 4 while C is
        // still alive must extend the arena.
        let captured = cr(vec![1024, 256, 200, 1000], vec![Some((1, 2)), Some((2, 3)), Some((4, 5)), Some((4, 6))]);
        let (entries, total) = plan_layout(&captured);
        let offsets = offsets(&entries);
        assert_eq!(offsets[0], Some(0));
        assert_eq!(offsets[1], Some(0)); // A's region
        assert_eq!(offsets[2], Some(0)); // coalesced A+B region
        assert_eq!(offsets[3], Some(1024)); // C still alive: new region
        assert_eq!(total, 2048);
    }

    #[test]
    fn coalescing_merges_adjacent_regions() {
        // A and B die together; C needs both regions coalesced.
        let captured = cr(vec![256, 256, 500], vec![Some((1, 3)), Some((2, 3)), Some((4, 5))]);
        let (entries, total) = plan_layout(&captured);
        assert_eq!(offsets(&entries)[2], Some(0));
        assert_eq!(total, 512);
    }
}

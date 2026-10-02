//! Bounded graph cache for ordinary expert GEMM sequences only.
//!
//! This owns launch metadata, never tensor backing. Every replay requires an
//! identical geometry/address/workspace key and the caller's normal invocation
//! leases and expert-stream completion fence. Grouped cuBLAS metadata copies
//! must not enter this cache.
use super::Bf16GemmPlan;
use cudarc::driver::{sys, CudaEvent, CudaGraph, CudaStream};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Key {
    groups: Vec<(Bf16GemmPlan, u64, u64, u64)>,
    workspace: u64,
    partitions: Vec<usize>,
    worker_start: usize,
}
impl Key {
    pub(crate) fn new(groups: &[(Bf16GemmPlan, u64, u64, u64)], workspace: u64) -> Self {
        Self {
            groups: groups.to_vec(),
            workspace,
            partitions: Vec::new(),
            worker_start: 0,
        }
    }
    pub(crate) fn partitioned(
        partitions: &[Vec<(Bf16GemmPlan, u64, u64, u64)>],
        worker_start: usize,
        workspace: u64,
    ) -> Self {
        Self {
            groups: partitions.iter().flatten().copied().collect(),
            workspace,
            partitions: partitions.iter().map(Vec::len).collect(),
            worker_start,
        }
    }
}
struct Graph {
    graph: CudaGraph,
    _events: Vec<CudaEvent>,
}
// SAFETY: access is serialized by the owning cuBLAS handle and cache mutex;
// cudarc binds the graph's context before launching or destroying it.
unsafe impl Send for Graph {}

pub(crate) struct Cache {
    capacity: usize,
    kind: &'static str,
    clock: u64,
    seen: HashMap<Key, u64>,
    graphs: HashMap<Key, (u64, Graph)>,
    pub(crate) hits: u64,
    pub(crate) captures: u64,
    evictions: u64,
}
impl Default for Cache {
    fn default() -> Self {
        let capacity = std::env::var("EFFECT_TORCH_CUDA_EXPERT_GRAPH_CACHE_ENTRIES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(512)
            .clamp(1, 2048);
        Self {
            capacity,
            kind: "ordinaryExpertGraphCache",
            clock: 0,
            seen: HashMap::new(),
            graphs: HashMap::new(),
            hits: 0,
            captures: 0,
            evictions: 0,
        }
    }
}
struct CaptureGuard<'a> {
    stream: &'a Arc<CudaStream>,
    active: bool,
}
impl Drop for CaptureGuard<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY,
            );
        }
    }
}
impl Cache {
    pub(crate) fn branch() -> Self {
        let mut cache = Self::default();
        cache.kind = "ordinaryExpertBranchGraphCache";
        cache
    }
    fn recurring(&mut self, key: &Key) -> bool {
        if self.seen.remove(key).is_some() {
            return true;
        }
        if self.seen.len() == self.capacity {
            let oldest = self
                .seen
                .iter()
                .min_by_key(|(_, tick)| **tick)
                .unwrap()
                .0
                .clone();
            self.seen.remove(&oldest);
        }
        self.seen.insert(key.clone(), self.clock);
        false
    }
    /// The single-worker caller holds its cuBLAS handle lock for the complete
    /// capture/launch. Joined branches additionally rely on the executor's
    /// device-wide graph lock to exclude unrelated worker submissions.
    pub(crate) fn execute(
        &mut self,
        stream: &Arc<CudaStream>,
        key: Key,
        submit: impl FnOnce() -> Result<(), String>,
    ) -> Result<(), String> {
        self.execute_with_events(stream, key, |_, _| submit())
            .map(|_| ())
    }
    pub(crate) fn execute_with_events(
        &mut self,
        stream: &Arc<CudaStream>,
        key: Key,
        submit: impl FnOnce(bool, &mut Vec<CudaEvent>) -> Result<(), String>,
    ) -> Result<Vec<CudaEvent>, String> {
        self.clock = self.clock.wrapping_add(1);
        if let Some((last, graph)) = self.graphs.get_mut(&key) {
            graph.graph.launch().map_err(|e| e.to_string())?;
            *last = self.clock;
            self.hits += 1;
            return Ok(Vec::new());
        }
        if !self.recurring(&key) {
            let mut events = Vec::new();
            submit(false, &mut events)?;
            return Ok(events);
        }
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|e| format!("ordinary expert capture start failed: {e}"))?;
        let mut events = Vec::new();
        let mut guard = CaptureGuard {
            stream,
            active: true,
        };
        submit(true, &mut events)?;
        let result = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY,
        );
        guard.active = false;
        let graph = result
            .map_err(|e| format!("ordinary expert capture end failed: {e}"))?
            .ok_or("ordinary expert capture returned no graph")?;
        graph.launch().map_err(|e| e.to_string())?;
        if self.graphs.len() == self.capacity {
            let oldest = self
                .graphs
                .iter()
                .min_by_key(|(_, (tick, _))| *tick)
                .unwrap()
                .0
                .clone();
            self.graphs.remove(&oldest);
            self.evictions += 1;
        }
        self.graphs.insert(
            key,
            (
                self.clock,
                Graph {
                    graph,
                    _events: events,
                },
            ),
        );
        self.captures += 1;
        Ok(Vec::new())
    }
}
impl Drop for Cache {
    fn drop(&mut self) {
        if let Some(path) = std::env::var_os("EFFECT_TORCH_CUDA_EXPERT_GRAPH_STATS_PATH") {
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                use std::io::Write;
                let _ = writeln!(
                    file,
                    "{}",
                    serde_json::json!({"kind":self.kind,
                    "capacityPerHandle":self.capacity,"cached":self.graphs.len(),"seen":self.seen.len(),
                    "hits":self.hits,"captures":self.captures,"evictions":self.evictions})
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn key(m: usize) -> Key {
        Key::new(
            &[(
                Bf16GemmPlan {
                    m,
                    n: 1408,
                    k: 2816,
                    batch: 1,
                    stride_x: m * 2816,
                    stride_weight: 0,
                    stride_out: m * 1408,
                },
                100,
                200,
                300,
            )],
            400,
        )
    }
    #[test]
    fn sequence_key_tracks_all_geometry_addresses_and_workspace() {
        let original = key(17);
        let mut variants = Vec::new();
        for field in 0..11 {
            let mut changed = original.clone();
            let (plan, x, w, out) = &mut changed.groups[0];
            match field {
                0 => plan.m += 1,
                1 => plan.n += 1,
                2 => plan.k += 1,
                3 => plan.batch += 1,
                4 => plan.stride_x += 1,
                5 => plan.stride_weight += 1,
                6 => plan.stride_out += 1,
                7 => *x += 1,
                8 => *w += 1,
                9 => *out += 1,
                _ => changed.workspace += 1,
            }
            assert_ne!(original, changed);
            variants.push(changed);
        }
        assert_eq!(
            variants
                .into_iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            11
        );
    }
    #[test]
    fn joined_key_tracks_worker_partition_and_origin() {
        let group = key(17).groups[0];
        let first = Key::partitioned(&[vec![group, group], vec![group]], 1, 400);
        assert_ne!(
            first,
            Key::partitioned(&[vec![group], vec![group, group]], 1, 400)
        );
        assert_ne!(
            first,
            Key::partitioned(&[vec![group, group], vec![group]], 2, 400)
        );
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn failed_submission_ends_capture_before_recovery() {
        let device = crate::CudaDevice::get(0).unwrap();
        let mut cache = Cache::default();
        cache.execute(&device.stream, key(17), || Ok(())).unwrap();
        let error = cache.execute(&device.stream, key(17), || {
            Err("injected submission failure".into())
        });
        assert_eq!(error.unwrap_err(), "injected submission failure");
        device
            .stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .unwrap();
        device
            .stream
            .end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY,
            )
            .unwrap();
        device.stream.synchronize().unwrap();
    }

    #[test]
    fn first_observation_is_eager_and_admission_is_bounded() {
        let mut cache = Cache::default();
        cache.capacity = 2;
        assert!(!cache.recurring(&key(17)));
        cache.clock += 1;
        assert!(!cache.recurring(&key(18)));
        cache.clock += 1;
        assert!(!cache.recurring(&key(19)));
        assert_eq!(cache.seen.len(), 2);
        assert!(cache.recurring(&key(18)));
        assert!(!cache.recurring(&key(17)));
    }
}

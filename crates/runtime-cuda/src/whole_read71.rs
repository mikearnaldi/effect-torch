//! Private, fixed-address storage for readonly decoder replay.
//!
//! Payload preparation is separate from submission: aliased state scratch is
//! refilled at its original command, never when the host prepares the table.
use super::*;

#[cfg(test)]
thread_local! {
    static CANCEL_AFTER_LAUNCH: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
impl CudaExecutable {
    /// Scope a one-shot cancellation to a successful graph71 launch on this
    /// thread. Unlike the generic kernel hooks, this does not disable admission.
    pub(crate) fn with_whole_read71_cancel_after_launch<T>(run: impl FnOnce() -> T) -> T {
        struct Restore(bool);
        impl Drop for Restore {
            fn drop(&mut self) {
                CANCEL_AFTER_LAUNCH.with(|pending| pending.set(self.0));
            }
        }
        let _restore = Restore(CANCEL_AFTER_LAUNCH.with(|pending| pending.replace(true)));
        run()
    }
}

#[cfg(test)]
pub(super) fn after_launch(cancelled: &CancellationFlag) {
    if CANCEL_AFTER_LAUNCH.with(|pending| pending.replace(false)) {
        cancelled.cancel();
    }
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;

    #[test]
    fn cancellation_is_one_shot_and_scoped_to_a_successful_launch() {
        let first = CancellationFlag::new();
        let second = CancellationFlag::new();
        CudaExecutable::with_whole_read71_cancel_after_launch(|| {
            assert!(!first.is_cancelled());
            after_launch(&first);
            after_launch(&second);
            assert!(first.is_cancelled());
            assert!(!second.is_cancelled());
        });
        // A scope that never launched must not cancel a later unrelated call.
        CudaExecutable::with_whole_read71_cancel_after_launch(|| ());
        after_launch(&second);
        assert!(!second.is_cancelled());
    }
}

pub(super) fn trace(
    event: &str,
    plan: Option<&whole_read71_admission::BodyPlan>,
    layers: usize,
) -> Result<(), String> {
    let Some(path) = std::env::var_os("EFFECT_TORCH_CUDA_WHOLE_READ71_TRACE_PATH") else {
        return Ok(());
    };
    let record = serde_json::json!({
        "event": event, "layers": layers,
        "start": plan.map(|plan| plan.start), "end": plan.map(|plan| plan.end),
        "hiddenInput": plan.map(|plan| plan.hidden_input.index()),
        "hiddenOutput": plan.map(|plan| plan.hidden_output.index()),
        "persistentResources": plan.map(|plan| plan.persistent.len()),
    });
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("whole-read71 trace open: {e}"))?;
    std::io::Write::write_all(&mut file, format!("{record}\n").as_bytes())
        .map_err(|e| format!("whole-read71 trace write: {e}"))
}

#[derive(Default)]
pub(super) struct Runtime {
    pub(super) frame: Option<Frame>,
    pub(super) plan: Option<Arc<whole_read71_admission::BodyPlan>>,
    pub(super) examined: bool,
}

pub(super) struct Frame {
    pub(super) graph: Option<Graph>,
    pub(super) storage: Option<Box<Storage>>,
    pub(super) geometry: Vec<u64>,
    pub(super) policy: Vec<(std::ffi::OsString, std::ffi::OsString)>,
    pub(super) warmed: bool,
    pub(super) quarantined: bool,
}

pub(super) struct Storage {
    pub(super) resources: InvocationResources,
    pub(super) live_ins: HashMap<ValueId, CudaValue>,
    pub(super) slab: CudaBuffer<u8>,
    pub(super) primary_blas: crate::cublas::CudaBlas,
    pub(super) dense_blas: crate::cublas::CudaBlas,
    pub(super) device: Arc<CudaDevice>,
    _constants: Vec<CudaValue>,
    _metadata: Vec<Option<CudaBuffer<u64>>>,
    _fused_functions: Vec<cudarc::driver::CudaFunction>,
    // Graph nodes reference native runner resources and packed gate/up weights
    // outside the ordinary value table. Keep their owners until graph teardown.
    _moe75_plans: Vec<Arc<crate::fused_moe75::Plan>>,
    prepared: Option<PreparedCache>,
}

struct PrefixLayerIdentity {
    descriptor: KvLayerDescriptor,
    start: u32,
    pages: Vec<Arc<CudaKvPage>>,
}

/// Retaining the Arc owners makes identity comparisons immune to allocator
/// address reuse. Device addresses alone are not a sufficient cache key.
struct PrefixIdentity(Vec<Vec<PrefixLayerIdentity>>);

fn same_owners<'a, T: 'a>(
    retained: &[Arc<T>],
    mut current: impl Iterator<Item = &'a Arc<T>>,
) -> bool {
    retained.iter().all(|owner| {
        current
            .next()
            .is_some_and(|other| Arc::ptr_eq(owner, other))
    }) && current.next().is_none()
}

impl PrefixIdentity {
    fn capture(state: &CudaStateInvocation) -> Result<Self, String> {
        let cache = state
            .cache
            .as_ref()
            .ok_or("whole-read71 prefix cache missing")?;
        if cache.sequences.len() != state.sequences.len() {
            return Err("whole-read71 prefix sequence count differs".into());
        }
        Ok(Self(
            cache
                .sequences
                .iter()
                .zip(&state.sequences)
                .map(|(snapshot, sequence)| {
                    snapshot
                        .layers
                        .iter()
                        .map(|layer| PrefixLayerIdentity {
                            descriptor: layer.descriptor,
                            start: layer.start_position,
                            // prepare_kernel_state discards previous readonly transaction
                            // pages starting at cursor, then inserts this frame's pages.
                            pages: layer
                                .pages
                                .iter()
                                .filter(|page| page.start < sequence.cursor)
                                .cloned()
                                .collect(),
                        })
                        .collect()
                })
                .collect(),
        ))
    }

    fn matches(&self, state: &CudaStateInvocation) -> bool {
        let Some(cache) = state.cache.as_ref() else {
            return false;
        };
        self.0.len() == cache.sequences.len()
            && self.0.len() == state.sequences.len()
            && self
                .0
                .iter()
                .zip(&cache.sequences)
                .zip(&state.sequences)
                .all(|((expected, actual), sequence)| {
                    expected.len() == actual.layers.len()
                        && expected
                            .iter()
                            .zip(&actual.layers)
                            .all(|(expected, actual)| {
                                expected.descriptor == actual.descriptor
                                    && expected.start == actual.start_position
                                    && same_owners(
                                        &expected.pages,
                                        actual
                                            .pages
                                            .iter()
                                            .filter(|page| page.start < sequence.cursor),
                                    )
                            })
                })
    }
}

struct PreparedCache {
    prefix: PrefixIdentity,
    commands: Arc<HashMap<usize, PreparedState>>,
    /// Invocation-local bookkeeping only. These current-row pages refer to
    /// fixed private frame storage; readonly success never publishes them.
    transactions: CudaKvCache,
}

#[cfg(test)]
mod prefix_identity_tests {
    use super::*;

    #[test]
    fn owners_require_identity_order_and_count_not_device_address_or_contents() {
        // Two owners can describe the same numerical device address after reuse,
        // or identical payloads at different allocations. Neither is a cache hit.
        let owner = Arc::new(0x1000_u64);
        let same_address_different_owner = Arc::new(0x1000_u64);
        let other = Arc::new(0x2000_u64);
        let retained = vec![owner.clone(), other.clone()];
        assert!(same_owners(
            &retained,
            [owner.clone(), other.clone()].iter()
        ));
        assert!(!same_owners(
            &retained,
            [same_address_different_owner, other.clone()].iter()
        ));
        assert!(!same_owners(
            &retained,
            [other.clone(), owner.clone()].iter()
        ));
        assert!(!same_owners(&retained, [owner.clone()].iter()));
        assert!(!same_owners(
            &retained,
            [owner.clone(), other.clone(), other].iter()
        ));

        let weak = Arc::downgrade(&owner);
        drop(owner);
        assert!(
            weak.upgrade().is_some(),
            "cache must retain the compared owner"
        );
        drop(retained);
        assert!(
            weak.upgrade().is_none(),
            "replacing the bounded cache releases owners"
        );
    }

    fn empty_state() -> CudaStateInvocation {
        let descriptor = KvLayerDescriptor {
            layer_id: 0,
            kv_heads: 1,
            head_dim: 256,
            dtype: DType::BF16,
            retention: None,
        };
        CudaStateInvocation {
            sequences: vec![CudaSequenceState {
                cursor: 0,
                keys: vec![],
                values: vec![],
                kda_states: vec![],
                conv_states: vec![],
                kv_storage: None,
            }],
            slots: vec![0],
            valid_lengths: vec![256],
            capacity: 1024,
            cache_dtype: DType::BF16,
            packed_rows_per_sequence: None,
            kv_layers: vec![descriptor],
            access: StateAccessMode::ReadOnly,
            cache: Some(CudaKvCache {
                sequences: vec![CudaKvSnapshot {
                    layers: vec![CudaKvLayer {
                        descriptor,
                        start_position: 0,
                        pages: vec![],
                    }],
                }],
            }),
        }
    }

    #[test]
    fn empty_prefix_still_checks_layer_schema_retained_start_and_sequence_grouping() {
        let mut state = empty_state();
        let key = PrefixIdentity::capture(&state).unwrap();
        assert!(key.matches(&state));
        state.cache.as_mut().unwrap().sequences[0].layers[0].start_position = 1;
        assert!(!key.matches(&state));
        state.cache.as_mut().unwrap().sequences[0].layers[0].start_position = 0;
        state.cache.as_mut().unwrap().sequences[0].layers[0]
            .descriptor
            .head_dim = 512;
        assert!(!key.matches(&state));
        state.cache.as_mut().unwrap().sequences.clear();
        assert!(!key.matches(&state));
        assert!(PrefixIdentity::capture(&state).is_err());
    }

    #[test]
    fn outer_geometry_changes_for_cursor_valid_lengths_and_slot_mapping() {
        let mut state = empty_state();
        let initial = geometry(&state);
        state.sequences[0].cursor = 1;
        assert_ne!(initial, geometry(&state));
        state.sequences[0].cursor = 0;
        state.valid_lengths[0] = 128;
        assert_ne!(initial, geometry(&state));
        state.valid_lengths[0] = 256;
        state.slots[0] = 1;
        assert_ne!(initial, geometry(&state));
    }
}

pub(super) struct Graph {
    raw: sys::CUgraph,
    exec: sys::CUgraphExec,
}

// Access is serialized by the executable runtime mutex and device graph gate.
unsafe impl Send for Graph {}

impl Graph {
    pub(super) fn finish(stream: &CudaStream, quarantined: &mut bool) -> Result<Self, String> {
        let mut raw = std::ptr::null_mut();
        unsafe { sys::cuStreamEndCapture(stream.cu_stream(), &mut raw) }
            .result()
            .map_err(|e| format!("whole-read71 end capture: {e}"))?;
        if raw.is_null() {
            return Err("whole-read71 capture returned no graph".into());
        }
        let mut graph = Self {
            raw,
            exec: std::ptr::null_mut(),
        };
        if let Err(error) = Self::validate_nodes(raw) {
            if let Err(cleanup) = graph.destroy() {
                *quarantined = true;
                return Err(cleanup);
            }
            return Err(error);
        }
        if let Err(error) = unsafe {
            sys::cuGraphInstantiateWithFlags(
                &mut graph.exec,
                raw,
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY as u64,
            )
        }
        .result()
        {
            if let Err(cleanup) = graph.destroy() {
                *quarantined = true;
                return Err(cleanup);
            }
            return Err(format!("whole-read71 instantiate: {error}"));
        }
        Ok(graph)
    }

    fn validate_nodes(graph: sys::CUgraph) -> Result<(), String> {
        let mut count = 0;
        unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut count) }
            .result()
            .map_err(|e| e.to_string())?;
        let mut nodes = vec![std::ptr::null_mut(); count];
        unsafe { sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut count) }
            .result()
            .map_err(|e| e.to_string())?;
        for node in nodes.into_iter().take(count) {
            let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
            unsafe { sys::cuGraphNodeGetType(node, &mut kind) }
                .result()
                .map_err(|e| e.to_string())?;
            match kind {
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL
                | sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY
                | sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET
                | sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY => {}
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_GRAPH => {
                    let mut child = std::ptr::null_mut();
                    unsafe { sys::cuGraphChildGraphNodeGetGraph(node, &mut child) }
                        .result()
                        .map_err(|e| e.to_string())?;
                    Self::validate_nodes(child)?;
                }
                // Internal stream-capture record/waits become DAG edges. An
                // external event or host node would require additional owners.
                _ => return Err(format!("whole-read71 unsupported captured node {kind:?}")),
            }
        }
        Ok(())
    }

    pub(super) fn launch(&self, stream: &CudaStream) -> Result<(), String> {
        unsafe { sys::cuGraphLaunch(self.exec, stream.cu_stream()) }
            .result()
            .map_err(|e| format!("whole-read71 launch: {e}"))
    }

    fn destroy(&mut self) -> Result<(), String> {
        if !self.exec.is_null() {
            unsafe { sys::cuGraphExecDestroy(self.exec) }
                .result()
                .map_err(|e| format!("whole-read71 destroy exec: {e}"))?;
            self.exec = std::ptr::null_mut();
        }
        if !self.raw.is_null() {
            unsafe { sys::cuGraphDestroy(self.raw) }
                .result()
                .map_err(|e| format!("whole-read71 destroy graph: {e}"))?;
            self.raw = std::ptr::null_mut();
        }
        Ok(())
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        let Some(storage) = self.storage.take() else {
            return;
        };
        let result = storage
            .device
            .stream
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())
            .and_then(|()| self.graph.as_mut().map_or(Ok(()), Graph::destroy));
        if let Err(error) = result {
            eprintln!("{error}; quarantining whole-read71 storage");
            std::mem::forget(storage);
        }
    }
}

pub(super) struct PreparedState {
    pub(super) args: CudaKernelArgs,
    pub(super) ranges: Vec<(usize, usize, usize)>,
    pub(super) refills: Vec<(usize, CudaBuffer<u8>)>,
}

pub(super) fn copy(
    stream: &CudaStream,
    source: u64,
    target: u64,
    bytes: usize,
) -> Result<(), String> {
    if bytes == 0 {
        return Ok(());
    }
    unsafe { sys::cuMemcpyDtoDAsync_v2(target, source, bytes, stream.cu_stream()) }
        .result()
        .map_err(|e| format!("whole-read71 copy: {e}"))
}

pub(super) fn geometry(state: &CudaStateInvocation) -> Vec<u64> {
    let mut key = vec![
        state.capacity as u64,
        state.packed_rows_per_sequence.unwrap_or(1) as u64,
        dtype_code(state.cache_dtype) as u64,
        state.slots.len() as u64,
    ];
    key.extend(state.slots.iter().map(|&v| v as u64));
    key.extend(state.valid_lengths.iter().map(|&v| v as u64));
    key.extend(state.sequences.iter().map(|s| s.cursor as u64));
    if let Some(cache) = &state.cache {
        for sequence in &cache.sequences {
            for layer in &sequence.layers {
                key.extend([
                    layer.descriptor.layer_id as u64,
                    layer.start_position as u64,
                ]);
            }
        }
    }
    key
}

pub(super) fn policy() -> Vec<(std::ffi::OsString, std::ffi::OsString)> {
    let mut settings = std::env::vars_os()
        .filter(|(key, _)| key.to_string_lossy().starts_with("EFFECT_TORCH_CUDA_"))
        .collect::<Vec<_>>();
    settings.sort();
    settings
}

impl CudaExecutable {
    pub(super) fn make_frame71(
        &self,
        plan: &whole_read71_admission::BodyPlan,
        geometry: Vec<u64>,
    ) -> Result<Frame, String> {
        let resources = workspace::acquire_private_segments71(
            self.device.ordinal,
            &self.memory.segments,
            &plan.segments,
        )?;
        let mut live_ins = HashMap::new();
        for &id in &plan.live_ins {
            let value = match self.memory.locations[id.index()] {
                effect_torch_runtime::Location::Segment { .. } => {
                    self.planned_value(&resources, id)?
                }
                _ => CudaValue::from_planned_buffer(
                    self.device.clone(),
                    self.program.values[id.index()].spec(),
                    workspace::acquire_private71(
                        self.device.ordinal,
                        self.program.values[id.index()].decl.bytes,
                    )?,
                )?,
            };
            live_ins.insert(id, value);
        }
        Ok(Frame {
            graph: None,
            warmed: false,
            quarantined: false,
            geometry,
            policy: policy(),
            storage: Some(Box::new(Storage {
                resources,
                live_ins,
                slab: workspace::acquire_private71(self.device.ordinal, 0)?,
                primary_blas: self.device.cublas.fork_for_graph()?,
                dense_blas: self.device.dense_cublas.fork_for_graph()?,
                device: self.device.clone(),
                _constants: self
                    .commands
                    .iter()
                    .filter_map(|command| match &command.kind {
                        CommandKind::Value(value) => Some(value.clone()),
                        _ => None,
                    })
                    .collect(),
                _metadata: self.metadata.clone(),
                _moe75_plans: self.commands[plan.start..plan.end]
                    .iter()
                    .filter_map(|command| match &command.kind {
                        CommandKind::FusedMoe75 { plan, .. } => Some(Arc::clone(plan)),
                        _ => None,
                    })
                    .collect(),
                _fused_functions: self.commands[plan.start..plan.end]
                    .iter()
                    .filter_map(|command| match &command.kind {
                        CommandKind::FusedElementwise { function, .. } => Some(function.clone()),
                        _ => None,
                    })
                    .collect(),
                prepared: None,
            })),
        })
    }

    pub(super) fn prepare_state71(
        &self,
        plan: &whole_read71_admission::BodyPlan,
        frame: &mut Frame,
        state: &mut CudaStateInvocation,
        pinned: &mut Option<cudarc::driver::PinnedHostSlice<u8>>,
    ) -> Result<Arc<HashMap<usize, PreparedState>>, String> {
        let storage = frame
            .storage
            .as_mut()
            .ok_or("whole-read71 storage missing")?;
        if let Some(cached) = storage
            .prepared
            .as_ref()
            .filter(|cached| cached.prefix.matches(state))
        {
            // Exact invocation geometry was checked before entering this method.
            // Clone the tentative page-list containers so normal rollback remains
            // invocation-local, while retaining the same immutable prefix owners.
            state.cache = Some(cached.transactions.clone());
            trace("prepared-hit", Some(plan), state.kv_layers.len())?;
            return Ok(cached.commands.clone());
        }
        let prefix = PrefixIdentity::capture(state)?;
        let mut prepared = HashMap::new();
        let mut payload = Vec::new();
        for &position in &plan.state_commands {
            let CommandKind::Kernel {
                args,
                scratch,
                state: access,
                state_buffers,
                ..
            } = &self.commands[position].kind
            else {
                return Err("whole-read71 invalid state command".into());
            };
            let scratch = scratch
                .iter()
                .map(|id| id.map(|id| self.buffer(&storage.resources, id)).transpose())
                .collect::<Result<Vec<_>, _>>()?;
            let transactions = state_buffers
                .iter()
                .map(|id| id.map(|id| self.buffer(&storage.resources, id)).transpose())
                .collect::<Result<Vec<_>, _>>()?;
            let mut args = *args;
            for (slot, buffer) in scratch.iter().enumerate() {
                if let Some(buffer) = buffer {
                    args.scratch[slot] = buffer.address();
                }
            }
            let mut ranges = Vec::new();
            let mut uploads = Vec::new();
            self.prepare_kernel_state(
                access,
                &mut args,
                &scratch,
                &transactions,
                Some(state),
                &mut ranges,
                Some(&mut uploads),
            )?;
            let mut refills = Vec::new();
            for upload in uploads {
                let offset = payload
                    .len()
                    .checked_next_multiple_of(8)
                    .ok_or("whole-read71 payload overflow")?;
                payload.resize(offset, 0);
                payload.extend_from_slice(&upload.bytes);
                refills.push((offset, upload.target));
            }
            prepared.insert(
                position,
                PreparedState {
                    args,
                    ranges,
                    refills,
                },
            );
        }
        if storage.slab.len() == 0 {
            storage.slab = workspace::acquire_private71(self.device.ordinal, payload.len())?;
        } else if storage.slab.len() != payload.len() {
            return Err("whole-read71 exact geometry changed payload extent".into());
        }
        // Immutable per-invocation host storage remains owned by the caller even
        // if the upload fails after submission. No pageable upload enters capture.
        let mut host = unsafe {
            self.device
                .stream
                .context()
                .alloc_pinned::<u8>(payload.len())
        }
        .map_err(|e| e.to_string())?;
        host.as_mut_slice()
            .map_err(|e| e.to_string())?
            .copy_from_slice(&payload);
        *pinned = Some(host);
        self.device
            .stream
            .memcpy_htod(pinned.as_ref().unwrap(), &mut storage.slab)
            .map_err(|e| e.to_string())?;
        let commands = Arc::new(prepared);
        storage.prepared = Some(PreparedCache {
            prefix,
            commands: commands.clone(),
            transactions: state
                .cache
                .clone()
                .ok_or("whole-read71 prepared transactions missing")?,
        });
        trace("prepared-refresh", Some(plan), state.kv_layers.len())?;
        Ok(commands)
    }

    pub(super) fn preflight_bindings71(
        &self,
        plan: &whole_read71_admission::BodyPlan,
        bindings: &[CudaValue],
    ) -> Result<(), String> {
        for &(id, binding) in &plan.bindings {
            let value = bindings
                .get(binding)
                .ok_or("whole-read71 missing binding")?;
            if value.ordinal() != self.device.ordinal
                || value.spec() != self.program.values[id.index()].spec()
            {
                return Err("whole-read71 binding violates its value specification".into());
            }
        }
        Ok(())
    }

    pub(super) fn enter_body71(
        &self,
        plan: &whole_read71_admission::BodyPlan,
        frame: &Frame,
        ordinary: &InvocationResources,
        bindings: &[CudaValue],
        values: &mut [Option<CudaValue>],
    ) -> Result<Vec<(ValueId, Option<CudaValue>)>, String> {
        let storage = frame
            .storage
            .as_ref()
            .ok_or("whole-read71 storage missing")?;
        let mut saved = Vec::new();
        for &id in &plan.live_ins {
            if id == plan.status {
                let source = self.buffer(ordinary, id)?;
                let target = self.buffer(&storage.resources, id)?;
                copy(
                    &self.device.stream,
                    source.address(),
                    target.address(),
                    source.len(),
                )?;
                continue;
            }
            let source =
                if let Some((_, binding)) = plan.bindings.iter().find(|(root, _)| *root == id) {
                    bindings
                        .get(*binding)
                        .ok_or("whole-read71 missing binding")?
                } else {
                    values[id.index()]
                        .as_ref()
                        .ok_or("whole-read71 missing live-in")?
                };
            let target = storage
                .live_ins
                .get(&id)
                .ok_or("whole-read71 missing staging")?;
            if source.ordinal() != self.device.ordinal || source.spec() != target.spec() {
                return Err("whole-read71 binding specification changed".into());
            }
            copy(
                &self.device.stream,
                source.storage_address(),
                target.storage_address(),
                source.storage_bytes(),
            )?;
            saved.push((id, values[id.index()].replace(target.clone())));
        }
        for &(id, position) in &plan.persistent_values {
            if values[id.index()].is_none() {
                let CommandKind::Value(value) = &self.commands[position].kind else {
                    return Err("whole-read71 persistent value unavailable".into());
                };
                values[id.index()] = Some(value.clone());
            }
        }
        for &id in &plan.live_in_aliases {
            let effect_torch_runtime::Location::Alias { root, byte_offset } =
                self.memory.locations[id.index()]
            else {
                return Err("whole-read71 expected normalized live-in alias".into());
            };
            let target = storage
                .live_ins
                .get(&root)
                .ok_or("whole-read71 missing alias staging")?;
            let bytes = self.program.values[id.index()].decl.bytes;
            let value = CudaValue::from_planned_buffer(
                self.device.clone(),
                self.program.values[id.index()].spec(),
                target.buffer.slice(
                    byte_offset
                        ..byte_offset
                            .checked_add(bytes)
                            .ok_or("whole-read71 alias overflow")?,
                )?,
            )?;
            saved.push((id, values[id.index()].replace(value)));
        }
        Ok(saved)
    }

    pub(super) fn leave_body71(
        &self,
        plan: &whole_read71_admission::BodyPlan,
        frame: &Frame,
        ordinary: &InvocationResources,
        values: &mut [Option<CudaValue>],
        saved: Vec<(ValueId, Option<CudaValue>)>,
    ) -> Result<(), String> {
        let storage = frame
            .storage
            .as_ref()
            .ok_or("whole-read71 storage missing")?;
        for (id, value) in saved {
            values[id.index()] = value;
        }
        for &id in &plan.live_outs {
            let source = self.buffer(&storage.resources, id)?;
            let target = self.planned_value(ordinary, id)?;
            copy(
                &self.device.stream,
                source.address(),
                target.storage_address(),
                source.len(),
            )?;
            values[id.index()] = Some(target);
        }
        for &id in &plan.live_out_aliases {
            values[id.index()] = Some(self.planned_value(ordinary, id)?);
        }
        Ok(())
    }
}

pub(super) struct StateUpload {
    pub(super) target: CudaBuffer<u8>,
    pub(super) bytes: Vec<u8>,
}

impl StateUpload {
    fn new(target: &Option<CudaBuffer<u8>>, bytes: Vec<u8>) -> Result<Self, String> {
        let target = target
            .as_ref()
            .ok_or("whole-read71 staging missing")?
            .slice(0..bytes.len())?;
        Ok(Self { target, bytes })
    }

    pub(super) fn u32(target: &Option<CudaBuffer<u8>>, values: &[u32]) -> Result<Self, String> {
        Self::new(
            target,
            values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        )
    }

    pub(super) fn u64(target: &Option<CudaBuffer<u8>>, values: &[u64]) -> Result<Self, String> {
        Self::new(
            target,
            values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        )
    }
}

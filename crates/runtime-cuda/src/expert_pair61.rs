//! Default-off explicit kernel graph for the proved device59 expert pair.
use super::*;
use crate::explicit_graph61::{ExplicitGraph61, KernelNodeSpec61, NodeArgs61};

pub(crate) fn enabled() -> bool {
    std::env::var("EFFECT_TORCH_CUDA_EXPERT_PAIR_GRAPH61").as_deref() == Ok("1")
}
#[derive(Clone, Debug)]
struct Group {
    position: usize,
    output: ValueId,
    x: ValueId,
    weight: ValueId,
    indexes: ValueId,
    rows: usize,
    columns: usize,
    inner: usize,
    source_rows: usize,
    control: ValueId,
    row_map: ValueId,
    gathered: ValueId,
    projected: ValueId,
    metadata: ValueId,
    status: ValueId,
    reuse: bool,
    input_sorted: bool,
    output_sorted: bool,
}
impl Group {
    fn read(position: usize, command: &Command) -> Option<Self> {
        let CommandKind::GroupedExpert {
            x,
            weight,
            indexes,
            rows,
            columns,
            inner,
            experts: 128,
            control,
            row_map,
            gathered,
            projected,
            reuse_routing,
            source_rows,
            input_sorted,
            output_sorted,
            inverse_routing: false,
            device_control: Some((metadata, status)),
            ..
        } = command.kind
        else {
            return None;
        };
        Some(Self {
            position,
            output: command.output?,
            x,
            weight,
            indexes,
            rows,
            columns,
            inner,
            source_rows,
            control,
            row_map,
            gathered,
            projected,
            metadata,
            status,
            reuse: reuse_routing,
            input_sorted,
            output_sorted,
        })
    }
}
#[derive(Clone, Debug)]
struct Pair {
    first: Group,
    second: Group,
    activation: usize,
}
struct Entry {
    key: Vec<u8>,
    graph: Arc<ExplicitGraph61>,
}
#[derive(Default)]
pub(super) struct Runtime {
    pairs: HashMap<usize, Pair>,
    cache: Mutex<HashMap<usize, Entry>>,
}
impl Runtime {
    pub(super) fn plan(executable: &CudaExecutable) -> Result<Self, String> {
        let mut runtime = Self::default();
        if !enabled() {
            return Ok(runtime);
        }
        let instructions = executable
            .program
            .instructions
            .iter()
            .filter(|i| {
                !matches!(
                    i.kind,
                    "state_prepare" | "state_commit" | "state_discard" | "status_check"
                )
            })
            .collect::<Vec<_>>();
        if instructions.len() != executable.commands.len() {
            return Err("graph61 command mapping mismatch".into());
        }
        let aliases = effect_torch_compiler::normalize_aliases(&executable.program.values)
            .map_err(|e| e.to_string())?;
        let uses = |lo: usize, hi: usize| {
            let mut reads = HashSet::new();
            let mut writes = HashSet::new();
            for i in &instructions[lo..=hi] {
                for u in i.resource_uses() {
                    let root = aliases[u.value.index()].root;
                    if u.access.reads() {
                        reads.insert(root);
                    }
                    if u.access.writes() {
                        writes.insert(root);
                    }
                }
            }
            (reads, writes)
        };
        for (lo, command) in executable.commands.iter().enumerate() {
            let Some(first) = Group::read(lo, command) else {
                continue;
            };
            if first.reuse
                || first.input_sorted
                || !first.output_sorted
                || !matches!(first.rows, 512 | 2048)
                || (first.columns, first.inner) != (1408, 2816)
            {
                continue;
            }
            let seconds = executable
                .commands
                .iter()
                .enumerate()
                .skip(lo + 1)
                .filter_map(|(p, c)| Group::read(p, c))
                .filter(|g| g.control == first.control)
                .collect::<Vec<_>>();
            if seconds.len() != 1 {
                continue;
            }
            let second = seconds[0].clone();
            let hi = second.position;
            if !second.reuse
                || !second.input_sorted
                || second.output_sorted
                || second.rows != first.rows
                || (second.columns, second.inner) != (2816, 704)
                || first.status != second.status
                || first.indexes != second.indexes
                || first.row_map != second.row_map
            {
                continue;
            }
            if [
                first.x,
                first.weight,
                first.output,
                second.weight,
                second.output,
            ]
            .iter()
            .any(|id| executable.program.values[id.index()].dtype != DType::BF16)
            {
                continue;
            }
            let mut activation = None;
            let mut valid = true;
            for p in lo..=hi {
                if executable.commands[p].overlap != CommandOverlap::Primary {
                    valid = false;
                }
                if p == lo || p == hi {
                    continue;
                }
                match &executable.commands[p].kind {
                    CommandKind::Prepare => {}
                    CommandKind::FusedElementwise {
                        wide_sum: false,
                        wide_arg: false,
                        graph61_function: Some(_),
                        inputs,
                        ..
                    } if instructions[p].kind == "group_sorted_activation"
                        && inputs[..2] == [Some(first.output); 2]
                        && inputs[2..].iter().all(Option::is_none)
                        && activation.is_none() =>
                    {
                        activation = Some(p);
                    }
                    _ => valid = false,
                }
            }
            let Some(activation) = activation else {
                continue;
            };
            let Some(activation_output) = executable.commands[activation].output else {
                continue;
            };
            if second.x != activation_output {
                continue;
            }
            let private = [
                aliases[first.output.index()].root,
                aliases[activation_output.index()].root,
            ];
            if executable
                .program
                .outputs
                .iter()
                .any(|v| private.contains(&aliases[v.index()].root))
            {
                valid = false;
            }
            for (p, i) in instructions.iter().enumerate() {
                if p < lo || p > hi {
                    if i.resource_uses()
                        .any(|u| private.contains(&aliases[u.value.index()].root))
                    {
                        valid = false;
                    }
                }
            }
            let (reads, writes) = uses(lo, hi);
            let mut branches = HashMap::<usize, (Vec<usize>, Vec<usize>)>::new();
            for (p, c) in executable.commands.iter().enumerate() {
                match c.overlap {
                    CommandOverlap::Worker { branch, .. } => {
                        branches.entry(branch).or_default().0.push(p)
                    }
                    CommandOverlap::Join { branch } => {
                        branches.entry(branch).or_default().1.push(p)
                    }
                    _ => {}
                }
            }
            for (_, (workers, joins)) in branches {
                if workers.is_empty() || joins.len() != 1 {
                    valid = false;
                    continue;
                }
                let begin = workers[0];
                let end = *workers.last().unwrap();
                let join = joins[0];
                let starts = workers
                    .iter()
                    .filter(|&&p| {
                        matches!(
                            executable.commands[p].overlap,
                            CommandOverlap::Worker { start: true, .. }
                        )
                    })
                    .copied()
                    .collect::<Vec<_>>();
                let finishes = workers
                    .iter()
                    .filter(|&&p| {
                        matches!(
                            executable.commands[p].overlap,
                            CommandOverlap::Worker { finish: true, .. }
                        )
                    })
                    .copied()
                    .collect::<Vec<_>>();
                if !dense_boundary_valid(begin, end, &starts, &finishes, &joins) {
                    valid = false;
                    continue;
                }
                if join < lo || begin > hi {
                    continue;
                }
                if end >= lo || join <= hi {
                    valid = false;
                }
                for p in workers {
                    let (r, w) = uses(p, p);
                    if !writes.is_disjoint(&r) || !writes.is_disjoint(&w) || !reads.is_disjoint(&w)
                    {
                        valid = false;
                    }
                }
            }
            if valid {
                runtime.pairs.insert(
                    lo,
                    Pair {
                        first,
                        second,
                        activation,
                    },
                );
            }
        }
        Ok(runtime)
    }
}

pub(super) struct Prepared {
    pub(super) graph: Arc<ExplicitGraph61>,
    pub(super) end: usize,
    pub(super) control: ValueId,
    pub(super) outputs: Vec<(ValueId, CudaValue)>,
}
impl CudaExecutable {
    pub(super) fn prepare_pair61(
        &self,
        position: usize,
        resources: &InvocationResources,
        values: &[Option<CudaValue>],
        deferred_status: Option<ValueId>,
        rows_block: bool,
        vector_copy: bool,
    ) -> Result<Option<Prepared>, String> {
        let Some(pair) = self.pair61.pairs.get(&position) else {
            return Ok(None);
        };
        if !self.device.device_expert_ready() || deferred_status != Some(pair.first.status) {
            return Ok(None);
        }
        let Some((merged, merged_blocks, m1_blocks)) = self.device.graph61_workers() else {
            return Ok(None);
        };
        // Validate both future external operands before materializing pair outputs.
        for (id, required) in [
            (pair.first.x, pair.first.source_rows * pair.first.inner * 2),
            (pair.first.indexes, pair.first.rows * 4),
            (pair.first.weight, 128 * 1408 * 2816 * 2),
            (pair.second.weight, 128 * 2816 * 704 * 2),
        ] {
            let Some(value) = values[id.index()].as_ref() else {
                return Ok(None);
            };
            let meta = &self.program.values[id.index()];
            if value.ordinal() != self.device.ordinal
                || value.shape() != meta.shape
                || value.dtype() != meta.dtype
                || meta.decl.bytes < required
                || value.storage_bytes() < meta.decl.bytes
                || value.spec().storage.representation != meta.storage.representation
                || value.allocation().1 < value.storage_bytes()
                || value
                    .storage_address()
                    .checked_add(meta.decl.bytes as u64)
                    .is_none()
                || ([pair.first.weight, pair.second.weight].contains(&id)
                    && value.storage_address() % 16 != 0)
            {
                return Ok(None);
            }
        }
        // Pure planned range/alignment checks include every scratch and output
        // operand. Current lease capacities are checked before any buffer view.
        let activation_output = self.commands[pair.activation]
            .output
            .ok_or("graph61 missing activation output")?;
        let planned = [
            (pair.first.output, pair.first.rows * 1408 * 2),
            (activation_output, pair.first.rows * 704 * 2),
            (pair.second.output, pair.second.rows * 2816 * 2),
            (pair.first.control, 130 * 4),
            (pair.first.row_map, pair.first.rows * 4),
            (pair.first.gathered, pair.first.rows * 2816 * 2),
            (pair.first.metadata, crate::expert_device::BYTES),
            (pair.second.projected, pair.second.rows * 2816 * 2),
            (pair.second.metadata, crate::expert_device::BYTES),
            (pair.first.status, 8),
        ];
        for (id, required) in planned {
            if self.program.values[id.index()].decl.bytes < required {
                return Ok(None);
            }
            let (location, extra) = match &self.memory.locations[id.index()] {
                effect_torch_runtime::Location::Alias { root, byte_offset } => {
                    (&self.memory.locations[root.index()], *byte_offset)
                }
                location => (location, 0),
            };
            let effect_torch_runtime::Location::Segment {
                segment, offset, ..
            } = location
            else {
                return Ok(None);
            };
            let Some(offset) = offset.checked_add(extra) else {
                return Ok(None);
            };
            if self.memory.segments[segment.index()].alignment < 16
                || offset % 16 != 0
                || resources
                    .preflight_graph61(location, extra, self.program.values[id.index()].decl.bytes)
                    .is_err()
            {
                return Ok(None);
            }
        }
        if [pair.first.metadata, pair.second.metadata]
            .iter()
            .any(|id| self.program.values[id.index()].decl.bytes < crate::expert_device::BYTES)
        {
            return Ok(None);
        }
        // Committed preparation starts here. No fallback is allowed after this
        // point: unexpected materialization/alignment errors use the fence.
        let mut outputs = Vec::new();
        for p in [position, pair.activation, pair.second.position] {
            let id = self.commands[p].output.ok_or("graph61 missing output")?;
            outputs.push((id, self.planned_value(resources, id)?));
        }
        let mut key = Vec::new();
        let mut addr = |id: ValueId| -> Result<u64, String> {
            let (address, allocation) = if let Some((_, v)) = outputs.iter().find(|(i, _)| *i == id)
            {
                (v.storage_address(), v.allocation())
            } else if let Some(v) = values[id.index()].as_ref() {
                (v.storage_address(), v.allocation())
            } else {
                let b = self.buffer(resources, id)?;
                (b.address(), b.allocation())
            };
            key.extend_from_slice(&id.get().to_le_bytes());
            key.extend_from_slice(&address.to_le_bytes());
            key.extend_from_slice(&allocation.0.to_le_bytes());
            key.extend_from_slice(&allocation.1.to_le_bytes());
            Ok(address)
        };
        let mut nodes = Vec::new();
        let mut tail = Vec::new();
        for (index, g) in [&pair.first, &pair.second].into_iter().enumerate() {
            let control = addr(g.control)?;
            let row_map = addr(g.row_map)?;
            let status = addr(g.status)?;
            let input = addr(g.x)?;
            let weight = addr(g.weight)?;
            let indexes = addr(g.indexes)?;
            let output = addr(g.output)?;
            let gathered = if g.input_sorted {
                input
            } else {
                addr(g.gathered)?
            };
            let projected = if g.output_sorted {
                output
            } else {
                addr(g.projected)?
            };
            let metadata = addr(g.metadata)?;
            if metadata % 16 != 0 || weight % 16 != 0 || gathered % 16 != 0 || projected % 16 != 0 {
                return Err("graph61 prepared pointer alignment invariant".into());
            }
            let mut args = CudaKernelArgs {
                output: control,
                elements: 130,
                output_dtype: dtype_code(DType::U32),
                ..Default::default()
            };
            args.inputs[0] = indexes;
            args.integers[0] = 128;
            args.scratch[3] = status;
            if index == 0 {
                push_typed(self, &mut nodes, &mut tail, "et_fill", args)?;
                args.elements = g.rows as u64;
                push_typed(self, &mut nodes, &mut tail, "et_grouped_counts", args)?;
                args.elements = 1;
                push_typed(self, &mut nodes, &mut tail, "et_grouped_offsets", args)?;
            }
            let mut packet = CudaKernelArgs::default();
            packet.inputs[..4].copy_from_slice(&[control, gathered, weight, row_map]);
            packet.output = projected;
            packet.scratch = [metadata, metadata + 8192, metadata + 9728, status];
            packet.integers[..4].copy_from_slice(&[
                g.columns as u64,
                g.inner as u64,
                g.rows as u64,
                0,
            ]);
            packet.error_context =
                u32::try_from(g.position).map_err(|_| "graph61 context overflow")?;
            packet.elements = (g.rows * g.columns) as u64;
            push_typed(
                self,
                &mut nodes,
                &mut tail,
                "et_expert_device_metadata59",
                packet,
            )?;
            push_typed(
                self,
                &mut nodes,
                &mut tail,
                "et_expert_device_sanitize59",
                packet,
            )?;
            if index == 0 {
                args.elements = g.rows as u64;
                args.inputs[1] = control;
                args.output = row_map;
                args.integers[3] = u64::from(rows_block);
                push_typed(self, &mut nodes, &mut tail, "et_grouped_rows", args)?;
            }
            args.inputs[0] = input;
            args.inputs[1] = row_map;
            args.output = gathered;
            args.output_dtype = dtype_code(DType::BF16);
            args.elements = (g.rows * g.inner) as u64;
            args.integers[0] = g.inner as u64;
            args.integers[1] = g.source_rows as u64;
            args.integers[2] = u64::from(vector_copy);
            if index == 0 {
                push_typed(self, &mut nodes, &mut tail, "et_grouped_gather", args)?;
            }
            let merged_index = nodes.len();
            nodes.push(KernelNodeSpec61 {
                kernel: merged.clone(),
                config: LaunchConfig {
                    grid_dim: (merged_blocks, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 36880,
                },
                args: NodeArgs61::Merged {
                    descriptors: metadata,
                    shapes: metadata + 8192,
                    count: 128,
                    columns: g.columns as u32,
                    inner: g.inner as u32,
                },
                parents: tail.clone(),
            });
            let name = if index == 0 {
                "et_expert_device_first59"
            } else {
                "et_expert_device_second59"
            };
            nodes.push(KernelNodeSpec61 {
                kernel: self
                    .device
                    .graph61_kernel(name)
                    .ok_or("graph61 M1 kernel missing")?,
                config: LaunchConfig {
                    grid_dim: (m1_blocks, 1, 1),
                    block_dim: (if index == 0 { 16 } else { 32 }, 4, 1),
                    shared_mem_bytes: 0,
                },
                args: NodeArgs61::M1 {
                    descriptors: metadata,
                    m1: metadata + 9728,
                },
                parents: tail.clone(),
            });
            tail = vec![merged_index, merged_index + 1];
            if index == 0 {
                let CommandKind::FusedElementwise {
                    args,
                    inputs,
                    graph61_function: Some(kernel),
                    ..
                } = &self.commands[pair.activation].kind
                else {
                    return Err("graph61 activation missing".into());
                };
                let mut args = *args;
                args.output = addr(self.commands[pair.activation].output.unwrap())?;
                for (slot, input) in inputs.iter().enumerate() {
                    if let Some(input) = input {
                        args.inputs[slot] = addr(*input)?;
                    }
                }
                let p = nodes.len();
                nodes.push(KernelNodeSpec61 {
                    kernel: kernel.clone(),
                    config: LaunchConfig {
                        grid_dim: (args.elements.div_ceil(256).min(65535) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    args: NodeArgs61::Typed(args),
                    parents: tail,
                });
                tail = vec![p];
            } else {
                args.inputs[0] = projected;
                args.inputs[1] = row_map;
                args.output = output;
                args.elements = (g.rows * g.columns) as u64;
                args.integers[0] = g.columns as u64;
                push_typed(self, &mut nodes, &mut tail, "et_grouped_scatter", args)?;
            }
        }
        if nodes.len() != 15 || nodes.iter().map(|n| n.parents.len()).sum::<usize>() != 16 {
            return Err("graph61 unexpected DAG".into());
        }
        for n in &nodes {
            key.extend(n.key_bytes());
        }
        let mut cache = self
            .pair61
            .cache
            .lock()
            .map_err(|_| "graph61 cache lock poisoned")?;
        let graph = if let Some(entry) = cache.get(&position).filter(|entry| entry.key == key) {
            entry.graph.clone()
        } else {
            // SAFETY: registered symbols and enum variants above match the three
            // exact device ABIs; construction copies all initialized host arguments.
            let graph =
                unsafe { ExplicitGraph61::build(self.device.stream.context().clone(), &nodes) }?;
            cache.insert(
                position,
                Entry {
                    key,
                    graph: graph.clone(),
                },
            );
            graph
        };
        Ok(Some(Prepared {
            graph,
            end: pair.second.position,
            control: pair.first.control,
            outputs,
        }))
    }
}
fn push_typed(
    executable: &CudaExecutable,
    nodes: &mut Vec<KernelNodeSpec61>,
    tail: &mut Vec<usize>,
    name: &str,
    args: CudaKernelArgs,
) -> Result<(), String> {
    let mut name = name;
    let config = if name == "et_expert_device_metadata59" || name == "et_expert_device_sanitize59" {
        LaunchConfig {
            grid_dim: (
                if name == "et_expert_device_metadata59" {
                    1
                } else {
                    32
                },
                1,
                1,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        }
    } else if name == "et_grouped_gather" || name == "et_grouped_scatter" {
        let (actual, threads) = grouped_copy_launch(name, &args);
        name = actual;
        LaunchConfig {
            grid_dim: ((args.elements / args.integers[0]).min(65535) as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        }
    } else if name == "et_grouped_rows" && args.integers[3] == 1 {
        name = "et_grouped_rows_block";
        LaunchConfig {
            grid_dim: (args.integers[0].min(65535) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        let work = if name == "et_grouped_rows" {
            args.integers[0] * 32
        } else {
            args.elements
        };
        LaunchConfig {
            grid_dim: (work.div_ceil(256).clamp(1, 65535) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    };
    let index = nodes.len();
    nodes.push(KernelNodeSpec61 {
        kernel: executable
            .device
            .graph61_kernel(name)
            .ok_or_else(|| format!("graph61 missing {name}"))?,
        config,
        args: NodeArgs61::Typed(args),
        parents: tail.clone(),
    });
    *tail = vec![index];
    Ok(())
}

#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fault {
    BeforeLaunch,
    AfterLaunch,
    CancelAfterLaunch,
}
#[cfg(test)]
thread_local! { static FAULT: std::cell::Cell<Option<Fault>> = const { std::cell::Cell::new(None) }; }
#[cfg(test)]
pub(crate) fn with_fault<T>(fault: Fault, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<Fault>);
    impl Drop for Restore {
        fn drop(&mut self) {
            FAULT.with(|f| f.set(self.0));
        }
    }
    let _restore = Restore(FAULT.with(|f| f.replace(Some(fault))));
    run()
}
#[cfg(test)]
pub(super) fn before_launch() -> Result<(), String> {
    if FAULT.with(|f| f.get()) == Some(Fault::BeforeLaunch) {
        return Err("injected graph61 before launch".into());
    }
    Ok(())
}
#[cfg(test)]
pub(super) fn after_launch(cancelled: &CancellationFlag) -> Result<(), String> {
    match FAULT.with(|f| f.get()) {
        Some(Fault::AfterLaunch) => Err("injected graph61 after launch".into()),
        Some(Fault::CancelAfterLaunch) => {
            cancelled.cancel();
            Ok(())
        }
        _ => Ok(()),
    }
}
#[cfg(test)]
impl CudaExecutable {
    pub(crate) fn graph61_test_counts(&self) -> (usize, usize) {
        (
            self.pair61.pairs.len(),
            self.pair61.cache.lock().unwrap().len(),
        )
    }
}

fn dense_boundary_valid(
    first: usize,
    last: usize,
    starts: &[usize],
    finishes: &[usize],
    joins: &[usize],
) -> bool {
    first <= last && starts == [first] && finishes == [last] && joins.len() == 1 && joins[0] > last
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn graph61_dense_boundary_requires_one_exact_fork_finish_join() {
        assert!(dense_boundary_valid(83, 96, &[83], &[96], &[147]));
        assert!(dense_boundary_valid(4, 4, &[4], &[4], &[7]));
        for (first, last, starts, finishes, joins) in [
            (83, 96, vec![], vec![96], vec![147]),
            (83, 96, vec![83, 90], vec![96], vec![147]),
            (83, 96, vec![83], vec![90], vec![147]),
            (83, 96, vec![83], vec![96], vec![96]),
            (83, 96, vec![83], vec![96], vec![147, 150]),
            (96, 83, vec![96], vec![83], vec![147]),
        ] {
            assert!(!dense_boundary_valid(
                first, last, &starts, &finishes, &joins
            ));
        }
    }
    #[test]
    fn graph61_fault_scope_preserves_checkpoint_boundary_and_restores_hook() {
        let cancelled = CancellationFlag::new();
        assert!(before_launch().is_ok());
        with_fault(Fault::BeforeLaunch, || {
            assert!(before_launch().is_err());
            assert!(after_launch(&cancelled).is_ok());
        });
        with_fault(Fault::AfterLaunch, || {
            assert!(before_launch().is_ok());
            assert!(after_launch(&cancelled).is_err());
        });
        with_fault(Fault::CancelAfterLaunch, || {
            assert!(after_launch(&cancelled).is_ok());
        });
        assert!(cancelled.is_cancelled());
        assert!(before_launch().is_ok());
        assert!(after_launch(&CancellationFlag::new()).is_ok());
    }
}

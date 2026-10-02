//! Conservative whole-decoder boundary selection from the current lowered IR.
//! This file does not allocate, submit, retain an invocation, or use archived IDs.
use super::*;
use effect_torch_runtime::{Location, SegmentId};

#[derive(Debug)]
pub(super) struct BodyPlan {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) hidden_input: ValueId,
    pub(super) hidden_output: ValueId,
    /// Non-persistent root values copied into the private context before replay.
    pub(super) live_ins: Vec<ValueId>,
    /// Root values copied back before the ordinary suffix executes.
    pub(super) live_outs: Vec<ValueId>,
    pub(super) live_in_aliases: Vec<ValueId>,
    pub(super) live_out_aliases: Vec<ValueId>,
    pub(super) bindings: Vec<(ValueId, usize)>,
    pub(super) persistent: Vec<ValueId>,
    /// Kernel metadata also has persistent storage, but is not a CudaValue.
    /// Only these declarations need entries in the ordinary value table.
    pub(super) persistent_values: Vec<(ValueId, usize)>,
    pub(super) state_commands: Vec<usize>,
    pub(super) status: ValueId,
    pub(super) segments: Vec<SegmentId>,
}

fn native_primary(overlap: CommandOverlap) -> bool {
    matches!(overlap, CommandOverlap::Primary)
}

fn supported(command: &Command) -> bool {
    match &command.kind {
        CommandKind::Prepare
        | CommandKind::Value(_)
        | CommandKind::Input { .. }
        | CommandKind::Alias { .. }
        | CommandKind::PlannedAlias
        | CommandKind::Gemm { .. }
        | CommandKind::GemmPair { .. }
        | CommandKind::LinearBias { .. }
        | CommandKind::FusedElementwise { .. } => true,
        // Native MoE plans own a runner bound to the device's primary stream.
        // A worker or join command cannot substitute its execution stream.
        CommandKind::FusedMoe75 { .. } => native_primary(command.overlap),
        CommandKind::GroupedExpert {
            device_control: Some(_),
            inverse_routing: false,
            experts: 128,
            ..
        } => true,
        CommandKind::Kernel { name, state, .. } => {
            !name.starts_with("et_random")
                && matches!(
                    state,
                    StateAccess::None | StateAccess::Rotary | StateAccess::Kv { .. }
                )
        }
        _ => false,
    }
}

/// No fork, finish, or join may cross a graph boundary. Branch identifiers are
/// the compiler's original identifiers, not inferred from physical stream IDs.
fn closed_overlap(commands: &[Command]) -> bool {
    let mut open = HashMap::<usize, bool>::new();
    for command in commands {
        match command.overlap {
            CommandOverlap::Primary => {}
            CommandOverlap::Worker {
                branch,
                start,
                finish,
            } => {
                if start && open.insert(branch, false).is_some() {
                    return false;
                }
                let Some(done) = open.get_mut(&branch) else {
                    return false;
                };
                if *done {
                    return false;
                }
                *done = finish;
            }
            CommandOverlap::Join { branch } => {
                if open.remove(&branch) != Some(true) {
                    return false;
                }
            }
        }
    }
    open.is_empty()
}

fn sorted(values: HashSet<ValueId>) -> Vec<ValueId> {
    let mut values = values.into_iter().collect::<Vec<_>>();
    values.sort_by_key(|id| id.index());
    values
}

pub(super) fn plan(executable: &CudaExecutable) -> Result<Option<BodyPlan>, String> {
    let Some(layout) = executable.state_layout.as_ref() else {
        return Ok(None);
    };
    if layout.access != StateAccessMode::ReadOnly || layout.dtype != DType::BF16 {
        return Ok(None);
    }
    let commands = &executable.commands;
    let program = &executable.program;
    let aliases = effect_torch_compiler::normalize_aliases(&program.values)
        .map_err(|error| error.to_string())?;
    let root = |id: ValueId| aliases[id.index()].root;
    let instructions = program
        .instructions
        .iter()
        .filter(|instruction| {
            !matches!(
                instruction.kind,
                "state_prepare" | "state_commit" | "state_discard" | "status_check"
            )
        })
        .collect::<Vec<_>>();
    if instructions.len() != commands.len() {
        return Err("whole-read71 command/instruction mapping mismatch".into());
    }
    let mut kv = Vec::new();
    let mut entrances = Vec::new();
    let mut tails = Vec::new();
    for (position, command) in commands.iter().enumerate() {
        match &command.kind {
            CommandKind::Kernel {
                state: StateAccess::Kv { layer, .. },
                kv_matmul: Some(_),
                ..
            } => {
                kv.push((position, *layer));
            }
            CommandKind::Kernel {
                name: "et_attention_ffn_entrance_bf16",
                inputs,
                ..
            } => {
                let Some(hidden) = inputs[1] else {
                    return Ok(None);
                };
                entrances.push((position, root(hidden)));
            }
            CommandKind::Kernel {
                name: "et_ffn_tail_bf16",
                ..
            } => {
                let Some(output) = command.output else {
                    return Ok(None);
                };
                tails.push((position, root(output)));
            }
            _ => {}
        }
    }
    // Admit the complete declared decoder, never a profitable-looking subset.
    let layers = layout.kv_layers.len();
    if layers == 0 || kv.len() != layers || entrances.len() != layers || tails.len() != layers {
        return Ok(None);
    }
    let declared = layout
        .kv_layers
        .iter()
        .map(|layer| layer.layer_id as usize)
        .collect::<HashSet<_>>();
    if kv.iter().map(|(_, layer)| *layer).collect::<HashSet<_>>() != declared {
        return Ok(None);
    }
    for index in 0..layers {
        if !(kv[index].0 < entrances[index].0 && entrances[index].0 < tails[index].0)
            || (index > 0
                && (tails[index - 1].0 >= kv[index].0 || tails[index - 1].1 != entrances[index].1))
        {
            return Ok(None);
        }
    }
    let hidden_input = entrances[0].1;
    let hidden_output = tails[layers - 1].1;
    let hidden = &program.values[hidden_input.index()];
    if hidden.dtype != DType::BF16
        || hidden.shape.last() != Some(&2816)
        || hidden.storage.representation != StorageRepresentation::Dense
        || hidden.decl.bytes == 0
        || program.values[hidden_output.index()].dtype != DType::BF16
        || program.values[hidden_output.index()].decl.bytes != hidden.decl.bytes
    {
        return Ok(None);
    }
    let Some(producer) = commands
        .iter()
        .position(|command| command.output == Some(hidden_input))
    else {
        return Ok(None);
    };
    let start = producer + 1;
    let end = tails[layers - 1].0 + 1;
    if start >= kv[0].0
        || !commands[start..end].iter().all(supported)
        || !closed_overlap(&commands[start..end])
    {
        return Ok(None);
    }

    let mut statuses = HashSet::new();
    let mut state_commands = Vec::new();
    let mut fresh_transactions = HashMap::<ValueId, usize>::new();
    let mut expert_count = 0usize;
    let mut bindings = Vec::new();
    for (offset, command) in commands[start..end].iter().enumerate() {
        match &command.kind {
            CommandKind::Kernel {
                checked,
                status,
                state,
                state_buffers,
                kv_matmul,
                ..
            } => {
                if *checked {
                    let Some(status) = status else {
                        return Ok(None);
                    };
                    statuses.insert(root(*status));
                }
                if !matches!(state, StateAccess::None) {
                    // Descriptor refills use the primary stream at this exact
                    // command position. A previously forked worker cannot read
                    // a refill without an additional dependency.
                    if matches!(command.overlap, CommandOverlap::Worker { .. }) {
                        return Ok(None);
                    }
                    state_commands.push(start + offset);
                }
                if matches!(state, StateAccess::Kv { .. }) && kv_matmul.is_some() {
                    for &value in state_buffers.iter().flatten() {
                        let id = root(value);
                        let Some(Location::Segment { segment, .. }) =
                            executable.memory.locations.get(id.index())
                        else {
                            return Ok(None);
                        };
                        if executable.memory.segments[segment.index()].ownership
                            != effect_torch_runtime::SegmentOwnership::StateTransaction
                            || fresh_transactions.insert(id, start + offset).is_some()
                        {
                            return Ok(None);
                        }
                    }
                }
            }
            CommandKind::GroupedExpert {
                device_control: Some((_, status)),
                ..
            } => {
                statuses.insert(root(*status));
                expert_count += 1;
            }
            CommandKind::FusedMoe75 { status, .. } => {
                statuses.insert(root(*status));
                // One native command replaces both expert projections.
                expert_count += 2;
            }
            CommandKind::Input { binding } => {
                let Some(output) = command.output else {
                    return Ok(None);
                };
                bindings.push((root(output), *binding));
            }
            _ => {}
        }
    }
    if statuses.len() != 1 || expert_count != layers * 2 {
        return Ok(None);
    }
    let status = *statuses.iter().next().unwrap();
    // The eager prefix must already establish the status lifetime. Its contents
    // (including an earlier failure) are input to the body, never reset by it.
    if !instructions[..start].iter().any(|instruction| {
        instruction
            .resource_uses()
            .any(|usage| root(usage.value) == status && usage.access.writes())
    }) {
        return Ok(None);
    }

    let mut written = HashSet::new();
    let mut live_ins = HashSet::new();
    let mut used = HashSet::new();
    for (offset, instruction) in instructions[start..end].iter().enumerate() {
        // Read/write uses in the same instruction must observe the incoming
        // value, regardless of resource-category ordering in the IR.
        for usage in instruction.resource_uses() {
            let id = root(usage.value);
            used.insert(id);
            // KV state transaction storage contains this invocation's new rows.
            // The admitted stepwise sequence stores those rows before gathering
            // them; its conservative ReadWrite IR use is not a caller input.
            // No other command may read or overwrite that private transaction.
            if fresh_transactions
                .get(&id)
                .is_some_and(|position| *position != start + offset)
            {
                return Ok(None);
            }
            if usage.access.reads() && !written.contains(&id) {
                live_ins.insert(id);
            }
        }
        for usage in instruction
            .resource_uses()
            .filter(|usage| usage.access.writes())
        {
            let id = root(usage.value);
            // Binding/view commands do not initialize memory. Current lowering
            // declares no writes for them; decline an inconsistent future IR.
            if matches!(
                commands[start + offset].kind,
                CommandKind::Value(_)
                    | CommandKind::Input { .. }
                    | CommandKind::Alias { .. }
                    | CommandKind::PlannedAlias
            ) || aliases[usage.value.index()].byte_offset != 0
                || program.values[usage.value.index()].decl.bytes
                    != program.values[id.index()].decl.bytes
            {
                // Root-level initialization tracking cannot prove that a write
                // through a partial view initializes a different view's bytes.
                return Ok(None);
            }
            written.insert(id);
        }
    }
    live_ins.insert(hidden_input);
    live_ins.insert(status);
    live_ins.extend(bindings.iter().map(|(id, _)| *id));
    used.extend(live_ins.iter().copied());
    live_ins.retain(|id| !fresh_transactions.contains_key(id));
    let mut live_outs = HashSet::new();
    for instruction in &instructions[end..] {
        for usage in instruction
            .resource_uses()
            .filter(|usage| usage.access.reads())
        {
            let id = root(usage.value);
            if written.contains(&id) {
                live_outs.insert(id);
            }
        }
    }
    for output in &program.outputs {
        let id = root(*output);
        if written.contains(&id) {
            live_outs.insert(id);
        }
    }
    live_outs.insert(hidden_output);
    live_outs.insert(status);
    if live_outs
        .iter()
        .any(|id| *id != hidden_output && *id != status)
    {
        return Ok(None);
    }

    let mut persistent = HashSet::new();
    let mut segments = HashSet::new();
    for &id in &used {
        match executable.memory.locations.get(id.index()) {
            Some(Location::Persistent { .. }) => {
                if written.contains(&id) {
                    return Ok(None);
                }
                persistent.insert(id);
            }
            Some(Location::Segment { segment, .. }) => {
                segments.insert(*segment);
            }
            Some(Location::External { .. }) if live_ins.contains(&id) => {}
            _ => return Ok(None),
        }
    }
    live_ins.retain(|id| !persistent.contains(id));
    let persistent_values = commands
        .iter()
        .enumerate()
        .filter_map(|(position, command)| {
            if !matches!(command.kind, CommandKind::Value(_)) {
                return None;
            }
            command
                .output
                .filter(|id| persistent.contains(id))
                .map(|id| (id, position))
        })
        .collect();
    let aliases_in_range = |range: std::ops::Range<usize>, roots: &HashSet<ValueId>| {
        sorted(
            commands[range]
                .iter()
                .filter_map(|command| command.output)
                .filter(|id| root(*id) != *id && roots.contains(&root(*id)))
                .collect(),
        )
    };
    let live_in_aliases = aliases_in_range(0..start, &live_ins);
    let live_out_aliases = aliases_in_range(start..end, &live_outs);
    // Include external bindings defined before the selected body as well.
    for command in &commands[..start] {
        if let CommandKind::Input { binding } = command.kind {
            if let Some(id) = command.output.map(root).filter(|id| live_ins.contains(id)) {
                bindings.push((id, binding));
            }
        }
    }
    bindings.sort_by_key(|(id, _)| id.index());
    bindings.dedup();
    let mut segments = segments.into_iter().collect::<Vec<_>>();
    segments.sort_by_key(|id| id.index());
    Ok(Some(BodyPlan {
        start,
        end,
        hidden_input,
        hidden_output,
        live_ins: sorted(live_ins),
        live_outs: sorted(live_outs),
        live_in_aliases,
        live_out_aliases,
        bindings,
        persistent: sorted(persistent),
        persistent_values,
        state_commands,
        status,
        segments,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_moe_graph_requires_its_owning_primary_stream() {
        assert!(native_primary(CommandOverlap::Primary));
        assert!(!native_primary(CommandOverlap::Join { branch: 0 }));
        for start in [false, true] {
            for finish in [false, true] {
                assert!(!native_primary(CommandOverlap::Worker {
                    branch: 0,
                    start,
                    finish,
                }));
            }
        }
    }
    fn command(overlap: CommandOverlap) -> Command {
        Command {
            output: None,
            kind: CommandKind::Prepare,
            overlap,
        }
    }
    #[test]
    fn graph_boundary_rejects_missing_duplicate_and_unfinished_worker_edges() {
        let worker = |start, finish| CommandOverlap::Worker {
            branch: 7,
            start,
            finish,
        };
        let join = CommandOverlap::Join { branch: 7 };
        assert!(closed_overlap(&[
            command(worker(true, false)),
            command(worker(false, true)),
            command(join)
        ]));
        assert!(closed_overlap(&[
            command(worker(true, true)),
            command(join)
        ]));
        for edges in [
            vec![worker(false, true), join],
            vec![worker(true, false), join],
            vec![worker(true, true)],
            vec![worker(true, false), worker(true, true), join],
            vec![worker(true, true), worker(false, false), join],
            vec![join],
        ] {
            assert!(!closed_overlap(
                &edges.into_iter().map(command).collect::<Vec<_>>()
            ));
        }
    }
    #[test]
    fn graph_admission_excludes_dynamic_scalar_and_cursor_commands() {
        for kind in [
            CommandKind::Scalar { binding: 0 },
            CommandKind::Cursor { tensor: true },
        ] {
            assert!(!supported(&Command {
                output: None,
                kind,
                overlap: CommandOverlap::Primary
            }));
        }
    }
}

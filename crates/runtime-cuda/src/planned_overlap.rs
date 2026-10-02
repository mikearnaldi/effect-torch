//! Conservative dense/expert fork/join scheduling with planned storage.
//!
//! The worker commands retain every referenced root until the primary stream
//! joins them. This includes borrowed inputs and GEMM scratch, not only outputs:
//! sequential last-use intervals are insufficient for asynchronous execution.
use crate::executable::StateAccess;
use crate::lowering::{Command, CommandKind, CommandOverlap, CudaLoweredProgram};
use effect_torch_compiler::{normalize_aliases, ValueUse};
use effect_torch_runtime::ValueId;
use std::collections::{BTreeSet, HashSet};

fn worker_command(command: &Command) -> bool {
    match &command.kind {
        CommandKind::Gemm { out_f32, .. } => !out_f32,
        CommandKind::PackedProjection77 { .. } => true,
        CommandKind::FusedElementwise { .. } => true,
        CommandKind::Kernel {
            checked,
            name,
            state,
            kv_matmul,
            ..
        } => {
            !checked
                && !name.starts_with("et_random_")
                && matches!(state, StateAccess::None)
                && kv_matmul.is_none()
        }
        _ => false,
    }
}

fn host_command(command: &Command) -> bool {
    matches!(
        command.kind,
        CommandKind::Prepare
            | CommandKind::Value(_)
            | CommandKind::Input { .. }
            | CommandKind::Alias { .. }
            | CommandKind::PlannedAlias
    )
}

/// Marks the existing independent three-projection dense branch. No arithmetic
/// or command ordering changes; runtime dispatch supplies the fork/join events.
pub(super) fn plan_dense_overlap(
    program: &mut CudaLoweredProgram,
    commands: &mut [Command],
) -> Result<usize, String> {
    let trace =
        std::env::var("EFFECT_TORCH_CUDA_DENSE_OVERLAP_TRACE").is_ok_and(|value| value == "1");
    if trace {
        let grouped = commands
            .iter()
            .filter(|command| matches!(command.kind, CommandKind::GroupedExpert { .. }))
            .count();
        let gemms = commands
            .iter()
            .filter(|command| matches!(command.kind, CommandKind::Gemm { .. }))
            .count();
        eprintln!(
            "CUDA overlap planning commands={} groups={grouped} gemms={gemms}",
            commands.len()
        );
    }
    // State staging/commit and deferred status instructions participate in
    // storage planning, but execute outside the command loop. Their positions
    // must never be mistaken for GPU commands when collecting a branch.
    let instruction_positions = program
        .instructions
        .iter()
        .enumerate()
        .filter_map(|(position, instruction)| {
            (!matches!(
                instruction.kind,
                "state_prepare" | "state_commit" | "state_discard" | "status_check"
            ))
            .then_some(position)
        })
        .collect::<Vec<_>>();
    if instruction_positions.len() != commands.len() {
        return Err("compile: CUDA overlap command/instruction mapping mismatch".into());
    }
    let aliases = normalize_aliases(&program.values).map_err(|error| error.to_string())?;
    let root = |id: ValueId| aliases[id.index()].root.index();
    if trace {
        if let Some(path) = std::env::var_os("EFFECT_TORCH_CUDA_DENSE_OVERLAP_DUMP_PATH") {
            let mut text = String::new();
            for (position, instruction) in program.instructions.iter().enumerate() {
                use std::fmt::Write;
                let _ = writeln!(
                    text,
                    "{position} {} inputs={:?} outputs={:?} resources={:?}",
                    instruction.kind,
                    instruction
                        .inputs
                        .iter()
                        .map(|input| root(input.value))
                        .collect::<Vec<_>>(),
                    instruction
                        .outputs
                        .iter()
                        .map(|output| (
                            root(output.value),
                            &program.values[output.value.index()].shape
                        ))
                        .collect::<Vec<_>>(),
                    instruction
                        .resource_uses()
                        .map(|value| (root(value.value), value.access))
                        .collect::<Vec<_>>()
                );
            }
            use std::hash::{Hash, Hasher};
            let mut fingerprint = std::collections::hash_map::DefaultHasher::new();
            text.hash(&mut fingerprint);
            let path = std::path::PathBuf::from(path).with_extension(format!(
                "{}.{:016x}.txt",
                commands.len(),
                fingerprint.finish()
            ));
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
            {
                Ok(mut file) => {
                    use std::io::Write;
                    file.write_all(text.as_bytes())
                        .map_err(|error| format!("CUDA overlap diagnostic dump: {error}"))?;
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(format!("CUDA overlap diagnostic dump: {error}")),
            }
        }
    }
    let mut producers = vec![None; program.values.len()];
    for (position, &instruction_position) in instruction_positions.iter().enumerate() {
        let instruction = &program.instructions[instruction_position];
        for output in &instruction.outputs {
            producers[root(output.value)].get_or_insert(position);
        }
    }

    // Boundary ancestors include the routed inputs, router and shared residual.
    // Traversal stops there when isolating a dense branch, so the worker never
    // repeats a shared producer or carries attention/state onto its stream.
    let ancestors = |seeds: Vec<usize>| {
        let mut found = HashSet::new();
        let mut pending = seeds;
        while let Some(value) = pending.pop() {
            if !found.insert(value) {
                continue;
            }
            if let Some(position) = producers[value] {
                for input in program.instructions[instruction_positions[position]].resource_uses() {
                    let dependency = root(input.value);
                    if input.access.reads() && dependency != value {
                        pending.push(dependency);
                    }
                }
            }
        }
        found
    };

    let mut planned = Vec::<(Vec<usize>, usize, BTreeSet<usize>)>::new();
    let mut occupied_until = 0;
    let mut traced_candidates = 0;
    for group in 0..commands.len() {
        if !matches!(
            commands[group].kind,
            CommandKind::GroupedExpert {
                reuse_routing: false,
                ..
            } | CommandKind::FusedMoe75 { .. }
        ) {
            continue;
        }
        let boundary = ancestors(
            program.instructions[instruction_positions[group]]
                .inputs
                .iter()
                .map(|input| root(input.value))
                .collect(),
        );
        let mut dependent = vec![false; program.values.len()];
        for output in &program.instructions[instruction_positions[group]].outputs {
            dependent[root(output.value)] = true;
        }
        for join in group + 1..commands.len() {
            let instruction = &program.instructions[instruction_positions[join]];
            let has_dependent = instruction
                .inputs
                .iter()
                .any(|input| dependent[root(input.value)]);
            if has_dependent {
                for output in &instruction.outputs {
                    dependent[root(output.value)] = true;
                }
            }
            if !has_dependent || !worker_command(&commands[join]) {
                continue;
            }
            let mut accepted = None;
            for input in &instruction.inputs {
                let value = root(input.value);
                if dependent[value] || boundary.contains(&value) {
                    continue;
                }
                let mut selected = BTreeSet::new();
                let mut pending = vec![value];
                let mut visited = HashSet::new();
                let mut valid = true;
                let mut rejected_producer = None;
                while let Some(value) = pending.pop() {
                    if boundary.contains(&value) || !visited.insert(value) {
                        continue;
                    }
                    let Some(position) = producers[value] else {
                        continue;
                    };
                    if position >= join {
                        rejected_producer = Some(position);
                        valid = false;
                        break;
                    }
                    let command = &commands[position];
                    if host_command(command) {
                        continue;
                    }
                    if !worker_command(command)
                        || program.instructions[instruction_positions[position]]
                            .effects
                            .has_side_effects
                    {
                        rejected_producer = Some(position);
                        valid = false;
                        break;
                    }
                    if selected.insert(position) {
                        for resource in
                            program.instructions[instruction_positions[position]].resource_uses()
                        {
                            if resource.access.reads() && root(resource.value) != value {
                                pending.push(root(resource.value));
                            }
                        }
                    }
                }
                let gemms = selected
                    .iter()
                    .map(|&position| match &commands[position].kind {
                        CommandKind::Gemm { .. } => 1,
                        CommandKind::PackedProjection77 { outputs, .. } => outputs.len(),
                        _ => 0,
                    })
                    .sum::<usize>();
                if trace && rejected_producer.is_some() && traced_candidates < 12 {
                    let position = rejected_producer.unwrap();
                    eprintln!(
                        "CUDA overlap candidate rejected group={group} join={join} input={value} shape={:?} producer={position} kind={} gemms={gemms}",
                        program.values[value].shape,
                        program.instructions[instruction_positions[position]].kind
                    );
                    traced_candidates += 1;
                }
                if trace && gemms != 0 {
                    eprintln!(
                        "CUDA overlap candidate group={group} join={join} input={value} valid={valid} gemms={gemms} commands={}",
                        selected.len()
                    );
                }
                if !valid || gemms != 3 {
                    continue;
                }
                let first = *selected
                    .first()
                    .expect("three GEMMs imply a nonempty branch");
                if first >= group {
                    continue;
                }
                if first < occupied_until {
                    if trace {
                        eprintln!(
                            "CUDA overlap rejected: branch starts {first} before prior join {occupied_until}"
                        );
                    }
                    continue;
                }
                let touched = selected
                    .iter()
                    .flat_map(|&position| {
                        program.instructions[instruction_positions[position]].resource_uses()
                    })
                    .map(|resource| root(resource.value))
                    .collect::<BTreeSet<_>>();
                // Every external GPU producer must precede the fork. The worker
                // waits once at its start; later primary dependencies would need
                // additional events and are deliberately rejected.
                if touched.iter().any(|&value| {
                    producers[value].is_some_and(|position| {
                        !selected.contains(&position)
                            && position >= first
                            && !host_command(&commands[position])
                    })
                }) {
                    if trace {
                        eprintln!(
                            "CUDA overlap rejected: external GPU dependency follows fork {first}"
                        );
                    }
                    continue;
                }
                // A logical write to borrowed storage in the overlap window is
                // unsafe even if sequential planning gave it a valid interval.
                if (first..join).any(|position| {
                    !selected.contains(&position)
                        && !host_command(&commands[position])
                        && program.instructions[instruction_positions[position]]
                            .resource_uses()
                            .any(|resource| {
                                resource.access.writes() && touched.contains(&root(resource.value))
                            })
                }) {
                    if trace {
                        eprintln!(
                            "CUDA overlap rejected: primary writes worker resource before join {join}"
                        );
                    }
                    continue;
                }
                // Any outside consumer of a worker write before the chosen
                // join would require an earlier completion dependency. Scratch
                // writes have the same hazard as ordinary branch outputs.
                let produced = selected
                    .iter()
                    .flat_map(|&position| {
                        program.instructions[instruction_positions[position]].resource_uses()
                    })
                    .filter(|resource| resource.access.writes())
                    .map(|resource| root(resource.value))
                    .collect::<HashSet<_>>();
                if (first..join).any(|position| {
                    !selected.contains(&position)
                        && !host_command(&commands[position])
                        && program.instructions[instruction_positions[position]]
                            .resource_uses()
                            .any(|resource| {
                                resource.access.reads() && produced.contains(&root(resource.value))
                            })
                }) {
                    if trace {
                        eprintln!(
                            "CUDA overlap rejected: primary consumes worker output before join {join}"
                        );
                    }
                    continue;
                }
                accepted = Some((selected.into_iter().collect(), join, touched));
                break;
            }
            if let Some(branch) = accepted {
                occupied_until = join + 1;
                planned.push(branch);
                break;
            }
        }
    }

    for (branch, (selected, join, touched)) in planned.iter().enumerate() {
        for (index, &position) in selected.iter().enumerate() {
            commands[position].overlap = CommandOverlap::Worker {
                branch,
                start: index == 0,
                finish: index + 1 == selected.len(),
            };
        }
        commands[*join].overlap = CommandOverlap::Join { branch };
        let instruction = &mut program.instructions[instruction_positions[*join]];
        let existing = instruction
            .resource_uses()
            .map(|resource| root(resource.value))
            .collect::<HashSet<_>>();
        let retained = touched
            .iter()
            .filter(|value| !existing.contains(value))
            .map(|&value| ValueUse::read(aliases[value].root));
        instruction.scratch = instruction
            .scratch
            .iter()
            .copied()
            .chain(retained)
            .collect();
    }
    Ok(planned.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cublas::Bf16GemmPlan;
    use crate::executable::CudaKernelArgs;
    use crate::lowering::CudaValueMeta;
    use crate::workspace::CudaMemorySpace;
    use effect_torch_compiler::{
        analyze_liveness, plan_memory, LoweredInstruction, LoweredProgram, MemoryPlannerConfig,
        OutputDecl, ValueDecl,
    };
    use effect_torch_runtime::{DType, InstructionId, SegmentOwnership, StorageMetadata};

    fn fixture() -> (CudaLoweredProgram, Vec<Command>) {
        let values = (0..16)
            .map(|index| CudaValueMeta {
                decl: ValueDecl::planned(
                    ValueId::from_index(index).unwrap(),
                    format!("value_{index}"),
                    256,
                    256,
                    CudaMemorySpace::Device,
                    SegmentOwnership::Workspace,
                ),
                shape: vec![128],
                dtype: DType::BF16,
                storage: StorageMetadata::dense(),
            })
            .collect::<Vec<_>>();
        let mut instructions = Vec::new();
        let mut commands = Vec::new();
        let mut emit = |kind: CommandKind, inputs: &[u32], outputs: &[u32], scratch: &[u32]| {
            let id = InstructionId::from_index(instructions.len()).unwrap();
            instructions.push(
                LoweredInstruction::new(
                    id,
                    "fixture",
                    inputs
                        .iter()
                        .map(|&id| ValueUse::read(ValueId::new(id)))
                        .collect::<Vec<_>>(),
                    outputs
                        .iter()
                        .map(|&id| OutputDecl::new(ValueId::new(id)))
                        .collect::<Vec<_>>(),
                )
                .with_resources(
                    scratch
                        .iter()
                        .map(|&id| ValueUse::read_write(ValueId::new(id)))
                        .collect::<Vec<_>>(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                ),
            );
            commands.push(Command {
                output: outputs.first().copied().map(ValueId::new),
                kind,
                overlap: CommandOverlap::Primary,
            });
        };
        emit(CommandKind::Prepare, &[], &[0], &[]);
        for (x, workspace, out) in [(0, 1, 2), (2, 3, 4), (4, 5, 6)] {
            emit(CommandKind::Prepare, &[], &[workspace], &[]);
            emit(
                CommandKind::Gemm {
                    x: ValueId::new(x),
                    weight: ValueId::new(0),
                    weight_transposed: true,
                    plan: Bf16GemmPlan {
                        m: 1,
                        n: 128,
                        k: 128,
                        batch: 1,
                        stride_x: 128,
                        stride_weight: 128 * 128,
                        stride_out: 128,
                    },
                    out_f32: false,
                    workspace: ValueId::new(workspace),
                },
                &[x, 0],
                &[out],
                &[workspace],
            );
        }
        emit(CommandKind::Prepare, &[], &[7, 8, 9, 10], &[]);
        emit(
            CommandKind::GroupedExpert {
                x: ValueId::new(0),
                weight: ValueId::new(0),
                indexes: ValueId::new(0),
                rows: 1,
                columns: 128,
                inner: 128,
                experts: 1,
                control: ValueId::new(7),
                row_map: ValueId::new(8),
                gathered: ValueId::new(9),
                projected: ValueId::new(10),
                workspace: None,
                reuse_routing: false,
                splitk_workspace: false,
                source_rows: 1,
                input_sorted: false,
                output_sorted: false,
                inverse_routing: false,
                device_control: None,
            },
            &[0],
            &[11],
            &[7, 8, 9, 10],
        );
        emit(CommandKind::Prepare, &[], &[12], &[]);
        let kernel = |ids: &[u32]| {
            let mut inputs = [None; 8];
            for (role, &id) in ids.iter().enumerate() {
                inputs[role] = Some(ValueId::new(id));
            }
            CommandKind::Kernel {
                name: "et_binary",
                args: CudaKernelArgs::default(),
                inputs,
                scratch: [None; 3],
                status: None,
                checked: false,
                metadata: Vec::new(),
                state: StateAccess::None,
                state_buffers: [None; 4],
                kv_matmul: None,
            }
        };
        emit(kernel(&[11]), &[11], &[13], &[]);
        emit(kernel(&[6, 13]), &[6, 13], &[14], &[]);
        emit(kernel(&[12]), &[12], &[15], &[]);
        (
            LoweredProgram::new(
                values,
                instructions,
                vec![ValueId::new(14), ValueId::new(15)],
            ),
            commands,
        )
    }

    #[test]
    fn packed_dense_keeps_worker_stream_and_scratch_lifetime() {
        let (mut program, mut commands) = fixture();
        let CommandKind::Gemm { plan, .. } = commands[2].kind else {
            panic!("fixture")
        };
        commands[2].kind = CommandKind::PackedProjection77 {
            skip_split: false,
            x: ValueId::new(0),
            weight: ValueId::new(0),
            outputs: vec![ValueId::new(2), ValueId::new(4)],
            widths: vec![64, 64],
            temporary: ValueId::new(3),
            workspace: ValueId::new(1),
            plan,
        };
        program.instructions[2].outputs = vec![
            OutputDecl::new(ValueId::new(2)),
            OutputDecl::new(ValueId::new(4)),
            OutputDecl::new(ValueId::new(3)),
        ]
        .into_boxed_slice();
        program.instructions[2].scratch =
            vec![ValueUse::read_write(ValueId::new(1))].into_boxed_slice();
        // The removed projection becomes a read-only view of its earlier result.
        commands[4].kind = CommandKind::PlannedAlias;
        program.instructions[4].inputs = vec![ValueUse::read(ValueId::new(4))].into_boxed_slice();
        program.instructions[4].outputs = Box::new([]);
        program.instructions[4].scratch = Box::new([]);
        // This synthetic scratch now belongs to the packed command.
        program.instructions[3].outputs = Box::new([]);
        commands[3].output = None;
        assert_eq!(plan_dense_overlap(&mut program, &mut commands).unwrap(), 1);
        assert!(matches!(
            commands[2].overlap,
            CommandOverlap::Worker { start: true, .. }
        ));
        let live = analyze_liveness(&program).unwrap();
        assert_eq!(live.intervals[3].unwrap().end.index(), 12);
    }

    #[test]
    fn overlap_retains_dense_inputs_intermediates_and_workspace_until_join() {
        let (mut program, mut commands) = fixture();
        let before = analyze_liveness(&program).unwrap();
        assert_eq!(before.intervals[1].unwrap().end.index(), 3);
        assert_eq!(before.intervals[2].unwrap().end.index(), 5);
        assert_eq!(plan_dense_overlap(&mut program, &mut commands).unwrap(), 1);
        let after = analyze_liveness(&program).unwrap();
        for value in 0..=6 {
            assert_eq!(after.intervals[value].unwrap().end.index(), 12);
        }
        assert_eq!(
            commands[2].overlap,
            CommandOverlap::Worker {
                branch: 0,
                start: true,
                finish: false
            }
        );
        assert_eq!(
            commands[6].overlap,
            CommandOverlap::Worker {
                branch: 0,
                start: false,
                finish: true
            }
        );
        assert_eq!(commands[11].overlap, CommandOverlap::Join { branch: 0 });
        let memory = plan_memory(
            &program,
            &MemoryPlannerConfig::uniform(CudaMemorySpace::Device, 1 << 20, 256, 256),
        )
        .unwrap();
        for retained in 0..=6 {
            assert_ne!(memory.locations[retained], memory.locations[12]);
        }
    }

    #[test]
    fn packed_ffn_next_norm_keeps_the_dense_join_and_both_output_roots() {
        let (mut program, mut commands) = fixture();
        // The fused tail writes one owner, then exposes the residual and next
        // normalized input as disjoint aliases. Both remain live to later reads.
        program.values[14].decl.bytes = 512;
        program.values[14].shape = vec![2, 128];
        let mut values = program.values.into_vec();
        for (id, offset) in [(16, 0), (17, 256)] {
            values.push(CudaValueMeta {
                decl: ValueDecl::alias(
                    ValueId::new(id),
                    format!("ffn_result_{id}"),
                    ValueId::new(14),
                    offset,
                    256,
                ),
                shape: vec![128],
                dtype: DType::BF16,
                storage: StorageMetadata::dense(),
            });
        }
        program.values = values.into_boxed_slice();
        let CommandKind::Kernel { name, .. } = &mut commands[11].kind else {
            unreachable!()
        };
        *name = "et_ffn_next_norm63_bf16";
        program.instructions[11].kind = "ffn_next_norm";
        let mut instructions = program.instructions.into_vec();
        for (position, id) in [(12, 16), (13, 17)] {
            instructions.insert(
                position,
                LoweredInstruction::new(
                    InstructionId::new(0),
                    "alias",
                    vec![ValueUse::read(ValueId::new(14))],
                    vec![OutputDecl::new(ValueId::new(id))],
                ),
            );
            commands.insert(
                position,
                Command {
                    output: Some(ValueId::new(id)),
                    kind: CommandKind::PlannedAlias,
                    overlap: CommandOverlap::Primary,
                },
            );
        }
        instructions[14].inputs = [12, 16, 17]
            .map(|id| ValueUse::read(ValueId::new(id)))
            .into();
        let CommandKind::Kernel { inputs, .. } = &mut commands[14].kind else {
            unreachable!()
        };
        inputs[..3].copy_from_slice(&[12, 16, 17].map(|id| Some(ValueId::new(id))));
        for (position, instruction) in instructions.iter_mut().enumerate() {
            instruction.id = InstructionId::from_index(position).unwrap();
        }
        program.instructions = instructions.into_boxed_slice();
        program.outputs =
            vec![ValueId::new(16), ValueId::new(17), ValueId::new(15)].into_boxed_slice();

        assert_eq!(plan_dense_overlap(&mut program, &mut commands).unwrap(), 1);
        assert_eq!(commands[11].overlap, CommandOverlap::Join { branch: 0 });
        assert!(commands[12..]
            .iter()
            .all(|command| command.overlap == CommandOverlap::Primary));
        let liveness = analyze_liveness(&program).unwrap();
        for value in 0..=6 {
            assert_eq!(liveness.intervals[value].unwrap().end.index(), 12);
        }
        assert!(liveness.intervals[14].unwrap().end.index() >= 15);
        assert_eq!(liveness.intervals[16], None);
        assert_eq!(liveness.intervals[17], None);
        let aliases = normalize_aliases(&program.values).unwrap();
        assert_eq!(aliases[16].root, ValueId::new(14));
        assert_eq!(aliases[17].root, ValueId::new(14));
        assert_eq!(aliases[17].byte_offset, 256);
        plan_memory(
            &program,
            &MemoryPlannerConfig::uniform(CudaMemorySpace::Device, 1 << 20, 256, 256),
        )
        .unwrap();
    }

    #[test]
    fn overlap_rejects_an_early_primary_consumer_of_worker_output() {
        let (mut program, mut commands) = fixture();
        program.instructions[10].inputs = vec![
            ValueUse::read(ValueId::new(11)),
            ValueUse::read(ValueId::new(2)),
        ]
        .into_boxed_slice();
        assert_eq!(plan_dense_overlap(&mut program, &mut commands).unwrap(), 0);
        assert!(commands
            .iter()
            .all(|command| command.overlap == CommandOverlap::Primary));
    }

    #[test]
    fn overlap_rejects_an_early_primary_consumer_of_worker_scratch() {
        let (mut program, mut commands) = fixture();
        program.instructions[10].inputs = vec![
            ValueUse::read(ValueId::new(11)),
            ValueUse::read(ValueId::new(1)),
        ]
        .into_boxed_slice();
        assert_eq!(plan_dense_overlap(&mut program, &mut commands).unwrap(), 0);
        assert!(commands
            .iter()
            .all(|command| command.overlap == CommandOverlap::Primary));
    }

    #[test]
    fn non_command_state_instructions_preserve_command_mapping_and_retention() {
        let (mut program, mut commands) = fixture();
        let mut instructions = program.instructions.into_vec();
        // Real state staging definitions are interleaved with the command
        // stream; commits and status checks are appended after its last entry.
        for position in [0, 4] {
            instructions.insert(
                position,
                LoweredInstruction::new(
                    InstructionId::new(0),
                    "state_prepare",
                    Vec::new(),
                    Vec::new(),
                ),
            );
        }
        for kind in ["state_commit", "state_discard", "status_check"] {
            instructions.push(LoweredInstruction::new(
                InstructionId::new(0),
                kind,
                Vec::new(),
                Vec::new(),
            ));
        }
        for (position, instruction) in instructions.iter_mut().enumerate() {
            instruction.id = InstructionId::from_index(position).unwrap();
        }
        program.instructions = instructions.into_boxed_slice();
        assert_eq!(plan_dense_overlap(&mut program, &mut commands).unwrap(), 1);
        assert_eq!(
            commands[2].overlap,
            CommandOverlap::Worker {
                branch: 0,
                start: true,
                finish: false
            }
        );
        assert_eq!(commands[11].overlap, CommandOverlap::Join { branch: 0 });
        let after = analyze_liveness(&program).unwrap();
        for value in 0..=6 {
            assert_eq!(after.intervals[value].unwrap().end.index(), 14);
        }
    }

    #[test]
    fn overlap_rejects_unknown_unmapped_instructions() {
        let (mut program, mut commands) = fixture();
        let mut instructions = program.instructions.into_vec();
        instructions.push(LoweredInstruction::new(
            InstructionId::new(13),
            "unexpected",
            Vec::new(),
            Vec::new(),
        ));
        program.instructions = instructions.into_boxed_slice();
        assert_eq!(
            plan_dense_overlap(&mut program, &mut commands).unwrap_err(),
            "compile: CUDA overlap command/instruction mapping mismatch"
        );
        assert!(commands
            .iter()
            .all(|command| command.overlap == CommandOverlap::Primary));
    }

    #[test]
    fn worker_alias_reads_extend_the_root_through_completion() {
        let (mut program, mut commands) = fixture();
        let mut values = program.values.into_vec();
        values.push(CudaValueMeta {
            decl: ValueDecl::alias(ValueId::new(16), "dense_view", ValueId::new(2), 0, 256),
            shape: vec![128],
            dtype: DType::BF16,
            storage: StorageMetadata::dense(),
        });
        program.values = values.into_boxed_slice();
        program.instructions[4].inputs[0] = ValueUse::read(ValueId::new(16));
        let CommandKind::Gemm { x, .. } = &mut commands[4].kind else {
            unreachable!()
        };
        *x = ValueId::new(16);
        assert_eq!(plan_dense_overlap(&mut program, &mut commands).unwrap(), 1);
        let liveness = analyze_liveness(&program).unwrap();
        assert_eq!(liveness.intervals[2].unwrap().end.index(), 12);
        assert_eq!(liveness.intervals[16], None);
    }

    #[test]
    fn dense_commands_can_straddle_the_primary_grouped_projection() {
        let (mut program, commands) = fixture();
        let mut instructions = program
            .instructions
            .into_vec()
            .into_iter()
            .map(Some)
            .collect::<Vec<_>>();
        let mut commands = commands.into_iter().map(Some).collect::<Vec<_>>();
        let order = [0, 1, 2, 3, 4, 7, 8, 5, 6, 9, 10, 11, 12];
        let mut reordered_commands = Vec::new();
        program.instructions = order
            .into_iter()
            .enumerate()
            .map(|(position, previous)| {
                reordered_commands.push(commands[previous].take().unwrap());
                let mut instruction = instructions[previous].take().unwrap();
                instruction.id = InstructionId::from_index(position).unwrap();
                instruction
            })
            .collect();
        assert_eq!(
            plan_dense_overlap(&mut program, &mut reordered_commands).unwrap(),
            1
        );
        assert_eq!(
            reordered_commands[8].overlap,
            CommandOverlap::Worker {
                branch: 0,
                start: false,
                finish: true
            }
        );
        assert_eq!(
            reordered_commands[11].overlap,
            CommandOverlap::Join { branch: 0 }
        );
        analyze_liveness(&program).unwrap();
    }
}

//! Pure physical admission for the screened packed local QKV normalization chain.
//! No allocation, CUDA access, or mutation occurs until every check succeeds.
use crate::executable::StateAccess;
use crate::lowering::{Command, CommandKind, CommandOverlap, CudaValueMeta};
use crate::triton_normrope101::{KV_BYTES, PACKED_BYTES, Q_BYTES};
use effect_torch_compiler::{normalize_aliases, LoweredInstruction, ValueAccess, ValueStorage};
use effect_torch_runtime::{DType, StorageRepresentation, ValueId};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Group101 {
    pub(crate) packed_command: usize,
    pub(crate) norm_commands: [usize; 3],
    pub(crate) norm_instructions: [usize; 3],
    pub(crate) dispatch_command: usize,
    pub(crate) dispatch_instruction: usize,
    pub(crate) prepare_instruction: usize,
    pub(crate) packed_temporary: ValueId,
    pub(crate) borrowed: [ValueId; 4],
    pub(crate) table_strides: [u32; 2],
    pub(crate) outputs: [ValueId; 3],
}
fn dense(values: &[CudaValueMeta], id: ValueId, bytes: usize) -> bool {
    values.get(id.index()).is_some_and(|value| {
        value.dtype == DType::BF16
            && value.storage.representation == StorageRepresentation::Dense
            && value.decl.bytes == bytes
    })
}
fn source_view(args: &crate::executable::CudaKernelArgs, heads: usize) -> bool {
    args.integers[0] == 256
        && args.integers[1] == 3
        && args.integers[2] == 0
        && args.integers[3..6] == [1, heads as u64, 256]
        && matches!(args.integers[6], 0 | 4096 | 2048)
        && (args.integers[6] == 0 || args.integers[6] == (heads * 256) as u64)
        && args.integers[7..9] == [1, heads as u64]
}
fn table_layout(metadata: &[u64]) -> bool {
    let Some(layouts) = metadata.get(metadata.len().saturating_sub(10)..) else {
        return false;
    };
    layouts.len() == 10
        && layouts.chunks_exact(5).all(|layout| {
            layout[0] == 0
                && layout[1] == 128
                && layout[2] == 1
                && matches!(layout[3], 0 | 256)
                && layout[4] == 128
        })
}
// Full-width tables need structural repeated-half provenance established by
// the semantic lowerer. Arbitrary full tables are intentionally not admitted.
fn table_strides(args: &crate::executable::CudaKernelArgs, metadata: &[u64]) -> Option<[u32; 2]> {
    if args.operation & !7 != 0 {
        return None;
    }
    let layouts = metadata.get(metadata.len().checked_sub(10)?..)?;
    let mut strides = [0; 2];
    for slot in 0..2 {
        let layout = &layouts[slot * 5..slot * 5 + 5];
        let view = args.operation & (1 << slot) != 0;
        strides[slot] = if view
            && layout[0] == 0
            && layout[1] == 128
            && layout[2] == 1
            && matches!(layout[3], 0 | 256)
            && layout[4] == 128
        {
            128
        } else if !view && layout == [0, 256, 1, 0, 0] && args.integers[15] & (1 << slot) != 0 {
            256
        } else {
            return None;
        };
    }
    Some(strides)
}

fn aliases_only(command: &Command, instruction: &LoweredInstruction<&'static str>) -> bool {
    matches!(
        command.kind,
        CommandKind::Alias { .. } | CommandKind::PlannedAlias
    ) && !instruction.effects.has_side_effects
        && !instruction.effects.may_fail
        && instruction
            .inputs
            .iter()
            .chain(instruction.scratch.iter())
            .chain(instruction.staging.iter())
            .chain(instruction.status.iter())
            .chain(instruction.state.iter())
            .all(|usage| usage.access == ValueAccess::Read)
}

pub(crate) fn admit(
    commands: &[Command],
    lowered: &[LoweredInstruction<&'static str>],
    values: &[CudaValueMeta],
    roots: &[ValueId],
    packed_command: usize,
) -> Option<Group101> {
    admit_diagnostic(commands, lowered, values, roots, packed_command, None)
}

pub(crate) fn admit_diagnostic(
    commands: &[Command],
    lowered: &[LoweredInstruction<&'static str>],
    values: &[CudaValueMeta],
    roots: &[ValueId],
    packed_command: usize,
    mut diagnostics: Option<&mut Vec<serde_json::Value>>,
) -> Option<Group101> {
    macro_rules! decline {
        ($reason:expr) => {{
            if let Some(records) = diagnostics.as_deref_mut() { records.push(serde_json::json!({"event":"decline", "reason":$reason, "line":line!()})); }
            return None;
        }};
    }
    macro_rules! detail {
        ($value:expr) => {
            if let Some(records) = diagnostics.as_deref_mut() {
                records.push($value);
            }
        };
    }
    let packed = commands.get(packed_command)?;
    let CommandKind::PackedProjection77 {
        outputs: raw,
        widths,
        temporary,
        plan,
        skip_split,
        ..
    } = &packed.kind
    else {
        decline!("not_packed");
    };
    let aliases = normalize_aliases(values).ok()?;
    let root = |id: ValueId| aliases.get(id.index()).map(|alias| alias.root);
    let mapping = lowered
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
    if mapping.len() != commands.len() {
        decline!("command_instruction_mapping");
    }
    let mut reverse = vec![None; lowered.len()];
    for (command, &instruction) in mapping.iter().enumerate() {
        reverse[instruction] = Some(command);
    }
    if packed.overlap != CommandOverlap::Primary
        || *skip_split
        || widths.as_slice() != [4096, 2048, 2048]
        || raw.len() != 3
        || plan.m != 256
        || plan.n != 8192
        || plan.k != 2816
        || plan.batch != 1
        || !dense(values, *temporary, PACKED_BYTES)
        || values[temporary.index()].shape != [256, 8192]
        || !matches!(
            values[temporary.index()].decl.storage,
            ValueStorage::Planned { .. }
        )
    {
        decline!("packed_geometry_or_storage");
    }
    let mut raw_roots = [*temporary; 3];
    for (role, &id) in raw.iter().enumerate() {
        if !dense(values, id, if role == 0 { Q_BYTES } else { KV_BYTES })
            || !matches!(
                values[id.index()].decl.storage,
                ValueStorage::Planned { .. }
            )
            || aliases.get(id.index())?.byte_offset != 0
        {
            decline!("raw_output_geometry_or_storage");
        }
        raw_roots[role] = root(id)?;
    }
    if raw_roots[0] == raw_roots[1]
        || raw_roots[0] == raw_roots[2]
        || raw_roots[1] == raw_roots[2]
        || roots
            .iter()
            .any(|&id| root(id).is_some_and(|id| raw_roots.contains(&id)))
    {
        decline!("raw_alias_or_root_escape");
    }
    let mut found: [Option<(usize, ValueId, [Option<ValueId>; 4], [u32; 2])>; 3] =
        [None, None, None];
    for (position, command) in commands.iter().enumerate().skip(packed_command + 1) {
        let CommandKind::Kernel {
            name,
            args,
            inputs,
            scratch,
            status,
            checked,
            metadata,
            state,
            state_buffers,
            kv_matmul,
        } = &command.kind
        else {
            continue;
        };
        let Some(source_id) = inputs[0] else {
            continue;
        };
        let Some(source) = root(source_id) else {
            continue;
        };
        if aliases.get(source_id.index())?.byte_offset != 0 {
            continue;
        }
        let Some(role) = raw_roots.iter().position(|&raw| raw == source) else {
            continue;
        };
        detail!(
            serde_json::json!({"event":"raw_reader", "command":position, "role":role, "kernel":name,
            "source":source_id.get(), "output":command.output.map(|id|id.get()), "integers":args.integers,
            "scalars":args.scalars, "elements":args.elements, "operation":args.operation,
            "computeDtype":args.compute_dtype,"outputDtype":args.output_dtype,"inputDtypes":args.input_dtypes,
            "inputs":inputs.iter().map(|id|id.map(|id|id.get())).collect::<Vec<_>>(), "metadata":metadata,
            "sourceViewMatches":source_view(args,if role==0{16}else{8}),"compactTableLayoutMatches":table_layout(metadata),"admittedTableStrides":table_strides(args,metadata),
            "inputValues":inputs.iter().flatten().map(|id|serde_json::json!({"id":id.get(),"shape":values[id.index()].shape,"dtype":format!("{:?}",values[id.index()].dtype),"bytes":values[id.index()].decl.bytes,"storage":format!("{:?}",values[id.index()].decl.storage),"alias":format!("{:?}",aliases[id.index()])})).collect::<Vec<_>>() })
        );
        let heads = if role == 0 { 16 } else { 8 };
        if command.overlap != CommandOverlap::Primary
            || *checked
            || status.is_some()
            || scratch.iter().any(Option::is_some)
            || state_buffers.iter().any(Option::is_some)
            || kv_matmul.is_some()
            || !matches!(state, StateAccess::None)
            || args.elements != (256 * heads * 256) as u64
            || args.compute_dtype != 1
            || args.output_dtype != 3
            || args.input_dtypes[0] != 3
            || args.scalars[0].to_bits() != 1e-6_f64.to_bits()
            || !source_view(args, heads)
        {
            detail!(
                serde_json::json!({"event":"reject_candidate","command":position,"role":role,"reason":"base_norm_geometry_dtype_state_or_source_view"})
            );
            continue;
        }
        let output = command.output?;
        if !dense(values, output, if role == 0 { Q_BYTES } else { KV_BYTES })
            || values[output.index()].shape != [1, heads, 256, 256]
            || !matches!(
                values[output.index()].decl.storage,
                ValueStorage::Planned { .. }
            )
        {
            detail!(
                serde_json::json!({"event":"reject_candidate","command":position,"role":role,"reason":"norm_output_shape_dtype_or_storage"})
            );
            continue;
        }
        let borrowed = if role < 2 {
            if *name != "et_norm_rope_bf16"
                || args.operation & !7 != 0
                || args.integers[10..12] != [heads as u64, 256]
                || inputs[4..].iter().any(Option::is_some)
                || args.input_dtypes[1] != 3
                || table_strides(args, metadata).is_none()
            {
                detail!(
                    serde_json::json!({"event":"reject_candidate","command":position,"role":role,"reason":"norm_rope_name_operation_or_table_layout"})
                );
                continue;
            }
            let strides = table_strides(args, metadata)?;
            let weight = inputs[1]?;
            let cosine = inputs[2]?;
            let sine = inputs[3]?;
            if !dense(values, weight, 512)
                || values[weight.index()].shape != [256]
                || !dense(values, cosine, 256 * strides[0] as usize * 2)
                || !dense(values, sine, 256 * strides[1] as usize * 2)
            {
                detail!(
                    serde_json::json!({"event":"reject_candidate","command":position,"role":role,"reason":"weight_or_half_table_storage"})
                );
                continue;
            }
            ([Some(weight), Some(cosine), Some(sine), None], strides)
        } else {
            if *name != "et_rms_norm_f32"
                || inputs[1..].iter().any(Option::is_some)
                // Flags0/1 select wide vector/static2816 implementations.
                // Width256 uses the generic kernel which ignores these bits.
                || args.operation & !3 != 0
            {
                continue;
            }
            ([None; 4], [0; 2])
        };
        if found[role].is_some() {
            decline!("duplicate_norm_reader");
        }
        found[role] = Some((position, output, borrowed.0, borrowed.1));
    }
    detail!(
        serde_json::json!({"event":"norm_matches", "commands":found.iter().map(|x|x.as_ref().map(|x|x.0)).collect::<Vec<_>>() })
    );
    let Some(found) = found.into_iter().collect::<Option<Vec<_>>>() else {
        decline!("missing_q_k_or_v_norm");
    };
    let same_table = |left: ValueId, right: ValueId| {
        aliases[left.index()].root == aliases[right.index()].root
            && aliases[left.index()].byte_offset == aliases[right.index()].byte_offset
    };
    if found[0].3 != found[1].3
        || !same_table(found[0].2[1]?, found[1].2[1]?)
        || !same_table(found[0].2[2]?, found[1].2[2]?)
    {
        decline!("distinct_qk_tables");
    }
    let norm_commands = [found[0].0, found[1].0, found[2].0];
    let outputs = [found[0].1, found[1].1, found[2].1];
    let borrowed = [
        found[0].2[0]?,
        found[1].2[0]?,
        found[0].2[1]?,
        found[0].2[2]?,
    ];
    let dispatch_command = *norm_commands.iter().max()?;
    let dispatch_instruction = mapping[dispatch_command];
    let prepare_command = dispatch_command.checked_sub(1)?;
    let prepare_instruction = mapping[prepare_command];
    if !matches!(commands[prepare_command].kind, CommandKind::Prepare)
        || commands[prepare_command].output.is_some()
        || lowered[dispatch_instruction].outputs.as_ref()
            != [effect_torch_compiler::OutputDecl::new(
                outputs[norm_commands
                    .iter()
                    .position(|&command| command == dispatch_command)?],
            )]
        || lowered[prepare_instruction].kind != "prepare"
        || prepare_instruction + 1 != dispatch_instruction
    {
        decline!("dispatch_prepare_mapping");
    }
    let output_roots = outputs.map(root).into_iter().collect::<Option<Vec<_>>>()?;
    if output_roots[0] == output_roots[1]
        || output_roots[0] == output_roots[2]
        || output_roots[1] == output_roots[2]
    {
        decline!("overlapping_output_roots");
    }
    for &id in &borrowed {
        let borrowed_root = root(id)?;
        if raw_roots.contains(&borrowed_root) || output_roots.contains(&borrowed_root) {
            decline!("borrowed_aliases_mutable_output");
        }
        // Only move this operand's original reads. In a normal lazy QKV
        // schedule KW is bound after Q's normalization, before K's. Requiring
        // every operand before Q incorrectly excludes the production graph.
        let first_reader = norm_commands
            .iter()
            .copied()
            .filter(|&command| match &commands[command].kind {
                CommandKind::Kernel { inputs, .. } => inputs
                    .iter()
                    .flatten()
                    .any(|&input| root(input) == Some(borrowed_root)),
                _ => false,
            })
            .min()?;
        detail!(
            serde_json::json!({"event":"borrowed", "id":id.get(),"root":borrowed_root.get(),"firstReader":first_reader,"dispatch":dispatch_command})
        );
        if !commands[..first_reader]
            .iter()
            .any(|command| command.output.and_then(root) == Some(borrowed_root))
        {
            decline!("borrowed_producer_after_first_reader");
        }
        if lowered[mapping[first_reader]..=dispatch_instruction]
            .iter()
            .any(|instruction| {
                instruction
                    .outputs
                    .iter()
                    .any(|output| root(output.value) == Some(borrowed_root))
                    || instruction.resource_uses().any(|usage| {
                        usage.access.writes() && root(usage.value) == Some(borrowed_root)
                    })
            })
        {
            decline!("borrowed_modified_before_dispatch");
        }
    }
    for (position, instruction) in lowered.iter().enumerate() {
        let command_position = reverse[position];
        let alias_only =
            command_position.is_some_and(|command| aliases_only(&commands[command], instruction));
        for usage in instruction.resource_uses() {
            let usage_root = root(usage.value)?;
            if raw_roots.contains(&usage_root) || output_roots.contains(&usage_root) {
                detail!(
                    serde_json::json!({"event":"resource_use", "instruction":position,"command":command_position,"kind":instruction.kind,"value":usage.value.get(),"root":usage_root.get(),"access":format!("{:?}",usage.access),"aliasOnly":alias_only,"dispatchInstruction":dispatch_instruction})
                );
            }
            if usage.access.reads()
                && raw_roots.contains(&usage_root)
                && !alias_only
                && !command_position.is_some_and(|command| norm_commands.contains(&command))
            {
                decline!("raw_payload_observer");
            }
            if usage.access.reads()
                && output_roots.contains(&usage_root)
                && position < dispatch_instruction
                && !alias_only
            {
                decline!("final_output_observed_before_dispatch");
            }
            if usage.access.writes()
                && raw_roots.contains(&usage_root)
                && !alias_only
                && command_position != Some(packed_command)
            {
                decline!("raw_payload_writer");
            }
        }
        if position > mapping[packed_command]
            && instruction
                .outputs
                .iter()
                .any(|output| root(output.value).is_some_and(|id| raw_roots.contains(&id)))
            && !alias_only
            && !command_position
                .is_some_and(|command| matches!(commands[command].kind, CommandKind::Prepare))
        {
            decline!("raw_redefinition");
        }
    }
    // No final output may be changed by another payload writer. Publication
    // roots are allowed: the existing invocation fence completes all stages.
    for (position, instruction) in lowered.iter().enumerate() {
        if norm_commands
            .iter()
            .any(|&command| mapping[command] == position)
        {
            continue;
        }
        let alias_only =
            reverse[position].is_some_and(|command| aliases_only(&commands[command], instruction));
        let preparation = reverse[position]
            .is_some_and(|command| matches!(commands[command].kind, CommandKind::Prepare));
        if !alias_only
            && (instruction.resource_uses().any(|usage| {
                usage.access.writes()
                    && root(usage.value).is_some_and(|id| output_roots.contains(&id))
            }) || (!preparation
                && instruction
                    .outputs
                    .iter()
                    .any(|output| root(output.value).is_some_and(|id| output_roots.contains(&id)))))
        {
            decline!("final_output_payload_writer");
        }
    }
    Some(Group101 {
        packed_command,
        norm_commands,
        norm_instructions: norm_commands.map(|command| mapping[command]),
        dispatch_command,
        dispatch_instruction,
        prepare_instruction,
        packed_temporary: *temporary,
        borrowed,
        table_strides: found[0].3,
        outputs,
    })
}
#[cfg(test)]
#[path = "normrope101_admission_tests.rs"]
mod tests;

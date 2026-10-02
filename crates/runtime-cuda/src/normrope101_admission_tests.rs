use super::*;
use crate::cublas::Bf16GemmPlan;
use crate::executable::CudaKernelArgs;
use crate::workspace::CudaMemorySpace;
use effect_torch_compiler::{OutputDecl, ValueDecl, ValueUse};
use effect_torch_runtime::{InstructionId, SegmentOwnership, StorageMetadata};
fn id(value: usize) -> ValueId {
    ValueId::from_index(value).unwrap()
}
fn fixture(
    stateful: bool,
) -> (
    Vec<Command>,
    Vec<LoweredInstruction<&'static str>>,
    Vec<CudaValueMeta>,
    Vec<ValueId>,
    usize,
) {
    let shapes = [
        vec![256, 2816],
        vec![8192, 2816],
        vec![1],
        vec![256, 8192],
        vec![1, 256, 4096],
        vec![1, 256, 2048],
        vec![1, 256, 2048],
        vec![256],
        vec![256],
        vec![1, 256, 128],
        vec![1, 256, 128],
        vec![1, 16, 256, 256],
        vec![1, 8, 256, 256],
        vec![1, 8, 256, 256],
    ];
    let values = shapes
        .into_iter()
        .enumerate()
        .map(|(index, shape)| {
            let bytes = shape.iter().product::<usize>() * 2;
            CudaValueMeta {
                decl: ValueDecl::planned(
                    id(index),
                    format!("v{index}"),
                    bytes,
                    256,
                    CudaMemorySpace::Device,
                    SegmentOwnership::Workspace,
                ),
                shape,
                dtype: DType::BF16,
                storage: StorageMetadata::dense(),
            }
        })
        .collect::<Vec<_>>();
    let mut commands = Vec::new();
    let mut lowered = Vec::new();
    if stateful {
        lowered.push(LoweredInstruction::new(
            InstructionId::from_index(0).unwrap(),
            "state_prepare",
            vec![],
            vec![],
        ));
    }
    let mut append = |kind, output, name, reads: Vec<ValueId>, writes: Vec<ValueId>| {
        commands.push(Command {
            kind,
            output,
            overlap: CommandOverlap::Primary,
        });
        lowered.push(LoweredInstruction::new(
            InstructionId::from_index(lowered.len()).unwrap(),
            name,
            reads.into_iter().map(ValueUse::read).collect::<Vec<_>>(),
            writes.into_iter().map(OutputDecl::new).collect::<Vec<_>>(),
        ));
    };
    for binding in 0..4 {
        let value = id(7 + binding);
        append(
            CommandKind::Input { binding },
            Some(value),
            "input",
            vec![],
            vec![value],
        );
    }
    append(
        CommandKind::PackedProjection77 {
            x: id(0),
            weight: id(1),
            outputs: vec![id(4), id(5), id(6)],
            widths: vec![4096, 2048, 2048],
            temporary: id(3),
            workspace: id(2),
            plan: Bf16GemmPlan {
                m: 256,
                n: 8192,
                k: 2816,
                batch: 1,
                stride_x: 256 * 2816,
                stride_weight: 8192 * 2816,
                stride_out: 256 * 8192,
            },
            skip_split: false,
        },
        Some(id(4)),
        "packed_projection77",
        vec![id(0), id(1)],
        vec![id(4), id(5), id(6), id(3)],
    );
    for role in 0..3 {
        let heads = if role == 0 { 16 } else { 8 };
        let output = id(11 + role);
        append(CommandKind::Prepare, None, "prepare", vec![], vec![]);
        let mut args = CudaKernelArgs::default();
        args.elements = (256 * heads * 256) as u64;
        args.compute_dtype = 1;
        args.output_dtype = 3;
        args.input_dtypes[0] = 3;
        args.integers[0] = 256;
        args.integers[1] = 3;
        args.integers[3] = 1;
        args.integers[4] = heads as u64;
        args.integers[5] = 256;
        args.integers[6] = (heads * 256) as u64;
        args.integers[7] = 1;
        args.integers[8] = heads as u64;
        args.scalars[0] = 1e-6;
        let mut inputs = [None; 8];
        inputs[0] = Some(id(4 + role));
        let mut reads = vec![id(4 + role)];
        let mut metadata = vec![0; 9];
        let name = if role < 2 {
            args.operation = 3;
            args.integers[10] = heads as u64;
            args.integers[11] = 256;
            args.input_dtypes[1] = 3;
            inputs[1] = Some(id(7 + role));
            inputs[2] = Some(id(9));
            inputs[3] = Some(id(10));
            reads.extend([id(7 + role), id(9), id(10)]);
            metadata.extend([0, 128, 1, 0, 128, 0, 128, 1, 0, 128]);
            "et_norm_rope_bf16"
        } else {
            "et_rms_norm_f32"
        };
        append(
            CommandKind::Kernel {
                name,
                args,
                inputs,
                scratch: [None; 3],
                status: None,
                checked: false,
                metadata,
                state: StateAccess::None,
                state_buffers: [None; 4],
                kv_matmul: None,
            },
            Some(output),
            name,
            reads,
            vec![output],
        );
    }
    (commands, lowered, values, vec![id(11), id(12), id(13)], 4)
}
#[test]
fn normrope101_admission_preserves_separate_stateful_instruction_indices_and_final_roots() {
    for stateful in [false, true] {
        let (commands, lowered, values, roots, packed) = fixture(stateful);
        let group = admit(&commands, &lowered, &values, &roots, packed).unwrap();
        assert_eq!(group.norm_commands, [6, 8, 10]);
        assert_eq!(group.dispatch_command, 10);
        assert_eq!(group.dispatch_instruction, 10 + usize::from(stateful));
        assert_eq!(group.prepare_instruction, 9 + usize::from(stateful));
        assert_eq!(
            group.norm_instructions,
            [6, 8, 10].map(|position| position + usize::from(stateful))
        );
        assert_eq!(group.outputs, [id(11), id(12), id(13)]);
        assert_eq!(group.borrowed, [id(7), id(8), id(9), id(10)]);
    }
}
#[test]
fn normrope101_admission_accepts_private_raw_alias_definitions_and_rejects_payload_writes() {
    let (mut commands, mut lowered, mut values, roots, packed) = fixture(false);
    let alias = id(values.len());
    values.push(CudaValueMeta {
        decl: ValueDecl::alias(alias, "rawalias", id(4), 0, Q_BYTES),
        shape: vec![1, 16, 256, 256],
        dtype: DType::BF16,
        storage: StorageMetadata::dense(),
    });
    commands.insert(
        5,
        Command {
            kind: CommandKind::Alias { source: id(4) },
            output: Some(alias),
            overlap: CommandOverlap::Primary,
        },
    );
    lowered.insert(
        5,
        LoweredInstruction::new(
            InstructionId::from_index(5).unwrap(),
            "alias",
            vec![ValueUse::read(id(4))],
            vec![OutputDecl::new(alias)],
        ),
    );
    for (index, instruction) in lowered.iter_mut().enumerate() {
        instruction.id = InstructionId::from_index(index).unwrap();
    }
    if let CommandKind::Kernel { inputs, .. } = &mut commands[7].kind {
        inputs[0] = Some(alias);
    }
    lowered[7].inputs = vec![
        ValueUse::read(alias),
        ValueUse::read(id(7)),
        ValueUse::read(id(9)),
        ValueUse::read(id(10)),
    ]
    .into();
    assert!(admit(&commands, &lowered, &values, &roots, packed).is_some());
    lowered[6].inputs = vec![ValueUse::write(id(5))].into();
    assert!(admit(&commands, &lowered, &values, &roots, packed).is_none());
}
#[test]
fn normrope101_admission_rejects_borrowed_mutation_inside_moved_span() {
    let (commands, mut lowered, values, roots, packed) = fixture(false);
    lowered[7].inputs = vec![ValueUse::write(id(9))].into();
    assert!(admit(&commands, &lowered, &values, &roots, packed).is_none());
}
#[test]
fn normrope101_admission_preserves_early_final_alias_without_payload_observers() {
    let (mut commands, mut lowered, mut values, mut roots, packed) = fixture(false);
    let alias = id(values.len());
    values.push(CudaValueMeta {
        decl: ValueDecl::alias(alias, "queryalias", id(11), 0, Q_BYTES),
        shape: vec![1, 16, 256, 256],
        dtype: DType::BF16,
        storage: StorageMetadata::dense(),
    });
    commands.insert(
        7,
        Command {
            kind: CommandKind::Alias { source: id(11) },
            output: Some(alias),
            overlap: CommandOverlap::Primary,
        },
    );
    lowered.insert(
        7,
        LoweredInstruction::new(
            InstructionId::from_index(7).unwrap(),
            "alias",
            vec![ValueUse::read(id(11))],
            vec![OutputDecl::new(alias)],
        ),
    );
    for (index, instruction) in lowered.iter_mut().enumerate() {
        instruction.id = InstructionId::from_index(index).unwrap();
    }
    roots[0] = alias;
    assert!(admit(&commands, &lowered, &values, &roots, packed).is_some());
    lowered[8].inputs = vec![ValueUse::write(id(11))].into();
    assert!(admit(&commands, &lowered, &values, &roots, packed).is_none());
}
#[test]
fn normrope101_admission_rejects_escaping_raw_root_alias_and_external_payload_observer() {
    let (commands, lowered, mut values, roots, packed) = fixture(false);
    for raw in [id(4), id(5), id(6)] {
        let mut exposed = roots.clone();
        exposed.push(raw);
        assert!(admit(&commands, &lowered, &values, &exposed, packed).is_none());
    }
    let alias = id(values.len());
    values.push(CudaValueMeta {
        decl: ValueDecl::alias(alias, "rawalias", id(4), 0, Q_BYTES),
        shape: vec![1, 16, 256, 256],
        dtype: DType::BF16,
        storage: StorageMetadata::dense(),
    });
    let mut exposed = roots.clone();
    exposed.push(alias);
    assert!(admit(&commands, &lowered, &values, &exposed, packed).is_none());
    let mut observed = lowered.clone();
    observed[1].inputs = vec![ValueUse::read(id(5))].into();
    assert!(admit(&commands, &observed, &values, &roots, packed).is_none());
}
#[test]
fn normrope101_admission_rejects_early_final_consumer_and_late_borrowed_input() {
    let (commands, lowered, values, roots, packed) = fixture(false);
    let mut observed = lowered.clone();
    observed[7].inputs = vec![ValueUse::read(id(11))].into();
    assert!(admit(&commands, &observed, &values, &roots, packed).is_none());
    let mut late = lowered.clone();
    late[9].outputs = vec![OutputDecl::new(id(9))].into(); // A second tabledefinition invalidates immutable borrowing.
    let mut late_commands = commands;
    late_commands[2].output = None;
    late_commands[9].output = Some(id(9));
    assert!(admit(&late_commands, &late, &values, &roots, packed).is_none());
}
#[test]
fn normrope101_admission_requires_exact_source_table_dtype_geometry_and_prepare_mapping() {
    for mutation in 0..12 {
        let (mut commands, mut lowered, mut values, roots, packed) = fixture(false);
        match mutation {
            0 => {
                if let CommandKind::PackedProjection77 { widths, .. } = &mut commands[packed].kind {
                    widths[0] = 8192
                }
            }
            1 => {
                if let CommandKind::PackedProjection77 { plan, .. } = &mut commands[packed].kind {
                    plan.m = 64
                }
            }
            2 => {
                if let CommandKind::Kernel { args, .. } = &mut commands[6].kind {
                    args.integers[2] = 1
                }
            }
            3 => {
                if let CommandKind::Kernel { args, .. } = &mut commands[8].kind {
                    args.integers[8] = 16
                }
            }
            4 => {
                if let CommandKind::Kernel { metadata, .. } = &mut commands[6].kind {
                    metadata[9] = 128
                }
            }
            5 => values[9].dtype = DType::F32,
            6 => values[10].decl.bytes -= 2,
            7 => {
                if let CommandKind::Kernel { inputs, .. } = &mut commands[10].kind {
                    inputs[1] = Some(id(8))
                }
            }
            8 => {
                if let CommandKind::Kernel { args, .. } = &mut commands[10].kind {
                    args.scalars[0] = 1e-5
                }
            }
            9 => lowered[9].kind = "unexpectedprepare",
            10 => {
                commands[8].overlap = CommandOverlap::Worker {
                    branch: 0,
                    start: true,
                    finish: true,
                }
            }
            _ => {
                lowered.remove(0);
            }
        }
        assert!(
            admit(&commands, &lowered, &values, &roots, packed).is_none(),
            "mutation{mutation}"
        );
    }
}

#[test]
fn normrope101_admission_accepts_lazy_key_weight_but_rejects_changed_original_operands() {
    for stateful in [false, true] {
        let (mut commands, mut lowered, values, roots, mut packed) = fixture(stateful);
        let offset = usize::from(stateful);
        // Move KW's original input definition between Q and K normalizations.
        let command = commands.remove(1);
        let instruction = lowered.remove(1 + offset);
        commands.insert(6, command);
        lowered.insert(6 + offset, instruction);
        packed -= 1;
        for (position, instruction) in lowered.iter_mut().enumerate() {
            instruction.id = InstructionId::from_index(position).unwrap();
        }
        assert!(admit(&commands, &lowered, &values, &roots, packed).is_some());
        // A shared table changes after Q originally read it: forbidden even
        // though K would see the replacement under the original schedule.
        let mut changed = lowered.clone();
        changed[6 + offset].inputs = vec![ValueUse::write(id(9))].into();
        assert!(admit(&commands, &changed, &values, &roots, packed).is_none());
        // KW changes after its own original read: also forbidden.
        let mut changed = lowered.clone();
        changed[9 + offset].inputs = vec![ValueUse::write(id(8))].into();
        assert!(admit(&commands, &changed, &values, &roots, packed).is_none());
    }
}

#[test]
fn normrope101_diagnostics_preserve_admission_and_identify_table_layout() {
    let (mut commands, lowered, values, roots, packed) = fixture(false);
    let mut records = Vec::new();
    assert_eq!(
        admit(&commands, &lowered, &values, &roots, packed),
        admit_diagnostic(
            &commands,
            &lowered,
            &values,
            &roots,
            packed,
            Some(&mut records)
        )
    );
    if let CommandKind::Kernel { args, metadata, .. } = &mut commands[6].kind {
        args.operation = 0;
        let n = metadata.len();
        metadata[n - 10..].copy_from_slice(&[0, 256, 1, 0, 0, 0, 256, 1, 0, 0]);
    }
    records.clear();
    assert!(admit_diagnostic(
        &commands,
        &lowered,
        &values,
        &roots,
        packed,
        Some(&mut records)
    )
    .is_none());
    assert!(records
        .iter()
        .any(|record| record["reason"] == "norm_rope_name_operation_or_table_layout"));
    assert!(records
        .iter()
        .any(|record| record["reason"] == "missing_q_k_or_v_norm"));
}

#[test]
fn normrope101_real_dump_full_tables_aliases_and_rms_flags_require_provenance() {
    // Reproduce the actual 25-group model diagnostic: full131072B tables,
    // Q/K different zero-offset aliases, canonical metadata, and V flags3.
    for mutation in 0..7 {
        let (mut commands, mut lowered, mut values, roots, packed) = fixture(false);
        for table in [9, 10] {
            values[table].shape = vec![1, 1, 256, 256];
            values[table].decl.bytes = 131072;
        }
        for command in [6, 8] {
            if let CommandKind::Kernel { args, metadata, .. } = &mut commands[command].kind {
                args.operation = 0;
                args.integers[15] = 3;
                let n = metadata.len();
                metadata[n - 10..].copy_from_slice(&[0, 256, 1, 0, 0, 0, 256, 1, 0, 0]);
            }
        }
        if let CommandKind::Kernel { args, .. } = &mut commands[10].kind {
            args.operation = 3;
        }
        for table in [9, 10] {
            let alias = id(values.len());
            let mut meta = values[table].clone();
            meta.decl = ValueDecl::alias(alias, "rotary66_alias", id(table), 0, 131072);
            values.push(meta);
            if let CommandKind::Kernel { inputs, .. } = &mut commands[8].kind {
                inputs[table - 7] = Some(alias);
            }
        }
        lowered[8].inputs = vec![
            ValueUse::read(id(5)),
            ValueUse::read(id(8)),
            ValueUse::read(id(14)),
            ValueUse::read(id(15)),
        ]
        .into();
        match mutation {
            0 => {}
            1 => {
                if let CommandKind::Kernel { args, .. } = &mut commands[6].kind {
                    args.integers[15] = 0;
                }
            }
            2 => {
                if let CommandKind::Kernel { args, .. } = &mut commands[8].kind {
                    args.integers[15] = 1;
                }
            }
            3 => {
                values[14].decl.storage = ValueStorage::Alias {
                    source: id(9),
                    byte_offset: 16,
                }
            }
            4 => {
                values[14].decl.storage = ValueStorage::Alias {
                    source: id(10),
                    byte_offset: 0,
                }
            }
            5 => {
                if let CommandKind::Kernel { args, .. } = &mut commands[10].kind {
                    args.operation = 4;
                }
            }
            _ => values[9].decl.bytes = 65536,
        }
        let result = admit(&commands, &lowered, &values, &roots, packed);
        if mutation == 0 {
            assert_eq!(result.unwrap().table_strides, [256, 256]);
        } else {
            assert!(result.is_none(), "mutation{mutation}");
        }
    }
}

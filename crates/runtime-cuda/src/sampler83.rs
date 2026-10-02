//! Opt-in fusion of the private division/RNG80/entropy81 sampler pipeline.
//! No semantic matcher or RNG identity changes: only already-admitted kernels.
use crate::executable::{CudaKernelArgs, StateAccess};
use crate::lowering::{Command, CommandKind, CommandOverlap, CudaLoweredProgram};
use effect_torch_compiler::{InstructionEffects, OutputDecl, ValueStorage, ValueUse};
use effect_torch_runtime::{DType, StorageClass, StorageRepresentation, ValueId};

pub(crate) const KERNEL: &str = "et_random_sampler83_f32";
pub(crate) fn enabled() -> bool {
    std::env::var("EFFECT_TORCH_CUDA_SAMPLER83").as_deref() == Ok("1")
}

fn plain(command: &Command) -> bool {
    matches!(&command.kind, CommandKind::Kernel { status: None, checked: false,
        state: StateAccess::None, state_buffers, kv_matmul: None, .. }
        if state_buffers.iter().all(Option::is_none))
        && command.overlap == CommandOverlap::Primary
}

// Follow every alias transitively, including RNG80's dead uniform tombstone.
// Such aliases are permitted only if entirely unobservable.
fn private(program: &CudaLoweredProgram, value: ValueId, allowed: &[usize]) -> bool {
    if !matches!(
        program.values[value.index()].decl.storage,
        ValueStorage::Planned {
            class: StorageClass::Workspace,
            ..
        }
    ) {
        return false;
    }
    let mut family = vec![value];
    loop {
        let n = family.len();
        for v in &program.values {
            if matches!(v.decl.storage, ValueStorage::Alias {source,..} if family.contains(&source))
                && !family.contains(&v.decl.id)
            {
                family.push(v.decl.id);
            }
        }
        if n == family.len() {
            break;
        }
    }
    if program.outputs.iter().any(|v| family.contains(v)) {
        return false;
    }
    let definitions: Vec<_> = program
        .instructions
        .iter()
        .flat_map(|i| i.outputs.iter())
        .filter(|o| family.contains(&o.value))
        .collect();
    if definitions.len() != 1 || definitions[0].value != value {
        return false;
    }
    !program.instructions.iter().enumerate().any(|(i, ins)| {
        ins.inputs
            .iter()
            .chain(ins.scratch.iter())
            .chain(ins.staging.iter())
            .chain(ins.status.iter())
            .chain(ins.state.iter())
            .any(|u| family.contains(&u.value) && (u.value != value || !allowed.contains(&i)))
    })
}

pub(super) fn fuse(
    program: &mut CudaLoweredProgram,
    commands: &mut [Command],
) -> Result<usize, String> {
    let positions: Vec<_> = program
        .instructions
        .iter()
        .enumerate()
        .filter_map(|(i, x)| {
            (!matches!(
                x.kind,
                "state_prepare" | "state_commit" | "state_discard" | "status_check"
            ))
            .then_some(i)
        })
        .collect();
    if positions.len() != commands.len() {
        return Err("sampler83: command mapping mismatch".into());
    }
    let mut count = 0;
    for div in 0..commands.len() {
        if !plain(&commands[div]) {
            continue;
        }
        let CommandKind::Kernel {
            name: "et_div_feedback",
            args: division,
            inputs,
            scratch,
            metadata,
            ..
        } = &commands[div].kind
        else {
            continue;
        };
        let (Some(processed), Some(x), Some(temp), Some(feedback)) =
            (commands[div].output, inputs[0], inputs[1], scratch[0])
        else {
            continue;
        };
        if division.compute_dtype != 1
            || division.output_dtype != 1
            || division.input_dtypes[..2] != [1, 1]
            || division.operation != 3
            || division.integers[..2] != [0, 0]
            || division.integers[2] != 1
            || !matches!(division.integers[3], 2 | 3 | 4)
            || inputs[2..].iter().any(Option::is_some)
            || scratch[1..].iter().any(Option::is_some)
        {
            continue;
        }
        let find = |name, source| {
            commands
                .iter()
                .enumerate()
                .skip(div + 1)
                .find_map(|(i, c)| match &c.kind {
                    CommandKind::Kernel {
                        name: n, inputs, ..
                    } if *n == name && inputs[0] == Some(source) => Some(i),
                    _ => None,
                })
        };
        let (Some(rng), Some(max), Some(ent)) = (
            find(crate::rng_arg80::KERNEL, processed),
            find("et_entropy_max", processed),
            find(crate::entropy81::KERNEL, processed),
        ) else {
            continue;
        };
        if ![rng, max, ent].iter().all(|&i| plain(&commands[i])) {
            continue;
        }
        let CommandKind::Kernel {
            args: random,
            inputs: ri,
            scratch: rs,
            ..
        } = &commands[rng].kind
        else {
            unreachable!()
        };
        let CommandKind::Kernel {
            args: ma,
            inputs: mi,
            scratch: ms,
            ..
        } = &commands[max].kind
        else {
            unreachable!()
        };
        let CommandKind::Kernel {
            args: ea,
            inputs: ei,
            scratch: es,
            ..
        } = &commands[ent].kind
        else {
            unreachable!()
        };
        let (Some(arg), Some(maximum), Some(entropy)) = (
            commands[rng].output,
            commands[max].output,
            commands[ent].output,
        ) else {
            continue;
        };
        let (rows, width) = (random.integers[2], random.integers[1]);
        let Some(elements) = rows.checked_mul(width) else {
            continue;
        };
        if !(1..=65535).contains(&rows)
            || !(4096..=i32::MAX as u64).contains(&width)
            || elements > u32::MAX as u64
            || division.elements != elements
            || random.elements != 2 * rows
            || random.compute_dtype != 1
            || random.output_dtype != 4
            || random.input_dtypes[0] != 1
            || ri[1..].iter().any(Option::is_some)
            || ma.integers[..2] != [width, rows]
            || ea.integers[..2] != [width, rows]
            || ma.compute_dtype != 1
            || ea.compute_dtype != 1
            || ma.output_dtype != 1
            || ea.output_dtype != 1
            || ma.input_dtypes[0] != 1
            || ea.input_dtypes[..2] != [1, 1]
            || mi[1..].iter().any(Option::is_some)
            || ei[1] != Some(maximum)
            || ei[2..].iter().any(Option::is_some)
            || rs.iter().chain(ms).chain(es).any(Option::is_some)
            || (division.integers[3] == 3 && division.integers[5] != width)
            || (division.integers[3] == 4
                && (division.integers[5] >= 32 || (1u64 << division.integers[5]) != width))
        {
            continue;
        }
        let per_row = matches!(division.integers[3], 3 | 4);
        let expected = [
            (x, DType::F32, elements * 4),
            (temp, DType::F32, if per_row { rows * 4 } else { 4 }),
            (processed, DType::F32, elements * 4),
            (feedback, DType::BF16, elements * 2),
            (arg, DType::I64, rows * 16),
            (maximum, DType::F32, rows * 4),
            (entropy, DType::F32, rows * 4),
        ];
        if expected.iter().any(|&(v, d, b)| {
            let m = &program.values[v.index()];
            m.dtype != d
                || m.decl.bytes as u64 != b
                || m.storage.representation != StorageRepresentation::Dense
        }) || [x, temp, processed, feedback, arg, maximum, entropy]
            .iter()
            .enumerate()
            .any(|(i, v)| [x, temp, processed, feedback, arg, maximum, entropy][..i].contains(v))
            || !private(
                program,
                processed,
                &[positions[rng], positions[max], positions[ent]],
            )
            || !private(program, maximum, &[positions[ent]])
        {
            continue;
        }
        if [(feedback,2),(arg,8),(entropy,4)].iter().any(|&(v,a)|
            !matches!(program.values[v.index()].decl.storage,ValueStorage::Planned{alignment,..} if alignment>=a)) {continue;}
        if [div, rng, max, ent].iter().any(|&i| {
            let ins = &program.instructions[positions[i]];
            !ins.scratch.is_empty()
                || !ins.staging.is_empty()
                || !ins.status.is_empty()
                || !ins.state.is_empty()
        }) {
            continue;
        }
        if [
            (div, vec![processed, feedback]),
            (rng, vec![arg]),
            (max, vec![maximum]),
            (ent, vec![entropy]),
        ]
        .iter()
        .any(|(i, expected)| {
            program.instructions[positions[*i]]
                .outputs
                .iter()
                .map(|o| o.value)
                .collect::<Vec<_>>()
                != *expected
        }) {
            continue;
        }
        let metadata = metadata.clone();
        // Move outputs to the original division point: its two operands are
        // already available, and every later observer retains its original order.
        let args = CudaKernelArgs {
            elements: elements,
            compute_dtype: 1,
            output_dtype: 3,
            input_dtypes: [1, 1, 0, 0, 0, 0, 0, 0],
            integers: {
                let mut a = [0; 16];
                a[0] = random.integers[0];
                a[1] = width;
                a[2] = rows;
                a[3] = u64::from(per_row);
                a
            },
            ..Default::default()
        };
        commands[div].output = Some(feedback);
        commands[div].kind = CommandKind::Kernel {
            name: KERNEL,
            args,
            inputs: [Some(x), Some(temp), None, None, None, None, None, None],
            scratch: [Some(arg), Some(entropy), None],
            status: None,
            checked: false,
            metadata,
            state: StateAccess::None,
            state_buffers: [None; 4],
            kv_matmul: None,
        };
        let mut effects = InstructionEffects::default();
        for i in [div, rng, max, ent] {
            effects.may_fail |= program.instructions[positions[i]].effects.may_fail;
            effects.has_side_effects |= program.instructions[positions[i]].effects.has_side_effects;
        }
        let ins = &mut program.instructions[positions[div]];
        ins.kind = KERNEL;
        ins.inputs = vec![ValueUse::read(x), ValueUse::read(temp)].into_boxed_slice();
        ins.outputs = vec![
            OutputDecl::new(feedback),
            OutputDecl::new(arg),
            OutputDecl::new(entropy),
        ]
        .into_boxed_slice();
        ins.effects = effects;
        for i in [rng, max, ent] {
            commands[i].output = None;
            commands[i].kind = CommandKind::Prepare;
            let ins = &mut program.instructions[positions[i]];
            ins.kind = "sampler83_absorbed";
            ins.inputs = Box::new([]);
            ins.outputs = Box::new([]);
            ins.scratch = Box::new([]);
            ins.staging = Box::new([]);
            ins.status = Box::new([]);
            ins.state = Box::new([]);
            ins.effects = InstructionEffects::default();
        }
        // Scratch outputs are planned writes; retain publication markers so
        // ordinary downstream commands and escaping roots acquire their owners.
        for (i, output) in [(rng, arg), (ent, entropy)] {
            commands[i].output = Some(output);
            commands[i].kind = CommandKind::PlannedAlias;
            let ins = &mut program.instructions[positions[i]];
            ins.kind = "sampler83_output";
            ins.inputs = vec![ValueUse::read(output)].into_boxed_slice();
        }
        program.values[processed.index()].decl.storage = ValueStorage::Alias {
            source: x,
            byte_offset: 0,
        };
        program.values[maximum.index()].decl.storage = ValueStorage::Alias {
            source: entropy,
            byte_offset: 0,
        };
        count += 1;
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lowering::CudaValueMeta;
    use crate::workspace::CudaMemorySpace;
    use effect_torch_compiler::{
        analyze_liveness, plan_memory, LoweredInstruction, LoweredProgram, MemoryPlannerConfig,
        ValueDecl,
    };
    use effect_torch_runtime::{InstructionId, SegmentOwnership, StorageMetadata};
    fn v(i: u32) -> ValueId {
        ValueId::new(i)
    }
    fn fixture(per_row: bool) -> (CudaLoweredProgram, Vec<Command>) {
        let values = [
            (DType::F32, 12288),
            (DType::F32, if per_row { 3 } else { 1 }),
            (DType::F32, 12288),
            (DType::BF16, 12288),
            (DType::I64, 6),
            (DType::F32, 3),
            (DType::F32, 3),
            (DType::F32, 12288),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, (dtype, n))| CudaValueMeta {
            decl: ValueDecl::planned(
                v(i as u32),
                format!("v{i}"),
                n * dtype.size_in_bytes(),
                256,
                CudaMemorySpace::Device,
                SegmentOwnership::Workspace,
            ),
            shape: vec![n],
            dtype,
            storage: StorageMetadata::dense(),
        })
        .collect::<Vec<_>>();
        let mut program = LoweredProgram::new(values, vec![], vec![v(3), v(4), v(6)]);
        program.values[7].decl.storage = ValueStorage::Alias {
            source: v(2),
            byte_offset: 0,
        };
        let mut instructions = vec![];
        let mut commands = vec![Command {
            output: None,
            kind: CommandKind::Prepare,
            overlap: CommandOverlap::Primary,
        }];
        instructions.push(LoweredInstruction::new(
            InstructionId::new(0),
            "input",
            vec![],
            vec![OutputDecl::new(v(0)), OutputDecl::new(v(1))],
        ));
        let mut add = |name, args, inputs: Vec<ValueId>, outputs: Vec<ValueId>| {
            let i = commands.len();
            let mut ins = [None; 8];
            for (j, &x) in inputs.iter().enumerate() {
                ins[j] = Some(x);
            }
            let mut scratch = [None; 3];
            for (j, &x) in outputs.iter().skip(1).enumerate() {
                scratch[j] = Some(x);
            }
            commands.push(Command {
                output: Some(outputs[0]),
                kind: CommandKind::Kernel {
                    name,
                    args,
                    inputs: ins,
                    scratch,
                    status: None,
                    checked: false,
                    metadata: vec![0, 1, 2],
                    state: StateAccess::None,
                    state_buffers: [None; 4],
                    kv_matmul: None,
                },
                overlap: CommandOverlap::Primary,
            });
            instructions.push(LoweredInstruction::new(
                InstructionId::new(i as u32),
                name,
                inputs.into_iter().map(ValueUse::read).collect::<Vec<_>>(),
                outputs.into_iter().map(OutputDecl::new).collect::<Vec<_>>(),
            ));
        };
        let mut div = CudaKernelArgs {
            elements: 12288,
            compute_dtype: 1,
            output_dtype: 1,
            operation: 3,
            ..Default::default()
        };
        div.input_dtypes[..2].copy_from_slice(&[1, 1]);
        div.integers[2] = 1;
        div.integers[3] = if per_row { 3 } else { 2 };
        div.integers[5] = 4096;
        add("et_div_feedback", div, vec![v(0), v(1)], vec![v(2), v(3)]);
        let mut rng = CudaKernelArgs {
            elements: 6,
            compute_dtype: 1,
            output_dtype: 4,
            ..Default::default()
        };
        rng.input_dtypes[0] = 1;
        rng.integers[..3].copy_from_slice(&[u64::MAX - 15, 4096, 3]);
        add(crate::rng_arg80::KERNEL, rng, vec![v(2)], vec![v(4)]);
        let mut ent = CudaKernelArgs {
            elements: 3,
            compute_dtype: 1,
            output_dtype: 1,
            ..Default::default()
        };
        ent.input_dtypes[..2].copy_from_slice(&[1, 1]);
        ent.integers[..2].copy_from_slice(&[4096, 3]);
        add("et_entropy_max", ent, vec![v(2)], vec![v(5)]);
        add(crate::entropy81::KERNEL, ent, vec![v(2), v(5)], vec![v(6)]);
        program.instructions = instructions.into_boxed_slice();
        (program, commands)
    }
    #[test]
    fn sampler83_private_pipeline_preserves_provenance_and_three_owned_outputs() {
        for mode in [2, 3, 4] {
            let per_row = mode != 2;
            let (mut p, mut c) = fixture(per_row);
            if let CommandKind::Kernel { args, .. } = &mut c[1].kind {
                args.integers[3] = mode;
                args.integers[5] = if mode == 4 { 12 } else { 4096 };
            }
            assert_eq!(fuse(&mut p, &mut c).unwrap(), 1);
            let CommandKind::Kernel {
                name,
                args,
                scratch,
                ..
            } = &c[1].kind
            else {
                panic!()
            };
            assert_eq!(*name, KERNEL);
            assert_eq!(
                args.integers[..4],
                [u64::MAX - 15, 4096, 3, u64::from(per_row)]
            );
            assert_eq!(*scratch, [Some(v(4)), Some(v(6)), None]);
            assert_eq!(p.instructions[1].outputs.len(), 3);
            analyze_liveness(&p).unwrap();
            let _ = plan_memory(
                &p,
                &MemoryPlannerConfig::uniform(CudaMemorySpace::Device, 1 << 30, 256, 256),
            )
            .unwrap();
            assert_eq!(fuse(&mut p, &mut c).unwrap(), 0);
        }
    }
    #[test]
    fn sampler83_rejects_escape_alias_observers_and_unsupported_descriptors() {
        for case in 0..17 {
            let (mut p, mut c) = fixture(true);
            match case {
                0 => p.outputs = vec![v(2)].into_boxed_slice(),
                1 => p.outputs = vec![v(5)].into_boxed_slice(),
                2 => p.outputs = vec![v(7)].into_boxed_slice(),
                3 => p.instructions[0].inputs = vec![ValueUse::read(v(7))].into_boxed_slice(),
                4 => {
                    if let CommandKind::Kernel { args, .. } = &mut c[1].kind {
                        args.integers[0] = 1;
                    }
                }
                5 => p.values[1].decl.bytes = 4,
                6 => {
                    if let CommandKind::Kernel { args, .. } = &mut c[1].kind {
                        args.integers[5] = 2048;
                    }
                }
                7 => {
                    if let CommandKind::Kernel { args, .. } = &mut c[1].kind {
                        args.integers[3] = 4;
                    }
                }
                8 => {
                    if let CommandKind::Kernel { args, .. } = &mut c[2].kind {
                        args.integers[1] = i32::MAX as u64 + 1;
                    }
                }
                9 => {
                    if let CommandKind::Kernel { name, .. } = &mut c[4].kind {
                        *name = "et_entropy_finish";
                    }
                }
                10 => {
                    if let CommandKind::Kernel { inputs, .. } = &mut c[4].kind {
                        inputs[1] = Some(v(0));
                    }
                }
                11 => {
                    if let CommandKind::Kernel { checked, .. } = &mut c[1].kind {
                        *checked = true;
                    }
                }
                12 => {
                    if let ValueStorage::Planned { alignment, .. } = &mut p.values[4].decl.storage {
                        *alignment = 4;
                    }
                }
                13 => {
                    p.values[6].decl.storage = ValueStorage::Alias {
                        source: v(5),
                        byte_offset: 0,
                    }
                }
                14 => p.instructions[0].inputs = vec![ValueUse::read(v(5))].into_boxed_slice(),
                15 => p.instructions[0].inputs = vec![ValueUse::read(v(2))].into_boxed_slice(),
                16 => p.instructions[0].outputs = vec![OutputDecl::new(v(7))].into_boxed_slice(),
                _ => unreachable!(),
            }
            assert_eq!(fuse(&mut p, &mut c).unwrap(), 0, "case {case}");
        }
    }
}

#[cfg(test)]
fn entropy_graph83(
    source: std::sync::Arc<effect_torch_graph::Node>,
    rows: usize,
    width: usize,
) -> std::sync::Arc<effect_torch_graph::Node> {
    use effect_torch_graph::{Device, Node, NodeKind};
    let dtype = DType::F32;
    let minimum = -(f32::MAX as f64);
    let make = |kind| Node::new(kind).unwrap();
    let maximum = make(NodeKind::Max {
        a: source.clone(),
        dims: vec![1],
        keepdims: true,
    });
    let shifted = make(NodeKind::Sub {
        a: source.clone(),
        b: maximum.clone(),
    });
    let exponential = make(NodeKind::Exp { a: shifted });
    let sum = make(NodeKind::Sum {
        a: exponential,
        dims: vec![1],
        keepdims: true,
    });
    let logarithm = make(NodeKind::Log { a: sum });
    let lse = make(NodeKind::Add {
        a: maximum,
        b: logarithm,
    });
    let normalized = make(NodeKind::Sub { a: source, b: lse });
    let minimum = make(NodeKind::Full {
        shape: vec![rows, width],
        value: minimum,
        dtype,
        device: Device::Cuda(0),
    });
    let clamped = make(NodeKind::Maximum {
        a: normalized.clone(),
        b: minimum,
    });
    let maximum = make(NodeKind::Max {
        a: normalized.clone(),
        dims: vec![1],
        keepdims: true,
    });
    let shifted = make(NodeKind::Sub {
        a: normalized,
        b: maximum,
    });
    let exponential = make(NodeKind::Exp { a: shifted });
    let denominator = make(NodeKind::Sum {
        a: exponential.clone(),
        dims: vec![1],
        keepdims: true,
    });
    let probability = make(NodeKind::Div {
        a: exponential,
        b: denominator,
    });
    let product = make(NodeKind::Mul {
        a: clamped,
        b: probability,
    });
    let sum = make(NodeKind::Sum {
        a: product,
        dims: vec![1],
        keepdims: false,
    });
    make(NodeKind::Neg { a: sum })
}

#[cfg(test)]
#[test]
#[ignore = "requires CUDA and sampler83, div-feedback, RNG80, entropy-recompute, relaxed-entropy81 flags"]
fn sampler83_hardware_matches_four_kernels_repeated_runs_and_retained_outputs() {
    use effect_torch_graph::{Device, Node, NodeKind};
    use effect_torch_runtime::{CancellationFlag, StorageMetadata};
    assert!(enabled() && crate::rng_arg80::enabled() && crate::entropy81::enabled());
    let device = crate::CudaDevice::get(0).unwrap();
    for width in [4096, 4097, 262144] {
        for per_row in [false, true] {
            let rows = 3;
            let shape = vec![rows, width];
            let input = |slot, shape| {
                Node::new(NodeKind::Input {
                    slot,
                    shape,
                    dtype: DType::F32,
                    device: Device::Cuda(0),
                    storage: StorageMetadata::dense(),
                })
                .unwrap()
            };
            let temp_shape = if per_row { vec![rows, 1] } else { vec![] };
            let processed = Node::new(NodeKind::Div {
                a: input(0, shape.clone()),
                b: input(1, temp_shape.clone()),
            })
            .unwrap();
            let feedback = Node::new(NodeKind::Cast {
                a: processed.clone(),
                dtype: DType::BF16,
            })
            .unwrap();
            let uniform = Node::new(NodeKind::Uniform {
                lo: 0.,
                hi: 1.,
                shape: shape.clone(),
                dtype: DType::F32,
                device: Device::Cuda(0),
            })
            .unwrap();
            let log = Node::new(NodeKind::Log { a: uniform }).unwrap();
            let neg = Node::new(NodeKind::Neg { a: log }).unwrap();
            let log = Node::new(NodeKind::Log { a: neg }).unwrap();
            let noise = Node::new(NodeKind::Neg { a: log }).unwrap();
            let sampled = Node::new(NodeKind::Argmax {
                a: Node::new(NodeKind::Add {
                    a: processed.clone(),
                    b: noise,
                })
                .unwrap(),
                dim: 1,
            })
            .unwrap();
            let plain = Node::new(NodeKind::Argmax {
                a: processed.clone(),
                dim: 1,
            })
            .unwrap();
            let entropy = entropy_graph83(processed.clone(), rows, width);
            let roots = vec![feedback, plain, sampled, entropy];
            let options = || effect_torch_compiler::CompileOptions {
                random_seed: Some(u64::MAX - 15),
                ..Default::default()
            };
            let fused = crate::compile_with_options(roots.clone(), 0, options()).unwrap();
            let mut reference_roots = roots;
            reference_roots.push(processed);
            let reference = crate::compile_with_options(reference_roots, 0, options()).unwrap();
            assert!(
                fused
                    .diagnostics()
                    .instructions
                    .iter()
                    .any(|x| x.kind == KERNEL),
                "width {width} per_row {per_row}"
            );
            assert!(!reference
                .diagnostics()
                .instructions
                .iter()
                .any(|x| x.kind == KERNEL));
            for kind in [
                "et_div_feedback",
                crate::rng_arg80::KERNEL,
                crate::entropy81::KERNEL,
            ] {
                assert!(
                    reference
                        .diagnostics()
                        .instructions
                        .iter()
                        .any(|x| x.kind == kind),
                    "reference lacks {kind}"
                );
            }
            let mut retained = Vec::new();
            for mode in 0..7 {
                let xs = (0..rows * width)
                    .map(|i| match mode {
                        1 if i % width == 0 => f64::NAN,
                        2 if i % width == 7 => f64::NAN,
                        3 => {
                            if i % 2 == 0 {
                                f64::NEG_INFINITY
                            } else {
                                f64::INFINITY
                            }
                        }
                        4 => {
                            if i % 2 == 0 {
                                -0.0
                            } else {
                                0.0
                            }
                        }
                        5 => 3.25,
                        6 => f64::NEG_INFINITY,
                        _ => ((i * 173) % 103) as f64 / 19. - 2.7,
                    })
                    .collect::<Vec<_>>();
                let temps = if per_row {
                    vec![0.5, 1., 1.75]
                } else {
                    vec![0.875]
                };
                let bindings = vec![
                    crate::CudaValue::from_host(device.clone(), shape.clone(), DType::F32, &xs)
                        .unwrap(),
                    crate::CudaValue::from_host(
                        device.clone(),
                        temp_shape.clone(),
                        DType::F32,
                        &temps,
                    )
                    .unwrap(),
                ];
                if mode == 4 {
                    let cancelled = CancellationFlag::new();
                    cancelled.cancel();
                    assert_eq!(
                        fused
                            .execute(&bindings, &[], &cancelled)
                            .err()
                            .expect("fused pre-cancel"),
                        reference
                            .execute(&bindings, &[], &cancelled)
                            .err()
                            .expect("reference pre-cancel")
                    );
                }
                let actual = fused
                    .execute(&bindings, &[], &CancellationFlag::new())
                    .unwrap();
                let expected = reference
                    .execute(&bindings, &[], &CancellationFlag::new())
                    .unwrap();
                let bytes = actual
                    .iter()
                    .map(|v| v.read_storage_bytes().unwrap())
                    .collect::<Vec<_>>();
                for i in 0..3 {
                    assert_eq!(
                        bytes[i],
                        expected[i].read_storage_bytes().unwrap(),
                        "output {i} mode {mode}"
                    );
                }
                let e = expected[3].read_storage_bytes().unwrap();
                for (a, b) in bytes[3].chunks_exact(4).zip(e.chunks_exact(4)) {
                    let a = f32::from_ne_bytes(a.try_into().unwrap());
                    let b = f32::from_ne_bytes(b.try_into().unwrap());
                    assert!(
                        a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan(),
                        "entropy {a} != {b}"
                    );
                }
                retained.push((actual, bytes));
            }
            drop(fused);
            drop(reference);
            for (outputs, bytes) in retained {
                for (v, b) in outputs.iter().zip(bytes) {
                    assert_eq!(v.read_storage_bytes().unwrap(), b);
                }
            }
        }
    }
}

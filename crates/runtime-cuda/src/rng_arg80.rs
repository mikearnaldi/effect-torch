//! Absorb a private, already-legalized uniform into the existing dual argmax.
//! The original provenance and per-invocation seed arithmetic are unchanged.
use crate::executable::StateAccess;
use crate::lowering::{Command, CommandKind, CommandOverlap, CudaLoweredProgram};
use effect_torch_compiler::{InstructionEffects, ValueStorage};
use effect_torch_runtime::{DType, StorageClass, StorageRepresentation};

pub(crate) fn enabled() -> bool {
    std::env::var("EFFECT_TORCH_CUDA_RNG_ARG80").as_deref() == Ok("1")
}
pub(crate) const KERNEL: &str = "et_random_dual_arg80_f32";

pub(super) fn fuse(
    program: &mut CudaLoweredProgram,
    commands: &mut [Command],
) -> Result<usize, String> {
    let positions: Vec<_> = program
        .instructions
        .iter()
        .enumerate()
        .filter_map(|(i, instruction)| {
            (!matches!(
                instruction.kind,
                "state_prepare" | "state_commit" | "state_discard" | "status_check"
            ))
            .then_some(i)
        })
        .collect();
    if positions.len() != commands.len() {
        return Err("compile: RNG80 command/instruction mapping mismatch".into());
    }
    let mut fused = 0;
    for target in 0..commands.len() {
        let CommandKind::Kernel {
            name: "et_dual_argmax",
            args,
            inputs,
            scratch,
            status: None,
            checked: false,
            state: StateAccess::None,
            state_buffers,
            kv_matmul: None,
            ..
        } = &commands[target].kind
        else {
            continue;
        };
        if commands[target].overlap != CommandOverlap::Primary
            || scratch.iter().any(Option::is_some)
            || state_buffers.iter().any(Option::is_some)
            || inputs[2..].iter().any(Option::is_some)
            || args.compute_dtype != 1
            || args.output_dtype != 4
            || args.input_dtypes[..2] != [1, 1]
            || args.integers[0] == 0
            || args.integers[0] > 65535
            || !(4096..=u32::MAX as u64).contains(&args.integers[1])
            || args.elements != args.integers[0] * 2
        {
            continue;
        }
        let (Some(logits), Some(uniform)) = (inputs[0], inputs[1]) else {
            continue;
        };
        if logits == uniform {
            continue;
        }
        let Some(elements) = args.integers[0]
            .checked_mul(args.integers[1])
            .filter(|n| *n <= u32::MAX as u64)
        else {
            continue;
        };
        let u = &program.values[uniform.index()];
        let x = &program.values[logits.index()];
        if u.dtype != DType::F32
            || x.dtype != DType::F32
            || u.shape != x.shape
            || u.decl.bytes != elements as usize * 4
            || x.decl.bytes != u.decl.bytes
            || u.storage.representation != StorageRepresentation::Dense
            || x.storage.representation != StorageRepresentation::Dense
            || !matches!(
                u.decl.storage,
                ValueStorage::Planned {
                    class: StorageClass::Workspace,
                    ..
                }
            )
            || program.outputs.contains(&uniform)
            || program.values.iter().any(
                |v| matches!(v.decl.storage,ValueStorage::Alias{source,..} if source == uniform),
            )
        {
            continue;
        }
        // No read, write, staging, state or output escape may observe the omitted buffer.
        if program
            .instructions
            .iter()
            .enumerate()
            .any(|(i, instruction)| {
                i != positions[target]
                    && instruction
                        .inputs
                        .iter()
                        .chain(instruction.scratch.iter())
                        .chain(instruction.staging.iter())
                        .chain(instruction.status.iter())
                        .chain(instruction.state.iter())
                        .any(|v| v.value == uniform)
            })
        {
            continue;
        }
        let Some(producer) = commands[..target]
            .iter()
            .position(|command| command.output == Some(uniform))
        else {
            continue;
        };
        let CommandKind::Kernel {
            name: "et_random_f32",
            args: random,
            inputs: random_inputs,
            scratch: random_scratch,
            status: None,
            checked: false,
            state: StateAccess::None,
            state_buffers: random_state,
            kv_matmul: None,
            ..
        } = &commands[producer].kind
        else {
            continue;
        };
        if commands[producer].overlap != CommandOverlap::Primary
            || random.elements != elements
            || random.compute_dtype != 1
            || random.output_dtype != 1
            || random.integers[1] != 0
            || random.scalars[0].to_bits() != 0f64.to_bits()
            || random.scalars[1].to_bits() != 1f64.to_bits()
            || random_inputs
                .iter()
                .chain(random_scratch)
                .chain(random_state)
                .any(Option::is_some)
        {
            continue;
        }
        let provenance = random.integers[0];
        let CommandKind::Kernel {
            name, args, inputs, ..
        } = &mut commands[target].kind
        else {
            unreachable!()
        };
        *name = KERNEL;
        args.integers[2] = args.integers[0];
        args.integers[0] = provenance;
        inputs[1] = None;
        args.input_dtypes[1] = 0;
        let instruction = &mut program.instructions[positions[target]];
        instruction.kind = KERNEL;
        instruction.inputs = instruction
            .inputs
            .iter()
            .filter(|v| v.value != uniform)
            .cloned()
            .collect();
        // The random source identity is already fixed by semantic lowering; RNG
        // is counter-based, not a mutable stream advanced by submission order.
        let instruction = &mut program.instructions[positions[producer]];
        instruction.kind = "rng_arg80_source";
        instruction.outputs = Box::new([]);
        instruction.inputs = Box::new([]);
        instruction.effects = InstructionEffects::default();
        // The planner requires a definition for every Planned value, including
        // dead ones. This proven-unobservable tombstone aliases the equal-size
        // logits only to remove its allocation; no command reads this alias.
        program.values[uniform.index()].decl.storage = ValueStorage::Alias {
            source: logits,
            byte_offset: 0,
        };
        commands[producer].output = None;
        commands[producer].kind = CommandKind::Prepare;
        fused += 1;
    }
    Ok(fused)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executable::CudaKernelArgs;
    use crate::lowering::CudaValueMeta;
    use crate::workspace::CudaMemorySpace;
    use effect_torch_compiler::{
        analyze_liveness, plan_memory, LoweredInstruction, LoweredProgram, MemoryPlannerConfig,
        OutputDecl, ValueDecl, ValueUse,
    };
    use effect_torch_runtime::{InstructionId, SegmentOwnership, StorageMetadata, ValueId};
    fn fixture() -> (CudaLoweredProgram, Vec<Command>) {
        let values: Vec<_> = [
            (DType::F32, vec![3, 4096]),
            (DType::F32, vec![3, 4096]),
            (DType::I64, vec![2, 3]),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, (dtype, shape))| CudaValueMeta {
            decl: ValueDecl::planned(
                ValueId::new(i as u32),
                format!("v{i}"),
                shape.iter().product::<usize>() * dtype.size_in_bytes(),
                256,
                CudaMemorySpace::Device,
                SegmentOwnership::Workspace,
            ),
            shape,
            dtype,
            storage: StorageMetadata::dense(),
        })
        .collect();
        let kernel = |name, args, inputs| CommandKind::Kernel {
            name,
            args,
            inputs,
            scratch: [None; 3],
            status: None,
            checked: false,
            metadata: Vec::new(),
            state: StateAccess::None,
            state_buffers: [None; 4],
            kv_matmul: None,
        };
        let mut random = CudaKernelArgs {
            elements: 3 * 4096,
            compute_dtype: 1,
            output_dtype: 1,
            ..Default::default()
        };
        random.integers[0] = 91;
        random.scalars[1] = 1.;
        let mut dual = CudaKernelArgs {
            elements: 6,
            compute_dtype: 1,
            output_dtype: 4,
            ..Default::default()
        };
        dual.integers[..2].copy_from_slice(&[3, 4096]);
        dual.input_dtypes[..2].copy_from_slice(&[1, 1]);
        let commands = vec![
            Command {
                output: Some(ValueId::new(0)),
                kind: CommandKind::Prepare,
                overlap: CommandOverlap::Primary,
            },
            Command {
                output: Some(ValueId::new(1)),
                kind: kernel("et_random_f32", random, [None; 8]),
                overlap: CommandOverlap::Primary,
            },
            Command {
                output: Some(ValueId::new(2)),
                kind: kernel(
                    "et_dual_argmax",
                    dual,
                    [
                        Some(ValueId::new(0)),
                        Some(ValueId::new(1)),
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ],
                ),
                overlap: CommandOverlap::Primary,
            },
        ];
        let instructions = vec![
            LoweredInstruction::new(
                InstructionId::new(0),
                "input",
                vec![],
                vec![OutputDecl::new(ValueId::new(0))],
            ),
            LoweredInstruction::new(
                InstructionId::new(1),
                "et_random_f32",
                vec![],
                vec![OutputDecl::new(ValueId::new(1))],
            ),
            LoweredInstruction::new(
                InstructionId::new(2),
                "et_dual_argmax",
                vec![
                    ValueUse::read(ValueId::new(0)),
                    ValueUse::read(ValueId::new(1)),
                ],
                vec![OutputDecl::new(ValueId::new(2))],
            ),
        ];
        (
            LoweredProgram::new(values, instructions, vec![ValueId::new(2)]),
            commands,
        )
    }
    #[test]
    fn private_uniform_removed_provenance_preserved_and_memory_plan_valid() {
        let (mut p, mut c) = fixture();
        assert_eq!(fuse(&mut p, &mut c).unwrap(), 1);
        let CommandKind::Kernel {
            name, args, inputs, ..
        } = &c[2].kind
        else {
            panic!("not kernel")
        };
        assert_eq!(*name, KERNEL);
        assert_eq!(args.integers[..3], [91, 4096, 3]);
        assert_eq!(inputs[1], None);
        assert!(p.instructions[1].outputs.is_empty());
        assert!(matches!(c[1].kind, CommandKind::Prepare));
        analyze_liveness(&p).unwrap();
        let memory = plan_memory(
            &p,
            &MemoryPlannerConfig::uniform(CudaMemorySpace::Device, 1 << 20, 256, 256),
        )
        .unwrap();
        assert_eq!(
            memory.locations[1],
            effect_torch_runtime::Location::Alias {
                root: ValueId::new(0),
                byte_offset: 0
            }
        );
        assert_ne!(memory.locations[0], memory.locations[2]);
        assert_eq!(fuse(&mut p, &mut c).unwrap(), 0);
    }
    #[test]
    fn sharing_escapes_aliases_dtype_distribution_and_geometry_reject() {
        for case in 0..13 {
            let (mut p, mut c) = fixture();
            match case {
                0 => p.outputs = vec![ValueId::new(2), ValueId::new(1)].into_boxed_slice(),
                1 => {
                    p.instructions[0].inputs =
                        vec![ValueUse::read(ValueId::new(1))].into_boxed_slice()
                }
                2 => {
                    p.values[2].decl.storage = ValueStorage::Alias {
                        source: ValueId::new(1),
                        byte_offset: 0,
                    }
                }
                3 => p.values[1].dtype = DType::F16,
                4 => p.values[1].shape = vec![4096, 3],
                5 => {
                    if let CommandKind::Kernel { args, .. } = &mut c[1].kind {
                        args.integers[1] = 1
                    }
                }
                6 => {
                    if let CommandKind::Kernel { args, .. } = &mut c[1].kind {
                        args.scalars[0] = -0.0
                    }
                }
                7 => {
                    if let CommandKind::Kernel { args, .. } = &mut c[1].kind {
                        args.scalars[1] = 2.0
                    }
                }
                8 => {
                    if let CommandKind::Kernel { args, .. } = &mut c[2].kind {
                        args.integers[1] = 4095
                    }
                }
                9 => {
                    if let CommandKind::Kernel { status, .. } = &mut c[1].kind {
                        *status = Some(ValueId::new(0))
                    }
                }
                10 => {
                    if let CommandKind::Kernel { scratch, .. } = &mut c[2].kind {
                        scratch[0] = Some(ValueId::new(0))
                    }
                }
                11 => {
                    c[1].overlap = CommandOverlap::Worker {
                        branch: 0,
                        start: true,
                        finish: true,
                    }
                }
                _ => {
                    p.instructions[0].state =
                        vec![ValueUse::write(ValueId::new(1))].into_boxed_slice()
                }
            }
            assert_eq!(fuse(&mut p, &mut c).unwrap(), 0, "case {case}");
            assert!(matches!(
                c[1].kind,
                CommandKind::Kernel {
                    name: "et_random_f32",
                    ..
                }
            ));
        }
    }
}

#[cfg(test)]
#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_RNG_ARG80=1"]
fn rng_arg80_hardware_matches_materialized_uniform_repeated_runs_and_retained_outputs() {
    use effect_torch_graph::{Device, Node, NodeKind};
    use effect_torch_runtime::{CancellationFlag, StorageMetadata};
    assert!(enabled());
    let device = crate::CudaDevice::get(0).unwrap();
    for width in [4096, 4097, 262144] {
        let shape = vec![3, width];
        let x = Node::new(NodeKind::Input {
            slot: 0,
            shape: shape.clone(),
            dtype: DType::F32,
            device: Device::Cuda(0),
            storage: StorageMetadata::dense(),
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
        let log = Node::new(NodeKind::Log { a: uniform.clone() }).unwrap();
        let neg = Node::new(NodeKind::Neg { a: log }).unwrap();
        let log = Node::new(NodeKind::Log { a: neg }).unwrap();
        let noise = Node::new(NodeKind::Neg { a: log }).unwrap();
        let add = Node::new(NodeKind::Add {
            a: x.clone(),
            b: noise,
        })
        .unwrap();
        let plain = Node::new(NodeKind::Argmax { a: x, dim: 1 }).unwrap();
        let sampled = Node::new(NodeKind::Argmax { a: add, dim: 1 }).unwrap();
        let options = || effect_torch_compiler::CompileOptions {
            random_seed: Some(0xffff_ffff_ffff_fff0),
            ..effect_torch_compiler::CompileOptions::default()
        };
        let optimized =
            crate::compile_with_options(vec![plain.clone(), sampled.clone()], 0, options())
                .unwrap();
        // Escaping the SAME random node prevents absorption. Both programs have
        // one random source (stable provenance zero) and the same explicit base
        // seed; successive draws also exercise wrapping invocation arithmetic.
        let reference =
            crate::compile_with_options(vec![plain, sampled, uniform], 0, options()).unwrap();
        assert!(optimized
            .diagnostics()
            .instructions
            .iter()
            .any(|i| i.kind == KERNEL));
        assert!(!reference
            .diagnostics()
            .instructions
            .iter()
            .any(|i| i.kind == KERNEL));
        let mut retained = Vec::new();
        for mode in 0..7 {
            let xs = (0..3 * width)
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
            let binding =
                crate::CudaValue::from_host(device.clone(), shape.clone(), DType::F32, &xs)
                    .unwrap();
            if mode == 4 {
                let cancelled = CancellationFlag::new();
                cancelled.cancel();
                let reference_error = reference
                    .execute(std::slice::from_ref(&binding), &[], &cancelled)
                    .err()
                    .expect("reference must reject pre-cancel");
                let optimized_error = optimized
                    .execute(std::slice::from_ref(&binding), &[], &cancelled)
                    .err()
                    .expect("optimized must reject pre-cancel");
                assert_eq!(optimized_error, reference_error);
                // Both arms obey the same executable-local counter semantics;
                // the immediately following successful draw verifies alignment.
            }
            let expected = reference
                .execute(
                    std::slice::from_ref(&binding),
                    &[],
                    &CancellationFlag::new(),
                )
                .unwrap();
            let actual = optimized
                .execute(&[binding], &[], &CancellationFlag::new())
                .unwrap();
            let bytes = expected[..2]
                .iter()
                .map(|v| v.read_storage_bytes().unwrap())
                .collect::<Vec<_>>();
            for (v, b) in actual.iter().zip(&bytes) {
                assert_eq!(v.read_storage_bytes().unwrap(), *b);
            }
            retained.push((actual, bytes));
        }
        drop(optimized);
        drop(reference);
        for (values, expected) in retained {
            for (v, b) in values.iter().zip(expected) {
                assert_eq!(v.read_storage_bytes().unwrap(), b);
            }
        }
    }
}

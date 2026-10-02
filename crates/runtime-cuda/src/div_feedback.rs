//! Fuse an already-legalized F32 division and its BF16 feedback store. Outputs
//! remain separate planned values; this pass changes no graph dtype policy.
use crate::executable::StateAccess;
use crate::lowering::{Command, CommandKind, CudaLoweredProgram};
use effect_torch_compiler::{InstructionEffects, OutputDecl, ValueStorage, ValueUse};
use effect_torch_runtime::{DType, StorageRepresentation, ValueId};
use std::collections::HashMap;

pub(super) fn fuse(
    program: &mut CudaLoweredProgram,
    commands: &mut [Command],
) -> Result<usize, String> {
    let positions = program
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
        .collect::<Vec<_>>();
    if positions.len() != commands.len() {
        return Err("compile: CUDA feedback command/instruction mapping mismatch".into());
    }
    let mut divisions = HashMap::<ValueId, usize>::new();
    let mut fused = 0;
    for position in 0..commands.len() {
        let command = &commands[position];
        let Some(output) = command.output else {
            continue;
        };
        let CommandKind::Kernel {
            name,
            args,
            inputs,
            scratch,
            status,
            checked,
            state,
            kv_matmul,
            ..
        } = &command.kind
        else {
            continue;
        };
        if *checked
            || status.is_some()
            || scratch.iter().any(Option::is_some)
            || !matches!(state, StateAccess::None)
            || kv_matmul.is_some()
        {
            continue;
        }
        if *name == "et_binary"
            && args.operation == 3
            && args.compute_dtype == 1
            && args.output_dtype == 1
            && args.input_dtypes[..2] == [1, 1]
            && args.integers[..2] == [0, 0]
            && args.elements != 0
        {
            divisions.insert(output, position);
            continue;
        }
        if *name != "et_convert"
            || args.output_dtype != 3
            || args.input_dtypes[0] != 1
            || inputs[1..].iter().any(Option::is_some)
        {
            continue;
        }
        let Some(source) = inputs[0] else {
            continue;
        };
        let Some(&producer) = divisions.get(&source) else {
            continue;
        };
        let source_meta = &program.values[source.index()];
        let output_meta = &program.values[output.index()];
        if source_meta.dtype != DType::F32
            || output_meta.dtype != DType::BF16
            || source_meta.shape != output_meta.shape
            || [source_meta, output_meta].iter().any(|value| {
                value.storage.representation != StorageRepresentation::Dense
                    || !matches!(value.decl.storage, ValueStorage::Planned { .. })
            })
        {
            continue;
        }
        let elements = args.elements;
        let CommandKind::Kernel {
            name,
            args: producer_args,
            scratch,
            ..
        } = &mut commands[producer].kind
        else {
            continue;
        };
        if *name != "et_binary" || producer_args.elements != elements {
            continue;
        }
        *name = "et_div_feedback";
        // The ABI scratch slot transports a second OUTPUT address. The memory
        // plan declares it as an output here, never as uninitialized scratch.
        scratch[0] = Some(output);
        let producer_instruction = &mut program.instructions[positions[producer]];
        producer_instruction.kind = "et_div_feedback";
        producer_instruction.outputs = producer_instruction
            .outputs
            .iter()
            .cloned()
            .chain([OutputDecl::new(output)])
            .collect();
        let cast_instruction = &mut program.instructions[positions[position]];
        cast_instruction.kind = "feedback_output";
        cast_instruction.inputs = vec![ValueUse::read(output)].into_boxed_slice();
        cast_instruction.outputs = Box::new([]);
        cast_instruction.scratch = Box::new([]);
        cast_instruction.staging = Box::new([]);
        cast_instruction.status = Box::new([]);
        cast_instruction.state = Box::new([]);
        cast_instruction.effects = InstructionEffects::default();
        commands[position].kind = CommandKind::PlannedAlias;
        divisions.remove(&source);
        fused += 1;
    }
    Ok(fused)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executable::CudaKernelArgs;
    use crate::lowering::{CommandOverlap, CudaValueMeta};
    use crate::workspace::CudaMemorySpace;
    use effect_torch_compiler::{
        analyze_liveness, plan_memory, LoweredInstruction, LoweredProgram, MemoryPlannerConfig,
        ValueDecl,
    };
    use effect_torch_runtime::{InstructionId, SegmentOwnership, StorageMetadata};

    fn fixture() -> (CudaLoweredProgram, Vec<Command>) {
        let values: Vec<_> = [
            (DType::F32, 1024),
            (DType::F32, 1),
            (DType::F32, 1024),
            (DType::BF16, 1024),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, (dtype, n))| CudaValueMeta {
            decl: ValueDecl::planned(
                ValueId::new(i as u32),
                format!("value{i}"),
                n * dtype.size_in_bytes(),
                256,
                CudaMemorySpace::Device,
                SegmentOwnership::Workspace,
            ),
            shape: vec![n],
            dtype,
            storage: StorageMetadata::dense(),
        })
        .collect();
        let kernel = |name, inputs, output_dtype, operation| CommandKind::Kernel {
            name,
            args: CudaKernelArgs {
                elements: 1024,
                compute_dtype: 1,
                output_dtype,
                operation,
                input_dtypes: [1; 8],
                ..Default::default()
            },
            inputs,
            scratch: [None; 3],
            status: None,
            checked: false,
            metadata: Vec::new(),
            state: StateAccess::None,
            state_buffers: [None; 4],
            kv_matmul: None,
        };
        let instructions = vec![
            LoweredInstruction::new(
                InstructionId::new(0),
                "prepare",
                Vec::new(),
                vec![
                    OutputDecl::new(ValueId::new(0)),
                    OutputDecl::new(ValueId::new(1)),
                ],
            ),
            LoweredInstruction::new(
                InstructionId::new(1),
                "et_binary",
                vec![
                    ValueUse::read(ValueId::new(0)),
                    ValueUse::read(ValueId::new(1)),
                ],
                vec![OutputDecl::new(ValueId::new(2))],
            ),
            LoweredInstruction::new(
                InstructionId::new(2),
                "et_convert",
                vec![ValueUse::read(ValueId::new(2))],
                vec![OutputDecl::new(ValueId::new(3))],
            ),
        ];
        let commands = vec![
            Command {
                output: None,
                kind: CommandKind::Prepare,
                overlap: CommandOverlap::Primary,
            },
            Command {
                output: Some(ValueId::new(2)),
                kind: kernel(
                    "et_binary",
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
                    1,
                    3,
                ),
                overlap: CommandOverlap::Primary,
            },
            Command {
                output: Some(ValueId::new(3)),
                kind: kernel(
                    "et_convert",
                    [
                        Some(ValueId::new(2)),
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ],
                    3,
                    0,
                ),
                overlap: CommandOverlap::Primary,
            },
        ];
        (
            LoweredProgram::new(values, instructions, vec![ValueId::new(2), ValueId::new(3)]),
            commands,
        )
    }
    #[test]
    fn feedback_definition_moves_before_use_and_outputs_cannot_alias() {
        let (mut program, mut commands) = fixture();
        assert_eq!(fuse(&mut program, &mut commands).unwrap(), 1);
        assert_eq!(program.instructions[1].outputs.len(), 2);
        assert!(program.instructions[2].outputs.is_empty());
        assert!(matches!(commands[2].kind, CommandKind::PlannedAlias));
        let liveness = analyze_liveness(&program).unwrap();
        assert_eq!(liveness.intervals[3].unwrap().start.index(), 1);
        let memory = plan_memory(
            &program,
            &MemoryPlannerConfig::uniform(CudaMemorySpace::Device, 1 << 20, 256, 256),
        )
        .unwrap();
        assert_ne!(memory.locations[2], memory.locations[3]);
        assert_eq!(fuse(&mut program, &mut commands).unwrap(), 0);
    }
    #[test]
    fn incompatible_dtype_shape_operation_and_scratch_are_rejected() {
        for case in 0..5 {
            let (mut program, mut commands) = fixture();
            match case {
                0 => program.values[3].dtype = DType::F16,
                1 => program.values[3].shape = vec![1023],
                2 => {
                    if let CommandKind::Kernel { args, .. } = &mut commands[1].kind {
                        args.operation = 0
                    }
                }
                3 => {
                    if let CommandKind::Kernel { scratch, .. } = &mut commands[1].kind {
                        scratch[0] = Some(ValueId::new(0))
                    }
                }
                _ => {
                    if let CommandKind::Kernel { checked, .. } = &mut commands[2].kind {
                        *checked = true
                    }
                }
            }
            assert_eq!(fuse(&mut program, &mut commands).unwrap(), 0, "case{case}");
        }
    }
    #[test]
    #[ignore = "requires CUDA and EFFECT_TORCH_CUDA_DIV_FEEDBACK=1"]
    fn div_feedback_preserves_bits_and_independent_output_lifetimes() {
        use crate::{CudaDevice, CudaValue};
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        use effect_torch_graph::{Device, Node, NodeKind};
        use effect_torch_runtime::CancellationFlag;
        assert_eq!(
            std::env::var("EFFECT_TORCH_CUDA_DIV_FEEDBACK").as_deref(),
            Ok("1")
        );
        let device = CudaDevice::get(0).unwrap();
        let n = 2048;
        let input = |slot, shape: Vec<usize>| {
            Node::new(NodeKind::Input {
                slot,
                shape,
                dtype: DType::F32,
                device: Device::Cuda(0),
                storage: StorageMetadata::dense(),
            })
            .unwrap()
        };
        let divided = Node::new(NodeKind::Div {
            a: input(0, vec![n]),
            b: input(1, vec![1]),
        })
        .unwrap();
        let feedback = Node::new(NodeKind::Cast {
            a: divided.clone(),
            dtype: DType::BF16,
        })
        .unwrap();
        let executable = crate::compile(vec![divided, feedback], 0).unwrap();
        assert!(executable
            .diagnostics()
            .instructions
            .iter()
            .any(|i| i.kind == "et_div_feedback" && i.count == 1));
        let bits: [u32; 18] = [
            0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x3f800000, 0xbf800000,
            0x3f808000, 0x3f818000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345,
            0xffc12345, 0x7f812345, 0xff812345,
        ];
        let bytes = (0..n)
            .flat_map(|i| bits[i % bits.len()].to_le_bytes())
            .collect::<Vec<_>>();
        let x = CudaValue::from_dense_bytes(device.clone(), vec![n], DType::F32, &bytes).unwrap();
        let allocate = |dtype: DType| {
            CudaValue::from_dense_bytes(
                device.clone(),
                vec![n],
                dtype,
                &vec![0; n * dtype.size_in_bytes()],
            )
            .unwrap()
        };
        let mut retained: Vec<(Vec<CudaValue>, Vec<Vec<u8>>)> = Vec::new();
        for temperature in [
            1.0f32,
            0.6,
            -1.0,
            0.0,
            f32::INFINITY,
            f32::from_bits(1),
            f32::NAN,
        ] {
            let scalar = CudaValue::from_dense_bytes(
                device.clone(),
                vec![1],
                DType::F32,
                &temperature.to_le_bytes(),
            )
            .unwrap();
            let full = allocate(DType::F32);
            let half = allocate(DType::BF16);
            let mut args = CudaKernelArgs {
                elements: n as u64,
                output: full.storage_address(),
                compute_dtype: 1,
                output_dtype: 1,
                operation: 3,
                ..Default::default()
            };
            args.inputs[0] = x.storage_address();
            args.inputs[1] = scalar.storage_address();
            args.input_dtypes[0] = 1;
            args.input_dtypes[1] = 1;
            args.integers[2] = 1;
            args.integers[3] = 2;
            let launch = |name: &str, args: &CudaKernelArgs| {
                let mut builder = device.stream.launch_builder(device.kernel(name).unwrap());
                builder.arg(args);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: ((n as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .unwrap();
            };
            launch("et_binary", &args);
            args.inputs[0] = full.storage_address();
            args.output = half.storage_address();
            args.output_dtype = 3;
            launch("et_convert", &args);
            let reference = vec![
                full.read_storage_bytes().unwrap(),
                half.read_storage_bytes().unwrap(),
            ];
            let actual = executable
                .execute(&[x.clone(), scalar], &[], &CancellationFlag::new())
                .unwrap();
            for (value, expected) in actual.iter().zip(&reference) {
                assert_eq!(value.read_storage_bytes().unwrap(), *expected);
            }
            retained.push((actual, reference));
            for (values, reference) in &retained {
                for (value, expected) in values.iter().zip(reference) {
                    assert_eq!(value.read_storage_bytes().unwrap(), *expected);
                }
            }
        }
        let cancelled = CancellationFlag::new();
        cancelled.cancel();
        let scalar = CudaValue::from_host(device.clone(), vec![1], DType::F32, &[1.0]).unwrap();
        assert!(executable.execute(&[x, scalar], &[], &cancelled).is_err());
        // Keep both roots owned independently even when sibling outputs drop.
        let (mut outputs, reference) = retained.pop().unwrap();
        outputs.remove(0);
        assert_eq!(outputs[0].read_storage_bytes().unwrap(), reference[1]);
    }
}

use crate::executable::Instruction;
use crate::value::element_count;
use crate::workspace::{CudaMemorySpace, CUDA_STORAGE_ALIGNMENT};
use effect_torch_compiler::{
    GraphIndex, InstructionEffects, LoweredInstruction, LoweredProgram, OutputDecl, ValueDecl,
    ValueStorage, ValueUse,
};
use effect_torch_graph::{node_children, Node};
use effect_torch_runtime::{
    DType, InstructionId, Location, SegmentOwnership, StorageClass, ValueId,
};

pub(super) type CudaLoweredProgram =
    LoweredProgram<&'static str, CudaMemorySpace, ValueDecl<CudaMemorySpace>>;

fn dense_storage_bytes(node: &Node) -> Result<usize, String> {
    let bytes_per_element = match node.dtype {
        DType::F32 => 4,
        DType::I64 => 16,
        _ => 8,
    };
    element_count(&node.shape)?
        .checked_mul(bytes_per_element)
        .ok_or_else(|| "compile: CUDA value storage byte size overflowed usize".to_string())
}

/// Builds the compiler-visible CUDA instruction and value tables. Dynamic
/// values and reusable operation scratch are assigned by the shared planner.
pub(super) fn lowered_program(
    index: &GraphIndex,
    instructions: &[Instruction],
) -> Result<CudaLoweredProgram, String> {
    if instructions.len() != index.order.len() {
        return Err("compile: CUDA instruction table does not match the graph index".to_string());
    }
    let mut escaping = vec![false; instructions.len()];
    for root in &index.roots {
        let mut value = root.index();
        while let Instruction::Alias { a, .. } = &instructions[value] {
            value = *a;
        }
        escaping[value] = true;
    }

    let scratch_bytes = instructions
        .iter()
        .map(Instruction::scratch_bytes)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    let scratch = if scratch_bytes == 0 {
        None
    } else {
        Some(
            ValueId::from_index(instructions.len())
                .ok_or_else(|| "compile: too many CUDA values".to_string())?,
        )
    };
    let mut values = Vec::with_capacity(instructions.len() + usize::from(scratch.is_some()));
    let mut lowered = Vec::with_capacity(instructions.len() + usize::from(scratch.is_some()));
    let mut paired_outputs = vec![false; instructions.len()];
    for instruction in instructions {
        if let Instruction::QuantizedLinear {
            pair: Some(pair), ..
        } = instruction
        {
            paired_outputs[pair.output] = true;
        }
    }
    if let Some(scratch) = scratch {
        lowered.push(
            LoweredInstruction::new(
                InstructionId::from_index(0)
                    .ok_or_else(|| "compile: too many CUDA instructions".to_string())?,
                "prepare_invocation",
                Vec::new(),
                vec![OutputDecl::new(scratch)],
            )
            .with_effects(InstructionEffects {
                may_fail: true,
                has_side_effects: false,
            }),
        );
    }
    for (position, (node, instruction)) in index.order.iter().zip(instructions).enumerate() {
        let value = ValueId::from_index(position)
            .ok_or_else(|| "compile: too many CUDA values".to_string())?;
        let bytes = match instruction {
            Instruction::Value(value) => value.storage_bytes()?,
            Instruction::Alias { a, .. } => values
                .get(*a)
                .map(|source: &ValueDecl<CudaMemorySpace>| source.bytes)
                .ok_or_else(|| "compile: CUDA alias source is out of range".to_string())?,
            _ => dense_storage_bytes(node)?,
        };
        let storage = match instruction {
            Instruction::Value(_) => ValueStorage::Fixed {
                class: StorageClass::PersistentConstant,
                location: Location::Persistent { slot: value.get() },
            },
            Instruction::Input {
                binding,
                scalar: false,
                ..
            } => ValueStorage::Fixed {
                class: StorageClass::ExternalInput,
                location: Location::External {
                    slot: u32::try_from(*binding)
                        .map_err(|_| "compile: too many CUDA tensor bindings".to_string())?,
                },
            },
            Instruction::Alias { a, .. } => ValueStorage::Alias {
                source: ValueId::from_index(*a)
                    .ok_or_else(|| "compile: CUDA alias source exceeds u32".to_string())?,
                byte_offset: 0,
            },
            _ => ValueStorage::Planned {
                class: if escaping[position] {
                    StorageClass::EscapingOutput
                } else {
                    StorageClass::Workspace
                },
                alignment: CUDA_STORAGE_ALIGNMENT,
                memory_space: CudaMemorySpace::Device,
                ownership: if escaping[position] {
                    SegmentOwnership::ProvisionalOutput
                } else {
                    SegmentOwnership::Workspace
                },
            },
        };
        values.push(ValueDecl {
            id: value,
            name: format!("{}_{position}", instruction.name()),
            bytes,
            storage,
        });

        let direct_inputs = match instruction {
            Instruction::Noop => Some(Vec::new()),
            Instruction::QuantizedLinear {
                x,
                weight,
                bias,
                pair,
                ..
            } => {
                let mut inputs = vec![Some(*x), Some(*weight), *bias];
                if let Some(pair) = pair {
                    inputs.push(Some(pair.weight));
                }
                if let Some(Instruction::MuseGate {
                    value,
                    gate,
                    multiplier,
                    ..
                }) = instructions.get(*x)
                {
                    inputs.extend([Some(*value), Some(*gate), *multiplier]);
                }
                Some(inputs)
            }
            Instruction::MuseGate {
                value,
                gate,
                multiplier,
                ..
            } => Some(vec![Some(*value), Some(*gate), *multiplier]),
            Instruction::MuseResidualRmsNorm {
                x,
                weight,
                residual,
                ..
            } => Some(vec![Some(*x), *weight, Some(*residual)]),
            Instruction::KvAttention {
                v,
                muse: Some(muse),
                ..
            } => Some(vec![
                Some(muse.q),
                Some(muse.k),
                Some(*v),
                Some(muse.q_weight),
                Some(muse.k_weight),
            ]),
            Instruction::ChunkedHeadCe {
                x,
                weight,
                bias,
                target,
                gradient,
                ..
            } => Some(vec![
                Some(*x),
                Some(*weight),
                Some(*bias),
                Some(*target),
                *gradient,
            ]),
            Instruction::LayerNorm {
                x, weight, other, ..
            } => Some(vec![Some(*x), Some(*weight), Some(*other)]),
            Instruction::Sdpa { q, k, v, g, .. } => Some(vec![Some(*q), Some(*k), Some(*v), *g]),
            Instruction::Optimizer {
                param,
                grad,
                state1,
                state2,
                lr,
                c1,
                c2,
                ..
            } => Some(vec![
                Some(*param),
                Some(*grad),
                Some(*state1),
                Some(*state2),
                Some(*lr),
                Some(*c1),
                Some(*c2),
            ]),
            Instruction::Kda {
                q,
                k,
                v,
                decay,
                beta,
                g,
                ..
            } => Some(vec![
                Some(*q),
                Some(*k),
                Some(*v),
                Some(*decay),
                Some(*beta),
                *g,
            ]),
            _ => None,
        };
        let mut inputs = if let Some(direct_inputs) = direct_inputs {
            let mut seen = Vec::new();
            direct_inputs
                .into_iter()
                .flatten()
                .filter(|input| {
                    if seen.contains(input) {
                        false
                    } else {
                        seen.push(*input);
                        true
                    }
                })
                .map(|input| {
                    ValueId::from_index(input)
                        .map(ValueUse::read)
                        .ok_or_else(|| "compile: CUDA instruction input exceeds u32".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            node_children(&node.kind)
                .into_iter()
                .map(|child| {
                    index
                        .dense_id(child.id)
                        .map(|dense| ValueUse::read(ValueId::new(dense.get())))
                        .ok_or_else(|| "compile: CUDA lowering lost a graph dependency".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        if instruction.scratch_bytes()? != 0 {
            inputs.push(ValueUse::read_write(
                scratch.expect("nonzero scratch instruction has scratch value"),
            ));
        }
        let outputs = if let Instruction::QuantizedLinear {
            pair: Some(pair), ..
        } = instruction
        {
            vec![
                OutputDecl::new(value),
                OutputDecl::new(
                    ValueId::from_index(pair.output)
                        .ok_or_else(|| "compile: paired CUDA output exceeds u32".to_string())?,
                ),
            ]
        } else if paired_outputs[position]
            || matches!(
                instruction,
                Instruction::Value(_)
                    | Instruction::Input { scalar: false, .. }
                    | Instruction::Alias { .. }
            )
        {
            Vec::new()
        } else {
            vec![OutputDecl::new(value)]
        };
        let stateful = matches!(
            instruction,
            Instruction::KvAttention { .. }
                | Instruction::Kda {
                    state_layer: Some(_),
                    ..
                }
                | Instruction::ShortConv {
                    state_layer: Some(_),
                    ..
                }
        );
        lowered.push(
            LoweredInstruction::new(
                InstructionId::from_index(position + usize::from(scratch.is_some()))
                    .ok_or_else(|| "compile: too many CUDA instructions".to_string())?,
                instruction.name(),
                inputs,
                outputs,
            )
            .with_effects(InstructionEffects {
                may_fail: instruction.encodes_command(),
                has_side_effects: stateful,
            }),
        );
    }
    if let Some(scratch) = scratch {
        values.push(ValueDecl {
            id: scratch,
            name: "invocation_scratch".to_string(),
            bytes: scratch_bytes,
            storage: ValueStorage::Planned {
                class: StorageClass::Workspace,
                alignment: CUDA_STORAGE_ALIGNMENT,
                memory_space: CudaMemorySpace::Device,
                ownership: SegmentOwnership::Workspace,
            },
        });
    }
    let outputs = index
        .roots
        .iter()
        .map(|root| {
            ValueId::from_index(root.index())
                .ok_or_else(|| "compile: CUDA output value exceeds u32".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(LoweredProgram::new(values, lowered, outputs))
}

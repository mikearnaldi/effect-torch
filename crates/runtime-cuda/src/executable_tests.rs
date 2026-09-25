use super::*;

#[path = "paged_state_tests.rs"]
mod paged_state_tests;
use crate::capabilities::CudaCapabilities;
use crate::executable::CudaKernelArgs;
use effect_torch_compiler::{
    CompileOptions, CompilerDriver, LoweringUnit, MemoryPlannerConfig, ProgramRequest,
};
use effect_torch_graph::Device;
use effect_torch_runtime::{GgmlKQuant, MemoryPlan, PackedFormat};
use std::sync::Arc;

fn input(slot: u32, shape: &[usize], dtype: DType) -> Arc<Node> {
    Node::new(NodeKind::Input {
        slot,
        shape: shape.to_vec(),
        dtype,
        device: Device::Cuda(0),
        storage: StorageMetadata::dense(),
    })
    .unwrap()
}
fn lower(
    roots: Vec<Arc<Node>>,
    optimize: bool,
    layout: Option<CudaStateLayout>,
) -> (
    CudaLoweredProgram,
    Vec<Command>,
    MemoryPlan<CudaMemorySpace>,
    usize,
    usize,
) {
    lower_with(roots, optimize, layout, true)
}

fn lower_with(
    roots: Vec<Arc<Node>>,
    optimize: bool,
    layout: Option<CudaStateLayout>,
    bf16_gemm: bool,
) -> (
    CudaLoweredProgram,
    Vec<Command>,
    MemoryPlan<CudaMemorySpace>,
    usize,
    usize,
) {
    let prepared = ProgramRequest::from_roots(
        roots,
        CompileOptions {
            optimize,
            ..Default::default()
        },
    )
    .prepare()
    .unwrap();
    let capabilities = CudaCapabilities::new(0, if bf16_gemm { 12 } else { 7 }, 0);
    let mut driver = CompilerDriver::new(&prepared, &capabilities).unwrap();
    let mut builder =
        CudaProgramBuilder::new(&prepared.index, layout, driver.legalization()).unwrap();
    driver
        .lower(|unit, index, optimization, plan| {
            let LoweringUnit::Node(id) = unit else {
                return Err("unexpected CUDA region".into());
            };
            let node = index.node(id).unwrap();
            let child = |node: &Arc<Node>| index.dense_id(node.id).unwrap().index();
            let instruction = match &node.kind {
                NodeKind::Input { slot, .. } => Instruction::Input {
                    binding: *slot as usize,
                    scalar: false,
                },
                NodeKind::Matmul { a, b } => Instruction::Matmul {
                    a: child(a),
                    b: child(b),
                },
                NodeKind::Permute { a, dims } => Instruction::Reindex {
                    op: 1,
                    a: child(a),
                    parameters: dims.iter().map(|dim| *dim as u64).collect(),
                },
                NodeKind::Reshape { a, .. } => Instruction::Alias { a: child(a) },
                NodeKind::BroadcastTo { a, .. } => Instruction::Reindex {
                    op: 0,
                    a: child(a),
                    parameters: Vec::new(),
                },
                NodeKind::Slice { a, ranges } => Instruction::Reindex {
                    op: 2,
                    a: child(a),
                    parameters: ranges
                        .iter()
                        .flat_map(|(start, end, step)| [*start as u64, *end as u64, *step as u64])
                        .collect(),
                },
                NodeKind::Neg { a } => Instruction::Unary {
                    op: 0,
                    a: child(a),
                    parameter: 0.0,
                },
                NodeKind::Concat { a, b, dim } => Instruction::Concat {
                    a: child(a),
                    b: child(b),
                    dim: *dim as u32,
                },
                NodeKind::ExpertLinearRows { x, weight, indexes } => {
                    Instruction::ExpertLinearRows {
                        x: child(x),
                        weight: child(weight),
                        indexes: child(indexes),
                        rows: x.shape[0],
                        columns: weight.shape[1],
                        inner: x.shape[1],
                        experts: weight.shape[0],
                    }
                }
                NodeKind::GroupedExpertLinearRows { x, weight, indexes } => {
                    Instruction::GroupedExpertLinearRows {
                        x: child(x),
                        weight: child(weight),
                        indexes: child(indexes),
                    }
                }
                NodeKind::Linear { x, weight, bias } => Instruction::Linear {
                    x: child(x),
                    weight: child(weight),
                    bias: child(bias),
                    k_width: weight.shape[0] as u32,
                    n_width: weight.shape[1] as u32,
                },
                NodeKind::Mul { a, b } => Instruction::Binary {
                    op: 2,
                    a: child(a),
                    b: child(b),
                },
                NodeKind::RmsNorm { x, weight, eps } => Instruction::RmsNorm {
                    x: child(x),
                    weight: weight.as_ref().map(child),
                    eps: *eps,
                },
                NodeKind::Gather { a, indexes, dim } => Instruction::Index {
                    op: 4,
                    a: child(a),
                    indexes: Some(child(indexes)),
                    src: None,
                    dim: *dim as u32,
                    width: a.shape[*dim],
                    trailing: *dim + 1 == a.shape.len(),
                },
                NodeKind::ScatterAdd {
                    a,
                    dim,
                    indexes,
                    src,
                } => Instruction::Index {
                    op: 5,
                    a: child(a),
                    indexes: Some(child(indexes)),
                    src: Some(child(src)),
                    dim: *dim as u32,
                    width: a.shape[*dim],
                    trailing: *dim + 1 == a.shape.len(),
                },
                NodeKind::QuantizedLinear { x, weight, bias } => {
                    let StorageRepresentation::Packed(PackedFormat::GgmlKQuant(codec)) =
                        weight.storage.representation
                    else {
                        panic!()
                    };
                    Instruction::QuantizedLinear {
                        x: child(x),
                        weight: child(weight),
                        bias: bias.as_ref().map(child),
                        codec,
                        rows: weight.shape[0] as u32,
                        columns: weight.shape[1] as u32,
                        row_bytes: codec.encoded_row_bytes(weight.shape[1]).unwrap() as u32,
                    }
                }
                NodeKind::Sdpa {
                    q,
                    k,
                    v,
                    scale,
                    causal,
                    ..
                } => Instruction::Sdpa {
                    op: 3,
                    q: child(q),
                    k: child(k),
                    v: child(v),
                    g: None,
                    scale: *scale,
                    causal: *causal,
                    window: None,
                },
                NodeKind::SdpaBackward {
                    q,
                    k,
                    v,
                    g,
                    scale,
                    causal,
                    ..
                } => Instruction::Sdpa {
                    op: 0,
                    q: child(q),
                    k: child(k),
                    v: child(v),
                    g: Some(child(g)),
                    scale: *scale,
                    causal: *causal,
                    window: None,
                },
                NodeKind::SdpaBackwardOut { of, .. } => Instruction::Alias { a: child(of) },
                NodeKind::KvAttention {
                    q,
                    k,
                    v,
                    scale,
                    layer,
                    window,
                    mode,
                    rounding,
                } => Instruction::KvAttention {
                    q: child(q),
                    k: child(k),
                    v: child(v),
                    q_shape: q.shape.clone(),
                    k_shape: k.shape.clone(),
                    scale: *scale,
                    layer: *layer as usize,
                    window: *window,
                    rounding: *rounding,
                    bidirectional: matches!(
                        mode,
                        effect_torch_graph::KvAttentionMode::BidirectionalBlock
                    ),
                },
                _ => return Err("unsupported host-test instruction".into()),
            };
            builder.add(id, node, index, optimization, instruction, plan)
        })
        .unwrap();
    let count = builder.conversion_count;
    let bytes = builder.conversion_bytes;
    let (program, commands) = builder.finish(&prepared.index).unwrap();
    let memory = driver
        .plan_memory(
            &program,
            &MemoryPlannerConfig::uniform(
                CudaMemorySpace::Device,
                usize::MAX / 2,
                CUDA_STORAGE_ALIGNMENT,
                CUDA_STORAGE_ALIGNMENT,
            ),
        )
        .unwrap();
    (program, commands, memory, count, bytes)
}

#[test]
fn kernel_descriptor_matches_cuda_layout() {
    assert_eq!(std::mem::size_of::<CudaKernelArgs>(), 360);
    assert_eq!(std::mem::offset_of!(CudaKernelArgs, metadata), 104);
    assert_eq!(std::mem::offset_of!(CudaKernelArgs, input_dtypes), 312);
}

#[test]
fn grouped_experts_plan_bounded_scratch_borrow_banks_and_report_host_completion() {
    for dtype in [DType::F32, DType::BF16] {
        for (rows, experts, columns, inner) in [
            (69, 7, 13, 37),
            (2, 512, 4096, 4096),
            (3, 2, 0, 5),
            (0, 1, 4, 5),
            (3, 2, 4, 0),
        ] {
            let root = Node::new(NodeKind::GroupedExpertLinearRows {
                x: input(0, &[rows, inner], dtype),
                weight: input(1, &[experts, columns, inner], dtype),
                indexes: input(2, &[rows], DType::U32),
            })
            .unwrap();
            let (program, commands, _, conversions, _) = lower(vec![root], true, None);
            assert_eq!(conversions, 0);
            let group = commands
                .iter()
                .find(|c| matches!(c.kind, CommandKind::GroupedExpert { .. }))
                .unwrap();
            let CommandKind::GroupedExpert {
                control,
                row_map,
                gathered,
                projected,
                weight,
                workspace,
                ..
            } = &group.kind
            else {
                unreachable!()
            };
            let bytes = |id: &ValueId| program.values[id.index()].decl.bytes;
            assert_eq!(bytes(control), (experts + 2) * 4);
            assert_eq!(bytes(row_map), rows * 4);
            assert_eq!(bytes(gathered), rows * inner * dtype.size_in_bytes());
            assert_eq!(bytes(projected), rows * columns * dtype.size_in_bytes());
            assert_eq!(
                bytes(weight),
                experts * columns * inner * dtype.size_in_bytes()
            );
            assert!(matches!(
                program.values[weight.index()].decl.storage,
                ValueStorage::Fixed {
                    class: StorageClass::ExternalInput,
                    ..
                }
            ));
            assert_eq!(
                workspace.is_some(),
                dtype == DType::BF16 && rows != 0 && columns != 0 && inner != 0
            );
            assert_eq!(
                crate::executable::physical_counts(&program, &commands).1,
                1 + usize::from(rows != 0)
            );
            assert!(program
                .instructions
                .iter()
                .any(|i| i.kind == "grouped_expert_linear_rows_host_control_non_capturable"));
        }
    }
}

#[test]
fn grouped_experts_reuse_unchanged_routing_control_and_row_map() {
    let (rows, experts, inner, middle, columns) = (64, 128, 2816, 704, 2816);
    let indexes = input(2, &[rows], DType::U32);
    let first = Node::new(NodeKind::GroupedExpertLinearRows {
        x: input(0, &[rows, inner], DType::BF16),
        weight: input(1, &[experts, middle, inner], DType::BF16),
        indexes: indexes.clone(),
    })
    .unwrap();
    let second = Node::new(NodeKind::GroupedExpertLinearRows {
        x: first,
        weight: input(3, &[experts, columns, middle], DType::BF16),
        indexes,
    })
    .unwrap();
    let (_, commands, _, _, _) = lower(vec![second], true, None);
    let groups = commands
        .iter()
        .filter_map(|command| match &command.kind {
            CommandKind::GroupedExpert {
                control,
                row_map,
                reuse_routing,
                ..
            } => Some((*control, *row_map, *reuse_routing)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(groups.len(), 2);
    assert!(!groups[0].2);
    assert!(groups[1].2);
    assert_eq!(groups[0].0, groups[1].0);
    assert_eq!(groups[0].1, groups[1].1);
}

#[test]
fn grouped_experts_gather_repeated_routes_from_compact_rows() {
    let rows = Node::new(NodeKind::Reshape {
        a: input(0, &[2, 5], DType::BF16),
        shape: vec![1, 2, 5],
    })
    .unwrap();
    let routed = Node::new(NodeKind::BroadcastTo {
        a: rows,
        shape: vec![3, 2, 5],
    })
    .unwrap();
    let expanded = Node::new(NodeKind::Reshape {
        a: routed,
        shape: vec![6, 5],
    })
    .unwrap();
    let root = Node::new(NodeKind::GroupedExpertLinearRows {
        x: expanded,
        weight: input(1, &[4, 7, 5], DType::BF16),
        indexes: input(2, &[6], DType::U32),
    })
    .unwrap();
    let (program, commands, _, _, _) = lower(vec![root], true, None);
    let CommandKind::GroupedExpert {
        x,
        rows,
        inner,
        source_rows,
        ..
    } = &commands
        .iter()
        .find(|command| matches!(command.kind, CommandKind::GroupedExpert { .. }))
        .expect("grouped expert command")
        .kind
    else {
        unreachable!()
    };
    assert_eq!((*rows, *inner, *source_rows), (6, 5, 2));
    assert_eq!(program.values[x.index()].shape, [1, 2, 5]);
    assert!(!commands.iter().any(|command| {
        matches!(
            command.kind,
            CommandKind::Kernel {
                name: "et_reindex",
                ..
            }
        ) && command
            .output
            .is_some_and(|output| program.values[output.index()].shape == [3, 2, 5])
    }));
}

#[test]
fn grouped_experts_materialize_general_broadcast_inputs() {
    let broadcast = Node::new(NodeKind::BroadcastTo {
        a: input(0, &[1, 2], DType::BF16),
        shape: vec![4, 2],
    })
    .unwrap();
    let root = Node::new(NodeKind::GroupedExpertLinearRows {
        x: broadcast,
        weight: input(1, &[3, 2, 2], DType::BF16),
        indexes: input(2, &[4], DType::U32),
    })
    .unwrap();
    let (program, commands, _, _, _) = lower(vec![root], true, None);
    let CommandKind::GroupedExpert {
        x,
        rows,
        source_rows,
        ..
    } = &commands
        .iter()
        .find(|command| matches!(command.kind, CommandKind::GroupedExpert { .. }))
        .expect("grouped expert command")
        .kind
    else {
        unreachable!()
    };
    assert_eq!((*rows, *source_rows), (4, 4));
    assert_eq!(program.values[x.index()].shape, [4, 2]);
    assert!(commands.iter().any(|command| {
        matches!(
            command.kind,
            CommandKind::Kernel {
                name: "et_reindex",
                ..
            }
        ) && command
            .output
            .is_some_and(|output| program.values[output.index()].shape == [4, 2])
    }));
}

#[test]
fn rms_norm_reads_outer_permutation_as_a_row_view() {
    let source = input(0, &[1, 2, 3, 5], DType::BF16);
    let permuted = Node::new(NodeKind::Permute {
        a: source,
        dims: vec![0, 2, 1, 3],
    })
    .unwrap();
    let root = Node::new(NodeKind::RmsNorm {
        x: permuted,
        weight: Some(input(1, &[5], DType::BF16)),
        eps: 1e-6,
    })
    .unwrap();
    let (_, commands, _, _, _) = lower(vec![root], true, None);
    assert!(!commands.iter().any(|command| matches!(
        command.kind,
        CommandKind::Kernel {
            name: "et_reindex",
            ..
        }
    )));
    let CommandKind::Kernel { args, .. } = &commands
        .iter()
        .find(|command| {
            matches!(
                command.kind,
                CommandKind::Kernel {
                    name: "et_rms_norm_f32",
                    ..
                }
            )
        })
        .expect("RMS command")
        .kind
    else {
        unreachable!()
    };
    assert_eq!(&args.integers[..9], &[5, 3, 0, 1, 3, 2, 6, 1, 3]);

    let f64_source = input(0, &[1, 2, 3, 5], DType::F64);
    let f64_permuted = Node::new(NodeKind::Permute {
        a: f64_source,
        dims: vec![0, 2, 1, 3],
    })
    .unwrap();
    let f64_root = Node::new(NodeKind::RmsNorm {
        x: f64_permuted,
        weight: None,
        eps: 1e-6,
    })
    .unwrap();
    let (_, f64_commands, _, _, _) = lower(vec![f64_root], true, None);
    assert!(f64_commands.iter().any(|command| matches!(
        command.kind,
        CommandKind::Kernel {
            name: "et_reindex",
            ..
        }
    )));
}

#[test]
fn rotary_half_reindex_folds_slices_negation_and_concat() {
    let source = input(0, &[2, 3, 8], DType::BF16);
    let first = Node::new(NodeKind::Slice {
        a: source.clone(),
        ranges: vec![(0, 2, 1), (0, 3, 1), (0, 4, 1)],
    })
    .unwrap();
    let second = Node::new(NodeKind::Slice {
        a: source.clone(),
        ranges: vec![(0, 2, 1), (0, 3, 1), (4, 8, 1)],
    })
    .unwrap();
    let negative = Node::new(NodeKind::Neg { a: second }).unwrap();
    let root = Node::new(NodeKind::Concat {
        a: negative,
        b: first,
        dim: 2,
    })
    .unwrap();
    let (program, commands, _, _, _) = lower(vec![root], true, None);
    let rotary = commands
        .iter()
        .find(|command| {
            matches!(
                command.kind,
                CommandKind::Kernel {
                    name: "et_rotary_reindex",
                    ..
                }
            )
        })
        .expect("rotary reindex command");
    let CommandKind::Kernel { args, inputs, .. } = &rotary.kind else {
        unreachable!()
    };
    assert_eq!(args.integers[0], 8);
    assert_eq!(program.values[inputs[0].unwrap().index()].shape, [2, 3, 8]);
    let kernels = commands
        .iter()
        .filter_map(|command| match &command.kind {
            CommandKind::Kernel { name, .. } => Some(*name),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        !kernels
            .iter()
            .any(|name| matches!(*name, "et_reindex" | "et_unary" | "et_concat")),
        "{kernels:?}"
    );
}

#[test]
fn scatter_add_uses_compact_indexes_for_an_inner_broadcast() {
    let compact = input(1, &[2, 3], DType::U32);
    let columns = Node::new(NodeKind::Reshape {
        a: compact.clone(),
        shape: vec![2, 3, 1],
    })
    .unwrap();
    let indexes = Node::new(NodeKind::BroadcastTo {
        a: columns,
        shape: vec![2, 3, 5],
    })
    .unwrap();
    let root = Node::new(NodeKind::ScatterAdd {
        a: input(0, &[2, 3, 5], DType::BF16),
        dim: 1,
        indexes,
        src: input(2, &[2, 3, 5], DType::BF16),
    })
    .unwrap();
    let (program, commands, _, _, _) = lower(vec![root], true, None);
    let CommandKind::Kernel { name, inputs, .. } = &commands
        .iter()
        .find(|command| {
            matches!(
                command.kind,
                CommandKind::Kernel {
                    name: "et_scatter_add_inner",
                    ..
                }
            )
        })
        .expect("compact scatter kernel")
        .kind
    else {
        unreachable!()
    };
    assert_eq!(*name, "et_scatter_add_inner");
    assert_eq!(program.values[inputs[1].unwrap().index()].shape, [2, 3]);
    assert!(!commands.iter().any(|command| {
        matches!(
            command.kind,
            CommandKind::Kernel {
                name: "et_reindex",
                ..
            }
        ) && command
            .output
            .is_some_and(|output| program.values[output.index()].shape == [2, 3, 5])
    }));
    assert_eq!(compact.shape, [2, 3]);
}

#[test]
fn expert_rows_keep_native_banks_and_u64_geometry_without_tensor_scratch() {
    for optimize in [false, true] {
        for dtype in [DType::F32, DType::BF16] {
            for (rows, experts, columns, inner) in [
                (64, 3, 13, 65),
                (1, 2, 65536, 65536), // Bank spans more than 2^32 elements.
                (u32::MAX as usize + 1, 1, 1, 0), // Row/grid arithmetic is also u64.
                (3, 2, 0, 13),        // Route validation still launches when O=0.
                (0, 2, 13, 65),
            ] {
                let root = Node::new(NodeKind::ExpertLinearRows {
                    x: input(0, &[rows, inner], dtype),
                    weight: input(1, &[experts, columns, inner], dtype),
                    indexes: input(2, &[rows], DType::U32),
                })
                .unwrap();
                // This direct F32 dot kernel does not require tensor core hardware.
                let (program, commands, memory, count, bytes) =
                    lower_with(vec![root], optimize, None, false);
                assert_eq!((count, bytes), (0, 0));
                let CommandKind::Kernel {
                    name,
                    args,
                    inputs,
                    scratch,
                    ..
                } = &commands
                    .iter()
                    .find(|command| {
                        matches!(
                            command.kind,
                            CommandKind::Kernel {
                                name: "et_expert_linear_rows",
                                ..
                            }
                        )
                    })
                    .unwrap()
                    .kind
                else {
                    unreachable!()
                };
                assert_eq!(*name, "et_expert_linear_rows");
                assert_eq!(
                    &args.integers[..4],
                    &[rows as u64, columns as u64, inner as u64, experts as u64]
                );
                assert_eq!(args.elements, (rows * columns) as u64);
                assert!(scratch.iter().all(Option::is_none));
                let weight = inputs[1].unwrap();
                assert_eq!(
                    program.values[weight.index()].decl.bytes,
                    experts * columns * inner * dtype.size_in_bytes()
                );
                assert!(matches!(
                    memory.locations[weight.index()],
                    Location::External { slot: 1 }
                ));
                assert_eq!(
                    &args.input_dtypes[..3],
                    &[dtype_code(dtype), dtype_code(dtype), dtype_code(DType::U32)]
                );
                assert_eq!(args.output_dtype, dtype_code(dtype));
                assert_eq!(
                    program.values[program.outputs[0].index()].decl.bytes,
                    rows * columns * dtype.size_in_bytes()
                );
                assert!(!program
                    .values
                    .iter()
                    .any(|value| value.decl.name.starts_with("scratch")));
                let statuses: Vec<_> = program
                    .values
                    .iter()
                    .filter(|value| value.decl.name.starts_with("status"))
                    .collect();
                assert_eq!(statuses.len(), 1);
                assert_eq!(statuses[0].decl.bytes, 8);
                assert_eq!(
                    crate::executable::physical_counts(&program, &commands).0,
                    2 + usize::from(rows != 0)
                );
            }
        }
    }
}

#[test]
fn f16_matmul_plans_f32_materialized_conversions() {
    for optimize in [false, true] {
        let a = input(0, &[2, 3], DType::F16);
        let b = input(1, &[3, 4], DType::F16);
        let root = Node::new(NodeKind::Matmul { a, b }).unwrap();
        let (program, commands, memory, count, bytes) = lower(vec![root], optimize, None);
        assert_eq!(count, 3);
        assert_eq!(bytes, 6 * 4 + 12 * 4 + 8 * 2);
        assert_eq!(program.values[program.outputs[0].index()].decl.bytes, 16);
        assert!(program.values.len() > 3);
        assert!(program.instructions.len() > 3);
        let CommandKind::Kernel { args, inputs, .. } = commands
            .iter()
            .find(|command| {
                matches!(
                    command.kind,
                    CommandKind::Kernel {
                        name: "et_matmul_f32",
                        ..
                    }
                )
            })
            .unwrap()
            .kind
        else {
            panic!()
        };
        assert_eq!(args.compute_dtype, dtype_code(DType::F32));
        for id in inputs.iter().flatten() {
            assert_eq!(program.values[id.index()].dtype, DType::F32);
        }
        assert!(matches!(
            memory.locations[program.outputs[0].index()],
            Location::Segment { bytes: 16, .. }
        ));
    }
}

#[test]
fn bf16_matmul_is_native_row_major_on_blackwell() {
    for optimize in [false, true] {
        let a = input(0, &[2, 3], DType::BF16);
        let b = input(1, &[3, 4], DType::BF16);
        let root = Node::new(NodeKind::Matmul { a, b }).unwrap();
        let (program, commands, _, count, bytes) = lower(vec![root], optimize, None);
        assert_eq!(count, 0);
        assert_eq!(bytes, 0);
        let gemm = commands
            .iter()
            .find(|command| matches!(command.kind, CommandKind::Gemm { .. }))
            .unwrap();
        let CommandKind::Gemm {
            x,
            weight,
            weight_transposed,
            plan,
            out_f32,
            ..
        } = &gemm.kind
        else {
            unreachable!()
        };
        assert!(!weight_transposed);
        assert!(!out_f32);
        assert_eq!((plan.m, plan.n, plan.k, plan.batch), (2, 4, 3, 1));
        assert_eq!(program.values[x.index()].dtype, DType::BF16);
        assert_eq!(program.values[weight.index()].dtype, DType::BF16);
        let result = gemm.output.unwrap();
        assert_eq!(program.values[result.index()].dtype, DType::BF16);
        assert_eq!(program.values[result.index()].decl.bytes, 8 * 2);
        assert!(!program.values.iter().any(|value| value.dtype == DType::F32));
        assert!(!commands.iter().any(|command| matches!(
            command.kind,
            CommandKind::Kernel {
                name: "et_matmul_f32",
                ..
            }
        )));
    }
}

#[test]
fn bf16_matmul_falls_back_to_f32_off_ampere() {
    let a = input(0, &[2, 3], DType::BF16);
    let b = input(1, &[3, 4], DType::BF16);
    let root = Node::new(NodeKind::Matmul { a, b }).unwrap();
    let (_, commands, _, count, _) = lower_with(vec![root], false, None, false);
    assert_eq!(count, 3);
    assert!(commands.iter().any(|command| matches!(
        command.kind,
        CommandKind::Kernel {
            name: "et_matmul_f32",
            ..
        }
    )));
    assert!(!commands
        .iter()
        .any(|command| matches!(command.kind, CommandKind::Gemm { .. })));
}

#[test]
fn bf16_linear_consumes_weight_directly() {
    let x = input(0, &[2, 3], DType::BF16);
    let weight = input(1, &[3, 2], DType::BF16);
    let bias = input(2, &[2], DType::BF16);
    let root = Node::new(NodeKind::Linear { x, weight, bias }).unwrap();
    let (program, commands, _, count, _) = lower(vec![root], true, None);
    assert_eq!(count, 0);
    let gemm = commands
        .iter()
        .find(|command| matches!(command.kind, CommandKind::Gemm { .. }))
        .unwrap();
    let CommandKind::Gemm {
        weight_transposed,
        out_f32,
        plan,
        ..
    } = &gemm.kind
    else {
        unreachable!()
    };
    assert!(!weight_transposed);
    assert!(out_f32);
    assert_eq!((plan.m, plan.n, plan.k), (2, 2, 3));
    assert!(commands
        .iter()
        .any(|command| matches!(command.kind, CommandKind::LinearBias { .. })));
    let weight = program
        .values
        .iter()
        .find(|value| value.decl.name.starts_with("input_1"))
        .unwrap();
    assert_eq!(weight.dtype, DType::BF16);
    assert_eq!(weight.decl.bytes, 6 * 2);
}

#[test]
fn bf16_linear_rows_folds_row_oriented_weight_without_copy() {
    let x = input(0, &[2, 3], DType::BF16);
    // Imported safetensors weight is row-oriented [N, K].
    let weight = input(1, &[2, 3], DType::BF16);
    let bias = input(2, &[2], DType::BF16);
    let transposed = Node::new(NodeKind::Permute {
        a: weight.clone(),
        dims: vec![1, 0],
    })
    .unwrap();
    let root = Node::new(NodeKind::Linear {
        x,
        weight: transposed,
        bias,
    })
    .unwrap();
    let (program, commands, _, count, _) = lower(vec![root], true, None);
    assert_eq!(count, 0);
    // The 2D transpose is a view, never a reindex or convert kernel.
    assert!(!commands.iter().any(|command| matches!(
        command.kind,
        CommandKind::Kernel {
            name: "et_reindex",
            ..
        }
    )));
    assert!(!program
        .values
        .iter()
        .any(|value| value.decl.name.starts_with("convert")));
    // Exactly one value carries the logical transposed weight shape, and it is
    // an allocation-free alias of the original BF16 bytes.
    let views = program
        .values
        .iter()
        .filter(|value| value.shape == vec![3, 2])
        .collect::<Vec<_>>();
    assert_eq!(views.len(), 1);
    assert_eq!(views[0].dtype, DType::BF16);
    assert_eq!(views[0].decl.bytes, 6 * 2);
    assert!(matches!(views[0].decl.storage, ValueStorage::Alias { .. }));
    let original = program
        .values
        .iter()
        .find(|value| value.decl.name.starts_with("input_1"))
        .unwrap();
    assert_eq!(original.shape, vec![2, 3]);
    assert_eq!(original.decl.bytes, 6 * 2);
    let gemm = commands
        .iter()
        .find(|command| matches!(command.kind, CommandKind::Gemm { .. }))
        .unwrap();
    let CommandKind::Gemm {
        weight_transposed,
        out_f32,
        plan,
        ..
    } = &gemm.kind
    else {
        unreachable!()
    };
    assert!(weight_transposed);
    assert!(out_f32);
    assert_eq!((plan.m, plan.n, plan.k), (2, 2, 3));
    assert!(commands
        .iter()
        .any(|command| matches!(command.kind, CommandKind::LinearBias { .. })));
}

#[test]
fn bf16_matmul_folds_row_oriented_weight_without_copy() {
    let x = input(0, &[2, 3], DType::BF16);
    let weight = input(1, &[2, 3], DType::BF16);
    let transposed = Node::new(NodeKind::Permute {
        a: weight.clone(),
        dims: vec![1, 0],
    })
    .unwrap();
    let root = Node::new(NodeKind::Matmul {
        a: x,
        b: transposed,
    })
    .unwrap();
    let (program, commands, _, count, _) = lower(vec![root], true, None);
    assert_eq!(count, 0);
    let gemm = commands
        .iter()
        .find(|command| matches!(command.kind, CommandKind::Gemm { .. }))
        .unwrap();
    let CommandKind::Gemm {
        weight_transposed,
        out_f32,
        plan,
        ..
    } = &gemm.kind
    else {
        unreachable!()
    };
    assert!(weight_transposed);
    assert!(!out_f32);
    assert_eq!((plan.m, plan.n, plan.k), (2, 2, 3));
    assert!(!program
        .values
        .iter()
        .any(|value| value.decl.name.starts_with("convert")));
}

#[test]
fn bf16_linear_odd_tail_dimensions_plan_native_gemm() {
    let x = input(0, &[3, 5], DType::BF16);
    let weight = input(1, &[5, 7], DType::BF16);
    let bias = input(2, &[7], DType::BF16);
    let root = Node::new(NodeKind::Linear { x, weight, bias }).unwrap();
    let (_, commands, _, count, _) = lower(vec![root], true, None);
    assert_eq!(count, 0);
    let CommandKind::Gemm { plan, .. } = commands
        .iter()
        .find(|command| matches!(command.kind, CommandKind::Gemm { .. }))
        .unwrap()
        .kind
    else {
        unreachable!()
    };
    assert_eq!((plan.m, plan.n, plan.k, plan.batch), (3, 7, 5, 1));
}

#[test]
fn bf16_row_weight_has_exact_storage_and_only_output_sized_f32_scratch() {
    for optimize in [false, true] {
        for with_bias in [false, true] {
            let x = input(0, &[2, 3, 5], DType::BF16);
            let rows = input(1, &[7, 5], DType::BF16);
            let w = Node::new(NodeKind::Permute {
                a: rows,
                dims: vec![1, 0],
            })
            .unwrap();
            let root = Node::new(if with_bias {
                NodeKind::Linear {
                    x,
                    weight: w,
                    bias: input(2, &[7], DType::BF16),
                }
            } else {
                NodeKind::Matmul { a: x, b: w }
            })
            .unwrap();
            let (program, commands, memory, count, bytes) = lower(vec![root], optimize, None);
            assert_eq!((count, bytes), (0, 0));
            let gemm = commands
                .iter()
                .find(|c| matches!(c.kind, CommandKind::Gemm { .. }))
                .unwrap();
            let CommandKind::Gemm {
                x,
                weight,
                workspace,
                plan,
                weight_transposed,
                ..
            } = gemm.kind
            else {
                unreachable!()
            };
            assert!(weight_transposed);
            assert_eq!((plan.batch, plan.stride_x, plan.stride_weight), (2, 15, 0));
            assert_eq!(program.values[x.index()].dtype, DType::BF16);
            let w = &program.values[weight.index()];
            assert_eq!(w.dtype, DType::BF16);
            assert_eq!(w.shape, [7, 5]);
            assert_eq!(w.decl.bytes, 2 * 7 * 5);
            assert!(matches!(
                w.decl.storage,
                ValueStorage::Fixed {
                    class: StorageClass::ExternalInput,
                    ..
                }
            ));
            assert!(matches!(
                memory.locations[weight.index()],
                Location::External { slot: 1 }
            ));
            let f32_values = program
                .values
                .iter()
                .filter(|v| v.dtype == DType::F32)
                .collect::<Vec<_>>();
            assert_eq!(f32_values.len(), usize::from(with_bias));
            // A native linear has no status word, metadata upload, or generic
            // kernel command. Its only scratch is the declared cuBLAS workspace
            // and, with bias, an output-sized F32 accumulator.
            assert!(!program
                .values
                .iter()
                .any(|v| v.decl.name.starts_with("status") || v.decl.name.starts_with("metadata")));
            assert!(!commands
                .iter()
                .any(|c| matches!(c.kind, CommandKind::Kernel { .. })));
            assert_eq!(
                crate::executable::physical_counts(&program, &commands),
                (1 + usize::from(with_bias), 1)
            );
            if with_bias {
                assert_eq!(f32_values[0].shape, [2, 3, 7]);
                assert_eq!(f32_values[0].decl.bytes, 2 * 3 * 7 * 4);
                let epilogue = commands
                    .iter()
                    .find(|c| matches!(c.kind, CommandKind::LinearBias { .. }))
                    .unwrap();
                let CommandKind::LinearBias { args, .. } = epilogue.kind else {
                    unreachable!()
                };
                assert_eq!(args.metadata, 0);
                assert_eq!(args.scratch, [0; 4]);
                assert_eq!(args.elements, 42);
                assert_eq!(args.integers[0], 7);
                assert_eq!(
                    &args.input_dtypes[..2],
                    &[dtype_code(DType::F32), dtype_code(DType::BF16)]
                );
            }
            assert!(!commands.iter().any(|c| matches!(
                c.kind,
                CommandKind::Kernel {
                    name: "et_reindex" | "et_convert",
                    ..
                }
            )));
            assert_eq!(
                program.values[workspace.index()].decl.bytes,
                CUBLAS_WORKSPACE_BYTES
            );
            assert!(matches!(
                memory.locations[workspace.index()],
                Location::Segment {
                    bytes: CUBLAS_WORKSPACE_BYTES,
                    ..
                }
            ));
        }
    }
}

#[test]
fn native_gemm_bypasses_transpose_even_when_a_canonical_output_needs_it() {
    let rows = input(1, &[7, 5], DType::BF16);
    let w = Node::new(NodeKind::Permute {
        a: rows,
        dims: vec![1, 0],
    })
    .unwrap();
    let root = Node::new(NodeKind::Matmul {
        a: input(0, &[3, 5], DType::BF16),
        b: w.clone(),
    })
    .unwrap();
    let (program, commands, _, count, _) = lower(vec![root, w], true, None);
    assert_eq!(count, 0);
    assert!(commands.iter().any(|c| matches!(
        c.kind,
        CommandKind::Kernel {
            name: "et_reindex",
            ..
        }
    )));
    let gemm = commands
        .iter()
        .find(|c| matches!(c.kind, CommandKind::Gemm { .. }))
        .unwrap();
    let CommandKind::Gemm {
        weight,
        weight_transposed,
        ..
    } = gemm.kind
    else {
        unreachable!()
    };
    assert!(weight_transposed);
    assert_eq!(program.values[weight.index()].shape, [7, 5]);
    let canonical = &program.values[program.outputs[1].index()];
    assert_eq!(canonical.shape, [5, 7]);
    assert!(matches!(
        canonical.decl.storage,
        ValueStorage::Planned { .. }
    ));
}

#[test]
fn zero_dimensions_have_an_explicit_materialized_legalization() {
    for (m, n, k) in [(0, 3, 5), (3, 0, 5), (3, 5, 0)] {
        let root = Node::new(NodeKind::Linear {
            x: input(0, &[m, k], DType::BF16),
            weight: input(1, &[k, n], DType::BF16),
            bias: input(2, &[n], DType::BF16),
        })
        .unwrap();
        let (_, commands, _, count, _) = lower(vec![root], true, None);
        assert_eq!(count, 4);
        assert!(!commands
            .iter()
            .any(|c| matches!(c.kind, CommandKind::Gemm { .. })));
        assert!(commands.iter().any(|c| matches!(
            c.kind,
            CommandKind::Kernel {
                name: "et_linear_f32",
                ..
            }
        )));
    }
}

#[test]
fn infallible_physical_diagnostics_omit_status_transfers_and_host_waits() {
    for elements in [0, 6] {
        let x = input(0, &[elements], DType::F32);
        let root = Node::new(NodeKind::Mul { a: x.clone(), b: x }).unwrap();
        let (program, commands, _, _, _) = lower(vec![root], false, None);
        // Infallible arithmetic only launches for nonempty outputs. The final
        // stream completion still precedes result publication.
        assert_eq!(
            crate::executable::physical_counts(&program, &commands),
            (usize::from(elements != 0), 1)
        );
    }
}

#[test]
fn checked_kernels_share_one_deferred_status_and_final_completion() {
    let values = input(0, &[4], DType::F32);
    let indexes = input(1, &[2], DType::U32);
    let first = Node::new(NodeKind::Gather {
        a: values.clone(),
        indexes: indexes.clone(),
        dim: 0,
    })
    .unwrap();
    let second = Node::new(NodeKind::Gather {
        a: values,
        indexes,
        dim: 0,
    })
    .unwrap();
    let (program, commands, _, _, _) = lower(vec![first, second], false, None);
    let statuses = program
        .values
        .iter()
        .filter(|value| value.decl.name.starts_with("status"))
        .collect::<Vec<_>>();
    assert_eq!(statuses.len(), 1);
    assert_eq!(statuses[0].decl.bytes, 8);
    assert_eq!(
        crate::executable::physical_counts(&program, &commands),
        (4, 1)
    );
}

#[test]
fn bf16_matmul_partial_broadcast_falls_back_to_f32() {
    // Both operands contribute distinct leading extents, which strided batched
    // GEMM cannot express. The explicit F32 route stays correct.
    let a = input(0, &[2, 1, 3, 4], DType::BF16);
    let b = input(1, &[1, 5, 4, 6], DType::BF16);
    let root = Node::new(NodeKind::Matmul { a, b }).unwrap();
    let (_, commands, _, count, _) = lower(vec![root], true, None);
    assert_eq!(count, 3);
    assert!(commands.iter().any(|command| matches!(
        command.kind,
        CommandKind::Kernel {
            name: "et_matmul_f32",
            ..
        }
    )));
}

#[test]
fn rms_norm_converts_half_storage_inside_the_kernel() {
    let x = input(0, &[3, 2816], DType::BF16);
    let weight = input(1, &[2816], DType::BF16);
    let root = Node::new(NodeKind::RmsNorm {
        x,
        weight: Some(weight),
        eps: 1e-6,
    })
    .unwrap();
    let (_, commands, _, count, _) = lower(vec![root], false, None);
    assert_eq!(count, 0);
    assert_eq!(
        commands
            .iter()
            .filter(|command| matches!(command.kind, CommandKind::Kernel { .. }))
            .count(),
        1
    );
    assert!(commands.iter().any(|command| matches!(
        command.kind,
        CommandKind::Kernel {
            name: "et_rms_norm_f32",
            ..
        }
    )));
}

#[test]
fn scalar_coercion_rounds_to_half_before_inline_opmath() {
    let tensor = input(0, &[1], DType::BF16);
    let scalar = input(1, &[], DType::F32);
    let root = Node::new(NodeKind::Mul {
        a: tensor,
        b: scalar,
    })
    .unwrap();
    let (program, commands, _, count, _) = lower(vec![root], false, None);
    assert_eq!(count, 1);
    let scalar_conversions = commands
        .iter()
        .filter_map(|command| match &command.kind {
            CommandKind::Kernel {
                name: "et_convert",
                inputs,
                ..
            } => Some((inputs[0].unwrap(), command.output.unwrap())),
            _ => None,
        })
        .filter(|(source, _)| program.values[source.index()].shape.is_empty())
        .map(|(source, out)| {
            (
                program.values[source.index()].dtype,
                program.values[out.index()].dtype,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(scalar_conversions, [(DType::F32, DType::BF16)]);
}

#[test]
fn integer_data_and_indexes_keep_distinct_exact_storage() {
    let a = input(0, &[2, 3], DType::I64);
    let indexes = input(1, &[2, 1], DType::U32);
    let root = Node::new(NodeKind::Gather { a, indexes, dim: 1 }).unwrap();
    let (program, commands, _, count, _) = lower(vec![root], true, None);
    assert_eq!(count, 0);
    assert_eq!(program.values[program.outputs[0].index()].decl.bytes, 16);
    let CommandKind::Kernel { args, .. } = commands
        .iter()
        .find(|command| {
            matches!(
                command.kind,
                CommandKind::Kernel {
                    name: "et_index",
                    ..
                }
            )
        })
        .unwrap()
        .kind
    else {
        panic!()
    };
    assert_eq!(
        &args.input_dtypes[..2],
        &[dtype_code(DType::I64), dtype_code(DType::U32)]
    );
}

#[test]
fn canonical_packed_weights_have_only_encoded_storage() {
    for codec in [
        GgmlKQuant::Q2K,
        GgmlKQuant::Q3K,
        GgmlKQuant::Q4K,
        GgmlKQuant::Q5K,
        GgmlKQuant::Q6K,
    ] {
        let x = input(0, &[16, 256], DType::F32);
        let weight = Node::new(NodeKind::Input {
            slot: 1,
            shape: vec![2, 256],
            dtype: DType::F32,
            device: Device::Cuda(0),
            storage: StorageMetadata::packed(codec),
        })
        .unwrap();
        let root = Node::new(NodeKind::QuantizedLinear {
            x,
            weight,
            bias: None,
        })
        .unwrap();
        let (program, commands, _, count, _) = lower(vec![root], true, None);
        assert_eq!(count, 0);
        let packed = program
            .values
            .iter()
            .find(|v| v.storage.representation != StorageRepresentation::Dense)
            .unwrap();
        assert_eq!(packed.decl.bytes, 2 * codec.encoded_row_bytes(256).unwrap());
        assert!(commands.iter().any(|c| matches!(
            c.kind,
            CommandKind::Kernel {
                name: "et_quantized_linear",
                ..
            }
        )));
        assert_eq!(
            program
                .values
                .iter()
                .filter(|v| v.decl.name.starts_with("scratch"))
                .count(),
            0
        );
    }
}

#[test]
fn backward_selectors_use_distinct_planned_outputs() {
    let q = input(0, &[1, 2, 3], DType::BF16);
    let k = input(1, &[1, 4, 3], DType::BF16);
    let v = input(2, &[1, 4, 5], DType::BF16);
    let g = input(3, &[1, 2, 5], DType::BF16);
    let fwd = Node::new(NodeKind::Sdpa {
        q: q.clone(),
        k: k.clone(),
        v: v.clone(),
        scale: 1.0,
        causal: false,
        window: effect_torch_graph::AttentionWindow::Inherit,
    })
    .unwrap();
    let of = Node::new(NodeKind::SdpaBackward {
        q,
        k,
        v,
        g,
        fwd,
        scale: 1.0,
        causal: false,
        window: effect_torch_graph::AttentionWindow::Inherit,
    })
    .unwrap();
    let roots = (0..3)
        .map(|index| {
            Node::new(NodeKind::SdpaBackwardOut {
                of: of.clone(),
                index,
            })
            .unwrap()
        })
        .collect();
    let (program, commands, _, count, _) = lower(roots, true, None);
    assert_eq!(count, 12);
    assert_eq!(
        program
            .outputs
            .iter()
            .map(|id| program.values[id.index()].shape.clone())
            .collect::<Vec<_>>(),
        [vec![1, 2, 3], vec![1, 4, 3], vec![1, 4, 5]]
    );
    assert_eq!(
        commands
            .iter()
            .filter(|c| matches!(
                c.kind,
                CommandKind::Kernel {
                    name: "et_sdpa_f32",
                    ..
                }
            ))
            .count(),
        4
    );
}

#[test]
fn kv_state_and_transaction_bytes_follow_configured_storage() {
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::U8] {
        let q = input(0, &[2, 4, 1, 8], DType::F32);
        let k = input(1, &[2, 2, 1, 8], DType::F32);
        let v = input(2, &[2, 2, 1, 8], DType::F32);
        let root = Node::new(NodeKind::KvAttention {
            q,
            k,
            v,
            scale: 1.0,
            layer: 0,
            window: None,
            mode: effect_torch_graph::KvAttentionMode::Causal,
            rounding: effect_torch_graph::AttentionRounding::Fused,
        })
        .unwrap();
        let layout = CudaStateLayout {
            capacity: 16,
            dtype,
            slots: 2,
            packed_rows_per_sequence: None,
            access: effect_torch_runtime::StateAccessMode::Append,
            kv_layers: vec![effect_torch_runtime::KvLayerDescriptor {
                layer_id: 0,
                kv_heads: 2,
                head_dim: 8,
                dtype,
                retention: None,
            }],
        };
        let (program, commands, memory, _, _) = lower(vec![root], false, Some(layout));
        let expected = 2 * 2 * 16 * 2 * 8 * dtype.size_in_bytes()
            + if dtype == DType::U8 {
                2 * 2 * 16 * 2 * 4
            } else {
                0
            };
        let state_bytes = program
            .values
            .iter()
            .filter(|v| {
                matches!(
                    v.decl.storage,
                    ValueStorage::Fixed {
                        class: StorageClass::PersistentState,
                        ..
                    }
                )
            })
            .map(|v| v.decl.bytes)
            .sum::<usize>();
        let staging_bytes = program
            .values
            .iter()
            .filter(|v| {
                matches!(
                    v.decl.storage,
                    ValueStorage::Planned {
                        ownership: SegmentOwnership::StateTransaction,
                        ..
                    }
                )
            })
            .map(|v| v.decl.bytes)
            .sum::<usize>();
        assert_eq!(state_bytes, expected);
        // A one-row private tail replaces the old whole-cache transaction.
        assert_eq!(staging_bytes, expected / 16);
        assert!(memory
            .segments
            .iter()
            .any(|s| s.ownership == SegmentOwnership::StateTransaction));
        let state_copies = commands
            .iter()
            .filter(|c| matches!(c.kind, CommandKind::Prepare))
            .count();
        assert_eq!(
            state_copies, 1,
            "only scratch/status preparation is submitted"
        );
    }
}

#[test]
fn every_dense_dtype_has_one_exact_allocation() {
    for dtype in [
        DType::F64,
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::I64,
        DType::U32,
        DType::U8,
    ] {
        let (program, _, _, _, _) = lower(vec![input(0, &[3, 5], dtype)], false, None);
        assert_eq!(program.values.len(), 1);
        assert_eq!(program.values[0].decl.bytes, 15 * dtype.size_in_bytes());
    }
}

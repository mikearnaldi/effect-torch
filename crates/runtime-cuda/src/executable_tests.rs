use super::*;
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
        .lower(|unit, index, _, plan| {
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
                NodeKind::Gather { a, indexes, dim } => Instruction::Index {
                    op: 4,
                    a: child(a),
                    indexes: Some(child(indexes)),
                    src: None,
                    dim: *dim as u32,
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
                    ..
                } => Instruction::KvAttention {
                    q: child(q),
                    k: child(k),
                    v: child(v),
                    q_shape: q.shape.clone(),
                    k_shape: k.shape.clone(),
                    scale: *scale,
                    layer: *layer as usize,
                    window: *window,
                    bidirectional: matches!(
                        mode,
                        effect_torch_graph::KvAttentionMode::BidirectionalBlock
                    ),
                },
                _ => return Err("unsupported host-test instruction".into()),
            };
            builder.add(id, node, index, instruction, plan)
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
fn physical_diagnostics_include_status_transfers_and_host_waits() {
    for elements in [0, 6] {
        let x = input(0, &[elements], DType::F32);
        let root = Node::new(NodeKind::Mul { a: x.clone(), b: x }).unwrap();
        let (program, commands, _, _, _) = lower(vec![root], false, None);
        // Reset status, launch only for nonempty outputs, read status, then
        // complete the stream before publishing results.
        assert_eq!(
            crate::executable::physical_counts(&program, &commands),
            (2 + usize::from(elements != 0), 2)
        );
    }
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
fn scalar_coercion_rounds_to_half_before_widening() {
    let tensor = input(0, &[1], DType::BF16);
    let scalar = input(1, &[], DType::F32);
    let root = Node::new(NodeKind::Mul {
        a: tensor,
        b: scalar,
    })
    .unwrap();
    let (program, commands, _, count, _) = lower(vec![root], false, None);
    assert_eq!(count, 4);
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
    assert_eq!(
        scalar_conversions,
        [(DType::F32, DType::BF16), (DType::BF16, DType::F32)]
    );
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
        })
        .unwrap();
        let layout = CudaStateLayout {
            capacity: 16,
            dtype,
            slots: 2,
            packed_rows_per_sequence: None,
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
        assert_eq!(staging_bytes, expected);
        assert!(memory
            .segments
            .iter()
            .any(|s| s.ownership == SegmentOwnership::StateTransaction));
        let commits = commands
            .iter()
            .filter(|c| matches!(c.kind, CommandKind::StateCopy { commit: true, .. }))
            .count();
        assert_eq!(commits, if dtype == DType::U8 { 4 } else { 2 });
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

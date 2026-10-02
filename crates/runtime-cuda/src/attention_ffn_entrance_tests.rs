use super::*;
use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
use effect_torch_graph::Node;
use effect_torch_runtime::{CancellationFlag, StorageMetadata, ValueSpec};
use std::sync::Arc;

struct Chain {
    outputs: Vec<Arc<Node>>,
    private: Vec<Arc<Node>>,
    view: Arc<Node>,
}
fn chain(rows: usize, eps: f64, reverse: bool, dependent: bool) -> Chain {
    chain_layout(rows, eps, reverse, dependent, 0)
}
fn chain_layout(rows: usize, eps: f64, reverse: bool, dependent: bool, layout: usize) -> Chain {
    let input = |slot, shape, dtype| {
        Node::new(NodeKind::Input {
            slot,
            shape,
            dtype,
            device: Device::Cuda(0),
            storage: StorageMetadata::dense(),
        })
        .unwrap()
    };
    let norm = |x, weight, eps| Node::new(NodeKind::RmsNorm { x, weight, eps }).unwrap();
    let shape = vec![1, rows, 2816];
    let a = match layout {
        1 => Node::new(NodeKind::Permute {
            a: input(0, vec![1, 2816, rows], DType::BF16),
            dims: vec![0, 2, 1],
        })
        .unwrap(),
        2 => Node::new(NodeKind::Slice {
            a: input(0, vec![1, rows, 2817], DType::BF16),
            ranges: vec![(0, 1, 1), (0, rows, 1), (1, 2817, 1)],
        })
        .unwrap(),
        3 => Node::new(NodeKind::BroadcastTo {
            a: input(0, vec![1, 1, 2816], DType::BF16),
            shape: shape.clone(),
        })
        .unwrap(),
        _ => input(0, shape.clone(), DType::BF16),
    };
    let h = input(1, shape.clone(), DType::BF16);
    let p = norm(a, Some(input(2, vec![2816], DType::BF16)), 1e-6);
    let r = Node::new(if reverse {
        NodeKind::Add { a: p.clone(), b: h }
    } else {
        NodeKind::Add { a: h, b: p.clone() }
    })
    .unwrap();
    let d = norm(r.clone(), Some(input(3, vec![2816], DType::BF16)), eps);
    let ew = if dependent {
        Node::new(NodeKind::Reshape {
            a: Node::new(NodeKind::Slice {
                a: d.clone(),
                ranges: vec![(0, 1, 1), (0, 1, 1), (0, 2816, 1)],
            })
            .unwrap(),
            shape: vec![2816],
        })
        .unwrap()
    } else {
        input(4, vec![2816], DType::BF16)
    };
    let ew = if dependent {
        Node::new(NodeKind::Add {
            a: ew,
            b: input(4, vec![2816], DType::BF16),
        })
        .unwrap()
    } else {
        ew
    };
    let e = norm(r.clone(), Some(ew), 1e-6);
    let view = Node::new(NodeKind::Reshape {
        a: r.clone(),
        shape: vec![rows, 2816],
    })
    .unwrap();
    let u = norm(view.clone(), None, 1e-6);
    let l = Node::new(NodeKind::Mul {
        a: u.clone(),
        b: input(5, vec![2816], DType::BF16),
    })
    .unwrap();
    let f = Node::new(NodeKind::Cast {
        a: l.clone(),
        dtype: DType::F32,
    })
    .unwrap();
    let m = Node::new(NodeKind::Mul {
        a: f.clone(),
        b: input(6, vec![1], DType::F32),
    })
    .unwrap();
    let s = Node::new(NodeKind::Cast {
        a: m.clone(),
        dtype: DType::BF16,
    })
    .unwrap();
    Chain {
        outputs: vec![r, d, e, s],
        private: vec![p, u, l, f, m],
        view,
    }
}

fn with_early_bindings(roots: Vec<Arc<Node>>) -> Vec<Arc<Node>> {
    let index = ProgramRequest::from_roots(roots.clone(), CompileOptions::default())
        .prepare()
        .unwrap();
    index
        .index
        .order
        .iter()
        .filter(|n| matches!(n.kind, NodeKind::Input { .. }))
        .cloned()
        .chain(roots)
        .collect()
}

#[test]
fn attention_ffn_entrance_boundaries_aliases_scheduling_and_policy() {
    for rows in [16, 64, 256, 512] {
        for enabled in [false, true] {
            for mutation in 0..12 {
                let c = chain(
                    rows,
                    if mutation == 7 { 1e-5 } else { 1e-6 },
                    mutation == 8,
                    mutation == 9,
                );
                let mut roots = c.outputs.clone();
                if mutation == 11 {
                    let indexes = Node::new(NodeKind::FromBytes {
                        data: 99999i64.to_le_bytes().to_vec(),
                        shape: vec![1],
                        dtype: DType::I64,
                        device: Device::Cuda(0),
                    })
                    .unwrap();
                    let checked = Node::new(NodeKind::IndexSelect {
                        a: c.outputs[0].clone(),
                        dim: 1,
                        indexes,
                    })
                    .unwrap();
                    roots.insert(1, checked);
                }
                if mutation < 5 {
                    roots.push(c.private[mutation].clone());
                }
                if mutation == 5 {
                    roots.push(c.view.clone());
                }
                let roots = if mutation == 10 {
                    roots
                } else {
                    with_early_bindings(roots)
                };
                let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
                    .prepare()
                    .unwrap();
                let mut caps = CudaCapabilities::new(0, 12, 0);
                caps.attention_ffn_entrance = enabled;
                let driver = CompilerDriver::new(&prepared, &caps).unwrap();
                let plan = driver.optimization();
                plan.validate(&prepared.index).unwrap();
                let selected = plan
                    .regions
                    .iter()
                    .find(|r| matches!(r, NativeRegion::AttentionFfnEntrance(_)));
                assert_eq!(
                    selected.is_some(),
                    enabled && matches!(rows, 64 | 256) && matches!(mutation, 5 | 6),
                    "rows={rows} enabled={enabled} mutation={mutation}"
                );
                if let Some(region) = selected {
                    assert_eq!(region.output_count(), 4);
                    for root in c.outputs.iter().chain([&c.view]) {
                        assert!(plan
                            .resolve(prepared.index.dense_id(root.id).unwrap())
                            .is_ok());
                    }
                    let NativeRegion::AttentionFfnEntrance(region) = region else {
                        unreachable!()
                    };
                    assert_eq!(region.residual_views.len(), 1);
                    assert_eq!(region.inputs.len(), 7);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_ATTN_FFN_ENTRANCE=1"]
fn attention_ffn_entrance_exact_unaligned_cancel_concurrent_retained_outputs() {
    assert_eq!(
        std::env::var("EFFECT_TORCH_CUDA_ATTN_FFN_ENTRANCE").as_deref(),
        Ok("1")
    );
    let device = crate::CudaDevice::get(0).unwrap();
    for rows in [64, 256] {
        for offset in [0, 2] {
            let c = chain(rows, 1e-6, false, false);
            let mut roots = c.outputs.clone();
            roots.push(c.view);
            let mut roots = with_early_bindings(roots);
            let optimized = Arc::new(crate::compile(roots.clone(), 0).unwrap());
            roots.extend(c.private);
            let reference = crate::compile(roots, 0).unwrap();
            assert!(optimized
                .diagnostics()
                .instructions
                .iter()
                .any(|i| i.kind == "et_attention_ffn_entrance_bf16"));
            assert!(!reference
                .diagnostics()
                .instructions
                .iter()
                .any(|i| i.kind == "et_attention_ffn_entrance_bf16"));
            let mut retained = Vec::new();
            let mut workers = Vec::new();
            for mode in 0..8 {
                let bindings = (0..7)
                    .map(|slot| {
                        let shape = if slot < 2 {
                            vec![1, rows, 2816]
                        } else if slot == 6 {
                            vec![1]
                        } else {
                            vec![2816]
                        };
                        let dtype = if slot == 6 { DType::F32 } else { DType::BF16 };
                        let data = (0..shape.iter().product())
                            .map(|i: usize| {
                                if slot == 6 {
                                    return match mode {
                                        2 => -0.,
                                        3 => f64::INFINITY,
                                        4 => f64::NAN,
                                        _ => 2816f64.powf(-0.5),
                                    };
                                }
                                let bits = ((i * 173 + slot * 917) % 65536) as u16;
                                match mode {
                                    1 => {
                                        if i % 2 == 0 {
                                            -0.
                                        } else {
                                            0.
                                        }
                                    }
                                    2 => half::bf16::from_bits((bits & 0x807f) | 1).to_f64(),
                                    3 => {
                                        if slot == 0 {
                                            f64::INFINITY
                                        } else {
                                            1.
                                        }
                                    }
                                    4 => half::bf16::from_bits(if i % 2 == 0 {
                                        0x7fc1
                                    } else {
                                        0xff81
                                    })
                                    .to_f64(),
                                    5 => half::bf16::from_bits(bits).to_f64(),
                                    6 => {
                                        if slot == 1 {
                                            -1.
                                        } else {
                                            1.
                                        }
                                    }
                                    7 => half::bf16::from_bits((bits & 0x807f) | 0x7f00).to_f64(),
                                    _ => ((i * 31 + slot * 17) % 103) as f64 / 19. - 2.7,
                                }
                            })
                            .collect::<Vec<_>>();
                        if slot == 0 && offset != 0 {
                            let raw = crate::value::dense_bytes_from_host(&data, dtype);
                            let mut padded = vec![0xa5; offset];
                            padded.extend(&raw);
                            padded.extend([0xa5; 2]);
                            let owner = Arc::new(device.stream.clone_htod(&padded).unwrap());
                            let buffer = crate::buffer::CudaBuffer::from_segment(
                                owner,
                                offset,
                                raw.len(),
                                None,
                            )
                            .unwrap();
                            let result = crate::CudaValue::from_planned_buffer(
                                device.clone(),
                                ValueSpec::dense(dtype, &shape),
                                buffer,
                            )
                            .unwrap();
                            assert_eq!(result.storage_address() % 8, 2);
                            result
                        } else {
                            crate::CudaValue::from_host(device.clone(), shape, dtype, &data)
                                .unwrap()
                        }
                    })
                    .collect::<Vec<_>>();
                let expected = reference
                    .execute(&bindings, &[], &CancellationFlag::new())
                    .unwrap()
                    .into_iter()
                    .skip(7)
                    .take(5)
                    .map(|v| v.read_storage_bytes().unwrap())
                    .collect::<Vec<_>>();
                let cancelled = CancellationFlag::new();
                cancelled.cancel();
                assert!(optimized.execute(&bindings, &[], &cancelled).is_err());
                let actual = optimized
                    .execute(&bindings, &[], &CancellationFlag::new())
                    .unwrap()
                    .into_iter()
                    .skip(7)
                    .collect::<Vec<_>>();
                for (v, b) in actual.iter().zip(&expected) {
                    assert_eq!(
                        &v.read_storage_bytes().unwrap(),
                        b,
                        "rows={rows} offset={offset} mode={mode}"
                    );
                }
                if mode < 2 {
                    let executable = optimized.clone();
                    let bytes = expected.clone();
                    workers.push(std::thread::spawn(move || {
                        let out = executable
                            .execute(&bindings, &[], &CancellationFlag::new())
                            .unwrap();
                        for (v, b) in out.iter().skip(7).zip(bytes) {
                            assert_eq!(v.read_storage_bytes().unwrap(), b);
                        }
                    }));
                }
                retained.push((actual, expected));
            }
            for worker in workers {
                worker.join().unwrap();
            }
            drop(optimized);
            drop(reference);
            for (values, bytes) in retained {
                for (v, b) in values.iter().zip(bytes) {
                    assert_eq!(v.read_storage_bytes().unwrap(), b);
                }
            }
        }
    }
}

#[test]
fn attention_ffn_entrance_physical_views_materialize_and_output_owner_is_explicit() {
    use crate::executable::Instruction;
    use crate::lowering::CudaProgramBuilder;
    use effect_torch_compiler::{LoweringUnit, ValueStorage};
    for layout in 0..4 {
        let c = chain_layout(64, 1e-6, false, false, layout);
        let mut roots = c.outputs;
        roots.push(c.view);
        let prepared =
            ProgramRequest::from_roots(with_early_bindings(roots), CompileOptions::default())
                .prepare()
                .unwrap();
        let mut caps = CudaCapabilities::new(0, 12, 0);
        caps.attention_ffn_entrance = true;
        let mut driver = CompilerDriver::new(&prepared, &caps).unwrap();
        let mut builder =
            CudaProgramBuilder::new(&prepared.index, None, driver.legalization()).unwrap();
        driver
            .lower(|unit, index, optimization, plan| {
                if let LoweringUnit::Region(id) = unit {
                    let NativeRegion::AttentionFfnEntrance(region) =
                        &optimization.regions[id.index()]
                    else {
                        return Err("unexpected region".into());
                    };
                    return builder.add_attention_ffn_entrance_region(
                        index,
                        optimization,
                        region,
                        plan,
                    );
                }
                let LoweringUnit::Node(id) = unit else {
                    unreachable!()
                };
                let node = index.node(id).unwrap();
                let child = |n: &Arc<Node>| index.dense_id(n.id).unwrap().index();
                let instruction = match &node.kind {
                    NodeKind::Input { slot, .. } => Instruction::Input {
                        binding: *slot as usize,
                        scalar: false,
                    },
                    NodeKind::Permute { a, dims } => Instruction::Reindex {
                        op: 1,
                        a: child(a),
                        parameters: dims.iter().map(|&d| d as u64).collect(),
                    },
                    NodeKind::Slice { a, ranges } => Instruction::Reindex {
                        op: 2,
                        a: child(a),
                        parameters: ranges
                            .iter()
                            .flat_map(|(s, e, t)| [*s as u64, *e as u64, *t as u64])
                            .collect(),
                    },
                    NodeKind::BroadcastTo { a, .. } => Instruction::Reindex {
                        op: 0,
                        a: child(a),
                        parameters: vec![],
                    },
                    _ => return Err("unexpected independent node".into()),
                };
                builder.add(id, node, index, optimization, instruction, plan)
            })
            .unwrap();
        let (program, _) = builder.finish(&prepared.index).unwrap();
        effect_torch_compiler::analyze_liveness(&program).unwrap();
        let fused = program
            .instructions
            .iter()
            .filter(|i| i.kind == "et_attention_ffn_entrance_bf16")
            .collect::<Vec<_>>();
        assert_eq!(fused.len(), 1);
        // All source views must materialize before the fused dense reader.
        if layout != 0 {
            assert!(program.instructions.iter().any(|i| i.kind == "et_reindex"));
        }
        let outputs = &program.outputs[7..];
        assert_eq!(outputs.len(), 5);
        let mut aliases = Vec::new();
        for &output in outputs {
            let value = &program.values[output.index()];
            let ValueStorage::Alias {
                source,
                byte_offset,
            } = value.decl.storage
            else {
                panic!("output must retain packed owner")
            };
            aliases.push((source, byte_offset));
        }
        assert!(aliases.iter().all(|a| a.0 == aliases[0].0));
        assert_eq!(
            aliases.iter().map(|a| a.1).collect::<Vec<_>>(),
            vec![0, 64 * 2816 * 2, 2 * 64 * 2816 * 2, 3 * 64 * 2816 * 2, 0]
        );
        assert_eq!(
            program.values[aliases[0].0.index()].decl.bytes,
            4 * 64 * 2816 * 2
        );
    }
}

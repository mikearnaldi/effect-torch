use super::*;
use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
use effect_torch_graph::Node;
use effect_torch_runtime::{CancellationFlag, StorageMetadata};
use std::sync::Arc;

struct Chain {
    inputs: Vec<Arc<Node>>,
    private: Vec<Arc<Node>>,
    outputs: Vec<Arc<Node>>,
}
fn chain(rows: usize, eps: f64, dtype: DType) -> Chain {
    chain_interleaved(rows, eps, dtype, false)
}
fn chain_interleaved(rows: usize, eps: f64, dtype: DType, computed_expert: bool) -> Chain {
    chain_order(rows, eps, dtype, u8::from(computed_expert))
}
fn chain_order(rows: usize, eps: f64, dtype: DType, mode: u8) -> Chain {
    let inputs = (0..8)
        .map(|slot| {
            Node::new(NodeKind::Input {
                slot,
                shape: if slot < 3 {
                    vec![1, rows, 2816]
                } else if slot == 6 {
                    vec![1]
                } else {
                    vec![2816]
                },
                dtype,
                device: Device::Cuda(0),
                storage: StorageMetadata::dense(),
            })
            .unwrap()
        })
        .collect::<Vec<_>>();
    let norm = |x, slot: usize, eps| {
        Node::new(NodeKind::RmsNorm {
            x,
            weight: Some(inputs[slot].clone()),
            eps,
        })
        .unwrap()
    };
    let d = norm(inputs[0].clone(), 3, 1e-6);
    let expert = if mode == 2 {
        Node::new(NodeKind::Mul {
            a: inputs[1].clone(),
            b: inputs[7].clone(),
        })
        .unwrap()
    } else if mode == 1 {
        Node::new(NodeKind::Neg {
            a: inputs[1].clone(),
        })
        .unwrap()
    } else {
        inputs[1].clone()
    };
    let e = norm(expert, 4, 1e-6);
    let sum = Node::new(NodeKind::Add {
        a: d.clone(),
        b: e.clone(),
    })
    .unwrap();
    let combined = norm(sum.clone(), 5, 1e-6);
    let added = Node::new(NodeKind::Add {
        a: inputs[2].clone(),
        b: combined.clone(),
    })
    .unwrap();
    let tail = Node::new(NodeKind::Mul {
        a: added.clone(),
        b: inputs[6].clone(),
    })
    .unwrap();
    let view = Node::new(NodeKind::Reshape {
        a: tail.clone(),
        shape: vec![rows, 2816],
    })
    .unwrap();
    let next = norm(view.clone(), 7, eps);
    Chain {
        inputs,
        private: vec![d, e, sum, combined, added],
        outputs: vec![tail, next, view],
    }
}
fn chain_exposed(rows: usize) -> Chain {
    let mut c = chain_interleaved(rows, 1e-6, DType::BF16, true);
    let first = Node::new(NodeKind::Expose {
        a: c.outputs[0].clone(),
        name: "model.hidden.0".into(),
    })
    .unwrap();
    let second = Node::new(NodeKind::Expose {
        a: first.clone(),
        name: "retained.hidden.0".into(),
    })
    .unwrap();
    let next = Node::new(NodeKind::RmsNorm {
        x: second.clone(),
        weight: Some(c.inputs[7].clone()),
        eps: 1e-6,
    })
    .unwrap();
    c.outputs = vec![c.outputs[0].clone(), next, first, second];
    c
}

#[test]
fn ffn_next_norm63_semantic_guards_and_two_output_routes() {
    for (rows, eps, dtype, enabled, expected) in [
        (64, 1e-6, DType::BF16, true, true),
        (256, 1e-6, DType::BF16, true, true),
        (16, 1e-6, DType::BF16, true, false),
        (512, 1e-6, DType::BF16, true, false),
        (64, 1e-5, DType::BF16, true, false),
        (64, 1e-6, DType::F32, true, false),
        (64, 1e-6, DType::BF16, false, false),
    ] {
        for escape in 0..7 {
            let c = chain(rows, eps, dtype);
            let mut roots = c.inputs.clone();
            roots.extend(c.outputs.clone());
            if escape < 5 {
                roots.push(c.private[escape].clone());
            } else if escape == 5 {
                roots.push(
                    Node::new(NodeKind::Neg {
                        a: c.outputs[0].clone(),
                    })
                    .unwrap(),
                );
            }
            let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
                .prepare()
                .unwrap();
            let mut caps = CudaCapabilities::new(0, 12, 0);
            caps.ffn_next_norm63 = enabled;
            let driver = CompilerDriver::new(&prepared, &caps).unwrap();
            let region = driver
                .optimization()
                .regions
                .iter()
                .find(|r| matches!(r, NativeRegion::FfnNextNorm(_)));
            assert_eq!(
                region.is_some(),
                expected && escape >= 5,
                "rows={rows} escape={escape}"
            );
            if let Some(region) = region {
                assert_eq!(region.output_count(), 2);
                assert_eq!(region.semantic_outputs().len(), 3);
            }
        }
    }
    let c = chain(64, 1e-6, DType::BF16);
    let prepared = ProgramRequest::from_roots(c.outputs, CompileOptions::default())
        .prepare()
        .unwrap();
    let mut caps = CudaCapabilities::new(0, 12, 0);
    caps.ffn_next_norm63 = true;
    let driver = CompilerDriver::new(&prepared, &caps).unwrap();
    assert!(
        !driver
            .optimization()
            .regions
            .iter()
            .any(|r| matches!(r, NativeRegion::FfnNextNorm(_))),
        "late external weight must not move before earlier bindings"
    );
}
#[test]
fn ffn_next_norm63_lowering_declares_both_aliases_and_owner() {
    use crate::executable::Instruction;
    use crate::lowering::CudaProgramBuilder;
    use effect_torch_compiler::{LoweringUnit, ValueStorage};
    for exposed in [false, true] {
        let c = if exposed {
            chain_exposed(64)
        } else {
            chain(64, 1e-6, DType::BF16)
        };
        let mut roots = c.inputs;
        roots.extend(c.outputs);
        let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
            .prepare()
            .unwrap();
        let mut caps = CudaCapabilities::new(0, 12, 0);
        caps.ffn_next_norm63 = true;
        let mut driver = CompilerDriver::new(&prepared, &caps).unwrap();
        let mut builder =
            CudaProgramBuilder::new(&prepared.index, None, driver.legalization()).unwrap();
        driver
            .lower(|unit, index, optimization, plan| match unit {
                LoweringUnit::Region(id) => {
                    let NativeRegion::FfnNextNorm(region) = &optimization.regions[id.index()]
                    else {
                        return Err("unexpected region".into());
                    };
                    builder.add_ffn_next_norm_region(index, optimization, region, plan)
                }
                LoweringUnit::Node(id) => {
                    let node = index.node(id).unwrap();
                    let instruction = match &node.kind {
                        NodeKind::Input { slot, .. } => Instruction::Input {
                            binding: *slot as usize,
                            scalar: false,
                        },
                        NodeKind::Neg { a } => Instruction::Unary {
                            op: 0,
                            a: index.dense_id(a.id).unwrap().index(),
                            parameter: 0.0,
                        },
                        _ => return Err("unexpected node".into()),
                    };
                    builder.add(id, node, index, optimization, instruction, plan)
                }
            })
            .unwrap();
        let (program, _) = builder.finish(&prepared.index).unwrap();
        effect_torch_compiler::analyze_liveness(&program).unwrap();
        assert_eq!(
            program
                .instructions
                .iter()
                .filter(|i| i.kind == "et_ffn_next_norm63_bf16")
                .count(),
            1
        );
        let aliases = program.outputs[8..]
            .iter()
            .map(|id| match program.values[id.index()].decl.storage {
                ValueStorage::Alias {
                    source,
                    byte_offset,
                } => (source, byte_offset),
                _ => panic!("explicit output alias required"),
            })
            .collect::<Vec<_>>();
        assert!(aliases.iter().all(|a| a.0 == aliases[0].0));
        assert_eq!(
            aliases.iter().map(|a| a.1).collect::<Vec<_>>(),
            if exposed {
                vec![0, 64 * 2816 * 2, 0, 0]
            } else {
                vec![0, 64 * 2816 * 2, 0]
            }
        );
        assert_eq!(
            program.values[aliases[0].0.index()].decl.bytes,
            2 * 64 * 2816 * 2
        );
    }
}
#[test]
fn ffn_next_norm63_declined_geometry_decomposes_named_exposures() {
    use crate::executable::Instruction;
    use crate::lowering::CudaProgramBuilder;
    use effect_torch_compiler::LoweringUnit;
    for exposed in [true] {
        let c = if exposed {
            chain_exposed(64)
        } else {
            chain(64, 1e-6, DType::BF16)
        };
        let mut roots = c.inputs;
        roots.extend(c.outputs);
        let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
            .prepare()
            .unwrap();
        let mut caps = CudaCapabilities::new(0, 12, 0);
        caps.ffn_next_norm63 = true;
        let mut driver = CompilerDriver::new(&prepared, &caps).unwrap();
        let mut builder =
            CudaProgramBuilder::new(&prepared.index, None, driver.legalization()).unwrap();
        driver
            .lower(|unit, index, optimization, plan| match unit {
                LoweringUnit::Region(id) => {
                    let NativeRegion::FfnNextNorm(region) = &optimization.regions[id.index()]
                    else {
                        return Err("unexpected region".into());
                    };
                    {
                        let mut declined = region.clone();
                        declined.rows = 16;
                        builder.add_ffn_next_norm_region(index, optimization, &declined, plan)
                    }
                }
                LoweringUnit::Node(id) => {
                    let node = index.node(id).unwrap();
                    let instruction = match &node.kind {
                        NodeKind::Input { slot, .. } => Instruction::Input {
                            binding: *slot as usize,
                            scalar: false,
                        },
                        NodeKind::Neg { a } => Instruction::Unary {
                            op: 0,
                            a: index.dense_id(a.id).unwrap().index(),
                            parameter: 0.0,
                        },
                        _ => return Err("unexpected node".into()),
                    };
                    builder.add(id, node, index, optimization, instruction, plan)
                }
            })
            .unwrap();
        let (program, _) = builder.finish(&prepared.index).unwrap();
        effect_torch_compiler::analyze_liveness(&program).unwrap();
        assert!(!program
            .instructions
            .iter()
            .any(|i| i.kind == "et_ffn_next_norm63_bf16"));
        let aliases = effect_torch_compiler::normalize_aliases(&program.values).unwrap();
        let outputs = &program.outputs[8..];
        assert_eq!(outputs.len(), 4);
        assert_eq!(
            aliases[outputs[0].index()].root,
            aliases[outputs[2].index()].root
        );
        assert_eq!(
            aliases[outputs[0].index()].root,
            aliases[outputs[3].index()].root
        );
        assert_ne!(
            aliases[outputs[0].index()].root,
            aliases[outputs[1].index()].root
        );
    }
}
#[test]
fn ffn_next_norm63_physical_dtype_and_extent_decline_before_owner() {
    for corruption in [0, 1] {
        use crate::executable::Instruction;
        use crate::lowering::CudaProgramBuilder;
        use effect_torch_compiler::LoweringUnit;
        let c = chain_exposed(64);
        let mut roots = c.inputs;
        roots.extend(c.outputs);
        let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
            .prepare()
            .unwrap();
        let mut caps = CudaCapabilities::new(0, 12, 0);
        caps.ffn_next_norm63 = true;
        let mut driver = CompilerDriver::new(&prepared, &caps).unwrap();
        let mut builder =
            CudaProgramBuilder::new(&prepared.index, None, driver.legalization()).unwrap();
        driver
            .lower(|unit, index, optimization, plan| match unit {
                LoweringUnit::Region(id) => {
                    let NativeRegion::FfnNextNorm(region) = &optimization.regions[id.index()]
                    else {
                        return Err("unexpected region".into());
                    };
                    let original_values = builder.values.len();
                    let value = builder
                        .values
                        .iter_mut()
                        .find(|value| value.shape == vec![1, 64, 2816])
                        .unwrap();
                    if corruption == 0 {
                        value.dtype = DType::F32;
                    } else {
                        value.decl.bytes -= 2;
                    }
                    // These intentionally inconsistent metadata fixtures need not
                    // yield an executable, but must never emit the fused owner.
                    let _ = builder.add_ffn_next_norm_region(index, optimization, region, plan);
                    assert!(builder.values[original_values..]
                        .iter()
                        .all(|value| value.shape != vec![2, 64 * 2816]));
                    assert!(builder.commands.iter().all(|command| !matches!(
                        &command.kind,
                        crate::lowering::CommandKind::Kernel {
                            name: "et_ffn_next_norm63_bf16",
                            ..
                        }
                    )));
                    Ok(())
                }
                LoweringUnit::Node(id) => {
                    let node = index.node(id).unwrap();
                    let instruction = match &node.kind {
                        NodeKind::Input { slot, .. } => Instruction::Input {
                            binding: *slot as usize,
                            scalar: false,
                        },
                        NodeKind::Neg { a } => Instruction::Unary {
                            op: 0,
                            a: index.dense_id(a.id).unwrap().index(),
                            parameter: 0.0,
                        },
                        _ => return Err("unexpected node".into()),
                    };
                    builder.add(id, node, index, optimization, instruction, plan)
                }
            })
            .unwrap();
    }
}
#[test]
#[ignore = "requires CUDA GPU; run with EFFECT_TORCH_CUDA_FFN_NEXT_NORM63=1"]
fn ffn_next_norm63_exact_offsets_cancel_errors_concurrent_retained() {
    assert_eq!(
        std::env::var("EFFECT_TORCH_CUDA_FFN_NEXT_NORM63").as_deref(),
        Ok("1")
    );
    let device = crate::CudaDevice::get(0).unwrap();
    for rows in [64, 256] {
        for offset in [0, 2] {
            let c = chain_exposed(rows);
            let mut roots = c.inputs.clone();
            roots.extend(c.outputs);
            let mut reference_roots = roots.clone();
            reference_roots.extend(c.private);
            let optimized = Arc::new(crate::compile(roots, 0).unwrap());
            let reference = crate::compile(reference_roots, 0).unwrap();
            assert!(optimized
                .diagnostics()
                .instructions
                .iter()
                .any(|i| i.kind == "et_ffn_next_norm63_bf16"));
            assert!(!reference
                .diagnostics()
                .instructions
                .iter()
                .any(|i| i.kind == "et_ffn_next_norm63_bf16"));
            let mut retained = Vec::new();
            let mut workers = Vec::new();
            for mode in 0..8 {
                let bindings = c
                    .inputs
                    .iter()
                    .enumerate()
                    .map(|(slot, node)| {
                        let data = (0..node.shape.iter().product())
                            .map(|i: usize| {
                                let bits = ((i * 173 + slot * 917) % 65536) as u16;
                                if slot == 6 {
                                    return [
                                        0.713,
                                        -0.321,
                                        0.,
                                        -0.,
                                        f64::INFINITY,
                                        f64::NEG_INFINITY,
                                        f64::NAN,
                                        1.,
                                    ][mode];
                                }
                                match mode {
                                    2 => {
                                        if i % 2 == 0 {
                                            -0.
                                        } else {
                                            0.
                                        }
                                    }
                                    3 => half::bf16::from_bits((bits & 0x807f) | 1).to_f64(),
                                    7 => half::bf16::from_bits(bits).to_f64(),
                                    _ => ((i * 31 + slot * 17) % 103) as f64 / 19. - 2.7,
                                }
                            })
                            .collect::<Vec<_>>();
                        if offset != 0 {
                            let mut raw = vec![0xa5; offset];
                            for v in &data {
                                raw.extend_from_slice(
                                    &half::bf16::from_f64(*v).to_bits().to_le_bytes(),
                                );
                            }
                            let owner = Arc::new(device.stream.clone_htod(&raw).unwrap());
                            let buffer = crate::buffer::CudaBuffer::from_segment(
                                owner,
                                offset,
                                raw.len() - offset,
                                None,
                            )
                            .unwrap();
                            crate::CudaValue::from_planned_buffer(
                                device.clone(),
                                effect_torch_runtime::ValueSpec::dense(DType::BF16, &node.shape),
                                buffer,
                            )
                            .unwrap()
                        } else {
                            crate::CudaValue::from_host(
                                device.clone(),
                                node.shape.clone(),
                                DType::BF16,
                                &data,
                            )
                            .unwrap()
                        }
                    })
                    .collect::<Vec<_>>();
                let expected = reference
                    .execute(&bindings, &[], &CancellationFlag::new())
                    .unwrap()
                    .into_iter()
                    .skip(8)
                    .take(4)
                    .map(|v| v.read_storage_bytes().unwrap())
                    .collect::<Vec<_>>();
                let cancel = CancellationFlag::new();
                cancel.cancel();
                assert!(optimized.execute(&bindings, &[], &cancel).is_err());
                assert!(optimized
                    .execute(&bindings[..7], &[], &CancellationFlag::new())
                    .is_err());
                let interrupted = CancellationFlag::new();
                let reached = std::cell::Cell::new(false);
                assert!(optimized
                    .execute_with_ffn63_hook(&bindings, &interrupted, &|| {
                        reached.set(true);
                        interrupted.cancel();
                    })
                    .is_err());
                assert!(reached.get(), "actual fused kernel hook must execute");
                let actual = optimized
                    .execute(&bindings, &[], &CancellationFlag::new())
                    .unwrap()
                    .into_iter()
                    .skip(8)
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
                        for (v, b) in out.iter().skip(8).zip(bytes) {
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
fn ffn_next_norm63_interleaved_expert_producer_reproduces_model_admission() {
    let c = chain_interleaved(64, 1e-6, DType::BF16, true);
    let mut roots = c.inputs;
    roots.extend(c.outputs);
    let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
        .prepare()
        .unwrap();
    let mut caps = CudaCapabilities::new(0, 12, 0);
    caps.ffn_tail = true;
    caps.ffn_next_norm63 = true;
    let driver = CompilerDriver::new(&prepared, &caps).unwrap();
    let region = driver
        .optimization()
        .regions
        .iter()
        .find_map(|r| {
            if let NativeRegion::FfnNextNorm(r) = r {
                Some(r)
            } else {
                None
            }
        })
        .expect(
            "existing tail accepts interleaved external expert producer and next norm must fuse",
        );
    assert!(region.inputs[1].index() > region.nodes[0].index());
    assert_eq!(region.outputs.len(), 2);
}

#[test]
fn ffn_next_norm63_rejects_new_weight_binding_mid_tail_and_after_tail() {
    for mode in [0, 2] {
        let c = chain_order(64, 1e-6, DType::BF16, mode);
        let mut roots = c.inputs[..7].to_vec();
        roots.extend(c.outputs.clone());
        let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
            .prepare()
            .unwrap();
        let mut caps = CudaCapabilities::new(0, 12, 0);
        caps.ffn_tail = true;
        caps.ffn_next_norm63 = true;
        let driver = CompilerDriver::new(&prepared, &caps).unwrap();
        let tail = driver
            .optimization()
            .regions
            .iter()
            .find_map(|r| {
                if let NativeRegion::FfnTail(r) = r {
                    Some(r)
                } else {
                    None
                }
            })
            .expect("tail remains legal");
        let weight = prepared.index.dense_id(c.inputs[7].id).unwrap().index();
        assert!(weight > tail.nodes[0].index());
        assert_eq!(
            weight < tail.output.index(),
            mode == 2,
            "mid-tail vs post-tail new binding fixture"
        );
        assert!(!driver
            .optimization()
            .regions
            .iter()
            .any(|r| matches!(r, NativeRegion::FfnNextNorm(_))));
    }
}

#[test]
fn ffn_next_norm63_exposed_model_chain_preserves_root_and_later_reader() {
    let c = chain_interleaved(64, 1e-6, DType::BF16, true);
    let exposed = Node::new(NodeKind::Expose {
        a: c.outputs[0].clone(),
        name: "model.hidden.0".into(),
    })
    .unwrap();
    let normalized = Node::new(NodeKind::RmsNorm {
        x: exposed.clone(),
        weight: Some(c.inputs[7].clone()),
        eps: 1e-6,
    })
    .unwrap();
    let later = Node::new(NodeKind::Add {
        a: exposed.clone(),
        b: c.inputs[2].clone(),
    })
    .unwrap();
    let mut roots = c.inputs;
    roots.extend([exposed.clone(), normalized, later]);
    let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
        .prepare()
        .unwrap();
    let mut caps = CudaCapabilities::new(0, 12, 0);
    caps.ffn_tail = true;
    caps.ffn_next_norm63 = true;
    let driver = CompilerDriver::new(&prepared, &caps).unwrap();
    let region = driver
        .optimization()
        .regions
        .iter()
        .find(|r| matches!(r, NativeRegion::FfnNextNorm(_)))
        .expect("actual model Expose wrapper must preserve fusion");
    let expose_id = prepared.index.dense_id(exposed.id).unwrap();
    assert!(region
        .semantic_outputs()
        .iter()
        .any(|out| out.semantic_node == expose_id && out.index == 0));
    let NodeKind::Expose { name, .. } = &prepared.index.order[expose_id.index()].kind else {
        panic!("exposure graph identity lost")
    };
    assert_eq!(name, "model.hidden.0");
}

#[test]
fn ffn_next_norm63_does_not_strip_checkpoint() {
    let c = chain_interleaved(64, 1e-6, DType::BF16, true);
    let checkpoint = Node::new(NodeKind::Checkpoint {
        a: c.outputs[0].clone(),
    })
    .unwrap();
    let next = Node::new(NodeKind::RmsNorm {
        x: checkpoint.clone(),
        weight: Some(c.inputs[7].clone()),
        eps: 1e-6,
    })
    .unwrap();
    let mut roots = c.inputs;
    roots.extend([checkpoint, next]);
    let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
        .prepare()
        .unwrap();
    let mut caps = CudaCapabilities::new(0, 12, 0);
    caps.ffn_tail = true;
    caps.ffn_next_norm63 = true;
    let driver = CompilerDriver::new(&prepared, &caps).unwrap();
    assert!(!driver
        .optimization()
        .regions
        .iter()
        .any(|r| matches!(r, NativeRegion::FfnNextNorm(_))));
}

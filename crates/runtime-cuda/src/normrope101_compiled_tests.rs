//! Real compiler/packed-GEMM lifecycle regression for the late group.
use crate::{CudaDevice, CudaValue};
use effect_torch_compiler::{CompileOptions, InferenceOptions};
use effect_torch_graph::{Device, Node, NodeKind};
use effect_torch_runtime::{CancellationFlag, DType, StorageMetadata};
use std::sync::Arc;

fn fixture() -> (Vec<Arc<Node>>, Arc<Node>) {
    fixture_with_tables(false)
}

fn fixture_with_tables(model_tables: bool) -> (Vec<Arc<Node>>, Arc<Node>) {
    let node = |kind| Node::new(kind).unwrap();
    let input = |slot, shape| {
        node(NodeKind::Input {
            slot,
            shape,
            dtype: DType::BF16,
            device: Device::Cuda(0),
            storage: StorageMetadata::dense(),
        })
    };
    let x = input(0, vec![1, 256, 2816]);
    let qw = input(1, vec![256]);
    let kw = input(2, vec![256]);
    let cosine = if model_tables {
        node(NodeKind::Input {
            slot: 3,
            shape: vec![1, 256],
            dtype: DType::U32,
            device: Device::Cuda(0),
            storage: StorageMetadata::dense(),
        })
    } else {
        input(3, vec![1, 1, 256, 128])
    };
    let sine = if model_tables {
        node(NodeKind::Input {
            slot: 4,
            shape: vec![128],
            dtype: DType::F32,
            device: Device::Cuda(0),
            storage: StorageMetadata::dense(),
        })
    } else {
        input(4, vec![1, 1, 256, 128])
    };
    let mut outputs = Vec::new();
    let mut raw_root = None;
    for (index, heads) in [16, 8, 8].into_iter().enumerate() {
        let data = (0..heads * 256 * 2816)
            .flat_map(|i| {
                let value = (((i * 17 + index * 13) % 31) as f32 + 1.0) / 2048.0;
                ((value.to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect();
        let weight = node(NodeKind::FromBytes {
            data,
            shape: vec![heads * 256, 2816],
            dtype: DType::BF16,
            device: Device::Cuda(0),
        });
        let transposed = node(NodeKind::Permute {
            a: weight,
            dims: vec![1, 0],
        });
        let raw = node(NodeKind::Matmul {
            a: x.clone(),
            b: transposed,
        });
        if index == 0 {
            raw_root = Some(raw.clone());
        }
        let reshape = node(NodeKind::Reshape {
            a: raw,
            shape: vec![1, 256, heads, 256],
        });
        let view = node(NodeKind::Permute {
            a: reshape,
            dims: vec![0, 2, 1, 3],
        });
        let norm = node(NodeKind::RmsNorm {
            x: view,
            weight: match index {
                0 => Some(qw.clone()),
                1 => Some(kw.clone()),
                _ => None,
            },
            eps: 1e-6,
        });
        if index == 2 {
            outputs.push(norm);
            continue;
        }
        let first = node(NodeKind::Slice {
            a: norm.clone(),
            ranges: vec![(0, 1, 1), (0, heads, 1), (0, 256, 1), (0, 128, 1)],
        });
        let second = node(NodeKind::Slice {
            a: norm.clone(),
            ranges: vec![(0, 1, 1), (0, heads, 1), (0, 256, 1), (128, 256, 1)],
        });
        let neg = node(NodeKind::Neg { a: second });
        let rotated = node(NodeKind::Concat {
            a: neg,
            b: first,
            dim: 3,
        });
        let (cos, sin) = if model_tables {
            // Exact model/rotary66 topology, independently constructed for Q/K:
            // positions→cast→reshape × cast(frequency)→Concat(phase,phase)
            // →Cos/Sin→BF16 cast→heads broadcast reshape.
            let pos = node(NodeKind::Cast {
                a: cosine.clone(),
                dtype: DType::F32,
            });
            let pos = node(NodeKind::Reshape {
                a: pos,
                shape: vec![1, 256, 1],
            });
            let freq = node(NodeKind::Cast {
                a: sine.clone(),
                dtype: DType::F32,
            });
            let phase = node(NodeKind::Mul { a: pos, b: freq });
            let doubled = node(NodeKind::Concat {
                a: phase.clone(),
                b: phase,
                dim: 2,
            });
            let cos = node(NodeKind::Cos { a: doubled.clone() });
            let sin = node(NodeKind::Sin { a: doubled });
            let cos = node(NodeKind::Cast {
                a: cos,
                dtype: DType::BF16,
            });
            let sin = node(NodeKind::Cast {
                a: sin,
                dtype: DType::BF16,
            });
            (
                node(NodeKind::Reshape {
                    a: cos,
                    shape: vec![1, 1, 256, 256],
                }),
                node(NodeKind::Reshape {
                    a: sin,
                    shape: vec![1, 1, 256, 256],
                }),
            )
        } else {
            (
                node(NodeKind::Concat {
                    a: cosine.clone(),
                    b: cosine.clone(),
                    dim: 3,
                }),
                node(NodeKind::Concat {
                    a: sine.clone(),
                    b: sine.clone(),
                    dim: 3,
                }),
            )
        };
        let direct = node(NodeKind::Mul { a: norm, b: cos });
        let cross = node(NodeKind::Mul { a: rotated, b: sin });
        outputs.push(node(NodeKind::Add {
            a: direct,
            b: cross,
        }));
    }
    // Bind borrowed inputs before the first normalization; the group may not
    // move dispatch across a late table/weight producer.
    let mut roots = vec![x, qw, kw, cosine, sine];
    roots.extend(outputs);
    (roots, raw_root.unwrap())
}
fn options() -> CompileOptions {
    let mut options = CompileOptions::from_environment();
    options.inference = Some(InferenceOptions {
        constant_weights: true,
    });
    options
}
fn bytes(values: &[CudaValue]) -> Vec<Vec<u8>> {
    values
        .iter()
        .map(|value| value.read_storage_bytes().unwrap())
        .collect()
}
fn close(got: &[u8], expected: &[u8]) {
    assert_eq!(got.len(), expected.len());
    for (got, expected) in got.chunks_exact(2).zip(expected.chunks_exact(2)) {
        let decode = |v: &[u8]| f32::from_bits(u32::from(u16::from_le_bytes([v[0], v[1]])) << 16);
        let got = decode(got);
        let expected = decode(expected);
        assert!(got.is_finite() && expected.is_finite());
        assert!(
            (got - expected).abs() <= 0.0234375 * expected.abs().max(0.05),
            "{got} vs {expected}"
        );
    }
}

#[test]
#[ignore = "requires SM120, packed77, norm-rope, mask and TRITON_NORMROPE101_DIRECTORY"]
fn normrope101_hardware_compiled_lifecycle() {
    let device = CudaDevice::get(0).unwrap();
    let (roots, raw) = fixture();
    let optimized = Arc::new(crate::compile_with_options(roots.clone(), 0, options()).unwrap());
    let count = |program: &crate::CudaExecutable, name| {
        program
            .diagnostics()
            .instructions
            .iter()
            .filter(|i| i.kind == name)
            .map(|i| i.count)
            .sum::<usize>()
    };
    assert_eq!(count(&optimized, "triton_normrope101"), 1);
    assert_eq!(count(&optimized, "packed_projection77"), 1);
    assert_eq!(count(&optimized, "et_norm_rope_bf16"), 0);
    assert_eq!(count(&optimized, "et_rms_norm_f32"), 0);
    let mut escaped = roots;
    escaped.push(raw);
    let reference = crate::compile_with_options(escaped, 0, options()).unwrap();
    assert_eq!(count(&reference, "triton_normrope101"), 0);
    assert_eq!(count(&reference, "packed_projection77"), 1);
    let mut retained = Vec::new();
    for seed in [3, 19] {
        let shapes = [
            vec![1, 256, 2816],
            vec![256],
            vec![256],
            vec![1, 1, 256, 128],
            vec![1, 1, 256, 128],
        ];
        let bindings = shapes
            .into_iter()
            .enumerate()
            .map(|(slot, shape)| {
                let data = (0..shape.iter().product())
                    .map(|i| match slot {
                        0 => 0.125 + ((i * 7 + seed) % 23) as f64 / 64.0,
                        1 | 2 => 0.75 + ((i + seed + slot) % 13) as f64 / 32.0,
                        3 => (((i / 128 + seed) % 17) as f64 / 32.0).cos(),
                        _ => (((i / 128 + seed) % 17) as f64 / 32.0).sin(),
                    })
                    .collect::<Vec<_>>();
                CudaValue::from_host(device.clone(), shape, DType::BF16, &data).unwrap()
            })
            .collect::<Vec<_>>();
        let borrowed = bytes(&bindings);
        let expected = reference
            .execute(&bindings, &[], &CancellationFlag::new())
            .unwrap();
        let cancelled = CancellationFlag::new();
        cancelled.cancel();
        assert!(optimized.execute(&bindings, &[], &cancelled).is_err());
        assert!(optimized
            .execute(&bindings[..4], &[], &CancellationFlag::new())
            .is_err());
        let cancelled = CancellationFlag::new();
        let reached = std::cell::Cell::new(false);
        assert!(optimized
            .execute_with_gemm_hook(&bindings, &cancelled, &|| {
                reached.set(true);
                cancelled.cancel();
            })
            .is_err());
        assert!(reached.get());
        let actual = optimized
            .execute(&bindings, &[], &CancellationFlag::new())
            .unwrap();
        for (got, want) in actual.iter().skip(5).zip(expected.iter().skip(5).take(3)) {
            close(
                &got.read_storage_bytes().unwrap(),
                &want.read_storage_bytes().unwrap(),
            );
        }
        let expected_bytes = bytes(&actual);
        let mut workers = Vec::new();
        for _ in 0..2 {
            let program = optimized.clone();
            let bindings = bindings.clone();
            let expected = expected_bytes.clone();
            workers.push(std::thread::spawn(move || {
                assert_eq!(
                    bytes(
                        &program
                            .execute(&bindings, &[], &CancellationFlag::new())
                            .unwrap()
                    ),
                    expected
                )
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(bytes(&bindings), borrowed);
        retained.push((actual, expected_bytes));
    }
    drop(optimized);
    drop(reference);
    for (values, expected) in retained {
        assert_eq!(bytes(&values), expected);
    }
}

#[test]
fn normrope101_audit_single_kv_root_schedule() {
    use effect_torch_compiler::{
        CompilerDriver, LoweringUnit, NativeRegion, ProgramRequest, StateCursorSlot,
    };
    let (roots, _) = fixture();
    let attention = Node::new(NodeKind::KvAttention {
        q: roots[5].clone(),
        k: roots[6].clone(),
        v: roots[7].clone(),
        scale: 1.0,
        layer: 0,
        window: Some(256),
        mode: effect_torch_graph::KvAttentionMode::BidirectionalBlock,
        rounding: effect_torch_graph::AttentionRounding::Stepwise,
    })
    .unwrap();
    let prepared = ProgramRequest::from_roots(vec![attention], options())
        .with_state_cursor(StateCursorSlot::new(20, true))
        .prepare()
        .unwrap();
    let caps = crate::capabilities::CudaCapabilities::new(0, 12, 0);
    let mut driver = CompilerDriver::new(&prepared, &caps).unwrap();
    let mut steps = Vec::new();
    driver
        .lower(|unit, index, optimization, _| {
            match unit {
                LoweringUnit::Node(id) => match &index.node(id).unwrap().kind {
                    NodeKind::Input { slot, .. } => steps.push(format!("input{slot}")),
                    NodeKind::KvAttention { .. } => steps.push("kv".into()),
                    NodeKind::RmsNorm { .. } => steps.push("rms".into()),
                    _ => {}
                },
                LoweringUnit::Region(id) => {
                    if let NativeRegion::NormRope(region) = &optimization.regions[id.index()] {
                        steps.push(format!("normrope{}", region.shape[1]));
                    }
                }
            }
            Ok(())
        })
        .unwrap();
    eprintln!("single-root physical lowering order: {steps:?}");
    assert_eq!(steps.last().map(String::as_str), Some("kv"));
    let first_norm = steps
        .iter()
        .position(|step| step == "normrope16" || step == "rms")
        .unwrap();
    let key_weight = steps.iter().position(|step| step == "input2").unwrap();
    assert!(
        first_norm < key_weight,
        "lazy KW must follow Q original normalization: {steps:?}"
    );
}

#[test]
#[ignore = "requires SM120 and native101 flags; lazy single-root KV append/read-only lifecycle"]
fn normrope101_hardware_lazy_kv_schedule() {
    lazy_kv_schedule(false);
}

#[test]
#[ignore = "requires SM120,101DIR,PACKED77,NORM_ROPE,ROTARY_REUSE66,RMS_VECTOR_LOADS,RMS_STATIC2816"]
fn normrope101_hardware_actual_model_tables_kv() {
    for flag in [
        "EFFECT_TORCH_CUDA_ROTARY_REUSE66",
        "EFFECT_TORCH_CUDA_RMS_VECTOR_LOADS",
        "EFFECT_TORCH_CUDA_RMS_STATIC2816",
    ] {
        assert_eq!(std::env::var(flag).as_deref(), Ok("1"));
    }
    lazy_kv_schedule(true);
}

fn lazy_kv_schedule(model_tables: bool) {
    use crate::{CudaSequenceState, CudaStateInvocation, CudaStateLayout};
    use effect_torch_runtime::{KvLayerDescriptor, StateAccessMode};
    let device = CudaDevice::get(0).unwrap();
    let (roots, raw) = fixture_with_tables(model_tables);
    let attention = Node::new(NodeKind::KvAttention {
        q: roots[5].clone(),
        k: roots[6].clone(),
        v: roots[7].clone(),
        scale: 1.0,
        layer: 0,
        window: Some(256),
        mode: effect_torch_graph::KvAttentionMode::BidirectionalBlock,
        rounding: effect_torch_graph::AttentionRounding::Stepwise,
    })
    .unwrap();
    let layout = |access| CudaStateLayout {
        capacity: 512,
        dtype: DType::BF16,
        slots: 1,
        packed_rows_per_sequence: None,
        access,
        kv_layers: vec![KvLayerDescriptor {
            layer_id: 0,
            kv_heads: 8,
            head_dim: 256,
            dtype: DType::BF16,
            retention: None,
        }],
    };
    let compile = |roots, access| {
        crate::compile_stateful_with_layout(roots, 0, 20, true, options(), layout(access)).unwrap()
    };
    let optimized = compile(vec![attention.clone()], StateAccessMode::Append);
    let reference = compile(vec![attention.clone(), raw], StateAccessMode::Append);
    let readonly = compile(vec![attention], StateAccessMode::ReadOnly);
    for program in [&optimized, &readonly] {
        let instructions = &program.diagnostics().instructions;
        assert_eq!(
            instructions
                .iter()
                .filter(|i| i.kind == "triton_normrope101")
                .map(|i| i.count)
                .sum::<usize>(),
            1
        );
        assert!(instructions.iter().any(|i| i.kind == "state_prepare"));
        if model_tables {
            assert!(
                instructions
                    .iter()
                    .filter(|i| i.kind == "rotary_reuse66")
                    .map(|i| i.count)
                    .sum::<usize>()
                    >= 2,
                "modelfixture must exercise distinct physical aliases from66"
            );
        }
    }
    assert!(!reference
        .diagnostics()
        .instructions
        .iter()
        .any(|i| i.kind == "triton_normrope101"));
    let shapes = [
        vec![1, 256, 2816],
        vec![256],
        vec![256],
        if model_tables {
            vec![1, 256]
        } else {
            vec![1, 1, 256, 128]
        },
        if model_tables {
            vec![128]
        } else {
            vec![1, 1, 256, 128]
        },
    ];
    let bindings = shapes
        .into_iter()
        .enumerate()
        .map(|(slot, shape)| {
            let data = (0..shape.iter().product())
                .map(|i| match slot {
                    0 => 0.25 + (i % 13) as f64 / 64.0,
                    1 | 2 => 1.0,
                    3 => {
                        if model_tables {
                            ((i * 3 + 17) % 997) as f64
                        } else {
                            0.875
                        }
                    }
                    _ => {
                        if model_tables {
                            0.0001 + (i % 17) as f64 * 0.0002
                        } else {
                            0.25
                        }
                    }
                })
                .collect::<Vec<_>>();
            let dtype = if model_tables && slot == 3 {
                DType::U32
            } else if model_tables && slot == 4 {
                DType::F32
            } else {
                DType::BF16
            };
            CudaValue::from_host(device.clone(), shape, dtype, &data).unwrap()
        })
        .collect::<Vec<_>>();
    let borrowed = bytes(&bindings);
    let invocation = |access, cursor, snapshot| CudaStateInvocation {
        sequences: vec![CudaSequenceState {
            cursor,
            keys: vec![],
            values: vec![],
            kda_states: vec![],
            conv_states: vec![],
            kv_storage: snapshot,
        }],
        slots: vec![0],
        valid_lengths: vec![256],
        capacity: 512,
        cache_dtype: DType::BF16,
        packed_rows_per_sequence: None,
        kv_layers: layout(access).kv_layers,
        access,
        cache: None,
    };
    let mut state = invocation(StateAccessMode::Append, 0, None);
    let mut expected_state = invocation(StateAccessMode::Append, 0, None);
    let actual = optimized
        .execute_stateful(&bindings, &[], &mut state, &CancellationFlag::new())
        .unwrap();
    let expected = reference
        .execute_stateful(
            &bindings,
            &[],
            &mut expected_state,
            &CancellationFlag::new(),
        )
        .unwrap();
    close(
        &actual[0].read_storage_bytes().unwrap(),
        &expected[0].read_storage_bytes().unwrap(),
    );
    optimized.readback_state(&mut state).unwrap();
    let snapshot = state.sequences[0].kv_storage.clone().unwrap();
    let mut state = invocation(StateAccessMode::ReadOnly, 256, Some(snapshot));
    let cancelled = CancellationFlag::new();
    cancelled.cancel();
    assert!(readonly
        .execute_stateful(&bindings, &[], &mut state, &cancelled)
        .is_err());
    let cancelled = CancellationFlag::new();
    let reached = std::cell::Cell::new(false);
    assert!(readonly
        .execute_stateful_with_kv_hook(&bindings, &mut state, &cancelled, &|| {
            reached.set(true);
            cancelled.cancel();
        })
        .is_err());
    assert!(reached.get());
    let read = readonly
        .execute_stateful(&bindings, &[], &mut state, &CancellationFlag::new())
        .unwrap();
    close(
        &read[0].read_storage_bytes().unwrap(),
        &actual[0].read_storage_bytes().unwrap(),
    );
    assert_eq!(bytes(&bindings), borrowed);
    let retained = bytes(&actual);
    let retained_read = bytes(&read);
    drop(optimized);
    drop(reference);
    drop(readonly);
    drop(state);
    assert_eq!(bytes(&actual), retained);
    assert_eq!(bytes(&read), retained_read);
}

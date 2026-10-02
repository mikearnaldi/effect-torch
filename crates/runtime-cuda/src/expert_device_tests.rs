//! Structural proof gates for device-controlled expert descriptors.
use crate::expert_device::{earlier_status, topk_rows, BYTES};
use effect_torch_graph::{Device, Node, NodeKind};
use effect_torch_runtime::{DType, StorageMetadata};
use std::sync::Arc;

fn node(kind: NodeKind) -> Arc<Node> {
    Node::new(kind).unwrap()
}
fn input(shape: &[usize], dtype: DType) -> Arc<Node> {
    node(NodeKind::Input {
        slot: 0,
        shape: shape.to_vec(),
        dtype,
        device: Device::Cuda(0),
        storage: StorageMetadata::dense(),
    })
}
fn full(shape: &[usize], dtype: DType, value: f64) -> Arc<Node> {
    node(NodeKind::Full {
        shape: shape.to_vec(),
        dtype,
        value,
        device: Device::Cuda(0),
    })
}
fn routes(scores: Arc<Node>, k: usize, dims: Vec<usize>) -> Arc<Node> {
    let rows = scores.shape[0] * k;
    node(NodeKind::Reshape {
        a: node(NodeKind::Permute {
            a: node(NodeKind::TopKIndices { a: scores, k }),
            dims,
        }),
        shape: vec![rows],
    })
}

#[test]
fn device_expert_topk_proof_accepts_exact_model_views_and_stable_ties() {
    for tokens in [64, 256] {
        let route = routes(input(&[tokens, 128], DType::F32), 8, vec![1, 0]);
        assert_eq!(topk_rows(&route), Some(tokens));
        // Stable TopK ties select distinct original indices. All-equal or
        // infinite scores do not weaken the per-expert <= token-count bound.
        for value in [0.0, f64::INFINITY, f64::NEG_INFINITY] {
            let tied = routes(full(&[tokens, 128], DType::F32, value), 8, vec![1, 0]);
            assert_eq!(topk_rows(&tied), Some(tokens));
        }
        assert!(tokens <= 512);
        assert_eq!(route.shape, [tokens * 8]);
    }
}

#[test]
fn device_expert_topk_proof_rejects_unproved_route_multiplicity_and_views() {
    for tokens in [64, 256] {
        let exact = routes(input(&[tokens, 128], DType::F32), 8, vec![1, 0]);
        let external = input(&[tokens * 8], DType::U32);
        let repeated = full(&[tokens * 8], DType::U32, 0.0);
        let broadcast = node(NodeKind::BroadcastTo {
            a: full(&[1], DType::U32, 0.0),
            shape: vec![tokens * 8],
        });
        let wrong_permute = routes(input(&[tokens, 128], DType::F32), 8, vec![0, 1]);
        let extra_reshape = node(NodeKind::Reshape {
            a: exact.clone(),
            shape: vec![tokens * 8],
        });
        let extra_slice = node(NodeKind::Slice {
            a: exact.clone(),
            ranges: vec![(0, tokens * 8, 1)],
        });
        let materialized_topk = node(NodeKind::Reshape {
            a: node(NodeKind::Permute {
                a: input(&[tokens, 8], DType::U32),
                dims: vec![1, 0],
            }),
            shape: vec![tokens * 8],
        });
        let wrong_flat_shape = node(NodeKind::Reshape {
            a: match &exact.kind {
                NodeKind::Reshape { a, .. } => a.clone(),
                _ => unreachable!(),
            },
            shape: vec![1, tokens * 8],
        });
        for rejected in [
            external,
            repeated,
            broadcast,
            wrong_permute,
            extra_reshape,
            extra_slice,
            materialized_topk,
            wrong_flat_shape,
        ] {
            assert_eq!(topk_rows(&rejected), None);
        }
        for k in [1, 7, 9, 16] {
            assert_eq!(
                topk_rows(&routes(input(&[tokens, 128], DType::F32), k, vec![1, 0])),
                None
            );
        }
        for experts in [64, 127, 129, 256] {
            assert_eq!(
                topk_rows(&routes(
                    input(&[tokens, experts], DType::F32),
                    8,
                    vec![1, 0]
                )),
                None
            );
        }
    }
    for tokens in [1, 32, 128, 512] {
        assert_eq!(
            topk_rows(&routes(input(&[tokens, 128], DType::F32), 8, vec![1, 0])),
            None
        );
    }
}

#[test]
fn device_expert_status_checkpoint_and_metadata_workspace_bounds() {
    assert!(BYTES >= 8192 + 1536 + 516);
    assert_eq!(BYTES % 16, 0);
    for position in [0, 1, 107, u32::MAX as usize] {
        let failure = ((position as u64) << 32) | 1;
        assert!(!earlier_status(failure, None));
        assert!(earlier_status(failure, Some(position)));
        if position > 0 {
            assert!(!earlier_status(failure, Some(position - 1)));
        }
    }
    assert!(!earlier_status(0, Some(usize::MAX)));
}

fn expert_fixture(tokens: usize) -> Arc<Node> {
    let binding = |slot, shape: Vec<usize>, dtype| {
        node(NodeKind::Input {
            slot,
            shape,
            dtype,
            device: Device::Cuda(0),
            storage: StorageMetadata::dense(),
        })
    };
    let rows = tokens * 8;
    let indexes = routes(binding(1, vec![tokens, 128], DType::F32), 8, vec![1, 0]);
    // Expert-specific small host banks expand on-device. Unlike a uniform
    // constant bank, these expose descriptor expert/weight-pointer mistakes.
    let bank = |slot, columns, inner| {
        node(NodeKind::BroadcastTo {
            a: binding(slot, vec![128, 1, 1], DType::BF16),
            shape: vec![128, columns, inner],
        })
    };
    let first = node(NodeKind::GroupedExpertLinearRows {
        x: binding(0, vec![rows, 2816], DType::BF16),
        weight: bank(2, 1408, 2816),
        indexes: indexes.clone(),
    });
    let gate = node(NodeKind::Slice {
        a: first.clone(),
        ranges: vec![(0, rows, 1), (0, 704, 1)],
    });
    let up = node(NodeKind::Slice {
        a: first,
        ranges: vec![(0, rows, 1), (704, 1408, 1)],
    });
    let activated = node(NodeKind::Gelu {
        a: gate,
        approximate: true,
    });
    let product = node(NodeKind::Mul {
        a: activated,
        b: up,
    });
    node(NodeKind::GroupedExpertLinearRows {
        x: product,
        weight: bank(3, 2816, 704),
        indexes,
    })
}

fn dense_fixture(expert: Arc<Node>, tokens: usize) -> Arc<Node> {
    let mut dense = input(&[tokens * 8, 2816], DType::BF16);
    for (inner, columns) in [(2816, 128), (128, 128), (128, 2816)] {
        dense = node(NodeKind::Matmul {
            a: dense,
            b: node(NodeKind::Permute {
                a: full(&[columns, inner], DType::BF16, 0.0009765625),
                dims: vec![1, 0],
            }),
        });
    }
    node(NodeKind::Add {
        a: dense,
        b: expert,
    })
}

fn finalized_fixture(projection: Arc<Node>, tokens: usize) -> Arc<Node> {
    let token = node(NodeKind::Permute {
        a: node(NodeKind::Reshape {
            a: projection,
            shape: vec![8, tokens, 2816],
        }),
        dims: vec![1, 0, 2],
    });
    let weighted = node(NodeKind::Cast {
        a: node(NodeKind::Mul {
            a: node(NodeKind::Cast {
                a: token,
                dtype: DType::F32,
            }),
            b: full(&[tokens, 8, 1], DType::F32, 0.125),
        }),
        dtype: DType::BF16,
    });
    let ranks = node(NodeKind::BroadcastTo {
        a: node(NodeKind::Reshape {
            a: full(&[tokens, 8], DType::U32, 0.),
            shape: vec![tokens, 8, 1],
        }),
        shape: vec![tokens, 8, 2816],
    });
    let scatter = node(NodeKind::ScatterAdd {
        a: node(NodeKind::Zeros {
            shape: vec![tokens, 8, 2816],
            dtype: DType::BF16,
            device: Device::Cuda(0),
        }),
        dim: 1,
        indexes: ranks,
        src: weighted,
    });
    let mut result = node(NodeKind::Zeros {
        shape: vec![tokens, 2816],
        dtype: DType::BF16,
        device: Device::Cuda(0),
    });
    for route in 0..8 {
        result = node(NodeKind::Add {
            a: result,
            b: node(NodeKind::Reshape {
                a: node(NodeKind::Slice {
                    a: scatter.clone(),
                    ranges: vec![(0, tokens, 1), (route, route + 1, 1), (0, 2816, 1)],
                }),
                shape: vec![tokens, 2816],
            }),
        });
    }
    result
}

#[test]
#[ignore = "requires CUDA, accepted merged artifact and EFFECT_TORCH_CUDA_EXPERT_DEVICE=1"]
fn device_expert_real_topk_exact_outputs_errors_retention_and_recovery() {
    use crate::{CudaDevice, CudaValue};
    use effect_torch_runtime::CancellationFlag;
    assert_eq!(std::env::var(crate::expert_device::ENV).as_deref(), Ok("1"));
    // Populate the device registry with native59 functions before disabling
    // only compile selection for the otherwise-identical baseline.
    let device = CudaDevice::get(0).unwrap();
    let host = |shape, dtype, data: &[f64]| {
        CudaValue::from_host(device.clone(), shape, dtype, data).unwrap()
    };
    let device_count = |executable: &crate::CudaExecutable| {
        executable
            .diagnostics()
            .instructions
            .iter()
            .filter(|i| i.kind == "grouped_expert_linear_rows_device59_non_capturable")
            .map(|i| i.count)
            .sum::<usize>()
    };
    let mut retained = Vec::new();
    for (tokens, finalized, dense) in [
        (64, false, false),
        (256, false, false),
        (64, true, false),
        (256, true, false),
        (64, false, true),
    ] {
        let root = expert_fixture(tokens);
        let root = if finalized {
            finalized_fixture(root, tokens)
        } else {
            root
        };
        let root = if dense {
            dense_fixture(root, tokens)
        } else {
            root
        };
        let baseline =
            crate::expert_device::with_test_policy(false, || crate::compile(vec![root.clone()], 0))
                .unwrap();
        let mut candidate =
            crate::expert_device::with_test_policy(true, || crate::compile(vec![root.clone()], 0))
                .unwrap();
        if dense {
            assert_eq!(candidate.enable_dense_overlap_for_test().unwrap(), 1);
        }
        let candidate = Arc::new(candidate);
        assert_eq!(device_count(&baseline), 0);
        assert_eq!(device_count(&candidate), 2);
        if crate::executable::expert_pair61::enabled() && !finalized {
            assert_eq!(candidate.graph61_test_counts().0, 1);
        }
        if finalized {
            assert!(candidate
                .diagnostics()
                .instructions
                .iter()
                .any(|i| i.kind == "et_ordered_sorted_reduce"));
        }
        for executable in [&baseline, candidate.as_ref()] {
            assert!(executable
                .diagnostics()
                .instructions
                .iter()
                .any(|i| i.kind == "group_sorted_activation"));
        }
        let rows = tokens * 8;
        let x = host(
            vec![rows, 2816],
            DType::BF16,
            &(0..rows * 2816)
                .map(|i| ((i * 31 % 103) as f64 - 51.) / 128.)
                .collect::<Vec<_>>(),
        );
        let first_bank = host(
            vec![128, 1, 1],
            DType::BF16,
            &(0..128)
                .map(|i| (i as f64 - 63.) / 8192.)
                .collect::<Vec<_>>(),
        );
        let second_bank = host(
            vec![128, 1, 1],
            DType::BF16,
            &(0..128)
                .map(|i| (65. - i as f64) / 8192.)
                .collect::<Vec<_>>(),
        );
        let bindings = |scores: &[f64]| {
            vec![
                x.clone(),
                host(vec![tokens, 128], DType::F32, scores),
                first_bank.clone(),
                second_bank.clone(),
            ]
        };
        let mut concurrent = Vec::new();
        for pattern in 0..3 {
            let scores = (0..tokens * 128)
                .map(|i| {
                    let token = i / 128;
                    let expert = i % 128;
                    match pattern {
                        0 => 0., // stable ties: eight experts with exactly T rows
                        1 => {
                            if (expert + 128 - token * 8 % 128) % 128 < 8 {
                                1.
                            } else {
                                -1.
                            }
                        }
                        _ => {
                            if (token == 0 && expert >= 120) || (token > 0 && expert < 8) {
                                1.
                            } else {
                                -1.
                            }
                        } // eight M1 experts
                    }
                })
                .collect::<Vec<_>>();
            let inputs = bindings(&scores);
            let expected = baseline
                .execute(&inputs, &[], &CancellationFlag::new())
                .unwrap()[0]
                .read_storage_bytes()
                .unwrap();
            let actual = candidate
                .execute(&inputs, &[], &CancellationFlag::new())
                .unwrap()
                .remove(0);
            assert_eq!(
                actual.read_storage_bytes().unwrap(),
                expected,
                "T{tokens} pattern{pattern}"
            );
            retained.push((actual, expected.clone()));
            if crate::executable::expert_pair61::enabled() && !finalized {
                assert_eq!(candidate.graph61_test_counts(), (1, 1));
                if pattern == 0 {
                    use crate::executable::expert_pair61::{with_fault, Fault};
                    for (fault, message) in [
                        (Fault::BeforeLaunch, "injected graph61 before launch"),
                        (Fault::AfterLaunch, "injected graph61 after launch"),
                        (Fault::CancelAfterLaunch, "operation aborted"),
                    ] {
                        let error = with_fault(fault, || {
                            candidate.execute(&inputs, &[], &CancellationFlag::new())
                        })
                        .err()
                        .unwrap();
                        assert_eq!(error, message);
                        assert_eq!(
                            candidate
                                .execute(&inputs, &[], &CancellationFlag::new())
                                .unwrap()[0]
                                .read_storage_bytes()
                                .unwrap(),
                            expected
                        );
                    }
                }
            }

            if pattern == 0 {
                for after_merged in [true, false] {
                    let result = crate::device::with_device_expert_failure(after_merged, || {
                        candidate.execute(&inputs, &[], &CancellationFlag::new())
                    });
                    assert!(
                        matches!(result, Err(error) if error.contains("injected expert submission failure"))
                    );
                    assert_eq!(
                        candidate
                            .execute(&inputs, &[], &CancellationFlag::new())
                            .unwrap()[0]
                            .read_storage_bytes()
                            .unwrap(),
                        expected
                    );
                }
            }
            if pattern < 2 {
                concurrent.push((inputs.clone(), expected.clone()));
            }
            let cancelled = CancellationFlag::new();
            cancelled.cancel();
            assert!(
                matches!(candidate.execute(&inputs, &[], &cancelled), Err(e) if e == "operation aborted")
            );
            let interrupted = CancellationFlag::new();
            assert!(
                matches!(candidate.execute_with_gemm_hook(&inputs, &interrupted, &|| interrupted.cancel()), Err(e) if e == "operation aborted")
            );
            assert_eq!(
                candidate
                    .execute(&inputs, &[], &CancellationFlag::new())
                    .unwrap()[0]
                    .read_storage_bytes()
                    .unwrap(),
                expected
            );
        }
        let mut nan_scores = vec![0.; tokens * 128];
        nan_scores[17] = f64::NAN;
        let invalid = bindings(&nan_scores);
        let baseline_error = baseline
            .execute(&invalid, &[], &CancellationFlag::new())
            .err()
            .expect("TopK NaN must fail");
        if crate::executable::expert_pair61::enabled() && !finalized {
            use crate::executable::expert_pair61::{with_fault, Fault};
            assert_eq!(
                with_fault(Fault::AfterLaunch, || candidate.execute(
                    &invalid,
                    &[],
                    &CancellationFlag::new()
                ))
                .err()
                .unwrap(),
                baseline_error
            );
            assert_eq!(
                with_fault(Fault::BeforeLaunch, || candidate.execute(
                    &invalid,
                    &[],
                    &CancellationFlag::new()
                ))
                .err()
                .unwrap(),
                "injected graph61 before launch"
            );
        }
        let candidate_error = candidate
            .execute(&invalid, &[], &CancellationFlag::new())
            .err()
            .expect("TopK NaN must fail");
        assert_eq!(candidate_error, baseline_error);
        // A later missing binding must not replace the earlier deferred TopK
        // error once the baseline's host-validation checkpoint was skipped.
        let late = node(NodeKind::Neg {
            a: node(NodeKind::Input {
                slot: 4,
                shape: vec![1],
                dtype: DType::F32,
                device: Device::Cuda(0),
                storage: StorageMetadata::dense(),
            }),
        });
        let late_baseline = crate::expert_device::with_test_policy(false, || {
            crate::compile(vec![root.clone(), late.clone()], 0)
        })
        .unwrap();
        let late_candidate = crate::expert_device::with_test_policy(true, || {
            crate::compile(vec![root.clone(), late.clone()], 0)
        })
        .unwrap();
        assert_eq!(device_count(&late_candidate), 2);
        assert_eq!(
            late_baseline
                .execute(&invalid, &[], &CancellationFlag::new())
                .err()
                .unwrap(),
            baseline_error
        );
        assert_eq!(
            late_candidate
                .execute(&invalid, &[], &CancellationFlag::new())
                .err()
                .unwrap(),
            baseline_error
        );
        let failed_status_read = crate::expert_device::with_failed_status_read(|| {
            late_candidate.execute(&invalid, &[], &CancellationFlag::new())
        });
        assert!(
            matches!(failed_status_read, Err(error) if error.contains("missing CUDA binding 4"))
        );

        // An error produced after the last first-group validation checkpoint
        // must not displace a later host binding error. Reuse does not advance it.
        let later_gpu_error = node(NodeKind::IndexSelect {
            a: root.clone(),
            dim: 0,
            indexes: full(&[1], DType::I64, 999999.),
        });
        let after_baseline = crate::expert_device::with_test_policy(false, || {
            crate::compile(vec![later_gpu_error.clone(), late.clone()], 0)
        })
        .unwrap();
        let after_candidate = crate::expert_device::with_test_policy(true, || {
            crate::compile(vec![later_gpu_error, late], 0)
        })
        .unwrap();
        assert_eq!(device_count(&after_candidate), 2);
        let valid = bindings(&vec![0.; tokens * 128]);
        let host_error = after_baseline
            .execute(&valid, &[], &CancellationFlag::new())
            .err()
            .unwrap();
        assert!(
            host_error.contains("missing CUDA binding 4"),
            "{host_error}"
        );
        assert_eq!(
            after_candidate
                .execute(&valid, &[], &CancellationFlag::new())
                .err()
                .unwrap(),
            host_error
        );
        let workers = concurrent
            .into_iter()
            .map(|(inputs, expected)| {
                let candidate = candidate.clone();
                std::thread::spawn(move || {
                    assert_eq!(
                        candidate
                            .execute(&inputs, &[], &CancellationFlag::new())
                            .unwrap()[0]
                            .read_storage_bytes()
                            .unwrap(),
                        expected
                    );
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    }
    for (value, expected) in retained {
        assert_eq!(value.read_storage_bytes().unwrap(), expected);
    }
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_EXPERT_DEVICE=1"]
fn device_expert_invalid_metadata_sanitizes_maps_outputs_and_preserves_first_status() {
    use crate::executable::CudaKernelArgs;
    use cudarc::driver::{DevicePtr, LaunchConfig, PushKernelArg};
    let device = crate::CudaDevice::get(0).unwrap();
    let stream = &device.stream;
    let total = 512usize;
    for case in 0..10 {
        let mut control = vec![0u32; 130];
        for expert in 0..128 {
            control[expert + 2] = ((expert + 1).min(8) * 64) as u32;
        }
        match case {
            0 => control[0] = 6,
            1 => control[1] = 1,
            2 => control[129] = 511,
            3 => control[4] = 1,
            4 => control[2] = u32::MAX,
            5 => control[2] = 513,
            7 => {
                control[2] = 257;
                control[3..].fill(512);
            }
            8 => control[2..].fill(512),
            9 => {
                control[2] = 65;
            }
            _ => {}
        }
        let prior = if case == 6 { (3u64 << 32) | 5 } else { 0 };
        let control = stream.clone_htod(&control).unwrap();
        let status = stream.clone_htod(&[prior]).unwrap();
        let descriptors = stream.clone_htod(&vec![u64::MAX; 1024]).unwrap();
        let shapes = stream.clone_htod(&vec![u32::MAX; 384]).unwrap();
        let m1 = stream.clone_htod(&vec![u32::MAX; 129]).unwrap();
        let rows = stream.clone_htod(&vec![u32::MAX; total + 4]).unwrap();
        let inverse = stream.clone_htod(&vec![u32::MAX; total + 4]).unwrap();
        let output = stream
            .clone_htod(&vec![0xffffu16; total * 1408 + 4])
            .unwrap();
        let ptr = |buffer: &cudarc::driver::CudaSlice<u32>| buffer.device_ptr(stream).0;
        let mut args = CudaKernelArgs::default();
        args.inputs[..5].copy_from_slice(&[ptr(&control), 16, 16, ptr(&rows), ptr(&inverse)]);
        args.output = output.device_ptr(stream).0;
        args.scratch = [
            descriptors.device_ptr(stream).0,
            ptr(&shapes),
            ptr(&m1),
            status.device_ptr(stream).0,
        ];
        args.integers[..4].copy_from_slice(&[1408, 2816, total as u64, 1]);
        args.elements = (total * 1408) as u64;
        args.error_context = 7;
        for (name, grid) in [
            ("et_expert_device_metadata59", 1),
            ("et_expert_device_sanitize59", 32),
        ] {
            let mut launch = stream.launch_builder(device.kernel(name).unwrap());
            launch.arg(&args);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .unwrap();
        }
        assert_eq!(
            stream.clone_dtoh(&status).unwrap()[0],
            if prior == 0 { (7 << 32) | 1 } else { prior }
        );
        assert!(stream
            .clone_dtoh(&descriptors)
            .unwrap()
            .iter()
            .all(|v| *v == 0));
        assert!(stream.clone_dtoh(&m1).unwrap().iter().all(|v| *v == 0));
        assert!(stream
            .clone_dtoh(&shapes)
            .unwrap()
            .chunks_exact(3)
            .all(|s| s == [0, 1408, 2816]));
        for map in [&rows, &inverse] {
            let map = stream.clone_dtoh(map).unwrap();
            assert_eq!(&map[..total], &(0..total as u32).collect::<Vec<_>>());
            assert_eq!(&map[total..], &[u32::MAX; 4]);
        }
        let output = stream.clone_dtoh(&output).unwrap();
        assert!(output[..total * 1408].iter().all(|v| *v == 0));
        assert_eq!(&output[total * 1408..], &[0xffff; 4]);
    }
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_EXPERT_DEVICE=1"]
fn device_expert_deferred_failure_restores_append_and_readonly_state() {
    use crate::{CudaSequenceState, CudaStateInvocation, CudaStateLayout, CudaValue};
    use effect_torch_runtime::{CancellationFlag, KvLayerDescriptor, StateAccessMode};
    let device = crate::CudaDevice::get(0).unwrap();
    let attention = node(NodeKind::KvAttention {
        q: full(&[1, 1, 1, 2], DType::BF16, 0.),
        k: full(&[1, 1, 1, 2], DType::BF16, 1.),
        v: full(&[1, 1, 1, 2], DType::BF16, 2.),
        scale: 1.,
        layer: 0,
        window: None,
        mode: effect_torch_graph::KvAttentionMode::Causal,
        rounding: effect_torch_graph::AttentionRounding::Stepwise,
    });
    let mut layout = CudaStateLayout {
        capacity: 4,
        dtype: DType::BF16,
        slots: 1,
        packed_rows_per_sequence: None,
        access: StateAccessMode::Append,
        kv_layers: vec![KvLayerDescriptor {
            layer_id: 0,
            kv_heads: 1,
            head_dim: 2,
            dtype: DType::BF16,
            retention: None,
        }],
    };
    let compile = |roots, layout| {
        crate::compile_stateful_with_layout(
            roots,
            0,
            20,
            true,
            effect_torch_compiler::CompileOptions::default(),
            layout,
        )
        .unwrap()
    };
    let mut state = CudaStateInvocation {
        sequences: vec![CudaSequenceState {
            cursor: 0,
            keys: vec![],
            values: vec![],
            kda_states: vec![],
            conv_states: vec![],
            kv_storage: None,
        }],
        slots: vec![0],
        valid_lengths: vec![1],
        capacity: 4,
        cache_dtype: DType::BF16,
        packed_rows_per_sequence: None,
        kv_layers: layout.kv_layers.clone(),
        access: StateAccessMode::Append,
        cache: None,
    };
    let seed = compile(vec![attention.clone()], layout.clone());
    seed.execute_stateful(&[], &[], &mut state, &CancellationFlag::new())
        .unwrap();
    seed.readback_state(&mut state).unwrap();
    let parent = state.sequences[0].kv_storage.clone().unwrap();
    let bytes = |snapshot: &crate::CudaKvSnapshot| {
        snapshot
            .layers
            .iter()
            .flat_map(|layer| {
                layer.pages.iter().flat_map(|page| {
                    [
                        device.stream.clone_dtoh(&page.keys).unwrap(),
                        device.stream.clone_dtoh(&page.values).unwrap(),
                    ]
                })
            })
            .collect::<Vec<_>>()
    };
    let before = bytes(&parent);
    let host = |shape, dtype, data: &[f64]| {
        CudaValue::from_host(device.clone(), shape, dtype, data).unwrap()
    };
    let mut inputs = vec![
        host(vec![512, 2816], DType::BF16, &vec![0.01; 512 * 2816]),
        host(vec![64, 128], DType::F32, &vec![f64::NAN; 64 * 128]),
        host(vec![128, 1, 1], DType::BF16, &vec![0.001; 128]),
        host(vec![128, 1, 1], DType::BF16, &vec![0.002; 128]),
    ];
    for access in [StateAccessMode::Append, StateAccessMode::ReadOnly] {
        layout.access = access;
        state.access = access;
        state.sequences[0].cursor = 1;
        state.sequences[0].kv_storage = Some(parent.clone());
        state.cache = None;
        let executable = compile(vec![attention.clone(), expert_fixture(64)], layout.clone());
        assert_eq!(
            executable
                .diagnostics()
                .instructions
                .iter()
                .filter(|i| i.kind == "grouped_expert_linear_rows_device59_non_capturable")
                .map(|i| i.count)
                .sum::<usize>(),
            2
        );
        let touched = std::sync::atomic::AtomicUsize::new(0);
        let error = executable
            .execute_stateful_with_kv_hook(&inputs, &mut state, &CancellationFlag::new(), &|| {
                touched.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            })
            .err()
            .unwrap();
        assert_eq!(error, "topKIndices: NaN input");
        assert_eq!(touched.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(state.sequences[0].cursor, 1);
        assert_eq!(
            bytes(state.sequences[0].kv_storage.as_ref().unwrap()),
            before
        );
        assert_eq!(bytes(&state.cache.as_ref().unwrap().sequences[0]), before);
        if crate::executable::expert_pair61::enabled() {
            // No hook: this failure must traverse the actual graph path.
            assert_eq!(
                executable
                    .execute_stateful(&inputs, &[], &mut state, &CancellationFlag::new())
                    .err()
                    .unwrap(),
                "topKIndices: NaN input"
            );
            assert_eq!(executable.graph61_test_counts(), (1, 1));
            assert_eq!(
                bytes(state.sequences[0].kv_storage.as_ref().unwrap()),
                before
            );
            inputs[1] = host(vec![64, 128], DType::F32, &vec![0.; 64 * 128]);
            use crate::executable::expert_pair61::{with_fault, Fault};
            assert_eq!(
                with_fault(Fault::AfterLaunch, || executable.execute_stateful(
                    &inputs,
                    &[],
                    &mut state,
                    &CancellationFlag::new()
                ))
                .err()
                .unwrap(),
                "injected graph61 after launch"
            );
            assert_eq!(
                bytes(state.sequences[0].kv_storage.as_ref().unwrap()),
                before
            );
            assert_eq!(state.sequences[0].cursor, 1);
        }
        inputs[1] = host(vec![64, 128], DType::F32, &vec![0.; 64 * 128]);
        executable
            .execute_stateful(&inputs, &[], &mut state, &CancellationFlag::new())
            .unwrap();
        assert_eq!(bytes(&parent), before);
        inputs[1] = host(vec![64, 128], DType::F32, &vec![f64::NAN; 64 * 128]);
    }
}

#[test]
#[ignore = "requires CUDA, accepted59 expert flags, EXPERT_PAIR_GRAPH61=1 and EXPERT_FINALIZE=1"]
fn explicit_pair61_exact_topk_dense_errors_lifetimes_and_state() {
    assert!(crate::executable::expert_pair61::enabled());
    assert_eq!(
        std::env::var("EFFECT_TORCH_CUDA_CACHE_SEGMENT_BUFFERS").as_deref(),
        Ok("0")
    );
    // Same source first compiled by ordinary fused lowering, then requested by
    // grouped lowering. Keep the exact retained NVRTC image; no recompilation.
    let device = crate::CudaDevice::get(0).unwrap();
    let source = r#"extern "C" __global__ void et_fused_elementwise(CudaKernelArgs a) { /* graph61 cache order */ if (et_thread() < a.elements) et_store(a.output, a.output_dtype, et_thread(), 1.0f); }"#;
    device.fused_elementwise(source).unwrap();
    assert!(device.graph61_fused(source).is_none());
    let image = device.graph61_retained_fused_image(source).unwrap();
    device.fused_elementwise_with_graph61(source, true).unwrap();
    assert!(device.graph61_fused(source).is_some());
    assert!(Arc::ptr_eq(
        &image,
        &device.graph61_retained_fused_image(source).unwrap()
    ));
    device_expert_real_topk_exact_outputs_errors_retention_and_recovery();
    device_expert_deferred_failure_restores_append_and_readonly_state();
}

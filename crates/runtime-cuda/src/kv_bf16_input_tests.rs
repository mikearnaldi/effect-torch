use super::*;

fn compile_policy(layout: CudaStateLayout, root: Arc<Node>, enabled: bool) -> CudaExecutable {
    crate::lowering::with_kv_bf16_input_test_policy(enabled, || compile(layout, root))
}
fn state_bytes(state: &CudaStateInvocation) -> Vec<(u32, u32, Vec<u8>, Vec<u8>)> {
    let device = CudaDevice::get(0).unwrap();
    state
        .sequences
        .iter()
        .flat_map(|sequence| {
            sequence.kv_storage.iter().flat_map(|snapshot| {
                snapshot.layers.iter().flat_map(|layer| {
                    layer.pages.iter().map(|page| {
                        (
                            sequence.cursor,
                            page.start,
                            device.stream.clone_dtoh(&page.keys).unwrap(),
                            device.stream.clone_dtoh(&page.values).unwrap(),
                        )
                    })
                })
            })
        })
        .collect()
}
fn assert_input_policy(executable: &CudaExecutable, enabled: bool) {
    let conversions = executable
        .diagnostics()
        .instructions
        .iter()
        .find(|instruction| instruction.kind == "et_convert")
        .unwrap()
        .count;
    assert_eq!(conversions, if enabled { 1 } else { 4 });
    let kv = executable
        .commands
        .iter()
        .find(|command| {
            matches!(
                command.kind,
                CommandKind::Kernel {
                    kv_matmul: Some(_),
                    ..
                }
            )
        })
        .unwrap();
    if let CommandKind::Kernel { args, .. } = &kv.kind {
        assert_eq!(args.compute_dtype, dtype_code(DType::F32));
        assert_eq!(args.output_dtype, dtype_code(DType::F32));
        assert_eq!(&args.input_dtypes[..3], &[if enabled { 3 } else { 1 }; 3]);
    }
}

#[test]
#[ignore = "requires CUDA"]
fn kv_bf16_inputs_exact_all_payloads_and_state_retention() {
    for access in [StateAccessMode::Append, StateAccessMode::ReadOnly] {
        let (tokens, dim) = (256, 256);
        let layout = layout(1, 1024, 1, dim, None, access);
        let graph = root(
            1,
            1,
            1,
            tokens,
            dim,
            access == StateAccessMode::ReadOnly,
            None,
            0.30157,
        );
        let baseline = compile_policy(layout.clone(), graph.clone(), false);
        let candidate = compile_policy(layout.clone(), graph, true);
        assert_input_policy(&baseline, false);
        assert_input_policy(&candidate, true);
        let mut retained = Vec::new();
        for pattern in 0..2 {
            let bindings = (0..3)
                .map(|role| {
                    let bytes = (0..65536u32)
                        .flat_map(|i| {
                            let bits = if pattern == 0 {
                                (i as u16).wrapping_add(role * 173)
                            } else {
                                half::bf16::from_f32(
                                    ((i + u32::from(role)) % 31) as f32 / 31.0 - 0.5,
                                )
                                .to_bits()
                            };
                            bits.to_le_bytes()
                        })
                        .collect::<Vec<_>>();
                    CudaValue::from_dense_bytes(
                        CudaDevice::get(0).unwrap(),
                        vec![1, 1, tokens, dim],
                        DType::BF16,
                        &bytes,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            for valid in [256, 17, 0] {
                let make = || {
                    invocation(
                        &layout,
                        vec![sequence(0, snapshot(layout.kv_layers[0], 0, 0, &[], &[]))],
                        vec![0],
                        vec![valid],
                    )
                };
                let mut off = make();
                let mut on = make();
                let expected = baseline
                    .execute_stateful(&bindings, &[], &mut off, &CancellationFlag::new())
                    .unwrap();
                let actual = candidate
                    .execute_stateful(&bindings, &[], &mut on, &CancellationFlag::new())
                    .unwrap();
                let expected_bytes = expected[0].read_storage_bytes().unwrap();
                assert_eq!(
                    actual[0].read_storage_bytes().unwrap(),
                    expected_bytes,
                    "access={access:?} pattern={pattern} valid={valid}"
                );
                assert_eq!(state_bytes(&on), state_bytes(&off));
                retained.push((actual, expected_bytes));
            }
        }
        drop(candidate);
        drop(baseline);
        for (actual, expected) in retained {
            assert_eq!(actual[0].read_storage_bytes().unwrap(), expected);
        }
    }
}

#[test]
#[ignore = "requires CUDA"]
fn kv_bf16_inputs_cancel_failure_rollback_then_recover() {
    let layout = layout(1, 32, 1, 8, None, StateAccessMode::Append);
    let candidate = compile_policy(layout.clone(), root(1, 1, 1, 4, 8, false, None, 1.0), true);
    let baseline = compile_policy(layout.clone(), root(1, 1, 1, 4, 8, false, None, 1.0), false);
    let bindings = [
        host(vec![1, 1, 4, 8], &[0.; 32]),
        host(vec![1, 1, 4, 8], &[0.; 32]),
        host(vec![1, 1, 4, 8], &[1.; 32]),
    ];
    let make = || {
        invocation(
            &layout,
            vec![sequence(0, snapshot(layout.kv_layers[0], 0, 0, &[], &[]))],
            vec![0],
            vec![4],
        )
    };
    let mut state = make();
    let original = state_bytes(&state);
    let cancelled = CancellationFlag::new();
    let result =
        candidate.execute_stateful_with_kv_hook(&bindings, &mut state, &cancelled, &|| {
            cancelled.cancel()
        });
    assert!(result.is_err());
    assert_eq!(state_bytes(&state), original);
    assert_eq!(state.sequences[0].cursor, 0);
    state.valid_lengths[0] = 5;
    assert!(candidate
        .execute_stateful(&bindings, &[], &mut state, &CancellationFlag::new())
        .is_err());
    assert_eq!(state_bytes(&state), original);
    state.valid_lengths[0] = 4;
    let actual = candidate
        .execute_stateful(&bindings, &[], &mut state, &CancellationFlag::new())
        .unwrap();
    let mut reference = make();
    let expected = baseline
        .execute_stateful(&bindings, &[], &mut reference, &CancellationFlag::new())
        .unwrap();
    assert_eq!(
        actual[0].read_storage_bytes().unwrap(),
        expected[0].read_storage_bytes().unwrap()
    );
    assert_eq!(state_bytes(&state), state_bytes(&reference));
}

#[test]
#[ignore = "requires CUDA"]
fn kv_bf16_inputs_gather_matches_production_conversion_for_every_payload() {
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    let device = CudaDevice::get(0).unwrap();
    let bytes = (0..=u16::MAX)
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let source =
        CudaValue::from_dense_bytes(device.clone(), vec![1, 1, 256, 256], DType::BF16, &bytes)
            .unwrap();
    let widened =
        CudaValue::from_dense_bytes(device.clone(), vec![65536], DType::F32, &vec![0; 65536 * 4])
            .unwrap();
    let expected = CudaValue::from_dense_bytes(
        device.clone(),
        vec![65536],
        DType::BF16,
        &vec![0xa5; 65536 * 2],
    )
    .unwrap();
    let launch = |name: &str, args: &CudaKernelArgs| {
        let mut launch = device.stream.launch_builder(device.kernel(name).unwrap());
        launch.arg(args);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .unwrap();
    };
    let mut conversion = CudaKernelArgs {
        output: widened.storage_address(),
        elements: 65536,
        output_dtype: 1,
        ..Default::default()
    };
    conversion.inputs[0] = source.storage_address();
    conversion.input_dtypes[0] = 3;
    launch("et_convert", &conversion);
    // Zero prefix positions isolates the actual production Q packing branch;
    // the K/V store branches are exhaustively compared via state bytes above.
    let mut gather = CudaKernelArgs::default();
    gather.inputs[0] = widened.storage_address();
    gather.input_dtypes[0] = 1;
    gather.inputs[5] = expected.storage_address();
    gather.integers[7] = 256;
    gather.integers[10] = 256;
    gather.integers[11] = 1;
    gather.integers[15] = 256;
    launch("et_kv_gemm_gather", &gather);
    let expected_bytes = expected.read_storage_bytes().unwrap();
    let poison = expected_bytes
        .iter()
        .map(|byte| byte ^ 0xff)
        .collect::<Vec<_>>();
    let actual =
        CudaValue::from_dense_bytes(device.clone(), vec![65536], DType::BF16, &poison).unwrap();
    gather.inputs[0] = source.storage_address();
    gather.input_dtypes[0] = 3;
    gather.inputs[5] = actual.storage_address();
    launch("et_kv_gemm_gather", &gather);
    assert_eq!(actual.read_storage_bytes().unwrap(), expected_bytes);
    assert_eq!(source.read_storage_bytes().unwrap(), bytes);
}

#[test]
#[ignore = "requires CUDA"]
fn kv_bf16_inputs_views_shared_alias_concurrency_and_retained_outputs() {
    for view in [false, true] {
        let source = input(0, &[1, 1, 5, 9], DType::BF16);
        let sliced = Node::new(NodeKind::Slice {
            a: source,
            ranges: vec![(0, 1, 1), (0, 1, 1), (1, 5, 1), (1, 9, 1)],
        })
        .unwrap();
        let value = if view {
            Node::new(NodeKind::Permute {
                a: sliced,
                dims: vec![0, 1, 3, 2],
            })
            .unwrap()
        } else {
            sliced
        };
        let (tokens, dim) = if view { (8, 4) } else { (4, 8) };
        let graph = Node::new(NodeKind::KvAttention {
            q: value.clone(),
            k: value.clone(),
            v: value,
            scale: 0.3,
            layer: 0,
            window: Some(3),
            mode: KvAttentionMode::Causal,
            rounding: AttentionRounding::Stepwise,
        })
        .unwrap();
        let layout = layout(1, 32, 1, dim, Some(2), StateAccessMode::Append);
        let baseline = compile_policy(layout.clone(), graph.clone(), false);
        let candidate = Arc::new(compile_policy(layout.clone(), graph, true));
        let values = (0..45)
            .map(|i| ((i % 11) as f64 - 5.) / 16.)
            .collect::<Vec<_>>();
        let bindings = vec![host(vec![1, 1, 5, 9], &values)];
        let make = || {
            invocation(
                &layout,
                vec![sequence(0, snapshot(layout.kv_layers[0], 0, 0, &[], &[]))],
                vec![0],
                vec![tokens as u32],
            )
        };
        let mut reference = make();
        let expected = baseline
            .execute_stateful(&bindings, &[], &mut reference, &CancellationFlag::new())
            .unwrap()[0]
            .read_storage_bytes()
            .unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let mut workers = Vec::new();
        for _ in 0..2 {
            let executable = candidate.clone();
            let inputs = bindings.clone();
            let mut state = make();
            let barrier = barrier.clone();
            workers.push(std::thread::spawn(move || {
                barrier.wait();
                let output = executable
                    .execute_stateful(&inputs, &[], &mut state, &CancellationFlag::new())
                    .unwrap();
                (output, state_bytes(&state))
            }));
        }
        barrier.wait();
        let results = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        drop(candidate);
        drop(baseline);
        drop(bindings);
        for (output, state) in results {
            assert_eq!(output[0].read_storage_bytes().unwrap(), expected);
            assert_eq!(state, state_bytes(&reference));
        }
    }
}

#[test]
#[ignore = "requires CUDA"]
fn attention82_unsupported_native_geometry_keeps_bf16_fallback_lanes_and_cache() {
    for access in [StateAccessMode::Append, StateAccessMode::ReadOnly] {
        let layout = layout(2, 32, 1, 8, None, access);
        let graph = root(
            2,
            1,
            1,
            3,
            8,
            access == StateAccessMode::ReadOnly,
            None,
            0.3,
        );
        let baseline = crate::lowering::with_attention82_test_policy(false, || {
            compile_policy(layout.clone(), graph.clone(), false)
        });
        let candidate = crate::lowering::with_attention82_test_policy(true, || {
            compile_policy(layout.clone(), graph, false)
        });
        let bindings = (0..3)
            .map(|role| {
                let data = (0..48)
                    .map(|i| ((i * 7 + role * 11) % 23) as f64 / 16.0 - 0.5)
                    .collect::<Vec<_>>();
                host(vec![2, 1, 3, 8], &data)
            })
            .collect::<Vec<_>>();
        let make = || {
            invocation(
                &layout,
                vec![
                    sequence(0, snapshot(layout.kv_layers[0], 0, 0, &[], &[])),
                    sequence(0, snapshot(layout.kv_layers[0], 0, 0, &[], &[])),
                ],
                vec![0, 1],
                vec![3, 1],
            )
        };
        let mut off = make();
        let mut on = make();
        let expected = baseline
            .execute_stateful(&bindings, &[], &mut off, &CancellationFlag::new())
            .unwrap();
        let actual = candidate
            .execute_stateful(&bindings, &[], &mut on, &CancellationFlag::new())
            .unwrap();
        assert_eq!(
            actual[0].read_storage_bytes().unwrap(),
            expected[0].read_storage_bytes().unwrap()
        );
        assert_eq!(state_bytes(&on), state_bytes(&off));
        assert!(candidate
            .commands
            .iter()
            .any(|command| matches!(&command.kind,
            CommandKind::Kernel { name: "et_kv_attention", args, kv_matmul: Some(_), .. }
            if args.output_dtype == 3 && args.integers[10] == 8)));
    }
}

#[test]
#[ignore = "requires CUDA"]
fn attention82_fallback_cancel_failure_rollback_then_recover() {
    crate::lowering::with_attention82_test_policy(
        true,
        kv_bf16_inputs_cancel_failure_rollback_then_recover,
    );
}

#[test]
#[ignore = "requires CUDA SM120 and ATTENTION75_DIRECTORY with ABI2 variants"]
fn attention82_native_parity_multilane_partial_layout_retention_and_cancel() {
    let device = CudaDevice::get(0).unwrap();
    assert!(
        device.cublas.attention75().is_some(),
        "ABI2 native variants required"
    );
    let mut retained = Vec::new();
    for dim in [256, 512] {
        for access in [StateAccessMode::Append, StateAccessMode::ReadOnly] {
            for token_major in [false, true] {
                let (tokens, heads, kv, prefix) = (256, 16, 2, 32);
                let layout = layout(2, 512, kv, dim, None, access);
                let graph = root(
                    2,
                    heads,
                    kv,
                    tokens,
                    dim,
                    access == StateAccessMode::ReadOnly,
                    if dim == 256 { Some(23) } else { None },
                    0.03125,
                );
                let graph = if token_major {
                    Node::new(NodeKind::Permute {
                        a: graph,
                        dims: vec![0, 2, 1, 3],
                    })
                    .unwrap()
                } else {
                    graph
                };
                let baseline = crate::lowering::with_attention82_test_policy(false, || {
                    compile_policy(layout.clone(), graph.clone(), false)
                });
                let candidate = crate::lowering::with_attention82_test_policy(true, || {
                    compile_policy(layout.clone(), graph, false)
                });
                for (executable, dtype) in [(&baseline, 1), (&candidate, 3)] {
                    let args = executable
                        .commands
                        .iter()
                        .find_map(|command| match &command.kind {
                            CommandKind::Kernel {
                                name: "et_kv_attention",
                                args,
                                ..
                            } => Some(args),
                            _ => None,
                        })
                        .unwrap();
                    assert_eq!(args.output_dtype, dtype);
                    assert_eq!(args.operation, u32::from(token_major));
                }
                let bindings = (0..3)
                    .map(|role| {
                        let h = if role == 0 { heads } else { kv };
                        let data = (0..2 * h * tokens * dim)
                            .map(|i| (((i * 17 + role * 23) % 127) as f64 - 63.0) / 128.0)
                            .collect::<Vec<_>>();
                        host(vec![2, h, tokens, dim], &data)
                    })
                    .collect::<Vec<_>>();
                let keys = encoded(
                    &(0..prefix * kv * dim)
                        .map(|i| ((i % 23) as f64 - 11.0) / 64.0)
                        .collect::<Vec<_>>(),
                );
                let vals = encoded(
                    &(0..prefix * kv * dim)
                        .map(|i| ((i % 31) as f64 - 15.0) / 64.0)
                        .collect::<Vec<_>>(),
                );
                let make = || {
                    invocation(
                        &layout,
                        vec![
                            sequence(
                                prefix,
                                snapshot(layout.kv_layers[0], 0, prefix, &keys, &vals),
                            ),
                            sequence(
                                prefix,
                                snapshot(layout.kv_layers[0], 0, prefix, &keys, &vals),
                            ),
                        ],
                        vec![0, 1],
                        vec![tokens as u32, 17],
                    )
                };
                let mut off = make();
                let mut on = make();
                let before = crate::attention75::test_launches();
                let expected = baseline
                    .execute_stateful(&bindings, &[], &mut off, &CancellationFlag::new())
                    .unwrap();
                assert_eq!(
                    crate::attention75::test_launches() - before,
                    2,
                    "legacy75 must really launch native attention"
                );
                let before = crate::attention75::test_launches();
                let actual = candidate
                    .execute_stateful(&bindings, &[], &mut on, &CancellationFlag::new())
                    .unwrap();
                assert_eq!(
                    crate::attention75::test_launches() - before,
                    2,
                    "BF16IO must really launch native attention"
                );
                let bytes = expected[0].read_storage_bytes().unwrap();
                assert_eq!(
                    actual[0].read_storage_bytes().unwrap(),
                    bytes,
                    "dim={dim} access={access:?} token_major={token_major}"
                );
                assert_eq!(state_bytes(&on), state_bytes(&off));
                let mut interrupted = make();
                let original = state_bytes(&interrupted);
                let cancelled = CancellationFlag::new();
                let before = crate::attention75::test_launches();
                assert!(candidate
                    .execute_stateful_with_kv_hook(&bindings, &mut interrupted, &cancelled, &|| {
                        cancelled.cancel()
                    })
                    .is_err());
                assert_eq!(crate::attention75::test_launches() - before, 2);
                assert_eq!(state_bytes(&interrupted), original);
                assert_eq!(interrupted.sequences[0].cursor, prefix as u32);
                let recovered = candidate
                    .execute_stateful(&bindings, &[], &mut interrupted, &CancellationFlag::new())
                    .unwrap();
                assert_eq!(recovered[0].read_storage_bytes().unwrap(), bytes);
                retained.push((actual, bytes));
                drop(candidate);
                drop(baseline);
                drop(bindings);
            }
        }
    }
    for (outputs, expected) in retained {
        assert_eq!(outputs[0].read_storage_bytes().unwrap(), expected);
    }
}

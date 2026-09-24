use super::*;
use effect_torch_runtime::{CancellationFlag, KvLayerDescriptor, StateAccessMode};

fn layout(access: StateAccessMode, dtype: DType) -> CudaStateLayout {
    CudaStateLayout {
        capacity: 4,
        dtype,
        slots: 1,
        packed_rows_per_sequence: None,
        access,
        kv_layers: vec![
            KvLayerDescriptor {
                layer_id: 0,
                kv_heads: 1,
                head_dim: 2,
                dtype,
                retention: None,
            },
            KvLayerDescriptor {
                layer_id: 1,
                kv_heads: 2,
                head_dim: 1,
                dtype,
                retention: Some(1),
            },
        ],
    }
}
fn roots(dtype: DType, bidirectional: bool) -> Vec<Arc<Node>> {
    [(2, 1, 2), (4, 2, 1)]
        .into_iter()
        .enumerate()
        .map(|(layer, (qh, kh, dim))| {
            Node::new(NodeKind::KvAttention {
                q: input(layer as u32 * 3, &[1, qh, 2, dim], dtype),
                k: input(layer as u32 * 3 + 1, &[1, kh, 2, dim], dtype),
                v: input(layer as u32 * 3 + 2, &[1, kh, 2, dim], dtype),
                scale: 0.30157,
                layer: layer as u32,
                window: None,
                mode: if bidirectional {
                    effect_torch_graph::KvAttentionMode::BidirectionalBlock
                } else {
                    effect_torch_graph::KvAttentionMode::Causal
                },
                rounding: effect_torch_graph::AttentionRounding::Stepwise,
            })
            .unwrap()
        })
        .collect()
}
fn compile(layout: CudaStateLayout, roots: Vec<Arc<Node>>) -> crate::CudaExecutable {
    crate::compile_stateful_with_layout(roots, 0, 20, true, CompileOptions::default(), layout)
        .unwrap()
}
fn invocation(
    layout: &CudaStateLayout,
    cursor: u32,
    snapshot: Option<crate::CudaKvSnapshot>,
) -> crate::CudaStateInvocation {
    crate::CudaStateInvocation {
        sequences: vec![crate::CudaSequenceState {
            cursor,
            keys: Vec::new(),
            values: Vec::new(),
            kda_states: Vec::new(),
            conv_states: Vec::new(),
            kv_storage: snapshot,
        }],
        slots: vec![0],
        valid_lengths: vec![2],
        capacity: layout.capacity,
        cache_dtype: layout.dtype,
        packed_rows_per_sequence: None,
        kv_layers: layout.kv_layers.clone(),
        access: layout.access,
        cache: None,
    }
}
fn bindings(dtype: DType, offset: f64) -> Vec<crate::CudaValue> {
    let device = crate::CudaDevice::get(0).expect("CUDA device 0 is required for this test");
    [(2, 1, 2), (4, 2, 1)]
        .into_iter()
        .flat_map(|(qh, kh, dim)| {
            (0..3).map({
                let device = device.clone();
                move |role| {
                    let shape = vec![1, if role == 0 { qh } else { kh }, 2, dim];
                    let n = shape.iter().product();
                    let values = if role == 2 {
                        (0..n).map(|i| i as f64 + 1.0 + offset).collect::<Vec<_>>()
                    } else {
                        vec![0.0; n]
                    };
                    crate::CudaValue::from_host(device.clone(), shape, dtype, &values).unwrap()
                }
            })
        })
        .collect()
}
fn bytes(snapshot: &crate::CudaKvSnapshot) -> Vec<Vec<u8>> {
    let device = crate::CudaDevice::get(0).unwrap();
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
        .collect()
}

#[test]
fn heterogeneous_paged_lowering_keeps_retention_and_stepwise_boundaries() {
    for dtype in [DType::F16, DType::BF16] {
        for access in [StateAccessMode::Append, StateAccessMode::ReadOnly] {
            let (program, commands, _, _, _) =
                lower(roots(dtype, true), false, Some(layout(access, dtype)));
            let kernels = commands
                .iter()
                .filter_map(|command| match &command.kind {
                    CommandKind::Kernel {
                        name: "et_kv_attention",
                        args,
                        state_buffers,
                        ..
                    } => Some((args, state_buffers)),
                    _ => None,
                })
                .collect::<Vec<_>>();
            if access == StateAccessMode::ReadOnly {
                for instruction in &program.instructions {
                    for usage in &instruction.state {
                        if matches!(
                            program.values[usage.value.index()].decl.storage,
                            ValueStorage::Fixed {
                                class: StorageClass::PersistentState,
                                ..
                            }
                        ) {
                            assert_eq!(
                                usage.access,
                                effect_torch_compiler::ValueAccess::Read,
                                "read-only lowering cannot write persistent state"
                            );
                        }
                    }
                }
            }
            assert_eq!(kernels.len(), 2);
            for (args, buffers) in kernels {
                assert_eq!(args.integers[6], 1);
                assert_eq!(args.integers[8], u64::from(dtype_code(dtype)));
                assert_eq!(
                    args.integers[3], 0,
                    "prefix retention must not mask canvas rows"
                );
                assert!(buffers[0].is_some() && buffers[1].is_some());
            }
            let bytes: usize = program
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
                .sum();
            assert_eq!(bytes, 2 * 2 * 2 * 2 * dtype.size_in_bytes());
        }
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn retention_larger_than_capacity_preserves_rows_and_rejects_overflow() {
    let mut append_layout = layout(StateAccessMode::Append, DType::BF16);
    for layer in &mut append_layout.kv_layers {
        layer.retention = Some(1023);
    }
    let append = compile(append_layout.clone(), roots(DType::BF16, false));
    let mut state = invocation(&append_layout, 0, None);
    for cursor in [0, 2] {
        state.sequences[0].cursor = cursor;
        append
            .execute_stateful(
                &bindings(DType::BF16, 0.0),
                &[],
                &mut state,
                &CancellationFlag::new(),
            )
            .unwrap();
        append.readback_state(&mut state).unwrap();
    }
    let prefix = state.sequences[0].kv_storage.clone().unwrap();
    assert!(prefix
        .layers
        .iter()
        .all(|layer| layer.start_position == 0 && layer.descriptor.retention == Some(1023)));
    let before = bytes(&prefix);
    let mut overflow = invocation(&append_layout, 4, Some(prefix.clone()));
    assert_eq!(
        append.prepare_state(&mut overflow).unwrap_err(),
        "execute: prefix exceeds KV capacity"
    );
    let mut read_layout = append_layout;
    read_layout.access = StateAccessMode::ReadOnly;
    let read = compile(read_layout.clone(), roots(DType::BF16, true));
    let mut canvas = invocation(&read_layout, 4, Some(prefix.clone()));
    read.execute_stateful(
        &bindings(DType::BF16, 0.0),
        &[],
        &mut canvas,
        &CancellationFlag::new(),
    )
    .unwrap();
    assert_eq!(bytes(&prefix), before);
}

#[test]
#[ignore = "requires a CUDA device"]
fn paged_prefix_shared_forks_readonly_concurrency_and_full_capacity_canvas() {
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let append_layout = layout(StateAccessMode::Append, dtype);
        let append = compile(append_layout.clone(), roots(dtype, false));
        let mut state = invocation(&append_layout, 0, None);
        append
            .execute_stateful(
                &bindings(dtype, 0.0),
                &[],
                &mut state,
                &CancellationFlag::new(),
            )
            .unwrap();
        append.readback_state(&mut state).unwrap();
        let parent = state.sequences[0].kv_storage.clone().unwrap();
        let parent_bytes = bytes(&parent);
        let mut fork = invocation(&append_layout, 2, Some(parent.clone()));
        append
            .execute_stateful(
                &bindings(dtype, 4.0),
                &[],
                &mut fork,
                &CancellationFlag::new(),
            )
            .unwrap();
        append.readback_state(&mut fork).unwrap();
        let prefix = fork.sequences[0].kv_storage.clone().unwrap();
        assert!(Arc::ptr_eq(
            &parent.layers[0].pages[0],
            &prefix.layers[0].pages[0]
        ));
        assert_eq!(prefix.layers[1].start_position, 3);
        assert_eq!(bytes(&parent), parent_bytes);
        let before = bytes(&prefix);
        let read_layout = layout(StateAccessMode::ReadOnly, dtype);
        let read = Arc::new(compile(read_layout.clone(), roots(dtype, true)));
        let mut workers = Vec::new();
        for _ in 0..2 {
            let prefix = prefix.clone();
            let layout = read_layout.clone();
            let read = read.clone();
            workers.push(std::thread::spawn(move || {
                let mut state = invocation(&layout, 4, Some(prefix));
                let result = read
                    .execute_stateful(
                        &bindings(dtype, 8.0),
                        &[],
                        &mut state,
                        &CancellationFlag::new(),
                    )
                    .unwrap();
                assert_eq!(state.sequences[0].cursor, 4);
                result
            }));
        }
        let first = workers.remove(0).join().unwrap();
        let second = workers.remove(0).join().unwrap();
        assert_eq!(first[0].readback().unwrap(), second[0].readback().unwrap());
        let expected = first[0].readback().unwrap();
        for (i, value) in expected.iter().enumerate() {
            assert!((value - if i % 2 == 0 { 6.0 } else { 7.0 }).abs() < 0.04);
        }
        let mut partial = invocation(&read_layout, 4, Some(prefix.clone()));
        partial.valid_lengths[0] = 1;
        let partial = read
            .execute_stateful(
                &bindings(dtype, 18.0),
                &[],
                &mut partial,
                &CancellationFlag::new(),
            )
            .unwrap();
        assert_eq!(&partial[0].readback().unwrap()[2..4], &[0.0, 0.0]);
        assert_eq!(
            first[0].readback().unwrap(),
            expected,
            "later calls must preserve output storage"
        );
        assert_eq!(bytes(&prefix), before);
        assert_eq!(bytes(&parent), parent_bytes);
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn paged_append_and_readonly_rollback_after_attention_failure_and_cancellation() {
    for dtype in [DType::F32, DType::BF16] {
        let append_layout = layout(StateAccessMode::Append, dtype);
        let append = compile(append_layout.clone(), roots(dtype, false));
        let mut seed = invocation(&append_layout, 0, None);
        append
            .execute_stateful(
                &bindings(dtype, 0.0),
                &[],
                &mut seed,
                &CancellationFlag::new(),
            )
            .unwrap();
        append.readback_state(&mut seed).unwrap();
        let parent = seed.sequences[0].kv_storage.clone().unwrap();
        let before = bytes(&parent);
        for access in [StateAccessMode::Append, StateAccessMode::ReadOnly] {
            let layout = layout(access, dtype);
            let roots = roots(dtype, access == StateAccessMode::ReadOnly);
            let mut failing_roots = roots.clone();
            failing_roots.push(
                Node::new(NodeKind::IndexSelect {
                    a: roots[0].clone(),
                    dim: 3,
                    indexes: input(6, &[1], DType::I64),
                })
                .unwrap(),
            );
            let failing = compile(layout.clone(), failing_roots);
            let mut bindings = bindings(dtype, 10.0);
            bindings.push(
                crate::CudaValue::from_host(
                    crate::CudaDevice::get(0).unwrap(),
                    vec![1],
                    DType::I64,
                    &[99.0],
                )
                .unwrap(),
            );
            let mut state = invocation(&layout, 2, Some(parent.clone()));
            assert!(failing
                .execute_stateful(&bindings, &[], &mut state, &CancellationFlag::new())
                .is_err());
            assert_eq!(state.sequences[0].cursor, 2);
            assert_eq!(
                bytes(state.sequences[0].kv_storage.as_ref().unwrap()),
                before
            );
            assert_eq!(bytes(&state.cache.as_ref().unwrap().sequences[0]), before);
            let executable = compile(layout.clone(), roots);
            let cancelled = CancellationFlag::new();
            let reached = std::sync::atomic::AtomicUsize::new(0);
            let error = executable
                .execute_stateful_with_kv_hook(&bindings[..6], &mut state, &cancelled, &|| {
                    reached.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    cancelled.cancel();
                })
                .err()
                .unwrap();
            assert_eq!(
                reached.load(std::sync::atomic::Ordering::Relaxed),
                1,
                "interrupt after actual KV device work"
            );
            assert!(error.contains("aborted"));
            assert_eq!(state.sequences[0].cursor, 2);
            assert_eq!(bytes(&parent), before);
            assert_eq!(bytes(&state.cache.as_ref().unwrap().sequences[0]), before);
            let result = executable
                .execute_stateful(&bindings[..6], &[], &mut state, &CancellationFlag::new())
                .unwrap();
            assert!(
                !result.is_empty(),
                "scratch must be reusable after rollback"
            );
        }
    }
}

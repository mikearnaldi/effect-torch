//! Independent stored-prefix attention regressions and exact uniform-score cases.
use super::*;
use effect_torch_graph::{AttentionRounding, KvAttentionMode};
use effect_torch_runtime::{CancellationFlag, KvLayerDescriptor, StateAccessMode};

fn input(slot: u32, shape: &[usize], dtype: DType) -> Arc<Node> {
    Node::new(NodeKind::Input {
        slot,
        shape: shape.to_vec(),
        dtype,
        device: effect_torch_graph::Device::Cuda(0),
        storage: effect_torch_runtime::StorageMetadata::dense(),
    })
    .unwrap()
}

fn root(
    batch: usize,
    heads: usize,
    kv: usize,
    tokens: usize,
    dim: usize,
    bidirectional: bool,
    window: Option<usize>,
    scale: f64,
) -> Arc<Node> {
    Node::new(NodeKind::KvAttention {
        q: input(0, &[batch, heads, tokens, dim], DType::BF16),
        k: input(1, &[batch, kv, tokens, dim], DType::BF16),
        v: input(2, &[batch, kv, tokens, dim], DType::BF16),
        scale,
        layer: 0,
        window,
        mode: if bidirectional {
            KvAttentionMode::BidirectionalBlock
        } else {
            KvAttentionMode::Causal
        },
        rounding: AttentionRounding::Stepwise,
    })
    .unwrap()
}
fn layout(
    slots: usize,
    capacity: usize,
    kv: usize,
    dim: usize,
    retention: Option<usize>,
    access: StateAccessMode,
) -> CudaStateLayout {
    CudaStateLayout {
        capacity: capacity as u32,
        dtype: DType::BF16,
        slots: slots as u32,
        packed_rows_per_sequence: None,
        access,
        kv_layers: vec![KvLayerDescriptor {
            layer_id: 0,
            kv_heads: kv,
            head_dim: dim,
            dtype: DType::BF16,
            retention,
        }],
    }
}
fn snapshot(
    descriptor: KvLayerDescriptor,
    start: usize,
    cursor: usize,
    k: &[u8],
    v: &[u8],
) -> CudaKvSnapshot {
    let device = CudaDevice::get(0).unwrap();
    let row = descriptor.kv_heads * descriptor.head_dim * 2;
    let split = (cursor - start) / 2;
    let mut pages = Vec::new();
    for (from, to) in [(0, split), (split, cursor - start)] {
        if from == to {
            continue;
        }
        pages.push(Arc::new(CudaKvPage {
            start: (start + from) as u32,
            count: (to - from) as u32,
            keys: CudaBuffer::from_slice(
                device.stream.clone_htod(&k[from * row..to * row]).unwrap(),
            ),
            values: CudaBuffer::from_slice(
                device.stream.clone_htod(&v[from * row..to * row]).unwrap(),
            ),
            key_scales: None,
            value_scales: None,
        }));
    }
    CudaKvSnapshot {
        layers: vec![CudaKvLayer {
            descriptor,
            start_position: start as u32,
            pages,
        }],
    }
}
fn sequence(cursor: usize, prefix: CudaKvSnapshot) -> CudaSequenceState {
    CudaSequenceState {
        cursor: cursor as u32,
        keys: vec![],
        values: vec![],
        kda_states: vec![],
        conv_states: vec![],
        kv_storage: Some(prefix),
    }
}
fn invocation(
    layout: &CudaStateLayout,
    sequences: Vec<CudaSequenceState>,
    slots: Vec<u32>,
    valid: Vec<u32>,
) -> CudaStateInvocation {
    CudaStateInvocation {
        sequences,
        slots,
        valid_lengths: valid,
        capacity: layout.capacity,
        cache_dtype: layout.dtype,
        packed_rows_per_sequence: layout.packed_rows_per_sequence,
        kv_layers: layout.kv_layers.clone(),
        access: layout.access,
        cache: None,
    }
}
fn compile(layout: CudaStateLayout, root: Arc<Node>) -> CudaExecutable {
    let executable = crate::compile_stateful_with_layout(
        vec![root],
        0,
        20,
        true,
        CompileOptions::default(),
        layout,
    )
    .unwrap();
    assert!(executable.commands.iter().any(|c| matches!(
        &c.kind,
        CommandKind::Kernel {
            kv_matmul: Some(_),
            ..
        }
    )));
    executable
}
fn host(shape: Vec<usize>, data: &[f64]) -> CudaValue {
    CudaValue::from_host(CudaDevice::get(0).unwrap(), shape, DType::BF16, data).unwrap()
}
fn encoded(data: &[f64]) -> Vec<u8> {
    data.iter()
        .flat_map(|&v| half::bf16::from_f64(v).to_bits().to_le_bytes())
        .collect()
}

#[test]
#[ignore = "requires a CUDA device"]
fn kv_gemm_full_canvas_changing_lengths_and_capture_rejection() {
    // Independent closed-form uniform-score oracle. BF16 probabilities times
    // these small integers have exact F32 sums at all tested lengths.
    let value = |p: usize, h: usize, d: usize| ((p + h + d) % 8) as f64 - 3.;
    let sum = |end: usize, h: usize, d: usize| {
        (end / 8) as f64 * 4. + (0..end % 8).map(|p| value(p, h, d)).sum::<f64>()
    };
    for dim in [256, 512] {
        for window in [None, Some(128), Some(1024)] {
            for (tokens, bidirectional) in [(278, false), (256, true)] {
                let (heads, kv) = (16, if dim == 512 { 2 } else { 8 });
                let retention = (window == Some(1024)).then_some(1023);
                let layout = layout(
                    1,
                    1024,
                    kv,
                    dim,
                    retention,
                    if bidirectional {
                        StateAccessMode::ReadOnly
                    } else {
                        StateAccessMode::Append
                    },
                );
                let executable = compile(
                    layout.clone(),
                    root(1, heads, kv, tokens, dim, bidirectional, window, 0.30157),
                );
                let mut lengths: Vec<(usize, usize)> = if bidirectional {
                    vec![
                        (278, 256),
                        (534, 256),
                        (7, 17),
                        (534, 1),
                        (278, 0),
                        (278, 256),
                    ]
                } else {
                    vec![(0, 278), (278, 17), (295, 1), (296, 0), (0, 278)]
                };
                if retention.is_some() {
                    lengths.push((1100, tokens));
                }
                let mut retained = Vec::new();
                for (cursor, valid) in lengths {
                    let retained_start = cursor.saturating_sub(retention.unwrap_or(cursor));
                    let prefix_values = (retained_start..cursor)
                        .flat_map(|p| {
                            (0..kv).flat_map(move |h| (0..dim).map(move |d| value(p, h, d)))
                        })
                        .collect::<Vec<_>>();
                    let prefix = snapshot(
                        layout.kv_layers[0],
                        retained_start,
                        cursor,
                        &vec![0; prefix_values.len() * 2],
                        &encoded(&prefix_values),
                    );
                    let mut state = invocation(
                        &layout,
                        vec![sequence(cursor, prefix)],
                        vec![0],
                        vec![valid as u32],
                    );
                    let current = (0..kv)
                        .flat_map(|h| {
                            (0..tokens)
                                .flat_map(move |t| (0..dim).map(move |d| value(cursor + t, h, d)))
                        })
                        .collect::<Vec<_>>();
                    let bindings = [
                        host(vec![1, heads, tokens, dim], &vec![0.; heads * tokens * dim]),
                        host(vec![1, kv, tokens, dim], &vec![0.; kv * tokens * dim]),
                        host(vec![1, kv, tokens, dim], &current),
                    ];
                    assert!(executable
                        .execute_stateful_graphed(
                            &bindings,
                            &[],
                            &mut state,
                            &CancellationFlag::new()
                        )
                        .unwrap()
                        .is_none());
                    assert_eq!(state.sequences[0].cursor, cursor as u32);
                    let output = executable
                        .execute_stateful(&bindings, &[], &mut state, &CancellationFlag::new())
                        .unwrap();
                    let actual = output[0].readback().unwrap();
                    for h in 0..heads {
                        for t in 0..tokens {
                            let end = cursor + if bidirectional { valid } else { t + 1 };
                            let start =
                                retained_start.max(end.saturating_sub(window.unwrap_or(end)));
                            let prob = half::bf16::from_f64(1. / (end - start) as f64).to_f64();
                            for d in 0..dim {
                                let expected = if t >= valid {
                                    0.
                                } else {
                                    half::bf16::from_f64(
                                        prob * (sum(end, h * kv / heads, d)
                                            - sum(start, h * kv / heads, d)),
                                    )
                                    .to_f64()
                                };
                                assert_eq!(actual[(h*tokens+t)*dim+d],expected,"D{dim} Q{tokens}/{valid} prefix{cursor} window{window:?} h{h} t{t} d{d}");
                            }
                        }
                    }
                    retained.push((output[0].clone(), actual));
                }
                drop(executable);
                for (output, expected) in retained {
                    assert_eq!(output.readback().unwrap(), expected);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn kv_gemm_packed_rows_keep_full_logical_pv_extent() {
    for bidirectional in [false, true] {
        let (heads, kv, dim, rows, cursor, valid) = (4, 2, 257, 5, 17, 3);
        let mut layout = layout(
            1,
            32,
            kv,
            dim,
            None,
            if bidirectional {
                StateAccessMode::ReadOnly
            } else {
                StateAccessMode::Append
            },
        );
        layout.packed_rows_per_sequence = Some(rows as u32);
        let executable = compile(
            layout.clone(),
            root(rows, heads, kv, 1, dim, bidirectional, Some(4), 1.),
        );
        let prefix_values = (0..cursor)
            .flat_map(|p| {
                (0..kv).flat_map(move |h| (0..dim).map(move |d| ((p + h + d) % 8) as f64 - 3.))
            })
            .collect::<Vec<_>>();
        let prefix = snapshot(
            layout.kv_layers[0],
            0,
            cursor,
            &vec![0; prefix_values.len() * 2],
            &encoded(&prefix_values),
        );
        let mut state = invocation(
            &layout,
            vec![sequence(cursor, prefix)],
            vec![0],
            vec![valid as u32],
        );
        let currents = (0..rows)
            .flat_map(|t| {
                (0..kv).flat_map(move |h| {
                    (0..dim).map(move |d| ((cursor + t + h + d) % 8) as f64 - 3.)
                })
            })
            .collect::<Vec<_>>();
        let result = executable
            .execute_stateful(
                &[
                    host(vec![rows, heads, 1, dim], &vec![0.; rows * heads * dim]),
                    host(vec![rows, kv, 1, dim], &vec![0.; rows * kv * dim]),
                    host(vec![rows, kv, 1, dim], &currents),
                ],
                &[],
                &mut state,
                &CancellationFlag::new(),
            )
            .unwrap()[0]
            .readback()
            .unwrap();
        for t in 0..rows {
            for h in 0..heads {
                for d in 0..dim {
                    let end = cursor + if bidirectional { valid } else { t + 1 };
                    let expected = if t >= valid {
                        0.
                    } else {
                        (end - 4..end)
                            .map(|p| ((p + h * kv / heads + d) % 8) as f64 - 3.)
                            .sum::<f64>()
                            / 4.
                    };
                    assert_eq!(result[(t * heads + h) * dim + d], expected);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a CUDA device; bounded host-wall timing, including output readback"]
fn kv_gemm_timing_full_geometries_after_exactness_checks() {
    for dim in [256, 512] {
        for (tokens, cursor, bidirectional) in [(278, 0, false), (256, 278, true), (256, 534, true)]
        {
            let (heads, kv) = (16, if dim == 512 { 2 } else { 8 });
            let layout = layout(1, 1024, kv, dim, None, StateAccessMode::ReadOnly);
            let prefix = snapshot(
                layout.kv_layers[0],
                0,
                cursor,
                &vec![0; cursor * kv * dim * 2],
                &encoded(&vec![1.; cursor * kv * dim]),
            );
            let mut executable = compile(
                layout.clone(),
                root(1, heads, kv, tokens, dim, bidirectional, None, 1.),
            );
            let bindings = [
                host(vec![1, heads, tokens, dim], &vec![0.; heads * tokens * dim]),
                host(vec![1, kv, tokens, dim], &vec![0.; kv * tokens * dim]),
                host(vec![1, kv, tokens, dim], &vec![1.; kv * tokens * dim]),
            ];
            let mut legacy_ms = Vec::new();
            let mut candidate_ms = Vec::new();
            let mut expected = None;
            for iteration in 0..4 {
                for candidate in [true, false] {
                    let mut plans = Vec::new();
                    if !candidate {
                        for (index, command) in executable.commands.iter_mut().enumerate() {
                            if let CommandKind::Kernel { kv_matmul, .. } = &mut command.kind {
                                if let Some(plan) = kv_matmul.take() {
                                    plans.push((index, plan));
                                }
                            }
                        }
                    }
                    let mut state = invocation(
                        &layout,
                        vec![sequence(cursor, prefix.clone())],
                        vec![0],
                        vec![tokens as u32],
                    );
                    let start = std::time::Instant::now();
                    let output = executable
                        .execute_stateful(&bindings, &[], &mut state, &CancellationFlag::new())
                        .unwrap()[0]
                        .read_storage_bytes()
                        .unwrap();
                    let elapsed = start.elapsed().as_secs_f64() * 1000.;
                    if let Some(expected) = &expected {
                        assert_eq!(&output, expected);
                    } else {
                        expected = Some(output);
                    }
                    for (index, plan) in plans {
                        let CommandKind::Kernel { kv_matmul, .. } =
                            &mut executable.commands[index].kind
                        else {
                            unreachable!()
                        };
                        *kv_matmul = Some(plan);
                    }
                    if iteration != 0 {
                        if candidate {
                            candidate_ms.push(elapsed);
                        } else {
                            legacy_ms.push(elapsed);
                        }
                    }
                }
            }
            let scratch = executable
                .commands
                .iter()
                .find_map(|command| match &command.kind {
                    CommandKind::Kernel {
                        kv_matmul: Some(plan),
                        ..
                    } => Some(plan.bytes),
                    _ => None,
                })
                .unwrap();
            eprintln!("KV_TIMING D={dim} Q={tokens} P={} causal={} workspaceBytes={scratch} candidateHostWallMs={candidate_ms:?} legacyHostWallMs={legacy_ms:?}",cursor+tokens,!bidirectional);
        }
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn kv_gemm_exact_lengths_padding_gqa_windows_and_retention() {
    for dim in [1, 7, 255, 256, 257, 511, 512, 513, 1025] {
        for (access, bidirectional) in [
            (StateAccessMode::Append, false),
            (StateAccessMode::ReadOnly, true),
        ] {
            for retention in [None, Some(0), Some(5)] {
                for window in [None, Some(1), Some(7)] {
                    let (heads, kv, tokens) = (4, 2, 3);
                    let layout = layout(3, 32, kv, dim, retention, access);
                    let executable = compile(
                        layout.clone(),
                        root(3, heads, kv, tokens, dim, bidirectional, window, 0.30157),
                    );
                    let value =
                        |p: usize, h: usize, d: usize| ((p * 3 + h * 5 + d) % 17) as f64 - 8.;
                    let mut prefixes = Vec::new();
                    for cursor in [17usize, 3] {
                        let start = cursor.saturating_sub(retention.unwrap_or(cursor));
                        let values = (start..cursor)
                            .flat_map(|p| {
                                (0..kv).flat_map(move |h| (0..dim).map(move |d| value(p, h, d)))
                            })
                            .collect::<Vec<_>>();
                        prefixes.push(snapshot(
                            layout.kv_layers[0],
                            start,
                            cursor,
                            &vec![0; values.len() * 2],
                            &encoded(&values),
                        ));
                    }
                    let mut state = invocation(
                        &layout,
                        vec![
                            sequence(17, prefixes[0].clone()),
                            sequence(3, prefixes[1].clone()),
                        ],
                        vec![2, 0],
                        vec![1, 0, 3],
                    );
                    let currents = (0..3)
                        .flat_map(|slot| {
                            (0..kv).flat_map(move |h| {
                                (0..tokens).flat_map(move |t| {
                                    (0..dim).map(move |d| {
                                        value(if slot == 2 { 17 + t } else { 3 + t }, h, d)
                                    })
                                })
                            })
                        })
                        .collect::<Vec<_>>();
                    let bindings = [
                        host(
                            vec![3, heads, tokens, dim],
                            &vec![0.; 3 * heads * tokens * dim],
                        ),
                        host(vec![3, kv, tokens, dim], &vec![0.; 3 * kv * tokens * dim]),
                        host(vec![3, kv, tokens, dim], &currents),
                    ];
                    let output = executable
                        .execute_stateful(&bindings, &[], &mut state, &CancellationFlag::new())
                        .unwrap()[0]
                        .readback()
                        .unwrap();
                    for slot in 0..3 {
                        for h in 0..heads {
                            for t in 0..tokens {
                                let valid = [1, 0, 3][slot];
                                let cursor = if slot == 2 { 17usize } else { 3 };
                                let end = cursor + if bidirectional { valid } else { t + 1 };
                                let start = cursor
                                    .saturating_sub(retention.unwrap_or(cursor))
                                    .max(end.saturating_sub(window.unwrap_or(end)));
                                let prob = half::bf16::from_f64(1. / (end - start) as f64).to_f64();
                                for d in 0..dim {
                                    let expected = if t >= valid {
                                        0.
                                    } else {
                                        half::bf16::from_f64(
                                            prob * (start..end)
                                                .map(|p| value(p, h * kv / heads, d))
                                                .sum::<f64>(),
                                        )
                                        .to_f64()
                                    };
                                    assert_eq!(output[((slot*heads+h)*tokens+t)*dim+d],expected,"D={dim} {access:?} retention={retention:?} window={window:?} slot={slot} head={h} token={t} dim={d}");
                                }
                            }
                        }
                    }
                    if access == StateAccessMode::ReadOnly {
                        assert_eq!(state.sequences[0].cursor, 17);
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_ATTENTION_FIXTURE_DIR from official capture 10"]
fn kv_gemm_captured_context_matches_independent_oracle_and_old_witness() {
    let path = std::path::PathBuf::from(
        std::env::var_os("EFFECT_TORCH_ATTENTION_FIXTURE_DIR").expect("capture 10 directory"),
    );
    let file = std::fs::read(path.join("layer0.safetensors")).unwrap();
    let length = u64::from_le_bytes(file[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&file[8..8 + length]).unwrap();
    let tensor = |name: &str| {
        let m = &header[name];
        assert_eq!(m["dtype"], "BF16");
        let lo = m["data_offsets"][0].as_u64().unwrap() as usize;
        let hi = m["data_offsets"][1].as_u64().unwrap() as usize;
        file[8 + length + lo..8 + length + hi].to_vec()
    };
    let (heads, kv, tokens, dim, cursor) = (16, 8, 16, 256, 278);
    let q = tensor("decoder.attention.query");
    let k = tensor("decoder.attention.key");
    let v = tensor("decoder.attention.value");
    let official = tensor("decoder.attention.output.0");
    let token_first = |data: &[u8], h: usize, t: usize| {
        let mut out = Vec::new();
        for row in 0..t {
            for head in 0..h {
                out.extend_from_slice(
                    &data[(head * t + row) * dim * 2..(head * t + row + 1) * dim * 2],
                );
            }
        }
        out
    };
    let extract = |data: &[u8], begin: usize, end: usize| {
        let mut out = Vec::new();
        for h in 0..kv {
            out.extend_from_slice(
                &data[(h * (cursor + tokens) + begin) * dim * 2
                    ..(h * (cursor + tokens) + end) * dim * 2],
            );
        }
        out
    };
    let device = CudaDevice::get(0).unwrap();
    let bindings = [
        CudaValue::from_dense_bytes(device.clone(), vec![1, heads, tokens, dim], DType::BF16, &q)
            .unwrap(),
        CudaValue::from_dense_bytes(
            device.clone(),
            vec![1, kv, tokens, dim],
            DType::BF16,
            &extract(&k, cursor, cursor + tokens),
        )
        .unwrap(),
        CudaValue::from_dense_bytes(
            device.clone(),
            vec![1, kv, tokens, dim],
            DType::BF16,
            &extract(&v, cursor, cursor + tokens),
        )
        .unwrap(),
    ];
    for capacity in [278, 544, 1024] {
        let layout = layout(1, capacity, kv, dim, None, StateAccessMode::ReadOnly);
        let prefix = snapshot(
            layout.kv_layers[0],
            0,
            cursor,
            &token_first(&extract(&k, 0, cursor), kv, cursor),
            &token_first(&extract(&v, 0, cursor), kv, cursor),
        );
        let mut executable = compile(
            layout.clone(),
            root(1, heads, kv, tokens, dim, true, None, 1.),
        );
        let make_state = || {
            invocation(
                &layout,
                vec![sequence(cursor, prefix.clone())],
                vec![0],
                vec![tokens as u32],
            )
        };
        let result = executable
            .execute_stateful(&bindings, &[], &mut make_state(), &CancellationFlag::new())
            .unwrap();
        let fixed = token_first(&result[0].read_storage_bytes().unwrap(), heads, tokens);
        assert_eq!(
            fixed, official,
            "actual P=294 must not become planned capacity {capacity}"
        );
        if capacity == 544 {
            for command in &mut executable.commands {
                if let CommandKind::Kernel { kv_matmul, .. } = &mut command.kind {
                    *kv_matmul = None;
                }
            }
            let old = executable
                .execute_stateful(&bindings, &[], &mut make_state(), &CancellationFlag::new())
                .unwrap();
            let old = token_first(&old[0].read_storage_bytes().unwrap(), heads, tokens);
            assert_eq!(
                old.chunks_exact(2)
                    .zip(official.chunks_exact(2))
                    .filter(|(a, b)| a != b)
                    .count(),
                232
            );
            let i = 26895 * 2;
            assert_eq!(
                u16::from_le_bytes(old[i..i + 2].try_into().unwrap()),
                0x3e1c
            );
            assert_eq!(
                u16::from_le_bytes(official[i..i + 2].try_into().unwrap()),
                0x3e1d
            );
        }
    }
}

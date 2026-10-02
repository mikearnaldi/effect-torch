//! Stateful integration: the standalone56 screen separately poisons all cache writes.
use super::*;

fn chain(tokens: usize, heads: usize, dim: usize, shared: bool, exposed: bool) -> Vec<Arc<Node>> {
    let raw = input(
        if shared { 1 } else { 2 },
        &[1, tokens, heads, dim],
        DType::BF16,
    );
    let view = Node::new(NodeKind::Permute {
        a: raw.clone(),
        dims: vec![0, 2, 1, 3],
    })
    .unwrap();
    let v = Node::new(NodeKind::RmsNorm {
        x: view.clone(),
        weight: None,
        eps: 1e-6,
    })
    .unwrap();
    let k = if shared {
        Node::new(NodeKind::RmsNorm {
            x: view,
            weight: Some(input(2, &[dim], DType::BF16)),
            eps: 1e-6,
        })
        .unwrap()
    } else {
        input(1, &[1, heads, tokens, dim], DType::BF16)
    };
    let attention = Node::new(NodeKind::KvAttention {
        q: input(0, &[1, 16, tokens, dim], DType::BF16),
        k,
        v: v.clone(),
        scale: 0.0625,
        layer: 0,
        window: None,
        mode: KvAttentionMode::BidirectionalBlock,
        rounding: AttentionRounding::Stepwise,
    })
    .unwrap();
    let mut roots = vec![attention, raw];
    if exposed {
        roots.push(v);
    }
    roots
}
fn compile_vnorm(
    layout: CudaStateLayout,
    roots: Vec<Arc<Node>>,
    enabled: bool,
    selected: bool,
) -> CudaExecutable {
    let executable = crate::vnorm_store::with_test_policy(enabled, || {
        crate::compile_stateful_with_layout(roots, 0, 20, true, CompileOptions::default(), layout)
    })
    .unwrap();
    assert_eq!(
        executable
            .diagnostics()
            .instructions
            .iter()
            .filter(|i| i.kind == "kv_stepwise_bf16_gemm_vnorm_store")
            .count(),
        usize::from(selected)
    );
    executable
}
fn payload(tokens: usize, heads: usize, dim: usize, pattern: usize) -> Vec<u8> {
    (0..tokens * heads * dim)
        .flat_map(|i| {
            let bits = match pattern {
                0 => i as u16,
                1 => ((i * 37 & 0x807f) | ((120 + i % 9) << 7)) as u16,
                2 => [0u16, 0x8000, 1, 0x8001, 0x7f, 0x80, 0x3f80, 0xbf80][i % 8],
                3 => {
                    if i % dim == 0 {
                        0x7f80
                    } else {
                        0x3f80
                    }
                }
                4 => {
                    if i % dim == 0 {
                        0xff81
                    } else {
                        0xbf80
                    }
                }
                5 => {
                    if i % dim == 0 {
                        0x7fff
                    } else {
                        0x7f7f
                    }
                }
                6 => {
                    if i % 2 == 0 {
                        0
                    } else {
                        0x8000
                    }
                }
                7 => ((i % 128) | ((i % 2) << 15)) as u16,
                _ => {
                    if i % dim < 4 {
                        0x4380
                    } else {
                        0xb880
                    }
                }
            };
            bits.to_le_bytes()
        })
        .collect()
}
fn bindings(
    tokens: usize,
    heads: usize,
    dim: usize,
    shared: bool,
    pattern: usize,
) -> Vec<CudaValue> {
    let raw = CudaValue::from_dense_bytes(
        CudaDevice::get(0).unwrap(),
        vec![1, tokens, heads, dim],
        DType::BF16,
        &payload(tokens, heads, dim, pattern),
    )
    .unwrap();
    let q = host(vec![1, 16, tokens, dim], &vec![0.; 16 * tokens * dim]);
    if shared {
        vec![q, raw, host(vec![dim], &vec![1.; dim])]
    } else {
        vec![
            q,
            host(vec![1, heads, tokens, dim], &vec![0.; heads * tokens * dim]),
            raw,
        ]
    }
}
fn state(layout: &CudaStateLayout, count: usize) -> CudaStateInvocation {
    let bytes = vec![0u8; 5 * layout.kv_layers[0].kv_heads * layout.kv_layers[0].head_dim * 2];
    invocation(
        layout,
        vec![sequence(
            5,
            snapshot(layout.kv_layers[0], 0, 5, &bytes, &bytes),
        )],
        vec![0],
        vec![count as u32],
    )
}
fn cache_bytes(state: &CudaStateInvocation) -> Vec<Vec<u8>> {
    let device = CudaDevice::get(0).unwrap();
    state
        .sequences
        .iter()
        .flat_map(|s| s.kv_storage.as_ref().unwrap().layers.iter())
        .flat_map(|l| l.pages.iter())
        .flat_map(|p| {
            [
                device.stream.clone_dtoh(&p.keys).unwrap(),
                device.stream.clone_dtoh(&p.values).unwrap(),
            ]
        })
        .collect()
}
#[test]
#[ignore = "requires CUDA; native56 exact cache/output and alias lifetime gates"]
fn vnorm_store_exact_shapes_specials_prefixes_and_retained_outputs() {
    let mut retained = Vec::new();
    for (heads, dim) in [(8, 256), (2, 512)] {
        for tokens in [64, 256] {
            for shared in [false, true] {
                let layout = layout(1, 1024, heads, dim, None, StateAccessMode::Append);
                let roots = chain(tokens, heads, dim, shared, false);
                let baseline = compile_vnorm(layout.clone(), roots.clone(), false, false);
                let candidate = compile_vnorm(layout.clone(), roots, true, true);
                for pattern in 0..9 {
                    for count in [0, 1, tokens / 2, tokens] {
                        let input = bindings(tokens, heads, dim, shared, pattern);
                        let original = input
                            .iter()
                            .map(|v| v.read_storage_bytes().unwrap())
                            .collect::<Vec<_>>();
                        let mut reference_state = state(&layout, count);
                        let mut actual_state = state(&layout, count);
                        let prefix = cache_bytes(&actual_state);
                        let expected = baseline
                            .execute_stateful(
                                &input,
                                &[],
                                &mut reference_state,
                                &CancellationFlag::new(),
                            )
                            .unwrap();
                        let actual = candidate
                            .execute_stateful(
                                &input,
                                &[],
                                &mut actual_state,
                                &CancellationFlag::new(),
                            )
                            .unwrap();
                        assert_eq!(
                            actual[0].read_storage_bytes().unwrap(),
                            expected[0].read_storage_bytes().unwrap(),
                            "T{tokens} H{heads} shared{shared} pattern{pattern} count{count}"
                        );
                        assert_eq!(cache_bytes(&actual_state), cache_bytes(&reference_state));
                        assert_eq!(
                            &cache_bytes(&actual_state)[..prefix.len()],
                            prefix.as_slice()
                        );
                        assert_eq!(
                            input
                                .iter()
                                .map(|v| v.read_storage_bytes().unwrap())
                                .collect::<Vec<_>>(),
                            original
                        );
                        assert_eq!(
                            actual[1].read_storage_bytes().unwrap(),
                            original[if shared { 1 } else { 2 }]
                        );
                        if pattern == 1 && count == tokens {
                            let bytes = actual[0].read_storage_bytes().unwrap();
                            retained.push((actual[0].clone(), bytes));
                        }
                    }
                }
            }
        }
    }
    for (output, expected) in retained {
        assert_eq!(output.read_storage_bytes().unwrap(), expected);
    }
}
#[test]
#[ignore = "requires CUDA; native56 unsupported escape, rollback, cancellation and concurrency"]
fn vnorm_store_transaction_rollback_cancel_concurrent_and_escape() {
    let (tokens, heads, dim) = (64, 8, 256);
    let layout = layout(1, 1024, heads, dim, None, StateAccessMode::ReadOnly);
    let roots = chain(tokens, heads, dim, true, false);
    let baseline = compile_vnorm(layout.clone(), roots.clone(), false, false);
    let candidate = Arc::new(compile_vnorm(layout.clone(), roots, true, true));
    let _escaped = compile_vnorm(
        layout.clone(),
        chain(tokens, heads, dim, true, true),
        true,
        false,
    );
    let input = bindings(tokens, heads, dim, true, 1);
    let make = || state(&layout, tokens);
    let mut expected_state = make();
    let expected = baseline
        .execute_stateful(&input, &[], &mut expected_state, &CancellationFlag::new())
        .unwrap()[0]
        .read_storage_bytes()
        .unwrap();
    let mut actual_state = make();
    let prefix = cache_bytes(&actual_state);
    let cancelled = CancellationFlag::new();
    assert!(candidate
        .execute_stateful_with_kv_hook(&input, &mut actual_state, &cancelled, &|| cancelled
            .cancel())
        .is_err());
    assert_eq!(actual_state.sequences[0].cursor, 5);
    assert_eq!(cache_bytes(&actual_state), prefix);
    for invalid in [tokens + 1, 2048] {
        actual_state.valid_lengths[0] = invalid as u32;
        assert!(candidate
            .execute_stateful(&input, &[], &mut actual_state, &CancellationFlag::new())
            .is_err());
        assert_eq!(cache_bytes(&actual_state), prefix);
    }
    actual_state.valid_lengths[0] = tokens as u32;
    let actual = candidate
        .execute_stateful(&input, &[], &mut actual_state, &CancellationFlag::new())
        .unwrap();
    assert_eq!(actual[0].read_storage_bytes().unwrap(), expected);
    assert_eq!(cache_bytes(&actual_state), prefix);
    let workers = (0..2)
        .map(|_| {
            let executable = candidate.clone();
            let layout = layout.clone();
            let input = input.clone();
            std::thread::spawn(move || {
                let mut state = state(&layout, tokens);
                executable
                    .execute_stateful(&input, &[], &mut state, &CancellationFlag::new())
                    .unwrap()[0]
                    .read_storage_bytes()
                    .unwrap()
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        assert_eq!(worker.join().unwrap(), expected);
    }
    assert_eq!(actual[0].read_storage_bytes().unwrap(), expected);
}

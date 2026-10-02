//! Native integration gates; standalone57 additionally poisons guarded outputs.
use crate::{CudaDevice, CudaValue};
use effect_torch_graph::{Device, Node, NodeKind};
use effect_torch_runtime::{CancellationFlag, DType, StorageMetadata};
use std::sync::Arc;
fn input(slot: u32, shape: &[usize]) -> Arc<Node> {
    Node::new(NodeKind::Input {
        slot,
        shape: shape.to_vec(),
        dtype: DType::BF16,
        device: Device::Cuda(0),
        storage: StorageMetadata::dense(),
    })
    .unwrap()
}
fn roots(normalized: bool) -> Vec<Arc<Node>> {
    // The activation is an actual alias, retained as a direct operand of both.
    let x = Node::new(NodeKind::Reshape {
        a: input(0, &[256, 2816]),
        shape: vec![1, 256, 2816],
    })
    .unwrap();
    let projection = |slot| {
        let wt = Node::new(NodeKind::Permute {
            a: input(slot, &[2048, 2816]),
            dims: vec![1, 0],
        })
        .unwrap();
        Node::new(NodeKind::Matmul {
            a: x.clone(),
            b: wt,
        })
        .unwrap()
    };
    let k = projection(1);
    let v = projection(2);
    let first = if normalized {
        Node::new(NodeKind::RmsNorm {
            x: k.clone(),
            weight: None,
            eps: 1e-6,
        })
        .unwrap()
    } else {
        k.clone()
    };
    vec![first, v, k, input(3, &[1])]
}
fn data(device: &Arc<CudaDevice>, shape: Vec<usize>, seed: u32, special: bool) -> CudaValue {
    let mut rng = seed;
    let n = shape.iter().product::<usize>();
    let mut bytes = Vec::with_capacity(n * 2);
    for i in 0..n {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        let bits = if special {
            let edge = [
                0u16, 0x8000, 1, 0x8001, 0x7f80, 0xff80, 0x7f81, 0xffc1, 0x3f80, 0xbf80,
            ];
            edge[(i + seed as usize) % edge.len()]
        } else {
            ((rng & 0x807f) | (((rng >> 24) % 12 + 120) << 7)) as u16
        };
        bytes.extend_from_slice(&bits.to_le_bytes());
    }
    CudaValue::from_dense_bytes(device.clone(), shape, DType::BF16, &bytes).unwrap()
}
fn bytes(values: &[CudaValue]) -> Vec<Vec<u8>> {
    values
        .iter()
        .map(|v| v.read_storage_bytes().unwrap())
        .collect()
}
#[test]
#[ignore = "requires pinned CUDA device, accepted ordinary K16 and standalone57 pair PTX"]
fn kv_pair_native_exact_retention_alias_fallback_cancellation_and_recovery() {
    let device = CudaDevice::get(0).unwrap();
    assert!(
        device.kv_pair.is_some(),
        "pair PTX must be loaded before test"
    );
    for normalized in [false, true] {
        let reference =
            crate::kv_pair::with_test_policy(false, || crate::compile(roots(normalized), 0))
                .unwrap();
        let candidate = Arc::new(
            crate::kv_pair::with_test_policy(true, || crate::compile(roots(normalized), 0))
                .unwrap(),
        );
        assert!(candidate
            .diagnostics()
            .instructions
            .iter()
            .any(|i| i.kind == "kv_pair_bf16_gemm"));
        assert!(!reference
            .diagnostics()
            .instructions
            .iter()
            .any(|i| i.kind == "kv_pair_bf16_gemm"));
        let mut retained = Vec::new();
        for special in [false, true] {
            for frame in 0..2 {
                let bindings = vec![
                    data(&device, vec![256, 2816], 17 + frame, special),
                    data(&device, vec![2048, 2816], 31 + frame, special),
                    data(&device, vec![2048, 2816], 313 + frame, special),
                    data(&device, vec![1], 0, false),
                ];
                let original = bytes(&bindings);
                let baseline = reference
                    .execute(&bindings, &[], &CancellationFlag::new())
                    .unwrap();
                let expected = bytes(&baseline);
                if !normalized {
                    // Independent poisoned allocations rule out stale/reused output bytes.
                    let poison =
                        |reference: &[u8]| reference.iter().map(|b| !b).collect::<Vec<_>>();
                    let direct_k = CudaValue::from_dense_bytes(
                        device.clone(),
                        vec![1, 256, 2048],
                        DType::BF16,
                        &poison(&expected[2]),
                    )
                    .unwrap();
                    let direct_v = CudaValue::from_dense_bytes(
                        device.clone(),
                        vec![1, 256, 2048],
                        DType::BF16,
                        &poison(&expected[1]),
                    )
                    .unwrap();
                    let addresses = [
                        bindings[0].storage_address(),
                        bindings[1].storage_address(),
                        bindings[2].storage_address(),
                        direct_k.storage_address(),
                        direct_v.storage_address(),
                    ];
                    let submitted = unsafe {
                        device
                            .kv_pair
                            .as_ref()
                            .unwrap()
                            .launch(&device.stream, addresses)
                    };
                    device.stream.synchronize().unwrap();
                    submitted.unwrap();
                    assert_eq!(direct_k.read_storage_bytes().unwrap(), expected[2]);
                    assert_eq!(direct_v.read_storage_bytes().unwrap(), expected[1]);
                }
                let output = candidate
                    .execute(&bindings, &[], &CancellationFlag::new())
                    .unwrap();
                assert_eq!(bytes(&output), expected);
                assert_eq!(bytes(&bindings), original);
                retained.push((output, expected.clone()));
                // Runtime-only weight aliasing forces fallback before submission.
                let mut alias = bindings.clone();
                alias[2] = alias[1].clone();
                let expected_alias = reference
                    .execute(&alias, &[], &CancellationFlag::new())
                    .unwrap();
                assert_eq!(
                    bytes(
                        &candidate
                            .execute(&alias, &[], &CancellationFlag::new())
                            .unwrap()
                    ),
                    bytes(&expected_alias)
                );
                if !special && frame == 0 {
                    let unavailable = crate::kv_pair::with_test_policy(false, || {
                        candidate.execute(&bindings, &[], &CancellationFlag::new())
                    })
                    .unwrap();
                    assert_eq!(bytes(&unavailable), expected);
                    // A late invalid binding fails after paired work is enqueued.
                    let mut invalid = bindings.clone();
                    invalid[3] = data(&device, vec![2], 0, false);
                    let off = reference
                        .execute(&invalid, &[], &CancellationFlag::new())
                        .err()
                        .unwrap();
                    let on = candidate
                        .execute(&invalid, &[], &CancellationFlag::new())
                        .err()
                        .unwrap();
                    assert_eq!(on, off);
                    let cancel = CancellationFlag::new();
                    assert!(candidate
                        .execute_with_gemm_hook(&bindings, &cancel, &|| cancel.cancel())
                        .is_err());
                    assert_eq!(
                        bytes(
                            &candidate
                                .execute(&bindings, &[], &CancellationFlag::new())
                                .unwrap()
                        ),
                        expected
                    );
                    let workers = (0..2)
                        .map(|_| {
                            let exec = candidate.clone();
                            let input = bindings.clone();
                            std::thread::spawn(move || {
                                bytes(&exec.execute(&input, &[], &CancellationFlag::new()).unwrap())
                            })
                        })
                        .collect::<Vec<_>>();
                    for worker in workers {
                        assert_eq!(worker.join().unwrap(), expected);
                    }
                }
            }
        }
        for (output, expected) in retained {
            assert_eq!(bytes(&output), expected);
        }
    }
}

#[test]
#[ignore = "requires pinned CUDA device, accepted ordinary K16 and standalone57 pair PTX"]
fn kv_pair_native_captured_constant_weights_preserve_norm_order() {
    let device = CudaDevice::get(0).unwrap();
    assert!(device.kv_pair.is_some());
    let weight = |seed| {
        Node::new(NodeKind::FromBytes {
            data: data(&device, vec![2048, 2816], seed, false)
                .read_storage_bytes()
                .unwrap(),
            shape: vec![2048, 2816],
            dtype: DType::BF16,
            device: Device::Cuda(0),
        })
        .unwrap()
    };
    let x = input(0, &[1, 256, 2816]);
    let projection = |weight| {
        let w = Node::new(NodeKind::Permute {
            a: weight,
            dims: vec![1, 0],
        })
        .unwrap();
        Node::new(NodeKind::Matmul { a: x.clone(), b: w }).unwrap()
    };
    let k = projection(weight(17));
    let v = projection(weight(31));
    let normalized = Node::new(NodeKind::RmsNorm {
        x: k.clone(),
        weight: None,
        eps: 1e-6,
    })
    .unwrap();
    let graph = vec![normalized, v, k];
    let reference =
        crate::kv_pair::with_test_policy(false, || crate::compile(graph.clone(), 0)).unwrap();
    let candidate = crate::kv_pair::with_test_policy(true, || crate::compile(graph, 0)).unwrap();
    assert!(candidate
        .diagnostics()
        .instructions
        .iter()
        .any(|i| i.kind == "kv_pair_bf16_gemm"));
    let mut retained = Vec::new();
    for frame in 0..2 {
        let inputs = vec![data(&device, vec![1, 256, 2816], 313 + frame, false)];
        let expected = reference
            .execute(&inputs, &[], &CancellationFlag::new())
            .unwrap();
        let output = candidate
            .execute(&inputs, &[], &CancellationFlag::new())
            .unwrap();
        assert_eq!(bytes(&output), bytes(&expected));
        retained.push((output, bytes(&expected)));
    }
    for (output, expected) in retained {
        assert_eq!(bytes(&output), expected);
    }
}

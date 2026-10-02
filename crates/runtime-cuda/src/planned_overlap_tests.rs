//! Hardware ownership and numerical gates for planned dense/expert overlap.
use crate::{CudaDevice, CudaValue};
use effect_torch_graph::{Device, Node, NodeKind};
use effect_torch_runtime::{CancellationFlag, DType, StorageMetadata};
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

fn projection(x: Arc<Node>, slot: u32, output: usize, inner: usize) -> Arc<Node> {
    Node::new(NodeKind::Matmul {
        a: x,
        b: Node::new(NodeKind::Permute {
            a: input(slot, &[output, inner], DType::BF16),
            dims: vec![1, 0],
        })
        .unwrap(),
    })
    .unwrap()
}

fn graph() -> Arc<Node> {
    let x = input(0, &[65, 37], DType::BF16);
    let dense_input = Node::new(NodeKind::RmsNorm {
        x: x.clone(),
        weight: None,
        eps: 1e-6,
    })
    .unwrap();
    let gate = projection(dense_input.clone(), 1, 53, 37);
    let up = projection(dense_input, 2, 53, 37);
    let product = Node::new(NodeKind::Mul { a: gate, b: up }).unwrap();
    let dense = projection(product, 3, 37, 53);
    let experts = Node::new(NodeKind::GroupedExpertLinearRows {
        x,
        weight: input(4, &[7, 37, 37], DType::BF16),
        indexes: input(5, &[65], DType::U32),
    })
    .unwrap();
    Node::new(NodeKind::Add {
        a: dense,
        b: experts,
    })
    .unwrap()
}

fn bindings(factor: f64, pass: usize) -> Vec<CudaValue> {
    let device = CudaDevice::get(0).unwrap();
    [
        [65, 37].as_slice(),
        &[53, 37],
        &[53, 37],
        &[37, 53],
        &[7, 37, 37],
    ]
    .into_iter()
    .enumerate()
    .map(|(slot, shape)| {
        let count = shape.iter().product::<usize>();
        let data = (0..count)
            .map(|index| {
                let value = (((index * (31 + slot * 6)) % 113) as f64 - 56.) / 61.;
                if slot == 0 {
                    value * factor
                } else {
                    value
                }
            })
            .collect::<Vec<_>>();
        CudaValue::from_host(device.clone(), shape.to_vec(), DType::BF16, &data).unwrap()
    })
    .chain(std::iter::once(
        CudaValue::from_host(
            device.clone(),
            vec![65],
            DType::U32,
            &(0..65)
                .map(|row| {
                    if pass == 0 {
                        ((row * 3 + row / 5) % 7) as f64
                    } else {
                        6.
                    }
                })
                .collect::<Vec<_>>(),
        )
        .unwrap(),
    ))
    .collect()
}

#[test]
#[ignore = "requires CUDA; run with EFFECT_TORCH_CUDA_DENSE_OVERLAP unset"]
fn planned_overlap_exact_dynamic_routing_cancel_and_concurrent_ownership() {
    assert!(!std::env::var("EFFECT_TORCH_CUDA_DENSE_OVERLAP").is_ok_and(|value| value == "1"));
    let root = graph();
    let serial = crate::compile(vec![root.clone()], 0).unwrap();
    let mut parallel = crate::compile(vec![root], 0).unwrap();
    assert_eq!(parallel.enable_dense_overlap_for_test().unwrap(), 1);
    let parallel = Arc::new(parallel);
    let mut retained = Vec::new();
    for (factor, pass) in [(1., 0), (0.5, 1), (-0.75, 0)] {
        let bindings = bindings(factor, pass);
        let originals = bindings
            .iter()
            .map(|value| value.read_storage_bytes().unwrap())
            .collect::<Vec<_>>();
        let expected = serial
            .execute(&bindings, &[], &CancellationFlag::new())
            .unwrap()[0]
            .read_storage_bytes()
            .unwrap();
        let output = parallel
            .execute(&bindings, &[], &CancellationFlag::new())
            .unwrap()
            .remove(0);
        assert_eq!(output.read_storage_bytes().unwrap(), expected);
        retained.push((output, expected));
        let cancelled = CancellationFlag::new();
        let submissions = std::cell::Cell::new(0);
        let result = parallel.execute_with_gemm_hook(&bindings, &cancelled, &|| {
            submissions.set(submissions.get() + 1);
            cancelled.cancel();
        });
        assert_eq!(submissions.get(), 1);
        assert!(matches!(result, Err(ref error) if error.contains("aborted")));
        let recovered = parallel
            .execute(&bindings, &[], &CancellationFlag::new())
            .unwrap();
        assert_eq!(
            recovered[0].read_storage_bytes().unwrap(),
            retained.last().unwrap().1
        );
        for (binding, original) in bindings.iter().zip(originals) {
            assert_eq!(binding.read_storage_bytes().unwrap(), original);
        }
        for (output, expected) in &retained {
            assert_eq!(output.read_storage_bytes().unwrap(), *expected);
        }
    }
    let cases = (0..4)
        .map(|index| {
            let bindings = bindings((index + 1) as f64 / 4., index % 2);
            let expected = serial
                .execute(&bindings, &[], &CancellationFlag::new())
                .unwrap()[0]
                .read_storage_bytes()
                .unwrap();
            (bindings, expected)
        })
        .collect::<Vec<_>>();
    let workers = cases
        .into_iter()
        .map(|(bindings, expected)| {
            let parallel = parallel.clone();
            std::thread::spawn(move || {
                let result = parallel
                    .execute(&bindings, &[], &CancellationFlag::new())
                    .unwrap();
                assert_eq!(result[0].read_storage_bytes().unwrap(), expected);
                (result, expected)
            })
        })
        .collect::<Vec<_>>();
    let results = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .collect::<Vec<_>>();
    for (result, expected) in results {
        assert_eq!(result[0].read_storage_bytes().unwrap(), expected);
    }
    for (output, expected) in retained {
        assert_eq!(output.read_storage_bytes().unwrap(), expected);
    }
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_OVERLAP_PRIMARY_GRAPHS=1; dense overlap env unset"]
fn planned_overlap_primary_graph_islands_replay_and_preserve_ownership() {
    assert!(
        std::env::var("EFFECT_TORCH_CUDA_OVERLAP_PRIMARY_GRAPHS").is_ok_and(|value| value == "1")
    );
    assert!(!std::env::var("EFFECT_TORCH_CUDA_DENSE_OVERLAP").is_ok_and(|value| value == "1"));
    let mut root = graph();
    for _ in 0..3 {
        root = Node::new(NodeKind::RmsNorm {
            x: root,
            weight: None,
            eps: 1e-6,
        })
        .unwrap();
    }
    let serial = crate::compile(vec![root.clone()], 0).unwrap();
    let mut parallel = crate::compile(vec![root], 0).unwrap();
    assert_eq!(parallel.enable_dense_overlap_for_test().unwrap(), 1);
    let inputs = bindings(0.75, 0);
    let originals = inputs
        .iter()
        .map(|v| v.read_storage_bytes().unwrap())
        .collect::<Vec<_>>();
    let expected = serial
        .execute(&inputs, &[], &CancellationFlag::new())
        .unwrap()[0]
        .read_storage_bytes()
        .unwrap();
    let mut retained = None;
    for iteration in 0..12 {
        let output = parallel
            .execute(&inputs, &[], &CancellationFlag::new())
            .unwrap()
            .remove(0);
        assert_eq!(output.read_storage_bytes().unwrap(), expected);
        if iteration == 5 {
            retained = Some(output);
        }
    }
    let (captures, hits) = parallel.graph_counts_for_test();
    assert!(
        captures > 0 && hits > 0,
        "expected actual primary graph capture/replay: {captures}/{hits}"
    );
    let cancelled = CancellationFlag::new();
    cancelled.cancel();
    assert!(
        matches!(parallel.execute(&inputs, &[], &cancelled), Err(error) if error.contains("aborted"))
    );
    assert_eq!(
        parallel
            .execute(&inputs, &[], &CancellationFlag::new())
            .unwrap()[0]
            .read_storage_bytes()
            .unwrap(),
        expected
    );
    assert_eq!(retained.unwrap().read_storage_bytes().unwrap(), expected);
    for (input, original) in inputs.iter().zip(originals) {
        assert_eq!(input.read_storage_bytes().unwrap(), original);
    }
    // The existing ownership gate adds cancellation during dense submission,
    // changing expert routing, retained output arenas, and concurrent callers.
    planned_overlap_exact_dynamic_routing_cancel_and_concurrent_ownership();
}

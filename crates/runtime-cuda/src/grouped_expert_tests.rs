//! Exact grouped matrix semantics and invocation lifetime tests on CUDA.
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
fn grouped(n: usize, e: usize, o: usize, k: usize, dtype: DType) -> Arc<Node> {
    Node::new(NodeKind::GroupedExpertLinearRows {
        x: input(0, &[n, k], dtype),
        weight: input(1, &[e, o, k], dtype),
        indexes: input(2, &[n], DType::U32),
    })
    .unwrap()
}
fn host(shape: &[usize], dtype: DType, data: &[f64]) -> CudaValue {
    CudaValue::from_host(CudaDevice::get(0).unwrap(), shape.to_vec(), dtype, data).unwrap()
}
fn matrix(
    rows: usize,
    columns: usize,
    inner: usize,
    dtype: DType,
    x: &[f64],
    weight: &[f64],
) -> Vec<u8> {
    let root = Node::new(NodeKind::Matmul {
        a: input(0, &[rows, inner], dtype),
        b: Node::new(NodeKind::Permute {
            a: input(1, &[columns, inner], dtype),
            dims: vec![1, 0],
        })
        .unwrap(),
    })
    .unwrap();
    crate::compile(vec![root], 0)
        .unwrap()
        .execute(
            &[
                host(&[rows, inner], dtype, x),
                host(&[columns, inner], dtype, weight),
            ],
            &[],
            &CancellationFlag::new(),
        )
        .unwrap()[0]
        .read_storage_bytes()
        .unwrap()
}

#[test]
#[ignore = "requires a CUDA device"]
fn grouped_dynamic_counts_stable_order_views_and_matrix_semantics() {
    let (n, e, o, k) = (69, 7, 13, 37);
    for dtype in [DType::F32, DType::BF16] {
        let x = (0..n * k)
            .map(|i| (((i * 31) % 103) as f64 - 51.0) / 17.0)
            .collect::<Vec<_>>();
        let w = (0..e * o * k)
            .map(|i| (((i * 41) % 109) as f64 - 54.0) / 23.0)
            .collect::<Vec<_>>();
        // Inputs and banks enter through nontrivial transpose views.
        let root = Node::new(NodeKind::GroupedExpertLinearRows {
            x: Node::new(NodeKind::Permute {
                a: input(0, &[k, n], dtype),
                dims: vec![1, 0],
            })
            .unwrap(),
            weight: Node::new(NodeKind::Permute {
                a: input(1, &[e, k, o], dtype),
                dims: vec![0, 2, 1],
            })
            .unwrap(),
            indexes: input(2, &[n], DType::U32),
        })
        .unwrap();
        let executable = crate::compile(vec![root], 0).unwrap();
        assert_eq!(
            executable
                .diagnostics()
                .legalization
                .materialized_conversions,
            0
        );
        let mut xt = vec![0.; n * k];
        let mut wt = vec![0.; e * o * k];
        for r in 0..n {
            for i in 0..k {
                xt[i * n + r] = x[r * k + i];
            }
        }
        for expert in 0..e {
            for c in 0..o {
                for i in 0..k {
                    wt[(expert * k + i) * o + c] = w[(expert * o + c) * k + i];
                }
            }
        }
        let xv = host(&[k, n], dtype, &xt);
        let wv = host(&[e, k, o], dtype, &wt);
        let original = wv.read_storage_bytes().unwrap();
        for pass in 0..3 {
            let ids = (0..n)
                .map(|r| {
                    if pass == 0 {
                        (r * 3 + r / 5) % 5
                    } else if pass == 1 {
                        6
                    } else {
                        (n - r) % 7
                    }
                })
                .collect::<Vec<_>>();
            let result = executable
                .execute(
                    &[
                        xv.clone(),
                        wv.clone(),
                        host(
                            &[n],
                            DType::U32,
                            &ids.iter().map(|&v| v as f64).collect::<Vec<_>>(),
                        ),
                    ],
                    &[],
                    &CancellationFlag::new(),
                )
                .unwrap();
            let bytes = result[0].read_storage_bytes().unwrap();
            let width = dtype.size_in_bytes();
            for expert in 0..e {
                let rows = (0..n).filter(|&r| ids[r] == expert).collect::<Vec<_>>();
                if rows.is_empty() {
                    continue;
                }
                let mut selected = Vec::new();
                for &r in &rows {
                    selected.extend_from_slice(&x[r * k..(r + 1) * k]);
                }
                let expected = matrix(
                    rows.len(),
                    o,
                    k,
                    dtype,
                    &selected,
                    &w[expert * o * k..(expert + 1) * o * k],
                );
                for (local, &r) in rows.iter().enumerate() {
                    assert_eq!(
                        &bytes[r * o * width..(r + 1) * o * width],
                        &expected[local * o * width..(local + 1) * o * width],
                        "{dtype:?} pass {pass} expert {expert} row {r}"
                    );
                }
            }
        }
        assert_eq!(wv.read_storage_bytes().unwrap(), original);
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn grouped_empty_dimensions_validate_indices_before_any_projection() {
    for dtype in [DType::F32, DType::BF16] {
        for (n, e, o, k) in [
            (0, 1, 3, 5),
            (0, 2, 0, 5),
            (3, 2, 0, 5),
            (3, 2, 4, 0),
            (3, 2, 0, 0),
        ] {
            let executable = crate::compile(vec![grouped(n, e, o, k, dtype)], 0).unwrap();
            let mut bindings = vec![
                host(&[n, k], dtype, &vec![1.; n * k]),
                host(&[e, o, k], dtype, &vec![1.; e * o * k]),
                host(&[n], DType::U32, &vec![0.; n]),
            ];
            let result = executable.execute(&bindings, &[], &CancellationFlag::new());
            if e == 0 && n != 0 {
                assert!(
                    matches!(result,Err(ref error) if error.contains("expert index is out of range"))
                );
            } else {
                assert_eq!(result.unwrap()[0].readback().unwrap(), vec![0.; n * o]);
            }
            if n != 0 {
                let mut ids = vec![0.; n];
                ids[n - 1] = e as f64;
                bindings[2] = host(&[n], DType::U32, &ids);
                let submitted = std::cell::Cell::new(0);
                let result =
                    executable.execute_with_gemm_hook(&bindings, &CancellationFlag::new(), &|| {
                        submitted.set(submitted.get() + 1)
                    });
                assert!(
                    matches!(result,Err(ref error) if error.contains("expert index is out of range"))
                );
                assert_eq!(submitted.get(), 0);
            }
        }
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn grouped_broadcast_duplicate_rows_cancel_and_concurrent_lifetimes() {
    for dtype in [DType::F32, DType::BF16] {
        let root = Node::new(NodeKind::GroupedExpertLinearRows {
            x: Node::new(NodeKind::BroadcastTo {
                a: input(0, &[1, 3], dtype),
                shape: vec![65, 3],
            })
            .unwrap(),
            weight: Node::new(NodeKind::BroadcastTo {
                a: input(1, &[1, 2, 3], dtype),
                shape: vec![4, 2, 3],
            })
            .unwrap(),
            indexes: input(2, &[65], DType::U32),
        })
        .unwrap();
        let executable = Arc::new(crate::compile(vec![root], 0).unwrap());
        let bindings = vec![
            host(&[1, 3], dtype, &[1., 2., 3.]),
            host(&[1, 2, 3], dtype, &[1., 0., 0., 0., 1., 1.]),
            host(
                &[65],
                DType::U32,
                &(0..65).map(|r| (r % 4) as f64).collect::<Vec<_>>(),
            ),
        ];
        let expected = [1., 5.].repeat(65);
        let retained = executable
            .execute(&bindings, &[], &CancellationFlag::new())
            .unwrap();
        let cancelled = CancellationFlag::new();
        let submissions = std::cell::Cell::new(0);
        let result = executable.execute_with_gemm_hook(&bindings, &cancelled, &|| {
            submissions.set(submissions.get() + 1);
            cancelled.cancel();
        });
        assert!(matches!(result,Err(ref error) if error == "operation aborted"));
        assert_eq!(submissions.get(), 1);
        assert!(executable.execute(&bindings, &[], &cancelled).is_err());
        let workers = (1..=3)
            .map(|factor| {
                let executable = executable.clone();
                let mut bindings = bindings.clone();
                bindings[0] = host(
                    &[1, 3],
                    dtype,
                    &[factor as f64, 2. * factor as f64, 3. * factor as f64],
                );
                std::thread::spawn(move || {
                    executable
                        .execute(&bindings, &[], &CancellationFlag::new())
                        .unwrap()[0]
                        .readback()
                        .unwrap()
                })
            })
            .collect::<Vec<_>>();
        for (i, worker) in workers.into_iter().enumerate() {
            assert_eq!(
                worker.join().unwrap(),
                expected
                    .iter()
                    .map(|v| v * (i + 1) as f64)
                    .collect::<Vec<_>>()
            );
        }
        drop(executable);
        drop(bindings);
        assert_eq!(retained[0].readback().unwrap(), expected);
    }
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_GROUPED_EXPERT_FIXTURE_DIR from bounded replay 22"]
fn grouped_captured_expert_projections_match_official_bf16_bytes() {
    let directory = std::path::PathBuf::from(
        std::env::var_os("EFFECT_TORCH_GROUPED_EXPERT_FIXTURE_DIR")
            .expect("replay 22 fixture directory required"),
    );
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("report.json")).unwrap()).unwrap();
    let device = CudaDevice::get(0).unwrap();
    for case in report["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let shape = case["shape"].as_array().unwrap();
        let (n, k, o) = (
            shape[0].as_u64().unwrap() as usize,
            shape[1].as_u64().unwrap() as usize,
            case["weightShape"][0].as_u64().unwrap() as usize,
        );
        let load =
            |suffix: &str| std::fs::read(directory.join(format!("{name}.{suffix}"))).unwrap();
        let x = CudaValue::from_dense_bytes(
            device.clone(),
            vec![n, k],
            DType::BF16,
            &load("input.bf16"),
        )
        .unwrap();
        let w = CudaValue::from_dense_bytes(
            device.clone(),
            vec![1, o, k],
            DType::BF16,
            &load("weight.bf16"),
        )
        .unwrap();
        let executable = crate::compile(vec![grouped(n, 1, o, k, DType::BF16)], 0).unwrap();
        let bindings = [x, w, host(&[n], DType::U32, &vec![0.; n])];
        let output = executable
            .execute(&bindings, &[], &CancellationFlag::new())
            .unwrap();
        assert_eq!(
            output[0].read_storage_bytes().unwrap(),
            load("official.bf16"),
            "{name}"
        );
        assert_eq!(
            bindings[1].read_storage_bytes().unwrap(),
            load("weight.bf16")
        );
        eprintln!("captured grouped exact: {name} [{n},{k},{o}]");
    }
}

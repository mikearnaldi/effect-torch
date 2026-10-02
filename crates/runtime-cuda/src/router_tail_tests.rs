use super::*;
use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
use effect_torch_graph::Node;
use effect_torch_runtime::{CancellationFlag, StorageMetadata};
use std::sync::Arc;

struct Tail {
    weights: Arc<Node>,
    indices: Arc<Node>,
    probabilities: Arc<Node>,
    private: Vec<Arc<Node>>,
}

fn chain(
    rows: usize,
    width: usize,
    k: usize,
    scale_dtype: DType,
    views: bool,
    reverse: bool,
) -> Tail {
    let input = |slot, shape, dtype| {
        Node::new(NodeKind::Input {
            slot,
            shape,
            dtype,
            device: Device::Cuda(0),
            storage: StorageMetadata::dense(),
        })
        .unwrap()
    };
    let source = if views {
        Node::new(NodeKind::Permute {
            a: input(0, vec![width, rows], DType::F32),
            dims: vec![1, 0],
        })
        .unwrap()
    } else {
        input(0, vec![rows, width], DType::F32)
    };
    let scale = if views {
        Node::new(NodeKind::Slice {
            a: input(1, vec![width + 1], scale_dtype),
            ranges: vec![(1, width + 1, 1)],
        })
        .unwrap()
    } else {
        input(1, vec![width], scale_dtype)
    };
    tail_from_inputs(source, scale, k, reverse)
}

fn tail_from_inputs(source: Arc<Node>, scale: Arc<Node>, k: usize, reverse: bool) -> Tail {
    let rows = source.shape[0];
    let scale_dtype = scale.dtype;
    let mut private = Vec::new();
    let mut make = |kind| {
        let n = Node::new(kind).unwrap();
        private.push(n.clone());
        n
    };
    let maximum = make(NodeKind::Max {
        a: source.clone(),
        dims: vec![1],
        keepdims: true,
    });
    let shifted = make(NodeKind::Sub {
        a: source,
        b: maximum,
    });
    let exponential = make(NodeKind::Exp { a: shifted });
    let denominator = make(NodeKind::Sum {
        a: exponential.clone(),
        dims: vec![1],
        keepdims: true,
    });
    let probabilities = make(NodeKind::Div {
        a: exponential,
        b: denominator,
    });
    let indices = Node::new(NodeKind::TopKIndices {
        a: probabilities.clone(),
        k,
    })
    .unwrap();
    let selected = make(NodeKind::Gather {
        a: probabilities.clone(),
        dim: 1,
        indexes: indices.clone(),
    });
    let total = make(NodeKind::Sum {
        a: selected.clone(),
        dims: vec![1],
        keepdims: true,
    });
    let normalized = make(NodeKind::Div {
        a: selected,
        b: total,
    });
    let flat = make(NodeKind::Reshape {
        a: indices.clone(),
        shape: vec![rows * k],
    });
    let taken = make(NodeKind::IndexSelect {
        a: scale,
        dim: 0,
        indexes: flat,
    });
    let scale_view = make(NodeKind::Reshape {
        a: taken,
        shape: vec![rows, k],
    });
    let scale_float = if scale_dtype == DType::F32 {
        scale_view
    } else {
        make(NodeKind::Cast {
            a: scale_view,
            dtype: DType::F32,
        })
    };
    let weights = Node::new(if reverse {
        NodeKind::Mul {
            a: scale_float,
            b: normalized,
        }
    } else {
        NodeKind::Mul {
            a: normalized,
            b: scale_float,
        }
    })
    .unwrap();
    Tail {
        weights,
        indices,
        probabilities,
        private,
    }
}

#[test]
fn router_tail_private_two_outputs_shape_dtype_order_and_policy_guards() {
    for (rows, width, k, dtype, enabled, reverse, expected) in [
        (1, 128, 8, DType::BF16, true, false, true),
        (64, 128, 8, DType::BF16, true, false, true),
        (256, 128, 8, DType::F32, true, false, true),
        (64, 128, 8, DType::BF16, false, false, false),
        (64, 128, 8, DType::F16, true, false, false),
        (64, 128, 8, DType::BF16, true, true, false),
        (64, 256, 8, DType::BF16, true, false, false),
        (64, 128, 4, DType::BF16, true, false, false),
    ] {
        let tail = chain(rows, width, k, dtype, false, reverse);
        for exposure in 0..(tail.private.len() * 2 + 3) {
            let mut roots = vec![tail.weights.clone()];
            if exposure == 1 {
                roots.push(tail.indices.clone());
            }
            if exposure == 2 {
                roots.push(
                    Node::new(NodeKind::Reshape {
                        a: tail.indices.clone(),
                        shape: vec![rows * k],
                    })
                    .unwrap(),
                );
            }
            if exposure >= 3 {
                let node = tail.private[(exposure - 3) / 2].clone();
                roots.push(if exposure % 2 == 1 {
                    node
                } else {
                    Node::new(NodeKind::Reshape {
                        a: node.clone(),
                        shape: vec![node.shape.iter().product()],
                    })
                    .unwrap()
                });
            }
            let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
                .prepare()
                .unwrap();
            let mut caps = CudaCapabilities::new(0, 12, 0);
            caps.router_tail = enabled;
            let driver = CompilerDriver::new(&prepared, &caps).unwrap();
            let selected = driver
                .optimization()
                .regions
                .iter()
                .find(|r| matches!(r, NativeRegion::RouterTail(_)));
            assert_eq!(
                selected.is_some(),
                expected && exposure < 3,
                "rows={rows} width={width} k={k} dtype={dtype:?} enabled={enabled} reverse={reverse} exposure={exposure}"
            );
            if let Some(region) = selected {
                assert_eq!(region.output_count(), 2);
                assert_eq!(region.semantic_outputs().len(), 2);
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_ROUTER_TAIL=1"]
fn router_tail_exact_probabilities_indices_weights_nan_views_cancel_concurrent_retained() {
    assert_eq!(
        std::env::var("EFFECT_TORCH_CUDA_ROUTER_TAIL").as_deref(),
        Ok("1")
    );
    let device = crate::CudaDevice::get(0).unwrap();
    for (rows, dtype, views) in [
        (1, DType::BF16, false),
        (64, DType::BF16, false),
        (256, DType::F32, false),
        (64, DType::BF16, true),
    ] {
        let tail = chain(rows, 128, 8, dtype, views, false);
        let optimized =
            Arc::new(crate::compile(vec![tail.weights.clone(), tail.indices.clone()], 0).unwrap());
        let reference =
            crate::compile(vec![tail.weights, tail.indices, tail.probabilities], 0).unwrap();
        assert_eq!(
            optimized
                .diagnostics()
                .instructions
                .iter()
                .filter(|i| i.kind == "et_router_tail")
                .count(),
            1
        );
        assert!(!reference
            .diagnostics()
            .instructions
            .iter()
            .any(|i| i.kind == "et_router_tail"));
        if rows == 64 && !views {
            let mut histogram = std::collections::BTreeMap::new();
            for i in &reference.diagnostics().instructions {
                *histogram.entry(i.kind.clone()).or_insert(0usize) += i.count;
            }
            eprintln!("router-tail reference diagnostic histogram: {histogram:?}");
        }
        let mut retained = Vec::new();
        let mut workers = Vec::new();
        for mode in 0..12 {
            let scores = (0..rows * 128)
                .map(|physical| {
                    let i = if views {
                        (physical % rows) * 128 + physical / rows
                    } else {
                        physical
                    };
                    match mode {
                        1 => {
                            if i % 2 == 0 {
                                -0.
                            } else {
                                0.
                            }
                        }
                        2 => ((i % 128) / 8) as f64,
                        3 => f32::from_bits(0x3f800000 + (i % 7) as u32) as f64,
                        4 => {
                            if i % 128 == 0 {
                                1000.
                            } else {
                                -1000.
                            }
                        }
                        5 => {
                            if i % 128 == 0 {
                                f64::NEG_INFINITY
                            } else {
                                (i % 19) as f64
                            }
                        }
                        8 => {
                            if i == 0 {
                                f64::NAN
                            } else {
                                1.
                            }
                        }
                        9 => {
                            if i == 128.min(rows * 128 - 1) {
                                f64::INFINITY
                            } else {
                                1.
                            }
                        }
                        10 => f64::NEG_INFINITY,
                        _ => ((i * 31 + 17) % 103) as f64 / 19. - 2.7,
                    }
                })
                .collect::<Vec<_>>();
            let scales = (0..128 + usize::from(views))
                .map(|i| match mode {
                    6 => match i % 5 {
                        0 => -0.,
                        1 => f64::INFINITY,
                        2 => f64::NEG_INFINITY,
                        3 => f64::NAN,
                        _ => 1.,
                    },
                    7 => half::bf16::from_bits((i as u16 & 0x7f) | 1).to_f64(),
                    _ => (i % 29) as f64 / 17. - 0.5,
                })
                .collect::<Vec<_>>();
            let bindings = vec![
                crate::CudaValue::from_host(
                    device.clone(),
                    if views {
                        vec![128, rows]
                    } else {
                        vec![rows, 128]
                    },
                    DType::F32,
                    &scores,
                )
                .unwrap(),
                crate::CudaValue::from_host(device.clone(), vec![scales.len()], dtype, &scales)
                    .unwrap(),
            ];
            let cancelled = CancellationFlag::new();
            cancelled.cancel();
            assert!(optimized.execute(&bindings, &[], &cancelled).is_err());
            let expected = reference.execute(&bindings, &[], &CancellationFlag::new());
            let actual = optimized.execute(&bindings, &[], &CancellationFlag::new());
            if (8..=10).contains(&mode) {
                assert_eq!(expected.err().unwrap(), "topKIndices: NaN input");
                assert_eq!(actual.err().unwrap(), "topKIndices: NaN input");
                continue;
            }
            let expected = expected
                .unwrap()
                .into_iter()
                .take(2)
                .map(|v| v.read_storage_bytes().unwrap())
                .collect::<Vec<_>>();
            let actual = actual.unwrap();
            for (v, bytes) in actual.iter().zip(&expected) {
                assert_eq!(
                    &v.read_storage_bytes().unwrap(),
                    bytes,
                    "rows={rows} dtype={dtype:?} views={views} mode={mode}"
                );
            }
            if mode < 2 {
                let executable = optimized.clone();
                let bytes = expected.clone();
                workers.push(std::thread::spawn(move || {
                    let out = executable
                        .execute(&bindings, &[], &CancellationFlag::new())
                        .unwrap();
                    for (v, b) in out.iter().zip(&bytes) {
                        assert_eq!(&v.read_storage_bytes().unwrap(), b);
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

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_ROUTER_TAIL=1"]
fn router_tail_mid_execution_cancel_releases_unpublished_outputs_and_recovers() {
    assert_eq!(
        std::env::var("EFFECT_TORCH_CUDA_ROUTER_TAIL").as_deref(),
        Ok("1")
    );
    let input = |slot, shape| {
        Node::new(NodeKind::Input {
            slot,
            shape,
            dtype: DType::BF16,
            device: Device::Cuda(0),
            storage: StorageMetadata::dense(),
        })
        .unwrap()
    };
    let projection = Node::new(NodeKind::Matmul {
        a: input(0, vec![64, 128]),
        b: input(2, vec![128, 128]),
    })
    .unwrap();
    let source = Node::new(NodeKind::Cast {
        a: projection,
        dtype: DType::F32,
    })
    .unwrap();
    let tail = tail_from_inputs(source, input(1, vec![128]), 8, false);
    let executable = crate::compile(vec![tail.weights, tail.indices], 0).unwrap();
    assert!(executable
        .diagnostics()
        .instructions
        .iter()
        .any(|i| i.kind == "et_router_tail"));
    let device = crate::CudaDevice::get(0).unwrap();
    let bindings = [vec![64, 128], vec![128], vec![128, 128]]
        .into_iter()
        .enumerate()
        .map(|(slot, shape)| {
            let data = (0..shape.iter().product())
                .map(|i| {
                    if slot == 2 {
                        if i / 128 == i % 128 {
                            1.
                        } else {
                            0.
                        }
                    } else {
                        ((i * 17) % 37) as f64 / 19.
                    }
                })
                .collect::<Vec<_>>();
            crate::CudaValue::from_host(device.clone(), shape, DType::BF16, &data).unwrap()
        })
        .collect::<Vec<_>>();
    let retained = executable
        .execute(&bindings, &[], &CancellationFlag::new())
        .unwrap();
    let expected = retained
        .iter()
        .map(|v| v.read_storage_bytes().unwrap())
        .collect::<Vec<_>>();
    let cancellation = CancellationFlag::new();
    let entered = std::sync::atomic::AtomicBool::new(false);
    let failed = executable.execute_with_gemm_hook(&bindings, &cancellation, &|| {
        entered.store(true, std::sync::atomic::Ordering::SeqCst);
        cancellation.cancel();
    });
    assert!(entered.load(std::sync::atomic::Ordering::SeqCst));
    assert!(failed.is_err());
    let recovered = executable
        .execute(&bindings, &[], &CancellationFlag::new())
        .unwrap();
    drop(executable);
    for values in [&retained, &recovered] {
        for (value, bytes) in values.iter().zip(&expected) {
            assert_eq!(&value.read_storage_bytes().unwrap(), bytes);
        }
    }
}

use super::*;
use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
use effect_torch_graph::Node;
use effect_torch_runtime::{CancellationFlag, StorageMetadata};
use std::sync::Arc;

fn input(slot: u32, shape: Vec<usize>, dtype: DType) -> Arc<Node> {
    Node::new(NodeKind::Input {
        slot,
        shape,
        dtype,
        device: Device::Cuda(0),
        storage: StorageMetadata::dense(),
    })
    .unwrap()
}
struct Pair {
    plain: Arc<Node>,
    noisy: Arc<Node>,
    private: Vec<Arc<Node>>,
    boundaries: [Arc<Node>; 2],
}
fn pair(processed: Arc<Node>, uniform: Arc<Node>, reverse: bool, axis: usize) -> Pair {
    let mut private = Vec::new();
    let mut make = |kind| {
        let node = Node::new(kind).unwrap();
        private.push(node.clone());
        node
    };
    let log = make(NodeKind::Log { a: uniform.clone() });
    let negative = make(NodeKind::Neg { a: log });
    let outer = make(NodeKind::Log { a: negative });
    let noise = make(NodeKind::Neg { a: outer });
    let added = make(if reverse {
        NodeKind::Add {
            a: noise,
            b: processed.clone(),
        }
    } else {
        NodeKind::Add {
            a: processed.clone(),
            b: noise,
        }
    });
    Pair {
        plain: Node::new(NodeKind::Argmax {
            a: processed.clone(),
            dim: axis,
        })
        .unwrap(),
        noisy: Node::new(NodeKind::Argmax {
            a: added,
            dim: axis,
        })
        .unwrap(),
        private,
        boundaries: [processed, uniform],
    }
}
#[test]
fn dual_argmax_private_outputs_shared_boundaries_order_axis_dtype_and_policy_guards() {
    for (shape, dtype, reverse, axis, enabled, expected) in [
        (vec![3, 4096], DType::F32, false, 1, true, true),
        (vec![1, 64, 4097], DType::F32, false, 2, true, true),
        (vec![4096], DType::F32, false, 0, true, true),
        (vec![3, 4095], DType::F32, false, 1, true, false),
        (vec![3, 4096], DType::BF16, false, 1, true, false),
        (vec![3, 4096], DType::F32, true, 1, true, false),
        (vec![3, 4096], DType::F32, false, 0, true, false),
        (vec![3, 4096], DType::F32, false, 1, false, false),
    ] {
        let p = pair(
            input(0, shape.clone(), dtype),
            input(1, shape.clone(), dtype),
            reverse,
            axis,
        );
        for exposure in 0..15 {
            let mut roots = vec![p.plain.clone(), p.noisy.clone()];
            if exposure < 10 {
                let n = p.private[exposure / 2].clone();
                roots.push(if exposure % 2 == 0 {
                    n
                } else {
                    Node::new(NodeKind::Neg { a: n }).unwrap()
                });
            }
            if exposure == 11 {
                roots.extend(p.boundaries.iter().cloned());
            }
            if exposure == 12 {
                roots = vec![
                    Node::new(NodeKind::Cast {
                        a: p.plain.clone(),
                        dtype: DType::U32,
                    })
                    .unwrap(),
                    Node::new(NodeKind::Cast {
                        a: p.noisy.clone(),
                        dtype: DType::U32,
                    })
                    .unwrap(),
                ];
            }
            if exposure == 13 {
                roots = vec![p.plain.clone()];
            }
            if exposure == 14 {
                roots = vec![p.noisy.clone()];
            }
            let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
                .prepare()
                .unwrap();
            let mut caps = CudaCapabilities::new(0, 12, 0);
            caps.dual_argmax = enabled;
            let driver = CompilerDriver::new(&prepared, &caps).unwrap();
            let selected = driver
                .optimization()
                .regions
                .iter()
                .find(|r| matches!(r, NativeRegion::DualArgmax(_)));
            assert_eq!(
                selected.is_some(),
                expected && (10..=12).contains(&exposure),
                "shape={shape:?} dtype={dtype:?} reverse={reverse} axis={axis} enabled={enabled} exposure={exposure}"
            );
            if let Some(r) = selected {
                assert_eq!(r.output_count(), 2);
                assert_eq!(r.semantic_outputs().len(), 2);
                assert_eq!(r.inputs().len(), 2);
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_DUAL_ARGMAX=1"]
fn dual_argmax_exact_i64_u32_nan_ties_views_concurrent_and_retained_outputs() {
    assert_eq!(
        std::env::var("EFFECT_TORCH_CUDA_DUAL_ARGMAX").as_deref(),
        Ok("1")
    );
    let device = crate::CudaDevice::get(0).unwrap();
    for (shape, views, aliased) in [
        (vec![3, 4097], false, false),
        (vec![1, 64, 4096], false, false),
        (vec![3, 4096], true, false),
        (vec![1, 256, 262144], false, false),
        (vec![3, 4096], false, true),
    ] {
        let width = *shape.last().unwrap();
        let rows = shape.iter().product::<usize>() / width;
        let xshape = if views {
            vec![width, rows]
        } else {
            shape.clone()
        };
        let ushape = if views {
            vec![rows, width + 1]
        } else {
            shape.clone()
        };
        let x = input(0, xshape.clone(), DType::F32);
        let u = if aliased {
            x.clone()
        } else {
            input(1, ushape.clone(), DType::F32)
        };
        let x = if views {
            Node::new(NodeKind::Permute {
                a: x,
                dims: vec![1, 0],
            })
            .unwrap()
        } else {
            x
        };
        let u = if views {
            Node::new(NodeKind::Slice {
                a: u,
                ranges: vec![(0, rows, 1), (1, width + 1, 1)],
            })
            .unwrap()
        } else {
            u
        };
        let p = pair(x, u, false, shape.len() - 1);
        let roots = vec![
            p.plain.clone(),
            p.noisy.clone(),
            Node::new(NodeKind::Cast {
                a: p.plain,
                dtype: DType::U32,
            })
            .unwrap(),
            Node::new(NodeKind::Cast {
                a: p.noisy,
                dtype: DType::U32,
            })
            .unwrap(),
        ];
        let optimized = Arc::new(crate::compile(roots.clone(), 0).unwrap());
        let mut reference_roots = roots;
        reference_roots.push(p.private[4].clone());
        let reference = crate::compile(reference_roots, 0).unwrap();
        assert_eq!(
            optimized
                .diagnostics()
                .instructions
                .iter()
                .filter(|i| i.kind == "et_dual_argmax")
                .map(|i| i.count)
                .sum::<usize>(),
            1
        );
        assert!(!reference
            .diagnostics()
            .instructions
            .iter()
            .any(|i| i.kind == "et_dual_argmax"));
        let mut retained = Vec::new();
        let mut workers = Vec::new();
        for mode in 0..if width == 262144 { 1 } else { 9 } {
            let xs = (0..xshape.iter().product())
                .map(|i| {
                    let col = if views { i / rows } else { i % width };
                    match mode {
                        1 => {
                            if col % 2 == 0 {
                                -0.
                            } else {
                                0.
                            }
                        }
                        2 => f64::NEG_INFINITY,
                        3 => f64::INFINITY,
                        4 => {
                            if col == 0 {
                                f64::NAN
                            } else {
                                1.
                            }
                        }
                        5 => {
                            if col == width - 1 {
                                f64::NAN
                            } else {
                                1.
                            }
                        }
                        6 => f64::NAN,
                        7 => {
                            if [31, 32, 1023, 1024, width - 1].contains(&col) {
                                100.
                            } else {
                                0.
                            }
                        }
                        _ => ((i * 173) % 103) as f64 / 19. - 2.7,
                    }
                })
                .collect::<Vec<_>>();
            let us = (0..ushape.iter().product())
                .map(|i| {
                    if mode == 8 {
                        [0., -0., 1., -1., f64::INFINITY, f64::NAN][i % 6]
                    } else {
                        0.5
                    }
                })
                .collect::<Vec<_>>();
            let mut bindings = vec![
                crate::CudaValue::from_host(device.clone(), xshape.clone(), DType::F32, &xs)
                    .unwrap(),
                crate::CudaValue::from_host(device.clone(), ushape.clone(), DType::F32, &us)
                    .unwrap(),
            ];
            if aliased {
                bindings.truncate(1);
            }
            let expected = reference
                .execute(&bindings, &[], &CancellationFlag::new())
                .unwrap()
                .into_iter()
                .take(4)
                .map(|v| v.read_storage_bytes().unwrap())
                .collect::<Vec<_>>();
            let cancelled = CancellationFlag::new();
            cancelled.cancel();
            assert!(optimized.execute(&bindings, &[], &cancelled).is_err());
            let actual = optimized
                .execute(&bindings, &[], &CancellationFlag::new())
                .unwrap();
            for (v, b) in actual.iter().zip(&expected) {
                assert_eq!(
                    &v.read_storage_bytes().unwrap(),
                    b,
                    "shape={shape:?} views={views} mode={mode}"
                );
            }
            if mode == 4 || mode == 6 {
                assert!(actual[0]
                    .read_storage_bytes()
                    .unwrap()
                    .iter()
                    .all(|&v| v == 0));
            }
            if mode < 2 && width < 262144 {
                let executable = optimized.clone();
                let bytes = expected.clone();
                workers.push(std::thread::spawn(move || {
                    let out = executable
                        .execute(&bindings, &[], &CancellationFlag::new())
                        .unwrap();
                    for (v, b) in out.iter().zip(bytes) {
                        assert_eq!(v.read_storage_bytes().unwrap(), b);
                    }
                }));
            }
            retained.push((actual, expected));
        }
        for w in workers {
            w.join().unwrap();
        }
        drop(optimized);
        drop(reference);
        for (vs, bs) in retained {
            for (v, b) in vs.iter().zip(bs) {
                assert_eq!(v.read_storage_bytes().unwrap(), b);
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_DUAL_ARGMAX=1"]
fn dual_argmax_mid_cancel_and_checked_failure_publish_no_outputs_then_recover() {
    assert_eq!(
        std::env::var("EFFECT_TORCH_CUDA_DUAL_ARGMAX").as_deref(),
        Ok("1")
    );
    let projection = Node::new(NodeKind::Matmul {
        a: input(0, vec![3, 16], DType::BF16),
        b: input(2, vec![16, 4096], DType::BF16),
    })
    .unwrap();
    let x = Node::new(NodeKind::Cast {
        a: projection,
        dtype: DType::F32,
    })
    .unwrap();
    let p = pair(x.clone(), input(1, vec![3, 4096], DType::F32), false, 1);
    let checked = Node::new(NodeKind::IndexSelect {
        a: x,
        dim: 1,
        indexes: input(3, vec![1], DType::U32),
    })
    .unwrap();
    let executable = crate::compile(vec![p.plain, p.noisy, checked], 0).unwrap();
    assert!(executable
        .diagnostics()
        .instructions
        .iter()
        .any(|i| i.kind == "et_dual_argmax"));
    let device = crate::CudaDevice::get(0).unwrap();
    let mut bindings = [
        (vec![3, 16], DType::BF16),
        (vec![3, 4096], DType::F32),
        (vec![16, 4096], DType::BF16),
        (vec![1], DType::U32),
    ]
    .into_iter()
    .enumerate()
    .map(|(slot, (shape, dtype))| {
        let data = (0..shape.iter().product())
            .map(|i| {
                if slot == 1 {
                    0.5
                } else if slot == 3 {
                    0.
                } else {
                    ((i * 17) % 37) as f64 / 19.
                }
            })
            .collect::<Vec<_>>();
        crate::CudaValue::from_host(device.clone(), shape, dtype, &data).unwrap()
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
    assert!(executable
        .execute_with_gemm_hook(&bindings, &cancellation, &|| {
            entered.store(true, std::sync::atomic::Ordering::SeqCst);
            cancellation.cancel();
        })
        .is_err());
    assert!(entered.load(std::sync::atomic::Ordering::SeqCst));
    bindings[3] =
        crate::CudaValue::from_host(device.clone(), vec![1], DType::U32, &[4096.]).unwrap();
    assert!(executable
        .execute(&bindings, &[], &CancellationFlag::new())
        .is_err());
    bindings[3] = crate::CudaValue::from_host(device, vec![1], DType::U32, &[0.]).unwrap();
    let recovered = executable
        .execute(&bindings, &[], &CancellationFlag::new())
        .unwrap();
    drop(executable);
    for vs in [&retained, &recovered] {
        for (v, b) in vs.iter().zip(&expected) {
            assert_eq!(&v.read_storage_bytes().unwrap(), b);
        }
    }
}

#[test]
fn dual_argmax_rng_boundary_alias_and_identity_guards() {
    for mode in 0..4 {
        let source = input(0, vec![3, 4096], DType::F32);
        let uniform = if mode == 0 {
            source.clone()
        } else {
            Node::new(NodeKind::Uniform {
                lo: 0.,
                hi: 1.,
                shape: vec![3, 4096],
                dtype: DType::F32,
                device: Device::Cuda(0),
            })
            .unwrap()
        };
        let p = pair(source.clone(), uniform.clone(), false, 1);
        let roots = match mode {
            2 => vec![
                Node::new(NodeKind::Argmax {
                    a: input(1, vec![3, 4096], DType::F32),
                    dim: 1,
                })
                .unwrap(),
                p.noisy,
            ],
            3 => vec![
                p.plain,
                Node::new(NodeKind::Argmax {
                    a: Node::new(NodeKind::Add {
                        a: source,
                        b: p.private[2].clone(),
                    })
                    .unwrap(),
                    dim: 1,
                })
                .unwrap(),
            ],
            _ => vec![p.noisy, p.plain, uniform.clone()],
        };
        let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
            .prepare()
            .unwrap();
        let mut caps = CudaCapabilities::new(0, 12, 0);
        caps.dual_argmax = true;
        let driver = CompilerDriver::new(&prepared, &caps).unwrap();
        let region = driver
            .optimization()
            .regions
            .iter()
            .find(|r| matches!(r, NativeRegion::DualArgmax(_)));
        assert_eq!(region.is_some(), mode < 2, "mode={mode}");
        if let Some(region) = region {
            let boundary = prepared.index.dense_id(uniform.id).unwrap();
            assert!(region.inputs().contains(&boundary));
            assert!(!region.nodes().contains(&boundary));
        }
    }
}

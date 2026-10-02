use super::*;
use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
use effect_torch_graph::Node;
use effect_torch_runtime::{CancellationFlag, StorageMetadata};
use std::sync::Arc;

fn chain(shape: Vec<usize>, eps: f64, dtype: DType, view: bool) -> Vec<Arc<Node>> {
    let width = *shape.last().unwrap();
    let input = |slot, shape| {
        Node::new(NodeKind::Input {
            slot,
            shape,
            dtype,
            device: Device::Cuda(0),
            storage: StorageMetadata::dense(),
        })
        .unwrap()
    };
    let matrix = |slot| {
        if view {
            let mut padded = shape.clone();
            *padded.last_mut().unwrap() += 1;
            let ranges = shape
                .iter()
                .enumerate()
                .map(|(i, &n)| {
                    if i + 1 == shape.len() {
                        (1, n + 1, 1)
                    } else {
                        (0, n, 1)
                    }
                })
                .collect();
            Node::new(NodeKind::Slice {
                a: input(slot, padded),
                ranges,
            })
            .unwrap()
        } else {
            input(slot, shape.clone())
        }
    };
    let norm = |x, slot| {
        Node::new(NodeKind::RmsNorm {
            x,
            weight: Some(input(slot, vec![width])),
            eps,
        })
        .unwrap()
    };
    let dense = norm(matrix(0), 3);
    let expert = norm(matrix(1), 4);
    let sum = Node::new(NodeKind::Add {
        a: dense.clone(),
        b: expert.clone(),
    })
    .unwrap();
    let combined = norm(sum.clone(), 5);
    let residual = Node::new(NodeKind::Add {
        a: matrix(2),
        b: combined.clone(),
    })
    .unwrap();
    let output = Node::new(NodeKind::Mul {
        a: residual.clone(),
        b: input(6, vec![1]),
    })
    .unwrap();
    vec![dense, expert, sum, combined, residual, output]
}

#[test]
fn ffn_tail_private_roots_consumers_rank_and_policy_guards() {
    for (shape, eps, dtype, enabled, expected) in [
        (vec![3, 2816], 1e-6, DType::BF16, true, true),
        (vec![1, 3, 2816], 1e-6, DType::BF16, true, true),
        (vec![2, 3, 2816], 1e-6, DType::BF16, true, false),
        (vec![3, 2815], 1e-6, DType::BF16, true, false),
        (vec![65536, 2816], 1e-6, DType::BF16, true, false),
        (vec![3, 2816], 1e-5, DType::BF16, true, false),
        (vec![3, 2816], 1e-6, DType::F32, true, false),
        (vec![3, 2816], 1e-6, DType::BF16, false, false),
    ] {
        let nodes = chain(shape.clone(), eps, dtype, false);
        for escape in 0..=10 {
            let mut roots = vec![nodes[5].clone()];
            if escape < 5 {
                roots.push(nodes[escape].clone());
            } else if escape < 10 {
                roots.push(
                    Node::new(NodeKind::Neg {
                        a: nodes[escape - 5].clone(),
                    })
                    .unwrap(),
                );
            }
            let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
                .prepare()
                .unwrap();
            let mut caps = CudaCapabilities::new(0, 12, 0);
            caps.ffn_tail = enabled;
            let driver = CompilerDriver::new(&prepared, &caps).unwrap();
            let selected = driver
                .optimization()
                .regions
                .iter()
                .find(|r| matches!(r, NativeRegion::FfnTail(_)));
            assert_eq!(
                selected.is_some(),
                expected && escape == 10,
                "shape={shape:?} eps={eps} dtype={dtype:?} enabled={enabled} escape={escape}"
            );
            if let Some(NativeRegion::FfnTail(region)) = selected {
                assert_eq!(region.shape, shape);
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_FFN_TAIL=1"]
fn ffn_tail_exact_views_specials_cancel_concurrent_retained_outputs() {
    assert_eq!(
        std::env::var("EFFECT_TORCH_CUDA_FFN_TAIL").as_deref(),
        Ok("1")
    );
    let device = crate::CudaDevice::get(0).unwrap();
    for (shape, view) in [
        (vec![1, 2816], false),
        (vec![1, 33, 2816], false),
        (vec![3, 2816], true),
        (vec![256, 2816], false),
    ] {
        let nodes = chain(shape.clone(), 1e-6, DType::BF16, view);
        let optimized = Arc::new(crate::compile(vec![nodes[5].clone()], 0).unwrap());
        let reference = crate::compile(
            std::iter::once(nodes[5].clone())
                .chain(nodes[..5].iter().cloned())
                .collect(),
            0,
        )
        .unwrap();
        assert!(optimized
            .diagnostics()
            .instructions
            .iter()
            .any(|i| i.kind == "et_ffn_tail_bf16"));
        assert!(!reference
            .diagnostics()
            .instructions
            .iter()
            .any(|i| i.kind == "et_ffn_tail_bf16"));
        let mut retained = Vec::new();
        let mut workers = Vec::new();
        for mode in 0..8 {
            let bindings = (0..7)
                .map(|slot| {
                    let mut dims = if slot < 3 {
                        shape.clone()
                    } else if slot < 6 {
                        vec![2816]
                    } else {
                        vec![1]
                    };
                    if view && slot < 3 {
                        *dims.last_mut().unwrap() += 1;
                    }
                    let data = (0..dims.iter().product())
                        .map(|i: usize| {
                            if slot == 6 {
                                return [
                                    0.713,
                                    -0.321,
                                    0.,
                                    -0.,
                                    f64::INFINITY,
                                    f64::NEG_INFINITY,
                                    f64::NAN,
                                    1.,
                                ][mode];
                            }
                            let bits = ((i * 173 + slot * 917) % 65536) as u16;
                            match mode {
                                2 => {
                                    if i % 2 == 0 {
                                        -0.
                                    } else {
                                        0.
                                    }
                                }
                                3 => half::bf16::from_bits((bits & 0x807f) | 1).to_f64(),
                                7 => half::bf16::from_bits(bits).to_f64(),
                                _ => ((i * 31 + slot * 17) % 103) as f64 / 19. - 2.7,
                            }
                        })
                        .collect::<Vec<_>>();
                    crate::CudaValue::from_host(device.clone(), dims, DType::BF16, &data).unwrap()
                })
                .collect::<Vec<_>>();
            let expected = reference
                .execute(&bindings, &[], &CancellationFlag::new())
                .unwrap()[0]
                .read_storage_bytes()
                .unwrap();
            let cancelled = CancellationFlag::new();
            cancelled.cancel();
            assert!(optimized.execute(&bindings, &[], &cancelled).is_err());
            let actual = optimized
                .execute(&bindings, &[], &CancellationFlag::new())
                .unwrap()
                .remove(0);
            assert_eq!(
                actual.read_storage_bytes().unwrap(),
                expected,
                "shape={shape:?} view={view} mode={mode}"
            );
            if mode < 2 {
                let executable = optimized.clone();
                let bytes = expected.clone();
                workers.push(std::thread::spawn(move || {
                    let output = executable
                        .execute(&bindings, &[], &CancellationFlag::new())
                        .unwrap();
                    assert_eq!(output[0].read_storage_bytes().unwrap(), bytes);
                }));
            }
            retained.push((actual, expected));
        }
        for worker in workers {
            worker.join().unwrap();
        }
        drop(optimized);
        drop(reference);
        for (value, bytes) in retained {
            assert_eq!(value.read_storage_bytes().unwrap(), bytes);
        }
    }
}

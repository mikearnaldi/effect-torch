use super::*;
use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
use effect_torch_graph::Node;
use effect_torch_runtime::{CancellationFlag, StorageMetadata};
use std::sync::Arc;

fn chain(
    shape: Vec<usize>,
    dtype: DType,
    weight_dtype: DType,
    view: bool,
    reverse: bool,
) -> [Arc<Node>; 2] {
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
    let width = *shape.last().unwrap();
    let source = if view {
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
            a: input(0, padded, dtype),
            ranges,
        })
        .unwrap()
    } else {
        input(0, shape.clone(), dtype)
    };
    let normalized = Node::new(NodeKind::RmsNorm {
        x: source,
        weight: Some(input(1, vec![width], weight_dtype)),
        eps: 1e-6,
    })
    .unwrap();
    let residual = input(2, shape, dtype);
    let output = Node::new(if reverse {
        NodeKind::Add {
            a: normalized.clone(),
            b: residual,
        }
    } else {
        NodeKind::Add {
            a: residual,
            b: normalized.clone(),
        }
    })
    .unwrap();
    [normalized, output]
}

#[test]
fn rms_residual_private_shape_dtype_order_and_policy_guards() {
    for (shape, dtype, enabled, reverse, expected) in [
        (vec![64, 2816], DType::BF16, true, false, true),
        (vec![1, 256, 2816], DType::BF16, true, false, true),
        (vec![32, 2816], DType::BF16, true, false, false),
        (vec![2, 64, 2816], DType::BF16, true, false, false),
        (vec![64, 2815], DType::BF16, true, false, false),
        (vec![64, 2816], DType::F32, true, false, false),
        (vec![64, 2816], DType::BF16, false, false, false),
        (vec![64, 2816], DType::BF16, true, true, false),
    ] {
        for exposure in 0..3 {
            let [norm, output] = chain(shape.clone(), dtype, dtype, false, reverse);
            let mut roots = vec![output];
            if exposure == 1 {
                roots.push(norm.clone());
            }
            if exposure == 2 {
                roots.push(Node::new(NodeKind::Neg { a: norm }).unwrap());
            }
            let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
                .prepare()
                .unwrap();
            let mut caps = CudaCapabilities::new(0, 12, 0);
            caps.rms_residual = enabled;
            let driver = CompilerDriver::new(&prepared, &caps).unwrap();
            let selected = driver
                .optimization()
                .regions
                .iter()
                .find(|r| matches!(r, NativeRegion::RmsResidual(_)));
            assert_eq!(
                selected.is_some(),
                expected && exposure == 0,
                "shape={shape:?} dtype={dtype:?} enabled={enabled} reverse={reverse} exposure={exposure}"
            );
            if let Some(NativeRegion::RmsResidual(region)) = selected {
                assert_eq!(region.shape.as_ref(), shape.as_slice());
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_RMS_RESIDUAL=1"]
fn rms_residual_exact_views_specials_cancel_concurrent_retained_outputs() {
    assert_eq!(
        std::env::var("EFFECT_TORCH_CUDA_RMS_RESIDUAL").as_deref(),
        Ok("1")
    );
    let device = crate::CudaDevice::get(0).unwrap();
    for (shape, view, weight_dtype) in [
        (vec![64, 2816], false, DType::BF16),
        (vec![1, 256, 2816], false, DType::BF16),
        (vec![64, 2816], true, DType::BF16),
    ] {
        let [norm, output] = chain(shape.clone(), DType::BF16, weight_dtype, view, false);
        let optimized = Arc::new(crate::compile(vec![output.clone()], 0).unwrap());
        let reference = crate::compile(vec![output, norm], 0).unwrap();
        assert!(optimized
            .diagnostics()
            .instructions
            .iter()
            .any(|i| i.kind == "et_rms_residual_bf16"));
        assert!(!reference
            .diagnostics()
            .instructions
            .iter()
            .any(|i| i.kind == "et_rms_residual_bf16"));
        let mut retained = Vec::new();
        let mut workers = Vec::new();
        for mode in 0..8 {
            let bindings = (0..3)
                .map(|slot| {
                    let mut dims = if slot == 1 { vec![2816] } else { shape.clone() };
                    if view && slot == 0 {
                        *dims.last_mut().unwrap() += 1;
                    }
                    let dtype = if slot == 1 { weight_dtype } else { DType::BF16 };
                    let data = (0..dims.iter().product())
                        .map(|i: usize| {
                            let bits = ((i * 173 + slot * 917) % 65536) as u16;
                            match mode {
                                1 => {
                                    if i % 2 == 0 {
                                        -0.
                                    } else {
                                        0.
                                    }
                                }
                                2 => half::bf16::from_bits((bits & 0x807f) | 1).to_f64(),
                                3 => {
                                    if slot == 0 {
                                        f64::INFINITY
                                    } else {
                                        1.
                                    }
                                }
                                4 => {
                                    if slot == 1 {
                                        f64::NAN
                                    } else {
                                        1.
                                    }
                                }
                                5 => {
                                    if slot == 2 {
                                        f64::NEG_INFINITY
                                    } else {
                                        1.
                                    }
                                }
                                6 => {
                                    if slot == 2 {
                                        -1.
                                    } else {
                                        1.
                                    }
                                }
                                7 => half::bf16::from_bits(bits).to_f64(),
                                _ => ((i * 31 + slot * 17) % 103) as f64 / 19. - 2.7,
                            }
                        })
                        .collect::<Vec<_>>();
                    crate::CudaValue::from_host(device.clone(), dims, dtype, &data).unwrap()
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
                "shape={shape:?} view={view} weight={weight_dtype:?} mode={mode}"
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

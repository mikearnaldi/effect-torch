//! RMS reduction accuracy, dtype boundaries, views and invocation ownership.
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
fn host(shape: &[usize], dtype: DType, data: &[f64]) -> CudaValue {
    CudaValue::from_host(CudaDevice::get(0).unwrap(), shape.to_vec(), dtype, data).unwrap()
}
fn seeded(count: usize, seed: u32) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let mut h = (i as u32).wrapping_add(seed);
            h = (h ^ (h >> 16)).wrapping_mul(0x7feb352d);
            h = (h ^ (h >> 15)).wrapping_mul(0x846ca68b);
            h ^= h >> 16;
            let bits = ((h >> 16) & 0x8000) | (((h >> 24) % 9 + 120) << 7) | ((h >> 8) & 127);
            half::bf16::from_bits(bits as u16).to_f64()
        })
        .collect()
}

#[test]
fn static_rms_requires_supported_storage_width_and_single_row_ctas() {
    use crate::executable::{rms_static2816_kernel, CudaKernelArgs};
    let mut args = CudaKernelArgs {
        operation: 2,
        elements: 256 * 2816,
        output_dtype: 1,
        ..Default::default()
    };
    args.integers[0] = 2816;
    args.input_dtypes[0] = 1;
    assert_eq!(
        rms_static2816_kernel(&args),
        Some("et_rms_norm_static2816_1_1")
    );
    args.operation = 1;
    assert_eq!(rms_static2816_kernel(&args), None);
    args.operation = 3;
    for elements in [0, 2815, 2817, 65536 * 2816] {
        args.elements = elements;
        assert_eq!(rms_static2816_kernel(&args), None);
    }
    args.elements = 65535 * 2816;
    assert!(rms_static2816_kernel(&args).is_some());
    for width in [0, 1024, 2815, 4096] {
        args.integers[0] = width;
        assert_eq!(rms_static2816_kernel(&args), None);
    }
    args.integers[0] = 2816;
    for dtype in [0, 2, 4, 5, 6] {
        args.input_dtypes[0] = dtype;
        assert_eq!(rms_static2816_kernel(&args), None);
    }
    args.input_dtypes[0] = 3;
    args.output_dtype = 3;
    assert_eq!(
        rms_static2816_kernel(&args),
        Some("et_rms_norm_static2816_3_3")
    );
    args.output_dtype = 2;
    assert_eq!(rms_static2816_kernel(&args), None);
}

#[test]
#[ignore = "requires a CUDA device"]
fn rms_fixed_independent_bf16_rounding_regression() {
    // Fixed PyTorch 2.10.0+cu128 oracle from bounded RMS replay 31. The old
    // sequential reduction produced BF16 bits 15825, 47989 and 47891.
    // These witnesses use rows 9, 14 and 31, without the diagnostic edge rows.
    let (rows, width) = (32, 2816);
    let root = Node::new(NodeKind::RmsNorm {
        x: input(0, &[rows, width], DType::BF16),
        weight: Some(input(1, &[width], DType::BF16)),
        eps: 1e-6,
    })
    .unwrap();
    let executable = crate::compile(vec![root], 0).unwrap();
    let result = executable
        .execute(
            &[
                host(&[rows, width], DType::BF16, &seeded(rows * width, 17)),
                host(&[width], DType::BF16, &seeded(width, 29)),
            ],
            &[],
            &CancellationFlag::new(),
        )
        .unwrap();
    let bytes = result[0].read_storage_bytes().unwrap();
    let actual =
        [26545, 39949, 87735].map(|i| u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]));
    assert_eq!(actual, [15826, 47988, 47890]);
}

#[test]
#[ignore = "requires a CUDA device"]
fn rms_outer_row_permute_matches_materialized_order() {
    let (batch, first, second, width) = (2, 3, 4, 256);
    let source = seeded(batch * first * second * width, 41);
    let weight = seeded(width, 73);
    let mut materialized = vec![0.0; source.len()];
    for b in 0..batch {
        for i in 0..first {
            for j in 0..second {
                let source_row = (b * first + i) * second + j;
                let destination_row = (b * second + j) * first + i;
                materialized[destination_row * width..(destination_row + 1) * width]
                    .copy_from_slice(&source[source_row * width..(source_row + 1) * width]);
            }
        }
    }
    let weight_node = || input(1, &[width], DType::BF16);
    let view_root = Node::new(NodeKind::RmsNorm {
        x: Node::new(NodeKind::Permute {
            a: input(0, &[batch, first, second, width], DType::BF16),
            dims: vec![0, 2, 1, 3],
        })
        .unwrap(),
        weight: Some(weight_node()),
        eps: 1e-6,
    })
    .unwrap();
    let dense_root = Node::new(NodeKind::RmsNorm {
        x: input(0, &[batch, second, first, width], DType::BF16),
        weight: Some(weight_node()),
        eps: 1e-6,
    })
    .unwrap();
    let view = crate::compile(vec![view_root], 0)
        .unwrap()
        .execute(
            &[
                host(&[batch, first, second, width], DType::BF16, &source),
                host(&[width], DType::BF16, &weight),
            ],
            &[],
            &CancellationFlag::new(),
        )
        .unwrap()[0]
        .read_storage_bytes()
        .unwrap();
    let dense = crate::compile(vec![dense_root], 0)
        .unwrap()
        .execute(
            &[
                host(&[batch, second, first, width], DType::BF16, &materialized),
                host(&[width], DType::BF16, &weight),
            ],
            &[],
            &CancellationFlag::new(),
        )
        .unwrap()[0]
        .read_storage_bytes()
        .unwrap();
    assert_eq!(view, dense);
}

#[test]
#[ignore = "requires a CUDA device"]
fn rms_widths_tails_views_and_dtype_contract() {
    for dtype in [DType::F32, DType::BF16, DType::F16, DType::F64] {
        for (rows, width) in [
            (0, 7),
            (3, 0),
            (1, 1),
            (3, 7),
            (7, 31),
            (2, 32),
            (19, 33),
            (1, 127),
            (7, 128),
            (16, 129),
            (33, 255),
            (19, 256),
            (3, 257),
            (1, 511),
            (17, 512),
            (7, 513),
            (3, 1023),
            (33, 2048),
            (32, 2816),
            (17, 4097),
            (2, 8193),
            (256, 512),
            (256, 2816),
            (4096, 256),
            (2048, 512),
        ] {
            for weighted in [false, true] {
                let x = seeded(rows * width, 17);
                let w = seeded(width, 29);
                // Transpose both the input view and its physical host values.
                let mut xt = vec![0.; x.len()];
                for r in 0..rows {
                    for c in 0..width {
                        xt[c * rows + r] = x[r * width + c];
                    }
                }
                let root = Node::new(NodeKind::RmsNorm {
                    x: Node::new(NodeKind::Permute {
                        a: input(0, &[width, rows], dtype),
                        dims: vec![1, 0],
                    })
                    .unwrap(),
                    weight: weighted.then(|| input(1, &[width], dtype)),
                    eps: 1e-6,
                })
                .unwrap();
                let executable = crate::compile(vec![root], 0).unwrap();
                let mut bindings = vec![host(&[width, rows], dtype, &xt)];
                if weighted {
                    bindings.push(host(&[width], dtype, &w));
                }
                let out = executable
                    .execute(&bindings, &[], &CancellationFlag::new())
                    .unwrap()[0]
                    .readback()
                    .unwrap();
                for r in 0..rows {
                    let sum = x[r * width..(r + 1) * width]
                        .iter()
                        .map(|v| v * v)
                        .sum::<f64>();
                    let inv = (sum / width as f64 + 1e-6).sqrt().recip();
                    for c in 0..width {
                        let expected = x[r * width + c] * inv * if weighted { w[c] } else { 1. };
                        let relative = match dtype {
                            DType::BF16 => 0.004,
                            DType::F16 => 0.0005,
                            DType::F64 => 1e-12,
                            _ => 2e-6,
                        };
                        let atol = if dtype == DType::F16 { 3e-8 } else { 1e-14 };
                        assert!(
                            (out[r * width + c] - expected).abs()
                                <= atol + expected.abs() * relative,
                            "{dtype:?} [{rows},{width}] weighted={weighted} at {r},{c}: {} != {expected}",
                            out[r * width + c]
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn rms_broadcast_extreme_finite_cancellation_and_retained_outputs() {
    for dtype in [DType::F32, DType::BF16] {
        let width = 513;
        let root = Node::new(NodeKind::RmsNorm {
            x: Node::new(NodeKind::BroadcastTo {
                a: input(0, &[1, width], dtype),
                shape: vec![17, width],
            })
            .unwrap(),
            weight: Some(
                Node::new(NodeKind::BroadcastTo {
                    a: input(1, &[1], dtype),
                    shape: vec![width],
                })
                .unwrap(),
            ),
            eps: 1e-6,
        })
        .unwrap();
        let executable = Arc::new(crate::compile(vec![root], 0).unwrap());
        let mut retained = Vec::new();
        for value in [0., 1e-37, 1., 1e16, 1e30] {
            let bindings = [
                host(&[1, width], dtype, &vec![value; width]),
                host(&[1], dtype, &[2.]),
            ];
            let cancelled = CancellationFlag::new();
            cancelled.cancel();
            assert!(executable.execute(&bindings, &[], &cancelled).is_err());
            let result = executable
                .execute(&bindings, &[], &CancellationFlag::new())
                .unwrap();
            let stored = bindings[0].readback().unwrap()[0];
            // F32 sum-of-squares overflow is part of the existing opmath contract.
            let expected = if value > 1e20 {
                0.
            } else {
                2. * stored / (stored * stored + 1e-6).sqrt()
            };
            for v in result[0].readback().unwrap() {
                assert!(
                    (v - expected).abs() <= expected.abs() * 0.004 + 1e-38,
                    "{dtype:?} {value}: {v} != {expected}"
                );
            }
            retained.push(result[0].clone());
        }
        let workers = (0..3)
            .map(|_| {
                let executable = executable.clone();
                std::thread::spawn(move || {
                    executable
                        .execute(
                            &[
                                host(&[1, width], dtype, &vec![1.; width]),
                                host(&[1], dtype, &[2.]),
                            ],
                            &[],
                            &CancellationFlag::new(),
                        )
                        .unwrap()[0]
                        .readback()
                        .unwrap()
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            assert!(worker.join().unwrap().iter().all(|v| (v - 2.).abs() < 2e-6));
        }
        drop(executable);
        assert_eq!(retained[0].readback().unwrap(), vec![0.; 17 * width]);
    }
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_SHARED_RMS=1"]
fn shared_rms_exact_products_views_retained_outputs_and_cancellation() {
    use effect_torch_compiler::CompileOptions;
    assert_eq!(std::env::var("EFFECT_TORCH_CUDA_SHARED_RMS").unwrap(), "1");
    for dtype in [DType::BF16, DType::F32] {
        for width in [1024, 1025, 2816] {
            let rows = 11;
            let x = input(0, &[1, rows, width], dtype);
            let view = Node::new(NodeKind::Reshape {
                a: x.clone(),
                shape: vec![rows, width],
            })
            .unwrap();
            let weight = input(1, &[width], dtype);
            let roots = vec![
                Node::new(NodeKind::RmsNorm {
                    x: view.clone(),
                    weight: None,
                    eps: 1e-6,
                })
                .unwrap(),
                Node::new(NodeKind::RmsNorm {
                    x: x.clone(),
                    weight: Some(weight),
                    eps: 1e-6,
                })
                .unwrap(),
                Node::new(NodeKind::RmsNorm {
                    x,
                    weight: Some(input(2, &[width], dtype)),
                    eps: 1e-6,
                })
                .unwrap(),
                view,
            ];
            let optimized =
                crate::compile_with_options(roots.clone(), 0, CompileOptions::default()).unwrap();
            assert!(optimized
                .diagnostics()
                .instructions
                .iter()
                .any(|i| i.kind == "et_shared_rms_norm_f32" && i.count == 1));
            let reference = crate::compile_with_options(
                roots,
                0,
                CompileOptions {
                    optimize: false,
                    ..Default::default()
                },
            )
            .unwrap();
            let mut retained = Vec::new();
            for seed in [17, 29, 41] {
                let mut values = seeded(rows * width, seed);
                for column in 0..width {
                    values[column] = -0.0;
                    values[width + column] = if column % 2 == 0 { -0.0 } else { 0.0 };
                    values[2 * width + column] = f32::from_bits(1 + column as u32 % 17) as f64;
                    values[3 * width + column] = 1e30;
                    values[4 * width + column] = if column == 0 { f64::NAN } else { 1.0 };
                    values[5 * width + column] = if column == 0 { f64::INFINITY } else { -1.0 };
                    values[6 * width + column] = if column == 0 { f64::NEG_INFINITY } else { 1.0 };
                }
                let bindings = [
                    host(&[1, rows, width], dtype, &values),
                    host(&[width], dtype, &seeded(width, seed + 101)),
                    host(
                        &[width],
                        dtype,
                        &(0..width)
                            .map(|i| [-0.0, 0.0, 1.0, -1.0][i % 4])
                            .collect::<Vec<_>>(),
                    ),
                ];
                let input_bytes = bindings[0].read_storage_bytes().unwrap();
                let cancelled = CancellationFlag::new();
                cancelled.cancel();
                assert!(optimized.execute(&bindings, &[], &cancelled).is_err());
                let expected = reference
                    .execute(&bindings, &[], &CancellationFlag::new())
                    .unwrap();
                let actual = optimized
                    .execute(&bindings, &[], &CancellationFlag::new())
                    .unwrap();
                for (left, right) in actual.iter().zip(&expected) {
                    let expected_bytes = right.read_storage_bytes().unwrap();
                    assert_eq!(
                        left.read_storage_bytes().unwrap(),
                        expected_bytes,
                        "{dtype:?} width={width} seed={seed}"
                    );
                    retained.push((left.clone(), expected_bytes));
                }
                assert_eq!(bindings[0].read_storage_bytes().unwrap(), input_bytes);
            }
            // All slices survive subsequent calls, executable drop, and owner
            // handle release. Each retained slice keeps the allocation lease.
            drop(optimized);
            drop(reference);
            for (output, expected) in retained {
                assert_eq!(output.read_storage_bytes().unwrap(), expected);
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_SHARED_RMS=1"]
fn shared_rms_materializes_outer_permute_before_shared_reduction() {
    use effect_torch_compiler::CompileOptions;
    assert_eq!(std::env::var("EFFECT_TORCH_CUDA_SHARED_RMS").unwrap(), "1");
    let width = 1024;
    let source = Node::new(NodeKind::Permute {
        a: input(0, &[2, 3, width], DType::BF16),
        dims: vec![1, 0, 2],
    })
    .unwrap();
    let roots = vec![
        Node::new(NodeKind::RmsNorm {
            x: source.clone(),
            weight: None,
            eps: 1e-6,
        })
        .unwrap(),
        Node::new(NodeKind::RmsNorm {
            x: source,
            weight: Some(input(1, &[width], DType::BF16)),
            eps: 1e-6,
        })
        .unwrap(),
    ];
    let bindings = [
        host(&[2, 3, width], DType::BF16, &seeded(6 * width, 17)),
        host(&[width], DType::BF16, &seeded(width, 31)),
    ];
    let optimized =
        crate::compile_with_options(roots.clone(), 0, CompileOptions::default()).unwrap();
    assert!(optimized
        .diagnostics()
        .instructions
        .iter()
        .any(|i| i.kind == "et_shared_rms_norm_f32"));
    let reference = crate::compile_with_options(
        roots,
        0,
        CompileOptions {
            optimize: false,
            ..Default::default()
        },
    )
    .unwrap();
    let expected = reference
        .execute(&bindings, &[], &CancellationFlag::new())
        .unwrap();
    let actual = optimized
        .execute(&bindings, &[], &CancellationFlag::new())
        .unwrap();
    for (left, right) in actual.iter().zip(expected) {
        assert_eq!(
            left.read_storage_bytes().unwrap(),
            right.read_storage_bytes().unwrap()
        );
    }
}

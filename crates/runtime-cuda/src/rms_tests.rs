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
                        assert!((out[r*width+c]-expected).abs() <= atol + expected.abs()*relative, "{dtype:?} [{rows},{width}] weighted={weighted} at {r},{c}: {} != {expected}",out[r*width+c]);
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

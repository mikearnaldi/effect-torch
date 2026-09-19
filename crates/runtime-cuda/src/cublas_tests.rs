//! Hardware numerical tests for the native row-major BF16 GEMM path.
//!
//! Hardware tests are ignored by default and require a CUDA device when run.
//! They cover row-oriented linearRows weights, bias rounding
//! boundaries, BF16 subnormal operands, odd tail dimensions, and strided
//! batched broadcast parity.

use crate::{CudaDevice, CudaValue};
use effect_torch_graph::{Device, Node, NodeKind};
use effect_torch_runtime::{CancellationFlag, DType, StorageMetadata};
use half::bf16;
use std::sync::Arc;

fn device() -> Arc<CudaDevice> {
    CudaDevice::get(0).expect("CUDA device 0 is required for this test")
}

fn bf16_round(value: f32) -> f64 {
    bf16::from_f32(value).to_f64()
}

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

fn host(device: &Arc<CudaDevice>, shape: Vec<usize>, values: &[f64]) -> CudaValue {
    CudaValue::from_host(device.clone(), shape, DType::BF16, values).unwrap()
}

fn run(roots: Vec<Arc<Node>>, bindings: Vec<CudaValue>) -> Vec<CudaValue> {
    let executable = crate::compile(roots, 0).unwrap();
    assert!(executable
        .diagnostics()
        .instructions
        .iter()
        .any(|instruction| instruction.kind == "cublas_bf16_gemm_f32_accum"));
    assert_eq!(
        executable
            .diagnostics()
            .legalization
            .materialized_conversions,
        0
    );
    executable
        .execute(&bindings, &[], &CancellationFlag::new())
        .unwrap()
}

fn reference_linear(x: &[f64], xs: &[usize], w: &[f64], ws: &[usize], bias: &[f64]) -> Vec<f64> {
    let m = xs[xs.len() - 2];
    let k = xs[xs.len() - 1];
    let n = ws[1];
    let batch = xs[..xs.len() - 2].iter().product::<usize>();
    let mut out = vec![0.0; batch * m * n];
    for b in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for p in 0..k {
                    acc += (x[b * m * k + i * k + p] as f32) * (w[p * n + j] as f32);
                }
                out[(b * m + i) * n + j] = bf16_round(acc + bias[j] as f32);
            }
        }
    }
    out
}

fn reference_matmul(x: &[f64], xs: &[usize], w: &[f64], ws: &[usize]) -> Vec<f64> {
    let m = xs[xs.len() - 2];
    let k = xs[xs.len() - 1];
    let n = ws[ws.len() - 1];
    let a_batch = xs[..xs.len() - 2].iter().product::<usize>();
    let b_batch = ws[..ws.len() - 2].iter().product::<usize>();
    let batch = a_batch.max(b_batch);
    let mut out = vec![0.0; batch * m * n];
    for b in 0..batch {
        let ab = if a_batch == 1 { 0 } else { b };
        let bb = if b_batch == 1 { 0 } else { b };
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for p in 0..k {
                    acc += (x[(ab * m + i) * k + p] as f32) * (w[(bb * k + p) * n + j] as f32);
                }
                out[(b * m + i) * n + j] = bf16_round(acc);
            }
        }
    }
    out
}

fn assert_close(found: &[f64], expected: &[f64]) {
    assert_eq!(found.len(), expected.len());
    for (index, (found, expected)) in found.iter().zip(expected.iter()).enumerate() {
        let tolerance = 1e-3f64.max(expected.abs() * 1e-2);
        assert!(
            (found - expected).abs() <= tolerance,
            "index {index}: found {found}, expected {expected}"
        );
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn linear_rows_bf16_bias_rounds_once() {
    let device = device();
    // The GEMM accumulator is 257, which would round to 256 before a bias
    // whose exact F32 sum is 1. Computing bias before the single BF16
    // rounding boundary must yield 1.
    let x = input(0, &[1, 2]);
    let weight = input(1, &[1, 2]);
    let bias = input(2, &[1]);
    let transposed = Node::new(NodeKind::Permute {
        a: weight.clone(),
        dims: vec![1, 0],
    })
    .unwrap();
    let root = Node::new(NodeKind::Linear {
        x,
        weight: transposed,
        bias,
    })
    .unwrap();
    let bindings = vec![
        host(&device, vec![1, 2], &[256.0, 1.0]),
        host(&device, vec![1, 2], &[1.0, 1.0]),
        host(&device, vec![1], &[-256.0]),
    ];
    let output = run(vec![root], bindings);
    assert_eq!(output[0].readback().unwrap(), vec![1.0]);

    // 257 is not BF16-representable. Adding bias 0.75 before rounding gives
    // 257.75, which rounds to 258. Rounding the GEMM result first would give
    // 256, then 256.75, which rounds back to 256.
    let x = input(0, &[1, 2]);
    let weight = input(1, &[2, 1]);
    let bias = input(2, &[1]);
    let root = Node::new(NodeKind::Linear { x, weight, bias }).unwrap();
    let bindings = vec![
        host(&device, vec![1, 2], &[256.0, 1.0]),
        host(&device, vec![2, 1], &[1.0, 1.0]),
        host(&device, vec![1], &[0.75]),
    ];
    let output = run(vec![root], bindings);
    assert_eq!(output[0].readback().unwrap(), vec![258.0]);
}

#[test]
#[ignore = "requires a CUDA device"]
fn linear_bf16_subnormal_operand_times_large_normal_survives() {
    let device = device();
    let subnormal = 2f64.powi(-130);
    let large = 2f64.powi(100);
    // Activation-side subnormal.
    let x = input(0, &[1, 2]);
    let weight = input(1, &[2, 1]);
    let bias = input(2, &[1]);
    let root = Node::new(NodeKind::Linear { x, weight, bias }).unwrap();
    let bindings = vec![
        host(&device, vec![1, 2], &[subnormal, 0.0]),
        host(&device, vec![2, 1], &[large, 0.0]),
        host(&device, vec![1], &[0.0]),
    ];
    let output = run(vec![root], bindings);
    assert_eq!(output[0].readback().unwrap(), vec![2f64.powi(-30)]);

    // Weight-side subnormal through the row-oriented transpose fold.
    let x = input(0, &[1, 2]);
    let weight = input(1, &[1, 2]);
    let bias = input(2, &[1]);
    let transposed = Node::new(NodeKind::Permute {
        a: weight.clone(),
        dims: vec![1, 0],
    })
    .unwrap();
    let root = Node::new(NodeKind::Linear {
        x,
        weight: transposed,
        bias,
    })
    .unwrap();
    let bindings = vec![
        host(&device, vec![1, 2], &[large, 0.0]),
        host(&device, vec![1, 2], &[subnormal, 0.0]),
        host(&device, vec![1], &[0.0]),
    ];
    let output = run(vec![root], bindings);
    assert_eq!(output[0].readback().unwrap(), vec![2f64.powi(-30)]);
}

#[test]
#[ignore = "requires a CUDA device"]
fn linear_bf16_tail_dimensions_match_reference() {
    let device = device();
    // Row-oriented [N=7, K=5] weight through the transpose fold.
    let x = input(0, &[3, 5]);
    let weight = input(1, &[7, 5]);
    let bias = input(2, &[7]);
    let transposed = Node::new(NodeKind::Permute {
        a: weight.clone(),
        dims: vec![1, 0],
    })
    .unwrap();
    let root = Node::new(NodeKind::Linear {
        x,
        weight: transposed,
        bias,
    })
    .unwrap();
    let x_values = [
        1.0, 2.0, 4.0, 8.0, 16.0, -1.0, -2.0, -4.0, -8.0, -16.0, 0.5, 0.25, 0.125, 0.0625, 0.03125,
    ];
    let w_values = [
        1.0, 0.5, 0.25, 2.0, 4.0, -1.0, -0.5, -0.25, -2.0, -4.0, 8.0, 16.0, 0.125, 0.0625, 32.0,
        1.0, 1.0, 1.0, 1.0, 1.0, -3.0, -3.0, -3.0, -3.0, -3.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ];
    let bias_values = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0];
    let bindings = vec![
        host(&device, vec![3, 5], &x_values),
        host(&device, vec![7, 5], &w_values),
        host(&device, vec![7], &bias_values),
    ];
    let x_exact = bindings[0].readback().unwrap();
    let w_exact = bindings[1].readback().unwrap();
    let bias_exact = bindings[2].readback().unwrap();
    let logical_weight = (0..5)
        .flat_map(|k| (0..7).map(move |n| (n, k)))
        .map(|(n, k)| w_exact[n * 5 + k])
        .collect::<Vec<_>>();
    let expected = reference_linear(&x_exact, &[3, 5], &logical_weight, &[5, 7], &bias_exact);
    let output = run(vec![root], bindings);
    assert_close(&output[0].readback().unwrap(), &expected);
}

#[test]
#[ignore = "requires a CUDA device"]
fn matmul_bf16_batched_broadcast_matches_reference() {
    let device = device();
    let x = input(0, &[2, 3, 4]);
    let weight = input(1, &[4, 5]);
    let root = Node::new(NodeKind::Matmul { a: x, b: weight }).unwrap();
    let x_values = (0..24)
        .map(|value| (value % 7) as f64 - 3.0)
        .collect::<Vec<_>>();
    let w_values = (0..20)
        .map(|value| (value % 5) as f64 - 2.0)
        .collect::<Vec<_>>();
    let bindings = vec![
        host(&device, vec![2, 3, 4], &x_values),
        host(&device, vec![4, 5], &w_values),
    ];
    let x_exact = bindings[0].readback().unwrap();
    let w_exact = bindings[1].readback().unwrap();
    let expected = reference_matmul(&x_exact, &[2, 3, 4], &w_exact, &[4, 5]);
    let output = run(vec![root], bindings);
    assert_close(&output[0].readback().unwrap(), &expected);
}

#[test]
#[ignore = "requires a CUDA device"]
fn matmul_bf16_strided_batch_matches_reference() {
    let device = device();
    let x = input(0, &[2, 3, 4]);
    let weight = input(1, &[2, 4, 5]);
    let root = Node::new(NodeKind::Matmul { a: x, b: weight }).unwrap();
    let x_values = (0..24)
        .map(|value| (value % 5) as f64 - 2.0)
        .collect::<Vec<_>>();
    let w_values = (0..40)
        .map(|value| (value % 3) as f64 - 1.0)
        .collect::<Vec<_>>();
    let bindings = vec![
        host(&device, vec![2, 3, 4], &x_values),
        host(&device, vec![2, 4, 5], &w_values),
    ];
    let x_exact = bindings[0].readback().unwrap();
    let w_exact = bindings[1].readback().unwrap();
    let expected = reference_matmul(&x_exact, &[2, 3, 4], &w_exact, &[2, 4, 5]);
    let output = run(vec![root], bindings);
    assert_close(&output[0].readback().unwrap(), &expected);
}

#[test]
fn native_gemm_geometry_handles_batch_broadcast_and_rejects_partial_broadcast() {
    use crate::cublas::{plan_row_bf16_gemm, RowGemmKind};
    for (x, w, out, stride_x, stride_weight, batch) in [
        (vec![2, 3, 5], vec![5, 7], vec![2, 3, 7], 15, 0, 2),
        (vec![3, 5], vec![2, 5, 7], vec![2, 3, 7], 0, 35, 2),
        (vec![2, 3, 5], vec![2, 5, 7], vec![2, 3, 7], 15, 35, 2),
        (vec![2, 3, 5], vec![1, 2, 5, 7], vec![1, 2, 3, 7], 15, 35, 2),
        (vec![3, 5], vec![1, 1, 5, 7], vec![1, 1, 3, 7], 15, 35, 1),
    ] {
        let plan = plan_row_bf16_gemm(RowGemmKind::Matmul, &x, &w, &out).unwrap();
        assert_eq!(
            (plan.stride_x, plan.stride_weight, plan.batch),
            (stride_x, stride_weight, batch)
        );
    }
    assert!(plan_row_bf16_gemm(
        RowGemmKind::Matmul,
        &[2, 1, 3, 5],
        &[1, 4, 5, 7],
        &[2, 4, 3, 7]
    )
    .is_none());
    for (x, w, out) in [
        (vec![0, 5], vec![5, 7], vec![0, 7]),
        (vec![3, 0], vec![0, 7], vec![3, 7]),
        (vec![3, 5], vec![5, 0], vec![3, 0]),
        (vec![0, 3, 5], vec![5, 7], vec![0, 3, 7]),
        (vec![3, 5], vec![4, 7], vec![3, 7]),
        (
            vec![i32::MAX as usize + 1, 5],
            vec![5, 7],
            vec![i32::MAX as usize + 1, 7],
        ),
    ] {
        assert!(plan_row_bf16_gemm(RowGemmKind::Matmul, &x, &w, &out).is_none());
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn bf16_native_range_and_subnormals_on_aligned_tiles() {
    let device = device();
    for transposed in [false, true] {
        for (a, b) in [
            (2f64.powi(-133), 2f64.powi(100)),
            (2f64.powi(100), 2f64.powi(-133)),
            (2f64.powi(80), 2f64.powi(-60)),
            (-2f64.powi(80), 2f64.powi(-60)),
        ] {
            let x = input(0, &[16, 16]);
            let w = input(1, &[16, 16]);
            let w = if transposed {
                Node::new(NodeKind::Permute {
                    a: w,
                    dims: vec![1, 0],
                })
                .unwrap()
            } else {
                w
            };
            let root = Node::new(NodeKind::Matmul { a: x, b: w }).unwrap();
            let mut xs = vec![0.; 256];
            let mut ws = vec![0.; 256];
            for i in 0..16 {
                xs[i * 16 + i] = a;
                ws[i * 16 + i] = b;
            }
            let out = run(
                vec![root],
                vec![
                    host(&device, vec![16, 16], &xs),
                    host(&device, vec![16, 16], &ws),
                ],
            );
            let result = out[0].readback().unwrap();
            for i in 0..16 {
                for j in 0..16 {
                    assert_eq!(result[i * 16 + j], if i == j { a * b } else { 0. });
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn linear_rows_batched_raw_weight_bytes_and_output_ownership() {
    let device = device();
    let x = input(0, &[2, 2, 3]);
    let w = input(1, &[2, 3]);
    let bias = input(2, &[2]);
    let w = Node::new(NodeKind::Permute {
        a: w,
        dims: vec![1, 0],
    })
    .unwrap();
    let root = Node::new(NodeKind::Linear { x, weight: w, bias }).unwrap();
    let bytes = [1., 2., 4., -1., 0.5, 0.]
        .into_iter()
        .flat_map(|v| bf16::from_f32(v).to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    let weight =
        CudaValue::from_dense_bytes(device.clone(), vec![2, 3], DType::BF16, &bytes).unwrap();
    assert_eq!(weight.storage_bytes(), 12);
    let pointer = weight.storage_address();
    let bindings = vec![
        host(
            &device,
            vec![2, 2, 3],
            &[1., 2., 3., -1., 0., 2., 1., 2., 3., -1., 0., 2.],
        ),
        weight.clone(),
        host(&device, vec![2], &[0.25, -0.25]),
    ];
    let executable = crate::compile(vec![root], 0).unwrap();
    assert_eq!(
        executable
            .diagnostics()
            .legalization
            .materialized_conversions,
        0
    );
    let first = executable
        .execute(&bindings, &[], &CancellationFlag::new())
        .unwrap();
    let second = executable
        .execute(&bindings, &[], &CancellationFlag::new())
        .unwrap();
    assert_eq!(weight.storage_address(), pointer);
    assert_eq!(weight.read_storage_bytes().unwrap(), bytes);
    drop(executable);
    drop(bindings);
    drop(weight);
    for out in [first, second] {
        assert_eq!(
            out[0].readback().unwrap(),
            vec![17.25, -0.25, 7.25, 0.75, 17.25, -0.25, 7.25, 0.75]
        );
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn matmul_bf16_broadcast_activation_matches_reference() {
    let device = device();
    let root = Node::new(NodeKind::Matmul {
        a: input(0, &[3, 5]),
        b: input(1, &[2, 5, 7]),
    })
    .unwrap();
    let xs = (0..15).map(|v| (v % 5) as f64 - 2.).collect::<Vec<_>>();
    let ws = (0..70).map(|v| (v % 7) as f64 - 3.).collect::<Vec<_>>();
    let expected = reference_matmul(&xs, &[3, 5], &ws, &[2, 5, 7]);
    let out = run(
        vec![root],
        vec![
            host(&device, vec![3, 5], &xs),
            host(&device, vec![2, 5, 7], &ws),
        ],
    );
    assert_eq!(out[0].readback().unwrap(), expected);
}

#[test]
#[ignore = "requires a CUDA device"]
fn bf16_zero_dimensions_use_the_declared_reference_route() {
    let device = device();
    for (m, n, k) in [(0, 3, 5), (3, 0, 5), (3, 5, 0)] {
        let root = Node::new(NodeKind::Linear {
            x: input(0, &[m, k]),
            weight: input(1, &[k, n]),
            bias: input(2, &[n]),
        })
        .unwrap();
        let executable = crate::compile(vec![root], 0).unwrap();
        assert!(!executable
            .diagnostics()
            .instructions
            .iter()
            .any(|i| i.kind == "cublas_bf16_gemm_f32_accum"));
        let out = executable
            .execute(
                &[
                    host(&device, vec![m, k], &vec![0.; m * k]),
                    host(&device, vec![k, n], &vec![0.; k * n]),
                    host(&device, vec![n], &vec![0.5; n]),
                ],
                &[],
                &CancellationFlag::new(),
            )
            .unwrap();
        assert_eq!(out[0].readback().unwrap(), vec![0.5; m * n]);
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn bf16_gemm_error_cleanup_cancellation_and_concurrent_invocations() {
    let device = device();
    let root = Node::new(NodeKind::Matmul {
        a: input(0, &[2, 3]),
        b: input(1, &[3, 2]),
    })
    .unwrap();
    // An invalid late binding fails after GEMM submission. The invocation fence
    // must finish that work before its output and cuBLAS workspace are reused.
    let executable = Arc::new(crate::compile(vec![root, input(2, &[1])], 0).unwrap());
    let bindings = vec![
        host(&device, vec![2, 3], &[1.; 6]),
        host(&device, vec![3, 2], &[2.; 6]),
        host(&device, vec![1], &[0.]),
    ];
    let mut invalid = bindings.clone();
    invalid[2] = host(&device, vec![2], &[0.; 2]);
    assert!(executable
        .execute(&invalid, &[], &CancellationFlag::new())
        .is_err());
    let cancelled = CancellationFlag::new();
    cancelled.cancel();
    assert!(executable.execute(&bindings, &[], &cancelled).is_err());
    let workers = (1..=4)
        .map(|factor| {
            let executable = executable.clone();
            let mut bindings = bindings.clone();
            bindings[0] = host(&device, vec![2, 3], &[factor as f64; 6]);
            std::thread::spawn(move || {
                let result = executable
                    .execute(&bindings, &[], &CancellationFlag::new())
                    .unwrap();
                assert_eq!(result[0].readback().unwrap(), vec![6. * factor as f64; 4]);
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        worker.join().unwrap();
    }
}

fn normal_operands_produce_subnormal_outputs(with_bias: bool) {
    let device = device();
    for (m, k, n) in [(1, 2, 1), (16, 16, 16)] {
        for row_weight in [false, true] {
            let x = input(0, &[m, k]);
            let ws = if row_weight { vec![n, k] } else { vec![k, n] };
            let w = input(1, &ws);
            let w = if row_weight {
                Node::new(NodeKind::Permute {
                    a: w,
                    dims: vec![1, 0],
                })
                .unwrap()
            } else {
                w
            };
            let root = Node::new(if with_bias {
                NodeKind::Linear {
                    x,
                    weight: w,
                    bias: input(2, &[n]),
                }
            } else {
                NodeKind::Matmul { a: x, b: w }
            })
            .unwrap();
            let mut xs = vec![0.; m * k];
            let mut weights = vec![0.; n * k];
            for i in 0..m {
                xs[i * k + i] = 2f64.powi(-126);
            }
            for i in 0..n {
                weights[i * if row_weight { k } else { n } + i] = 0.5;
            }
            let mut bindings = vec![host(&device, vec![m, k], &xs), host(&device, ws, &weights)];
            if with_bias {
                bindings.push(host(&device, vec![n], &vec![0.; n]));
            }
            let output = run(vec![root], bindings);
            // BF16 2^-127 is subnormal 0x0040. Check storage bits directly so
            // neither a host conversion nor an absolute tolerance can hide FTZ.
            let bits = output[0].read_storage_bytes().unwrap();
            for i in 0..m {
                for j in 0..n {
                    let offset = 2 * (i * n + j);
                    assert_eq!(
                        u16::from_le_bytes([bits[offset], bits[offset + 1]]),
                        if i == j { 0x0040 } else { 0 }
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn matmul_bf16_normal_operands_produce_subnormal_outputs() {
    normal_operands_produce_subnormal_outputs(false);
}

#[test]
#[ignore = "requires a CUDA device"]
fn linear_bf16_normal_operands_produce_subnormal_outputs() {
    normal_operands_produce_subnormal_outputs(true);
}

#[test]
#[ignore = "requires a CUDA device"]
fn linear_bf16_status_free_epilogues_and_late_cancellation() {
    let device = device();
    let x = input(0, &[2, 3]);
    let w = input(1, &[3, 3]);
    let bias = input(2, &[3]);
    let first = Node::new(NodeKind::Linear {
        x,
        weight: w.clone(),
        bias: bias.clone(),
    })
    .unwrap();
    let root = Node::new(NodeKind::Linear {
        x: first,
        weight: w,
        bias,
    })
    .unwrap();
    let executable = crate::compile(vec![root], 0).unwrap();
    assert_eq!(executable.diagnostics().command_count, 4);
    assert_eq!(executable.diagnostics().synchronization_count, 1);
    let bindings = vec![
        host(&device, vec![2, 3], &[1., 2., 3., 4., 5., 6.]),
        host(&device, vec![3, 3], &[1., 0., 0., 0., 1., 0., 0., 0., 1.]),
        host(&device, vec![3], &[0.5, -0.5, 1.]),
    ];
    let retained = executable
        .execute(&bindings, &[], &CancellationFlag::new())
        .unwrap();
    let cancelled = CancellationFlag::new();
    let submissions = std::cell::Cell::new(0);
    let result = executable.execute_with_gemm_hook(&bindings, &cancelled, &|| {
        submissions.set(submissions.get() + 1);
        cancelled.cancel();
    });
    assert_eq!(submissions.get(), 1);
    assert!(matches!(result, Err(ref error) if error == "operation aborted"));
    // A fresh invocation can reuse scratch after the cancelled submission has
    // drained. The older escaping output still owns its original allocation.
    let next = executable
        .execute(&bindings, &[], &CancellationFlag::new())
        .unwrap();
    drop(executable);
    drop(bindings);
    for output in [retained, next] {
        assert_eq!(output[0].readback().unwrap(), vec![2., 1., 5., 5., 4., 8.]);
    }
}

use super::*;
use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
use effect_torch_graph::Node;
use effect_torch_runtime::CancellationFlag;
use std::sync::Arc;

fn chain(width: usize, dtype: DType, minimum: f64) -> Vec<Arc<Node>> {
    chain_rows(9, width, dtype, minimum)
}

fn chain_rows(rows: usize, width: usize, dtype: DType, minimum: f64) -> Vec<Arc<Node>> {
    let mut nodes = Vec::new();
    let mut make = |kind| {
        let n = Node::new(kind).unwrap();
        nodes.push(n.clone());
        n
    };
    let source = make(NodeKind::Input {
        slot: 0,
        shape: vec![rows, width],
        dtype,
        device: Device::Cuda(0),
        storage: effect_torch_runtime::StorageMetadata::dense(),
    });
    let maximum = make(NodeKind::Max {
        a: source.clone(),
        dims: vec![1],
        keepdims: true,
    });
    let shifted = make(NodeKind::Sub {
        a: source.clone(),
        b: maximum.clone(),
    });
    let exponential = make(NodeKind::Exp { a: shifted });
    let sum = make(NodeKind::Sum {
        a: exponential,
        dims: vec![1],
        keepdims: true,
    });
    let logarithm = make(NodeKind::Log { a: sum });
    let lse = make(NodeKind::Add {
        a: maximum,
        b: logarithm,
    });
    let normalized = make(NodeKind::Sub { a: source, b: lse });
    let minimum = make(NodeKind::Full {
        shape: vec![rows, width],
        value: minimum,
        dtype,
        device: Device::Cuda(0),
    });
    let clamped = make(NodeKind::Maximum {
        a: normalized.clone(),
        b: minimum,
    });
    let maximum = make(NodeKind::Max {
        a: normalized.clone(),
        dims: vec![1],
        keepdims: true,
    });
    let shifted = make(NodeKind::Sub {
        a: normalized,
        b: maximum,
    });
    let exponential = make(NodeKind::Exp { a: shifted });
    let denominator = make(NodeKind::Sum {
        a: exponential.clone(),
        dims: vec![1],
        keepdims: true,
    });
    let probability = make(NodeKind::Div {
        a: exponential,
        b: denominator,
    });
    let product = make(NodeKind::Mul {
        a: clamped,
        b: probability,
    });
    let sum = make(NodeKind::Sum {
        a: product,
        dims: vec![1],
        keepdims: false,
    });
    make(NodeKind::Neg { a: sum });
    nodes
}

#[test]
fn entropy_selection_requires_exact_f32_pattern_private_intermediates_and_opt_in() {
    for (enabled, width, dtype, minimum, expected) in [
        (true, 4096, DType::F32, -(f32::MAX as f64), true),
        (true, 262144, DType::F32, -(f32::MAX as f64), true),
        (false, 4096, DType::F32, -(f32::MAX as f64), false),
        (true, 4095, DType::F32, -(f32::MAX as f64), false),
        (true, 4096, DType::F16, -(f32::MAX as f64), false),
        (true, 4096, DType::F32, -100., false),
    ] {
        let nodes = chain(width, dtype, minimum);
        for escape in 0..nodes.len() - 1 {
            let mut roots = vec![nodes.last().unwrap().clone()];
            if escape > 0 {
                roots.push(nodes[escape].clone());
            }
            let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
                .prepare()
                .unwrap();
            let mut caps = CudaCapabilities::new(0, 12, 0);
            caps.entropy_recompute = enabled;
            let driver = CompilerDriver::new(&prepared, &caps).unwrap();
            assert_eq!(
                driver
                    .optimization()
                    .regions
                    .iter()
                    .any(|r| matches!(r, NativeRegion::Entropy(_))),
                expected && escape == 0,
                "width{width} dtype{dtype:?} escape{escape}"
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_ENTROPY_RECOMPUTE=1"]
fn entropy_exact_special_values_retained_outputs_and_cancellation() {
    assert_eq!(
        std::env::var("EFFECT_TORCH_CUDA_ENTROPY_RECOMPUTE").unwrap(),
        "1"
    );
    for width in [4096, 4097, 16385, 262144] {
        let root = chain(width, DType::F32, -(f32::MAX as f64)).pop().unwrap();
        let optimized =
            crate::compile_with_options(vec![root.clone()], 0, CompileOptions::default()).unwrap();
        assert!(optimized
            .diagnostics()
            .instructions
            .iter()
            .any(|i| i.kind == "et_entropy_finish" && i.count == 1));
        let reference = crate::compile_with_options(
            vec![root],
            0,
            CompileOptions {
                optimize: false,
                ..Default::default()
            },
        )
        .unwrap();
        let mut retained = Vec::new();
        for seed in [17usize, 313] {
            let data = (0..9 * width)
                .map(|i| {
                    let col = i % width;
                    match i / width {
                        0 => [-0., 0.][col % 2],
                        1 => f32::from_bits(1 + col as u32 % 17) as f64,
                        2 => f64::NEG_INFINITY,
                        3 => {
                            if col == 0 {
                                f64::NAN
                            } else {
                                -1.
                            }
                        }
                        4 => {
                            if col == width - 1 {
                                f64::INFINITY
                            } else {
                                2.
                            }
                        }
                        5 => {
                            if col % 2 == 0 {
                                -1e30
                            } else {
                                1e30
                            }
                        }
                        6 => -(col as f64 % 200.),
                        _ => (((i * 101 + seed) % 65521) as f64 - 32760.) / 1024.,
                    }
                })
                .collect::<Vec<_>>();
            let bindings = [crate::CudaValue::from_host(
                crate::CudaDevice::get(0).unwrap(),
                vec![9, width],
                DType::F32,
                &data,
            )
            .unwrap()];
            let before = bindings[0].read_storage_bytes().unwrap();
            let cancelled = CancellationFlag::new();
            cancelled.cancel();
            assert!(optimized.execute(&bindings, &[], &cancelled).is_err());
            let expected = reference
                .execute(&bindings, &[], &CancellationFlag::new())
                .unwrap()[0]
                .read_storage_bytes()
                .unwrap();
            let actual = optimized
                .execute(&bindings, &[], &CancellationFlag::new())
                .unwrap();
            assert_eq!(
                actual[0].read_storage_bytes().unwrap(),
                expected,
                "width{width} seed{seed}"
            );
            assert_eq!(bindings[0].read_storage_bytes().unwrap(), before);
            retained.push((actual[0].clone(), expected));
        }
        drop(optimized);
        drop(reference);
        for (output, expected) in retained {
            assert_eq!(output.read_storage_bytes().unwrap(), expected);
        }
    }
}

#[test]
fn entropy_selection_preserves_fallback_above_one_cta_per_row_grid_limit() {
    for (rows, expected) in [(65535, true), (65536, false)] {
        let root = chain_rows(rows, 4096, DType::F32, -(f32::MAX as f64))
            .pop()
            .unwrap();
        let prepared = ProgramRequest::from_roots(vec![root], CompileOptions::default())
            .prepare()
            .unwrap();
        let mut caps = CudaCapabilities::new(0, 12, 0);
        caps.entropy_recompute = true;
        let driver = CompilerDriver::new(&prepared, &caps).unwrap();
        assert_eq!(
            driver
                .optimization()
                .regions
                .iter()
                .any(|r| matches!(r, NativeRegion::Entropy(_))),
            expected,
            "rows={rows}"
        );
    }
}

#[test]
fn entropy81_opt_in_reuses_only_private_dense_f32_entropy_pattern_and_geometry() {
    for (enabled, rows, width, dtype, minimum, expected) in [
        (true, 9, 4096, DType::F32, -(f32::MAX as f64), true),
        (true, 256, 262144, DType::F32, -(f32::MAX as f64), true),
        (true, 65535, 4097, DType::F32, -(f32::MAX as f64), true),
        (false, 9, 4096, DType::F32, -(f32::MAX as f64), false),
        (true, 65536, 4096, DType::F32, -(f32::MAX as f64), false),
        (true, 9, 4095, DType::F32, -(f32::MAX as f64), false),
        (true, 9, 4096, DType::F16, -(f32::MAX as f64), false),
        (true, 9, 4096, DType::F32, -100., false),
    ] {
        let nodes = chain_rows(rows, width, dtype, minimum);
        for escape in 0..nodes.len() - 1 {
            let mut roots = vec![nodes.last().unwrap().clone()];
            if escape > 0 {
                roots.push(nodes[escape].clone());
            }
            let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
                .prepare()
                .unwrap();
            let mut caps = CudaCapabilities::new(0, 12, 0);
            caps.entropy_recompute = false;
            caps.entropy_relaxed81 = enabled;
            let driver = CompilerDriver::new(&prepared, &caps).unwrap();
            assert_eq!(
                driver
                    .optimization()
                    .regions
                    .iter()
                    .any(|r| matches!(r, NativeRegion::Entropy(_))),
                expected && escape == 0,
                "enabled={enabled} rows={rows} width={width} dtype={dtype:?} escape={escape}"
            );
            assert!(caps
                .fingerprint()
                .features
                .iter()
                .any(|feature| feature.starts_with("entropy81-experimental-f32-moment-v1-")));
        }
    }
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_RELAXED_ENTROPY81=1; experimental numerical diagnostic"]
fn entropy81_native_descriptor_true_entropy_nonfinite_and_retained_output() {
    assert!(crate::entropy81::enabled());
    let rows = 256;
    let width = 262144;
    let root = chain_rows(rows, width, DType::F32, -(f32::MAX as f64))
        .pop()
        .unwrap();
    let executable = crate::compile(vec![root], 0).unwrap();
    let instructions = &executable.diagnostics().instructions;
    assert!(instructions
        .iter()
        .any(|i| i.kind == "et_entropy_max" && i.count == 1));
    assert!(instructions
        .iter()
        .any(|i| i.kind == crate::entropy81::KERNEL && i.count == 1));
    assert!(!instructions.iter().any(|i| matches!(
        i.kind.as_str(),
        "et_entropy_sum"
            | "et_entropy_normalized_max"
            | "et_entropy_normalized_sum"
            | "et_entropy_finish"
    )));
    let data: Vec<f64> = (0..rows * width)
        .map(|i| {
            let row = i / width;
            let column = i % width;
            let value: f32 = match row {
                0 => f32::NAN,
                1 => f32::NEG_INFINITY,
                2 => {
                    if column == width - 1 {
                        f32::INFINITY
                    } else {
                        2.
                    }
                }
                3 => 10.,
                4 => {
                    if column % 2 == 0 {
                        -1e30
                    } else {
                        1e30
                    }
                }
                _ => (((i * 101 + 313) % 65521) as f32 - 32760.) / 1024.,
            };
            f64::from(value)
        })
        .collect();
    let mut expected = Vec::with_capacity(rows);
    for (row, values) in data.chunks_exact(width).enumerate() {
        if row < 3 {
            expected.push(f64::NAN);
            continue;
        }
        let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let (sum, moment) = values.iter().fold((0., 0.), |(sum, moment), &x| {
            let shifted = x - maximum;
            let exponential = shifted.exp();
            (
                sum + exponential,
                moment
                    + if exponential == 0. {
                        0.
                    } else {
                        exponential * shifted
                    },
            )
        });
        expected.push(sum.ln() - moment / sum);
    }
    let input = crate::CudaValue::from_host(
        crate::CudaDevice::get(0).unwrap(),
        vec![rows, width],
        DType::F32,
        &data,
    )
    .unwrap();
    drop(data);
    let cancelled = CancellationFlag::new();
    cancelled.cancel();
    assert!(executable
        .execute(std::slice::from_ref(&input), &[], &cancelled)
        .is_err());
    let mut retained = Vec::new();
    for _ in 0..2 {
        let outputs = executable
            .execute(std::slice::from_ref(&input), &[], &CancellationFlag::new())
            .unwrap();
        let bytes = outputs[0].read_storage_bytes().unwrap();
        assert_eq!(bytes.len(), rows * 4);
        for (row, (actual, &reference)) in bytes.chunks_exact(4).zip(&expected).enumerate() {
            let actual = f32::from_le_bytes(actual.try_into().unwrap()) as f64;
            if reference.is_nan() {
                assert!(
                    actual.is_nan(),
                    "nonfinite classification row={row} actual={actual}"
                );
            } else {
                assert!(
                    actual.is_finite() && (actual - reference).abs() <= 2e-5,
                    "true entropy row={row} actual={actual} reference={reference}"
                );
            }
        }
        retained.push((outputs[0].clone(), bytes));
    }
    drop(executable);
    drop(input);
    for (output, bytes) in retained {
        assert_eq!(output.read_storage_bytes().unwrap(), bytes);
    }
}

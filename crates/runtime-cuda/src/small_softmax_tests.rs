use super::*;
use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
use effect_torch_graph::Node;
use effect_torch_runtime::{CancellationFlag, StorageMetadata};
use std::sync::Arc;

fn chain(width: usize, dtype: DType, dim: usize) -> Vec<Arc<Node>> {
    let make = |kind| Node::new(kind).unwrap();
    let input = make(NodeKind::Input {
        slot: 0,
        shape: vec![9, width],
        dtype,
        device: Device::Cuda(0),
        storage: StorageMetadata::dense(),
    });
    let maximum = make(NodeKind::Max {
        a: input.clone(),
        dims: vec![dim],
        keepdims: true,
    });
    let shifted = make(NodeKind::Sub {
        a: input.clone(),
        b: maximum.clone(),
    });
    let exponential = make(NodeKind::Exp { a: shifted.clone() });
    let sum = make(NodeKind::Sum {
        a: exponential.clone(),
        dims: vec![dim],
        keepdims: true,
    });
    let output = make(NodeKind::Div {
        a: exponential.clone(),
        b: sum.clone(),
    });
    vec![output, input, maximum, shifted, exponential, sum]
}

#[test]
fn small_softmax_selection_requires_exact_f32_width_axis_and_private_intermediates() {
    for (enabled, width, dtype, dim, escape, expected) in [
        (true, 128, DType::F32, 1, 0, true),
        (false, 128, DType::F32, 1, 0, false),
        (true, 127, DType::F32, 1, 0, false),
        (true, 129, DType::F32, 1, 0, false),
        (true, 128, DType::BF16, 1, 0, false),
        (true, 128, DType::F64, 1, 0, false),
        (true, 128, DType::F32, 0, 0, false),
        (true, 128, DType::F32, 1, 1, true),
        (true, 128, DType::F32, 1, 2, false),
        (true, 128, DType::F32, 1, 3, false),
        (true, 128, DType::F32, 1, 4, false),
        (true, 128, DType::F32, 1, 5, false),
    ] {
        let nodes = chain(width, dtype, dim);
        let mut roots = vec![nodes[0].clone()];
        if escape != 0 {
            roots.push(nodes[escape].clone());
        }
        let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
            .prepare()
            .unwrap();
        let mut capabilities = CudaCapabilities::new(0, 12, 0);
        capabilities.small_softmax = enabled;
        let driver = CompilerDriver::new(&prepared, &capabilities).unwrap();
        let selected = driver
            .optimization()
            .regions
            .iter()
            .any(|r| matches!(r, NativeRegion::SmallSoftmax(_)));
        assert_eq!(
            selected, expected,
            "enabled={enabled} width={width} dtype={dtype:?} dim={dim} escape={escape}"
        );
    }
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_SMALL_SOFTMAX=1"]
fn small_softmax_exact_bits_changed_inputs_retention_and_cancellation() {
    assert_eq!(
        std::env::var("EFFECT_TORCH_CUDA_SMALL_SOFTMAX").unwrap(),
        "1"
    );
    let roots = vec![chain(128, DType::F32, 1)[0].clone()];
    let optimized =
        crate::compile_with_options(roots.clone(), 0, CompileOptions::default()).unwrap();
    assert!(optimized
        .diagnostics()
        .instructions
        .iter()
        .any(|i| i.kind == "et_small_softmax_f32" && i.count == 1));
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
    for seed in [17usize, 313] {
        let data = (0..9 * 128)
            .map(|i| {
                let col = i % 128;
                match i / 128 {
                    0 => {
                        if col % 2 == 0 {
                            -0.0
                        } else {
                            0.0
                        }
                    }
                    1 => f32::from_bits((1 + col) as u32) as f64,
                    2 => f64::NEG_INFINITY,
                    3 => {
                        if col == seed % 128 {
                            f64::NAN
                        } else {
                            -1.0
                        }
                    }
                    4 => {
                        if col == seed % 128 {
                            f64::INFINITY
                        } else {
                            2.0
                        }
                    }
                    5 => {
                        if col % 2 == 0 {
                            1e30
                        } else {
                            -1e30
                        }
                    }
                    6 => -(col as f64 % 100.),
                    _ => (((i * 101 + seed) % 65521) as f64 - 32760.) / 1024.,
                }
            })
            .collect::<Vec<_>>();
        let bindings = [crate::CudaValue::from_host(
            crate::CudaDevice::get(0).unwrap(),
            vec![9, 128],
            DType::F32,
            &data,
        )
        .unwrap()];
        let borrowed = bindings[0].read_storage_bytes().unwrap();
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
            "seed={seed}"
        );
        assert_eq!(bindings[0].read_storage_bytes().unwrap(), borrowed);
        retained.push((actual[0].clone(), expected));
    }
    drop(optimized);
    drop(reference);
    for (value, expected) in retained {
        assert_eq!(value.read_storage_bytes().unwrap(), expected);
    }
    let unsupported = crate::compile_with_options(
        vec![chain(129, DType::F32, 1)[0].clone()],
        0,
        CompileOptions::default(),
    )
    .unwrap();
    assert!(!unsupported
        .diagnostics()
        .instructions
        .iter()
        .any(|i| i.kind == "et_small_softmax_f32"));
}

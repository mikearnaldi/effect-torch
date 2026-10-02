use super::*;
use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
use effect_torch_graph::Node;
use effect_torch_runtime::CancellationFlag;
use std::sync::Arc;

pub(super) fn chain(
    rows: usize,
    width: usize,
    shared_indexes: bool,
    reverse_mul: bool,
    approximate: bool,
) -> Vec<Arc<Node>> {
    let input = |slot, shape, dtype| {
        Node::new(NodeKind::Input {
            slot,
            shape,
            dtype,
            device: Device::Cuda(0),
            storage: effect_torch_runtime::StorageMetadata::dense(),
        })
        .unwrap()
    };
    let x = input(0, vec![rows, width], DType::BF16);
    let indexes = input(1, vec![rows], DType::U32);
    // Graph constants avoid giant host bank allocations in the hardware fixture.
    let bank = |shape, value| {
        Node::new(NodeKind::Full {
            shape,
            value,
            dtype: DType::BF16,
            device: Device::Cuda(0),
        })
        .unwrap()
    };
    let first = Node::new(NodeKind::GroupedExpertLinearRows {
        x,
        weight: Node::new(NodeKind::Concat {
            a: bank(vec![128, 704, width], 0.00390625),
            b: bank(vec![128, 704, width], -0.0078125),
            dim: 1,
        })
        .unwrap(),
        indexes: indexes.clone(),
    })
    .unwrap();
    let gate = Node::new(NodeKind::Slice {
        a: first.clone(),
        ranges: vec![(0, rows, 1), (0, 704, 1)],
    })
    .unwrap();
    let up = Node::new(NodeKind::Slice {
        a: first.clone(),
        ranges: vec![(0, rows, 1), (704, 1408, 1)],
    })
    .unwrap();
    let activated = Node::new(NodeKind::Gelu {
        a: gate.clone(),
        approximate,
    })
    .unwrap();
    let product = Node::new(NodeKind::Mul {
        a: if reverse_mul {
            up.clone()
        } else {
            activated.clone()
        },
        b: if reverse_mul {
            activated.clone()
        } else {
            up.clone()
        },
    })
    .unwrap();
    let output = Node::new(NodeKind::GroupedExpertLinearRows {
        x: product.clone(),
        weight: bank(vec![128, 2816, 704], 0.0078125),
        indexes: if shared_indexes {
            indexes
        } else {
            input(2, vec![rows], DType::U32)
        },
    })
    .unwrap();
    vec![first, gate, up, activated, product, output]
}

#[test]
fn grouped_retained_selection_private_roots_geometry_operand_order_and_rounding() {
    for (enabled, width, same_indexes, reverse, approximate, expected) in [
        (true, 2816, true, false, true, true),
        (false, 2816, true, false, true, false),
        (true, 2815, true, false, true, false),
        (true, 2816, false, false, true, false),
        (true, 2816, true, true, true, false),
        (true, 2816, true, false, false, false),
    ] {
        let nodes = chain(9, width, same_indexes, reverse, approximate);
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
            caps.grouped_retained_layout = enabled;
            let driver = CompilerDriver::new(&prepared, &caps).unwrap();
            let selected = driver
                .optimization()
                .regions
                .iter()
                .enumerate()
                .find(|(_, region)| matches!(region, NativeRegion::GroupedExpertGated(_)));
            assert_eq!(
                selected.is_some(),
                expected && escape == 10,
                "enabled={enabled} width={width} same={same_indexes} reverse={reverse} approximate={approximate} escape={escape}"
            );
            if let Some((position, region)) = selected {
                let unit = driver.legalization().units().iter().find(|unit| matches!(unit.unit(), effect_torch_compiler::LoweringUnit::Region(id) if id.index() == position)).unwrap();
                let expressions =
                    effect_torch_compiler::legalize_region_expressions(region, unit.disposition())
                        .unwrap();
                let effect_torch_compiler::KernelExpr::RoundTo(product, DType::BF16) =
                    &expressions[0]
                else {
                    panic!("missing product BF16 boundary")
                };
                let effect_torch_compiler::KernelExpr::Mul(gate, _) = product.as_ref() else {
                    panic!("missing product")
                };
                assert!(
                    matches!(
                        gate.as_ref(),
                        effect_torch_compiler::KernelExpr::RoundTo(_, DType::BF16)
                    ),
                    "missing GELU BF16 boundary"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_GROUPED_RETAINED_LAYOUT=1"]
fn grouped_retained_exact_rooted_fallback_changing_routes_cancel_and_outputs() {
    assert_eq!(
        std::env::var("EFFECT_TORCH_CUDA_GROUPED_RETAINED_LAYOUT").as_deref(),
        Ok("1")
    );
    let rows = 17;
    let nodes = chain(rows, 2816, true, false, true);
    let optimized = Arc::new(crate::compile(vec![nodes[5].clone()], 0).unwrap());
    // Exposing each private result forces the original route-order path while
    // keeping every other optimization and the exact GEMM policy enabled.
    let reference = crate::compile(
        vec![
            nodes[5].clone(),
            nodes[0].clone(),
            nodes[1].clone(),
            nodes[2].clone(),
            nodes[3].clone(),
            nodes[4].clone(),
        ],
        0,
    )
    .unwrap();
    assert!(optimized
        .diagnostics()
        .instructions
        .iter()
        .any(|i| i.kind == "group_sorted_activation"));
    assert!(!reference
        .diagnostics()
        .instructions
        .iter()
        .any(|i| i.kind == "group_sorted_activation"));
    let device = crate::CudaDevice::get(0).unwrap();
    let host = |shape, dtype, data: &[f64]| {
        crate::CudaValue::from_host(device.clone(), shape, dtype, data).unwrap()
    };
    let mut retained = Vec::new();
    let mut concurrent_inputs = Vec::new();
    for mode in 0..7 {
        let routes = (0..rows)
            .map(|row| match mode {
                0 => row as f64,
                1 => (row / 2) as f64,
                2 => 127.,
                _ => ((row * 47) % 128) as f64,
            })
            .collect::<Vec<_>>();
        let input = (0..rows * 2816)
            .map(|i| match mode {
                4 => {
                    if i / 2816 % 2 == 0 {
                        -0.
                    } else {
                        0.
                    }
                }
                5 => f32::from_bits(1 << 16) as f64 * ((i % 3) as f64 - 1.),
                6 => match i / 2816 % 5 {
                    0 => f64::INFINITY,
                    1 => f64::NEG_INFINITY,
                    2 => f64::NAN,
                    _ => ((i * 31 % 103) as f64 - 51.) / 32.,
                },
                _ => ((i * 31 % 103) as f64 - 51.) / 32.,
            })
            .collect::<Vec<_>>();
        let bindings = [
            host(vec![rows, 2816], DType::BF16, &input),
            host(vec![rows], DType::U32, &routes),
        ];
        let expected = reference
            .execute(&bindings, &[], &CancellationFlag::new())
            .unwrap()[0]
            .read_storage_bytes()
            .unwrap();
        let cancelled = CancellationFlag::new();
        cancelled.cancel();
        assert!(optimized.execute(&bindings, &[], &cancelled).is_err());
        let interrupted = CancellationFlag::new();
        let result =
            optimized.execute_with_gemm_hook(&bindings, &interrupted, &|| interrupted.cancel());
        assert!(matches!(result, Err(ref error) if error == "operation aborted"));
        let actual = optimized
            .execute(&bindings, &[], &CancellationFlag::new())
            .unwrap()
            .remove(0);
        assert_eq!(
            actual.read_storage_bytes().unwrap(),
            expected,
            "mode={mode}"
        );
        if mode < 2 {
            concurrent_inputs.push((bindings.clone(), expected.clone()));
        }
        retained.push((actual, expected));
    }
    let invalid = [
        host(vec![rows, 2816], DType::BF16, &vec![0.; rows * 2816]),
        host(vec![rows], DType::U32, &vec![128.; rows]),
    ];
    assert!(optimized
        .execute(&invalid, &[], &CancellationFlag::new())
        .is_err());
    let workers = concurrent_inputs
        .into_iter()
        .map(|(bindings, expected)| {
            let executable = optimized.clone();
            std::thread::spawn(move || {
                let actual = executable
                    .execute(&bindings, &[], &CancellationFlag::new())
                    .unwrap();
                assert_eq!(actual[0].read_storage_bytes().unwrap(), expected);
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        worker.join().unwrap();
    }
    drop(optimized);
    drop(reference);
    for (value, bytes) in retained {
        assert_eq!(value.read_storage_bytes().unwrap(), bytes);
    }
}

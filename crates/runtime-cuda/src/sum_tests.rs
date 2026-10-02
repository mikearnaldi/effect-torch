//! F32 Sum accuracy, semantic narrowing, materialized views and routing capture.
use crate::{CudaDevice, CudaValue};
use effect_torch_graph::{Device, Node, NodeKind};
use effect_torch_runtime::{CancellationFlag, DType, StorageMetadata};
use std::{path::PathBuf, sync::Arc};

fn node(kind: NodeKind) -> Arc<Node> {
    Node::new(kind).unwrap()
}
fn input(slot: u32, shape: &[usize], dtype: DType) -> Arc<Node> {
    node(NodeKind::Input {
        slot,
        shape: shape.to_vec(),
        dtype,
        device: Device::Cuda(0),
        storage: StorageMetadata::dense(),
    })
}
fn host(shape: &[usize], dtype: DType, values: &[f64]) -> CudaValue {
    CudaValue::from_host(CudaDevice::get(0).unwrap(), shape.to_vec(), dtype, values).unwrap()
}
fn sum(a: Arc<Node>, dims: &[usize], keepdims: bool) -> Arc<Node> {
    node(NodeKind::Sum {
        a,
        dims: dims.to_vec(),
        keepdims,
    })
}

#[test]
fn small_max_dispatch_requires_opt_in_and_short_trailing_f32_max() {
    use crate::executable::{small_max_eligible, CudaKernelArgs};
    let mut args = CudaKernelArgs {
        elements: 256,
        operation: 2,
        compute_dtype: 1,
        ..Default::default()
    };
    args.integers[1] = 128;
    args.integers[2] = 1;
    assert!(small_max_eligible("et_reduce_f32", &args, true));
    assert!(!small_max_eligible("et_reduce_f32", &args, false));
    assert!(!small_max_eligible("et_reduce_f64", &args, true));
    for dtype in [0, 2, 3, 4, 5, 6] {
        args.compute_dtype = dtype;
        assert!(!small_max_eligible("et_reduce_f32", &args, true));
    }
    args.compute_dtype = 1;
    for width in [0, 4096, 262144] {
        args.integers[1] = width;
        assert!(!small_max_eligible("et_reduce_f32", &args, true));
    }
    args.integers[1] = 128;
    for operation in [0, 1, 3, 4] {
        args.operation = operation;
        assert!(!small_max_eligible("et_reduce_f32", &args, true));
    }
    args.operation = 2;
    args.integers[2] = 0;
    assert!(!small_max_eligible("et_reduce_f32", &args, true));
    args.integers[2] = 1;
    args.elements = 0;
    assert!(!small_max_eligible("et_reduce_f32", &args, true));
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_SMALL_MAX=1"]
fn small_max_matches_generic_bits_for_finite_nan_zero_and_infinity() {
    assert_eq!(std::env::var("EFFECT_TORCH_CUDA_SMALL_MAX").unwrap(), "1");
    // A transposed host layout forces the generic nontrailing reduction while
    // keeping each row's value order identical. No process-global flag changes.
    for width in [1, 7, 31, 32, 33, 128, 257, 4095] {
        let rows = if width == 128 { 256 } else { 17 };
        let values: Vec<f64> = (0..rows * width)
            .map(|index| {
                let row = index / width;
                let column = index % width;
                match row {
                    0 => f64::NAN,
                    1 => f64::NEG_INFINITY,
                    2 => f64::INFINITY,
                    3 => -0.0,
                    4 => 0.0,
                    5 => {
                        if column % 2 == 0 {
                            -0.0
                        } else {
                            0.0
                        }
                    }
                    6 => {
                        if column % 2 == 0 {
                            0.0
                        } else {
                            -0.0
                        }
                    }
                    7 => [f64::NAN, -0.0, f64::NEG_INFINITY][column % 3],
                    8 => [f64::NAN, f64::INFINITY, -1.0][column % 3],
                    9 => [f64::NAN, -3.0, -1.0][column % 3],
                    10 => f32::from_bits(1 + column as u32 % 17) as f64,
                    11 => -(f32::from_bits(1 + column as u32 % 17) as f64),
                    12 => [f32::MIN_POSITIVE as f64, -0.0, f64::NAN][column % 3],
                    13 => [f32::MAX as f64, -(f32::MAX as f64)][column % 2],
                    _ => ((index * 101 % 4093) as f64 - 2046.0) / 128.0,
                }
            })
            .collect();
        let transposed: Vec<_> = (0..width * rows)
            .map(|index| values[(index % rows) * width + index / rows])
            .collect();
        let mut outputs = Vec::new();
        for (shape, dims, data) in [
            ([rows, width], vec![1], &values),
            ([width, rows], vec![0], &transposed),
        ] {
            let root = node(NodeKind::Max {
                a: input(0, &shape, DType::F32),
                dims,
                keepdims: false,
            });
            let executable = crate::compile(vec![root], 0).unwrap();
            let output = executable
                .execute(
                    &[host(&shape, DType::F32, data)],
                    &[],
                    &CancellationFlag::new(),
                )
                .unwrap();
            outputs.push(output[0].read_storage_bytes().unwrap());
        }
        assert_eq!(outputs[0], outputs[1], "width={width}");
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn sum_independent_rational_rounding_witness() {
    // Exact dyadic sum = 355686261 / 1073741824. The sequential kernel rounded
    // to 0x3ea99abb; nearest F32 is 0x3ea99abc. No model/scales needed.
    let values = [
        1040849724, 1036246331, 1016280329, 1016130760, 1015983510, 1015910743, 1015007911,
        1014115770,
    ]
    .map(|bits| f32::from_bits(bits) as f64);
    let exact = 355686261.0 / 1073741824.0;
    assert_eq!(values.iter().sum::<f64>(), exact);
    let old = values.iter().fold(0_f32, |total, x| total + *x as f32);
    assert_ne!(old, exact as f32);
    let exe = crate::compile(vec![sum(input(0, &[8], DType::F32), &[0], false)], 0).unwrap();
    let out = exe
        .execute(
            &[host(&[8], DType::F32, &values)],
            &[],
            &CancellationFlag::new(),
        )
        .unwrap();
    assert_eq!(
        out[0].read_storage_bytes().unwrap(),
        (exact as f32).to_le_bytes()
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn sum_wide_preserves_small_positive_contributions() {
    // Exactly 32 + 2^-18, one F32 ULP above 32. A single warp gives lane
    // zero 1 followed by 64 additions of 2^-24, all lost to rounding.
    // A block assigns those small terms their own partial before the tree.
    let width = 262144;
    let mut values = vec![0.0; width];
    values[..32].fill(1.0);
    for i in (32..width).step_by(4096) {
        values[i] = 2_f64.powi(-24);
    }
    let exact = 32.0 + 2_f64.powi(-18);
    assert_eq!(values.iter().sum::<f64>(), exact);
    let exe = crate::compile(vec![sum(input(0, &[1, width], DType::F32), &[1], true)], 0).unwrap();
    let output = exe
        .execute(
            &[host(&[1, width], DType::F32, &values)],
            &[],
            &CancellationFlag::new(),
        )
        .unwrap();
    assert_eq!(
        output[0].read_storage_bytes().unwrap(),
        (exact as f32).to_le_bytes()
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn sum_widths_tails_axes_views_and_opmath() {
    for dtype in [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::F64,
        DType::I64,
        DType::U32,
        DType::U8,
    ] {
        for (rows, width) in [
            (0, 7),
            (3, 0),
            (1, 1),
            (3, 7),
            (3, 8),
            (3, 9),
            (3, 31),
            (3, 32),
            (3, 33),
            (3, 127),
            (3, 128),
            (3, 129),
            (3, 255),
            (3, 256),
            (3, 257),
            (3, 511),
            (3, 512),
            (3, 513),
            (3, 1023),
            (3, 1024),
            (3, 1025),
            (3, 4095),
            (3, 4096),
            (3, 4097),
            (3, 8191),
            (3, 8192),
            (1, 65537),
        ] {
            // Integer dyadics give an exact independent F64 sum for every dtype.
            // Permute forces materialization, and reducing axis 0 is strided in
            // that materialized [width,rows] input.
            let values = (0..rows * width)
                .map(|i| (i % 5 == 0) as u8 as f64)
                .collect::<Vec<_>>();
            let x = input(0, &[rows, width], dtype);
            let view = node(NodeKind::Permute {
                a: x.clone(),
                dims: vec![1, 0],
            });
            let exe = crate::compile(
                vec![
                    sum(x.clone(), &[1], true),
                    sum(view, &[0], false),
                    sum(x, &[0, 1], false),
                ],
                0,
            )
            .unwrap();
            let bindings = [host(&[rows, width], dtype, &values)];
            let results = exe
                .execute(&bindings, &[], &CancellationFlag::new())
                .unwrap();
            let narrowed = |x: f64| {
                if dtype == DType::U8 {
                    (x as u64 % 256) as f64
                } else {
                    x
                }
            };
            let expected = (0..rows)
                .map(|r| {
                    narrowed(
                        values[r * width..(r + 1) * width]
                            .iter()
                            .fold(0.0_f64, |a, b| a + b),
                    )
                })
                .collect::<Vec<_>>();
            let expected_bytes = crate::value::dense_bytes_from_host(&expected, dtype);
            assert_eq!(
                results[0].read_storage_bytes().unwrap(),
                expected_bytes,
                "{dtype:?} {rows} {width}"
            );
            assert_eq!(results[1].read_storage_bytes().unwrap(), expected_bytes);
            assert_eq!(
                results[2].read_storage_bytes().unwrap(),
                crate::value::dense_bytes_from_host(
                    &[narrowed(values.iter().fold(0.0_f64, |a, b| a + b))],
                    dtype
                )
            );
        }
    }
    // Each lane accumulates 256 ones before its small tails. Narrow partials
    // would discard those tails in either half dtype; one final cast keeps them.
    for dtype in [DType::BF16, DType::F16] {
        let values = [vec![1.; 8192], vec![0.015625; 8192]].concat();
        let exe = crate::compile(vec![sum(input(0, &[16384], dtype), &[0], false)], 0).unwrap();
        assert_eq!(
            exe.execute(
                &[host(&[16384], dtype, &values)],
                &[],
                &CancellationFlag::new()
            )
            .unwrap()[0]
                .readback()
                .unwrap(),
            vec![8320.]
        );
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn sum_nondyadic_accuracy_special_values_and_ownership() {
    for width in [3, 7, 33, 129, 513, 1025, 4097, 65537] {
        let values = (0..width)
            .map(|i| (((i * 157 % 997) as f32 - 480.) / 997.) as f64)
            .collect::<Vec<_>>();
        let exact = values.iter().sum::<f64>();
        let absolute = values.iter().map(|x| x.abs()).sum::<f64>();
        let exe =
            crate::compile(vec![sum(input(0, &[width], DType::F32), &[0], false)], 0).unwrap();
        let result = exe
            .execute(
                &[host(&[width], DType::F32, &values)],
                &[],
                &CancellationFlag::new(),
            )
            .unwrap()[0]
            .readback()
            .unwrap()[0];
        // Standard sequential F32 summation bound, independent of the tree.
        let nu = width as f64 * 2_f64.powi(-24);
        assert!((result - exact).abs() <= nu / (1. - nu) * absolute);
    }
    let x = input(0, &[1, 3], DType::F32);
    let root = sum(
        node(NodeKind::BroadcastTo {
            a: x,
            shape: vec![17, 3],
        }),
        &[1],
        false,
    );
    let exe = Arc::new(crate::compile(vec![root], 0).unwrap());
    let mut retained = Vec::new();
    for values in [
        [0., -0., 0.],
        [f64::INFINITY, 1., 2.],
        [f64::NEG_INFINITY, 1., 2.],
        [f64::NAN, 1., 2.],
        [f64::INFINITY, f64::NEG_INFINITY, 1.],
        [1e-40, -1e-40, 1e-40],
    ] {
        let binding = host(&[1, 3], DType::F32, &values);
        let expected = values.into_iter().fold(0_f32, |s, x| s + x as f32);
        let cancelled = CancellationFlag::new();
        cancelled.cancel();
        assert!(exe.execute(&[binding.clone()], &[], &cancelled).is_err());
        let out = exe
            .execute(&[binding], &[], &CancellationFlag::new())
            .unwrap()
            .remove(0);
        for value in out.readback().unwrap() {
            assert!(
                (value.is_nan() && expected.is_nan())
                    || (value as f32).to_bits() == expected.to_bits()
            );
        }
        retained.push(out);
    }
    let threads = (0..4)
        .map(|_| {
            let exe = exe.clone();
            std::thread::spawn(move || {
                exe.execute(
                    &[host(&[1, 3], DType::F32, &[1., 2., 3.])],
                    &[],
                    &CancellationFlag::new(),
                )
                .unwrap()[0]
                    .readback()
                    .unwrap()
            })
        })
        .collect::<Vec<_>>();
    for thread in threads {
        assert_eq!(thread.join().unwrap(), vec![6.; 17]);
    }
    drop(exe);
    assert_eq!(retained[0].readback().unwrap(), vec![0.; 17]);
    // More rows than the bounded 65535-block launch: exercise its grid stride.
    let rows = 65535 * 8 + 1;
    let exe = crate::compile(vec![sum(input(0, &[rows, 1], DType::F32), &[1], false)], 0).unwrap();
    assert_eq!(
        exe.execute(
            &[host(&[rows, 1], DType::F32, &vec![2.; rows])],
            &[],
            &CancellationFlag::new()
        )
        .unwrap()[0]
            .readback()
            .unwrap(),
        vec![2.; rows]
    );
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_SUM_CAPTURE pointing to bounded replay 46"]
fn sum_captured_compiled_softmax_routing_chain() {
    let path = PathBuf::from(
        std::env::var("EFFECT_TORCH_SUM_CAPTURE").expect("bounded replay 46 fixture directory"),
    );
    let read = |name: &str| std::fs::read(path.join(name)).unwrap();
    let floats = |name: &str| {
        read(name)
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64)
            .collect::<Vec<_>>()
    };
    let legacy = std::env::var_os("EFFECT_TORCH_SUM_LEGACY_CONTROL").is_some();
    let scores = node(NodeKind::Cast {
        a: input(0, &[278, 128], DType::BF16),
        dtype: DType::F32,
    });
    let maximum = node(NodeKind::Max {
        a: scores.clone(),
        dims: vec![1],
        keepdims: true,
    });
    let exp = node(NodeKind::Exp {
        a: node(NodeKind::Sub {
            a: scores,
            b: maximum.clone(),
        }),
    });
    let denominator = sum(exp.clone(), &[1], true);
    let probabilities = node(NodeKind::Div {
        a: exp.clone(),
        b: denominator.clone(),
    });
    let indices = node(NodeKind::TopKIndices {
        a: probabilities.clone(),
        k: 8,
    });
    let selected = node(NodeKind::Gather {
        a: probabilities.clone(),
        indexes: indices.clone(),
        dim: 1,
    });
    let total = sum(selected.clone(), &[1], true);
    let normalized = node(NodeKind::Div {
        a: selected.clone(),
        b: total.clone(),
    });
    let scales = node(NodeKind::Cast {
        a: input(1, &[278, 8], DType::BF16),
        dtype: DType::F32,
    });
    let weights = node(NodeKind::Mul {
        a: normalized.clone(),
        b: scales,
    });
    let exe = crate::compile(
        vec![
            maximum,
            exp,
            denominator,
            probabilities,
            indices,
            selected,
            total,
            normalized,
            weights,
        ],
        0,
    )
    .unwrap();
    let out = exe
        .execute(
            &[
                host(&[278, 128], DType::BF16, &floats("old-scores.f32")),
                host(&[278, 8], DType::BF16, &floats("selectedScale.f32")),
            ],
            &[],
            &CancellationFlag::new(),
        )
        .unwrap();
    let names = if legacy {
        [
            "old-maximum.f32",
            "old-exp.f32",
            "old-denominator.f32",
            "old-probabilities.f32",
            "native-indices.u32",
            "old-selected.f32",
            "sum0-renorm0-total.f32",
            "sum0-renorm0-normalized.f32",
            "sum0-renorm0-weights.f32",
        ]
    } else {
        [
            "old-maximum.f32",
            "old-exp.f32",
            "warp-denominator.f32",
            "torch-softmax.f32",
            "native-indices.u32",
            "warp-selected.f32",
            "sum1-renorm1-total.f32",
            "sum1-renorm1-normalized.f32",
            "sum1-renorm1-weights.f32",
        ]
    };
    for (value, name) in out.iter().zip(names) {
        assert_eq!(
            value.read_storage_bytes().unwrap(),
            read(name),
            "stage {name}"
        );
    }
    eprintln!(
        "compiled routing chain exact, legacy={legacy}, probabilities=35584 weights=2224 all 278 rows and native indices preserved"
    );
}

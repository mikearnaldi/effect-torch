use super::*;
use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
use effect_torch_graph::Node;
use effect_torch_runtime::{CancellationFlag, StorageMetadata};
use std::sync::Arc;

fn chain(
    rows: usize,
    routes: usize,
    dtype: DType,
    variant: u8,
) -> (Vec<Arc<Node>>, Vec<Arc<Node>>) {
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
    let experts = if variant == 5 {
        Node::new(NodeKind::Permute {
            a: input(0, vec![routes, rows]),
            dims: vec![1, 0],
        })
        .unwrap()
    } else {
        input(0, vec![rows, routes])
    };
    let positions = if variant == 5 {
        Node::new(NodeKind::Slice {
            a: input(1, vec![routes + 1]),
            ranges: vec![(1, routes + 1, 1)],
        })
        .unwrap()
    } else {
        input(
            1,
            if variant == 4 {
                vec![1, routes]
            } else {
                vec![routes]
            },
        )
    };
    let reshape = |a, shape| Node::new(NodeKind::Reshape { a, shape }).unwrap();
    let right = reshape(experts.clone(), vec![rows, 1, routes]);
    let left = reshape(experts.clone(), vec![rows, routes, 1]);
    let column = reshape(positions.clone(), vec![1, routes]);
    let row = reshape(
        if variant == 2 {
            input(2, vec![routes])
        } else {
            positions.clone()
        },
        vec![routes, 1],
    );
    let smaller = Node::new(NodeKind::Lt {
        a: if variant == 1 {
            left.clone()
        } else {
            right.clone()
        },
        b: if variant == 1 {
            right.clone()
        } else {
            left.clone()
        },
    })
    .unwrap();
    let same = Node::new(NodeKind::Eq {
        a: right.clone(),
        b: left.clone(),
    })
    .unwrap();
    let earlier = Node::new(NodeKind::Lt {
        a: column.clone(),
        b: row.clone(),
    })
    .unwrap();
    let tie = Node::new(NodeKind::Mul {
        a: same.clone(),
        b: earlier.clone(),
    })
    .unwrap();
    let precedes = Node::new(NodeKind::Add {
        a: smaller.clone(),
        b: tie.clone(),
    })
    .unwrap();
    let floating = Node::new(NodeKind::Cast {
        a: precedes.clone(),
        dtype: DType::F32,
    })
    .unwrap();
    let counts = Node::new(NodeKind::Sum {
        a: floating.clone(),
        dims: vec![if variant == 3 { 1 } else { 2 }],
        keepdims: false,
    })
    .unwrap();
    let rank = Node::new(NodeKind::Cast {
        a: counts.clone(),
        dtype: DType::U32,
    })
    .unwrap();
    (
        vec![
            right, left, column, row, smaller, same, earlier, tie, precedes, floating, counts, rank,
        ],
        vec![experts, positions],
    )
}

#[test]
fn expert_route_rank_private_shapes_identity_dtype_policy_and_escape_guards() {
    for (rows, routes, dtype, variant, enabled, expected) in [
        (3, 8, DType::U32, 0, true, true),
        (1, 1, DType::U32, 0, true, true),
        (3, 32, DType::U32, 0, true, true),
        (3, 8, DType::U32, 5, true, true),
        (3, 33, DType::U32, 0, true, false),
        (3, 8, DType::F32, 0, true, false),
        (3, 8, DType::U32, 1, true, false),
        (3, 8, DType::U32, 2, true, false),
        (3, 8, DType::U32, 3, true, false),
        (3, 8, DType::U32, 4, true, false),
        (3, 8, DType::U32, 0, false, false),
    ] {
        let (nodes, inputs) = chain(rows, routes, dtype, variant);
        for escape in 0..=23 {
            let mut roots = vec![nodes[11].clone()];
            if escape < 11 {
                roots.push(nodes[escape].clone());
            } else if escape < 22 {
                roots.push(
                    Node::new(NodeKind::Cast {
                        a: nodes[escape - 11].clone(),
                        dtype: DType::F64,
                    })
                    .unwrap(),
                );
            } else if escape == 22 {
                roots.extend(inputs.iter().cloned());
            }
            let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
                .prepare()
                .unwrap();
            let mut caps = CudaCapabilities::new(0, 12, 0);
            caps.expert_route_rank = enabled;
            let driver = CompilerDriver::new(&prepared, &caps).unwrap();
            let selected = driver
                .optimization()
                .regions
                .iter()
                .any(|r| matches!(r, NativeRegion::ExpertRouteRank(_)));
            assert_eq!(
                selected,
                expected && escape >= 22,
                "rows={rows} routes={routes} dtype={dtype:?} variant={variant} enabled={enabled} escape={escape}"
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_EXPERT_ROUTE_RANK=1"]
fn expert_route_rank_exact_full_u32_positions_duplicates_cancel_concurrent_and_retained() {
    assert_eq!(
        std::env::var("EFFECT_TORCH_CUDA_EXPERT_ROUTE_RANK").as_deref(),
        Ok("1")
    );
    let device = crate::CudaDevice::get(0).unwrap();
    for (rows, routes, view) in [
        (1, 1, false),
        (3, 2, false),
        (33, 8, false),
        (33, 31, false),
        (33, 32, false),
        (256, 8, false),
        (33, 8, true),
    ] {
        let (nodes, _) = chain(rows, routes, DType::U32, if view { 5 } else { 0 });
        let executable = Arc::new(crate::compile(vec![nodes[11].clone()], 0).unwrap());
        let reference = crate::compile(
            std::iter::once(nodes[11].clone())
                .chain(nodes[..11].iter().cloned())
                .collect(),
            0,
        )
        .unwrap();
        assert!(executable
            .diagnostics()
            .instructions
            .iter()
            .any(|i| i.kind == "et_expert_route_rank"));
        assert!(!reference
            .diagnostics()
            .instructions
            .iter()
            .any(|i| i.kind == "et_expert_route_rank"));
        let mut retained = Vec::new();
        for mode in 0..8 {
            let positions = (0..routes)
                .map(|r| match mode {
                    4 => 0,
                    5 => (routes - r) as u32,
                    6 => u32::MAX - r as u32,
                    7 => (r as u32).wrapping_mul(2654435761),
                    _ => r as u32,
                })
                .collect::<Vec<_>>();
            let experts = (0..rows * routes)
                .map(|i| match mode {
                    1 | 4 => u32::MAX,
                    2 | 5 => (i % 3) as u32,
                    3 => u32::MAX - (i % 5) as u32,
                    _ => (i as u32).wrapping_mul(2654435761),
                })
                .collect::<Vec<_>>();
            let expected = (0..rows * routes)
                .map(|i| {
                    let row = i / routes;
                    let r = i % routes;
                    (0..routes)
                        .filter(|&c| {
                            experts[row * routes + c] < experts[i]
                                || (experts[row * routes + c] == experts[i]
                                    && positions[c] < positions[r])
                        })
                        .count() as u32
                })
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>();
            let value = |shape, values: &[u32]| {
                crate::CudaValue::from_dense_bytes(
                    device.clone(),
                    shape,
                    DType::U32,
                    &values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap()
            };
            let bindings = if view {
                let transposed = (0..routes)
                    .flat_map(|r| (0..rows).map(move |n| (n, r)))
                    .map(|(n, r)| experts[n * routes + r])
                    .collect::<Vec<_>>();
                let padded = std::iter::once(917u32)
                    .chain(positions.iter().copied())
                    .collect::<Vec<_>>();
                vec![
                    value(vec![routes, rows], &transposed),
                    value(vec![routes + 1], &padded),
                ]
            } else {
                vec![
                    value(vec![rows, routes], &experts),
                    value(vec![routes], &positions),
                ]
            };
            let original = reference
                .execute(&bindings, &[], &CancellationFlag::new())
                .unwrap();
            assert_eq!(original[0].read_storage_bytes().unwrap(), expected);
            let cancelled = CancellationFlag::new();
            cancelled.cancel();
            assert!(executable.execute(&bindings, &[], &cancelled).is_err());
            let result = executable
                .execute(&bindings, &[], &CancellationFlag::new())
                .unwrap()
                .remove(0);
            assert_eq!(
                result.read_storage_bytes().unwrap(),
                expected,
                "rows={rows} routes={routes} mode={mode}"
            );
            if mode < 2 {
                let jobs = (0..2)
                    .map(|_| {
                        let exe = executable.clone();
                        let inputs = bindings.clone();
                        let bytes = expected.clone();
                        std::thread::spawn(move || {
                            let out = exe.execute(&inputs, &[], &CancellationFlag::new()).unwrap();
                            assert_eq!(out[0].read_storage_bytes().unwrap(), bytes);
                        })
                    })
                    .collect::<Vec<_>>();
                for job in jobs {
                    job.join().unwrap();
                }
            }
            retained.push((result, expected));
        }
        drop(executable);
        drop(reference);
        for (value, bytes) in retained {
            assert_eq!(value.read_storage_bytes().unwrap(), bytes);
        }
    }
}

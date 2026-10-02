use super::*;
use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
use effect_torch_graph::Node;
use effect_torch_runtime::{CancellationFlag, StorageMetadata};
use std::sync::Arc;

fn chain(rows: usize, routes: usize, reverse: bool) -> (Arc<Node>, Vec<Arc<Node>>) {
    let make = |kind| Node::new(kind).unwrap();
    let mut covered = super::grouped_retained_tests::chain(rows * routes, 2816, true, false, true);
    let input = |slot, shape, dtype| {
        make(NodeKind::Input {
            slot,
            shape,
            dtype,
            device: Device::Cuda(0),
            storage: StorageMetadata::dense(),
        })
    };
    let projection = covered[5].clone();
    let mut add = |kind| {
        let node = make(kind);
        covered.push(node.clone());
        node
    };
    let route = add(NodeKind::Reshape {
        a: projection,
        shape: vec![routes, rows, 2816],
    });
    let token = add(NodeKind::Permute {
        a: route,
        dims: vec![1, 0, 2],
    });
    let floating = add(NodeKind::Cast {
        a: token,
        dtype: DType::F32,
    });
    let product = add(NodeKind::Mul {
        a: floating,
        b: input(2, vec![rows, routes, 1], DType::F32),
    });
    let weighted = add(NodeKind::Cast {
        a: product,
        dtype: DType::BF16,
    });
    let ranks = add(NodeKind::Reshape {
        a: input(3, vec![rows, routes], DType::U32),
        shape: vec![rows, routes, 1],
    });
    let ranks = add(NodeKind::BroadcastTo {
        a: ranks,
        shape: vec![rows, routes, 2816],
    });
    let zero = add(NodeKind::Zeros {
        shape: vec![rows, routes, 2816],
        dtype: DType::BF16,
        device: Device::Cuda(0),
    });
    let scatter = add(NodeKind::ScatterAdd {
        a: zero,
        dim: 1,
        indexes: ranks,
        src: weighted,
    });
    let mut output = add(NodeKind::Zeros {
        shape: vec![rows, 2816],
        dtype: DType::BF16,
        device: Device::Cuda(0),
    });
    for position in 0..routes {
        let r = if reverse {
            routes - 1 - position
        } else {
            position
        };
        let sliced = add(NodeKind::Slice {
            a: scatter.clone(),
            ranges: vec![(0, rows, 1), (r, r + 1, 1), (0, 2816, 1)],
        });
        let selected = add(NodeKind::Reshape {
            a: sliced,
            shape: vec![rows, 2816],
        });
        output = add(NodeKind::Add {
            a: output,
            b: selected,
        });
    }
    (output, covered)
}

#[test]
fn expert_finalize_private_projection_intermediates_order_and_independent_opt_out() {
    for (enabled, retained, weighted, reverse, routes, expected) in [
        (true, true, true, false, 3, true),
        (false, true, true, false, 3, false),
        (true, false, true, false, 3, false),
        (true, true, false, false, 3, false),
        (true, true, true, true, 3, false),
        (true, true, true, false, 33, false),
    ] {
        let (output, nodes) = chain(2, routes, reverse);
        for escape in 0..nodes.len() * 2 - 1 {
            let last = nodes.len() - 1;
            let mut roots = vec![output.clone()];
            if escape < last {
                roots.push(nodes[escape].clone());
            } else if escape < last * 2 {
                roots.push(
                    Node::new(NodeKind::Neg {
                        a: nodes[escape - last].clone(),
                    })
                    .unwrap(),
                );
            }
            let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
                .prepare()
                .unwrap();
            let mut caps = CudaCapabilities::new(0, 12, 0);
            caps.expert_finalize = enabled;
            caps.grouped_retained_layout = retained;
            caps.ordered_scatter_fusion = true;
            caps.ordered_scatter_weighted = weighted;
            let driver = CompilerDriver::new(&prepared, &caps).unwrap();
            let selected =
                driver.optimization().regions.iter().any(
                    |r| matches!(r,NativeRegion::GroupedExpertGated(g) if g.finalizer.is_some()),
                );
            assert_eq!(
                selected,
                expected && escape == last * 2,
                "enabled={enabled} retained={retained} weighted={weighted} reverse={reverse} routes={routes} escape={escape}"
            );
            if !enabled && escape == last * 2 {
                assert!(driver.optimization().regions.iter().any(
                    |r| matches!(r,NativeRegion::GroupedExpertGated(g) if g.finalizer.is_none())
                ));
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_EXPERT_FINALIZE=1 with retained and weighted ordered flags"]
fn expert_finalize_exact_duplicate_ranks_nonfinite_cancel_concurrent_and_retained() {
    assert_eq!(
        std::env::var("EFFECT_TORCH_CUDA_EXPERT_FINALIZE").as_deref(),
        Ok("1")
    );
    let rows = 3;
    let routes = 3;
    let routed = rows * routes;
    let (output, nodes) = chain(rows, routes, false);
    let executable = Arc::new(crate::compile(vec![output.clone()], 0).unwrap());
    // A retained projected down output forces31scatter while retaining both GEMMs.
    let reference = crate::compile(vec![output, nodes[5].clone()], 0).unwrap();
    assert!(executable
        .diagnostics()
        .instructions
        .iter()
        .any(|i| i.kind == "et_ordered_sorted_reduce"));
    assert!(!reference
        .diagnostics()
        .instructions
        .iter()
        .any(|i| i.kind == "et_ordered_sorted_reduce"));
    assert!(reference
        .diagnostics()
        .instructions
        .iter()
        .any(|i| i.kind == "group_sorted_activation"));
    let device = crate::CudaDevice::get(0).unwrap();
    let host = |shape, dtype, data: &[f64]| {
        crate::CudaValue::from_host(device.clone(), shape, dtype, data).unwrap()
    };
    let mut retained = Vec::new();
    for mode in 0..8 {
        let experts = (0..routed)
            .map(|i| match mode {
                1 => 127.,
                2 => (routed - i - 1) as f64,
                _ => ((i * 47) % 128) as f64,
            })
            .collect::<Vec<_>>();
        let ranks = (0..rows * routes)
            .map(|i| {
                if mode == 5 {
                    0.
                } else {
                    ((i % routes + 1) % routes) as f64
                }
            })
            .collect::<Vec<_>>();
        let input = (0..routed * 2816)
            .map(|i| match mode {
                2 => [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.][(i / 2816) % 4],
                3 => {
                    if i % 2 == 0 {
                        -0.
                    } else {
                        0.
                    }
                }
                4 => half::bf16::from_bits(((i % 128) as u16) | 1).to_f64(),
                _ => ((i * 31 % 103) as f64 - 51.) / 19.,
            })
            .collect::<Vec<_>>();
        let weights = (0..routed)
            .map(|i| {
                if mode == 6 {
                    [f64::NAN, f64::INFINITY, -0.][i % 3]
                } else {
                    (i as f64 - 3.7) / 5.3
                }
            })
            .collect::<Vec<_>>();
        let bindings = vec![
            host(vec![routed, 2816], DType::BF16, &input),
            host(vec![routed], DType::U32, &experts),
            host(vec![rows, routes, 1], DType::F32, &weights),
            host(vec![rows, routes], DType::U32, &ranks),
        ];
        let expected = reference
            .execute(&bindings, &[], &CancellationFlag::new())
            .unwrap()[0]
            .read_storage_bytes()
            .unwrap();
        let cancelled = CancellationFlag::new();
        cancelled.cancel();
        assert!(executable.execute(&bindings, &[], &cancelled).is_err());
        let interrupted = CancellationFlag::new();
        assert!(executable
            .execute_with_gemm_hook(&bindings, &interrupted, &|| interrupted.cancel())
            .is_err());
        let actual = executable
            .execute(&bindings, &[], &CancellationFlag::new())
            .unwrap()
            .remove(0);
        assert_eq!(
            actual.read_storage_bytes().unwrap(),
            expected,
            "mode={mode}"
        );
        if mode < 2 {
            let jobs = (0..2)
                .map(|_| {
                    let exe = executable.clone();
                    let values = bindings.clone();
                    let bytes = expected.clone();
                    std::thread::spawn(move || {
                        let out = exe.execute(&values, &[], &CancellationFlag::new()).unwrap();
                        assert_eq!(out[0].read_storage_bytes().unwrap(), bytes);
                    })
                })
                .collect::<Vec<_>>();
            for job in jobs {
                job.join().unwrap();
            }
        }
        if mode == 0 {
            for slot in [1, 3] {
                let mut invalid = bindings.clone();
                invalid[slot] = if slot == 1 {
                    host(vec![routed], DType::U32, &vec![128.; routed])
                } else {
                    host(vec![rows, routes], DType::U32, &vec![routes as f64; routed])
                };
                assert!(executable
                    .execute(&invalid, &[], &CancellationFlag::new())
                    .is_err());
                assert!(reference
                    .execute(&invalid, &[], &CancellationFlag::new())
                    .is_err());
            }
        }
        retained.push((actual, expected));
    }
    drop(executable);
    drop(reference);
    for (value, bytes) in retained {
        assert_eq!(value.read_storage_bytes().unwrap(), bytes);
    }
}

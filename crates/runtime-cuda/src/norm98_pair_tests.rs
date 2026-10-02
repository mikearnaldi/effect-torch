use super::*;
use effect_torch_graph::{Device, Node};
use effect_torch_runtime::StorageMetadata;
use std::sync::Arc;

pub(crate) struct FixtureGraph {
    pub(crate) roots: Vec<Arc<Node>>,
    pub(crate) entrance_inputs: Vec<Arc<Node>>,
    pub(crate) entrance_outputs: [Arc<Node>; 4],
    pub(crate) entrance_nodes: Vec<Arc<Node>>,
    pub(crate) alias: Arc<Node>,
    pub(crate) tail_inputs: Vec<Arc<Node>>,
    pub(crate) tail_outputs: [Arc<Node>; 2],
    pub(crate) tail_nodes: Vec<Arc<Node>>,
}

pub(crate) fn fixture(rows: usize, dynamic_rho: bool) -> FixtureGraph {
    let node = |kind| Node::new(kind).unwrap();
    let matrix = |slot| {
        node(NodeKind::Input {
            slot,
            shape: vec![1, rows, 2816],
            dtype: DType::BF16,
            device: Device::Cuda(0),
            storage: StorageMetadata::dense(),
        })
    };
    let a = matrix(0);
    let h = matrix(1);
    let d = matrix(2);
    let e = matrix(3);
    let weight = node(NodeKind::Full {
        shape: vec![2816],
        dtype: DType::BF16,
        value: 1.0,
        device: Device::Cuda(0),
    });
    // Distinct entrance weights keep dense/expert outputs distinct after CSE.
    let dense_weight = node(NodeKind::Full {
        shape: vec![2816],
        dtype: DType::BF16,
        value: 1.125,
        device: Device::Cuda(0),
    });
    let expert_weight = node(NodeKind::Full {
        shape: vec![2816],
        dtype: DType::BF16,
        value: 0.875,
        device: Device::Cuda(0),
    });
    let other_weight = |value| {
        node(NodeKind::Full {
            shape: vec![2816],
            dtype: DType::BF16,
            value,
            device: Device::Cuda(0),
        })
    };
    let router_weight = other_weight(1.25);
    let dense_post = other_weight(1.5);
    let expert_post = other_weight(1.75);
    let combined_post = other_weight(0.5);
    let next_weight = other_weight(0.75);
    let scale = node(NodeKind::Full {
        shape: vec![1],
        dtype: DType::BF16,
        value: 1.0,
        device: Device::Cuda(0),
    });
    let rho = if dynamic_rho {
        node(NodeKind::Input {
            slot: 4,
            shape: vec![],
            dtype: DType::F32,
            device: Device::Cuda(0),
            storage: StorageMetadata::dense(),
        })
    } else {
        node(NodeKind::Full {
            shape: vec![],
            dtype: DType::F32,
            value: 2816_f64.sqrt().recip(),
            device: Device::Cuda(0),
        })
    };
    let norm = |x, w| {
        node(NodeKind::RmsNorm {
            x,
            weight: w,
            eps: 1e-6,
        })
    };
    let attention = norm(a.clone(), Some(weight.clone()));
    let residual = node(NodeKind::Add {
        a: h.clone(),
        b: attention.clone(),
    });
    let dense = norm(residual.clone(), Some(dense_weight.clone()));
    let expert = norm(residual.clone(), Some(expert_weight.clone()));
    let alias = node(NodeKind::Reshape {
        a: residual.clone(),
        shape: vec![rows, 2816],
    });
    let router_norm = norm(alias.clone(), None);
    let learned = node(NodeKind::Mul {
        a: router_norm.clone(),
        b: router_weight.clone(),
    });
    let widened = node(NodeKind::Cast {
        a: learned.clone(),
        dtype: DType::F32,
    });
    let scaled = node(NodeKind::Mul {
        a: widened.clone(),
        b: rho.clone(),
    });
    let router = node(NodeKind::Cast {
        a: scaled.clone(),
        dtype: DType::BF16,
    });
    let dense_norm = norm(d.clone(), Some(dense_post.clone()));
    let expert_norm = norm(e.clone(), Some(expert_post.clone()));
    let combined = node(NodeKind::Add {
        a: dense_norm.clone(),
        b: expert_norm.clone(),
    });
    let combined_norm = norm(combined.clone(), Some(combined_post.clone()));
    let added = node(NodeKind::Add {
        a: residual.clone(),
        b: combined_norm.clone(),
    });
    let next = node(NodeKind::Mul {
        a: added.clone(),
        b: scale.clone(),
    });
    let next_norm = norm(next.clone(), Some(next_weight.clone()));
    FixtureGraph {
        roots: vec![
            dense.clone(),
            expert.clone(),
            router.clone(),
            next.clone(),
            next_norm.clone(),
        ],
        entrance_inputs: vec![
            a,
            h,
            weight.clone(),
            dense_weight,
            expert_weight,
            router_weight,
            rho,
        ],
        entrance_outputs: [
            residual.clone(),
            dense.clone(),
            expert.clone(),
            router.clone(),
        ],
        entrance_nodes: vec![
            attention,
            residual.clone(),
            dense,
            expert,
            alias.clone(),
            router_norm,
            learned,
            widened,
            scaled,
            router,
        ],
        alias: alias.clone(),
        tail_inputs: vec![
            d,
            e,
            residual,
            dense_post,
            expert_post,
            combined_post,
            scale,
            next_weight,
        ],
        tail_outputs: [next.clone(), next_norm.clone()],
        tail_nodes: vec![
            dense_norm,
            expert_norm,
            combined,
            combined_norm,
            added,
            next,
            next_norm,
        ],
    }
}

fn indexed(
    graph: &FixtureGraph,
    roots: &[Arc<Node>],
    rows: usize,
) -> (GraphIndex, AttentionFfnEntranceRegion, Vec<NativeRegion>) {
    let index = GraphIndex::new(roots).unwrap();
    let id = |node: &Arc<Node>| index.dense_id(node.id).unwrap();
    let entrance = AttentionFfnEntranceRegion {
        nodes: graph.entrance_nodes.iter().map(id).collect(),
        inputs: graph.entrance_inputs.iter().map(id).collect(),
        outputs: graph.entrance_outputs.clone().map(|node| id(&node)),
        residual_views: vec![id(&graph.alias)].into_boxed_slice(),
        rows,
    };
    let tail = FfnNextNormRegion {
        nodes: graph.tail_nodes.iter().map(id).collect(),
        inputs: graph.tail_inputs.iter().map(id).collect(),
        outputs: graph.tail_outputs.clone().map(|node| id(&node)),
        residual_views: vec![].into_boxed_slice(),
        rows,
    };
    let regions = vec![
        NativeRegion::AttentionFfnEntrance(entrance.clone()),
        NativeRegion::FfnNextNorm(tail),
    ];
    (index, entrance, regions)
}

fn roots_with_early_kv(graph: &FixtureGraph) -> Vec<Arc<Node>> {
    let kv_inputs = (4..7)
        .map(|slot| {
            Node::new(NodeKind::Input {
                slot,
                shape: vec![1, 1, 2, 2],
                dtype: DType::BF16,
                device: Device::Cuda(0),
                storage: StorageMetadata::dense(),
            })
            .unwrap()
        })
        .collect::<Vec<_>>();
    let attention = Node::new(NodeKind::KvAttention {
        q: kv_inputs[0].clone(),
        k: kv_inputs[1].clone(),
        v: kv_inputs[2].clone(),
        scale: 0.5,
        layer: 0,
        window: None,
        mode: effect_torch_graph::KvAttentionMode::BidirectionalBlock,
        rounding: effect_torch_graph::AttentionRounding::Stepwise,
    })
    .unwrap();
    let mut roots = graph.entrance_inputs[..2].to_vec();
    roots.extend(graph.tail_inputs[..2].iter().cloned());
    roots.extend(kv_inputs);
    // KV transaction declarations precede the entrance region, creating a
    // real difference between command and lowered-instruction positions.
    roots.push(attention);
    roots.extend(graph.roots.iter().cloned());
    roots
}

#[test]
fn actual_compiler_preserves_private_pair_after_earlier_stateful_kv() {
    use crate::capabilities::CudaCapabilities;
    use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest, StateCursorSlot};
    for rows in [64, 256] {
        let graph = fixture(rows, false);
        let prepared =
            ProgramRequest::from_roots(roots_with_early_kv(&graph), CompileOptions::default())
                .with_state_cursor(StateCursorSlot::new(20, true))
                .prepare()
                .unwrap();
        let mut caps = CudaCapabilities::new(0, 12, 0);
        caps.enable_norm98_fixture_regions();
        let driver = CompilerDriver::new(&prepared, &caps).unwrap();
        let entrances = driver
            .optimization()
            .regions
            .iter()
            .filter_map(|region| match region {
                NativeRegion::AttentionFfnEntrance(region) => Some(region),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(entrances.len(), 1);
        assert!(find_pair(&prepared.index, driver.optimization(), entrances[0]).is_some());
        let kv = prepared
            .index
            .order
            .iter()
            .position(|node| matches!(node.kind, NodeKind::KvAttention { .. }))
            .unwrap();
        assert!(
            kv < entrances[0]
                .nodes
                .iter()
                .map(|id| id.index())
                .min()
                .unwrap()
        );
    }
}

#[test]
#[ignore = "requires CUDA SM120, TRITON_NORM98_DIRECTORY, entrance/FFN63 and DIV_FEEDBACK=1"]
fn norm98_hardware_stateful_mapping() {
    use crate::{
        CudaDevice, CudaExecutable, CudaSequenceState, CudaStateInvocation, CudaStateLayout,
        CudaValue,
    };
    use effect_torch_compiler::CompileOptions;
    use effect_torch_runtime::{CancellationFlag, KvLayerDescriptor, StateAccessMode};
    assert_eq!(
        std::env::var("EFFECT_TORCH_CUDA_DIV_FEEDBACK").as_deref(),
        Ok("1")
    );
    let device = CudaDevice::get(0).unwrap();
    let bytes = |snapshot: &crate::CudaKvSnapshot| {
        snapshot
            .layers
            .iter()
            .flat_map(|layer| {
                layer.pages.iter().flat_map(|page| {
                    [
                        device.stream.clone_dtoh(&page.keys).unwrap(),
                        device.stream.clone_dtoh(&page.values).unwrap(),
                    ]
                })
            })
            .collect::<Vec<_>>()
    };
    let mut retained = Vec::new();
    for rows in [64, 256] {
        let graph = fixture(rows, false);
        let roots = roots_with_early_kv(&graph);
        let mut reference_roots = roots.clone();
        reference_roots.push(graph.entrance_outputs[0].clone());
        let layout = |access| CudaStateLayout {
            capacity: 4,
            dtype: DType::BF16,
            slots: 1,
            packed_rows_per_sequence: None,
            access,
            kv_layers: vec![KvLayerDescriptor {
                layer_id: 0,
                kv_heads: 1,
                head_dim: 2,
                dtype: DType::BF16,
                retention: None,
            }],
        };
        let compile = |roots, access| {
            crate::compile_stateful_with_layout(
                roots,
                0,
                20,
                true,
                CompileOptions::default(),
                layout(access),
            )
            .unwrap()
        };
        let optimized = compile(roots.clone(), StateAccessMode::Append);
        let reference = compile(reference_roots, StateAccessMode::Append);
        let read_only = compile(roots, StateAccessMode::ReadOnly);
        let count = |executable: &CudaExecutable, name| {
            executable
                .diagnostics()
                .instructions
                .iter()
                .filter(|instruction| instruction.kind == name)
                .map(|instruction| instruction.count)
                .sum::<usize>()
        };
        for executable in [&optimized, &read_only] {
            assert_eq!(count(executable, "triton_norm98_entrance"), 1);
            assert_eq!(count(executable, "triton_norm98_tail"), 1);
            assert_eq!(count(executable, "et_attention_ffn_entrance_bf16"), 0);
            assert_eq!(count(executable, "et_ffn_next_norm63_bf16"), 0);
            assert!(
                count(executable, "state_prepare") > 0,
                "earlier KV state preparation must survive retrospective entrance replacement"
            );
        }
        assert_eq!(
            count(&optimized, "state_prepare"),
            count(&reference, "state_prepare")
        );
        assert_eq!(
            count(&optimized, "state_commit"),
            count(&reference, "state_commit")
        );
        assert!(count(&optimized, "state_commit") > 0);
        assert!(count(&read_only, "state_discard") > 0);
        assert_eq!(count(&reference, "triton_norm98_entrance"), 0);
        let mut bindings = [0.25, 0.5, 0.25, 0.25]
            .map(|value| {
                CudaValue::from_host(
                    device.clone(),
                    vec![1, rows, 2816],
                    DType::BF16,
                    &vec![value; rows * 2816],
                )
                .unwrap()
            })
            .to_vec();
        for data in [vec![0.0; 4], vec![0.0; 4], vec![1.0, 2.0, 3.0, 4.0]] {
            bindings.push(
                CudaValue::from_host(device.clone(), vec![1, 1, 2, 2], DType::BF16, &data).unwrap(),
            );
        }
        let borrowed = bindings
            .iter()
            .map(|value| value.read_storage_bytes().unwrap())
            .collect::<Vec<_>>();
        let invocation = |access, cursor, snapshot| CudaStateInvocation {
            sequences: vec![CudaSequenceState {
                cursor,
                keys: vec![],
                values: vec![],
                kda_states: vec![],
                conv_states: vec![],
                kv_storage: snapshot,
            }],
            slots: vec![0],
            valid_lengths: vec![2],
            capacity: 4,
            cache_dtype: DType::BF16,
            packed_rows_per_sequence: None,
            kv_layers: layout(access).kv_layers,
            access,
            cache: None,
        };
        let mut optimized_state = invocation(StateAccessMode::Append, 0, None);
        let mut reference_state = invocation(StateAccessMode::Append, 0, None);
        let actual = optimized
            .execute_stateful(
                &bindings,
                &[],
                &mut optimized_state,
                &CancellationFlag::new(),
            )
            .unwrap();
        let expected = reference
            .execute_stateful(
                &bindings,
                &[],
                &mut reference_state,
                &CancellationFlag::new(),
            )
            .unwrap();
        assert_eq!(actual.len(), 13);
        assert_eq!(
            actual[7].read_storage_bytes().unwrap(),
            expected[7].read_storage_bytes().unwrap(),
            "KV result is preserved before paired normalization"
        );
        for (got, expected) in actual[8..13].iter().zip(&expected[8..13]) {
            let got = got.read_storage_bytes().unwrap();
            let expected = expected.read_storage_bytes().unwrap();
            for (got, expected) in got.chunks_exact(2).zip(expected.chunks_exact(2)) {
                let got = f32::from_bits(u32::from(u16::from_le_bytes([got[0], got[1]])) << 16);
                let expected =
                    f32::from_bits(u32::from(u16::from_le_bytes([expected[0], expected[1]])) << 16);
                assert!((got - expected).abs() <= 0.0078125 * expected.abs().max(0.01));
            }
        }
        optimized.readback_state(&mut optimized_state).unwrap();
        reference.readback_state(&mut reference_state).unwrap();
        let prefix = optimized_state.sequences[0].kv_storage.clone().unwrap();
        let prefix_bytes = bytes(&prefix);
        assert_eq!(
            prefix_bytes,
            bytes(reference_state.sequences[0].kv_storage.as_ref().unwrap())
        );
        assert_eq!(prefix.layers.len(), 1);
        let mut read_state = invocation(StateAccessMode::ReadOnly, 2, Some(prefix.clone()));
        let cancelled = CancellationFlag::new();
        cancelled.cancel();
        assert!(read_only
            .execute_stateful(&bindings, &[], &mut read_state, &cancelled)
            .is_err());
        assert!(read_only
            .execute_stateful(
                &bindings[..6],
                &[],
                &mut read_state,
                &CancellationFlag::new()
            )
            .is_err());
        let interrupted = CancellationFlag::new();
        let reached = std::cell::Cell::new(false);
        assert!(read_only
            .execute_stateful_with_kv_hook(&bindings, &mut read_state, &interrupted, &|| {
                reached.set(true);
                interrupted.cancel();
            })
            .is_err());
        assert!(
            reached.get(),
            "actual earlier KV stage cancellation hook required"
        );
        assert_eq!(bytes(&prefix), prefix_bytes);
        let read = read_only
            .execute_stateful(&bindings, &[], &mut read_state, &CancellationFlag::new())
            .unwrap();
        assert_eq!(read.len(), 13);
        assert_eq!(
            bytes(&prefix),
            prefix_bytes,
            "read-only execution preserves source prefix"
        );
        for (value, expected) in bindings.iter().zip(&borrowed) {
            assert_eq!(value.read_storage_bytes().unwrap(), *expected);
        }
        let saved = actual
            .iter()
            .map(|value| value.read_storage_bytes().unwrap())
            .collect::<Vec<_>>();
        retained.push((actual, saved));
        let saved = read
            .iter()
            .map(|value| value.read_storage_bytes().unwrap())
            .collect::<Vec<_>>();
        retained.push((read, saved));
    }
    for (values, expected) in retained {
        for (value, expected) in values.iter().zip(expected) {
            assert_eq!(value.read_storage_bytes().unwrap(), expected);
        }
    }
}

#[test]
fn actual_compiler_selects_one_private_pair_for_both_row_counts() {
    use crate::capabilities::CudaCapabilities;
    use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
    for rows in [64, 256] {
        let graph = fixture(rows, false);
        let mut roots = graph.entrance_inputs[..2].to_vec();
        roots.extend(graph.tail_inputs[..2].iter().cloned());
        roots.extend(graph.roots.iter().cloned());
        let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
            .prepare()
            .unwrap();
        let mut caps = CudaCapabilities::new(0, 12, 0);
        caps.enable_norm98_fixture_regions();
        let driver = CompilerDriver::new(&prepared, &caps).unwrap();
        let regions = &driver.optimization().regions;
        let entrances = regions
            .iter()
            .filter_map(|region| match region {
                NativeRegion::AttentionFfnEntrance(region) => Some(region),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(entrances.len(), 1, "rows={rows} regions={regions:#?}");
        assert_eq!(
            regions
                .iter()
                .filter(|region| matches!(region, NativeRegion::FfnNextNorm(_)))
                .count(),
            1,
            "rows={rows}"
        );
        assert_eq!(
            find_pair(&prepared.index, driver.optimization(), entrances[0]),
            Some((
                prepared.index.dense_id(graph.tail_outputs[0].id).unwrap(),
                2816_f64.sqrt().recip() as f32,
            ))
        );
    }
}

#[test]
fn accepts_only_private_reshape_residual_and_immutable_f32_rho() {
    for rows in [64, 256] {
        let graph = fixture(rows, false);
        let (index, entrance, regions) = indexed(&graph, &graph.roots, rows);
        let pair = find_in_regions(&index, &regions, &entrance).unwrap();
        assert_eq!(pair.0, index.dense_id(graph.tail_outputs[0].id).unwrap());
        assert_eq!(pair.1, 2816_f64.sqrt().recip() as f32);
        let graph = fixture(rows, true);
        let (index, entrance, regions) = indexed(&graph, &graph.roots, rows);
        assert!(find_in_regions(&index, &regions, &entrance).is_none());
    }
}

#[test]
fn rejects_direct_or_aliased_residual_outputs_and_external_observers() {
    let graph = fixture(64, false);
    let residual = graph.entrance_outputs[0].clone();
    let extra_alias = Node::new(NodeKind::Reshape {
        a: graph.alias.clone(),
        shape: vec![1, 64, 2816],
    })
    .unwrap();
    let observer = Node::new(NodeKind::Mul {
        a: extra_alias.clone(),
        b: graph.tail_inputs[6].clone(),
    })
    .unwrap();
    for escaped in [residual, graph.alias.clone(), extra_alias, observer] {
        let mut roots = graph.roots.clone();
        roots.push(escaped);
        let (index, entrance, regions) = indexed(&graph, &roots, 64);
        assert!(find_in_regions(&index, &regions, &entrance).is_none());
    }
}

#[test]
fn rejects_another_tail_consumer_and_mismatched_geometry() {
    let graph = fixture(64, false);
    let second = Node::new(NodeKind::Add {
        a: graph.alias.clone(),
        b: graph.tail_outputs[0].clone(),
    })
    .unwrap();
    let mut roots = graph.roots.clone();
    roots.push(second);
    let (index, entrance, regions) = indexed(&graph, &roots, 64);
    assert!(find_in_regions(&index, &regions, &entrance).is_none());
    let (index, mut entrance, regions) = indexed(&graph, &graph.roots, 64);
    entrance.rows = 256;
    assert!(find_in_regions(&index, &regions, &entrance).is_none());
}

#[test]
#[ignore = "requires CUDA SM120, TRITON_NORM98_DIRECTORY, entrance and FFN63 flags"]
fn norm98_hardware_paired_lifecycle() {
    use crate::{CudaDevice, CudaValue};
    use effect_torch_runtime::CancellationFlag;
    let device = CudaDevice::get(0).unwrap();
    let mut retained = Vec::new();
    for rows in [64, 256] {
        let graph = fixture(rows, false);
        // Input roots establish early binding boundaries without exposing R.
        let mut roots = graph.entrance_inputs[..2].to_vec();
        roots.extend(graph.tail_inputs[..2].iter().cloned());
        roots.extend(graph.roots.iter().cloned());
        let optimized = Arc::new(crate::compile(roots.clone(), 0).unwrap());
        let count = |name| {
            optimized
                .diagnostics()
                .instructions
                .iter()
                .filter(|instruction| instruction.kind == name)
                .map(|instruction| instruction.count)
                .sum::<usize>()
        };
        assert_eq!(count("triton_norm98_entrance"), 1);
        assert_eq!(count("triton_norm98_tail"), 1);
        assert_eq!(count("et_attention_ffn_entrance_bf16"), 0);
        assert_eq!(count("et_ffn_next_norm63_bf16"), 0);
        // Observing the entrance residual must preserve the old numerical path.
        roots.push(graph.entrance_outputs[0].clone());
        let reference = crate::compile(roots, 0).unwrap();
        assert!(!reference
            .diagnostics()
            .instructions
            .iter()
            .any(|instruction| instruction.kind == "triton_norm98_entrance"));
        for magnitude in [0.25, 1.0] {
            let bindings = [magnitude, 0.5, magnitude, magnitude].map(|value| {
                CudaValue::from_host(
                    device.clone(),
                    vec![1, rows, 2816],
                    DType::BF16,
                    &vec![value; rows * 2816],
                )
                .unwrap()
            });
            let borrowed = bindings
                .iter()
                .map(|value| value.read_storage_bytes().unwrap())
                .collect::<Vec<_>>();
            let expected = reference
                .execute(&bindings, &[], &CancellationFlag::new())
                .unwrap()
                .into_iter()
                .skip(4)
                .take(5)
                .map(|value| value.read_storage_bytes().unwrap())
                .collect::<Vec<_>>();
            let pre_cancel = CancellationFlag::new();
            pre_cancel.cancel();
            assert!(optimized.execute(&bindings, &[], &pre_cancel).is_err());
            assert!(optimized
                .execute(&bindings[..3], &[], &CancellationFlag::new())
                .is_err());
            let interrupted = CancellationFlag::new();
            let reached = std::cell::Cell::new(false);
            assert!(optimized
                .execute_with_ffn63_hook(&bindings, &interrupted, &|| {
                    reached.set(true);
                    interrupted.cancel();
                })
                .is_err());
            assert!(reached.get(), "actual98 tail cancellation hook required");
            let actual = optimized
                .execute(&bindings, &[], &CancellationFlag::new())
                .unwrap()
                .into_iter()
                .skip(4)
                .collect::<Vec<_>>();
            assert_eq!(actual.len(), 5);
            let decode = |bytes: &[u8]| {
                bytes
                    .chunks_exact(2)
                    .map(|bits| {
                        f32::from_bits(u32::from(u16::from_le_bytes([bits[0], bits[1]])) << 16)
                    })
                    .collect::<Vec<_>>()
            };
            for (value, expected) in actual.iter().zip(&expected) {
                let got = decode(&value.read_storage_bytes().unwrap());
                let expected = decode(expected);
                for (got, expected) in got.iter().zip(expected) {
                    assert!((got - expected).abs() <= 0.0078125 * expected.abs().max(0.01));
                }
            }
            let hidden = actual[3].storage_address();
            let normalized = actual[4].storage_address();
            let bytes_per_output = (rows * 2816 * 2) as u64;
            assert!(
                hidden + bytes_per_output <= normalized || normalized + bytes_per_output <= hidden,
                "nextHidden and nextNorm have disjoint storage ranges"
            );
            for (value, expected) in bindings.iter().zip(&borrowed) {
                assert_eq!(value.read_storage_bytes().unwrap(), *expected);
            }
            let bytes = actual
                .iter()
                .map(|value| value.read_storage_bytes().unwrap())
                .collect::<Vec<_>>();
            let mut workers = Vec::new();
            for _ in 0..2 {
                let executable = optimized.clone();
                let bindings = bindings.clone();
                let bytes = bytes.clone();
                workers.push(std::thread::spawn(move || {
                    let output = executable
                        .execute(&bindings, &[], &CancellationFlag::new())
                        .unwrap();
                    for (value, expected) in output.iter().skip(4).zip(bytes) {
                        assert_eq!(value.read_storage_bytes().unwrap(), expected);
                    }
                }));
            }
            for worker in workers {
                worker.join().unwrap();
            }
            // A sibling output may share the ordinary allocation lease while
            // owning a distinct range. Drop nextHidden and retain nextNorm to
            // prove its lease survives independently of the sibling wrapper.
            let next_norm = actual[4].clone();
            let next_norm_bytes = bytes[4].clone();
            retained.push((actual, bytes));
            retained.push((vec![next_norm], vec![next_norm_bytes]));
        }
        drop(optimized);
        drop(reference);
    }
    for (values, expected) in retained {
        for (value, expected) in values.iter().zip(expected) {
            assert_eq!(value.read_storage_bytes().unwrap(), expected);
        }
    }
}

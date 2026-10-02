use super::*;
use crate::capabilities::CudaCapabilities;
use effect_torch_compiler::{
    CompileOptions, CompilerDriver, ProgramRequest, TargetDTypeCapabilities,
};
use effect_torch_graph::Device;

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
fn cast(a: Arc<Node>, dtype: DType) -> Arc<Node> {
    Node::new(NodeKind::Cast { a, dtype }).unwrap()
}
struct Fixture {
    inputs: Vec<Arc<Node>>,
    roots: Vec<Arc<Node>>,
    private: Vec<Arc<Node>>,
}
fn fixture(rows: usize, count: usize) -> Fixture {
    fixture_width(rows, count, 128)
}
fn fixture_width(rows: usize, count: usize, half: usize) -> Fixture {
    let positions = input(0, &[1, rows], DType::U32);
    let frequency = input(1, &[half], DType::F32);
    fixture_sources(rows, count, positions, frequency)
}
fn fixture_sources(
    rows: usize,
    count: usize,
    positions: Arc<Node>,
    frequency: Arc<Node>,
) -> Fixture {
    let mut f = Fixture {
        inputs: vec![positions.clone(), frequency.clone()],
        roots: Vec::new(),
        private: Vec::new(),
    };
    for _ in 0..count {
        let pos = cast(positions.clone(), DType::F32);
        let view = Node::new(NodeKind::Reshape {
            a: pos.clone(),
            shape: vec![1, rows, 1],
        })
        .unwrap();
        let freq = cast(frequency.clone(), DType::F32);
        let phase = Node::new(NodeKind::Mul {
            a: view,
            b: freq.clone(),
        })
        .unwrap();
        let doubled = Node::new(NodeKind::Concat {
            a: phase.clone(),
            b: phase.clone(),
            dim: 2,
        })
        .unwrap();
        let cosine = cast(
            Node::new(NodeKind::Cos { a: doubled.clone() }).unwrap(),
            DType::BF16,
        );
        let sine = cast(
            Node::new(NodeKind::Sin { a: doubled }).unwrap(),
            DType::BF16,
        );
        f.roots.extend([cosine, sine]);
        f.private.extend([pos, freq, phase]);
    }
    f
}
fn prepared(f: &Fixture) -> effect_torch_compiler::PreparedProgram {
    ProgramRequest::from_roots(f.roots.clone(), CompileOptions::default())
        .prepare()
        .unwrap()
}
#[test]
fn rotary66_semantic_plan_preserves_graph_and_scopes_shapes() {
    for rows in [16, 64, 256, 512] {
        let mut f = fixture(rows, 50);
        let global = fixture_sources(rows, 10, f.inputs[0].clone(), input(2, &[256], DType::F32));
        f.inputs.push(global.inputs[1].clone());
        f.roots.extend(global.roots);
        f.private.extend(global.private);
        let p = prepared(&f);
        let caps = CudaCapabilities::new(0, 12, 0).with_rotary_reuse66(true);
        let driver = CompilerDriver::new(&p, &caps).unwrap();
        let before = p.index.order.iter().map(|n| n.id).collect::<Vec<_>>();
        let reuse = RotaryReuse66::new(&p.index, driver.optimization(), driver.legalization());
        assert_eq!(
            reuse.pairs.len(),
            if matches!(rows, 64 | 256) { 58 } else { 0 }
        );
        assert_eq!(
            before,
            p.index.order.iter().map(|n| n.id).collect::<Vec<_>>()
        );
        for pair in &reuse.pairs {
            assert_eq!(pair.canonical.contracts, pair.duplicate.contracts);
            assert!(pair.canonical.positions.last() < pair.duplicate.positions.first());
        }
    }
}
#[test]
fn rotary66_observed_private_values_reject_whole_chain() {
    for private in 0..3 {
        let mut f = fixture(64, 2);
        f.roots.push(f.private[3 + private].clone());
        let p = prepared(&f);
        let caps = CudaCapabilities::new(0, 12, 0).with_rotary_reuse66(true);
        let driver = CompilerDriver::new(&p, &caps).unwrap();
        assert!(
            RotaryReuse66::new(&p.index, driver.optimization(), driver.legalization())
                .pairs
                .is_empty()
        );
    }
}
#[test]
fn rotary66_flag_off_fingerprint_and_policy_are_unchanged() {
    let off = CudaCapabilities::new(0, 12, 0).with_rotary_reuse66(false);
    let on = off.clone().with_rotary_reuse66(true);
    assert_eq!(on.policy_revision(), 66);
    assert_ne!(on.fingerprint(), off.fingerprint());
    assert_eq!(
        on.with_rotary_reuse66(false).fingerprint(),
        off.fingerprint()
    );
    let f = fixture(64, 2);
    let p = prepared(&f);
    let d = CompilerDriver::new(&p, &off).unwrap();
    assert!(
        RotaryReuse66::new(&p.index, d.optimization(), d.legalization())
            .pairs
            .is_empty()
    );
}

fn seed(builder: &mut CudaProgramBuilder, index: &GraphIndex, c: &Chain) {
    for (slot, semantic) in [c.nodes[0], c.nodes[3], c.nodes[8], c.nodes[10]]
        .into_iter()
        .enumerate()
    {
        let n = &index.order[semantic.index()];
        let value = builder
            .planned(n.shape.clone(), n.dtype, "fixture")
            .unwrap();
        if slot < 2 {
            builder.values[value.index()].decl.storage = ValueStorage::Fixed {
                class: StorageClass::ExternalInput,
                location: Location::External { slot: slot as u32 },
            };
        }
        builder
            .emit(
                "fixture",
                CommandKind::Prepare,
                Some(value),
                Vec::new(),
                slot >= 2,
            )
            .unwrap();
        builder.semantic_values[semantic.index()] = Some(value);
    }
}
#[test]
fn rotary66_rooted_aliases_extend_owner_lifetimes() {
    let f = fixture(64, 2);
    let p = prepared(&f);
    let caps = CudaCapabilities::new(0, 12, 0).with_rotary_reuse66(true);
    let mut driver = CompilerDriver::new(&p, &caps).unwrap();
    let mut reuse = RotaryReuse66::new(&p.index, driver.optimization(), driver.legalization());
    assert_eq!(reuse.pairs.len(), 1);
    let canonical = reuse.pairs[0].canonical.clone();
    let duplicate = reuse.pairs[0].duplicate.clone();
    let mut builder = CudaProgramBuilder::new(&p.index, None, driver.legalization()).unwrap();
    seed(&mut builder, &p.index, &canonical);
    for unit in duplicate.units {
        assert!(reuse.lower(unit, &p.index, &mut builder).unwrap());
    }
    let (program, commands) = builder.finish(&p.index).unwrap();
    assert_eq!(
        commands
            .iter()
            .filter(|c| matches!(c.kind, CommandKind::Alias { .. }))
            .count(),
        2
    );
    for (alias, owner) in program.outputs[2..].iter().zip(&program.outputs[..2]) {
        assert!(
            matches!(program.values[alias.index()].decl.storage,ValueStorage::Alias{source,byte_offset:0} if source==*owner)
        );
        assert!(matches!(
            program.values[owner.index()].decl.storage,
            ValueStorage::Planned {
                class: StorageClass::EscapingOutput,
                ownership: SegmentOwnership::ProvisionalOutput,
                ..
            }
        ));
    }
    let memory = driver
        .plan_memory(
            &program,
            &effect_torch_compiler::MemoryPlannerConfig::uniform(
                CudaMemorySpace::Device,
                usize::MAX / 2,
                CUDA_STORAGE_ALIGNMENT,
                CUDA_STORAGE_ALIGNMENT,
            ),
        )
        .unwrap();
    for (alias, owner) in program.outputs[2..].iter().zip(&program.outputs[..2]) {
        assert!(
            matches!(memory.locations[alias.index()],Location::Alias{root,byte_offset:0} if root==*owner)
        );
    }
}
#[test]
fn rotary66_physical_preflight_declines_every_unit_before_suppression() {
    for defect in 0..7 {
        let f = fixture(64, 2);
        let p = prepared(&f);
        let caps = CudaCapabilities::new(0, 12, 0).with_rotary_reuse66(true);
        let driver = CompilerDriver::new(&p, &caps).unwrap();
        let mut reuse = RotaryReuse66::new(&p.index, driver.optimization(), driver.legalization());
        let canonical = reuse.pairs[0].canonical.clone();
        let units = reuse.pairs[0].duplicate.units.clone();
        let mut builder = CudaProgramBuilder::new(&p.index, None, driver.legalization()).unwrap();
        seed(&mut builder, &p.index, &canonical);
        let cosine = builder.resolve(canonical.nodes[8].index()).unwrap();
        let sine = builder.resolve(canonical.nodes[10].index()).unwrap();
        match defect {
            0 => builder.semantic_values[canonical.nodes[10].index()] = None,
            1 => builder.values[sine.index()].decl.bytes -= 2,
            2 => builder.values[sine.index()].storage = StorageMetadata::unconstrained(),
            3 => {
                builder.values[sine.index()].decl.storage = ValueStorage::Alias {
                    source: cosine,
                    byte_offset: 0,
                }
            }
            4 => {
                builder.values[sine.index()].decl.storage = ValueStorage::Alias {
                    source: cosine,
                    byte_offset: 2,
                }
            }
            5 => {
                let middle = builder.planned(vec![1], DType::BF16, "too_short").unwrap();
                builder.values[middle.index()].decl.storage = ValueStorage::Alias {
                    source: cosine,
                    byte_offset: 0,
                };
                builder.values[sine.index()].decl.storage = ValueStorage::Alias {
                    source: middle,
                    byte_offset: 0,
                };
            }
            _ => {
                let raw = builder.resolve(canonical.nodes[0].index()).unwrap();
                builder.values[raw.index()].decl.storage = ValueStorage::Fixed {
                    class: StorageClass::PersistentState,
                    location: Location::Persistent { slot: 0 },
                };
            }
        }
        let before = builder.commands.len();
        for unit in units {
            assert!(!reuse.lower(unit, &p.index, &mut builder).unwrap());
        }
        assert_eq!(builder.commands.len(), before);
        assert!(matches!(reuse.states[0], State::Declined));
    }
}

#[test]
fn rotary66_equal_slots_different_semantic_versions_do_not_merge() {
    let mut f = fixture(64, 1);
    f.roots.extend(fixture(64, 1).roots);
    let p = prepared(&f);
    let caps = CudaCapabilities::new(0, 12, 0).with_rotary_reuse66(true);
    let driver = CompilerDriver::new(&p, &caps).unwrap();
    assert!(
        RotaryReuse66::new(&p.index, driver.optimization(), driver.legalization())
            .pairs
            .is_empty()
    );
}
#[test]
fn rotary66_modified_region_layout_expression_or_members_declines() {
    for defect in 0..3 {
        let f = fixture(64, 2);
        let p = prepared(&f);
        let caps = CudaCapabilities::new(0, 12, 0).with_rotary_reuse66(true);
        let driver = CompilerDriver::new(&p, &caps).unwrap();
        let good = RotaryReuse66::new(&p.index, driver.optimization(), driver.legalization());
        let output = good.pairs[0].duplicate.nodes[8];
        let mut plan = driver.optimization().clone();
        let rid = plan.node_region[output.index()].unwrap();
        let NativeRegion::Elementwise(r) = &mut plan.regions[rid.index()] else {
            panic!()
        };
        match defect {
            0 => r.lane_strides[0][2] = 2,
            1 => r.output.expression = KernelExpr::Input(0),
            _ => r.nodes = vec![output].into_boxed_slice(),
        };
        assert!(RotaryReuse66::new(&p.index, &plan, driver.legalization())
            .pairs
            .is_empty());
    }
}

#[test]
#[ignore = "requires CUDA GPU; run with EFFECT_TORCH_CUDA_ROTARY_REUSE66=1"]
fn rotary66_exact_inputs_cancel_errors_concurrent_retained() {
    use effect_torch_runtime::CancellationFlag;
    assert_eq!(
        std::env::var("EFFECT_TORCH_CUDA_ROTARY_REUSE66").as_deref(),
        Ok("1")
    );
    let device = crate::CudaDevice::get(0).unwrap();
    for rows in [64, 256] {
        for half in [128, 256] {
            for offset in [0, 4] {
                let mut f = fixture_width(rows, 2, half);
                for (slot, table) in f.roots[2..].to_vec().into_iter().enumerate() {
                    let view = Node::new(NodeKind::Reshape {
                        a: table,
                        shape: vec![1, 1, rows, 2 * half],
                    })
                    .unwrap();
                    f.roots.push(
                        Node::new(NodeKind::Expose {
                            a: view,
                            name: format!("retained.rotary66.{slot}"),
                        })
                        .unwrap(),
                    );
                }
                let optimized = Arc::new(
                    with_test_policy(true, || crate::compile(f.roots.clone(), 0)).unwrap(),
                );
                let reference =
                    with_test_policy(false, || crate::compile(f.roots.clone(), 0)).unwrap();
                assert_eq!(
                    optimized
                        .diagnostics()
                        .instructions
                        .iter()
                        .filter(|i| i.kind == "rotary_reuse66")
                        .map(|i| i.count)
                        .sum::<usize>(),
                    2
                );
                assert!(!reference
                    .diagnostics()
                    .instructions
                    .iter()
                    .any(|i| i.kind == "rotary_reuse66"));
                let mut retained = Vec::new();
                let mut workers = Vec::new();
                for mode in 0..12u32 {
                    let bindings = f
                        .inputs
                        .iter()
                        .enumerate()
                        .map(|(slot, n)| {
                            let mut bytes = vec![0xa5; offset];
                            for i in 0..n.shape.iter().product::<usize>() {
                                let h = (i as u32)
                                    .wrapping_mul(0x9e3779b9)
                                    .wrapping_add(mode.wrapping_mul(0x7feb352d));
                                let bits = if slot == 0 {
                                    match mode % 6 {
                                        0 => 256 + i as u32,
                                        1 => 0x1000000 - 16 + (i % 33) as u32,
                                        2 => u32::MAX - (i % 65) as u32,
                                        3 => 0,
                                        _ => h,
                                    }
                                } else {
                                    let special = [
                                        0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
                                        0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321, 0x7f800001,
                                        0xff800001, 0x7f7fffff, 0xff7fffff, 0x3f800000, 0xbf800000,
                                    ];
                                    match mode {
                                        0..=4 => {
                                            if half == 128 {
                                                CAPTURED_LOCAL66[i]
                                            } else {
                                                CAPTURED_GLOBAL66[i]
                                            }
                                        }
                                        5..=9 => special[(i + mode as usize) % special.len()],
                                        _ => h,
                                    }
                                };
                                bytes.extend_from_slice(&bits.to_le_bytes());
                            }
                            let owner = Arc::new(device.stream.clone_htod(&bytes).unwrap());
                            let buffer = crate::buffer::CudaBuffer::from_segment(
                                owner,
                                offset,
                                bytes.len() - offset,
                                None,
                            )
                            .unwrap();
                            crate::CudaValue::from_planned_buffer(
                                device.clone(),
                                effect_torch_runtime::ValueSpec::dense(n.dtype, &n.shape),
                                buffer,
                            )
                            .unwrap()
                        })
                        .collect::<Vec<_>>();
                    let expected = reference
                        .execute(&bindings, &[], &CancellationFlag::new())
                        .unwrap()
                        .into_iter()
                        .map(|v| v.read_storage_bytes().unwrap())
                        .collect::<Vec<_>>();
                    assert_eq!(expected.len(), f.roots.len());
                    let cancel = CancellationFlag::new();
                    cancel.cancel();
                    assert!(optimized.execute(&bindings, &[], &cancel).is_err());
                    assert!(optimized
                        .execute(&bindings[..1], &[], &CancellationFlag::new())
                        .is_err());
                    if mode == 0 {
                        let cancelled = CancellationFlag::new();
                        let reached = std::cell::Cell::new(false);
                        assert!(optimized
                            .execute_with_rotary66_hook(&bindings, &cancelled, &|| {
                                reached.set(true);
                                cancelled.cancel();
                            })
                            .is_err());
                        assert!(
                            reached.get(),
                            "both canonical table launches must precede this alias hook"
                        );
                    }
                    let actual = optimized
                        .execute(&bindings, &[], &CancellationFlag::new())
                        .unwrap();
                    assert_eq!(actual.len(), expected.len());
                    for (v, b) in actual.iter().zip(&expected) {
                        assert_eq!(
                            &v.read_storage_bytes().unwrap(),
                            b,
                            "rows={rows} half={half} offset={offset} mode={mode}"
                        );
                    }
                    if mode < 2 {
                        let exe = optimized.clone();
                        let bytes = expected.clone();
                        workers.push(std::thread::spawn(move || {
                            let output = exe
                                .execute(&bindings, &[], &CancellationFlag::new())
                                .unwrap();
                            assert_eq!(output.len(), bytes.len());
                            for (v, b) in output.iter().zip(bytes) {
                                assert_eq!(v.read_storage_bytes().unwrap(), b);
                            }
                        }));
                    }
                    retained.push((actual, expected));
                }
                for w in workers {
                    w.join().unwrap();
                }
                drop(optimized);
                drop(reference);
                for (values, bytes) in retained {
                    for (v, b) in values.iter().zip(bytes) {
                        assert_eq!(v.read_storage_bytes().unwrap(), b);
                    }
                }
            }
        }
    }
}

#[test]
fn rotary66_undominated_duplicate_and_observed_concat_decline() {
    for observed in [false, true] {
        let mut f = fixture(64, 2);
        if observed {
            let NodeKind::Cast { a: cos, .. } = &f.roots[2].kind else {
                panic!()
            };
            let NodeKind::Cos { a: doubled } = &cos.kind else {
                panic!()
            };
            f.roots.push(doubled.clone());
        } else {
            f.roots.swap(1, 2);
        }
        let p = prepared(&f);
        let caps = CudaCapabilities::new(0, 12, 0).with_rotary_reuse66(true);
        let driver = CompilerDriver::new(&p, &caps).unwrap();
        assert!(
            RotaryReuse66::new(&p.index, driver.optimization(), driver.legalization())
                .pairs
                .is_empty()
        );
    }
}
#[test]
fn rotary66_raw_dtype_and_half_width_contracts_decline() {
    for (position_dtype, frequency_dtype, half) in [
        (DType::I64, DType::F32, 128),
        (DType::U32, DType::BF16, 128),
        (DType::U32, DType::F32, 64),
    ] {
        let f = fixture_sources(
            64,
            2,
            input(0, &[1, 64], position_dtype),
            input(1, &[half], frequency_dtype),
        );
        let p = prepared(&f);
        let caps = CudaCapabilities::new(0, 12, 0).with_rotary_reuse66(true);
        let driver = CompilerDriver::new(&p, &caps).unwrap();
        assert!(
            RotaryReuse66::new(&p.index, driver.optimization(), driver.legalization())
                .pairs
                .is_empty()
        );
    }
}

// Exact initialized-rope.safetensors bytes; source SHA256
// 69c413815d1e62b529c88cc1d393b1242a2910a8237cce73f60290206fee9c60.
const CAPTURED_LOCAL66: [u32; 128] = [
    0x3f800000, 0x3f6e39f8, 0x3f5dafd7, 0x3f4e4bad, 0x3f3ff911, 0x3f32a506, 0x3f263de0, 0x3f1ab32b,
    0x3f0ff59a, 0x3f05f6ef, 0x3ef953cf, 0x3ee8045f, 0x3ed7e89b, 0x3ec8eb24, 0x3ebaf81b, 0x3eadfcff,
    0x3ea1e89b, 0x3e96aaea, 0x3e8c3504, 0x3e827909, 0x3e72d423, 0x3e61f835, 0x3e5247ed, 0x3e43ae7c,
    0x3e361887, 0x3e297409, 0x3e1db040, 0x3e12bd91, 0x3e088d77, 0x3dfe24e0, 0x3dec7fd6, 0x3ddc1466,
    0x3dcccccd, 0x3dbe94c6, 0x3db15978, 0x3da50956, 0x3d99940d, 0x3d8eea6b, 0x3d84fe4d, 0x3d778513,
    0x3d6655c2, 0x3d5657e4, 0x3d47763f, 0x3d399d19, 0x3d2cba15, 0x3d20bc1d, 0x3d159348, 0x3d0b30cc,
    0x3d0186e3, 0x3cf11177, 0x3ce054d2, 0x3cd0c1a8, 0x3cc2434f, 0x3cb4c691, 0x3ca8398b, 0x3c9c8b97,
    0x3c91ad39, 0x3c879008, 0x3c7c4d33, 0x3c6ac8e7, 0x3c5a7bf2, 0x3c4b50b3, 0x3c3d3311, 0x3c301052,
    0x3c23d70a, 0x3c187705, 0x3c0de12d, 0x3c040779, 0x3bf5b9b0, 0x3be4aa46, 0x3bd4ca15, 0x3bc6040f,
    0x3bb8449c, 0x3bab7983, 0x3b9f91cc, 0x3b947dae, 0x3b8a2e77, 0x3b80967d, 0x3b6f520e, 0x3b5eb47a,
    0x3b4f3e38, 0x3b40dac5, 0x3b33770f, 0x3b270153, 0x3b1b690d, 0x3b109edb, 0x3b06946f, 0x3afa78f0,
    0x3ae91528, 0x3ad8e673, 0x3ac9d75c, 0x3abbd3ed, 0x3aaec98e, 0x3aa2a6f7, 0x3a975c0e, 0x3a8cd9db,
    0x3a83126f, 0x3a73f1a3, 0x3a6301e2, 0x3a533f28, 0x3a44948c, 0x3a36ee9e, 0x3a2a3b44, 0x3a1e69a5,
    0x3a136a16, 0x3a092e02, 0x39ff4fac, 0x39ed95e2, 0x39dd1725, 0x39cdbd96, 0x39bf74d7, 0x39b229fb,
    0x39a5cb60, 0x399a489e, 0x398f9272, 0x39859aa9, 0x3978a815, 0x39676492, 0x395753e4, 0x394860c1,
    0x393a7753, 0x392d8529, 0x39217916, 0x39164324, 0x390bd472, 0x39021f2b, 0x38f22ce2, 0x38e15c91,
];

// Exact initialized-rope.safetensors bytes; source SHA256
// 69c413815d1e62b529c88cc1d393b1242a2910a8237cce73f60290206fee9c60.
const CAPTURED_GLOBAL66: [u32; 256] = [
    0x3f800000, 0x3f728cf8, 0x3f65ced3, 0x3f59bc10, 0x3f4e4bad, 0x3f437523, 0x3f39305c, 0x3f2f75b1,
    0x3f263de0, 0x3f1d8209, 0x3f153ba8, 0x3f0d6492, 0x3f05f6ef, 0x3efdda64, 0x3ef0843c, 0x3ee3e172,
    0x3ed7e89b, 0x3ecc90c7, 0x3ec1d182, 0x3eb7a2c7, 0x3eadfcff, 0x3ea4d8f8, 0x3e9c2fe1, 0x3e93fb45,
    0x3e8c3504, 0x3e84d752, 0x3e7bb964, 0x3e6e7fde, 0x3e61f835, 0x3e561912, 0x3e4ad998, 0x3e403165,
    0x3e361887, 0x3e2c8776, 0x3e23770f, 0x3e1ae090, 0x3e12bd91, 0x3e0b0801, 0x3e03ba20, 0x3df99cf8,
    0x3dec7fd6, 0x3de01313, 0x3dd44d6c, 0x3dc92617, 0x3dbe94c6, 0x3db49196, 0x3dab150f, 0x3da2181d,
    0x3d99940d, 0x3d918287, 0x3d89dd84, 0x3d829f52, 0x3d778513, 0x3d6a8417, 0x3d5e3201, 0x3d5285a1,
    0x3d47763f, 0x3d3cfb9e, 0x3d330dec, 0x3d29a5c2, 0x3d20bc1d, 0x3d184a56, 0x3d104a21, 0x3d08b588,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0,
];

#[test]
fn rotary66_captured_frequency_fixture_provenance() {
    use sha2::{Digest, Sha256};
    for (bits, expected) in [
        (
            CAPTURED_LOCAL66.as_slice(),
            "cc63341a0ac42a60b986ed638fd0d45b838b72fabeffec059c463eac4ed9ea15",
        ),
        (
            CAPTURED_GLOBAL66.as_slice(),
            "71a80d617056a8be95eeeb385ebcc01c7fb2c73d5430a8ce862ed6c94179d076",
        ),
    ] {
        let bytes = bits
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), expected);
    }
    assert_eq!(CAPTURED_GLOBAL66.iter().filter(|&&b| b == 0).count(), 192);
}

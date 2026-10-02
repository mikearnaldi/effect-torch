use super::*;
use effect_torch_graph::Device;
use effect_torch_runtime::{Location, StorageMetadata};

#[test]
fn graph_address_scratch_discards_previous_attempts_and_wraps() {
    let mut scratch = GraphAddressScratch::new(2);
    let value = GraphValueAddress {
        address: 123,
        dtype: DType::BF16,
        representation: StorageRepresentation::Dense,
        bytes: 2,
    };
    scratch.begin();
    scratch.insert(ValueId::new(0), value);
    assert_eq!(scratch.get(ValueId::new(0)).unwrap().address, 123);
    scratch.begin();
    assert!(scratch.get(ValueId::new(0)).is_none());
    scratch.insert(ValueId::new(1), value);
    scratch.generation = u64::MAX;
    scratch.begin();
    assert_eq!(scratch.generation, 1);
    assert!(scratch.get(ValueId::new(0)).is_none());
    assert!(scratch.get(ValueId::new(1)).is_none());
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

#[test]
#[ignore = "requires CUDA"]
fn planned_graph_keys_equal_materialized_keys_with_aliases_bindings_constants_and_workspaces() {
    let x = input(0, &[4, 8]);
    let weight = input(1, &[8, 8]);
    let product = Node::new(NodeKind::Matmul { a: x, b: weight }).unwrap();
    let reshaped = Node::new(NodeKind::Reshape {
        a: product.clone(),
        shape: vec![2, 16],
    })
    .unwrap();
    let sliced = Node::new(NodeKind::Slice {
        a: reshaped.clone(),
        ranges: vec![(1, 2, 1), (0, 16, 1)],
    })
    .unwrap();
    let output = Node::new(NodeKind::RmsNorm {
        x: sliced.clone(),
        weight: None,
        eps: 1e-6,
    })
    .unwrap();
    let mut executable = compile_with_options(
        vec![product, reshaped, sliced, output],
        0,
        CompileOptions {
            optimize: false,
            ..Default::default()
        },
    )
    .unwrap();
    let device = executable.device.clone();
    let bindings = |factor| {
        vec![
            CudaValue::from_host(
                device.clone(),
                vec![4, 8],
                DType::BF16,
                &(0..32).map(|i| i as f64 * factor).collect::<Vec<_>>(),
            )
            .unwrap(),
            CudaValue::from_host(
                device.clone(),
                vec![8, 8],
                DType::BF16,
                &(0..64).map(|i| i as f64 / 64.).collect::<Vec<_>>(),
            )
            .unwrap(),
        ]
    };
    let first = bindings(0.25);
    let second = bindings(0.5);
    assert_ne!(first[0].storage_address(), second[0].storage_address());
    assert!(executable
        .commands
        .iter()
        .any(|c| matches!(c.kind, CommandKind::Gemm { .. })));
    assert!(executable
        .commands
        .iter()
        .any(|c| matches!(c.kind, CommandKind::Alias { .. })));
    // Make a nonzero planned alias explicit even when this lowering chooses a
    // slice kernel. Key construction must honor the memory plan's byte offset.
    let root = executable
        .commands
        .iter()
        .find_map(|c| {
            matches!(c.kind, CommandKind::Gemm { .. })
                .then_some(c.output)
                .flatten()
        })
        .unwrap();
    let alias = executable.commands.last().unwrap().output.unwrap();
    let alias_bytes = executable.program.values[alias.index()].decl.bytes;
    assert!(executable.program.values[root.index()].decl.bytes >= alias_bytes + 16);
    executable.memory.locations[alias.index()] = Location::Alias {
        root,
        byte_offset: 16,
    };
    executable.commands.last_mut().unwrap().kind = CommandKind::PlannedAlias;
    let resources = workspace::acquire(0, &executable.memory.segments).unwrap();
    let resources_other = workspace::acquire(0, &executable.memory.segments).unwrap();
    let mut scratch = GraphAddressScratch::new(executable.program.values.len());
    let check = |bindings: &[CudaValue],
                 resources: &InvocationResources,
                 scratch: &mut GraphAddressScratch| {
        let mut values = vec![None; executable.program.values.len()];
        let end = executable.commands.len();
        let actual = executable
            .planned_graph_key(resources, bindings, &values, scratch, 0, end)
            .unwrap();
        executable
            .materialize_graph_values(resources, bindings, &mut values, 0, end)
            .unwrap();
        assert_eq!(
            actual,
            executable.graph_key(resources, &values, 0, end).unwrap()
        );
        assert_eq!(
            values[alias.index()].as_ref().unwrap().storage_address(),
            executable.buffer(resources, root).unwrap().address() + 16
        );
        actual
    };
    let original = check(&first, &resources, &mut scratch);
    assert_ne!(original, check(&second, &resources, &mut scratch));
    assert_ne!(original, check(&first, &resources_other, &mut scratch));
    // Materialize an explicit constant with the same declaration and pointer.
    let input_command = executable
        .commands
        .iter_mut()
        .find(|c| matches!(c.kind, CommandKind::Input { binding: 1 }))
        .unwrap();
    input_command.kind = CommandKind::Value(first[1].clone());
    let mut values = vec![None; executable.program.values.len()];
    let end = executable.commands.len();
    let actual = executable
        .planned_graph_key(&resources, &first, &values, &mut scratch, 0, end)
        .unwrap();
    executable
        .materialize_graph_values(&resources, &first, &mut values, 0, end)
        .unwrap();
    assert_eq!(
        actual,
        executable.graph_key(&resources, &values, 0, end).unwrap()
    );
    assert_eq!(original, actual);
    let missing = executable
        .planned_graph_key(&resources, &[], &[], &mut scratch, 0, end)
        .unwrap_err();
    let baseline_missing = executable
        .materialize_graph_values(&resources, &[], &mut values, 0, end)
        .unwrap_err();
    assert_eq!(missing, baseline_missing);
    let invalid = vec![first[1].clone()];
    assert_eq!(
        executable
            .planned_graph_key(&resources, &invalid, &values, &mut scratch, 0, end)
            .unwrap_err(),
        executable
            .materialize_graph_values(&resources, &invalid, &mut values, 0, end)
            .unwrap_err()
    );
}

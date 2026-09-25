//! Exact CUDA half-rotation reindexing.
use crate::{CudaDevice, CudaValue};
use effect_torch_graph::{Device, Node, NodeKind};
use effect_torch_runtime::{CancellationFlag, DType, StorageMetadata};
use std::sync::Arc;

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

fn host(shape: &[usize], dtype: DType, data: &[f64]) -> CudaValue {
    CudaValue::from_host(CudaDevice::get(0).unwrap(), shape.to_vec(), dtype, data).unwrap()
}

#[test]
#[ignore = "requires a CUDA device"]
fn rotary_half_reindex_matches_materialized_order_exactly() {
    let shape = [2, 3, 8];
    let row = [0.0, -0.0, 1.0, -2.0, 3.5, -4.5, 5.0, -6.0];
    let expected_row = [-3.5, 4.5, -5.0, 6.0, 0.0, -0.0, 1.0, -2.0];
    let source = row.repeat(6);
    let expected = expected_row.repeat(6);

    for dtype in [DType::F64, DType::F32, DType::F16, DType::BF16] {
        let x = input(0, &shape, dtype);
        let first = Node::new(NodeKind::Slice {
            a: x.clone(),
            ranges: vec![(0, 2, 1), (0, 3, 1), (0, 4, 1)],
        })
        .unwrap();
        let second = Node::new(NodeKind::Slice {
            a: x,
            ranges: vec![(0, 2, 1), (0, 3, 1), (4, 8, 1)],
        })
        .unwrap();
        let negative = Node::new(NodeKind::Neg { a: second }).unwrap();
        let root = Node::new(NodeKind::Concat {
            a: negative,
            b: first,
            dim: 2,
        })
        .unwrap();
        let executable = crate::compile(vec![root], 0).unwrap();
        let binding = host(&shape, dtype, &source);
        let original = binding.read_storage_bytes().unwrap();
        let actual = executable
            .execute(&[binding.clone()], &[], &CancellationFlag::new())
            .unwrap()[0]
            .read_storage_bytes()
            .unwrap();
        let expected = host(&shape, dtype, &expected).read_storage_bytes().unwrap();

        assert_eq!(actual, expected, "{dtype:?}");
        assert_eq!(binding.read_storage_bytes().unwrap(), original, "{dtype:?}");
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn repeated_angles_match_materialized_cosine_exactly() {
    let shape = [2, 3, 4];
    let values = [
        -3.25, -1.5, -0.0, 0.125, 0.5, 1.0, 2.0, 4.0, -2.5, -0.75, 0.25, 3.0, -4.0, -2.0, -1.0,
        -0.5, 0.0, 0.75, 1.5, 2.5, -3.0, -0.25, 1.25, 3.5,
    ];
    let root = |repeated: bool| {
        let first = input(0, &shape, DType::F32);
        let second = if repeated {
            first.clone()
        } else {
            input(1, &shape, DType::F32)
        };
        let duplicated = Node::new(NodeKind::Concat {
            a: first,
            b: second,
            dim: 2,
        })
        .unwrap();
        let cosine = Node::new(NodeKind::Cos { a: duplicated }).unwrap();
        Node::new(NodeKind::Cast {
            a: cosine,
            dtype: DType::BF16,
        })
        .unwrap()
    };
    let repeated = crate::compile(vec![root(true)], 0).unwrap();
    let materialized = crate::compile(vec![root(false)], 0).unwrap();
    let binding = host(&shape, DType::F32, &values);
    let original = binding.read_storage_bytes().unwrap();
    let actual = repeated
        .execute(&[binding.clone()], &[], &CancellationFlag::new())
        .unwrap()[0]
        .read_storage_bytes()
        .unwrap();
    let expected = materialized
        .execute(
            &[binding.clone(), binding.clone()],
            &[],
            &CancellationFlag::new(),
        )
        .unwrap()[0]
        .read_storage_bytes()
        .unwrap();

    assert_eq!(actual, expected);
    assert_eq!(binding.read_storage_bytes().unwrap(), original);
}

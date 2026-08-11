mod decode;
mod diagnostics;
mod driver;
mod ir;
mod lowered;
mod optimization;
mod planner;
mod request;
mod schedule;

pub use decode::*;
pub use diagnostics::*;
pub use driver::*;
pub use ir::*;
pub use lowered::*;
pub use optimization::*;
pub use planner::*;
pub use request::*;
pub use schedule::{
    DenseNodeId, GeneratedBinding, GraphIndex, GraphIndexWork, ProgramSlot, RandomSource,
    RandomSourceId, RandomSourceKind, RandomSourceMetadata,
};

#[cfg(test)]
mod tests {
    use super::*;
    use effect_torch_graph::{Device, Node, NodeKind};
    use effect_torch_runtime::DType;
    use std::sync::Arc;

    #[test]
    fn optimization_plan_preserves_semantic_node_identity() {
        let x = Node::new(NodeKind::Input {
            slot: 0,
            shape: vec![4],
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap();
        let y = Node::new(NodeKind::Input {
            slot: 1,
            shape: vec![4],
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap();
        let sum = Node::new(NodeKind::Add { a: x, b: y }).unwrap();
        let root = Node::new(NodeKind::Tanh { a: sum }).unwrap();
        let index = GraphIndex::new(std::slice::from_ref(&root)).unwrap();
        let identities = index
            .order
            .iter()
            .map(|node| (node.id, Arc::as_ptr(node) as usize))
            .collect::<Vec<_>>();

        let plan = OptimizationPlan::select(&index, &CompileOptions::default()).unwrap();

        assert!(matches!(plan.regions[0], NativeRegion::Elementwise(_)));
        assert_eq!(plan.work.semantic_nodes_rebuilt, 0);
        assert_eq!(
            identities,
            index
                .order
                .iter()
                .map(|node| (node.id, Arc::as_ptr(node) as usize))
                .collect::<Vec<_>>()
        );
        assert!(matches!(
            plan.resolve(index.dense_id(root.id).unwrap()),
            Ok(ValueSource::Region(_))
        ));
    }

    #[test]
    fn optimization_plan_is_stack_safe_for_deep_graphs() {
        std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                let leaf = Node::new(NodeKind::Input {
                    slot: 0,
                    shape: vec![1],
                    dtype: DType::F32,
                    device: Device::Cpu,
                })
                .unwrap();
                let mut root = leaf;
                for _ in 0..50_000 {
                    root = Node::new(NodeKind::Neg { a: root }).unwrap();
                }
                let index = GraphIndex::new(std::slice::from_ref(&root)).unwrap();
                let plan = OptimizationPlan::select(&index, &CompileOptions::default()).unwrap();
                assert_eq!(index.order.len(), 50_001);
                assert_eq!(plan.regions.len(), 1);
                assert_eq!(plan.work.semantic_nodes_rebuilt, 0);
                assert!(matches!(
                    plan.resolve(index.dense_id(root.id).unwrap()),
                    Ok(ValueSource::Region(_))
                ));
            })
            .unwrap()
            .join()
            .unwrap();
    }
}

//! Backend-neutral graph compiler: from a semantic `Node` graph to a
//! validated, memory-planned program ready for backend execution.
//!
//! Compilation proceeds through a fixed sequence of stages, each producing
//! an immutable artifact consumed by the next:
//!
//! 1. **Request/preparation** (`request`, `driver`): a [`ProgramRequest`]
//!    is validated (non-empty roots, single device, sound options) and its
//!    invocation contract — bindings, scalars, outputs — is either taken
//!    from the caller or derived from the graph.
//! 2. **Graph indexing** (`schedule`): the semantic graph is discovered
//!    iteratively into a [`GraphIndex`], a dense postorder with child/consumer
//!    adjacency, slot declarations, and random-source metadata. Exactly one
//!    index is built per program generation.
//! 3. **Optimization** (`optimization`): region selection fuses eligible
//!    nodes into codegen regions (elementwise, fused-reduce, GEMM epilogues,
//!    optimizer steps, multi-output merges) without ever rebuilding a
//!    semantic node, and emits a deterministic topological lowering order.
//! 4. **Decode specialization** (`decode`, optional, ahead of indexing):
//!    training-time ops are rewritten into their stateful inference
//!    counterparts (KV attention, KDA recurrence, conv state) with the
//!    geometry contract the runtime must honor.
//! 5. **Lowering** (`lowered`): backends translate lowering units into a
//!    dense [`LoweredProgram`] of instructions and value declarations.
//! 6. **Memory planning** (`planner`): liveness analysis and alias
//!    normalization drive a deterministic best-fit segment packer that
//!    guarantees overlapping live intervals never share address space and
//!    records reuse edges for the runtime.
//! 7. **Diagnostics** (`diagnostics`): structural work counters and phase
//!    timings are assembled into reports; timings are observational only and
//!    never part of cache identity.
//!
//! The elementwise fusion IR itself lives in `ir`. Throughout, all
//! traversals are iterative so compiler depth is bounded by heap, not stack,
//! and every stage is deterministic: identical inputs yield identical
//! artifacts.

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

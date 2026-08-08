mod ir;
mod rewrite;
mod schedule;

pub use ir::*;
pub use rewrite::{fuse_roots, gemm_epilogue_pass};
pub use schedule::{collect_program_slots, ProgramSlot};

#[cfg(test)]
mod tests {
    use super::*;
    use effect_torch_graph::{Device, Node, NodeKind};
    use effect_torch_runtime::DType;

    #[test]
    fn neutral_ir_is_interpreted_by_the_cpu_compiler_path() {
        let expr = Expr::Add(
            Box::new(Expr::Input(0)),
            Box::new(Expr::Mul(
                Box::new(Expr::Input(1)),
                Box::new(Expr::cst(2.0)),
            )),
        );
        let a = [1.0f32, 2.0];
        let b = [3.0f32, 4.0];
        let output = interpret_core(&[expr], &[&a, &b], None, &[], 2, &[2]);
        assert_eq!(output, [vec![7.0, 10.0]]);
    }

    #[test]
    fn production_graph_rewrite_builds_a_fused_region() {
        let x = Node::<Expr>::new(NodeKind::Input {
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
        let fused = fuse_roots(&[root]).unwrap();
        assert!(matches!(fused[0].kind, NodeKind::FusedElementwise { .. }));
    }

    // Long elementwise chains fuse into one deep expression region; every
    // walk over it (evaluate, clone, compare, hash, count, drop) must be
    // stack-safe. This test lives on a 256 KiB stack to prove it.
    #[test]
    fn deep_fused_regions_never_recurse() {
        std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                let mut expr = Expr::Input(0);
                for _ in 0..100_000 {
                    expr = Expr::Add(Box::new(expr), Box::new(Expr::Input(1)));
                }
                assert_eq!(expr.ops(), 200_001);
                let cloned = expr.clone();
                assert!(cloned == expr);
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                std::hash::Hash::hash(&expr, &mut hasher);
                let a = [1.0f32];
                let b = [2.0f32];
                let out = interpret_core(&[expr], &[&a, &b], None, &[], 1, &[1]);
                assert_eq!(out, [vec![200_001.0f32]]);
                // The 100k-deep expression drops at scope end on this
                // small-stack thread.
            })
            .unwrap()
            .join()
            .unwrap();
    }

    // The same depth discipline for whole graphs: fusion's traversals and
    // the node destructors are iterative.
    #[test]
    fn deep_graphs_fuse_and_drop_without_recursion() {
        std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                let leaf = || {
                    Node::<Expr>::new(NodeKind::Input {
                        slot: 0,
                        shape: vec![1],
                        dtype: DType::F32,
                        device: Device::Cpu,
                    })
                    .unwrap()
                };
                let mut node = leaf();
                for _ in 0..50_000 {
                    node = Node::new(NodeKind::Add { a: node, b: leaf() }).unwrap();
                }
                let fused = fuse_roots(&[node]).unwrap();
                assert!(matches!(fused[0].kind, NodeKind::FusedElementwise { .. }));
                drop(fused);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn frozen_program_planner_uses_real_placeholder_nodes() {
        let input = Node::<Expr>::new(NodeKind::Input {
            slot: 0,
            shape: vec![2, 3],
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap();
        let scalar = Node::new(NodeKind::ScalarInput {
            slot: 1,
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap();
        let root = Node::new(NodeKind::Mul {
            a: input,
            b: scalar,
        })
        .unwrap();
        let (slots, leaves) = collect_program_slots(&[root]).unwrap();
        assert_eq!(slots.len(), 2);
        assert_eq!(slots[0].signature(), "2x3:f32@cpu");
        assert_eq!(slots[1].signature(), "scalar:f32@cpu");
        assert_eq!(leaves.len(), 2);
    }
}

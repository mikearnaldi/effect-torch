use crate::Expr;
use effect_torch_graph::{node_children, Device, Node as GraphNode, NodeKind as GraphNodeKind};
use effect_torch_runtime::DType;
use std::collections::HashSet;
use std::sync::Arc;

type Node = GraphNode<Expr>;
type NodeKind = GraphNodeKind<Expr>;

/// Deterministic, stack-safe postorder over roots and children in caller order.
/// Shared subgraphs occur once; hash tables are used only for membership.
pub fn graph_post_order(roots: &[Arc<Node>]) -> Vec<Arc<Node>> {
    let mut visited = HashSet::new();
    let mut order = Vec::new();
    let mut stack: Vec<(Arc<Node>, bool)> = roots
        .iter()
        .rev()
        .map(|root| (root.clone(), false))
        .collect();
    while let Some((node, processed)) = stack.pop() {
        if processed {
            order.push(node);
            continue;
        }
        if !visited.insert(node.id) {
            continue;
        }
        stack.push((node.clone(), true));
        for child in node_children(&node.kind).into_iter().rev() {
            stack.push((child, false));
        }
    }
    order
}

#[derive(Clone)]
pub struct ProgramSlot {
    pub scalar: bool,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub device: Device,
}

impl ProgramSlot {
    pub fn signature(&self) -> String {
        let shape = if self.scalar {
            "scalar".to_string()
        } else {
            self.shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x")
        };
        format!("{}:{}@{}", shape, self.dtype.name(), self.device.name())
    }
}
pub fn collect_program_slots(
    roots: &[Arc<Node>],
) -> std::result::Result<(Vec<ProgramSlot>, Vec<(u64, u32)>), String> {
    let mut slots: Vec<Option<ProgramSlot>> = Vec::new();
    let mut leaves: Vec<(u64, u32)> = Vec::new();
    for node in graph_post_order(roots) {
        let declared = match &node.kind {
            NodeKind::Input {
                slot,
                shape,
                dtype,
                device,
            } => Some((
                *slot,
                ProgramSlot {
                    scalar: false,
                    shape: shape.clone(),
                    dtype: *dtype,
                    device: device.clone(),
                },
            )),
            NodeKind::ScalarInput {
                slot,
                dtype,
                device,
            } => Some((
                *slot,
                ProgramSlot {
                    scalar: true,
                    shape: vec![],
                    dtype: *dtype,
                    device: device.clone(),
                },
            )),
            _ => None,
        };
        if let Some((slot, declared)) = declared {
            leaves.push((node.id, slot));
            let slot = slot as usize;
            if slot >= slots.len() {
                slots.resize_with(slot + 1, || None);
            }
            match &slots[slot] {
                Some(existing) => {
                    if existing.scalar != declared.scalar
                        || existing.shape != declared.shape
                        || existing.dtype != declared.dtype
                        || existing.device != declared.device
                    {
                        return Err(format!(
                            "compile: slot {slot} is used with conflicting signatures ({} vs {})",
                            existing.signature(),
                            declared.signature()
                        ));
                    }
                }
                None => slots[slot] = Some(declared),
            }
        }
    }
    let mut out = Vec::with_capacity(slots.len());
    for (slot, declared) in slots.into_iter().enumerate() {
        out.push(
            declared.ok_or_else(|| format!("compile: slot {slot} is declared but never used"))?,
        );
    }
    Ok((out, leaves))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(slot: u32) -> Arc<Node> {
        Node::new(NodeKind::Input {
            slot,
            shape: vec![1],
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap()
    }

    #[test]
    fn multi_root_postorder_preserves_root_and_child_order_with_deduplication() {
        let x = input(0);
        let y = input(1);
        let shared = Node::new(NodeKind::Add {
            a: x.clone(),
            b: y.clone(),
        })
        .unwrap();
        let left = Node::new(NodeKind::Neg { a: shared.clone() }).unwrap();
        let right = Node::new(NodeKind::Tanh { a: shared.clone() }).unwrap();
        let order = graph_post_order(&[left.clone(), right.clone(), left.clone()]);
        assert_eq!(
            order.iter().map(|node| node.id).collect::<Vec<_>>(),
            [x.id, y.id, shared.id, left.id, right.id]
        );
    }

    #[test]
    fn postorder_is_stack_safe_for_deep_graphs() {
        std::thread::Builder::new()
            .stack_size(128 * 1024)
            .spawn(|| {
                let leaf = input(0);
                let leaf_id = leaf.id;
                let mut root = leaf;
                for _ in 0..50_000 {
                    root = Node::new(NodeKind::Neg { a: root }).unwrap();
                }
                let root_id = root.id;
                let order = graph_post_order(&[root]);
                assert_eq!(order.len(), 50_001);
                assert_eq!(order.first().unwrap().id, leaf_id);
                assert_eq!(order.last().unwrap().id, root_id);
            })
            .unwrap()
            .join()
            .unwrap();
    }
}

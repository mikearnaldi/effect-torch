use crate::Expr;
use effect_torch_graph::{node_children, Device, Node as GraphNode, NodeKind as GraphNodeKind};
use effect_torch_runtime::DType;
use std::collections::HashSet;
use std::sync::Arc;

type Node = GraphNode<Expr>;
type NodeKind = GraphNodeKind<Expr>;

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
    let mut visited = HashSet::new();
    let mut stack: Vec<Arc<Node>> = roots.to_vec();
    while let Some(node) = stack.pop() {
        if !visited.insert(node.id) {
            continue;
        }
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
        stack.extend(node_children(&node.kind));
    }
    let mut out = Vec::with_capacity(slots.len());
    for (slot, declared) in slots.into_iter().enumerate() {
        out.push(
            declared.ok_or_else(|| format!("compile: slot {slot} is declared but never used"))?,
        );
    }
    Ok((out, leaves))
}

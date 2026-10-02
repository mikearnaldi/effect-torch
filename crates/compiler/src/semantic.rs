//! Mandatory native preparation after decode specialization.

use effect_torch_graph::{decompose_semantic, node_children, remap_children, Node};
use std::collections::HashMap;
use std::sync::Arc;

/// Retains source graphs and lowers configured ordinary operations into exact
/// same-device graphs. This is independent of optional optimization. Unchanged
/// nodes, including all caller bindings and random sources, retain identity.
pub(crate) fn prepare_semantics(roots: &[Arc<Node>]) -> Result<Vec<Arc<Node>>, String> {
    let mut mapped = HashMap::<u64, Arc<Node>>::new();
    let mut stack = roots
        .iter()
        .rev()
        .map(|root| (root.clone(), false))
        .collect::<Vec<_>>();
    while let Some((node, ready)) = stack.pop() {
        if mapped.contains_key(&node.id) {
            continue;
        }
        let children = node_children(&node.kind);
        if !ready {
            stack.push((node.clone(), true));
            for child in children.into_iter().rev() {
                if !mapped.contains_key(&child.id) {
                    stack.push((child, false));
                }
            }
            continue;
        }
        let changed = children
            .iter()
            .any(|child| !Arc::ptr_eq(child, &mapped[&child.id]));
        let current = if changed {
            Node::new(remap_children(&node.kind, &|child| {
                mapped[&child.id].clone()
            }))?
        } else {
            node.clone()
        };
        let prepared = decompose_semantic(&current.kind)?.unwrap_or(current);
        mapped.insert(node.id, prepared);
    }
    Ok(roots.iter().map(|root| mapped[&root.id].clone()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompileOptions, ProgramRequest};
    use effect_torch_graph::{AttentionRounding, AttentionWindow, Device, NodeKind};
    use effect_torch_runtime::{DType, StorageMetadata};

    #[test]
    fn preparation_preserves_source_identity_metadata_and_duplicate_roots() {
        let q = Node::new(NodeKind::Input {
            slot: 0,
            shape: vec![1, 2, 4],
            dtype: DType::BF16,
            device: Device::Cpu(0),
            storage: StorageMetadata::dense(),
        })
        .unwrap();
        let source = Node::new(NodeKind::SdpaConfigured {
            q: q.clone(),
            k: q.clone(),
            v: q.clone(),
            scale: 0.3,
            causal: true,
            window: AttentionWindow::Local(1),
            rounding: AttentionRounding::Stepwise,
            layer_id: Some(19),
            retention: AttentionWindow::Local(0),
        })
        .unwrap();
        for optimize in [false, true] {
            let prepared = ProgramRequest::from_roots(
                vec![source.clone(), source.clone()],
                CompileOptions {
                    optimize,
                    ..Default::default()
                },
            )
            .prepare()
            .unwrap();
            assert!(Arc::ptr_eq(&prepared.source_roots[0], &source));
            assert!(Arc::ptr_eq(&prepared.roots[0], &prepared.roots[1]));
            assert_eq!(prepared.roots[0].shape, source.shape);
            assert_eq!(prepared.roots[0].dtype, source.dtype);
            assert!(prepared.index.dense_id(q.id).is_some());
            assert!(!prepared
                .index
                .order
                .iter()
                .any(|node| matches!(node.kind, NodeKind::SdpaConfigured { .. })));
            assert!(matches!(
                source.kind,
                NodeKind::SdpaConfigured {
                    rounding: AttentionRounding::Stepwise,
                    layer_id: Some(19),
                    retention: AttentionWindow::Local(0),
                    ..
                }
            ));
        }
    }
}

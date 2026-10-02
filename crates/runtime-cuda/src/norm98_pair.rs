//! Compile-time proof that the entrance residual is private to one matched tail.
//! Physical admission remains the lowerer's responsibility: replace the earlier
//! entrance only after the matched tail's existing dense preflight succeeds.
use effect_torch_compiler::{
    AttentionFfnEntranceRegion, DenseNodeId, FfnNextNormRegion, GraphIndex, NativeRegion,
    OptimizationPlan,
};
use effect_torch_graph::NodeKind;
use effect_torch_runtime::{DType, StorageRepresentation};
use std::collections::{HashSet, VecDeque};

fn extent(index: &GraphIndex, id: DenseNodeId, dtype: DType, elements: usize) -> bool {
    index.node(id).is_some_and(|node| {
        node.dtype == dtype
            && node
                .shape
                .iter()
                .try_fold(1_usize, |n, &d| n.checked_mul(d))
                == Some(elements)
            && index
                .value_storage
                .get(id.index())
                .is_some_and(|storage| storage.representation == StorageRepresentation::Dense)
    })
}

fn reshape_source(index: &GraphIndex, mut id: DenseNodeId) -> Option<DenseNodeId> {
    loop {
        let node = index.node(id)?;
        match &node.kind {
            NodeKind::Reshape { a, .. } => {
                let source = index.dense_id(a.id)?;
                if node.dtype != a.dtype
                    || node
                        .shape
                        .iter()
                        .try_fold(1_usize, |n, &d| n.checked_mul(d))
                        != a.shape.iter().try_fold(1_usize, |n, &d| n.checked_mul(d))
                {
                    return None;
                }
                id = source;
            }
            _ => return Some(id),
        }
    }
}

fn rho(index: &GraphIndex, id: DenseNodeId) -> Option<f32> {
    if !extent(index, id, DType::F32, 1) {
        return None;
    }
    let source = index.node(reshape_source(index, id)?)?;
    let NodeKind::Full {
        shape,
        dtype: DType::F32,
        value,
        ..
    } = &source.kind
    else {
        return None;
    };
    if !shape.is_empty() || !value.is_finite() {
        return None;
    }
    let value = *value as f32;
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + (bits >> 16 & 1)) >> 16;
    (value.is_finite() && rounded & 0x7f80 != 0x7f80).then_some(value)
}

fn private_residual(
    index: &GraphIndex,
    entrance: &AttentionFfnEntranceRegion,
    tail: &FfnNextNormRegion,
) -> bool {
    let residual = entrance.outputs[0];
    let elements = entrance.rows * 2816;
    if reshape_source(index, tail.inputs[2]) != Some(residual) {
        return false;
    }
    let allowed = entrance
        .nodes
        .iter()
        .chain(tail.nodes.iter())
        .copied()
        .collect::<HashSet<_>>();
    let mut seen = HashSet::new();
    let mut pending = VecDeque::from([residual]);
    while let Some(id) = pending.pop_front() {
        if !seen.insert(id) {
            continue;
        }
        if index.roots.contains(&id) || !extent(index, id, DType::BF16, elements) {
            return false;
        }
        let Some(consumers) = index.consumers_of(id) else {
            return false;
        };
        for &consumer in consumers {
            let Some(node) = index.node(consumer) else {
                return false;
            };
            // Walk every reshape, including aliases outside selected regions,
            // so a root or another consumer behind an alias cannot escape proof.
            if matches!(node.kind, NodeKind::Reshape { .. }) {
                if reshape_source(index, consumer) != Some(residual) {
                    return false;
                }
                pending.push_back(consumer);
            } else if !allowed.contains(&consumer) {
                return false;
            }
        }
    }
    entrance.residual_views.iter().all(|id| seen.contains(id)) && seen.contains(&tail.inputs[2])
}

fn find_in_regions(
    index: &GraphIndex,
    regions: &[NativeRegion],
    entrance: &AttentionFfnEntranceRegion,
) -> Option<(DenseNodeId, f32)> {
    if !matches!(entrance.rows, 64 | 256) || entrance.inputs.len() != 7 {
        return None;
    }
    let elements = entrance.rows.checked_mul(2816)?;
    if !entrance
        .outputs
        .iter()
        .chain(entrance.residual_views.iter())
        .all(|&id| {
            extent(index, id, DType::BF16, elements)
                && index
                    .node(id)
                    .is_some_and(|node| node.shape.last() == Some(&2816))
        })
    {
        return None;
    }
    let rho = rho(index, entrance.inputs[6])?;
    let mut result = None;
    for region in regions {
        let NativeRegion::FfnNextNorm(tail) = region else {
            continue;
        };
        if tail.rows != entrance.rows
            || tail.inputs.len() != 8
            || !tail
                .outputs
                .iter()
                .chain(tail.residual_views.iter())
                .all(|&id| {
                    extent(index, id, DType::BF16, elements)
                        && index
                            .node(id)
                            .is_some_and(|node| node.shape.last() == Some(&2816))
                })
            || !private_residual(index, entrance, tail)
        {
            continue;
        }
        if result.is_some() {
            return None;
        }
        result = Some((tail.outputs[0], rho));
    }
    result
}

pub(crate) fn find_pair(
    index: &GraphIndex,
    optimization: &OptimizationPlan,
    entrance: &AttentionFfnEntranceRegion,
) -> Option<(DenseNodeId, f32)> {
    find_in_regions(index, &optimization.regions, entrance)
}

#[cfg(test)]
#[path = "norm98_pair_tests.rs"]
mod tests;

//! Graph-level program transformations: reverse-mode automatic
//! differentiation ([`grad`]) and vectorization ([`vmap`]) over the semantic
//! `Node` graph.
//!
//! Both transforms consume an immutable graph and produce a *new* graph
//! generation built from ordinary `NodeKind`s; the compiler afterwards sees
//! a plain forward graph and needs no autodiff awareness. Design contracts:
//!
//! - **Reverse mode** ([`grad`]) walks one topological order of the loss
//!   graph backwards, accumulating a cotangent graph per node. Broadcasting
//!   is undone with sum-to-shape reductions; non-float nodes stop gradient
//!   flow (their mathematical gradient is zero almost everywhere); ops
//!   without a closed form lower to dedicated backward nodes
//!   (`SdpaBackward`, `LayerNormBackward`, `KdaBackward`, …) whose outputs
//!   are selected per differentiable input. Backward nodes themselves are
//!   deliberately *not* differentiable, so second derivatives fail loudly
//!   instead of silently producing wrong graphs.
//! - **Checkpoints** (`NodeKind::Checkpoint`) are rebuilt during the
//!   backward walk: the region's interior is deep-copied with fresh node
//!   IDs and the adjoint is built over the copy, so forward intermediates
//!   are recomputed in the backward phase rather than retained. Region
//!   inputs and constructor leaves stay shared — random draws and constants
//!   are not re-run.
//! - **Vectorization** ([`vmap`]) rebuilds only the subgraph that descends
//!   from the mapped input, inserting a batch axis at the requested dim.
//!   Elementwise ops and matmul are unchanged (broadcasting carries the
//!   batch); shape, slice, permutation, and reduction metadata shifts around
//!   the inserted axis; random sources draw per batch element; ops with
//!   data-dependent indexing are rejected explicitly.
//!
//! All traversals are iterative; depth is bounded by heap, not the call
//! stack.

use effect_torch_graph::{
    node_children, remap_children, Device, Node as GraphNode, NodeKind as GraphNodeKind,
    PositionOffset,
};
use effect_torch_runtime::DType;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

type Node = GraphNode;
type NodeKind = GraphNodeKind;

/// Node constructor shorthand: every transform funnels through `Node::new`
/// so shape/dtype validation is always applied to rebuilt graphs.
fn mk(kind: NodeKind) -> std::result::Result<Arc<Node>, String> {
    Node::new(kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use effect_torch_graph::AttentionWindow;

    fn input(slot: u32, shape: Vec<usize>) -> Arc<Node> {
        Node::new(NodeKind::Input {
            slot,
            shape,
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap()
    }

    #[test]
    fn reverse_mode_builds_gradients_from_the_production_graph() {
        let x = input(0, vec![3]);
        let square = Node::new(NodeKind::Mul {
            a: x.clone(),
            b: x.clone(),
        })
        .unwrap();
        let loss = Node::new(NodeKind::Sum {
            a: square,
            dims: vec![0],
            keepdims: false,
        })
        .unwrap();
        let gradient = grad(&loss, std::slice::from_ref(&x)).unwrap().remove(0);
        assert_eq!(gradient.shape, [3]);
        assert!(matches!(gradient.kind, NodeKind::Add { .. }));
    }

    #[test]
    fn vmap_and_checkpoint_transform_real_nodes() {
        let x = input(0, vec![3]);
        let y = Node::new(NodeKind::Tanh { a: x.clone() }).unwrap();
        let batched = input(1, vec![5, 3]);
        let mapped = vmap(&y, &x, &batched, 0).unwrap();
        assert_eq!(mapped.shape, [5, 3]);

        let checkpoint = Node::new(NodeKind::Checkpoint { a: y }).unwrap();
        let loss = Node::new(NodeKind::Sum {
            a: checkpoint,
            dims: vec![0],
            keepdims: false,
        })
        .unwrap();
        assert_eq!(grad(&loss, &[x]).unwrap()[0].shape, [3]);
    }

    #[test]
    fn grouped_query_attention_rejects_autodiff_explicitly() {
        let q = input(0, vec![1, 4, 2, 2]);
        let k = input(1, vec![1, 2, 2, 2]);
        let v = input(2, vec![1, 2, 2, 2]);
        let attention = Node::new(NodeKind::Sdpa {
            q: q.clone(),
            k,
            v,
            scale: 1.0,
            causal: true,
            window: AttentionWindow::Inherit,
        })
        .unwrap();
        let loss = Node::new(NodeKind::Sum {
            a: attention,
            dims: vec![0, 1, 2, 3],
            keepdims: false,
        })
        .unwrap();

        let error = match grad(&loss, &[q]) {
            Ok(_) => panic!("grouped-query attention unexpectedly differentiated"),
            Err(error) => error,
        };
        assert!(error.contains("grouped-query attention with unequal heads"));
    }
}

fn full(value: f64, dtype: DType, device: &Device) -> std::result::Result<Arc<Node>, String> {
    mk(NodeKind::Full {
        shape: vec![],
        value,
        dtype,
        device: device.clone(),
    })
}

fn zeros_like(target: &Node) -> std::result::Result<Arc<Node>, String> {
    mk(NodeKind::Zeros {
        shape: target.shape.clone(),
        dtype: target.dtype,
        device: target.device.clone(),
    })
}

fn ones(dtype: DType, device: &Device) -> std::result::Result<Arc<Node>, String> {
    mk(NodeKind::Ones {
        shape: vec![],
        dtype,
        device: device.clone(),
    })
}

fn add(a: Arc<Node>, b: Arc<Node>) -> std::result::Result<Arc<Node>, String> {
    mk(NodeKind::Add { a, b })
}

fn sub(a: Arc<Node>, b: Arc<Node>) -> std::result::Result<Arc<Node>, String> {
    mk(NodeKind::Sub { a, b })
}

fn mul(a: Arc<Node>, b: Arc<Node>) -> std::result::Result<Arc<Node>, String> {
    mk(NodeKind::Mul { a, b })
}

fn div(a: Arc<Node>, b: Arc<Node>) -> std::result::Result<Arc<Node>, String> {
    mk(NodeKind::Div { a, b })
}

fn neg(a: Arc<Node>) -> std::result::Result<Arc<Node>, String> {
    mk(NodeKind::Neg { a })
}

fn cast(a: Arc<Node>, dtype: DType) -> std::result::Result<Arc<Node>, String> {
    if a.dtype == dtype {
        return Ok(a);
    }
    mk(NodeKind::Cast { a, dtype })
}

fn reshape(a: Arc<Node>, shape: Vec<usize>) -> std::result::Result<Arc<Node>, String> {
    if a.shape == shape {
        return Ok(a);
    }
    mk(NodeKind::Reshape { a, shape })
}

fn broadcast_to(a: Arc<Node>, shape: &[usize]) -> std::result::Result<Arc<Node>, String> {
    if a.shape == shape {
        return Ok(a);
    }
    mk(NodeKind::BroadcastTo {
        a,
        shape: shape.to_vec(),
    })
}

fn transpose2(a: &Arc<Node>) -> std::result::Result<Arc<Node>, String> {
    let rank = a.shape.len();
    let mut dims: Vec<usize> = (0..rank).collect();
    dims.swap(rank - 2, rank - 1);
    mk(NodeKind::Permute { a: a.clone(), dims })
}

// Sum g over the dims that broadcasting expanded, then reshape to target.
fn sum_to_shape(g: &Arc<Node>, target: &[usize]) -> std::result::Result<Arc<Node>, String> {
    if g.shape == target {
        return Ok(g.clone());
    }
    if g.shape.len() < target.len() {
        return Err(format!(
            "grad: cannot reduce {:?} to higher-rank shape {target:?}",
            g.shape
        ));
    }
    let extra = g.shape.len() - target.len();
    let mut dims: Vec<usize> = (0..extra).collect();
    for i in extra..g.shape.len() {
        if target[i - extra] == 1 && g.shape[i] != 1 {
            dims.push(i);
        }
    }
    let out = if dims.is_empty() {
        g.clone()
    } else {
        mk(NodeKind::Sum {
            a: g.clone(),
            dims,
            keepdims: true,
        })?
    };
    reshape(out, target.to_vec())
}

// Broadcast a reduced cotangent (and output) back to the input shape,
// re-inserting size-1 dims when keepdims was false.
fn expand_reduced(
    g: &Arc<Node>,
    dims: &[usize],
    keepdims: bool,
    target: &[usize],
) -> std::result::Result<Arc<Node>, String> {
    let g = if keepdims {
        g.clone()
    } else {
        let kept: Vec<usize> = target
            .iter()
            .enumerate()
            .map(|(i, &d)| if dims.contains(&i) { 1 } else { d })
            .collect();
        reshape(g.clone(), kept)?
    };
    broadcast_to(g, target)
}

/// Iterative postorder from `loss`, deduplicated by node ID — the single
/// order both `grad` and `vmap` build their transforms over.
fn topo(loss: &Arc<Node>) -> Vec<Arc<Node>> {
    let mut visited = HashSet::new();
    let mut order = Vec::new();
    let mut stack = vec![(loss.clone(), false)];
    while let Some((node, processed)) = stack.pop() {
        if processed {
            order.push(node);
            continue;
        }
        if !visited.insert(node.id) {
            continue;
        }
        stack.push((node.clone(), true));
        for child in node_children(&node.kind) {
            stack.push((child, false));
        }
    }
    order
}

// Nodes of `root`'s graph whose subtree contains `x_id` — the subgraph
// that must be rebuilt under vmap.
fn descendants_of(root: &Arc<Node>, x_id: u64) -> HashSet<u64> {
    let mut set = HashSet::new();
    for node in topo(root) {
        if node.id == x_id
            || node_children(&node.kind)
                .iter()
                .any(|c| set.contains(&c.id))
        {
            set.insert(node.id);
        }
    }
    set
}

fn shift_dim(d: usize, batch_dim: usize) -> usize {
    if d >= batch_dim {
        d + 1
    } else {
        d
    }
}

fn insert_batch(shape: &[usize], dim: usize, batch: usize) -> Vec<usize> {
    let mut out = shape.to_vec();
    out.insert(dim.min(out.len()), batch);
    out
}

// Unsqueezes shared indexes at the batch dim and broadcasts them across
// it, so rank-matched indexing kernels apply per batch element.
fn broadcast_batch_indexes(
    indexes: &Arc<Node>,
    dim: usize,
    batch: usize,
) -> std::result::Result<Arc<Node>, String> {
    let unsqueezed = mk(NodeKind::Reshape {
        a: indexes.clone(),
        shape: insert_batch(&indexes.shape, dim, 1),
    })?;
    mk(NodeKind::BroadcastTo {
        a: unsqueezed,
        shape: insert_batch(&indexes.shape, dim, batch),
    })
}

// Per-op batching rules: rebuild a node for a graph whose input gained
// a leading-dim-style batch axis at `dim`. Elementwise ops, matmul and
// wrappers are unchanged (broadcasting carries the batch); shape and
// reduction metadata shifts around the inserted axis; random sources
// draw per batch element; indexing with data-dependent indexes and
// gather/scatter are rejected for now.
fn vmap_rebuild(
    node: &Node,
    dim: usize,
    batch: usize,
    f: &dyn Fn(&Arc<Node>) -> Arc<Node>,
    is_batched: &dyn Fn(u64) -> bool,
) -> std::result::Result<NodeKind, String> {
    let shift_dims = |dims: &[usize]| dims.iter().map(|&d| shift_dim(d, dim)).collect();
    match &node.kind {
        NodeKind::Randn {
            shape,
            dtype,
            device,
        } => Ok(NodeKind::Randn {
            shape: insert_batch(shape, dim, batch),
            dtype: *dtype,
            device: device.clone(),
        }),
        NodeKind::Uniform {
            lo,
            hi,
            shape,
            dtype,
            device,
        } => Ok(NodeKind::Uniform {
            lo: *lo,
            hi: *hi,
            shape: insert_batch(shape, dim, batch),
            dtype: *dtype,
            device: device.clone(),
        }),
        NodeKind::Sum { a, dims, keepdims } => Ok(NodeKind::Sum {
            a: f(a),
            dims: shift_dims(dims),
            keepdims: *keepdims,
        }),
        NodeKind::Mean { a, dims, keepdims } => Ok(NodeKind::Mean {
            a: f(a),
            dims: shift_dims(dims),
            keepdims: *keepdims,
        }),
        NodeKind::Max { a, dims, keepdims } => Ok(NodeKind::Max {
            a: f(a),
            dims: shift_dims(dims),
            keepdims: *keepdims,
        }),
        NodeKind::Min { a, dims, keepdims } => Ok(NodeKind::Min {
            a: f(a),
            dims: shift_dims(dims),
            keepdims: *keepdims,
        }),
        NodeKind::Prod { a, dims, keepdims } => Ok(NodeKind::Prod {
            a: f(a),
            dims: shift_dims(dims),
            keepdims: *keepdims,
        }),
        NodeKind::Argmax { a, dim: d } => Ok(NodeKind::Argmax {
            a: f(a),
            dim: shift_dim(*d, dim),
        }),
        NodeKind::Argmin { a, dim: d } => Ok(NodeKind::Argmin {
            a: f(a),
            dim: shift_dim(*d, dim),
        }),
        NodeKind::Cumsum { a, dim: d } => Ok(NodeKind::Cumsum {
            a: f(a),
            dim: shift_dim(*d, dim),
        }),
        NodeKind::Reshape { a, shape } => Ok(NodeKind::Reshape {
            a: f(a),
            shape: insert_batch(shape, dim, batch),
        }),
        NodeKind::Permute { a, dims } => {
            let mut out: Vec<usize> = dims.iter().map(|&d| shift_dim(d, dim)).collect();
            out.insert(dim, dim);
            Ok(NodeKind::Permute { a: f(a), dims: out })
        }
        NodeKind::Slice { a, ranges } => {
            let mut out = ranges.clone();
            out.insert(dim, (0, batch, 1));
            Ok(NodeKind::Slice {
                a: f(a),
                ranges: out,
            })
        }
        NodeKind::Concat { a, b, dim: d } => Ok(NodeKind::Concat {
            a: f(a),
            b: f(b),
            dim: shift_dim(*d, dim),
        }),
        NodeKind::BroadcastTo { a, shape } => Ok(NodeKind::BroadcastTo {
            a: f(a),
            shape: insert_batch(shape, dim, batch),
        }),
        NodeKind::IndexSelect { a, dim: d, indexes } => {
            if is_batched(indexes.id) {
                return Err(
                    "vmap: index_select with data-dependent indexes is not supported".to_string(),
                );
            }
            Ok(NodeKind::IndexSelect {
                a: f(a),
                dim: shift_dim(*d, dim),
                indexes: indexes.clone(),
            })
        }
        NodeKind::Gather { indexes, .. } | NodeKind::ScatterAdd { indexes, .. } => {
            if is_batched(indexes.id) {
                Err(
                    "vmap: gather and scatterAdd with data-dependent indexes are not supported"
                        .to_string(),
                )
            } else {
                Err(
                        "vmap: gather and scatterAdd with shared indexes are not supported under vmap (requires a batched-gather kernel)"
                            .to_string(),
                    )
            }
        }
        NodeKind::CrossEntropy { .. } | NodeKind::CrossEntropyBackward { .. } => Err(
            "vmap: crossEntropy uses data-dependent indexing and is not supported under vmap"
                .to_string(),
        ),
        NodeKind::SdpaBackward { .. } | NodeKind::SdpaBackwardOut { .. } => {
            Err("vmap: sdpa backward nodes are internal to autodiff".to_string())
        }
        NodeKind::PositionEmbedding { .. } | NodeKind::KvAttention { .. } => Err(
            "vmap: position embedding and kv attention nodes are not supported under vmap"
                .to_string(),
        ),
        NodeKind::LastTokenRow { .. } => Err(
            "vmap: last token row nodes are inference-only and not supported under vmap"
                .to_string(),
        ),
        NodeKind::QuantizedLinear { .. } | NodeKind::QuantizedEmbedding { .. } => {
            Err("vmap: encoded quantized operations are not supported under vmap".to_string())
        }
        NodeKind::KdaChunk { .. } => {
            Err("vmap: kda chunk nodes are not supported under vmap".to_string())
        }
        NodeKind::KdaRecurrence { .. }
        | NodeKind::ConvState { .. }
        | NodeKind::KdaBackward { .. }
        | NodeKind::KdaBackwardOut { .. }
        | NodeKind::ShortConv1dBackwardX { .. }
        | NodeKind::ShortConv1dBackwardW { .. }
        | NodeKind::ChunkedHeadCeBackward { .. }
        | NodeKind::ChunkedHeadCeBackwardOut { .. } => {
            Err("vmap: stateful decode and backward nodes are internal".to_string())
        }
        NodeKind::ChunkedHeadCe { .. } => {
            Err("vmap: chunked head ce nodes are not supported under vmap".to_string())
        }
        NodeKind::ShortConv1d { .. } => {
            Err("vmap: short conv nodes are not supported under vmap".to_string())
        }
        NodeKind::RotaryEmbedding { .. } => {
            Err("vmap: rotary embedding nodes are not supported under vmap".to_string())
        }
        NodeKind::Conv1d { .. }
        | NodeKind::Conv2d { .. }
        | NodeKind::ConvTranspose1d { .. }
        | NodeKind::ConvTranspose2d { .. }
        | NodeKind::Conv1dBackwardW { .. }
        | NodeKind::Conv2dBackwardW { .. } => {
            Err("vmap: convolution nodes are not supported under vmap".to_string())
        }
        _ => Ok(remap_children(&node.kind, f)),
    }
}

/// Maps the output of `y = f(x)` to `f` applied elementwise over a batch
/// axis of `batched`, where `batched` is `x` with one extra dimension of
/// size `batch` inserted at `dim`. The result is a graph computing the
/// batched output with the batch axis at `dim`.
///
/// Only the subgraph descending from `x` is rebuilt; the rest is shared.
/// Returns an error when shapes or dtypes don't match, when the output does
/// not depend on the input, or when the subgraph contains an op with no
/// batching rule (data-dependent indexing, stateful decode nodes, quantized
/// ops, convolutions — see `vmap_rebuild`).
pub fn vmap(
    y: &Arc<Node>,
    x: &Arc<Node>,
    batched: &Arc<Node>,
    dim: usize,
) -> std::result::Result<Arc<Node>, String> {
    if batched.shape.len() != x.shape.len() + 1 || dim >= batched.shape.len() {
        return Err(format!(
                "vmap: batched input shape {:?} must be the input shape {:?} with one dimension inserted",
                batched.shape, x.shape
            ));
    }
    for (i, &d) in x.shape.iter().enumerate() {
        let at = if i < dim { i } else { i + 1 };
        if batched.shape[at] != d {
            return Err(format!(
                "vmap: batched input shape {:?} does not match input shape {:?} outside dim {dim}",
                batched.shape, x.shape
            ));
        }
    }
    if batched.dtype != x.dtype {
        return Err(format!(
            "vmap: dtype mismatch, got {:?} and {:?}",
            batched.dtype, x.dtype
        ));
    }
    let batch = batched.shape[dim];
    let descendants = descendants_of(y, x.id);
    if !descendants.contains(&y.id) {
        return Err("vmap: the output does not depend on the input".to_string());
    }
    let mut map: HashMap<u64, Arc<Node>> = HashMap::new();
    map.insert(x.id, batched.clone());
    for node in topo(y) {
        // random sources inside the mapped graph draw per batch element
        // even when they do not depend on the input; everything else is
        // rebuilt only when it descends from the input
        let is_random = matches!(node.kind, NodeKind::Randn { .. } | NodeKind::Uniform { .. });
        if node.id == x.id || (!is_random && !descendants.contains(&node.id)) {
            continue;
        }
        let child_of =
            |child: &Arc<Node>| map.get(&child.id).cloned().unwrap_or_else(|| child.clone());
        let rebuilt = match &node.kind {
            // shared indexes are reshaped and broadcast across the batch
            // dim so the rank-matched gather/scatter kernels apply per
            // batch element
            NodeKind::Gather { a, dim: d, indexes }
                if !descendants.contains(&indexes.id) && !is_random =>
            {
                let idx = broadcast_batch_indexes(indexes, dim, batch)?;
                mk(NodeKind::Gather {
                    a: child_of(a),
                    dim: shift_dim(*d, dim),
                    indexes: idx,
                })?
            }
            NodeKind::ScatterAdd {
                a,
                dim: d,
                indexes,
                src,
            } if !descendants.contains(&indexes.id)
                && descendants.contains(&a.id)
                && !is_random =>
            {
                let idx = broadcast_batch_indexes(indexes, dim, batch)?;
                mk(NodeKind::ScatterAdd {
                    a: child_of(a),
                    dim: shift_dim(*d, dim),
                    indexes: idx,
                    src: child_of(src),
                })?
            }
            _ => mk(vmap_rebuild(&node, dim, batch, &child_of, &|id: u64| {
                map.contains_key(&id)
            })?)?,
        };
        map.insert(node.id, rebuilt);
    }
    Ok(map.get(&y.id).expect("vmap root").clone())
}

/// Reverse-mode gradients of a scalar loss with respect to `wrt`, in the
/// same order. A target the loss does not depend on receives an explicit
/// zeros graph, keeping the walk total.
///
/// The loss must be 0-d and floating point, and every target must be
/// floating point. Quantized (encoded) ops are inference-only and rejected
/// up front; inference-only nodes (`KvAttention`, `LastTokenRow`, stateful
/// decode nodes) and backward nodes (no second-order support) fail during
/// the walk with an explicit error.
pub fn grad(loss: &Arc<Node>, wrt: &[Arc<Node>]) -> std::result::Result<Vec<Arc<Node>>, String> {
    if !loss.shape.is_empty() {
        return Err(format!(
            "grad: expected a scalar (0-d) loss, got shape {:?}",
            loss.shape
        ));
    }
    if !loss.dtype.is_float() {
        return Err(format!(
            "grad: loss dtype must be floating point, got {:?}",
            loss.dtype
        ));
    }
    for target in wrt {
        if !target.dtype.is_float() {
            return Err(format!(
                "grad: cannot differentiate with respect to non-float dtype {:?}",
                target.dtype
            ));
        }
    }
    let order = topo(loss);
    if order.iter().any(|node| {
        matches!(
            node.kind,
            NodeKind::QuantizedLinear { .. } | NodeKind::QuantizedEmbedding { .. }
        )
    }) {
        return Err(
            "grad: encoded quantized operations are inference-only and not differentiable"
                .to_string(),
        );
    }
    let mut cotangents: HashMap<u64, Arc<Node>> = HashMap::new();
    cotangents.insert(loss.id, ones(loss.dtype, &loss.device)?);
    backward(&order, &mut cotangents)?;
    Ok(wrt
        .iter()
        .map(|target| match cotangents.get(&target.id) {
            Some(g) => g.clone(),
            None => zeros_like(target).expect("zeros_like"),
        })
        .collect())
}

// Nodes reachable from the walk's root without passing through the
// checkpoint — these are the region's inputs and stay shared.
fn outside_set(order: &[Arc<Node>], checkpoint_id: u64) -> HashSet<u64> {
    let mut visited = HashSet::new();
    if let Some(root) = order.last() {
        let mut stack = vec![root.clone()];
        while let Some(node) = stack.pop() {
            if node.id == checkpoint_id {
                continue;
            }
            if !visited.insert(node.id) {
                continue;
            }
            for child in node_children(&node.kind) {
                stack.push(child);
            }
        }
    }
    visited
}

/// The reverse-mode walk: iterates the forward topological order backwards,
/// extending `cotangents` with each node's adjoint contribution. Cotangents
/// of nodes with multiple consumers accumulate with `Add` graphs, matching
/// the total-derivative rule. Per-op adjoint rules are inline; ops with a
/// closed multi-output adjoint emit one backward node plus one picker per
/// differentiable input (the `SdpaBackward` pattern).
fn backward(
    order: &[Arc<Node>],
    cotangents: &mut HashMap<u64, Arc<Node>>,
) -> std::result::Result<(), String> {
    for node in order.iter().rev() {
        let Some(g) = cotangents.get(&node.id).cloned() else {
            continue;
        };
        // Gradients do not flow through non-float nodes (comparisons,
        // integer arithmetic): their mathematical gradient is zero
        // almost everywhere, so the cotangent is dropped here.
        if !node.dtype.is_float() {
            continue;
        }
        let mut accumulate =
            |input: &Arc<Node>, contribution: std::result::Result<Arc<Node>, String>| {
                let contribution = contribution?;
                cotangents
                    .entry(input.id)
                    .and_modify(|existing| {
                        *existing = add(existing.clone(), contribution.clone())
                            .expect("grad accumulation broadcast")
                    })
                    .or_insert(contribution);
                Ok::<(), String>(())
            };
        match &node.kind {
            NodeKind::Add { a, b } => {
                accumulate(a, sum_to_shape(&g, &a.shape))?;
                accumulate(b, sum_to_shape(&g, &b.shape))?;
            }
            NodeKind::Sub { a, b } => {
                accumulate(a, sum_to_shape(&g, &a.shape))?;
                accumulate(b, sum_to_shape(&neg(g)?, &b.shape))?;
            }
            NodeKind::Mul { a, b } => {
                accumulate(a, sum_to_shape(&mul(g.clone(), b.clone())?, &a.shape))?;
                accumulate(b, sum_to_shape(&mul(g.clone(), a.clone())?, &b.shape))?;
            }
            NodeKind::Div { a, b } => {
                accumulate(a, sum_to_shape(&div(g.clone(), b.clone())?, &a.shape))?;
                let gb = neg(div(mul(g.clone(), a.clone())?, mul(b.clone(), b.clone())?)?)?;
                accumulate(b, sum_to_shape(&gb, &b.shape))?;
            }
            NodeKind::Neg { a } => {
                accumulate(a, neg(g))?;
            }
            NodeKind::Maximum { a, b } | NodeKind::Minimum { a, b } => {
                let is_max = matches!(&node.kind, NodeKind::Maximum { .. });
                // ties route the gradient to the left operand
                let (mask_a, mask_b) = if is_max {
                    (
                        mk(NodeKind::Ge {
                            a: a.clone(),
                            b: b.clone(),
                        })?,
                        mk(NodeKind::Lt {
                            a: a.clone(),
                            b: b.clone(),
                        })?,
                    )
                } else {
                    (
                        mk(NodeKind::Le {
                            a: a.clone(),
                            b: b.clone(),
                        })?,
                        mk(NodeKind::Gt {
                            a: a.clone(),
                            b: b.clone(),
                        })?,
                    )
                };
                let dtype = node.dtype;
                let ga = mul(g.clone(), cast(mask_a, dtype)?)?;
                let gb = mul(g, cast(mask_b, dtype)?)?;
                accumulate(a, sum_to_shape(&ga, &a.shape))?;
                accumulate(b, sum_to_shape(&gb, &b.shape))?;
            }
            NodeKind::Abs { a } => {
                let zero = full(0.0, a.dtype, &a.device)?;
                let sign = mk(NodeKind::Sub {
                    a: cast(
                        mk(NodeKind::Gt {
                            a: a.clone(),
                            b: zero.clone(),
                        })?,
                        a.dtype,
                    )?,
                    b: cast(
                        mk(NodeKind::Lt {
                            a: a.clone(),
                            b: zero,
                        })?,
                        a.dtype,
                    )?,
                })?;
                accumulate(a, mul(g, sign))?;
            }
            NodeKind::Sqrt { a } => {
                let half = full(0.5, node.dtype, &node.device)?;
                accumulate(a, div(mul(g, half)?, node.clone()))?;
            }
            NodeKind::Exp { a } => {
                accumulate(a, mul(g, node.clone()))?;
            }
            NodeKind::Log { a } => {
                accumulate(a, div(g, a.clone()))?;
            }
            NodeKind::Sin { a } => {
                accumulate(a, mul(g, mk(NodeKind::Cos { a: a.clone() })?))?;
            }
            NodeKind::Cos { a } => {
                accumulate(a, neg(mul(g, mk(NodeKind::Sin { a: a.clone() })?)?))?;
            }
            NodeKind::Tanh { a } => {
                let one = full(1.0, node.dtype, &node.device)?;
                accumulate(a, mul(g, add(one, neg(mul(node.clone(), node.clone())?)?)?))?;
            }
            NodeKind::Relu { a } => {
                let zero = full(0.0, a.dtype, &a.device)?;
                let mask = cast(
                    mk(NodeKind::Gt {
                        a: a.clone(),
                        b: zero,
                    })?,
                    a.dtype,
                )?;
                accumulate(a, mul(g, mask))?;
            }
            NodeKind::Erf { a } => {
                let c = full(2.0 / std::f64::consts::PI.sqrt(), a.dtype, &a.device)?;
                let e = mk(NodeKind::Exp {
                    a: neg(mul(a.clone(), a.clone())?)?,
                })?;
                accumulate(a, mul(mul(g, c)?, e))?;
            }
            // exact: d/dx gelu(x) = Φ(x) + x·φ(x) with
            // Φ(x) = ½(1+erf(x/√2)), φ(x) = e^{-x²/2}/√(2π).
            // tanh: ½(1+t) + ½x(1-t²)·c(1+3k·x²) with
            // t = tanh(c(x+kx³)), c = √(2/π), k = 0.044715.
            NodeKind::Gelu { a, approximate } => {
                let dt = a.dtype;
                let dv = &a.device;
                let half = full(0.5, dt, dv)?;
                let one = full(1.0, dt, dv)?;
                let dg = if *approximate {
                    let c = full((2.0f64 / std::f64::consts::PI).sqrt(), dt, dv)?;
                    let k = full(0.044715, dt, dv)?;
                    let x2 = mul(a.clone(), a.clone())?;
                    let u = mul(
                        add(a.clone(), mul(mul(x2.clone(), a.clone())?, k.clone())?)?,
                        c.clone(),
                    )?;
                    let t = mk(NodeKind::Tanh { a: u })?;
                    let sech2 = add(one.clone(), neg(mul(t.clone(), t.clone())?)?)?;
                    let du = mul(c, add(one.clone(), mul(full(3.0, dt, dv)?, mul(k, x2)?)?)?)?;
                    add(
                        mul(half.clone(), add(one, t)?)?,
                        mul(mul(mul(half, a.clone())?, sech2)?, du)?,
                    )?
                } else {
                    let inner = mk(NodeKind::Erf {
                        a: mul(a.clone(), full(std::f64::consts::FRAC_1_SQRT_2, dt, dv)?)?,
                    })?;
                    let phi_up = mul(half, add(one, inner)?)?;
                    let neg_x2_half = div(neg(mul(a.clone(), a.clone())?)?, full(2.0, dt, dv)?)?;
                    let xphi = mul(
                        a.clone(),
                        mul(
                            full(1.0 / (2.0 * std::f64::consts::PI).sqrt(), dt, dv)?,
                            mk(NodeKind::Exp { a: neg_x2_half })?,
                        )?,
                    )?;
                    add(phi_up, xphi)?
                };
                accumulate(a, mul(g, dg))?;
            }
            // zero almost everywhere; the cotangent is an explicit zero
            // rather than a drop so higher-order walks stay total
            NodeKind::Floor { a }
            | NodeKind::Ceil { a }
            | NodeKind::Round { a }
            | NodeKind::Sign { a } => {
                accumulate(a, zeros_like(a))?;
            }
            NodeKind::Where { cond, a, b } => {
                let zero = full(0.0, node.dtype, &node.device)?;
                let ga = mk(NodeKind::Where {
                    cond: cond.clone(),
                    a: g.clone(),
                    b: zero.clone(),
                })?;
                let gb = mk(NodeKind::Where {
                    cond: cond.clone(),
                    a: zero,
                    b: g.clone(),
                })?;
                accumulate(a, sum_to_shape(&ga, &a.shape))?;
                accumulate(b, sum_to_shape(&gb, &b.shape))?;
            }
            NodeKind::Cumsum { a, dim } => {
                // d out[i] / d x[j] = 1 when i >= j, so the adjoint is the
                // reverse cumulative sum: total - cumsum(g) + g
                let total = mk(NodeKind::Sum {
                    a: g.clone(),
                    dims: vec![*dim],
                    keepdims: true,
                })?;
                let total = broadcast_to(total, &a.shape)?;
                let cs = mk(NodeKind::Cumsum {
                    a: g.clone(),
                    dim: *dim,
                })?;
                accumulate(a, add(g.clone(), sub(total, cs)?))?;
            }
            NodeKind::IndexSelect { a, dim, indexes } => {
                // scatter the cotangent back into a zero tensor of the
                // input shape at the selected positions
                let mut ishape = vec![1usize; a.shape.len()];
                ishape[*dim] = indexes.shape[0];
                let idx = reshape(indexes.clone(), ishape)?;
                let idx = broadcast_to(idx, &g.shape)?;
                accumulate(
                    a,
                    mk(NodeKind::ScatterAdd {
                        a: zeros_like(a.as_ref())?,
                        dim: *dim,
                        indexes: idx,
                        src: g,
                    }),
                )?;
            }
            NodeKind::ScatterAdd {
                a,
                dim,
                indexes,
                src,
            } => {
                accumulate(a, Ok(g.clone()))?;
                accumulate(
                    src,
                    mk(NodeKind::Gather {
                        a: g,
                        dim: *dim,
                        indexes: indexes.clone(),
                    }),
                )?;
            }
            NodeKind::Gather { a, dim, indexes } => {
                // the scatter kernel requires src to match the target
                // outside dim, so pad the cotangent and the indexes with
                // harmless zeros (index 0, value 0) where they are smaller
                let mut g_padded = g;
                let mut idx_padded = indexes.clone();
                for i in 0..a.shape.len() {
                    if i == *dim {
                        continue;
                    }
                    let missing = a.shape[i].saturating_sub(g_padded.shape[i]);
                    if missing > 0 {
                        let mut zshape = g_padded.shape.clone();
                        zshape[i] = missing;
                        let mut ishape = idx_padded.shape.clone();
                        ishape[i] = missing;
                        g_padded = mk(NodeKind::Concat {
                            a: g_padded.clone(),
                            b: mk(NodeKind::Zeros {
                                shape: zshape,
                                dtype: g_padded.dtype,
                                device: g_padded.device.clone(),
                            })?,
                            dim: i,
                        })?;
                        idx_padded = mk(NodeKind::Concat {
                            a: idx_padded.clone(),
                            b: mk(NodeKind::Zeros {
                                shape: ishape,
                                dtype: DType::I64,
                                device: idx_padded.device.clone(),
                            })?,
                            dim: i,
                        })?;
                    }
                }
                accumulate(
                    a,
                    mk(NodeKind::ScatterAdd {
                        a: zeros_like(a.as_ref())?,
                        dim: *dim,
                        indexes: idx_padded,
                        src: g_padded,
                    }),
                )?;
            }
            NodeKind::Prod { a, dims, keepdims } => {
                // d prod / d x_i = prod / x_i; undefined when any factor
                // is zero (the true adjoint needs the zero-free subproducts)
                let out_b = expand_reduced(&node.clone(), dims, *keepdims, &a.shape)?;
                let g_b = expand_reduced(&g, dims, *keepdims, &a.shape)?;
                accumulate(a, div(mul(g_b, out_b)?, a.clone()))?;
            }
            NodeKind::Pow { a, exp } => {
                let c = full(*exp, a.dtype, &a.device)?;
                let base = mk(NodeKind::Pow {
                    a: a.clone(),
                    exp: exp - 1.0,
                })?;
                accumulate(a, mul(mul(g, c)?, base))?;
            }
            NodeKind::Cast { a, .. } => {
                if a.dtype.is_float() {
                    accumulate(a, cast(g, a.dtype))?;
                }
            }
            NodeKind::Sum { a, dims, keepdims } => {
                accumulate(a, expand_reduced(&g, dims, *keepdims, &a.shape))?;
            }
            NodeKind::Mean { a, dims, keepdims } => {
                let count: usize = dims.iter().map(|&d| a.shape[d]).product();
                let scaled = div(g, full(count as f64, a.dtype, &a.device)?)?;
                accumulate(a, expand_reduced(&scaled, dims, *keepdims, &a.shape))?;
            }
            NodeKind::Max { a, dims, keepdims } | NodeKind::Min { a, dims, keepdims } => {
                let kept: Vec<usize> = a
                    .shape
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| if dims.contains(&i) { 1 } else { d })
                    .collect();
                let out_r = if *keepdims {
                    node.clone()
                } else {
                    reshape(node.clone(), kept.clone())?
                };
                let g_r = if *keepdims {
                    g.clone()
                } else {
                    reshape(g, kept)?
                };
                let out_b = broadcast_to(out_r, &a.shape)?;
                let mask = cast(
                    mk(NodeKind::Eq {
                        a: a.clone(),
                        b: out_b,
                    })?,
                    a.dtype,
                )?;
                let denom = broadcast_to(
                    mk(NodeKind::Sum {
                        a: mask.clone(),
                        dims: dims.clone(),
                        keepdims: true,
                    })?,
                    &a.shape,
                )?;
                accumulate(a, div(mul(broadcast_to(g_r, &a.shape)?, mask)?, denom))?;
            }
            NodeKind::Reshape { a, .. } => {
                accumulate(a, reshape(g, a.shape.clone()))?;
            }
            NodeKind::Permute { a, dims } => {
                let mut inverse = vec![0usize; dims.len()];
                for (i, &d) in dims.iter().enumerate() {
                    inverse[d] = i;
                }
                accumulate(
                    a,
                    mk(NodeKind::Permute {
                        a: g,
                        dims: inverse,
                    }),
                )?;
            }
            NodeKind::Slice { a, ranges } => {
                let mut cur = g;
                for (dim, &(start, stop, stride)) in ranges.iter().enumerate() {
                    let n = a.shape[dim];
                    if stride != 1 && cur.shape[dim] > 0 {
                        // dilate the cotangent along the sliced dim by
                        // interleaving stride-1 zeros, so it lines up with
                        // the positions the forward pass actually read
                        let len = cur.shape[dim];
                        let mut g_shape = cur.shape.clone();
                        g_shape.insert(dim + 1, 1);
                        let mut z_shape = cur.shape.clone();
                        z_shape.insert(dim + 1, stride - 1);
                        let mut expanded = cur.shape.clone();
                        expanded[dim] = len * stride;
                        let g_r = reshape(cur, g_shape)?;
                        let z = mk(NodeKind::Zeros {
                            shape: z_shape,
                            dtype: g_r.dtype,
                            device: g_r.device.clone(),
                        })?;
                        let cat = mk(NodeKind::Concat {
                            a: g_r,
                            b: z,
                            dim: dim + 1,
                        })?;
                        let mut wide = reshape(cat, expanded)?;
                        let keep = (len - 1) * stride + 1;
                        if keep < len * stride {
                            let trim: Vec<(usize, usize, usize)> = (0..wide.shape.len())
                                .map(|i| {
                                    if i == dim {
                                        (0, keep, 1)
                                    } else {
                                        (0, wide.shape[i], 1)
                                    }
                                })
                                .collect();
                            wide = mk(NodeKind::Slice {
                                a: wide,
                                ranges: trim,
                            })?;
                        }
                        cur = wide;
                    }
                    if start > 0 {
                        let mut zshape = cur.shape.clone();
                        zshape[dim] = start;
                        cur = mk(NodeKind::Concat {
                            a: mk(NodeKind::Zeros {
                                shape: zshape,
                                dtype: cur.dtype,
                                device: cur.device.clone(),
                            })?,
                            b: cur.clone(),
                            dim,
                        })?;
                    }
                    if stop < n {
                        let mut zshape = cur.shape.clone();
                        zshape[dim] = n - stop;
                        cur = mk(NodeKind::Concat {
                            a: cur.clone(),
                            b: mk(NodeKind::Zeros {
                                shape: zshape,
                                dtype: cur.dtype,
                                device: cur.device.clone(),
                            })?,
                            dim,
                        })?;
                    }
                }
                accumulate(a, Ok(cur))?;
            }
            NodeKind::Concat { a, b, dim } => {
                let mut offset = 0usize;
                for input in [a, b] {
                    let len = input.shape[*dim];
                    let ranges: Vec<(usize, usize, usize)> = (0..g.shape.len())
                        .map(|i| {
                            if i == *dim {
                                (offset, offset + len, 1)
                            } else {
                                (0, g.shape[i], 1)
                            }
                        })
                        .collect();
                    accumulate(
                        input,
                        mk(NodeKind::Slice {
                            a: g.clone(),
                            ranges,
                        }),
                    )?;
                    offset += len;
                }
            }
            NodeKind::BroadcastTo { a, .. } => {
                accumulate(a, sum_to_shape(&g, &a.shape))?;
            }
            NodeKind::Matmul { a, b } => {
                let ga = mk(NodeKind::Matmul {
                    a: g.clone(),
                    b: transpose2(b)?,
                })?;
                accumulate(a, sum_to_shape(&ga, &a.shape))?;
                let gb = mk(NodeKind::Matmul {
                    a: transpose2(a)?,
                    b: g.clone(),
                })?;
                accumulate(b, sum_to_shape(&gb, &b.shape))?;
            }
            NodeKind::Inverse { a } => {
                // d inv = -inv^T @ g @ inv^T
                let t = transpose2(&node)?;
                accumulate(
                    a,
                    neg(mk(NodeKind::Matmul {
                        a: mk(NodeKind::Matmul { a: t.clone(), b: g })?,
                        b: t,
                    })?),
                )?;
            }
            NodeKind::Det { a } => {
                // d det = det * inv^T; the batch-shaped det and cotangent
                // are expanded across the matrix dimensions
                let inv_t = transpose2(&mk(NodeKind::Inverse { a: a.clone() })?)?;
                let rank = a.shape.len();
                let dims = vec![rank - 2, rank - 1];
                let det_b = expand_reduced(&node.clone(), &dims, false, &a.shape)?;
                let g_b = expand_reduced(&g, &dims, false, &a.shape)?;
                accumulate(a, mul(g_b, mul(det_b, inv_t)?))?;
            }
            NodeKind::Solve { a, b } => {
                // out = a^-1 b; g_b = a^-T g; g_a = -g_b @ out^T
                let inv_t = transpose2(&mk(NodeKind::Inverse { a: a.clone() })?)?;
                let gb = mk(NodeKind::Matmul {
                    a: inv_t,
                    b: g.clone(),
                })?;
                accumulate(b, Ok(gb.clone()))?;
                accumulate(
                    a,
                    neg(mk(NodeKind::Matmul {
                        a: gb,
                        b: transpose2(&node)?,
                    })?),
                )?;
            }
            NodeKind::StopGradient { .. } => {}
            NodeKind::CrossEntropy {
                logits,
                target,
                ignore_index,
                reduction,
            } => {
                let gb = mk(NodeKind::CrossEntropyBackward {
                    logits: logits.clone(),
                    target: target.clone(),
                    ignore_index: *ignore_index,
                    reduction: *reduction,
                })?;
                accumulate(logits, mul(g, gb))?;
            }
            NodeKind::ChunkedHeadCe {
                x,
                weight,
                bias,
                target,
                ignore_index,
            } => {
                // Closed-form chunked adjoint: one backward node, one
                // picker per differentiable input (the SdpaBackward
                // pattern).
                let bw = mk(NodeKind::ChunkedHeadCeBackward {
                    x: x.clone(),
                    weight: weight.clone(),
                    bias: bias.clone(),
                    target: target.clone(),
                    g: g.clone(),
                    ignore_index: *ignore_index,
                })?;
                for (input, index) in [(x, 0u8), (weight, 1u8), (bias, 2u8)] {
                    let out = mk(NodeKind::ChunkedHeadCeBackwardOut {
                        of: bw.clone(),
                        index,
                    })?;
                    accumulate(input, Ok(out))?;
                }
            }
            NodeKind::ChunkedHeadCeBackward { .. } | NodeKind::ChunkedHeadCeBackwardOut { .. } => {
                return Err(
                    "grad: chunked head ce backward nodes are not differentiable (no second-order)"
                        .to_string(),
                );
            }
            NodeKind::CrossEntropyBackward { .. } => {
                return Err(
                    "grad: cross-entropy backward nodes are not differentiable (no second-order)"
                        .to_string(),
                );
            }
            NodeKind::RotaryEmbeddingBackward { .. } => {
                return Err(
                    "grad: rotary backward nodes are not differentiable (no second-order)"
                        .to_string(),
                );
            }
            NodeKind::LayerNorm {
                x,
                weight,
                bias,
                eps,
            } => {
                let bwd = mk(NodeKind::LayerNormBackward {
                    x: x.clone(),
                    weight: weight.clone(),
                    g: g.clone(),
                    eps: *eps,
                })?;
                accumulate(x, Ok(bwd.clone()))?;
                accumulate(
                    weight,
                    mk(NodeKind::LayerNormBackwardOut {
                        of: bwd.clone(),
                        index: 1,
                    }),
                )?;
                accumulate(
                    bias,
                    mk(NodeKind::LayerNormBackwardOut { of: bwd, index: 2 }),
                )?;
            }
            NodeKind::LayerNormBackward { .. } | NodeKind::LayerNormBackwardOut { .. } => {
                return Err(
                    "grad: layer norm backward nodes are not differentiable (no second-order)"
                        .to_string(),
                );
            }
            NodeKind::RmsNorm { .. } => {
                return Err("grad: RMS norm is not differentiable yet".to_string());
            }
            NodeKind::Linear { x, weight, bias } => {
                // y = x·W + b over the last dim: dx = g·Wᵀ,
                // dw = xᵀ·g (reduced over leading dims), db = Σ g.
                let wt = mk(NodeKind::Permute {
                    a: weight.clone(),
                    dims: vec![1, 0],
                })?;
                accumulate(
                    x,
                    mk(NodeKind::Matmul {
                        a: g.clone(),
                        b: wt,
                    }),
                )?;
                let rank = x.shape.len();
                let (k, n) = (weight.shape[0], weight.shape[1]);
                let rows: usize = x.shape[..rank - 1].iter().product();
                let x2d = mk(NodeKind::Reshape {
                    a: x.clone(),
                    shape: vec![rows, k],
                })?;
                let x2d_t = mk(NodeKind::Permute {
                    a: x2d,
                    dims: vec![1, 0],
                })?;
                let g2d = mk(NodeKind::Reshape {
                    a: g.clone(),
                    shape: vec![rows, n],
                })?;
                accumulate(
                    weight,
                    mk(NodeKind::Matmul {
                        a: x2d_t,
                        b: g2d.clone(),
                    }),
                )?;
                let reduce_dims: Vec<usize> = (0..rank - 1).collect();
                accumulate(
                    bias,
                    mk(NodeKind::Sum {
                        a: g.clone(),
                        dims: reduce_dims,
                        keepdims: false,
                    }),
                )?;
            }
            NodeKind::QuantizedLinear { .. } | NodeKind::QuantizedEmbedding { .. } => {
                return Err(
                    "grad: encoded quantized operations are inference-only and not differentiable"
                        .to_string(),
                );
            }
            NodeKind::Sdpa {
                q,
                k,
                v,
                scale,
                causal,
                window,
            } => {
                let rank = q.shape.len();
                if rank >= 3 && q.shape[rank - 3] != k.shape[rank - 3] {
                    return Err(
                        "grad: grouped-query attention with unequal heads is not differentiable"
                            .to_string(),
                    );
                }
                let bw = mk(NodeKind::SdpaBackward {
                    q: q.clone(),
                    k: k.clone(),
                    v: v.clone(),
                    g: g.clone(),
                    fwd: node.clone(),
                    scale: *scale,
                    causal: *causal,
                    window: *window,
                })?;
                for (input, index) in [(q, 0u8), (k, 1u8), (v, 2u8)] {
                    let out = mk(NodeKind::SdpaBackwardOut {
                        of: bw.clone(),
                        index,
                    })?;
                    accumulate(input, Ok(out))?;
                }
            }
            NodeKind::SdpaBackward { .. } | NodeKind::SdpaBackwardOut { .. } => {
                return Err(
                    "grad: sdpa backward nodes are not differentiable (no second-order)"
                        .to_string(),
                );
            }
            NodeKind::PositionEmbedding { weight, seq_len } => {
                // dW: rows 0..seq_len-1 accumulate the cotangent, the
                // rest stay zero — scatter-add of g into zeros_like(W)
                // at rows arange(seq_len) (indexes padded to g's shape
                // per the scatter contract).
                let t = *seq_len;
                let e = weight.shape[1];
                let rows = mk(NodeKind::Arange {
                    start: 0.0,
                    end: t as f64,
                    step: 1.0,
                    dtype: DType::I64,
                    device: weight.device.clone(),
                })?;
                let rows = mk(NodeKind::Reshape {
                    a: rows,
                    shape: vec![t, 1],
                })?;
                let indexes = mk(NodeKind::BroadcastTo {
                    a: rows,
                    shape: vec![t, e],
                })?;
                accumulate(
                    weight,
                    mk(NodeKind::ScatterAdd {
                        a: zeros_like(weight.as_ref())?,
                        dim: 0,
                        indexes,
                        src: g.clone(),
                    }),
                )?;
            }
            NodeKind::KvAttention { .. } => {
                return Err(
                    "grad: kv attention is an inference-only node and is not differentiable"
                        .to_string(),
                );
            }
            NodeKind::LastTokenRow { .. } => {
                return Err(
                    "grad: last token row is an inference-only node and is not differentiable"
                        .to_string(),
                );
            }
            NodeKind::KdaChunk {
                q,
                k,
                v,
                log_decay,
                beta,
                scale,
            } => {
                // Closed-form adjoint (RFC 0018 phase 4): one backward
                // node, one picker per differentiable input — the
                // SdpaBackward pattern.
                let bw = mk(NodeKind::KdaBackward {
                    q: q.clone(),
                    k: k.clone(),
                    v: v.clone(),
                    log_decay: log_decay.clone(),
                    beta: beta.clone(),
                    g: g.clone(),
                    scale: *scale,
                })?;
                for (input, index) in [(q, 0u8), (k, 1u8), (v, 2u8), (log_decay, 3u8), (beta, 4u8)]
                {
                    let out = mk(NodeKind::KdaBackwardOut {
                        of: bw.clone(),
                        index,
                    })?;
                    accumulate(input, Ok(out))?;
                }
            }
            NodeKind::KdaBackward { .. } | NodeKind::KdaBackwardOut { .. } => {
                return Err(
                    "grad: kda backward nodes are not differentiable (no second-order)".to_string(),
                );
            }
            NodeKind::KdaRecurrence { .. } | NodeKind::ConvState { .. } => {
                return Err(
                    "grad: stateful decode nodes are inference-only and are not differentiable"
                        .to_string(),
                );
            }
            NodeKind::ShortConv1d { x, weight } => {
                // dx is the full correlation of the cotangent with the
                // (unflipped) weight; dw is the per-tap correlation of
                // cotangent and input over the causal window.
                accumulate(
                    x,
                    mk(NodeKind::ShortConv1dBackwardX {
                        x: x.clone(),
                        weight: weight.clone(),
                        g: g.clone(),
                    }),
                )?;
                accumulate(
                    weight,
                    mk(NodeKind::ShortConv1dBackwardW {
                        x: x.clone(),
                        weight: weight.clone(),
                        g: g.clone(),
                    }),
                )?;
            }
            NodeKind::ShortConv1dBackwardX { .. } | NodeKind::ShortConv1dBackwardW { .. } => {
                return Err(
                    "grad: short conv backward nodes are not differentiable (no second-order)"
                        .to_string(),
                );
            }
            NodeKind::RotaryEmbedding {
                x,
                seq_len,
                theta,
                offset,
                layout,
            } => {
                if *offset != PositionOffset::Absolute {
                    return Err(
                        "grad: cursor-offset rotary embedding is not differentiable".to_string()
                    );
                }
                // y = R x with R orthogonal per position: dx = Rᵀ g,
                // the same rotation with negated angles — a single
                // semantic node (the fused kernel's sign flip).
                accumulate(
                    x,
                    mk(NodeKind::RotaryEmbeddingBackward {
                        g: g.clone(),
                        shape: x.shape.clone(),
                        seq_len: *seq_len,
                        theta: *theta,
                        layout: *layout,
                    }),
                )?;
            }
            NodeKind::Conv1d {
                x,
                w,
                stride,
                padding,
                dilation,
                groups,
            } => {
                // dX is the full convolution of the cotangent with the
                // weight: a transposed convolution whose output_padding
                // fills the stride remainder; since our forward is a
                // correlation, the same unflipped weight is the adjoint
                // kernel.
                let out_pad = x.shape[2]
                    - ((g.shape[2] - 1) * stride + dilation * (w.shape[2] - 1) + 1 - 2 * padding);
                accumulate(
                    x,
                    mk(NodeKind::ConvTranspose1d {
                        x: g.clone(),
                        w: w.clone(),
                        stride: *stride,
                        padding: *padding,
                        output_padding: out_pad,
                        dilation: *dilation,
                        groups: *groups,
                    }),
                )?;
                accumulate(
                    w,
                    mk(NodeKind::Conv1dBackwardW {
                        x: x.clone(),
                        g,
                        kernel: w.shape[2],
                        out_channels: w.shape[0],
                        stride: *stride,
                        padding: *padding,
                        dilation: *dilation,
                        groups: *groups,
                    }),
                )?;
            }
            NodeKind::Conv2d {
                x,
                w,
                stride,
                padding,
                dilation,
                groups,
            } => {
                let out_pad_h = x.shape[2]
                    - ((g.shape[2] - 1) * stride + dilation * (w.shape[2] - 1) + 1 - 2 * padding);
                let out_pad_w = x.shape[3]
                    - ((g.shape[3] - 1) * stride + dilation * (w.shape[3] - 1) + 1 - 2 * padding);
                // candle's conv_transpose2d takes a single output_padding;
                // when the per-dim stride remainders differ, compute with
                // the smaller one and append the missing strip explicitly
                // — remainder strips beyond the full convolution are
                // always zeros, so this is exact
                let min_pad = out_pad_h.min(out_pad_w);
                let mut dx = mk(NodeKind::ConvTranspose2d {
                    x: g.clone(),
                    w: w.clone(),
                    stride: *stride,
                    padding: *padding,
                    output_padding: min_pad,
                    dilation: *dilation,
                    groups: *groups,
                })?;
                for (dim, out_pad) in [(2usize, out_pad_h), (3usize, out_pad_w)] {
                    if out_pad > min_pad {
                        let mut zshape = dx.shape.clone();
                        zshape[dim] = out_pad - min_pad;
                        dx = mk(NodeKind::Concat {
                            a: dx,
                            b: mk(NodeKind::Zeros {
                                shape: zshape,
                                dtype: g.dtype,
                                device: g.device.clone(),
                            })?,
                            dim,
                        })?;
                    }
                }
                accumulate(x, Ok(dx))?;
                accumulate(
                    w,
                    mk(NodeKind::Conv2dBackwardW {
                        x: x.clone(),
                        g,
                        kernel: [w.shape[2], w.shape[3]],
                        out_channels: w.shape[0],
                        stride: *stride,
                        padding: *padding,
                        dilation: *dilation,
                        groups: *groups,
                    }),
                )?;
            }
            NodeKind::ConvTranspose1d { .. }
            | NodeKind::ConvTranspose2d { .. }
            | NodeKind::Conv1dBackwardW { .. }
            | NodeKind::Conv2dBackwardW { .. } => {
                return Err(
                    "grad: convolution backward nodes are not differentiable (no second-order)"
                        .to_string(),
                );
            }
            NodeKind::AdamWStep { .. }
            | NodeKind::AdamWOut { .. }
            | NodeKind::SgdStep { .. }
            | NodeKind::SgdOut { .. } => {
                return Err("grad: optimizer update nodes are not differentiable".to_string());
            }
            NodeKind::Checkpoint { a } => {
                // Deep-copy the region's interior with fresh node ids and
                // build the adjoint over the copy: forward intermediates
                // are recomputed in the backward phase instead of being
                // retained. Region inputs (nodes also reachable from
                // outside the checkpoint) and constructor leaves are
                // shared, so randn draws and constants are not re-run.
                let outside = outside_set(order, node.id);
                let region_topo = topo(a);
                let mut map: HashMap<u64, Arc<Node>> = HashMap::new();
                let mut shared: HashMap<u64, Arc<Node>> = HashMap::new();
                for rn in &region_topo {
                    if outside.contains(&rn.id) || node_children(&rn.kind).is_empty() {
                        shared.insert(rn.id, rn.clone());
                        continue;
                    }
                    let kind = remap_children(&rn.kind, &|child: &Arc<Node>| {
                        map.get(&child.id).cloned().unwrap_or_else(|| child.clone())
                    });
                    let copied = Node::new(kind)?;
                    map.insert(rn.id, copied);
                }
                if let Some(copied_root) = map.get(&a.id).cloned() {
                    let copied_order: Vec<Arc<Node>> = region_topo
                        .iter()
                        .filter_map(|rn| map.get(&rn.id).cloned())
                        .collect();
                    let copy_ids: HashSet<u64> = map.values().map(|n| n.id).collect();
                    let mut sub: HashMap<u64, Arc<Node>> = HashMap::new();
                    sub.insert(copied_root.id, g.clone());
                    backward(&copied_order, &mut sub)?;
                    for (id, contribution) in sub {
                        if copy_ids.contains(&id) {
                            continue;
                        }
                        if let Some(input) = shared.get(&id) {
                            accumulate(input, Ok(contribution))?;
                        }
                    }
                } else {
                    accumulate(a, Ok(g.clone()))?;
                }
            }
            NodeKind::Leaf(_)
            | NodeKind::Input { .. }
            | NodeKind::ScalarInput { .. }
            | NodeKind::FromBytes { .. }
            | NodeKind::Zeros { .. }
            | NodeKind::Ones { .. }
            | NodeKind::Full { .. }
            | NodeKind::Randn { .. }
            | NodeKind::Uniform { .. }
            | NodeKind::Arange { .. }
            | NodeKind::Eye { .. } => {}
            NodeKind::Eq { .. }
            | NodeKind::Gt { .. }
            | NodeKind::Lt { .. }
            | NodeKind::Ge { .. }
            | NodeKind::Le { .. }
            | NodeKind::Argmax { .. }
            | NodeKind::Argmin { .. } => {
                unreachable!("non-float nodes are filtered above")
            }
        }
    }
    Ok(())
}

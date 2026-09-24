//! Exact same-device decompositions shared by native preparation and autodiff.
//! These run only after state specialization has consumed attention metadata.

use crate::{AttentionRounding, AttentionWindow, Node, NodeKind, RotaryLayout};
use effect_torch_runtime::DType;
use std::sync::Arc;

type Result = std::result::Result<Arc<Node>, String>;

fn reshape(a: Arc<Node>, shape: Vec<usize>) -> Result {
    Node::new(NodeKind::Reshape { a, shape })
}

fn cast(a: Arc<Node>, dtype: DType) -> Result {
    Node::new(NodeKind::Cast { a, dtype })
}

fn scalar(like: &Node, value: f64, dtype: DType) -> Result {
    Node::new(NodeKind::Full {
        shape: vec![],
        value,
        dtype,
        device: like.device.clone(),
    })
}

fn repeat_heads(a: &Arc<Node>, heads: usize) -> Result {
    let rank = a.shape.len();
    if rank < 3 || a.shape[rank - 3] == heads {
        return Ok(a.clone());
    }
    let mut grouped = a.shape.clone();
    grouped.insert(rank - 2, 1);
    let value = reshape(a.clone(), grouped.clone())?;
    grouped[rank - 2] = heads / a.shape[rank - 3];
    let value = Node::new(NodeKind::BroadcastTo {
        a: value,
        shape: grouped,
    })?;
    let mut shape = a.shape.clone();
    shape[rank - 3] = heads;
    reshape(value, shape)
}

fn attention(
    q: &Arc<Node>,
    k: &Arc<Node>,
    v: &Arc<Node>,
    scale: f64,
    causal: bool,
    window: AttentionWindow,
) -> Result {
    let rank = q.shape.len();
    let queries = q.shape[rank - 2];
    let keys = k.shape[rank - 2];
    let heads = if rank >= 3 { q.shape[rank - 3] } else { 1 };
    let mut dims = (0..rank).collect::<Vec<_>>();
    dims.swap(rank - 2, rank - 1);
    let kt = Node::new(NodeKind::Permute {
        a: repeat_heads(k, heads)?,
        dims,
    })?;
    let mut scores = Node::new(NodeKind::Matmul {
        a: q.clone(),
        b: kt,
    })?;
    if scale != 1.0 {
        let float = cast(scores, DType::F32)?;
        let multiplier = scalar(q, scale, DType::F32)?;
        scores = cast(
            Node::new(NodeKind::Mul {
                a: float,
                b: multiplier,
            })?,
            q.dtype,
        )?;
    }
    if causal {
        let offset = keys.saturating_sub(queries);
        let rows = Node::new(NodeKind::Arange {
            start: offset as f64,
            end: (offset + queries) as f64,
            step: 1.0,
            dtype: DType::I64,
            device: q.device.clone(),
        })?;
        let columns = Node::new(NodeKind::Arange {
            start: 0.0,
            end: keys as f64,
            step: 1.0,
            dtype: DType::I64,
            device: q.device.clone(),
        })?;
        let row = reshape(rows, vec![queries, 1])?;
        let column = reshape(columns, vec![1, keys])?;
        let mut allowed = Node::new(NodeKind::Le {
            a: column.clone(),
            b: row.clone(),
        })?;
        if let Some(window) = window.local() {
            let distance = Node::new(NodeKind::Sub { a: row, b: column })?;
            let inside = Node::new(NodeKind::Lt {
                a: distance,
                b: scalar(q, window as f64, DType::I64)?,
            })?;
            allowed = Node::new(NodeKind::Mul {
                a: allowed,
                b: inside,
            })?;
        }
        let minimum = match q.dtype {
            DType::BF16 => -3.3895313892515355e38,
            DType::F16 => -65504.0,
            _ => -(f32::MAX as f64),
        };
        let mask = Node::new(NodeKind::Where {
            cond: allowed,
            a: scalar(q, 0.0, q.dtype)?,
            b: scalar(q, minimum, q.dtype)?,
        })?;
        scores = Node::new(NodeKind::Add { a: scores, b: mask })?;
    }
    let scores = cast(scores, DType::F32)?;
    let max = Node::new(NodeKind::Max {
        a: scores.clone(),
        dims: vec![rank - 1],
        keepdims: true,
    })?;
    let shifted = Node::new(NodeKind::Sub { a: scores, b: max })?;
    let exp = Node::new(NodeKind::Exp { a: shifted })?;
    let sum = Node::new(NodeKind::Sum {
        a: exp.clone(),
        dims: vec![rank - 1],
        keepdims: true,
    })?;
    let probabilities = cast(Node::new(NodeKind::Div { a: exp, b: sum })?, q.dtype)?;
    Node::new(NodeKind::Matmul {
        a: probabilities,
        b: repeat_heads(v, heads)?,
    })
}

fn rotary(
    x: &Arc<Node>,
    positions: &Arc<Node>,
    frequencies: &Arc<Node>,
    layout: RotaryLayout,
) -> Result {
    let rank = x.shape.len();
    let width = x.shape[rank - 1];
    let mut columns = positions.shape.clone();
    columns.push(1);
    let positions_float = reshape(cast(positions.clone(), DType::F32)?, columns)?;
    let angles = Node::new(NodeKind::Mul {
        a: positions_float,
        b: cast(frequencies.clone(), DType::F32)?,
    })?;
    let doubled = match layout {
        RotaryLayout::HalfSplit => Node::new(NodeKind::Concat {
            a: angles.clone(),
            b: angles.clone(),
            dim: angles.shape.len() - 1,
        })?,
        RotaryLayout::InterleavedPairs => {
            let mut shape = angles.shape.clone();
            shape.push(1);
            let columns = reshape(angles.clone(), shape.clone())?;
            *shape.last_mut().unwrap() = 2;
            let pairs = Node::new(NodeKind::BroadcastTo { a: columns, shape })?;
            let mut shape = angles.shape.clone();
            *shape.last_mut().unwrap() = width;
            reshape(pairs, shape)?
        }
    };
    let mut table_shape = positions.shape.clone();
    if rank >= 3 {
        table_shape.insert(table_shape.len() - 1, 1);
    }
    table_shape.push(width);
    let cosine = reshape(
        cast(Node::new(NodeKind::Cos { a: doubled.clone() })?, x.dtype)?,
        table_shape.clone(),
    )?;
    let sine = reshape(
        cast(Node::new(NodeKind::Sin { a: doubled })?, x.dtype)?,
        table_shape,
    )?;
    let pair_input = match layout {
        RotaryLayout::HalfSplit => x.clone(),
        RotaryLayout::InterleavedPairs => {
            let mut shape = x.shape.clone();
            *shape.last_mut().unwrap() = width / 2;
            shape.push(2);
            reshape(x.clone(), shape)?
        }
    };
    let dim = pair_input.shape.len() - 1;
    let half = pair_input.shape[dim] / 2;
    let mut first = pair_input
        .shape
        .iter()
        .map(|&size| (0, size, 1))
        .collect::<Vec<_>>();
    first[dim].1 = half;
    let mut second = first.clone();
    second[dim] = (half, half * 2, 1);
    let first = Node::new(NodeKind::Slice {
        a: pair_input.clone(),
        ranges: first,
    })?;
    let second = Node::new(NodeKind::Slice {
        a: pair_input,
        ranges: second,
    })?;
    let rotated = Node::new(NodeKind::Concat {
        a: Node::new(NodeKind::Neg { a: second })?,
        b: first,
        dim,
    })?;
    let rotated = reshape(rotated, x.shape.clone())?;
    let direct = Node::new(NodeKind::Mul {
        a: x.clone(),
        b: cosine,
    })?;
    let cross = Node::new(NodeKind::Mul {
        a: rotated,
        b: sine,
    })?;
    Node::new(NodeKind::Add {
        a: direct,
        b: cross,
    })
}

/// Decomposes one ordinary semantic operation, retaining its operand nodes.
/// Stateful KV attention is deliberately excluded. Explicit casts mark every
/// precision boundary, independent of subsequent optimization settings.
pub fn decompose_semantic(kind: &NodeKind) -> std::result::Result<Option<Arc<Node>>, String> {
    match kind {
        NodeKind::SdpaConfigured {
            q,
            k,
            v,
            scale,
            causal,
            window,
            rounding,
            ..
        } => {
            let result = match rounding {
                AttentionRounding::Fused => Node::new(NodeKind::Sdpa {
                    q: q.clone(),
                    k: k.clone(),
                    v: v.clone(),
                    scale: *scale,
                    causal: *causal,
                    window: *window,
                })?,
                AttentionRounding::Stepwise => attention(q, k, v, *scale, *causal, *window)?,
            };
            Ok(Some(result))
        }
        NodeKind::RotaryEmbeddingExplicit {
            x,
            positions,
            inverse_frequencies,
            layout,
        } => Ok(Some(rotary(x, positions, inverse_frequencies, *layout)?)),
        _ => Ok(None),
    }
}

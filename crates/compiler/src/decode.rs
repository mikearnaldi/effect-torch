//! Specializes inference graphs for stateful decode.
//!
//! [`specialize_decode`] converts training-style operations to their stateful
//! autoregressive forms. It replaces causal attention with KV-cached attention,
//! chunked KDA with a recurrence, and short convolutions with state updates. It
//! also makes rotary positions cursor-relative and rebuilds learned position
//! embeddings as cursor-indexed gathers. The function creates a new graph with
//! fresh node IDs, preserves shared subgraphs, and does not mutate the source.
//!
//! The returned [`DecodeGeometry`] tells the runtime how to allocate state.
//! Rewrite validation enforces these rules:
//!
//! - Stateful layers in each family use the same geometry. Mixed attention,
//!   KDA, or convolution geometry is an error.
//! - Layer ordinals follow the historical encounter order, with the last root
//!   first. Repeated specialization of the same graph produces the same order.
//! - The runtime provides the state cursor at `cursor_slot`, one slot after
//!   the highest caller input. It is not a caller argument. One dense graph row
//!   uses a scalar. Multiple rows use an `i64 [graph_rows]` tensor.
//! - `allows_window_eviction` is true only if every attention layer has a
//!   finite window. A global retention window smaller than an explicit local
//!   window is an error.

use effect_torch_graph::{
    node_children, remap_children, KvAttentionMode, Node, NodeKind, PositionOffset,
};
use effect_torch_runtime::DType;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Uniform KDA recurrence geometry used to size recurrent state buffers.
/// All fields are zero when the graph has no KDA layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KdaGeometry {
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub value_dim: usize,
    pub dtype: DType,
}

impl Default for KdaGeometry {
    fn default() -> Self {
        Self {
            layers: 0,
            heads: 0,
            head_dim: 0,
            value_dim: 0,
            dtype: DType::F32,
        }
    }
}

/// Uniform channel count and kernel width for short-convolution state layers.
/// All fields are zero when the graph has no such layers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ConvGeometry {
    pub layers: usize,
    pub channels: usize,
    pub kernel: usize,
}

/// Stateful decode geometry returned with the specialized roots.
///
/// `layers` is the attention layer count. `kv_heads` and `head_dim` define
/// their uniform KV-cache geometry. All three are zero when the graph has no
/// attention. `cursor_slot` and `cursor_tensor` locate the runtime state
/// cursor among the program inputs. `allows_window_eviction` says whether the
/// runtime may evict KV state outside the maximum attention window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecodeGeometry {
    pub layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub allows_window_eviction: bool,
    pub kda: KdaGeometry,
    pub conv: ConvGeometry,
    pub cursor_slot: u32,
    pub cursor_tensor: bool,
}

/// Maps physical sequence lanes to rows in the traced graph.
///
/// Dense prefill and decode use one graph row per physical lane. Packed
/// causal-chain verification traces each candidate position as a separate
/// one-token row but retains one state sequence per physical lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecodeLayout {
    Dense {
        batch: usize,
    },
    PackedCausalChains {
        batch: usize,
        rows_per_sequence: usize,
    },
}

/// Output-row policy for one decode root. Policies follow source-root order,
/// and split roots retain lane order within each source root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecodeOutputSelection {
    /// Preserve every row produced by the semantic root.
    AllRows,
    /// Emit one last-valid `[V]` output root per physical lane.
    SplitLastTokenRow,
    /// Emit one batched last-valid `[batch, V]` output root.
    BatchedLastTokenRow,
}

/// Visibility policy for rows staged by one decode invocation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum CurrentBlockAttention {
    #[default]
    Causal,
    Bidirectional,
}

impl CurrentBlockAttention {
    const fn kv_mode(self) -> KvAttentionMode {
        match self {
            Self::Causal => KvAttentionMode::Causal,
            Self::Bidirectional => KvAttentionMode::BidirectionalBlock,
        }
    }
}

impl DecodeLayout {
    pub const fn dense(batch: usize) -> Self {
        Self::Dense { batch }
    }

    pub const fn packed_causal_chains(batch: usize, rows_per_sequence: usize) -> Self {
        Self::PackedCausalChains {
            batch,
            rows_per_sequence,
        }
    }

    pub const fn batch(self) -> usize {
        match self {
            Self::Dense { batch } | Self::PackedCausalChains { batch, .. } => batch,
        }
    }

    pub fn graph_rows(self) -> Result<usize, String> {
        let (batch, rows_per_sequence) = match self {
            Self::Dense { batch } => (batch, 1),
            Self::PackedCausalChains {
                batch,
                rows_per_sequence,
            } => (batch, rows_per_sequence),
        };
        if batch == 0 || rows_per_sequence == 0 {
            return Err("decode: batch and rows per sequence must be positive".to_string());
        }
        batch
            .checked_mul(rows_per_sequence)
            .ok_or_else(|| "decode: packed causal-chain graph row count overflow".to_string())
    }
}

impl DecodeGeometry {
    /// Declares the state cursor slot as internal runtime metadata, not a
    /// caller-visible argument.
    pub const fn state_cursor(&self) -> crate::request::StateCursorSlot {
        crate::request::StateCursorSlot::new(self.cursor_slot, self.cursor_tensor)
    }
}

/// Builds the stateful decode specialization of an inference graph.
///
/// Traversal and reconstruction are iterative. Layer ordinals follow the
/// historical decode order. This order visits roots and children from last to
/// first. Returned roots remain in caller order.
///
/// With `last_token_row`, every rewritten root must be `[batch, T, V]`.
/// Batch 1 returns one inference-only `LastTokenRow` selector. Larger batches
/// return `batch` selectors per source root, each over a static one-row slice
/// in row order. The default value, `false`, returns rewritten roots unchanged.
pub fn specialize_decode(
    roots: &[Arc<Node>],
    window: Option<usize>,
    batch: usize,
    last_token_row: bool,
) -> Result<(Vec<Arc<Node>>, DecodeGeometry), String> {
    specialize_decode_layout(roots, window, DecodeLayout::dense(batch), last_token_row)
}

/// Builds a stateful decode specialization with an explicit physical-to-graph
/// row layout. Packed causal-chain roots remain all-row outputs.
pub fn specialize_decode_layout(
    roots: &[Arc<Node>],
    window: Option<usize>,
    layout: DecodeLayout,
    last_token_row: bool,
) -> Result<(Vec<Arc<Node>>, DecodeGeometry), String> {
    let output_selections = vec![
        if last_token_row {
            DecodeOutputSelection::SplitLastTokenRow
        } else {
            DecodeOutputSelection::AllRows
        };
        roots.len()
    ];
    specialize_decode_layout_outputs(roots, window, layout, &output_selections)
}

fn push_last_token_row_through_head(
    root: &Arc<Node>,
    batched: bool,
) -> Result<Option<Arc<Node>>, String> {
    fn descend(node: &Arc<Node>) -> Result<Option<Arc<Node>>, String> {
        match &node.kind {
            NodeKind::QuantizedLinear {
                x,
                weight,
                bias,
                codec,
                weight_shape,
            } if x.shape.len() == 3 && x.shape[0] == 1 => {
                let columns = x.shape[2];
                let selected = Node::new(NodeKind::LastTokenRow { a: x.clone() })?;
                let selected = Node::new(NodeKind::Reshape {
                    a: selected,
                    shape: vec![1, columns],
                })?;
                Node::new(NodeKind::QuantizedLinear {
                    x: selected,
                    weight: weight.clone(),
                    bias: bias.clone(),
                    codec: *codec,
                    weight_shape: *weight_shape,
                })
                .map(Some)
            }
            NodeKind::Linear { x, weight, bias } if x.shape.len() == 3 && x.shape[0] == 1 => {
                let columns = x.shape[2];
                let selected = Node::new(NodeKind::LastTokenRow { a: x.clone() })?;
                let selected = Node::new(NodeKind::Reshape {
                    a: selected,
                    shape: vec![1, columns],
                })?;
                Node::new(NodeKind::Linear {
                    x: selected,
                    weight: weight.clone(),
                    bias: bias.clone(),
                })
                .map(Some)
            }
            NodeKind::Tanh { a } => match descend(a)? {
                Some(a) => Node::new(NodeKind::Tanh { a }).map(Some),
                None => Ok(None),
            },
            NodeKind::Mul { a, b } | NodeKind::Div { a, b } => {
                let (sequence, scalar, sequence_is_left) = if a.shape.is_empty() {
                    (b, a, false)
                } else if b.shape.is_empty() {
                    (a, b, true)
                } else {
                    return Ok(None);
                };
                let Some(sequence) = descend(sequence)? else {
                    return Ok(None);
                };
                let (a, b) = if sequence_is_left {
                    (sequence, scalar.clone())
                } else {
                    (scalar.clone(), sequence)
                };
                let kind = match &node.kind {
                    NodeKind::Mul { .. } => NodeKind::Mul { a, b },
                    NodeKind::Div { .. } => NodeKind::Div { a, b },
                    _ => unreachable!(),
                };
                Node::new(kind).map(Some)
            }
            _ => Ok(None),
        }
    }
    let Some(pushed) = descend(root)? else {
        return Ok(None);
    };
    if pushed.shape.len() != 2 || pushed.shape[0] != 1 {
        return Ok(None);
    }
    let width = pushed.shape[1];
    Node::new(NodeKind::Reshape {
        a: pushed,
        shape: if batched { vec![1, width] } else { vec![width] },
    })
    .map(Some)
}

/// Builds a stateful decode specialization with an explicit output-row policy
/// for every source root.
pub fn specialize_decode_layout_outputs(
    roots: &[Arc<Node>],
    window: Option<usize>,
    layout: DecodeLayout,
    output_selections: &[DecodeOutputSelection],
) -> Result<(Vec<Arc<Node>>, DecodeGeometry), String> {
    specialize_decode_layout_outputs_with_attention(
        roots,
        window,
        layout,
        output_selections,
        CurrentBlockAttention::Causal,
    )
}

/// Builds a stateful decode specialization with explicit output and current
/// block attention policies.
pub fn specialize_decode_layout_outputs_with_attention(
    roots: &[Arc<Node>],
    window: Option<usize>,
    layout: DecodeLayout,
    output_selections: &[DecodeOutputSelection],
    current_block_attention: CurrentBlockAttention,
) -> Result<(Vec<Arc<Node>>, DecodeGeometry), String> {
    let graph_rows = layout.graph_rows()?;
    let batch = layout.batch();
    if output_selections.len() != roots.len() {
        return Err(format!(
            "decode: expected one output selection per root ({}), got {}",
            roots.len(),
            output_selections.len()
        ));
    }
    if matches!(layout, DecodeLayout::PackedCausalChains { .. })
        && output_selections
            .iter()
            .any(|selection| *selection != DecodeOutputSelection::AllRows)
    {
        return Err(
            "decode: packed causal-chain verification requires all-row outputs".to_string(),
        );
    }
    if matches!(layout, DecodeLayout::PackedCausalChains { .. })
        && current_block_attention == CurrentBlockAttention::Bidirectional
    {
        return Err(
            "decode: packed causal-chain verification requires causal current-block attention"
                .to_string(),
        );
    }
    let mut maximum_slot = None;
    let mut visited = HashSet::new();
    let mut stack = roots.to_vec();
    while let Some(node) = stack.pop() {
        if !visited.insert(node.id) {
            continue;
        }
        match &node.kind {
            NodeKind::Input { slot, .. } => {
                maximum_slot = Some(maximum_slot.map_or(*slot, |current: u32| current.max(*slot)));
            }
            NodeKind::ScalarInput { .. } => {
                return Err(
                    "decode: runtime scalar inputs are not supported in inference graphs"
                        .to_string(),
                );
            }
            _ => {}
        }
        stack.extend(node_children(&node.kind));
    }
    let cursor_slot = maximum_slot.map_or(0, |slot| slot + 1);

    let mut visited = HashSet::new();
    let mut order = Vec::new();
    let mut stack = roots
        .iter()
        .map(|root| (root.clone(), false))
        .collect::<Vec<_>>();
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

    let mut remapped = HashMap::new();
    let mut layers = 0usize;
    let mut kda_layers = 0usize;
    let mut conv_layers = 0usize;
    let mut cursor_tensor = false;
    let mut geometry: Option<(usize, usize)> = None;
    let mut allows_window_eviction = true;
    let mut maximum_attention_window = None;
    let mut kda_geometry: Option<(usize, usize, usize, DType)> = None;
    let mut conv_geometry: Option<(usize, usize)> = None;
    for node in order {
        let remap = |child: &Arc<Node>| {
            remapped
                .get(&child.id)
                .cloned()
                .unwrap_or_else(|| child.clone())
        };
        let kind = match &node.kind {
            NodeKind::Sdpa {
                q,
                k,
                v,
                scale,
                causal,
                window: attention_window,
            } => {
                if !causal && current_block_attention != CurrentBlockAttention::Bidirectional {
                    return Err(
                        "decode: only causal attention is cacheable, found a non-causal sdpa"
                            .to_string(),
                    );
                }
                let rank = k.shape.len();
                if rank != 4 || k.shape[..rank - 3].iter().product::<usize>() != graph_rows {
                    return Err(format!(
                        "decode: kv caching expects attention of shape [{graph_rows}, H, T, D], got {:?}",
                        k.shape
                    ));
                }
                let current = (k.shape[rank - 3], k.shape[rank - 1]);
                if let Some(previous) = geometry {
                    if previous != current {
                        if k.device.is_metal() {
                            return Err(format!(
                                "decode: attention layers disagree on head geometry ([{}, {}] vs [{}, {}])",
                                previous.0, previous.1, current.0, current.1
                            ));
                        }
                        return Err(format!(
                            "decode: attention layers disagree on head geometry ({previous:?} vs {current:?})"
                        ));
                    }
                } else {
                    geometry = Some(current);
                }
                let layer = layers;
                layers += 1;
                let resolved_window = attention_window.resolve(window);
                allows_window_eviction &= resolved_window.is_some();
                if let Some(resolved_window) = resolved_window {
                    maximum_attention_window = Some(
                        maximum_attention_window.map_or(resolved_window, |current: usize| {
                            current.max(resolved_window)
                        }),
                    );
                }
                NodeKind::KvAttention {
                    q: remap(q),
                    k: remap(k),
                    v: remap(v),
                    scale: *scale,
                    layer: layer as u32,
                    window: resolved_window,
                    mode: current_block_attention.kv_mode(),
                }
            }
            NodeKind::KdaChunk {
                q,
                k,
                v,
                log_decay,
                beta,
                scale,
            } => {
                let rank = q.shape.len();
                if rank != 4 || q.shape[..rank - 3].iter().product::<usize>() != graph_rows {
                    return Err(format!(
                        "decode: kda state caching expects layers of shape [{graph_rows}, H, T, D], got {:?}",
                        q.shape
                    ));
                }
                if q.device.is_cpu() && !matches!(q.dtype, DType::F32 | DType::F64) {
                    return Err(format!(
                        "decode: stateful KDA requires f32 or f64, got {}",
                        q.dtype.name()
                    ));
                }
                let current = (
                    q.shape[rank - 3],
                    q.shape[rank - 1],
                    v.shape[rank - 1],
                    q.dtype,
                );
                if let Some(previous) = kda_geometry {
                    let dimensions_disagree = previous.0 != current.0
                        || previous.1 != current.1
                        || previous.2 != current.2;
                    let dtype_disagrees = q.device.is_cpu() && previous.3 != current.3;
                    if dimensions_disagree || dtype_disagrees {
                        if q.device.is_metal() {
                            let previous = (previous.0, previous.1, previous.2);
                            let current = (current.0, current.1, current.2);
                            return Err(format!(
                                "decode: kda layers disagree on head geometry ({previous:?} vs {current:?})"
                            ));
                        }
                        return Err(format!(
                            "decode: kda layers disagree on head geometry ({previous:?} vs {current:?})"
                        ));
                    }
                } else {
                    kda_geometry = Some(current);
                }
                let layer = kda_layers;
                kda_layers += 1;
                NodeKind::KdaRecurrence {
                    q: remap(q),
                    k: remap(k),
                    v: remap(v),
                    log_decay: remap(log_decay),
                    beta: remap(beta),
                    scale: *scale,
                    layer: layer as u32,
                }
            }
            NodeKind::ShortConv1d { x, weight } => {
                let rank = x.shape.len();
                if rank != 3 || x.shape[..rank - 2].iter().product::<usize>() != graph_rows {
                    return Err(format!(
                        "decode: conv state caching expects layers of shape [{graph_rows}, T, C], got {:?}",
                        x.shape
                    ));
                }
                let current = (x.shape[rank - 1], weight.shape[1]);
                if let Some(previous) = conv_geometry {
                    if previous != current {
                        return Err(format!(
                            "decode: short conv layers disagree on geometry ({previous:?} vs {current:?})"
                        ));
                    }
                } else {
                    conv_geometry = Some(current);
                }
                let layer = conv_layers;
                conv_layers += 1;
                NodeKind::ConvState {
                    x: remap(x),
                    weight: remap(weight),
                    layer: layer as u32,
                }
            }
            NodeKind::RotaryEmbedding {
                x,
                seq_len,
                theta,
                layout,
                ..
            } => NodeKind::RotaryEmbedding {
                x: remap(x),
                seq_len: *seq_len,
                theta: *theta,
                offset: PositionOffset::Cursor,
                layout: *layout,
            },
            NodeKind::PositionEmbedding { weight, seq_len } => {
                let tokens = *seq_len;
                let width = weight.shape[1];
                let device = weight.device.clone();
                if graph_rows > 1 {
                    cursor_tensor = true;
                    let cursors = Node::new(NodeKind::Input {
                        slot: cursor_slot,
                        shape: vec![graph_rows],
                        dtype: DType::I64,
                        device: device.clone(),
                    })?;
                    let positions = Node::new(NodeKind::Add {
                        a: Node::new(NodeKind::Reshape {
                            a: cursors,
                            shape: vec![graph_rows, 1],
                        })?,
                        b: Node::new(NodeKind::BroadcastTo {
                            a: Node::new(NodeKind::Reshape {
                                a: Node::new(NodeKind::Arange {
                                    start: 0.0,
                                    end: tokens as f64,
                                    step: 1.0,
                                    dtype: DType::I64,
                                    device: device.clone(),
                                })?,
                                shape: vec![1, tokens],
                            })?,
                            shape: vec![graph_rows, tokens],
                        })?,
                    })?;
                    let indexes = Node::new(NodeKind::BroadcastTo {
                        a: Node::new(NodeKind::Reshape {
                            a: positions,
                            shape: vec![graph_rows * tokens, 1],
                        })?,
                        shape: vec![graph_rows * tokens, width],
                    })?;
                    NodeKind::Reshape {
                        a: Node::new(NodeKind::Gather {
                            a: remap(weight),
                            dim: 0,
                            indexes,
                        })?,
                        shape: vec![graph_rows, tokens, width],
                    }
                } else {
                    let positions = Node::new(NodeKind::Add {
                        a: Node::new(NodeKind::Arange {
                            start: 0.0,
                            end: tokens as f64,
                            step: 1.0,
                            dtype: DType::I64,
                            device: device.clone(),
                        })?,
                        b: Node::new(NodeKind::ScalarInput {
                            slot: cursor_slot,
                            dtype: DType::I64,
                            device,
                        })?,
                    })?;
                    let indexes = Node::new(NodeKind::BroadcastTo {
                        a: Node::new(NodeKind::Reshape {
                            a: positions,
                            shape: vec![tokens, 1],
                        })?,
                        shape: vec![tokens, width],
                    })?;
                    NodeKind::Gather {
                        a: remap(weight),
                        dim: 0,
                        indexes,
                    }
                }
            }
            kind => remap_children(kind, &remap),
        };
        remapped.insert(node.id, Node::new(kind)?);
    }

    let (kv_heads, head_dim) = geometry.unwrap_or((0, 0));
    let kda = kda_geometry
        .map(|(heads, head_dim, value_dim, dtype)| KdaGeometry {
            layers: kda_layers,
            heads,
            head_dim,
            value_dim,
            dtype,
        })
        .unwrap_or_default();
    let conv = conv_geometry
        .map(|(channels, kernel)| ConvGeometry {
            layers: conv_layers,
            channels,
            kernel,
        })
        .unwrap_or_default();
    if allows_window_eviction {
        if let (Some(retained), Some(required)) = (window, maximum_attention_window) {
            if retained < required {
                return Err(format!(
                    "decode: global attention window {retained} cannot retain explicit local window {required}"
                ));
            }
        }
    }
    let roots = roots
        .iter()
        .map(|root| {
            remapped
                .get(&root.id)
                .cloned()
                .unwrap_or_else(|| root.clone())
        })
        .collect::<Vec<_>>();
    if matches!(layout, DecodeLayout::PackedCausalChains { .. }) {
        for root in &roots {
            if root.shape.len() != 3 || root.shape[0] != graph_rows || root.shape[1] != 1 {
                return Err(format!(
                    "decode: packed causal-chain roots must be [{graph_rows}, 1, V], got {:?}",
                    root.shape
                ));
            }
        }
    }
    let mut selected = Vec::new();
    for (root, selection) in roots.iter().zip(output_selections) {
        match selection {
            DecodeOutputSelection::AllRows => selected.push(root.clone()),
            DecodeOutputSelection::SplitLastTokenRow
            | DecodeOutputSelection::BatchedLastTokenRow => {
                if root.shape.len() != 3 || root.shape[0] != batch {
                    return Err(format!(
                        "decode: last-token-row roots must be [{batch}, T, V], got {:?}",
                        root.shape
                    ));
                }
                let (tokens, width) = (root.shape[1], root.shape[2]);
                if batch == 1 {
                    if let Some(pushed) = push_last_token_row_through_head(
                        root,
                        *selection == DecodeOutputSelection::BatchedLastTokenRow,
                    )? {
                        selected.push(pushed);
                        continue;
                    }
                }
                let mut rows = Vec::with_capacity(batch);
                for row in 0..batch {
                    let source = if batch == 1 {
                        root.clone()
                    } else {
                        Node::new(NodeKind::Slice {
                            a: root.clone(),
                            ranges: vec![(row, row + 1, 1), (0, tokens, 1), (0, width, 1)],
                        })?
                    };
                    rows.push(Node::new(NodeKind::LastTokenRow { a: source })?);
                }
                if *selection == DecodeOutputSelection::SplitLastTokenRow {
                    selected.extend(rows);
                } else {
                    let mut rows = rows.into_iter();
                    let first = rows.next().expect("decode batch is positive");
                    let mut batched = Node::new(NodeKind::Reshape {
                        a: first,
                        shape: vec![1, width],
                    })?;
                    for row in rows {
                        let row = Node::new(NodeKind::Reshape {
                            a: row,
                            shape: vec![1, width],
                        })?;
                        batched = Node::new(NodeKind::Concat {
                            a: batched,
                            b: row,
                            dim: 0,
                        })?;
                    }
                    selected.push(batched);
                }
            }
        }
    }
    Ok((
        selected,
        DecodeGeometry {
            layers,
            kv_heads,
            head_dim,
            allows_window_eviction,
            kda,
            conv,
            cursor_slot,
            cursor_tensor,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::graph_post_order;
    use effect_torch_graph::{AttentionWindow, Device, RotaryLayout};

    fn tensor(shape: &[usize], dtype: DType, device: Device) -> Arc<Node> {
        Node::new(NodeKind::Zeros {
            shape: shape.to_vec(),
            dtype,
            device,
        })
        .unwrap()
    }

    fn input(slot: u32, shape: &[usize], dtype: DType, device: Device) -> Arc<Node> {
        Node::new(NodeKind::Input {
            slot,
            shape: shape.to_vec(),
            dtype,
            device,
        })
        .unwrap()
    }

    fn attention(
        batch: usize,
        heads: usize,
        tokens: usize,
        dim: usize,
        scale: f64,
        causal: bool,
        device: Device,
    ) -> Arc<Node> {
        let operand = tensor(&[batch, heads, tokens, dim], DType::F32, device);
        Node::new(NodeKind::Sdpa {
            q: operand.clone(),
            k: operand.clone(),
            v: operand,
            scale,
            causal,
            window: AttentionWindow::Inherit,
        })
        .unwrap()
    }

    fn kda(
        batch: usize,
        heads: usize,
        tokens: usize,
        key_dim: usize,
        value_dim: usize,
        dtype: DType,
        scale: f64,
        device: Device,
    ) -> Arc<Node> {
        let q = tensor(&[batch, heads, tokens, key_dim], dtype, device.clone());
        let v = tensor(&[batch, heads, tokens, value_dim], dtype, device.clone());
        Node::new(NodeKind::KdaChunk {
            q: q.clone(),
            k: q.clone(),
            v,
            log_decay: q,
            beta: tensor(&[batch, heads, tokens, 1], dtype, device),
            scale,
        })
        .unwrap()
    }

    fn short_conv(
        batch: usize,
        tokens: usize,
        channels: usize,
        kernel: usize,
        device: Device,
    ) -> Arc<Node> {
        Node::new(NodeKind::ShortConv1d {
            x: tensor(&[batch, tokens, channels], DType::F32, device.clone()),
            weight: tensor(&[channels, kernel], DType::F32, device),
        })
        .unwrap()
    }

    fn layer(kind: &NodeKind) -> u32 {
        match kind {
            NodeKind::KvAttention { layer, .. }
            | NodeKind::KdaRecurrence { layer, .. }
            | NodeKind::ConvState { layer, .. } => *layer,
            _ => panic!("expected a stateful decode node"),
        }
    }

    #[test]
    fn specializes_all_stateful_semantics_in_one_new_graph_generation() {
        for device in [Device::Cpu(0), Device::Metal(0)] {
            let shared = tensor(&[1, 1, 2, 4], DType::F32, device.clone());
            let attention = Node::new(NodeKind::Sdpa {
                q: shared.clone(),
                k: shared.clone(),
                v: shared.clone(),
                scale: 0.5,
                causal: true,
                window: AttentionWindow::Inherit,
            })
            .unwrap();
            let rotary = Node::new(NodeKind::RotaryEmbedding {
                x: shared,
                seq_len: 2,
                theta: 10_000.0,
                offset: PositionOffset::Absolute,
                layout: RotaryLayout::HalfSplit,
            })
            .unwrap();
            let kda = kda(1, 2, 2, 3, 5, DType::F32, 0.25, device.clone());
            let conv = short_conv(1, 2, 5, 3, device);
            let roots = vec![attention, rotary, kda, conv];
            let source = graph_post_order(&roots);
            let source_ids = source.iter().map(|node| node.id).collect::<HashSet<_>>();

            let (specialized, geometry) = specialize_decode(&roots, Some(32), 1, false).unwrap();

            assert!(matches!(
                specialized[0].kind,
                NodeKind::KvAttention {
                    scale: 0.5,
                    layer: 0,
                    window: Some(32),
                    ..
                }
            ));
            assert!(matches!(
                specialized[1].kind,
                NodeKind::RotaryEmbedding {
                    offset: PositionOffset::Cursor,
                    ..
                }
            ));
            assert!(matches!(
                specialized[2].kind,
                NodeKind::KdaRecurrence {
                    scale: 0.25,
                    layer: 0,
                    ..
                }
            ));
            assert!(matches!(
                specialized[3].kind,
                NodeKind::ConvState { layer: 0, .. }
            ));
            let NodeKind::KvAttention { q, .. } = &specialized[0].kind else {
                unreachable!()
            };
            let NodeKind::RotaryEmbedding { x, .. } = &specialized[1].kind else {
                unreachable!()
            };
            assert!(Arc::ptr_eq(q, x), "shared source remains shared");
            let generated = graph_post_order(&specialized);
            assert_eq!(generated.len(), source.len());
            assert!(generated.iter().all(|node| !source_ids.contains(&node.id)));
            assert_eq!(geometry.layers, 1);
            assert_eq!((geometry.kv_heads, geometry.head_dim), (1, 4));
            assert!(geometry.allows_window_eviction);
            assert_eq!(geometry.kda.layers, 1);
            assert_eq!(geometry.conv.layers, 1);
        }
    }

    #[test]
    fn bidirectional_mode_specializes_non_causal_attention_and_rejects_packed_layout() {
        for device in [Device::Cpu(0), Device::Metal(0)] {
            let causal = attention(1, 2, 3, 4, 0.5, true, device.clone());
            let (causal_specialized, _) = specialize_decode_layout_outputs(
                &[causal],
                None,
                DecodeLayout::dense(1),
                &[DecodeOutputSelection::AllRows],
            )
            .unwrap();
            assert!(matches!(
                causal_specialized[0].kind,
                NodeKind::KvAttention {
                    mode: KvAttentionMode::Causal,
                    ..
                }
            ));

            let root = attention(1, 2, 3, 4, 0.5, false, device);
            let causal_error = specialize_decode_layout_outputs(
                std::slice::from_ref(&root),
                None,
                DecodeLayout::dense(1),
                &[DecodeOutputSelection::AllRows],
            )
            .err()
            .expect("causal mode must reject non-causal SDPA");
            assert!(causal_error.contains("non-causal sdpa"));

            let (specialized, _) = specialize_decode_layout_outputs_with_attention(
                std::slice::from_ref(&root),
                None,
                DecodeLayout::dense(1),
                &[DecodeOutputSelection::AllRows],
                CurrentBlockAttention::Bidirectional,
            )
            .unwrap();
            assert!(matches!(
                specialized[0].kind,
                NodeKind::KvAttention {
                    mode: KvAttentionMode::BidirectionalBlock,
                    ..
                }
            ));

            let packed_error = specialize_decode_layout_outputs_with_attention(
                &[root],
                None,
                DecodeLayout::packed_causal_chains(1, 1),
                &[DecodeOutputSelection::AllRows],
                CurrentBlockAttention::Bidirectional,
            )
            .err()
            .expect("packed layouts must reject bidirectional current blocks");
            assert!(packed_error.contains("requires causal current-block attention"));
        }
    }

    #[test]
    fn learned_positions_build_scalar_and_batched_cursor_graphs() {
        for (device, batch) in [
            (Device::Cpu(0), 1usize),
            (Device::Cpu(0), 3usize),
            (Device::Metal(0), 1usize),
            (Device::Metal(0), 3usize),
        ] {
            let q = input(0, &[batch, 1, 2, 4], DType::F32, device.clone());
            let k = input(1, &[batch, 1, 2, 4], DType::F32, device.clone());
            let v = input(2, &[batch, 1, 2, 4], DType::F32, device.clone());
            let attention = Node::new(NodeKind::Sdpa {
                q,
                k,
                v,
                scale: 0.5,
                causal: true,
                window: AttentionWindow::Inherit,
            })
            .unwrap();
            let positions = Node::new(NodeKind::PositionEmbedding {
                weight: tensor(&[128, 6], DType::F32, device.clone()),
                seq_len: 2,
            })
            .unwrap();

            let (roots, geometry) =
                specialize_decode(&[attention, positions], None, batch, false).unwrap();

            assert_eq!(geometry.cursor_slot, 3);
            assert_eq!(geometry.cursor_tensor, batch > 1);
            let cursor_nodes = graph_post_order(&roots)
                .into_iter()
                .filter(|node| match &node.kind {
                    NodeKind::Input { slot, .. } | NodeKind::ScalarInput { slot, .. } => {
                        *slot == geometry.cursor_slot
                    }
                    _ => false,
                })
                .collect::<Vec<_>>();
            assert_eq!(cursor_nodes.len(), 1);
            if batch == 1 {
                assert!(matches!(
                    cursor_nodes[0].kind,
                    NodeKind::ScalarInput {
                        dtype: DType::I64,
                        ..
                    }
                ));
                assert!(matches!(roots[1].kind, NodeKind::Gather { .. }));
                assert_eq!(roots[1].shape, [2, 6]);
            } else {
                assert!(matches!(
                    &cursor_nodes[0].kind,
                    NodeKind::Input {
                        shape,
                        dtype: DType::I64,
                        ..
                    } if shape == &[batch]
                ));
                assert!(matches!(roots[1].kind, NodeKind::Reshape { .. }));
                assert_eq!(roots[1].shape, [batch, 2, 6]);
            }
            assert!(cursor_nodes[0].device.same_device(&device));
        }
    }

    #[test]
    fn geometry_and_dtype_validation_preserve_backend_errors() {
        let cpu_attention = [
            attention(1, 2, 1, 4, 1.0, true, Device::Cpu(0)),
            attention(1, 3, 1, 4, 1.0, true, Device::Cpu(0)),
        ];
        assert_eq!(
            specialize_decode(&cpu_attention, None, 1, false)
                .err()
                .unwrap(),
            "decode: attention layers disagree on head geometry ((3, 4) vs (2, 4))"
        );
        let metal_attention = [
            attention(1, 2, 1, 4, 1.0, true, Device::Metal(0)),
            attention(1, 3, 1, 4, 1.0, true, Device::Metal(0)),
        ];
        assert_eq!(
            specialize_decode(&metal_attention, None, 1, false)
                .err()
                .unwrap(),
            "decode: attention layers disagree on head geometry ([3, 4] vs [2, 4])"
        );
        let cpu_kda = [
            kda(1, 2, 1, 4, 3, DType::F32, 1.0, Device::Cpu(0)),
            kda(1, 3, 1, 4, 3, DType::F32, 1.0, Device::Cpu(0)),
        ];
        assert_eq!(
            specialize_decode(&cpu_kda, None, 1, false).err().unwrap(),
            "decode: kda layers disagree on head geometry ((3, 4, 3, F32) vs (2, 4, 3, F32))"
        );
        let conv = [
            short_conv(1, 1, 3, 2, Device::Cpu(0)),
            short_conv(1, 1, 4, 2, Device::Cpu(0)),
            attention(1, 1, 1, 2, 1.0, true, Device::Cpu(0)),
        ];
        assert_eq!(
            specialize_decode(&conv, None, 1, false).err().unwrap(),
            "decode: short conv layers disagree on geometry ((4, 2) vs (3, 2))"
        );

        let malformed_kda = kda(1, 1, 1, 2, 2, DType::F32, 1.0, Device::Cpu(0));
        let NodeKind::KdaChunk {
            q,
            k,
            v,
            log_decay,
            beta,
            scale,
        } = &malformed_kda.kind
        else {
            unreachable!()
        };
        let malformed_kda = Node::new(NodeKind::KdaChunk {
            q: Node::new(NodeKind::Reshape {
                a: q.clone(),
                shape: vec![1, 1, 2],
            })
            .unwrap(),
            k: Node::new(NodeKind::Reshape {
                a: k.clone(),
                shape: vec![1, 1, 2],
            })
            .unwrap(),
            v: Node::new(NodeKind::Reshape {
                a: v.clone(),
                shape: vec![1, 1, 2],
            })
            .unwrap(),
            log_decay: Node::new(NodeKind::Reshape {
                a: log_decay.clone(),
                shape: vec![1, 1, 2],
            })
            .unwrap(),
            beta: Node::new(NodeKind::Reshape {
                a: beta.clone(),
                shape: vec![1, 1, 1],
            })
            .unwrap(),
            scale: *scale,
        })
        .unwrap();
        assert_eq!(
            specialize_decode(&[malformed_kda], None, 1, false)
                .err()
                .unwrap(),
            "decode: kda state caching expects layers of shape [1, H, T, D], got [1, 1, 2]"
        );
        assert_eq!(
            specialize_decode(&[short_conv(1, 2, 3, 2, Device::Cpu(0))], None, 2, false)
                .err()
                .unwrap(),
            "decode: conv state caching expects layers of shape [2, T, C], got [1, 2, 3]"
        );
        assert_eq!(
            specialize_decode(
                &[attention(1, 1, 1, 2, 1.0, false, Device::Cpu(0))],
                None,
                1,
                false
            )
            .err()
            .unwrap(),
            "decode: only causal attention is cacheable, found a non-causal sdpa"
        );
        assert_eq!(
            specialize_decode(
                &[kda(1, 1, 1, 2, 2, DType::BF16, 1.0, Device::Cpu(0))],
                None,
                1,
                false
            )
            .err()
            .unwrap(),
            "decode: stateful KDA requires f32 or f64, got bf16"
        );
        let scalar = Node::new(NodeKind::ScalarInput {
            slot: 0,
            dtype: DType::I64,
            device: Device::Cpu(0),
        })
        .unwrap();
        assert_eq!(
            specialize_decode(&[scalar], None, 1, false).err().unwrap(),
            "decode: runtime scalar inputs are not supported in inference graphs"
        );
        let (stateless, stateless_geometry) =
            specialize_decode(&[tensor(&[1], DType::F32, Device::Cpu(0))], None, 1, false).unwrap();
        assert_eq!(stateless.len(), 1);
        assert_eq!(stateless_geometry.layers, 0);
        assert_eq!(stateless_geometry.kda.layers, 0);
        assert_eq!(stateless_geometry.conv.layers, 0);

        let (_, conv_geometry) =
            specialize_decode(&[short_conv(1, 2, 3, 2, Device::Cpu(0))], None, 1, false).unwrap();
        assert_eq!(conv_geometry.layers, 0);
        assert_eq!(conv_geometry.kda.layers, 0);
        assert_eq!(conv_geometry.conv.layers, 1);

        let (_, cpu_geometry) = specialize_decode(
            &[kda(1, 1, 1, 2, 3, DType::F64, 1.0, Device::Cpu(0))],
            None,
            1,
            false,
        )
        .unwrap();
        assert_eq!(cpu_geometry.kda.dtype, DType::F64);
        for dtype in [DType::F32, DType::BF16] {
            let (_, metal_geometry) = specialize_decode(
                &[kda(1, 1, 1, 2, 3, dtype, 1.0, Device::Metal(0))],
                None,
                1,
                false,
            )
            .unwrap();
            assert_eq!(metal_geometry.kda.dtype, dtype);
        }
        let (_, mixed_metal_geometry) = specialize_decode(
            &[
                kda(1, 1, 1, 2, 3, DType::F32, 1.0, Device::Metal(0)),
                kda(1, 1, 1, 2, 3, DType::BF16, 1.0, Device::Metal(0)),
            ],
            None,
            1,
            false,
        )
        .unwrap();
        assert_eq!(mixed_metal_geometry.kda.layers, 2);
        assert_eq!(mixed_metal_geometry.kda.dtype, DType::BF16);
        assert!(Node::new(NodeKind::Zeros {
            shape: vec![1, 1, 1, 2],
            dtype: DType::F64,
            device: Device::Metal(0),
        })
        .is_err());
    }

    #[test]
    fn root_order_and_layer_numbering_are_stable() {
        let roots = vec![
            attention(1, 2, 1, 4, 1.0, true, Device::Cpu(0)),
            attention(1, 2, 1, 4, 2.0, true, Device::Cpu(0)),
            kda(1, 2, 1, 4, 3, DType::F32, 3.0, Device::Cpu(0)),
            kda(1, 2, 1, 4, 3, DType::F32, 4.0, Device::Cpu(0)),
            short_conv(1, 1, 3, 2, Device::Cpu(0)),
            short_conv(1, 1, 3, 2, Device::Cpu(0)),
        ];

        for _ in 0..2 {
            let (specialized, geometry) = specialize_decode(&roots, Some(7), 1, false).unwrap();
            assert_eq!(specialized.len(), roots.len());
            assert_eq!(
                specialized
                    .iter()
                    .map(|node| layer(&node.kind))
                    .collect::<Vec<_>>(),
                [1, 0, 1, 0, 1, 0]
            );
            assert!(matches!(
                specialized[0].kind,
                NodeKind::KvAttention { scale: 1.0, .. }
            ));
            assert!(matches!(
                specialized[1].kind,
                NodeKind::KvAttention { scale: 2.0, .. }
            ));
            assert!(matches!(
                specialized[2].kind,
                NodeKind::KdaRecurrence { scale: 3.0, .. }
            ));
            assert!(matches!(
                specialized[3].kind,
                NodeKind::KdaRecurrence { scale: 4.0, .. }
            ));
            assert_eq!(geometry.layers, 2);
            assert_eq!(geometry.kda.layers, 2);
            assert_eq!(geometry.conv.layers, 2);
        }
    }

    #[test]
    fn resolved_attention_windows_determine_retention_policy() {
        let make = |window| {
            Node::new(NodeKind::Sdpa {
                q: tensor(&[1, 4, 1, 2], DType::F32, Device::Cpu(0)),
                k: tensor(&[1, 2, 1, 2], DType::F32, Device::Cpu(0)),
                v: tensor(&[1, 2, 1, 2], DType::F32, Device::Cpu(0)),
                scale: 1.0,
                causal: true,
                window,
            })
            .unwrap()
        };
        let windows = |specialized: &[Arc<Node>]| {
            specialized
                .iter()
                .map(|node| match node.kind {
                    NodeKind::KvAttention { window, .. } => window,
                    _ => panic!("expected KV attention"),
                })
                .collect::<Vec<_>>()
        };

        let (specialized, geometry) = specialize_decode(
            &[
                make(AttentionWindow::Inherit),
                make(AttentionWindow::Local(3)),
            ],
            Some(11),
            1,
            false,
        )
        .unwrap();
        assert_eq!(windows(&specialized), [Some(11), Some(3)]);
        assert!(geometry.allows_window_eviction);
        assert_eq!((geometry.kv_heads, geometry.head_dim), (2, 2));

        let (specialized, geometry) = specialize_decode(
            &[make(AttentionWindow::Local(3)), make(AttentionWindow::Full)],
            Some(11),
            1,
            false,
        )
        .unwrap();
        assert_eq!(windows(&specialized), [Some(3), None]);
        assert!(!geometry.allows_window_eviction);

        let error = match specialize_decode(&[make(AttentionWindow::Local(13))], Some(11), 1, false)
        {
            Ok(_) => panic!("expected an unsafe retention window to fail"),
            Err(error) => error,
        };
        assert!(error.contains("cannot retain explicit local window 13"));
    }

    #[test]
    fn last_token_row_moves_through_scalar_head_epilogue_before_linear() {
        let x = input(0, &[1, 4, 256], DType::F32, Device::Cpu(0));
        let weight = tensor(&[256, 4], DType::F32, Device::Cpu(0));
        let bias = tensor(&[4], DType::F32, Device::Cpu(0));
        let scalar = tensor(&[], DType::F32, Device::Cpu(0));
        let head = Node::new(NodeKind::Linear { x, weight, bias }).unwrap();
        let scaled = Node::new(NodeKind::Mul {
            a: head,
            b: scalar.clone(),
        })
        .unwrap();
        let divided = Node::new(NodeKind::Div {
            a: scaled,
            b: scalar.clone(),
        })
        .unwrap();
        let activated = Node::new(NodeKind::Tanh { a: divided }).unwrap();
        let root = Node::new(NodeKind::Mul {
            a: activated,
            b: scalar,
        })
        .unwrap();

        let (roots, _) = specialize_decode(&[root], None, 1, true).unwrap();
        assert_eq!(roots[0].shape, [4]);
        let mut stack = vec![roots[0].clone()];
        let mut found_selector = false;
        let mut head_input_shape = None;
        while let Some(node) = stack.pop() {
            match &node.kind {
                NodeKind::LastTokenRow { .. } => found_selector = true,
                NodeKind::Linear { x, .. } => head_input_shape = Some(x.shape.clone()),
                _ => {}
            }
            stack.extend(node_children(&node.kind));
        }
        assert!(found_selector);
        assert_eq!(head_input_shape.as_deref(), Some(&[1, 256][..]));
    }

    #[test]
    fn last_token_row_rewrites_roots_into_state_driven_selectors() {
        for device in [Device::Cpu(0), Device::Metal(0)] {
            let single = input(0, &[1, 4, 8], DType::F32, device.clone());
            let (roots, _) = specialize_decode(&[single], None, 1, true).unwrap();
            assert_eq!(roots.len(), 1);
            let NodeKind::LastTokenRow { a } = &roots[0].kind else {
                panic!("batch 1 rewrites the root into one last-token-row selector")
            };
            assert_eq!(roots[0].shape, [8]);
            assert_eq!(a.shape, [1, 4, 8]);
            assert!(roots[0].device.same_device(&device));

            let batched = input(0, &[3, 4, 8], DType::F32, device.clone());
            let (roots, _) = specialize_decode(&[batched], None, 3, true).unwrap();
            assert_eq!(roots.len(), 3);
            for (row, root) in roots.iter().enumerate() {
                assert_eq!(root.shape, [8]);
                let NodeKind::LastTokenRow { a } = &root.kind else {
                    panic!("batched roots are last-token-row selectors")
                };
                let NodeKind::Slice { a: source, ranges } = &a.kind else {
                    panic!("batched selectors read a static one-row slice")
                };
                assert_eq!(source.shape, [3, 4, 8]);
                assert_eq!(ranges.as_slice(), [(row, row + 1, 1), (0, 4, 1), (0, 8, 1)]);
            }

            let stateful = attention(1, 2, 4, 2, 1.0, true, device.clone());
            let logits = Node::new(NodeKind::Reshape {
                a: stateful,
                shape: vec![1, 4, 4],
            })
            .unwrap();
            let (roots, geometry) = specialize_decode(&[logits], None, 1, true).unwrap();
            assert_eq!(geometry.layers, 1);
            assert!(matches!(roots[0].kind, NodeKind::LastTokenRow { .. }));

            let malformed = input(0, &[1, 8], DType::F32, device.clone());
            assert_eq!(
                specialize_decode(&[malformed], None, 1, true)
                    .err()
                    .unwrap(),
                "decode: last-token-row roots must be [1, T, V], got [1, 8]"
            );
            let wrong_batch = input(0, &[2, 4, 8], DType::F32, device);
            assert_eq!(
                specialize_decode(&[wrong_batch], None, 1, true)
                    .err()
                    .unwrap(),
                "decode: last-token-row roots must be [1, T, V], got [2, 4, 8]"
            );
        }
    }

    #[test]
    fn per_root_output_selection_keeps_mapping_and_batches_hidden_rows() {
        for device in [Device::Cpu(0), Device::Metal(0)] {
            let logits = input(0, &[2, 4, 8], DType::F32, device.clone());
            let hidden = input(1, &[2, 4, 6], DType::F32, device);
            let (roots, _) = specialize_decode_layout_outputs(
                &[logits, hidden],
                None,
                DecodeLayout::dense(2),
                &[
                    DecodeOutputSelection::SplitLastTokenRow,
                    DecodeOutputSelection::BatchedLastTokenRow,
                ],
            )
            .unwrap();

            assert_eq!(roots.len(), 3);
            assert_eq!(roots[0].shape, [8]);
            assert_eq!(roots[1].shape, [8]);
            assert_eq!(roots[2].shape, [2, 6]);
            assert!(matches!(roots[0].kind, NodeKind::LastTokenRow { .. }));
            assert!(matches!(roots[1].kind, NodeKind::LastTokenRow { .. }));
            assert!(matches!(roots[2].kind, NodeKind::Concat { .. }));
        }

        let root = input(0, &[2, 1, 8], DType::F32, Device::Cpu(0));
        let error = specialize_decode_layout_outputs(
            std::slice::from_ref(&root),
            None,
            DecodeLayout::dense(2),
            &[],
        )
        .err()
        .expect("selection count must be validated");
        assert!(error.contains("one output selection per root"));
        let error = specialize_decode_layout_outputs(
            &[root],
            None,
            DecodeLayout::packed_causal_chains(1, 2),
            &[DecodeOutputSelection::BatchedLastTokenRow],
        )
        .err()
        .expect("packed roots must retain all rows");
        assert_eq!(
            error,
            "decode: packed causal-chain verification requires all-row outputs"
        );
    }

    #[test]
    fn packed_causal_chains_separate_physical_batch_from_graph_rows() {
        for device in [Device::Cpu(0), Device::Metal(0)] {
            let layout = DecodeLayout::packed_causal_chains(2, 3);
            assert_eq!(layout.batch(), 2);
            assert_eq!(layout.graph_rows().unwrap(), 6);

            let stateful = attention(6, 2, 1, 4, 1.0, true, device.clone());
            let logits = Node::new(NodeKind::Reshape {
                a: stateful,
                shape: vec![6, 1, 8],
            })
            .unwrap();
            let positions = Node::new(NodeKind::PositionEmbedding {
                weight: tensor(&[128, 8], DType::F32, device),
                seq_len: 1,
            })
            .unwrap();

            let (roots, geometry) =
                specialize_decode_layout(&[logits, positions], None, layout, false).unwrap();

            assert_eq!(roots[0].shape, [6, 1, 8]);
            assert_eq!(roots[1].shape, [6, 1, 8]);
            assert!(geometry.cursor_tensor);
            let cursors = graph_post_order(&roots)
                .into_iter()
                .filter(|node| {
                    matches!(
                        &node.kind,
                        NodeKind::Input { slot, shape, dtype: DType::I64, .. }
                            if *slot == geometry.cursor_slot && shape == &[6]
                    )
                })
                .count();
            assert_eq!(cursors, 1);
        }

        let malformed = input(0, &[6, 2, 8], DType::F32, Device::Cpu(0));
        assert_eq!(
            specialize_decode_layout(
                &[malformed],
                None,
                DecodeLayout::packed_causal_chains(2, 3),
                false,
            )
            .err()
            .unwrap(),
            "decode: packed causal-chain roots must be [6, 1, V], got [6, 2, 8]"
        );
        assert_eq!(
            specialize_decode_layout(
                &[input(0, &[6, 1, 8], DType::F32, Device::Cpu(0))],
                None,
                DecodeLayout::packed_causal_chains(2, 3),
                true,
            )
            .err()
            .unwrap(),
            "decode: packed causal-chain verification requires all-row outputs"
        );
    }

    #[test]
    fn specialization_is_stack_safe_for_deep_graphs() {
        std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                let mut root = attention(1, 1, 1, 2, 1.0, true, Device::Cpu(0));
                for _ in 0..50_000 {
                    root = Node::new(NodeKind::Neg { a: root }).unwrap();
                }
                let (specialized, geometry) = specialize_decode(&[root], None, 1, false).unwrap();
                assert_eq!(geometry.layers, 1);
                assert_eq!(graph_post_order(&specialized).len(), 50_002);
            })
            .unwrap()
            .join()
            .unwrap();
    }
}

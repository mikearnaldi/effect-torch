use crate as fusion;
use effect_torch_graph::{
    node_children, remap_children, Device, Node as GraphNode, NodeKind as GraphNodeKind,
};
use effect_torch_runtime::DType;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

type Node = GraphNode<crate::Expr>;
type NodeKind = GraphNodeKind<crate::Expr>;

// Stack-safe post-order traversal: children in left-to-right order,
// shared nodes once (the autodiff `topo` pattern). Graph walks must
// never recurse — deep graphs are bounded by heap, not the call stack.
fn post_order(roots: &[Arc<Node>]) -> Vec<Arc<Node>> {
    let mut visited = HashSet::new();
    let mut order = Vec::new();
    let mut stack: Vec<(Arc<Node>, bool)> =
        roots.iter().rev().map(|r| (r.clone(), false)).collect();
    while let Some((node, processed)) = stack.pop() {
        if processed {
            order.push(node);
            continue;
        }
        if !visited.insert(node.id) {
            continue;
        }
        stack.push((node.clone(), true));
        for c in node_children(&node.kind).into_iter().rev() {
            stack.push((c, false));
        }
    }
    order
}

fn reduced_shape(shape: &[usize], dims: &[usize], keepdims: bool) -> Vec<usize> {
    if keepdims {
        shape
            .iter()
            .enumerate()
            .map(|(i, &d)| if dims.contains(&i) { 1 } else { d })
            .collect()
    } else {
        shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !dims.contains(i))
            .map(|(_, &d)| d)
            .collect()
    }
}

// RFC 0007 phase 2: folds maximal single-consumer chains of elementwise
// ops into FusedElementwise nodes, each evaluated as one kernel. Runs on a
// throwaway rewrite at evaluation time, so autodiff, vmap and checkpoint
// always see the unfused graph. EFFECT_TORCH_NO_FUSION disables it.
// Freeze-time optimizer grouping: same-shape AdamW steps collapse
// into AdamWStepGroup nodes of ≤4 params (one fused launch per group,
// 4 lanes + 3 outputs per param + 3 scalars against Metal's
// 31-buffer limit). DISABLED by default: measured at GPT-scale
// (≤64K-element params), the 28-lane mega-kernel's per-element cost
// is ~10× a single-param fused step and outweighs the saved launches
// (12.9ms vs 11.5ms per step). Set EFFECT_TORCH_OPT_GROUPS=1 to
// enable — it should win once parameters are large enough that kernel
// efficiency dominates launch count.
fn group_optimizer_steps(roots: &[Arc<Node>]) -> std::result::Result<Vec<Arc<Node>>, String> {
    if std::env::var_os("EFFECT_TORCH_OPT_GROUPS").is_none() {
        return Ok(roots.to_vec());
    }
    let order = post_order(roots);
    // Bucket steps by (shape, dtype, hyperparameters).
    type Key = (Vec<usize>, DType, u64, u64, u64, u64);
    let mut buckets: HashMap<Key, Vec<Arc<Node>>> = HashMap::new();
    let mut bucket_order: Vec<Key> = Vec::new();
    for node in &order {
        if let NodeKind::AdamWStep {
            param,
            beta1,
            beta2,
            eps,
            weight_decay,
            ..
        } = &node.kind
        {
            let key: Key = (
                param.shape.clone(),
                param.dtype,
                beta1.to_bits(),
                beta2.to_bits(),
                eps.to_bits(),
                weight_decay.to_bits(),
            );
            buckets
                .entry(key.clone())
                .or_insert_with(|| {
                    bucket_order.push(key);
                    Vec::new()
                })
                .push(node.clone());
        }
    }
    if bucket_order.iter().all(|k| buckets[k].len() < 2) {
        return Ok(roots.to_vec());
    }
    // step id -> (group node, param index)
    let mut grouped: HashMap<u64, (Arc<Node>, u32)> = HashMap::new();
    for key in &bucket_order {
        let steps = &buckets[key];
        for chunk in steps.chunks(4) {
            if chunk.len() < 2 {
                continue;
            }
            let first = &chunk[0];
            let NodeKind::AdamWStep {
                lr,
                c1,
                c2,
                beta1,
                beta2,
                eps,
                weight_decay,
                ..
            } = &first.kind
            else {
                unreachable!()
            };
            let group = Node::new(NodeKind::AdamWStepGroup {
                params: chunk
                    .iter()
                    .map(|s| match &s.kind {
                        NodeKind::AdamWStep { param, .. } => param.clone(),
                        _ => unreachable!(),
                    })
                    .collect(),
                grads: chunk
                    .iter()
                    .map(|s| match &s.kind {
                        NodeKind::AdamWStep { grad, .. } => grad.clone(),
                        _ => unreachable!(),
                    })
                    .collect(),
                ms: chunk
                    .iter()
                    .map(|s| match &s.kind {
                        NodeKind::AdamWStep { m, .. } => m.clone(),
                        _ => unreachable!(),
                    })
                    .collect(),
                vs: chunk
                    .iter()
                    .map(|s| match &s.kind {
                        NodeKind::AdamWStep { v, .. } => v.clone(),
                        _ => unreachable!(),
                    })
                    .collect(),
                lr: lr.clone(),
                c1: c1.clone(),
                c2: c2.clone(),
                beta1: *beta1,
                beta2: *beta2,
                eps: *eps,
                weight_decay: *weight_decay,
            })?;
            for (i, step) in chunk.iter().enumerate() {
                grouped.insert(step.id, (group.clone(), i as u32));
            }
        }
    }
    // Rebuild bottom-up: outs and direct references remap to the group.
    let mut map: HashMap<u64, Arc<Node>> = HashMap::new();
    for node in &order {
        let rebuilt = match &node.kind {
            NodeKind::AdamWStep { .. } if grouped.contains_key(&node.id) => {
                let (group, param) = &grouped[&node.id];
                NodeKind::AdamWGroupOut {
                    of: group.clone(),
                    param: *param,
                    index: 0,
                }
            }
            NodeKind::AdamWOut { step, index } if grouped.contains_key(&step.id) => {
                let (group, param) = &grouped[&step.id];
                NodeKind::AdamWGroupOut {
                    of: group.clone(),
                    param: *param,
                    index: *index,
                }
            }
            kind => {
                let remap = |child: &Arc<Node>| {
                    map.get(&child.id).cloned().unwrap_or_else(|| child.clone())
                };
                remap_children(kind, &remap)
            }
        };
        map.insert(node.id, Node::new(rebuilt)?);
    }
    Ok(roots
        .iter()
        .map(|r| map.get(&r.id).cloned().unwrap_or_else(|| r.clone()))
        .collect())
}

// RFC 0016 phase 3: folds gemm epilogues before the elementwise fold
// sees the graph. Two patterns, both Metal-only with gemm-supported
// dtypes:
//
// - Add(Linear(x, w, b), r) with the Linear's only consumer being this
//   Add and r exactly output-shaped becomes LinearResidual — the
//   epilogue adds r to the accumulator and the standalone proj output
//   never materializes. Linear backward reads x/w and the routed grad,
//   never its own output, so dropping the node is safe.
// - Gelu(Linear(x, w, b)) becomes LinearGelu. When the pre-activation
//   has further consumers (the backward gelu chain), the dual variant
//   writes it as output 0 and consumers read both through FusedPick;
//   otherwise the pre-activation buffer disappears entirely.
pub fn gemm_epilogue_pass(roots: &[Arc<Node>]) -> std::result::Result<Vec<Arc<Node>>, String> {
    let order = post_order(roots);
    let mut consumers: HashMap<u64, usize> = HashMap::new();
    for n in &order {
        for c in node_children(&n.kind) {
            *consumers.entry(c.id).or_insert(0) += 1;
        }
    }
    // A Linear the epilogue pass may absorb: Metal, f32/bf16 (the fused
    // gemm's supported dtypes).
    fn absorbable_linear(node: &Arc<Node>) -> Option<(Arc<Node>, Arc<Node>, Arc<Node>)> {
        if !matches!(node.device, Device::Metal) || !matches!(node.dtype, DType::F32 | DType::BF16)
        {
            return None;
        }
        match &node.kind {
            NodeKind::Linear { x, weight, bias } => Some((x.clone(), weight.clone(), bias.clone())),
            _ => None,
        }
    }
    let mut map: HashMap<u64, Arc<Node>> = HashMap::new();
    for node in &order {
        let remap = |ch: &Arc<Node>| map.get(&ch.id).cloned().unwrap_or_else(|| ch.clone());
        match &node.kind {
            NodeKind::Add { a: a0, b: b0 } => {
                let a = remap(a0);
                let b = remap(b0);
                // (linear, residual) or (residual, linear); consumer
                // counts are keyed by the ORIGINAL child ids, since the
                // remap minted fresh ones.
                let fused = [(a.clone(), b.clone(), a0.id), (b.clone(), a.clone(), b0.id)]
                    .into_iter()
                    .find_map(|(cand, res, orig_id)| {
                        let (x, weight, bias) = absorbable_linear(&cand)?;
                        // The Linear's only consumer must be this Add;
                        // Linear backward reads x/w and the routed grad,
                        // never its own output.
                        if consumers.get(&orig_id).copied().unwrap_or(0) != 1 {
                            return None;
                        }
                        if res.shape != cand.shape
                            || res.dtype != cand.dtype
                            || !matches!(res.device, Device::Metal)
                        {
                            return None;
                        }
                        Some(Node::new(NodeKind::LinearResidual {
                            x,
                            weight,
                            bias,
                            residual: res,
                        }))
                    });
                match fused {
                    Some(fused) => {
                        map.insert(node.id, fused?);
                    }
                    None => {
                        map.insert(node.id, Node::new(NodeKind::Add { a, b })?);
                    }
                }
            }
            NodeKind::Gelu { a: a0, approximate } => {
                let a = remap(a0);
                match absorbable_linear(&a) {
                    Some((x, weight, bias)) => {
                        let orig_id = a0.id;
                        if consumers.get(&orig_id).copied().unwrap_or(0) == 1 {
                            // Nothing else reads the pre-activation (no
                            // backward): the gemm stores only the gelu.
                            map.insert(
                                node.id,
                                Node::new(NodeKind::LinearGelu {
                                    x,
                                    weight,
                                    bias,
                                    approximate: *approximate,
                                    dual: false,
                                })?,
                            );
                        } else {
                            let dual = Node::new(NodeKind::LinearGelu {
                                x,
                                weight,
                                bias,
                                approximate: *approximate,
                                dual: true,
                            })?;
                            map.insert(
                                orig_id,
                                Node::new(NodeKind::FusedPick {
                                    of: dual.clone(),
                                    index: 0,
                                })?,
                            );
                            map.insert(
                                node.id,
                                Node::new(NodeKind::FusedPick { of: dual, index: 1 })?,
                            );
                        }
                    }
                    None => {
                        map.insert(
                            node.id,
                            Node::new(NodeKind::Gelu {
                                a,
                                approximate: *approximate,
                            })?,
                        );
                    }
                }
            }
            _ => {
                let rebuilt = remap_children(&node.kind, &remap);
                map.insert(node.id, Node::new(rebuilt)?);
            }
        }
    }
    Ok(roots
        .iter()
        .map(|r| map.get(&r.id).cloned().unwrap_or_else(|| r.clone()))
        .collect())
}

pub fn fuse_roots(roots: &[Arc<Node>]) -> std::result::Result<Vec<Arc<Node>>, String> {
    let roots = &group_optimizer_steps(roots)?;
    if std::env::var_os("EFFECT_TORCH_NO_FUSION").is_some() {
        return Ok(roots.to_vec());
    }
    // EFFECT_TORCH_NO_EPILOGUE isolates the pass for A/B measurement.
    let roots = &if std::env::var_os("EFFECT_TORCH_NO_EPILOGUE").is_some() {
        roots.to_vec()
    } else {
        gemm_epilogue_pass(roots)?
    };
    use fusion::Expr as E;

    let order = post_order(roots);
    let mut consumers: HashMap<u64, usize> = HashMap::new();
    for n in &order {
        for c in node_children(&n.kind) {
            *consumers.entry(c.id).or_insert(0) += 1;
        }
    }

    enum OpT {
        Unary(Box<dyn Fn(E) -> E>),
        Binary(Box<dyn Fn(E, E) -> E>),
        // where(cmp(x, y), a, b): the comparison constructor, applied to
        // the comparison's two inputs. Logical children are [x, y, a, b].
        Select(Box<dyn Fn(E, E) -> E>),
        // A reduce terminates the region feeding it (RFC 0007 phase 3a):
        // the chain compiles into the reduce loop instead of
        // materializing. Carries (op, dims, keepdims).
        Reduce(fusion::ReduceOp, Vec<usize>, bool),
    }
    // An input qualifies when it broadcasts into the output shape
    // (right-aligned dims equal or 1; scalars included). Broadcast lanes
    // are read through stride-0 dims inside the region instead of being
    // materialized at the output shape.
    fn input_ok(c: &Node, out: &[usize]) -> bool {
        fusion::broadcast_compatible(&c.shape, out)
    }
    let fusable = |node: &Node| -> Option<OpT> {
        if !fusion::is_supported(&node.device, node.dtype) {
            return None;
        }
        match &node.kind {
            NodeKind::Add { a, b } if input_ok(a, &node.shape) && input_ok(b, &node.shape) => {
                Some(OpT::Binary(Box::new(|a, b| {
                    E::Add(Box::new(a), Box::new(b))
                })))
            }
            NodeKind::Sub { a, b } if input_ok(a, &node.shape) && input_ok(b, &node.shape) => {
                Some(OpT::Binary(Box::new(|a, b| {
                    E::Sub(Box::new(a), Box::new(b))
                })))
            }
            NodeKind::Mul { a, b } if input_ok(a, &node.shape) && input_ok(b, &node.shape) => {
                Some(OpT::Binary(Box::new(|a, b| {
                    E::Mul(Box::new(a), Box::new(b))
                })))
            }
            NodeKind::Div { a, b } if input_ok(a, &node.shape) && input_ok(b, &node.shape) => {
                Some(OpT::Binary(Box::new(|a, b| {
                    E::Div(Box::new(a), Box::new(b))
                })))
            }
            NodeKind::Maximum { a, b } if input_ok(a, &node.shape) && input_ok(b, &node.shape) => {
                Some(OpT::Binary(Box::new(|a, b| {
                    E::Max(Box::new(a), Box::new(b))
                })))
            }
            NodeKind::Minimum { a, b } if input_ok(a, &node.shape) && input_ok(b, &node.shape) => {
                Some(OpT::Binary(Box::new(|a, b| {
                    E::Min(Box::new(a), Box::new(b))
                })))
            }
            NodeKind::Neg { .. } => Some(OpT::Unary(Box::new(|a| E::Neg(Box::new(a))))),
            NodeKind::Sqrt { .. } => Some(OpT::Unary(Box::new(|a| E::Sqrt(Box::new(a))))),
            NodeKind::Exp { .. } => Some(OpT::Unary(Box::new(|a| E::Exp(Box::new(a))))),
            NodeKind::Log { .. } => Some(OpT::Unary(Box::new(|a| E::Log(Box::new(a))))),
            NodeKind::Sin { .. } => Some(OpT::Unary(Box::new(|a| E::Sin(Box::new(a))))),
            NodeKind::Cos { .. } => Some(OpT::Unary(Box::new(|a| E::Cos(Box::new(a))))),
            NodeKind::Relu { .. } => Some(OpT::Unary(Box::new(|a| {
                E::Max(Box::new(a), Box::new(E::cst(0.0)))
            }))),
            NodeKind::Tanh { .. } => Some(OpT::Unary(Box::new(|a| E::Tanh(Box::new(a))))),
            NodeKind::Gelu { approximate, .. } => {
                let approximate = *approximate;
                Some(OpT::Unary(Box::new(move |a| {
                    if approximate {
                        E::GeluTanh(Box::new(a))
                    } else {
                        E::Gelu(Box::new(a))
                    }
                })))
            }
            NodeKind::Abs { .. } => Some(OpT::Unary(Box::new(|a| E::Abs(Box::new(a))))),
            NodeKind::Erf { .. } => Some(OpT::Unary(Box::new(|a| E::Erf(Box::new(a))))),
            NodeKind::Floor { .. } => Some(OpT::Unary(Box::new(|a| E::Floor(Box::new(a))))),
            NodeKind::Ceil { .. } => Some(OpT::Unary(Box::new(|a| E::Ceil(Box::new(a))))),
            NodeKind::Round { .. } => Some(OpT::Unary(Box::new(|a| E::Round(Box::new(a))))),
            NodeKind::Pow { exp, .. } => {
                let exp = *exp;
                Some(OpT::Unary(Box::new(move |a| fusion::pow_expr(a, exp))))
            }
            // sign(x) = (x > 0) ? 1 : ((x < 0) ? -1 : 0); NaN yields 0,
            // matching candle's CPU and Metal kernels.
            NodeKind::Sign { .. } => Some(OpT::Unary(Box::new(|a| {
                E::Select(
                    Box::new(E::Gt(Box::new(a.clone()), Box::new(E::cst(0.0)))),
                    Box::new(E::cst(1.0)),
                    Box::new(E::Select(
                        Box::new(E::Lt(Box::new(a), Box::new(E::cst(0.0)))),
                        Box::new(E::cst(-1.0)),
                        Box::new(E::cst(0.0)),
                    )),
                )
            }))),
            // A dtype-preserving cast is the identity inside a region.
            // Cross-dtype casts stay region boundaries: lanes are loaded in
            // the region's single dtype.
            NodeKind::Cast { a, dtype } if a.dtype == *dtype => Some(OpT::Unary(Box::new(|a| a))),
            // where(cond, a, b) fuses only when cond is a comparison with
            // no other consumer: the comparison lowers to a float mask
            // feeding a true select, so the u8 mask never materializes.
            // A cond shared with another consumer must stay a real u8
            // tensor, which a float region cannot produce.
            NodeKind::Where { cond, a, b }
                if consumers.get(&cond.id).copied().unwrap_or(0) == 1
                    && input_ok(a, &node.shape)
                    && input_ok(b, &node.shape) =>
            {
                let cmp: Option<Box<dyn Fn(E, E) -> E>> = match &cond.kind {
                    NodeKind::Eq { a: x, b: y }
                        if input_ok(x, &node.shape)
                            && input_ok(y, &node.shape)
                            && fusion::is_supported(&x.device, x.dtype) =>
                    {
                        Some(Box::new(|a, b| E::Eq(Box::new(a), Box::new(b))))
                    }
                    NodeKind::Gt { a: x, b: y }
                        if input_ok(x, &node.shape)
                            && input_ok(y, &node.shape)
                            && fusion::is_supported(&x.device, x.dtype) =>
                    {
                        Some(Box::new(|a, b| E::Gt(Box::new(a), Box::new(b))))
                    }
                    NodeKind::Lt { a: x, b: y }
                        if input_ok(x, &node.shape)
                            && input_ok(y, &node.shape)
                            && fusion::is_supported(&x.device, x.dtype) =>
                    {
                        Some(Box::new(|a, b| E::Lt(Box::new(a), Box::new(b))))
                    }
                    NodeKind::Ge { a: x, b: y }
                        if input_ok(x, &node.shape)
                            && input_ok(y, &node.shape)
                            && fusion::is_supported(&x.device, x.dtype) =>
                    {
                        Some(Box::new(|a, b| E::Ge(Box::new(a), Box::new(b))))
                    }
                    NodeKind::Le { a: x, b: y }
                        if input_ok(x, &node.shape)
                            && input_ok(y, &node.shape)
                            && fusion::is_supported(&x.device, x.dtype) =>
                    {
                        Some(Box::new(|a, b| E::Le(Box::new(a), Box::new(b))))
                    }
                    _ => None,
                };
                cmp.map(OpT::Select)
            }
            NodeKind::Sum { dims, keepdims, .. } if !dims.is_empty() => {
                Some(OpT::Reduce(fusion::ReduceOp::Sum, dims.clone(), *keepdims))
            }
            NodeKind::Mean { dims, keepdims, .. } if !dims.is_empty() => {
                Some(OpT::Reduce(fusion::ReduceOp::Mean, dims.clone(), *keepdims))
            }
            NodeKind::Max { dims, keepdims, .. } if !dims.is_empty() => {
                Some(OpT::Reduce(fusion::ReduceOp::Max, dims.clone(), *keepdims))
            }
            NodeKind::Min { dims, keepdims, .. } if !dims.is_empty() => {
                Some(OpT::Reduce(fusion::ReduceOp::Min, dims.clone(), *keepdims))
            }
            _ => None,
        }
    };
    // A uniform-value child folds into an IR constant when it broadcasts
    // into the output shape (scalar, broadcast-smaller, or output-shaped).
    let const_value = |child: &Node, out_shape: &[usize]| -> Option<f64> {
        match &child.kind {
            NodeKind::Full { shape, value, .. }
                if fusion::broadcast_compatible(shape, out_shape) =>
            {
                Some(*value)
            }
            NodeKind::Zeros { shape, .. } if fusion::broadcast_compatible(shape, out_shape) => {
                Some(0.0)
            }
            _ => None,
        }
    };

    struct Region {
        expr: E,
        inputs: Vec<Arc<Node>>,
        lane_of: HashMap<u64, u32>,
        ops: usize,
    }
    impl Region {
        fn empty() -> Self {
            Region {
                expr: E::cst(0.0),
                inputs: Vec::new(),
                lane_of: HashMap::new(),
                ops: 0,
            }
        }
        fn lane(&mut self, n: &Arc<Node>) -> E {
            if let Some(&k) = self.lane_of.get(&n.id) {
                return E::Input(k);
            }
            let k = self.inputs.len() as u32;
            self.inputs.push(n.clone());
            self.lane_of.insert(n.id, k);
            E::Input(k)
        }
        // Takes ownership of another region's lanes; returns its expr with
        // lane indices remapped into this region's namespace. Lanes the
        // two regions share are reused, not duplicated.
        fn absorb(&mut self, other: Region) -> E {
            let mut remap: HashMap<u32, u32> = HashMap::new();
            for (k, input) in other.inputs.iter().enumerate() {
                let idx = match self.lane_of.get(&input.id) {
                    Some(&existing) => existing,
                    None => {
                        let idx = self.inputs.len() as u32;
                        self.lane_of.insert(input.id, idx);
                        self.inputs.push(input.clone());
                        idx
                    }
                };
                remap.insert(k as u32, idx);
            }
            self.ops += other.ops;
            other.expr.remap_lanes(&remap)
        }
    }

    // Metal kernels accept at most 31 buffer arguments; one slot is the
    // output, so regions are capped at 30 input lanes. Overflow closes the
    // region (it materializes as a fused node) and the op continues with
    // that fused node as a plain input lane.
    const MAX_LANES: usize = 30;
    // Emits a closed region as a FusedElementwise node, or rebuilds the
    // node plainly (children already emitted) when fusion does not apply:
    // single-op regions, lane-less constant regions, regions too large for
    // the Metal kernel's i32 indexing, or a lane that fails to broadcast
    // (unreachable by construction, handled defensively).
    fn emit_region(
        node: &Node,
        region: Region,
        map: &mut HashMap<u64, Arc<Node>>,
    ) -> std::result::Result<(), String> {
        let n: usize = node.shape.iter().product();
        let strides: Option<Vec<Vec<usize>>> = region
            .inputs
            .iter()
            .map(|lane| fusion::lane_strides(&lane.shape, &node.shape))
            .collect();
        let fused = match strides {
            Some(strides)
                if region.ops >= 2
                    && !region.inputs.is_empty()
                    && !(matches!(node.device, Device::Metal) && n > i32::MAX as usize) =>
            {
                Node::new(NodeKind::FusedElementwise {
                    inputs: region.inputs,
                    strides,
                    shape: node.shape.clone(),
                    expr: region.expr,
                })?
            }
            _ => Node::new(remap_children(&node.kind, &|ch| {
                map.get(&ch.id).cloned().unwrap_or_else(|| ch.clone())
            }))?,
        };
        map.insert(node.id, fused);
        Ok(())
    }
    let mut open: HashMap<u64, Region> = HashMap::new();
    let mut map: HashMap<u64, Arc<Node>> = HashMap::new();
    for node in &order {
        let children = node_children(&node.kind);
        let opt = fusable(node);
        // close regions this node will not extend
        for c in &children {
            if open.contains_key(&c.id)
                && (opt.is_none() || consumers.get(&c.id).copied().unwrap_or(0) != 1)
            {
                let region = open.remove(&c.id).unwrap();
                emit_region(c, region, &mut map)?;
            }
        }
        match opt {
            None => {
                let rebuilt = remap_children(&node.kind, &|ch| {
                    map.get(&ch.id).cloned().unwrap_or_else(|| ch.clone())
                });
                map.insert(node.id, Node::new(rebuilt)?);
            }
            Some(OpT::Unary(f)) => {
                let c = children[0].clone();
                let (mut region, expr) = match open.remove(&c.id) {
                    Some(mut r) => {
                        // Owned region, overwritten just below: take the
                        // expr instead of cloning the whole tree.
                        let e = f(std::mem::replace(&mut r.expr, E::cst(0.0)));
                        (r, e)
                    }
                    None => {
                        let mut r = Region::empty();
                        let l = if let Some(v) = const_value(&c, &node.shape) {
                            E::cst(v)
                        } else {
                            r.lane(&map.get(&c.id).cloned().unwrap_or_else(|| c.clone()))
                        };
                        (r, f(l))
                    }
                };
                region.expr = expr;
                region.ops += 1;
                open.insert(node.id, region);
            }
            Some(OpT::Select(cmpf)) => {
                let (ca, cb, a, b) = match &node.kind {
                    NodeKind::Where { cond, a, b } => match &cond.kind {
                        NodeKind::Eq { a: x, b: y }
                        | NodeKind::Gt { a: x, b: y }
                        | NodeKind::Lt { a: x, b: y }
                        | NodeKind::Ge { a: x, b: y }
                        | NodeKind::Le { a: x, b: y } => (x, y, a, b),
                        _ => unreachable!("fusion: select guard"),
                    },
                    _ => unreachable!("fusion: select guard"),
                };
                // Fold the four logical children into one region. The
                // comparison's inputs never have open regions (the
                // non-fusable comparison closed them when it was visited);
                // the branches may. On lane-cap overflow with nothing left
                // to give, abandon: dropped regions are still covered by
                // the original subgraphs, and the node rebuilds plain.
                let logical: [&Arc<Node>; 4] = [ca, cb, a, b];
                let mut region = Region::empty();
                let mut exprs: Vec<E> = Vec::with_capacity(4);
                let mut abandon = false;
                for child in logical {
                    if abandon {
                        break;
                    }
                    if let Some(r) = open.remove(&child.id) {
                        if region.inputs.len() + r.inputs.len() > MAX_LANES {
                            emit_region(child, r, &mut map)?;
                            let resolved = map.get(&child.id).cloned().unwrap();
                            if region.inputs.len() >= MAX_LANES
                                && !region.lane_of.contains_key(&resolved.id)
                            {
                                abandon = true;
                            } else {
                                exprs.push(region.lane(&resolved));
                            }
                        } else {
                            exprs.push(region.absorb(r));
                        }
                    } else if let Some(v) = const_value(child, &node.shape) {
                        exprs.push(E::cst(v));
                    } else {
                        let resolved = map.get(&child.id).cloned().unwrap_or_else(|| child.clone());
                        if region.inputs.len() >= MAX_LANES
                            && !region.lane_of.contains_key(&resolved.id)
                        {
                            abandon = true;
                        } else {
                            exprs.push(region.lane(&resolved));
                        }
                    }
                }
                if abandon {
                    let rebuilt = remap_children(&node.kind, &|ch| {
                        map.get(&ch.id).cloned().unwrap_or_else(|| ch.clone())
                    });
                    map.insert(node.id, Node::new(rebuilt)?);
                } else {
                    let mut it = exprs.into_iter();
                    let (e0, e1, e2, e3) = (
                        it.next().unwrap(),
                        it.next().unwrap(),
                        it.next().unwrap(),
                        it.next().unwrap(),
                    );
                    region.expr = E::Select(Box::new(cmpf(e0, e1)), Box::new(e2), Box::new(e3));
                    region.ops += 1;
                    open.insert(node.id, region);
                }
            }
            Some(OpT::Binary(f)) => {
                let a = children[0].clone();
                let b = children[1].clone();
                let mut ra = open.remove(&a.id);
                let mut rb = open.remove(&b.id);
                // A merged region must stay within the lane cap; otherwise
                // the smaller side materializes first.
                if let (Some(r1), Some(r2)) = (&ra, &rb) {
                    if r1.inputs.len() + r2.inputs.len() > MAX_LANES {
                        emit_region(&b, rb.take().unwrap(), &mut map)?;
                    }
                }
                // Extending a region with a brand-new lane must stay within
                // the cap; otherwise that region materializes first and the
                // op reads it back as a plain lane.
                if let Some(r) = &ra {
                    if rb.is_none() {
                        let resolved = map.get(&b.id).map(|n| n.id).unwrap_or(b.id);
                        if const_value(&b, &node.shape).is_none()
                            && !r.lane_of.contains_key(&resolved)
                            && r.inputs.len() >= MAX_LANES
                        {
                            let region = ra.take().unwrap();
                            emit_region(&a, region, &mut map)?;
                        }
                    }
                }
                if let Some(r) = &rb {
                    if ra.is_none() {
                        let resolved = map.get(&a.id).map(|n| n.id).unwrap_or(a.id);
                        if const_value(&a, &node.shape).is_none()
                            && !r.lane_of.contains_key(&resolved)
                            && r.inputs.len() >= MAX_LANES
                        {
                            let region = rb.take().unwrap();
                            emit_region(&b, region, &mut map)?;
                        }
                    }
                }
                let (mut region, expr) = match (ra, rb) {
                    (Some(mut r1), Some(r2)) => {
                        let b_expr = r1.absorb(r2);
                        let e = f(std::mem::replace(&mut r1.expr, E::cst(0.0)), b_expr);
                        (r1, e)
                    }
                    (Some(mut r), None) => {
                        let l = if let Some(v) = const_value(&b, &node.shape) {
                            E::cst(v)
                        } else {
                            r.lane(&map.get(&b.id).cloned().unwrap_or_else(|| b.clone()))
                        };
                        let e = f(std::mem::replace(&mut r.expr, E::cst(0.0)), l);
                        (r, e)
                    }
                    (None, Some(mut r)) => {
                        let l = if let Some(v) = const_value(&a, &node.shape) {
                            E::cst(v)
                        } else {
                            r.lane(&map.get(&a.id).cloned().unwrap_or_else(|| a.clone()))
                        };
                        let e = f(l, std::mem::replace(&mut r.expr, E::cst(0.0)));
                        (r, e)
                    }
                    (None, None) => {
                        let mut r = Region::empty();
                        let la = if let Some(v) = const_value(&a, &node.shape) {
                            E::cst(v)
                        } else {
                            r.lane(&map.get(&a.id).cloned().unwrap_or_else(|| a.clone()))
                        };
                        let lb = if let Some(v) = const_value(&b, &node.shape) {
                            E::cst(v)
                        } else {
                            r.lane(&map.get(&b.id).cloned().unwrap_or_else(|| b.clone()))
                        };
                        (r, f(la, lb))
                    }
                };
                region.expr = expr;
                region.ops += 1;
                open.insert(node.id, region);
            }
            Some(OpT::Reduce(op, mut dims, keepdims)) => {
                let a = children[0].clone();
                let in_shape = a.shape.clone();
                dims.sort_unstable();
                dims.dedup();
                let rank = in_shape.len();
                let guards_ok = !dims.is_empty()
                    && dims.iter().all(|&d| d < rank)
                    && dims.iter().map(|&d| in_shape[d]).product::<usize>() > 0
                    && !(matches!(node.device, Device::Metal) && {
                        let in_n: usize = in_shape.iter().product();
                        let out_n: usize =
                            reduced_shape(&in_shape, &dims, keepdims).iter().product();
                        in_n > i32::MAX as usize || out_n > i32::MAX as usize
                    });
                match open.remove(&a.id) {
                    Some(region) if guards_ok && !region.inputs.is_empty() => {
                        let strides: Option<Vec<Vec<usize>>> = region
                            .inputs
                            .iter()
                            .map(|lane| fusion::lane_strides(&lane.shape, &in_shape))
                            .collect();
                        match strides {
                            Some(strides) => {
                                let fused = Node::new(NodeKind::FusedReduce {
                                    inputs: region.inputs,
                                    strides,
                                    in_shape: in_shape.clone(),
                                    expr: region.expr,
                                    op,
                                    dims: dims.clone(),
                                    keepdims,
                                    shape: reduced_shape(&in_shape, &dims, keepdims),
                                })?;
                                map.insert(node.id, fused);
                            }
                            None => {
                                emit_region(&a, region, &mut map)?;
                                let rebuilt = remap_children(&node.kind, &|ch| {
                                    map.get(&ch.id).cloned().unwrap_or_else(|| ch.clone())
                                });
                                map.insert(node.id, Node::new(rebuilt)?);
                            }
                        }
                    }
                    region => {
                        // No single-consumer region to compile into the
                        // reduce loop (or a degenerate reduce): emit the
                        // region plainly if one stayed open and rebuild.
                        if let Some(r) = region {
                            emit_region(&a, r, &mut map)?;
                        }
                        let rebuilt = remap_children(&node.kind, &|ch| {
                            map.get(&ch.id).cloned().unwrap_or_else(|| ch.clone())
                        });
                        map.insert(node.id, Node::new(rebuilt)?);
                    }
                }
            }
        }
    }
    // close regions whose end has no consumer in the graph (graph roots)
    for (id, region) in open.drain() {
        let node = order.iter().find(|n| n.id == id).unwrap();
        emit_region(node, region, &mut map)?;
    }
    Ok(roots
        .iter()
        .map(|r| map.get(&r.id).cloned().unwrap_or_else(|| r.clone()))
        .collect::<Vec<_>>())
    .and_then(|roots| merge_shared_regions(&roots))
}

// RFC 0007 multi-output merge: when a fused region materializes because
// its value has several consumers, and some of those consumers are
// themselves fused regions of one common shape, compile the prefix and
// the continuations as a single multi-output kernel (the prefix's
// expression is inlined into each continuation, so the shared
// intermediate stays in registers instead of round-tripping through a
// buffer). Runs as a post-pass on the rewritten graph: kernel signatures
// are fixed at compile time, so the merge needs the full consumer set,
// which only exists once the region sweep has finished. Repeats to a
// fixpoint; each merge removes at least one FusedElementwise node, so it
// terminates.
fn merge_shared_regions(roots: &[Arc<Node>]) -> std::result::Result<Vec<Arc<Node>>, String> {
    if std::env::var_os("EFFECT_TORCH_NO_MULTI_FUSION").is_some() {
        return Ok(roots.to_vec());
    }
    // Total expression size bound for one merged kernel: keeps register
    // pressure and shader compile time sane on pathological shares.
    const MAX_MERGED_OPS: usize = 512;

    struct Plan {
        // the shared prefix node (a FusedElementwise)
        prefix: u64,
        // fused continuations (FusedElementwise nodes of one common shape)
        group: Vec<u64>,
        // whether the prefix must stay materialized for unfused consumers
        keep_prefix: bool,
        multi: NodeKind,
    }

    fn analyze(roots: &[Arc<Node>]) -> (Vec<Arc<Node>>, HashMap<u64, Vec<u64>>) {
        let order = post_order(roots);
        let mut consumers: HashMap<u64, Vec<u64>> = HashMap::new();
        for n in &order {
            for c in node_children(&n.kind) {
                consumers.entry(c.id).or_default().push(n.id);
            }
        }
        (order, consumers)
    }

    fn find_merge(order: &[Arc<Node>], consumers: &HashMap<u64, Vec<u64>>) -> Option<Plan> {
        let by_id: HashMap<u64, &Arc<Node>> = order.iter().map(|n| (n.id, n)).collect();
        for node in order {
            let NodeKind::FusedElementwise {
                inputs,
                strides,
                shape,
                expr,
            } = &node.kind
            else {
                continue;
            };
            let Some(cons) = consumers.get(&node.id) else {
                continue;
            };
            if cons.len() < 2 {
                continue;
            }
            // group fused consumers by common output shape, keeping the
            // largest group (encounter order breaks ties deterministically)
            let mut groups: Vec<(Vec<usize>, Vec<&Arc<Node>>)> = Vec::new();
            for cid in cons {
                let c = by_id[cid];
                if let NodeKind::FusedElementwise { .. } = &c.kind {
                    match groups.iter_mut().find(|(s, _)| s == &c.shape) {
                        Some((_, g)) => g.push(c),
                        None => groups.push((c.shape.clone(), vec![c])),
                    }
                }
            }
            groups.sort_by_key(|(_, g)| std::cmp::Reverse(g.len()));
            for (out_shape, group) in groups {
                let group_ids: HashSet<u64> = group.iter().map(|g| g.id).collect();
                // a continuation that reads another group member would need
                // nested inlining; skip the whole group in that case
                if group.iter().any(|g| {
                    node_children(&g.kind)
                        .iter()
                        .any(|ch| group_ids.contains(&ch.id))
                }) {
                    continue;
                }
                let keep_prefix = cons.iter().any(|cid| !group_ids.contains(cid));
                if group.len() + usize::from(keep_prefix) < 2 {
                    continue;
                }
                // A materialized prefix output is evaluated at the group's
                // coordinates, so it only equals the prefix's own value
                // when the shapes match; a broadcast-smaller prefix is
                // only safe to inline (never to emit).
                if keep_prefix && shape != &out_shape {
                    continue;
                }
                let Some(f_as_lane) = fusion::lane_strides(shape, &out_shape) else {
                    continue;
                };
                let offset = out_shape.len() - shape.len();
                // merged lanes: the prefix's lanes first (strides composed
                // through the prefix's own broadcast into out_shape), then
                // each continuation's extra lanes
                let mut lanes: Vec<Arc<Node>> = Vec::new();
                let mut lane_strides_out: Vec<Vec<usize>> = Vec::new();
                let mut lane_index: HashMap<u64, u32> = HashMap::new();
                for (input, s) in inputs.iter().zip(strides.iter()) {
                    lane_index.insert(input.id, lanes.len() as u32);
                    lanes.push(input.clone());
                    lane_strides_out.push(
                        f_as_lane
                            .iter()
                            .enumerate()
                            .map(|(d, &fs)| if fs == 0 { 0 } else { s[d - offset] })
                            .collect(),
                    );
                }
                let mut exprs: Vec<fusion::Expr> = Vec::new();
                if keep_prefix {
                    exprs.push(expr.clone());
                }
                let mut total_ops: usize = expr.ops();
                let mut ok = true;
                for g in &group {
                    let NodeKind::FusedElementwise {
                        inputs: g_inputs,
                        strides: g_strides,
                        expr: g_expr,
                        ..
                    } = &g.kind
                    else {
                        unreachable!()
                    };
                    let f_lane = g_inputs
                        .iter()
                        .position(|i| i.id == node.id)
                        .expect("fusion: group member must read the prefix")
                        as u32;
                    let mut remap: HashMap<u32, u32> = HashMap::new();
                    for (j, (input, s)) in g_inputs.iter().zip(g_strides.iter()).enumerate() {
                        if input.id == node.id {
                            continue;
                        }
                        let idx = match lane_index.get(&input.id) {
                            Some(&k) => k,
                            None => {
                                let k = lanes.len() as u32;
                                lane_index.insert(input.id, k);
                                lanes.push(input.clone());
                                lane_strides_out.push(s.clone());
                                k
                            }
                        };
                        remap.insert(j as u32, idx);
                    }
                    let merged = g_expr.merge_lane(f_lane, expr, &remap);
                    total_ops += merged.ops();
                    exprs.push(merged);
                }
                // Metal allows 31 buffer arguments per kernel; every lane
                // and every output takes one
                ok &= lanes.len() + exprs.len() <= 31;
                ok &= total_ops <= MAX_MERGED_OPS;
                ok &= !group
                    .iter()
                    .any(|g| g.dtype != node.dtype || !g.device.same_device(&node.device));
                if matches!(node.device, Device::Metal) {
                    ok &= out_shape.iter().product::<usize>() <= i32::MAX as usize;
                }
                if !ok {
                    continue;
                }
                return Some(Plan {
                    prefix: node.id,
                    group: group.iter().map(|g| g.id).collect(),
                    keep_prefix,
                    multi: NodeKind::FusedElementwiseMulti {
                        inputs: lanes,
                        strides: lane_strides_out,
                        shape: out_shape.clone(),
                        exprs,
                    },
                });
            }
        }
        None
    }

    let mut current = roots.to_vec();
    loop {
        let (order, consumers) = analyze(&current);
        if std::env::var_os("EFFECT_TORCH_FUSION_DEBUG").is_some() {
            let mut fe = 0;
            let mut multi = 0;
            let mut pick = 0;
            let mut red = 0;
            for n in &order {
                match &n.kind {
                    NodeKind::FusedElementwise { .. } => fe += 1,
                    NodeKind::FusedElementwiseMulti { .. } => multi += 1,
                    NodeKind::FusedPick { .. } => pick += 1,
                    NodeKind::FusedReduce { .. } => red += 1,
                    _ => {}
                }
            }
            eprintln!(
                "[fusion] analyze: {} nodes (fe {fe}, multi {multi}, pick {pick}, reduce {red})",
                order.len()
            );
        }
        let Some(plan) = find_merge(&order, &consumers) else {
            return Ok(current);
        };
        if std::env::var_os("EFFECT_TORCH_FUSION_DEBUG").is_some() {
            eprintln!(
                "[fusion] multi-merge: prefix {} -> group {:?} (keep {})",
                plan.prefix, plan.group, plan.keep_prefix
            );
        }
        // Remaps a node depth-first through rebuilt subtrees, memoized
        // into `map`. The multi's lanes are remapped this way BEFORE the
        // main rewrite so the multi never references the original lane
        // nodes: keeping the originals would retain their whole
        // ancestry (fused regions included), which the next fixpoint
        // round would see and merge again — duplicating a generation of
        // the subgraph per round. A lane's ancestry can include the
        // prefix itself (a continuation's extra lane descending from
        // it); that path rebuilds the prefix plainly — single-consumer,
        // so it cannot be re-merged and the duplication is bounded.
        fn remap_deep(
            n: &Arc<Node>,
            map: &mut HashMap<u64, Arc<Node>>,
        ) -> std::result::Result<Arc<Node>, String> {
            if let Some(r) = map.get(&n.id) {
                return Ok(r.clone());
            }
            let children = node_children(&n.kind);
            let mut resolved: HashMap<u64, Arc<Node>> = HashMap::with_capacity(children.len());
            for ch in &children {
                resolved.insert(ch.id, remap_deep(ch, map)?);
            }
            let kind = remap_children(&n.kind, &|ch| {
                resolved.get(&ch.id).cloned().unwrap_or_else(|| ch.clone())
            });
            let rebuilt = Node::new(kind)?;
            map.insert(n.id, rebuilt.clone());
            Ok(rebuilt)
        }
        let mut map: HashMap<u64, Arc<Node>> = HashMap::new();
        let multi = {
            let NodeKind::FusedElementwiseMulti {
                inputs,
                strides,
                shape,
                exprs,
            } = &plan.multi
            else {
                unreachable!("fusion: merge plan must build a multi node")
            };
            let mut remapped_inputs = Vec::with_capacity(inputs.len());
            for lane in inputs {
                remapped_inputs.push(remap_deep(lane, &mut map)?);
            }
            Node::new(NodeKind::FusedElementwiseMulti {
                inputs: remapped_inputs,
                strides: strides.clone(),
                shape: shape.clone(),
                exprs: exprs.clone(),
            })?
        };
        let mut pick_index = u8::from(plan.keep_prefix);
        let mut picks: HashMap<u64, Arc<Node>> = HashMap::new();
        if plan.keep_prefix {
            picks.insert(
                plan.prefix,
                Node::new(NodeKind::FusedPick {
                    of: multi.clone(),
                    index: 0,
                })?,
            );
        }
        for gid in &plan.group {
            picks.insert(
                *gid,
                Node::new(NodeKind::FusedPick {
                    of: multi.clone(),
                    index: pick_index,
                })?,
            );
            pick_index += 1;
        }
        for node in &order {
            if map.contains_key(&node.id) {
                continue;
            }
            if let Some(pick) = picks.get(&node.id) {
                map.insert(node.id, pick.clone());
                continue;
            }
            let rebuilt = remap_children(&node.kind, &|ch| {
                map.get(&ch.id).cloned().unwrap_or_else(|| ch.clone())
            });
            map.insert(node.id, Node::new(rebuilt)?);
        }
        current = current
            .iter()
            .map(|r| map.get(&r.id).cloned().unwrap_or_else(|| r.clone()))
            .collect();
    }
}

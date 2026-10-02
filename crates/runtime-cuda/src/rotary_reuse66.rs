//! Invocation-local rotary table reuse. Semantic graphs and arithmetic stay intact.
use super::*;
use effect_torch_compiler::{DTypeExecution, ElementwiseRegion, KernelExpr};
use std::collections::{HashMap, HashSet};

pub(crate) fn enabled() -> bool {
    let enabled = std::env::var("EFFECT_TORCH_CUDA_ROTARY_REUSE66").as_deref() == Ok("1");
    #[cfg(test)]
    let enabled = TEST_POLICY.with(|p| p.get().unwrap_or(enabled));
    enabled
}
#[cfg(test)]
thread_local! { static TEST_POLICY: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) }; }
#[cfg(test)]
fn with_test_policy<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_POLICY.with(|p| p.set(self.0));
        }
    }
    let _restore = Restore(TEST_POLICY.with(|p| p.replace(Some(enabled))));
    run()
}

pub(crate) const FEATURE: &str = "rotary-reuse66-semantic-two-table-alias-v1";

#[derive(Clone, Debug)]
struct Chain {
    nodes: [DenseNodeId; 11],
    units: Vec<LoweringUnit>,
    positions: Vec<usize>,
    contracts: Vec<DTypeExecution>,
    rows: usize,
    half: usize,
}
impl Chain {
    fn key(&self) -> (DenseNodeId, DenseNodeId, usize, usize) {
        (self.nodes[0], self.nodes[3], self.rows, self.half)
    }
    fn tables(&self) -> [DenseNodeId; 2] {
        [self.nodes[8], self.nodes[10]]
    }
}

fn unit_of(plan: &OptimizationPlan, node: DenseNodeId) -> LoweringUnit {
    plan.node_region[node.index()]
        .map(LoweringUnit::Region)
        .unwrap_or(LoweringUnit::Node(node))
}

fn elementwise<'a>(
    plan: &'a OptimizationPlan,
    output: DenseNodeId,
    nodes: &[DenseNodeId],
    inputs: &[DenseNodeId],
    expression: KernelExpr,
) -> Option<&'a ElementwiseRegion> {
    let NativeRegion::Elementwise(r) = &plan.regions[plan.node_region[output.index()]?.index()]
    else {
        return None;
    };
    let mut expected = nodes.to_vec();
    expected.sort();
    (r.nodes.as_ref() == expected
        && r.inputs.as_ref() == inputs
        && r.output.semantic_node == output
        && r.output.expression == expression)
        .then_some(r)
}

fn candidate(
    index: &GraphIndex,
    plan: &OptimizationPlan,
    legal: &LegalizationPlan,
    table: &Arc<Node>,
) -> Option<Chain> {
    let NodeKind::Cast { a: cos, .. } = &table.kind else {
        return None;
    };
    let NodeKind::Cos { a: doubled } = &cos.kind else {
        return None;
    };
    let NodeKind::Concat {
        a: phase,
        b: second,
        dim,
    } = &doubled.kind
    else {
        return None;
    };
    if phase.id != second.id || *dim != 2 {
        return None;
    }
    let NodeKind::Mul {
        a: pos_view,
        b: freq_cast,
    } = &phase.kind
    else {
        return None;
    };
    let NodeKind::Reshape { a: pos_cast, .. } = &pos_view.kind else {
        return None;
    };
    let NodeKind::Cast { a: raw_pos, .. } = &pos_cast.kind else {
        return None;
    };
    let NodeKind::Cast { a: raw_freq, .. } = &freq_cast.kind else {
        return None;
    };
    if !matches!(&raw_pos.kind, NodeKind::Input { .. } | NodeKind::Leaf(_))
        || !matches!(&raw_freq.kind, NodeKind::Input { .. } | NodeKind::Leaf(_))
    {
        return None;
    }
    let &[1, rows] = raw_pos.shape.as_slice() else {
        return None;
    };
    let &[half] = raw_freq.shape.as_slice() else {
        return None;
    };
    if !matches!(rows, 64 | 256) || !matches!(half, 128 | 256) {
        return None;
    }
    let doubled_id = index.dense_id(doubled.id)?;
    let sin = index.consumers[doubled_id.index()]
        .iter()
        .filter_map(|id| {
            let n = &index.order[id.index()];
            matches!(&n.kind,NodeKind::Sin{a} if a.id==doubled.id).then_some(n)
        })
        .next()?;
    let sin_id = index.dense_id(sin.id)?;
    let sin_table = index.consumers[sin_id.index()]
        .iter()
        .filter_map(|id| {
            let n = &index.order[id.index()];
            matches!(&n.kind,NodeKind::Cast{a,..} if a.id==sin.id && n.dtype==DType::BF16)
                .then_some(n)
        })
        .next()?;
    let refs = [
        raw_pos, pos_cast, pos_view, raw_freq, freq_cast, phase, doubled, cos, table, sin,
        sin_table,
    ];
    let expected_shapes = [
        vec![1, rows],
        vec![1, rows],
        vec![1, rows, 1],
        vec![half],
        vec![half],
        vec![1, rows, half],
        vec![1, rows, 2 * half],
        vec![1, rows, 2 * half],
        vec![1, rows, 2 * half],
        vec![1, rows, 2 * half],
        vec![1, rows, 2 * half],
    ];
    let mut nodes = [doubled_id; 11];
    for (i, n) in refs.iter().enumerate() {
        let dtype = match i {
            0 => DType::U32,
            8 | 10 => DType::BF16,
            _ => DType::F32,
        };
        nodes[i] = index.dense_id(n.id)?;
        if n.dtype != dtype
            || n.shape != expected_shapes[i]
            || index.value_storage[nodes[i].index()].representation != StorageRepresentation::Dense
            || !matches!(
                index.value_storage[nodes[i].index()].layout,
                effect_torch_runtime::StorageLayout::Canonical
                    | effect_torch_runtime::StorageLayout::Unconstrained
            )
        {
            return None;
        }
    }
    let covered = nodes
        .iter()
        .enumerate()
        .filter_map(|(i, n)| (!matches!(i, 0 | 3)).then_some(*n))
        .collect::<HashSet<_>>();
    for (i, n) in nodes.iter().enumerate() {
        if matches!(i, 0 | 3 | 8 | 10) {
            continue;
        }
        if index.roots.contains(n)
            || index.consumers[n.index()]
                .iter()
                .any(|c| !covered.contains(c))
        {
            return None;
        }
    }
    use KernelExpr as E;
    let angle = E::Mul(
        Box::new(E::Input(1)),
        Box::new(E::Cast(Box::new(E::Input(0)), DType::F32).semantic(nodes[4], DType::F32)),
    )
    .semantic(nodes[5], DType::F32);
    let ar = elementwise(plan, nodes[5], &nodes[4..6], &[nodes[3], nodes[2]], angle)?;
    if ar.shape.as_ref() != expected_shapes[5]
        || ar.dtype != DType::F32
        || ar.lane_strides.as_ref()
            != [
                vec![0, 0, 1].into_boxed_slice(),
                vec![rows, 1, 0].into_boxed_slice(),
            ]
    {
        return None;
    }
    for (trig, out, is_cos) in [(7, 8, true), (9, 10, false)] {
        let inner = if is_cos {
            E::Cos(Box::new(E::Input(0)))
        } else {
            E::Sin(Box::new(E::Input(0)))
        };
        let expression = E::Cast(
            Box::new(inner.semantic(nodes[trig], DType::F32)),
            DType::BF16,
        )
        .semantic(nodes[out], DType::BF16);
        let r = elementwise(
            plan,
            nodes[out],
            &[nodes[trig], nodes[out]],
            &[nodes[6]],
            expression,
        )?;
        if r.shape.as_ref() != expected_shapes[out]
            || r.dtype != DType::BF16
            || r.lane_strides.as_ref() != [vec![rows * 2 * half, 2 * half, 1].into_boxed_slice()]
        {
            return None;
        }
    }
    for i in [1, 2, 6] {
        if plan.node_region[nodes[i].index()].is_some() {
            return None;
        }
    }
    let mut units = Vec::new();
    let mut positions = Vec::new();
    let mut contracts = Vec::new();
    for (position, entry) in legal.units().iter().enumerate() {
        let unit = entry.unit();
        if !covered.iter().any(|n| unit_of(plan, *n) == unit) {
            continue;
        }
        if let LoweringUnit::Region(r) = unit {
            if plan.regions[r.index()]
                .nodes()
                .iter()
                .any(|n| !covered.contains(n))
            {
                return None;
            }
        }
        let ExecutableDTypePlan::Native(execution) = entry.disposition() else {
            return None;
        };
        if execution.realization != ExecutionRealization::DirectKernel {
            return None;
        }
        let mut normalized = (**execution).clone();
        for operation in &mut normalized.operations {
            if operation.random.is_some()
                || operation
                    .operands
                    .iter()
                    .any(|o| o.preparation != OperandPreparation::Direct)
                || operation
                    .results
                    .iter()
                    .any(|r| r.completion != ResultCompletion::Direct)
            {
                return None;
            }
            operation.node =
                DenseNodeId::from_index(nodes.iter().position(|n| *n == operation.node)?)?;
            for boundary in &mut operation.rounding_boundaries {
                boundary.node =
                    DenseNodeId::from_index(nodes.iter().position(|n| *n == boundary.node)?)?;
            }
        }
        units.push(unit);
        positions.push(position);
        contracts.push(normalized);
    }
    // Three private view/convert nodes and three exact elementwise regions.
    if units.len() != 6 {
        return None;
    }
    Some(Chain {
        nodes,
        units,
        positions,
        contracts,
        rows,
        half,
    })
}

#[derive(Clone, Debug)]
struct Duplicate {
    canonical: Chain,
    duplicate: Chain,
}
#[derive(Clone, Copy, Debug)]
enum State {
    Pending,
    Declined,
    Active([ValueId; 2]),
}

/// Compile-local immutable admission plan plus physical preflight state.
#[derive(Default)]
pub(crate) struct RotaryReuse66 {
    pairs: Vec<Duplicate>,
    actions: HashMap<LoweringUnit, (usize, Option<usize>)>,
    states: Vec<State>,
}
impl RotaryReuse66 {
    pub(crate) fn new(
        index: &GraphIndex,
        plan: &OptimizationPlan,
        legal: &LegalizationPlan,
    ) -> Self {
        let mut result = Self::default();
        if !legal.target().features.iter().any(|f| f == FEATURE) {
            return result;
        }
        let mut first = HashMap::<_, Chain>::new();
        for node in &index.order {
            let Some(chain) = candidate(index, plan, legal, node) else {
                continue;
            };
            let Some(canonical) = first.get(&chain.key()) else {
                first.insert(chain.key(), chain);
                continue;
            };
            if chain.contracts != canonical.contracts
                || canonical.positions.last() >= chain.positions.first()
                || chain.units.iter().any(|u| result.actions.contains_key(u))
            {
                continue;
            }
            let pair = result.pairs.len();
            for unit in &chain.units {
                let slot = chain
                    .tables()
                    .iter()
                    .position(|n| unit_of(plan, *n) == *unit);
                result.actions.insert(*unit, (pair, slot));
            }
            result.pairs.push(Duplicate {
                canonical: canonical.clone(),
                duplicate: chain,
            });
            result.states.push(State::Pending);
        }
        result
    }

    /// Returns true only when this whole-chain-admitted unit was replaced.
    pub(crate) fn lower(
        &mut self,
        unit: LoweringUnit,
        index: &GraphIndex,
        builder: &mut CudaProgramBuilder,
    ) -> Result<bool, String> {
        let Some(&(pair, slot)) = self.actions.get(&unit) else {
            return Ok(false);
        };
        let p = &self.pairs[pair];
        if matches!(self.states[pair], State::Pending) {
            // This is the first duplicate unit in unchanged legalization order.
            self.states[pair] = preflight(builder, &p.canonical)
                .map(State::Active)
                .unwrap_or(State::Declined);
        }
        let State::Active(sources) = self.states[pair] else {
            return Ok(false);
        };
        if let Some(slot) = slot {
            let semantic = p.duplicate.tables()[slot];
            let n = &index.order[semantic.index()];
            let source = sources[slot];
            let output = builder.value(
                n.shape.clone(),
                n.dtype,
                StorageMetadata::dense(),
                "rotary_reuse66",
                ValueStorage::Alias {
                    source,
                    byte_offset: 0,
                },
            )?;
            builder.emit(
                "rotary_reuse66",
                CommandKind::Alias { source },
                Some(output),
                vec![ValueUse::read(source)],
                false,
            )?;
            builder.semantic_values[semantic.index()] = Some(output);
            builder.semantic_results[semantic.index()] = vec![output];
        }
        Ok(true)
    }
}

fn physical(
    builder: &CudaProgramBuilder,
    semantic: DenseNodeId,
    shape: &[usize],
    dtype: DType,
    owned: bool,
) -> Option<(ValueId, ValueId)> {
    let id = builder.resolve(semantic.index()).ok()?;
    let v = builder.values.get(id.index())?;
    let bytes = shape
        .iter()
        .try_fold(dtype.size_in_bytes(), |a, b| a.checked_mul(*b))?;
    if v.shape != shape
        || v.dtype != dtype
        || v.storage != StorageMetadata::dense()
        || v.decl.bytes != bytes
    {
        return None;
    }
    let mut root = id;
    for _ in 0..=builder.values.len() {
        let value = builder.values.get(root.index())?;
        if value.decl.bytes < bytes
            || value.dtype != dtype
            || value.storage != StorageMetadata::dense()
        {
            return None;
        }
        match value.decl.storage {
            ValueStorage::Alias {
                source,
                byte_offset,
            } => {
                if byte_offset != 0 {
                    return None;
                }
                root = source;
            }
            ValueStorage::Planned {
                class: StorageClass::Workspace | StorageClass::EscapingOutput,
                ..
            } => return owned.then_some((id, root)),
            ValueStorage::Fixed {
                class: StorageClass::ExternalInput,
                location: Location::External { .. },
            }
            | ValueStorage::Fixed {
                class: StorageClass::PersistentConstant,
                location: Location::Persistent { .. },
            } => return (!owned).then_some((id, root)),
            _ => return None,
        }
    }
    None
}
fn preflight(builder: &CudaProgramBuilder, c: &Chain) -> Option<[ValueId; 2]> {
    physical(builder, c.nodes[0], &[1, c.rows], DType::U32, false)?;
    physical(builder, c.nodes[3], &[c.half], DType::F32, false)?;
    let a = physical(
        builder,
        c.nodes[8],
        &[1, c.rows, 2 * c.half],
        DType::BF16,
        true,
    )?;
    let b = physical(
        builder,
        c.nodes[10],
        &[1, c.rows, 2 * c.half],
        DType::BF16,
        true,
    )?;
    (a.1 != b.1).then_some([a.0, b.0])
}

#[cfg(test)]
#[path = "rotary_reuse66_tests.rs"]
mod tests;

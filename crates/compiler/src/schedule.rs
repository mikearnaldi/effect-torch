use effect_torch_graph::{node_children, Device, Node as GraphNode, NodeKind as GraphNodeKind};
use effect_torch_runtime::{DType, DenseId};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

type Node = GraphNode;
type NodeKind = GraphNodeKind;

struct GraphDiscovery {
    order: Vec<Arc<Node>>,
    children_by_node: HashMap<u64, Vec<u64>>,
    edge_visits: usize,
}

fn discover_graph(roots: &[Arc<Node>]) -> GraphDiscovery {
    let mut visited = HashSet::new();
    let mut order = Vec::new();
    let mut children_by_node = HashMap::new();
    let mut edge_visits = 0usize;
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
        let children = node_children(&node.kind);
        edge_visits = edge_visits.saturating_add(children.len());
        children_by_node.insert(node.id, children.iter().map(|child| child.id).collect());
        stack.push((node, true));
        for child in children.into_iter().rev() {
            stack.push((child, false));
        }
    }
    GraphDiscovery {
        order,
        children_by_node,
        edge_visits,
    }
}

/// Deterministic, stack-safe postorder over roots and children in caller order.
/// Shared subgraphs occur once; hash tables are used only for membership.
#[cfg(test)]
pub(crate) fn graph_post_order(roots: &[Arc<Node>]) -> Vec<Arc<Node>> {
    discover_graph(roots).order
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

fn collect_program_slots_from_order(
    order: &[Arc<Node>],
) -> std::result::Result<(Vec<ProgramSlot>, Vec<(u64, u32)>), String> {
    let mut slots: Vec<Option<ProgramSlot>> = Vec::new();
    let mut leaves: Vec<(u64, u32)> = Vec::new();
    for node in order {
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
fn collect_program_slots(
    roots: &[Arc<Node>],
) -> std::result::Result<(Vec<ProgramSlot>, Vec<(u64, u32)>), String> {
    collect_program_slots_from_order(&graph_post_order(roots))
}

/// Dense postorder identity for one immutable graph generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DenseNodeId(u32);

impl DenseNodeId {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub fn from_index(index: usize) -> Option<Self> {
        u32::try_from(index).ok().map(Self)
    }
}

impl DenseId for DenseNodeId {
    fn from_u32(value: u32) -> Self {
        Self(value)
    }

    fn as_u32(self) -> u32 {
        self.0
    }
}

impl fmt::Display for DenseNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Dense identity assigned to random operations in semantic postorder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RandomSourceId(u32);

impl RandomSourceId {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub fn from_index(index: usize) -> Option<Self> {
        u32::try_from(index).ok().map(Self)
    }
}

impl DenseId for RandomSourceId {
    fn from_u32(value: u32) -> Self {
        Self(value)
    }

    fn as_u32(self) -> u32 {
        self.0
    }
}

impl fmt::Display for RandomSourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RandomSourceKind {
    Randn,
    Uniform,
}

/// Stable metadata for one semantic random operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RandomSource {
    pub id: RandomSourceId,
    pub node: DenseNodeId,
    pub provenance: u64,
    pub kind: RandomSourceKind,
}

pub type RandomSourceMetadata = RandomSource;

/// Metadata for a concrete graph leaf parameterized by native compilation.
/// The concrete value remains owned only through the graph's `LeafSlot`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedBinding {
    pub node: DenseNodeId,
    pub node_id: u64,
    pub slot_identity: usize,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub device: Device,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GraphIndexWork {
    pub graph_index_builds: usize,
    pub semantic_node_visits: usize,
    pub graph_edge_visits: usize,
}

/// Dense, deterministic analysis of one immutable semantic graph generation.
#[derive(Clone)]
pub struct GraphIndex {
    pub order: Box<[Arc<Node>]>,
    pub dense_by_node: HashMap<u64, DenseNodeId>,
    pub children: Box<[Box<[DenseNodeId]>]>,
    pub consumers: Box<[Box<[DenseNodeId]>]>,
    pub roots: Box<[DenseNodeId]>,
    pub slots: Box<[ProgramSlot]>,
    pub slot_leaves: Box<[(DenseNodeId, u32)]>,
    pub leaves: Box<[GeneratedBinding]>,
    pub random_sources: Box<[Option<RandomSourceId>]>,
    pub random_source_order: Box<[RandomSource]>,
    pub work: GraphIndexWork,
}

impl GraphIndex {
    pub fn new(roots: &[Arc<Node>]) -> Result<Self, String> {
        let GraphDiscovery {
            order,
            mut children_by_node,
            edge_visits,
        } = discover_graph(roots);

        let mut dense_by_node = HashMap::with_capacity(order.len());
        for (index, node) in order.iter().enumerate() {
            let dense = DenseNodeId::from_index(index)
                .ok_or_else(|| "compile: graph contains too many nodes".to_string())?;
            dense_by_node.insert(node.id, dense);
        }

        let mut children = Vec::with_capacity(order.len());
        let mut consumers = vec![Vec::new(); order.len()];
        for (index, node) in order.iter().enumerate() {
            let parent = DenseNodeId::from_index(index)
                .expect("graph node count was validated while assigning dense IDs");
            let dense_children = children_by_node
                .remove(&node.id)
                .unwrap_or_default()
                .into_iter()
                .map(|child| {
                    dense_by_node
                        .get(&child)
                        .copied()
                        .expect("discovered child must have a dense ID")
                })
                .collect::<Vec<_>>();
            for child in &dense_children {
                consumers[child.index()].push(parent);
            }
            children.push(dense_children.into_boxed_slice());
        }

        let roots = roots
            .iter()
            .map(|root| {
                dense_by_node
                    .get(&root.id)
                    .copied()
                    .expect("every root must have a dense ID")
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let (slots, slot_leaves) = collect_program_slots_from_order(&order)?;
        let slot_leaves = slot_leaves
            .into_iter()
            .map(|(node, slot)| {
                (
                    dense_by_node
                        .get(&node)
                        .copied()
                        .expect("every slot leaf must have a dense ID"),
                    slot,
                )
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let mut leaves = Vec::new();
        let mut random_sources = vec![None; order.len()];
        let mut random_source_order = Vec::new();
        for (index, node) in order.iter().enumerate() {
            let dense = DenseNodeId::from_index(index)
                .expect("graph node count was validated while assigning dense IDs");
            match &node.kind {
                NodeKind::Leaf(slot) => leaves.push(GeneratedBinding {
                    node: dense,
                    node_id: node.id,
                    slot_identity: Arc::as_ptr(slot) as usize,
                    shape: node.shape.clone(),
                    dtype: node.dtype,
                    device: node.device.clone(),
                }),
                NodeKind::Randn { .. } | NodeKind::Uniform { .. } => {
                    let source = RandomSourceId::from_index(random_source_order.len())
                        .expect("random source count cannot exceed graph node count");
                    let kind = match &node.kind {
                        NodeKind::Randn { .. } => RandomSourceKind::Randn,
                        NodeKind::Uniform { .. } => RandomSourceKind::Uniform,
                        _ => unreachable!(),
                    };
                    random_sources[index] = Some(source);
                    random_source_order.push(RandomSource {
                        id: source,
                        node: dense,
                        provenance: node.id,
                        kind,
                    });
                }
                _ => {}
            }
        }

        Ok(Self {
            work: GraphIndexWork {
                graph_index_builds: 1,
                semantic_node_visits: order.len(),
                graph_edge_visits: edge_visits,
            },
            order: order.into_boxed_slice(),
            dense_by_node,
            children: children.into_boxed_slice(),
            consumers: consumers
                .into_iter()
                .map(Vec::into_boxed_slice)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            roots,
            slots: slots.into_boxed_slice(),
            slot_leaves,
            leaves: leaves.into_boxed_slice(),
            random_sources: random_sources.into_boxed_slice(),
            random_source_order: random_source_order.into_boxed_slice(),
        })
    }

    pub fn dense_id(&self, node_id: u64) -> Option<DenseNodeId> {
        self.dense_by_node.get(&node_id).copied()
    }

    pub fn node(&self, id: DenseNodeId) -> Option<&Arc<Node>> {
        self.order.get(id.index())
    }

    pub fn children_of(&self, id: DenseNodeId) -> Option<&[DenseNodeId]> {
        self.children.get(id.index()).map(Box::as_ref)
    }

    pub fn consumers_of(&self, id: DenseNodeId) -> Option<&[DenseNodeId]> {
        self.consumers.get(id.index()).map(Box::as_ref)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use effect_torch_graph::{LeafSlot, LeafValue};
    use std::any::Any;

    #[derive(Clone)]
    struct TestLeaf;

    impl LeafValue for TestLeaf {
        fn shape(&self) -> Vec<usize> {
            vec![1]
        }

        fn dtype(&self) -> DType {
            DType::F32
        }

        fn device(&self) -> Device {
            Device::Cpu
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn input(slot: u32) -> Arc<Node> {
        Node::new(NodeKind::Input {
            slot,
            shape: vec![1],
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap()
    }

    fn generated_leaf() -> (Arc<Node>, Arc<LeafSlot>) {
        let slot = Arc::new(LeafSlot::new(TestLeaf));
        let node = Node::new(NodeKind::Leaf(slot.clone())).unwrap();
        (node, slot)
    }

    fn randn() -> Arc<Node> {
        Node::new(NodeKind::Randn {
            shape: vec![1],
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap()
    }

    fn uniform() -> Arc<Node> {
        Node::new(NodeKind::Uniform {
            lo: -1.0,
            hi: 1.0,
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
    fn graph_index_preserves_postorder_and_child_order() {
        let x = input(0);
        let y = input(1);
        let root = Node::new(NodeKind::Sub {
            a: x.clone(),
            b: y.clone(),
        })
        .unwrap();
        let index = GraphIndex::new(std::slice::from_ref(&root)).unwrap();
        let root_dense = index.dense_id(root.id).unwrap();

        assert_eq!(
            index.order.iter().map(|node| node.id).collect::<Vec<_>>(),
            [x.id, y.id, root.id]
        );
        assert_eq!(
            index.children_of(root_dense).unwrap(),
            [index.dense_id(x.id).unwrap(), index.dense_id(y.id).unwrap()]
        );
    }

    #[test]
    fn graph_index_preserves_duplicate_root_order() {
        let x = input(0);
        let left = Node::new(NodeKind::Neg { a: x.clone() }).unwrap();
        let right = Node::new(NodeKind::Tanh { a: x }).unwrap();
        let index = GraphIndex::new(&[right.clone(), left.clone(), right.clone()]).unwrap();

        assert_eq!(
            index.roots.as_ref(),
            [
                index.dense_id(right.id).unwrap(),
                index.dense_id(left.id).unwrap(),
                index.dense_id(right.id).unwrap(),
            ]
        );
    }

    #[test]
    fn graph_index_deduplicates_shared_nodes() {
        let x = input(0);
        let shared = Node::new(NodeKind::Neg { a: x }).unwrap();
        let left = Node::new(NodeKind::Tanh { a: shared.clone() }).unwrap();
        let right = Node::new(NodeKind::Relu { a: shared.clone() }).unwrap();
        let index = GraphIndex::new(&[left.clone(), right.clone()]).unwrap();
        let shared_dense = index.dense_id(shared.id).unwrap();

        assert_eq!(index.order.len(), 4);
        assert_eq!(
            index
                .order
                .iter()
                .filter(|node| node.id == shared.id)
                .count(),
            1
        );
        assert_eq!(
            index.consumers_of(shared_dense).unwrap(),
            [
                index.dense_id(left.id).unwrap(),
                index.dense_id(right.id).unwrap(),
            ]
        );
    }

    #[test]
    fn consumer_adjacency_preserves_duplicate_edge_multiplicity() {
        let x = input(0);
        let root = Node::new(NodeKind::Add {
            a: x.clone(),
            b: x.clone(),
        })
        .unwrap();
        let index = GraphIndex::new(std::slice::from_ref(&root)).unwrap();
        let x_dense = index.dense_id(x.id).unwrap();
        let root_dense = index.dense_id(root.id).unwrap();

        assert_eq!(index.children_of(root_dense).unwrap(), [x_dense, x_dense]);
        assert_eq!(
            index.consumers_of(x_dense).unwrap(),
            [root_dense, root_dense]
        );
        assert_eq!(index.work.graph_edge_visits, 2);
    }

    #[test]
    fn graph_index_orders_slots_generated_leaves_and_random_sources() {
        let (generated_a, slot_a) = generated_leaf();
        let input_one = input(1);
        let random_a = randn();
        let left = Node::new(NodeKind::Add {
            a: generated_a.clone(),
            b: input_one.clone(),
        })
        .unwrap();
        let left = Node::new(NodeKind::Add {
            a: left,
            b: random_a.clone(),
        })
        .unwrap();

        let (generated_b, slot_b) = generated_leaf();
        let scalar_zero = Node::new(NodeKind::ScalarInput {
            slot: 0,
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap();
        let random_b = uniform();
        let right = Node::new(NodeKind::Add {
            a: generated_b.clone(),
            b: scalar_zero.clone(),
        })
        .unwrap();
        let right = Node::new(NodeKind::Add {
            a: right,
            b: random_b.clone(),
        })
        .unwrap();

        let slot_a_owners = Arc::strong_count(&slot_a);
        let slot_b_owners = Arc::strong_count(&slot_b);
        let index = GraphIndex::new(&[left, right]).unwrap();

        assert_eq!(index.slots.len(), 2);
        assert!(index.slots[0].scalar);
        assert!(!index.slots[1].scalar);
        assert_eq!(
            index.slot_leaves.as_ref(),
            [
                (index.dense_id(input_one.id).unwrap(), 1),
                (index.dense_id(scalar_zero.id).unwrap(), 0),
            ]
        );
        assert_eq!(
            index
                .leaves
                .iter()
                .map(|binding| binding.node_id)
                .collect::<Vec<_>>(),
            [generated_a.id, generated_b.id]
        );
        assert_eq!(index.leaves[0].slot_identity, Arc::as_ptr(&slot_a) as usize);
        assert_eq!(index.leaves[1].slot_identity, Arc::as_ptr(&slot_b) as usize);
        assert_eq!(Arc::strong_count(&slot_a), slot_a_owners);
        assert_eq!(Arc::strong_count(&slot_b), slot_b_owners);
        assert_eq!(
            index
                .random_source_order
                .iter()
                .map(|source| (source.provenance, source.kind))
                .collect::<Vec<_>>(),
            [
                (random_a.id, RandomSourceKind::Randn),
                (random_b.id, RandomSourceKind::Uniform),
            ]
        );
        assert_eq!(
            index.random_sources[index.dense_id(random_a.id).unwrap().index()],
            Some(RandomSourceId::new(0))
        );
        assert_eq!(
            index.random_sources[index.dense_id(random_b.id).unwrap().index()],
            Some(RandomSourceId::new(1))
        );
    }

    #[test]
    fn slot_validation_errors_remain_compatible() {
        assert_eq!(
            collect_program_slots(&[input(1)]).unwrap_err(),
            "compile: slot 0 is declared but never used"
        );

        let first = input(0);
        let second = Node::new(NodeKind::Input {
            slot: 0,
            shape: vec![2],
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap();
        let root = Node::new(NodeKind::Add {
            a: first,
            b: second,
        })
        .unwrap();
        assert_eq!(
            GraphIndex::new(&[root]).err().unwrap(),
            "compile: slot 0 is used with conflicting signatures (1:f32@cpu vs 2:f32@cpu)"
        );
    }

    #[test]
    fn graph_index_is_stack_safe_for_deep_graphs() {
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
                let index = GraphIndex::new(&[root]).unwrap();
                assert_eq!(index.order.len(), 50_001);
                assert_eq!(index.order.first().unwrap().id, leaf_id);
                assert_eq!(index.order.last().unwrap().id, root_id);
                assert_eq!(index.work.semantic_node_visits, 50_001);
                assert_eq!(index.work.graph_edge_visits, 50_000);
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
